"""Binance adapter logic, offline.

The venue is stubbed so these run hermetically: URI parsing, pagination, the refusal to
ingest an unclosed bar, interval cross-checks, symbol resolution, and retry behaviour.
The timestamp-label proof lives in ``tests/contracts/test_binance_contract.py``.
"""

from __future__ import annotations

import datetime as dt
import urllib.error
from decimal import Decimal
from typing import Any

import pytest

from qresearch.data.adapters.base import IngestRequest
from qresearch.data.adapters.binance import (
    MAX_LIMIT,
    SOURCE_NAME,
    BinanceError,
    BinanceKlineAdapter,
    BinanceQuery,
    build_uri,
)
from qresearch.data.contracts import AssetClass, Instrument, PriceAdjustment, SymbolAlias
from qresearch.data.manifests import DuplicatePolicy, NormalizationPolicy, TimestampLabel

T0 = dt.datetime(2024, 3, 4, 12, 0, tzinfo=dt.UTC)
BTC, ETH = "CRYPTO:BTCUSDT", "CRYPTO:ETHUSDT"
LISTED = dt.datetime(2017, 1, 1, tzinfo=dt.UTC)


def kline(
    minute: int, *, close: float = 100.0, volume: float = 10.0, close_ms_offset: int = 59_999
) -> list[Any]:
    """One raw kline in Binance's array shape."""
    open_ms = int((T0 + dt.timedelta(minutes=minute)).timestamp() * 1000)
    return [
        open_ms,
        f"{close - 1:.8f}",
        f"{close + 1:.8f}",
        f"{close - 2:.8f}",
        f"{close:.8f}",
        f"{volume:.8f}",
        open_ms + close_ms_offset,
        f"{volume * close:.8f}",
        42,
        "0",
        "0",
        "0",
    ]


def instrument(
    instrument_id: str, symbol: str, *, effective_from: dt.datetime = LISTED
) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        asset_class=AssetClass.CRYPTO,
        venue="BINANCE",
        quote_currency="USDT",
        base_currency=symbol.removesuffix("USDT"),
        price_increment=Decimal("0.01"),
        quantity_increment=Decimal("0.00001"),
        calendar_id="24x7:1",
        aliases=(SymbolAlias(symbol=symbol, source=SOURCE_NAME, effective_from=effective_from),),
    )


def make_request(
    *,
    start: dt.datetime = T0,
    hours: int = 1,
    instruments: tuple[Instrument, ...] | None = None,
    bar_size: str = "1m",
    interval: str | None = None,
    **policy_overrides: Any,
) -> IngestRequest:
    base: dict[str, Any] = {
        "timestamp_label": TimestampLabel.BAR_START,
        "publication_latency": dt.timedelta(seconds=2),
        "price_adjustment": PriceAdjustment.NOT_APPLICABLE,
        "duplicates": DuplicatePolicy.ERROR,
        "volume_unit": "base_asset",
    }
    return IngestRequest(
        uri=build_uri(
            interval=interval or bar_size, start=start, end=start + dt.timedelta(hours=hours)
        ),
        bar_size=bar_size,
        policy=NormalizationPolicy.model_validate(base | policy_overrides),
        instruments=instruments or (instrument(BTC, "BTCUSDT"),),
        source_name=SOURCE_NAME,
    )


class StubAdapter(BinanceKlineAdapter):
    """Serves canned klines, recording every request it would have made."""

    def __init__(
        self, pages: list[Any], *, now: Any = None, per_symbol: dict[str, list[Any]] | None = None
    ) -> None:
        super().__init__(pace=0.0, now=now or (lambda: dt.datetime(2024, 3, 5, tzinfo=dt.UTC)))
        self._pages = pages
        self._per_symbol = per_symbol
        self.calls: list[dict[str, Any]] = []

    def _get(self, path: str, **params: Any) -> list[Any]:
        self.calls.append(params)
        source = self._per_symbol[params["symbol"]] if self._per_symbol else self._pages
        start = params["startTime"]
        return [k for k in source if k[0] >= start][: params["limit"]]


# -- URI ------------------------------------------------------------------------------


def test_uri_parses_and_round_trips() -> None:
    query = BinanceQuery.parse(build_uri(interval="5m", start=T0, end=T0 + dt.timedelta(hours=2)))
    assert (query.market, query.interval) == ("spot", "5m")
    assert query.start == T0 and query.end == T0 + dt.timedelta(hours=2)
    assert query.start_ms == int(T0.timestamp() * 1000)


@pytest.mark.parametrize(
    ("uri", "match"),
    [
        ("https://example.com", "expected a binance:// uri"),
        (
            "binance://futures?interval=1m&start=2024-01-01T00:00:00Z&end=2024-01-02T00:00:00Z",
            "only the spot market",
        ),
        ("binance://spot?interval=1m", "missing"),
        (
            "binance://spot?interval=1m&start=2024-01-02T00:00:00Z&end=2024-01-01T00:00:00Z",
            "must precede",
        ),
    ],
)
def test_bad_uris_are_refused(uri: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        BinanceQuery.parse(uri)


def test_naive_timestamps_in_a_uri_are_refused() -> None:
    with pytest.raises(ValueError):
        BinanceQuery.parse(
            "binance://spot?interval=1m&start=2024-01-01T00:00:00&end=2024-01-02T00:00:00"
        )


# -- mapping --------------------------------------------------------------------------


def test_klines_become_canonical_bars() -> None:
    frame = StubAdapter([kline(i, close=100.0 + i) for i in range(3)]).read(make_request()).frame
    assert frame.height == 3
    row = frame.row(0, named=True)
    assert row["instrument_id"] == BTC and row["bar_size"] == "1m"
    assert row["bar_start"] == T0 and row["bar_end"] == T0 + dt.timedelta(minutes=1)
    assert row["available_at"] == T0 + dt.timedelta(minutes=1, seconds=2)
    assert row["open"] == pytest.approx(99.0) and row["close"] == pytest.approx(100.0)
    assert row["source"] == SOURCE_NAME
    assert row["source_key"].startswith(f"{BTC}:")


def test_vwap_is_quote_volume_over_base_volume() -> None:
    frame = StubAdapter([kline(0, close=100.0, volume=10.0)]).read(make_request()).frame
    assert frame.item(0, "vwap") == pytest.approx(1000.0 / 10.0)


def test_zero_volume_yields_a_null_vwap_not_a_division_error() -> None:
    frame = StubAdapter([kline(0, volume=0.0)]).read(make_request()).frame
    assert frame.item(0, "vwap") is None
    assert frame.item(0, "volume") == 0.0


def test_multiple_instruments_are_fetched_and_sorted() -> None:
    adapter = StubAdapter(
        [],
        per_symbol={
            "BTCUSDT": [kline(i, close=100.0 + i) for i in range(2)],
            "ETHUSDT": [kline(i, close=50.0 + i) for i in range(2)],
        },
    )
    request = make_request(instruments=(instrument(ETH, "ETHUSDT"), instrument(BTC, "BTCUSDT")))
    frame = adapter.read(request).frame
    assert frame.height == 4
    assert frame.get_column("instrument_id").to_list() == [BTC, BTC, ETH, ETH]
    assert {c["symbol"] for c in adapter.calls} == {"BTCUSDT", "ETHUSDT"}


# -- unclosed bars ----------------------------------------------------------------------


def test_an_unclosed_bar_is_dropped() -> None:
    """The final kline of a live pull is still forming; ingesting it would write a bar
    that later changes."""
    adapter = StubAdapter(
        [kline(0), kline(1), kline(2)],
        now=lambda: T0 + dt.timedelta(minutes=2, seconds=30),
    )
    batch = adapter.read(make_request())
    assert batch.frame.height == 2, "the 12:02 bar has not closed at 12:02:30"
    assert batch.frame.get_column("bar_start").max() == T0 + dt.timedelta(minutes=1)
    assert batch.dropped_unmapped_symbols == ("BTCUSDT:1 unclosed",)


def test_all_bars_unclosed_is_an_error_not_an_empty_dataset() -> None:
    adapter = StubAdapter([kline(0)], now=lambda: T0 + dt.timedelta(seconds=30))
    with pytest.raises(BinanceError, match="no closed"):
        adapter.read(make_request())


# -- interval consistency -----------------------------------------------------------------


def test_a_close_time_inconsistent_with_the_interval_is_refused() -> None:
    """If field[6] does not sit one ms before the next open, the venue's interval is not
    what the adapter assumes and the dataset could be shifted in time."""
    adapter = StubAdapter([kline(0, close_ms_offset=119_999)])
    with pytest.raises(BinanceError, match="inconsistent with a"):
        adapter.read(make_request())


def test_bar_size_must_map_to_a_venue_interval() -> None:
    with pytest.raises(ValueError, match="no Binance interval"):
        StubAdapter([kline(0)]).read(make_request(bar_size="7m", interval="7m"))


def test_uri_interval_must_agree_with_bar_size() -> None:
    with pytest.raises(ValueError, match="disagrees with bar_size"):
        StubAdapter([kline(0)]).read(make_request(bar_size="5m", interval="1m"))


# -- symbols ---------------------------------------------------------------------------


def test_symbols_resolve_point_in_time() -> None:
    later = instrument(BTC, "BTCUSDT", effective_from=T0 + dt.timedelta(days=1))
    with pytest.raises(ValueError, match="no instrument has a"):
        StubAdapter([kline(0)]).read(make_request(instruments=(later,)))


# -- provenance --------------------------------------------------------------------------


def test_identical_pulls_hash_identically_and_differing_ones_do_not() -> None:
    """The content hash is what makes a re-pull resolve to the same dataset id."""
    pages = [kline(i) for i in range(3)]
    first = StubAdapter(pages).read(make_request()).source
    second = StubAdapter(list(pages)).read(make_request()).source
    assert first.content_sha256 == second.content_sha256
    changed = StubAdapter([kline(0, close=999.0), kline(1), kline(2)]).read(make_request()).source
    assert changed.content_sha256 != first.content_sha256


def test_source_ref_records_the_uri_and_feed() -> None:
    source = StubAdapter([kline(0)]).read(make_request()).source
    assert source.provider == SOURCE_NAME
    assert source.uri.startswith("binance://spot?")
    assert source.feed == "spot/1m"


# -- pagination and retries ------------------------------------------------------------------


def test_pagination_walks_forward_without_repeating_a_bar() -> None:
    adapter = StubAdapter(
        [kline(i) for i in range(MAX_LIMIT + 5)],
        now=lambda: T0 + dt.timedelta(days=2),  # everything has closed
    )
    frame = adapter.read(make_request(hours=24)).frame
    assert frame.height == MAX_LIMIT + 5
    assert frame.get_column("bar_start").n_unique() == MAX_LIMIT + 5
    assert len(adapter.calls) >= 2, "more than one page was needed"


class FailingTransport(BinanceKlineAdapter):
    """Stubs the network at ``_fetch``, so the real retry policy in ``_get`` runs."""

    def __init__(
        self, error: Exception, *, succeed_after: int | None = None, max_retries: int = 3
    ) -> None:
        super().__init__(max_retries=max_retries, pace=0.0, now=lambda: T0 + dt.timedelta(days=2))
        self._error = error
        self._succeed_after = succeed_after
        self.attempts = 0

    def _fetch(self, url: str) -> Any:
        self.attempts += 1
        if self._succeed_after is not None and self.attempts >= self._succeed_after:
            return [kline(0)]
        raise self._error


def _http_error(code: int, retry_after: str | None = "0") -> urllib.error.HTTPError:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError("/api/v3/klines", code, "stub", headers, None)  # type: ignore[arg-type]


def test_rate_limits_are_retried_then_succeed() -> None:
    adapter = FailingTransport(_http_error(429), succeed_after=3)
    assert adapter.read(make_request()).frame.height == 1
    assert adapter.attempts == 3


def test_persistent_rate_limiting_is_surfaced_after_the_retry_budget() -> None:
    adapter = FailingTransport(_http_error(429), max_retries=2)
    with pytest.raises(BinanceError, match="HTTP 429"):
        adapter.read(make_request())
    assert adapter.attempts == 2


def test_server_errors_are_retried() -> None:
    adapter = FailingTransport(_http_error(503), succeed_after=2)
    assert adapter.read(make_request()).frame.height == 1
    assert adapter.attempts == 2


def test_a_client_error_is_not_retried() -> None:
    """A 400 is our bug, not the venue's; surface it instead of hammering the endpoint."""
    adapter = FailingTransport(_http_error(400, retry_after=None))
    with pytest.raises(BinanceError, match="HTTP 400"):
        adapter.read(make_request())
    assert adapter.attempts == 1


def test_a_transport_failure_is_retried_then_surfaced() -> None:
    adapter = FailingTransport(urllib.error.URLError("dns"), max_retries=2)
    with pytest.raises(BinanceError, match="failed after 2 attempts"):
        adapter.read(make_request())
    assert adapter.attempts == 2


def test_a_non_list_payload_is_refused() -> None:
    class Weird(BinanceKlineAdapter):
        def __init__(self) -> None:
            super().__init__(pace=0.0, now=lambda: T0 + dt.timedelta(days=2))

        def _fetch(self, url: str) -> Any:
            return {"code": -1121, "msg": "Invalid symbol."}

    with pytest.raises(BinanceError, match="expected a list"):
        Weird().read(make_request())
