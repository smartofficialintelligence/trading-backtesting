"""Binance provider contract: the timestamp label, proven against raw trades.

DEVELOPMENT_PLAN.md Stage 1 requires provider timestamp behaviour to be captured in
contract tests. For a real venue that means more than repeating the documentation: a
kline's ``field[0]`` is compared against the trades that actually occurred in
``[field[0], field[0] + interval)``. If the label were the interval *end*, the open and
close would belong to the previous minute and would not match.

The proof runs offline from a captured fixture, so it protects the mapping on every test
run. A ``network``-marked twin re-pulls the same window from the venue and asserts the
fixture still describes reality; it is deselected by default
(``QRESEARCH_NETWORK_TESTS=1`` to enable).
"""

from __future__ import annotations

import datetime as dt
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from tests.unit.test_binance_adapter import BTC, StubAdapter, make_request

from qresearch.data.adapters.base import IngestRequest
from qresearch.data.adapters.binance import (
    SOURCE_NAME,
    BinanceKlineAdapter,
    BinanceQuery,
    build_uri,
)
from qresearch.data.contracts import PriceAdjustment
from qresearch.data.manifests import DuplicatePolicy, NormalizationPolicy, TimestampLabel

FIXTURE = Path(__file__).parent / "data" / "binance_btcusdt_1m_20240304T1200Z.json"
WINDOW_START = dt.datetime(2024, 3, 4, 12, 0, tzinfo=dt.UTC)


@pytest.fixture(scope="module")
def captured() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def test_the_label_is_bar_start_proven_against_raw_trades(captured: dict[str, Any]) -> None:
    """The decisive test: OHLC from the kline equals OHLC rebuilt from the trades that
    happened inside ``[field[0], field[0] + 60s)``."""
    kline = captured["klines"][0]
    derived = captured["derived_from_trades"]

    assert int(kline[0]) == int(WINDOW_START.timestamp() * 1000)
    assert Decimal(kline[1]) == Decimal(derived["open"]), "open == first trade in the window"
    assert Decimal(kline[2]) == Decimal(derived["high"]), "high == max trade in the window"
    assert Decimal(kline[3]) == Decimal(derived["low"]), "low == min trade in the window"
    assert Decimal(kline[4]) == Decimal(derived["close"]), "close == last trade in the window"

    first = dt.datetime.fromtimestamp(derived["first_trade_at_ms"] / 1000, dt.UTC)
    last = dt.datetime.fromtimestamp(derived["last_trade_at_ms"] / 1000, dt.UTC)
    assert WINDOW_START <= first < last < WINDOW_START + dt.timedelta(minutes=1)


def test_field_six_is_the_inclusive_close(captured: dict[str, Any]) -> None:
    """field[6] sits one millisecond before the next kline's field[0]."""
    first, second = captured["klines"][0], captured["klines"][1]
    assert int(first[6]) == int(first[0]) + 59_999
    assert int(first[6]) + 1 == int(second[0])


def test_the_adapter_places_the_interval_where_the_trades_are(captured: dict[str, Any]) -> None:
    """End to end through the adapter: the captured klines become bars on [12:00, 12:01)."""
    adapter = StubAdapter(captured["klines"], now=lambda: dt.datetime(2024, 3, 5, tzinfo=dt.UTC))
    frame = adapter.read(make_request(start=WINDOW_START, hours=1)).frame
    row = frame.row(0, named=True)
    derived = captured["derived_from_trades"]

    assert row["bar_start"] == WINDOW_START
    assert row["bar_end"] == WINDOW_START + dt.timedelta(minutes=1)
    assert row["open"] == pytest.approx(float(derived["open"]))
    assert row["close"] == pytest.approx(float(derived["close"]))
    assert row["available_at"] == row["bar_end"] + dt.timedelta(seconds=2), "declared latency"


def test_a_bar_end_policy_is_refused(captured: dict[str, Any]) -> None:
    """Declaring the wrong convention must fail loudly, not shift the dataset by a bar."""
    adapter = StubAdapter(captured["klines"], now=lambda: dt.datetime(2024, 3, 5, tzinfo=dt.UTC))
    request = make_request(start=WINDOW_START, hours=1, timestamp_label=TimestampLabel.BAR_END)
    with pytest.raises(ValueError, match="labelled by the interval start"):
        adapter.read(request)


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("QRESEARCH_NETWORK_TESTS") != "1",
    reason="set QRESEARCH_NETWORK_TESTS=1 to hit the live venue",
)
def test_the_venue_still_returns_what_the_fixture_captured(captured: dict[str, Any]) -> None:
    """Freshness check: historical klines are immutable, so a re-pull must be identical."""
    request = make_request(start=WINDOW_START, hours=1)
    batch = BinanceKlineAdapter().read(request)
    live = batch.frame.filter(batch.frame["instrument_id"] == BTC).row(0, named=True)
    kline = captured["klines"][0]
    assert live["open"] == pytest.approx(float(kline[1]))
    assert live["high"] == pytest.approx(float(kline[2]))
    assert live["low"] == pytest.approx(float(kline[3]))
    assert live["close"] == pytest.approx(float(kline[4]))
    assert live["volume"] == pytest.approx(float(kline[5]))


def test_uri_round_trips() -> None:
    uri = build_uri(interval="1m", start=WINDOW_START, end=WINDOW_START + dt.timedelta(hours=3))
    query = BinanceQuery.parse(uri)
    assert query.interval == "1m" and query.market == "spot"
    assert query.start == WINDOW_START
    assert query.end == WINDOW_START + dt.timedelta(hours=3)


def test_policy_used_by_the_contract_tests_is_the_documented_one() -> None:
    """Guards the example config: latency is a collection property, stated explicitly."""
    policy = make_request(start=WINDOW_START, hours=1).policy
    assert policy.timestamp_label is TimestampLabel.BAR_START
    assert policy.publication_latency == dt.timedelta(seconds=2)
    assert policy.price_adjustment is PriceAdjustment.NOT_APPLICABLE
    assert policy.duplicates is DuplicatePolicy.ERROR
    assert isinstance(policy, NormalizationPolicy)


def test_request_declares_the_binance_source() -> None:
    request = make_request(start=WINDOW_START, hours=1)
    assert isinstance(request, IngestRequest)
    assert request.source_name == SOURCE_NAME
