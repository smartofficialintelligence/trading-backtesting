"""Bar and Instrument invariants.

These are the row-level guarantees the rest of the system is allowed to assume.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from qresearch.data.contracts import AssetClass, Bar, Instrument, SymbolAlias

T0 = dt.datetime(2024, 3, 4, 14, 30, tzinfo=dt.UTC)


def make_bar(**overrides: Any) -> Bar:
    base: dict[str, Any] = {
        "instrument_id": "CRYPTO:BTCUSD",
        "bar_size": "1m",
        "bar_start": T0,
        "bar_end": T0 + dt.timedelta(minutes=1),
        "available_at": T0 + dt.timedelta(minutes=1, seconds=2),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000.0,
        "source": "test",
        "ingested_at": T0,
    }
    return Bar.model_validate(base | overrides)


def test_a_valid_bar_reports_its_publication_latency() -> None:
    assert make_bar().publication_latency == dt.timedelta(seconds=2)


def test_availability_may_not_precede_bar_end() -> None:
    """The core look-ahead guard: a bar cannot be known before its interval closes."""
    with pytest.raises(ValidationError, match="cannot be available before"):
        make_bar(available_at=T0 + dt.timedelta(seconds=30))


def test_availability_may_equal_bar_end() -> None:
    """Zero latency is optimistic but representable; validation warns, the contract allows."""
    assert make_bar(available_at=T0 + dt.timedelta(minutes=1)).publication_latency == dt.timedelta()


def test_bar_size_label_must_match_the_interval() -> None:
    """A '5m' label on a 1-minute interval would corrupt every downstream resample."""
    with pytest.raises(ValidationError, match="does not match declared bar_size"):
        make_bar(bar_size="5m")


def test_empty_interval_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must precede bar_end"):
        make_bar(bar_end=T0, available_at=T0)


@pytest.mark.parametrize(
    ("field", "value"),
    [("low", 100.6), ("high", 100.4), ("low", 101.5)],
    ids=["low-above-close", "high-below-close", "low-above-high"],
)
def test_broken_ohlc_relationships_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=r"OHLC relationships violated|exceeds high"):
        make_bar(**{field: value})


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_non_positive_prices_are_rejected(price: float) -> None:
    with pytest.raises(ValidationError, match="must be positive"):
        make_bar(open=price, low=min(price, 99.0))


def test_negative_volume_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_bar(volume=-1.0)


def test_vwap_outside_the_bar_range_is_rejected() -> None:
    with pytest.raises(ValidationError, match="outside the bar range"):
        make_bar(vwap=105.0)


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_bar(bar_start=T0.replace(tzinfo=None))


def test_unknown_fields_are_rejected() -> None:
    """extra='forbid' turns a typo into an error instead of a silent default."""
    with pytest.raises(ValidationError):
        make_bar(closs=100.5)


def test_bars_are_immutable() -> None:
    with pytest.raises(ValidationError):
        make_bar().close = 1.0  # type: ignore[misc]


def test_key_is_the_natural_key() -> None:
    assert make_bar().key == ("CRYPTO:BTCUSD", "1m", T0)


# -- Instrument ---------------------------------------------------------------------


def make_instrument(aliases: tuple[SymbolAlias, ...]) -> Instrument:
    return Instrument(
        instrument_id="EQ:ACME",
        asset_class=AssetClass.EQUITY,
        venue="XNAS",
        quote_currency="USD",
        price_increment=Decimal("0.01"),
        quantity_increment=Decimal("1"),
        calendar_id="XNYS:2024.1",
        aliases=aliases,
    )


def test_symbol_resolves_point_in_time_across_a_ticker_change() -> None:
    """A rename must not retroactively rewrite the identity of older bars."""
    change = dt.datetime(2024, 6, 1, tzinfo=dt.UTC)
    instrument = make_instrument(
        (
            SymbolAlias(
                symbol="OLD",
                source="prov",
                effective_from=T0 - dt.timedelta(days=500),
                effective_to=change,
            ),
            SymbolAlias(symbol="NEW", source="prov", effective_from=change),
        )
    )
    assert instrument.symbol_at(change - dt.timedelta(seconds=1), source="prov") == "OLD"
    assert instrument.symbol_at(change, source="prov") == "NEW"
    assert instrument.symbol_at(change, source="other") is None


def test_overlapping_aliases_for_one_source_are_rejected() -> None:
    """A symbol must resolve uniquely at every instant, or ingestion is ambiguous."""
    with pytest.raises(ValidationError, match="overlapping symbol aliases"):
        make_instrument(
            (
                SymbolAlias(symbol="A", source="prov", effective_from=T0),
                SymbolAlias(symbol="B", source="prov", effective_from=T0 + dt.timedelta(days=1)),
            )
        )


def test_the_same_symbol_may_be_reused_by_different_sources() -> None:
    instrument = make_instrument(
        (
            SymbolAlias(symbol="ACME", source="a", effective_from=T0),
            SymbolAlias(symbol="ACME.US", source="b", effective_from=T0),
        )
    )
    assert instrument.symbol_at(T0, source="b") == "ACME.US"


def test_alias_interval_must_be_non_empty() -> None:
    with pytest.raises(ValidationError, match="effective_to <= effective_from"):
        SymbolAlias(symbol="A", source="p", effective_from=T0, effective_to=T0)
