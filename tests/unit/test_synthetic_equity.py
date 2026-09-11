"""The synthetic equity sample respects the XNYS calendar."""

from __future__ import annotations

import datetime as dt

from qresearch.data.calendars import XNYSCalendar
from qresearch.data.contracts import AssetClass
from qresearch.data.synthetic import SyntheticSpec, generate_rows, instruments_for


def test_equity_rows_fall_only_inside_sessions() -> None:
    spec = SyntheticSpec.equity()
    rows = generate_rows(spec)
    calendar = XNYSCalendar()
    assert rows, "two sessions of bars"
    assert all(calendar.session_at(r.timestamp) is not None for r in rows)
    dates = {r.timestamp.astimezone(dt.timezone(dt.timedelta(hours=-5))).date() for r in rows}
    assert dates == {dt.date(2024, 3, 4), dt.date(2024, 3, 5)}
    per_symbol = {}
    for r in rows:
        per_symbol.setdefault(r.symbol, set()).add(r.timestamp)
    assert all(len(v) <= 2 * 390 for v in per_symbol.values())


def test_equity_instruments_are_whole_share_xnys() -> None:
    instruments = instruments_for(SyntheticSpec.equity())
    assert {i.asset_class for i in instruments} == {AssetClass.EQUITY}
    assert {str(i.quantity_increment) for i in instruments} == {"1"}
    assert {i.calendar_id for i in instruments} == {"XNYS:1"}
    assert all(i.base_currency is None for i in instruments)
