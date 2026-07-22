"""Shared fixtures: hand-built quotes and small synthetic markets."""

from datetime import UTC, date, datetime

import pytest

from xsp_research.domain import BearCallSpread, OptionContract, OptionQuote, OptionType, SpreadQuote

TS = datetime(2023, 3, 1, 20, 30, tzinfo=UTC)
EXP = date(2023, 3, 31)


def make_call(strike: float) -> OptionContract:
    return OptionContract(root="XSP", expiration=EXP, strike=strike, option_type=OptionType.CALL)


def make_quote(strike: float, bid: float, ask: float, ts: datetime = TS) -> OptionQuote:
    return OptionQuote(
        contract=make_call(strike),
        ts=ts,
        bid=bid,
        ask=ask,
        bid_size=50,
        ask_size=50,
        volume=100,
        open_interest=500,
        underlying_price=400.0,
    )


@pytest.fixture
def spread_400_405() -> BearCallSpread:
    return BearCallSpread(short_leg=make_call(400.0), long_leg=make_call(405.0))


@pytest.fixture
def spread_quote_standard() -> SpreadQuote:
    """Short 400 @ 1.00/1.10, long 405 @ 0.40/0.50.

    Hand-computed: natural credit 0.50, mid credit 0.60,
    close natural 0.70, close mid 0.60.
    """
    return SpreadQuote(
        short_quote=make_quote(400.0, 1.00, 1.10),
        long_quote=make_quote(405.0, 0.40, 0.50),
    )
