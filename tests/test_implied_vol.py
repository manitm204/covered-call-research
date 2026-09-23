"""Implied-vol inversion: round trips and refusal to emit garbage."""

import pytest

from covered_call.domain import OptionType
from covered_call.options.black_scholes import bs_price
from covered_call.options.implied_vol import implied_vol


@pytest.mark.parametrize("vol", [0.08, 0.15, 0.23, 0.60, 1.50])
@pytest.mark.parametrize("strike", [360.0, 400.0, 440.0])
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_round_trip(vol, strike, option_type):
    s, t, r, q = 400.0, 30 / 365, 0.045, 0.015
    price = bs_price(s, strike, t, vol, r, q, option_type)
    if price < 1e-8:
        pytest.skip("price numerically zero; IV unidentifiable")
    recovered = implied_vol(price, s, strike, t, r, q, option_type)
    assert recovered is not None
    assert recovered == pytest.approx(vol, abs=1e-6)


def test_below_intrinsic_returns_none(_s=400.0):
    # Deep ITM call priced below its arbitrage floor
    assert implied_vol(5.0, _s, 350.0, 30 / 365, 0.045, 0.0, OptionType.CALL) is None


def test_absurd_price_returns_none():
    # Call priced above the discounted forward
    assert implied_vol(500.0, 400.0, 410.0, 30 / 365, 0.045, 0.0, OptionType.CALL) is None


def test_expired_returns_none():
    assert implied_vol(1.0, 400.0, 410.0, 0.0, 0.045, 0.0, OptionType.CALL) is None


def test_zero_price_returns_none():
    assert implied_vol(0.0, 400.0, 410.0, 30 / 365, 0.045, 0.0, OptionType.CALL) is None
