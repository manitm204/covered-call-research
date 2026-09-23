"""Black-Scholes pricing and Greeks against hand-known values and structure."""

import math

import pytest

from covered_call.domain import OptionType
from covered_call.options.black_scholes import (
    black76_price,
    bs_greeks,
    bs_price,
    forward_price,
    year_fraction,
)


class TestKnownValues:
    def test_textbook_atm_call(self):
        # Classic reference: S=100, K=100, T=1, r=5%, q=0, vol=20% -> 10.4506
        price = bs_price(100.0, 100.0, 1.0, 0.20, 0.05, 0.0, OptionType.CALL)
        assert price == pytest.approx(10.450584, abs=1e-4)

    def test_put_via_parity(self):
        call = bs_price(100.0, 100.0, 1.0, 0.20, 0.05, 0.0, OptionType.CALL)
        put = bs_price(100.0, 100.0, 1.0, 0.20, 0.05, 0.0, OptionType.PUT)
        # C - P = S e^{-qT} - K e^{-rT}
        assert call - put == pytest.approx(100.0 - 100.0 * math.exp(-0.05), abs=1e-10)

    def test_put_call_parity_with_dividends(self):
        s, k, t, vol, r, q = 400.0, 410.0, 30 / 365, 0.18, 0.045, 0.015
        call = bs_price(s, k, t, vol, r, q, OptionType.CALL)
        put = bs_price(s, k, t, vol, r, q, OptionType.PUT)
        assert call - put == pytest.approx(s * math.exp(-q * t) - k * math.exp(-r * t), abs=1e-10)

    def test_black76_matches_spot_form(self):
        s, k, t, vol, r, q = 400.0, 420.0, 0.1, 0.2, 0.05, 0.02
        f = forward_price(s, r, q, t)
        assert black76_price(f, k, t, vol, math.exp(-r * t), OptionType.CALL) == pytest.approx(
            bs_price(s, k, t, vol, r, q, OptionType.CALL), abs=1e-12
        )


class TestDegenerateCases:
    def test_expired_option_is_intrinsic(self):
        assert bs_price(110.0, 100.0, 0.0, 0.2, 0.05, 0.0, OptionType.CALL) == pytest.approx(10.0)
        assert bs_price(90.0, 100.0, 0.0, 0.2, 0.05, 0.0, OptionType.CALL) == 0.0

    def test_zero_vol_is_discounted_forward_intrinsic(self):
        s, k, t, r = 100.0, 90.0, 1.0, 0.05
        expected = math.exp(-r * t) * (forward_price(s, r, 0.0, t) - k)
        assert bs_price(s, k, t, 0.0, r, 0.0, OptionType.CALL) == pytest.approx(expected)


class TestGreeks:
    S, K, T, VOL, R, Q = 400.0, 420.0, 30 / 365, 0.18, 0.045, 0.015

    def _fd(self, f, x, h):
        return (f(x + h) - f(x - h)) / (2 * h)

    def test_delta_matches_finite_difference(self):
        g = bs_greeks(self.S, self.K, self.T, self.VOL, self.R, self.Q, OptionType.CALL)
        fd = self._fd(
            lambda s: bs_price(s, self.K, self.T, self.VOL, self.R, self.Q, OptionType.CALL),
            self.S,
            1e-3,
        )
        assert g.delta == pytest.approx(fd, abs=1e-6)

    def test_gamma_matches_finite_difference(self):
        g = bs_greeks(self.S, self.K, self.T, self.VOL, self.R, self.Q, OptionType.CALL)
        h = 1e-2

        def f(s):
            return bs_price(s, self.K, self.T, self.VOL, self.R, self.Q, OptionType.CALL)

        fd = (f(self.S + h) - 2 * f(self.S) + f(self.S - h)) / h**2
        assert g.gamma == pytest.approx(fd, rel=1e-4)

    def test_vega_matches_finite_difference(self):
        g = bs_greeks(self.S, self.K, self.T, self.VOL, self.R, self.Q, OptionType.CALL)
        fd = self._fd(
            lambda v: bs_price(self.S, self.K, self.T, v, self.R, self.Q, OptionType.CALL),
            self.VOL,
            1e-5,
        )
        assert g.vega == pytest.approx(fd, rel=1e-6)

    def test_theta_matches_finite_difference(self):
        g = bs_greeks(self.S, self.K, self.T, self.VOL, self.R, self.Q, OptionType.CALL)
        fd = self._fd(
            lambda t: bs_price(self.S, self.K, t, self.VOL, self.R, self.Q, OptionType.CALL),
            self.T,
            1e-6,
        )
        assert g.theta == pytest.approx(-fd, rel=1e-4)

    def test_put_delta_negative_call_delta_positive(self):
        c = bs_greeks(self.S, self.K, self.T, self.VOL, self.R, self.Q, OptionType.CALL)
        p = bs_greeks(self.S, self.K, self.T, self.VOL, self.R, self.Q, OptionType.PUT)
        assert 0 < c.delta < 1
        assert -1 < p.delta < 0


def test_year_fraction():
    from datetime import date

    assert year_fraction(date(2023, 1, 1), date(2023, 1, 31)) == pytest.approx(30 / 365)
    assert year_fraction(date(2023, 1, 31), date(2023, 1, 1)) == 0.0
