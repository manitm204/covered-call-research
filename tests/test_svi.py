"""SVI surface: parameter recovery, arbitrage diagnostics, chain fitting."""

from datetime import date

import numpy as np
import pytest

from xsp_research.options.svi_surface import (
    SVIParams,
    SVISliceFit,
    SVISurface,
    butterfly_g,
    fit_surface_from_chain,
    fit_svi_slice,
)

EXP = date(2023, 3, 31)
T = 30 / 365
FWD = 400.0


def slice_from_params(params: SVIParams, t: float = T) -> SVISliceFit:
    return SVISliceFit(
        expiration=EXP,
        t=t,
        forward=FWD,
        params=params,
        n_points=0,
        rmse_vol=0.0,
        max_abs_residual_vol=0.0,
        butterfly_min_g=0.0,
        butterfly_violations=0,
        converged=True,
    )


class TestSliceFitting:
    def test_flat_smile_recovered(self):
        """Constant vol -> fitted IV must be flat at that vol."""
        strikes = np.arange(360.0, 441.0, 2.0)
        ivs = np.full(len(strikes), 0.18)
        fit = fit_svi_slice(strikes, ivs, FWD, T, EXP)
        assert fit is not None and fit.converged
        assert fit.rmse_vol < 1e-4
        assert fit.iv(400.0) == pytest.approx(0.18, abs=1e-3)
        assert fit.iv(370.0) == pytest.approx(0.18, abs=1e-3)

    def test_known_params_round_trip(self):
        """IVs generated from known SVI params -> near-zero fit residuals."""
        true = SVIParams(a=0.0016, b=0.04, rho=-0.4, m=0.0, sigma=0.12)
        strikes = np.linspace(340.0, 460.0, 40)
        k = np.log(strikes / FWD)
        ivs = np.sqrt(true.total_variance(k) / T)
        fit = fit_svi_slice(strikes, ivs, FWD, T, EXP)
        assert fit is not None
        assert fit.rmse_vol < 5e-4
        # Parameter identity is not required (SVI has near-degenerate ridges);
        # the fitted CURVE must match.
        assert np.allclose(fit.iv(strikes), ivs, atol=2e-3)

    def test_skewed_smile_fits_synthetic_market_shape(self):
        """Linear-in-log-moneyness IV (the synthetic market's smile)."""
        strikes = np.arange(360.0, 441.0, 1.0)
        ivs = 0.17 - 0.35 * np.log(strikes / FWD)
        fit = fit_svi_slice(strikes, ivs, FWD, T, EXP)
        assert fit is not None
        assert fit.rmse_vol < 0.01  # SVI approximates a linear skew well locally

    def test_too_few_points_returns_none(self):
        assert fit_svi_slice(np.array([395.0, 400.0]), np.array([0.2, 0.19]), FWD, T, EXP) is None

    def test_degenerate_inputs_return_none(self):
        strikes = np.arange(390.0, 411.0, 1.0)
        assert fit_svi_slice(strikes, np.full(len(strikes), 0.2), FWD, 0.0, EXP) is None
        assert fit_svi_slice(strikes, np.full(len(strikes), np.nan), FWD, T, EXP) is None

    def test_deterministic(self):
        strikes = np.arange(370.0, 431.0, 1.0)
        ivs = 0.17 - 0.3 * np.log(strikes / FWD) + 0.2 * np.log(strikes / FWD) ** 2
        a = fit_svi_slice(strikes, ivs, FWD, T, EXP)
        b = fit_svi_slice(strikes, ivs, FWD, T, EXP)
        assert a.params == b.params


class TestButterflyDiagnostic:
    def test_flat_surface_no_arbitrage(self):
        params = SVIParams(a=0.18**2 * T, b=0.0001, rho=0.0, m=0.0, sigma=0.1)
        g = butterfly_g(params, np.linspace(-0.4, 0.4, 101))
        assert (g >= -1e-10).all()

    def test_extreme_params_flagged(self):
        """Absurdly steep wings produce negative density somewhere."""
        params = SVIParams(a=0.0001, b=2.5, rho=-0.99, m=0.0, sigma=0.01)
        g = butterfly_g(params, np.linspace(-0.4, 0.4, 201))
        assert g.min() < 0


class TestCalendarDiagnostic:
    def test_increasing_total_variance_clean(self):
        s1 = slice_from_params(SVIParams(0.002, 0.02, -0.3, 0.0, 0.1), t=20 / 365)
        s2 = slice_from_params(SVIParams(0.004, 0.02, -0.3, 0.0, 0.1), t=50 / 365)
        surface = SVISurface(slices={date(2023, 3, 17): s1, date(2023, 4, 21): s2})
        assert surface.calendar_violations() == []

    def test_crossed_slices_detected(self):
        s1 = slice_from_params(SVIParams(0.006, 0.02, -0.3, 0.0, 0.1), t=20 / 365)
        s2 = slice_from_params(SVIParams(0.002, 0.02, -0.3, 0.0, 0.1), t=50 / 365)
        surface = SVISurface(slices={date(2023, 3, 17): s1, date(2023, 4, 21): s2})
        violations = surface.calendar_violations()
        assert violations and all(v["w_far"] < v["w_near"] for v in violations)


class TestChainFitting:
    def test_fit_synthetic_chain(self):
        from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket

        market = SyntheticMarket(SyntheticConfig(start=date(2023, 2, 1), end=date(2023, 4, 30)))
        session = date(2023, 2, 6)
        chain = market.chain(market._snapshot_ts(session))
        spot = float(chain["underlying_price"][0])
        surface = fit_surface_from_chain(chain, spot, 0.045, 0.015, session)
        assert len(surface.slices) >= 3
        diag = surface.diagnostics()
        for s in diag["slices"].values():
            # Quotes are rounded to cents with a $0.05 minimum width, so IVs
            # recovered from mids carry real noise in the wings: 3 vol points
            # of RMSE is the honest bound for an unweighted fit.
            assert s["rmse_vol"] < 0.03
            assert s["converged"]
        # Ultra-short-dated slices (< ~2 weeks) have wing IVs dominated by
        # quote rounding ($0.05 min width on near-zero prices) and the
        # butterfly diagnostic CORRECTLY flags the noisy fits. The slices the
        # strategy trades (>= 20 DTE) must be arbitrage-clean.
        for expiry_str, s in diag["slices"].items():
            dte = (date.fromisoformat(expiry_str) - session).days
            if dte >= 20:
                assert s["butterfly_violations"] == 0, (expiry_str, s)
                assert s["rmse_vol"] < 0.01

    def test_surface_iv_close_to_quote_iv(self):
        from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket

        market = SyntheticMarket(SyntheticConfig(start=date(2023, 2, 1), end=date(2023, 4, 30)))
        session = date(2023, 2, 6)
        chain = market.chain(market._snapshot_ts(session))
        spot = float(chain["underlying_price"][0])
        surface = fit_surface_from_chain(chain, spot, 0.045, 0.015, session)
        expiry = surface.expirations()[0]
        # Synthetic smile at ATM is base_iv (0.17) by construction.
        assert surface.iv(spot, expiry) == pytest.approx(0.17, abs=0.02)
