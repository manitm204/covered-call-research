"""Raw-SVI volatility surface fitting with arbitrage diagnostics.

Purpose (research brief section 11): a stable, arbitrage-aware representation
of the implied-vol surface — NOT an alpha source. A quote far from the fitted
surface may be stale, illiquid, bad, a temporary dislocation, a fitting error,
or occasionally real; downstream logic must require executable quotes and
neighboring-strike confirmation before calling anything a mispricing.

Parameterization (Gatheral raw SVI), per expiry slice, in log-forward-moneyness
k = ln(K/F) and total implied variance w = iv^2 * T:

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

Constraints enforced in the fit: b >= 0, |rho| < 1, sigma > 0, and
min_k w(k) = a + b*sigma*sqrt(1-rho^2) >= 0.

Diagnostics:
- Butterfly arbitrage: Gatheral's g(k) >= 0 on a dense grid (g < 0 implies a
  negative implied density).
- Calendar arbitrage (surface level): total variance non-decreasing in T at
  fixed k across fitted slices.
- Fit quality: RMSE and max abs residual in vol points, per-point residuals.

Everything here is deterministic (fixed multi-starts, no randomness) and
consumes a single timestamp's quotes — strict snapshot control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import minimize

MIN_POINTS = 5


@dataclass(frozen=True, slots=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        km = k - self.m
        return self.a + self.b * (self.rho * km + np.sqrt(km**2 + self.sigma**2))

    def w_prime(self, k: np.ndarray) -> np.ndarray:
        km = k - self.m
        return self.b * (self.rho + km / np.sqrt(km**2 + self.sigma**2))

    def w_second(self, k: np.ndarray) -> np.ndarray:
        km = k - self.m
        return self.b * self.sigma**2 / (km**2 + self.sigma**2) ** 1.5


@dataclass(frozen=True, slots=True)
class SVISliceFit:
    expiration: date
    t: float
    forward: float
    params: SVIParams
    n_points: int
    rmse_vol: float
    max_abs_residual_vol: float
    butterfly_min_g: float
    butterfly_violations: int
    converged: bool

    def iv(self, strike: float | np.ndarray) -> np.ndarray:
        k = np.log(np.asarray(strike, dtype=float) / self.forward)
        w = np.maximum(self.params.total_variance(k), 1e-12)
        return np.sqrt(w / self.t)


def butterfly_g(params: SVIParams, k: np.ndarray) -> np.ndarray:
    """Gatheral's g(k); negative values imply butterfly arbitrage."""
    w = np.maximum(params.total_variance(k), 1e-12)
    wp = params.w_prime(k)
    wpp = params.w_second(k)
    return (1.0 - k * wp / (2.0 * w)) ** 2 - (wp**2 / 4.0) * (1.0 / w + 0.25) + wpp / 2.0


def fit_svi_slice(
    strikes: np.ndarray,
    ivs: np.ndarray,
    forward: float,
    t: float,
    expiration: date,
    weights: np.ndarray | None = None,
) -> SVISliceFit | None:
    """Fit one raw-SVI slice. None when inputs are insufficient/degenerate."""
    strikes = np.asarray(strikes, dtype=float)
    ivs = np.asarray(ivs, dtype=float)
    mask = np.isfinite(strikes) & np.isfinite(ivs) & (strikes > 0) & (ivs > 0)
    strikes, ivs = strikes[mask], ivs[mask]
    if len(strikes) < MIN_POINTS or t <= 0 or forward <= 0:
        return None
    k = np.log(strikes / forward)
    w_obs = ivs**2 * t
    wts = np.ones_like(w_obs) if weights is None else np.asarray(weights, dtype=float)[mask]
    wts = wts / wts.sum()

    k_lo, k_hi = float(k.min()), float(k.max())
    w_max = float(w_obs.max())
    bounds = [
        (-w_max, w_max),  # a
        (1e-8, 10.0),  # b
        (-0.999, 0.999),  # rho
        (2 * k_lo - 0.5, 2 * k_hi + 0.5),  # m
        (1e-4, 2.0),  # sigma
    ]

    def objective(x: np.ndarray) -> float:
        p = SVIParams(*x)
        return float(np.sum(wts * (p.total_variance(k) - w_obs) ** 2))

    def min_var_constraint(x: np.ndarray) -> float:
        a, b, rho, _m, sigma = x
        return a + b * sigma * math.sqrt(max(1.0 - rho**2, 0.0))

    starts = [
        np.array([float(np.median(w_obs)) * 0.8, 0.1, r, m0, 0.1])
        for r in (-0.7, 0.0, 0.7)
        for m0 in (k_lo / 2, 0.0, k_hi / 2)
    ]
    best, best_obj = None, np.inf
    for x0 in starts:
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": min_var_constraint}],
            options={"maxiter": 500, "ftol": 1e-14},
        )
        if res.fun < best_obj and min_var_constraint(res.x) > -1e-10:
            best, best_obj = res, res.fun
    if best is None:
        return None

    params = SVIParams(*best.x)
    w_fit = np.maximum(params.total_variance(k), 1e-12)
    iv_fit = np.sqrt(w_fit / t)
    resid = iv_fit - ivs
    grid = np.linspace(k_lo - 0.1, k_hi + 0.1, 201)
    g = butterfly_g(params, grid)
    return SVISliceFit(
        expiration=expiration,
        t=t,
        forward=forward,
        params=params,
        n_points=len(k),
        rmse_vol=float(np.sqrt(np.mean(resid**2))),
        max_abs_residual_vol=float(np.max(np.abs(resid))),
        butterfly_min_g=float(g.min()),
        butterfly_violations=int((g < -1e-10).sum()),
        converged=bool(best.success),
    )


@dataclass(slots=True)
class SVISurface:
    """Fitted slices for one snapshot, keyed by expiration (ascending)."""

    slices: dict[date, SVISliceFit]

    def expirations(self) -> list[date]:
        return sorted(self.slices)

    def iv(self, strike: float, expiration: date) -> float | None:
        s = self.slices.get(expiration)
        return float(s.iv(strike)[()]) if s is not None else None

    def calendar_violations(self, n_grid: int = 41, tol: float = 1e-8) -> list[dict]:
        """Total variance must be non-decreasing in T at fixed forward-moneyness.
        Returns one record per violating (expiry pair, k)."""
        out: list[dict] = []
        exps = self.expirations()
        grid = np.linspace(-0.3, 0.3, n_grid)
        for e1, e2 in zip(exps, exps[1:], strict=False):
            s1, s2 = self.slices[e1], self.slices[e2]
            w1 = s1.params.total_variance(grid)
            w2 = s2.params.total_variance(grid)
            bad = np.where(w2 < w1 - tol)[0]
            for i in bad:
                out.append(
                    {
                        "near": str(e1),
                        "far": str(e2),
                        "k": float(grid[i]),
                        "w_near": float(w1[i]),
                        "w_far": float(w2[i]),
                    }
                )
        return out

    def diagnostics(self) -> dict:
        return {
            "n_slices": len(self.slices),
            "slices": {
                str(e): {
                    "n_points": s.n_points,
                    "rmse_vol": s.rmse_vol,
                    "max_abs_residual_vol": s.max_abs_residual_vol,
                    "butterfly_min_g": s.butterfly_min_g,
                    "butterfly_violations": s.butterfly_violations,
                    "converged": s.converged,
                }
                for e, s in sorted(self.slices.items())
            },
            "calendar_violations": len(self.calendar_violations()),
        }


def fit_surface_from_chain(
    chain,
    spot: float,
    r: float,
    q: float,
    as_of: date,
    max_expiries: int = 8,
    min_points: int = MIN_POINTS,
) -> SVISurface:
    """Fit SVI slices from a canonical chain snapshot (calls with valid two-sided
    markets; IV from quote mids; forward from continuous carry)."""
    import polars as pl

    from xsp_research.domain import OptionType
    from xsp_research.options.black_scholes import forward_price, year_fraction
    from xsp_research.options.implied_vol import implied_vol

    slices: dict[date, SVISliceFit] = {}
    calls = chain.filter(
        (pl.col("option_type") == "C") & (pl.col("bid") > 0) & (pl.col("ask") > pl.col("bid"))
    )
    for expiry in sorted(calls["expiration"].unique().to_list())[:max_expiries]:
        t = year_fraction(as_of, expiry)
        if t <= 0:
            continue
        sl = calls.filter(pl.col("expiration") == expiry)
        strikes, ivs = [], []
        for row in sl.iter_rows(named=True):
            mid = 0.5 * (row["bid"] + row["ask"])
            iv = implied_vol(mid, spot, row["strike"], t, r, q, OptionType.CALL)
            if iv is not None:
                strikes.append(row["strike"])
                ivs.append(iv)
        if len(strikes) < min_points:
            continue
        fwd = forward_price(spot, r, q, t)
        fit = fit_svi_slice(np.array(strikes), np.array(ivs), fwd, t, expiry)
        if fit is not None:
            slices[expiry] = fit
    return SVISurface(slices=slices)
