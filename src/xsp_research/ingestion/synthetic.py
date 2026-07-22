"""SYNTHETIC market data generator — FOR SOFTWARE TESTING ONLY.

============================================================================
WARNING: Everything produced here is simulated. It exists so the backtester,
ledger, selection and execution logic can be exercised end-to-end and unit
tested against known-good values. Results computed on this data are NEVER
evidence about real-world strategy profitability, and every provider tags
its output with data_source="synthetic" so downstream reports can refuse to
present it as such.
============================================================================

Model: geometric Brownian motion underlying (deterministic seed), a simple
static smile (linear put-side skew in log-moneyness), monthly (3rd Friday) and
weekly (Friday) expirations, strikes on a $1 grid, and quotes built as
Black-Scholes theoretical value +/- half of a price-dependent bid-ask spread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from xsp_research.domain import OptionType
from xsp_research.ingestion.base import validate_chain_frame
from xsp_research.options.black_scholes import bs_price, year_fraction

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


def _fridays(start: date, end: date) -> list[date]:
    d = start + timedelta(days=(4 - start.weekday()) % 7)
    out = []
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


def _third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
    return first_friday + timedelta(days=14)


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 7
    start: date = date(2023, 1, 2)
    end: date = date(2024, 12, 31)
    s0: float = 400.0  # XSP-like level
    drift: float = 0.07
    vol: float = 0.16  # realized vol of the GBM path
    rate: float = 0.045
    div_yield: float = 0.015
    base_iv: float = 0.17  # ATM implied vol used to quote options
    skew: float = -0.35  # dIV/dlog-moneyness (negative => call wing cheaper)
    min_tick_spread: float = 0.05  # minimum quote width, index points
    rel_spread: float = 0.06  # quote width as fraction of theo price
    strike_pct_range: float = 0.25  # strikes within +/- this fraction of spot
    snapshot_time_et: time = time(15, 30)


class SyntheticMarket:
    """Deterministic synthetic market; provides all three provider Protocols."""

    def __init__(self, cfg: SyntheticConfig | None = None) -> None:
        self.cfg = cfg = cfg if cfg is not None else SyntheticConfig()
        self._dates: list[date] = [
            d
            for d in (cfg.start + timedelta(days=i) for i in range((cfg.end - cfg.start).days + 1))
            if d.weekday() < 5
        ]
        rng = np.random.default_rng(cfg.seed)
        n = len(self._dates)
        dt = 1.0 / 252.0
        shocks = rng.standard_normal(n)
        log_path = np.cumsum((cfg.drift - 0.5 * cfg.vol**2) * dt + cfg.vol * math.sqrt(dt) * shocks)
        closes = cfg.s0 * np.exp(np.concatenate(([0.0], log_path[:-1])))
        self._close = dict(zip(self._dates, closes.tolist(), strict=False))

        expiries = set(_fridays(cfg.start, cfg.end + timedelta(days=60)))
        y_m = set()
        d = cfg.start
        while d <= cfg.end + timedelta(days=60):
            y_m.add((d.year, d.month))
            d += timedelta(days=28)
        expiries |= {_third_friday(y, m) for (y, m) in y_m}
        self._expirations = sorted(expiries)

    # ------------------------------------------------------------------ shared
    @property
    def data_source(self) -> str:
        return "synthetic"

    # ------------------------------------------------- UnderlyingProvider API
    def trading_dates(self, start: date, end: date) -> list[date]:
        return [d for d in self._dates if start <= d <= end]

    def close(self, as_of: date) -> float | None:
        return self._close.get(as_of)

    def closes(self, start: date, end: date) -> pl.DataFrame:
        ds = self.trading_dates(start, end)
        return pl.DataFrame({"date": ds, "close": [self._close[d] for d in ds]})

    # ------------------------------------------------------ RatesProvider API
    def rate(self, as_of: date) -> float:
        return self.cfg.rate

    # ---------------------------------------------------- OptionsProvider API
    def expirations(self, as_of: date) -> list[date]:
        return [e for e in self._expirations if e >= as_of]

    def _iv(self, spot: float, strike: float) -> float:
        return max(0.05, self.cfg.base_iv + self.cfg.skew * math.log(strike / spot))

    def chain(self, as_of: datetime) -> pl.DataFrame:
        """Snapshot at the fixture's single daily snapshot time (<= as_of required)."""
        d = as_of.date()
        spot = self._close.get(d)
        if spot is None:
            return pl.DataFrame(schema=list(_chain_schema().items()))
        snap = self._snapshot_ts(d)
        if as_of < snap:
            # The day's snapshot is not yet observable: nothing to return.
            return pl.DataFrame(schema=list(_chain_schema().items()))

        rows: dict[str, list] = {k: [] for k in _chain_schema()}
        lo = int(spot * (1 - self.cfg.strike_pct_range))
        hi = int(spot * (1 + self.cfg.strike_pct_range)) + 1
        for exp in self._expirations:
            if not (d < exp <= d + timedelta(days=70)):
                continue
            t = year_fraction(d, exp)
            for strike in range(lo, hi):
                k = float(strike)
                iv = self._iv(spot, k)
                for ot in (OptionType.CALL, OptionType.PUT):
                    theo = bs_price(spot, k, t, iv, self.cfg.rate, self.cfg.div_yield, ot)
                    half = max(self.cfg.min_tick_spread, self.cfg.rel_spread * theo) / 2.0
                    bid = max(0.0, round(theo - half, 2))
                    ask = round(theo + half, 2)
                    if ask <= bid:
                        ask = bid + 0.05
                    rows["ts"].append(snap)
                    rows["root"].append("XSP")
                    rows["expiration"].append(exp)
                    rows["strike"].append(k)
                    rows["option_type"].append(ot.value)
                    rows["bid"].append(bid)
                    rows["ask"].append(ask)
                    rows["bid_size"].append(50)
                    rows["ask_size"].append(50)
                    rows["volume"].append(100)
                    rows["open_interest"].append(500)
                    rows["underlying_price"].append(spot)
        df = pl.DataFrame(rows, schema=_chain_schema())
        validate_chain_frame(df)
        return df

    def _snapshot_ts(self, d: date) -> datetime:
        t = self.cfg.snapshot_time_et
        return datetime(d.year, d.month, d.day, t.hour, t.minute, tzinfo=_ET).astimezone(_UTC)


def _chain_schema() -> dict[str, pl.DataType]:
    from xsp_research.ingestion.base import CHAIN_SCHEMA

    return dict(CHAIN_SCHEMA)


def research_bundle(market: SyntheticMarket, seed: int = 11):
    """SYNTHETIC research universe for feature/experiment testing only.

    Builds ETF/VIX/rate proxies deterministically from the market's underlying
    path via a one-factor model (market beta + seeded idiosyncratic noise).
    Every series is simulated; results derived from it are software validation,
    never research evidence.
    """
    from xsp_research.features.registry import MarketDataBundle

    rng = np.random.default_rng(seed)
    dates = market._dates
    closes = np.array([market._close[d] for d in dates])
    mkt_ret = np.diff(np.log(closes), prepend=np.log(closes[0]))
    mkt_ret[0] = 0.0
    n = len(dates)

    def factor_series(beta: float, idio: float, s0: float = 100.0) -> pl.DataFrame:
        rets = beta * mkt_ret + idio * rng.standard_normal(n)
        rets[0] = 0.0
        return pl.DataFrame({"date": dates, "close": s0 * np.exp(np.cumsum(rets))})

    # Trailing realized vol (annualized) drives the synthetic VIX level.
    rv = np.full(n, market.cfg.base_iv)
    for i in range(20, n):
        rv[i] = float(np.std(mkt_ret[i - 19 : i + 1], ddof=1) * math.sqrt(252.0))
    vix = np.clip(100.0 * rv * 1.15 + rng.normal(0.0, 0.6, n), 9.0, 90.0)
    vix9d = np.clip(vix * (0.97 + 0.06 * rng.standard_normal(n) * 0.1), 8.0, 95.0)
    vvix = np.clip(
        85.0 + 4.0 * (vix - vix.mean()) / max(vix.std(), 1e-9) + rng.normal(0.0, 2.0, n),
        60.0,
        180.0,
    )
    rate = np.clip(market.cfg.rate + np.cumsum(rng.normal(0.0, 1e-4, n)), 0.0, 0.10)

    series: dict[str, pl.DataFrame] = {
        "UNDERLYING": pl.DataFrame({"date": dates, "close": closes}),
        "SPY": factor_series(1.0, 0.0005, 450.0),
        "RSP": factor_series(0.95, 0.002, 150.0),
        "QQQ": factor_series(1.15, 0.004, 380.0),
        "IWM": factor_series(1.05, 0.006, 190.0),
        "HYG": factor_series(0.30, 0.002, 75.0),
        "LQD": factor_series(0.10, 0.003, 105.0),
        "VIX": pl.DataFrame({"date": dates, "close": vix}),
        "VIX9D": pl.DataFrame({"date": dates, "close": vix9d}),
        "VVIX": pl.DataFrame({"date": dates, "close": vvix}),
        "RATE_3M": pl.DataFrame({"date": dates, "close": rate}),
    }
    sector_symbols = tuple(f"SEC{i}" for i in range(1, 9))
    betas = np.linspace(0.7, 1.3, len(sector_symbols))
    for sym, beta in zip(sector_symbols, betas, strict=True):
        series[sym] = factor_series(float(beta), 0.008)
    return MarketDataBundle(series=series, sector_symbols=sector_symbols)
