"""EXPLORATORY veto split test (descriptive; all three chain datasets burned).

Unified Rule v2 veto = (a) rebound window: index closed >=10% below its running
high within the past 60 sessions, OR (b) index below its 200-session MA, OR
(c) own vol index in its top tercile (2018-08..2026-07 window). All flags
lagged one session. For each ETF, engine cap-2 base fills on three arms:
veto-CLEAR entries only, veto-ACTIVE entries only, and unfiltered baseline.
Question: how much is the veto saving?
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from xsp_research.backtest.american import DividendCalendar
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.config import EXECUTION_SCENARIOS, load_strategy_config
from xsp_research.evaluation.robustness import bootstrap_mean_ci
from xsp_research.ingestion.file_provider import ParquetUnderlyingProvider, SeriesRatesProvider

sys.path.insert(0, "scripts")
from full_ablation_sweep import RATES, PreloadedOptionsProvider  # noqa: E402
from phaseA_aggression_frontier import episode_boot_ci, episode_ids  # noqa: E402

import datetime as dt

W0, W1 = dt.date(2018, 8, 1), dt.date(2026, 7, 22)

ETFS = {
    "SPY": {
        "config": "configs/strategy_spy_hold_to_expiry.yaml", "width": 8.0,
        "glob": "data/normalized/options/spy/chain_*.parquet",
        "und": "data/normalized/options/spy/underlying_eod.parquet",
        "div": "data/normalized/aux/SPY_DIVIDENDS.parquet",
        "index": "data/normalized/aux/SPX.parquet", "vol": "data/normalized/aux/VIX.parquet",
    },
    "QQQ": {
        "config": "configs/strategy_qqq_prereg.yaml", "width": None,
        "glob": "data/normalized/options/qqq/chain_*.parquet",
        "und": "data/normalized/options/qqq/underlying_eod.parquet",
        "div": "data/normalized/aux/QQQ_DIVIDENDS.parquet",
        "index": "data/normalized/aux/NDX.parquet", "vol": "data/normalized/aux/VXN.parquet",
    },
    "IWM": {
        "config": "configs/strategy_iwm_prereg_h1.yaml", "width": None,
        "glob": "data/normalized/options/iwm/chain_*.parquet",
        "und": "data/normalized/options/iwm/underlying_eod.parquet",
        "div": "data/normalized/aux/IWM_DIVIDENDS.parquet",
        "index": "data/normalized/aux/RUT.parquet", "vol": "data/normalized/aux/RVX.parquet",
    },
}


def veto_flags(index_path: str, vol_path: str) -> dict:
    """date -> True if veto ACTIVE (flags computed at date, consumed lagged)."""
    idx = pl.read_parquet(index_path).sort("date")
    dates = idx["date"].to_list()
    closes = np.array(idx["close"].to_list())
    runmax = np.maximum.accumulate(closes)
    dd = closes / runmax - 1
    dd_min60 = np.array([dd[max(0, i - 60):i + 1].min() for i in range(len(dd))])
    ma200 = np.full(len(closes), np.nan)
    for i in range(199, len(closes)):
        ma200[i] = closes[i - 199:i + 1].mean()
    below_ma = closes < ma200

    vol = pl.read_parquet(vol_path).sort("date")
    vwin = vol.filter((pl.col("date") >= W0) & (pl.col("date") <= W1))
    vhi = float(np.quantile(vwin["close"].to_numpy(), 2 / 3))
    vmap = dict(zip(vol["date"].to_list(), vol["close"].to_list()))

    flags = {}
    for i, d in enumerate(dates):
        v = vmap.get(d)
        if np.isnan(ma200[i]) or v is None:
            flags[d] = None  # undefined -> no trade in either arm
            continue
        flags[d] = bool(dd_min60[i] <= -0.10 or below_ma[i] or v >= vhi)
    return flags


def make_gate(flags: dict, want_active: bool):
    dates = sorted(flags)

    def gate(session):
        import bisect
        j = bisect.bisect_left(dates, session) - 1  # latest flag date < session
        if j < 0:
            return False, "no history"
        f = flags[dates[j]]
        if f is None:
            return False, "flag undefined"
        return (f == want_active, "ok")

    return gate


for sym, spec in ETFS.items():
    flags = veto_flags(spec["index"], spec["vol"])
    n_def = [f for d, f in flags.items() if W0 <= d <= W1 and f is not None]
    print(f"\n{'='*74}\n{sym}: veto active {np.mean(n_def):.0%} of {len(n_def)} in-window sessions")

    cfg0 = load_strategy_config(spec["config"])
    sel = cfg0.selection.model_copy(update={"fixed_width": spec["width"]}) \
        if spec["width"] else cfg0.selection
    cfg = cfg0.model_copy(update={
        "selection": sel,
        "execution": EXECUTION_SCENARIOS["base"],
        "sizing": cfg0.sizing.model_copy(update={
            "contracts_per_entry": 1, "max_open_positions": 2}),
    })
    options = PreloadedOptionsProvider(spec["glob"], sym, cfg0.backtest.mark_time_et)
    und = ParquetUnderlyingProvider(spec["und"], sym)
    rates = SeriesRatesProvider(RATES)
    div = DividendCalendar.from_parquet(spec["div"])

    print(f"{'arm':14s} {'n':>3s} {'mean$':>7s} {'tradeCI':>18s} {'epCI':>16s} {'nep':>3s} "
          f"{'win':>5s} {'worst':>6s} {'total':>6s}")
    arms = [("veto-CLEAR", make_gate(flags, False)),
            ("veto-ACTIVE", make_gate(flags, True)),
            ("unfiltered", None)]
    detail = {}
    for name, gate in arms:
        res = BacktestEngine(cfg, options, und, rates, execution_scenario="base",
                             entry_gate=gate, dividends=div).run()
        t = res.trades_frame()
        if not t.height:
            print(f"{name:14s}   0")
            continue
        t = t.with_columns(
            (pl.col("realized_net") / pl.col("qty")).alias("pnl"),
            pl.col("entry_ts").dt.convert_time_zone("America/New_York")
            .dt.date().alias("entry_date"))
        per = t["pnl"].to_numpy()
        ci = bootstrap_mean_ci(per, n_boot=2000, block=5, seed=7)
        lo, hi, nep = episode_boot_ci(per, episode_ids(t["entry_date"].to_numpy()))
        detail[name] = t
        print(f"{name:14s} {t.height:3d} {per.mean():7.2f} "
              f"[{ci['ci_low']:7.2f},{ci['ci_high']:7.2f}] [{lo:6.1f},{hi:6.1f}] {nep:3d} "
              f"{(per > 0).mean():5.2f} {per.min():6.0f} {per.sum():6.0f}")

    print("yearly means (n):")
    years = list(range(2018, 2027))
    for name, t in detail.items():
        yr = t["entry_date"].dt.year().to_numpy()
        per = t["pnl"].to_numpy()
        cells = []
        for y in years:
            m = yr == y
            cells.append(f"{per[m].mean():6.0f}({m.sum():2d})" if m.sum() else "     .    ")
        print(f"  {name:12s} " + " ".join(cells))
