"""EXPLORATORY confirmation-layer ablation on top of the frozen veto.

Architecture under test = Unified Rule v2: the veto (rebound window OR below
MA200 OR own-vol top tercile, lagged 1 session) is ALWAYS applied; the
ablation sweeps every AND-combination of the IC-bridge confirmation signals
per ETF, engine cap-2, base fills. Includes a veto-only baseline (mask 0).
Descriptive only — all three chain datasets are burned.

  SPY (31 combos): seccorr60<0.45, trend84>10%, absorb_fall, rsi>60, ma200>+5%
  QQQ (63 combos): trend84>10%, seccorr60<0.45, absorb_fall, beta>1.30,
                   ma200>+5%, rsi>60
  IWM (31 combos): seccorr60<0.45, beta>1.30, rvx<25, trend84>10%, ivrank<0.50
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
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

AUX = "data/normalized/aux"
W0, W1 = dt.date(2018, 8, 1), dt.date(2026, 7, 22)
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]
SPY_FEATURES = "reports/experiments/spy_regime_rsi70_ivrich-20260723-204901-b6454793/features.parquet"
REGIME_V2 = "results/ablation_full/regime_features_v2.parquet"

ETFS = {
    "SPY": {
        "config": "configs/strategy_spy_hold_to_expiry.yaml", "width": 8.0,
        "glob": "data/normalized/options/spy/chain_*.parquet",
        "und": "data/normalized/options/spy/underlying_eod.parquet",
        "div": f"{AUX}/SPY_DIVIDENDS.parquet",
        "index": f"{AUX}/SPX.parquet", "vol": f"{AUX}/VIX.parquet",
        "features": SPY_FEATURES,
        "signals": {
            "seccorr60": ("sector_corr_60", "<", 0.45),
            "trend84":   ("ret_84d", ">", 0.10),
            "absorb":    ("absorption_chg_20d", "<", 0.0),
            "rsi60":     ("rsi_14", ">", 60.0),
            "ma200_5":   ("dist_ma200", ">", 0.05),
        },
    },
    "QQQ": {
        "config": "configs/strategy_qqq_prereg.yaml", "width": None,
        "glob": "data/normalized/options/qqq/chain_*.parquet",
        "und": "data/normalized/options/qqq/underlying_eod.parquet",
        "div": f"{AUX}/QQQ_DIVIDENDS.parquet",
        "index": f"{AUX}/NDX.parquet", "vol": f"{AUX}/VXN.parquet",
        "features": "results/qqq_prereg/features_qqq.parquet",
        "signals": {
            "trend84":   ("ret_84d", ">", 0.10),
            "seccorr60": ("sector_corr_60", "<", 0.45),
            "absorb":    ("absorption_chg_20d", "<", 0.0),
            "beta13":    ("qqq_beta_spy_60", ">", 1.30),
            "ma200_5":   ("dist_ma200", ">", 0.05),
            "rsi60":     ("rsi_14", ">", 60.0),
        },
    },
    "IWM": {
        "config": "configs/strategy_iwm_prereg_h1.yaml", "width": None,
        "glob": "data/normalized/options/iwm/chain_*.parquet",
        "und": "data/normalized/options/iwm/underlying_eod.parquet",
        "div": f"{AUX}/IWM_DIVIDENDS.parquet",
        "index": f"{AUX}/RUT.parquet", "vol": f"{AUX}/RVX.parquet",
        "features": "results/iwm_prereg/features_iwm.parquet",
        "signals": {
            "seccorr60": ("sector_corr_60", "<", 0.45),
            "beta13":    ("iwm_beta_spy_60", ">", 1.30),
            "rvx25":     ("vix_level", "<", 25.0),
            "trend84":   ("ret_84d", ">", 0.10),
            "ivrank50":  ("vol_rank_252", "<", 0.50),
        },
    },
}


def sector_corr_60() -> pl.DataFrame:
    frames = [pl.read_parquet(f"{AUX}/{s}.parquet").sort("date")
              .with_columns((pl.col("close") / pl.col("close").shift(1)).log().alias(s))
              .select("date", s) for s in SECTORS]
    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, on="date", how="inner")
    df = df.drop_nulls().sort("date")
    dates = df["date"].to_list()
    R = df.select(SECTORS).to_numpy()
    iu = np.triu_indices(len(SECTORS), k=1)
    out = [{"date": dates[i], "sector_corr_60": float(np.corrcoef(R[i - 59:i + 1].T)[iu].mean())}
           for i in range(59, len(dates))]
    return pl.DataFrame(out)


def series_extras(index_path: str, vol_path: str) -> pl.DataFrame:
    idx = pl.read_parquet(index_path).sort("date").with_columns(
        (pl.col("close") / pl.col("close").shift(84) - 1).alias("ret_84d"))
    vol = pl.read_parquet(vol_path).sort("date")
    v = vol["close"].to_numpy()
    rank = np.full(len(v), np.nan)
    for i in range(251, len(v)):
        rank[i] = (v[i - 251:i + 1] <= v[i]).mean()
    vol = vol.with_columns(pl.Series("vol_rank_252", rank))
    return idx.select("date", "ret_84d").join(
        vol.select("date", "vol_rank_252"), on="date", how="full", coalesce=True).sort("date")


def veto_flags(index_path: str, vol_path: str) -> tuple[list, dict]:
    idx = pl.read_parquet(index_path).sort("date")
    dates = idx["date"].to_list()
    closes = np.array(idx["close"].to_list())
    dd = closes / np.maximum.accumulate(closes) - 1
    dd_min60 = np.array([dd[max(0, i - 60):i + 1].min() for i in range(len(dd))])
    ma200 = np.full(len(closes), np.nan)
    for i in range(199, len(closes)):
        ma200[i] = closes[i - 199:i + 1].mean()
    vol = pl.read_parquet(vol_path).sort("date")
    vwin = vol.filter((pl.col("date") >= W0) & (pl.col("date") <= W1))
    vhi = float(np.quantile(vwin["close"].to_numpy(), 2 / 3))
    vmap = dict(zip(vol["date"].to_list(), vol["close"].to_list()))
    flags = {}
    for i, d in enumerate(dates):
        v = vmap.get(d)
        flags[d] = None if (np.isnan(ma200[i]) or v is None) else \
            bool(dd_min60[i] <= -0.10 or closes[i] < ma200[i] or v >= vhi)
    return sorted(flags), flags


SC60 = sector_corr_60()

for sym, spec in ETFS.items():
    names = list(spec["signals"])
    feats = pl.read_parquet(spec["features"]).join(
        pl.read_parquet(REGIME_V2), on="date", how="left")
    feats = feats.join(
        series_extras(spec["index"], spec["vol"]).sort("date").with_columns(
            pl.col("ret_84d", "vol_rank_252").shift(1)),
        on="date", how="left").join(
        SC60.sort("date").with_columns(pl.col("sector_corr_60").shift(1)),
        on="date", how="left")
    rows = {r["date"]: r for r in feats.iter_rows(named=True)}
    vdates, vflags = veto_flags(spec["index"], spec["vol"])

    def veto_clear(session):
        j = bisect.bisect_left(vdates, session) - 1
        if j < 0:
            return False
        f = vflags[vdates[j]]
        return f is False

    def sig_on(r, name):
        col, op, thr = spec["signals"][name]
        v = r.get(col)
        if v is None:
            raise TypeError
        return (v < thr) if op == "<" else (v > thr)

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

    results = []
    for mask in range(0, 1 << len(names)):
        active = [names[i] for i in range(len(names)) if mask >> i & 1]

        def gate(session, active=active):
            if not veto_clear(session):
                return False, "veto"
            r = rows.get(session)
            if r is None:
                return False, "no row"
            try:
                return (all(sig_on(r, a) for a in active), "ok")
            except TypeError:
                return False, "na"

        res = BacktestEngine(cfg, options, und, rates, execution_scenario="base",
                             entry_gate=gate, dividends=div).run()
        t = res.trades_frame()
        row = {"mask": mask, "signals": "+".join(active) or "(veto only)", "n": t.height}
        if t.height:
            t = t.with_columns(
                (pl.col("realized_net") / pl.col("qty")).alias("pnl"),
                pl.col("entry_ts").dt.convert_time_zone("America/New_York")
                .dt.date().alias("entry_date"))
            per = t["pnl"].to_numpy()
            row.update({"mean": float(per.mean()), "win": float((per > 0).mean()),
                        "worst": float(per.min()), "total": float(per.sum()),
                        "credit": float((t["entry_credit"] * 100).mean())})
            if t.height >= 10:
                ci = bootstrap_mean_ci(per, n_boot=1000, block=5, seed=7)
                lo, hi, nep = episode_boot_ci(
                    per, episode_ids(t["entry_date"].to_numpy()), n_boot=1000)
                row.update({"ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                            "ep_ci_low": lo, "ep_ci_high": hi, "n_episodes": nep})
        results.append(row)

    with open(f"results/iwm_prereg/confirm_ablation_{sym.lower()}.jsonl", "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    base = next(r for r in results if r["mask"] == 0)
    print(f"\n{'='*84}\n{sym} — veto-only baseline: n={base['n']} "
          f"mean=${base.get('mean', 0):.2f} total=${base.get('total', 0):.0f}")
    ok = [r for r in results if r["mask"] and r["n"] >= 15]
    top = sorted(ok, key=lambda r: -r["mean"])[:10]
    print(f"top 10 (of {len(ok)} combos with n>=15):")
    print(f"{'signals':44s} {'n':>3s} {'mean$':>7s} {'credit$':>7s} {'epCI':>16s} "
          f"{'win':>5s} {'worst':>6s} {'total':>6s}")
    for r in top:
        ep = f"[{r['ep_ci_low']:6.1f},{r['ep_ci_high']:6.1f}]" if "ep_ci_low" in r else " " * 16
        print(f"{r['signals']:44s} {r['n']:3d} {r['mean']:7.2f} {r['credit']:7.1f} "
              f"{ep:>16s} {r['win']:5.2f} {r['worst']:6.0f} {r['total']:6.0f}")
    from collections import Counter
    c = Counter()
    for r in top:
        c.update(r["signals"].split("+"))
    print("inclusion in top 10: " +
          ", ".join(f"{k} {v*10}%" for k, v in c.most_common()))
