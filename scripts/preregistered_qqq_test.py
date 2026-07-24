"""PRE-REGISTERED QQQ out-of-sample test. Protocol: docs/PREREGISTRATION_QQQ.md.

Frozen before any QQQ options data was examined. SINGLE USE: the script
refuses to run a second time (delete the marker only if the first run crashed
before producing results — and say so in the report).

Rule under test (identical to the SPY final hypothesis, QQQ analogs):
  gate: RSI14(NDX) > 70 AND >= 2 of { VXN/100 - RV20(NDX) > 0,
        sector_avg_corr_20 < 0.45, absorption_chg_20d < 0 }
  structure: 0.15 delta / $8 wide / 30 DTE / weekly / hold to expiry / cap 2

Endpoints (declared in advance):
  PRIMARY    base fills: per-trade moving-block bootstrap 95% CI entirely > $0
  SECONDARY  conservative-fill mean > 0; episode-level 95% CI entirely > $0;
             win rate >= 0.75
  VERDICT    GO      = primary met AND conservative mean > 0
             WEAK    = base mean > 0 but primary CI spans zero (inconclusive;
                       NOT a license to tune)
             NO-GO   = base mean <= 0
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

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

CONFIG = "configs/strategy_qqq_prereg.yaml"
OPTIONS_GLOB = "data/normalized/options/qqq/chain_*.parquet"
UNDERLYING = "data/normalized/options/qqq/underlying_eod.parquet"
DIVIDENDS = "data/normalized/aux/QQQ_DIVIDENDS.parquet"
FEATURES = "results/qqq_prereg/features_qqq.parquet"
SECTOR_CORR = "results/ablation_full/regime_features_v2.parquet"
MARKER = Path("results/qqq_prereg/TEST_EXECUTED.marker")
OUT = Path("results/qqq_prereg/verdict.json")


def make_gate(feats: pl.DataFrame):
    rows = {r["date"]: r for r in feats.iter_rows(named=True)}

    def gate(session):
        r = rows.get(session)
        if r is None:
            return False, "no feature row"
        needed = ("rsi_14", "iv_minus_rv20", "sector_avg_corr_20", "absorption_chg_20d")
        if any(r.get(k) is None for k in needed):
            return False, "feature unavailable"
        if not r["rsi_14"] > 70:
            return False, "rsi off"
        votes = sum([
            r["iv_minus_rv20"] > 0,
            r["sector_avg_corr_20"] < 0.45,
            r["absorption_chg_20d"] < 0,
        ])
        return (votes >= 2, f"{votes}/3 confirmations")

    return gate


def main() -> int:
    if MARKER.exists():
        print(f"REFUSING TO RUN: {MARKER} exists — this test is single-use.\n"
              f"Executed at: {MARKER.read_text().strip()}")
        return 1
    from datetime import datetime, timezone
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(datetime.now(timezone.utc).isoformat() + "\n")

    feats = pl.read_parquet(FEATURES).join(
        pl.read_parquet(SECTOR_CORR).select(["date", "sector_avg_corr_20"]),
        on="date", how="left",
    )
    gate = make_gate(feats)
    cfg0 = load_strategy_config(CONFIG)
    options = PreloadedOptionsProvider(OPTIONS_GLOB, "QQQ", cfg0.backtest.mark_time_et)
    und = ParquetUnderlyingProvider(UNDERLYING, "QQQ")
    rates = SeriesRatesProvider(RATES)
    div = DividendCalendar.from_parquet(DIVIDENDS)

    out: dict[str, dict] = {}
    for scen in ("base", "conservative"):
        cfg = cfg0.model_copy(update={"execution": EXECUTION_SCENARIOS[scen]})
        res = BacktestEngine(
            cfg, options, und, rates, execution_scenario=scen,
            entry_gate=gate, dividends=div,
        ).run()
        t = res.trades_frame()
        if not t.height:
            out[scen] = {"n": 0}
            continue
        t = t.with_columns(
            (pl.col("realized_net") / pl.col("qty")).alias("pnl"),
            pl.col("entry_ts").dt.convert_time_zone("America/New_York")
            .dt.date().alias("entry_date"),
        )
        per = t["pnl"].to_numpy()
        block = max(1, math.ceil(float(t["days_in_trade"].mean() or 0) / 7))
        ci = bootstrap_mean_ci(per, n_boot=2000, block=block, seed=7)
        eids = episode_ids(t["entry_date"].to_numpy())
        elo, ehi, n_ep = episode_boot_ci(per, eids)
        yr = t["entry_date"].dt.year().to_numpy()
        yearly = {int(y): round(float(per[yr == y].mean()), 2) for y in sorted(set(yr))}
        out[scen] = {
            "n": t.height, "mean": float(per.mean()),
            "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "ep_ci_low": elo, "ep_ci_high": ehi, "n_episodes": n_ep,
            "win": float((per > 0).mean()), "worst": float(per.min()),
            "total": float(per.sum()), "yearly_mean": yearly,
        }

    b = out.get("base", {})
    c = out.get("conservative", {})
    primary = bool(b.get("n") and b["ci_low"] > 0)
    if b.get("n") and b["mean"] > 0:
        verdict = "GO" if primary and c.get("n") and c["mean"] > 0 else "WEAK"
    else:
        verdict = "NO-GO"
    out["verdict"] = {
        "primary_ci_above_zero": primary,
        "secondary_conservative_mean_pos": bool(c.get("n") and c["mean"] > 0),
        "secondary_episode_ci_above_zero": bool(b.get("n") and b["ep_ci_low"] > 0),
        "secondary_win_rate_ge_75": bool(b.get("n") and b["win"] >= 0.75),
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nVERDICT: {verdict}  (wrote {OUT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
