"""Phase A: aggression frontier on gated entry dates (exploratory, NOT pre-registered).

Entry gate is FROZEN at the final-hypothesis rule: RSI(14)>70 mandatory, plus at
least 2 of {iv_minus_rv20>0, sector_avg_corr_20<0.45, absorption_chg_20d<0}.
Sizing frozen at cap-2 (max 2 open, 1 contract each). DTE frozen at 30, weekly
entries, hold to expiry. Sweep is ONLY delta x width x fill scenario, scored on
risk-adjusted metrics: return on collateral, CVaR, worst trade, and episode-level
bootstrap CI (trades cluster into ~25 regime episodes; per-trade CIs flatter us).
"""

from __future__ import annotations

import json
import math
import sys
import time as _clock
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from xsp_research.backtest.american import DividendCalendar
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.config import EXECUTION_SCENARIOS, load_strategy_config
from xsp_research.evaluation.robustness import bootstrap_mean_ci
from xsp_research.ingestion.file_provider import ParquetUnderlyingProvider, SeriesRatesProvider

sys.path.insert(0, "scripts")
from full_ablation_sweep import (  # noqa: E402
    BASE_CONFIG, DIVIDENDS, OPTIONS_GLOB, RATES, UNDERLYING, PreloadedOptionsProvider,
)

EXP_FEATURES = (
    "reports/experiments/spy_regime_rsi70_ivrich-20260723-204901-b6454793/features.parquet"
)
REGIME_V2 = "results/ablation_full/regime_features_v2.parquet"
OUT = Path("results/ablation_full/phaseA_frontier.jsonl")

DELTAS = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]
WIDTHS = [2.0, 3.0, 5.0, 8.0, 10.0, 15.0]
SCENARIOS = ["base", "conservative"]
EPISODE_GAP_DAYS = 30  # entries more than this far apart start a new episode


def make_final_gate(feats: pl.DataFrame):
    rows: dict[date, dict] = {r["date"]: r for r in feats.iter_rows(named=True)}

    def gate(session: date) -> tuple[bool, str]:
        r = rows.get(session)
        if r is None:
            return False, "no feature row"
        needed = ("rsi_14", "iv_minus_rv20", "sector_avg_corr_20", "absorption_chg_20d")
        if any(r.get(k) is None for k in needed):
            return False, "feature unavailable"
        if not r["rsi_14"] > 70:
            return False, f"rsi_14={r['rsi_14']:.1f} <= 70"
        votes = sum([
            r["iv_minus_rv20"] > 0,
            r["sector_avg_corr_20"] < 0.45,
            r["absorption_chg_20d"] < 0,
        ])
        if votes < 2:
            return False, f"only {votes}/3 confirmations"
        return True, f"rsi>70 and {votes}/3 confirmations"

    return gate


def episode_ids(entry_dates: np.ndarray) -> np.ndarray:
    """Assign consecutive entries within EPISODE_GAP_DAYS of the previous one
    to the same episode."""
    order = np.argsort(entry_dates)
    eid = np.zeros(len(entry_dates), dtype=int)
    cur = 0
    prev = None
    for idx in order:
        d = entry_dates[idx]
        if prev is not None and (d - prev).astype("timedelta64[D]").astype(int) > EPISODE_GAP_DAYS:
            cur += 1
        eid[idx] = cur
        prev = d
    return eid


def episode_boot_ci(per: np.ndarray, eids: np.ndarray, n_boot: int = 2000, seed: int = 7):
    """Bootstrap CI of the per-trade mean, resampling whole episodes."""
    rng = np.random.default_rng(seed)
    groups = [per[eids == e] for e in np.unique(eids)]
    k = len(groups)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, size=k)
        sample = np.concatenate([groups[i] for i in pick])
        means[b] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), k


def main() -> int:
    feats = pl.read_parquet(EXP_FEATURES).join(
        pl.read_parquet(REGIME_V2), on="date", how="left"
    )
    gate = make_final_gate(feats)
    base = load_strategy_config(BASE_CONFIG)
    options = PreloadedOptionsProvider(OPTIONS_GLOB, "SPY", base.backtest.mark_time_et)
    und = ParquetUnderlyingProvider(UNDERLYING, "SPY")
    rates = SeriesRatesProvider(RATES)
    div = DividendCalendar.from_parquet(DIVIDENDS)

    done: set[str] = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            done.add(json.loads(line)["config_id"])

    with OUT.open("a") as fh:
        for scen in SCENARIOS:
            for delta in DELTAS:
                for width in WIDTHS:
                    cid = f"{scen}|d{delta:.2f}_w{width:g}"
                    if cid in done:
                        continue
                    cfg = base.model_copy(update={
                        "name": f"phaseA_{scen}",
                        "execution": EXECUTION_SCENARIOS[scen],
                        "exits": [],
                        "selection": base.selection.model_copy(update={
                            "short_delta_target": delta, "fixed_width": width,
                            "target_dte": 30, "min_dte": 25, "max_dte": 35,
                        }),
                        "sizing": base.sizing.model_copy(update={
                            "contracts_per_entry": 1, "max_open_positions": 2,
                        }),
                    })
                    t0 = _clock.time()
                    try:
                        res = BacktestEngine(
                            cfg, options, und, rates, execution_scenario=scen,
                            entry_gate=gate, dividends=div,
                        ).run()
                        t = res.trades_frame()
                        row = {"config_id": cid, "scenario": scen,
                               "short_delta": delta, "width": width, "n_trades": t.height}
                        if t.height:
                            t = t.with_columns(
                                (pl.col("realized_net") / pl.col("qty")).alias("pnl"),
                                (pl.col("entry_credit") / pl.col("qty")).alias("credit"),
                                (pl.col("max_loss_dollars") / pl.col("qty")).alias("risk"),
                            )
                            per = t["pnl"].to_numpy()
                            credit = t["credit"].to_numpy()
                            risk = t["risk"].to_numpy()
                            entry_d = (
                                t["entry_ts"].dt.convert_time_zone("America/New_York")
                                .dt.date().to_numpy()
                            )
                            eids = episode_ids(entry_d)
                            elo, ehi, n_ep = episode_boot_ci(per, eids)
                            block = max(1, math.ceil(float(t["days_in_trade"].mean() or 0) / 7))
                            ci = bootstrap_mean_ci(per, n_boot=2000, block=block, seed=7)
                            k_tail = max(1, math.ceil(0.05 * len(per)))
                            cvar5 = float(np.sort(per)[:k_tail].mean())
                            row.update({
                                "mean_pnl": float(per.mean()),
                                "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                                "ep_ci_low": elo, "ep_ci_high": ehi, "n_episodes": n_ep,
                                "win_rate": float((per > 0).mean()),
                                "worst": float(per.min()),
                                "total": float(per.sum()),
                                "mean_credit": float(credit.mean()),
                                "mean_risk": float(risk.mean()),
                                "ret_on_risk_pct": float(100 * per.mean() / risk.mean()),
                                "cvar5": cvar5,
                                "mean_short_delta": float(
                                    t["short_delta_at_entry"].abs().mean()
                                ),
                            })
                    except Exception as exc:
                        row = {"config_id": cid, "scenario": scen,
                               "short_delta": delta, "width": width,
                               "error": f"{type(exc).__name__}: {exc}"}
                    row["elapsed_s"] = round(_clock.time() - t0, 2)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(cid, row.get("mean_pnl"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
