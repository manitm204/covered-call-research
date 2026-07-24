"""Regime-conditioned structure ablation (exploratory, NOT pre-registered).

For each of 4 regime entry gates (mined from the 417-trade dataset — results
are hypotheses), sweep spread structure: delta x width x DTE x exit style,
weekly entries, base scenario. Answers "which structure works in which regime".
"""

from __future__ import annotations

import json
import math
import sys
import time as _clock
from pathlib import Path

import numpy as np
import polars as pl

from xsp_research.backtest.american import DividendCalendar
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.config import EXECUTION_SCENARIOS, ExitRuleConfig, load_strategy_config
from xsp_research.evaluation.robustness import bootstrap_mean_ci
from xsp_research.experiments.filters import EntryFilterConfig, FilterClause, make_entry_gate
from xsp_research.ingestion.file_provider import ParquetUnderlyingProvider, SeriesRatesProvider

sys.path.insert(0, "scripts")
from full_ablation_sweep import (  # noqa: E402
    BASE_CONFIG, DIVIDENDS, OPTIONS_GLOB, RATES, UNDERLYING, PreloadedOptionsProvider,
)

EXP_FEATURES = (
    "reports/experiments/spy_regime_rsi70_ivrich-20260723-204901-b6454793/features.parquet"
)
REGIME_V2 = "results/ablation_full/regime_features_v2.parquet"
OUT = Path("results/ablation_full/regime_structure.jsonl")

GATES = {
    "rsi70_ivrich": [("rsi_14", ">", 70.0), ("iv_minus_rv20", ">", 0.0)],
    "absorb_falling_vix25": [("absorption_chg_20d", "<", 0.0), ("vix_level", "<", 25.0)],
    "sectorcorr45_vix25": [("sector_avg_corr_20", "<", 0.45), ("vix_level", "<", 25.0)],
    "sectorcorr45_breadth_up": [
        ("sector_avg_corr_20", "<", 0.45), ("rspspy_ema5_20", ">", 0.0),
    ],
    "unfiltered": [],
}
DELTAS = [0.10, 0.15, 0.20]
WIDTHS = [2.0, 5.0, 10.0]
DTES = [(21, 17, 25), (30, 25, 35), (45, 38, 52)]
EXITS = {
    "hold": [],
    "pt50_sl2": [
        ExitRuleConfig(kind="profit_target", profit_target_pct=0.50),
        ExitRuleConfig(kind="stop_loss", stop_loss_multiple=2.0),
    ],
}


def main() -> int:
    feats = pl.read_parquet(EXP_FEATURES).join(
        pl.read_parquet(REGIME_V2), on="date", how="left"
    )
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
        for gate_name, clauses in GATES.items():
            gate = (
                make_entry_gate(
                    feats,
                    EntryFilterConfig(
                        clauses=[FilterClause(feature=f, op=o, value=v) for f, o, v in clauses]
                    ),
                )
                if clauses
                else None
            )
            for delta in DELTAS:
                for width in WIDTHS:
                    for tgt, lo, hi in DTES:
                        for exit_name, exits in EXITS.items():
                            cid = f"{gate_name}|d{delta:.2f}_w{width:g}_dte{tgt}_{exit_name}"
                            if cid in done:
                                continue
                            cfg = base.model_copy(update={
                                "name": f"regime_{gate_name}",
                                "execution": EXECUTION_SCENARIOS["base"],
                                "exits": exits,
                                "selection": base.selection.model_copy(update={
                                    "short_delta_target": delta, "fixed_width": width,
                                    "target_dte": tgt, "min_dte": lo, "max_dte": hi,
                                }),
                                "sizing": base.sizing.model_copy(update={
                                    "contracts_per_entry": 1, "max_open_positions": 12,
                                }),
                            })
                            t0 = _clock.time()
                            try:
                                res = BacktestEngine(
                                    cfg, options, und, rates, execution_scenario="base",
                                    entry_gate=gate, dividends=div,
                                ).run()
                                t = res.trades_frame()
                                row = {
                                    "config_id": cid, "gate": gate_name, "short_delta": delta,
                                    "width": width, "target_dte": tgt, "exit_style": exit_name,
                                    "n_trades": t.height,
                                }
                                if t.height:
                                    per = (t["realized_net"] / t["qty"]).to_numpy()
                                    block = max(
                                        1, math.ceil(float(t["days_in_trade"].mean() or 0) / 7)
                                    )
                                    ci = bootstrap_mean_ci(per, n_boot=2000, block=block, seed=7)
                                    row.update({
                                        "mean_pnl": float(np.mean(per)),
                                        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                                        "win_rate": float((per > 0).mean()),
                                        "worst": float(per.min()),
                                        "total": float(per.sum()),
                                    })
                            except Exception as exc:
                                row = {"config_id": cid, "gate": gate_name,
                                       "error": f"{type(exc).__name__}: {exc}"}
                            row["elapsed_s"] = round(_clock.time() - t0, 2)
                            fh.write(json.dumps(row) + "\n")
                            fh.flush()
                            print(cid, row.get("mean_pnl"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
