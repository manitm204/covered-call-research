"""EXPLORATORY QQQ structure ablation under the frozen regime gate.

QQQ is burned as a test set (the pre-registered test was run and failed);
this sweep is exploration for possible NEW hypotheses only. Grid: delta x
width x DTE, weekly entries, hold to expiry, cap-2, base fills.
Output: results/qqq_prereg/structure_ablation.jsonl
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
from xsp_research.config import EXECUTION_SCENARIOS, load_strategy_config
from xsp_research.evaluation.robustness import bootstrap_mean_ci
from xsp_research.ingestion.file_provider import ParquetUnderlyingProvider, SeriesRatesProvider

sys.path.insert(0, "scripts")
from full_ablation_sweep import RATES, PreloadedOptionsProvider  # noqa: E402
from phaseA_aggression_frontier import episode_boot_ci, episode_ids  # noqa: E402
from preregistered_qqq_test import (  # noqa: E402
    CONFIG, DIVIDENDS, FEATURES, OPTIONS_GLOB, SECTOR_CORR, UNDERLYING, make_gate,
)

OUT = Path("results/qqq_prereg/structure_ablation.jsonl")
DELTAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
WIDTHS = [5.0, 10.0, 15.0]
DTES = [(7, 5, 10), (14, 10, 18), (21, 17, 25), (30, 25, 35), (45, 38, 52)]


def main() -> int:
    feats = pl.read_parquet(FEATURES).join(
        pl.read_parquet(SECTOR_CORR).select(["date", "sector_avg_corr_20"]),
        on="date", how="left",
    )
    gate = make_gate(feats)
    base = load_strategy_config(CONFIG)
    options = PreloadedOptionsProvider(OPTIONS_GLOB, "QQQ", base.backtest.mark_time_et)
    und = ParquetUnderlyingProvider(UNDERLYING, "QQQ")
    rates = SeriesRatesProvider(RATES)
    div = DividendCalendar.from_parquet(DIVIDENDS)

    done: set[str] = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            done.add(json.loads(line)["config_id"])

    with OUT.open("a") as fh:
        for delta in DELTAS:
            for width in WIDTHS:
                for tgt, lo, hi in DTES:
                    cid = f"d{delta:.2f}_w{width:g}_dte{tgt}"
                    if cid in done:
                        continue
                    cfg = base.model_copy(update={
                        "name": f"qqq_ablate_{cid}",
                        "execution": EXECUTION_SCENARIOS["base"],
                        "exits": [],
                        "selection": base.selection.model_copy(update={
                            "short_delta_target": delta, "fixed_width": width,
                            "target_dte": tgt, "min_dte": lo, "max_dte": hi,
                        }),
                        "sizing": base.sizing.model_copy(update={
                            "contracts_per_entry": 1, "max_open_positions": 2,
                        }),
                    })
                    t0 = _clock.time()
                    try:
                        res = BacktestEngine(
                            cfg, options, und, rates, execution_scenario="base",
                            entry_gate=gate, dividends=div,
                        ).run()
                        t = res.trades_frame()
                        row = {"config_id": cid, "short_delta": delta,
                               "width": width, "target_dte": tgt, "n_trades": t.height}
                        if t.height:
                            t = t.with_columns(
                                (pl.col("realized_net") / pl.col("qty")).alias("pnl"),
                                (pl.col("entry_credit") / pl.col("qty")).alias("credit"),
                                (pl.col("max_loss_dollars") / pl.col("qty")).alias("risk"),
                                pl.col("entry_ts").dt.convert_time_zone("America/New_York")
                                .dt.date().alias("entry_date"),
                            )
                            per = t["pnl"].to_numpy()
                            block = max(1, math.ceil(
                                float(t["days_in_trade"].mean() or 0) / 7))
                            ci = bootstrap_mean_ci(per, n_boot=2000, block=block, seed=7)
                            eids = episode_ids(t["entry_date"].to_numpy())
                            elo, ehi, n_ep = episode_boot_ci(per, eids)
                            k = max(1, math.ceil(0.05 * len(per)))
                            row.update({
                                "mean_pnl": float(per.mean()),
                                "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                                "ep_ci_low": elo, "ep_ci_high": ehi,
                                "n_episodes": n_ep,
                                "win_rate": float((per > 0).mean()),
                                "worst": float(per.min()),
                                "total": float(per.sum()),
                                "mean_credit": float(t["credit"].mean()),
                                "mean_risk": float(t["risk"].mean()),
                                "ret_on_risk_pct": float(
                                    100 * per.mean() / t["risk"].mean()),
                                "cvar5": float(np.sort(per)[:k].mean()),
                                "mean_width": float(t["width"].mean()),
                                "mean_delta": float(
                                    t["short_delta_at_entry"].abs().mean()),
                            })
                    except Exception as exc:
                        row = {"config_id": cid, "error": f"{type(exc).__name__}: {exc}"}
                    row["elapsed_s"] = round(_clock.time() - t0, 2)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(cid, row.get("mean_pnl"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
