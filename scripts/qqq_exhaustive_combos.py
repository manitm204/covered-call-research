"""EXPLORATORY exhaustive QQQ gate sweep: all 1,023 non-empty AND-combinations
of the 10 candidate signals, engine cap-2, base fills. Overfit by construction
(the winner of 1,023 comparisons has no honest CI); descriptive only.

Usage: python scripts/qqq_exhaustive_combos.py --shard i/n
Output: results/qqq_prereg/exhaustive_combos_shard<i>.jsonl (resume-safe)
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _clock
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
from preregistered_qqq_test import FEATURES, SECTOR_CORR  # noqa: E402
from phaseA_aggression_frontier import episode_boot_ci, episode_ids  # noqa: E402
from qqq_signal_ablation import SIGNALS  # noqa: E402
from xsp_research.evaluation.robustness import bootstrap_mean_ci  # noqa: E402
from xsp_research.backtest.american import DividendCalendar  # noqa: E402
from xsp_research.backtest.engine import BacktestEngine  # noqa: E402
from xsp_research.config import EXECUTION_SCENARIOS, load_strategy_config  # noqa: E402
from xsp_research.ingestion.file_provider import (  # noqa: E402
    ParquetUnderlyingProvider, SeriesRatesProvider,
)
from full_ablation_sweep import RATES, PreloadedOptionsProvider  # noqa: E402
from preregistered_qqq_test import CONFIG, DIVIDENDS, OPTIONS_GLOB, UNDERLYING  # noqa: E402

NAMES = list(SIGNALS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    out = Path(f"results/qqq_prereg/exhaustive_combos_shard{si}.jsonl")

    feats = pl.read_parquet(FEATURES).join(
        pl.read_parquet(SECTOR_CORR), on="date", how="left")
    # precompute per-date signal bits
    bits: dict = {}
    for r in feats.iter_rows(named=True):
        b = 0
        ok = True
        for i, name in enumerate(NAMES):
            col, op, thr = SIGNALS[name]
            v = r.get(col)
            if v is None:
                ok = False
                break
            if (v < thr) if op == "<" else (v > thr):
                b |= 1 << i
        if ok:
            bits[r["date"]] = b

    base = load_strategy_config(CONFIG)
    options = PreloadedOptionsProvider(OPTIONS_GLOB, "QQQ", base.backtest.mark_time_et)
    und = ParquetUnderlyingProvider(UNDERLYING, "QQQ")
    rates = SeriesRatesProvider(RATES)
    div = DividendCalendar.from_parquet(DIVIDENDS)
    cfg = base.model_copy(update={
        "execution": EXECUTION_SCENARIOS["base"],
        "sizing": base.sizing.model_copy(update={
            "contracts_per_entry": 1, "max_open_positions": 2}),
    })

    done: set[int] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            done.add(json.loads(line)["mask"])

    with out.open("a") as fh:
        for mask in range(1, 1024):
            if mask % sn != si or mask in done:
                continue

            def gate(s, mask=mask):
                b = bits.get(s)
                if b is None:
                    return False, "na"
                return ((b & mask) == mask, "ok")

            t0 = _clock.time()
            res = BacktestEngine(
                cfg, options, und, rates, execution_scenario="base",
                entry_gate=gate, dividends=div,
            ).run()
            t = res.trades_frame()
            row = {"mask": mask,
                   "signals": [NAMES[i] for i in range(10) if mask >> i & 1],
                   "n": t.height}
            if t.height:
                t = t.with_columns(
                    (pl.col("realized_net") / pl.col("qty")).alias("pnl"),
                    pl.col("entry_ts").dt.convert_time_zone("America/New_York")
                    .dt.date().alias("entry_date"),
                )
                per = t["pnl"].to_numpy()
                row.update({"mean": float(per.mean()),
                            "win": float((per > 0).mean()),
                            "worst": float(per.min()), "total": float(per.sum())})
                if t.height >= 10:
                    ci = bootstrap_mean_ci(per, n_boot=1000, block=5, seed=7)
                    eids = episode_ids(t["entry_date"].to_numpy())
                    lo, hi, nep = episode_boot_ci(per, eids, n_boot=1000)
                    row.update({"ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                                "ep_ci_low": lo, "ep_ci_high": hi, "n_episodes": nep})
            row["elapsed_s"] = round(_clock.time() - t0, 2)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    print(f"shard {si}/{sn} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
