"""EXPLORATORY exhaustive IWM gate sweep: all 2,047 non-empty AND-combinations
of the 11 candidate signals (QQQ's 10 + rsi70), engine cap-2, base fills.
Overfit by construction (the winner of 2,047 comparisons has no honest CI);
descriptive only — IWM is already burned as confirmation data.

Usage: python scripts/iwm_exhaustive_combos.py --shard i/n
Output: results/iwm_prereg/exhaustive_combos_shard<i>.jsonl (resume-safe)
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _clock
from pathlib import Path

import polars as pl

sys.path.insert(0, "scripts")
from preregistered_iwm_test import (  # noqa: E402
    DIVIDENDS, FEATURES, OPTIONS_GLOB, SECTOR_CORR, UNDERLYING,
)
from phaseA_aggression_frontier import episode_boot_ci, episode_ids  # noqa: E402
from xsp_research.evaluation.robustness import bootstrap_mean_ci  # noqa: E402
from xsp_research.backtest.american import DividendCalendar  # noqa: E402
from xsp_research.backtest.engine import BacktestEngine  # noqa: E402
from xsp_research.config import EXECUTION_SCENARIOS, load_strategy_config  # noqa: E402
from xsp_research.ingestion.file_provider import (  # noqa: E402
    ParquetUnderlyingProvider, SeriesRatesProvider,
)
from full_ablation_sweep import RATES, PreloadedOptionsProvider  # noqa: E402

CONFIG = "configs/strategy_iwm_prereg_h1.yaml"

# keep identical to iwm_regime_scan.SIGNALS (inlined so shards skip its panel build)
SIGNALS = {
    "rsi70":       ("rsi_14", ">", 70.0),
    "trend120":    ("ret_120d", ">", 0.15),
    "trend60":     ("ret_60d", ">", 0.08),
    "ma200_10":    ("dist_ma200", ">", 0.10),
    "ma50_up":     ("ma50_slope_20d", ">", 0.02),
    "beta_hi":     ("iwm_beta_spy_60", ">", 1.30),
    "absorb_fall": ("absorption_chg_20d", "<", 0.0),
    "seccorr_lo":  ("sector_avg_corr_20", "<", 0.45),
    "iv_rich":     ("iv_minus_rv20", ">", 0.0),
    "iwm_decor":   ("corr_spy_iwm_20", "<", 0.80),
    "rvx_lo":      ("vix_level", "<", 25.0),
}
NAMES = list(SIGNALS)
N_MASKS = (1 << len(NAMES)) - 1  # 2047


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    out = Path(f"results/iwm_prereg/exhaustive_combos_shard{si}.jsonl")

    feats = pl.read_parquet(FEATURES).join(
        pl.read_parquet(SECTOR_CORR), on="date", how="left")
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
    options = PreloadedOptionsProvider(OPTIONS_GLOB, "IWM", base.backtest.mark_time_et)
    und = ParquetUnderlyingProvider(UNDERLYING, "IWM")
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
        for mask in range(1, N_MASKS + 1):
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
                   "signals": [NAMES[i] for i in range(len(NAMES)) if mask >> i & 1],
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
