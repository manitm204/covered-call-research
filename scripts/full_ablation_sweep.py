"""Full exploratory ablation sweep over spread-structure parameters.

2,000 configurations: 8 short-delta targets x 5 widths x 5 DTE targets x
2 entry frequencies x 5 exit styles, all run with the BASE execution scenario
on real SPY data. EXPLORATORY (not pre-registered): with 2,000 configs some
cells will look positive by chance; results carry bootstrap CIs and the report
must treat any "winner" as a hypothesis for fresh pre-registration, not a
conclusion.

Speed: the engine's per-session chain snapshot is config-independent, so all
snapshots are precomputed once (PreloadedOptionsProvider) and shared by every
config. Supports sharding (--shard i/n) so several processes can split the
grid, and resumes by skipping config ids already present in the output JSONL.

Run:  python scripts/full_ablation_sweep.py --shard 0/4
Validate fidelity vs the committed baseline first:
      python scripts/full_ablation_sweep.py --validate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time as _clock
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from xsp_research.backtest.american import DividendCalendar
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.config import (
    EXECUTION_SCENARIOS,
    EntryFrequency,
    ExitRuleConfig,
    load_strategy_config,
)
from xsp_research.evaluation.metrics import equity_metrics, trade_metrics
from xsp_research.evaluation.robustness import bootstrap_mean_ci
from xsp_research.ingestion.base import CHAIN_SCHEMA, validate_chain_frame
from xsp_research.ingestion.file_provider import (
    ParquetUnderlyingProvider,
    SeriesRatesProvider,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

OPTIONS_GLOB = "data/normalized/options/spy/chain_*.parquet"
UNDERLYING = "data/normalized/options/spy/underlying_eod.parquet"
RATES = "data/normalized/rates/tbill_4w.parquet"
DIVIDENDS = "data/normalized/aux/SPY_DIVIDENDS.parquet"
BASE_CONFIG = "configs/strategy_spy_hold_to_expiry.yaml"
OUT_DIR = Path("results/ablation_full")

DELTAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
WIDTHS = [1.0, 2.0, 3.0, 5.0, 10.0]
DTES = [(7, 5, 10), (14, 10, 18), (21, 17, 25), (30, 25, 35), (45, 38, 52)]
FREQS = [EntryFrequency.WEEKLY, EntryFrequency.MONTHLY]


def _exit_sets() -> dict[str, list[ExitRuleConfig]]:
    pt50 = ExitRuleConfig(kind="profit_target", profit_target_pct=0.50)
    pt25 = ExitRuleConfig(kind="profit_target", profit_target_pct=0.25)
    sl2 = ExitRuleConfig(kind="stop_loss", stop_loss_multiple=2.0)
    return {
        "hold": [],
        "pt50": [pt50],
        "pt50_sl2": [pt50, sl2],
        "pt25_sl2": [pt25, sl2],
        "sl2": [sl2],
    }


class PreloadedOptionsProvider:
    """Same chain() semantics as ParquetOptionsProvider, but every session
    snapshot is computed once up front and served from memory."""

    def __init__(self, dataset_glob: str, root: str, mark_time_et: time) -> None:
        self._root = root
        self._path = dataset_glob
        self._cache: dict[date, pl.DataFrame] = {}
        files = sorted(Path().glob(dataset_glob))
        if not files:
            raise FileNotFoundError(f"no files match {dataset_glob}")
        empty: pl.DataFrame | None = None
        t0 = _clock.time()
        for f in files:
            df = pl.read_parquet(f).filter(pl.col("root") == root)
            if df.is_empty():
                continue
            for d in df.select(pl.col("ts").dt.date().unique().sort()).to_series():
                snap_ts = datetime(
                    d.year, d.month, d.day, mark_time_et.hour, mark_time_et.minute, tzinfo=ET
                ).astimezone(UTC)
                snap = (
                    df.filter((pl.col("ts").dt.date() == d) & (pl.col("ts") <= snap_ts))
                    .sort("ts")
                    .group_by(
                        ["root", "expiration", "strike", "option_type"], maintain_order=True
                    )
                    .last()
                    .select(list(CHAIN_SCHEMA))
                )
                validate_chain_frame(snap)
                self._cache[d] = snap
                if empty is None:
                    empty = snap.clear()
        self._empty = empty if empty is not None else pl.DataFrame()
        print(
            f"preloaded {len(self._cache)} session snapshots in {_clock.time() - t0:.0f}s",
            file=sys.stderr,
            flush=True,
        )

    @property
    def data_source(self) -> str:
        return f"user_parquet:{self._root}:{self._path}"

    def chain(self, as_of: datetime) -> pl.DataFrame:
        return self._cache.get(as_of.astimezone(UTC).date(), self._empty)

    def expirations(self, as_of: date) -> list[date]:
        snap = self._cache.get(as_of)
        if snap is None or snap.is_empty():
            return []
        return sorted(snap.filter(pl.col("expiration") >= as_of)["expiration"].unique())


def build_configs(base_cfg):
    exit_sets = _exit_sets()
    for delta in DELTAS:
        for width in WIDTHS:
            for target, lo, hi in DTES:
                for freq in FREQS:
                    for exit_name, exits in exit_sets.items():
                        cid = (
                            f"d{delta:.2f}_w{width:g}_dte{target}_"
                            f"{freq.value}_{exit_name}"
                        )
                        cfg = base_cfg.model_copy(
                            update={
                                "name": f"sweep_{cid}",
                                "selection": base_cfg.selection.model_copy(
                                    update={
                                        "short_delta_target": delta,
                                        "fixed_width": width,
                                        "target_dte": target,
                                        "min_dte": lo,
                                        "max_dte": hi,
                                    }
                                ),
                                "exits": exits,
                                "execution": EXECUTION_SCENARIOS["base"],
                                "backtest": base_cfg.backtest.model_copy(
                                    update={"entry_frequency": freq}
                                ),
                                "sizing": base_cfg.sizing.model_copy(
                                    update={"contracts_per_entry": 1, "max_open_positions": 12}
                                ),
                            }
                        )
                        params = {
                            "config_id": cid,
                            "short_delta": delta,
                            "width": width,
                            "target_dte": target,
                            "entry_frequency": freq.value,
                            "exit_style": exit_name,
                        }
                        yield params, cfg


def run_one(cfg, params, options, underlying, rates, dividends) -> dict:
    engine = BacktestEngine(
        cfg, options, underlying, rates, execution_scenario="base", dividends=dividends
    )
    result = engine.run()
    trades = result.trades_frame()
    tm = trade_metrics(trades)
    em = equity_metrics(result.equity_curve)
    row = {
        **params,
        "data_source": result.data_source,
        "n_trades": tm.get("n_trades", 0),
        "win_rate": tm.get("win_rate"),
        "expectancy_net_per_spread": tm.get("expectancy_net_per_spread"),
        "total_net_pnl": tm.get("total_net_pnl"),
        "avg_return_on_max_risk": tm.get("avg_return_on_max_risk"),
        "worst_trade_net": tm.get("worst_trade_net"),
        "total_fees": tm.get("total_fees"),
        "max_drawdown": em.get("max_drawdown"),
        "final_equity": em.get("final_equity"),
        "sharpe": em.get("sharpe"),
        "n_entry_attempts": len(result.entry_attempts),
        "n_filled": sum(1 for a in result.entry_attempts if a.filled),
    }
    if not trades.is_empty():
        per_spread = (trades["realized_net"] / trades["qty"]).to_numpy()
        interval = 7 if params["entry_frequency"] == "weekly" else 30
        mean_days = float(trades["days_in_trade"].mean() or 0.0)
        block = max(1, math.ceil(mean_days / interval))
        ci = bootstrap_mean_ci(per_spread, n_boot=2000, block=block, seed=7)
        row["expectancy_ci_low"] = ci.get("ci_low")
        row["expectancy_ci_high"] = ci.get("ci_high")
        row["bootstrap_block"] = block
        reasons = trades["exit_reason"].value_counts()
        row["exit_reasons"] = {
            r["exit_reason"]: r["count"] for r in reasons.iter_rows(named=True)
        }
    return row


def validate_against_baseline(options, underlying, rates, dividends) -> int:
    """The preloaded provider must reproduce the committed baseline exactly."""
    cfg = load_strategy_config("configs/strategy_spy_baseline.yaml")
    cfg = cfg.model_copy(update={"execution": EXECUTION_SCENARIOS["base"]})
    engine = BacktestEngine(
        cfg, options, underlying, rates, execution_scenario="base", dividends=dividends
    )
    result = engine.run()
    tm = trade_metrics(result.trades_frame())
    committed = json.loads(Path("results/spy_baseline/base/summary.json").read_text())
    ok = True
    for key in ("n_trades", "expectancy_net_per_spread", "total_net_pnl", "win_rate"):
        got, want = tm.get(key), committed["trades"].get(key)
        match = got == want or (
            isinstance(got, float) and isinstance(want, float) and abs(got - want) < 1e-9
        )
        print(f"{key}: got {got}  committed {want}  {'OK' if match else 'MISMATCH'}")
        ok &= match
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n: run configs where index %% n == i")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="stop after N configs (timing)")
    args = ap.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    base_cfg = load_strategy_config(BASE_CONFIG)
    options = PreloadedOptionsProvider(OPTIONS_GLOB, "SPY", base_cfg.backtest.mark_time_et)
    underlying = ParquetUnderlyingProvider(UNDERLYING, "SPY")
    rates = SeriesRatesProvider(RATES)
    dividends = DividendCalendar.from_parquet(DIVIDENDS)

    if args.validate:
        return validate_against_baseline(options, underlying, rates, dividends)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"sweep_results_shard{shard_i}.jsonl"
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["config_id"])
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"resuming: {len(done)} configs already done", file=sys.stderr, flush=True)

    n_run = 0
    with out_path.open("a") as fh:
        for idx, (params, cfg) in enumerate(build_configs(base_cfg)):
            if idx % shard_n != shard_i or params["config_id"] in done:
                continue
            t0 = _clock.time()
            try:
                row = run_one(cfg, params, options, underlying, rates, dividends)
            except Exception as exc:  # record and continue; a bad cell must not kill the sweep
                row = {**params, "error": f"{type(exc).__name__}: {exc}"}
            row["elapsed_s"] = round(_clock.time() - t0, 2)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n_run += 1
            print(
                f"[shard {shard_i}/{shard_n}] {n_run}: {params['config_id']} "
                f"({row['elapsed_s']}s)",
                file=sys.stderr,
                flush=True,
            )
            if args.limit and n_run >= args.limit:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
