"""Command-line interface.

Commands:
  xsp run-backtest -c configs/strategy_baseline.yaml --synthetic [--scenario base]
  xsp validate-data --options data/normalized/options [--root XSP]
  xsp info
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from xsp_research.config import EXECUTION_SCENARIOS, load_strategy_config

SYNTHETIC_BANNER = (
    "=" * 78 + "\nWARNING: running on SYNTHETIC data. Output validates software behavior only\n"
    "and is NOT evidence about real-world strategy performance.\n" + "=" * 78
)


def _cmd_run_backtest(args: argparse.Namespace) -> int:
    from xsp_research.backtest.engine import BacktestEngine
    from xsp_research.evaluation.metrics import summarize

    cfg = load_strategy_config(args.config)
    if args.scenario:
        cfg = cfg.model_copy(update={"execution": EXECUTION_SCENARIOS[args.scenario]})

    if args.synthetic:
        from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket

        print(SYNTHETIC_BANNER, file=sys.stderr)
        market = SyntheticMarket(
            SyntheticConfig(start=cfg.backtest.start, end=cfg.backtest.end, seed=args.seed)
        )
        options = underlying = rates = market
    else:
        if not (args.options_data and args.underlying_data and args.rates_data):
            print(
                "error: real-data runs need --options-data, --underlying-data and "
                "--rates-data (or use --synthetic for software validation)",
                file=sys.stderr,
            )
            return 2
        from xsp_research.ingestion.file_provider import (
            ParquetOptionsProvider,
            ParquetUnderlyingProvider,
            SeriesRatesProvider,
        )

        options = ParquetOptionsProvider(args.options_data, cfg.selection.root)
        underlying = ParquetUnderlyingProvider(args.underlying_data, cfg.selection.root)
        rates = SeriesRatesProvider(args.rates_data)

    engine = BacktestEngine(
        cfg, options, underlying, rates, execution_scenario=args.scenario or "config"
    )
    result = engine.run()
    summary = summarize(result)
    print(json.dumps(summary, indent=2, default=str))

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        result.equity_curve.write_parquet(out / "equity_curve.parquet")
        tf = result.trades_frame()
        if not tf.is_empty():
            tf.write_parquet(out / "trades.parquet")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        (out / "config_snapshot.json").write_text(json.dumps(result.config_snapshot, indent=2))
        print(f"artifacts written to {out}", file=sys.stderr)
    return 0


def _cmd_validate_data(args: argparse.Namespace) -> int:
    import polars as pl

    from xsp_research.ingestion.base import ChainSchemaError, validate_chain_frame

    df = pl.read_parquet(args.options)
    try:
        validate_chain_frame(df)
    except ChainSchemaError as exc:
        print(f"INVALID: {exc}")
        return 1
    crossed = df.filter(pl.col("bid") > pl.col("ask")).height
    zero_bid = df.filter(pl.col("bid") == 0).height
    print(
        json.dumps(
            {
                "rows": df.height,
                "date_range": [str(df["ts"].min()), str(df["ts"].max())],
                "crossed_markets": crossed,
                "zero_bids": zero_bid,
                "status": "schema ok (full validation suite arrives in Phase 2)",
            },
            indent=2,
        )
    )
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    print("xsp-research: XSP bear call credit spread research framework (Phase 1)")
    print("See docs/PLAN.md for architecture, assumptions, and P&L definitions.")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="xsp")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run-backtest", help="run a configured backtest")
    p_run.add_argument("-c", "--config", required=True)
    p_run.add_argument("--scenario", choices=sorted(EXECUTION_SCENARIOS), default=None)
    p_run.add_argument("--synthetic", action="store_true", help="use synthetic TEST data")
    p_run.add_argument("--seed", type=int, default=7)
    p_run.add_argument("--options-data", help="path to canonical options parquet dataset")
    p_run.add_argument("--underlying-data", help="path to (date, close) file")
    p_run.add_argument("--rates-data", help="path to (date, rate) file")
    p_run.add_argument("-o", "--output", help="directory for result artifacts")
    p_run.set_defaults(func=_cmd_run_backtest)

    p_val = sub.add_parser("validate-data", help="validate an options parquet dataset")
    p_val.add_argument("--options", required=True)
    p_val.set_defaults(func=_cmd_validate_data)

    p_info = sub.add_parser("info")
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
