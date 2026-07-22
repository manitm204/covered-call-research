"""Command-line interface.

Commands:
  xsp run-backtest -c configs/strategy_baseline.yaml --synthetic [--scenario base] [--report]
  xsp ingest --vendor cboe_datashop --input raw.csv --output data/normalized/options/x.parquet
  xsp validate-data --options data/normalized/options/x.parquet
  xsp info
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
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

    stock_closes = None
    if args.synthetic:
        from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket

        print(SYNTHETIC_BANNER, file=sys.stderr)
        market = SyntheticMarket(
            SyntheticConfig(start=cfg.backtest.start, end=cfg.backtest.end, seed=args.seed)
        )
        options = underlying = rates = market
        # Overlay proxy for synthetic runs: the synthetic index itself as the
        # "stock portfolio" (labeled synthetic like everything else).
        stock_closes = market.closes(cfg.backtest.start, cfg.backtest.end)
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
        if args.stock_data:
            stock_closes = ParquetUnderlyingProvider(args.stock_data, "STOCK").closes(
                cfg.backtest.start, cfg.backtest.end
            )

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

    if args.report:
        if not args.output:
            print("error: --report requires -o/--output", file=sys.stderr)
            return 2
        from xsp_research.backtest.portfolio_overlay import (
            OverlayConfig,
            build_overlay_frame,
            compare_portfolios,
        )
        from xsp_research.evaluation.reporting import generate_report
        from xsp_research.evaluation.stress_tests import stress_report

        overlay_frame = overlay_comparison = stress = None
        if stock_closes is not None:
            overlay_frame = build_overlay_frame(stock_closes, result, OverlayConfig())
            overlay_comparison = compare_portfolios(overlay_frame, result)
            stress = stress_report(overlay_frame, result.trades_frame())
        path = generate_report(
            result,
            args.output,
            overlay_frame=overlay_frame,
            overlay_comparison=overlay_comparison,
            stress=stress,
        )
        print(f"report written to {path}", file=sys.stderr)
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from xsp_research.ingestion.vendors import PRESETS, VendorMapping, ingest_file

    if args.mapping:
        import yaml

        with open(args.mapping) as fh:
            mapping = VendorMapping.model_validate(yaml.safe_load(fh))
    else:
        mapping = PRESETS[args.vendor]
    manifest = ingest_file(
        args.input,
        args.output,
        mapping,
        spx_proxy=args.spx_proxy,
        manifest_dir=args.manifest_dir,
    )
    print(json.dumps(manifest, indent=2))
    if args.spx_proxy:
        print(
            "NOTE: SPX->XSP proxy transform applied; root labeled *_PROXY. "
            "Execution costs on proxy data are NOT representative of XSP.",
            file=sys.stderr,
        )
    return 0


def _load_bundle(args: argparse.Namespace, start, end):
    """Bundle from --synthetic or from a --bundle-dir of <SYMBOL>.parquet files."""
    if args.synthetic:
        from xsp_research.ingestion.synthetic import (
            SyntheticConfig,
            SyntheticMarket,
            research_bundle,
        )

        print(SYNTHETIC_BANNER, file=sys.stderr)
        market = SyntheticMarket(SyntheticConfig(start=start, end=end, seed=args.seed))
        return market, research_bundle(market, seed=args.seed)

    import polars as pl

    from xsp_research.features.registry import MarketDataBundle

    if not args.bundle_dir:
        print("error: need --synthetic or --bundle-dir", file=sys.stderr)
        raise SystemExit(2)
    series = {}
    for p in sorted(Path(args.bundle_dir).glob("*.parquet")):
        series[p.stem] = pl.read_parquet(p).select(["date", "close"])
    sectors = tuple(s.strip() for s in args.sectors.split(",") if s.strip()) if args.sectors else ()
    return None, MarketDataBundle(series=series, sector_symbols=sectors)


def _cmd_build_features(args: argparse.Namespace) -> int:
    from xsp_research.features import build_features

    _, bundle = _load_bundle(args, args.start, args.end)
    built = build_features(bundle, families=args.families.split(",") if args.families else None)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    built.frame.write_parquet(out)
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(built.manifest(), indent=2))
    print(
        json.dumps(
            {
                "features_built": len(built.used),
                "skipped": built.skipped,
                "rows": built.frame.height,
                "output": str(out),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


def _cmd_run_experiment(args: argparse.Namespace) -> int:
    from xsp_research.experiments.runner import load_experiment_config, run_experiment

    exp = load_experiment_config(args.config)
    if not args.synthetic:
        print(
            "error: real-data experiment runs arrive with the data providers; "
            "use --synthetic for software validation",
            file=sys.stderr,
        )
        return 2
    from xsp_research.config import load_strategy_config

    scfg = load_strategy_config(exp.strategy_config)
    market, bundle = _load_bundle(args, scfg.backtest.start, scfg.backtest.end)
    run = run_experiment(exp, market, market, market, bundle, out_root=args.output_root)
    print(
        json.dumps(
            {
                "experiment_id": run.experiment_id,
                "out_dir": str(run.out_dir),
                "scenarios_completed": sorted(run.summaries),
                "errors": run.errors,
            },
            indent=2,
        )
    )
    return 0 if not run.errors else 1


def _cmd_validate_data(args: argparse.Namespace) -> int:
    import polars as pl

    from xsp_research.ingestion.base import ChainSchemaError, validate_chain_frame
    from xsp_research.ingestion.validation import validate_options_dataset

    df = pl.read_parquet(args.options)
    try:
        validate_chain_frame(df)
    except ChainSchemaError as exc:
        print(json.dumps({"passed": False, "schema_error": str(exc)}, indent=2))
        return 1
    report = validate_options_dataset(df, dataset_name=str(args.options))
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


def _cmd_info(args: argparse.Namespace) -> int:
    print("xsp-research: XSP bear call credit spread research framework (Phase 2)")
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
    p_run.add_argument(
        "--stock-data", help="path to stock-portfolio (date, close) file for overlay"
    )
    p_run.add_argument("-o", "--output", help="directory for result artifacts")
    p_run.add_argument(
        "--report",
        action="store_true",
        help="generate markdown report (+overlay/stress if possible)",
    )
    p_run.set_defaults(func=_cmd_run_backtest)

    p_ing = sub.add_parser("ingest", help="normalize a vendor file to canonical parquet + manifest")
    p_ing.add_argument("--vendor", choices=["generic", "cboe_datashop"], default="generic")
    p_ing.add_argument(
        "--mapping", help="YAML file with a custom VendorMapping (overrides --vendor)"
    )
    p_ing.add_argument("--input", required=True)
    p_ing.add_argument("--output", required=True, help="output .parquet path")
    p_ing.add_argument("--manifest-dir", default="data/manifests")
    p_ing.add_argument(
        "--spx-proxy",
        action="store_true",
        help="apply explicit SPX->XSP /10 proxy transform (root becomes *_PROXY)",
    )
    p_ing.set_defaults(func=_cmd_ingest)

    p_val = sub.add_parser("validate-data", help="run the full data-quality report on a dataset")
    p_val.add_argument("--options", required=True)
    p_val.set_defaults(func=_cmd_validate_data)

    p_feat = sub.add_parser("build-features", help="build the daily feature frame + manifest")
    p_feat.add_argument("--synthetic", action="store_true")
    p_feat.add_argument("--seed", type=int, default=7)
    p_feat.add_argument("--bundle-dir", help="directory of <SYMBOL>.parquet (date, close) files")
    p_feat.add_argument("--sectors", help="comma-separated sector symbols within the bundle dir")
    p_feat.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 2))
    p_feat.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    p_feat.add_argument("--families", help="comma-separated feature families (default: all)")
    p_feat.add_argument("-o", "--output", required=True, help="output parquet path")
    p_feat.set_defaults(func=_cmd_build_features)

    p_exp = sub.add_parser("run-experiment", help="run a tracked, config-driven experiment")
    p_exp.add_argument("-c", "--config", required=True, help="experiment YAML")
    p_exp.add_argument("--synthetic", action="store_true")
    p_exp.add_argument("--seed", type=int, default=7)
    p_exp.add_argument("--bundle-dir")
    p_exp.add_argument("--sectors")
    p_exp.add_argument("--output-root", default="reports/experiments")
    p_exp.set_defaults(func=_cmd_run_experiment)

    p_info = sub.add_parser("info")
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
