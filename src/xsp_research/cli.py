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


def _load_dividends(cfg, dividends_data: str | None):
    """Dividend calendar for American-exercise configs (None for European)."""
    if cfg.selection.exercise_style != "american":
        return None
    from xsp_research.backtest.american import DividendCalendar

    div_path = Path(
        dividends_data or f"data/normalized/aux/{cfg.selection.root}_DIVIDENDS.parquet"
    )
    if not div_path.exists():
        print(
            f"error: American exercise needs a dividend calendar; {div_path} not found "
            "(run `xsp ingest-aux` or pass --dividends-data)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return DividendCalendar.from_parquet(div_path, cfg.selection.root)


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

    dividends = _load_dividends(cfg, getattr(args, "dividends_data", None))

    engine = BacktestEngine(
        cfg,
        options,
        underlying,
        rates,
        execution_scenario=args.scenario or "config",
        dividends=dividends,
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

    if not args.bundle_dir:
        print("error: need --synthetic or --bundle-dir", file=sys.stderr)
        raise SystemExit(2)
    from xsp_research.ingestion.aux_data import load_aux_bundle

    bundle = load_aux_bundle(args.bundle_dir)
    if args.sectors:  # explicit override of the auto-detected sector list
        from xsp_research.features.registry import MarketDataBundle

        sectors = tuple(s.strip() for s in args.sectors.split(",") if s.strip())
        bundle = MarketDataBundle(series=bundle.series, sector_symbols=sectors)
    return None, bundle


def _cmd_ingest_aux(args: argparse.Namespace) -> int:
    """Download the free FRED/Stooq/Cboe auxiliary bundle + write manifest."""
    from xsp_research.ingestion.aux_data import DEFAULT_UNIVERSE, ingest_aux_bundle

    only = tuple(s.strip() for s in args.only.split(",")) if args.only else None
    manifest = ingest_aux_bundle(
        args.output, DEFAULT_UNIVERSE, start=args.start, manifest_dir=args.manifest_dir, only=only
    )
    summary = {
        "series_ok": len(manifest["series"]),
        "series_failed": len(manifest["errors"]),
        "errors": manifest["errors"],
        "sectors_detected": manifest["sector_symbols"],
        "out_dir": manifest["out_dir"],
        "manifest": str(Path(args.manifest_dir) / "aux_bundle.manifest.json"),
    }
    print(json.dumps(summary, indent=2))
    return 0 if not manifest["errors"] else 1


def _cmd_terminal_start(args: argparse.Namespace) -> int:
    from xsp_research.ingestion.thetadata import launch_terminal, terminal_running

    if terminal_running():
        print("Theta Terminal already running.")
        return 0
    proc = launch_terminal()
    print(f"Theta Terminal started (pid {proc.pid}) and answering.")
    return 0


def _cmd_pull_thetadata(args: argparse.Namespace) -> int:
    from xsp_research.ingestion.thetadata import (
        TerminalNotRunningError,
        ThetaDataClient,
        pull_chain_history,
        terminal_running,
    )

    if not terminal_running():
        print(TerminalNotRunningError(ThetaDataClient().base_url), file=sys.stderr)
        return 2
    client = ThetaDataClient()
    manifest = pull_chain_history(
        client,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        out_dir=args.output,
        snapshot_et=args.snapshot,
        max_dte=args.max_dte,
        with_open_interest=not args.no_oi,
        refresh=args.refresh,
        progress=lambda msg: print(msg, file=sys.stderr),
    )
    from xsp_research.ingestion.thetadata import pull_underlying_eod

    eod_path = Path(args.output) / "underlying_eod.parquet"
    try:
        eod = pull_underlying_eod(client, args.symbol, args.start, args.end, eod_path)
        print(f"underlying EOD closes: {eod.height} rows -> {eod_path}", file=sys.stderr)
    except Exception as exc:
        print(f"warning: underlying EOD pull failed: {exc}", file=sys.stderr)
    print(
        json.dumps(
            {
                "months_written": len(manifest["months_written"]),
                "months_skipped_existing": len(manifest["months_skipped_existing"]),
                "empty_sessions": len(manifest["empty_sessions"]),
                "session_errors": manifest["session_errors"],
                "underlying_price_methods": manifest["underlying_price_methods"],
                "out_dir": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if not manifest["session_errors"] else 1


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
    from xsp_research.config import load_strategy_config
    from xsp_research.experiments.runner import load_experiment_config, run_experiment

    exp = load_experiment_config(args.config)
    scfg = load_strategy_config(exp.strategy_config)
    dividends = None
    if args.synthetic:
        market, bundle = _load_bundle(args, scfg.backtest.start, scfg.backtest.end)
        options = underlying = rates = market
    else:
        if not (
            args.options_data and args.underlying_data and args.rates_data and args.bundle_dir
        ):
            print(
                "error: real-data experiments need --options-data, --underlying-data, "
                "--rates-data and --bundle-dir (or use --synthetic)",
                file=sys.stderr,
            )
            return 2
        from xsp_research.ingestion.file_provider import (
            ParquetOptionsProvider,
            ParquetUnderlyingProvider,
            SeriesRatesProvider,
        )

        options = ParquetOptionsProvider(args.options_data, scfg.selection.root)
        underlying = ParquetUnderlyingProvider(args.underlying_data, scfg.selection.root)
        rates = SeriesRatesProvider(args.rates_data)
        _, bundle = _load_bundle(args, scfg.backtest.start, scfg.backtest.end)
        dividends = _load_dividends(scfg, args.dividends_data)
    run = run_experiment(
        exp, options, underlying, rates, bundle, out_root=args.output_root, dividends=dividends
    )
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


def _cmd_evaluate_model(args: argparse.Namespace) -> int:
    """Walk-forward benchmark-model evaluation on a research dataset."""
    import polars as pl

    from xsp_research.models.evaluate import run_family_ablation, run_walkforward
    from xsp_research.models.walkforward import WalkForwardConfig

    dataset = pl.read_parquet(args.research_dataset)
    wf_cfg = WalkForwardConfig(
        n_folds=args.folds,
        embargo_days=args.embargo_days,
        final_test_start=args.final_test_start,
    )
    if args.ablation:
        result = run_family_ablation(dataset, args.label, wf_cfg, model_name=args.model)
    elif args.nested_boosting:
        from xsp_research.models.boosting import nested_boosting_walkforward
        from xsp_research.models.evaluate import feature_family_columns

        fams = feature_family_columns(dataset)
        if args.families:
            wanted = set(args.families.split(","))
            fams = {k: v for k, v in fams.items() if k in wanted}
        cols = sorted({c for cols in fams.values() for c in cols})
        result = nested_boosting_walkforward(dataset, cols, args.label, wf_cfg)
    else:
        from xsp_research.models.evaluate import feature_family_columns

        fams = feature_family_columns(dataset)
        if args.families:
            wanted = set(args.families.split(","))
            fams = {k: v for k, v in fams.items() if k in wanted}
        cols = sorted({c for cols in fams.values() for c in cols})
        result = run_walkforward(dataset, cols, args.label, wf_cfg)
    print(json.dumps(result, indent=2, default=str))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    return 0 if "error" not in result else 1


def _cmd_fit_surface(args: argparse.Namespace) -> int:
    """Fit SVI slices to one snapshot and print arbitrage/fit diagnostics."""
    from datetime import timedelta

    from xsp_research.options.svi_surface import fit_surface_from_chain

    if args.synthetic:
        from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket

        print(SYNTHETIC_BANNER, file=sys.stderr)
        market = SyntheticMarket(
            SyntheticConfig(
                start=args.session - timedelta(days=5),
                end=args.session + timedelta(days=90),
                seed=args.seed,
            )
        )
        session = next(
            (d for d in market.trading_dates(args.session, args.session + timedelta(days=7))),
            None,
        )
        if session is None:
            print("error: no synthetic session near that date", file=sys.stderr)
            return 2
        chain = market.chain(market._snapshot_ts(session))
    elif args.options_data:
        import polars as pl

        df = pl.read_parquet(args.options_data)
        chain = df.filter(pl.col("ts").dt.date() == args.session)
        session = args.session
        if chain.is_empty():
            print(f"error: no rows for session {args.session}", file=sys.stderr)
            return 2
    else:
        print("error: need --synthetic or --options-data", file=sys.stderr)
        return 2

    spot = float(chain["underlying_price"][0])
    surface = fit_surface_from_chain(chain, spot, args.rate, args.div_yield, session)
    print(json.dumps(surface.diagnostics(), indent=2))
    return 0


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
    print("xsp-research: XSP bear call credit spread research framework (all 5 phases complete)")
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
    p_run.add_argument(
        "--dividends-data", help="(ex_date, amount) parquet for American-exercise modeling"
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

    p_aux = sub.add_parser("ingest-aux", help="download the free FRED/Stooq/Cboe auxiliary bundle")
    p_aux.add_argument("-o", "--output", default="data/normalized/aux")
    p_aux.add_argument("--start", type=date.fromisoformat, default=date(2016, 1, 1))
    p_aux.add_argument("--manifest-dir", default="data/manifests")
    p_aux.add_argument("--only", help="comma-separated symbols to (re)fetch, merging the manifest")
    p_aux.set_defaults(func=_cmd_ingest_aux)

    p_term = sub.add_parser("terminal-start", help="download/launch the ThetaData terminal")
    p_term.set_defaults(func=_cmd_terminal_start)

    p_pull = sub.add_parser(
        "pull-thetadata", help="pull daily option-chain snapshots from ThetaData"
    )
    p_pull.add_argument("--symbol", default="SPY")
    p_pull.add_argument("--start", type=date.fromisoformat, required=True)
    p_pull.add_argument("--end", type=date.fromisoformat, required=True)
    p_pull.add_argument("--snapshot", default="15:30:00", help="ET snapshot time HH:MM:SS")
    p_pull.add_argument("--max-dte", type=int, default=70)
    p_pull.add_argument("--no-oi", action="store_true", help="skip open-interest requests")
    p_pull.add_argument("--refresh", action="store_true", help="re-pull existing months")
    p_pull.add_argument("-o", "--output", default="data/normalized/options/spy")
    p_pull.set_defaults(func=_cmd_pull_thetadata)

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
    p_exp.add_argument("--options-data", help="path to canonical options parquet dataset")
    p_exp.add_argument("--underlying-data", help="path to (date, close) file")
    p_exp.add_argument("--rates-data", help="path to (date, rate) file")
    p_exp.add_argument(
        "--dividends-data", help="(ex_date, amount) parquet for American-exercise modeling"
    )
    p_exp.add_argument("--output-root", default="reports/experiments")
    p_exp.set_defaults(func=_cmd_run_experiment)

    p_eval = sub.add_parser(
        "evaluate-model", help="walk-forward benchmark models on a research dataset"
    )
    p_eval.add_argument("--research-dataset", required=True, help="research_dataset.parquet path")
    p_eval.add_argument(
        "--label",
        default="label_expire_itm",
        help="target column (label_expire_itm, label_touch, label_net_pnl, label_mae, ...)",
    )
    p_eval.add_argument("--folds", type=int, default=4)
    p_eval.add_argument("--embargo-days", type=int, default=5)
    p_eval.add_argument(
        "--final-test-start",
        type=date.fromisoformat,
        default=None,
        help="samples whose label window reaches this date are excluded (untouched final test)",
    )
    p_eval.add_argument("--families", help="comma-separated feature families to use")
    p_eval.add_argument(
        "--ablation", action="store_true", help="run the feature-family ablation grid"
    )
    p_eval.add_argument(
        "--nested-boosting",
        action="store_true",
        help="tuned XGBoost with nested inner splits + benchmark admission comparison",
    )
    p_eval.add_argument("--model", default="logistic", help="model for --ablation runs")
    p_eval.add_argument("-o", "--output", help="write result JSON here")
    p_eval.set_defaults(func=_cmd_evaluate_model)

    p_surf = sub.add_parser("fit-surface", help="fit SVI slices to a chain snapshot + diagnostics")
    p_surf.add_argument("--synthetic", action="store_true")
    p_surf.add_argument("--seed", type=int, default=7)
    p_surf.add_argument("--options-data", help="canonical chain parquet")
    p_surf.add_argument("--session", type=date.fromisoformat, required=True)
    p_surf.add_argument("--rate", type=float, default=0.045)
    p_surf.add_argument("--div-yield", type=float, default=0.015)
    p_surf.set_defaults(func=_cmd_fit_surface)

    p_info = sub.add_parser("info")
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
