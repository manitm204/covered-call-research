"""Experiment runner: artifacts, provenance, research dataset, filter effects."""

import json
from datetime import date

import polars as pl
import pytest

from xsp_research.experiments.filters import FilterClause
from xsp_research.experiments.runner import (
    ExperimentConfig,
    build_research_dataset,
    load_experiment_config,
    run_experiment,
)
from xsp_research.features.surface import SURFACE_FEATURE_NAMES
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket, research_bundle

START, END = date(2023, 1, 2), date(2023, 9, 29)


@pytest.fixture(scope="module")
def market():
    return SyntheticMarket(SyntheticConfig(start=START, end=END, strike_pct_range=0.12))


@pytest.fixture(scope="module")
def bundle(market):
    return research_bundle(market)


@pytest.fixture(scope="module")
def strategy_yaml(tmp_path_factory):
    """Baseline strategy config with the test window."""
    import yaml

    from xsp_research.config import load_strategy_config

    cfg = load_strategy_config("configs/strategy_baseline.yaml")
    raw = cfg.model_dump(mode="json")
    raw["backtest"]["start"] = str(START)
    raw["backtest"]["end"] = str(END)
    p = tmp_path_factory.mktemp("cfg") / "strategy.yaml"
    p.write_text(yaml.safe_dump(raw))
    return str(p)


@pytest.fixture(scope="module")
def baseline_run(market, bundle, strategy_yaml, tmp_path_factory):
    exp = ExperimentConfig(
        name="test_mechanical", strategy_config=strategy_yaml, scenarios=["base"]
    )
    out_root = tmp_path_factory.mktemp("experiments")
    return run_experiment(exp, market, market, market, bundle, out_root=out_root)


class TestArtifacts:
    def test_clean_run_no_errors(self, baseline_run):
        assert baseline_run.errors == []
        assert (baseline_run.out_dir / "errors.log").read_text() == ""

    def test_provenance_contents(self, baseline_run):
        prov = json.loads((baseline_run.out_dir / "provenance.json").read_text())
        assert prov["git_commit"], "repo has commits; hash must be captured"
        assert prov["data_sources"]["options"] == "synthetic"
        assert prov["config_hash"][:8] in baseline_run.experiment_id

    def test_feature_manifest_written(self, baseline_run):
        manifest = json.loads((baseline_run.out_dir / "feature_manifest.json").read_text())
        assert len(manifest["features"]) >= 30
        assert manifest["skipped"] == {}

    def test_scenario_artifacts(self, baseline_run):
        scen = baseline_run.out_dir / "base"
        assert (scen / "summary.json").exists()
        assert (scen / "trades.parquet").exists()
        assert (scen / "research_dataset.parquet").exists()

    def test_summary_labeled_synthetic(self, baseline_run):
        assert "SYNTHETIC" in baseline_run.summaries["base"]["WARNING"]


class TestResearchDataset:
    def test_one_row_per_trade_with_features(self, baseline_run):
        scen = baseline_run.out_dir / "base"
        research = pl.read_parquet(scen / "research_dataset.parquet")
        trades = pl.read_parquet(scen / "trades.parquet")
        assert research.height == trades.height
        assert "vix_level" in research.columns  # daily feature at entry date
        for name in SURFACE_FEATURE_NAMES:
            assert name in research.columns
        # Entry-time surface delta should match the trade's recorded short delta.
        joined = research.select(["short_delta_at_entry", "surf_short_delta"]).drop_nulls()
        for a, b in joined.iter_rows():
            assert a == pytest.approx(b, abs=1e-9)

    def test_empty_result_yields_empty_dataset(self, bundle):
        from xsp_research.backtest.engine import BacktestResult
        from xsp_research.backtest.ledger import Ledger

        empty = BacktestResult(
            data_source="synthetic",
            config_snapshot={"backtest": {"initial_cash": 1.0}},
            execution_scenario="base",
            trades=[],
            equity_curve=pl.DataFrame(),
            entry_attempts=[],
            ledger=Ledger(),
            positions=[],
        )
        assert build_research_dataset(empty, pl.DataFrame()).is_empty()


class TestFilteredExperiment:
    def test_impossible_filter_blocks_everything(
        self, market, bundle, strategy_yaml, tmp_path_factory
    ):
        exp = ExperimentConfig(
            name="test_impossible_filter",
            strategy_config=strategy_yaml,
            scenarios=["base"],
            entry_filters=[FilterClause(feature="vix_level", op="<", value=-1.0)],
        )
        run = run_experiment(
            exp, market, market, market, bundle, out_root=tmp_path_factory.mktemp("exp")
        )
        assert run.errors == []
        assert run.summaries["base"]["trades"]["n_trades"] == 0
        rejected = run.summaries["base"]["entry_attempts"]["rejected"]
        assert rejected and all("filtered" in r["reason"] for r in rejected)

    def test_filter_changes_id_via_config_hash(self, strategy_yaml):
        a = ExperimentConfig(name="x", strategy_config=strategy_yaml)
        b = ExperimentConfig(
            name="x",
            strategy_config=strategy_yaml,
            entry_filters=[FilterClause(feature="vix_level", op="<", value=20.0)],
        )
        assert a.content_hash() != b.content_hash()


def test_load_experiment_config_yaml():
    exp = load_experiment_config("configs/experiments/exp_vix_filter.yaml")
    assert exp.name == "vix_below_20"
    assert exp.entry_filters[0].feature == "vix_level"
    assert exp.scenarios == ["base", "conservative"]


def test_unknown_scenario_rejected(market, bundle, strategy_yaml, tmp_path):
    exp = ExperimentConfig(name="bad_scenario", strategy_config=strategy_yaml, scenarios=["nope"])
    with pytest.raises(ValueError, match="unknown execution scenarios"):
        run_experiment(exp, market, market, market, bundle, out_root=tmp_path)
