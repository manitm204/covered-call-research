"""Tracked, configuration-driven experiment runs.

Every run produces a self-describing artifact directory:

  reports/experiments/<experiment_id>/
    experiment_config.json    the experiment definition as executed
    strategy_snapshot.json    resolved strategy config
    feature_manifest.json     specs of every feature built (+skipped, with reasons)
    provenance.json           git commit, data-source labels, run timestamp, config hash
    features.parquet          the daily feature frame used for gating/research
    <scenario>/summary.json   metrics per execution scenario
    <scenario>/trades.parquet
    <scenario>/research_dataset.parquet   trades x (daily features @ entry + surface features)
    errors.log                exceptions per scenario (empty file = clean run)

The experiment_id embeds a content hash of the experiment definition so runs
of the same configuration are identifiable across time.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from pydantic import BaseModel

from xsp_research.backtest.engine import BacktestEngine, BacktestResult
from xsp_research.config import EXECUTION_SCENARIOS, load_strategy_config
from xsp_research.evaluation.metrics import summarize
from xsp_research.experiments.filters import EntryFilterConfig, FilterClause, make_entry_gate
from xsp_research.features import build_features
from xsp_research.features.registry import MarketDataBundle
from xsp_research.features.surface import compute_surface_features
from xsp_research.ingestion.base import OptionsProvider, RatesProvider, UnderlyingProvider

DAILY_FAMILIES = ["volatility", "trend", "breadth", "cross_asset", "pca_regime"]


class ExperimentConfig(BaseModel):
    name: str
    strategy_config: str  # path to a strategy YAML
    scenarios: list[str] = ["base"]
    feature_families: list[str] = DAILY_FAMILIES
    entry_filters: list[FilterClause] = []
    allow_entry_on_missing_feature: bool = False
    capture_entry_features: bool = True

    def content_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with open(path) as fh:
        return ExperimentConfig.model_validate(yaml.safe_load(fh))


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except OSError:
        return None


def build_research_dataset(result: BacktestResult, features: pl.DataFrame) -> pl.DataFrame:
    """Trade-level modeling table: outcomes + entry-date daily features + surface
    features. This is the input for Phase 4 label/model work."""
    trades = result.trades_frame()
    if trades.is_empty():
        return pl.DataFrame()
    trades = trades.with_columns(pl.col("entry_ts").dt.date().alias("entry_date"))
    daily = features.rename({"date": "entry_date"})
    out = trades.join(daily, on="entry_date", how="left")
    surf = result.entry_features_frame()
    if not surf.is_empty():
        out = out.join(surf, on="trade_id", how="left")
    return out


@dataclass(slots=True)
class ExperimentRun:
    experiment_id: str
    out_dir: Path
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def run_experiment(
    exp: ExperimentConfig,
    options: OptionsProvider,
    underlying: UnderlyingProvider,
    rates: RatesProvider,
    bundle: MarketDataBundle,
    out_root: str | Path = "reports/experiments",
) -> ExperimentRun:
    run_ts = datetime.now(UTC)
    experiment_id = f"{exp.name}-{run_ts:%Y%m%d-%H%M%S}-{exp.content_hash()[:8]}"
    out_dir = Path(out_root) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run = ExperimentRun(experiment_id=experiment_id, out_dir=out_dir)

    strategy_cfg = load_strategy_config(exp.strategy_config)
    built = build_features(bundle, families=exp.feature_families)

    gate = None
    if exp.entry_filters:
        gate = make_entry_gate(
            built.frame,
            EntryFilterConfig(
                clauses=exp.entry_filters,
                allow_on_missing=exp.allow_entry_on_missing_feature,
            ),
        )

    hook = None
    if exp.capture_entry_features:

        def hook(session, snap_ts, chain, sel, spot, rate):  # noqa: ANN001
            return compute_surface_features(chain, sel, spot, rate, 0.0, session)

    # --- provenance & static artifacts (written before runs so failures keep them)
    (out_dir / "experiment_config.json").write_text(
        json.dumps(exp.model_dump(mode="json"), indent=2)
    )
    (out_dir / "strategy_snapshot.json").write_text(json.dumps(strategy_cfg.snapshot(), indent=2))
    (out_dir / "feature_manifest.json").write_text(json.dumps(built.manifest(), indent=2))
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "run_ts_utc": run_ts.isoformat(timespec="seconds"),
                "git_commit": _git_commit(),
                "config_hash": exp.content_hash(),
                "data_sources": {
                    "options": options.data_source,
                    "underlying": underlying.data_source,
                    "rates": rates.data_source,
                },
            },
            indent=2,
        )
    )
    built.frame.write_parquet(out_dir / "features.parquet")

    unknown = [s for s in exp.scenarios if s not in EXECUTION_SCENARIOS]
    if unknown:
        raise ValueError(f"unknown execution scenarios: {unknown}")

    for scenario in exp.scenarios:
        scen_dir = out_dir / scenario
        scen_dir.mkdir(exist_ok=True)
        cfg = strategy_cfg.model_copy(update={"execution": EXECUTION_SCENARIOS[scenario]})
        try:
            engine = BacktestEngine(
                cfg,
                options,
                underlying,
                rates,
                execution_scenario=scenario,
                entry_gate=gate,
                entry_feature_hook=hook,
            )
            result = engine.run()
            summary = summarize(result)
            run.summaries[scenario] = summary
            (scen_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
            trades = result.trades_frame()
            if not trades.is_empty():
                trades.write_parquet(scen_dir / "trades.parquet")
            research = build_research_dataset(result, built.frame)
            if not research.is_empty():
                research.write_parquet(scen_dir / "research_dataset.parquet")
        except Exception as exc:  # log-and-continue: one scenario must not kill the rest
            run.errors.append(f"{scenario}: {type(exc).__name__}: {exc}")

    (out_dir / "errors.log").write_text("\n".join(run.errors))
    return run
