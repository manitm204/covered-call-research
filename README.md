# xsp-research

Research and backtesting framework for an **XSP bear call credit spread** overlay strategy
(European-style, cash-settled, $100 multiplier), designed for correctness, reproducibility,
realistic execution, and protection against data leakage and overfitting.

The two research questions this framework exists to answer:

1. Does the spread have positive standalone expected value **after all costs**?
2. Does adding the spread to a long stock portfolio improve **total-portfolio**
   risk-adjusted performance?

See `docs/PLAN.md` for the architecture, data dictionary, material assumptions, and the
authoritative definitions of every P&L and performance calculation.

## Status

**Phase 1 complete**: configuration system, validated domain models, provider interfaces,
Black-Scholes/IV/Greeks, deterministic audited contract selection, execution-cost model,
double-entry ledger with collateral and interest accrual, daily backtest engine, metrics,
CLI.

**Phase 2 complete**: vendor-file ingestion (declarative column mappings, Cboe DataShop
preset, explicit opt-in SPX→XSP `/10` proxy transform labeled `*_PROXY`, SHA-256
manifests), full data-quality validation suite (duplicates, crossed/locked, zero-bid,
below-intrinsic, abnormal spreads, stale quotes, underlying misalignment, strike/session
gaps, put-call-parity dispersion), portfolio overlay (stock-only vs overlay vs combined,
margin-overlay and carve-out capital models, downside beta/capture/CVaR), stress-window
reporting (Q4-2018, COVID, rebounds, 2022 bear — uncovered windows reported, never
skipped), and Markdown/JSON/chart report generation.

**Phase 3 complete**: feature registry (39 daily features across volatility/trend/
breadth/cross-asset/PCA-regime families, each with definition, sources, mandatory
lag >= 1 session, missing-value policy), snapshot-exact options-surface features,
empirical leakage validation (prefix-consistency: removing future data must never
change past feature values — enforced in CI for every registered feature),
declarative entry filters, and a tracked experiment runner (`xsp run-experiment`)
producing config snapshots, feature manifests, git/data provenance, and per-scenario
trade-level research datasets.

**Phase 4 complete**: economically meaningful labels (expiration-ITM, close-based
touch, net P&L, max adverse excursion — each with a `label_end` purge window);
purged/embargoed forward-chaining walk-forward splits with an untouched final-test
guard; the mandated benchmark-model ladder (base rate first, then logistic, L1,
shallow tree — fold-local preprocessing only); calibration diagnostics (Brier, log
loss, reliability tables, calibration slope/intercept, ECE) plus realized-P&L-by-
predicted-decile tables; and the required feature-family ablation grid
(`xsp evaluate-model [--ablation]`).

**Phase 5 complete**: raw-SVI surface fitting with butterfly/calendar arbitrage
diagnostics (`xsp fit-surface`); nested-tuned XGBoost with per-fold inner purged
splits and an explicit admission rule against simpler benchmarks
(`xsp evaluate-model --nested-boosting`); robustness suite (block-bootstrap CIs for
trade expectancy and Sharpe, short-delta × width grid re-runs); a single-use
final-test evaluator with an on-disk usage guard; and pre-registered research
conclusions / go-no-go gates in `docs/CONCLUSIONS.md`. 285 tests.

All five phases of the framework are built and verified. The single remaining
blocker for real research results is licensed XSP/SPX options history.

**No real market data is bundled.** Runs against synthetic data are for software
validation only and are labeled as such in every output. Historical XSP/SPX options data
must be licensed separately (ThetaData, Polygon, Cboe DataShop, ORATS, or user-supplied
files) and placed under `data/` (git-ignored).

## Install

```bash
pip install -e ".[dev]"       # add [ml,viz] for later phases
pytest                        # run the full test suite
```

## Usage

```bash
# Software-validation run on labeled synthetic data:
xsp run-backtest -c configs/strategy_baseline.yaml --synthetic --scenario base -o reports/dev

# Real data (canonical Parquet schemas documented in docs/PLAN.md):
xsp run-backtest -c configs/strategy_baseline.yaml \
  --options-data data/normalized/options \
  --underlying-data data/normalized/underlying/xsp.parquet \
  --rates-data data/normalized/rates/tbill_4w.parquet \
  --scenario conservative -o reports/run1

# Schema/quality check of an options dataset:
xsp validate-data --options data/normalized/options/2023-01-03.parquet
```

Execution scenarios (`--scenario`): `optimistic` (midpoint), `base` (halfway between
natural and midpoint), `conservative` (natural + slippage). Results should always be
compared across all three; gross and net P&L are reported separately.

## Data licensing

Options quote data is licensed by its vendor and must not be committed to this
repository (`.gitignore` enforces this). Record vendor, license terms, and file hashes
in `data/manifests/`.

## Honesty guarantees baked into the code

- Every `BacktestResult` carries `data_source`; synthetic runs print a warning banner
  and embed a warning in the summary JSON.
- Every candidate contract at entry gets an accept/reject audit record.
- Every cash movement is a balanced double-entry posting; the engine hard-fails the run
  if ledger equity stops reconciling with position marks.
- Providers are queried strictly as-of; a regression test asserts the engine's query
  stream never moves backward or beyond the simulation clock.
- Interest income is ledgered separately from trading P&L and never mixed into it.
