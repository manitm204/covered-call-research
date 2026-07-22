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
CLI, and a 112-test suite.

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
