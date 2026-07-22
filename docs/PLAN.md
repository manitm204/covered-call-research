# XSP Bear Call Credit Spread — Research Framework Plan

Status: living document. Created 2026-07-22 at project inception.

---

## 1. Repository summary (at inception)

The repository was empty. Everything described below is built from scratch. The environment
provides Python 3.11.9 with numpy/scipy/pandas/polars/duckdb/pydantic/pytest/scikit-learn/
xgboost/lightgbm/optuna available.

## 2. System architecture

```
configs/                  YAML configuration (data, strategy, costs, experiments)
data/
  raw/                    provider-native files (never committed)
  normalized/             canonical Parquet datasets (see data dictionary)
  features/               feature matrices (Phase 3+)
  manifests/              JSON manifests: file hashes, row counts, date coverage
src/xsp_research/
  config.py               pydantic config models + YAML loader
  domain.py               validated domain objects (contracts, quotes, spreads, positions)
  cli.py                  command-line interface
  ingestion/
    base.py               provider Protocols + canonical chain schema + schema validation
    file_provider.py      user-supplied Parquet/CSV adapter (provider-agnostic)
    synthetic.py          SYNTHETIC test fixture generator (never evidence of profitability)
  options/
    black_scholes.py      forward pricing, Black-Scholes / Black-76, analytic Greeks
    implied_vol.py        robust IV inversion (bracketed Brent with no-arb bounds)
    greeks.py             per-quote and spread-level Greek aggregation
  strategy/
    candidate_selection.py  deterministic, fully audited expiration/strike selection
    exit_rules.py           configurable prioritized exit policies
    sizing.py               position sizing (fixed qty Phase 1; risk-budget later)
  backtest/
    engine.py             timestamp-correct daily event loop
    execution.py          fill models (natural/mid/pct-between), costs, rejections
    ledger.py             double-entry ledger, collateral tracking, interest accrual
  evaluation/
    metrics.py            trade- and equity-curve-level performance metrics
  features/, models/      Phases 3-5
tests/                    pytest suite incl. hand-computed examples and invariants
docs/                     this plan, data licensing notes
reports/                  generated outputs (regenerable, not committed)
```

Design principles:
- **Provider isolation.** The engine consumes three `Protocol`s (`OptionsProvider`,
  `UnderlyingProvider`, `RatesProvider`). Vendor-specific code lives only in
  `ingestion/`; a new provider is a new adapter mapping to the canonical schema.
- **Timestamp discipline.** Every provider query takes an explicit `as_of` timestamp and
  must return only data at or before it. The engine never constructs a future timestamp.
  A test asserts the engine's query stream is monotone and ≤ simulation time.
- **Auditability.** Every candidate contract considered at entry is recorded with an
  accept/reject reason. Every cash movement is a balanced double-entry posting.
- **Honest data labeling.** `BacktestResult.data_source` is carried through to every
  report. Synthetic runs print a banner and are labeled `synthetic` everywhere.

## 3. Data dictionary (canonical normalized schemas)

### 3.1 Options chain snapshot (`data/normalized/options/`, Parquet, hive-partitioned by date)

| column            | type              | description |
|-------------------|-------------------|-------------|
| ts                | datetime[us, UTC] | quote snapshot timestamp (exchange timestamp, converted to UTC) |
| root              | str               | option root, e.g. `XSP` (or `SPX` for proxy data — never mixed silently) |
| expiration        | date              | expiration date (exchange calendar date) |
| strike            | f64               | strike price in index points |
| option_type       | str               | `C` or `P` |
| bid               | f64               | best bid (0.0 allowed = no bid) |
| ask               | f64               | best ask |
| bid_size          | i64 (nullable)    | size at bid, contracts |
| ask_size          | i64 (nullable)    | size at ask, contracts |
| volume            | i64 (nullable)    | cumulative day volume at ts |
| open_interest     | i64 (nullable)    | prior-day open interest |
| underlying_price  | f64               | underlying level aligned to the same ts |

Settlement/exercise metadata is contract-level, held in `domain.OptionContract`
(`exercise_style=european`, `settlement=PM` for XSP; standard 3rd-Friday XSP is also
PM-settled per Cboe spec — verified against contract specs before real-data runs).

### 3.2 Underlying daily bars (`data/normalized/underlying/`)
`date, open, high, low, close, volume` per symbol (XSP, SPY, RSP, QQQ, IWM, sectors, ...).

### 3.3 Rates (`data/normalized/rates/`)
`date, tenor, rate` — annualized decimal. Baseline: 4-week T-bill (FRED DTB4WK) or SOFR.
Used for (a) discounting/forwards, (b) cash interest accrual. Both configurable separately.

### 3.4 Auxiliary series (Phase 2+)
VIX family, credit ETFs, commodities, dollar — `date, symbol, value`.

### 3.5 Manifests
Every normalized dataset gets `data/manifests/<name>.json`: source, license note, file
SHA-256 hashes, row counts, date coverage, ingestion timestamp, adapter version.

## 4. Material assumptions (each affects results; all configurable or flagged)

1. **XSP pre-2022 data**: XSP was relaunched with market-maker incentives in 2022;
   earlier quotes are sparse and wide. Either restrict to 2022+, or use SPX/10 as a
   structural proxy for 2018-2021 (identical European/cash-settled mechanics; strike
   grid 10x coarser: a $5 XSP width = $50 SPX width, which exists). Proxy runs are
   labeled `root=SPX_PROXY` and reported separately. **Never silently mixed.**
2. **Snapshot granularity**: exits (profit target, stop, delta stop) are evaluated only
   at available snapshot timestamps (baseline: one intraday snapshot near 15:30 ET plus
   settlement). Real intraday stops would trigger earlier; our stops are therefore
   evaluated pessimistically late, and intratrade MAE is a lower bound. Documented in
   every report.
3. **Settlement**: XSP is PM-settled → settlement value = official closing index value on
   expiration day. We use the underlying close from the data provider; the official Cboe
   settlement value can differ by small amounts. AM settlement is unsupported (not needed
   for XSP; guarded by an explicit error).
4. **Collateral**: width × $100 × qty is restricted while the spread is open (Reg-T
   style); the credit received is unrestricted cash. Broker-specific treatment differs;
   configurable.
5. **Interest**: default accrues on free cash only (cash minus restricted collateral) at
   T-bill proxy minus a configurable spread. Never assumes collateral earns interest
   unless explicitly configured.
6. **Greeks/IV are model-derived** (Black-Scholes on parity-implied forward when both
   sides available, else continuous carry). Vendor deltas, if present, are ignored for
   selection so results are provider-independent. Delta targeting therefore inherits BS
   model error — that is the point of Section 12 of the research brief (models of real
   probabilities are built and compared against BS quantities, not assumed equal).
7. **Fees**: baseline $0.65/contract commission + $0.60/contract exchange+regulatory
   (XSP is a Cboe proprietary product with an index-option fee; verify against your
   broker's schedule). Scenario multipliers apply.
8. **Fill realism**: base scenario fills credit spreads halfway between natural and mid;
   conservative fills at natural plus slippage; optimistic at mid. No fill if quotes are
   crossed/locked/stale-flagged or relative width exceeds the configured cap.
9. **Calendar**: trading calendar = dates present in the underlying dataset (no invented
   sessions). Interest accrues on calendar days (ACT/365).

## 5. Phased implementation plan

- **Phase 1 (this build)**: architecture, configs, domain, provider interfaces,
  synthetic fixtures, BS/IV/Greeks, selection, execution model, ledger, daily engine,
  metrics, CLI, tests.
- **Phase 2 (done)**: vendor-file ingestion with declarative mappings + manifests
  (`ingestion/vendors.py`; live REST clients deferred until a vendor is licensed),
  data-validation reports (crossed/locked/zero-bid/below-intrinsic/stale/parity/gap
  diagnostics in `ingestion/validation.py`), portfolio overlay with margin-overlay and
  carve-out capital models (`backtest/portfolio_overlay.py`), stress-window and
  baseline reporting (`evaluation/stress_tests.py`, `evaluation/reporting.py`).
  Overlay capital model note: margin_overlay credits NO cash interest to the overlay
  (capital is fully invested in stock); carve_out reserves a cash sleeve that does.
- **Phase 3 (done)**: feature registry with per-feature lag/missing-policy/availability
  metadata (`features/registry.py`); 39 daily features across volatility, trend,
  breadth, cross-asset, and PCA-regime families plus snapshot-based surface features
  (`features/surface.py`); empirical leakage validation via prefix-consistency checks
  (`features/leakage.py`); declarative entry filters and a tracked experiment runner
  (`experiments/`) writing config snapshots, feature manifests, provenance (git commit,
  config hash, data-source labels), and per-scenario research datasets.
  Timestamp convention: daily features carry a mandatory lag >= 1 session (the 15:30 ET
  decision precedes that day's close); surface features are lag-0 snapshot-exact.
- **Phase 4**: walk-forward with purge/embargo for overlapping 30-day labels; base-rate
  and logistic benchmarks; calibration reports; expected-P&L and MAE regression targets;
  ablations.
- **Phase 5**: SVI surface with arbitrage diagnostics; gradient boosting with nested
  tuning; final untouched 2025+ evaluation; research conclusions.

Chronological research design (adjusted once real data coverage is known):
2018-2021 development/training → 2022 first OOS regime → 2023-2024 rolling validation →
2025-present final untouched test (used once).

## 6. Required external data vs. what proceeds without it

Required (licensed): XSP (and/or SPX) options quotes with timestamps 2018-present;
underlying daily bars; VIX family; Treasury/SOFR rates; ETF series (SPY/RSP/QQQ/IWM/
sectors/HYG/LQD); optionally MOVE (licensing).

Proceeds without licensed data (Phase 1, complete): everything listed under Phase 1 —
all logic is exercised against clearly-labeled synthetic fixtures. No performance claims
are made from synthetic data.

Blocked without licensed data: any statement about historical profitability, regime
behavior, model training, or overlay benefit.

## 7. Baseline strategy configuration

See `configs/strategy_baseline.yaml`. Summary: sell XSP call nearest 0.15 delta at the
expiration nearest 30 DTE within [25, 35]; buy the call $5 higher; enter monthly at the
first eligible session at 15:30 ET snapshot; 1 spread per entry; exits (priority order):
expiration settlement, profit target 50% of max profit, stop when cost-to-close ≥ 2.0x
entry credit, time exit at 7 DTE (disabled by default in the pure baseline), delta stop
at 0.50 (disabled by default); collateral = width×100; interest on free cash at T-bill
proxy; base execution scenario.

## 8. P&L and performance definitions (authoritative)

Let `c` = entry credit per spread (index points), `d` = exit debit per spread, `w` =
width, `q` = quantity, `M` = 100 (multiplier). All fees `F` are itemized separately.

- **Entry cash flow** = `+c·M·q − F_open`
- **Cost to close (mark)** = per configured mark rule; default mid: `(short.mid − long.mid)` at mark ts, floored at 0 and capped at `w` (invariant)
- **Unrealized P&L** = `(c − mark)·M·q − F_open` (fees at entry are sunk)
- **Realized P&L (closed early)** = `(c − d)·M·q − F_open − F_close`
- **Settlement value** at expiration with settlement price `S_T`:
  `settle = min(max(S_T − K_short, 0), w) − ... ` precisely `max(S_T−K_short,0) − max(S_T−K_long,0)` (∈ [0, w])
- **Realized P&L (expired)** = `(c − settle)·M·q − F_open − F_settle`
- **Max profit** = `c·M·q` (gross of fees); **Max loss** = `(w − c)·M·q` (gross of fees). Invariants tested.
- **Return on max risk** = realized P&L ÷ `(w − c)·M·q`
- **% of max credit captured** = `(c − d or settle)/c`
- **MFE/MAE** = max/min over marks of unrealized P&L (gross of fees), snapshot-limited (Assumption 2)
- **Equity** = ledger cash + option position MTM (= −cost-to-close of open shorts) ; reconciled daily to the penny against the double-entry trial balance
- **Trading P&L vs interest**: interest income is a separate ledger account and is never mixed into trade P&L; reports show both separately and combined
- **CAGR** = `(E_T/E_0)^(365.25/days) − 1` on the daily equity curve
- **Sharpe** = mean(daily excess return over configured rf) / std · √252 ; **Sortino** uses downside std of excess returns; **Calmar** = CAGR/|MaxDD|
- **Max drawdown** = min over t of `E_t/max_{s≤t}E_s − 1`
- **Win rate / profit factor / expectancy** computed on realized net P&L per closed trade
- **Gross vs net**: every trade stores gross P&L (before all fees/slippage-vs-mid) and net; reports show both.

## 9. Leakage and overfitting controls (summary; enforced from Phase 1)

- Providers are as-of queried; engine query stream monotonicity is tested.
- Marks for exits use the same snapshot the decision is made at; execution uses that
  snapshot's quotes (no future quote selection).
- Later phases: purged/embargoed walk-forward, nested tuning, single-use final test
  period, ablations, and multiple-execution-scenario reporting are mandatory before any
  claim.
