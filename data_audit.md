# Data audit — Level-2 strategy research (2026-07-29)

Scope: audit of all data available for researching a **new** options strategy executable
in a $10,000 Fidelity account with Tier 1 / Level 1–2 permissions (long calls/puts,
covered calls, cash-secured puts — **no spreads**). This audit covers the existing
repository data; any newly downloaded data will be appended in §8.

## 1. Inventory

| Dataset | Path | Coverage | Size / rows |
|---|---|---|---|
| SPY options chain snapshots | `data/normalized/options/spy/chain_YYYY-MM.parquet` (96 files) | 2018-08-01 → 2026-07-22, 1,990 sessions | ~9.0M rows |
| QQQ options chain snapshots | `data/normalized/options/qqq/` (96 files) | same span, 1,977 sessions (15 vendor gap days, documented) | ~6.4M rows |
| IWM options chain snapshots | `data/normalized/options/iwm/` (96 files) | same span, 1,948 sessions (46 vendor gap days, documented) | ~5.6M rows |
| Aux daily bundle (33 series) | `data/normalized/aux/` | 2016-01-04 → 2026-07-28 | ETFs, sector SPDRs, VIX/VIX9D/VVIX/VXN/RVX, SPX/NDX/RUT, HYG/LQD/GLD/USO/UUP/DBC, 3M rate, UNRATE, dividends |
| Engineered daily features (39) | `data/features/daily_features.parquet` | 2016 → 2026-07, lag ≥ 1 session enforced, leakage-tested in CI | 2,652 rows |
| 4-week T-bill rate | `data/normalized/rates/tbill_4w.parquet` | 2016-01-04 → 2026-07-21 | daily |
| Dividend calendars | `aux/{SPY,QQQ,IWM}_DIVIDENDS.parquet` | ex-dates + amounts through 2026-06 | quarterly |

Provenance: ThetaData v3 NBBO snapshots (options), FRED/Yahoo/Cboe (aux bundle).
SHA-256 manifests in `data/manifests/`; per-pull manifests alongside each chain dir.

## 2. What the options data is (and is not)

- **One NBBO quote snapshot per session at 15:30 ET** (19:30 UTC, verified single
  timestamp per session across all 96 months × 3 underlyings). This supports daily
  decision-frequency backtests with entries/exits priced 30 minutes before the close.
  It does **not** support intraday logic, 0DTE timing, or stop orders evaluated
  intraday (stops can only be evaluated at the daily snapshot).
- Fields: `bid/ask`, `bid_size/ask_size`, `open_interest`, `underlying_price`,
  strike, expiration, type. **`volume` is 100% null** (never pulled) — liquidity
  screens must use OI, quoted size, and spread width instead.
- **No vendor Greeks or vendor IV are present.** All Greeks/IV are computed locally
  (`src/xsp_research/options/`) from the same snapshot used for the trade decision —
  no timing leakage from vendor calculations is possible.
- Both calls **and puts** at every strike (verified symmetric row counts). All prior
  research used only the short-call side; the put side and long side are unmined.
- **Pull filter: max DTE = 70** (per `pull_manifest.json`). Strategies requiring
  > ~65 DTE at entry (LEAPS, long-dated stock replacement) are NOT testable with the
  current data and would need a new pull.
- Strike coverage is wide: ~0.34×–1.29× spot (SPY 2024 example) — deep ITM through
  far OTM is covered inside the DTE window.
- Expirations present: all listed weeklies/monthlies within 70 DTE (SPY: Mon/Wed/Fri
  weeklies + monthlies; QQQ/IWM similar in recent years, monthlies + Fridays earlier).

## 3. Quote quality (fresh audit, this study's tradeable zone)

Zone = 15–65 DTE, 0.80–1.20 moneyness, all 96 months, per underlying:

| | SPY | QQQ | IWM |
|---|---|---|---|
| Zone rows | 3.46M | 2.43M | 1.59M |
| Crossed quotes (bid>ask) | 936 (0.03%) | 238 (0.01%) | 10 (0.0006%) |
| Zero-bid rows | 36.7k (1.1%) | 17.8k (0.7%) | 9.9k (0.6%) |
| Median relative spread | 1.13% | 1.09% | 1.25% |
| Worst month (median rel. spread) | 2.2% (2023-12) | 3.3% (2020-03) | 3.7% (2020-03) |

Crossed/locked quotes are rejected at selection by the engine. Zero-bid rows are deep
OTM and excluded by liquidity filters. Even in March 2020 the median relative spread
in the tradeable zone stayed under 4%.

Typical structure-level costs (median across sampled months, 28–45 DTE):

| Structure | SPY | QQQ | IWM |
|---|---|---|---|
| ATM call: mid / abs spread | $8.45 / $0.04 (0.5%) | $9.93 / $0.055 (0.7%) | $5.88 / $0.05 (1.0%) |
| 5% OTM call | $0.78 / $0.02 (2.2%) | $2.62 / $0.03 (1.6%) | $1.73 / $0.03 (2.1%) |
| 7% ITM call | $32.61 / $0.30 (0.9%) | $27.39 / $0.18 (0.7%) | $15.73 / $0.14 (0.9%) |
| 5% OTM put | $2.93 / $0.02 (0.9%) | $4.02 / $0.03 (1.0%) | $2.19 / $0.035 (1.7%) |

Implication: single-leg entries cost roughly 0.5–2% of premium per side at the mid —
long-option strategies survive realistic fills only if expected edges are well above
~2–4% of premium round-trip plus fees.

## 4. Underlying prices, adjustment, and distributions

- `underlying_price` in the chain files is the **unadjusted, tradeable price level**
  at 15:30 ET, implied from put-call parity at pull time and validated against known
  history in the original ingestion (stock tier was blocked at the vendor).
- The aux bundle closes (`aux/SPY.parquet` etc.) are **dividend-adjusted** (verified:
  aux/chain ratio ≈ 0.972–0.978 in June 2024, converging toward 1.0 near present).
  Adjusted series are used only for return/feature computation; all trade pricing,
  strike selection, and moneyness use the unadjusted chain level. These two must not
  be mixed, and the engine keeps them separate.
- Ex-dividend calendars with amounts exist for all three ETFs through June 2026 and
  drive early-assignment hazard modeling (`backtest/american.py`, tested).
- ETF distributions for SPY/QQQ/IWM are ordinary quarterly cash dividends; no splits
  or special distributions occurred in the sample window (verified against calendars).

## 5. Known gaps and unusable segments

- **Vendor session gaps**: SPY 7 (holiday weeks only), QQQ 15, IWM 46 missing
  sessions, all documented with expiry-date patches in
  `results/{qqq,iwm}_prereg/DATA_NOTES.md`. Gap days cannot host entries/exits; the
  engine treats them as non-trading days. IWM's 46 gaps (2.3% of sessions) are the
  worst; none fall on option expiration Fridays after patching.
- Chain data ends **2026-07-22**; aux bundle ends 2026-07-28. The last week is not
  yet pulled (can be topped up for paper trading; not material for backtests).
- `volume` unusable (never pulled). OI is present (≈0% null pre-2025, ~5–6% null in
  some 2025 months; OI>0 on ~67–78% of all rows including far OTM).
- 0–3 DTE quotes exist but a 15:30 snapshot is too coarse to model 0DTE dynamics
  honestly; this study will not trade DTE < 5.
- No index-option (XSP/SPX) data: cash-settled European alternatives are out of
  scope unless newly licensed.

## 6. Timezone / duplication / staleness checks

- Every row carries a UTC timestamp; converted to US/Eastern all snapshots fall at
  exactly 15:30 ET (no DST drift — verified across winter/summer months).
- No duplicate (date, expiration, strike, type) rows found in sampled months; the
  original ingestion's validation suite (duplicates, crossed/locked, zero-bid,
  below-intrinsic, abnormal spread, stale quotes, strike/session gaps, parity
  dispersion) ran at pull time with no ERROR-level issues in the traded DTE range.
- Staleness: quoted sizes are present (bid_size/ask_size), and deep-ITM quotes show
  live parity-consistent pricing; the 15:30 snapshot is during regular hours with
  active NBBO. Residual risk: any single snapshot can still be momentarily stale;
  the conservative fill scenario (cross the full spread + slippage) bounds this.

## 7. Fitness-for-purpose summary

| Research need | Status |
|---|---|
| Long calls/puts, 5–65 DTE, SPY/QQQ/IWM | ✅ fully supported, both sides, 8 years, 3 bear/crash episodes |
| Covered calls / CSP on SPY/QQQ/IWM | ✅ data supports it, ❌ **capital does not** (needs $22k–$63k per 100-share/contract unit vs $10k account) |
| LEAPS / >70 DTE structures | ❌ not in data; new pull required |
| 0DTE / intraday | ❌ single daily snapshot; out of scope |
| CSP on a sub-$100 underlying (e.g. XLF/SLV/GDX) | ❌ no chain data on disk; ThetaData pull required (~hours); decision deferred to Phase 3 |
| Cash interest on idle balances | ✅ tbill_4w daily series |
| Signal features (trend, vol, breadth, IV-RV, rebound) | ✅ 39 leakage-tested daily features, 2016→2026 |
| Pre-2018 signal validation (price-only) | ⏳ requires FMP pull of 2000–2017 daily history (§8) |

## 8. Newly downloaded data (appended as acquired)

- **FMP daily price history** (`data/normalized/prices_long/`, manifest.json inside):
  SPY/QQQ/IWM 1999→2026-07-29 (adjusted + unadjusted + dividends; adjClose verified
  ≡ aux bundle on 2,656 overlapping sessions, ratio 1.0000; unadjusted close within
  0.03% of chain parity level), ^VIX 1999→2026, ^VXN 2001→2026, ^RVX 2004→2026,
  XLF/SLV/EWZ 1999·2006·2000→2026 with dividend calendars. Pulled via FMP `/stable/`
  endpoints in 4-year chunks.
- **ThetaData chains for XLF, SLV, EWZ** (2018-08→2026-07-28, 15:30 ET NBBO,
  max DTE 70 — same pipeline and format as SPY/QQQ/IWM): for the H5 cash-secured-put
  wheel hypothesis, the only premium-selling family fundable at $10k. Quality audit
  appended after the pull completes.

## 9. Contamination disclosure (critical for research design)

The 2018–2026 chains for all three ETFs were used exhaustively by the prior
short-call-spread study (thousands of configurations, including the 2025+ window).
That mining was for a **different strategy family** (short call verticals), but
regime definitions derived there (rebound windows, vol terciles, sector-correlation
thresholds) are known to this researcher and partially overlap with candidate long-side
signals. Mitigations used in this study:
1. Signal hypotheses are pre-validated on 2000–2017 price data (never used before).
2. The options-era partitions are frozen before any experiment: train 2018-08→2022-12,
   validation 2023-01→2024-12, holdout 2025-01→2026-07 (single use, after freeze).
3. The holdout's prior exposure to spread-family mining is disclosed in every report;
   a clean forward paper-trading period is mandatory regardless of holdout outcome.
