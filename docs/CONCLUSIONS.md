# Research conclusions — framework phase (2026-07-22)

Status at the end of Phase 5. **No licensed market data has been ingested; nothing in
this document is a claim about the real-world profitability of the XSP bear call
spread strategy.** These are the conclusions that CAN be drawn honestly, the go/no-go
criteria for the real study, and the standing risks to the strategy thesis.

## 1. What is established

**The research machine works and is verified.** 285 tests cover pricing math against
hand-computed values, ledger accounting to the penny, arbitrage-band invariants,
purge/embargo correctness, prefix-consistency of every feature, and determinism of
every run. The full path — ingest → validate → backtest → features → experiments →
walk-forward models → calibration → robustness → guarded final test — runs end-to-end
from the CLI.

**The guardrails demonstrably guard.** Three deliberate self-tests:
1. A feature computed with a full-sample statistic is caught by the leakage checker.
2. On synthetic GBM data (no true signal), the base-rate model beats every fitted
   model out-of-sample, and adding features monotonically worsens fitted models.
3. Nested-tuned XGBoost on the same no-signal data selects the simplest grid point,
   ties the base rate, and is refused admission by the complexity rule.

A framework that cannot produce false positives on noise is the prerequisite for
trusting any positive result later. That prerequisite is now met.

**Execution-cost sensitivity is first-class.** All results are produced under
optimistic/base/conservative fills with gross and net separated; even on synthetic
data, fees consumed ~14% of gross wins at 1-lot scale — the cost drag is material at
XSP size and will be a, possibly the, deciding factor.

## 2. What is NOT established

- Whether the spread has positive standalone expected value after costs (question 1
  of the brief) — requires licensed XSP/SPX options history.
- Whether the overlay improves total-portfolio risk-adjusted performance (question 2).
- Whether any feature family predicts spread outcomes.
- Anything about regime behavior (2018 selloff, COVID, 2022 bear, rebounds).

## 3. Go/no-go criteria for the real study (pre-registered here)

1. **Data gate**: validation report on ingested data must show no ERROR-level issues
   in the traded DTE range; pre-2022 XSP (if used at all) only via the labeled
   SPX/10 proxy, reported separately.
2. **Strategy gate (question 1)**: mechanical baseline expectancy per spread, net,
   under the CONSERVATIVE execution scenario, with a block-bootstrap 95% CI that
   excludes zero — else the standalone strategy is not established, whatever the
   mean says. Compare against T-bill interest on the same collateral: the spread
   must beat doing nothing with the cash.
3. **Model gate**: any filter/model must beat (a) the always-trade baseline and
   (b) the base-rate/logistic benchmarks, on purged walk-forward, after costs, and
   survive the family ablations without depending on a single family's presence.
4. **Final test**: one evaluation of the frozen configuration on 2025+, recorded by
   the usage guard. If it fails there, the answer is no — not "iterate and retry".

## 4. Standing risks to the strategy thesis (flagged early, monitor throughout)

1. **Rebound exposure is structural.** A bear call spread caps its benefit in a
   selloff but keeps losing in the rip back up; the 2020-style V-rebound is the
   expected worst case. Stress reporting treats rebound windows as first-class.
2. **Cost drag at XSP scale.** ~$2.50 round-turn per spread against credits of
   ~$50-100 per spread is a 3-5% haircut before slippage; conservative-fill results
   govern.
3. **Cash interest competes.** In high-rate regimes, T-bills on the collateral may
   dominate the spread's expectancy; the framework separates interest from trading
   P&L everywhere so this comparison cannot be fudged.
4. **Index call skew is thin.** Selling 0.15-delta index calls collects less vol
   premium than the put side; the variance-risk-premium literature is mostly a
   put-side phenomenon. The thesis needs the data to prove the call side pays.
5. **Small sample.** Monthly entries over 7 years ≈ 84 independent-ish trades;
   CIs will be wide, and the framework reports them rather than hiding them.

## 5. Next actions

1. License data (ThetaData / Polygon / Cboe DataShop / ORATS) — the only blocker.
2. `xsp ingest` + `xsp validate-data`; fix or scope out data failing the gate.
3. Run the pre-registered experiment ladder (mechanical baseline first) under all
   three execution scenarios; publish gross/net, CI'd expectancy, overlay panel,
   stress windows.
4. Only then: features → filters → models, in the mandated complexity order.

---

# Real-data results — SPY (2026-07-23)

Instrument switched to SPY by user decision (American exercise, physical
settlement, dividend assignment hazard modeled). Data: ThetaData 15:30 ET NBBO
snapshots, 2018-08-01 → 2026-07-22, 1,990 sessions, 8.98M quotes, underlying via
put-call-parity implied (validated against known history; stock tier blocked).

## Data gate: PASS (with noted caveats)

`validate-data` over all 96 monthly files: 1,213 crossed quotes (0.014% of rows,
engine rejects them at selection), 748 locked (rejected likewise), 892k zero-bid
rows (deep OTM, untradable shorts — excluded by min_short_bid), 7 session gaps
(all holiday weeks). No structural failures in the traded DTE range.

## Strategy gate (question 1): FAIL — the unconditional strategy loses money

Pre-registered mechanical baseline (0.15Δ short, $5 wide, ~30 DTE, monthly,
50% profit target / 2x stop), 96 trades over 8 years:

| scenario     | net expectancy/spread | block-bootstrap 95% CI | win rate | profit factor |
|--------------|----------------------|------------------------|----------|---------------|
| optimistic   | −$18.40              | [−$29.78, −$8.71]      | 55%      | 0.50          |
| base         | −$21.17              | [−$32.91, −$10.98]     | 53%      | 0.45          |
| conservative | −$24.46              | [−$35.51, −$15.09]     | 52%      | 0.38          |

The CI excludes zero in the WRONG direction under every scenario, including
optimistic mid fills — this is not an execution-cost artifact; the gross edge
itself is negative. Per-year: negative in 8 of 9 years; only 2022 (the bear
year) was positive (+$44). Worst year 2020 (−$842, 17% win rate) — the V-rebound,
exactly the structural risk flagged in §4.1. Standing risk §4.4 (thin call skew)
is confirmed: median credit $0.57 on a $5 width does not pay for the drift.
Early assignment never triggered (stops fire long before shorts go deep ITM);
interest on collateral (+$20.7k on $100k) dwarfs trading P&L (−$2.0k base) —
standing risk §4.3 realized.

**Conclusion (honest, as pre-registered): selling unconditional 15-delta SPY
call spreads at monthly cadence has significantly negative expectancy after
costs. "Do nothing and hold T-bills" dominates. The remaining research question
is question 3: can a conditioning model identify the subset of regimes (e.g.
2022-style downtrends) where the trade pays — and beat the always-trade AND
never-trade baselines on purged walk-forward? The never-trade baseline is now
the one to beat.**

## Filter & model gates (run 2026-07-23, real data): FAIL — study concludes NO-GO

**VIX<20 filter (pre-registered experiment 3).** Improves expectancy but does
not rescue it: base −$9.14/spread (95% CI [−$21.69, +$1.32]), conservative
−$13.40 (CI [−$26.01, −$2.88]), n=62. Calm-regime selection removes some
losses; the conservative CI still excludes zero on the wrong side.

**Label generation (417 weekly hold-to-expiry trades, 2018–2026).** American
mechanics fired for real: 7 early assignments on ex-div eves, 4 intrinsic
expiry closes. Win rate 77% but expectancy −$26.67/spread (CI [−$48.89,
−$6.26]) — tail losses dominate, the classic short-premium profile.

**Model gate (purged walk-forward, 4 folds, 5-day embargo, 2025+ excluded as
the untouched final window; 263 OOS samples).** On BOTH pre-registered labels:

| model (label_expire_itm) | Brier | ECE |
|---|---|---|
| base rate | **0.1757** | **0.028** |
| logistic | 0.2983 | 0.296 |
| logistic L1 | 0.3319 | 0.330 |
| shallow tree | 0.2410 | 0.210 |
| nested-tuned XGBoost | 0.1924 | 0.107 |

Boosting refused admission (rule: must beat every simpler benchmark; it does
not). Same ordering on label_touch (base rate 0.2295; every fitted model
worse; boosting 0.2475, refused). The family ablation is monotone in the
wrong direction — baseline-only (3 features) 0.2043, all 54 features 0.2983,
and NO feature set beats the base rate. This is precisely the no-signal
signature the framework was validated to produce on synthetic GBM noise.

**Final test: NOT RUN, by protocol.** The 2025+ window remains untouched. The
final test exists to confirm a configuration that passed the gates; nothing
passed. Running it anyway would be result-shopping.

**Overall verdict (question-by-question):**
1. Standalone expectancy after costs: NO — significantly negative in all
   execution scenarios.
2. Conditioning/filters: VIX filtering shrinks losses but stays ≤ 0;
   no admissible model finds a tradable subset on 39 daily + 15 surface
   features across 8 years including two bear markets.
3. The honest recommendation from this study: do not trade this strategy.
   The collateral earns more in T-bills. A future study could revisit with
   (a) put-side spreads (where the variance risk premium actually lives),
   (b) intraday/0DTE structures, or (c) different underlyings — each would
   need its own pre-registration under this same framework.

---

## Addendum (2026-07-23/24): exhaustive structure sweep + exploratory regime study

Everything in this addendum is EXPLORATORY — thresholds and combinations were
chosen while looking at the same 2018–2026 data (including the 2025+ window
the original protocol reserved). Nothing here upgrades the NO-GO verdict on
the unconditional strategy; the regime rule at the end is a pre-registrable
HYPOTHESIS, not a validated strategy.

### Full 2,000-config structure ablation (`scripts/full_ablation_sweep.py`)

8 short-deltas (0.05–0.30) x 5 widths ($1–$10) x 5 DTEs (7–45) x
{weekly, monthly} x 5 exit styles, base scenario, real SPY data
(results/ablation_full/SUMMARY.md). Verdict: the NO-GO is global, not a
bad-parameter artifact. 10/2000 configs positive (4 CIs above zero vs ~50
expected by chance under a no-edge null); expectancy worsens monotonically in
BOTH delta (−$6 mean at 0.05Δ → −$28 at 0.30Δ) and width (−$10 at $1 →
−$26 at $10). The only "winners" are 0.05Δ/$1-wide configs earning ~$5–8 per
spread — sub-T-bill, sitting on the $0.05 min-bid floor, fill-model artifacts.

### Regime conditioning (the interesting part)

Conditioning the 417-trade weekly hold-to-expiry panel on entry-date features
(and confirming with gated engine runs, base AND conservative scenarios):

- Bad regimes: IV<RV (−$97/spread), deep below 200d MA (−$89), post-selloff
  (−$87), rising absorption ratio (−$85), VIX>30 (−$80), backwardated VIX
  term structure (−$54). Selling "fat premium into fear" is where the
  catastrophic losses live.
- Good regimes: RSI(14)>70 (+$12), falling absorption (+$20 tercile), sector
  average pairwise correlation <0.45 (+$13), IV>RV as a veto. Calm,
  overbought, internally-diversified grinds are the only profitable habitat.
- Signal audit on the locked structure (0.15Δ/$8-wide/30 DTE/weekly/hold,
  unfiltered −$34/spread): RSI>70 is the one orthogonal, all-subperiod-
  consistent signal; {absorption, sector-corr, SPY-RSP-corr} form one
  redundant cluster (phi ~0.4–0.5); {VIX, 200d MA, IV-RV} form a second
  veto-ish cluster; RSP/SPY breadth EMA and QQQ-beta are standalone noise.
- Structure under good gates inverts the unconditional findings: edge scales
  WITH width, 30 DTE is the sweet spot, and hold-to-expiry beats
  profit-target/stop management (the gate is the risk management).
- Overlap honesty: trades cluster into ~25 regime episodes over 8 years
  (~3/yr). The strict gate survives episode-level scrutiny (23/25 episodes
  positive); k-of-n voting's apparent extra edge was mostly stacking size on
  the same episodes and degrades under a 2-position cap.

### The resulting hypothesis (fully specified, ready to pre-register)

Enter weekly when RSI(14)>70 AND at least 2 of {IV>RV20, sector_avg_corr_20
< 0.45, absorption_chg_20d < 0}; short 0.15Δ SPY call spread, $8 wide,
~30 DTE; hold to expiry; max 2 open positions. On 2018–2026 SPY (base
fills): n=53 (6.6/yr), +$46.27/spread, bootstrap 95% CI [+22.94, +72.83],
92% win, worst trade −$787, total +$2,452. Conservative fills degrade the
strict variant by only ~$3/spread. Effective sample ~25 independent
episodes. Every gate variant shares the same −$787 worst trade (2022): the
tail is irreducible at this width.

Required next step before any capital: pre-register this exact rule and test
on unmined data — other underlyings (QQQ/IWM/XSP chains) or forward paper
trading. The thresholds must not move during that test.

## Addendum (2026-07-24, second session): structure frontier, surface study, portfolio framing

All exploratory (same 53 gated entries / ~26 episodes as the hypothesis above).

### Phase A — aggression frontier (scripts/phaseA_aggression_frontier.py)

108 configs (delta 0.08–0.35 × width $2–$15 × base/conservative) on the frozen
gate, cap-2. Verdict:

- Delta has a hard ceiling at 0.20: at 0.25–0.35 the means go negative/noisy,
  win rate falls to 58–74%, and every episode CI spans zero. The gate does not
  rescue high delta.
- Return on collateral peaks at 0.15Δ/$2–$5 wide (~8.1%/trade base fills);
  width buys total dollars at declining risk-efficiency (collateral grows
  faster than credit).
- An ultra-conservative variant exists: 0.08Δ/$15 — 98% win, worst trade −$127,
  episode CI [+28, +42], but only ~2.7%/trade on collateral.
- The efficient region is 0.15±0.03Δ × $5–$8; differences inside it are within
  each other's CIs (picking the single best cell would be mining).

### Phase B — surface-relative strike selection (phaseB_surface_diagnostic.py, phaseB_engine_ab.py)

Question: can SVI-smile "mispricing" pick a better short strike than fixed
0.15Δ? Answer: NO, and the mispricing story is backwards.

- Strikes rich vs the fitted smile UNDERPERFORM when sold (within-date partial
  correlation −0.20 controlling delta; sell-the-richest rule loses $4–8/spread
  in every test). Local richness is information, not free premium.
- The mirror rule (sell the cheapest strike in the 0.10–0.20 band) showed
  +$12/spread out-of-engine but degraded to +$5.3 with episode CI [−4.8, +16.0]
  in the full engine — and to $0.0 under a different (equally reasonable)
  smile-fit implementation. Sign stable, magnitude fragile → not
  pre-registrable. Fixed delta stands. XGBoost strike ranker (Phase C) skipped:
  the signal it would learn is sub-vol-point, wrong-signed vs intuition, and
  implementation-dependent.

### Portfolio framing (scripts/portfolio_comparison.py)

$100k, 2018-08→2026-07, base fills, dividends reinvested:

| portfolio | CAGR | vol | Sharpe | maxDD | beta |
|---|---|---|---|---|---|
| T-bills | 2.71% | 0.2% | — | ~0% | 0.00 |
| SPY (TR) | 14.97% | 18.5% | 0.70 | −34.7% | 1.00 |
| strategy standalone (cash-collateralized) | 2.74% | 0.6% | 0.06 | −0.8% | −0.008 |
| SPY + margin overlay | 15.09% | 18.2% | 0.72 | −34.4% | 0.99 |

- KEY: cash collateral forfeits T-bill interest while spreads are open —
  $2,142 of the $2,453 trading P&L over 8 years. Standalone cash-collateralized
  = T-bills + 3bp/yr, i.e. not worth running. The strategy only makes economic
  sense as a MARGIN OVERLAY on an existing SPY position (+$311/yr per 2-spread
  unit, alpha +0.23%/yr, beta −0.008, corr −0.26 — genuinely diversifying;
  scales linearly in contracts, as does the tail).

### Status

Exploratory phase CLOSED. The only remaining step that produces new evidence
is the pre-registered test of the frozen rule on unmined data (QQQ/IWM
chains or forward paper trading).

## Pre-registered QQQ out-of-sample test: NO-GO (2026-07-24)

Protocol: docs/PREREGISTRATION_QQQ.md (frozen and committed before the QQQ
data pull; single-use runner). Rule: RSI14(NDX)>70 + >=2 of {VXN>RV20(NDX),
sector_avg_corr_20<0.45, absorption_chg_20d<0}; 0.15Δ/$8/30 DTE/weekly/hold/
cap-2 on QQQ, 2018-08 -> 2026-07 (1,977 sessions, data notes in
results/qqq_prereg/DATA_NOTES.md).

Result (base fills): n=50, mean −$35.99/spread, bootstrap 95% CI
[−91.17, +37.92], episode CI [−108.18, +31.57] (27 episodes), win 80%,
worst −$900, total −$1,800. Conservative: mean −$40.26, total −$2,013.
Yearly means: +41.5 (2018), −11.4 (2019), −106.2 (2020), +72.1 (2021),
+80.5 (2022), −20.5 (2023), −77.2 (2024), +57.8 (2025), −441.1 (2026).
Verdict per pre-declared criteria: NO-GO (primary failed, both mean-based
secondaries failed; only the win-rate secondary passed — the short-premium
base rate, which the tails erase).

### What this means

The SPY hypothesis (+$46/spread, CI [+23, +73]) failed its first and only
out-of-sample test. The honest conclusion: the mined SPY edge does not
generalize to the closest comparable underlying, which is exactly the
signature of selection bias over ~26 effective episodes rather than real,
transferable market structure. The regime story ("sell calls into calm
overbought diversified grinds") is not supported off-instrument.

Per protocol, no threshold may be tuned in response, and any new hypothesis
(e.g. SPY-specific microstructure arguments, different vol index mapping)
would require fresh pre-registration on data not yet touched. The study's
final state: framework validated, unconditional selling NO-GO, conditional
SPY result NOT confirmed out-of-sample. Do not deploy capital on this rule.

## Addendum (2026-07-25): post-NO-GO QQQ exploration — the two mined rules, side by side

FRAMING, READ FIRST: everything in this section was HAND-PICKED from the data
it is evaluated on. After the pre-registered QQQ test failed, we deliberately
mined QQQ (labeled exploration): a 52-feature regime scan, a 10-signal
ablation with phi-redundancy analysis, an exhaustive sweep of ALL 1,023
signal combinations, and a 128-config structure ablation inside the chosen
gate. The QQQ rule below is the survivor of roughly 1,300 engine runs on
~27 effective episodes; the SPY rule survived a comparable in-sample gauntlet
and then FAILED its only out-of-sample test (QQQ, NO-GO). The backtest
numbers are arithmetic on selected data — 100%-win cells and positive worst
trades are selection artifacts, not properties of any real strategy. Neither
rule is validated. Neither should trade real capital as-is.

### The two rules

|                    | SPY                                   | QQQ                                        |
|--------------------|---------------------------------------|--------------------------------------------|
| Trigger            | RSI(14) > 70 (short-term overbought)  | ret_120d > 15% AND dist_ma200 > +10% (mature long-run trend) |
| Confirmation       | >=2 of {IV>RV20, sector_corr<0.45, absorption falling} | sector_avg_corr_20 < 0.45 |
| Delta              | 0.15 (efficient region 0.12-0.18)     | 0.12 (mean keeps rising to 0.30 but tails/CIs degrade; 0.08-0.12 is the robust band) |
| Width              | $8 (efficient $5-$8; edge scales with width) | $15-$20 (width = capital knob, RoR flat) |
| DTE                | ~30 (25-35) — clear sweet spot        | ~30 (25-35) — clear sweet spot (14/21 DTE fail, 45 comparable) |
| Entries / sizing   | weekly, hold to expiry, max 2 open, 1 contract | same |
| In-sample result   | n=53, +$46.27/spread, CI [+22.9,+72.8], 92% win, worst -$787 | n=40, +$91.33/spread (0.12/$20), CI [+76.7,+106.5], 97% win, worst -$17 |
| Trades/yr          | ~6.6                                  | ~5.0                                       |
| Out-of-sample      | FAILED (QQQ pre-registered test: -$36/spread, NO-GO) | NEVER TESTED |

### What transfers between instruments and what does not

- The DANGER regimes are universal: rising absorption, high sector/index
  correlations, below the 200d MA, IV<RV, high vol index. Both instruments
  agree, with similar magnitudes. "Never sell into fear, lockstep markets,
  or broken trends" is the study's most robust finding.
- The OPPORTUNITY trigger is instrument-personality: SPY sells short-term
  overbought (mean-reversion index); RSI is completely flat on QQQ
  (momentum index), which instead conditions on OLD, extended rallies.
- The confirmation layer (diversified internals: sector-corr, absorption)
  is shared; on QQQ, sector_corr<0.45 alone carries it, and phi-redundant
  variants (SPY-IWM decorrelation) add fragility, not information.
- Structure agrees on 30 DTE and hold-to-expiry; QQQ tolerates (in-sample)
  higher delta than SPY's hard 0.20 ceiling, but the robust band is lower
  delta on both.

### Provenance

QQQ exploration artifacts: results/qqq_prereg/{POSTMORTEM.md,
regime_scan_panel.parquet, signal_phi_matrix.json, exhaustive_combos.parquet,
regime_structure_qqq.parquet}; scripts/qqq_{postmortem,regime_scan,
signal_ablation,exhaustive_combos,structure_ablation}.py. The exhaustive
sweep's top-20 all share one shape (mature trend + diversified internals);
simplifying any champion rule collapses its edge, the classic signature of
threshold-sculpted selection.

### If either rule is ever taken forward

Freeze it exactly as specified above and run ONE pre-registered test on data
neither rule has touched: IWM chains (pull available), XSP/SPX (needs index-
options data tier), or forward paper trading. SPY's rule already spent its
QQQ shot; QQQ's rule has never faced unmined data. Expected outcome under
the null (mined noise): ~$0/spread. The SPY->QQQ NO-GO is the base-rate
warning for how these tests tend to go.

## Pre-registered IWM test (2026-07-25): BOTH mined rules — verdict WEAK / WEAK

Protocol: docs/PREREGISTRATION_IWM.md, committed (`52826a5`) before any IWM
options data was downloaded. Two hypotheses on one unmined pull, multiplicity
declared in advance (family-wise false-positive ~10% at nominal 5% per test).
Single-use runner executed once; marker written. Data: 1,948 of 1,990
sessions (46 persistent vendor gaps, documented with expiry-date patches in
results/iwm_prereg/DATA_NOTES.md). Recorded verbatim from
results/iwm_prereg/verdict.json — no tuning, no re-runs.

### H1 — the SPY rule on IWM (RSI trigger, 0.15Δ/$3/30DTE): **WEAK**

- Base fills: n=33, mean **+$10.86**/spread, moving-block CI
  **[−36.63, +34.93]** (primary endpoint FAILED — CI spans zero), episode CI
  [−18.45, +34.34] (23 episodes), win 90.9%, worst **−$315.75**, total +$358.
- Conservative fills: mean +$7.35 (> 0, secondary met).
- Yearly means positive every year except 2020 (−$70.57).

### H2 — the mined QQQ rule on IWM (trend trigger, 0.12Δ/$6/30DTE): **WEAK**

- Base fills: n=24, mean **+$14.89**/spread, moving-block CI
  **[−10.59, +41.04]** (primary FAILED), episode CI [−39.13, +41.53] (only
  **10 episodes**), win 95.8%, worst **−$514.25**, total +$357.
- Conservative fills: mean +$11.42 (> 0, secondary met).
- Fired only in 2020–2021 and 2024–2026 (IWM rarely satisfies the
  mature-trend gate); 2021 mean −$53.25.

### Reading (written per the frozen protocol)

WEAK = inconclusive and explicitly NOT a license to tune. Both rules were
net positive on data they had never seen — the first time either has
survived contact with unmined data — but each result is one large loss away
from breakeven (H1's −$316 in 2020; H2's −$514 in 2021), which is exactly
the short-vol profile where 30 wins hide in a CI that spans zero. Per the
protocol neither WEAK may be promoted; the honest summary is "consistent
with a small edge, and consistent with zero." Remaining unmined venues for
a decisive test: XSP/SPX chains (needs index data tier) or forward paper
trading; IWM is now burned for both rules.

## Unified Rule v2 (2026-07-26): veto + per-ETF confirmations

Successor to every prior mined gate. Unlike earlier rules, each leg here is
supported by an INDEPENDENT 20-year price study (Spearman ICs on 2005+
weekly data in a separate project: mean-return, breach, tail-size, and
vol-normalized-return targets) and only then checked on our 2018-2026
options panels. Still not validated forward — see caveats.

### The veto (all ETFs — when NOT to sell)

No entry if ANY of the following, computed on the underlying index
(SPX/NDX/RUT) and own vol index (VIX/VXN/RVX), lagged >= 1 session:

1. **Rebound window**: index closed >= 10% below its running high at any
   point in the past 60 sessions. (The big one-month upside tails that
   breach short calls are rebound rallies out of drawdowns — 20-yr breach
   ICs; confirmed in our panels: rebound windows carry 58%/72%/93% of all
   losses on SPY/QQQ/IWM.)
2. **Below trend**: index under its 200-session MA.
3. **High vol**: own vol index in its top tercile (2018-08..2026-07
   terciles: VIX >= ~20.8, VXN >= ~25.8, RVX >= ~26.4). Rationale is loss
   SEVERITY, not frequency: high-vol breaches lose ~2x more while credit
   rises only ~30%.

Veto split test (engine cap-2, weekly, base fills): veto-active trades
lost -$64.71/spread on SPY (CI entirely negative, total -$6,342) and
-$49.14 on QQQ (total -$5,700); veto-clear arms: SPY +$6.72, QQQ -$12.37,
IWM +$18.12 (CI [+0.7,+31.5], the only standalone-positive arm).

### Confirmations (when TO sell, within veto-clear windows)

- **SPY**: sector_corr_60 < 0.45 AND ret_84d > 10%.
  n=27, +$57.45/spread, epCI [+38.1,+78.2], win 89%, worst -$87.
- **QQQ**: sector_corr_60 < 0.45 AND ret_84d > 10%; absorption_chg_20d < 0
  is a useful additional check (with it: n=22, +$61.49, epCI [+34.2,+80.2],
  worst -$125; without: n=28, +$43.71 but epCI spans zero).
- **IWM**: sector_corr_60 < 0.45 only. n=26, +$27.69, epCI [+17.3,+33.2],
  win 96%, worst -$46. RVX low is a useful additional check but largely
  covered by the veto's vol leg. A trend condition is REDUNDANT on IWM:
  every veto-clear + seccorr session already has ret_84d > +6% (the veto
  clears only 14% of IWM sessions and its rebound/MA200 legs force an
  uptrend); adding any trend threshold only shrinks n and total P&L.

sector_corr_60 = mean pairwise 60-session correlation of the nine SPDR
sector ETFs (the 60d variant, per the price study; stronger than our
original 20d). ret_84d = 84-session index return ("4-month trend"). The
+10% cutoff sits at the ~70th percentile of veto-clear r84 on both SPX and
RUT (53rd on NDX) — frequency-fair across instruments, not tuned.

### Structure (unchanged from the frozen tests)

~0.15 delta short call, width ~1-1.3% of spot ($8 SPY, $8 QQQ, $3 IWM; the
QQQ prereg width), ~30 DTE, weekly entry checks, hold to expiry, max 2
open, 1 contract per entry, American/physical with dividend calendars.
Expect roughly 3-4 entries/yr/ETF; credits ~$70-80 (SPY/QQQ), ~$30 (IWM).

### Why this one is different from (and replaces) the earlier mined rules

The SPY RSI rule (failed QQQ prereg), the QQQ mature-trend rule and the
IWM beta rule (both mined) are RETIRED. The v2 legs were each selected by
the external study BEFORE being tested here: rebound danger (breach ICs),
sector-corr and trend (top vol-normalized ICs), vol veto (severity
decomposition), RSI demoted (weak after normalization; on our panels it
only shrinks n), beta dropped (dead on 20-yr tail targets; its IWM
backtest showing was a small-sample artifact). IC bridge test: all 21
factor rows (7 factors x 3 ETFs) had the price-study "good side" beat the
bad side in mean $/spread on the real options panels.

### Caveats (unchanged in kind)

All three chain datasets are burned; the confirmation-layer ablation was
~128 more comparisons on them, and the veto's rebound definition (10%/60d)
was informed by the same panels it was then tested on. The 100%-win cells
in the IWM ablation are selection artifacts (n=15-20 subsets that dodge
every loser). Numbers above are optimistic by construction. The ONLY
remaining honest validation is forward: paper-trade this exact rule set,
frozen verbatim, on all three ETFs (and XSP if the index data tier is ever
added). Provenance: scripts/veto_split_test.py, scripts/ic_bridge_test.py,
scripts/confirm_ablation.py, results/iwm_prereg/{veto_split,ic_bridge,
confirm_ablation,iwm_trend_threshold,three_checks}.log and
confirm_ablation_*.jsonl.
