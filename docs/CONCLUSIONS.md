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
