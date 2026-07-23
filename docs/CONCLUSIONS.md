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
