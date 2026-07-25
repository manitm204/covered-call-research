# QQQ NO-GO postmortem (EXPLORATORY — verdict unchanged)

Script: scripts/qqq_postmortem.py. Regenerates the deterministic gated trade
list plus an unfiltered QQQ baseline for diagnosis. Nothing here modifies the
pre-registered verdict.

## Findings

1. **The gate did provide lift on QQQ — just nowhere near enough.**
   Unfiltered QQQ (weekly, 0.15Δ, ~30 DTE, hold): n=414, mean −$46.79/spread.
   Gated: −$35.99. Lift +$11 vs SPY's +$80 (−$34.4 → +$46.3).

2. **The regime dates were largely the SAME dates.** 41 of 50 QQQ gated
   entries share their entry date with SPY gated entries. On those shared
   dates: SPY +$39.51/spread, QQQ −$32.48 (P&L correlation 0.71). SPY lost on
   0 dates where QQQ won; QQQ lost on 5 dates where SPY won. This is an
   INSTRUMENT-level failure, not (primarily) a signal-mapping failure: the
   gate identified the same windows, and SPY's calls died worthless while NDX
   ripped through its short strikes.

3. **The mechanism: NDX's right tail in these regimes.** All 10 losing trades
   breached the short strike; losers' average entry→expiry NDX move was
   +7.0% (range +4.3% to +11.2%) vs −1.4% for winners. The 0.15Δ short strike
   sits ~5–6% OTM; calm/overbought/diversified regimes are precisely when
   mega-cap tech melts up 6–11% in a month. The premium (≈$1.0–1.3 on an
   effectively $10-wide spread — QQQ's $5 strike grid widened the nearest-to-
   $8 selection to $10) does not pay for that tail.

4. **The signals still ordered risk correctly on QQQ.** Vote-pattern means:
   all 3 confirmations on → +$5.24 (n=24, the only non-negative group);
   absorption off → −$101.94; VXN>RV off (admitted purely by S&P-specific
   votes) → −$163.19, the worst group. So sector-corr/absorption were not
   meaningless for QQQ — consensus still ranked outcomes — but even the best
   sub-regime was ~breakeven, not +$46.

5. **2026's −$441 yearly mean is one trade**: 2026-04-27 entry, 705/715,
   $1.03 credit, NDX +11.2% by expiry, full loss −$899.75 (n=2 trades in
   2026).

## Interpretation

The refined post-hoc story: "in calm/overbought/diversified regimes the S&P's
right tail is sellable at 0.15Δ; the Nasdaq's is not — same regimes, fatter
melt-up tail, insufficient premium." That is a coherent instrument-specific
hypothesis (and more precise than "the signals don't map"), but it is
POST-HOC: it was formulated after seeing the QQQ failure. It does not rescue
the SPY result; it narrows it. Remaining honest tests of the narrowed
hypothesis: XSP/SPX spreads (same index, different market) or forward paper
trading of the frozen SPY rule. QQQ data is burned for testing and usable
only for exploration.
