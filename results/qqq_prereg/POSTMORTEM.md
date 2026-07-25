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

## Structure ablation under the frozen gate (exploratory, 2026-07-24)

105 configs (delta 0.05-0.25 x width $5/$10/$15 x DTE 7/14/21/30/45), weekly,
hold, cap-2, base fills: scripts/qqq_structure_ablation.py ->
results/qqq_prereg/structure_ablation.{jsonl,parquet}.

- ZERO of 105 cells have an episode-level CI above zero (~2-3 expected by
  chance even under a no-edge null): the QQQ failure is not a structure
  problem. Expectancy is monotone-worse in delta (-$2.3 mean at 0.05 ->
  -$26.9 at 0.25) and worse with width -- the same signature as SPY's
  UNCONDITIONAL grid, i.e. the gate adds nothing structural on QQQ.
- Shorter DTE does not dodge the melt-up tail (7 DTE: every delta negative);
  30 DTE is the WORST tenor (-$32.7 avg) -- SPY's sweet spot inverted.
- The only ~positive corner is 0.05 delta / wide / 21-45 DTE (best cell
  +$13.5, ep CI [-3.0, +25.8]): pennies of credit, the classic fake-winner
  corner selection bias produces from 105 draws.

Conclusion: on QQQ, in the same regimes, short call spreads lose at every
setup. Reinforces the instrument-level story: NDX's right tail in these
regimes cannot be structured around with delta/width/DTE.

## Full regime scan on QQQ (exploratory/overfit, 2026-07-24)

scripts/qqq_regime_scan.py; panel = 414 unfiltered trades x 52 features
(results/qqq_prereg/regime_scan_panel.parquet).

- VETOES TRANSFER: absorption rising (hi tercile -$120), high sector-corr
  (-$100), below MA200 (off -$103), IV<RV (-$60), high VXN (-$64), high
  index-pair correlations (-$78..-$101) — all bad on QQQ exactly as on SPY.
  The "when NOT to sell" side is universal market structure.
- THE TRIGGER DOES NOT: RSI>70, SPY's load-bearing entry signal, is DEAD FLAT
  on QQQ (on -$47.5 vs off -$46.6). QQQ's only positive conditioners are
  LONG-horizon trend maturity: ret_120d / ret_60d / dist_ma200 top terciles
  (+$21..+$27). Short-term overbought means nothing when the index melts up.
- Even deliberately overfit composites can't reach SPY's number: best found
  (all vetoes + ret_120d>10%, cap-2) = n=56, +$32.45, trade CI [-10.0,+68.1],
  episode CI [-21.7,+68.3] — spans zero. SPY's mined rule on SPY: +$46.27,
  CI [+22.9,+72.8]. Mining QQQ with the same effort cannot manufacture a
  significant rule: the habitat itself is worse (2023/2024/2026 melt-up years
  are -$92/-$58/-$144 unfiltered; only 2018 and 2022 were sellable at all).

Bottom line: QQQ shares SPY's danger regimes but not its opportunity regime.
The strategy's edge (if any) is not "calm overbought grinds" generically —
on QQQ the closest analog is late-stage mature uptrends, and even that is
statistically nothing after honest error bars.
