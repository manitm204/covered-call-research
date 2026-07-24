# Phase A: aggression frontier on the frozen gate (exploratory)

Run 2026-07-24. Gate frozen at final hypothesis: RSI(14)>70 AND >=2 of
{IV>RV20, sector_avg_corr_20<0.45, absorption_chg_20d<0}; weekly entries,
30 DTE, hold to expiry, cap-2 sizing. Sweep: 9 deltas (0.08-0.35) x 6 widths
($2-$15) x {base, conservative} fills = 108 configs, all n=53 trades
(same 53 gated entry dates; delta/width don't change which dates fire).
Driver: scripts/phaseA_aggression_frontier.py ->
results/ablation_full/phaseA_frontier.{jsonl,parquet}.

Metrics: per-trade mean $/spread, block-bootstrap trade CI, episode-bootstrap
CI (~27 episodes, 30-day gap clustering — the honest one), win rate, worst,
CVaR5 (mean of worst 5%), mean credit, mean collateral-at-risk, return on
risk % (mean pnl / mean max loss, per ~30d trade).

## Findings

1. **Aggression dies above 0.20 delta.** 0.25/0.30/0.35 delta: means mostly
   negative, all episode CIs span zero (many trade CIs too), win rate falls
   58-74%, CVaR5 -300 to -900. The premium bought with higher delta is fully
   paid back in losses. This mirrors the unconditional sweep's monotone-worse-
   in-delta result — the gate does not rescue high delta.

2. **The edge lives at 0.08-0.20 delta; the sweet band is 0.12-0.18.**
   Mean $/spread peaks around 0.15-0.20 at wide widths, but episode CIs are
   tightest at 0.08-0.15.

3. **Width scales dollars sub-linearly in risk.** Total P&L rises with width
   at every delta, but return-on-risk peaks at $2-$8 wide and declines at
   $10-$15 (collateral grows faster than credit). CVaR grows with width.

4. **Return on collateral peaks at 0.15 delta / $2-$5 wide: ~8.1%/trade
   (~30 days deployed), base fills; 6.5-7.3% conservative.** The committed
   0.15/$8 sits at 6.2%.

5. Notable frontier cells (base fills; conservative in parens):

   | profile | cell | mean$ | episode CI | RoR% | worst | CVaR5 |
   |---|---|---|---|---|---|---|
   | max safety | 0.08 / $15 | 35.7 (33.0) | [28.2, 41.9] | 2.7 | -127 | -34 |
   | best risk-adjusted | 0.15 / $5 | 35.1 (32.1) | [17.9, 48.8] | 8.1 | -347 | -198 |
   | committed hypothesis | 0.15 / $8 | 46.3 (43.3) | [9.6, 70.3] | 6.2 | -787 | -327 |
   | best total, still robust | 0.18 / $8 | 53.4 (50.4) | [22.8, 79.4] | 7.3 | -590 | -289 |
   | max premium that survives | 0.20 / $10 | 58.3 (55.2) | [19.1, 93.8] | 6.7 | -743 | -397 |

   The 0.08-delta row is remarkable: 98% win, worst trade -$127 to -$161,
   episode CI entirely positive with the tightest bounds in the grid —
   a genuinely conservative variant that still clears T-bill-plus.

6. **Conservative fills subtract a near-constant ~$3/spread** at every cell
   (tight spreads at low delta); no cell flips sign from base to conservative
   within the 0.08-0.20 band. Fill assumptions are not driving the result.

## Honest read

All 108 cells reuse the same 53 entry dates / ~27 episodes, so cherry-picking
the single best cell (0.18/$8) is mining; differences inside the 0.12-0.18 x
$5-$10 block are within each other's CIs. The robust, structure-level
conclusions are: (a) delta <= 0.20 hard ceiling, (b) width buys total P&L at
declining risk-efficiency, (c) 0.15 +/- 0.03 delta and $5-$8 width is the
efficient region, (d) an ultra-conservative 0.08-delta variant exists with a
categorically smaller tail. Whatever cell goes into pre-registration must be
picked once, on judgment, and frozen.
