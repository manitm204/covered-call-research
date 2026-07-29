# Research cycle 3 — user-directed ideas: margins, deciles, single names, interaction trees
*2026-07-30. Ideas from the user, tested under the standing gates; ledger rows in
`experiment_log.csv`. Companion to `final_report.md` and `research_cycle2.md`.*

## Ideas taken up this cycle

1. **Sector profit-margin trends → trade sector ETFs** (user idea). Built from a
   new pull of 48,801 statement-quarters (501 current S&P members, FMP, with
   filing-date point-in-time discipline; survivorship caveat disclosed).
   Monthly panel 2003→2026 of per-sector trailing-4Q operating margin medians,
   YoY margin trend, and margin-improvement breadth, matched to sector-SPDR
   forward relative returns.
   **Result: dead.** All 12 marginal IC tests fail in BOTH eras (2005–17 and
   2018–24; |IC| ≤ 0.05, |t| ≤ 0.7); the pooled-panel depth-3 interaction tree
   fails OOF (Brier 0.2526 vs base 0.2498) with incoherent in-sample splits; the
   market-level aggregate variant (margin trend → SPY forward returns) is
   slightly *negative* in both eras. Reading: quarterly margins are stale, widely
   watched, and priced by filing time.
2. **Interaction trees aimed at strategy-relevant labels** (user's core method).
   Whipsaw-prediction trees for TG-CER's known failure mode (label: MA200-band
   flip(s) in the next 21 sessions): both label variants "pass" the Brier
   comparison by probability shading but **no leaf ever predicts the event**, and
   the top split is a narrow consumer-sentiment window — episode memorization.
   The mechanical driver of whipsaw is simply distance-to-MA, which the existing
   hysteresis band already encodes. Closed as practically useless.
3. **Cross-ETF comparisons / momentum rotation** (user idea, "random ETFs").
   22-ETF universe (equities, bonds, credit, gold, silver, oil, dollar, EM, REITs,
   biotech, sectors), 12-1 momentum, top-3 above their MA200, 2007–2017:
   +1.02%/21d vs +0.61% universe control — right direction, CI90 [−0.53, +1.25]
   spans zero → **fail** at the gate.
4. **SPY internal deciles (large vs small inside the index)** (user idea):
   deferred — the coarse proxies already in the feature set (RSP/SPY equal-weight
   ratio, IWM/SPY differentials) were IC-screened in cycle 2 and were weak/
   unstable. A true decile study needs point-in-time constituent membership +
   500 price histories; queued as future work, expected value low given the
   proxy results.
5. **Options on single names inside SPY** (user idea): in progress. Ex-ante
   universe (price $15–65, S&P member, liquid options, dividend payer, four
   different sectors — chosen before any backtest): **PFE, VZ, BAC, DOW**.
   Chains downloading; the wheel engine gained an **earnings-avoidance rule**
   (skip any new short option spanning a scheduled report, from the on-disk
   earnings calendar). Test protocol when data lands: same wheel cell as the SLV
   pre-registration, train grid → only if a diversified 2-name ladder clears the
   T-bill+2%/yr bar with DD consistency does it advance to valid.

## Cycle-3 scorecard so far

22 new logged tests: **0 passes.** Combined with cycle 2 (13 tests) and the
original program, the running theme is stable: broad, popular, low-frequency
public data (fundamentals, macro, valuation, positioning, seasonality) carries
no monthly-horizon edge detectable at our sample sizes — and the honest gates
catch every seductive in-era pattern (eigen cluster, sentiment windows,
CPI splits). The only signals that have ever survived out-of-era checks remain:
1. MA200 left-tail collapse (drives TG-CER), and
2. deep-drawdown rebound right-tail (H1, marginal).

## Standing recommendation (unchanged)

Paper trade the brake-amended TG-CER + reduced-size three-sleeve combination;
no live capital. The single-name CSP ladder may add a fourth sleeve if it clears
its gates when the chains arrive.
