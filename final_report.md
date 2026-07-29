# Final report — Level-2 options strategy research for a $10,000 Fidelity account
*2026-07-29. All artifacts referenced here are in this repository; every engine run
is row-logged in `experiment_log.csv` (148 backtest runs + pre-validation entries).*

## 1. Mandate and constraint reality

Find a systematic options strategy with credible evidence of positive risk-adjusted
returns after realistic costs, executable in a **$10k Fidelity account with options
Level 1–2** (covered calls; long calls/puts; cash-secured puts — **no spreads, no
naked shorts, no margin**).

The capital constraint does most of the strategy selection by itself: 100-share
units of SPY (~$63k), QQQ (~$57k) or IWM (~$22k) are unfundable, so covered calls
and CSPs on the liquid big-three ETFs are out. What remains:
(a) **long calls/puts** on SPY/QQQ/IWM (data on hand, 2018–2026), or
(b) **CSP/covered-call wheels on sub-$100 ETFs** (XLF/SLV/EWZ chains newly pulled).

## 2. What was tested (hypothesis ledger)

| Hypothesis | Economic story | Pre-validation (2000–2017 prices, untouched by any options backtest) | Options-era outcome |
|---|---|---|---|
| H1 rebound calls | post-drawdown V-recoveries have fat right tails | narrow PASS (SPY deep-dd variant only: P(+10%/42d) 11.2% vs 4.3%, CI90 [+0.7,+17.8]) | positive but n=9 trades in train — too few; not promoted |
| H2 trend calls | above 200d MA the left tail collapses (SPY/QQQ CI90 excludes 0); a premium-capped call monetizes that shape while cash earns T-bill interest | PASS (SPY, QQQ) | **promoted — see §3** |
| H3 cheap convexity (IV<RV) | buy premium when IV underprices RV | **FAIL** — VRP stays ~4 vol pts negative even conditioned (fair per-instrument test: QQQ/VXN, IWM/RVX) | not pursued |
| H4 trend puts (control) | mirror of H2 | FAIL as predicted | control confirmed (SPY/QQQ negative) |
| H5 CSP wheel (XLF/SLV/EWZ) | put-side variance risk premium, fundable at $10k | n/a (options-native) | XLF train: +2–3% CAGR, Sharpe ≤0.54 — ~T-bills+2%; SLV/EWZ pending (§6) |

Also rejected by design: anything needing >70 DTE (no data), 0DTE (snapshot too
coarse), IWM anything (trend signal dead there — G1 and options-era agree).

## 3. The primary candidate: TG-CER (trend-gated convex equity replacement, QQQ)

Full rules in `strategy_spec.md` (frozen and committed at `e411fdd` **before** the
validation and holdout runs). In one paragraph: ~90% of the account sits in the
money-market core; while QQQ's adjusted close is >1% above its 200-session MA
(computed through the prior close), hold one QQQ call — expiry nearest 60 DTE in
[40,70], the highest-delta strike in [0.30,0.62] whose premium fits a 10%-of-equity
budget (prefer 0.50Δ) — roll at 21 DTE, sell everything when the close drops >1%
below the MA, never hold under 5 DTE, and apply a drawdown brake (half size below
85% of the 12-month equity high; no new entries below 75%).

**Why an edge is plausible:** it is mostly *structure*, not forecasting. The 200d-MA
state does not predict higher mean returns (2000–2017: CI spans zero) — it predicts
*fewer left-tail states* (P(−10% in 42d): QQQ 4.9% vs 10.4%, CI90 excludes 0). A
long call whose premium is capped at 10% of equity converts that distributional
shape into: full participation in trends, strictly bounded loss per cycle, and
T-bill interest on the rest. In high-rate regimes the interest pays ~40–100% of the
annual theta bill. Costs are structurally small (ATM QQQ spreads ~0.7%, ~9
single-leg 1-contract orders/year).

### Results by partition (frozen spec, brake included)

| Partition | Scenario | CAGR | Sharpe | maxDD | Trades | Final ($10k) |
|---|---|---|---|---|---|---|
| Train 2018-08–2022 | base | +10.0% | 0.56 | −19.4% | 36 | $15,201 |
| Train | conservative | +9.3% | 0.52 | −19.8% | 36 | $14,797 |
| Validation 2023–24 | base | **+37.0%** | 0.99 | −19.4% | 17 | $18,723 |
| Validation | conservative | +36.4% | 0.98 | −19.5% | 17 | $18,579 |
| **Holdout 2025–2026-07** | base | **−0.8%** | −0.09 | −20.4% | 8 | **$9,876** |
| Holdout | conservative | −1.1% | −0.10 | −20.5% | 8 | $9,830 |
| Holdout | stress (spread+3 ticks, 2× fees) | −1.4% | −0.11 | −20.6% | 8 | $9,780 |
| Continuous 2018–2026 (one account) | base | +15.2% | 0.66 | −19.4% | 30 | $30,906 |

Benchmarks, same spans: SPY B&H train CAGR 9.1% (maxDD −34%), validation 25.9%;
QQQ B&H continuous ≈ +18%/yr with −35% drawdowns; T-bills 0.0–5.3%/yr; unoptimized
always-ATM-call benchmark: QQQ +20%/yr at −36% DD (train), SPY +8.7%, IWM **−8.7%**
(long premium unconditionally bleeds — the engine is not a money printer).

### The holdout verdict — and why the recommendation is what it is

**The frozen spec fails its pre-registered holdout gate (G5).** Recorded verbatim:

- With the brake (the actual spec): net **negative** (−1.2% total). The 2025
  whipsaw (QQQ crashed and V-recovered inside the year, finishing +21%) drew the
  strategy down ~20%; the brake then halved/zeroed size exactly when the 2026
  trend resumed. A fresh-$10k account started 2025-01 ended at $9,876.
- Without the brake (diagnostic only — the engine initially omitted the brake;
  both holdout touches are disclosed): +34.5% total (+21.1% CAGR) but maxDD
  −31.7%, breaching the pre-set −30% line.

Per protocol, neither variant may be selected after seeing these results, and no
threshold may be tuned. The failure is *marginal and mechanism-specific* (the brake
interacts badly with V-recoveries; the core trend-gate + call structure remained
profitable before sizing overlays), but the gates exist precisely so that
"marginal, explainable" failures don't get promoted. Two mitigating facts are also
disclosed: the holdout window was previously exposed to a different strategy
family's mining (documented in `data_audit.md` §9), and the continuous-account view
of the same frozen rule (2018→2026, +15.2%/yr, maxDD −19.4%) shows the fresh-start
2025 result is substantially sequence risk. Neither fact upgrades the verdict.

### Robustness evidence (train-only, before any validation/holdout look)

- **Parameter neighborhood: 18/18 cells positive** (delta 0.4–0.6 × roll 14/21/30 ×
  DTE 45/60): CAGR +8.1% to +26.5%; the chosen config is mid-pack, not the argmax.
- Execution: optimistic → conservative costs ~0.3 CAGR pts; **stress** fills (cross
  full spread +3 ticks, double fees) cost ~1.7 pts. No cliff.
- Timing: 1–2 session entry/exit delay: Sharpe 0.75 → 0.74/0.65.
- Outliers: P&L without best trade +$5.3k; without two best +$3.9k; without best
  year (2020) +$3.7k (of +$7.2k total).
- Trade-EV block-bootstrap CI95: **[+$54, +$440]** per trade (n=36, base).
- Honest negatives: daily-return Sharpe CI95 [−0.12, +1.72] includes zero (4.4y of
  trend-concentrated returns cannot statistically pin a Sharpe); the **edge is
  QQQ-specific in-sample** — the same rule on SPY has EV CI [−$172, +$209] and is
  outlier-dependent; on IWM it is negative. G1 supported SPY and QQQ equally, so
  instrument concentration is a real, disclosed selection risk.
- Multiple testing: 148 engine runs total; the H2 family used a 12-config coarse
  grid + 26-run neighborhood. All results (including failures) are in the ledger;
  the neighborhood's 18/18 positivity — not a single surviving cell — is the
  anti-mining evidence.

### Worst historical outcomes (frozen spec, base fills)

Worst trade −$1,438 (14% of a $10k account); worst month −13.9% (Mar-2020); worst
regime windows: COVID crash −18.2% (QQQ −27.9%), 2018Q4 −10.9% (QQQ −16.1%),
2022 bear −5.6% (QQQ −33.2%), 2025 whipsaw +0.6% while QQQ finished +21% — the
strategy's characteristic failure is *paying premium repeatedly into failed rallies
and then under-participating in the V-recovery*.

## 4. Fidelity execution (summary; full checklist in `fidelity_execution_checklist.md`)

Permission needed: **Level 2 — "buy calls and puts"** (long calls only; zero
assignment risk; the only operational hazard is accidental auto-exercise of an ITM
call held to expiry, made structurally impossible by the ≤5-DTE rule). Orders:
single-leg day-limit orders, ~15:15–15:45 ET, ~9/year. Required cash: premium ≤10%
of equity + $0.65/contract; ≥85% of the account never leaves the money-market core.

## 5. Backup candidate status (H5 wheel, XLF/SLV/EWZ)

XLF train (2018–2022): CAGR +2.2–3.4%, Sharpe ≤0.54, maxDD ~−11% across the 6-cell
grid — economically "T-bills + ~2%/yr" from ~57% collateral utilization; assignment
mechanics (COVID 2020) handled and survivable. This does not clear the promotion
bar on its own. SLV/EWZ chains (higher IV, fatter premiums) were still downloading
at report time; their grids will be appended to `candidate_results.csv` and this
section when complete. Until then **no backup strategy is recommended**.

## 6. Final recommendation

**Conduct more research — do not deploy capital.** Specifically:

1. The TG-CER result is a *near-miss with a diagnosed failure mechanism*, not a
   validated strategy and not an obvious reject: G0–G4 all passed (economic story
   anchored on 18 years of pre-options data; conservative-cost train CI positive;
   validation +36% conservative; 18/18 robustness cells), and the single
   pre-registered holdout failed marginally on the sizing overlay (brake), not on
   the signal or structure.
2. The honest path forward, consistent with this repo's standards: **redesign the
   brake** (e.g. re-entry at half-size on a fresh trend_entry signal rather than
   equity-level recovery; or replace the equity brake with the trend gate alone,
   whose worst observed DD unbraked was −31.7%) — then **pre-register the amended
   spec and paper trade it forward 6–12 months** (`scripts/tg_cer_signal.py` is the
   generator; protocol in the checklist §9). Forward data is the only unmined
   sample left for QQQ.
3. Do **not** trade the wheel for yield at XLF-like premium levels; revisit only if
   SLV/EWZ change the picture materially.
4. A negative-but-honest bottom line for today's $10k: hold the cash at ~4% in the
   core fund while the paper-trading evidence accumulates. That is the standard
   every options strategy here had to beat, and none has yet done so out-of-sample.
