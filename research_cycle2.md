# Research cycle 2 — exogenous-data expedition, trees, and sleeve combination
*2026-07-29/30. Continuation of the program in `final_report.md`, at the user's
direction: broaden the data universe aggressively, screen "random" exogenous
features, train interpretable trees, and study multi-strategy combination. Same
gates, same ledger (`experiment_log.csv`).*

## 1. New data pulled (documented in `data/normalized/exog/manifest.json`)

| Dataset | Span | Notes / lag policy |
|---|---|---|
| Treasury yield curve (FMP, daily) | 2000→2026 | 10y−2y, 10y−3m slopes; lag 1 session |
| CFTC COT, E-mini S&P net spec positioning | 2010→2026 (864 wks) | lag 5 calendar days (Fri release of Tue data) |
| Sector P/E snapshots (FMP, weekly) | 2017→2026 | median sector P/E, tech/defensive P/E ratio; lag 1 session |
| Earnings calendar, S&P-500 members | 2016→2026 | trailing-90d beat rate, median EPS surprise, upcoming-21d report density; lag 2 days; **survivorship caveat**: current constituent list |
| FRED macro (CPI, UNRATE, UMich, INDPRO, retail) | 1947→2026 | lag 45 calendar days (release clearance) |
| Sector SPDR history (9 ETFs) | 1999→2026 | to rebuild eigen features pre-2018 |
| VIX3M | 2006→2026 | term-structure tests |

Combined with the existing 39 leakage-tested features → **57-feature point-in-time
matrix** (`data/features/cycle2_features.parquet`, builder `scripts/l2_exog_features.py`).

## 2. Hypothesis pre-validations (all logged; all FAILED their gates)

| Hypothesis | Test (out-of-era where possible) | Result |
|---|---|---|
| H6 credit gate (HYG confirm on MA200) | 2007–17, left-tail reduction | right direction (5.7% vs 8.5%) but CI90 spans 0 → fail |
| H8 VIX/VIX3M backwardation → long puts | 2006–17 | no mean edge; vol NOT underpriced in backwardation → fail |
| H9 momentum-leader rotation (SPY/QQQ/IWM) | 2001–17 | leader ≈ equal-weight (diff CI [−0.8%, +0.7%]) → fail |
| H12 Nov–Apr seasonality | 2000–17 | flat → fail |
| **H13 eigen gate** (see §3) | 2000–17 | **fail — the cycle's key lesson** |

## 3. The IC screen, the tree, and the eigen mirage (the cycle's main finding)

Spearman ICs of all 57 features vs forward-21d return/left-tail/vol on train
(2018-08→2022) with validation-stability check (2023–24): 336 pairs, 85
train-significant, 41 sign-stable (vs ~34 expected false positives at the 10%
level — most "signals" are noise, as expected). The standouts, strong in BOTH
periods, formed one cluster — the **eigenstructure family**: `pc1_loading_dispersion`
(IC −0.34 train / −0.23 valid vs QQQ fwd returns) and `absorption_ratio`
(+0.34/+0.25), plus the related vol-level features (`rv_60` +0.24/+0.43).

A depth-3 decision tree (purged 5-fold walk-forward, 21-session purge, 10%
min-leaf) independently selected the same features. Its in-sample logic, in plain
language: *"if the market trades as one tide (even PC1 loadings across sectors)
and CPI YoY < 6.4%, next month is up; fragmented leadership or 2022-style
inflation → down."* But **out-of-fold the tree does not beat the base rate**
(Brier 0.265–0.277 vs 0.233) — no tradable classifier. The CPI split is
one-episode memorization (2022).

The decisive test: eigen features were rebuilt from sector-ETF data back to 1999
(identical 120-session definition) and G1-tested on 2000–2017 — **all four
variants fail** (e.g. QQQ, low-dispersion tercile: +0.65% vs +0.56% fwd-21d,
CI90 [−1.55, +1.85]). The strongest in-era feature cluster of 2018–2024 simply
does not exist in 2000–2017. That is the signature of a period-specific artifact
(COVID/2022 crash-recovery epochs), and it is exactly what the out-of-era gate is
for. **No eigen-gated strategy advances.**

## 4. Sleeve-combination study (the user's diversification thesis)

Three sleeves with any positive-expectation evidence, run simultaneously in one
simulated $10k account, 2018-08→2024-12, base fills (portfolio return = T-bill +
Σ sleeve excess returns, so cash interest is not double-counted):

| Sleeve | CAGR | Sharpe | maxDD |
|---|---|---|---|
| TG-CER (QQQ trend calls, frozen spec) | +17.7% | 0.93 | −19.4% |
| H1 rebound calls (SPY deep-drawdown, 12 trades) | +8.0% | 0.47 | −23.1% |
| SLV CSP wheel (frozen backup) | +3.9% | 0.49 | −5.8% |
| **Combined** | **+24.8%** | **0.96** | **−27.7%** |

Pairwise daily excess-return correlations: **0.14–0.20** — the sleeves are
genuinely different exposures (trend-long-tech, crash-rebound, short-silver-vol),
which supports the diversification premise directionally. The honest
decomposition, however: at these sizings the combination mostly **stacks**
exposure (return ↑ 40%, drawdown ↑ 43%) rather than diversifies it (Sharpe only
0.93 → 0.96). Capital feasibility in one account: worst-case simultaneous usage
≈ 10% + 10% + ~55% collateral ≈ 75% — fundable, but the 20%-reserve rule binds
exactly in crisis regimes when all three deploy at once.

**Caveat that governs everything:** no constituent sleeve has passed a holdout
(TG-CER and SLV wheel each failed theirs; H1 has 12 trades). A portfolio of
unvalidated sleeves is an unvalidated portfolio — with better marketing. The
combination becomes interesting only after individual sleeves earn forward
(paper-trading) validation.

## 5. Cycle-2 conclusions and recommended next steps

1. The exogenous expedition (curve, macro, valuation, earnings, positioning,
   eigenstructure) surfaced **no new durable, gate-passing signal**. Eleven new
   pre-registered tests, eleven honest failures — all logged. This is the
   expected base rate of such expeditions and the strongest available evidence
   that the machinery doesn't manufacture edges.
2. The **left-tail-collapse trend gate (H2) remains the only out-of-era-validated
   signal** in the entire program; the deep-drawdown rebound (H1) is second,
   marginal. Everything else failed either out-of-era or out-of-fold.
3. Recommended path (unchanged in kind, sharpened in detail):
   - Paper trade the **brake-amended TG-CER** (re-entry on fresh trend signal
     rather than equity-level recovery) — pre-register the amendment first.
   - Paper trade the **three-sleeve combination at reduced size** (e.g. half
     budgets: 5% + 5% + 1 CSP) alongside, to collect forward correlation data.
   - Two of the queued data ideas were tested before closing the cycle, and both
     also failed: the **correlation-risk-premium proxy** (sector 21d-return
     dispersion / SPY RV21, 2000–17 terciles: high-dispersion +0.88% vs +0.55%
     unconditional, CI90 [−0.35, +0.93] → fail) and the **put/call OI ratio**
     built from the QQQ chains (train IC +0.03, CI spans zero, validation
     unstable → fail). The remaining untested direction is intraday-to-close
     return splits, which needs a second daily ThetaData snapshot per session.
4. Deployment recommendation: **unchanged — no live capital**. The honest
   baseline for the $10k remains the ~4% money-market core until forward
   evidence exists.
