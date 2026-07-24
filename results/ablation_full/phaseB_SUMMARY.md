# Phase B: surface-relative strike selection (exploratory)

Run 2026-07-24. Question: within the frozen gate (RSI>70 + >=2 of 3) and frozen
structure ($8 wide, 30 DTE, weekly, hold, cap-2), can smile-relative
"mispricing" pick a better short strike than fixed 0.15 delta?

Scripts: scripts/phaseB_surface_diagnostic.py (out-of-engine, 640 candidate
spreads across 55 gated dates; SVI slice fit per date/expiry; base-fill
hold-to-expiry approximation validated 0.998 corr vs engine P&L) and
scripts/phaseB_engine_ab.py (full-fidelity engine A/B via selection override,
base + conservative fills, cap-2). Data:
results/ablation_full/phaseB_candidates.parquet, phaseB_engine_ab.json.

## Findings

1. **The mispricing story is backwards.** Strikes trading RICH vs the fitted
   SVI smile UNDERPERFORM when sold: within-date partial correlation of spread
   P&L with richness residual (controlling delta) is -0.20; a sell-the-richest-
   strike rule loses $4-8/spread vs fixed 0.15d in every test. Local richness
   at these deltas is information (strikes being bid for a reason), not free
   premium.

2. **The mirror rule (sell the CHEAPEST strike in the 0.10-0.20 band) showed
   +$12/spread in the diagnostic (episode CI [+3.2,+21.2]) but did not survive
   the engine.** Engine-level, quality-filtered smile fit: +$5.3/spread, episode
   CI [-4.8,+16.0] — spans zero, base and conservative alike. With an
   unfiltered smile fit the effect was $0.0. The sign (cheap > fixed > rich) is
   stable across every implementation; the magnitude is not, and half the
   diagnostic gain was delta drift (cheap picks average 0.163-0.172 delta),
   which Phase A already prices.

3. **Decision (per the criterion set before running): keep fixed delta.** A
   replacement rule had to beat fixed 0.15d on episode CI and conservative
   fills; both residual rules fail. A strike rule whose edge halves between two
   reasonable smile-fit implementations is not pre-registrable.

4. Implication for Phase C (XGBoost strike ranker): the cross-sectional signal
   it would learn from is weak, sign-flipped vs intuition, and implementation-
   fragile; recommend skipping unless new features (term structure, put wing)
   are added.

## Caveats

Same 53-65 gated entries / ~26 episodes as everything else; the diagnostic
ignores early assignment (7/417 historically) and expiry fees; SVI residuals at
0.10-0.20 delta on SPY are typically < 1 vol point, i.e. a few cents — inside
noise for a $0.74 credit.
