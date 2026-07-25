# Pre-registered out-of-sample test: BOTH frozen rules on IWM

Registered 2026-07-25, BEFORE any IWM options data was downloaded or examined.
The commit containing this document, `configs/strategy_iwm_prereg_h1.yaml`,
`configs/strategy_iwm_prereg_h2.yaml`, and `scripts/preregistered_iwm_test.py`
is the registration timestamp. Nothing in this protocol may change after the
IWM chain pull completes.

## Motivation

Two rules have been mined so far, neither validated:

- **H1 (the SPY rule)**: mined from SPY 2018–2026 (53 trades, ~26 episodes);
  took its one pre-registered shot on QQQ 2026-07-24 and FAILED (NO-GO,
  −$35.99/spread). This is its second and final out-of-sample attempt.
- **H2 (the QQQ rule)**: mined post-NO-GO from the burned QQQ data — the
  survivor of ~1,300 engine runs (exhaustive 1,023-combo sweep + structure
  ablation) on ~27 effective episodes. It has NEVER been tested on unmined
  data. The honest prior for H2 is ≈ $0.

IWM chains are unmined. The QQQ postmortem taught that opportunity triggers
are instrument-specific while danger regimes are universal, so plausible
outcomes include both rules failing on IWM for trigger-mismatch reasons.
That result would still be informative and will be reported as such.

## Multiplicity (declared in advance)

This is TWO pre-registered tests on one data pull. Each hypothesis is judged
against its own endpoints; at a nominal 5% level per test the family-wise
false-positive rate is ≈ 10%. Any GO claim will be reported with this caveat.
Both verdicts get reported regardless of outcome. The two hypotheses may
share entry dates; they are backtested and evaluated independently.

## H1 — the SPY rule (frozen; IWM analogs)

- Entry gate, evaluated weekly with all features lagged >= 1 session:
  **RSI(14) of RUT > 70** (mandatory) AND **at least 2 of 3**:
  1. `iv_minus_rv20 > 0` — RVX/100 minus 20-session realized vol of RUT
  2. `sector_avg_corr_20 < 0.45` — mean pairwise 20d correlation of the nine
     SPDR sector ETFs (market-wide series, identical to the SPY/QQQ studies)
  3. `absorption_chg_20d < 0` — 20d change in top-eigenvalue share of sector
     returns (market-wide, identical to the SPY/QQQ studies)
- Structure: short 0.15Δ IWM call, long **+$3**, ~30 DTE (25–35), weekly
  entries, hold to expiry, max 2 open positions, 1 contract per entry.

## H2 — the QQQ rule (frozen; IWM analogs)

- Entry gate, evaluated weekly with all features lagged >= 1 session
  (ALL three required — no voting):
  1. `ret_120d > 0.15` — 120-session return of RUT above +15%
  2. `dist_ma200 > 0.10` — RUT more than +10% above its 200-session MA
  3. `sector_avg_corr_20 < 0.45` — market-wide, as above
- Structure: short 0.12Δ IWM call, long **+$6**, ~30 DTE (25–35), weekly
  entries, hold to expiry, max 2 open positions, 1 contract per entry.

Both: American exercise + physical settlement with the IWM dividend calendar.
Window: 2018-08-01 → 2026-07-22 (matching the SPY and QQQ studies).

### Symbol-mapping decisions (made now, before seeing results)

Underlying features computed on **RUT** (Russell 2000 index, Yahoo `^RUT`);
vol index **RVX** (Cboe Russell 2000 Volatility Index; primary source the
Cboe public index-history CSV, fallback Yahoo `^RVX` if the CDN lacks it —
whichever succeeds is recorded in the bundle manifest). Feature frame built
from a bundle where UNDERLYING=RUT and VIX=RVX, all market-wide series
identical to the SPY/QQQ studies — zero feature-code changes.

Width scaling matches percent-of-spot at registration (2026-07-22 closes:
SPY $747.41, QQQ $705.35, IWM $293.79), rounded to IWM's $1 strike grid:
H1 $8/SPY = 1.07% → $3.15 → **$3**; H2 $15/QQQ = 2.13% → $6.26 → **$6**.
Thresholds are unchanged round numbers: 70, 0, 0.45, 0 (H1); 0.15, 0.10,
0.45 (H2). Deltas unchanged: 0.15 (H1), 0.12 (H2).

## Endpoints (declared in advance, per hypothesis — identical to the QQQ test)

- **Primary**: base-fill per-trade moving-block bootstrap 95% CI entirely
  above $0.
- **Secondary**: conservative-fill mean > 0; episode-level (30-day-gap
  clustering) bootstrap 95% CI entirely above $0; win rate >= 75%.
- **Verdict**: GO = primary met AND conservative mean > 0. WEAK = base mean
  > 0 but primary CI spans zero — reported as inconclusive, and explicitly
  NOT a license to tune thresholds. NO-GO = base mean <= 0.
- All numbers get reported regardless of outcome, including yearly means and
  the worst trade, for both hypotheses.

## Procedure

1. Commit this protocol (done before data pull).
2. Fetch RUT, RVX, IWM dividends into `data/normalized/aux`; build the
   `aux_iwm` bundle (UNDERLYING=RUT, VIX=RVX) and the IWM feature frame.
3. Pull IWM chains 2018-08-01 → 2026-07-22 via `xsp pull-thetadata`
   (15:30 ET snapshots, same pipeline as SPY/QQQ) + `xsp validate-data`.
   Underlying EOD derived from parity-implied chain prices as before; any
   vendor gaps handled exactly as documented in
   `results/qqq_prereg/DATA_NOTES.md` (Yahoo unadjusted-close patches,
   documented per-session).
4. Run `python scripts/preregistered_iwm_test.py` ONCE. The script runs BOTH
   hypotheses in a single invocation, writes a single-use marker, and
   refuses to run twice.
5. Record both verdicts in docs/CONCLUSIONS.md verbatim.

## What is NOT allowed

Changing any threshold, delta/width/DTE, the cap, the entry weekday, the
feature definitions, or the endpoints after seeing any IWM result;
re-running with variations; reporting only favorable scenarios; promoting a
WEAK to a GO; treating one hypothesis's result as license to re-judge the
other. If both fail, the conclusion is "neither mined rule transfers to
IWM" — full stop. Any new hypothesis derived from IWM data would itself
need fresh pre-registration on other unmined data (XSP/SPX chains or
forward paper trading).
