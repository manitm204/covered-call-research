# Pre-registered out-of-sample test: the frozen rule on QQQ

Registered 2026-07-24, BEFORE any QQQ options data was downloaded or examined.
The commit containing this document, `configs/strategy_qqq_prereg.yaml`, and
`scripts/preregistered_qqq_test.py` is the registration timestamp. Nothing in
this protocol may change after the QQQ chain pull completes.

## Motivation

The SPY hypothesis (docs/CONCLUSIONS.md) was mined from 2018–2026 SPY data:
53 trades, ~26 effective episodes, hundreds of implicit comparisons. Its
CIs overstate confidence by construction. QQQ options are unmined data: if
the regime edge is real market structure (calm/overbought/diversified grinds
are safe to sell into), it should transfer; if it was SPY-specific noise, it
should not.

## The rule (frozen — identical to the SPY final hypothesis)

- Entry gate, evaluated weekly with all features lagged >= 1 session:
  **RSI(14) of NDX > 70** (mandatory) AND **at least 2 of 3**:
  1. `iv_minus_rv20 > 0` — VXN/100 minus 20-session realized vol of NDX
  2. `sector_avg_corr_20 < 0.45` — mean pairwise 20d correlation of the nine
     SPDR sector ETFs (market-wide series, identical to the SPY study)
  3. `absorption_chg_20d < 0` — 20d change in top-eigenvalue share of sector
     returns (market-wide, identical to the SPY study)
- Structure: short 0.15Δ QQQ call, long +$8, ~30 DTE (25–35), weekly entries,
  hold to expiry, max 2 open positions, 1 contract per entry.
- American exercise + physical settlement with the QQQ dividend calendar.
- Window: 2018-08-01 → 2026-07-22 (matching the SPY study).

### Symbol-mapping decisions (made now, before seeing results)

The SPY study computed underlying features on the S&P index (SPX) and used
VIX. The faithful QQQ analog: underlying features on **NDX**, vol index
**VXN** (both fetched 2026-07-24 into `data/normalized/aux`; QQQ feature
frame built at `results/qqq_prereg/features_qqq.parquet` from a bundle where
UNDERLYING=NDX and VIX=VXN — manifest alongside). Width stays $8 (QQQ ≈ $560
vs SPY ≈ $630; 1.4% vs 1.3% of spot — comparable). Thresholds are unchanged
round numbers: 70, 0, 0.45, 0.

## Endpoints (declared in advance)

- **Primary**: base-fill per-trade moving-block bootstrap 95% CI entirely
  above $0.
- **Secondary**: conservative-fill mean > 0; episode-level (30-day-gap
  clustering) bootstrap 95% CI entirely above $0; win rate >= 75%.
- **Verdict**: GO = primary met AND conservative mean > 0. WEAK = base mean
  > 0 but primary CI spans zero — reported as inconclusive, and explicitly
  NOT a license to tune thresholds. NO-GO = base mean <= 0.
- All numbers get reported regardless of outcome, including yearly means and
  the worst trade.

## Procedure

1. Commit this protocol (done before data pull).
2. Pull QQQ chains 2018-08-01 → 2026-07-22 via `xsp pull-thetadata --symbol
   QQQ` (15:30 ET snapshots, same pipeline as SPY) + `xsp validate-data`.
3. Run `python scripts/preregistered_qqq_test.py` ONCE. The script writes a
   single-use marker and refuses to run twice.
4. Record the verdict in docs/CONCLUSIONS.md verbatim.

## What is NOT allowed

Changing any threshold, the delta/width/DTE, the cap, the entry weekday, the
feature definitions, or the endpoints after seeing any QQQ result; re-running
with variations; reporting only favorable scenarios. If the test fails, the
conclusion is "the SPY result does not transfer" — full stop. Any new
hypothesis derived from QQQ data would itself need fresh pre-registration on
other unmined data.
