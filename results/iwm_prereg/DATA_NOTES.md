# IWM chain-pull data notes (2026-07-25)

Pull: `xsp pull-thetadata --symbol IWM --start 2018-08-01 --end 2026-07-22`
(15:30 ET snapshots, max-dte 70, same pipeline as SPY/QQQ). 96 monthly files,
4,131,714 rows, 1,948 sessions with quotes vs SPY's 1,990 (97.9%).

## Vendor gaps: 46 trading sessions have no IWM chain snapshot

Cross-checked against the SPY session list (ground truth for trading days).
Two persistent error classes from ThetaData, identical in character to the
QQQ pull (results/qqq_prereg/DATA_NOTES.md):

- ~19 sessions: `/v3/option/history/quote` HTTP 500, concentrated 2021–2022
  (e.g. 2021-06-11, 2022-03-09, 2022-12-05).
- ~27 sessions: "no underlying price (stock quote unavailable and parity
  fallback lacked usable pairs)", concentrated 2023–2025 (e.g. 2023-09-18,
  2024-12-09, 2025-07-03).

A resume pass re-pulled incomplete months and recovered nothing (errors are
persistent server-side). Effect on the backtest: a missing session on an
entry day silently skips that week's entry; positions are marked on the next
available session. 46/1990 = 2.3% of sessions.

## underlying_eod.parquet

Derived as per-date median parity-implied chain price (same derivation as
SPY/QQQ; free ThetaData tier 403s the stock EOD endpoint). 1,948 parity
sessions + 45 patches = 1,993 dates.

**45 expiry-date patches from Yahoo unadjusted closes**: IWM lists M/W/F
weeklies, so many gap sessions are also expiration dates, where a settlement
price is required. All 45 in-window expiry dates lacking a parity price were
patched with the Yahoo raw (unadjusted) close — full list = the gap list
above minus non-expiry dates; spot-checks are continuous with neighboring
parity-implied closes.

**Special case 2025-01-09**: US markets were closed (national day of
mourning for President Carter) — no close exists. Options labeled with that
expiration actually settled at the next session; the date is mapped to the
2025-01-10 close ($216.83).

## Validation

`xsp validate-data`: passed=false driven solely by 51 crossed-market rows
out of 4,131,714 (0.0012%) — an order of magnitude cleaner than the accepted
SPY dataset (1,213 / 0.014%) and better than QQQ (402 / 0.006%). Warnings
(zero bids far OTM, strike gaps, stale far-OTM quotes) match the SPY/QQQ
profile. Accepted on the same precedent.
