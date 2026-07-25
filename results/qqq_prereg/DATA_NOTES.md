# QQQ dataset notes (written BEFORE the pre-registered test was run)

Pull completed 2026-07-24: 96 monthly files, 1,977 chain sessions,
6,749,994 rows, 2018-08-01 -> 2026-07-22, 15:30 ET snapshots, underlying
prices parity-implied (stock EOD endpoint 403s on the free tier — identical
to the SPY dataset).

## Deviations, decided and documented before any backtest was run

1. **15 sessions are missing** (0.75%): 2020-06-23, 2021-01-05, 2021-06-01,
   2021-06-14, 2021-07-21, 2021-08-05, 2021-08-06, 2021-09-10, 2021-10-18,
   2021-11-17, 2021-11-19, 2022-03-15, 2022-08-02, 2023-08-24, 2023-11-15.
   ThetaData returns persistent HTTP 500/472 for QQQ on these dates across
   retries and refreshes (SPY has data on all of them — vendor-side QQQ gaps).
   Effect: any weekly entry scheduled on those dates is skipped (no chain →
   selection fails, recorded as a failed entry attempt). One additional
   session (2022-02-02) recovered on refresh.
2. **underlying_eod.parquet** is derived from the per-date median
   parity-implied chain price (same derivation as SPY). The 15 missing dates
   are patched with Yahoo UNADJUSTED daily closes so that spreads expiring on
   the three missing Fridays (2021-08-06, 2021-09-10, 2021-11-19) have a
   settlement price. Both sources are actual (unadjusted) prices; patched
   values sit consistently between parity-implied neighbors.
3. **validate-data**: 402 crossed markets (0.006% of rows) — flagged "error"
   by the suite, but BETTER than the accepted SPY study dataset (1,213 /
   0.014%). Crossed quotes are rejected at selection by the engine in both
   datasets. All other findings are warn/info level and consistent with the
   SPY dataset's profile.

No rule, threshold, endpoint, or config was changed. These notes only record
data-quality handling.
