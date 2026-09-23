# Does Selectively Selling Covered Calls Improve Total Return or Risk-Adjusted Return Compared with Simply Holding the ETF?

This repo is the paper and everything needed to reproduce it: an empirical study of
covered-call writing on SPY, QQQ, and IWM (Aug 2018 – Jul 2026), testing (1) whether an
unconditional monthly covered-call program beats buy-and-hold, and (2) whether a small,
signal-gated rule that skips writing the call in high-breach-risk months can fix it.

**Read the paper:** [`reports/covered_call_writeup.html`](reports/covered_call_writeup.html) (interactive, with the equity-curve chart — see [Viewing the report](#viewing-the-report) below) or [`reports/covered_call_writeup.pdf`](reports/covered_call_writeup.pdf) (static, for quick reading/printing).

## Repo layout

```
reports/covered_call_writeup.html   the paper (self-contained HTML, no build step)
reports/covered_call_writeup.pdf    the same paper, as a static PDF
results/covered_call/               signal-research outputs (breach-probability study)
results/covered_call_sweep_rule/    the three-strategy backtest bundle the paper's
                                     tables and figures are sourced from
scripts/                            the three scripts that produced everything above
src/covered_call/                   the backtest engine + strategies those scripts use
tests/                              unit tests for the engine and option-pricing math
data/normalized/                    input price/options data (not committed — see below)
```

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest        # 58 tests, should all pass
```

## Data

`data/normalized/` is not committed (176 MB of licensed options-chain data). To
reproduce the scripts below, populate it with:

```
data/normalized/options/{spy,qqq,iwm}/chain_YYYY-MM.parquet   monthly 15:30 ET NBBO snapshots
data/normalized/prices_long/{SPY,QQQ,IWM}.parquet              daily adjusted OHLC
data/normalized/prices_long/{SPY,QQQ,IWM}_dividends.parquet    ex-dividend history
data/normalized/prices_long/{VIX,VXN,RVX}.parquet               each fund's own vol index
data/normalized/prices_long/eigen_features_long.parquet         market absorption-ratio series
data/normalized/prices_long/sectors/{XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY}.parquet
data/normalized/rates/tbill_4w.parquet                           annualized decimal T-bill rate
```

`results/` and `reports/covered_call_writeup.html` are already checked in, so you don't
need the data just to read the paper — only to regenerate the numbers behind it.

## Reproducing the paper's numbers

Three scripts, run from the repo root, in any order (none depend on each other):

```bash
# Breach-probability + 10-signal IC study, 2005-2026 weekly sample (Sections 5-6)
python scripts/covered_call_signal_research.py      # -> results/covered_call/signal_research.json

# Decile-cutoff breach-rate sweep for every signal (used to pick cutoffs, Section 7)
python scripts/covered_call_threshold_sweep.py       # -> results/covered_call/threshold_sweep.json

# The three-strategy backtest: buy-and-hold, naive covered call, rule-gated
# covered call, for SPY/QQQ/IWM (Sections 4, 8 — this is the paper's main result)
python scripts/covered_call_sweep_rule_backtest.py   # -> results/covered_call_sweep_rule/
```

The third script is the one that matters most: it writes
`results/covered_call_sweep_rule/artifact_bundle.json`, which is where every number and
every chart series in the paper comes from. The per-fund gating rule (Table 4 in the
paper) is hardcoded near the top of that script, with the reasoning for each leg in
comments.

## Viewing the report

The report is a single static HTML file with no external dependencies — open it
directly in a browser, or serve it locally:

```bash
cd reports && python3 -m http.server 8935
# then open http://localhost:8935/covered_call_writeup.html
```

## Code

- `src/covered_call/engine.py` — the event-driven backtest engine (orders, fills, a
  double-entry-ish account, assignment/expiry handling)
- `src/covered_call/market.py` — loads option chains and daily price/dividend/rate data
- `src/covered_call/selection.py` — deterministic contract selection (Black-Scholes delta
  computed locally from the quote midpoint, never trusted from vendor fields)
- `src/covered_call/options/` — Black-Scholes pricing and implied-vol solving
- `src/covered_call/signals.py` — builds the regime-signal panel (RSI, trend, MA200,
  sector correlation, absorption shift, vol index, etc.) used by the gate
- `src/covered_call/strategies.py` — `BuyHoldShares` / `DripBuyHoldShares` (baselines)
  and `CoveredCallStrategy` / `ReinvestingCoveredCallStrategy` (naive vs. gated covered
  call; the paper uses the reinvesting variant throughout)
- `src/covered_call/metrics.py` — CAGR, Sharpe, Sortino, Calmar, drawdown, yearly returns

## Caveats

The paper's own Section 10 (Limitations) states these plainly: the gating rule was
constructed and evaluated on the same 2018–2026 sample, which is honest in-sample
research, not an out-of-sample test. Read that section before drawing conclusions beyond
what the paper claims.
