# Research plan — a Level-2-compatible options strategy for a $10,000 Fidelity account
*(written 2026-07-29, before any new experiments were run; partitions and gates frozen here)*

## 0. Mandate and hard constraints

Design, validate, and (only if evidence supports it) recommend for **paper trading** a
systematic options strategy executable in a $10,000 Fidelity brokerage account with
**Tier 1 / Level 1–2 options permissions**:

| Capability | Fidelity level | Available here |
|---|---|---|
| Covered calls / buy-writes | Level 1 | yes, if 100 shares affordable |
| Buy calls and puts | Level 2 | **yes** |
| Cash-secured puts | Level 2 | yes, if strike×100 affordable |
| Any spread (vertical/calendar/diagonal) | Level 3 | **NO — excluded** |
| Naked anything, portfolio margin | 4–5 | NO — excluded by mandate |

**Capital reality at $10k** (spot levels ≈ July 2026): 100 shares SPY ≈ $63k,
QQQ ≈ $57k, IWM ≈ $22k. Cash-secured puts / covered calls on the three underlyings
with existing data are **not fundable**. Therefore the executable strategy families are:

- **F1: Long calls / long puts** on SPY, QQQ, IWM (existing 8-yr chain data).
- **F2: CSP / wheel on a liquid sub-$100 ETF** (XLF, SLV, GDX, EEM…) — requires a new
  ThetaData pull; considered only if F1 hypotheses fail or as a comparison arm.

All prior research in this repository (short call verticals) is both invalidated for
live use by the permission constraint and retired on its own evidence. This is a new
program; nothing from the prior strategy is assumed.

## 1. Data and partitions (frozen)

Data per `data_audit.md`. Chronological partitions:

| Partition | Span | Use |
|---|---|---|
| **Signal pre-validation** | 2000-01 → 2017-12 (FMP daily prices, no options) | test signal→forward-underlying-return links on data never touched by any backtest here |
| **Train** | 2018-08 → 2022-12 (options) | hypothesis development, coarse grids. Contains Q4-2018 selloff, COVID crash + V-rebound, 2021 low-vol bull, 2022 bear/rate shock |
| **Validation** | 2023-01 → 2024-12 (options) | confirm frozen-ish candidates, tune nothing new. 2023–24 recovery + tech rally |
| **Holdout** | 2025-01 → 2026-07 (options) | **single use** after full freeze & pre-registration commit. Includes 2025 vol events + most recent conditions |

Contamination disclosure: the holdout window was previously exposed to short-spread
mining (different family). Disclosed in every report; forward paper trading is
mandatory regardless of holdout result. Signal pre-validation on 2000–2017 exists
precisely to anchor hypotheses outside all contaminated data.

Within train, per-year folds (2018H2, 2019, 2020, 2021, 2022) serve as a coarse
walk-forward stability check: a candidate must not owe its result to one fold.

Look-ahead rules: all signals computed from data lagged ≥ 1 session (the existing
leakage-tested feature registry) or from the same 15:30 snapshot used for pricing;
Greeks/IV computed locally from that snapshot only; no vendor Greeks.

## 2. Hypotheses (economically distinct, stated before testing)

**H1 — Drawdown-rebound long calls.** Buy near-ATM calls when the index is emerging
from a deep drawdown (e.g. was ≥ X% below its high within the last N sessions and
short-term momentum has turned up).
*Edge source:* the largest one-to-two-month upside tails in equity indexes cluster in
post-drawdown rebound windows (V-recoveries). Independently supported by the user's
20-year price study and, mechanically, by the prior study's finding that rebound
windows carried 58–93% of all short-call losses — what bleeds call sellers pays call
buyers. The open question is whether elevated IV in those windows already prices the
tail. *Should fail when:* the drawdown resumes (2008-style cascade); or IV is so high
that even a strong rally doesn't beat the premium paid.

**H2 — Trend-following long calls (defined-risk stock replacement).** While a
long-term trend filter is on (e.g. index above 200-session MA), hold an ITM call
(Δ≈0.7–0.8, 45–60 DTE), roll before expiry; step aside when trend is off.
*Edge source:* equity risk premium + documented time-series momentum, accessed with
strictly capped downside (premium) instead of unfundable stock. *Should fail when:*
markets chop around the filter (whipsaw theta+spread churn) or gap down hard from
above the MA (2020-02).

**H3 — Cheap-convexity timing (IV vs RV).** Buy premium (calls, or symmetric
strangles) only when IV is cheap vs recent realized (IV−RV20 < 0 and/or IV rank low),
direction from a momentum tiebreak.
*Edge source:* the variance risk premium is time-varying and occasionally inverts;
long premium may be +EV precisely then. *Should fail when:* low IV correctly predicts
even lower RV (usual case — this is the standard VRP tax).

**H4 — Long puts in downtrends (control).** Mirror of H2 with puts under the 200d MA.
*Expected to fail* (puts are the expensive side of skew; downtrend IV already
elevated). Included as a control for the testing machinery and to document the
negative result honestly.

**H5 — CSP wheel on a sub-$100 ETF (contingent).** Sell 30-DTE ~0.25Δ cash-secured
puts on XLF (or SLV/GDX), take assignment, exit via covered calls.
*Edge source:* the put-side variance risk premium — the one option premium with broad
literature support. *Costs:* needs a new data pull; single-position concentration in a
$10k account; thin premia on cheap ETFs after fees. Activated only if F1 fails or as
a benchmark arm if time permits.

Structure variables explored (coarse only): DTE ∈ {21, 30, 45, 60}, entry delta ∈
{0.35, 0.50, 0.65, 0.80}, exits {time stop, trend-off, profit multiple, expiry},
underlying ∈ {SPY, QQQ, IWM}. No fine grids; neighborhoods only for finalists.

## 3. Execution model (all results reported under all three)

Single-leg fills from the 15:30 ET NBBO snapshot; no order may fill on a gap day, at
a crossed/locked quote, zero bid (for sells), or outside quoted size ≥ 1 contract.

| Scenario | Buy fill | Sell fill | Fees/contract/side |
|---|---|---|---|
| Optimistic | mid | mid | $0.70 |
| Base | mid + 50% of half-spread | mid − 50% of half-spread | $0.70 |
| Conservative | ask + $0.01 | bid − $0.01 | $1.00 |

Justification: marketable limit orders on penny-pilot ETF options routinely fill
between mid and ~75% of the half-spread; base takes the midpoint of that experience;
conservative assumes crossing the full spread plus a tick of adverse movement — an
upper bound for 1–2 lot retail flow in these names (median relative spreads 0.5–2%,
audit §3). Fees: Fidelity $0.65/contract commission + ~$0.05 regulatory (base);
$1.00 stress. Exercise/assignment: $0 at Fidelity; modeled explicitly (American,
physical settlement, dividend-driven early exercise for ITM calls; expiration
auto-exercise if ITM ≥ $0.01 unless sold; all expiring longs are sold at the last
snapshot if they have a bid, else abandoned).

Cash earns the 4-week T-bill rate daily; a strategy must beat leaving the cash alone.

## 4. Portfolio and risk rules (search space, not final)

- Account $10,000; positions in whole contracts only.
- Max premium at risk per position: tested at {5%, 10%, 15%} of equity; hard cap 20%.
- Max aggregate premium at risk: tested at {10%, 20%, 30%}.
- Delta-notional leverage cap: total |Δ|×spot×100 ≤ 1.0× equity (tested 0.5×/1.0×/1.5×).
- Min cash reserve: 20% of equity, never invested.
- Max simultaneous positions: ≤ 3.
- Drawdown brake: if equity < 85% of trailing-12m high, halve size; < 75%, stop, review.
- Expiration week: no long option held into its final 5 DTE (sold or rolled earlier).
- Compounding: sizes recomputed from current equity; sizing must survive all tested
  percentages, not be picked as the historical argmax.

## 5. Metrics, benchmarks, uncertainty

Full metric set per the mandate (CAGR, vol, Sharpe, Sortino, maxDD, Calmar, win rate,
avg win/loss, profit factor, EV/trade, trade count, holding period, utilization,
worst day/week/month/trade, longest DD, per-year and per-regime tables, all under all
three fill scenarios). Benchmarks: (a) buy-and-hold SPY (total return), (b) T-bills,
(c) unoptimized family benchmark — always-long 30-DTE ATM call rolled monthly, sized
identically. Uncertainty: moving-block bootstrap CIs on per-trade EV and on
daily-equity Sharpe; episode-level (cluster) CIs when entries overlap. A high Sharpe
from < ~40 independent trades will not be treated as evidence.

## 6. Multiple-testing discipline

- Every engine run is appended to `experiment_log.csv` (id, date, hypothesis, config
  hash, partition, scenario, headline stats, verdict, notes) — failures included.
- Coarse grids only during exploration (≤ ~50 configs per hypothesis); finalists get
  neighborhoods, not re-optimization.
- The final rule is frozen in `strategy_spec.md` and committed **before** the holdout
  run; the holdout is evaluated once, results recorded verbatim whatever they say.
- Selection-bias estimate reported: number of configurations examined per family vs
  the survivor's margin (a survivor whose edge is within the noise band implied by
  the search width is rejected).

## 7. Decision gates (frozen now)

- **G0 (economic):** hypothesis has a stated edge source that does not depend on the
  2018–2026 options sample itself.
- **G1 (signal):** on 2000–2017 prices, the entry signal's forward 20–40 session
  underlying return distribution must differ from unconditional in the direction the
  strategy needs (mean shift or tail asymmetry, block-bootstrap 90% CI excluding 0),
  on at least 2 of 3 underlyings (or the one underlying it will trade).
- **G2 (train):** net EV/trade > 0 under conservative fills; base-fill bootstrap 95%
  CI excludes 0; no single year or single trade accounts for the sign.
- **G3 (validation):** base-fill net > 0 and drawdown profile consistent with train;
  no new tuning after looking (one pre-declared simplification allowed, logged).
- **G4 (robustness):** survives ±1 step in DTE/delta/thresholds, 2× fees, +50%
  spread widening, 1-day entry delay, best-single-trade removal, best-year removal.
- **G5 (holdout, single-use):** frozen spec, base fills: net > 0 and maxDD within
  1.5× of train+validation worst; conservative fills: not catastrophically negative.
- If nothing passes: **report that no reliable strategy was found** (explicitly an
  acceptable outcome), with the CSP arm (H5) documented as future work if untested.

## 8. Deliverables

As mandated: this plan, `data_audit.md`, `experiment_log.csv`, reproducible pipeline +
event-driven backtester (`src/level2_research/`), unit tests, `candidate_results.csv`,
sensitivity charts/tables, `final_report.md`, `strategy_spec.md`,
`fidelity_execution_checklist.md`, paper-trading signal generator, README section.
