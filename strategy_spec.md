# Strategy specification — "Trend-Gated Convex Equity Replacement" (TG-CER)

**STATUS: DRAFT — frozen for validation/holdout testing on 2026-07-29.**
Numbers in this file define the rule; they may not be changed in response to
validation or holdout results. Any change requires a new pre-registration.

## 0. One-paragraph description

Hold ~90% of the account in cash earning money-market/T-bill interest. While QQQ
is in a confirmed long-term uptrend (price above its 200-session moving average by
a 1% hysteresis band), hold one ~50-delta QQQ call with ~60 days to expiry, sized
so the premium at risk is at most 10% of account equity; roll it at 21 DTE. When
the trend filter turns off, sell the call and sit fully in cash. Maximum loss per
roll cycle is the premium paid (≤10% of equity); the account can never lose more
than the option premium plus foregone interest in any market crash.

## 1. Universe and instruments

- Underlying: **QQQ** (primary). SPY/IWM results are reported for transfer
  reference but are NOT part of the strategy.
- Instrument: exchange-listed QQQ **call options** (American, physical delivery).
- Account: $10,000 initial, Fidelity cash account, options Level 2 required
  ("buy calls and puts"). No margin, no spreads, no short options.

## 2. Signal (point-in-time, computed at previous session's close)

Let `P_t` = QQQ dividend-adjusted close, `MA200_t` = 200-session simple moving
average of adjusted closes, both through session t−1 (signals are lagged one
session; the decision on session t uses only data through t−1).

- `trend_entry(t)` = P_{t-1} > 1.01 × MA200_{t-1}
- `trend_hold(t)`  = P_{t-1} > 0.99 × MA200_{t-1}

## 3. Position rules

Evaluated once per session, in the afternoon (target ~15:30 ET):

1. **Exit-all**: if `trend_hold` is false → sell every open call at market
   (marketable limit at the NBBO bid or better). Do not re-enter until
   `trend_entry` is true again.
2. **Roll**: if an open call has ≤ 21 calendar DTE and `trend_hold` is true →
   sell it; the replacement purchase follows rule 3 the same session.
3. **Entry**: if no call is open (after rolls) and `trend_entry` is true →
   buy 1..N contracts of the selected call (rule 4) where
   `N = floor(0.10 × equity / (100 × ask + fees))`. If N = 0 (premium too
   expensive), buy nothing and re-check next session.
4. **Contract selection (affordability-aware)**: expiration nearest to 60
   calendar days within [40, 70]; among that expiry's calls with valid quotes
   (bid > 0 for exits, uncrossed, relative spread ≤ 10%, OI ≥ 100) and
   Black-Scholes delta (from the quote midpoint IV at the decision snapshot)
   in **[0.30, 0.62]**, buy the **highest-delta strike whose ask×100 + fee fits
   the budget** (preferred delta 0.50 when affordable). If no strike in the
   band is affordable, no trade. [Rationale: a fresh $10k account cannot fund a
   0.50Δ 60-DTE QQQ call at 2026 price levels; walking down the delta band is
   the pre-declared small-account adaptation, backtested identically:
   train CAGR 12.4% base / 12.0% conservative, Sharpe 0.66, maxDD −21.6%.]
5. **Expiry-week backstop**: never hold a call with ≤ 5 DTE (roll or exit
   before; the ≤21-DTE roll rule makes this unreachable in normal operation).
6. **No averaging down, no adds, max 1 open expiry** at a time.

## 4. Risk and account rules

- Premium at risk per cycle: ≤ 10% of current equity (the sizing rule).
- Minimum cash reserve: ≥ 85% of equity is always in cash/core; never deploy it.
- Drawdown brake: if account equity < 85% of its trailing-12-month high, halve
  the budget to 5% until a new high; if < 75%, stop trading and review.
- Compounding: N is computed from current equity each entry.
- No positions in other instruments while the strategy runs (the H1 rebound
  sleeve is documented in research but NOT part of this spec).
- Expected activity: ~9 rolls/year in a persistent trend; long flat spells
  (weeks–months) below the MA are normal and correct.

## 5. Worked example

Equity $10,000; QQQ adjusted close yesterday 570.00, MA200 545.00 →
570 > 1.01×545 = 550.45 → trend_entry true. At 15:30 the 59-DTE expiry (within
[40,70]) shows: 570C at 19.80×20.10 (Δ≈0.51, cost $2,010.65 — exceeds the
$1,000 budget), 595C at 9.40×9.60 (Δ≈0.36, cost $960.65 — fits), 605C at
6.90×7.10 (Δ≈0.29 — below the 0.30 floor). Selection: highest affordable delta
in [0.30, 0.62] → **buy 1× 595C with a marketable limit at $9.60**. In a larger
account ($25k+), the same rule buys the 0.50-delta 570C. If nothing in the band
fits the budget (extreme IV), no trade that session.

Exit example: three weeks later the call has 21 DTE, QQQ 590, call worth 27.00 ×
27.30. Roll: sell at ≥ 27.00, then buy the new ~60-DTE 0.50-delta call if
trend_hold still true.

## 6. Pseudocode

```
each session t (afternoon):
    s = signals through close(t-1)
    for pos in open_calls:
        if not trend_hold(s):            sell(pos); continue
        if dte(pos) <= 21:               sell(pos)
    if trend_entry(s) and no open_calls:
        budget = 0.10 * equity
        c = highest_affordable_delta_call(
                dte_target=60, dte_band=[40,70], delta_band=[0.30,0.62],
                prefer_delta=0.50, max_cost=budget,
                max_rel_spread=0.10, min_oi=100)
        if c is not None:
            n = floor(budget / (100*c.ask + fee))
            if n >= 1: buy(c, n)
```

## 7. Why this should work (economic rationale)

1. **Equity risk premium with a floor.** The long call collects equity upside;
   the maximum loss per cycle is the premium. ~90% of the account earns the
   T-bill rate, which in high-rate regimes pays a large fraction of the theta.
2. **Trend filter as regime selector, not alpha.** On 2000–2017 data (never used
   by any options backtest here), being above the 200d MA does not raise the
   mean much, but it collapses the left tail (P(−10% in 42d): 4.9% vs 10.4%
   unconditional on QQQ; CI excludes 0) and reduces vol ~30%. A long-call
   structure monetizes exactly that shape: full upside participation, and the
   flat-or-down states that bleed theta are partially avoided.
3. **Costs are structurally small.** ATM QQQ options trade at ~0.7% relative
   spread; ~9 single-leg orders/year on 1–2 contracts. The strategy survives a
   stress model (cross full spread + 3 ticks + double fees) with ~1.5 CAGR
   points of degradation.
4. **When it fails** (declared in advance): choppy markets that oscillate around
   the 200d MA (whipsaw premium bleed — mitigated but not removed by the 1%
   hysteresis band); gap crashes from above the MA (loss capped at premium);
   long low-rate + low-trend regimes where neither interest nor trend pays.

## 8. Backtest provenance (train period only; validation/holdout pending)

Train 2018-08→2022-12, base fills, $10k: CAGR 13.9%, Sharpe 0.75, maxDD −20.0%,
36 trades, EV +$204/trade (CI95 [+54, +440]), yearly: −8.0/+36.1/+31.9/+12.9/−4.7.
Conservative fills: CAGR 13.6%. Stress: 12.2%. All 18 parameter-neighborhood
cells positive. QQQ B&H same window: CAGR ≈ 12.4%, maxDD −35%. Runs recorded in
`experiment_log.csv`; artifacts under `results/level2/`.

---

# Backup strategy specification — SLV cash-secured put wheel (pre-registered 2026-07-29)

**STATUS: frozen before its validation/holdout runs.** Chosen as the H5Wheel class
defaults (not the train argmax); all 6 SLV train cells were positive (Sharpe
0.63–0.93), and the rule survives stress fills (+2.5% CAGR vs +2.9% base;
T-bill baseline ~1.5% in the same window).

Rules: while holding no SLV shares and no short put, sell 1 cash-secured SLV put,
expiry nearest 35 DTE in [25,50], delta closest to −0.25 (local BS delta from the
quote mid), quote filters as primary spec; collateral = strike × 100 ≤ 80% of
equity (else no trade). Buy it back at 50% of the credit, or at ≤ $0.10 with ≤ 7
DTE; otherwise take assignment. When assigned, sell ~0.25Δ covered calls (same
DTE band) at strikes ≥ cost basis; let shares be called away; repeat. Dividends:
none (SLV pays none). Early assignment modeled/expected when a short put's
extrinsic < $0.03. Fidelity permission: Level 1 (covered calls) + Level 2
(cash-secured puts). Fresh-$10k fundability at 2026 prices: ~$4.5–5k collateral ✓.

Train record (2018-08→2022-12, $10k): base +2.9%/yr, Sharpe 0.70, maxDD −5.8%,
41 trades; conservative +2.8%; stress +2.5%. Economic role: harvest the put-side
variance risk premium on a hard asset, uncorrelated with the primary's equity
trend exposure. Failure mode: sustained silver bear (assigned above market,
covered calls capped below basis) and premium too thin to beat T-bills in
high-rate/low-IV regimes.
