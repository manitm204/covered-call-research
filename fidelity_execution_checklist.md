# Fidelity execution checklist — TG-CER (trend-gated QQQ calls)

Applies to the strategy in `strategy_spec.md`. Print this; check every line
before any order. **Paper trade first — do not deploy live until the paper
period (§9) is complete.**

## 1. Permissions and account setup

- [ ] Fidelity brokerage (cash) account, **options Level 2** ("Purchases of
  calls and puts") approved. Level 1 is NOT sufficient (it covers only covered
  calls); no level above 2 is needed. If Fidelity's tier wording differs, the
  required capability is exactly: **buy and sell long calls**.
- [ ] Core position = money market (SPAXX/FDRXX) so idle cash earns interest —
  ~90% of this strategy's balance sits there by design.
- [ ] No margin agreement needed; never enable "sell short" or spread trading
  for this strategy.

## 2. Buying power / funding

- [ ] Required settled cash before an entry: option cost (≤10% of equity)
  + $0.65/contract commission. Options require settled cash in a cash account —
  check "Available to trade without margin impact".
- [ ] Reserve check: after the buy, ≥85% of account equity remains in core.
- [ ] Never fund an entry by selling other holdings the same day (settlement).

## 3. Order construction (entry)

- [ ] Single-leg order: **Buy to Open, QQQ call**, per spec §3.4 (expiry nearest
  60 DTE inside 40–70; highest-delta strike in [0.30, 0.62] whose cost fits the
  budget; prefer 0.50Δ when affordable).
- [ ] Verify on the Fidelity chain before submitting: bid > 0, quote uncrossed,
  (ask−bid)/mid ≤ 10%, open interest ≥ 100, delta within band.
- [ ] Order type: **LIMIT, day**. Price at the ask (small size fills immediately);
  if you want price improvement, start at mid + half the spread's width and step
  up once after ~30s. Never market orders. Never leave GTC orders.
- [ ] Time: between ~15:15 and 15:45 ET (matches the backtest snapshot). Avoid
  the first 30 minutes and the last 5 minutes.
- [ ] Contracts: `floor(0.10 × equity / (100×ask + 0.65))` — recompute at order
  time; if 0, no trade.
- [ ] MAX-LOSS check (say it out loud): "If QQQ goes to zero I lose exactly
  premium + commission = $____ , which is ≤ 10% of my account."

## 4. Exits and rolls

- [ ] Roll when DTE ≤ 21 (sell old, then buy new per §3): two separate
  single-leg orders, sell first, confirm fill, then buy. Do not leg a "spread".
- [ ] Trend-off exit: if yesterday's close < 0.99 × MA200 → **Sell to Close
  everything today**, limit at the bid (step down a cent if unfilled in 60s).
- [ ] Hard rule: never hold any long option with ≤ 5 DTE.
- [ ] If a quote is crossed/locked or bid = $0 at exit time: wait 15 minutes,
  retry; if still broken near the close, sell at the best available bid ≥ $0.01;
  escalate to phone desk only if the position has ≤ 2 DTE.

## 5. Expiration / exercise / assignment precautions

- [ ] This strategy holds **long options only** — no assignment risk exists.
  The only exercise risk is *accidental auto-exercise*: an ITM long call held to
  expiration Friday auto-exercises into 100 QQQ shares (~$68k) you cannot fund →
  forced liquidation. The ≤5-DTE rule makes this impossible; check DTE weekly.
- [ ] If somehow still long an ITM call on expiration day: SELL it before
  ~15:30 ET, or call Fidelity and file "do not exercise" before 16:30 ET.
- [ ] Ex-dividend (QQQ: late Mar/Jun/Sep/Dec): long calls lose the dividend in
  price naturally — no action needed; never exercise early to capture a
  dividend under this strategy (violates the spec, consumes unfundable capital).

## 6. Conditions under which NO trade is placed

- Yesterday's close below the +1% entry band (even if today it rallies).
- No strike in the Δ [0.30, 0.62] band affordable within budget.
- Quote filters fail (spread > 10%, OI < 100, crossed, zero bid).
- Market half-days, or any day you cannot supervise the 15:15–15:45 window.
- Equity < 75% of trailing-12m high (drawdown stop — see spec §4).
- Within 24h of a scheduled Fed decision **only if** IV on the target contract
  is > 1.5× its 20-day average (avoid paying event premium; check IV column).

## 7. Emergency procedures

- **Accidental exercise notice** (shares appear): sell the shares at market
  open immediately, call Fidelity to confirm no margin call; log the incident;
  the strategy rules were violated — run the §9 review.
- **Fat-finger (wrong strike/size)**: close the wrong position immediately at
  limit≈mid; do not "manage" it into a different strategy.
- **Halted underlying / no quotes**: do nothing while halted; on reopen, apply
  the normal exit rules.
- **Account drawdown > 25% from high**: flatten everything, stop trading,
  re-run the research review before resuming.

## 8. Weekly 10-minute routine

1. Run `python scripts/tg_cer_signal.py --equity <current>` (or compute MA200
   manually) after Friday's close.
2. Record: equity, open position, DTE, trend state, action for next week.
3. Verify the open call's DTE > 21, else schedule the roll.
4. Confirm core cash ≥ 85% of equity.

## 9. Paper-trading protocol (mandatory before capital)

- Duration: minimum 6 months or 6 completed trades, whichever is longer.
- Execute every rule with a simulated $10k ledger, logging real quotes at
  order time (screenshot the chain).
- Pass criteria (set now): tracking error vs the signal generator ≤ 1 missed
  action; realized fills within the base/conservative band of the backtest's
  assumptions; no rule ambiguities encountered (any ambiguity → fix spec first).
