"""Unit tests for the Level-2 engine: fills, selection, accounting, expiration,
assignment, dividends, interest — penny-exact where deterministic."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from covered_call.engine import (BuyToOpen, Engine, SellCallToOpen,
                                    SellPutToOpen, SellToClose)
from covered_call.execution import FEES, fill_price, quote_ok
from covered_call.selection import Contract, pick_by_delta
from covered_call.signals import build_signals


# ------------------------------------------------------------------ fixtures
def make_chain(session: date, spot: float, rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        bid, ask = r["bid"], r["ask"]
        mid = (bid + ask) / 2
        recs.append(dict(
            session=session, expiration=r["exp"], strike=r["strike"],
            option_type=r["type"], bid=bid, ask=ask, mid=mid,
            rel_spread=(ask - bid) / mid if mid > 0 else np.inf,
            dte=(r["exp"] - session).days, open_interest=r.get("oi", 1000),
            bid_size=10, ask_size=10, underlying_price=spot,
        ))
    return pd.DataFrame(recs)


class FakeStore:
    def __init__(self, days: dict[date, pd.DataFrame]):
        self._days = days

    def sessions(self):
        return sorted(self._days)

    def chain(self, session):
        return self._days.get(session, pd.DataFrame())


class FakeDaily:
    def __init__(self, closes: dict[date, float], rate=0.0, divs=None):
        self._closes = closes
        self._rate = rate
        self.dividends = pd.DataFrame(divs or [], columns=["ex_date", "amount"])
        if len(self.dividends):
            self.dividends["ex_date"] = pd.to_datetime(self.dividends["ex_date"])

    def close_on(self, d):
        ks = [k for k in self._closes if k <= d]
        return self._closes[max(ks)] if ks else None

    def rate_on(self, d):
        return self._rate

    def dividend_on(self, d):
        if not len(self.dividends):
            return 0.0
        m = self.dividends[self.dividends.ex_date == pd.Timestamp(d)]
        return float(m.amount.sum()) if len(m) else 0.0


class Script:
    """Strategy that emits a pre-scripted order list per session."""

    def __init__(self, script):
        self.script = script

    def on_session(self, ctx):
        return self.script.get(ctx.session, [])


D1, D2, D3 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
EXP = date(2024, 1, 19)


def contract(exp=EXP, k=100.0, t="C", bid=2.0, ask=2.10):
    return Contract(exp, k, t, bid, ask, (bid + ask) / 2, (exp - D1).days, None, None, 1000)


# ------------------------------------------------------------------ execution
def test_fill_prices_all_scenarios():
    bid, ask = 2.00, 2.20  # mid 2.10, half 0.10
    assert fill_price("buy", bid, ask, "optimistic") == pytest.approx(2.10)
    assert fill_price("buy", bid, ask, "base") == pytest.approx(2.15)
    assert fill_price("buy", bid, ask, "conservative") == pytest.approx(2.21)
    assert fill_price("sell", bid, ask, "optimistic") == pytest.approx(2.10)
    assert fill_price("sell", bid, ask, "base") == pytest.approx(2.05)
    assert fill_price("sell", bid, ask, "conservative") == pytest.approx(1.99)


def test_quote_validity():
    assert not quote_ok(2.2, 2.0, for_sell=False)   # crossed
    assert not quote_ok(0.0, 0.05, for_sell=True)   # zero bid for sell
    assert quote_ok(0.0, 0.05, for_sell=False)      # zero bid ok for buy
    assert not quote_ok(1.0, 1.1, for_sell=True, bid_size=0)


# ------------------------------------------------------------------ selection
def test_pick_by_delta_selects_nearest_delta_and_respects_filters():
    session, spot = D1, 100.0
    exp_near, exp_far = date(2024, 1, 12), date(2024, 2, 16)
    rows = []
    for k in [80, 90, 95, 100, 105, 110, 120]:
        # rough plausible call quotes at ~20 vol
        iv = 0.20
        from covered_call.domain import OptionType
        from covered_call.options.black_scholes import bs_price, year_fraction
        for exp in (exp_near, exp_far):
            t = year_fraction(session, exp)
            px = bs_price(spot, k, t, iv, 0.03, 0.013, OptionType.CALL)
            if px < 0.03:
                continue
            rows.append(dict(exp=exp, strike=float(k), type="C",
                             bid=round(px * 0.98, 2), ask=round(px * 1.02, 2)))
    # one illiquid row that would otherwise win: OI below floor
    rows.append(dict(exp=exp_far, strike=102.0, type="C", bid=3.0, ask=3.1, oi=5))
    chain = make_chain(session, spot, rows)
    c = pick_by_delta(chain, spot, session, 0.03, option_type="C", target_delta=0.50,
                      dte_target=45, dte_lo=20, dte_hi=70)
    assert c is not None
    assert c.expiration == exp_far      # nearest to 45 within [20,70]
    assert c.strike in (100.0,)         # ATM ~ 0.5 delta, illiquid 102 excluded
    assert abs(abs(c.delta) - 0.5) < 0.12


# ------------------------------------------------------------------ engine paths
def test_long_round_trip_accounting_to_the_penny():
    ch1 = make_chain(D1, 100, [dict(exp=EXP, strike=100.0, type="C", bid=2.00, ask=2.10)])
    ch2 = make_chain(D2, 103, [dict(exp=EXP, strike=100.0, type="C", bid=4.00, ask=4.10)])
    eng = Engine(FakeStore({D1: ch1, D2: ch2}), FakeDaily({}), Script({
        D1: [BuyToOpen(contract(), 2)],
        D2: [SellToClose(1)],
    }), scenario="base", initial_cash=10_000)
    res = eng.run()
    # base buy: mid 2.05 + half*0.5 = 2.075; cost = 2*100*2.075 + 1.40 = 416.40
    # base sell: mid 4.05 - 0.025 = 4.025; proceeds = 805.00 - 1.40 = 803.60
    assert eng.acct.cash == pytest.approx(10_000 - 416.40 + 803.60)
    t = res.trades.iloc[0]
    assert t.pnl == pytest.approx((4.025 - 2.075) * 200 - 2.80)


def test_long_otm_expires_worthless_and_itm_forced_settlement():
    exp = date(2024, 1, 5)
    ch1 = make_chain(D1, 100, [
        dict(exp=exp, strike=105.0, type="C", bid=0.50, ask=0.55),
        dict(exp=exp, strike=95.0, type="C", bid=5.20, ask=5.40),
    ])
    ch_next = make_chain(date(2024, 1, 8), 101, [dict(exp=date(2024, 2, 16), strike=100.0,
                                                      type="C", bid=3.0, ask=3.2)])
    daily = FakeDaily({exp: 101.0})
    # buy both; engine force-sells at dte<=1 but D1->exp is 3 days and there is no
    # snapshot on the last day, so settlement happens via expiry path
    eng = Engine(FakeStore({D1: ch1, date(2024, 1, 8): ch_next}), daily, Script({
        D1: [BuyToOpen(contract(exp=exp, k=105.0, bid=0.50, ask=0.55), 1),
             BuyToOpen(contract(exp=exp, k=95.0, bid=5.20, ask=5.40), 1)],
    }), scenario="optimistic", initial_cash=10_000)
    res = eng.run()
    tr = res.trades.set_index("strike")
    assert tr.loc[105.0, "close_price"] == 0.0                      # OTM worthless
    assert tr.loc[95.0, "close_price"] == pytest.approx(6.0 - 0.05)  # intrinsic - haircut
    assert tr.loc[95.0, "close_how"] == "expiry_forced"


def test_csp_assignment_and_collateral():
    exp = date(2024, 1, 5)
    put = dict(exp=exp, strike=100.0, type="P", bid=1.50, ask=1.60)
    ch1 = make_chain(D1, 101, [put])
    after = date(2024, 1, 8)
    ch2 = make_chain(after, 96, [dict(exp=date(2024, 2, 16), strike=95.0, type="C",
                                      bid=2.0, ask=2.2)])
    daily = FakeDaily({exp: 96.0})
    eng = Engine(FakeStore({D1: ch1, after: ch2}), daily, Script({
        D1: [SellPutToOpen(contract(exp=exp, k=100.0, t="P", bid=1.50, ask=1.60), 1)],
    }), scenario="optimistic", initial_cash=10_000)
    res = eng.run()
    # credit 1.55*100 - 0.70; assignment: -10,000 cash, +100 shares basis 100
    assert eng.acct.stock.shares == 100
    assert eng.acct.cash == pytest.approx(10_000 + 155 - 0.70 - 10_000)
    assert res.events.iloc[0]["event"] == "put_assignment"
    # reserve was released exactly once
    assert eng.acct.reserved == 0


def test_csp_rejected_when_not_fundable():
    put = dict(exp=EXP, strike=200.0, type="P", bid=1.0, ask=1.1)
    ch1 = make_chain(D1, 210, [put])
    eng = Engine(FakeStore({D1: ch1}), FakeDaily({}), Script({
        D1: [SellPutToOpen(contract(exp=EXP, k=200.0, t="P", bid=1.0, ask=1.1), 1)],
    }), scenario="base", initial_cash=10_000)
    eng.run()  # 200*100 = 20k > 10k cash -> rejected
    assert not eng.acct.short_puts and eng.acct.cash == 10_000


def test_early_assignment_when_extrinsic_gone():
    exp = date(2024, 2, 16)
    ch1 = make_chain(D1, 100, [dict(exp=exp, strike=95.0, type="P", bid=0.90, ask=1.00)])
    # next day: spot crashes to 80, put trades at parity (extrinsic ~ 0)
    ch2 = make_chain(D2, 80, [dict(exp=exp, strike=95.0, type="P", bid=14.98, ask=15.02)])
    eng = Engine(FakeStore({D1: ch1, D2: ch2}), FakeDaily({}), Script({
        D1: [SellPutToOpen(contract(exp=exp, k=95.0, t="P", bid=0.90, ask=1.00), 1)],
    }), scenario="optimistic", initial_cash=10_000)
    res = eng.run()
    assert eng.acct.stock.shares == 100
    assert "early" in res.trades.iloc[0]["close_how"]


def test_covered_call_cycle_and_dividend():
    exp = date(2024, 1, 19)
    # start with assigned stock: engineer via CSP assignment first? Simpler: seed stock
    ch1 = make_chain(D1, 100, [dict(exp=exp, strike=105.0, type="C", bid=1.00, ask=1.10)])
    after = date(2024, 1, 22)
    ch2 = make_chain(after, 108, [dict(exp=date(2024, 2, 16), strike=110.0, type="C",
                                       bid=1.0, ask=1.1)])
    daily = FakeDaily({exp: 108.0}, divs=[dict(ex_date="2024-01-10", amount=0.50)])
    eng = Engine(FakeStore({D1: ch1, after: ch2}), daily, Script({
        D1: [SellCallToOpen(contract(exp=exp, k=105.0, bid=1.00, ask=1.10), 1)],
    }), scenario="optimistic", initial_cash=0)
    eng.acct.stock.shares = 100
    eng.acct.stock.basis = 100.0
    res = eng.run()
    # credit 105 - fee 0.70 + dividend 50 + called away 10,500
    assert eng.acct.stock.shares == 0
    assert eng.acct.cash == pytest.approx(105 - 0.70 + 50 + 10_500)
    assert set(res.events.event) == {"dividend", "call_assignment"}


def test_uncovered_call_rejected():
    ch1 = make_chain(D1, 100, [dict(exp=EXP, strike=105.0, type="C", bid=1.0, ask=1.1)])
    eng = Engine(FakeStore({D1: ch1}), FakeDaily({}), Script({
        D1: [SellCallToOpen(contract(exp=EXP, k=105.0, bid=1.0, ask=1.1), 1)],
    }), scenario="base", initial_cash=10_000)
    eng.run()
    assert not eng.acct.short_calls  # no shares -> rejected


def test_interest_accrual_exact():
    ch1 = make_chain(D1, 100, [dict(exp=EXP, strike=100.0, type="C", bid=1.0, ask=1.1)])
    ch2 = make_chain(date(2024, 1, 12), 100, [dict(exp=EXP, strike=100.0, type="C",
                                                   bid=1.0, ask=1.1)])
    eng = Engine(FakeStore({D1: ch1, date(2024, 1, 12): ch2}), FakeDaily({}, rate=0.05),
                 Script({}), scenario="base", initial_cash=10_000)
    eng.run()
    assert eng.acct.cash == pytest.approx(10_000 * (1 + 0.05 * 10 / 365))


def test_signals_have_no_lookahead():
    sig = build_signals("SPY")
    px = pd.read_parquet("data/normalized/prices_long/SPY.parquet")
    px["date"] = pd.to_datetime(px.date)
    px = px.set_index("date").adjClose
    d = pd.Timestamp("2020-03-16")  # -12% day
    # signal dated D must NOT reflect D's own return
    assert sig.loc[d, "ret1"] == pytest.approx(px.loc["2020-03-13"] / px.loc["2020-03-12"] - 1)
    # and the 200d MA dated D uses data only through D-1
    manual = px.loc[:"2020-03-13"].rolling(200).mean().iloc[-1]
    assert sig.loc[d, "ma200"] == pytest.approx(manual)
