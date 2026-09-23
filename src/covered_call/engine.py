"""Event-driven daily backtest engine for a Level-2 cash account.

Position kinds: long options, cash-secured short puts, stock (from assignment),
covered short calls. One session step per 15:30 ET snapshot. Cash-account
semantics throughout: no margin, no naked shorts, collateral reserved for CSPs.

Settlement conventions (documented assumptions):
- Expiration ITM/OTM decided on the official close of the expiry date (FMP
  unadjusted close), not the 15:30 snapshot; auto-exercise threshold $0.01.
- Long options are force-sold at the last snapshot with dte <= FORCE_EXIT_DTE
  (backstop; strategies should exit earlier). If ITM at expiry with no earlier
  exit (no bid), we credit intrinsic minus a $0.05/share haircut (broker
  liquidation), floor 0 — tagged 'expiry_forced'.
- Short put assignment: at expiry if close < strike; early when the snapshot
  extrinsic < $0.03 and ITM (deterministic-conservative hazard).
- Covered call early assignment on ex-div eve when ITM and extrinsic < dividend.
- Dividends on held stock credited on ex-date (simplification, noted).
- All cash (incl. CSP collateral, per Fidelity core-position behavior) accrues
  the 4-week T-bill rate daily (ACT/365 on calendar days between sessions).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Protocol

import pandas as pd

from .execution import FEES, execute, quote_ok
from .market import ChainStore, DailyData
from .selection import Contract, lookup

FORCE_EXIT_DTE = 1
EXPIRY_HAIRCUT = 0.05
AUTO_EXERCISE = 0.01
EARLY_ASSIGN_EXTRINSIC = 0.03


# ------------------------------------------------------------------ positions
@dataclass
class LongOption:
    pos_id: int
    contract: Contract
    contracts: int
    entry_session: date
    entry_price: float
    entry_fee: float
    tag: str = ""


@dataclass
class ShortPut:
    pos_id: int
    contract: Contract
    contracts: int
    entry_session: date
    credit: float
    entry_fee: float
    tag: str = ""

    @property
    def collateral(self) -> float:
        return self.contract.strike * 100.0 * self.contracts


@dataclass
class ShortCall:  # covered by stock only
    pos_id: int
    contract: Contract
    contracts: int
    entry_session: date
    credit: float
    entry_fee: float
    tag: str = ""


@dataclass
class Stock:
    shares: int = 0
    basis: float = 0.0  # per share, from assignment strike


# ------------------------------------------------------------------ orders
@dataclass(frozen=True)
class BuyToOpen:
    contract: Contract
    contracts: int
    tag: str = ""


@dataclass(frozen=True)
class SellToClose:
    pos_id: int
    tag: str = ""


@dataclass(frozen=True)
class SellPutToOpen:  # cash-secured
    contract: Contract
    contracts: int
    tag: str = ""


@dataclass(frozen=True)
class BuyToCloseShort:
    pos_id: int
    tag: str = ""


@dataclass(frozen=True)
class SellCallToOpen:  # covered
    contract: Contract
    contracts: int
    tag: str = ""


@dataclass(frozen=True)
class SellStock:
    shares: int
    tag: str = ""


@dataclass(frozen=True)
class BuyStock:
    shares: int
    tag: str = ""


Order = (BuyToOpen | SellToClose | SellPutToOpen | BuyToCloseShort | SellCallToOpen
         | SellStock | BuyStock)


@dataclass
class Ctx:
    session: date
    chain: pd.DataFrame
    spot: float
    account: "Account"
    daily: DailyData
    rate: float
    brake_mult: float = 1.0  # drawdown brake: 1.0 / 0.5 / 0.0 (spec §4)


class Strategy(Protocol):
    def on_session(self, ctx: Ctx) -> list[Order]: ...


@dataclass
class Account:
    cash: float
    stock: Stock = field(default_factory=Stock)
    long_options: dict[int, LongOption] = field(default_factory=dict)
    short_puts: dict[int, ShortPut] = field(default_factory=dict)
    short_calls: dict[int, ShortCall] = field(default_factory=dict)

    @property
    def reserved(self) -> float:
        return sum(p.collateral for p in self.short_puts.values())

    @property
    def available_cash(self) -> float:
        return self.cash - self.reserved


class Engine:
    def __init__(self, store: ChainStore, daily: DailyData, strategy: Strategy,
                 scenario: str, initial_cash: float = 10_000.0,
                 mark: str = "mid"):
        self.store, self.daily, self.strategy = store, daily, strategy
        self.scenario, self.mark = scenario, mark
        self.acct = Account(cash=initial_cash)
        self._ids = itertools.count(1)
        self.trades: list[dict] = []
        self.equity_rows: list[dict] = []
        self.events: list[dict] = []
        self._open_legs: dict[int, dict] = {}

    # -------------------------------------------------------------- ledger
    def _record_open(self, pos_id, kind, session, contract, contracts, price, fee, tag):
        self._open_legs[pos_id] = dict(
            pos_id=pos_id, kind=kind, open_session=session,
            expiration=getattr(contract, "expiration", None),
            strike=getattr(contract, "strike", None),
            option_type=getattr(contract, "option_type", None),
            contracts=contracts, open_price=price, open_fee=fee, tag=tag,
            open_delta=getattr(contract, "delta", None),
            open_iv=getattr(contract, "iv", None), open_dte=getattr(contract, "dte", None),
        )

    def _record_close(self, pos_id, session, price, fee, how):
        leg = self._open_legs.pop(pos_id)
        n = leg["contracts"]
        if leg["kind"] == "long":
            pnl = (price - leg["open_price"]) * 100 * n - leg["open_fee"] - fee
        else:  # short premium
            pnl = (leg["open_price"] - price) * 100 * n - leg["open_fee"] - fee
        self.trades.append(dict(leg, close_session=session, close_price=price,
                                close_fee=fee, close_how=how, pnl=round(pnl, 2)))

    # -------------------------------------------------------------- steps
    def _accrue_interest(self, prev: date | None, today: date):
        if prev is None:
            return
        days = (today - prev).days
        r = self.daily.rate_on(prev)
        self.acct.cash *= (1.0 + r * days / 365.0)

    def _sweep_between(self, prev: date | None, today: date):
        """Chronological sweep of the gap (prev, today]: for each calendar day,
        credit dividends (entitlement = holdings at prior close, i.e. current
        state), then settle any expiration dated that day."""
        start = (prev + timedelta(days=1)) if prev else today
        d = start
        while d <= today:
            amt = self.daily.dividend_on(d)
            if amt > 0 and self.acct.stock.shares > 0:
                self.acct.cash += amt * self.acct.stock.shares
                self.events.append(dict(session=d, event="dividend", amount=amt,
                                        shares=self.acct.stock.shares))
            if d < today:
                self._settle_expirations_on(d)
            d += timedelta(days=1)
        # anything expired before today not yet settled (e.g. first session)
        self._settle_expirations(today)

    def _settle_expirations_on(self, d: date):
        acct = self.acct
        for coll, handler in ((acct.long_options, self._settle_long),
                              (acct.short_puts, self._settle_short_put),
                              (acct.short_calls, self._settle_short_call)):
            for pid, p in list(coll.items()):
                if p.contract.expiration == d:
                    handler(pid, p)
                    del coll[pid]

    def _settle_long(self, pid, p):
        close = self.daily.close_on(p.contract.expiration) or 0.0
        k, ot = p.contract.strike, p.contract.option_type
        intr = max(close - k, 0.0) if ot == "C" else max(k - close, 0.0)
        value = max(intr - EXPIRY_HAIRCUT, 0.0) if intr >= AUTO_EXERCISE else 0.0
        self.acct.cash += value * 100 * p.contracts
        self._record_close(pid, p.contract.expiration, value, 0.0, "expiry_forced")

    def _settle_short_put(self, pid, p):
        close = self.daily.close_on(p.contract.expiration) or 0.0
        k = p.contract.strike
        if k - close >= AUTO_EXERCISE:
            self.acct.cash -= k * 100 * p.contracts
            self.acct.stock.shares += 100 * p.contracts
            self.acct.stock.basis = k
            self._record_close(pid, p.contract.expiration, 0.0, 0.0, "assigned")
            self.events.append(dict(session=p.contract.expiration, event="put_assignment",
                                    strike=k, contracts=p.contracts))
        else:
            self._record_close(pid, p.contract.expiration, 0.0, 0.0, "expired")

    def _settle_short_call(self, pid, p):
        close = self.daily.close_on(p.contract.expiration) or 0.0
        k = p.contract.strike
        if close - k >= AUTO_EXERCISE:
            self.acct.cash += k * 100 * p.contracts
            self.acct.stock.shares -= 100 * p.contracts
            self._record_close(pid, p.contract.expiration, 0.0, 0.0, "called_away")
            self.events.append(dict(session=p.contract.expiration, event="call_assignment",
                                    strike=k, contracts=p.contracts))
        else:
            self._record_close(pid, p.contract.expiration, 0.0, 0.0, "expired")

    def _settle_expirations(self, today: date):
        """Settle anything expired strictly before today (fallback path)."""
        acct = self.acct
        for coll, handler in ((acct.long_options, self._settle_long),
                              (acct.short_puts, self._settle_short_put),
                              (acct.short_calls, self._settle_short_call)):
            for pid, p in list(coll.items()):
                if p.contract.expiration < today:
                    handler(pid, p)
                    del coll[pid]

    def _early_assignments(self, today: date, chain: pd.DataFrame, spot: float):
        acct = self.acct
        for pid, p in list(acct.short_puts.items()):
            k = p.contract.strike
            if spot >= k:
                continue
            q = lookup(chain, p.contract.expiration, k, "P")
            if q is None:
                continue
            extrinsic = q.mid - (k - spot)
            if extrinsic < EARLY_ASSIGN_EXTRINSIC:
                acct.cash -= k * 100 * p.contracts
                acct.stock.shares += 100 * p.contracts
                acct.stock.basis = k
                self._record_close(pid, today, 0.0, 0.0, "early_assigned")
                self.events.append(dict(session=today, event="early_put_assignment", strike=k))
                del acct.short_puts[pid]
        # covered-call ex-div early assignment
        next_div = self.daily.dividends[self.daily.dividends.ex_date > pd.Timestamp(today)]
        ex1 = next_div.ex_date.min() if len(next_div) else None
        if ex1 is not None and (ex1 - pd.Timestamp(today)).days <= 1:
            amt = float(next_div.loc[next_div.ex_date == ex1, "amount"].sum())
            for pid, p in list(acct.short_calls.items()):
                k = p.contract.strike
                if spot <= k or p.contract.expiration <= today:
                    continue
                q = lookup(chain, p.contract.expiration, k, "C")
                if q is None:
                    continue
                extrinsic = q.mid - (spot - k)
                if extrinsic < amt:
                    acct.cash += k * 100 * p.contracts
                    acct.stock.shares -= 100 * p.contracts
                    self._record_close(pid, today, 0.0, 0.0, "called_away_exdiv")
                    self.events.append(dict(session=today, event="exdiv_call_assignment", strike=k))
                    del acct.short_calls[pid]

    def _force_exits(self, today: date, chain: pd.DataFrame) -> list[Order]:
        orders = []
        for pid, p in self.acct.long_options.items():
            dte = (p.contract.expiration - today).days
            if dte <= FORCE_EXIT_DTE:
                orders.append(SellToClose(pos_id=pid, tag="force_dte"))
        return orders

    # -------------------------------------------------------------- orders
    def _exec_order(self, o: Order, today: date, chain: pd.DataFrame, spot: float):
        acct = self.acct
        if isinstance(o, BuyToOpen):
            c = lookup(chain, o.contract.expiration, o.contract.strike, o.contract.option_type)
            if c is None or not quote_ok(c.bid, c.ask, for_sell=False):
                return
            f = execute("buy", c.bid, c.ask, o.contracts, self.scenario)
            cost = f.price * 100 * o.contracts + f.fee
            if cost > acct.available_cash:
                return
            acct.cash -= cost
            pid = next(self._ids)
            cc = Contract(c.expiration, c.strike, c.option_type, c.bid, c.ask, c.mid,
                          c.dte, o.contract.delta, o.contract.iv, c.open_interest)
            acct.long_options[pid] = LongOption(pid, cc, o.contracts, today, f.price, f.fee, o.tag)
            self._record_open(pid, "long", today, cc, o.contracts, f.price, f.fee, o.tag)
        elif isinstance(o, SellToClose):
            p = acct.long_options.get(o.pos_id)
            if p is None:
                return
            c = lookup(chain, p.contract.expiration, p.contract.strike, p.contract.option_type)
            if c is None or not quote_ok(c.bid, c.ask, for_sell=True):
                return  # cannot exit today; backstop/expiry path will handle
            f = execute("sell", c.bid, c.ask, p.contracts, self.scenario)
            acct.cash += f.price * 100 * p.contracts - f.fee
            self._record_close(o.pos_id, today, f.price, f.fee, o.tag or "close")
            del acct.long_options[o.pos_id]
        elif isinstance(o, SellPutToOpen):
            c = lookup(chain, o.contract.expiration, o.contract.strike, "P")
            if c is None or not quote_ok(c.bid, c.ask, for_sell=True):
                return
            need = o.contract.strike * 100 * o.contracts
            f = execute("sell", c.bid, c.ask, o.contracts, self.scenario)
            credit = f.price * 100 * o.contracts - f.fee
            if need > acct.available_cash + credit:
                return  # cannot secure the put
            acct.cash += credit
            pid = next(self._ids)
            cc = Contract(c.expiration, c.strike, "P", c.bid, c.ask, c.mid, c.dte,
                          o.contract.delta, o.contract.iv, c.open_interest)
            acct.short_puts[pid] = ShortPut(pid, cc, o.contracts, today, f.price, f.fee, o.tag)
            self._record_open(pid, "short", today, cc, o.contracts, f.price, f.fee, o.tag)
        elif isinstance(o, BuyToCloseShort):
            p = acct.short_puts.get(o.pos_id) or acct.short_calls.get(o.pos_id)
            if p is None:
                return
            c = lookup(chain, p.contract.expiration, p.contract.strike, p.contract.option_type)
            if c is None or not quote_ok(c.bid, c.ask, for_sell=False):
                return
            f = execute("buy", c.bid, c.ask, p.contracts, self.scenario)
            acct.cash -= f.price * 100 * p.contracts + f.fee
            self._record_close(o.pos_id, today, f.price, f.fee, o.tag or "close")
            acct.short_puts.pop(o.pos_id, None)
            acct.short_calls.pop(o.pos_id, None)
        elif isinstance(o, SellCallToOpen):
            covered = self.acct.stock.shares - 100 * sum(
                p.contracts for p in acct.short_calls.values())
            if o.contracts * 100 > covered:
                return  # never uncovered
            c = lookup(chain, o.contract.expiration, o.contract.strike, "C")
            if c is None or not quote_ok(c.bid, c.ask, for_sell=True):
                return
            f = execute("sell", c.bid, c.ask, o.contracts, self.scenario)
            acct.cash += f.price * 100 * o.contracts - f.fee
            pid = next(self._ids)
            cc = Contract(c.expiration, c.strike, "C", c.bid, c.ask, c.mid, c.dte,
                          o.contract.delta, o.contract.iv, c.open_interest)
            acct.short_calls[pid] = ShortCall(pid, cc, o.contracts, today, f.price, f.fee, o.tag)
            self._record_open(pid, "short", today, cc, o.contracts, f.price, f.fee, o.tag)
        elif isinstance(o, SellStock):
            n = min(o.shares, acct.stock.shares)
            if n <= 0:
                return
            covered_needed = 100 * sum(p.contracts for p in acct.short_calls.values())
            if acct.stock.shares - n < covered_needed:
                return  # would uncover a short call
            px = spot  # marketable at snapshot; slippage folded into spread scenarios elsewhere
            acct.cash += px * n
            self.trades.append(dict(pos_id=None, kind="stock", open_session=None,
                                    close_session=today, contracts=n, open_price=acct.stock.basis,
                                    close_price=px, pnl=round((px - acct.stock.basis) * n, 2),
                                    close_how=o.tag or "sell_stock", tag=o.tag))
            acct.stock.shares -= n
        elif isinstance(o, BuyStock):
            cost = spot * o.shares
            if cost > acct.available_cash or o.shares <= 0:
                return
            new_shares = acct.stock.shares + o.shares
            acct.stock.basis = ((acct.stock.basis * acct.stock.shares + spot * o.shares)
                                 / new_shares) if new_shares else 0.0
            acct.stock.shares = new_shares
            acct.cash -= cost
            self.trades.append(dict(pos_id=None, kind="stock", open_session=today,
                                    close_session=None, contracts=o.shares, open_price=spot,
                                    close_price=None, pnl=None,
                                    close_how=o.tag or "buy_stock", tag=o.tag))

    # -------------------------------------------------------------- marking
    def _mark(self, today: date, chain: pd.DataFrame, spot: float):
        acct = self.acct
        val = acct.cash + acct.stock.shares * spot
        liq = val
        for p in acct.long_options.values():
            q = lookup(chain, p.contract.expiration, p.contract.strike, p.contract.option_type)
            m = q.mid if q else 0.0
            b = q.bid if q else 0.0
            val += m * 100 * p.contracts
            liq += b * 100 * p.contracts
        for p in list(acct.short_puts.values()) + list(acct.short_calls.values()):
            q = lookup(chain, p.contract.expiration, p.contract.strike, p.contract.option_type)
            m = q.mid if q else 0.0
            a = q.ask if q else 0.0
            val -= m * 100 * p.contracts
            liq -= a * 100 * p.contracts
        self.equity_rows.append(dict(session=today, equity=round(val, 2),
                                     equity_liq=round(liq, 2), cash=round(acct.cash, 2),
                                     reserved=round(acct.reserved, 2),
                                     shares=acct.stock.shares,
                                     n_long=len(acct.long_options),
                                     n_short=len(acct.short_puts) + len(acct.short_calls)))

    # -------------------------------------------------------------- main loop
    def run(self, start: date | None = None, end: date | None = None) -> "Result":
        prev: date | None = None
        for today in self.store.sessions():
            if start and today < start:
                continue
            if end and today > end:
                break
            chain = self.store.chain(today)
            if not len(chain):
                continue
            spot = float(chain["underlying_price"].iloc[0])
            self._accrue_interest(prev, today)
            self._sweep_between(prev, today)
            self._early_assignments(today, chain, spot)
            for o in self._force_exits(today, chain):
                self._exec_order(o, today, chain, spot)
            brake = 1.0
            if self.equity_rows:
                hist = [r["equity"] for r in self.equity_rows[-252:]]
                high = max(hist)
                cur = hist[-1]
                if cur < 0.75 * high:
                    brake = 0.0
                elif cur < 0.85 * high:
                    brake = 0.5
            ctx = Ctx(today, chain, spot, self.acct, self.daily,
                      self.daily.rate_on(today), brake_mult=brake)
            for o in self.strategy.on_session(ctx):
                self._exec_order(o, today, chain, spot)
            if self.acct.cash < -1e-6 and not self.acct.stock.shares:
                raise RuntimeError(f"negative cash {self.acct.cash} on {today}")
            self._mark(today, chain, spot)
            prev = today
        return Result(pd.DataFrame(self.equity_rows), pd.DataFrame(self.trades),
                      pd.DataFrame(self.events))


@dataclass
class Result:
    equity: pd.DataFrame
    trades: pd.DataFrame
    events: pd.DataFrame
