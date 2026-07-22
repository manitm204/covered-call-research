"""Timestamp-correct daily backtest engine for bear call credit spreads.

Event order within each session (no step may see data from a later step):
  1. 15:30 ET snapshot (configurable): mark open positions, evaluate exit rules,
     execute triggered exits at that same snapshot's quotes.
  2. Same snapshot: scheduled entries (selection + sizing + execution).
  3. Session close: cash-settle positions expiring today (XSP is PM-settled).
  4. End of day: interest accrual (calendar-day ACT/365) and equity snapshot,
     with a hard reconciliation between the ledger and position marks.

The engine only ever queries providers with the current simulation timestamp;
a regression test asserts the query stream is monotone and never in the future.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from xsp_research.backtest import execution
from xsp_research.backtest.ledger import Account, InterestAccruer, Ledger
from xsp_research.config import StrategyConfig
from xsp_research.domain import (
    ExitReason,
    Fill,
    OptionQuote,
    PositionStatus,
    SpreadPosition,
    SpreadQuote,
)
from xsp_research.ingestion.base import OptionsProvider, RatesProvider, UnderlyingProvider
from xsp_research.options.greeks import spread_greeks
from xsp_research.strategy.candidate_selection import select_bear_call_spread
from xsp_research.strategy.exit_rules import ExitContext, evaluate_exit
from xsp_research.strategy.sizing import size_entry

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


@dataclass(slots=True)
class EntryAttempt:
    session: date
    selected: bool
    filled: bool
    reason: str


@dataclass(slots=True)
class TradeRecord:
    trade_id: str
    entry_ts: datetime
    exit_ts: datetime
    expiration: date
    short_strike: float
    long_strike: float
    width: float
    qty: int
    entry_credit: float
    exit_price: float
    exit_reason: str
    dte_at_entry: int
    short_delta_at_entry: float | None
    short_iv_at_entry: float | None
    realized_gross: float
    realized_net: float
    fees: float
    max_loss_dollars: float
    return_on_max_risk: float
    pct_credit_captured: float
    mfe_dollars: float
    mae_dollars: float
    days_in_trade: int


# Optional engine hooks (used by the experiment runner):
# entry gate:   session -> (allow, reason); blocks scheduled entries pre-selection.
# feature hook: (session, snap_ts, chain, selection, spot, rate) -> feature dict,
#               captured per trade_id at fill time. Both see only current-time data.
EntryGate = Callable[[date], tuple[bool, str]]
EntryFeatureHook = Callable[[date, datetime, pl.DataFrame, Any, float, float], dict[str, Any]]


@dataclass(slots=True)
class BacktestResult:
    data_source: str
    config_snapshot: dict[str, Any]
    execution_scenario: str
    trades: list[TradeRecord]
    equity_curve: pl.DataFrame
    entry_attempts: list[EntryAttempt]
    ledger: Ledger
    positions: list[SpreadPosition]
    entry_features: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)

    def trades_frame(self) -> pl.DataFrame:
        if not self.trades:
            return pl.DataFrame()
        return pl.DataFrame([dataclasses.asdict(t) for t in self.trades])

    def entry_features_frame(self) -> pl.DataFrame:
        if not self.entry_features:
            return pl.DataFrame()
        rows = [{"trade_id": tid, **feats} for tid, feats in self.entry_features.items()]
        return pl.DataFrame(rows)


class BacktestEngine:
    def __init__(
        self,
        config: StrategyConfig,
        options: OptionsProvider,
        underlying: UnderlyingProvider,
        rates: RatesProvider,
        execution_scenario: str = "custom",
        *,
        entry_gate: EntryGate | None = None,
        entry_feature_hook: EntryFeatureHook | None = None,
    ) -> None:
        self.cfg = config
        self.options = options
        self.underlying = underlying
        self.rates = rates
        self.scenario = execution_scenario
        self.entry_gate = entry_gate
        self.entry_feature_hook = entry_feature_hook
        self._entry_features: dict[str, dict[str, Any]] = {}
        self._ledger = Ledger()
        self._accruer = InterestAccruer(config.interest, rates)
        self._open: list[SpreadPosition] = []
        self._closed: list[SpreadPosition] = []
        self._booked_mtm: dict[str, float] = {}  # trade_id -> liability dollars on ledger
        self._selection_meta: dict[str, dict[str, Any]] = {}
        self._trades: list[TradeRecord] = []
        self._attempts: list[EntryAttempt] = []
        self._seq = 0

    # ------------------------------------------------------------------ run
    def run(self) -> BacktestResult:
        bt = self.cfg.backtest
        sessions = self.underlying.trading_dates(bt.start, bt.end)
        if not sessions:
            raise ValueError("no trading sessions in backtest window (check data coverage)")

        start_ts = self._session_ts(sessions[0], bt.mark_time_et)
        self._ledger.deposit_capital(start_ts, bt.initial_cash)

        entry_sessions = self._entry_sessions(sessions)
        equity_rows: list[dict[str, Any]] = []

        for i, session in enumerate(sessions):
            snap_ts = self._session_ts(session, bt.mark_time_et)
            chain = self.options.chain(snap_ts)
            spot = self._spot_from_chain(chain) or self.underlying.close(session)

            # 1. Mark open positions and evaluate exits at the snapshot.
            if spot is not None:
                self._mark_and_exit(session, snap_ts, chain, spot)

            # 2. Scheduled entries.
            if session in entry_sessions and spot is not None:
                self._try_entry(session, snap_ts, chain, spot)

            # 3. Cash settlement at the close for today's expirations.
            self._settle_expiring(session)

            # 4. Force-close anything still open on the final session (marked-to-market
            #    exit so the reported equity has no open-liability ambiguity).
            if i == len(sessions) - 1 and self._open:
                self._force_close_remaining(session, snap_ts, chain)

            # 5. Interest accrual and end-of-day snapshot + reconciliation.
            interest = self._accruer.accrue(
                session, self._ledger.cash, self._restricted_collateral()
            )
            if interest:
                self._ledger.accrue_interest(snap_ts, interest)
            equity_rows.append(self._eod_snapshot(session))

        return BacktestResult(
            data_source=self.options.data_source,
            config_snapshot=self.cfg.snapshot(),
            execution_scenario=self.scenario,
            trades=self._trades,
            equity_curve=pl.DataFrame(equity_rows),
            entry_attempts=self._attempts,
            ledger=self._ledger,
            positions=self._closed + self._open,
            entry_features=self._entry_features,
        )

    # ------------------------------------------------------------- internals
    def _session_ts(self, session: date, t) -> datetime:
        return datetime(
            session.year, session.month, session.day, t.hour, t.minute, tzinfo=ET
        ).astimezone(UTC)

    @staticmethod
    def _spot_from_chain(chain: pl.DataFrame) -> float | None:
        if chain.is_empty():
            return None
        return float(chain["underlying_price"][0])

    def _entry_sessions(self, sessions: list[date]) -> set[date]:
        chosen: set[date] = set()
        seen: set[tuple[int, int]] = set()
        for s in sessions:
            key = (
                (s.year, s.month)
                if self.cfg.backtest.entry_frequency.value == "monthly"
                else (s.isocalendar().year, s.isocalendar().week)
            )
            if key not in seen:
                seen.add(key)
                chosen.add(s)
        return chosen

    def _leg_quote(
        self, chain: pl.DataFrame, position: SpreadPosition, leg: str
    ) -> OptionQuote | None:
        contract = position.spread.short_leg if leg == "short" else position.spread.long_leg
        rows = chain.filter(
            (pl.col("expiration") == contract.expiration)
            & (pl.col("strike") == contract.strike)
            & (pl.col("option_type") == contract.option_type.value)
        )
        if rows.is_empty():
            return None
        r = rows.row(0, named=True)
        return OptionQuote(
            contract=contract,
            ts=r["ts"],
            bid=r["bid"],
            ask=r["ask"],
            bid_size=r["bid_size"],
            ask_size=r["ask_size"],
            volume=r["volume"],
            open_interest=r["open_interest"],
            underlying_price=r["underlying_price"],
        )

    def _position_quote(self, chain: pl.DataFrame, position: SpreadPosition) -> SpreadQuote | None:
        sq = self._leg_quote(chain, position, "short")
        lq = self._leg_quote(chain, position, "long")
        if sq is None or lq is None:
            return None
        return SpreadQuote(short_quote=sq, long_quote=lq)

    def _mark_and_exit(
        self, session: date, snap_ts: datetime, chain: pl.DataFrame, spot: float
    ) -> None:
        rate = self.rates.rate(session)
        for position in list(self._open):
            if position.spread.expiration <= session:
                continue  # settles at today's close instead
            sq = self._position_quote(chain, position)
            if sq is None:
                logger.warning(
                    "no quotes for %s at %s; mark is stale", position.position_id, session
                )
                continue
            mark = min(max(sq.close_debit_mid, 0.0), position.spread.width)
            position.record_mark(snap_ts, mark)
            mark_dollars = mark * position.spread.multiplier * position.qty
            self._ledger.mark_spread(
                snap_ts, position.position_id, self._booked_mtm[position.position_id], mark_dollars
            )
            self._booked_mtm[position.position_id] = mark_dollars

            greeks = spread_greeks(sq, spot, rate, 0.0, session)
            ctx = ExitContext(
                position=position,
                ts=snap_ts,
                cost_to_close_mid=mark,
                short_delta=greeks.short_leg_delta if greeks else None,
                dte=(position.spread.expiration - session).days,
            )
            reason = evaluate_exit(self.cfg.exits, ctx)
            if reason is None:
                continue
            report = execution.close_spread(sq, position.qty, self.cfg.execution, self.cfg.costs)
            if not report.filled:
                logger.warning(
                    "exit %s for %s not filled: %s", reason, position.position_id, report.reason
                )
                continue
            self._close_position(
                position, snap_ts, report.price, report.fees_dollars, reason, "close"
            )

    def _try_entry(
        self, session: date, snap_ts: datetime, chain: pl.DataFrame, spot: float
    ) -> None:
        if self.entry_gate is not None:
            allowed, why = self.entry_gate(session)
            if not allowed:
                self._attempts.append(EntryAttempt(session, False, False, f"filtered: {why}"))
                return
        if chain.is_empty():
            self._attempts.append(EntryAttempt(session, False, False, "no chain data"))
            return
        rate = self.rates.rate(session)
        sel = select_bear_call_spread(chain, snap_ts, spot, rate, 0.0, self.cfg.selection)
        if not sel.selected:
            self._attempts.append(
                EntryAttempt(session, False, False, f"selection failed: {sel.reject_summary()}")
            )
            return
        sq = sel.spread_quote
        qty, size_reason = size_entry(
            self.cfg.sizing, sq.spread, self._buying_power(), len(self._open)
        )
        if qty == 0:
            self._attempts.append(EntryAttempt(session, True, False, size_reason))
            return
        report = execution.open_spread(sq, qty, self.cfg.execution, self.cfg.costs)
        if not report.filled:
            self._attempts.append(EntryAttempt(session, True, False, report.reason))
            return

        self._seq += 1
        trade_id = f"T{self._seq:04d}"
        credit_dollars = report.price * sq.spread.multiplier * qty
        position = SpreadPosition(
            position_id=trade_id,
            spread=sq.spread,
            qty=qty,
            entry_fill=Fill(
                ts=snap_ts, price=report.price, qty=qty, fees_dollars=report.fees_dollars
            ),
        )
        self._open.append(position)
        self._booked_mtm[trade_id] = credit_dollars
        self._selection_meta[trade_id] = {
            "dte": sel.dte,
            "short_delta": sel.short_delta,
            "short_iv": sel.short_iv,
        }
        self._ledger.open_spread(snap_ts, trade_id, credit_dollars, report.fees_dollars)
        self._attempts.append(EntryAttempt(session, True, True, f"filled {trade_id}"))
        if self.entry_feature_hook is not None:
            try:
                self._entry_features[trade_id] = self.entry_feature_hook(
                    session, snap_ts, chain, sel, spot, rate
                )
            except Exception as exc:  # feature capture must never kill a backtest
                logger.warning("entry feature hook failed for %s: %s", trade_id, exc)
                self._entry_features[trade_id] = {"feature_error": str(exc)}

    def _settle_expiring(self, session: date) -> None:
        close_ts = self._session_ts(session, time(16, 0))
        for position in list(self._open):
            if position.spread.expiration != session:
                continue
            settle_price = self.underlying.close(session)
            if settle_price is None:
                raise ValueError(f"no settlement price for {session}")
            settle_value = position.spread.settlement_value(settle_price)
            fees = execution.settlement_fees(position.qty, self.cfg.costs)
            self._close_position(
                position, close_ts, settle_value, fees, ExitReason.EXPIRY_SETTLEMENT, "settle"
            )

    def _force_close_remaining(self, session: date, snap_ts: datetime, chain: pl.DataFrame) -> None:
        for position in list(self._open):
            sq = self._position_quote(chain, position)
            if sq is None:
                logger.warning(
                    "cannot price %s at end of backtest; closing at last mark", position.position_id
                )
                price = (
                    position.last_mark if position.last_mark is not None else position.entry_credit
                )
                self._close_position(
                    position, snap_ts, price, 0.0, ExitReason.END_OF_BACKTEST, "close"
                )
                continue
            report = execution.close_spread(sq, position.qty, self.cfg.execution, self.cfg.costs)
            price = (
                report.price
                if report.filled
                else min(max(sq.close_debit_mid, 0.0), position.spread.width)
            )
            fees = report.fees_dollars if report.filled else 0.0
            self._close_position(
                position, snap_ts, price, fees, ExitReason.END_OF_BACKTEST, "close"
            )

    def _close_position(
        self,
        position: SpreadPosition,
        ts: datetime,
        price: float,
        fees: float,
        reason: ExitReason,
        memo: str,
    ) -> None:
        qty, mult = position.qty, position.spread.multiplier
        exit_dollars = price * mult * qty
        self._ledger.close_spread(
            ts,
            position.position_id,
            position.entry_credit * mult * qty,
            self._booked_mtm[position.position_id],
            exit_dollars,
            fees,
            memo,
        )
        self._booked_mtm.pop(position.position_id)
        position.exit_fill = Fill(ts=ts, price=price, qty=qty, fees_dollars=fees)
        position.exit_reason = reason
        position.status = (
            PositionStatus.EXPIRED
            if reason is ExitReason.EXPIRY_SETTLEMENT
            else PositionStatus.CLOSED
        )
        self._open.remove(position)
        self._closed.append(position)
        meta = self._selection_meta.get(position.position_id, {})
        max_loss = position.spread.max_loss_dollars(position.entry_credit, qty)
        net = position.realized_net_dollars()
        self._trades.append(
            TradeRecord(
                trade_id=position.position_id,
                entry_ts=position.entry_fill.ts,
                exit_ts=ts,
                expiration=position.spread.expiration,
                short_strike=position.spread.short_leg.strike,
                long_strike=position.spread.long_leg.strike,
                width=position.spread.width,
                qty=qty,
                entry_credit=position.entry_credit,
                exit_price=price,
                exit_reason=reason.value,
                dte_at_entry=meta.get("dte") or 0,
                short_delta_at_entry=meta.get("short_delta"),
                short_iv_at_entry=meta.get("short_iv"),
                realized_gross=position.realized_gross_dollars(),
                realized_net=net,
                fees=position.entry_fill.fees_dollars + fees,
                max_loss_dollars=max_loss,
                return_on_max_risk=net / max_loss if max_loss > 0 else 0.0,
                pct_credit_captured=(
                    (position.entry_credit - price) / position.entry_credit
                    if position.entry_credit > 0
                    else 0.0
                ),
                mfe_dollars=position.mfe_dollars,
                mae_dollars=position.mae_dollars,
                days_in_trade=position.days_in_trade(),
            )
        )

    def _restricted_collateral(self) -> float:
        return sum(p.collateral_dollars() for p in self._open)

    def _buying_power(self) -> float:
        return self._ledger.cash - self._restricted_collateral()

    def _eod_snapshot(self, session: date) -> dict[str, Any]:
        # Reconciliation: ledger equity must equal cash minus sum of open marks.
        options_mtm = self._ledger.balance(Account.OPTIONS_MTM)
        expected_mtm = -sum(self._booked_mtm.values())
        if abs(options_mtm - expected_mtm) > 1e-4 or not self._ledger.trial_balance_ok():
            raise RuntimeError(f"ledger reconciliation failed on {session}")
        return {
            "date": session,
            "equity": self._ledger.equity,
            "cash": self._ledger.cash,
            "options_mtm": options_mtm,
            "restricted_collateral": self._restricted_collateral(),
            "buying_power": self._buying_power(),
            "open_positions": len(self._open),
            "realized_pnl_gross_cum": self._ledger.realized_pnl_gross(),
            "interest_income_cum": self._ledger.interest_income(),
            "fees_cum": self._ledger.total_fees(),
        }
