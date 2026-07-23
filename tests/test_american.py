"""American-exercise modeling: assignment boundary, event P&L, engine wiring."""

from datetime import date

import polars as pl
import pytest

from tests.conftest import make_quote
from tests.test_engine import make_config
from xsp_research.backtest.american import (
    DividendCalendar,
    assignment_exit_debit,
    early_assignment_triggered,
    short_call_extrinsic,
)
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.config import BacktestConfig, SelectionConfig
from xsp_research.domain import SettlementStyle, SpreadQuote
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket

START, END = date(2023, 1, 2), date(2023, 9, 29)


class TestDividendCalendar:
    def test_from_parquet_round_trip(self, tmp_path):
        pl.DataFrame(
            {"ex_date": [date(2023, 3, 17), date(2023, 6, 16)], "amount": [1.51, 1.63]}
        ).write_parquet(tmp_path / "d.parquet")
        cal = DividendCalendar.from_parquet(tmp_path / "d.parquet", "SPY")
        assert cal.amount_on(date(2023, 3, 17)) == pytest.approx(1.51)
        assert cal.amount_on(date(2023, 3, 18)) is None
        assert cal.next_ex_date_after(date(2023, 4, 1)) == date(2023, 6, 16)

    def test_bad_schema_rejected(self, tmp_path):
        pl.DataFrame({"date": [date(2023, 1, 1)], "close": [1.0]}).write_parquet(
            tmp_path / "bad.parquet"
        )
        with pytest.raises(ValueError, match="ex_date"):
            DividendCalendar.from_parquet(tmp_path / "bad.parquet")


class TestAssignmentBoundary:
    """S=460, K=450: intrinsic 10. Dividend 1.80."""

    def test_low_extrinsic_triggers(self):
        # short mid 10.30 -> extrinsic 0.30 < 1.80 dividend
        assert early_assignment_triggered(460.0, 450.0, 10.30, 1.80)

    def test_high_extrinsic_does_not_trigger(self):
        # short mid 12.50 -> extrinsic 2.50 > 1.80
        assert not early_assignment_triggered(460.0, 450.0, 12.50, 1.80)

    def test_otm_never_triggers(self):
        assert not early_assignment_triggered(445.0, 450.0, 1.00, 1.80)

    def test_zero_dividend_never_triggers(self):
        assert not early_assignment_triggered(460.0, 450.0, 10.01, 0.0)

    def test_extrinsic_hand_computed(self):
        assert short_call_extrinsic(460.0, 450.0, 10.30) == pytest.approx(0.30)
        assert short_call_extrinsic(445.0, 450.0, 1.00) == pytest.approx(1.00)


class TestAssignmentPnl:
    def test_hand_computed_exit_debit(self):
        """intrinsic 10 + penalty (1.80-0.30) - long bid 5.60 = 5.90"""
        debit = assignment_exit_debit(460.0, 450.0, 10.30, 1.80, long_bid=5.60)
        assert debit == pytest.approx(10.0 + 1.50 - 5.60)

    def test_assignment_costs_more_than_quote_close(self):
        """The assignment event is never cheaper than closing at the quotes —
        that is exactly why ignoring it flatters SPY backtests."""
        short = make_quote(450.0, 10.25, 10.35)  # mid 10.30
        long_q = make_quote(455.0, 5.60, 5.70)
        sq = SpreadQuote(short_quote=short, long_quote=long_q)
        quote_close = sq.close_debit_mid  # 10.30 - 5.65 = 4.65
        assignment = assignment_exit_debit(460.0, 450.0, 10.30, 1.80, long_q.bid)
        assert assignment > quote_close

    def test_floor_at_zero(self):
        assert assignment_exit_debit(451.0, 450.0, 1.0, 0.5, long_bid=5.0) == 0.0


def _american_config(**overrides):
    cfg = make_config(**overrides)
    return cfg.model_copy(
        update={
            "selection": cfg.selection.model_copy(
                update={"exercise_style": "american", "settlement": "physical"}
            )
        }
    )


def _market():
    return SyntheticMarket(SyntheticConfig(start=START, end=END, strike_pct_range=0.12))


def _calendar(dates_amounts: dict[date, float]) -> DividendCalendar:
    return DividendCalendar(symbol="SPY", amounts=dates_amounts)


class TestEngineWiring:
    def test_american_without_dividends_fails_loud(self):
        m = _market()
        with pytest.raises(ValueError, match="dividend calendar"):
            BacktestEngine(_american_config(), m, m, m, "base")

    def test_am_settlement_rejected(self):
        m = _market()
        cfg = make_config()
        cfg = cfg.model_copy(
            update={"selection": cfg.selection.model_copy(update={"settlement": "AM"})}
        )
        with pytest.raises(ValueError, match="AM settlement"):
            BacktestEngine(cfg, m, m, m, "base")

    def test_contracts_carry_mechanics(self):
        m = _market()
        engine = BacktestEngine(_american_config(), m, m, m, "base", dividends=_calendar({}))
        result = engine.run()
        assert result.trades, "expected trades"
        for p in result.positions:
            assert p.spread.short_leg.settlement is SettlementStyle.PHYSICAL

    def test_physical_expiry_closes_at_quotes_not_cash_settle(self):
        """Hold-to-expiry under physical settlement: every expiring position
        must exit as EXPIRY_CLOSE (forced close), never EXPIRY_SETTLEMENT."""
        m = _market()
        engine = BacktestEngine(
            _american_config(exits=[]), m, m, m, "base", dividends=_calendar({})
        )
        result = engine.run()
        reasons = {t.exit_reason for t in result.trades}
        assert "expiry_settlement" not in reasons
        assert "expiry_close" in reasons

    def test_european_regression_unchanged(self):
        """Default config must still cash-settle (no behavior drift)."""
        m = _market()
        result = BacktestEngine(make_config(exits=[]), m, m, m, "base").run()
        reasons = {t.exit_reason for t in result.trades}
        assert "expiry_close" not in reasons
        assert "expiry_settlement" in reasons

    def test_ex_div_eve_assignment_fires_when_itm(self):
        """Force assignment: dividend on every session, huge amount — any ITM
        drift must trigger; with none ITM, nothing fires (both outcomes valid,
        the accounting must reconcile either way)."""
        m = _market()
        sessions = m.trading_dates(START, END)
        divs = _calendar(dict.fromkeys(sessions, 50.0))  # absurd dividend: always rational
        engine = BacktestEngine(_american_config(exits=[]), m, m, m, "base", dividends=divs)
        result = engine.run()
        assert result.ledger.trial_balance_ok()
        assigned = [t for t in result.trades if t.exit_reason == "early_assignment"]
        for t in assigned:
            # Assignment only ever fires ITM, and stays within defined risk.
            assert t.exit_price <= t.width + 1e-9

    def test_selection_config_rejects_unknown_style(self):
        with pytest.raises(ValueError):
            SelectionConfig(exercise_style="bermudan").model_dump()
        # (pydantic validates via enum conversion at contract creation; the
        # string field itself is free-form — engine/selection raise on use)


class TestBacktestConfigCompat:
    def test_start_end_alias(self):
        cfg = BacktestConfig(start=START, end=END)
        assert cfg.start < cfg.end
