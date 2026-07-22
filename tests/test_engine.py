"""End-to-end engine tests on labeled synthetic data.

These validate accounting, invariants, determinism, and the absence of
look-ahead — they say nothing about real-world profitability.
"""

from datetime import date, datetime

import pytest

from xsp_research.backtest.engine import BacktestEngine
from xsp_research.config import (
    BacktestConfig,
    ExitRuleConfig,
    InterestConfig,
    SizingConfig,
    StrategyConfig,
)
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket

START, END = date(2023, 1, 2), date(2023, 9, 29)


def make_config(**overrides) -> StrategyConfig:
    base = dict(
        name="test",
        exits=[
            ExitRuleConfig(kind="profit_target", profit_target_pct=0.50),
            ExitRuleConfig(kind="stop_loss", stop_loss_multiple=2.0),
        ],
        interest=InterestConfig(rate_source="fixed", fixed_rate=0.04),
        sizing=SizingConfig(contracts_per_entry=1, max_open_positions=2),
        backtest=BacktestConfig(start=START, end=END, initial_cash=100_000.0),
    )
    base.update(overrides)
    return StrategyConfig.model_validate(base)


@pytest.fixture(scope="module")
def market():
    return SyntheticMarket(SyntheticConfig(start=START, end=END, strike_pct_range=0.12))


@pytest.fixture(scope="module")
def result(market):
    engine = BacktestEngine(make_config(), market, market, market, "base")
    return engine.run()


class TestBasicRun:
    def test_trades_happen(self, result):
        assert len(result.trades) >= 5  # monthly entries over 9 months

    def test_data_source_labeled_synthetic(self, result):
        assert result.data_source == "synthetic"

    def test_all_positions_terminal(self, result):
        from xsp_research.domain import PositionStatus

        assert all(p.status is not PositionStatus.OPEN for p in result.positions)


class TestAccountingIdentities:
    def test_final_equity_decomposition(self, result):
        """equity = initial + realized gross - fees + interest (nothing open)."""
        led = result.ledger
        expected = 100_000.0 + led.realized_pnl_gross() - led.total_fees() + led.interest_income()
        final_equity = result.equity_curve["equity"][-1]
        assert final_equity == pytest.approx(expected, abs=1e-4)

    def test_trade_records_match_ledger(self, result):
        total_net = sum(t.realized_net for t in result.trades)
        led = result.ledger
        assert total_net == pytest.approx(led.realized_pnl_gross() - led.total_fees(), abs=1e-4)

    def test_interest_accrued_and_separated(self, result):
        assert result.ledger.interest_income() > 0.0

    def test_trial_balance(self, result):
        assert result.ledger.trial_balance_ok()


class TestTradeInvariants:
    def test_prices_within_arbitrage_band(self, result):
        for t in result.trades:
            assert 0.0 < t.entry_credit < t.width
            assert -1e-9 <= t.exit_price <= t.width + 1e-9

    def test_realized_gross_formula(self, result):
        for t in result.trades:
            assert t.realized_gross == pytest.approx(
                (t.entry_credit - t.exit_price) * 100 * t.qty, abs=1e-6
            )

    def test_max_loss_never_exceeded_net_of_fees(self, result):
        for t in result.trades:
            assert t.realized_net >= -(t.max_loss_dollars + t.fees) - 1e-6

    def test_dte_within_configured_window(self, result):
        for t in result.trades:
            assert 25 <= t.dte_at_entry <= 35

    def test_marks_bounded_by_width(self, result):
        for p in result.positions:
            for _, mark in p.marks:
                assert 0.0 <= mark <= p.spread.width + 1e-9

    def test_exit_after_entry(self, result):
        for t in result.trades:
            assert t.exit_ts >= t.entry_ts


class TestCollateralAndSizing:
    def test_buying_power_never_negative(self, result):
        assert (result.equity_curve["buying_power"] >= -1e-6).all()

    def test_collateral_restricted_when_open(self, result):
        df = result.equity_curve
        open_days = df.filter(df["open_positions"] > 0)
        assert (open_days["restricted_collateral"] > 0).all()


class TestDeterminism:
    def test_same_seed_same_result(self, result):
        market2 = SyntheticMarket(SyntheticConfig(start=START, end=END, strike_pct_range=0.12))
        engine2 = BacktestEngine(make_config(), market2, market2, market2, "base")
        result2 = engine2.run()
        assert result.equity_curve["equity"].to_list() == result2.equity_curve["equity"].to_list()
        assert [t.trade_id for t in result.trades] == [t.trade_id for t in result2.trades]


class RecordingMarket:
    """Wraps the synthetic market, recording every chain query timestamp."""

    def __init__(self, inner: SyntheticMarket) -> None:
        self._inner = inner
        self.chain_queries: list[datetime] = []

    @property
    def data_source(self) -> str:
        return self._inner.data_source

    def chain(self, as_of):
        self.chain_queries.append(as_of)
        return self._inner.chain(as_of)

    def expirations(self, as_of):
        return self._inner.expirations(as_of)

    def trading_dates(self, s, e):
        return self._inner.trading_dates(s, e)

    def close(self, d):
        return self._inner.close(d)

    def closes(self, s, e):
        return self._inner.closes(s, e)

    def rate(self, d):
        return self._inner.rate(d)


class TestNoLookahead:
    def test_chain_queries_monotone_and_in_window(self, market):
        rec = RecordingMarket(market)
        cfg = make_config(
            backtest=BacktestConfig(start=START, end=date(2023, 3, 31), initial_cash=100_000.0)
        )
        BacktestEngine(cfg, rec, rec, rec, "base").run()
        qs = rec.chain_queries
        assert qs, "engine never queried the chain"
        assert qs == sorted(qs), "chain queries went backwards in time"
        assert all(q.date() <= date(2023, 3, 31) for q in qs), "queried beyond backtest end"


class TestExecutionScenariosOrdering:
    def test_conservative_never_beats_optimistic(self, market):
        from xsp_research.config import EXECUTION_SCENARIOS

        results = {}
        for name in ("optimistic", "conservative"):
            cfg = make_config(execution=EXECUTION_SCENARIOS[name])
            m = SyntheticMarket(SyntheticConfig(start=START, end=END, strike_pct_range=0.12))
            results[name] = BacktestEngine(cfg, m, m, m, name).run()
        opt = results["optimistic"].equity_curve["equity"][-1]
        con = results["conservative"].equity_curve["equity"][-1]
        assert con <= opt + 1e-6
