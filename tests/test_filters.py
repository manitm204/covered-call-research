"""Entry filters: clause evaluation, missing-value policy, engine integration."""

from datetime import date

import polars as pl
import pytest

from tests.test_engine import END, START, make_config
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.experiments.filters import EntryFilterConfig, FilterClause, make_entry_gate
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket

FEATURES = pl.DataFrame(
    {
        "date": [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
        "vix_level": [18.0, 25.0, None],
    }
)


def gate_for(clauses, allow_on_missing=False):
    return make_entry_gate(
        FEATURES, EntryFilterConfig(clauses=clauses, allow_on_missing=allow_on_missing)
    )


class TestGateLogic:
    def test_pass_and_fail(self):
        gate = gate_for([FilterClause(feature="vix_level", op="<", value=20.0)])
        allowed, why = gate(date(2023, 1, 2))
        assert allowed and "passed" in why
        blocked, why = gate(date(2023, 1, 3))
        assert not blocked and "fails" in why

    def test_missing_value_blocks_by_default(self):
        gate = gate_for([FilterClause(feature="vix_level", op="<", value=20.0)])
        allowed, why = gate(date(2023, 1, 4))
        assert not allowed and "unavailable" in why

    def test_missing_value_allowed_when_configured(self):
        gate = gate_for(
            [FilterClause(feature="vix_level", op="<", value=20.0)], allow_on_missing=True
        )
        allowed, _ = gate(date(2023, 1, 4))
        assert allowed

    def test_missing_session_row(self):
        gate = gate_for([FilterClause(feature="vix_level", op="<", value=20.0)])
        allowed, why = gate(date(2023, 6, 1))
        assert not allowed and "no feature row" in why

    def test_unknown_feature_raises_at_build_time(self):
        with pytest.raises(ValueError, match="unknown features"):
            gate_for([FilterClause(feature="nonexistent", op="<", value=1.0)])

    def test_clauses_are_anded(self):
        gate = gate_for(
            [
                FilterClause(feature="vix_level", op=">", value=10.0),
                FilterClause(feature="vix_level", op="<", value=20.0),
            ]
        )
        assert gate(date(2023, 1, 2))[0]
        assert not gate(date(2023, 1, 3))[0]  # 25 fails the second clause


class TestEngineIntegration:
    def test_blocking_gate_prevents_all_trades(self):
        market = SyntheticMarket(
            SyntheticConfig(start=START, end=date(2023, 4, 28), strike_pct_range=0.12)
        )

        def deny_all(session):
            return False, "test gate"

        engine = BacktestEngine(make_config(), market, market, market, "base", entry_gate=deny_all)
        result = engine.run()
        assert result.trades == []
        filtered = [a for a in result.entry_attempts if a.reason.startswith("filtered:")]
        assert filtered and all(not a.filled for a in filtered)

    def test_open_gate_matches_no_gate(self):
        cfg = make_config()
        m1 = SyntheticMarket(SyntheticConfig(start=START, end=END, strike_pct_range=0.12))
        m2 = SyntheticMarket(SyntheticConfig(start=START, end=END, strike_pct_range=0.12))
        base = BacktestEngine(cfg, m1, m1, m1, "base").run()
        gated = BacktestEngine(cfg, m2, m2, m2, "base", entry_gate=lambda s: (True, "open")).run()
        assert [t.trade_id for t in base.trades] == [t.trade_id for t in gated.trades]
        assert base.equity_curve["equity"].to_list() == gated.equity_curve["equity"].to_list()
