"""Execution model: hand-computed fills, costs, and rejections."""

import pytest

from tests.conftest import make_quote
from xsp_research.backtest.execution import close_spread, open_spread
from xsp_research.config import CostConfig, ExecutionConfig, FillModel
from xsp_research.domain import SpreadQuote

COSTS = CostConfig(commission_per_contract=0.65, fees_per_contract=0.60)


class TestOpenFills:
    def test_natural(self, spread_quote_standard):
        cfg = ExecutionConfig(fill_model=FillModel.NATURAL)
        rep = open_spread(spread_quote_standard, 1, cfg, COSTS)
        assert rep.filled and rep.price == pytest.approx(0.50)

    def test_midpoint(self, spread_quote_standard):
        cfg = ExecutionConfig(fill_model=FillModel.MIDPOINT)
        rep = open_spread(spread_quote_standard, 1, cfg, COSTS)
        assert rep.filled and rep.price == pytest.approx(0.60)

    def test_pct_between(self, spread_quote_standard):
        cfg = ExecutionConfig(fill_model=FillModel.PCT_BETWEEN, fill_pct=0.5)
        rep = open_spread(spread_quote_standard, 1, cfg, COSTS)
        assert rep.filled and rep.price == pytest.approx(0.55)

    def test_slippage_reduces_credit(self, spread_quote_standard):
        cfg = ExecutionConfig(fill_model=FillModel.NATURAL, slippage_per_spread=0.02)
        rep = open_spread(spread_quote_standard, 1, cfg, COSTS)
        assert rep.price == pytest.approx(0.48)

    def test_fees_hand_computed(self, spread_quote_standard):
        # 3 spreads x 2 legs x ($0.65 + $0.60) = $7.50
        cfg = ExecutionConfig(fill_model=FillModel.NATURAL)
        rep = open_spread(spread_quote_standard, 3, cfg, COSTS)
        assert rep.fees_dollars == pytest.approx(7.50)


class TestCloseFills:
    def test_natural_close(self, spread_quote_standard):
        cfg = ExecutionConfig(fill_model=FillModel.NATURAL)
        rep = close_spread(spread_quote_standard, 1, cfg, COSTS)
        assert rep.filled and rep.price == pytest.approx(0.70)

    def test_pct_between_close(self, spread_quote_standard):
        cfg = ExecutionConfig(fill_model=FillModel.PCT_BETWEEN, fill_pct=0.5)
        rep = close_spread(spread_quote_standard, 1, cfg, COSTS)
        assert rep.filled and rep.price == pytest.approx(0.65)

    def test_close_slippage_increases_debit(self, spread_quote_standard):
        cfg = ExecutionConfig(fill_model=FillModel.NATURAL, slippage_per_spread=0.02)
        rep = close_spread(spread_quote_standard, 1, cfg, COSTS)
        assert rep.price == pytest.approx(0.72)


class TestRejections:
    def test_rejects_wide_leg(self):
        wide = SpreadQuote(
            short_quote=make_quote(400.0, 1.00, 1.10),
            long_quote=make_quote(405.0, 0.05, 0.50),  # rel spread 1.64
        )
        rep = open_spread(wide, 1, ExecutionConfig(max_leg_rel_spread=0.5), COSTS)
        assert not rep.filled and "rel spread" in rep.reason

    def test_rejects_below_min_credit(self):
        thin = SpreadQuote(
            short_quote=make_quote(400.0, 0.45, 0.50),
            long_quote=make_quote(405.0, 0.40, 0.44),
        )
        rep = open_spread(
            thin, 1, ExecutionConfig(fill_model=FillModel.NATURAL, min_credit=0.05), COSTS
        )
        assert not rep.filled and "below minimum" in rep.reason

    def test_rejects_one_sided_market(self):
        dead = SpreadQuote(
            short_quote=make_quote(400.0, 0.0, 0.0),
            long_quote=make_quote(405.0, 0.40, 0.50),
        )
        # bid==ask==0 -> invalid market
        rep = open_spread(dead, 1, ExecutionConfig(), COSTS)
        assert not rep.filled
