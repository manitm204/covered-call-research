"""Spread payoff, settlement, and max gain/loss invariants (hand-computed)."""

import pytest

from tests.conftest import make_call
from xsp_research.domain import BearCallSpread


class TestSettlementValue:
    def test_below_short_strike_worthless(self, spread_400_405):
        assert spread_400_405.settlement_value(390.0) == 0.0
        assert spread_400_405.settlement_value(400.0) == 0.0

    def test_between_strikes_intrinsic(self, spread_400_405):
        assert spread_400_405.settlement_value(402.0) == pytest.approx(2.0)

    def test_above_long_strike_capped_at_width(self, spread_400_405):
        assert spread_400_405.settlement_value(405.0) == pytest.approx(5.0)
        assert spread_400_405.settlement_value(450.0) == pytest.approx(5.0)

    def test_always_within_zero_and_width(self, spread_400_405):
        for s in [0.0, 300.0, 399.99, 400.01, 404.99, 405.01, 10_000.0]:
            v = spread_400_405.settlement_value(s)
            assert 0.0 <= v <= spread_400_405.width


class TestMaxGainLoss:
    def test_hand_computed(self, spread_400_405):
        # credit 1.00, width 5, qty 2:
        assert spread_400_405.max_profit_dollars(1.0, 2) == pytest.approx(200.0)
        assert spread_400_405.max_loss_dollars(1.0, 2) == pytest.approx(800.0)

    def test_identity_profit_plus_loss_equals_width(self, spread_400_405):
        credit, qty = 1.37, 3
        total = spread_400_405.max_profit_dollars(credit, qty) + spread_400_405.max_loss_dollars(
            credit, qty
        )
        assert total == pytest.approx(spread_400_405.width * 100 * qty)


class TestSpreadValidation:
    def test_rejects_inverted_strikes(self):
        with pytest.raises(ValueError):
            BearCallSpread(short_leg=make_call(405.0), long_leg=make_call(400.0))

    def test_rejects_mismatched_expiration(self):
        from datetime import date

        from xsp_research.domain import OptionContract, OptionType

        other = OptionContract(
            root="XSP", expiration=date(2023, 4, 28), strike=405.0, option_type=OptionType.CALL
        )
        with pytest.raises(ValueError):
            BearCallSpread(short_leg=make_call(400.0), long_leg=other)
