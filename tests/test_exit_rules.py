"""Exit-rule evaluation: triggers, priority order, and non-triggers."""

from datetime import UTC, datetime

from tests.conftest import make_call
from xsp_research.config import ExitRuleConfig
from xsp_research.domain import BearCallSpread, ExitReason, Fill, SpreadPosition
from xsp_research.strategy.exit_rules import ExitContext, evaluate_exit

TS = datetime(2023, 3, 10, 20, 30, tzinfo=UTC)


def make_position(credit: float = 1.0) -> SpreadPosition:
    spread = BearCallSpread(short_leg=make_call(400.0), long_leg=make_call(405.0))
    return SpreadPosition(
        position_id="T0001",
        spread=spread,
        qty=1,
        entry_fill=Fill(ts=TS, price=credit, qty=1, fees_dollars=2.5),
    )


def ctx(cost_to_close: float, short_delta: float | None = 0.15, dte: int = 20) -> ExitContext:
    return ExitContext(
        position=make_position(),
        ts=TS,
        cost_to_close_mid=cost_to_close,
        short_delta=short_delta,
        dte=dte,
    )


PROFIT_50 = ExitRuleConfig(kind="profit_target", profit_target_pct=0.50)
STOP_2X = ExitRuleConfig(kind="stop_loss", stop_loss_multiple=2.0)
DTE_7 = ExitRuleConfig(kind="dte", dte_threshold=7)
DELTA_35 = ExitRuleConfig(kind="delta_stop", delta_threshold=0.35)
RULES = [PROFIT_50, STOP_2X, DTE_7, DELTA_35]


def test_profit_target_triggers_at_half_credit():
    assert evaluate_exit(RULES, ctx(cost_to_close=0.50)) is ExitReason.PROFIT_TARGET
    assert evaluate_exit(RULES, ctx(cost_to_close=0.51)) is None


def test_stop_loss_triggers_at_2x_credit():
    assert evaluate_exit(RULES, ctx(cost_to_close=2.0)) is ExitReason.STOP_LOSS
    assert evaluate_exit(RULES, ctx(cost_to_close=1.99)) is None


def test_dte_exit():
    assert evaluate_exit(RULES, ctx(cost_to_close=1.0, dte=7)) is ExitReason.DTE
    assert evaluate_exit(RULES, ctx(cost_to_close=1.0, dte=8)) is None


def test_delta_stop():
    assert evaluate_exit(RULES, ctx(cost_to_close=1.0, short_delta=0.40)) is ExitReason.DELTA_STOP


def test_delta_stop_skipped_when_delta_unavailable():
    assert evaluate_exit([DELTA_35], ctx(cost_to_close=1.0, short_delta=None)) is None


def test_priority_order_first_rule_wins():
    # Both stop (ctc >= 2x) and dte would trigger; list order decides.
    assert evaluate_exit([STOP_2X, DTE_7], ctx(cost_to_close=3.0, dte=5)) is ExitReason.STOP_LOSS
    assert evaluate_exit([DTE_7, STOP_2X], ctx(cost_to_close=3.0, dte=5)) is ExitReason.DTE


def test_no_rules_never_exits():
    assert evaluate_exit([], ctx(cost_to_close=0.0)) is None
