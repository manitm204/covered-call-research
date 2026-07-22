"""Contract selection on a synthetic chain: determinism, targets, audits."""

from datetime import date

import pytest

from xsp_research.config import SelectionConfig, WidthMethod
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket
from xsp_research.strategy.candidate_selection import (
    select_bear_call_spread,
    select_expiration,
)

R, Q = 0.045, 0.015


@pytest.fixture(scope="module")
def market():
    return SyntheticMarket(SyntheticConfig(start=date(2023, 2, 1), end=date(2023, 4, 30)))


@pytest.fixture(scope="module")
def snapshot(market):
    session = date(2023, 2, 6)
    ts = market._snapshot_ts(session)
    chain = market.chain(ts)
    spot = float(chain["underlying_price"][0])
    return ts, chain, spot


class TestExpirationSelection:
    def test_within_window_closest_to_target(self):
        cfg = SelectionConfig()
        exps = [date(2023, 3, 3), date(2023, 3, 10), date(2023, 3, 17), date(2023, 4, 21)]
        chosen, audits = select_expiration(exps, date(2023, 2, 6), cfg)
        # DTEs: 25, 32, 39, 74 -> eligible {25, 32}; |32-30| < |25-30|
        assert chosen == date(2023, 3, 10)
        rejected = {a.candidate for a in audits if not a.accepted}
        assert str(date(2023, 4, 21)) in rejected

    def test_none_when_no_expiration_in_window(self):
        cfg = SelectionConfig()
        chosen, audits = select_expiration([date(2023, 2, 10)], date(2023, 2, 6), cfg)
        assert chosen is None
        assert all(not a.accepted for a in audits)


class TestSpreadSelection:
    def test_baseline_selection(self, snapshot):
        ts, chain, spot = snapshot
        cfg = SelectionConfig()  # 0.15 delta, $5 fixed width
        res = select_bear_call_spread(chain, ts, spot, R, Q, cfg)
        assert res.selected
        assert 25 <= res.dte <= 35
        assert res.short_delta == pytest.approx(0.15, abs=0.05)  # $1 strike grid granularity
        sq = res.spread_quote
        assert sq.spread.width == pytest.approx(5.0)
        assert sq.spread.short_leg.strike > spot  # OTM
        assert any(a.stage == "spread" and a.accepted for a in res.audits)

    def test_deterministic(self, snapshot):
        ts, chain, spot = snapshot
        cfg = SelectionConfig()
        a = select_bear_call_spread(chain, ts, spot, R, Q, cfg)
        b = select_bear_call_spread(chain, ts, spot, R, Q, cfg)
        assert a.spread_quote.spread == b.spread_quote.spread

    def test_impossible_liquidity_filter_rejects_with_audit(self, snapshot):
        ts, chain, spot = snapshot
        cfg = SelectionConfig(min_short_bid=999.0)
        res = select_bear_call_spread(chain, ts, spot, R, Q, cfg)
        assert not res.selected
        assert res.reject_summary()  # human-readable reasons available

    def test_long_delta_method(self, snapshot):
        ts, chain, spot = snapshot
        cfg = SelectionConfig(width_method=WidthMethod.LONG_DELTA, long_delta_target=0.05)
        res = select_bear_call_spread(chain, ts, spot, R, Q, cfg)
        assert res.selected
        assert res.long_delta < res.short_delta
        assert res.long_delta == pytest.approx(0.05, abs=0.04)

    def test_pct_underlying_method(self, snapshot):
        ts, chain, spot = snapshot
        cfg = SelectionConfig(width_method=WidthMethod.PCT_UNDERLYING, pct_underlying=0.02)
        res = select_bear_call_spread(chain, ts, spot, R, Q, cfg)
        assert res.selected
        assert res.spread_quote.spread.width == pytest.approx(0.02 * spot, abs=1.0)

    def test_max_loss_budget_method(self, snapshot):
        ts, chain, spot = snapshot
        budget = 300.0
        cfg = SelectionConfig(width_method=WidthMethod.MAX_LOSS_BUDGET, max_loss_budget=budget)
        res = select_bear_call_spread(chain, ts, spot, R, Q, cfg)
        assert res.selected
        sq = res.spread_quote
        est_max_loss = (sq.spread.width - sq.credit_mid) * 100
        assert est_max_loss <= budget + 1e-6

    def test_expected_move_method(self, snapshot):
        ts, chain, spot = snapshot
        cfg = SelectionConfig(
            width_method=WidthMethod.EXPECTED_MOVE_FRACTION, expected_move_fraction=0.5
        )
        res = select_bear_call_spread(chain, ts, spot, R, Q, cfg)
        assert res.selected
        assert res.spread_quote.spread.width >= 1.0

    def test_missing_param_raises_at_config_time(self):
        with pytest.raises(ValueError):
            SelectionConfig(width_method=WidthMethod.LONG_DELTA)
