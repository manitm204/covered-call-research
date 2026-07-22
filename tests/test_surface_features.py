"""Surface features from a synthetic chain snapshot."""

from datetime import date

import pytest

from xsp_research.config import SelectionConfig
from xsp_research.features.surface import SURFACE_FEATURE_NAMES, compute_surface_features
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket
from xsp_research.strategy.candidate_selection import SelectionResult, select_bear_call_spread

R, Q = 0.045, 0.015


@pytest.fixture(scope="module")
def snapshot():
    market = SyntheticMarket(SyntheticConfig(start=date(2023, 2, 1), end=date(2023, 4, 30)))
    session = date(2023, 2, 6)
    ts = market._snapshot_ts(session)
    chain = market.chain(ts)
    spot = float(chain["underlying_price"][0])
    sel = select_bear_call_spread(chain, ts, spot, R, Q, SelectionConfig())
    assert sel.selected
    return chain, sel, spot, session


class TestSurfaceFeatures:
    def test_all_names_present(self, snapshot):
        chain, sel, spot, session = snapshot
        feats = compute_surface_features(chain, sel, spot, R, Q, session)
        assert set(feats) == set(SURFACE_FEATURE_NAMES)

    def test_atm_iv_matches_synthetic_base(self, snapshot):
        chain, sel, spot, session = snapshot
        feats = compute_surface_features(chain, sel, spot, R, Q, session)
        # Synthetic smile: iv = base + skew*ln(K/S); at ATM ln(K/S) ~ 0.
        assert feats["surf_atm_iv"] == pytest.approx(0.17, abs=0.02)

    def test_wing_slope_negative_under_negative_skew(self, snapshot):
        chain, sel, spot, session = snapshot
        feats = compute_surface_features(chain, sel, spot, R, Q, session)
        assert feats["surf_wing_slope"] is not None and feats["surf_wing_slope"] < 0

    def test_credit_ratios_consistent(self, snapshot):
        chain, sel, spot, session = snapshot
        feats = compute_surface_features(chain, sel, spot, R, Q, session)
        sq = sel.spread_quote
        credit, width = sq.credit_mid, sq.spread.width
        assert feats["surf_credit_over_width"] == pytest.approx(credit / width)
        assert feats["surf_credit_over_max_loss"] == pytest.approx(credit / (width - credit))
        assert 0 < feats["surf_credit_over_width"] < 1

    def test_net_delta_negative_for_bear_call(self, snapshot):
        chain, sel, spot, session = snapshot
        feats = compute_surface_features(chain, sel, spot, R, Q, session)
        assert feats["surf_net_delta"] < 0

    def test_short_delta_and_dte_passthrough(self, snapshot):
        chain, sel, spot, session = snapshot
        feats = compute_surface_features(chain, sel, spot, R, Q, session)
        assert feats["surf_short_delta"] == pytest.approx(sel.short_delta)
        assert feats["surf_dte"] == float(sel.dte)

    def test_unselected_returns_all_none(self, snapshot):
        chain, _, spot, session = snapshot
        empty = SelectionResult(spread_quote=None)
        feats = compute_surface_features(chain, empty, spot, R, Q, session)
        assert all(v is None for v in feats.values())
