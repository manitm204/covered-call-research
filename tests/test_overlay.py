"""Portfolio overlay: hand-computed combination math and metric panels."""

from datetime import date

import polars as pl
import pytest

from tests.test_engine import END, START, make_config
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.backtest.portfolio_overlay import (
    OverlayConfig,
    build_overlay_frame,
    compare_portfolios,
    overlay_trading_pnl,
    stock_returns_from_closes,
)
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket


@pytest.fixture(scope="module")
def market():
    return SyntheticMarket(SyntheticConfig(start=START, end=END, strike_pct_range=0.12))


@pytest.fixture(scope="module")
def result(market):
    return BacktestEngine(make_config(), market, market, market, "base").run()


@pytest.fixture(scope="module")
def stock_closes(market):
    return market.closes(START, END)


class TestStockReturns:
    def test_hand_computed(self):
        closes = pl.DataFrame(
            {
                "date": [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
                "close": [100.0, 101.0, 99.99],
            }
        )
        rets = stock_returns_from_closes(closes)["stock_ret"].to_list()
        assert rets[0] == 0.0
        assert rets[1] == pytest.approx(0.01)
        assert rets[2] == pytest.approx(99.99 / 101.0 - 1.0)


class TestTradingPnl:
    def test_excludes_interest(self, result):
        pnl = overlay_trading_pnl(result)
        led = result.ledger
        # Sum of daily trading P&L must equal realized gross - fees (all closed).
        assert pnl["overlay_pnl"].sum() == pytest.approx(
            led.realized_pnl_gross() - led.total_fees(), abs=1e-4
        )


class TestMarginOverlayCombination:
    def test_combined_equals_stock_plus_cum_pnl(self, result, stock_closes):
        frame = build_overlay_frame(stock_closes, result, OverlayConfig(mode="margin_overlay"))
        stock = frame["stock_equity"].to_numpy()
        cum_pnl = frame["overlay_pnl"].to_numpy().cumsum()
        combined = frame["combined_equity"].to_numpy()
        assert combined == pytest.approx(stock + cum_pnl)

    def test_initial_capital_preserved(self, result, stock_closes):
        frame = build_overlay_frame(stock_closes, result, OverlayConfig())
        # Day 1: stock at initial (0% first ret) + day-1 trading P&L.
        assert frame["stock_equity"][0] == pytest.approx(100_000.0)

    def test_calendar_misalignment_raises(self, result):
        bad = pl.DataFrame({"date": [date(2020, 1, 2), date(2020, 1, 3)], "close": [100.0, 101.0]})
        with pytest.raises(ValueError, match="misaligned"):
            build_overlay_frame(bad, result, OverlayConfig())


class TestCarveOut:
    def test_sleeve_scaling(self, result, stock_closes):
        frame = build_overlay_frame(
            stock_closes, result, OverlayConfig(mode="carve_out", carve_out_pct=0.10)
        )
        # First session: 90% stock + 10% sleeve scaled overlay equity.
        expected = 90_000.0 + 0.10 * result.equity_curve["equity"][0]
        assert frame["combined_equity"][0] == pytest.approx(expected, rel=1e-9)


class TestComparison:
    def test_panel_structure(self, result, stock_closes):
        frame = build_overlay_frame(stock_closes, result, OverlayConfig())
        panel = compare_portfolios(frame, result)
        for key in ("stock_only", "overlay_standalone", "combined", "combined_minus_stock"):
            assert key in panel
        assert panel["data_source"] == "synthetic"
        assert "cagr" in panel["combined_minus_stock"]
        assert "downside_beta" in panel["combined_vs_stock_benchmark"]

    def test_rally_drag_is_reported(self, result, stock_closes):
        frame = build_overlay_frame(stock_closes, result, OverlayConfig())
        panel = compare_portfolios(frame, result)
        assert "overlay_pnl_on_strong_up_days_total" in panel
