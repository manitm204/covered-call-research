"""Stress-window slicing and report generation."""

from datetime import date

import pytest

from tests.test_engine import END, START, make_config
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.backtest.portfolio_overlay import (
    OverlayConfig,
    build_overlay_frame,
    compare_portfolios,
)
from xsp_research.evaluation.reporting import generate_report
from xsp_research.evaluation.stress_tests import STRESS_WINDOWS, StressWindow, stress_report
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket


@pytest.fixture(scope="module")
def setup():
    market = SyntheticMarket(SyntheticConfig(start=START, end=END, strike_pct_range=0.12))
    result = BacktestEngine(make_config(), market, market, market, "base").run()
    frame = build_overlay_frame(market.closes(START, END), result, OverlayConfig())
    return market, result, frame


class TestStressWindows:
    def test_uncovered_windows_reported_not_skipped(self, setup):
        _, result, frame = setup
        report = stress_report(frame, result.trades_frame())
        # Synthetic data covers 2023 only: 2018/2020/2021/2022 windows must be
        # explicitly labeled not_covered.
        assert report["covid_crash"]["status"] == "not_covered"
        assert report["bear_2022"]["status"] == "not_covered"

    def test_covered_window_has_full_row(self, setup):
        _, result, frame = setup
        report = stress_report(frame, result.trades_frame())
        row = report["rebound_2023"]
        assert row["status"] == "covered"
        for key in ("stock_return", "combined_return", "overlay_pnl", "overlay_helped"):
            assert key in row

    def test_summary_counts(self, setup):
        _, result, frame = setup
        report = stress_report(frame, result.trades_frame())
        s = report["_summary"]
        assert s["windows_covered"] + s["windows_not_covered"] == len(STRESS_WINDOWS)

    def test_custom_window(self, setup):
        _, result, frame = setup
        w = StressWindow("mid_2023", "custom", date(2023, 3, 1), date(2023, 6, 30), "bull")
        report = stress_report(frame, result.trades_frame(), windows=(w,))
        assert report["mid_2023"]["status"] == "covered"


class TestReportGeneration:
    def test_full_report_written(self, setup, tmp_path):
        _, result, frame = setup
        panel = compare_portfolios(frame, result)
        stress = stress_report(frame, result.trades_frame())
        path = generate_report(
            result, tmp_path, overlay_frame=frame, overlay_comparison=panel, stress=stress
        )
        text = path.read_text()
        assert "SYNTHETIC DATA" in text  # warning block present
        assert "## Portfolio overlay" in text
        assert "## Stress windows" in text
        assert "not covered by data" in text
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "config_snapshot.json").exists()
        assert (tmp_path / "stress.json").exists()

    def test_report_without_overlay(self, setup, tmp_path):
        _, result, _ = setup
        path = generate_report(result, tmp_path / "plain")
        text = path.read_text()
        assert "## Trade metrics" in text
        assert "## Portfolio overlay" not in text

    def test_chart_written_when_matplotlib_present(self, setup, tmp_path):
        pytest.importorskip("matplotlib")
        _, result, frame = setup
        generate_report(result, tmp_path / "charts", overlay_frame=frame)
        assert (tmp_path / "charts" / "equity.png").exists()
