"""Metrics and summary reporting, including the synthetic-data warning path."""

import json
from datetime import date

import pytest

from tests.test_engine import make_config
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.evaluation.metrics import equity_metrics, summarize, trade_metrics
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket


@pytest.fixture(scope="module")
def result():
    start, end = date(2023, 1, 2), date(2023, 6, 30)
    cfg = make_config()
    cfg = cfg.model_copy(
        update={"backtest": cfg.backtest.model_copy(update={"start": start, "end": end})}
    )
    m = SyntheticMarket(SyntheticConfig(start=start, end=end, strike_pct_range=0.12))
    return BacktestEngine(cfg, m, m, m, "base").run()


def test_summarize_is_json_serializable(result):
    summary = summarize(result)
    json.dumps(summary, default=str)  # must not raise


def test_synthetic_warning_present(result):
    summary = summarize(result)
    assert "SYNTHETIC" in summary.get("WARNING", "")


def test_trade_metrics_populated(result):
    tm = trade_metrics(result.trades_frame())
    assert tm["n_trades"] == len(result.trades)
    assert 0.0 <= tm["win_rate"] <= 1.0
    assert tm["total_net_pnl"] == pytest.approx(sum(t.realized_net for t in result.trades))


def test_equity_metrics_structure(result):
    em = equity_metrics(result.equity_curve)
    assert em["final_equity"] == pytest.approx(result.equity_curve["equity"][-1])
    assert em["max_drawdown"] <= 0.0


def test_interest_separated_from_trading_pnl(result):
    summary = summarize(result)
    assert summary["interest_income_total"] > 0
    assert summary["realized_trading_pnl_gross"] == pytest.approx(
        sum(t.realized_gross for t in result.trades)
    )


def test_empty_trades_frame_handled():
    import polars as pl

    assert trade_metrics(pl.DataFrame()) == {"n_trades": 0}
