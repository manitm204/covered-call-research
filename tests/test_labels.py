"""Label construction: hand-checked outcomes on synthetic hold-to-expiry runs."""

from datetime import date

import polars as pl
import pytest

from tests.test_engine import make_config
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.config import BacktestConfig, EntryFrequency
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket
from xsp_research.models.labels import LABEL_COLUMNS, build_labels

START, END = date(2023, 1, 2), date(2023, 12, 29)


@pytest.fixture(scope="module")
def market():
    return SyntheticMarket(SyntheticConfig(start=START, end=END, strike_pct_range=0.12))


@pytest.fixture(scope="module")
def result(market):
    cfg = make_config(
        exits=[],  # hold to expiration: untruncated labels
        backtest=BacktestConfig(
            start=START, end=END, initial_cash=100_000.0, entry_frequency=EntryFrequency.WEEKLY
        ),
    )
    cfg = cfg.model_copy(update={"sizing": cfg.sizing.model_copy(update={"max_open_positions": 6})})
    return BacktestEngine(cfg, market, market, market, "base").run()


@pytest.fixture(scope="module")
def labels(result, market):
    return build_labels(result, market)


class TestLabelConstruction:
    def test_one_row_per_trade(self, result, labels):
        assert labels.height == len(result.trades)
        assert set(LABEL_COLUMNS) <= set(labels.columns)

    def test_expire_itm_matches_settlement_price(self, result, market, labels):
        by_id = {r["trade_id"]: r for r in labels.iter_rows(named=True)}
        for t in result.trades:
            settle = market.close(t.expiration)
            row = by_id[t.trade_id]
            if settle is None:
                assert row["label_expire_itm"] is None
            else:
                assert row["label_expire_itm"] == int(settle > t.short_strike)

    def test_expire_itm_consistent_with_settlement_pnl(self, result, labels):
        """Hold-to-expiry OTM settlements must capture the full credit."""
        by_id = {r["trade_id"]: r for r in labels.iter_rows(named=True)}
        for t in result.trades:
            if t.exit_reason != "expiry_settlement":
                continue
            row = by_id[t.trade_id]
            if row["label_expire_itm"] == 0:
                assert t.exit_price == pytest.approx(0.0, abs=1e-9)
                assert row["label_pct_credit"] == pytest.approx(1.0)

    def test_touch_implies_at_least_no_contradiction(self, result, market, labels):
        """ITM at expiration implies the closes touched the strike (the
        expiration close itself counts)."""
        by_id = {r["trade_id"]: r for r in labels.iter_rows(named=True)}
        for t in result.trades:
            row = by_id[t.trade_id]
            if row["label_expire_itm"] == 1:
                assert row["label_touch"] == 1

    def test_touch_hand_check(self, result, market, labels):
        by_id = {r["trade_id"]: r for r in labels.iter_rows(named=True)}
        t = result.trades[0]
        closes = market.closes(t.entry_ts.date(), t.expiration).filter(
            pl.col("date") > t.entry_ts.date()
        )
        expected = int(bool((closes["close"] > t.short_strike).any()))
        assert by_id[t.trade_id]["label_touch"] == expected

    def test_label_end_is_expiration(self, result, labels):
        by_id = {r["trade_id"]: r for r in labels.iter_rows(named=True)}
        for t in result.trades:
            assert by_id[t.trade_id]["label_end"] == t.expiration

    def test_per_spread_normalization(self, result, labels):
        by_id = {r["trade_id"]: r for r in labels.iter_rows(named=True)}
        for t in result.trades:
            assert by_id[t.trade_id]["label_net_pnl"] == pytest.approx(t.realized_net / t.qty)


def test_empty_result_gives_typed_empty_frame(market):
    from xsp_research.backtest.engine import BacktestResult
    from xsp_research.backtest.ledger import Ledger

    empty = BacktestResult(
        data_source="synthetic",
        config_snapshot={},
        execution_scenario="base",
        trades=[],
        equity_curve=pl.DataFrame(),
        entry_attempts=[],
        ledger=Ledger(),
        positions=[],
    )
    frame = build_labels(empty, market)
    assert frame.is_empty()
    assert "label_expire_itm" in frame.columns
