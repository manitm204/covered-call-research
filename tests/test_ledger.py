"""Ledger accounting: hand-computed lifecycle, balance, interest, collateral."""

from datetime import UTC, date, datetime

import pytest

from xsp_research.backtest.ledger import Account, InterestAccruer, Ledger
from xsp_research.config import InterestConfig, InterestMode
from xsp_research.ingestion.file_provider import FixedRatesProvider

TS = datetime(2023, 3, 1, 20, 30, tzinfo=UTC)


class TestSpreadLifecycle:
    """Open credit 1.50 x 2 spreads (=$300), fees $5; mark to 1.20; close at
    0.50 (=$100 debit), fees $5. Realized gross = $200, net = $190."""

    def _run(self) -> Ledger:
        led = Ledger()
        led.deposit_capital(TS, 100_000.0)
        led.open_spread(TS, "T0001", credit_dollars=300.0, fees=5.0)
        led.mark_spread(TS, "T0001", old_mtm=300.0, new_mtm=240.0)  # 1.20 * 100 * 2
        led.close_spread(
            TS,
            "T0001",
            entry_credit_dollars=300.0,
            current_mtm=240.0,
            exit_debit_dollars=100.0,
            fees=5.0,
            memo="close",
        )
        return led

    def test_cash(self):
        led = self._run()
        assert led.cash == pytest.approx(100_000.0 + 300.0 - 5.0 - 100.0 - 5.0)

    def test_realized_and_fees(self):
        led = self._run()
        assert led.realized_pnl_gross() == pytest.approx(200.0)
        assert led.total_fees() == pytest.approx(10.0)

    def test_liability_retired_and_unrealized_flat(self):
        led = self._run()
        assert led.balance(Account.OPTIONS_MTM) == pytest.approx(0.0)
        assert led.balance(Account.UNREALIZED_PNL) == pytest.approx(0.0)

    def test_equity(self):
        led = self._run()
        assert led.equity == pytest.approx(100_000.0 + 190.0)

    def test_trial_balance(self):
        assert self._run().trial_balance_ok()


class TestSettlementLoss:
    def test_full_loss_at_settlement(self):
        """Credit 1.00 x 1 on a 5-wide; settles fully ITM: debit $500, net -$400."""
        led = Ledger()
        led.deposit_capital(TS, 10_000.0)
        led.open_spread(TS, "T0001", credit_dollars=100.0, fees=2.5)
        led.close_spread(
            TS,
            "T0001",
            entry_credit_dollars=100.0,
            current_mtm=100.0,
            exit_debit_dollars=500.0,
            fees=0.0,
            memo="settle",
        )
        assert led.realized_pnl_gross() == pytest.approx(-400.0)
        assert led.equity == pytest.approx(10_000.0 - 400.0 - 2.5)
        assert led.trial_balance_ok()


class TestPostingValidation:
    def test_unbalanced_posting_rejected(self):
        led = Ledger()
        with pytest.raises(ValueError, match="unbalanced"):
            led.post(TS, "bad", [(Account.CASH, 100.0), (Account.CAPITAL, -99.0)])


class TestInterestAccrual:
    def test_act365_hand_computed(self):
        # $10,000 at 3.65% for 10 calendar days = $10,000 * 0.0365 * 10/365 = $10.00
        cfg = InterestConfig(
            mode=InterestMode.FREE_CASH_ONLY, rate_source="fixed", fixed_rate=0.0365
        )
        acc = InterestAccruer(cfg, FixedRatesProvider(0.0))
        assert acc.accrue(date(2023, 3, 1), 10_000.0, 0.0) == 0.0  # first call sets baseline
        got = acc.accrue(date(2023, 3, 11), 10_000.0, 0.0)
        assert got == pytest.approx(10.0)

    def test_free_cash_only_excludes_collateral(self):
        cfg = InterestConfig(mode=InterestMode.FREE_CASH_ONLY, rate_source="fixed", fixed_rate=0.05)
        acc = InterestAccruer(cfg, FixedRatesProvider(0.0))
        assert acc.eligible_balance(cash=10_000.0, restricted_collateral=4_000.0) == 6_000.0

    def test_collateral_mode_includes_it(self):
        cfg = InterestConfig(
            mode=InterestMode.FREE_CASH_AND_COLLATERAL, rate_source="fixed", fixed_rate=0.05
        )
        acc = InterestAccruer(cfg, FixedRatesProvider(0.0))
        assert acc.eligible_balance(cash=10_000.0, restricted_collateral=4_000.0) == 10_000.0

    def test_none_mode(self):
        cfg = InterestConfig(mode=InterestMode.NONE)
        acc = InterestAccruer(cfg, FixedRatesProvider(0.05))
        assert acc.eligible_balance(10_000.0, 0.0) == 0.0

    def test_spread_bps_applied(self):
        cfg = InterestConfig(
            mode=InterestMode.FREE_CASH_ONLY,
            rate_source="fixed",
            fixed_rate=0.05,
            spread_bps=-100.0,  # 5% - 1% = 4%
        )
        acc = InterestAccruer(cfg, FixedRatesProvider(0.0))
        acc.accrue(date(2023, 3, 1), 10_000.0, 0.0)
        got = acc.accrue(date(2024, 3, 1), 10_000.0, 0.0)  # 366 days (2024 leap)
        assert got == pytest.approx(10_000.0 * 0.04 * 366 / 365)
