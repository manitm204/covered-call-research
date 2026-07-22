"""Data-quality validation: every check must fire on seeded defects and stay
quiet on clean data."""

from datetime import UTC, date, datetime, timedelta

import polars as pl

from xsp_research.ingestion.base import CHAIN_SCHEMA
from xsp_research.ingestion.validation import Severity, validate_options_dataset

TS0 = datetime(2023, 7, 3, 19, 30, tzinfo=UTC)
EXP = date(2023, 8, 4)


def make_rows(**overrides) -> dict:
    row = {
        "ts": TS0,
        "root": "XSP",
        "expiration": EXP,
        "strike": 450.0,
        "option_type": "C",
        "bid": 1.00,
        "ask": 1.10,
        "bid_size": 10,
        "ask_size": 10,
        "volume": 5,
        "open_interest": 100,
        "underlying_price": 440.0,
    }
    row.update(overrides)
    return row


def frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=dict(CHAIN_SCHEMA))


def clean_chain() -> list[dict]:
    """A small internally-consistent chain: several strikes of calls and puts."""
    rows = []
    for k in range(430, 455, 5):
        # crude but consistent quotes: calls decrease in strike, puts increase
        c_mid = max(440.0 - k + 5.0, 0.3)
        p_mid = max(k - 440.0 + 5.0, 0.3)
        rows.append(make_rows(strike=float(k), option_type="C", bid=c_mid - 0.05, ask=c_mid + 0.05))
        rows.append(make_rows(strike=float(k), option_type="P", bid=p_mid - 0.05, ask=p_mid + 0.05))
    return rows


def issues_by_check(report):
    return {i.check: i for i in report.issues}


class TestCleanData:
    def test_clean_chain_passes(self):
        report = validate_options_dataset(frame(clean_chain()))
        assert report.passed, report.to_dict()

    def test_empty_dataset_fails(self):
        report = validate_options_dataset(frame([]))
        assert not report.passed


class TestSeededDefects:
    def test_duplicates_detected(self):
        rows = clean_chain() + [make_rows(strike=430.0)]  # 430C exists already
        report = validate_options_dataset(frame(rows))
        found = issues_by_check(report)
        assert "duplicates" in found and found["duplicates"].severity is Severity.ERROR
        assert not report.passed

    def test_crossed_market_detected(self):
        rows = clean_chain() + [make_rows(strike=460.0, bid=1.50, ask=1.00)]
        report = validate_options_dataset(frame(rows))
        assert "crossed_market" in issues_by_check(report)
        assert not report.passed

    def test_locked_market_detected(self):
        rows = clean_chain() + [make_rows(strike=460.0, bid=1.10, ask=1.10)]
        report = validate_options_dataset(frame(rows))
        found = issues_by_check(report)
        assert "locked_market" in found and found["locked_market"].severity is Severity.WARN
        assert report.passed  # warn-only

    def test_zero_bid_reported(self):
        rows = clean_chain() + [make_rows(strike=480.0, bid=0.0, ask=0.05)]
        report = validate_options_dataset(frame(rows))
        assert "zero_bid" in issues_by_check(report)

    def test_below_intrinsic_detected(self):
        # 400C with underlying 440: intrinsic 40, ask 5 is impossible
        rows = clean_chain() + [make_rows(strike=400.0, bid=4.0, ask=5.0)]
        report = validate_options_dataset(frame(rows))
        assert "below_intrinsic" in issues_by_check(report)

    def test_abnormal_spread_detected(self):
        rows = clean_chain() + [make_rows(strike=460.0, bid=0.10, ask=2.00)]
        report = validate_options_dataset(frame(rows))
        assert "abnormal_spread" in issues_by_check(report)

    def test_inconsistent_underlying_detected(self):
        rows = clean_chain() + [make_rows(strike=465.0, underlying_price=444.0)]
        report = validate_options_dataset(frame(rows))
        found = issues_by_check(report)
        assert "inconsistent_underlying" in found
        assert found["inconsistent_underlying"].severity is Severity.ERROR

    def test_stale_quotes_detected(self):
        rows = []
        for i in range(8):  # same bid/ask for 8 consecutive sessions
            rows.append(make_rows(ts=TS0 + timedelta(days=i), bid=1.00, ask=1.10))
        report = validate_options_dataset(frame(rows), stale_sessions_warn=5)
        assert "stale_quotes" in issues_by_check(report)

    def test_strike_gap_detected(self):
        rows = [make_rows(strike=float(k)) for k in range(430, 445)]  # 1-wide grid...
        rows += [make_rows(strike=470.0)]  # ...then a 26-point hole
        report = validate_options_dataset(frame(rows))
        assert "strike_gaps" in issues_by_check(report)

    def test_session_gap_detected(self):
        rows = clean_chain()
        rows += [make_rows(ts=TS0 + timedelta(days=10), strike=430.0)]
        report = validate_options_dataset(frame(rows), max_session_gap_days=4)
        assert "session_gaps" in issues_by_check(report)

    def test_parity_dispersion_detected(self):
        rows = clean_chain()
        # Corrupt one put mid massively: parity forward for that strike shifts.
        rows.append(make_rows(strike=445.0, option_type="P", bid=30.0, ask=30.2))
        rows.append(make_rows(strike=445.0, option_type="C", bid=0.55, ask=0.65))
        report = validate_options_dataset(frame(rows), parity_dispersion_warn=0.005)
        assert "parity_dispersion" in issues_by_check(report)


def test_report_serializes():
    report = validate_options_dataset(frame(clean_chain()))
    d = report.to_dict()
    assert d["passed"] is True
    assert isinstance(d["issues"], list)
