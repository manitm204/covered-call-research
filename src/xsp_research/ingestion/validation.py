"""Data-quality validation for canonical options datasets.

Produces a structured ValidationReport listing every detected issue with a
severity, count, and examples. ERROR-level issues indicate data that should not
be backtested without remediation; WARN indicates known market-microstructure
artifacts to keep in mind (and which the execution model must handle); INFO is
descriptive.

Checks implemented (research brief section 5):
  duplicates, crossed markets, locked markets, zero bids, one-sided quotes,
  prices below intrinsic, abnormal relative spreads, stale quotes (unchanged
  across many sessions), inconsistent underlying price within a snapshot,
  strike-grid gaps, session-coverage gaps, put-call parity dispersion.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import polars as pl


class Severity(enum.StrEnum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    check: str
    severity: Severity
    count: int
    description: str
    examples: tuple[str, ...] = ()


@dataclass(slots=True)
class ValidationReport:
    dataset: str
    rows: int
    ts_min: str
    ts_max: str
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.severity is Severity.ERROR for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "ts_range": [self.ts_min, self.ts_max],
            "passed": self.passed,
            "issues": [
                {
                    "check": i.check,
                    "severity": i.severity.value,
                    "count": i.count,
                    "description": i.description,
                    "examples": list(i.examples),
                }
                for i in sorted(self.issues, key=lambda i: (i.severity.value, -i.count))
            ],
        }


def _examples(df: pl.DataFrame, cols: list[str], n: int = 5) -> tuple[str, ...]:
    rows = df.select([c for c in cols if c in df.columns]).head(n).iter_rows(named=True)
    return tuple(", ".join(f"{k}={v}" for k, v in r.items()) for r in rows)


_KEY = ["ts", "root", "expiration", "strike", "option_type"]


def validate_options_dataset(
    df: pl.DataFrame,
    dataset_name: str = "options",
    *,
    rel_spread_warn: float = 1.0,
    stale_sessions_warn: int = 5,
    parity_dispersion_warn: float = 0.005,
    max_session_gap_days: int = 4,
) -> ValidationReport:
    report = ValidationReport(
        dataset=dataset_name,
        rows=df.height,
        ts_min=str(df["ts"].min()),
        ts_max=str(df["ts"].max()),
    )
    if df.is_empty():
        report.issues.append(
            ValidationIssue("empty", Severity.ERROR, 0, "dataset contains no rows")
        )
        return report
    add = report.issues.append

    # --- duplicates (same contract observed twice at one timestamp)
    dupes = df.group_by(_KEY).len().filter(pl.col("len") > 1)
    if not dupes.is_empty():
        add(
            ValidationIssue(
                "duplicates",
                Severity.ERROR,
                dupes.height,
                "duplicate contract observations at the same timestamp",
                _examples(dupes, _KEY),
            )
        )

    # --- crossed / locked markets
    crossed = df.filter(pl.col("bid") > pl.col("ask"))
    if not crossed.is_empty():
        add(
            ValidationIssue(
                "crossed_market",
                Severity.ERROR,
                crossed.height,
                "bid above ask (unusable quote)",
                _examples(crossed, _KEY + ["bid", "ask"]),
            )
        )
    locked = df.filter((pl.col("bid") == pl.col("ask")) & (pl.col("ask") > 0))
    if not locked.is_empty():
        add(
            ValidationIssue(
                "locked_market",
                Severity.WARN,
                locked.height,
                "bid equals ask; execution model treats these as invalid",
                _examples(locked, _KEY + ["bid", "ask"]),
            )
        )

    # --- zero bids / one-sided quotes (normal for deep OTM, but must be known)
    zero_bid = df.filter((pl.col("bid") == 0) & (pl.col("ask") > 0))
    if not zero_bid.is_empty():
        add(
            ValidationIssue(
                "zero_bid",
                Severity.INFO,
                zero_bid.height,
                "no-bid quotes (typically deep OTM); short legs here are untradable",
            )
        )
    no_ask = df.filter(pl.col("ask") == 0)
    if not no_ask.is_empty():
        add(
            ValidationIssue(
                "no_ask",
                Severity.WARN,
                no_ask.height,
                "quotes with no offer; treated as invalid markets",
                _examples(no_ask, _KEY),
            )
        )

    # --- prices below intrinsic (tolerance for European discounting effects)
    intrinsic_call = (pl.col("underlying_price") - pl.col("strike")).clip(lower_bound=0.0)
    intrinsic_put = (pl.col("strike") - pl.col("underlying_price")).clip(lower_bound=0.0)
    intrinsic = pl.when(pl.col("option_type") == "C").then(intrinsic_call).otherwise(intrinsic_put)
    below = df.filter((pl.col("ask") > 0) & (pl.col("ask") < intrinsic * 0.985 - 0.05))
    if not below.is_empty():
        add(
            ValidationIssue(
                "below_intrinsic",
                Severity.WARN,
                below.height,
                "ask below spot intrinsic beyond discounting tolerance (stale or bad quote)",
                _examples(below, _KEY + ["ask", "underlying_price"]),
            )
        )

    # --- abnormal relative spreads
    mid = (pl.col("bid") + pl.col("ask")) / 2
    wide = df.filter((mid > 0.10) & ((pl.col("ask") - pl.col("bid")) / mid > rel_spread_warn))
    if not wide.is_empty():
        add(
            ValidationIssue(
                "abnormal_spread",
                Severity.WARN,
                wide.height,
                f"relative bid-ask spread above {rel_spread_warn:.0%} with mid > $0.10",
                _examples(wide, _KEY + ["bid", "ask"]),
            )
        )

    # --- inconsistent underlying price within one snapshot timestamp
    incons = (
        df.group_by("ts")
        .agg((pl.col("underlying_price").max() - pl.col("underlying_price").min()).alias("rng"))
        .filter(pl.col("rng") > 1e-6)
    )
    if not incons.is_empty():
        add(
            ValidationIssue(
                "inconsistent_underlying",
                Severity.ERROR,
                incons.height,
                "underlying_price differs across rows sharing a timestamp "
                "(misaligned underlying join)",
                _examples(incons, ["ts", "rng"]),
            )
        )

    # --- stale quotes: identical bid/ask across many consecutive sessions
    per_session = (
        df.with_columns(pl.col("ts").dt.date().alias("session"))
        .group_by(["root", "expiration", "strike", "option_type", "session"])
        .agg(pl.col("bid").last(), pl.col("ask").last())
        .sort("session")
    )
    contract_key = ["root", "expiration", "strike", "option_type"]
    changed = per_session.with_columns(
        (
            (pl.col("bid") != pl.col("bid").shift(1).over(contract_key))
            | (pl.col("ask") != pl.col("ask").shift(1).over(contract_key))
        )
        .fill_null(True)
        .alias("changed")
    )
    runs = changed.with_columns(pl.col("changed").cum_sum().over(contract_key).alias("run_id"))
    stale = (
        runs.group_by(contract_key + ["run_id"])
        .agg(pl.len().alias("run_len"), pl.col("bid").first(), pl.col("ask").first())
        .filter((pl.col("run_len") >= stale_sessions_warn) & (pl.col("ask") > 0.10))
    )
    if not stale.is_empty():
        add(
            ValidationIssue(
                "stale_quotes",
                Severity.WARN,
                stale.height,
                f"non-trivial quotes unchanged for >= {stale_sessions_warn} consecutive "
                "sessions (possible stale feed)",
                _examples(stale, contract_key + ["run_len", "bid", "ask"]),
            )
        )

    # --- strike-grid gaps per (session, expiration)
    grid = (
        df.with_columns(pl.col("ts").dt.date().alias("session"))
        .select(["session", "root", "expiration", "strike"])
        .unique()
        .sort("strike")
        .group_by(["session", "root", "expiration"])
        .agg(
            pl.col("strike").diff().drop_nulls().median().alias("median_step"),
            pl.col("strike").diff().drop_nulls().max().alias("max_step"),
            pl.len().alias("n_strikes"),
        )
        .filter((pl.col("n_strikes") >= 10) & (pl.col("max_step") > 3 * pl.col("median_step")))
    )
    if not grid.is_empty():
        add(
            ValidationIssue(
                "strike_gaps",
                Severity.WARN,
                grid.height,
                "strike grid has gaps > 3x the median step (missing strikes)",
                _examples(grid, ["session", "expiration", "median_step", "max_step"]),
            )
        )

    # --- session coverage gaps
    sessions = df.select(pl.col("ts").dt.date().unique().sort().alias("d"))["d"].to_list()
    gaps = [
        (a, b, (b - a).days)
        for a, b in zip(sessions, sessions[1:], strict=False)
        if (b - a).days > max_session_gap_days
    ]
    if gaps:
        add(
            ValidationIssue(
                "session_gaps",
                Severity.WARN,
                len(gaps),
                f"gaps > {max_session_gap_days} calendar days between quote sessions "
                "(missing data or extended closures)",
                tuple(f"{a} -> {b} ({d}d)" for a, b, d in gaps[:5]),
            )
        )

    # --- put-call parity: implied-forward dispersion near ATM per (ts, expiration)
    calls = df.filter((pl.col("option_type") == "C") & (pl.col("bid") > 0))
    puts = df.filter((pl.col("option_type") == "P") & (pl.col("bid") > 0))
    joined = calls.join(
        puts, on=["ts", "root", "expiration", "strike"], how="inner", suffix="_p"
    ).with_columns(
        (
            pl.col("strike")
            + (pl.col("bid") + pl.col("ask")) / 2
            - (pl.col("bid_p") + pl.col("ask_p")) / 2
        ).alias("implied_forward")
    )
    near_atm = joined.filter((pl.col("strike") / pl.col("underlying_price") - 1.0).abs() < 0.05)
    if not near_atm.is_empty():
        parity = (
            near_atm.group_by(["ts", "expiration"])
            .agg(
                pl.col("implied_forward").median().alias("fwd_med"),
                (
                    (pl.col("implied_forward").max() - pl.col("implied_forward").min())
                    / pl.col("implied_forward").median()
                ).alias("dispersion"),
                pl.len().alias("n"),
            )
            .filter((pl.col("n") >= 5) & (pl.col("dispersion") > parity_dispersion_warn))
        )
        if not parity.is_empty():
            add(
                ValidationIssue(
                    "parity_dispersion",
                    Severity.WARN,
                    parity.height,
                    f"put-call-parity implied forwards disperse > "
                    f"{parity_dispersion_warn:.2%} across near-ATM strikes "
                    "(stale/inconsistent quotes within snapshot)",
                    _examples(parity, ["ts", "expiration", "fwd_med", "dispersion"]),
                )
            )
    else:
        add(
            ValidationIssue(
                "parity_unavailable",
                Severity.INFO,
                0,
                "no overlapping call/put quotes near ATM; parity diagnostics skipped",
            )
        )

    return report
