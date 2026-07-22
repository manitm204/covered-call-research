"""Provider interfaces and the canonical chain schema.

The backtester depends only on these Protocols. Vendor-specific logic lives in
adapters that normalize into the canonical schema. Every query takes an explicit
``as_of`` timestamp and MUST return only data observable at or before it.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, runtime_checkable

import polars as pl

# Canonical options-chain snapshot schema (see docs/PLAN.md data dictionary).
CHAIN_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us", "UTC"),
    "root": pl.Utf8,
    "expiration": pl.Date,
    "strike": pl.Float64,
    "option_type": pl.Utf8,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "bid_size": pl.Int64,
    "ask_size": pl.Int64,
    "volume": pl.Int64,
    "open_interest": pl.Int64,
    "underlying_price": pl.Float64,
}


class ChainSchemaError(ValueError):
    """Raised when a provider emits data violating the canonical schema."""


def validate_chain_frame(df: pl.DataFrame) -> None:
    """Structural validation of a chain snapshot. Raises ChainSchemaError."""
    missing = [c for c in CHAIN_SCHEMA if c not in df.columns]
    if missing:
        raise ChainSchemaError(f"chain frame missing columns: {missing}")
    if df.is_empty():
        return
    bad_type = df.filter(~pl.col("option_type").is_in(["C", "P"]))
    if not bad_type.is_empty():
        raise ChainSchemaError("option_type must be 'C' or 'P'")
    neg = df.filter((pl.col("bid") < 0) | (pl.col("ask") < 0) | (pl.col("strike") <= 0))
    if not neg.is_empty():
        raise ChainSchemaError("negative bid/ask or non-positive strike present")
    dupes = (
        df.group_by(["ts", "root", "expiration", "strike", "option_type"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if not dupes.is_empty():
        raise ChainSchemaError(f"{dupes.height} duplicate contract observations at same ts")


@runtime_checkable
class OptionsProvider(Protocol):
    """Options chain snapshots. `chain` returns the latest snapshot at/before as_of
    on as_of's date, in the canonical schema."""

    @property
    def data_source(self) -> str:  # e.g. "synthetic", "user_parquet:XSP"
        ...

    def expirations(self, as_of: date) -> list[date]:
        """Expirations listed and still alive as of this date."""
        ...

    def chain(self, as_of: datetime) -> pl.DataFrame: ...


@runtime_checkable
class UnderlyingProvider(Protocol):
    @property
    def data_source(self) -> str: ...

    def trading_dates(self, start: date, end: date) -> list[date]: ...

    def close(self, as_of: date) -> float | None:
        """Official close for as_of's session; None if not a session."""
        ...

    def closes(self, start: date, end: date) -> pl.DataFrame:
        """Frame of (date, close) over [start, end]."""
        ...


@runtime_checkable
class RatesProvider(Protocol):
    @property
    def data_source(self) -> str: ...

    def rate(self, as_of: date) -> float:
        """Annualized continuously-compounded risk-free proxy observable at as_of."""
        ...
