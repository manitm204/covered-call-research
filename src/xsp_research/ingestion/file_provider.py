"""User-supplied Parquet/CSV adapters (provider-agnostic ingestion target).

Expected layout (configured in configs/data.yaml):
  options:    one Parquet dataset in the canonical chain schema (any partitioning
              readable by polars.scan_parquet, e.g. hive `date=YYYY-MM-DD/`).
  underlying: Parquet/CSV with columns (date, close) per symbol file.
  rates:      Parquet/CSV with columns (date, rate) — annualized decimals.

These adapters only *select* data as of a timestamp; they never fabricate it.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date, datetime
from pathlib import Path

import polars as pl

from xsp_research.ingestion.base import CHAIN_SCHEMA, validate_chain_frame


class ParquetOptionsProvider:
    """Canonical-schema Parquet chain data. `chain(as_of)` returns, per contract,
    the latest quote row on as_of's date with ts <= as_of."""

    def __init__(self, dataset_path: str | Path, root: str) -> None:
        self._lf = pl.scan_parquet(str(dataset_path))
        self._root = root
        self._path = str(dataset_path)

    @property
    def data_source(self) -> str:
        return f"user_parquet:{self._root}:{self._path}"

    def expirations(self, as_of: date) -> list[date]:
        rows = (
            self._lf.filter(
                (pl.col("root") == self._root)
                & (pl.col("ts").dt.date() == as_of)
                & (pl.col("expiration") >= as_of)
            )
            .select(pl.col("expiration").unique().sort())
            .collect()
        )
        return rows["expiration"].to_list()

    def chain(self, as_of: datetime) -> pl.DataFrame:
        df = (
            self._lf.filter(
                (pl.col("root") == self._root)
                & (pl.col("ts").dt.date() == as_of.date())
                & (pl.col("ts") <= as_of)
            )
            .sort("ts")
            .group_by(["root", "expiration", "strike", "option_type"], maintain_order=True)
            .last()
            .select(list(CHAIN_SCHEMA))
            .collect()
        )
        validate_chain_frame(df)
        return df


class ParquetUnderlyingProvider:
    def __init__(self, path: str | Path, symbol: str) -> None:
        p = Path(path)
        df = pl.read_csv(p) if p.suffix == ".csv" else pl.read_parquet(p)
        self._df = (
            df.select(pl.col("date").cast(pl.Date), pl.col("close").cast(pl.Float64))
            .unique(subset="date")
            .sort("date")
        )
        self._symbol = symbol
        self._closes = dict(
            zip(self._df["date"].to_list(), self._df["close"].to_list(), strict=False)
        )

    @property
    def data_source(self) -> str:
        return f"user_file:{self._symbol}"

    def trading_dates(self, start: date, end: date) -> list[date]:
        return [d for d in self._df["date"].to_list() if start <= d <= end]

    def close(self, as_of: date) -> float | None:
        return self._closes.get(as_of)

    def closes(self, start: date, end: date) -> pl.DataFrame:
        return self._df.filter((pl.col("date") >= start) & (pl.col("date") <= end))


class SeriesRatesProvider:
    """Rates from a (date, rate) file; returns the last observation <= as_of
    (no forward-filling from the future)."""

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        df = pl.read_csv(p) if p.suffix == ".csv" else pl.read_parquet(p)
        df = (
            df.select(pl.col("date").cast(pl.Date), pl.col("rate").cast(pl.Float64))
            .unique(subset="date")
            .sort("date")
        )
        self._dates = df["date"].to_list()
        self._rates = df["rate"].to_list()
        self._path = str(p)

    @property
    def data_source(self) -> str:
        return f"user_file:rates:{self._path}"

    def rate(self, as_of: date) -> float:
        i = bisect_right(self._dates, as_of)
        if i == 0:
            raise ValueError(f"no rate observation at or before {as_of}")
        return self._rates[i - 1]


class FixedRatesProvider:
    def __init__(self, rate: float) -> None:
        self._rate = rate

    @property
    def data_source(self) -> str:
        return f"fixed_rate:{self._rate}"

    def rate(self, as_of: date) -> float:
        return self._rate
