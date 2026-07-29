"""Market data access for the Level-2 engine.

ChainStore lazily loads the monthly 15:30 ET NBBO snapshot parquets produced by the
existing ThetaData pipeline and serves per-session chains with derived fields. All
trade pricing uses the unadjusted snapshot level; signals use adjusted closes from
`data/normalized/prices_long` (lagged by the caller).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

OPTIONS_ROOT = Path("data/normalized/options")
PRICES_LONG = Path("data/normalized/prices_long")
RATES = Path("data/normalized/rates/tbill_4w.parquet")
AUX = Path("data/normalized/aux")


class ChainStore:
    def __init__(self, symbol: str, options_root: Path = OPTIONS_ROOT):
        self.symbol = symbol.upper()
        self.dir = options_root / symbol.lower()
        files = sorted(self.dir.glob("chain_*.parquet"))
        if not files:
            raise FileNotFoundError(f"no chain files under {self.dir}")
        self._files = {f.stem[-7:]: f for f in files}  # "YYYY-MM" -> path
        self._sessions: list[date] | None = None

    @lru_cache(maxsize=8)
    def _month(self, ym: str) -> pd.DataFrame:
        df = pd.read_parquet(self._files[ym])
        df["session"] = df["ts"].dt.tz_convert("US/Eastern").dt.date
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
        df["dte"] = (
            pd.to_datetime(df["expiration"]) - pd.to_datetime(df["session"])
        ).dt.days
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
        df["rel_spread"] = np.where(df["mid"] > 0, (df["ask"] - df["bid"]) / df["mid"], np.inf)
        return df

    def sessions(self) -> list[date]:
        if self._sessions is None:
            out: list[date] = []
            for ym in sorted(self._files):
                out.extend(sorted(self._month(ym)["session"].unique()))
            self._sessions = out
        return self._sessions

    def chain(self, session: date) -> pd.DataFrame:
        ym = f"{session.year:04d}-{session.month:02d}"
        if ym not in self._files:
            return pd.DataFrame()
        m = self._month(ym)
        return m[m["session"] == session]

    def spot(self, session: date) -> float | None:
        c = self.chain(session)
        return float(c["underlying_price"].iloc[0]) if len(c) else None


@dataclass(frozen=True)
class DailyData:
    """Point-in-time daily series for signals and settlement."""

    prices: pd.DataFrame  # date-indexed: open/high/low/close/adjClose
    dividends: pd.DataFrame  # ex_date, amount
    tbill: pd.Series  # date-indexed annualized decimal rate

    @classmethod
    def load(cls, symbol: str) -> "DailyData":
        px = pd.read_parquet(PRICES_LONG / f"{symbol.upper()}.parquet")
        px["date"] = pd.to_datetime(px["date"])
        px = px.set_index("date").sort_index()
        divp = PRICES_LONG / f"{symbol.upper()}_dividends.parquet"
        div = pd.read_parquet(divp) if divp.exists() else pd.DataFrame(columns=["ex_date", "amount"])
        if len(div):
            div["ex_date"] = pd.to_datetime(div["ex_date"])
        r = pd.read_parquet(RATES)
        r["date"] = pd.to_datetime(r["date"])
        rate = r.set_index("date")["rate"].sort_index()  # already decimal (0.036 = 3.6%)
        if rate.max() > 0.30:
            raise ValueError("tbill rate series looks like percent, expected decimal")
        return cls(prices=px, dividends=div, tbill=rate)

    def close_on(self, d: date) -> float | None:
        """Unadjusted close on d, or the most recent prior session's close."""
        ts = pd.Timestamp(d)
        s = self.prices["close"]
        s = s.loc[:ts]
        return float(s.iloc[-1]) if len(s) else None

    def rate_on(self, d: date) -> float:
        s = self.tbill.loc[: pd.Timestamp(d)]
        return float(s.iloc[-1]) if len(s) else 0.0

    def dividend_on(self, d: date) -> float:
        if not len(self.dividends):
            return 0.0
        m = self.dividends[self.dividends["ex_date"] == pd.Timestamp(d)]
        return float(m["amount"].sum()) if len(m) else 0.0
