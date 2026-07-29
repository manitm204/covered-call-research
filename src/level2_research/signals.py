"""Point-in-time signal frames. Every column is shifted so the value dated D is
computable from closes up to D-1 — safe to read on session D at 15:30."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .market import PRICES_LONG


def build_signals(symbol: str) -> pd.DataFrame:
    px = pd.read_parquet(PRICES_LONG / f"{symbol.upper()}.parquet")
    px["date"] = pd.to_datetime(px["date"])
    df = px.set_index("date").sort_index()
    a = df["adjClose"]
    out = pd.DataFrame(index=df.index)
    out["ret1"] = a.pct_change()
    out["ret5"] = a.pct_change(5)
    out["ma20"] = a.rolling(20).mean()
    out["ma200"] = a.rolling(200).mean()
    out["px"] = a
    out["above_ma200"] = a > out["ma200"]
    out["above_ma200_up"] = a > out["ma200"] * 1.01  # entry band
    out["above_ma200_dn"] = a > out["ma200"] * 0.99  # exit band (False -> exit)
    out["rv20"] = out["ret1"].rolling(20).std() * np.sqrt(252)
    roll_max = a.rolling(252, min_periods=60).max()
    out["dd"] = a / roll_max - 1.0
    out["deep_dd_60"] = out["dd"].rolling(60).min()  # most negative dd in last 60 sess
    out = out.shift(1)  # <- the lag: everything above is knowable at D-1 close
    return out


def asof(sig: pd.DataFrame, session) -> pd.Series | None:
    ts = pd.Timestamp(session)
    if ts in sig.index:
        return sig.loc[ts]
    prior = sig.loc[:ts]
    return prior.iloc[-1] if len(prior) else None
