"""Point-in-time signal frames. Every column is shifted so the value dated D is
computable from closes up to D-1 — safe to read on session D at 15:30."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .market import PRICES_LONG

_SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
_VOL_SYM = {"SPY": "VIX", "QQQ": "VXN", "IWM": "RVX"}


def _own_vol(symbol: str) -> pd.Series:
    vsym = _VOL_SYM.get(symbol.upper())
    if vsym is None:
        return pd.Series(dtype=float)
    d = pd.read_parquet(PRICES_LONG / f"{vsym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    col = "vix" if "vix" in d.columns else "close"
    return d.set_index("date")[col].sort_index()


def _absorption_shift() -> pd.Series:
    eigen = pd.read_parquet(PRICES_LONG / "eigen_features_long.parquet")
    return eigen["ar1"].diff(20)


def _rsi14(a: pd.Series) -> pd.Series:
    delta = a.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _sector_returns() -> pd.DataFrame:
    rets = {}
    for s in _SECTORS:
        p = PRICES_LONG / "sectors" / f"{s}.parquet"
        if not p.exists():
            p = PRICES_LONG / f"{s}.parquet"
        d = pd.read_parquet(p)
        d["date"] = pd.to_datetime(d["date"])
        rets[s] = d.set_index("date")["adjClose"].sort_index().pct_change()
    return pd.concat(rets, axis=1)


def _market_sector_corr_60() -> pd.Series:
    """Market-wide regime signal: mean pairwise 60-day correlation among the 9
    sector SPDRs (is the market trading as one tide, or fragmented?) — shared
    across every fund, not fund-specific. A fund's own correlation *to* the
    sector average is nearly tautological for a broad index like SPY (it's
    approximately a cap-weighted sum of those same sectors), so it carries
    almost no information; the pairwise version is what the regime-signal
    research notes actually used."""
    rets = _sector_returns()
    pairs = [(rets[a], rets[b]) for i, a in enumerate(_SECTORS) for b in _SECTORS[i + 1:]]
    corrs = [x.rolling(60).corr(y) for x, y in pairs]
    return pd.concat(corrs, axis=1).mean(axis=1)


def build_signals(symbol: str) -> pd.DataFrame:
    px = pd.read_parquet(PRICES_LONG / f"{symbol.upper()}.parquet")
    px["date"] = pd.to_datetime(px["date"])
    df = px.set_index("date").sort_index()
    a = df["adjClose"]
    out = pd.DataFrame(index=df.index)
    out["ret1"] = a.pct_change()
    out["ret5"] = a.pct_change(5)
    out["ret21"] = a.pct_change(21)
    out["ret21_pctile"] = out["ret21"].rolling(756, min_periods=252).rank(pct=True)
    out["ret84"] = a.pct_change(84)  # "4-month trend" in the regime-signal research notes
    out["rsi14"] = _rsi14(a)
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
    out["sector_corr_60"] = _market_sector_corr_60().reindex(a.index).ffill()
    out["vol_own"] = _own_vol(symbol).reindex(a.index).ffill()
    out["absorption_shift"] = _absorption_shift().reindex(a.index).ffill()
    out = out.shift(1)  # <- the lag: everything above is knowable at D-1 close
    return out


def asof(sig: pd.DataFrame, session) -> pd.Series | None:
    ts = pd.Timestamp(session)
    if ts in sig.index:
        return sig.loc[ts]
    prior = sig.loc[:ts]
    return prior.iloc[-1] if len(prior) else None
