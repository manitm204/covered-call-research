"""Mandated evaluation metrics from a daily equity curve and trade ledger."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _drawdown(eq: pd.Series) -> tuple[pd.Series, float, int]:
    peak = eq.cummax()
    dd = eq / peak - 1.0
    trough = dd.idxmin() if len(dd) else None
    longest = 0
    run = 0
    for v in (eq >= peak).values:
        run = 0 if v else run + 1
        longest = max(longest, run)
    return dd, float(dd.min()) if len(dd) else 0.0, longest


def equity_metrics(equity: pd.DataFrame, rf: pd.Series | None = None) -> dict:
    eq = equity.set_index("session")["equity"].astype(float)
    rets = eq.pct_change().dropna()
    n = len(eq)
    years = max((pd.Timestamp(eq.index[-1]) - pd.Timestamp(eq.index[0])).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    vol = rets.std() * np.sqrt(252)
    rf_daily = 0.0
    if rf is not None:
        rf_daily = float(rf.reindex(pd.to_datetime(rets.index), method="ffill").mean()) / 252
    ex = rets - rf_daily
    sharpe = ex.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    downside = rets[rets < 0].std()
    sortino = ex.mean() / downside * np.sqrt(252) if downside and downside > 0 else np.nan
    dd, maxdd, longest = _drawdown(eq)
    weekly = eq.resample("W", on=None).last().pct_change().dropna() if isinstance(eq.index, pd.DatetimeIndex) else pd.Series(dtype=float)
    eqd = eq.copy(); eqd.index = pd.to_datetime(eqd.index)
    weekly = eqd.resample("W").last().pct_change().dropna()
    monthly = eqd.resample("ME").last().pct_change().dropna()
    return dict(
        start=str(eq.index[0]), end=str(eq.index[-1]), sessions=n,
        final_equity=round(float(eq.iloc[-1]), 2),
        total_return=round(float(eq.iloc[-1] / eq.iloc[0] - 1), 4),
        cagr=round(float(cagr), 4), ann_vol=round(float(vol), 4),
        sharpe=round(float(sharpe), 2), sortino=round(float(sortino), 2) if sortino == sortino else None,
        max_drawdown=round(maxdd, 4), calmar=round(float(cagr / abs(maxdd)), 2) if maxdd < 0 else None,
        longest_dd_sessions=longest,
        worst_day=round(float(rets.min()), 4) if len(rets) else 0,
        worst_week=round(float(weekly.min()), 4) if len(weekly) else 0,
        worst_month=round(float(monthly.min()), 4) if len(monthly) else 0,
    )


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or not len(trades):
        return dict(n_trades=0)
    t = trades[trades.pnl.notna()].copy()
    wins, losses = t[t.pnl > 0], t[t.pnl <= 0]
    hold = None
    if "open_session" in t and "close_session" in t:
        ho = pd.to_datetime(t.close_session) - pd.to_datetime(t.open_session)
        hold = float(ho.dt.days.median())
    pf = float(wins.pnl.sum() / abs(losses.pnl.sum())) if len(losses) and losses.pnl.sum() != 0 else np.inf
    return dict(
        n_trades=int(len(t)), win_rate=round(float((t.pnl > 0).mean()), 3),
        avg_win=round(float(wins.pnl.mean()), 2) if len(wins) else 0,
        avg_loss=round(float(losses.pnl.mean()), 2) if len(losses) else 0,
        profit_factor=round(pf, 2), ev_per_trade=round(float(t.pnl.mean()), 2),
        total_pnl=round(float(t.pnl.sum()), 2),
        worst_trade=round(float(t.pnl.min()), 2), best_trade=round(float(t.pnl.max()), 2),
        median_holding_days=hold,
    )


def yearly_returns(equity: pd.DataFrame) -> pd.Series:
    eq = equity.set_index("session")["equity"].astype(float)
    eq.index = pd.to_datetime(eq.index)
    yearly = eq.resample("YE").last()
    first = eq.iloc[0]
    out = yearly.pct_change()
    out.iloc[0] = yearly.iloc[0] / first - 1
    out.index = out.index.year
    return out.round(4)


def block_bootstrap_ci(x: np.ndarray, stat=np.mean, block: int = 10, n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 5:
        return (float("nan"), float("nan"))
    k = max(1, int(np.ceil(n / block)))
    starts = rng.integers(0, max(1, n - block + 1), size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    vals = np.apply_along_axis(stat, 1, x[idx])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def capital_utilization(equity: pd.DataFrame) -> float:
    """Mean fraction of equity not sitting as free cash (reserved + position value)."""
    e = equity.copy()
    used = (e["equity"] - (e["cash"] - e["reserved"])) / e["equity"]
    return round(float(used.mean()), 3)
