"""Robustness checks: bootstrap confidence intervals and parameter grids.

- Trade-expectancy CIs use a moving-block bootstrap over time-ordered trades
  (adjacent 30-day spreads share market conditions; iid resampling would
  understate uncertainty).
- Equity-metric CIs use a moving-block bootstrap over daily returns.
- The short-delta x width grid re-runs the full backtest per combination
  (research brief experiments 18/19) — sensitivity of conclusions to the two
  most important structural parameters.

All bootstraps are seeded and deterministic. CIs are honest about width: with
few trades they will be wide, and that IS the finding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from xsp_research.backtest.engine import BacktestEngine
from xsp_research.config import StrategyConfig
from xsp_research.evaluation.metrics import equity_metrics, trade_metrics


def _block_bootstrap_sample(values: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    if block <= 1:
        return values[rng.integers(0, n, n)]
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), n_blocks)
    out = np.concatenate([values[s : s + block] for s in starts])
    return out[:n]


def bootstrap_mean_ci(
    values: np.ndarray | list[float],
    n_boot: int = 2000,
    block: int = 1,
    ci: float = 0.95,
    seed: int = 7,
) -> dict[str, Any]:
    """Block-bootstrap CI for the mean of a (time-ordered) series."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"error": "no observations"}
    rng = np.random.default_rng(seed)
    means = np.array(
        [float(np.mean(_block_bootstrap_sample(values, block, rng))) for _ in range(n_boot)]
    )
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
        "ci_level": ci,
        "n_obs": int(len(values)),
        "n_boot": n_boot,
        "block": block,
    }


def bootstrap_sharpe_ci(
    daily_returns: np.ndarray | list[float],
    n_boot: int = 2000,
    block: int = 10,
    ci: float = 0.95,
    seed: int = 7,
) -> dict[str, Any]:
    """Block-bootstrap CI for annualized Sharpe of a daily return series."""
    rets = np.asarray(daily_returns, dtype=float)
    if len(rets) < 40:
        return {"error": "too few observations for a Sharpe CI"}
    rng = np.random.default_rng(seed)

    def sharpe(x: np.ndarray) -> float:
        sd = np.std(x, ddof=1)
        return float(np.mean(x) / sd * np.sqrt(252)) if sd > 0 else 0.0

    stats = np.array([sharpe(_block_bootstrap_sample(rets, block, rng)) for _ in range(n_boot)])
    alpha = (1.0 - ci) / 2.0
    return {
        "sharpe": sharpe(rets),
        "ci_low": float(np.quantile(stats, alpha)),
        "ci_high": float(np.quantile(stats, 1.0 - alpha)),
        "ci_level": ci,
        "n_obs": int(len(rets)),
        "block": block,
    }


def result_robustness(result, n_boot: int = 2000, seed: int = 7) -> dict[str, Any]:
    """Bootstrap panel for one BacktestResult: expectancy + Sharpe CIs."""
    trades = result.trades_frame()
    eq = result.equity_curve["equity"].to_numpy()
    rets = np.diff(eq) / eq[:-1]
    out: dict[str, Any] = {"data_source": result.data_source}
    if not trades.is_empty():
        per_spread = (trades["realized_net"] / trades["qty"]).to_numpy()
        out["expectancy_per_spread_ci"] = bootstrap_mean_ci(
            per_spread, n_boot=n_boot, block=3, seed=seed
        )
        out["expectancy_includes_zero"] = (
            out["expectancy_per_spread_ci"]["ci_low"]
            <= 0.0
            <= out["expectancy_per_spread_ci"]["ci_high"]
        )
    out["sharpe_ci"] = bootstrap_sharpe_ci(rets, n_boot=n_boot, seed=seed)
    return out


def delta_width_grid(
    base_cfg: StrategyConfig,
    options,
    underlying,
    rates,
    deltas: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25),
    widths: tuple[float, ...] = (2.0, 5.0, 10.0),
    scenario: str = "grid",
) -> pl.DataFrame:
    """Re-run the backtest per (short delta, width): experiments 18/19."""
    rows: list[dict[str, Any]] = []
    for delta in deltas:
        for width in widths:
            cfg = base_cfg.model_copy(
                update={
                    "selection": base_cfg.selection.model_copy(
                        update={"short_delta_target": delta, "fixed_width": width}
                    )
                }
            )
            result = BacktestEngine(cfg, options, underlying, rates, scenario).run()
            tm = trade_metrics(result.trades_frame())
            em = equity_metrics(result.equity_curve)
            rows.append(
                {
                    "short_delta": delta,
                    "width": width,
                    "n_trades": tm.get("n_trades", 0),
                    "win_rate": tm.get("win_rate"),
                    "total_net_pnl": tm.get("total_net_pnl"),
                    "expectancy_net_per_spread": tm.get("expectancy_net_per_spread"),
                    "avg_return_on_max_risk": tm.get("avg_return_on_max_risk"),
                    "worst_trade_net": tm.get("worst_trade_net"),
                    "total_fees": tm.get("total_fees"),
                    "max_drawdown": em.get("max_drawdown"),
                    "final_equity": em.get("final_equity"),
                    "data_source": result.data_source,
                }
            )
    return pl.DataFrame(rows)
