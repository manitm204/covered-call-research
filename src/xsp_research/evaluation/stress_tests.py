"""Stress-period reporting over named historical windows.

Windows outside the data's coverage are reported as ``not_covered`` — never
silently skipped — so a report always states which stress regimes the data
could and could not test. Rapid rebounds get first-class treatment: they are
the scenario most likely to hurt a bear call spread (capped benefit on the way
down, open-ended pain as the market rips back through the short strike).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl

MULT_DD_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class StressWindow:
    key: str
    label: str
    start: date
    end: date
    kind: str  # "selloff" | "rebound" | "bull" | "bear"


STRESS_WINDOWS: tuple[StressWindow, ...] = (
    StressWindow("q4_2018", "Q4 2018 selloff", date(2018, 10, 1), date(2018, 12, 24), "selloff"),
    StressWindow("covid_crash", "COVID crash", date(2020, 2, 19), date(2020, 3, 23), "selloff"),
    StressWindow("rebound_2020", "2020 rebound", date(2020, 3, 24), date(2020, 8, 31), "rebound"),
    StressWindow("bull_2021", "2021 low-vol bull", date(2021, 1, 4), date(2021, 12, 31), "bull"),
    StressWindow("bear_2022", "2022 bear market", date(2022, 1, 3), date(2022, 10, 12), "bear"),
    StressWindow(
        "rebound_2023", "2023 recovery rally", date(2023, 1, 3), date(2023, 7, 31), "rebound"
    ),
)


def _window_slice(frame: pl.DataFrame, w: StressWindow) -> pl.DataFrame:
    return frame.filter((pl.col("date") >= w.start) & (pl.col("date") <= w.end))


def _period_return(equity: pl.Series) -> float:
    return float(equity[-1] / equity[0] - 1.0)


def _max_drawdown(equity: pl.Series) -> float:
    eq = equity.to_numpy()
    peak = eq[0]
    worst = 0.0
    for v in eq:
        peak = max(peak, v)
        worst = min(worst, v / max(peak, MULT_DD_FLOOR) - 1.0)
    return worst


def stress_report(
    overlay_frame: pl.DataFrame,
    trades: pl.DataFrame,
    windows: tuple[StressWindow, ...] = STRESS_WINDOWS,
) -> dict[str, Any]:
    """Per-window behavior of stock, overlay, and combined portfolios.

    ``overlay_frame`` is the output of build_overlay_frame; ``trades`` the
    trades frame (may be empty).
    """
    out: dict[str, Any] = {}
    for w in windows:
        sl = _window_slice(overlay_frame, w)
        # Require meaningful coverage: at least 60% of the window's calendar span.
        if sl.height < 2 or (sl["date"][-1] - sl["date"][0]).days < 0.6 * (w.end - w.start).days:
            out[w.key] = {"label": w.label, "kind": w.kind, "status": "not_covered"}
            continue
        row: dict[str, Any] = {
            "label": w.label,
            "kind": w.kind,
            "status": "covered",
            "sessions": sl.height,
            "stock_return": _period_return(sl["stock_equity"]),
            "combined_return": _period_return(sl["combined_equity"]),
            "overlay_pnl": float(sl["overlay_pnl"].sum()),
            "stock_max_drawdown": _max_drawdown(sl["stock_equity"]),
            "combined_max_drawdown": _max_drawdown(sl["combined_equity"]),
        }
        row["overlay_helped"] = row["combined_return"] > row["stock_return"]
        if not trades.is_empty():
            wt = trades.filter(
                (pl.col("exit_ts").dt.date() >= w.start) & (pl.col("exit_ts").dt.date() <= w.end)
            )
            row["trades_closed_in_window"] = wt.height
            if not wt.is_empty():
                row["worst_trade_net"] = float(wt["realized_net"].min())
                row["window_trades_net"] = float(wt["realized_net"].sum())
        out[w.key] = row

    rebounds = [
        v for v in out.values() if v.get("kind") == "rebound" and v.get("status") == "covered"
    ]
    out["_summary"] = {
        "windows_covered": sum(1 for v in out.values() if v.get("status") == "covered"),
        "windows_not_covered": sum(1 for v in out.values() if v.get("status") == "not_covered"),
        "rebound_windows_where_overlay_hurt": sum(
            1 for v in rebounds if not v.get("overlay_helped", True)
        ),
        "note": (
            "Rebound windows are the expected worst case for a bear call overlay; "
            "losses there are structural, not incidental."
        ),
    }
    return out
