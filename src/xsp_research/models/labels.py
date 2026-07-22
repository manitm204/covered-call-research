"""Label construction for the trade-level research dataset.

Labels are OUTCOMES (future information) and exist only as model targets —
they must never appear on the feature side. Each label row carries
``label_end`` (the last date the label depends on) so walk-forward splits can
purge overlapping windows.

Economically meaningful targets (research brief section 12):
  label_expire_itm   Model A: 1 if settlement price > short strike. Depends only
                     on the expiration close, so it is defined for every trade
                     whose expiration lies inside the data window — including
                     trades the strategy exited early.
  label_touch        Model B (close-based): 1 if any session close in
                     (entry, expiration] exceeds the short strike. Intraday
                     touches between closes are invisible at daily granularity;
                     this is a documented lower bound on true breach frequency.
  label_net_pnl      Model C: realized net P&L per spread (dollars / qty), as
                     executed (management rules included).
  label_mae          Model D: max adverse excursion per spread (dollars / qty),
                     snapshot-limited (lower bound in magnitude).
  label_pct_credit   fraction of entry credit captured.

For clean Model A/B estimation use a hold-to-expiration label run (exits: [])
so management does not truncate observation windows.
"""

from __future__ import annotations

import polars as pl

from xsp_research.backtest.engine import BacktestResult
from xsp_research.ingestion.base import UnderlyingProvider

LABEL_COLUMNS = (
    "label_expire_itm",
    "label_touch",
    "label_net_pnl",
    "label_mae",
    "label_pct_credit",
    "label_end",
)


def build_labels(result: BacktestResult, underlying: UnderlyingProvider) -> pl.DataFrame:
    """One row per closed trade: (trade_id, label_*, label_end).

    Expiration-dependent labels are null when the expiration lies beyond the
    available underlying data (never guessed).
    """
    rows: list[dict] = []
    positions = {p.position_id: p for p in result.positions}
    for t in result.trades:
        pos = positions[t.trade_id]
        entry_date = t.entry_ts.date()
        expiration = t.expiration

        settle = underlying.close(expiration)
        expire_itm: int | None = None
        touch: int | None = None
        if settle is not None:
            expire_itm = int(settle > t.short_strike)
            closes = underlying.closes(entry_date, expiration)
            window = closes.filter(pl.col("date") > entry_date)
            if not window.is_empty():
                touch = int(bool((window["close"] > t.short_strike).any()))

        rows.append(
            {
                "trade_id": t.trade_id,
                "label_expire_itm": expire_itm,
                "label_touch": touch,
                "label_net_pnl": t.realized_net / t.qty,
                "label_mae": pos.mae_dollars / t.qty,
                "label_pct_credit": t.pct_credit_captured,
                # The label window always extends to expiration: even early-exit
                # trades have expiration-dependent labels (Model A/B).
                "label_end": expiration,
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "trade_id": pl.Utf8,
                "label_expire_itm": pl.Int64,
                "label_touch": pl.Int64,
                "label_net_pnl": pl.Float64,
                "label_mae": pl.Float64,
                "label_pct_credit": pl.Float64,
                "label_end": pl.Date,
            }
        )
    return pl.DataFrame(rows)
