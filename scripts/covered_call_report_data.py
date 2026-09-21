"""Build chart images + metrics tables for the covered-call research report.
Reads results/<root>/<SYMBOL>/<strategy>/{equity,trades}.parquet and
summary.json, writes results/<root>/report_data.json (base64 PNG charts +
tables) for the HTML report to consume directly. Run once for the flat
single-lot variant and once for the reinvesting variant (see __main__).
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SYMBOLS = ["SPY", "QQQ", "IWM"]


def load(root: Path, symbol: str, strat: str):
    d = root / symbol / strat
    eq = pd.read_parquet(d / "equity.parquet")
    eq["session"] = pd.to_datetime(eq["session"])
    summ = json.loads((d / "summary.json").read_text())
    trades = None
    if (d / "trades.parquet").exists():
        trades = pd.read_parquet(d / "trades.parquet")
    events = None
    if (d / "events.parquet").exists():
        events = pd.read_parquet(d / "events.parquet")
    return eq, summ, trades, events


def equity_chart_png(root: Path, symbol: str, strats: list[str], colors: dict, labels: dict) -> str:
    fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=200)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for strat in strats:
        eq, summ, _, _ = load(root, symbol, strat)
        norm = eq["equity"] / eq["equity"].iloc[0] * 100
        ax.plot(eq["session"], norm, label=labels[strat], color=colors[strat],
                linewidth=1.9, solid_capstyle="round")
    ax.axhline(100, color="#8a8f98", linewidth=0.7, alpha=0.5, linestyle=(0, (1, 3)))
    ax.set_ylabel("Equity (start = 100)", color="#8a8f98", fontsize=10)
    ax.tick_params(colors="#8a8f98", labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#8a8f98")
        ax.spines[spine].set_alpha(0.4)
    ax.grid(axis="y", color="#8a8f98", alpha=0.15, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9.5, labelcolor="#8a8f98", loc="upper left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def drawdown_chart_png(root: Path, symbol: str, strats: list[str], colors: dict) -> str:
    fig, ax = plt.subplots(figsize=(9.2, 2.6), dpi=200)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for strat in strats:
        eq, summ, _, _ = load(root, symbol, strat)
        s = eq["equity"]
        dd = s / s.cummax() - 1.0
        ax.fill_between(eq["session"], dd * 100, 0, color=colors[strat], alpha=0.18, linewidth=0)
        ax.plot(eq["session"], dd * 100, color=colors[strat], linewidth=1.1)
    ax.set_ylabel("Drawdown %", color="#8a8f98", fontsize=10)
    ax.tick_params(colors="#8a8f98", labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#8a8f98")
        ax.spines[spine].set_alpha(0.4)
    ax.grid(axis="y", color="#8a8f98", alpha=0.15, linewidth=0.6)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build(root: Path, strats: list[str], colors: dict, labels: dict, extra_fields: list[str] = ()):
    out = {}
    for symbol in SYMBOLS:
        sdata = {"equity_chart": equity_chart_png(root, symbol, strats, colors, labels),
                 "dd_chart": drawdown_chart_png(root, symbol, strats, colors),
                 "strategies": {}}
        for strat in strats:
            eq, summ, trades, events = load(root, symbol, strat)
            m = summ["metrics"]
            tm = summ["trade_metrics"]
            yearly = summ["yearly"]
            called_pnl = None
            if trades is not None and len(trades) and "close_how" in trades.columns:
                cc = trades[trades.tag == "cc_write"]
                called_pnl = dict(
                    n=int(len(cc)),
                    win_rate=round(float((cc.pnl > 0).mean()), 3) if len(cc) else None,
                    total_premium_pnl=round(float(cc.pnl.sum()), 2) if len(cc) else None,
                )
            row = dict(
                cagr=m["cagr"], sharpe=m["sharpe"], sortino=m.get("sortino"),
                max_drawdown=m["max_drawdown"], ann_vol=m["ann_vol"],
                final_equity=m["final_equity"], total_return=m["total_return"],
                calmar=m.get("calmar"),
                yearly=yearly,
                n_calls_written=summ["n_calls_written"], n_called_away=summ["n_called_away"],
                call_pnl=called_pnl,
            )
            for f in extra_fields:
                row[f] = summ.get(f)
            sdata["strategies"][strat] = row
        out[symbol] = sdata
    (root / "report_data.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote", root / "report_data.json")


if __name__ == "__main__":
    build(
        Path("results/covered_call"),
        ["buyhold", "naive_cc", "gated_cc", "fwdret_cc"],
        {"buyhold": "#8a8f98", "naive_cc": "#e0806a", "gated_cc": "#1f7a6c", "fwdret_cc": "#4a6fb3"},
        {"buyhold": "Buy & hold", "naive_cc": "Naive monthly covered call",
         "gated_cc": "Breach-gated covered call", "fwdret_cc": "Forward-return-gated covered call"},
    )
    build(
        Path("results/covered_call_reinvest"),
        ["buyhold_drip", "naive_cc_reinvest", "gated_cc_reinvest"],
        {"buyhold_drip": "#8a8f98", "naive_cc_reinvest": "#b3492f", "gated_cc_reinvest": "#1f7a6c"},
        {"buyhold_drip": "Buy & hold (DRIP)", "naive_cc_reinvest": "Naive covered call (reinvest)",
         "gated_cc_reinvest": "Gated covered call (reinvest)"},
        extra_fields=["initial_shares", "final_shares"],
    )
