"""Compare the breach-probability gate vs the plain-forward-return gate:
buy-hold, gated_cc (breach), fwdret_cc (forward-return), flat single-lot,
0.25Δ. Writes results/covered_call_gate_compare.json for the report.
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

ROOT = Path("results/covered_call")
SYMBOLS = ["SPY", "QQQ", "IWM"]
SERIES = [
    ("buyhold", "Buy & hold", "#8a8f98", "-"),
    ("gated_cc", "Gated (breach)", "#1f7a6c", "-"),
    ("fwdret_cc", "Gated (forward return)", "#4a6fb3", "-"),
]


def load_eq(sym, strat):
    eq = pd.read_parquet(ROOT / sym / strat / "equity.parquet")
    eq["session"] = pd.to_datetime(eq["session"])
    return eq


def load_summary(sym, strat):
    return json.loads((ROOT / sym / strat / "summary.json").read_text())


def equity_chart(sym) -> str:
    fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=200)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for strat, label, color, ls in SERIES:
        eq = load_eq(sym, strat)
        norm = eq["equity"] / eq["equity"].iloc[0] * 100
        ax.plot(eq["session"], norm, label=label, color=color, linewidth=1.9,
                linestyle=ls, solid_capstyle="round")
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


def build():
    out = {}
    for sym in SYMBOLS:
        rows = {}
        for strat, label, *_ in SERIES:
            s = load_summary(sym, strat)
            rows[label] = dict(cagr=s["metrics"]["cagr"], sharpe=s["metrics"]["sharpe"],
                               max_drawdown=s["metrics"]["max_drawdown"],
                               n_calls=s["n_calls_written"], n_called_away=s["n_called_away"])
        out[sym] = dict(chart=equity_chart(sym), rows=rows)
    Path("results/covered_call_gate_compare.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote results/covered_call_gate_compare.json")


if __name__ == "__main__":
    build()
