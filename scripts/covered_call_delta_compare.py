"""Build delta-sensitivity comparison charts + data: 0.25Δ vs 0.15Δ, for both
naive and gated, flat single-lot and reinvesting variants. Writes
results/covered_call_delta_compare.json for the report to consume.
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

FLAT = dict(
    root25=Path("results/covered_call"), root15=Path("results/covered_call_d15"),
    buyhold="buyhold", naive="naive_cc", gated="gated_cc", fwdret="fwdret_cc",
)
REINVEST = dict(
    root25=Path("results/covered_call_reinvest"), root15=Path("results/covered_call_reinvest_d15"),
    buyhold="buyhold_drip", naive="naive_cc_reinvest", gated="gated_cc_reinvest",
)

FLAT_SERIES = [  # (root_key, strat_key, label, color, linestyle)
    ("root25", "buyhold", "Buy & hold", "#8a8f98", "-"),
    ("root25", "naive", "Naive, 0.25Δ", "#e0806a", "-"),
    ("root15", "naive", "Naive, 0.15Δ", "#b3492f", "--"),
    ("root25", "gated", "Gated (breach), 0.25Δ", "#4fb3a0", "-"),
    ("root15", "gated", "Gated (breach), 0.15Δ", "#1f7a6c", "--"),
    ("root25", "fwdret", "Gated (fwd-ret), 0.25Δ", "#7c9fd9", "-"),
    ("root15", "fwdret", "Gated (fwd-ret), 0.15Δ", "#4a6fb3", "--"),
]
REINVEST_SERIES = [  # (root_key, strat_key, label, color, linestyle)
    ("root25", "buyhold", "Buy & hold", "#8a8f98", "-"),
    ("root25", "naive", "Naive, 0.25Δ", "#e0806a", "-"),
    ("root15", "naive", "Naive, 0.15Δ", "#b3492f", "--"),
    ("root25", "gated", "Gated, 0.25Δ", "#4fb3a0", "-"),
    ("root15", "gated", "Gated, 0.15Δ", "#1f7a6c", "--"),
]


def load_eq(cfg, root_key, strat_key, sym):
    root = cfg[root_key]
    strat = cfg[strat_key]
    eq = pd.read_parquet(root / sym / strat / "equity.parquet")
    eq["session"] = pd.to_datetime(eq["session"])
    return eq


def load_summary(cfg, root_key, strat_key, sym):
    root = cfg[root_key]
    strat = cfg[strat_key]
    return json.loads((root / sym / strat / "summary.json").read_text())


def equity_chart(cfg, sym, series) -> str:
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=200)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for root_key, strat_key, label, color, ls in series:
        eq = load_eq(cfg, root_key, strat_key, sym)
        norm = eq["equity"] / eq["equity"].iloc[0] * 100
        ax.plot(eq["session"], norm, label=label, color=color, linewidth=1.7,
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
    ax.legend(frameon=False, fontsize=8.5, labelcolor="#8a8f98", loc="upper left", ncol=1)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build():
    out = {"flat": {}, "reinvest": {}}
    for variant, cfg, series in [("flat", FLAT, FLAT_SERIES), ("reinvest", REINVEST, REINVEST_SERIES)]:
        for sym in SYMBOLS:
            rows = {}
            for root_key, strat_key, label, *_ in series:
                s = load_summary(cfg, root_key, strat_key, sym)
                rows[label] = dict(cagr=s["metrics"]["cagr"], sharpe=s["metrics"]["sharpe"],
                                   max_drawdown=s["metrics"]["max_drawdown"],
                                   n_calls=s["n_calls_written"], n_called_away=s["n_called_away"])
            out[variant][sym] = dict(chart=equity_chart(cfg, sym, series), rows=rows)
    Path("results/covered_call_delta_compare.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote results/covered_call_delta_compare.json")


if __name__ == "__main__":
    build()
