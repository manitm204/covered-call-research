"""Vol-regime time-series charts (own vol index vs frozen tercile threshold)
for the rip-risk rule report. Appends 'vol_charts' (base64 PNG per ticker) to
results/covered_call/signal_research.json.
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

PL = Path("data/normalized/prices_long")
FUNDS = {"SPY": "VIX", "QQQ": "VXN", "IWM": "RVX"}
COLOR = "#1f7a6c"

research = json.loads(Path("results/covered_call/signal_research.json").read_text())


def load_vol(sym: str) -> pd.Series:
    d = pd.read_parquet(PL / f"{sym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    col = "vix" if "vix" in d.columns else "close"
    return d.set_index("date")[col].sort_index()


def chart(sym: str, vsym: str) -> str:
    v = load_vol(vsym)
    v = v[v.index >= "2015-01-01"]
    weekly = v.resample("W-FRI").last()
    thresh = research["vol_tercile"][sym]
    fig, ax = plt.subplots(figsize=(4.4, 2.3), dpi=200)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.plot(weekly.index, weekly.values, color=COLOR, linewidth=1.1)
    ax.axhline(thresh, color="#b3492f", linewidth=1.0, linestyle=(0, (4, 2)))
    ymax = float(weekly.values.max()) * 1.05
    ax.axhspan(thresh, ymax, color="#b3492f", alpha=0.06, linewidth=0)
    ax.set_ylim(bottom=0, top=ymax)
    ax.set_title(f"{sym} — {vsym}  (latest {weekly.iloc[-1]:.1f}, tercile {thresh:.1f})",
                fontsize=8.5, color="#8a8f98", loc="left")
    ax.tick_params(colors="#8a8f98", labelsize=7.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#8a8f98")
        ax.spines[spine].set_alpha(0.4)
    ax.grid(axis="y", color="#8a8f98", alpha=0.15, linewidth=0.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


vol_charts = {sym: chart(sym, vsym) for sym, vsym in FUNDS.items()}
research["vol_charts"] = vol_charts
Path("results/covered_call/signal_research.json").write_text(json.dumps(research, indent=1, default=str))
print("added vol_charts")
