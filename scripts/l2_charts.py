"""Chart generation for Level-2 research: equity, drawdown, yearly, regime,
parameter-stability panels. Writes PNGs under reports/level2/."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("reports/level2")
OUT.mkdir(parents=True, exist_ok=True)

C = dict(strategy="#4269d0", bench="#efb118", spy="#3ca951", cash="#9c6b4e",
         dd="#ff725c")


def equity_and_drawdown(curves: dict[str, pd.Series], title: str, fname: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2.2, 1]))
    for i, (name, eq) in enumerate(curves.items()):
        color = list(C.values())[i % len(C)]
        ax1.plot(eq.index, eq.values, label=name, lw=1.4, color=color)
        dd = eq / eq.cummax() - 1
        ax2.plot(dd.index, dd.values * 100, lw=1.1, color=color)
    ax1.set_yscale("log")
    ax1.set_ylabel("equity ($, log)")
    ax1.legend(loc="upper left", frameon=False)
    ax1.set_title(title)
    ax2.set_ylabel("drawdown (%)")
    ax2.axhline(0, color="gray", lw=0.5)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def yearly_bars(yearly: dict[str, pd.Series], title: str, fname: str):
    years = sorted(set().union(*[set(s.index) for s in yearly.values()]))
    n = len(yearly)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (name, s) in enumerate(yearly.items()):
        vals = [s.get(y, np.nan) * 100 for y in years]
        ax.bar(np.arange(len(years)) + i * w, vals, width=w, label=name,
               color=list(C.values())[i % len(C)])
    ax.set_xticks(np.arange(len(years)) + 0.4 - w / 2)
    ax.set_xticklabels(years)
    ax.set_ylabel("return (%)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.legend(frameon=False)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def param_heatmap(df: pd.DataFrame, x: str, y: str, z: str, title: str, fname: str):
    piv = df.pivot_table(index=y, columns=x, values=z)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)
    fig.colorbar(im, ax=ax, label=z)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def load_equity(run_dir: str) -> pd.Series:
    eq = pd.read_parquet(Path(run_dir) / "equity.parquet")
    s = eq.set_index("session")["equity"].astype(float)
    s.index = pd.to_datetime(s.index)
    return s
