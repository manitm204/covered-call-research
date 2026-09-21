"""SPY buy & hold + Unified Rule v2 overlay on margin, with a $1k cash buffer.

Account mechanics (user-specified, 2026-07-26):
  - $10k starts fully in SPY; dividends reinvest into shares (same as the
    buy & hold baseline, so the gap between lines is purely the overlay).
  - Spread P&L (from reports/portfolio_v2_trades.parquet, the frozen v2 rule
    on SPY+QQQ+IWM) is booked to cash on exit date.
  - Cash above the $1k buffer is immediately swept into SPY at that close.
  - Losses draw cash negative = margin debit, charged T-bill + 1.5% (ACT/365)
    until wins repay it. Positive cash earns the T-bill rate.

Spread collateral is assumed satisfied against the SPY position (margin
overlay, no cash carve-out). Descriptive only — burned data, v2 caveats apply.
Output: reports/portfolio_v2_margin_overlay.png (+ curves parquet).
"""

from __future__ import annotations

import bisect

import numpy as np
import polars as pl

AUX = "data/normalized/aux"
CAPITAL = 10_000.0
BUFFER = 1_000.0
MARGIN_SPREAD = 0.015

spy = pl.read_parquet(f"{AUX}/SPY.parquet").sort("date").filter(
    (pl.col("date") >= pl.date(2018, 8, 1)) & (pl.col("date") <= pl.date(2026, 7, 22)))
dates = spy["date"].to_list()
px = spy["close"].to_numpy()

rt = pl.read_parquet("data/normalized/rates/tbill_4w.parquet").sort("date")
rdates, rvals = rt["date"].to_list(), rt["rate"].to_list()


def tbill(d) -> float:
    j = bisect.bisect_right(rdates, d) - 1
    return rvals[j] if j >= 0 else 0.0


div = pl.read_parquet(f"{AUX}/SPY_DIVIDENDS.parquet")
divmap = dict(zip(div["ex_date"].to_list(), div["amount"].to_list()))

trades = pl.read_parquet("reports/portfolio_v2_trades.parquet")
pnl_by_exit: dict = {}
for r in trades.iter_rows(named=True):
    pnl_by_exit[r["exit_date"]] = pnl_by_exit.get(r["exit_date"], 0.0) + r["realized_net"]

n = len(dates)
base = np.empty(n)        # SPY buy & hold
combo = np.empty(n)       # SPY + overlay on margin
cash_path = np.zeros(n)
sh_base = CAPITAL / px[0]
sh = CAPITAL / px[0]
cash = 0.0
base[0] = combo[0] = CAPITAL
sweeps = 0
for i in range(1, n):
    gap = (dates[i] - dates[i - 1]).days
    r = tbill(dates[i - 1]) + (MARGIN_SPREAD if cash < 0 else 0.0)
    cash *= 1 + r * gap / 365
    if dates[i] in divmap:
        d = divmap[dates[i]]
        sh_base += sh_base * d / px[i]
        sh += sh * d / px[i]
    cash += pnl_by_exit.get(dates[i], 0.0)
    if cash > BUFFER:
        sh += (cash - BUFFER) / px[i]
        cash = BUFFER
        sweeps += 1
    base[i] = sh_base * px[i]
    combo[i] = sh * px[i] + cash
    cash_path[i] = cash

pl.DataFrame({"date": dates, "spy": base, "spy_plus_overlay": combo,
              "cash": cash_path}).write_parquet(
    "reports/portfolio_v2_margin_overlay_curves.parquet")

yrs = (dates[-1] - dates[0]).days / 365.25
for name, v in [("SPY", base), ("SPY+overlay", combo)]:
    peak = np.maximum.accumulate(v)
    print(f"{name}: end=${v[-1]:,.0f} CAGR={(v[-1]/v[0])**(1/yrs)-1:+.2%} "
          f"maxDD={((v - peak)/peak).min():.2%}", flush=True)
on_margin = (cash_path < 0).mean()
print(f"cash: min=${cash_path.min():,.0f} sweeps into SPY={sweeps} "
      f"days on margin={on_margin:.1%} end=${cash_path[-1]:,.0f}", flush=True)
print(f"overlay added ${combo[-1] - base[-1]:+,.0f} vs buy & hold", flush=True)

# ---- plot -----------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, BASE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
C_COMBO, C_SPY = "#2a78d6", "#e87ba4"
SURF = "#fcfcfb"

fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(11, 7.5), dpi=160, sharex=True,
    gridspec_kw={"height_ratios": [2.6, 1], "hspace": 0.12})
fig.set_facecolor(SURF)
for a in (ax, ax2):
    a.set_facecolor(SURF)
    a.grid(True, axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        a.spines[side].set_color(BASE)
    a.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=9)

ax.plot(dates, combo, color=C_COMBO, lw=2,
        label="SPY + Rule v2 overlay on margin ($1k buffer, sweep wins to SPY)")
ax.plot(dates, base, color=C_SPY, lw=2, label="SPY buy & hold (divs reinvested)")
ax.annotate(f"SPY+overlay  ${combo[-1]:,.0f}", (dates[-1], combo[-1]),
            xytext=(8, 6), textcoords="offset points", color=C_COMBO,
            fontsize=9.5, fontweight="bold", va="center")
ax.annotate(f"SPY  ${base[-1]:,.0f}", (dates[-1], base[-1]),
            xytext=(8, -8), textcoords="offset points", color=C_SPY,
            fontsize=9.5, fontweight="bold", va="center")
ax.set_ylabel("Account value ($10k start)", color=MUTED, fontsize=9.5)
ax.yaxis.set_major_formatter(lambda x, _: f"${x/1000:,.0f}k")
ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
ax.set_title("Backtest 2018-08 → 2026-07: SPY vs SPY + Rule v2 overlay on margin",
             color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)

ax2.plot(dates, cash_path, color=C_COMBO, lw=1.6)
ax2.axhline(0, color=BASE, lw=1)
ax2.axhline(BUFFER, color=GRID, lw=1, ls="--")
ax2.fill_between(dates, cash_path, 0, where=cash_path < 0,
                 color="#e34948", alpha=0.25, linewidth=0)
ax2.annotate("$1k buffer cap", (dates[len(dates) // 10], BUFFER),
             xytext=(0, 5), textcoords="offset points", color=MUTED, fontsize=8)
ax2.set_ylabel("Overlay cash ($)", color=MUTED, fontsize=9.5)
ax2.annotate(f"min ${cash_path.min():,.0f}",
             (dates[int(np.argmin(cash_path))], cash_path.min()),
             xytext=(6, -10), textcoords="offset points", color="#e34948",
             fontsize=8.5, fontweight="bold")
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.text(0.01, 0.005,
         "Trade P&L books to cash at exit; cash>$1k sweeps into SPY same day; "
         "negative cash pays T-bill+1.5% margin interest; positive cash earns "
         "T-bill. Collateral vs SPY position. Burned data — descriptive only.",
         color=MUTED, fontsize=7.5)
fig.subplots_adjust(left=0.08, right=0.84, top=0.93, bottom=0.07)
fig.savefig("reports/portfolio_v2_margin_overlay.png", facecolor=SURF)
print("wrote reports/portfolio_v2_margin_overlay.png", flush=True)
