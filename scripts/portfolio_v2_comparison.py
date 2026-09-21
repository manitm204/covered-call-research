"""Account-level comparison: T-bills vs Unified Rule v2 (3 ETFs) vs SPY.

Runs the FROZEN Unified Rule v2 (docs/CONCLUSIONS.md 2026-07-26) once per ETF:
  veto (rebound 10%/60d OR below MA200 OR own-vol top tercile, lagged 1d)
  + confirmations: SPY/QQQ seccorr60<0.45 AND ret84>10%; IWM seccorr60<0.45.
Structure: prereg configs, cap-2, 1 contract, base fills, hold to expiry.

Then builds daily $10k account curves, 2018-08-01 -> 2026-07-22:
  TBILL — 4w T-bill roll, ACT/365 calendar-day accrual
  STRAT — same T-bill accrual on the full balance (treasuries-as-collateral
          assumption) + realized options P&L of all three ETF overlays,
          booked on exit date
  SPY   — buy & hold, dividends reinvested on ex-date

Descriptive only: the chain data is burned and the rule was frozen on these
same panels — see the v2 caveats. Output: reports/portfolio_v2_comparison.png
plus trades/curves parquet next to it.
"""

from __future__ import annotations

import bisect
import datetime as dt
import sys

import numpy as np
import polars as pl

from xsp_research.backtest.american import DividendCalendar
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.config import EXECUTION_SCENARIOS, load_strategy_config
from xsp_research.ingestion.file_provider import ParquetUnderlyingProvider, SeriesRatesProvider

sys.path.insert(0, "scripts")
from full_ablation_sweep import RATES, PreloadedOptionsProvider  # noqa: E402

W0, W1 = dt.date(2018, 8, 1), dt.date(2026, 7, 22)
CAPITAL = 10_000.0
AUX = "data/normalized/aux"
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]

# ETF specs and feature/veto builders duplicated from confirm_ablation.py,
# which cannot be imported (its ablation loop runs at module level).
ETFS = {
    "SPY": {
        "config": "configs/strategy_spy_hold_to_expiry.yaml", "width": 8.0,
        "glob": "data/normalized/options/spy/chain_*.parquet",
        "und": "data/normalized/options/spy/underlying_eod.parquet",
        "div": f"{AUX}/SPY_DIVIDENDS.parquet",
        "index": f"{AUX}/SPX.parquet", "vol": f"{AUX}/VIX.parquet",
        "features": "reports/experiments/spy_regime_rsi70_ivrich-20260723-204901-b6454793/features.parquet",
        "signals": {
            "seccorr60": ("sector_corr_60", "<", 0.45),
            "trend84":   ("ret_84d", ">", 0.10),
        },
    },
    "QQQ": {
        "config": "configs/strategy_qqq_prereg.yaml", "width": None,
        "glob": "data/normalized/options/qqq/chain_*.parquet",
        "und": "data/normalized/options/qqq/underlying_eod.parquet",
        "div": f"{AUX}/QQQ_DIVIDENDS.parquet",
        "index": f"{AUX}/NDX.parquet", "vol": f"{AUX}/VXN.parquet",
        "features": "results/qqq_prereg/features_qqq.parquet",
        "signals": {
            "seccorr60": ("sector_corr_60", "<", 0.45),
            "trend84":   ("ret_84d", ">", 0.10),
        },
    },
    "IWM": {
        "config": "configs/strategy_iwm_prereg_h1.yaml", "width": None,
        "glob": "data/normalized/options/iwm/chain_*.parquet",
        "und": "data/normalized/options/iwm/underlying_eod.parquet",
        "div": f"{AUX}/IWM_DIVIDENDS.parquet",
        "index": f"{AUX}/RUT.parquet", "vol": f"{AUX}/RVX.parquet",
        "features": "results/iwm_prereg/features_iwm.parquet",
        "signals": {
            "seccorr60": ("sector_corr_60", "<", 0.45),
        },
    },
}


def sector_corr_60() -> pl.DataFrame:
    frames = [pl.read_parquet(f"{AUX}/{s}.parquet").sort("date")
              .with_columns((pl.col("close") / pl.col("close").shift(1)).log().alias(s))
              .select("date", s) for s in SECTORS]
    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, on="date", how="inner")
    df = df.drop_nulls().sort("date")
    dates = df["date"].to_list()
    R = df.select(SECTORS).to_numpy()
    iu = np.triu_indices(len(SECTORS), k=1)
    out = [{"date": dates[i], "sector_corr_60": float(np.corrcoef(R[i - 59:i + 1].T)[iu].mean())}
           for i in range(59, len(dates))]
    return pl.DataFrame(out)


def series_extras(index_path: str, vol_path: str) -> pl.DataFrame:
    idx = pl.read_parquet(index_path).sort("date").with_columns(
        (pl.col("close") / pl.col("close").shift(84) - 1).alias("ret_84d"))
    vol = pl.read_parquet(vol_path).sort("date")
    v = vol["close"].to_numpy()
    rank = np.full(len(v), np.nan)
    for i in range(251, len(v)):
        rank[i] = (v[i - 251:i + 1] <= v[i]).mean()
    vol = vol.with_columns(pl.Series("vol_rank_252", rank))
    return idx.select("date", "ret_84d").join(
        vol.select("date", "vol_rank_252"), on="date", how="full", coalesce=True).sort("date")


def veto_flags(index_path: str, vol_path: str) -> tuple[list, dict]:
    idx = pl.read_parquet(index_path).sort("date")
    dates = idx["date"].to_list()
    closes = np.array(idx["close"].to_list())
    dd = closes / np.maximum.accumulate(closes) - 1
    dd_min60 = np.array([dd[max(0, i - 60):i + 1].min() for i in range(len(dd))])
    ma200 = np.full(len(closes), np.nan)
    for i in range(199, len(closes)):
        ma200[i] = closes[i - 199:i + 1].mean()
    vol = pl.read_parquet(vol_path).sort("date")
    vwin = vol.filter((pl.col("date") >= W0) & (pl.col("date") <= W1))
    vhi = float(np.quantile(vwin["close"].to_numpy(), 2 / 3))
    vmap = dict(zip(vol["date"].to_list(), vol["close"].to_list()))
    flags = {}
    for i, d in enumerate(dates):
        v = vmap.get(d)
        flags[d] = None if (np.isnan(ma200[i]) or v is None) else \
            bool(dd_min60[i] <= -0.10 or closes[i] < ma200[i] or v >= vhi)
    return sorted(flags), flags
CONFIRMATIONS = {
    "SPY": ["seccorr60", "trend84"],
    "QQQ": ["seccorr60", "trend84"],
    "IWM": ["seccorr60"],
}


def run_etf(sym: str) -> pl.DataFrame:
    spec = ETFS[sym]
    feats = pl.read_parquet(spec["features"]).join(
        series_extras(spec["index"], spec["vol"]).sort("date").with_columns(
            pl.col("ret_84d", "vol_rank_252").shift(1)),
        on="date", how="left").join(
        SC60.sort("date").with_columns(pl.col("sector_corr_60").shift(1)),
        on="date", how="left")
    rows = {r["date"]: r for r in feats.iter_rows(named=True)}
    vdates, vflags = veto_flags(spec["index"], spec["vol"])
    active = CONFIRMATIONS[sym]

    def gate(session):
        j = bisect.bisect_left(vdates, session) - 1
        if j < 0 or vflags[vdates[j]] is not False:
            return False, "veto"
        r = rows.get(session)
        if r is None:
            return False, "no row"
        for name in active:
            col, op, thr = spec["signals"][name]
            v = r.get(col)
            if v is None:
                return False, "na"
            if not ((v < thr) if op == "<" else (v > thr)):
                return False, "off"
        return True, "ok"

    cfg0 = load_strategy_config(spec["config"])
    sel = cfg0.selection.model_copy(update={"fixed_width": spec["width"]}) \
        if spec["width"] else cfg0.selection
    cfg = cfg0.model_copy(update={
        "selection": sel,
        "execution": EXECUTION_SCENARIOS["base"],
        "sizing": cfg0.sizing.model_copy(update={
            "contracts_per_entry": 1, "max_open_positions": 2}),
    })
    res = BacktestEngine(
        cfg,
        PreloadedOptionsProvider(spec["glob"], sym, cfg0.backtest.mark_time_et),
        ParquetUnderlyingProvider(spec["und"], sym),
        SeriesRatesProvider(RATES),
        execution_scenario="base",
        entry_gate=gate,
        dividends=DividendCalendar.from_parquet(spec["div"]),
    ).run()
    t = res.trades_frame()
    if not t.height:
        return pl.DataFrame()
    return t.with_columns(
        pl.lit(sym).alias("symbol"),
        pl.col("entry_ts").dt.convert_time_zone("America/New_York").dt.date()
        .alias("entry_date"),
        pl.col("exit_ts").dt.convert_time_zone("America/New_York").dt.date()
        .alias("exit_date"),
    ).select("symbol", "entry_date", "exit_date", "qty", "entry_credit",
             "realized_net")


SC60 = sector_corr_60()
all_trades = []
for sym in ETFS:
    t = run_etf(sym)
    per = t["realized_net"].to_numpy() / t["qty"].to_numpy()
    print(f"{sym}: n={t.height} mean=${per.mean():+.2f} total=${per.sum():+.2f} "
          f"win={(per > 0).mean():.0%} worst=${per.min():+.2f}", flush=True)
    all_trades.append(t)
trades = pl.concat(all_trades).sort("exit_date")
trades.write_parquet("reports/portfolio_v2_trades.parquet")

# ---- daily curves ---------------------------------------------------------
spy = pl.read_parquet(f"{AUX}/SPY.parquet").sort("date").filter(
    (pl.col("date") >= W0) & (pl.col("date") <= W1))
dates = spy["date"].to_list()
px = spy["close"].to_numpy()

rt = pl.read_parquet(RATES).sort("date")
rdates, rvals = rt["date"].to_list(), rt["rate"].to_list()


def tbill_rate(d: dt.date) -> float:
    j = bisect.bisect_right(rdates, d) - 1
    return rvals[j] if j >= 0 else 0.0


div = pl.read_parquet(f"{AUX}/SPY_DIVIDENDS.parquet")
divmap = dict(zip(div["ex_date"].to_list(), div["amount"].to_list()))

pnl_by_exit: dict[dt.date, float] = {}
for r in trades.iter_rows(named=True):
    pnl_by_exit[r["exit_date"]] = pnl_by_exit.get(r["exit_date"], 0.0) + r["realized_net"]

n = len(dates)
tbill = np.empty(n)
strat = np.empty(n)
spy_eq = np.empty(n)
tbill[0] = strat[0] = spy_eq[0] = CAPITAL
shares = CAPITAL / px[0]
for i in range(1, n):
    gap = (dates[i] - dates[i - 1]).days
    acc = 1 + tbill_rate(dates[i - 1]) * gap / 365
    tbill[i] = tbill[i - 1] * acc
    strat[i] = strat[i - 1] * acc + pnl_by_exit.get(dates[i], 0.0)
    if dates[i] in divmap:
        shares += shares * divmap[dates[i]] / px[i]
    spy_eq[i] = shares * px[i]

curves = pl.DataFrame({"date": dates, "tbill": tbill, "strat": strat, "spy": spy_eq})
curves.write_parquet("reports/portfolio_v2_curves.parquet")

yrs = (dates[-1] - dates[0]).days / 365.25
for name, v in [("TBILL", tbill), ("STRAT", strat), ("SPY", spy_eq)]:
    peak = np.maximum.accumulate(v)
    print(f"{name}: end=${v[-1]:,.0f} CAGR={(v[-1]/v[0])**(1/yrs)-1:+.2%} "
          f"maxDD={((v - peak)/peak).min():.2%}", flush=True)

# ---- plot -----------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, BASE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
C_STRAT, C_TBILL, C_SPY = "#2a78d6", "#008300", "#e87ba4"
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

ax.plot(dates, spy_eq, color=C_SPY, lw=2, label="SPY buy & hold (divs reinvested)")
ax.plot(dates, tbill, color=C_TBILL, lw=2, label="T-bills (4w roll)")
ax.plot(dates, strat, color=C_STRAT, lw=2,
        label="Unified Rule v2 on SPY+QQQ+IWM (cash in T-bills)")
for v, c, txt in [(spy_eq, C_SPY, f"SPY  ${spy_eq[-1]:,.0f}"),
                  (strat, C_STRAT, f"Rule v2  ${strat[-1]:,.0f}"),
                  (tbill, C_TBILL, f"T-bills  ${tbill[-1]:,.0f}")]:
    ax.annotate(txt, (dates[-1], v[-1]), xytext=(8, 0),
                textcoords="offset points", color=c, fontsize=9.5,
                fontweight="bold", va="center")
ax.set_ylabel("Account value ($10k start)", color=MUTED, fontsize=9.5)
ax.yaxis.set_major_formatter(lambda x, _: f"${x/1000:,.0f}k")
ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
ax.set_title(
    "Backtest 2018-08 → 2026-07: T-bills vs Unified Rule v2 vs SPY",
    color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)

edates = [W0] + trades["exit_date"].to_list()
cum = np.concatenate([[0.0], trades["realized_net"].cum_sum().to_numpy()])
ax2.step(edates, cum, where="post", color=C_STRAT, lw=2)
ax2.axhline(0, color=BASE, lw=1)
ax2.set_ylabel("Options P&L alone ($)", color=MUTED, fontsize=9.5)
ax2.annotate(f"${cum[-1]:+,.0f}", (edates[-1], cum[-1]), xytext=(8, 0),
             textcoords="offset points", color=C_STRAT, fontsize=9.5,
             fontweight="bold", va="center")
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.text(0.01, 0.005,
         "Base fills, cap-2/ETF, 1 contract, hold to expiry. Strategy account: "
         "full balance accrues T-bill interest (treasuries as collateral) + "
         "realized spread P&L at exit. Burned data — descriptive, not validation.",
         color=MUTED, fontsize=7.5)
fig.subplots_adjust(left=0.08, right=0.86, top=0.93, bottom=0.07)
fig.savefig("reports/portfolio_v2_comparison.png", facecolor=SURF)
print("wrote reports/portfolio_v2_comparison.png", flush=True)
