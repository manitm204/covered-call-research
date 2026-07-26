"""EXPLORATORY IC bridge test: the other project's top vol-normalized-IC
factors, scored against ACTUAL options P&L on our unfiltered panels.

For each ETF: Spearman IC of factor value at entry vs per-trade P&L
(sign-flipped so positive = factor's seller-favorable side per the price
study), plus on/off mean split at the panel median. Descriptive only.

Freshly computed here (not in existing feature frames): sector_corr_60
(mean pairwise 60d corr of the 9 SPDR sector ETFs), ret_84d (4-month trend
on the underlying index), vol_rank_252 (vol index percentile over 1y).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]
AUX = "data/normalized/aux"


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
    out = []
    iu = np.triu_indices(len(SECTORS), k=1)
    for i in range(59, len(dates)):
        C = np.corrcoef(R[i - 59:i + 1].T)
        out.append({"date": dates[i], "sector_corr_60": float(C[iu].mean())})
    return pl.DataFrame(out)


def series_features(index_path: str, vol_path: str) -> pl.DataFrame:
    idx = pl.read_parquet(index_path).sort("date")
    idx = idx.with_columns(
        (pl.col("close") / pl.col("close").shift(84) - 1).alias("ret_84d"))
    vol = pl.read_parquet(vol_path).sort("date").rename({"close": "vol_close"})
    vol = vol.with_columns(
        pl.col("vol_close").rank().rolling_map(  # placeholder replaced below
            lambda s: s, window_size=1).alias("_tmp")).drop("_tmp")
    v = vol["vol_close"].to_numpy()
    rank = np.full(len(v), np.nan)
    for i in range(251, len(v)):
        w = v[i - 251:i + 1]
        rank[i] = (w <= v[i]).mean()
    vol = vol.with_columns(pl.Series("vol_rank_252", rank))
    return idx.select("date", "ret_84d").join(
        vol.select("date", "vol_rank_252"), on="date", how="full", coalesce=True).sort("date")


SC60 = sector_corr_60()

PANELS = {
    "SPY": ("results/ablation_full/signal_study_panel.parquet",
            f"{AUX}/SPX.parquet", f"{AUX}/VIX.parquet"),
    "QQQ": ("results/qqq_prereg/regime_scan_panel.parquet",
            f"{AUX}/NDX.parquet", f"{AUX}/VXN.parquet"),
    "IWM": ("results/iwm_prereg/regime_scan_panel.parquet",
            f"{AUX}/RUT.parquet", f"{AUX}/RVX.parquet"),
}

# (factor, column, good_side) — good side = seller-favorable per the price
# study's vol-normalized IC sign ("lo" = low values good for the call seller)
FACTORS = {
    "SPY": [
        ("sector_corr_60", "sector_corr_60", "lo"),
        ("4m_trend", "ret_84d", "hi"),
        ("rsi_14", "rsi_14", "hi"),
        ("absorption_shift", "absorption_chg_20d", "lo"),
        ("price_vs_ma200", "dist_ma200", "hi"),
        ("own_vol (VIX)", "vix_level", "lo"),       # extra: IC says dead after norm
        ("iv_minus_rv20", "iv_minus_rv20", "hi"),   # extra: our old confirmation
    ],
    "QQQ": [
        ("4m_trend", "ret_84d", "hi"),
        ("sector_corr_60", "sector_corr_60", "lo"),
        ("absorption_shift", "absorption_chg_20d", "lo"),
        ("beta_to_spy_60", "qqq_beta_spy_60", "hi"),
        ("price_vs_ma200", "dist_ma200", "hi"),
        ("own_vol (VXN)", "vix_level", "lo"),       # extra
        ("rsi_14", "rsi_14", "hi"),                 # extra
    ],
    "IWM": [
        ("sector_corr_60", "sector_corr_60", "lo"),
        ("own_vol (RVX)", "vix_level", "lo"),
        ("rsi_14", "rsi_14", "hi"),
        ("4m_trend", "ret_84d", "hi"),
        ("iv_rank_252", "vol_rank_252", "lo"),
        ("absorption_shift", "absorption_chg_20d", "lo"),  # extra
        ("beta_to_spy_60", "iwm_beta_spy_60", "hi"),       # extra: dead in price study
    ],
}

for sym, (panel_path, index_path, vol_path) in PANELS.items():
    panel = pl.read_parquet(panel_path)
    extra = series_features(index_path, vol_path).join(SC60, on="date", how="left")
    # lag extras by one session to match feature convention
    extra = extra.sort("date").with_columns(
        pl.col("ret_84d", "vol_rank_252", "sector_corr_60").shift(1))
    panel = panel.join(extra, left_on="entry_date", right_on="date", how="left")
    pnl = panel["pnl"].to_numpy()
    print(f"\n{'='*76}\n{sym}: panel n={len(pnl)} mean=${pnl.mean():.1f}")
    print(f"{'factor':18s} {'IC(P&L)':>8s} {'p':>6s} | {'good-side mean$':>15s} "
          f"{'bad-side mean$':>14s} {'gap$':>7s} {'n_ok':>5s}")
    for name, col, good in FACTORS[sym]:
        if col not in panel.columns:
            print(f"{name:18s}  MISSING COLUMN {col}")
            continue
        v = panel[col].to_numpy().astype(float)
        ok = np.isfinite(v) & np.isfinite(pnl)
        if ok.sum() < 50:
            print(f"{name:18s}  n={ok.sum()} too few")
            continue
        rho, p = spearmanr(v[ok], pnl[ok])
        # sign-adjust: positive IC(P&L) = factor helps the seller in its
        # price-study direction
        adj = rho if good == "hi" else -rho
        med = np.median(v[ok])
        side_hi = pnl[ok & (v > med)]
        side_lo = pnl[ok & (v <= med)]
        g, b = (side_hi, side_lo) if good == "hi" else (side_lo, side_hi)
        print(f"{name:18s} {adj:+8.3f} {p:6.3f} | {g.mean():15.1f} {b.mean():14.1f} "
              f"{g.mean() - b.mean():7.1f} {ok.sum():5d}")
