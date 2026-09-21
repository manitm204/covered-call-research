"""Redo the IC / rank-IC / breach-probability regime-signal study (in the style
of the 2026-07 'Unified Rule v2' research notes) but scored against the target
that actually matters for a covered-call writer: does the underlying breach an
OTM strike within the next month? Real daily price data, SPY/QQQ/IWM, their own
vol indices (VIX/VXN/RVX), 2005-01 -> latest. Weekly (Friday) sampling to match
the original study's cadence; targets computed from the underlying daily series.

Outputs results/covered_call/signal_research.json for the report to consume,
and prints the derived per-ticker rip-risk rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PL = Path("data/normalized/prices_long")
START = pd.Timestamp("2005-01-01")
HORIZON = 21  # trading days ~ 1 month
FUNDS = {"SPY": "VIX", "QQQ": "VXN", "IWM": "RVX"}
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
RNG_SEED = 11


def load_px(sym: str) -> pd.Series:
    d = pd.read_parquet(PL / f"{sym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["adjClose"].sort_index()


def load_vol(sym: str) -> pd.Series:
    d = pd.read_parquet(PL / f"{sym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    col = "vix" if "vix" in d.columns else "close"
    return d.set_index("date")[col].sort_index()


def rsi14(px: pd.Series) -> pd.Series:
    delta = px.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def rolling_beta(ret: pd.Series, mkt_ret: pd.Series, window: int = 60) -> pd.Series:
    cov = ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    return cov / var


def fwd_max_ret(px: pd.Series, horizon: int) -> pd.Series:
    """Max close over the next `horizon` sessions, as a return from today."""
    fwd = pd.concat({i: px.shift(-i) for i in range(1, horizon + 1)}, axis=1)
    return fwd.max(axis=1) / px - 1.0


def block_bootstrap_ic(x: np.ndarray, y: np.ndarray, block: int = 8,
                        n_boot: int = 1500, seed: int = RNG_SEED):
    n = len(x)
    if n < 20:
        return float("nan"), (float("nan"), float("nan"))
    rho0 = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        ii = idx[b]
        vals[b] = spearmanr(x[ii], y[ii]).statistic
    lo, hi = np.quantile(vals, [0.05, 0.95])
    return float(rho0), (float(lo), float(hi))


def bucket_diff_ci(tgt: pd.Series, cond: pd.Series, block: int = 8,
                    n_boot: int = 1500, seed: int = RNG_SEED):
    """Mean(tgt | cond) - Mean(tgt | ~cond), block-bootstrap 90% CI."""
    df = pd.DataFrame({"c": cond, "t": tgt}).dropna()
    if df["c"].sum() < 10 or (~df["c"]).sum() < 10:
        return None
    point = df.loc[df.c, "t"].mean() - df.loc[~df.c, "t"].mean()
    n = len(df)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    c = df["c"].to_numpy()
    t = df["t"].to_numpy()
    starts = rng.integers(0, max(1, n - block + 1), size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        ii = idx[b]
        cc, tt = c[ii], t[ii]
        if cc.sum() < 3 or (~cc.astype(bool)).sum() < 3:
            vals[b] = np.nan
            continue
        vals[b] = tt[cc.astype(bool)].mean() - tt[~cc.astype(bool)].mean()
    vals = vals[~np.isnan(vals)]
    lo, hi = np.quantile(vals, [0.05, 0.95]) if len(vals) else (np.nan, np.nan)
    return dict(point=round(float(point) * 100, 3), lo=round(float(lo) * 100, 3),
               hi=round(float(hi) * 100, 3), n=int(df.c.sum()))


def build():
    px = {s: load_px(s) for s in FUNDS}
    vol = {s: load_vol(v) for s, v in FUNDS.items()}
    sector_px = {}
    for s in SECTORS:
        d = pd.read_parquet(PL / "sectors" / f"{s}.parquet")
        d["date"] = pd.to_datetime(d["date"])
        sector_px[s] = d.set_index("date")["adjClose"].sort_index()
    sector_ret = pd.concat({s: sector_px[s].pct_change() for s in SECTORS}, axis=1)
    # market-wide regime signal: mean pairwise 60d correlation among the 9
    # sector SPDRs (is the market trading as one tide, or fragmented?) —
    # NOT each fund's own correlation to the sector average, which is nearly
    # tautological for a broad index like SPY (it IS ~a sum of those sectors)
    _pairs = [(sector_ret[a], sector_ret[b]) for i, a in enumerate(SECTORS) for b in SECTORS[i + 1:]]
    market_sector_corr_60 = pd.concat([x.rolling(60).corr(y) for x, y in _pairs], axis=1).mean(axis=1)

    eigen = pd.read_parquet(PL / "eigen_features_long.parquet")
    absorption = eigen["ar1"]
    absorption_shift = absorption.diff(20)

    spy_ret = px["SPY"].pct_change()

    signals, targets = {}, {}
    for sym, vsym in FUNDS.items():
        p = px[sym]
        ret = p.pct_change()
        v = vol[sym].reindex(p.index).ffill()
        rv20 = ret.rolling(20).std() * np.sqrt(252)
        ma200 = p.rolling(200).mean()
        dd = p / p.rolling(252, min_periods=60).max() - 1.0
        sig = pd.DataFrame(index=p.index)
        sig["rsi14"] = rsi14(p)
        sig["sector_corr_60"] = market_sector_corr_60.reindex(p.index).ffill()
        sig["trend_84d"] = p.pct_change(84)
        sig["px_vs_ma200"] = p / ma200 - 1.0
        sig["absorption_shift"] = absorption_shift.reindex(p.index).ffill()
        sig["vol_own"] = v
        sig["vol_rank_252"] = v.rolling(252, min_periods=100).apply(
            lambda s: (s < s.iloc[-1]).mean(), raw=False)
        sig["vrp_proxy"] = v / 100.0 - rv20
        sig["deep_dd_60"] = dd.rolling(60).min()
        if sym != "SPY":
            sig["beta_spy_60"] = rolling_beta(ret, spy_ret.reindex(p.index), 60)
        else:
            sig["beta_spy_60"] = 1.0
        sig = sig.shift(1)  # point-in-time: known at prior close

        implied_sigma_m = (v / 100.0) * np.sqrt(HORIZON / 252)
        fwd_ret = p.shift(-HORIZON) / p - 1.0
        fmax = fwd_max_ret(p, HORIZON)
        tgt = pd.DataFrame(index=p.index)
        tgt["fwd_ret"] = fwd_ret
        tgt["fwd_ret_rel_spy"] = fwd_ret - (px["SPY"].shift(-HORIZON) / px["SPY"] - 1.0).reindex(p.index)
        tgt["upside_tail"] = fmax
        tgt["vol_norm_ret"] = fwd_ret / implied_sigma_m
        for mult in (1.0, 1.2, 1.5):
            tgt[f"breach_{mult}"] = (fmax > mult * implied_sigma_m).astype(float)

        signals[sym] = sig
        targets[sym] = tgt

    # weekly (Friday) sampling window, matching the reference study's cadence
    def to_weekly(df):
        w = df.resample("W-FRI").last()
        return w[(w.index >= START)]

    wsig = {s: to_weekly(signals[s]) for s in FUNDS}
    wtgt = {s: to_weekly(targets[s]) for s in FUNDS}

    sig_cols = ["rsi14", "sector_corr_60", "trend_84d", "px_vs_ma200",
               "absorption_shift", "vol_own", "vrp_proxy", "vol_rank_252", "beta_spy_60"]
    tgt_cols = ["fwd_ret", "upside_tail", "vol_norm_ret", "breach_1.2"]

    ic_table = {}  # tgt -> sym -> sig -> {ic, lo, hi, n}
    for tc in tgt_cols:
        ic_table[tc] = {}
        for sym in FUNDS:
            ic_table[tc][sym] = {}
            t = wtgt[sym][tc]
            for sc in sig_cols:
                s = wsig[sym][sc]
                df = pd.DataFrame({"s": s, "t": t}).dropna()
                if len(df) < 30:
                    continue
                ic, (lo, hi) = block_bootstrap_ic(df["s"].to_numpy(), df["t"].to_numpy())
                ic_table[tc][sym][sc] = dict(ic=round(ic, 3), lo=round(lo, 3), hi=round(hi, 3),
                                             n=len(df), sig=bool(lo > 0 or hi < 0))

    # breach rates per fund at the 3 multipliers
    breach_rates = {}
    for sym in FUNDS:
        breach_rates[sym] = {}
        for mult in (1.0, 1.2, 1.5):
            s = wtgt[sym][f"breach_{mult}"].dropna()
            breach_rates[sym][str(mult)] = round(float(s.mean()) * 100, 1)
        breach_rates[sym]["median_upside_tail"] = round(float(wtgt[sym]["upside_tail"].dropna().median()) * 100, 2)

    # vol terciles per fund (frozen thresholds derived from this sample)
    vol_tercile = {sym: round(float(wsig[sym]["vol_own"].dropna().quantile(2 / 3)), 1) for sym in FUNDS}

    # exploratory cutoffs, computed per-fund from each signal's own weekly
    # distribution: lower/upper tercile splits for sector correlation and
    # the vol index, and a top-decile split for the 4-month trend — testing
    # a wider range of thresholds than the fixed round numbers above.
    cutoffs = {sym: {} for sym in FUNDS}
    for sym in FUNDS:
        s = wsig[sym]
        cutoffs[sym]["sector_corr_lo_tercile"] = round(float(s["sector_corr_60"].dropna().quantile(1 / 3)), 3)
        cutoffs[sym]["sector_corr_hi_tercile"] = round(float(s["sector_corr_60"].dropna().quantile(2 / 3)), 3)
        cutoffs[sym]["trend_84d_top_decile"] = round(float(s["trend_84d"].dropna().quantile(0.90)), 4)
        cutoffs[sym]["vol_lo_tercile"] = round(float(s["vol_own"].dropna().quantile(1 / 3)), 1)
        cutoffs[sym]["vol_hi_tercile"] = vol_tercile[sym]
        cutoffs[sym]["rsi14_lo_tercile"] = round(float(s["rsi14"].dropna().quantile(1 / 3)), 1)

    # threshold/bucket forest-plot data for the signals that matter for breach risk
    thresholds = {
        "rsi14_lt_40": lambda s: s["rsi14"] < 40,
        "sector_corr_lt_0.45": lambda s: s["sector_corr_60"] < 0.45,
        "sector_corr_gt_0.70": lambda s: s["sector_corr_60"] > 0.70,
        "trend_84d_lt_0": lambda s: s["trend_84d"] < 0,
        "px_below_ma200": lambda s: s["px_vs_ma200"] < 0,
        "deep_dd_60_lt_-0.12": lambda s: s["deep_dd_60"] <= -0.12,
        "vol_top_tercile": None,  # filled per-fund below
        "sector_corr_lo_tercile": None,
        "sector_corr_hi_tercile": None,
        "trend_84d_top_decile": None,
        "vol_lo_tercile": None,
        "vol_hi_tercile": None,
        "rsi14_lo_tercile": None,
    }
    forest = {}
    for sym in FUNDS:
        forest[sym] = {}
        s = wsig[sym]
        t_breach = wtgt[sym]["breach_1.2"]
        t_upside = wtgt[sym]["upside_tail"]
        t_fwdret = wtgt[sym]["fwd_ret"]
        vt = vol_tercile[sym]
        co = cutoffs[sym]
        conds = dict(thresholds)
        conds["vol_top_tercile"] = lambda s, vt=vt: s["vol_own"] >= vt
        conds["sector_corr_lo_tercile"] = lambda s, c=co["sector_corr_lo_tercile"]: s["sector_corr_60"] < c
        conds["sector_corr_hi_tercile"] = lambda s, c=co["sector_corr_hi_tercile"]: s["sector_corr_60"] > c
        conds["trend_84d_top_decile"] = lambda s, c=co["trend_84d_top_decile"]: s["trend_84d"] > c
        conds["vol_lo_tercile"] = lambda s, c=co["vol_lo_tercile"]: s["vol_own"] < c
        conds["vol_hi_tercile"] = lambda s, c=co["vol_hi_tercile"]: s["vol_own"] > c
        conds["rsi14_lo_tercile"] = lambda s, c=co["rsi14_lo_tercile"]: s["rsi14"] < c
        for name, fn in conds.items():
            cond = fn(s).reindex(t_breach.index)
            r_breach = bucket_diff_ci(t_breach, cond)
            r_upside = bucket_diff_ci(t_upside, cond)
            r_fwdret = bucket_diff_ci(t_fwdret, cond)
            forest[sym][name] = dict(breach=r_breach, upside_tail=r_upside, fwd_ret=r_fwdret)

    out = dict(
        start=str(START.date()), horizon_days=HORIZON,
        breach_rates=breach_rates, vol_tercile=vol_tercile, cutoffs=cutoffs,
        ic_table=ic_table, forest=forest,
        n_weekly_obs={sym: int(len(wsig[sym].dropna(how="all"))) for sym in FUNDS},
    )
    Path("results/covered_call").mkdir(parents=True, exist_ok=True)
    Path("results/covered_call/signal_research.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote results/covered_call/signal_research.json")
    for sym in FUNDS:
        print(sym, "vol tercile:", vol_tercile[sym], "breach@1.2σ:", breach_rates[sym]["1.2"], "%")


if __name__ == "__main__":
    build()
