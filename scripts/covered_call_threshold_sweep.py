"""Full threshold-sweep companion to covered_call_signal_research.py.

For every regime signal, every fund (SPY/QQQ/IWM), and every breach
multiplier (1.0 / 1.2 / 1.5 sigma), sweep a grid of quantile cutoffs
(10..90 in steps of 10, i.e. decile/tercile/quintile/top-20-40-60-80% all
fall out of the same grid) and compute the breach probability in the
"top X%" and "bottom X%" buckets with a 90% block-bootstrap CI, plus the
unconditional (all-history) breach rate as a baseline.

Outputs results/covered_call/threshold_sweep.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PL = Path("data/normalized/prices_long")
START = pd.Timestamp("2005-01-01")
HORIZON = 21
FUNDS = {"SPY": "VIX", "QQQ": "VXN", "IWM": "RVX"}
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
RNG_SEED = 11
QUANTILES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
MULTS = [1.0, 1.2, 1.5]
N_BOOT = 1000
BLOCK = 8


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
    fwd = pd.concat({i: px.shift(-i) for i in range(1, horizon + 1)}, axis=1)
    return fwd.max(axis=1) / px - 1.0


def block_bootstrap_mean(x: np.ndarray, block: int = BLOCK, n_boot: int = N_BOOT,
                          seed: int = RNG_SEED):
    """90% block-bootstrap CI on mean(x). x is a 0/1 breach indicator array."""
    n = len(x)
    if n < 15:
        return None
    point = float(np.mean(x))
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    vals = x[idx].mean(axis=1)
    lo, hi = np.quantile(vals, [0.05, 0.95])
    return dict(point=round(point * 100, 2), lo=round(float(lo) * 100, 2),
                hi=round(float(hi) * 100, 2), n=int(n))


def bucket_result(tgt: pd.Series, cond: pd.Series):
    df = pd.DataFrame({"c": cond.astype(bool), "t": tgt}).dropna()
    top = block_bootstrap_mean(df.loc[df.c, "t"].to_numpy())
    bot = block_bootstrap_mean(df.loc[~df.c, "t"].to_numpy())
    return top, bot


def diff_ci(tgt: pd.Series, cond: pd.Series, block: int = BLOCK, n_boot: int = N_BOOT,
            seed: int = RNG_SEED):
    """Paired block-bootstrap CI on mean(top) - mean(bottom), same resample used
    for both groups each draw (not two independently-bootstrapped bars)."""
    df = pd.DataFrame({"c": cond.astype(bool), "t": tgt}).dropna()
    if df["c"].sum() < 10 or (~df["c"]).sum() < 10:
        return None
    c = df["c"].to_numpy()
    t = df["t"].to_numpy()
    n = len(df)
    point = t[c].mean() - t[~c].mean()
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    vals = np.full(n_boot, np.nan)
    for b in range(n_boot):
        ii = idx[b]
        cc, tt = c[ii], t[ii]
        if cc.sum() < 3 or (~cc).sum() < 3:
            continue
        vals[b] = tt[cc].mean() - tt[~cc].mean()
    vals = vals[~np.isnan(vals)]
    lo, hi = np.quantile(vals, [0.05, 0.95]) if len(vals) else (np.nan, np.nan)
    return dict(point=round(float(point) * 100, 2), lo=round(float(lo) * 100, 2),
                hi=round(float(hi) * 100, 2), n_top=int(c.sum()), n_bottom=int((~c).sum()))


def build():
    px = {s: load_px(s) for s in FUNDS}
    vol = {s: load_vol(v) for s, v in FUNDS.items()}
    sector_px = {}
    for s in SECTORS:
        d = pd.read_parquet(PL / "sectors" / f"{s}.parquet")
        d["date"] = pd.to_datetime(d["date"])
        sector_px[s] = d.set_index("date")["adjClose"].sort_index()
    sector_ret = pd.concat({s: sector_px[s].pct_change() for s in SECTORS}, axis=1)
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
        sig = sig.shift(1)

        implied_sigma_m = (v / 100.0) * np.sqrt(HORIZON / 252)
        fmax = fwd_max_ret(p, HORIZON)
        tgt = pd.DataFrame(index=p.index)
        for mult in MULTS:
            tgt[f"breach_{mult}"] = (fmax > mult * implied_sigma_m).astype(float)

        signals[sym] = sig
        targets[sym] = tgt

    def to_weekly(df):
        w = df.resample("W-FRI").last()
        return w[(w.index >= START)]

    wsig = {s: to_weekly(signals[s]) for s in FUNDS}
    wtgt = {s: to_weekly(targets[s]) for s in FUNDS}

    sig_cols = ["rsi14", "sector_corr_60", "trend_84d", "px_vs_ma200",
                "absorption_shift", "vol_own", "vrp_proxy", "vol_rank_252",
                "beta_spy_60", "deep_dd_60"]

    out = {"funds": list(FUNDS), "signals": sig_cols,
           "mults": [str(m) for m in MULTS], "quantiles": QUANTILES,
           "unconditional": {}, "sweep": {}}

    for sym in FUNDS:
        out["unconditional"][sym] = {}
        for mult in MULTS:
            t = wtgt[sym][f"breach_{mult}"].dropna()
            out["unconditional"][sym][str(mult)] = block_bootstrap_mean(t.to_numpy())

    for sym in FUNDS:
        print(f"=== {sym} ===")
        out["sweep"][sym] = {}
        s_df = wsig[sym]
        for sc in sig_cols:
            if sym == "SPY" and sc == "beta_spy_60":
                continue  # degenerate (beta to itself == 1)
            print(f"  {sc}")
            s = s_df[sc]
            out["sweep"][sym][sc] = {}
            for mult in MULTS:
                t = wtgt[sym][f"breach_{mult}"]
                out["sweep"][sym][sc][str(mult)] = {}
                for q in QUANTILES:
                    cutoff = float(s.dropna().quantile(q))
                    cond_top = (s >= cutoff)
                    top, bot = bucket_result(t, cond_top)
                    d = diff_ci(t, cond_top)
                    out["sweep"][sym][sc][str(mult)][f"{int(q*100)}"] = dict(
                        cutoff=round(cutoff, 4),
                        top_pct=round((1 - q) * 100, 0),
                        bottom_pct=round(q * 100, 0),
                        top=top, bottom=bot, diff=d,
                    )

    Path("results/covered_call").mkdir(parents=True, exist_ok=True)
    Path("results/covered_call/threshold_sweep.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote results/covered_call/threshold_sweep.json")


if __name__ == "__main__":
    build()
