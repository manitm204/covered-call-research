"""G1 signal pre-validation on 2000-2017 daily prices (no options data involved).

Pre-declared variants only (research_plan.md §2, gate G1). For each signal we compare
the conditional forward-return distribution against the unconditional one on the SAME
dates universe, with a moving-block bootstrap CI on the mean difference and tail-odds
ratios (long calls monetize the right tail, so P(fwd > +5%/+10%) matters as much as
the mean).

Outputs: results/level2/g1_signal_prevalidation.csv (+ .md summary), and rows
appended to experiment_log.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from l2_common import append_experiment_log  # noqa: E402

RNG = np.random.default_rng(20260729)
START, END = "2000-01-01", "2017-12-31"
HORIZONS = (21, 42)


def load(sym: str) -> pd.DataFrame:
    px = pd.read_parquet(f"data/normalized/prices_long/{sym}.parquet")
    vix = pd.read_parquet("data/normalized/prices_long/VIX.parquet")
    df = px.merge(vix, on="date", how="left").set_index("date").sort_index()
    df["ret1"] = df.adjClose.pct_change()
    df["ma20"] = df.adjClose.rolling(20).mean()
    df["ma50"] = df.adjClose.rolling(50).mean()
    df["ma200"] = df.adjClose.rolling(200).mean()
    df["ret5"] = df.adjClose.pct_change(5)
    df["rv20"] = df.ret1.rolling(20).std() * np.sqrt(252) * 100
    df["roll_max252"] = df.adjClose.rolling(252, min_periods=60).max()
    df["dd"] = df.adjClose / df.roll_max252 - 1.0
    for h in HORIZONS:
        df[f"fwd{h}"] = df.adjClose.shift(-h) / df.adjClose - 1.0
        df[f"fwdrv{h}"] = (
            df.ret1.shift(-h).rolling(h).std().shift(-(h - 1)) * np.sqrt(252) * 100
        )
    # signals are LAGGED one session: what was knowable at yesterday's close governs
    # today's (15:30-style) decision
    return df


def block_bootstrap_diff(cond: np.ndarray, uncond: np.ndarray, block: int, n_boot: int = 2000):
    """CI on mean(cond) - mean(uncond) with moving-block resampling of each."""

    def mbb_mean(x: np.ndarray) -> np.ndarray:
        n = len(x)
        k = max(1, int(np.ceil(n / block)))
        starts = RNG.integers(0, max(1, n - block + 1), size=(n_boot, k))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
        return x[idx].mean(axis=1)

    d = mbb_mean(cond) - mbb_mean(uncond)
    return float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))


def stats_row(df, mask, h, sym, sig_name):
    d = df.loc[START:END]
    m = mask.reindex(d.index).fillna(False).astype(bool).shift(1).fillna(False)
    cond = d.loc[m, f"fwd{h}"].dropna()
    uncond = d[f"fwd{h}"].dropna()
    if len(cond) < 60:
        return None
    lo, hi = block_bootstrap_diff(cond.values, uncond.values, block=2 * h)
    # episode count: contiguous runs of the signal
    runs = (m != m.shift(1)).cumsum()[m]
    episodes = runs.nunique()
    return dict(
        symbol=sym, signal=sig_name, horizon=h, n_days=len(cond), n_episodes=int(episodes),
        pct_days=round(100 * m.mean(), 1),
        cond_mean=round(cond.mean() * 100, 2), uncond_mean=round(uncond.mean() * 100, 2),
        diff_ci90_lo=round(lo * 100, 2), diff_ci90_hi=round(hi * 100, 2),
        cond_p_gt5=round((cond > 0.05).mean() * 100, 1), uncond_p_gt5=round((uncond > 0.05).mean() * 100, 1),
        cond_p_gt10=round((cond > 0.10).mean() * 100, 1), uncond_p_gt10=round((uncond > 0.10).mean() * 100, 1),
        cond_p_lt_m10=round((cond < -0.10).mean() * 100, 1), uncond_p_lt_m10=round((uncond < -0.10).mean() * 100, 1),
        cond_vol=round(cond.std() * 100, 1), uncond_vol=round(uncond.std() * 100, 1),
    )


def main():
    rows = []
    for sym in ["SPY", "QQQ", "IWM"]:
        df = load(sym)
        d = df
        recent_deep = {x: (d.dd.rolling(60).min() <= -x / 100) for x in (8, 10, 15)}
        signals = {}
        # H1 rebound variants: was in deep drawdown recently AND turn confirmation
        for x in (8, 10, 15):
            signals[f"H1_dd{x}_ma20"] = recent_deep[x] & (d.adjClose > d.ma20)
            signals[f"H1_dd{x}_ret5"] = recent_deep[x] & (d.ret5 >= 0.02)
            signals[f"H1_dd{x}_zone"] = recent_deep[x]
        # H2 trend
        signals["H2_above_ma200"] = d.adjClose > d.ma200
        signals["H2_above_ma200_and_ma50"] = (d.adjClose > d.ma200) & (d.adjClose > d.ma50)
        # H4 control (puts side)
        signals["H4_below_ma200"] = d.adjClose < d.ma200
        for name, mask in signals.items():
            for h in HORIZONS:
                r = stats_row(df, mask, h, sym, name)
                if r:
                    rows.append(r)
        # H3: VIX vs RV — does VIX<RV20 predict fwd RV >= VIX? (straddle cheapness)
        for h in HORIZONS:
            dd = df.loc[START:END]
            m = (dd.vix < dd.rv20).shift(1).fillna(False)
            gap_cond = (dd.loc[m, f"fwdrv{h}"] - dd.loc[m, "vix"]).dropna()
            gap_unc = (dd[f"fwdrv{h}"] - dd.vix).dropna()
            if len(gap_cond) > 60:
                lo, hi = block_bootstrap_diff(gap_cond.values, gap_unc.values, block=2 * h)
                rows.append(dict(symbol=sym, signal="H3_vix_lt_rv20", horizon=h,
                                 n_days=len(gap_cond), n_episodes=int(((m != m.shift(1)).cumsum()[m]).nunique()),
                                 pct_days=round(100 * m.mean(), 1),
                                 cond_mean=round(gap_cond.mean(), 2), uncond_mean=round(gap_unc.mean(), 2),
                                 diff_ci90_lo=round(lo, 2), diff_ci90_hi=round(hi, 2),
                                 cond_p_gt5=np.nan, uncond_p_gt5=np.nan, cond_p_gt10=np.nan,
                                 uncond_p_gt10=np.nan, cond_p_lt_m10=np.nan, uncond_p_lt_m10=np.nan,
                                 cond_vol=round(gap_cond.std(), 1), uncond_vol=round(gap_unc.std(), 1)))
    out = pd.DataFrame(rows)
    Path("results/level2").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/level2/g1_signal_prevalidation.csv", index=False)
    for _, r in out.iterrows():
        append_experiment_log(
            hypothesis=r.signal.split("_")[0], partition="preval_2000_2017",
            underlying=r.symbol, config=f"{r.signal}_h{r.horizon}", scenario="n/a",
            n=int(r.n_days), metric="fwd_mean_diff_pct" if not r.signal.startswith("H3") else "fwdRV_minus_VIX_diff",
            value=float(r.cond_mean - r.uncond_mean), ci_lo=float(r.diff_ci90_lo), ci_hi=float(r.diff_ci90_hi),
            verdict="", notes=f"episodes={r.n_episodes}, pct_days={r.pct_days}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
