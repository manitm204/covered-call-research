"""Cycle-2 G1 pre-validation on price data (2000/2007-2017, pre-options era).

Pre-declared variants only; every row logged to experiment_log.csv.

H8  VIX term-structure: VIX/VIX3M > 1 (backwardation) -> do forward returns/vol
    justify long puts (or avoiding long calls)?
H9  Cross-ETF momentum: hold the 6-3-1m blended momentum leader of SPY/QQQ/IWM
    (only if above its MA200) -> does the leader beat the average ETF forward?
H12 Seasonality: Nov-Apr vs May-Oct forward returns (the old chestnut, tested
    honestly).
H6  Credit gate (2007-2017 only, HYG history): equity above MA200 AND HYG above
    its MA100 vs equity-above-MA200 alone -> does the credit confirm cut left
    tail further?
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from l2_common import append_experiment_log  # noqa: E402

RNG = np.random.default_rng(20260730)
P = "data/normalized/prices_long"


def mbb_diff(c, u, block, n_boot=2000):
    def m(x):
        n = len(x); k = max(1, int(np.ceil(n / block)))
        st = RNG.integers(0, max(1, n - block + 1), size=(n_boot, k))
        idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
        return x[idx].mean(axis=1)
    d = m(c) - m(u)
    return float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))


def load_adj(sym):
    df = pd.read_parquet(f"{P}/{sym}.parquet")
    df["date"] = pd.to_datetime(df.date)
    return df.set_index("date").adjClose.sort_index()


def log_row(hyp, sym, cfg, n, metric, val, lo, hi, note=""):
    verdict = "PASS" if (lo > 0 if val >= 0 else hi < 0) else "fail"
    append_experiment_log(hypothesis=hyp, partition="preval_pre2018", underlying=sym,
                          config=cfg, scenario="n/a", n=n, metric=metric,
                          value=round(val, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                          verdict=verdict, notes=note)
    print(f"{hyp} {sym} {cfg}: {metric}={val:+.3f} CI90 [{lo:+.3f},{hi:+.3f}] {verdict} {note}")


def h8_term_structure():
    spy = load_adj("SPY")
    vix = pd.read_parquet(f"{P}/VIX.parquet"); vix["date"] = pd.to_datetime(vix.date)
    v3 = pd.read_parquet(f"{P}/VIX3M.parquet"); v3["date"] = pd.to_datetime(v3.date)
    df = pd.DataFrame(dict(px=spy)).join(vix.set_index("date").vix).join(
        v3.set_index("date").close.rename("vix3m")).dropna()
    df = df.loc["2006-08-01":"2017-12-31"]
    df["ret1"] = df.px.pct_change()
    df["bw"] = (df.vix / df.vix3m) > 1.0
    for h in (10, 21):
        df[f"fwd{h}"] = df.px.shift(-h) / df.px - 1
        m = df.bw.shift(1).fillna(False).astype(bool)
        cond = df.loc[m, f"fwd{h}"].dropna(); unc = df[f"fwd{h}"].dropna()
        lo, hi = mbb_diff(cond.values, unc.values, 2 * h)
        log_row("H8", "SPY", f"bw_vix_gt_vix3m_h{h}", len(cond),
                "fwd_mean_diff", cond.mean() - unc.mean(), lo, hi,
                f"cond_vol={cond.std():.3f} unc_vol={unc.std():.3f}")
        # for long puts you also need the move DOWN to exceed elevated IV; check
        # realized fwd vol vs entry VIX (is vol still underpriced in backwardation?)
        fwd_rv = df.ret1.shift(-h).rolling(h).std().shift(-(h - 1)) * np.sqrt(252) * 100
        gap_c = (fwd_rv[m] - df.vix[m]).dropna(); gap_u = (fwd_rv - df.vix).dropna()
        lo, hi = mbb_diff(gap_c.values, gap_u.values, 2 * h)
        log_row("H8", "SPY", f"bw_fwdRV_minus_VIX_h{h}", len(gap_c),
                "vol_gap_diff", gap_c.mean() - gap_u.mean(), lo, hi)


def h9_momentum_leader():
    px = {s: load_adj(s) for s in ("SPY", "QQQ", "IWM")}
    df = pd.DataFrame(px).dropna().loc["2001-01-01":"2017-12-31"]
    mom = 0.5 * df.pct_change(126) + 0.3 * df.pct_change(63) + 0.2 * df.pct_change(21)
    ma200 = df.rolling(200).mean()
    h = 21
    fwd = df.shift(-h) / df - 1
    leader = mom.idxmax(axis=1)
    above = df > ma200
    lead_ret, avg_ret = [], []
    for t in df.index[200:-h]:
        lt = leader.shift(1).loc[t]
        if pd.isna(lt) or not above.shift(1).loc[t, lt]:
            continue
        lead_ret.append(fwd.loc[t, lt])
        avg_ret.append(fwd.loc[t].mean())
    lead, avg = pd.Series(lead_ret).dropna(), pd.Series(avg_ret).dropna()
    lo, hi = mbb_diff(lead.values, avg.values, 2 * h)
    log_row("H9", "SPY/QQQ/IWM", f"mom_leader_vs_eqw_h{h}", len(lead),
            "fwd_mean_diff", lead.mean() - avg.mean(), lo, hi,
            "leader (above MA200) vs equal-weight of the three")


def h12_seasonality():
    for sym in ("SPY", "QQQ"):
        px = load_adj(sym).loc["2000-01-01":"2017-12-31"]
        h = 21
        fwd = (px.shift(-h) / px - 1).dropna()
        winter = fwd[fwd.index.month.isin([11, 12, 1, 2, 3, 4])]
        lo, hi = mbb_diff(winter.values, fwd.values, 2 * h)
        log_row("H12", sym, "nov_apr_h21", len(winter),
                "fwd_mean_diff", winter.mean() - fwd.mean(), lo, hi)


def h6_credit_gate():
    for sym in ("SPY", "QQQ"):
        px = load_adj(sym)
        hyg = load_adj("HYG") if (pd.io.common.file_exists(f"{P}/HYG.parquet")) else None
        if hyg is None:
            print("H6: no HYG in prices_long — pulling")
            from level2_research.fmp import pull_daily_history
            d = pull_daily_history("HYG", "2007-01-01", "2026-07-29")
            d.to_parquet(f"{P}/HYG.parquet", index=False)
            hyg = load_adj("HYG")
        df = pd.DataFrame(dict(px=px, hyg=hyg)).dropna().loc["2007-06-01":"2017-12-31"]
        h = 42
        df["fwd"] = df.px.shift(-h) / df.px - 1
        base_m = (df.px > df.px.rolling(200).mean()).shift(1).fillna(False).astype(bool)
        credit_ok = (df.hyg > df.hyg.rolling(100).mean()).shift(1).fillna(False).astype(bool)
        both = base_m & credit_ok
        a = df.loc[base_m, "fwd"].dropna()
        b = df.loc[both, "fwd"].dropna()
        # does the credit confirm reduce the left tail beyond MA200 alone?
        def tail(x):
            return (x < -0.05).mean()
        n_boot, block = 2000, 2 * h
        def mbb_tail(x):
            n = len(x); k = max(1, int(np.ceil(n / block)))
            st = RNG.integers(0, max(1, n - block + 1), size=(n_boot, k))
            idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
            return (x[idx] < -0.05).mean(axis=1)
        d = mbb_tail(b.values) - mbb_tail(a.values)
        lo, hi = float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))
        log_row("H6", sym, "ma200+hyg100_lefttail5_h42", len(b),
                "tail_prob_diff", tail(b) - tail(a), lo, hi,
                f"P(<-5%): both={tail(b):.3f} ma200only={tail(a):.3f}")


if __name__ == "__main__":
    h8_term_structure()
    h9_momentum_leader()
    h12_seasonality()
    h6_credit_gate()
