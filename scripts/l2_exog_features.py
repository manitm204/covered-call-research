"""Build the cycle-2 extended feature matrix (point-in-time, all lagged).

Sources -> features (lag policy in parentheses):
- treasury curve (1 session): slope_10y2y, slope_10y3m, y10_level, slope chg 20d
- COT ES (5 calendar days: Tue data, Fri release): net non-commercial z-score 52w
- sector P/E weekly (1 session, ffill): median sector PE, tech/defensive PE ratio,
  13w change in median PE
- earnings calendar (2 days): trailing-90d S&P beat rate; trailing-90d median EPS
  YoY growth; upcoming-21d report count (earnings-season density). Constituent
  list is TODAY's S&P membership — survivorship caveat documented; acceptable for
  broad regime aggregates, not for stock selection.
- FRED macro (45 calendar days to clear release lag): CPI YoY, unemployment chg
  3m, sentiment level+chg, INDPRO YoY
- existing 39 daily features (already lagged >=1 session) incl. the eigen family:
  absorption_ratio, absorption_chg_20d, ncomp_80pct, pc1_loading_dispersion

Output: data/features/cycle2_features.parquet indexed by date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXOG = "data/normalized/exog"


def daily_index() -> pd.DatetimeIndex:
    spy = pd.read_parquet("data/normalized/prices_long/SPY.parquet")
    return pd.DatetimeIndex(pd.to_datetime(spy.date)).sort_values()


def treasury_features(idx):
    t = pd.read_parquet(f"{EXOG}/treasury_curve.parquet")
    t["date"] = pd.to_datetime(t.date)
    t = t.set_index("date").sort_index()
    f = pd.DataFrame(index=t.index)
    f["slope_10y2y"] = t.year10 - t.year2
    f["slope_10y3m"] = t.year10 - t.month3
    f["y10_level"] = t.year10
    f["slope_10y2y_chg20"] = f.slope_10y2y.diff(20)
    f["y10_chg20"] = f.y10_level.diff(20)
    return f.reindex(idx, method="ffill").shift(1)


def cot_features(idx):
    c = pd.read_parquet(f"{EXOG}/cot_es.parquet")
    c["date"] = pd.to_datetime(c.date)
    c = c.set_index("date").sort_index()
    net_cols = [x for x in c.columns if "noncomm" in x.lower() and "long" in x.lower()]
    short_cols = [x for x in c.columns if "noncomm" in x.lower() and "short" in x.lower()
                  and "spread" not in x.lower()]
    if not net_cols or not short_cols:
        return pd.DataFrame(index=idx)
    net = c[net_cols[0]] - c[short_cols[0]]
    z = (net - net.rolling(52).mean()) / net.rolling(52).std()
    f = pd.DataFrame({"cot_es_net_z52": z})
    f.index = f.index + pd.Timedelta(days=5)  # release lag
    return f.reindex(idx, method="ffill")


def sector_pe_features(idx):
    s = pd.read_parquet(f"{EXOG}/sector_pe_weekly.parquet")
    s["date"] = pd.to_datetime(s.date)
    piv = s.pivot_table(index="date", columns="sector", values="pe", aggfunc="mean")
    f = pd.DataFrame(index=piv.index)
    f["sector_pe_median"] = piv.median(axis=1)
    tech = piv.get("Technology")
    defens = piv[[c for c in ("Consumer Defensive", "Utilities") if c in piv]].mean(axis=1)
    f["pe_tech_over_defensive"] = tech / defens
    f["sector_pe_median_chg13w"] = f.sector_pe_median.pct_change(13)
    return f.reindex(idx, method="ffill").shift(1)


def earnings_features(idx):
    e = pd.read_parquet(f"{EXOG}/earnings_calendar.parquet")
    members = set(pd.read_parquet(f"{EXOG}/sp500_constituents_current.parquet").symbol)
    e = e[e.symbol.isin(members)].copy()
    e["date"] = pd.to_datetime(e.date)
    e = e.dropna(subset=["date"])
    rep = e.dropna(subset=["epsActual", "epsEstimated"]).sort_values("date")
    rep["beat"] = rep.epsActual > rep.epsEstimated
    rep["surprise"] = ((rep.epsActual - rep.epsEstimated)
                       / rep.epsEstimated.abs().clip(lower=0.05)).clip(-2, 2)
    # event-window aggregates: reports in (t-92d, t-2d], min 30 reports
    rep_d = rep.set_index("date").sort_index()
    f = pd.DataFrame(index=idx)
    beats, surprises = [], []
    for t in idx:
        w = rep_d.loc[t - pd.Timedelta(days=92): t - pd.Timedelta(days=2)]
        if len(w) >= 30:
            beats.append(w.beat.mean())
            surprises.append(w.surprise.median())
        else:
            beats.append(np.nan)
            surprises.append(np.nan)
    f["earn_beat_rate_90d"] = beats
    f["earn_surprise_med_90d"] = surprises
    # upcoming report density: counts by scheduled date (known in advance)
    sched = e.groupby(e.date.dt.normalize()).size()
    dens = sched.reindex(pd.date_range(idx.min(), idx.max() + pd.Timedelta(days=30)), fill_value=0)
    f["earn_upcoming_21d"] = pd.Series(
        [dens.loc[t: t + pd.Timedelta(days=21)].sum() for t in idx], index=idx)
    return f


def macro_features(idx):
    def load(name):
        d = pd.read_parquet(f"{EXOG}/fred_{name}.parquet")
        d["date"] = pd.to_datetime(d.date)
        return d.set_index("date").value
    cpi, un, sent, ind = load("cpi"), load("unrate"), load("umich_sentiment"), load("indpro")
    f = pd.DataFrame(index=cpi.index.union(un.index).union(sent.index))
    f["cpi_yoy"] = cpi.pct_change(12).reindex(f.index)
    f["unrate"] = un.reindex(f.index)
    f["unrate_chg3m"] = un.diff(3).reindex(f.index)
    f["sentiment"] = sent.reindex(f.index)
    f["sentiment_chg6m"] = sent.diff(6).reindex(f.index)
    f["indpro_yoy"] = ind.pct_change(12).reindex(f.index)
    f = f.sort_index().ffill()
    f.index = f.index + pd.Timedelta(days=45)  # release-lag clearance
    return f.reindex(idx, method="ffill")


def existing_features(idx):
    f = pd.read_parquet("data/features/daily_features.parquet")
    f["date"] = pd.to_datetime(f.date)
    return f.set_index("date").reindex(idx)


def build() -> pd.DataFrame:
    idx = daily_index()
    parts = [treasury_features(idx), cot_features(idx), sector_pe_features(idx),
             earnings_features(idx), macro_features(idx), existing_features(idx)]
    out = pd.concat(parts, axis=1)
    out.index.name = "date"
    return out


if __name__ == "__main__":
    df = build()
    df.to_parquet("data/features/cycle2_features.parquet")
    print(df.shape, "features:", len(df.columns))
    print("non-null coverage 2018+:",
          df.loc["2018-08-01":].notna().mean().sort_values().head(8).round(2).to_dict())
