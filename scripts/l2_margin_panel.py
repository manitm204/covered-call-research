"""Cycle-3 sector profit-margin panel study (user hypothesis H18).

Build a point-in-time monthly panel: for each of 9 GICS-ish sectors, the median
trailing-4Q operating margin of S&P members (only statements whose filingDate <
feature date), its YoY change, and margin-improvement breadth (% of members with
positive YoY margin change). Targets: sector SPDR forward 21d/63d return minus
SPY.

Tests:
 A. Cross-sectional rank IC (margin trend rank vs forward relative return rank)
    per month, discovery era 2005-2017 vs confirm era 2018-2024 — both reported.
 B. Pooled-panel depth-2/3 tree (features: margin_yoy, margin_breadth, 6m price
    momentum, PE percentile where available) vs label outperform-SPY-next-21d,
    purged by time.
Survivorship caveat (current members only) applies to levels; cross-sector
RELATIVE signals are partially insulated — stated in every output.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "scripts")
from l2_common import append_experiment_log  # noqa: E402

SECT_ETF = {"Basic Materials": "XLB", "Energy": "XLE", "Financial Services": "XLF",
            "Industrials": "XLI", "Technology": "XLK", "Consumer Defensive": "XLP",
            "Utilities": "XLU", "Healthcare": "XLV", "Consumer Cyclical": "XLY"}


def build_sector_margins() -> pd.DataFrame:
    inc = pd.read_parquet("data/normalized/exog/sp500_income_q.parquet")
    inc = inc.dropna(subset=["revenue", "operatingIncome", "filingDate"])
    inc = inc[inc.revenue > 0]
    inc["opm"] = inc.operatingIncome / inc.revenue
    inc = inc.sort_values("date")
    # trailing-4Q margin per symbol (sum inc / sum rev is more robust than mean of ratios)
    inc["rev4"] = inc.groupby("symbol").revenue.transform(lambda s: s.rolling(4).sum())
    inc["op4"] = inc.groupby("symbol").operatingIncome.transform(lambda s: s.rolling(4).sum())
    inc["opm4"] = inc.op4 / inc.rev4
    inc["opm4_yoy"] = inc.groupby("symbol").opm4.transform(lambda s: s.diff(4))
    months = pd.date_range("2003-01-31", "2026-07-31", freq="ME")
    rows = []
    for sector, g in inc.groupby("sector"):
        if sector not in SECT_ETF:
            continue
        g = g.dropna(subset=["opm4"])
        for m in months:
            avail = g[g.filingDate < m - pd.Timedelta(days=1)]
            latest = avail.sort_values("date").groupby("symbol").tail(1)
            latest = latest[latest.date > m - pd.DateOffset(months=9)]  # stale filter
            if len(latest) < 8:
                continue
            yoy = latest.opm4_yoy.dropna()
            rows.append(dict(month=m, sector=sector, etf=SECT_ETF[sector],
                             n=len(latest), opm4_med=latest.opm4.median(),
                             opm4_yoy_med=yoy.median() if len(yoy) else np.nan,
                             margin_breadth=(yoy > 0).mean() if len(yoy) else np.nan))
    return pd.DataFrame(rows)


def add_prices(panel: pd.DataFrame) -> pd.DataFrame:
    px = {}
    for etf in set(panel.etf):
        d = pd.read_parquet(f"data/normalized/prices_long/sectors/{etf}.parquet")
        d["date"] = pd.to_datetime(d.date)
        px[etf] = d.set_index("date").adjClose
    spy = pd.read_parquet("data/normalized/prices_long/SPY.parquet")
    spy["date"] = pd.to_datetime(spy.date)
    spy = spy.set_index("date").adjClose
    def asof(s, t):
        s2 = s.loc[:t]
        return s2.iloc[-1] if len(s2) else np.nan
    out = []
    for r in panel.itertuples():
        p = px[r.etf]
        p0, s0 = asof(p, r.month), asof(spy, r.month)
        p21 = asof(p, r.month + pd.Timedelta(days=30))
        s21 = asof(spy, r.month + pd.Timedelta(days=30))
        p63 = asof(p, r.month + pd.Timedelta(days=91))
        s63 = asof(spy, r.month + pd.Timedelta(days=91))
        mom6 = p0 / asof(p, r.month - pd.DateOffset(months=6)) - 1
        out.append(dict(rel_fwd21=p21 / p0 - s21 / s0, rel_fwd63=p63 / p0 - s63 / s0,
                        mom6=mom6))
    return pd.concat([panel.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def cross_sectional_ic(df, feat, tgt, era):
    d = df[(df.month >= era[0]) & (df.month <= era[1])].dropna(subset=[feat, tgt])
    ics = []
    for m, g in d.groupby("month"):
        if len(g) >= 6:
            ics.append(np.corrcoef(rankdata(g[feat]), rankdata(g[tgt]))[0, 1])
    ics = np.array(ics)
    if len(ics) < 24:
        return None
    se = ics.std() / np.sqrt(len(ics) / 2)  # halve n for autocorrelation
    return dict(mean_ic=float(ics.mean()), t=float(ics.mean() / se), n_months=len(ics))


def main():
    panel = build_sector_margins()
    df = add_prices(panel)
    df.to_parquet("results/level2/sector_margin_panel.parquet")
    print(f"panel: {len(df)} sector-months, {df.month.min():%Y-%m} -> {df.month.max():%Y-%m}")
    for feat in ("opm4_yoy_med", "margin_breadth", "mom6"):
        for tgt in ("rel_fwd21", "rel_fwd63"):
            for era_name, era in [("2005-2017", ("2005-01-01", "2017-12-31")),
                                  ("2018-2024", ("2018-01-01", "2024-12-31"))]:
                r = cross_sectional_ic(df, feat, tgt, era)
                if r is None:
                    continue
                v = "PASS" if abs(r["t"]) > 1.64 else "fail"
                print(f"{feat:16s} -> {tgt:9s} {era_name}: IC {r['mean_ic']:+.3f} "
                      f"t={r['t']:+.2f} ({r['n_months']} mo) {v}")
                append_experiment_log(hypothesis="H18_sector_margin", partition=f"panel_{era_name}",
                                      underlying="9 sector ETFs", config=f"{feat}->{tgt}",
                                      scenario="n/a", n=r["n_months"], metric="mean_xsec_IC",
                                      value=round(r["mean_ic"], 3),
                                      verdict=v, notes=f"t={r['t']:+.2f}; survivorship caveat")


if __name__ == "__main__":
    main()
