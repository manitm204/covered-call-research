"""Phase B diagnostic: is smile-relative richness a usable strike-selection
signal on the gated entry dates? (exploratory, NOT pre-registered)

For every gated entry date (RSI>70 + >=2 of 3 confirmations, same rule as
Phase A), at the actually-traded ~30 DTE expiration:
  1. fit a raw-SVI slice to the call smile (quote-mid IVs),
  2. per candidate short strike (0.05-0.30 delta, OTM, quality-passing):
     richness residual = market IV - fitted IV, and dollar edge =
     (market mid credit - SVI model credit) for the $8-wide spread,
  3. price the hypothetical $8-wide spread with base fills and settle it at
     expiry intrinsic (underlying EOD close) -- hold-to-expiry approximation
     that ignores early assignment (7/417 historically) and expiry fees.

Analyses:
  A. sanity: approximate P&L at the 0.15-delta strike vs the engine's panel P&L
  B. strike-level: within-date partial correlation of P&L with residual/edge,
     controlling for delta
  C. rule A/B: fixed 0.15-delta vs max-residual vs max-edge strike in the
     0.10-0.20 band, paired per date, episode-bootstrap CI on the difference

Output: results/ablation_full/phaseB_candidates.parquet + printed report.
"""

from __future__ import annotations

import math
import sys
from datetime import date

import numpy as np
import polars as pl

from xsp_research.config import load_strategy_config
from xsp_research.domain import OptionType
from xsp_research.ingestion.file_provider import ParquetUnderlyingProvider, SeriesRatesProvider
from xsp_research.options.black_scholes import bs_greeks, bs_price, year_fraction
from xsp_research.options.implied_vol import implied_vol
from xsp_research.options.svi_surface import fit_svi_slice

sys.path.insert(0, "scripts")
from full_ablation_sweep import (  # noqa: E402
    BASE_CONFIG, OPTIONS_GLOB, RATES, UNDERLYING, PreloadedOptionsProvider,
)
from phaseA_aggression_frontier import episode_boot_ci, episode_ids  # noqa: E402

PANEL = "results/ablation_full/signal_study_panel.parquet"
OUT = "results/ablation_full/phaseB_candidates.parquet"
WIDTH = 8.0
FEES_OPEN = 2.5  # 2 legs x (0.65 commission + 0.60 fees)
DELTA_BAND = (0.10, 0.20)  # band for the rule A/B
CAND_BAND = (0.05, 0.30)


def main() -> int:
    panel = pl.read_parquet(PANEL).with_columns(
        (
            (pl.col("rsi_14") > 70)
            & (
                (pl.col("iv_minus_rv20") > 0).cast(pl.Int8)
                + (pl.col("sector_avg_corr_20") < 0.45).cast(pl.Int8)
                + (pl.col("absorption_chg_20d") < 0).cast(pl.Int8)
                >= 2
            )
        ).alias("gated")
    )
    gated = panel.filter(pl.col("gated"))
    print(f"gated entries: {gated.height} of {panel.height}", flush=True)

    base = load_strategy_config(BASE_CONFIG)
    options = PreloadedOptionsProvider(OPTIONS_GLOB, "SPY", base.backtest.mark_time_et)
    und = ParquetUnderlyingProvider(UNDERLYING, "SPY")
    rates = SeriesRatesProvider(RATES)

    rows: list[dict] = []
    skipped: list[str] = []
    for tr in gated.iter_rows(named=True):
        d: date = tr["entry_date"]
        expiry: date = tr["expiration"]
        chain = options.chain(tr["entry_ts"])
        s_t = und.close(expiry)
        if chain.is_empty() or s_t is None:
            skipped.append(f"{d}: no chain or no settlement close")
            continue
        calls = chain.filter(
            (pl.col("expiration") == expiry) & (pl.col("option_type") == "C")
        ).sort("strike")
        if calls.is_empty():
            skipped.append(f"{d}: no calls at {expiry}")
            continue
        spot = float(calls["underlying_price"].median())
        r = rates.rate(d)
        t = year_fraction(d, expiry)

        # quote-quality pass + IV/delta per strike (same filters as selection)
        quotes: dict[float, dict] = {}
        for row in calls.iter_rows(named=True):
            bid, ask = row["bid"], row["ask"]
            if bid is None or ask is None or bid <= 0 or ask < bid:
                continue
            mid = 0.5 * (bid + ask)
            if mid <= 0 or (ask - bid) / mid > 0.50:
                continue
            iv = implied_vol(mid, spot, row["strike"], t, r, 0.0, OptionType.CALL)
            if iv is None:
                continue
            delta = bs_greeks(spot, row["strike"], t, iv, r, 0.0, OptionType.CALL).delta
            quotes[row["strike"]] = {
                "bid": bid, "ask": ask, "mid": mid, "iv": iv, "delta": delta,
            }
        if len(quotes) < 8:
            skipped.append(f"{d}: only {len(quotes)} usable call quotes")
            continue

        ks = np.array(sorted(quotes))
        ivs = np.array([quotes[k]["iv"] for k in ks])
        fwd = spot * math.exp(r * t)
        fit = fit_svi_slice(ks, ivs, fwd, t, expiry)
        if fit is None:
            skipped.append(f"{d}: SVI fit failed")
            continue
        fit_ivs = {k: float(fit.iv(k)) for k in ks}

        strikes_sorted = list(ks)
        for k in strikes_sorted:
            qs = quotes[k]
            if not (CAND_BAND[0] <= qs["delta"] <= CAND_BAND[1]):
                continue
            if k <= spot or qs["bid"] < 0.05:
                continue
            # long leg: nearest available quality strike to K + WIDTH
            longs = [kk for kk in strikes_sorted if kk > k]
            if not longs:
                continue
            kl = min(longs, key=lambda kk: abs(kk - (k + WIDTH)))
            if abs(kl - (k + WIDTH)) > 1.0:  # demand the width we asked for
                continue
            ql = quotes[kl]
            natural = qs["bid"] - ql["ask"]
            mid_credit = qs["mid"] - ql["mid"]
            fill = natural + 0.5 * (mid_credit - natural)  # base scenario
            if fill < 0.05:
                continue
            width = kl - k
            intrinsic = max(s_t - k, 0.0) - max(s_t - kl, 0.0)
            pnl = fill * 100.0 - intrinsic * 100.0 - FEES_OPEN
            model_credit = bs_price(spot, k, t, fit_ivs[k], r, 0.0, OptionType.CALL) - bs_price(
                spot, kl, t, fit_ivs[kl], r, 0.0, OptionType.CALL
            )
            rows.append({
                "entry_date": d, "expiration": expiry, "dte": (expiry - d).days,
                "spot": spot, "settle": s_t,
                "short_strike": k, "long_strike": kl, "width": width,
                "delta": qs["delta"], "iv": qs["iv"],
                "resid": qs["iv"] - fit_ivs[k],
                "resid_long": ql["iv"] - fit_ivs[kl],
                "edge_dollars": (mid_credit - model_credit) * 100.0,
                "credit_fill": fill, "pnl": pnl,
                "panel_strike": tr["short_strike"], "panel_pnl": tr["pnl"],
            })
    print(f"dates skipped: {len(skipped)}", flush=True)
    for s in skipped[:10]:
        print("  ", s, flush=True)

    cand = pl.DataFrame(rows)
    cand.write_parquet(OUT)
    n_dates = cand["entry_date"].n_unique()
    print(f"\ncandidate rows: {cand.height} across {n_dates} dates", flush=True)

    # ---- A. sanity: approximation vs engine panel P&L at the traded strike
    sanity = (
        cand.filter((pl.col("short_strike") - pl.col("panel_strike")).abs() < 0.01)
        .select(["entry_date", "pnl", "panel_pnl"])
    )
    if sanity.height:
        a = sanity["pnl"].to_numpy()
        b = sanity["panel_pnl"].to_numpy()
        print(
            f"\n[A] sanity on {len(a)} matched trades: approx mean ${a.mean():.2f} "
            f"vs engine ${b.mean():.2f}; corr {np.corrcoef(a, b)[0, 1]:.3f}; "
            f"mean abs diff ${np.abs(a - b).mean():.2f}"
        )

    # ---- B. within-date partial correlation, controlling for delta
    df = cand.to_pandas()
    for col in ("pnl", "resid", "edge_dollars", "delta"):
        df[f"_{col}_dm"] = df[col] - df.groupby("entry_date")[col].transform("mean")

    def partial_corr(y, x, z):
        beta_y = np.polyfit(z, y, 2)
        beta_x = np.polyfit(z, x, 2)
        ry = y - np.polyval(beta_y, z)
        rx = x - np.polyval(beta_x, z)
        return float(np.corrcoef(ry, rx)[0, 1])

    print("\n[B] strike-level signal content (within-date demeaned):")
    for sig in ("resid", "edge_dollars"):
        raw = float(np.corrcoef(df["_pnl_dm"], df[f"_{sig}_dm"])[0, 1])
        pc = partial_corr(
            df["_pnl_dm"].to_numpy(), df[f"_{sig}_dm"].to_numpy(), df["_delta_dm"].to_numpy()
        )
        print(f"  {sig:12s}: corr with pnl {raw:+.3f}; partial (| delta) {pc:+.3f}")

    # ---- C. rule A/B in the 0.10-0.20 band, paired per date
    band = cand.filter(
        (pl.col("delta") >= DELTA_BAND[0]) & (pl.col("delta") <= DELTA_BAND[1])
    )
    picks: list[dict] = []
    for (d,), g in band.group_by(["entry_date"], maintain_order=True):
        gg = g.to_dicts()
        baseline = min(gg, key=lambda x: abs(x["delta"] - 0.15))
        max_resid = max(gg, key=lambda x: x["resid"])
        max_edge = max(gg, key=lambda x: x["edge_dollars"])
        picks.append({
            "entry_date": d, "n_cands": len(gg),
            "pnl_base": baseline["pnl"], "pnl_resid": max_resid["pnl"],
            "pnl_edge": max_edge["pnl"],
            "delta_base": baseline["delta"], "delta_resid": max_resid["delta"],
            "delta_edge": max_edge["delta"],
            "same_resid": baseline["short_strike"] == max_resid["short_strike"],
            "same_edge": baseline["short_strike"] == max_edge["short_strike"],
        })
    pk = pl.DataFrame(picks).sort("entry_date")
    dates = pk["entry_date"].to_numpy()
    eids = episode_ids(dates)
    print(f"\n[C] rule A/B on {pk.height} dates ({pk['n_cands'].mean():.1f} candidates/date, "
          f"{len(np.unique(eids))} episodes):")
    base_pnl = pk["pnl_base"].to_numpy()
    for rule in ("resid", "edge"):
        p = pk[f"pnl_{rule}"].to_numpy()
        diff = p - base_pnl
        lo, hi, _ = episode_boot_ci(diff, eids)
        same = pk[f"same_{rule}"].mean()
        print(
            f"  max-{rule:5s}: mean ${p.mean():7.2f} vs baseline ${base_pnl.mean():7.2f} "
            f"| paired diff ${diff.mean():+6.2f} epCI [{lo:+.2f},{hi:+.2f}] "
            f"| same strike as baseline {same:.0%} "
            f"| avg delta {pk[f'delta_{rule}'].mean():.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
