"""Phase B3: engine-level A/B of strike-selection rules on the frozen gate.

Rules, all 0.10-0.20 delta band, $8 wide, 30 DTE, weekly, hold, cap-2:
  fixed015   — standard selection, short_delta_target=0.15 (control)
  min_resid  — short strike = lowest SVI-residual (cheapest vs fitted smile)
  max_resid  — short strike = highest residual (the original mispricing story;
               expected to LOSE per the diagnostic — included as a check)

Selection override is a monkeypatch of engine.select_bear_call_spread that
replicates the stock pipeline (same expiration choice, same quote filters,
same long-leg logic) but ranks the short leg by smile residual instead of
delta distance. SVI slice fits are cached per (date, expiry).
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime

import numpy as np
import polars as pl

import xsp_research.backtest.engine as engine_mod
from xsp_research.backtest.american import DividendCalendar
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.config import EXECUTION_SCENARIOS, SelectionConfig, load_strategy_config
from xsp_research.domain import ExerciseStyle, OptionType, SettlementStyle, SpreadQuote
from xsp_research.evaluation.robustness import bootstrap_mean_ci
from xsp_research.ingestion.file_provider import ParquetUnderlyingProvider, SeriesRatesProvider
from xsp_research.options.black_scholes import bs_greeks, year_fraction
from xsp_research.options.implied_vol import implied_vol
from xsp_research.options.svi_surface import fit_svi_slice
from xsp_research.strategy.candidate_selection import (
    AuditRecord,
    SelectionResult,
    _CallCandidate,
    _passes_quote_filters,
    _pick_long_leg,
    _row_to_quote,
    select_bear_call_spread,
    select_expiration,
)

import sys
sys.path.insert(0, "scripts")
from full_ablation_sweep import (  # noqa: E402
    BASE_CONFIG, DIVIDENDS, OPTIONS_GLOB, RATES, UNDERLYING, PreloadedOptionsProvider,
)
from phaseA_aggression_frontier import (  # noqa: E402
    episode_boot_ci, episode_ids, make_final_gate,
)

EXP_FEATURES = (
    "reports/experiments/spy_regime_rsi70_ivrich-20260723-204901-b6454793/features.parquet"
)
REGIME_V2 = "results/ablation_full/regime_features_v2.parquet"
OUT = "results/ablation_full/phaseB_engine_ab.json"
BAND = (0.10, 0.20)

_fit_cache: dict[tuple[date, date], object] = {}
_original_select = select_bear_call_spread


def _resid_select(rank_sign: float):
    """Build a select_bear_call_spread replacement ranking the short leg by
    rank_sign * residual (min_resid: -1 ... pick most-negative residual)."""

    def select(
        chain: pl.DataFrame,
        as_of: datetime,
        spot: float,
        r: float,
        q_yield: float,
        cfg: SelectionConfig,
    ) -> SelectionResult:
        result = SelectionResult(spread_quote=None)
        expirations = chain.filter(pl.col("root") == cfg.root)["expiration"].unique().to_list()
        expiry, exp_audits = select_expiration(expirations, as_of.date(), cfg)
        result.audits.extend(exp_audits)
        if expiry is None:
            result.audits.append(AuditRecord("spread", "-", False, "no expiration in window"))
            return result
        result.expiration = expiry
        result.dte = (expiry - as_of.date()).days
        t = year_fraction(as_of.date(), expiry)

        calls = chain.filter(
            (pl.col("root") == cfg.root)
            & (pl.col("expiration") == expiry)
            & (pl.col("option_type") == "C")
        ).sort("strike")
        candidates: list[_CallCandidate] = []
        for row in calls.iter_rows(named=True):
            quote = _row_to_quote(row, {
                "exercise_style": ExerciseStyle(cfg.exercise_style),
                "settlement": SettlementStyle(cfg.settlement),
            })
            iv = implied_vol(quote.mid, spot, quote.contract.strike, t, r, q_yield,
                             OptionType.CALL)
            if iv is None:
                continue
            delta = bs_greeks(spot, quote.contract.strike, t, iv, r, q_yield,
                              OptionType.CALL).delta
            candidates.append(_CallCandidate(quote=quote, iv=iv, delta=delta))
        if not candidates:
            result.audits.append(AuditRecord("spread", "-", False, "no recoverable IVs"))
            return result

        key = (as_of.date(), expiry)
        fit = _fit_cache.get(key)
        if key not in _fit_cache:
            # fit only on quality-passing quotes (same filter set as the
            # diagnostic script) so residuals aren't shaped by junk markets
            fit_pool = [
                c for c in candidates
                if c.quote.has_valid_market and c.quote.rel_spread <= 0.50
            ] or candidates
            ks = np.array([c.quote.contract.strike for c in fit_pool])
            ivs = np.array([c.iv for c in fit_pool])
            fwd = spot * math.exp((r - q_yield) * t)
            fit = fit_svi_slice(ks, ivs, fwd, t, expiry)
            _fit_cache[key] = fit
        if fit is None:
            # fall back to the stock selector so the run doesn't silently thin out
            return _original_select(chain, as_of, spot, r, q_yield, cfg)

        short_pool: list[tuple[float, _CallCandidate]] = []
        for c in candidates:
            ok, reason = _passes_quote_filters(c.quote, cfg, is_short_leg=True)
            label = f"K={c.quote.contract.strike:g} d={c.delta:.3f}"
            if not ok:
                result.audits.append(AuditRecord("short_leg", label, False, reason))
                continue
            if c.quote.contract.strike <= spot:
                result.audits.append(AuditRecord("short_leg", label, False, "not OTM"))
                continue
            if not (BAND[0] <= c.delta <= BAND[1]):
                result.audits.append(AuditRecord("short_leg", label, False, "outside band"))
                continue
            resid = c.iv - float(fit.iv(c.quote.contract.strike))
            short_pool.append((rank_sign * resid, c))
        if not short_pool:
            return _original_select(chain, as_of, spot, r, q_yield, cfg)
        _, short = max(short_pool, key=lambda rc: (rc[0], -rc[1].quote.contract.strike))
        result.audits.append(AuditRecord(
            "short_leg", f"K={short.quote.contract.strike:g} d={short.delta:.3f}",
            True, "residual-ranked",
        ))

        long_pool = [
            c for c in candidates
            if c.quote.contract.strike > short.quote.contract.strike
            and _passes_quote_filters(c.quote, cfg, is_short_leg=False)[0]
        ]
        if not long_pool:
            result.audits.append(AuditRecord("spread", "-", False, "no long strikes"))
            return result
        long_c = _pick_long_leg(short, long_pool, spot, t, cfg, result)
        if long_c is None:
            return result
        sq = SpreadQuote(short_quote=short.quote, long_quote=long_c.quote)
        result.spread_quote = sq
        result.short_iv, result.short_delta = short.iv, short.delta
        result.long_iv, result.long_delta = long_c.iv, long_c.delta
        result.audits.append(AuditRecord(
            "spread", f"{short.quote.contract.strike:g}/{long_c.quote.contract.strike:g}",
            True, f"width={sq.spread.width:g} residual-rule",
        ))
        return result

    return select


def main() -> int:
    feats = pl.read_parquet(EXP_FEATURES).join(
        pl.read_parquet(REGIME_V2), on="date", how="left"
    )
    gate = make_final_gate(feats)
    base = load_strategy_config(BASE_CONFIG)
    options = PreloadedOptionsProvider(OPTIONS_GLOB, "SPY", base.backtest.mark_time_et)
    und = ParquetUnderlyingProvider(UNDERLYING, "SPY")
    rates = SeriesRatesProvider(RATES)
    div = DividendCalendar.from_parquet(DIVIDENDS)

    rules = {
        "fixed015": None,
        "min_resid": _resid_select(-1.0),
        "max_resid": _resid_select(+1.0),
    }
    out: dict[str, dict] = {}
    frames: dict[str, pl.DataFrame] = {}
    for scen in ("base", "conservative"):
        for rule, selector in rules.items():
            engine_mod.select_bear_call_spread = selector or _original_select
            cfg = base.model_copy(update={
                "name": f"phaseB_{rule}_{scen}",
                "execution": EXECUTION_SCENARIOS[scen],
                "exits": [],
                "selection": base.selection.model_copy(update={
                    "short_delta_target": 0.15, "fixed_width": 8.0,
                    "target_dte": 30, "min_dte": 25, "max_dte": 35,
                }),
                "sizing": base.sizing.model_copy(update={
                    "contracts_per_entry": 1, "max_open_positions": 2,
                }),
            })
            res = BacktestEngine(
                cfg, options, und, rates, execution_scenario=scen,
                entry_gate=gate, dividends=div,
            ).run()
            t = res.trades_frame().with_columns(
                (pl.col("realized_net") / pl.col("qty")).alias("pnl"),
                pl.col("entry_ts").dt.convert_time_zone("America/New_York")
                .dt.date().alias("entry_date"),
            )
            frames[f"{rule}|{scen}"] = t
            per = t["pnl"].to_numpy()
            eids = episode_ids(t["entry_date"].to_numpy())
            elo, ehi, n_ep = episode_boot_ci(per, eids)
            ci = bootstrap_mean_ci(per, n_boot=2000, block=5, seed=7)
            out[f"{rule}|{scen}"] = {
                "n": t.height, "mean": float(per.mean()),
                "ci": [ci["ci_low"], ci["ci_high"]], "ep_ci": [elo, ehi],
                "win": float((per > 0).mean()), "worst": float(per.min()),
                "total": float(per.sum()),
                "mean_delta": float(t["short_delta_at_entry"].abs().mean()),
                "mean_credit": float((t["entry_credit"] / t["qty"]).mean()),
            }
            r = out[f"{rule}|{scen}"]
            print(f"{rule:9s}|{scen:12s} n={r['n']} mean=${r['mean']:.2f} "
                  f"CI=[{r['ci'][0]:.2f},{r['ci'][1]:.2f}] "
                  f"epCI=[{r['ep_ci'][0]:.2f},{r['ep_ci'][1]:.2f}] win={r['win']:.2f} "
                  f"worst={r['worst']:.0f} total={r['total']:.0f} "
                  f"avg_d={r['mean_delta']:.3f} credit={r['mean_credit']:.2f}", flush=True)
    engine_mod.select_bear_call_spread = _original_select

    # paired comparison on common entry dates
    for scen in ("base", "conservative"):
        a = frames[f"fixed015|{scen}"].select(["entry_date", "pnl"])
        for rule in ("min_resid", "max_resid"):
            b = frames[f"{rule}|{scen}"].select(["entry_date", "pnl"])
            j = a.join(b, on="entry_date", suffix="_r").sort("entry_date")
            diff = (j["pnl_r"] - j["pnl"]).to_numpy()
            eids = episode_ids(j["entry_date"].to_numpy())
            lo, hi, n_ep = episode_boot_ci(diff, eids)
            out[f"paired:{rule}|{scen}"] = {
                "n_common": j.height, "mean_diff": float(diff.mean()),
                "ep_ci": [lo, hi], "n_episodes": n_ep,
            }
            print(f"paired {rule} vs fixed015 [{scen}]: n={j.height} "
                  f"diff=${diff.mean():+.2f} epCI=[{lo:+.2f},{hi:+.2f}]", flush=True)

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
