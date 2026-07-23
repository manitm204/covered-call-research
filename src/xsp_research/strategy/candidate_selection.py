"""Deterministic, auditable bear-call-spread selection.

Every expiration and strike considered is recorded with an accept/reject
reason so any historical trade (or absence of one) can be explained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime

import polars as pl

from xsp_research.config import SelectionConfig, WidthMethod
from xsp_research.domain import (
    ExerciseStyle,
    OptionContract,
    OptionQuote,
    OptionType,
    SettlementStyle,
    SpreadQuote,
)
from xsp_research.options.black_scholes import bs_greeks, year_fraction
from xsp_research.options.implied_vol import implied_vol


@dataclass(frozen=True, slots=True)
class AuditRecord:
    stage: str  # "expiration" | "short_leg" | "long_leg" | "spread"
    candidate: str
    accepted: bool
    reason: str


@dataclass(slots=True)
class SelectionResult:
    spread_quote: SpreadQuote | None
    audits: list[AuditRecord] = field(default_factory=list)
    # Metadata for the ledger/features (model-derived at selection time)
    short_iv: float | None = None
    short_delta: float | None = None
    long_iv: float | None = None
    long_delta: float | None = None
    expiration: date | None = None
    dte: int | None = None

    @property
    def selected(self) -> bool:
        return self.spread_quote is not None

    def reject_summary(self) -> str:
        rejected = [a for a in self.audits if not a.accepted]
        return "; ".join(f"[{a.stage}] {a.candidate}: {a.reason}" for a in rejected[-5:])


def select_expiration(
    expirations: list[date], as_of: date, cfg: SelectionConfig
) -> tuple[date | None, list[AuditRecord]]:
    """Nearest expiration to target DTE within [min_dte, max_dte]. Ties -> earlier."""
    audits: list[AuditRecord] = []
    eligible: list[date] = []
    for exp in sorted(expirations):
        dte = (exp - as_of).days
        if dte < cfg.min_dte:
            audits.append(
                AuditRecord("expiration", str(exp), False, f"dte {dte} < min {cfg.min_dte}")
            )
        elif dte > cfg.max_dte:
            audits.append(
                AuditRecord("expiration", str(exp), False, f"dte {dte} > max {cfg.max_dte}")
            )
        else:
            eligible.append(exp)
    if not eligible:
        return None, audits
    chosen = min(eligible, key=lambda e: (abs((e - as_of).days - cfg.target_dte), e))
    for exp in eligible:
        ok = exp == chosen
        audits.append(
            AuditRecord(
                "expiration",
                str(exp),
                ok,
                "closest to target dte" if ok else "eligible but farther from target dte",
            )
        )
    return chosen, audits


@dataclass(frozen=True, slots=True)
class _CallCandidate:
    quote: OptionQuote
    iv: float
    delta: float


def _row_to_quote(row: dict, root_meta: dict) -> OptionQuote:
    contract = OptionContract(
        root=row["root"],
        expiration=row["expiration"],
        strike=row["strike"],
        option_type=OptionType(row["option_type"]),
        **root_meta,
    )
    return OptionQuote(
        contract=contract,
        ts=row["ts"],
        bid=row["bid"],
        ask=row["ask"],
        bid_size=row["bid_size"],
        ask_size=row["ask_size"],
        volume=row["volume"],
        open_interest=row["open_interest"],
        underlying_price=row["underlying_price"],
    )


def _passes_quote_filters(
    q: OptionQuote, cfg: SelectionConfig, *, is_short_leg: bool
) -> tuple[bool, str]:
    if not q.has_valid_market:
        return False, "invalid/crossed/one-sided market"
    if is_short_leg and q.bid < cfg.min_short_bid:
        return False, f"short bid {q.bid:.2f} < min {cfg.min_short_bid:.2f}"
    if q.rel_spread > cfg.max_leg_rel_spread:
        return False, f"rel spread {q.rel_spread:.2f} > max {cfg.max_leg_rel_spread:.2f}"
    if cfg.min_open_interest is not None and (q.open_interest or 0) < cfg.min_open_interest:
        return False, f"open interest {q.open_interest} < min {cfg.min_open_interest}"
    if (
        cfg.min_quote_size is not None
        and min(q.bid_size or 0, q.ask_size or 0) < cfg.min_quote_size
    ):
        return False, f"quote size < min {cfg.min_quote_size}"
    return True, "ok"


def select_bear_call_spread(
    chain: pl.DataFrame,
    as_of: datetime,
    spot: float,
    r: float,
    q_yield: float,
    cfg: SelectionConfig,
) -> SelectionResult:
    """Full selection pipeline on one chain snapshot (no data beyond `chain`)."""
    result = SelectionResult(spread_quote=None)

    expirations = chain.filter(pl.col("root") == cfg.root)["expiration"].unique().to_list()
    expiry, exp_audits = select_expiration(expirations, as_of.date(), cfg)
    result.audits.extend(exp_audits)
    if expiry is None:
        result.audits.append(AuditRecord("spread", "-", False, "no expiration in DTE window"))
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
        quote = _row_to_quote(
            row,
            {
                "exercise_style": ExerciseStyle(cfg.exercise_style),
                "settlement": SettlementStyle(cfg.settlement),
            },
        )
        iv = implied_vol(quote.mid, spot, quote.contract.strike, t, r, q_yield, OptionType.CALL)
        if iv is None:
            continue  # unrecoverable IV: not audited per-strike to keep audits readable
        delta = bs_greeks(spot, quote.contract.strike, t, iv, r, q_yield, OptionType.CALL).delta
        candidates.append(_CallCandidate(quote=quote, iv=iv, delta=delta))
    if not candidates:
        result.audits.append(AuditRecord("spread", "-", False, "no calls with recoverable IV"))
        return result

    # ---- short leg: delta closest to target among quality-passing OTM calls
    short_pool: list[_CallCandidate] = []
    for c in candidates:
        ok, reason = _passes_quote_filters(c.quote, cfg, is_short_leg=True)
        label = f"K={c.quote.contract.strike:g} d={c.delta:.3f}"
        if not ok:
            result.audits.append(AuditRecord("short_leg", label, False, reason))
        elif c.quote.contract.strike <= spot:
            result.audits.append(AuditRecord("short_leg", label, False, "not OTM"))
        else:
            short_pool.append(c)
    if not short_pool:
        result.audits.append(AuditRecord("spread", "-", False, "no eligible short strikes"))
        return result
    short = min(
        short_pool,
        key=lambda c: (abs(c.delta - cfg.short_delta_target), c.quote.contract.strike),
    )
    result.audits.append(
        AuditRecord(
            "short_leg",
            f"K={short.quote.contract.strike:g} d={short.delta:.3f}",
            True,
            f"closest to target delta {cfg.short_delta_target:.2f}",
        )
    )

    # ---- long leg by configured width method
    long_pool = [
        c
        for c in candidates
        if c.quote.contract.strike > short.quote.contract.strike
        and _passes_quote_filters(c.quote, cfg, is_short_leg=False)[0]
    ]
    if not long_pool:
        result.audits.append(AuditRecord("spread", "-", False, "no eligible long strikes"))
        return result
    long_c = _pick_long_leg(short, long_pool, spot, t, cfg, result)
    if long_c is None:
        return result

    sq = SpreadQuote(short_quote=short.quote, long_quote=long_c.quote)
    result.spread_quote = sq
    result.short_iv, result.short_delta = short.iv, short.delta
    result.long_iv, result.long_delta = long_c.iv, long_c.delta
    result.audits.append(
        AuditRecord(
            "spread",
            f"{short.quote.contract.strike:g}/{long_c.quote.contract.strike:g}",
            True,
            f"width={sq.spread.width:g} method={cfg.width_method.value}",
        )
    )
    return result


def _pick_long_leg(
    short: _CallCandidate,
    pool: list[_CallCandidate],
    spot: float,
    t: float,
    cfg: SelectionConfig,
    result: SelectionResult,
) -> _CallCandidate | None:
    ks = short.quote.contract.strike

    def nearest_to_strike(target: float) -> _CallCandidate:
        return min(
            pool, key=lambda c: (abs(c.quote.contract.strike - target), c.quote.contract.strike)
        )

    if cfg.width_method is WidthMethod.FIXED_WIDTH:
        chosen = nearest_to_strike(ks + cfg.fixed_width)
    elif cfg.width_method is WidthMethod.PCT_UNDERLYING:
        chosen = nearest_to_strike(ks + cfg.pct_underlying * spot)
    elif cfg.width_method is WidthMethod.EXPECTED_MOVE_FRACTION:
        expected_move = spot * short.iv * math.sqrt(t)
        chosen = nearest_to_strike(ks + cfg.expected_move_fraction * expected_move)
    elif cfg.width_method is WidthMethod.LONG_DELTA:
        chosen = min(
            pool, key=lambda c: (abs(c.delta - cfg.long_delta_target), c.quote.contract.strike)
        )
    elif cfg.width_method is WidthMethod.MAX_LOSS_BUDGET:
        mult = short.quote.contract.multiplier
        affordable = []
        for c in pool:
            width = c.quote.contract.strike - ks
            est_credit = short.quote.mid - c.quote.mid  # mid estimate for budgeting only
            max_loss = (width - est_credit) * mult
            if max_loss <= cfg.max_loss_budget:
                affordable.append((width, c))
            else:
                result.audits.append(
                    AuditRecord(
                        "long_leg",
                        f"K={c.quote.contract.strike:g}",
                        False,
                        f"max loss ${max_loss:,.0f} > budget ${cfg.max_loss_budget:,.0f}",
                    )
                )
        if not affordable:
            result.audits.append(
                AuditRecord("spread", "-", False, "no width within max-loss budget")
            )
            return None
        chosen = max(affordable, key=lambda wc: wc[0])[1]  # widest width within budget
    else:  # pragma: no cover - enum is exhaustive
        raise ValueError(f"unsupported width method {cfg.width_method}")

    result.audits.append(
        AuditRecord(
            "long_leg",
            f"K={chosen.quote.contract.strike:g} d={chosen.delta:.3f}",
            True,
            f"selected by {cfg.width_method.value}",
        )
    )
    return chosen
