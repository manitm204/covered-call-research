"""Configuration models and YAML loading.

Every knob that materially affects results lives here, is validated by pydantic,
and is snapshotted into each backtest result for reproducibility.
"""

from __future__ import annotations

import enum
from datetime import date, time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class WidthMethod(enum.StrEnum):
    """How the long-leg strike is chosen relative to the short leg."""

    FIXED_WIDTH = "fixed_width"
    LONG_DELTA = "long_delta"
    PCT_UNDERLYING = "pct_underlying"
    EXPECTED_MOVE_FRACTION = "expected_move_fraction"
    MAX_LOSS_BUDGET = "max_loss_budget"


class FillModel(enum.StrEnum):
    NATURAL = "natural"  # credit at short bid - long ask (worst executable)
    MIDPOINT = "midpoint"  # credit at mid of both legs (optimistic)
    PCT_BETWEEN = "pct_between"  # interpolate natural -> mid by `fill_pct`


class InterestMode(enum.StrEnum):
    NONE = "none"
    FREE_CASH_ONLY = "free_cash_only"
    FREE_CASH_AND_COLLATERAL = "free_cash_and_collateral"


class EntryFrequency(enum.StrEnum):
    MONTHLY = "monthly"  # first eligible session of each calendar month
    WEEKLY = "weekly"  # first eligible session of each ISO week


class SelectionConfig(BaseModel):
    """Deterministic contract selection parameters (research brief section 6)."""

    root: str = "XSP"
    # Contract mechanics: default is XSP-style European/PM-cash-settled.
    # SPY requires american + physical, which activates the engine's
    # early-assignment hazard and expiry-day forced close (backtest/american.py).
    exercise_style: Literal["european", "american"] = "european"
    settlement: Literal["PM", "AM", "physical"] = "PM"  # AM parses but the engine rejects it
    target_dte: int = 30
    min_dte: int = 25
    max_dte: int = 35
    short_delta_target: float = Field(0.15, gt=0.0, lt=1.0)
    width_method: WidthMethod = WidthMethod.FIXED_WIDTH
    fixed_width: float = 5.0
    long_delta_target: float | None = None  # for WidthMethod.LONG_DELTA
    pct_underlying: float | None = None  # for WidthMethod.PCT_UNDERLYING
    expected_move_fraction: float | None = None  # for WidthMethod.EXPECTED_MOVE_FRACTION
    max_loss_budget: float | None = None  # dollars per spread, for MAX_LOSS_BUDGET
    # Quote-quality / liquidity filters
    min_short_bid: float = 0.05
    max_leg_rel_spread: float = 0.50  # (ask-bid)/mid per leg
    min_open_interest: int | None = None
    min_quote_size: int | None = None

    @model_validator(mode="after")
    def _width_params_present(self) -> SelectionConfig:
        required = {
            WidthMethod.LONG_DELTA: self.long_delta_target,
            WidthMethod.PCT_UNDERLYING: self.pct_underlying,
            WidthMethod.EXPECTED_MOVE_FRACTION: self.expected_move_fraction,
            WidthMethod.MAX_LOSS_BUDGET: self.max_loss_budget,
        }
        if self.width_method in required and required[self.width_method] is None:
            raise ValueError(f"width_method={self.width_method} requires its parameter to be set")
        return self


class ExitRuleConfig(BaseModel):
    """One exit rule. Rules are evaluated in list order (priority)."""

    kind: str  # profit_target | stop_loss | dte | delta_stop
    profit_target_pct: float | None = None  # e.g. 0.50 => close at 50% of max profit
    stop_loss_multiple: float | None = None  # close when cost-to-close >= mult * entry credit
    dte_threshold: int | None = None
    delta_threshold: float | None = None

    @model_validator(mode="after")
    def _param_present(self) -> ExitRuleConfig:
        needed = {
            "profit_target": self.profit_target_pct,
            "stop_loss": self.stop_loss_multiple,
            "dte": self.dte_threshold,
            "delta_stop": self.delta_threshold,
        }
        if self.kind not in needed:
            raise ValueError(f"unknown exit rule kind: {self.kind}")
        if needed[self.kind] is None:
            raise ValueError(f"exit rule {self.kind} requires its parameter")
        return self


class ExecutionConfig(BaseModel):
    fill_model: FillModel = FillModel.PCT_BETWEEN
    fill_pct: float = Field(0.5, ge=0.0, le=1.0)  # 0 = natural, 1 = midpoint
    slippage_per_spread: float = 0.0  # index points, worsens fill
    max_leg_rel_spread: float = 0.50  # refuse to trade through very wide quotes
    min_credit: float = 0.05  # refuse entries below this credit (index points)


class CostConfig(BaseModel):
    commission_per_contract: float = 0.65
    fees_per_contract: float = 0.60  # exchange (proprietary index product) + regulatory
    settlement_fee_per_contract: float = 0.0

    def per_spread(self) -> float:
        """Round-turn cost of one side (open or close) of a 2-leg spread, dollars."""
        return 2.0 * (self.commission_per_contract + self.fees_per_contract)


class InterestConfig(BaseModel):
    mode: InterestMode = InterestMode.FREE_CASH_ONLY
    # Either a fixed annual rate, or "provider" to use the RatesProvider series.
    rate_source: str = "provider"  # "provider" | "fixed"
    fixed_rate: float = 0.0
    spread_bps: float = 0.0  # added to (usually subtracted from) the benchmark, e.g. -25


class SizingConfig(BaseModel):
    contracts_per_entry: int = Field(1, ge=1)
    max_open_positions: int = Field(1, ge=1)


class BacktestConfig(BaseModel):
    start: date
    end: date
    initial_cash: float = 100_000.0
    entry_frequency: EntryFrequency = EntryFrequency.MONTHLY
    entry_time_et: time = time(15, 30)
    mark_time_et: time = time(15, 30)

    @model_validator(mode="after")
    def _range_ok(self) -> BacktestConfig:
        if self.end <= self.start:
            raise ValueError("backtest end must be after start")
        return self


class StrategyConfig(BaseModel):
    """Top-level configuration: one file = one reproducible run definition."""

    name: str
    selection: SelectionConfig = SelectionConfig()
    exits: list[ExitRuleConfig] = []
    execution: ExecutionConfig = ExecutionConfig()
    costs: CostConfig = CostConfig()
    interest: InterestConfig = InterestConfig()
    sizing: SizingConfig = SizingConfig()
    backtest: BacktestConfig

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe snapshot stored with every result."""
        return self.model_dump(mode="json")


# Execution scenarios (research brief section 7): reported side by side.
EXECUTION_SCENARIOS: dict[str, ExecutionConfig] = {
    "optimistic": ExecutionConfig(fill_model=FillModel.MIDPOINT, fill_pct=1.0),
    "base": ExecutionConfig(fill_model=FillModel.PCT_BETWEEN, fill_pct=0.5),
    "conservative": ExecutionConfig(
        fill_model=FillModel.NATURAL, fill_pct=0.0, slippage_per_spread=0.02
    ),
}


def load_strategy_config(path: str | Path) -> StrategyConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return StrategyConfig.model_validate(raw)
