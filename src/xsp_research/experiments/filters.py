"""Declarative entry filters over the daily feature frame.

A filter is a list of clauses ANDed together; each clause compares one feature
value to a threshold. No eval(), no arbitrary expressions — every experiment's
gating logic is fully serializable and auditable. Missing feature values block
the entry by default (conservative: an untestable condition is a failed
condition), configurable via ``allow_on_missing``.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from datetime import date
from typing import Literal

import polars as pl
from pydantic import BaseModel

_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


class FilterClause(BaseModel):
    feature: str
    op: Literal["<", "<=", ">", ">=", "==", "!="]
    value: float

    def describe(self) -> str:
        return f"{self.feature} {self.op} {self.value:g}"


class EntryFilterConfig(BaseModel):
    clauses: list[FilterClause]
    allow_on_missing: bool = False


def make_entry_gate(
    features: pl.DataFrame, cfg: EntryFilterConfig
) -> Callable[[date], tuple[bool, str]]:
    """Build an engine entry-gate closure from the feature frame.

    The frame is keyed by decision date and already lag-shifted by the builder,
    so looking up the session's row involves no future information.
    """
    unknown = [c.feature for c in cfg.clauses if c.feature not in features.columns]
    if unknown:
        raise ValueError(f"filter references unknown features: {unknown}")
    rows: dict[date, dict] = {r["date"]: r for r in features.iter_rows(named=True)}

    def gate(session: date) -> tuple[bool, str]:
        row = rows.get(session)
        if row is None:
            return cfg.allow_on_missing, f"no feature row for {session}"
        for clause in cfg.clauses:
            v = row.get(clause.feature)
            if v is None:
                if cfg.allow_on_missing:
                    continue
                return False, f"{clause.feature} unavailable (warm-up or missing source)"
            if not _OPS[clause.op](v, clause.value):
                return False, f"{clause.feature}={v:.4g} fails {clause.describe()}"
        return True, "passed " + " and ".join(c.describe() for c in cfg.clauses)

    return gate
