"""Minimal domain enum needed by option selection/pricing."""

from __future__ import annotations

import enum


class OptionType(enum.StrEnum):
    CALL = "C"
    PUT = "P"
