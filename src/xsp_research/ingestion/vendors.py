"""Vendor-format adapters: normalize provider files into the canonical schema.

Adapters are declarative column mappings, so a new vendor is a new
``VendorMapping`` (or a YAML-supplied one), not new code. Ingestion always ends
with schema validation and a manifest (source hash, coverage, transforms) so a
backtest can state exactly which data produced it.

The SPX->XSP proxy transform is available ONLY as an explicit opt-in: it scales
strikes/prices by 1/10 and renames the root to ``<ROOT>_PROXY`` so proxy data
can never silently masquerade as native XSP quotes. Execution costs measured on
proxy data are NOT representative of XSP; see docs/PLAN.md assumption 1.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel

from xsp_research.ingestion.base import CHAIN_SCHEMA, validate_chain_frame

_REQUIRED = ("ts", "root", "expiration", "strike", "option_type", "bid", "ask", "underlying_price")
_OPTIONAL_INTS = ("bid_size", "ask_size", "volume", "open_interest")


class VendorMapping(BaseModel):
    """Declarative mapping from one vendor file layout to the canonical schema."""

    name: str
    columns: dict[str, str]  # canonical column -> vendor column
    ts_timezone: str = "America/New_York"  # timezone of naive vendor timestamps
    ts_format: str | None = None  # strptime format; None = polars auto-parse
    expiration_format: str | None = None
    strike_scale: float = 1.0  # e.g. 0.001 when strikes arrive in millis
    price_scale: float = 1.0  # applied to bid/ask/underlying


PRESETS: dict[str, VendorMapping] = {
    # Files already in canonical column names (e.g. exported by this tool).
    "generic": VendorMapping(name="generic", columns={c: c for c in CHAIN_SCHEMA}),
    # Cboe DataShop option-quote CSVs (verify against your delivered layout;
    # any mismatch is a mapping edit, not a code change).
    "cboe_datashop": VendorMapping(
        name="cboe_datashop",
        columns={
            "ts": "quote_datetime",
            "root": "root",
            "expiration": "expiration",
            "strike": "strike",
            "option_type": "option_type",
            "bid": "bid",
            "ask": "ask",
            "bid_size": "bid_size",
            "ask_size": "ask_size",
            "volume": "trade_volume",
            "open_interest": "open_interest",
            "underlying_price": "active_underlying_price",
        },
    ),
}


class VendorFormatError(ValueError):
    """Raised when a vendor file cannot be normalized under the given mapping."""


def _ts_expr(src: pl.Expr, dtype: pl.DataType, mapping: VendorMapping) -> pl.Expr:
    if dtype == pl.Utf8:
        parsed = src.str.to_datetime(format=mapping.ts_format, time_unit="us")
    elif isinstance(dtype, pl.Datetime):
        parsed = src.dt.cast_time_unit("us")
        if dtype.time_zone is not None:
            return parsed.dt.convert_time_zone("UTC")
    else:
        raise VendorFormatError(f"unsupported timestamp dtype {dtype}")
    return parsed.dt.replace_time_zone(mapping.ts_timezone).dt.convert_time_zone("UTC")


def _expiration_expr(src: pl.Expr, dtype: pl.DataType, mapping: VendorMapping) -> pl.Expr:
    if dtype == pl.Utf8:
        return src.str.to_date(format=mapping.expiration_format)
    if dtype == pl.Date:
        return src
    if isinstance(dtype, pl.Datetime):
        return src.dt.date()
    raise VendorFormatError(f"unsupported expiration dtype {dtype}")


def normalize_vendor_frame(df: pl.DataFrame, mapping: VendorMapping) -> pl.DataFrame:
    """Map a vendor frame into the validated canonical chain schema."""
    missing = [
        c for c in _REQUIRED if c not in mapping.columns or mapping.columns[c] not in df.columns
    ]
    if missing:
        raise VendorFormatError(
            f"mapping '{mapping.name}' cannot supply required columns {missing}; "
            f"vendor file has {df.columns}"
        )

    exprs: list[pl.Expr] = [
        _ts_expr(pl.col(mapping.columns["ts"]), df.schema[mapping.columns["ts"]], mapping).alias(
            "ts"
        ),
        pl.col(mapping.columns["root"]).cast(pl.Utf8).alias("root"),
        _expiration_expr(
            pl.col(mapping.columns["expiration"]),
            df.schema[mapping.columns["expiration"]],
            mapping,
        ).alias("expiration"),
        (pl.col(mapping.columns["strike"]).cast(pl.Float64) * mapping.strike_scale).alias("strike"),
        pl.col(mapping.columns["option_type"])
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.slice(0, 1)
        .alias("option_type"),
        (pl.col(mapping.columns["bid"]).cast(pl.Float64) * mapping.price_scale).alias("bid"),
        (pl.col(mapping.columns["ask"]).cast(pl.Float64) * mapping.price_scale).alias("ask"),
        (pl.col(mapping.columns["underlying_price"]).cast(pl.Float64) * mapping.price_scale).alias(
            "underlying_price"
        ),
    ]
    for c in _OPTIONAL_INTS:
        src = mapping.columns.get(c)
        if src is not None and src in df.columns:
            exprs.append(pl.col(src).cast(pl.Int64).alias(c))
        else:
            exprs.append(pl.lit(None, dtype=pl.Int64).alias(c))

    out = (
        df.select(exprs)
        .select(list(CHAIN_SCHEMA))
        .sort(["ts", "expiration", "strike", "option_type"])
    )
    validate_chain_frame(out)
    return out


def read_vendor_file(path: str | Path, mapping: VendorMapping) -> pl.DataFrame:
    p = Path(path)
    raw = pl.read_csv(p, infer_schema_length=10_000) if p.suffix == ".csv" else pl.read_parquet(p)
    return normalize_vendor_frame(raw, mapping)


def to_xsp_proxy(df: pl.DataFrame, divisor: float = 10.0) -> pl.DataFrame:
    """Scale an SPX-class chain to XSP levels (1/10) and relabel the root.

    The result is labeled ``<ROOT>_PROXY`` and must always be reported as a
    structural proxy: strike granularity is 10x coarser and quoted spreads do
    NOT represent achievable XSP execution costs.
    """
    out = df.with_columns(
        (pl.col("strike") / divisor).alias("strike"),
        (pl.col("bid") / divisor).alias("bid"),
        (pl.col("ask") / divisor).alias("ask"),
        (pl.col("underlying_price") / divisor).alias("underlying_price"),
        (pl.col("root") + "_PROXY").alias("root"),
    )
    validate_chain_frame(out)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(
    source: Path, output: Path, mapping: VendorMapping, df: pl.DataFrame, transforms: list[str]
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "vendor_mapping": mapping.name,
        "source_file": str(source),
        "source_sha256": _sha256(source),
        "output_file": str(output),
        "transforms": transforms,
        "rows": df.height,
        "ts_min": str(df["ts"].min()),
        "ts_max": str(df["ts"].max()),
        "roots": sorted(df["root"].unique().to_list()),
        "n_expirations": df["expiration"].n_unique(),
        "n_strikes": df["strike"].n_unique(),
        "license_note": "data licensed by vendor; do not redistribute or commit",
    }


def ingest_file(
    input_path: str | Path,
    output_path: str | Path,
    mapping: VendorMapping,
    *,
    spx_proxy: bool = False,
    manifest_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize one vendor file to canonical Parquet + manifest. Returns the manifest."""
    src, out = Path(input_path), Path(output_path)
    df = read_vendor_file(src, mapping)
    transforms = []
    if spx_proxy:
        df = to_xsp_proxy(df)
        transforms.append("spx_to_xsp_proxy(/10)")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    manifest = build_manifest(src, out, mapping, df, transforms)
    mdir = Path(manifest_dir) if manifest_dir else out.parent
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{out.stem}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
