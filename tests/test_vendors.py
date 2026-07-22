"""Vendor ingestion: normalization, timezone handling, proxy transform, manifests."""

import json
from datetime import date, datetime

import polars as pl
import pytest

from xsp_research.ingestion.base import CHAIN_SCHEMA, validate_chain_frame
from xsp_research.ingestion.vendors import (
    PRESETS,
    VendorFormatError,
    VendorMapping,
    ingest_file,
    normalize_vendor_frame,
    to_xsp_proxy,
)


def cboe_style_frame() -> pl.DataFrame:
    """Two quotes in Cboe DataShop-style columns, naive ET timestamps."""
    return pl.DataFrame(
        {
            "quote_datetime": ["2023-07-03 15:30:00", "2023-07-03 15:30:00"],
            "root": ["XSP", "XSP"],
            "expiration": ["2023-08-04", "2023-08-04"],
            "strike": [450.0, 455.0],
            "option_type": ["C", "C"],
            "bid": [1.00, 0.40],
            "ask": [1.10, 0.50],
            "bid_size": [10, 12],
            "ask_size": [11, 13],
            "trade_volume": [5, 6],
            "open_interest": [100, 120],
            "active_underlying_price": [440.0, 440.0],
        }
    )


class TestNormalization:
    def test_cboe_preset_maps_and_validates(self):
        out = normalize_vendor_frame(cboe_style_frame(), PRESETS["cboe_datashop"])
        assert out.columns == list(CHAIN_SCHEMA)
        validate_chain_frame(out)
        assert out["volume"].to_list() == [5, 6]
        assert out["expiration"][0] == date(2023, 8, 4)

    def test_et_timestamps_convert_to_utc_dst_aware(self):
        out = normalize_vendor_frame(cboe_style_frame(), PRESETS["cboe_datashop"])
        # July 3 is EDT (UTC-4): 15:30 ET -> 19:30 UTC.
        ts = out["ts"][0]
        assert (ts.hour, ts.minute) == (19, 30)
        assert str(out.schema["ts"].time_zone) == "UTC"

    def test_option_type_normalized(self):
        df = cboe_style_frame().with_columns(pl.lit("call").alias("option_type"))
        out = normalize_vendor_frame(df, PRESETS["cboe_datashop"])
        assert set(out["option_type"].to_list()) == {"C"}

    def test_missing_required_column_raises(self):
        df = cboe_style_frame().drop("bid")
        with pytest.raises(VendorFormatError, match="bid"):
            normalize_vendor_frame(df, PRESETS["cboe_datashop"])

    def test_strike_scale(self):
        mapping = PRESETS["cboe_datashop"].model_copy(update={"strike_scale": 0.001})
        df = cboe_style_frame().with_columns((pl.col("strike") * 1000).alias("strike"))
        out = normalize_vendor_frame(df, mapping)
        assert out["strike"].to_list() == [450.0, 455.0]


class TestProxyTransform:
    def test_scales_and_relabels(self):
        spx = normalize_vendor_frame(
            cboe_style_frame().with_columns(
                pl.lit("SPX").alias("root"),
                (pl.col("strike") * 10).alias("strike"),
                (pl.col("bid") * 10).alias("bid"),
                (pl.col("ask") * 10).alias("ask"),
                (pl.col("active_underlying_price") * 10).alias("active_underlying_price"),
            ),
            PRESETS["cboe_datashop"],
        )
        proxy = to_xsp_proxy(spx)
        assert proxy["root"].to_list() == ["SPX_PROXY", "SPX_PROXY"]
        assert proxy["strike"].to_list() == [450.0, 455.0]
        assert proxy["underlying_price"][0] == pytest.approx(440.0)

    def test_never_silently_labeled_xsp(self):
        spx = normalize_vendor_frame(
            cboe_style_frame().with_columns(pl.lit("SPX").alias("root")),
            PRESETS["cboe_datashop"],
        )
        assert "XSP" not in to_xsp_proxy(spx)["root"].to_list()


class TestIngestFile:
    def test_end_to_end_with_manifest(self, tmp_path):
        src = tmp_path / "vendor.csv"
        cboe_style_frame().write_csv(src)
        out = tmp_path / "normalized" / "xsp.parquet"
        manifest = ingest_file(src, out, PRESETS["cboe_datashop"], manifest_dir=tmp_path / "m")
        assert out.exists()
        validate_chain_frame(pl.read_parquet(out))
        assert manifest["rows"] == 2
        assert manifest["transforms"] == []
        assert len(manifest["source_sha256"]) == 64
        on_disk = json.loads((tmp_path / "m" / "xsp.manifest.json").read_text())
        assert on_disk["source_sha256"] == manifest["source_sha256"]

    def test_proxy_transform_recorded_in_manifest(self, tmp_path):
        src = tmp_path / "spx.csv"
        cboe_style_frame().with_columns(pl.lit("SPX").alias("root")).write_csv(src)
        out = tmp_path / "proxy.parquet"
        manifest = ingest_file(src, out, PRESETS["cboe_datashop"], spx_proxy=True)
        assert manifest["transforms"] == ["spx_to_xsp_proxy(/10)"]
        assert manifest["roots"] == ["SPX_PROXY"]


def test_custom_mapping_from_dict():
    mapping = VendorMapping(
        name="custom",
        columns={
            "ts": "t",
            "root": "sym",
            "expiration": "exp",
            "strike": "k",
            "option_type": "cp",
            "bid": "b",
            "ask": "a",
            "underlying_price": "u",
        },
        ts_timezone="UTC",
    )
    df = pl.DataFrame(
        {
            "t": [datetime(2023, 7, 3, 19, 30)],
            "sym": ["XSP"],
            "exp": [date(2023, 8, 4)],
            "k": [450.0],
            "cp": ["C"],
            "b": [1.0],
            "a": [1.1],
            "u": [440.0],
        }
    )
    out = normalize_vendor_frame(df, mapping)
    validate_chain_frame(out)
    assert out["bid_size"][0] is None  # optional columns null-filled
