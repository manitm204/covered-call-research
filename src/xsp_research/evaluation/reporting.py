"""Reproducible report generation (Markdown + JSON + optional charts).

Reports are regenerable artifacts: everything they contain derives from a
BacktestResult (with its embedded config snapshot and data-source label) plus
optional overlay/stress inputs. Synthetic data sources produce a prominent
warning block at the top of every report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from xsp_research.backtest.engine import BacktestResult
from xsp_research.evaluation.metrics import summarize

SYNTHETIC_BLOCK = (
    "> **WARNING — SYNTHETIC DATA.** This report was generated from simulated market\n"
    "> data and validates software behavior only. Nothing here is evidence about\n"
    "> real-world strategy performance.\n"
)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 100 else f"{v:,.2f}"
    return str(v)


def _dict_table(d: dict[str, Any], keys: list[str] | None = None) -> str:
    keys = keys or [k for k, v in d.items() if not isinstance(v, (dict, list))]
    lines = ["| metric | value |", "| --- | --- |"]
    lines += [f"| {k} | {_fmt(d[k])} |" for k in keys if k in d]
    return "\n".join(lines)


def _by_year(trades: pl.DataFrame) -> str:
    if trades.is_empty():
        return "_no trades_"
    by = (
        trades.with_columns(pl.col("exit_ts").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("trades"),
            pl.col("realized_net").sum().alias("net_pnl"),
            pl.col("realized_gross").sum().alias("gross_pnl"),
            pl.col("fees").sum().alias("fees"),
            (pl.col("realized_net") > 0).mean().alias("win_rate"),
        )
        .sort("year")
    )
    lines = [
        "| year | trades | gross P&L | net P&L | fees | win rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in by.iter_rows(named=True):
        lines.append(
            f"| {r['year']} | {r['trades']} | {r['gross_pnl']:,.2f} | {r['net_pnl']:,.2f} "
            f"| {r['fees']:,.2f} | {r['win_rate']:.1%} |"
        )
    return "\n".join(lines)


def _by_exit_reason(trades: pl.DataFrame) -> str:
    if trades.is_empty():
        return "_no trades_"
    by = (
        trades.group_by("exit_reason")
        .agg(
            pl.len().alias("trades"),
            pl.col("realized_net").sum().alias("net_pnl"),
            pl.col("realized_net").mean().alias("avg_net"),
            pl.col("days_in_trade").mean().alias("avg_days"),
        )
        .sort("net_pnl", descending=True)
    )
    lines = [
        "| exit reason | trades | net P&L | avg net | avg days |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in by.iter_rows(named=True):
        lines.append(
            f"| {r['exit_reason']} | {r['trades']} | {r['net_pnl']:,.2f} "
            f"| {r['avg_net']:,.2f} | {r['avg_days']:.1f} |"
        )
    return "\n".join(lines)


def _stress_section(stress: dict[str, Any]) -> str:
    lines = [
        "| window | kind | status | stock ret | combined ret | overlay P&L | overlay helped |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, row in stress.items():
        if key.startswith("_"):
            continue
        if row.get("status") != "covered":
            lines.append(
                f"| {row['label']} | {row['kind']} | not covered by data | – | – | – | – |"
            )
            continue
        lines.append(
            f"| {row['label']} | {row['kind']} | covered | {row['stock_return']:.2%} "
            f"| {row['combined_return']:.2%} | {row['overlay_pnl']:,.2f} "
            f"| {'yes' if row['overlay_helped'] else 'NO'} |"
        )
    summary = stress.get("_summary", {})
    if summary:
        lines.append("")
        lines.append(f"_{summary.get('note', '')}_")
    return "\n".join(lines)


def _charts(result: BacktestResult, overlay_frame: pl.DataFrame | None, outdir: Path) -> list[str]:
    """Equity/drawdown charts; silently skipped if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written: list[str] = []

    eq = result.equity_curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 1])
    dates = eq["date"].to_list()
    ax1.plot(dates, eq["equity"].to_list(), label="strategy equity")
    if overlay_frame is not None:
        ax1.plot(
            overlay_frame["date"].to_list(),
            overlay_frame["stock_equity"].to_list(),
            label="stock only",
            alpha=0.8,
        )
        ax1.plot(
            overlay_frame["date"].to_list(),
            overlay_frame["combined_equity"].to_list(),
            label="stock + overlay",
            alpha=0.8,
        )
    ax1.set_title(f"Equity ({result.data_source} / {result.execution_scenario})")
    ax1.legend()
    e = eq["equity"].to_numpy()
    peak = e.copy()
    for i in range(1, len(peak)):
        peak[i] = max(peak[i - 1], peak[i])
    ax2.fill_between(dates, e / peak - 1.0, 0, color="firebrick", alpha=0.5)
    ax2.set_title("drawdown")
    fig.tight_layout()
    path = outdir / "equity.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(path.name)
    return written


def generate_report(
    result: BacktestResult,
    outdir: str | Path,
    *,
    overlay_frame: pl.DataFrame | None = None,
    overlay_comparison: dict[str, Any] | None = None,
    stress: dict[str, Any] | None = None,
    scenario_summaries: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Write report.md (+ summary.json, charts) into outdir; returns report path."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(result)
    trades = result.trades_frame()
    charts = _charts(result, overlay_frame, out)

    parts: list[str] = [f"# Backtest report — {result.config_snapshot['name']}", ""]
    if result.data_source == "synthetic":
        parts += [SYNTHETIC_BLOCK, ""]
    parts += [
        f"- data source: `{result.data_source}`",
        f"- execution scenario: `{result.execution_scenario}`",
        f"- period: {summary['equity'].get('start')} → {summary['equity'].get('end')}",
        "- config: see `config_snapshot.json` alongside this report",
        "",
        "## Equity metrics",
        _dict_table(summary["equity"]),
        "",
        "## Trade metrics (net of all modeled costs)",
        _dict_table(summary["trades"]),
        "",
        "## Income attribution",
        _dict_table(
            {
                "trading_pnl_gross": summary["realized_trading_pnl_gross"],
                "total_fees": summary["total_fees"],
                "interest_income (separate from trading)": summary["interest_income_total"],
            }
        ),
        "",
        "## Trades by year",
        _by_year(trades),
        "",
        "## Trades by exit reason",
        _by_exit_reason(trades),
        "",
    ]

    if scenario_summaries:
        parts += ["## Execution-scenario comparison", ""]
        keys = ["total_return", "sharpe", "max_drawdown", "final_equity"]
        parts.append("| scenario | " + " | ".join(keys) + " |")
        parts.append("| --- | " + " | ".join(["---"] * len(keys)) + " |")
        for name, s in scenario_summaries.items():
            eqm = s["equity"]
            parts.append(f"| {name} | " + " | ".join(_fmt(eqm.get(k, "–")) for k in keys) + " |")
        parts += [
            "",
            "_Gross vs net and scenario spread indicate execution-cost sensitivity; "
            "conclusions must hold under the conservative scenario._",
            "",
        ]

    if overlay_comparison:
        parts += [
            "## Portfolio overlay (stock vs stock+overlay)",
            "",
            "### Combined minus stock-only",
            _dict_table(overlay_comparison["combined_minus_stock"]),
            "",
            "### Combined vs stock benchmark",
            _dict_table(overlay_comparison["combined_vs_stock_benchmark"]),
            "",
            f"- overlay trading P&L total: {_fmt(overlay_comparison['overlay_trading_pnl_total'])}",
            f"- overlay P&L on strong stock up-days (>1%): "
            f"{_fmt(overlay_comparison['overlay_pnl_on_strong_up_days_total'])}"
            "  _(capped-upside drag shows up here)_",
            "",
        ]

    if stress:
        parts += ["## Stress windows", _stress_section(stress), ""]

    if charts:
        parts += ["## Charts", *[f"![{c}]({c})" for c in charts], ""]

    parts += [
        "## Entry attempts",
        f"- attempted: {summary['entry_attempts']['attempted']}, "
        f"filled: {summary['entry_attempts']['filled']}",
    ]
    rejected = summary["entry_attempts"]["rejected"]
    if rejected:
        parts += ["", "Rejected entries:"] + [
            f"- {r['session']}: {r['reason']}" for r in rejected[:20]
        ]

    report_path = out / "report.md"
    report_path.write_text("\n".join(parts))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "config_snapshot.json").write_text(
        json.dumps(result.config_snapshot, indent=2, default=str)
    )
    if overlay_comparison:
        (out / "overlay_comparison.json").write_text(
            json.dumps(overlay_comparison, indent=2, default=str)
        )
    if stress:
        (out / "stress.json").write_text(json.dumps(stress, indent=2, default=str))
    return report_path
