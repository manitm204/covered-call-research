"""Aggregate the full ablation sweep (results/ablation_full/*.jsonl) into a
deduped parquet + markdown summary with multiple-testing honesty."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

OUT_DIR = Path("results/ablation_full")


def load() -> pl.DataFrame:
    rows = []
    for f in sorted(OUT_DIR.glob("sweep_results_shard*.jsonl")):
        for line in f.read_text().splitlines():
            rows.append(json.loads(line))
    df = pl.DataFrame(rows, infer_schema_length=None)
    df = df.unique(subset="config_id", keep="first")
    return df


def main() -> None:
    df = load()
    n_total = df.height
    errs = df.filter(pl.col("error").is_not_null()) if "error" in df.columns else pl.DataFrame()
    ok = df.filter(pl.col("error").is_null()) if "error" in df.columns else df
    ok = ok.filter(pl.col("n_trades") > 0)
    ok.drop([c for c in ("exit_reasons",) if c in ok.columns]).write_parquet(
        OUT_DIR / "sweep_results.parquet"
    )

    exp = pl.col("expectancy_net_per_spread")
    lines = []
    lines.append(f"# Full ablation sweep — {n_total} configs (exploratory, NOT pre-registered)")
    lines.append("")
    lines.append(
        f"- configs with trades: {ok.height}; errored: {errs.height}; "
        f"zero-trade: {n_total - ok.height - errs.height}"
    )
    pos = ok.filter(exp > 0).height
    ci_pos = ok.filter(pl.col("expectancy_ci_low") > 0).height
    lines.append(
        f"- positive expectancy: {pos}/{ok.height} ({100 * pos / max(ok.height, 1):.1f}%); "
        f"bootstrap 95% CI entirely above zero: {ci_pos} "
        f"(chance expectation under a no-edge null at 2.5% one-sided: "
        f"~{0.025 * ok.height:.0f})"
    )
    q = ok.select(
        exp.quantile(0.05).alias("p5"),
        exp.quantile(0.25).alias("p25"),
        exp.median().alias("median"),
        exp.quantile(0.75).alias("p75"),
        exp.quantile(0.95).alias("p95"),
    ).row(0)
    lines.append(
        f"- expectancy per spread distribution: p5 {q[0]:.1f}, p25 {q[1]:.1f}, "
        f"median {q[2]:.1f}, p75 {q[3]:.1f}, p95 {q[4]:.1f} ($)"
    )
    lines.append("")

    def marginal(col: str) -> None:
        t = (
            ok.group_by(col)
            .agg(
                pl.len().alias("n"),
                exp.median().alias("median_exp"),
                exp.mean().alias("mean_exp"),
                (exp > 0).mean().alias("frac_positive"),
            )
            .sort(col)
        )
        lines.append(f"## Marginal: {col}")
        lines.append("")
        lines.append("| " + col + " | n | median exp $ | mean exp $ | % positive |")
        lines.append("|---|---|---|---|---|")
        for r in t.iter_rows(named=True):
            lines.append(
                f"| {r[col]} | {r['n']} | {r['median_exp']:.2f} | {r['mean_exp']:.2f} "
                f"| {100 * r['frac_positive']:.0f}% |"
            )
        lines.append("")

    for col in ("short_delta", "width", "target_dte", "entry_frequency", "exit_style"):
        marginal(col)

    lines.append("## Top 20 by expectancy (treat as hypotheses, not conclusions)")
    lines.append("")
    lines.append("| config | n_trades | exp $ | 95% CI | win rate | max DD |")
    lines.append("|---|---|---|---|---|---|")
    for r in ok.sort(exp, descending=True).head(20).iter_rows(named=True):
        lines.append(
            f"| {r['config_id']} | {r['n_trades']} | {r['expectancy_net_per_spread']:.2f} "
            f"| [{r.get('expectancy_ci_low', float('nan')):.2f}, "
            f"{r.get('expectancy_ci_high', float('nan')):.2f}] "
            f"| {100 * (r['win_rate'] or 0):.0f}% | {100 * (r['max_drawdown'] or 0):.2f}% |"
        )
    lines.append("")
    lines.append("## Bottom 5 by expectancy")
    lines.append("")
    for r in ok.sort(exp).head(5).iter_rows(named=True):
        lines.append(
            f"- {r['config_id']}: {r['expectancy_net_per_spread']:.2f} "
            f"$/spread over {r['n_trades']} trades"
        )
    out = OUT_DIR / "SUMMARY.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {out} and sweep_results.parquet")


if __name__ == "__main__":
    main()
