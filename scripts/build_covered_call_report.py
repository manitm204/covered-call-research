"""Assemble the covered-call research report HTML: signal research (IC / rank
IC / bootstrap CI / breach probabilities), the derived rip-risk rule board, and
the backtest (buy-hold vs naive vs gated covered call) — from
results/covered_call/{report_data,signal_research}.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from level2_research.signals import asof, build_signals

DATA = json.loads(Path("results/covered_call/report_data.json").read_text())
DATA2 = json.loads(Path("results/covered_call_reinvest/report_data.json").read_text())
RESEARCH = json.loads(Path("results/covered_call/signal_research.json").read_text())
DELTACMP = json.loads(Path("results/covered_call_delta_compare.json").read_text())
GATECMP = json.loads(Path("results/covered_call_gate_compare.json").read_text())
OUT = Path("reports/covered_call_report.html")
STRATS2 = ["buyhold_drip", "naive_cc_reinvest", "gated_cc_reinvest"]
LABELS2 = {"buyhold_drip": "Buy &amp; hold (DRIP)", "naive_cc_reinvest": "Naive covered call (reinvest)",
          "gated_cc_reinvest": "Gated covered call (reinvest)"}

FUNDS = ["SPY", "QQQ", "IWM"]
VOL_SYM = {"SPY": "VIX", "QQQ": "VXN", "IWM": "RVX"}

FMT_PCT = lambda x: f"{x*100:+.1f}%" if x is not None else "—"
FMT_PCT0 = lambda x: f"{x*100:.1f}%" if x is not None else "—"

SIG_LABEL = {
    "rsi14": "RSI(14)", "sector_corr_60": "Sector corr 60d", "trend_84d": "4-month trend",
    "px_vs_ma200": "Price vs MA200", "absorption_shift": "Absorption shift",
    "vol_own": "Own vol index", "vrp_proxy": "VRP (vol idx − realized)",
    "vol_rank_252": "Vol-index rank (252d)", "beta_spy_60": "Beta to SPY (60d)",
}
SIG_ORDER = ["rsi14", "sector_corr_60", "trend_84d", "px_vs_ma200", "absorption_shift",
            "vol_own", "vrp_proxy", "vol_rank_252", "beta_spy_60"]


# --------------------------------------------------------------- IC heatmap
def ic_cell(v: dict | None) -> str:
    if v is None or v.get("ic") != v.get("ic"):  # missing or NaN
        return '<td class="num icdud">—</td>'
    ic = v["ic"]
    mag = min(abs(ic) / 0.20, 1.0)
    if ic >= 0:
        bg = f"rgba(31,122,108,{0.10 + 0.55*mag:.2f})"
    else:
        bg = f"rgba(179,73,47,{0.10 + 0.55*mag:.2f})"
    star = " *" if v.get("sig") else ""
    return f'<td class="num" style="background:{bg}">{ic:+.2f}{star}</td>'


def ic_heatmap(target: str, caption: str) -> str:
    rows = []
    for sc in SIG_ORDER:
        cells = "".join(ic_cell(RESEARCH["ic_table"][target][sym].get(sc)) for sym in FUNDS)
        rows.append(f"<tr><td>{SIG_LABEL[sc]}</td>{cells}</tr>")
    head = "".join(f"<th>{s}</th>" for s in FUNDS)
    return f"""
    <div class="table-wrap">
    <table class="heatmap">
      <thead><tr><th>Signal</th>{head}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    </div>
    <p class="fig-note">{caption} Cell = Spearman rank IC, weekly sampling, 2005&ndash;2026 (n&asymp;1,126/fund).
    * marks a 90% block-bootstrap interval (block=8 weeks, 1,500 resamples) that excludes zero.</p>"""


# --------------------------------------------------------------- forest rows
FOREST_LABEL = {
    "rsi14_lt_40": "RSI(14) < 40", "trend_84d_lt_0": "4-month trend < 0%",
    "px_below_ma200": "Price below MA200", "deep_dd_60_lt_-0.12": "60d drawdown ≤ −12%",
    "vol_top_tercile": "Own vol index in top tercile", "sector_corr_gt_0.70": "Sector corr > 0.70",
    "sector_corr_lt_0.45": "Sector corr < 0.45",
}
FOREST_ORDER = ["rsi14_lt_40", "trend_84d_lt_0", "px_below_ma200", "deep_dd_60_lt_-0.12",
               "vol_top_tercile", "sector_corr_gt_0.70"]

# Exploratory threshold sweep: cutoffs derived per-fund from each signal's own
# weekly distribution (lower/upper tercile, top decile) rather than fixed round
# numbers, tested against both breach probability and plain forward return.
EXPLORE_ORDER = ["sector_corr_lo_tercile", "sector_corr_hi_tercile",
                 "trend_84d_top_decile", "vol_lo_tercile", "vol_hi_tercile"]


def explore_label(sym: str, name: str) -> str:
    c = RESEARCH["cutoffs"][sym]
    return {
        "sector_corr_lo_tercile": f"Sector corr &lt; {c['sector_corr_lo_tercile']:.2f} (lo tercile)",
        "sector_corr_hi_tercile": f"Sector corr &gt; {c['sector_corr_hi_tercile']:.2f} (hi tercile)",
        "trend_84d_top_decile": f"4-month trend &gt; {c['trend_84d_top_decile']*100:.1f}% (top decile)",
        "vol_lo_tercile": f"{VOL_SYM[sym]} &lt; {c['vol_lo_tercile']:.1f} (lo tercile)",
        "vol_hi_tercile": f"{VOL_SYM[sym]} &gt; {c['vol_hi_tercile']:.1f} (hi tercile)",
    }[name]


def forest_plot(sym: str, order=FOREST_ORDER, labels=None, target="breach",
                target_label="breach probability") -> str:
    rows = []
    vals = []
    for name in order:
        v = RESEARCH["forest"][sym].get(name, {}).get(target)
        if v is None:
            continue
        vals.extend([v["lo"], v["hi"]])
    if not vals:
        return ""
    span = max(max(vals, default=1), -min(vals, default=-1), 1.0) * 1.15
    for name in order:
        v = RESEARCH["forest"][sym].get(name, {}).get(target)
        if v is None:
            continue
        label = labels[name] if labels is not None else (
            FOREST_LABEL[name] if name in FOREST_LABEL else explore_label(sym, name))
        lo_pct = (v["lo"] + span) / (2 * span) * 100
        hi_pct = (v["hi"] + span) / (2 * span) * 100
        pt_pct = (v["point"] + span) / (2 * span) * 100
        sig = v["lo"] > 0 or v["hi"] < 0
        color = "var(--accent)" if sig else "#8a8f98"
        rows.append(f"""
        <div class="forest-row">
          <div class="forest-label">{label}</div>
          <div class="forest-track">
            <div class="forest-zero"></div>
            <div class="forest-bar" style="left:{lo_pct:.1f}%; width:{max(hi_pct-lo_pct,0.5):.1f}%; background:{color}"></div>
            <div class="forest-pt" style="left:{pt_pct:.1f}%; background:{color}"></div>
          </div>
          <div class="forest-val num">{v['point']:+.1f}pp <span class="muted">n={v['n']}</span></div>
        </div>""")
    return f"""
    <div class="forest">
      <div class="forest-row forest-axis">
        <div class="forest-label"></div>
        <div class="forest-track"><span>&minus;{span:.1f}</span><span style="margin-left:auto">+{span:.1f}pp</span></div>
        <div class="forest-val"></div>
      </div>
      {''.join(rows)}
    </div>
    <p class="fig-note">Change in {target_label} when the condition is true vs
    false, with a 90% block-bootstrap interval. Teal = interval excludes zero.</p>"""


# --------------------------------------------------------------- rule board
# Per-ticker leg selection, mirrors scripts/covered_call_backtest.py exactly —
# kept in sync manually since this script only renders, it doesn't backtest.
BREACH_USE_MA200 = {"SPY": True, "QQQ": True, "IWM": False}
FWDRET_THRESH = {
    "SPY": dict(rsi=50.0, corr=0.677, trend=0.0, vol=19.6, use_ma200=False),
    "QQQ": dict(rsi=48.2, corr=0.677, trend=0.0, vol=23.2, use_ma200=False),
    "IWM": dict(rsi=47.5, corr=0.677, trend=0.0, vol=25.2, use_ma200=True),
}
def rule_board() -> str:
    cards = []
    for sym in FUNDS:
        sig_df = build_signals(sym)
        s = asof(sig_df, sig_df.index[-1])
        oversold = s["rsi14"] < 40
        weak_trend = s["ret84"] < 0
        below_ma = BREACH_USE_MA200[sym] and s["px"] < s["ma200"]
        veto = bool(oversold or weak_trend or below_ma)
        badge = "SKIP" if veto else "WRITE"
        badge_cls = "veto" if veto else "enter"
        legs = [
            ("RSI(14)", f"{s['rsi14']:.1f}", "veto if &lt; 40", oversold),
            ("4-month trend", f"{s['ret84']*100:+.1f}%", "veto if &lt; 0%", weak_trend),
        ]
        if BREACH_USE_MA200[sym]:
            legs.append(("Price vs MA200", f"{(s['px']/s['ma200']-1)*100:+.1f}%",
                        "veto if below (&lt;0%)", below_ma))
        else:
            legs.append(("Price vs MA200", f"{(s['px']/s['ma200']-1)*100:+.1f}%",
                        "not used &mdash; CI crosses zero on {}".format(sym), False))
        leg_html = "".join(f"""
          <div class="leg">
            <span class="leg-check">{'&#10007;' if fired else '&#10003;'}</span>
            <span class="leg-name">{name}</span>
            <span class="leg-val num">{val}</span>
            <span class="leg-rule muted">{rule}</span>
          </div>""" for name, val, rule, fired in legs)
        confidence_html = ""
        if sym == "SPY":
            hi = s["ret84"] > 0.1279
            confidence_html = f"""
          <div class="leg confidence">
            <span class="leg-check">{'&#9679;' if hi else '&middot;'}</span>
            <span class="leg-name">4-month trend</span>
            <span class="leg-val num">{s['ret84']*100:+.1f}%</span>
            <span class="leg-rule muted">&gt; +12.8% (top decile) &rarr; breach risk measurably
            *lower*, informational only, doesn't change write/skip</span>
          </div>"""
        elif sym == "IWM":
            confidence_html = f"""
          <div class="leg confidence">
            <span class="leg-check">&middot;</span>
            <span class="leg-name">{VOL_SYM[sym]} (own vol index)</span>
            <span class="leg-val num">&mdash;</span>
            <span class="leg-rule muted">bottom tercile &rarr; breach risk measurably lower,
            informational only, doesn't change write/skip</span>
          </div>"""
        cards.append(f"""
        <div class="rule-card">
          <div class="rule-card-head">
            <span class="tick">{sym}</span> <span class="muted">/ {VOL_SYM[sym]}</span>
            <span class="badge {badge_cls}">{sym} &middot; {badge}</span>
          </div>
          {leg_html}
          {confidence_html}
          <p class="rule-note muted">As of {s.name.date()} close (session t&minus;1 signal, governs a
          write placed now).</p>
        </div>""")
    return f'<div class="rule-grid">{"".join(cards)}</div>'


def fwdret_rule_board() -> str:
    cards = []
    for sym in FUNDS:
        sig_df = build_signals(sym)
        s = asof(sig_df, sig_df.index[-1])
        th = FWDRET_THRESH[sym]
        oversold = s["rsi14"] < th["rsi"]
        high_corr = s["sector_corr_60"] > th["corr"]
        weak_trend = s["ret84"] < th["trend"]
        high_vol = s[f"vol_own"] > th["vol"] if "vol_own" in s.index else False
        below_ma = th["use_ma200"] and s["px"] < s["ma200"]
        veto = bool(oversold or high_corr or weak_trend or high_vol or below_ma)
        badge = "SKIP" if veto else "WRITE"
        badge_cls = "veto" if veto else "enter"
        legs = [
            ("RSI(14)", f"{s['rsi14']:.1f}", f"veto if &lt; {th['rsi']:.1f} (lo tercile)", oversold, True),
            ("Sector corr 60d", f"{s['sector_corr_60']:.2f}", f"veto if &gt; {th['corr']:.3f} (hi tercile)", high_corr, True),
            ("4-month trend", f"{s['ret84']*100:+.1f}%", "veto if &lt; 0% &mdash; weaker evidence, see note below", weak_trend, False),
            (f"{VOL_SYM[sym]} (own vol index)", f"{s.get('vol_own', float('nan')):.1f}",
             f"veto if &gt; {th['vol']:.1f} (hi tercile){'&mdash; weaker evidence for SPY/QQQ, see note below' if sym != 'IWM' else ''}",
             high_vol, sym == "IWM"),
        ]
        if th["use_ma200"]:
            legs.append(("Price vs MA200", f"{(s['px']/s['ma200']-1)*100:+.1f}%",
                        "veto if below (&lt;0%) &mdash; weaker evidence, see note below", below_ma, False))
        leg_html = "".join(f"""
          <div class="leg">
            <span class="leg-check">{'&#10007;' if fired else '&#10003;'}</span>
            <span class="leg-name">{name}</span>
            <span class="leg-val num">{val}</span>
            <span class="leg-rule muted">{rule}</span>
          </div>""" for name, val, rule, fired, solid in legs)
        cards.append(f"""
        <div class="rule-card">
          <div class="rule-card-head">
            <span class="tick">{sym}</span> <span class="muted">/ {VOL_SYM[sym]}</span>
            <span class="badge {badge_cls}">{sym} &middot; {badge}</span>
          </div>
          {leg_html}
          <p class="rule-note muted">As of {s.name.date()} close (session t&minus;1 signal, governs a
          write placed now).</p>
        </div>""")
    return f'<div class="rule-grid">{"".join(cards)}</div>'


def vol_chart_grid() -> str:
    imgs = "".join(f"""
      <figure class="chart small">
        <img src="data:image/png;base64,{RESEARCH['vol_charts'][sym]}" alt="{sym} vol regime" />
      </figure>""" for sym in FUNDS)
    return f'<div class="chart-grid">{imgs}</div>'


# --------------------------------------------------------------- backtest tables
def metrics_table(sym: str, data=DATA, strats=("buyhold", "naive_cc", "gated_cc", "fwdret_cc"),
                  labels=None, hl=("gated_cc", "fwdret_cc"), extra_col=None) -> str:
    labels = labels or {"buyhold": "Buy &amp; hold", "naive_cc": "Naive monthly covered call",
                        "gated_cc": "Breach-gated covered call",
                        "fwdret_cc": "Forward-return-gated covered call"}
    hl = (hl,) if isinstance(hl, str) else hl
    rows = []
    for strat in strats:
        s = data[sym]["strategies"][strat]
        cls = " class=\"hl\"" if strat in hl else ""
        extra = f'<td class="num">{s[extra_col[1]]}</td>' if extra_col else ""
        rows.append(f"""
        <tr{cls}>
          <td>{labels[strat]}</td>
          <td class="num">{FMT_PCT(s['cagr'])}</td>
          <td class="num">{s['sharpe']:.2f}</td>
          <td class="num">{FMT_PCT0(s['max_drawdown'])}</td>
          <td class="num">${s['final_equity']:,.0f}</td>
          <td class="num">{s['n_calls_written']}</td>
          <td class="num">{s['n_called_away']}</td>
          {extra}
        </tr>""")
    extra_head = f"<th>{extra_col[0]}</th>" if extra_col else ""
    return f"""
    <div class="table-wrap">
    <table>
      <thead><tr><th>Strategy</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th>
      <th>Final equity</th><th>Calls written</th><th>Called away</th>{extra_head}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    </div>"""


def yearly_table(sym: str, data=DATA, strats=("buyhold", "naive_cc", "gated_cc", "fwdret_cc"),
                 labels=None, hl=("gated_cc", "fwdret_cc")) -> str:
    labels = labels or {"buyhold": "Buy &amp; hold", "naive_cc": "Naive CC", "gated_cc": "Breach CC",
                        "fwdret_cc": "Fwd-ret CC"}
    hl = (hl,) if isinstance(hl, str) else hl
    years = sorted(data[sym]["strategies"][strats[0]]["yearly"].keys())
    head = "".join(f"<th>{y}</th>" for y in years)
    rows = []
    for strat in strats:
        yy = data[sym]["strategies"][strat]["yearly"]
        cls = " class=\"hl\"" if strat in hl else ""
        cells = "".join(
            f'<td class="num {"neg" if yy.get(y, 0) < 0 else ""}">{FMT_PCT(yy.get(y))}</td>'
            for y in years)
        rows.append(f'<tr{cls}><td>{labels[strat]}</td>{cells}</tr>')
    return f"""
    <div class="table-wrap">
    <table class="yearly">
      <thead><tr><th>Strategy</th>{head}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    </div>"""


def section(sym: str, narrative: str) -> str:
    d = DATA[sym]
    return f"""
    <section id="{sym.lower()}">
      <h2><span class="tick">{sym}</span> backtest</h2>
      {narrative}
      <figure class="chart">
        <img src="data:image/png;base64,{d['equity_chart']}" alt="{sym} equity curves" />
        <figcaption>Equity, one 100-share lot, rebased to 100 at inception (2018-08-01).</figcaption>
      </figure>
      <figure class="chart dd">
        <img src="data:image/png;base64,{d['dd_chart']}" alt="{sym} drawdowns" />
        <figcaption>Drawdown from trailing peak.</figcaption>
      </figure>
      {metrics_table(sym)}
      <h3>Year by year</h3>
      {yearly_table(sym)}
    </section>"""


def reinvest_section(sym: str, narrative: str) -> str:
    d = DATA2[sym]
    shares = d["strategies"]["gated_cc_reinvest"]
    return f"""
    <section id="{sym.lower()}-reinvest">
      <h2><span class="tick">{sym}</span> reinvesting variant</h2>
      {narrative}
      <figure class="chart">
        <img src="data:image/png;base64,{d['equity_chart']}" alt="{sym} reinvesting equity curves" />
        <figcaption>Equity, 200-share start, rebased to 100 at inception (2018-08-01).</figcaption>
      </figure>
      <figure class="chart dd">
        <img src="data:image/png;base64,{d['dd_chart']}" alt="{sym} reinvesting drawdowns" />
        <figcaption>Drawdown from trailing peak.</figcaption>
      </figure>
      {metrics_table(sym, data=DATA2, strats=STRATS2, labels=LABELS2, hl="gated_cc_reinvest",
                      extra_col=("Shares 200&rarr;", "final_shares"))}
      <h3>Year by year</h3>
      {yearly_table(sym, data=DATA2, strats=STRATS2, labels=LABELS2, hl="gated_cc_reinvest")}
    </section>"""


# --------------------------------------------------------------- narrative copy
SPY_NARRATIVE = """
<p>SPY is the clean win, and it's the breach gate that delivers it. Skipping
a write whenever RSI(14) &lt; 40, the 4-month trend is negative, or price is
below its 200-day MA doesn't just close the gap to buy-and-hold, it
<b>closes it and keeps going</b>: 14.3% CAGR vs buy-and-hold's 14.0%, Sharpe
0.75 vs 0.70, for essentially the same drawdown (&minus;32.8% vs
&minus;33.2%). The naive version, by contrast, gives up about 2 CAGR points a
year (12.1%) for barely any drawdown protection. The mechanism: the veto rule
keeps the account uncovered specifically through the oversold/weak-trend
windows this project's IC study found carry SPY's largest upside tails
(median tail size when RSI&lt;40: +1.76pp higher than unconditional, 90% CI
excludes zero &mdash; see <a href="#research">Signal research</a>), so it
avoids capping exactly the rebounds that would otherwise blow through the
strike. The forward-return gate (13.0% CAGR, Sharpe 0.64, 40 calls written vs
the breach gate's 63) is more conservative on SPY &mdash; it sits out more
weeks than it needs to for breach purposes specifically, trading away some of
the outperformance.</p>
"""

QQQ_NARRATIVE = """
<p>QQQ is the closest race in the study: naive 18.35% CAGR, breach-gated
18.38%, forward-return-gated 17.88%, buy-and-hold 19.18% &mdash; every
covered-call variant lands within 1.3 points of buy-and-hold and within half
a point of each other. What separates them is mostly a single assignment: a
call struck $198&ndash;209 got assigned on 2020-05-08 while QQQ was already
trading near $225, right in the middle of the COVID-recovery melt-up. The
account buys back as many shares as the proceeds afford and tops the rest up
over time as dividends and interest come in, so it spends the next several
years running a few shares light of a full 100-share lot rather than losing
the position outright &mdash; real, measurable drag, but a recoverable one.
<b>The honest lesson: an early bad assignment during a persistent rally costs
real ground, but a cash account that keeps redeploying whatever it has
recovers most of the way, given years to do it.</b> The forward-return gate,
built to sit out rally-into-correlated-market regimes, writes fewer calls
(18 vs the breach gate's 20) but is narrowly the weakest of the three CC
variants here &mdash; on QQQ the extra caution doesn't pay for itself the way
it does on IWM.</p>
"""

IWM_NARRATIVE = """
<p>IWM is where the two rules genuinely diverge, and the forward-return gate
wins. Breach-gated (RSI and 4-month trend only &mdash; price vs MA200 doesn't
clear a 90% CI on IWM, so it's dropped for this ticker) comes in at 8.55%
CAGR, Sharpe 0.38 &mdash; actually <i>below</i> naive's 9.04%/0.40 on both
counts, though it does have the shallowest drawdown of the four variants
(&minus;38.8%). The forward-return gate does better on every axis at once:
<b>9.32% CAGR, Sharpe 0.42</b> &mdash; the best of all four IWM variants,
including buy-and-hold's 0.36 &mdash; with a drawdown of &minus;39.1%, nearly
as shallow as the breach gate's. It writes 37 calls over the backtest, a
normal cadence, using RSI, sector correlation, and IWM's own vol index (RVX)
rather than the trend/MA signals that don't hold up as well on this ticker.
Given this repository's general experience that IWM signals travel less
reliably than SPY/QQQ's, treat the size of this edge cautiously &mdash; but
the direction is consistent and the mechanism is the same story as elsewhere:
skip weeks that are set up to be simply good, rather than trying to time a
specific breach.</p>
"""

OVERVIEW = f"""
<section id="overview">
  <p class="lede">You own 100 shares of SPY, QQQ, or IWM. Every month you can sell
  an out-of-the-money call against them and collect income &mdash; the textbook
  covered call. The problem, as you put it: if the stock rips, the naive version
  of this trade loses to just holding the shares, because the call caps exactly
  the move you needed. This report redoes this project's regime-signal research
  from scratch &mdash; rank IC, 90% block-bootstrap confidence intervals, breach
  probabilities &mdash; scored specifically against the question a covered-call
  writer cares about (does the underlying breach the strike?), derives two
  veto rules from what actually clears, and backtests them against a
  naive monthly covered call and plain buy-and-hold on real 2018&ndash;2026
  options-chain data.</p>

  <h3>Method</h3>
  <p>Nine regime signals &mdash; RSI(14), 60-day sector correlation, 4-month
  (84-session) trend, price vs. 200-day MA, a market-wide absorption-ratio
  shift, each fund's own vol index level (VIX for SPY, VXN for QQQ, RVX for
  IWM), a vol-risk-premium proxy, the vol index's own 252-day percentile rank,
  and 60-day beta to SPY &mdash; scored weekly (Friday close), 2005-01 through
  2026-07, against three one-month-forward targets that matter for a short
  call: plain forward return, <b>breach</b> (does the max close over the next
  21 sessions pierce a strike 1.2&times; the fund's own implied monthly
  &sigma; away), and the continuous upside-tail size. Every correlation is a
  Spearman rank IC with a 90% block-bootstrap confidence interval (8-week
  blocks, 1,500 resamples) so a signal only counts as real if the interval
  excludes zero. The options backtest itself is separate and later: real daily
  chain snapshots for SPY/QQQ/IWM, one 100-share lot each, three variants
  (buy-and-hold, naive monthly covered call, and the gated version), this
  project's existing execution-cost and cash-account assignment mechanics.</p>

  <h3 id="research">Signal research</h3>
  <p>The headline finding flips the intuition most covered-call writers start
  with: <b>breach risk is highest in weak tape, not strong.</b> RSI, 4-month
  trend, and price-vs-MA200 all correlate <i>negatively</i> with breach
  probability on every fund &mdash; the dangerous weeks to sell a call are the
  oversold ones, because the big one-month upside tails are rebound rallies out
  of drawdowns, not continuations of a melt-up. And a fund's own vol index
  cannot predict a breach of its own vol-scaled band: it has the largest IC on
  raw tail <i>size</i> of anything tested (elevated vol means bigger moves,
  mechanically), but is indistinguishable from zero on <i>breach</i>, because
  the option is already priced for that.</p>
  <h4>Rank IC vs breach probability (1.2&sigma;, 21 sessions)</h4>
  {ic_heatmap("breach_1.2", "The target that matters for a short call: does the underlying pierce a strike 1.2&times; the implied monthly move away within a month?")}
  <h4>Rank IC vs plain forward return</h4>
  {ic_heatmap("fwd_ret", "For comparison: the conventional target (does the fund go up next month?). Weaker and less consistent than the breach columns above.")}

  <h3>Breach rates and the vol-index terciles</h3>
  <div class="table-wrap">
  <table>
    <thead><tr><th>Ticker</th><th>Breach rate (1.0&sigma;)</th><th>Breach rate (1.2&sigma;)</th>
    <th>Breach rate (1.5&sigma;)</th><th>Median upside tail</th><th>Vol tercile cutoff</th></tr></thead>
    <tbody>
    {''.join(f'''<tr><td>{sym}</td><td class="num">{RESEARCH['breach_rates'][sym]['1.0']}%</td>
    <td class="num">{RESEARCH['breach_rates'][sym]['1.2']}%</td>
    <td class="num">{RESEARCH['breach_rates'][sym]['1.5']}%</td>
    <td class="num">+{RESEARCH['breach_rates'][sym]['median_upside_tail']}%</td>
    <td class="num">{VOL_SYM[sym]} &ge; {RESEARCH['vol_tercile'][sym]}</td></tr>''' for sym in FUNDS)}
    </tbody>
  </table>
  </div>
  {vol_chart_grid()}

  <h3>Threshold breach-probability deltas</h3>
  <p>For each ticker: how much higher is 1.2&sigma;-breach probability when the
  condition is true, vs when it's false? A 90% block-bootstrap interval that
  excludes zero (teal) is what promotes a condition into the rule below.</p>
  {''.join(f'<h4>{sym}</h4>' + forest_plot(sym) for sym in FUNDS)}

  <h3 id="sensitivity">Threshold sensitivity: testing cutoffs beyond the ones in the rule</h3>
  <p>The legs in the rule below use either a fixed conventional cutoff (RSI 40,
  trend 0%, MA200) or a top-tercile split (sector correlation, the vol index,
  in the forward-return gate). To check whether other natural cutoffs tell a
  different story, five more splits &mdash; each derived from that signal's
  own weekly distribution per ticker, not hand-picked &mdash; were tested
  against <b>both</b> breach probability and plain forward return:</p>
  <ul>
    <li><b>Sector correlation, lower tercile</b> (below the bottom third of
    its own range) and <b>upper tercile</b> (above the top third) &mdash;
    testing both ends instead of just the high side already in the rule.</li>
    <li><b>4-month trend, top decile</b> (the strongest tenth of readings)
    &mdash; does an already-strong trend behave differently from just
    "trend &gt; 0%"?</li>
    <li><b>Each fund's own vol index, lower and upper tercile</b> &mdash;
    testing the bottom third as well as the top third.</li>
  </ul>
  <h4>vs. breach probability</h4>
  {''.join(f'<h5>{sym}</h5>' + forest_plot(sym, order=EXPLORE_ORDER, target="breach", target_label="breach probability") for sym in FUNDS)}
  <h4>vs. plain forward return</h4>
  {''.join(f'<h5>{sym}</h5>' + forest_plot(sym, order=EXPLORE_ORDER, target="fwd_ret", target_label="plain forward return") for sym in FUNDS)}
  <p class="fig-note">Two of these five did clear the bar and are now in
  Rule 2 below (sector correlation upper tercile; RSI lower tercile, tested
  the same way against breach here and shown again there). The rest are
  informational: 4-month trend top decile lowers breach risk on SPY
  specifically (added to that rule board as a confidence note, not a veto);
  vol index lower tercile does the same for IWM. Nothing here changes a rule
  without clearing the same 90% CI bar and having a clear mechanism, the same
  standard applied throughout this study.</p>

  <h3 id="why-not">Two different questions: "will it breach?" vs. "will it be a good month?"</h3>
  <p>Sector correlation (the mean pairwise 60-day correlation among the 9
  sector SPDRs &mdash; is the market trading as one tide, or fragmented?) and
  each fund's own vol index turn out to answer a different question than the
  one this rule needs. Both have a real, significant IC on <b>plain forward
  return</b> for all three funds (sector correlation: SPY +0.15, QQQ +0.15,
  IWM +0.16, all 90% CI exclude zero) &mdash; a correlated ("one tide") market
  predicts a better next month. But on the target that actually matters for a
  short call &mdash; <b>breach probability</b> &mdash; both are
  indistinguishable from zero on all three funds. They correctly tell you
  whether to expect a good month, not whether that month's move will be
  violent enough to blow through a specific strike (which is already
  vol-scaled, so elevated vol is already priced in) &mdash; so they stay out
  of the breach-based veto rule below, and instead drive a second rule built
  specifically around plain forward return (see
  <a href="#gatecompare">the forward-return gate</a>).</p>

  <h3 id="rule">Rule 1 &mdash; the breach-probability veto (derived, current state)</h3>
  <p>Skip writing a new covered call this cycle if <b>any</b> leg fires. Every
  leg below cleared a 90% CI on breach probability, on every fund it's applied
  to, in the study above.</p>
  {rule_board()}

  <h3 id="rule2">Rule 2 &mdash; the forward-return veto</h3>
  <p>A second, differently-built rule: instead of every signal that predicts
  a <i>breach</i>, this one uses every signal whose continuous rank IC clears
  a 90% CI against <b>plain forward return</b> on that fund (dropping
  absorption-ratio shift and the vol index's own 252-day rank as
  near-duplicates of sector correlation and the vol index level &mdash; see
  the caveat above on correlated signals). Skip writing this cycle if
  <b>any</b> leg below fires. Two legs, RSI lower-tercile and sector-corr
  upper-tercile, also clear a bucket-level 90% CI test the same way Rule 1's
  legs do (see <a href="#sensitivity">threshold sensitivity</a>); the
  4-month-trend and vol-index legs are real on the continuous correlation but
  weaker at this specific cutoff &mdash; included because the underlying
  signal is real, flagged so the strength of the evidence is honest rather
  than uniform.</p>
  {fwdret_rule_board()}
  <p class="fig-note">Full head-to-head backtest results for both rules are in
  <a href="#gatecompare">Breach vs. forward-return gate</a> below.</p>

  <h3>Cross-ticker backtest summary</h3>
  <div class="table-wrap">
  <table>
    <thead><tr><th>Ticker</th><th>Strategy</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr></thead>
    <tbody>
    {(lambda strat_label={'buyhold':'Buy &amp; hold','naive_cc':'Naive covered call','gated_cc':'Breach-gated covered call','fwdret_cc':'Forward-return-gated covered call'}: ''.join(f'''
    <tr class="{'hl' if strat in ('gated_cc','fwdret_cc') else ''}">
      <td>{sym if strat=='buyhold' else ''}</td>
      <td>{strat_label[strat]}</td>
      <td class="num">{FMT_PCT(DATA[sym]['strategies'][strat]['cagr'])}</td>
      <td class="num">{DATA[sym]['strategies'][strat]['sharpe']:.2f}</td>
      <td class="num">{FMT_PCT0(DATA[sym]['strategies'][strat]['max_drawdown'])}</td>
    </tr>''' for sym in FUNDS for strat in ["buyhold","naive_cc","gated_cc","fwdret_cc"]))()}
    </tbody>
  </table>
  </div>

  <div class="callout">
    <p><b>What decides most of the remaining SPY/QQQ gap isn't "premium
    collected vs. foregone upside" month to month &mdash; it's what happens
    after an assignment.</b> This backtest models a strict cash account: get
    called away, and the next session it buys back as many shares as it can
    afford, redeploying more as dividends and interest accumulate. When the
    stock keeps running past the strike, that can leave the account a few
    shares short of a full lot for a long stretch &mdash; real, measurable
    drag, but one the account works back down over the following years rather
    than a permanent loss of position.</p>
  </div>

  <h3>Per-ticker verdict</h3>
  <ul class="verdicts">
    <li><b><a href="#spy">SPY</a></b> &mdash; the <b>breach gate</b> wins
    outright: higher CAGR <i>and</i> higher Sharpe than buy-and-hold, same
    drawdown (14.3% / 0.75 vs 14.0% / 0.70). The forward-return gate is more
    conservative here and gives some of that back (13.0%).</li>
    <li><b><a href="#qqq">QQQ</a></b> &mdash; all three covered-call variants
    land within 1.3 CAGR points of buy-and-hold and within half a point of
    each other (17.9&ndash;18.4% vs 19.2%); breach-gated is narrowly best of
    the three here. A 2020 assignment during the COVID-recovery rally costs
    real ground but the account recovers most of it over the following
    years.</li>
    <li><b><a href="#iwm">IWM</a></b> &mdash; the <b>forward-return gate</b>
    wins outright here: higher CAGR, higher Sharpe, and nearly as shallow a
    drawdown as the breach gate (9.32% / 0.42 vs buy-and-hold's 8.29% /
    0.36). The breach gate actually trails naive on CAGR and Sharpe on this
    ticker, though it does have the single shallowest drawdown of the four
    variants. Treat the size of either edge cautiously given this repo's
    track record with IWM signals elsewhere.</li>
  </ul>
</section>"""

SPY_REINVEST_NARRATIVE = """
<p>Same rule, different account structure: start with 200 shares instead of
100, write covered calls on only the first 100 (the other 100 stay
permanently uncovered as a buffer), and sweep every dollar that comes in
&mdash; premium, assignment proceeds, dividends, interest &mdash; straight
back into more whole shares instead of letting it sit as cash. On SPY this
essentially matches the flat single-lot result: the gated variant edges out
DRIP buy-and-hold on both CAGR (14.8% vs 14.8%, a hair ahead) and Sharpe
(0.72 vs 0.70), with a very slightly shallower drawdown &mdash; real, but
modest, not a blowout.</p>
"""

QQQ_REINVEST_NARRATIVE = """
<p>The gated variant <b>beats DRIP buy-and-hold on Sharpe (0.82 vs 0.79) and
drawdown (&minus;34.3% vs &minus;34.9%), and is essentially tied on CAGR
(19.7% vs 19.7%)</b> &mdash; a real, if modest, risk-adjusted win. Inside the
reinvesting structure, gated also clearly beats naive (19.7% vs 15.8%), the
same story as everywhere else in this report: avoiding writes into the
rip-risk weeks pays off. The flat single-lot and reinvesting structures were
built independently, so treat a direct flat-vs-reinvest comparison as
suggestive rather than a clean apples-to-apples result.</p>
"""

IWM_REINVEST_NARRATIVE = """
<p>Gated reinvesting is IWM's clearest, most robust win across every variant
in this report: CAGR 9.7% vs buy-and-hold's 8.8%, Sharpe 0.41 vs 0.37, and the
shallowest drawdown of the three (&minus;42.1% vs &minus;42.6%), on a normal
trading cadence (52 calls in 8 years). The naive version actually
<b>underperforms</b> buy-and-hold here: 8.0% CAGR, &minus;43.8% drawdown
&mdash; selling calls every month on IWM with no regard for regime is a net
negative once cash is kept properly working. As before, treat the size of the
gated edge more cautiously than SPY's or QQQ's given this repository's track
record with IWM signals elsewhere &mdash; but the gated-vs-naive gap here is
the widest and most consistent in the whole study.</p>
"""

REINVEST_OVERVIEW = f"""
<section id="reinvest">
  <h2>The reinvesting variant</h2>
  <p class="lede">A follow-up structure: own 200 shares instead of 100, write
  covered calls on only the first 100 (the other 100 permanently uncovered as
  a buffer), and sweep every dollar that comes in &mdash; option premium,
  assignment proceeds, dividends, interest, all of it &mdash; straight back
  into more whole shares rather than letting it sit as cash. Same rip-risk
  gate, same execution model, same real chain data; the only change is that
  the position compounds instead of staying pinned at one lot, and only half
  of it is ever exposed to the call-capping risk at a time. Baseline is now a
  DRIP buy-and-hold (200 shares, dividends reinvested into more shares) for a
  fair like-for-like comparison.</p>

  <div class="table-wrap">
  <table>
    <thead><tr><th>Ticker</th><th>Strategy</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Shares 200&rarr;</th></tr></thead>
    <tbody>
    {(lambda strat_label=LABELS2: ''.join(f'''
    <tr class="{'hl' if strat=='gated_cc_reinvest' else ''}">
      <td>{sym if strat=='buyhold_drip' else ''}</td>
      <td>{strat_label[strat]}</td>
      <td class="num">{FMT_PCT(DATA2[sym]['strategies'][strat]['cagr'])}</td>
      <td class="num">{DATA2[sym]['strategies'][strat]['sharpe']:.2f}</td>
      <td class="num">{FMT_PCT0(DATA2[sym]['strategies'][strat]['max_drawdown'])}</td>
      <td class="num">{DATA2[sym]['strategies'][strat]['final_shares']}</td>
    </tr>''' for sym in FUNDS for strat in STRATS2))()}
    </tbody>
  </table>
  </div>

  <div class="callout">
    <p><b>Headline: the gated variant beats its DRIP buy-and-hold baseline on
    Sharpe and drawdown for all three tickers</b>, and roughly ties or edges
    it on CAGR (SPY, QQQ) or clearly beats it (IWM).</p>
  </div>
</section>"""

def delta_table(variant: str, sym: str) -> str:
    rows_d = DELTACMP[variant][sym]["rows"]
    gated_label = "Gated (breach), 0.25Δ" if variant == "flat" else "Gated, 0.25Δ"
    gated_label15 = "Gated (breach), 0.15Δ" if variant == "flat" else "Gated, 0.15Δ"
    order = ["Buy & hold", "Naive, 0.25Δ", "Naive, 0.15Δ", gated_label, gated_label15]
    if variant == "flat":
        order += ["Gated (fwd-ret), 0.25Δ", "Gated (fwd-ret), 0.15Δ"]
    rows = []
    for label in order:
        r = rows_d[label]
        cls = " class=\"hl\"" if label in (gated_label15, "Gated (fwd-ret), 0.15Δ") else ""
        rows.append(f"""
        <tr{cls}>
          <td>{label}</td>
          <td class="num">{FMT_PCT(r['cagr'])}</td>
          <td class="num">{r['sharpe']:.2f}</td>
          <td class="num">{FMT_PCT0(r['max_drawdown'])}</td>
          <td class="num">{r['n_calls']}</td>
          <td class="num">{r['n_called_away']}</td>
        </tr>""")
    return f"""
    <div class="table-wrap">
    <table>
      <thead><tr><th>Strategy</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th>
      <th>Calls</th><th>Called away</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    </div>"""


def delta_chart(variant: str, sym: str) -> str:
    img = DELTACMP[variant][sym]["chart"]
    return f"""
    <figure class="chart">
      <img src="data:image/png;base64,{img}" alt="{sym} {variant} delta comparison" />
      <figcaption>Solid = 0.25&Delta; (this report's default so far). Dashed = 0.15&Delta;.</figcaption>
    </figure>"""


DELTA_SECTION = f"""
<section id="delta">
  <h2>Delta sensitivity: 0.25&Delta; vs 0.15&Delta;</h2>
  <p class="lede">2.7&ndash;4.6% out-of-the-money (this report's 0.25&Delta;
  target) is tight for a monthly write. Re-ran everything &mdash; naive and
  gated, flat single-lot and reinvesting, all three tickers &mdash; at
  0.15&Delta; instead, which pushes the strike further out (roughly
  5&ndash;9% OTM depending on the ticker's own volatility) for a smaller,
  safer premium.</p>

  <h3>What changed</h3>
  <p>QQQ is the one place the breach gate has a visible blind spot: a call
  written 2020-04-14, during a confirmed uptrend the gate correctly didn't
  veto, got blown through by the COVID-recovery melt-up and assigned at a
  strike well below where QQQ kept running &mdash; it protects against
  rebounds out of weak tape, not against a strong rally continuing past an
  already-written strike. Even so, QQQ breach-gated runs 18.3&ndash;18.4%
  CAGR at <i>both</i> deltas, close to buy-and-hold, because the account
  recovers the lost ground over the following years.</p>
  <p>For the breach gate, the delta effect is modest and mixed rather than
  dramatic: on SPY, 0.15&Delta; helps naive (12.1%&rarr;13.1%) and very
  slightly hurts gated (14.3%&rarr;14.0%); on QQQ both deltas land within
  half a point of each other; on IWM 0.15&Delta; now runs ahead of 0.25&Delta;
  (8.55%&rarr;8.96%) since the leaner IWM rule (RSI + trend only, no MA200)
  writes more often and the smaller premium at 0.15&Delta; matters less. The
  <b>forward-return gate is more delta-sensitive than the breach gate</b>:
  QQQ's forward-return CAGR jumps from 17.88% at 0.25&Delta; to <b>18.64% at
  0.15&Delta;</b> (Sharpe 0.74&rarr;0.79, matching buy-and-hold's Sharpe
  exactly) as it writes more than twice as many calls (18&rarr;41) at the
  wider, safer strike. There's no single consistent "better" delta across
  every ticker and rule combination here &mdash; see the caveat below on why
  two points aren't enough to fully characterize this.</p>

  {"".join(f'<h4>{sym} &mdash; flat single-lot</h4>' + delta_chart("flat", sym) + delta_table("flat", sym) for sym in FUNDS)}
  {"".join(f'<h4>{sym} &mdash; reinvesting</h4>' + delta_chart("reinvest", sym) + delta_table("reinvest", sym) for sym in FUNDS)}
</section>"""

def gate_table(sym: str) -> str:
    rows_d = GATECMP[sym]["rows"]
    order = ["Buy & hold", "Gated (breach)", "Gated (forward return)"]
    rows = []
    for label in order:
        r = rows_d[label]
        cls = " class=\"hl\"" if label == "Gated (forward return)" else ""
        rows.append(f"""
        <tr{cls}>
          <td>{label}</td>
          <td class="num">{FMT_PCT(r['cagr'])}</td>
          <td class="num">{r['sharpe']:.2f}</td>
          <td class="num">{FMT_PCT0(r['max_drawdown'])}</td>
          <td class="num">{r['n_calls']}</td>
          <td class="num">{r['n_called_away']}</td>
        </tr>""")
    return f"""
    <div class="table-wrap">
    <table>
      <thead><tr><th>Strategy</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th>
      <th>Calls</th><th>Called away</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    </div>"""


def gate_chart(sym: str) -> str:
    return f"""
    <figure class="chart">
      <img src="data:image/png;base64,{GATECMP[sym]['chart']}" alt="{sym} gate comparison" />
      <figcaption>Flat single-lot, 0.25&Delta;.</figcaption>
    </figure>"""


GATE_SECTION = f"""
<section id="gatecompare">
  <h2>Head to head: breach-probability gate vs. forward-return gate</h2>
  <p class="lede">Sector correlation, the absorption-ratio shift, and each
  fund's own vol-index level all look weak on <i>breach probability</i> (the
  target Rule 1 is built on) but strong &mdash; consistently significant, IC
  +0.10 to +0.18, all three funds &mdash; on <i>plain forward return</i>.
  That's not a contradiction: a signal can tell you a month will be good
  without telling you whether that goodness will specifically pierce a given
  strike. Rule 2 (defined above) acts on that second question instead &mdash;
  skip weeks that are simply likely to be <i>good</i>, rather than weeks
  specifically at risk of a breach. Here's how the two rules actually compare
  on the backtest.</p>

  <div class="callout">
    <p><b>The two rules split by ticker, not uniformly.</b> Breach wins on
    SPY (14.3% vs 13.0%), is a hair ahead on QQQ (18.4% vs 17.9% at
    0.25&Delta; &mdash; though forward-return overtakes it at 0.15&Delta;,
    see <a href="#delta">delta sensitivity</a>), and <b>loses clearly on IWM</b>
    (8.55% vs 9.32%, and lower Sharpe too: 0.38 vs 0.42). The more interesting
    difference is <i>when</i> each rule stays out of the market. On QQQ, the
    forward-return gate went completely silent from <b>2020-01-16 to
    2021-04-05</b> &mdash; 15 months, correctly reading the whole COVID
    crash-and-recovery as a persistent "stay out" regime &mdash; and resumed
    only once QQQ had already re-rated to $345+. The breach gate, by
    contrast, flips back to "write" as soon as RSI/trend recover, <i>even
    mid-rally</i>, which is exactly when a QQQ call written 2020-04-14 got
    caught and assigned well below where the stock kept running (see the
    <a href="#qqq">QQQ section</a>). The forward-return gate's signals move
    slower and persist through a whole regime; the breach gate's signals
    (RSI, 84-day trend) recover faster and re-open the door earlier &mdash;
    a real difference in behavior, even where the two rules land close
    together on CAGR.</p>
  </div>

  {"".join(f'<h3>{sym}</h3>' + gate_chart(sym) + gate_table(sym) for sym in FUNDS)}
</section>"""

CAVEATS = """
<section id="caveats">
  <h2>Caveats</h2>
  <ul>
    <li><b>The rules are derived, not pre-registered.</b> Thresholds (RSI 40,
    trend 0%, sector-corr top tercile) were chosen after seeing which legs
    cleared a 90% CI in this same sample the backtest runs on &mdash; there is
    no held-out validation/holdout split here, unlike this project's TG-CER
    research, which froze a spec before looking at validation or holdout
    data. Treat this as a well-evidenced first pass, not a validated
    strategy.</li>
    <li><b>The account model is strict cash, single-lot, all-or-nothing.</b>
    A real investor might re-enter a partial position with margin, or sell a
    cash-secured put to re-enter lower (the wheel pattern already used
    elsewhere in this codebase). None of that is modeled here.</li>
    <li><b>One configuration, one path.</b> One DTE band (25&ndash;45), one
    1.2&sigma; breach multiplier, one historical path (2018&ndash;2026). No
    parameter-neighborhood or stress-fill robustness sweep has been run yet
    for this strategy family.</li>
    <li><b>The forward-return gate was only tested on the flat single-lot
    structure</b> &mdash; not yet run through the reinvesting variant.</li>
    <li><b>The delta sweep is two points (0.25, 0.15), not a real sweep.</b>
    Most ticker/strategy combinations land close together across the two, but
    the forward-return gate on QQQ moved a full 0.76 CAGR points between them
    (see <a href="#delta">delta sensitivity</a>) &mdash; two points can't rule
    out a cliff or a smooth trend somewhere else in the range; a real
    neighborhood sweep is the honest next step before trusting any single
    delta choice.</li>
    <li><b>Some Rule 2 legs are weaker evidence than others.</b> RSI
    lower-tercile and sector-correlation upper-tercile clear a bucket-level
    90% CI on all three funds; the 4-month-trend and vol-index legs (and, for
    IWM, price-vs-MA200) are real on the continuous rank IC but the specific
    cutoff used here doesn't clear the same bucket-level bar for every fund
    (see <a href="#sensitivity">threshold sensitivity</a>). They're kept in
    the rule because the underlying correlation is real, not because the
    binary split itself was independently confirmed.</li>
    <li><b>The reinvesting variant's 100-share buffer size wasn't tuned or
    swept</b> &mdash; it's a round number, not a result of testing
    50/100/150-share buffers against each other. Nor was "always exactly 1
    covered lot, buffer grows unboundedly" compared against letting covered
    lots scale up as the position compounds.</li>
    <li><b>The three forward-return signals (sector correlation, absorption
    shift, vol index) are meaningfully correlated with each other</b> &mdash;
    vol spikes, correlated markets, and rising absorption tend to happen
    together, a classic stress-event signature &mdash; so three "confirming"
    legs are closer to one strong signal wearing three names than three
    independent pieces of evidence.</li>
  </ul>
</section>"""

STYLE = """
:root {
  --bg: #f4f2ec;
  --surface: #ffffff;
  --ink: #201d1a;
  --muted: #6d675e;
  --line: rgba(32,29,26,.12);
  --accent: #1f7a6c;
  --accent-ink: #0f4038;
  --warn: #b3492f;
  --pos: #1f7a6c;
  --neg: #b3492f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16181a;
    --surface: #1e2124;
    --ink: #ece8e1;
    --muted: #9a958c;
    --line: rgba(236,232,225,.12);
    --accent: #4fb3a0;
    --accent-ink: #8fd8c9;
    --warn: #e0806a;
    --pos: #4fb3a0;
    --neg: #e0806a;
  }
}
:root[data-theme="dark"] {
  --bg: #16181a; --surface: #1e2124; --ink: #ece8e1; --muted: #9a958c;
  --line: rgba(236,232,225,.12); --accent: #4fb3a0; --accent-ink: #8fd8c9;
  --warn: #e0806a; --pos: #4fb3a0; --neg: #e0806a;
}
:root[data-theme="light"] {
  --bg: #f4f2ec; --surface: #ffffff; --ink: #201d1a; --muted: #6d675e;
  --line: rgba(32,29,26,.12); --accent: #1f7a6c; --accent-ink: #0f4038;
  --warn: #b3492f; --pos: #1f7a6c; --neg: #b3492f;
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background: var(--bg); color: var(--ink);
  font-family: -apple-system, "Segoe UI", ui-sans-serif, system-ui, sans-serif;
  line-height: 1.6;
}
.wrap { max-width: 960px; margin: 0 auto; padding: 0 24px 96px; }
header.masthead {
  padding: 64px 24px 40px; text-align: left; border-bottom: 1px solid var(--line);
  margin-bottom: 8px;
}
header.masthead .inner { max-width: 960px; margin: 0 auto; }
header.masthead .eyebrow {
  text-transform: uppercase; letter-spacing: .12em; font-size: .72rem;
  color: var(--accent-ink); font-weight: 600; margin-bottom: 14px;
}
h1 {
  font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
  font-size: clamp(2rem, 4vw, 2.9rem); margin: 0 0 14px; text-wrap: balance;
  letter-spacing: -0.01em;
}
header.masthead p.dek {
  max-width: 66ch; color: var(--muted); font-size: 1.08rem; margin: 0;
}
nav.tabs {
  position: sticky; top: 0; z-index: 10; background: var(--bg);
  border-bottom: 1px solid var(--line); backdrop-filter: blur(6px);
}
nav.tabs .inner {
  max-width: 960px; margin: 0 auto; padding: 0 24px; display: flex; gap: 4px;
  overflow-x: auto;
}
nav.tabs a {
  color: var(--muted); text-decoration: none; padding: 14px 14px 12px;
  font-size: .88rem; font-weight: 600; border-bottom: 2px solid transparent;
  white-space: nowrap;
}
nav.tabs a:hover, nav.tabs a:focus-visible { color: var(--ink); }
section { padding: 48px 0 16px; border-bottom: 1px solid var(--line); }
section:last-of-type { border-bottom: none; }
h2 { font-family: Georgia, "Iowan Old Style", serif; font-size: 1.9rem; margin: 0 0 4px; }
h2 .tick {
  display: inline-block; font-family: ui-monospace, "SF Mono", Menlo, monospace;
  background: var(--surface); border: 1px solid var(--line); border-radius: 6px;
  padding: 2px 10px; font-size: 1.5rem; letter-spacing: .02em;
}
h3 { font-size: 1.15rem; margin: 40px 0 10px; }
h4 { font-size: .95rem; margin: 24px 0 4px; color: var(--muted); font-weight: 700; }
h5 { font-size: .82rem; margin: 16px 0 2px; color: var(--muted); font-weight: 600; }
p, li { max-width: 68ch; font-size: 1rem; }
p.lede { font-size: 1.18rem; max-width: 70ch; color: var(--ink); }
p.fig-note { color: var(--muted); font-size: .82rem; max-width: 68ch; margin-top: 6px; }
ul, ol { padding-left: 1.3em; }
a { color: var(--accent-ink); }
code {
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .88em;
  background: var(--surface); border: 1px solid var(--line); border-radius: 4px;
  padding: .05em .35em;
}
.muted { color: var(--muted); }
.callout {
  background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--accent);
  border-radius: 8px; padding: 18px 22px; margin: 28px 0; max-width: 760px;
}
.callout p { margin: 0; max-width: none; }
ul.verdicts { list-style: none; padding: 0; display: grid; gap: 10px; max-width: 760px; }
ul.verdicts li {
  background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
  padding: 12px 16px; max-width: none;
}
figure.chart { margin: 24px 0 8px; }
figure.chart img { width: 100%; height: auto; display: block; }
figure.chart figcaption { color: var(--muted); font-size: .82rem; margin-top: 6px; }
figure.dd { margin-top: -4px; }
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 8px; margin: 12px 0; }
figure.chart.small { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 6px; }
.table-wrap { overflow-x: auto; margin: 14px 0 8px; }
table { border-collapse: collapse; width: 100%; font-size: .92rem; min-width: 420px; }
table.yearly { min-width: 760px; }
table.heatmap { min-width: 380px; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 600; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }
td.num, th.num { font-variant-numeric: tabular-nums; text-align: right; }
td.icdud { color: var(--muted); text-align: right; }
tr.hl td:first-child { color: var(--accent-ink); font-weight: 600; }
td.neg { color: var(--neg); }

.forest { max-width: 760px; margin: 8px 0; }
.forest-row { display: grid; grid-template-columns: 160px 1fr 110px; align-items: center; gap: 10px; padding: 5px 0; }
.forest-axis { color: var(--muted); font-size: .74rem; }
.forest-axis .forest-track { display: flex; }
.forest-label { font-size: .86rem; }
.forest-track { position: relative; height: 16px; background: var(--surface); border: 1px solid var(--line); border-radius: 4px; }
.forest-zero { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--muted); opacity: .5; }
.forest-bar { position: absolute; top: 6px; height: 4px; border-radius: 2px; opacity: .8; }
.forest-pt { position: absolute; top: 3px; width: 10px; height: 10px; margin-left: -5px; border-radius: 50%; }
.forest-val { font-size: .82rem; text-align: right; }

.rule-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin: 14px 0; }
.rule-card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; }
.rule-card-head { display: flex; align-items: baseline; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }
.badge { margin-left: auto; font-size: .72rem; font-weight: 700; letter-spacing: .03em; padding: 3px 9px; border-radius: 999px; }
.badge.enter { background: rgba(31,122,108,.15); color: var(--accent-ink); }
.badge.veto { background: rgba(179,73,47,.15); color: var(--warn); }
.leg { display: grid; grid-template-columns: 18px 1fr auto; align-items: baseline; gap: 8px; padding: 4px 0; font-size: .86rem; }
.leg.confidence { border-top: 1px dashed var(--line); margin-top: 4px; padding-top: 8px; opacity: .82; }
.leg-check { text-align: center; }
.leg-val { font-weight: 600; }
.leg-rule { grid-column: 2 / span 2; font-size: .74rem; margin-top: -2px; }
.rule-note { margin-top: 10px; font-size: .78rem; }

footer { padding: 40px 0 0; color: var(--muted); font-size: .84rem; }
"""

HTML = f"""<title>Rip-Risk Gated Covered Calls — SPY, QQQ, IWM</title>
<style>{STYLE}</style>
<header class="masthead">
  <div class="inner">
    <div class="eyebrow">Level-2 research &middot; new work, 2026-09</div>
    <h1>Does knowing when <em>not</em> to sell the call save the covered call?</h1>
    <p class="dek">A regime-signal study (rank IC, 90% bootstrap CIs, breach
    probabilities) in the house style of this project's earlier research
    notes, redone against the target a covered-call writer actually cares
    about &mdash; then a derived per-ticker veto rule, backtested against a
    naive monthly covered call and plain buy-and-hold on 100-share lots of
    SPY, QQQ, and IWM using real 2018&ndash;2026 options-chain data.</p>
  </div>
</header>
<nav class="tabs">
  <div class="inner">
    <a href="#overview">Overview</a>
    <a href="#research">Signal research</a>
    <a href="#rule">Veto rules</a>
    <a href="#spy">SPY</a>
    <a href="#qqq">QQQ</a>
    <a href="#iwm">IWM</a>
    <a href="#delta">Delta sensitivity</a>
    <a href="#gatecompare">Breach vs forward-return</a>
    <a href="#reinvest">Reinvesting variant</a>
    <a href="#caveats">Caveats</a>
  </div>
</nav>
<div class="wrap">
{OVERVIEW}
{section("SPY", SPY_NARRATIVE)}
{section("QQQ", QQQ_NARRATIVE)}
{section("IWM", IWM_NARRATIVE)}
{DELTA_SECTION}
{GATE_SECTION}
{REINVEST_OVERVIEW}
{reinvest_section("SPY", SPY_REINVEST_NARRATIVE)}
{reinvest_section("QQQ", QQQ_REINVEST_NARRATIVE)}
{reinvest_section("IWM", IWM_REINVEST_NARRATIVE)}
{CAVEATS}
<footer>Data: licensed daily options-chain + daily price/vol-index history,
SPY/QQQ/IWM, 2005-01&ndash;2026-07 (signal study) and 2018-08&ndash;2026-07
(options backtest) &middot; Signal research:
<code>scripts/covered_call_signal_research.py</code> &middot; Engine:
<code>src/level2_research/engine.py</code> &middot; Strategy:
<code>CoveredCallStrategy</code> in <code>src/level2_research/strategies.py</code>
&middot; Backtest runner: <code>scripts/covered_call_backtest.py</code> &middot;
Scenario: base fills.</footer>
</div>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
print("wrote", OUT, len(HTML), "bytes")
