"""
report.py — renders the scan into a single self-contained HTML file.

Design brief: an end-of-day tape sheet, read once after the SGX close. The
thing you should be able to read in two seconds is WHY a name did or did not
qualify, so the signature element is the gate rail — eight fixed segments per
row, always in the same order, so the shape of a near-miss is recognisable
without reading any text.

No build step, no CDN dependency for anything structural. Fonts load from
Google if online and fall back to local stacks if not.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from params import FINGERPRINT, describe
from rules import GATE_NAMES

# --- tokens ---------------------------------------------------------------
CSS = """
:root{
  --ink:#0B1220; --panel:#131E32; --panel-2:#18253C; --rule:#24344F;
  --text:#E2E9F4; --muted:#7D8DA8; --dim:#4A5C78;
  --pass:#4FB286; --fail:#B4485C; --amber:#E0A33E;
  --mono:"IBM Plex Mono",ui-monospace,Consolas,"SF Mono",monospace;
  --disp:"Archivo","Archivo Narrow",-apple-system,"Segoe UI",sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--ink);color:var(--text);font-family:var(--mono);
  font-size:13px;line-height:1.5;padding:28px 20px 64px;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1100px;margin:0 auto}

/* masthead */
.mast{display:flex;justify-content:space-between;align-items:flex-end;
  border-bottom:2px solid var(--text);padding-bottom:10px;margin-bottom:6px;gap:16px;flex-wrap:wrap}
.mast h1{font-family:var(--disp);font-size:30px;font-weight:700;letter-spacing:-.02em;
  text-transform:uppercase}
.mast .sub{color:var(--muted);font-size:11px;letter-spacing:.14em;text-transform:uppercase}
.stamp{text-align:right;font-size:11px;color:var(--muted);white-space:nowrap}
.stamp b{color:var(--text);display:block;font-size:15px}
.fp{border-top:1px solid var(--rule);margin-bottom:28px;padding-top:6px;
  font-size:10px;color:var(--dim);letter-spacing:.08em}

/* counters */
.counts{display:flex;gap:0;border:1px solid var(--rule);margin-bottom:30px}
.counts div{flex:1;padding:14px 16px;border-right:1px solid var(--rule)}
.counts div:last-child{border-right:0}
.counts .n{font-family:var(--disp);font-size:26px;font-weight:700;line-height:1}
.counts .l{font-size:10px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-top:5px}
.counts .hit .n{color:var(--amber)}

h2{font-family:var(--disp);font-size:12px;letter-spacing:.18em;text-transform:uppercase;
  color:var(--muted);margin:34px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--rule)}

/* card */
.card{border:1px solid var(--rule);background:var(--panel);margin-bottom:12px}
.card.hit{border-color:var(--amber);border-left-width:3px}
.card-top{display:flex;justify-content:space-between;align-items:center;
  padding:12px 16px;gap:14px;flex-wrap:wrap}
.tick{font-family:var(--disp);font-size:21px;font-weight:700;letter-spacing:-.01em}
.tick span{font-family:var(--mono);font-size:11px;color:var(--muted);font-weight:400;
  margin-left:9px;letter-spacing:.04em}
.px{text-align:right;font-size:16px}
.px small{display:block;font-size:10px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase}

/* SIGNATURE: gate rail */
.rail{display:grid;grid-template-columns:repeat(8,1fr);border-top:1px solid var(--rule)}
.g{padding:8px 4px;text-align:center;border-right:1px solid var(--rule);position:relative}
.g:last-child{border-right:0}
.g .bar{height:3px;background:var(--dim);margin-bottom:6px}
.g.p .bar{background:var(--pass)}
.g.f .bar{background:var(--fail)}
.g .nm{font-size:9px;letter-spacing:.08em;color:var(--muted)}
.g.p .nm{color:var(--text)}
.g.f .nm{color:var(--fail)}

/* detail */
.detail{border-top:1px solid var(--rule);padding:11px 16px;font-size:11px;color:var(--muted);
  display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:3px 22px}
.detail b{color:var(--text);font-weight:400}
.detail .x{color:var(--fail)}
.levels{border-top:1px solid var(--rule);background:var(--panel-2);padding:11px 16px;
  font-size:11px;display:flex;gap:26px;flex-wrap:wrap}
.levels i{font-style:normal;color:var(--muted)}
.spark{padding:0 16px 12px}

table{width:100%;border-collapse:collapse;font-size:11px}
th{text-align:left;color:var(--muted);font-weight:400;font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;padding:7px 10px;border-bottom:1px solid var(--rule)}
td{padding:6px 10px;border-bottom:1px solid var(--panel-2)}
td.num{text-align:right}
.pos{color:var(--pass)} .neg{color:var(--fail)}

.note{border-left:2px solid var(--dim);padding:11px 15px;margin:18px 0;
  color:var(--muted);font-size:11px;line-height:1.7}
.empty{border:1px dashed var(--rule);padding:40px 20px;text-align:center;color:var(--muted)}
.empty b{display:block;color:var(--text);font-family:var(--disp);font-size:16px;margin-bottom:6px}
footer{margin-top:44px;padding-top:14px;border-top:1px solid var(--rule);
  color:var(--dim);font-size:10px;line-height:1.8}
@media (max-width:640px){
  .rail{grid-template-columns:repeat(4,1fr)}
  .g{border-bottom:1px solid var(--rule)}
  .counts{flex-wrap:wrap}.counts div{min-width:50%}
}
@media (prefers-reduced-motion:no-preference){
  .card{animation:in .32s ease both}
  @keyframes in{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
}
"""


def _sparkline(df: pd.DataFrame, bars: int = 60) -> str:
    """Inline SVG of recent closes with the EMA21 overlaid."""
    d = df.iloc[-bars:]
    c = d["Close"].astype(float).tolist()
    e = d["ema_mid"].astype(float).tolist()
    vals = [v for v in c + e if pd.notna(v)]
    if len(vals) < 4:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    w, h = 1000.0, 60.0

    def path(series):
        pts = []
        for i, v in enumerate(series):
            if pd.isna(v):
                continue
            x = i / (len(series) - 1) * w
            y = h - (v - lo) / rng * h
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    return (
        f'<div class="spark"><svg viewBox="0 0 {w:.0f} {h:.0f}" preserveAspectRatio="none" '
        f'style="width:100%;height:52px" role="img" aria-label="60-day close with EMA21">'
        f'<polyline fill="none" stroke="#4A5C78" stroke-width="1.5" stroke-dasharray="3 3" points="{path(e)}"/>'
        f'<polyline fill="none" stroke="#E2E9F4" stroke-width="2" points="{path(c)}"/>'
        f'</svg></div>'
    )


def _card(item: dict) -> str:
    esc = html.escape
    hit = item["fired"]
    rail = "".join(
        f'<div class="g {"p" if g["passed"] else "f"}">'
        f'<div class="bar"></div><div class="nm">{esc(g["name"])}</div></div>'
        for g in item["gates"]
    )
    detail = "".join(
        f'<div class="{"x" if not g["passed"] else ""}">'
        f'{esc(g["name"])} &middot; <b>{esc(g["detail"])}</b></div>'
        for g in item["gates"]
    )
    lv = item.get("levels", {})
    levels = ""
    if lv:
        parts = [
            f'<span><i>20d range</i> {lv["range_20_low"]:.3f} – {lv["range_20_high"]:.3f}</span>',
            f'<span><i>stop ref</i> {lv["stop_ref"]:.3f}</span>',
            f'<span><i>ATR</i> {lv["atr_pct"]:.1f}%</span>',
            f'<span><i>upper band</i> {lv["bb_upper"]:.3f}</span>',
            f'<span><i>60d high</i> {lv["swing_high_60"]:.3f}</span>',
        ]
        if "base_rate_20d_win" in lv:
            parts.append(
                f'<span><i>base rate 20d</i> {lv["base_rate_20d_win"]:.0f}% up, '
                f'median {lv["base_rate_20d_median"]:+.1f}% '
                f'(n={lv["base_rate_sample"]})</span>'
            )
        levels = f'<div class="levels">{"".join(parts)}</div>'

    return f"""<article class="card {'hit' if hit else ''}">
<div class="card-top">
  <div class="tick">{esc(item['ticker'])}<span>{esc(item.get('name',''))}</span></div>
  <div class="px">{item['close']:.3f}<small>{item['score']}/8 gates</small></div>
</div>
<div class="rail">{rail}</div>
{_sparkline(item['frame']) if item.get('frame') is not None else ''}
<div class="detail">{detail}</div>
{levels}
</article>"""


def _baserate_table(summary: dict) -> str:
    if not summary or summary.get("n", 0) == 0:
        return ('<div class="note">No historical signals in the tested window, so there '
                'is no base rate to report. Treat any live alert as untested.</div>')
    rows = ""
    for h, s in summary.get("horizons", {}).items():
        cls = "pos" if s["median"] > 0 else "neg"
        rows += (f"<tr><td>{h} days</td>"
                 f"<td class='num'>{s['win_rate']:.0f}%</td>"
                 f"<td class='num {cls}'>{s['median']:+.2f}%</td>"
                 f"<td class='num'>{s['mean']:+.2f}%</td>"
                 f"<td class='num'>{s['p25']:+.2f}%</td>"
                 f"<td class='num'>{s['p75']:+.2f}%</td>"
                 f"<td class='num'>{s['worst']:+.2f}%</td></tr>")
    yrs = "".join(
        f"<tr><td>{y}</td><td class='num'>{d['n']}</td>"
        f"<td class='num {'pos' if d['median']>0 else 'neg'}'>{d['median']:+.2f}%</td></tr>"
        for y, d in sorted(summary.get("by_year", {}).items()))
    return f"""
<table><thead><tr><th>Horizon</th><th class="num">Up</th><th class="num">Median</th>
<th class="num">Mean</th><th class="num">25th</th><th class="num">75th</th>
<th class="num">Worst</th></tr></thead><tbody>{rows}</tbody></table>
<div class="note">
Sample: {summary['n']} signals across {summary['names']} names,
{summary['first']} to {summary['last']}.
Median best-case move within 20 days was {summary['mfe_20_median']:+.2f}%; median worst
drawdown before that was {summary['mae_20_median']:+.2f}% — that second number is the one
that decides your position size.
Entry is modelled at the next open after the signal bar closes, never at the signal close.
</div>
<h2>Median 20-day result by year</h2>
<table><thead><tr><th>Year</th><th class="num">Signals</th>
<th class="num">Median 20d</th></tr></thead><tbody>{yrs}</tbody></table>
<div class="note">If one year carries the whole result, the edge is a regime artefact,
not a strategy.</div>"""


def render(hits: list[dict], near: list[dict], summary: dict,
           scanned: int, out_path: str | Path) -> Path:
    now = datetime.now()
    hits_html = "".join(_card(h) for h in hits) or (
        '<div class="empty"><b>No qualifying setups</b>'
        'Every name failed at least one gate. This is the normal outcome on most days.</div>')
    near_html = "".join(_card(n) for n in near[:12])

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SGX Signal Sheet — {now:%d %b %Y}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><div class="wrap">

<header class="mast">
  <div><h1>Signal Sheet</h1>
  <div class="sub">SGX daily &middot; locked rule set</div></div>
  <div class="stamp"><b>{now:%a %d %b %Y}</b>{now:%H:%M} SGT &middot; {scanned} counters scanned</div>
</header>
<div class="fp">RULE FINGERPRINT {FINGERPRINT} &middot; any change to params.py voids the base rates below</div>

<div class="counts">
  <div class="hit"><div class="n">{len(hits)}</div><div class="l">Alerts</div></div>
  <div><div class="n">{len(near)}</div><div class="l">7 of 8 gates</div></div>
  <div><div class="n">{scanned}</div><div class="l">Scanned</div></div>
  <div><div class="n">{summary.get('n',0)}</div><div class="l">Historical signals</div></div>
</div>

<h2>Alerts — all eight gates</h2>
{hits_html}

<h2>Near misses — one gate short</h2>
{near_html or '<div class="empty"><b>Nothing close</b>No name reached seven of eight gates.</div>'}

<h2>Base rates — what happened after past signals</h2>
{_baserate_table(summary)}

<footer>
Generated by the locked SGX screener. Levels shown are volatility-derived reference points,
not forecasts: an ATR band describes how far a stock typically travels, and says nothing
about direction. Base rates are historical frequencies, not probabilities for any single trade.
Yahoo Finance data may contain gaps on thin counters — confirm on POEMS before acting.<br>
This is a personal research tool. It is not financial advice.
</footer>
</div></body></html>"""

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return p
