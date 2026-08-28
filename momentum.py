"""
momentum.py — long-horizon cross-sectional momentum, tested honestly.

A DIFFERENT HYPOTHESIS, NOT A REARRANGEMENT OF THE OLD ONE
Everything before this asked: does a chart pattern predict the next 20 days?
Answer, ten ways: no.

This asks something else. Rank every stock by its return over the past 12
months, skipping the most recent month. Buy the top slice. Hold for months,
not weeks. Rebalance. Repeat.

Why it is worth a separate test:
  * Documented since Jegadeesh & Titman (1993) and replicated across decades,
    countries and asset classes. Unlike chart patterns, it has survived
    out-of-sample scrutiny by people trying hard to kill it.
  * It is CROSS-SECTIONAL — a stock's rank relative to others, not a pattern
    in its own history. Different information.
  * Few trades. Your gross edge was +1.7% against 2.2% costs; a strategy that
    rebalances quarterly pays a fraction of what one trading every 20 days does.
    Cost was the thing killing you, and this attacks it directly.

The skipped month is not arbitrary. Short-term reversal runs the other way —
you measured it yourself at 28 standard errors in the 2010-2019 holdout — so
including the most recent month contaminates the signal with its opposite.

THE CONTROL
Ranking is replaced with random selection, many times, holding everything else
identical. p = fraction of random-selection runs that match or beat momentum.
Also compares against equal-weight buy-and-hold of the same universe, which is
the benchmark that actually matters.

    python momentum.py --top-pct 10 --hold 3 --perms 300
    python momentum.py --top-pct 20 --hold 6 --lookback 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import _load_frames

HERE = Path(__file__).parent


def build_panel(frames, min_price, min_turnover):
    """Monthly close panel plus an eligibility mask."""
    closes, elig = {}, {}
    for t, d in frames.items():
        if len(d) < 300:
            continue
        m = d["Close"].resample("ME").last()
        if m.notna().sum() < 24:
            continue
        turn = (d["Close"] * d["Volume"]).rolling(20).mean().resample("ME").last()
        closes[t] = m
        elig[t] = (m >= min_price) & (turn >= min_turnover)
    if not closes:
        raise SystemExit("No usable tickers.")
    px = pd.DataFrame(closes).sort_index()
    ok = pd.DataFrame(elig).reindex_like(px).fillna(False)
    return px, ok


def run(px, ok, lookback, skip, top_pct, hold, cost_pct, rng=None,
        random_pick=False, max_names=0):
    """Monthly rebalance into overlapping `hold`-month sleeves."""
    idx = px.index
    rets = px.pct_change()
    # momentum: return from t-lookback to t-skip
    mom = px.shift(skip) / px.shift(lookback) - 1

    sleeves = []          # each: (exit_month_index, list of tickers, weight)
    equity = 1.0
    curve = []
    n_trades = 0

    start = lookback + 1
    for i in range(start, len(idx) - 1):
        # this month's return on all open sleeves
        month_ret = 0.0
        live = [s for s in sleeves if s[0] > i]
        if live:
            w = 1.0 / len(live)
            for _, names, _ in live:
                r = rets.iloc[i + 1][names].dropna()
                if len(r):
                    month_ret += w * float(r.mean())
        equity *= (1 + month_ret)
        curve.append((idx[i], equity))
        sleeves = live

        # open a new sleeve if there is room
        if len(sleeves) < hold:
            cand = mom.iloc[i][ok.iloc[i]].dropna()
            if len(cand) >= 20:
                k = max(5, int(len(cand) * top_pct / 100))
                if max_names > 0:
                    k = min(k, max_names)
                if random_pick:
                    names = list(rng.choice(cand.index, size=k, replace=False))
                else:
                    names = list(cand.nlargest(k).index)
                sleeves.append((i + hold, names, 1.0))
                n_trades += k
                equity *= (1 - cost_pct / 100 / hold)   # cost amortised

    if not curve:
        return None
    c = pd.DataFrame(curve, columns=["date", "eq"]).set_index("date")
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    cagr = (c["eq"].iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
    dd = ((c["eq"] - c["eq"].cummax()) / c["eq"].cummax() * 100).min()
    mr = c["eq"].pct_change().dropna()
    sharpe = mr.mean() / mr.std() * np.sqrt(12) if mr.std() > 0 else 0.0
    return {"cagr": float(cagr), "max_dd": float(dd), "sharpe": float(sharpe),
            "final": float(c["eq"].iloc[-1]), "years": float(yrs),
            "trades": n_trades}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=12, help="months")
    ap.add_argument("--skip", type=int, default=1, help="months skipped (reversal)")
    ap.add_argument("--top-pct", type=float, default=10.0)
    ap.add_argument("--hold", type=int, default=3, help="months held")
    ap.add_argument("--cost", type=float, default=0.20)
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    ap.add_argument("--cache-tag", default="",
                    help="alternate cache, e.g. _holdout for 2010-2019")
    ap.add_argument("--max-names", type=int, default=0,
                    help="cap positions per sleeve; 0 = no cap")
    a = ap.parse_args()

    if a.market == "us":
        from params_us import GATES as G
    else:
        from params import GATES as G

    if a.cache_tag:
        import data_us
        files = sorted(data_us.CACHE_DIR.glob(f"us{a.cache_tag}_*.parquet"))
        if not files:
            raise SystemExit(f"No cache us{a.cache_tag}_*.parquet")
        print(f"  reading {files[-1].name}")
        frames = data_us._unflatten(pd.read_parquet(files[-1]))
    else:
        frames = _load_frames(a.market)
    print(f"  building monthly panel from {len(frames):,} tickers...")
    px, ok = build_panel(frames, G.min_price_sgd, G.min_avg_turnover_sgd)
    print(f"  {px.shape[1]:,} tickers x {px.shape[0]} months "
          f"({px.index.min().date()} to {px.index.max().date()})")
    print(f"  {a.lookback}-month lookback, skip {a.skip}, "
          f"top {a.top_pct:g}%, hold {a.hold} months\n")

    real = run(px, ok, a.lookback, a.skip, a.top_pct, a.hold, a.cost,
               max_names=a.max_names)
    if real is None:
        raise SystemExit("Not enough history.")

    # equal-weight buy and hold of the same eligible universe
    rets = px.pct_change()
    ew = []
    eq = 1.0
    for i in range(a.lookback + 1, len(px.index) - 1):
        r = rets.iloc[i + 1][ok.iloc[i]].dropna()
        eq *= (1 + (float(r.mean()) if len(r) else 0.0))
        ew.append(eq)
    ew_years = real["years"]
    ew_cagr = (ew[-1] ** (1 / ew_years) - 1) * 100 if ew else 0.0

    print(f"  {'strategy':<26}{'CAGR':>9}{'max DD':>10}{'Sharpe':>9}")
    print("  " + "-" * 56)
    print(f"  {'momentum, top ' + str(int(a.top_pct)) + '%':<26}"
          f"{real['cagr']:>8.1f}%{real['max_dd']:>9.1f}%{real['sharpe']:>9.2f}")
    print(f"  {'equal-weight universe':<26}{ew_cagr:>8.1f}%{'—':>10}{'—':>9}")

    print(f"\n  running {a.perms} random-selection controls...")
    rng = np.random.default_rng(0)
    null = []
    for p in range(a.perms):
        r = run(px, ok, a.lookback, a.skip, a.top_pct, a.hold, a.cost,
                rng=np.random.default_rng(1000 + p), random_pick=True,
                max_names=a.max_names)
        if r:
            null.append(r["cagr"])
        if (p + 1) % 100 == 0:
            print(f"    {p + 1}/{a.perms}", end="\r")
    print(" " * 24, end="\r")

    null = np.asarray(null)
    beat = int((null >= real["cagr"]).sum())
    pval = (beat + 1) / (len(null) + 1)
    print(f"\n  random selection: median {np.median(null):.1f}%  "
          f"5-95% {np.percentile(null,5):.1f}% to {np.percentile(null,95):.1f}%")
    print(f"  random runs matching or beating momentum: {beat}/{len(null)}")
    print(f"\n  p = {pval:.4f}")

    if pval < 0.05 and real["cagr"] > ew_cagr:
        print("  -> Momentum beat BOTH random selection and buy-and-hold.")
        print("     This is the first thing in this project to do that.")
    elif pval < 0.05:
        print("  -> Beat random selection but not equal-weight buy-and-hold.")
        print("     The ranking does something, but not enough to justify trading.")
    else:
        print("  -> Consistent with random selection. No edge here either.")

    per_sleeve = real["trades"] / max(1, (len(px.index) - a.lookback - 2))
    print(f"\n  {real['trades']:,} position entries over {real['years']:.1f} years, "
          f"~{per_sleeve:.0f} names per rebalance.")
    if per_sleeve > 50:
        print(f"  WARNING: {per_sleeve:.0f} positions is not tradeable at retail size.")
        print(f"  Re-run with --max-names 30 to see if the edge survives")
        print(f"  concentration. If it does not, it lived in the long tail.")
    (HERE / "momentum.json").write_text(json.dumps(
        {"real": real, "ew_cagr": ew_cagr, "p": pval}, indent=2, default=float))
    print("  Saved: momentum.json")


if __name__ == "__main__":
    main()