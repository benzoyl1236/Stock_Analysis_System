"""
combo.py — do the two surviving rules work better together?

FROM basics.py, THE TWO THAT CLEARED BOTH DECADES:
    Close below lower BB       +0.81% vs random, p 0.001 / 0.001
    EMA 8 crosses above 21     +0.24% vs random, p 0.013 / 0.047

They may not combine the way you would hope. If price sits 2 standard
deviations BELOW its 20-day average, EMA8 is very unlikely to be crossing
ABOVE EMA21 on that same bar — the two conditions nearly exclude each other.
So this tests four readings, not one:

    A  BB lower alone                       (the baseline to beat)
    B  EMA cross alone
    C  BOTH on the same bar                 (probably rare)
    D  BB lower first, EMA cross within N days   (dip, then turn)
    E  EMA cross first, BB lower within N days   (turn, then dip)

D is the interesting one: buy the fall, wait for it to stop falling.

A WARNING ABOUT WHERE WE ARE
basics.py already ran 10 rules x 2 periods = 20 tests. This adds 5 more x 2 =
10. At a 5% threshold, roughly 1.5 false passes are expected across all 30.
A combination that beats its own components by a small margin is exactly what
a false pass looks like. The bar for believing this should be higher than for
the original rules, not lower — a combination must clearly beat rule A alone,
in BOTH decades, to be worth anything.

    python combo.py --both --top 600 --perms 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import indicators as ind
from basics import load, ptest, HOLD

HERE = Path(__file__).parent


def _cross_up(a, b):
    pa, pb = np.roll(a, 1), np.roll(b, 1)
    pa[0] = pb[0] = np.nan
    return (pa <= pb) & (a > b)


def _within(flags, window):
    """True where `flags` was True at any point in the last `window` bars."""
    s = pd.Series(flags).rolling(window, min_periods=1).max()
    return s.fillna(0).to_numpy().astype(bool)


def rules_for(d, window):
    c = d["Close"].to_numpy(float)
    e8, e21 = d["ema_fast"].to_numpy(float), d["ema_mid"].to_numpy(float)
    lo = d["bb_lower"].to_numpy(float)

    below = c < lo
    cross = _cross_up(e8, e21)
    below_recent = _within(below, window)
    cross_recent = _within(cross, window)

    return {
        "A  BB lower alone": below,
        "B  EMA 8x21 alone": cross,
        "C  both same bar": below & cross,
        f"D  BB lower then EMA x ({window}d)": cross & below_recent & ~below,
        f"E  EMA x then BB lower ({window}d)": below & cross_recent,
    }


def collect(built, G, cost, window):
    out = {}
    for t, df in built.items():
        c = df["Close"].to_numpy(float)
        o = df["Open"].to_numpy(float)
        n = len(c)
        elig = np.arange(150, n - HOLD - 1)
        if len(elig) < 40:
            continue
        entry, exitp = o[elig + 1], c[elig + 1 + HOLD]
        ok = (np.isfinite(entry) & (entry > 0) & np.isfinite(exitp)
              & (df["turnover_avg"].to_numpy(float)[elig] >= G.min_avg_turnover_sgd)
              & (c[elig] >= G.min_price_sgd))
        elig, entry, exitp = elig[ok], entry[ok], exitp[ok]
        if len(elig) < 40:
            continue
        pool = (exitp / entry - 1) * 100 - cost
        pos = {v: j for j, v in enumerate(elig)}

        for name, fires in rules_for(df, window).items():
            hits, last = [], -10**9
            for i in np.flatnonzero(fires):
                if i not in pos or i - last < 20:
                    continue
                hits.append(pos[i])
                last = i
            if not hits:
                continue
            e = out.setdefault(name, {"real": [], "pools": [], "counts": []})
            e["real"].extend(pool[hits])
            e["pools"].append(pool)
            e["counts"].append(len(hits))
    return out


def run_period(tag, market, top, perms, cost, G, window):
    frames = load(tag, market)
    liq = sorted(((t, float((d["Close"] * d["Volume"]).tail(250).median()))
                  for t, d in frames.items() if len(d) >= 400), key=lambda x: -x[1])
    built = {}
    for t in [x for x, _ in liq[:top]]:
        try:
            built[t] = ind.build(frames[t])
        except Exception:
            pass
    print(f"    {len(built):,} tickers")
    data = collect(built, G, cost, window)
    return {k: ptest(v, perms) for k, v in data.items() if len(v["real"]) >= 100}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=600)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--cost", type=float, default=0.20)
    ap.add_argument("--window", type=int, default=10, help="days linking the two")
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    ap.add_argument("--both", action="store_true")
    a = ap.parse_args()

    if a.market == "us":
        from params_us import GATES as G
    else:
        from params import GATES as G

    print("Combining the two rules that survived both decades")
    print(f"  {a.perms:,} permutations, {a.cost}% cost, {HOLD}-day hold, "
          f"{a.window}-day link window\n")

    print("  period 1: recent")
    r1 = run_period("", a.market, a.top, a.perms, a.cost, G, a.window)
    r2 = None
    if a.both:
        print("  period 2: holdout (2010-2019)")
        r2 = run_period("_holdout", a.market, a.top, a.perms, a.cost, G, a.window)

    hdr = f"\n  {'rule':<34}{'n':>8}{'vs random':>11}{'p':>8}"
    if r2:
        hdr += f"{'vs rand 2010s':>15}{'p 2010s':>10}{'both?':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))

    base = r1.get("A  BB lower alone", {}).get("edge", 0.0)
    base2 = (r2 or {}).get("A  BB lower alone", {}).get("edge", 0.0)
    rows = {}
    for name in sorted(r1):
        s = r1[name]
        line = f"  {name:<34}{s['n']:>8,}{s['edge']:>10.2f}%{s['p']:>8.3f}"
        if r2:
            s2 = r2.get(name)
            if s2:
                ok = (s["p"] < 0.05 and s2["p"] < 0.05
                      and np.sign(s["edge"]) == np.sign(s2["edge"]))
                line += f"{s2['edge']:>14.2f}%{s2['p']:>10.3f}{('YES' if ok else 'no'):>8}"
                rows[name] = (s, s2, ok)
            else:
                line += f"{'—':>15}{'—':>10}{'—':>8}"
        print(line)

    if r2 and rows:
        print(f"\n  Does any combination BEAT 'BB lower alone' in both decades?")
        print(f"  (BB lower alone: {base:+.2f}% recent, {base2:+.2f}% 2010s)")
        winners = []
        for name, (s, s2, ok) in rows.items():
            if name.startswith("A "):
                continue
            if ok and s["edge"] > base and s2["edge"] > base2:
                winners.append(name)
                print(f"    {name}: {s['edge']:+.2f}% / {s2['edge']:+.2f}%  BEATS IT")
        if not winners:
            print(f"    No. Combining does not improve on the single rule.")
            print(f"    That is the common outcome — extra conditions cut the")
            print(f"    sample without adding information.")

    print(f"\n  Reminder: this is 10 more tests on top of the 20 in basics.py.")
    print(f"  A marginal winner here is more likely noise than discovery.")
    (HERE / "combo.json").write_text(json.dumps({"p1": r1, "p2": r2},
                                                indent=2, default=float))
    print("  Saved: combo.json")


if __name__ == "__main__":
    main()
