"""
basics.py — one indicator at a time, tested properly, in two decades.

WHY START OVER HERE
Everything complicated failed. The eight gates, the options overlays, the
genetic search, the evolving colony, cross-sectional momentum. Before adding
anything back, the honest question is whether ANY single indicator does
anything at all on its own.

simple.py asked this before, but it had no p-values and only one period. This
version fixes both.

WHAT IT DOES
For each rule, one indicator, read the plain way:
  * find every signal
  * measure the forward return, entering at the NEXT open
  * permutation test: keep the signals, reassign which forward return attaches
    to each, 1,000 times, and see where the real result falls
  * repeat the whole thing on 2010-2019, a decade the rule was not chosen in

THE BAR, SET BEFORE RUNNING
A rule must clear p < 0.05 in BOTH periods AND have the same sign in both.
Ten rules across two periods is twenty tests; at a 5% threshold you would
expect one false pass by chance, so a single lucky period means nothing. Two
periods agreeing is roughly a 1-in-400 accident.

Nothing here is combined, stacked or weighted. That is the point — find out if
any single piece works before assembling pieces again.

    python basics.py                       2020-2026
    python basics.py --cache-tag _holdout  2010-2019
    python basics.py --both                run and compare both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import indicators as ind

HERE = Path(__file__).parent
HOLD = 20


def _cross_up(a, b):
    pa, pb = np.roll(a, 1), np.roll(b, 1)
    pa[0] = pb[0] = np.nan
    return (pa <= pb) & (a > b)


def rules_for(d):
    c = d["Close"].to_numpy(float)
    e8, e21, e100 = (d["ema_fast"].to_numpy(float), d["ema_mid"].to_numpy(float),
                     d["ema_slow"].to_numpy(float))
    k, dd = d["stoch_k"].to_numpy(float), d["stoch_d"].to_numpy(float)
    macd, sig = d["macd"].to_numpy(float), d["macd_signal"].to_numpy(float)
    mtm = d["mtm"].to_numpy(float)
    lo, up, mid = (d["bb_lower"].to_numpy(float), d["bb_upper"].to_numpy(float),
                   d["bb_mid"].to_numpy(float))
    z = np.zeros_like(macd)
    return {
        "EMA 8 crosses above 21": _cross_up(e8, e21),
        "Close crosses above EMA100": _cross_up(c, e100),
        "MACD crosses above signal": _cross_up(macd, sig),
        "MACD crosses above zero": _cross_up(macd, z),
        "Stoch %K crosses %D under 30": _cross_up(k, dd) & (k < 30),
        "Stoch %K rises above 30": _cross_up(k, np.full_like(k, 30.0)),
        "Close below lower BB": c < lo,
        "Close crosses above upper BB": _cross_up(c, up),
        "Close crosses above BB mid": _cross_up(c, mid),
        "Momentum 28 crosses above 0": _cross_up(mtm, z),
    }


def collect(built, G, cost):
    """Per rule: real returns, and the per-ticker pool for permuting."""
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

        for name, fires in rules_for(df).items():
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


def ptest(entry, perms, seed=0):
    real = np.asarray(entry["real"])
    rng = np.random.default_rng(seed)
    null = np.empty(perms)
    for i in range(perms):
        parts = [rng.choice(p, size=min(k, len(p)), replace=len(p) < k)
                 for p, k in zip(entry["pools"], entry["counts"])]
        null[i] = np.concatenate(parts).mean()
    rm = float(real.mean())
    beat = int((null >= rm).sum())
    return {"n": len(real), "mean": rm, "null": float(null.mean()),
            "p": (beat + 1) / (perms + 1),
            "edge": rm - float(null.mean())}


LOADED = []


def load(tag, market):
    import data_us, re as _re
    if tag:
        files = sorted(data_us.CACHE_DIR.glob(f"us{tag}_*.parquet"))
    else:
        files = sorted(f for f in data_us.CACHE_DIR.glob("us_*.parquet")
                       if _re.fullmatch(r"us_\d{4}-\d{2}-\d{2}\.parquet", f.name))
    if not files:
        raise SystemExit(f"No cache for tag '{tag or '(main)'}'")
    LOADED.append(files[-1].name)
    print(f"    reading {files[-1].name}")
    if len(LOADED) == 2 and LOADED[0] == LOADED[1]:
        raise SystemExit(
            "\n  ABORT: both periods resolved to the SAME cache file.\n"
            "  A two-period test on one file is not a test. Check cache_us/.")
    return data_us._unflatten(pd.read_parquet(files[-1]))


def run_period(tag, market, top, perms, cost, G):
    frames = load(tag, market)
    # guard: identical p-values across periods means the same file was read twice
    liq = sorted(((t, float((d["Close"] * d["Volume"]).tail(250).median()))
                  for t, d in frames.items() if len(d) >= 400), key=lambda x: -x[1])
    picks = [t for t, _ in liq[:top]]
    built = {}
    for t in picks:
        try:
            built[t] = ind.build(frames[t])
        except Exception:
            pass
    print(f"    {len(built):,} tickers")
    data = collect(built, G, cost)
    return {name: ptest(e, perms) for name, e in data.items() if len(e["real"]) >= 100}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=600)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--cost", type=float, default=0.20)
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    ap.add_argument("--cache-tag", default="")
    ap.add_argument("--both", action="store_true",
                    help="run 2020-2026 AND 2010-2019, require both to pass")
    a = ap.parse_args()

    if a.market == "us":
        from params_us import GATES as G
    else:
        from params import GATES as G

    print("One indicator at a time, permutation tested")
    print(f"  {a.perms:,} permutations, {a.cost}% cost, {HOLD}-day hold\n")

    print("  period 1: recent")
    r1 = run_period(a.cache_tag, a.market, a.top, a.perms, a.cost, G)
    r2 = None
    if a.both:
        print("  period 2: holdout (2010-2019)")
        r2 = run_period("_holdout", a.market, a.top, a.perms, a.cost, G)

    hdr = f"\n  {'rule':<32}{'n':>8}{'mean':>9}{'vs random':>11}{'p':>8}"
    if r2:
        hdr += f"{'p (2010s)':>11}{'passes?':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))

    passed = []
    for name in sorted(r1, key=lambda x: r1[x]["p"]):
        s = r1[name]
        line = (f"  {name:<32}{s['n']:>8,}{s['mean']:>8.2f}%"
                f"{s['edge']:>10.2f}%{s['p']:>8.3f}")
        if r2:
            s2 = r2.get(name)
            if s2:
                ok = (s["p"] < 0.05 and s2["p"] < 0.05
                      and np.sign(s["edge"]) == np.sign(s2["edge"]))
                line += f"{s2['p']:>11.3f}{('YES' if ok else 'no'):>9}"
                if ok:
                    passed.append(name)
            else:
                line += f"{'—':>11}{'—':>9}"
        print(line)

    print(f"\n  'vs random' is the rule's mean minus what random timing on the")
    print(f"  same tickers produced. Negative means the rule picks worse days.")

    if r2:
        if passed:
            print(f"\n  PASSED BOTH DECADES: {', '.join(passed)}")
            print(f"  Worth a closer look. Check the edge is bigger than costs.")
        else:
            print(f"\n  No single indicator cleared p<0.05 in both decades.")
            print(f"  With 10 rules x 2 periods you would expect ~1 false pass")
            print(f"  by chance, so finding zero is a clean negative.")
    else:
        print(f"\n  Single period only. Re-run with --both before believing any")
        print(f"  low p-value here — one period is where false positives live.")

    (HERE / "basics.json").write_text(json.dumps(
        {"period1": r1, "period2": r2}, indent=2, default=float))
    print("  Saved: basics.json")


if __name__ == "__main__":
    main()
