"""
permtest.py — exact p-values by permutation.

Two different questions, two tests. Both work the same way: keep everything
about the strategy fixed, destroy only the link between a signal and what
followed it, repeat many times, and see where the real result falls in the
resulting distribution.

    TEST A — does the STRATEGY beat chance?
        Null: signals fire on the same days, but each is paired with a random
        forward return drawn from the same ticker's eligible bars.
        p = fraction of permutations reaching a mean return >= the real one.

    TEST B — does the OPTIMISER find more than chance?
        Null: run the whole genetic search on shuffled data, repeatedly, with
        different seeds. Collect each run's champion fitness.
        p = fraction of shuffled GA runs reaching fitness >= the real champion.

WHY A PERMUTATION TEST RATHER THAN A t-TEST
Trading returns are fat-tailed, serially correlated, and cross-correlated
across tickers on the same day. Every assumption behind a t-test is violated,
and the t-test would report a p-value that is too small — often dramatically.
A permutation test assumes nothing about the distribution. It builds the null
from your own data by breaking the one thing you claim exists.

READING THE RESULT
    p < 0.01   strong evidence the result is not chance
    p < 0.05   conventional significance
    p > 0.05   consistent with chance; you cannot rule out luck
    p > 0.50   chance does this BETTER than half the time

One caution the tool prints and you should hold on to: a small p-value means
the effect is unlikely to be chance. It does not mean the effect is large
enough to trade after costs. Those are separate questions and this only
answers the first.

    python permtest.py --top 600 --perms 2000
    python permtest.py --top 600 --perms 2000 --ga-runs 15   (adds Test B, slow)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fast_rules
import indicators as ind
from baseline import _load_frames
from params import INDICATORS as I

HERE = Path(__file__).parent
HOLD = 20


def collect(built, G, cost):
    """Real signals and, per ticker, the pool of forward returns to draw from.

    Returns (real_returns, pools) where pools[i] is the array of every eligible
    forward return for the ticker that produced real_returns[i]'s signal. The
    null then draws from that same pool, so the permutation preserves which
    tickers and how many signals each contributed.
    """
    real, pools, counts = [], [], []
    for t, df in built.items():
        try:
            fired = fast_rules.evaluate_all(df, I, G)["FIRED"].to_numpy()
        except Exception:
            continue
        o = df["Open"].to_numpy(float)
        c = df["Close"].to_numpy(float)
        n = len(c)
        elig = np.arange(150, n - HOLD - 1)
        if len(elig) < 30:
            continue
        entries = o[elig + 1]
        exits = c[elig + 1 + HOLD]
        ok = np.isfinite(entries) & (entries > 0) & np.isfinite(exits)
        elig, entries, exits = elig[ok], entries[ok], exits[ok]
        if len(elig) < 30:
            continue
        pool = (exits / entries - 1) * 100 - cost

        last = -10 ** 9
        hits = []
        for j, i in enumerate(elig):
            if not fired[i] or i - last < 20:
                continue
            hits.append(j)
            last = i
        if not hits:
            continue
        real.extend(pool[hits])
        pools.append(pool)
        counts.append(len(hits))
    return np.asarray(real), pools, counts


def permute(pools, counts, rng):
    """One null sample: same tickers, same signal counts, random timing."""
    out = []
    for pool, k in zip(pools, counts):
        out.append(rng.choice(pool, size=k, replace=False) if k <= len(pool)
                   else rng.choice(pool, size=k, replace=True))
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=600)
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--cost", type=float, default=0.20)
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    ap.add_argument("--ga-runs", type=int, default=0,
                    help="also run Test B with N shuffled GA runs (slow)")
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--gens", type=int, default=15)
    a = ap.parse_args()

    if a.market == "us":
        from params_us import GATES as G
    else:
        from params import GATES as G

    frames = _load_frames(a.market)
    liq = sorted(((t, float((d["Close"] * d["Volume"]).tail(250).median()))
                  for t, d in frames.items() if len(d) >= 400), key=lambda x: -x[1])
    picks = [t for t, _ in liq[:a.top]]
    print(f"  building indicators for {len(picks):,} tickers...")
    built = {}
    for t in picks:
        try:
            built[t] = ind.build(frames[t])
        except Exception:
            pass
    print(f"  {len(built):,} ready\n")

    # ---------------- TEST A ----------------
    print("=" * 68)
    print("TEST A — does the strategy beat chance?")
    print("=" * 68)
    real, pools, counts = collect(built, G, a.cost)
    if len(real) < 30:
        raise SystemExit("Too few real signals.")
    real_mean = float(real.mean())
    print(f"  real: {len(real):,} signals, mean net return {real_mean:+.3f}%")
    print(f"  running {a.perms:,} permutations...")

    rng = np.random.default_rng(0)
    null = np.empty(a.perms)
    for i in range(a.perms):
        null[i] = permute(pools, counts, rng).mean()
        if (i + 1) % 500 == 0:
            print(f"    {i + 1:,}/{a.perms:,}", end="\r")
    print(" " * 30, end="\r")

    beat = int((null >= real_mean).sum())
    p = (beat + 1) / (a.perms + 1)          # add-one, the unbiased estimator
    print(f"\n  null distribution: mean {null.mean():+.3f}%, "
          f"sd {null.std():.3f}%")
    print(f"  5th-95th percentile: {np.percentile(null, 5):+.3f}% to "
          f"{np.percentile(null, 95):+.3f}%")
    print(f"  permutations matching or beating the real result: {beat:,}/{a.perms:,}")
    print(f"\n  p = {p:.4f}")
    z = (real_mean - null.mean()) / null.std() if null.std() > 0 else 0.0
    print(f"  effect size: {z:+.2f} standard deviations above the null mean")
    if p < 0.01:
        print("  -> strong evidence this is not chance")
    elif p < 0.05:
        print("  -> significant at the conventional 5% level")
    elif p > 0.5:
        print("  -> chance beats this result more often than not")
    else:
        print("  -> consistent with chance; luck cannot be ruled out")
    print(f"\n  NOTE: even a small p only says 'probably not luck'. Whether")
    print(f"  {real_mean:+.3f}% per signal is worth trading after real costs")
    print(f"  is a separate question this test does not answer.")

    result = {"test_a": {"n": len(real), "real_mean": real_mean, "p": p,
                         "null_mean": float(null.mean()),
                         "null_sd": float(null.std()), "z": float(z)}}

    # ---------------- TEST B ----------------
    if a.ga_runs > 0:
        import ga as ga_mod
        print("\n" + "=" * 68)
        print(f"TEST B — does the optimiser find more than chance? "
              f"({a.ga_runs} shuffled GA runs)")
        print("=" * 68)
        lo = min(d.index.min() for d in built.values())
        hi = max(d.index.max() for d in built.values())
        split = lo + (hi - lo) * 0.7
        train = ga_mod.prepare(built, lo, split)

        ga_mod.MIN_SIGNALS = max(30, int(len(real) * 0.4))
        print(f"  MIN_SIGNALS floor {ga_mod.MIN_SIGNALS:,}")
        print("  evolving on REAL data...")
        _, fit_real, n_real, _ = ga_mod.evolve(train, G, a.cost, a.pop, a.gens,
                                               np.random.default_rng(0))
        nulls = []
        for k in range(a.ga_runs):
            _, f, _, _ = ga_mod.evolve(train, G, a.cost, a.pop, a.gens,
                                       np.random.default_rng(100 + k),
                                       shuffle_rng=np.random.default_rng(500 + k),
                                       label=f"[null {k+1}/{a.ga_runs}] ")
            nulls.append(f)
        nulls = np.asarray(nulls)
        beat_b = int((nulls >= fit_real).sum())
        p_b = (beat_b + 1) / (len(nulls) + 1)
        print(f"\n  real champion fitness   : {fit_real:+.3f}%")
        print(f"  shuffled GA fitness     : mean {nulls.mean():+.3f}%, "
              f"max {nulls.max():+.3f}%")
        print(f"  shuffled runs >= real   : {beat_b}/{len(nulls)}")
        print(f"\n  p = {p_b:.4f}")
        if p_b > 0.5:
            print("  -> the optimiser does BETTER on data with no signal in it.")
        elif p_b > 0.05:
            print("  -> optimisation results are consistent with chance.")
        else:
            print("  -> optimisation found more than chance would produce.")
        result["test_b"] = {"real": fit_real, "null_mean": float(nulls.mean()),
                            "null_max": float(nulls.max()), "p": p_b,
                            "runs": len(nulls)}

    (HERE / "permtest.json").write_text(json.dumps(result, indent=2))
    print("\n  Saved: permtest.json")


if __name__ == "__main__":
    main()
