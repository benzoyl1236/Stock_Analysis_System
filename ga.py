"""
ga.py — genetic optimisation of the gate thresholds, with a null control.

WHAT IT DOES
Evolves a population of gate parameter sets. Each genome is scored on a
training window; the best breed, mutate and pass on. Standard GA.

WHY THE NULL CONTROL IS THE POINT
A GA evaluating population 30 over 25 generations tries 750 parameter sets. At
a 5% threshold that is ~37 combinations that look good purely by chance. A GA
will therefore ALWAYS report a high-fitness champion, on any data, including
data with no signal in it whatsoever. Fitness alone tells you nothing.

So this runs the identical GA twice:

  REAL      the actual price data
  SHUFFLED  the same signals, but forward returns randomly reassigned in time,
            which destroys any relationship between a setup and what followed
            while preserving the return distribution exactly

If the shuffled run reaches similar fitness, evolution was fitting noise and
the champion is worthless. That comparison — not the fitness number — is the
result.

The champion is then scored on a held-out window neither run touched.

WHAT IS AND IS NOT EVOLVED
Only the GATE THRESHOLDS. The indicator periods (EMA 8/21/100, BB 20/2,
Stochastic 9/3/3, MACD 9/18/9, Momentum 28) are your POEMS template and stay
fixed. Evolving those too would multiply the search space and the overfitting
with it, and they are the one part of this system you did not invent.

    python ga.py --top 600 --pop 30 --gens 25
    python ga.py --top 600 --pop 30 --gens 25 --no-null   (faster, less honest)
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

# gene name -> (low, high, is_integer)
GENES = {
    "pullback_lookback": (3, 25, True),
    "pullback_ema_tolerance": (0.0, 0.05, False),
    "stoch_trigger_ceiling": (35.0, 85.0, False),
    "confirm_window": (1, 10, True),
    "min_rel_volume": (0.8, 2.5, False),
    "max_pct_of_bb_range": (0.50, 1.00, False),
}

MIN_SIGNALS = 60          # a genome with fewer is not evaluable
HOLD = 20


def random_genome(rng, base):
    g = {}
    for k, (lo, hi, is_int) in GENES.items():
        g[k] = int(rng.integers(lo, hi + 1)) if is_int else float(rng.uniform(lo, hi))
    return g


def mutate(g, rng, rate=0.3):
    out = dict(g)
    for k, (lo, hi, is_int) in GENES.items():
        if rng.random() < rate:
            span = (hi - lo) * 0.25
            v = out[k] + rng.normal(0, span)
            v = min(max(v, lo), hi)
            out[k] = int(round(v)) if is_int else float(v)
    return out


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in GENES}


def prepare(built, start, end):
    """Freeze the per-ticker arrays once so the GA only pays for gate evaluation."""
    prepped = []
    for t, df in built.items():
        mask = (df.index >= start) & (df.index < end)
        if mask.sum() < 30:
            continue
        prepped.append((df, np.flatnonzero(mask),
                        df["Open"].to_numpy(float), df["Close"].to_numpy(float)))
    return prepped


def fitness(genome, prepped, base_G, cost, shuffle_rng=None):
    """Mean net return per signal. Returns (fitness, n)."""
    G = dataclasses.replace(base_G, **genome)
    rets = []
    for df, pos, o, c in prepped:
        try:
            fired = fast_rules.evaluate_all(df, I, G)["FIRED"].to_numpy()
        except Exception:
            continue
        last = -10 ** 9
        idx = []
        for i in pos:
            if i < 150 or i + HOLD + 1 >= len(c):
                continue
            if not fired[i] or i - last < 20:
                continue
            idx.append(i)
            last = i
        if not idx:
            continue
        if shuffle_rng is not None:
            # destroy the signal->outcome link: pair each signal with a random
            # bar's forward return from the same ticker and window
            pool = pos[(pos >= 150) & (pos + HOLD + 1 < len(c))]
            if len(pool) == 0:
                continue
            picks = shuffle_rng.choice(pool, size=len(idx), replace=True)
            for j in picks:
                e = o[j + 1]
                if np.isfinite(e) and e > 0:
                    rets.append((c[j + 1 + HOLD] / e - 1) * 100 - cost)
        else:
            for i in idx:
                e = o[i + 1]
                if np.isfinite(e) and e > 0:
                    rets.append((c[i + 1 + HOLD] / e - 1) * 100 - cost)

    if len(rets) < MIN_SIGNALS:
        return -99.0, len(rets)
    return float(np.mean(rets)), len(rets)


def evolve(prepped, base_G, cost, pop_size, gens, rng, shuffle_rng=None, label=""):
    pop = [random_genome(rng, base_G) for _ in range(pop_size)]
    pop[0] = {k: getattr(base_G, k) for k in GENES}      # seed with your settings
    best, best_fit, best_n = None, -1e9, 0
    history = []

    for gen in range(gens):
        scored = []
        for g in pop:
            f, n = fitness(g, prepped, base_G, cost, shuffle_rng)
            scored.append((f, n, g))
        scored.sort(key=lambda x: -x[0])
        if scored[0][0] > best_fit:
            best_fit, best_n, best = scored[0][0], scored[0][1], scored[0][2]
        history.append(scored[0][0])
        print(f"    {label}gen {gen + 1:>3}/{gens}  best {scored[0][0]:+.3f}%  "
              f"(n={scored[0][1]:,})", end="\r")

        elite = [g for _, _, g in scored[:max(2, pop_size // 5)]]
        nxt = list(elite)
        while len(nxt) < pop_size:
            a, b = rng.choice(len(elite), 2, replace=True)
            nxt.append(mutate(crossover(elite[a], elite[b], rng), rng))
        pop = nxt
    print()
    return best, best_fit, best_n, history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=600)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--cost", type=float, default=0.20)
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    ap.add_argument("--no-null", action="store_true",
                    help="skip the shuffled control (not recommended)")
    a = ap.parse_args()

    if a.market == "us":
        from params_us import GATES as base_G
    else:
        from params import GATES as base_G

    evals = a.pop * a.gens
    print("Genetic optimisation of gate thresholds")
    print(f"  population {a.pop} x {a.gens} generations = {evals:,} parameter sets tried")
    print(f"  at a 5% threshold, ~{evals * 0.05:.0f} of those will look good by chance\n")

    frames = _load_frames(a.market)
    liq = sorted(((t, float((d["Close"] * d["Volume"]).tail(250).median()))
                  for t, d in frames.items() if len(d) >= 400), key=lambda x: -x[1])
    picks = [t for t, _ in liq[:a.top]]
    print(f"  building indicators for {len(picks):,} tickers...")
    built = {}
    for n, t in enumerate(picks, 1):
        try:
            built[t] = ind.build(frames[t])
        except Exception:
            pass
    print(f"  {len(built):,} ready")

    lo = min(d.index.min() for d in built.values())
    hi = max(d.index.max() for d in built.values())
    split = lo + (hi - lo) * 0.7
    print(f"  TRAIN {lo.date()} to {split.date()}   HOLDOUT {split.date()} to {hi.date()}\n")

    train = prepare(built, lo, split)
    test = prepare(built, split, hi)

    rng = np.random.default_rng(0)
    print("  Evolving on REAL data")
    champ, fit_real, n_real, _ = evolve(train, base_G, a.cost, a.pop, a.gens, rng)

    fit_null = None
    if not a.no_null:
        print("\n  Evolving on SHUFFLED data (signal->outcome link destroyed)")
        rng2 = np.random.default_rng(1)
        _, fit_null, n_null, _ = evolve(train, base_G, a.cost, a.pop, a.gens,
                                        rng2, shuffle_rng=np.random.default_rng(2))

    seed_fit, seed_n = fitness({k: getattr(base_G, k) for k in GENES},
                               train, base_G, a.cost)
    champ_test, champ_test_n = fitness(champ, test, base_G, a.cost)
    seed_test, seed_test_n = fitness({k: getattr(base_G, k) for k in GENES},
                                     test, base_G, a.cost)

    print("\n  " + "=" * 68)
    print("  CHAMPION GENOME")
    for k in GENES:
        cur = getattr(base_G, k)
        v = champ[k]
        mark = "" if np.isclose(float(v), float(cur)) else "  <- changed"
        print(f"    {k:<26} {v}{mark}   (yours: {cur})")

    print(f"\n  {'':<26}{'TRAIN':>12}{'HOLDOUT':>12}")
    print("  " + "-" * 52)
    def _f(v, n):
        return f"too few ({n})".rjust(12) if v <= -98 else f"{v:>11.3f}%"
    print(f"  {'your settings':<26}{_f(seed_fit, seed_n)}{_f(seed_test, seed_test_n)}")
    print(f"  {'GA champion':<26}{_f(fit_real, n_real)}{_f(champ_test, champ_test_n)}")
    if fit_null is not None:
        print(f"  {'GA on SHUFFLED data':<26}{fit_null:>11.3f}%{'—':>12}")

    print("\n  " + "=" * 68)
    if fit_null is not None:
        gap = fit_real - fit_null
        print(f"  Real fitness {fit_real:+.3f}% vs shuffled {fit_null:+.3f}%  "
              f"(gap {gap:+.3f}pp)")
        if gap < 0.2:
            print("\n  -> The GA reached comparable fitness on data with NO signal in it.")
            print("     Evolution was fitting noise. The champion is meaningless,")
            print("     however good its training number looks.")
        else:
            print("\n  -> Real beat shuffled by a clear margin. Something may be there.")
    if np.isfinite(champ_test) and np.isfinite(seed_test):
        if champ_test <= -98:
            print(f"  -> On the HOLDOUT the champion produced too few signals to")
            print(f"     evaluate ({champ_test_n}). It evolved into a rule so narrow it")
            print(f"     barely fires outside the data it was fitted to.")
        elif champ_test <= seed_test:
            print(f"  -> On the HOLDOUT the champion ({champ_test:+.3f}%) did not beat")
            print(f"     your untouched settings ({seed_test:+.3f}%). Optimisation lost.")
        else:
            print(f"  -> Champion beat your settings out of sample "
                  f"({champ_test:+.3f}% vs {seed_test:+.3f}%).")

    print("\n  Reminder: adopting this champion means a NEW fingerprint and")
    print("  invalidates every base rate computed under the old one.")
    (HERE / "ga_champion.json").write_text(json.dumps(
        {"champion": champ, "train_fitness": fit_real, "holdout": champ_test,
         "shuffled_fitness": fit_null, "seed_train": seed_fit,
         "seed_holdout": seed_test}, indent=2, default=float))
    print("  Saved: ga_champion.json")


if __name__ == "__main__":
    main()
