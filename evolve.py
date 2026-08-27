"""
evolve.py — survival colony where genomes mutate on replication.

THE DESIGN
Each agent carries a genome: its own gate thresholds. It trades only the
signals its genome produces. Every month it pays compute. At zero it dies. At
double its starting capital it splits, and the child inherits the parent's
genome with ONE gene mutated one step.

No fitness function, no explicit optimiser. Selection is survival. Genomes that
keep their agent alive spread; genomes that don't, vanish with their host. This
is closer to artificial life than to a genetic algorithm, and it is the design
Conway's automaton describes.

THE CONTROL, WHICH IS THE POINT
Run the identical colony on PERMUTED returns. If the population "evolves"
toward particular genomes just as readily when signals predict nothing, then
what you are watching is drift and survivorship, not adaptation.

Two things to compare at the end:
  * final capital, real colony vs the distribution of permuted colonies
  * whether the surviving genome distribution differs from the starting one
    MORE in the real run than in the permuted runs

A population converging on some genome proves nothing on its own. Random
lineages converge too — the dead ones just aren't there to argue.

EFFICIENCY
Genomes live on a discrete lattice (each gene has a handful of allowed values)
and their signal sets are cached, so recurring genomes cost nothing to
re-evaluate. Mutation moves one gene one step along the lattice, which keeps
cache hits high.

    python evolve.py --top 300 --agents 40 --burn 100
    python evolve.py --top 300 --agents 40 --burn 100 --perms 30
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

# Discrete lattice. Mutation moves one gene one step left or right.
LATTICE = {
    "pullback_lookback": [5, 8, 10, 14, 20],
    "stoch_trigger_ceiling": [40.0, 50.0, 60.0, 70.0, 80.0],
    "confirm_window": [2, 3, 5, 8, 12],
    "min_rel_volume": [0.9, 1.1, 1.2, 1.5, 2.0],
    "max_pct_of_bb_range": [0.70, 0.80, 0.90, 0.95, 1.00],
}
GENES = list(LATTICE)


def seed_genome(base_G):
    """Index of each gene's value on the lattice, starting from your settings."""
    g = []
    for k in GENES:
        cur = getattr(base_G, k)
        vals = LATTICE[k]
        g.append(int(np.argmin([abs(float(v) - float(cur)) for v in vals])))
    return tuple(g)


def mutate(genome, rng, ledger=None, strength=1.0):
    """Mutate one gene. With a ledger, biased AWAY from values that killed.

    The ledger records every agent that ever lived — dead ones included. Pure
    selection only learns from survivors, which is survivorship bias baked into
    the algorithm: the failures carry just as much information about which
    gene values are lethal, and discarding them throws away most of the data.
    """
    g = list(genome)
    i = int(rng.integers(len(g)))
    vals = LATTICE[GENES[i]]

    if ledger is None or strength <= 0:
        step = 1 if rng.random() < 0.5 else -1
        g[i] = int(np.clip(g[i] + step, 0, len(vals) - 1))
        return tuple(g)

    # score each value of this gene by the mean outcome of every agent that
    # ever carried it, alive or dead
    scores = np.zeros(len(vals))
    seen = np.zeros(len(vals))
    for gen, outcome in ledger:
        v = gen[i]
        scores[v] += outcome
        seen[v] += 1
    mean = np.where(seen > 0, scores / np.maximum(seen, 1), np.nan)
    if np.all(np.isnan(mean)):
        step = 1 if rng.random() < 0.5 else -1
        g[i] = int(np.clip(g[i] + step, 0, len(vals) - 1))
        return tuple(g)

    # softmax over observed means; unseen values get the average so they are
    # still explored rather than written off
    filled = np.where(np.isnan(mean), np.nanmean(mean), mean)
    z = (filled - filled.mean()) / (filled.std() + 1e-9)
    w = np.exp(strength * z)
    w = w / w.sum()
    g[i] = int(rng.choice(len(vals), p=w))
    return tuple(g)


def to_gates(genome, base_G):
    kw = {k: LATTICE[k][genome[j]] for j, k in enumerate(GENES)}
    return dataclasses.replace(base_G, **kw)


class SignalCache:
    """genome -> (sorted dates, returns). Computed once per distinct genome."""

    def __init__(self, built, base_G, hold, cost):
        self.built, self.base_G = built, base_G
        self.hold, self.cost = hold, cost
        self.cache = {}
        self.misses = 0

    def get(self, genome):
        if genome in self.cache:
            return self.cache[genome]
        self.misses += 1
        G = to_gates(genome, self.base_G)
        ds, rs = [], []
        for t, df in self.built.items():
            try:
                fired = fast_rules.evaluate_all(df, I, G)["FIRED"].to_numpy()
            except Exception:
                continue
            o = df["Open"].to_numpy(float)
            c = df["Close"].to_numpy(float)
            idx = df.index
            last = -10 ** 9
            for i in np.flatnonzero(fired):
                if i < 150 or i + self.hold + 1 >= len(c) or i - last < 20:
                    continue
                e = o[i + 1]
                if not np.isfinite(e) or e <= 0:
                    continue
                ds.append(idx[i + 1])
                rs.append((c[i + 1 + self.hold] / e - 1) * 100 - self.cost)
                last = i
        if not ds:
            self.cache[genome] = (np.array([], dtype="datetime64[ns]"),
                                  np.array([]))
            return self.cache[genome]
        order = np.argsort(np.array(ds))
        self.cache[genome] = (pd.DatetimeIndex(np.array(ds)[order]),
                              np.array(rs)[order])
        return self.cache[genome]


def run(cache, seed, n_agents, start_cap, slots, hold, burn, rng,
        shuffle=False, max_pop=400, learn=0.0):
    """One colony. Agents trade their own genome's signals."""
    ledger = [] if learn > 0 else None      # (genome, outcome) for EVERY agent
    genomes = [seed] * n_agents
    caps = [float(start_cap)] * n_agents
    alive = [True] * n_agents
    slots_used = [[] for _ in range(n_agents)]
    births = deaths = 0

    # global timeline from the seed genome (all genomes share the same window)
    all_d, _ = cache.get(seed)
    if len(all_d) == 0:
        return None
    months = pd.date_range(all_d.min(), all_d.max(), freq="MS")

    # per-agent cursor into its own signal array
    cursor = [0] * n_agents

    for mi in range(len(months) - 1):
        m0, m1 = months[mi], months[mi + 1]
        for a in range(len(genomes)):
            if not alive[a]:
                continue
            caps[a] -= burn
            if caps[a] <= 0:
                alive[a] = False
                deaths += 1
                if ledger is not None:
                    ledger.append((genomes[a], caps[a] - start_cap))
                continue

            d, r = cache.get(genomes[a])
            if len(d) == 0:
                continue
            if shuffle:
                r = rng.permutation(r)
            # signals for this agent inside this month
            lo = np.searchsorted(d, m0)
            hi = np.searchsorted(d, m1)
            for k in range(lo, hi):
                slots_used[a] = [t for t in slots_used[a] if t > d[k]]
                if len(slots_used[a]) >= slots:
                    continue
                stake = caps[a] / slots
                caps[a] += stake * (r[k] / 100.0)
                slots_used[a].append(d[k] + pd.Timedelta(days=hold * 1.45))
                if caps[a] <= 0:
                    alive[a] = False
                    deaths += 1
                    if ledger is not None:
                        ledger.append((genomes[a], caps[a] - start_cap))
                    break
            if not alive[a]:
                continue
            if caps[a] >= 2 * start_cap and len(genomes) < max_pop:
                caps[a] /= 2
                genomes.append(mutate(genomes[a], rng, ledger, learn))
                caps.append(caps[a])
                alive.append(True)
                slots_used.append([])
                cursor.append(0)
                births += 1

    if ledger is not None:
        for i in range(len(genomes)):
            if alive[i]:
                ledger.append((genomes[i], caps[i] - start_cap))
    surv_idx = [i for i in range(len(genomes)) if alive[i]]
    return {
        "survivors": len(surv_idx),
        "started": n_agents,
        "births": births,
        "deaths": deaths,
        "total_capital": float(sum(caps[i] for i in surv_idx)),
        "best": float(max((caps[i] for i in surv_idx), default=0.0)),
        "genomes": [genomes[i] for i in surv_idx],
        "ledger_size": len(ledger) if ledger is not None else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--agents", type=int, default=40)
    ap.add_argument("--capital", type=float, default=10_000)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--cost", type=float, default=0.20)
    ap.add_argument("--burn", type=float, default=100.0)
    ap.add_argument("--learn", type=float, default=0.0,
                    help="learn from dead agents; 0 = off, 1-3 = bias strength")
    ap.add_argument("--perms", type=int, default=25)
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    a = ap.parse_args()

    if a.market == "us":
        from params_us import GATES as base_G
    else:
        from params import GATES as base_G

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
    print(f"  {len(built):,} ready")

    cache = SignalCache(built, base_G, a.hold, a.cost)
    seed = seed_genome(base_G)
    print(f"  seed genome (your settings): "
          f"{ {k: LATTICE[k][seed[j]] for j, k in enumerate(GENES)} }")
    print(f"  compute burn ${a.burn:,.0f}/month/agent")
    print(f"  learning from deaths: "
          f"{'OFF (pure selection)' if a.learn == 0 else f'ON, strength {a.learn}'}\n")

    rng = np.random.default_rng(0)
    real = run(cache, seed, a.agents, a.capital, a.slots, a.hold, a.burn, rng,
               learn=a.learn)
    if real is None:
        raise SystemExit("Seed genome produced no signals.")

    print("  REAL")
    print(f"    survivors {real['survivors']}/{real['started']}   "
          f"births {real['births']}   deaths {real['deaths']}")
    print(f"    total capital ${real['total_capital']:,.0f} of "
          f"${a.agents * a.capital:,.0f} deployed")
    print(f"    best lineage  ${real['best']:,.0f}")

    if real["genomes"]:
        print("\n    surviving genome distribution:")
        for j, k in enumerate(GENES):
            vals = [LATTICE[k][g[j]] for g in real["genomes"]]
            seedv = LATTICE[k][seed[j]]
            print(f"      {k:<24} median {np.median(vals):<8.2f} "
                  f"(seed {seedv})")

    print(f"\n  running {a.perms} permuted colonies...")
    caps_null, births_null = [], []
    for p in range(a.perms):
        prng = np.random.default_rng(500 + p)
        res = run(cache, seed, a.agents, a.capital, a.slots, a.hold, a.burn,
                  prng, shuffle=True, learn=a.learn)
        if res:
            caps_null.append(res["total_capital"])
            births_null.append(res["births"])
        print(f"    {p + 1}/{a.perms}", end="\r")
    print(" " * 24, end="\r")

    caps_null = np.asarray(caps_null)
    beat = int((caps_null >= real["total_capital"]).sum())
    p_val = (beat + 1) / (len(caps_null) + 1)
    print(f"\n  {'':<20}{'REAL':>14}{'permuted median':>18}{'p':>9}")
    print("  " + "-" * 62)
    print(f"  {'total capital':<20}{real['total_capital']:>14,.0f}"
          f"{np.median(caps_null):>18,.0f}{p_val:>9.3f}")
    print(f"  {'replications':<20}{real['births']:>14,}"
          f"{np.median(births_null):>18,.0f}")

    print()
    if real["births"] == 0 and np.median(births_null) == 0:
        print("  -> Nothing replicated in EITHER run. With this compute burn the")
        print("     colony cannot compound fast enough to double. Evolution never")
        print("     starts: selection only kills. Lower --burn to let it run.")
    elif p_val > 0.5:
        print("  -> Permuted colonies did better. The population is drifting,")
        print("     not adapting.")
    elif p_val < 0.05:
        print("  -> The real colony genuinely outgrew permuted ones.")
    else:
        print("  -> Real and permuted colonies are indistinguishable.")

    print(f"\n  cache: {cache.misses} distinct genomes evaluated")
    (HERE / "evolve.json").write_text(json.dumps(
        {"real": {k: v for k, v in real.items() if k != "genomes"},
         "null_median": float(np.median(caps_null)), "p": p_val},
        indent=2, default=float))
    print("  Saved: evolve.json")


if __name__ == "__main__":
    main()
