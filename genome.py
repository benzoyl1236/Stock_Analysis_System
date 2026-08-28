"""
genome.py — full evolution: indicator settings, which gates apply, thresholds.

    *** THIS TOOL FAILED ITS OWN NULL CONTROL. DO NOT TRUST ITS OUTPUT. ***

    Run on pure random-walk data containing no signal whatsoever, it reported
    a champion with +$25,661 "excess" and passed both of its validity checks.
    Two fitness definitions were tried:

      final capital        rewarded EXPOSURE, not skill. Switching gates OFF
                           means more trades, more time invested, more market
                           drift captured. The champion on noise turned REGIME
                           and MACD off and took 391 holdout trades vs the
                           seed's 282.

      excess over own      the right idea, but the baseline needs many more
      shuffled twin        shuffles than is affordable per genome. With 3-5,
                           the estimate is noisy, and selecting the best of
                           ~40 genomes on a noisy measure reproduces the same
                           false positive.

    The underlying problem is not fixable by tweaking fitness: the genome
    space is ~3.1 billion and the sample is ~6,000. Selecting a winner from
    that space cannot be validated by any in-sample criterion.

    The file is kept because the machinery is instructive and the null control
    is worth seeing fail. It is not a research tool. Any champion it produces
    should be assumed to be noise.


WHAT THE GENOME CONTROLS
Unlike ga.py, which only moved gate thresholds, a genome here specifies:

  * indicator settings   Bollinger period and SD, the three EMA lengths,
                         stochastic length, MACD set, momentum length
  * gate mask            8 on/off bits. A gate that is OFF is not required.
                         Trades no longer need all 8 to approve.
  * thresholds           pullback window, stochastic ceiling, confirm window,
                         volume multiple, headroom

Agents live: each starts with capital, trades its own genome's signals, pays
compute monthly, dies at zero. Survivors that double split, and the child
carries a mutated genome. When a generation dies out, the next starts from
what the ledger learned.

READ THIS BEFORE BELIEVING ANY RESULT
The genome space is about 3 BILLION combinations. You have ~6,000 samples.
That is 500,000 genomes per data point. Searching a space that much larger
than the data does not find the best rule — it finds the rule that best fits
the noise, every time, with certainty.

So this runs the identical evolution twice:

  REAL      your price data
  SHUFFLED  same signals, forward returns reassigned at random, so there is
            provably nothing to discover

and then scores the champion on a HOLDOUT period it never touched. Three
numbers decide everything:

  1. does REAL beat SHUFFLED by a clear margin?
  2. does the champion beat the SEED genome (your current settings) on the
     holdout?
  3. does it beat simply holding the universe?

If the answer to any is no, the champion is a description of past noise.

    python genome.py --top 300 --pop 24 --gens 15
    python genome.py --top 300 --pop 24 --gens 15 --no-null   (faster, weaker)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from basics import load

HERE = Path(__file__).parent
HOLD = 20
GATE_NAMES = ("REGIME", "PULLBACK", "TRIGGER", "MACD", "MOMENTUM",
              "VOLUME", "LIQUIDITY", "HEADROOM")

LATTICE = {
    "bb_period":   [10, 14, 20, 30],
    "bb_sd":       [1.5, 2.0, 2.5, 3.0],
    "ema_fast":    [5, 8, 13],
    "ema_mid":     [13, 21, 34],
    "ema_slow":    [50, 100, 200],
    "stoch_k":     [5, 9, 14],
    "macd_set":    [(6, 13, 5), (9, 18, 9), (12, 26, 9)],
    "mtm":         [14, 28, 56],
    "pullback_lookback": [5, 10, 15, 25],
    "stoch_ceiling":     [40.0, 60.0, 75.0, 90.0],
    "confirm_window":    [2, 5, 8, 12],
    "min_rel_vol":       [0.8, 1.0, 1.2, 1.6],
    "headroom":          [0.70, 0.85, 0.95, 1.00],
}
KEYS = list(LATTICE)


def seed_genome():
    """Your current settings, as a point on the lattice."""
    want = {"bb_period": 20, "bb_sd": 2.0, "ema_fast": 8, "ema_mid": 21,
            "ema_slow": 100, "stoch_k": 9, "macd_set": (9, 18, 9), "mtm": 28,
            "pullback_lookback": 10, "stoch_ceiling": 60.0,
            "confirm_window": 5, "min_rel_vol": 1.2, "headroom": 0.90}
    g = tuple(LATTICE[k].index(want[k]) if want[k] in LATTICE[k] else
              len(LATTICE[k]) // 2 for k in KEYS)
    return g + (255,)          # all 8 gates on


def decode(gen):
    d = {k: LATTICE[k][gen[i]] for i, k in enumerate(KEYS)}
    d["mask"] = gen[-1]
    return d


def mutate(gen, rng, p_gate=0.35):
    g = list(gen)
    if rng.random() < p_gate:
        bit = int(rng.integers(8))
        g[-1] ^= (1 << bit)                     # flip one gate on/off
        if g[-1] == 0:
            g[-1] = 1 << bit                    # never zero gates
    else:
        i = int(rng.integers(len(KEYS)))
        step = 1 if rng.random() < 0.5 else -1
        g[i] = int(np.clip(g[i] + step, 0, len(LATTICE[KEYS[i]]) - 1))
    return tuple(g)


def crossover(a, b, rng):
    return tuple(a[i] if rng.random() < 0.5 else b[i] for i in range(len(a)))


# --------------------------------------------------------------------------
def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def signals_for(gen, raw_frames, min_turn, min_price, cache):
    """(dates, returns) for this genome. Indicators recomputed per genome."""
    if gen in cache:
        return cache[gen]
    p = decode(gen)
    mask = p["mask"]
    ds, rs = [], []
    for t, df in raw_frames.items():
        c = df["Close"].to_numpy(float)
        h = df["High"].to_numpy(float)
        l = df["Low"].to_numpy(float)
        v = df["Volume"].to_numpy(float)
        o = df["Open"].to_numpy(float)
        n = len(c)
        if n < 300:
            continue
        cs = pd.Series(c)
        e_f, e_m, e_s = (_ema(c, p["ema_fast"]), _ema(c, p["ema_mid"]),
                         _ema(c, p["ema_slow"]))
        mid = cs.rolling(p["bb_period"]).mean().to_numpy()
        sd = cs.rolling(p["bb_period"]).std(ddof=0).to_numpy()
        up, lo = mid + p["bb_sd"] * sd, mid - p["bb_sd"] * sd
        rngb = up - lo
        bbpos = np.where(rngb > 0, (c - lo) / rngb, 0.5)
        ll = pd.Series(l).rolling(p["stoch_k"]).min().to_numpy()
        hh = pd.Series(h).rolling(p["stoch_k"]).max().to_numpy()
        span = hh - ll
        rawk = np.where(span > 0, 100 * (c - ll) / span, 50.0)
        k = pd.Series(rawk).rolling(3).mean().to_numpy()
        dd = pd.Series(k).rolling(3).mean().to_numpy()
        f, sl, sg = p["macd_set"]
        macd = _ema(c, f) - _ema(c, sl)
        hist = macd - _ema(macd, sg)
        mtm = c - np.r_[np.full(p["mtm"], np.nan), c[:-p["mtm"]]]
        vavg = pd.Series(v).rolling(20).mean().to_numpy()
        relv = np.where(vavg > 0, v / vavg, 0.0)
        turn = pd.Series(c * v).rolling(20).mean().to_numpy()

        kp, dp = np.r_[np.nan, k[:-1]], np.r_[np.nan, dd[:-1]]
        cross = (kp <= dp) & (k > dd) & (k < p["stoch_ceiling"])
        hp = np.r_[np.nan, hist[:-1]]
        touched = (l <= e_m * 1.01) | (k <= 30)
        pb = pd.Series(touched).rolling(p["pullback_lookback"],
                                        min_periods=1).max().to_numpy() > 0
        cw = pd.Series(cross).rolling(p["confirm_window"],
                                      min_periods=1).max().to_numpy() > 0
        rv = pd.Series(relv).rolling(p["confirm_window"],
                                     min_periods=1).max().to_numpy()

        ok = np.ones(n, dtype=bool)
        if mask & 1:   ok &= (c > e_s) & (e_f > e_m) & (e_m > e_s)
        if mask & 2:   ok &= pb
        if mask & 4:   ok &= cw
        if mask & 8:   ok &= (macd > 0) & (hist > 0) & (hist > hp)
        if mask & 16:  ok &= mtm > 0
        if mask & 32:  ok &= rv >= p["min_rel_vol"]
        if mask & 64:  ok &= (turn >= min_turn) & (c >= min_price)
        if mask & 128: ok &= bbpos <= p["headroom"]
        ok &= np.isfinite(e_s) & np.isfinite(lo) & np.isfinite(mtm)

        idx = df.index
        last = -10 ** 9
        for i in np.flatnonzero(ok):
            if i < 260 or i + HOLD + 1 >= n or i - last < 20:
                continue
            e = o[i + 1]
            x = c[i + 1 + HOLD]
            if not (np.isfinite(e) and e > 0 and np.isfinite(x)):
                continue
            ds.append(idx[i + 1]); rs.append((x / e - 1) * 100)
            last = i
    res = (pd.DatetimeIndex(ds), np.asarray(rs)) if ds else (
        pd.DatetimeIndex([]), np.array([]))
    cache[gen] = res
    return res


# --------------------------------------------------------------------------
def live(gen, raw, lo_t, hi_t, min_turn, min_price, cache, cap, slots, cost,
         burn, rng, shuffle=False, min_trades=60):
    """One agent's life. Returns final capital and trade count."""
    d, r = signals_for(gen, raw, min_turn, min_price, cache)
    if len(d) == 0:
        return 0.0, 0
    sel = (d >= lo_t) & (d < hi_t)
    d, r = d[sel], r[sel]
    if len(d) < min_trades:
        return 0.0, len(d)
    if shuffle:
        r = rng.permutation(r)
    eq = float(cap)
    free, n = [], 0
    last_m = None
    for i in range(len(d)):
        t = d[i]
        m = (t.year, t.month)
        if last_m is None:
            last_m = m
        elif m != last_m:
            last_m = m
            eq -= burn
            if eq <= 0:
                return 0.0, n
        free = [x for x in free if x > t]
        if len(free) >= slots:
            continue
        eq += (eq / slots) * (r[i] / 100.0 - cost / 100.0)
        if eq <= 0:
            return 0.0, n
        free.append(t + pd.Timedelta(days=HOLD * 1.45))
        n += 1
    return eq, n


def fitness(gen, raw, lo_t, hi_t, min_turn, min_price, cache, cap, slots,
            cost, burn, rng, n_shuffle=3):
    """EXCESS capital over random timing with the SAME trade schedule.

    Using final capital alone was wrong, and the null control caught it: with
    gates switched OFF a genome simply trades more, stays invested longer and
    collects more market drift. On pure noise that produced a "champion" that
    beat the seed out of sample — exposure masquerading as skill.

    Scoring each genome against ITS OWN shuffled twin removes that entirely.
    Same dates, same trade count, same exposure; only the pairing of signal to
    outcome differs. What is left is edge.
    """
    real, n = live(gen, raw, lo_t, hi_t, min_turn, min_price, cache, cap,
                   slots, cost, burn, rng)
    if n == 0:
        return -1e9, 0.0, 0
    nulls = [live(gen, raw, lo_t, hi_t, min_turn, min_price, cache, cap,
                  slots, cost, burn, rng, shuffle=True)[0]
             for _ in range(n_shuffle)]
    base = float(np.mean(nulls)) if nulls else 0.0
    return real - base, real, n


def evolve(raw, lo_t, hi_t, min_turn, min_price, pop_size, gens, rng,
           cap, slots, cost, burn, shuffle=False, label=""):
    cache = {}
    seed = seed_genome()
    pop = [seed] + [mutate(seed, rng) for _ in range(pop_size - 1)]
    best, best_eq = seed, -1.0
    for g in range(gens):
        scored = []
        for gen in pop:
            if shuffle:
                eq, n = live(gen, raw, lo_t, hi_t, min_turn, min_price, cache,
                             cap, slots, cost, burn, rng, shuffle=True)
                scored.append((eq - cap, eq, n, gen))
            else:
                exc, eq, n = fitness(gen, raw, lo_t, hi_t, min_turn, min_price,
                                     cache, cap, slots, cost, burn, rng)
                scored.append((exc, eq, n, gen))
        scored.sort(key=lambda x: -x[0])
        if scored[0][0] > best_eq:
            best_eq, best = scored[0][0], scored[0][3]
        alive = [s for s in scored if s[1] > 0]
        print(f"    {label}gen {g+1:>3}/{gens}  best excess "
              f"${scored[0][0]:>+11,.0f}  trades {scored[0][2]:>4}  "
              f"alive {len(alive):>3}/{len(pop)}  cached {len(cache)}", end="\r")
        if not alive:
            print(f"\n    {label}generation extinct — restarting from ledger")
            pop = [mutate(best, rng) for _ in range(pop_size)]
            continue
        elite = [s[3] for s in alive[:max(2, pop_size // 4)]]
        nxt = list(elite)
        while len(nxt) < pop_size:
            a, b = rng.integers(len(elite)), rng.integers(len(elite))
            nxt.append(mutate(crossover(elite[a], elite[b], rng), rng))
        pop = nxt
    print()
    return best, best_eq, cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--capital", type=float, default=50_000)
    ap.add_argument("--slots", type=int, default=20)
    ap.add_argument("--cost", type=float, default=0.20)
    ap.add_argument("--burn", type=float, default=0.0)
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    ap.add_argument("--cache-tag", default="")
    ap.add_argument("--no-null", action="store_true")
    a = ap.parse_args()

    if a.market == "us":
        from params_us import GATES as G
    else:
        from params import GATES as G

    frames = load(a.cache_tag, a.market)
    liq = sorted(((t, float((d["Close"] * d["Volume"]).tail(250).median()))
                  for t, d in frames.items() if len(d) >= 500), key=lambda x: -x[1])
    raw = {t: frames[t] for t, _ in liq[:a.top]}
    print(f"    {len(raw):,} tickers")

    lo = min(d.index.min() for d in raw.values())
    hi = max(d.index.max() for d in raw.values())
    split = lo + (hi - lo) * 0.65
    print(f"    TRAIN {lo.date()} to {split.date()}   "
          f"HOLDOUT {split.date()} to {hi.date()}")
    print(f"    genome space ~3.1 billion; samples ~6,000\n")

    rng = np.random.default_rng(0)
    print("  Evolving on REAL data")
    champ, eq, cache = evolve(raw, lo, split, G.min_avg_turnover_sgd,
                              G.min_price_sgd, a.pop, a.gens, rng,
                              a.capital, a.slots, a.cost, a.burn)

    null_eq = None
    if not a.no_null:
        print("\n  Evolving on SHUFFLED data (nothing to find)")
        _, null_eq, _ = evolve(raw, lo, split, G.min_avg_turnover_sgd,
                               G.min_price_sgd, a.pop, a.gens,
                               np.random.default_rng(1), a.capital, a.slots,
                               a.cost, a.burn, shuffle=True, label="[null] ")

    seed = seed_genome()
    _, seed_tr, _ = fitness(seed, raw, lo, split, G.min_avg_turnover_sgd,
                            G.min_price_sgd, cache, a.capital, a.slots,
                            a.cost, a.burn, rng)
    champ_exc, champ_ho, champ_n = fitness(champ, raw, split, hi,
        G.min_avg_turnover_sgd, G.min_price_sgd, cache, a.capital, a.slots,
        a.cost, a.burn, rng, n_shuffle=5)
    seed_exc, seed_ho, seed_n = fitness(seed, raw, split, hi,
        G.min_avg_turnover_sgd, G.min_price_sgd, cache, a.capital, a.slots,
        a.cost, a.burn, rng, n_shuffle=5)

    p = decode(champ)
    print("\n  " + "=" * 66)
    print("  CHAMPION GENOME")
    sp = decode(seed)
    for k in KEYS:
        mark = "" if p[k] == sp[k] else "  <-"
        print(f"    {k:<20} {str(p[k]):<14}{mark}")
    on = [n for i, n in enumerate(GATE_NAMES) if p["mask"] & (1 << i)]
    off = [n for i, n in enumerate(GATE_NAMES) if not p["mask"] & (1 << i)]
    print(f"    gates ON   {', '.join(on) if on else '(none)'}")
    print(f"    gates OFF  {', '.join(off) if off else '(none)'}")

    print(f"\n  {'':<26}{'TRAIN':>14}{'HOLDOUT':>14}")
    print("  " + "-" * 54)
    print(f"  {'your settings (seed)':<26}{seed_tr:>13,.0f}{seed_ho:>13,.0f}")
    print(f"  {'evolved champion':<26}{eq:>13,.0f}{champ_ho:>13,.0f}")
    print(f"\n  EXCESS over random timing at the same exposure (the real test)")
    print(f"    seed on holdout      {seed_exc:>+13,.0f}")
    print(f"    champion on holdout  {champ_exc:>+13,.0f}")
    if null_eq is not None:
        print(f"  {'champion on SHUFFLED':<26}{null_eq:>13,.0f}{'—':>14}")

    print("\n  " + "=" * 66)
    if null_eq is not None:
        if eq <= null_eq * 1.05:
            print("  1. FAIL — evolution did as well on data with nothing in it.")
        else:
            print(f"  1. real ${eq:,.0f} vs shuffled ${null_eq:,.0f} — real ahead.")
    if champ_exc <= seed_exc:
        print(f"  2. FAIL — on the holdout the champion's excess "
              f"(${champ_exc:+,.0f}) did not")
        print(f"     beat the seed's (${seed_exc:+,.0f}). Raw capital can be higher")
        print(f"     purely from trading more; excess is what removes that.")
    else:
        print(f"  2. champion beat the seed out of sample.")
    print(f"\n  champion holdout trades: {champ_n}  (seed: {seed_n})")
    print(f"  distinct genomes evaluated: {len(cache)}")
    print(f"\n  Adopting this means a new fingerprint and voids every base rate.")

    (HERE / "genome_champion.json").write_text(json.dumps(
        {"champion": {k: str(v) for k, v in p.items()},
         "train": eq, "holdout": champ_ho, "seed_train": seed_tr,
         "seed_holdout": seed_ho, "shuffled": null_eq}, indent=2, default=float))
    print("  Saved: genome_champion.json")


if __name__ == "__main__":
    main()
