"""
walkforward.py — adaptive tuning, with an honesty check built in.

WHAT YOU ASKED FOR
Start on the first 6 months of 2020. Try changing one parameter. If profit
improves, keep it. Walk forward 6 months. Repeat.

WHAT THIS ADDS, AND WHY IT MATTERS MORE THAN THE OPTIMIZER
That loop will ALWAYS find improvements. With 5 parameters and 2 alternatives
each, you run 10 tests per window. At the usual 5% threshold you expect roughly
0.5 false discoveries per window by chance alone — over 11 windows, about 5
parameter changes that look profitable and mean nothing. The optimizer cannot
tell those apart from real ones, because in-sample it is exactly what an
improvement looks like.

So every adopted change is then tested on the NEXT window, which the optimizer
had not seen when it chose. That gives the number that decides whether any of
this works:

    SURVIVAL RATE — of the changes adopted because they improved in-sample,
    what fraction also improved out-of-sample?

  ~50%  the optimizer is a coin flip. Adapting adds nothing over never
        changing anything, and you have built an expensive random number
        generator.
  >65%  the adaptation is finding something real.
  <40%  adapting is actively WORSE than leaving the parameters alone.

The run also compares the adaptive path against a FROZEN baseline that never
changes. If frozen wins, that is your answer.

CONFIDENCE
Every performance difference gets a bootstrap 95% confidence interval, because
a +0.3% mean improvement from 40 trades is noise and a report that prints it
without an interval is misleading. Differences whose interval spans zero are
marked accordingly and are NOT counted as real improvements.

WHERE TO RUN IT
Locally. It needs the cache_us/ parquet (several hundred MB) and a few minutes
of CPU. GitHub Actions would have to rebuild the cache every run, and the
free-tier machines are slower than yours. This is not a scheduled job — it is
an experiment you run once and read carefully.

    python walkforward.py --top 800
    python walkforward.py --top 800 --months 6 --cost 0.20
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fast_rules
from params import INDICATORS as _I
import indicators as ind
from baseline import _load_frames

HERE = Path(__file__).parent

# One parameter changed at a time. Indicator PERIODS are not touched — those
# are your POEMS template. Only the gate thresholds move.
VARIANTS = {
    "pullback_lookback": [5, 15],
    "stoch_trigger_ceiling": [50.0, 70.0],
    "confirm_window": [3, 8],
    "min_rel_volume": [1.0, 1.5],
    "max_pct_of_bb_range": [0.80, 1.00],
}


def evaluate(frames_idx, G, start, end, hold=20, cost=0.20):
    """Mean net return per signal in [start, end). Returns (mean, n, trades)."""
    rets = []
    for t, df in frames_idx.items():
        mask = (df.index >= start) & (df.index < end)
        if mask.sum() < 30:
            continue
        gm = fast_rules.evaluate_all(df, _I, G)
        fired = gm["FIRED"].to_numpy()
        o = df["Open"].to_numpy(float)
        c = df["Close"].to_numpy(float)
        pos = np.flatnonzero(mask)
        last = -10 ** 9
        for i in pos:
            if i < 150 or i + hold + 1 >= len(c):
                continue
            if not fired[i] or i - last < 20:
                continue
            entry = o[i + 1]
            if not np.isfinite(entry) or entry <= 0:
                continue
            rets.append((c[i + 1 + hold] / entry - 1) * 100 - cost)
            last = i
    if not rets:
        return np.nan, 0, []
    return float(np.mean(rets)), len(rets), rets


def boot_ci(a, b, n_boot=2000, seed=0):
    """Bootstrap 95% CI on the difference in means (a - b)."""
    if not a or not b:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (rng.choice(a, len(a), replace=True).mean()
                    - rng.choice(b, len(b), replace=True).mean())
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=800, help="most liquid N tickers")
    ap.add_argument("--months", type=int, default=6, help="window length")
    ap.add_argument("--cost", type=float, default=0.20, help="round-trip cost %%")
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    a = ap.parse_args()

    if a.market == "us":
        from params_us import GATES as G0
    else:
        from params import GATES as G0

    print("Walk-forward adaptive tuning")
    print(f"  windows of {a.months} months, cost {a.cost}% per round trip\n")

    frames = _load_frames(a.market)
    liq = []
    for t, d in frames.items():
        if len(d) < 400:
            continue
        liq.append((t, float((d["Close"] * d["Volume"]).tail(250).median())))
    liq.sort(key=lambda x: -x[1])
    picks = [t for t, _ in liq[:a.top]]
    print(f"  building indicators for {len(picks):,} tickers...")

    built = {}
    for n, t in enumerate(picks, 1):
        try:
            built[t] = ind.build(frames[t])
        except Exception:
            pass
        if n % 200 == 0:
            print(f"    {n:,}/{len(picks):,}")
    print(f"  {len(built):,} ready\n")

    all_dates = pd.DatetimeIndex(sorted({d.index.min() for d in built.values()}))
    start = max(pd.Timestamp("2020-07-01"), all_dates.min())
    end_all = max(d.index.max() for d in built.values())
    bounds = pd.date_range(start, end_all, freq=f"{a.months}MS")
    if len(bounds) < 4:
        raise SystemExit("Not enough history for walk-forward.")

    G_live = G0            # adaptive: changes as it learns
    G_frozen = G0          # control: never changes
    adopted, survived, tested = 0, 0, 0
    live_rets, frozen_rets = [], []
    log = []

    print(f"  {'window':<22}{'adopted change':<34}{'in-samp':>9}{'next-win':>10}{'held?':>7}")
    print("  " + "-" * 84)

    for w in range(len(bounds) - 2):
        tr_s, tr_e = bounds[w], bounds[w + 1]
        te_s, te_e = bounds[w + 1], bounds[w + 2]

        base_mean, base_n, base_rets = evaluate(built, G_live, tr_s, tr_e, cost=a.cost)
        if base_n < 20:
            continue

        # --- try one change at a time, in sample ---
        best = None
        for field, alts in VARIANTS.items():
            for val in alts:
                G_try = dataclasses.replace(G_live, **{field: val})
                m, n, rr = evaluate(built, G_try, tr_s, tr_e, cost=a.cost)
                if n < 20 or not np.isfinite(m):
                    continue
                lo, hi = boot_ci(rr, base_rets)
                real = lo > 0                       # CI must exclude zero
                if real and (best is None or m - base_mean > best["gain"]):
                    best = {"field": field, "val": val, "gain": m - base_mean,
                            "G": G_try, "lo": lo, "hi": hi, "n": n}

        # --- frozen control on the test window ---
        fm, fn, fr = evaluate(built, G_frozen, te_s, te_e, cost=a.cost)
        if fr:
            frozen_rets.extend(fr)

        if best is None:
            lm, ln, lr = evaluate(built, G_live, te_s, te_e, cost=a.cost)
            if lr:
                live_rets.extend(lr)
            log.append({"window": str(tr_s.date()), "change": None})
            print(f"  {str(tr_s.date()):<22}{'(nothing beat noise)':<34}"
                  f"{'—':>9}{'—':>10}{'—':>7}")
            continue

        # --- does the adopted change survive on unseen data? ---
        adopted += 1
        old_te, _, old_rets = evaluate(built, G_live, te_s, te_e, cost=a.cost)
        new_te, new_n, new_rets = evaluate(built, best["G"], te_s, te_e, cost=a.cost)
        held = np.isfinite(new_te) and np.isfinite(old_te) and new_te > old_te
        tested += 1
        survived += int(held)

        G_live = best["G"]                     # adopt regardless, as you described
        if new_rets:
            live_rets.extend(new_rets)

        label = f"{best['field']} -> {best['val']}"
        log.append({"window": str(tr_s.date()), "change": label,
                    "in_sample_gain": best["gain"],
                    "oos_gain": float(new_te - old_te) if np.isfinite(new_te) else None,
                    "held": bool(held)})
        print(f"  {str(tr_s.date()):<22}{label:<34}{best['gain']:>+8.2f}%"
              f"{(new_te - old_te if np.isfinite(new_te) else float('nan')):>+9.2f}%"
              f"{('YES' if held else 'no'):>7}")

    # ---------------- verdict ----------------
    print("\n  " + "=" * 70)
    if tested == 0:
        print("  No change ever beat its bootstrap interval. The optimizer found")
        print("  nothing to adopt, which is itself a clean result: on this data")
        print("  no single parameter change is distinguishable from noise.")
        return

    rate = survived / tested * 100
    print(f"  Changes adopted (looked good in-sample) : {adopted}")
    print(f"  Of those, still better on the NEXT window: {survived}  ({rate:.0f}%)")
    print()
    if rate >= 65:
        print("  -> Adaptation is finding something real. Worth pursuing.")
    elif rate <= 40:
        print("  -> Adapting is WORSE than leaving the parameters alone.")
        print("     The optimizer is reliably fitting noise.")
    else:
        print("  -> Around a coin flip. The optimizer adds nothing over never")
        print("     changing anything. Every 'improvement' it found was luck.")

    if live_rets and frozen_rets:
        lm, fm = float(np.mean(live_rets)), float(np.mean(frozen_rets))
        lo, hi = boot_ci(live_rets, frozen_rets)
        print(f"\n  Adaptive  mean net return/signal : {lm:+.3f}%  ({len(live_rets):,} trades)")
        print(f"  Frozen    mean net return/signal : {fm:+.3f}%  ({len(frozen_rets):,} trades)")
        print(f"  Difference 95% CI                : {lo:+.3f}% to {hi:+.3f}%")
        if lo <= 0 <= hi:
            print("  -> interval spans zero: adaptive and frozen are indistinguishable.")
        elif lo > 0:
            print("  -> adaptive genuinely beat frozen.")
        else:
            print("  -> frozen genuinely beat adaptive.")

    n_tests = len(VARIANTS) * 2
    print(f"\n  Multiple comparisons: {n_tests} tests per window x {adopted + 1} windows")
    print(f"  At a 5% threshold you would expect ~{n_tests * (adopted + 1) * 0.05:.1f} false")
    print(f"  discoveries by chance alone. Weigh the adoption count against that.")

    (HERE / "walkforward_log.json").write_text(json.dumps(log, indent=2))
    print(f"\n  Full log: walkforward_log.json")


if __name__ == "__main__":
    main()
