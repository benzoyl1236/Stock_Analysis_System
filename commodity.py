"""
commodity.py — run the surviving rule on gold, metals and commodities.

WHY NOT JUST XAUUSD ON ITS OWN
One instrument produces roughly 20-60 signals over six years. With return
volatility around 9%, the standard error on the mean of 50 trades is 1.3%,
so you can only detect effects larger than about 2.5%. The effect you are
looking for is under 1%. Testing gold alone cannot answer the question — it
can only produce a backtest you have no way to evaluate.

So this uses a BASKET: 40 instruments spanning metals, energy, agriculture,
broad commodity indices and miners. That gives cross-sectional sample the same
way 600 stocks did, while still being a genuinely different asset class with
different participants and different drivers.

You can still look at gold alone afterwards — the per-instrument breakdown
shows it — but the statistical verdict comes from the basket.

WHAT CHANGES, AND WHAT DOES NOT
Signal gates are identical to the US profile. Only the tradability floors
move: turnover $20M -> $1M and price $5 -> $1, because commodity ETFs are
smaller and futures quote differently. params_commodity.py carries its own
fingerprint so results can never be mixed with equity results.

DATA NOTES
  * spot XAUUSD has no meaningful volume on Yahoo, so the VOLUME gate cannot
    work on it. GLD and GC=F both carry real volume.
  * futures tickers (GC=F etc.) are continuous contracts — Yahoo splices them,
    which introduces roll artefacts. ETFs are cleaner for a first pass.
  * some tickers may fail; that is expected and reported.

    python commodity.py --fetch
    python commodity.py --test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
TAG = "_commodity"
GOLD_TAG = "_gold"
XAU_TAG = "_xau"


def fetch(years: int, gold: bool = False, xau: bool = False):
    import data_us
    uni = HERE / ("universe_xau.csv" if xau else
                  "universe_gold.csv" if gold else "universe_commodity.csv")
    if not uni.exists():
        raise SystemExit(f"{uni.name} missing — download it into this folder")
    tickers = pd.read_csv(uni)["ticker"].dropna().astype(str).tolist()
    print(f"Fetching {len(tickers)} commodity instruments, {years}y")
    frames = data_us.load_bulk(tickers, years=years,
                               tag=(XAU_TAG if xau else GOLD_TAG if gold else TAG),
                               min_bars=300,
                               batch_size=40)
    print(f"\n  {len(frames)} usable. Now: python commodity.py --test")


def test(perms: int, cost: float, gold: bool = False):
    import data_us
    import indicators as ind
    from basics import collect, ptest
    from params_commodity import GATES as G

    tg = GOLD_TAG if gold else TAG
    files = sorted(data_us.CACHE_DIR.glob(f"us{tg}_*.parquet"))
    if not files:
        raise SystemExit("No commodity cache. Run: python commodity.py --fetch")
    print(f"  reading {files[-1].name}")
    frames = data_us._unflatten(pd.read_parquet(files[-1]))
    built = {}
    for t, d in frames.items():
        if len(d) < 400:
            continue
        try:
            built[t] = ind.build(d)
        except Exception:
            pass
    print(f"  {len(built)} instruments with enough history\n")
    if len(built) < 10:
        print("  WARNING: fewer than 10 instruments. Cross-sectional sample is")
        print("  thin and the p-values below will be weak.\n")

    data = collect(built, G, cost)
    res = {k: ptest(v, perms) for k, v in data.items() if len(v["real"]) >= 100}

    print(f"  {'rule':<32}{'n':>8}{'mean':>9}{'vs random':>11}{'p':>8}")
    print("  " + "-" * 70)
    for name in sorted(res, key=lambda x: res[x]["p"]):
        s = res[name]
        print(f"  {name:<32}{s['n']:>8,}{s['mean']:>8.2f}%"
              f"{s['edge']:>10.2f}%{s['p']:>8.3f}")

    # gold specifically, for interest — not for inference
    print(f"\n  Gold instruments alone (too few signals for a verdict):")
    for t in ("GLD", "IAU", "GC=F"):
        if t not in built:
            continue
        df = built[t]
        c = df["Close"].to_numpy(float)
        o = df["Open"].to_numpy(float)
        lo = df["bb_lower"].to_numpy(float)
        hits, last = [], -10 ** 9
        for i in range(150, len(c) - 22):
            if c[i] < lo[i] and i - last >= 20:
                e, x = o[i + 1], c[i + 21]
                if np.isfinite(e) and e > 0 and np.isfinite(x):
                    hits.append((x / e - 1) * 100 - cost)
                    last = i
        if hits:
            a = np.array(hits)
            print(f"    {t:<8} {len(a):>4} signals   mean {a.mean():+.2f}%   "
                  f"win {(a > 0).mean()*100:.0f}%")
    print(f"\n  Those per-instrument numbers are for curiosity only. With a few")
    print(f"  dozen signals the error bar is wider than any effect worth having.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--years", type=int, default=15)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--cost", type=float, default=0.20)
    ap.add_argument("--xau", action="store_true",
                    help="spot XAUUSD and close proxies only")
    ap.add_argument("--gold", action="store_true",
                    help="gold complex only: metal, miners, related metals")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.years, a.gold, a.xau)
    elif a.test:
        test(a.perms, a.cost, a.gold)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()