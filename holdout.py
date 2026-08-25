"""
holdout.py — test the reversal finding on data neither of us has seen.

THE PRE-REGISTERED HYPOTHESIS
Written down BEFORE running this, from the rank.py result on 2021-2026 data:

    Among signals that pass all eight gates, those with LOWER 28-bar momentum
    (mtm) produce HIGHER forward 20-day returns.
    Measured on 2021-2026: IC = -0.080 out of sample.

PRE-REGISTERED PASS/FAIL — decided before seeing any result:

    PASS      IC is negative AND |IC| >= 0.03
    FAIL      IC is positive, or |IC| < 0.03
    Anything else is a FAIL. There is no "promising but".

Why this matters: the finding was DISCOVERED in 2021-2026 data. Testing it
again there proves nothing, because that is where it was found. This fetches
2010-2019 — a different decade, a different rate environment, a different
market structure — which the discovery never touched.

It also reports whether the eight gates add anything, by computing the same
IC on ALL eligible bars rather than only gated signals. If reversal works
equally well without the gates, the gates are decoration.

    python holdout.py --fetch --top 600
    python holdout.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import data_us
import fast_rules
import indicators as ind
from params_us import GATES as G, INDICATORS as I

HERE = Path(__file__).parent
TAG = "_holdout"
START, END = "2010-01-01", "2019-12-31"

IC_THRESHOLD = 0.03          # pre-registered
EXPECTED_SIGN = -1           # pre-registered: negative


def _ic(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 100:
        return np.nan, len(d)
    return float(d["x"].rank().corr(d["y"].rank())), len(d)


def gather(frames: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (gated signals, all eligible bars) with mtm and forward return."""
    sig_rows, all_rows = [], []
    for n, (t, raw) in enumerate(frames.items(), 1):
        try:
            df = ind.build(raw)
        except Exception:
            continue
        if len(df) < 200:
            continue

        gm = fast_rules.evaluate_all(df, I, G)
        fired = gm["FIRED"].to_numpy()
        openp = df["Open"].to_numpy(float)
        closep = df["Close"].to_numpy(float)
        mtm = df["mtm"].to_numpy(float)
        turn = df["turnover_avg"].to_numpy(float)

        last = -10 ** 9
        for i in range(150, len(df) - 22):
            if not np.isfinite(turn[i]) or turn[i] < G.min_avg_turnover_sgd:
                continue
            if closep[i] < G.min_price_sgd:
                continue
            entry = openp[i + 1]
            if not np.isfinite(entry) or entry <= 0 or not np.isfinite(mtm[i]):
                continue
            r20 = (closep[i + 21] / entry - 1) * 100
            # normalise momentum by price so it compares across tickers
            row = {"mtm": mtm[i] / closep[i] * 100, "ret_20": r20}
            all_rows.append(row)
            if fired[i] and i - last >= 20:
                sig_rows.append(row)
                last = i

        if n % 100 == 0:
            print(f"  {n:,}/{len(frames):,}  ({len(sig_rows):,} signals)")

    return pd.DataFrame(sig_rows), pd.DataFrame(all_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--top", type=int, default=600)
    a = ap.parse_args()

    if a.fetch:
        uni = HERE / "universe_us.csv"
        if not uni.exists():
            raise SystemExit("Run: python universe_us.py")
        tickers = pd.read_csv(uni)["ticker"].dropna().astype(str).tolist()[:a.top * 2]
        print(f"Fetching {START} to {END} for up to {len(tickers):,} tickers")
        print("Many will not have existed then; that is expected.")
        data_us.load_bulk(tickers, start=START, end=END, tag=TAG, min_bars=300)
        print("\nNow run: python holdout.py")
        return

    files = sorted(data_us.CACHE_DIR.glob(f"us{TAG}_*.parquet"))
    if not files:
        raise SystemExit("No holdout cache. Run: python holdout.py --fetch --top 600")
    print(f"  reading {files[-1].name}")
    frames = data_us._unflatten(pd.read_parquet(files[-1]))
    frames = {t: d for t, d in frames.items() if len(d) >= 300}
    print(f"  {len(frames):,} tickers with pre-2020 history\n")

    sig, allb = gather(frames)
    if sig.empty:
        raise SystemExit("No signals found in the holdout period.")

    ic_sig, n_sig = _ic(sig["mtm"], sig["ret_20"])
    ic_all, n_all = _ic(allb["mtm"], allb["ret_20"])
    se = 1 / math.sqrt(max(n_sig - 3, 4))

    print(f"\n  {'':<28}{'IC':>9}{'n':>10}{'std errors':>13}")
    print("  " + "-" * 60)
    print(f"  {'gated signals':<28}{ic_sig:>9.3f}{n_sig:>10,}{abs(ic_sig)/se:>12.2f}")
    print(f"  {'all eligible bars':<28}{ic_all:>9.3f}{n_all:>10,}"
          f"{abs(ic_all)/(1/math.sqrt(max(n_all-3,4))):>12.2f}")

    print(f"\n  Discovery sample (2021-2026): IC = -0.080")
    print(f"  Pre-registered: PASS needs IC negative and |IC| >= {IC_THRESHOLD}")

    passed = (np.isfinite(ic_sig) and np.sign(ic_sig) == EXPECTED_SIGN
              and abs(ic_sig) >= IC_THRESHOLD)
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'}")
    if passed:
        print("  The reversal effect replicated in a decade it was not found in.")
        print("  That is real evidence. It is still one replication, and the")
        print("  effect size is small relative to trading costs.")
    else:
        print("  It did not replicate. The 2021-2026 result was most likely a")
        print("  feature of that period. Do not trade it.")

    if np.isfinite(ic_all) and np.isfinite(ic_sig):
        print(f"\n  Do the gates add anything?")
        if abs(ic_all) >= abs(ic_sig) * 0.9:
            print(f"    No. Reversal is as strong on ALL bars ({ic_all:+.3f}) as on")
            print(f"    gated signals ({ic_sig:+.3f}). The eight gates are not")
            print(f"    contributing; low momentum alone carries it.")
        else:
            print(f"    Possibly. Reversal is stronger on gated signals "
                  f"({ic_sig:+.3f}) than on all bars ({ic_all:+.3f}).")


if __name__ == "__main__":
    main()
