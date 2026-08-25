"""
rank.py — can you tell the good signals from the bad ones in advance?

THE IDEA
Your slots are limited, so you take ~6% of signals and arrival order decides
which. If signal quality varies predictably, ranking beats arrival order and
the wide outcome spread narrows.

THE TRAP, AND THE GUARDRAIL
Test ten features on one dataset and the best will look good by chance. That
is how backtests get fooled, and it is precisely what your locked-parameter
fingerprint exists to prevent. So this script splits your history:

    TRAIN  the earlier 60% of signals — features are ranked here
    TEST   the later 40% — never consulted until the ranking is fixed

A feature that works in TRAIN and fails in TEST was noise. That is the normal
outcome and you should expect it. Only a feature that holds up in TEST is
worth anything, and even then it is one test.

The split is chronological, not random. Random splitting would leak: two
signals from the same week in the same market regime would land on both sides
and inflate the result.

    python rank.py
    python rank.py --hold 20 --top 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

FEATURES = [
    "stoch_k", "macd_hist", "mtm", "rel_vol", "bb_pos", "bb_squeeze",
    "dist_ema21", "dist_ema100", "turnover_avg", "atr_pct", "trend_age",
]


def _ic(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation between a feature and forward return.

    Rank correlation, not linear — we only care whether higher feature means
    higher return in ORDER, which is all a ranking uses.
    """
    if x.nunique() < 5 or len(x) < 30:
        return np.nan
    return float(x.rank().corr(y.rank()))


def _decile_spread(x: pd.Series, y: pd.Series, q: int = 5) -> float:
    """Median return of the top bucket minus the bottom bucket."""
    try:
        b = pd.qcut(x, q, labels=False, duplicates="drop")
    except ValueError:
        return np.nan
    if b is None or pd.Series(b).nunique() < 2:
        return np.nan
    d = pd.DataFrame({"b": b, "y": y})
    top = d[d["b"] == d["b"].max()]["y"].median()
    bot = d[d["b"] == d["b"].min()]["y"].median()
    return float(top - bot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=20, choices=[5, 10, 20])
    ap.add_argument("--top", type=int, default=20,
                    help="%% of signals a ranking strategy would take")
    ap.add_argument("--file", default=None)
    a = ap.parse_args()

    path = Path(a.file) if a.file else HERE / "output" / "backtest_trades.csv"
    if not path.exists():
        raise SystemExit(f"No trades file at {path}")

    tr = pd.read_csv(path)
    ycol = f"ret_{a.hold}"
    missing = [f for f in FEATURES if f not in tr.columns]
    if missing:
        raise SystemExit(
            f"Trades file lacks ranking features: {', '.join(missing)}\n"
            f"Re-run: python scan.py --market us --backtest")

    tr = tr.dropna(subset=[ycol, "entry_date"]).copy()
    tr["entry_date"] = pd.to_datetime(tr["entry_date"], errors="coerce")
    tr = tr.dropna(subset=["entry_date"]).sort_values("entry_date")

    cut = int(len(tr) * 0.6)
    train, test = tr.iloc[:cut], tr.iloc[cut:]
    print(f"Signal ranking, {a.hold}-day return\n")
    print(f"  TRAIN {len(train):,} signals  "
          f"{train['entry_date'].min().date()} to {train['entry_date'].max().date()}")
    print(f"  TEST  {len(test):,} signals  "
          f"{test['entry_date'].min().date()} to {test['entry_date'].max().date()}")

    rows = []
    for f in FEATURES:
        rows.append({
            "feature": f,
            "ic_train": _ic(train[f], train[ycol]),
            "ic_test": _ic(test[f], test[ycol]),
            "spread_train": _decile_spread(train[f], train[ycol]),
            "spread_test": _decile_spread(test[f], test[ycol]),
        })
    r = pd.DataFrame(rows).sort_values("ic_train", key=abs, ascending=False)

    print(f"\n  {'feature':<16}{'IC train':>10}{'IC test':>10}"
          f"{'spread train':>15}{'spread test':>14}{'holds up?':>12}")
    print("  " + "-" * 77)
    for _, x in r.iterrows():
        holds = (np.isfinite(x["ic_train"]) and np.isfinite(x["ic_test"])
                 and abs(x["ic_train"]) > 0.03
                 and np.sign(x["ic_train"]) == np.sign(x["ic_test"])
                 and abs(x["ic_test"]) > 0.02)
        print(f"  {x['feature']:<16}{x['ic_train']:>10.3f}{x['ic_test']:>10.3f}"
              f"{x['spread_train']:>14.2f}%{x['spread_test']:>13.2f}%"
              f"{('YES' if holds else 'no'):>12}")

    survivors = [x["feature"] for _, x in r.iterrows()
                 if np.isfinite(x["ic_train"]) and np.isfinite(x["ic_test"])
                 and abs(x["ic_train"]) > 0.03
                 and np.sign(x["ic_train"]) == np.sign(x["ic_test"])
                 and abs(x["ic_test"]) > 0.02]

    print(f"\n  IC = rank correlation between the feature and forward return.")
    print(f"  |IC| below ~0.03 is noise. Even 0.05 is a weak but usable signal.")
    print(f"  'Holds up' means same sign in TRAIN and TEST, and not tiny.")

    if not survivors:
        print(f"\n  RESULT: no feature survived out-of-sample.")
        print(f"  Ranking will not help. The signals are not distinguishable")
        print(f"  in advance, which means arrival order is as good as anything.")
        return

    print(f"\n  RESULT: {len(survivors)} feature(s) held up: {', '.join(survivors)}")

    # What would taking the best N% by the top surviving feature have done?
    best = survivors[0]
    sign = np.sign(r[r["feature"] == best]["ic_train"].iloc[0])
    for label, part in (("TRAIN", train), ("TEST", test)):
        s = part.copy()
        s["score"] = s[best] * sign
        n = max(10, int(len(s) * a.top / 100))
        picked = s.nlargest(n, "score")
        print(f"\n  {label}: top {a.top}% by {best}  ({len(picked):,} signals)")
        print(f"    median {picked[ycol].median():+.2f}%  vs "
              f"all-signal median {part[ycol].median():+.2f}%  "
              f"(improvement {picked[ycol].median() - part[ycol].median():+.2f}pp)")

    print(f"\n  Only the TEST line counts. TRAIN is where the feature was chosen,")
    print(f"  so it is guaranteed to look good there.")


if __name__ == "__main__":
    main()
