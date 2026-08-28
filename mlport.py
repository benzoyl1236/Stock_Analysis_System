"""
mlport.py — turn the model's IC into an equity curve.

An information coefficient is not a return. This takes the out-of-fold scores
from mlmodel.py, keeps only the best-scoring slice, and runs them as a
portfolio you could actually hold: limited slots, real costs, compounding,
signals skipped when nothing is free.

THREE THINGS ARE COMPARED, AND ALL THREE MATTER

  ALL SIGNALS      every BB-lower candidate, unranked. This is bbtest.py.
  TOP QUINTILE     only the best-scoring 20% by out-of-fold model score.
  EQUAL WEIGHT     just holding the same universe. The thing you would do
                   instead of any of this.

The model earns its place only if TOP QUINTILE beats ALL SIGNALS by more than
noise, AND beats EQUAL WEIGHT by more than the measured benchmark bias.

SCORES ARE OUT-OF-FOLD, WHICH IS THE WHOLE POINT
Every score used here was produced by a model that never saw that sample
during training, under purged and embargoed CV. Ranking on in-sample scores
would be circular and would look spectacular.

    python mlport.py --top 600 --slots 20 --quantile 0.2
    python mlport.py --top 600 --slots 20 --cache-tag _holdout
    python mlport.py --top 600 --slots 20 --cost 0.35     realistic fills
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import indicators as ind
from basics import load
from mlmodel import build_samples, oof_predict, FEATURES, HOLD

HERE = Path(__file__).parent


def portfolio(sig, capital, slots, cost, seed=None):
    """Compound through a dated signal stream with limited slots."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        sig = sig.iloc[rng.permutation(len(sig))].sort_values(
            "date", kind="stable").reset_index(drop=True)
    else:
        sig = sig.sort_values("date").reset_index(drop=True)
    eq = float(capital)
    free_at, taken = [], []
    for _, r in sig.iterrows():
        d = r["date"]
        free_at = [t for t in free_at if t > d]
        if len(free_at) >= slots:
            continue
        stake = eq / slots
        eq += stake * (r["y"] / 100.0 - cost / 100.0)
        free_at.append(d + pd.Timedelta(days=HOLD * 1.45))
        taken.append({"date": d, "ret": r["y"] - cost, "eq": eq})
    if not taken:
        return None
    t = pd.DataFrame(taken)
    yrs = (t["date"].max() - t["date"].min()).days / 365.25
    curve = t.set_index("date")["eq"]
    peak = curve.cummax()
    return {"trades": len(t), "years": yrs, "final": eq,
            "cagr": ((eq / capital) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0,
            "max_dd": float(((curve - peak) / peak * 100).min()),
            "win": float((t["ret"] > 0).mean() * 100),
            "avg": float(t["ret"].mean()),
            "per_year": len(t) / yrs if yrs > 0 else 0.0,
            "table": t}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=600)
    ap.add_argument("--capital", type=float, default=50_000)
    ap.add_argument("--slots", type=int, default=20)
    ap.add_argument("--cost", type=float, default=0.20)
    ap.add_argument("--quantile", type=float, default=0.2,
                    help="fraction of best-scoring signals to keep")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--splits", type=int, default=6)
    ap.add_argument("--cache-tag", default="")
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    ap.add_argument("--drop-momentum", action="store_true",
                    help="remove mom_12_1, which failed its own holdout")
    a = ap.parse_args()

    if a.market == "us":
        from params_us import GATES as G
    else:
        from params import GATES as G

    frames = load(a.cache_tag, a.market)
    liq = sorted(((t, float((d["Close"] * d["Volume"]).tail(250).median()))
                  for t, d in frames.items() if len(d) >= 400), key=lambda x: -x[1])
    built = {}
    for t in [x for x, _ in liq[:a.top]]:
        try:
            built[t] = ind.build(frames[t])
        except Exception:
            pass
    print(f"    {len(built):,} tickers")

    d = build_samples(built, G)
    feats = [f for f in FEATURES if not (a.drop_momentum and f == "mom_12_1")]
    if a.drop_momentum:
        import mlmodel
        mlmodel.FEATURES = feats
        print(f"  mom_12_1 dropped — {len(feats)} features")
    print(f"  {len(d):,} candidate signals\n")

    params = dict(n_estimators=400, learning_rate=0.02, num_leaves=7,
                  max_depth=3, min_child_samples=100, subsample=0.7,
                  subsample_freq=1, colsample_bytree=0.6,
                  reg_alpha=1.0, reg_lambda=5.0, verbose=-1, n_jobs=-1)
    ic, preds, ok = oof_predict(d, params, n_splits=a.splits)
    if ic is None:
        raise SystemExit("Too few out-of-fold predictions.")
    print(f"  out-of-fold IC {ic:+.4f}\n")

    d = d.loc[ok].copy()
    d["score"] = preds[ok]
    cut = d["score"].quantile(1 - a.quantile)
    top = d[d["score"] >= cut]

    # equal-weight benchmark, rebalanced daily
    daily = pd.DataFrame({t_: df["Close"].pct_change() for t_, df in built.items()})
    ew = daily.mean(axis=1).dropna()
    ewc = (1 + ew).cumprod()
    yrs = (ew.index[-1] - ew.index[0]).days / 365.25
    bh = (ewc.iloc[-1] ** (1 / yrs) - 1) * 100
    bh_dd = float(((ewc - ewc.cummax()) / ewc.cummax() * 100).min())

    rows = []
    for label, sub in (("all signals", d), (f"top {a.quantile:.0%} by model", top)):
        r = portfolio(sub, a.capital, a.slots, a.cost)
        if r:
            rows.append((label, r, len(sub)))

    print(f"  ${a.capital:,.0f}, {a.slots} slots, {a.cost}% round trip\n")
    print(f"  {'strategy':<26}{'signals':>9}{'trades/yr':>11}{'avg':>8}"
          f"{'CAGR':>9}{'maxDD':>9}{'win%':>7}")
    print("  " + "-" * 79)
    for label, r, n in rows:
        print(f"  {label:<26}{n:>9,}{r['per_year']:>11,.0f}{r['avg']:>7.2f}%"
              f"{r['cagr']:>8.1f}%{r['max_dd']:>8.1f}%{r['win']:>6.1f}%")
    print(f"  {'equal-weight universe':<26}{'—':>9}{'—':>11}{'—':>8}"
          f"{bh:>8.1f}%{bh_dd:>8.1f}%{'—':>7}")

    NOISE = 2.0
    if len(rows) == 2:
        allr, topr = rows[0][1], rows[1][1]
        d_model = topr["cagr"] - allr["cagr"]
        d_bench = topr["cagr"] - bh
        print(f"\n  model ranking vs taking everything : {d_model:+.1f}pp")
        print(f"  model ranking vs equal weight      : {d_bench:+.1f}pp"
              f"   (noise floor {NOISE:.0f}pp)")
        if d_model > 0 and d_bench > NOISE:
            print(f"  -> the model earns its place AND beats doing nothing.")
        elif d_model > 0:
            print(f"  -> ranking helps, but the whole thing still does not")
            print(f"     clearly beat just holding the universe.")
        else:
            print(f"  -> ranking does not help. Taking every signal was better.")

    # year by year for the ranked version
    if len(rows) == 2:
        t = rows[1][1]["table"].copy()
        t["yr"] = pd.DatetimeIndex(t["date"]).year
        print(f"\n  Top-quintile portfolio, year by year")
        print(f"  {'year':<8}{'trades':>8}{'avg':>9}{'end $':>13}{'yr %':>9}")
        print("  " + "-" * 47)
        prev = a.capital
        for y, g in t.groupby("yr"):
            end = g["eq"].iloc[-1]
            print(f"  {y:<8}{len(g):>8,}{g['ret'].mean():>8.2f}%"
                  f"{end:>13,.0f}{(end/prev-1)*100:>8.1f}%")
            prev = end

    if a.trials > 0 and len(rows) == 2:
        print(f"\n  Robustness: {a.trials} runs reshuffling which signals get taken")
        for label, sub in (("all", d), ("top", top)):
            cs = [portfolio(sub, a.capital, a.slots, a.cost, seed=s)["cagr"]
                  for s in range(a.trials)]
            cs = np.asarray(cs)
            print(f"    {label:<5} median {np.median(cs):+6.1f}%   "
                  f"range {cs.min():+.1f}% to {cs.max():+.1f}%   "
                  f"beats EW+{NOISE:.0f}pp in {(cs > bh + NOISE).mean()*100:.0f}%")

    (HERE / f"mlport{a.cache_tag or '_recent'}.json").write_text(json.dumps(
        {"ic": ic, "bench": bh,
         "rows": [{"label": l, **{k: v for k, v in r.items() if k != "table"}}
                  for l, r, _ in rows]}, indent=2, default=float))
    print("\n  Saved.")


if __name__ == "__main__":
    main()
