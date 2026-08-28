"""
mlmodel.py — LightGBM meta-labelling on the surviving rule, validated honestly.

THE ARCHITECTURE, AND WHY IT IS SHAPED THIS WAY

The eight gates were eight AND-ed thresholds: an axis-aligned box in R^8, VC
dimension 16, one representable boundary, one bit of output. It cannot express
interactions, diagonal boundaries or confidence. That was a bad design.

This replaces it with:

  PRIMARY RULE (unchanged)  Close below lower Bollinger band. This is the only
                            rule that cleared p<0.05 in BOTH decades. It picks
                            the CANDIDATES and fixes the direction.

  META MODEL                LightGBM scores each candidate on continuous
                            features. Output is a graded score, not a gate, so
                            you can rank and take the best N. That alone uses
                            breadth far better than yes/no.

Meta-labelling (Lopez de Prado, ch. 3) is the right shape here: predicting
"which of these candidates work" is a far easier problem than predicting
direction from scratch, and the primary rule already carries the direction.

WHY LIGHTGBM AND NOT A NEURAL NET
Tabular, low signal-to-noise, ~6k samples. Tree ensembles are state of the art
on medium tabular data and the gap survives tuning. A 2x128 MLP has ~50k
parameters against ~6k samples; that memorises. Depth buys nothing because
these features have no compositional structure.

VALIDATION
  * purged + embargoed CV — labels are 20-day forward returns and overlap, so
    ordinary k-fold puts the answer in the training set
  * out-of-fold IC, not in-sample fit
  * a permutation control: shuffle the labels, refit the whole pipeline, and
    see what IC the model reaches on data with nothing in it
  * both decades

THE CEILING, STATED UP FRONT
The strongest effect measured in this data is IC 0.044, which is R^2 of 0.0019.
By Grinold, IR ~= IC*sqrt(breadth); at ~250 trades a year with realistic
cross-correlation that caps out near IR 0.2-0.3 before costs. A better model
cannot exceed the information in the features. If this pipeline reports much
more than that, suspect leakage rather than success.

    python mlmodel.py --top 600
    python mlmodel.py --top 600 --cache-tag _holdout
    python mlmodel.py --top 600 --perms 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import indicators as ind
from basics import load
from purged_cv import purged_train_test

HERE = Path(__file__).parent
HOLD = 20

FEATURES = [
    "depth_below_bb",    # how far below the lower band, in ATRs
    "bb_width",          # band width as % of price
    "bb_squeeze",        # band width percentile vs own 6-month history
    "stoch_k", "stoch_d", "stoch_spread",
    "macd_n", "macd_hist_n",
    "mtm_n",
    "dist_ema8", "dist_ema21", "dist_ema100",
    "rel_vol",
    "atr_pct",
    "ret_5", "ret_20", "ret_60", "mom_12_1",
    "log_turnover",
]


def build_samples(built, G):
    """One row per BB-lower signal: continuous features + forward return."""
    rows = []
    for t, df in built.items():
        c = df["Close"].to_numpy(float)
        o = df["Open"].to_numpy(float)
        lo = df["bb_lower"].to_numpy(float)
        up = df["bb_upper"].to_numpy(float)
        mid = df["bb_mid"].to_numpy(float)
        atr = df["atr"].to_numpy(float)
        turn = df["turnover_avg"].to_numpy(float)
        idx = df.index
        n = len(c)
        g = {k: df[k].to_numpy(float) for k in
             ("stoch_k", "stoch_d", "macd", "macd_hist", "mtm", "ema_fast",
              "ema_mid", "ema_slow", "rel_vol", "atr_pct",
              "bb_squeeze_pct")}
        last = -10 ** 9
        for i in range(150, n - HOLD - 2):
            if not (c[i] < lo[i]) or i - last < 20:
                continue
            if not np.isfinite(turn[i]) or turn[i] < G.min_avg_turnover_sgd:
                continue
            if c[i] < G.min_price_sgd:
                continue
            e, x = o[i + 1], c[i + 1 + HOLD]
            if not (np.isfinite(e) and e > 0 and np.isfinite(x)):
                continue
            a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else np.nan
            row = {
                "ticker": t, "date": idx[i + 1], "pos": i,
                "y": (x / e - 1) * 100,
                "depth_below_bb": (lo[i] - c[i]) / a if a == a else np.nan,
                "bb_width": (up[i] - lo[i]) / c[i] * 100,
                "bb_squeeze": g["bb_squeeze_pct"][i],
                "stoch_k": g["stoch_k"][i], "stoch_d": g["stoch_d"][i],
                "stoch_spread": g["stoch_k"][i] - g["stoch_d"][i],
                "macd_n": g["macd"][i] / c[i] * 100,
                "macd_hist_n": g["macd_hist"][i] / c[i] * 100,
                "mtm_n": g["mtm"][i] / c[i] * 100,
                "dist_ema8": (c[i] / g["ema_fast"][i] - 1) * 100,
                "dist_ema21": (c[i] / g["ema_mid"][i] - 1) * 100,
                "dist_ema100": (c[i] / g["ema_slow"][i] - 1) * 100,
                "rel_vol": g["rel_vol"][i],
                "atr_pct": g["atr_pct"][i],
                "ret_5": (c[i] / c[i - 5] - 1) * 100,
                "ret_20": (c[i] / c[i - 20] - 1) * 100,
                "ret_60": (c[i] / c[i - 60] - 1) * 100,
                "mom_12_1": (c[i - 21] / c[i - 252] - 1) * 100 if i >= 252 else np.nan,
                "log_turnover": np.log10(max(turn[i], 1.0)),
            }
            rows.append(row)
            last = i
    d = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return d.dropna(subset=["y"])


def oof_predict(d, params, n_splits=6, embargo_frac=0.02, shuffle_y=False,
                seed=0):
    """Out-of-fold predictions under purged, embargoed CV."""
    import lightgbm as lgb
    X = d[FEATURES].to_numpy(float)
    y = d["y"].to_numpy(float)
    if shuffle_y:
        y = np.random.default_rng(seed).permutation(y)
    n = len(d)
    # label of a sample ends HOLD bars later; in row terms, find the row whose
    # date is >= this row's date + HOLD trading days
    end_dates = pd.DatetimeIndex(d["date"]) + pd.Timedelta(days=HOLD * 1.45)
    t1_pos = np.searchsorted(pd.DatetimeIndex(d["date"]).values,
                             end_dates.values).astype(int)
    t1_pos = np.minimum(t1_pos, n - 1)
    embargo = max(1, int(n * embargo_frac))

    preds = np.full(n, np.nan)
    bounds = [(k * n // n_splits, (k + 1) * n // n_splits)
              for k in range(n_splits)]
    for a, b in bounds:
        tr, te = purged_train_test(n, t1_pos, a, b, embargo)
        if len(tr) < 300:
            continue
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    ok = ~np.isnan(preds)
    if ok.sum() < 100:
        return None, None, None
    ic = float(pd.Series(preds[ok]).corr(pd.Series(y[ok]), method="spearman"))
    return ic, preds, ok


def decile_spread(preds, y, ok, q=5):
    p = pd.Series(preds[ok])
    yy = pd.Series(y[ok])
    try:
        b = pd.qcut(p, q, labels=False, duplicates="drop")
    except ValueError:
        return np.nan, np.nan, np.nan
    top = yy[b == b.max()].mean()
    bot = yy[b == b.min()].mean()
    return float(top), float(bot), float(top - bot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=600)
    ap.add_argument("--cache-tag", default="")
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    ap.add_argument("--perms", type=int, default=20,
                    help="label-shuffled refits for the null distribution")
    ap.add_argument("--splits", type=int, default=6)
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
    print(f"  {len(d):,} candidate signals, {len(FEATURES)} features")
    print(f"  n/p = {len(d)/len(FEATURES):,.0f}  (want >100 for low SNR)\n")

    # heavy regularisation: shallow trees, strong shrinkage, subsampling
    params = dict(n_estimators=400, learning_rate=0.02, num_leaves=7,
                  max_depth=3, min_child_samples=100, subsample=0.7,
                  subsample_freq=1, colsample_bytree=0.6,
                  reg_alpha=1.0, reg_lambda=5.0, verbose=-1, n_jobs=-1)

    ic, preds, ok = oof_predict(d, params, n_splits=a.splits)
    if ic is None:
        raise SystemExit("Too few out-of-fold predictions.")
    y = d["y"].to_numpy(float)
    top, bot, spread = decile_spread(preds, y, ok)

    print(f"  OUT-OF-FOLD, purged + embargoed")
    print(f"    IC (Spearman)        {ic:+.4f}")
    print(f"    top quintile mean    {top:+.2f}%")
    print(f"    bottom quintile mean {bot:+.2f}%")
    print(f"    spread               {spread:+.2f}%")
    print(f"    all candidates mean  {y[ok].mean():+.2f}%")

    if a.perms > 0:
        print(f"\n  {a.perms} label-shuffled refits (the null)...")
        nulls = []
        for s in range(a.perms):
            i2, p2, o2 = oof_predict(d, params, n_splits=a.splits,
                                     shuffle_y=True, seed=s)
            if i2 is not None:
                nulls.append(i2)
            print(f"    {s+1}/{a.perms}", end="\r")
        print(" " * 20, end="\r")
        nulls = np.asarray(nulls)
        beat = int((nulls >= ic).sum())
        p = (beat + 1) / (len(nulls) + 1)
        print(f"\n    shuffled IC: mean {nulls.mean():+.4f}, "
              f"max {nulls.max():+.4f}")
        print(f"    shuffled runs >= real: {beat}/{len(nulls)}")
        print(f"    p = {p:.4f}")
        if p > 0.05:
            print(f"    -> the model finds no more than it finds in noise.")
        else:
            print(f"    -> the model beats its own noise floor.")

    # feature importance, fitted on everything (descriptive only)
    import lightgbm as lgb
    m = lgb.LGBMRegressor(**params).fit(d[FEATURES], d["y"])
    imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values(
        ascending=False)
    print(f"\n  Top features (descriptive, in-sample):")
    for k, v in imp.head(8).items():
        print(f"    {k:<18}{v:>6.0f}")

    print(f"\n  Ceiling check: IC {ic:+.4f} implies IR ~ "
          f"{abs(ic)*np.sqrt(250/ (1+19*0.4)):.2f} at 250 trades/yr with")
    print(f"  realistic cross-correlation. Anything above ~0.5 here would be")
    print(f"  surprising enough to suspect leakage rather than skill.")

    (HERE / f"mlmodel{a.cache_tag or '_recent'}.json").write_text(json.dumps(
        {"n": len(d), "ic": ic, "top": top, "bottom": bot, "spread": spread},
        indent=2, default=float))
    print("  Saved.")


if __name__ == "__main__":
    main()
