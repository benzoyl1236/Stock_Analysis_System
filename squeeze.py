"""
squeeze.py — does a Bollinger squeeze predict a breakout big enough to trade?

THIS ASKS A DIFFERENT QUESTION FROM EVERYTHING BEFORE IT.

The eight-gate system and simple.py both asked: does the signal predict UP?
Answer: barely (+0.48% median), nowhere near enough to pay for an option that
needs ~5.8%.

This asks: does a squeeze predict a BIG MOVE, in either direction? That is the
question that actually matters for options, because a straddle (call + put)
profits from magnitude regardless of sign. Volatility clustering — quiet
periods preceding active ones — is one of the more reliable regularities in
markets, unlike direction.

It uses YOUR Bollinger setting (20, 2 SD). Band width is ranked against its own
previous 126 bars (~6 months), so a "squeeze" means this stock is quieter than
usual for this stock, not quiet compared to some other stock.

WHAT IS MEASURED
  * abs move    — |return| over the horizon. Direction-agnostic.
  * excursion   — largest absolute move at ANY point in the window, which is
                  what you could capture by closing early rather than holding
                  to expiry.
  * vol ratio   — realised volatility after vs before. Above 1.0 means the
                  squeeze really did precede expansion.
  * straddle    — buy a call AND a put. Profits if the move is large either way.

THE BIG CAVEAT, READ IT
  Premiums here are modelled from ATR at signal time, so during a squeeze the
  modelled option looks CHEAP. Real implied volatility is forward-looking and
  market makers can see the same squeeze you can. Real straddles into a known
  squeeze cost more than this model says — sometimes much more. Treat any edge
  shown here as an UPPER BOUND, and verify against real option chains before
  believing it.

    python squeeze.py --market us
    python squeeze.py --market us --pct 5      (tighter squeeze definition)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import indicators as ind
from baseline import _load_frames

HERE = Path(__file__).parent
HORIZON = 20
COOLDOWN = 20


def _stats(label: str, moves, exc, volr, prem) -> dict:
    moves, exc = np.asarray(moves), np.asarray(exc)
    volr, prem = np.asarray(volr), np.asarray(prem)
    # Straddle: pay ~2x the ATM call, profit on |move| beyond that.
    straddle_cost = prem * 2.0
    pnl_hold = (np.abs(moves) - straddle_cost) / straddle_cost * 100.0
    pnl_best = (exc - straddle_cost) / straddle_cost * 100.0
    return {
        "label": label,
        "n": int(len(moves)),
        "median_abs_move": float(np.median(np.abs(moves))),
        "p75_abs_move": float(np.percentile(np.abs(moves), 75)),
        "median_excursion": float(np.median(exc)),
        "median_vol_ratio": float(np.median(volr)),
        "median_breakeven": float(np.median(straddle_cost)),
        "straddle_hold_hit": float((np.abs(moves) > straddle_cost).mean() * 100),
        "straddle_hold_mean": float(np.mean(pnl_hold)),
        "straddle_best_hit": float((exc > straddle_cost).mean() * 100),
        "straddle_best_mean": float(np.mean(pnl_best)),
    }


def run(market: str, pct: float, limit: int | None = None) -> dict:
    frames = _load_frames(market)
    if market == "us":
        from params_us import GATES as G
    else:
        from params import GATES as G

    tickers = sorted(frames)
    if limit:
        tickers = tickers[:limit]
    print(f"  {len(tickers):,} tickers, squeeze = bottom {pct:g}% of own band width")

    sq = {"m": [], "e": [], "v": [], "p": []}
    nm = {"m": [], "e": [], "v": [], "p": []}
    rng = np.random.default_rng(0)

    for n, t in enumerate(tickers, 1):
        try:
            df = ind.build(frames[t])
        except Exception:
            continue
        if len(df) < 200 + HORIZON:
            continue

        ok = ((df["turnover_avg"] >= G.min_avg_turnover_sgd)
              & (df["Close"] >= G.min_price_sgd)).to_numpy()
        sqp = df["bb_squeeze_pct"].to_numpy(float)
        openp = df["Open"].to_numpy(float)
        closep = df["Close"].to_numpy(float)
        high = df["High"].to_numpy(float)
        low = df["Low"].to_numpy(float)
        atrp = df["atr_pct"].to_numpy(float)
        ret1 = pd.Series(closep).pct_change()

        last = -10 ** 9
        for i in range(150, len(df) - HORIZON - 2):
            if not ok[i] or not np.isfinite(sqp[i]) or not np.isfinite(atrp[i]):
                continue
            is_sq = sqp[i] <= pct
            # sample non-squeeze bars sparsely to keep the comparison balanced
            if not is_sq and rng.random() > 0.02:
                continue
            if is_sq and i - last < COOLDOWN:
                continue

            entry = openp[i + 1]
            if not np.isfinite(entry) or entry <= 0:
                continue
            j = i + 1 + HORIZON
            if j >= len(closep):
                continue

            move = (closep[j] / entry - 1) * 100
            win_hi = np.nanmax(high[i + 1:j + 1])
            win_lo = np.nanmin(low[i + 1:j + 1])
            exc = max(abs(win_hi / entry - 1), abs(win_lo / entry - 1)) * 100

            v_before = ret1.iloc[max(0, i - 19):i + 1].std() * 100
            v_after = ret1.iloc[i + 1:j + 1].std() * 100
            volr = v_after / v_before if v_before and np.isfinite(v_before) else np.nan
            if not np.isfinite(volr):
                continue

            sigma = np.clip(atrp[i] / 100 * np.sqrt(252) * 1.15, 0.10, 3.0)
            prem = 0.4 * sigma * np.sqrt(30 / 365) * 100   # ATM call, % of spot

            tgt = sq if is_sq else nm
            tgt["m"].append(move); tgt["e"].append(exc)
            tgt["v"].append(volr); tgt["p"].append(prem)
            if is_sq:
                last = i

        if n % 500 == 0:
            print(f"  {n:,}/{len(tickers):,}  (squeeze {len(sq['m']):,})")

    out = {"market": market, "pct": pct, "groups": []}
    for lab, g in (("SQUEEZE", sq), ("NORMAL BARS", nm)):
        if g["m"]:
            out["groups"].append(_stats(lab, g["m"], g["e"], g["v"], g["p"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    ap.add_argument("--pct", type=float, default=10.0,
                    help="squeeze = band width in bottom N%% of its 6-month range")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    print(f"Bollinger squeeze -> breakout test [{a.market.upper()}], {HORIZON}-day horizon\n")
    res = run(a.market, a.pct, a.limit)
    if not res["groups"]:
        raise SystemExit("No data.")

    print(f"\n  {'':<14}{'n':>9}{'|move|':>9}{'excursion':>11}{'vol after/before':>18}")
    print("  " + "-" * 62)
    for g in res["groups"]:
        print(f"  {g['label']:<14}{g['n']:>9,}{g['median_abs_move']:>8.2f}%"
              f"{g['median_excursion']:>10.2f}%{g['median_vol_ratio']:>17.2f}x")

    print(f"\n  Straddle economics (call + put, breakeven = 2x premium)\n")
    print(f"  {'':<14}{'breakeven':>11}{'hit (hold)':>12}{'P&L (hold)':>12}"
          f"{'hit (best)':>12}{'P&L (best)':>12}")
    print("  " + "-" * 74)
    for g in res["groups"]:
        print(f"  {g['label']:<14}{g['median_breakeven']:>10.2f}%"
              f"{g['straddle_hold_hit']:>11.1f}%{g['straddle_hold_mean']:>11.1f}%"
              f"{g['straddle_best_hit']:>11.1f}%{g['straddle_best_mean']:>11.1f}%")

    if len(res["groups"]) == 2:
        s, nrm = res["groups"]
        print(f"\n  Vol expansion: squeeze {s['median_vol_ratio']:.2f}x vs "
              f"normal {nrm['median_vol_ratio']:.2f}x")
        if s["median_vol_ratio"] > nrm["median_vol_ratio"] * 1.05:
            print("  -> squeezes DO precede more expansion than normal bars")
        else:
            print("  -> squeezes do NOT precede more expansion than normal bars")

    print("\n  'hold' = to expiry. 'best' = perfect exit at the largest excursion,")
    print("  which is unachievable but bounds the upside.")
    print("  Premiums are modelled from ATR, so squeeze options look artificially")
    print("  cheap here. Real implied vol prices the squeeze in. UPPER BOUND ONLY.")

    (HERE / f"squeeze_{a.market}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
