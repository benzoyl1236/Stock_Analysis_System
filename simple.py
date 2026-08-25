"""
simple.py — your indicators, on their own, the plain way.

WHY THIS EXISTS

The eight-gate system was my construction, not yours. Your screenshots contained
indicator settings; the regime stacking, pullback requirement, trigger ceiling
of 60, headroom rule and long-only bias were all invented by me. When that
system showed only a small edge, there was no way to tell whether your
indicators were weak or my scaffolding was.

This strips the scaffolding out. Each rule below uses ONE of your indicators,
read the way the indicator is designed to be read, with no stacking.

EVERY NUMBER COMES FROM YOUR OWN POEMS DIALOGS
    EMA periods 8 / 21 / 100          from the EMA dialog
    Bollinger 20, 2 SD                from the Bollinger dialog
    Stochastic 9 / 3 / 3              from the Stochastic dialog
    buy zone 30, sell zone 70         from the Stochastic dialog
    MACD 9 / 18 / 9                   from the MACD dialog
    Momentum 28                       from the Momentum dialog

There are no thresholds of mine anywhere in this file. Note in particular that
the sell zone of 70 finally gets used — the eight-gate system never read it,
which is why it had entries but no exits.

TWO CHOICES I HAD TO MAKE, both disclosed:
  * Entry at the next open after the signal bar closes. Entering at the signal
    close would be lookahead and would flatter every number here.
  * A 20-bar cooldown per rule per ticker, so one long trend is not counted as
    twenty separate signals. Without it, trending names dominate the sample.

    python simple.py --market us
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import indicators as ind
from baseline import _load_frames
from params import INDICATORS as I

HERE = Path(__file__).parent
HORIZONS = (5, 10, 20)
COOLDOWN = 20


# --------------------------------------------------------------------------
# The rules. Each returns a boolean array: True where the rule says "buy".
# --------------------------------------------------------------------------

def _cross_up(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a crosses above b on this bar."""
    prev_a, prev_b = np.roll(a, 1), np.roll(b, 1)
    prev_a[0] = prev_b[0] = np.nan
    return (prev_a <= prev_b) & (a > b)


def rules_for(df: pd.DataFrame) -> dict[str, np.ndarray]:
    c = df["Close"].to_numpy(float)
    e8 = df["ema_fast"].to_numpy(float)
    e21 = df["ema_mid"].to_numpy(float)
    e100 = df["ema_slow"].to_numpy(float)
    k = df["stoch_k"].to_numpy(float)
    d = df["stoch_d"].to_numpy(float)
    macd = df["macd"].to_numpy(float)
    sig = df["macd_signal"].to_numpy(float)
    mtm = df["mtm"].to_numpy(float)
    bb_lo = df["bb_lower"].to_numpy(float)
    bb_up = df["bb_upper"].to_numpy(float)
    bb_mid = df["bb_mid"].to_numpy(float)
    zeros = np.zeros_like(macd)

    return {
        # --- EMA (8 / 21 / 100) ---
        "EMA 8 crosses above 21": _cross_up(e8, e21),
        "Close crosses above EMA100": _cross_up(c, e100),

        # --- MACD (9 / 18 / 9) ---
        "MACD crosses above signal": _cross_up(macd, sig),
        "MACD crosses above zero": _cross_up(macd, zeros),

        # --- Stochastic (9 / 3 / 3), zones from your dialog ---
        "Stoch %K crosses %D under 30": _cross_up(k, d) & (k < I.stoch_buy_zone),
        "Stoch %K rises above 30": _cross_up(k, np.full_like(k, I.stoch_buy_zone)),

        # --- Bollinger (20, 2 SD) ---
        "Close below lower band": c < bb_lo,
        "Close crosses above upper band": _cross_up(c, bb_up),
        "Close crosses above BB mid": _cross_up(c, bb_mid),

        # --- Momentum (28) ---
        "Momentum 28 crosses above 0": _cross_up(mtm, zeros),
    }


# --------------------------------------------------------------------------

def _forward(openp, closep, i, horizons):
    entry = openp[i + 1]
    if not np.isfinite(entry) or entry <= 0:
        return None
    out = {}
    for h in horizons:
        j = i + 1 + h
        out[h] = (closep[j] / entry - 1) * 100 if j < len(closep) else np.nan
    return out


def _stoch_round_trip(df, openp, closep, k):
    """Your stochastic rule with YOUR exit: buy when %K crosses %D below 30,
    sell when %K reaches 70. This is the only rule here with a real exit."""
    d = df["stoch_d"].to_numpy(float)
    entries = _cross_up(k, d) & (k < I.stoch_buy_zone)
    trades = []
    i, n = 150, len(df) - 2
    while i < n:
        if not entries[i]:
            i += 1
            continue
        entry = openp[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue
        exit_i = None
        for j in range(i + 1, min(i + 250, len(df))):
            if np.isfinite(k[j]) and k[j] >= I.stoch_sell_zone:
                exit_i = j
                break
        if exit_i is None:
            i += 1
            continue
        trades.append({"ret": (closep[exit_i] / entry - 1) * 100,
                       "bars": exit_i - i})
        i = exit_i + 1
    return trades


def run(market: str, limit: int | None = None) -> dict:
    frames = _load_frames(market)
    if market == "us":
        from params_us import GATES as G
    else:
        from params import GATES as G

    tickers = sorted(frames)
    if limit:
        tickers = tickers[:limit]
    print(f"  {len(tickers):,} tickers")

    acc: dict[str, list] = {}
    rt_trades: list[dict] = []
    scanned = 0

    for n, t in enumerate(tickers, 1):
        try:
            df = ind.build(frames[t])
        except Exception:
            continue
        if len(df) < G.min_bars_required + max(HORIZONS) + 2:
            continue

        # Same tradability floor as the signal used, so the comparison is fair.
        ok = ((df["turnover_avg"] >= G.min_avg_turnover_sgd)
              & (df["Close"] >= G.min_price_sgd)).to_numpy()
        openp = df["Open"].to_numpy(float)
        closep = df["Close"].to_numpy(float)
        scanned += 1

        for name, fires in rules_for(df).items():
            bucket = acc.setdefault(name, [])
            last = -10 ** 9
            hits = np.flatnonzero(fires & ok)
            for i in hits:
                if i < 150 or i >= len(df) - max(HORIZONS) - 2:
                    continue
                if i - last < COOLDOWN:
                    continue
                fwd = _forward(openp, closep, i, HORIZONS)
                if fwd:
                    bucket.append(fwd)
                    last = i

        k = df["stoch_k"].to_numpy(float)
        rt_trades.extend(_stoch_round_trip(df, openp, closep, k))

        if n % 500 == 0:
            print(f"  {n:,}/{len(tickers):,}")

    out = {"market": market, "scanned": scanned, "rules": {}}
    for name, rows in acc.items():
        if not rows:
            continue
        stat = {"n": len(rows)}
        for h in HORIZONS:
            col = pd.Series([r[h] for r in rows]).dropna()
            if len(col) == 0:
                continue
            stat[str(h)] = {"win": float((col > 0).mean() * 100),
                            "median": float(col.median()),
                            "p25": float(col.quantile(0.25))}
        out["rules"][name] = stat

    if rt_trades:
        r = pd.Series([t["ret"] for t in rt_trades])
        b = pd.Series([t["bars"] for t in rt_trades])
        out["stoch_round_trip"] = {
            "n": len(r), "win": float((r > 0).mean() * 100),
            "median": float(r.median()), "mean": float(r.mean()),
            "p25": float(r.quantile(0.25)),
            "median_bars_held": float(b.median()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    ap.add_argument("--limit", type=int, default=None, help="first N tickers only")
    a = ap.parse_args()

    print(f"Your indicators, one at a time [{a.market.upper()}]")
    res = run(a.market, a.limit)

    bl = HERE / f"baseline_{a.market}.json"
    base = json.loads(bl.read_text()) if bl.exists() else {}
    b20 = (base.get("horizons") or {}).get("20", {})

    print(f"\n  {'rule':<32}{'n':>8}{'20d up%':>10}{'20d med':>10}{'20d p25':>10}{'vs random':>12}")
    print("  " + "-" * 82)
    ranked = sorted(res["rules"].items(),
                    key=lambda kv: -(kv[1].get("20", {}).get("median", -99)))
    for name, s in ranked:
        d = s.get("20", {})
        if not d:
            continue
        edge = (d["median"] - b20["median"]) if b20 else None
        etxt = f"{edge:+.2f}pp" if edge is not None else "—"
        print(f"  {name:<32}{s['n']:>8,}{d['win']:>9.1f}%{d['median']:>9.2f}%"
              f"{d['p25']:>9.2f}%{etxt:>12}")

    if b20:
        print(f"  {'RANDOM ENTRY (benchmark)':<32}{base.get('n',0):>8,}"
              f"{b20.get('win_rate',0):>9.1f}%{b20['median']:>9.2f}%"
              f"{b20['p25']:>9.2f}%{'—':>12}")

    rt = res.get("stoch_round_trip")
    if rt:
        print(f"\n  Stochastic round trip — buy %K crosses %D under 30, "
              f"sell when %K hits 70")
        print(f"    trades {rt['n']:,}   win {rt['win']:.1f}%   "
              f"median {rt['median']:+.2f}%   mean {rt['mean']:+.2f}%   "
              f"held {rt['median_bars_held']:.0f} bars")
        print("    (the only rule here with a real exit — it uses your sell zone)")

    print("\n  Costs are not included. Subtract roughly 0.1-0.3% per round trip.")
    (HERE / f"simple_{a.market}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
