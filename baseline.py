"""
baseline.py — the comparison that decides whether a signal is worth anything.

A backtest in isolation cannot tell you if a signal works. "51.8% up over 20
days" sounds like an edge until you ask the only question that matters:

    What would random entries into the same stocks over the same period have
    returned?

If a coin flip on the same universe gives you 53% and +0.9%, then a signal
giving 51.8% and +0.48% is not a weak edge. It is worse than nothing — you did
work, took risk, paid spreads, and underperformed picking at random.

This script measures that benchmark from the SAME cached price data the
backtest used, so it is a true apples-to-apples comparison rather than an
index return pulled from elsewhere.

    python baseline.py --market us
    python baseline.py --market us --n 40000

It applies the identical eligibility filters as the backtest (enough history,
liquidity floor, price floor) so the only difference between the two samples is
whether the eight gates fired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import indicators as ind

HERE = Path(__file__).parent


def _load_frames(market: str) -> dict[str, pd.DataFrame]:
    if market == "us":
        import data_us
        # Only plain us_YYYY-MM-DD.parquet. A tagged cache like
        # us_holdout_2026-08-27.parquet also matches "us_*" and would silently
        # shadow the main cache — which once made a two-period test read the
        # same decade twice and report identical p-values.
        import re as _re
        files = sorted(f for f in data_us.CACHE_DIR.glob("us_*.parquet")
                       if _re.fullmatch(r"us_\d{4}-\d{2}-\d{2}\.parquet", f.name))
        if not files:
            raise SystemExit("No US cache found. Run: python scan.py --market us --backtest")
        print(f"  reading {files[-1].name}")
        return data_us._unflatten(pd.read_parquet(files[-1]))
    import data
    frames = {}
    for f in data.CACHE_DIR.glob("*.parquet"):
        t = f.stem.rsplit("_", 3)[0].replace("_", ".")
        try:
            frames[t] = pd.read_parquet(f)
        except Exception:
            pass
    if not frames:
        raise SystemExit("No SGX cache found. Run: python scan.py --backtest")
    return frames


def run(market: str, n_samples: int, horizons=(5, 10, 20), seed: int = 0,
        restrict_gates: list[str] | None = None) -> dict:
    if market == "us":
        from params_us import GATES as G, INDICATORS as I
    else:
        from params import GATES as G, INDICATORS as I

    frames = _load_frames(market)
    print(f"  {len(frames):,} tickers in cache")
    if restrict_gates:
        print(f"  restricting random draws to bars passing: {', '.join(restrict_gates)}")

    rng = np.random.default_rng(seed)
    tickers = [t for t, d in frames.items() if len(d) >= G.min_bars_required + max(horizons) + 2]
    print(f"  {len(tickers):,} eligible")

    # ------------------------------------------------------------------
    # Pass 1 — count eligible bars per ticker.
    #
    # This matters more than it looks. Allocating the same number of random
    # draws to every ticker gives a stock that was briefly liquid the same
    # weight as one liquid for six straight years. Rule-based signals are not
    # distributed that way — they arise in proportion to eligible bars, so
    # long-surviving healthy companies dominate. Sampling flat while comparing
    # against proportional signals builds a survivorship gap into the
    # benchmark, and it shows up as a uniform edge on every rule tested,
    # including rules that contradict each other.
    #
    # Only turnover and close are needed here, so this pass is cheap.
    # ------------------------------------------------------------------
    counts: dict[str, int] = {}
    vp = I.vol_avg_period
    for t in tickers:
        d = frames[t]
        turn = (d["Close"] * d["Volume"]).rolling(vp, min_periods=vp).mean()
        ok = ((turn >= G.min_avg_turnover_sgd) & (d["Close"] >= G.min_price_sgd))
        lo, hi = 150, len(d) - max(horizons) - 1
        counts[t] = int(ok.to_numpy()[lo:hi].sum()) if hi > lo else 0

    total_eligible = sum(counts.values())
    if total_eligible == 0:
        raise SystemExit("No eligible bars.")
    rate = min(1.0, n_samples / total_eligible)
    print(f"  {total_eligible:,} eligible bars -> sampling {rate * 100:.2f}% of them")

    rows = []

    for n, t in enumerate(tickers, 1):
        raw = frames[t]
        try:
            df = ind.build(raw)
        except Exception:
            continue
        lo, hi = 150, len(df) - max(horizons) - 1
        if hi <= lo:
            continue

        # Same eligibility as the LIQUIDITY gate — otherwise the benchmark
        # would include microcaps the signal was never allowed to pick.
        ok = ((df["turnover_avg"] >= G.min_avg_turnover_sgd)
              & (df["Close"] >= G.min_price_sgd)).to_numpy()

        # Optionally require some of the signal's own gates to pass, so the
        # comparison isolates what the REMAINING gates contribute.
        if restrict_gates:
            try:
                import fast_rules
                gm = fast_rules.evaluate_all(df, I, G)
                for g in restrict_gates:
                    ok = ok & gm[g].to_numpy()
            except Exception:
                continue

        cand = np.flatnonzero(ok[lo:hi]) + lo
        if len(cand) == 0:
            continue

        # Draw in proportion to this ticker's share of eligible bars, so the
        # random sample has the same ticker composition as the rule signals.
        n_draw = rng.binomial(len(cand), rate)
        if n_draw == 0:
            continue
        pick = rng.choice(cand, size=min(n_draw, len(cand)), replace=False)
        openp = df["Open"].to_numpy(float)
        closep = df["Close"].to_numpy(float)
        for i in pick:
            # Entry at next open, exactly as the backtest models it.
            entry = openp[i + 1]
            if not np.isfinite(entry) or entry <= 0:
                continue
            r = {"ticker": t}
            for h in horizons:
                j = i + h
                r[f"r{h}"] = (closep[j] / entry - 1) * 100 if j < len(closep) else np.nan
            rows.append(r)

        if n % 500 == 0:
            print(f"  sampled {n:,}/{len(tickers):,} ({len(rows):,} draws)")

    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("No samples drawn.")

    out = {"market": market, "n": int(len(d)), "horizons": {}}
    for h in horizons:
        col = d[f"r{h}"].dropna()
        out["horizons"][str(h)] = {
            "win_rate": float((col > 0).mean() * 100),
            "median": float(col.median()),
            "mean": float(col.mean()),
            "p25": float(col.quantile(0.25)),
            "p75": float(col.quantile(0.75)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    ap.add_argument("--n", type=int, default=30000, help="random entries to draw")
    ap.add_argument("--gate", action="append", default=None,
                    help="restrict random draws to bars passing this gate "
                         "(repeatable, e.g. --gate REGIME)")
    a = ap.parse_args()

    print(f"Random-entry benchmark [{a.market.upper()}]")
    base = run(a.market, a.n, restrict_gates=a.gate)

    bt_file = HERE / ("baserates_us.json" if a.market == "us" else "baserates.json")
    sig = json.loads(bt_file.read_text()) if bt_file.exists() else {}

    print(f"\n{'':>5}  {'RANDOM ENTRY':>28}   {'YOUR SIGNAL':>28}")
    print(f"{'':>5}  {'up%':>7}{'median':>10}{'p25':>11}   {'up%':>7}{'median':>10}{'p25':>11}   verdict")
    print("  " + "-" * 78)

    for h in ("5", "10", "20"):
        b = base["horizons"][h]
        s = (sig.get("horizons") or {}).get(h)
        left = f"{b['win_rate']:6.1f}%{b['median']:+9.2f}%{b['p25']:+10.2f}%"
        if s:
            right = f"{s['win_rate']:6.1f}%{s['median']:+9.2f}%{s['p25']:+10.2f}%"
            edge = s["median"] - b["median"]
            verdict = f"{edge:+.2f}pp {'BETTER' if edge > 0 else 'WORSE'} than random"
        else:
            right, verdict = f"{'—':>27}", "no backtest found"
        print(f"  {h:>3}d  {left}   {right}   {verdict}")

    print(f"\n  random draws: {base['n']:,}   signals: {sig.get('n', 0):,}")
    print("\n  A signal only has value if it beats the random column.")
    print("  Neither column includes commission, spread or slippage.")

    (HERE / f"baseline_{a.market}.json").write_text(json.dumps(base, indent=2))


if __name__ == "__main__":
    main()
