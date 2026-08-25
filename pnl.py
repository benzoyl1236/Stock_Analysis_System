"""
pnl.py — what would you actually have made?

Median return per signal is not profitability. Profitability depends on how
many trades you can hold at once, how much capital each consumes, what costs
you pay, and what you would have earned doing nothing instead.

This simulates trading your real signals in date order:
  * capital split across a maximum number of concurrent positions
  * each position held for the fixed horizon, then closed
  * signals ignored when no slot is free (this is what really happens)
  * round-trip cost applied to every trade

    python pnl.py
    python pnl.py --capital 50000 --slots 5 --cost 0.20

COMPARISON
  The number that matters is not whether the curve goes up. Over 2020-2026
  most things went up. It is whether it beat holding an index fund, which
  required no research, no screening and no evenings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def simulate(tr: pd.DataFrame, capital: float, slots: int, hold: int,
             cost_pct: float, jitter_seed: int | None = None,
             rank_by: str | None = None, rank_asc: bool = True,
             random_rank_seed: int | None = None) -> dict:
    """If rank_by is set, each day's competing signals are sorted by that
    feature and the best fill the free slots, instead of arrival order."""
    col = f"ret_{hold}"
    tr = tr.dropna(subset=[col, "entry_date"]).copy()
    tr["entry_date"] = pd.to_datetime(tr["entry_date"], errors="coerce")
    tr = tr.dropna(subset=["entry_date"])

    # Which signals you actually take is decided by which happened to arrive
    # when a slot was free. That ordering is arbitrary. Shuffling ties lets us
    # ask how much of the result is the strategy and how much is luck of the
    # draw over which 6% of signals got taken.
    if jitter_seed is not None:
        rng = np.random.default_rng(jitter_seed)
        tr = tr.iloc[rng.permutation(len(tr))]
    if random_rank_seed is not None:
        rng = np.random.default_rng(random_rank_seed)
        tr = tr.copy()
        tr["_rand"] = rng.normal(size=len(tr))
        rank_by, rank_asc = "_rand", True
    tr = tr.sort_values("entry_date", kind="stable")

    equity = capital
    free_at: list[pd.Timestamp] = []      # when each occupied slot frees up
    taken, skipped = [], 0
    curve = []

    # Group by day so same-day signals compete on merit rather than file order
    for d, day_rows in tr.groupby("entry_date", sort=True):
        free_at = [t for t in free_at if t > d]
        if rank_by and rank_by in day_rows.columns:
            day_rows = day_rows.sort_values(rank_by, ascending=rank_asc)
        for _, row in day_rows.iterrows():
            free_at = [t for t in free_at if t > d]
            if len(free_at) >= slots:
                skipped += 1
                continue

            stake = equity / slots
            gross = row[col] / 100.0
            net = gross - cost_pct / 100.0        # round-trip cost
            pnl = stake * net
            equity += pnl
            taken.append({"date": d, "ret": net * 100, "pnl": pnl,
                          "equity": equity})
            free_at.append(d + pd.Timedelta(days=hold * 1.45))
            curve.append((d, equity))

    if not taken:
        return {}

    t = pd.DataFrame(taken)
    c = pd.DataFrame(curve, columns=["date", "equity"]).set_index("date")
    years = (t["date"].max() - t["date"].min()).days / 365.25
    cagr = ((equity / capital) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    peak = c["equity"].cummax()
    dd = ((c["equity"] - peak) / peak * 100).min()
    # per-trade Sharpe-like ratio, annualised by trades per year
    tpy = len(t) / years if years > 0 else 0
    sharpe = (t["ret"].mean() / t["ret"].std() * np.sqrt(tpy)) if t["ret"].std() else 0.0

    return {
        "trades_taken": len(t), "signals_skipped": skipped,
        "years": years, "final": equity, "cagr": cagr,
        "total_return": (equity / capital - 1) * 100,
        "max_dd": dd, "win_rate": float((t["ret"] > 0).mean() * 100),
        "avg_trade": float(t["ret"].mean()), "sharpe": sharpe,
        "trades_per_year": tpy,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=50000)
    ap.add_argument("--slots", type=int, default=5, help="max concurrent positions")
    ap.add_argument("--hold", type=int, default=20, choices=[5, 10, 20])
    ap.add_argument("--cost", type=float, default=0.20,
                    help="round-trip cost per trade in %% (default 0.20)")
    ap.add_argument("--compare-rank", default=None, metavar="FEATURE",
                    help="test whether ranking by FEATURE beats ranking at random")
    ap.add_argument("--rank-by", default=None,
                    help="feature to rank same-day signals by, e.g. mtm")
    ap.add_argument("--rank-desc", action="store_true",
                    help="prefer HIGH values (default prefers LOW)")
    ap.add_argument("--trials", type=int, default=200,
                    help="robustness runs reshuffling signal order (0 to skip)")
    ap.add_argument("--file", default=None)
    a = ap.parse_args()

    path = Path(a.file) if a.file else HERE / "output" / "backtest_trades.csv"
    if not path.exists():
        raise SystemExit(f"No trades file at {path}")

    tr = pd.read_csv(path)
    if "entry_date" not in tr.columns:
        raise SystemExit("trades file has no entry_date column")

    if a.compare_rank:
        f = a.compare_rank
        if f not in tr.columns:
            raise SystemExit(f"'{f}' is not a column in the trades file.")
        print(f"Does ranking by '{f}' beat ranking at random?")
        print(f"  ${a.capital:,.0f}, {a.slots} slots, {a.hold}d hold, "
              f"{a.cost}% cost, {a.trials} random-rank trials\n")
        null = []
        for sd in range(max(a.trials, 50)):
            rr = simulate(tr, a.capital, a.slots, a.hold, a.cost,
                          random_rank_seed=sd)
            if rr:
                null.append(rr["cagr"])
        null = np.array(null)
        for direction, asc in (("LOW first", True), ("HIGH first", False)):
            rr = simulate(tr, a.capital, a.slots, a.hold, a.cost,
                          rank_by=f, rank_asc=asc)
            if not rr:
                continue
            pct = float((null < rr["cagr"]).mean() * 100)
            verdict = ("beats random" if pct >= 95 else
                       "worse than random" if pct <= 5 else
                       "inside the random range - adds nothing")
            print(f"  rank by {f}, {direction:<11} CAGR {rr['cagr']:+6.1f}%"
                  f"   percentile {pct:5.1f}   {verdict}")
        print(f"\n  random ranking: median {np.median(null):+.1f}%   "
              f"5th-95th {np.percentile(null,5):+.1f}% to {np.percentile(null,95):+.1f}%")
        print(f"\n  A feature only helps if it lands above the 95th percentile.")
        print(f"  Landing inside the range means you gained nothing over")
        print(f"  picking at random, however good the raw CAGR looks.")
        print(f"\n  WARNING: if this trades file is the data the feature was")
        print(f"  discovered in, a good result here proves nothing.")
        return

    if a.rank_by:
        print(f"  NOTE: ranked selection is deterministic, so --trials cannot")
        print(f"  vary the outcome. Use --compare-rank {a.rank_by} instead.\n")

    print(f"Trading your signals: ${a.capital:,.0f}, {a.slots} slots, "
          f"{a.hold}-day hold, {a.cost}% round-trip cost\n")

    r = simulate(tr, a.capital, a.slots, a.hold, a.cost,
                 rank_by=a.rank_by, rank_asc=not a.rank_desc)
    if not r:
        raise SystemExit("No trades simulated.")

    print(f"  period                {r['years']:.1f} years")
    print(f"  signals available     {len(tr):,}")
    print(f"  trades taken          {r['trades_taken']:,}  "
          f"({r['trades_per_year']:.0f}/yr, {r['signals_skipped']:,} skipped, no free slot)")
    print(f"  win rate              {r['win_rate']:.1f}%")
    print(f"  average trade         {r['avg_trade']:+.2f}%  (after costs)")
    print()
    print(f"  final equity          ${r['final']:,.0f}")
    print(f"  total return          {r['total_return']:+.1f}%")
    print(f"  CAGR                  {r['cagr']:+.1f}%")
    print(f"  max drawdown          {r['max_dd']:.1f}%")
    print(f"  Sharpe (approx)       {r['sharpe']:.2f}")

    print("\n  Cost sensitivity (CAGR):")
    for c in (0.0, 0.10, 0.20, 0.40):
        rr = simulate(tr, a.capital, a.slots, a.hold, c,
                      rank_by=a.rank_by, rank_asc=not a.rank_desc)
        if rr:
            print(f"    {c:.2f}% per round trip -> {rr['cagr']:+6.1f}% CAGR")

    # ---- robustness: how much of the result is which signals you happened to take?
    if a.trials > 0:
        print(f"\n  Robustness — {a.trials} runs, reshuffling which signals get taken")
        print(f"  (same strategy, same costs, only the arrival order of ties changes)")
        cagrs, dds, wins = [], [], []
        for s in range(a.trials):
            rr = simulate(tr, a.capital, a.slots, a.hold, a.cost,
                          jitter_seed=s, rank_by=a.rank_by,
                          rank_asc=not a.rank_desc)
            if rr:
                cagrs.append(rr["cagr"]); dds.append(rr["max_dd"]); wins.append(rr["win_rate"])
        if cagrs:
            c = np.array(cagrs)
            print(f"    CAGR       median {np.median(c):+6.1f}%   "
                  f"range {c.min():+.1f}% to {c.max():+.1f}%")
            print(f"    worse than zero in {(c < 0).mean() * 100:.0f}% of runs")
            print(f"    max drawdown   median {np.median(dds):.1f}%")
            print(f"    win rate       median {np.median(wins):.1f}%")
            if c.min() < 0 < c.max():
                print("    -> the sign of the result depends on which signals you take.")
                print("       That is luck, not edge.")

    print("\n  Benchmark: a US index fund over roughly this period returned")
    print("  a solidly positive CAGR for zero effort and near-zero cost.")
    print("  Compare against THAT, not against zero.")


if __name__ == "__main__":
    main()
