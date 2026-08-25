"""
spreads.py — vertical spreads priced against YOUR signals.

A bull call spread (buy ATM call, sell an OTM call) is a genuinely better fit
for a modest directional view than a naked call:

  * cheaper, so the breakeven move is smaller
  * less exposed to time decay
  * the short leg recovers some of the premium you would otherwise lose

The catch is that it caps the upside. You give up the tail in exchange for a
lower breakeven, and that trade is priced fairly by the market. A spread
changes the SHAPE of the payoff, not its expected value.

This reads output/backtest_trades.csv — your real signals — and prices both
legs with Black-Scholes using each signal's own measured volatility, then
applies your actual forward return to see what the position would have made.

    python spreads.py                       widths 3/5/10% at 30 DTE
    python spreads.py --dte 45 --cost 0.03

BEAR PUT SPREAD ("the reversal") is computed too: buy ATM put, sell an OTM put.
It profits when price falls. Note that your signal is long-only and was never
tested for downside, so treat that column as what happens if you bet against
a bullish signal.

WHAT IS AND IS NOT INCLUDED
  Included: Black-Scholes pricing of both legs, and an optional round-trip
  cost as a fraction of net debit (--cost, default 0.05 = 5%). A vertical has
  TWO legs, so you cross two bid-ask spreads getting in and two getting out.
  On liquid US names 5% of debit is reasonable; on wide markets it is worse.

  Not included: early assignment, dividends, IV skew (real OTM calls trade at
  different implied vols than ATM, usually cheaper, which slightly helps the
  spread seller and therefore slightly helps you as the buyer of the spread).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def _ncdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def bs_call(S, K, T, sigma, r=0.04):
    S, K = np.asarray(S, float), np.asarray(K, float)
    sig_t = sigma * np.sqrt(T)
    sig_t = np.where(sig_t <= 1e-9, 1e-9, sig_t)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sig_t
    d2 = d1 - sig_t
    return S * _ncdf(d1) - K * np.exp(-r * T) * _ncdf(d2)


def bs_put(S, K, T, sigma, r=0.04):
    S, K = np.asarray(S, float), np.asarray(K, float)
    return bs_call(S, K, T, sigma, r) - S + K * np.exp(-r * T)


def analyse(tr: pd.DataFrame, width_pct: float, dte: int, hold: int,
            cost_frac: float) -> dict:
    T = dte / 365.0
    S = 100.0                                    # normalise; strikes scale with it
    ret = tr[f"ret_{hold}"].to_numpy(float) / 100.0
    atrp = tr["atr_pct"].to_numpy(float) / 100.0
    sigma = np.clip(atrp * np.sqrt(252) * 1.15, 0.10, 3.0)

    K1 = S                                        # long leg, at the money
    K2 = S * (1 + width_pct / 100.0)              # short leg, out of the money

    # --- Bull call spread ---
    debit = bs_call(S, K1, T, sigma) - bs_call(S, K2, T, sigma)
    debit = np.maximum(debit, 0.01)
    ST = S * (1 + ret)
    payoff = np.clip(ST - K1, 0, K2 - K1)
    net = payoff - debit
    total_cost = debit * (1 + cost_frac)
    pnl = (payoff - total_cost) / total_cost * 100.0
    be_move = (K1 + debit) / S - 1

    # --- Bear put spread (the reversal) ---
    K2p = S * (1 - width_pct / 100.0)
    debit_p = np.maximum(bs_put(S, K1, T, sigma) - bs_put(S, K2p, T, sigma), 0.01)
    payoff_p = np.clip(K1 - ST, 0, K1 - K2p)
    total_cost_p = debit_p * (1 + cost_frac)
    pnl_p = (payoff_p - total_cost_p) / total_cost_p * 100.0

    return {
        "width": width_pct,
        "median_debit_pct_of_spot": float(np.median(debit)),
        "median_breakeven_move": float(np.median(be_move) * 100),
        "max_gain_on_debit": float(np.median((K2 - K1 - debit) / debit) * 100),
        "bull_win": float((payoff > total_cost).mean() * 100),
        "bull_mean": float(np.mean(pnl)),
        "bull_median": float(np.median(pnl)),
        "bull_maxloss_rate": float((payoff <= 0).mean() * 100),
        "bear_win": float((payoff_p > total_cost_p).mean() * 100),
        "bear_mean": float(np.mean(pnl_p)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dte", type=int, default=30)
    ap.add_argument("--hold", type=int, default=20, choices=[5, 10, 20])
    ap.add_argument("--cost", type=float, default=0.05,
                    help="round-trip cost as fraction of net debit (default 0.05)")
    ap.add_argument("--file", default=None)
    a = ap.parse_args()

    path = Path(a.file) if a.file else HERE / "output" / "backtest_trades.csv"
    if not path.exists():
        raise SystemExit(f"No trades file at {path}\nRun: python scan.py --market us --backtest")

    tr = pd.read_csv(path)
    col = f"ret_{a.hold}"
    if col not in tr.columns or "atr_pct" not in tr.columns:
        raise SystemExit(f"{path} needs {col} and atr_pct")
    tr = tr.dropna(subset=[col, "atr_pct"])
    tr = tr[tr["atr_pct"] > 0]

    print(f"Vertical spreads on your signals — {len(tr):,} trades, "
          f"{a.dte} DTE, held {a.hold}d, costs {a.cost*100:.0f}% of debit\n")
    print(f"  {'width':>7}{'debit':>8}{'breakeven':>11}{'max gain':>10}"
          f"{'BULL win':>10}{'BULL P&L':>10}{'total loss':>12}{'BEAR P&L':>10}")
    print("  " + "-" * 78)

    for w in (2.5, 3.0, 5.0, 7.5, 10.0):
        r = analyse(tr, w, a.dte, a.hold, a.cost)
        print(f"  {w:>6.1f}%{r['median_debit_pct_of_spot']:>7.2f}%"
              f"{r['median_breakeven_move']:>10.2f}%{r['max_gain_on_debit']:>9.0f}%"
              f"{r['bull_win']:>9.1f}%{r['bull_mean']:>9.1f}%"
              f"{r['bull_maxloss_rate']:>11.1f}%{r['bear_mean']:>9.1f}%")

    print("\n  debit      = net cost as % of stock price")
    print("  breakeven  = how far the stock must rise to break even")
    print("  max gain   = best case return on the debit (stock at or above short strike)")
    print("  total loss = share of signals where the spread expired worthless")
    print("\n  A spread lowers the breakeven and caps the gain. Both legs are")
    print("  fairly priced, so with no directional edge the expected value is")
    print("  roughly zero minus costs, regardless of the width you choose.")


if __name__ == "__main__":
    main()
