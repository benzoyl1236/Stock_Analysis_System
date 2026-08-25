"""
options.py — what your signals would have done as options rather than shares.

Buying a call is not a leveraged version of buying the stock. It is a different
bet: the stock must move ENOUGH, and it must move IN TIME. A move that makes
money on shares can lose everything on a call.

This reads output/backtest_trades.csv — your real signals, not a simulation —
and prices an at-the-money call against each one using that signal's own
measured volatility (atr_pct), so the premium reflects the actual stock rather
than a generic assumption.

    python options.py
    python options.py --dte 45 --market us

MODEL AND ITS LIMITS
  ATM premium is approximated as 0.4 * sigma * sqrt(T) * S, the standard
  Brenner-Subrahmanyam approximation. Sigma is annualised from atr_pct
  (daily ATR% * sqrt(252) * 1.15; the 1.15 accounts for ATR understating
  close-to-close vol on gappy names).

  Held to expiry, valued at intrinsic. This IGNORES:
    - the option bid-ask spread, typically 2-10% of premium
    - commission per contract
    - the chance to sell early and recover time value
    - implied volatility being richer than realised vol, which is usually
      true and makes real options MORE expensive than modelled here

  The first two make real results worse than this shows. The third can make
  them better if you actively manage. On balance treat these numbers as
  OPTIMISTIC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def price_atm(sigma_annual: np.ndarray, dte: int) -> np.ndarray:
    """ATM premium as a fraction of spot (Brenner-Subrahmanyam)."""
    T = dte / 365.0
    return 0.4 * sigma_annual * np.sqrt(T)


def analyse(trades: pd.DataFrame, dte: int, ret_col: str) -> dict:
    r = trades[ret_col].to_numpy(float) / 100.0          # underlying move
    atrp = trades["atr_pct"].to_numpy(float) / 100.0     # daily ATR as fraction
    sigma = atrp * np.sqrt(252) * 1.15                   # annualised vol estimate
    sigma = np.clip(sigma, 0.10, 3.0)

    prem = price_atm(sigma, dte)                         # cost, fraction of spot
    intrinsic = np.maximum(r, 0.0)                       # ATM call at expiry
    pnl = (intrinsic - prem) / prem * 100.0              # % return on premium

    breakeven = prem * 100.0                             # % move needed
    cleared = (r * 100.0) > breakeven

    # Puts, for the bearish side
    intrinsic_p = np.maximum(-r, 0.0)
    pnl_p = (intrinsic_p - prem) / prem * 100.0
    cleared_p = (-r * 100.0) > breakeven

    return {
        "n": len(r),
        "median_move": float(np.median(r) * 100),
        "median_breakeven": float(np.median(breakeven)),
        "call_hit": float(cleared.mean() * 100),
        "call_mean_pnl": float(np.mean(pnl)),
        "call_median_pnl": float(np.median(pnl)),
        "call_total_loss": float((intrinsic <= 0).mean() * 100),
        "put_hit": float(cleared_p.mean() * 100),
        "put_mean_pnl": float(np.mean(pnl_p)),
        "put_median_pnl": float(np.median(pnl_p)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    ap.add_argument("--dte", type=int, default=30, help="days to expiry")
    ap.add_argument("--file", default=None)
    a = ap.parse_args()

    path = Path(a.file) if a.file else HERE / "output" / "backtest_trades.csv"
    if not path.exists():
        raise SystemExit(f"No trades file at {path}\n"
                         f"Run: python scan.py --market {a.market} --backtest")

    tr = pd.read_csv(path)
    need = {"ret_20", "atr_pct"}
    if not need.issubset(tr.columns):
        raise SystemExit(f"{path} is missing {need - set(tr.columns)}")
    tr = tr.dropna(subset=["ret_20", "atr_pct"])
    tr = tr[tr["atr_pct"] > 0]

    print(f"Options on your signals — {len(tr):,} trades, {a.dte} DTE ATM\n")

    hcol = {20: "ret_20", 10: "ret_10", 5: "ret_5"}
    for h in (5, 10, 20):
        col = hcol[h]
        if col not in tr.columns:
            continue
        res = analyse(tr, a.dte, col)
        print(f"  held {h} days")
        print(f"    median move {res['median_move']:+6.2f}%   "
              f"median breakeven needed {res['median_breakeven']:5.2f}%")
        print(f"    CALL  cleared breakeven {res['call_hit']:5.1f}%   "
              f"mean P&L {res['call_mean_pnl']:+7.1f}%   "
              f"expired worthless {res['call_total_loss']:5.1f}%")
        print(f"    PUT   cleared breakeven {res['put_hit']:5.1f}%   "
              f"mean P&L {res['put_mean_pnl']:+7.1f}%")
        print()

    print("  P&L is percent of premium paid. -100% means the option expired worthless.")
    print("  Excludes option spread and commission, so real results are WORSE.")
    print("  Note the put column: your signal is long-only and was never tested")
    print("  for downside, so those numbers show what betting bearish on a")
    print("  bullish signal would have done.")


if __name__ == "__main__":
    main()
