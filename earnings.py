"""
earnings.py — post-earnings announcement drift (PEAD).

THE HYPOTHESIS, AND WHY IT IS DIFFERENT FROM EVERYTHING WE TESTED

Every rule so far transformed past prices. Past prices are public and fully
traded, which is why they predicted nothing. PEAD is different: it says the
market UNDER-reacts to earnings news, so a stock that jumps on a surprise keeps
drifting that way for weeks while the information diffuses.

Documented since Ball & Brown (1968) and repeatedly since. It is directional,
the moves are large enough to clear an option breakeven, and the mechanism is
a behavioural one (slow diffusion of information) rather than a chart pattern.

HOW SURPRISE IS MEASURED HERE

Two ways, and the script reports both:

  1. ANALYST SURPRISE — reported EPS vs consensus estimate, from yfinance.
     The textbook measure. yfinance's history is shallow, so the sample is
     smaller.

  2. REACTION — the stock's own move across the announcement. This IS the
     surprise as priced by the market, needs no estimate data, and works on
     every event. Arguably the better signal: a company can beat consensus
     and still fall if guidance was poor, and the reaction captures that.

NO LOOKAHEAD
  The reaction window is [close before announcement -> close after]. Entry is
  the NEXT open after that window closes. You could actually have traded this.

USAGE
    python earnings.py --fetch --top 400     # download earnings dates (slow)
    python earnings.py                       # analyse

The fetch is one API call per ticker, so it is throttled and cached. Start with
the most liquid few hundred names; PEAD is documented across the whole market
but liquid names are what you could actually trade.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import indicators as ind
from baseline import _load_frames

HERE = Path(__file__).parent
CACHE = HERE / "cache_earnings.parquet"
DRIFT_HORIZONS = (5, 20, 60)


# --------------------------------------------------------------------------

def fetch(tickers: list[str], pause: float = 0.35) -> pd.DataFrame:
    """One call per ticker. yfinance returns a handful of recent quarters."""
    import yfinance as yf

    rows, fails = [], 0
    for n, t in enumerate(tickers, 1):
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=40)
        except Exception:
            fails += 1
            df = None
        if df is not None and len(df):
            df = df.reset_index()
            dcol = df.columns[0]
            for _, r in df.iterrows():
                rows.append({
                    "ticker": t,
                    "date": pd.to_datetime(r[dcol], utc=True, errors="coerce"),
                    "eps_est": pd.to_numeric(r.get("EPS Estimate"), errors="coerce"),
                    "eps_act": pd.to_numeric(r.get("Reported EPS"), errors="coerce"),
                    "surprise_pct": pd.to_numeric(r.get("Surprise(%)"), errors="coerce"),
                })
        if n % 25 == 0:
            print(f"  fetched {n}/{len(tickers)}  ({len(rows):,} events, {fails} failures)")
        time.sleep(pause)

    out = pd.DataFrame(rows).dropna(subset=["date"])
    if len(out):
        out["date"] = out["date"].dt.tz_localize(None).dt.normalize()
        out = out.drop_duplicates(subset=["ticker", "date"])
    return out


# --------------------------------------------------------------------------

def build_events(earn: pd.DataFrame, frames: dict, G) -> pd.DataFrame:
    """Join earnings dates to prices; compute reaction and forward drift."""
    recs = []
    for t, grp in earn.groupby("ticker"):
        if t not in frames:
            continue
        try:
            df = ind.build(frames[t])
        except Exception:
            continue
        idx = df.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        idx = pd.DatetimeIndex(idx).normalize()
        close = df["Close"].to_numpy(float)
        openp = df["Open"].to_numpy(float)
        turn = df["turnover_avg"].to_numpy(float)

        for _, e in grp.iterrows():
            pos = idx.searchsorted(e["date"])
            # need a bar before the window and room for the longest horizon
            if pos < 5 or pos + max(DRIFT_HORIZONS) + 3 >= len(idx):
                continue
            if not np.isfinite(turn[pos]) or turn[pos] < G.min_avg_turnover_sgd:
                continue
            if close[pos] < G.min_price_sgd:
                continue

            # Reaction spans the announcement whether it landed pre- or post-market
            pre, post = pos - 1, min(pos + 1, len(close) - 1)
            if close[pre] <= 0:
                continue
            reaction = (close[post] / close[pre] - 1) * 100

            entry_i = post + 1                      # next open, no lookahead
            entry = openp[entry_i]
            if not np.isfinite(entry) or entry <= 0:
                continue

            rec = {"ticker": t, "date": e["date"], "reaction": reaction,
                   "surprise_pct": e.get("surprise_pct", np.nan)}
            for h in DRIFT_HORIZONS:
                j = entry_i + h
                rec[f"d{h}"] = (close[j] / entry - 1) * 100 if j < len(close) else np.nan
            recs.append(rec)

    return pd.DataFrame(recs)


def _table(ev: pd.DataFrame, by: str, label: str, nbuckets: int = 5) -> None:
    d = ev.dropna(subset=[by]).copy()
    if len(d) < nbuckets * 20:
        print(f"\n  {label}: only {len(d):,} events, too few to bucket.")
        return
    try:
        d["bucket"] = pd.qcut(d[by], nbuckets, labels=False, duplicates="drop")
    except ValueError:
        print(f"\n  {label}: could not form buckets.")
        return

    print(f"\n  Drift by {label} (bucket 0 = most negative, "
          f"{nbuckets - 1} = most positive)\n")
    hdr = f"  {'bucket':<9}{'n':>7}{'avg ' + by:>12}"
    for h in DRIFT_HORIZONS:
        hdr += f"{'d' + str(h) + ' med':>10}{'up%':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for b, g in d.groupby("bucket"):
        line = f"  {int(b):<9}{len(g):>7,}{g[by].mean():>11.1f}%"
        for h in DRIFT_HORIZONS:
            col = g[f"d{h}"].dropna()
            if len(col):
                line += f"{col.median():>9.2f}%{(col > 0).mean() * 100:>6.0f}%"
            else:
                line += f"{'—':>10}{'—':>7}"
        print(line)

    top = d[d["bucket"] == d["bucket"].max()]
    bot = d[d["bucket"] == d["bucket"].min()]
    for h in DRIFT_HORIZONS:
        tm, bm = top[f"d{h}"].median(), bot[f"d{h}"].median()
        if np.isfinite(tm) and np.isfinite(bm):
            print(f"    {h:>2}d spread (top minus bottom): {tm - bm:+.2f}pp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download earnings dates")
    ap.add_argument("--top", type=int, default=400, help="most liquid N tickers")
    ap.add_argument("--market", default="us", choices=["us", "sgx"])
    ap.add_argument("--split", action="store_true",
                    help="chronological out-of-sample split (60/40)")
    a = ap.parse_args()

    frames = _load_frames(a.market)
    if a.market == "us":
        from params_us import GATES as G
    else:
        from params import GATES as G

    if a.fetch:
        liq = []
        for t, d in frames.items():
            if len(d) < 300:
                continue
            liq.append((t, float((d["Close"] * d["Volume"]).tail(250).median())))
        liq.sort(key=lambda x: -x[1])
        picks = [t for t, _ in liq[:a.top]]
        print(f"Fetching earnings dates for the {len(picks)} most liquid names")
        earn = fetch(picks)
        if earn.empty:
            raise SystemExit("No earnings data returned.")
        earn.to_parquet(CACHE, index=False)
        print(f"  saved {len(earn):,} events to {CACHE.name}")
        return

    if not CACHE.exists():
        raise SystemExit("No earnings cache. Run: python earnings.py --fetch --top 400")

    earn = pd.read_parquet(CACHE)
    print(f"Post-earnings drift [{a.market.upper()}] — {len(earn):,} raw events")
    ev = build_events(earn, frames, G)
    if ev.empty:
        raise SystemExit("No events matched the price cache.")
    print(f"  {len(ev):,} usable events across {ev['ticker'].nunique()} tickers")
    print(f"  {ev['date'].min().date()} to {ev['date'].max().date()}")

    if a.split:
        ev = ev.sort_values("date")
        cut = int(len(ev) * 0.6)
        tr_, te_ = ev.iloc[:cut], ev.iloc[cut:]
        print(f"\n  OUT-OF-SAMPLE SPLIT")
        print(f"    TRAIN {len(tr_):,} events  {tr_['date'].min().date()} to {tr_['date'].max().date()}")
        print(f"    TEST  {len(te_):,} events  {te_['date'].min().date()} to {te_['date'].max().date()}")
        for lab, part in (("TRAIN", tr_), ("TEST", te_)):
            print(f"\n  ===== {lab} =====")
            _table(part, "reaction", "announcement reaction")
            if part["surprise_pct"].notna().sum() > 200:
                _table(part, "surprise_pct", "analyst EPS surprise")
        print("\n  The TEST spread is the only one that counts. If it collapses")
        print("  relative to TRAIN, the effect was a feature of that period.")
        (HERE / "output").mkdir(parents=True, exist_ok=True)
        ev.to_csv(HERE / "output" / "earnings_events.csv", index=False)
        return

    _table(ev, "reaction", "announcement reaction")
    if ev["surprise_pct"].notna().sum() > 200:
        _table(ev, "surprise_pct", "analyst EPS surprise")
    else:
        print(f"\n  Analyst surprise: only {ev['surprise_pct'].notna().sum():,} "
              f"events have estimate data, skipping.")

    print("\n  What to look for: drift should INCREASE monotonically across")
    print("  buckets. A positive top-minus-bottom spread that grows with the")
    print("  horizon is the PEAD signature. A flat or noisy pattern is not.")
    print("\n  Costs excluded. Compare the spread against the ~1.1% breakeven")
    print("  from spreads.py before concluding anything is tradeable.")
    (HERE / "output").mkdir(parents=True, exist_ok=True)
    ev.to_csv(HERE / "output" / "earnings_events.csv", index=False)
    print(f"\n  Event detail written to output/earnings_events.csv")


if __name__ == "__main__":
    main()
