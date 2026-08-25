"""
scan.py — the runner. Load universe, evaluate the locked gates, write the
webpage, push Telegram alerts.

Usage:
    python scan.py                     scan and write the report
    python scan.py --notify            also push alerts to Telegram
    python scan.py --backtest          rebuild base rates (slow, do weekly)
    python scan.py --check MZH.SI      evaluate a single counter
    python scan.py --verify-universe   check which tickers actually return data
    python scan.py --rules             print the locked rule set

Run it after the SGX close. Every indicator in the template is Close-based, so
an intraday reading is provisional and the gates can flip before 5pm.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import backtest as bt
import data
import indicators as ind
import report
import rules
import telegram_bot as tg
from params import FINGERPRINT, describe

HERE = Path(__file__).parent
UNIVERSE = HERE / "universe.csv"
OUT_DIR = HERE / "output"
BASERATE_FILE = HERE / "baserates.json"

MARKET = "sgx"


def activate_market(market: str) -> None:
    """Point the runner at a market profile.

    The SGX and US profiles share identical signal gates and identical
    indicator settings; they differ only in the two tradability floors (see
    params_us.py). Each carries its own fingerprint, its own universe file, its
    own cache and its own base rates, so results from one market can never be
    silently mixed into the other.
    """
    global UNIVERSE, BASERATE_FILE, FINGERPRINT, MARKET, describe
    MARKET = market
    if market == "us":
        import params_us as P
        # rules.py reads its thresholds from module-level G / I. Swap them for
        # the US profile. Indicators are byte-identical between profiles, so
        # only the gate thresholds actually change.
        rules.G = P.GATES
        rules.I = P.INDICATORS
        bt.G = P.GATES
        UNIVERSE = HERE / "universe_us.csv"
        BASERATE_FILE = HERE / "baserates_us.json"
        FINGERPRINT = P.FINGERPRINT
        describe = P.describe
    elif market != "sgx":
        raise SystemExit(f"Unknown market '{market}'. Use sgx or us.")


# --------------------------------------------------------------------------

def _load(tickers, use_cache: bool = True):
    """Bulk loader for the US universe, per-ticker loader for SGX.

    Below ~300 names the simple loader is fine and gives clearer per-ticker
    errors. Above that, batching is the only way to avoid Yahoo throttling.
    """
    if MARKET == "us" or len(tickers) > 300:
        import data_us
        return data_us.load_bulk(tickers, use_cache=use_cache)
    return data.load_universe(tickers, use_cache=use_cache)


def _names() -> dict[str, str]:
    try:
        df = pd.read_csv(UNIVERSE, comment="#")
        return dict(zip(df["ticker"], df.get("name", df["ticker"])))
    except Exception:
        return {}


def _load_baserates() -> dict:
    if BASERATE_FILE.exists():
        try:
            return json.loads(BASERATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _evaluate(ticker: str, raw: pd.DataFrame, names: dict,
              summary: dict) -> dict | None:
    """Score one counter on its most recent bar."""
    try:
        df = ind.build(raw)
    except Exception:
        return None
    if len(df) < 150:
        return None

    i = len(df) - 1
    gates, fired = rules.evaluate(df, i)
    item = {
        "ticker": ticker,
        "name": names.get(ticker, ""),
        "close": float(df["Close"].iloc[i]),
        "date": str(df.index[i].date()),
        "score": rules.score(gates),
        "fired": fired,
        "missing": rules.failed_gates(gates),
        "gates": [asdict(g) for g in gates],
        "frame": df,
    }
    if fired or item["score"] >= 7:
        item["levels"] = bt.reference_levels(df, summary)
    return item


# --------------------------------------------------------------------------

def run_scan(notify: bool = False, use_cache: bool = True) -> dict:
    """Full universe scan. Returns hits, near-misses and the report path."""
    names = _names()
    tickers = data.read_universe_file(UNIVERSE)
    summary = _load_baserates()

    print(f"Scanning {len(tickers):,} tickers  [{MARKET.upper()} rules {FINGERPRINT}]")
    universe = _load(tickers, use_cache=use_cache)

    hits, near, all_items = [], [], []
    for t, raw in universe.items():
        item = _evaluate(t, raw, names, summary)
        if item is None:
            continue
        all_items.append(item)
        if item["fired"]:
            hits.append(item)
        elif item["score"] == 7:
            near.append(item)

    hits.sort(key=lambda x: x["ticker"])
    near.sort(key=lambda x: -x["close"])

    OUT_DIR.mkdir(exist_ok=True)
    path = report.render(hits, near, summary, len(universe),
                         OUT_DIR / "signals.html")

    print(f"\n{len(hits)} alert(s), {len(near)} near miss(es), "
          f"{len(universe)} scanned")
    for h in hits:
        print(f"  ALERT  {h['ticker']:<10} {h['close']:.3f}")
    print(f"Report: {path}")

    if notify and tg.configured():
        tg.send(tg.format_summary(hits, near, len(universe), FINGERPRINT))
        for h in hits[:6]:
            tg.send(tg.format_alert(h))
        if hits:
            tg.send_document(path, "Full signal sheet")

    return {"hits": hits, "near": near, "all": all_items,
            "scanned": len(universe), "report_path": str(path),
            "fingerprint": FINGERPRINT}


def check_one(ticker: str) -> dict | None:
    """Evaluate a single counter — used by the bot's /check command."""
    raw = data.load(ticker)
    if raw.empty:
        return None
    item = _evaluate(ticker, raw, _names(), _load_baserates())
    if item and "levels" not in item:
        item["levels"] = bt.reference_levels(ind.build(raw), _load_baserates())
    return item


def run_backtest() -> dict:
    """Rebuild base rates across the universe. Slow — run it weekly, not daily."""
    tickers = data.read_universe_file(UNIVERSE)
    print(f"Backtesting {len(tickers):,} tickers  [{MARKET.upper()} rules {FINGERPRINT}]")
    universe = _load(tickers)
    trades = bt.run(universe)
    summary = bt.summarise(trades)

    BASERATE_FILE.write_text(json.dumps(summary, indent=2))
    if not trades.empty:
        OUT_DIR.mkdir(exist_ok=True)
        trades.to_csv(OUT_DIR / "backtest_trades.csv", index=False)

    print(f"\nSignals found: {summary.get('n', 0)}")
    for h, s in summary.get("horizons", {}).items():
        print(f"  {h:>2}d  up {s['win_rate']:>5.1f}%   median {s['median']:+6.2f}%   "
              f"p25 {s['p25']:+6.2f}%  p75 {s['p75']:+6.2f}%")
    if summary.get("n", 0) < 30:
        print("\n  Fewer than 30 signals — too small a sample to conclude anything.")
    return summary


def verify_universe() -> None:
    tickers = data.read_universe_file(UNIVERSE)
    good, bad = [], []
    for t in tickers:
        df = data.load(t)
        (good if len(df) >= 150 else bad).append((t, len(df)))
        print(f"  {'ok ' if len(df) >= 150 else 'BAD'} {t:<10} {len(df):>5} bars")
    print(f"\n{len(good)} usable, {len(bad)} to remove from universe.csv")
    if bad:
        print("Remove: " + ", ".join(t for t, _ in bad))


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SGX screener, locked rule set.")
    ap.add_argument("--market", default="sgx", choices=["sgx", "us"],
                    help="market profile (default sgx)")
    ap.add_argument("--notify", action="store_true", help="push alerts to Telegram")
    ap.add_argument("--backtest", action="store_true", help="rebuild base rates")
    ap.add_argument("--check", metavar="TICKER", help="evaluate one counter")
    ap.add_argument("--verify-universe", action="store_true")
    ap.add_argument("--rules", action="store_true", help="print locked parameters")
    ap.add_argument("--no-cache", action="store_true", help="force refetch")
    a = ap.parse_args()
    activate_market(a.market)

    if a.rules:
        print(describe()); return
    if a.verify_universe:
        verify_universe(); return
    if a.backtest:
        run_backtest(); return
    if a.check:
        item = check_one(a.check.upper())
        if not item:
            print("No data."); return
        print(f"\n{item['ticker']}  {item['close']:.3f}  "
              f"{item['score']}/8  {'ALERT' if item['fired'] else ''}")
        for g in item["gates"]:
            print(f"  {'PASS' if g['passed'] else 'fail'}  {g['name']:<10} {g['detail']}")
        return

    data.clear_stale_cache()
    run_scan(notify=a.notify, use_cache=not a.no_cache)


if __name__ == "__main__":
    main()
