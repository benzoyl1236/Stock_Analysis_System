"""
universe_us.py — build the full US listed universe.

Source is the Nasdaq Trader symbol directory, which is the official listing
file, free, and needs no API key:

    nasdaqlisted.txt   every Nasdaq-listed security
    otherlisted.txt    NYSE, NYSE American, NYSE Arca, BATS, IEX

This is downloaded fresh rather than typed from memory — the SGX list I wrote
by hand had three dead tickers in it (S51, 5CP, 5G3). Don't repeat that.

    python universe_us.py                 common stock only  (~5,800)
    python universe_us.py --include-etfs  add ETFs           (~9,000)
    python universe_us.py --all           everything, warts and all

What gets dropped by default and why:
  * Test issues        — Nasdaq's own dummy symbols, never tradeable
  * ETFs               — the gates were designed for single stocks; an ETF
                         has no earnings and different mean-reversion
  * Non-common classes — warrants, units, rights, preferreds, notes. These
                         have thin books and behave nothing like the equity.
  * Symbols with . or $ — mostly preferreds and when-issued lines

Nothing here filters on liquidity. That is the LIQUIDITY gate's job at scan
time, so the decision stays inside the locked rule set.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
OUT = HERE / "universe_us.csv"

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Suffixes and words that mark a non-common-stock line
_JUNK_WORDS = (
    "warrant", "unit", " right", "preferred", "depositary", "notes due",
    "when issued", "when-issued", "%", "convertible",
)


def _fetch(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    text = r.text
    # Both files end with a "File Creation Time" trailer line.
    lines = [ln for ln in text.splitlines() if not ln.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|")


def build(include_etfs: bool = False, keep_all: bool = False) -> pd.DataFrame:
    nq = _fetch(NASDAQ_URL)
    ot = _fetch(OTHER_URL)

    nq = nq.rename(columns={"Symbol": "ticker", "Security Name": "name"})
    nq["exchange"] = "NASDAQ"
    # 'ACT Symbol' is the tradeable symbol in otherlisted.txt
    ot = ot.rename(columns={"ACT Symbol": "ticker", "Security Name": "name",
                            "Exchange": "exchange"})

    cols = ["ticker", "name", "exchange", "ETF", "Test Issue"]
    for df in (nq, ot):
        for c in cols:
            if c not in df.columns:
                df[c] = "N"

    all_df = pd.concat([nq[cols], ot[cols]], ignore_index=True)
    all_df = all_df.dropna(subset=["ticker"])
    all_df["ticker"] = all_df["ticker"].astype(str).str.strip()
    all_df["name"] = all_df["name"].astype(str)
    before = len(all_df)

    if not keep_all:
        # Nasdaq's own test symbols
        all_df = all_df[all_df["Test Issue"].astype(str).str.upper() != "Y"]
        if not include_etfs:
            all_df = all_df[all_df["ETF"].astype(str).str.upper() != "Y"]
        # Warrants / units / preferreds carry a dot or dollar in the symbol
        all_df = all_df[~all_df["ticker"].str.contains(r"[.$]", regex=True, na=False)]
        low = all_df["name"].str.lower()
        mask = pd.Series(False, index=all_df.index)
        for w in _JUNK_WORDS:
            mask |= low.str.contains(w, regex=False, na=False)
        all_df = all_df[~mask]
        # Five-character Nasdaq symbols ending in W/R/U are warrants/rights/units
        all_df = all_df[~((all_df["ticker"].str.len() == 5)
                          & all_df["ticker"].str[-1].isin(list("WRU")))]

    all_df = all_df.drop_duplicates(subset=["ticker"]).sort_values("ticker")
    print(f"  {before:,} raw -> {len(all_df):,} after filters")
    return all_df[["ticker", "name", "exchange"]].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-etfs", action="store_true")
    ap.add_argument("--all", action="store_true", help="no filtering at all")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    print("Fetching Nasdaq Trader symbol directory...")
    try:
        df = build(include_etfs=a.include_etfs, keep_all=a.all)
    except Exception as exc:
        print(f"\nFailed: {type(exc).__name__}: {exc}")
        print("If nasdaqtrader.com is unreachable, the SEC file is a fallback:")
        print("  https://www.sec.gov/files/company_tickers.json")
        raise SystemExit(1)

    df.to_csv(a.out, index=False)
    print(f"\nWrote {len(df):,} tickers to {a.out}")
    print(df["exchange"].value_counts().to_string())
    print("\nNext:  python scan.py --market us --backtest")


if __name__ == "__main__":
    main()
