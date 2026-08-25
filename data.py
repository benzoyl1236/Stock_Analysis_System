"""
data.py — price history retrieval with local caching.

Source is Yahoo Finance via yfinance. SGX tickers carry a .SI suffix
(Nanofilm = MZH.SI, AEM = AWX.SI, Genting Singapore = G13.SI). Yahoo's SGX
coverage is adjusted for splits but occasionally has gaps on thin counters;
the loader drops any name that fails the history-length check in params.

Cache lives in ./cache as one parquet file per ticker. A second run on the same
day reuses the cache instead of re-hitting Yahoo, which keeps you well under
their rate limits when scanning a few hundred names.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd

from params import GATES, BACKTEST

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('.', '_')}_{date.today().isoformat()}.parquet"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a yfinance frame to plain OHLCV, ascending, no dupes."""
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance returns a MultiIndex column frame for multi-ticker downloads and
    # sometimes for single ones too. Flatten to the first level.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)

    keep = [c for c in _OHLCV if c in df.columns]
    if len(keep) < 5:
        return pd.DataFrame()

    df = df[keep].copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])
    # Yahoo emits zero-volume placeholder rows on trading halts.
    df = df[df["Close"] > 0]
    return df


def load(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch daily history for one ticker. Returns empty frame on failure."""
    cp = _cache_path(ticker)
    if use_cache and cp.exists():
        try:
            return pd.read_parquet(cp)
        except Exception:
            cp.unlink(missing_ok=True)

    try:
        import yfinance as yf
        raw = yf.download(
            ticker,
            period=f"{BACKTEST.years_of_history}y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        print(f"  ! {ticker}: fetch failed ({type(exc).__name__})")
        return pd.DataFrame()

    df = _clean(raw)
    if not df.empty:
        try:
            df.to_parquet(cp)
        except Exception:
            pass
    return df


def load_universe(tickers: list[str], use_cache: bool = True,
                  pause: float = 0.12) -> dict[str, pd.DataFrame]:
    """Fetch many tickers, skipping any with insufficient history.

    `pause` throttles requests. Yahoo will start returning empties if you hammer
    it; 0.12s between calls handles a few hundred names comfortably.
    """
    out: dict[str, pd.DataFrame] = {}
    total = len(tickers)
    for n, t in enumerate(tickers, 1):
        cached = _cache_path(t).exists()
        df = load(t, use_cache=use_cache)
        if len(df) >= GATES.min_bars_required:
            out[t] = df
        else:
            print(f"  - {t}: skipped ({len(df)} bars, need {GATES.min_bars_required})")
        if n % 25 == 0 or n == total:
            print(f"  loaded {n}/{total} ({len(out)} usable)")
        if not cached:
            time.sleep(pause)
    return out


def read_universe_file(path: str | Path) -> list[str]:
    """Read tickers from a CSV with a 'ticker' column. Blank lines and rows
    starting with # are ignored."""
    df = pd.read_csv(path, comment="#")
    if "ticker" not in df.columns:
        raise ValueError(f"{path} needs a 'ticker' column.")
    return [t.strip() for t in df["ticker"].dropna().astype(str) if t.strip()]


def clear_stale_cache(keep_today: bool = True) -> int:
    """Delete cache files from previous days. Returns count removed."""
    today = date.today().isoformat()
    removed = 0
    for f in CACHE_DIR.glob("*.parquet"):
        if keep_today and f.stem.endswith(today):
            continue
        f.unlink(missing_ok=True)
        removed += 1
    return removed
