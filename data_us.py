"""
data_us.py — bulk price loading for universes of thousands.

data.py fetches one ticker per request with a small pause. That is right for 60
SGX counters and completely wrong for 6,000 US tickers: it would mean 6,000
round trips, well over an hour, and Yahoo would start returning empty frames
partway through.

This module instead:
  * downloads in batches (yfinance accepts a space-separated list),
  * writes ONE consolidated parquet per day rather than 6,000 small files
    (Windows filesystems handle one 400MB file far better than 6,000 tiny ones),
  * is resumable — an interrupted run picks up from the last completed batch,
  * backs off and retries when Yahoo throttles.

Expect the first full run to take 20-40 minutes and produce a few hundred MB of
cache. Subsequent same-day runs load from parquet in seconds.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache_us"
CACHE_DIR.mkdir(exist_ok=True)

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _store(day: str | None = None, tag: str = "") -> Path:
    day = day or date.today().isoformat()
    return CACHE_DIR / f"us{tag}_{day}.parquet"


def _progress(day: str | None = None, tag: str = "") -> Path:
    day = day or date.today().isoformat()
    return CACHE_DIR / f"us{tag}_{day}_done.txt"


def drop_incomplete_bar(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Remove today's bar if the US session has not closed yet.

    Every indicator in the template reads Close. A bar formed 30 minutes into
    the session is provisional: EMAs, %K, MACD and the Bollinger position will
    all move before 4pm ET. Scanning it produces gates that may be false by
    the bell. This drops that bar so a pre-close run uses the last COMPLETE
    session instead of a half-formed one.
    """
    from datetime import datetime, timezone, timedelta
    now_utc = datetime.now(timezone.utc)
    # US Eastern without a tz library: EDT (UTC-4) Mar-Nov, EST (UTC-5) otherwise.
    offset = -4 if 3 <= now_utc.month <= 11 else -5
    et = now_utc + timedelta(hours=offset)
    closed = (et.hour > 16) or (et.hour == 16 and et.minute >= 0)
    if closed or et.weekday() >= 5:
        return frames
    today = et.date()
    out, trimmed = {}, 0
    for t, d in frames.items():
        if len(d) and pd.Timestamp(d.index[-1]).date() == today:
            d = d.iloc[:-1]
            trimmed += 1
        out[t] = d
    if trimmed:
        print(f"  dropped today's incomplete bar on {trimmed:,} tickers "
              f"(US session still open, {et:%H:%M} ET)")
    return out


def _tidy(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Split a multi-ticker yfinance frame into per-ticker OHLCV frames."""
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out

    if isinstance(raw.columns, pd.MultiIndex):
        present = list(dict.fromkeys(raw.columns.get_level_values(1)))
        for t in present:
            try:
                sub = raw.xs(t, axis=1, level=1)
            except (KeyError, ValueError):
                continue
            keep = [c for c in _OHLCV if c in sub.columns]
            if len(keep) < 5:
                continue
            sub = sub[keep].dropna(subset=["Close"])
            sub = sub[sub["Close"] > 0]
            sub = sub[~sub.index.duplicated(keep="last")].sort_index()
            if len(sub):
                out[t] = sub
    elif len(tickers) == 1:
        keep = [c for c in _OHLCV if c in raw.columns]
        if len(keep) == 5:
            sub = raw[keep].dropna(subset=["Close"])
            sub = sub[sub["Close"] > 0].sort_index()
            if len(sub):
                out[tickers[0]] = sub
    return out


def _flatten(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long-format frame for compact parquet storage."""
    rows = []
    for t, df in frames.items():
        d = df.copy()
        d["ticker"] = t
        d.index.name = "Date"
        rows.append(d.reset_index())
    if not rows:
        return pd.DataFrame(columns=["Date", "ticker"] + _OHLCV)
    return pd.concat(rows, ignore_index=True)


def _unflatten(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for t, g in long.groupby("ticker", sort=False):
        d = g.drop(columns=["ticker"]).set_index("Date").sort_index()
        out[str(t)] = d
    return out


def load_bulk(tickers: list[str], years: int = 6, batch_size: int = 150,
              min_bars: int = 150, use_cache: bool = True,
              pause: float = 1.0, max_retries: int = 3,
              start: str | None = None, end: str | None = None,
              tag: str = "") -> dict[str, pd.DataFrame]:
    """Download history for a large ticker list. Resumable and cached.

    Pass start/end (YYYY-MM-DD) for an explicit window instead of the trailing
    `years`. Use `tag` to keep that cache separate from the main one.
    """
    import yfinance as yf

    store, prog = _store(tag=tag), _progress(tag=tag)

    # ---- reuse today's cache if complete ----
    done: set[str] = set()
    frames: dict[str, pd.DataFrame] = {}
    if use_cache and store.exists():
        try:
            frames = _unflatten(pd.read_parquet(store))
            if prog.exists():
                done = set(prog.read_text().split())
            print(f"  cache: {len(frames):,} tickers already loaded today")
        except Exception:
            frames, done = {}, set()

    todo = [t for t in tickers if t not in done]
    if not todo:
        print("  cache complete, no fetching needed")
    else:
        batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
        print(f"  fetching {len(todo):,} tickers in {len(batches)} batches "
              f"of {batch_size}")

        for bn, batch in enumerate(batches, 1):
            got = {}
            for attempt in range(1, max_retries + 1):
                try:
                    kw = ({"start": start, "end": end} if start
                          else {"period": f"{years}y"})
                    raw = yf.download(
                        " ".join(batch), interval="1d",
                        auto_adjust=True, progress=False, threads=True,
                        group_by="column", **kw,
                    )
                    got = _tidy(raw, batch)
                    if got:
                        break
                except Exception as exc:
                    print(f"    batch {bn} attempt {attempt} failed "
                          f"({type(exc).__name__})")
                if attempt < max_retries:
                    wait = pause * (2 ** attempt)
                    print(f"    backing off {wait:.0f}s")
                    time.sleep(wait)

            frames.update(got)
            done.update(batch)

            # checkpoint every batch so an interrupted run resumes
            try:
                _flatten(frames).to_parquet(store, index=False)
                prog.write_text(" ".join(sorted(done)))
            except Exception as exc:
                print(f"    cache write failed: {type(exc).__name__}")

            pct = bn / len(batches) * 100
            print(f"  batch {bn}/{len(batches)} ({pct:.0f}%) — "
                  f"{len(frames):,} tickers held")
            time.sleep(pause)

    frames = drop_incomplete_bar(frames)
    usable = {t: d for t, d in frames.items() if len(d) >= min_bars}
    print(f"  {len(usable):,} of {len(frames):,} have >= {min_bars} bars")
    return usable


def clear_cache() -> int:
    n = 0
    for f in list(CACHE_DIR.glob("*.parquet")) + list(CACHE_DIR.glob("*_done.txt")):
        f.unlink(missing_ok=True)
        n += 1
    return n
