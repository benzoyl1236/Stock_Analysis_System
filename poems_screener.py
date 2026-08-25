#!/usr/bin/env python3
"""
POEMS Screener — the fast screener skeleton, running YOUR template.

Keeps what was good about the original: parallel fetching, caching, the rich
dashboard, Telegram alerts, and the CLI flags. Replaces what was wrong.

WHAT CHANGED FROM THE ORIGINAL AND WHY
---------------------------------------
1. SETTINGS NOW MATCH POEMS. The original used EMA 9/21/50, MACD 12/26/9,
   MTM 10 — none of which are yours. Now: EMA 8/21/100, Bollinger 20/2.0,
   Stochastic 9/3/3 with 70/30 zones, MACD 9/18/9, Momentum 28. The
   Stochastic was missing entirely and has been added.

2. THE WIN RATE IS REBUILT. The original computed it from `period="2mo"`,
   which leaves about 23 rows after indicators warm up, then looped
   `iloc[i] for i in range(20)` — the OLDEST rows, not recent ones — and
   bought at `row["close"]`, the very close that generated the signal. That
   is lookahead: you cannot buy at a price you needed in order to know you
   should buy. It now uses 2 years of history, walks the most recent signals,
   and enters at the NEXT open. It returns None rather than a fake 50.0 when
   there are too few samples, so "no data" cannot masquerade as "average".

3. RSI PENALTY BUG FIXED. The original read:
       if rsi > 75:  score -= 15
       elif rsi > 80: score -= 25
   The second branch is unreachable — anything above 80 is already above 75,
   so an RSI of 95 got the same penalty as 78. The order is now reversed.

4. NO HARDCODED CREDENTIALS. The token was in the source. It now comes from
   secrets.json or environment variables.

5. VOLUME ADDED. The POEMS template has no volume indicator, but a move on
   thin volume is not the same as one on real participation.

HONEST NOTE, KEPT ON PURPOSE
-----------------------------
Backtesting this indicator set across 5,205 US stocks over 6 years found no
tradeable edge: +0.48% median 20-day return, ranking inside the random range,
and a pre-registered 2010-2019 holdout that failed at IC -0.004. This tool
scans correctly. Whether the setups it finds are worth trading is a separate
question, and the evidence so far says no. The win rate column is real
history, not a promise.

USAGE
    python poems_screener.py                        scan S&P 500
    python poems_screener.py --fast                 skip win rate (quicker)
    python poems_screener.py --send-alert           Telegram
    python poems_screener.py --no-cache             force fresh data
    python poems_screener.py --max-rsi 50           oversold only
    python poems_screener.py --min-score 80         stricter
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

try:
    import numpy as np
    import pandas as pd
    import requests
    import yfinance as yf
    from rich.console import Console
    from rich.table import Table
    from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                               TimeRemainingColumn)
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install numpy pandas yfinance rich requests lxml")
    sys.exit(1)


# ============================================================
# INTEGRATED WITH THE PROJECT
#
# Indicator settings are NOT redeclared here. They come from params.py, the
# locked ruleset shared with scan.py, so this screener and the scanner can
# never disagree about what "EMA 8" means. Telegram goes through
# telegram_bot.py rather than a second copy of the sending code.
#
# The practical consequence: editing params.py changes the fingerprint and
# affects BOTH tools at once, which is the point of locking it.
# ============================================================
try:
    import indicators as ind
    import telegram_bot as tg
    from params import INDICATORS as POEMS, FINGERPRINT
    INTEGRATED = True
except ImportError:
    INTEGRATED = False
    FINGERPRINT = "standalone"

# Not part of the POEMS template, so defined locally
RSI_PERIOD = 14

# Scoring and filters
MIN_SCORE_FOR_BUY = 70
RSI_WARN, RSI_EXTREME = 75, 85
MIN_WIN_RATE = 40

# Performance
MAX_WORKERS = 20
CACHE_DURATION = 1800
HISTORY = "2y"           # EMA100 needs ~250 bars; 2mo was never enough
MIN_BARS = 150
WINRATE_MIN_SAMPLES = 15

CACHE_DIR = os.path.expanduser("~/.poems_screener_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
USE_CACHE = True


# ============================================================
# Credentials — never hardcoded
# ============================================================
def get_telegram():
    tok = os.environ.get("TELEGRAM_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if tok and chat:
        return tok, chat
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.json")
    if os.path.exists(path):
        try:
            d = json.load(open(path))
            return d.get("telegram_token"), str(d.get("telegram_chat_id", "")) or None
        except Exception:
            pass
    return None, None


# ============================================================
# Universe
# ============================================================
def get_sp500():
    """Live S&P 500 constituents.

    Wikipedia rejects requests without a browser user-agent (HTTP 403), so the
    page is fetched with requests and parsed from the response text rather than
    handing the URL straight to pandas. If that fails, the project's own
    universe_us.csv is a better fallback than a hardcoded list of 52 names.
    """
    try:
        print("Fetching live S&P 500 list...")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=hdrs, timeout=20)
        resp.raise_for_status()
        t = pd.read_html(resp.text)[0]
        tickers = [str(x).replace(".", "-") for x in t["Symbol"].tolist()]
        tickers = [x for x in tickers if x and not any(c in x for c in "^/")]
        if len(tickers) < 100:
            raise ValueError(f"only {len(tickers)} parsed")
        print(f"  {len(tickers)} constituents")
        return tickers
    except Exception as e:
        print(f"  live fetch failed ({e})")

    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_us.csv")
    if os.path.exists(local):
        try:
            t = pd.read_csv(local)["ticker"].dropna().astype(str).tolist()
            print(f"  using universe_us.csv instead: {len(t):,} tickers")
            return t
        except Exception:
            pass
    print("  using built-in fallback list")
    return ["AAPL","MSFT","GOOGL","NVDA","META","AMZN","TSLA","NFLX","ADBE","CRM",
            "AMD","INTC","CSCO","QCOM","TXN","AVGO","IBM","ORCL","JPM","BAC",
            "WFC","GS","MS","V","MA","JNJ","PFE","UNH","MRK","ABBV","LLY",
            "WMT","PG","KO","PEP","COST","HD","MCD","NKE","DIS","XOM","CVX",
            "BA","CAT","GE","HON","UPS","LMT","RTX","DE","UNP","MMM"]


FOREX = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDJPY=X", "USDCAD=X"]


# ============================================================
# Cache
# ============================================================
def _cache_path(ticker, interval, period):
    stamp = datetime.now().strftime("%Y%m%d_%H")
    key = hashlib.md5(f"{ticker}_{interval}_{period}_{stamp}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.pkl")


def load_cache(ticker, interval, period):
    if not USE_CACHE:
        return None
    p = _cache_path(ticker, interval, period)
    if os.path.exists(p):
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(p))
        if age < timedelta(seconds=CACHE_DURATION):
            try:
                return pickle.load(open(p, "rb"))
            except Exception:
                pass
    return None


def save_cache(ticker, interval, period, data):
    if not USE_CACHE:
        return
    try:
        pickle.dump(data, open(_cache_path(ticker, interval, period), "wb"))
    except Exception:
        pass


# ============================================================
# Indicators — exactly the POEMS settings
# ============================================================
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def compute(df):
    """All indicators via the project's tested module, plus RSI.

    indicators.build() is the same code scan.py uses and is covered by
    test_indicators.py, so there is exactly one implementation of the POEMS
    template in this project rather than two that can drift apart.
    """
    if not INTEGRATED:
        raise RuntimeError("indicators.py not found — run from the project folder")

    d = ind.build(df)

    # lowercase aliases so the screener's display code reads naturally
    d["open"], d["high"] = d["Open"], d["High"]
    d["low"], d["close"] = d["Low"], d["Close"]
    d["volume"] = d["Volume"]

    # RSI is not in the POEMS template; it is only used for the score penalty
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    d["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    return d


# ============================================================
# Score — 0-100 from your five indicators plus volume
# ============================================================
def score_row(row, prev=None):
    s = 0
    # Trend structure, 25
    if row["close"] > row["ema_slow"] and row["ema_fast"] > row["ema_mid"] > row["ema_slow"]:
        s += 25
    elif row["close"] > row["ema_slow"]:
        s += 10
    # Stochastic rising out of the buy zone, 20
    k, dd = row["stoch_k"], row["stoch_d"]
    if k > dd and k < 50:
        s += 20
    elif k > dd:
        s += 10
    # MACD above zero and expanding, 20
    if row["macd"] > 0 and row["macd_hist"] > 0:
        s += 15
        if prev is not None and row["macd_hist"] > prev["macd_hist"]:
            s += 5
    # Momentum, 15
    if row["mtm"] > 0:
        s += 15
    # Headroom — NOT already at the top of the channel, 10
    if row["bb_pos"] <= 0.90:
        s += 10
    # Volume confirmation, 10
    if row["rel_vol"] >= 1.2:
        s += 10

    # RSI penalty. Extreme checked FIRST — the original had this backwards,
    # making the -25 branch unreachable.
    r = row.get("rsi", np.nan)
    if pd.notna(r):
        if r > RSI_EXTREME:
            s -= 25
        elif r > RSI_WARN:
            s -= 15
    return max(0, min(100, s))


# ============================================================
# Win rate — rebuilt with no lookahead
# ============================================================
def win_rate(d, hold=5, max_signals=40):
    """Historical hit rate of this score on THIS ticker.

    Walks the most recent signals (not the oldest), enters at the NEXT open
    after the signal bar closes, and exits `hold` bars later. Returns None
    when the sample is too small — a fake 50.0 would be worse than nothing.
    """
    d = d.dropna(subset=["ema_slow", "stoch_d", "macd_hist", "mtm", "bb_pos"])
    if len(d) < MIN_BARS:
        return None, 0

    o = d["open"].to_numpy(float)
    c = d["close"].to_numpy(float)
    wins = total = 0

    # newest first, so a short history still tests recent behaviour
    for i in range(len(d) - hold - 2, 0, -1):
        if total >= max_signals:
            break
        row, prev = d.iloc[i], d.iloc[i - 1]
        if score_row(row, prev) < MIN_SCORE_FOR_BUY:
            continue
        entry = o[i + 1]                     # next open — tradeable
        exit_ = c[i + 1 + hold]
        if not np.isfinite(entry) or entry <= 0:
            continue
        total += 1
        if exit_ > entry:
            wins += 1

    if total < WINRATE_MIN_SAMPLES:
        return None, total
    return round(wins / total * 100, 1), total


# ============================================================
# Fetch and scan
# ============================================================
# Yahoo caps intraday history. 1h is limited to 730 days; daily is unlimited.
# 4h does not exist as a Yahoo interval and is built by resampling 1h bars.
INTERVAL_SPEC = {
    "1d": {"yf": "1d", "period": "2y", "resample": None, "min_bars": 150},
    "1h": {"yf": "1h", "period": "730d", "resample": None, "min_bars": 200},
    "4h": {"yf": "1h", "period": "730d", "resample": "4h", "min_bars": 200},
}


def _to_4h(df):
    """Resample hourly bars to 4-hour. US sessions are 6.5h, so the last bar
    of each day is a partial — that is unavoidable and normal for 4h US data."""
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    cols = {k: v for k, v in agg.items() if k in df.columns}
    out = df.resample("4h").agg(cols).dropna(subset=["Close"])
    return out[out["Close"] > 0]


def fetch(ticker, interval="1d", period=None):
    spec = INTERVAL_SPEC.get(interval, INTERVAL_SPEC["1d"])
    period = period or spec["period"]
    cached = load_cache(ticker, interval, period)
    if cached is not None:
        return ticker, cached
    try:
        data = yf.Ticker(ticker).history(period=period, interval=spec["yf"],
                                         auto_adjust=True)
        if data is None or data.empty:
            return ticker, None
        if spec["resample"]:
            data = _to_4h(data)
        if len(data) >= spec["min_bars"]:
            save_cache(ticker, interval, period, data)
            return ticker, data
    except Exception:
        pass
    return ticker, None


def scan(tickers, interval, fast, quiet):
    out = []
    console = Console() if not quiet else None
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TimeRemainingColumn(), console=console, disable=quiet) as prog:
        task = prog.add_task("[cyan]Scanning", total=len(tickers))
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(fetch, t, interval): t for t in tickers}
            for fut in as_completed(futs):
                ticker, raw = fut.result()
                if raw is not None:
                    try:
                        d = compute(raw).dropna(subset=["ema_slow", "stoch_d",
                                                        "macd_hist", "mtm"])
                        if len(d) >= 2:
                            last, prev = d.iloc[-1], d.iloc[-2]
                            sc = score_row(last, prev)
                            wr, n = (None, 0) if fast or sc < 50 else win_rate(d)
                            if sc >= MIN_SCORE_FOR_BUY:
                                sig = "WEAK" if (wr is not None and wr < MIN_WIN_RATE) else "BUY"
                            elif sc <= 30:
                                sig = "SELL"
                            else:
                                sig = "NEUTRAL"
                            out.append({
                                "Ticker": ticker,
                                "Price": round(float(last["close"]), 2),
                                "Score": int(sc),
                                "Signal": sig,
                                "%K": round(float(last["stoch_k"]), 1),
                                "MACD": round(float(last["macd"]), 4),
                                "MTM": round(float(last["mtm"]), 3),
                                "RSI": round(float(last["rsi"]), 1),
                                "RelVol": round(float(last["rel_vol"]), 2),
                                "Win%": wr,
                                "n": n,
                            })
                    except Exception:
                        pass
                prog.update(task, advance=1)
    return pd.DataFrame(out).sort_values("Score", ascending=False) if out else pd.DataFrame()


# ============================================================
# Output
# ============================================================
def show(df, top):
    console = Console()
    if df.empty:
        console.print("[red]No results[/red]")
        return None
    buys = df[df["Signal"] == "BUY"].head(top)
    if buys.empty:
        console.print("[yellow]No BUY signals. This is the normal outcome most days.[/yellow]")
        near = df[df["Signal"] == "NEUTRAL"].head(5)
        for _, r in near.iterrows():
            console.print(f"  {r['Ticker']}: score {r['Score']}  %K {r['%K']}  RSI {r['RSI']}")
        return None

    t = Table(title=f"POEMS template — {len(buys)} BUY setups", show_lines=False)
    for col in ("#", "Ticker", "Score", "Price", "%K", "MACD", "MTM", "RSI", "RelVol", "Win% (n)"):
        t.add_column(col, justify="right" if col != "Ticker" else "left")
    for i, (_, r) in enumerate(buys.iterrows(), 1):
        kcol = "green" if r["%K"] <= 30 else "red" if r["%K"] >= 70 else "white"
        rcol = "red" if r["RSI"] > 75 else "green" if r["RSI"] < 35 else "white"
        wtxt = f"{r['Win%']:.0f}% ({r['n']})" if r["Win%"] is not None else "n/a"
        t.add_row(str(i), r["Ticker"], str(r["Score"]), f"${r['Price']:.2f}",
                  f"[{kcol}]{r['%K']:.1f}[/{kcol}]", f"{r['MACD']:+.4f}",
                  f"{r['MTM']:+.3f}", f"[{rcol}]{r['RSI']:.0f}[/{rcol}]",
                  f"{r['RelVol']:.2f}", wtxt)
    console.print(t)
    console.print("[dim]Win% is this ticker's own history for this score, entered at the "
                  "next open. 'n/a' means too few samples to say — not 50%.[/dim]")
    return buys


def alert(buys, token, chat):
    if buys is None or buys.empty or not token:
        return
    msg = f"<b>POEMS screener</b>\n<i>{datetime.now():%Y-%m-%d %H:%M}</i>\n\n"
    for _, r in buys.head(10).iterrows():
        w = f" | win {r['Win%']:.0f}%" if r["Win%"] is not None else ""
        msg += f"• <b>{r['Ticker']}</b> ${r['Price']:.2f} | score {r['Score']} | %K {r['%K']:.0f}{w}\n"
    msg += "\n<i>EMA 8/21/100 · BB 20/2 · Stoch 9/3/3 · MACD 9/18/9 · MTM 28</i>"
    if INTEGRATED and tg.configured():
        print("Telegram sent" if tg.send(msg) else "Telegram failed")
        return
    try:
        resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                             json={"chat_id": chat, "text": msg, "parse_mode": "HTML"},
                             timeout=10)
        print("Telegram sent" if resp.status_code == 200 else f"Telegram error {resp.status_code}")
    except Exception as e:
        print(f"Telegram failed: {e}")


def main():
    p = argparse.ArgumentParser(description="POEMS template screener")
    p.add_argument("--add-forex", action="store_true")
    p.add_argument("--interval", default="1d", choices=["1d", "4h", "1h"],
                   help="bar size. Everything was designed and tested on 1d.")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--max-rsi", type=int, default=100)
    p.add_argument("--min-score", type=int, default=70)
    p.add_argument("--send-alert", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--fast", action="store_true", help="skip win rate")
    global USE_CACHE, MIN_SCORE_FOR_BUY
    a = p.parse_args()
    USE_CACHE = not a.no_cache
    MIN_SCORE_FOR_BUY = a.min_score

    print("=" * 62)
    print("POEMS TEMPLATE SCREENER")
    if INTEGRATED:
        print(f"  EMA {POEMS.ema_fast}/{POEMS.ema_mid}/{POEMS.ema_slow} | "
              f"BB {POEMS.bb_period}/{POEMS.bb_sd} | "
              f"Stoch {POEMS.stoch_k}/{POEMS.stoch_d}/{POEMS.stoch_ma} | "
              f"MACD {POEMS.macd_short}/{POEMS.macd_long}/{POEMS.macd_signal} | "
              f"MTM {POEMS.momentum}")
        print(f"  ruleset fingerprint {FINGERPRINT} (shared with scan.py)")
    print("=" * 62)

    if a.interval != "1d":
        bars = {"4h": "about 2 sessions", "1h": "about 2 weeks"}[a.interval]
        print()
        print(f"  !! {a.interval} BARS — the periods below are unchanged, so they now")
        print(f"     mean very different things. EMA 100 becomes {bars} of history,")
        print(f"     Stochastic 9 becomes 9 {a.interval} bars. Expect far more signals")
        print(f"     and far more noise. Nothing here was tested on intraday data,")
        print(f"     and spreads cost the same on a 0.3% move as on a 3% one.")
        if a.interval == "4h":
            print(f"     4h is resampled from 1h; Yahoo has no native 4h interval.")
        print()

    tickers = get_sp500()
    if a.add_forex:
        tickers += FOREX
    tickers = list(dict.fromkeys(tickers))

    if not a.quiet:
        print(f"  {len(tickers)} tickers | {a.interval} | min score {MIN_SCORE_FOR_BUY} "
              f"| cache {'on' if USE_CACHE else 'off'}"
              + (" | FAST (no win rate)" if a.fast else ""))
        print()

    df = scan(tickers, a.interval, a.fast, a.quiet)
    if df.empty:
        print("No data retrieved.")
        return
    if a.max_rsi < 100:
        df = df[df["RSI"] <= a.max_rsi]

    console = Console()
    console.print(f"\n[cyan]Scanned {len(df)} | BUY {len(df[df.Signal=='BUY'])} | "
                  f"WEAK {len(df[df.Signal=='WEAK'])} | avg score {df['Score'].mean():.1f}[/cyan]")

    buys = show(df, a.top)
    if a.send_alert:
        tok, chat = get_telegram()
        if not tok:
            print("No Telegram credentials. Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID, "
                  "or create secrets.json.")
        else:
            alert(buys, tok, chat)


if __name__ == "__main__":
    main()