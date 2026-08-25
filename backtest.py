"""
backtest.py — historical base rates for the locked rule set.

This is the honest replacement for "predict where it will go". No model here
forecasts a price. Instead it answers a narrower, answerable question:

    When this exact eight-gate signal fired in the past, what actually happened
    over the next 5, 10 and 20 trading days?

Output is a distribution — hit rate, median move, worst drawdown before the
best gain — not a target. If the base rate comes back at 50% with a median near
zero, that is a real result and it means the signal has no edge. Do not respond
to that by adjusting params.py.

Method notes:
  - Entry is assumed at the NEXT bar's open after the signal bar closes, because
    the signal cannot be known until the close. Entering at the signal close
    would be lookahead and would flatter every number here.
  - A cooldown (params.BACKTEST.cooldown_bars) stops one sustained trend from
    being counted as twenty independent observations.
  - MFE/MAE are measured intrabar on highs and lows, so they reflect what the
    position actually did, not just closing marks.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

import indicators as ind
import rules


def _fp() -> str:
    """Active fingerprint. scan.activate_market() may swap the profile."""
    import scan
    return getattr(scan, "FINGERPRINT", FINGERPRINT)

from params import BACKTEST as B, FINGERPRINT  # noqa: F401 (rebound by scan.activate_market)


@dataclass
class Trade:
    ticker: str
    signal_date: str
    entry_date: str
    entry: float
    ret_5: float
    ret_10: float
    ret_20: float
    mfe_20: float          # best unrealised gain within 20 bars, %
    mae_20: float          # worst unrealised loss within 20 bars, %
    atr_pct: float
    # --- signal-time features, for ranking. All readable at the signal bar. ---
    stoch_k: float = 0.0          # how deep the oversold dip was
    macd_hist: float = 0.0        # strength of MACD expansion
    mtm: float = 0.0              # 28-bar momentum
    rel_vol: float = 0.0          # volume vs 20-day average
    bb_pos: float = 0.0           # position in the Bollinger channel
    bb_squeeze: float = 0.0       # band width percentile (coiled or not)
    dist_ema21: float = 0.0       # % above EMA21 (how extended)
    dist_ema100: float = 0.0      # % above EMA100 (trend maturity)
    turnover_avg: float = 0.0     # liquidity
    trend_age: float = 0.0        # bars since EMA8 crossed above EMA21


def _f(df, col, i) -> float:
    try:
        v = float(df[col].iloc[i])
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _pct_above(df, col, i) -> float:
    try:
        c, m = float(df["Close"].iloc[i]), float(df[col].iloc[i])
        return (c / m - 1.0) * 100.0 if m > 0 and np.isfinite(m) else 0.0
    except Exception:
        return 0.0


def _trend_age(df, i, cap: int = 120) -> float:
    """Bars since EMA8 last crossed above EMA21. Young trend or old one?"""
    try:
        f = df["ema_fast"].to_numpy(float)
        m = df["ema_mid"].to_numpy(float)
        for k in range(i, max(0, i - cap), -1):
            if k >= 1 and f[k - 1] <= m[k - 1] and f[k] > m[k]:
                return float(i - k)
        return float(cap)
    except Exception:
        return 0.0


def _forward(df: pd.DataFrame, sig_i: int) -> Trade | None:
    """Build a Trade from a signal at position sig_i, or None if truncated."""
    entry_i = sig_i + 1
    horizon = max(B.horizons)
    if entry_i + horizon >= len(df):
        return None

    entry = float(df["Open"].iloc[entry_i])
    if not np.isfinite(entry) or entry <= 0:
        return None

    def ret(h: int) -> float:
        return (float(df["Close"].iloc[entry_i + h - 1]) / entry - 1.0) * 100.0

    window = df.iloc[entry_i:entry_i + horizon]
    mfe = (float(window["High"].max()) / entry - 1.0) * 100.0
    mae = (float(window["Low"].min()) / entry - 1.0) * 100.0

    return Trade(
        ticker="",
        signal_date=str(df.index[sig_i].date()),
        entry_date=str(df.index[entry_i].date()),
        entry=entry,
        ret_5=ret(5), ret_10=ret(10), ret_20=ret(20),
        mfe_20=mfe, mae_20=mae,
        atr_pct=float(df["atr_pct"].iloc[sig_i]),
        stoch_k=_f(df, "stoch_k", sig_i),
        macd_hist=_f(df, "macd_hist", sig_i),
        mtm=_f(df, "mtm", sig_i),
        rel_vol=_f(df, "rel_vol", sig_i),
        bb_pos=_f(df, "bb_pos", sig_i),
        bb_squeeze=_f(df, "bb_squeeze_pct", sig_i),
        dist_ema21=_pct_above(df, "ema_mid", sig_i),
        dist_ema100=_pct_above(df, "ema_slow", sig_i),
        turnover_avg=_f(df, "turnover_avg", sig_i),
        trend_age=_trend_age(df, sig_i),
    )


def run_one(ticker: str, raw: pd.DataFrame) -> list[Trade]:
    """Find every historical signal in one name and measure what followed.

    Gate evaluation is vectorized (fast_rules), which is ~500x faster than the
    bar-by-bar loop and is asserted identical to it by test_fast_rules.py. The
    cooldown is still applied sequentially, because whether a bar is eligible
    depends on which earlier bars already fired.
    """
    df = ind.build(raw)
    trades: list[Trade] = []
    last_signal = -10 ** 9

    try:
        import fast_rules
        fired_col = fast_rules.evaluate_all(df, rules.I, rules.G)["FIRED"].to_numpy()
    except Exception:
        fired_col = None

    for i in range(150, len(df) - 1):
        if i - last_signal < B.cooldown_bars:
            continue
        if fired_col is not None:
            fired = bool(fired_col[i])
        else:
            try:
                _, fired = rules.evaluate(df, i)
            except Exception:
                continue
        if not fired:
            continue
        t = _forward(df, i)
        if t is not None:
            t.ticker = ticker
            trades.append(t)
        last_signal = i

    return trades


def run(universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Backtest the whole universe. Returns a trade-level DataFrame."""
    rows: list[dict] = []
    for n, (ticker, raw) in enumerate(universe.items(), 1):
        for t in run_one(ticker, raw):
            rows.append(asdict(t))
        if n % 25 == 0:
            print(f"  backtested {n}/{len(universe)} ({len(rows)} signals so far)")
    return pd.DataFrame(rows)


def summarise(trades: pd.DataFrame) -> dict:
    """Aggregate base rates. This is what gets shown instead of a forecast."""
    if trades.empty:
        return {"fingerprint": _fp(), "n": 0,
                "note": "No historical signals found. Nothing to base an estimate on."}

    out: dict = {
        "fingerprint": _fp(),
        "n": int(len(trades)),
        "names": int(trades["ticker"].nunique()),
        "first": trades["signal_date"].min(),
        "last": trades["signal_date"].max(),
        "horizons": {},
    }

    for h in B.horizons:
        col = f"ret_{h}"
        s = trades[col].dropna()
        if s.empty:
            continue
        out["horizons"][h] = {
            "win_rate": round(float((s > 0).mean() * 100), 1),
            "median": round(float(s.median()), 2),
            "mean": round(float(s.mean()), 2),
            "p25": round(float(s.quantile(0.25)), 2),
            "p75": round(float(s.quantile(0.75)), 2),
            "best": round(float(s.max()), 2),
            "worst": round(float(s.min()), 2),
        }

    out["mfe_20_median"] = round(float(trades["mfe_20"].median()), 2)
    out["mae_20_median"] = round(float(trades["mae_20"].median()), 2)

    # Expectancy per trade at the 20-bar horizon, in percent.
    s20 = trades["ret_20"].dropna()
    if not s20.empty:
        wins, losses = s20[s20 > 0], s20[s20 <= 0]
        wr = len(wins) / len(s20)
        aw = float(wins.mean()) if len(wins) else 0.0
        al = float(losses.mean()) if len(losses) else 0.0
        out["expectancy_20"] = round(wr * aw + (1 - wr) * al, 3)

    # A signal that only worked in one year is a signal that worked once.
    yr = trades.copy()
    yr["year"] = yr["signal_date"].str[:4]
    by_year = yr.groupby("year")["ret_20"].agg(["count", "median"])
    out["by_year"] = {
        y: {"n": int(r["count"]), "median": round(float(r["median"]), 2)}
        for y, r in by_year.iterrows()
    }

    return out


def reference_levels(df: pd.DataFrame, summary: dict | None = None) -> dict:
    """Volatility-derived reference levels for a live signal.

    These are NOT predictions. An ATR band says how far this stock typically
    travels in twenty days given its current volatility; it says nothing about
    direction. Presented so position sizing and stop placement have a number to
    work from.
    """
    row = df.iloc[-1]
    close = float(row["Close"])
    atr = float(row["atr"])

    # Volatility scales with the square root of time, so a 20-bar expected
    # range is roughly ATR * sqrt(20), not ATR * 20.
    band_20 = atr * np.sqrt(20)

    out = {
        "close": round(close, 4),
        "atr": round(atr, 4),
        "atr_pct": round(float(row["atr_pct"]), 2),
        "range_20_low": round(close - band_20, 4),
        "range_20_high": round(close + band_20, 4),
        "bb_upper": round(float(row["bb_upper"]), 4),
        "ema_mid": round(float(row["ema_mid"]), 4),
        "swing_high_60": round(float(df["High"].iloc[-60:].max()), 4),
        # A stop below the pullback low, sized off volatility rather than a
        # round percentage.
        "stop_ref": round(float(df["Low"].iloc[-10:].min()) - 0.5 * atr, 4),
    }

    if summary and summary.get("n", 0) >= 30:
        h = summary.get("horizons", {}).get(20)
        if h:
            out["base_rate_20d_win"] = h["win_rate"]
            out["base_rate_20d_median"] = h["median"]
            out["base_rate_sample"] = summary["n"]

    return out
