"""
indicators.py — indicator math matched to the POEMS Technical View template.

All parameters come from params.py and are LOCKED. This module takes no
settings arguments by design: there is no way to call an indicator here with a
period other than the one in the frozen rule set.

Verification against a live POEMS reading (Nanofilm / MZH, 2026-04-21):
    MACD 0.0612, MACD(EMA) 0.0435, MACD(Diff) 0.0177
    0.0612 - 0.0435 = 0.0177, confirming Diff = MACD line - Signal line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from params import INDICATORS as P


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """EMA with alpha = 2/(period+1). adjust=False matches how charting
    platforms seed and roll the average forward."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)


# --------------------------------------------------------------------------
# Template indicators — no tunable arguments
# --------------------------------------------------------------------------

def bollinger(close: pd.Series):
    """Returns (mid, upper, lower, bandwidth_pct). Period 20, 2 SD, Simple MA.

    Population standard deviation (ddof=0), which is what charting packages
    use. Sample stdev would widen the bands and shift every signal.
    """
    mid = sma(close, P.bb_period)
    sd = close.rolling(P.bb_period, min_periods=P.bb_period).std(ddof=0)
    upper = mid + P.bb_sd * sd
    lower = mid - P.bb_sd * sd
    return mid, upper, lower, (upper - lower) / mid * 100.0


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series):
    """Slow stochastic, K 9 / D 3 / MA 3. Returns (percent_k, percent_d).

    raw %K   = 100 * (C - LL9) / (HH9 - LL9)
    %K shown = SMA(raw %K, 3)   <- the 'Moving Average' field in the POEMS dialog
    %D       = SMA(%K shown, 3)
    """
    lowest = low.rolling(P.stoch_k, min_periods=P.stoch_k).min()
    highest = high.rolling(P.stoch_k, min_periods=P.stoch_k).max()
    span = highest - lowest
    raw = np.where(span > 0, 100.0 * (close - lowest) / span, 50.0)
    raw = pd.Series(raw, index=close.index).where(span.notna())
    k = sma(raw, P.stoch_ma)
    return k, sma(k, P.stoch_d)


def macd(close: pd.Series):
    """MACD 9 / 18 / 9. Returns (macd_line, signal_line, diff)."""
    line = ema(close, P.macd_short) - ema(close, P.macd_long)
    sig = ema(line, P.macd_signal)
    return line, sig, line - sig


def momentum(close: pd.Series) -> pd.Series:
    """Momentum 28, in price units — POEMS reports the absolute change, not a
    ratio. Cross-checked: MZH close 0.945 with MTM 0.3 implies 0.645 twenty-
    eight bars earlier."""
    return close - close.shift(P.momentum)


def atr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """ATR 14 with Wilder's smoothing (alpha = 1/period)."""
    return true_range(high, low, close).ewm(alpha=1.0 / P.atr_period, adjust=False).mean()


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Attach every indicator column to an OHLCV frame.

    `df` needs Open/High/Low/Close/Volume on an ascending date index.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Price history must be sorted oldest-first.")

    out = df.copy()
    c, h, l, v = out["Close"], out["High"], out["Low"], out["Volume"]

    out["ema_fast"] = ema(c, P.ema_fast)
    out["ema_mid"] = ema(c, P.ema_mid)
    out["ema_slow"] = ema(c, P.ema_slow)

    out["bb_mid"], out["bb_upper"], out["bb_lower"], out["bb_width"] = bollinger(c)
    out["stoch_k"], out["stoch_d"] = stochastic(h, l, c)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(c)
    out["mtm"] = momentum(c)

    out["atr"] = atr(h, l, c)
    out["atr_pct"] = out["atr"] / c * 100.0

    # Position within the Bollinger channel: 0 = lower band, 1 = upper band.
    span = out["bb_upper"] - out["bb_lower"]
    out["bb_pos"] = ((c - out["bb_lower"]) / span).where(span > 0)

    # --- Volume: the piece the POEMS template was missing ---
    out["vol_avg"] = sma(v, P.vol_avg_period)
    out["rel_vol"] = v / out["vol_avg"]
    out["turnover"] = c * v
    out["turnover_avg"] = sma(out["turnover"], P.vol_avg_period)

    # Bollinger squeeze percentile: where today's bandwidth sits within its own
    # six-month history. Low number = coiled.
    out["bb_squeeze_pct"] = (
        out["bb_width"].rolling(126, min_periods=60).rank(pct=True) * 100.0
    )

    return out
