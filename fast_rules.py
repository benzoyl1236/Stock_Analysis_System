"""
fast_rules.py — the same eight gates as rules.py, evaluated across every bar at
once instead of one bar at a time.

Why this exists
---------------
rules.evaluate() walks bar by bar in Python. That is fine for 60 SGX counters.
A US scan is ~6,000 tickers x ~1,500 bars = 9 million evaluations, which the
loop version cannot do in reasonable time. This module produces identical
results using column operations.

THE GATES ARE NOT REDEFINED HERE. This is a performance rewrite, nothing else.
test_fast_rules.py asserts cell-for-cell agreement with rules.evaluate() across
every bar of many synthetic series. If the two ever disagree, rules.py is the
authority and this file is wrong.

Thresholds still come from a locked params profile, passed in as `P`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rules import GATE_NAMES


def evaluate_all(df: pd.DataFrame, I, G) -> pd.DataFrame:
    """Boolean gate matrix for every bar of `df`.

    Returns a DataFrame indexed like `df` with one column per gate name plus
    'FIRED' (all eight true) and 'SCORE' (count passed).

    `I` = indicator params, `G` = gate params, from a locked profile.
    """
    n = len(df)
    idx = df.index
    close = df["Close"].to_numpy(float)
    low = df["Low"].to_numpy(float)

    def col(name):
        return df[name].to_numpy(float)

    ema_f, ema_m, ema_s = col("ema_fast"), col("ema_mid"), col("ema_slow")
    k, d = col("stoch_k"), col("stoch_d")
    macd, hist = col("macd"), col("macd_hist")
    mtm, rel_vol = col("mtm"), col("rel_vol")
    turn_avg, bb_pos = col("turnover_avg"), col("bb_pos")

    def nz(a):
        """NaN-safe: NaN never passes a gate."""
        return np.nan_to_num(a, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

    # ---- Gate 1: REGIME --------------------------------------------------
    have1 = ~(np.isnan(close) | np.isnan(ema_f) | np.isnan(ema_m) | np.isnan(ema_s))
    regime = have1 & (close > ema_s) & (ema_f > ema_m) & (ema_m > ema_s)

    # ---- Gate 2: PULLBACK ------------------------------------------------
    # Rolling ANY over the lookback window, matching df.iloc[start:i+1] in
    # rules.py (inclusive of the current bar, window shrinks at the head).
    touched = pd.Series(low <= ema_m * (1 + G.pullback_ema_tolerance), index=idx)
    dipped = pd.Series(nz(k) <= I.stoch_buy_zone, index=idx) & pd.Series(~np.isnan(k), index=idx)
    w = G.pullback_lookback
    pullback = (
        touched.rolling(w, min_periods=1).max().astype(bool)
        | dipped.rolling(w, min_periods=1).max().astype(bool)
    ).to_numpy()

    # ---- Gate 3: TRIGGER -------------------------------------------------
    # A cross bar: %K crossed above %D from at-or-below, while %K < ceiling.
    kp, dp = np.roll(k, 1), np.roll(d, 1)
    kp[0] = dp[0] = np.nan
    valid = ~(np.isnan(k) | np.isnan(d) | np.isnan(kp) | np.isnan(dp))
    cross = valid & (kp <= dp) & (k > d) & (k < G.stoch_trigger_ceiling)
    cross[0] = False  # rules.py requires j >= 1

    # Most recent cross within the confirmation window; trig_age = i - trig_at.
    trig_age = np.full(n, -1, dtype=int)
    for off in range(G.confirm_window):
        shifted = np.zeros(n, dtype=bool)
        if off == 0:
            shifted = cross
        else:
            shifted[off:] = cross[:-off]
        # only fill positions not already claimed by a more recent cross
        trig_age = np.where((trig_age < 0) & shifted, off, trig_age)
    trigger = trig_age >= 0

    # ---- Gate 4: MACD ----------------------------------------------------
    hp = np.roll(hist, 1)
    hp[0] = np.nan
    have4 = ~(np.isnan(macd) | np.isnan(hist) | np.isnan(hp))
    macd_ok = have4 & (macd > 0) & (hist > 0) & (hist > hp)

    # ---- Gate 5: MOMENTUM ------------------------------------------------
    momentum_ok = (~np.isnan(mtm)) & (mtm > 0)

    # ---- Gate 6: VOLUME --------------------------------------------------
    # Peak rel_vol from the trigger bar through the current bar. When no
    # trigger, rules.py uses the current bar alone.
    peak_rv = np.full(n, np.nan)
    span = np.where(trig_age >= 0, trig_age, 0)
    for off in range(G.confirm_window):
        cand = np.full(n, np.nan)
        if off == 0:
            cand = rel_vol
        else:
            cand[off:] = rel_vol[:-off]
        active = span >= off
        peak_rv = np.where(active, np.fmax(np.nan_to_num(peak_rv, nan=-np.inf), 
                                            np.nan_to_num(cand, nan=-np.inf)), peak_rv)
    peak_rv = np.where(np.isneginf(peak_rv), np.nan, peak_rv)
    volume_ok = (~np.isnan(peak_rv)) & (peak_rv >= G.min_rel_volume)

    # ---- Gate 7: LIQUIDITY -----------------------------------------------
    liquidity = ((~np.isnan(turn_avg)) & (turn_avg >= G.min_avg_turnover_sgd)
                 & (close >= G.min_price_sgd))

    # ---- Gate 8: HEADROOM ------------------------------------------------
    headroom = (~np.isnan(bb_pos)) & (bb_pos <= G.max_pct_of_bb_range)

    out = pd.DataFrame(
        {
            "REGIME": regime, "PULLBACK": pullback, "TRIGGER": trigger,
            "MACD": macd_ok, "MOMENTUM": momentum_ok, "VOLUME": volume_ok,
            "LIQUIDITY": liquidity, "HEADROOM": headroom,
        },
        index=idx,
    )
    out["SCORE"] = out[list(GATE_NAMES)].sum(axis=1)
    out["FIRED"] = out[list(GATE_NAMES)].all(axis=1)
    out["trig_age"] = trig_age
    return out
