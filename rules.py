"""
rules.py — the eight signal gates. Thresholds come from params.GATES and are LOCKED.

Design intent: the gates are ordered the way the setup actually develops, from
slowest-moving to fastest. A name must pass all eight on the same bar to alert.
Every gate reports its own pass/fail plus the measured value, so a near-miss
tells you exactly which gate stopped it rather than just disappearing.

Gate order:
    1 REGIME     price above EMA100 and EMAs stacked 8 > 21 > 100
    2 PULLBACK   a genuine retrace happened recently, not a vertical run
    3 TRIGGER    stochastic %K crossed above %D from a non-extended level
    4 MACD       histogram positive and expanding, with MACD above zero
    5 MOMENTUM   28-bar momentum above zero
    6 VOLUME     trigger bar traded above its 20-day average
    7 LIQUIDITY  enough daily turnover to actually get filled
    8 HEADROOM   not already pinned against the upper Bollinger band

Gates 7 and 8 are the ones that keep you out of the Nanofilm-at-90-stochastic
trade: liquidity screens out names you cannot exit, headroom screens out names
where the move already happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from params import GATES as G, INDICATORS as I


GATE_NAMES = ("REGIME", "PULLBACK", "TRIGGER", "MACD", "MOMENTUM",
              "VOLUME", "LIQUIDITY", "HEADROOM")


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


def _safe(value) -> bool:
    """True when a value is usable (not NaN/None)."""
    return value is not None and pd.notna(value)


def evaluate(df: pd.DataFrame, i: int) -> tuple[list[GateResult], bool]:
    """Evaluate all eight gates at integer position `i` of an indicator frame.

    Only data at or before bar `i` is read. Nothing here looks forward.
    Returns (results, all_passed).
    """
    if i < 1 or i >= len(df):
        raise IndexError("Bar index out of range.")

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    res: list[GateResult] = []

    # ---- Gate 1: REGIME -------------------------------------------------
    ok = all(_safe(row[c]) for c in ("Close", "ema_fast", "ema_mid", "ema_slow"))
    stacked = ok and (row["Close"] > row["ema_slow"]
                      and row["ema_fast"] > row["ema_mid"] > row["ema_slow"])
    res.append(GateResult(
        "REGIME", bool(stacked),
        f"C {row['Close']:.3f} / E8 {row['ema_fast']:.3f} / "
        f"E21 {row['ema_mid']:.3f} / E100 {row['ema_slow']:.3f}"
        if ok else "insufficient history"))

    # ---- Gate 2: PULLBACK -----------------------------------------------
    # Within the lookback window, price must have either touched near EMA21 or
    # the stochastic must have entered the buy zone. A vertical run does neither.
    start = max(0, i - G.pullback_lookback + 1)
    win = df.iloc[start:i + 1]
    touched_ema = bool((win["Low"] <= win["ema_mid"] * (1 + G.pullback_ema_tolerance)).any())
    dipped_stoch = bool((win["stoch_k"] <= I.stoch_buy_zone).any())
    pulled = touched_ema or dipped_stoch
    res.append(GateResult(
        "PULLBACK", pulled,
        f"{'EMA21 touch' if touched_ema else ''}"
        f"{' + ' if touched_ema and dipped_stoch else ''}"
        f"{'stoch<' + str(int(I.stoch_buy_zone)) if dipped_stoch else ''}"
        or f"no retrace in {G.pullback_lookback} bars"))

    # ---- Gate 3: TRIGGER ------------------------------------------------
    # The stochastic cross may have occurred on this bar or within the
    # confirmation window, because a 9-period stochastic structurally leads a
    # 9/18/9 MACD. See params.Gates.confirm_window for the measurement.
    trig_at = None
    w_start = max(1, i - G.confirm_window + 1)
    for j in range(i, w_start - 1, -1):
        a, b = df.iloc[j], df.iloc[j - 1]
        if not all(_safe(x) for x in (a["stoch_k"], a["stoch_d"],
                                      b["stoch_k"], b["stoch_d"])):
            continue
        if (b["stoch_k"] <= b["stoch_d"] and a["stoch_k"] > a["stoch_d"]
                and a["stoch_k"] < G.stoch_trigger_ceiling):
            trig_at = j
            break
    if trig_at is not None:
        age = i - trig_at
        detail = (f"%K crossed %D at {df.index[trig_at].date()} "
                  f"({age} bar{'s' if age != 1 else ''} ago, "
                  f"%K was {df.iloc[trig_at]['stoch_k']:.1f})")
    else:
        detail = f"no cross below {G.stoch_trigger_ceiling:.0f} in {G.confirm_window} bars"
    res.append(GateResult("TRIGGER", trig_at is not None, detail))

    # ---- Gate 4: MACD ---------------------------------------------------
    have = all(_safe(x) for x in (row["macd"], row["macd_hist"], prev["macd_hist"]))
    macd_ok = have and (row["macd"] > 0
                        and row["macd_hist"] > 0
                        and row["macd_hist"] > prev["macd_hist"])
    res.append(GateResult(
        "MACD", bool(macd_ok),
        f"line {row['macd']:+.4f} hist {row['macd_hist']:+.4f} "
        f"({'expanding' if have and row['macd_hist'] > prev['macd_hist'] else 'contracting'})"
        if have else "no data"))

    # ---- Gate 5: MOMENTUM -----------------------------------------------
    mtm_ok = _safe(row["mtm"]) and row["mtm"] > 0
    res.append(GateResult(
        "MOMENTUM", bool(mtm_ok),
        f"{row['mtm']:+.3f} over {I.momentum} bars" if _safe(row["mtm"]) else "no data"))

    # ---- Gate 6: VOLUME -------------------------------------------------
    # The participation surge can arrive on any bar of the thrust, not
    # necessarily the exact cross bar.
    v_start = trig_at if trig_at is not None else i
    v_win = df["rel_vol"].iloc[v_start:i + 1].dropna()
    peak_rv = float(v_win.max()) if len(v_win) else float("nan")
    vol_ok = _safe(peak_rv) and peak_rv >= G.min_rel_volume
    res.append(GateResult(
        "VOLUME", bool(vol_ok),
        f"peak {peak_rv:.2f}x avg since trigger (need {G.min_rel_volume:.2f}x)"
        if _safe(peak_rv) else "no data"))

    # ---- Gate 7: LIQUIDITY ----------------------------------------------
    liq_ok = (_safe(row["turnover_avg"])
              and row["turnover_avg"] >= G.min_avg_turnover_sgd
              and row["Close"] >= G.min_price_sgd)
    res.append(GateResult(
        "LIQUIDITY", bool(liq_ok),
        f"S${row['turnover_avg']:,.0f}/day avg"
        if _safe(row["turnover_avg"]) else "no data"))

    # ---- Gate 8: HEADROOM -----------------------------------------------
    head_ok = _safe(row["bb_pos"]) and row["bb_pos"] <= G.max_pct_of_bb_range
    res.append(GateResult(
        "HEADROOM", bool(head_ok),
        f"{row['bb_pos'] * 100:.0f}% of band range"
        if _safe(row["bb_pos"]) else "no data"))

    return res, all(r.passed for r in res)


def score(results: list[GateResult]) -> int:
    """How many gates passed. Used to rank near-misses, never to relax a gate."""
    return sum(1 for r in results if r.passed)


def failed_gates(results: list[GateResult]) -> list[str]:
    return [r.name for r in results if not r.passed]
