"""
params.py — LOCKED parameters. Do not tune these to fit a trade.

Every number here is frozen at import time. There is no config file, no CLI
override, and no setter. Changing a value requires editing this file, which
changes FINGERPRINT, which invalidates every backtest result you have already
recorded. That is the point: it makes parameter drift loud instead of silent.

PROVENANCE — where each number came from:

  From the POEMS Technical View property dialogs (screenshots, verified):
      EMA 8 / 21 / 100, field = Close
      Bollinger Bands: MA type Simple, period 20, standard deviation 2
      Stochastic Oscillator: K 9, D 3, Moving Average 3, sell 70, buy 30
      MACD: short 9, long 18, signal 9, field = Close
      Momentum: period 28

  Added because the POEMS template had no volume component:
      Volume average period 20, ATR period 14, and the gate thresholds below.
      These are conventional defaults, chosen before any backtest was run.
      They were NOT selected by trying values and keeping what scored best.

If you ever want different settings, do not edit in place. Copy this file to
params_v2.py, change it there, and re-run the full backtest from scratch so the
two rule sets are compared honestly against the same history.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, fields
from typing import Any


class _Frozen:
    """Blocks attribute assignment after construction."""

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"Parameters are locked. Refusing to set '{name}'. "
            "Edit params.py deliberately if you truly mean to change the rule set."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Parameters are locked. Refusing to delete attributes.")


@dataclass(frozen=True)
class Indicators(_Frozen):
    # --- Straight from the POEMS dialogs ---
    ema_fast: int = 8
    ema_mid: int = 21
    ema_slow: int = 100

    bb_period: int = 20
    bb_sd: float = 2.0

    stoch_k: int = 9
    stoch_d: int = 3
    stoch_ma: int = 3
    stoch_sell_zone: float = 70.0
    stoch_buy_zone: float = 30.0

    macd_short: int = 9
    macd_long: int = 18
    macd_signal: int = 9

    momentum: int = 28

    # --- Added: volume and volatility ---
    vol_avg_period: int = 20
    atr_period: int = 14


@dataclass(frozen=True)
class Gates(_Frozen):
    """Pass/fail thresholds for the eight signal gates.

    Locked for the same reason as the indicators. A gate you can loosen after a
    losing week is not a gate.
    """

    # Gate 2 — pullback must have happened within this many bars
    pullback_lookback: int = 10
    # ...price came within this fraction of EMA21, or stochastic entered buy zone
    pullback_ema_tolerance: float = 0.01

    # Gate 3 — stochastic trigger must fire below this level (no chasing)
    stoch_trigger_ceiling: float = 60.0

    # Confirmation window: bars after the stochastic trigger during which the
    # slower gates may still confirm.
    #
    # Rationale, set BEFORE any return was measured: a 9-period stochastic
    # structurally leads a 9/18/9 MACD. Measured on 2,326 synthetic triggers,
    # only 5.5% had MACD confirming on the same bar; ~29% confirmed within 5.
    # Requiring simultaneity is a logic error, not a strictness setting. Five
    # bars is one trading week — long enough for the slower indicator to catch
    # up, short enough that the entry is still near the pullback low.
    confirm_window: int = 5

    # Gate 6 — volume on the trigger bar, as a multiple of the 20-day average
    min_rel_volume: float = 1.20

    # Gate 7 — liquidity floor, average daily turnover in SGD
    min_avg_turnover_sgd: float = 300_000.0
    # ...and a minimum price, to exclude sub-cent tick-noise counters
    min_price_sgd: float = 0.05

    # Gate 8 — reject anything already pinned to the upper band
    max_pct_of_bb_range: float = 0.90

    # Universe hygiene — bars of history required before a name is scoreable
    min_bars_required: int = 150


@dataclass(frozen=True)
class Backtest(_Frozen):
    """Forward-return horizons for base-rate measurement."""

    horizons: tuple = (5, 10, 20)
    years_of_history: int = 6
    # Bars after a signal before the same name can signal again, so one trend
    # does not get counted as twenty independent observations.
    cooldown_bars: int = 20


INDICATORS = Indicators()
GATES = Gates()
BACKTEST = Backtest()


def _fingerprint() -> str:
    """Short hash over every locked value. Stamped onto all output."""
    blob = json.dumps(
        {
            "indicators": asdict(INDICATORS),
            "gates": asdict(GATES),
            "backtest": {k: list(v) if isinstance(v, tuple) else v
                         for k, v in asdict(BACKTEST).items()},
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


FINGERPRINT = _fingerprint()


def describe() -> str:
    """Human-readable dump for the report header and audit trail."""
    lines = [f"Rule set fingerprint: {FINGERPRINT}", ""]
    for label, obj in (("INDICATORS", INDICATORS), ("GATES", GATES), ("BACKTEST", BACKTEST)):
        lines.append(label)
        for f in fields(obj):
            lines.append(f"    {f.name:<26} {getattr(obj, f.name)}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
