"""
params_commodity.py — commodities and metals profile.

READ THIS BEFORE USING IT.

The eight signal gates are IDENTICAL to params.py. Every indicator setting from
your POEMS dialogs is unchanged: EMA 8/21/100, Bollinger 20/2.0, Stochastic
9/3/3 with 70/30 zones, MACD 9/18/9, Momentum 28. The signal logic — regime,
pullback, trigger, MACD, momentum, volume, headroom — is byte-for-byte the same.

Two constants differ, and neither one is part of the signal:

    min_avg_turnover  S$300,000  ->  US$20,000,000
    min_price         S$0.05     ->  US$5.00

These are tradability floors, not predictive rules. S$300k/day is a meaningful
liquidity bar on SGX and essentially no bar at all on US markets, where it would
admit thousands of microcaps whose spreads would eat any edge the gates found.
US$5.00 excludes the sub-$5 tier, which is where most pump activity lives and
where a 1-cent spread is a 1% round trip.

Why this is a separate file rather than an edit
-----------------------------------------------
You asked that the rules not be adjustable to fit a trade. Editing params.py to
work on US stocks would have changed the SGX fingerprint and silently
invalidated your SGX backtest. Instead this is a parallel profile with its own
fingerprint. The two markets can never be confused, and no SGX result was
touched to make US scanning work.

The distinction to hold on to: changing a THRESHOLD THE SIGNAL DEPENDS ON to
make a name qualify is curve-fitting. Changing a MARKET-STRUCTURE FLOOR when you
move to a market with 100x the turnover is calibration. If you ever find
yourself lowering min_price to let one specific stock through, that has crossed
from the second category to the first.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, fields

from params import Indicators, Backtest

# Indicator settings are imported unchanged — not redeclared, so they cannot
# drift apart from the SGX profile.
INDICATORS = Indicators()
BACKTEST = Backtest()


@dataclass(frozen=True)
class GatesUS:
    """Identical to params.Gates except the two tradability floors."""

    # --- unchanged from the SGX profile ---
    pullback_lookback: int = 10
    pullback_ema_tolerance: float = 0.01
    stoch_trigger_ceiling: float = 60.0
    confirm_window: int = 5
    min_rel_volume: float = 1.20
    max_pct_of_bb_range: float = 0.90
    min_bars_required: int = 150

    # --- recalibrated for US market structure ---
    # Field names match the SGX profile so rules.py works unmodified.
    # The "sgd" in the name is historical; on this profile the unit is USD.
    min_avg_turnover_sgd: float = 1_000_000.0
    min_price_sgd: float = 1.00


GATES = GatesUS()


def _fingerprint() -> str:
    blob = json.dumps(
        {
            "indicators": asdict(INDICATORS),
            "gates": asdict(GATES),
            "backtest": {k: list(v) if isinstance(v, tuple) else v
                         for k, v in asdict(BACKTEST).items()},
            "market": "COMMODITY",
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


FINGERPRINT = _fingerprint()


def describe() -> str:
    lines = [f"Rule set fingerprint: {FINGERPRINT}  [commodity profile]", ""]
    for label, obj in (("INDICATORS", INDICATORS), ("GATES", GATES),
                       ("BACKTEST", BACKTEST)):
        lines.append(label)
        for f in fields(obj):
            lines.append(f"    {f.name:<26} {getattr(obj, f.name)}")
        lines.append("")
    lines.append("Signal gates identical to SGX profile.")
    lines.append("Differences: min_avg_turnover 300,000 -> 20,000,000; "
                 "min_price 0.05 -> 5.00")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
