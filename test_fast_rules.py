"""Assert fast_rules.evaluate_all() == rules.evaluate() on every bar.

If this fails, rules.py is correct and fast_rules.py is wrong.
"""
import numpy as np
import pandas as pd

import fast_rules
import indicators as ind
import rules
from params import GATES as G, INDICATORS as I
from rules import GATE_NAMES


def synth(seed, n=600, drift=0.0006, vol=0.019):
    r = np.random.default_rng(seed)
    close = pd.Series(
        1.2 * np.exp(np.cumsum(r.normal(drift, vol, n))),
        index=pd.bdate_range("2021-06-01", periods=n),
    )
    nz = abs(r.normal(0, 0.007, n))
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close),
        "High": close * (1 + nz),
        "Low": close * (1 - nz),
        "Close": close,
        "Volume": pd.Series(r.lognormal(13.4, 0.5, n), index=close.index),
    })


mismatches = 0
compared = 0
gate_mismatch = {g: 0 for g in GATE_NAMES}

# Mix of regimes so every gate gets exercised in both states.
cases = (
    [(s, 0.0006, 0.019) for s in range(200, 215)]
    + [(s, -0.0008, 0.022) for s in range(300, 308)]   # downtrends
    + [(s, 0.0025, 0.012) for s in range(400, 408)]    # strong uptrends
    + [(s, 0.0000, 0.006) for s in range(500, 505)]    # flat / low vol
)

for seed, drift, vol in cases:
    df = ind.build(synth(seed, drift=drift, vol=vol))
    fast = fast_rules.evaluate_all(df, I, G)
    for i in range(1, len(df)):
        slow_gates, slow_fired = rules.evaluate(df, i)
        compared += 1
        for gr in slow_gates:
            if bool(gr.passed) != bool(fast[gr.name].iloc[i]):
                gate_mismatch[gr.name] += 1
                mismatches += 1
        if bool(slow_fired) != bool(fast["FIRED"].iloc[i]):
            mismatches += 1

print(f"bars compared : {compared:,}")
print(f"mismatches    : {mismatches}")
for g, c in gate_mismatch.items():
    if c:
        print(f"   {g}: {c}")

fired_fast = sum(
    int(fast_rules.evaluate_all(ind.build(synth(s, drift=d, vol=v)), I, G)["FIRED"].sum())
    for s, d, v in cases
)
print(f"signals found : {fired_fast}")
print("RESULT        :", "PASS" if mismatches == 0 else "FAIL")
