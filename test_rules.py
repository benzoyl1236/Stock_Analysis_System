"""Construct a synthetic textbook setup and confirm the gates fire on it,
then break each condition one at a time and confirm the right gate fails."""
import numpy as np, pandas as pd
import indicators as ind, rules

def make(uptrend=True, pullback=True, volume=True, liquid=True, extended=False):
    n = 260
    if uptrend:
        base = np.linspace(1.00, 1.60, n)
    else:
        base = np.linspace(1.60, 1.00, n)
    close = base + np.sin(np.arange(n)/9)*0.012
    if pullback:
        close[-12:-4] -= np.linspace(0.005, 0.055, 8)   # dip into EMA21
        close[-4:]    += np.linspace(0.012, 0.050, 4)   # hook back up
    if extended:
        close[-6:] += np.linspace(0.04, 0.20, 6)        # vertical blowoff
    close = pd.Series(close)
    idx = pd.bdate_range("2024-01-01", periods=n)
    close.index = idx
    high = close*1.008; low = close*0.992
    v = pd.Series(np.full(n, 800_000.0), index=idx)
    if volume: v.iloc[-1] = 1_600_000.0
    else:      v.iloc[-1] = 400_000.0
    if not liquid: v[:] = v/900.0
    return ind.build(pd.DataFrame({"Open":close.shift(1).fillna(close),
        "High":high,"Low":low,"Close":close,"Volume":v}))

def report(tag, df):
    # scan the last 8 bars for the best-scoring day
    best = max(range(len(df)-8, len(df)), key=lambda i: rules.score(rules.evaluate(df,i)[0]))
    r,ok = rules.evaluate(df,best)
    return tag, rules.score(r), ok, rules.failed_gates(r)

print(f"{'scenario':<26}{'score':>6}{'alert':>7}  failed")
for tag,kw in [("ideal setup",{}),
               ("downtrend",{"uptrend":False}),
               ("no pullback (vertical)",{"pullback":False}),
               ("thin volume",{"volume":False}),
               ("illiquid counter",{"liquid":False}),
               ("already extended",{"extended":True})]:
    t,s,ok,f = report(tag, make(**kw))
    print(f"{t:<26}{s:>4}/8{str(ok):>7}  {','.join(f) if f else '-'}")
