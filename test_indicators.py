"""Validation: check indicator output against independently computed values."""
import numpy as np, pandas as pd
import indicators as ind
from params import INDICATORS as P

rng = np.random.default_rng(7)
n = 400
close = pd.Series(np.cumsum(rng.normal(0, 0.02, n)) + 5.0)
close = close.clip(lower=0.5)
high = close * (1 + abs(rng.normal(0, 0.01, n)))
low  = close * (1 - abs(rng.normal(0, 0.01, n)))
vol  = pd.Series(rng.integers(50_000, 500_000, n).astype(float))
idx  = pd.bdate_range("2023-01-02", periods=n)
for s in (close, high, low, vol): s.index = idx
df = pd.DataFrame({"Open": close.shift(1).fillna(close), "High": high,
                   "Low": low, "Close": close, "Volume": vol})

out = ind.build(df)
fails = []

# 1. EMA recursion, computed manually
alpha = 2/(P.ema_fast+1); man = close.iloc[0]
for x in close.iloc[1:]: man = alpha*x + (1-alpha)*man
if not np.isclose(man, out["ema_fast"].iloc[-1]): fails.append("EMA8 recursion")

# 2. Bollinger: mid must equal 20-bar mean; band half-width = 2 * population sd
w = close.iloc[-20:]
if not np.isclose(w.mean(), out["bb_mid"].iloc[-1]): fails.append("BB mid")
if not np.isclose(w.std(ddof=0)*2, out["bb_upper"].iloc[-1]-out["bb_mid"].iloc[-1]): fails.append("BB 2sd")
if not np.isclose(out["bb_upper"].iloc[-1]-out["bb_mid"].iloc[-1],
                  out["bb_mid"].iloc[-1]-out["bb_lower"].iloc[-1]): fails.append("BB symmetry")

# 3. Stochastic: recompute the last value from scratch
ll = low.rolling(9).min(); hh = high.rolling(9).max()
rawk = 100*(close-ll)/(hh-ll)
k = rawk.rolling(3).mean(); d = k.rolling(3).mean()
if not np.isclose(k.iloc[-1], out["stoch_k"].iloc[-1]): fails.append("Stoch %K")
if not np.isclose(d.iloc[-1], out["stoch_d"].iloc[-1]): fails.append("Stoch %D")
if not (out["stoch_k"].dropna().between(0,100).all()): fails.append("Stoch range 0-100")

# 4. MACD identity: diff == line - signal  (the POEMS cross-check)
if not np.allclose((out["macd"]-out["macd_signal"]).dropna(),
                   out["macd_hist"].dropna()): fails.append("MACD diff identity")

# 5. Momentum: absolute 28-bar change
if not np.isclose(out["mtm"].iloc[-1], close.iloc[-1]-close.iloc[-29]): fails.append("Momentum 28")

# 6. ATR positive, and Wilder alpha (not 2/(n+1))
tr = ind.true_range(high, low, close)
wilder = tr.ewm(alpha=1/14, adjust=False).mean()
if not np.isclose(wilder.iloc[-1], out["atr"].iloc[-1]): fails.append("ATR Wilder")

# 7. bb_pos bounded sensibly, rel_vol sane
if out["bb_pos"].dropna().between(-1.5, 2.5).mean() < 0.99: fails.append("bb_pos range")
if not np.isclose(out["rel_vol"].iloc[-1], vol.iloc[-1]/vol.iloc[-20:].mean()): fails.append("rel_vol")

# 8. NO LOOKAHEAD: indicators at bar t must not change when future bars appear
trunc = ind.build(df.iloc[:300])
for col in ["ema_fast","ema_mid","bb_mid","bb_upper","stoch_k","stoch_d","macd","macd_hist","mtm","atr","rel_vol"]:
    a, b = trunc[col].iloc[250], out[col].iloc[250]
    if not (np.isnan(a) and np.isnan(b)) and not np.isclose(a, b):
        fails.append(f"LOOKAHEAD in {col}")

print("checks run: 8 groups")
print("FAILURES:", fails if fails else "none")
print(f"\nsample last bar: close={close.iloc[-1]:.3f} ema8={out['ema_fast'].iloc[-1]:.3f} "
      f"ema21={out['ema_mid'].iloc[-1]:.3f} ema100={out['ema_slow'].iloc[-1]:.3f}")
print(f"  stoch K={out['stoch_k'].iloc[-1]:.2f} D={out['stoch_d'].iloc[-1]:.2f} "
      f"macd={out['macd'].iloc[-1]:.4f} hist={out['macd_hist'].iloc[-1]:.4f} mtm={out['mtm'].iloc[-1]:.3f}")
