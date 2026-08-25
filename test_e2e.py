"""End-to-end: fake universe -> backtest -> scan items -> HTML render."""
import numpy as np, pandas as pd, json
import indicators as ind, rules, backtest as bt, report
from dataclasses import asdict

rng = np.random.default_rng(3)
def synth(seed, n=700):
    r = np.random.default_rng(seed)
    ret = r.normal(0.0006, 0.019, n)
    close = pd.Series(1.2*np.exp(np.cumsum(ret)))
    idx = pd.bdate_range("2021-06-01", periods=n); close.index = idx
    nz = abs(r.normal(0, 0.007, n))
    return pd.DataFrame({"Open":close.shift(1).fillna(close),"High":close*(1+nz),
        "Low":close*(1-nz),"Close":close,
        "Volume":pd.Series(r.lognormal(13.4,0.5,n),index=idx)})

uni = {f"T{i:02d}.SI": synth(100+i) for i in range(40)}

# backtest
trades = bt.run(uni)
summary = bt.summarise(trades)
print(f"backtest signals: {summary.get('n',0)}")
for h,s in summary.get("horizons",{}).items():
    print(f"  {h:>2}d  up {s['win_rate']:5.1f}%  median {s['median']:+6.2f}%  worst {s['worst']:+6.2f}%")
print("expectancy_20:", summary.get("expectancy_20"))

# scan last bar
hits, near = [], []
for t, raw in uni.items():
    df = ind.build(raw); i = len(df)-1
    g, fired = rules.evaluate(df, i)
    item = {"ticker":t,"name":"Test Counter","close":float(df["Close"].iloc[i]),
            "score":rules.score(g),"fired":fired,"missing":rules.failed_gates(g),
            "gates":[asdict(x) for x in g],"frame":df}
    if fired or item["score"]>=7: item["levels"]=bt.reference_levels(df, summary)
    if fired: hits.append(item)
    elif item["score"]==7: near.append(item)
print(f"\nscan: {len(hits)} hits, {len(near)} near-miss")

# force one hit so the alert path renders
if not hits and near:
    n = near.pop(0); n["fired"]=True
    for gg in n["gates"]: gg["passed"]=True
    n["score"]=8; hits.append(n)

p = report.render(hits, near, summary, len(uni), "output/signals.html")
size = p.stat().st_size
html = p.read_text()
print(f"\nHTML: {size:,} bytes")
checks = {
 "fingerprint stamped": "RULE FINGERPRINT" in html,
 "gate rail rendered": html.count('class="g ') >= 8,
 "sparkline svg": "<svg" in html,
 "no unescaped braces": "{" not in html.split("<style>")[0],
 "all 8 gate names": all(g in html for g in rules.GATE_NAMES),
 "base rate table": "Horizon" in html,
 "disclaimer": "not financial advice" in html,
}
for k,v in checks.items(): print(f"  {'OK ' if v else 'FAIL'} {k}")

# telegram formatting
import telegram_bot as tg
msg = tg.format_alert(hits[0])
print(f"\ntelegram alert len={len(msg)} (cap 4096)")
print(msg[:400])
print("\nsummary msg:"); print(tg.format_summary(hits, near, len(uni), "abc123")[:300])
