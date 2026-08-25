# Stock Screener — locked rule set, SGX + US

## START HERE

```bash
pip install -r requirements.txt

# 1. SGX — 66 counters, a few minutes
python scan.py --backtest          # THE GO/NO-GO. Read the output.
python scan.py                     # daily scan, writes output/signals.html

# 2. US — ~5,800 tickers, 20-40 min on first run
python universe_us.py              # fetch the real ticker list
python scan.py --market us --backtest
python scan.py --market us
```

Run SGX after 5:05pm SGT. Run US after 5:30am SGT (4pm ET, next morning your
time). Every indicator is Close-based, so intraday readings are provisional and
gates can flip before the bell.

**Do not trade a signal until you have read a backtest.** The pipeline is
tested; the edge is not. If the 20-day win rate comes back near 50% with a
median near zero, the signal does not work on that market and the correct
response is to stop — not to edit params.

---


Scans SGX counters for the eight-gate setup built from your POEMS Technical View
template, writes an HTML signal sheet, and pushes alerts to Telegram.

---

## The rule set

Indicator settings are taken from your POEMS property dialogs and are **locked**
in `params.py`. There is no config file and no CLI override. Changing a value
requires editing that file, which changes the fingerprint hash stamped on every
report — so parameter drift is loud instead of silent.

| | Setting | Source |
|---|---|---|
| EMA | 8 / 21 / 100, Close | POEMS dialog |
| Bollinger | Simple, 20, 2 SD, Close | POEMS dialog |
| Stochastic | K 9, D 3, MA 3, zones 70/30 | POEMS dialog |
| MACD | 9 / 18 / 9, Close | POEMS dialog |
| Momentum | 28 | POEMS dialog |
| Volume average | 20 | added — template had none |
| ATR | 14 | added — for stops and sizing |

### The eight gates

A name must pass all eight to alert.

1. **REGIME** — close above EMA100, EMAs stacked 8 > 21 > 100
2. **PULLBACK** — a real retrace in the last 10 bars, not a vertical run
3. **TRIGGER** — stochastic %K crossed %D below 60, within the last 5 bars
4. **MACD** — line above zero, histogram positive and expanding
5. **MOMENTUM** — 28-bar momentum above zero
6. **VOLUME** — at least 1.2× the 20-day average since the trigger
7. **LIQUIDITY** — S$300k average daily turnover minimum
8. **HEADROOM** — not already above 90% of the Bollinger channel

Gates 7 and 8 are what keep you out of the Nanofilm-at-90-stochastic trade.

**On the 5-bar confirmation window:** a 9-period stochastic structurally leads a
9/18/9 MACD. Measured across 2,326 test triggers, only 5.5% had MACD confirming
on the same bar; about 29% confirmed within five. Requiring same-bar agreement
was a logic error that discarded 94% of valid setups. The window was set from
that timing measurement, before any return was looked at.

---

## Setup

```bash
pip install yfinance pandas numpy requests pyarrow
```

Verify the ticker list first — it was written from memory, not fetched from SGX:

```bash
python scan.py --verify-universe
```

Delete any line it reports as BAD. To scan the whole exchange, download the
counter list from sgx.com, append `.SI` to each code, and paste them into
`universe.csv`. The liquidity gate discards the untradeable ones anyway.

---

## Daily use

```bash
python scan.py                  # scan, write output/signals.html
python scan.py --notify         # scan and push to Telegram
python scan.py --check MZH.SI   # one counter, gate by gate
python scan.py --rules          # print the locked parameters
```

**Run it after 5:05pm SGT.** Every indicator in the template is Close-based, so
an intraday reading is provisional — gates can flip before the bell.

### Weekly

```bash
python scan.py --backtest
```

Rebuilds `baserates.json` from six years of history. Slow. Once it exists, live
alerts carry the historical hit rate alongside them.

---

## Telegram

1. Message `@BotFather`, send `/newbot`, copy the token.
2. Message your new bot once — it cannot start a conversation with you.
3. Create `secrets.json`:

```json
{"telegram_token": "123456:ABC-your-token", "telegram_chat_id": "987654321"}
```

Don't know your chat id? `python telegram_bot.py --whoami`
Test it: `python telegram_bot.py --test`

Interactive mode — `python telegram_bot.py` — gives you `/scan`, `/check TICKER`,
`/near`, `/rules` from your phone.

**The token is a password.** Keep `secrets.json` out of any repo you push.

---

## Scheduling on Windows

Task Scheduler → Create Task → Triggers: Daily, 5:15pm, weekdays only.
Action: `python.exe` with arguments `scan.py --notify`, "Start in" set to this
folder. Tick "Run whether user is logged on or not".

---

## What this does not do

**It does not predict prices.** You asked for that and I didn't build it,
because a number labelled "target: 1.42" would get used for position sizing and
it would be noise wearing a decimal point. What you get instead:

- **Base rates.** When this exact signal fired historically, what happened over
  5, 10 and 20 days — hit rate, median, quartiles, worst case, and a per-year
  breakdown. If one year carries the whole result, the edge is a regime
  artefact, not a strategy.
- **ATR reference levels.** How far the stock typically travels in 20 days given
  current volatility. That's a range, not a direction.
- **A stop reference.** Recent pullback low minus half an ATR, so sizing has a
  number to work from.

If the backtest comes back near 50% with a median near zero, that is a real
result and it means this signal has no edge on SGX. The correct response is to
stop trading it — not to edit `params.py`.

Also worth knowing: entry is modelled at the next open after the signal bar
closes, never at the signal close. Modelling entry at the close would be
lookahead and would flatter every number in the report.

Yahoo Finance has gaps on thin SGX counters. Confirm on POEMS before acting.
Backtests exclude commission, spread and slippage — on SGX small caps the spread
alone can be a meaningful fraction of a 20-day move.

This is a personal research tool, not financial advice.

---

## US market

Same eight gates, same indicator settings, separate profile in `params_us.py`
with its own fingerprint (`93bab20c088f` vs SGX `6acfe71eb6f4`).

Only two constants differ, and neither is part of the signal:

| | SGX | US |
|---|---|---|
| min average turnover | S$300,000/day | US$20,000,000/day |
| min price | S$0.05 | US$5.00 |

These are tradability floors. S$300k/day is a real liquidity bar on SGX and no
bar at all on US markets. Changing a floor when moving to a market with 100x the
turnover is calibration; lowering it to let one specific stock through is
curve-fitting. Same edit, different intent — know which one you are doing.

`universe_us.py` pulls the live Nasdaq Trader symbol directory rather than a
hand-typed list. Test issues, warrants, units and preferreds are filtered out.
ETFs excluded by default; add `--include-etfs` to keep them.

**Expect roughly 15-20 alerts a day** once ~2,000 US names clear the liquidity
gate. That is a firehose, not an alert system. Decide how to triage it after you
see the real number — ranked top-N, tighter confirm_window, or sector caps.

### Performance

`fast_rules.py` evaluates the gates as column operations instead of bar by bar:
494x faster, which turns a 2.6-hour full-market backtest into about 20 seconds.
It is tested for cell-for-cell agreement with `rules.py` across 21,564 bars in
four market regimes, zero mismatches. If the two ever disagree, `rules.py` is
the authority.

`data_us.py` batches 150 tickers per request, checkpoints after each batch so an
interrupted run resumes, and writes one consolidated parquet rather than 6,000
small files.

### Tests

```bash
python test_indicators.py    # indicator math vs hand-computed values
python test_rules.py         # gate behaviour across scenarios
python test_fast_rules.py    # vectorized == loop, every bar
python test_e2e.py           # full pipeline, synthetic universe
```
