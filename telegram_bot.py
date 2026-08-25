"""
telegram_bot.py — alerts plus an interactive bot. Uses `requests` only.

Two ways to run it:

  1. Push mode (called by scan.py). After a scan finishes, any alert is sent to
     your chat immediately. This is the part you actually want at 5:15pm.

  2. Poll mode (`python telegram_bot.py`). A long-polling loop so you can query
     from your phone:
         /scan            run the full universe now
         /check MZH.SI    evaluate one counter, gate by gate
         /rules           print the locked rule set and fingerprint
         /near            names sitting at seven of eight gates
         /help

Setup:
  1. Message @BotFather on Telegram, send /newbot, copy the token.
  2. Message your new bot once (it cannot start a conversation with you).
  3. Put the token and your chat id in secrets.json, or set the environment
     variables TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.
  4. `python telegram_bot.py --whoami` prints your chat id if you don't know it.

The token is a password. Anyone holding it can post as your bot, so keep
secrets.json out of any repo you push.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

API = "https://api.telegram.org/bot{token}/{method}"
SECRETS = Path(__file__).parent / "secrets.json"


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------

def _creds() -> tuple[str | None, str | None]:
    token = os.environ.get("TELEGRAM_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat:
        return token, chat
    if SECRETS.exists():
        try:
            d = json.loads(SECRETS.read_text())
            return d.get("telegram_token"), str(d.get("telegram_chat_id", "")) or None
        except Exception:
            pass
    return token, chat


def configured() -> bool:
    t, c = _creds()
    return bool(t and c)


# --------------------------------------------------------------------------
# Sending
# --------------------------------------------------------------------------

def send(text: str, chat_id: str | None = None, silent: bool = False) -> bool:
    """Send a Markdown message. Returns False rather than raising, so a
    notification failure never kills a scan."""
    token, default_chat = _creds()
    chat = chat_id or default_chat
    if not token or not chat:
        print("  ! Telegram not configured — skipping notification.")
        return False

    # Telegram hard-caps messages at 4096 characters.
    for chunk in [text[i:i + 3900] for i in range(0, len(text), 3900)] or [""]:
        try:
            r = requests.post(
                API.format(token=token, method="sendMessage"),
                json={"chat_id": chat, "text": chunk,
                      "parse_mode": "Markdown",
                      "disable_web_page_preview": True,
                      "disable_notification": silent},
                timeout=20,
            )
            if r.status_code != 200:
                print(f"  ! Telegram {r.status_code}: {r.text[:160]}")
                return False
        except Exception as exc:
            print(f"  ! Telegram send failed: {type(exc).__name__}")
            return False
    return True


def send_document(path: str | Path, caption: str = "") -> bool:
    """Attach the HTML sheet so you can open it on your phone."""
    token, chat = _creds()
    if not token or not chat:
        return False
    try:
        with open(path, "rb") as fh:
            r = requests.post(
                API.format(token=token, method="sendDocument"),
                data={"chat_id": chat, "caption": caption[:1000]},
                files={"document": fh}, timeout=60)
        return r.status_code == 200
    except Exception as exc:
        print(f"  ! Telegram document failed: {type(exc).__name__}")
        return False


# --------------------------------------------------------------------------
# Message formatting
# --------------------------------------------------------------------------

def format_alert(item: dict) -> str:
    """One alert. Deliberately includes the stop reference and the base rate,
    because an alert without a risk number invites position sizing by feeling."""
    lv = item.get("levels", {})
    lines = [
        f"*{item['ticker']}* — all 8 gates",
        f"`{item.get('name','')}`" if item.get("name") else "",
        f"Close *{item['close']:.3f}*   ATR {lv.get('atr_pct', 0):.1f}%",
        "",
        "*Gates*",
    ]
    for g in item["gates"]:
        lines.append(f"{'✅' if g['passed'] else '❌'} {g['name']} — {g['detail']}")

    if lv:
        lines += [
            "",
            "*Reference levels* (not targets)",
            f"20d ATR range  `{lv['range_20_low']:.3f} – {lv['range_20_high']:.3f}`",
            f"Stop reference `{lv['stop_ref']:.3f}`  "
            f"({abs(item['close'] - lv['stop_ref']) / item['close'] * 100:.1f}% away)",
            f"Upper band     `{lv['bb_upper']:.3f}`",
        ]
        if "base_rate_20d_win" in lv:
            lines.append(
                f"Base rate 20d  {lv['base_rate_20d_win']:.0f}% up, "
                f"median {lv['base_rate_20d_median']:+.1f}% (n={lv['base_rate_sample']})")

    return "\n".join(l for l in lines if l != "" or True)


def format_summary(hits: list[dict], near: list[dict], scanned: int, fingerprint: str) -> str:
    if not hits:
        head = f"*SGX scan* — no setups\n{scanned} counters, nothing passed all 8 gates."
    else:
        head = (f"*SGX scan* — {len(hits)} alert{'s' if len(hits) != 1 else ''}\n"
                + "\n".join(f"• `{h['ticker']}` {h['close']:.3f}" for h in hits))
    if near:
        head += ("\n\n_7 of 8:_ " +
                 ", ".join(f"`{n['ticker']}`({n['missing'][0]})" for n in near[:8]))
    return head + f"\n\n`rules {fingerprint}`"


# --------------------------------------------------------------------------
# Polling bot
# --------------------------------------------------------------------------

HELP = (
    "*SGX screener bot*\n\n"
    "/scan — run the full universe now (takes a few minutes)\n"
    "/check TICKER — evaluate one counter, e.g. `/check MZH.SI`\n"
    "/near — names at 7 of 8 gates from the last scan\n"
    "/rules — the locked rule set and its fingerprint\n"
    "/help — this message\n\n"
    "_Reference levels are volatility bands, not price targets._"
)


def _handle(text: str, chat: str) -> None:
    """Dispatch one command. Imports are local so the module stays importable
    even if a scan dependency is missing."""
    cmd, _, arg = text.strip().partition(" ")
    cmd = cmd.lower().split("@")[0]

    if cmd in ("/start", "/help"):
        send(HELP, chat); return

    if cmd == "/rules":
        from params import describe
        send(f"```\n{describe()}\n```", chat); return

    if cmd == "/check":
        if not arg.strip():
            send("Give me a ticker, e.g. `/check MZH.SI`", chat); return
        import scan
        send(f"Checking `{arg.strip().upper()}`…", chat, silent=True)
        try:
            item = scan.check_one(arg.strip().upper())
        except Exception as exc:
            send(f"Failed: {type(exc).__name__}", chat); return
        send(format_alert(item) if item else f"No usable data for `{arg}`.", chat)
        return

    if cmd in ("/scan", "/near"):
        import scan
        send("Running the scan — this takes a few minutes.", chat, silent=True)
        try:
            result = scan.run_scan(notify=False)
        except Exception as exc:
            send(f"Scan failed: {type(exc).__name__}: {exc}", chat); return
        if cmd == "/near":
            near = result["near"]
            if not near:
                send("Nothing at 7 of 8.", chat); return
            send("*7 of 8 gates*\n" + "\n".join(
                f"• `{n['ticker']}` {n['close']:.3f} — missing {n['missing'][0]}"
                for n in near[:15]), chat)
            return
        send(format_summary(result["hits"], result["near"],
                            result["scanned"], result["fingerprint"]), chat)
        for h in result["hits"][:6]:
            send(format_alert(h), chat)
        if result.get("report_path"):
            send_document(result["report_path"], "Full signal sheet")
        return

    send("Unknown command. Try /help", chat)


def poll(interval: float = 2.0) -> None:
    token, _ = _creds()
    if not token:
        raise SystemExit("No Telegram token. See the setup notes at the top of this file.")

    print("Bot polling. Ctrl-C to stop.")
    offset = None
    while True:
        try:
            r = requests.get(API.format(token=token, method="getUpdates"),
                             params={"timeout": 30, "offset": offset}, timeout=40)
            for upd in r.json().get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message")
                if not msg or "text" not in msg:
                    continue
                chat = str(msg["chat"]["id"])
                print(f"  <- {chat}: {msg['text'][:60]}")
                try:
                    _handle(msg["text"], chat)
                except Exception as exc:
                    print(f"  ! handler error: {type(exc).__name__}: {exc}")
                    send(f"Something broke: {type(exc).__name__}", chat)
        except KeyboardInterrupt:
            print("\nStopped."); return
        except Exception as exc:
            print(f"  ! poll error: {type(exc).__name__}; retrying in 10s")
            time.sleep(10)
        time.sleep(interval)


def whoami() -> None:
    """Print the chat id of whoever last messaged the bot."""
    token, _ = _creds()
    if not token:
        raise SystemExit("Set TELEGRAM_TOKEN first.")
    r = requests.get(API.format(token=token, method="getUpdates"), timeout=30)
    res = r.json().get("result", [])
    if not res:
        print("No messages yet. Send your bot any message, then run this again.")
        return
    for upd in res[-5:]:
        msg = upd.get("message") or {}
        chat = msg.get("chat", {})
        print(f"chat_id {chat.get('id')}  ({chat.get('first_name') or chat.get('title')})")


if __name__ == "__main__":
    import sys
    if "--whoami" in sys.argv:
        whoami()
    elif "--test" in sys.argv:
        print("sent" if send("Screener bot connected. ✅") else "failed")
    else:
        poll()
