"""Log prices by replying to an alert in Telegram.

Reply to any watchflip alert or reminder with what you saw:

    sold 540 575 610        real sale prices (eBay "sold items" by default)
    sold 590 wh             a sale you saw on Willhaben
    c24 450 520             asking prices on Chrono24
    wh 480                  an asking price on Willhaben
    ebay 499                an asking price on eBay
    c24 450 plated          add plated / steel / gold if it differs from the lot,
    wh 520 serviced         and serviced if the watch was sold serviced
    undo                    (as a reply to the bot's confirmation) removes those prices

The bot knows which watch the alert was about, so you never type the
reference. Prices go into market_comps exactly as `python market.py add` would
put them; the material is the lot's own unless you say otherwise. Only
messages from your own chat (TELEGRAM_CHAT_ID) are read.

Runs every minute from run.py serve:  python run.py replies  does one pass.
Nothing here reads any website: it only talks to Telegram's bot API.
"""

from __future__ import annotations

import logging
import os
import re

import db
import market
import model

log = logging.getLogger("replies")

SCHEMA = """
create table if not exists tg_messages (
    message_id  bigint primary key,
    lot_id      text,
    entries     text,                 -- market_comps ids, for "undo" on a confirmation
    sent_at     timestamptz not null default now()
);
alter table tg_messages enable row level security;
create table if not exists bot_state (
    key    text primary key,
    value  text
);
alter table bot_state enable row level security;
"""

CHANNEL_WORDS = {
    "c24": "chrono24", "chrono": "chrono24", "chrono24": "chrono24",
    "wh": "willhaben", "willhaben": "willhaben",
    "ebay": "ebay", "eb": "ebay",
    "insta": "instagram", "instagram": "instagram",
    "dealer": "dealer", "dorotheum": "dorotheum", "catawiki": "catawiki",
}
MATERIAL_WORDS = {"plated": "plated", "gp": "plated", "steel": "steel", "ss": "steel", "gold": "gold"}
SERVICED_WORDS = {"serviced", "service", "revised", "warranty"}
MIN_PRICE, MAX_PRICE = 20, 50_000

HELP = ("Reply to an alert with what you saw, for example:\n"
        "sold 540 575 · c24 450 520 · wh 480 · ebay 499\n"
        "Add plated / steel / gold or serviced if it differs. Reply undo to my confirmation "
        "to remove it.")


def _setup(conn) -> None:
    conn.execute(SCHEMA)
    conn.commit()


def remember(message_id, lot_id: str | None, entries: list[int] | None = None) -> None:
    """Note which lot a sent message was about, so a reply to it can be matched."""
    if not message_id:
        return
    try:
        with db.connect() as conn:
            _setup(conn)
            conn.execute("insert into tg_messages (message_id, lot_id, entries) values (%s, %s, %s) "
                         "on conflict (message_id) do nothing",
                         (int(message_id), lot_id, ",".join(map(str, entries)) if entries else None))
            conn.commit()
    except Exception:
        log.exception("could not remember telegram message %s", message_id)


def _price(tok: str) -> float | None:
    t = tok.strip("€eur").replace("€", "")
    if not re.fullmatch(r"\d[\d.,]*", t):
        return None
    if "," in t and "." in t:                           # 2.450,00 / 2,450.00
        dec = "," if t.rfind(",") > t.rfind(".") else "."
        t = t.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif re.fullmatch(r"\d{1,3}([.,]\d{3})+", t):        # 1.200 / 1,200
        t = re.sub(r"[.,]", "", t)
    else:
        t = t.replace(",", ".")
    try:
        p = float(t)
    except ValueError:
        return None
    return p if MIN_PRICE <= p <= MAX_PRICE else None


def parse(text: str) -> dict | None:
    """'sold 540 575 wh plated' → {kind, channel, prices, material, serviced}."""
    toks = re.split(r"[\s;/]+", text.strip().lower())
    out = {"kind": "ask", "channel": None, "prices": [], "material": None, "serviced": False}
    for t in toks:
        if not t:
            continue
        if t in ("sold", "verkauft", "sale"):
            out["kind"] = "sold"
        elif t in CHANNEL_WORDS:
            out["channel"] = CHANNEL_WORDS[t]
        elif t in MATERIAL_WORDS:
            out["material"] = MATERIAL_WORDS[t]
        elif t in SERVICED_WORDS:
            out["serviced"] = True
        else:
            p = _price(t)
            if p is not None:
                out["prices"].append(p)
    if not out["prices"]:
        return None
    if out["channel"] is None:
        if out["kind"] != "sold":
            return None                     # an ask needs to say where it was seen
        out["channel"] = "ebay"             # sold prices come from eBay's sold items by default
    return out


def _lot(lot_id: str) -> dict | None:
    with db.connect() as conn:
        rows = conn.execute("""
            select l.lot_id, l.reference_number, l.title, l.description,
                   jsonb_path_query_first(l.raw, '$.lotDetailsData.specifications[*] ? (@.name == "Case material").value') #>> '{}' as case_material
            from lots l where l.lot_id = %s""", (lot_id,)).fetchall()
    return rows[0] if rows else None


def _send(token: str, chat: str, text: str, reply_to: int | None = None):
    import requests
    data = {"chat_id": chat, "text": text, "disable_web_page_preview": "true"}
    if reply_to:
        data["reply_to_message_id"] = reply_to
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=20)
    return r.json().get("result", {}).get("message_id") if r.ok else None


def handle(msg: dict, token: str, chat: str) -> None:
    text = (msg.get("text") or "").strip()
    mid = msg.get("message_id")
    replied = (msg.get("reply_to_message") or {}).get("message_id")
    if not text:
        return
    if text.lower() in ("/help", "help", "/start"):
        _send(token, chat, HELP, mid)
        return

    with db.connect() as conn:
        _setup(conn)
        known = conn.execute("select lot_id, entries from tg_messages where message_id = %s",
                             (replied,)).fetchall() if replied else []
    if replied and text.lower() == "undo":
        entries = known[0]["entries"] if known else None
        if not entries:
            _send(token, chat, "Nothing to undo on that message.", mid)
            return
        n = market.drop_ids([int(x) for x in entries.split(",")])
        _send(token, chat, f"Removed {n} price{'s' if n != 1 else ''}.", mid)
        return

    cmd = parse(text)
    if cmd is None:
        if re.search(r"\d", text):
            _send(token, chat, "I could not read that. " + HELP, mid)
        return
    if not known or not known[0]["lot_id"]:
        _send(token, chat, "Reply to an alert (swipe it left) so I know which watch you mean. "
              + HELP, mid)
        return

    lot = _lot(known[0]["lot_id"])
    ref = lot["reference_number"] if lot else None
    if not ref or not model.ref_key(ref):
        _send(token, chat, "That lot has no reference number, so the prices cannot be matched to "
              "other lots. Log them with: python market.py add REFERENCE …", mid)
        return
    mat = cmd["material"]
    if mat is None:
        lm = model.lot_material(lot)
        mat = {"steel": "steel", "plated": "plated", "gold": "gold", "bicolor": "gold"}.get(lm)
    ids = market.log_prices(ref, cmd["channel"], cmd["kind"], cmd["prices"], mat, cmd["serviced"],
                            note="telegram")
    what = "sold price" if cmd["kind"] == "sold" else f"{cmd['channel']} ask"
    n = len(ids)
    reply = (f"✅ Logged {n} {what}{'s' if n > 1 else ''} for {ref}"
             + (f" ({mat}{', serviced' if cmd['serviced'] else ''})" if mat else "")
             + ": " + ", ".join(f"€{p:,.0f}" for p in cmd["prices"])
             + ".\nThey count from the next run. Reply undo to this message to remove them.")
    remember(_send(token, chat, reply, mid), None, ids)
    log.info("logged %d %s for %s from telegram", n, what, ref)


def poll() -> None:
    """One pass: read new messages to the bot, act on replies from your chat."""
    import requests
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    with db.connect() as conn:
        _setup(conn)
        row = conn.execute("select value from bot_state where key = 'tg_offset'").fetchall()
    offset = int(row[0]["value"]) if row else None
    params = {"timeout": 0, "allowed_updates": '["message"]'}
    if offset is not None:
        params["offset"] = offset
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", params=params, timeout=20)
    if not r.ok:
        log.warning("telegram getUpdates %s: %s", r.status_code, r.text[:200])
        return
    updates = r.json().get("result", [])
    if not updates:
        return
    for u in updates:
        msg = u.get("message") or {}
        if str((msg.get("chat") or {}).get("id")) != str(chat):
            continue                         # only your own chat
        try:
            handle(msg, token, chat)
        except Exception:
            log.exception("could not handle telegram message %s", msg.get("message_id"))
    with db.connect() as conn:
        conn.execute("insert into bot_state (key, value) values ('tg_offset', %s) "
                     "on conflict (key) do update set value = excluded.value",
                     (str(max(u["update_id"] for u in updates) + 1),))
        conn.commit()
