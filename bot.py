#!/usr/bin/env python3
"""bot.py v12 - 7/24 otomatik bot (sanal / paper islem)

Yenilikler v12:
    * Telegram bildirimi (notify.send)
    * MAE/MFE/hold/ret_pct takibi (analytics.trade_stats icin)
    * entry_high/entry_low pozisyon acikken guncellenir

Kullanim:
    python bot.py              # normal calisma (GitHub Actions)
    python bot.py --dry-run    # karar uret, hicbir dosyaya yazma
    python bot.py --force      # borsa saati / veri tazeligi kontrolunu atla (test)
"""
import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

try:
    import numpy as np  # noqa: F401
    import pandas as pd
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "yfinance", "pandas", "numpy"])
    import numpy as np  # noqa: F401
    import pandas as pd
    import yfinance as yf

try:
    from ai_brain import Bot, guru_score, BOT_VERSION
    from self_improve import SelfImprover
    from indicators import add_ind
except ImportError as e:
    print(f"Import hatasi: {e}")
    sys.exit(1)

# notify opsiyonel - yoksa sessizce atlar
try:
    from notify import send as notify_send
except ImportError:
    def notify_send(text):
        return False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bot")

TR = ZoneInfo("Europe/Istanbul")
SYMBOL = os.getenv("BOT_SYMBOL", "THYAO.IS")
STATE = os.getenv("BOT_STATE", "state.json")
BOT_FILE = os.getenv("BOT_FILE", "bot.pkl")
MAX_TRADES = 500
MAX_EQUITY = 1000

DEFAULTS = {
    "cash": 100000.0, "shares": 0, "initial": 100000.0, "trades": [],
    "last_run": None, "generation": 0, "wins": 0, "losses": 0,
    "tp_pct": 6.0, "sl_pct": 3.0, "risk_pct": 0.25, "min_conf": 0.25,
    "max_daily_trades": 4, "cooldown_min": 30,
    "max_drawdown_pct": 20.0, "max_stale_min": 45,
    "peak_value": None, "equity": [],
    # v12 ek alanlar (MAE/MFE takibi)
    "entry_price": None, "entry_cost": None,
    "entry_high": None, "entry_low": None,
    "entry_ts": None, "entry_features": None,
}


def now_tr():
    return datetime.now(TR)


def market_open():
    n = now_tr()
    return n.weekday() < 5 and dtime(9, 55) <= n.time() <= dtime(18, 10)


# ---------------------------------------------------------------- state
def load_state():
    state = {}
    if os.path.exists(STATE):
        try:
            with open(STATE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            log.error("state.json bozuk (%s) -> state.json.bak olarak saklandi", e)
            try:
                os.replace(STATE, STATE + ".bak")
            except OSError:
                pass
            state = {}
    for k, v in DEFAULTS.items():
        state.setdefault(k, list(v) if isinstance(v, list) else v)
    if state["peak_value"] is None:
        state["peak_value"] = state["initial"]
    return state


def atomic_write(path, writer, mode):
    tmp = path + ".tmp"
    with open(tmp, mode, **({"encoding": "utf-8"} if "b" not in mode else {})) as f:
        writer(f)
    os.replace(tmp, path)


def save_state(s):
    atomic_write(STATE, lambda f: json.dump(s, f, indent=2, ensure_ascii=False, default=str), "w")


# ---------------------------------------------------------------- veri
def fetch(period, interval, tries=3):
    for i in range(tries):
        try:
            df = yf.Ticker(SYMBOL).history(period=period, interval=interval)
            if not df.empty:
                return df
        except Exception as e:
            log.warning("yfinance hata (%d/%d): %s", i + 1, tries, e)
        time.sleep(2 ** i)
    return None


def live_price():
    df = fetch("1d", "1m", tries=2)
    if df is None:
        return None
    ts = df.index[-1]
    ts = ts.tz_localize(TR) if ts.tzinfo is None else ts.tz_convert(TR)
    age = (now_tr() - ts.to_pydatetime()).total_seconds() / 60
    return float(df["Close"].iloc[-1]), age


# ---------------------------------------------------------------- bot
def restore_bot(state, df=None):
    """Bot'u diskten geri yukler; entry_high/entry_low state.json'dan gelir (MAE/MFE icin)."""
    bot = None
    if os.path.exists(BOT_FILE):
        try:
            with open(BOT_FILE, "rb") as f:
                bot = pickle.load(f)
            if getattr(bot, "version", 1) != BOT_VERSION:
                log.warning("Eski surum bot (v%s) atildi", getattr(bot, "version", 1))
                bot = None
            else:
                log.info("Bot diskten yuklendi (%s)", BOT_FILE)
        except Exception as e:
            log.warning("Bot yuklenemedi: %s", e)
            bot = None
    if bot is None:
        bot = Bot("Alpha", state["cash"], "balanced")
        m = bot.pretrain(df) if df is not None else None
        if m:
            log.info("On-egitim: %d ornek | dogrulama %%%.0f (taban %%%.0f)",
                     m["n"], m["val_acc"] * 100, m["baseline"] * 100)
    bot.cash = state["cash"]
    bot.shares = state["shares"]
    bot.initial = state["initial"]
    bot.wins, bot.losses = state["wins"], state["losses"]
    bot.tp_pct, bot.sl_pct = state["tp_pct"], state["sl_pct"]
    bot.risk_pct, bot.min_conf = state["risk_pct"], state["min_conf"]
    # v12: pozisyon acikken entry_high/entry_low korunur (MAE/MFE hesabi icin)
    if bot.shares > 0:
        if state.get("entry_price"):
            bot.entry_price = state["entry_price"]
            bot.entry_cost = state.get("entry_cost")
        bot._entry_high = state.get("entry_high") or bot.entry_price or 0.0
        bot._entry_low = state.get("entry_low") or bot.entry_price or 0.0
        bot._entry_ts = state.get("entry_ts")
    else:
        bot.entry_price = bot.entry_cost = bot.entry_features = None
        bot._entry_high = bot._entry_low = None
        bot._entry_ts = None
    return bot


def track_excursion(bot, price):
    """Her cagride entry_high/entry_low'u guncelle (pozisyon acikken)."""
    if bot.shares > 0 and getattr(bot, "_entry_high", None):
        bot._entry_high = max(bot._entry_high, price)
        bot._entry_low = min(bot._entry_low, price)


def trade_extras(bot, price, pnl, cost, date):
    """Kapanan islem icin MAE/MFE/hold/ret_pct hesapla."""
    ret_pct = (pnl / cost * 100) if cost else 0.0
    entry = bot.entry_price or 0
    high = getattr(bot, "_entry_high", entry) or entry
    low = getattr(bot, "_entry_low", entry) or entry
    mae = (low / entry - 1) * 100 if entry else None
    mfe = (high / entry - 1) * 100 if entry else None
    hold = None
    ts = getattr(bot, "_entry_ts", None)
    if ts:
        try:
            t0 = datetime.fromisoformat(ts)
            hold = (now_tr().replace(tzinfo=None) - t0).total_seconds() / 86400
        except Exception:
            pass
    return {"ret_pct": round(ret_pct, 3), "mae": round(mae, 3) if mae is not None else None,
            "mfe": round(mfe, 3) if mfe is not None else None,
            "hold": round(hold, 2) if hold is not None else None}


def risk_guard(state, n, value):
    today = n.strftime("%Y-%m-%d")
    todays = [t for t in state["trades"]
              if str(t.get("ts", "")).startswith(today) and t.get("action") == "AL"]
    if len(todays) >= state["max_daily_trades"]:
        return f"gunluk alis limiti ({len(todays)}/{state['max_daily_trades']})"
    if todays:
        try:
            last_ts = datetime.fromisoformat(todays[-1]["ts"])
            if n - last_ts < timedelta(minutes=state["cooldown_min"]):
                return f"bekleme suresi ({state['cooldown_min']} dk)"
        except Exception:
            pass
    peak = max(state["peak_value"], value)
    dd = (1 - value / peak) * 100 if peak else 0
    if dd >= state["max_drawdown_pct"]:
        return f"devre kesici: zirveden %{dd:.1f} dusus"
    return None


def gh_summary(msg):
    p = os.getenv("GITHUB_STEP_SUMMARY")
    if p:
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    state = load_state()
    n = now_tr()

    raw = fetch("2y", "1d")
    if raw is None:
        log.error("Gunluk veri alinamadi")
        return 1
    raw = raw.reset_index()
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
    df = add_ind(raw)
    if df.empty:
        log.error("Yetersiz veri")
        return 1
    last = df.iloc[-1]

    live = live_price()
    price, age = (live if live else (float(last["Close"]), None))

    bot = restore_bot(state, df)
    bot.last_price = price

    # v12: her calismada entry_high/entry_low guncelle (MAE/MFE)
    track_excursion(bot, price)

    action, probs, feat = bot.decide(last, price)
    gs = guru_score(last, "BUY" if action == "AL" else "SELL")
    log.info("%s %.2f | Karar: %s | P(AL): %.2f | Usta: %%%.0f",
             SYMBOL, price, action, probs[2], gs * 100)

    fresh = age is not None and age <= state["max_stale_min"]
    if args.force:
        tradeable, why = True, ""
    elif not market_open():
        tradeable, why = False, "Borsa kapali"
    elif not fresh:
        tradeable, why = False, f"Canli veri bayat/yok (yas: {age if age is None else round(age)} dk)"
    else:
        tradeable, why = True, ""

    traded = None
    if not tradeable:
        log.info("%s - islem yok", why)
    else:
        exec_action = action
        if action == "AL":
            block = risk_guard(state, n, bot.value(price))
            if block:
                log.info("AL engellendi: %s", block)
                exec_action = "TUT"

        # v12: MAE/MFE icin giris bilgisi state'e kaydedilecek
        prev_cost = bot.entry_cost
        prev_entry = bot.entry_price

        ok = bot.execute(exec_action, price, last["Date"], feat,
                         reason=f"AI {exec_action} | Usta %{gs*100:.0f}")
        if ok:
            t = bot.trades[-1]
            extras = {}
            if t["action"] == "SAT":
                extras = trade_extras(bot, price, t.get("pnl") or 0,
                                      prev_cost or 0, last["Date"])
            elif t["action"] == "AL":
                bot._entry_high = t["price"]
                bot._entry_low = t["price"]
                bot._entry_ts = n.isoformat(timespec="seconds")

            traded = {"date": str(t["date"]), "ts": n.isoformat(timespec="seconds"),
                      "action": t["action"], "price": round(t["price"], 2), "qty": t["qty"],
                      "value": round(t["value"], 2), "reason": t["reason"],
                      "pnl": round(t.get("pnl") or 0, 2)}
            traded.update(extras)
            state["trades"].append(traded)
            log.info("ISLEM: %s @ %.2f", t["action"], price)

            # v12: Telegram bildirimi
            try:
                extra_txt = ""
                if "ret_pct" in extras and extras.get("ret_pct") is not None:
                    extra_txt = f" | PnL: {extras['ret_pct']:+.2f}%"
                notify_send(f"🤖 {SYMBOL} {t['action']}\n"
                            f"Fiyat: {t['price']:.2f} TL\n"
                            f"Adet: {t['qty']}{extra_txt}\n"
                            f"Usta: %{gs*100:.0f}\n"
                            f"Portfoy: {bot.value(price):,.0f} TL")
            except Exception as e:
                log.warning("notify atlandi: %s", e)

        try:
            bot.train(3, 32)
        except Exception as e:
            log.warning("train atlandi: %s", e)

    # durum guncelle
    value = float(bot.value(price))
    state.update(cash=bot.cash, shares=bot.shares, wins=bot.wins, losses=bot.losses,
                 entry_price=bot.entry_price, entry_cost=bot.entry_cost,
                 entry_high=getattr(bot, "_entry_high", None),
                 entry_low=getattr(bot, "_entry_low", None),
                 entry_ts=getattr(bot, "_entry_ts", None))
    state["peak_value"] = max(state["peak_value"], value)
    state["trades"] = state["trades"][-MAX_TRADES:]
    state["last_run"] = n.strftime("%Y-%m-%d %H:%M")
    state["generation"] += 1
    state["last_price"], state["last_action"] = round(price, 2), action
    if tradeable:
        state["equity"].append({"ts": n.isoformat(timespec="minutes"),
                                "value": round(value, 2), "price": round(price, 2)})
        state["equity"] = state["equity"][-MAX_EQUITY:]

    # oneri motoru (v12: GitHub token varsa API ile de calisir)
    try:
        imp = SelfImprover(state,
                           gh_token=os.getenv("GITHUB_TOKEN"),
                           gh_repo=os.getenv("GITHUB_REPOSITORY", "devrantogtay0-blip/thyao_bot"))
        applied = imp.apply_approved_to_state()
        if applied:
            log.info("Uygulanan oneriler: %s", applied)
        new = imp.analyze()
        if new:
            log.info("%d yeni oneri", len(new))
    except Exception as e:
        log.warning("SelfImprove: %s", e)

    if args.dry_run:
        log.info("DRY-RUN: dosyalara yazilmadi | Portfoy: %s TL", f"{value:,.0f}")
        return 0

    save_state(state)
    try:
        atomic_write(BOT_FILE, lambda f: pickle.dump(bot, f), "wb")
    except Exception as e:
        log.warning("Bot kaydedilemedi: %s", e)

    ret = (value / state["initial"] - 1) * 100
    log.info("Portfoy: %s TL (%+.2f%%)", f"{value:,.0f}", ret)
    gh_summary(f"**{SYMBOL}** {price:.2f} | karar `{action}` | "
               f"{'islem: ' + traded['action'] if traded else 'islem yok'} | "
               f"portfoy {value:,.0f} TL ({ret:+.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
