# bot.py - v10 7/24 otomatik bot
import json, os, sys
from datetime import datetime
from zoneinfo import ZoneInfo

try:
    from ai_brain import Bot, guru_score
    from self_improve import SelfImprover
except ImportError as e:
    print(f"Import hatasi: {e}"); sys.exit(1)

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    os.system("pip install yfinance pandas numpy -q")
    import yfinance as yf
    import pandas as pd
    import numpy as np

TR = ZoneInfo("Europe/Istanbul")
STATE = "state.json"

def now_tr(): return datetime.now(TR)

def market_open():
    n = now_tr()
    if n.weekday() >= 5: return False
    return (9, 55) <= (n.hour, n.minute) <= (18, 10)

def load_state():
    if os.path.exists(STATE):
        try:
            with open(STATE, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    return {"cash": 100000.0, "shares": 0, "initial": 100000.0, "trades": [],
            "last_run": None, "generation": 0, "wins": 0, "losses": 0,
            "tp_pct": 6.0, "sl_pct": 3.0, "risk_pct": 0.25, "min_conf": 0.25}

def save_state(s):
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False, default=str)

def add_ind(df):
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    d = c.diff()
    g = d.where(d > 0, 0).rolling(14).mean(); ls = (-d.where(d < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + g / ls))
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]
    df["SMA20"] = c.rolling(20).mean(); df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["BB_mid"] = df["SMA20"]; df["BB_std"] = c.rolling(20).std()
    df["BB_up"] = df["BB_mid"] + 2 * df["BB_std"]; df["BB_dn"] = df["BB_mid"] - 2 * df["BB_std"]
    df["BB_width"] = (df["BB_up"] - df["BB_dn"]) / df["BB_mid"]
    lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
    df["K"] = 100 * (c - lo14) / (hi14 - lo14); df["D"] = df["K"].rolling(3).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["WILLR"] = -100 * (hi14 - c) / (hi14 - lo14)
    up = h.diff(); dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr14 = tr.rolling(14).sum()
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(14).sum() / tr14
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(14).sum() / tr14
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["ADX"] = dx.rolling(14).mean()
    df["Vol_SMA"] = v.rolling(20).mean(); df["Vol_ratio"] = v / df["Vol_SMA"]
    df["Returns"] = c.pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252)
    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["EMA21"] = c.ewm(span=21, adjust=False).mean()
    df["Mom10"] = c.pct_change(10); df["Mom30"] = c.pct_change(30)
    df["HL_ratio"] = (h - l) / c
    df["Gap"] = (df["Open"] - c.shift()) / c.shift()
    return df.dropna().reset_index(drop=True)

def main():
    state = load_state()
    n = now_tr()
    try:
        raw = yf.Ticker("THYAO.IS").history(period="2y", interval="1d")
        if raw.empty: print("Veri yok"); return
        raw = raw.reset_index()
        raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
        df = add_ind(raw)
        if df.empty: print("Yetersiz veri"); return
        last = df.iloc[-1]; price = float(last["Close"])
    except Exception as e:
        print(f"Veri hatasi: {e}"); return

    bot = Bot("Alpha", state["cash"], "balanced")
    bot.shares = state["shares"]; bot.initial = state["initial"]
    bot.wins = state.get("wins", 0); bot.losses = state.get("losses", 0)
    bot.tp_pct = state.get("tp_pct", 6.0); bot.sl_pct = state.get("sl_pct", 3.0)
    bot.risk_pct = state.get("risk_pct", 0.25); bot.min_conf = state.get("min_conf", 0.25)

    action, probs, feat = bot.decide(last, price)
    gs = guru_score(last, "BUY" if action == "AL" else "SELL")
    print(f"Fiyat: {price:.2f} | Karar: {action} | P(AL): {probs[2]:.2f} | Usta: %{gs*100:.0f}")

    if not market_open():
        print("Borsa kapali.")
    else:
        ok = bot.execute(action, price, last["Date"], feat, reason=f"AI {action} | Usta %{gs*100:.0f}")
        if ok:
            state["cash"] = bot.cash; state["shares"] = bot.shares
            state["wins"] = bot.wins; state["losses"] = bot.losses
            t = bot.trades[-1]
            state["trades"].append({"date": str(t["date"]), "action": t["action"],
                                     "price": round(t["price"], 2), "qty": t["qty"],
                                     "value": round(t["value"], 2), "reason": t["reason"],
                                     "pnl": round(t.get("pnl", 0), 2)})
            print(f"ISLEM: {action} @ {price:.2f}")

    state["last_run"] = n.strftime("%Y-%m-%d %H:%M")
    state["generation"] = state.get("generation", 0) + 1

    # Oneri motoru
    try:
        imp = SelfImprover(state)
        applied = imp.apply_approved_to_state()
        if applied: print(f"Uygulanan oneriler: {applied}")
        new = imp.analyze()
        if new: print(f"{len(new)} yeni oneri")
    except Exception as e:
        print(f"SelfImprove: {e}")

    save_state(state)
    print(f"Portfoy: {bot.value(price):,.0f} TL")

if __name__ == "__main__":
    main()
