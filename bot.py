import json
import os
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo

TR = ZoneInfo("Europe/Istanbul")
STATE_FILE = "state.json"

def now_tr():
    return datetime.now(TR)

def market_open():
    n = now_tr()
    if n.weekday() >= 5:
        return False
    return (9, 55) <= (n.hour, n.minute) <= (18, 10)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "cash": 100000.0,
        "shares": 0,
        "initial": 100000.0,
        "trades": [],
        "weights": {"rsi": 0.35, "macd": 0.35, "trend": 0.30},
        "memory": [],
        "last_run": None,
    }

def save_state(s):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False, default=str)

def get_price_and_row():
    df = yf.Ticker("THYAO.IS").history(period="6mo", interval="1d")
    if df.empty:
        return None, None
    df = df.reset_index()
    c = df["Close"]
    d = c.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + g / l))
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        return None, None
    return float(df["Close"].iloc[-1]), df.iloc[-1].to_dict()

def decide(row, weights):
    sig = {}
    rsi = row.get("RSI", 50)
    sig["rsi"] = 1 if rsi < 35 else (-1 if rsi > 65 else 0)
    mh = row.get("MACD_hist", 0)
    sig["macd"] = 1 if mh > 0 else (-1 if mh < 0 else 0)
    s20, s50 = row.get("SMA20", 0), row.get("SMA50", 0)
    sig["trend"] = 1 if s20 > s50 else (-1 if s20 < s50 else 0)
    score = sum(weights.get(k, 0) * v for k, v in sig.items())
    action = "AL" if score > 0.3 else ("SAT" if score < -0.3 else "TUT")
    return action, score, sig

def execute(state, action, price, reason):
    if action == "AL" and state["cash"] > price * 10:
        qty = int((state["cash"] * 0.25) / price)
        if qty < 1:
            return False
        cost = qty * price
        state["cash"] -= cost
        state["shares"] += qty
        state["trades"].append({
            "date": now_tr().strftime("%Y-%m-%d %H:%M"),
            "action": "AL", "price": price, "qty": qty,
            "total": cost, "cash": state["cash"],
            "shares": state["shares"], "reason": reason
        })
        return True
    if action == "SAT" and state["shares"] > 0:
        qty = state["shares"]
        rev = qty * price
        state["cash"] += rev
        state["shares"] = 0
        state["trades"].append({
            "date": now_tr().strftime("%Y-%m-%d %H:%M"),
            "action": "SAT", "price": price, "qty": qty,
            "total": rev, "cash": state["cash"],
            "shares": 0, "reason": reason
        })
        return True
    return False

def learn(state, was_right):
    state["memory"].append(1 if was_right else 0)
    w = state["weights"]
    for k in w:
        if was_right:
            w[k] = min(0.6, w[k] + 0.008)
        else:
            w[k] = max(0.1, w[k] - 0.008)
    t = sum(w.values())
    for k in w:
        w[k] /= t

def main():
    state = load_state()
    price, row = get_price_and_row()
    if price is None:
        print("Veri alınamadı")
        return
    action, score, sig = decide(row, state["weights"])
    print(f"Fiyat: {price:.2f} | Karar: {action} | Skor: {score:+.2f}")

    if not market_open():
        print("Borsa kapalı, işlem yapılmadı.")
        state["last_run"] = now_tr().strftime("%Y-%m-%d %H:%M")
        save_state(state)
        return

    ok = execute(state, action, price, f"OTOMATIK {action} (skor {score:+.2f})")
    if ok:
        learn(state, True)
        print(f"İşlem yapıldı: {action} @ {price:.2f}")
    else:
        print("İşlem koşulu yok.")

    state["last_run"] = now_tr().strftime("%Y-%m-%d %H:%M")
    save_state(state)

if __name__ == "__main__":
    main()
