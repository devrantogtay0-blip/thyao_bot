import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="THYAO AI Trader", page_icon="✈️", layout="wide")
st.title("✈️ THYAO Yapay Zeka Trader")
st.caption("Türk Hava Yolları - Gerçek veri, hayali para, kendi kendine işlem.")


@st.cache_data(ttl=300)
def get_data(period="6mo"):
    df = yf.Ticker("THYAO.IS").history(period=period, interval="1d")
    if df.empty:
        return None
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def add_indicators(df):
    df = df.copy()
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
    return df.dropna().reset_index(drop=True)


class Bot:
    def __init__(self, cash=100000.0):
        self.cash = cash
        self.shares = 0
        self.initial = cash
        self.trades = []
        self.w = {"rsi": 0.35, "macd": 0.35, "trend": 0.30}
        self.memory = []

    def value(self, price):
        return self.cash + self.shares * price

    def decide(self, row):
        sig = {}
        rsi = row["RSI"]
        sig["rsi"] = 1 if rsi < 35 else (-1 if rsi > 65 else 0)
        mh = row["MACD_hist"]
        sig["macd"] = 1 if mh > 0 else (-1 if mh < 0 else 0)
        sig["trend"] = 1 if row["SMA20"] > row["SMA50"] else (-1 if row["SMA20"] < row["SMA50"] else 0)
        score = sum(self.w[k] * sig[k] for k in sig)
        action = "AL" if score > 0.3 else ("SAT" if score < -0.3 else "TUT")
        return action, score, sig

    def execute(self, action, price, date, reason="AI"):
        if action == "AL" and self.cash > price * 10:
            qty = int((self.cash * 0.25) / price)
            if qty < 1:
                return False
            cost = qty * price
            self.cash -= cost
            self.shares += qty
            self.trades.append({"date": date, "action": "AL", "price": price,
                                "qty": qty, "total": cost, "cash": self.cash,
                                "shares": self.shares, "value": self.value(price),
                                "reason": reason})
            return True
        if action == "SAT" and self.shares > 0:
            qty = self.shares
            rev = qty * price
            self.cash += rev
            self.shares = 0
            self.trades.append({"date": date, "action": "SAT", "price": price,
                                "qty": qty, "total": rev, "cash": self.cash,
                                "shares": 0, "value": self.value(price),
                                "reason": reason})
            return True
        return False

    def learn(self, was_right):
        self.memory.append(1 if was_right else 0)
        lr = 0.08
        for k in self.w:
            if was_right:
                self.w[k] = min(0.6, self.w[k] + lr * 0.1)
            else:
                self.w[k] = max(0.1, self.w[k] - lr * 0.1)
        t = sum(self.w.values())
        for k in self.w:
            self.w[k] /= t


if "bot" not in st.session_state:
    st.session_state.bot = Bot(100000)
bot = st.session_state.bot

df = get_data()
if df is None:
    st.error("THYAO verisi çekilemedi. Birkaç saniye sonra tekrar dene.")
    st.stop()

df = add_indicators(df)
last = df.iloc[-1]
price = float(last["Close"])
date = last["Date"]

with st.sidebar:
    st.header("⚙️ Panel")
    st.metric("THYAO", f"{price:.2f} TL")

    val = bot.value(price)
    pnl = val - bot.initial
    st.metric("Portföy", f"{val:,.0f} TL", f"{pnl:+,.0f} TL")
    st.metric("Nakit", f"{bot.cash:,.0f} TL")
    st.metric("Hisse", f"{bot.shares} adet")

    if bot.memory:
        acc = sum(bot.memory) / len(bot.memory) * 100
        st.metric("AI Doğruluk", f"{acc:.0f}%")
    st.write("**Ağırlıklar:**")
    for k, v in bot.w.items():
        st.progress(min(v, 1.0), text=f"{k}: {v:.2f}")

    st.divider()
    q = st.number_input("Adet", 1, 10000, 10)
    c1, c2 = st.columns(2)
    if c1.button("🟢 AL", use_container_width=True):
        if bot.cash >= q * price:
            bot.cash -= q * price
            bot.shares += q
            bot.trades.append({"date": datetime.now(), "action": "AL", "price": price,
                               "qty": q, "total": q * price, "cash": bot.cash,
                               "shares": bot.shares, "value": bot.value(price),
                               "reason": "Manuel"})
            st.rerun()
    if c2.button("🔴 SAT", use_container_width=True):
        if bot.shares >= q:
            bot.cash += q * price
            bot.shares -= q
            bot.trades.append({"date": datetime.now(), "action": "SAT", "price": price,
                               "qty": q, "total": q * price, "cash": bot.cash,
                               "shares": bot.shares, "value": bot.value(price),
                               "reason": "Manuel"})
            st.rerun()

    if st.button("🔄 Sıfırla", use_container_width=True):
        st.session_state.bot = Bot(100000)
        st.rerun()

c1, c2, c3 = st.columns(3)
c1.metric("Fiyat", f"{price:.2f} TL")
c2.metric("RSI", f"{last['RSI']:.1f}",
          "Aşırı Alım" if last["RSI"] > 70 else ("Aşırı Satım" if last["RSI"] < 30 else "Nötr"))
c3.metric("MACD", f"{last['MACD_hist']:.2f}",
          "Pozitif" if last["MACD_hist"] > 0 else "Negatif")

st.subheader("📈 Grafik")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
                              low=df["Low"], close=df["Close"], name="THYAO"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], name="SMA20", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="SMA50", line=dict(color="cyan")))
for t in bot.trades:
    try:
        td = pd.to_datetime(t["date"])
        color = "lime" if t["action"] == "AL" else "red"
        sym = "triangle-up" if t["action"] == "AL" else "triangle-down"
        fig.add_trace(go.Scatter(x=[td], y=[t["price"]], mode="markers",
                                  marker=dict(color=color, size=14, symbol=sym),
                                  showlegend=False, hovertext=f"{t['action']} @ {t['price']:.2f}"))
    except Exception:
        pass
fig.update_layout(height=450, xaxis_rangeslider_visible=False,
                  template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, use_container_width=True)

action, score, sig = bot.decide(last)
st.subheader("🤖 AI Kararı")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Karar", action, f"Skor: {score:+.2f}")
with c2:
    st.write("**Sinyaller**")
    for k, v in sig.items():
        st.write(f"• {k.upper()}: {'AL' if v > 0 else 'SAT' if v < 0 else 'TUT'}")
with c3:
    st.write("**Ağırlıklar**")
    for k, v in bot.w.items():
        st.write(f"• {k}: {v:.2f}")

if st.button("🚀 AI Kararını Uygula", type="primary", use_container_width=True):
    ok = bot.execute(action, price, date, reason=f"AI {action} (skor {score:+.2f})")
    if ok:
        st.success(f"AI {action} yaptı: {price:.2f} TL")
        st.rerun()
    else:
        st.info(f"AI '{action}' dedi ama koşul yok.")

st.subheader("📜 İşlem Geçmişi")
if bot.trades:
    tdf = pd.DataFrame(bot.trades)
    tdf["date"] = pd.to_datetime(tdf["date"]).dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(tdf[["date", "action", "price", "qty", "total", "cash", "shares", "value", "reason"]],
                 use_container_width=True, hide_index=True)
else:
    st.info("Henüz işlem yok.")

st.subheader("📊 AI vs Al-Tut")
if bot.trades:
    first = df["Close"].iloc[0]
    bh = (price / first - 1) * 100
    ai_ret = (bot.value(price) / bot.initial - 1) * 100
    c1, c2 = st.columns(2)
    c1.metric("AI Getiri", f"{ai_ret:+.2f}%")
    c2.metric("Al-Tut Getiri", f"{bh:+.2f}%", f"{ai_ret - bh:+.2f}%")
    if ai_ret > bh:
        st.success("🎉 AI, Al-Tut'u yendi!")
    else:
        st.warning("AI, Al-Tut'un gerisinde.")

st.divider()
st.caption("⚠️ Hayali simülasyon. Yatırım tavsiyesi değildir.")
