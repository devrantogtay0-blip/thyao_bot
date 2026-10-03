import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh
import json
import hashlib

st.set_page_config(page_title="THYAO AI Ogrenen Bot", page_icon="✈️", layout="wide")
st.title("✈️ THYAO Ogrenen AI Trader")
st.caption("Sadece kar/zarar degil, NEDENINI de ogrenir.")

TR = ZoneInfo("Europe/Istanbul")

def now_tr():
    return datetime.now(TR)

def market_open():
    n = now_tr()
    if n.weekday() >= 5:
        return False
    return dtime(9, 55) <= n.time() <= dtime(18, 10)

qp = st.query_params
auto_on = qp.get("auto", "0") == "1"
test_mode = qp.get("test", "0") == "1"
interval_min = int(qp.get("int", "2"))
min_confidence = float(qp.get("conf", "0.3"))

@st.cache_data(ttl=60)
def get_data(period="1y", interval="1d"):
    df = yf.Ticker("THYAO.IS").history(period=period, interval=interval)
    if df.empty:
        return None
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df

def add_indicators(df):
    df = df.copy()
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]
    d = c.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    ls = (-d.where(d < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + g / ls))
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["BB_mid"] = df["SMA20"]
    df["BB_std"] = c.rolling(20).std()
    df["BB_up"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_dn"] = df["BB_mid"] - 2 * df["BB_std"]
    lo14 = l.rolling(14).min()
    hi14 = h.rolling(14).max()
    df["K"] = 100 * (c - lo14) / (hi14 - lo14)
    df["D"] = df["K"].rolling(3).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["Vol_SMA"] = v.rolling(20).mean()
    df["Vol_ratio"] = v / df["Vol_SMA"]
    return df.dropna().reset_index(drop=True)


def get_state(row):
    """Piyasa durumunu kategorize et - 4 karakter kod."""
    rsi = row["RSI"]
    rsi_c = "L" if rsi < 35 else ("H" if rsi > 65 else "M")
    macd = row["MACD_hist"]
    macd_c = "P" if macd > 0 else "N"
    trend = "U" if row["SMA20"] > row["SMA50"] else "D"
    adx = row["ADX"] if "ADX" in row else 25
    adx_c = "S" if adx > 25 else "W"
    return f"{rsi_c}{macd_c}{trend}{adx_c}"


class LearningBot:
    def __init__(self, cash=100000.0):
        self.cash = cash
        self.shares = 0
        self.initial = cash
        self.trades = []
        self.w = {"rsi": 0.2, "macd": 0.2, "trend": 0.2, "bb": 0.2, "stoch": 0.2}
        self.experience = []          # Tum islem deneyimleri
        self.state_performance = {}   # Durum -> basari orani
        self.signal_accuracy = {      # Her sinyalin dogru cikma orani
            "rsi": {"correct": 0, "total": 0},
            "macd": {"correct": 0, "total": 0},
            "trend": {"correct": 0, "total": 0},
            "bb": {"correct": 0, "total": 0},
            "stoch": {"correct": 0, "total": 0},
        }
        self.lessons = []              # Ogrenilen dersler
        self.last_auto_time = None
        self.auto_count = 0
        self.entry_price = None
        self.entry_state = None
        self.entry_signals = None

    def value(self, price):
        return self.cash + self.shares * price

    def decide(self, row):
        """Karar ver + aciklama uret."""
        sig = {}
        rsi = row["RSI"]
        sig["rsi"] = 1 if rsi < 35 else (-1 if rsi > 65 else 0)
        mh = row["MACD_hist"]
        sig["macd"] = 1 if mh > 0 else (-1 if mh < 0 else 0)
        sig["trend"] = 1 if row["SMA20"] > row["SMA50"] else (-1 if row["SMA20"] < row["SMA50"] else 0)
        c = row["Close"]
        sig["bb"] = 1 if c < row["BB_dn"] else (-1 if c > row["BB_up"] else 0)
        k, d = row["K"], row["D"]
        sig["stoch"] = 1 if k < 20 and k > d else (-1 if k > 80 and k < d else 0)

        # Temel skor
        base_score = sum(self.w[k] * sig[k] for k in sig)

        # Durum benzerligi - gecmis deneyimlerden ogrenilmis bonus
        state = get_state(row)
        state_bonus = 0.0
        state_history = []
        if state in self.state_performance:
            perf = self.state_performance[state]
            if perf["total"] >= 3:
                win_rate = perf["wins"] / perf["total"]
                state_bonus = (win_rate - 0.5) * 0.6
                state_history.append(f"{perf['total']} gecmis islem, basari %{win_rate*100:.0f}")

        # Benzer deneyimler
        similar = [e for e in self.experience if e["state"] == state]
        similar_bonus = 0.0
        if len(similar) >= 3:
            avg_pnl = np.mean([e["pnl_pct"] for e in similar])
            similar_bonus = np.clip(avg_pnl / 100, -0.3, 0.3)

        total_score = base_score + state_bonus + similar_bonus

        # Guven skoru
        confidence = min(abs(total_score), 1.0)

        # Karar
        if total_score > min_confidence:
            action = "AL"
        elif total_score < -min_confidence:
            action = "SAT"
        else:
            action = "TUT"

        # Aciklama uret
        explanations = []
        if state_history:
            explanations.append(f"Durum {state}: {state_history[0]}")
        if len(similar) >= 3:
            avg = np.mean([e["pnl_pct"] for e in similar])
            explanations.append(f"Benzer {len(similar)} islem ort. {avg:+.1f}%")
        if sig["rsi"] != 0:
            explanations.append(f"RSI {rsi:.0f} -> {'AL' if sig['rsi']>0 else 'SAT'}")
        if sig["macd"] != 0:
            explanations.append(f"MACD {'pozitif' if sig['macd']>0 else 'negatif'}")
        if sig["trend"] != 0:
            explanations.append(f"Trend {'yukari' if sig['trend']>0 else 'asagi'}")

        return action, total_score, sig, confidence, explanations, state, similar

    def execute(self, action, price, date, row, reason="AI", signals=None):
        """Islem yap ve deneyim kaydet."""
        state = get_state(row)
        if action == "AL" and self.cash > price * 10:
            qty = int((self.cash * 0.25) / price)
            if qty < 1:
                return False
            cost = qty * price
            self.cash -= cost
            self.shares += qty
            self.entry_price = price
            self.entry_state = state
            self.entry_signals = signals
            self.trades.append({
                "date": date, "action": "AL", "price": price, "qty": qty,
                "total": cost, "cash": self.cash, "shares": self.shares,
                "value": self.value(price), "reason": reason, "state": state
            })
            return True

        if action == "SAT" and self.shares > 0:
            qty = self.shares
            rev = qty * price
            pnl_pct = (price / self.entry_price - 1) * 100 if self.entry_price else 0

            # DENEYIM KAYDET
            if self.entry_price and self.entry_signals:
                exp = {
                    "state": self.entry_state,
                    "signals": dict(self.entry_signals),
                    "buy_price": self.entry_price,
                    "sell_price": price,
                    "pnl_pct": pnl_pct,
                    "date": str(date),
                }
                self.experience.append(exp)

                # Durum performansini guncelle
                if self.entry_state not in self.state_performance:
                    self.state_performance[self.entry_state] = {"wins": 0, "total": 0}
                self.state_performance[self.entry_state]["total"] += 1
                if pnl_pct > 0:
                    self.state_performance[self.entry_state]["wins"] += 1

                # SINYAL SORUMLULUGU - hangi sinyal dogruydu?
                for sig_name, sig_val in self.entry_signals.items():
                    if sig_name not in self.signal_accuracy:
                        continue
                    if sig_val == 0:
                        continue
                    # Sinyal AL dediyse ve kar ettiysek -> dogru
                    # Sinyal SAT dediyse ve zarar ettiysek -> dogru (cikmak iyiydi)
                    is_correct = (sig_val > 0 and pnl_pct > 0) or (sig_val < 0 and pnl_pct < 0)
                    self.signal_accuracy[sig_name]["total"] += 1
                    if is_correct:
                        self.signal_accuracy[sig_name]["correct"] += 1
                        self.w[sig_name] = min(0.5, self.w[sig_name] + 0.015)
                    else:
                        self.w[sig_name] = max(0.02, self.w[sig_name] - 0.015)

                # DERS CIKAR
                if pnl_pct > 3:
                    self.lessons.append(f"✅ {self.entry_state} durumunda AL dogruydu ({pnl_pct:+.1f}%)")
                elif pnl_pct < -3:
                    self.lessons.append(f"❌ {self.entry_state} durumunda AL yanlisti ({pnl_pct:+.1f}%)")

                # Normalize
                t = sum(self.w.values())
                if t > 0:
                    for k in self.w:
                        self.w[k] /= t

            self.cash += rev
            self.shares = 0
            self.entry_price = None
            self.entry_state = None
            self.entry_signals = None
            self.trades.append({
                "date": date, "action": "SAT", "price": price, "qty": qty,
                "total": rev, "cash": self.cash, "shares": 0,
                "value": self.value(price), "reason": reason, "state": state
            })
            return True

        return False

    def get_learning_report(self):
        """Ogrenme raporu olustur."""
        report = {}
        # Durum performansi
        state_stats = []
        for state, perf in self.state_performance.items():
            wr = perf["wins"] / perf["total"] * 100 if perf["total"] else 0
            state_stats.append({
                "Durum": state,
                "Islem": perf["total"],
                "Basari_%": round(wr, 1),
            })
        report["states"] = sorted(state_stats, key=lambda x: x["Islem"], reverse=True)

        # Sinyal dogruluk
        sig_stats = []
        for name, acc in self.signal_accuracy.items():
            if acc["total"] == 0:
                continue
            correct_pct = acc["correct"] / acc["total"] * 100
            sig_stats.append({
                "Sinyal": name.upper(),
                "Toplam": acc["total"],
                "Dogru": acc["correct"],
                "Dogruluk_%": round(correct_pct, 1),
                "Agirlik": round(self.w.get(name, 0), 3),
            })
        report["signals"] = sig_stats

        # Ogrenilen dersler
        report["lessons"] = self.lessons[-10:]

        return report


if "bot" not in st.session_state:
    st.session_state.bot = LearningBot(100000)
if "log" not in st.session_state:
    st.session_state.log = []

bot = st.session_state.bot

if auto_on:
    st_autorefresh(interval=60 * 1000, key="auto_refresh")

df = get_data()
if df is None:
    st.error("Veri yok")
    st.stop()

df = add_indicators(df)
last = df.iloc[-1]
price = float(last["Close"])
date = last["Date"]

action, score, sig, confidence, explanations, state, similar = bot.decide(last)
auto_msg = None

if auto_on:
    n = now_tr()
    should_run = (bot.last_auto_time is None) or ((n - bot.last_auto_time).total_seconds() >= interval_min * 60)
    if should_run:
        if market_open() or test_mode:
            ok = bot.execute(action, price, date, last, reason=f"AI {action} (guven %{confidence*100:.0f})", signals=sig)
            bot.last_auto_time = n
            bot.auto_count += 1
            auto_msg = f"{action}: {price:.2f} TL (guven %{confidence*100:.0f})" if ok else f"Karar {action} - kosul yok"
            st.session_state.log.append(f"{n.strftime('%H:%M:%S')} - {auto_msg}")
        else:
            bot.last_auto_time = n
            st.session_state.log.append(f"{now_tr().strftime('%H:%M:%S')} - Borsa kapali")

# ---- SIDEBAR ----
with st.sidebar:
    st.header("Panel")
    st.metric("THYAO", f"{price:.2f} TL")
    val = bot.value(price)
    pnl = val - bot.initial
    st.metric("Portfoy", f"{val:,.0f} TL", f"{pnl:+,.0f} TL")
    c1, c2 = st.columns(2)
    c1.metric("Nakit", f"{bot.cash:,.0f}")
    c2.metric("Hisse", f"{bot.shares}")

    if bot.entry_price:
        ch = (price / bot.entry_price - 1) * 100
        st.metric("Pozisyon", f"{ch:+.2f}%", f"Giris: {bot.entry_price:.2f}")

    st.divider()
    st.subheader("Otomatik")
    auto_w = st.toggle("Kendi kendine", value=auto_on)
    if auto_w != auto_on:
        st.query_params["auto"] = "1" if auto_w else "0"
        st.rerun()
    test_w = st.toggle("Test Modu", value=test_mode)
    if test_w != test_mode:
        st.query_params["test"] = "1" if test_w else "0"
        st.rerun()

    st.divider()
    st.subheader("AI Ogrenmesi")
    st.write(f"Deneyim: **{len(bot.experience)}** islem")
    st.write(f"Durum: **{len(bot.state_performance)}** farkli")
    st.write(f"Ders: **{len(bot.lessons)}**")

    if st.button("Sifirla", use_container_width=True):
        st.session_state.bot = LearningBot(100000)
        st.session_state.log = []
        st.rerun()

# ---- UST ----
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fiyat", f"{price:.2f} TL")
c2.metric("RSI", f"{last['RSI']:.1f}")
c3.metric("MACD", f"{last['MACD_hist']:.2f}")
c4.metric("Durum", state)
c5.metric("Guven", f"%{confidence*100:.0f}")

# ---- SEKMELER ----
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 AI Karari", "🧠 Ogrenme Raporu", "📈 Grafik", "📜 Islemler", "📡 Log"
])

with tab1:
    st.subheader(f"Karar: {action}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Skor", f"{score:+.3f}")
        st.metric("Guven", f"%{confidence*100:.0f}")
    with c2:
        st.write("**Sinyaller**")
        for k, v in sig.items():
            st.write(f"• {k.upper()}: {'🟢 AL' if v>0 else '🔴 SAT' if v<0 else '⚪ TUT'}")
    with c3:
        st.write("**Agirliklar**")
        for k, v in bot.w.items():
            st.progress(min(v, 0.5)/0.5, text=f"{k}: {v:.3f}")

    st.divider()
    st.subheader("🔍 Neden Bu Karar?")
    if explanations:
        for e in explanations:
            st.write(f"• {e}")
    else:
        st.caption("Henuz yeterli deneyim yok. Bot ogrenmeye devam ediyor.")

    if similar and len(similar) >= 3:
        st.divider()
        st.subheader(f"📚 Benzer Gecmis Deneyimler ({len(similar)} adet)")
        sim_df = pd.DataFrame([{
            "Tarih": e["date"][:10],
            "Alis": f"{e['buy_price']:.2f}",
            "Satis": f"{e['sell_price']:.2f}",
            "Kar %": f"{e['pnl_pct']:+.2f}%"
        } for e in similar[-5:]])
        st.dataframe(sim_df, use_container_width=True, hide_index=True)

    if st.button("🚀 Simdi Uygula", type="primary", use_container_width=True):
        ok = bot.execute(action, price, date, last, reason=f"Manuel {action}", signals=sig)
        if ok:
            st.success(f"{action} yapildi")
            st.rerun()
        else:
            st.info("Kosul yok")

with tab2:
    st.subheader("🧠 Ogrenme Raporu")
    rep = bot.get_learning_report()

    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Deneyim", len(bot.experience))
    c2.metric("Ogrenilen Durum", len(bot.state_performance))
    c3.metric("Cikarilan Ders", len(bot.lessons))

    st.divider()
    st.subheader("📊 Sinyal Dogrulugu")
    if rep["signals"]:
        sdf = pd.DataFrame(rep["signals"])
        st.dataframe(sdf, use_container_width=True, hide_index=True)
        st.caption("Yuksek dogruluk = bu sinyal guvenilir. Bot agirligini otomatik artiriyor.")
    else:
        st.info("Henuz sinyal verisi yok. Bot islem yaptikca burasi dolacak.")

    st.divider()
    st.subheader("🎯 Durum Performansi")
    if rep["states"]:
        stdf = pd.DataFrame(rep["states"])
        st.dataframe(stdf, use_container_width=True, hide_index=True)
        st.caption("Hangi piyasa durumunda basarili oldu. Orn: LUPS = RSI dusuk, MACD pozitif, trend yukari, ADX guclu")
    else:
        st.info("Henuz durum verisi yok.")

    st.divider()
    st.subheader("💡 Ogrenilen Dersler")
    if rep["lessons"]:
        for lesson in reversed(rep["lessons"]):
            st.write(lesson)
    else:
        st.info("Henuz ders yok. Islem yaptikca bot ders cikaracak.")

with tab3:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Fiyat"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], name="SMA20", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="SMA50", line=dict(color="cyan")), row=1, col=1)
    for t in bot.trades:
        try:
            td = pd.to_datetime(t["date"])
            color = "lime" if t["action"] == "AL" else "red"
            sym = "triangle-up" if t["action"] == "AL" else "triangle-down"
            fig.add_trace(go.Scatter(x=[td], y=[t["price"]], mode="markers",
                                      marker=dict(color=color, size=12, symbol=sym), showlegend=False), row=1, col=1)
        except Exception:
            pass
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    if bot.trades:
        tdf = pd.DataFrame(bot.trades)
        tdf["date"] = pd.to_datetime(tdf["date"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(tdf[["date", "action", "price", "qty", "total", "value", "state", "reason"]],
                     use_container_width=True, hide_index=True)

        if bot.experience:
            st.divider()
            st.subheader("🧠 Deneyim Kayitlari")
            edf = pd.DataFrame([{
                "Durum": e["state"],
                "Alis": f"{e['buy_price']:.2f}",
                "Satis": f"{e['sell_price']:.2f}",
                "Kar %": f"{e['pnl_pct']:+.2f}%",
            } for e in bot.experience])
            st.dataframe(edf, use_container_width=True, hide_index=True)
    else:
        st.info("Henuz islem yok.")

with tab5:
    for line in reversed(st.session_state.log[-30:]):
        st.text(line)

st.divider()
st.caption(f"🕐 {now_tr().strftime('%Y-%m-%d %H:%M:%S')} - Borsa: {'ACIK' if market_open() else 'KAPALI'}")
