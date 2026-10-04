import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh
import requests

st.set_page_config(page_title="THYAO AI Pro v5", page_icon="✈️", layout="wide")
st.title("✈️ THYAO AI Pro Trader v5")
st.caption("Canli veri + alarm + coklu hisse + AI + Telegram + backtest")

TR = ZoneInfo("Europe/Istanbul")

def now_tr():
    return datetime.now(TR)

def market_open():
    n = now_tr()
    if n.weekday() >= 5:
        return False
    return dtime(9, 55) <= n.time() <= dtime(18, 10)

# ====== HISSELER ======
HISSELER = {
    "THYAO": "THYAO.IS",
    "GARAN": "GARAN.IS",
    "ASELS": "ASELS.IS",
    "AKBNK": "AKBNK.IS",
    "EREGL": "EREGL.IS",
    "TUPRS": "TUPRS.IS",
    "SISE": "SISE.IS",
    "KCHOL": "KCHOL.IS",
}

qp = st.query_params
auto_on = qp.get("auto", "0") == "1"
test_mode = qp.get("test", "0") == "1"
interval_min = int(qp.get("int", "2"))
min_confidence = float(qp.get("conf", "0.3"))
secili_hisse = qp.get("hisse", "THYAO")
refresh_sn = int(qp.get("rs", "30"))

# ====== BILGI BANKASI ======
BILGI = {
    "Temel": {
        "Hisse Senedi": "Bir sirketin ortakligini temsil eden pay.",
        "BIST": "Borsa Istanbul. Turkiye'nin resmi borsasi.",
        "Endeks": "Birden fazla hissenin ortalama performansi. BIST100 en onemli.",
        "Tavan/Taban": "BIST'te gunluk %10 limit.",
        "Volatilite": "Fiyat dalgalanma siddeti. Yuksek = yuksek risk.",
    },
    "Gostergeler": {
        "RSI": "0-100. 70 ustu asiri alim, 30 alti asiri satim.",
        "MACD": "EMA12 - EMA26. Signal EMA9. Histogram = MACD - Signal.",
        "SMA": "Basit hareketli ortalama. SMA20/50/200.",
        "EMA": "Ustel hareketli ortalama. SMA'dan hizli tepki.",
        "Bollinger": "SMA20 +/- 2 std. Ust=pahali, alt=ucuz.",
        "Stochastic": "%K ve %D. 80 ustu asiri alim.",
        "ATR": "Ortalama gercek aralik. Volatilite olcusu.",
        "ADX": "Trend gucu. 25 ustu = guclu trend.",
        "OBV": "Hacim bazli momentum.",
        "CCI": "+100 asiri alim, -100 asiri satim.",
    },
    "Formasyonlar": {
        "Bas-Omuz-Bas": "Dusus formasyonu.",
        "Ikili Dip": "W formu, alis sinyali.",
        "Ikili Tepe": "M formu, satis sinyali.",
        "Ucgen": "Simetrik/yukselen/alcalan. Kirilim yonu onemli.",
        "Bayrak": "Konsolidasyon, trend devam.",
    },
    "Temel Analiz": {
        "F/K": "Fiyat/Kazanc. Dusuk = ucuz.",
        "PD/DD": "Piyasa/Defter. 1 alti ucuz.",
        "ROE": "Ozkaynak karliligi. %15+ iyi.",
        "Temettu": "Kar payi. Yuksek = iyi pasif gelir.",
        "PEG": "F/K / Buyume. 1 alti ucuz.",
    },
    "Stratejiler": {
        "Trend": "Yukselen trendde al.",
        "Mean Reversion": "Asiri satimda al, asiri alimda sat.",
        "Momentum": "Guclu gideni al.",
        "Breakout": "Kirilim stratejisi.",
        "Swing": "Gunler-haftalar.",
    },
    "Risk": {
        "Stop-Loss": "Zarar kesme. %2-5.",
        "Take-Profit": "Kar alma. Stop'un 2-3 kati.",
        "Pozisyon": "Sermayenin %1-2'sini riske at.",
        "Sharpe": "(Getiri-Risksiz)/Std. 1+ iyi.",
        "Drawdown": "En yuksek zirveden dusus.",
    },
    "Psikoloji": {
        "FOMO": "Kacirma korkusu.",
        "Panik": "Korkuyla zararina satmak.",
        "Disiplin": "Planina sadik kal.",
        "Sabir": "Iyi firsat icin bekle.",
    },
    "BIST": {
        "Seans": "09:55-18:00. Tek seans.",
        "T+2": "Takas suresi 2 gun.",
        "Temettu": "Nisan-Mayis.",
        "Halka Arz": "IPO.",
    },
}

def bilgi_ara(konu):
    konu = konu.lower()
    sonuc = []
    for kat, mad in BILGI.items():
        for b, i in mad.items():
            if konu in b.lower() or konu in i.lower():
                sonuc.append({"Kategori": kat, "Konu": b, "Aciklama": i})
    return sonuc

# ====== 25 USTA ======
GURUS = [
    {"name": "Warren Buffett", "check": lambda r: r["Close"] > r["SMA200"] and r["Volatility"] < 0.3},
    {"name": "Benjamin Graham", "check": lambda r: r["Close"] < r["BB_dn"]},
    {"name": "Peter Lynch", "check": lambda r: 50 < r["RSI"] < 70 and r["MACD_hist"] > 0},
    {"name": "Charlie Munger", "check": lambda r: r["SMA20"] > r["SMA50"] and r["ADX"] > 20},
    {"name": "Philip Fisher", "check": lambda r: r["SMA50"] > r.get("SMA200", r["SMA50"]) * 0.95},
    {"name": "John Templeton", "check": lambda r: r["RSI"] < 35},
    {"name": "George Soros", "check": lambda r: r["MACD"] > r["MACD_sig"]},
    {"name": "Jesse Livermore", "check": lambda r: r["SMA20"] > r["SMA50"] and r["Close"] > r["SMA20"]},
    {"name": "Paul Tudor Jones", "check": lambda r: r["Close"] > r["SMA50"]},
    {"name": "Ray Dalio", "check": lambda r: r["Volatility"] < 0.35},
    {"name": "Stanley Druckenmiller", "check": lambda r: r["ADX"] > 25 and r["MACD_hist"] > 0},
    {"name": "William O'Neil", "check": lambda r: r["RSI"] > 60 and r["Vol_ratio"] > 1.2},
    {"name": "Mark Minervini", "check": lambda r: r.get("BB_width", 1) < 0.08 and r["ADX"] > 20},
    {"name": "Nicolas Darvas", "check": lambda r: r["Close"] > r["BB_up"] * 0.98},
    {"name": "Jim Simons", "check": lambda r: r["RSI"] < 30},
    {"name": "Michael Burry", "check": lambda r: r["RSI"] < 25 and r["WILLR"] < -85},
    {"name": "Carl Icahn", "check": lambda r: r["Close"] < r["BB_mid"] and r["ADX"] < 20},
    {"name": "Bill Ackman", "check": lambda r: r["SMA20"] > r["SMA50"] and r["Volatility"] < 0.4},
    {"name": "Seth Klarman", "check": lambda r: r["Close"] < r["BB_dn"] * 1.02},
    {"name": "Howard Marks", "check": lambda r: r["RSI"] < 40 and r["MACD_hist"] > -0.5},
    {"name": "David Tepper", "check": lambda r: r["RSI"] < 35 and r["BB_dn"] > r["Close"] * 0.95},
    {"name": "Ken Griffin", "check": lambda r: r["Vol_ratio"] > 1.0 and abs(r["MACD_hist"]) > 0.1},
    {"name": "Steven Cohen", "check": lambda r: r["RSI"] > 55 and r["Vol_ratio"] > 1.3},
    {"name": "Larry Williams", "check": lambda r: r["K"] < 20 and r["ATR"] > 0},
    {"name": "Ed Seykota", "check": lambda r: r["SMA20"] > r["SMA50"] and r["ADX"] > 22},
]

def guru_score(row, side="BUY"):
    c = []
    for g in GURUS:
        try:
            v = bool(g["check"](row))
            c.append(v if side == "BUY" else not v)
        except Exception:
            c.append(False)
    return sum(1 for x in c if x) / len(GURUS)

def guru_details(row):
    r = []
    for g in GURUS:
        try:
            onay = bool(g["check"](row))
        except Exception:
            onay = False
        r.append({"Usta": g["name"], "Onay": "AL" if onay else "BEKLE"})
    return r

# ====== HABER DUYGU ======
POZ = ["yuksel", "artis", "kar", "buyume", "guclu", "rekor", "anlasma", "yatirim", "pozitif", "kazanc", "ihracat", "siparis", "hedef", "basari"]
NEG = ["dusus", "azal", "zarar", "kriz", "zayif", "risk", "ceza", "sorusturma", "negatif", "kayip", "iflas", "iptal", "gerileme", "uyari"]

def duygu_puanla(text):
    if not text:
        return 0, "Notr"
    t = text.lower()
    p = sum(1 for w in POZ if w in t)
    n = sum(1 for w in NEG if w in t)
    if p > n:
        return 1, "Pozitif"
    if n > p:
        return -1, "Negatif"
    return 0, "Notr"

@st.cache_data(ttl=180)
def get_news(symbol):
    try:
        n = yf.Ticker(symbol).news or []
        out = []
        for item in n[:10]:
            title = item.get("title", "")
            pub = item.get("publisher", "")
            link = item.get("link", "")
            ts = item.get("providerPublishTime", 0)
            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
            _, duygu = duygu_puanla(title)
            out.append({"Baslik": title, "Kaynak": pub, "Tarih": dt, "Link": link, "Duygu": duygu})
        return out
    except Exception:
        return []

# ====== TELEGRAM ======
def telegram_gonder(token, chat_id, mesaj):
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": mesaj, "parse_mode": "HTML"}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# ====== AI DANISMAN ======
def ai_sor(api_key, base_url, model, soru, baglam=""):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        sys = "Sen THYAO ve borsa uzmanisin. Kisa, net Turkce cevap ver."
        msgs = [{"role": "system", "content": sys}]
        if baglam:
            msgs.append({"role": "system", "content": f"Baglam: {baglam}"})
        msgs.append({"role": "user", "content": soru})
        r = client.chat.completions.create(model=model, messages=msgs, temperature=0.7, max_tokens=800)
        return r.choices[0].message.content
    except Exception as e:
        return f"AI hatasi: {str(e)}"

# ====== VERI ======
@st.cache_data(ttl=30)
def get_data(symbol, period="1y", interval="1d"):
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    if df.empty:
        return None
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df

@st.cache_data(ttl=15)
def get_canli_fiyat(symbol):
    try:
        df = yf.Ticker(symbol).history(period="1d", interval="1m")
        if df.empty:
            return None
        return {
            "price": float(df["Close"].iloc[-1]),
            "open": float(df["Open"].iloc[0]),
            "high": float(df["High"].max()),
            "low": float(df["Low"].min()),
            "volume": int(df["Volume"].sum()),
            "change_pct": float((df["Close"].iloc[-1] / df["Open"].iloc[0] - 1) * 100),
        }
    except Exception:
        return None

def add_indicators(df):
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
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
    df["SMA200"] = c.rolling(200).mean()
    df["BB_mid"] = df["SMA20"]
    df["BB_std"] = c.rolling(20).std()
    df["BB_up"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_dn"] = df["BB_mid"] - 2 * df["BB_std"]
    df["BB_width"] = (df["BB_up"] - df["BB_dn"]) / df["BB_mid"]
    lo14 = l.rolling(14).min()
    hi14 = h.rolling(14).max()
    df["K"] = 100 * (c - lo14) / (hi14 - lo14)
    df["D"] = df["K"].rolling(3).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["WILLR"] = -100 * (hi14 - c) / (hi14 - lo14)
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr14 = tr.rolling(14).sum()
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(14).sum() / tr14
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(14).sum() / tr14
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["ADX"] = dx.rolling(14).mean()
    df["Vol_SMA"] = v.rolling(20).mean()
    df["Vol_ratio"] = v / df["Vol_SMA"]
    df["Returns"] = c.pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252)
    return df.dropna().reset_index(drop=True)

def get_state(row):
    r = "L" if row["RSI"] < 35 else ("H" if row["RSI"] > 65 else "M")
    m = "P" if row["MACD_hist"] > 0 else "N"
    t = "U" if row["SMA20"] > row["SMA50"] else "D"
    a = "S" if row.get("ADX", 25) > 25 else "W"
    return f"{r}{m}{t}{a}"

def ai_tahmin(df, gun=5):
    """Basit lineer regresyon ile fiyat tahmini."""
    if len(df) < 30:
        return None
    y = df["Close"].values[-60:]
    x = np.arange(len(y))
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    gelecek_x = np.arange(len(y), len(y) + gun)
    tahminler = p(gelecek_x)
    return tahminler.tolist(), float(z[0])

def backtest(df, cash=100000.0):
    """Basit backtest - SMA20/50 kesisimi."""
    c = cash
    s = 0
    trades = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row["Close"]
        # AL: SMA20 > SMA50 kesisimi
        if prev["SMA20"] <= prev["SMA50"] and row["SMA20"] > row["SMA50"] and c > price * 10:
            q = int((c * 0.95) / price)
            c -= q * price
            s += q
            trades.append({"date": row["Date"], "action": "AL", "price": price, "qty": q})
        # SAT
        elif prev["SMA20"] >= prev["SMA50"] and row["SMA20"] < row["SMA50"] and s > 0:
            c += s * price
            trades.append({"date": row["Date"], "action": "SAT", "price": price, "qty": s})
            s = 0
    final = c + s * df["Close"].iloc[-1]
    return {
        "baslangic": cash,
        "son": final,
        "getiri_pct": (final / cash - 1) * 100,
        "islem_sayisi": len(trades),
        "al_tut_pct": (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100,
    }

# ====== BOT ======
class LearningBot:
    def __init__(self, cash=100000.0):
        self.cash = cash
        self.shares = 0
        self.initial = cash
        self.trades = []
        self.w = {"rsi": 0.2, "macd": 0.2, "trend": 0.2, "bb": 0.2, "stoch": 0.2}
        self.experience = []
        self.state_performance = {}
        self.signal_accuracy = {k: {"correct": 0, "total": 0} for k in ["rsi", "macd", "trend", "bb", "stoch"]}
        self.lessons = []
        self.last_auto_time = None
        self.auto_count = 0
        self.entry_price = None
        self.entry_state = None
        self.entry_signals = None

    def value(self, price):
        return self.cash + self.shares * price

    def decide(self, row):
        sig = {}
        sig["rsi"] = 1 if row["RSI"] < 35 else (-1 if row["RSI"] > 65 else 0)
        sig["macd"] = 1 if row["MACD_hist"] > 0 else (-1 if row["MACD_hist"] < 0 else 0)
        sig["trend"] = 1 if row["SMA20"] > row["SMA50"] else (-1 if row["SMA20"] < row["SMA50"] else 0)
        sig["bb"] = 1 if row["Close"] < row["BB_dn"] else (-1 if row["Close"] > row["BB_up"] else 0)
        k, d = row["K"], row["D"]
        sig["stoch"] = 1 if k < 20 and k > d else (-1 if k > 80 and k < d else 0)
        base = sum(self.w[k] * sig[k] for k in sig)
        state = get_state(row)
        sb = 0.0
        if state in self.state_performance and self.state_performance[state]["total"] >= 3:
            perf = self.state_performance[state]
            wr = perf["wins"] / perf["total"]
            sb = (wr - 0.5) * 0.6
        sim = [e for e in self.experience if e["state"] == state]
        sib = 0.0
        if len(sim) >= 3:
            sib = np.clip(np.mean([e["pnl_pct"] for e in sim]) / 100, -0.3, 0.3)
        total = base + sb + sib
        gs = guru_score(row, "BUY" if total > 0 else "SELL")
        total = total * 0.7 + (gs - 0.5) * 0.6
        conf = min(abs(total), 1.0)
        if total > min_confidence:
            action = "AL"
        elif total < -min_confidence:
            action = "SAT"
        else:
            action = "TUT"
        exp = []
        if sim and len(sim) >= 3:
            exp.append(f"Benzer {len(sim)} islem ort. {np.mean([e['pnl_pct'] for e in sim]):+.1f}%")
        if sig["rsi"] != 0:
            exp.append(f"RSI {row['RSI']:.0f}")
        if sig["macd"] != 0:
            exp.append(f"MACD {'pozitif' if sig['macd']>0 else 'negatif'}")
        if sig["trend"] != 0:
            exp.append(f"Trend {'yukari' if sig['trend']>0 else 'asagi'}")
        gd = guru_details(row)
        ap = sum(1 for g in gd if g["Onay"] == "AL")
        exp.append(f"👑 {ap}/25 usta")
        return action, total, sig, conf, exp, state, sim, gs, gd

    def execute(self, action, price, date, row, reason="AI", signals=None):
        state = get_state(row)
        if action == "AL" and self.cash > price * 10:
            q = int((self.cash * 0.25) / price)
            if q < 1:
                return False
            cost = q * price
            self.cash -= cost
            self.shares += q
            self.entry_price = price
            self.entry_state = state
            self.entry_signals = signals
            self.trades.append({"date": date, "action": "AL", "price": price, "qty": q,
                                "total": cost, "cash": self.cash, "shares": self.shares,
                                "value": self.value(price), "reason": reason, "state": state})
            return True
        if action == "SAT" and self.shares > 0:
            q = self.shares
            rev = q * price
            pnl_pct = (price / self.entry_price - 1) * 100 if self.entry_price else 0
            if self.entry_price and self.entry_signals:
                self.experience.append({"state": self.entry_state, "signals": dict(self.entry_signals),
                                        "buy_price": self.entry_price, "sell_price": price,
                                        "pnl_pct": pnl_pct, "date": str(date)})
                if self.entry_state not in self.state_performance:
                    self.state_performance[self.entry_state] = {"wins": 0, "total": 0}
                self.state_performance[self.entry_state]["total"] += 1
                if pnl_pct > 0:
                    self.state_performance[self.entry_state]["wins"] += 1
                for sn, sv in self.entry_signals.items():
                    if sn not in self.signal_accuracy or sv == 0:
                        continue
                    ok = (sv > 0 and pnl_pct > 0) or (sv < 0 and pnl_pct < 0)
                    self.signal_accuracy[sn]["total"] += 1
                    if ok:
                        self.signal_accuracy[sn]["correct"] += 1
                        self.w[sn] = min(0.5, self.w[sn] + 0.015)
                    else:
                        self.w[sn] = max(0.02, self.w[sn] - 0.015)
                if pnl_pct > 3:
                    self.lessons.append(f"✅ {self.entry_state} AL dogru ({pnl_pct:+.1f}%)")
                elif pnl_pct < -3:
                    self.lessons.append(f"❌ {self.entry_state} AL yanlis ({pnl_pct:+.1f}%)")
                t = sum(self.w.values())
                if t > 0:
                    for k in self.w:
                        self.w[k] /= t
            self.cash += rev
            self.shares = 0
            self.entry_price = None
            self.entry_state = None
            self.entry_signals = None
            self.trades.append({"date": date, "action": "SAT", "price": price, "qty": q,
                                "total": rev, "cash": self.cash, "shares": 0,
                                "value": self.value(price), "reason": reason, "state": state})
            return True
        return False

# ====== SESSION ======
if "bot" not in st.session_state:
    st.session_state.bot = LearningBot(100000)
if "log" not in st.session_state:
    st.session_state.log = []
if "chat" not in st.session_state:
    st.session_state.chat = []
if "alarmlar" not in st.session_state:
    st.session_state.alarmlar = []
if "alarm_gecmis" not in st.session_state:
    st.session_state.alarm_gecmis = []
bot = st.session_state.bot

if auto_on:
    st_autorefresh(interval=refresh_sn * 1000, key="auto_refresh")

# ====== VERI CEK ======
symbol = HISSELER[secili_hisse]
df = get_data(symbol)
if df is None:
    st.error(f"{secili_hisse} verisi yok")
    st.stop()

df = add_indicators(df)
last = df.iloc[-1]
canli = get_canli_fiyat(symbol)
price = canli["price"] if canli else float(last["Close"])
date = last["Date"]

action, score, sig, conf, explanations, state, similar, g_score, g_details = bot.decide(last)

# ====== ALARM KONTROL ======
for a in st.session_state.alarmlar:
    if a["hisse"] == secili_hisse and not a.get("tetiklendi", False):
        if a["yon"] == "ust" and price >= a["fiyat"]:
            msg = f"🚨 {secili_hisse} {a['fiyat']:.2f} TL'yi gecti! Su an: {price:.2f}"
            st.session_state.alarm_gecmis.append(f"{now_tr().strftime('%H:%M:%S')} - {msg}")
            a["tetiklendi"] = True
            if a.get("telegram"):
                telegram_gonder(qp.get("tgt", ""), qp.get("tgc", ""), msg)
        elif a["yon"] == "alt" and price <= a["fiyat"]:
            msg = f"🚨 {secili_hisse} {a['fiyat']:.2f} TL'nin altina indi! Su an: {price:.2f}"
            st.session_state.alarm_gecmis.append(f"{now_tr().strftime('%H:%M:%S')} - {msg}")
            a["tetiklendi"] = True
            if a.get("telegram"):
                telegram_gonder(qp.get("tgt", ""), qp.get("tgc", ""), msg)

# ====== OTOMATIK ======
auto_msg = None
if auto_on:
    n = now_tr()
    sr = (bot.last_auto_time is None) or ((n - bot.last_auto_time).total_seconds() >= interval_min * 60)
    if sr:
        if market_open() or test_mode:
            ok = bot.execute(action, price, date, last, reason=f"AI {action}", signals=sig)
            bot.last_auto_time = n
            bot.auto_count += 1
            auto_msg = f"{action}: {price:.2f} TL" if ok else f"Karar {action}"
            st.session_state.log.append(f"{n.strftime('%H:%M:%S')} - {auto_msg}")
            if ok and qp.get("tgt") and qp.get("tgc"):
                telegram_gonder(qp.get("tgt", ""), qp.get("tgc", ""),
                                f"🤖 {secili_hisse} {action}\nFiyat: {price:.2f}\nGuven: %{conf*100:.0f}\nUsta: %{g_score*100:.0f}")

# ====== SIDEBAR ======
with st.sidebar:
    st.header("⚙️ Panel")
    st.subheader("📊 Hisse Sec")
    hisse_sec = st.selectbox("Hisse", list(HISSELER.keys()), index=list(HISSELER.keys()).index(secili_hisse))
    if hisse_sec != secili_hisse:
        st.query_params["hisse"] = hisse_sec
        st.rerun()

    st.divider()
    st.metric(f"{secili_hisse} Canli", f"{price:.2f} TL",
              f"{canli['change_pct']:+.2f}%" if canli else "")
    val = bot.value(price)
    st.metric("Portfoy", f"{val:,.0f} TL", f"{val-bot.initial:+,.0f}")
    c1, c2 = st.columns(2)
    c1.metric("Nakit", f"{bot.cash:,.0f}")
    c2.metric("Hisse", f"{bot.shares}")

    st.divider()
    st.subheader("🔄 Canli Veri")
    rs = st.slider("Yenileme (sn)", 10, 300, refresh_sn)
    if rs != refresh_sn:
        st.query_params["rs"] = str(rs)
        st.rerun()

    st.divider()
    st.subheader("🔔 Fiyat Alarmi")
    a_f = st.number_input("Fiyat", value=float(round(price, 2)), step=0.5)
    a_y = st.selectbox("Yon", ["ust", "alt"])
    a_tg = st.checkbox("Telegram'a gonder", value=False)
    if st.button("➕ Alarm Ekle", use_container_width=True):
        st.session_state.alarmlar.append({"hisse": secili_hisse, "fiyat": a_f, "yon": a_y,
                                            "tetiklendi": False, "telegram": a_tg})
        st.success(f"Alarm eklendi: {secili_hisse} {a_y} {a_f}")
    if st.session_state.alarmlar:
        st.caption(f"Aktif alarm: {len([a for a in st.session_state.alarmlar if not a.get('tetiklendi')])}")
        if st.button("🗑️ Temizle", use_container_width=True):
            st.session_state.alarmlar = []
            st.rerun()

    st.divider()
    st.subheader("🤖 Telegram")
    tg_t = st.text_input("Bot Token", type="password", value=qp.get("tgt", ""))
    tg_c = st.text_input("Chat ID", value=qp.get("tgc", ""))
    if tg_t != qp.get("tgt", "") or tg_c != qp.get("tgc", ""):
        st.query_params["tgt"] = tg_t
        st.query_params["tgc"] = tg_c
    if tg_t and tg_c:
        if st.button("📤 Test"):
            ok = telegram_gonder(tg_t, tg_c, "✅ Test mesaji")
            st.success("Gonderildi") if ok else st.error("Basarisiz")

    st.divider()
    st.subheader("🤖 AI Danisman")
    ai_key = st.text_input("API Key", type="password", value=qp.get("aik", ""))
    ai_url = st.text_input("Base URL", value=qp.get("aiu", "https://api.openai.com/v1"))
    ai_model = st.text_input("Model", value=qp.get("aim", "gpt-4o-mini"))
    if ai_key and ai_key != qp.get("aik", ""):
        st.query_params["aik"] = ai_key
        st.query_params["aiu"] = ai_url
        st.query_params["aim"] = ai_model
        st.rerun()

    st.divider()
    st.subheader("🤖 Otomatik")
    aw = st.toggle("Kendi kendine", value=auto_on)
    if aw != auto_on:
        st.query_params["auto"] = "1" if aw else "0"
        st.rerun()
    tw = st.toggle("Test Modu", value=test_mode)
    if tw != test_mode:
        st.query_params["test"] = "1" if tw else "0"
        st.rerun()

    if st.button("🔄 Sifirla", use_container_width=True):
        st.session_state.bot = LearningBot(100000)
        st.session_state.log = []
        st.rerun()

# ====== UST ======
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{secili_hisse}", f"{price:.2f} TL",
          f"{canli['change_pct']:+.2f}%" if canli else "")
c2.metric("RSI", f"{last['RSI']:.1f}")
c3.metric("MACD", f"{last['MACD_hist']:.2f}")
c4.metric("Durum", state)
c5.metric("Usta", f"%{g_score*100:.0f}")

if st.session_state.alarm_gecmis:
    for msg in st.session_state.alarm_gecmis[-3:]:
        st.warning(msg)
if auto_msg:
    st.info(auto_msg)

# ====== SEKMELER ======
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📡 Canli", "🤖 AI", "🧠 Ogrenme", "👑 Ustalar", "📰 Haberler",
    "🎯 Backtest", "🔮 Tahmin", "📚 Bilgi", "📜 Islemler"
])

with tab1:
    st.subheader(f"📡 {secili_hisse} Canli Veri")
    c1, c2, c3, c4 = st.columns(4)
    if canli:
        c1.metric("Fiyat", f"{canli['price']:.2f}")
        c2.metric("Yuksek", f"{canli['high']:.2f}")
        c3.metric("Dusuk", f"{canli['low']:.2f}")
        c4.metric("Hacim", f"{canli['volume']:,}")
    st.caption(f"Son guncelleme: {now_tr().strftime('%H:%M:%S')} - Her {refresh_sn} sn yenilenir")

    st.divider()
    st.subheader(f"📊 {secili_hisse} Grafik")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
                                  low=df["Low"], close=df["Close"], name="Fiyat"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], name="SMA20", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="SMA50", line=dict(color="cyan")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_up"], name="BB+", line=dict(color="gray", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_dn"], name="BB-", line=dict(color="gray", dash="dot")), row=1, col=1)
    for t in bot.trades:
        try:
            td = pd.to_datetime(t["date"])
            col = "lime" if t["action"] == "AL" else "red"
            sm = "triangle-up" if t["action"] == "AL" else "triangle-down"
            fig.add_trace(go.Scatter(x=[td], y=[t["price"]], mode="markers",
                                      marker=dict(color=col, size=12, symbol=sm), showlegend=False), row=1, col=1)
        except Exception:
            pass
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.alarmlar:
        st.divider()
        st.subheader("🔔 Aktif Alarmlar")
        for i, a in enumerate(st.session_state.alarmlar):
            durum = "✅ Tetiklendi" if a.get("tetiklendi") else "⏳ Bekliyor"
            st.write(f"{i+1}. {a['hisse']} {a['yon']} {a['fiyat']:.2f} - {durum}")

with tab2:
    st.subheader(f"Karar: {action}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Skor", f"{score:+.3f}")
    c1.metric("Guven", f"%{conf*100:.0f}")
    with c2:
        st.write("**Sinyaller**")
        for k, v in sig.items():
            st.write(f"• {k.upper()}: {'🟢 AL' if v>0 else '🔴 SAT' if v<0 else '⚪'}")
    with c3:
        st.write("**Agirliklar**")
        for k, v in bot.w.items():
            st.progress(min(v, 0.5)/0.5, text=f"{k}: {v:.3f}")
    st.divider()
    st.subheader("🔍 Neden?")
    for e in explanations:
        st.write(f"• {e}")
    st.divider()
    st.subheader("🤖 AI Yorumu")
    if not ai_key:
        st.info("Sol panelden API key gir")
    else:
        if st.button("AI'a Sor", use_container_width=True):
            with st.spinner("..."):
                bag = f"{secili_hisse}: {price:.2f}, RSI {last['RSI']:.0f}, MACD {last['MACD_hist']:.3f}, karar {action}"
                st.success(ai_sor(ai_key, ai_url, ai_model, f"Botun '{action}' karari mantikli mi? 2-3 cumle.", bag))
    if st.button("🚀 Simdi Uygula", type="primary", use_container_width=True):
        ok = bot.execute(action, price, date, last, reason=f"Manuel", signals=sig)
        if ok:
            st.success(f"{action} yapildi")
            if tg_t and tg_c:
                telegram_gonder(tg_t, tg_c, f"🤖 {secili_hisse} {action} @ {price:.2f}")
            st.rerun()

with tab3:
    st.subheader("🧠 Ogrenme Raporu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Deneyim", len(bot.experience))
    c2.metric("Durum", len(bot.state_performance))
    c3.metric("Ders", len(bot.lessons))
    st.divider()
    st.subheader("📊 Sinyal Dogrulugu")
    sdata = []
    for n, a in bot.signal_accuracy.items():
        if a["total"] > 0:
            sdata.append({"Sinyal": n.upper(), "Toplam": a["total"],
                          "Dogru": a["correct"],
                          "Dogruluk_%": round(a["correct"]/a["total"]*100, 1),
                          "Agirlik": round(bot.w.get(n, 0), 3)})
    if sdata:
        st.dataframe(pd.DataFrame(sdata), use_container_width=True, hide_index=True)
    else:
        st.info("Veri yok")
    st.divider()
    st.subheader("💡 Dersler")
    for l in reversed(bot.lessons[-10:]):
        st.write(l)

with tab4:
    st.subheader(f"👑 Usta Skoru: %{g_score*100:.0f}")
    st.dataframe(pd.DataFrame(g_details), use_container_width=True, hide_index=True)

with tab5:
    st.subheader(f"📰 {secili_hisse} Haberler")
    if st.button("🔄 Yenile", use_container_width=True):
        st.cache_data.clear()
    news = get_news(symbol)
    if not news:
        st.info("Haber yok")
    else:
        for n in news:
            with st.expander(f"{n['Duygu']} | {n['Baslik'][:80]}... ({n['Kaynak']})"):
                st.write(f"**{n['Baslik']}**")
                st.caption(f"{n['Tarih']} - {n['Kaynak']}")
                st.write(f"Duygu: **{n['Duygu']}**")
                if n["Link"]:
                    st.markdown(f"[Haberin tamami]({n['Link']})")

with tab6:
    st.subheader("🎯 Backtest (SMA20/50 Kesisimi)")
    st.caption("Gecmis veriyle strateji testi")
    if st.button("▶️ Calistir", use_container_width=True):
        with st.spinner("Test ediliyor..."):
            bt = backtest(df)
            c1, c2, c3 = st.columns(3)
            c1.metric("Baslangic", f"{bt['baslangic']:,.0f} TL")
            c2.metric("Son", f"{bt['son']:,.0f} TL", f"{bt['getiri_pct']:+.2f}%")
            c3.metric("Al-Tut", f"{bt['al_tut_pct']:+.2f}%")
            st.write(f"Islem sayisi: **{bt['islem_sayisi']}**")
            if bt["getiri_pct"] > bt["al_tut_pct"]:
                st.success("🎉 Strateji Al-Tut'u yendi!")
            else:
                st.warning("Al-Tut daha iyi")

with tab7:
    st.subheader("🔮 Fiyat Tahmini (Lineer Regresyon)")
    gun = st.slider("Kac gun ilerisi?", 1, 15, 5)
    tahmin = ai_tahmin(df, gun)
    if tahmin:
        t, egim = tahmin
        yon = "📈 YUKSELIS" if egim > 0 else "📉 DUSUS"
        st.metric("Yon", yon, f"Egim: {egim:+.3f}/gun")
        hist = df["Date"].tail(30).tolist()
        hist_f = df["Close"].tail(30).tolist()
        future = [(df["Date"].iloc[-1] + timedelta(days=i+1)).date() for i in range(gun)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist, y=hist_f, name="Gecmis", line=dict(color="cyan")))
        fig.add_trace(go.Scatter(x=[hist[-1]] + future, y=[hist_f[-1]] + t,
                                  name="Tahmin", line=dict(color="orange", dash="dash")))
        fig.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ Tahmin sadece matematiksel, gercek degil")
    else:
        st.info("Yeterli veri yok")

with tab8:
    st.subheader("📚 Borsa Bilgi Bankasi")
    arama = st.text_input("Ara", placeholder="RSI, temettu, stop-loss...")
    if arama:
        for s in bilgi_ara(arama):
            with st.expander(f"**{s['Konu']}** ({s['Kategori']})"):
                st.write(s["Aciklama"])
    else:
        for kat, mad in BILGI.items():
            with st.expander(f"📂 {kat} ({len(mad)})"):
                for b, i in mad.items():
                    st.markdown(f"**{b}**: {i}")

with tab9:
    if bot.trades:
        tdf = pd.DataFrame(bot.trades)
        tdf["date"] = pd.to_datetime(tdf["date"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(tdf[["date", "action", "price", "qty", "value", "reason"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("Islem yok")
    st.divider()
    st.subheader("📡 Otomatik Log")
    for line in reversed(st.session_state.log[-20:]):
        st.text(line)

st.divider()
st.caption(f"🕐 {now_tr().strftime('%Y-%m-%d %H:%M:%S')} - Borsa: {'ACIK' if market_open() else 'KAPALI'}")
