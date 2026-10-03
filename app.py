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

st.set_page_config(page_title="THYAO AI Pro Trader v4", page_icon="✈️", layout="wide")
st.title("✈️ THYAO AI Pro Trader v4")
st.caption("25 usta + Kendi kendine ogrenme + AI Danisman + Borsa Bilgi Bankasi")

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

# ====== BORSA BILGI BANKASI ======
BILGI = {
    "Temel Kavramlar": {
        "Hisse Senedi": "Bir sirketin ortakligini temsil eden pay. Alinca sirkete ortak olursun, kar payi (temettu) ve deger artisindan faydalanirsin.",
        "BIST": "Borsa Istanbul. Turkiye'nin resmi borsasi. 1986'da kuruldu.",
        "Endeks": "Birden fazla hissenin ortalama performansini gosteren sayi. BIST100 en onemli endekstir.",
        "Lot": "Hisse alim-satim birimi. 1 lot = 1 hisse (bazi borsalarda farkli).",
        "Tavan/Taban": "Bir hissenin gunluk maksimum/minimum fiyat hareketi. BIST'te %10.",
        "Likidite": "Bir varligin ne kadar hizli nakde cevrilebilecegi. Yuksek likidite = hizli alim-satim.",
        "Volatilite": "Fiyat dalgalanma siddeti. Yuksek volatilite = yuksek risk + yuksek firsat.",
        "Spread": "Alis ve satis fiyati arasindaki fark. Dusuk spread = likit piyasa.",
    },
    "Teknik Gostergeler": {
        "RSI": "Relative Strength Index. 0-100 arasi. 70 ustu asiri alim, 30 alti asiri satim. 14 gunluk standart.",
        "MACD": "Moving Average Convergence Divergence. EMA12 - EMA26. Signal cizgisi EMA9. Histogram = MACD - Signal.",
        "SMA": "Simple Moving Average. Basit hareketli ortalama. SMA20 kisa, SMA50 orta, SMA200 uzun vade.",
        "EMA": "Exponential Moving Average. Yeni fiyatlara daha fazla agirlik verir. SMA'dan hizli tepki verir.",
        "Bollinger Bantlari": "SMA20 +/- 2 standart sapma. Fiyat ust bantta = pahali, alt bantta = ucuz.",
        "Stochastic": "%K ve %D. 80 ustu asiri alim, 20 alti asiri satim.",
        "ATR": "Average True Range. Volatilite olcusu. Yuksek ATR = yuksek volatilite.",
        "ADX": "Average Directional Index. Trend gucu. 25 ustu = guclu trend.",
        "OBV": "On-Balance Volume. Hacim bazli momentum. Fiyat ile uyumlu ise trend saglam.",
        "CCI": "Commodity Channel Index. +100 ustu asiri alim, -100 alti asiri satim.",
        "Williams %R": "-100 ile 0 arasi. -20 ustu asiri alim, -80 alti asiri satim.",
        "Ichimoku": "Japon bulut sistemi. Tenkan, Kijun, Senkou A/B. Destek-direnc ve trend analizi.",
        "Fibonacci": "Altin oran seviyeleri: 23.6%, 38.2%, 50%, 61.8%, 78.6%. Destek-direnc noktalari.",
    },
    "Formasyonlar": {
        "Bas-Omuz-Bas": "Dusus formasyonu. Sol omuz - bas - sag omuz. Boyun cizgisi kirilinca satis.",
        "Ters Bas-Omuz-Bas": "Yukselis formasyonu. Yukari kirilimda alis sinyali.",
        "Ucgen": "Simetrik, yukselen, alcalan. Kirilim yonu onemli.",
        "Bayrak": "Kisa konsolidasyon. Trend yonunde devam sinyali.",
        "Ikili Dip": "W formasyonu. Ikinci dipte alis sinyali.",
        "Ikili Tepe": "M formasyonu. Ikinci tepede satis sinyali.",
        "Kanal": "Paralel destek-direnc cizgileri. Yatay, yukselen, alcalan.",
        "Kama": "Daralan ucgen. Genellikle trend donus sinyali.",
    },
    "Temel Analiz": {
        "F/K": "Fiyat/Kazanc orani. Dusuk = ucuz. Sektor ortalamasiyla karsilastir.",
        "PD/DD": "Piyasa Degeri/Defter Degeri. 1 altinda = defter degerinin altinda.",
        "FD/FAVOK": "Firma Degeri / FAVOK. Sirket degerleme olcusu.",
        "ROE": "Ozkaynak Karliligi. %15+ iyi, %20+ mukemmel.",
        "ROA": "Aktif Karliligi. Sirketin varliklarini ne kadar verimli kullandigi.",
        "Borclanma": "Borcluluk orani. Yuksek = riskli. Sektor ortalamasiyla karsilastir.",
        "Temettu Verimi": "Yillik temettu / Hisse fiyati. Yuksek = iyi pasif gelir.",
        "PEG": "F/K / Buyume orani. 1 altinda = ucuz buyume hissesi.",
        "Net Kar Marji": "Net kar / Ciro. Yuksek = verimli sirket.",
        "Serbest Nakit Akisi": "Isletme nakit akisi - Yatirim harcamalari. Pozitif olmali.",
    },
    "Stratejiler": {
        "Trend Takibi": "Yukselen trendde al, dusen trendde sat. SMA ve ADX kullan.",
        "Mean Reversion": "Ortalamaya donus. Asiri satimda al, asiri alimda sat. RSI + BB.",
        "Momentum": "Guclu gideni al. RSI 50 ustu + hacim + MACD.",
        "Breakout": "Kirilim stratejisi. Direnc kirilinca al, destek kirilinca sat.",
        "Scalping": "Cok kisa vadeli. Dakikalar- saatler. Yuksek likidite gerekir.",
        "Swing Trading": "Gunler-haftalar. Orta vadeli. Trend + destek-direnc.",
        "Pozisyon": "Aylar-yillar. Uzun vade. Temel analiz + sabir.",
        "Dollar-Cost Averaging": "Duzenli alim. Volatiliteye karsi ortalama maliyet.",
        "Pairs Trading": "Iki korelasyonlu hisse arasinda arbitraj.",
        "Grid Trading": "Belirli araliklarda al-sat emirleri. Yatay piyasada.",
    },
    "Risk Yonetimi": {
        "Stop-Loss": "Zarar kesme seviyesi. Genelde %2-5. Sermayeyi korur.",
        "Take-Profit": "Kar alma seviyesi. Genelde stop-loss'un 2-3 kati.",
        "Pozisyon Buyuklugu": "Tek islemde sermayenin %1-2'sini riske at.",
        "Risk/Getiri": "Minimum 1:2 orani. 1 TL riske 2 TL kazanc.",
        "Cesitlendirme": "Farkli sektorler, varliklar. Korelasyon dusuk olmali.",
        "Kelly Kriteri": "Optimal pozisyon buyuklugu formulu. Agresif.",
        "Trailing Stop": "Fiyat lehine hareket ettikce stop'u kaydir.",
        "Maksimum Drawdown": "En yuksek zirveden en dusuk noktaya kayip.",
        "Sharpe Orani": "(Getiri - Risksiz) / Std. 1+ iyi, 2+ mukemmel.",
        "Sortino Orani": "Sadece asagi yonlu volatiliteyi dikkate alan Sharpe.",
    },
    "Psikoloji": {
        "FOMO": "Fear of Missing Out. Kacirma korkusu. Yanlis kararlar verdirir.",
        "Panik Satis": "Korkuyla zararina satmak. En buyuk hata.",
        "Asiri Guven": "Kazandiktan sonra daha fazla risk almak. Tehlikeli.",
        "Kayip Avutma": "Zarari kabul edememek. Kucuk zarar buyuk olur.",
        "Intikam Trading": "Zarar sonrasi ofkeyle islem. En tehlikeli mod.",
        "Disiplin": "Planina sadik kal. Duygularla hareket etme.",
        "Sabir": "Iyi firsat icin bekle. Her gun islem yapmak zorunlu degil.",
        "Kabul": "Kayip normaldir. Kazanma orani %60+ iyi.",
    },
    "BIST Ozel": {
        "Seans Saatleri": "09:55 acilis, 18:00 kapanis. Ara: 13:00-14:00 yok (tek seans).",
        "Tavan-Taban": "%10 gunluk limit. Tavan = +%10, Taban = -%10.",
        "Lot": "1 lot = 1 hisse. Minimum alim 1 lot.",
        "Komisyon": "Aracilara gore %0.1-0.3 arasi degisir.",
        "T+2": "Takas suresi. Sattigin gun para 2 gun sonra hesaba gecer.",
        "Temettu": "Kar payi dagitimi. Genelde Nisan-Mayis.",
        "Bedelsiz": "Sermaye artirimi. Hisse sayisi artar, fiyat duser.",
        "Bedelli": "Nakit sermaye artirimi. Mevcut ortaklara rucan hakki.",
        "Halka Arz": "IPO. Sirketin borsaya ilk kez acilmasi.",
        "Brut Takas": "Spekulatif hisselere uygulanan kisitlama.",
    },
}

def bilgi_ara(konu):
    """Bilgi bankasinda anahtar kelime ara."""
    konu = konu.lower()
    sonuclar = []
    for kategori, maddeler in BILGI.items():
        for baslik, icerik in maddeler.items():
            if konu in baslik.lower() or konu in icerik.lower():
                sonuclar.append({"Kategori": kategori, "Konu": baslik, "Aciklama": icerik})
    return sonuclar

# ====== 25 USTA PRENSIBI ======
GURUS = [
    {"name": "Warren Buffett", "rule": "Guvenlik marji + istikrar",
     "check": lambda r: r["Close"] > r["SMA200"] and r["Volatility"] < 0.3},
    {"name": "Benjamin Graham", "rule": "Derin deger (BB alti)",
     "check": lambda r: r["Close"] < r["BB_dn"]},
    {"name": "Peter Lynch", "rule": "Buyume + momentum",
     "check": lambda r: 50 < r["RSI"] < 70 and r["MACD_hist"] > 0},
    {"name": "Charlie Munger", "rule": "Kaliteli is, guclu trend",
     "check": lambda r: r["SMA20"] > r["SMA50"] and r["ADX"] > 20},
    {"name": "Philip Fisher", "rule": "Uzun vadeli kalite",
     "check": lambda r: r["SMA50"] > r.get("SMA200", r["SMA50"]) * 0.95},
    {"name": "John Templeton", "rule": "Kontra - en kotu al",
     "check": lambda r: r["RSI"] < 35},
    {"name": "George Soros", "rule": "Refleksivite - trend takip",
     "check": lambda r: r["MACD"] > r["MACD_sig"]},
    {"name": "Jesse Livermore", "rule": "Trende katil",
     "check": lambda r: r["SMA20"] > r["SMA50"] and r["Close"] > r["SMA20"]},
    {"name": "Paul Tudor Jones", "rule": "200 gunluk ustu",
     "check": lambda r: r["Close"] > r["SMA50"]},
    {"name": "Ray Dalio", "rule": "Dongu - dusuk volatilite",
     "check": lambda r: r["Volatility"] < 0.35},
    {"name": "Stanley Druckenmiller", "rule": "Konsantre + makro",
     "check": lambda r: r["ADX"] > 25 and r["MACD_hist"] > 0},
    {"name": "William O'Neil", "rule": "CANSLIM - momentum",
     "check": lambda r: r["RSI"] > 60 and r["Vol_ratio"] > 1.2},
    {"name": "Mark Minervini", "rule": "VCP - volatilite sikismasi",
     "check": lambda r: r.get("BB_width", 1) < 0.08 and r["ADX"] > 20},
    {"name": "Nicolas Darvas", "rule": "Kutu kirilimi",
     "check": lambda r: r["Close"] > r["BB_up"] * 0.98},
    {"name": "Jim Simons", "rule": "Kantitatif - mean reversion",
     "check": lambda r: r["RSI"] < 30},
    {"name": "Michael Burry", "rule": "Derin kontra",
     "check": lambda r: r["RSI"] < 25 and r["WILLR"] < -85},
    {"name": "Carl Icahn", "rule": "Ucuz + aktivist",
     "check": lambda r: r["Close"] < r["BB_mid"] and r["ADX"] < 20},
    {"name": "Bill Ackman", "rule": "Konsantre + kalite",
     "check": lambda r: r["SMA20"] > r["SMA50"] and r["Volatility"] < 0.4},
    {"name": "Seth Klarman", "rule": "Guvenlik marji",
     "check": lambda r: r["Close"] < r["BB_dn"] * 1.02},
    {"name": "Howard Marks", "rule": "Ikinci seviye - dongu",
     "check": lambda r: r["RSI"] < 40 and r["MACD_hist"] > -0.5},
    {"name": "David Tepper", "rule": "Distressed + kontra",
     "check": lambda r: r["RSI"] < 35 and r["BB_dn"] > r["Close"] * 0.95},
    {"name": "Ken Griffin", "rule": "Kant + likidite",
     "check": lambda r: r["Vol_ratio"] > 1.0 and abs(r["MACD_hist"]) > 0.1},
    {"name": "Steven Cohen", "rule": "Momentum + bilgi",
     "check": lambda r: r["RSI"] > 55 and r["Vol_ratio"] > 1.3},
    {"name": "Larry Williams", "rule": "Volatilite + sezon",
     "check": lambda r: r["K"] < 20 and r["ATR"] > 0},
    {"name": "Ed Seykota", "rule": "Trend + risk kontrol",
     "check": lambda r: r["SMA20"] > r["SMA50"] and r["ADX"] > 22},
]

def guru_score(row, side="BUY"):
    checks = []
    for g in GURUS:
        try:
            v = bool(g["check"](row))
            checks.append(v if side == "BUY" else not v)
        except Exception:
            checks.append(False)
    return sum(1 for c in checks if c) / len(GURUS)

def guru_details(row):
    result = []
    for g in GURUS:
        try:
            onay = bool(g["check"](row))
        except Exception:
            onay = False
        result.append({"Usta": g["name"], "Prensip": g["rule"], "Onay": "AL" if onay else "BEKLE"})
    return result

# ====== AI DANISMAN ======
def ai_danisma(api_key, base_url, model, soru, baglam=""):
    """AI'a soru sor, cevap al. OpenAI uyumlu API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        system_msg = """Sen THYAO (Turk Hava Yollari) hissesi ve borsa hakkinda uzman bir danismansin.
Kisa, net ve Turkce cevap ver. Teknik terimleri acikla. Yatirim tavsiyesi verme, bilgi ver.
Bilgi bankandan faydalan ve profesyonel ama anlasilir ol."""
        
        messages = [{"role": "system", "content": system_msg}]
        if baglam:
            messages.append({"role": "system", "content": f"Guncel baglam: {baglam}"})
        messages.append({"role": "user", "content": soru})
        
        r = client.chat.completions.create(model=model, messages=messages, temperature=0.7, max_tokens=800)
        return r.choices[0].message.content
    except Exception as e:
        return f"AI hatasi: {str(e)}"

def ai_karar_danisma(api_key, base_url, model, row, action, score):
    """Botun karari icin AI'a danis."""
    baglam = f"""THYAO analizi:
- Fiyat: {row['Close']:.2f}
- RSI: {row['RSI']:.1f}
- MACD Hist: {row['MACD_hist']:.3f}
- SMA20: {row['SMA20']:.2f}, SMA50: {row['SMA50']:.2f}
- ADX: {row['ADX']:.1f}
- Volatilite: {row['Volatility']*100:.1f}%
- Bot karari: {action} (skor {score:+.2f})"""

    soru = f"Bu verilere gore botun '{action}' karari mantikli mi? 2-3 cumle ile degerlendir."
    return ai_danisma(api_key, base_url, model, soru, baglam)

# ====== VERI ======
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
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).sum() / tr14
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / tr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.rolling(14).mean()
    df["Vol_SMA"] = v.rolling(20).mean()
    df["Vol_ratio"] = v / df["Vol_SMA"]
    df["Returns"] = c.pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252)
    return df.dropna().reset_index(drop=True)

def get_state(row):
    rsi_c = "L" if row["RSI"] < 35 else ("H" if row["RSI"] > 65 else "M")
    macd_c = "P" if row["MACD_hist"] > 0 else "N"
    trend = "U" if row["SMA20"] > row["SMA50"] else "D"
    adx_c = "S" if row.get("ADX", 25) > 25 else "W"
    return f"{rsi_c}{macd_c}{trend}{adx_c}"

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
        c = row["Close"]
        sig["bb"] = 1 if c < row["BB_dn"] else (-1 if c > row["BB_up"] else 0)
        k, d = row["K"], row["D"]
        sig["stoch"] = 1 if k < 20 and k > d else (-1 if k > 80 and k < d else 0)

        base_score = sum(self.w[k] * sig[k] for k in sig)
        state = get_state(row)
        state_bonus = 0.0
        state_history = []
        if state in self.state_performance and self.state_performance[state]["total"] >= 3:
            perf = self.state_performance[state]
            wr = perf["wins"] / perf["total"]
            state_bonus = (wr - 0.5) * 0.6
            state_history.append(f"{perf['total']} gecmis islem, basari %{wr*100:.0f}")

        similar = [e for e in self.experience if e["state"] == state]
        similar_bonus = 0.0
        if len(similar) >= 3:
            avg_pnl = np.mean([e["pnl_pct"] for e in similar])
            similar_bonus = np.clip(avg_pnl / 100, -0.3, 0.3)

        total_score = base_score + state_bonus + similar_bonus

        g_score = guru_score(row, "BUY" if total_score > 0 else "SELL")
        total_score = total_score * 0.7 + (g_score - 0.5) * 0.6
        confidence = min(abs(total_score), 1.0)

        if total_score > min_confidence:
            action = "AL"
        elif total_score < -min_confidence:
            action = "SAT"
        else:
            action = "TUT"

        explanations = []
        if state_history:
            explanations.append(f"Durum {state}: {state_history[0]}")
        if len(similar) >= 3:
            avg = np.mean([e["pnl_pct"] for e in similar])
            explanations.append(f"Benzer {len(similar)} islem ort. {avg:+.1f}%")
        if sig["rsi"] != 0:
            explanations.append(f"RSI {row['RSI']:.0f} -> {'AL' if sig['rsi']>0 else 'SAT'}")
        if sig["macd"] != 0:
            explanations.append(f"MACD {'pozitif' if sig['macd']>0 else 'negatif'}")
        if sig["trend"] != 0:
            explanations.append(f"Trend {'yukari' if sig['trend']>0 else 'asagi'}")

        g_details = guru_details(row)
        approved = [g["Usta"] for g in g_details if g["Onay"] == "AL"]
        if len(approved) >= 15:
            explanations.append(f"👑 {len(approved)}/25 usta onayliyor")
        elif len(approved) >= 8:
            explanations.append(f"👍 {len(approved)}/25 usta onayliyor")
        elif len(approved) <= 5:
            explanations.append(f"⚠️ Sadece {len(approved)}/25 usta onayliyor")

        return action, total_score, sig, confidence, explanations, state, similar, g_score, g_details

    def execute(self, action, price, date, row, reason="AI", signals=None):
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
            self.trades.append({"date": date, "action": "AL", "price": price, "qty": qty,
                                "total": cost, "cash": self.cash, "shares": self.shares,
                                "value": self.value(price), "reason": reason, "state": state})
            return True
        if action == "SAT" and self.shares > 0:
            qty = self.shares
            rev = qty * price
            pnl_pct = (price / self.entry_price - 1) * 100 if self.entry_price else 0
            if self.entry_price and self.entry_signals:
                exp = {"state": self.entry_state, "signals": dict(self.entry_signals),
                       "buy_price": self.entry_price, "sell_price": price,
                       "pnl_pct": pnl_pct, "date": str(date)}
                self.experience.append(exp)
                if self.entry_state not in self.state_performance:
                    self.state_performance[self.entry_state] = {"wins": 0, "total": 0}
                self.state_performance[self.entry_state]["total"] += 1
                if pnl_pct > 0:
                    self.state_performance[self.entry_state]["wins"] += 1
                for sig_name, sig_val in self.entry_signals.items():
                    if sig_name not in self.signal_accuracy or sig_val == 0:
                        continue
                    is_correct = (sig_val > 0 and pnl_pct > 0) or (sig_val < 0 and pnl_pct < 0)
                    self.signal_accuracy[sig_name]["total"] += 1
                    if is_correct:
                        self.signal_accuracy[sig_name]["correct"] += 1
                        self.w[sig_name] = min(0.5, self.w[sig_name] + 0.015)
                    else:
                        self.w[sig_name] = max(0.02, self.w[sig_name] - 0.015)
                if pnl_pct > 3:
                    self.lessons.append(f"✅ {self.entry_state} durumunda AL dogruydu ({pnl_pct:+.1f}%)")
                elif pnl_pct < -3:
                    self.lessons.append(f"❌ {self.entry_state} durumunda AL yanlisti ({pnl_pct:+.1f}%)")
                t = sum(self.w.values())
                if t > 0:
                    for k in self.w:
                        self.w[k] /= t
            self.cash += rev
            self.shares = 0
            self.entry_price = None
            self.entry_state = None
            self.entry_signals = None
            self.trades.append({"date": date, "action": "SAT", "price": price, "qty": qty,
                                "total": rev, "cash": self.cash, "shares": 0,
                                "value": self.value(price), "reason": reason, "state": state})
            return True
        return False

    def get_learning_report(self):
        state_stats = []
        for state, perf in self.state_performance.items():
            wr = perf["wins"] / perf["total"] * 100 if perf["total"] else 0
            state_stats.append({"Durum": state, "Islem": perf["total"], "Basari_%": round(wr, 1)})
        sig_stats = []
        for name, acc in self.signal_accuracy.items():
            if acc["total"] == 0:
                continue
            correct_pct = acc["correct"] / acc["total"] * 100
            sig_stats.append({"Sinyal": name.upper(), "Toplam": acc["total"],
                              "Dogru": acc["correct"], "Dogruluk_%": round(correct_pct, 1),
                              "Agirlik": round(self.w.get(name, 0), 3)})
        return {"states": sorted(state_stats, key=lambda x: x["Islem"], reverse=True),
                "signals": sig_stats, "lessons": self.lessons[-10:]}

# ====== SESSION ======
if "bot" not in st.session_state:
    st.session_state.bot = LearningBot(100000)
if "log" not in st.session_state:
    st.session_state.log = []
if "chat" not in st.session_state:
    st.session_state.chat = []
if "ai_karar_cache" not in st.session_state:
    st.session_state.ai_karar_cache = None
bot = st.session_state.bot

if auto_on:
    st_autorefresh(interval=60 * 1000, key="auto_refresh")

df = get_data()
if df is None:
    st.error("Veri cekilemedi")
    st.stop()

df = add_indicators(df)
last = df.iloc[-1]
price = float(last["Close"])
date = last["Date"]

action, score, sig, confidence, explanations, state, similar, g_score, g_details = bot.decide(last)
auto_msg = None

if auto_on:
    n = now_tr()
    should_run = (bot.last_auto_time is None) or ((n - bot.last_auto_time).total_seconds() >= interval_min * 60)
    if should_run:
        if market_open() or test_mode:
            ok = bot.execute(action, price, date, last, reason=f"AI {action} (guven %{confidence*100:.0f})", signals=sig)
            bot.last_auto_time = n
            bot.auto_count += 1
            auto_msg = f"{action}: {price:.2f} TL (guven %{confidence*100:.0f}, usta %{g_score*100:.0f})" if ok else f"Karar {action} - kosul yok"
            st.session_state.log.append(f"{n.strftime('%H:%M:%S')} - {auto_msg}")
        else:
            bot.last_auto_time = n
            st.session_state.log.append(f"{now_tr().strftime('%H:%M:%S')} - Borsa kapali")

# ====== SIDEBAR ======
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
    st.subheader("🤖 AI Danisman")
    st.caption("Bana baglanmak icin API key gir")
    ai_key = st.text_input("API Key", type="password", value=qp.get("aik", ""))
    ai_url = st.text_input("Base URL", value=qp.get("aiu", "https://api.openai.com/v1"))
    ai_model = st.text_input("Model", value=qp.get("aim", "gpt-4o-mini"))
    if ai_key and (ai_key != qp.get("aik", "") or ai_url != qp.get("aiu", "")):
        st.query_params["aik"] = ai_key
        st.query_params["aiu"] = ai_url
        st.query_params["aim"] = ai_model
        st.rerun()
    ai_ready = bool(ai_key)

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
    st.write(f"Deneyim: **{len(bot.experience)}**")
    st.write(f"Durum: **{len(bot.state_performance)}**")
    st.write(f"Ders: **{len(bot.lessons)}**")

    if st.button("Sifirla", use_container_width=True):
        st.session_state.bot = LearningBot(100000)
        st.session_state.log = []
        st.session_state.chat = []
        st.rerun()

# ====== UST ======
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fiyat", f"{price:.2f} TL")
c2.metric("RSI", f"{last['RSI']:.1f}")
c3.metric("MACD", f"{last['MACD_hist']:.2f}")
c4.metric("Durum", state)
c5.metric("Usta", f"%{g_score*100:.0f}")

# ====== SEKMELER ======
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🤖 AI Karari", "🧠 Ogrenme", "👑 Ustalar", "🤖 AI Danisman", "📚 Borsa Bilgi", "📈 Grafik", "📜 Islemler"
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

    # AI yorumu
    st.divider()
    st.subheader("🤖 AI'in Yorumu")
    if not ai_ready:
        st.info("AI yorumu icin sol panelden API key gir")
    else:
        if st.button("AI'a Sor", use_container_width=True):
            with st.spinner("AI dusunuyor..."):
                yorum = ai_karar_danisma(ai_key, ai_url, ai_model, last, action, score)
                st.session_state.ai_karar_cache = yorum
        if st.session_state.ai_karar_cache:
            st.success(st.session_state.ai_karar_cache)

    st.divider()
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
        st.dataframe(pd.DataFrame(rep["signals"]), use_container_width=True, hide_index=True)
    else:
        st.info("Henuz veri yok")
    st.divider()
    st.subheader("🎯 Durum Performansi")
    if rep["states"]:
        st.dataframe(pd.DataFrame(rep["states"]), use_container_width=True, hide_index=True)
    else:
        st.info("Henuz veri yok")
    st.divider()
    st.subheader("💡 Ogrenilen Dersler")
    if rep["lessons"]:
        for lesson in reversed(rep["lessons"]):
            st.write(lesson)
    else:
        st.info("Henuz ders yok")

with tab3:
    st.subheader(f"👑 Usta Onay Skoru: %{g_score*100:.0f}")
    onayli = [g for g in g_details if g["Onay"] == "AL"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Onaylayan", f"{len(onayli)}/25")
    c2.metric("Skor", f"%{g_score*100:.0f}")
    c3.metric("Karar", "GUCLU" if g_score > 0.6 else ("ORTA" if g_score > 0.4 else "ZAYIF"))
    st.divider()
    st.dataframe(pd.DataFrame(g_details), use_container_width=True, hide_index=True)
    if g_score > 0.6:
        st.success("🔥 Ustularin cogu AL diyor")
    elif g_score < 0.3:
        st.warning("⚠️ Ustularin cogu BEKLE diyor")

with tab4:
    st.subheader("🤖 AI Danisman")
    if not ai_ready:
        st.warning("Sol panelden API key gir. Ornekler:")
        st.write("- **Groq (ucretsiz):** https://console.groq.com/keys")
        st.write("  Base URL: `https://api.groq.com/openai/v1`")
        st.write("  Model: `llama-3.3-70b-versatile`")
        st.write("- **OpenRouter:** https://openrouter.ai/keys")
        st.write("  Base URL: `https://openrouter.ai/api/v1`")
        st.write("- **OpenAI:** https://platform.openai.com/api-keys")
    else:
        st.success("✅ AI hazir - soru sorabilirsin")
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if prompt := st.chat_input("Borsa hakkinda soru sor..."):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Dusunuyor..."):
                    baglam = f"THYAO: {price:.2f} TL, RSI: {last['RSI']:.1f}, MACD: {last['MACD_hist']:.3f}, Durum: {state}"
                    cevap = ai_danisma(ai_key, ai_url, ai_model, prompt, baglam)
                    st.write(cevap)
                    st.session_state.chat.append({"role": "assistant", "content": cevap})

with tab5:
    st.subheader("📚 Borsa Bilgi Bankasi")
    st.caption(f"Toplam {sum(len(v) for v in BILGI.values())} konu, {len(BILGI)} kategori")
    arama = st.text_input("Ara", placeholder="Ornek: RSI, temettu, stop-loss...")
    if arama:
        sonuc = bilgi_ara(arama)
        if sonuc:
            for s in sonuc:
                with st.expander(f"**{s['Konu']}** ({s['Kategori']})"):
                    st.write(s["Aciklama"])
        else:
            st.warning("Sonuc bulunamadi")
    else:
        for kategori, maddeler in BILGI.items():
            with st.expander(f"📂 {kategori} ({len(maddeler)} konu)"):
                for baslik, icerik in maddeler.items():
                    st.markdown(f"**{baslik}**: {icerik}")

with tab6:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Fiyat"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], name="SMA20", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="SMA50", line=dict(color="cyan")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_up"], name="BB Up", line=dict(color="gray", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_dn"], name="BB Dn", line=dict(color="gray", dash="dot")), row=1, col=1)
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

with tab7:
    if bot.trades:
        tdf = pd.DataFrame(bot.trades)
        tdf["date"] = pd.to_datetime(tdf["date"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(tdf[["date", "action", "price", "qty", "total", "value", "state", "reason"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("Henuz islem yok")
    st.divider()
    st.subheader("📡 Otomatik Log")
    for line in reversed(st.session_state.log[-20:]):
        st.text(line)

st.divider()
st.caption(f"🕐 {now_tr().strftime('%Y-%m-%d %H:%M:%S')} - Borsa: {'ACIK' if market_open() else 'KAPALI'}")
