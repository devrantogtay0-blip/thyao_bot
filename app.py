import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh
import random
import copy

st.set_page_config(page_title="THYAO AI v7 - Super Brain", page_icon="🧠", layout="wide")
st.title("🧠 THYAO AI Pro Trader v7 - Super Brain")
st.caption("Sinir agi + Genetik + Multi-Bot Arena + Actor-Critic")

TR = ZoneInfo("Europe/Istanbul")
def now_tr(): return datetime.now(TR)
def market_open():
    n = now_tr()
    if n.weekday() >= 5: return False
    return dtime(9, 55) <= n.time() <= dtime(18, 10)

HISSELER = {"THYAO": "THYAO.IS", "GARAN": "GARAN.IS", "ASELS": "ASELS.IS", "AKBNK": "AKBNK.IS"}
qp = st.query_params
auto_on = qp.get("auto", "0") == "1"
test_mode = qp.get("test", "0") == "1"
interval_min = int(qp.get("int", "2"))
secili_hisse = qp.get("hisse", "THYAO")
refresh_sn = int(qp.get("rs", "30"))
min_conf = float(qp.get("conf", "0.25"))

# ====== USTALAR ======
GURUS = [
    {"name": "Buffett", "check": lambda r: r["Close"] > r["SMA200"] and r["Volatility"] < 0.3},
    {"name": "Graham", "check": lambda r: r["Close"] < r["BB_dn"]},
    {"name": "Lynch", "check": lambda r: 50 < r["RSI"] < 70 and r["MACD_hist"] > 0},
    {"name": "Munger", "check": lambda r: r["SMA20"] > r["SMA50"] and r["ADX"] > 20},
    {"name": "Fisher", "check": lambda r: r["SMA50"] > r.get("SMA200", r["SMA50"]) * 0.95},
    {"name": "Templeton", "check": lambda r: r["RSI"] < 35},
    {"name": "Soros", "check": lambda r: r["MACD"] > r["MACD_sig"]},
    {"name": "Livermore", "check": lambda r: r["SMA20"] > r["SMA50"] and r["Close"] > r["SMA20"]},
    {"name": "TudorJones", "check": lambda r: r["Close"] > r["SMA50"]},
    {"name": "Dalio", "check": lambda r: r["Volatility"] < 0.35},
    {"name": "Druckenmiller", "check": lambda r: r["ADX"] > 25 and r["MACD_hist"] > 0},
    {"name": "ONeil", "check": lambda r: r["RSI"] > 60 and r["Vol_ratio"] > 1.2},
    {"name": "Minervini", "check": lambda r: r.get("BB_width", 1) < 0.08 and r["ADX"] > 20},
    {"name": "Darvas", "check": lambda r: r["Close"] > r["BB_up"] * 0.98},
    {"name": "Simons", "check": lambda r: r["RSI"] < 30},
    {"name": "Burry", "check": lambda r: r["RSI"] < 25 and r["WILLR"] < -85},
    {"name": "Icahn", "check": lambda r: r["Close"] < r["BB_mid"] and r["ADX"] < 20},
    {"name": "Ackman", "check": lambda r: r["SMA20"] > r["SMA50"] and r["Volatility"] < 0.4},
    {"name": "Klarman", "check": lambda r: r["Close"] < r["BB_dn"] * 1.02},
    {"name": "Marks", "check": lambda r: r["RSI"] < 40 and r["MACD_hist"] > -0.5},
    {"name": "Tepper", "check": lambda r: r["RSI"] < 35 and r["BB_dn"] > r["Close"] * 0.95},
    {"name": "Griffin", "check": lambda r: r["Vol_ratio"] > 1.0 and abs(r["MACD_hist"]) > 0.1},
    {"name": "Cohen", "check": lambda r: r["RSI"] > 55 and r["Vol_ratio"] > 1.3},
    {"name": "LWilliams", "check": lambda r: r["K"] < 20 and r["ATR"] > 0},
    {"name": "Seykota", "check": lambda r: r["SMA20"] > r["SMA50"] and r["ADX"] > 22},
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
        try: onay = bool(g["check"](row))
        except Exception: onay = False
        r.append({"Usta": g["name"], "Onay": "AL" if onay else "BEKLE"})
    return r

# ====== VERI ======
@st.cache_data(ttl=30)
def get_data(symbol, period="2y", interval="1d"):
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    if df.empty: return None
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df

@st.cache_data(ttl=15)
def get_canli(symbol):
    try:
        df = yf.Ticker(symbol).history(period="1d", interval="1m")
        if df.empty: return None
        return {"price": float(df["Close"].iloc[-1]), "open": float(df["Open"].iloc[0]),
                "high": float(df["High"].max()), "low": float(df["Low"].min()),
                "volume": int(df["Volume"].sum()),
                "change_pct": float((df["Close"].iloc[-1] / df["Open"].iloc[0] - 1) * 100)}
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
    up = h.diff(); dn = -l.diff()
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
    # Ekstra
    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["EMA21"] = c.ewm(span=21, adjust=False).mean()
    df["Mom10"] = c.pct_change(10)
    df["Mom30"] = c.pct_change(30)
    df["HL_ratio"] = (h - l) / c
    df["Gap"] = (df["Open"] - c.shift()) / c.shift()
    return df.dropna().reset_index(drop=True)

def get_state(row):
    r = "L" if row["RSI"] < 35 else ("H" if row["RSI"] > 65 else "M")
    m = "P" if row["MACD_hist"] > 0 else "N"
    t = "U" if row["SMA20"] > row["SMA50"] else "D"
    a = "S" if row.get("ADX", 25) > 25 else "W"
    return f"{r}{m}{t}{a}"

def extract_features(row, prev_rows=None):
    """20+ ozellik cikar."""
    f = [
        row["RSI"] / 100,
        np.clip(row["MACD_hist"] * 10, -1, 1),
        (row["SMA20"] / row["SMA50"] - 1) * 10 if row["SMA50"] else 0,
        (row["Close"] - row["BB_mid"]) / (row["BB_up"] - row["BB_dn"] + 1e-9),
        row["K"] / 100,
        row["D"] / 100,
        np.clip(row["ADX"] / 50, 0, 1),
        np.clip(row["Volatility"] * 3, 0, 1),
        np.clip(row["Vol_ratio"] - 1, -1, 1),
        np.clip(row["Mom10"] * 10, -1, 1),
        np.clip(row["Mom30"] * 10, -1, 1),
        np.clip(row["HL_ratio"] * 20, 0, 1),
        np.clip(row["Gap"] * 20, -1, 1),
        np.clip((row["Close"] / row["SMA200"] - 1) * 5 if row["SMA200"] else 0, -1, 1),
        np.clip((row["EMA9"] - row["EMA21"]) / row["Close"] * 20, -1, 1),
        np.clip(row["WILLR"] / 100, -1, 0),
        (1 if row["Close"] > row["SMA20"] else 0),
        (1 if row["Close"] > row["SMA50"] else 0),
        (1 if row["MACD"] > row["MACD_sig"] else 0),
        (1 if row["SMA20"] > row["SMA50"] else 0),
    ]
    return np.array(f, dtype=np.float64)

# ====== SIFIRDAN NEURAL NETWORK (numpy) ======
class NeuralNet:
    """2 katmanli MLP. numpy ile sifirdan."""
    def __init__(self, n_in=20, n_h1=24, n_h2=12, n_out=3, lr=0.01):
        np.random.seed(42)
        self.lr = lr
        # Xavier init
        self.W1 = np.random.randn(n_in, n_h1) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_h1)
        self.W2 = np.random.randn(n_h1, n_h2) * np.sqrt(2.0 / n_h1)
        self.b2 = np.zeros(n_h2)
        self.W3 = np.random.randn(n_h2, n_out) * np.sqrt(2.0 / n_h2)
        self.b3 = np.zeros(n_out)
        self.loss_history = []

    def relu(self, x): return np.maximum(0, x)
    def relu_d(self, x): return (x > 0).astype(float)
    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-9)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)
        z3 = a2 @ self.W3 + self.b3
        return self.softmax(z3), (x, z1, a1, z2, a2, z3)

    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        probs, _ = self.forward(x)
        return probs[0]  # [P(SAT), P(TUT), P(AL)]

    def train_step(self, X, y):
        """Batch egitim. y: 0=SAT, 1=TUT, 2=AL"""
        n = X.shape[0]
        probs, (x, z1, a1, z2, a2, z3) = self.forward(X)
        # Cross-entropy loss
        y_one_hot = np.zeros_like(probs)
        y_one_hot[np.arange(n), y] = 1
        loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-9), axis=1))
        self.loss_history.append(float(loss))
        if len(self.loss_history) > 200: self.loss_history = self.loss_history[-200:]

        # Backprop
        dz3 = (probs - y_one_hot) / n
        dW3 = a2.T @ dz3
        db3 = dz3.sum(axis=0)
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.relu_d(z2)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_d(z1)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        # Gradyan inişi
        self.W3 -= self.lr * dW3; self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

        return loss

    def mutate(self, rate=0.05, scale=0.1):
        for W in [self.W1, self.W2, self.W3]:
            mask = np.random.random(W.shape) < rate
            W += mask * np.random.randn(*W.shape) * scale

    def clone(self):
        new = NeuralNet(self.W1.shape[0], self.W1.shape[1], self.W2.shape[1], self.W3.shape[1], self.lr)
        new.W1 = self.W1.copy(); new.b1 = self.b1.copy()
        new.W2 = self.W2.copy(); new.b2 = self.b2.copy()
        new.W3 = self.W3.copy(); new.b3 = self.b3.copy()
        return new

    def crossover(self, other):
        new = self.clone()
        for attr in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            a = getattr(self, attr); b = getattr(other, attr)
            mask = np.random.random(a.shape) < 0.5
            setattr(new, attr, np.where(mask, a, b))
        return new

    def size(self):
        return sum(w.size for w in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])

# ====== EXPERIENCE REPLAY ======
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = []
        self.capacity = capacity

    def add(self, features, label):
        self.buffer.append((features, label))
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]

    def sample(self, batch_size=32):
        if len(self.buffer) < batch_size: return None
        batch = random.sample(self.buffer, batch_size)
        X = np.array([b[0] for b in batch])
        y = np.array([b[1] for b in batch])
        return X, y

    def size(self): return len(self.buffer)

# ====== AKTOR (Tek Bot) ======
class Bot:
    """Tek yarışmacı bot. Kendi NN'si + hafizasi var."""
    def __init__(self, name="Bot", cash=100000.0, strategy="balanced"):
        self.name = name
        self.cash = cash
        self.shares = 0
        self.initial = cash
        self.trades = []
        self.nn = NeuralNet()
        self.replay = ReplayBuffer(500)
        self.strategy = strategy  # aggressive, balanced, conservative
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.predictions = []  # (features, predicted, actual, correct)
        self.entry_price = None
        self.entry_features = None
        self.score = 0.0

    def value(self, price): return self.cash + self.shares * price

    def decide(self, row, price):
        feat = extract_features(row)
        probs = self.nn.predict(feat)  # [P_SAT, P_TUT, P_AL]
        # Strateji bazli esikler
        thresholds = {"aggressive": (0.3, 0.5), "balanced": (0.35, 0.45), "conservative": (0.4, 0.4)}[self.strategy]
        sat_t, al_t = thresholds
        if probs[2] > al_t: action = "AL"
        elif probs[0] > sat_t: action = "SAT"
        else: action = "TUT"
        return action, probs, feat

    def execute(self, action, price, date, feat, reason="AI"):
        if action == "AL" and self.cash > price * 10:
            qty = int((self.cash * 0.25) / price)
            if qty < 1: return False
            self.cash -= qty * price
            self.shares += qty
            self.entry_price = price
            self.entry_features = feat
            self.trades.append({"date": date, "action": "AL", "price": price, "qty": qty, "value": self.value(price), "reason": reason})
            return True
        if action == "SAT" and self.shares > 0:
            qty = self.shares
            pnl = (price - self.entry_price) * qty if self.entry_price else 0
            self.cash += qty * price
            self.shares = 0
            self.total_pnl += pnl
            # Etiket: dogru mu yapti?
            if pnl > 0: self.wins += 1
            else: self.losses += 1
            # NN egitimi icin ornek
            if self.entry_features is not None:
                label = 2 if pnl > 0 else 0  # AL iyi = 2 (AL), kotu = 0 (SAT)
                self.replay.add(self.entry_features, label)
            # Prediction kaydet
            if self.entry_features is not None:
                self.predictions.append({"correct": pnl > 0})
            self.trades.append({"date": date, "action": "SAT", "price": price, "qty": qty, "value": self.value(price), "reason": reason, "pnl": pnl})
            self.entry_price = None
            self.entry_features = None
            return True
        return False

    def train_from_replay(self, epochs=3, batch=32):
        if self.replay.size() < batch: return None
        total_loss = 0.0
        cnt = 0
        for _ in range(epochs):
            s = self.replay.sample(batch)
            if s is None: break
            X, y = s
            # TUT etiketleri ekle (her zaman olası)
            loss = self.nn.train_step(X, y)
            total_loss += loss; cnt += 1
        return total_loss / cnt if cnt else None

    def win_rate(self):
        t = self.wins + self.losses
        return self.wins / t if t > 0 else 0.0

    def fitness(self):
        """Genetik algoritma icin uygunluk."""
        if not self.trades: return -999
        val = self.value(self.last_price) if hasattr(self, "last_price") else self.initial + self.total_pnl
        return (val / self.initial - 1) * 100 + self.win_rate() * 20

# ====== ARENA (Multi-Bot) ======
class Arena:
    def __init__(self, cash=100000.0):
        self.bots = [
            Bot("Alpha", cash, "aggressive"),
            Bot("Beta", cash, "balanced"),
            Bot("Gamma", cash, "conservative"),
            Bot("Delta", cash, "balanced"),
            Bot("Epsilon", cash, "aggressive"),
            Bot("Zeta", cash, "conservative"),
        ]
        self.generation = 1
        self.champion_history = []
        self.last_evolve_at = 0

    def best_bot(self, price):
        for b in self.bots: b.last_price = price
        return max(self.bots, key=lambda b: b.value(price))

    def worst_bot(self, price):
        for b in self.bots: b.last_price = price
        return min(self.bots, key=lambda b: b.value(price))

    def evolve(self, price):
        """En iyi 2 botun cocugu eski botlari degistirir."""
        for b in self.bots: b.last_price = price
        sorted_bots = sorted(self.bots, key=lambda b: b.value(price), reverse=True)
        best1, best2 = sorted_bots[0], sorted_bots[1]
        worst1, worst2 = sorted_bots[-1], sorted_bots[-2]

        # Cocuk NN: crossover + mutation
        child1_nn = best1.nn.crossover(best2.nn); child1_nn.mutate(0.05, 0.1)
        child2_nn = best2.nn.crossover(best1.nn); child2_nn.mutate(0.05, 0.1)

        # Worst botlari sifirla ve cocuk NN ver
        for bot, new_nn, name in [(worst1, child1_nn, f"Gen{self.generation+1}a"), (worst2, child2_nn, f"Gen{self.generation+1}b")]:
            bot.cash = 100000.0
            bot.shares = 0
            bot.initial = 100000.0
            bot.total_pnl = 0.0
            bot.wins = 0; bot.losses = 0
            bot.trades = []
            bot.nn = new_nn
            bot.replay = ReplayBuffer(500)  # Hafiza sifirla
            bot.name = name
        self.generation += 1
        self.champion_history.append({"gen": self.generation-1, "champion": best1.name, "value": best1.value(price)})

    def leaderboard(self, price):
        for b in self.bots: b.last_price = price
        rows = []
        for b in self.bots:
            v = b.value(price)
            rows.append({
                "Bot": b.name,
                "Strateji": b.strategy,
                "Portfoy": f"{v:,.0f}",
                "Getiri %": f"{(v/b.initial-1)*100:+.2f}",
                "Islem": len(b.trades),
                "Kazanma %": f"%{b.win_rate()*100:.0f}",
                "NN Deneyim": b.replay.size(),
            })
        return sorted(rows, key=lambda x: float(x["Portfoy"].replace(",", "")), reverse=True)

# ====== SESSION ======
if "arena" not in st.session_state:
    st.session_state.arena = Arena(100000)
if "log" not in st.session_state:
    st.session_state.log = []
arena = st.session_state.arena

if auto_on:
    st_autorefresh(interval=refresh_sn * 1000, key="auto_refresh")

symbol = HISSELER[secili_hisse]
df = get_data(symbol)
if df is None:
    st.error("Veri yok"); st.stop()

df = add_indicators(df)
last = df.iloc[-1]
canli = get_canli(symbol)
price = canli["price"] if canli else float(last["Close"])
date = last["Date"]

# Her bot karar verir
decisions = {}
for b in arena.bots:
    b.last_price = price
    action, probs, feat = b.decide(last, price)
    decisions[b.name] = (action, probs, feat)

champion = arena.best_bot(price)

# Otomatik islem
auto_msgs = []
if auto_on:
    n = now_tr()
    if market_open() or test_mode:
        for b in arena.bots:
            action, probs, feat = decisions[b.name]
            ok = b.execute(action, price, date, feat, reason=f"AI {action}")
            if ok:
                auto_msgs.append(f"{b.name}: {action} @ {price:.2f}")

        # NN egitimi
        for b in arena.bots:
            loss = b.train_from_replay(epochs=2, batch=16)
        
        # Nesil ilerlemesi - 20 islem sonrasi evrim
        arena.last_evolve_at += 1
        if arena.last_evolve_at >= 50:
            arena.evolve(price)
            arena.last_evolve_at = 0
            st.session_state.log.append(f"{n.strftime('%H:%M:%S')} - 🧬 Yeni nesil! Gen {arena.generation}")

        if auto_msgs:
            st.session_state.log.append(f"{n.strftime('%H:%M:%S')} - " + " | ".join(auto_msgs))

# ====== SIDEBAR ======
with st.sidebar:
    st.header("🧠 Super Brain v7")
    hs = st.selectbox("Hisse", list(HISSELER.keys()), index=list(HISSELER.keys()).index(secili_hisse))
    if hs != secili_hisse:
        st.query_params["hisse"] = hs; st.rerun()

    st.metric(f"{secili_hisse}", f"{price:.2f}", f"{canli['change_pct']:+.2f}%" if canli else "")

    st.divider()
    st.subheader("🏆 Sampiyon")
    st.metric(champion.name, f"{champion.value(price):,.0f} TL",
              f"{(champion.value(price)/champion.initial-1)*100:+.2f}%")
    st.caption(f"Strateji: {champion.strategy} | Kazanma: %{champion.win_rate()*100:.0f}")

    st.divider()
    st.subheader("Kontroller")
    rs = st.slider("Yenile (sn)", 10, 300, refresh_sn)
    if rs != refresh_sn:
        st.query_params["rs"] = str(rs); st.rerun()

    aw = st.toggle("Otomatik", value=auto_on)
    if aw != auto_on:
        st.query_params["auto"] = "1" if aw else "0"; st.rerun()
    tw = st.toggle("Test Modu", value=test_mode)
    if tw != test_mode:
        st.query_params["test"] = "1" if tw else "0"; st.rerun()

    st.divider()
    st.subheader("🧬 Evrim")
    st.write(f"Nesil: **{arena.generation}**")
    st.write(f"Sonraki evrim: **{50 - arena.last_evolve_at}** islem")

    if st.button("🔄 Sifirla", use_container_width=True):
        st.session_state.arena = Arena(100000)
        st.session_state.log = []
        st.rerun()

# ====== ÜST ======
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{secili_hisse}", f"{price:.2f}")
c2.metric("Rejim", "TREND" if last.get("ADX", 20) > 25 else "RANGE")
c3.metric("Volatilite", f"{last['Volatility']*100:.1f}%")
c4.metric("Nesil", f"Gen {arena.generation}")

if auto_msgs:
    st.info(" | ".join(auto_msgs[:3]))

# ====== SEKMELER ======
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏟️ Arena", "🧠 Sinir Agi", "🧬 Genetik", "📊 Kararlar", "📜 Islemler", "📡 Log"
])

with tab1:
    st.subheader("🏟️ Multi-Bot Arena - 6 Bot Yarisi")
    lb = arena.leaderboard(price)
    st.dataframe(pd.DataFrame(lb), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📊 Portfoy Karsilastirmasi")
    fig = go.Figure()
    for b in arena.bots:
        vals = [b.initial]
        for t in b.trades:
            vals.append(t["value"])
        vals.append(b.value(price))
        fig.add_trace(go.Scatter(y=vals, name=b.name, mode="lines+markers"))
    fig.update_layout(height=400, template="plotly_dark", xaxis_title="Islem Sirasi", yaxis_title="Portfoy (TL)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🧠 Sinir Agi Analizi")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mimari", "20-24-12-3")
    c2.metric("Parametre", f"{champion.nn.size()}")
    c3.metric("Egitim Adimi", len(champion.nn.loss_history))
    st.caption("Giris: 20 ozellik | Gizli: 24 + 12 noron | Cikis: SAT/TUT/AL")

    if champion.nn.loss_history:
        st.divider()
        st.subheader("📉 Egitim Kaybi (Loss)")
        st.line_chart(pd.DataFrame({"Loss": champion.nn.loss_history}))

    st.divider()
    st.subheader("🎯 Aktif Botun Sinyalleri")
    action, probs, feat = decisions[champion.name]
    c1, c2, c3 = st.columns(3)
    c1.metric("SAT Olasiligi", f"%{probs[0]*100:.1f}")
    c2.metric("TUT Olasiligi", f"%{probs[1]*100:.1f}")
    c3.metric("AL Olasiligi", f"%{probs[2]*100:.1f}")
    st.write(f"**Karar: {action}**")

with tab3:
    st.subheader("🧬 Genetik Algoritma")
    st.caption("En iyi 2 bot cocuk uretir, en kotu 2 bot yerini cocuklara birakir.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nesil", arena.generation)
    c2.metric("Toplam Evrim", len(arena.champion_history))
    c3.metric("Bot Sayisi", len(arena.bots))

    if arena.champion_history:
        st.divider()
        st.subheader("🏆 Sampiyon Gecmisi")
        st.dataframe(pd.DataFrame(arena.champion_history), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🧪 Her Botun NN Istatistigi")
    rows = []
    for b in arena.bots:
        rows.append({
            "Bot": b.name,
            "Strateji": b.strategy,
            "Hafiza": b.replay.size(),
            "Egitim": len(b.nn.loss_history),
            "Son Loss": f"{b.nn.loss_history[-1]:.4f}" if b.nn.loss_history else "-",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("📊 Tum Botlarin Kararlari")
    rows = []
    for name, (a, p, f) in decisions.items():
        rows.append({
            "Bot": name,
            "Karar": a,
            "P(SAT)": f"%{p[0]*100:.0f}",
            "P(TUT)": f"%{p[1]*100:.0f}",
            "P(AL)": f"%{p[2]*100:.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("👑 Usta Onayi")
    gd = guru_details(last)
    onay = sum(1 for g in gd if g["Onay"] == "AL")
    st.metric("Onaylayan Usta", f"{onay}/25")
    st.dataframe(pd.DataFrame(gd), use_container_width=True, hide_index=True)

with tab5:
    st.subheader("📜 Sampiyon Islemleri")
    if champion.trades:
        tdf = pd.DataFrame(champion.trades)
        tdf["date"] = pd.to_datetime(tdf["date"]).dt.strftime("%Y-%m-%d %H:%M")
        cols = [c for c in ["date", "action", "price", "qty", "value", "pnl", "reason"] if c in tdf.columns]
        st.dataframe(tdf[cols], use_container_width=True, hide_index=True)
    else:
        st.info("Islem yok")

with tab6:
    for line in reversed(st.session_state.log[-30:]):
        st.text(line)

st.divider()
st.caption(f"🕐 {now_tr().strftime('%Y-%m-%d %H:%M:%S')} - Borsa: {'ACIK' if market_open() else 'KAPALI'}")
