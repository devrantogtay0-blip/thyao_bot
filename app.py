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

st.set_page_config(page_title="THYAO AI v8", page_icon="🧠", layout="wide")
st.title("🧠 THYAO AI Trader v8")
st.caption("Self-Evolving AI: Sinir agi + Genetik + Meta-Evrim + Arena + 25 Usta")

TR = ZoneInfo("Europe/Istanbul")
def now_tr(): return datetime.now(TR)
def market_open():
    n = now_tr()
    if n.weekday() >= 5: return False
    return dtime(9, 55) <= n.time() <= dtime(18, 10)

# ====== HISSELER ======
HISSELER = {
    "THYAO": "THYAO.IS", "GARAN": "GARAN.IS", "ASELS": "ASELS.IS",
    "AKBNK": "AKBNK.IS", "EREGL": "EREGL.IS", "TUPRS": "TUPRS.IS",
    "SISE": "SISE.IS", "KCHOL": "KCHOL.IS",
}

# ====== URL PARAMS (guvenli) ======
qp = st.query_params
def qp_get(key, default):
    try:
        v = qp.get(key)
        return v if v is not None else default
    except Exception:
        return default

def qp_set(key, value):
    try:
        st.query_params[key] = str(value)
    except Exception:
        pass

hisse_param = qp_get("hisse", "THYAO")
secili_hisse = hisse_param if hisse_param in HISSELER else "THYAO"
auto_on = qp_get("auto", "0") == "1"
test_mode = qp_get("test", "0") == "1"
try:
    interval_min = int(qp_get("int", "2"))
except Exception:
    interval_min = 2
try:
    refresh_sn = int(qp_get("rs", "30"))
except Exception:
    refresh_sn = 30

# ====== 25 USTA ======
GURUS = [
    {"n": "Buffett",      "f": lambda r: r["Close"] > r["SMA200"] and r["Volatility"] < 0.3},
    {"n": "Graham",       "f": lambda r: r["Close"] < r["BB_dn"]},
    {"n": "Lynch",        "f": lambda r: 50 < r["RSI"] < 70 and r["MACD_hist"] > 0},
    {"n": "Munger",       "f": lambda r: r["SMA20"] > r["SMA50"] and r["ADX"] > 20},
    {"n": "Fisher",       "f": lambda r: r["SMA50"] > r.get("SMA200", r["SMA50"]) * 0.95},
    {"n": "Templeton",    "f": lambda r: r["RSI"] < 35},
    {"n": "Soros",        "f": lambda r: r["MACD"] > r["MACD_sig"]},
    {"n": "Livermore",    "f": lambda r: r["SMA20"] > r["SMA50"] and r["Close"] > r["SMA20"]},
    {"n": "TudorJones",   "f": lambda r: r["Close"] > r["SMA50"]},
    {"n": "Dalio",        "f": lambda r: r["Volatility"] < 0.35},
    {"n": "Druckenmiller","f": lambda r: r["ADX"] > 25 and r["MACD_hist"] > 0},
    {"n": "ONeil",        "f": lambda r: r["RSI"] > 60 and r["Vol_ratio"] > 1.2},
    {"n": "Minervini",    "f": lambda r: r.get("BB_width", 1) < 0.08 and r["ADX"] > 20},
    {"n": "Darvas",       "f": lambda r: r["Close"] > r["BB_up"] * 0.98},
    {"n": "Simons",       "f": lambda r: r["RSI"] < 30},
    {"n": "Burry",        "f": lambda r: r["RSI"] < 25 and r["WILLR"] < -85},
    {"n": "Icahn",        "f": lambda r: r["Close"] < r["BB_mid"] and r["ADX"] < 20},
    {"n": "Ackman",       "f": lambda r: r["SMA20"] > r["SMA50"] and r["Volatility"] < 0.4},
    {"n": "Klarman",      "f": lambda r: r["Close"] < r["BB_dn"] * 1.02},
    {"n": "Marks",        "f": lambda r: r["RSI"] < 40 and r["MACD_hist"] > -0.5},
    {"n": "Tepper",       "f": lambda r: r["RSI"] < 35 and r["BB_dn"] > r["Close"] * 0.95},
    {"n": "Griffin",      "f": lambda r: r["Vol_ratio"] > 1.0 and abs(r["MACD_hist"]) > 0.1},
    {"n": "Cohen",        "f": lambda r: r["RSI"] > 55 and r["Vol_ratio"] > 1.3},
    {"n": "LWilliams",    "f": lambda r: r["K"] < 20 and r["ATR"] > 0},
    {"n": "Seykota",      "f": lambda r: r["SMA20"] > r["SMA50"] and r["ADX"] > 22},
]

def guru_score(row, side="BUY"):
    c = 0
    for g in GURUS:
        try:
            v = bool(g["f"](row))
            if side == "SELL": v = not v
            if v: c += 1
        except Exception:
            pass
    return c / len(GURUS)

def guru_details(row):
    r = []
    for g in GURUS:
        try: ok = bool(g["f"](row))
        except Exception: ok = False
        r.append({"Usta": g["n"], "Onay": "AL" if ok else "BEKLE"})
    return r

# ====== VERI ======
@st.cache_data(ttl=30)
def get_data(symbol, period="2y"):
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d")
        if df.empty: return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df
    except Exception:
        return None

@st.cache_data(ttl=15)
def get_canli(symbol):
    try:
        df = yf.Ticker(symbol).history(period="1d", interval="1m")
        if df.empty: return None
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
    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["EMA21"] = c.ewm(span=21, adjust=False).mean()
    df["Mom10"] = c.pct_change(10)
    df["Mom30"] = c.pct_change(30)
    df["HL_ratio"] = (h - l) / c
    df["Gap"] = (df["Open"] - c.shift()) / c.shift()
    return df.dropna().reset_index(drop=True)

def extract_features(row):
    f = [
        row["RSI"] / 100,
        np.clip(row["MACD_hist"] * 10, -1, 1),
        np.clip((row["SMA20"] / row["SMA50"] - 1) * 10, -1, 1) if row["SMA50"] else 0,
        np.clip((row["Close"] - row["BB_mid"]) / (row["BB_up"] - row["BB_dn"] + 1e-9), -1, 1),
        row["K"] / 100,
        np.clip(row["ADX"] / 50, 0, 1),
        np.clip(row["Volatility"] * 3, 0, 1),
        np.clip(row["Vol_ratio"] - 1, -1, 1),
        np.clip(row["Mom10"] * 10, -1, 1),
        np.clip(row["HL_ratio"] * 20, 0, 1),
        np.clip(row["Gap"] * 20, -1, 1),
        np.clip((row["Close"] / row["SMA200"] - 1) * 5, -1, 1) if row["SMA200"] else 0,
        np.clip((row["EMA9"] - row["EMA21"]) / row["Close"] * 20, -1, 1),
        1 if row["Close"] > row["SMA20"] else 0,
        1 if row["Close"] > row["SMA50"] else 0,
        1 if row["MACD"] > row["MACD_sig"] else 0,
        1 if row["SMA20"] > row["SMA50"] else 0,
        np.clip(row["WILLR"] / 100, -1, 0),
        np.clip(row["Mom30"] * 10, -1, 1),
        1 if row["Close"] > row["BB_mid"] else 0,
    ]
    return np.array(f, dtype=np.float64)

# ====== SIFIRDAN NEURAL NETWORK ======
class NeuralNet:
    def __init__(self, n_in=20, h1=24, h2=12, n_out=3, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(n_in, h1) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, n_out) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(n_out)
        self.loss_history = []

    def _relu(self, x): return np.maximum(0, x)
    def _relu_d(self, x): return (x > 0).astype(float)
    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        z3 = a2 @ self.W3 + self.b3
        return self._softmax(z3), (x, z1, a1, z2, a2, z3)

    def predict(self, x):
        if x.ndim == 1: x = x.reshape(1, -1)
        p, _ = self.forward(x)
        return p[0]

    def train_step(self, X, y):
        n = X.shape[0]
        probs, (x, z1, a1, z2, a2, z3) = self.forward(X)
        y1h = np.zeros_like(probs)
        y1h[np.arange(n), y] = 1
        loss = -np.mean(np.sum(y1h * np.log(probs + 1e-9), axis=1))
        self.loss_history.append(float(loss))
        if len(self.loss_history) > 100: self.loss_history = self.loss_history[-100:]
        dz3 = (probs - y1h) / n
        dW3 = a2.T @ dz3; db3 = dz3.sum(0)
        da2 = dz3 @ self.W3.T; dz2 = da2 * self._relu_d(z2)
        dW2 = a1.T @ dz2; db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T; dz1 = da1 * self._relu_d(z1)
        dW1 = x.T @ dz1; db1 = dz1.sum(0)
        self.W3 -= self.lr * dW3; self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
        return loss

    def mutate(self, rate=0.05, scale=0.1):
        for W in [self.W1, self.W2, self.W3]:
            m = np.random.random(W.shape) < rate
            W += m * np.random.randn(*W.shape) * scale

    def clone(self):
        n = NeuralNet(self.W1.shape[0], self.W1.shape[1], self.W2.shape[1], self.W3.shape[1], self.lr)
        n.W1 = self.W1.copy(); n.b1 = self.b1.copy()
        n.W2 = self.W2.copy(); n.b2 = self.b2.copy()
        n.W3 = self.W3.copy(); n.b3 = self.b3.copy()
        return n

    def crossover(self, other):
        n = self.clone()
        for a in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            A = getattr(self, a); B = getattr(other, a)
            m = np.random.random(A.shape) < 0.5
            setattr(n, a, np.where(m, A, B))
        return n

    def n_params(self):
        return sum(w.size for w in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])

# ====== REPLAY BUFFER ======
class ReplayBuffer:
    def __init__(self, cap=500):
        self.buf = []
        self.cap = cap
    def add(self, feat, label):
        self.buf.append((feat, label))
        if len(self.buf) > self.cap: self.buf = self.buf[-self.cap:]
    def sample(self, bs=32):
        if len(self.buf) < bs: return None
        b = random.sample(self.buf, bs)
        return np.array([x[0] for x in b]), np.array([x[1] for x in b])
    def size(self): return len(self.buf)

# ====== BOT ======
class Bot:
    def __init__(self, name, cash=100000.0, strategy="balanced"):
        self.name = name
        self.cash = cash
        self.shares = 0
        self.initial = cash
        self.trades = []
        self.nn = NeuralNet()
        self.replay = ReplayBuffer(500)
        self.strategy = strategy
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.entry_price = None
        self.entry_features = None
        self.last_price = cash
        # Meta DNA
        self.risk_pct = 0.25
        self.min_conf = 0.25
        self.sl_pct = 3.0
        self.tp_pct = 6.0

    def value(self, p): return self.cash + self.shares * p

    def decide(self, row, price):
        feat = extract_features(row)
        probs = self.nn.predict(feat)
        thr = {"aggressive": (0.30, 0.50), "balanced": (0.35, 0.45), "conservative": (0.40, 0.40)}.get(self.strategy, (0.35, 0.45))
        if probs[2] > thr[1]: a = "AL"
        elif probs[0] > thr[0]: a = "SAT"
        else: a = "TUT"
        return a, probs, feat

    def execute(self, action, price, date, feat, reason="AI"):
        # Stop / TP kontrol
        if self.shares > 0 and self.entry_price:
            pct = (price / self.entry_price - 1) * 100
            if pct <= -self.sl_pct:
                action = "SAT"; reason = f"STOP-LOSS ({pct:+.1f}%)"
            elif pct >= self.tp_pct:
                action = "SAT"; reason = f"TAKE-PROFIT ({pct:+.1f}%)"

        if action == "AL" and self.cash > price * 10:
            q = int((self.cash * self.risk_pct) / price)
            if q < 1: return False
            self.cash -= q * price
            self.shares += q
            self.entry_price = price
            self.entry_features = feat
            self.trades.append({"date": date, "action": "AL", "price": price, "qty": q,
                                "value": self.value(price), "reason": reason})
            return True
        if action == "SAT" and self.shares > 0:
            q = self.shares
            pnl = (price - self.entry_price) * q if self.entry_price else 0
            self.cash += q * price
            self.shares = 0
            self.total_pnl += pnl
            if pnl > 0: self.wins += 1
            else: self.losses += 1
            if self.entry_features is not None:
                label = 2 if pnl > 0 else 0
                self.replay.add(self.entry_features, label)
            self.trades.append({"date": date, "action": "SAT", "price": price, "qty": q,
                                "value": self.value(price), "reason": reason, "pnl": pnl})
            self.entry_price = None
            self.entry_features = None
            return True
        return False

    def train(self, epochs=2, bs=16):
        if self.replay.size() < bs: return None
        losses = []
        for _ in range(epochs):
            s = self.replay.sample(bs)
            if s is None: break
            X, y = s
            losses.append(self.nn.train_step(X, y))
        return float(np.mean(losses)) if losses else None

    def win_rate(self):
        t = self.wins + self.losses
        return self.wins / t if t > 0 else 0.0

    def fitness(self):
        v = self.value(self.last_price)
        return (v / self.initial - 1) * 100 + self.win_rate() * 20

# ====== ARENA ======
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
        self.evolve_counter = 0
        self.champion_name = "Alpha"

    def best(self, price):
        for b in self.bots: b.last_price = price
        return max(self.bots, key=lambda b: b.value(price))

    def evolve(self, price):
        for b in self.bots: b.last_price = price
        sb = sorted(self.bots, key=lambda b: b.value(price), reverse=True)
        b1, b2 = sb[0], sb[1]
        self.champion_name = b1.name
        w1, w2 = sb[-1], sb[-2]
        c1 = b1.nn.crossover(b2.nn); c1.mutate(0.05, 0.1)
        c2 = b2.nn.crossover(b1.nn); c2.mutate(0.05, 0.1)
        for bot, nn, nm in [(w1, c1, f"G{self.generation+1}a"), (w2, c2, f"G{self.generation+1}b")]:
            bot.cash = 100000.0
            bot.shares = 0
            bot.initial = 100000.0
            bot.total_pnl = 0.0
            bot.wins = 0; bot.losses = 0
            bot.trades = []
            bot.nn = nn
            bot.replay = ReplayBuffer(500)
            bot.name = nm
        self.generation += 1
        self.champion_history.append({"gen": self.generation-1, "champion": b1.name,
                                       "value": round(b1.value(price), 0)})

    def leaderboard(self, price):
        for b in self.bots: b.last_price = price
        rows = []
        for b in self.bots:
            v = b.value(price)
            rows.append({
                "Bot": b.name, "Strateji": b.strategy,
                "Portfoy": f"{v:,.0f}",
                "Getiri%": f"{(v/b.initial-1)*100:+.2f}",
                "Islem": len(b.trades),
                "Kazanma%": f"%{b.win_rate()*100:.0f}",
                "Hafiza": b.replay.size(),
            })
        return sorted(rows, key=lambda x: float(x["Portfoy"].replace(",", "")), reverse=True)

# ====== META EVOLVER ======
class MetaEvolver:
    def __init__(self, n=8):
        self.n = n
        self.pop = [self._rand() for _ in range(n)]
        self.hall = []
        self.gen = 0
        self.hist = []
        self.reflections = []

    def _rand(self):
        return {
            "risk_pct": float(np.random.uniform(0.1, 0.4)),
            "min_conf": float(np.random.uniform(0.15, 0.45)),
            "sl_pct": float(np.random.uniform(1.5, 5.0)),
            "tp_pct": float(np.random.uniform(3.0, 12.0)),
            "use_bb": bool(np.random.choice([True, False])),
            "use_adx": bool(np.random.choice([True, False])),
        }

    def _mut(self, dna, rate=0.3):
        new = dict(dna)
        for k, v in new.items():
            if np.random.random() < rate:
                if isinstance(v, bool):
                    new[k] = not v
                elif isinstance(v, float):
                    new[k] = v * float(np.random.uniform(0.85, 1.15))
                    if k == "risk_pct": new[k] = float(np.clip(new[k], 0.05, 0.5))
                    if k == "min_conf": new[k] = float(np.clip(new[k], 0.1, 0.5))
                    if k == "sl_pct": new[k] = float(np.clip(new[k], 1.0, 8.0))
                    if k == "tp_pct": new[k] = float(np.clip(new[k], 2.0, 20.0))
        return new

    def _cross(self, a, b):
        return {k: (a[k] if np.random.random() < 0.5 else b[k]) for k in a}

    def _backtest(self, dna, df):
        if len(df) < 60: return -999
        cash = 100000.0; shares = 0; entry = None
        wins = losses = 0
        peak = cash; max_dd = 0
        for i in range(1, len(df)):
            row = df.iloc[i]; price = float(row["Close"])
            sig = 0
            if row["RSI"] < 35: sig += 1
            if row["RSI"] > 65: sig -= 1
            if row["MACD_hist"] > 0: sig += 1
            else: sig -= 1
            if dna.get("use_bb"):
                if row["Close"] < row["BB_dn"]: sig += 1
                if row["Close"] > row["BB_up"]: sig -= 1
            if dna.get("use_adx") and row.get("ADX", 25) < 18:
                sig = int(np.sign(sig))
            if sig >= 2 and shares == 0 and cash > price * 10:
                q = int((cash * dna["risk_pct"]) / price)
                if q > 0:
                    cash -= q * price; shares = q; entry = price
            elif shares > 0 and entry:
                pct = (price / entry - 1) * 100
                if sig <= -2 or pct <= -dna["sl_pct"] or pct >= dna["tp_pct"]:
                    cash += shares * price
                    if price > entry: wins += 1
                    else: losses += 1
                    shares = 0; entry = None
            val = cash + shares * price
            peak = max(peak, val)
            max_dd = max(max_dd, (peak - val) / peak * 100)
        final = cash + shares * float(df["Close"].iloc[-1])
        ret = (final / 100000.0 - 1) * 100
        t = wins + losses
        wr = wins / t if t > 0 else 0
        return ret + wr * 20 - max_dd * 0.5

    def evolve(self, df):
        scored = [(self._backtest(d, df), d) for d in self.pop]
        scored.sort(key=lambda x: x[0], reverse=True)
        self.gen += 1
        best_fit, best_dna = scored[0]
        self.hist.append(best_fit)
        self.hall.append({"gen": self.gen, "fit": round(best_fit, 2), "dna": dict(best_dna)})
        self.hall = sorted(self.hall, key=lambda x: x["fit"], reverse=True)[:8]

        new_pop = [scored[0][1], scored[1][1]]
        while len(new_pop) < self.n:
            if np.random.random() < 0.6:
                a = scored[np.random.randint(0, min(4, len(scored)))][1]
                b = scored[np.random.randint(0, min(4, len(scored)))][1]
                new_pop.append(self._mut(self._cross(a, b), 0.2))
            else:
                new_pop.append(self._mut(scored[0][1], 0.4))
        self.pop = new_pop

        refl = self._reflect(best_dna)
        if refl: self.reflections.append(refl)
        self.reflections = self.reflections[-15:]
        return best_dna, best_fit

    def _reflect(self, dna):
        parts = []
        parts.append("Bollinger kullaniyorum" if dna["use_bb"] else "Bollinger'siz calisiyorum")
        parts.append("ADX filtresi aktif" if dna["use_adx"] else "ADX kullanmiyorum")
        parts.append(f"SL: %{dna['sl_pct']:.1f}")
        parts.append(f"TP: %{dna['tp_pct']:.1f}")
        parts.append(f"Risk: %{dna['risk_pct']*100:.0f}")
        trend = ""
        if len(self.hist) >= 2:
            trend = "📈" if self.hist[-1] > self.hist[-2] else "📉"
        return f"Gen {self.gen}: {trend} " + " | ".join(parts)

# ====== SESSION ======
if "arena" not in st.session_state:
    st.session_state.arena = Arena(100000)
if "meta" not in st.session_state:
    st.session_state.meta = MetaEvolver(8)
if "log" not in st.session_state:
    st.session_state.log = []
if "last_meta" not in st.session_state:
    st.session_state.last_meta = 0
if "alarms" not in st.session_state:
    st.session_state.alarms = []

arena = st.session_state.arena
meta = st.session_state.meta

if auto_on:
    st_autorefresh(interval=refresh_sn * 1000, key="rf")

# ====== VERI CEK ======
symbol = HISSELER[secili_hisse]
df = get_data(symbol)
if df is None or df.empty:
    st.error(f"{secili_hisse} verisi cekilemedi. Farkli hisse sec.")
    st.stop()

df = add_indicators(df)
if df.empty:
    st.error("Yetersiz veri")
    st.stop()

last = df.iloc[-1]
canli = get_canli(symbol)
price = canli["price"] if canli else float(last["Close"])
date = last["Date"]

# ====== ALARM ======
for a in st.session_state.alarms:
    if not a.get("done") and a["hisse"] == secili_hisse:
        if (a["yon"] == "ust" and price >= a["fiyat"]) or (a["yon"] == "alt" and price <= a["fiyat"]):
            a["done"] = True
            st.session_state.log.append(f"{now_tr().strftime('%H:%M:%S')} - 🚨 {a['hisse']} {a['yon']} {a['fiyat']} tetiklendi ({price:.2f})")

# ====== HER BOT KARAR ======
decisions = {}
for b in arena.bots:
    b.last_price = price
    a, p, f = b.decide(last, price)
    decisions[b.name] = (a, p, f)

champion = arena.best(price)

# ====== OTOMATIK ======
auto_msgs = []
if auto_on and (market_open() or test_mode):
    n = now_tr()
    for b in arena.bots:
        a, p, f = decisions[b.name]
        if b.execute(a, price, date, f, reason=f"AI {a}"):
            auto_msgs.append(f"{b.name}:{a}")
    for b in arena.bots:
        b.train(2, 16)
    arena.evolve_counter += 1
    if arena.evolve_counter >= 50:
        arena.evolve(price)
        arena.evolve_counter = 0
        st.session_state.log.append(f"{n.strftime('%H:%M:%S')} - 🧬 Arena Gen {arena.generation}")
    if auto_msgs:
        st.session_state.log.append(f"{n.strftime('%H:%M:%S')} - " + " | ".join(auto_msgs))

# ====== META EVRIM (100 islemde bir) ======
total_tr = sum(len(b.trades) for b in arena.bots)
if total_tr > 0 and total_tr // 100 > st.session_state.last_meta:
    st.session_state.last_meta = total_tr // 100
    best_dna, fit = meta.evolve(df)
    champion.risk_pct = best_dna["risk_pct"]
    champion.min_conf = best_dna["min_conf"]
    champion.sl_pct = best_dna["sl_pct"]
    champion.tp_pct = best_dna["tp_pct"]
    st.session_state.log.append(f"{now_tr().strftime('%H:%M:%S')} - 🔬 Meta Gen {meta.gen} fit {fit:+.2f}")

# ====== SIDEBAR ======
with st.sidebar:
    st.header("⚙️ Panel")
    hs = st.selectbox("Hisse", list(HISSELER.keys()), index=list(HISSELER.keys()).index(secili_hisse))
    if hs != secili_hisse:
        qp_set("hisse", hs); st.rerun()

    st.metric(secili_hisse, f"{price:.2f}", f"{canli['change_pct']:+.2f}%" if canli else "")
    st.metric("Sampiyon", f"{champion.name} - {champion.value(price):,.0f}",
              f"{(champion.value(price)/champion.initial-1)*100:+.1f}%")

    st.divider()
    rs = st.slider("Yenile (sn)", 10, 300, refresh_sn)
    if rs != refresh_sn: qp_set("rs", rs); st.rerun()
    aw = st.toggle("Otomatik", value=auto_on)
    if aw != auto_on: qp_set("auto", "1" if aw else "0"); st.rerun()
    tw = st.toggle("Test Modu", value=test_mode)
    if tw != test_mode: qp_set("test", "1" if tw else "0"); st.rerun()

    st.divider()
    st.subheader("🔔 Alarm")
    af = st.number_input("Fiyat", value=float(round(price, 2)), step=0.5)
    ay = st.selectbox("Yon", ["ust", "alt"])
    if st.button("➕ Ekle", use_container_width=True):
        st.session_state.alarms.append({"hisse": secili_hisse, "fiyat": af, "yon": ay, "done": False})
        st.success("Eklendi")

    st.divider()
    st.write(f"🧬 Arena Gen: **{arena.generation}**")
    st.write(f"🔬 Meta Gen: **{meta.gen}**")
    if st.button("🔄 Sifirla", use_container_width=True):
        st.session_state.arena = Arena(100000)
        st.session_state.meta = MetaEvolver(8)
        st.session_state.log = []
        st.rerun()

# ====== UST ======
c1, c2, c3, c4 = st.columns(4)
c1.metric(secili_hisse, f"{price:.2f}")
c2.metric("RSI", f"{last['RSI']:.1f}")
c3.metric("Arena Gen", arena.generation)
c4.metric("Meta Gen", meta.gen)

if auto_msgs: st.info(" | ".join(auto_msgs[:4]))

# ====== SEKMELER ======
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "🏟️ Arena", "🧠 NN", "🧬 Genetik", "🔬 Meta", "💭 Yansima", "📜 Islemler", "📡 Log"
])

with t1:
    st.subheader("🏟️ Multi-Bot Arena")
    st.dataframe(pd.DataFrame(arena.leaderboard(price)), use_container_width=True, hide_index=True)
    st.divider()
    fig = go.Figure()
    for b in arena.bots:
        vals = [b.initial] + [t["value"] for t in b.trades] + [b.value(price)]
        fig.add_trace(go.Scatter(y=vals, name=b.name, mode="lines+markers"))
    fig.update_layout(height=400, template="plotly_dark", xaxis_title="Islem Sirasi", yaxis_title="TL")
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.subheader("🧠 Neural Network")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mimari", "20-24-12-3")
    c2.metric("Parametre", f"{champion.nn.n_params()}")
    c3.metric("Egitim Adimi", len(champion.nn.loss_history))
    if champion.nn.loss_history:
        st.line_chart(pd.DataFrame({"Loss": champion.nn.loss_history}))
    st.divider()
    st.subheader("Sampiyon Karar Olasiliklari")
    a, p, f = decisions[champion.name]
    c1, c2, c3 = st.columns(3)
    c1.metric("P(SAT)", f"%{p[0]*100:.1f}")
    c2.metric("P(TUT)", f"%{p[1]*100:.1f}")
    c3.metric("P(AL)", f"%{p[2]*100:.1f}")
    st.write(f"**Karar: {a}**")

with t3:
    st.subheader("🧬 Arena Genetik Evrimi")
    if arena.champion_history:
        st.dataframe(pd.DataFrame(arena.champion_history), use_container_width=True, hide_index=True)
    else:
        st.info("50 islemde bir evrim olur")

with t4:
    st.subheader("🔬 Meta-Evolution (Kendi DNA'si)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nesil", meta.gen)
    c2.metric("En Iyi Fitness", f"{max(meta.hist) if meta.hist else 0:+.2f}")
    c3.metric("Son Fitness", f"{meta.hist[-1] if meta.hist else 0:+.2f}")
    st.caption("100 islemde bir evrimlesir.")
    if meta.hall:
        rows = []
        for h in meta.hall:
            d = h["dna"]
            rows.append({
                "Nesil": h["gen"], "Fitness": f"{h['fit']:+.2f}",
                "Risk %": f"%{d['risk_pct']*100:.0f}",
                "SL%": f"{d['sl_pct']:.1f}", "TP%": f"{d['tp_pct']:.1f}",
                "BB": "✅" if d["use_bb"] else "❌",
                "ADX": "✅" if d["use_adx"] else "❌",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if meta.hist:
        st.line_chart(pd.DataFrame({"Fitness": meta.hist}))

with t5:
    st.subheader("💭 Botun Yansimalari")
    if meta.reflections:
        for r in reversed(meta.reflections):
            st.info(r)
    else:
        st.warning("Bot henuz ogrenmedi. 100 islem gerekiyor.")

with t6:
    st.subheader(f"📜 {champion.name} Islemleri")
    if champion.trades:
        tdf = pd.DataFrame(champion.trades)
        tdf["date"] = pd.to_datetime(tdf["date"]).dt.strftime("%Y-%m-%d %H:%M")
        cols = [c for c in ["date", "action", "price", "qty", "value", "pnl", "reason"] if c in tdf.columns]
        st.dataframe(tdf[cols], use_container_width=True, hide_index=True)
    else:
        st.info("Islem yok")

with t7:
    for line in reversed(st.session_state.log[-40:]):
        st.text(line)

st.divider()
st.caption(f"🕐 {now_tr().strftime('%Y-%m-%d %H:%M:%S')} - Borsa: {'ACIK' if market_open() else 'KAPALI'}")
