# ai_brain.py - THYAO AI v10
import numpy as np
import random

# ==================== BILGI BANKASI ====================
BILGI = {
    "Temel": {
        "Hisse Senedi": "Sirket ortakligi. Kar payi + deger artisi.",
        "BIST": "Borsa Istanbul, 1986.",
        "Endeks": "BIST100 en onemli.",
        "Tavan/Taban": "BIST'te gunluk %10.",
        "Volatilite": "Dalgalanma siddeti.",
        "Likidite": "Nakde cevrilme hizi.",
    },
    "Gostergeler": {
        "RSI": "70 ustu asiri alim, 30 alti asiri satim.",
        "MACD": "EMA12-EMA26. Signal EMA9.",
        "SMA": "Basit ortalama. 20/50/200.",
        "EMA": "Ustel ortalama, hizli tepki.",
        "Bollinger": "SMA20 +/- 2std.",
        "Stochastic": "%K %D. 80+ asiri alim.",
        "ATR": "Volatilite olcusu.",
        "ADX": "Trend gucu. 25+ guclu.",
        "OBV": "Hacim momentumu.",
        "CCI": "+100 asiri alim.",
    },
    "Formasyon": {
        "Bas-Omuz-Bas": "Dusus formu.",
        "Ikili Dip": "W, alis sinyali.",
        "Ikili Tepe": "M, satis sinyali.",
        "Ucgen": "Kirilim yonu onemli.",
        "Bayrak": "Trend devam.",
    },
    "Temel Analiz": {
        "F/K": "Dusuk = ucuz.",
        "PD/DD": "1 alti ucuz.",
        "ROE": "%15+ iyi.",
        "Temettu": "Pasif gelir.",
        "PEG": "1 alti ucuz.",
    },
    "Strateji": {
        "Trend": "Yukselen trendde al.",
        "Mean Reversion": "Asiri satimda al.",
        "Momentum": "Guclu gideni al.",
        "Breakout": "Kirilim stratejisi.",
        "Swing": "Gunler-haftalar.",
    },
    "Risk": {
        "Stop-Loss": "Zarar kesme. %2-5.",
        "Take-Profit": "Kar alma. SL'in 2-3 kati.",
        "Sharpe": "1+ iyi.",
        "Drawdown": "Zirveden dusus.",
    },
    "Psikoloji": {
        "FOMO": "Kacirma korkusu.",
        "Disiplin": "Planina sadik kal.",
        "Sabir": "Iyi firsat bekle.",
    },
    "BIST": {
        "Seans": "09:55-18:00.",
        "T+2": "Takas 2 gun.",
        "Temettu": "Nisan-Mayis.",
    },
}

def bilgi_ara(konu):
    konu = konu.lower()
    out = []
    for k, mad in BILGI.items():
        for b, i in mad.items():
            if konu in b.lower() or konu in i.lower():
                out.append({"Kategori": k, "Konu": b, "Aciklama": i})
    return out

# ==================== 25 USTA ====================
GURUS = [
    {"n": "Buffett", "f": lambda r: r["Close"] > r["SMA200"] and r["Volatility"] < 0.3},
    {"n": "Graham", "f": lambda r: r["Close"] < r["BB_dn"]},
    {"n": "Lynch", "f": lambda r: 50 < r["RSI"] < 70 and r["MACD_hist"] > 0},
    {"n": "Munger", "f": lambda r: r["SMA20"] > r["SMA50"] and r["ADX"] > 20},
    {"n": "Fisher", "f": lambda r: r["SMA50"] > r.get("SMA200", r["SMA50"]) * 0.95},
    {"n": "Templeton", "f": lambda r: r["RSI"] < 35},
    {"n": "Soros", "f": lambda r: r["MACD"] > r["MACD_sig"]},
    {"n": "Livermore", "f": lambda r: r["SMA20"] > r["SMA50"] and r["Close"] > r["SMA20"]},
    {"n": "TudorJones", "f": lambda r: r["Close"] > r["SMA50"]},
    {"n": "Dalio", "f": lambda r: r["Volatility"] < 0.35},
    {"n": "Druckenmiller", "f": lambda r: r["ADX"] > 25 and r["MACD_hist"] > 0},
    {"n": "ONeil", "f": lambda r: r["RSI"] > 60 and r["Vol_ratio"] > 1.2},
    {"n": "Minervini", "f": lambda r: r.get("BB_width", 1) < 0.08 and r["ADX"] > 20},
    {"n": "Darvas", "f": lambda r: r["Close"] > r["BB_up"] * 0.98},
    {"n": "Simons", "f": lambda r: r["RSI"] < 30},
    {"n": "Burry", "f": lambda r: r["RSI"] < 25 and r["WILLR"] < -85},
    {"n": "Icahn", "f": lambda r: r["Close"] < r["BB_mid"] and r["ADX"] < 20},
    {"n": "Ackman", "f": lambda r: r["SMA20"] > r["SMA50"] and r["Volatility"] < 0.4},
    {"n": "Klarman", "f": lambda r: r["Close"] < r["BB_dn"] * 1.02},
    {"n": "Marks", "f": lambda r: r["RSI"] < 40 and r["MACD_hist"] > -0.5},
    {"n": "Tepper", "f": lambda r: r["RSI"] < 35 and r["BB_dn"] > r["Close"] * 0.95},
    {"n": "Griffin", "f": lambda r: r["Vol_ratio"] > 1.0 and abs(r["MACD_hist"]) > 0.1},
    {"n": "Cohen", "f": lambda r: r["RSI"] > 55 and r["Vol_ratio"] > 1.3},
    {"n": "LWilliams", "f": lambda r: r["K"] < 20 and r["ATR"] > 0},
    {"n": "Seykota", "f": lambda r: r["SMA20"] > r["SMA50"] and r["ADX"] > 22},
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

# ==================== FEATURES ====================
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

# ==================== NEURAL NET (Adam + Dropout) ====================
class NeuralNet:
    def __init__(self, n_in=20, h1=32, h2=16, n_out=3, lr=0.005, dropout=0.15):
        self.W1 = np.random.randn(n_in, h1) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, n_out) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(n_out)
        self.lr = lr
        self.b1a, self.b2a, self.eps = 0.9, 0.999, 1e-8
        self.t = 0
        self.m = {k: np.zeros_like(getattr(self, k)) for k in ["W1","b1","W2","b2","W3","b3"]}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in ["W1","b1","W2","b2","W3","b3"]}
        self.dropout = dropout
        self.loss_history = []

    def _relu(self, x): return np.maximum(0, x)
    def _relu_d(self, x): return (x > 0).astype(float)
    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

    def forward(self, x, training=False):
        z1 = x @ self.W1 + self.b1
        a1 = self._relu(z1)
        if training and self.dropout > 0:
            m = (np.random.random(a1.shape) > self.dropout).astype(float)
            a1 = a1 * m / (1 - self.dropout)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        if training and self.dropout > 0:
            m = (np.random.random(a2.shape) > self.dropout).astype(float)
            a2 = a2 * m / (1 - self.dropout)
        z3 = a2 @ self.W3 + self.b3
        return self._softmax(z3), (x, z1, a1, z2, a2, z3)

    def predict(self, x):
        if x.ndim == 1: x = x.reshape(1, -1)
        p, _ = self.forward(x, training=False)
        return p[0]

    def train_step(self, X, y):
        n = X.shape[0]
        probs, (x, z1, a1, z2, a2, z3) = self.forward(X, training=True)
        y1h = np.zeros_like(probs); y1h[np.arange(n), y] = 1
        loss = -np.mean(np.sum(y1h * np.log(probs + 1e-9), axis=1))
        self.loss_history.append(float(loss))
        if len(self.loss_history) > 150: self.loss_history = self.loss_history[-150:]
        dz3 = (probs - y1h) / n
        dW3 = a2.T @ dz3; db3 = dz3.sum(0)
        da2 = dz3 @ self.W3.T; dz2 = da2 * self._relu_d(z2)
        dW2 = a1.T @ dz2; db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T; dz1 = da1 * self._relu_d(z1)
        dW1 = x.T @ dz1; db1 = dz1.sum(0)
        self.t += 1
        for k, g in {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}.items():
            self.m[k] = self.b1a * self.m[k] + (1 - self.b1a) * g
            self.v[k] = self.b2a * self.v[k] + (1 - self.b2a) * (g * g)
            mh = self.m[k] / (1 - self.b1a ** self.t)
            vh = self.v[k] / (1 - self.b2a ** self.t)
            setattr(self, k, getattr(self, k) - self.lr * mh / (np.sqrt(vh) + self.eps))
        return loss

    def mutate(self, rate=0.05, scale=0.1):
        for W in [self.W1, self.W2, self.W3]:
            m = np.random.random(W.shape) < rate
            W += m * np.random.randn(*W.shape) * scale

    def clone(self):
        n = NeuralNet(self.W1.shape[0], self.W1.shape[1], self.W2.shape[1], self.W3.shape[1], self.lr, self.dropout)
        for k in ["W1","b1","W2","b2","W3","b3"]:
            setattr(n, k, getattr(self, k).copy())
        return n

    def crossover(self, other):
        n = self.clone()
        for k in ["W1","b1","W2","b2","W3","b3"]:
            A = getattr(self, k); B = getattr(other, k)
            m = np.random.random(A.shape) < 0.5
            setattr(n, k, np.where(m, A, B))
        return n

    def n_params(self):
        return sum(w.size for w in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ["W1","b1","W2","b2","W3","b3"]}

    def from_dict(self, d):
        for k in ["W1","b1","W2","b2","W3","b3"]:
            setattr(self, k, np.array(d[k]))

# ==================== PRIORITIZED REPLAY ====================
class PrioritizedReplay:
    def __init__(self, cap=1000, alpha=0.6):
        self.cap = cap; self.alpha = alpha; self.buf = []
    def add(self, feat, label, priority=1.0):
        self.buf.append((feat, label, priority))
        if len(self.buf) > self.cap: self.buf = self.buf[-self.cap:]
    def sample(self, bs=32):
        if len(self.buf) < bs: return None
        prios = np.array([b[2] ** self.alpha for b in self.buf])
        probs = prios / prios.sum()
        idx = np.random.choice(len(self.buf), bs, p=probs, replace=False)
        b = [self.buf[i] for i in idx]
        return np.array([x[0] for x in b]), np.array([x[1] for x in b])
    def size(self): return len(self.buf)

# ==================== ENSEMBLE ====================
class EnsemblePredictor:
    def __init__(self):
        self.models = {
            "trend": {"w": 0.25, "h": []}, "momentum": {"w": 0.20, "h": []},
            "mean_rev": {"w": 0.15, "h": []}, "sma_cross": {"w": 0.15, "h": []},
            "volatility": {"w": 0.15, "h": []}, "seasonal": {"w": 0.10, "h": []},
        }
    def predict(self, df, gun=5):
        if len(df) < 60: return None
        last = df.iloc[-1]; c = last["Close"]; preds = {}
        y = df["Close"].tail(60).values; x = np.arange(len(y))
        z = np.polyfit(x, y, 1); egim = z[0]
        preds["trend"] = c + egim * gun
        mom = 0.0
        if last["RSI"] > 60: mom += 0.4
        elif last["RSI"] < 40: mom -= 0.4
        if last["MACD_hist"] > 0: mom += 0.3
        else: mom -= 0.3
        if last["Vol_ratio"] > 1.2: mom += 0.3 * (1 if mom > 0 else -1)
        preds["momentum"] = c * (1 + mom * 0.03 * gun / 5)
        bb_pos = (c - last["BB_dn"]) / (last["BB_up"] - last["BB_dn"]) if (last["BB_up"] - last["BB_dn"]) > 0 else 0.5
        mr_k = 0.5 if bb_pos < 0.2 else (-0.5 if bb_pos > 0.8 else 0)
        preds["mean_rev"] = c * (1 + mr_k * 0.04 * gun / 5)
        cross = 1 if last["SMA20"] > last["SMA50"] else -1
        guc = min(1.0, abs(last["SMA20"] - last["SMA50"]) / c * 20)
        preds["sma_cross"] = c * (1 + cross * guc * 0.03 * gun / 5)
        vol = last["Volatility"]
        if vol > 0.5: preds["volatility"] = c * (1 - 0.015 * gun / 5)
        elif vol < 0.2: preds["volatility"] = c * (1 + 0.01 * gun / 5)
        else: preds["volatility"] = c
        son5 = df["Close"].tail(5).mean()
        preds["seasonal"] = c * (1 + (c - son5) / son5 * 0.3)
        tw = sum(m["w"] for m in self.models.values())
        ensemble = sum(preds[k] * self.models[k]["w"] for k in preds) / tw if tw > 0 else c
        return ensemble, preds, egim
    def update(self, gercek, preds):
        for name, tahmin in preds.items():
            if tahmin and gercek:
                acc = max(0.0, 1 - abs(tahmin - gercek) / gercek * 10)
                self.models[name]["h"].append(acc)
                son = self.models[name]["h"][-20:]
                self.models[name]["w"] = max(0.05, float(np.mean(son))) if son else 0.15
        tw = sum(m["w"] for m in self.models.values())
        if tw > 0:
            for m in self.models.values(): m["w"] /= tw
    def stats(self):
        out = []
        for n, m in self.models.items():
            h = m["h"][-20:]
            out.append({"Model": n.upper(), "Agirlik": round(m["w"], 3),
                        "Dogruluk": round(float(np.mean(h)) * 100 if h else 0, 1),
                        "Sayi": len(m["h"])})
        return out

# ==================== PREDICTION TRACKER ====================
class PredictionTracker:
    def __init__(self): self.predictions = []; self.counter = 0
    def add(self, tarih, hedef, tahmin, giris):
        self.counter += 1
        self.predictions.append({"id": self.counter, "tarih": tarih, "hedef": hedef,
                                  "tahmin": float(tahmin), "giris": float(giris),
                                  "gercek": None, "hata": None, "yon_ok": None,
                                  "dogrulandi": False,
                                  "yon": "UP" if tahmin > giris else "DOWN"})
    def dogrula(self, bugun, fiyat):
        for p in self.predictions:
            if p["dogrulandi"]: continue
            if p["hedef"] <= bugun:
                p["gercek"] = float(fiyat)
                p["hata"] = abs(p["tahmin"] - fiyat) / fiyat * 100
                p["yon_ok"] = (("UP" if fiyat > p["giris"] else "DOWN") == p["yon"])
                p["dogrulandi"] = True
    def stats(self):
        v = [p for p in self.predictions if p["dogrulandi"]]
        if not v:
            return {"toplam": len(self.predictions), "dogrulanan": 0, "mape": 0,
                    "yon": 0, "gercekcilik": 0, "guven": "YETERSIZ",
                    "yorum": "Dogrulanmis tahmin yok."}
        mape = float(np.mean([p["hata"] for p in v]))
        yon = sum(1 for p in v if p["yon_ok"]) / len(v) * 100
        ger = max(0.0, min(100.0, 100 - mape * 5))
        if len(v) < 5: g, y = "YETERSIZ", f"{len(v)} tahmin, 5 gerekli."
        elif mape < 3: g, y = "COK YUKSEK", "Guvenilir."
        elif mape < 7: g, y = "YUKSEK", "Tutarli."
        elif mape < 15: g, y = "ORTA", "Dikkatli."
        elif mape < 25: g, y = "DUSUK", "Zayif."
        else: g, y = "COK DUSUK", "Guvenilmez."
        return {"toplam": len(self.predictions), "dogrulanan": len(v),
                "mape": round(mape, 2), "yon": round(yon, 1),
                "gercekcilik": round(ger, 1), "guven": g, "yorum": y}

# ==================== META EVOLVER ====================
class MetaEvolver:
    def __init__(self, n=10):
        self.n = n; self.pop = [self._rand() for _ in range(n)]
        self.hall = []; self.gen = 0; self.hist = []; self.reflections = []
    def _rand(self):
        return {"risk_pct": float(np.random.uniform(0.1, 0.4)),
                "min_conf": float(np.random.uniform(0.15, 0.45)),
                "sl_pct": float(np.random.uniform(1.5, 5.0)),
                "tp_pct": float(np.random.uniform(3.0, 12.0)),
                "use_bb": bool(np.random.choice([True, False])),
                "use_adx": bool(np.random.choice([True, False])),
                "use_stoch": bool(np.random.choice([True, False]))}
    def _mut(self, dna, rate=0.3):
        new = dict(dna)
        for k, v in new.items():
            if np.random.random() < rate:
                if isinstance(v, bool): new[k] = not v
                elif isinstance(v, float):
                    new[k] = v * float(np.random.uniform(0.85, 1.15))
                    if k == "risk_pct": new[k] = float(np.clip(new[k], 0.05, 0.5))
                    if k == "min_conf": new[k] = float(np.clip(new[k], 0.1, 0.5))
                    if k == "sl_pct": new[k] = float(np.clip(new[k], 1.0, 8.0))
                    if k == "tp_pct": new[k] = float(np.clip(new[k], 2.0, 20.0))
        return new
    def _cross(self, a, b):
        return {k: (a[k] if np.random.random() < 0.5 else b[k]) for k in a}
    def _bt(self, dna, df):
        if len(df) < 60: return -999
        cash = 100000.0; shares = 0; entry = None
        wins = losses = 0; peak = cash; mdd = 0
        for i in range(1, len(df)):
            row = df.iloc[i]; p = float(row["Close"])
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
            if sig >= 2 and shares == 0 and cash > p * 10:
                q = int((cash * dna["risk_pct"]) / p)
                if q > 0: cash -= q * p; shares = q; entry = p
            elif shares > 0 and entry:
                pct = (p / entry - 1) * 100
                if sig <= -2 or pct <= -dna["sl_pct"] or pct >= dna["tp_pct"]:
                    cash += shares * p
                    wins += 1 if p > entry else 0
                    losses += 1 if p <= entry else 0
                    shares = 0; entry = None
            val = cash + shares * p
            peak = max(peak, val); mdd = max(mdd, (peak - val) / peak * 100)
        final = cash + shares * float(df["Close"].iloc[-1])
        ret = (final / 100000.0 - 1) * 100
        t = wins + losses; wr = wins / t if t > 0 else 0
        return ret + wr * 20 - mdd * 0.5
    def evolve(self, df):
        sc = [(self._bt(d, df), d) for d in self.pop]
        sc.sort(key=lambda x: x[0], reverse=True)
        self.gen += 1
        bf, bd = sc[0]
        self.hist.append(bf)
        self.hall.append({"gen": self.gen, "fit": round(bf, 2), "dna": dict(bd)})
        self.hall = sorted(self.hall, key=lambda x: x["fit"], reverse=True)[:8]
        np_new = [sc[0][1], sc[1][1]]
        while len(np_new) < self.n:
            if np.random.random() < 0.6:
                a = sc[np.random.randint(0, min(4, len(sc)))][1]
                b = sc[np.random.randint(0, min(4, len(sc)))][1]
                np_new.append(self._mut(self._cross(a, b), 0.2))
            else:
                np_new.append(self._mut(sc[0][1], 0.4))
        self.pop = np_new
        refl = self._reflect(bd)
        if refl: self.reflections.append(refl)
        self.reflections = self.reflections[-20:]
        return bd, bf
    def _reflect(self, dna):
        parts = ["BB aktif" if dna["use_bb"] else "BB kapali",
                 "ADX filtre" if dna["use_adx"] else "ADX yok",
                 "Stoch kullaniyorum" if dna.get("use_stoch") else "Stoch kapali",
                 f"SL %{dna['sl_pct']:.1f}", f"TP %{dna['tp_pct']:.1f}",
                 f"Risk %{dna['risk_pct']*100:.0f}"]
        trend = ""
        if len(self.hist) >= 2: trend = "📈" if self.hist[-1] > self.hist[-2] else "📉"
        return f"Gen {self.gen}: {trend} " + " | ".join(parts)

# ==================== SELF DATA ====================
class SelfDataGenerator:
    def __init__(self): self.data = []
    def generate(self, df, n=80, seed=42):
        if len(df) < 100: return []
        np.random.seed(seed)
        out = []
        for _ in range(n):
            i = np.random.randint(60, len(df) - 10)
            w = df.iloc[i-60:i]; f = df.iloc[i:i+10]
            if len(w) < 30 or len(f) < 5: continue
            e = float(w["Close"].iloc[-1]); x = float(f["Close"].iloc[-1])
            ret = (x / e - 1) * 100
            lab = 2 if ret > 2 else (0 if ret < -2 else 1)
            out.append({"feat": extract_features(w.iloc[-1]), "label": lab, "ret": ret})
        self.data = out
        return out
    def summary(self):
        if not self.data:
            return {"total": 0, "AL": 0, "SAT": 0, "TUT": 0, "avg_ret": 0}
        l = [d["label"] for d in self.data]
        return {"total": len(self.data), "AL": l.count(2), "SAT": l.count(0),
                "TUT": l.count(1), "avg_ret": round(float(np.mean([d["ret"] for d in self.data])), 2)}

# ==================== BOT ====================
class Bot:
    def __init__(self, name, cash=100000.0, strategy="balanced"):
        self.name = name; self.cash = cash; self.shares = 0
        self.initial = cash; self.trades = []
        self.nn = NeuralNet(); self.replay = PrioritizedReplay(1000)
        self.strategy = strategy; self.wins = 0; self.losses = 0
        self.total_pnl = 0.0; self.entry_price = None
        self.entry_features = None; self.last_price = cash
        self.risk_pct = 0.25; self.min_conf = 0.25
        self.sl_pct = 3.0; self.tp_pct = 6.0
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
        if self.shares > 0 and self.entry_price:
            pct = (price / self.entry_price - 1) * 100
            if pct <= -self.sl_pct: action = "SAT"; reason = f"STOP-LOSS ({pct:+.1f}%)"
            elif pct >= self.tp_pct: action = "SAT"; reason = f"TAKE-PROFIT ({pct:+.1f}%)"
        if action == "AL" and self.cash > price * 10:
            q = int((self.cash * self.risk_pct) / price)
            if q < 1: return False
            self.cash -= q * price; self.shares += q
            self.entry_price = price; self.entry_features = feat
            self.trades.append({"date": date, "action": "AL", "price": price, "qty": q,
                                "value": self.value(price), "reason": reason})
            return True
        if action == "SAT" and self.shares > 0:
            q = self.shares
            pnl = (price - self.entry_price) * q if self.entry_price else 0
            self.cash += q * price; self.shares = 0; self.total_pnl += pnl
            if pnl > 0: self.wins += 1
            else: self.losses += 1
            if self.entry_features is not None:
                lab = 2 if pnl > 0 else 0
                prio = abs(pnl) / 100 + 1.0
                self.replay.add(self.entry_features, lab, prio)
            self.trades.append({"date": date, "action": "SAT", "price": price, "qty": q,
                                "value": self.value(price), "reason": reason, "pnl": pnl})
            self.entry_price = None; self.entry_features = None
            return True
        return False
    def train(self, epochs=3, bs=32):
        if self.replay.size() < bs: return None
        ls = []
        for _ in range(epochs):
            s = self.replay.sample(bs)
            if s is None: break
            X, y = s
            ls.append(self.nn.train_step(X, y))
        return float(np.mean(ls)) if ls else None
    def win_rate(self):
        t = self.wins + self.losses
        return self.wins / t if t > 0 else 0.0
    def sharpe(self):
        pnls = [t.get("pnl", 0) for t in self.trades if t["action"] == "SAT"]
        if len(pnls) < 2: return 0.0
        return float(np.mean(pnls) / (np.std(pnls) + 1e-9))
    def max_dd(self):
        if not self.trades: return 0.0
        v = [self.initial] + [t["value"] for t in self.trades]
        peak = v[0]; dd = 0
        for x in v:
            peak = max(peak, x); dd = max(dd, (peak - x) / peak * 100)
        return dd

# ==================== ARENA ====================
class Arena:
    def __init__(self, cash=100000.0):
        self.bots = [Bot("Alpha", cash, "aggressive"), Bot("Beta", cash, "balanced"),
                     Bot("Gamma", cash, "conservative"), Bot("Delta", cash, "balanced"),
                     Bot("Epsilon", cash, "aggressive"), Bot("Zeta", cash, "conservative")]
        self.generation = 1; self.history = []; self.counter = 0
    def best(self, price):
        for b in self.bots: b.last_price = price
        return max(self.bots, key=lambda b: b.value(price))
    def evolve(self, price):
        for b in self.bots: b.last_price = price
        sb = sorted(self.bots, key=lambda b: b.value(price), reverse=True)
        b1, b2 = sb[0], sb[1]; w1, w2 = sb[-1], sb[-2]
        c1 = b1.nn.crossover(b2.nn); c1.mutate(0.05, 0.1)
        c2 = b2.nn.crossover(b1.nn); c2.mutate(0.05, 0.1)
        for bot, nn, nm in [(w1, c1, f"G{self.generation+1}a"), (w2, c2, f"G{self.generation+1}b")]:
            bot.cash = 100000.0; bot.shares = 0; bot.initial = 100000.0
            bot.total_pnl = 0.0; bot.wins = 0; bot.losses = 0; bot.trades = []
            bot.nn = nn; bot.replay = PrioritizedReplay(1000); bot.name = nm
        self.generation += 1
        self.history.append({"Gen": self.generation-1, "Sampiyon": b1.name,
                              "Deger": round(b1.value(price), 0)})
    def leaderboard(self, price):
        for b in self.bots: b.last_price = price
        rows = []
        for b in self.bots:
            v = b.value(price)
            rows.append({"Bot": b.name, "Strateji": b.strategy,
                         "Portfoy": f"{v:,.0f}",
                         "Getiri%": f"{(v/b.initial-1)*100:+.2f}",
                         "Islem": len(b.trades),
                         "Kazanma%": f"%{b.win_rate()*100:.0f}",
                         "Sharpe": f"{b.sharpe():.2f}",
                         "MaxDD%": f"{b.max_dd():.1f}",
                         "Hafiza": b.replay.size()})
        return sorted(rows, key=lambda x: float(x["Portfoy"].replace(",", "")), reverse=True)
