#!/usr/bin/env python3
"""
bot.py v25 - GitHub Actions otomatik trader
Telefon kapalı olsa bile 20 hisse + NN + walk-forward + meta-evrim
"""
import os, re, json, math, time, pickle, sqlite3
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    os.system("pip install -q yfinance")
    import yfinance as yf

BOT_VERSION = 25
HORIZON = 10
TB_K = 1.2
FEE = 0.0015
SLIPPAGE = 0.001
RT_COST = 2 * (FEE + SLIPPAGE)
N_FEAT_TECH = 34
N_STOCKS = 20
N_FEAT = N_FEAT_TECH + N_STOCKS
BAG_N = 5
BAG_SEED = 100
DB_FILE = "bot_v25.db"
STATE_FILE = "state_v25.pkl"
TR = ZoneInfo("Europe/Istanbul")

HISSELER = {
    "THYAO": "THYAO.IS", "GARAN": "GARAN.IS", "ASELS": "ASELS.IS",
    "AKBNK": "AKBNK.IS", "EREGL": "EREGL.IS", "TUPRS": "TUPRS.IS",
    "SISE": "SISE.IS", "KCHOL": "KCHOL.IS", "PGSUS": "PGSUS.IS",
    "TCELL": "TCELL.IS", "TTKOM": "TTKOM.IS", "BIMAS": "BIMAS.IS",
    "MGROS": "MGROS.IS", "ENKAI": "ENKAI.IS", "TAVHL": "TAVHL.IS",
    "ULKER": "ULKER.IS", "CCOLA": "CCOLA.IS", "AKSEN": "AKSEN.IS",
    "FROTO": "FROTO.IS", "TOASO": "TOASO.IS",
}
STOCK_LIST = list(HISSELER.keys())
CROSS_SYMBOLS = {"USDTRY": "USDTRY=X", "XU100": "XU100.IS", "XAUUSD": "GC=F"}

DEFAULT_RISK = {
    "risk_per_trade": 0.01, "sl_atr": 2.5, "trail_act_r": 1.5, "trail_atr": 2.5,
    "p_buy": 0.45, "p_sell": 0.45, "margin": 0.10,
    "max_pos": 0.15, "use_regime": True, "max_hold": 120, "max_dd": 15.0,
    "daily_loss": 3.0, "max_trades_day": 5, "loss_streak": 3, "cooldown_h": 24,
    "gate_on": True, "gate_min_n": 8, "auto_adopt": True, "unc_max": 0.25,
    "conf_sizing": True, "regime_bull_adj": -0.03, "regime_bear_adj": 0.08,
    "regime_range_adj": 0.03, "regime_vol_adj": 0.10,
    "max_positions": 8, "max_exposure": 0.85,
}

REGIMES = ["BULL", "BEAR", "VOL", "RANGE"]


def now_tr():
    return datetime.now(TR)


def borsa_acik():
    n = now_tr()
    return n.weekday() < 5 and dtime(9, 55) <= n.time() <= dtime(18, 10)


def log(msg):
    print(f"{now_tr().isoformat(timespec='seconds')} | {msg}")


def safe_float(x, d=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else d
    except (TypeError, ValueError):
        return d


# ============================================================
# DB
# ============================================================
def db_conn():
    return sqlite3.connect(DB_FILE, timeout=15.0)


def db_init():
    with db_conn() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, stock TEXT, action TEXT, price REAL, qty INTEGER, pnl REAL, reason TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS equity (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, value REAL, cash REAL, npos INTEGER)")
        c.execute("CREATE TABLE IF NOT EXISTS train (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, step INTEGER, vl REAL, va REAL, note TEXT)")


def db_add_trade(s, a, p, q, pnl, r):
    try:
        with db_conn() as c:
            c.execute("INSERT INTO trades (ts,stock,action,price,qty,pnl,reason) VALUES (?,?,?,?,?,?,?)",
                      (now_tr().isoformat(timespec="seconds"), s, a, float(p), int(q), float(pnl or 0), str(r)[:300]))
    except Exception:
        pass


def db_eq_add(v, cash, npos):
    try:
        with db_conn() as c:
            c.execute("INSERT INTO equity (ts,value,cash,npos) VALUES (?,?,?,?)",
                      (now_tr().isoformat(timespec="seconds"), float(v), float(cash), int(npos)))
    except Exception:
        pass


def db_log_train(step, vl, va, note=""):
    try:
        with db_conn() as c:
            c.execute("INSERT INTO train (ts,step,vl,va,note) VALUES (?,?,?,?,?)",
                      (now_tr().isoformat(timespec="seconds"), int(step), float(vl), float(va), note))
    except Exception:
        pass


db_init()


# ============================================================
# GOSTERGELER
# ============================================================
def add_ind(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    d = c.diff()
    g = d.clip(lower=0).rolling(14).mean()
    ls = (-d.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["MACDs"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDh"] = df["MACD"] - df["MACDs"]
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["EMA21"] = c.ewm(span=21, adjust=False).mean()
    df["BBm"] = df["SMA20"]
    df["BBs"] = c.rolling(20).std()
    df["BBu"] = df["BBm"] + 2 * df["BBs"]
    df["BBd"] = df["BBm"] - 2 * df["BBs"]
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    rng = (hi14 - lo14).replace(0, np.nan)
    df["K"] = 100 * (c - lo14) / rng
    df["D"] = df["K"].rolling(3).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATRp"] = df["ATR"] / c
    df["WILLR"] = -100 * (hi14 - c) / rng
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr14 = tr.rolling(14).sum().replace(0, np.nan)
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(14).sum() / tr14
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(14).sum() / tr14
    df["ADX"] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).rolling(14).mean()
    df["Vol_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
    for k in (1, 5, 10, 20, 30):
        df[f"Ret{k}"] = c.pct_change(k)
    df["Volatility"] = df["Ret1"].rolling(20).std() * np.sqrt(252)
    df["HL"] = (h - l) / c
    df["Gap"] = (df["Open"] - c.shift()) / c.shift()
    df["HI120"] = c / c.rolling(120).max() - 1
    s = df.set_index("Date")["Close"]
    w = s.resample("W-FRI").last().dropna()
    wd = w.diff()
    wg = wd.clip(lower=0).rolling(14).mean()
    wl = (-wd.clip(upper=0)).rolling(14).mean()
    wk = pd.DataFrame({"WDate": pd.DatetimeIndex(w.index).astype("datetime64[ns]")})
    wk["W_RSI"] = (100 - 100 / (1 + wg / wl.replace(0, np.nan))).to_numpy()
    we9 = w.ewm(span=9, adjust=False).mean()
    we21 = w.ewm(span=21, adjust=False).mean()
    wk["W_Trend"] = ((we9 - we21) / we21.replace(0, np.nan)).to_numpy()
    wsma = w.rolling(20).mean()
    wk["W_Strength"] = ((w - wsma) / wsma.replace(0, np.nan)).to_numpy()
    wk = wk.replace([np.inf, -np.inf], np.nan).dropna()
    df = pd.merge_asof(df, wk, left_on="Date", right_on="WDate", direction="backward", allow_exact_matches=False).drop(columns=["WDate"])
    tu = (c > df["SMA200"]) & (df["SMA50"] > df["SMA200"])
    td = (c < df["SMA200"]) & (df["SMA50"] < df["SMA200"])
    vh = df["Volatility"] > df["Volatility"].rolling(60).quantile(0.75)
    adx = df["ADX"] > 25
    df["Regime_Bull"] = (tu & ~vh & adx).astype(int)
    df["Regime_Bear"] = (td & ~vh & adx).astype(int)
    df["Regime_Vol"] = vh.astype(int)
    df["Regime_Range"] = (~tu & ~td & ~vh).astype(int)
    for cc, wnd in [("Regime_Bull", 10), ("Regime_Bear", 10), ("Regime_Vol", 5), ("Regime_Range", 10)]:
        df[cc + "_sm"] = df[cc].rolling(wnd, min_periods=1).mean()
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def add_cross(df, cross):
    df = df.copy()
    idx = pd.DatetimeIndex(df["Date"])
    for name in CROSS_SYMBOLS:
        s = (cross or {}).get(name)
        if s is None or len(s) < 30:
            df[f"{name}_Ret5"] = 0.0
            df[f"{name}_Corr20"] = 0.0
            continue
        s = s[~s.index.duplicated()].sort_index()
        al = pd.Series(s.reindex(idx, method="ffill").to_numpy(), index=df.index)
        if name in ("USDTRY", "XAUUSD"):
            al = al.shift(1)
        df[f"{name}_Ret5"] = al.pct_change(5).fillna(0.0)
        df[f"{name}_Corr20"] = df["Ret1"].rolling(20).corr(al.pct_change()).fillna(0.0)
    return df.replace([np.inf, -np.inf], 0.0)


# ============================================================
# FEATURES
# ============================================================
def feature_matrix(df, stock_name=None):
    n = len(df)
    g = lambda c, d=0.0: df[c].to_numpy(float) if c in df.columns else np.full(n, d)
    div = lambda a, b: np.divide(a, b, out=np.zeros(n), where=(b != 0) & np.isfinite(b))
    cl = np.clip
    c = g("Close")
    sma20, sma50, sma200 = g("SMA20"), g("SMA50"), g("SMA200")
    bbm, bbu, bbd = g("BBm"), g("BBu"), g("BBd")
    f = [
        g("RSI", 50) / 100,
        cl(div(g("MACDh"), c) * 200, -1, 1),
        cl((div(sma20, sma50) - 1) * 10, -1, 1),
        cl(div(c - bbm, bbu - bbd + 1e-9), -1, 1),
        g("K", 50) / 100, cl(g("ADX") / 50, 0, 1),
        cl(g("Volatility") * 3, 0, 1), cl(g("Vol_ratio", 1) - 1, -1, 1),
        cl(g("Ret10") * 10, -1, 1), cl(g("HL") * 20, 0, 1),
        cl(g("Gap") * 20, -1, 1), cl((div(c, sma200) - 1) * 5, -1, 1),
        cl(div(g("EMA9") - g("EMA21"), c) * 20, -1, 1),
        (c > sma20).astype(float), (c > sma50).astype(float),
        (g("MACD") > g("MACDs")).astype(float), (sma20 > sma50).astype(float),
        cl(g("WILLR", -50) / 100, -1, 0), cl(g("Ret30") * 10, -1, 1),
        (c > bbm).astype(float), cl(g("ATRp") * 20, 0, 1),
        cl(g("Ret5") * 10, -1, 1), cl(g("HI120") * 5, -1, 0),
        cl((g("K", 50) - g("D", 50)) / 30, -1, 1),
        cl(g("W_RSI", 50) / 100, 0, 1), cl(g("W_Trend") * 10, -1, 1), cl(g("W_Strength") * 5, -1, 1),
        g("Regime_Bull_sm"), g("Regime_Bear_sm"), g("Regime_Vol_sm"), g("Regime_Range_sm"),
        cl(g("USDTRY_Ret5") * 20, -1, 1), cl(g("XU100_Ret5") * 10, -1, 1), cl(g("XU100_Corr20"), -1, 1),
    ]
    M = np.nan_to_num(np.column_stack(f))
    oh = np.zeros((n, N_STOCKS))
    if stock_name in STOCK_LIST:
        oh[:, STOCK_LIST.index(stock_name)] = 1.0
    return np.hstack([M, oh])


def tb_label(hi, lo, cl, atr, i, horizon=HORIZON, k=TB_K):
    e, a = cl[i], atr[i]
    if not np.isfinite(a) or a <= 0:
        return 1, 0.0
    up, dn = e + k * a, e - k * a
    end = min(i + horizon, len(cl) - 1)
    ret = cl[end] / e - 1
    for j in range(i + 1, end + 1):
        if lo[j] <= dn:
            return 0, ret
        if hi[j] >= up:
            return 2, ret
    return 1, ret


def make_ds(df, stock=None):
    n = len(df)
    if n < HORIZON + 150:
        return None
    F = feature_matrix(df, stock)
    hi, lo = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    cl, atr = df["Close"].to_numpy(float), df["ATR"].to_numpy(float)
    m = n - HORIZON
    y = np.ones(m, dtype=int)
    f = np.zeros(m)
    for i in range(m):
        y[i], f[i] = tb_label(hi, lo, cl, atr, i)
    d = df["Date"].to_numpy()
    return {"F": F, "X": F[:m], "y": y, "f": f, "d": d[:m], "dates": d}


def merge_ds(lst):
    lst = [d for d in lst if d]
    if not lst:
        return None
    return {"X": np.vstack([d["X"] for d in lst]), "y": np.concatenate([d["y"] for d in lst]),
            "f": np.concatenate([d["f"] for d in lst]), "d": np.concatenate([d["d"] for d in lst])}


def split_ds(ds, val_frac=0.2, purge=HORIZON):
    d = ds["d"]
    ud = np.unique(d)
    if len(ud) < 60:
        return None
    cut = ud[int(len(ud) * (1 - val_frac))]
    gap = np.timedelta64(int(purge * 1.5) + 1, "D")
    tr, va = d < (cut - gap), d >= cut
    if tr.sum() < 200 or va.sum() < 40:
        return None
    return {"Xtr": ds["X"][tr], "ytr": ds["y"][tr], "ftr": ds["f"][tr],
            "Xva": ds["X"][va], "yva": ds["y"][va], "fva": ds["f"][va]}


# ============================================================
# NN
# ============================================================
_KEYS = ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "Att"]


class NN:
    def __init__(self, n_in=N_FEAT, h1=64, h2=48, h3=32, n_out=3, lr=0.003, dropout=0.20, wd=0.015, seed=None):
        rg = np.random.default_rng(seed)
        self.W1 = rg.standard_normal((n_in, h1)) * np.sqrt(2.0 / n_in); self.b1 = np.zeros(h1)
        self.W2 = rg.standard_normal((h1, h2)) * np.sqrt(2.0 / h1); self.b2 = np.zeros(h2)
        self.W3 = rg.standard_normal((h2, h3)) * np.sqrt(2.0 / h2); self.b3 = np.zeros(h3)
        self.W4 = rg.standard_normal((h3, n_out)) * np.sqrt(2.0 / h3); self.b4 = np.zeros(n_out)
        self.Att = np.ones(n_in)
        self.lr, self.dropout, self.wd = lr, dropout, wd
        self.eps, self.t, self.T = 1e-8, 0, 1.0
        self.m = {k: np.zeros_like(getattr(self, k)) for k in _KEYS}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in _KEYS}

    @staticmethod
    def _sm(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

    def _scale(self):
        a = np.abs(self.Att)
        return a / (a.sum() + 1e-9) * len(a)

    def forward(self, x, training=False):
        xa = x * self._scale()
        z1 = xa @ self.W1 + self.b1; a1 = np.maximum(0, z1)
        if training and self.dropout > 0:
            a1 = a1 * (np.random.random(a1.shape) > self.dropout) / (1 - self.dropout)
        z2 = a1 @ self.W2 + self.b2; a2 = np.maximum(0, z2)
        if training and self.dropout > 0:
            a2 = a2 * (np.random.random(a2.shape) > self.dropout) / (1 - self.dropout)
        z3 = a2 @ self.W3 + self.b3; a3 = np.maximum(0, z3)
        if training and self.dropout > 0:
            a3 = a3 * (np.random.random(a3.shape) > self.dropout) / (1 - self.dropout)
        z4 = a3 @ self.W4 + self.b4
        zt = z4 if training else z4 / max(self.T, 1e-3)
        return self._sm(zt), (xa, z1, a1, z2, a2, z3, a3)

    def logits(self, x):
        xa = x * self._scale()
        a1 = np.maximum(0, xa @ self.W1 + self.b1)
        a2 = np.maximum(0, a1 @ self.W2 + self.b2)
        a3 = np.maximum(0, a2 @ self.W3 + self.b3)
        return a3 @ self.W4 + self.b4

    def predict(self, x):
        return self.forward(x.reshape(1, -1) if x.ndim == 1 else x)[0][0]

    def evaluate(self, X, y):
        p, _ = self.forward(X)
        return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-9))), float(np.mean(p.argmax(1) == y))

    def train_step(self, X, y, w=None, smooth=0.05):
        n = X.shape[0]
        probs, (xa, z1, a1, z2, a2, z3, a3) = self.forward(X, training=True)
        y1h = np.full_like(probs, smooth / 3)
        y1h[np.arange(n), y] += 1.0 - smooth
        sw = np.ones(n) if w is None else np.asarray(w, dtype=float)
        loss = -np.mean(sw * np.sum(y1h * np.log(probs + 1e-9), axis=1))
        dz4 = (probs - y1h) * sw[:, None] / n
        dW4, db4 = a3.T @ dz4, dz4.sum(0)
        dz3 = (dz4 @ self.W4.T) * (z3 > 0)
        dW3, db3 = a2.T @ dz3, dz3.sum(0)
        dz2 = (dz3 @ self.W3.T) * (z2 > 0)
        dW2, db2 = a1.T @ dz2, dz2.sum(0)
        dz1 = (dz2 @ self.W2.T) * (z1 > 0)
        dW1, db1 = xa.T @ dz1, dz1.sum(0)
        dxa = dz1 @ self.W1.T
        gs = (dxa * X).sum(0)
        absA = np.abs(self.Att)
        S = absA.sum() + 1e-9
        datt = np.sign(self.Att) * len(absA) * (gs / S - (gs @ absA) / S ** 2)
        self.t += 1
        for k, gr in {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3, "W4": dW4, "b4": db4, "Att": datt}.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * gr
            self.v[k] = 0.999 * self.v[k] + 0.001 * gr * gr
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            old = getattr(self, k)
            new = old - self.lr * mh / (np.sqrt(vh) + self.eps)
            if self.wd and k[0] == "W":
                new = new - self.lr * self.wd * old
            setattr(self, k, new)
        return float(loss)

    def snapshot(self):
        return {k: getattr(self, k).copy() for k in _KEYS}

    def restore(self, s):
        for k in _KEYS:
            setattr(self, k, s[k].copy())


def fit_temp(nn, X, y):
    z = nn.logits(X)
    bl, bt = 1e9, 1.0
    for T in np.linspace(0.5, 3.0, 26):
        zz = z / T
        zz = zz - zz.max(1, keepdims=True)
        p = np.exp(zz)
        p /= p.sum(1, keepdims=True)
        l = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-9))
        if l < bl:
            bl, bt = l, float(T)
    return bt


def fit_model(nn, Xtr, ytr, Xva=None, yva=None, steps=400, bs=64, seed=None, smooth=0.05):
    rng = np.random.default_rng(seed)
    cw = (len(ytr) / (3.0 * (np.bincount(ytr, minlength=3) + 1.0))) ** 0.5
    lr0, T0 = nn.lr, nn.T
    nn.T = 1.0
    bl, best = float("inf"), None
    hv = Xva is not None and len(Xva) >= 20
    bs = min(bs, len(Xtr))
    for s in range(steps):
        nn.lr = lr0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / max(1, steps - 1))))
        idx = rng.integers(0, len(Xtr), bs)
        nn.train_step(Xtr[idx], ytr[idx], w=cw[ytr[idx]], smooth=smooth)
        if hv and (s % 20 == 19 or s == steps - 1):
            vl, _ = nn.evaluate(Xva, yva)
            if vl < bl:
                bl, best = vl, nn.snapshot()
    nn.lr = lr0
    if best is not None:
        nn.restore(best)
    nn.T = fit_temp(nn, Xva, yva) if hv else T0
    return nn


def eval_p(P, y, f, rp=None):
    rp = rp or DEFAULT_RISK
    loss = float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-9)))
    acc = float(np.mean(P.argmax(1) == y))
    base = float(np.bincount(y, minlength=3).max() / len(y))
    msk = (P[:, 2] >= rp["p_buy"]) & ((P[:, 2] - np.maximum(P[:, 0], P[:, 1])) >= rp["margin"])
    na = int(msk.sum())
    return {"loss": loss, "acc": acc, "base": base, "al_n": na,
            "al_ret": float(f[msk].mean()) if na else 0.0,
            "al_prec": float((y[msk] == 2).mean()) if na else 0.0, "n": int(len(y))}


class BaggedNN:
    def __init__(self, n=BAG_N, seed=BAG_SEED):
        self.nets = [NN(seed=seed + i * 7) for i in range(n)]
        self.perf = np.ones(n) / n
        self.weights = np.ones(n) / n
        self.best_vl = float("inf")
        self.plateau = 0
        self.evo = 0
        self.hard = []

    def _stack(self, X):
        return np.array([nn.forward(X)[0] for nn in self.nets])

    def predict_batch(self, X):
        return np.tensordot(self.weights, self._stack(X), axes=1)

    def predict_unc(self, X):
        S = self._stack(X)
        return np.tensordot(self.weights, S, axes=1), S.std(axis=0).mean(axis=1)

    def evaluate(self, X, y):
        P = self.predict_batch(X)
        return (float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-9))),
                float(np.mean(P.argmax(1) == y)), float(np.bincount(y, minlength=3).max() / len(y)))

    def fit_all(self, Xtr, ytr, Xva=None, yva=None, steps=350, seed=42, bootstrap=True, lr_scale=1.0, guard=False):
        rolled = 0
        for i, nn in enumerate(self.nets):
            rg = np.random.default_rng(seed + i * 7)
            if bootstrap:
                idx = rg.integers(0, len(Xtr), int(0.85 * len(Xtr)))
                X, y = Xtr[idx], ytr[idx]
            else:
                X, y = Xtr, ytr
            old = nn.evaluate(Xva, yva)[0] if (guard and Xva is not None) else None
            snap, T0, lr0 = nn.snapshot(), nn.T, nn.lr
            nn.lr = lr0 * lr_scale
            fit_model(nn, X, y, Xva, yva, steps=steps, seed=seed + i * 7)
            nn.lr = lr0
            if old is not None and nn.evaluate(Xva, yva)[0] > old * 1.03:
                nn.restore(snap)
                nn.T = T0
                rolled += 1
        return rolled

    def update_w(self, X, y):
        losses = np.array([nn.evaluate(X, y)[0] for nn in self.nets])
        inv = 1.0 / (losses + 1e-6)
        self.perf = 0.8 * self.perf + 0.2 * inv / inv.sum()
        self.perf /= self.perf.sum()
        self.weights = self.perf.copy()

    def evolve_step(self, Xtr, ytr, Xva, yva, n_batches=5):
        self.evo += 1
        if len(Xva) > 3000:
            sel = np.random.default_rng(self.evo).choice(len(Xva), 3000, replace=False)
            Xva, yva = Xva[sel], yva[sel]
        snaps = [(nn.snapshot(), nn.T) for nn in self.nets]
        vl0, _, _ = self.evaluate(Xva, yva)
        for _ in range(n_batches):
            n_hard = 64 // 3
            idx = np.random.randint(0, len(Xtr), 64 - n_hard)
            Xb, yb = Xtr[idx], ytr[idx]
            if len(self.hard) >= n_hard:
                ch = np.random.choice(len(self.hard), n_hard, replace=False)
                Xb = np.vstack([Xb, np.array([self.hard[i][0] for i in ch])])
                yb = np.concatenate([yb, np.array([self.hard[i][1] for i in ch])])
            for nn in self.nets:
                nn.train_step(Xb, yb)
        for nn in self.nets:
            nn.T = fit_temp(nn, Xva, yva)
        vl1, va1, _ = self.evaluate(Xva, yva)
        rolled = vl1 > vl0 * 1.02
        if rolled:
            for nn, (s, T) in zip(self.nets, snaps):
                nn.restore(s)
                nn.T = T
            vl1, va1, _ = self.evaluate(Xva, yva)
        self.update_w(Xva, yva)
        if vl1 < self.best_vl * 0.98:
            self.best_vl, self.plateau = vl1, 0
            for nn in self.nets:
                nn.dropout = max(0.05, nn.dropout * 0.95)
        else:
            self.plateau += 1
            if self.plateau >= 3:
                for nn in self.nets:
                    nn.dropout = min(0.40, nn.dropout * 1.15)
                self.plateau = 0
        P = self.predict_batch(Xva)
        wrong = np.where(P.argmax(1) != yva)[0][:20]
        for i in wrong:
            self.hard.append((Xva[i].copy(), int(yva[i])))
        self.hard = self.hard[-800:]
        db_log_train(self.evo, vl1, va1, "rollback" if rolled else "")
        return {"vl": vl1, "va": va1, "rolled": bool(rolled), "step": self.evo}

    def get_state(self):
        return {"nets": [{"w": nn.snapshot(), "T": nn.T, "t": nn.t, "dropout": nn.dropout, "lr": nn.lr} for nn in self.nets],
                "perf": self.perf.tolist(), "weights": self.weights.tolist(), "bvl": self.best_vl,
                "plateau": self.plateau, "evo": self.evo, "hard": self.hard[-300:]}

    def set_state(self, st):
        for nn, s in zip(self.nets, st["nets"]):
            nn.restore(s["w"])
            nn.T, nn.t, nn.dropout, nn.lr = s["T"], s["t"], s["dropout"], s["lr"]
        self.perf, self.weights = np.array(st["perf"]), np.array(st["weights"])
        self.best_vl, self.plateau, self.evo = st["bvl"], st["plateau"], st["evo"]
        self.hard = st.get("hard", [])


# ============================================================
# RISK / SINYAL
# ============================================================
def regime_of(row):
    vals = [safe_float(row.get("Regime_Bull_sm", 0)), safe_float(row.get("Regime_Bear_sm", 0)),
            safe_float(row.get("Regime_Vol_sm", 0)), safe_float(row.get("Regime_Range_sm", 0))]
    return REGIMES[int(np.argmax(vals))]


def regime_adj(rp):
    return np.array([rp.get("regime_bull_adj", 0), rp.get("regime_bear_adj", 0),
                     rp.get("regime_vol_adj", 0), rp.get("regime_range_adj", 0)], dtype=float)


def signal_from_probs(p, rp, regime=None):
    code = REGIMES.index(regime) if regime in REGIMES else None
    pb, ps = rp["p_buy"], rp["p_sell"]
    if code is not None:
        a = regime_adj(rp)[code]
        pb += a
        ps -= a * 0.5
    if p[2] >= pb and p[2] - max(p[0], p[1]) >= rp["margin"]:
        return "AL"
    if p[0] >= ps:
        return "SAT"
    return "TUT"


def conf_mult(p, rp):
    conf = float(p[2] - max(p[0], p[1]))
    base = rp.get("margin", 0.10)
    ratio = max(0.0, (conf - base) / max(1e-9, 1.0 - base))
    return 0.6 + 0.8 * min(1.0, ratio * 2.0)


def size_pos(eq, cash, expo, px, atr, probs, rp, fee=FEE):
    sd = rp["sl_atr"] * atr
    if sd <= 0 or px <= 0 or not np.isfinite(sd):
        return 0, sd
    cm = conf_mult(probs, rp) if (probs is not None and rp.get("conf_sizing", True)) else 1.0
    room = max(0.0, rp["max_exposure"] * eq - expo)
    q = int(min(eq * rp["risk_per_trade"] * cm / sd, eq * rp["max_pos"] / px, room / px, cash / (px * (1 + fee))))
    return max(q, 0), sd


def stop_level(entry, high, sd0, atr, rp):
    stop, trail = entry - sd0, False
    if high >= entry + rp["trail_act_r"] * sd0:
        ts = high - rp["trail_atr"] * atr
        if ts > stop:
            stop, trail = ts, True
    return stop, trail


# ============================================================
# GLOBAL BOT
# ============================================================
class GlobalBot:
    def __init__(self, cash=100000.0):
        self.cash, self.initial = cash, cash
        self.positions, self.trades = {}, []
        self.nn = BaggedNN()
        self.mem = []
        self.wins = self.losses = 0
        self.total_pnl = 0.0
        self.eq_hist = []
        self.peak_eq = cash
        self.halted, self.halt_reason, self.halt_until, self.streak = False, "", None, 0
        self.day_key, self.day_eq0, self.day_trades = None, cash, 0
        self.pretrained, self.val_stats, self.hold = False, None, None
        self.last_train_n = 0
        self.oos_ok, self.oos_note = True, "walk-forward yok"

    def total_value(self, prices):
        return self.cash + sum(p["qty"] * prices.get(s, p["entry"]) for s, p in self.positions.items())

    def exposure(self, prices):
        return sum(p["qty"] * prices.get(s, p["entry"]) for s, p in self.positions.items())

    def record(self, v):
        if not self.eq_hist or abs(self.eq_hist[-1] - v) > 1e-6:
            self.eq_hist = (self.eq_hist + [float(v)])[-3000:]

    def win_rate(self):
        t = self.wins + self.losses
        return self.wins / t if t else 0.0

    def max_dd(self):
        peak, dd = self.initial, 0.0
        for x in [self.initial] + self.eq_hist:
            peak = max(peak, x)
            dd = max(dd, (peak - x) / peak * 100)
        return dd

    def state_text(self):
        return f"HALT: {self.halt_reason}" if self.halted else "AKTIF"

    def pretrain(self, dss_list, rp, steps=600):
        merged = merge_ds(dss_list)
        sp = split_ds(merged) if merged else None
        if sp is None:
            return None
        self.nn.fit_all(sp["Xtr"], sp["ytr"], sp["Xva"], sp["yva"], steps=steps)
        self.nn.update_w(sp["Xva"], sp["yva"])
        self.hold = (sp["Xva"][-4000:], sp["yva"][-4000:], sp["fva"][-4000:])
        self.mem = list(zip(sp["Xtr"][-3000:], sp["ytr"][-3000:]))
        self.reeval(rp)
        for _ in range(2):
            self.nn.evolve_step(sp["Xtr"], sp["ytr"], sp["Xva"], sp["yva"])
        self.pretrained = True
        return self.val_stats

    def daily_update(self, dss_list, rp, steps=120):
        merged = merge_ds(dss_list)
        sp = split_ds(merged) if merged else None
        if sp is None:
            return None
        rolled = self.nn.fit_all(sp["Xtr"], sp["ytr"], sp["Xva"], sp["yva"], steps=steps, lr_scale=0.5, guard=True, seed=int(time.time()) % 10000)
        evo = self.nn.evolve_step(sp["Xtr"], sp["ytr"], sp["Xva"], sp["yva"])
        self.hold = (sp["Xva"][-4000:], sp["yva"][-4000:], sp["fva"][-4000:])
        self.reeval(rp)
        evo["fit_rolled"] = rolled
        return evo

    def reeval(self, rp):
        if self.hold:
            X, y, f = self.hold
            self.val_stats = eval_p(self.nn.predict_batch(X), y, f, rp)

    def edge_ok(self, rp):
        vs = self.val_stats
        return bool(vs and self.oos_ok and vs["al_n"] >= rp["gate_min_n"] and vs["al_ret"] > RT_COST * 1.5)

    def scan(self, names, F, rows, rp):
        P, U = self.nn.predict_unc(F)
        gate_ok = self.edge_ok(rp)
        out = {}
        for i, nm in enumerate(names):
            p, u, row = P[i], float(U[i]), rows[nm]
            reg = regime_of(row)
            act, note = signal_from_probs(p, rp, reg), ""
            if act == "AL":
                if rp["use_regime"] and not safe_float(row.get("Close")) > safe_float(row.get("SMA200")):
                    act, note = "TUT", "rejim"
                elif rp["gate_on"] and not gate_ok:
                    act, note = "TUT", "kapi"
                elif u > rp["unc_max"]:
                    act, note = "TUT", f"belirsizlik {u:.2f}"
            out[nm] = {"action": act, "probs": p, "unc": u, "regime": reg, "note": note,
                       "atr": safe_float(row.get("ATR"))}
        return out

    def halt(self, reason, hours, now):
        self.halted, self.halt_reason = True, reason
        self.halt_until = (now + timedelta(hours=hours)).isoformat(timespec="seconds")

    def resume(self, prices):
        self.halted, self.halt_reason, self.halt_until, self.streak = False, "", None, 0
        self.peak_eq = self.total_value(prices)

    def _sell(self, s, price, date, reason):
        p = self.positions.pop(s)
        px = price * (1 - SLIPPAGE)
        proceeds = p["qty"] * px * (1 - FEE)
        pnl = proceeds - p["cost"]
        self.cash += proceeds
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
            self.streak = 0
        else:
            self.losses += 1
            self.streak += 1
        self.trades = (self.trades + [{"date": date, "action": "SAT", "stock": s, "price": px, "qty": p["qty"], "pnl": pnl, "reason": reason[:200]}])[-500:]
        db_add_trade(s, "SAT", px, p["qty"], pnl, reason)
        return pnl

    def _buy(self, s, price, date, q, sd, atr, reason):
        px = price * (1 + SLIPPAGE)
        cost = q * px * (1 + FEE)
        self.cash -= cost
        self.positions[s] = {"qty": q, "entry": px, "cost": cost, "stop0": px - sd, "sd0": sd, "atr": atr,
                             "high": px, "ts": now_tr().isoformat(timespec="seconds")}
        self.day_trades += 1
        self.trades = (self.trades + [{"date": date, "action": "AL", "stock": s, "price": px, "qty": q, "reason": reason[:200]}])[-500:]
        db_add_trade(s, "AL", px, q, 0.0, reason)

    def run_cycle(self, dec, prices, rp, date, now=None):
        now = now or now_tr()
        msgs = []
        eq = self.total_value(prices)
        dk = now.strftime("%Y-%m-%d")
        if self.day_key != dk:
            self.day_key, self.day_eq0, self.day_trades = dk, eq, 0
        if self.halted and self.halt_until and now.isoformat(timespec="seconds") >= self.halt_until:
            self.resume(prices)
        self.peak_eq = max(self.peak_eq, eq)
        dd = (self.peak_eq - eq) / self.peak_eq * 100 if self.peak_eq > 0 else 0.0
        if dd >= rp["max_dd"] and not self.halted:
            self.halt(f"KILL-SWITCH DD {dd:.1f}%", rp["cooldown_h"], now)
            for s in list(self.positions):
                if s in prices:
                    msgs.append(f"{s}: SAT {self._sell(s, prices[s], date, 'KILL'):+.0f} TL")
            return msgs
        for s in list(self.positions):
            p, px = self.positions[s], prices.get(s)
            if not px:
                continue
            p["high"] = max(p["high"], px)
            stop, trail = stop_level(p["entry"], p["high"], p["sd0"], p["atr"], rp)
            pct = (px / p["entry"] - 1) * 100
            reason = None
            if px <= stop:
                reason = f"{'TRAILING' if trail else 'STOP'} {pct:+.1f}%"
            else:
                try:
                    if (now - datetime.fromisoformat(p["ts"])).days >= rp["max_hold"]:
                        reason = "ZAMAN STOPU"
                except Exception:
                    pass
            if reason is None and dec.get(s, {}).get("action") == "SAT":
                reason = "SINYAL SAT"
            if reason:
                msgs.append(f"{s}: SAT {self._sell(s, px, date, reason):+.0f} TL")
        if self.streak >= rp["loss_streak"] and not self.halted:
            self.halt(f"SOGUMA ({self.streak})", rp["cooldown_h"], now)
        if self.halted:
            return msgs
        eq = self.total_value(prices)
        if self.day_eq0 > 0 and (eq - self.day_eq0) / self.day_eq0 * 100 <= -rp["daily_loss"]:
            return msgs
        cands = sorted([(d["probs"][2] - max(d["probs"][0], d["probs"][1]), s) for s, d in dec.items()
                        if d["action"] == "AL" and s not in self.positions and prices.get(s)], reverse=True)
        for _, s in cands:
            if len(self.positions) >= rp["max_positions"] or self.day_trades >= rp["max_trades_day"]:
                break
            d = dec[s]
            px = prices[s] * (1 + SLIPPAGE)
            q, sd = size_pos(eq, self.cash, self.exposure(prices), px, d["atr"] or px * 0.02, d["probs"], rp)
            if q >= 1:
                self._buy(s, prices[s], date, q, sd, d["atr"] or px * 0.02, "AI AL")
                msgs.append(f"{s}: AL {q} @ {px:.2f}")
        return msgs

    def get_state(self):
        keys = ["cash", "initial", "positions", "trades", "mem", "wins", "losses", "total_pnl", "eq_hist", "peak_eq",
                "halted", "halt_reason", "halt_until", "streak", "day_key", "day_eq0", "day_trades", "pretrained",
                "val_stats", "hold", "last_train_n", "oos_ok", "oos_note"]
        d = {k: getattr(self, k) for k in keys}
        d["nn"] = self.nn.get_state()
        return d

    def set_state(self, d):
        for k, v in d.items():
            if k != "nn":
                setattr(self, k, v)
        self.nn.set_state(d["nn"])


def load_state(path=STATE_FILE):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj if isinstance(obj, dict) and obj.get("ver") == BOT_VERSION else None
    except Exception:
        return None


def save_state(bot, rp, path=STATE_FILE):
    try:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"ver": BOT_VERSION, "bot": bot.get_state(), "rp": rp}, f)
        os.replace(tmp, path)
        return True
    except Exception as e:
        log(f"save_state hata: {e}")
        return False


# ============================================================
# VERI
# ============================================================
def _fetch(sym, period="5y"):
    for _ in range(2):
        try:
            d = yf.Ticker(sym).history(period=period, interval="1d")
            if d is not None and not d.empty:
                d = d.reset_index()
                d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
                return d[["Date", "Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            time.sleep(0.4)
    return None


def fetch_all():
    items = list(HISSELER.items()) + [(f"X:{n}", t) for n, t in CROSS_SYMBOLS.items()]
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = dict(ex.map(lambda kv: (kv[0], _fetch(kv[1])), items))
    cross = {n: res[f"X:{n}"].set_index("Date")["Close"] for n in CROSS_SYMBOLS if res.get(f"X:{n}") is not None}
    dfs = {}
    for nm in HISSELER:
        d = res.get(nm)
        if d is None:
            continue
        try:
            d = add_cross(add_ind(d), cross)
            if len(d) >= 300:
                dfs[nm] = d
        except Exception as e:
            log(f"{nm} isleme hata: {e}")
    return dfs


# ============================================================
# MAIN
# ============================================================
def main():
    log(f"=== v{BOT_VERSION} START ===")
    log(f"Borsa: {'ACIK' if borsa_acik() else 'KAPALI'}")

    state = load_state()
    bot = GlobalBot()
    rp = dict(DEFAULT_RISK)
    if state:
        try:
            bot.set_state(state["bot"])
            rp.update(state.get("rp", {}))
            log(f"State yuklendi (pretrained={bot.pretrained})")
        except Exception as e:
            log(f"state hatasi: {e}")
            bot = GlobalBot()

    for k, v in DEFAULT_RISK.items():
        rp.setdefault(k, v)

    now = now_tr()
    today = now.strftime("%Y-%m-%d")

    with open("_last_run.txt", "w") as f:
        f.write(today)

    dfs = fetch_all()
    log(f"Veri: {len(dfs)} hisse")
    if not dfs:
        log("HATA: veri yok")
        return 1

    dss = {nm: make_ds(d, nm) for nm, d in dfs.items()}
    datasets = [v for v in dss.values() if v]

    if not bot.pretrained:
        log("Bastan egitim...")
        bot.pretrain(datasets, rp, steps=500)
        log(f"Egitim tamam: val_acc={bot.val_stats['acc']*100:.1f}% taban={bot.val_stats['base']*100:.1f}%")
    else:
        log("Gunluk ince ayar...")
        bot.daily_update(datasets, rp, steps=100)

    names, rows, feats = [], {}, []
    for nm, d in dfs.items():
        i = len(d) - 1
        rows[nm] = d.iloc[i]
        feats.append(feature_matrix(d.iloc[[i]], nm)[0])
        names.append(nm)
    F_live = np.array(feats)
    prices = {nm: float(dfs[nm]["Close"].iloc[-1]) for nm in names}
    dec = bot.scan(names, F_live, rows, rp)

    n_al = sum(1 for d in dec.values() if d["action"] == "AL")
    n_sat = sum(1 for d in dec.values() if d["action"] == "SAT")
    log(f"Karar: AL={n_al} SAT={n_sat} TUT={len(dec)-n_al-n_sat}")

    if borsa_acik():
        msgs = bot.run_cycle(dec, prices, rp, now, now)
        for m in msgs:
            log(f"  {m}")
    else:
        log("Borsa kapali, islem yok")

    val = bot.total_value(prices)
    db_eq_add(val, bot.cash, len(bot.positions))
    log(f"Portfoy: {val:,.0f} TL | Pozisyon: {len(bot.positions)}")

    save_state(bot, rp)
    log(f"=== DONE ===")
    return 0


if __name__ == "__main__":
    exit(main())
