# ============================================================
# THYAO AI v25 KOMPAKT - 20 hisse, tek beyin, walk-forward
# Kagit islem simulasyonu. Yatirim tavsiyesi degildir.
# ============================================================
import os, re, json, math, time, html, pickle, sqlite3
from contextlib import contextmanager
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

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
MODELS_DIR = "checkpoints_v25"
CHAT_FILE = "chat_v25.json"
os.makedirs(MODELS_DIR, exist_ok=True)
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
RISK_PRESETS = {
    "dusuk": {"risk_per_trade": 0.005, "max_pos": 0.10, "sl_atr": 2.0, "max_dd": 10.0, "max_exposure": 0.6},
    "orta": {"risk_per_trade": 0.01, "max_pos": 0.15, "sl_atr": 2.5, "max_dd": 15.0, "max_exposure": 0.85},
    "yuksek": {"risk_per_trade": 0.02, "max_pos": 0.25, "sl_atr": 3.0, "max_dd": 25.0, "max_exposure": 1.0},
}


def now_tr():
    return datetime.now(TR)


def borsa_acik():
    n = now_tr()
    return n.weekday() < 5 and dtime(9, 55) <= n.time() <= dtime(18, 10)


def safe_float(x, d=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else d
    except (TypeError, ValueError):
        return d


# ============================================================
# DB
# ============================================================
@contextmanager
def db():
    c = sqlite3.connect(DB_FILE, timeout=15.0)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def db_init():
    with db() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, stock TEXT, action TEXT, price REAL, qty INTEGER, pnl REAL, reason TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS equity (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, value REAL, cash REAL, npos INTEGER)")
        c.execute("CREATE TABLE IF NOT EXISTS train (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, step INTEGER, vl REAL, va REAL, note TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS snaps (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, step INTEGER, vl REAL, va REAL, filepath TEXT)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_eq ON equity(ts)")


def _q(sql, args=()):
    try:
        with db() as c:
            return c.execute(sql, args).fetchall()
    except Exception:
        return []


def db_add_trade(s, a, p, q, pnl, r):
    _q("INSERT INTO trades (ts,stock,action,price,qty,pnl,reason) VALUES (?,?,?,?,?,?,?)",
       (now_tr().isoformat(timespec="seconds"), s, a, float(p), int(q), float(pnl or 0), str(r)[:300]))


def db_trades(limit=300):
    r = _q("SELECT ts,stock,action,price,qty,pnl,reason FROM trades ORDER BY id DESC LIMIT ?", (limit,))
    return [dict(zip(["ts", "stock", "action", "price", "qty", "pnl", "reason"], x)) for x in r]


def db_eq_add(v, cash, np_):
    _q("INSERT INTO equity (ts,value,cash,npos) VALUES (?,?,?,?)",
       (now_tr().isoformat(timespec="seconds"), float(v), float(cash), int(np_)))


def db_eq(limit=20000):
    r = _q("SELECT ts,value,cash,npos FROM equity ORDER BY id DESC LIMIT ?", (limit,))
    return [{"ts": x[0], "value": x[1], "cash": x[2], "npos": x[3]} for x in reversed(r)]


def db_log_train(step, vl, va, note=""):
    _q("INSERT INTO train (ts,step,vl,va,note) VALUES (?,?,?,?,?)",
       (now_tr().isoformat(timespec="seconds"), int(step), float(vl), float(va), note))


def db_train(limit=100):
    r = _q("SELECT ts,step,vl,va,note FROM train ORDER BY id DESC LIMIT ?", (limit,))
    return [dict(zip(["ts", "step", "vl", "va", "note"], x)) for x in r]


def db_snap(bag, step, vl, va):
    fpath = os.path.join(MODELS_DIR, f"m{step}_{now_tr():%Y%m%d_%H%M%S}.pkl")
    try:
        with open(fpath, "wb") as f:
            pickle.dump({"ver": BOT_VERSION, "s": bag.get_state()}, f)
        _q("INSERT INTO snaps (ts,step,vl,va,filepath) VALUES (?,?,?,?,?)",
           (now_tr().isoformat(timespec="seconds"), int(step), float(vl), float(va), fpath))
        olds = _q("SELECT id,filepath FROM snaps ORDER BY id DESC LIMIT -1 OFFSET 10")
        for i, fp in olds:
            try:
                os.remove(fp)
            except OSError:
                pass
            _q("DELETE FROM snaps WHERE id=?", (i,))
    except Exception:
        pass


def db_snaps(limit=10):
    r = _q("SELECT ts,step,vl,va,filepath FROM snaps ORDER BY id DESC LIMIT ?", (limit,))
    return [dict(zip(["ts", "step", "vl", "va", "filepath"], x)) for x in r]


def db_sum():
    c = lambda t: (_q(f"SELECT COUNT(*) FROM {t}") or [(0,)])[0][0]
    sz = os.path.getsize(DB_FILE) / 1024 if os.path.exists(DB_FILE) else 0
    return {"trades": c("trades"), "equity": c("equity"), "train": c("train"), "snaps": c("snaps"), "kb": round(sz, 1)}


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
# FEATURE MATRIX
# ============================================================
FEAT_NAMES = [
    "RSI", "MACDh", "SMA_trend", "BB_pos", "Stoch", "ADX", "Volatilite", "Vol_ratio",
    "Mom10", "HL_ratio", "Gap", "SMA200_uzak", "EMA_trend", "C>SMA20", "C>SMA50",
    "MACD_poz", "SMA20>50", "WilliamsR", "Mom30", "C>BB_mid", "ATR%", "Ret5", "HI120",
    "K-D_fark", "W_RSI", "W_Trend", "W_Strength",
    "Reg_Bull", "Reg_Bear", "Reg_Vol", "Reg_Range",
    "USDTRY_etki", "XU100_etki", "XU100_kor"] + [f"Hisse_{s}" for s in STOCK_LIST]


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
        self.loss_history = []

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
        self.loss_history = (self.loss_history + [float(loss)])[-300:]
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

    def n_params(self):
        return sum(getattr(self, k).size for k in _KEYS)


def fit_temp(nn, X, y):
    z = nn.logits(X)
    best_l, best_t = 1e9, 1.0
    for T in np.linspace(0.5, 3.0, 26):
        zz = z / T
        zz = zz - zz.max(1, keepdims=True)
        p = np.exp(zz)
        p /= p.sum(1, keepdims=True)
        l = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-9))
        if l < best_l:
            best_l, best_t = l, float(T)
    return best_t


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
        self.imp = np.ones(N_FEAT) / N_FEAT
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
            X, y = (Xtr[rg.integers(0, len(Xtr), int(0.85 * len(Xtr)))], ytr[rg.integers(0, len(ytr), int(0.85 * len(ytr)))]) if bootstrap else (Xtr, ytr)
            if bootstrap:
                idx = rg.integers(0, len(Xtr), int(0.85 * len(Xtr)))
                X, y = Xtr[idx], ytr[idx]
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
        if self.evo % 10 == 0:
            self._update_imp(Xva[:300], yva[:300])
        db_log_train(self.evo, vl1, va1, "rollback" if rolled else "")
        return {"vl": vl1, "va": va1, "rolled": bool(rolled), "step": self.evo}

    def _update_imp(self, X, y):
        if len(X) < 50:
            return
        bl, _, _ = self.evaluate(X, y)
        imp = np.zeros(X.shape[1])
        rg = np.random.default_rng(self.evo)
        for j in range(N_FEAT_TECH):
            Xp = X.copy()
            Xp[:, j] = rg.permutation(Xp[:, j])
            imp[j] = max(0.0, self.evaluate(Xp, y)[0] - bl)
        if imp.sum() > 0:
            imp /= imp.sum()
            self.imp = 0.7 * self.imp + 0.3 * imp

    def top_feats(self, k=10):
        return [int(i) for i in np.argsort(-self.imp)[:k]]

    def status(self):
        return {"step": self.evo, "best": int(np.argmax(self.perf)),
                "perf": [round(float(p), 3) for p in self.perf], "hard": len(self.hard),
                "bvl": round(self.best_vl, 4) if np.isfinite(self.best_vl) else None,
                "plateau": self.plateau}

    def n_params(self):
        return sum(nn.n_params() for nn in self.nets)

    def get_state(self):
        return {"nets": [{"w": nn.snapshot(), "T": nn.T, "t": nn.t, "dropout": nn.dropout, "lr": nn.lr} for nn in self.nets],
                "perf": self.perf.tolist(), "weights": self.weights.tolist(), "bvl": self.best_vl,
                "plateau": self.plateau, "evo": self.evo, "imp": self.imp.tolist(), "hard": self.hard[-300:]}

    def set_state(self, st):
        for nn, s in zip(self.nets, st["nets"]):
            nn.restore(s["w"])
            nn.T, nn.t, nn.dropout, nn.lr = s["T"], s["t"], s["dropout"], s["lr"]
        self.perf, self.weights = np.array(st["perf"]), np.array(st["weights"])
        self.best_vl, self.plateau, self.evo = st["bvl"], st["plateau"], st["evo"]
        self.imp, self.hard = np.array(st["imp"]), st.get("hard", [])


# ============================================================
# ONLINE LEARNER
# ============================================================
class OnlineLearner:
    def __init__(self):
        self.pending = {}
        self.done = 0
        self.hits = 0

    def register(self, stock, date, feat, pred):
        d = str(pd.Timestamp(date).date())
        lst = self.pending.setdefault(stock, [])
        item = {"d": d, "feat": np.array(feat), "pred": int(pred)}
        if lst and lst[-1]["d"] == d:
            lst[-1] = item
        else:
            lst.append(item)
        self.pending[stock] = lst[-40:]

    def resolve(self, dfs):
        out = []
        for stock, lst in self.pending.items():
            df = dfs.get(stock)
            if df is None:
                continue
            pos = {str(pd.Timestamp(x).date()): i for i, x in enumerate(df["Date"].to_numpy())}
            hi, lo = df["High"].to_numpy(float), df["Low"].to_numpy(float)
            cl, atr = df["Close"].to_numpy(float), df["ATR"].to_numpy(float)
            keep = []
            for it in lst:
                i = pos.get(it["d"])
                if i is None or i + HORIZON > len(df) - 1:
                    keep.append(it)
                    continue
                lab, _ = tb_label(hi, lo, cl, atr, i)
                out.append((stock, it["feat"], lab))
                self.done += 1
                self.hits += int(lab == it["pred"])
            self.pending[stock] = keep
        return out

    def stats(self):
        return {"cozulen": self.done, "bekleyen": sum(len(v) for v in self.pending.values()),
                "isabet": round(self.hits / self.done * 100, 1) if self.done else 0.0}


# ============================================================
# SIMULATOR + WALK-FORWARD
# ============================================================
REGIMES = ["BULL", "BEAR", "VOL", "RANGE"]


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


def build_pf(dfs, dss):
    syms = [s for s in dfs if dss.get(s)]
    if not syms:
        return None
    mx = max(len(dfs[s]) for s in syms)
    syms = [s for s in syms if len(dfs[s]) >= 0.9 * mx]
    common = None
    for s in syms:
        idx = pd.DatetimeIndex(dfs[s]["Date"])
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) < 400:
        return None
    common = common.sort_values()
    pf = {"syms": syms, "dates": common.to_numpy(), "O": {}, "H": {}, "L": {}, "C": {}, "A": {}, "S200": {}, "REG": {}, "F": {}}
    cols = ["Regime_Bull_sm", "Regime_Bear_sm", "Regime_Vol_sm", "Regime_Range_sm"]
    for s in syms:
        d = dfs[s]
        m = d["Date"].isin(common).to_numpy()
        sub = d[m].reset_index(drop=True)
        for key, col in (("O", "Open"), ("H", "High"), ("L", "Low"), ("C", "Close"), ("A", "ATR"), ("S200", "SMA200")):
            pf[key][s] = sub[col].to_numpy(float)
        pf["REG"][s] = np.column_stack([sub[c].to_numpy(float) for c in cols]).argmax(1)
        pf["F"][s] = dss[s]["F"][m]
    return pf


def pooled_oos(pf, dss, n_folds=4, steps=400, seed=0, min_frac=0.4):
    dates, syms = pf["dates"], pf["syms"]
    n = len(dates)
    P = {s: np.full((n, 3), np.nan) for s in syms}
    start = int(max(250, n * min_frac))
    if n - start < 60:
        return P, []
    merged = merge_ds([dss[s] for s in syms])
    bounds = np.linspace(start, n, n_folds + 1).astype(int)
    gap = np.timedelta64(int(HORIZON * 1.5) + 1, "D")
    folds = []
    for k in range(n_folds):
        a, b = bounds[k], bounds[k + 1]
        tr = merged["d"] < (dates[a] - gap)
        if tr.sum() < 600:
            continue
        X, y, f, d = merged["X"][tr], merged["y"][tr], merged["f"][tr], merged["d"][tr]
        cut = np.unique(d)[int(len(np.unique(d)) * 0.85)]
        trm, vam = d < (cut - gap), d >= cut
        if trm.sum() < 400 or vam.sum() < 50:
            continue
        nn = NN(seed=seed + k)
        fit_model(nn, X[trm], y[trm], X[vam], y[vam], steps=steps, seed=seed + k)
        for s in syms:
            P[s][a:b] = nn.forward(pf["F"][s][a:b])[0]
        pv = nn.forward(X[vam])[0]
        e = eval_p(pv, y[vam], f[vam])
        folds.append({"Fold": k + 1, "Test": f"{pd.Timestamp(dates[a]):%Y-%m-%d} → {pd.Timestamp(dates[b - 1]):%Y-%m-%d}",
                      "Egitim": int(trm.sum()), "Val_acc": round(e["acc"] * 100, 1),
                      "Taban": round(e["base"] * 100, 1), "AL": e["al_n"], "AL_ret": round(e["al_ret"] * 100, 2)})
    return P, folds


def pf_sim(pf, P, rp, lo=None, hi=None, cash0=1_000_000.0, fee_mult=1.0):
    syms, n = pf["syms"], len(pf["dates"])
    starts = [int(np.argmax(~np.isnan(P[s][:, 2]))) for s in syms if (~np.isnan(P[s][:, 2])).any()]
    if not starts:
        return None
    first = min(starts)
    lo = first if lo is None else max(lo, first)
    hi = n if hi is None else min(hi, n)
    if hi - lo < 30:
        return None
    fee, slip = FEE * fee_mult, SLIPPAGE * fee_mult
    O, H, L, C, A, S2, RG = pf["O"], pf["H"], pf["L"], pf["C"], pf["A"], pf["S200"], pf["REG"]
    adj = regime_adj(rp)
    maxn, mg = int(rp["max_positions"]), rp["margin"]
    cash, pos, trades = cash0, {}, []
    ps, pb = {}, []
    eq = np.full(n, np.nan)
    peak, halt_until, expo_sum, npos_sum = cash0, -1, 0.0, 0

    def close(s, px, j, reason):
        nonlocal cash
        p = pos.pop(s)
        proceeds = p["sh"] * px * (1 - fee)
        cash += proceeds
        trades.append({"Hisse": s, "Giris": pd.Timestamp(pf["dates"][p["ei"]]).strftime("%Y-%m-%d"),
                       "Cikis": pd.Timestamp(pf["dates"][j]).strftime("%Y-%m-%d"),
                       "Giris_f": round(p["entry"], 2), "Cikis_f": round(px, 2),
                       "pnl": proceeds - p["cost"], "ret": proceeds / p["cost"] - 1, "bar": j - p["ei"], "neden": reason})

    for j in range(lo, hi):
        for s, why in list(ps.items()):
            if s in pos:
                close(s, O[s][j] * (1 - slip), j, why)
        ps = {}
        if pb and j > lo and j > halt_until:
            vopen = cash + sum(p["sh"] * O[s2][j] for s2, p in pos.items())
            for s, pr in pb:
                if s in pos or len(pos) >= maxn:
                    continue
                px = O[s][j] * (1 + slip)
                expo = sum(p["sh"] * O[s2][j] for s2, p in pos.items())
                q, sd = size_pos(vopen, cash, expo, px, A[s][j - 1], pr, rp, fee)
                if q >= 1:
                    cost = q * px * (1 + fee)
                    cash -= cost
                    pos[s] = {"sh": q, "entry": px, "hi": px, "sd0": sd, "atr": A[s][j - 1], "cost": cost, "ei": j}
        pb = []
        for s, p in list(pos.items()):
            stop, trail = stop_level(p["entry"], p["hi"], p["sd0"], p["atr"], rp)
            if L[s][j] <= stop:
                close(s, min(O[s][j], stop) * (1 - slip), j, "TRAILING" if trail else "STOP")
            else:
                p["hi"] = max(p["hi"], H[s][j])
        mv = sum(p["sh"] * C[s][j] for s, p in pos.items())
        eq[j] = cash + mv
        expo_sum += mv / eq[j]
        npos_sum += len(pos)
        peak = max(peak, eq[j])
        if (peak - eq[j]) / peak * 100 >= rp["max_dd"] and j > halt_until:
            ps = {s: "KILL" for s in pos}
            halt_until = j + max(1, int(rp["cooldown_h"] / 24))
            peak = eq[j]
        if j < hi - 1 and j > halt_until:
            cands = []
            for s in syms:
                pr = P[s][j]
                if pr[2] != pr[2]:
                    continue
                code = RG[s][j]
                pbu, pse = rp["p_buy"] + adj[code], rp["p_sell"] - adj[code] * 0.5
                if s in pos:
                    if pr[0] >= pse or (j - pos[s]["ei"]) >= rp["max_hold"]:
                        ps.setdefault(s, "SINYAL")
                elif pr[2] >= pbu and pr[2] - max(pr[0], pr[1]) >= mg and (not rp["use_regime"] or C[s][j] > S2[s][j]):
                    cands.append((pr[2] - max(pr[0], pr[1]), s, pr))
            cands.sort(key=lambda x: x[0], reverse=True)
            slots = maxn - len(pos) + len(ps)
            pb = [(s, pr) for _, s, pr in cands[:max(0, slots)]]
    for s in list(pos):
        close(s, C[s][hi - 1] * (1 - slip), hi - 1, "ACIK")
    eq[hi - 1] = cash
    e = eq[lo:hi]
    e = e[~np.isnan(e)]
    if len(e) < 5:
        return None
    dr = np.diff(e) / e[:-1]
    pk = np.maximum.accumulate(e)
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    loss = [t["pnl"] for t in trades if t["pnl"] <= 0]
    bh = float(np.mean([C[s][hi - 1] / C[s][lo] - 1 for s in syms]) * 100)
    stats = {"ret": float((e[-1] / cash0 - 1) * 100), "bh": bh,
             "sharpe": float(dr.mean() / dr.std() * math.sqrt(252)) if len(dr) > 2 and dr.std() > 0 else 0.0,
             "mdd": float(((pk - e) / pk).max() * 100), "n_trades": len(trades),
             "win": float(len(wins) / len(trades) * 100) if trades else 0.0,
             "pf": float(sum(wins) / abs(sum(loss))) if loss and sum(loss) != 0 else (99.0 if wins else 0.0),
             "exposure": float(expo_sum / max(1, hi - lo) * 100), "avg_pos": float(npos_sum / max(1, hi - lo))}
    return {"stats": stats, "trades": trades, "eq": eq, "first": lo, "hi": hi}


def fitness(res):
    if not res:
        return -50.0
    s = res["stats"]
    f = s["ret"] - 0.7 * s["mdd"] + 3.0 * float(np.clip(s["sharpe"], -2, 3))
    return float(f - 10.0 if s["n_trades"] < 3 else f)


def trade_sig(trades, n_boot=2000, seed=0):
    r = np.array([t["ret"] for t in trades], dtype=float)
    if len(r) < 5:
        return None
    rg = np.random.default_rng(seed)
    means = rg.choice(r, (n_boot, len(r))).mean(1)
    t = r.mean() / (r.std(ddof=1) / math.sqrt(len(r)) + 1e-12)
    return {"n": len(r), "mean": float(r.mean() * 100), "lo": float(np.percentile(means, 5) * 100),
            "hi": float(np.percentile(means, 95) * 100), "t": float(t), "p_pos": float((means > 0).mean())}


def oos_verdict(res):
    if not res:
        return True, "walk-forward yok"
    s, sig = res["stats"], trade_sig(res["trades"])
    if sig and sig["hi"] < 0:
        return False, f"OOS ort. %{sig['mean']:+.2f} (üst %{sig['hi']:+.2f} < 0)"
    if s["n_trades"] >= 10 and s["ret"] < 0 and s["sharpe"] < 0:
        return False, f"OOS getiri %{s['ret']:+.1f}, Sharpe {s['sharpe']:.2f}"
    return True, f"OOS getiri %{s['ret']:+.1f}, Sharpe {s['sharpe']:.2f}"


# ============================================================
# META EVOLVER
# ============================================================
DNA_BOUNDS = {"risk_per_trade": (0.003, 0.03), "sl_atr": (1.2, 4.5), "trail_act_r": (0.8, 3.5),
              "trail_atr": (1.2, 4.0), "p_buy": (0.36, 0.65), "p_sell": (0.36, 0.70),
              "margin": (0.02, 0.25), "max_pos": (0.08, 0.35)}
DNA_KEYS = list(DNA_BOUNDS) + ["use_regime"]


class MetaEvolver:
    def __init__(self, n=12):
        self.n, self.pop, self.hall = n, [], []
        self.gen, self.hist, self.refl, self.last = 0, [], [], None

    def _rand(self):
        d = {k: float(np.random.uniform(lo, hi)) for k, (lo, hi) in DNA_BOUNDS.items()}
        d["use_regime"] = bool(np.random.rand() < 0.6)
        return d

    def _mut(self, dna, rate=0.3):
        new = dict(dna)
        for k, (lo, hi) in DNA_BOUNDS.items():
            if np.random.rand() < rate:
                new[k] = float(np.clip(new[k] + np.random.normal(0, 0.12 * (hi - lo)), lo, hi))
        if np.random.rand() < rate * 0.5:
            new["use_regime"] = not new["use_regime"]
        return new

    def evolve(self, pf, P, base_rp, gens=6):
        ok = np.zeros(len(pf["dates"]), dtype=bool)
        for s in pf["syms"]:
            ok |= ~np.isnan(P[s][:, 2])
        if ok.sum() < 120:
            return None
        n, first = len(ok), int(np.argmax(ok))
        mid = (first + n) // 2
        base = {k: base_rp[k] for k in DNA_KEYS}
        run = lambda d, a, b: pf_sim(pf, P, {**base_rp, **d}, a, b)
        if not self.pop:
            self.pop = [self._rand() for _ in range(self.n)]
        self.pop[0] = dict(base)
        best = base
        for _ in range(gens):
            sc = sorted(((fitness(run(d, first, mid)), d) for d in self.pop), key=lambda x: x[0], reverse=True)
            self.gen += 1
            self.hist.append(float(sc[0][0]))
            best = sc[0][1]
            new = [sc[0][1], sc[1][1]]
            while len(new) < self.n:
                if np.random.rand() < 0.6:
                    a_ = sc[np.random.randint(0, min(4, len(sc)))][1]
                    b_ = sc[np.random.randint(0, min(4, len(sc)))][1]
                    new.append(self._mut({k: (a_[k] if np.random.rand() < 0.5 else b_[k]) for k in a_}, 0.2))
                else:
                    new.append(self._mut(sc[0][1], 0.4))
            self.pop = new
        bA, bB = run(best, first, mid), run(best, mid, n)
        cA, cB = run(base, first, mid), run(base, mid, n)
        fbA, fbB, fcA, fcB = fitness(bA), fitness(bB), fitness(cA), fitness(cB)
        adopt = bool(fbB > fcB + 1.0 and fbA >= fcA and bB and bB["stats"]["n_trades"] >= 5 and bB["stats"]["ret"] > 0)
        info = {"dna": dict(best), "fit_A": fbA, "fit_B": fbB, "base_A": fcA, "base_B": fcB, "adopt": adopt,
                "stats_B": bB["stats"] if bB else None}
        self.last = info
        self.hall = sorted(self.hall + [{"gen": self.gen, "fit_A": round(fbA, 2), "fit_B": round(fbB, 2),
                                         "adopt": adopt, "dna": dict(best)}], key=lambda x: x["fit_B"], reverse=True)[:8]
        sB = info["stats_B"] or {}
        self.refl = (self.refl + [f"Gen {self.gen}: A {fbA:+.1f} | B(OOS) {fbB:+.1f} vs {fcB:+.1f} | getiri %{sB.get('ret', 0):+.1f} | {'KABUL' if adopt else 'RED'}"])[-20:]
        return info


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
        return f"⛔ {self.halt_reason}" if self.halted else "✅ AKTİF"

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
        vl, va, _ = self.nn.evaluate(*self.hold[:2])
        db_snap(self.nn, self.nn.evo, vl, va)
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

    def learn(self, labeled):
        for _, feat, lab in labeled:
            self.mem.append((feat, int(lab)))
        self.mem = self.mem[-6000:]
        if len(self.mem) >= 100 and len(self.mem) - self.last_train_n >= 15:
            self.last_train_n = len(self.mem)
            X = np.array([m[0] for m in self.mem[-1500:]])
            y = np.array([m[1] for m in self.mem[-1500:]])
            self.nn.fit_all(X, y, steps=30, bootstrap=False, lr_scale=0.3)

    def scan(self, names, F, rows, rp):
        P, U = self.nn.predict_unc(F)
        gate_ok = self.edge_ok(rp)
        top = [FEAT_NAMES[i] for i in self.nn.top_feats(3)]
        out = {}
        for i, nm in enumerate(names):
            p, u, row = P[i], float(U[i]), rows[nm]
            reg = regime_of(row)
            act, note = signal_from_probs(p, rp, reg), ""
            if act == "AL":
                if rp["use_regime"] and not safe_float(row.get("Close")) > safe_float(row.get("SMA200")):
                    act, note = "TUT", "rejim filtresi"
                elif rp["gate_on"] and not gate_ok:
                    act, note = "TUT", "model kapisi"
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

    def kill(self, prices, date, rp, reason="ACIL DUR"):
        for s in list(self.positions):
            if s in prices:
                self._sell(s, prices[s], date, reason)
        self.halt(reason, rp["cooldown_h"] * 365, now_tr())

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
            self.halt(f"KILL-SWITCH DD %{dd:.1f}", rp["cooldown_h"], now)
            for s in list(self.positions):
                if s in prices:
                    msgs.append(f"{s}: SAT ₺{self._sell(s, prices[s], date, 'KILL'):+.0f}")
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
                reason = f"{'TRAILING' if trail else 'STOP'} ({pct:+.1f}%)"
            else:
                try:
                    if (now - datetime.fromisoformat(p["ts"])).days >= rp["max_hold"]:
                        reason = "ZAMAN STOPU"
                except Exception:
                    pass
            if reason is None and dec.get(s, {}).get("action") == "SAT":
                reason = "SINYAL SAT"
            if reason:
                msgs.append(f"{s}: SAT ₺{self._sell(s, px, date, reason):+.0f}")
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


def save_state(payload, path=STATE_FILE):
    try:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"ver": BOT_VERSION, **payload}, f)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def load_state(path=STATE_FILE):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj if isinstance(obj, dict) and obj.get("ver") == BOT_VERSION else None
    except Exception:
        return None


# ============================================================
# BILGI + SOHBET
# ============================================================
BILGI = {
    "Risk": {"ATR": "Volatilite ölçüsü; stop ve pozisyon boyutu bağlı.", "Kill-Switch": "DD limitinde hepsi kapanır.",
             "Sharpe": "Risk başına getiri. 1+ iyi.", "Sortino": "Sadece aşağı sapma.",
             "Calmar": "CAGR/MaxDD.", "VaR": "%95 azami kayıp.", "CVaR": "En kötü %5 ortalaması."},
    "AI": {"Triple Barrier": "+1.2 ATR/-1.2 ATR/zaman.", "Walk-forward": "Geçmişle eğit, ileri test.",
           "Kalibrasyon": "Temperature: %60 → ~%60.", "Model Kapısı": "Edge yoksa AL yok.",
           "Bagging": "5 ağ ortalaması.", "Online Öğrenme": "Gerçek etiketlerle güncelle."},
    "Göstergeler": {"RSI": "70+/30-", "MACD": "EMA12-EMA26", "ADX": "25+ güçlü trend",
                    "Bollinger": "SMA20±2std", "Rejim": "BULL/BEAR/VOL/RANGE"},
}


def bilgi_ara(q):
    q = q.lower()
    return [{"Kategori": k, "Konu": b, "Aciklama": i} for k, m in BILGI.items() for b, i in m.items()
            if q in b.lower() or q in i.lower()]


def chat_reply(soru, ctx):
    tok = set(re.findall(r"[a-zçğıöşü0-9]+", soru.lower()))
    has = lambda *w: any(x in tok for x in w)
    p = ctx["probs"]
    if has("merhaba", "selam", "hey"):
        return f"Merhaba! {ctx['sym']} {ctx['fiyat']:.2f} TL, karar {ctx['karar']}."
    if has("yardim", "komut"):
        return "Soru: durum, neden, risk, portfoy, rejim, kapi."
    if has("neden"):
        return f"{ctx['sym']} {ctx['karar']}"
    if has("portfoy", "para"):
        return f"₺{ctx['portfoy']:,.0f} | {ctx['npos']} pozisyon"
    if has("rejim"):
        return f"Rejim {ctx['regime']}"
    if has("risk", "kapi"):
        return f"Kapi: {'ACIK' if ctx['gate'] else 'KAPALI'} | Belirsizlik {ctx['unc']:.2f}"
    if has("durum"):
        return f"{ctx['sym']} ₺{ctx['fiyat']:.2f} | {ctx['karar']} | P(AL) %{p[2] * 100:.0f}"
    return f"{ctx['sym']}: {ctx['karar']}. 'yardim' yaz."


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="THYAO AI v25", page_icon="🧠", layout="wide")
h1, h2, h3 = st.columns([3, 1, 1])
h1.title("🧠 THYAO AI v25 Pro")
h1.caption("Tek beyin · 20 hisse · walk-forward · kill-switch")
(h2.success if borsa_acik() else h2.error)("🟢 AÇIK" if borsa_acik() else "🔴 KAPALI")
h3.caption(f"🕐 {now_tr():%H:%M:%S}")


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


@st.cache_resource(ttl=900, show_spinner=False)
def load_uni(day_key):
    items = list(HISSELER.items()) + [(f"X:{n}", t) for n, t in CROSS_SYMBOLS.items()]
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = dict(ex.map(lambda kv: (kv[0], _fetch(kv[1])), items))
    cross = {n: res[f"X:{n}"].set_index("Date")["Close"] for n in CROSS_SYMBOLS if res.get(f"X:{n}") is not None}
    dfs, errs = {}, []
    for nm in HISSELER:
        d = res.get(nm)
        if d is None:
            errs.append(nm)
            continue
        try:
            d = add_cross(add_ind(d), cross)
            if len(d) >= 300:
                dfs[nm] = d
            else:
                errs.append(f"{nm}(kısa)")
        except Exception as e:
            errs.append(f"{nm}:{e}")
    return {"dfs": dfs, "errs": errs, "ts": now_tr().strftime("%H:%M")}


@st.cache_data(ttl=30, show_spinner=False)
def live_prices(tickers):
    try:
        d = yf.download(list(tickers), period="1d", interval="1m", group_by="ticker", progress=False, threads=True, auto_adjust=True)
        out = {}
        for t in tickers:
            try:
                c = d[t]["Close"].dropna()
                if len(c):
                    out[t] = float(c.iloc[-1])
            except Exception:
                pass
        return out
    except Exception:
        return {}


def sig_idx(d, now):
    if len(d) > 1 and d["Date"].iloc[-1].date() == now.date() and now.time() < dtime(18, 20):
        return len(d) - 2
    return len(d) - 1


def init_book():
    b = {"bot": GlobalBot(), "meta": MetaEvolver(12), "rp": dict(DEFAULT_RISK), "risk_prev": None,
         "online": OnlineLearner(), "wf": None, "last_daily": None, "log": [], "evo": None, "mc": None,
         "ticks": 0, "last_tick": 0.0}
    s = load_state()
    if s:
        try:
            b["bot"].set_state(s["bot"])
            b["rp"].update(s.get("rp", {}))
            for k in ("meta", "online"):
                if s.get(k):
                    b[k].__dict__.update(s[k])
            b["last_daily"], b["log"], b["ticks"] = s.get("last_daily"), s.get("log", []), s.get("ticks", 0)
            b["risk_prev"] = s.get("risk_prev")
        except Exception:
            b["bot"] = GlobalBot()
    return b


def persist():
    b = st.session_state.book
    return save_state({"bot": b["bot"].get_state(), "rp": b["rp"], "meta": b["meta"].__dict__,
                       "online": b["online"].__dict__, "last_daily": b["last_daily"],
                       "log": b["log"][-100:], "ticks": b["ticks"], "risk_prev": b["risk_prev"]})


if "book" not in st.session_state:
    st.session_state.book = init_book()
    st.session_state.setdefault("chat", [])
    try:
        if os.path.exists(CHAT_FILE):
            st.session_state.chat = json.load(open(CHAT_FILE, encoding="utf-8"))
    except Exception:
        pass
st.session_state.setdefault("auto_on", False)
st.session_state.setdefault("test_mode", False)
st.session_state.setdefault("interval", 60)
st.session_state.setdefault("profile_risk", "orta")
book = st.session_state.book
bot, meta, rp, online = book["bot"], book["meta"], book["rp"], book["online"]
for _k, _v in DEFAULT_RISK.items():
    rp.setdefault(_k, _v)


def set_rp(updates):
    bk = st.session_state.book
    for k, v in updates.items():
        bk["rp"][k] = v
        if f"rp_{k}" in st.session_state:
            st.session_state[f"rp_{k}"] = v


now = now_tr()
with st.spinner("📥 Veri yükleniyor..."):
    uni = load_uni(now.strftime("%Y-%m-%d"))
dfs = uni["dfs"]
if not dfs:
    st.error("Veri yok: " + ", ".join(uni["errs"][:5]))
    st.stop()

if "dss" not in st.session_state or st.session_state.get("dss_key") != (uni["ts"], len(dfs)):
    st.session_state.dss = {nm: make_ds(d, nm) for nm, d in dfs.items()}
    st.session_state.dss_key = (uni["ts"], len(dfs))
dss = st.session_state.dss
datasets = [v for v in dss.values() if v]

today = now.strftime("%Y-%m-%d")
if not bot.pretrained:
    with st.spinner("🧠 Beyin eğitiliyor..."):
        bot.pretrain(datasets, rp, steps=600)
elif book["last_daily"] != today:
    with st.spinner("🔄 İnce ayar + walk-forward..."):
        book["evo"] = bot.daily_update(datasets, rp)
if bot.pretrained and book["last_daily"] != today:
    pf_ = build_pf(dfs, dss)
    if pf_:
        P_, folds_ = pooled_oos(pf_, dss, n_folds=4, steps=300)
        res_ = pf_sim(pf_, P_, rp)
        book["wf"] = {"P": P_, "folds": folds_, "res": res_, "pf": pf_}
        bot.oos_ok, bot.oos_note = oos_verdict(res_)
        info_ = meta.evolve(pf_, P_, rp, gens=6)
        if info_ and info_["adopt"] and rp.get("auto_adopt"):
            book["risk_prev"] = {k: rp[k] for k in DNA_KEYS}
            set_rp({k: info_["dna"][k] for k in DNA_KEYS})
            book["log"].append(f"{now:%H:%M:%S} Meta DNA uygulandı")
    book["last_daily"] = today
    persist()
bot.reeval(rp)

names, rows, feats = [], {}, []
for nm, d in dfs.items():
    i = sig_idx(d, now)
    rows[nm] = d.iloc[i]
    feats.append(feature_matrix(d.iloc[[i]], nm)[0])
    names.append(nm)
F_live = np.array(feats)
live = live_prices(tuple(HISSELER[n] for n in names))
prices = {nm: live.get(HISSELER[nm], float(dfs[nm]["Close"].iloc[-1])) for nm in names}
dec = bot.scan(names, F_live, rows, rp)

for i, nm in enumerate(names):
    online.register(nm, rows[nm]["Date"], F_live[i], int(np.argmax(dec[nm]["probs"])))
labeled = online.resolve(dfs)
if labeled:
    bot.learn(labeled)

secili = st.session_state.get("sel", "THYAO")
if secili not in dfs:
    secili = names[0]
df, last, sd_ = dfs[secili], rows[secili], dec[secili]
price = prices[secili]

auto_msgs = []
interval = st.session_state.interval
if st.session_state.auto_on and time.time() - book["last_tick"] >= interval - 1 \
        and (borsa_acik() or st.session_state.test_mode) and bot.pretrained:
    book["last_tick"] = time.time()
    book["ticks"] += 1
    auto_msgs = bot.run_cycle(dec, prices, rp, now, now)
    v_ = bot.total_value(prices)
    bot.record(v_)
    db_eq_add(v_, bot.cash, len(bot.positions))
    if auto_msgs:
        book["log"] = (book["log"] + [f"{now:%H:%M:%S} " + " | ".join(auto_msgs)])[-100:]
    if book["ticks"] % 5 == 0:
        persist()
if st.session_state.auto_on:
    st_autorefresh(interval=interval * 1000, key="auto_refresh")
cur_val = bot.total_value(prices)
bot.record(cur_val)

with st.sidebar:
    st.markdown("## ⚙️ Panel")
    st.selectbox("📊 Hisse", names, index=names.index(secili), key="sel")
    st.metric(secili, f"{price:.2f} TL", f"{(price / float(df['Close'].iloc[-2]) - 1) * 100:+.2f}%")
    st.metric("💰 Portföy", f"{cur_val:,.0f} TL", f"{cur_val - bot.initial:+,.0f} TL")
    st.divider()
    st.toggle("🟢 Otomatik işlem", key="auto_on")
    st.toggle("🧪 Test modu", key="test_mode")
    st.slider("Yenileme (sn)", 30, 300, step=10, key="interval")
    if st.session_state.auto_on:
        st.caption(f"Tick: {book['ticks']} · " + ("çalışıyor" if (borsa_acik() or st.session_state.test_mode) else "kapalı"))
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Pozisyon", len(bot.positions))
    c2.metric("Hafıza", len(bot.mem))
    st.caption(f"Evrim {bot.nn.evo} · Meta {meta.gen} · {bot.state_text()}")
    st.caption(f"Kapı: {'✅ açık' if bot.edge_ok(rp) else '⛔ kapalı'}")
    st.divider()
    if st.button("💾 Kaydet", use_container_width=True):
        st.toast("✅" if persist() else "❌")
    if st.button("🛑 ACİL DUR", use_container_width=True, type="primary"):
        bot.kill(prices, now, rp)
        persist()
        st.rerun()
    if st.button("▶️ Devam", use_container_width=True):
        bot.resume(prices)
        st.rerun()
    if st.button("🗑️ Sıfırla", use_container_width=True):
        st.session_state.pop("book", None)
        for f_ in (STATE_FILE, DB_FILE):
            if os.path.exists(f_):
                os.remove(f_)
        st.rerun()

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Fiyat", f"{price:.2f}")
m2.metric("Karar", sd_["action"], f"P(AL) %{sd_['probs'][2] * 100:.0f}")
m3.metric("Portföy", f"{cur_val:,.0f}", f"%{(cur_val / bot.initial - 1) * 100:+.2f}")
m4.metric("İşlem", len(bot.trades), f"kaz %{bot.win_rate() * 100:.0f}" if bot.trades else None)
m5.metric("Rejim", sd_["regime"])
m6.metric("Belirsizlik", f"{sd_['unc']:.2f}")
if sd_["note"]:
    st.warning(f"AL engellendi: {sd_['note']}")
if auto_msgs:
    st.info(" | ".join(auto_msgs[:5]))

tabs = st.tabs(["🎛️ Kontrol", "📈 Grafik", "🎯 Karar", "🧠 Beyin", "🧪 Backtest",
                "🧬 Meta", "🎲 Monte Carlo", "📊 Metrik", "💾 DB", "📚 Bilgi", "💬 Sohbet"])
tK, tG, tD, tB, tBT, tM, tMC, tMT, tDB, tBI, tCH = tabs


def _cb_rp(name, key):
    st.session_state.book["rp"][name] = st.session_state[key]


def rp_slider(name, label, lo, hi, step):
    key = f"rp_{name}"
    if key not in st.session_state:
        st.session_state[key] = type(step)(max(lo, min(hi, rp[name])))
    st.slider(label, lo, hi, step=step, key=key, on_change=_cb_rp, args=(name, key))


def rp_toggle(name, label):
    key = f"rp_{name}"
    if key not in st.session_state:
        st.session_state[key] = bool(rp[name])
    st.toggle(label, key=key, on_change=_cb_rp, args=(name, key))


def _cb_def(): set_rp(dict(DEFAULT_RISK))
def _cb_pre(): set_rp(dict(RISK_PRESETS[st.session_state.profile_risk]))


def _cb_dna():
    bk = st.session_state.book
    if bk["meta"].last:
        bk["risk_prev"] = {k: bk["rp"][k] for k in DNA_KEYS}
        set_rp({k: bk["meta"].last["dna"][k] for k in DNA_KEYS})


def _cb_rb():
    bk = st.session_state.book
    if bk["risk_prev"]:
        set_rp(dict(bk["risk_prev"]))
        bk["risk_prev"] = None


with tK:
    st.subheader("🎛️ Kontrol Merkezi")
    cA, cB, cC = st.columns(3)
    with cA:
        st.markdown("**Pozisyon / Stop**")
        rp_slider("risk_per_trade", "Risk %", 0.002, 0.05, 0.001)
        rp_slider("max_pos", "Tek poz max", 0.05, 0.5, 0.01)
        rp_slider("max_exposure", "Toplam maruziyet", 0.2, 1.0, 0.05)
        rp_slider("max_positions", "Max poz sayısı", 1, 20, 1)
        rp_slider("sl_atr", "Stop (ATR)", 1.0, 6.0, 0.1)
        rp_slider("trail_act_r", "Trail açılış (R)", 0.5, 4.0, 0.1)
        rp_slider("trail_atr", "Trail mesafe", 1.0, 5.0, 0.1)
    with cB:
        st.markdown("**Sinyal**")
        rp_slider("p_buy", "P(AL) eşik", 0.34, 0.80, 0.01)
        rp_slider("p_sell", "P(SAT) eşik", 0.34, 0.80, 0.01)
        rp_slider("margin", "Karar marjı", 0.0, 0.40, 0.01)
        rp_slider("unc_max", "Max belirsizlik", 0.05, 0.50, 0.01)
        rp_slider("max_hold", "Zaman stop", 5, 400, 5)
        rp_toggle("use_regime", "Rejim filtresi")
        rp_toggle("conf_sizing", "Güvenle boyut")
    with cC:
        st.markdown("**Yönetişim**")
        rp_slider("max_dd", "Kill-switch DD %", 3.0, 50.0, 0.5)
        rp_slider("daily_loss", "Günlük zarar %", 0.5, 15.0, 0.5)
        rp_slider("max_trades_day", "Günlük işlem", 1, 30, 1)
        rp_slider("loss_streak", "Soğuma streak", 1, 10, 1)
        rp_slider("cooldown_h", "Soğuma (saat)", 1, 240, 1)
        rp_toggle("gate_on", "Model kapısı")
        rp_slider("gate_min_n", "Kapı min AL", 3, 100, 1)
        rp_toggle("auto_adopt", "Meta DNA otomatik")
    st.divider()
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.button("🔄 Varsayılan", on_click=_cb_def, use_container_width=True)
    b2.selectbox("Profil", ["dusuk", "orta", "yuksek"], key="profile_risk", label_visibility="collapsed")
    b3.button("🛡️ Profil", on_click=_cb_pre, use_container_width=True)
    b4.button("🎯 Meta DNA", on_click=_cb_dna, disabled=not meta.last, use_container_width=True)
    b5.button("↩️ Geri", on_click=_cb_rb, disabled=not book["risk_prev"], use_container_width=True)
    st.divider()
    st.markdown("### 📦 Açık pozisyonlar")
    if not bot.positions:
        st.info("Pozisyon yok")
    else:
        prow = []
        for nm, p in bot.positions.items():
            cur = prices.get(nm, p["entry"])
            stop, trail = stop_level(p["entry"], p["high"], p["sd0"], p["atr"], rp)
            prow.append({"Hisse": nm, "Adet": p["qty"], "Giriş": round(p["entry"], 2),
                         "Şu an": round(cur, 2), "K/Z ₺": round((cur - p["entry"]) * p["qty"]),
                         "K/Z %": round((cur / p["entry"] - 1) * 100, 2), "Stop": round(stop, 2)})
        st.dataframe(pd.DataFrame(prow), use_container_width=True, hide_index=True)
    st.markdown("### ✋ Manuel emir")
    mc1, mc2 = st.columns(2)
    if mc1.button("Manuel AL", use_container_width=True):
        if secili in bot.positions or bot.halted:
            st.toast("Reddedildi")
        else:
            q_, sdv = size_pos(bot.total_value(prices), bot.cash, bot.exposure(prices), price,
                               sd_["atr"] or price * 0.02, sd_["probs"], rp)
            if q_ >= 1:
                bot._buy(secili, price, now, q_, sdv, sd_["atr"] or price * 0.02, "MANUEL AL")
                persist()
                st.rerun()
    if mc2.button("Manuel SAT", use_container_width=True):
        if secili in bot.positions:
            bot._sell(secili, price, now, "MANUEL SAT")
            persist()
            st.rerun()
        else:
            st.toast("Pozisyon yok")

with tG:
    n_show = st.select_slider("Gün", options=[60, 120, 180, 250, 400], value=180)
    d = df.tail(n_show)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.15, 0.2, 0.15])
    fig.add_trace(go.Candlestick(x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Fiyat"), row=1, col=1)
    for c_, clr in [("SMA20", "#f5a623"), ("SMA50", "#4a90e2"), ("SMA200", "#bd10e0")]:
        fig.add_trace(go.Scatter(x=d["Date"], y=d[c_], name=c_, line=dict(width=1.2, color=clr)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["BBu"], showlegend=False, line=dict(width=0.5, color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["BBd"], showlegend=False, fill="tonexty", line=dict(width=0.5, color="gray")), row=1, col=1)
    pos_ = bot.positions.get(secili)
    if pos_:
        stp, _ = stop_level(pos_["entry"], pos_["high"], pos_["sd0"], pos_["atr"], rp)
        fig.add_hline(y=stp, line_dash="dash", line_color="red", row=1, col=1)
    for t in bot.trades[-200:]:
        if t["stock"] == secili:
            try:
                td_ = pd.Timestamp(t["date"])
                if td_ >= d["Date"].iloc[0]:
                    fig.add_trace(go.Scatter(x=[td_], y=[t["price"]], mode="markers", showlegend=False,
                                             marker=dict(color="lime" if t["action"] == "AL" else "red", size=13,
                                                         symbol="triangle-up" if t["action"] == "AL" else "triangle-down")), row=1, col=1)
            except Exception:
                pass
    fig.add_trace(go.Bar(x=d["Date"], y=d["Volume"], showlegend=False, marker_color=np.where(d["Close"] >= d["Open"], "#26a69a", "#ef5350")), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["RSI"], showlegend=False, line=dict(color="#50e3c2")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    fig.add_trace(go.Bar(x=d["Date"], y=d["MACDh"], showlegend=False, marker_color=np.where(d["MACDh"] >= 0, "#26a69a", "#ef5350")), row=4, col=1)
    fig.update_layout(height=820, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with tD:
    st.subheader(f"🎯 {secili} Karar")
    pr_ = sd_["probs"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Karar", sd_["action"]); c2.metric("P(AL)", f"%{pr_[2] * 100:.1f}")
    c3.metric("P(TUT)", f"%{pr_[1] * 100:.1f}"); c4.metric("P(SAT)", f"%{pr_[0] * 100:.1f}")
    st.caption(f"Sinyal mumu: {pd.Timestamp(last['Date']):%Y-%m-%d}")
    st.divider()
    st.markdown("### 20 hisse tarayıcı")
    trows = [{"Hisse": nm, "Fiyat": round(prices[nm], 2), "Karar": d_["action"],
              "P(AL) %": round(d_["probs"][2] * 100, 0), "Belirsizlik": round(d_["unc"], 2),
              "Rejim": d_["regime"], "RSI": round(safe_float(rows[nm].get("RSI", 50)), 0), "Not": d_["note"]}
             for nm, d_ in dec.items()]
    st.dataframe(pd.DataFrame(sorted(trows, key=lambda x: -x["P(AL) %"])), use_container_width=True, hide_index=True)
    vs = bot.val_stats
    if vs:
        st.markdown("### 🚪 Model Kapısı")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Val acc", f"%{vs['acc'] * 100:.1f}", f"taban %{vs['base'] * 100:.1f}")
        k2.metric("AL çağrı", vs["al_n"]); k3.metric("AL isabet", f"%{vs['al_prec'] * 100:.1f}")
        k4.metric("AL ort.", f"%{vs['al_ret'] * 100:+.2f}")
        k5.metric("Kapı", "✅" if bot.edge_ok(rp) else "⛔")
        (st.success if bot.oos_ok else st.error)(bot.oos_note)

with tB:
    st.subheader("🧠 Beyin")
    s_ = bot.nn.status()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evrim", s_["step"]); c2.metric("En iyi net", f"#{s_['best']}")
    c3.metric("Zor örnek", s_["hard"]); c4.metric("Plateau", s_["plateau"])
    c1, c2 = st.columns(2)
    with c1:
        for i, p in enumerate(s_["perf"]):
            st.progress(float(min(max(p, 0.0), 1.0)), text=f"Net #{i}: {p:.3f}")
    with c2:
        st.metric("Parametre", f"{bot.nn.n_params():,}")
        st.metric("Mimari", f"{N_FEAT}→64→48→32→3 ×{BAG_N}")
        if s_["bvl"] is not None:
            st.metric("En iyi val loss", f"{s_['bvl']:.4f}")
    st.markdown("### Önemli 10 özellik")
    st.dataframe(pd.DataFrame([{"Özellik": FEAT_NAMES[i], "Önem": round(float(bot.nn.imp[i]), 4)} for i in bot.nn.top_feats(10)]),
                 use_container_width=True, hide_index=True)
    ol = online.stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Online etiket", ol["cozulen"]); c2.metric("Canlı isabet", f"%{ol['isabet']}")
    c3.metric("Bekleyen", ol["bekleyen"])
    st.divider()
    c1, c2 = st.columns(2)
    sp_ = split_ds(merge_ds(datasets)) if datasets else None
    if c1.button("🔄 10 adım evrim", use_container_width=True) and sp_:
        with st.spinner("Evrim..."):
            for _ in range(10):
                book["evo"] = bot.nn.evolve_step(sp_["Xtr"], sp_["ytr"], sp_["Xva"], sp_["yva"])
        bot.reeval(rp)
        st.rerun()
    if c2.button("🔁 Yeniden eğit", use_container_width=True) and datasets:
        with st.spinner("Eğitim..."):
            book["evo"] = bot.daily_update(datasets, rp, steps=250)
        st.rerun()

with tBT:
    st.subheader("🧪 Walk-forward Backtest")
    wf = book.get("wf")
    if wf and wf.get("res"):
        res = wf["res"]; s_ = res["stats"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Strateji", f"%{s_['ret']:+.1f}"); c2.metric("Al-tut", f"%{s_['bh']:+.1f}")
        c3.metric("Sharpe", f"{s_['sharpe']:.2f}"); c4.metric("MaxDD", f"%{s_['mdd']:.1f}")
        c5.metric("İşlem", s_["n_trades"], f"kaz %{s_['win']:.0f}")
        sig = trade_sig(res["trades"])
        if sig:
            ok_ = sig["lo"] > 0 and sig["n"] >= 20
            (st.success if ok_ else st.warning)(
                f"İşlem başı %{sig['mean']:+.2f} (%90: %{sig['lo']:+.2f} … %{sig['hi']:+.2f}) · t={sig['t']:.2f} · n={sig['n']}")
        eq, f0 = res["eq"], res["first"]
        x_ = pd.to_datetime(wf["pf"]["dates"][f0:res["hi"]])
        bh = np.mean([wf["pf"]["C"][s][f0:res["hi"]] / wf["pf"]["C"][s][f0] for s in wf["pf"]["syms"]], axis=0) * 1_000_000
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_, y=eq[f0:res["hi"]], name="Strateji", line=dict(color="#26a69a", width=2)))
        fig.add_trace(go.Scatter(x=x_, y=bh, name="Al-tut", line=dict(color="gray", dash="dot")))
        fig.update_layout(height=360, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        if wf.get("folds"):
            st.dataframe(pd.DataFrame(wf["folds"]), use_container_width=True, hide_index=True)
        if res["trades"]:
            tdf = pd.DataFrame(res["trades"])
            by = tdf.groupby("Hisse").agg(Islem=("pnl", "size"), PnL=("pnl", "sum"), Ort_ret=("ret", "mean")).reset_index()
            by["PnL"] = by["PnL"].round(0)
            by["Ort_ret"] = (by["Ort_ret"] * 100).round(2)
            st.dataframe(by.sort_values("PnL", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Sonuç yok")

with tM:
    st.subheader("🧬 Meta-evrim")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nesil", meta.gen)
    c2.metric("En iyi (B)", f"{meta.hall[0]['fit_B']:+.2f}" if meta.hall else "-")
    c3.metric("Son (A)", f"{meta.hist[-1]:+.2f}" if meta.hist else "-")
    if st.button("▶️ Evrim çalıştır", use_container_width=True) and book.get("wf") and book["wf"].get("pf"):
        with st.spinner("Evrim..."):
            meta.evolve(book["wf"]["pf"], book["wf"]["P"], rp, gens=8)
        st.rerun()
    if meta.hall:
        st.dataframe(pd.DataFrame([{"Nesil": h["gen"], "Fit A": h["fit_A"], "Fit B": h["fit_B"],
                                    "Kabul": "✅" if h["adopt"] else "—",
                                    "Risk %": round(h["dna"]["risk_per_trade"] * 100, 2),
                                    "SL ATR": round(h["dna"]["sl_atr"], 1),
                                    "Rejim": "evet" if h["dna"]["use_regime"] else "hayır"} for h in meta.hall]),
                     use_container_width=True, hide_index=True)
    if meta.hist:
        st.line_chart(pd.DataFrame({"Fit A": meta.hist}))
    for r in reversed(meta.refl):
        (st.success if r.endswith("KABUL") else st.info)(r)

with tMC:
    st.subheader("🎲 Monte Carlo")
    eq_d = np.array([e["value"] for e in db_eq(20000)], float)
    if len(eq_d) < 30:
        st.info(f"Yetersiz veri ({len(eq_d)}/30)")
    else:
        c1, c2 = st.columns(2)
        n_paths = c1.slider("Simülasyon", 200, 3000, 1000, 100)
        horizon = c2.slider("Ufuk (gün)", 60, 504, 252, 21)
        if st.button("▶️ Çalıştır", use_container_width=True):
            r = np.diff(eq_d) / eq_d[:-1]
            rg = np.random.default_rng(42)
            idx = rg.integers(0, len(r), (n_paths, horizon))
            finals = (1 + r[idx]).prod(axis=1) * eq_d[-1]
            pc = lambda q: float(np.percentile(finals, q))
            book["mc"] = {"baş": float(eq_d[-1]), "p5": pc(5), "p25": pc(25), "med": pc(50),
                          "p75": pc(75), "p95": pc(95), "kayıp": float((finals < eq_d[-1]).mean() * 100),
                          "finals": finals[:600].tolist()}
        mc = book.get("mc")
        if mc:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Başlangıç", f"₺{mc['baş']:,.0f}")
            c2.metric("Medyan", f"₺{mc['med']:,.0f}")
            c3.metric("P5", f"₺{mc['p5']:,.0f}")
            c4.metric("P95", f"₺{mc['p95']:,.0f}")
            st.metric("Kayıp olasılığı", f"%{mc['kayıp']:.1f}")
            if mc.get("finals"):
                fig = go.Figure(data=[go.Histogram(x=mc["finals"], nbinsx=40, marker_color="#4a90e2")])
                fig.add_vline(x=mc["baş"], line_dash="dash", line_color="yellow")
                fig.add_vline(x=mc["med"], line_dash="dash", line_color="lime")
                fig.update_layout(height=350, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

with tMT:
    st.subheader("📊 Metrikler")
    eq_d = np.array([e["value"] for e in db_eq(20000)], float)
    if len(eq_d) < 10:
        st.info("Yetersiz veri")
    else:
        r = np.diff(eq_d) / eq_d[:-1]
        pk = np.maximum.accumulate(eq_d)
        mdd = float(((pk - eq_d) / pk).max() * 100)
        sharpe = float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0
        cagr = ((eq_d[-1] / eq_d[0]) ** (252 / max(1, len(r))) - 1) * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe", f"{sharpe:.2f}")
        c2.metric("CAGR", f"%{cagr:.1f}")
        c3.metric("MaxDD", f"%{mdd:.1f}")
        c4.metric("VaR 95", f"%{np.percentile(r, 5) * 100:.2f}")
        st.line_chart(pd.DataFrame({"Portföy": eq_d}))

with tDB:
    st.subheader("💾 Veri Tabanı")
    sm = db_sum()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("İşlem", sm["trades"]); c2.metric("Equity", sm["equity"])
    c3.metric("Eğitim", sm["train"]); c4.metric("Snapshot", sm["snaps"])
    c5.metric("Boyut KB", sm["kb"])
    sub = st.radio("Görüntüle", ["İşlemler", "Eğitim", "Snapshotlar", "Equity"], horizontal=True)
    if sub == "İşlemler":
        r = db_trades(300)
        if r:
            df_t = pd.DataFrame(r)
            df_t["ts"] = pd.to_datetime(df_t["ts"]).dt.strftime("%m-%d %H:%M")
            st.dataframe(df_t, use_container_width=True, hide_index=True)
        else:
            st.info("Yok")
    elif sub == "Eğitim":
        r = db_train(100)
        if r:
            st.dataframe(pd.DataFrame(r), use_container_width=True, hide_index=True)
        else:
            st.info("Yok")
    elif sub == "Snapshotlar":
        r = db_snaps(10)
        if r:
            st.dataframe(pd.DataFrame(r), use_container_width=True, hide_index=True)
        else:
            st.info("Yok")
    else:
        r = db_eq(500)
        if r:
            st.dataframe(pd.DataFrame(r), use_container_width=True, hide_index=True)
        else:
            st.info("Yok")

with tBI:
    st.subheader("📚 Bilgi Bankası")
    q = st.text_input("Ara", placeholder="RSI, sharpe, kill-switch...")
    if q:
        for r in bilgi_ara(q):
            with st.expander(f"**{r['Konu']}** ({r['Kategori']})"):
                st.write(r["Aciklama"])
    else:
        for k, m in BILGI.items():
            with st.expander(f"📂 {k} ({len(m)})"):
                for b, i in m.items():
                    st.markdown(f"**{b}**: {i}")

with tCH:
    st.subheader("💬 Sohbet")
    ctx = {"sym": secili, "fiyat": price, "karar": sd_["action"], "probs": [float(x) for x in sd_["probs"]],
           "portfoy": cur_val, "npos": len(bot.positions), "regime": sd_["regime"],
           "unc": sd_["unc"], "gate": bot.edge_ok(rp)}
    for msg in st.session_state.chat[-30:]:
        with st.chat_message(msg.get("role", "user")):
            st.write(msg.get("content", ""))
    if p_ := st.chat_input("Sor..."):
        st.session_state.chat.append({"role": "user", "content": p_})
        st.session_state.chat.append({"role": "assistant", "content": chat_reply(p_, ctx)})
        st.session_state.chat = st.session_state.chat[-100:]
        try:
            json.dump(st.session_state.chat, open(CHAT_FILE, "w", encoding="utf-8"), ensure_ascii=False, default=str)
        except Exception:
            pass
        st.rerun()
    if st.session_state.chat and st.button("🗑️ Temizle"):
        st.session_state.chat = []
        st.rerun()

st.divider()
c1, c2, c3 = st.columns(3)
c1.caption(f"🕐 {now:%Y-%m-%d %H:%M:%S}")
c2.caption(f"Borsa: {'🟢 AÇIK' if borsa_acik() else '🔴 KAPALI'} · v{BOT_VERSION}")
c3.caption("⚠️ Kâğıt işlem simülasyonu - yatırım tavsiyesi değildir")
