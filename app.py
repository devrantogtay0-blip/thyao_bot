# THYAO AI v33 - VERIFIED MoE EDITION
from __future__ import annotations
import os, math, time, pickle, sqlite3
from contextlib import contextmanager
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from numpy.lib.stride_tricks import sliding_window_view

BOT_VERSION = 33
HORIZON = 10; HORIZON_LONG = 40
TB_K = 1.2; TB_K_LONG = 1.5
ALPHA_LONG = 0.35
FEE = 0.0015; SLIPPAGE = 0.001
RT_COST = 2 * (FEE + SLIPPAGE)
HW = 90
NEWS_ENABLED = False
N_NEWS = 3 if NEWS_ENABLED else 0
N_RANK = 5
LIQ_FLAG_TL = 5_000_000
N_FEAT_TECH = 40 + N_NEWS + 2 + N_RANK
H_DIM = 64; BAG_N = 5; BAG_SEED = 100; HOLD_N = 6000
USE_ROUTER = True; ROUTER_H = 32; ROUTER_MIX = 0.5
USE_DSR = True
DB_FILE = "bot_v33.db"; STATE_FILE = "state_v33.pkl"; MEMORY_FILE = "memory_v33.bin"
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
STOCK_LIST = list(HISSELER.keys()); N_STOCKS = len(STOCK_LIST)
N_FEAT = N_FEAT_TECH + N_STOCKS
CROSS_SYMBOLS = {"USDTRY": "USDTRY=X", "XU100": "XU100.IS"}

DEFAULT_RISK = {
    "risk_per_trade": 0.01, "sl_atr": 2.5, "trail_act_r": 1.5, "trail_atr": 2.5,
    "p_buy": 0.45, "p_sell": 0.45, "p_buy_long": 0.40, "margin": 0.10, "max_pos": 0.15,
    "use_regime": True, "use_long_gate": True, "max_hold": 120, "max_dd": 15.0,
    "daily_loss": 3.0, "max_trades_day": 5, "loss_streak": 3, "cooldown_h": 24,
    "gate_on": True, "gate_min_n": 8, "auto_adopt": True, "unc_max": 0.25,
    "conf_sizing": True, "max_positions": 8, "max_exposure": 0.85,
    "regime_bull_adj": -0.03, "regime_bear_adj": 0.08,
    "regime_range_adj": 0.03, "regime_vol_adj": 0.10,
    "si_auto": True, "si_interval_min": 30, "si_budget_s": 10,
    "min_daily_turnover": 5_000_000, "dsr_confidence": 0.90,
    "consec_improve": 2, "test_score_tol": 0.25,
}
REGIMES = ["BULL", "BEAR", "VOL", "RANGE"]
RISK_BOUNDS = {
    "risk_per_trade": (0.002, 0.05), "max_pos": (0.05, 0.50), "max_exposure": (0.2, 1.0),
    "max_positions": (1, 20), "sl_atr": (1.0, 6.0), "trail_act_r": (0.5, 4.0),
    "trail_atr": (1.0, 5.0), "p_buy": (0.34, 0.80), "p_sell": (0.34, 0.80),
    "p_buy_long": (0.34, 0.80), "margin": (0.0, 0.40), "unc_max": (0.05, 0.50),
    "max_hold": (5, 400), "max_dd": (3.0, 50.0), "daily_loss": (0.5, 15.0),
    "max_trades_day": (1, 30), "loss_streak": (1, 10), "cooldown_h": (1, 240),
    "gate_min_n": (3, 100), "regime_bull_adj": (-0.2, 0.3), "regime_bear_adj": (-0.2, 0.3),
    "regime_range_adj": (-0.2, 0.3), "regime_vol_adj": (-0.2, 0.3),
    "si_interval_min": (5, 720), "si_budget_s": (2, 60),
    "min_daily_turnover": (0, 1e9), "dsr_confidence": (0.5, 0.999),
    "consec_improve": (1, 5), "test_score_tol": (0.0, 2.0),
}
INT_KEYS = {"max_positions", "max_hold", "max_trades_day", "loss_streak", "cooldown_h",
            "gate_min_n", "si_interval_min", "si_budget_s", "consec_improve"}
TUNE_BOUNDS = {
    "p_buy": (0.36, 0.75), "p_buy_long": (0.36, 0.75), "margin": (0.0, 0.35),
    "unc_max": (0.05, 0.50), "sl_atr": (1.0, 5.0),
    "regime_bull_adj": (-0.10, 0.15), "regime_bear_adj": (0.0, 0.25),
    "regime_range_adj": (-0.05, 0.15), "regime_vol_adj": (0.0, 0.25),
}

def now_tr(): return datetime.now(TR)
def borsa_acik():
    n = now_tr()
    return n.weekday() < 5 and dtime(9, 55) <= n.time() <= dtime(18, 10)
def safe_float(x, d=0.0):
    try:
        v = float(x); return v if np.isfinite(v) else d
    except (TypeError, ValueError): return d
def clamp(v, lo, hi): return max(lo, min(hi, v))

def validate_risk(rp):
    out = {**DEFAULT_RISK, **(rp or {})}
    for k, (lo, hi) in RISK_BOUNDS.items():
        out[k] = clamp(safe_float(out.get(k), DEFAULT_RISK.get(k, 0)), lo, hi)
        if k in INT_KEYS: out[k] = int(round(out[k]))
    return out

@st.cache_resource(show_spinner=False)
def _db():
    c = sqlite3.connect(DB_FILE, timeout=30.0, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA synchronous=NORMAL")
    c.executescript("""
        CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, stock TEXT,
            action TEXT, price REAL, qty INTEGER, pnl REAL, reason TEXT);
        CREATE TABLE IF NOT EXISTS equity (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, value REAL,
            cash REAL, npos INTEGER);
        CREATE TABLE IF NOT EXISTS train (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, step INTEGER,
            vl REAL, va REAL, note TEXT);
        CREATE TABLE IF NOT EXISTS improve (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, reason TEXT,
            detail TEXT, before REAL, after REAL, adopted INTEGER);
        CREATE TABLE IF NOT EXISTS calibrate (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT,
            n INTEGER, brier REAL, ece REAL, acc REAL, base REAL);
    """)
    c.commit(); return c

@contextmanager
def db_cur():
    c = _db()
    try: yield c; c.commit()
    except Exception: c.rollback(); raise

def _ts(): return now_tr().isoformat(timespec="seconds")
def db_add_trade(s, a, p, q, pnl, r):
    try:
        with db_cur() as c:
            c.execute("INSERT INTO trades (ts,stock,action,price,qty,pnl,reason) VALUES (?,?,?,?,?,?,?)",
                      (_ts(), str(s)[:20], str(a)[:10], float(p), int(q), float(pnl or 0), str(r)[:300]))
    except Exception: pass
def db_trades(limit=300):
    try:
        with db_cur() as c:
            r = c.execute("SELECT ts,stock,action,price,qty,pnl,reason FROM trades ORDER BY id DESC LIMIT ?",
                          (int(limit),)).fetchall()
        return [dict(zip(["ts","stock","action","price","qty","pnl","reason"], x)) for x in r]
    except Exception: return []
def db_eq_add(v, cash, npos):
    try:
        with db_cur() as c:
            c.execute("INSERT INTO equity (ts,value,cash,npos) VALUES (?,?,?,?)", (_ts(), float(v), float(cash), int(npos)))
    except Exception: pass
def db_eq(limit=2000):
    try:
        with db_cur() as c:
            r = c.execute("SELECT ts,value,cash,npos FROM equity ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [{"ts": x[0], "value": x[1], "cash": x[2], "npos": x[3]} for x in reversed(r)]
    except Exception: return []
def db_log_train(step, vl, va, note=""):
    try:
        with db_cur() as c:
            c.execute("INSERT INTO train (ts,step,vl,va,note) VALUES (?,?,?,?,?)", (_ts(), int(step), float(vl), float(va), note))
    except Exception: pass
def db_train(limit=50):
    try:
        with db_cur() as c:
            r = c.execute("SELECT ts,step,vl,va,note FROM train ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(zip(["ts","step","vl","va","note"], x)) for x in r]
    except Exception: return []
def db_improve_add(reason, detail, before, after, adopted):
    try:
        with db_cur() as c:
            c.execute("INSERT INTO improve (ts,reason,detail,before,after,adopted) VALUES (?,?,?,?,?,?)",
                      (_ts(), str(reason)[:30], str(detail)[:500], float(before), float(after), int(adopted)))
    except Exception: pass
def db_calib_add(n, brier, ece, acc, base):
    try:
        with db_cur() as c:
            c.execute("INSERT INTO calibrate (ts,n,brier,ece,acc,base) VALUES (?,?,?,?,?,?)",
                      (_ts(), int(n), float(brier), float(ece), float(acc), float(base)))
    except Exception: pass
def db_clear():
    try:
        with db_cur() as c:
            for t in ("trades","equity","train","improve","calibrate"): c.execute(f"DELETE FROM {t}")
    except Exception: pass

class ExperienceBuffer:
    def __init__(self, capacity=5000, n_feat=N_FEAT):
        self.capacity, self.n_feat = capacity, n_feat
        self.X = np.zeros((capacity, n_feat), dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.int64)
        self.y_long = np.full(capacity, -1, dtype=np.int64)
        self.p = np.zeros(capacity, dtype=np.float32)
        self.ptr = self.n = self.total_added = 0
    def add(self, feat, label, label_long=-1, priority=1.0):
        self.X[self.ptr] = np.asarray(feat, dtype=np.float32)
        self.y[self.ptr] = int(label); self.y_long[self.ptr] = int(label_long)
        self.p[self.ptr] = max(1e-3, float(priority))
        self.ptr = (self.ptr + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity); self.total_added += 1
    def add_many(self, X, y, y_long=None, p=None):
        for i in range(len(X)):
            self.add(X[i], y[i], -1 if y_long is None else y_long[i], 1.0 if p is None else p[i])
    def sample(self, k, rng=None):
        if self.n < 8: return None
        rng = rng or np.random.default_rng()
        k = min(k, self.n)
        cdf = np.cumsum(self.p[:self.n].astype(np.float64))
        idx = np.minimum(np.searchsorted(cdf, rng.random(k) * cdf[-1]), self.n - 1)
        return self.X[idx], self.y[idx], self.y_long[idx]
    def save(self, path):
        try:
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(f, X=self.X[:self.n], y=self.y[:self.n],
                                    y_long=self.y_long[:self.n], p=self.p[:self.n])
            os.replace(tmp, path); return True
        except Exception: return False
    def load(self, path):
        if not os.path.exists(path): return False
        try:
            with np.load(path, allow_pickle=False) as z:
                if z["X"].shape[1] != self.n_feat: return False
                self.add_many(z["X"], z["y"], z["y_long"], z["p"])
            return True
        except Exception: return False
    def size(self): return self.n

def add_ind(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    d = c.diff()
    g = d.clip(lower=0).rolling(14).mean(); ls = (-d.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26; df["MACDs"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDh"] = df["MACD"] - df["MACDs"]
    df["SMA20"] = c.rolling(20).mean(); df["SMA50"] = c.rolling(50).mean(); df["SMA200"] = c.rolling(200).mean()
    df["EMA9"] = c.ewm(span=9, adjust=False).mean(); df["EMA21"] = c.ewm(span=21, adjust=False).mean()
    df["BBm"] = df["SMA20"]; df["BBs"] = c.rolling(20).std()
    df["BBu"] = df["BBm"] + 2 * df["BBs"]; df["BBd"] = df["BBm"] - 2 * df["BBs"]
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    rng = (hi14 - lo14).replace(0, np.nan)
    df["K"] = 100 * (c - lo14) / rng; df["D"] = df["K"].rolling(3).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean(); df["ATRp"] = df["ATR"] / c
    df["WILLR"] = -100 * (hi14 - c) / rng
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr14 = tr.rolling(14).sum().replace(0, np.nan)
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(14).sum() / tr14
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(14).sum() / tr14
    df["ADX"] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).rolling(14).mean()
    df["Vol_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
    for k in (1, 5, 10, 20, 30): df[f"Ret{k}"] = c.pct_change(k)
    df["Volatility"] = df["Ret1"].rolling(20).std() * np.sqrt(252)
    df["HL"] = (h - l) / c; df["Gap"] = (df["Open"] - c.shift()) / c.shift()
    df["HI120"] = c / c.rolling(120).max() - 1
    r20 = (h.rolling(20).max() - l.rolling(20).min()).replace(0, np.nan)
    df["Donch20"] = (c - l.rolling(20).min()) / r20
    df["RSI_slope"] = df["RSI"].diff(3); df["SMA50_slope"] = df["SMA50"].pct_change(10)
    df["ATR_chg"] = df["ATRp"] / df["ATRp"].rolling(50).mean() - 1
    df["Turnover20"] = (c * v).rolling(20).mean()
    s = df.set_index("Date")["Close"]; w = s.resample("W-FRI").last().dropna()
    wd = w.diff(); wg = wd.clip(lower=0).rolling(14).mean(); wl = (-wd.clip(upper=0)).rolling(14).mean()
    wk = pd.DataFrame({"WDate": pd.DatetimeIndex(w.index).astype("datetime64[ns]")})
    wk["W_RSI"] = (100 - 100 / (1 + wg / wl.replace(0, np.nan))).to_numpy()
    we9 = w.ewm(span=9, adjust=False).mean(); we21 = w.ewm(span=21, adjust=False).mean()
    wk["W_Trend"] = ((we9 - we21) / we21.replace(0, np.nan)).to_numpy()
    wsma = w.rolling(20).mean(); wk["W_Strength"] = ((w - wsma) / wsma.replace(0, np.nan)).to_numpy()
    wk = wk.replace([np.inf, -np.inf], np.nan).dropna()
    df = pd.merge_asof(df, wk, left_on="Date", right_on="WDate", direction="backward",
                       allow_exact_matches=False).drop(columns=["WDate"])
    tu = (c > df["SMA200"]) & (df["SMA50"] > df["SMA200"])
    td = (c < df["SMA200"]) & (df["SMA50"] < df["SMA200"])
    vh = df["Volatility"] > df["Volatility"].rolling(60).quantile(0.75); adx = df["ADX"] > 25
    df["Regime_Bull"] = (tu & ~vh & adx).astype(int); df["Regime_Bear"] = (td & ~vh & adx).astype(int)
    df["Regime_Vol"] = vh.astype(int); df["Regime_Range"] = (~tu & ~td & ~vh).astype(int)
    for cc, wnd in [("Regime_Bull", 10), ("Regime_Bear", 10), ("Regime_Vol", 5), ("Regime_Range", 10)]:
        df[cc + "_sm"] = df[cc].rolling(wnd, min_periods=1).mean()
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

def add_cross(df, cross):
    df = df.copy(); idx = pd.DatetimeIndex(df["Date"]); df["XU100_Ret20"] = 0.0
    for name in CROSS_SYMBOLS:
        s = (cross or {}).get(name)
        if s is None or len(s) < 30:
            df[f"{name}_Ret5"] = 0.0; df[f"{name}_Corr20"] = 0.0; continue
        s = s[~s.index.duplicated()].sort_index()
        al = pd.Series(s.reindex(idx, method="ffill").to_numpy(), index=df.index)
        if name == "USDTRY": al = al.shift(1)
        df[f"{name}_Ret5"] = al.pct_change(5, fill_method=None).fillna(0.0)
        df[f"{name}_Corr20"] = df["Ret1"].rolling(20).corr(al.pct_change(fill_method=None)).fillna(0.0)
        if name == "XU100": df["XU100_Ret20"] = al.pct_change(20, fill_method=None).fillna(0.0)
    df["RelStr20"] = df["Ret20"] - df["XU100_Ret20"]
    return df.replace([np.inf, -np.inf], 0.0)

def load_news_for(stock, dates):
    n = len(dates)
    return (np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32))

CROSS_RANK_COLS = ["RSI", "Ret5", "Ret20", "Vol_ratio", "RelStr20"]

def build_cross_rank(dfs):
    frames = []
    for s, d in dfs.items():
        if d is None or len(d) == 0: continue
        sub = d[["Date"] + CROSS_RANK_COLS].copy(); sub["stock"] = s
        frames.append(sub)
    if not frames: return {}
    all_df = pd.concat(frames, ignore_index=True)
    for col in CROSS_RANK_COLS:
        all_df[f"rk_{col}"] = all_df.groupby("Date")[col].rank(pct=True)
    out = {}
    for s, g in all_df.groupby("stock"):
        out[s] = g.sort_values("Date")[["Date"] + [f"rk_{c}" for c in CROSS_RANK_COLS]].reset_index(drop=True)
    return out

def inject_cross_rank(df, rank_df):
    if rank_df is None or len(rank_df) == 0:
        for c in CROSS_RANK_COLS: df[f"rk_{c}"] = 0.5
        return df
    df = df.merge(rank_df, on="Date", how="left")
    for c in CROSS_RANK_COLS: df[f"rk_{c}"] = df[f"rk_{c}"].fillna(0.5)
    return df

FEAT_NAMES = [
    "RSI","MACDh","SMA_trend","BB_pos","Stoch","ADX","Volatilite","Vol_ratio",
    "Mom10","HL_ratio","Gap","SMA200_uzak","EMA_trend","C>SMA20","C>SMA50",
    "MACD_poz","SMA20>50","WilliamsR","Mom30","C>BB_mid","ATR%","Ret5","HI120",
    "K-D_fark","W_RSI","W_Trend","W_Strength",
    "Reg_Bull","Reg_Bear","Reg_Vol","Reg_Range",
    "USDTRY_etki","XU100_etki","XU100_kor",
    "Ret20","RelStr20","Donchian20","RSI_egim","SMA50_egim","ATR_degisim",
] + (["News_Sent", "News_Vol", "News_Novelty"] if NEWS_ENABLED else []) + [
    "Turnover_log", "Liquidity_flag",
] + [f"Rank_{c}" for c in CROSS_RANK_COLS] + [f"Hisse_{s}" for s in STOCK_LIST]
assert len(FEAT_NAMES) == N_FEAT

def feature_matrix(df, stock_name=None):
    n = len(df)
    g = lambda c, d=0.0: df[c].to_numpy(np.float32) if c in df.columns else np.full(n, d, dtype=np.float32)
    div = lambda a, b: np.divide(a, b, out=np.zeros(n, dtype=np.float32), where=(b != 0) & np.isfinite(b))
    cl = np.clip
    c = g("Close"); sma20, sma50, sma200 = g("SMA20"), g("SMA50"), g("SMA200")
    bbm, bbu, bbd = g("BBm"), g("BBu"), g("BBd"); turn = g("Turnover20")
    f = [
        g("RSI", 50) / 100, cl(div(g("MACDh"), c) * 200, -1, 1),
        cl((div(sma20, sma50) - 1) * 10, -1, 1), cl(div(c - bbm, bbu - bbd + 1e-9), -1, 1),
        g("K", 50) / 100, cl(g("ADX") / 50, 0, 1), cl(g("Volatility") * 3, 0, 1),
        cl(g("Vol_ratio", 1) - 1, -1, 1), cl(g("Ret10") * 10, -1, 1),
        cl(g("HL") * 20, 0, 1), cl(g("Gap") * 20, -1, 1), cl((div(c, sma200) - 1) * 5, -1, 1),
        cl(div(g("EMA9") - g("EMA21"), c) * 20, -1, 1),
        (c > sma20).astype(np.float32), (c > sma50).astype(np.float32),
        (g("MACD") > g("MACDs")).astype(np.float32), (sma20 > sma50).astype(np.float32),
        cl(g("WILLR", -50) / 100, -1, 0), cl(g("Ret30") * 10, -1, 1),
        (c > bbm).astype(np.float32), cl(g("ATRp") * 20, 0, 1),
        cl(g("Ret5") * 10, -1, 1), cl(g("HI120") * 5, -1, 0),
        cl((g("K", 50) - g("D", 50)) / 30, -1, 1),
        cl(g("W_RSI", 50) / 100, 0, 1), cl(g("W_Trend") * 10, -1, 1),
        cl(g("W_Strength") * 5, -1, 1),
        g("Regime_Bull_sm"), g("Regime_Bear_sm"), g("Regime_Vol_sm"), g("Regime_Range_sm"),
        cl(g("USDTRY_Ret5") * 20, -1, 1), cl(g("XU100_Ret5") * 10, -1, 1), cl(g("XU100_Corr20"), -1, 1),
        cl(g("Ret20") * 8, -1, 1), cl(g("RelStr20") * 8, -1, 1), cl(g("Donch20", 0.5), 0, 1),
        cl(g("RSI_slope") / 30, -1, 1), cl(g("SMA50_slope") * 10, -1, 1), cl(g("ATR_chg"), -1, 1),
    ]
    if NEWS_ENABLED:
        ns, nv, nn_ = load_news_for(stock_name, df["Date"].to_numpy())
        f += [cl(ns, -1, 1), cl(nv, 0, 5), cl(nn_, 0, 1)]
    f += [cl(np.log1p(np.maximum(turn, 0)) / 20.0, 0, 1), (turn >= LIQ_FLAG_TL).astype(np.float32)]
    f += [cl(g(f"rk_{c_}", 0.5) * 2 - 1, -1, 1) for c_ in CROSS_RANK_COLS]
    M = np.nan_to_num(np.column_stack(f)).astype(np.float32)
    oh = np.zeros((n, N_STOCKS), dtype=np.float32)
    if stock_name in STOCK_LIST: oh[:, STOCK_LIST.index(stock_name)] = 1.0
    return np.hstack([M, oh]).astype(np.float32)

def regime_code(df):
    return np.column_stack([df["Regime_Bull_sm"], df["Regime_Bear_sm"],
                            df["Regime_Vol_sm"], df["Regime_Range_sm"]]).argmax(1)

BASE_KEYS = ["X", "y", "y_long", "f", "d", "lmin", "atrp", "rg", "a200", "turn"]
WIN_KEYS = ["fh", "fl", "fc"]
DS_KEYS = BASE_KEYS + WIN_KEYS

def _triple_barrier(cl_, hi, lo, atr, K, H):
    n = len(cl_); m = n - H
    if m <= 0: return None
    Hh = sliding_window_view(hi[1:], H)[:m]; Ll = sliding_window_view(lo[1:], H)[:m]
    e, a = cl_[:m], atr[:m]
    ok = np.isfinite(a) & (a > 0) & (e > 0)
    up, dn = e + K * a, e - K * a
    hu, hd = Hh >= up[:, None], Ll <= dn[:, None]
    fu = np.where(hu.any(1), hu.argmax(1), H); fd = np.where(hd.any(1), hd.argmax(1), H)
    y = np.where(fd <= fu, np.where(fd < H, 0, 1), 2).astype(np.int64)
    y[~ok] = 1
    es = np.where(e > 0, e, 1.0)
    ret = np.where(ok, cl_[H:H + m] / es - 1, 0.0)
    lmin = np.where(ok, Ll.min(1) / es - 1, 0.0)
    return y, ret.astype(np.float32), lmin.astype(np.float32)

def _fwd_window(arr, m, base):
    pad = np.concatenate([arr[1:], np.full(HW, np.nan)])
    W = sliding_window_view(pad, HW)[:m]
    return (W / np.where(base > 0, base, 1.0)[:, None] - 1.0).astype(np.float32)

def make_ds(df, stock=None):
    n = len(df)
    if n < HORIZON_LONG + 150: return None
    F = feature_matrix(df, stock)
    hi, lo = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    cl_, atr = df["Close"].to_numpy(float), df["ATR"].to_numpy(float)
    r_s = _triple_barrier(cl_, hi, lo, atr, TB_K, HORIZON)
    r_l = _triple_barrier(cl_, hi, lo, atr, TB_K_LONG, HORIZON_LONG)
    if r_s is None or r_l is None: return None
    y_s, f_s, lmin_s = r_s
    m = len(y_s)
    y_l = np.full(m, -1, dtype=np.int64); y_l[:len(r_l[0])] = r_l[0]
    e, a = cl_[:m], atr[:m]
    atrp = np.where(np.isfinite(a) & (a > 0) & (e > 0), a / np.where(e > 0, e, 1.0), 0.02).astype(np.float32)
    turn = df["Turnover20"].to_numpy(float)[:m] if "Turnover20" in df.columns else np.zeros(m)
    return {"X": F[:m], "y": y_s, "y_long": y_l, "f": f_s, "d": df["Date"].to_numpy()[:m],
            "lmin": lmin_s, "atrp": atrp, "rg": regime_code(df)[:m].astype(np.int64),
            "a200": (cl_ > df["SMA200"].to_numpy(float))[:m], "turn": turn.astype(np.float32),
            "fh": _fwd_window(hi, m, cl_[:m]), "fl": _fwd_window(lo, m, cl_[:m]),
            "fc": _fwd_window(cl_, m, cl_[:m])}

def merge_ds(lst):
    lst = [d for d in lst if d]
    if not lst: return None
    out = {k: np.concatenate([d[k] for d in lst]) for k in DS_KEYS}
    out["X"] = out["X"].astype(np.float32)
    return out

def truncate_windows(seg, ud, seg_end):
    avail = np.searchsorted(ud, seg_end, side="left") - np.searchsorted(ud, seg["d"], side="right")
    cut = np.arange(HW)[None, :] >= np.maximum(avail, 0)[:, None]
    for k in WIN_KEYS:
        w = seg[k].copy(); w[cut] = np.nan; seg[k] = w
    return seg

def split_ds(ds, val_frac=0.15, test_frac=0.15, purge=HORIZON_LONG):
    d0 = ds["d"]; ud = np.unique(d0)
    if len(ud) < 120: return None
    order = np.argsort(d0, kind="stable"); d = d0[order]
    n_u = len(ud)
    te_start = ud[int(n_u * (1 - test_frac))]
    va_start = ud[int(n_u * (1 - test_frac - val_frac))]
    gap = np.timedelta64(int(purge * 1.5) + 1, "D")
    va_end = te_start - gap
    trm = d < (va_start - gap); vam = (d >= va_start) & (d < va_end); tem = d >= te_start
    if trm.sum() < 300 or vam.sum() < 100 or tem.sum() < 100: return None
    take = lambda m, keys: {k: ds[k][order[m]] for k in keys}
    tr, va, te = take(trm, BASE_KEYS), take(vam, DS_KEYS), take(tem, DS_KEYS)
    va = truncate_windows(va, ud, va_end)
    ut = np.unique(tr["d"])
    wtr = (0.5 + np.searchsorted(ut, tr["d"]) / max(1, len(ut) - 1)).astype(np.float32)
    return {"tr": tr, "va": va, "te": te,
            "Xtr": tr["X"], "ytr": tr["y"], "ytr_long": tr["y_long"], "wtr": wtr,
            "Xva": va["X"], "yva": va["y"], "yva_long": va["y_long"],
            "Xte": te["X"], "yte": te["y"], "yte_long": te["y_long"]}

def _ln(z, g, b):
    mu = z.mean(1, keepdims=True); xc = z - mu
    std = np.sqrt((xc * xc).mean(1, keepdims=True) + 1e-5)
    xh = xc / std
    return g * xh + b, (xh, std)

def _ln_back(dy, c, g):
    xh, std = c; dxh = dy * g
    dz = (dxh - dxh.mean(1, keepdims=True) - xh * (dxh * xh).mean(1, keepdims=True)) / std
    return dz, (dy * xh).sum(0), dy.sum(0)

def _softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)

def _onehot(y, smooth=0.05, k=3):
    Y = np.full((len(y), k), smooth / k, dtype=np.float32)
    Y[np.arange(len(y)), y] += 1.0 - smooth
    return Y

def _onehot_m(y, smooth=0.05, k=3):
    y = np.asarray(y); v = y >= 0
    Y = np.full((len(y), k), 1.0 / k, dtype=np.float32)
    if v.any(): Y[v] = _onehot(y[v], smooth, k)
    return Y, v.astype(np.float32)

def _comb_loss_vec(ps, pl, y_s, y_l, alpha):
    n = len(y_s); ar = np.arange(n)
    ls = -np.log(ps[ar, y_s] + 1e-9)
    v = y_l >= 0
    ll = -np.log(pl[ar, np.where(v, y_l, 0)] + 1e-9)
    return np.where(v, alpha * ls + (1 - alpha) * ll, ls)

def fit_temp_logits(z, y, valid=None):
    if valid is not None: z, y = z[valid], y[valid]
    if len(y) < 10: return 1.0
    Ts = np.linspace(0.5, 3.0, 26, dtype=np.float32)
    zz = z[None, :, :] / Ts[:, None, None]
    zz = zz - zz.max(-1, keepdims=True)
    lse = np.log(np.exp(zz).sum(-1))
    ll = zz[:, np.arange(len(y)), y] - lse
    return float(Ts[int(np.argmax(ll.mean(1)))])

class NN:
    def __init__(self, n_in=N_FEAT, h=H_DIM, n_out=3, lr=0.003, dropout=0.15, wd=0.01,
                 seed=None, ema=0.98, alpha=ALPHA_LONG):
        rg = np.random.default_rng(seed)
        self.rng = np.random.default_rng(None if seed is None else seed + 991)
        f32 = np.float32
        def W(i, o, s=1.0): return (rg.standard_normal((i, o)) * np.sqrt(2.0 / i) * s).astype(f32)
        self.P = {
            "Att": np.ones(n_in, f32),
            "W0": W(n_in, h), "b0": np.zeros(h, f32), "g0": np.ones(h, f32), "e0": np.zeros(h, f32),
            "W1": W(h, h, 0.5), "b1": np.zeros(h, f32), "g1": np.ones(h, f32), "e1": np.zeros(h, f32),
            "W2": W(h, h, 0.5), "b2": np.zeros(h, f32), "g2": np.ones(h, f32), "e2": np.zeros(h, f32),
            "W3s": W(h, n_out, 0.5), "b3s": np.zeros(n_out, f32),
            "W3l": W(h, n_out, 0.5), "b3l": np.zeros(n_out, f32),
        }
        self.E = {k: v.copy() for k, v in self.P.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.lr, self.dropout, self.wd, self.ema, self.alpha = lr, dropout, wd, ema, alpha
        self.t, self.Ts, self.Tl, self.use_ema, self.swa_wins = 0, 1.0, 1.0, True, 0
    @staticmethod
    def _scale(att):
        a = np.abs(att); return a / (a.sum() + 1e-9) * len(a)
    @staticmethod
    def _fwd(P, X, drop=0.0, rng=None):
        xa = X * NN._scale(P["Att"])
        n0, c0 = _ln(xa @ P["W0"] + P["b0"], P["g0"], P["e0"]); h0 = np.maximum(n0, 0)
        n1, c1 = _ln(h0 @ P["W1"] + P["b1"], P["g1"], P["e1"]); r1 = np.maximum(n1, 0)
        k1 = None
        if drop > 0:
            k1 = ((rng.random(r1.shape) > drop) / (1 - drop)).astype(r1.dtype); r1 = r1 * k1
        h1 = h0 + r1
        n2, c2 = _ln(h1 @ P["W2"] + P["b2"], P["g2"], P["e2"]); r2 = np.maximum(n2, 0)
        k2 = None
        if drop > 0:
            k2 = ((rng.random(r2.shape) > drop) / (1 - drop)).astype(r2.dtype); r2 = r2 * k2
        h2 = h1 + r2
        return (h2 @ P["W3s"] + P["b3s"], h2 @ P["W3l"] + P["b3l"],
                (xa, n0, c0, h0, n1, c1, k1, h1, n2, c2, k2, h2))
    @staticmethod
    def loss_grad(P, X, Ys, Yl, ml, w=None, drop=0.0, rng=None, alpha=ALPHA_LONG):
        n = X.shape[0]
        z3s, z3l, ca = NN._fwd(P, X, drop, rng)
        ps, pl = _softmax(z3s), _softmax(z3l)
        sw = np.ones(n, dtype=ps.dtype) if w is None else np.asarray(w, dtype=ps.dtype)
        swl = sw * np.asarray(ml, dtype=ps.dtype)
        loss = float(alpha * -np.mean(sw * np.sum(Ys * np.log(ps + 1e-9), axis=1))
                     + (1 - alpha) * -np.mean(swl * np.sum(Yl * np.log(pl + 1e-9), axis=1)))
        dz3s = alpha * (ps - Ys) * (sw[:, None] / n)
        dz3l = (1 - alpha) * (pl - Yl) * (swl[:, None] / n)
        xa, n0, c0, h0, n1, c1, k1, h1, n2, c2, k2, h2 = ca
        G = {"W3s": h2.T @ dz3s, "b3s": dz3s.sum(0), "W3l": h2.T @ dz3l, "b3l": dz3l.sum(0)}
        dh2 = dz3s @ P["W3s"].T + dz3l @ P["W3l"].T
        dn2 = (dh2 if k2 is None else dh2 * k2) * (n2 > 0)
        dz2, G["g2"], G["e2"] = _ln_back(dn2, c2, P["g2"])
        G["W2"], G["b2"] = h1.T @ dz2, dz2.sum(0)
        dh1 = dh2 + dz2 @ P["W2"].T
        dn1 = (dh1 if k1 is None else dh1 * k1) * (n1 > 0)
        dz1, G["g1"], G["e1"] = _ln_back(dn1, c1, P["g1"])
        G["W1"], G["b1"] = h0.T @ dz1, dz1.sum(0)
        dh0 = dh1 + dz1 @ P["W1"].T
        dn0 = dh0 * (n0 > 0)
        dz0, G["g0"], G["e0"] = _ln_back(dn0, c0, P["g0"])
        G["W0"], G["b0"] = xa.T @ dz0, dz0.sum(0)
        dxa = dz0 @ P["W0"].T
        gs = (dxa * X).sum(0)
        a = np.abs(P["Att"]); S = a.sum() + 1e-9
        G["Att"] = np.sign(P["Att"]) * len(a) * (gs / S - (gs @ a) / S ** 2)
        return loss, G
    def _step(self, X, Ys, Yl, ml, w=None, lr_mult=1.0, clip=1.0):
        loss, G = self.loss_grad(self.P, X, Ys, Yl, ml, w, self.dropout, self.rng, self.alpha)
        gn = math.sqrt(sum(float((g * g).sum()) for g in G.values()))
        sc = min(1.0, clip / (gn + 1e-9))
        self.t += 1; lr = self.lr * lr_mult
        b1, b2 = 0.9, 0.999
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for k, g in G.items():
            g = g * sc
            m, v, p = self.m[k], self.v[k], self.P[k]
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            if self.wd and k[0] == "W": p *= (1 - lr * self.wd)
            p -= (lr * (m / c1) / (np.sqrt(v / c2) + 1e-8)).astype(np.float32)
            e = self.E[k]; e *= self.ema; e += (1 - self.ema) * p
        return loss
    def train_step(self, X, y_s, y_l, w=None, smooth=0.05, lr_mult=1.0):
        Yl, ml = _onehot_m(y_l, smooth)
        return self._step(X, _onehot(y_s, smooth), Yl, ml, w, lr_mult)
    def _infer_P(self):
        return self.E if (self.use_ema and self.t >= 30) else self.P
    def logits_pair(self, X, P=None):
        z3s, z3l, _ = self._fwd(P if P is not None else self._infer_P(), X)
        return z3s, z3l
    def proba_pair(self, X):
        zs, zl = self.logits_pair(np.atleast_2d(X))
        return (_softmax(zs / max(self.Ts, 1e-3)).astype(np.float32),
                _softmax(zl / max(self.Tl, 1e-3)).astype(np.float32))
    def evaluate(self, X, y_s, y_l=None):
        ps, pl = self.proba_pair(X)
        if y_l is None: y_l = np.full(len(y_s), -1, dtype=np.int64)
        loss = float(_comb_loss_vec(ps, pl, y_s, y_l, self.alpha).mean())
        v = y_l >= 0
        return loss, (float(np.mean(ps.argmax(1) == y_s)),
                      float(np.mean(pl.argmax(1)[v] == y_l[v])) if v.any() else 0.0)
    def distill_from(self, Ys, Yl, X, steps=5):
        ml = np.ones(len(X), dtype=np.float32)
        for _ in range(steps): self._step(X, Ys, Yl, ml, None, lr_mult=0.5)
    def snapshot(self):
        return {"P": {k: v.copy() for k, v in self.P.items()},
                "E": {k: v.copy() for k, v in self.E.items()}, "Ts": self.Ts, "Tl": self.Tl}
    def restore(self, s):
        self.P = {k: v.copy() for k, v in s["P"].items()}
        self.E = {k: v.copy() for k, v in s["E"].items()}
        self.Ts, self.Tl = s["Ts"], s["Tl"]
    def get_state(self):
        d = self.snapshot()
        d.update(m={k: v.copy() for k, v in self.m.items()}, v={k: v.copy() for k, v in self.v.items()},
                 t=self.t, lr=self.lr, dropout=self.dropout, wd=self.wd, swa_wins=self.swa_wins)
        return d
    def set_state(self, s):
        self.restore(s); self.m, self.v = s["m"], s["v"]
        self.t, self.lr, self.dropout, self.wd = s["t"], s["lr"], s["dropout"], s["wd"]
        self.swa_wins = s.get("swa_wins", 0)

def fit_temp_pair(nn, X, y_s, y_l):
    zs, zl = nn.logits_pair(X)
    return fit_temp_logits(zs, y_s), fit_temp_logits(zl, y_l, valid=y_l >= 0)

def fit_model(nn, Xtr, ytr_s, ytr_l, Xva=None, yva_s=None, yva_l=None, steps=300, bs=64,
              seed=None, warmup=20, sw=None, swa=True):
    rng = np.random.default_rng(seed)
    cnt = np.bincount(ytr_s, minlength=3).astype(float) + 1.0
    cw = ((len(ytr_s) / (3.0 * cnt)) ** 0.5).astype(np.float32)
    cdf = None if sw is None else np.cumsum(np.asarray(sw, dtype=np.float64))
    lr0 = nn.lr
    nn.Ts = nn.Tl = 1.0
    hv = Xva is not None and len(Xva) >= 20
    if hv and len(Xva) > 4000:
        sel = rng.choice(len(Xva), 4000, replace=False)
        Xe, ye_s, ye_l = Xva[sel], yva_s[sel], yva_l[sel]
    else:
        Xe, ye_s, ye_l = Xva, yva_s, yva_l
    bl, best = float("inf"), None
    bs = min(bs, len(Xtr))
    swa_start, swa_every = int(steps * 0.6), max(5, steps // 20)
    avg, navg = None, 0
    for s in range(steps):
        if s < warmup: nn.lr = lr0 * ((s + 1) / max(1, warmup))
        else:
            prog = (s - warmup) / max(1, steps - warmup)
            nn.lr = lr0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        if cdf is None: idx = rng.integers(0, len(Xtr), bs)
        else: idx = np.minimum(np.searchsorted(cdf, rng.random(bs) * cdf[-1]), len(Xtr) - 1)
        nn.train_step(Xtr[idx], ytr_s[idx], ytr_l[idx], w=cw[ytr_s[idx]])
        if hv and (s % 20 == 19 or s == steps - 1):
            vl, _ = nn.evaluate(Xe, ye_s, ye_l)
            if vl < bl: bl, best = vl, nn.snapshot()
        if swa and hv and s >= swa_start and (s - swa_start) % swa_every == 0:
            Pe = nn._infer_P()
            if avg is None: avg = {k: v.copy() for k, v in Pe.items()}
            else:
                for k, v in Pe.items(): avg[k] += (v - avg[k]) / (navg + 1)
            navg += 1
    nn.lr = lr0
    if best is not None: nn.restore(best)
    if avg is not None and navg >= 3:
        cur = nn.snapshot()
        nn.P = {k: v.copy() for k, v in avg.items()}
        nn.E = {k: v.copy() for k, v in avg.items()}
        vl_avg, _ = nn.evaluate(Xe, ye_s, ye_l)
        if vl_avg < bl * 0.999: nn.swa_wins += 1
        else: nn.restore(cur)
    if hv: nn.Ts, nn.Tl = fit_temp_pair(nn, Xe, ye_s, ye_l)
    else: nn.Ts = nn.Tl = 1.0
    return nn

class RouterNN:
    def __init__(self, n_in, n_bags, h=ROUTER_H, seed=None, lr=0.005, wd=0.001):
        rg = np.random.default_rng(seed); f32 = np.float32
        self.P = {"W1": (rg.standard_normal((n_in, h)) * 0.05).astype(f32), "b1": np.zeros(h, f32),
                  "W2": (rg.standard_normal((h, n_bags)) * 0.05).astype(f32), "b2": np.zeros(n_bags, f32)}
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.lr, self.wd, self.t, self.n_bags = lr, wd, 0, n_bags
    def forward(self, F):
        h1 = np.maximum(F @ self.P["W1"] + self.P["b1"], 0)
        return _softmax(h1 @ self.P["W2"] + self.P["b2"]), (F, h1)
    def backward(self, dg, cache):
        F, h1 = cache
        g = _softmax(h1 @ self.P["W2"] + self.P["b2"])
        dz = g * (dg - (dg * g).sum(1, keepdims=True))
        dh1 = (dz @ self.P["W2"].T) * (h1 > 0)
        return {"W1": F.T @ dh1, "b1": dh1.sum(0), "W2": h1.T @ dz, "b2": dz.sum(0)}
    def loss_grad(self, F, ps, pl, Ys, Yl, ml, alpha=ALPHA_LONG):
        g, cache = self.forward(F)
        Ps = np.einsum("nb,bnc->nc", g, ps); Pl = np.einsum("nb,bnc->nc", g, pl)
        n = len(F); eps = 1e-9
        loss = float(alpha * -np.mean(np.sum(Ys * np.log(Ps + eps), 1))
                     + (1 - alpha) * -np.mean(ml * np.sum(Yl * np.log(Pl + eps), 1)))
        dPs = -alpha * Ys / (Ps + eps) / n
        dPl = -(1 - alpha) * ml[:, None] * Yl / (Pl + eps) / n
        dg = np.einsum("nc,bnc->nb", dPs, ps) + np.einsum("nc,bnc->nb", dPl, pl)
        return loss, self.backward(dg, cache)
    def step(self, G):
        self.t += 1
        b1, b2 = 0.9, 0.999
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for k, g in G.items():
            m, v, p = self.m[k], self.v[k], self.P[k]
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            if self.wd and k[0] == "W": p *= (1 - self.lr * self.wd)
            p -= (self.lr * (m / c1) / (np.sqrt(v / c2) + 1e-8)).astype(np.float32)
    def get_state(self):
        return {"P": {k: v.copy() for k, v in self.P.items()}, "m": {k: v.copy() for k, v in self.m.items()},
                "v": {k: v.copy() for k, v in self.v.items()}, "t": self.t, "lr": self.lr, "wd": self.wd}
    def set_state(self, s):
        self.P = {k: v.copy() for k, v in s["P"].items()}
        self.m = {k: v.copy() for k, v in s["m"].items()}
        self.v = {k: v.copy() for k, v in s["v"].items()}
        self.t, self.lr, self.wd = s["t"], s["lr"], s["wd"]

def _mix(w, p):
    if w.shape[0] == 1: return np.tensordot(w[0], p, axes=1).astype(np.float32)
    return np.einsum("nb,bnc->nc", w, p).astype(np.float32)

class BaggedNN:
    def __init__(self, n=BAG_N, seed=BAG_SEED, replay_cap=5000):
        self.nets = [NN(seed=seed + i * 7) for i in range(n)]
        self.perf = np.ones(n, dtype=np.float32) / n
        self.weights = self.perf.copy()
        self.best_vl = float("inf"); self.plateau = 0; self.evo = 0
        self.pbt_acc = self.pbt_n = 0
        self.replay = ExperienceBuffer(replay_cap)
        self.router = RouterNN(N_STOCKS + 2 * n, n, seed=seed + 991) if USE_ROUTER else None
        self.router_on, self.router_gain = False, 0.0
    def _Z(self, X): return [nn.logits_pair(X) for nn in self.nets]
    def _temps(self): return [(nn.Ts, nn.Tl) for nn in self.nets]
    @staticmethod
    def _probs(Z, temps):
        ps = np.array([_softmax(z[0] / max(t[0], 1e-3)) for z, t in zip(Z, temps)], dtype=np.float32)
        pl = np.array([_softmax(z[1] / max(t[1], 1e-3)) for z, t in zip(Z, temps)], dtype=np.float32)
        return ps, pl
    def _net_losses(self, Z, temps, y_s, y_l):
        ps, pl = self._probs(Z, temps)
        return np.array([_comb_loss_vec(ps[i], pl[i], y_s, y_l, ALPHA_LONG).mean() for i in range(len(self.nets))])
    @staticmethod
    def _router_inputs(X, ps, pl):
        cs = ps[:, :, 2] - np.maximum(ps[:, :, 0], ps[:, :, 1])
        cl = pl[:, :, 2] - np.maximum(pl[:, :, 0], pl[:, :, 1])
        return np.hstack([X[:, -N_STOCKS:], cs.T, cl.T]).astype(np.float32)
    def _weights_for(self, X, ps, pl):
        if self.router is None or not self.router_on or X is None: return self.weights[None, :]
        g, _ = self.router.forward(self._router_inputs(X, ps, pl))
        return (1 - ROUTER_MIX) * self.weights[None, :] + ROUTER_MIX * g
    def _ens_eval(self, Z, temps, y_s, y_l, X=None):
        ps, pl = self._probs(Z, temps)
        w = self._weights_for(X, ps, pl)
        Ps, Pl = _mix(w, ps), _mix(w, pl)
        return (float(_comb_loss_vec(Ps, Pl, y_s, y_l, ALPHA_LONG).mean()),
                float(np.mean(Ps.argmax(1) == y_s)), Ps, Pl)
    def update_w_from_losses(self, losses):
        inv = (1.0 / (np.asarray(losses) + 1e-6)).astype(np.float32)
        self.perf = (0.8 * self.perf + 0.2 * inv / inv.sum()).astype(np.float32)
        self.perf /= self.perf.sum()
        self.weights = self.perf.copy()
    def update_w(self, X, y_s, y_l=None):
        if y_l is None: y_l = np.full(len(y_s), -1, dtype=np.int64)
        self.update_w_from_losses(self._net_losses(self._Z(X), self._temps(), y_s, y_l))
    def _stack(self, X):
        pr = [nn.proba_pair(X) for nn in self.nets]
        return (np.array([p[0] for p in pr], dtype=np.float32), np.array([p[1] for p in pr], dtype=np.float32))
    def predict_batch(self, X):
        ps, pl = self._stack(X)
        w = self._weights_for(X, ps, pl)
        return _mix(w, ps), _mix(w, pl)
    def predict_unc(self, X):
        ps, pl = self._stack(X)
        w = self._weights_for(X, ps, pl)
        return _mix(w, ps), _mix(w, pl), ps.std(axis=0).mean(axis=1)
    def evaluate(self, X, y_s, y_l=None):
        if y_l is None: y_l = np.full(len(y_s), -1, dtype=np.int64)
        vl, acc_s, _, Pl = self._ens_eval(self._Z(X), self._temps(), y_s, y_l, X=X)
        v = y_l >= 0
        return (vl, (acc_s, float(np.mean(Pl.argmax(1)[v] == y_l[v])) if v.any() else 0.0),
                float(np.bincount(y_s, minlength=3).max() / len(y_s)))
    def drift(self, X, y_s, y_l=None, frac=0.2):
        if y_l is None: y_l = np.full(len(y_s), -1, dtype=np.int64)
        ps, pl = self.predict_batch(X)
        l = _comb_loss_vec(ps, pl, y_s, y_l, ALPHA_LONG)
        k = int(len(l) * (1 - frac))
        return float(l[k:].mean() / (l[:k].mean() + 1e-9)) if k > 10 else 1.0
    def fit_router(self, sp, steps=150, bs=128, seed=7):
        if self.router is None: return None
        Xva, yva, ylv = sp["Xva"], sp["yva"], sp["yva_long"]
        k = int(len(Xva) * 0.6)
        if k < 200 or len(Xva) - k < 100: return None
        temps = self._temps()
        psA, plA = self._probs(self._Z(Xva[:k]), temps)
        FA = self._router_inputs(Xva[:k], psA, plA)
        YsA = _onehot(yva[:k], 0.05); YlA, mlA = _onehot_m(ylv[:k], 0.05)
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            idx = rng.integers(0, k, min(bs, k))
            _, G = self.router.loss_grad(FA[idx], psA[:, idx], plA[:, idx], YsA[idx], YlA[idx], mlA[idx])
            self.router.step(G)
        psB, plB = self._probs(self._Z(Xva[k:]), temps)
        yB, ylB = yva[k:], ylv[k:]
        st = float(_comb_loss_vec(_mix(self.weights[None, :], psB), _mix(self.weights[None, :], plB), yB, ylB, ALPHA_LONG).mean())
        self.router_on = True
        w = self._weights_for(Xva[k:], psB, plB)
        rt = float(_comb_loss_vec(_mix(w, psB), _mix(w, plB), yB, ylB, ALPHA_LONG).mean())
        self.router_gain = st - rt
        self.router_on = bool(rt < st * 0.998)
        return {"static": st, "router": rt, "on": self.router_on}
    def fit_all(self, Xtr, ytr_s, ytr_l, Xva=None, yva_s=None, yva_l=None, steps=300, seed=42, sw=None):
        def job(i):
            nn = self.nets[i]
            rg = np.random.default_rng(seed + i * 7)
            idx = rg.integers(0, len(Xtr), int(0.85 * len(Xtr)))
            Xb, ybs, ybl = Xtr[idx], ytr_s[idx], ytr_l[idx]
            wb = None if sw is None else sw[idx]
            s = self.replay.sample(300, rg) if self.replay.size() >= 200 else None
            if s is not None:
                Xb = np.vstack([Xb, s[0]])
                ybs = np.concatenate([ybs, s[1]]); ybl = np.concatenate([ybl, s[2]])
                if wb is not None:
                    wb = np.concatenate([wb, np.full(len(s[1]), float(wb.mean()), np.float32)])
            fit_model(nn, Xb, ybs, ybl, Xva, yva_s, yva_l, steps=steps, seed=seed + i * 7, sw=wb)
        with ThreadPoolExecutor(max_workers=max(1, min(len(self.nets), os.cpu_count() or 2))) as ex:
            list(ex.map(job, range(len(self.nets))))
    def distill(self, X, y_s, y_l):
        bi = int(np.argmax(self.perf)); t = self.nets[bi]
        zs, zl = t.logits_pair(X)
        soft_s = _softmax(zs / t.Ts / 2.0).astype(np.float32)
        soft_l = _softmax(zl / t.Tl / 2.0).astype(np.float32)
        Ys = 0.5 * soft_s + 0.5 * _onehot(y_s, 0.05)
        Yh, mv = _onehot_m(y_l, 0.05)
        Yl = np.where(mv[:, None] > 0, 0.5 * soft_l + 0.5 * Yh, soft_l).astype(np.float32)
        for i, nn in enumerate(self.nets):
            if i != bi and self.perf[i] < 0.9 * self.perf[bi]:
                nn.distill_from(Ys, Yl, X)
    def evolve_step(self, sp, n_batches=6, lr_mult=0.5):
        self.evo += 1
        Xtr, ytr_s, ytr_l, wtr = sp["Xtr"], sp["ytr"], sp["ytr_long"], sp.get("wtr")
        Xva, yva_s, yva_l = sp["Xva"], sp["yva"], sp["yva_long"]
        rg = np.random.default_rng(self.evo * 31 + 5)
        if len(Xva) > 2500:
            sel = rg.choice(len(Xva), 2500, replace=False)
            Xva, yva_s, yva_l = Xva[sel], yva_s[sel], yva_l[sel]
        snaps = [nn.snapshot() for nn in self.nets]
        Z0, t0 = self._Z(Xva), self._temps()
        vl0, _, _, _ = self._ens_eval(Z0, t0, yva_s, yva_l, X=Xva)
        cdf = None if wtr is None else np.cumsum(wtr.astype(np.float64))
        cnt = np.bincount(ytr_s, minlength=3) + 1.0
        cw = ((len(ytr_s) / (3.0 * cnt)) ** 0.5).astype(np.float32)
        for _ in range(n_batches):
            for nn in self.nets:
                if cdf is None: idx = rg.integers(0, len(Xtr), 64)
                else: idx = np.minimum(np.searchsorted(cdf, rg.random(64) * cdf[-1]), len(Xtr) - 1)
                Xb, ybs, ybl = Xtr[idx], ytr_s[idx], ytr_l[idx]
                s = self.replay.sample(24, rg)
                if s is not None:
                    Xb = np.vstack([Xb, s[0]])
                    ybs = np.concatenate([ybs, s[1]]); ybl = np.concatenate([ybl, s[2]])
                nn.train_step(Xb, ybs, ybl, w=cw[ybs], lr_mult=lr_mult)
        Z1 = self._Z(Xva)
        t1 = [(fit_temp_logits(z[0], yva_s), fit_temp_logits(z[1], yva_l, valid=yva_l >= 0)) for z in Z1]
        vl1, va1, _, _ = self._ens_eval(Z1, t1, yva_s, yva_l, X=Xva)
        rolled = vl1 > vl0 * 1.02
        if rolled:
            for nn, s in zip(self.nets, snaps): nn.restore(s)
            Z, tt = Z0, t0
            vl1, va1, _, _ = self._ens_eval(Z0, t0, yva_s, yva_l, X=Xva)
        else:
            Z, tt = Z1, t1
            for nn, (a, b) in zip(self.nets, t1): nn.Ts, nn.Tl = a, b
        self.update_w_from_losses(self._net_losses(Z, tt, yva_s, yva_l))
        if self.evo % 3 == 0:
            sel = rg.choice(len(Xtr), min(400, len(Xtr)), replace=False)
            self.distill(Xtr[sel], ytr_s[sel], ytr_l[sel])
        if vl1 < self.best_vl * 0.98: self.best_vl, self.plateau = vl1, 0
        else: self.plateau += 1
        db_log_train(self.evo, vl1, va1, "rollback" if rolled else "")
        return {"vl": vl1, "va": va1, "rolled": bool(rolled), "step": self.evo}
    def pbt_step(self, sp, steps=60):
        rg = np.random.default_rng(self.evo * 13 + 7)
        Xva, yva_s, yva_l = sp["Xva"], sp["yva"], sp["yva_long"]
        if len(Xva) > 3000:
            sel = rg.choice(len(Xva), 3000, replace=False)
            Xva, yva_s, yva_l = Xva[sel], yva_s[sel], yva_l[sel]
        losses = self._net_losses(self._Z(Xva), self._temps(), yva_s, yva_l)
        bi, wi = int(losses.argmin()), int(losses.argmax())
        if bi == wi: return None
        nn, best = self.nets[wi], self.nets[bi]
        old, old_hp, old_t, old_loss = nn.snapshot(), (nn.lr, nn.dropout, nn.wd), nn.t, float(losses[wi])
        old_m, old_v = nn.m, nn.v
        nn.restore(best.snapshot())
        nn.m = {k: v.copy() for k, v in best.m.items()}
        nn.v = {k: v.copy() for k, v in best.v.items()}
        nn.t = best.t
        nn.lr = clamp(best.lr * float(rg.choice([0.7, 1.0, 1.4])), 5e-4, 8e-3)
        nn.dropout = clamp(best.dropout + float(rg.choice([-0.05, 0.0, 0.05])), 0.0, 0.4)
        nn.wd = clamp(best.wd * float(rg.choice([0.5, 1.0, 2.0])), 1e-3, 0.1)
        for k in nn.P:
            if k[0] == "W":
                nn.P[k] += (rg.standard_normal(nn.P[k].shape) * 0.02 * nn.P[k].std()).astype(np.float32)
                nn.E[k] = nn.P[k].copy()
        fit_model(nn, sp["Xtr"], sp["ytr"], sp["ytr_long"], Xva, yva_s, yva_l, steps=steps,
                  seed=int(rg.integers(1 << 30)), warmup=5, sw=sp.get("wtr"))
        new_loss = float(nn.evaluate(Xva, yva_s, yva_l)[0])
        ok = new_loss < old_loss * 0.995
        self.pbt_n += 1
        hp = {"lr": round(nn.lr, 5), "dropout": round(nn.dropout, 2), "wd": round(nn.wd, 4)}
        if ok:
            self.pbt_acc += 1; losses[wi] = new_loss
        else:
            nn.restore(old); nn.lr, nn.dropout, nn.wd = old_hp
            nn.t, nn.m, nn.v = old_t, old_m, old_v
        self.update_w_from_losses(losses)
        return {"accepted": bool(ok), "slot": wi, "old": old_loss, "new": new_loss, "hp": hp}
    def get_state(self):
        return {"nets": [nn.get_state() for nn in self.nets], "perf": self.perf.tolist(),
                "weights": self.weights.tolist(), "bvl": self.best_vl, "plateau": self.plateau,
                "evo": self.evo, "pbt_acc": self.pbt_acc, "pbt_n": self.pbt_n,
                "router": self.router.get_state() if self.router else None,
                "router_on": self.router_on, "router_gain": self.router_gain}
    def set_state(self, s):
        for nn, ns in zip(self.nets, s["nets"]): nn.set_state(ns)
        self.perf = np.array(s["perf"], dtype=np.float32)
        self.weights = np.array(s["weights"], dtype=np.float32)
        self.best_vl, self.plateau, self.evo = s["bvl"], s["plateau"], s["evo"]
        self.pbt_acc, self.pbt_n = s.get("pbt_acc", 0), s.get("pbt_n", 0)
        if self.router is not None and s.get("router") is not None: self.router.set_state(s["router"])
        self.router_on, self.router_gain = s.get("router_on", False), s.get("router_gain", 0.0)

def regime_of(row):
    vals = [safe_float(row.get("Regime_Bull_sm", 0)), safe_float(row.get("Regime_Bear_sm", 0)),
            safe_float(row.get("Regime_Vol_sm", 0)), safe_float(row.get("Regime_Range_sm", 0))]
    return REGIMES[int(np.argmax(vals))]
def regime_adj(rp):
    return np.array([rp.get("regime_bull_adj", 0), rp.get("regime_bear_adj", 0),
                     rp.get("regime_vol_adj", 0), rp.get("regime_range_adj", 0)], dtype=np.float32)
def signal_from_probs(p_s, p_l, rp, regime=None):
    code = REGIMES.index(regime) if regime in REGIMES else None
    pb = rp["p_buy"]; pb_l = rp.get("p_buy_long", 0.40); ps = rp["p_sell"]
    if code is not None:
        a = regime_adj(rp)[code]; pb += a; pb_l += a * 0.5; ps -= a * 0.5
    long_bull = p_l[2] >= pb_l; long_bear = p_l[0] >= ps
    short_al = p_s[2] >= pb and p_s[2] - max(p_s[0], p_s[1]) >= rp["margin"]
    if p_s[0] >= ps or long_bear: return "SAT"
    if rp.get("use_long_gate", True) and not long_bull: return "TUT"
    if short_al and long_bull: return "AL"
    return "TUT"
def conf_mult_vec(Ps, Pl, rp):
    cs = Ps[:, 2] - np.maximum(Ps[:, 0], Ps[:, 1])
    cl = Pl[:, 2] - np.maximum(Pl[:, 0], Pl[:, 1])
    base = rp.get("margin", 0.10); d = max(1e-9, 1.0 - base)
    rs, rl = np.maximum(0.0, (cs - base) / d), np.maximum(0.0, (cl - base) / d)
    return 0.6 + 0.8 * np.minimum(1.0, (rs * 0.6 + rl * 0.4) * 2.0)
def conf_mult(p_s, p_l, rp):
    return float(conf_mult_vec(np.asarray(p_s)[None], np.asarray(p_l)[None], rp)[0])
def size_pos(eq, cash, expo, px, atr, probs_s, probs_l, rp):
    sd = rp["sl_atr"] * atr
    if sd <= 0 or px <= 0 or not np.isfinite(sd): return 0, sd
    cm = conf_mult(probs_s, probs_l, rp) if rp.get("conf_sizing", True) else 1.0
    room = max(0.0, rp["max_exposure"] * eq - expo)
    q = int(min(eq * rp["risk_per_trade"] * cm / sd, eq * rp["max_pos"] / px,
                room / px, cash / (px * (1 + FEE))))
    return max(q, 0), sd
def stop_level(entry, high, sd0, atr, rp):
    stop, trail = entry - sd0, False
    if high >= entry + rp["trail_act_r"] * sd0:
        ts = high - rp["trail_atr"] * atr
        if ts > stop: stop, trail = ts, True
    return stop, trail

def make_V(hold, Ps, Pl, U):
    ud, di = np.unique(hold["d"], return_inverse=True)
    V = dict(hold); V.update(Ps=Ps, Pl=Pl, U=U, di=di, nd=len(ud))
    V["conf_s"] = (Ps[:, 2] - np.maximum(Ps[:, 0], Ps[:, 1])).astype(np.float32)
    fh, fl, fc = hold["fh"], hold["fl"], hold["fc"]
    V["cmax"] = np.maximum.accumulate(np.where(np.isfinite(fh), 1.0 + fh, -np.inf), axis=1)
    V["lo1"], V["cl1"] = 1.0 + fl, 1.0 + fc
    V["nval"] = np.isfinite(fl).sum(1)
    return V
def sim_signals(V, rp):
    Ps, Pl = V["Ps"], V["Pl"]
    adj = regime_adj(rp)[V["rg"]]
    m = ((Ps[:, 2] >= rp["p_buy"] + adj) & (V["conf_s"] >= rp["margin"])
         & (Pl[:, 2] >= rp.get("p_buy_long", 0.40) + adj * 0.5) & (V["U"] <= rp["unc_max"])
         & (V["nval"] >= 1) & (V["turn"] >= rp.get("min_daily_turnover", 0)))
    if rp["use_regime"]: m &= V["a200"]
    return m
def simulate_entries(V, idx, rp):
    k = len(idx)
    Hc = int(np.clip(round(rp["max_hold"] * 5 / 7), 1, HW))
    cm, lo, cl = V["cmax"][idx, :Hc], V["lo1"][idx, :Hc], V["cl1"][idx, :Hc]
    nval = np.minimum(V["nval"][idx], Hc)
    atr = V["atrp"][idx]
    e = 1.0 + SLIPPAGE
    sd0 = rp["sl_atr"] * atr
    stop_fixed = (e - sd0)[:, None]
    hwm_prev = np.empty_like(cm); hwm_prev[:, 0] = e
    if Hc > 1: hwm_prev[:, 1:] = np.maximum(e, cm[:, :-1])
    trail_on = hwm_prev >= (e + rp["trail_act_r"] * sd0)[:, None]
    stop = np.where(trail_on, np.maximum(stop_fixed, hwm_prev - (rp["trail_atr"] * atr)[:, None]), stop_fixed)
    valid = np.arange(Hc)[None, :] < nval[:, None]
    hit = valid & (lo <= stop)
    anyh, first = hit.any(1), hit.argmax(1)
    ar = np.arange(k)
    ex = np.where(anyh, stop[ar, first], cl[ar, np.maximum(nval - 1, 0)])
    return (ex * (1 - SLIPPAGE) * (1 - FEE) / (e * (1 + FEE)) - 1.0).astype(np.float32)
def run_sim(V, rp):
    idx = np.flatnonzero(sim_signals(V, rp))
    if len(idx) == 0: return idx, None, None
    rets = simulate_entries(V, idx, rp)
    cm = conf_mult_vec(V["Ps"][idx], V["Pl"][idx], rp) if rp.get("conf_sizing", True) else 1.0
    sd = np.maximum(rp["sl_atr"] * V["atrp"][idx], 1e-6)
    frac = np.minimum(rp["risk_per_trade"] * cm / sd, rp["max_pos"])
    di, nd = V["di"][idx], V["nd"]
    dayfrac = np.bincount(di, weights=frac, minlength=nd)
    scale = np.minimum(1.0, rp["max_exposure"] / np.maximum(dayfrac[di], 1e-9))
    daily = np.bincount(di, weights=frac * scale * rets, minlength=nd)
    return idx, rets, daily
def score_params(V, rp, min_n=20):
    idx, rets, daily = run_sim(V, rp)
    n = len(idx)
    st_ = {"n": n, "gross": 0.0, "net": 0.0, "prec": 0.0, "dd": 0.0, "wr": 0.0}
    if n == 0: return -5.0, st_
    st_.update(net=float(rets.mean()), gross=float(rets.mean() + RT_COST), wr=float((rets > 0).mean()),
               prec=float((V["y"][idx] == 2).mean()))
    if n < min_n: return -5.0 + n / max(1, min_n), st_
    cum = np.cumsum(daily)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum))
    t = float(daily.mean() / (daily.std() + 1e-9) * math.sqrt(V["nd"]))
    st_["dd"] = dd
    return t - 20.0 * dd, st_

def _norm_cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def _norm_ppf(p):
    if p <= 0: return -8.0
    if p >= 1: return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
def deflated_sharpe(returns, n_trials):
    r = np.asarray(returns, dtype=np.float64); r = r[np.isfinite(r)]
    n = len(r)
    if n < 30: return 0.0, 0.0, 0.0
    mu, sdv = r.mean(), r.std(ddof=1)
    if sdv < 1e-12: return 0.0, 0.0, 0.0
    sr = mu / sdv
    z = (r - mu) / sdv
    g3, g4 = float((z ** 3).mean()), float((z ** 4).mean())
    N = max(int(n_trials), 2); em = 0.5772156649
    sr0 = math.sqrt(1.0 / (n - 1)) * ((1 - em) * _norm_ppf(1 - 1.0 / N) + em * _norm_ppf(1 - 1.0 / (N * math.e)))
    den = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2, 1e-6))
    dsr = float(_norm_cdf((sr - sr0) * math.sqrt(n - 1) / den))
    return dsr, float(sr * math.sqrt(252)), float(sr0 * math.sqrt(252))

def _sub(hold, mask):
    return {k: v[mask] for k, v in hold.items()}
def tune_risk(hold, bag, rp, budget=3.0, seed=0, min_n=20, hist=None, hold_te=None, max_evals=800):
    Ps, Pl, U = bag.predict_unc(hold["X"])
    d = hold["d"]; ud = np.unique(d)
    if len(ud) < 30: return None
    cut = ud[int(len(ud) * 0.6)]
    ma = d < cut; mb = ~ma
    if ma.sum() < 200 or mb.sum() < 100: return None
    segA = truncate_windows(_sub(hold, ma), ud, cut)
    VA = make_V(segA, Ps[ma], Pl[ma], U[ma])
    VB = make_V(_sub(hold, mb), Ps[mb], Pl[mb], U[mb])
    nA, nB = min_n, max(8, int(min_n * 0.6))
    rng = np.random.default_rng(seed)
    full = lambda c: {**rp, **c}
    cur = {k: float(rp[k]) for k in TUNE_BOUNDS}
    curA, _ = score_params(VA, full(cur), nA)
    curB, _ = score_params(VB, full(cur), nB)
    best, bestA = dict(cur), curA
    for h in (hist or []):
        hc = {k: float(h[k]) for k in TUNE_BOUNDS if k in h}
        if len(hc) == len(TUNE_BOUNDS):
            s, _ = score_params(VA, full(hc), nA)
            if s > bestA: best, bestA = hc, s
    sig = {k: 0.12 * (hi - lo) for k, (lo, hi) in TUNE_BOUNDS.items()}
    keys = list(TUNE_BOUNDS)
    t0, evals, tries, succ = time.time(), 0, 0, 0
    while evals < max_evals and time.time() - t0 < budget:
        cand, nm = dict(best), 0
        for k in keys:
            if rng.random() < 0.4:
                lo, hi = TUNE_BOUNDS[k]
                cand[k] = float(clamp(best[k] + rng.normal() * sig[k], lo, hi)); nm += 1
        if nm == 0:
            k = keys[int(rng.integers(len(keys)))]; lo, hi = TUNE_BOUNDS[k]
            cand[k] = float(clamp(best[k] + rng.normal() * sig[k], lo, hi))
        sc, _ = score_params(VA, full(cand), nA); evals += 1; tries += 1
        if sc > bestA + 1e-9: best, bestA = cand, sc; succ += 1
        if tries == 25:
            f = 1.3 if succ / 25 > 0.2 else 0.8
            for k, (lo, hi) in TUNE_BOUNDS.items():
                sig[k] = clamp(sig[k] * f, 0.01 * (hi - lo), 0.3 * (hi - lo))
            tries = succ = 0
    bestB, stB = score_params(VB, full(best), nB)
    dsr, sr, sr0 = 0.0, 0.0, 0.0
    if USE_DSR:
        _, _, daily = run_sim(VB, full(best))
        if daily is not None: dsr, sr, sr0 = deflated_sharpe(daily, evals + 1)
    curT = bestT = None; stT = None
    if hold_te is not None and len(hold_te["d"]) >= 100:
        Pt, Plt, Ut = bag.predict_unc(hold_te["X"])
        VT = make_V(hold_te, Pt, Plt, Ut)
        curT, _ = score_params(VT, full(cur), 8)
        bestT, stT = score_params(VT, full(best), 8)
    changed = any(abs(best[k] - cur[k]) > 1e-9 for k in keys)
    improved = bool(changed and stB["n"] >= nB and bestB > curB + 0.25 and bestA > curA + 0.25 and stB["net"] > 0)
    return {"cur": cur, "best": best, "curA": curA, "bestA": bestA, "curB": curB, "bestB": bestB,
            "evals": evals, "improved": improved, "stB": stB, "dsr": dsr, "sr": sr, "sr0": sr0,
            "curT": curT, "bestT": bestT, "stT": stT}

def _near(a, b, tol=0.25):
    return all(abs(a[k] - b[k]) <= tol * (TUNE_BOUNDS[k][1] - TUNE_BOUNDS[k][0]) for k in TUNE_BOUNDS)
class SelfImprover:
    def __init__(self):
        self.history, self.rp_hist = [], []
        self.last_run, self.last_data, self.runs = 0.0, None, 0
        self.proposal, self.drift = None, 1.0
        self.consec_improve, self.pending_rp = 0, None
    def due(self, now_ts, data_date, interval_min):
        return (self.last_data != data_date) or (now_ts - self.last_run >= interval_min * 60)
    @staticmethod
    def calibrate(bag, X, y_s):
        if len(y_s) < 30: return None
        ps, _ = bag.predict_batch(X)
        p_al = ps[:, 2]; y_al = (y_s == 2).astype(np.float32)
        brier = float(np.mean((p_al - y_al) ** 2))
        bins = np.linspace(0, 1, 11); ece = 0.0; n = len(p_al); cal = []
        for i in range(10):
            m = (p_al >= bins[i]) & ((p_al < bins[i + 1]) if i < 9 else (p_al <= bins[i + 1]))
            if m.sum() > 0:
                ece += m.sum() / n * abs(p_al[m].mean() - y_al[m].mean())
                cal.append({"bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}", "pred": float(p_al[m].mean()),
                            "actual": float(y_al[m].mean()), "n": int(m.sum())})
        return {"brier": brier, "ece": float(ece), "acc": float(np.mean(ps.argmax(1) == y_s)),
                "base": float(np.bincount(y_s, minlength=3).max() / len(y_s)), "cal": cal, "n": n}
    def run(self, bot, datasets, rp, budget=8.0, reason="auto"):
        t0 = time.time(); merged = merge_ds(datasets)
        sp = split_ds(merged) if merged else None
        rep = {"ts": _ts(), "reason": reason}
        if sp is None:
            rep["error"] = "veri yetersiz"; return None, rep
        nn = bot.nn
        rep["drift"] = self.drift = nn.drift(sp["Xva"], sp["yva"], sp["yva_long"])
        vl0, va0, _ = nn.evaluate(sp["Xva"], sp["yva"], sp["yva_long"])
        steps = rolled = 0
        n_steps = 12 if rep["drift"] > 1.15 else 6
        while time.time() - t0 < budget * 0.45 and steps < n_steps:
            r = nn.evolve_step(sp); steps += 1; rolled += int(r["rolled"])
        pb = nn.pbt_step(sp) if time.time() - t0 < budget * 0.65 else None
        rr = nn.fit_router(sp)
        vl1, va1, _ = nn.evaluate(sp["Xva"], sp["yva"], sp["yva_long"])
        rep.update(vl0=vl0, vl1=vl1, va0=va0, va1=va1, steps=steps, rolled=rolled,
                   pbt=("kabul" if pb and pb["accepted"] else "red" if pb else "-"),
                   router=("açık" if nn.router_on else "kapalı") if rr else "-")
        te_loss, te_acc, te_base = nn.evaluate(sp["Xte"], sp["yte"], sp["yte_long"])
        rep.update(te_loss=te_loss, te_acc=te_acc[0], te_base=te_base, gen_gap=te_loss / max(vl1, 1e-9))
        cal = self.calibrate(nn, sp["Xte"], sp["yte"])
        if cal:
            rep.update(brier=cal["brier"], ece=cal["ece"])
            db_calib_add(cal["n"], cal["brier"], cal["ece"], cal["acc"], cal["base"])
        bot.set_hold(sp); bot.set_ref(sp); bot.reeval(rp)
        new_rp, adopted = None, False
        tr = tune_risk(bot.hold, nn, rp, budget=min(max(1.0, budget - (time.time() - t0)), 5.0),
                       seed=nn.evo, min_n=20, hist=self.rp_hist, hold_te=bot.hold_te)
        if tr:
            rep.update(scoreB0=tr["curB"], scoreB1=tr["bestB"], evals=tr["evals"], dsr=tr["dsr"], sr=tr["sr"])
            if tr["bestT"] is not None: rep.update(scoreT0=tr["curT"], scoreT1=tr["bestT"])
            dsr_ok = (not USE_DSR) or tr["dsr"] >= rp.get("dsr_confidence", 0.9)
            te_ok = True
            if tr["stT"] is not None:
                te_ok = (tr["stT"]["n"] >= 8 and tr["stT"]["net"] > 0
                         and tr["bestT"] >= tr["curT"] - rp.get("test_score_tol", 0.25))
            if tr["improved"] and dsr_ok and te_ok:
                chg = {k: round(tr["best"][k], 4) for k in TUNE_BOUNDS if abs(tr["best"][k] - tr["cur"][k]) > 1e-9}
                rep["changes"] = chg
                need = int(rp.get("consec_improve", 2))
                if self.pending_rp is not None and _near(self.pending_rp, tr["best"]): self.consec_improve += 1
                else: self.consec_improve = 1
                self.pending_rp = dict(tr["best"])
                if self.consec_improve >= need and rp.get("auto_adopt", True):
                    new_rp = {**rp, **{k: float(v) for k, v in tr["best"].items()}}
                    self.rp_hist = (self.rp_hist + [dict(tr["cur"])])[-5:]
                    adopted, self.proposal = True, None
                    self.consec_improve, self.pending_rp = 0, None
                else:
                    self.proposal = {"params": tr["best"], "gain": tr["bestB"] - tr["curB"], "ts": rep["ts"],
                                     "wait": max(0, need - self.consec_improve)}
            else:
                rep["dsr_reject"] = not dsr_ok; rep["te_reject"] = not te_ok
                self.consec_improve, self.pending_rp = 0, None
        rep["adopted"] = adopted; rep["sec"] = round(time.time() - t0, 1)
        self.history = (self.history + [rep])[-40:]
        self.runs += 1
        db_improve_add(reason, f"drift={rep['drift']:.2f} steps={steps} pbt={rep['pbt']} router={rep['router']} chg={rep.get('changes', {})}",
                       tr["curB"] if tr else vl0, tr["bestB"] if tr else vl1, adopted)
        return new_rp, rep
    def get_state(self):
        return {"history": self.history, "rp_hist": self.rp_hist, "last_run": self.last_run,
                "last_data": self.last_data, "runs": self.runs, "proposal": self.proposal,
                "drift": self.drift, "consec_improve": self.consec_improve, "pending_rp": self.pending_rp}
    def set_state(self, s):
        for k in ("history","rp_hist","last_run","last_data","runs","proposal","drift","consec_improve","pending_rp"):
            if k in s: setattr(self, k, s[k])

class Bot:
    def __init__(self, cash=100000.0):
        self.cash = self.initial = cash
        self.positions, self.trades = {}, []
        self.nn = BaggedNN(); self.si = SelfImprover()
        self.wins = self.losses = 0
        self.eq_hist = []; self.peak_eq = cash
        self.halted, self.halt_reason, self.halt_until, self.streak = False, "", None, 0
        self.day_key, self.day_eq0, self.day_trades = None, cash, 0
        self.pretrained, self.val_stats, self.hold, self.hold_te, self.ref = False, None, None, None, None
        self.last_exit = {}
    def total_value(self, prices):
        return self.cash + sum(p["qty"] * prices.get(s, p["entry"]) for s, p in self.positions.items())
    def exposure(self, prices):
        return sum(p["qty"] * prices.get(s, p["entry"]) for s, p in self.positions.items())
    def record(self, v):
        if not self.eq_hist or abs(self.eq_hist[-1] - v) > 1e-6:
            self.eq_hist = (self.eq_hist + [float(v)])[-1000:]
    def win_rate(self):
        t = self.wins + self.losses; return self.wins / t if t else 0.0
    def set_hold(self, sp):
        self.hold = {k: sp["va"][k][-HOLD_N:].copy() for k in DS_KEYS}
        self.hold_te = {k: sp["te"][k][-HOLD_N:].copy() for k in DS_KEYS}
    def set_ref(self, sp):
        self.ref = (sp["Xtr"].mean(0), sp["Xtr"].std(0) + 1e-6)
    def feat_drift(self, F):
        if not self.ref: return 0.0
        mu, sd = self.ref; k = N_FEAT_TECH
        return float(np.mean(np.minimum(np.abs(F[:, :k].mean(0) - mu[:k]) / sd[:k], 5.0)))
    def pretrain(self, dss_list, rp, steps=400):
        merged = merge_ds(dss_list); sp = split_ds(merged) if merged else None
        if sp is None: return None
        k = min(2000, len(sp["Xtr"]))
        sel = np.random.default_rng(1).choice(len(sp["Xtr"]), k, replace=False)
        self.nn.replay.add_many(sp["Xtr"][sel], sp["ytr"][sel], sp["ytr_long"][sel])
        self.nn.fit_all(sp["Xtr"], sp["ytr"], sp["ytr_long"], sp["Xva"], sp["yva"], sp["yva_long"],
                        steps=steps, sw=sp["wtr"])
        self.nn.update_w(sp["Xva"], sp["yva"], sp["yva_long"])
        self.set_hold(sp); self.set_ref(sp)
        for _ in range(3): self.nn.evolve_step(sp)
        self.nn.fit_router(sp)
        self.reeval(rp); self.pretrained = True
        return self.val_stats
    def reeval(self, rp):
        if not self.hold: return
        Ps, Pl, U = self.nn.predict_unc(self.hold["X"])
        sc, st_ = score_params(make_V(self.hold, Ps, Pl, U), rp, min_n=1)
        y = self.hold["y"]
        self.val_stats = {"loss": float(-np.mean(np.log(Ps[np.arange(len(y)), y] + 1e-9))),
                          "acc": float(np.mean(Ps.argmax(1) == y)),
                          "base": float(np.bincount(y, minlength=3).max() / len(y)),
                          "al_n": st_["n"], "al_ret": st_["gross"], "al_prec": st_["prec"],
                          "net": st_["net"], "dd": st_["dd"], "score": sc, "n": int(len(y)), "wr": st_["wr"]}
    def edge_ok(self, rp):
        vs = self.val_stats
        return bool(vs and vs["al_n"] >= rp["gate_min_n"] and vs["al_ret"] > RT_COST * 1.5)
    def scan(self, names, F, rows, rp):
        Ps, Pl, U = self.nn.predict_unc(F)
        gate_ok = self.edge_ok(rp); out = {}
        for i, nm in enumerate(names):
            p_s, p_l, u, row = Ps[i], Pl[i], float(U[i]), rows[nm]
            reg = regime_of(row)
            act = signal_from_probs(p_s, p_l, rp, reg); note = ""
            if act == "AL":
                if rp["use_regime"] and not safe_float(row.get("Close")) > safe_float(row.get("SMA200")):
                    act, note = "TUT", "rejim"
                elif rp["gate_on"] and not gate_ok:
                    act, note = "TUT", "kapi"
                elif u > rp["unc_max"]:
                    act, note = "TUT", f"belirsizlik {u:.2f}"
                elif safe_float(row.get("Turnover20")) < rp.get("min_daily_turnover", 0):
                    act, note = "TUT", "likidite"
            out[nm] = {"action": act, "probs": p_s, "probs_long": p_l, "unc": u,
                       "regime": reg, "note": note, "atr": safe_float(row.get("ATR")), "feat": F[i]}
        return out
    def halt(self, reason, hours, now):
        self.halted, self.halt_reason = True, reason
        self.halt_until = (now + timedelta(hours=hours)).isoformat(timespec="seconds")
    def resume(self, prices):
        self.halted, self.halt_reason, self.halt_until, self.streak = False, "", None, 0
        self.peak_eq = self.total_value(prices)
    def _sell(self, s, price, date, reason):
        p = self.positions.pop(s)
        px = price * (1 - SLIPPAGE); proceeds = p["qty"] * px * (1 - FEE)
        pnl = proceeds - p["cost"]; self.cash += proceeds
        if pnl > 0: self.wins += 1; self.streak = 0
        else: self.losses += 1; self.streak += 1
        self.trades = (self.trades + [{"date": str(date), "action": "SAT", "stock": s, "price": px,
                                       "qty": p["qty"], "pnl": pnl, "reason": reason[:200]}])[-500:]
        db_add_trade(s, "SAT", px, p["qty"], pnl, reason)
        self.last_exit[s] = now_tr().strftime("%Y-%m-%d")
        if p.get("feat") is not None and p["cost"] > 0:
            net = pnl / p["cost"]
            label_s = 2 if net > RT_COST else (0 if net < -RT_COST else 1)
            self.nn.replay.add(p["feat"], label_s, -1, 1.0 + min(abs(net), 0.2) * 20)
        return pnl
    def _buy(self, s, price, date, q, sd, atr, feat, reason):
        px = price * (1 + SLIPPAGE); cost = q * px * (1 + FEE); self.cash -= cost
        self.positions[s] = {"qty": q, "entry": px, "cost": cost, "stop0": px - sd, "sd0": sd,
                             "atr": atr, "high": px, "feat": feat, "ts": _ts()}
        self.day_trades += 1
        self.trades = (self.trades + [{"date": str(date), "action": "AL", "stock": s, "price": px,
                                       "qty": q, "reason": reason[:200]}])[-500:]
        db_add_trade(s, "AL", px, q, 0.0, reason)
    def run_cycle(self, dec, prices, rp, date, now=None):
        now = now or now_tr(); msgs = []; eq = self.total_value(prices)
        dk = now.strftime("%Y-%m-%d")
        if self.day_key != dk: self.day_key, self.day_eq0, self.day_trades = dk, eq, 0
        if self.halted and self.halt_until and now.isoformat(timespec="seconds") >= self.halt_until:
            self.resume(prices)
        self.peak_eq = max(self.peak_eq, eq)
        dd = (self.peak_eq - eq) / self.peak_eq * 100 if self.peak_eq > 0 else 0.0
        if dd >= rp["max_dd"] and not self.halted:
            self.halt(f"KILL DD {dd:.1f}%", rp["cooldown_h"], now)
            for s in list(self.positions):
                if s in prices: msgs.append(f"{s}: SAT {self._sell(s, prices[s], date, 'KILL'):+.0f} TL")
            return msgs
        for s in list(self.positions):
            p, px = self.positions[s], prices.get(s)
            if not px: continue
            p["high"] = max(p["high"], px)
            stop, trail = stop_level(p["entry"], p["high"], p["sd0"], p["atr"], rp)
            pct = (px / p["entry"] - 1) * 100; reason = None
            if px <= stop: reason = f"{'TRAILING' if trail else 'STOP'} {pct:+.1f}%"
            else:
                try:
                    if (now - datetime.fromisoformat(p["ts"])).days >= rp["max_hold"]: reason = "ZAMAN STOPU"
                except Exception: pass
            if reason is None and dec.get(s, {}).get("action") == "SAT": reason = "SINYAL SAT"
            if reason: msgs.append(f"{s}: SAT {self._sell(s, px, date, reason):+.0f} TL")
        if self.streak >= rp["loss_streak"] and not self.halted:
            self.halt(f"SOGUMA ({self.streak})", rp["cooldown_h"], now)
        if self.halted: return msgs
        eq = self.total_value(prices)
        if self.day_eq0 > 0 and (eq - self.day_eq0) / self.day_eq0 * 100 <= -rp["daily_loss"]:
            self.halt("GUNLUK LIMIT", max(1, rp["cooldown_h"] // 4), now)
            return msgs
        cands = sorted([((d["probs"][2] + d["probs_long"][2]) / 2 - max(d["probs"][0], d["probs"][1]), s)
                        for s, d in dec.items() if d["action"] == "AL" and s not in self.positions
                        and prices.get(s) and self.last_exit.get(s) != dk], reverse=True)
        for _, s in cands:
            if len(self.positions) >= rp["max_positions"] or self.day_trades >= rp["max_trades_day"]: break
            d = dec[s]; px = prices[s] * (1 + SLIPPAGE); atr = d["atr"] or px * 0.02
            q, sd = size_pos(eq, self.cash, self.exposure(prices), px, atr, d["probs"], d["probs_long"], rp)
            if q >= 1:
                self._buy(s, prices[s], date, q, sd, atr, d.get("feat"), "AI AL")
                msgs.append(f"{s}: AL {q} @ {px:.2f}")
        return msgs
    def get_state(self):
        keys = ["cash","initial","positions","trades","wins","losses","eq_hist","peak_eq",
                "halted","halt_reason","halt_until","streak","day_key","day_eq0","day_trades",
                "pretrained","val_stats","hold","hold_te","ref","last_exit"]
        d = {k: getattr(self, k) for k in keys}
        d["nn"] = self.nn.get_state(); d["si"] = self.si.get_state()
        return d
    def set_state(self, d):
        for k, v in d.items():
            if k == "nn": self.nn.set_state(v)
            elif k == "si": self.si.set_state(v)
            else: setattr(self, k, v)

def save_state(bot, rp, path=STATE_FILE):
    try:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"ver": BOT_VERSION, "bot": bot.get_state(), "rp": rp}, f)
        os.replace(tmp, path); bot.nn.replay.save(MEMORY_FILE)
        return True
    except Exception: return False
def load_state(path=STATE_FILE):
    if not os.path.exists(path): return None
    try:
        with open(path, "rb") as f: obj = pickle.load(f)
        return obj if isinstance(obj, dict) and obj.get("ver") == BOT_VERSION else None
    except Exception: return None

def _fetch(sym, period="5y"):
    for _ in range(2):
        try:
            d = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
            if d is not None and not d.empty:
                d = d.reset_index()
                d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
                return d[["Date","Open","High","Low","Close","Volume"]]
        except Exception: time.sleep(0.4)
    return None
@st.cache_resource(ttl=900, show_spinner=False)
def load_uni(day_key):
    items = list(HISSELER.items()) + [(f"X:{n}", t) for n, t in CROSS_SYMBOLS.items()]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_fetch, sym): name for name, sym in items}
        res = {}
        for f in as_completed(futures):
            try: res[futures[f]] = f.result()
            except Exception: res[futures[f]] = None
    cross = {n: res[f"X:{n}"].set_index("Date")["Close"] for n in CROSS_SYMBOLS if res.get(f"X:{n}") is not None}
    dfs, errs = {}, []
    for nm in HISSELER:
        d = res.get(nm)
        if d is None: errs.append(nm); continue
        try:
            d = add_cross(add_ind(d), cross)
            if len(d) >= 300: dfs[nm] = d
            else: errs.append(f"{nm}(kısa)")
        except Exception as e: errs.append(f"{nm}:{e}")
    try:
        rk = build_cross_rank(dfs)
        for nm in list(dfs.keys()): dfs[nm] = inject_cross_rank(dfs[nm], rk.get(nm))
    except Exception as e:
        errs.append(f"rank:{e}")
    return {"dfs": dfs, "errs": errs, "ts": now_tr().strftime("%H:%M")}
@st.cache_data(ttl=30, show_spinner=False)
def live_prices(tickers):
    try:
        d = yf.download(list(tickers), period="1d", interval="1m", group_by="ticker",
                        progress=False, threads=True, auto_adjust=True)
        out = {}
        for t in tickers:
            try:
                c = d[t]["Close"].dropna()
                if len(c): out[t] = float(c.iloc[-1])
            except Exception: pass
        return out
    except Exception: return {}
def sig_idx(d, now):
    if len(d) > 1 and d["Date"].iloc[-1].date() == now.date() and now.time() < dtime(18, 20):
        return len(d) - 2
    return len(d) - 1
def run_improve(bot, datasets, rp, data_date, reason):
    with st.spinner("🧬 Öz-gelişim çalışıyor..."):
        new_rp, rep = bot.si.run(bot, datasets, rp, budget=float(rp["si_budget_s"]), reason=reason)
    bot.si.last_run, bot.si.last_data = time.time(), data_date
    if rep.get("error"):
        st.session_state.log.append(f"{now_tr():%H:%M:%S} Öz-gelişim: {rep['error']}")
        return rp
    out = rp
    if new_rp:
        out = validate_risk(new_rp); st.session_state.rp = out
        for k in TUNE_BOUNDS: st.session_state.pop(f"rp_{k}", None)
        bot.reeval(out)
    extra = f" te_acc {rep['te_acc']:.3f}(taban {rep['te_base']:.3f}) gap {rep['gen_gap']:.2f}"
    if "dsr" in rep: extra += f" DSR {rep['dsr']:.3f}"
    st.session_state.log = (st.session_state.log + [
        f"{now_tr():%H:%M:%S} 🧬 {reason}: vl {rep['vl0']:.4f}→{rep['vl1']:.4f} drift {rep['drift']:.2f} "
        f"pbt {rep['pbt']} router {rep['router']}{extra} " + (f"UYGULANDI {rep.get('changes')}" if rep.get("adopted") else "")])[-100:]
    save_state(bot, out)
    return out

def main():
    st.set_page_config(page_title="THYAO AI v33", page_icon="🧬", layout="wide")
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.title("🧬 THYAO AI v33 - VERIFIED MoE")
    c1.caption("Kısa+Uzun kafa · Doğrulamalı Router · Cross-rank · 3'lü split · DSR · hisse-doğru simülatör")
    (c2.success if borsa_acik() else c2.error)("🟢 AÇIK" if borsa_acik() else "🔴 KAPALI")
    c3.caption(f"🕐 {now_tr():%H:%M:%S}")
    for k, v in {"bot": None, "auto": False, "test_mode": False, "interval": 60,
                 "last_tick": 0.0, "ticks": 0, "log": [], "sel": "THYAO", "rp": None}.items():
        if k not in st.session_state: st.session_state[k] = v
    if st.session_state.bot is None:
        b, loaded = Bot(), load_state()
        if loaded:
            try:
                b.set_state(loaded["bot"]); b.nn.replay.load(MEMORY_FILE)
                st.session_state.rp = loaded.get("rp")
            except Exception as e:
                st.warning(f"State yüklenemedi, sıfırdan başlıyor: {e}")
                b = Bot()
        st.session_state.bot = b
    bot = st.session_state.bot
    st.session_state.rp = validate_risk(st.session_state.rp); rp = st.session_state.rp
    now = now_tr()
    with st.spinner("📥 20 hisse + cross + rank verisi yükleniyor..."):
        uni = load_uni(now.strftime("%Y-%m-%d"))
    dfs = uni["dfs"]
    if not dfs:
        st.error("❌ Veri yok: " + ", ".join(uni["errs"][:5])); st.stop()
    if st.session_state.get("dss_key") != (uni["ts"], len(dfs)):
        st.session_state.dss = {nm: make_ds(d, nm) for nm, d in dfs.items()}
        st.session_state.dss_key = (uni["ts"], len(dfs))
    datasets = [v for v in st.session_state.dss.values() if v]
    data_date = str(max(d["Date"].iloc[-1] for d in dfs.values()).date())
    if not bot.pretrained and datasets:
        with st.spinner("🧠 Eğitim (dual-head, SWA, router)..."):
            bot.pretrain(datasets, rp, steps=400)
        bot.si.last_run, bot.si.last_data = time.time(), data_date
        st.session_state.log.append(f"{now:%H:%M:%S} Eğitim tamam")
        save_state(bot, rp)
    names, rows, feats = [], {}, []
    for nm, d in dfs.items():
        i = sig_idx(d, now); rows[nm] = d.iloc[i]
        feats.append(feature_matrix(d.iloc[[i]], nm)[0]); names.append(nm)
    F_live = np.array(feats, dtype=np.float32)
    live = live_prices(tuple(HISSELER[n] for n in names))
    prices = {nm: live.get(HISSELER[nm], float(dfs[nm]["Close"].iloc[-1])) for nm in names}
    feat_dr = bot.feat_drift(F_live)
    if rp["si_auto"] and bot.pretrained and datasets:
        t_ = time.time()
        if bot.si.due(t_, data_date, rp["si_interval_min"]) or (feat_dr > 1.2 and t_ - bot.si.last_run > 600):
            rp = run_improve(bot, datasets, rp, data_date, "auto")
    dec = bot.scan(names, F_live, rows, rp)
    sec = st.session_state.sel
    if sec not in dfs: sec = names[0]
    df, sd_ = dfs[sec], dec[sec]; price = prices[sec]
    auto_msgs = []
    if (st.session_state.auto and time.time() - st.session_state.last_tick >= st.session_state.interval - 1
            and (borsa_acik() or st.session_state.test_mode) and bot.pretrained):
        st.session_state.last_tick = time.time(); st.session_state.ticks += 1
        auto_msgs = bot.run_cycle(dec, prices, rp, now.strftime("%Y-%m-%d %H:%M"), now)
        v_ = bot.total_value(prices); bot.record(v_)
        db_eq_add(v_, bot.cash, len(bot.positions))
        if auto_msgs:
            st.session_state.log = (st.session_state.log + [f"{now:%H:%M:%S} " + " | ".join(auto_msgs)])[-100:]
        if st.session_state.ticks % 10 == 0: save_state(bot, rp)
    if st.session_state.auto: st_autorefresh(interval=st.session_state.interval * 1000, key="rf")
    cur_val = bot.total_value(prices); bot.record(cur_val)
    with st.sidebar:
        st.markdown("## ⚙️ Panel")
        st.selectbox("📊 Hisse", names, key="sel")
        st.metric(sec, f"{price:.2f} TL", f"{(price / float(df['Close'].iloc[-2]) - 1) * 100:+.2f}%")
        st.metric("💰 Portföy", f"{cur_val:,.0f} TL", f"{cur_val - bot.initial:+,.0f}")
        st.divider()
        st.toggle("🟢 Otomatik işlem", key="auto")
        st.toggle("🧪 Test modu", key="test_mode")
        st.slider("Yenileme (sn)", 30, 300, step=10, key="interval")
        if st.session_state.auto:
            dsc = "çalışıyor" if (borsa_acik() or st.session_state.test_mode) else "kapalı"
            st.caption(f"Tick: {st.session_state.ticks} · {dsc}")
        st.divider()
        s1, s2, s3 = st.columns(3)
        s1.metric("Poz", len(bot.positions)); s2.metric("Hafıza", bot.nn.replay.size())
        s3.metric("Evrim", bot.nn.evo)
        vl_str = f"{bot.nn.best_vl:.4f}" if np.isfinite(bot.nn.best_vl) else "-"
        st.caption(f"Val loss: {vl_str} · 🧬 tur: {bot.si.runs}")
        st.caption(f"Router: {'AÇIK' if bot.nn.router_on else 'kapalı'} (kazanç {bot.nn.router_gain:+.4f})")
        if bot.halted: st.error(f"⛔ {bot.halt_reason}")
        if st.button("💾 Kaydet", use_container_width=True):
            st.toast("✅" if save_state(bot, rp) else "❌")
        if st.button("🛑 ACİL DUR", use_container_width=True, type="primary"):
            for s in list(bot.positions):
                if s in prices: bot._sell(s, prices[s], now, "ACIL")
            bot.halted, bot.halt_reason = True, "ACİL"; st.rerun()
        if st.button("▶️ Devam", use_container_width=True): bot.resume(prices); st.rerun()
        if st.button("🗑️ Sıfırla", use_container_width=True):
            for f_ in (STATE_FILE, MEMORY_FILE):
                try: os.remove(f_)
                except OSError: pass
            db_clear(); st.session_state.bot = Bot(); st.session_state.log = []; st.rerun()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Fiyat", f"{price:.2f}")
    m2.metric("Karar", sd_["action"], f"P(AL) %{sd_['probs'][2] * 100:.0f}")
    m3.metric("Portföy", f"{cur_val:,.0f}", f"%{(cur_val / bot.initial - 1) * 100:+.2f}")
    m4.metric("İşlem", len(bot.trades), f"kaz %{bot.win_rate() * 100:.0f}" if bot.trades else None)
    m5.metric("Rejim", sd_["regime"]); m6.metric("Belirsizlik", f"{sd_['unc']:.2f}")
    if sd_["note"]: st.warning(f"AL engellendi: {sd_['note']}")
    if auto_msgs: st.info(" | ".join(auto_msgs[:5]))
    tK, tG, tD, tB, tSI, tCAL, tDB, tCH = st.tabs(
        ["🎛️ Kontrol","📈 Grafik","🎯 Karar","🧠 Beyin","🧬 Öz-Gelişim","📊 Kalibrasyon","💾 DB","💬 Log"])
    def _cb(name, key):
        st.session_state.rp[name] = clamp(safe_float(st.session_state[key]), *RISK_BOUNDS.get(name, (0, 1)))
    def rp_sl(name, label, lo, hi, step):
        key = f"rp_{name}"
        if key not in st.session_state: st.session_state[key] = type(step)(clamp(rp[name], lo, hi))
        st.slider(label, lo, hi, step=step, key=key, on_change=_cb, args=(name, key))
    def rp_tg(name, label):
        key = f"rp_{name}"
        if key not in st.session_state: st.session_state[key] = bool(rp[name])
        st.toggle(label, key=key, on_change=lambda n=name, k=key: st.session_state.rp.__setitem__(n, bool(st.session_state[k])))
    with tK:
        cA, cB, cC = st.columns(3)
        with cA:
            st.markdown("**Pozisyon**")
            rp_sl("risk_per_trade","Risk %",0.002,0.05,0.001)
            rp_sl("max_pos","Tek poz max",0.05,0.5,0.01)
            rp_sl("max_exposure","Toplam maruziyet",0.2,1.0,0.05)
            rp_sl("max_positions","Max poz sayısı",1,20,1)
            rp_sl("sl_atr","Stop ATR",1.0,6.0,0.1)
            rp_sl("trail_act_r","Trail açılış R",0.5,4.0,0.1)
            rp_sl("trail_atr","Trail ATR",1.0,5.0,0.1)
            rp_sl("min_daily_turnover","Min günlük hacim (TL)",0.0,1e8,1e6)
        with cB:
            st.markdown("**Sinyal**")
            rp_sl("p_buy","P(AL) kısa eşik",0.34,0.80,0.01)
            rp_sl("p_buy_long","P(AL) uzun eşik",0.34,0.80,0.01)
            rp_sl("p_sell","P(SAT) eşik",0.34,0.80,0.01)
            rp_sl("margin","Karar marjı",0.0,0.40,0.01)
            rp_sl("unc_max","Max belirsizlik",0.05,0.50,0.01)
            rp_sl("max_hold","Zaman stop (takvim günü)",5,400,5)
            rp_tg("use_regime","Rejim filtresi")
            rp_tg("use_long_gate","Uzun vade hakem")
            rp_tg("conf_sizing","Güvenle boyut")
        with cC:
            st.markdown("**Yönetişim**")
            rp_sl("max_dd","Kill DD %",3.0,50.0,0.5)
            rp_sl("daily_loss","Günlük zarar %",0.5,15.0,0.5)
            rp_sl("max_trades_day","Günlük işlem",1,30,1)
            rp_sl("loss_streak","Soğuma streak",1,10,1)
            rp_sl("cooldown_h","Soğuma saat",1,240,1)
            rp_tg("gate_on","Model kapısı")
            rp_sl("gate_min_n","Kapı min AL",3,100,1)
            rp_sl("dsr_confidence","DSR güven eşiği",0.5,0.999,0.005)
            rp_sl("consec_improve","Ardışık doğrulama",1,5,1)
            rp_sl("test_score_tol","Test skoru toleransı",0.0,2.0,0.05)
        st.divider()
        st.markdown("### 📦 Açık Pozisyonlar")
        if not bot.positions: st.info("Yok")
        else:
            prow = []
            for nm, p in bot.positions.items():
                cur = prices.get(nm, p["entry"])
                stop, _ = stop_level(p["entry"], p["high"], p["sd0"], p["atr"], rp)
                prow.append({"Hisse": nm, "Adet": p["qty"], "Giriş": round(p["entry"], 2),
                             "Şu an": round(cur, 2), "K/Z ₺": round((cur - p["entry"]) * p["qty"]),
                             "K/Z %": round((cur / p["entry"] - 1) * 100, 2), "Stop": round(stop, 2)})
            st.dataframe(pd.DataFrame(prow), use_container_width=True, hide_index=True)
    with tG:
        n_show = st.select_slider("Gün", options=[60, 120, 180, 250], value=120)
        d = df.tail(n_show)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Fiyat"), row=1, col=1)
        for col, clr in [("SMA20","#f5a623"), ("SMA50","#4a90e2"), ("SMA200","#bd10e0")]:
            fig.add_trace(go.Scatter(x=d["Date"], y=d[col], name=col, line=dict(width=1.2, color=clr)), row=1, col=1)
        pos_ = bot.positions.get(sec)
        if pos_:
            stp, _ = stop_level(pos_["entry"], pos_["high"], pos_["sd0"], pos_["atr"], rp)
            fig.add_hline(y=stp, line_dash="dash", line_color="red", row=1, col=1)
        for t in bot.trades[-100:]:
            if t["stock"] == sec:
                try:
                    td_ = pd.Timestamp(t["date"])
                    if td_.tzinfo is not None: td_ = td_.tz_localize(None)
                    if td_ >= d["Date"].iloc[0]:
                        up_ = t["action"] == "AL"
                        fig.add_trace(go.Scatter(x=[td_], y=[t["price"]], mode="markers", showlegend=False,
                                                 marker=dict(color="lime" if up_ else "red", size=13,
                                                             symbol="triangle-up" if up_ else "triangle-down")), row=1, col=1)
                except Exception: pass
        fig.add_trace(go.Scatter(x=d["Date"], y=d["RSI"], showlegend=False, line=dict(color="#50e3c2")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        fig.add_trace(go.Bar(x=d["Date"], y=d["MACDh"], showlegend=False,
                             marker_color=np.where(d["MACDh"] >= 0, "#26a69a", "#ef5350")), row=3, col=1)
        fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with tD:
        st.subheader(f"🎯 {sec} Karar")
        ps_, pl_ = sd_["probs"], sd_["probs_long"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Karar", sd_["action"])
        k2.metric("P(AL) kısa", f"%{ps_[2] * 100:.1f}", f"P(TUT) %{ps_[1] * 100:.0f} · P(SAT) %{ps_[0] * 100:.0f}")
        k3.metric("P(AL) uzun", f"%{pl_[2] * 100:.1f}", f"P(TUT) %{pl_[1] * 100:.0f} · P(SAT) %{pl_[0] * 100:.0f}")
        k4.metric("Uzun hakem", "🟢 BULL" if pl_[2] >= rp["p_buy_long"] else ("🔴 BEAR" if pl_[0] >= rp["p_sell"] else "⚪ NÖTR"))
        st.divider()
        trows = [{"Hisse": nm, "Fiyat": round(prices[nm], 2), "Karar": d_["action"],
                  "P(AL) kısa %": round(d_["probs"][2] * 100, 0), "P(AL) uzun %": round(d_["probs_long"][2] * 100, 0),
                  "Belirsizlik": round(d_["unc"], 2), "Rejim": d_["regime"], "Not": d_["note"]}
                 for nm, d_ in dec.items()]
        st.dataframe(pd.DataFrame(sorted(trows, key=lambda x: -(x["P(AL) kısa %"] + x["P(AL) uzun %"]) / 2)),
                     use_container_width=True, hide_index=True)
        vs = bot.val_stats
        if vs:
            st.markdown("### 🚪 Model Kapısı (val)")
            g1, g2, g3, g4, g5, g6 = st.columns(6)
            g1.metric("Val acc", f"%{vs['acc'] * 100:.1f}", f"taban %{vs['base'] * 100:.1f}")
            g2.metric("AL çağrı", vs["al_n"]); g3.metric("AL isabet", f"%{vs['al_prec'] * 100:.1f}")
            g4.metric("AL ort. (brüt)", f"%{vs['al_ret'] * 100:+.2f}")
            g5.metric("Kazanma", f"%{vs.get('wr', 0) * 100:.1f}")
            g6.metric("Kapı", "✅" if bot.edge_ok(rp) else "⛔")
    with tB:
        st.subheader("🧠 Beyin v33")
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Evrim", bot.nn.evo); b2.metric("En iyi net", f"#{int(np.argmax(bot.nn.perf))}")
        b3.metric("Hafıza", bot.nn.replay.size()); b4.metric("PBT kabul", f"{bot.nn.pbt_acc}/{bot.nn.pbt_n}")
        b5.metric("Router", "AÇIK" if bot.nn.router_on else "kapalı", f"{bot.nn.router_gain:+.4f}")
        for i, p in enumerate(bot.nn.perf):
            nn_i = bot.nn.nets[i]
            st.progress(float(min(max(p, 0.0), 1.0)),
                        text=f"Net #{i}: {p:.3f} · lr {nn_i.lr:.4f} · drop {nn_i.dropout:.2f} · SWA {nn_i.swa_wins}")
        st.caption("Router, val'in ilk %60'ında eğitilir, son %40'ında statik ağırlıklara karşı sınanır; kazanmazsa kapalı kalır.")
        st.divider()
        if st.button("🔄 10 Adım Hızlı Evrim", use_container_width=True) and datasets:
            sp_ = split_ds(merge_ds(datasets))
            if sp_:
                with st.spinner("Evrim..."):
                    for _ in range(10): bot.nn.evolve_step(sp_)
                    bot.nn.fit_router(sp_)
                bot.reeval(rp); save_state(bot, rp)
                st.success("Tamam"); st.rerun()
    with tSI:
        st.subheader("🧬 Öz-Gelişim (DSR + test hakemi + ardışık)")
        si = bot.si
        i1, i2, i3, i4, i5 = st.columns(5)
        i1.metric("Tur", si.runs)
        i2.metric("Drift", f"{si.drift:.2f}", "bayat" if si.drift > 1.15 else "sağlıklı",
                  delta_color="inverse" if si.drift > 1.15 else "normal")
        i3.metric("Özellik kayması", f"{feat_dr:.2f}")
        i4.metric("Ardışık", f"{si.consec_improve}/{rp['consec_improve']}")
        i5.metric("Son tur", f"{int((time.time() - si.last_run) / 60)} dk" if si.last_run else "-")
        q1, q2 = st.columns(2)
        with q1:
            rp_tg("si_auto", "🧬 Otomatik öz-gelişim")
            rp_tg("auto_adopt", "✅ Otomatik uygula")
        with q2:
            rp_sl("si_interval_min", "Tur aralığı (dk)", 5, 720, 5)
            rp_sl("si_budget_s", "Tur bütçesi (sn)", 2, 60, 1)
        a1, a2, a3 = st.columns(3)
        if a1.button("🧬 Şimdi Geliştir", use_container_width=True) and datasets:
            run_improve(bot, datasets, rp, data_date, "manuel"); st.rerun()
        if si.proposal and a2.button("✅ Öneriyi Uygula", use_container_width=True):
            st.session_state.rp = validate_risk({**rp, **{k: float(v) for k, v in si.proposal["params"].items()}})
            si.proposal = None; si.consec_improve = 0; si.pending_rp = None
            for k in TUNE_BOUNDS: st.session_state.pop(f"rp_{k}", None)
            bot.reeval(st.session_state.rp); st.rerun()
        if si.rp_hist and a3.button("↩️ Önceki ayara dön", use_container_width=True):
            prev = si.rp_hist.pop(); st.session_state.rp = validate_risk({**rp, **prev})
            for k in TUNE_BOUNDS: st.session_state.pop(f"rp_{k}", None)
            bot.reeval(st.session_state.rp); st.rerun()
        if si.proposal:
            st.info(f"Öneri (≥{si.proposal.get('wait', 0)} tur daha): "
                    f"{({k: round(v, 3) for k, v in si.proposal['params'].items()})}")
        st.markdown("**Şu anki evrimleşen parametreler**")
        st.dataframe(pd.DataFrame([{k: round(float(rp[k]), 3) for k in TUNE_BOUNDS}]),
                     use_container_width=True, hide_index=True)
        if si.history:
            hrows = [{"Zaman": h["ts"][5:16], "Neden": h["reason"], "Drift": round(h.get("drift", 0), 2),
                      "Val loss": f"{h.get('vl0', 0):.4f}→{h.get('vl1', 0):.4f}",
                      "te_acc": (f"{h.get('te_acc', 0):.3f}" if "te_acc" in h else "-"),
                      "gap": (f"{h.get('gen_gap', 0):.2f}" if "gen_gap" in h else "-"),
                      "DSR": (f"{h.get('dsr', 0):.3f}" if "dsr" in h else "-"),
                      "PBT": h.get("pbt", "-"), "Router": h.get("router", "-"),
                      "Uyum": "✅" if h.get("adopted") else "-", "sn": h.get("sec", 0)}
                     for h in reversed(si.history[-15:])]
            st.dataframe(pd.DataFrame(hrows), use_container_width=True, hide_index=True)
        if len(bot.eq_hist) > 2:
            fe = go.Figure(go.Scatter(y=bot.eq_hist, mode="lines", line=dict(color="#50e3c2")))
            fe.update_layout(height=260, template="plotly_dark",
                             margin=dict(l=10, r=10, t=30, b=10), title="Portföy eğrisi")
            st.plotly_chart(fe, use_container_width=True)
        st.caption("Güvenlik: öz-gelişim yalnız sinyal/stop parametrelerine dokunur; risk %, max pozisyon, kill-DD sizin.")
    with tCAL:
        st.subheader("📊 Kalibrasyon Paneli (TEST)")
        st.caption("Sıcaklık val'de fit edilir; ölçüm TEST'te yapılır (sızıntısız).")
        if bot.pretrained and datasets:
            sp_ = split_ds(merge_ds(datasets))
            if sp_:
                if st.button("🧪 Şimdi Ölç", use_container_width=True):
                    cal = bot.si.calibrate(bot.nn, sp_["Xte"], sp_["yte"])
                    if cal:
                        st.session_state["_cal"] = cal
                        db_calib_add(cal["n"], cal["brier"], cal["ece"], cal["acc"], cal["base"])
                cal = st.session_state.get("_cal")
                if cal is None:
                    cal = bot.si.calibrate(bot.nn, sp_["Xte"], sp_["yte"])
                    st.session_state["_cal"] = cal
                if cal:
                    kk = st.columns(4)
                    kk[0].metric("Brier", f"{cal['brier']:.4f}")
                    kk[1].metric("ECE", f"{cal['ece']:.4f}")
                    kk[2].metric("İsabet", f"%{cal['acc'] * 100:.1f}", f"taban %{cal['base'] * 100:.1f}")
                    kk[3].metric("Örnek", cal["n"])
                    if cal["cal"]:
                        cdf = pd.DataFrame(cal["cal"])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=cdf["pred"], y=cdf["actual"], mode="markers+lines",
                                                 marker=dict(size=cdf["n"] / max(cdf["n"].max(), 1) * 20 + 5),
                                                 name="Güvenilirlik"))
                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                                 line=dict(dash="dash", color="gray"), name="Mükemmel"))
                        fig.update_layout(height=350, template="plotly_dark",
                                          xaxis_title="Tahmin P(AL)", yaxis_title="Gerçek P(AL)",
                                          title="Reliability Diagram (TEST)")
                        st.plotly_chart(fig, use_container_width=True)
                    st.caption("Brier: düşük iyi · ECE: düşük iyi · Çapraz çizgi mükemmel kalibrasyon")
        else:
            st.info("Önce model eğitilmeli")
    with tDB:
        tr, eq, tn = db_trades(100), db_eq(200), db_train(20)
        d1, d2, d3 = st.columns(3)
        d1.metric("İşlem", len(tr)); d2.metric("Equity", len(eq)); d3.metric("Eğitim", len(tn))
        if tr: st.dataframe(pd.DataFrame(tr), use_container_width=True, hide_index=True)
        else: st.info("Yok")
    with tCH:
        if st.session_state.log:
            for line in reversed(st.session_state.log[-30:]): st.text(line)
        else: st.info("Yok")
    st.divider()
    st.caption(f"🕐 {now:%Y-%m-%d %H:%M:%S} | Borsa: {'🟢 AÇIK' if borsa_acik() else '🔴 KAPALI'} | v{BOT_VERSION}")
    st.caption("⚠️ Kâğıt işlem simülasyonu - yatırım tavsiyesi değildir")

if not os.environ.get("THYAO_NO_UI"):
    main()
