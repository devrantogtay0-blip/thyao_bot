# -*- coding: utf-8 -*-
# ============================================================
# 🐋 SEEK DEEP v2.5 PRO — DÜZELTİLMİŞ TAM SÜRÜM
# PARÇA 1/4: Foundation
# Düzeltmeler: _fetch/sanitize yukarı, seq_store sektör-bazlı
# ============================================================
from __future__ import annotations
import os, math, time, pickle, sqlite3, threading, json, hashlib, logging, copy
import urllib.request
from contextlib import contextmanager
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from numpy.lib.stride_tricks import sliding_window_view

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

# ════════════════════════════════════════════════════════════
# 1. CONFIG
# ════════════════════════════════════════════════════════════
BOT_VERSION = "SeekDeep-2.5-Pro"
BOT_NAME = "🐋 SEEK DEEP"
HORIZON, HORIZON_LONG = 10, 40
TB_K, TB_K_LONG = 1.2, 1.5
ALPHA_LONG = 0.35
FEE, SLIPPAGE = 0.0015, 0.001
RT_COST = 2 * (FEE + SLIPPAGE)
HW = 90

N_TECH_BASE, N_NEWS, N_LIQ, N_RANK, N_FUND, N_MACRO = 49, 0, 2, 5, 0, 8
N_FEAT_TECH = N_TECH_BASE + N_NEWS + N_LIQ + N_RANK + N_FUND + N_MACRO  # 64
N_STOCKS = 24
N_FEAT = N_FEAT_TECH + N_STOCKS  # 88

H_DIM, BAG_N, BAG_SEED, HOLD_N = 64, 5, 100, 6000
USE_ROUTER, ROUTER_H, ROUTER_MIX, USE_DSR = True, 32, 0.5, True
SEQ_LEN, N_HEADS, TRANS_DIM, TRANS_LAYERS = 20, 4, 64, 2
MLP_WEIGHT, TRANS_WEIGHT = 0.6, 0.4
CPCV_ENABLED, CPCV_N_GROUPS, CPCV_N_TEST_GROUPS = True, 6, 2
PBO_ENABLED = True
DYNAMIC_SLIPPAGE, SLIPPAGE_BASE = True, 0.0005
RISK_PARITY = True
VAR_CONFIDENCE = 0.95

NEWS_ENABLED, NEWS_CACHE_HOURS, NEWS_MAX_PER_STOCK = True, 6, 5
NEWS_LLM_PER_CYCLE = 5  # 🐛 SORUN 3: Rate limit için LLM batch limit
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
KAP_ENABLED = True
LIQ_FLAG_TL = 5_000_000
TR = ZoneInfo("Europe/Istanbul")

SEKTOR_ENV = os.environ.get("SEKTOR", "BANKACILIK").upper()
_sfx = SEKTOR_ENV.lower()
DB_FILE = f"seekdeep_{_sfx}.db"
STATE_FILE = f"seekdeep_{_sfx}.pkl"
MEMORY_FILE = f"seekdeep_{_sfx}.npz"
NEWS_FILE = f"seekdeep_news_{_sfx}.json"
FUND_FILE = f"seekdeep_fund_{_sfx}.json"
KAP_FILE = f"seekdeep_kap_{_sfx}.json"
METRICS_FILE = f"metrics_{_sfx}.prom"

log = logging.getLogger("seekdeep")
if not log.handlers:
    log.setLevel(logging.INFO)
    try:
        h = logging.FileHandler("seekdeep.log", encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
    except Exception:
        pass

# ════════════════════════════════════════════════════════════
# 2. 3-BOT KONFİGÜRASYONU
# ════════════════════════════════════════════════════════════
SEKTOR_BOTLARI = {
    "BANKACILIK": {"bot_id": 1, "sermaye": 33333.0,
        "hisseler": ["GARAN","AKBNK","ISCTR","YKBNK","HALKB","VAKBN","QNBFB","TSKB","ALBRK","SKBNK","ICBCT","KLNMA",
                     "QNBFL","KCHOL","SAHOL","AGHOL","GLYHO","TURSG","ANSGR","AKGRT","ISMEN","ISFIN","HLGYO","ISGYO"],
        "aciklama": "Bankacılık + Holding + Sigorta + GYO",
        "risk_profil": {"risk_per_trade": 0.008, "max_positions": 6, "sl_atr": 2.0, "daily_loss": 2.0}},
    "HAVACILIK": {"bot_id": 2, "sermaye": 33334.0,
        "hisseler": ["THYAO","PGSUS","TAVHL","CLEBI","DOAS","FROTO","TOASO","TTRAK","OTKAR","ASUZU","KARSN","BRISA",
                     "GOODY","TUKAS","ULKER","CCOLA","AEFES","BIMAS","MGROS","SOKM","MAVI","MEPET","BIZIM","ULUFA"],
        "aciklama": "Havacılık + Otomotiv + Perakende + Gıda",
        "risk_profil": {"risk_per_trade": 0.012, "max_positions": 8, "sl_atr": 3.0, "daily_loss": 3.5}},
    "ENERJI": {"bot_id": 3, "sermaye": 33333.0,
        "hisseler": ["TUPRS","PETKM","AKSEN","ENJSA","ENERY","ZOREN","ODAS","AYDEM","EREGL","KRDMD","ISDMR","SISE",
                     "TRKCM","SASA","AKSA","GUBRF","ASELS","LOGO","NETAS","ARDYZ","KAREL","KONTR","PAPIL","FORTE"],
        "aciklama": "Enerji + Demir-Çelik + Kimya + Teknoloji",
        "risk_profil": {"risk_per_trade": 0.010, "max_positions": 7, "sl_atr": 2.5, "daily_loss": 3.0}},
}
AKTIF_SEKTOR = SEKTOR_ENV if SEKTOR_ENV in SEKTOR_BOTLARI else "BANKACILIK"
BOT_ID = SEKTOR_BOTLARI[AKTIF_SEKTOR]["bot_id"]
BOT_HISSELER = SEKTOR_BOTLARI[AKTIF_SEKTOR]["hisseler"]
BOT_SERMAYE = SEKTOR_BOTLARI[AKTIF_SEKTOR]["sermaye"]
SEKTOR_RISK = SEKTOR_BOTLARI[AKTIF_SEKTOR]["risk_profil"]
assert len(BOT_HISSELER) == N_STOCKS
HISSELER = {c: f"{c}.IS" for c in BOT_HISSELER}
STOCK_LIST = list(HISSELER.keys())
CROSS_SYMBOLS = {"USDTRY": "USDTRY=X", "XU100": "XU100.IS"}
MACRO_KEYS = ["usdtry", "eurtry", "gold", "brent"]
log.info(f"🐋 Bot {BOT_ID} ({AKTIF_SEKTOR}): {N_STOCKS} hisse, ₺{BOT_SERMAYE:,.0f}")

BIST_TUM = [
    "ACSEL","ADEL","ADESE","AFYON","AGHOL","AGYO","AHGAZ","AKBNK","AKCNS","AKENR","AKFGY","AKFYE","AKGRT","AKMGY","AKSA",
    "AKSEN","ALARK","ALBRK","ALCAR","ALCTL","ALFAS","ALGYO","ALKA","ALKIM","ALTNY","ANACM","ANHYT","ANSGR","ARASE","ARCLK",
    "ARDYZ","ARENA","ARSAN","ARTMS","ASELS","ASGYO","ASTOR","ASUZU","ATAGY","ATAKP","ATATP","ATEKS","ATLAS","AVOD","AVPGY",
    "AYCES","AYDEM","AYEN","AYGAZ","AZTEK","BAGFS","BAKAB","BALSU","BANVT","BARMA","BASGZ","BATAS","BAYRK","BERA","BEYAZ",
    "BFREN","BIENY","BIGCH","BIMAS","BINHO","BIOEN","BIZIM","BJKAS","BLCYT","BLUME","BMSCH","BMSTL","BNTAS","BOBET","BORSK",
    "BOSSA","BRISA","BRKSN","BRMEN","BRSAN","BRYAT","BSOKE","BTCIM","BUCIM","BULGS","BURCE","BURVA","BVSAN","BYDNR","CANTE",
    "CASA","CCOLA","CEMAS","CEMTS","CIMSA","CLEBI","CMBTN","CMENT","CONSE","COSMO","CRDFA","CRFSA","CUSAN","CVKMD","CWENE",
    "DAGHL","DAGI","DAPGM","DARDL","DENGE","DERHL","DERIM","DESA","DESPC","DEVA","DGKLB","DGNMO","DIRIT","DITAS","DMRGD",
    "DMSAS","DNISI","DOAS","DOBUR","DOHOL","DOKTA","DURDO","DURKN","DYOBY","DZGYO","EBEBK","ECILC","ECZYT","EDATA","EDIP",
    "EFOR","EGEEN","EGGUB","EGPRO","EGSER","EKGYO","EKIZ","EKSUN","ELITE","EMKEL","EMNIS","ENDAE","ENERY","ENJSA","ENKAI",
    "ENSRI","ENTRA","EPLAS","ERBOS","ERCB","EREGL","ERSU","ESCAR","ESCOM","ESEN","ETILR","EUPWR","EUREN","EVDRE","EVKUR",
    "EVREN","FADE","FENER","FLAP","FMIZP","FONET","FORMT","FORTE","FRIGO","FROTO","FZLGY","GARAN","GARFA","GEDIK","GEDZA",
    "GENIL","GENTS","GEREL","GESAN","GIPTA","GLBMD","GLCVY","GLRYH","GLYHO","GMTAS","GOKNR","GOLTS","GOODY","GOZDE","GRNYO",
    "GSDDE","GSDHO","GSRAY","GUBRF","GUNDG","GWIND","HALKB","HATEK","HATSN","HAYAT","HEDEF","HEKTS","HKTM","HLGYO","HOROZ",
    "HTTBT","HUBVC","HUNER","HURGZ","ICBCT","ICUGS","IDEAS","IDGYO","IEYHO","IHAAS","IHEVA","IHGZT","IHLAS","IHLGM","IHYAY",
    "IMASM","INDES","INFO","INGRM","INTEM","INVEO","ISATR","ISBIR","ISDMR","ISFIN","ISGYO","ISKUR","ISMEN","ISSEN","IZENR",
    "IZFAS","IZMDC","JANTS","KAPLM","KAREL","KARSN","KARTN","KATMR","KAYSE","KBORU","KCAER","KCHOL","KENT","KERVN","KERVT",
    "KFEIN","KGYO","KIMMR","KLKIM","KLMSN","KLNMA","KLRHO","KLSER","KLSYN","KLYPV","KMPUR","KNFRT","KONKA","KONTR","KONYA",
    "KORDS","KOZAA","KOZAL","KRDMA","KRDMB","KRDMD","KRGYO","KRONT","KRPLS","KRSTL","KRTEK","KRVGD","KSTUR","KTLEV","KTSKR",
    "KUTPO","KUVVA","KUYAS","KZBGY","LIDER","LILAK","LINK","LOGO","LRSHO","LUKSK","MAALT","MACKO","MAGEN","MAKIM","MAKTK",
    "MANAS","MARBL","MARKA","MARTI","MAVI","MEDTR","MEGAP","MEKAG","MEPET","MERCN","MERIT","MERKO","METRO","MEYSU","MGROS",
    "MHRGY","MIATK","MILAS","MIPAZ","MMCAS","MNDRS","MNDTR","MOBTL","MOGAN","MPARK","MRGYO","MRSHL","MSGYO","MTRKS","MTRYO",
    "MZHLD","NATEN","NETAS","NIBAS","NTGAZ","NTHOL","NUGYO","NUHCM","OBAMS","ODAS","OFSYM","ONCSM","ORCAY","ORGE","ORMA",
    "OSMEN","OSTIM","OTKAR","OTTO","OYAKC","OZGYO","OZKGY","OZSUB","PAGYO","PAMEL","PAPIL","PARSN","PASEU","PAYSN","PENGD",
    "PENTA","PETKM","PETUN","PGSUS","PINSU","PKART","PLTUR","PNLSN","PNSUT","POLHO","POLTK","PRKAB","PRKME","PRZMA","PSDTC",
    "PSGYO","QNBFB","QNBFL","QUAGR","RALYH","RAYSG","RGYAS","RNPOL","RODRG","RTALB","RUBNS","RYGYO","RYSAS","SAHOL","SAMAT",
    "SANEL","SANFM","SANKO","SARKY","SASA","SAYAS","SDTTR","SEGYO","SEKFK","SEKUR","SELEC","SELGD","SERNT","SEYKM","SILVR",
    "SISE","SKBNK","SKTAS","SKYMD","SMART","SMRTG","SNGYO","SNPAM","SODSN","SOKM","SONME","SRVGY","SUMAS","SUNTK","SURGY",
    "SUWEN","SVGYO","TABGD","TARKM","TATEN","TATGD","TAVHL","TCELL","TCKRC","TDGYO","TEKTU","TERA","TETMT","TEZOL","TGSAS",
    "THYAO","TIBET","TKFEN","TKNSA","TLMAN","TMPOL","TMSN","TOASO","TRCAS","TRGYO","TRILC","TSKB","TSPOR","TTKOM","TTRAK",
    "TUCLK","TUKAS","TUPRS","TUREX","TURGG","TURSG","ULAS","ULKER","ULUFA","ULUSE","UNLU","USAK","VAKBN","VAKKO","VANGD",
    "VBTYZ","VERUS","VESBE","VESTL","VKFYO","VKING","VRGYO","YAPRK","YATAS","YAYLA","YEOTK","YESIL","YGGYO","YGYO","YKBNK",
    "YKSLN","YONGA","YUNSA","YYLGD","ZEDUR","ZOREN","ZRGYO",
]
BIST_YF = {c: f"{c}.IS" for c in BIST_TUM}

# ════════════════════════════════════════════════════════════
# 3. RİSK KONFİGÜRASYONU
# ════════════════════════════════════════════════════════════
DEFAULT_RISK = {
    "risk_per_trade": 0.01, "sl_atr": 2.5, "trail_act_r": 1.5, "trail_atr": 2.5,
    "p_buy": 0.45, "p_sell": 0.45, "p_buy_long": 0.40, "margin": 0.10, "max_pos": 0.15,
    "use_regime": True, "use_long_gate": True, "max_hold": 120, "max_dd": 15.0,
    "daily_loss": 3.0, "max_trades_day": 5, "loss_streak": 3, "cooldown_h": 24,
    "gate_on": True, "gate_min_n": 8, "auto_adopt": True, "unc_max": 0.25,
    "conf_sizing": True, "max_positions": 8, "max_exposure": 0.85, "max_sector": 0.40,
    "regime_bull_adj": -0.03, "regime_bear_adj": 0.08, "regime_range_adj": 0.03, "regime_vol_adj": 0.10,
    "si_auto": True, "si_interval_min": 30, "si_budget_s": 30,
    "min_daily_turnover": 5_000_000, "dsr_confidence": 0.90, "consec_improve": 2, "test_score_tol": 0.25,
    "tp_on": True, "tp_r": 2.0, "tp_frac": 0.5, "be_on": True, "be_r": 1.0,
    "dd_throttle": True, "corr_on": True, "corr_max": 0.85,
    "overlay_on": True, "rl_veto": True, "rl_veto_p": 0.25, "fed_beta": 0.5,
}
DEFAULT_RISK.update(SEKTOR_RISK)
REGIMES = ["BULL", "BEAR", "VOL", "RANGE"]
RISK_BOUNDS = {
    "risk_per_trade": (0.002, 0.05), "max_pos": (0.05, 0.50), "max_exposure": (0.2, 1.0), "max_sector": (0.1, 1.0),
    "max_positions": (1, 20), "sl_atr": (1.0, 6.0), "trail_act_r": (0.5, 4.0), "trail_atr": (1.0, 5.0),
    "p_buy": (0.34, 0.80), "p_sell": (0.34, 0.80), "p_buy_long": (0.34, 0.80), "margin": (0.0, 0.40),
    "unc_max": (0.05, 0.50), "max_hold": (5, 400), "max_dd": (3.0, 50.0), "daily_loss": (0.5, 15.0),
    "max_trades_day": (1, 30), "loss_streak": (1, 10), "cooldown_h": (1, 240), "gate_min_n": (3, 100),
    "regime_bull_adj": (-0.2, 0.3), "regime_bear_adj": (-0.2, 0.3), "regime_range_adj": (-0.2, 0.3),
    "regime_vol_adj": (-0.2, 0.3), "si_interval_min": (5, 720), "si_budget_s": (2, 120),
    "min_daily_turnover": (0, 1e9), "dsr_confidence": (0.5, 0.999), "consec_improve": (1, 5),
    "test_score_tol": (0.0, 2.0), "tp_r": (0.5, 6.0), "tp_frac": (0.1, 0.9), "be_r": (0.3, 3.0),
    "corr_max": (0.5, 0.99), "rl_veto_p": (0.05, 0.6), "fed_beta": (0.0, 1.0),
}
INT_KEYS = {"max_positions", "max_hold", "max_trades_day", "loss_streak", "cooldown_h", "gate_min_n",
            "si_interval_min", "si_budget_s", "consec_improve"}
BOOL_KEYS = ("use_regime", "use_long_gate", "conf_sizing", "tp_on", "be_on", "dd_throttle", "corr_on",
             "gate_on", "auto_adopt", "si_auto", "overlay_on", "rl_veto")
TUNE_BOUNDS = {
    "p_buy": (0.36, 0.75), "p_buy_long": (0.36, 0.75), "margin": (0.0, 0.35), "unc_max": (0.05, 0.50),
    "sl_atr": (1.0, 5.0), "trail_act_r": (0.5, 4.0), "trail_atr": (1.0, 5.0), "tp_r": (0.5, 6.0),
    "tp_frac": (0.1, 0.9), "be_r": (0.3, 3.0), "max_hold": (10, 200), "risk_per_trade": (0.003, 0.025),
    "max_pos": (0.05, 0.35), "conf_sizing": (0, 1), "regime_bull_adj": (-0.10, 0.15),
    "regime_bear_adj": (0.0, 0.25), "regime_range_adj": (-0.05, 0.15), "regime_vol_adj": (0.0, 0.25),
    "min_daily_turnover": (0, 5e7), "use_regime": (0, 1), "use_long_gate": (0, 1), "cooldown_h": (6, 72),
    "max_trades_day": (2, 15), "loss_streak": (2, 6), "corr_max": (0.6, 0.95),
}
TUNE_BINARY = {"conf_sizing", "use_regime", "use_long_gate"}
TUNE_INT = {"max_hold", "max_trades_day", "loss_streak", "cooldown_h"}
LOCKED_KEYS = ("max_dd", "daily_loss", "max_positions", "max_exposure", "max_sector")

# ════════════════════════════════════════════════════════════
# 4. YARDIMCI + VERİ FONKSİYONLARI (BUG 1 DÜZELTİLDİ: _fetch/sanitize YUKARI)
# ════════════════════════════════════════════════════════════
def now_tr(): return datetime.now(TR)

def borsa_acik():
    n = now_tr()
    return n.weekday() < 5 and dtime(9, 55) <= n.time() <= dtime(18, 10)

def safe_float(x, d=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else d
    except (TypeError, ValueError):
        return d

def clamp(v, lo, hi): return max(lo, min(hi, v))

def validate_risk(rp):
    out = {**DEFAULT_RISK, **(rp or {})}
    for k, (lo, hi) in RISK_BOUNDS.items():
        out[k] = clamp(safe_float(out.get(k), DEFAULT_RISK.get(k, 0)), lo, hi)
        if k in INT_KEYS: out[k] = int(round(out[k]))
    for k in BOOL_KEYS: out[k] = bool(out.get(k, True))
    return out

def tick_size(p):
    for lim, t in ((20, .01), (50, .02), (100, .05), (250, .10), (500, .25), (1000, .5), (2500, 1.0)):
        if p < lim: return t
    return 2.5

def tick_round(p, up=True):
    t = tick_size(p)
    return round((math.ceil(p / t - 1e-9) if up else math.floor(p / t + 1e-9)) * t, 4)

def bday_count(a, b):
    try: return int(np.busday_count(np.datetime64(a.date()), np.datetime64(b.date())))
    except Exception: return (b - a).days

# ─── 🐛 BUG 1 DÜZELTME: _fetch/sanitize ARTIK EN BAŞTA ──────
def sanitize(d):
    cols = ["Open", "High", "Low", "Close"]
    d = d.dropna(subset=cols).drop_duplicates("Date").sort_values("Date")
    d = d[(d[cols] > 0).all(axis=1)].copy()
    d["High"], d["Low"] = d[cols].max(axis=1), d[cols].min(axis=1)
    d["Volume"] = d["Volume"].fillna(0).clip(lower=0)
    d = d[d["Close"].pct_change().abs().fillna(0) < 0.5]
    return d.reset_index(drop=True)

def _fetch(sym, period="5y"):
    for _ in range(3):
        try:
            d = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
            if d is not None and not d.empty:
                d = d.reset_index()
                d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
                return sanitize(d[["Date", "Open", "High", "Low", "Close", "Volume"]])
        except Exception:
            time.sleep(0.5)
    return None

def _fetch_hizli(sym, period="6mo"):
    try:
        d = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
        if d is None or len(d) < 100: return None
        d = d.reset_index()
        d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
        return d[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception: return None

def _titles(sym):
    out = []
    try:
        for n in (yf.Ticker(sym).news or []):
            t = n.get("title") or (n.get("content") or {}).get("title") or ""
            if t: out.append(t)
    except Exception: pass
    return out[:NEWS_MAX_PER_STOCK]

# ════════════════════════════════════════════════════════════
# 5. DB (thread-safe)
# ════════════════════════════════════════════════════════════
_DB_LOCK = threading.RLock()

@st.cache_resource(show_spinner=False)
def _db():
    c = sqlite3.connect(DB_FILE, timeout=30.0, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA synchronous=NORMAL")
    c.executescript("""
        CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, stock TEXT, action TEXT, price REAL, qty INTEGER, pnl REAL, reason TEXT);
        CREATE TABLE IF NOT EXISTS equity (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, value REAL, cash REAL, npos INTEGER);
        CREATE TABLE IF NOT EXISTS train (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, step INTEGER, vl REAL, va REAL, note TEXT);
        CREATE TABLE IF NOT EXISTS improve (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, reason TEXT, detail TEXT, before REAL, after REAL, adopted INTEGER);
        CREATE TABLE IF NOT EXISTS calibrate (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, n INTEGER, brier REAL, ece REAL, acc REAL, base REAL);
        CREATE TABLE IF NOT EXISTS cpcv (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, mean REAL, std REAL, worst REAL, best REAL, n INTEGER, sharpe REAL);
        CREATE TABLE IF NOT EXISTS pbo (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, pbo REAL, mean REAL, median REAL, n INTEGER, interp TEXT);
        CREATE TABLE IF NOT EXISTS forecast (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, stock TEXT, current_price REAL, target_price REAL,
            expected_return REAL, lower_bound REAL, upper_bound REAL, p_up REAL, p_down REAL, p_flat REAL, bull_price REAL, base_price REAL,
            bear_price REAL, confidence REAL, uncertainty REAL, regime TEXT, horizon INTEGER);
        CREATE INDEX IF NOT EXISTS ix_fc ON forecast(stock, id);
    """)
    c.commit()
    return c

@contextmanager
def db_cur():
    with _DB_LOCK:
        c = _db()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback(); raise

def _ts(): return now_tr().isoformat(timespec="seconds")

def _x(sql, args=()):
    try:
        with db_cur() as c: c.execute(sql, args)
    except Exception as e: log.warning(f"db: {e}")

def _q(sql, args=()):
    try:
        with db_cur() as c: return c.execute(sql, args).fetchall()
    except Exception as e:
        log.warning(f"db: {e}"); return []

def db_add_trade(s, a, p, q, pnl, r):
    _x("INSERT INTO trades (ts,stock,action,price,qty,pnl,reason) VALUES (?,?,?,?,?,?,?)",
       (_ts(), str(s)[:20], str(a)[:10], float(p), int(q), float(pnl or 0), str(r)[:300]))

def db_trades(limit=300):
    r = _q("SELECT ts,stock,action,price,qty,pnl,reason FROM trades ORDER BY id DESC LIMIT ?", (int(limit),))
    return [dict(zip(["ts","stock","action","price","qty","pnl","reason"], x)) for x in r]

def db_eq_add(v, cash, npos):
    _x("INSERT INTO equity (ts,value,cash,npos) VALUES (?,?,?,?)", (_ts(), float(v), float(cash), int(npos)))

def db_eq(limit=2000):
    r = _q("SELECT ts,value,cash,npos FROM equity ORDER BY id DESC LIMIT ?", (int(limit),))
    return [{"ts": x[0], "value": x[1], "cash": x[2], "npos": x[3]} for x in reversed(r)]

def db_train(limit=50):
    r = _q("SELECT ts,step,vl,va,note FROM train ORDER BY id DESC LIMIT ?", (int(limit),))
    return [dict(zip(["ts","step","vl","va","note"], x)) for x in r]

def db_log_train(step, vl, va, note=""):
    _x("INSERT INTO train (ts,step,vl,va,note) VALUES (?,?,?,?,?)", (_ts(), int(step), float(vl), float(va), note))

def db_improve_add(reason, detail, before, after, adopted):
    _x("INSERT INTO improve (ts,reason,detail,before,after,adopted) VALUES (?,?,?,?,?,?)",
       (_ts(), str(reason)[:30], str(detail)[:500], float(before), float(after), int(adopted)))

def db_calib_add(n, brier, ece, acc, base):
    _x("INSERT INTO calibrate (ts,n,brier,ece,acc,base) VALUES (?,?,?,?,?,?)", (_ts(), int(n), float(brier), float(ece), float(acc), float(base)))

def db_cpcv_add(mean, std, worst, best, n, sharpe):
    _x("INSERT INTO cpcv (ts,mean,std,worst,best,n,sharpe) VALUES (?,?,?,?,?,?,?)",
       (_ts(), float(mean), float(std), float(worst), float(best), int(n), float(sharpe)))

def db_pbo_add(pbo, mean, median, n, interp):
    _x("INSERT INTO pbo (ts,pbo,mean,median,n,interp) VALUES (?,?,?,?,?,?)",
       (_ts(), float(pbo), float(mean), float(median), int(n), str(interp)[:50]))

def db_forecast_add(f):
    _x("""INSERT INTO forecast (ts,stock,current_price,target_price,expected_return,lower_bound,upper_bound,p_up,p_down,p_flat,
         bull_price,base_price,bear_price,confidence,uncertainty,regime,horizon) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
       (_ts(), f.stock, f.current_price, f.target_price, f.expected_return_pct, f.lower_bound, f.upper_bound, f.p_up, f.p_down,
        f.p_flat, f.bull_price, f.base_price, f.bear_price, f.confidence, f.uncertainty, f.regime, f.horizon_days))

def db_forecast_history(stock, limit=50):
    r = _q("SELECT ts,current_price,target_price,expected_return,confidence FROM forecast WHERE stock=? ORDER BY id DESC LIMIT ?",
           (stock, int(limit)))
    return [dict(zip(["ts","current","target","ret","conf"], x)) for x in r]

def db_clear():
    for t in ("trades","equity","train","improve","calibrate","cpcv","pbo","forecast"):
        _x(f"DELETE FROM {t}")

# ════════════════════════════════════════════════════════════
# 6. LLM HABER (BUG 3 düzeltildi: LLM çağrıları batch'lenir)
# ════════════════════════════════════════════════════════════
NEWS_PROMPT = """Sen BIST haber analistisin. {stock} hissesi için:
Haber: {title}
JSON ver (başka bir şey yazma):
{{"sentiment":<-1..1>,"confidence":<0..1>}}"""

class NewsAnalyzer:
    def __init__(self):
        self.cache, self.client = {}, None
        self._lock = threading.Lock()
        self._llm_calls_this_cycle = 0
        self.enabled = NEWS_ENABLED and GROQ_OK and bool(GROQ_API_KEY)
        self._load()
        if self.enabled:
            try: self.client = Groq(api_key=GROQ_API_KEY)
            except Exception as e:
                log.warning(f"Groq init: {e}"); self.enabled = False

    def reset_cycle(self): self._llm_calls_this_cycle = 0

    def _load(self):
        try:
            if os.path.exists(NEWS_FILE):
                with open(NEWS_FILE, "r", encoding="utf-8") as f: data = json.load(f)
                now = time.time()
                self.cache = {h: v for h, v in data.items() if now - v.get("ts", 0) < NEWS_CACHE_HOURS * 3600}
        except Exception: pass

    def _save(self):
        try:
            with open(NEWS_FILE, "w", encoding="utf-8") as f: json.dump(self.cache, f, ensure_ascii=False)
        except Exception: pass

    def analyze(self, stock, title):
        if not self.enabled or not title.strip(): return None
        h = hashlib.md5((stock + title).encode()).hexdigest()[:12]
        with self._lock:
            c = self.cache.get(h)
            if c: return (c["sentiment"], c["confidence"])
            if self._llm_calls_this_cycle >= NEWS_LLM_PER_CYCLE:
                return None
            self._llm_calls_this_cycle += 1
        try:
            r = self.client.chat.completions.create(
                model=GROQ_MODEL, temperature=0.1, max_tokens=80,
                messages=[{"role": "user", "content": NEWS_PROMPT.format(stock=stock, title=title[:400])}],
                response_format={"type": "json_object"})
            d = json.loads(r.choices[0].message.content)
            item = {"sentiment": clamp(float(d.get("sentiment", 0)), -1, 1),
                    "confidence": clamp(float(d.get("confidence", 0.5)), 0, 1), "ts": time.time()}
            with self._lock:
                self.cache[h] = item
                if len(self.cache) % 10 == 0: self._save()
            return (item["sentiment"], item["confidence"])
        except Exception as e:
            log.warning(f"LLM: {e}"); return None

    def sentiment(self, stock, titles):
        items = [r for r in (self.analyze(stock, t) for t in titles[:NEWS_MAX_PER_STOCK]) if r]
        if not items: return 0.0, 0.0, 0
        s = np.array([i[0] for i in items]); c = np.array([i[1] for i in items])
        return float((s * c).sum() / (c.sum() + 1e-9)), float(c.mean()), len(items)

@st.cache_resource(show_spinner=False)
def get_news(): return NewsAnalyzer()

# ════════════════════════════════════════════════════════════
# 7. KAP
# ════════════════════════════════════════════════════════════
class KAPFetcher:
    KW = ["esas sözleşme", "temettü", "sermaye", "birleşme", "satın alma", "yeni iş", "geri alım", "bedelsiz"]
    def __init__(self):
        self.cache = {}
        try:
            if os.path.exists(KAP_FILE):
                with open(KAP_FILE, "r", encoding="utf-8") as f: self.cache = json.load(f)
        except Exception: pass

    def _save(self):
        try:
            with open(KAP_FILE, "w", encoding="utf-8") as f: json.dump(self.cache, f, ensure_ascii=False)
        except Exception: pass

    def fetch(self, code, days=7):
        key = f"{code}_{days}"
        if key in self.cache and time.time() - self.cache[key].get("_ts", 0) < 3600:
            return self.cache[key].get("items", [])
        try:
            url = f"https://www.kap.org.tr/tr/api/disclosures?mkkMemberOidList={code}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read().decode("utf-8"))
            items = [{"title": str(d.get("subject", ""))[:200], "date": d.get("publishDate", "")}
                     for d in (data if isinstance(data, list) else data.get("items", []))]
            self.cache[key] = {"items": items, "_ts": time.time()}; self._save()
            return items
        except Exception:
            self.cache[key] = {"items": [], "_ts": time.time()}
            return []

    def get_features(self, code):
        items = self.fetch(code)
        if not items: return np.zeros(3, np.float32)
        n = min(len(items), 20)
        score = sum(1 for it in items[:n] if any(k in it["title"].lower() for k in self.KW))
        return np.array([min(n / 10.0, 2.0), min(score / 5.0, 2.0), 1.0], np.float32)

@st.cache_resource(show_spinner=False)
def get_kap(): return KAPFetcher()

# ════════════════════════════════════════════════════════════
# 8. TEMEL VERİ (overlay)
# ════════════════════════════════════════════════════════════
@dataclass
class FundamentalData:
    pe: float = 0.0; pb: float = 0.0; ps: float = 0.0; ev_ebitda: float = 0.0
    roe: float = 0.0; debt_eq: float = 0.0; div_yield: float = 0.0; market_cap: float = 0.0; updated: str = ""

class FundamentalFetcher:
    def __init__(self):
        self.cache, self._lock = {}, threading.Lock()
        try:
            if os.path.exists(FUND_FILE):
                with open(FUND_FILE, "r") as f: self.cache = json.load(f)
        except Exception: pass

    def _save(self):
        try:
            with open(FUND_FILE, "w") as f: json.dump(self.cache, f)
        except Exception: pass

    def _fresh(self, code):
        ts = self.cache.get(code, {}).get("updated", "")
        try: return bool(ts) and (now_tr() - datetime.fromisoformat(ts)).days < 7
        except Exception: return False

    def fetch(self, code, force=False):
        if not force and self._fresh(code): return FundamentalData(**self.cache[code])
        try:
            info = yf.Ticker(f"{code}.IS").info or {}
            fd = FundamentalData(
                pe=safe_float(info.get("trailingPE")), pb=safe_float(info.get("priceToBook")),
                ps=safe_float(info.get("priceToSalesTrailing12Months")), ev_ebitda=safe_float(info.get("enterpriseToEbitda")),
                roe=safe_float(info.get("returnOnEquity")), debt_eq=safe_float(info.get("debtToEquity")),
                div_yield=safe_float(info.get("dividendYield")), market_cap=safe_float(info.get("marketCap")),
                updated=now_tr().isoformat(timespec="seconds"))
            with self._lock:
                self.cache[code] = fd.__dict__
                self._save()
            return fd
        except Exception:
            return FundamentalData(**self.cache[code]) if code in self.cache else FundamentalData()

@st.cache_resource(show_spinner=False)
def get_fund(): return FundamentalFetcher()

def fund_score(fd, sektor=""):
    s = 0.0
    if 0 < fd.pe < 12: s += 0.35
    elif fd.pe > 35: s -= 0.35
    if 0 < fd.pb < 1.5: s += 0.2
    elif fd.pb > 6: s -= 0.2
    if fd.roe > 0.15: s += 0.3
    elif fd.roe < 0: s -= 0.3
    if sektor not in ("Bankacılık", "Finans", "Sigorta") and fd.debt_eq > 250: s -= 0.2
    return clamp(s, -1.0, 1.0)

# ════════════════════════════════════════════════════════════
# 9. MAKRO (BUG 1 DÜZELTİLDİ: _fetch artık yukarıda)
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_macro_hist():
    syms = {"usdtry": "USDTRY=X", "eurtry": "EURTRY=X", "gold": "GC=F", "brent": "BZ=F"}
    cols = {}
    for k, s in syms.items():
        d = _fetch(s, "5y")  # ✅ Artık çalışıyor
        if d is not None and len(d) > 100: cols[k] = d.set_index("Date")["Close"]
    if not cols: return None
    m = pd.DataFrame(cols).sort_index()
    return m[~m.index.duplicated()].ffill()

def load_macro():
    m = load_macro_hist()
    if m is None or len(m) < 25: return {}
    out = {}
    for k in m.columns:
        out[k] = float(m[k].iloc[-1]); out[f"{k}_ret20"] = float(m[k].iloc[-1] / m[k].iloc[-21] - 1)
    return out

def add_macro(df, macro):
    df = df.copy()
    idx = pd.DatetimeIndex(df["Date"])
    for k in MACRO_KEYS:
        if macro is None or k not in macro.columns:
            df[f"M_{k}_z"] = 0.0; df[f"M_{k}_r20"] = 0.0; continue
        s = macro[k].reindex(macro.index.union(idx)).ffill().reindex(idx).shift(1)
        s = pd.Series(s.to_numpy(), index=df.index)
        ls = np.log(s.where(s > 0))
        z = (ls - ls.rolling(250, min_periods=60).mean()) / ls.rolling(250, min_periods=60).std().replace(0, np.nan)
        df[f"M_{k}_z"] = z.fillna(0.0)
        df[f"M_{k}_r20"] = s.pct_change(20, fill_method=None).fillna(0.0)
    return df.replace([np.inf, -np.inf], 0.0)

# ════════════════════════════════════════════════════════════
# 10. EXPERIENCE BUFFER
# ════════════════════════════════════════════════════════════
class ExperienceBuffer:
    def __init__(self, capacity=5000, n_feat=N_FEAT):
        self.capacity, self.n_feat = capacity, n_feat
        self.X = np.zeros((capacity, n_feat), dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.int64)
        self.y_long = np.full(capacity, -1, dtype=np.int64)
        self.p = np.zeros(capacity, dtype=np.float32)
        self.ptr = self.n = self.total_added = 0

    def add(self, feat, label, label_long=-1, priority=1.0):
        self.X[self.ptr] = np.asarray(feat, np.float32)
        self.y[self.ptr], self.y_long[self.ptr] = int(label), int(label_long)
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
                np.savez_compressed(f, X=self.X[:self.n], y=self.y[:self.n], y_long=self.y_long[:self.n], p=self.p[:self.n])
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

# ════════════════════════════════════════════════════════════
# 11. FEDERATED
# ════════════════════════════════════════════════════════════
class FederatedAggregator:
    def __init__(self, storage_dir="federated"):
        self.dir = storage_dir
        os.makedirs(self.dir, exist_ok=True)
        self.rounds, self.history = 0, []
        try:
            with open(os.path.join(self.dir, "global.json")) as f:
                m = json.load(f); self.rounds, self.history = m.get("rounds", 0), m.get("history", [])
        except Exception: pass

    def _p(self, name): return os.path.join(self.dir, name)

    @staticmethod
    def _flat(nets_state):
        out = {}
        for i, n in enumerate(nets_state):
            for k, v in n["P"].items(): out[f"P{i}__{k}"] = np.asarray(v, np.float32)
            for k, v in n["E"].items(): out[f"E{i}__{k}"] = np.asarray(v, np.float32)
        return out

    def push(self, bot_id, nn_state, score):
        try:
            flat = self._flat(nn_state["nets"])
            tmp = self._p(f"bot_{bot_id}.npz.tmp")
            with open(tmp, "wb") as f: np.savez_compressed(f, **flat)
            os.replace(tmp, self._p(f"bot_{bot_id}.npz"))
            with open(self._p(f"bot_{bot_id}.json"), "w") as f:
                json.dump({"bot_id": bot_id, "score": float(score), "ts": _ts(), "n_nets": len(nn_state["nets"]),
                           "n_feat": N_FEAT}, f)
            return True
        except Exception as e:
            log.warning(f"federated push {bot_id}: {e}"); return False

    def aggregate(self):
        bots = []
        for bid in (1, 2, 3):
            try:
                with open(self._p(f"bot_{bid}.json")) as f: meta = json.load(f)
                if meta.get("n_feat") != N_FEAT: continue
                with np.load(self._p(f"bot_{bid}.npz"), allow_pickle=False) as z:
                    bots.append((meta, {k: z[k] for k in z.files}))
            except Exception: continue
        if len(bots) < 2: return None
        scores = np.array([max(0.05, b[0]["score"]) for b in bots])
        w = scores / scores.sum()
        keys = set(bots[0][1].keys())
        for _, d in bots[1:]: keys &= set(d.keys())
        glob = {}
        for k in keys:
            try: glob[k] = sum(wi * d[k] for wi, (_, d) in zip(w, bots)).astype(np.float32)
            except Exception: pass
        tmp = self._p("global.npz.tmp")
        with open(tmp, "wb") as f: np.savez_compressed(f, **glob)
        os.replace(tmp, self._p("global.npz"))
        self.rounds += 1
        self.history = (self.history + [{"round": self.rounds, "ts": _ts(), "n_bots": len(bots),
                                         "weights": w.tolist(), "scores": scores.tolist()}])[-50:]
        with open(self._p("global.json"), "w") as f:
            json.dump({"rounds": self.rounds, "history": self.history}, f)
        log.info(f"Federated round {self.rounds}: {len(bots)} bot")
        return glob

    def blend(self, bag, beta=0.5):
        try:
            with np.load(self._p("global.npz"), allow_pickle=False) as z:
                G = {k: z[k] for k in z.files}
        except Exception: return None
        out = []
        for i, nn in enumerate(bag.nets):
            P, E = {}, {}
            for src, dst, pre in ((nn.P, P, "P"), (nn.E, E, "E")):
                for k, loc in src.items():
                    g = G.get(f"{pre}{i}__{k}")
                    if g is None or g.shape != loc.shape: dst[k] = loc.copy(); continue
                    new = ((1 - beta) * loc + beta * g).astype(np.float32)
                    if k in ("Att", "W0"): new[-N_STOCKS:] = loc[-N_STOCKS:]
                    dst[k] = new
            out.append({"P": P, "E": E, "Ts": nn.Ts, "Tl": nn.Tl})
        return out

    def has_global(self): return os.path.exists(self._p("global.npz"))
    def summary(self): return {"rounds": self.rounds, "history": self.history[-10:], "has_global": self.has_global()}

@st.cache_resource(show_spinner=False)
def get_federated(): return FederatedAggregator()

# ════════════════════════════════════════════════════════════
# 12. GÖSTERGELER
# ════════════════════════════════════════════════════════════
def add_ind(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    d = c.diff()
    g = d.clip(lower=0).rolling(14).mean(); ls = (-d.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["MACDs"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDh"] = df["MACD"] - df["MACDs"]
    df["SMA20"], df["SMA50"], df["SMA200"] = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    df["EMA9"], df["EMA21"] = c.ewm(span=9, adjust=False).mean(), c.ewm(span=21, adjust=False).mean()
    df["BBm"] = df["SMA20"]; df["BBs"] = c.rolling(20).std()
    df["BBu"], df["BBd"] = df["BBm"] + 2 * df["BBs"], df["BBm"] - 2 * df["BBs"]
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    rng = (hi14 - lo14).replace(0, np.nan)
    df["K"] = 100 * (c - lo14) / rng
    df["D"] = df["K"].rolling(3).mean()
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
    for k in (1, 5, 10, 20, 30, 60): df[f"Ret{k}"] = c.pct_change(k)
    df["Volatility"] = df["Ret1"].rolling(20).std() * np.sqrt(252)
    df["HL"] = (h - l) / c
    df["Gap"] = (df["Open"] - c.shift()) / c.shift()
    df["HI120"] = c / c.rolling(120).max() - 1
    r20 = (h.rolling(20).max() - l.rolling(20).min()).replace(0, np.nan)
    df["Donch20"] = (c - l.rolling(20).min()) / r20
    df["RSI_slope"] = df["RSI"].diff(3)
    df["SMA50_slope"] = df["SMA50"].pct_change(10)
    df["ATR_chg"] = df["ATRp"] / df["ATRp"].rolling(50).mean() - 1
    df["Turnover20"] = (c * v).rolling(20).mean()
    tp = (h + l + c) / 3
    df["TPz"] = (tp - tp.rolling(20).mean()) / tp.rolling(20).std().replace(0, np.nan)
    mf, dtp = tp * v, tp.diff()
    pmf, nmf = mf.where(dtp > 0, 0.0).rolling(14).sum(), mf.where(dtp < 0, 0.0).rolling(14).sum()
    df["MFI"] = (100 - 100 / (1 + pmf / nmf.replace(0, np.nan))).fillna(50.0)
    df["BBW"] = (df["BBu"] - df["BBd"]) / df["BBm"]
    obv = (np.sign(d).fillna(0) * v).cumsum()
    df["OBV_slope"] = ((obv - obv.shift(10)) / (v.rolling(20).mean().replace(0, np.nan) * 10)).fillna(0.0)
    df["CLV"] = (((c - l) - (h - c)) / (h - l).replace(0, np.nan)).fillna(0.0)
    df["UpRatio10"] = (d > 0).astype(float).rolling(10).mean()
    df["Skew20"] = df["Ret1"].rolling(20).skew()
    s = df.set_index("Date")["Close"]
    w = s.resample("W-FRI").last().dropna()
    wd = w.diff()
    wg, wl = wd.clip(lower=0).rolling(14).mean(), (-wd.clip(upper=0)).rolling(14).mean()
    wk = pd.DataFrame({"WDate": pd.DatetimeIndex(w.index).astype("datetime64[ns]")})
    wk["W_RSI"] = (100 - 100 / (1 + wg / wl.replace(0, np.nan))).to_numpy()
    we9, we21 = w.ewm(span=9, adjust=False).mean(), w.ewm(span=21, adjust=False).mean()
    wk["W_Trend"] = ((we9 - we21) / we21.replace(0, np.nan)).to_numpy()
    wsma = w.rolling(20).mean()
    wk["W_Strength"] = ((w - wsma) / wsma.replace(0, np.nan)).to_numpy()
    wk = wk.replace([np.inf, -np.inf], np.nan).dropna()
    df = pd.merge_asof(df, wk, left_on="Date", right_on="WDate", direction="backward",
                       allow_exact_matches=False).drop(columns=["WDate"])
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
    df["XU100_Ret20"] = 0.0
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

CROSS_RANK_COLS = ["RSI", "Ret5", "Ret20", "Vol_ratio", "RelStr20"]

def build_cross_rank(dfs):
    frames = []
    for s, d in dfs.items():
        if d is None or len(d) == 0: continue
        sub = d[["Date"] + CROSS_RANK_COLS].copy()
        sub["above"] = (d["Close"] > d["SMA50"]).astype(float).to_numpy()
        sub["stock"] = s
        frames.append(sub)
    if not frames: return {}
    all_df = pd.concat(frames, ignore_index=True)
    all_df["Breadth"] = all_df.groupby("Date")["above"].transform("mean")
    for col in CROSS_RANK_COLS:
        all_df[f"rk_{col}"] = all_df.groupby("Date")[col].rank(pct=True)
    keep = ["Date"] + [f"rk_{c}" for c in CROSS_RANK_COLS] + ["Breadth"]
    return {s: g.sort_values("Date")[keep].reset_index(drop=True) for s, g in all_df.groupby("stock")}

def inject_cross_rank(df, rank_df):
    if rank_df is None or len(rank_df) == 0:
        for c in CROSS_RANK_COLS: df[f"rk_{c}"] = 0.5
        df["Breadth"] = 0.5
        return df
    df = df.merge(rank_df, on="Date", how="left")
    for c in CROSS_RANK_COLS: df[f"rk_{c}"] = df[f"rk_{c}"].fillna(0.5)
    df["Breadth"] = df["Breadth"].fillna(0.5)
    return df

# ════════════════════════════════════════════════════════════
# 13. FEAT_NAMES
# ════════════════════════════════════════════════════════════
FEAT_NAMES = [
    "RSI","MACDh","SMA_trend","BB_pos","Stoch","ADX","Volatilite","Vol_ratio","Mom10","HL_ratio","Gap","SMA200_uzak",
    "EMA_trend","C>SMA20","C>SMA50","MACD_poz","SMA20>50","WilliamsR","Mom30","C>BB_mid","ATR%","Ret5","HI120","K-D_fark",
    "W_RSI","W_Trend","W_Strength","Reg_Bull","Reg_Bear","Reg_Vol","Reg_Range","USDTRY_etki","XU100_etki","XU100_kor",
    "Ret20","RelStr20","Donchian20","RSI_egim","SMA50_egim","ATR_degisim","TP_z","MFI","BB_genislik","OBV_egim","CLV",
    "Yukari_oran10","Carpiklik20","Ret60","Piyasa_genislik",
] + ["Turnover_log", "Liquidity_flag"] + [f"Rank_{c}" for c in CROSS_RANK_COLS] \
  + [f"Macro_{k.upper()}_z" for k in MACRO_KEYS] + [f"Macro_{k.upper()}_r20" for k in MACRO_KEYS] \
  + [f"Hisse_{s}" for s in STOCK_LIST]
assert len(FEAT_NAMES) == N_FEAT, f"FEAT_NAMES={len(FEAT_NAMES)} != N_FEAT={N_FEAT}"

# ════════════════════════════════════════════════════════════
# 14. FEATURE MATRIX
# ════════════════════════════════════════════════════════════
def feature_matrix(df, stock_name=None):
    n = len(df)
    g = lambda c, d=0.0: df[c].to_numpy(np.float32) if c in df.columns else np.full(n, d, dtype=np.float32)
    def div(a, b): return np.divide(a, b, out=np.zeros(n, dtype=np.float32), where=(b != 0) & np.isfinite(b))
    cl = np.clip
    c = g("Close")
    sma20, sma50, sma200 = g("SMA20"), g("SMA50"), g("SMA200")
    bbm, bbu, bbd = g("BBm"), g("BBu"), g("BBd")
    turn = g("Turnover20")
    f = [
        g("RSI", 50) / 100, cl(div(g("MACDh"), c) * 200, -1, 1), cl((div(sma20, sma50) - 1) * 10, -1, 1),
        cl(div(c - bbm, bbu - bbd + 1e-9), -1, 1), g("K", 50) / 100, cl(g("ADX") / 50, 0, 1),
        cl(g("Volatility") * 3, 0, 1), cl(g("Vol_ratio", 1) - 1, -1, 1), cl(g("Ret10") * 10, -1, 1),
        cl(g("HL") * 20, 0, 1), cl(g("Gap") * 20, -1, 1), cl((div(c, sma200) - 1) * 5, -1, 1),
        cl(div(g("EMA9") - g("EMA21"), c) * 20, -1, 1),
        (c > sma20).astype(np.float32), (c > sma50).astype(np.float32),
        (g("MACD") > g("MACDs")).astype(np.float32), (sma20 > sma50).astype(np.float32),
        cl(g("WILLR", -50) / 100, -1, 0), cl(g("Ret30") * 10, -1, 1), (c > bbm).astype(np.float32),
        cl(g("ATRp") * 20, 0, 1), cl(g("Ret5") * 10, -1, 1), cl(g("HI120") * 5, -1, 0),
        cl((g("K", 50) - g("D", 50)) / 30, -1, 1), cl(g("W_RSI", 50) / 100, 0, 1), cl(g("W_Trend") * 10, -1, 1),
        cl(g("W_Strength") * 5, -1, 1), g("Regime_Bull_sm"), g("Regime_Bear_sm"), g("Regime_Vol_sm"), g("Regime_Range_sm"),
        cl(g("USDTRY_Ret5") * 20, -1, 1), cl(g("XU100_Ret5") * 10, -1, 1), cl(g("XU100_Corr20"), -1, 1),
        cl(g("Ret20") * 8, -1, 1), cl(g("RelStr20") * 8, -1, 1), cl(g("Donch20", 0.5), 0, 1),
        cl(g("RSI_slope") / 30, -1, 1), cl(g("SMA50_slope") * 10, -1, 1), cl(g("ATR_chg"), -1, 1),
        cl(g("TPz") / 3, -1, 1), cl(g("MFI", 50) / 100, 0, 1), cl(g("BBW") * 5, 0, 1), cl(g("OBV_slope"), -1, 1),
        cl(g("CLV"), -1, 1), cl(g("UpRatio10", 0.5), 0, 1), cl(g("Skew20") / 2, -1, 1), cl(g("Ret60") * 4, -1, 1),
        cl(g("Breadth", 0.5) * 2 - 1, -1, 1),
        cl(np.log1p(np.maximum(turn, 0)) / 20.0, 0, 1), (turn >= LIQ_FLAG_TL).astype(np.float32),
    ]
    f += [cl(g(f"rk_{c_}", 0.5) * 2 - 1, -1, 1) for c_ in CROSS_RANK_COLS]
    f += [cl(g(f"M_{k}_z") / 3, -1, 1) for k in MACRO_KEYS]
    f += [cl(g(f"M_{k}_r20") * 8, -1, 1) for k in MACRO_KEYS]
    M = np.nan_to_num(np.column_stack(f)).astype(np.float32)
    assert M.shape[1] == N_FEAT_TECH, f"teknik özellik sayısı {M.shape[1]} != {N_FEAT_TECH}"
    oh = np.zeros((n, N_STOCKS), dtype=np.float32)
    if stock_name in STOCK_LIST: oh[:, STOCK_LIST.index(stock_name)] = 1.0
    return np.hstack([M, oh]).astype(np.float32)

def regime_code(df):
    return np.column_stack([df["Regime_Bull_sm"], df["Regime_Bear_sm"], df["Regime_Vol_sm"], df["Regime_Range_sm"]]).argmax(1)

# ════════════════════════════════════════════════════════════
# 15. SEQ STORE (🐛 BUG 4 DÜZELTME: SEKTÖR BAZLI — sızıntı yok!)
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def seq_store(sektor_key=""):
    """Sektöre özel sequence cache. Sektör değişince farklı key → farklı cache."""
    return {}

def gather_seq(sid, row, sektor_key=None, L=SEQ_LEN):
    """(sid,row) çiftlerinden her hissenin KENDİ geçmişinden (L,N_FEAT) dizi çıkarır."""
    sektor_key = sektor_key or AKTIF_SEKTOR
    sid, row = np.asarray(sid), np.asarray(row)
    out = np.zeros((len(sid), L, N_FEAT), dtype=np.float32)
    store = seq_store(sektor_key)   # ✅ Sektör bazlı cache
    for s in np.unique(sid):
        F = store.get(int(s))
        if F is None: continue
        m = sid == s
        r = row[m][:, None] - np.arange(L - 1, -1, -1)[None, :]
        out[m] = F[np.clip(r, 0, len(F) - 1)]
    return out

# ════════════════════════════════════════════════════════════
# 16. DATASET
# ════════════════════════════════════════════════════════════
BASE_KEYS = ["X", "y", "y_long", "f", "d", "lmin", "atrp", "rg", "a200", "turn", "sid", "row"]
WIN_KEYS = ["fh", "fl", "fc"]
DS_KEYS = BASE_KEYS + WIN_KEYS

def _triple_barrier(cl_, hi, lo, atr, K, H):
    n = len(cl_); m = n - H
    if m <= 0: return None
    Hh, Ll = sliding_window_view(hi[1:], H)[:m], sliding_window_view(lo[1:], H)[:m]
    e, a = cl_[:m], atr[:m]
    ok = np.isfinite(a) & (a > 0) & (e > 0)
    up, dn = e + K * a, e - K * a
    hu, hd = Hh >= up[:, None], Ll <= dn[:, None]
    fu, fd = np.where(hu.any(1), hu.argmax(1), H), np.where(hd.any(1), hd.argmax(1), H)
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

def make_ds(df, stock=None, sektor_key=None):
    sektor_key = sektor_key or AKTIF_SEKTOR
    n = len(df)
    F = feature_matrix(df, stock)
    sid = STOCK_LIST.index(stock) if stock in STOCK_LIST else -1
    seq_store(sektor_key)[sid] = F   # ✅ Sektör bazlı kayıt
    if n < HORIZON_LONG + 150: return None
    hi, lo = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    cl_, atr = df["Close"].to_numpy(float), df["ATR"].to_numpy(float)
    r_s, r_l = _triple_barrier(cl_, hi, lo, atr, TB_K, HORIZON), _triple_barrier(cl_, hi, lo, atr, TB_K_LONG, HORIZON_LONG)
    if r_s is None or r_l is None: return None
    y_s, f_s, lmin_s = r_s
    m = len(y_s)
    y_l = np.full(m, -1, dtype=np.int64); y_l[:len(r_l[0])] = r_l[0]
    e, a = cl_[:m], atr[:m]
    atrp = np.where(np.isfinite(a) & (a > 0) & (e > 0), a / np.where(e > 0, e, 1.0), 0.02).astype(np.float32)
    turn = df["Turnover20"].to_numpy(float)[:m] if "Turnover20" in df.columns else np.zeros(m)
    return {"X": F[:m], "y": y_s, "y_long": y_l, "f": f_s, "d": df["Date"].to_numpy()[:m], "lmin": lmin_s, "atrp": atrp,
            "rg": regime_code(df)[:m].astype(np.int64), "a200": (cl_ > df["SMA200"].to_numpy(float))[:m],
            "turn": turn.astype(np.float32), "sid": np.full(m, sid, dtype=np.int64), "row": np.arange(m, dtype=np.int64),
            "fh": _fwd_window(hi, m, cl_[:m]), "fl": _fwd_window(lo, m, cl_[:m]), "fc": _fwd_window(cl_, m, cl_[:m])}

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
    order = np.argsort(d0, kind="stable"); d = d0[order]; n_u = len(ud)
    te_start = ud[int(n_u * (1 - test_frac))]
    va_start = ud[int(n_u * (1 - test_frac - val_frac))]
    gap = np.timedelta64(int(purge * 1.5) + 1, "D")
    va_end = te_start - gap
    trm, vam, tem = d < (va_start - gap), (d >= va_start) & (d < va_end), d >= te_start
    if trm.sum() < 300 or vam.sum() < 100 or tem.sum() < 100: return None
    take = lambda m, keys: {k: ds[k][order[m]] for k in keys}
    tr, va, te = take(trm, BASE_KEYS), take(vam, DS_KEYS), take(tem, DS_KEYS)
    va = truncate_windows(va, ud, va_end)
    ut = np.unique(tr["d"])
    wtr = (0.5 + np.searchsorted(ut, tr["d"]) / max(1, len(ut) - 1)).astype(np.float32)
    return {"tr": tr, "va": va, "te": te, "Xtr": tr["X"], "ytr": tr["y"], "ytr_long": tr["y_long"], "wtr": wtr,
            "Xva": va["X"], "yva": va["y"], "yva_long": va["y_long"],
            "Xte": te["X"], "yte": te["y"], "yte_long": te["y_long"]}

# ════════════════════════════════════════════════════════════
# 17. UNIVERSE SCANNER
# ════════════════════════════════════════════════════════════
def scan_universe(top_n=40, min_turnover=LIQ_FLAG_TL, period="6mo"):
    t0, results = time.time(), []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_fetch_hizli, BIST_YF[code], period): code for code in BIST_TUM}
        for f in as_completed(futures):
            code = futures[f]
            try:
                d = f.result()
                if d is None or len(d) < 100: continue
                c = d["Close"]
                turnover = (c * d["Volume"]).tail(20).mean()
                if turnover < min_turnover: continue
                sma20, sma50 = c.rolling(20).mean().iloc[-1], c.rolling(50).mean().iloc[-1]
                sma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else sma50
                trend = (1.0 if sma20 > sma50 else 0) + (1.0 if sma50 > sma200 else 0) + (0.5 if c.iloc[-1] > sma20 else 0)
                ret20 = (c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else 0
                ret60 = (c.iloc[-1] / c.iloc[-61] - 1) if len(c) > 61 else 0
                mom = clamp(ret20 * 5 + ret60 * 2, -2, 3)
                tr = pd.concat([d["High"] - d["Low"], (d["High"] - c.shift()).abs(), (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
                atr_pct = tr.rolling(14).mean().iloc[-1] / c.iloc[-1]
                vol_score = clamp(1.0 - abs(atr_pct - 0.025) * 20, 0, 1)
                vr = d["Volume"].tail(5).mean() / (d["Volume"].tail(20).mean() + 1e-9)
                hcm = clamp((vr - 1.0) * 2, -1, 2)
                liq = clamp(np.log1p(turnover / 1e6) / 5, 0, 1.5)
                results.append({"code": code, "total": trend * 1.5 + mom + vol_score * 0.5 + hcm * 0.8 + liq * 1.2,
                                "turnover": turnover, "trend": trend, "mom": mom, "atr_pct": atr_pct,
                                "vol_ratio": vr, "price": float(c.iloc[-1])})
            except Exception: pass
    log.info(f"Universe scan: {len(results)} hisse, {time.time() - t0:.1f}s")
    if not results: return [], []
    results.sort(key=lambda x: x["total"], reverse=True)
    return [r["code"] for r in results[:top_n]], results

# ════════════════════════════════════════════════════════════
# 18. VERİ YÜKLEME
# ════════════════════════════════════════════════════════════
@st.cache_resource(ttl=900, show_spinner=False)
def load_uni(day_key, sektor_key=None):
    sektor_key = sektor_key or AKTIF_SEKTOR
    hd = {c: f"{c}.IS" for c in SEKTOR_BOTLARI[sektor_key]["hisseler"]}
    items = list(hd.items()) + [(f"X:{n}", t) for n, t in CROSS_SYMBOLS.items()]
    res = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_fetch, sym): name for name, sym in items}
        for f in as_completed(futs):
            try: res[futs[f]] = f.result()
            except Exception: res[futs[f]] = None
    cross = {n: res[f"X:{n}"].set_index("Date")["Close"] for n in CROSS_SYMBOLS if res.get(f"X:{n}") is not None}
    macro = None
    try: macro = load_macro_hist()
    except Exception as e: log.warning(f"macro: {e}")
    dfs, errs = {}, []
    for nm in hd:
        d = res.get(nm)
        if d is None: errs.append(nm); continue
        try:
            d = add_macro(add_cross(add_ind(d), cross), macro)
            if len(d) >= 300: dfs[nm] = d
            else: errs.append(f"{nm}(kısa)")
        except Exception as e: errs.append(f"{nm}:{e}")
    try:
        rk = build_cross_rank(dfs)
        for nm in list(dfs): dfs[nm] = inject_cross_rank(dfs[nm], rk.get(nm))
    except Exception as e: errs.append(f"rank:{e}")
    # 🐛 BUG 3 DÜZELTME: LLM çağrılarını reset et + top hisselerle sınırla
    news, fund = {}, {}
    try: get_news().reset_cycle()
    except Exception: pass
    def _nf(nm):
        titles = _titles(hd[nm])
        s = get_news().sentiment(nm, titles) if titles else (0.0, 0.0, 0)
        fd = get_fund().fetch(nm)
        return nm, titles, s, fd
    with ThreadPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(_nf, nm) for nm in dfs]):
            try:
                nm, titles, s, fd = f.result()
                news[nm] = {"titles": titles, "sent": s[0], "conf": s[1], "n": s[2]}
                fund[nm] = fd
            except Exception as e: errs.append(f"nf:{e}")
    return {"dfs": dfs, "errs": errs, "ts": now_tr().strftime("%H:%M"), "news": news, "fund": fund}

@st.cache_data(ttl=30, show_spinner=False)
def live_prices(tickers):
    try:
        tk = list(tickers)
        d = yf.download(tk, period="1d", interval="1m", group_by="ticker", progress=False, threads=True, auto_adjust=True)
        out = {}
        for t in tk:
            try:
                c = (d[t]["Close"] if len(tk) > 1 or isinstance(d.columns, pd.MultiIndex) else d["Close"]).dropna()
                if len(c): out[t] = float(c.iloc[-1])
            except Exception: pass
        return out
    except Exception: return {}

def sig_idx(d, now):
    if len(d) > 1 and d["Date"].iloc[-1].date() == now.date() and now.time() < dtime(18, 20):
        return len(d) - 2
    return len(d) - 1

# ════════════════════════════════════════════════════════════
# 19. SEKTÖR HARİTASI
# ════════════════════════════════════════════════════════════
SEKTOR = {}
for _sek, _lst in {
    "Bankacılık": "GARAN AKBNK ISCTR YKBNK HALKB VAKBN QNBFB TSKB ALBRK SKBNK ICBCT KLNMA",
    "Finans": "QNBFL ISMEN ISFIN", "Holding": "KCHOL SAHOL AGHOL GLYHO", "Sigorta": "TURSG ANSGR AKGRT", "GYO": "HLGYO ISGYO",
    "Havacılık": "THYAO PGSUS TAVHL CLEBI", "Otomotiv": "DOAS FROTO TOASO TTRAK OTKAR ASUZU KARSN BRISA GOODY",
    "Gıda": "TUKAS ULKER CCOLA AEFES", "Perakende": "BIMAS MGROS SOKM MAVI MEPET BIZIM ULUFA",
    "Enerji": "TUPRS PETKM AKSEN ENJSA ENERY ZOREN ODAS AYDEM", "Demir-Çelik": "EREGL KRDMD ISDMR", "Cam": "SISE TRKCM",
    "Kimya": "SASA AKSA GUBRF", "Savunma": "ASELS", "Teknoloji": "LOGO NETAS ARDYZ KAREL KONTR PAPIL FORTE",
    "Telekom": "TCELL TTKOM", "İnşaat": "ENKAI",
}.items():
    for _c in _lst.split(): SEKTOR[_c] = _sek

def sektor_of(code): return SEKTOR.get(code, "Diğer")

def sektor_exposure(positions, prices):
    out = {}
    for s, p in positions.items():
        sek = sektor_of(s)
        out[sek] = out.get(sek, 0.0) + p["qty"] * prices.get(s, p["entry"])
    return out# ════════════════════════════════════════════════════════════
# 🐋 PARÇA 2/4: BEYİN
# NN + Router + Transformer + PPO + HMM + EWC + Reptile + ErrorAnalyzer + BaggedNN
# Düzeltme: BUG 3 — set_state nn_kwargs restore (NAS uyumlu)
# ════════════════════════════════════════════════════════════

def _ln(z, g, b):
    mu = z.mean(1, keepdims=True); xc = z - mu
    std = np.sqrt((xc * xc).mean(1, keepdims=True) + 1e-5)
    return g * (xc / std) + b, (xc / std, std)

def _ln_back(dy, c, g):
    xh, std = c
    dxh = dy * g
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

def _comb_loss_vec(ps, pl, y_s, y_l, alpha=ALPHA_LONG):
    ar = np.arange(len(y_s))
    ls = -np.log(ps[ar, y_s] + 1e-9)
    v = y_l >= 0
    ll = -np.log(pl[ar, np.where(v, y_l, 0)] + 1e-9)
    return np.where(v, alpha * ls + (1 - alpha) * ll, ls)

def short_logloss(P, y): return float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-9)))

def fit_temp_logits(z, y, valid=None):
    if valid is not None: z, y = z[valid], y[valid]
    if len(y) < 10: return 1.0
    Ts = np.linspace(0.5, 3.0, 26, dtype=np.float32)
    zz = z[None, :, :] / Ts[:, None, None]
    zz = zz - zz.max(-1, keepdims=True)
    lse = np.log(np.exp(zz).sum(-1))
    ll = zz[:, np.arange(len(y)), y] - lse
    return float(Ts[int(np.argmax(ll.mean(1)))])

# ════════════════════════════════════════════════════════════
# 1. NN
# ════════════════════════════════════════════════════════════
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
        a = np.abs(att)
        return a / (a.sum() + 1e-9) * len(a)

    @staticmethod
    def _fwd(P, X, drop=0.0, rng=None):
        xa = X * NN._scale(P["Att"])
        n0, c0 = _ln(xa @ P["W0"] + P["b0"], P["g0"], P["e0"]); h0 = np.maximum(n0, 0)
        n1, c1 = _ln(h0 @ P["W1"] + P["b1"], P["g1"], P["e1"]); r1 = np.maximum(n1, 0); k1 = None
        if drop > 0:
            k1 = ((rng.random(r1.shape) > drop) / (1 - drop)).astype(r1.dtype); r1 = r1 * k1
        h1 = h0 + r1
        n2, c2 = _ln(h1 @ P["W2"] + P["b2"], P["g2"], P["e2"]); r2 = np.maximum(n2, 0); k2 = None
        if drop > 0:
            k2 = ((rng.random(r2.shape) > drop) / (1 - drop)).astype(r2.dtype); r2 = r2 * k2
        h2 = h1 + r2
        return h2 @ P["W3s"] + P["b3s"], h2 @ P["W3l"] + P["b3l"], (xa, n0, c0, h0, n1, c1, k1, h1, n2, c2, k2, h2)

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

    def _step(self, X, Ys, Yl, ml, w=None, lr_mult=1.0, clip=1.0, extra=None):
        loss, G = self.loss_grad(self.P, X, Ys, Yl, ml, w, self.dropout, self.rng, self.alpha)
        if extra is not None:
            for k, g in extra(self.P).items(): G[k] = G[k] + g
        gn = math.sqrt(sum(float((g * g).sum()) for g in G.values()))
        sc = min(1.0, clip / (gn + 1e-9))
        self.t += 1
        lr = self.lr * lr_mult
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

    def train_step(self, X, y_s, y_l, w=None, smooth=0.05, lr_mult=1.0, extra=None):
        Yl, ml = _onehot_m(y_l, smooth)
        return self._step(X, _onehot(y_s, smooth), Yl, ml, w, lr_mult, extra=extra)

    def _infer_P(self): return self.E if (self.use_ema and self.t >= 30) else self.P

    def logits_pair(self, X, P=None):
        z3s, z3l, _ = self._fwd(P if P is not None else self._infer_P(), X)
        return z3s, z3l

    def proba_pair(self, X):
        zs, zl = self.logits_pair(np.atleast_2d(X))
        return (_softmax(zs / max(self.Ts, 1e-3)).astype(np.float32), _softmax(zl / max(self.Tl, 1e-3)).astype(np.float32))

    def evaluate(self, X, y_s, y_l=None):
        ps, pl = self.proba_pair(X)
        if y_l is None: y_l = np.full(len(y_s), -1, dtype=np.int64)
        loss = float(_comb_loss_vec(ps, pl, y_s, y_l, self.alpha).mean())
        v = y_l >= 0
        return loss, (float(np.mean(ps.argmax(1) == y_s)), float(np.mean(pl.argmax(1)[v] == y_l[v])) if v.any() else 0.0)

    def distill_from(self, Ys, Yl, X, steps=5):
        ml = np.ones(len(X), dtype=np.float32)
        for _ in range(steps): self._step(X, Ys, Yl, ml, None, lr_mult=0.5)

    def snapshot(self):
        return {"P": {k: v.copy() for k, v in self.P.items()}, "E": {k: v.copy() for k, v in self.E.items()}, "Ts": self.Ts, "Tl": self.Tl}

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
        self.restore(s)
        self.m, self.v = {k: v.copy() for k, v in s["m"].items()}, {k: v.copy() for k, v in s["v"].items()}
        self.t, self.lr, self.dropout, self.wd = s["t"], s["lr"], s["dropout"], s["wd"]
        self.swa_wins = s.get("swa_wins", 0)

def fit_temp_pair(nn, X, y_s, y_l):
    zs, zl = nn.logits_pair(X)
    return fit_temp_logits(zs, y_s), fit_temp_logits(zl, y_l, valid=y_l >= 0)

def fit_model(nn, Xtr, ytr_s, ytr_l, Xva=None, yva_s=None, yva_l=None, steps=300, bs=64, seed=None, warmup=20, sw=None, swa=True):
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
        idx = rng.integers(0, len(Xtr), bs) if cdf is None else np.minimum(np.searchsorted(cdf, rng.random(bs) * cdf[-1]), len(Xtr) - 1)
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
        nn.P = {k: v.copy() for k, v in avg.items()}; nn.E = {k: v.copy() for k, v in avg.items()}
        vl_avg, _ = nn.evaluate(Xe, ye_s, ye_l)
        if vl_avg < bl * 0.999: nn.swa_wins += 1
        else: nn.restore(cur)
    if hv: nn.Ts, nn.Tl = fit_temp_pair(nn, Xe, ye_s, ye_l)
    else: nn.Ts = nn.Tl = 1.0
    return nn

# ════════════════════════════════════════════════════════════
# 2. ROUTER
# ════════════════════════════════════════════════════════════
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

    def loss_grad(self, F, ps, pl, Ys, Yl, ml, alpha=ALPHA_LONG):
        g, (F_, h1) = self.forward(F)
        Ps, Pl = np.einsum("nb,bnc->nc", g, ps), np.einsum("nb,bnc->nc", g, pl)
        n, eps = len(F), 1e-9
        loss = float(alpha * -np.mean(np.sum(Ys * np.log(Ps + eps), 1)) + (1 - alpha) * -np.mean(ml * np.sum(Yl * np.log(Pl + eps), 1)))
        dPs = -alpha * Ys / (Ps + eps) / n
        dPl = -(1 - alpha) * ml[:, None] * Yl / (Pl + eps) / n
        dg = np.einsum("nc,bnc->nb", dPs, ps) + np.einsum("nc,bnc->nb", dPl, pl)
        dz = g * (dg - (dg * g).sum(1, keepdims=True))
        dh1 = (dz @ self.P["W2"].T) * (h1 > 0)
        return loss, {"W1": F_.T @ dh1, "b1": dh1.sum(0), "W2": h1.T @ dz, "b2": dz.sum(0)}

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
        self.m = {k: v.copy() for k, v in s["m"].items()}; self.v = {k: v.copy() for k, v in s["v"].items()}
        self.t, self.lr, self.wd = s["t"], s["lr"], s["wd"]

def _mix(w, p):
    if w.shape[0] == 1: return np.tensordot(w[0], p, axes=1).astype(np.float32)
    return np.einsum("nb,bnc->nc", w, p).astype(np.float32)

# ════════════════════════════════════════════════════════════
# 3. TRANSFORMER
# ════════════════════════════════════════════════════════════
def _lnf(z, g, b, eps=1e-5):
    mu = z.mean(-1, keepdims=True); var = z.var(-1, keepdims=True)
    return g * (z - mu) / np.sqrt(var + eps) + b

class TransformerHead:
    def __init__(self, n_feat=N_FEAT, seq_len=SEQ_LEN, d_model=TRANS_DIM, n_heads=N_HEADS, n_layers=TRANS_LAYERS,
                 hidden=48, seed=None, lr=0.003):
        rg = np.random.default_rng(seed); f32 = np.float32
        self.n_feat, self.seq_len, self.d_model, self.n_heads, self.n_layers = n_feat, seq_len, d_model, n_heads, n_layers
        self.d_k, self.lr, self.t, self.Ts = d_model // n_heads, lr, 0, 1.3
        self.P = {"W_in": (rg.standard_normal((n_feat, d_model)) * np.sqrt(2.0 / n_feat)).astype(f32), "b_in": np.zeros(d_model, f32)}
        pos, div = np.arange(seq_len)[:, None], np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = np.zeros((seq_len, d_model), f32); pe[:, 0::2] = np.sin(pos * div); pe[:, 1::2] = np.cos(pos * div)
        self.PE = pe
        for l in range(n_layers):
            for nm, sh in [("Wq", (d_model, d_model)), ("Wk", (d_model, d_model)), ("Wv", (d_model, d_model)),
                           ("Wo", (d_model, d_model)), ("Wf1", (d_model, d_model * 2)), ("Wf2", (d_model * 2, d_model))]:
                self.P[f"{nm}{l}"] = (rg.standard_normal(sh) * np.sqrt(2.0 / sh[0])).astype(f32)
            for nm, sz in [("bf1", d_model * 2), ("bf2", d_model), ("g1", d_model), ("n1", d_model), ("g2", d_model), ("n2", d_model)]:
                self.P[f"{nm}{l}"] = np.ones(sz, f32) if nm.startswith("g") else np.zeros(sz, f32)
        self.H = {"W1": (rg.standard_normal((2 * d_model, hidden)) * np.sqrt(2.0 / (2 * d_model))).astype(f32), "b1": np.zeros(hidden, f32),
                  "W2": (rg.standard_normal((hidden, 3)) * np.sqrt(2.0 / hidden)).astype(f32), "b2": np.zeros(3, f32)}
        self.m = {k: np.zeros_like(v) for k, v in self.H.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.H.items()}

    def encode(self, X):
        B, L = X.shape[0], self.seq_len
        h = X @ self.P["W_in"] + self.P["b_in"] + self.PE[None]
        for l in range(self.n_layers):
            sp = lambda W: (h @ self.P[f"{W}{l}"]).reshape(B, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
            Q, K, V = sp("Wq"), sp("Wk"), sp("Wv")
            att = _softmax(Q @ K.transpose(0, 1, 3, 2) / math.sqrt(self.d_k))
            ao = (att @ V).transpose(0, 2, 1, 3).reshape(B, L, self.d_model) @ self.P[f"Wo{l}"]
            h = _lnf(h + ao, self.P[f"g1{l}"], self.P[f"n1{l}"])
            ffn = np.maximum(h @ self.P[f"Wf1{l}"] + self.P[f"bf1{l}"], 0) @ self.P[f"Wf2{l}"] + self.P[f"bf2{l}"]
            h = _lnf(h + ffn, self.P[f"g2{l}"], self.P[f"n2{l}"])
        return np.concatenate([h.mean(1), h[:, -1, :]], axis=1).astype(np.float32)

    def _head(self, Z):
        h = np.maximum(Z @ self.H["W1"] + self.H["b1"], 0)
        return h, h @ self.H["W2"] + self.H["b2"]

    def predict_proba(self, X):
        if X.ndim == 2: X = X[None]
        return _softmax(self._head(self.encode(X))[1] / self.Ts).astype(np.float32)

    def update(self, X, y, lr_mult=1.0, w=None):
        Z = self.encode(X); h, lg = self._head(Z); p = _softmax(lg); n = len(y)
        d = (p - _onehot(y, 0.05)) / n
        if w is not None: d = d * w[:, None]
        G = {"W2": h.T @ d, "b2": d.sum(0)}
        dh = (d @ self.H["W2"].T) * (h > 0)
        G["W1"], G["b1"] = Z.T @ dh, dh.sum(0)
        self.t += 1
        b1, b2 = 0.9, 0.999
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for k, g in G.items():
            self.m[k] *= b1; self.m[k] += (1 - b1) * g
            self.v[k] *= b2; self.v[k] += (1 - b2) * g * g
            self.H[k] -= (self.lr * lr_mult * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + 1e-8)).astype(np.float32)
        return float(-np.mean(np.log(p[np.arange(n), y] + 1e-9)))

    def get_state(self):
        return {"P": {k: v.copy() for k, v in self.P.items()}, "H": {k: v.copy() for k, v in self.H.items()},
                "m": {k: v.copy() for k, v in self.m.items()}, "v": {k: v.copy() for k, v in self.v.items()},
                "t": self.t, "lr": self.lr, "Ts": self.Ts}

    def set_state(self, s):
        self.P = {k: v.copy() for k, v in s["P"].items()}; self.H = {k: v.copy() for k, v in s["H"].items()}
        self.m = {k: v.copy() for k, v in s["m"].items()}; self.v = {k: v.copy() for k, v in s["v"].items()}
        self.t, self.lr, self.Ts = s["t"], s["lr"], s.get("Ts", 1.3)

class TransformerEnsemble:
    def __init__(self, n=2, seed=BAG_SEED):
        self.trans = [TransformerHead(seed=seed + 500 + i * 13) for i in range(n)]
        self.weights = np.ones(n, dtype=np.float32) / n
        self.on, self.gain = False, 0.0

    def predict_proba(self, X_seq):
        preds = np.array([t.predict_proba(X_seq) for t in self.trans], dtype=np.float32)
        return np.tensordot(self.weights, preds, axes=1)

    def predict_idx(self, sid, row, chunk=512, sektor_key=None):
        out = []
        for s in range(0, len(sid), chunk):
            out.append(self.predict_proba(gather_seq(sid[s:s + chunk], row[s:s + chunk], sektor_key)))
        return np.concatenate(out) if out else np.zeros((0, 3), np.float32)

    def fit(self, sid, row, y, steps=100, bs=32, seed=42, sektor_key=None):
        rng = np.random.default_rng(seed)
        cnt = np.bincount(y, minlength=3).astype(float) + 1.0
        cw = ((len(y) / (3.0 * cnt)) ** 0.5).astype(np.float32)
        for t in self.trans:
            for _ in range(steps):
                idx = rng.integers(0, len(y), min(bs, len(y)))
                t.update(gather_seq(sid[idx], row[idx], sektor_key), y[idx], w=cw[y[idx]])

    def sample_val(self, sid, row, y, k=1500, seed=3):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), min(k, len(y)), replace=False) if len(y) > k else np.arange(len(y))
        return idx

    def update_w(self, sid, row, y, sektor_key=None):
        ls = []
        for t in self.trans:
            P = np.concatenate([t.predict_proba(gather_seq(sid[s:s + 512], row[s:s + 512], sektor_key))
                                for s in range(0, len(sid), 512)])
            ls.append(short_logloss(P, y))
        inv = 1.0 / (np.array(ls) + 1e-6)
        self.weights = (inv / inv.sum()).astype(np.float32)

    def get_state(self):
        return {"trans": [t.get_state() for t in self.trans], "weights": self.weights.tolist(), "on": self.on, "gain": self.gain}

    def set_state(self, s):
        try:
            for t, ts in zip(self.trans, s["trans"]): t.set_state(ts)
            self.weights = np.array(s["weights"], dtype=np.float32)
            self.on, self.gain = s.get("on", False), s.get("gain", 0.0)
        except Exception as e:
            log.warning(f"trans state: {e}"); self.on = False

# ════════════════════════════════════════════════════════════
# 4. RL (PPO) — VETO katmanı
# ════════════════════════════════════════════════════════════
class PPOAgent:
    def __init__(self, n_in=N_FEAT, n_act=3, h=64, lr=0.001, seed=None, gamma=0.95):
        rg = np.random.default_rng(seed); f32 = np.float32
        self.rng = np.random.default_rng(None if seed is None else seed + 1)
        def W(i, o): return (rg.standard_normal((i, o)) * np.sqrt(2.0 / i)).astype(f32)
        self.P = {"W_pi1": W(n_in, h), "b_pi1": np.zeros(h, f32), "W_pi2": W(h, n_act) * 0.1, "b_pi2": np.zeros(n_act, f32),
                  "W_v1": W(n_in, h), "b_v1": np.zeros(h, f32), "W_v2": W(h, 1) * 0.1, "b_v2": np.zeros(1, f32)}
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.lr, self.gamma, self.t, self.clip = lr, gamma, 0, 0.2

    def _forward(self, X):
        h_pi = np.maximum(X @ self.P["W_pi1"] + self.P["b_pi1"], 0)
        logits = h_pi @ self.P["W_pi2"] + self.P["b_pi2"]
        h_v = np.maximum(X @ self.P["W_v1"] + self.P["b_v1"], 0)
        return _softmax(logits), (h_v @ self.P["W_v2"] + self.P["b_v2"])[:, 0], (h_pi, h_v)

    def proba(self, X): return self._forward(np.atleast_2d(X))[0]

    def act(self, x, greedy=False):
        proba, v, _ = self._forward(np.atleast_2d(x)); p = proba[0]
        a = int(p.argmax()) if greedy else int(self.rng.choice(len(p), p=p / p.sum()))
        return a, float(p[a]), float(v[0])

    def update(self, X, actions, old_probs, rewards, epochs=4, ent=0.01):
        n, ar = len(X), np.arange(len(X))
        old = np.clip(old_probs, 1e-9, 1)
        for _ in range(epochs):
            proba, v, (h_pi, h_v) = self._forward(X)
            adv = rewards - v
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = np.clip(proba[ar, actions], 1e-9, 1) / old
            clipped = ((adv > 0) & (ratio > 1 + self.clip)) | ((adv < 0) & (ratio < 1 - self.clip))
            coef = np.where(clipped, 0.0, -adv * ratio) / n
            d_lg = coef[:, None] * (np.eye(proba.shape[1])[actions] - proba)
            H = -(proba * np.log(proba + 1e-9)).sum(1)
            d_lg += ent * proba * (np.log(proba + 1e-9) + H[:, None]) / n
            d_v = (v - rewards)[:, None] / n
            G = {"W_pi2": h_pi.T @ d_lg, "b_pi2": d_lg.sum(0), "W_v2": h_v.T @ d_v, "b_v2": d_v.sum(0)}
            dh_pi = (d_lg @ self.P["W_pi2"].T) * (h_pi > 0)
            G["W_pi1"], G["b_pi1"] = X.T @ dh_pi, dh_pi.sum(0)
            dh_v = (d_v @ self.P["W_v2"].T) * (h_v > 0)
            G["W_v1"], G["b_v1"] = X.T @ dh_v, dh_v.sum(0)
            self.t += 1
            b1, b2 = 0.9, 0.999
            c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
            for k, g in G.items():
                self.m[k] *= b1; self.m[k] += (1 - b1) * g
                self.v[k] *= b2; self.v[k] += (1 - b2) * g * g
                self.P[k] -= (self.lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + 1e-8)).astype(np.float32)
        return float(rewards.mean())

    def get_state(self):
        return {"P": {k: v.copy() for k, v in self.P.items()}, "m": {k: v.copy() for k, v in self.m.items()},
                "v": {k: v.copy() for k, v in self.v.items()}, "t": self.t, "lr": self.lr, "gamma": self.gamma}

    def set_state(self, s):
        for k in ("P", "m", "v"): setattr(self, k, {kk: vv.copy() for kk, vv in s[k].items()})
        self.t, self.lr, self.gamma = s["t"], s["lr"], s["gamma"]

# ════════════════════════════════════════════════════════════
# 5. HMM
# ════════════════════════════════════════════════════════════
class HMMRegime:
    def __init__(self, n_states=4, seed=42):
        self.n_states = n_states
        self.A = np.full((n_states, n_states), 0.1 / (n_states - 1)); np.fill_diagonal(self.A, 0.9)
        self.mu = np.array([0.001, -0.001, 0.0, 0.0005]); self.sigma = np.array([0.01, 0.015, 0.03, 0.008])
        self.pi = np.ones(n_states) / n_states
        self.fitted, self.n_iter, self.fit_date = False, 0, ""
        self.names = list(REGIMES)

    def _emis(self, obs):
        z = (obs[:, None] - self.mu[None]) / (self.sigma[None] + 1e-9)
        return np.exp(-0.5 * z * z) / (self.sigma[None] * math.sqrt(2 * math.pi) + 1e-9) + 1e-12

    def _fb(self, obs):
        T, N = len(obs), self.n_states
        B = self._emis(obs)
        alpha, c = np.zeros((T, N)), np.zeros(T)
        a = self.pi * B[0]; c[0] = a.sum() + 1e-300; alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * B[t]; c[t] = a.sum() + 1e-300; alpha[t] = a / c[t]
        beta = np.ones((T, N))
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        return alpha, beta, c, B

    def _label(self):
        order_sig = np.argsort(-self.sigma)
        vol = int(order_sig[0]); rest = [i for i in range(self.n_states) if i != vol]
        rest.sort(key=lambda i: self.mu[i])
        names = [None] * self.n_states
        names[vol], names[rest[0]], names[rest[-1]] = "VOL", "BEAR", "BULL"
        for i in rest[1:-1]: names[i] = "RANGE"
        self.names = names

    def fit(self, returns, n_iter=20, date=""):
        obs = np.asarray(returns, dtype=np.float64)
        obs = obs[np.isfinite(obs)]
        if len(obs) < 120: return self
        q = np.quantile(obs, [0.1, 0.4, 0.6, 0.9]); sd = obs.std()
        self.mu = np.array([q[3], q[0], obs.mean(), q[2]]) * 0.3
        self.sigma = np.array([sd * 0.8, sd * 1.0, sd * 2.0, sd * 0.6])
        for it in range(n_iter):
            alpha, beta, c, B = self._fb(obs)
            gamma = alpha * beta; gamma /= gamma.sum(1, keepdims=True) + 1e-300
            xi = np.einsum("ti,ij,tj->ij", alpha[:-1], self.A, (B[1:] * beta[1:]) / c[1:, None])
            self.A = xi / (xi.sum(1, keepdims=True) + 1e-300)
            g_sum = gamma.sum(0) + 1e-12
            self.mu = (gamma * obs[:, None]).sum(0) / g_sum
            self.sigma = np.sqrt((gamma * (obs[:, None] - self.mu[None]) ** 2).sum(0) / g_sum + 1e-8)
            self.pi = gamma[0] / (gamma[0].sum() + 1e-12)
            self.n_iter = it + 1
        self._label()
        self.fitted, self.fit_date = True, date
        return self

    def posterior(self, recent):
        if not self.fitted: return np.ones(self.n_states) / self.n_states
        alpha, _, _, _ = self._fb(np.asarray(recent, dtype=np.float64))
        return alpha[-1]

    def predict_name(self, recent):
        if not self.fitted or len(recent) < 10: return None, 0.0
        p = self.posterior(recent); i = int(p.argmax())
        return self.names[i], float(p[i])

    def get_state(self):
        return {"A": self.A.tolist(), "mu": self.mu.tolist(), "sigma": self.sigma.tolist(), "pi": self.pi.tolist(),
                "fitted": self.fitted, "n_iter": self.n_iter, "names": self.names, "fit_date": self.fit_date}

    def set_state(self, s):
        self.A, self.mu = np.array(s["A"]), np.array(s["mu"])
        self.sigma, self.pi = np.array(s["sigma"]), np.array(s["pi"])
        self.fitted, self.n_iter = s.get("fitted", False), s.get("n_iter", 0)
        self.names, self.fit_date = s.get("names", list(REGIMES)), s.get("fit_date", "")

@st.cache_resource(show_spinner=False)
def get_hmm(): return HMMRegime()

# ════════════════════════════════════════════════════════════
# 6. EWC
# ════════════════════════════════════════════════════════════
class EWC:
    def __init__(self, lam=5.0):
        self.lam, self.fisher, self.optimal, self.n = lam, [], [], 0

    def consolidate(self, nets, X, y_s, y_l, n_samples=240, chunk=8):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)
        X, ys, yl = X[idx], y_s[idx], y_l[idx]
        new_f, new_o = [], []
        for nn in nets:
            acc, nch = {k: np.zeros_like(v) for k, v in nn.P.items()}, 0
            for s in range(0, len(X), chunk):
                Yl, ml = _onehot_m(yl[s:s + chunk], 0.05)
                _, G = NN.loss_grad(nn.P, X[s:s + chunk], _onehot(ys[s:s + chunk], 0.05), Yl, ml, None, 0.0, None, nn.alpha)
                for k, g in G.items(): acc[k] += g * g
                nch += 1
            tot = sum(float(a.sum()) for a in acc.values()) / max(1, sum(a.size for a in acc.values())) + 1e-18
            new_f.append({k: (a / nch / tot).astype(np.float32) for k, a in acc.items()})
            new_o.append({k: v.copy() for k, v in nn.P.items()})
        if self.fisher and len(self.fisher) == len(new_f):
            for i in range(len(new_f)):
                for k in new_f[i]:
                    new_f[i][k] = 0.7 * self.fisher[i][k] + 0.3 * new_f[i][k]
                    new_o[i][k] = 0.5 * self.optimal[i][k] + 0.5 * new_o[i][k]
        self.fisher, self.optimal = new_f, new_o
        self.n += 1
        log.info(f"EWC konsolide: tur {self.n}")

    def penalty(self, i, nn):
        if not self.fisher: return 0.0
        return 0.5 * self.lam * sum(float((self.fisher[i][k] * (nn.P[k] - self.optimal[i][k]) ** 2).sum()) for k in self.fisher[i])

    def extra(self, i):
        if not self.fisher or i >= len(self.fisher): return None
        F, O, lam = self.fisher[i], self.optimal[i], self.lam
        return lambda P: {k: lam * F[k] * (P[k] - O[k]) for k in F if k in P}

    def get_state(self): return {"fisher": self.fisher, "optimal": self.optimal, "lam": self.lam, "n": self.n}

    def set_state(self, s):
        self.fisher, self.optimal = s.get("fisher", []), s.get("optimal", [])
        self.lam, self.n = s.get("lam", 5.0), s.get("n", 0)

# ════════════════════════════════════════════════════════════
# 7. REPTILE
# ════════════════════════════════════════════════════════════
class Reptile:
    def __init__(self, inner_lr=0.01, eps=0.3, n_inner=5):
        self.inner_lr, self.eps, self.n_inner, self.history = inner_lr, eps, n_inner, []

    @staticmethod
    def tasks(X, y_s, y_l, d, n_tasks=5):
        ud = np.unique(d)
        if len(ud) < n_tasks * 10: return []
        chunk, out = len(ud) // n_tasks, []
        for i in range(n_tasks):
            seg = ud[i * chunk:(i + 1) * chunk if i < n_tasks - 1 else len(ud)]
            idx = np.flatnonzero(np.isin(d, seg))
            if len(idx) < 50: continue
            out.append((X[idx], y_s[idx], y_l[idx]))
        return out

    def meta_train(self, bag, sp, n_epochs=3, seed=0):
        tasks = self.tasks(sp["Xtr"], sp["ytr"], sp["ytr_long"], sp["tr"]["d"])
        if len(tasks) < 2: return None
        Xv, yv, ylv = sp["Xva"][:2500], sp["yva"][:2500], sp["yva_long"][:2500]
        rng = np.random.default_rng(seed)
        for ep in range(n_epochs):
            before, states = bag.evaluate(Xv, yv, ylv)[0], [nn.get_state() for nn in bag.nets]
            for nn in bag.nets:
                theta0 = {k: v.copy() for k, v in nn.P.items()}; st0 = nn.get_state(); delta = {k: np.zeros_like(v) for k, v in theta0.items()}
                for (Xt, yt, ylt) in tasks:
                    nn.P = {k: v.copy() for k, v in theta0.items()}
                    lr0 = nn.lr; nn.lr = self.inner_lr
                    for _ in range(self.n_inner):
                        idx = rng.integers(0, len(Xt), min(32, len(Xt)))
                        nn.train_step(Xt[idx], yt[idx], ylt[idx])
                    nn.lr = lr0
                    for k in delta: delta[k] += (nn.P[k] - theta0[k]) / len(tasks)
                nn.set_state(st0)
                nn.P = {k: (theta0[k] + self.eps * delta[k]).astype(np.float32) for k in theta0}
                nn.E = {k: v.copy() for k, v in nn.P.items()}
            after = bag.evaluate(Xv, yv, ylv)[0]
            ok = after <= before * 1.002
            if not ok:
                for nn, s in zip(bag.nets, states): nn.set_state(s)
            self.history.append({"epoch": ep, "before": before, "after": after, "kept": bool(ok), "n_tasks": len(tasks)})
        return self.history

    def get_state(self): return {"inner_lr": self.inner_lr, "eps": self.eps, "n_inner": self.n_inner, "history": self.history[-30:]}

    def set_state(self, s):
        self.inner_lr, self.eps, self.n_inner = s.get("inner_lr", 0.01), s.get("eps", 0.3), s.get("n_inner", 5)
        self.history = s.get("history", [])

# ════════════════════════════════════════════════════════════
# 8. ERROR ANALYZER
# ════════════════════════════════════════════════════════════
class ErrorAnalyzer:
    def __init__(self):
        self.trades = []
        self.reg_mult = {r: 1.0 for r in REGIMES}
        self.patterns = {}
        self.n_analyzed = 0

    def record_trade(self, feat, regime, net):
        self.trades.append({"feat": np.asarray(feat, np.float32).copy(), "regime": regime if regime in REGIMES else "RANGE", "net": float(net)})
        self.trades = self.trades[-600:]

    def analyze(self, min_trades=20):
        if len(self.trades) < min_trades: return None
        self.n_analyzed += 1
        for r in REGIMES:
            sub = [t for t in self.trades if t["regime"] == r]
            if len(sub) >= 8:
                loss_rate = (sum(1 for t in sub if t["net"] < 0) + 1) / (len(sub) + 2)
                self.reg_mult[r] = clamp(1.0 + (0.5 - loss_rate) * 1.2, 0.5, 1.2)
            else:
                self.reg_mult[r] = 1.0
        F = np.array([t["feat"] for t in self.trades]); net = np.array([t["net"] for t in self.trades])
        lose, win = F[net < 0], F[net >= 0]
        self.patterns = {}
        if len(lose) >= 8 and len(win) >= 8:
            sd = np.sqrt((lose.var(0) + win.var(0)) / 2) + 1e-6
            d = (lose.mean(0) - win.mean(0)) / sd
            for i in np.argsort(-np.abs(d))[:5]:
                if i < len(FEAT_NAMES) and i < N_FEAT_TECH:
                    self.patterns[f"{FEAT_NAMES[i]} {'YÜKSEK' if d[i] > 0 else 'DÜŞÜK'}"] = round(float(d[i]), 2)
        return {"n_trades": len(self.trades), "regime_mult": dict(self.reg_mult), "patterns": dict(self.patterns)}

    def regime_mult(self, regime): return float(self.reg_mult.get(regime, 1.0))

    def report(self):
        if len(self.trades) < 10: return "Yetersiz işlem verisi (≥10 kapanmış işlem gerekir)"
        net = np.array([t["net"] for t in self.trades])
        lines = [f"📊 {len(net)} kapanmış işlem · kayıp oranı %{(net < 0).mean() * 100:.0f} · ort. net %{net.mean() * 100:+.2f}",
                 "Rejim çarpanları: " + ", ".join(f"{r} ×{m:.2f}" for r, m in self.reg_mult.items())]
        if self.patterns:
            lines.append("Kayıplı işlemlerde ayrışan özellikler (Cohen d):")
            lines += [f"  • {k}: {v:+.2f}" for k, v in self.patterns.items()]
        return "\n".join(lines)

    def get_state(self):
        return {"trades": self.trades[-300:], "reg_mult": self.reg_mult, "patterns": self.patterns, "n_analyzed": self.n_analyzed}

    def set_state(self, s):
        self.trades, self.patterns = s.get("trades", []), s.get("patterns", {})
        self.reg_mult = {r: s.get("reg_mult", {}).get(r, 1.0) for r in REGIMES}
        self.n_analyzed = s.get("n_analyzed", 0)

# ════════════════════════════════════════════════════════════
# 9. BAGGED NN — 🐛 BUG 3 DÜZELTME: set_state nn_kwargs restore
# ════════════════════════════════════════════════════════════
class BaggedNN:
    def __init__(self, n=BAG_N, seed=BAG_SEED, replay_cap=5000, nn_kwargs=None, replay=None):
        self.nn_kwargs = dict(nn_kwargs or {})
        self.nets = [NN(seed=seed + i * 7, **self.nn_kwargs) for i in range(n)]
        self.perf = np.ones(n, dtype=np.float32) / n
        self.weights = self.perf.copy()
        self.best_vl, self.plateau, self.evo = float("inf"), 0, 0
        self.pbt_acc = self.pbt_n = 0
        self.replay = replay if replay is not None else ExperienceBuffer(replay_cap)
        self.router = RouterNN(N_STOCKS + 2 * n, n, seed=seed + 991) if USE_ROUTER else None
        self.router_on, self.router_gain = False, 0.0
        self.trans_ens = TransformerEnsemble(n=2, seed=seed + 500)
        self.ewc, self.ewc_consolidated = EWC(), False
        self.reptile = Reptile()
        self.error_analyzer = ErrorAnalyzer()

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
        return float(_comb_loss_vec(Ps, Pl, y_s, y_l, ALPHA_LONG).mean()), float(np.mean(Ps.argmax(1) == y_s)), Ps, Pl

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
        return np.array([p[0] for p in pr], dtype=np.float32), np.array([p[1] for p in pr], dtype=np.float32)

    def predict_batch(self, X):
        ps, pl = self._stack(X)
        w = self._weights_for(X, ps, pl)
        return _mix(w, ps), _mix(w, pl)

    def predict_unc(self, X, sid=None, row=None, seq=None):
        ps, pl = self._stack(X)
        w = self._weights_for(X, ps, pl)
        Ps, Pl = _mix(w, ps), _mix(w, pl)
        U = ps.std(axis=0).mean(axis=1)
        if self.trans_ens.on:
            Pt = None
            try:
                if seq is not None: Pt = self.trans_ens.predict_proba(seq)
                elif sid is not None: Pt = self.trans_ens.predict_idx(np.asarray(sid), np.asarray(row))
            except Exception as e:
                log.warning(f"trans pred: {e}")
            if Pt is not None and len(Pt) == len(Ps):
                Ps = (MLP_WEIGHT * Ps + TRANS_WEIGHT * Pt).astype(np.float32)
        return Ps, Pl, U

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
                Xb, ybs, ybl = np.vstack([Xb, s[0]]), np.concatenate([ybs, s[1]]), np.concatenate([ybl, s[2]])
                if wb is not None: wb = np.concatenate([wb, np.full(len(s[1]), float(wb.mean()), np.float32)])
            fit_model(nn, Xb, ybs, ybl, Xva, yva_s, yva_l, steps=steps, seed=seed + i * 7, sw=wb)
        with ThreadPoolExecutor(max_workers=max(1, min(len(self.nets), os.cpu_count() or 2))) as ex:
            list(ex.map(job, range(len(self.nets))))

    def distill(self, X, y_s, y_l):
        bi = int(np.argmax(self.perf)); t = self.nets[bi]
        zs, zl = t.logits_pair(X)
        soft_s, soft_l = _softmax(zs / t.Ts / 2.0).astype(np.float32), _softmax(zl / t.Tl / 2.0).astype(np.float32)
        Ys = 0.5 * soft_s + 0.5 * _onehot(y_s, 0.05)
        Yh, mv = _onehot_m(y_l, 0.05)
        Yl = np.where(mv[:, None] > 0, 0.5 * soft_l + 0.5 * Yh, soft_l).astype(np.float32)
        for i, nn in enumerate(self.nets):
            if i != bi and self.perf[i] < 0.9 * self.perf[bi]: nn.distill_from(Ys, Yl, X)

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
            for i, nn in enumerate(self.nets):
                idx = rg.integers(0, len(Xtr), 64) if cdf is None else np.minimum(np.searchsorted(cdf, rg.random(64) * cdf[-1]), len(Xtr) - 1)
                Xb, ybs, ybl = Xtr[idx], ytr_s[idx], ytr_l[idx]
                s = self.replay.sample(24, rg)
                if s is not None:
                    Xb, ybs, ybl = np.vstack([Xb, s[0]]), np.concatenate([ybs, s[1]]), np.concatenate([ybl, s[2]])
                ex = self.ewc.extra(i) if self.ewc_consolidated else None
                nn.train_step(Xb, ybs, ybl, w=cw[ybs], lr_mult=lr_mult, extra=ex)
        Z1 = self._Z(Xva)
        t1 = [(fit_temp_logits(z[0], yva_s), fit_temp_logits(z[1], yva_l, valid=yva_l >= 0)) for z in Z1]
        vl1, va1, _, _ = self._ens_eval(Z1, t1, yva_s, yva_l, X=Xva)
        rolled = vl1 > vl0 * 1.02
        if rolled:
            for nn, s in zip(self.nets, snaps): nn.restore(s)
            vl1, va1, _, _ = self._ens_eval(Z0, t0, yva_s, yva_l, X=Xva)
        else:
            for nn, (a, b) in zip(self.nets, t1): nn.Ts, nn.Tl = a, b
        self.update_w_from_losses(self._net_losses(Z1, t1, yva_s, yva_l))
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
        old, old_hp, old_t, old_loss, old_m, old_v = nn.snapshot(), (nn.lr, nn.dropout, nn.wd), nn.t, float(losses[wi]), nn.m, nn.v
        nn.restore(best.snapshot())
        nn.m, nn.v, nn.t = {k: v.copy() for k, v in best.m.items()}, {k: v.copy() for k, v in best.v.items()}, best.t
        nn.lr = clamp(best.lr * float(rg.choice([0.7, 1.0, 1.4])), 5e-4, 8e-3)
        nn.dropout = clamp(best.dropout + float(rg.choice([-0.05, 0.0, 0.05])), 0.0, 0.4)
        nn.wd = clamp(best.wd * float(rg.choice([0.5, 1.0, 2.0])), 1e-3, 0.1)
        for k in nn.P:
            if k[0] == "W":
                nn.P[k] += (rg.standard_normal(nn.P[k].shape) * 0.02 * nn.P[k].std()).astype(np.float32)
                nn.E[k] = nn.P[k].copy()
        fit_model(nn, sp["Xtr"], sp["ytr"], sp["ytr_long"], Xva, yva_s, yva_l, steps=steps, seed=int(rg.integers(1 << 30)), warmup=5, sw=sp.get("wtr"))
        new_loss = float(nn.evaluate(Xva, yva_s, yva_l)[0])
        ok = new_loss < old_loss * 0.995
        self.pbt_n += 1
        hp = {"lr": round(nn.lr, 5), "dropout": round(nn.dropout, 2), "wd": round(nn.wd, 4)}
        if ok:
            self.pbt_acc += 1; losses[wi] = new_loss
        else:
            nn.restore(old); nn.lr, nn.dropout, nn.wd = old_hp; nn.t, nn.m, nn.v = old_t, old_m, old_v
        self.update_w_from_losses(losses)
        return {"accepted": bool(ok), "slot": wi, "old": old_loss, "new": float(new_loss), "hp": hp}

    def consolidate_ewc(self, X, y_s, y_l):
        try:
            self.ewc.consolidate(self.nets, X, y_s, y_l)
            self.ewc_consolidated = True
            return True
        except Exception as e:
            log.warning(f"EWC: {e}"); return False

    def fast_adapt(self, X, y_s, y_l, n_steps=5, lr=0.01):
        if len(X) < 60: return False
        k = int(len(X) * 0.7)
        Xa, ya, yla, Xc, yc, ylc = X[:k], y_s[:k], y_l[:k], X[k:], y_s[k:], y_l[k:]
        states, l0 = [nn.get_state() for nn in self.nets], self.evaluate(Xc, yc, ylc)[0]
        rng = np.random.default_rng(self.evo)
        for nn in self.nets:
            old = nn.lr; nn.lr = lr
            for _ in range(n_steps):
                idx = rng.integers(0, k, min(32, k)); nn.train_step(Xa[idx], ya[idx], yla[idx])
            nn.lr = old
        if self.evaluate(Xc, yc, ylc)[0] < l0 * 0.999: return True
        for nn, s in zip(self.nets, states): nn.set_state(s)
        return False

    def get_state(self):
        return {"nets": [nn.get_state() for nn in self.nets], "perf": self.perf.tolist(), "weights": self.weights.tolist(),
                "bvl": self.best_vl, "plateau": self.plateau, "evo": self.evo, "pbt_acc": self.pbt_acc, "pbt_n": self.pbt_n,
                "router": self.router.get_state() if self.router else None, "router_on": self.router_on, "router_gain": self.router_gain,
                "trans": self.trans_ens.get_state(), "ewc": self.ewc.get_state(), "ewc_consolidated": self.ewc_consolidated,
                "reptile": self.reptile.get_state(), "error_analyzer": self.error_analyzer.get_state(),
                "nn_kwargs": dict(self.nn_kwargs)}   # ✅ nn_kwargs kaydedilir

    def set_state(self, s):
        # 🐛 BUG 3 DÜZELTME: nn_kwargs restore + NN'leri yeniden başlat
        nk = s.get("nn_kwargs")
        if nk and nk != self.nn_kwargs:
            self.nn_kwargs = dict(nk)
            self.nets = [NN(seed=BAG_SEED + i * 7, **self.nn_kwargs) for i in range(len(self.nets))]
            log.info(f"NAS state: yeni nn_kwargs uygulandı {nk}")
        for nn, ns in zip(self.nets, s["nets"]): nn.set_state(ns)
        self.perf, self.weights = np.array(s["perf"], dtype=np.float32), np.array(s["weights"], dtype=np.float32)
        self.best_vl, self.plateau, self.evo = s["bvl"], s["plateau"], s["evo"]
        self.pbt_acc, self.pbt_n = s.get("pbt_acc", 0), s.get("pbt_n", 0)
        if self.router is not None and s.get("router") is not None: self.router.set_state(s["router"])
        self.router_on, self.router_gain = s.get("router_on", False), s.get("router_gain", 0.0)
        if s.get("trans"): self.trans_ens.set_state(s["trans"])
        if s.get("ewc"): self.ewc.set_state(s["ewc"])
        self.ewc_consolidated = s.get("ewc_consolidated", False)
        if s.get("reptile"): self.reptile.set_state(s["reptile"])
        if s.get("error_analyzer"): self.error_analyzer.set_state(s["error_analyzer"])# ════════════════════════════════════════════════════════════
# 🐋 PARÇA 3/4: RİSK + SİM + TUNER + ÖZ-GELİŞİM + FORECAST + BOT
# Düzeltmeler: BUG 2 (pretrain XU100), SORUN 1 (federated_pull optimizer reset)
# ════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════
# 1. SİNYAL
# ════════════════════════════════════════════════════════════
def regime_of(row):
    vals = [safe_float(row.get(k, 0)) for k in ("Regime_Bull_sm", "Regime_Bear_sm", "Regime_Vol_sm", "Regime_Range_sm")]
    return REGIMES[int(np.argmax(vals))]

def regime_adj(rp):
    return np.array([rp.get("regime_bull_adj", 0), rp.get("regime_bear_adj", 0),
                     rp.get("regime_vol_adj", 0), rp.get("regime_range_adj", 0)], dtype=np.float32)

def signal_from_probs(p_s, p_l, rp, regime=None):
    code = REGIMES.index(regime) if regime in REGIMES else None
    pb, pb_l, ps = rp["p_buy"], rp.get("p_buy_long", 0.40), rp["p_sell"]
    if code is not None:
        a = regime_adj(rp)[code]; pb += a; pb_l += a * 0.5; ps -= a * 0.5
    long_bull, long_bear = p_l[2] >= pb_l, p_l[0] >= ps
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

def conf_mult(p_s, p_l, rp): return float(conf_mult_vec(np.asarray(p_s)[None], np.asarray(p_l)[None], rp)[0])

def apply_tilt(p, tilt):
    if abs(tilt) < 1e-9: return p
    q = np.array(p, dtype=np.float64); q[2] += tilt; q[0] -= tilt
    q = np.clip(q, 1e-3, None)
    return (q / q.sum()).astype(np.float32)

# ════════════════════════════════════════════════════════════
# 2. PORTFÖY
# ════════════════════════════════════════════════════════════
def hrp_weights(cov, corr):
    n = len(corr)
    if n < 2: return np.ones(n)
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1)); np.fill_diagonal(dist, 0)
        link = linkage(squareform(dist, checks=False), method="single")
        def cvar(idx):
            if len(idx) == 1: return float(cov[idx[0], idx[0]])
            sub = cov[np.ix_(idx, idx)]; ivp = 1.0 / np.diag(sub); ivp /= ivp.sum()
            return float(ivp @ sub @ ivp)
        def bisect(idx):
            if len(idx) == 1: return {tuple(idx): 1.0}
            mid = len(idx) // 2; left, right = idx[:mid], idx[mid:]
            lv, rv = cvar(left), cvar(right); alpha = 1.0 - lv / (lv + rv + 1e-9)
            w = {k: v * alpha for k, v in bisect(left).items()}
            w.update({k: v * (1 - alpha) for k, v in bisect(right).items()})
            return w
        order = dendrogram(link, no_plot=True)["leaves"]
        out = np.zeros(n)
        for (i,), v in bisect(order).items(): out[i] = v
        return out / (out.sum() + 1e-9)
    except Exception:
        return np.ones(n) / n

def markowitz_weights(mu, cov, risk_aversion=3.0):
    n = len(mu)
    if n < 2: return np.ones(n)
    try:
        w = np.clip(np.linalg.pinv(cov + np.eye(n) * 1e-6) @ mu / max(risk_aversion, 1e-6), 0, None)
        return w / (w.sum() + 1e-9) if w.sum() > 0 else np.ones(n) / n
    except Exception: return np.ones(n) / n

def risk_parity_weights(positions):
    if not RISK_PARITY or not positions: return {}
    w = {}
    for s, p in positions.items():
        vol = p.get("atr", 0) / max(p.get("entry", 1), 1e-9)
        w[s] = 1.0 / (vol if vol > 0 else 0.02)
    tot = sum(w.values())
    return {s: x / tot for s, x in w.items()} if tot > 0 else {}

def portfolio_suggestion(dfs, names):
    if len(names) < 2: return None
    R = pd.concat([dfs[n].set_index("Date")["Ret1"].tail(120).rename(n) for n in names if n in dfs], axis=1).dropna()
    if len(R) < 40 or R.shape[1] < 2: return None
    cov, corr = R.cov().to_numpy(), R.corr().to_numpy()
    return pd.DataFrame({"HRP %": hrp_weights(cov, corr) * 100,
                         "Markowitz %": markowitz_weights(R.mean().to_numpy(), cov) * 100,
                         "Eşit %": 100.0 / R.shape[1]}, index=R.columns).round(1)

# ════════════════════════════════════════════════════════════
# 3. VaR / CVaR
# ════════════════════════════════════════════════════════════
def var_cvar(pnl_history, confidence=VAR_CONFIDENCE):
    if len(pnl_history) < 20: return 0.0, 0.0
    s = np.sort(pnl_history); idx = max(1, int(len(s) * (1 - confidence)))
    return float(s[idx - 1]), float(s[:idx].mean())

def calc_portfolio_var(bot, rp):
    pnls = [t["pnl"] / bot.initial for t in bot.trades if t.get("action") == "SAT" and "pnl" in t]
    if len(pnls) < 20: return {"var95": 0.0, "cvar95": 0.0, "var99": 0.0, "cvar99": 0.0, "n": len(pnls)}
    arr = np.array(pnls, dtype=float)
    v95, c95 = var_cvar(arr, 0.95); v99, c99 = var_cvar(arr, 0.99)
    return {"var95": v95, "cvar95": c95, "var99": v99, "cvar99": c99, "n": len(arr)}

# ════════════════════════════════════════════════════════════
# 4. Slippage / Almgren-Chriss
# ════════════════════════════════════════════════════════════
def almgren_chriss_impact(order_size, daily_volume, atr_pct, sigma_daily=0.02, eta=0.1):
    if daily_volume <= 0: return 0.01
    pv = order_size / daily_volume
    return float(clamp(eta * pv * sigma_daily + eta * np.sqrt(max(pv, 0)) * sigma_daily, 0.0001, 0.05))

def dynamic_slippage(price, volume_tl, atr_pct, order_size_tl=1e6):
    if not DYNAMIC_SLIPPAGE: return SLIPPAGE
    vf = clamp((1e7 / max(volume_tl, 1e5)) ** 0.5, 0.5, 3.0)
    af = clamp(1.0 + (atr_pct - 0.02) * 50, 0.8, 3.0)
    return clamp(SLIPPAGE_BASE * vf * af + almgren_chriss_impact(order_size_tl, volume_tl, atr_pct), 0.0002, 0.02)

def simulate_partial_fill(order_size, daily_volume, atr_pct):
    if daily_volume <= 0: return 0.0
    p = order_size / daily_volume
    return 1.0 if p < 0.01 else 0.95 if p < 0.05 else 0.80 if p < 0.10 else 0.60 if p < 0.20 else 0.40

# ════════════════════════════════════════════════════════════
# 5. Pozisyon + Stop
# ════════════════════════════════════════════════════════════
def size_pos(eq, cash, expo, px, atr, probs_s, probs_l, rp, positions=None, rmult=1.0):
    sd = rp["sl_atr"] * atr
    if sd <= 0 or px <= 0 or not np.isfinite(sd): return 0, sd
    cm = conf_mult(probs_s, probs_l, rp) if rp.get("conf_sizing", True) else 1.0
    rp_mult = 1.0
    if RISK_PARITY and positions:
        w = risk_parity_weights(positions)
        if w: rp_mult = clamp(np.mean(list(w.values())) / 0.2, 0.5, 1.5)
    room = max(0.0, rp["max_exposure"] * eq - expo)
    q = int(min(eq * rp["risk_per_trade"] * cm * rp_mult * rmult / sd,
                eq * rp["max_pos"] / px, room / px, cash / (px * (1 + FEE))))
    return max(q, 0), sd

def stop_level(entry, high, sd0, atr, rp):
    stop, trail = entry - sd0, False
    if rp.get("be_on", True) and high >= entry + rp.get("be_r", 1.0) * sd0:
        stop = max(stop, entry * (1 + RT_COST))
    if high >= entry + rp["trail_act_r"] * sd0:
        ts = high - rp["trail_atr"] * atr
        if ts > stop: stop, trail = ts, True
    return stop, trail

# ════════════════════════════════════════════════════════════
# 6. SİMÜLATÖR
# ════════════════════════════════════════════════════════════
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
         & (Pl[:, 2] >= rp.get("p_buy_long", 0.40) + adj * 0.5)
         & (V["U"] <= rp["unc_max"]) & (V["nval"] >= 1)
         & (V["turn"] >= rp.get("min_daily_turnover", 0)))
    if rp["use_regime"]: m &= V["a200"]
    return m

def simulate_entries(V, idx, rp):
    k = len(idx)
    Hc = int(np.clip(round(rp["max_hold"] * 5 / 7), 1, HW))
    cm, lo, cl = V["cmax"][idx, :Hc], V["lo1"][idx, :Hc], V["cl1"][idx, :Hc]
    nval, atr = np.minimum(V["nval"][idx], Hc), V["atrp"][idx]
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
    st_ = {"n": n, "gross": 0.0, "net": 0.0, "prec": 0.0, "dd": 0.0, "wr": 0.0, "pf": 0.0, "sharpe": 0.0, "sortino": 0.0, "t_stat": 0.0}
    if n == 0: return -5.0, st_
    st_.update(net=float(rets.mean()), gross=float(rets.mean() + RT_COST),
               wr=float((rets > 0).mean()), prec=float((V["y"][idx] == 2).mean()))
    if n < min_n: return -5.0 + n / max(1, min_n), st_
    cum = np.cumsum(daily)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum))
    st_["dd"] = dd
    ds = daily.std() + 1e-9
    neg = daily[daily < 0]
    nstd = neg.std() + 1e-9 if len(neg) > 1 else ds
    st_["sharpe"] = float(daily.mean() / ds * math.sqrt(252))
    st_["sortino"] = float(daily.mean() / nstd * math.sqrt(252))
    wp, lp = rets[rets > 0].sum(), -rets[rets < 0].sum()
    st_["pf"] = float(wp / lp) if lp > 1e-9 else 5.0
    st_["t_stat"] = float(daily.mean() / ds * math.sqrt(len(daily)))
    base = st_["t_stat"] + 0.3 * clamp(st_["sharpe"], -3, 5) + 0.5 * clamp(st_["pf"] - 1.0, -1, 3) + 0.4 * clamp((st_["wr"] - 0.5) * 4, -1, 2)
    base -= 15.0 * dd + 0.001 * max(0, n - 100)
    return float(base), st_

# ════════════════════════════════════════════════════════════
# 7. DSR
# ════════════════════════════════════════════════════════════
def _norm_cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _norm_ppf(p):
    if p <= 0: return -8.0
    if p >= 1: return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
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
    r = np.asarray(returns, dtype=np.float64); r = r[np.isfinite(r)]; n = len(r)
    if n < 30: return 0.0, 0.0, 0.0
    mu, sdv = r.mean(), r.std(ddof=1)
    if sdv < 1e-12: return 0.0, 0.0, 0.0
    sr = mu / sdv; z = (r - mu) / sdv
    g3, g4 = float((z ** 3).mean()), float((z ** 4).mean())
    N, em = max(int(n_trials), 2), 0.5772156649
    sr0 = math.sqrt(1.0 / (n - 1)) * ((1 - em) * _norm_ppf(1 - 1.0 / N) + em * _norm_ppf(1 - 1.0 / (N * math.e)))
    den = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2, 1e-6))
    return float(_norm_cdf((sr - sr0) * math.sqrt(n - 1) / den)), float(sr * math.sqrt(252)), float(sr0 * math.sqrt(252))

# ════════════════════════════════════════════════════════════
# 8. CPCV-OOS + PBO (CSCV)
# ════════════════════════════════════════════════════════════
def cpcv_oos(bot, rp, n_groups=CPCV_N_GROUPS, n_test=CPCV_N_TEST_GROUPS, max_splits=15):
    if not CPCV_ENABLED or bot.hold is None or bot.hold_te is None: return None
    pool = {k: np.concatenate([bot.hold[k], bot.hold_te[k]]) for k in DS_KEYS}
    ud = np.unique(pool["d"])
    if len(ud) < n_groups * 10: return None
    Ps, Pl, U = bot.nn.predict_unc(pool["X"], pool["sid"], pool["row"])
    gs = len(ud) // n_groups
    groups = [ud[i * gs:(len(ud) if i == n_groups - 1 else (i + 1) * gs)] for i in range(n_groups)]
    combos = list(combinations(range(n_groups), n_test))
    rng = np.random.default_rng(42)
    if len(combos) > max_splits:
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]
    scores = []
    for combo in combos:
        m = np.isin(pool["d"], np.concatenate([groups[i] for i in combo]))
        if m.sum() < 50: continue
        V = make_V({k: pool[k][m] for k in DS_KEYS}, Ps[m], Pl[m], U[m])
        sc, _ = score_params(V, rp, min_n=5)
        if np.isfinite(sc): scores.append(sc)
    if not scores: return None
    return {"mean_score": float(np.mean(scores)), "std_score": float(np.std(scores)),
            "worst": float(np.min(scores)), "best": float(np.max(scores)),
            "n_splits": len(scores),
            "sharpe_cpcv": float(np.mean(scores) / (np.std(scores) + 1e-9)),
            "per_split": [round(s, 3) for s in scores]}

def cscv_pbo(M, n_blocks=8, max_combos=70, seed=0):
    if M is None or M.ndim != 2: return None
    T, K = M.shape
    if K < 4 or T < n_blocks * 4: return None
    S = n_blocks - n_blocks % 2; bl = T // S
    blocks = [M[i * bl:(i + 1) * bl] for i in range(S)]
    combos = list(combinations(range(S), S // 2))
    rng = np.random.default_rng(seed)
    if len(combos) > max_combos:
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]
    sr = lambda A: A.mean(0) / (A.std(0) + 1e-9)
    logits = []
    for c in combos:
        rest = [i for i in range(S) if i not in c]
        s_tr, s_te = sr(np.vstack([blocks[i] for i in c])), sr(np.vstack([blocks[i] for i in rest]))
        best = int(np.argmax(s_tr))
        omega = ((s_te < s_te[best]).sum() + 1) / (K + 1)
        logits.append(math.log(omega / (1 - omega)))
    lg = np.array(logits); pbo = float((lg <= 0).mean())
    return {"pbo": pbo, "n_combos": len(lg), "K": K, "logit_mean": float(lg.mean()),
            "n_sims": len(lg), "mean": float(lg.mean()), "median": float(np.median(lg)),
            "interpretation": "GÜVENİLİR" if pbo < 0.3 else ("ŞÜPHELİ" if pbo < 0.5 else "OVERFIT RİSKİ")}

# ════════════════════════════════════════════════════════════
# 9. XAI / ÖNEM / KALİBRASYON
# ════════════════════════════════════════════════════════════
def feat_importance(bot):
    imp = np.zeros(N_FEAT)
    for nn in bot.nn.nets: imp += NN._scale(nn.E["Att"]) * np.linalg.norm(nn.E["W0"], axis=1)
    return imp / (imp.sum() + 1e-9)

def permutation_importance(bot, X, y, n_perm=3, max_n=1500):
    rng = np.random.default_rng(42)
    if len(X) > max_n:
        sel = rng.choice(len(X), max_n, replace=False); X, y = X[sel], y[sel]
    base, _, _ = bot.nn.evaluate(X, y, None)
    imp = np.zeros(X.shape[1], dtype=np.float32)
    for j in range(X.shape[1]):
        ls = []
        for _ in range(n_perm):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            ls.append(bot.nn.evaluate(Xp, y, None)[0])
        imp[j] = max(0.0, np.mean(ls) - base)
    return imp / (imp.sum() + 1e-9)

def shap_values(bot, x_sample, n_samples=30):
    if bot.hold is None or len(bot.hold["X"]) < 10: return None
    ref, rng = bot.hold["X"], np.random.default_rng(42)
    x = np.atleast_2d(x_sample).astype(np.float32)[0]
    n = len(x); sv = np.zeros(n, dtype=np.float32)
    for _ in range(n_samples):
        z, perm = ref[rng.integers(0, len(ref))], rng.permutation(n)
        H = np.tile(z, (n + 1, 1))
        for i, j in enumerate(perm): H[i + 1:, j] = x[j]
        p = bot.nn.predict_batch(H)[0][:, 2]
        sv[perm] += (p[1:] - p[:-1]) / n_samples
    out = [(FEAT_NAMES[i], float(sv[i])) for i in range(n)]
    out.sort(key=lambda t: abs(t[1]), reverse=True)
    return out[:20]

def lime_explain(bot, x_sample, n_samples=300):
    if bot.hold is None: return None
    ref, x = bot.hold["X"], np.atleast_2d(x_sample).astype(np.float32)
    n_feat, rng = x.shape[1], np.random.default_rng(42)
    mask = rng.random((n_samples, n_feat)) < 0.5
    Z = np.where(mask, ref[rng.integers(0, len(ref), n_samples)], x)
    y = bot.nn.predict_batch(Z)[0][:, 2]
    dist = np.linalg.norm((Z - x) / (ref.std(0) + 1e-6), axis=1)
    w = np.exp(-(dist / (dist.std() + 1e-9)) ** 2 / 2)
    A = np.hstack([Z - Z.mean(0), np.ones((n_samples, 1))]) * np.sqrt(w)[:, None]
    try: coef = np.linalg.solve(A.T @ A + 1e-2 * np.eye(n_feat + 1), A.T @ ((y - y.mean()) * np.sqrt(w)))
    except Exception: return None
    out = [(FEAT_NAMES[i], float(coef[i])) for i in range(n_feat)]
    out.sort(key=lambda t: abs(t[1]), reverse=True)
    return out[:15]

def partial_dependence(bot, X, feat_idx, n_grid=20):
    if bot.hold is None: return None
    ref = bot.hold["X"][:1500]
    vals = np.linspace(np.percentile(X[:, feat_idx], 5), np.percentile(X[:, feat_idx], 95), n_grid)
    out = []
    for v in vals:
        T = ref.copy(); T[:, feat_idx] = v
        out.append(float(bot.nn.predict_batch(T)[0][:, 2].mean()))
    return list(zip(vals.tolist(), out))

def calibration(bot):
    if not bot.hold: return None
    Ps, Pl, U = bot.nn.predict_unc(bot.hold["X"], bot.hold["sid"], bot.hold["row"])
    p, y = Ps[:, 2], (bot.hold["y"] == 2).astype(float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 9)))
    if len(edges) < 3: return None
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows, ece = [], 0.0
    for i in range(len(edges) - 1):
        mk = ids == i
        if mk.any():
            rows.append({"Bin": i, "Tahmin %": round(p[mk].mean() * 100, 1),
                         "Gerçek %": round(y[mk].mean() * 100, 1), "n": int(mk.sum())})
            ece += mk.mean() * abs(p[mk].mean() - y[mk].mean())
    return rows, float(ece)

# ════════════════════════════════════════════════════════════
# 10. STRESS TEST
# ════════════════════════════════════════════════════════════
KRIZ_SENARYOLARI = {
    "2008_Kriz": {"shock": -0.55, "vol_mult": 3.0, "name": "2008 Finans Krizi"},
    "2020_Covid": {"shock": -0.35, "vol_mult": 2.5, "name": "2020 Covid Çöküşü"},
    "2023_Deprem": {"shock": -0.15, "vol_mult": 2.0, "name": "2023 Deprem Şoku"},
    "2018_Kur": {"shock": -0.25, "vol_mult": 2.2, "name": "2018 Kur Krizi"},
    "2022_Enflasyon": {"shock": -0.20, "vol_mult": 1.8, "name": "2022 Enflasyon"},
    "Faiz_+500bp": {"shock": -0.15, "vol_mult": 1.5, "name": "Faiz +500bp"},
}

def stress_test(bot, prices, rp, dfs=None):
    total_eq, expo = bot.total_value(prices), bot.exposure(prices)
    if total_eq <= 0: return []
    res = []
    for cfg in KRIZ_SENARYOLARI.values():
        loss = expo * abs(cfg["shock"]) + expo * min(cfg["vol_mult"] * 0.02, 0.15)
        res.append({"Senaryo": cfg["name"], "Şok %": cfg["shock"] * 100,
                    "Portföy Kayıp ₺": loss, "Kayıp %": loss / total_eq * 100,
                    "Yeni Portföy": total_eq - loss, "DD %": loss / total_eq * 100})
    if dfs:
        loss = 0.0
        for s, p in bot.positions.items():
            d = dfs.get(s)
            if d is None: continue
            w = (d["Close"].pct_change(10)).min()
            loss += p["qty"] * prices.get(s, p["entry"]) * abs(min(0.0, safe_float(w)))
        res.append({"Senaryo": "Tarihsel en kötü 10 gün (pozisyon bazlı)", "Şok %": float("nan"),
                    "Portföy Kayıp ₺": loss, "Kayıp %": loss / total_eq * 100,
                    "Yeni Portföy": total_eq - loss, "DD %": loss / total_eq * 100})
    return res

# ════════════════════════════════════════════════════════════
# 11. TUNE (25 otonom) + CSCV-PBO
# ════════════════════════════════════════════════════════════
def _sub(hold, mask): return {k: v[mask] for k, v in hold.items()}

def _mat(rp, c):
    out = dict(rp)
    for k, v in c.items():
        out[k] = bool(v > 0.5) if k in TUNE_BINARY else (int(round(v)) if k in TUNE_INT else float(v))
    return out

def tune_risk(hold, bag, rp, budget=5.0, seed=0, min_n=20, hist=None, hold_te=None, max_evals=1200):
    Ps, Pl, U = bag.predict_unc(hold["X"], hold["sid"], hold["row"])
    d = hold["d"]; ud = np.unique(d)
    if len(ud) < 30: return None
    cut = ud[int(len(ud) * 0.6)]
    ma = d < cut; mb = ~ma
    if ma.sum() < 200 or mb.sum() < 100: return None
    VA = make_V(truncate_windows(_sub(hold, ma), ud, cut), Ps[ma], Pl[ma], U[ma])
    VB = make_V(_sub(hold, mb), Ps[mb], Pl[mb], U[mb])
    nA, nB = min_n, max(8, int(min_n * 0.6))
    rng = np.random.default_rng(seed)
    full = lambda c: _mat(rp, c)
    cur = {}
    for k in TUNE_BOUNDS:
        if k in rp:
            val = float(rp[k])
            if k in TUNE_BINARY: val = 1.0 if val else 0.0
            if k in TUNE_INT: val = float(round(val))
            cur[k] = val
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
    t0, evals, tries, succ, hs = time.time(), 0, 0, 0, []
    while evals < max_evals and time.time() - t0 < budget:
        if evals > 5 and rng.random() < 0.4 and hs:
            top = sorted(hs, key=lambda x: x[1], reverse=True)[:5]
            w_arr = np.array([max(0.01, h[1] + 6.0) for h in top]); w_arr /= w_arr.sum()
            cand = dict(best)
            for k in keys:
                if rng.random() < 0.5:
                    vals = np.array([h[0].get(k, best[k]) for h in top])
                    sd = max(np.std(vals), 0.08 * (TUNE_BOUNDS[k][1] - TUNE_BOUNDS[k][0]))
                    cand[k] = float(clamp((vals * w_arr).sum() + rng.normal() * sd, *TUNE_BOUNDS[k]))
        else:
            cand, nm = dict(best), 0
            for k in keys:
                if rng.random() < 0.4:
                    cand[k] = float(clamp(best[k] + rng.normal() * sig[k], *TUNE_BOUNDS[k])); nm += 1
            if nm == 0:
                k = keys[int(rng.integers(len(keys)))]
                cand[k] = float(clamp(best[k] + rng.normal() * sig[k], *TUNE_BOUNDS[k]))
        sc, _ = score_params(VA, full(cand), nA)
        evals += 1; tries += 1
        hs.append((dict(cand), sc))
        if len(hs) > 60: hs.pop(0)
        if sc > bestA + 1e-9: best, bestA, succ = cand, sc, succ + 1
        if tries == 25:
            f = 1.3 if succ / 25 > 0.2 else 0.8
            for k, (lo, hi) in TUNE_BOUNDS.items():
                sig[k] = clamp(sig[k] * f, 0.01 * (hi - lo), 0.3 * (hi - lo))
            tries = succ = 0
    bestB, stB = score_params(VB, full(best), nB)
    dsr = sr = sr0 = 0.0
    if USE_DSR:
        _, _, daily = run_sim(VB, full(best))
        if daily is not None: dsr, sr, sr0 = deflated_sharpe(daily, evals + 1)
    pbo = None
    if PBO_ENABLED:
        try:
            Vf = make_V(hold, Ps, Pl, U)
            pool, seen = [cur, best] + [c for c, _ in sorted(hs, key=lambda x: x[1], reverse=True)[:14]], set()
            cols = []
            for c in pool:
                key = tuple(round(c[k], 3) for k in keys)
                if key in seen: continue
                seen.add(key)
                _, _, dly = run_sim(Vf, full(c))
                col = dly if dly is not None else np.zeros(Vf["nd"])
                if col.std() > 1e-12: cols.append(col)
            if len(cols) >= 4: pbo = cscv_pbo(np.column_stack(cols))
        except Exception as e:
            log.warning(f"cscv: {e}")
    curT = bestT = stT = None
    if hold_te is not None and len(hold_te["d"]) >= 100:
        Pt, Plt, Ut = bag.predict_unc(hold_te["X"], hold_te["sid"], hold_te["row"])
        VT = make_V(hold_te, Pt, Plt, Ut)
        curT, _ = score_params(VT, full(cur), 8)
        bestT, stT = score_params(VT, full(best), 8)
    changed = any(abs(best[k] - cur[k]) > 1e-9 for k in keys)
    improved = bool(changed and stB["n"] >= nB and bestB > curB + 0.25 and bestA > curA + 0.25 and stB["net"] > 0)
    clean = lambda dct: {k: (bool(v > 0.5) if k in TUNE_BINARY else int(round(v)) if k in TUNE_INT else float(v)) for k, v in dct.items()}
    return {"cur": clean(cur), "best": clean(best), "curA": curA, "bestA": bestA, "curB": curB, "bestB": bestB,
            "evals": evals, "improved": improved, "stB": stB, "dsr": dsr, "sr": sr, "sr0": sr0,
            "curT": curT, "bestT": bestT, "stT": stT, "pbo": pbo}

def run_pbo(bot, rp, budget=4.0):
    if bot.hold is None: return None
    tr = tune_risk(bot.hold, bot.nn, rp, budget=budget, seed=bot.nn.evo + 1, hold_te=None)
    return tr["pbo"] if tr else None

def _near(a, b, tol=0.35):
    if not a or not b: return False
    common = set(a) & set(b)
    if not common: return False
    for k in common:
        if k not in TUNE_BOUNDS: continue
        lo, hi = TUNE_BOUNDS[k]
        if abs(float(a[k]) - float(b[k])) > tol * max(hi - lo, 1e-9): return False
    return True

# ════════════════════════════════════════════════════════════
# 12. TRANSFORMER EĞİTİM YARDIMCISI (sektör-aware)
# ════════════════════════════════════════════════════════════
def train_transformer(nn, sp, steps=30, seed=0):
    te, tr, va = nn.trans_ens, sp["tr"], sp["va"]
    sk = AKTIF_SEKTOR
    te.fit(tr["sid"], tr["row"], tr["y"], steps=steps, bs=16, seed=seed, sektor_key=sk)
    vi = te.sample_val(va["sid"], va["row"], va["y"])
    sid, row, y = va["sid"][vi], va["row"][vi], va["y"][vi]
    te.update_w(sid, row, y, sektor_key=sk)
    Pt = te.predict_idx(sid, row, sektor_key=sk)
    Pm = nn.predict_batch(va["X"][vi])[0]
    ml, bl = short_logloss(Pm, y), short_logloss(MLP_WEIGHT * Pm + TRANS_WEIGHT * Pt, y)
    te.gain, te.on = ml - bl, bool(bl < ml * 0.998)

# ════════════════════════════════════════════════════════════
# 13. SELF-IMPROVER
# ════════════════════════════════════════════════════════════
class SelfImprover:
    def __init__(self):
        self.history, self.rp_hist = [], []
        self.last_run, self.last_data, self.runs = 0.0, None, 0
        self.proposal, self.drift, self.consec_improve, self.pending_rp = None, 1.0, 0, None

    def due(self, now_ts, data_date, interval_min):
        return (self.last_data != data_date) or (now_ts - self.last_run >= interval_min * 60)

    @staticmethod
    def calibrate(bag, X, y_s):
        if len(y_s) < 30: return None
        ps, _ = bag.predict_batch(X)
        p_al, y_al = ps[:, 2], (y_s == 2).astype(np.float32)
        bins, n, ece, cal = np.linspace(0, 1, 11), len(ps), 0.0, []
        for i in range(10):
            m = (p_al >= bins[i]) & ((p_al < bins[i + 1]) if i < 9 else (p_al <= bins[i + 1]))
            if m.sum() > 0:
                ece += m.sum() / n * abs(p_al[m].mean() - y_al[m].mean())
                cal.append({"bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}", "pred": float(p_al[m].mean()),
                            "actual": float(y_al[m].mean()), "n": int(m.sum())})
        return {"brier": float(np.mean((p_al - y_al) ** 2)), "ece": float(ece),
                "acc": float(np.mean(ps.argmax(1) == y_s)),
                "base": float(np.bincount(y_s, minlength=3).max() / len(y_s)), "cal": cal, "n": n}

    def run(self, sh, datasets, rp, budget=30.0, reason="auto"):
        t0 = time.time()
        merged = merge_ds(datasets)
        sp = split_ds(merged) if merged else None
        rep = {"ts": _ts(), "reason": reason}
        if sp is None:
            rep["error"] = "veri yetersiz"; return None, rep
        nn = sh.nn
        rep["drift"] = self.drift = nn.drift(sp["Xva"], sp["yva"], sp["yva_long"])
        vl0, va0, _ = nn.evaluate(sp["Xva"], sp["yva"], sp["yva_long"])
        steps = rolled = 0
        n_steps = 12 if rep["drift"] > 1.15 else 6
        while time.time() - t0 < budget * 0.35 and steps < n_steps:
            r = nn.evolve_step(sp); steps += 1; rolled += int(r["rolled"])
        pb = nn.pbt_step(sp) if time.time() - t0 < budget * 0.5 else None
        rr = nn.fit_router(sp)
        try: train_transformer(nn, sp, steps=30, seed=nn.evo)
        except Exception as e: log.warning(f"trans: {e}")
        adapted = False
        if rep["drift"] > 1.15:
            try:
                adapted = nn.fast_adapt(sp["Xva"][-300:], sp["yva"][-300:], sp["yva_long"][-300:])
                if time.time() - t0 < budget * 0.55:
                    nn.reptile.meta_train(nn, sp, n_epochs=1, seed=nn.evo)
            except Exception as e: log.warning(f"adapt: {e}")
        rep["adapt"] = adapted
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
        if (not nn.ewc_consolidated or nn.evo % 20 == 0) and len(sp["Xva"]) > 200:
            nn.consolidate_ewc(sp["Xva"], sp["yva"], sp["yva_long"])
        err = nn.error_analyzer.analyze()
        if err: rep["error_analysis"] = err
        sh.set_hold(sp); sh.set_ref(sp); sh.reeval(rp)
        try:
            sh._rl_train(sp, epochs=2)
            sh.rl_on = sh._rl_validate(sp)
        except Exception as e: log.warning(f"rl: {e}")
        new_rp, adopted = None, False
        tr = tune_risk(sh.hold, nn, rp, budget=min(max(2.0, budget - (time.time() - t0)), 15.0),
                       seed=nn.evo, min_n=20, hist=self.rp_hist, hold_te=sh.hold_te)
        if tr:
            rep.update(scoreB0=tr["curB"], scoreB1=tr["bestB"], evals=tr["evals"], dsr=tr["dsr"], sr=tr["sr"])
            if tr["bestT"] is not None: rep.update(scoreT0=tr["curT"], scoreT1=tr["bestT"])
            dsr_ok = (not USE_DSR) or tr["dsr"] >= rp.get("dsr_confidence", 0.9)
            te_ok = True
            if tr["stT"] is not None:
                te_ok = tr["stT"]["n"] >= 8 and tr["stT"]["net"] > 0 and tr["bestT"] >= tr["curT"] - rp.get("test_score_tol", 0.25)
            cand_rp = _mat(rp, tr["best"])
            cp = cpcv_oos(sh, cand_rp) if CPCV_ENABLED else None
            if cp:
                rep["cpcv"] = cp
                db_cpcv_add(cp["mean_score"], cp["std_score"], cp["worst"], cp["best"], cp["n_splits"], cp["sharpe_cpcv"])
            pbo = tr.get("pbo")
            if pbo:
                rep["pbo"] = pbo
                db_pbo_add(pbo["pbo"], pbo["mean"], pbo["median"], pbo["n_sims"], pbo["interpretation"])
            cpcv_ok = cp is None or cp.get("sharpe_cpcv", 0) > 0.3
            pbo_ok = pbo is None or pbo["pbo"] < 0.5
            min_trades_ok = tr["stB"].get("n", 0) >= 30
            risk_ok = tr["stB"].get("dd", 1.0) < 0.30
            if tr["improved"] and dsr_ok and te_ok and cpcv_ok and pbo_ok and min_trades_ok and risk_ok:
                rep["changes"] = {k: round(float(tr["best"][k]), 4) for k in TUNE_BOUNDS
                                  if k in tr["best"] and k in tr["cur"] and abs(float(tr["best"][k]) - float(tr["cur"][k])) > 1e-9}
                need = int(rp.get("consec_improve", 2))
                if self.pending_rp is not None and _near(self.pending_rp, tr["best"]): self.consec_improve += 1
                else: self.consec_improve = 1
                self.pending_rp = dict(tr["best"])
                if self.consec_improve >= need and rp.get("auto_adopt", True):
                    new_rp = {**rp, **{k: tr["best"][k] for k in TUNE_BOUNDS if k in tr["best"]}}
                    self.rp_hist = (self.rp_hist + [dict(tr["cur"])])[-5:]
                    adopted, self.proposal, self.consec_improve, self.pending_rp = True, None, 0, None
                else:
                    self.proposal = {"params": tr["best"], "gain": tr["bestB"] - tr["curB"], "ts": rep["ts"],
                                     "wait": max(0, need - self.consec_improve)}
            else:
                rep.update(dsr_reject=not dsr_ok, te_reject=not te_ok, cpcv_reject=not cpcv_ok,
                           pbo_reject=not pbo_ok, min_trades_reject=not min_trades_ok, risk_reject=not risk_ok)
                self.consec_improve, self.pending_rp = 0, None
        rep["adopted"], rep["sec"] = adopted, round(time.time() - t0, 1)
        self.history = (self.history + [rep])[-40:]
        self.runs += 1
        db_improve_add(reason, f"drift={rep['drift']:.2f} steps={steps} pbt={rep['pbt']} router={rep['router']}",
                       tr["curB"] if tr else vl0, tr["bestB"] if tr else vl1, adopted)
        return new_rp, rep

    def get_state(self):
        return {k: getattr(self, k) for k in ("history", "rp_hist", "last_run", "last_data", "runs",
                                              "proposal", "drift", "consec_improve", "pending_rp")}

    def set_state(self, s):
        for k in ("history", "rp_hist", "last_run", "last_data", "runs", "proposal", "drift",
                  "consec_improve", "pending_rp"):
            if k in s: setattr(self, k, s[k])

class BackgroundImprover:
    def __init__(self):
        self.thread, self.result, self.owner, self.started = None, None, None, 0.0

    def busy(self): return self.thread is not None and self.thread.is_alive()

    def start(self, bot, datasets, rp, data_date, reason, budget):
        if self.busy(): return False
        shadow = bot.clone_for_improve()
        self.owner, self.started = id(bot), time.time()
        def job():
            try:
                new_rp, rep = bot.si.run(shadow, datasets, rp, budget=budget, reason=reason)
                self.result = {"new_rp": new_rp, "rep": rep, "shadow": shadow, "reason": reason,
                               "data_date": data_date, "err": None}
            except Exception as e:
                log.exception("improve"); self.result = {"err": repr(e), "reason": reason, "data_date": data_date}
        self.thread = threading.Thread(target=job, daemon=True)
        self.thread.start()
        return True

    def pop(self, owner):
        if self.result is not None and self.owner == owner:
            r, self.result = self.result, None
            return r
        return None

@st.cache_resource(show_spinner=False)
def get_bgi(): return BackgroundImprover()

def handle_improve_result(bot, res, rp):
    if res is None: return rp
    bot.si.last_run, bot.si.last_data = time.time(), res.get("data_date")
    if res.get("err"):
        st.session_state.log.append(f"{now_tr():%H:%M:%S} ❌ öz-gelişim hata: {res['err'][:120]}"); return rp
    rep = res["rep"]
    if rep.get("error"):
        st.session_state.log.append(f"{now_tr():%H:%M:%S} {rep['error']}"); return rp
    bot.adopt(res["shadow"])
    out = rp
    if res.get("new_rp"):
        out = validate_risk(res["new_rp"]); st.session_state.rp = out
        for k in TUNE_BOUNDS: st.session_state.pop(f"rp_{k}", None)
        bot.reeval(out)
    extra = f" te_acc {rep.get('te_acc', 0):.3f}(taban {rep.get('te_base', 0):.3f})"
    if "dsr" in rep: extra += f" DSR {rep['dsr']:.3f}"
    if "pbo" in rep: extra += f" PBO {rep['pbo']['pbo']:.2f}"
    if "cpcv" in rep: extra += f" CPCV {rep['cpcv']['sharpe_cpcv']:.2f}"
    st.session_state.log = (st.session_state.log + [
        f"{now_tr():%H:%M:%S} 🐋 {res['reason']}: vl {rep['vl0']:.4f}→{rep['vl1']:.4f} drift {rep['drift']:.2f} "
        f"pbt {rep['pbt']} router {rep['router']}{extra} " +
        (f"✅ UYGULANDI {rep.get('changes')}" if rep.get("adopted") else "")])[-100:]
    save_state(bot, out)
    return out

# ════════════════════════════════════════════════════════════
# 14. FORECAST
# ════════════════════════════════════════════════════════════
@dataclass
class Forecast:
    stock: str; current_price: float; horizon_days: int; target_price: float; expected_return_pct: float
    lower_bound: float; upper_bound: float; p_up: float; p_down: float; p_flat: float
    bull_price: float; base_price: float; bear_price: float; confidence: float; uncertainty: float
    regime: str; regime_adj: float; ts: str

def forecast_stock(bot, stock, df, price, dec, rp, horizon_days=None):
    h = int(horizon_days or HORIZON)
    r1 = df["Ret1"].dropna().tail(60).to_numpy()
    atr_pct = safe_float(df["ATR"].iloc[-1], price * 0.02) / max(price, 1e-9)
    sig_d = max(float(r1.std()) if len(r1) >= 20 else atr_pct * 0.8, 0.004)
    ps, pl, unc, regime = dec["probs"], dec["probs_long"], dec["unc"], dec["regime"]
    wl = clamp(h / 40.0, 0.2, 0.8)
    bias = (1 - wl) * float(ps[2] - ps[0]) + wl * float(pl[2] - pl[0])
    sigma_h = sig_d * math.sqrt(h) * 1.1
    exp_ret = bias * 0.5 * sigma_h
    mp, mn = {"BULL": (1.15, 0.85), "BEAR": (0.85, 1.15), "VOL": (0.8, 0.8), "RANGE": (1.0, 1.0)}.get(regime, (1.0, 1.0))
    reg_adj = mp if exp_ret > 0 else mn
    exp_ret *= reg_adj
    z = 1.2816
    px = lambda r: max(price * (1 + r), price * 0.01)
    confidence = clamp(1.0 - unc * 2.0 + (float(ps[2]) - max(float(ps[0]), float(ps[1]))) * 0.5, 0.0, 1.0)
    return Forecast(stock=stock, current_price=price, horizon_days=h, target_price=px(exp_ret),
                    expected_return_pct=exp_ret * 100,
                    lower_bound=px(exp_ret - z * sigma_h), upper_bound=px(exp_ret + z * sigma_h),
                    p_up=float(ps[2]), p_down=float(ps[0]), p_flat=float(ps[1]),
                    bull_price=px(exp_ret + z * sigma_h), base_price=px(exp_ret), bear_price=px(exp_ret - z * sigma_h),
                    confidence=confidence, uncertainty=unc, regime=regime, regime_adj=reg_adj, ts=_ts())

def forecast_all(bot, names, dfs, prices, dec, rp, horizon_days=None):
    rows = []
    for nm in names:
        try:
            f = forecast_stock(bot, nm, dfs[nm], prices[nm], dec[nm], rp, horizon_days)
            rows.append({"Hisse": nm, "Şu an": round(f.current_price, 2), "Hedef": round(f.target_price, 2),
                         "Değişim %": round(f.expected_return_pct, 2), "Alt (%80)": round(f.lower_bound, 2),
                         "Üst (%80)": round(f.upper_bound, 2), "P(Yukarı) %": round(f.p_up * 100, 0),
                         "P(Aşağı) %": round(f.p_down * 100, 0), "Senaryo Boğa": round(f.bull_price, 2),
                         "Senaryo Baz": round(f.base_price, 2), "Senaryo Ayı": round(f.bear_price, 2),
                         "Güven %": round(f.confidence * 100, 0), "Belirsizlik": round(f.uncertainty, 3),
                         "Rejim": f.regime, "Karar": dec[nm]["action"], "_obj": f})
        except Exception as e: log.warning(f"forecast {nm}: {e}")
    return rows

def forecast_to_text(f):
    yon = "YUKARI" if f.expected_return_pct > 1 else ("AŞAĞI" if f.expected_return_pct < -1 else "YATAY")
    emoji = "🟢" if f.expected_return_pct > 1 else ("🔴" if f.expected_return_pct < -1 else "🟡")
    return f"""{emoji} **{f.stock}** — {f.horizon_days} günlük tahmin
- **Şu an:** ₺{f.current_price:.2f}
- **Hedef:** ₺{f.target_price:.2f} ({f.expected_return_pct:+.2f}%) — **{yon}**
- **%80 aralık:** ₺{f.lower_bound:.2f} — ₺{f.upper_bound:.2f}
- **Olasılıklar:** P(Yukarı) %{f.p_up*100:.0f} · P(Yatay) %{f.p_flat*100:.0f} · P(Aşağı) %{f.p_down*100:.0f}
- **Senaryolar:** 🐂 ₺{f.bull_price:.2f} · 🎯 ₺{f.base_price:.2f} · 🐻 ₺{f.bear_price:.2f}
- **Güven:** %{f.confidence*100:.0f} · **Belirsizlik:** {f.uncertainty:.3f}
- **Rejim:** {f.regime} (çarpan ×{f.regime_adj:.2f})"""

def forecast_scorecard(dfs, limit=600):
    rows = _q("SELECT ts,stock,current_price,target_price,horizon FROM forecast ORDER BY id DESC LIMIT ?", (int(limit),))
    out = []
    for ts, stock, cp, tp, h in rows:
        d = dfs.get(stock)
        if d is None or cp <= 0: continue
        try:
            t0 = datetime.fromisoformat(ts).replace(tzinfo=None)
            i = int(np.searchsorted(d["Date"].to_numpy(), np.datetime64(t0.date()), side="right")) - 1
            if i < 0 or i + int(h) >= len(d): continue
            act = float(d["Close"].iloc[i + int(h)]) / cp - 1
            pred = tp / cp - 1
            out.append({"Hisse": stock, "Zaman": ts[:16], "Tahmin %": round(pred * 100, 2),
                        "Gerçek %": round(act * 100, 2), "Hata %": round((pred - act) * 100, 2),
                        "Yön ✓": int(np.sign(pred) == np.sign(act))})
        except Exception: continue
    return pd.DataFrame(out)

def check_forecast_alerts(rows, threshold=5.0):
    return [r["_obj"] for r in rows if abs(r["_obj"].expected_return_pct) >= threshold and r["_obj"].confidence > 0.5]

# ════════════════════════════════════════════════════════════
# 15. MULTI-AGENT
# ════════════════════════════════════════════════════════════
class Agent:
    def __init__(self, agent_id, seed, profile="balanced"):
        self.id, self.seed, self.profile = agent_id, seed, profile
        self.score, self.stats, self.weight, self.rp = 0.0, {}, 0.0, {}

    def to_dict(self):
        s = self.stats
        return {"id": self.id, "profile": self.profile, "score": self.score, "n": s.get("n", 0),
                "net": s.get("net", 0.0), "wr": s.get("wr", 0.0), "pf": s.get("pf", 0.0),
                "sharpe": s.get("sharpe", 0.0), "dd": s.get("dd", 0.0), "weight": self.weight}

class MultiAgent:
    PROF = {"aggressive": {"p": -0.03, "risk": 1.4, "sl": 0.85, "m": -0.03},
            "balanced": {"p": 0.0, "risk": 1.0, "sl": 1.0, "m": 0.0},
            "conservative": {"p": 0.04, "risk": 0.7, "sl": 1.2, "m": 0.04}}

    def __init__(self, n_agents=12):
        names = list(self.PROF)
        self.n = n_agents
        self.agents = [Agent(i, BAG_SEED + i * 137, names[i % 3]) for i in range(n_agents)]
        self.rounds = 0

    def _rp(self, a, rp):
        rg, o = np.random.default_rng(a.seed + self.rounds), self.PROF[a.profile]
        out = dict(rp)
        out["p_buy"] = clamp(rp["p_buy"] + o["p"] + rg.normal() * 0.02, 0.36, 0.75)
        out["margin"] = clamp(rp["margin"] + o["m"] + rg.normal() * 0.01, 0.0, 0.35)
        out["risk_per_trade"] = clamp(rp["risk_per_trade"] * o["risk"] * (1 + rg.normal() * 0.1), 0.003, 0.025)
        out["sl_atr"] = clamp(rp["sl_atr"] * o["sl"] * (1 + rg.normal() * 0.08), 1.0, 5.0)
        return out

    def run(self, V, rp, min_n=10):
        self.rounds += 1
        for a in self.agents:
            a.rp = self._rp(a, rp)
            a.score, a.stats = score_params(V, a.rp, min_n)
        sc = np.array([a.score for a in self.agents])
        w = np.exp((sc - sc.max()) / 2.0); w /= w.sum()
        for a, wi in zip(self.agents, w): a.weight = float(wi)

    def best(self): return max(self.agents, key=lambda a: a.score)

    def summary(self):
        return {"n_agents": self.n, "rounds": self.rounds,
                "top_agent": self.best().id if self.rounds else -1,
                "avg_score": float(np.mean([a.score for a in self.agents])),
                "agents": [a.to_dict() for a in self.agents]}

# ════════════════════════════════════════════════════════════
# 16. NAS
# ════════════════════════════════════════════════════════════
NAS_CANDIDATES = [
    {"h": 32, "dropout": 0.10, "lr": 0.004, "wd": 0.01},
    {"h": 64, "dropout": 0.15, "lr": 0.003, "wd": 0.01},
    {"h": 96, "dropout": 0.20, "lr": 0.003, "wd": 0.02},
    {"h": 128, "dropout": 0.25, "lr": 0.0025, "wd": 0.02},
    {"h": 64, "dropout": 0.05, "lr": 0.003, "wd": 0.005},
    {"h": 48, "dropout": 0.30, "lr": 0.004, "wd": 0.03},
]

def nas_search(sp, seed=42, steps=100):
    results, best_loss, best_cfg = [], float("inf"), NAS_CANDIDATES[1]
    for cfg in NAS_CANDIDATES:
        try:
            nn = NN(seed=seed, **cfg)
            fit_model(nn, sp["Xtr"], sp["ytr"], sp["ytr_long"], sp["Xva"], sp["yva"], sp["yva_long"],
                      steps=steps, seed=seed)
            vl, _ = nn.evaluate(sp["Xva"], sp["yva"], sp["yva_long"])
            tl, _ = nn.evaluate(sp["Xte"], sp["yte"], sp["yte_long"])
            results.append({"cfg": cfg, "val_loss": float(vl), "test_loss": float(tl)})
            if vl < best_loss: best_loss, best_cfg = vl, cfg
        except Exception as e: log.warning(f"nas: {e}")
    return best_cfg, results

# ════════════════════════════════════════════════════════════
# 17. BOT — 🐛 BUG 2 + SORUN 1 DÜZELTME
# ════════════════════════════════════════════════════════════
class Bot:
    def __init__(self, cash=None):
        self.cash = self.initial = cash or BOT_SERMAYE
        self.positions, self.trades = {}, []
        self.nn, self.si = BaggedNN(), SelfImprover()
        self.wins = self.losses = 0
        self.eq_hist = []
        self.peak_eq = self.cash
        self.halted, self.halt_reason, self.halt_until, self.streak = False, "", None, 0
        self.day_key, self.day_eq0, self.day_trades = None, self.cash, 0
        self.pretrained, self.val_stats = False, None
        self.hold = self.hold_te = self.ref = None
        self.last_exit = {}
        self.rl, self.rl_on, self.rl_gain = PPOAgent(seed=BAG_SEED + 777), False, 0.0
        self.federated_score, self.last_push = 0.0, 0.0
        self.lock = threading.RLock()

    def total_value(self, prices):
        return self.cash + sum(p["qty"] * prices.get(s, p["entry"]) for s, p in self.positions.items())

    def exposure(self, prices):
        return sum(p["qty"] * prices.get(s, p["entry"]) for s, p in self.positions.items())

    def record(self, v):
        if not self.eq_hist or abs(self.eq_hist[-1] - v) > 1e-6:
            self.eq_hist = (self.eq_hist + [float(v)])[-1000:]

    def win_rate(self):
        t = self.wins + self.losses
        return self.wins / t if t else 0.0

    def set_hold(self, sp):
        self.hold = {k: sp["va"][k][-HOLD_N:].copy() for k in DS_KEYS}
        self.hold_te = {k: sp["te"][k][-HOLD_N:].copy() for k in DS_KEYS}

    def set_ref(self, sp): self.ref = (sp["Xtr"].mean(0), sp["Xtr"].std(0) + 1e-6)

    def feat_drift(self, F):
        if not self.ref: return 0.0
        mu, sd = self.ref
        k = N_FEAT_TECH
        return float(np.mean(np.minimum(np.abs(F[:, :k].mean(0) - mu[:k]) / sd[:k], 5.0)))

    def rebuild_hold(self, datasets, rp):
        merged = merge_ds(datasets)
        sp = split_ds(merged) if merged else None
        if sp is None: return False
        self.set_hold(sp); self.set_ref(sp); self.reeval(rp)
        return True

    def clone_for_improve(self):
        sh = Bot(cash=self.cash)
        sh.nn = BaggedNN(nn_kwargs=self.nn.nn_kwargs, replay=self.nn.replay)
        with self.lock: st0 = copy.deepcopy(self.nn.get_state())
        sh.nn.set_state(st0)
        sh.hold, sh.hold_te, sh.ref, sh.val_stats = self.hold, self.hold_te, self.ref, self.val_stats
        sh.rl.set_state(self.rl.get_state()); sh.rl_on, sh.rl_gain = self.rl_on, self.rl_gain
        sh.pretrained = True
        return sh

    def adopt(self, sh):
        with self.lock:
            self.nn.set_state(copy.deepcopy(sh.nn.get_state()))
            self.hold, self.hold_te, self.ref, self.val_stats = sh.hold, sh.hold_te, sh.ref, sh.val_stats
            self.rl.set_state(sh.rl.get_state()); self.rl_on, self.rl_gain = sh.rl_on, sh.rl_gain

    # 🐛 BUG 2 DÜZELTME: pretrain XU100'ü otomatik çeker
    def pretrain(self, dss_list, rp, steps=400, data_date=""):
        merged = merge_ds(dss_list)
        sp = split_ds(merged) if merged else None
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
        try: train_transformer(self.nn, sp, steps=80, seed=1)
        except Exception as e: log.warning(f"trans: {e}")
        try:
            self._rl_train(sp, epochs=6); self.rl_on = self._rl_validate(sp)
        except Exception as e: log.warning(f"rl: {e}")
        try:
            if len(sp["Xva"]) > 100:
                self.nn.consolidate_ewc(sp["Xva"], sp["yva"], sp["yva_long"])
        except Exception as e: log.warning(f"ewc: {e}")
        # HMM için XU100 otomatik çek
        try:
            xu = _fetch("XU100.IS", "5y")
            if xu is not None and len(xu) > 150:
                mkt_ret = xu.set_index("Date")["Close"].pct_change().dropna().to_numpy()
                get_hmm().fit(mkt_ret, n_iter=20, date=data_date)
                log.info(f"HMM eğitildi: {get_hmm().n_iter} iter, {len(mkt_ret)} gün")
            else:
                log.warning("XU100 verisi alınamadı, HMM eğitilmedi")
        except Exception as e:
            log.warning(f"HMM: {e}")
        self.reeval(rp)
        self.pretrained = True
        return self.val_stats

    def _rl_train(self, sp, epochs=3):
        X, f = sp["Xtr"], sp["tr"]["f"]
        rng = np.random.default_rng(42 + self.rl.t)
        for _ in range(epochs):
            idx = rng.permutation(len(X))[:min(2000, len(X))]
            proba = self.rl.proba(X[idx])
            actions = np.array([rng.choice(3, p=p / p.sum()) for p in proba])
            old = proba[np.arange(len(idx)), actions]
            rew = np.where(actions == 2, f[idx] - RT_COST, 0.0).astype(np.float32)
            self.rl.update(X[idx], actions, old, rew, epochs=4)

    def _rl_validate(self, sp):
        Xv, fv = sp["Xva"][:3000], sp["va"]["f"][:3000]
        p = self.rl.proba(Xv)[:, 2]
        sel = p > 0.5
        if sel.sum() < 30:
            self.rl_gain = 0.0; return False
        self.rl_gain = float((fv[sel] - RT_COST).mean() - (fv - RT_COST).mean())
        return bool(self.rl_gain > 0.002)

    def reeval(self, rp):
        if not self.hold: return
        Ps, Pl, U = self.nn.predict_unc(self.hold["X"], self.hold["sid"], self.hold["row"])
        sc, st_ = score_params(make_V(self.hold, Ps, Pl, U), rp, min_n=1)
        y = self.hold["y"]
        self.val_stats = {
            "loss": float(-np.mean(np.log(Ps[np.arange(len(y)), y] + 1e-9))),
            "acc": float(np.mean(Ps.argmax(1) == y)),
            "base": float(np.bincount(y, minlength=3).max() / len(y)),
            "al_n": st_["n"], "al_ret": st_["gross"], "al_prec": st_["prec"],
            "net": st_["net"], "dd": st_["dd"], "score": sc, "n": int(len(y)),
            "wr": st_.get("wr", 0.0), "pf": st_.get("pf", 0.0),
            "sharpe": st_.get("sharpe", 0.0), "sortino": st_.get("sortino", 0.0),
            "t_stat": st_.get("t_stat", 0.0)}

    def edge_ok(self, rp):
        vs = self.val_stats
        return bool(vs and vs["al_n"] >= rp["gate_min_n"] and vs["al_ret"] > RT_COST * 1.5)

    def scan(self, names, F, rows, rp, seq=None, mkt=None, overlay=None):
        Ps, Pl, U = self.nn.predict_unc(F, seq=seq)
        gate_ok, out = self.edge_ok(rp), {}
        rl_p = self.rl.proba(F)[:, 2] if self.rl_on else None
        for i, nm in enumerate(names):
            p_s, p_l, u, row = Ps[i], Pl[i], float(U[i]), rows[nm]
            reg = regime_of(row)
            if mkt and mkt[0] in ("BEAR", "VOL") and mkt[1] >= 0.6: reg = mkt[0]
            tilt = float((overlay or {}).get(nm, 0.0)) if rp.get("overlay_on", True) else 0.0
            if tilt: p_s = apply_tilt(p_s, tilt)
            act, note = signal_from_probs(p_s, p_l, rp, reg), ""
            if act == "AL":
                if rp["use_regime"] and not safe_float(row.get("Close")) > safe_float(row.get("SMA200")):
                    act, note = "TUT", "rejim"
                elif rp["gate_on"] and not gate_ok: act, note = "TUT", "kapi"
                elif u > rp["unc_max"]: act, note = "TUT", f"belirsizlik {u:.2f}"
                elif safe_float(row.get("Turnover20")) < rp.get("min_daily_turnover", 0):
                    act, note = "TUT", "likidite"
                elif rp.get("rl_veto", True) and rl_p is not None and rl_p[i] < rp.get("rl_veto_p", 0.25):
                    act, note = "TUT", f"rl-veto {rl_p[i]:.2f}"
            out[nm] = {"action": act, "probs": p_s, "probs_long": p_l, "unc": u, "regime": reg,
                       "note": note, "atr": safe_float(row.get("ATR")), "feat": F[i],
                       "tilt": tilt, "rl_p": float(rl_p[i]) if rl_p is not None else None}
        return out

    def halt(self, reason, hours, now):
        self.halted, self.halt_reason = True, reason
        self.halt_until = (now + timedelta(hours=hours)).isoformat(timespec="seconds")

    def resume(self, prices):
        self.halted, self.halt_reason, self.halt_until, self.streak = False, "", None, 0
        self.peak_eq = self.total_value(prices)

    def _sell(self, s, price, date, reason, qty=None, turn=0.0):
        p = self.positions[s]
        q = p["qty"] if qty is None else int(min(max(qty, 1), p["qty"]))
        full = q >= p["qty"]
        slip = dynamic_slippage(price, safe_float(p.get("turn", turn), 1e7), p.get("atr", 0) / max(price, 1e-9))
        px = tick_round(price * (1 - slip), up=False)
        proceeds = q * px * (1 - FEE)
        cp = p["cost"] * q / p["qty"]
        pnl = proceeds - cp
        self.cash += proceeds
        tot = pnl + p.get("realized", 0.0)
        if full:
            self.positions.pop(s)
            if tot > 0: self.wins += 1; self.streak = 0
            else: self.losses += 1; self.streak += 1
        else:
            p["qty"] -= q; p["cost"] -= cp; p["realized"] = p.get("realized", 0.0) + pnl; p["tp_done"] = True
        self.trades = (self.trades + [{"date": str(date), "action": "SAT", "stock": s, "price": px,
                                        "qty": q, "pnl": pnl, "reason": reason[:200]}])[-500:]
        db_add_trade(s, "SAT", px, q, pnl, reason)
        if full:
            self.last_exit[s] = now_tr().strftime("%Y-%m-%d")
            c0 = p.get("cost0", p["cost"])
            if p.get("feat") is not None and c0 > 0:
                net = tot / c0
                lab = 2 if net > RT_COST else (0 if net < -RT_COST else 1)
                self.nn.replay.add(p["feat"], lab, -1, 1.0 + min(abs(net), 0.2) * 20)
                self.nn.error_analyzer.record_trade(p["feat"], p.get("regime_at_entry", "RANGE"), net)
        return pnl

    def _buy(self, s, price, date, q, sd, atr, feat, reason, turn=0.0, probs=None, regime="RANGE"):
        slip = dynamic_slippage(price, safe_float(turn, 1e7), atr / max(price, 1e-9))
        px = tick_round(price * (1 + slip), up=True)
        cost = q * px * (1 + FEE)
        self.cash -= cost
        self.positions[s] = {"qty": q, "entry": px, "cost": cost, "cost0": cost, "stop0": px - sd, "sd0": sd,
                             "atr": atr, "high": px, "feat": feat, "ts": _ts(), "tp_done": False,
                             "realized": 0.0, "turn": turn, "probs_at_entry": probs, "regime_at_entry": regime}
        self.day_trades += 1
        self.trades = (self.trades + [{"date": str(date), "action": "AL", "stock": s, "price": px,
                                        "qty": q, "reason": reason[:200]}])[-500:]
        db_add_trade(s, "AL", px, q, 0.0, reason)

    def _corr_ok(self, s, rets, thr):
        a = rets.get(s)
        if a is None or len(a) < 30: return True
        for p in self.positions:
            b = rets.get(p)
            if b is None: continue
            n = min(len(a), len(b))
            c = np.corrcoef(a[-n:], b[-n:])[0, 1]
            if np.isfinite(c) and c > thr: return False
        return True

    def run_cycle(self, dec, prices, rp, date, now=None, rows=None, rets=None, stale=False):
        now, rows, rets, msgs = now or now_tr(), rows or {}, rets or {}, []
        with self.lock:
            eq = self.total_value(prices)
            dk = now.strftime("%Y-%m-%d")
            if self.day_key != dk: self.day_key, self.day_eq0, self.day_trades = dk, eq, 0
            if self.halted and self.halt_until and now.isoformat(timespec="seconds") >= self.halt_until:
                self.resume(prices)
            self.peak_eq = max(self.peak_eq, eq)
            dd = (self.peak_eq - eq) / self.peak_eq * 100 if self.peak_eq > 0 else 0.0
            if dd >= rp["max_dd"] and not self.halted:
                self.halt(f"KILL DD {dd:.1f}%", rp["cooldown_h"], now)
                for s in list(self.positions):
                    if s in prices:
                        msgs.append(f"{s}: SAT {self._sell(s, prices[s], date, 'KILL'):+.0f} TL")
                return msgs
            hold_days = max(1, round(rp["max_hold"] * 5 / 7))
            for s in list(self.positions):
                p, px = self.positions[s], prices.get(s)
                if not px: continue
                p["high"] = max(p["high"], px)
                stop, trail = stop_level(p["entry"], p["high"], p["sd0"], p["atr"], rp)
                pct, reason = (px / p["entry"] - 1) * 100, None
                if px <= stop: reason = f"{'TRAILING' if trail else 'STOP'} {pct:+.1f}%"
                else:
                    try:
                        if bday_count(datetime.fromisoformat(p["ts"]), now) >= hold_days: reason = "ZAMAN STOPU"
                    except Exception: pass
                if reason is None and dec.get(s, {}).get("action") == "SAT": reason = "SINYAL SAT"
                if reason: msgs.append(f"{s}: SAT {self._sell(s, px, date, reason):+.0f} TL")
                elif rp.get("tp_on", True) and not p.get("tp_done") and p["qty"] >= 2 and px >= p["entry"] + rp["tp_r"] * p["sd0"]:
                    q = max(1, int(p["qty"] * rp["tp_frac"]))
                    msgs.append(f"{s}: TP {q} {self._sell(s, px, date, f'TP {pct:+.1f}%', q):+.0f} TL")
            if self.streak >= rp["loss_streak"] and not self.halted:
                self.halt(f"SOGUMA ({self.streak})", rp["cooldown_h"], now)
            if self.halted: return msgs
            eq = self.total_value(prices)
            if self.day_eq0 > 0 and (eq - self.day_eq0) / self.day_eq0 * 100 <= -rp["daily_loss"]:
                self.halt("GUNLUK LIMIT", max(1, rp["cooldown_h"] // 4), now); return msgs
            if stale: return msgs
            rmult = clamp(1 - 0.7 * dd / max(rp["max_dd"], 1e-9), 0.3, 1.0) if rp.get("dd_throttle", True) else 1.0
            cands = sorted([((d["probs"][2] + d["probs_long"][2]) / 2 - max(d["probs"][0], d["probs"][1]), s)
                            for s, d in dec.items()
                            if d["action"] == "AL" and s not in self.positions
                            and prices.get(s) and self.last_exit.get(s) != dk], reverse=True)
            for _, s in cands:
                if len(self.positions) >= rp["max_positions"] or self.day_trades >= rp["max_trades_day"]: break
                if rp.get("corr_on", True) and not self._corr_ok(s, rets, rp["corr_max"]): continue
                d = dec[s]
                px = prices[s] * (1 + SLIPPAGE)
                atr = d["atr"] or px * 0.02
                total_eq = self.total_value(prices)
                q, sd = size_pos(eq, self.cash, self.exposure(prices), px, atr,
                                 d["probs"], d["probs_long"], rp, self.positions,
                                 rmult * self.nn.error_analyzer.regime_mult(d["regime"]))
                sek_now = sektor_exposure(self.positions, prices).get(sektor_of(s), 0.0)
                q = min(q, int(max(0.0, rp["max_sector"] * total_eq - sek_now) / px)) if total_eq > 0 else 0
                if q >= 1:
                    turn = safe_float(rows[s].get("Turnover20") if s in rows else None, 1e7)
                    self._buy(s, prices[s], date, q, sd, atr, d.get("feat"), "AI AL",
                              turn=turn, probs=d["probs"], regime=d["regime"])
                    msgs.append(f"{s}: AL {q} @ {px:.2f}")
            return msgs

    def federated_push(self):
        try:
            vs = self.val_stats or {}
            self.federated_score = float(vs.get("t_stat", 0) + 0.3 * vs.get("sharpe", 0))
            with self.lock: st0 = self.nn.get_state()
            return get_federated().push(BOT_ID, st0, self.federated_score)
        except Exception: return False

    # 🐛 SORUN 1 DÜZELTME: federated_pull optimizer state sıfırlar
    def federated_pull(self, beta=0.5):
        if self.hold is None: return False, "hold yok"
        fed = get_federated()
        blended = fed.blend(self.nn, beta)
        if blended is None: return False, "global model yok"
        h = self.hold
        with self.lock:
            old = [nn.snapshot() for nn in self.nn.nets]
            l0 = self.nn.evaluate(h["X"], h["y"], h["y_long"])[0]
            for nn, b in zip(self.nn.nets, blended): nn.restore(b)
            l1 = self.nn.evaluate(h["X"], h["y"], h["y_long"])[0]
            if l1 <= l0 * 1.005:
                # ✅ Optimizer state sıfırla (Adam momentum uyumsuzluğunu önler)
                for nn in self.nn.nets:
                    nn.m = {k: np.zeros_like(v) for k, v in nn.P.items()}
                    nn.v = {k: np.zeros_like(v) for k, v in nn.P.items()}
                    nn.t = 0
                return True, f"uygulandı (kayıp {l0:.4f}→{l1:.4f})"
            for nn, o in zip(self.nn.nets, old): nn.restore(o)
        return False, f"reddedildi (kayıp {l0:.4f}→{l1:.4f})"

    def get_state(self):
        keys = ["cash", "initial", "positions", "trades", "wins", "losses", "eq_hist", "peak_eq",
                "halted", "halt_reason", "halt_until", "streak", "day_key", "day_eq0", "day_trades",
                "pretrained", "val_stats", "last_exit", "rl_on", "rl_gain", "federated_score"]
        d = {k: getattr(self, k) for k in keys}
        with self.lock: d["nn"] = self.nn.get_state()
        d["si"], d["rl"], d["hmm"] = self.si.get_state(), self.rl.get_state(), get_hmm().get_state()
        return d

    def set_state(self, d):
        for k, v in d.items():
            if k == "nn": self.nn.set_state(v)
            elif k == "si": self.si.set_state(v)
            elif k == "rl" and v: self.rl.set_state(v)
            elif k == "hmm" and v: get_hmm().set_state(v)
            else: setattr(self, k, v)

# ════════════════════════════════════════════════════════════
# 18. SAVE / LOAD / METRICS
# ════════════════════════════════════════════════════════════
def save_state(bot, rp, path=STATE_FILE):
    try:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"ver": BOT_VERSION, "bot": bot.get_state(), "rp": rp},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        bot.nn.replay.save(MEMORY_FILE)
        return True
    except Exception as e:
        log.warning(f"save: {e}"); return False

def load_state(path=STATE_FILE):
    if not os.path.exists(path): return None
    try:
        with open(path, "rb") as f: obj = pickle.load(f)
        return obj if isinstance(obj, dict) and obj.get("ver") == BOT_VERSION else None
    except Exception: return None

def export_prometheus(bot, rp, prices=None):
    try:
        eq = bot.total_value(prices or {})
        vl = bot.val_stats.get("loss", 0) if bot.val_stats else 0
        lines = [
            "# HELP seekdeep_portfolio Portföy değeri", "# TYPE seekdeep_portfolio gauge",
            f'seekdeep_portfolio{{bot="{BOT_ID}",sektor="{AKTIF_SEKTOR}"}} {eq}',
            f'seekdeep_positions{{bot="{BOT_ID}"}} {len(bot.positions)}',
            f'seekdeep_trades{{bot="{BOT_ID}"}} {len(bot.trades)}',
            f'seekdeep_win_rate{{bot="{BOT_ID}"}} {bot.win_rate()}',
            f'seekdeep_evolution{{bot="{BOT_ID}"}} {bot.nn.evo}',
            f'seekdeep_val_loss{{bot="{BOT_ID}"}} {vl}',
            f'seekdeep_router_on{{bot="{BOT_ID}"}} {int(bot.nn.router_on)}',
            f'seekdeep_transformer_on{{bot="{BOT_ID}"}} {int(bot.nn.trans_ens.on)}',
            f'seekdeep_rl_on{{bot="{BOT_ID}"}} {int(bot.rl_on)}',
        ]
        with open(METRICS_FILE, "w") as f: f.write("\n".join(lines) + "\n")
    except Exception: pass# ════════════════════════════════════════════════════════════
# 🐋 PARÇA 4/4: UI — STREAMLIT (21 SEKME)
# ════════════════════════════════════════════════════════════

# ─── Forecast grafikleri ────────────────────────────────
def plot_forecast(f: Forecast, df: pd.DataFrame, n_hist: int = 60):
    d = df.tail(n_hist).copy()
    future_dates = pd.date_range(
        start=pd.Timestamp(d["Date"].iloc[-1]) + pd.Timedelta(days=1),
        periods=f.horizon_days, freq="B")
    today = d["Close"].iloc[-1]
    n_pts = len(future_dates) + 1
    base_line = np.linspace(today, f.base_price, n_pts)
    upper_line = np.linspace(today, f.upper_bound, n_pts)
    lower_line = np.linspace(today, f.lower_bound, n_pts)
    bull_line = np.linspace(today, f.bull_price, n_pts)
    bear_line = np.linspace(today, f.bear_price, n_pts)
    x_all = [d["Date"].iloc[-1]] + list(future_dates)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"], mode="lines",
                              name="Geçmiş", line=dict(color="#50e3c2", width=2)))
    fig.add_trace(go.Scatter(x=x_all + x_all[::-1],
                              y=list(upper_line) + list(lower_line)[::-1],
                              fill="toself", fillcolor="rgba(80, 227, 194, 0.15)",
                              line=dict(color="rgba(255,255,255,0)"),
                              name="%80 Güven Aralığı"))
    fig.add_trace(go.Scatter(x=x_all, y=base_line, mode="lines+markers",
                              name=f"Baz (₺{f.base_price:.2f})",
                              line=dict(color="#f5a623", width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=x_all, y=bull_line, mode="lines",
                              name=f"Boğa (₺{f.bull_price:.2f})",
                              line=dict(color="#26a69a", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=x_all, y=bear_line, mode="lines",
                              name=f"Ayı (₺{f.bear_price:.2f})",
                              line=dict(color="#ef5350", width=1.5, dash="dot")))
    fig.add_vline(x=d["Date"].iloc[-1], line_dash="dot", line_color="gray",
                  annotation_text="Bugün", annotation_position="top")
    fig.update_layout(
        height=500, template="plotly_dark",
        title=f"🎯 {f.stock} Tahmin — {f.horizon_days} gün · Güven %{f.confidence*100:.0f} · Rejim {f.regime}",
        xaxis_title="Tarih", yaxis_title="Fiyat (₺)",
        margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified")
    return fig

def plot_scenarios_bar(f: Forecast):
    labels = ["Ayı", "Alt (%80)", "Baz", "Üst (%80)", "Boğa"]
    prices_ = [f.bear_price, f.lower_bound, f.base_price, f.upper_bound, f.bull_price]
    colors = ["#ef5350", "#f5a623", "#50e3c2", "#f5a623", "#26a69a"]
    fig = go.Figure(go.Bar(x=labels, y=prices_, marker_color=colors,
                            text=[f"₺{p:.2f}" for p in prices_], textposition="outside"))
    fig.add_hline(y=f.current_price, line_dash="dash", line_color="white",
                  annotation_text=f"Şu an ₺{f.current_price:.2f}")
    fig.update_layout(height=350, template="plotly_dark",
                       title=f"{f.stock} Senaryo Analizi", yaxis_title="Fiyat (₺)",
                       margin=dict(l=10, r=10, t=50, b=10))
    return fig

def plot_probability_gauge(f: Forecast):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=f.p_up * 100,
        title={"text": f"{f.stock} P(Yukarı) %"},
        delta={"reference": 50, "increasing": {"color": "#26a69a"},
               "decreasing": {"color": "#ef5350"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#50e3c2"},
            "steps": [
                {"range": [0, 33], "color": "#ef5350"},
                {"range": [33, 66], "color": "#f5a623"},
                {"range": [66, 100], "color": "#26a69a"},
            ],
            "threshold": {"line": {"color": "white", "width": 3},
                          "thickness": 0.75, "value": 50},
        }
    ))
    fig.update_layout(height=280, template="plotly_dark",
                       margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ════════════════════════════════════════════════════════════
# ANA UI
# ════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="SeekDeep", page_icon="🐋", layout="wide")
    sektor_emoji = {"BANKACILIK": "🏦", "HAVACILIK": "✈️", "ENERJI": "⚡"}

    # Mobil CSS
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .stMetric { font-size: 0.8em; }
        .stTabs [data-baseweb="tab-list"] { overflow-x: auto; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Başlık
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.title(f"{BOT_NAME} — Bot #{BOT_ID}")
    c1.caption(f"{sektor_emoji.get(AKTIF_SEKTOR,'🐋')} {AKTIF_SEKTOR} · "
               f"{SEKTOR_BOTLARI[AKTIF_SEKTOR]['aciklama']}")
    (c2.success if borsa_acik() else c2.error)("🟢 AÇIK" if borsa_acik() else "🔴 KAPALI")
    c3.caption(f"🕐 {now_tr():%H:%M:%S}")

    # Üst metrikler
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sürüm", BOT_VERSION)
    k2.metric("Otonom Param", len(TUNE_BOUNDS))
    k3.metric("Kilitli Risk", str(len(LOCKED_KEYS)))
    k4.metric("Doğrulama", "CPCV+PBO+DSR")
    k5.metric("Mimari", "MoE+Trans+RL+Meta")
    st.divider()

    # Session state
    for k, v in {"bot": None, "auto": False, "test_mode": False, "interval": 60,
                 "last_tick": 0.0, "ticks": 0, "log": [], "sel": None, "rp": None,
                 "universe": None, "universe_scores": [], "ma": None,
                 "nas_results": None, "nas_best": None, "_cpcv": None,
                 "_pbo": None, "_stress": None, "_shap": None, "_lime": None,
                 "_pdp": None, "last_agg": 0.0, "_portf": None}.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Bot yükle
    if st.session_state.bot is None:
        b, loaded = Bot(), load_state()
        if loaded:
            try:
                b.set_state(loaded["bot"])
                b.nn.replay.load(MEMORY_FILE)
                st.session_state.rp = loaded.get("rp")
            except Exception as e:
                st.warning(f"State yüklenemedi: {e}")
                b = Bot()
        st.session_state.bot = b

    bot = st.session_state.bot
    st.session_state.rp = validate_risk(st.session_state.rp)
    rp = st.session_state.rp
    now = now_tr()

    # LLM durumu
    na = get_news()
    if NEWS_ENABLED:
        if na.enabled:
            st.toast("🤖 LLM aktif (Groq)", icon="✅")
        else:
            st.toast("⚠️ LLM kapalı (GROQ_API_KEY eksik)", icon="⚠️")

    # Veri yükle
    with st.spinner(f"📥 {AKTIF_SEKTOR} verisi yükleniyor ({N_STOCKS} hisse)..."):
        uni = load_uni(now.strftime("%Y-%m-%d"), AKTIF_SEKTOR)
    dfs = uni["dfs"]
    if not dfs:
        st.error("❌ Veri yok: " + ", ".join(uni["errs"][:5]))
        st.stop()
    if uni["errs"]:
        st.caption("⚠️ Atlanan: " + ", ".join(uni["errs"][:8]))

    # Dataset cache
    if st.session_state.get("dss_key") != (uni["ts"], AKTIF_SEKTOR):
        st.session_state.dss = {nm: make_ds(d, nm, AKTIF_SEKTOR) for nm, d in dfs.items()}
        st.session_state.dss_key = (uni["ts"], AKTIF_SEKTOR)
    datasets = [v for v in st.session_state.dss.values() if v]
    data_date = str(max(d["Date"].iloc[-1] for d in dfs.values()).date())

    # İlk eğitim
    if not bot.pretrained and datasets:
        with st.spinner("🧠 Eğitim (MoE + Transformer + RL + Meta + HMM)..."):
            bot.pretrain(datasets, rp, steps=400, data_date=data_date)
        bot.si.last_run = time.time()
        bot.si.last_data = data_date
        st.session_state.log.append(f"{now:%H:%M:%S} Eğitim tamam")
        save_state(bot, rp)

    # Feature hazırlık
    names, rows, feats = [], {}, []
    for nm, d in dfs.items():
        i = sig_idx(d, now)
        rows[nm] = d.iloc[i]
        feats.append(feature_matrix(d.iloc[[i]], nm)[0])
        names.append(nm)
    F_live = np.array(feats, dtype=np.float32)

    # Sequence (canlı)
    sid_live = np.array([STOCK_LIST.index(nm) for nm in names], dtype=np.int64)
    row_live = np.array([len(dfs[nm]) - 1 if sig_idx(dfs[nm], now) == len(dfs[nm]) - 1 else len(dfs[nm]) - 2
                          for nm in names], dtype=np.int64)
    F_live_seq = None
    if len(F_live) >= SEQ_LEN:
        try:
            F_live_seq = gather_seq(sid_live, row_live, AKTIF_SEKTOR)
        except Exception as e:
            log.warning(f"seq: {e}")

    # Canlı fiyat
    live = live_prices(tuple(HISSELER[n] for n in names))
    prices = {nm: live.get(HISSELER[nm], float(dfs[nm]["Close"].iloc[-1])) for nm in names}

    # Aktif hisse
    if st.session_state.sel is None or st.session_state.sel not in dfs:
        st.session_state.sel = names[0]

    # Recent returns (HMM)
    df_recent_returns = {nm: dfs[nm]["Ret1"].dropna().to_numpy()[-30:] for nm in names}
    # Piyasa rejimi (XU100 üzerinden)
    mkt_regime = None
    try:
        xu = dfs.get("XU100")
        if xu is not None and get_hmm().fitted:
            mkt_regime = get_hmm().predict_name(dfs[list(dfs.keys())[0]]["Ret1"].dropna().to_numpy()[-30:])
    except Exception:
        mkt_regime = None

    # Overlay (haber + temel)
    overlay = {}
    try:
        for nm in names:
            nf = uni.get("news", {}).get(nm, {})
            fd = uni.get("fund", {}).get(nm)
            tilt = 0.0
            if nf.get("n", 0) > 0:
                tilt += 0.02 * float(nf.get("sent", 0)) * float(nf.get("conf", 0.5))
            if fd is not None:
                tilt += 0.015 * fund_score(fd, sektor_of(nm))
            overlay[nm] = clamp(tilt, -0.04, 0.04)
    except Exception as e:
        log.warning(f"overlay: {e}")

    # Feature drift
    feat_dr = bot.feat_drift(F_live)

    # Background improver sonucu al
    bgi = get_bgi()
    res = bgi.pop(id(bot))
    if res is not None:
        rp = handle_improve_result(bot, res, rp)

    # Otomatik öz-gelişim tetikle (background)
    if rp["si_auto"] and bot.pretrained and datasets and not bgi.busy():
        t_ = time.time()
        if bot.si.due(t_, data_date, rp["si_interval_min"]) or (feat_dr > 1.2 and t_ - bot.si.last_run > 600):
            if bgi.start(bot, datasets, rp, data_date, "auto", float(rp["si_budget_s"])):
                st.toast("🧬 Öz-gelişim arka planda çalışıyor", icon="🔄")

    # Scan
    dec = bot.scan(names, F_live, rows, rp, F_live_seq, mkt_regime, overlay)
    sec = st.session_state.sel
    df, sd_ = dfs[sec], dec[sec]
    price = prices[sec]

    # Bayat veri kontrolü
    last_data_ts = pd.Timestamp(df["Date"].iloc[-1]).date()
    stale = (now.date() - last_data_ts).days > 5

    # Otomatik döngü
    auto_msgs = []
    if (st.session_state.auto
            and time.time() - st.session_state.last_tick >= st.session_state.interval - 1
            and (borsa_acik() or st.session_state.test_mode)
            and bot.pretrained):
        st.session_state.last_tick = time.time()
        st.session_state.ticks += 1
        rets_live = {nm: dfs[nm]["Ret1"].dropna().to_numpy()[-60:] for nm in names}
        auto_msgs = bot.run_cycle(dec, prices, rp, now.strftime("%Y-%m-%d %H:%M"), now,
                                   rows=rows, rets=rets_live, stale=stale)
        v_ = bot.total_value(prices)
        bot.record(v_)
        db_eq_add(v_, bot.cash, len(bot.positions))
        if auto_msgs:
            st.session_state.log = (st.session_state.log + [f"{now:%H:%M:%S} " + " | ".join(auto_msgs)])[-100:]
        if st.session_state.ticks % 10 == 0:
            save_state(bot, rp)
        if st.session_state.ticks % 100 == 0:
            bot.federated_push()
            if time.time() - st.session_state.last_agg > 3600:
                if get_federated().aggregate():
                    ok, msg = bot.federated_pull(rp.get("fed_beta", 0.5))
                    st.session_state.log.append(f"{now:%H:%M:%S} Federated: {msg}")
                st.session_state.last_agg = time.time()
        export_prometheus(bot, rp, prices)

    if st.session_state.auto:
        st_autorefresh(interval=st.session_state.interval * 1000, key="rf")
    cur_val = bot.total_value(prices)
    bot.record(cur_val)

    # ════════════════════════════════════════════════════════
    # SIDEBAR
    # ════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown(f"## {sektor_emoji.get(AKTIF_SEKTOR,'🐋')} SeekDeep")
        st.caption(f"Bot #{BOT_ID} · {AKTIF_SEKTOR}")
        st.selectbox("📊 Hisse", names, key="sel")
        st.metric(sec, f"{price:.2f} TL",
                  f"{(price / float(df['Close'].iloc[-2]) - 1) * 100:+.2f}%")
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
        s1.metric("Poz", len(bot.positions))
        s2.metric("Hafıza", bot.nn.replay.size())
        s3.metric("Evrim", bot.nn.evo)
        vl_str = f"{bot.nn.best_vl:.4f}" if np.isfinite(bot.nn.best_vl) else "-"
        st.caption(f"Val loss: {vl_str} · 🧬 tur: {bot.si.runs}")
        st.caption(f"Router: {'AÇIK' if bot.nn.router_on else 'kapalı'} ({bot.nn.router_gain:+.4f})")
        st.caption(f"Transformer: {'AÇIK' if bot.nn.trans_ens.on else 'kapalı'} ({bot.nn.trans_ens.gain:+.4f})")
        st.caption(f"RL: {'AÇIK' if bot.rl_on else 'kapalı'} · HMM: {'eğitildi' if get_hmm().fitted else 'bekliyor'}")
        st.caption(f"EWC: {'konsolide' if bot.nn.ewc_consolidated else 'bekliyor'}")
        if bgi.busy():
            st.info("🧬 Öz-gelişim çalışıyor (arka plan)")

        # Aktif hisse tahmini
        st.divider()
        st.markdown("### 🔮 Tahmin")
        try:
            f_aktif = forecast_stock(bot, sec, df, price, sd_, rp)
            yon_emoji = ("🟢" if f_aktif.expected_return_pct > 1
                         else ("🔴" if f_aktif.expected_return_pct < -1 else "🟡"))
            st.markdown(f"{yon_emoji} **Hedef:** ₺{f_aktif.target_price:.2f} "
                        f"({f_aktif.expected_return_pct:+.2f}%)")
            st.caption(f"Aralık: ₺{f_aktif.lower_bound:.2f} — ₺{f_aktif.upper_bound:.2f}")
            st.caption(f"P(Yukarı) %{f_aktif.p_up*100:.0f} · Güven %{f_aktif.confidence*100:.0f}")
        except Exception:
            pass

        if bot.halted:
            st.error(f"⛔ {bot.halt_reason}")
        if stale:
            st.warning("⚠️ Veri bayat (>5 gün)")

        if st.button("💾 Kaydet", use_container_width=True):
            st.toast("✅" if save_state(bot, rp) else "❌")
        if st.button("📤 Federated Push", use_container_width=True):
            if bot.federated_push(): st.toast("✅ Gönderildi")
        if st.button("📥 Federated Pull", use_container_width=True):
            ok, msg = bot.federated_pull(rp.get("fed_beta", 0.5))
            st.toast(("✅ " if ok else "⚠️ ") + msg)
            if ok: st.rerun()
        if st.button("🛑 ACİL DUR", use_container_width=True, type="primary"):
            for s in list(bot.positions):
                if s in prices:
                    bot._sell(s, prices[s], now, "ACIL")
            bot.halted, bot.halt_reason = True, "ACİL"
            st.rerun()
        if st.button("▶️ Devam", use_container_width=True):
            bot.resume(prices); st.rerun()
        if st.button("🗑️ Sıfırla", use_container_width=True):
            for f_ in (STATE_FILE, MEMORY_FILE):
                try: os.remove(f_)
                except OSError: pass
            db_clear()
            st.session_state.bot = Bot()
            st.session_state.log = []
            st.rerun()

    # ════════════════════════════════════════════════════════
    # ÜST METRİKLER
    # ════════════════════════════════════════════════════════
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Fiyat", f"{price:.2f}")
    m2.metric("Karar", sd_["action"], f"P(AL) %{sd_['probs'][2] * 100:.0f}")
    m3.metric("Portföy", f"{cur_val:,.0f}",
              f"%{(cur_val / bot.initial - 1) * 100:+.2f}")
    m4.metric("İşlem", len(bot.trades),
              f"kaz %{bot.win_rate() * 100:.0f}" if bot.trades else None)
    m5.metric("Rejim", sd_["regime"])
    m6.metric("Belirsizlik", f"{sd_['unc']:.2f}")
    if sd_["note"]:
        st.warning(f"AL engellendi: {sd_['note']}")
    if auto_msgs:
        st.info(" | ".join(auto_msgs[:5]))

    # Forecast uyarı
    if bot.pretrained and len(bot.positions) < rp["max_positions"]:
        try:
            fc_rows_alert = forecast_all(bot, names, dfs, prices, dec, rp)
            alerts = check_forecast_alerts(fc_rows_alert, threshold=5.0)
            high = [a for a in alerts if a.expected_return_pct > 0 and a.p_up > 0.55]
            if high:
                top3 = sorted(high, key=lambda x: -x.expected_return_pct)[:3]
                txt = " | ".join([f"🔮 {a.stock}: ₺{a.current_price:.2f}→₺{a.target_price:.2f} "
                                  f"({a.expected_return_pct:+.1f}%)" for a in top3])
                st.info(f"**Yüksek potansiyel:** {txt}")
        except Exception:
            pass

    # ════════════════════════════════════════════════════════
    # SEKMELER
    # ════════════════════════════════════════════════════════
    tabs = st.tabs([
        "🌍 Evren", "🎛️ Kontrol", "📈 Grafik", "🎯 Karar", "🧠 Beyin",
        "🧬 Öz-Gelişim", "📊 Analiz", "📊 Kalibrasyon", "🔬 CPCV/PBO",
        "💼 Temel", "🌐 Makro", "👥 Ajanlar", "🔬 NAS", "🔄 Federated",
        "⚡ Stress", "🔍 XAI", "📢 KAP", "🔮 Tahmin", "🧠 Meta",
        "💾 DB", "💬 Log"
    ])
    (tU, tK, tG, tD, tB, tSI, tA, tCAL, tCPCV, tFUN, tMAC, tMA,
     tNAS, tFED, tSTRESS, tXAI, tKAP, tFCAST, tMETA, tDB, tCH) = tabs

    # Slider helpers
    def _cb(name, key):
        st.session_state.rp[name] = clamp(safe_float(st.session_state[key]),
                                          *RISK_BOUNDS.get(name, (0, 1)))
    def rp_sl(name, label, lo, hi, step):
        key = f"rp_{name}"
        if key not in st.session_state:
            st.session_state[key] = type(step)(clamp(rp[name], lo, hi))
        st.slider(label, lo, hi, step=step, key=key, on_change=_cb, args=(name, key))
    def rp_tg(name, label):
        key = f"rp_{name}"
        if key not in st.session_state:
            st.session_state[key] = bool(rp[name])
        st.toggle(label, key=key,
                  on_change=lambda n=name, k=key: st.session_state.rp.__setitem__(n, bool(st.session_state[k])))

    # ─── 🌍 EVREN ───────────────────────────────────────────
    with tU:
        st.subheader(f"🌍 Dinamik Evren · {AKTIF_SEKTOR}")
        st.caption(f"Bot #{BOT_ID} · {N_STOCKS} hisse · ₺{BOT_SERMAYE:,.0f}")
        if st.button("🔄 Universe'ü Yenile (Tüm BIST)", use_container_width=True):
            with st.spinner("500+ hisse taranıyor..."):
                top_codes, all_scores = scan_universe(top_n=40)
                st.session_state.universe = top_codes
                st.session_state.universe_scores = all_scores
                st.success(f"✅ {len(top_codes)} hisse seçildi")
                st.rerun()
        scores = st.session_state.get("universe_scores", [])
        if scores:
            sdf = pd.DataFrame(scores).head(50)
            sdf = sdf[["code", "total", "trend", "mom", "atr_pct", "vol_ratio", "turnover", "price"]]
            sdf.columns = ["Kod", "Skor", "Trend", "Momentum", "ATR%", "Hacim Oranı", "Hacim (TL)", "Fiyat"]
            sdf["ATR%"] = sdf["ATR%"] * 100
            sdf["Hacim (TL)"] = sdf["Hacim (TL)"].apply(lambda x: f"{x/1e6:.1f}M")
            st.dataframe(sdf, use_container_width=True, hide_index=True)
        st.divider()
        st.markdown("### 📋 Bu Botun Hisseleri")
        st.write(", ".join(BOT_HISSELER))

    # ─── 🎛️ KONTROL ─────────────────────────────────────────
    with tK:
        cA, cB, cC = st.columns(3)
        with cA:
            st.markdown("**🔒 Kilitli (Sende)**")
            rp_sl("max_dd", "Kill DD %", 3.0, 50.0, 0.5)
            rp_sl("daily_loss", "Günlük zarar %", 0.5, 15.0, 0.5)
            rp_sl("max_positions", "Max poz", 1, 20, 1)
            rp_sl("max_exposure", "Maruziyet", 0.2, 1.0, 0.05)
            rp_sl("max_sector", "Sektör max", 0.1, 1.0, 0.05)
            st.divider()
            st.markdown(f"**🤖 Otonom ({len(TUNE_BOUNDS)} param)**")
            st.caption("Bot kendisi öğreniyor")
        with cB:
            st.markdown("**Model Kapısı**")
            rp_tg("gate_on", "Aktif")
            rp_sl("gate_min_n", "Min AL", 3, 100, 1)
            rp_sl("dsr_confidence", "DSR eşiği", 0.5, 0.999, 0.005)
            rp_sl("consec_improve", "Ardışık", 1, 5, 1)
            rp_tg("auto_adopt", "Oto uygula")
            rp_tg("rl_veto", "RL veto")
            rp_sl("rl_veto_p", "RL veto eşiği", 0.05, 0.6, 0.01)
        with cC:
            st.markdown("**Öz-Gelişim**")
            rp_tg("si_auto", "🧬 Aktif")
            rp_sl("si_interval_min", "Tur aralığı", 5, 720, 5)
            rp_sl("si_budget_s", "Bütçe (sn)", 2, 120, 1)
            st.divider()
            st.markdown("**Federated**")
            rp_sl("fed_beta", "Global blend β", 0.0, 1.0, 0.05)
        st.divider()
        st.markdown("### 📦 Açık Pozisyonlar")
        if not bot.positions:
            st.info("Yok")
        else:
            prow = []
            for nm, p in bot.positions.items():
                cur = prices.get(nm, p["entry"])
                stop, _ = stop_level(p["entry"], p["high"], p["sd0"], p["atr"], rp)
                prow.append({"Hisse": nm, "Adet": p["qty"],
                             "Giriş": round(p["entry"], 2),
                             "Şu an": round(cur, 2),
                             "K/Z ₺": round((cur - p["entry"]) * p["qty"] + p.get("realized", 0.0)),
                             "K/Z %": round((cur / p["entry"] - 1) * 100, 2),
                             "Stop": round(stop, 2),
                             "TP": "✅" if p.get("tp_done") else "-"})
            st.dataframe(pd.DataFrame(prow), use_container_width=True, hide_index=True)

    # ─── 📈 GRAFİK ──────────────────────────────────────────
    with tG:
        n_show = st.select_slider("Gün", options=[60, 120, 180, 250], value=120)
        d = df.tail(n_show)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=d["Date"], open=d["Open"], high=d["High"],
                                      low=d["Low"], close=d["Close"], name="Fiyat"), row=1, col=1)
        for col, clr in [("SMA20", "#f5a623"), ("SMA50", "#4a90e2"), ("SMA200", "#bd10e0")]:
            fig.add_trace(go.Scatter(x=d["Date"], y=d[col], name=col,
                                      line=dict(width=1.2, color=clr)), row=1, col=1)
        pos_ = bot.positions.get(sec)
        if pos_:
            stp, _ = stop_level(pos_["entry"], pos_["high"], pos_["sd0"], pos_["atr"], rp)
            fig.add_hline(y=stp, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=pos_["entry"] + rp["tp_r"] * pos_["sd0"],
                          line_dash="dash", line_color="lime", row=1, col=1)
        for t in bot.trades[-100:]:
            if t["stock"] == sec:
                try:
                    td_ = pd.Timestamp(t["date"])
                    if td_.tzinfo is not None:
                        td_ = td_.tz_localize(None)
                    if td_ >= d["Date"].iloc[0]:
                        up_ = t["action"] == "AL"
                        fig.add_trace(go.Scatter(x=[td_], y=[t["price"]],
                                                  mode="markers", showlegend=False,
                                                  marker=dict(color="lime" if up_ else "red",
                                                              size=13,
                                                              symbol="triangle-up" if up_ else "triangle-down")),
                                      row=1, col=1)
                except Exception:
                    pass
        fig.add_trace(go.Scatter(x=d["Date"], y=d["RSI"], showlegend=False,
                                  line=dict(color="#50e3c2")), row=2, col=1)
        fig.add_trace(go.Scatter(x=d["Date"], y=d["MFI"], showlegend=False,
                                  line=dict(color="#f5a623", width=1)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        fig.add_trace(go.Bar(x=d["Date"], y=d["MACDh"], showlegend=False,
                              marker_color=np.where(d["MACDh"] >= 0, "#26a69a", "#ef5350")),
                      row=3, col=1)
        fig.update_layout(height=700, template="plotly_dark",
                           xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # ─── 🎯 KARAR ───────────────────────────────────────────
    with tD:
        st.subheader(f"🎯 {sec} Karar")
        ps_, pl_ = sd_["probs"], sd_["probs_long"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Karar", sd_["action"])
        k2.metric("P(AL) kısa", f"%{ps_[2] * 100:.1f}",
                  f"TUT %{ps_[1] * 100:.0f} · SAT %{ps_[0] * 100:.0f}")
        k3.metric("P(AL) uzun", f"%{pl_[2] * 100:.1f}",
                  f"TUT %{pl_[1] * 100:.0f} · SAT %{pl_[0] * 100:.0f}")
        k4.metric("Uzun hakem",
                  "🟢 BULL" if pl_[2] >= rp["p_buy_long"]
                  else ("🔴 BEAR" if pl_[0] >= rp["p_sell"] else "⚪ NÖTR"))
        st.divider()
        trows = [{"Hisse": nm, "Fiyat": round(prices[nm], 2), "Karar": d_["action"],
                  "P(AL) kısa %": round(d_["probs"][2] * 100, 0),
                  "P(AL) uzun %": round(d_["probs_long"][2] * 100, 0),
                  "Tilt": round(d_.get("tilt", 0), 3),
                  "RL P(AL)": round(d_.get("rl_p", 0) or 0, 2),
                  "Belirsizlik": round(d_["unc"], 2),
                  "Rejim": d_["regime"], "Not": d_["note"]}
                 for nm, d_ in dec.items()]
        st.dataframe(pd.DataFrame(sorted(trows,
                                          key=lambda x: -(x["P(AL) kısa %"] + x["P(AL) uzun %"]) / 2)),
                     use_container_width=True, hide_index=True)
        fb = go.Figure([
            go.Bar(name="Kısa", x=[r["Hisse"] for r in trows],
                   y=[r["P(AL) kısa %"] for r in trows]),
            go.Bar(name="Uzun", x=[r["Hisse"] for r in trows],
                   y=[r["P(AL) uzun %"] for r in trows])
        ])
        fb.update_layout(height=260, template="plotly_dark", barmode="group",
                          margin=dict(l=10, r=10, t=30, b=10), title="P(AL) karşılaştırma")
        st.plotly_chart(fb, use_container_width=True)
        vs = bot.val_stats
        if vs:
            st.markdown("### 🤖 Otonom Performans")
            o1, o2, o3, o4, o5 = st.columns(5)
            o1.metric("T-stat", f"{vs.get('t_stat', 0):.2f}",
                      "anlamlı" if vs.get('t_stat', 0) > 2 else "zayıf")
            o2.metric("Sharpe", f"{vs.get('sharpe', 0):.2f}")
            o3.metric("Sortino", f"{vs.get('sortino', 0):.2f}")
            o4.metric("PF", f"{vs.get('pf', 0):.2f}")
            o5.metric("DD", f"%{vs.get('dd', 0) * 100:.1f}")
            st.markdown("### 🚪 Model Kapısı")
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("Val acc", f"%{vs['acc'] * 100:.1f}",
                      f"taban %{vs['base'] * 100:.1f}")
            g2.metric("AL çağrı", vs["al_n"])
            g3.metric("AL isabet", f"%{vs['al_prec'] * 100:.1f}")
            g4.metric("AL ort.", f"%{vs['al_ret'] * 100:+.2f}")
            g5.metric("Kapı", "✅" if bot.edge_ok(rp) else "⛔")

    # ─── 🧠 BEYİN ───────────────────────────────────────────
    with tB:
        st.subheader("🧠 Beyin")
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Evrim", bot.nn.evo)
        b2.metric("En iyi net", f"#{int(np.argmax(bot.nn.perf))}")
        b3.metric("Hafıza", bot.nn.replay.size())
        b4.metric("PBT", f"{bot.nn.pbt_acc}/{bot.nn.pbt_n}")
        b5.metric("Router", "AÇIK" if bot.nn.router_on else "kapalı",
                  f"{bot.nn.router_gain:+.4f}")
        for i, p in enumerate(bot.nn.perf):
            nn_i = bot.nn.nets[i]
            st.progress(float(min(max(p, 0.0), 1.0)),
                        text=f"Net #{i}: {p:.3f} · lr {nn_i.lr:.4f} · "
                             f"drop {nn_i.dropout:.2f} · SWA {nn_i.swa_wins}")
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🐋 Transformer")
            te = bot.nn.trans_ens
            st.metric("Durum", "🟢 AÇIK" if te.on else "🔴 kapalı")
            st.metric("Kazanç", f"{te.gain:+.4f}")
        with col2:
            st.markdown("### 🤖 RL (PPO)")
            st.metric("Durum", "🟢 AÇIK" if bot.rl_on else "🔴 kapalı")
            st.metric("Güncelleme", bot.rl.t if bot.rl else 0)
            st.metric("Kazanç", f"{bot.rl_gain:+.4f}")
        with col3:
            st.markdown("### 🧬 Meta")
            st.metric("HMM", "eğitildi" if get_hmm().fitted else "bekliyor")
            st.metric("EWC", "konsolide" if bot.nn.ewc_consolidated else "bekliyor")
            st.metric("Reptile", f"{len(bot.nn.reptile.history)} tur")
        imp = feat_importance(bot)
        top = np.argsort(imp)[::-1][:20][::-1]
        fi = go.Figure(go.Bar(
            x=imp[top] * 100,
            y=[FEAT_NAMES[i] if i < len(FEAT_NAMES) else f"f{i}" for i in top],
            orientation="h"))
        fi.update_layout(height=480, template="plotly_dark",
                          margin=dict(l=10, r=10, t=30, b=10), title="Feature Önemi (%)")
        st.plotly_chart(fi, use_container_width=True)
        if st.button("🔄 10 Adım Hızlı Evrim", use_container_width=True) and datasets:
            sp_ = split_ds(merge_ds(datasets))
            if sp_:
                with st.spinner("Evrim..."):
                    for _ in range(10):
                        bot.nn.evolve_step(sp_)
                    bot.nn.fit_router(sp_)
                    try: train_transformer(bot.nn, sp_, steps=20, seed=bot.nn.evo)
                    except Exception: pass
                bot.reeval(rp); save_state(bot, rp)
                st.success("Tamam"); st.rerun()

    # ─── 🧬 ÖZ-GELİŞİM ──────────────────────────────────────
    with tSI:
        st.subheader("🧬 Öz-Gelişim (25 Otonom Parametre)")
        si = bot.si
        i1, i2, i3, i4, i5 = st.columns(5)
        i1.metric("Tur", si.runs)
        i2.metric("Drift", f"{si.drift:.2f}",
                  "bayat" if si.drift > 1.15 else "sağlıklı",
                  delta_color="inverse" if si.drift > 1.15 else "normal")
        i3.metric("Özellik kayması", f"{feat_dr:.2f}")
        i4.metric("Ardışık", f"{si.consec_improve}/{rp['consec_improve']}")
        i5.metric("Son tur", f"{int((time.time() - si.last_run) / 60)} dk" if si.last_run else "-")
        a1, a2, a3 = st.columns(3)
        if a1.button("🧬 Şimdi Geliştir (Arka Plan)", use_container_width=True) and datasets:
            if bgi.start(bot, datasets, rp, data_date, "manuel", float(rp["si_budget_s"])):
                st.success("Arka planda başlatıldı"); st.rerun()
            else:
                st.warning("Zaten çalışıyor")
        if si.proposal and a2.button("✅ Öneriyi Uygula", use_container_width=True):
            st.session_state.rp = validate_risk({**rp,
                **{k: si.proposal["params"][k] for k in TUNE_BOUNDS if k in si.proposal["params"]}})
            si.proposal = None
            si.consec_improve = 0
            si.pending_rp = None
            for k in TUNE_BOUNDS:
                st.session_state.pop(f"rp_{k}", None)
            bot.reeval(st.session_state.rp); st.rerun()
        if si.rp_hist and a3.button("↩️ Geri Dön", use_container_width=True):
            prev = si.rp_hist.pop()
            st.session_state.rp = validate_risk({**rp, **prev})
            for k in TUNE_BOUNDS:
                st.session_state.pop(f"rp_{k}", None)
            bot.reeval(st.session_state.rp); st.rerun()
        if si.proposal:
            st.info(f"Öneri (≥{si.proposal.get('wait', 0)} tur daha)")
        st.markdown("### 🤖 Otonom Öğrenilen (25)")
        groups = {
            "🎯 Sinyal": ["p_buy", "p_buy_long", "margin", "unc_max", "use_long_gate", "use_regime"],
            "🛑 Stop/TP": ["sl_atr", "trail_act_r", "trail_atr", "tp_r", "tp_frac", "be_r", "max_hold"],
            "⚖️ Risk": ["risk_per_trade", "max_pos", "conf_sizing", "cooldown_h",
                         "max_trades_day", "loss_streak", "corr_max"],
            "🌊 Rejim/Filtre": ["regime_bull_adj", "regime_bear_adj",
                                 "regime_range_adj", "regime_vol_adj", "min_daily_turnover"],
        }
        gcols = st.columns(4)
        for i, (title, keys) in enumerate(groups.items()):
            with gcols[i]:
                st.caption(title)
                for k in keys:
                    if k in rp:
                        st.text(f"{k}: {rp[k]}")
        st.markdown("### 🔒 Kilitli (Sende)")
        lc = st.columns(len(LOCKED_KEYS))
        for i, k in enumerate(LOCKED_KEYS):
            lc[i].metric(k, f"{rp[k]}")
        st.markdown("### 📊 Hata Analizi")
        st.text(bot.nn.error_analyzer.report())
        if si.history:
            hrows = [{"Zaman": h["ts"][5:16], "Neden": h["reason"],
                      "Drift": round(h.get("drift", 0), 2),
                      "VL": f"{h.get('vl0',0):.4f}→{h.get('vl1',0):.4f}",
                      "te_acc": f"{h.get('te_acc',0):.3f}" if "te_acc" in h else "-",
                      "DSR": f"{h.get('dsr',0):.3f}" if "dsr" in h else "-",
                      "PBO": f"{h.get('pbo',{}).get('pbo',0):.2f}" if "pbo" in h else "-",
                      "CPCV": f"{h.get('cpcv',{}).get('sharpe_cpcv',0):.2f}" if "cpcv" in h else "-",
                      "Router": h.get("router", "-"),
                      "✅": "EVET" if h.get("adopted") else "-",
                      "sn": h.get("sec", 0)}
                     for h in reversed(si.history[-15:])]
            st.dataframe(pd.DataFrame(hrows), use_container_width=True, hide_index=True)
        if len(bot.eq_hist) > 2:
            fe = go.Figure(go.Scatter(y=bot.eq_hist, mode="lines",
                                       line=dict(color="#50e3c2")))
            fe.update_layout(height=260, template="plotly_dark",
                              margin=dict(l=10, r=10, t=30, b=10), title="Portföy Eğrisi")
            st.plotly_chart(fe, use_container_width=True)

    # ─── 📊 ANALİZ ──────────────────────────────────────────
    with tA:
        st.subheader("📊 Performans & Risk")
        eq_ = db_eq(5000)
        if len(eq_) > 3:
            df_eq = pd.DataFrame(eq_)
            df_eq["day"] = df_eq["ts"].str[:10]
            v = df_eq.groupby("day")["value"].last().to_numpy(float)
            if len(v) > 3:
                r = np.diff(v) / v[:-1]
                neg = r[r < 0]
                sh = float(r.mean() / (r.std() + 1e-12) * math.sqrt(252))
                so = (float(r.mean() / (neg.std() + 1e-12) * math.sqrt(252))
                      if len(neg) > 1 else 0.0)
                peak = np.maximum.accumulate(v)
                dd = float(((peak - v) / peak).max())
                tot = float(v[-1] / v[0] - 1)
                e1, e2, e3, e4, e5 = st.columns(5)
                e1.metric("Sharpe", f"{sh:.2f}")
                e2.metric("Sortino", f"{so:.2f}")
                e3.metric("Max DD", f"%{dd*100:.1f}")
                e4.metric("Getiri", f"%{tot*100:+.1f}")
                e5.metric("Calmar", f"{tot/dd:.2f}" if dd > 1e-9 else "∞")
        var_res = calc_portfolio_var(bot, rp)
        if var_res["n"] >= 20:
            st.markdown("### 📉 VaR / CVaR")
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("VaR %95", f"%{var_res['var95']*100:.2f}")
            v2.metric("CVaR %95", f"%{var_res['cvar95']*100:.2f}")
            v3.metric("VaR %99", f"%{var_res['var99']*100:.2f}")
            v4.metric("CVaR %99", f"%{var_res['cvar99']*100:.2f}")
        pn = [t["pnl"] / bot.initial for t in bot.trades
              if t.get("action") == "SAT" and "pnl" in t]
        if len(pn) >= 8:
            rng_mc = np.random.default_rng(1)
            sims = rng_mc.choice(pn, (2000, 100))
            cum = np.cumsum(sims, axis=1)
            final = cum[:, -1]
            full = np.concatenate([np.zeros((2000, 1)), cum], axis=1)
            dds = (np.maximum.accumulate(full, axis=1) - full).max(axis=1)
            st.markdown("**🎲 Monte Carlo (100 işlem)**")
            c_ = st.columns(5)
            c_[0].metric("Medyan", f"%{np.percentile(final, 50)*100:+.1f}")
            c_[1].metric("P5/P95",
                          f"%{np.percentile(final, 5)*100:+.1f}/%{np.percentile(final, 95)*100:+.1f}")
            c_[2].metric("Zarar", f"%{(final < 0).mean()*100:.0f}")
            c_[3].metric("Medyan DD", f"%{np.percentile(dds, 50)*100:.1f}")
            c_[4].metric("DD P95", f"%{np.percentile(dds, 95)*100:.1f}")
        st.divider()
        st.markdown("### 📐 Portföy Önerisi (HRP / Markowitz)")
        if st.button("📐 Öneri Hesapla", use_container_width=True):
            try:
                st.session_state._portf = portfolio_suggestion(dfs, names)
            except Exception as e:
                st.warning(f"Hata: {e}")
        ps_ = st.session_state.get("_portf")
        if ps_ is not None:
            st.dataframe(ps_, use_container_width=True)

    # ─── 📊 KALİBRASYON ─────────────────────────────────────
    with tCAL:
        st.subheader("📊 Kalibrasyon (TEST)")
        if bot.pretrained and datasets:
            merged = merge_ds(datasets)
            sp_ = split_ds(merged) if merged else None
            if sp_:
                cal = bot.si.calibrate(bot.nn, sp_["Xte"], sp_["yte"])
                if cal:
                    kk = st.columns(4)
                    kk[0].metric("Brier", f"{cal['brier']:.4f}")
                    kk[1].metric("ECE", f"{cal['ece']:.4f}")
                    kk[2].metric("İsabet", f"%{cal['acc']*100:.1f}",
                                  f"taban %{cal['base']*100:.1f}")
                    kk[3].metric("Örnek", cal["n"])
                    if cal["cal"]:
                        cdf = pd.DataFrame(cal["cal"])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=cdf["pred"], y=cdf["actual"],
                                                  mode="markers+lines",
                                                  marker=dict(size=cdf["n"] / max(cdf["n"].max(), 1) * 20 + 5),
                                                  name="Güvenilirlik"))
                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                                  line=dict(dash="dash", color="gray"),
                                                  name="Mükemmel"))
                        fig.update_layout(height=350, template="plotly_dark",
                                          title="Reliability Diagram (TEST)")
                        st.plotly_chart(fig, use_container_width=True)

    # ─── 🔬 CPCV/PBO ────────────────────────────────────────
    with tCPCV:
        st.subheader("🔬 CPCV (OOS) + PBO (CSCV)")
        st.caption("CPCV: modelin eğitimde görmediği val+test havuzunda parametre kombinasyonlarının stabilitesi. "
                   "PBO: CSCV ile aday parametreler arası overfit olasılığı.")
        if bot.pretrained and datasets:
            c1_, c2_ = st.columns(2)
            with c1_:
                st.markdown("### 📊 CPCV-OOS")
                if st.button("🧪 CPCV Çalıştır", use_container_width=True):
                    with st.spinner("CPCV 15 split..."):
                        res = cpcv_oos(bot, rp)
                    if res:
                        st.session_state["_cpcv"] = res
                        db_cpcv_add(res["mean_score"], res["std_score"], res["worst"],
                                    res["best"], res["n_splits"], res["sharpe_cpcv"])
                res = st.session_state.get("_cpcv")
                if res:
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Ort", f"{res['mean_score']:.3f}")
                    k2.metric("Std", f"{res['std_score']:.3f}")
                    k3.metric("CPCV Sharpe", f"{res['sharpe_cpcv']:.2f}")
                    if res["sharpe_cpcv"] > 1.0:
                        st.success("✅ Sağlam")
                    elif res["sharpe_cpcv"] > 0.5:
                        st.warning("⚠️ Dikkatli")
                    else:
                        st.error("❌ Overfit")
            with c2_:
                st.markdown("### 🎲 PBO (CSCV)")
                if st.button("🎲 PBO Çalıştır (Tuner)", use_container_width=True):
                    with st.spinner("Hızlı tuner + CSCV..."):
                        res = run_pbo(bot, rp, budget=4.0)
                    if res:
                        st.session_state["_pbo"] = res
                        db_pbo_add(res["pbo"], res["mean"], res["median"],
                                   res["n_sims"], res["interpretation"])
                res = st.session_state.get("_pbo")
                if res:
                    st.metric("PBO", f"%{res['pbo']*100:.1f}", res["interpretation"])
                    if res["pbo"] < 0.3:
                        st.success("✅ Güvenilir")
                    elif res["pbo"] < 0.5:
                        st.warning("⚠️ Şüpheli")
                    else:
                        st.error("❌ Overfit!")

    # ─── 💼 TEMEL ───────────────────────────────────────────
    with tFUN:
        st.subheader("💼 Temel Analiz (Overlay)")
        if st.button("🔄 Verileri Çek", use_container_width=True):
            with st.spinner("Finansallar çekiliyor..."):
                for nm in names:
                    get_fund().fetch(nm, force=True)
                st.success("Tamamlandı")
        rows_f = []
        for nm in names:
            fd = get_fund().fetch(nm)
            sc = fund_score(fd, sektor_of(nm))
            rows_f.append({"Hisse": nm, "F/K": round(fd.pe, 2), "PD/DD": round(fd.pb, 2),
                           "ROE": f"%{fd.roe*100:.1f}", "Borç/ÖS": round(fd.debt_eq, 2),
                           "Temettü": f"%{fd.div_yield*100:.2f}",
                           "PD (TL)": f"{fd.market_cap/1e9:.1f}M",
                           "Skor": round(sc, 2),
                           "Tilt": round(overlay.get(nm, 0), 3)})
        st.dataframe(pd.DataFrame(rows_f), use_container_width=True, hide_index=True)

    # ─── 🌐 MAKRO ───────────────────────────────────────────
    with tMAC:
        st.subheader("🌐 Makro")
        m = load_macro()
        if m:
            c1_, c2_, c3_, c4_ = st.columns(4)
            c1_.metric("USDTRY", f"{m.get('usdtry',0):.2f}",
                        f"%{m.get('usdtry_ret20',0)*100:+.2f}")
            c2_.metric("EURTRY", f"{m.get('eurtry',0):.2f}",
                        f"%{m.get('eurtry_ret20',0)*100:+.2f}")
            c3_.metric("Altın ($)", f"{m.get('gold',0):.0f}",
                        f"%{m.get('gold_ret20',0)*100:+.2f}")
            c4_.metric("Brent ($)", f"{m.get('brent',0):.2f}",
                        f"%{m.get('brent_ret20',0)*100:+.2f}")
        st.caption("Makro veriler her hissenin feature matrisine zaman hizalı (1 gün gecikmeli) eklenir.")

    # ─── 👥 AJANLAR ─────────────────────────────────────────
    with tMA:
        st.subheader("👥 Multi-Agent (Gerçek Yarış)")
        st.caption("Ajanlar aynı portföy verisinde farklı risk profilleriyle yarışır.")
        if st.button("🔄 Yarış Başlat", use_container_width=True) and bot.hold is not None:
            with st.spinner("12 ajan test ediliyor..."):
                if st.session_state.ma is None:
                    st.session_state.ma = MultiAgent(12)
                Ps_h, Pl_h, U_h = bot.nn.predict_unc(bot.hold["X"], bot.hold["sid"], bot.hold["row"])
                Vh = make_V(bot.hold, Ps_h, Pl_h, U_h)
                st.session_state.ma.run(Vh, rp)
                st.success(f"✅ Tur {st.session_state.ma.rounds} tamamlandı")
        ma = st.session_state.get("ma")
        if ma:
            summary = ma.summary()
            m1, m2, m3 = st.columns(3)
            m1.metric("Ajan", summary["n_agents"])
            m2.metric("Tur", summary["rounds"])
            m3.metric("Ort. Skor", f"{summary['avg_score']:.3f}")
            adf = pd.DataFrame(summary["agents"])
            adf = adf[["id", "profile", "score", "n", "net", "wr", "pf", "dd", "weight"]]
            adf.columns = ["ID", "Profil", "Skor", "İşlem", "Net", "Kazanma", "PF", "DD", "Ağırlık"]
            st.dataframe(adf, use_container_width=True, hide_index=True)
            fig = go.Figure(go.Bar(x=adf["ID"], y=adf["Skor"],
                                    marker_color=np.where(adf["Skor"] > 0, "#26a69a", "#ef5350")))
            fig.update_layout(height=300, template="plotly_dark", title="Ajan Skorları")
            st.plotly_chart(fig, use_container_width=True)

    # ─── 🔬 NAS ─────────────────────────────────────────────
    with tNAS:
        st.subheader("🔬 Neural Architecture Search")
        st.caption("6 farklı mimari (h, dropout, lr, wd) train/val/test üzerinde karşılaştırılır.")
        if st.button("🧪 NAS Çalıştır", use_container_width=True) and datasets:
            merged = merge_ds(datasets)
            sp_ = split_ds(merged) if merged else None
            if sp_:
                with st.spinner("6 mimari test ediliyor (~2 dk)..."):
                    best_cfg, results = nas_search(sp_, steps=80)
                    st.session_state.nas_results = results
                    st.session_state.nas_best = best_cfg
                st.success(f"✅ En iyi: h={best_cfg['h']}, drop={best_cfg['dropout']}, "
                           f"lr={best_cfg['lr']}, wd={best_cfg['wd']}")
        if st.session_state.get("nas_results"):
            rdf = pd.DataFrame([{"h": r["cfg"]["h"], "Dropout": r["cfg"]["dropout"],
                                  "LR": r["cfg"]["lr"], "WD": r["cfg"]["wd"],
                                  "Val Loss": round(r["val_loss"], 4),
                                  "Test Loss": round(r["test_loss"], 4)}
                                 for r in st.session_state["nas_results"]])
            st.dataframe(rdf, use_container_width=True, hide_index=True)
            st.caption(f"Mevcut mimari: h={bot.nn.nn_kwargs.get('h', H_DIM)}, "
                       f"drop={bot.nn.nn_kwargs.get('dropout', 0.15)}")
            if st.button("✅ En İyi Mimariyi Uygula (yeni ağ başlat)", use_container_width=True):
                cfg = st.session_state.get("nas_best")
                if cfg:
                    with bot.lock:
                        bot.nn = BaggedNN(nn_kwargs=cfg, replay=bot.nn.replay)
                    bot.pretrained = False
                    st.success("Yeni mimari hazır. Sıfırdan eğitim için sayfayı yenile.")
                    save_state(bot, rp)

    # ─── 🔄 FEDERATED ───────────────────────────────────────
    with tFED:
        st.subheader("🔄 Federated Learning (3-Bot)")
        fed = get_federated()
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Aktif Bot", f"#{BOT_ID} ({AKTIF_SEKTOR[:4]})")
        f2.metric("Sektör", AKTIF_SEKTOR)
        f3.metric("Federated Tur", fed.rounds)
        f4.metric("Skor", f"{bot.federated_score:.3f}")
        st.divider()
        a1, a2, a3 = st.columns(3)
        if a1.button("📤 Gönder", use_container_width=True):
            if bot.federated_push(): st.success("✅ Gönderildi")
        if a2.button("📥 Al (blend + rollback)", use_container_width=True):
            ok, msg = bot.federated_pull(rp.get("fed_beta", 0.5))
            st.toast(("✅ " if ok else "⚠️ ") + msg)
            if ok: st.rerun()
        if a3.button("🔗 Şimdi Birleştir", use_container_width=True):
            with st.spinner("3 bot ortalanıyor..."):
                gw = fed.aggregate()
            if gw: st.success(f"✅ Tur {fed.rounds} tamamlandı")
            else: st.warning("⚠️ En az 2 bot gerekli")
        st.divider()
        st.markdown("### 🤖 3 Bot Durumu")
        bot_rows = []
        for sektor, cfg in SEKTOR_BOTLARI.items():
            f_ = os.path.join("federated", f"bot_{cfg['bot_id']}.npz")
            mevcut = os.path.exists(f_)
            bot_rows.append({"Bot": f"#{cfg['bot_id']}", "Sektör": sektor,
                             "Hisse": len(cfg["hisseler"]),
                             "Sermaye": f"₺{cfg['sermaye']:,.0f}",
                             "Durum": "🟢 Aktif" if mevcut else "⚪ Bekliyor",
                             "Ben": "✅" if sektor == AKTIF_SEKTOR else ""})
        st.dataframe(pd.DataFrame(bot_rows), use_container_width=True, hide_index=True)
        if fed.history:
            st.markdown("### 📊 Federated Geçmişi")
            hdf = pd.DataFrame([{"Tur": h["round"], "Zaman": h["ts"][5:16],
                                  "Bot": h["n_bots"],
                                  "Skorlar": ", ".join([f"{s:.2f}" for s in h["scores"]])}
                                 for h in reversed(fed.history[-10:])])
            st.dataframe(hdf, use_container_width=True, hide_index=True)

    # ─── ⚡ STRESS ──────────────────────────────────────────
    with tSTRESS:
        st.subheader("⚡ Stress Testing")
        if st.button("🧪 Senaryoları Çalıştır", use_container_width=True) and bot.positions:
            res = stress_test(bot, prices, rp, dfs=dfs)
            st.session_state["_stress"] = res
        res = st.session_state.get("_stress")
        if res:
            sdf = pd.DataFrame(res)
            st.dataframe(sdf, use_container_width=True, hide_index=True)
            fig = go.Figure(go.Bar(
                x=sdf["Senaryo"], y=sdf["Kayıp %"],
                marker_color=["#ef5350" if x > 15 else "#f5a623" if x > 8 else "#26a69a"
                              for x in sdf["Kayıp %"]]))
            fig.update_layout(height=350, template="plotly_dark",
                              title="Senaryo Bazlı Kayıp (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pozisyon varsa ve test çalıştırıldıysa sonuç burada görünür.")

    # ─── 🔍 XAI ─────────────────────────────────────────────
    with tXAI:
        st.subheader("🔍 Açıklanabilir AI")
        if bot.pretrained and bot.hold is not None:
            xai1, xai2, xai3 = st.tabs(["SHAP", "LIME", "PDP"])
            with xai1:
                if st.button("🧮 SHAP Hesapla", use_container_width=True):
                    with st.spinner("SHAP (30 sample)..."):
                        if sec in names:
                            idx = names.index(sec)
                            sv = shap_values(bot, F_live[idx], n_samples=30)
                            st.session_state["_shap"] = sv
                sv = st.session_state.get("_shap")
                if sv:
                    sdf = pd.DataFrame(sv, columns=["Feature", "SHAP"])
                    st.dataframe(sdf, use_container_width=True, hide_index=True)
                    fig = go.Figure(go.Bar(x=sdf["SHAP"], y=sdf["Feature"],
                                            orientation="h",
                                            marker_color=np.where(sdf["SHAP"] > 0, "#26a69a", "#ef5350")))
                    fig.update_layout(height=500, template="plotly_dark",
                                       title="SHAP Değerleri")
                    st.plotly_chart(fig, use_container_width=True)
            with xai2:
                if st.button("🧪 LIME Hesapla", use_container_width=True):
                    if sec in names:
                        idx = names.index(sec)
                        lv = lime_explain(bot, F_live[idx])
                        st.session_state["_lime"] = lv
                lv = st.session_state.get("_lime")
                if lv:
                    ldf = pd.DataFrame(lv, columns=["Feature", "LIME"])
                    st.dataframe(ldf, use_container_width=True, hide_index=True)
            with xai3:
                st.caption("Feature etkisi (PDP)")
                feat_choice = st.selectbox("Feature", FEAT_NAMES[:N_FEAT_TECH], key="pdp_feat")
                if st.button("📈 PDP Çiz", use_container_width=True):
                    fi = FEAT_NAMES.index(feat_choice)
                    pdp_res = partial_dependence(bot, bot.hold["X"], fi)
                    if pdp_res:
                        pdf = pd.DataFrame(pdp_res, columns=["Değer", "P(AL)"])
                        fig = go.Figure(go.Scatter(x=pdf["Değer"], y=pdf["P(AL)"],
                                                    mode="lines+markers"))
                        fig.update_layout(height=350, template="plotly_dark",
                                          title=f"PDP: {feat_choice}")
                        st.plotly_chart(fig, use_container_width=True)

    # ─── 📢 KAP ─────────────────────────────────────────────
    with tKAP:
        st.subheader("📢 KAP Bildirimleri")
        if st.button("🔄 KAP Verilerini Çek", use_container_width=True):
            with st.spinner("KAP API sorgulanıyor..."):
                for nm in names:
                    get_kap().fetch(nm)
                st.success("✅ Tamamlandı")
        kap_rows = []
        for nm in names:
            feats = get_kap().get_features(nm)
            items = get_kap().fetch(nm)
            kap_rows.append({"Hisse": nm, "Bildirim": len(items),
                             "Önem Skoru": round(float(feats[1]), 2),
                             "Son Haber": items[0]["title"][:60] if items else "-"})
        st.dataframe(pd.DataFrame(kap_rows), use_container_width=True, hide_index=True)
        st.caption("KAP bildirimleri şu an overlay olarak modele eklenmez; bilgi amaçlıdır.")

    # ─── 🔮 TAHMİN ──────────────────────────────────────────
    with tFCAST:
        st.subheader("🔮 Fiyat Tahminleri")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            horizon = st.select_slider("Tahmin ufku (gün)",
                                        options=[5, 10, 20, 40, 60],
                                        value=HORIZON, key="fc_h")
        with fc2:
            sort_by = st.selectbox("Sıralama",
                                    ["Güven", "Değişim %", "P(Yukarı)", "Belirsizlik"],
                                    key="fc_sort")
        with fc3:
            if st.button("💾 Tahminleri Kaydet", use_container_width=True):
                fc_rows_ = forecast_all(bot, names, dfs, prices, dec, rp, horizon)
                for r in fc_rows_:
                    db_forecast_add(r["_obj"])
                st.success(f"✅ {len(fc_rows_)} tahmin kaydedildi")
        fc_rows = forecast_all(bot, names, dfs, prices, dec, rp, horizon)
        if not fc_rows:
            st.warning("Tahmin üretilemedi")
        else:
            if sort_by == "Güven":
                fc_rows.sort(key=lambda x: -x["Güven %"])
            elif sort_by == "Değişim %":
                fc_rows.sort(key=lambda x: -x["Değişim %"])
            elif sort_by == "P(Yukarı)":
                fc_rows.sort(key=lambda x: -x["P(Yukarı) %"])
            elif sort_by == "Belirsizlik":
                fc_rows.sort(key=lambda x: x["Belirsizlik"])
            df_show = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                                     for r in fc_rows])
            st.dataframe(df_show, use_container_width=True, hide_index=True)
            st.divider()
            st.markdown("### 🏆 En Yüksek Potansiyelli 5")
            top5 = fc_rows[:5]
            fig_top = go.Figure()
            for r in top5:
                f = r["_obj"]
                x_ = [0, f.horizon_days]
                y_base = [f.current_price, f.base_price]
                fig_top.add_trace(go.Scatter(x=x_, y=y_base, mode="lines+markers",
                                              name=f"{f.stock} ({f.expected_return_pct:+.1f}%)",
                                              line=dict(width=2)))
            fig_top.update_layout(height=400, template="plotly_dark",
                                    title="Tahmin Karşılaştırması",
                                    xaxis_title="Gün", yaxis_title="Fiyat (₺)")
            st.plotly_chart(fig_top, use_container_width=True)
            st.divider()
            st.markdown("### 🎯 Detaylı Tahmin")
            sec_fc = st.selectbox("Hisse seç", names, key="fc_sel")
            row_fc = next((r for r in fc_rows if r["Hisse"] == sec_fc), None)
            if row_fc:
                f = row_fc["_obj"]
                st.markdown(forecast_to_text(f))
                st.divider()
                vc1, vc2 = st.columns([2, 1])
                with vc1:
                    st.plotly_chart(plot_forecast(f, dfs[sec_fc]), use_container_width=True)
                with vc2:
                    st.plotly_chart(plot_probability_gauge(f), use_container_width=True)
                    st.plotly_chart(plot_scenarios_bar(f), use_container_width=True)
                st.markdown("### 📊 Tahmin Kalitesi")
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Güven", f"%{f.confidence*100:.0f}",
                          "yüksek" if f.confidence > 0.6 else "orta" if f.confidence > 0.3 else "düşük")
                q2.metric("Belirsizlik", f"{f.uncertainty:.3f}",
                          "düşük" if f.uncertainty < 0.15 else "yüksek",
                          delta_color="inverse" if f.uncertainty > 0.15 else "normal")
                q3.metric("ATR%",
                          f"%{(safe_float(dfs[sec_fc]['ATR'].iloc[-1]) / prices[sec_fc]) * 100:.2f}")
                q4.metric("Rejim ×", f"{f.regime_adj:.2f}")
        st.divider()
        st.markdown("### 📜 Geçmiş Tahminler")
        hist_sec = st.selectbox("Hisse (geçmiş)", names, key="fc_hist")
        hist = db_forecast_history(hist_sec, limit=30)
        if hist:
            hdf = pd.DataFrame(hist)
            hdf.columns = ["Zaman", "Şu an", "Hedef", "Beklenen %", "Güven"]
            st.dataframe(hdf, use_container_width=True, hide_index=True)
        else:
            st.info("Bu hisse için kayıtlı tahmin yok")
        st.divider()
        st.markdown("### 📋 Tahmin Karnesi (Gerçekleşen vs Tahmin)")
        if st.button("📋 Karne Hesapla", use_container_width=True):
            sc_df = forecast_scorecard(dfs, limit=600)
            st.session_state["_fcast_score"] = sc_df
        sc_df = st.session_state.get("_fcast_score")
        if sc_df is not None and len(sc_df) > 0:
            hit = sc_df["Yön ✓"].mean() * 100
            mae = sc_df["Hata %"].abs().mean()
            s1, s2, s3 = st.columns(3)
            s1.metric("Yön isabet", f"%{hit:.1f}")
            s2.metric("Ort. |Hata|", f"%{mae:.2f}")
            s3.metric("Örnek", len(sc_df))
            st.dataframe(sc_df.tail(30), use_container_width=True, hide_index=True)
        elif sc_df is not None:
            st.info("Henüz karşılaştırılabilir tahmin yok (ufuk dolmamış).")

    # ─── 🧠 META ────────────────────────────────────────────
    with tMETA:
        st.subheader("🧠 Meta-Learning (HMM + EWC + Reptile)")
        # HMM
        st.markdown("### 🔮 HMM Rejim Tespiti")
        h = get_hmm()
        hm1, hm2, hm3 = st.columns(3)
        hm1.metric("Durum", "🟢 Eğitildi" if h.fitted else "🔴 Eğitilmedi")
        hm2.metric("İterasyon", h.n_iter)
        hm3.metric("Fit tarihi", h.fit_date or "-")
        if h.fitted:
            fig_hmm = go.Figure(go.Heatmap(z=h.A, x=h.names, y=h.names,
                                            colorscale="Blues", zmin=0, zmax=1))
            fig_hmm.update_layout(height=300, template="plotly_dark",
                                   title="Rejim Geçiş Matrisi")
            st.plotly_chart(fig_hmm, use_container_width=True)
            reg_df = pd.DataFrame({
                "Rejim": h.names,
                "Ort. Getiri %": h.mu * 100,
                "Volatilite %": h.sigma * 100,
            })
            st.dataframe(reg_df, use_container_width=True, hide_index=True)
            # Aktif rejim
            if sec in df_recent_returns:
                nm_reg, prob = h.predict_name(df_recent_returns[sec])
                if nm_reg:
                    st.info(f"📊 {sec} anlık rejim: **{nm_reg}** (olasılık %{prob*100:.0f})")
        st.divider()
        # EWC
        st.markdown("### 🛡️ EWC (Sürekli Öğrenme)")
        em1, em2, em3 = st.columns(3)
        em1.metric("Durum", "🟢 Konsolide" if bot.nn.ewc_consolidated else "🔴 Bekliyor")
        em2.metric("Fisher Ağ", len(bot.nn.ewc.fisher))
        em3.metric("Lambda", f"{bot.nn.ewc.lam:.1f}")
        st.caption("Bot yeni öğrenirken eski bilgiyi unutmaz (Elastic Weight Consolidation).")
        st.divider()
        # Reptile
        st.markdown("### ⚡ Reptile (Meta-Öğrenme)")
        rp_m = bot.nn.reptile
        rm1, rm2, rm3 = st.columns(3)
        rm1.metric("Inner LR", f"{rp_m.inner_lr:.4f}")
        rm2.metric("Epsilon", f"{rp_m.eps:.2f}")
        rm3.metric("Tur geçmişi", len(rp_m.history))
        if rp_m.history:
            hist_df = pd.DataFrame(rp_m.history)
            hist_df["before"] = hist_df["before"].astype(float)
            hist_df["after"] = hist_df["after"].astype(float)
            fig_maml = go.Figure()
            fig_maml.add_trace(go.Scatter(x=hist_df.index, y=hist_df["before"],
                                            mode="lines+markers", name="Önce"))
            fig_maml.add_trace(go.Scatter(x=hist_df.index, y=hist_df["after"],
                                            mode="lines+markers", name="Sonra"))
            fig_maml.update_layout(height=300, template="plotly_dark",
                                    title="Reptile Meta-Eğitim (kabul edilen turlar)")
            st.plotly_chart(fig_maml, use_container_width=True)
        if st.button("🧪 Reptile Meta-Train (2 epoch)", use_container_width=True) and datasets:
            merged = merge_ds(datasets)
            sp_ = split_ds(merged) if merged else None
            if sp_:
                with st.spinner("Meta-eğitim..."):
                    hist = bot.nn.reptile.meta_train(bot.nn, sp_, n_epochs=2, seed=bot.nn.evo)
                    if hist:
                        st.success(f"✅ {len(hist)} epoch")
                    bot.reeval(rp)
                    save_state(bot, rp)

    # ─── 💾 DB ──────────────────────────────────────────────
    with tDB:
        tr, eq2, tn = db_trades(200), db_eq(500), db_train(50)
        d1, d2, d3 = st.columns(3)
        d1.metric("İşlem", len(tr))
        d2.metric("Equity", len(eq2))
        d3.metric("Eğitim", len(tn))
        st.markdown("### 📥 Dışa Aktar")
        cc1, cc2, cc3 = st.columns(3)
        if tr:
            tdf = pd.DataFrame(tr)
            cc1.download_button("📄 İşlemler CSV", tdf.to_csv(index=False).encode("utf-8"),
                                 f"trades_{AKTIF_SEKTOR}.csv", "text/csv")
        if eq2:
            edf = pd.DataFrame(eq2)
            cc2.download_button("📈 Equity CSV", edf.to_csv(index=False).encode("utf-8"),
                                 f"equity_{AKTIF_SEKTOR}.csv", "text/csv")
        if tn:
            tndf = pd.DataFrame(tn)
            cc3.download_button("🧠 Eğitim CSV", tndf.to_csv(index=False).encode("utf-8"),
                                 f"train_{AKTIF_SEKTOR}.csv", "text/csv")
        st.divider()
        if tr:
            st.dataframe(pd.DataFrame(tr).head(50), use_container_width=True, hide_index=True)
        else:
            st.info("İşlem kaydı yok")

    # ─── 💬 LOG ─────────────────────────────────────────────
    with tCH:
        if st.session_state.log:
            for line in reversed(st.session_state.log[-30:]):
                st.text(line)
        else:
            st.info("Log yok")

    # Alt bilgi
    st.divider()
    st.caption(f"{BOT_NAME} {BOT_VERSION} · Bot #{BOT_ID} · {AKTIF_SEKTOR} · "
               f"{now:%Y-%m-%d %H:%M:%S} | Borsa: "
               f"{'🟢 AÇIK' if borsa_acik() else '🔴 KAPALI'}")
    st.caption("⚠️ Kâğıt işlem simülasyonu - yatırım tavsiyesi değildir")


# ════════════════════════════════════════════════════════════
# AGGREGATOR MODU + ANA ÇAĞRI
# ════════════════════════════════════════════════════════════
if os.environ.get("MODE") == "aggregator":
    log.info("Aggregator modu aktif")
    while True:
        time.sleep(3600)
        try:
            get_federated().aggregate()
        except Exception as e:
            log.warning(f"agg: {e}")
else:
    if not os.environ.get("THYAO_NO_UI"):
        main()
