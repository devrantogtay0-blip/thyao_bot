import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh

from ai_brain import (
    BILGI, bilgi_ara, guru_score, guru_details,
    NeuralNet, PrioritizedReplay, EnsemblePredictor, PredictionTracker,
    MetaEvolver, SelfDataGenerator, Bot, Arena, extract_features, GURUS,
)

st.set_page_config(page_title="THYAO AI v9", page_icon="🧠", layout="wide")
st.title("🧠 THYAO AI Trader v9")
st.caption("Adam + Dropout + Prioritized Replay + Ensemble + Meta-Evolution + 25 Usta")

TR = ZoneInfo("Europe/Istanbul")
def now_tr(): return datetime.now(TR)
def market_open():
    n = now_tr()
    if n.weekday() >= 5: return False
    return dtime(9, 55) <= n.time() <= dtime(18, 10)

HISSELER = {
    "THYAO": "THYAO.IS", "GARAN": "GARAN.IS", "ASELS": "ASELS.IS",
    "AKBNK": "AKBNK.IS", "EREGL": "EREGL.IS", "TUPRS": "TUPRS.IS",
    "SISE": "SISE.IS", "KCHOL": "KCHOL.IS",
}

qp = st.query_params
def qget(k, d):
    try:
        v = qp.get(k)
        return v if v is not None else d
    except Exception:
        return d
def qset(k, v):
    try: st.query_params[k] = str(v)
    except Exception: pass

hisse_p = qget("hisse", "THYAO")
secili_hisse = hisse_p if hisse_p in HISSELER else "THYAO"
auto_on = qget("auto", "0") == "1"
test_mode = qget("test", "0") == "1"
try: interval_min = int(qget("int", "2"))
except Exception: interval_min = 2
try: refresh_sn = int(qget("rs", "30"))
except Exception: refresh_sn = 30

# ====== VERI ======
@st.cache_data(ttl=30)
def get_data(sym, period="2y"):
    try:
        df = yf.Ticker(sym).history(period=period, interval="1d")
        if df.empty: return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df
    except Exception:
        return None

@st.cache_data(ttl=15)
def get_canli(sym):
    try:
        df = yf.Ticker(sym).history(period="1d", interval="1m")
        if df.empty: return None
        return {"price": float(df["Close"].iloc[-1]), "open": float(df["Open"].iloc[0]),
                "high": float(df["High"].max()), "low": float(df["Low"].min()),
                "volume": int(df["Volume"].sum()),
                "change_pct": float((df["Close"].iloc[-1] / df["Open"].iloc[0] - 1) * 100)}
    except Exception:
        return None

def add_ind(df):
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

# ====== SESSION ======
if "arena" not in st.session_state:
    st.session_state.arena = Arena(100000)
if "meta" not in st.session_state:
    st.session_state.meta = MetaEvolver(10)
if "predictor" not in st.session_state:
    st.session_state.predictor = EnsemblePredictor()
if "tracker" not in st.session_state:
    st.session_state.tracker = PredictionTracker()
if "selfdata" not in st.session_state:
    st.session_state.selfdata = SelfDataGenerator()
if "log" not in st.session_state:
    st.session_state.log = []
if "last_meta" not in st.session_state:
    st.session_state.last_meta = 0
if "chat" not in st.session_state:
    st.session_state.chat = []

arena = st.session_state.arena
meta = st.session_state.meta
predictor = st.session_state.predictor
tracker = st.session_state.tracker
selfdata = st.session_state.selfdata

if auto_on:
    st_autorefresh(interval=refresh_sn * 1000, key="rf")

# ====== VERI CEK ======
sym = HISSELER[secili_hisse]
df = get_data(sym)
if df is None or df.empty:
    st.error("Veri cekilemedi")
    st.stop()
df = add_ind(df)
if df.empty:
    st.error("Yetersiz veri")
    st.stop()

last = df.iloc[-1]
canli = get_canli(sym)
price = canli["price"] if canli else float(last["Close"])
date = last["Date"]

# ====== TAHMIN KAYDET & DOGRULA ======
tracker.dogrula(date.date() if hasattr(date, "date") else date, price)
pred_result = predictor.predict(df, 5)
if pred_result:
    ensemble, preds, egim = pred_result
    predictor.update(price, preds)

# ====== BOT KARARLARI ======
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
        b.train(3, 32)
    arena.counter += 1
    if arena.counter >= 50:
        arena.evolve(price)
        arena.counter = 0
        st.session_state.log.append(f"{n.strftime('%H:%M:%S')} - 🧬 Arena Gen {arena.generation}")

    # Meta evrim
    total_tr = sum(len(b.trades) for b in arena.bots)
    if total_tr > 0 and total_tr // 100 > st.session_state.last_meta:
        st.session_state.last_meta = total_tr // 100
        best_dna, fit = meta.evolve(df)
        champion.risk_pct = best_dna["risk_pct"]
        champion.min_conf = best_dna["min_conf"]
        champion.sl_pct = best_dna["sl_pct"]
        champion.tp_pct = best_dna["tp_pct"]
        sd = selfdata.generate(df, n=80, seed=st.session_state.last_meta)
        st.session_state.log.append(f"{n.strftime('%H:%M:%S')} - 🔬 Meta Gen {meta.gen} | {len(sd)} veri uretildi")

    if auto_msgs:
        st.session_state.log.append(f"{n.strftime('%H:%M:%S')} - " + " | ".join(auto_msgs))

# ====== SIDEBAR ======
with st.sidebar:
    st.header("⚙️ Panel")
    hs = st.selectbox("Hisse", list(HISSELER.keys()), index=list(HISSELER.keys()).index(secili_hisse))
    if hs != secili_hisse:
        qset("hisse", hs); st.rerun()

    st.metric(secili_hisse, f"{price:.2f}", f"{canli['change_pct']:+.2f}%" if canli else "")
    st.metric("Sampiyon", f"{champion.name}", f"{champion.value(price):,.0f} TL")

    st.divider()
    rs = st.slider("Yenile (sn)", 10, 300, refresh_sn)
    if rs != refresh_sn: qset("rs", rs); st.rerun()
    aw = st.toggle("Otomatik", value=auto_on)
    if aw != auto_on: qset("auto", "1" if aw else "0"); st.rerun()
    tw = st.toggle("Test Modu", value=test_mode)
    if tw != test_mode: qset("test", "1" if tw else "0"); st.rerun()

    st.divider()
    st.write(f"🧬 Arena Gen: **{arena.generation}**")
    st.write(f"🔬 Meta Gen: **{meta.gen}**")
    st.write(f"🔮 Tahmin: **{tracker.stats()['toplam']}**")

    if st.button("🔄 Sifirla", use_container_width=True):
        st.session_state.arena = Arena(100000)
        st.session_state.meta = MetaEvolver(10)
        st.session_state.predictor = EnsemblePredictor()
        st.session_state.tracker = PredictionTracker()
        st.session_state.selfdata = SelfDataGenerator()
        st.session_state.log = []
        st.rerun()

# ====== UST ======
c1, c2, c3, c4 = st.columns(4)
c1.metric(secili_hisse, f"{price:.2f}", f"{canli['change_pct']:+.2f}%" if canli else "")
c2.metric("RSI", f"{last['RSI']:.1f}")
c3.metric("Arena Gen", arena.generation)
c4.metric("Meta Gen", meta.gen)
if auto_msgs: st.info(" | ".join(auto_msgs[:4]))

# ====== SEKMELER ======
t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
    "🏟️ Arena", "🧠 NN", "🧬 Genetik", "🔬 Meta", "💭 Yansima",
    "🔮 Tahmin", "📊 Ensemble", "📚 Bilgi", "📜 Islemler"
])

with t1:
    st.subheader("🏟️ Multi-Bot Arena")
    st.dataframe(pd.DataFrame(arena.leaderboard(price)), use_container_width=True, hide_index=True)
    st.divider()
    fig = go.Figure()
    for b in arena.bots:
        vals = [b.initial] + [t["value"] for t in b.trades] + [b.value(price)]
        fig.add_trace(go.Scatter(y=vals, name=b.name, mode="lines+markers"))
    fig.update_layout(height=400, template="plotly_dark", xaxis_title="Islem", yaxis_title="TL")
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.subheader("🧠 Neural Network (Adam + Dropout)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mimari", "20-32-16-3")
    c2.metric("Parametre", f"{champion.nn.n_params()}")
    c3.metric("Egitim", len(champion.nn.loss_history))
    if champion.nn.loss_history:
        st.line_chart(pd.DataFrame({"Loss": champion.nn.loss_history}))
    st.divider()
    a, p, f = decisions[champion.name]
    c1, c2, c3 = st.columns(3)
    c1.metric("P(SAT)", f"%{p[0]*100:.1f}")
    c2.metric("P(TUT)", f"%{p[1]*100:.1f}")
    c3.metric("P(AL)", f"%{p[2]*100:.1f}")
    st.write(f"**Karar: {a}**")

with t3:
    st.subheader("🧬 Arena Genetik Evrim")
    if arena.history:
        st.dataframe(pd.DataFrame(arena.history), use_container_width=True, hide_index=True)
    else:
        st.info("50 islemde bir evrim")

with t4:
    st.subheader("🔬 Meta-Evolution")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nesil", meta.gen)
    c2.metric("En Iyi", f"{max(meta.hist) if meta.hist else 0:+.2f}")
    c3.metric("Son", f"{meta.hist[-1] if meta.hist else 0:+.2f}")
    if meta.hall:
        rows = []
        for h in meta.hall:
            d = h["dna"]
            rows.append({"Nesil": h["gen"], "Fitness": f"{h['fit']:+.2f}",
                         "Risk%": f"%{d['risk_pct']*100:.0f}",
                         "SL%": f"{d['sl_pct']:.1f}", "TP%": f"{d['tp_pct']:.1f}",
                         "BB": "✅" if d["use_bb"] else "❌",
                         "ADX": "✅" if d["use_adx"] else "❌"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if meta.hist:
        st.line_chart(pd.DataFrame({"Fitness": meta.hist}))
    st.divider()
    st.subheader("🔬 Self-Data (R-Zero)")
    sd = selfdata.summary()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Toplam", sd["total"])
    c2.metric("AL", sd["AL"]); c3.metric("SAT", sd["SAT"]); c4.metric("TUT", sd["TUT"])
    c5.metric("Ort Getiri", f"%{sd['avg_ret']}")
    if st.button("🔬 Yeni Veri Uret"):
        s = selfdata.generate(df, n=100, seed=np.random.randint(1, 9999))
        st.success(f"{len(s)} senaryo!")
        st.rerun()

with t5:
    st.subheader("💭 Botun Yansimalari")
    if meta.reflections:
        for r in reversed(meta.reflections):
            st.info(r)
    else:
        st.warning("100 islem gerekiyor")

with t6:
    st.subheader("🔮 Fiyat Tahmini + Karne")
    s = tracker.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam", s["toplam"])
    c2.metric("Dogrulanan", s["dogrulanan"])
    c3.metric("MAPE", f"%{s['mape']}")
    c4.metric("Gercekcilik", f"%{s['gercekcilik']}")
    g = s["gercekcilik"]
    if g >= 80: st.success(f"🎯 {s['yorum']}")
    elif g >= 60: st.info(f"👍 {s['yorum']}")
    elif g >= 40: st.warning(f"⚠️ {s['yorum']}")
    else: st.error(f"❌ {s['yorum']}")

    st.divider()
    if pred_result:
        ensemble, preds, egim = pred_result
        pdf = pd.DataFrame([{"Model": k.upper(), "Tahmin": f"{v:.2f}",
                             "Fark%": f"{(v/price-1)*100:+.2f}"} for k, v in preds.items()])
        st.dataframe(pdf, use_container_width=True, hide_index=True)
        st.metric("Ensemble", f"{ensemble:.2f}", f"{(ensemble/price-1)*100:+.2f}%")
        gun = st.slider("Kac gun?", 1, 15, 5)
        if st.button("📌 Kaydet"):
            hedef = (date.date() if hasattr(date, "date") else date) + timedelta(days=gun)
            tracker.add(str(date.date() if hasattr(date, "date") else date), hedef, ensemble, price)
            st.success(f"{hedef} icin {ensemble:.2f} kaydedildi")
            st.rerun()
    if tracker.predictions:
        st.divider()
        rows = []
        for p in tracker.predictions[-20:]:
            rows.append({"Hedef": str(p["hedef"]), "Tahmin": f"{p['tahmin']:.2f}",
                         "Giris": f"{p['giris']:.2f}",
                         "Gercek": f"{p['gercek']:.2f}" if p["gercek"] else "-",
                         "Hata%": f"{p['hata']:.2f}" if p["hata"] is not None else "-",
                         "Yon": "✅" if p["yon_ok"] else ("❌" if p["yon_ok"] is False else "-")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with t7:
    st.subheader("📊 Ensemble Predictor (6 Model)")
    st.dataframe(pd.DataFrame(predictor.stats()), use_container_width=True, hide_index=True)
    st.divider()
    for m in predictor.stats():
        st.progress(m["Agirlik"], text=f"{m['Model']}: %{m['Agirlik']*100:.1f} (dogruluk %{m['Dogruluk']})")

with t8:
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

with t9:
    st.subheader(f"📜 {champion.name} Islemleri")
    if champion.trades:
        tdf = pd.DataFrame(champion.trades)
        tdf["date"] = pd.to_datetime(tdf["date"]).dt.strftime("%Y-%m-%d %H:%M")
        cols = [c for c in ["date", "action", "price", "qty", "value", "pnl", "reason"] if c in tdf.columns]
        st.dataframe(tdf[cols], use_container_width=True, hide_index=True)
    else:
        st.info("Islem yok")
    st.divider()
    st.subheader("📡 Log")
    for line in reversed(st.session_state.log[-25:]):
        st.text(line)

st.divider()
st.caption(f"🕐 {now_tr().strftime('%Y-%m-%d %H:%M:%S')} - Borsa: {'ACIK' if market_open() else 'KAPALI'}")
