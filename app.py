import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh
import requests

from ai_brain import (BILGI, bilgi_ara, guru_score, guru_details, EnsemblePredictor,
                       PredictionTracker, MetaEvolver, SelfDataGenerator, Bot, Arena)
from self_improve import SelfImprover

st.set_page_config(page_title="THYAO AI v10", page_icon="🧠", layout="wide")
st.title("🧠 THYAO AI Trader v10")
st.caption("Sinir Agi + Arena + Meta-Evrim + Yari-Otonom Oneri + 25 Usta")

TR = ZoneInfo("Europe/Istanbul")
def now_tr(): return datetime.now(TR)
def market_open():
    n = now_tr()
    if n.weekday() >= 5: return False
    return dtime(9, 55) <= n.time() <= dtime(18, 10)

HISSELER = {"THYAO": "THYAO.IS", "GARAN": "GARAN.IS", "ASELS": "ASELS.IS",
            "AKBNK": "AKBNK.IS", "EREGL": "EREGL.IS", "TUPRS": "TUPRS.IS",
            "SISE": "SISE.IS", "KCHOL": "KCHOL.IS"}

qp = st.query_params
def qget(k, d):
    try:
        v = qp.get(k); return v if v is not None else d
    except Exception: return d
def qset(k, v):
    try: st.query_params[k] = str(v)
    except Exception: pass

hp = qget("hisse", "THYAO")
secili = hp if hp in HISSELER else "THYAO"
auto_on = qget("auto", "0") == "1"
test_mode = qget("test", "0") == "1"
try: rsn = int(qget("rs", "30"))
except Exception: rsn = 30

@st.cache_data(ttl=30)
def get_data(sym, period="2y"):
    try:
        df = yf.Ticker(sym).history(period=period, interval="1d")
        if df.empty: return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df
    except Exception: return None

@st.cache_data(ttl=15)
def get_canli(sym):
    try:
        df = yf.Ticker(sym).history(period="1d", interval="1m")
        if df.empty: return None
        return {"price": float(df["Close"].iloc[-1]), "open": float(df["Open"].iloc[0]),
                "high": float(df["High"].max()), "low": float(df["Low"].min()),
                "volume": int(df["Volume"].sum()),
                "change_pct": float((df["Close"].iloc[-1] / df["Open"].iloc[0] - 1) * 100)}
    except Exception: return None

def add_ind(df):
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    d = c.diff()
    g = d.where(d > 0, 0).rolling(14).mean(); ls = (-d.where(d < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + g / ls))
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]
    df["SMA20"] = c.rolling(20).mean(); df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["BB_mid"] = df["SMA20"]; df["BB_std"] = c.rolling(20).std()
    df["BB_up"] = df["BB_mid"] + 2 * df["BB_std"]; df["BB_dn"] = df["BB_mid"] - 2 * df["BB_std"]
    df["BB_width"] = (df["BB_up"] - df["BB_dn"]) / df["BB_mid"]
    lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
    df["K"] = 100 * (c - lo14) / (hi14 - lo14); df["D"] = df["K"].rolling(3).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["WILLR"] = -100 * (hi14 - c) / (hi14 - lo14)
    up = h.diff(); dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr14 = tr.rolling(14).sum()
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(14).sum() / tr14
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(14).sum() / tr14
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["ADX"] = dx.rolling(14).mean()
    df["Vol_SMA"] = v.rolling(20).mean(); df["Vol_ratio"] = v / df["Vol_SMA"]
    df["Returns"] = c.pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252)
    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["EMA21"] = c.ewm(span=21, adjust=False).mean()
    df["Mom10"] = c.pct_change(10); df["Mom30"] = c.pct_change(30)
    df["HL_ratio"] = (h - l) / c
    df["Gap"] = (df["Open"] - c.shift()) / c.shift()
    return df.dropna().reset_index(drop=True)

# SESSION
if "arena" not in st.session_state: st.session_state.arena = Arena(100000)
if "meta" not in st.session_state: st.session_state.meta = MetaEvolver(10)
if "predictor" not in st.session_state: st.session_state.predictor = EnsemblePredictor()
if "tracker" not in st.session_state: st.session_state.tracker = PredictionTracker()
if "selfdata" not in st.session_state: st.session_state.selfdata = SelfDataGenerator()
if "log" not in st.session_state: st.session_state.log = []
if "last_meta" not in st.session_state: st.session_state.last_meta = 0

arena = st.session_state.arena; meta = st.session_state.meta
predictor = st.session_state.predictor; tracker = st.session_state.tracker
selfdata = st.session_state.selfdata

if auto_on: st_autorefresh(interval=rsn * 1000, key="rf")

sym = HISSELER[secili]
df = get_data(sym)
if df is None: st.error("Veri yok"); st.stop()
df = add_ind(df)
if df.empty: st.error("Yetersiz veri"); st.stop()
last = df.iloc[-1]; canli = get_canli(sym)
price = canli["price"] if canli else float(last["Close"])
date = last["Date"]

tracker.dogrula(date.date() if hasattr(date, "date") else date, price)
pred_result = predictor.predict(df, 5)
if pred_result:
    ensemble, preds, egim = pred_result
    predictor.update(price, preds)

decisions = {}
for b in arena.bots:
    b.last_price = price
    a, p, f = b.decide(last, price)
    decisions[b.name] = (a, p, f)
champion = arena.best(price)

auto_msgs = []
if auto_on and (market_open() or test_mode):
    n = now_tr()
    for b in arena.bots:
        a, p, f = decisions[b.name]
        if b.execute(a, price, date, f, reason=f"AI {a}"):
            auto_msgs.append(f"{b.name}:{a}")
    for b in arena.bots: b.train(3, 32)
    arena.counter += 1
    if arena.counter >= 50:
        arena.evolve(price); arena.counter = 0
        st.session_state.log.append(f"{n.strftime('%H:%M:%S')} - Arena Gen {arena.generation}")
    total_tr = sum(len(b.trades) for b in arena.bots)
    if total_tr > 0 and total_tr // 100 > st.session_state.last_meta:
        st.session_state.last_meta = total_tr // 100
        bd, fit = meta.evolve(df)
        champion.risk_pct = bd["risk_pct"]; champion.min_conf = bd["min_conf"]
        champion.sl_pct = bd["sl_pct"]; champion.tp_pct = bd["tp_pct"]
        selfdata.generate(df, n=80, seed=st.session_state.last_meta)
        st.session_state.log.append(f"{n.strftime('%H:%M:%S')} - Meta Gen {meta.gen}")
    if auto_msgs:
        st.session_state.log.append(f"{n.strftime('%H:%M:%S')} - " + " | ".join(auto_msgs))

# SIDEBAR
with st.sidebar:
    st.header("Panel")
    hs = st.selectbox("Hisse", list(HISSELER.keys()), index=list(HISSELER.keys()).index(secili))
    if hs != secili: qset("hisse", hs); st.rerun()
    st.metric(secili, f"{price:.2f}", f"{canli['change_pct']:+.2f}%" if canli else "")
    st.metric("Sampiyon", champion.name, f"{champion.value(price):,.0f} TL")
    st.divider()
    rs = st.slider("Yenile (sn)", 10, 300, rsn)
    if rs != rsn: qset("rs", rs); st.rerun()
    aw = st.toggle("Otomatik", value=auto_on)
    if aw != auto_on: qset("auto", "1" if aw else "0"); st.rerun()
    tw = st.toggle("Test Modu", value=test_mode)
    if tw != test_mode: qset("test", "1" if tw else "0"); st.rerun()
    st.divider()
    st.write(f"Arena Gen: **{arena.generation}**")
    st.write(f"Meta Gen: **{meta.gen}**")
    st.write(f"Tahmin: **{tracker.stats()['toplam']}**")
    if st.button("Sifirla", use_container_width=True):
        for k in ["arena","meta","predictor","tracker","selfdata","log"]:
            st.session_state.pop(k, None)
        st.rerun()

c1, c2, c3, c4 = st.columns(4)
c1.metric(secili, f"{price:.2f}")
c2.metric("RSI", f"{last['RSI']:.1f}")
c3.metric("Arena Gen", arena.generation)
c4.metric("Meta Gen", meta.gen)
if auto_msgs: st.info(" | ".join(auto_msgs[:4]))

t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs([
    "🏟️ Arena", "🧠 NN", "🧬 Genetik", "🔬 Meta", "💭 Yansima",
    "🔮 Tahmin", "📊 Ensemble", "📚 Bilgi", "📜 Islemler", "📋 Oneriler"
])

with t1:
    st.subheader("Multi-Bot Arena")
    st.dataframe(pd.DataFrame(arena.leaderboard(price)), use_container_width=True, hide_index=True)
    fig = go.Figure()
    for b in arena.bots:
        vals = [b.initial] + [t["value"] for t in b.trades] + [b.value(price)]
        fig.add_trace(go.Scatter(y=vals, name=b.name, mode="lines+markers"))
    fig.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.subheader("Neural Network")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mimari", "20-32-16-3")
    c2.metric("Parametre", f"{champion.nn.n_params()}")
    c3.metric("Egitim", len(champion.nn.loss_history))
    if champion.nn.loss_history:
        st.line_chart(pd.DataFrame({"Loss": champion.nn.loss_history}))
    a, p, f = decisions[champion.name]
    c1, c2, c3 = st.columns(3)
    c1.metric("P(SAT)", f"%{p[0]*100:.1f}")
    c2.metric("P(TUT)", f"%{p[1]*100:.1f}")
    c3.metric("P(AL)", f"%{p[2]*100:.1f}")
    st.write(f"Karar: **{a}**")

with t3:
    if arena.history: st.dataframe(pd.DataFrame(arena.history), use_container_width=True, hide_index=True)
    else: st.info("50 islemde bir evrim")

with t4:
    c1, c2, c3 = st.columns(3)
    c1.metric("Nesil", meta.gen)
    c2.metric("En Iyi", f"{max(meta.hist) if meta.hist else 0:+.2f}")
    c3.metric("Son", f"{meta.hist[-1] if meta.hist else 0:+.2f}")
    if meta.hall:
        rows = []
        for h in meta.hall:
            d = h["dna"]
            rows.append({"Nesil": h["gen"], "Fit": f"{h['fit']:+.2f}",
                         "Risk%": f"%{d['risk_pct']*100:.0f}",
                         "SL%": f"{d['sl_pct']:.1f}", "TP%": f"{d['tp_pct']:.1f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if meta.hist: st.line_chart(pd.DataFrame({"Fit": meta.hist}))

with t5:
    if meta.reflections:
        for r in reversed(meta.reflections): st.info(r)
    else: st.warning("100 islem gerekiyor")

with t6:
    s = tracker.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam", s["toplam"]); c2.metric("Dogrulanan", s["dogrulanan"])
    c3.metric("MAPE", f"%{s['mape']}"); c4.metric("Gercekcilik", f"%{s['gercekcilik']}")
    g = s["gercekcilik"]
    if g >= 80: st.success(s["yorum"])
    elif g >= 60: st.info(s["yorum"])
    elif g >= 40: st.warning(s["yorum"])
    else: st.error(s["yorum"])
    if pred_result:
        ensemble, preds, egim = pred_result
        st.dataframe(pd.DataFrame([{"Model": k.upper(), "Tahmin": f"{v:.2f}",
                                     "Fark%": f"{(v/price-1)*100:+.2f}"} for k, v in preds.items()]),
                     use_container_width=True, hide_index=True)
        st.metric("Ensemble", f"{ensemble:.2f}", f"{(ensemble/price-1)*100:+.2f}%")
        if st.button("Kaydet"):
            hedef = (date.date() if hasattr(date, "date") else date) + timedelta(days=5)
            tracker.add(str(date.date() if hasattr(date, "date") else date), hedef, ensemble, price)
            st.success("Kaydedildi"); st.rerun()

with t7:
    st.dataframe(pd.DataFrame(predictor.stats()), use_container_width=True, hide_index=True)

with t8:
    arama = st.text_input("Ara", placeholder="RSI, temettu...")
    if arama:
        for s in bilgi_ara(arama):
            with st.expander(f"{s['Konu']} ({s['Kategori']})"): st.write(s["Aciklama"])
    else:
        for k, m in BILGI.items():
            with st.expander(f"{k} ({len(m)})"):
                for b, i in m.items(): st.markdown(f"**{b}**: {i}")

with t9:
    if champion.trades:
        tdf = pd.DataFrame(champion.trades)
        tdf["date"] = pd.to_datetime(tdf["date"]).dt.strftime("%Y-%m-%d %H:%M")
        cols = [c for c in ["date","action","price","qty","value","pnl","reason"] if c in tdf.columns]
        st.dataframe(tdf[cols], use_container_width=True, hide_index=True)
    else: st.info("Islem yok")
    for line in reversed(st.session_state.log[-15:]): st.text(line)

with t10:
    st.subheader("Yari-Otonom Oneri Sistemi")
    st.caption("Bot oneri uretir, sen onaylarsin.")
    try:
        url = "https://raw.githubusercontent.com/devrantogtay0-blip/thyao_bot/main/state.json"
        try:
            r = requests.get(url, timeout=5)
            sdata = r.json() if r.status_code == 200 else {}
        except Exception: sdata = {}
        imp = SelfImprover(sdata)
        if st.button("Simdi Analiz Et", use_container_width=True):
            new = imp.analyze()
            st.success(f"{len(new)} yeni oneri") if new else st.info("Yeni oneri yok (20 islem gerekli)")
            st.rerun()
        pending = imp.pending()
        st.subheader(f"Bekleyen ({len(pending)})")
        for p in pending:
            tip = "Kod" if p.get("tip") == "kod" else "Param"
            with st.expander(f"[{tip}] {p['baslik']}"):
                st.write(f"**Oneri:** {p['oneri']}")
                st.write(f"**Sebep:** {p['sebep']}")
                if p.get("tip") == "param":
                    c1, c2 = st.columns(2)
                    c1.metric("Eski", p.get("eski", "-")); c2.metric("Yeni", p.get("yeni", "-"))
                if p.get("kod_ornek"):
                    st.code(p["kod_ornek"], language="python")
                c1, c2 = st.columns(2)
                if c1.button(f"Onayla #{p['id']}", key=f"a{p['id']}", use_container_width=True):
                    imp.approve(p["id"]); st.rerun()
                if c2.button(f"Reddet #{p['id']}", key=f"r{p['id']}", use_container_width=True):
                    imp.reject(p["id"]); st.rerun()
        approved = imp.approved()
        if approved:
            st.subheader(f"Onaylanan ({len(approved)})")
            rows = [{"ID": p["id"], "Tip": p.get("tip"), "Baslik": p["baslik"],
                     "Tarih": p.get("onay_tarih", "-")} for p in approved[-10:]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("GitHub Actions sonraki calismada uygular.")
    except Exception as e:
        st.error(f"Hata: {e}")

st.divider()
st.caption(f"{now_tr().strftime('%Y-%m-%d %H:%M:%S')} - Borsa: {'ACIK' if market_open() else 'KAPALI'}")
