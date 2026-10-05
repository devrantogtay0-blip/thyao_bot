import time
import pickle
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh
import requests

from ai_brain import (BILGI, bilgi_ara, guru_score, guru_details, EnsemblePredictor,
                      PredictionTracker, MetaEvolver, SelfDataGenerator, Bot, Arena, BOT_VERSION)
from self_improve import SelfImprover
from indicators import add_ind

try:
    from analytics import perf, trade_stats, daily_curve, bootstrap_mean, perm_pvalue
    HAS_ANALYTICS = True
except ImportError:
    HAS_ANALYTICS = False

st.set_page_config(page_title="THYAO AI v11", page_icon="🧠", layout="wide")
st.title("🧠 THYAO AI Trader v11")
st.caption("Sinir Agi + Arena + Meta-Evrim + Yari-Otonom Oneri + 25 Usta  |  "
           "Sanal (paper) islem - yatirim tavsiyesi degildir")

TR = ZoneInfo("Europe/Istanbul")
GH_REPO = "devrantogtay0-blip/thyao_bot"
STATE_URL = f"https://raw.githubusercontent.com/{GH_REPO}/main/state.json"
SAVE_FILE = "books.pkl"


def now_tr():
    return datetime.now(TR)


def market_open():
    n = now_tr()
    return n.weekday() < 5 and dtime(9, 55) <= n.time() <= dtime(18, 10)


HISSELER = {"THYAO": "THYAO.IS", "GARAN": "GARAN.IS", "ASELS": "ASELS.IS",
            "AKBNK": "AKBNK.IS", "EREGL": "EREGL.IS", "TUPRS": "TUPRS.IS",
            "SISE": "SISE.IS", "KCHOL": "KCHOL.IS"}

qp = st.query_params


def qget(k, d):
    try:
        v = qp.get(k)
        return v if v is not None else d
    except Exception:
        return d


def qset(k, v):
    try:
        st.query_params[k] = str(v)
    except Exception:
        pass


hp = qget("hisse", "THYAO")
secili = hp if hp in HISSELER else "THYAO"
try:
    rsn = min(300, max(10, int(qget("rs", "30"))))
except Exception:
    rsn = 30

if "auto_on" not in st.session_state:
    st.session_state.auto_on = qget("auto", "0") == "1"
if "test_mode" not in st.session_state:
    st.session_state.test_mode = qget("test", "0") == "1"


def on_hisse():
    qset("hisse", st.session_state.w_hisse)


def on_rs():
    qset("rs", st.session_state.w_rs)


def on_auto():
    qset("auto", "1" if st.session_state.auto_on else "0")


def on_test():
    qset("test", "1" if st.session_state.test_mode else "0")


auto_on = st.session_state.auto_on
test_mode = st.session_state.test_mode


@st.cache_data(ttl=300, show_spinner=False)
def get_data(sym, period="2y"):
    for _ in range(2):
        try:
            df = yf.Ticker(sym).history(period=period, interval="1d")
            if not df.empty:
                df = df.reset_index()
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                return df
        except Exception:
            time.sleep(0.5)
    return None


@st.cache_data(ttl=15, show_spinner=False)
def get_canli(sym):
    try:
        df = yf.Ticker(sym).history(period="1d", interval="1m")
        if df.empty:
            return None
        o = float(df["Open"].iloc[0])
        c = float(df["Close"].iloc[-1])
        return {"price": c, "open": o, "high": float(df["High"].max()),
                "low": float(df["Low"].min()), "volume": int(df["Volume"].sum()),
                "change_pct": (c / o - 1) * 100 if o else 0.0}
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def get_remote_state():
    try:
        r = requests.get(STATE_URL, timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def gh_conf():
    try:
        tok = st.secrets.get("GITHUB_TOKEN")
    except Exception:
        tok = None
    return {"gh_token": tok, "gh_repo": GH_REPO, "gh_branch": "main"}


def teknik_ozet(last):
    s = []
    r = last["RSI"]
    s.append(("RSI", f"{r:.0f} - " + ("asiri satim" if r < 30 else "asiri alim" if r > 70 else "notr"),
              1 if r < 30 else -1 if r > 70 else 0))
    s.append(("MACD", "pozitif" if last["MACD_hist"] > 0 else "negatif",
              1 if last["MACD_hist"] > 0 else -1))
    s.append(("Trend (SMA50/200)", "yukari" if last["SMA50"] > last["SMA200"] else "asagi",
              1 if last["SMA50"] > last["SMA200"] else -1))
    s.append(("Fiyat / SMA20", "ustunde" if last["Close"] > last["SMA20"] else "altinda",
              1 if last["Close"] > last["SMA20"] else -1))
    k = last["K"]
    s.append(("Stokastik", f"{k:.0f}", 1 if k < 20 else -1 if k > 80 else 0))
    s.append(("ADX", f"{last['ADX']:.0f} - " + ("guclu trend" if last["ADX"] > 25 else "zayif trend"), 0))
    return sum(x[2] for x in s), s


def equity_stats(bot, price):
    vals = [bot.initial] + [t.get("value", bot.initial) for t in bot.trades] + [bot.value(price)]
    arr = np.array(vals, dtype=float)
    peak = np.maximum.accumulate(arr)
    mdd = float(((arr - peak) / peak).min() * 100)
    pnls = [t["pnl"] for t in bot.trades if t.get("pnl") not in (None, 0)]
    win = (sum(1 for p in pnls if p > 0) / len(pnls) * 100) if pnls else 0.0
    return {"getiri": (arr[-1] / bot.initial - 1) * 100, "islem": len(bot.trades),
            "kazanma": win, "mdd": mdd}


def new_book():
    return {"arena": Arena(100000), "meta": MetaEvolver(10), "predictor": EnsemblePredictor(),
            "tracker": PredictionTracker(), "selfdata": SelfDataGenerator(),
            "last_meta": 0, "last_tick": 0.0, "last_pred_key": None,
            "pretrained": False, "pretrain": None}


books = st.session_state.setdefault("books", {})
st.session_state.setdefault("log", [])
book = books.setdefault(secili, new_book())
arena, meta, predictor = book["arena"], book["meta"], book["predictor"]
tracker, selfdata = book["tracker"], book["selfdata"]

if auto_on:
    st_autorefresh(interval=rsn * 1000, key="rf")

sym = HISSELER[secili]
with st.spinner("Veri yukleniyor..."):
    df = get_data(sym)
if df is None:
    st.error("Veri alinamadi (yfinance). Biraz sonra tekrar dene.")
    st.stop()
df = add_ind(df)
if df.empty:
    st.error("Yetersiz veri")
    st.stop()
last = df.iloc[-1]
canli = get_canli(sym)
price = canli["price"] if canli else float(last["Close"])
date = last["Date"]
date_d = date.date() if hasattr(date, "date") else date

if not book.get("pretrained"):
    with st.spinner("Sinir aglari gecmis veriyle on-egitiliyor..."):
        book["pretrain"] = arena.pretrain(df)
    book["pretrained"] = True

tracker.dogrula(date_d, price, df[["Date", "Close"]])
pred_result = predictor.predict(df, 5)
if pred_result:
    ensemble, preds, egim = pred_result
    key = (str(date_d), round(price, 2))
    if book["last_pred_key"] != key:
        predictor.update(price, preds)
        book["last_pred_key"] = key

decisions = {}
for b in arena.bots:
    b.last_price = price
    decisions[b.name] = b.decide(last, price)
champion = arena.best(price)

auto_msgs = []
tick_due = auto_on and (time.time() - book["last_tick"] >= max(rsn - 1, 5))
if tick_due and (market_open() or test_mode):
    book["last_tick"] = time.time()
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
        st.session_state.log.append(f"{n:%H:%M:%S} [{secili}] Arena Gen {arena.generation}")
    total_tr = sum(len(b.trades) for b in arena.bots)
    if total_tr > 0 and total_tr // 100 > book["last_meta"]:
        book["last_meta"] = total_tr // 100
        bd, fit = meta.evolve(df)
        champion.risk_pct = bd["risk_pct"]
        champion.min_conf = bd["min_conf"]
        champion.sl_pct = bd["sl_pct"]
        champion.tp_pct = bd["tp_pct"]
        selfdata.generate(df, n=80, seed=book["last_meta"])
        st.session_state.log.append(f"{n:%H:%M:%S} [{secili}] Meta Gen {meta.gen}")
    if auto_msgs:
        st.session_state.log.append(f"{n:%H:%M:%S} [{secili}] " + " | ".join(auto_msgs))
    st.session_state.log = st.session_state.log[-200:]

with st.sidebar:
    st.header("Panel")
    st.selectbox("Hisse", list(HISSELER.keys()), index=list(HISSELER.keys()).index(secili),
                 key="w_hisse", on_change=on_hisse)
    st.metric(secili, f"{price:.2f}", f"{canli['change_pct']:+.2f}%" if canli else "")
    st.metric("Sampiyon", champion.name, f"{champion.value(price):,.0f} TL")
    st.caption(f"Borsa: {'🟢 ACIK' if market_open() else '🔴 KAPALI'}")
    st.divider()
    st.slider("Yenile (sn)", 10, 300, rsn, key="w_rs", on_change=on_rs)
    st.toggle("Otomatik", key="auto_on", on_change=on_auto)
    st.toggle("Test Modu (borsa kapaliyken de islem)", key="test_mode", on_change=on_test)
    st.divider()
    st.write(f"Arena Gen: **{arena.generation}**")
    st.write(f"Meta Gen: **{meta.gen}**")
    st.write(f"Tahmin: **{tracker.stats()['toplam']}**")
    c1, c2 = st.columns(2)
    if c1.button("💾 Kaydet", use_container_width=True):
        try:
            with open(SAVE_FILE, "wb") as f:
                pickle.dump(books, f)
            st.toast("Diske kaydedildi")
        except Exception as e:
            st.error(f"Kayit hatasi: {e}")
    if c2.button("📂 Yukle", use_container_width=True):
        try:
            with open(SAVE_FILE, "rb") as f:
                loaded = pickle.load(f)
                ok = {k: v for k, v in loaded.items()
                      if all(getattr(b, "version", 1) == BOT_VERSION for b in v["arena"].bots)}
                if len(ok) < len(loaded):
                    st.toast("Eski surum kayitlar atlandi")
                st.session_state.books = ok
            st.rerun()
        except FileNotFoundError:
            st.warning("Kayit dosyasi yok")
        except Exception as e:
            st.error(f"Yukleme hatasi: {e}")
    if st.button("Bu hisseyi sifirla", use_container_width=True):
        books[secili] = new_book()
        st.rerun()

es = equity_stats(champion, price)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(secili, f"{price:.2f}", f"{canli['change_pct']:+.2f}%" if canli else None)
c2.metric("RSI", f"{last['RSI']:.1f}")
c3.metric("Sampiyon Getiri", f"%{es['getiri']:+.2f}")
c4.metric("Arena Gen", arena.generation)
c5.metric("Meta Gen", meta.gen)
if auto_msgs:
    st.info(" | ".join(auto_msgs[:4]))

tabs = st.tabs(["📈 Grafik", "🏟️ Arena", "🧠 NN", "🧬 Genetik", "🔬 Meta", "💭 Yansima",
                "🔮 Tahmin", "📊 Ensemble", "📚 Bilgi", "📜 Islemler", "📋 Oneriler",
                "📉 Analiz"])
(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11) = tabs

with t0:
    n_show = st.select_slider("Gun", options=[60, 120, 180, 250, 400], value=180)
    d = df.tail(n_show)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.5, 0.15, 0.2, 0.15])
    fig.add_trace(go.Candlestick(x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"],
                                 close=d["Close"], name="Fiyat"), row=1, col=1)
    for col, clr in [("SMA20", "#f5a623"), ("SMA50", "#4a90e2"), ("SMA200", "#bd10e0")]:
        fig.add_trace(go.Scatter(x=d["Date"], y=d[col], name=col, line=dict(width=1, color=clr)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["BB_up"], name="BB", line=dict(width=0.5, color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["BB_dn"], showlegend=False, fill="tonexty",
                             line=dict(width=0.5, color="gray")), row=1, col=1)
    fig.add_trace(go.Bar(x=d["Date"], y=d["Volume"], name="Hacim", marker_color="#555"), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["RSI"], name="RSI", line=dict(color="#50e3c2")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    fig.add_trace(go.Bar(x=d["Date"], y=d["MACD_hist"], name="MACD Hist",
                         marker_color=np.where(d["MACD_hist"] >= 0, "#26a69a", "#ef5350")), row=4, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["MACD"], name="MACD", line=dict(width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["MACD_sig"], name="Sinyal", line=dict(width=1)), row=4, col=1)
    fig.update_layout(height=720, template="plotly_dark", xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    puan, rows = teknik_ozet(last)
    st.subheader("Teknik Ozet")
    st.write(f"Toplam puan: **{puan:+d}** / {sum(1 for r in rows if r[2] != 0) or 1}  "
             f"({'olumlu' if puan > 1 else 'olumsuz' if puan < -1 else 'notr'})")
    st.dataframe(pd.DataFrame([{"Gosterge": a, "Durum": b,
                                "Sinyal": "🟢" if c > 0 else "🔴" if c < 0 else "⚪"} for a, b, c in rows]),
                 use_container_width=True, hide_index=True)

with t1:
    st.subheader("Multi-Bot Arena")
    st.dataframe(pd.DataFrame(arena.leaderboard(price)), use_container_width=True, hide_index=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampiyon getiri", f"%{es['getiri']:+.2f}")
    c2.metric("Islem", es["islem"])
    c3.metric("Kazanma orani", f"%{es['kazanma']:.0f}")
    c4.metric("Maks. dusus", f"%{es['mdd']:.1f}")
    fig = go.Figure()
    for b in arena.bots:
        vals = [b.initial] + [t.get("value", b.initial) for t in b.trades] + [b.value(price)]
        fig.add_trace(go.Scatter(y=vals, name=b.name, mode="lines",
                                 line=dict(width=3 if b.name == champion.name else 1)))
    fig.update_layout(height=400, template="plotly_dark", yaxis_title="Deger (TL)",
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.subheader("Neural Network")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mimari", "20-32-16-3")
    c2.metric("Parametre", f"{champion.nn.n_params()}")
    c3.metric("Egitim adimi", champion.nn.t)
    pt = book.get("pretrain")
    if pt:
        st.caption(f"On-egitim: {pt['n']} ornek | dogrulama dogrulugu %{pt['val_acc']*100:.0f} "
                   f"(taban %{pt['baseline']*100:.0f})")
    if champion.nn.loss_history:
        st.line_chart(pd.DataFrame({"Loss": champion.nn.loss_history[-500:]}))
    a, p, f = decisions[champion.name]
    c1, c2, c3 = st.columns(3)
    c1.metric("P(SAT)", f"%{p[0]*100:.1f}")
    c2.metric("P(TUT)", f"%{p[1]*100:.1f}")
    c3.metric("P(AL)", f"%{p[2]*100:.1f}")
    st.write(f"Karar: **{a}**")

with t3:
    if arena.history:
        st.dataframe(pd.DataFrame(arena.history), use_container_width=True, hide_index=True)
    else:
        st.info("50 turda bir evrim")

with t4:
    c1, c2, c3 = st.columns(3)
    c1.metric("Nesil", meta.gen)
    c2.metric("En Iyi", f"{max(meta.hist) if meta.hist else 0:+.2f}")
    c3.metric("Son", f"{meta.hist[-1] if meta.hist else 0:+.2f}")
    if meta.hall:
        rows = []
        for h in meta.hall:
            d_ = h["dna"]
            rows.append({"Nesil": h["gen"], "Fit": f"{h['fit']:+.2f}",
                         "Risk%": f"%{d_['risk_pct']*100:.0f}",
                         "SL%": f"{d_['sl_pct']:.1f}", "TP%": f"{d_['tp_pct']:.1f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if meta.hist:
        st.line_chart(pd.DataFrame({"Fit": meta.hist}))

with t5:
    if meta.reflections:
        for r in reversed(meta.reflections):
            st.info(r)
    else:
        st.warning("100 islem gerekiyor")

with t6:
    s = tracker.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam", s["toplam"])
    c2.metric("Dogrulanan", s["dogrulanan"])
    c3.metric("MAPE", f"%{s['mape']}")
    c4.metric("Gercekcilik", f"%{s['gercekcilik']}")
    g = s["gercekcilik"]
    (st.success if g >= 80 else st.info if g >= 60 else st.warning if g >= 40 else st.error)(s["yorum"])
    if pred_result:
        ensemble, preds, egim = pred_result
        st.dataframe(pd.DataFrame([{"Model": k.upper(), "Tahmin": f"{v:.2f}",
                                    "Fark%": f"{(v/price-1)*100:+.2f}"} for k, v in preds.items()]),
                     use_container_width=True, hide_index=True)
        st.metric("Ensemble (5 gun)", f"{ensemble:.2f}", f"{(ensemble/price-1)*100:+.2f}%")
        if st.button("Tahmini Kaydet"):
            tracker.add(str(date_d), date_d + timedelta(days=5), ensemble, price)
            st.toast("Kaydedildi")
            st.rerun()

with t7:
    st.dataframe(pd.DataFrame(predictor.stats()), use_container_width=True, hide_index=True)

with t8:
    arama = st.text_input("Ara", placeholder="RSI, temettu...")
    if arama:
        for s_ in bilgi_ara(arama):
            with st.expander(f"{s_['Konu']} ({s_['Kategori']})"):
                st.write(s_["Aciklama"])
    else:
        for k, m in BILGI.items():
            with st.expander(f"{k} ({len(m)})"):
                for b, i in m.items():
                    st.markdown(f"**{b}**: {i}")

with t9:
    if champion.trades:
        tdf = pd.DataFrame(champion.trades)
        tdf["date"] = pd.to_datetime(tdf["date"]).dt.strftime("%Y-%m-%d %H:%M")
        cols = [c for c in ["date", "action", "price", "qty", "value", "pnl", "reason"] if c in tdf.columns]
        st.dataframe(tdf[cols].iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button("CSV indir", tdf[cols].to_csv(index=False).encode("utf-8"),
                           file_name=f"{secili}_islemler.csv", mime="text/csv")
    else:
        st.info("Islem yok")
    for line in reversed(st.session_state.log[-15:]):
        st.text(line)

with t10:
    st.subheader("Yari-Otonom Oneri Sistemi")
    st.caption("Bot oneri uretir, sen onaylarsin.")
    try:
        imp = SelfImprover(get_remote_state(), **gh_conf())
        if imp.backend == "local":
            st.warning("GitHub token tanimli degil: onaylar sadece bu sunucuda kalir, bot gormez. "
                       "Streamlit secrets icine GITHUB_TOKEN ekle.")
        if st.button("Simdi Analiz Et", use_container_width=True):
            new = imp.analyze()
            (st.success(f"{len(new)} yeni oneri") if new else st.info("Yeni oneri yok (20 islem gerekli)"))
            get_remote_state.clear()
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
                    c1.metric("Eski", p.get("eski", "-"))
                    c2.metric("Yeni", p.get("yeni", "-"))
                if p.get("kod_ornek"):
                    st.code(p["kod_ornek"], language="python")
                c1, c2 = st.columns(2)
                if c1.button(f"Onayla #{p['id']}", key=f"a{p['id']}", use_container_width=True):
                    imp.approve(p["id"])
                    st.rerun()
                if c2.button(f"Reddet #{p['id']}", key=f"r{p['id']}", use_container_width=True):
                    imp.reject(p["id"])
                    st.rerun()
        approved = imp.approved()
        if approved:
            st.subheader(f"Onaylanan ({len(approved)})")
            st.dataframe(pd.DataFrame([{"ID": p["id"], "Tip": p.get("tip"), "Baslik": p["baslik"],
                                        "Durum": p.get("durum"), "Tarih": p.get("onay_tarih", "-")}
                                       for p in approved[-10:]]),
                         use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Hata: {e}")

with t11:
    st.subheader("📉 Performans ve İstatistiksel Anlamlılık")
    if not HAS_ANALYTICS:
        st.warning("analytics.py yuklu degil. Once o dosyayi ekle.")
    else:
        remote = get_remote_state()
        equity = remote.get("equity") or []
        trades = remote.get("trades") or []

        if not equity and not trades:
            st.info("Henuz GitHub Actions verisi yok. Ilk calismayi bekle.")
        else:
            curve = daily_curve(equity)
            st.markdown("### 📊 Varlık Eğrisi Metrikleri")
            p = perf(curve) if curve else None
            if p:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Toplam Getiri", f"%{p['toplam_pct']}")
                c2.metric("CAGR", f"%{p['cagr_pct']}" if p['cagr_pct'] is not None else "-")
                c3.metric("Sharpe", p['sharpe'] if p['sharpe'] is not None else "-")
                c4.metric("Sortino", p['sortino'] if p['sortino'] is not None else "-")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Volatilite (yillik)", f"%{p['vol_pct']}")
                c2.metric("Max Drawdown", f"%{p['maxdd_pct']}")
                c3.metric("Calmar", p['calmar'] if p['calmar'] is not None else "-")
                c4.metric("Gun", p['gun'])
                with st.expander("Ham veri"):
                    st.json(p)
            else:
                st.info("Yeterli equity verisi yok (en az 3 gun gerekli).")

            st.divider()
            st.markdown("### 💰 İşlem İstatistikleri")
            ts = trade_stats(trades)
            if ts:
                c1, c2, c3 = st.columns(3)
                c1.metric("İşlem", ts["islem"])
                c2.metric("Kazanma %", f"%{ts['kazanma_pct']}")
                c3.metric("Kar Faktörü", ts["kar_faktoru"] if ts["kar_faktoru"] is not None else "-")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Beklenti (TL)", ts["beklenti_tl"])
                c2.metric("Beklenti %", f"%{ts['beklenti_pct']}" if ts['beklenti_pct'] is not None else "-")
                c3.metric("Ort. Kazanç (TL)", ts["ort_kazanc_tl"])
                c4.metric("Ort. Kayıp (TL)", ts["ort_kayip_tl"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Ödeme Oranı", ts["odeme_orani"] if ts["odeme_orani"] is not None else "-")
                c2.metric("Ort. Tutma (gün)", ts["ort_tutma_gun"] if ts["ort_tutma_gun"] is not None else "-")
                c3.metric("Maks. Üst Üste Zarar", ts["maks_ust_uste_zarar"])
                c1, c2 = st.columns(2)
                c1.metric("Ort. MAE %", ts["ort_mae_pct"] if ts["ort_mae_pct"] is not None else "-")
                c2.metric("Ort. MFE %", ts["ort_mfe_pct"] if ts["ort_mfe_pct"] is not None else "-")
                st.caption("MAE: pozisyon açıkken görülen en kötü kayıp. MFE: en iyi kâr.")
            else:
                st.info("Henuz kapanmis islem yok.")

            st.divider()
            st.markdown("### 🎲 Bootstrap Güven Aralığı")
            if len(curve) >= 10:
                rets = np.diff(curve) / curve[:-1]
                bs = bootstrap_mean(rets, block=5, seed=42)
                if bs:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Ort. Günlük Getiri", f"{bs['mean']*100:.3f}%")
                    c2.metric("%10-%90 CI", f"[{bs['lo']*100:.3f}%, {bs['hi']*100:.3f}%]")
                    c3.metric("P(pozitif)", f"%{bs['p_pos']*100:.1f}")
                    if bs["p_pos"] > 0.6:
                        st.success("Güven aralığının çoğu pozitif — strateji muhtemelen gerçek edge taşıyor.")
                    elif bs["p_pos"] > 0.4:
                        st.warning("Sonuçlar kararsız — daha fazla veri lazım.")
                    else:
                        st.error("Günlük getiri büyük olasılıkla negatif — strateji şu an para kaybediyor.")
            else:
                st.info("Bootstrap için en az 10 gün gerekli.")

            st.divider()
            st.markdown("### 🧪 Permütasyon Testi")
            if len(curve) >= 50:
                rets = np.diff(curve) / curve[:-1]
                pos = np.where(rets > 0, 1.0, 0.0)
                pp = perm_pvalue(pos, rets, seed=7)
                if pp:
                    c1, c2 = st.columns(2)
                    c1.metric("Gözlenen Ort. Getiri", f"{pp['obs']*100:.4f}%")
                    c2.metric("p-değeri", f"{pp['p']:.3f}")
                    if pp["p"] < 0.05:
                        st.success(f"p={pp['p']:.3f} < 0.05 → Bu sonuç tesadüf olma ihtimali düşük. **İstatistiksel olarak anlamlı.**")
                    elif pp["p"] < 0.10:
                        st.warning(f"p={pp['p']:.3f} → SınırdA. Daha fazla veri topla.")
                    else:
                        st.error(f"p={pp['p']:.3f} → Sonuç tesadüfi olabilir.")
            else:
                st.info("Permütasyon testi için en az 50 gün gerekli.")

st.divider()
st.caption(f"{now_tr():%Y-%m-%d %H:%M:%S} - Borsa: {'ACIK' if market_open() else 'KAPALI'}")
