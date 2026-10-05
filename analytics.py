"""analytics.py - Performans metrikleri ve istatistiksel anlamlilik testleri."""
import numpy as np


def _f(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def maxdd(curve):
    """Maksimum dusus (%, negatif)."""
    a = np.asarray(curve, dtype=float)
    if len(a) == 0:
        return 0.0
    peak = np.maximum.accumulate(a)
    return float(((a - peak) / peak).min() * 100)


def daily_curve(equity):
    """[{'ts','value'}...] -> gun sonu degerleri (gunun son kaydi)."""
    last = {}
    for e in equity or []:
        try:
            last[str(e["ts"])[:10]] = float(e["value"])
        except (KeyError, TypeError, ValueError):
            continue
    return [last[k] for k in sorted(last)]


def perf(curve, ppy=252, rf=0.0):
    """Gunluk varlik egrisinden CAGR, Sharpe, Sortino, Calmar, maks. dusus."""
    a = np.asarray(curve, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return None
    r = a[1:] / a[:-1] - 1
    total = a[-1] / a[0] - 1
    yrs = len(r) / ppy
    cagr = (1 + total) ** (1 / yrs) - 1 if yrs > 0 and 1 + total > 0 else float("nan")
    sd = r.std(ddof=1)
    sharpe = (r.mean() - rf / ppy) / sd * np.sqrt(ppy) if sd > 0 else 0.0
    dd_dev = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    sortino = r.mean() / dd_dev * np.sqrt(ppy) if dd_dev > 0 else 0.0
    mdd = maxdd(a)
    calmar = cagr / abs(mdd / 100) if mdd < 0 and np.isfinite(cagr) else 0.0
    return {"gun": len(r), "toplam_pct": _f(total * 100, 2), "cagr_pct": _f(cagr * 100, 2),
            "vol_pct": _f(sd * np.sqrt(ppy) * 100, 2), "sharpe": _f(sharpe, 2),
            "sortino": _f(sortino, 2), "maxdd_pct": _f(mdd, 2), "calmar": _f(calmar, 2)}


def trade_stats(trades):
    """Kapanan (SAT) islemlerden: kazanma, kar faktoru, beklenti, ort. tutma, MAE/MFE..."""
    sells = [t for t in trades or [] if t.get("action") == "SAT" and t.get("pnl") is not None]
    if not sells:
        return None
    pnl = np.array([float(t["pnl"]) for t in sells])
    wins, losses = pnl[pnl > 0], -pnl[pnl < 0]
    streak = cur = 0
    for p in pnl:
        cur = cur + 1 if p <= 0 else 0
        streak = max(streak, cur)

    def avg(key):
        v = [float(t[key]) for t in sells if t.get(key) is not None]
        return _f(np.mean(v), 2) if v else None

    return {"islem": len(sells), "kazanma_pct": _f(len(wins) / len(pnl) * 100, 1),
            "kar_faktoru": _f(wins.sum() / losses.sum(), 2) if losses.sum() > 0 else None,
            "beklenti_tl": _f(pnl.mean(), 1), "beklenti_pct": avg("ret_pct"),
            "ort_kazanc_tl": _f(wins.mean(), 1) if len(wins) else 0.0,
            "ort_kayip_tl": _f(-losses.mean(), 1) if len(losses) else 0.0,
            "odeme_orani": _f(wins.mean() / losses.mean(), 2) if len(wins) and len(losses) else None,
            "ort_tutma_gun": avg("hold"), "ort_mae_pct": avg("mae"), "ort_mfe_pct": avg("mfe"),
            "maks_ust_uste_zarar": streak}


def bootstrap_mean(x, block=1, n=2000, seed=0, q_lo=10):
    """Blok bootstrap ile ortalama ve guven araligi [q_lo, 100-q_lo] yuzdelik (ust uste binen /
    otokorelasyonlu getiriler icin blok>1)."""
    x = np.asarray(x, dtype=float)
    m = len(x)
    if m < 2:
        return None
    block = max(1, min(int(block), m))
    rng = np.random.default_rng(seed)
    k = int(np.ceil(m / block))
    starts = rng.integers(0, m - block + 1, size=(n, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :m]
    means = x[idx].mean(axis=1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(means, q_lo)),
            "hi": float(np.percentile(means, 100 - q_lo)), "p_pos": float((means > 0).mean())}


def perm_pvalue(pos, ret, n=2000, seed=0):
    """Pozisyon serisini dairesel kaydirarak sans dagilimi: 'bu getiri tesadufen olur muydu?'
    pos: gunluk 0/1 pozisyon, ret: ayni gunlerin getirisi. Kucuk p = tesadufi olma ihtimali dusuk."""
    pos = np.asarray(pos, dtype=float)
    ret = np.asarray(ret, dtype=float)
    L = len(pos)
    if L < 50 or L != len(ret):
        return None
    obs = float(np.mean(pos * ret))
    rng = np.random.default_rng(seed)
    lo = max(5, L // 10)
    shifts = rng.integers(lo, L - lo, n)
    null = np.array([np.mean(np.roll(pos, k) * ret) for k in shifts])
    return {"obs": obs, "p": float((np.sum(null >= obs) + 1) / (n + 1)), "null_mean": float(null.mean())}
