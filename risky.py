"""risk.py - Profesyonel risk yonetimi: saf (durumsuz) fonksiyonlar.

* ATR tabanli stop mesafesi ve en az 2R hedef
* Islem basina sabit varlik riski (stop'ta kayip = varligin %1'i) ile pozisyon boyutu
* Kara gecince devreye giren takip eden stop (trailing)
* Rejim filtresi: onayli dusus trendinde veya turbulansta yeni alis yok
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class RiskConfig:
    atr_k: float = 2.0            # stop mesafesi = atr_k * ATR%  (alt sinir: bot.sl_pct)
    stop_max: float = 8.0         # stop ust siniri (%)
    rr_min: float = 2.0           # kar hedefi en az rr_min * stop
    trail_act_r: float = 1.0      # takip eden stop, +1R kara gecince aktif olur
    trail_r: float = 1.0          # takip mesafesi (R cinsinden)
    risk_per_trade: float = 0.01  # stop'ta kaybedilecek varlik orani
    max_loss_streak: int = 3      # ust uste zarar -> bekleme
    cooldown_days: int = 3
    vol_halt: float = 0.80        # yillik volatilite bunun ustundeyse yeni alis yok
    regime_filter: bool = True


def atr_pct(row):
    try:
        v = float(row["ATR"]) / float(row["Close"]) * 100
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def stop_distance(atr_p, floor, rc):
    """% cinsinden stop mesafesi: [floor, stop_max] araliginda atr_k*ATR%."""
    if atr_p is None:
        return float(floor)
    return float(min(max(rc.atr_k * atr_p, floor), max(floor, rc.stop_max)))


def tp_target(stop, base_tp, rc):
    return max(float(base_tp), rc.rr_min * stop)


def trail_level(high, entry, stop, rc):
    """Takip eden stop fiyati (aktif degilse None)."""
    if high >= entry * (1 + rc.trail_act_r * stop / 100):
        return high * (1 - rc.trail_r * stop / 100)
    return None


def size_position(equity, cash, price, stop, fee, slip, rc, max_frac, risk_mult=1.0):
    """Adet: stop'ta kayip <= equity*risk_per_trade*risk_mult; ayrica pozisyon <= max_frac*equity
    ve nakit."""
    risk_mult = max(0.0, float(risk_mult))
    px = price * (1 + slip)
    loss_ps = px * stop / 100 + 2 * px * fee          # stop'ta hisse basi kayip + gidis-donus komisyon
    if px <= 0 or loss_ps <= 0:
        return 0
    q_risk = equity * rc.risk_per_trade * risk_mult / loss_ps
    q_cap = max_frac * equity / (px * (1 + fee))
    q_cash = cash / (px * (1 + fee))
    return int(max(0, np.floor(min(q_risk, q_cap, q_cash))))


def regime_of(row, rc):
    try:
        c, s50, s200 = float(row["Close"]), float(row["SMA50"]), float(row["SMA200"])
        vol = float(row["Volatility"])
    except (KeyError, TypeError, ValueError):
        return {"trend": "bilinmiyor", "vol": 0.0, "turbulans": False, "ok_long": True}
    if c < s200 and s50 < s200:
        trend = "dusus"
    elif c > s200 and s50 >= s200:
        trend = "yukselis"
    else:
        trend = "yatay"
    turb = vol > rc.vol_halt
    ok = True if not rc.regime_filter else (trend != "dusus" and not turb)
    return {"trend": trend, "vol": vol, "turbulans": turb, "ok_long": ok}
