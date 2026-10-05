"""Ortak gosterge hesaplama - app.py ve bot.py ayni fonksiyonu kullanir.

Boylece NN'e giren ozellikler canli panelde ve 7/24 botta birebir ayni olur.
"""
import numpy as np
import pandas as pd


def add_ind(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    d = c.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    ls = (-d.where(d < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))

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
    rng = (hi14 - lo14).replace(0, np.nan)
    df["K"] = 100 * (c - lo14) / rng
    df["D"] = df["K"].rolling(3).mean()

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["WILLR"] = -100 * (hi14 - c) / rng

    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr14 = tr.rolling(14).sum().replace(0, np.nan)
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(14).sum() / tr14
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(14).sum() / tr14
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["ADX"] = dx.rolling(14).mean()

    df["Vol_SMA"] = v.rolling(20).mean()
    df["Vol_ratio"] = v / df["Vol_SMA"].replace(0, np.nan)
    df["Returns"] = c.pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252)
    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["EMA21"] = c.ewm(span=21, adjust=False).mean()
    df["Mom10"] = c.pct_change(10)
    df["Mom30"] = c.pct_change(30)
    df["HL_ratio"] = (h - l) / c
    df["Gap"] = (df["Open"] - c.shift()) / c.shift()

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
