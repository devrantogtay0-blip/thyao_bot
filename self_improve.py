# self_improve.py - Yari-otonom oneri motoru v10
import json
import os
from datetime import datetime

PROPOSALS_FILE = "proposals.json"

class SelfImprover:
    def __init__(self, state):
        self.state = state
        self.proposals = self._load()

    def _load(self):
        if os.path.exists(PROPOSALS_FILE):
            try:
                with open(PROPOSALS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save(self):
        with open(PROPOSALS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.proposals, f, indent=2, ensure_ascii=False, default=str)

    def analyze(self):
        trades = self.state.get("trades", [])
        if len(trades) < 20: return []
        new_props = []
        sells = [t for t in trades if t.get("action") == "SAT"]
        if len(sells) < 5: return []
        wins = sum(1 for t in sells if t.get("pnl", 0) > 0)
        wr = wins / len(sells)
        if wr < 0.4:
            new_props.append({"tip": "param", "baslik": f"Kazanma orani dusuk (%{int(wr*100)})",
                              "oneri": "Kar esigini (TP) artir: 6% -> 8%",
                              "eski": 6.0, "yeni": 8.0, "anahtar": "tp_pct",
                              "sebep": "Daha buyuk hareketleri bekle."})
            new_props.append({"tip": "param", "baslik": "Cok fazla zararli islem",
                              "oneri": "Zarar kesmeyi (SL) sikilastir: 3% -> 2%",
                              "eski": 3.0, "yeni": 2.0, "anahtar": "sl_pct",
                              "sebep": "Kucuk zararlar buyumesin."})
        if wr > 0.7:
            new_props.append({"tip": "param", "baslik": f"Kazanma orani yuksek (%{int(wr*100)})",
                              "oneri": "Risk oranini artir: 25% -> 30%",
                              "eski": 25.0, "yeni": 30.0, "anahtar": "risk_pct",
                              "sebep": "Iyi calisiyorsun."})
        avg = sum(t.get("pnl", 0) for t in sells) / len(sells)
        if avg < 0:
            new_props.append({"tip": "param", "baslik": "Ortalama kar negatif",
                              "oneri": "Esik yukari: 0.25 -> 0.35",
                              "eski": 0.25, "yeni": 0.35, "anahtar": "min_conf",
                              "sebep": "Sadece guclu sinyaller."})
        new_props.extend([
            {"tip": "kod", "baslik": "Yeni feature: hacim agirligi",
             "oneri": "extract_features'a 'Vol_MA5/Vol_MA20' ekle",
             "anahtar": "kod_1", "sebep": "Hacim trendi sinyal kalitesini artirabilir.",
             "kod_ornek": "df['Vol_MA5'] = df['Volume'].rolling(5).mean()\ndf['Vol_MA20'] = df['Volume'].rolling(20).mean()\ndf['Vol_ratio_5_20'] = df['Vol_MA5'] / df['Vol_MA20']"},
            {"tip": "kod", "baslik": "Yeni gosterge: OBV diverjansi",
             "oneri": "OBV ile fiyat arasindaki uyumsuzlugu sinyal yap",
             "anahtar": "kod_2", "sebep": "Diverjans donus sinyali olabilir.",
             "kod_ornek": "df['OBV_slope'] = df['OBV'].diff(5)"},
        ])
        existing = {p["baslik"] for p in self.proposals if p.get("durum") == "bekliyor"}
        for p in new_props:
            if p["baslik"] not in existing:
                p["durum"] = "bekliyor"
                p["tarih"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                p["id"] = len(self.proposals) + 1
                self.proposals.append(p)
        self.proposals = self.proposals[-30:]
        self._save()
        return new_props

    def approve(self, pid):
        for p in self.proposals:
            if p.get("id") == pid and p.get("durum") == "bekliyor":
                p["durum"] = "onaylandi"
                p["onay_tarih"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                self._save()
                return p
        return None

    def reject(self, pid):
        for p in self.proposals:
            if p.get("id") == pid and p.get("durum") == "bekliyor":
                p["durum"] = "reddedildi"
                p["onay_tarih"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                self._save()
                return p
        return None

    def pending(self): return [p for p in self.proposals if p.get("durum") == "bekliyor"]
    def approved(self): return [p for p in self.proposals if p.get("durum") == "onaylandi"]

    def apply_approved_to_state(self):
        applied = []
        for p in self.approved():
            if p.get("tip") == "param" and "anahtar" in p:
                self.state[p["anahtar"]] = p["yeni"]
                applied.append(p["anahtar"])
        return applied
