# self_improve.py - Yari-otonom oneri motoru
import json
import os
from datetime import datetime

PROPOSALS_FILE = "proposals.json"


class SelfImprover:
    """Botun kendi performansini analiz edip oneri uretir."""

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
        """Performansi analiz et, oneri uret."""
        trades = self.state.get("trades", [])
        if len(trades) < 20:
            return []

        new_props = []
        sells = [t for t in trades if t.get("action") == "SAT"]
        buys = [t for t in trades if t.get("action") == "AL"]

        if len(sells) < 5:
            return []

        # 1. Kazanma oranina gore
        wins = sum(1 for t in sells if t.get("pnl", 0) > 0)
        wr = wins / len(sells)

        if wr < 0.4:
            new_props.append({
                "tip": "param",
                "baslik": "Kazanma orani dusuk (%" + str(int(wr*100)) + ")",
                "oneri": "Kar esigini (TP) artir: 6% -> 8%",
                "eski": 6.0, "yeni": 8.0,
                "anahtar": "tp_pct",
                "sebep": "Az sayida islem karli, daha buyuk hareketleri bekle.",
            })
            new_props.append({
                "tip": "param",
                "baslik": "Cok fazla zararli islem",
                "oneri": "Zarar kesmeyi (SL) sikilastir: 3% -> 2%",
                "eski": 3.0, "yeni": 2.0,
                "anahtar": "sl_pct",
                "sebep": "Kucuk zararlar buyumesin.",
            })

        if wr > 0.7:
            new_props.append({
                "tip": "param",
                "baslik": "Kazanma orani yuksek (%" + str(int(wr*100)) + ")",
                "oneri": "Risk oranini artir: 25% -> 30%",
                "eski": 25.0, "yeni": 30.0,
                "anahtar": "risk_pct",
                "sebep": "Iyi calisiyorsun, daha buyuk pozisyon alabilirsin.",
            })

        # 2. Ortalama kar analizi
        avg_pnl = sum(t.get("pnl", 0) for t in sells) / len(sells)
        if avg_pnl < 0:
            new_props.append({
                "tip": "param",
                "baslik": "Ortalama kar negatif",
                "oneri": "Islem sikligi azalt, esik yukari: 0.25 -> 0.35",
                "eski": 0.25, "yeni": 0.35,
                "anahtar": "min_conf",
                "sebep": "Sadece cok guclu sinyallerde al.",
            })

        # 3. Cok islem yapiyor mu?
        if len(trades) > 100 and len(sells) > 50:
            new_props.append({
                "tip": "param",
                "baslik": "Cok fazla islem (komisyon zarari)",
                "oneri": "Minimum usta onayi: 8 -> 12",
                "eski": 8, "yeni": 12,
                "anahtar": "min_guru",
                "sebep": "Daha az ama daha kaliteli islem.",
            })

        # 4. Kod degisikligi onerileri (kod analizi)
        code_props = self._code_suggestions()
        new_props.extend(code_props)

        # Yeni olanlari kaydet
        existing_titles = {p["baslik"] for p in self.proposals if p.get("durum") == "bekliyor"}
        for p in new_props:
            if p["baslik"] not in existing_titles:
                p["durum"] = "bekliyor"
                p["tarih"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                p["id"] = len(self.proposals) + 1
                self.proposals.append(p)

        self.proposals = self.proposals[-30:]  # Son 30
        self._save()
        return new_props

    def _code_suggestions(self):
        """Kod degisikligi onerileri (sadece metin, otomatik uygulanmaz)."""
        return [
            {
                "tip": "kod",
                "baslik": "Yeni feature: hacim agirligi",
                "oneri": "extract_features fonksiyonuna 21. ozellik olarak 'Volume_MA5/Volume_MA20' ekle",
                "anahtar": "kod_feature_1",
                "sebep": "Hacim trendi sinyal kalitesini artirabilir.",
                "kod_ornek": "df['Vol_MA5'] = df['Volume'].rolling(5).mean()\ndf['Vol_MA20'] = df['Volume'].rolling(20).mean()\ndf['Vol_ratio_5_20'] = df['Vol_MA5'] / df['Vol_MA20']",
            },
            {
                "tip": "kod",
                "baslik": "Yeni gosterge: OBV diverjansi",
                "oneri": "OBV ile fiyat arasindaki uyumsuzlugu sinyal olarak ekle",
                "anahtar": "kod_feature_2",
                "sebep": "Diverjans donus sinyali olabilir.",
                "kod_ornek": "# OBV zaten var, diverjans hesapla\ndf['OBV_slope'] = df['OBV'].diff(5)",
            },
        ]

    def approve(self, prop_id):
        """Oneriyi onayla ve uygula."""
        for p in self.proposals:
            if p.get("id") == prop_id and p.get("durum") == "bekliyor":
                p["durum"] = "onaylandi"
                p["onay_tarih"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                self._save()
                return p
        return None

    def reject(self, prop_id):
        for p in self.proposals:
            if p.get("id") == prop_id and p.get("durum") == "bekliyor":
                p["durum"] = "reddedildi"
                p["onay_tarih"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                self._save()
                return p
        return None

    def pending(self):
        return [p for p in self.proposals if p.get("durum") == "bekliyor"]

    def approved(self):
        return [p for p in self.proposals if p.get("durum") == "onaylandi"]

    def apply_approved_to_state(self):
        """Onaylanmis param onerilerini state'e uygula."""
        applied = []
        for p in self.approved():
            if p.get("tip") == "param" and "anahtar" in p:
                self.state[p["anahtar"]] = p["yeni"]
                applied.append(p["anahtar"])
        return applied
