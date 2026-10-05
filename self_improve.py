# self_improve.py - Yari-otonom oneri motoru v11
"""Bot istatistiklerinden parametre/kod onerisi uretir; sen onaylarsin, bot uygular.

Depolama (proposals.json):
  * local  : calisma dizinindeki dosya (GitHub Actions icinde bot bunu kullanir)
  * github : GitHub Contents API (Streamlit paneli bunu kullanir, boylece panelde
             verdigin onay repoya yazilir ve bot gercekten gorur)
  GitHub icin: SelfImprover(state, gh_token=..., gh_repo="kullanici/repo")
"""
import base64
import copy
import json
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TR = ZoneInfo("Europe/Istanbul")
PROPOSALS_FILE = os.getenv("PROPOSALS_FILE", "proposals.json")

PARAMS = {
    "tp_pct": (1.0, 30.0, 6.0, "Kar al (TP) %"),
    "sl_pct": (0.5, 15.0, 3.0, "Zarar kes (SL) %"),
    "risk_pct": (0.02, 0.5, 0.25, "Risk orani"),
    "min_conf": (0.05, 0.9, 0.25, "Guven esigi"),
}

MIN_TRADES = 20
MIN_SELLS = 5
MIN_SELLS_PARAM = 10
WINDOW = 30
COOLDOWN_DAYS = 7
EXPIRE_DAYS = 14
KEEP_HISTORY = 50
CACHE_TTL = 20

KOD_ONERILERI = [
    {"tip": "kod", "anahtar": "kod_1", "baslik": "Yeni feature: hacim agirligi",
     "oneri": "extract_features'a 'Vol_MA5/Vol_MA20' ekle",
     "sebep": "Hacim trendi sinyal kalitesini artirabilir.",
     "kod_ornek": "df['Vol_MA5'] = df['Volume'].rolling(5).mean()\n"
                  "df['Vol_MA20'] = df['Volume'].rolling(20).mean()\n"
                  "df['Vol_ratio_5_20'] = df['Vol_MA5'] / df['Vol_MA20']"},
    {"tip": "kod", "anahtar": "kod_2", "baslik": "Yeni gosterge: OBV diverjansi",
     "oneri": "OBV ile fiyat arasindaki uyumsuzlugu sinyal yap",
     "sebep": "Diverjans donus sinyali olabilir.",
     "kod_ornek": "df['OBV'] = (np.sign(df['Close'].diff()).fillna(0) * df['Volume']).cumsum()\n"
                  "df['OBV_slope'] = df['OBV'].diff(5)\n"
                  "df['Price_slope'] = df['Close'].diff(5)\n"
                  "df['OBV_div'] = np.sign(df['OBV_slope']) - np.sign(df['Price_slope'])"},
]


def _now():
    return datetime.now(TR).replace(tzinfo=None)


def _stamp():
    return _now().strftime("%Y-%m-%d %H:%M")


def _parse(s):
    if not s:
        return None
    try:
        d = datetime.fromisoformat(str(s))
    except ValueError:
        return None
    return d.replace(tzinfo=None) if d.tzinfo else d


def _num(x):
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _clamp(key, v):
    lo, hi = PARAMS[key][:2]
    return min(max(v, lo), hi)


def _norm(key, val):
    if key not in PARAMS:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if v != v or abs(v) == float("inf"):
        return None
    if key == "risk_pct" and v > 1:
        v /= 100.0
    return round(_clamp(key, v), 4)


def _cur(state, key):
    v = _norm(key, state.get(key))
    return v if v is not None else PARAMS[key][2]


def _drawdown(state):
    try:
        eq = state.get("equity") or []
        peak = float(state.get("peak_value") or 0)
        if not eq or peak <= 0:
            return None
        return (1 - float(eq[-1]["value"]) / peak) * 100
    except Exception:
        return None


def _stats(sells):
    pnls = [_num(t.get("pnl")) for t in sells]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return {"n": n, "wr": len(wins) / n, "exp": sum(pnls) / n,
            "payoff": (avg_win / avg_loss) if avg_loss else None,
            "pf": (sum(wins) / sum(losses)) if losses else float("inf")}


def _pub(m):
    return {k: (None if v is None or v == float("inf") else round(v, 3)) for k, v in m.items()}


def _migrate(data):
    if not isinstance(data, list):
        return []
    data = [p for p in data if isinstance(p, dict)]
    for p in data:
        if p.get("anahtar") == "risk_pct":
            for f in ("eski", "yeni"):
                v = _num(p.get(f))
                if v > 1:
                    p[f] = round(v / 100.0, 4)
    return data


def _prune(data):
    last_by_key = {p.get("anahtar"): p.get("id") for p in data}
    decided = [p for p in data if p.get("durum") != "bekliyor"]
    keep = {p.get("id") for p in data if p.get("durum") == "bekliyor"}
    keep |= {p.get("id") for p in decided[-KEEP_HISTORY:]}
    keep |= set(last_by_key.values())
    return [p for p in data if p.get("id") in keep]


def _blocked(data, key):
    now = _now()
    for p in data:
        if p.get("anahtar") != key:
            continue
        d = p.get("durum")
        if d == "bekliyor":
            return True
        if d in ("onaylandi", "uygulandi", "reddedildi"):
            if key.startswith("kod_"):
                return True
            t = _parse(p.get("onay_tarih"))
            if t is None or now - t < timedelta(days=COOLDOWN_DAYS):
                return True
    return False


class _Conflict(Exception):
    pass


class _GitHubStore:
    def __init__(self, token, repo, path, branch="main"):
        self.url = f"https://api.github.com/repos/{repo}/contents/{path}"
        self.branch = branch
        self.h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                  "X-GitHub-Api-Version": "2022-11-28"}

    def read(self):
        import requests
        r = requests.get(self.url, headers=self.h, params={"ref": self.branch}, timeout=10)
        if r.status_code == 404:
            return [], None
        r.raise_for_status()
        j = r.json()
        raw = base64.b64decode(j["content"]).decode("utf-8")
        return json.loads(raw or "[]"), j["sha"]

    def write(self, data, sha, msg):
        import requests
        body = {"message": msg, "branch": self.branch,
                "content": base64.b64encode(json.dumps(data, indent=2, ensure_ascii=False,
                                                       default=str).encode("utf-8")).decode()}
        if sha:
            body["sha"] = sha
        r = requests.put(self.url, headers=self.h, json=body, timeout=15)
        if r.status_code in (409, 422):
            raise _Conflict()
        r.raise_for_status()


_CACHE = {}


class SelfImprover:
    def __init__(self, state, gh_token=None, gh_repo=None, gh_branch=None):
        self.state = state
        token = gh_token or os.getenv("PROPOSALS_GH_TOKEN")
        repo = gh_repo or os.getenv("PROPOSALS_GH_REPO")
        branch = gh_branch or os.getenv("PROPOSALS_GH_BRANCH") or "main"
        self._gh = _GitHubStore(token, repo, os.path.basename(PROPOSALS_FILE), branch) \
            if token and repo else None
        self._ckey = (repo, branch)
        self.proposals = self._load()

    @property
    def backend(self):
        return "github" if self._gh else "local"

    def _read_local(self):
        if not os.path.exists(PROPOSALS_FILE):
            return []
        try:
            with open(PROPOSALS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            try:
                os.replace(PROPOSALS_FILE, PROPOSALS_FILE + ".bak")
            except OSError:
                pass
            return []

    def _read(self):
        data, sha = self._gh.read() if self._gh else (self._read_local(), None)
        return _migrate(data), sha

    def _write(self, data, sha, msg):
        if self._gh:
            self._gh.write(data, sha, msg)
            return
        tmp = PROPOSALS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, PROPOSALS_FILE)

    def _load(self):
        if self._gh:
            hit = _CACHE.get(self._ckey)
            if hit and time.time() - hit[0] < CACHE_TTL:
                return copy.deepcopy(hit[1])
        data, _ = self._read()
        if self._gh:
            _CACHE[self._ckey] = (time.time(), copy.deepcopy(data))
        return data

    def _update(self, fn, msg):
        for i in range(4):
            data, sha = self._read()
            result, changed = fn(data)
            if not changed:
                self.proposals = data
                return result
            data = _prune(data)
            try:
                self._write(data, sha, msg)
            except _Conflict:
                time.sleep(0.5 * (i + 1))
                continue
            self.proposals = data
            if self._gh:
                _CACHE[self._ckey] = (time.time(), copy.deepcopy(data))
            return result
        raise RuntimeError("proposals.json guncellenemedi")

    def _expire(self, data):
        changed = False
        now = _now()
        for p in data:
            if p.get("durum") != "bekliyor" or p.get("tip") != "param":
                continue
            key = p.get("anahtar")
            t = _parse(p.get("tarih"))
            moved = key in PARAMS and abs(_cur(self.state, key) - _num(p.get("eski"))) > 1e-9
            if moved:
                p["durum"], changed = "gecersiz", True
            elif t and now - t > timedelta(days=EXPIRE_DAYS):
                p["durum"], changed = "suresi_doldu", True
        return changed

    @staticmethod
    def _cutoff(data):
        ts = [_parse(p.get("uygulama_ts")) for p in data if p.get("tip") == "param"]
        ts = [t for t in ts if t]
        return max(ts) if ts else None

    def _param_candidates(self, sells):
        res = []
        if len(sells) < MIN_SELLS_PARAM:
            return res
        m = _stats(sells)
        pub = _pub(m)
        dd = _drawdown(self.state)
        wr, pay, pf, exp, n = m["wr"], m["payoff"], m["pf"], m["exp"], m["n"]

        def add(key, new, baslik, sebep):
            cur = _cur(self.state, key)
            new = round(_clamp(key, new), 4)
            if abs(new - cur) < 1e-9:
                return
            res.append({"tip": "param", "baslik": baslik, "anahtar": key,
                        "oneri": f"{PARAMS[key][3]}: {cur:g} -> {new:g}",
                        "eski": cur, "yeni": new, "sebep": sebep, "metrik": pub})

        conf = _cur(self.state, "min_conf")
        if wr < 0.40:
            add("min_conf", conf + 0.05, f"Kazanma orani dusuk (%{wr*100:.0f})",
                f"Son {n} satista isabet dusuk.")
        elif exp < 0:
            add("min_conf", conf + 0.05, "Ortalama islem zararda",
                f"Son {n} satista ortalama PnL {exp:+.0f} TL.")
        if pay is not None and pay < 0.8:
            sl = _cur(self.state, "sl_pct")
            add("sl_pct", max(0.5, round(sl * 0.8 * 2) / 2), "Zararlar karlardan buyuk",
                f"Ortalama kar/zarar orani {pay:.2f}.")
        risk = _cur(self.state, "risk_pct")
        if wr > 0.70 and pf >= 1.5 and exp > 0 and n >= 15 and (dd is None or dd < 5):
            if risk < 0.40:
                add("risk_pct", min(risk + 0.05, 0.40), f"Kazanma orani yuksek (%{wr*100:.0f})",
                    f"Son {n} satis, PF {pf:.1f}.")
        if dd is not None and dd >= 15:
            add("risk_pct", round(risk * 0.7, 2), f"Portfoy zirveden %{dd:.0f} dustu",
                "Riski azalt.")
        return res

    def analyze(self):
        trades = self.state.get("trades", [])
        sells_all = [t for t in trades if t.get("action") == "SAT"]
        if len(trades) < MIN_TRADES or len(sells_all) < MIN_SELLS:
            return []
        added = []

        def fn(data):
            added.clear()
            changed = self._expire(data)
            cutoff = self._cutoff(data)
            sells = sells_all
            if cutoff:
                sells = [t for t in sells_all if (_parse(t.get("ts")) or datetime.min) > cutoff]
            cands = self._param_candidates(sells[-WINDOW:]) + [dict(k) for k in KOD_ONERILERI]
            next_id = max([p.get("id", 0) for p in data] + [0]) + 1
            for p in cands:
                if _blocked(data, p["anahtar"]):
                    continue
                p.update(durum="bekliyor", tarih=_stamp(), id=next_id)
                next_id += 1
                data.append(p)
                added.append(p)
                changed = True
            return list(added), changed

        return self._update(fn, "proposals: yeni oneri [skip ci]")

    def _decide(self, pid, durum):
        out = []

        def fn(data):
            out.clear()
            for p in data:
                if p.get("id") == pid and p.get("durum") == "bekliyor":
                    p["durum"], p["onay_tarih"] = durum, _stamp()
                    out.append(dict(p))
                    return out, True
            return out, False

        self._update(fn, f"proposals: #{pid} {durum} [skip ci]")
        return out[0] if out else None

    def approve(self, pid):
        return self._decide(pid, "onaylandi")

    def reject(self, pid):
        return self._decide(pid, "reddedildi")

    def pending(self):
        return [p for p in self.proposals if p.get("durum") == "bekliyor"]

    def approved(self):
        return [p for p in self.proposals if p.get("durum") in ("onaylandi", "uygulandi")]

    def apply_approved_to_state(self):
        applied = []

        def fn(data):
            applied.clear()
            changed = False
            for p in data:
                if p.get("durum") != "onaylandi" or p.get("tip") != "param":
                    continue
                key = p.get("anahtar")
                val = _norm(key, p.get("yeni"))
                if val is None:
                    p["durum"], changed = "gecersiz", True
                    continue
                self.state[key] = val
                p["durum"], p["uygulama_tarih"] = "uygulandi", _stamp()
                p["uygulama_ts"] = _now().isoformat(timespec="seconds")
                applied.append(key)
                changed = True
            return list(applied), changed

        self._update(fn, "proposals: uygulandi [skip ci]")
        for key in PARAMS:
            if key in self.state:
                fixed = _norm(key, self.state[key])
                if fixed is not None and fixed != self.state[key]:
                    self.state[key] = fixed
                    applied.append(f"{key}(duzeltildi)")
        return list(applied)
