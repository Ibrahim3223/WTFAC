# autoshorts_daily.py — Topic-locked Gemini • Per-video search_terms • Robust Pexels
# Captions: ALL CAPS + karaoke (kelime highlight) — drawtext/subtitles fallback
# -*- coding: utf-8 -*-
import os, sys, re, json, time, random, datetime, tempfile, pathlib, subprocess, hashlib, math, shutil
from typing import List, Optional, Tuple, Dict, Any, Set

# ---- novelty guard (30g anti-repeat + semantic) ----
from novelty_guard import NoveltyGuard  # novelty_guard.py eklendiğine göre direkt içeri alıyoruz
STATE_DIR = os.getenv("STATE_DIR", ".state")
from state_guard import StateGuard  # <-- YENİ

# ---- focus-entity cooldown (stronger anti-repeat) ----
ENTITY_COOLDOWN_DAYS = int(os.getenv("ENTITY_COOLDOWN_DAYS", os.getenv("NOVELTY_WINDOW", "30")))

_GENERIC_SKIP = {
    # very common generic words to ignore for entity extraction
    "country","countries","people","history","stories","story","facts","fact","amazing","weird","random","culture","cultural",
    "animal","animals","nature","wild","pattern","patterns","science","eco","habit","habits","waste","tip","tips","daily","news",
    "world","today","minute","short","video","watch","more","better","twist","comment","voice","narration","hook","topic",
    "secret","secrets","unknown","things","life","lived","modern","time","times","explained","guide","quick","fix","fixes",
    "color","colors","skin","cells","cell","temperature","light","lights","effect","effects",
    "land","nation","state","city","capital","border","flag","heritage",
    "travel","tourism","planet","earth","place","region","area"
}

def _tok_words_loose(s: str) -> List[str]:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return [w for w in s.split() if len(w) >= 3]

def _derive_focus_entity(topic: str, mode: str, sentences: list[str]) -> str:
    """
    Heuristic entity pick used for cooldown:
      - For 'country_' modes: prefer proper nouns / frequent tokens (e.g., 'japan')
      - For animal/nature modes: frequent concrete noun (e.g., 'octopus','chameleon')
      - Else: most frequent non-generic keyword (e.g., 'food waste' -> 'waste')
    """
    txt = " ".join(sentences or []) + " " + (topic or "")
    words = _tok_words_loose(txt)
    words = [w[:-1] if w.endswith("s") and len(w) >= 5 else w for w in words]
    from collections import Counter as _C
    cnt = _C([w for w in words if w not in _GENERIC_SKIP])
    if not cnt:
        return ""
    # try bigrams first for eco patterns like 'food waste'
    bigrams = _C([" ".join(words[i:i+2]) for i in range(len(words)-1)])
    for bg,_ in bigrams.most_common(10):
        if all(w not in _GENERIC_SKIP for w in bg.split()) and len(bg) >= 7:
            parts = [w for w in bg.split() if w not in _GENERIC_SKIP]
            if parts:
                return parts[-1]
    # fallback to unigram
    for w,_ in cnt.most_common(20):
        if len(w) >= 4:
            return w
    return next(iter(cnt.keys())) if cnt else ""

def _entity_key(mode: str, ent: str) -> str:
    ent = re.sub(r"[^a-z0-9]+","-", (ent or "").lower()).strip("-")
    mode = (mode or "").lower()
    return f"{mode}:{ent}" if ent else ""

def _entities_state_load() -> dict:
    try:
        gst = _global_topics_load()
    except Exception:
        gst = {}
    ents = (gst.get("entities") if isinstance(gst, dict) else None) or {}
    if not isinstance(ents, dict): ents = {}
    return ents

def _entities_state_save(ents: dict):
    try:
        gst = _global_topics_load()
    except Exception:
        gst = {}
    if isinstance(gst, dict):
        gst["entities"] = ents
        # cap total to avoid unbounded growth
        if len(ents) > 12000:
            oldest = sorted(ents.items(), key=lambda kv: kv[1])[:2000]
            for k,_ in oldest: ents.pop(k, None)
            gst["entities"] = ents
        _global_topics_save(gst)

def _entity_in_cooldown(key: str, days: int) -> bool:
    if not key or days <= 0: 
        return False
    ents = _entities_state_load()
    ts = ents.get(key)
    if not ts: 
        return False
    try:
        age = time.time() - float(ts)
    except Exception:
        return False
    return age < days * 86400

def _entity_touch(key: str):
    if not key: 
        return
    ents = _entities_state_load()
    ents[key] = time.time()
    _entities_state_save(ents)

# ---------- helpers (ÖNCE gelmeli) ----------
def _env_int(name: str, default: int) -> int:
    s = os.getenv(name)
    if s is None: return default
    s = str(s).strip()
    if s == "" or s.lower() == "none": return default
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))  # "68.0" gibi değerler
        except Exception:
            return default

def _env_float(name: str, default: float) -> float:
    s = os.getenv(name)
    if s is None: return default
    s = str(s).strip()
    if s == "" or s.lower() == "none": return default
    try:
        return float(s)
    except Exception:
        return default

def _sanitize_lang(val: Optional[str]) -> str:
    val = (val or "").strip()
    if not val: return "en"
    m = re.match(r"([A-Za-z]{2})", val)
    return (m.group(1).lower() if m else "en")

def _sanitize_privacy(val: Optional[str]) -> str:
    v = (val or "").strip().lower()
    return v if v in {"public", "unlisted", "private"} else "public"

KARAOKE_OFFSET_MS = int(os.getenv("KARAOKE_OFFSET_MS", "0"))
KARAOKE_SPEED = float(os.getenv("KARAOKE_SPEED", "1.0"))

def _adj_time(t_seconds: float) -> float:
    """
    Vurgu zamanlarını topluca öne/al ve çok küçük bir hız düzeltmesi uygula.
    Negatif offset => daha erken vurgu.
    """
    return max(0.0, (t_seconds + KARAOKE_OFFSET_MS / 1000.0) / max(KARAOKE_SPEED, 1e-6))


# ==================== ENV / constants ====================
VOICE_STYLE    = os.getenv("TTS_STYLE", "narration-professional")
TARGET_MIN_SEC = _env_float("TARGET_MIN_SEC", 22.0)
TARGET_MAX_SEC = _env_float("TARGET_MAX_SEC", 42.0)

CHANNEL_NAME   = os.getenv("CHANNEL_NAME", "DefaultChannel")
MODE           = os.getenv("MODE", "freeform").strip().lower()

LANG           = _sanitize_lang(os.getenv("VIDEO_LANG") or os.getenv("LANG") or "en")
VISIBILITY     = _sanitize_privacy(os.getenv("VISIBILITY"))
ROTATION_SEED  = _env_int("ROTATION_SEED", 0)
REQUIRE_CAPTIONS = os.getenv("REQUIRE_CAPTIONS", "0") == "1"
KARAOKE_CAPTIONS = os.getenv("KARAOKE_CAPTIONS", "1") == "1"

# ---- Entity görsel kapsama kontrolü ----
ENTITY_VISUAL_MIN = _env_float("ENTITY_VISUAL_MIN", 0.95)  # sahnelerin en az %50'si odak varlıkla alakalı olsun
ENTITY_VISUAL_STRICT = os.getenv("ENTITY_VISUAL_STRICT", "1") == "1"  # 1=eksikse agresif tamamla

SCENE_QUERY_MODE = os.getenv("SCENE_QUERY_MODE", "entity").strip().lower()

# Karaoke renkleri (ASS stili)
KARAOKE_ACTIVE   = os.getenv("KARAOKE_ACTIVE",   "#3EA6FF")
KARAOKE_INACTIVE = os.getenv("KARAOKE_INACTIVE", "#FFD700")
KARAOKE_OUTLINE  = os.getenv("KARAOKE_OUTLINE",  "#000000")
CAPTION_LEAD_MS  = int(os.getenv("CAPTION_LEAD_MS", "60"))

OUT_DIR        = "out"; pathlib.Path(OUT_DIR).mkdir(exist_ok=True)

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
USE_GEMINI     = os.getenv("USE_GEMINI", "1") == "1"
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_PROMPT  = (os.getenv("GEMINI_PROMPT") or "").strip()
GEMINI_TEMP    = _env_float("GEMINI_TEMP", 0.85)

# ---- Contextual CTA (comments-focused) ----
CTA_ENABLE      = os.getenv("CTA_ENABLE", "1") == "1"
CTA_SHOW_SEC    = _env_float("CTA_SHOW_SEC", 2.8)     # CTA sadece son X sn görünsün
CTA_MAX_CHARS   = _env_int("CTA_MAX_CHARS", 64)       # overlay kısalığı
CTA_TEXT_FORCE  = (os.getenv("CTA_TEXT") or "").strip()  # elle override istersek

# ---- Topic & user seed terms ----
TOPIC_RAW = os.getenv("TOPIC", "").strip()
TOPIC = re.sub(r'^[\'"]|[\'"]$', '', TOPIC_RAW).strip()

def _parse_terms(s: str) -> List[str]:
    s = (s or "").strip()
    if not s: return []
    try:
        data = json.loads(s)
        if isinstance(data, list): return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    s = re.sub(r'^[\[\(]|\s*[\]\)]$', '', s)
    parts = re.split(r'\s*,\s*', s)
    return [p.strip().strip('"').strip("'") for p in parts if p.strip()]

SEARCH_TERMS_ENV = _parse_terms(os.getenv("SEARCH_TERMS", ""))

TARGET_FPS       = 25
CRF_VISUAL       = 22

CAPTION_MAX_LINE  = int(os.getenv("CAPTION_MAX_LINE",  "28"))
CAPTION_MAX_LINES = int(os.getenv("CAPTION_MAX_LINES", "6"))

# ---------- Pexels ayarları ----------
PEXELS_PER_PAGE            = int(os.getenv("PEXELS_PER_PAGE", "30"))
PEXELS_MAX_USES_PER_CLIP   = int(os.getenv("PEXELS_MAX_USES_PER_CLIP", "1"))
# Varsayılan: tekrar YOK. Gerekirse PEXELS_ALLOW_REUSE=1
PEXELS_ALLOW_REUSE         = os.getenv("PEXELS_ALLOW_REUSE", "0") == "1"
PEXELS_ALLOW_LANDSCAPE     = os.getenv("PEXELS_ALLOW_LANDSCAPE", "1") == "1"
PEXELS_MIN_DURATION        = int(os.getenv("PEXELS_MIN_DURATION", "3"))
PEXELS_MAX_DURATION        = int(os.getenv("PEXELS_MAX_DURATION", "13"))
PEXELS_MIN_HEIGHT          = int(os.getenv("PEXELS_MIN_HEIGHT",   "720"))
PEXELS_STRICT_VERTICAL     = os.getenv("PEXELS_STRICT_VERTICAL", "0") == "1"

ALLOW_PIXABAY_FALLBACK     = os.getenv("ALLOW_PIXABAY_FALLBACK", "1") == "1"
PIXABAY_API_KEY            = os.getenv("PIXABAY_API_KEY", "").strip()

# ---- State dosyaları (legacy uyumlu) ----
STATE_FILE = f"state_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json"
GLOBAL_TOPIC_STATE = "state_global_topics.json"
LEGACY_STATE_FILE = f"state_{CHANNEL_NAME}.json"
LEGACY_GLOBAL_STATE = "state_global.json"

# === NOVELTY (tekrar engelleme) — ENV ===
NOVELTY_ENFORCE       = os.getenv("NOVELTY_ENFORCE", "1") == "1"
NOVELTY_WINDOW        = _env_int("NOVELTY_WINDOW", 40)
NOVELTY_JACCARD_MAX   = _env_float("NOVELTY_JACCARD_MAX", 0.55)
NOVELTY_RETRIES       = _env_int("NOVELTY_RETRIES", 4)

# === BGM (arka müzik) — ENV ===
BGM_ENABLE  = os.getenv("BGM_ENABLE", "0") == "1"
BGM_DB      = _env_float("BGM_DB", -26.0)          # temel müzik seviyesi (dB)
BGM_DUCK_DB = _env_float("BGM_DUCK_DB", -12.0)     # konuşmada kısılacak miktar (dB) — sidechaincompress ile
BGM_FADE    = _env_float("BGM_FADE", 0.8)          # giriş/çıkış fade saniyesi
BGM_DIR     = os.getenv("BGM_DIR", "bgm").strip()
BGM_URLS    = _parse_terms(os.getenv("BGM_URLS", ""))  # JSON/virgül listesini destekler

# ==================== deps (auto-install) ====================
def _pip(p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=True)
try:
    import requests
except ImportError:
    _pip("requests"); import requests
try:
    import edge_tts, nest_asyncio
except ImportError:
    _pip("edge-tts"); _pip("nest_asyncio"); import edge_tts, nest_asyncio
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:
    _pip("google-api-python-client"); from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
except ImportError:
    _pip("google-auth"); from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError

# ==================== Voices ====================
VOICE_OPTIONS = {
    "en": [
        "en-US-JennyNeural","en-US-JasonNeural","en-US-AriaNeural","en-US-GuyNeural",
        "en-AU-NatashaNeural","en-GB-SoniaNeural","en-CA-LiamNeural","en-US-DavisNeural","en-US-AmberNeural"
    ],
    "tr": ["tr-TR-EmelNeural","tr-TR-AhmetNeural"]
}
VOICE = os.getenv("TTS_VOICE", VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])[0])

# ==================== Utils ====================
def run(cmd, check=True):
    res = subprocess.run(cmd, text=True, capture_output=True)
    if check and res.returncode != 0:
        raise RuntimeError(res.stderr[:4000])
    return res

def ffprobe_dur(p):
    try:
        out = run(["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0", p]).stdout.strip()
        return float(out) if out else 0.0
    except:
        return 0.0

def ffmpeg_has_filter(name: str) -> bool:
    try:
        out = run(["ffmpeg","-hide_banner","-filters"], check=False).stdout
        return bool(re.search(rf"\b{name}\b", out))
    except Exception:
        return False

_HAS_DRAWTEXT   = ffmpeg_has_filter("drawtext")
_HAS_SUBTITLES  = ffmpeg_has_filter("subtitles")

def font_path():
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "C:/Windows/Fonts/arial.ttf"]:
        if pathlib.Path(p).exists():
            return p
    return ""

def _ff_sanitize_font(font_path_str: str) -> str:
    if not font_path_str: return ""
    return font_path_str.replace(":", r"\:").replace(",", r"\,").replace("\\", "/")

def normalize_sentence(raw: str) -> str:
    s = (raw or "").strip()
    s = s.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(re.sub(r"\s+", " ", ln).strip() for ln in s.split("\n"))
    s = s.replace("—", "-").replace("–", "-").replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    return s

# ---------- CTA keyword helpers ----------
_STOP_EN = set("the a an and or but if while of to in on at from by with for about into over after before between during under above across around through this that these those is are was were be been being have has had do does did can could should would may might will your you we our they their he she it its as than then so very more most many much just also only even still yet".split())
_STOP_TR = set("ve ya ama eğer iken ile için üzerine altında üzerinde arasında boyunca sonra önce boyunca altında üstünde hakkında üzerinden arasında bu şu o bir birisi şunlar bunlar biz siz onlar var yok çok daha en ise çünkü gibi kadar zaten sadece yine hâlâ".split())

def _kw_tokens(text: str, lang: str) -> list[str]:
    t = re.sub(r"[^A-Za-zçğıöşüÇĞİÖŞÜ0-9 ]+", " ", (text or "")).lower()
    ws = [w for w in t.split() if len(w) >= 4 and w not in (_STOP_TR if lang.startswith("tr") else _STOP_EN)]
    return ws

def _top_keywords(topic: str, sentences: list[str], lang: str, k: int = 6) -> list[str]:
    from collections import Counter
    cnt = Counter()
    for s in [topic] + list(sentences or []):
        for w in _kw_tokens(s, lang):
            cnt[w] += 1
    # iki kelimelik öbekleri de dene
    bigr = Counter()
    toks_all = _kw_tokens(" ".join([topic] + sentences), lang)
    for i in range(len(toks_all)-1):
        bigr[toks_all[i] + " " + toks_all[i+1]] += 1
    # skoru: (bigr*2) + unigram
    scored = []
    for w,c in cnt.items():
        scored.append((c, w))
    for bg,c in bigr.items():
        scored.append((c*2, bg))
    scored.sort(reverse=True)
    out=[]
    for _,w in scored:
        if w not in out:
            out.append(w)
        if len(out) >= k: break
    return out

def build_contextual_cta(topic: str, sentences: list[str], lang: str) -> str:
    """Return short, video-specific, comments-oriented CTA (no 'subscribe/like')."""
    if CTA_TEXT_FORCE:
        return CTA_TEXT_FORCE.strip()

    kws = _top_keywords(topic or "", sentences or [], lang)
    # En az bir/iki aday
    a = (kws[0] if kws else (topic or "").lower())
    b = (kws[1] if len(kws) > 1 else "")
    rng = random.Random((ROTATION_SEED or int(time.time())) + len("".join(sentences)))

    if lang.startswith("tr"):
        templates = [
            lambda a,b: f"Sence en şaşırtan neydi: {a} mı {b} mi?" if b else f"Sence en şaşırtan neydi: {a}?",
            lambda a,b: f"{a} için daha iyi bir fikir var mı? Yorumla!",
            lambda a,b: f"İlk hangisini denerdin: {a} mı {b} mi?" if b else f"{a} sence işe yarar mı?",
            lambda a,b: f"3 kelimeyle yorumla: {a}",
            lambda a,b: f"Detayı yakaladın mı? Nerede? Yaz 📝"
        ]
    else:
        templates = [
            lambda a,b: f"Which surprised you more: {a} or {b}?" if b else f"What surprised you most about {a}?",
            lambda a,b: f"Got a smarter fix for {a}? Drop it below!",
            lambda a,b: f"First pick: {a} or {b}?" if b else f"Would you try {a} first?",
            lambda a,b: f"Sum it up in 3 words: {a}",
            lambda a,b: f"Spot the tiny clue? Where? Comment!"
        ]

    # Seç, kısalt, biçimle
    for _ in range(10):
        t = templates[rng.randrange(len(templates))](a, b).strip()
        t = re.sub(r"\s+", " ", t)
        if len(t) <= CTA_MAX_CHARS:
            return t
    return (templates[0](a,b))[:CTA_MAX_CHARS]

# ==================== State ====================
def _load_json(path, default):
    try: return json.load(open(path, "r", encoding="utf-8"))
    except: return default

def _save_json(path, data):
    txt = json.dumps(data, indent=2, ensure_ascii=False)
    pathlib.Path(path).write_text(txt, encoding="utf-8")
    # Legacy eş-yazım (cache uyumluluğu)
    try:
        if path == STATE_FILE:
            pathlib.Path(LEGACY_STATE_FILE).write_text(txt, encoding="utf-8")
        if path == GLOBAL_TOPIC_STATE:
            pathlib.Path(LEGACY_GLOBAL_STATE).write_text(txt, encoding="utf-8")
    except Exception:
        pass

def _state_load() -> dict:
    # Önce modern dosya, yoksa legacy'den yükleyip promote et
    if pathlib.Path(STATE_FILE).exists():
        return _load_json(STATE_FILE, {"recent": [], "used_pexels_ids": []})
    if pathlib.Path(LEGACY_STATE_FILE).exists():
        st = _load_json(LEGACY_STATE_FILE, {"recent": [], "used_pexels_ids": []})
        _save_json(STATE_FILE, st)
        return st
    return {"recent": [], "used_pexels_ids": []}

def _state_save(st: dict):
    st["recent"] = st.get("recent", [])[-1200:]
    st["used_pexels_ids"] = st.get("used_pexels_ids", [])[-5000:]
    _save_json(STATE_FILE, st)

def _global_topics_load() -> dict:
    default = {"recent_topics": []}
    if pathlib.Path(GLOBAL_TOPIC_STATE).exists():
        return _load_json(GLOBAL_TOPIC_STATE, default)
    if pathlib.Path(LEGACY_GLOBAL_STATE).exists():
        gst = _load_json(LEGACY_GLOBAL_STATE, default)
        _save_json(GLOBAL_TOPIC_STATE, gst)
        return gst
    return default

def _global_topics_save(gst: dict):
    gst["recent_topics"] = gst.get("recent_topics", [])[-4000:]
    _save_json(GLOBAL_TOPIC_STATE, gst)

def _hash12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _record_recent(h: str, mode: str, topic: str, fp: Optional[List[str]] = None):
    st = _state_load()
    rec = {"h":h,"mode":mode,"topic":topic,"ts":time.time()}
    if fp: rec["fp"] = list(fp)
    st.setdefault("recent", []).append(rec)
    _state_save(st)
    gst = _global_topics_load()
    if topic and topic not in gst["recent_topics"]:
        gst["recent_topics"].append(topic)
        _global_topics_save(gst)

def _blocklist_add_pexels(ids: List[int], days=30):
    st = _state_load()
    now = int(time.time())
    for vid in ids:
        st.setdefault("used_pexels_ids", []).append({"id": int(vid), "ts": now})
    cutoff = now - days*86400
    st["used_pexels_ids"] = [x for x in st.get("used_pexels_ids", []) if x.get("ts",0) >= cutoff]
    _state_save(st)

def _blocklist_get_pexels() -> set:
    st = _state_load()
    return {int(x["id"]) for x in st.get("used_pexels_ids", [])}

def _recent_topics_for_prompt(limit=20) -> List[str]:
    gst = _global_topics_load()
    topics = list(reversed(gst.get("recent_topics", [])))
    uniq=[]
    for t in topics:
        if t and t not in uniq: uniq.append(t)
        if len(uniq) >= limit: break
    return uniq

# ---- novelty helpers ----
def _tok_words(s: str) -> List[str]:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return [w for w in s.split() if len(w) >= 3]

def _trigrams(words: List[str]) -> Set[str]:
    return {" ".join(words[i:i+3]) for i in range(len(words)-2)} if len(words) >= 3 else set()

def _sentences_fp(sentences: List[str]) -> Set[str]:
    ws = _tok_words(" ".join(sentences or []))
    return _trigrams(ws)

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return (inter / union) if union else 0.0

def _recent_fps_from_state(limit: int = NOVELTY_WINDOW) -> List[Set[str]]:
    st = _state_load()
    out=[]
    for item in reversed(st.get("recent", [])):
        fp = item.get("fp")
        if isinstance(fp, list):
            out.append(set(fp))
        if len(out) >= limit: break
    return out

def _novelty_ok(sentences: List[str]) -> Tuple[bool, List[str]]:
    """Dön: (yeterince yeni mi?, kaçınma-terimleri)"""
    if not NOVELTY_ENFORCE:
        return True, []
    cur = _sentences_fp(sentences)
    if not cur: return True, []
    for fp in _recent_fps_from_state(NOVELTY_WINDOW):
        sim = _jaccard(cur, fp)
        if sim > NOVELTY_JACCARD_MAX:
            common = list(cur & fp)
            terms = []
            for tri in common[:40]:
                for w in tri.split():
                    if len(w) >= 4 and w not in terms:
                        terms.append(w)
                    if len(terms) >= 12: break
                if len(terms) >= 12: break
            return False, terms
    return True, []

# ==================== Caption helpers ====================
CAPTION_COLORS = ["0xFFD700","0xFF6B35","0x00F5FF","0x32CD32","0xFF1493","0x1E90FF","0xFFA500","0xFF69B4"]

def _ff_color(c: str) -> str:
    c = (c or "").strip()
    if c.startswith("#"): return "0x" + c[1:].upper()
    if re.fullmatch(r"0x[0-9A-Fa-f]{6}", c): return c
    return "white"

def clean_caption_text(s: str) -> str:
    t = (s or "").strip()
    t = (t.replace("—", "-").replace("–", "-").replace("“", '"').replace("”", '"').replace("’", "'").replace("`",""))
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t

def wrap_mobile_lines(text: str, max_line_length: int = CAPTION_MAX_LINE, max_lines: int = CAPTION_MAX_LINES) -> str:
    text = (text or "").strip()
    if not text: return text
    words = text.split()
    HARD_CAP = max_lines + 2
    def distribute_into(k: int) -> list[str]:
        per = math.ceil(len(words) / k)
        chunks = [" ".join(words[i*per:(i+1)*per]) for i in range(k)]
        return [c for c in chunks if c]
    for k in range(2, max_lines + 1):
        cand = distribute_into(k)
        if cand and all(len(c) <= max_line_length for c in cand):
            return "\n".join(cand)
    def greedy(width: int, k_cap: int) -> list[str]:
        lines=[]; buf=[]; L=0
        for w in words:
            add=(1 if buf else 0)+len(w)
            if L+add>width and buf:
                lines.append(" ".join(buf)); buf=[w]; L=len(w)
            else:
                buf.append(w); L+=add
        if buf: lines.append(" ".join(buf))
        if len(lines)>k_cap and k_cap<HARD_CAP: return greedy(width, HARD_CAP)
        return lines
    lines = greedy(max_line_length, max_lines)
    return "\n".join([ln.strip() for ln in lines if ln.strip()])

# ==================== TTS ====================
def _rate_to_atempo(rate_str: str, default: float = 1.10) -> float:
    try:
        if not rate_str: return default
        rate_str = rate_str.strip()
        if rate_str.endswith("%"):
            val = float(rate_str.replace("%","")); return max(0.5, min(2.0, 1.0 + val/100.0))
        if rate_str.endswith(("x","X")):
            return max(0.5, min(2.0, float(rate_str[:-1])))
        v = float(rate_str); return max(0.5, min(2.0, v))
    except Exception:
        return default

def _edge_stream_tts(text: str, voice: str, rate_env: str, mp3_out: str) -> List[Dict[str,Any]]:
    """Edge-TTS stream → mp3 bytes + word boundaries. Returns marks list [{t0, t1, text}] in seconds."""
    import asyncio
    marks: List[Dict[str,Any]] = []
    async def _run():
        audio = bytearray()
        comm = edge_tts.Communicate(text, voice=voice, rate=rate_env)
        async for chunk in comm.stream():
            t = chunk.get("type")
            if t == "audio":
                audio.extend(chunk.get("data", b""))
            elif t == "WordBoundary":
                off = float(chunk.get("offset", 0))/10_000_000.0
                dur = float(chunk.get("duration",0))/10_000_000.0
                marks.append({"t0": off, "t1": off+dur, "text": str(chunk.get("text",""))})
        open(mp3_out, "wb").write(bytes(audio))
    try:
        asyncio.run(_run())
    except RuntimeError:
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_run())
    return marks

def _merge_marks_to_words(text: str, marks: List[Dict[str,Any]], total: float) -> List[Tuple[str,float]]:
    """
    Edge-TTS word boundaries'lerini kullanarak kelime sürelerini hesapla.
    Args:
        text: Orijinal metin
        marks: Edge-TTS'den gelen [{t0, t1, text}] listesi (atempo ÖNCESİ)
        total: Final audio süresi (atempo SONRASI)
    Returns:
        [(WORD, duration_seconds), ...]
    
    Strateji: Edge-TTS boundary'lerini OLDUĞU GİBİ scale et, ekstra düzeltme yapma.
    """
    words = [w for w in re.split(r"\s+", (text or "").strip()) if w]
    if not words:
        return []
    
    out = []
    
    # Edge-TTS marks varsa ve güvenilirse
    if marks and len(marks) >= len(words) * 0.7:  # En az %70 coverage
        N = min(len(words), len(marks))
        
        # Ham süreler (atempo öncesi)
        raw_durs = [max(0.05, float(marks[i]["t1"] - marks[i]["t0"])) for i in range(N)]
        sum_raw = sum(raw_durs)
        
        # Scale factor: atempo sonrası gerçek süreye uyarla
        scale = (total / sum_raw) if sum_raw > 0 else 1.0
        
        # Scale edilmiş süreler
        for i in range(N):
            scaled_dur = max(0.08, raw_durs[i] * scale)  # min 80ms
            out.append((words[i], scaled_dur))
        
        # Kalan kelimeler varsa (N < len(words))
        if len(words) > N:
            used_time = sum(d for _, d in out)
            remain = max(0.0, total - used_time)
            each = remain / (len(words) - N) if (len(words) - N) > 0 else 0.1
            
            for i in range(N, len(words)):
                out.append((words[i], max(0.08, each)))
        
        # Son düzeltme: toplam süreyi garanti et
        current_total = sum(d for _, d in out)
        if abs(current_total - total) > 0.05:  # 50ms'den fazla fark varsa
            diff = total - current_total
            # Farkı son kelimeye ekle/çıkar
            if out:
                last_word, last_dur = out[-1]
                out[-1] = (last_word, max(0.08, last_dur + diff))
    
    else:
        # Fallback: Marks yok/yetersiz → equal split
        each = max(0.08, total / max(1, len(words)))
        out = [(w, each) for w in words]
        
        # Son kelimeyi düzelt
        if out:
            current_sum = sum(d for _, d in out)
            diff = total - current_sum
            if abs(diff) > 0.01:  # 10ms'den fazla fark
                last_word, last_dur = out[-1]
                out[-1] = (last_word, max(0.08, last_dur + diff))
    
    return out

def tts_to_wav(text: str, wav_out: str) -> Tuple[float, List[Tuple[str,float]]]:
    """
    Returns (duration_seconds, word_durations_list) where list = [(WORD, seconds), ...]
    - Edge-TTS stream ile word boundaries yakala
    - Universal SSML prosody (tüm konular için)
    - Atempo uygula
    - Word durations'ı final duration'a scale et
    """
    import asyncio
    from aiohttp.client_exceptions import WSServerHandshakeError
    
    text = (text or "").strip()
    if not text:
        run(["ffmpeg","-y","-f","lavfi","-t","1.0","-i","anullsrc=r=48000:cl=mono", wav_out])
        return 1.0, []

    # ===== UNIVERSAL SSML ENHANCEMENT =====
    use_ssml = os.getenv("TTS_SSML", "1") == "1"
    
    if use_ssml:
        # Numbers → always emphasize (works for any topic)
        text = re.sub(r'\b(\d+)\b', r'<emphasis level="strong">\1</emphasis>', text)
        
        # Question words → pitch up (universal)
        text = re.sub(
            r'\b(Why|What|How|When|Where|Who|Which)\b', 
            r'<prosody pitch="+5%">\1</prosody>', 
            text, flags=re.IGNORECASE
        )
        
        # Exclamation → volume up (universal emotion)
        if '!' in text:
            parts = text.split('!')
            enhanced = []
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    enhanced.append(f'<prosody volume="+10%">{part}</prosody>!')
                else:
                    enhanced.append(part)
            text = ''.join(enhanced)
        
        # Contrast words → pause before (universal drama)
        contrast_words = ['but', 'however', 'although', 'yet', 'still', 'despite', 'actually']
        for word in contrast_words:
            text = re.sub(
                rf'\b({word})\b', 
                rf'<break time="150ms"/>\1', 
                text, flags=re.IGNORECASE
            )
        
        # Action verbs → moderate stress (universal energy)
        action_pattern = r'\b(makes|shows|reveals|changes|stops|starts|turns|becomes|creates|breaks|moves)\b'
        text = re.sub(
            action_pattern, 
            r'<emphasis level="moderate">\1</emphasis>', 
            text, flags=re.IGNORECASE
        )
        
        # Final sentence → slight slowdown (universal closure)
        sentences_split = re.split(r'[.!?]', text)
        if len(sentences_split) > 1 and sentences_split[-2].strip():
            last = sentences_split[-2].strip()
            # Sadece ilk bulduğunu değiştir
            text = text.replace(last, f'<prosody rate="-5%">{last}</prosody>', 1)

    mp3 = wav_out.replace(".wav", ".mp3")
    
    # TTS hızı ENV'den (workflow'da kontrol edilir)
    rate_env = os.getenv("TTS_RATE", "+10%")
    atempo = _rate_to_atempo(rate_env, default=1.10)
    
    available = VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])
    selected_voice = VOICE if VOICE in available else available[0]
    
    marks: List[Dict[str,Any]] = []
    
    # TRY 1: Edge-TTS stream (word boundaries ile)
    try:
        marks = _edge_stream_tts(text, selected_voice, rate_env, mp3)
        run([
            "ffmpeg","-y","-hide_banner","-loglevel","error",
            "-i", mp3,
            "-ar","48000","-ac","1","-acodec","pcm_s16le",
            "-af", f"dynaudnorm=g=7:f=250,atempo={atempo}",
            wav_out
        ])
        pathlib.Path(mp3).unlink(missing_ok=True)
        dur = ffprobe_dur(wav_out) or 0.0
        words = _merge_marks_to_words(text, marks, dur)
        ssml_status = "SSML" if use_ssml else "plain"
        print(f"   TTS: {len(words)} words | {dur:.2f}s | atempo={atempo:.2f} | {ssml_status}")
        return dur, words
        
    except WSServerHandshakeError as e:
        if getattr(e, "status", None) != 401 and "401" not in str(e):
            print(f"⚠️ edge-tts stream fail: {e}")
    except Exception as e:
        print(f"⚠️ edge-tts stream fail: {e}")

    # FALLBACK 1: Edge-TTS save (no marks)
    try:
        async def _edge_save_simple():
            comm = edge_tts.Communicate(text, voice=selected_voice, rate=rate_env)
            await comm.save(mp3)
        
        try:
            asyncio.run(_edge_save_simple())
        except RuntimeError:
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_edge_save_simple())
        
        run([
            "ffmpeg","-y","-hide_banner","-loglevel","error",
            "-i", mp3,
            "-ar","48000","-ac","1","-acodec","pcm_s16le",
            "-af", f"dynaudnorm=g=7:f=300,atempo={atempo}",
            wav_out
        ])
        pathlib.Path(mp3).unlink(missing_ok=True)
        dur = ffprobe_dur(wav_out) or 0.0
        words = _merge_marks_to_words(text, [], dur)
        print(f"   TTS (no marks): {len(words)} words | {dur:.2f}s | atempo={atempo:.2f}")
        return dur, words
        
    except Exception as e:
        print(f"⚠️ edge-tts 401 → hızlı fallback TTS")

    # FALLBACK 2: Google TTS (no marks, no SSML support)
    try:
        # SSML etiketlerini temizle (Google desteklemiyor)
        clean_text = re.sub(r'<[^>]+>', '', text) if use_ssml else text
        
        q = requests.utils.quote(clean_text.replace('"','').replace("'",""))
        lang_code = LANG or "en"
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={q}&tl={lang_code}&client=tw-ob&ttsspeed=1.0"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        open(mp3, "wb").write(r.content)
        
        run([
            "ffmpeg","-y","-hide_banner","-loglevel","error",
            "-i", mp3,
            "-ar","48000","-ac","1","-acodec","pcm_s16le",
            "-af", f"dynaudnorm=g=6:f=300,atempo={atempo}",
            wav_out
        ])
        pathlib.Path(mp3).unlink(missing_ok=True)
        dur = ffprobe_dur(wav_out) or 0.0
        words = _merge_marks_to_words(clean_text, [], dur)
        print(f"   TTS (Google fallback): {len(words)} words | {dur:.2f}s")
        return dur, words
        
    except Exception as e2:
        print(f"❌ TTS tüm yollar başarısız, sessizlik üretilecek: {e2}")
        run(["ffmpeg","-y","-f","lavfi","-t","4.0","-i","anullsrc=r=48000:cl=mono", wav_out])
        return 4.0, []

# ==================== Video helpers ====================
def quantize_to_frames(seconds: float, fps: int = TARGET_FPS) -> Tuple[int, float]:
    frames = max(2, int(round(seconds * fps)))
    return frames, frames / float(fps)

def make_segment(src: str, dur_s: float, outp: str):
    """Universal motion effects (adapts to any content)"""
    frames, qdur = quantize_to_frames(dur_s, TARGET_FPS)
    fade = max(0.05, min(0.12, qdur/8.0))
    fade_out_st = max(0.0, qdur - fade)
    
    # ===== ADAPTIVE MOTION =====
    use_motion = os.getenv("VIDEO_MOTION", "1") == "1"
    motion_intensity = os.getenv("MOTION_INTENSITY", "subtle").lower()  # subtle/moderate/dynamic
    
    if use_motion and qdur > 2.5:
        # Intensity mapping
        if motion_intensity == "dynamic":
            zoom_range = (1.0, 1.20)
            speed = 0.002
        elif motion_intensity == "moderate":
            zoom_range = (1.0, 1.12)
            speed = 0.0015
        else:  # subtle
            zoom_range = (1.0, 1.08)
            speed = 0.001
        
        # Random motion (works for any footage)
        motion_types = ['zoom_in', 'zoom_out', 'pan_right', 'pan_left', 'static']
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Prefer zoom in
        motion_type = random.choices(motion_types, weights=weights)[0]
        
        if motion_type == 'zoom_in':
            zoom = f"zoompan=z='min({zoom_range[1]},1+{speed}*on)':d={frames}:s=1080x1920"
        elif motion_type == 'zoom_out':
            zoom = f"zoompan=z='max(1.0,{zoom_range[1]}-{speed}*on)':d={frames}:s=1080x1920"
        elif motion_type in ['pan_right', 'pan_left']:
            direction = '+' if motion_type == 'pan_right' else '-'
            zoom = f"zoompan=z=1.15:x='iw/2-(iw/zoom/2){direction}min(iw/zoom-iw,iw*{speed}*on)':d={frames}:s=1080x1920"
        else:
            zoom = ""
    else:
        zoom = ""
    
    # Build filter chain
    base_filters = [
        "scale=1080:1920:force_original_aspect_ratio=increase",
        "crop=1080:1920",
    ]
    
    if zoom:
        base_filters.append(zoom)
    
    base_filters.extend([
        "setsar=1",
        "eq=brightness=0.02:contrast=1.08:saturation=1.1",
        f"fps={TARGET_FPS}",
        f"setpts=N/{TARGET_FPS}/TB",
        f"trim=start_frame=0:end_frame={frames}",
        f"fade=t=in:st=0:d={fade:.2f}",
        f"fade=t=out:st={fade_out_st:.2f}:d={fade:.2f}"
    ])
    
    vf = ",".join(base_filters)
    
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", src,
        "-vf", vf,
        "-r", str(TARGET_FPS), "-vsync","cfr",
        "-an",
        "-c:v","libx264","-preset","fast","-crf",str(CRF_VISUAL),
        "-pix_fmt","yuv420p","-movflags","+faststart",
        outp
    ])

def enforce_video_exact_frames(video_in: str, target_frames: int, outp: str):
    target_frames = max(2, int(target_frames))
    vf = f"setsar=1,fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB,trim=start_frame=0:end_frame={target_frames}"
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", video_in,
        "-vf", vf,
        "-r", str(TARGET_FPS), "-vsync","cfr",
        "-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),
        "-pix_fmt","yuv420p","-movflags","+faststart",
        outp
    ])

def _ass_time(s: float) -> str:
    """Float seconds'ı ASS time formatına çevir (H:MM:SS.CS)"""
    h = int(s // 3600)
    s -= h * 3600
    m = int(s // 60)
    s -= m * 60
    return f"{h:d}:{m:02d}:{s:05.2f}"

def _build_karaoke_ass(text: str, seg_dur: float, words: List[Tuple[str,float]], is_hook: bool) -> str:
    """
    Karaoke-style ASS subtitle oluştur (Universal - tüm konular için).
    - Edge-TTS word boundaries'lerine güvenir
    - Minimal düzeltme yapar (SPEEDUP_PCT ve EARLY_END_MS)
    - Adaptive effects (subtle/moderate/dynamic)
    - RAMP ve LEAD kullanmaz (tutarlılık için)
    """
    # ASS renk notasyonu: &HAABBGGRR (A=alpha, BGR sırası)
    def _to_ass(c: str) -> str:
        c = c.strip()
        if c.startswith("0x"): c = c[2:]
        if c.startswith("#"):  c = c[1:]
        if len(c) == 6:        c = "00" + c
        rr, gg, bb = c[-6:-4], c[-4:-2], c[-2:]
        return f"&H00{bb}{gg}{rr}"

    fontname = "DejaVu Sans"
    fontsize = 58 if is_hook else 52
    margin_v = 270 if is_hook else 330
    outline  = 4  if is_hook else 3

    # Kelimeleri UPPERCASE + süreleri al
    words_upper = [(re.sub(r"\s+", " ", w.upper()), d) for w, d in words if str(w).strip()]
    if not words_upper:
        # Fallback: eşit bölme
        split_words = (text or "…").split()
        each_dur = seg_dur / max(1, len(split_words))
        words_upper = [(w.upper(), each_dur) for w in split_words]

    n = len(words_upper)

    # Hedef toplam süre (centisecond)
    total_cs = int(round(seg_dur * 100))

    # Ham süreler (centisecond) - Edge-TTS'den gelen verileri KORU
    ds = [max(8, int(round(d * 100))) for _, d in words_upper]  # min 80ms
    if sum(ds) == 0:
        ds = [max(8, int(total_cs / n)) for _ in range(n)]

    # --- SADECE MİNİMAL GLOBAL DÜZELTME ---
    try:
        speedup_pct = float(os.getenv("KARAOKE_SPEEDUP_PCT", "0.0"))
    except Exception:
        speedup_pct = 0.0
    speedup_pct = max(0.0, min(5.0, speedup_pct))

    try:
        early_end_ms = int(os.getenv("KARAOKE_EARLY_END_MS", "0"))
    except Exception:
        early_end_ms = 0
    early_end_cs = max(0, int(round(early_end_ms / 10.0)))

    # Target hesapla (minimal düzeltme)
    target_cs = int(round(total_cs * (1.0 - (speedup_pct / 100.0)))) - early_end_cs
    target_cs = max(8 * n, target_cs)  # min 80ms/kelime

    # Global scale (tüm kelimelere eşit oranda uygula)
    s = sum(ds)
    scale = 1.0
    if s > 0 and s != target_cs:
        scale = target_cs / s
        ds = [max(8, int(round(x * scale))) for x in ds]

    # Fine-tune: tam hedefe ulaş (1 cs hassasiyetle)
    while sum(ds) > target_cs and any(d > 8 for d in ds):
        for i in range(n):
            if sum(ds) <= target_cs: break
            if ds[i] > 8: ds[i] -= 1
    
    while sum(ds) < target_cs:
        # Uzun kelimelere öncelik ver (daha doğal)
        longest_idx = max(range(n), key=lambda i: ds[i])
        ds[longest_idx] += 1

    # ===== ADAPTIVE ASS EFFECTS (Universal) =====
    use_effects = os.getenv("KARAOKE_EFFECTS", "1") == "1"
    effect_style = os.getenv("EFFECT_STYLE", "moderate").lower()  # subtle/moderate/dynamic
    
    if use_effects:
        if effect_style == "dynamic":
            # Aggressive (gaming, comedy, fast topics)
            shake = r"{\t(0,40,\fscx108\fscy108)\t(40,80,\fscx92\fscy92)\t(80,120,\fscx100\fscy100)}"
            blur = r"{\blur4}"
            scale_in = r"{\fscx85\fscy85\t(0,100,\fscx100\fscy100)}"
            shadow = "3"
        elif effect_style == "subtle":
            # Minimal (educational, serious topics)
            shake = ""
            blur = r"{\blur1}"
            scale_in = ""
            shadow = "1"
        else:  # moderate (default - works for everything)
            shake = r"{\t(0,50,\fscx103\fscy103)\t(50,100,\fscx97\fscy97)\t(100,150,\fscx100\fscy100)}" if is_hook else ""
            blur = r"{\blur2}"
            scale_in = r"{\fscx90\fscy90\t(0,80,\fscx100\fscy100)}" if is_hook else ""
            shadow = "2"
    else:
        shake = ""
        blur = ""
        scale_in = ""
        shadow = "0"
    
    # Build karaoke line with effects
    initial = scale_in if scale_in else ""
    kline = initial
    
    for i in range(n):
        word_text = words_upper[i][0]
        duration_cs = ds[i]
        
        # Hook words get extra effect, normal words get blur only
        effect_this = shake if (is_hook and use_effects and shake) else ""
        kline += f"{{\\k{duration_cs}{effect_this}{blur}}}{word_text} "
    
    kline = kline.strip()

    # ===== ASS OUTPUT =====
    ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Base,{fontname},{fontsize},{_to_ass(KARAOKE_INACTIVE)},{_to_ass(KARAOKE_ACTIVE)},{_to_ass(KARAOKE_OUTLINE)},&H7F000000,1,0,0,0,100,100,0,0,1,{outline},{shadow},2,50,50,{margin_v},0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,{_ass_time(seg_dur)},Base,,0,0,{margin_v},,{{\\bord{outline}\\shad{shadow}}}{kline}
"""

    # Debug log
    effect_status = f"{effect_style} effects" if use_effects else "no effects"
    print(f"   🎵 Karaoke: {n} words | seg={seg_dur:.2f}s | target={target_cs/100:.2f}s | scale={scale:.3f} | {effect_status}")

    return ass

def draw_capcut_text(seg: str, text: str, color: str, font: str, outp: str, is_hook: bool=False, words: Optional[List[Tuple[str,float]]]=None):
    seg_dur = ffprobe_dur(seg)
    frames = max(2, int(round(seg_dur * TARGET_FPS)))

    if KARAOKE_CAPTIONS and _HAS_SUBTITLES:
        words = words or []
        ass_txt = _build_karaoke_ass(text, seg_dur, words, is_hook)
        ass_path = str(pathlib.Path(seg).with_suffix(".ass"))
        pathlib.Path(ass_path).write_text(ass_txt, encoding="utf-8")
        tmp_out = str(pathlib.Path(outp).with_suffix(".tmp.mp4"))
        try:
            run([
                "ffmpeg","-y","-hide_banner","-loglevel","error",
                "-i", seg, "-vf", f"subtitles='{ass_path}',setsar=1,fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB,trim=start_frame=0:end_frame={frames}",
                "-r", str(TARGET_FPS), "-vsync","cfr",
                "-an","-c:v","libx264","-preset","medium","-crf",str(max(16,CRF_VISUAL-3)),
                "-pix_fmt","yuv420p","-movflags","+faststart", tmp_out
            ])
            enforce_video_exact_frames(tmp_out, frames, outp)
        finally:
            pathlib.Path(ass_path).unlink(missing_ok=True)
            pathlib.Path(tmp_out).unlink(missing_ok=True)
        return

    if _HAS_DRAWTEXT:
        wrapped = wrap_mobile_lines(clean_caption_text(text).upper(), CAPTION_MAX_LINE, CAPTION_MAX_LINES)
        tf = str(pathlib.Path(seg).with_suffix(".caption.txt"))
        pathlib.Path(tf).write_text(wrapped, encoding="utf-8")

        lines = wrapped.split("\n")
        n_lines = max(1, len(lines))
        maxchars = max((len(l) for l in lines), default=1)

        base = 60 if is_hook else 50
        ratio = CAPTION_MAX_LINE / max(1, maxchars)
        fs = int(base * min(1.0, max(0.50, ratio)))
        if n_lines >= 5: fs = int(fs * 0.92)
        if n_lines >= 6: fs = int(fs * 0.88)
        if n_lines >= 7: fs = int(fs * 0.84)
        if n_lines >= 8: fs = int(fs * 0.80)
        fs = max(22, fs)

        if n_lines >= 6:   y_pos = "(h*0.55 - text_h/2)"
        elif n_lines >= 4: y_pos = "(h*0.58 - text_h/2)"
        else:              y_pos = "h-h/3-text_h/2"

        font_arg = f":fontfile={_ff_sanitize_font(font)}" if font else ""
        col = _ff_color(color)
        common = f"textfile='{tf}':fontsize={fs}:x=(w-text_w)/2:y={y_pos}:line_spacing=10"

        shadow = f"drawtext={common}{font_arg}:fontcolor=black@0.85:borderw=0"
        box    = f"drawtext={common}{font_arg}:fontcolor=white@0.0:box=1:boxborderw={(22 if is_hook else 18)}:boxcolor=black@0.65"
        main   = f"drawtext={common}{font_arg}:fontcolor={col}:borderw={(5 if is_hook else 4)}:bordercolor=black@0.9"

        vf = f"{shadow},{box},{main},setsar=1,fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB,trim=start_frame=0:end_frame={frames}"
        tmp_out = str(pathlib.Path(outp).with_suffix(".tmp.mp4"))
        try:
            run([
                "ffmpeg","-y","-hide_banner","-loglevel","error",
                "-i", seg, "-vf", vf,
                "-r", str(TARGET_FPS), "-vsync","cfr",
                "-an",
                "-c:v","libx264","-preset","medium","-crf",str(max(16,CRF_VISUAL-3)),
                "-pix_fmt","yuv420p","-movflags","+faststart", tmp_out
            ])
            enforce_video_exact_frames(tmp_out, frames, outp)
        finally:
            pathlib.Path(tf).unlink(missing_ok=True)
            pathlib.Path(tmp_out).unlink(missing_ok=True)
        return

    if REQUIRE_CAPTIONS:
        raise RuntimeError("Captions required but neither 'drawtext' nor 'subtitles' filter is available in ffmpeg.")
    print("⚠️ FFmpeg 'drawtext' ve 'subtitles' filtreleri yok — caption atlanıyor.")
    enforce_video_exact_frames(seg, frames, outp)

    # ---- drawtext fallback (ALL CAPS)
    if _HAS_DRAWTEXT:
        wrapped = wrap_mobile_lines(clean_caption_text(text).upper(), CAPTION_MAX_LINE, CAPTION_MAX_LINES)
        tf = str(pathlib.Path(seg).with_suffix(".caption.txt"))
        pathlib.Path(tf).write_text(wrapped, encoding="utf-8")

        lines = wrapped.split("\n")
        n_lines = max(1, len(lines))
        maxchars = max((len(l) for l in lines), default=1)

        base = 60 if is_hook else 50
        ratio = CAPTION_MAX_LINE / max(1, maxchars)
        fs = int(base * min(1.0, max(0.50, ratio)))
        if n_lines >= 5: fs = int(fs * 0.92)
        if n_lines >= 6: fs = int(fs * 0.88)
        if n_lines >= 7: fs = int(fs * 0.84)
        if n_lines >= 8: fs = int(fs * 0.80)
        fs = max(22, fs)

        if n_lines >= 6:   y_pos = "(h*0.55 - text_h/2)"
        elif n_lines >= 4: y_pos = "(h*0.58 - text_h/2)"
        else:              y_pos = "h-h/3-text_h/2"

        font_arg = f":fontfile={_ff_sanitize_font(font)}" if font else ""
        col = _ff_color(color)
        common = f"textfile='{tf}':fontsize={fs}:x=(w-text_w)/2:y={y_pos}:line_spacing=10"

        shadow = f"drawtext={common}{font_arg}:fontcolor=black@0.85:borderw=0"
        box    = f"drawtext={common}{font_arg}:fontcolor=white@0.0:box=1:boxborderw={(22 if is_hook else 18)}:boxcolor=black@0.65"
        main   = f"drawtext={common}{font_arg}:fontcolor={col}:borderw={(5 if is_hook else 4)}:bordercolor=black@0.9"

        vf_overlay = f"{shadow},{box},{main}"
        vf = f"{vf_overlay},fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB,trim=start_frame=0:end_frame={frames}"
        tmp_out = str(pathlib.Path(outp).with_suffix(".tmp.mp4"))
        try:
            run([
                "ffmpeg","-y","-hide_banner","-loglevel","error",
                "-i", seg, "-vf", vf,
                "-r", str(TARGET_FPS), "-vsync","cfr",
                "-an",
                "-c:v","libx264","-preset","medium","-crf",str(max(16,CRF_VISUAL-3)),
                "-pix_fmt","yuv420p","-movflags","+faststart", tmp_out
            ])
            enforce_video_exact_frames(tmp_out, frames, outp)
        finally:
            pathlib.Path(tf).unlink(missing_ok=True)
            pathlib.Path(tmp_out).unlink(missing_ok=True)
        return

    if REQUIRE_CAPTIONS:
        raise RuntimeError("Captions required but neither 'drawtext' nor 'subtitles' filter is available in ffmpeg.")
    print("⚠️ FFmpeg 'drawtext' ve 'subtitles' filtreleri yok — caption atlanıyor.")
    enforce_video_exact_frames(seg, frames, outp)

def pad_video_to_duration(video_in: str, target_sec: float, outp: str):
    vdur = ffprobe_dur(video_in)
    if vdur >= target_sec - 0.02:
        pathlib.Path(outp).write_bytes(pathlib.Path(video_in).read_bytes())
        return
    extra = max(0.0, target_sec - vdur)
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", video_in,
        "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={extra:.3f},setsar=1,fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB[v]",
        "-map","[v]",
        "-r", str(TARGET_FPS), "-vsync","cfr",
        "-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),
        "-pix_fmt","yuv420p","-movflags","+faststart",
        outp
    ])

def concat_videos_filter(files: List[str], outp: str):
    if not files: raise RuntimeError("concat_videos_filter: empty")
    inputs = []; filters = []
    for i, p in enumerate(files):
        inputs += ["-i", p]
        # Her giriş: SAR=1, FPS kilidi, tutarlı zaman tabanı
        filters.append(f"[{i}:v]setsar=1,fps={TARGET_FPS},settb=AVTB,setpts=N/{TARGET_FPS}/TB[v{i}]")
    filtergraph = ";".join(filters) + ";" + "".join(f"[v{i}]" for i in range(len(files))) + f"concat=n={len(files)}:v=1:a=0[v]"
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        *inputs,
        "-filter_complex", filtergraph,
        "-map","[v]",
        "-r", str(TARGET_FPS), "-vsync","cfr",
        "-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),
        "-pix_fmt","yuv420p","-movflags","+faststart",
        outp
    ])

def overlay_cta_tail(video_in: str, text: str, outp: str, show_sec: float, font: str):
    vdur = ffprobe_dur(video_in)
    if vdur <= 0.1 or not text.strip():
        pathlib.Path(outp).write_bytes(pathlib.Path(video_in).read_bytes())
        return
    t0 = max(0.0, vdur - max(0.8, show_sec))
    tf = str(pathlib.Path(outp).with_suffix(".cta.txt"))
    wrapped = wrap_mobile_lines(text.upper(), max_line_length=26, max_lines=3)
    pathlib.Path(tf).write_text(wrapped, encoding="utf-8")
    font_arg = f":fontfile={_ff_sanitize_font(font)}" if font else ""
    common = f"textfile='{tf}':fontsize=52:x=(w-text_w)/2:y=h*0.18:line_spacing=10"
    box    = f"drawtext={common}{font_arg}:fontcolor=white@0.0:box=1:boxborderw=18:boxcolor=black@0.55:enable='gte(t,{t0:.3f})'"
    main   = f"drawtext={common}{font_arg}:fontcolor={_ff_color('#3EA6FF')}:borderw=5:bordercolor=black@0.9:enable='gte(t,{t0:.3f})'"
    vf     = f"{box},{main},setsar=1,fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB"
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", video_in, "-vf", vf,
        "-r", str(TARGET_FPS), "-vsync","cfr",
        "-an","-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),
        "-pix_fmt","yuv420p","-movflags","+faststart", outp
    ])
    pathlib.Path(tf).unlink(missing_ok=True)

# ==================== Audio concat (lossless) ====================
def concat_audios(files: List[str], outp: str):
    if not files: raise RuntimeError("concat_audios: empty file list")
    lst = str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst, "w", encoding="utf-8") as f:
        for p in files:
            f.write(f"file '{p}'\n")
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-f","concat","-safe","0","-i", lst,
        "-c","copy",
        outp
    ])
    pathlib.Path(lst).unlink(missing_ok=True)

def lock_audio_duration(audio_in: str, target_frames: int, outp: str):
    dur = target_frames / float(TARGET_FPS)
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", audio_in,
        "-af", f"atrim=end={dur:.6f},asetpts=N/SR/TB",
        "-ar","48000","-ac","1",
        "-c:a","pcm_s16le",
        outp
    ])

def mux(video: str, audio: str, outp: str):
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", video, "-i", audio,
        "-map","0:v:0","-map","1:a:0",
        "-c:v","copy",
        "-c:a","aac","-b:a","256k",
        "-movflags","+faststart",
        "-muxpreload","0","-muxdelay","0",
        "-avoid_negative_ts","make_zero",
        outp
    ])

# ==================== Template selection (by TOPIC) ====================
def _select_template_key(topic: str) -> str:
    t = (topic or "").lower()
    geo_kw = ("country", "geograph", "city", "capital", "border", "population", "continent", "flag")
    if any(k in t for k in geo_kw):
        return "country_facts"
    return "_default"

# ==================== Gemini (topic-locked) ====================
ENHANCED_GEMINI_TEMPLATES = {
    "_default": """Create a viral 25-40s YouTube Short that STOPS THE SCROLL.

Return STRICT JSON: topic, focus, sentences (7-8), search_terms (4-10), title, description, tags.

🎯 FOCUS RULE:
- ONE specific, visual, filmable subject
- Must have abundant stock footage
- Physical subjects only (no abstract concepts)

🔥 HOOK FORMULA (Sentence 1 - First 3 Seconds):
Pick the BEST pattern for your topic:

**Pattern A: Number + Shock**
"[Number] seconds/ways/reasons [subject] [unexpected action]"
Examples: "3 seconds is all this takes" | "5 reasons nobody tells you"

**Pattern B: Contradiction**
"[Common belief] is wrong. Here's why"
Examples: "You think you know this. You don't" | "Everything you learned is backwards"

**Pattern C: Question Hook**
"What if [surprising claim]?"
Examples: "What if I told you this changes everything?"

**Pattern D: Challenge**
"Try to [action] before [time/condition]"
Examples: "Try to spot the difference" | "Find the hidden detail"

**Pattern E: POV/Relatability**
"POV: You just [discovered/learned/realized] [X]"
Examples: "POV: You finally understand" | "When you realize"

**Pattern F: Mystery/Gap**
"Nobody knows why [X]... until now"
Examples: "The secret behind" | "What they don't show you"

🧲 RETENTION ARCHITECTURE (Sentences 2-7):

**Early (2-3): Build Intrigue**
- Plant a question you'll answer later
- Use: "But here's the crazy part"
- Introduce contrast/surprise

**Middle (4-5): Pattern Interrupt**
- Break expected flow
- Use: "Wait", "Stop", "Watch closely"
- Visual cue: "Look at [specific element]"

**Late (6-7): Climax**
- Deliver on hook promise
- Peak surprise/payoff
- Use: "And that's not even..."

📝 CTA (Sentence 8 - Last 3 Seconds):

**Comment Bait** (universal):
- "Which one surprised you?"
- "A or B? Comment below"
- "Did you catch it? Drop your answer"
- "Agree or disagree?"

⚡ UNIVERSAL RULES:
- 6-12 words per sentence
- Every sentence = ONE filmable action
- NO: "it's interesting", "you won't believe", "subscribe/like"
- Build: tension → peak → satisfying end
- End HIGH (not fade out)

Language: {lang}
""",

    "country_facts": """Create viral geographic/cultural facts.

[Same structure as default but with]:
- Focus on visual landmarks OR cultural elements
- Hook patterns adapted for places/culture
- Everything else stays universal

Language: {lang}
"""
}

BANNED_PHRASES = [
    "one clear tip", "see it", "learn it", "plot twist",
    "soap-opera narration", "repeat once", "takeaway action",
    "in 60 seconds", "just the point", "crisp beats",
    "sum it up", "watch till the end", "mind-blowing fact",
    "you won't believe", "wait for it", "mind blown",
    "this will shock you", "number will surprise",
    "keep watching", "stick around", "coming up",
]

def _universal_quality_score(sentences: List[str], title: str = "") -> dict:
    """
    Universal content quality scoring (works for ALL topics)
    Returns: {quality: float, viral: float, retention: float}
    """
    text_all = (" ".join(sentences) + " " + title).lower()
    
    scores = {
        'quality': 5.0,
        'viral': 5.0,
        'retention': 5.0
    }
    
    # ===== QUALITY SIGNALS (Universal) =====
    # Conciseness (short sentences = clearer)
    avg_words = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
    if avg_words <= 12:
        scores['quality'] += 1.5
    elif avg_words > 15:
        scores['quality'] -= 1.0
    
    # Specificity (numbers = concrete)
    num_count = len(re.findall(r'\b\d+\b', text_all))
    scores['quality'] += min(2.0, num_count * 0.5)
    
    # Active voice detection (verbs present)
    action_verbs = len(re.findall(r'\b(is|does|makes|shows|reveals|changes|breaks|creates|moves|stops|starts|turns)\b', text_all))
    scores['quality'] += min(1.5, action_verbs * 0.3)
    
    # ===== VIRAL SIGNALS (Universal) =====
    # Hook strength (question/number in first sentence)
    if sentences:
        hook = sentences[0].lower()
        if '?' in hook:
            scores['viral'] += 1.0
        if re.search(r'\b\d+\b', hook):
            scores['viral'] += 0.8
        if any(w in hook for w in ['secret', 'hidden', 'never', 'nobody', 'why', 'how']):
            scores['viral'] += 0.6
    
    # Curiosity gap (unanswered questions)
    question_marks = text_all.count('?')
    scores['viral'] += min(1.2, question_marks * 0.4)
    
    # Emotional triggers (universal words)
    triggers = ['shocking', 'insane', 'crazy', 'mind', 'unbelievable', 'secret', 'hidden']
    scores['viral'] += sum(0.3 for t in triggers if t in text_all)
    
    # Contrast markers (creates tension)
    contrasts = ['but', 'however', 'actually', 'surprisingly', 'turns out', 'wait']
    scores['viral'] += sum(0.25 for c in contrasts if c in text_all)
    
    # ===== RETENTION SIGNALS (Universal) =====
    # Pattern interrupts
    interrupts = ['wait', 'stop', 'look', 'watch', 'check', 'see', 'notice']
    scores['retention'] += sum(0.4 for i in interrupts if i in text_all)
    
    # Temporal cues (creates urgency)
    temporal = ['now', 'right now', 'immediately', 'seconds', 'instantly']
    scores['retention'] += sum(0.3 for t in temporal if t in text_all)
    
    # Visual references (guides attention)
    visual_refs = ['look at', 'watch', 'see', 'notice', 'spot', 'check']
    scores['retention'] += sum(0.35 for v in visual_refs if v in text_all)
    
    # Callback to hook (narrative closure)
    if len(sentences) >= 2 and sentences[-1] and sentences[0]:
        hook_words = set(sentences[0].lower().split()[:5])
        end_words = set(sentences[-1].lower().split())
        if hook_words & end_words:  # Overlap = callback
            scores['retention'] += 1.0
    
    # ===== NEGATIVE SIGNALS (Universal) =====
    # Generic filler
    bad_words = ['interesting', 'amazing', 'great', 'nice', 'good', 'cool', 'awesome']
    penalty = sum(0.5 for b in bad_words if b in text_all)
    scores['quality'] -= penalty
    scores['viral'] -= penalty
    
    # Meta references (breaks immersion)
    meta = ['this video', 'in this', 'today we', 'i\'m going', 'we\'re going', 'subscribe', 'like']
    meta_penalty = sum(0.6 for m in meta if m in text_all)
    scores['retention'] -= meta_penalty
    
    # Too long (loses attention)
    if any(len(s.split()) > 18 for s in sentences):
        scores['retention'] -= 1.0
    
    # Normalize to 0-10
    for key in scores:
        scores[key] = max(0.0, min(10.0, scores[key]))
    
    return scores

def _gemini_call(prompt: str, model: str, temp: float) -> dict:
    if not GEMINI_API_KEY: raise RuntimeError("GEMINI_API_KEY missing")
    headers = {"Content-Type":"application/json","x-goog-api-key":GEMINI_API_KEY}
    payload = {"contents":[{"parts":[{"text": prompt}]}],
               "generationConfig":{"temperature":temp}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    txt = ""
    try: txt = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception: txt = json.dumps(data)
    m = re.search(r"\{(?:.|\n)*\}", txt)
    if not m: raise RuntimeError("Gemini response parse error (no JSON)")
    raw = re.sub(r"^```json\s*|\s*```$", "", m.group(0).strip(), flags=re.MULTILINE)
    return json.loads(raw)

def _terms_normalize(terms: List[str]) -> List[str]:
    out=[]; seen=set()
    BAD={"great","nice","good","bad","things","stuff","concept","concepts","idea","ideas"}
    for t in terms or []:
        tt = re.sub(r"[^A-Za-z0-9 ]+"," ", str(t)).strip().lower()
        tt = " ".join([w for w in tt.split() if w and len(w)>2 and w not in BAD])[:64]
        if not tt: continue
        if tt not in seen:
            seen.add(tt); out.append(tt)
    return out[:12]

def _derive_terms_from_text(topic: str, sentences: List[str]) -> List[str]:
    pool=set()
    def tok(s):
        s=re.sub(r"[^A-Za-z0-9 ]+"," ", s.lower())
        return [w for w in s.split() if len(w)>3]
    for s in [topic] + sentences:
        ws=tok(s or "")
        for i in range(len(ws)-1):
            pool.add(ws[i]+" "+ws[i+1])
    base=list(pool); random.shuffle(base)
    return _terms_normalize(base)[:8]

def build_via_gemini(channel_name: str, topic_lock: str, user_terms: List[str], banlist: List[str]):
    tpl_key = _select_template_key(topic_lock)
    template = ENHANCED_GEMINI_TEMPLATES[tpl_key]
    avoid = "\n".join(f"- {b}" for b in banlist[:15]) if banlist else "(none)"
    terms_hint = ", ".join(user_terms[:10]) if user_terms else "(none)"
    extra = (("\nADDITIONAL STYLE:\n"+GEMINI_PROMPT) if GEMINI_PROMPT else "")

    guardrails = """
RULES (MANDATORY):
- STAY ON TOPIC exactly as provided.
- Return ONLY JSON, no prose/markdown.
- Keys required: topic, focus, sentences, search_terms, title, description, tags.

🎯 FOCUS SELECTION GUIDE:
- Pick ONE visual subject with abundant stock footage
- Good: "chameleon", "Tokyo tower", "lightning storm", "gears"
- Bad: "innovation", "happiness", "success" (too abstract)
- Must be filmable from multiple angles
"""
    jitter = ((ROTATION_SEED or int(time.time())) % 13) * 0.01
    temp = max(0.6, min(1.2, GEMINI_TEMP + (jitter - 0.06)))

    prompt = f"""{template}

Channel: {channel_name}
Language: {LANG}
TOPIC (hard lock): {topic_lock}
Seed search terms (use and expand): {terms_hint}
Avoid overlap for 180 days:
{avoid}{extra}
{guardrails}
"""
    data = _gemini_call(prompt, GEMINI_MODEL, temp)

    # Parse & normalize
    topic   = topic_lock
    sentences = [clean_caption_text(s) for s in (data.get("sentences") or [])]
    sentences = [s for s in sentences if s][:8]

    terms = data.get("search_terms") or []
    if isinstance(terms, str): terms=[terms]
    terms = _terms_normalize(terms)
    if user_terms:
        seed = _terms_normalize(user_terms)
        terms = _terms_normalize(seed + terms)

    title = (data.get("title") or "").strip()
    desc  = (data.get("description") or "").strip()
    tags  = [t.strip() for t in (data.get("tags") or []) if isinstance(t,str) and t.strip()]

    # ⭐ FOCUS extraction ve temizleme
    focus = (data.get("focus") or "").strip()
    if not focus:
        # Fallback: title -> topic -> first search term
        focus = (title or data.get("topic") or topic_lock or (terms[0] if terms else "")).strip()
    
    # Focus'u 1-2 kelimeye indir
    focus = _simplify_query(focus, keep=2)
    
    # Çok generic ise search_terms'den al
    if not focus or focus in ["great", "thing", "concept", "idea", "topic", "story"]:
        focus = (terms[0] if terms else _simplify_query(topic_lock, keep=1)) or "macro detail"
    
    print(f"🎯 Extracted FOCUS: '{focus}' (from Gemini)")

    return topic, sentences, terms, title, desc, tags, focus  # ⭐ focus'u da döndür
# --- Regenerate helper (novelty guard önerilerine göre) ---
def regenerate_with_llm(topic_lock: str, seed_term: Optional[str], avoid_list: List[str], base_user_terms: List[str], banlist: List[str]):
    """
    Regenerate content with novelty suggestions.
    Returns: (topic, sentences, search_terms, title, desc, tags, focus)
    """
    seed_user_terms = list(base_user_terms or [])
    if seed_term:
        seed_user_terms = [seed_term] + seed_user_terms
    extended_ban = list((avoid_list or [])) + list((banlist or []))
    
    # build_via_gemini artık focus da döndürüyor
    tpc, sents, search_terms, ttl, desc, tags, focus = build_via_gemini(
        CHANNEL_NAME, 
        topic_lock, 
        seed_user_terms, 
        extended_ban
    )
    
    sents = _polish_hook_cta(sents)
    
    # ⭐ focus'u da döndür
    return tpc, sents, search_terms, ttl, desc, tags, focus

# ==================== Per-scene queries ====================
_STOP = set("""
a an the and or but if while of to in on at from by with for about into over after before between during under above across around through
this that these those is are was were be been being have has had do does did can could should would may might will shall
you your we our they their he she it its as than then so such very more most many much just also only even still yet
""".split())
_GENERIC_BAD = {
    "great","good","bad","big","small","old","new","many","more","most","thing","things","stuff",
    "once","next","feature","features","precisely","signal","signals","masters","master",
    "ways","way","track","tracks","uncover","gripping","limb","emotion","emotions"
}

def _proper_phrases(texts: List[str]) -> List[str]:
    phrases=[]
    for t in texts:
        for m in re.finditer(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", t or ""):
            phrase = re.sub(r"^(The|A|An)\s+", "", m.group(0))
            ws = [w.lower() for w in phrase.split()]
            for i in range(len(ws)-1):
                phrases.append(f"{ws[i]} {ws[i+1]}")
    seen=set(); out=[]
    for p in phrases:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def _domain_synonyms(all_text: str) -> List[str]:
    t = (all_text or "").lower()
    s = set()
    if any(k in t for k in ["bridge","tunnel","arch","span"]):
        s.update(["suspension bridge","cable stayed","stone arch","viaduct","aerial city bridge"])
    if any(k in t for k in ["ocean","coast","tide","wave","storm"]):
        s.update(["ocean waves","coastal storm","rocky coast","lighthouse coast"])
    if any(k in t for k in ["timelapse","growth","melt","cloud"]):
        s.update(["city timelapse","plant growth","melting ice","cloud timelapse"])
    if any(k in t for k in ["mechanism","gears","pulley","cam"]):
        s.update(["macro gears","belt pulley","cam follower","robotic arm macro"])
    if any(k in t for k in ["chameleon","bukalemun","lizard","kertenkele","reptile","renk","color"]):
        s.update([
            "chameleon close up",
            "lizard macro",
            "gecko close up",
            "iguana macro",
            "reptile skin texture",
            "animal scales macro",
            "reptile eye close up"
        ])
    return list(s)

def _entity_synonyms(ent: str, lang: str) -> list[str]:
    """
    Odak varlık için aramada işimize yarayacak yakın/eş anlamlı terimler.
    """
    e = (ent or "").lower().strip()
    if not e:
        return []

    base = [e]
    if e.endswith("s") and len(e) > 4:
        base.append(e[:-1])

    # TR özel eşlemeler
    if lang.startswith("tr"):
        table_tr = {
            "bukalemun": ["bukalemun", "kertenkele", "gecko", "iguana", "sürüngen"],
            "yunus": ["yunus", "deniz memelisi", "şişeburun yunus"],
            "japonya": ["japonya", "tokyo", "kyoto", "fuji dağı", "japon tapınağı"],
            "ahtapot": ["ahtapot", "kafadan bacaklı", "deniz canavarı"],
            "kartal": ["kartal", "yırtıcı kuş", "şahin", "atmaca"],
        }
        for k, vals in table_tr.items():
            if k in e:
                return list(dict.fromkeys(vals + base))
        return list(dict.fromkeys(base))

    # EN eşlemeler
    table_en = {
        "chameleon": ["chameleon", "lizard", "gecko", "iguana", "reptile"],
        "dolphin": ["dolphin", "marine mammal", "bottlenose dolphin"],
        "octopus": ["octopus", "cephalopod", "tentacles", "sea creature"],
        "japan": ["japan", "tokyo", "kyoto", "mt fuji", "japanese temple"],
        "italy": ["italy", "rome", "venice", "colosseum", "venetian canal"],
        "eagle": ["eagle", "raptor", "bird of prey", "hawk"],
        "bridge": ["suspension bridge", "cable stayed bridge", "arch bridge"],
        "ocean": ["ocean", "sea", "waves", "marine", "underwater"],
        "lightning": ["lightning", "storm", "thunder", "electric storm"],
        "volcano": ["volcano", "volcanic", "lava", "magma", "eruption"],
    }
    for k, vals in table_en.items():
        if k in e:
            return list(dict.fromkeys(vals + base))

    return list(dict.fromkeys(base))

    # EN eşlemeler
    table_en = {
        "chameleon": ["chameleon", "lizard", "gecko", "iguana", "reptile", "reptile skin", "chameleon close up"],
        "dolphin": ["dolphin", "ocean dolphin", "spinner dolphin", "bottlenose dolphin", "marine mammal", "pod of dolphins"],
        "octopus": ["octopus", "cephalopod", "cuttlefish", "squid", "tentacles", "octopus macro"],
        "japan": ["japan", "tokyo", "kyoto", "mt fuji", "fuji", "shibuya crossing", "torii gate", "japanese temple", "japanese street"],
        "italy": ["italy", "rome", "venice", "florence", "colosseum", "venetian canal", "duomo"],
        "eagle": ["eagle", "raptor", "bird of prey", "falcon", "hawk"],
        "bridge": ["suspension bridge", "cable stayed bridge", "stone arch bridge", "viaduct"],
    }
    for k, vals in table_en.items():
        if k in e:
            return list(dict.fromkeys(vals + base))

    return list(dict.fromkeys(base))

def _required_tokens_for_focus(focus: str, lang: str) -> set[str]:
    """URL içinde görmek istediğimiz küçük odak-token kümesini üretir.
    (ör. 'chameleon' için {'chameleon','lizard','reptile','gecko',...})"""
    syns = _entity_synonyms(focus, lang) if focus else []
    toks: set[str] = set()
    for s in (syns + [focus]):
        toks |= _url_tokens(s)
    return {t for t in toks if len(t) >= 3}

def build_global_queries(focus: str, search_terms: List[str], mode: str, lang: str) -> List[str]:
    """Kısa ve sade, 1–2 kelimelik genel sorgular üretir."""
    qs = []
    focus_q = _simplify_query(focus, keep=2)
    if focus_q: qs.append(focus_q)
    # sinonimlerden
    for s in _entity_synonyms(focus, lang) or []:
        q = _simplify_query(s, keep=2)
        if q and q not in qs: qs.append(q)
    # gemini search_terms
    for t in (search_terms or []):
        q = _simplify_query(t, keep=2)
        if q and q not in qs: qs.append(q)
    return qs[:20]

STRICT_ENTITY_FILTER = os.getenv("STRICT_ENTITY_FILTER","1") == "1"

def build_per_scene_queries(sentences: List[str], fallback_terms: List[str], topic: Optional[str]=None) -> List[str]:
    topic = (topic or "").strip()
    texts_cap = [topic] + sentences
    texts_all = " ".join([topic] + sentences)
    phrase_pool = _proper_phrases(texts_cap) + _domain_synonyms(texts_all)

    def _tok4(s: str) -> List[str]:
        s = re.sub(r"[^A-Za-z0-9 ]+", " ", (s or "").lower())
        toks = [w for w in s.split() if len(w) >= 4 and w not in _STOP and w not in _GENERIC_BAD]
        return toks

    fb=[]
    for t in (fallback_terms or []):
        t = re.sub(r"[^A-Za-z0-9 ]+"," ", str(t)).strip().lower()
        if not t: continue
        ws = [w for w in t.split() if w not in _STOP and w not in _GENERIC_BAD]
        if ws:
            fb.append(" ".join(ws[:2]))

    topic_keys = _tok4(topic)[:2]
    topic_key_join = " ".join(topic_keys) if topic_keys else ""

    queries=[]
    fb_idx = 0

    # Aydınlatma odaklı küçük sözlük → daha hedefli sonuçlar
    lex = [
        ("window", "window light"),
        ("curtain", "sheer curtains"),
        ("uplight", "floor lamp uplight"),
        ("floor lamp", "floor lamp corner"),
        ("desk", "desk near window"),
        ("strip", "led strip ambient"),
        ("wall wash", "wall wash"),
        ("corner", "corner lamp"),
        ("glare", "reduce glare"),
        ("ambient", "ambient lighting"),
    ]

    fb_strong = [t for t in (fallback_terms or []) if t]

    for s in sentences:
        s_low = " " + (s or "").lower() + " "
        picked=None

        # Önce özel lexicon
        for key, val in lex:
            if key in s_low:
                picked = val; break

        if not picked:
            for ph in phrase_pool:
                if f" {ph} " in s_low:
                    picked = ph; break

        if not picked:
            toks = _tok4(s)
            if len(toks) >= 2:
                picked = f"{toks[-2]} {toks[-1]}"
            elif len(toks) == 1:
                picked = toks[0]

        if (not picked or len(picked) < 4) and (fb_strong or fb):
            seedlist = (fb_strong if fb_strong else fb)
            picked = seedlist[fb_idx % len(seedlist)]; fb_idx += 1

        if (not picked or len(picked) < 4) and topic_key_join:
            picked = topic_key_join

        if not picked or picked in ("great","nice","good","bad","things","stuff"):
            picked = "macro detail"

        if len(picked.split()) > 2:
            w = picked.split(); picked = f"{w[-2]} {w[-1]}"

        queries.append(picked)

    return queries

# ==================== TOPIC tabanlı arama sadeleştirici ====================
def _simplify_query(q: str, keep: int = 4) -> str:
    q = (q or "").lower()
    q = re.sub(r"[^a-z0-9 ]+", " ", q)
    toks = [t for t in q.split() if t and t not in _STOP]
    return " ".join(toks[:keep]) if toks else (q.strip()[:40] if q else "")

def _gen_topic_query_candidates(topic: str, terms: List[str]) -> List[str]:
    out: List[str] = []
    base = _simplify_query(topic, keep=4)
    if base: out += [base, _simplify_query(base, keep=2)]
    for t in (terms or []):
        tt = _simplify_query(t, keep=2)
        if tt and tt not in out: out.append(tt)
    if base:
        for w in base.split():
            if w not in out: out.append(w)
    for g in ["city timelapse","ocean waves","forest path","night skyline","macro detail","street crowd","mountain landscape"]:
        if g not in out: out.append(g)
    return out[:20]

# ==================== Pexels (robust) ====================
_USED_PEXELS_IDS_RUNTIME: Set[int] = set()
# Pexels sayfa URL'lerini (slug) sıralamada kullanmak için
_PEXELS_PAGE_URL: Dict[int, str] = {}

def _pexels_headers():
    if not PEXELS_API_KEY: raise RuntimeError("PEXELS_API_KEY missing")
    return {"Authorization": PEXELS_API_KEY}

def _is_vertical_ok(w: int, h: int) -> bool:
    if PEXELS_STRICT_VERTICAL:
        return h > w and h >= PEXELS_MIN_HEIGHT
    return (h >= PEXELS_MIN_HEIGHT) and (h >= w or PEXELS_ALLOW_LANDSCAPE)

def _pexels_search(query: str, locale: str, page: int = 1, per_page: int = None) -> List[Tuple[int, str, int, int, float]]:
    """
    Pexels video arama:
    - orientation=portrait KALDIRILDI (relevans düşürüyordu)
    - Puanlamada sayfa URL slug'ını da kullanacağız (_PEXELS_PAGE_URL'e yazarız)
    Dönen tuple: (vid, file_link, w, h, dur)
    """
    per_page = per_page or max(10, min(80, PEXELS_PER_PAGE))
    url = "https://api.pexels.com/videos/search"
    try:
        r = requests.get(
            url,
            headers=_pexels_headers(),
            params={
                "query": query,
                "per_page": per_page,
                "page": page,
                # "orientation": "portrait",  # <-- KALDIRILDI
                "size": "large",
                "locale": locale
            },
            timeout=30
        )
    except Exception:
        return []
    if r.status_code != 200:
        return []
    data = r.json() or {}
    out=[]
    for v in data.get("videos", []):
        vid = int(v.get("id", 0))
        dur = float(v.get("duration",0.0))
        if dur < PEXELS_MIN_DURATION or dur > PEXELS_MAX_DURATION:
            continue

        # Sayfa URL'sini kaydet → slug'da kelimeler var (relevans için)
        page_url = (v.get("url") or "").strip()
        if page_url:
            _PEXELS_PAGE_URL[vid] = page_url

        picks=[]
        for x in (v.get("video_files", []) or []):
            w = int(x.get("width",0)); h = int(x.get("height",0))
            if _is_vertical_ok(w, h):
                picks.append((w,h,x.get("link")))
        if not picks:
            continue
        # Yükseklik hedefimize yakın + büyük çözünürlük
        picks.sort(key=lambda t: (abs(t[1]-1600), -(t[0]*t[1])))
        w,h,link = picks[0]
        out.append((vid, link, w, h, dur))
    return out

def _pexels_popular(locale: str, page: int = 1, per_page: int = 40) -> List[Tuple[int, str, int, int, float]]:
    url = "https://api.pexels.com/videos/popular"
    try:
        r = requests.get(url, headers=_pexels_headers(),
                         params={"per_page": per_page, "page": page, "locale": locale},
                         timeout=30)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    data = r.json() or {}
    out=[]
    for v in data.get("videos", []):
        vid = int(v.get("id", 0))
        dur = float(v.get("duration",0.0))
        if dur < PEXELS_MIN_DURATION or dur > PEXELS_MAX_DURATION:
            continue
        page_url = (v.get("url") or "").strip()
        if page_url:
            _PEXELS_PAGE_URL[vid] = page_url
        picks=[]
        for x in (v.get("video_files", []) or []):
            w = int(x.get("width",0)); h = int(x.get("height",0))
            if _is_vertical_ok(w, h):
                picks.append((w,h,x.get("link")))
        if not picks:
            continue
        picks.sort(key=lambda t: (abs(t[1]-1600), -(t[0]*t[1])))
        w,h,link = picks[0]
        out.append((vid, link, w, h, dur))
    return out

def _pixabay_fallback(q: str, need: int, locale: str, syn_tokens: Optional[Set[str]] = None, strict: bool = False) -> List[Tuple[int, str]]:
    """
    Pixabay'de 'tags' alanı var → eş-anlam tokenları ile filtrele.
    strict=True ise syn_tokens ile kesişmeyenler elenir.
    """
    syn_tokens = syn_tokens or set()
    if not (ALLOW_PIXABAY_FALLBACK and PIXABAY_API_KEY):
        return []
    try:
        params = {
            "key": PIXABAY_API_KEY,
            "q": q,
            "safesearch": "true",
            "per_page": min(50, max(10, need*4)),
            "video_type": "film",
            "order": "popular",
            "lang": "en" if not locale.lower().startswith("tr") else "tr"
        }
        r = requests.get("https://pixabay.com/api/videos/", params=params, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json() or {}
        outs=[]
        for h in data.get("hits", []):
            dur = float(h.get("duration",0.0))
            if dur < PEXELS_MIN_DURATION or dur > PEXELS_MAX_DURATION:
                continue
            tags = (h.get("tags") or "").lower()
            tag_tokens = set(re.findall(r"[a-z0-9]+", tags))
            if strict and syn_tokens and not (tag_tokens & syn_tokens):
                continue
            vids = h.get("videos", {})
            chosen=None
            for qual in ("large","medium","small","tiny"):
                v = vids.get(qual)
                if not v:
                    continue
                w,hh = int(v.get("width",0)), int(v.get("height",0))
                if _is_vertical_ok(w, hh):
                    chosen = (w,hh,v.get("url"))
                    break
            if chosen:
                outs.append( (int(h.get("id",0)), chosen[2]) )
        return outs[:need]
    except Exception:
        return []

def _rank_and_dedup(
    items: List[Tuple[int, str, int, int, float]],
    qtokens: Set[str],
    block: Set[int],
    syn_tokens: Optional[Set[str]] = None,
    strict: bool = False
) -> List[Tuple[int,str]]:
    """
    Puanlama:
      - Sayfa slug tokenları + dosya link tokenları birlikte
      - syn_tokens ile çakışma +2x ağırlık
      - strict=True ise syn_tokens ile kesişmeyenler elenir
    """
    syn_tokens = syn_tokens or set()
    cand=[]
    for tup in items:
        vid, link, w, h, dur = tup
        if vid in block or vid in _USED_PEXELS_IDS_RUNTIME:
            continue
        page_url = _PEXELS_PAGE_URL.get(vid, "")
        tokens = _url_tokens(link) | _url_tokens(page_url)
        if strict and syn_tokens and not (tokens & syn_tokens):
            continue
        overlap_q   = len(tokens & qtokens)
        overlap_syn = len(tokens & syn_tokens)
        score = (overlap_q*1.0) + (overlap_syn*2.0) + (1.0 if 2.0 <= dur <= 12.0 else 0.0) + (1.0 if h >= 1440 else 0.0)
        cand.append((score, vid, link))
    cand.sort(key=lambda x: x[0], reverse=True)
    out=[]; seen=set()
    for _, vid, link in cand:
        if vid in seen:
            continue
        seen.add(vid)
        out.append((vid, link))
    return out

def _url_tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

def _ensure_entity_coverage(pool: List[Tuple[int,str]], need: int, locale: str, synonyms: List[str]) -> List[Tuple[int,str]]:
    want = max(1, int(math.ceil(need * max(0.0, min(1.0, ENTITY_VISUAL_MIN)))))
    synset = set([w for w in synonyms if w and len(w) >= 3])
    hits = [(vid,link) for (vid,link) in pool if _url_tokens(link) & synset]
    if len(hits) >= want or not ENTITY_VISUAL_STRICT:
        return pool
    # Eksikse: önce Pexels, olmazsa Pixabay ile tamamla
    missing = want - len(hits)
    extra: List[Tuple[int,str]] = []
    for term in synonyms:
        if missing <= 0: break
        try:
            merged = []
            for page in (1,2):
                merged += _pexels_search(term, locale, page=page, per_page=40)
                if len(merged) >= missing*3: break
            ranked = _rank_and_dedup(merged, set([term]), _blocklist_get_pexels())
            for (vid,link) in ranked[:missing]:
                if (vid,link) not in pool and (vid,link) not in extra:
                    extra.append((vid,link)); missing -= 1
        except Exception:
            pass
    if missing > 0:
        pix = _pixabay_fallback(synonyms[0], missing, locale)
        for (vid,link) in pix:
            if (vid,link) not in pool and (vid,link) not in extra:
                extra.append((vid,link))
    # Önceliği isabetlilere ver
    prioritized = hits + [p for p in pool if p not in hits] + extra
    # Stabil tut
    seen=set(); out=[]
    for it in prioritized:
        if it not in seen:
            seen.add(it); out.append(it)
        if len(out) >= max(need, len(pool)): break
    return out

def _entity_topic_queries(topic: str, ent: str, lang: str, user_terms: List[str]) -> List[str]:
    """
    Tek sahne yerine tüm sahneler için aynı (konu/varlık) odaklı sade sorgular.
    - Öncelik: eş anlamlar
    - Sonra: topic'ten temiz 1-2 kelimelik anahtarlar
    - Sonra: kullanıcı verdi ise user_terms (sadeleştirilmiş)
    """
    def _simpl(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9 ]+"," ", (s or "").lower()).strip()
        toks = [t for t in s.split() if len(t) >= 3 and t not in _STOP and t not in _GENERIC_BAD]
        return " ".join(toks[:2]) if toks else s[:40]

    syns = _entity_synonyms(ent, lang) if ent else []
    qs: List[str] = []

    # 1) eş anlamlar
    for s in syns:
        ss = _simpl(s)
        if ss and ss not in qs:
            qs.append(ss)

    # 2) topic’ten 1–2 kelime
    base = _simpl(topic)
    if base and base not in qs:
        qs.append(base)
    if base:
        for w in base.split():
            if w not in qs: qs.append(w)

    # 3) user_terms sade
    for u in (user_terms or []):
        uu = _simpl(u)
        if uu and uu not in qs: qs.append(uu)

    # 4) Genel güvenli fallback’ler (alan bazlı)
    if lang.startswith("tr"):
        fallbacks = ["yakın plan", "detay çekim", "hareketli su", "okyanus dalga"]
    else:
        fallbacks = ["macro detail", "close up", "ocean wave", "underwater"]
    for f in fallbacks:
        if f not in qs: qs.append(f)

    return qs[:20]

def build_pexels_pool(focus: str, search_terms: List[str], need: int, rotation_seed: int = 0) -> List[Tuple[int,str]]:
    """
    FOCUS-FIRST STRATEGY:
    Tüm videolar tek bir 'focus' keyword etrafında toplanır.
    Args:
        focus: Ana görsel anahtar kelime (ör: "chameleon", "Tokyo")
        search_terms: Ek arama terimleri (Gemini'den)
        need: Kaç klip gerekli
    """
    random.seed(rotation_seed or int(time.time()))
    
    # EN-US öncelik (daha zengin içerik)
    locale = "en-US"
    block = _blocklist_get_pexels()
    
    # Ana focus keyword
    main_focus = focus or (search_terms[0] if search_terms else "macro detail")
    main_focus = _simplify_query(main_focus, keep=2)
    
    # Synonyms ve variations
    syns = _entity_synonyms(main_focus, LANG)
    syn_tokens = set(re.findall(r"[a-z0-9]+", " ".join(syns + [main_focus]).lower()))
    
    # Sorgu havuzu: focus + synonyms (maksimum 6)
    queries = [main_focus] + syns[:5]
    queries = list(dict.fromkeys(queries))  # dedup
    
    print(f"🎯 FOCUS-FIRST: '{main_focus}' | Synonyms: {syns[:3]}")
    
    pool: List[Tuple[int,str]] = []
    target = need * 4  # Bol havuz
    
    # 1) Ana focus ile DEEP search (7 sayfaya kadar)
    for q in queries:
        qtokens = set(re.findall(r"[a-z0-9]+", q.lower()))
        merged = []
        
        for page in range(1, 8):
            merged += _pexels_search(q, locale, page=page, per_page=80)
            if len(merged) >= target:
                break
        
        # STRICT ranking: SADECE focus ile ilgili olanlar
        ranked = _rank_and_dedup(
            merged, qtokens, block, 
            syn_tokens=syn_tokens, 
            strict=STRICT_ENTITY_FILTER  # ENV'den
        )
        
        pool += ranked
        print(f"   Query '{q}': {len(ranked)} clips")
        
        if len(pool) >= target:
            break
    
    # 2) Eksikse Popular'dan focus-matching olanları
    if len(pool) < need * 2:
        print(f"   ⚠️ Need more, checking popular...")
        merged = []
        for page in range(1, 4):
            merged += _pexels_popular(locale, page=page, per_page=80)
        
        pop_rank = _rank_and_dedup(
            merged, syn_tokens, block, 
            syn_tokens=syn_tokens, 
            strict=STRICT_ENTITY_FILTER
        )
        pool += pop_rank[:max(0, need*2 - len(pool))]
    
    # 3) Son çare: Pixabay (strict)
    if len(pool) < need and ALLOW_PIXABAY_FALLBACK:
        print(f"   ⚠️ Pixabay fallback...")
        pix = _pixabay_fallback(
            main_focus, need*2, locale, 
            syn_tokens=syn_tokens, 
            strict=STRICT_ENTITY_FILTER
        )
        pool += [(vid, link) for (vid, link) in pix]
    
    # Dedup
    seen=set(); dedup=[]
    for vid, link in pool:
        if vid in seen: continue
        seen.add(vid); dedup.append((vid, link))
    
    # Fresh olanları önceliklendir
    fresh = [(vid,link) for vid,link in dedup if vid not in block]
    final = fresh[:need*2] if fresh else dedup[:need*2]
    
    print(f"   ✅ Final pool: {len(final)} clips (all '{main_focus}' related)")
    return final

def build_pool_topic_only(focus: str, search_terms: List[str], need: int, rotation_seed: int = 0) -> List[Tuple[int,str]]:
    random.seed(rotation_seed or int(time.time()))
    locale = "tr-TR" if LANG.startswith("tr") else "en-US"
    block = _blocklist_get_pexels()

    queries = build_global_queries(focus, search_terms, MODE, LANG)
    pool: List[Tuple[int,str]] = []
    qtokens_cache: Dict[str, Set[str]] = {}

    for q in queries:
        qtokens_cache[q] = set(re.findall(r"[a-z0-9]+", q.lower()))
        merged: List[Tuple[int, str, int, int, float]] = []
        for page in (1, 2, 3):
            merged += _pexels_search(q, locale, page=page, per_page=PEXELS_PER_PAGE)
            if len(merged) >= need*3: break
        ranked = _rank_and_dedup(merged, qtokens_cache[q], block)
        pool += ranked[:max(3, need//2)]
        if len(pool) >= need*2: break

        # Pexels zayıfsa anında Pixabay takviye
        if len(ranked) < 2:
            pix = _pixabay_fallback(q, 3, locale)
            pool += [(vid, link) for (vid, link) in pix]

    # dedup
    seen=set(); dedup=[]
    for vid,link in pool:
        if vid in seen: continue
        seen.add(vid); dedup.append((vid,link))

    # STRICT: focus/sinonim tokenlarını zorunlu kıl
    req = _required_tokens_for_focus(focus, LANG)
    if req:
        hits = [(vid,link) for (vid,link) in dedup if (_url_tokens(link) & req)]
        if hits and (STRICT_ENTITY_FILTER or len(hits) >= max(1, int(math.ceil(need*0.75)))):
            dedup = hits + [p for p in dedup if p not in hits]

    final = dedup[:max(need, 6)]
    if len(final) < need:
        # populer + pixabay ile doldur
        merged=[]
        for page in (1,2,3):
            merged += _pexels_popular(locale, page=page, per_page=40)
            if len(merged) >= need*3: break
        pop_rank = _rank_and_dedup(merged, set(), block)
        final += [p for p in pop_rank if p not in final][:need-len(final)]
        if len(final) < need:
            pix = _pixabay_fallback(focus, need-len(final), locale)
            final += [p for p in pix if p not in final][:need-len(final)]

    print(f"   Pexels candidates (topic-only): q={len(queries)} | pool={len(final)} | focus='{focus}' | strict={STRICT_ENTITY_FILTER}")
    return final

# ==================== YouTube ====================
def yt_service():
    cid  = os.getenv("YT_CLIENT_ID")
    csec = os.getenv("YT_CLIENT_SECRET")
    rtok = os.getenv("YT_REFRESH_TOKEN")
    if not (cid and csec and rtok):
        raise RuntimeError("Missing YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN")
    creds = Credentials(
        token=None, refresh_token=rtok, token_uri="https://oauth2.googleapis.com/token",
        client_id=cid, client_secret=csec, scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds, cache_discovery=False)

def upload_youtube(video_path: str, meta: dict) -> str:
    y = yt_service()
    body = {
        "snippet": {
            "title": meta["title"], "description": meta["description"], "tags": meta.get("tags", []),
            "categoryId": "27",
            "defaultLanguage": meta.get("defaultLanguage", LANG),
            "defaultAudioLanguage": meta.get("defaultAudioLanguage", LANG)
        },
        "status": {"privacyStatus": meta.get("privacy", VISIBILITY), "selfDeclaredMadeForKids": False}
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    try:
        req = y.videos().insert(part="snippet,status", body=body, media_body=media)
        resp = req.execute()
        return resp.get("id", "")
    except HttpError as e:
        raise RuntimeError(f"YouTube upload error: {e}")

# ==================== Long SEO Description ====================
def build_long_description(channel: str, topic: str, sentences: List[str], tags: List[str]) -> Tuple[str, str, List[str]]:
    """
    Universal viral title & description generator (works for ALL topics)
    """
    hook = (sentences[0].rstrip(" .!?") if sentences else topic or channel)
    
    # ===== DYNAMIC TITLE GENERATION (Topic-Agnostic) =====
    # Extract key elements from content
    numbers = re.findall(r'\d+', hook + " " + topic)
    has_question = '?' in hook
    has_negation = any(w in hook.lower() for w in ['no', 'not', 'never', 'nobody', 'nothing'])
    
    # Get first noun/main keyword (generic extraction)
    words = [w for w in topic.split() if len(w) >= 3 and w.lower() not in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}]
    main_keyword = words[0] if words else "This"
    
    # Universal emoji pool (works for any topic)
    neutral_emojis = ['🤯', '😱', '🔥', '⚡', '✨', '💡', '👀', '🎯']
    emoji = random.choice(neutral_emojis)
    
    # Universal title patterns (no topic-specific assumptions)
    patterns = []
    
    if numbers:
        patterns.extend([
            f"{numbers[0]} Things About {main_keyword} That'll Blow Your Mind {emoji}",
            f"{numbers[0]} {main_keyword} Facts You Didn't Know {emoji}",
            f"Why {numbers[0]} Changes Everything About {main_keyword} {emoji}",
        ])
    
    if has_question:
        patterns.extend([
            f"{hook} {emoji}",
            f"{hook.replace('?', '')} {emoji}",
        ])
    
    if has_negation:
        patterns.extend([
            f"{hook} {emoji}",
            f"The {main_keyword} Truth Nobody Tells You {emoji}",
        ])
    
    # Generic fallbacks (always work)
    patterns.extend([
        f"{hook} {emoji}",
        f"Wait Until You See This About {main_keyword} {emoji}",
        f"The {main_keyword} Secret {emoji}",
        f"{main_keyword}: You Won't Believe This {emoji}",
        f"This {main_keyword} Fact Will Change You {emoji}",
        f"POV: You Just Learned About {main_keyword} {emoji}",
    ])
    
    # Pick shortest that fits YouTube limit
    title = None
    for p in patterns:
        if len(p) <= 95:
            title = p
            break
    if not title:
        title = patterns[0][:92] + "..."
    
    # ===== UNIVERSAL DESCRIPTION (Works for All Topics) =====
    keyword = main_keyword.lower()
    
    # Dynamic intro (adapts to content)
    explainer = (
        f"🎬 {' '.join(sentences[:2])}\n\n"
        f"Quick breakdown of {keyword} in under 60 seconds. "
        f"Perfect for curious minds! 🧠\n\n"
    )
    
    # Key points (universal formatting)
    takeaways = "📌 Key Points:\n"
    emoji_bullets = ["🔹", "💡", "⚡", "✨", "🎯", "🔥", "💥", "🌟"]
    for i, s in enumerate(sentences[:6]):
        emoji = emoji_bullets[i % len(emoji_bullets)]
        takeaways += f"{emoji} {s}\n"
    
    # Why watch (universal hook)
    why_watch = (
        f"\n💭 Why Watch:\n"
        f"Clear visuals + concise facts = instant understanding. "
        f"Each scene reinforces one idea. Watch till the end! {emoji}\n\n"
    )
    
    # Universal CTA
    cta_section = (
        f"👇 Take Action:\n"
        f"🔔 Subscribe for daily shorts\n"
        f"💬 Share your thoughts below\n"
        f"🔄 Send this to someone\n"
        f"📲 Save for later\n\n"
    )
    
    # Universal hashtag strategy
    tagset = []
    
    # Extract main keywords (max 3)
    for word in words[:3]:
        if len(word) >= 3:
            tagset.append(f"#{word.lower()}")
    
    # Universal viral tags (work for everything)
    universal_tags = [
        "#shorts", "#viral", "#trending", "#fyp", 
        "#educational", "#mindblown", "#didyouknow", 
        "#facts", "#learnontiktok", "#quicklearn"
    ]
    tagset.extend(universal_tags)
    
    # Topic-specific tags (if provided)
    if tags:
        for t in tags[:12]:
            tclean = re.sub(r"[^A-Za-z0-9]+", "", t).lower()
            if tclean and len(tclean) >= 3 and ("#"+tclean) not in tagset:
                tagset.append("#"+tclean)
    
    # Combine
    body = explainer + takeaways + why_watch + cta_section + " ".join(tagset[:30])
    
    if len(body) > 4900:
        body = body[:4900] + "..."
    
    # YouTube tags
    yt_tags = [h[1:] for h in tagset if h][:30]
    yt_tags.extend(["shorts", "viral", "trending", "educational"])
    
    return title, body, list(dict.fromkeys(yt_tags))[:30]

# ==================== HOOK/CTA cilası ====================
HOOK_MAX_WORDS = _env_int("HOOK_MAX_WORDS", 10)
CTA_STYLE      = os.getenv("CTA_STYLE", "soft_comment")
LOOP_HINT      = os.getenv("LOOP_HINT", "1") == "1"

def _polish_hook_cta(sentences: List[str]) -> List[str]:
    if not sentences: return sentences
    ss = sentences[:]

    # HOOK: ilk cümle ≤ 10 kelime ve POWER WORDS ile başlasın
    hook = clean_caption_text(ss[0])
    words = hook.split()
    
    # Power patterns - bunlardan biri yoksa ekle
    power_starters = [
        (r"^\d+", ""),  # Sayı ile başlıyorsa zaten güçlü
        (r"^(Why|How|What|When|Where)", ""),  # Soru kelimeleri
        (r"^(Secret|Hidden|Unknown|Rare)", ""),  # Merak uyandıran
        (r"^(Never|Always|Only|Just)", ""),  # Mutlak ifadeler
    ]
    
    has_power = any(re.search(pattern, hook, re.IGNORECASE) for pattern, _ in power_starters)
    
    if not has_power and len(words) > 3:
        # Güçlü bir başlangıç ekle
        first_word = words[0].lower()
        if first_word not in ["the", "a", "an", "this", "that"]:
            # Sayı varsa öne çıkar
            numbers = re.findall(r'\d+', hook)
            if numbers:
                hook = f"{numbers[0]} {hook}"
            # Yoksa "How" veya "Why" ekle
            elif "?" not in hook:
                hook = f"How {hook.lower()}"
            elif not hook.endswith("?"):
                hook = hook.rstrip(".!") + "?"
    
    # Maksimum kelime sınırı
    if len(words) > HOOK_MAX_WORDS:
        hook = " ".join(words[:HOOK_MAX_WORDS])
        if not re.search(r"[?!]$", hook):
            hook = hook.rstrip(".") + "?"
    
    ss[0] = hook

    # CTA temiz ve noktalı
    if ss and not re.search(r'[.!?]$', ss[-1].strip()):
        ss[-1] = ss[-1].strip() + '.'
    
    return ss

# ==================== BGM helpers (download, loop, duck, mix) ====================
def _pick_bgm_source(tmpdir: str) -> Optional[str]:
    # 1) repo içi bgm/ klasörü
    try:
        p = pathlib.Path(BGM_DIR)
        if p.exists():
            files = [str(x) for x in p.glob("*.mp3")] + [str(x) for x in p.glob("*.wav")]
            if files:
                random.shuffle(files)
                return files[0]
    except Exception:
        pass
    # 2) URL listesinden indir
    urls = list(BGM_URLS or [])
    random.shuffle(urls)
    for u in urls:
        try:
            ext = ".mp3" if ".mp3" in u.lower() else ".wav"
            outp = str(pathlib.Path(tmpdir) / f"bgm_src{ext}")
            with requests.get(u, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(outp, "wb") as f:
                    for ch in r.iter_content(8192): f.write(ch)
            if pathlib.Path(outp).stat().st_size > 100_000:
                return outp
        except Exception:
            continue
    return None

def _make_bgm_looped(src: str, dur: float, out_wav: str):
    fade = max(0.3, float(BGM_FADE))
    endst = max(0.0, dur - fade)
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-stream_loop","-1","-i", src,
        "-t", f"{dur:.3f}",
        "-af", f"loudnorm=I=-21:TP=-2.0:LRA=11,"
               f"afade=t=in:st=0:d={fade:.2f},afade=t=out:st={endst:.2f}:d={fade:.2f},"
               "aresample=48000,pan=mono|c0=0.5*FL+0.5*FR",
        "-ar","48000","-ac","1","-c:a","pcm_s16le",
        out_wav
    ])

# --- BGM helpers (ekleyin / değiştirin) ---

def _find_bgm_candidates() -> List[str]:
    """repo kökünde bgm/ klasöründen veya BGM_URLS (virgül/boşlukla ayrılmış http linkleri) listesinden adayları döndürür."""
    out = []
    try:
        d = pathlib.Path("bgm")
        if d.exists():
            for pat in ("*.mp3","*.wav","*.m4a","*.flac","*.ogg"):
                out += [str(p) for p in d.glob(pat)]
    except Exception:
        pass

    urls = (os.getenv("BGM_URLS") or "").strip()
    if urls:
        for u in re.split(r"[,\s]+", urls):
            u = u.strip()
            if u.startswith("http"):
                out.append(u)
    return out

def _loop_to_duration(bgm_in: str, target_sec: float, outp: str):
    """
    BGM'i tam süreye uzatır (gerekirse loop). Yerel dosya bekler; URL verdiysek önce indirildiğini varsayıyoruz.
    """
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-stream_loop","-1","-t",f"{max(0.1,target_sec):.3f}",
        "-i", bgm_in,
        "-ar","48000","-ac","1",
        "-af","dynaudnorm=f=250:g=4",
        outp
    ])

def _duck_and_mix(voice_in: str, bgm_in: str, outp: str):
    """
    Seslendirme + BGM karışımı:
      1) BGM'i önce istediğimiz dB'e çekeriz.
      2) BGM'i, seslendirmeyi sidechain olarak kullanarak bastırırız (duck).
      3) Sonra VOICE + DUCKED_BGM'i karıştırırız.
    """
    bgm_gain_db   = float(os.getenv("BGM_GAIN_DB", "-10"))     # daha yüksek için -8 / -6 deneyebilirsiniz
    thr           = float(os.getenv("BGM_DUCK_THRESH", "0.03"))
    ratio         = float(os.getenv("BGM_DUCK_RATIO",  "10"))
    attack_ms     = int(os.getenv("BGM_DUCK_ATTACK_MS","6"))
    release_ms    = int(os.getenv("BGM_DUCK_RELEASE_MS","180"))

    # ÖNEMLİ: sidechaincompress SIRASI [PROGRAM][SIDECHAIN] → [BGM][VOICE]
    # Ayrıca makeup en az 1 olmalı (0 hatası alıyordunuz).
    sc = (
        f"sidechaincompress="
        f"threshold={thr}:ratio={ratio}:attack={attack_ms}:release={release_ms}:"
        f"makeup=1.0:level_in=1.0:level_sc=1.0"
    )

    # 0:a = VOICE, 1:a = BGM
    # BGM'i önce volume ile kıs, sonra VOICE ile duck et, sonra VOICE ile amix
    filter_complex = (
        f"[1:a]volume={bgm_gain_db}dB[b];"
        f"[b][0:a]{sc}[duck];"
        f"[0:a][duck]amix=inputs=2:duration=shortest,aresample=48000,alimiter=limit=0.98[mix]"
    )
    
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", voice_in, "-i", bgm_in,
        "-filter_complex", filter_complex,
        "-map","[mix]",
        "-ar","48000","-ac","1",
        "-c:a","pcm_s16le",
        outp
    ])

# ==================== Main ====================
def main():
    print(f"==> {CHANNEL_NAME} | MODE={MODE} | topic-first build")
    if not (_HAS_DRAWTEXT or _HAS_SUBTITLES):
        msg = "⚠️ UYARI: ffmpeg'te ne 'drawtext' ne 'subtitles' var. Altyazılar üretilemez."
        if REQUIRE_CAPTIONS: raise RuntimeError(msg + " REQUIRE_CAPTIONS=1 olduğu için durduruldu.")
        else: print(msg + " (devam edilecek)")

    random.seed(ROTATION_SEED or int(time.time()))
    topic_lock = TOPIC or "Interesting Visual Explainers"
    user_terms = SEARCH_TERMS_ENV

    # NoveltyGuard'ı kanal + pencere ile başlat
    GUARD = NoveltyGuard(state_dir=STATE_DIR, window_days=ENTITY_COOLDOWN_DAYS)
    SG = StateGuard(channel=CHANNEL_NAME)  # <-- YENİ

    # 1) İçerik üretim (topic-locked) + kalite kontrol + NOVELTY

    attempts = 0
    best = None
    best_score = -1.0
    banlist = _recent_topics_for_prompt()
    novelty_tries = 0
    selected_search_term_final = None
    
    while attempts < max(3, NOVELTY_RETRIES):
        attempts += 1
        
        if USE_GEMINI and GEMINI_API_KEY:
            try:
                # ⭐ focus'u da yakala
                tpc, sents, search_terms, ttl, desc, tags, focus = build_via_gemini(
                    CHANNEL_NAME, 
                    topic_lock, 
                    user_terms, 
                    banlist
                )
            except Exception as e:
                print(f"Gemini error: {str(e)[:200]}")
                tpc = topic_lock
                sents = []
                search_terms = user_terms or []
                ttl = ""
                desc = ""
                tags = []
                focus = topic_lock  # ⭐ fallback focus
        else:
            tpc = topic_lock
            sents = [
                f"{tpc} comes alive in small vivid scenes.",
                "Each beat shows one concrete detail to remember.",
                "The story moves forward without fluff or filler.",
                "You can picture it clearly as you listen.",
                "A tiny contrast locks the idea in memory.",
                "No meta talk—just what matters on screen.",
                "Replay to catch micro-details and patterns.",
                "What would you add? Tell me below."
            ]
            search_terms = _terms_normalize(user_terms or ["macro detail","timelapse","clean b-roll"])
            ttl = ""
            desc = ""
            tags = []
            focus = _simplify_query(tpc, keep=1)  # ⭐ fallback focus

        # Hook + CTA cilası
        sents = _polish_hook_cta(sents)

        # --- StateGuard: script semantik dup kontrolü ---
        try:
            if SG.script_semantic_duplicate(" ".join(sents or [])):
                novelty_tries += 1
                print(f"🚫 StateGuard: script too similar (>=0.90) → rebuilding… ({novelty_tries}/{NOVELTY_RETRIES})")
                banlist = [tpc] + banlist
                continue
        except Exception as e:
            print(f"⚠️ SG script dup check skipped: {e}")

        # ---- NoveltyGuard: LRU term seçimi + semantik kontrol ----
        selected_term = None
        try:
            if search_terms:
                selected_term = GUARD.pick_search_term(channel=CHANNEL_NAME, candidates=search_terms)
        except Exception as e:
            print(f"⚠️ pick_search_term warn: {e}")
    
        def _guard_recent_titles(n=10):
            try:
                return [it["title"] for it in GUARD.recent_items(CHANNEL_NAME, ENTITY_COOLDOWN_DAYS)[:n] if it.get("title")]
            except Exception:
                return []

        # Semantik benzerlik + cooldown kontrolü
        try:
            title_for_check = (ttl or "").strip() or (sents[0] if sents else tpc)
            script_for_check = " ".join(sents or [])
            decision = GUARD.check_novelty(
                channel=CHANNEL_NAME,
                title=title_for_check,
                script=script_for_check,
                search_term=selected_term,
                category=MODE,
                mode=MODE,
                lang=LANG
            )
            if not decision.ok:
                novelty_tries += 1
                print(f"🚫 NoveltyGuard block ({novelty_tries}/{NOVELTY_RETRIES}): {decision.reason}")
                alt_terms = (decision.suggestions or {}).get("alt_terms", [])
                avoid_list = _guard_recent_titles(12)
                seed_alt = (alt_terms[0] if alt_terms else None)
                
                # ⭐ regenerate_with_llm artık focus da döndürüyor
                tpc, sents, search_terms, ttl, desc, tags, focus = regenerate_with_llm(
                    topic_lock, 
                    seed_alt, 
                    avoid_list, 
                    (user_terms or []), 
                    banlist
                )
                banlist = avoid_list + banlist
                continue
            else:
                selected_search_term_final = selected_term or (search_terms[0] if search_terms else None)
        except Exception as e:
            print(f"⚠️ NoveltyGuard check skipped: {e}")
            selected_search_term_final = (search_terms[0] if search_terms else None)

        # NOVELTY (yerel jaccard) kontrolü
        ok, avoid_terms = _novelty_ok(sents)
        if not ok and novelty_tries < NOVELTY_RETRIES:
            novelty_tries += 1
            print(f"⚠️ Similar to recent videos (try {novelty_tries}/{NOVELTY_RETRIES}) → rebuilding with bans: {avoid_terms[:8]}")
            banlist = avoid_terms + banlist
            if selected_search_term_final:
                user_terms = [selected_search_term_final] + (user_terms or [])
            continue

        # Focus-entity cooldown
        if ENTITY_COOLDOWN_DAYS > 0:
            ent = _derive_focus_entity(tpc, MODE, sents)
            if ent:
                try:
                    if SG.is_on_cooldown(ent) or SG.entity_too_similar(ent):
                        novelty_tries += 1
                        print(f"⚠️ Entity blocked (cooldown/alias): '{ent}' → rebuilding… ({novelty_tries}/{NOVELTY_RETRIES})")
                        banlist = [ent] + banlist
                        continue
                except Exception as e:
                    print(f"⚠️ SG entity check skipped: {e}")

        # ⭐ UNIVERSAL QUALITY SCORING
        scores = _universal_quality_score(sents, ttl)
        quality = scores['quality']
        viral = scores['viral']
        retention = scores['retention']
        overall = (quality * 0.4 + viral * 0.35 + retention * 0.25)
    
        print(f"📝 Content: {tpc} | {len(sents)} lines")
        print(f"   📊 Quality={quality:.1f} | Viral={viral:.1f} | Retention={retention:.1f} | Overall={overall:.1f}")
    
        # ⭐ Best'i kaydet (focus dahil)
        if overall >= 7.0 and quality >= 6.5:
            best = (tpc, sents, search_terms, ttl, desc, tags, selected_search_term_final, focus)
            best_score = overall
            break
        else:
            # Score yine de best'ten iyiyse kaydet
            if overall > best_score:
                best = (tpc, sents, search_terms, ttl, desc, tags, selected_search_term_final, focus)
                best_score = overall
            
            print(f"⚠️ Low scores → rebuilding…")
            banlist = [tpc] + banlist
            time.sleep(0.5)

    # ⭐ Döngü sonrası: best None olabilir mi kontrolü
    if best is None:
        # Fallback: son denenen değerleri kullan
        print(f"⚠️ No content passed threshold, using last attempt (score={overall:.1f})")
        best = (tpc, sents, search_terms, ttl, desc, tags, selected_search_term_final, focus)
    
    # Final seçilen içerik
    tpc, sentences, search_terms, ttl, desc, tags, selected_search_term_final, focus = best
    sig = f"{CHANNEL_NAME}|{tpc}|{sentences[0] if sentences else ''}"
    fp = sorted(list(_sentences_fp(sentences)))[:500]
    _record_recent(_hash12(sig), MODE, tpc, fp=fp)

    # Record focus entity cooldown
    try:
        __ent = _derive_focus_entity(tpc, MODE, sentences)
        __ek = _entity_key(MODE, __ent)
        _entity_touch(__ek)
    except Exception:
        pass

    # debug meta
    _dump_debug_meta(f"{OUT_DIR}/meta_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json", {
        "channel": CHANNEL_NAME, "topic": tpc, "sentences": sentences, "search_terms": search_terms,
        "lang": LANG, "model": GEMINI_MODEL, "ts": time.time(), "selected_search_term": selected_search_term_final
    })

    print(f"📊 Sentences: {len(sentences)}")

    # 2) TTS (kelime zamanları ile)
    tmp = tempfile.mkdtemp(prefix="enhanced_shorts_")
    font = font_path()
    wavs, metas = [], []
    print("🎤 TTS…")
    for i, s in enumerate(sentences):
        base = normalize_sentence(s)
        w = str(pathlib.Path(tmp) / f"sent_{i:02d}.wav")
        d, words = tts_to_wav(base, w)
        wavs.append(w); metas.append((base, d, words))
        print(f"   {i+1}/{len(sentences)}: {d:.2f}s")

    need_clips = max(6, min(12, int(os.getenv("SCENE_COUNT", "8"))))

    # 3) Pexels — TOPIC-ONLY (tüm sahneler aynı odaktan beslensin)
    need_clips = max(6, min(12, int(os.getenv("SCENE_COUNT", "8"))))

    if os.getenv("SCENE_STRATEGY","topic_only").lower() == "topic_only":
        pool: List[Tuple[int,str]] = build_pexels_pool(
            focus=(focus or tpc),
            search_terms=(search_terms or user_terms or []),
            need=need_clips,
            rotation_seed=ROTATION_SEED
        )
    else:
        # Eski davranış (hybrid/per-scene) - artık kullanılmıyor
        per_scene_queries = build_per_scene_queries(
            [m[0] for m in metas], 
            (search_terms or user_terms or []), 
            topic=tpc
        )
        print("🔎 Per-scene queries:")
        for q in per_scene_queries: 
            print(f"   • {q}")
        pool: List[Tuple[int,str]] = build_pexels_pool(
            focus=focus,
            search_terms=(search_terms or user_terms or []),
            need=need_clips,
            rotation_seed=ROTATION_SEED
        )

    # İlk kontrol
    if not pool:
        raise RuntimeError("Pexels: no suitable clips (after all fallbacks).")

    # NoveltyGuard PEXELS tekrar filtresi (SQLite state)
    try:
        cand_ids = [vid for (vid, _) in pool]
        fresh_ids = GUARD.filter_new_pexels(channel=CHANNEL_NAME, candidate_ids=cand_ids, days=ENTITY_COOLDOWN_DAYS)
        if fresh_ids:
            fresh_set = set(map(int, fresh_ids))
            pool = [(vid, link) for (vid, link) in pool if int(vid) in fresh_set]
            print(f"🧹 NoveltyGuard Pexels filter → {len(pool)} fresh candidates")
        if not pool:
            raise RuntimeError("Pexels: no suitable NEW clips (all seen in last 30 days).")
    except Exception as e:
        print(f"⚠️ NoveltyGuard Pexels filter skipped: {e}")

    # 4) İndir ve dağıt
    downloads: Dict[int,str] = {}
    print("⬇️ Download pool…")
    for idx, (vid, link) in enumerate(pool):
        try:
            f = str(pathlib.Path(tmp) / f"pool_{idx:02d}_{vid}.mp4")
            with requests.get(link, stream=True, timeout=120) as rr:
                rr.raise_for_status()
                with open(f, "wb") as w:
                    for ch in rr.iter_content(8192): w.write(ch)
            if pathlib.Path(f).stat().st_size > 300_000:
                downloads[vid] = f
        except Exception as e:
            print(f"⚠️ download fail ({vid}): {e}")
    if not downloads: raise RuntimeError("Pexels pool empty after downloads.")
    print(f"   Downloaded unique clips: {len(downloads)}")

    # Yeterli benzersiz klip yoksa otomatik doldur
    if not PEXELS_ALLOW_REUSE and len(downloads) < len(metas):
        print(f"   Not enough uniques ({len(downloads)}/{len(metas)}). Backfilling from popular…")
        locale = "tr-TR" if LANG.startswith("tr") else "en-US"
        need_more = len(metas) - len(downloads)
        page = 1
        tried = set(downloads.keys())
        while need_more > 0 and page <= 4:
            pops = _pexels_popular(locale, page=page, per_page=50)
            page += 1
            for vid, link, w,h,dur, page_url in pops:
                if vid in tried: continue
                try:
                    f = str(pathlib.Path(tmp) / f"pop_{vid}.mp4")
                    with requests.get(link, stream=True, timeout=120) as rr:
                        rr.raise_for_status()
                        with open(f, "wb") as wfd:
                            for ch in rr.iter_content(8192): wfd.write(ch)
                    if pathlib.Path(f).stat().st_size > 300_000:
                        downloads[vid] = f; tried.add(vid); need_more -= 1
                        if need_more <= 0: break
                except Exception:
                    continue
        # Pixabay fallback
        if need_more > 0:
            pix = _pixabay_fallback("interior lighting", need_more, locale)
            for vid, link in pix:
                try:
                    f = str(pathlib.Path(tmp) / f"pix_{vid}.mp4")
                    with requests.get(link, stream=True, timeout=120) as rr:
                        rr.raise_for_status()
                        with open(f, "wb") as wfd:
                            for ch in rr.iter_content(8192): wfd.write(ch)
                    if pathlib.Path(f).stat().st_size > 300_000:
                        downloads[vid] = f
                        need_more -= 1
                        if need_more <= 0: break
                except Exception:
                    pass
        print(f"   After backfill uniques: {len(downloads)}")

    usage = {vid:0 for vid in downloads.keys()}
    chosen_files: List[str] = []; chosen_ids: List[int] = []
    if not PEXELS_ALLOW_REUSE:
        ordered = list(downloads.items())[:len(metas)]
        if len(ordered) < len(metas):
            print("⚠️ Still short on unique clips; enabling minimal reuse for remaining.")
        for i in range(len(metas)):
            if i < len(ordered):
                vid, pathv = ordered[i]
            else:
                vid = min(usage.keys(), key=lambda k: usage[k]); pathv = downloads[vid]
            usage[vid] += 1
            chosen_files.append(pathv); chosen_ids.append(vid); _USED_PEXELS_IDS_RUNTIME.add(vid)
    else:
        for i in range(len(metas)):
            picked_vid = None
            for vid in list(usage.keys()):
                if usage[vid] < PEXELS_MAX_USES_PER_CLIP:
                    picked_vid = vid; break
            if picked_vid is None:
                picked_vid = min(usage.keys(), key=lambda k: usage[k])
            usage[picked_vid] += 1
            chosen_files.append(downloads[picked_vid]); chosen_ids.append(picked_vid); _USED_PEXELS_IDS_RUNTIME.add(picked_vid)

    # --- StateGuard: pre-render idempotency (aynı içerikse render’a girmeden terk et) ---
    script_text = " ".join([m[0] for m in metas])
    try:
        content_hash = StateGuard.make_content_hash(script_text, chosen_files, wavs[0] if wavs else None)
    except Exception:
        content_hash = hashlib.sha1(("fallback:"+script_text).encode("utf-8")).hexdigest()

    if SG.was_uploaded(content_hash):
        print(f"⏭️ Skipping build: identical content already uploaded (hash={content_hash[:8]})")
        return

    # 5) Segment + altyazı (KARAOKE)
    print("🎬 Segments…")
    segs = []
    for i, ((base_text, d, words), src) in enumerate(zip(metas, chosen_files)):
        base   = str(pathlib.Path(tmp) / f"seg_{i:02d}.mp4")
        make_segment(src, d, base)
        colored = str(pathlib.Path(tmp) / f"segsub_{i:02d}.mp4")
        draw_capcut_text(
            base,
            base_text,
            CAPTION_COLORS[i % len(CAPTION_COLORS)],
            font,
            colored,
            is_hook=(i == 0),
            words=words
        )
        segs.append(colored)

    # 6) Birleştir
    print("🎞️ Assemble…")
    vcat = str(pathlib.Path(tmp) / "video_concat.mp4"); concat_videos_filter(segs, vcat)
    acat = str(pathlib.Path(tmp) / "audio_concat.wav"); concat_audios(wavs, acat)

    # 7) Süre & kare kilitleme
    adur = ffprobe_dur(acat); vdur = ffprobe_dur(vcat)
    if vdur + 0.02 < adur:
        vcat_padded = str(pathlib.Path(tmp) / "video_padded.mp4")
        pad_video_to_duration(vcat, adur, vcat_padded); vcat = vcat_padded
    a_frames = max(2, int(round(adur * TARGET_FPS)))
    vcat_exact = str(pathlib.Path(tmp) / "video_exact.mp4"); enforce_video_exact_frames(vcat, a_frames, vcat_exact); vcat = vcat_exact
    acat_exact = str(pathlib.Path(tmp) / "audio_exact.wav"); lock_audio_duration(acat, a_frames, acat_exact); acat = acat_exact
    vdur2 = ffprobe_dur(vcat); adur2 = ffprobe_dur(acat)
    print(f"🔒 Locked A/V: video={vdur2:.3f}s | audio={adur2:.3f}s | fps={TARGET_FPS}")

    # 7.1) Contextual CTA (overlay only at tail)
    cta_text = ""
    try:
        if CTA_ENABLE:
            cta_text = build_contextual_cta(tpc, [m[0] for m in metas], LANG)
            if cta_text:
                print(f"💬 CTA: {cta_text}")
                vcat_cta = str(pathlib.Path(tmp) / "video_cta.mp4")
                overlay_cta_tail(vcat, cta_text, vcat_cta, CTA_SHOW_SEC, font)
                # aynı kare sayısını koru:
                vcat_exact2 = str(pathlib.Path(tmp) / "video_exact_cta.mp4")
                enforce_video_exact_frames(vcat_cta, a_frames, vcat_exact2)
                vcat = vcat_exact2
    except Exception as e:
        print(f"⚠️ CTA overlay skipped: {e}")

    # 7.5) BGM mix (opsiyonel)
    if BGM_ENABLE:
        bgm_src = _pick_bgm_source(tmp)
        if bgm_src:
            print("🎧 BGM: mixing with sidechain ducking…")
            bgm_loop = str(pathlib.Path(tmp) / "bgm_loop.wav")
            _make_bgm_looped(bgm_src, adur2, bgm_loop)
            a_mix = str(pathlib.Path(tmp) / "audio_with_bgm.wav")
            _duck_and_mix(acat, bgm_loop, a_mix)
            # yeniden tam kare süreye kilitle
            a_mix_exact = str(pathlib.Path(tmp) / "audio_with_bgm_exact.wav")
            lock_audio_duration(a_mix, max(2, int(round(adur2 * TARGET_FPS))), a_mix_exact)
            acat = a_mix_exact
        else:
            print("🎧 BGM: kaynak bulunamadı (BGM_DIR veya BGM_URLS).")

    # 8) Mux
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^A-Za-z0-9]+', '_', tpc)[:60] or "Short"
    outp = f"{OUT_DIR}/{CHANNEL_NAME}_{safe_topic}_{ts}.mp4"
    print("🔄 Mux…"); mux(vcat, acat, outp)
    final = ffprobe_dur(outp); print(f"✅ Saved: {outp} ({final:.2f}s)")

    # 9) Metadata (long SEO)
    title, description, yt_tags = build_long_description(CHANNEL_NAME, tpc, [m[0] for m in metas], tags)
    meta = {"title": title,"description": description,"tags": yt_tags,"privacy": VISIBILITY,
            "defaultLanguage": LANG,"defaultAudioLanguage": LANG}

    # 10) Upload (varsa env)
    try:
        if os.getenv("UPLOAD_TO_YT","1") == "1":
            print("📤 Uploading to YouTube…")
            vid_id = upload_youtube(outp, meta)
            print(f"🎉 YouTube Video ID: {vid_id}\n🔗 https://youtube.com/watch?v={vid_id}")
        else:
            print("⏭️ Upload disabled (UPLOAD_TO_YT != 1)")
    except Exception as e:
        print(f"❌ Upload skipped: {e}")

    # --- StateGuard: başarı kaydı (entity + embedding + uploads) ---
    try:
        final_entity = __ent if '__ent' in locals() and __ent else _derive_focus_entity(tpc, MODE, [m[0] for m in metas])
        script_text = " ".join([m[0] for m in metas])
        try:
            content_hash  # varsa kullan
        except NameError:
            content_hash = StateGuard.make_content_hash(script_text, chosen_files, wavs[0] if wavs else None)
        SG.mark_uploaded(
            entity=final_entity or "",
            script_text=script_text,
            content_hash=content_hash,
            video_path=outp,
            title=title
        )
    except Exception as e:
        print(f"⚠️ SG mark_uploaded warn: {e}")

    # 11.5) NoveltyGuard'a final kayıt (başlık + script + pexels)
    try:
        GUARD.register_item(
            channel=CHANNEL_NAME,
            title=title,
            script=" ".join([m[0] for m in metas]),
            search_term=(selected_search_term_final or (search_terms[0] if search_terms else "")),
            category=MODE,
            mode=MODE,
            lang=LANG,
            topic=tpc,
            pexels_ids=list(map(int, chosen_ids or []))
        )
    except Exception as e:
        print(f"⚠️ NoveltyGuard register warn: {e}")

    # 12) Temizlik
    try: shutil.rmtree(tmp); print("🧹 Cleaned temp files")
    except: pass

# ==================== Debug meta ====================
def _dump_debug_meta(path: str, obj: dict):
    try:
        pathlib.Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

if __name__ == "__main__":
    main()

















