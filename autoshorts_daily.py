# autoshorts_daily.py — Relevance-first Pexels + SSML fix + reuse guard
# -*- coding: utf-8 -*-
import os, sys, re, json, time, uuid, random, datetime, tempfile, pathlib, subprocess, hashlib, math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# -------------------- Constants / ENV --------------------
VOICE_STYLE = os.getenv("TTS_STYLE", "narration-professional")
TARGET_MIN_SEC = float(os.getenv("TARGET_MIN_SEC", "22"))
TARGET_MAX_SEC = float(os.getenv("TARGET_MAX_SEC", "42"))

CHANNEL_NAME  = os.getenv("CHANNEL_NAME", "DefaultChannel")
MODE          = os.getenv("MODE", "country_facts").strip().lower()
LANG          = os.getenv("LANG", "en")
VISIBILITY    = os.getenv("VISIBILITY", "public")
ROTATION_SEED = int(os.getenv("ROTATION_SEED", "0"))
OUT_DIR = "out"; pathlib.Path(OUT_DIR).mkdir(exist_ok=True)

# APIs
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_GEMINI     = os.getenv("USE_GEMINI", "0") == "1"
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_PROMPT  = (os.getenv("GEMINI_PROMPT") or "").strip() or None

# Video rendering
TARGET_FPS     = 25
CRF_VISUAL     = 22
CAPTION_COLORS = ["#FFD700","#FF6B35","#00F5FF","#32CD32","#FF1493","#1E90FF","#FFA500","#FF69B4"]
CAPTION_MAX_LINE = 22

# State
STATE_FILE = f"state_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json"

# -------------------- Lightweight deps (auto-install) --------------------
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

# -------------------- Voices --------------------
VOICE_OPTIONS = {
    "en": [
        "en-US-JennyNeural","en-US-JasonNeural","en-US-AriaNeural","en-US-GuyNeural",
        "en-AU-NatashaNeural","en-GB-SoniaNeural","en-CA-LiamNeural","en-US-DavisNeural","en-US-AmberNeural"
    ],
    "tr": ["tr-TR-EmelNeural","tr-TR-AhmetNeural"]
}
VOICE = os.getenv("TTS_VOICE", VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])[0])
VOICE_RATE = os.getenv("TTS_RATE", "+10%")

# -------------------- Utils (proc, fonts, ffprobe) --------------------
def run(cmd, check=True):
    res = subprocess.run(cmd, text=True, capture_output=True)
    if check and res.returncode != 0:
        raise RuntimeError(res.stderr[:2000])
    return res

def ffprobe_dur(p):
    try:
        out = run(["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0", p]).stdout.strip()
        return float(out) if out else 0.0
    except:
        return 0.0

def font_path():
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "C:/Windows/Fonts/arial.ttf"]:
        if pathlib.Path(p).exists():
            return p
    return ""

def escape_drawtext(s: str) -> str:
    return (s.replace("\\","\\\\").replace(":", "\\:").replace(",", "\\,")
             .replace("'", "\\'").replace("%","\\%"))

# -------------------- State management --------------------
def _state_load() -> dict:
    try: return json.load(open(STATE_FILE, "r", encoding="utf-8"))
    except: return {"recent": [], "used_pexels_ids": []}

def _state_save(st: dict):
    st["recent"] = st.get("recent", [])[-1200:]
    st["used_pexels_ids"] = st.get("used_pexels_ids", [])[-5000:]
    pathlib.Path(STATE_FILE).write_text(json.dumps(st, indent=2), encoding="utf-8")

def _hash12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _is_recent(h: str, window_days=365) -> bool:
    now = time.time()
    for r in _state_load().get("recent", []):
        if r.get("h")==h and (now - r.get("ts",0)) < window_days*86400:
            return True
    return False

def _record_recent(h: str, mode: str, topic: str):
    st = _state_load()
    st.setdefault("recent", []).append({"h":h,"mode":mode,"topic":topic,"ts":time.time()})
    _state_save(st)

def _blocklist_add_pexels(ids: List[int], days=30):
    st = _state_load()
    now = int(time.time())
    for vid in ids:
        st.setdefault("used_pexels_ids", []).append({"id": int(vid), "ts": now})
    # purge old
    cutoff = now - days*86400
    st["used_pexels_ids"] = [x for x in st["used_pexels_ids"] if x.get("ts",0) >= cutoff]
    _state_save(st)

def _blocklist_get_pexels() -> set:
    st = _state_load()
    return {int(x["id"]) for x in st.get("used_pexels_ids", [])}

def _recent_topics_for_prompt(limit=20) -> List[str]:
    st = _state_load()
    topics = [r.get("topic","") for r in reversed(st.get("recent", [])) if r.get("topic")]
    uniq=[]
    for t in topics:
        if t and t not in uniq: uniq.append(t)
        if len(uniq) >= limit: break
    return uniq

# -------------------- Text prep (captions) --------------------
def clean_caption_text(s: str) -> str:
    t = (s or "").strip().replace("—","-").replace('"',"").replace("`","")
    t = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', t)
    t = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', t)
    t = re.sub(r'\s+',' ', t)
    if t and t[0].islower(): t = t[0].upper() + t[1:]
    if len(t) > 80:
        words = t.split()
        t = " ".join(words[:12]) + "."
    return t.strip()

def wrap_mobile_lines(text: str, max_line_length: int = CAPTION_MAX_LINE) -> str:
    text = (text or "").strip()
    if not text: return text
    words = text.split(); n = len(words)
    target = 3 if n > 12 else 2
    per = math.ceil(n/target)
    chunks = [" ".join(words[i*per:(i+1)*per]) for i in range(target)]
    chunks = [c for c in chunks if c]
    if any(len(c)>max_line_length for c in chunks) and target==2:
        target=3; per=math.ceil(n/target)
        chunks = [" ".join(words[i*per:(i+1)*per]) for i in range(target)]
        chunks = [c for c in chunks if c]
    if len(chunks)==1 and n>1:
        mid=n//2; chunks=[" ".join(words[:mid]), " ".join(words[mid:])]
    return "\n".join(chunks[:3])

# -------------------- TTS (Edge-TTS with SSML) --------------------
def _build_ssml(text: str, voice: str) -> str:
    txt = (text or "").strip()
    # basic pauses for naturalness
    txt = re.sub(r'\.(\s|$)', '.<break time="500ms"/> ', txt)
    txt = re.sub(r',(\\s|$)', ',<break time="250ms"/> ', txt)
    return f"""
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{ 'tr-TR' if LANG.startswith('tr') else 'en-US' }">
  <voice name="{voice}">
    <prosody rate="{VOICE_RATE}" pitch="+0Hz">{txt}</prosody>
  </voice>
</speak>
""".strip()

def tts_to_wav(text: str, wav_out: str) -> float:
    import asyncio
    def _run_ff(args): subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y", *args], check=True)
    def _probe(path: str, default: float = 3.5) -> float:
        try:
            pr = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1", path],
                                capture_output=True, text=True, check=True)
            return float((pr.stdout or "0").strip())
        except Exception: return default

    mp3 = wav_out.replace(".wav",".mp3")
    available = VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])
    selected = VOICE if VOICE in available else available[0]
    clean_text = text.strip()

    async def _edge_save():
        ssml = _build_ssml(clean_text, selected)
        comm = edge_tts.Communicate(ssml, voice=selected, ssml=True)
        await comm.save(mp3)

    try:
        try: asyncio.run(_edge_save())
        except RuntimeError:
            nest_asyncio.apply(); loop=asyncio.get_event_loop(); loop.run_until_complete(_edge_save())
        # safe filter chain (no deesser)
        _run_ff(["-i", mp3, "-ar", "48000", "-ac", "1", "-acodec", "pcm_s16le",
                 "-af", "volume=0.92,highpass=f=75,lowpass=f=15000,dynaudnorm=g=7:f=250:r=0.95,acompressor=threshold=-20dB:ratio=2:attack=5:release=60",
                 wav_out])
        pathlib.Path(mp3).unlink(missing_ok=True)
        return _probe(wav_out, 3.5)
    except Exception as e:
        # Fallback simple
        try:
            async def _edge_simple():
                comm = edge_tts.Communicate(clean_text, voice=selected, rate=VOICE_RATE)
                await comm.save(mp3)
            try: asyncio.run(_edge_simple())
            except RuntimeError:
                nest_asyncio.apply(); loop=asyncio.get_event_loop(); loop.run_until_complete(_edge_simple())
            _run_ff(["-i", mp3, "-ar","44100","-ac","1","-acodec","pcm_s16le","-af","volume=0.9,dynaudnorm=g=5:f=200", wav_out])
            pathlib.Path(mp3).unlink(missing_ok=True)
            return _probe(wav_out, 3.5)
        except:
            _run_ff(["-f","lavfi","-t","4.0","-i","anullsrc=r=44100:cl=mono", wav_out]); return 4.0

# -------------------- Video compose helpers --------------------
def make_segment(src: str, dur: float, outp: str):
    dur = max(0.8, min(dur, 5.0))
    fade = max(0.05, min(0.12, dur/8))
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "eq=brightness=0.02:contrast=1.08:saturation=1.1,"
        f"fade=t=in:st=0:d={fade:.2f},"
        f"fade=t=out:st={max(0.0,dur-fade):.2f}:d={fade:.2f}"
    )
    run(["ffmpeg","-y","-i",src,"-t",f"{dur:.3f}","-vf",vf,"-r","25","-an",
         "-c:v","libx264","-preset","fast","-crf","22","-pix_fmt","yuv420p", outp])

# --- COLORS: use 0xRRGGBB for ffmpeg drawtext ---
CAPTION_COLORS = ["0xFFD700","0xFF6B35","0x00F5FF","0x32CD32","0xFF1493","0x1E90FF","0xFFA500","0xFF69B4"]

def _ff_color(c: str) -> str:
    """
    Convert #RRGGBB -> 0xRRGGBB for ffmpeg. Accepts already-0x too.
    """
    c = (c or "").strip()
    if c.startswith("#"):
        return "0x" + c[1:].upper()
    if re.fullmatch(r"0x[0-9A-Fa-f]{6}", c):
        return c
    # fallback to named color
    return "white"

def draw_capcut_text(seg: str, text: str, color: str, font: str, outp: str, is_hook: bool=False):
    """
    Safer drawtext: renkler 0xRRGGBB, metin kaçışları tam.
    Hata olursa sade fallback ile yeniden dener.
    """
    wrapped = wrap_mobile_lines(clean_caption_text(text), CAPTION_MAX_LINE)
    # drawtext multi-line \n destekliyor; metni agresif kaçır:
    esc = escape_drawtext(wrapped).replace("\n", r"\n")

    lines = wrapped.count("\n")+1
    base_fs = (58 if is_hook else 48)
    if lines >= 3: base_fs -= 6

    y_pos = "h-h/3-text_h/2"
    common = f"text='{esc}':fontsize={base_fs}:x=(w-text_w)/2:y={y_pos}:line_spacing=8"
    col = _ff_color(color)

    shadow = f"drawtext={common}:fontcolor=black@0.85:borderw=0"
    box    = f"drawtext={common}:fontcolor=white@0.0:box=1:boxborderw={20 if is_hook else 16}:boxcolor=black@0.65"
    main   = f"drawtext={common}:fontcolor={col}:borderw={5 if is_hook else 4}:bordercolor=black@0.9"

    if font:
        fp = font.replace(":","\\:").replace(",","\\,").replace("\\","/")
        shadow += f":fontfile={fp}"
        box    += f":fontfile={fp}"
        main   += f":fontfile={fp}"

    vf = f"{shadow},{box},{main}"

    try:
        run([
            "ffmpeg","-y","-i",seg,"-vf",vf,
            "-c:v","libx264","-preset","medium",
            "-crf",str(max(16,CRF_VISUAL-3)),
            "-movflags","+faststart",
            outp
        ])
    except Exception as e:
        # Fallback: sadece ana metin, beyaz renk; kutu/gölge yok
        print(f"⚠️ drawtext advanced failed: {e}\n   → retrying with minimal overlay")
        main_min = f"drawtext={common}:fontcolor=white:borderw=4:bordercolor=black@0.9"
        if font:
            fp = font.replace(":","\\:").replace(",","\\,").replace("\\","/")
            main_min += f":fontfile={fp}"
        run([
            "ffmpeg","-y","-i",seg,"-vf",main_min,
            "-c:v","libx264","-preset","medium","-crf","20","-movflags","+faststart", outp
        ])

def concat_videos(files: List[str], outp: str):
    lst = str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst,"w") as f:
        for p in files: f.write(f"file '{p}'\n")
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",lst,"-c","copy", outp])

def concat_audios(files: List[str], outp: str):
    lst = str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst,"w") as f:
        for p in files: f.write(f"file '{p}'\n")
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",lst,"-af","volume=0.9,dynaudnorm", outp])

def mux(video: str, audio: str, outp: str):
    try:
        vd, ad = ffprobe_dur(video), ffprobe_dur(audio)
        if abs(vd-ad) > 1.0:
            md = min(vd, ad, 45.0)
            tv = video.replace(".mp4","_temp.mp4")
            ta = audio.replace(".wav","_temp.wav")
            run(["ffmpeg","-y","-i",video,"-t",f"{md:.2f}","-c","copy", tv])
            run(["ffmpeg","-y","-i",audio,"-t",f"{md:.2f}","-c","copy", ta])
            video, audio = tv, ta
        run(["ffmpeg","-y","-i",video,"-i",audio,"-map","0:v:0","-map","1:a:0",
             "-c:v","copy","-c:a","aac","-b:a","256k","-movflags","+faststart","-shortest","-avoid_negative_ts","make_zero", outp])
        for f in (video, audio):
            if f.endswith("_temp.mp4") or f.endswith("_temp.wav"):
                pathlib.Path(f).unlink(missing_ok=True)
    except Exception:
        run(["ffmpeg","-y","-i",video,"-i",audio,"-c","copy","-shortest", outp])

# -------------------- Gemini content --------------------
ENHANCED_GEMINI_TEMPLATES = {
    "_default": """Create a 25–40s YouTube Short.
Return JSON: country, topic, sentences (7–8), search_terms (4–8), title, description, tags.""",
    "country_facts": """Create amazing country facts.
EXACTLY 7–8 sentences; each 6–12 words.
Return JSON only with: country, topic, sentences, search_terms, title, description, tags."""
}

def _gemini_call(prompt: str, model: str) -> dict:
    if not GEMINI_API_KEY: raise RuntimeError("GEMINI_API_KEY missing")
    headers = {"Content-Type":"application/json","x-goog-api-key":GEMINI_API_KEY}
    payload = {"contents":[{"parts":[{"text": prompt}]}], "generationConfig":{"temperature":0.8}}
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

def build_via_gemini(mode: str, channel_name: str, banlist: List[str]) -> Tuple[str,str,List[str],List[str],str,str,List[str]]:
    template = ENHANCED_GEMINI_TEMPLATES.get(mode, ENHANCED_GEMINI_TEMPLATES["_default"])
    avoid = "\n".join(f"- {b}" for b in banlist[:15]) if banlist else "(none)"
    prompt = f"""{template}

Channel: {channel_name}
Language: {LANG}
Avoid for 180 days:
{avoid}
"""
    data = _gemini_call(prompt, GEMINI_MODEL)
    country = str(data.get("country") or "World").strip()
    topic   = str(data.get("topic") or "Amazing Facts").strip()
    sentences = [clean_caption_text(s) for s in (data.get("sentences") or [])]
    sentences = [s for s in sentences if s][:8]
    terms = data.get("search_terms") or []
    if isinstance(terms, str): terms=[terms]
    terms = [t.strip() for t in terms if isinstance(t,str) and t.strip()]
    title = (data.get("title") or "").strip()
    desc  = (data.get("description") or "").strip()
    tags  = [t.strip() for t in (data.get("tags") or []) if isinstance(t,str) and t.strip()]
    return country, topic, sentences, terms, title, desc, tags

# -------------------- Query terms per-video (from sentences) --------------------
STOP = set("""
a an the and or but if while of to in on at from by with for about into over after before between during under above across around through
this that these those is are was were be been being have has had do does did can could should would may might will shall
you your we our they their he she it its as than then so such very more most many much just also only even still yet
""".split())

def _words(s: str) -> List[str]:
    s = re.sub(r"[^A-Za-z0-9\- ]+"," ", s.lower())
    return [w for w in s.split() if w and w not in STOP and len(w)>2]

MYTH_BOOSTERS = {
    "myth_gods": [
        # sahne / obje
        "ancient greek statue", "marble bust close up", "temple columns doric",
        "torch flame in darkness", "smoke swirling in cave", "obsidian rock texture",
        "underworld cave river", "lava flow close up", "storm clouds dramatic sky",
        "black robe silhouette", "gold laurel crown", "gates doorway ancient",
        # mekan
        "ruined temple night", "cave entrance mist", "catacombs corridor", "underground cavern",
        # hareket
        "slow motion smoke", "embers floating", "torchlight flicker", "shadow moving wall"
    ],
    # istersen diğer modlara da ek deposu açarız
}

def derive_search_terms(sentences: List[str], fallback: List[str], channel_hint: str="") -> List[str]:
    """
    Cümlelerden işe yarar bigram/terimler çıkarır; moda özel booster’larla harmanlar.
    Myth_gods gibi nişte, somut görsel objelere zorlar (statue, torch, cave, smoke, lava, columns…).
    """
    # 1) Çekirdek kelimeler
    bag = []
    for s in sentences:
        bag += _words(s)
    # özel isimleri (Topic vb.) da zorla ekleyelim
    main_subjects = []
    for s in sentences[:2]:
        ms = re.findall(r"[A-Z][a-z]{2,}", s)
        main_subjects += [m.lower() for m in ms]
    bag = main_subjects + bag

    # 2) uniq sırayı koru
    uniq=[]; seen=set()
    for w in bag:
        if w not in seen:
            seen.add(w); uniq.append(w)

    # 3) moda özel booster
    boosters = []
    if MODE in MYTH_BOOSTERS:
        boosters = MYTH_BOOSTERS[MODE][:]

    # 4) cümle kökenli bigramları oluştur
    bigrams=set()
    for s in sentences:
        ws = _words(s)
        for a,b in zip(ws, ws[1:]):
            if a!=b:
                bigrams.add(f"{a} {b}")

    # 5) soruları ve çöplerini at
    trash = {"think","evil","eldest","time","very","really","thing","something"}
    core = [w for w in uniq if w not in trash][:10]

    # 6) candidate queries
    queries=[]
    subj = (core[0] if core else "")
    if channel_hint:
        queries.append(f"{channel_hint} logo intro vertical 4k")  # genel b-roll seçeneği

    # Subjects’i boost et (ör: hades / zeus)
    if subj:
        queries += [
            f"{subj} statue portrait 4k", f"{subj} marble bust portrait 4k",
            f"{subj} temple columns vertical 4k", f"{subj} torch flame cave vertical 4k",
        ]

    # bigramlardan işe yarayanları ekle
    for bg in list(bigrams)[:8]:
        queries.append(f"{bg} portrait 4k")
        queries.append(f"{bg} vertical video")

    # moda özel boosterları ekle
    for b in boosters[:14]:
        queries.append(f"{b} portrait 4k")
        queries.append(f"{b} vertical 4k")

    # güvenli b-roll’lar
    queries += ["cinematic smoke portrait 4k", "dramatic sky vertical 4k", "dark cave portrait 4k"]

    # fallbackler
    for t in (fallback or []):
        if t:
            queries.append(f"{t} portrait 4k")

    # dedupe+temizle
    cleaned=[]
    seen=set()
    for q in queries:
        q = re.sub(r"\s+"," ", q.strip())
        if len(q.split()) < 2: 
            continue
        if q not in seen:
            seen.add(q); cleaned.append(q)

    random.shuffle(cleaned)
    # myth_gods ise sayıyı biraz yüksek tutalım; yoksa 10–12
    limit = 18 if MODE=="myth_gods" else 12
    return cleaned[:limit] if cleaned else ["portrait 4k"]

# -------------------- Pexels search with relevance & reuse guard --------------------
def _pexels_headers(): 
    if not PEXELS_API_KEY: raise RuntimeError("PEXELS_API_KEY missing")
    return {"Authorization": PEXELS_API_KEY}

def _pexels_locale(lang: str) -> str:
    return "tr-TR" if lang.startswith("tr") else "en-US"

def _score_video(v: dict, query_terms: List[str]) -> float:
    w = 0.0
    url = (v.get("url") or "").lower()
    dur = float(v.get("duration") or 0)
    width = 0; height = 0
    try:
        # choose best file to read dimensions
        files = v.get("video_files",[]) or []
        if files:
            best = max(files, key=lambda x: (x.get("height",0)*x.get("width",0)))
            width, height = int(best.get("width",0)), int(best.get("height",0))
    except: pass
    # orientation reward
    if height >= width and height >= 1080: w += 2.0
    # duration reward (short b-roll 2–8s best per segment)
    if 2.0 <= dur <= 15.0: w += 1.0
    # keyword overlap on slug
    tokens = set(re.findall(r"[a-z0-9]+", url))
    overlap = len(tokens.intersection(set(query_terms)))
    w += overlap * 1.5
    return w

def pexels_pick_clips(per_video_queries: List[str], need: int) -> List[Tuple[int,str]]:
    """
    Returns list of (video_id, best_file_url) length==need
    """
    headers = _pexels_headers()
    locale  = _pexels_locale(LANG)
    seen_files=set()
    block = _blocklist_get_pexels()
    pool: Dict[int, dict] = {}  # id -> best candidate
    # gather candidates from multiple queries
    for q in per_video_queries:
        try:
            r = requests.get("https://api.pexels.com/videos/search",
                             headers=headers,
                             params={"query": q, "per_page": 15, "orientation":"portrait","size":"large","locale":locale},
                             timeout=30)
            data = r.json() if r.status_code==200 else {}
            for v in data.get("videos", []):
                vid = int(v.get("id", 0))
                if vid in block: 
                    continue
                # choose best file (highest area, mp4)
                files = v.get("video_files",[]) or []
                if not files: 
                    continue
                best = max(files, key=lambda x: (int(x.get("height",0))*int(x.get("width",0))))
                link = best.get("link")
                if not link or link in seen_files:
                    continue
                # score
                terms = _words(q)
                score = _score_video(v, terms)
                if score <= 0:
                    continue
                cand = pool.get(vid, {"score":-1})
                if score > cand["score"]:
                    pool[vid] = {"score":score, "url": link, "meta": v}
                seen_files.add(link)
        except Exception:
            continue
    # rank & pick
    ranked = sorted(pool.items(), key=lambda kv: kv[1]["score"], reverse=True)
    picked=[]; used_ids=[]
    for vid, info in ranked:
        picked.append((vid, info["url"]))
        used_ids.append(vid)
        if len(picked) >= max(need, 3):
            break
    if len(picked) < max(3, need//2):
        raise RuntimeError("Not enough relevant Pexels videos collected")
    # record used ids to avoid reuse across channels/runs (30d)
    _blocklist_add_pexels(used_ids, days=30)
    # log
    print("🔎 Pexels selected:")
    for vid, url in picked:
        print(f"   • id={vid} | {url}")
    return picked[:need]

# -------------------- YouTube upload --------------------
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
    req = y.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = req.execute()
    return resp.get("id", "")

# -------------------- Main --------------------
def main():
    print(f"==> {CHANNEL_NAME} | MODE={MODE} | Relevance-first build")

    # 1) Content
    if USE_GEMINI and GEMINI_API_KEY:
        banlist = _recent_topics_for_prompt()
        chosen=None; last=None
        for _ in range(8):
            try:
                ctry, tpc, sents, terms, ttl, desc, tags = build_via_gemini(MODE, CHANNEL_NAME, banlist)
                last=(ctry,tpc,sents,terms,ttl,desc,tags)
                sig = f"{MODE}|{tpc}|{sents[0] if sents else ''}"
                h = _hash12(sig)
                if not _is_recent(h, window_days=180):
                    _record_recent(h, MODE, tpc); chosen=last; break
                else:
                    banlist.insert(0, tpc); time.sleep(1.5)
            except Exception as e:
                print(f"Gemini error: {str(e)[:160]}")
                time.sleep(1)
        if chosen is None:
            ctry, tpc, sents, terms, ttl, desc, tags = last if last else ("World","Daily Facts",
                ["Amazing fact you didn’t know.","People are surprised by this.","Science reveals hidden truths.",
                 "History explains modern mysteries.","Nature has incredible designs.","Technology changes everything quickly.",
                 "What do you think about this?"], ["documentary 4k"], "", "", [])
    else:
        ctry, tpc = "World","Amazing Facts"
        sents = [
            "Did you know Turkey hides a city underground?",
            "Derinkuyu stretches many levels beneath the surface.",
            "Ancient builders carved rooms from volcanic rock.",
            "People once lived safely there for decades.",
            "The tunnels connect to nearby hidden chambers.",
            "Ventilation shafts still work remarkably well.",
            "It’s older than most famous empires.",
            "Which underground mystery fascinates you most?"
        ]
        terms = ["cappadocia underground", "ancient tunnels", "turkey caves 4k"]
        ttl = desc = ""; tags=[]

    sentences = sents
    print(f"📝 Content: {ctry} | {tpc}")
    print(f"📊 Sentences: {len(sentences)}")

    # 2) TTS
    tmp = tempfile.mkdtemp(prefix="enhanced_shorts_")
    font = font_path()
    wavs=[]; metas=[]
    print("🎤 TTS…")
    for i, s in enumerate(sentences):
        w = str(pathlib.Path(tmp)/f"sent_{i:02d}.wav")
        dur = tts_to_wav(s, w)
        wavs.append(w); metas.append((s, dur))
        print(f"   {i+1}/{len(sentences)}: {dur:.1f}s")

    # 3) Per-video search terms
    channel_hint = CHANNEL_NAME.replace("_"," ").replace("-"," ")
    per_video_queries = derive_search_terms(sentences, terms, channel_hint=channel_hint)
    print("🔎 Per-video queries:")
    for q in per_video_queries: print(f"   • {q}")

    # 4) Pexels download with relevance & reuse guard
    picked = pexels_pick_clips(per_video_queries, need=len(sentences))
    # download files
    clips=[]
    for idx,(vid,link) in enumerate(picked):
        f = str(pathlib.Path(tmp)/f"clip_{idx:02d}_{vid}.mp4")
        with requests.get(link, stream=True, timeout=120) as rr:
            rr.raise_for_status()
            with open(f,"wb") as w:
                for ch in rr.iter_content(8192): w.write(ch)
        if pathlib.Path(f).stat().st_size > 500_000:
            clips.append(f)
    if len(clips) < len(sentences):
        print("⚠️ Not enough downloaded clips after filtering; cycling clips to fill.")
    # 5) Segments + captions
    print("🎬 Segments…")
    segs=[]
    for i,(s,d) in enumerate(metas):
        base = str(pathlib.Path(tmp)/f"seg_{i:02d}.mp4")
        make_segment(clips[i % len(clips)], d, base)
        colored = str(pathlib.Path(tmp)/f"segsub_{i:02d}.mp4")
        draw_capcut_text(base, s, CAPTION_COLORS[i % len(CAPTION_COLORS)], font, colored, is_hook=(i==0))
        segs.append(colored)

    # 6) Assemble
    print("🎞️ Assemble…")
    vcat = str(pathlib.Path(tmp)/"video_concat.mp4"); concat_videos(segs, vcat)
    acat = str(pathlib.Path(tmp)/"audio_concat.wav"); concat_audios(wavs, acat)

    # pad if too short
    total = ffprobe_dur(acat)
    print(f"📏 Total audio: {total:.1f}s (target {TARGET_MIN_SEC}-{TARGET_MAX_SEC}s)")
    if total < TARGET_MIN_SEC:
        deficit = TARGET_MIN_SEC - total
        extra = min(deficit, 5.0)
        if extra > 0.1:
            padded = str(pathlib.Path(tmp)/"audio_padded.wav")
            run(["ffmpeg","-y","-f","lavfi","-t",f"{extra:.2f}","-i","anullsrc=r=48000:cl=mono",
                 "-i",acat,"-filter_complex","[1:a][0:a]concat=n=2:v=0:a=1", padded])
            acat = padded

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^A-Za-z0-9]+','_', tpc)[:60] or "Short"
    outp = f"{OUT_DIR}/{ctry}_{safe_topic}_{ts}.mp4"
    print("🔄 Mux…")
    mux(vcat, acat, outp)
    final = ffprobe_dur(outp)
    print(f"✅ Saved: {outp} ({final:.1f}s)")

    # 7) Metadata
    def _ok_str(x): return isinstance(x,str) and len(x.strip())>0
    if _ok_str(ttl):
        meta = {"title": ttl[:95], "description": (desc or "")[:4900], "tags": (tags[:15] if isinstance(tags,list) else []),
                "privacy": VISIBILITY, "defaultLanguage": LANG, "defaultAudioLanguage": LANG}
    else:
        hook = (sentences[0].rstrip(" .!?") if sentences else f"{ctry} secrets")
        title = f"{hook} — {ctry} Facts"
        description = "• " + "\n• ".join(sentences[:6]) + f"\n\n#{ctry.lower()} #shorts #facts"
        meta = {"title": title[:95], "description": description[:4900],
                "tags": ["shorts","facts",ctry.lower(),"education","interesting","broll","documentary","learn","trivia","history"],
                "privacy": VISIBILITY, "defaultLanguage": LANG, "defaultAudioLanguage": LANG}

    # 8) Upload
    try:
        print("📤 Uploading to YouTube…")
        vid_id = upload_youtube(outp, meta)
        print(f"🎉 YouTube Video ID: {vid_id}\n🔗 https://youtube.com/watch?v={vid_id}")
    except Exception as e:
        print(f"❌ Upload skipped: {e}")

    # 9) Cleanup
    try:
        import shutil; shutil.rmtree(tmp); print("🧹 Cleaned temp files")
    except: pass

if __name__ == "__main__":
    main()

