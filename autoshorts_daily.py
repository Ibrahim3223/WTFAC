# autoshorts_daily.py — Relevance-first Pexels + per-scene queries + sync captions
# -*- coding: utf-8 -*-
import os, sys, re, json, time, uuid, random, datetime, tempfile, pathlib, subprocess, hashlib, math
from typing import List, Optional, Tuple, Dict

# -------------------- ENV / constants --------------------
VOICE_STYLE   = os.getenv("TTS_STYLE", "narration-professional")
TARGET_MIN_SEC = float(os.getenv("TARGET_MIN_SEC", "22"))
TARGET_MAX_SEC = float(os.getenv("TARGET_MAX_SEC", "42"))

CHANNEL_NAME  = os.getenv("CHANNEL_NAME", "DefaultChannel")
MODE          = os.getenv("MODE", "country_facts").strip().lower()
LANG          = os.getenv("LANG", "en")
VISIBILITY    = os.getenv("VISIBILITY", "public")
ROTATION_SEED = int(os.getenv("ROTATION_SEED", "0"))
OUT_DIR = "out"; pathlib.Path(OUT_DIR).mkdir(exist_ok=True)

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_GEMINI     = os.getenv("USE_GEMINI", "0") == "1"
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

TARGET_FPS     = 25
CRF_VISUAL     = 22
CAPTION_MAX_LINE = 22

STATE_FILE = f"state_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json"

# -------------------- deps (auto-install) --------------------
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
VOICE_RATE = os.getenv("TTS_RATE", "+12%")

# -------------------- Utils --------------------
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

def _ff_sanitize_font(font_path_str: str) -> str:
    if not font_path_str: return ""
    return font_path_str.replace(":", r"\:").replace(",", r"\,").replace("\\", "/")

def escape_drawtext(s: str) -> str:
    return (s.replace("\\","\\\\").replace(":", "\\:").replace(",", "\\,")
             .replace("'", "\\'").replace("%","\\%"))

def normalize_sentence(raw: str) -> str:
    s = (raw or "").strip()
    s = s.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(re.sub(r"\s+", " ", ln).strip() for ln in s.split("\n"))
    s = s.replace("—", "-").replace("–", "-").replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    return s

# -------------------- State --------------------
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

# -------------------- Caption text & wrap --------------------
def clean_caption_text(s: str) -> str:
    t = (s or "").strip()
    t = t.replace("—","-").replace("`","")
    t = re.sub(r"\s+"," ", t)
    if t and t[0].islower(): t = t[0].upper()+t[1:]
    if len(t) > 120:  # biraz esnettik
        words = t.split()
        t = " ".join(words[:22]) + "…"
    return t

def wrap_mobile_lines(text: str, max_line_length: int = CAPTION_MAX_LINE) -> str:
    text = (text or "").strip()
    if not text: return text
    words = text.split()

    # Dinamik satır sayısı: 2–4 arası
    # çok uzun kelime sayısı arttıkça satır sayısını yükselt
    target = 2
    if len(words) > 10: target = 3
    if len(words) > 18: target = 4

    # karakter dağılımına göre böl
    total = len(text)
    ideal = max(10, min(max_line_length, math.ceil(total/target)))

    lines=[]; cur=[]; L=0
    for w in words:
        add = (1 if cur else 0) + len(w)
        if L + add > ideal and len(lines)+1 < target:
            lines.append(" ".join(cur)); cur=[w]; L=len(w)
        else:
            cur.append(w); L += add
    if cur: lines.append(" ".join(cur))

    # 1 satıra düşerse zorla 2'ye böl
    if len(lines) == 1 and len(words) > 1:
        mid = len(words)//2
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]

    # son rötuş: aşırı uzun satırları kırp
    fixed=[]
    for ln in lines:
        if len(ln) <= max_line_length+8:
            fixed.append(ln)
        else:
            ws = ln.split(); acc=[]; c=0
            for w in ws:
                add = (1 if acc else 0) + len(w)
                if c + add > max_line_length: break
                acc.append(w); c += add
            fixed.append(" ".join(acc))
    return "\n".join([ln.strip() for ln in fixed if ln.strip()][:4])

# -------------------- TTS --------------------
def _rate_to_atempo(rate_str: str, default: float = 1.12) -> float:
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

def tts_to_wav(text: str, wav_out: str) -> float:
    import asyncio
    text = (text or "").strip()
    if not text:
        run(["ffmpeg","-y","-f","lavfi","-t","1.0","-i","anullsrc=r=48000:cl=mono", wav_out])
        return 1.0

    mp3 = wav_out.replace(".wav", ".mp3")
    rate_env = VOICE_RATE
    atempo = _rate_to_atempo(rate_env, default=1.12)
    available = VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])
    selected_voice = VOICE if VOICE in available else available[0]

    async def _edge_save_simple():
        comm = edge_tts.Communicate(text, voice=selected_voice, rate=rate_env)
        await comm.save(mp3)

    for attempt in range(2):
        try:
            try:
                asyncio.run(_edge_save_simple())
            except RuntimeError:
                nest_asyncio.apply(); loop = asyncio.get_event_loop(); loop.run_until_complete(_edge_save_simple())
            run([
                "ffmpeg","-y","-hide_banner","-loglevel","error",
                "-i", mp3, "-ar","48000","-ac","1","-acodec","pcm_s16le",
                "-af", f"dynaudnorm=g=7:f=250,atempo={atempo}",
                wav_out
            ])
            pathlib.Path(mp3).unlink(missing_ok=True)
            return ffprobe_dur(wav_out) or 0.0
        except Exception as e:
            print(f"⚠️ edge-tts deneme {attempt+1}/2 başarısız: {e}"); time.sleep(1.0)

    # fallback
    try:
        q = requests.utils.quote(text.replace('"','').replace("'",""))
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={q}&tl={(LANG or 'en')}&client=tw-ob&ttsspeed=1.0"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=30); r.raise_for_status()
        open(mp3,"wb").write(r.content)
        run([
            "ffmpeg","-y","-hide_banner","-loglevel","error",
            "-i", mp3, "-ar","48000","-ac","1","-acodec","pcm_s16le",
            "-af", f"dynaudnorm=g=6:f=300,atempo={atempo}",
            wav_out
        ])
        pathlib.Path(mp3).unlink(missing_ok=True)
        return ffprobe_dur(wav_out) or 0.0
    except Exception as e2:
        print(f"❌ TTS fallback başarısız: {e2}")
        run(["ffmpeg","-y","-f","lavfi","-t","4.0","-i","anullsrc=r=48000:cl=mono", wav_out])
        return 4.0

# -------------------- Video compose --------------------
CAPTION_COLORS = ["0xFFD700","0xFF6B35","0x00F5FF","0x32CD32","0xFF1493","0x1E90FF","0xFFA500","0xFF69B4"]

def _ff_color(c: str) -> str:
    c = (c or "").strip()
    if c.startswith("#"): return "0x" + c[1:].upper()
    if re.fullmatch(r"0x[0-9A-Fa-f]{6}", c): return c
    return "white"

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
    run(["ffmpeg","-y","-i",src,"-t",f"{dur:.3f}","-vf",vf,"-r",str(TARGET_FPS),"-an",
         "-c:v","libx264","-preset","fast","-crf","22","-pix_fmt","yuv420p", outp])

def draw_capcut_text(seg: str, text: str, color: str, font: str, outp: str, is_hook: bool=False):
    wrapped = wrap_mobile_lines(clean_caption_text(text), CAPTION_MAX_LINE)
    tf = str(pathlib.Path(seg).with_suffix(".caption.txt"))
    pathlib.Path(tf).write_text(wrapped, encoding="utf-8")

    lines = wrapped.count("\n")+1
    fs = 58 if is_hook else 48
    if lines >= 3: fs -= 6
    if lines >= 4: fs -= 6
    y_pos = "h-h/3-text_h/2"
    col = _ff_color(color)
    font_arg = f":fontfile={_ff_sanitize_font(font)}" if font else ""
    common = f"textfile='{tf}':fontsize={fs}:x=(w-text_w)/2:y={y_pos}:line_spacing=10"

    shadow = f"drawtext={common}{font_arg}:fontcolor=black@0.85:borderw=0"
    box    = f"drawtext={common}{font_arg}:fontcolor=white@0.0:box=1:boxborderw={(22 if is_hook else 16)}:boxcolor=black@0.65"
    main   = f"drawtext={common}{font_arg}:fontcolor={col}:borderw={(5 if is_hook else 4)}:bordercolor=black@0.9"
    vf_advanced = f"{shadow},{box},{main}"
    vf_simple   = f"drawtext={common}{font_arg}:fontcolor=white:borderw=3:bordercolor=black@0.85"

    try:
        run(["ffmpeg","-y","-i",seg,"-vf",vf_advanced,"-c:v","libx264","-preset","medium",
             "-crf",str(max(16,CRF_VISUAL-3)),"-movflags","+faststart", outp])
    except Exception as e:
        print(f"⚠️ drawtext advanced failed, fallback simple: {e}")
        run(["ffmpeg","-y","-i",seg,"-vf",vf_simple,"-c:v","libx264","-preset","medium",
             "-crf",str(max(16,CRF_VISUAL-2)),"-movflags","+faststart", outp])
    finally:
        pathlib.Path(tf).unlink(missing_ok=True)

def concat_videos(files: List[str], outp: str):
    lst = str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst,"w",encoding="utf-8") as f:
        for p in files: f.write(f"file '{p}'\n")
    run([
        "ffmpeg","-y","-f","concat","-safe","0","-i",lst,
        "-vsync","cfr","-r",str(TARGET_FPS),
        "-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),
        "-pix_fmt","yuv420p","-movflags","+faststart", outp
    ])

def concat_audios(files: List[str], outp: str):
    lst = str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst,"w",encoding="utf-8") as f:
        for p in files: f.write(f"file '{p}'\n")
    run([
        "ffmpeg","-y","-f","concat","-safe","0","-i",lst,
        "-ar","48000","-ac","1",
        "-af","dynaudnorm=g=6:f=300",
        outp
    ])

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

# -------------------- Gemini (optional) --------------------
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

# -------------------- Per-scene query builder (1–2 kelime) --------------------
_STOP = set("""
a an the and or but if while of to in on at from by with for about into over after before between during under above across around through
this that these those is are was were be been being have has had do does did can could should would may might will shall
you your we our they their he she it its as than then so such very more most many much just also only even still yet
""".split())
_GENERIC_BAD = {"great","good","bad","big","small","old","new","many","more","most","thing","things","stuff"}

def _lower_tokens(s: str) -> List[str]:
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s.lower())
    return [w for w in s.split() if w and len(w)>2 and w not in _STOP and w not in _GENERIC_BAD]

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
    if any(k in t for k in ["garbage","trash","plastic"]):
        s.update(["garbage patch","ocean plastic","plastic pollution","floating debris","sea cleanup","ocean gyre"])
    if "pacific" in t: s.update(["pacific ocean","north pacific"])
    if "brics" in t: s.update(["trade bloc","global south"])
    return list(s)

def build_per_scene_queries(sentences: List[str], fallback_terms: List[str], mode: str, topic: Optional[str]=None) -> List[str]:
    topic = topic or ""
    texts_cap = [topic] + sentences
    texts_all = " ".join([topic] + sentences)

    phrase_pool = _proper_phrases(texts_cap) + _domain_synonyms(texts_all)
    global_tokens=[]
    for s in sentences: global_tokens += _lower_tokens(s)
    seen=set(); global_tokens=[w for w in global_tokens if not (w in seen or seen.add(w))]

    fb=[]
    for t in (fallback_terms or []):
        t = re.sub(r"[^A-Za-z0-9 ]+"," ", str(t)).strip().lower()
        if not t: continue
        ws = [w for w in t.split() if w not in _STOP and w not in _GENERIC_BAD]
        if ws: fb.append(" ".join(ws[:2]))

    queries=[]
    for s in sentences:
        s_low = " " + (s or "").lower() + " "
        picked=None
        for ph in phrase_pool:
            if f" {ph} " in s_low:
                picked = ph; break
        if not picked:
            toks = [w for w in _lower_tokens(s) if w not in _GENERIC_BAD]
            if len(toks)>=2: picked = f"{toks[0]} {toks[1]}"
            elif len(toks)==1: picked = toks[0]
        if not picked and global_tokens: picked = global_tokens[0]
        if not picked and fb: picked = fb[0]
        if picked and len(picked.split())>2:
            ws = picked.split(); picked = f"{ws[-2]} {ws[-1]}"
        if picked in ("great","nice","good","bad",None,""):
            picked = (fb[0] if fb else (global_tokens[0] if global_tokens else "portrait"))
        queries.append(picked)
    return queries

# -------------------- Pexels search (tek odak) --------------------
def _pexels_headers():
    if not PEXELS_API_KEY: raise RuntimeError("PEXELS_API_KEY missing")
    return {"Authorization": PEXELS_API_KEY}

def _pexels_locale(lang: str) -> str:
    return "tr-TR" if lang.startswith("tr") else "en-US"

def pexels_pick_one(query: str) -> Tuple[Optional[int], Optional[str]]:
    headers = _pexels_headers()
    locale  = _pexels_locale(LANG)
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers,
            params={"query": query, "per_page": 15, "orientation":"portrait", "size":"large", "locale": locale},
            timeout=30
        )
        if r.status_code != 200: return None, None
        data = r.json() or {}
        cand = []
        block = _blocklist_get_pexels()
        for v in data.get("videos", []):
            vid = int(v.get("id", 0))
            if vid in block: continue
            files = v.get("video_files", []) or []
            if not files: continue
            best = max(files, key=lambda x: int(x.get("height",0))*int(x.get("width",0)))
            w = int(best.get("width",0)); h = int(best.get("height",0))
            if h < w or h < 1080: continue
            dur = float(v.get("duration",0))
            dur_bonus = 1.0 if 2.0 <= dur <= 12.0 else 0.0
            tokens = set(re.findall(r"[a-z0-9]+", (v.get("url") or "").lower()))
            qtokens= set(re.findall(r"[a-z0-9]+", query.lower()))
            overlap = len(tokens & qtokens)
            score = overlap*2.0 + dur_bonus + (1.0 if h>=1920 else 0.0)
            cand.append((score, vid, best.get("link")))
        if not cand: return None, None
        cand.sort(key=lambda x: x[0], reverse=True)
        top = cand[0]
        _blocklist_add_pexels([top[1]], days=30)
        print(f"   → Pexels pick [{query}] -> id={top[1]} | {top[2]}")
        return top[1], top[2]
    except Exception:
        return None, None

# -------------------- YouTube --------------------
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

    # Content
    if USE_GEMINI and GEMINI_API_KEY:
        banlist = _recent_topics_for_prompt()
        chosen=None; last=None
        for _ in range(8):
            try:
                ctry, tpc, sents, search_terms, ttl, desc, tags = build_via_gemini(MODE, CHANNEL_NAME, banlist)
                last=(ctry,tpc,sents,search_terms,ttl,desc,tags)
                sig = f"{MODE}|{tpc}|{sents[0] if sents else ''}"
                h = _hash12(sig)
                if not _is_recent(h, window_days=180):
                    _record_recent(h, MODE, tpc); chosen=last; break
                else:
                    banlist.insert(0, tpc); time.sleep(1.2)
            except Exception as e:
                print(f"Gemini error: {str(e)[:160]}"); time.sleep(1)
        if chosen is None:
            ctry, tpc, sents, search_terms, ttl, desc, tags = last if last else (
                "World","Amazing Facts",
                ["Did you know Turkey hides a city underground?",
                 "Derinkuyu stretches many levels beneath the surface.",
                 "Ancient builders carved rooms from volcanic rock.",
                 "People once lived safely there for decades.",
                 "The tunnels connect to nearby hidden chambers.",
                 "Ventilation shafts still work remarkably well.",
                 "It’s older than most famous empires.",
                 "Which underground mystery fascinates you most?"],
                ["cappadocia underground","ancient tunnels","turkey caves 4k"], "", "", []
            )
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
        search_terms = ["cappadocia underground","ancient tunnels","turkey caves 4k"]
        ttl = desc = ""; tags=[]

    sentences = sents
    print(f"📝 Content: {ctry} | {tpc}")
    print(f"📊 Sentences: {len(sentences)}")

    # TTS – aynı metin
    tmp = tempfile.mkdtemp(prefix="enhanced_shorts_")
    font = font_path()
    wavs, metas = [], []           # [(text, dur)]
    processed_sentences = []
    print("🎤 TTS…")
    for i, s in enumerate(sentences):
        base = normalize_sentence(s)
        processed_sentences.append(base)
        w = str(pathlib.Path(tmp)/f"sent_{i:02d}.wav")
        d = tts_to_wav(base, w)
        wavs.append(w); metas.append((base, d))
        print(f"   {i+1}/{len(sentences)}: {d:.1f}s")
    sentences = processed_sentences

    # Per-scene queries (1–2 kelime)
    per_scene_queries = build_per_scene_queries(sentences, search_terms, MODE, topic=tpc)
    print("🔎 Per-scene queries:")
    for q in per_scene_queries: print(f"   • {q}")

    # Pexels – tek odak seçim + download
    picked = []
    for q in per_scene_queries:
        vid, link = pexels_pick_one(q)
        if vid and link: picked.append((vid, link))
    if not picked: raise RuntimeError("Pexels: hiçbir sonuç bulunamadı (per-scene).")

    clips=[]
    for idx,(vid,link) in enumerate(picked):
        try:
            f = str(pathlib.Path(tmp)/f"clip_{idx:02d}_{vid}.mp4")
            with requests.get(link, stream=True, timeout=120) as rr:
                rr.raise_for_status()
                with open(f,"wb") as w:
                    for ch in rr.iter_content(8192): w.write(ch)
            if pathlib.Path(f).stat().st_size > 500_000:
                clips.append(f)
        except Exception as e:
            print(f"⚠️ download fail ({vid}): {e}")
    if len(clips) < len(sentences):
        print("⚠️ Yeterli klip yok; eldeki klipler döndürülerek kullanılacak.")

    # Segments + captions (senkron)
    print("🎬 Segments…")
    segs=[]
    for i,(base_text, d) in enumerate(metas):
        base = str(pathlib.Path(tmp)/f"seg_{i:02d}.mp4")
        make_segment(clips[i % len(clips)], d, base)
        colored = str(pathlib.Path(tmp)/f"segsub_{i:02d}.mp4")
        draw_capcut_text(base, base_text, CAPTION_COLORS[i % len(CAPTION_COLORS)], font, colored, is_hook=(i==0))
        segs.append(colored)

    # Assemble
    print("🎞️ Assemble…")
    vcat = str(pathlib.Path(tmp)/"video_concat.mp4"); concat_videos(segs, vcat)
    acat = str(pathlib.Path(tmp)/"audio_concat.wav"); concat_audios(wavs, acat)

    total = ffprobe_dur(acat)
    print(f"📏 Total audio: {total:.1f}s (target {TARGET_MIN_SEC}-{TARGET_MAX_SEC}s)")
    if total < TARGET_MIN_SEC:
        deficit = TARGET_MIN_SEC - total
        extra = min(deficit, 5.0)
        if extra > 0.1:
            padded = str(pathlib.Path(tmp)/"audio_padded.wav")
            run(["ffmpeg","-y","-f","lavfi","-t",f"{extra:.2f}","-i","anullsrc=r=48000:cl=mono","-i",acat,"-filter_complex","[1:a][0:a]concat=n=2:v=0:a=1", padded])
            acat = padded

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^A-Za-z0-9]+','_', tpc)[:60] or "Short"
    outp = f"{OUT_DIR}/{ctry}_{safe_topic}_{ts}.mp4"
    print("🔄 Mux…")
    mux(vcat, acat, outp)
    final = ffprobe_dur(outp)
    print(f"✅ Saved: {outp} ({final:.1f}s)")

    # Metadata
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

    # Upload (opsiyonel; env yoksa atlar)
    try:
        print("📤 Uploading to YouTube…")
        vid_id = upload_youtube(outp, meta)
        print(f"🎉 YouTube Video ID: {vid_id}\n🔗 https://youtube.com/watch?v={vid_id}")
    except Exception as e:
        print(f"❌ Upload skipped: {e}")

    # Cleanup
    try:
        import shutil; shutil.rmtree(tmp); print("🧹 Cleaned temp files")
    except: pass

if __name__ == "__main__":
    main()
