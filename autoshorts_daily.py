# autoshorts_daily.py — Topic-driven guardrails + per-scene Pexels + unique clips + hard A/V lock
# -*- coding: utf-8 -*-
import os, sys, re, json, time, random, datetime, tempfile, pathlib, subprocess, hashlib, math, shutil
from typing import List, Optional, Tuple, Dict

# ===================== ENV / constants =====================
VOICE_STYLE    = os.getenv("TTS_STYLE", "narration-professional")
TARGET_MIN_SEC = float(os.getenv("TARGET_MIN_SEC", "22"))
TARGET_MAX_SEC = float(os.getenv("TARGET_MAX_SEC", "42"))

CHANNEL_NAME   = os.getenv("CHANNEL_NAME", "DefaultChannel").strip()
MODE           = os.getenv("MODE", "").strip().lower() or "general"
LANG           = (os.getenv("LANG", "en") or "en").strip().lower()
VISIBILITY     = os.getenv("VISIBILITY", "public").strip().lower()
ROTATION_SEED  = int(os.getenv("ROTATION_SEED", "0"))
TOPIC_ENV      = os.getenv("TOPIC", "").strip()             # yeni: environment'tan gelebilir
TERMS_ENV      = os.getenv("SEARCH_TERMS", "").strip()      # yeni: virgül/JSON listesi kabul

OUT_DIR        = "out"; pathlib.Path(OUT_DIR).mkdir(exist_ok=True)

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
USE_GEMINI     = os.getenv("USE_GEMINI", "0") == "1"
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

TARGET_FPS       = 25
CRF_VISUAL       = 22

# Caption tuneables
CAPTION_MAX_LINE  = int(os.getenv("CAPTION_MAX_LINE",  "28"))
CAPTION_MAX_LINES = int(os.getenv("CAPTION_MAX_LINES", "6"))

STATE_FILE = f"state_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json"

# ===================== deps (auto-install) =====================
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

# ===================== helpers =====================
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

# -------- state --------
def _state_load() -> dict:
    try: return json.load(open(STATE_FILE, "r", encoding="utf-8"))
    except: return {"recent": [], "used_pexels_ids": []}

def _state_save(st: dict):
    st["recent"] = st.get("recent", [])[-1200:]
    st["used_pexels_ids"] = st.get("used_pexels_ids", [])[-6000:]
    pathlib.Path(STATE_FILE).write_text(json.dumps(st, indent=2), encoding="utf-8")

def _hash12(s: str) -> str: return hashlib.sha1((s or "").encode("utf-8")).hexdigest()[:12]

def _is_recent(h: str, window_days=180) -> bool:
    now = time.time()
    for r in _state_load().get("recent", []):
        if r.get("h")==h and (now - r.get("ts",0)) < window_days*86400:
            return True
    return False

def _record_recent(h: str, topic: str):
    st = _state_load()
    st.setdefault("recent", []).append({"h":h,"topic":topic,"ts":time.time()})
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

def _recent_topics_for_prompt(limit=25) -> List[str]:
    st = _state_load()
    topics = [r.get("topic","") for r in reversed(st.get("recent", [])) if r.get("topic")]
    uniq=[]
    for t in topics:
        if t and t not in uniq: uniq.append(t)
        if len(uniq) >= limit: break
    return uniq

# -------- caption helpers --------
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
        lines = []; buf=[]; L=0
        for w in words:
            add = (1 if buf else 0) + len(w)
            if L + add > width and buf:
                lines.append(" ".join(buf)); buf=[w]; L=len(w)
            else:
                buf.append(w); L += add
        if buf: lines.append(" ".join(buf))
        if len(lines) > k_cap and k_cap < HARD_CAP:
            return greedy(width, HARD_CAP)
        return lines
    lines = greedy(max_line_length, max_lines)
    return "\n".join([ln.strip() for ln in lines if ln.strip()])

# -------- TTS --------
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

VOICE_OPTIONS = {
    "en": ["en-US-JennyNeural","en-US-JasonNeural","en-US-AriaNeural","en-US-GuyNeural","en-GB-SoniaNeural"],
    "tr": ["tr-TR-EmelNeural","tr-TR-AhmetNeural"]
}
VOICE = os.getenv("TTS_VOICE", VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])[0])
VOICE_RATE = os.getenv("TTS_RATE", "+12%")

def tts_to_wav(text: str, wav_out: str) -> float:
    import asyncio
    from aiohttp.client_exceptions import WSServerHandshakeError

    text = (text or "").strip()
    if not text:
        run(["ffmpeg","-y","-f","lavfi","-t","1.0","-i","anullsrc=r=48000:cl=mono", wav_out]); return 1.0
    mp3 = wav_out.replace(".wav", ".mp3")
    atempo = _rate_to_atempo(VOICE_RATE, default=1.15)
    async def _edge_save():
        comm = edge_tts.Communicate(text, voice=VOICE, rate=VOICE_RATE); await comm.save(mp3)
    for attempt in range(2):
        try:
            try: asyncio.run(_edge_save())
            except RuntimeError:
                nest_asyncio.apply(); loop = asyncio.get_event_loop(); loop.run_until_complete(_edge_save())
            run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", mp3,"-ar","48000","-ac","1","-acodec","pcm_s16le","-af", f"dynaudnorm=g=6:f=250,atempo={atempo}", wav_out])
            pathlib.Path(mp3).unlink(missing_ok=True); return ffprobe_dur(wav_out) or 0.0
        except WSServerHandshakeError as e:
            if getattr(e,"status",None)==401 or "401" in str(e): break
            time.sleep(0.7)
        except Exception: time.sleep(0.7)
    # ultra-fallback
    run(["ffmpeg","-y","-f","lavfi","-t","2.0","-i","anullsrc=r=48000:cl=mono", wav_out]); return 2.0

# -------- Video helpers --------
def quantize_to_frames(seconds: float, fps: int = TARGET_FPS) -> Tuple[int, float]:
    frames = max(2, int(round(seconds * fps))); return frames, frames / float(fps)

def make_segment(src: str, dur_s: float, outp: str):
    frames, qdur = quantize_to_frames(dur_s, TARGET_FPS)
    fade = max(0.05, min(0.12, qdur/8.0)); fade_out_st = max(0.0, qdur - fade)
    vf = ("scale=1080:1920:force_original_aspect_ratio=increase,"
          "crop=1080:1920,eq=brightness=0.02:contrast=1.08:saturation=1.08,"
          f"fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB,trim=start_frame=0:end_frame={frames},"
          f"fade=t=in:st=0:d={fade:.2f},fade=t=out:st={fade_out_st:.2f}:d={fade:.2f}")
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", src,"-vf", vf,"-r", str(TARGET_FPS), "-vsync","cfr","-an","-c:v","libx264","-preset","fast","-crf",str(CRF_VISUAL),"-pix_fmt","yuv420p","-movflags","+faststart", outp])

def enforce_video_exact_frames(video_in: str, target_frames: int, outp: str):
    target_frames = max(2, int(target_frames))
    vf = f"fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB,trim=start_frame=0:end_frame={target_frames}"
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", video_in,"-vf", vf,"-r", str(TARGET_FPS), "-vsync","cfr","-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),"-pix_fmt","yuv420p","-movflags","+faststart", outp])

def draw_capcut_text(seg: str, text: str, color: str, font: str, outp: str, is_hook: bool=False):
    wrapped = wrap_mobile_lines(clean_caption_text(text), CAPTION_MAX_LINE, CAPTION_MAX_LINES)
    tf = str(pathlib.Path(seg).with_suffix(".caption.txt")); pathlib.Path(tf).write_text(wrapped, encoding="utf-8")
    seg_dur = ffprobe_dur(seg); frames = max(2, int(round(seg_dur * TARGET_FPS)))
    lines = wrapped.split("\n"); n_lines=max(1,len(lines)); maxchars = max((len(l) for l in lines), default=1)
    base = 60 if is_hook else 50; ratio = CAPTION_MAX_LINE / max(1, maxchars)
    fs = int(base * min(1.0, max(0.50, ratio))); 
    if n_lines>=5: fs=int(fs*0.92)
    if n_lines>=6: fs=int(fs*0.88)
    if n_lines>=7: fs=int(fs*0.84)
    if n_lines>=8: fs=int(fs*0.80)
    fs = max(22, fs)
    y_pos = "(h*0.58 - text_h/2)" if n_lines>=4 else "h-h/3-text_h/2"
    col = _ff_color(color); font_arg = f":fontfile={_ff_sanitize_font(font)}" if font else ""
    common = f"textfile='{tf}':fontsize={fs}:x=(w-text_w)/2:y={y_pos}:line_spacing=10"
    shadow = f"drawtext={common}{font_arg}:fontcolor=black@0.85:borderw=0"
    box    = f"drawtext={common}{font_arg}:fontcolor=white@0.0:box=1:boxborderw={(22 if is_hook else 18)}:boxcolor=black@0.65"
    main   = f"drawtext={common}{font_arg}:fontcolor={col}:borderw={(5 if is_hook else 4)}:bordercolor=black@0.9"
    vf = f"{shadow},{box},{main},fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB,trim=start_frame=0:end_frame={frames}"
    tmp_out = str(pathlib.Path(outp).with_suffix(".tmp.mp4"))
    try:
        run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", seg,"-vf", vf,"-r", str(TARGET_FPS), "-vsync","cfr","-an","-c:v","libx264","-preset","medium","-crf",str(max(16,CRF_VISUAL-3)),"-pix_fmt","yuv420p","-movflags","+faststart", tmp_out])
        enforce_video_exact_frames(tmp_out, frames, outp)
    finally:
        pathlib.Path(tf).unlink(missing_ok=True); pathlib.Path(tmp_out).unlink(missing_ok=True)

def pad_video_to_duration(video_in: str, target_sec: float, outp: str):
    vdur = ffprobe_dur(video_in)
    if vdur >= target_sec - 0.02:
        pathlib.Path(outp).write_bytes(pathlib.Path(video_in).read_bytes()); return
    extra = max(0.0, target_sec - vdur)
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", video_in,"-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={extra:.3f},fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB[v]","-map","[v]","-r", str(TARGET_FPS), "-vsync","cfr","-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),"-pix_fmt","yuv420p","-movflags","+faststart", outp])

def concat_videos_filter(files: List[str], outp: str):
    if not files: raise RuntimeError("concat_videos_filter: empty")
    inputs = []; filters = []
    for i, p in enumerate(files):
        inputs += ["-i", p]; filters.append(f"[{i}:v]fps={TARGET_FPS},settb=AVTB,setpts=N/{TARGET_FPS}/TB[v{i}]")
    filtergraph = ";".join(filters) + ";" + "".join(f"[v{i}]" for i in range(len(files))) + f"concat=n={len(files)}:v=1:a=0[v]"
    run(["ffmpeg","-y","-hide_banner","-loglevel","error", *inputs,"-filter_complex", filtergraph,"-map","[v]","-r", str(TARGET_FPS), "-vsync","cfr","-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),"-pix_fmt","yuv420p","-movflags","+faststart", outp])

def concat_audios(files: List[str], outp: str):
    if not files: raise RuntimeError("concat_audios: empty file list")
    lst = str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst, "w", encoding="utf-8") as f:
        for p in files: f.write(f"file '{p}'\n")
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i", lst,"-c","copy", outp])
    pathlib.Path(lst).unlink(missing_ok=True)

def lock_audio_duration(audio_in: str, target_frames: int, outp: str):
    dur = target_frames / float(TARGET_FPS)
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", audio_in,"-af", f"atrim=end={dur:.6f},asetpts=N/SR/TB","-ar","48000","-ac","1","-c:a","pcm_s16le", outp])

def mux(video: str, audio: str, outp: str):
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", video, "-i", audio,"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","256k","-movflags","+faststart","-muxpreload","0","-muxdelay","0","-avoid_negative_ts","make_zero", outp])

# ===================== Content generation (Gemini optional) =====================
ENHANCED_TEMPLATES = {
    "country_facts": "Create amazing COUNTRY facts. EXACTLY 7–8 short sentences (6–12 words).",
    "history_story": "Create a 7–8 sentence historical micro-story (true, lesser-known).",
    "quotes":        "Pick one famous quote and explain its wisdom in 7 lines.",
    "tech_news":     "Summarize one tech news item with 7 crisp lines (no hype).",
    "space_news":    "Summarize one space discovery with 7 lines (no speculation).",
    "animal_facts":  "Give a single animal behavior/ability explained in 7 lines.",
    "fixit_fast":    "Give one practical, safe how-to routine in 7 steps.",
    "general":       "Write a tight 7–8 line educational explainer.",
}

GENERIC_BAN = ["country facts","mythology","celebrity gossip","politics rant"]

def _gemini_call(prompt: str, model: str) -> dict:
    if not GEMINI_API_KEY: raise RuntimeError("GEMINI_API_KEY missing")
    headers = {"Content-Type":"application/json","x-goog-api-key":GEMINI_API_KEY}
    payload = {"contents":[{"parts":[{"text": prompt}]}], "generationConfig":{"temperature":0.7}}
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

def build_via_gemini(mode: str, channel_name: str, topic_hint: str, ban_recent: List[str]) -> Tuple[str,str,List[str],List[str],str,str,List[str]]:
    template = ENHANCED_TEMPLATES.get(mode, ENHANCED_TEMPLATES["general"])
    avoid = "\n".join(f"- {b}" for b in (ban_recent[:20] or GENERIC_BAN))
    prompt = f"""{template}
Channel: {channel_name}
Language: {LANG}
Channel topic (HARD CONSTRAINT): {topic_hint}
Forbidden / avoid for 180 days:
{avoid}

Return JSON ONLY with keys: country, topic, sentences (array), search_terms (array), title, description, tags.
Sentences MUST reflect the channel topic exactly; no drift.
"""
    data = _gemini_call(prompt, GEMINI_MODEL)
    country = str(data.get("country") or "World").strip()
    topic   = str(data.get("topic") or topic_hint or "Explainer").strip()
    sentences = [clean_caption_text(s) for s in (data.get("sentences") or [])]
    sentences = [s for s in sentences if s][:8]
    terms = data.get("search_terms") or []
    if isinstance(terms, str): terms=[terms]
    terms = [t.strip() for t in terms if isinstance(t,str) and t.strip()]
    title = (data.get("title") or "").strip()
    desc  = (data.get("description") or "").strip()
    tags  = [t.strip() for t in (data.get("tags") or []) if isinstance(t,str) and t.strip()]
    return country, topic, sentences, terms, title, desc, tags

# ===================== Per-scene queries =====================
_STOP = set("""
a an the and or but if while of to in on at from by with for about into over after before between during under above across around through
this that these those is are was were be been being have has had do does did can could should would may might will shall
you your we our they their he she it its as than then so such very more most many much just also only even still yet
""".split())
_GENERIC_BAD = {"great","good","bad","big","small","old","new","many","more","most","thing","things","stuff"}

def _lower_tokens(s: str) -> List[str]:
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", (s or "").lower())
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

def _parse_terms_env(s: str) -> List[str]:
    if not s: return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return [t.strip() for t in s.split(",") if t.strip()]

def build_per_scene_queries(sentences: List[str], fallback_terms: List[str], topic: str) -> List[str]:
    texts_cap = [topic] + sentences
    texts_all = " ".join([topic] + sentences)
    phrase_pool = _proper_phrases(texts_cap)
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
            if f" {ph} " in s_low: picked = ph; break
        if not picked:
            toks = [w for w in _lower_tokens(s) if w not in _GENERIC_BAD]
            if len(toks)>=2: picked = f"{toks[0]} {toks[1]}"
            elif len(toks)==1: picked = toks[0]
        if not picked and global_tokens: picked = global_tokens[0]
        if not picked and fb: picked = fb[0]
        if picked and len(picked.split())>2:
            ws = picked.split(); picked = f"{ws[-2]} {ws[-1]}"
        if not picked: picked = "portrait"
        queries.append(picked)
    return queries

# ===================== Pexels (unique picks + graceful fallback) =====================
_USED_PEXELS_IDS_RUNTIME = set()

def _pexels_headers():
    if not PEXELS_API_KEY: raise RuntimeError("PEXELS_API_KEY missing")
    return {"Authorization": PEXELS_API_KEY}

def _pexels_locale(lang: str) -> str:
    return "tr-TR" if lang.startswith("tr") else "en-US"

def _pexels_search(q: str, page: int) -> dict:
    r = requests.get("https://api.pexels.com/videos/search",
                     headers=_pexels_headers(),
                     params={"query": q, "per_page": 30, "orientation":"portrait", "size":"large",
                             "locale": _pexels_locale(LANG), "page": page},
                     timeout=30)
    if r.status_code != 200: return {}
    return r.json() or {}

def _best_link(video: dict) -> Optional[Tuple[int,str,int,int,float]]:
    vid = int(video.get("id", 0)); files = video.get("video_files", []) or []
    if not files: return None
    # dikey ve 1080p üstü öncelik
    cand=[f for f in files if str(f.get("file_type","")).endswith("mp4")]
    if not cand: return None
    def score(f):
        w=int(f.get("width",0)); h=int(f.get("height",0))
        portrait = 2.0 if h>=w and h>=1080 else 0.0
        return portrait + min(h,1920)/1080.0
    cand.sort(key=score, reverse=True)
    f=cand[0]; return (vid, f.get("link"), int(f.get("width",0)), int(f.get("height",0)), float(video.get("duration",0)))

def pexels_pick_many(queries: List[str], need: int) -> List[Tuple[int,str]]:
    block = _blocklist_get_pexels(); picks=[]; tried=set()
    pages=[1,2,3,4,5]
    for q in queries:
        for page in random.sample(pages, k=len(pages)):
            key=f"{q}|{page}"
            if key in tried: continue
            tried.add(key)
            data=_pexels_search(q, page)
            vids=data.get("videos",[]) or []
            random.shuffle(vids)
            for v in vids:
                best=_best_link(v)
                if not best: continue
                vid,link,w,h,dur = best
                if vid in block or vid in _USED_PEXELS_IDS_RUNTIME: continue
                # çok kısa/çok uzun temizliği
                if dur<2.0 or dur>18.0: continue
                _USED_PEXELS_IDS_RUNTIME.add(vid); _blocklist_add_pexels([vid], days=30)
                picks.append((vid,link))
                if len(picks)>=need: return picks
            time.sleep(0.35)
    return picks

# ===================== YouTube =====================
def yt_service():
    cid  = os.getenv("YT_CLIENT_ID"); csec = os.getenv("YT_CLIENT_SECRET"); rtok = os.getenv("YT_REFRESH_TOKEN")
    if not (cid and csec and rtok): raise RuntimeError("Missing YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN")
    creds = Credentials(token=None, refresh_token=rtok, token_uri="https://oauth2.googleapis.com/token",
                        client_id=cid, client_secret=csec, scopes=["https://www.googleapis.com/auth/youtube.upload"])
    creds.refresh(Request()); return build("youtube", "v3", credentials=creds, cache_discovery=False)

def upload_youtube(video_path: str, meta: dict) -> str:
    y = yt_service()
    body = {"snippet": {"title": meta["title"], "description": meta["description"], "tags": meta.get("tags", []),
                        "categoryId": "27","defaultLanguage": meta.get("defaultLanguage", LANG),
                        "defaultAudioLanguage": meta.get("defaultAudioLanguage", LANG)},
            "status": {"privacyStatus": meta.get("privacy", VISIBILITY), "selfDeclaredMadeForKids": False}}
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    req = y.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = req.execute(); return resp.get("id","")

# ===================== MAIN =====================
def main():
    print(f"==> {CHANNEL_NAME} | MODE={MODE} | topic-first build")

    # 1) İçerik (topic guard)
    topic_hint = TOPIC_ENV or MODE.replace("_"," ")
    env_terms = _parse_terms_env(TERMS_ENV)

    if USE_GEMINI and GEMINI_API_KEY:
        ban_recent = _recent_topics_for_prompt()
        chosen, last = None, None
        for _ in range(6):
            try:
                ctry, tpc, sents, search_terms, ttl, desc, tags = build_via_gemini(MODE, CHANNEL_NAME, topic_hint, ban_recent)
                last = (ctry, tpc, sents, search_terms, ttl, desc, tags)
                sig = f"{MODE}|{tpc}|{sents[0] if sents else ''}"
                h = _hash12(sig)
                if not _is_recent(h, window_days=180):
                    _record_recent(h, tpc); chosen = last; break
                else:
                    ban_recent.insert(0, tpc); time.sleep(0.6)
            except Exception as e:
                print(f"Gemini error: {str(e)[:160]}"); time.sleep(0.5)
        if chosen is None:
            ctry, tpc, sents, search_terms, ttl, desc, tags = last if last else ("World", topic_hint, [
                "One striking fact stated clearly.",
                "Explain the core idea in simple words.",
                "Give a concrete example people recognize.",
                "Add one number or comparison for weight.",
                "Offer a short tip or implication.",
                "Avoid hype; stay precise and visual.",
                "End with a short question to engage.",
            ], env_terms or [topic_hint], "", "", [])
    else:
        ctry, tpc = "World", (TOPIC_ENV or topic_hint or "Explainer").title()
        sents = [
            "One striking fact stated clearly.",
            "Explain the core idea in simple words.",
            "Give a concrete example people recognize.",
            "Add one number or comparison for weight.",
            "Offer a short tip or implication.",
            "Avoid hype; stay precise and visual.",
            "End with a short question to engage.",
        ]
        search_terms = env_terms or [topic_hint]
        ttl = desc = ""; tags = []

    sentences = [normalize_sentence(x) for x in sents if x]
    print(f"📝 Content: {ctry} | {tpc} | {len(sentences)} lines")

    # 2) TTS
    tmp = tempfile.mkdtemp(prefix="shorts_")
    font = font_path()
    wavs, metas = [], []
    for i, s in enumerate(sentences):
        w = str(pathlib.Path(tmp)/f"sent_{i:02d}.wav"); d = tts_to_wav(s, w)
        wavs.append(w); metas.append((s, d)); print(f"   TTS {i+1}/{len(sentences)}: {d:.2f}s")

    # 3) Pexels — sahneye özel ve benzersiz klip zorunluluğu
    per_scene_queries = build_per_scene_queries(sentences, (search_terms or []) + env_terms, tpc)
    print("🔎 per-scene queries:", per_scene_queries)

    # ilk tur
    need = len(sentences)
    picks = pexels_pick_many(per_scene_queries, need)

    # eksikse ikinci tur: topic + env_terms + generics
    if len(picks) < need:
        broaden = list(dict.fromkeys(
            per_scene_queries + (search_terms or []) + env_terms + [tpc, MODE.replace("_"," "), "b-roll vertical"]
        ))
        more = pexels_pick_many(broaden, need - len(picks))
        picks.extend(more)

    # hâlâ eksikse zarif pas (job düşmez)
    if not picks:
        print("⚠️ No Pexels results; skipping render gracefully.")
        return

    # 4) Klipleri indir + segment + altyazı
    clips=[]
    for idx,(vid,link) in enumerate(picks):
        f = str(pathlib.Path(tmp)/f"clip_{idx:02d}_{vid}.mp4")
        try:
            with requests.get(link, stream=True, timeout=120) as rr:
                rr.raise_for_status()
                with open(f,"wb") as w:
                    for ch in rr.iter_content(8192): w.write(ch)
            if pathlib.Path(f).stat().st_size > 300_000:
                clips.append(f)
        except Exception as e:
            print(f"⚠️ download fail {vid}: {e}")

    if not clips:
        print("⚠️ All downloads failed; skipping gracefully."); return

    segs=[]
    for i,(text,dur) in enumerate(metas):
        base = str(pathlib.Path(tmp)/f"seg_{i:02d}.mp4")
        make_segment(clips[i % len(clips)], dur, base)
        colored = str(pathlib.Path(tmp)/f"segsub_{i:02d}.mp4")
        draw_capcut_text(base, text, CAPTION_COLORS[i % len(CAPTION_COLORS)], font, colored, is_hook=(i==0))
        segs.append(colored)

    # 5) Assemble video + audio
    vcat = str(pathlib.Path(tmp)/"video_concat.mp4"); concat_videos_filter(segs, vcat)
    acat = str(pathlib.Path(tmp)/"audio_concat.wav"); concat_audios(wavs, acat)

    # 6) Lock A/V durations
    adur = ffprobe_dur(acat); a_frames = max(2, int(round(adur * TARGET_FPS)))
    vcat_exact = str(pathlib.Path(tmp)/"video_exact.mp4")
    enforce_video_exact_frames(vcat, a_frames, vcat_exact); vcat = vcat_exact
    acat_exact = str(pathlib.Path(tmp)/"audio_exact.wav")
    lock_audio_duration(acat, a_frames, acat_exact); acat = acat_exact
    print(f"🔒 Locked A/V frames={a_frames} | fps={TARGET_FPS}")

    # 7) Mux
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^A-Za-z0-9]+', '_', tpc)[:60] or "Short"
    outp = f"{OUT_DIR}/{CHANNEL_NAME}_{safe_topic}_{ts}.mp4"
    mux(vcat, acat, outp); final = ffprobe_dur(outp)
    print(f"✅ Saved: {outp} ({final:.2f}s)")

    # 8) Metadata + Upload
    title = (ttl or (sentences[0].rstrip(" .!?") + " — " + tpc))[:95]
    description = (desc or ("• " + "\n• ".join(sentences[:6])))[:4900]
    tags = (tags[:15] if isinstance(tags,list) else [])
    meta = {"title": title, "description": description, "tags": tags, "privacy": VISIBILITY,
            "defaultLanguage": LANG, "defaultAudioLanguage": LANG}
    try:
        print("📤 Uploading to YouTube…")
        vid_id = upload_youtube(outp, meta)
        print(f"🎉 Video ID: {vid_id}  https://youtube.com/watch?v={vid_id}")
    except Exception as e:
        print(f"❌ Upload skipped: {e}")

    # 9) cleanup
    try: shutil.rmtree(tmp); print("🧹 Cleaned temp files")
    except: pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("⛔ Error:", str(e)[:400]); sys.exit(1)
