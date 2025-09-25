# autoshorts_daily.py — Topic-locked Gemini • Multi-clip Pexels • Hard A/V lock
# Kokoro TTS (primary) • WORD-LEVEL ASS captions (3–4 word window, live highlight) • Music via URL/dir + sidechain ducking
# -*- coding: utf-8 -*-
import os, sys, re, json, time, random, datetime, tempfile, pathlib, subprocess, hashlib, math, shutil
from typing import List, Optional, Tuple, Dict

# =============================================================================
# ENV / constants
# =============================================================================
VOICE_STYLE    = os.getenv("TTS_STYLE", "narration-professional")
TARGET_MIN_SEC = float(os.getenv("TARGET_MIN_SEC", "22"))
TARGET_MAX_SEC = float(os.getenv("TARGET_MAX_SEC", "42"))

CHANNEL_NAME   = os.getenv("CHANNEL_NAME", "DefaultChannel")
MODE           = os.getenv("MODE", "freeform").strip().lower()
LANG           = os.getenv("LANG", "en")
VISIBILITY     = os.getenv("VISIBILITY", "public")
ROTATION_SEED  = int(os.getenv("ROTATION_SEED", "0"))
OUT_DIR        = "out"; pathlib.Path(OUT_DIR).mkdir(exist_ok=True)

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_GEMINI     = os.getenv("USE_GEMINI", "1") == "1"
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ---- TTS backends
TTS_ENGINE   = os.getenv("TTS_ENGINE", "kokoro").lower()  # kokoro | edge
KOKORO_URL   = os.getenv("KOKORO_URL", "").strip()        # POST {text, voice} -> audio bytes
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sky")

# ---- Subtitles
SUBTITLES_MODE  = os.getenv("SUBTITLES_MODE", "ass").lower()     # ass | drawtext
WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "small")            # tiny/base/small/medium
SUB_CHUNK_WORDS = max(1, min(6, int(os.getenv("SUB_CHUNK_WORDS","3"))))  # ekranda aynı anda kaç kelime

# ---- Music (mood + remote)
USE_MUSIC      = os.getenv("USE_MUSIC", "1") == "1"
MUSIC_DIR      = os.getenv("MUSIC_DIR", "music")
MUSIC_MOOD     = os.getenv("MUSIC_MOOD", "cinematic")
MUSIC_GAIN_DB  = os.getenv("MUSIC_GAIN_DB", "-13")
DUCKING        = os.getenv("DUCKING", "1") == "1"
MUSIC_URL      = os.getenv("MUSIC_URL","").strip()
def _parse_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s: return []
    try:
        data = json.loads(s)
        if isinstance(data, list): return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    return [p.strip() for p in re.split(r"\s*,\s*", s) if p.strip()]
MUSIC_URLS     = _parse_list(os.getenv("MUSIC_URLS",""))

# ---- Audio smoothing
USE_AUDIO_XFADE = os.getenv("USE_AUDIO_XFADE", "1") == "1"
AUDIO_XFADE_MS  = float(os.getenv("AUDIO_XFADE_MS", "0.08"))

# ---- Visuals / encoding
TARGET_FPS       = 25
CRF_VISUAL       = 22
CAPTION_MAX_LINE  = int(os.getenv("CAPTION_MAX_LINE",  "28"))
CAPTION_MAX_LINES = int(os.getenv("CAPTION_MAX_LINES", "6"))
VFX_XFADE_FIRST   = os.getenv("VFX_XFADE_FIRST", "1") == "1"
VFX_XFADE_MS      = float(os.getenv("VFX_XFADE_MS", "0.22"))

# ---- Topic & search terms
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

# ---- Pexels tuning
PEXELS_PER_PAGE          = int(os.getenv("PEXELS_PER_PAGE", "30"))
PEXELS_MAX_USES_PER_CLIP = int(os.getenv("PEXELS_MAX_USES_PER_CLIP", "1"))
PEXELS_ALLOW_LANDSCAPE   = os.getenv("PEXELS_ALLOW_LANDSCAPE", "1") == "1"

STATE_FILE = f"state_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json"
GLOBAL_TOPIC_STATE = "state_global_topics.json"

# =============================================================================
# deps (auto-install)
# =============================================================================
def _pip(p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=True)
try: import requests
except ImportError: _pip("requests"); import requests
try: import edge_tts, nest_asyncio
except ImportError: _pip("edge-tts"); _pip("nest_asyncio"); import edge_tts, nest_asyncio
# YouTube
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
# Whisper
try:
    from faster_whisper import WhisperModel
except ImportError:
    _pip("faster-whisper"); from faster_whisper import WhisperModel

# =============================================================================
# Voices (Edge fallback)
# =============================================================================
VOICE_OPTIONS = {
    "en": [
        "en-US-JennyNeural","en-US-JasonNeural","en-US-AriaNeural","en-US-GuyNeural",
        "en-AU-NatashaNeural","en-GB-SoniaNeural","en-CA-LiamNeural","en-US-DavisNeural","en-US-AmberNeural"
    ],
    "tr": ["tr-TR-EmelNeural","tr-TR-AhmetNeural"]
}
VOICE = os.getenv("TTS_VOICE", VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])[0])

# =============================================================================
# Utils
# =============================================================================
def run(cmd, check=True):
    res = subprocess.run(cmd, text=True, capture_output=True)
    if check and res.returncode != 0:
        raise RuntimeError(res.stderr[:4000])
    return res

def ffprobe_dur(p):
    try:
        out = run(["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0", p]).stdout.strip()
        return float(out) if out else 0.0
    except: return 0.0

def font_path():
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "C:/Windows/Fonts/arial.ttf"]:
        if pathlib.Path(p).exists(): return p
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

# =============================================================================
# State
# =============================================================================
def _load_json(path, default):
    try: return json.load(open(path, "r", encoding="utf-8"))
    except: return default
def _save_json(path, data):
    pathlib.Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
def _state_load() -> dict:
    return _load_json(STATE_FILE, {"recent": [], "used_pexels_ids": []})
def _state_save(st: dict):
    st["recent"] = st.get("recent", [])[-1200:]
    st["used_pexels_ids"] = st.get("used_pexels_ids", [])[-5000:]
    _save_json(STATE_FILE, st)
def _global_topics_load() -> dict:
    return _load_json(GLOBAL_TOPIC_STATE, {"recent_topics": []})
def _global_topics_save(gst: dict):
    gst["recent_topics"] = gst.get("recent_topics", [])[-4000:]; _save_json(GLOBAL_TOPIC_STATE, gst)
def _hash12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
def _is_recent(h: str, window_days=365) -> bool:
    now = time.time()
    for r in _state_load().get("recent", []):
        if r.get("h")==h and (now - r.get("ts",0)) < window_days*86400: return True
    return False
def _record_recent(h: str, mode: str, topic: str):
    st = _state_load(); st.setdefault("recent", []).append({"h":h,"mode":mode,"topic":topic,"ts":time.time()}); _state_save(st)
    gst = _global_topics_load()
    if topic and topic not in gst["recent_topics"]:
        gst["recent_topics"].append(topic); _global_topics_save(gst)
def _blocklist_add_pexels(ids: List[int], days=30):
    st = _state_load(); now = int(time.time())
    for vid in ids: st.setdefault("used_pexels_ids", []).append({"id": int(vid), "ts": now})
    cutoff = now - days*86400
    st["used_pexels_ids"] = [x for x in st.get("used_pexels_ids", []) if x.get("ts",0) >= cutoff]; _save_json(STATE_FILE, st)
def _blocklist_get_pexels() -> set:
    return {int(x["id"]) for x in _state_load().get("used_pexels_ids", [])}
def _recent_topics_for_prompt(limit=20) -> List[str]:
    topics = list(reversed(_global_topics_load().get("recent_topics", [])))
    uniq=[]; 
    for t in topics:
        if t and t not in uniq: uniq.append(t)
        if len(uniq) >= limit: break
    return uniq

# =============================================================================
# Caption text & wrap (kept)
# =============================================================================
CAPTION_COLORS = ["0xFFD700","0xFF6B35","0x00F5FF","0x32CD32","0xFF1493","0x1E90FF","0xFFA500","0xFF69B4"]
def _ff_color(c: str) -> str:
    c = (c or "").strip()
    if c.startswith("#"): return "0x" + c[1:].upper()
    if re.fullmatch(r"0x[0-9A-Fa-f]{6}", c): return c
    return "white"
def clean_caption_text(s: str) -> str:
    t = (s or "").strip()
    t = (t.replace("—","-").replace("–","-").replace("“",'"').replace("”",'"').replace("’","'").replace("`",""))
    t = re.sub(r"\s+"," ", t).strip()
    if t and t[0].islower(): t = t[0].upper() + t[1:]
    return t
def wrap_mobile_lines(text: str, max_line_length: int = CAPTION_MAX_LINE, max_lines: int = CAPTION_MAX_LINES) -> str:
    text = (text or "").strip()
    if not text: return text
    words = text.split(); HARD_CAP = max_lines + 2
    def distribute_into(k: int) -> list[str]:
        per = math.ceil(len(words) / k)
        chunks = [" ".join(words[i*per:(i+1)*per]) for i in range(k)]
        return [c for c in chunks if c]
    for k in range(2, max_lines + 1):
        cand = distribute_into(k)
        if cand and all(len(c) <= max_line_length for c in cand): return "\n".join(cand)
    def greedy(width: int, k_cap: int) -> list[str]:
        lines=[]; buf=[]; L=0
        for w in words:
            add=(1 if buf else 0)+len(w)
            if L + add > width and buf: lines.append(" ".join(buf)); buf=[w]; L=len(w)
            else: buf.append(w); L+=add
        if buf: lines.append(" ".join(buf))
        if len(lines) > k_cap and k_cap < HARD_CAP: return greedy(max_line_length, HARD_CAP)
        return lines
    lines = greedy(max_line_length, max_lines)
    return "\n".join([ln.strip() for ln in lines if ln.strip()])

# =============================================================================
# TTS
# =============================================================================
def _rate_to_atempo(rate_str: str, default: float = 1.10) -> float:
    try:
        if not rate_str: return default
        rate_str = rate_str.strip()
        if rate_str.endswith("%"): return max(0.5, min(2.0, 1.0 + float(rate_str[:-1])/100.0))
        if rate_str.endswith(("x","X")): return max(0.5, min(2.0, float(rate_str[:-1])))
        return max(0.5, min(2.0, float(rate_str)))
    except: return default

def _tts_edge_to_mp3(text: str, mp3_out: str, rate_env: str, selected_voice: str):
    import asyncio
    from aiohttp.client_exceptions import WSServerHandshakeError
    async def _edge_save_simple():
        comm = edge_tts.Communicate(text, voice=selected_voice, rate=rate_env)
        await comm.save(mp3_out)
    for attempt in range(2):
        try:
            try: asyncio.run(_edge_save_simple())
            except RuntimeError:
                nest_asyncio.apply(); loop = asyncio.get_event_loop(); loop.run_until_complete(_edge_save_simple())
            return True
        except WSServerHandshakeError as e:
            if getattr(e,"status",None)==401 or "401" in str(e): break
            time.sleep(0.8)
        except Exception: time.sleep(0.8)
    return False

def _tts_kokoro_http(text: str, tmp_out: str) -> bool:
    if not KOKORO_URL: return False
    try:
        r = requests.post(KOKORO_URL, json={"text": text, "voice": KOKORO_VOICE}, timeout=60); r.raise_for_status()
        raw = r.content
        tmp = pathlib.Path(tmp_out); tmp.write_bytes(raw)
        # normalize unknown input → wav 48k mono
        wav = str(tmp.with_suffix(".wav"))
        run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", str(tmp), "-ar","48000","-ac","1","-acodec","pcm_s16le", wav])
        tmp.unlink(missing_ok=True)
        pathlib.Path(tmp_out).write_bytes(pathlib.Path(wav).read_bytes()); pathlib.Path(wav).unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"⚠️ kokoro http fail: {e}"); return False

def tts_to_wav(text: str, wav_out: str) -> float:
    text = (text or "").strip()
    if not text:
        run(["ffmpeg","-y","-f","lavfi","-t","1.0","-i","anullsrc=r=48000:cl=mono", wav_out]); return 1.0
    rate_env = os.getenv("TTS_RATE", "+12%"); atempo=_rate_to_atempo(rate_env,1.12)
    tmp_in = str(pathlib.Path(wav_out).with_suffix(".tts_in.wav"))
    used=False
    if TTS_ENGINE=="kokoro" and _tts_kokoro_http(text,tmp_in): used=True
    if not used:
        avail=VOICE_OPTIONS.get(LANG,["en-US-JennyNeural"])
        mp3=str(pathlib.Path(wav_out).with_suffix(".mp3"))
        if _tts_edge_to_mp3(text, mp3, rate_env, avail[0]):
            run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", mp3,"-ar","48000","-ac","1","-acodec","pcm_s16le", tmp_in]); pathlib.Path(mp3).unlink(missing_ok=True); used=True
    if not used:
        try:
            q = requests.utils.quote(text.replace('"','').replace("'","")); url=f"https://translate.google.com/translate_tts?ie=UTF-8&q={q}&tl={(LANG or 'en')}&client=tw-ob&ttsspeed=1.0"
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=30); r.raise_for_status()
            mp3=str(pathlib.Path(wav_out).with_suffix(".mp3")); open(mp3,"wb").write(r.content)
            run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", mp3,"-ar","48000","-ac","1","-acodec","pcm_s16le", tmp_in]); pathlib.Path(mp3).unlink(missing_ok=True)
        except Exception as e2:
            print(f"❌ TTS failed: {e2}"); run(["ffmpeg","-y","-f","lavfi","-t","4.0","-i","anullsrc=r=48000:cl=mono", wav_out]); return 4.0
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", tmp_in,"-ar","48000","-ac","1","-acodec","pcm_s16le","-af", f"dynaudnorm=g=7:f=250,atempo={atempo}", wav_out])
    pathlib.Path(tmp_in).unlink(missing_ok=True)
    return ffprobe_dur(wav_out) or 0.0

# =============================================================================
# Video helpers
# =============================================================================
def quantize_to_frames(seconds: float, fps: int = TARGET_FPS) -> Tuple[int,float]:
    frames=max(2,int(round(seconds*fps))); return frames, frames/float(fps)
def make_segment(src: str, dur_s: float, outp: str):
    frames,qdur=quantize_to_frames(dur_s,TARGET_FPS); fade=max(0.05,min(0.12,qdur/8.0)); fade_out=max(0.0,qdur-fade)
    vf=("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,eq=brightness=0.02:contrast=1.08:saturation=1.1,"
        f"fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB,trim=start_frame=0:end_frame={frames},"
        f"fade=t=in:st=0:d={fade:.2f},fade=t=out:st={fade_out:.2f}:d={fade:.2f}")
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",src,"-vf",vf,"-r",str(TARGET_FPS),"-vsync","cfr","-an","-c:v","libx264","-preset","fast","-crf",str(CRF_VISUAL),"-pix_fmt","yuv420p","-movflags","+faststart",outp])
def enforce_video_exact_frames(video_in: str, target_frames: int, outp: str):
    vf=f"fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB,trim=start_frame=0:end_frame={max(2,int(target_frames))}"
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",video_in,"-vf",vf,"-r",str(TARGET_FPS),"-vsync","cfr","-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),"-pix_fmt","yuv420p","-movflags","+faststart",outp])

# =============================================================================
# ASS subtitles — WORD-LEVEL WINDOW (active word = blue pill)
# =============================================================================
_WHISPER_SINGLETON=None
def _whisper_model():
    global _WHISPER_SINGLETON
    if _WHISPER_SINGLETON is None:
        _WHISPER_SINGLETON = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _WHISPER_SINGLETON

def _ass_color(a,b,g,r): return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"

def transcribe_words(wav_path: str, lang_hint: str="en") -> List[Tuple[float,float,str]]:
    model=_whisper_model()
    segs,_ = model.transcribe(wav_path, language=(lang_hint[:2] if lang_hint else "en"),
                              vad_filter=True, vad_parameters=dict(min_silence_duration_ms=180),
                              word_timestamps=True)
    words=[]
    for s in segs:
        if not getattr(s,'words',None): continue
        for w in s.words:
            st=float(getattr(w,'start',0.0) or 0.0); en=float(getattr(w,'end',st+0.01) or st+0.01)
            tok=str(getattr(w,'word','')).strip()
            tok = re.sub(r"[^\w'-]+","", tok).upper()
            if tok: words.append((st,en,tok))
    return words

def make_ass_word_window(words: List[Tuple[float,float,str]], ass_path: str, chunk_words: int = 3):
    # styles
    font = _ff_sanitize_font(font_path())
    size = 56  # büyük, Shorts için
    primary_white = _ass_color(0x00,0xFF,0xFF,0xFF)
    outline_black = _ass_color(0x00,0x00,0x00,0x00)
    back_blue     = _ass_color(0x20,175,64,30)   # #1E40AF, hafif alpha
    back_none     = _ass_color(0xFF,0,0,0)

    header = "[Script Info]\nScriptType: v4.00+\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n" \
             "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, " \
             "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
    style_base = f"Style: Base,{pathlib.Path(font).name if font else 'Arial'},{size},{primary_white},{primary_white},{outline_black},{back_none},0,0,0,0,100,100,0,0,1,2,2,2,28,20,44,1"
    style_hl   = f"Style: HL,{pathlib.Path(font).name if font else 'Arial'},{size},{primary_white},{primary_white},{outline_black},{back_blue},0,0,0,0,100,100,0,0,3,2,1,2,28,20,44,1"
    events = "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    def ts(t):
        h=int(t//3600); m=int((t%3600)//60); s=t%60
        return f"{h:d}:{m:02d}:{s:05.2f}".replace(".",",")

    lines=[]
    if not words:
        lines.append(f"Dialogue: 0,0:00:00.00,0:00:01.00,Base,,0,0,0,,{{\\rHL}}HI")
    else:
        k = max(1, chunk_words)
        n = len(words)
        i = 0
        while i < n:
            window = words[i:i+k]
            tokens = [t for _,_,t in window]
            # her kelime için: o kelime HL, diğerleri Base
            for j,(st,en,_) in enumerate(window):
                parts=[]
                for idx,tok in enumerate(tokens):
                    parts.append(("{\\rHL}"+tok+"{\\rBase}") if idx==j else tok)
                text_ass = " ".join(parts)
                lines.append(f"Dialogue: 0,{ts(st)},{ts(en)},Base,,0,0,0,,{text_ass}")
            i += k

    pathlib.Path(ass_path).write_text(header + style_base + "\n" + style_hl + "\n\n" + events + "\n".join(lines), encoding="utf-8")

def overlay_subtitles(seg_in: str, ass_path: str, seg_out: str):
    ass_q = str(pathlib.Path(ass_path).as_posix()).replace(":", r"\:")
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", seg_in, "-vf", f"subtitles='{ass_q}'",
        "-r", str(TARGET_FPS), "-vsync","cfr",
        "-an","-c:v","libx264","-preset","medium","-crf",str(max(16,CRF_VISUAL-3)),
        "-pix_fmt","yuv420p","-movflags","+faststart", seg_out
    ])

# =============================================================================
# Music (remote/local) + sidechain ducking
# =============================================================================
def _download(url: str, to_path: str) -> Optional[str]:
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(to_path,"wb") as f:
                for ch in r.iter_content(8192): f.write(ch)
        return to_path if pathlib.Path(to_path).stat().st_size>64_000 else None
    except Exception as e:
        print(f"⚠️ music download fail: {e}"); return None

def pick_music(mood: str) -> Optional[str]:
    # remote first
    tmpdir = tempfile.mkdtemp(prefix="music_")
    if MUSIC_URL:
        p=str(pathlib.Path(tmpdir)/"track.bin"); d=_download(MUSIC_URL,p)
        if d: return d
    if MUSIC_URLS:
        urls = MUSIC_URLS[:]; random.shuffle(urls)
        for u in urls:
            p=str(pathlib.Path(tmpdir)/("t_"+_hash12(u)+".bin")); d=_download(u,p)
            if d: return d
    # local fallback (by mood hint)
    mdir=pathlib.Path(MUSIC_DIR)
    if mdir.exists():
        cand=[p for p in mdir.glob("**/*") if p.suffix.lower() in (".mp3",".wav",".m4a",".flac")]
        if cand:
            mood=(mood or "").lower()
            cand.sort(key=lambda p:(0 if mood and mood in p.stem.lower() else 1, -p.stat().st_size))
            return str(cand[0])
    return None

def mix_voice_and_music(voice_wav: str, music_in: str, out_wav: str, duck=True, gain_db="-13"):
    if duck:
        fc=(f"[1:a]volume={gain_db}dB,aloop=loop=999999:size=40000000:start=0,apad[a1];"
            f"[0:a][a1]sidechaincompress=threshold=-28dB:ratio=6:attack=12:release=180:makeup=6:mix=0.7[a]")
        run(["ffmpeg","-y","-i",voice_wav,"-stream_loop","-1","-i",music_in,"-filter_complex",fc,"-map","[a]","-ar","48000","-ac","1","-c:a","pcm_s16le",out_wav])
    else:
        run(["ffmpeg","-y","-i",voice_wav,"-stream_loop","-1","-i",music_in,"-filter_complex",
             f"[1:a]volume={gain_db}dB,apad[a1];[0:a][a1]amix=inputs=2:duration=first:dropout_transition=0[a]",
             "-map","[a]","-ar","48000","-ac","1","-c:a","pcm_s16le",out_wav])

# =============================================================================
# Template selection / Gemini / search queries (unchanged from your file)
# =============================================================================
def _select_template_key(topic: str) -> str:
    t = (topic or "").lower()
    if any(k in t for k in ("country","geograph","city","capital","border","population","continent","flag")):
        return "country_facts"
    return "_default"

ENHANCED_GEMINI_TEMPLATES = {
    "_default": """Create a 25–40s YouTube Short.
Return STRICT JSON with keys: topic, sentences (7–8), search_terms (4–10), title, description, tags.

CONTENT RULES:
- Stay laser-focused on the provided TOPIC (no pivoting).
- Coherent, causally linked beats; each sentence advances a single concrete idea.
- No meta-instructions like "one clear tip", "see/learn", "plot twist", "soap-opera narration".
- Sentences must be visually anchorable with stock b-roll (objects, places, actions, phenomena).
- Avoid vague fillers (nice/great/thing/stuff). No list headers. No numbering in sentences.
- Keep 6–12 words per sentence. 7–8 sentences total.""",
    "country_facts": """Create amazing country/city facts.
Return STRICT JSON with keys: topic, sentences (7–8), search_terms (4–10), title, description, tags.
Rules:
- Facts must be specific (culture, geography, record-holding places, history).
- No meta-instructions (no "one clear tip", "see/learn").
- 6–12 words per sentence; 7–8 sentences total; visually anchorable beats."""
}
BANNED_PHRASES = ["one clear tip","see it","learn it","plot twist","soap-opera narration","repeat once","takeaway action","in 60 seconds","just the point","crisp beats"]

def _content_score(sentences: List[str]) -> float:
    if not sentences: return 0.0
    bad=0
    for s in sentences:
        low=(s or "").lower()
        if any(bp in low for bp in BANNED_PHRASES): bad+=1
        if len(low.split())<5: bad+=0.5
    return max(0.0, 10.0 - (bad*1.4))

def _gemini_call(prompt: str, model: str) -> dict:
    if not GEMINI_API_KEY: raise RuntimeError("GEMINI_API_KEY missing")
    headers={"Content-Type":"application/json","x-goog-api-key":GEMINI_API_KEY}
    payload={"contents":[{"parts":[{"text": prompt}]}], "generationConfig":{"temperature":0.75}}
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r=requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code!=200: raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
    data=r.json()
    try: txt=data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception: txt=json.dumps(data)
    m=re.search(r"\{(?:.|\n)*\}", txt)
    if not m: raise RuntimeError("Gemini response parse error (no JSON)")
    raw=re.sub(r"^```json\s*|\s*```$", "", m.group(0).strip(), flags=re.MULTILINE)
    return json.loads(raw)

def build_via_gemini(channel_name: str, topic_lock: str, user_terms: List[str], banlist: List[str]) -> Tuple[str,List[str],List[str],str,str,List[str]]:
    tpl_key=_select_template_key(topic_lock); template=ENHANCED_GEMINI_TEMPLATES[tpl_key]
    avoid="\n".join(f"- {b}" for b in banlist[:15]) if banlist else "(none)"
    terms_hint=", ".join(user_terms[:10]) if user_terms else "(none)"
    guardrails = """
RULES (MANDATORY):
- STAY ON TOPIC exactly as provided.
- No country/geography pivot unless topic is about geography.
- No meta sentences, no headers. Return ONLY JSON, no prose/markdown."""
    prompt=f"""{template}

Channel: {channel_name}
Language: {LANG}
TOPIC (hard lock): {topic_lock}
Seed search terms (use and expand): {terms_hint}
Avoid overlap for 180 days:
{avoid}
{guardrails}
"""
    data=_gemini_call(prompt, GEMINI_MODEL)
    topic=topic_lock
    sentences=[clean_caption_text(s) for s in (data.get("sentences") or [])][:8]
    terms=data.get("search_terms") or []
    if isinstance(terms,str): terms=[terms]
    terms=[t.strip() for t in terms if isinstance(t,str) and t.strip()]
    if user_terms:
        pref=[t for t in user_terms if t not in terms]; terms=pref+terms
    title=(data.get("title") or "").strip(); desc=(data.get("description") or "").strip()
    tags=[t.strip() for t in (data.get("tags") or []) if isinstance(t,str) and t.strip()]
    return topic, sentences, terms, title, desc, tags

# =============================================================================
# Per-scene queries (kept)
# =============================================================================
_STOP=set("""a an the and or but if while of to in on at from by with for about into over after before between during under above across around through
this that these those is are was were be been being have has had do does did can could should would may might will shall
you your we our they their he she it its as than then so such very more most many much just also only even still yet
""".split())
_GENERIC_BAD={"great","good","bad","big","small","old","new","many","more","most","thing","things","stuff"}
def _lower_tokens(s: str) -> List[str]:
    s=re.sub(r"[^A-Za-z0-9 ]+"," ", s.lower()); return [w for w in s.split() if w and len(w)>2 and w not in _STOP and w not in _GENERIC_BAD]
def _proper_phrases(texts: List[str]) -> List[str]:
    phrases=[]
    for t in texts:
        for m in re.finditer(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", t or ""):
            phrase=re.sub(r"^(The|A|An)\s+","", m.group(0)); ws=[w.lower() for w in phrase.split()]
            for i in range(len(ws)-1): phrases.append(f"{ws[i]} {ws[i+1]}")
    seen=set(); out=[]
    for p in phrases:
        if p not in seen: seen.add(p); out.append(p)
    return out
def _domain_synonyms(all_text: str) -> List[str]:
    t=(all_text or "").lower(); s=set()
    if any(k in t for k in ["bridge","tunnel","arch","span"]): s.update(["suspension bridge","cable stayed","stone arch","viaduct","aerial city bridge"])
    if any(k in t for k in ["ocean","coast","tide","wave","storm"]): s.update(["ocean waves","coastal storm","rocky coast","lighthouse coast"])
    if any(k in t for k in ["timelapse","growth","melt","cloud"]): s.update(["city timelapse","plant growth","melting ice","cloud timelapse"])
    if any(k in t for k in ["mechanism","gears","pulley","cam"]): s.update(["macro gears","belt pulley","cam follower","robotic arm macro"])
    return list(s)
def build_per_scene_queries(sentences: List[str], fallback_terms: List[str], topic: Optional[str]=None) -> List[str]:
    topic=(topic or "").strip()
    texts_cap=[topic]+sentences; texts_all=" ".join([topic]+sentences)
    phrase_pool=_proper_phrases(texts_cap)+_domain_synonyms(texts_all)
    def _tok4(s: str) -> List[str]:
        s=re.sub(r"[^A-Za-z0-9 ]+"," ", (s or "").lower()); return [w for w in s.split() if len(w)>=4 and w not in _STOP and w not in _GENERIC_BAD]
    fb=[]
    for t in (fallback_terms or []):
        t=re.sub(r"[^A-Za-z0-9 ]+"," ", str(t)).strip().lower()
        if not t: continue
        ws=[w for w in t.split() if w not in _STOP and w not in _GENERIC_BAD]
        if ws: fb.append(" ".join(ws[:2]))
    topic_keys=_tok4(topic)[:2]; topic_key_join=" ".join(topic_keys) if topic_keys else ""
    queries=[]; fb_idx=0
    for s in sentences:
        s_low=" "+(s or "").lower()+" "; picked=None
        for ph in phrase_pool:
            if f" {ph} " in s_low: picked=ph; break
        if not picked:
            toks=_tok4(s)
            if len(toks)>=2: picked=f"{toks[0]} {toks[1]}"
            elif len(toks)==1: picked=toks[0]
        if (not picked or len(picked)<4) and fb:
            picked=fb[fb_idx % len(fb)]; fb_idx+=1
        if (not picked or len(picked)<4) and topic_key_join: picked=topic_key_join
        if not picked or picked in ("great","nice","good","bad","things","stuff"): picked="macro detail"
        if len(picked.split())>2: w=picked.split(); picked=f"{w[-2]} {w[-1]}"
        queries.append(picked)
    return queries

# =============================================================================
# Pexels (multi-pick + variety)
# =============================================================================
_USED_PEXELS_IDS_RUNTIME=set()
def _pexels_headers():
    if not PEXELS_API_KEY: raise RuntimeError("PEXELS_API_KEY missing")
    return {"Authorization": PEXELS_API_KEY}
def _pexels_search(query: str, locale: str) -> List[Tuple[int,str,int,int,float]]:
    url="https://api.pexels.com/videos/search"
    r=requests.get(url, headers=_pexels_headers(), params={"query":query,"per_page":max(10,min(80,PEXELS_PER_PAGE)),"orientation":"portrait","size":"large","locale":locale}, timeout=30)
    if r.status_code!=200: return []
    data=r.json() or {}; out=[]
    for v in data.get("videos", []):
        vid=int(v.get("id",0)); dur=float(v.get("duration",0.0)); files=v.get("video_files", []) or []
        if not files: continue
        pf=[]
        for x in files:
            w=int(x.get("width",0)); h=int(x.get("height",0))
            if h>=1080 and (h>=w or PEXELS_ALLOW_LANDSCAPE): pf.append((w,h,x.get("link")))
        if not pf: continue
        pf.sort(key=lambda t:(abs(t[1]-1440), t[0]*t[1])); w,h,link=pf[0]
        out.append((vid,link,w,h,dur))
    return out
def pexels_pick_many(query: str) -> List[Tuple[int,str]]:
    locale="tr-TR" if LANG.startswith("tr") else "en-US"
    items=_pexels_search(query, locale)
    if not items and locale=="tr-TR": items=_pexels_search(query, "en-US")
    if not items: return []
    block=_blocklist_get_pexels(); cand=[]; qtokens=set(re.findall(r"[a-z0-9]+", query.lower()))
    for vid,link,w,h,dur in items:
        if vid in block or vid in _USED_PEXELS_IDS_RUNTIME: continue
        dur_bonus=1.0 if 2.0<=dur<=12.0 else 0.0
        tokens=set(re.findall(r"[a-z0-9]+", (link or "").lower()))
        overlap=len(tokens & qtokens); score=overlap*2.0 + dur_bonus + (1.0 if 1080 <= h else 0.0)
        cand.append((score,vid,link))
    cand.sort(key=lambda x:x[0], reverse=True)
    out=[]
    for _,vid,link in cand:
        if vid not in _USED_PEXELS_IDS_RUNTIME: out.append((vid,link))
    return out[:5]

# =============================================================================
# YouTube
# =============================================================================
def yt_service():
    cid=os.getenv("YT_CLIENT_ID"); csec=os.getenv("YT_CLIENT_SECRET"); rtok=os.getenv("YT_REFRESH_TOKEN")
    if not (cid and csec and rtok): raise RuntimeError("Missing YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN")
    creds=Credentials(token=None, refresh_token=rtok, token_uri="https://oauth2.googleapis.com/token",
                      client_id=cid, client_secret=csec, scopes=["https://www.googleapis.com/auth/youtube.upload"])
    creds.refresh(Request()); return build("youtube","v3",credentials=creds,cache_discovery=False)
def upload_youtube(video_path: str, meta: dict) -> str:
    y=yt_service()
    body={"snippet":{"title":meta["title"],"description":meta["description"],"tags":meta.get("tags",[]),"categoryId":"27","defaultLanguage":meta.get("defaultLanguage",LANG),"defaultAudioLanguage":meta.get("defaultAudioLanguage",LANG)},
          "status":{"privacyStatus":meta.get("privacy",VISIBILITY),"selfDeclaredMadeForKids":False}}
    media=MediaFileUpload(video_path, chunksize=-1, resumable=True)
    req=y.videos().insert(part="snippet,status", body=body, media_body=media); resp=req.execute()
    return resp.get("id","")

# =============================================================================
# Long SEO Description
# =============================================================================
def build_long_description(channel: str, topic: str, sentences: List[str], tags: List[str]) -> Tuple[str,str,List[str]]:
    hook=(sentences[0].rstrip(" .!?") if sentences else topic or channel); title=(hook[:1].upper()+hook[1:])[:95]
    para=" ".join(sentences)
    explainer=(f"{para} This short explores “{topic}” with clear, visual steps so you can grasp it at a glance. "
               f"Rewatch to catch tiny details, save for later, and share with someone who’ll enjoy it.")
    tagset=[]; base_terms=[w for w in re.findall(r\"[A-Za-z]{3,}\", (topic or \"\"))][:5]
    for t in base_terms: tagset.append(\"#\"+t.lower())
    tagset += [\"#shorts\", \"#learn\", \"#visual\", \"#broll\", \"#education\"]
    if tags:
        for t in tags[:10]:
            tclean=re.sub(r\"[^A-Za-z0-9]+\",\"\", t).lower()
            if tclean and (\"#\"+tclean) not in tagset: tagset.append(\"#\"+tclean)
    body=(f\"{explainer}\\n\\n— Key takeaways —\\n\" + \"\\n\".join([f\"• {s}\" for s in sentences[:8]]) +
          \"\\n\\n— Why it matters —\\nThis topic sticks because it ties a vivid visual to a single idea per scene. "
          "That’s how your brain remembers faster and better.\\n\\n— Watch next —\\n"
          f\"Subscribe for more {topic.lower()} in clear, repeatable visuals.\\n\\n\" + \" \".join(tagset))
    if len(body)>4900: body=body[:4900]
    yt_tags=[]; 
    for h in tagset:
        k=h[1:]; 
        if k and k not in yt_tags: yt_tags.append(k)
        if len(yt_tags)>=15: break
    return title, body, yt_tags

# =============================================================================
# Audio concat / smoothing / music / mux helpers
# =============================================================================
def concat_videos_filter(files: List[str], outp: str):
    if not files: raise RuntimeError("concat_videos_filter: empty")
    inputs=[]; filters=[]
    for i,p in enumerate(files):
        inputs += ["-i", p]; filters.append(f"[{i}:v]fps={TARGET_FPS},settb=AVTB,setpts=N/{TARGET_FPS}/TB[v{i}]")
    filtergraph = ";".join(filters) + ";" + "".join(f"[v{i}]" for i in range(len(files))) + f"concat=n={len(files)}:v=1:a=0[v]"
    run(["ffmpeg","-y","-hide_banner","-loglevel","error",*inputs,"-filter_complex",filtergraph,"-map","[v]","-r",str(TARGET_FPS),"-vsync","cfr","-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),"-pix_fmt","yuv420p","-movflags","+faststart",outp])

def pad_video_to_duration(video_in: str, target_sec: float, outp: str):
    vdur=ffprobe_dur(video_in)
    if vdur >= target_sec - 0.02:
        pathlib.Path(outp).write_bytes(pathlib.Path(video_in).read_bytes()); return
    extra=max(0.0, target_sec - vdur)
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",video_in,"-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={extra:.3f},fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB[v]","-map","[v]","-r",str(TARGET_FPS),"-vsync","cfr","-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),"-pix_fmt","yuv420p","-movflags","+faststart",outp])

def concat_audios_raw(files: List[str], outp: str):
    if not files: raise RuntimeError("concat_audios_raw: empty")
    lst=str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst,"w",encoding="utf-8") as f:
        for p in files: f.write(f"file '{p}'\n")
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i",lst,"-c","copy",outp]); pathlib.Path(lst).unlink(missing_ok=True)

def concat_audios_xfade(files: List[str], outp: str, ms: float = 0.08):
    if not files: raise RuntimeError("concat_audios_xfade: empty")
    tmp_current=files[0]
    for i in range(1,len(files)):
        nxt=files[i]; out_i=str(pathlib.Path(outp).with_suffix(f".xf{i:02d}.wav"))
        run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",tmp_current,"-i",nxt,"-filter_complex",f"[0:a][1:a]acrossfade=d={max(0.02,min(0.20,ms))}:c1=tri:c2=tri[a]","-map","[a]","-ar","48000","-ac","1","-c:a","pcm_s16le",out_i])
        if tmp_current != files[0]: pathlib.Path(tmp_current).unlink(missing_ok=True)
        tmp_current=out_i
    pathlib.Path(outp).write_bytes(pathlib.Path(tmp_current).read_bytes())
    if tmp_current != files[0]: pathlib.Path(tmp_current).unlink(missing_ok=True)

def lock_audio_duration(audio_in: str, target_frames: int, outp: str):
    dur=target_frames/float(TARGET_FPS)
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",audio_in,"-af",f"atrim=end={dur:.6f},asetpts=N/SR/TB","-ar","48000","-ac","1","-c:a","pcm_s16le",outp])

def mux(video: str, audio: str, outp: str):
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",video,"-i",audio,"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","256k","-movflags","+faststart","-muxpreload","0","-muxdelay","0","-avoid_negative_ts","make_zero",outp])

# =============================================================================
# Main
# =============================================================================
def main():
    print(f"==> {CHANNEL_NAME} | MODE={MODE} | topic-first build")
    random.seed(ROTATION_SEED or int(time.time()))

    topic_lock = TOPIC or "Interesting Visual Explainers"
    user_terms = SEARCH_TERMS_ENV

    # 1) Content build (Gemini) + quality
    attempts=0; best=None; best_score=-1.0; banlist=_recent_topics_for_prompt()
    while attempts<3:
        attempts+=1
        if USE_GEMINI and GEMINI_API_KEY:
            try: tpc,sents,search_terms,ttl,desc,tags=build_via_gemini(CHANNEL_NAME, topic_lock, user_terms, banlist)
            except Exception as e:
                print(f"Gemini error: {str(e)[:200]}"); tpc=topic_lock; sents=[]; search_terms=user_terms or []; ttl=""; desc=""; tags=[]
        else:
            tpc=topic_lock; sents=[f"{tpc} comes alive in small vivid scenes.","Each beat shows one concrete detail to remember.","The story moves forward without fluff or filler.","You can picture it clearly as you listen.","A tiny contrast locks the idea in memory.","No meta talk—just what matters on screen.","Replay to catch micro-details and patterns."]; search_terms=user_terms or ["macro detail","timelapse","clean b-roll"]; ttl=""; desc=""; tags=[]
        score=_content_score(sents); print(f"📝 Content: {tpc} | {len(sents)} lines | score={score:.2f}")
        if score>best_score: best=(tpc,sents,search_terms,ttl,desc,tags); best_score=score
        if score>=7.2: break
        print("⚠️ Low content score → rebuilding…"); banlist=[tpc]+banlist; time.sleep(0.5)

    tpc, sentences, search_terms, ttl, desc, tags = best
    sig=f"{CHANNEL_NAME}|{tpc}|{sentences[0] if sentences else ''}"; _record_recent(_hash12(sig), MODE, tpc)
    print(f"📊 Sentences: {len(sentences)}")

    # 2) TTS per sentence
    tmp=tempfile.mkdtemp(prefix="enhanced_shorts_"); font=font_path()
    wavs, metas = [], []
    print("🎤 TTS…")
    for i,s in enumerate(sentences):
        base=normalize_sentence(s); w=str(pathlib.Path(tmp)/f"sent_{i:02d}.wav")
        d=tts_to_wav(base,w); wavs.append(w); metas.append((base,d)); print(f"   {i+1}/{len(sentences)}: {d:.2f}s")

    # 2.5) Optional smooth audio concat (voice only)
    acat_voice=str(pathlib.Path(tmp)/"audio_voice.wav")
    if USE_AUDIO_XFADE and len(wavs)>=2: concat_audios_xfade(wavs, acat_voice, ms=AUDIO_XFADE_MS)
    else: concat_audios_raw(wavs, acat_voice)

    # 3) Pexels search pool
    per_scene_queries=build_per_scene_queries([m[0] for m in metas], (search_terms or user_terms or []), topic=tpc)
    print("🔎 Per-scene queries:"); [print(f"   • {q}") for q in per_scene_queries]
    pool=[]; seen_ids=set()
    for q in per_scene_queries:
        picks=pexels_pick_many(q)
        for vid,link in picks:
            if vid not in seen_ids: seen_ids.add(vid); pool.append((vid,link))
    if len(pool)<len(metas):
        extras=["macro detail","city timelapse","nature macro","clean interior","close up hands","ocean waves","night skyline","forest path","nature","globe","space","ocean"]
        for q in (user_terms or []) + extras:
            for vid,link in pexels_pick_many(q):
                if vid not in seen_ids: seen_ids.add(vid); pool.append((vid,link))
                if len(pool)>=len(metas)*2: break
            if len(pool)>=len(metas)*2: break
    if not pool: raise RuntimeError("Pexels: no suitable clips found.")

    # Download pool
    downloads={}
    print("⬇️ Download pool…")
    for idx,(vid,link) in enumerate(pool):
        try:
            f=str(pathlib.Path(tmp)/f"pool_{idx:02d}_{vid}.mp4")
            with requests.get(link, stream=True, timeout=120) as rr:
                rr.raise_for_status()
                with open(f,"wb") as w:
                    for ch in rr.iter_content(8192): w.write(ch)
            if pathlib.Path(f).stat().st_size>300_000: downloads[vid]=f
        except Exception as e: print(f"⚠️ download fail ({vid}): {e}")

    # Assign per sentence (limit reuse)
    usage={vid:0 for vid in downloads.keys()}; chosen_files=[]
    for i in range(len(metas)):
        picked_path=None
        for vid,p in downloads.items():
            if usage[vid] < PEXELS_MAX_USES_PER_CLIP:
                picked_path=p; usage[vid]+=1; break
        if not picked_path:
            if not usage: raise RuntimeError("Pexels pool empty after filtering.")
            vid=min(usage.keys(), key=lambda k: usage[k]); usage[vid]+=1; picked_path=downloads[vid]
        chosen_files.append(picked_path)

    # 4) Segments + WORD-LEVEL ASS subtitles
    print("🎬 Segments…")
    segs=[]
    for i,((base_text,d),src) in enumerate(zip(metas, chosen_files)):
        base=str(pathlib.Path(tmp)/f"seg_{i:02d}.mp4"); make_segment(src, d, base)
        colored=str(pathlib.Path(tmp)/f"segsub_{i:02d}.mp4")
        if SUBTITLES_MODE=="ass":
            seg_ass=str(pathlib.Path(tmp)/f"seg_{i:02d}.ass")
            words=transcribe_words(wavs[i], lang_hint=("en" if LANG.startswith("en") else LANG))
            if not words: words=[(0.0, d, clean_caption_text(base_text).upper())]  # failsafe
            make_ass_word_window(words, seg_ass, chunk_words=SUB_CHUNK_WORDS)
            overlay_subtitles(base, seg_ass, colored)
        else:
            # legacy drawtext (not recommended)
            draw_capcut_text(base, base_text, CAPTION_COLORS[i % len(CAPTION_COLORS)], font, colored, is_hook=(i==0))
        segs.append(colored)

    # 4.5) Optional xfade for the first join
    if VFX_XFADE_FIRST and len(segs)>=2:
        cf0=str(pathlib.Path(tmp)/"seg_cf_00.mp4")
        dur_a=ffprobe_dur(segs[0]); fade=max(0.12, min(0.40, VFX_XFADE_MS))
        run(["ffmpeg","-y","-i",segs[0],"-i",segs[1],"-filter_complex",f"[0:v][1:v]xfade=transition=fade:duration={fade}:offset={max(0.0,dur_a-fade)}[v]","-map","[v]","-r",str(TARGET_FPS),"-vsync","cfr","-an","-c:v","libx264","-crf",str(CRF_VISUAL),cf0])
        segs = [cf0] + segs[2:]

    # 5) Assemble video
    print("🎞️ Assemble…")
    vcat=str(pathlib.Path(tmp)/"video_concat.mp4"); concat_videos_filter(segs, vcat)

    # 5.5) Music mix
    acat_voice_path=str(pathlib.Path(tmp)/"audio_voice.wav")
    acat=acat_voice
    if USE_MUSIC:
        music=pick_music(MUSIC_MOOD)
        if music:
            print(f"🎵 Music: {music}")
            mixed=str(pathlib.Path(tmp)/"audio_mixed.wav"); mix_voice_and_music(acat, music, mixed, duck=DUCKING, gain_db=MUSIC_GAIN_DB); acat=mixed
        else:
            print("🎵 No music found, skipping music mix.")

    # 6) Hard A/V lock
    adur=ffprobe_dur(acat); vdur=ffprobe_dur(vcat)
    if vdur + 0.02 < adur:
        vcat_padded=str(pathlib.Path(tmp)/"video_padded.mp4"); pad_video_to_duration(vcat, adur, vcat_padded); vcat=vcat_padded
    a_frames=max(2,int(round(adur*TARGET_FPS)))
    vcat_exact=str(pathlib.Path(tmp)/"video_exact.mp4"); enforce_video_exact_frames(vcat, a_frames, vcat_exact); vcat=vcat_exact
    acat_exact=str(pathlib.Path(tmp)/"audio_exact.wav"); lock_audio_duration(acat, a_frames, acat_exact); acat=acat_exact
    vdur2=ffprobe_dur(vcat); adur2=ffprobe_dur(acat)
    print(f"🔒 Locked A/V: video={vdur2:.3f}s | audio={adur2:.3f}s | fps={TARGET_FPS}")

    # 7) Mux
    ts=datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic=re.sub(r'[^A-Za-z0-9]+','_', tpc)[:60] or "Short"
    outp=f"{OUT_DIR}/{CHANNEL_NAME}_{safe_topic}_{ts}.mp4"
    print("🔄 Mux…"); mux(vcat, acat, outp); final=ffprobe_dur(outp); print(f"✅ Saved: {outp} ({final:.2f}s)")

    # 8) Metadata (long SEO)
    title,description,yt_tags=build_long_description(CHANNEL_NAME, tpc, [m[0] for m in metas], tags)
    meta={"title":title,"description":description,"tags":yt_tags,"privacy":VISIBILITY,"defaultLanguage":LANG,"defaultAudioLanguage":LANG}

    # 9) Upload (optional)
    try:
        if os.getenv("UPLOAD_TO_YT","1") == "1":
            print("📤 Uploading to YouTube…"); vid_id=upload_youtube(outp, meta); print(f"🎉 YouTube Video ID: {vid_id}\n🔗 https://youtube.com/watch?v={vid_id}")
        else:
            print("⏭️ Upload disabled (UPLOAD_TO_YT != 1)")
    except Exception as e:
        print(f"❌ Upload skipped: {e}")

    # 10) Cleanup
    try: shutil.rmtree(tmp); print("🧹 Cleaned temp files")
    except: pass

if __name__ == "__main__":
    main()
