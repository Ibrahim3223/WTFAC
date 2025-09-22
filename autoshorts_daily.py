# autoshorts_daily.py — Relevance-first Pexels + per-scene queries + hard A/V lock
# -*- coding: utf-8 -*-
import os, sys, re, json, time, random, datetime, tempfile, pathlib, subprocess, hashlib, math, shutil
from typing import List, Optional, Tuple, Dict

# -------------------- ENV / constants --------------------
VOICE_STYLE    = os.getenv("TTS_STYLE", "narration-professional")
TARGET_MIN_SEC = float(os.getenv("TARGET_MIN_SEC", "22"))
TARGET_MAX_SEC = float(os.getenv("TARGET_MAX_SEC", "42"))

CHANNEL_NAME   = os.getenv("CHANNEL_NAME", "DefaultChannel")
MODE           = os.getenv("MODE", "country_facts").strip().lower()
LANG           = os.getenv("LANG", "en")
VISIBILITY     = os.getenv("VISIBILITY", "public")
ROTATION_SEED  = int(os.getenv("ROTATION_SEED", "0"))
OUT_DIR        = "out"; pathlib.Path(OUT_DIR).mkdir(exist_ok=True)

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_GEMINI     = os.getenv("USE_GEMINI", "0") == "1"
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

TARGET_FPS       = 25
CRF_VISUAL       = 22
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

def wrap_mobile_lines(text: str, max_line_length: int = CAPTION_MAX_LINE, max_lines: int = 5) -> str:
    text = (text or "").strip()
    if not text: return text
    words = text.split()
    def distribute_into(k: int) -> List[str]:
        per = math.ceil(len(words) / k)
        chunks = [" ".join(words[i*per:(i+1)*per]) for i in range(k)]
        return [c for c in chunks if c]
    chosen = None
    for k in range(2, max_lines + 1):
        cand = distribute_into(k)
        if cand and all(len(c) <= max_line_length for c in cand):
            chosen = cand; break
    if not chosen:
        chosen = distribute_into(max_lines)
        fixed=[]
        for c in chosen:
            if len(c) <= max_line_length + 6:
                fixed.append(c)
            else:
                ws = c.split(); buf=[]; L=0
                for w in ws:
                    add = (1 if buf else 0) + len(w)
                    if L + add > max_line_length and buf:
                        fixed.append(" ".join(buf)); buf=[w]; L=len(w)
                    else:
                        buf.append(w); L += add
                if buf: fixed.append(" ".join(buf))
        chosen = fixed[:max_lines]
    if len(chosen) == 1 and len(words) > 1:
        mid = len(words)//2
        chosen = [" ".join(words[:mid]), " ".join(words[mid:])]
    return "\n".join([c.strip() for c in chosen if c.strip()])

# -------------------- TTS --------------------
def _rate_to_atempo(rate_str: str, default: float = 1.08) -> float:
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
    from aiohttp.client_exceptions import WSServerHandshakeError

    text = (text or "").strip()
    if not text:
        run(["ffmpeg","-y","-f","lavfi","-t","1.0","-i","anullsrc=r=48000:cl=mono", wav_out])
        return 1.0

    mp3 = wav_out.replace(".wav", ".mp3")
    rate_env = os.getenv("TTS_RATE", "+15%")
    atempo = _rate_to_atempo(rate_env, default=1.15)

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
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                loop.run_until_complete(_edge_save_simple())

            run([
                "ffmpeg","-y","-hide_banner","-loglevel","error",
                "-i", mp3,
                "-ar","48000","-ac","1","-acodec","pcm_s16le",
                "-af", f"dynaudnorm=g=7:f=250,atempo={atempo}",
                wav_out
            ])
            pathlib.Path(mp3).unlink(missing_ok=True)
            return ffprobe_dur(wav_out) or 0.0

        except WSServerHandshakeError as e:
            if getattr(e, "status", None) == 401 or "401" in str(e):
                print("⚠️ edge-tts 401 → hızlı fallback TTS")
                break
            print(f"⚠️ edge-tts deneme {attempt+1}/2 başarısız: {e}")
            time.sleep(0.8)
        except Exception as e:
            print(f"⚠️ edge-tts deneme {attempt+1}/2 başarısız: {e}")
            time.sleep(0.8)

    # Google fallback
    try:
        q = requests.utils.quote(text.replace('"','').replace("'",""))
        lang_code = (LANG or "en")
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={q}&tl={lang_code}&client=tw-ob&ttsspeed=1.0"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=30); r.raise_for_status()
        open(mp3, "wb").write(r.content)

        run([
            "ffmpeg","-y","-hide_banner","-loglevel","error",
            "-i", mp3,
            "-ar","48000","-ac","1","-acodec","pcm_s16le",
            "-af", f"dynaudnorm=g=6:f=300,atempo={atempo}",
            wav_out
        ])
        pathlib.Path(mp3).unlink(missing_ok=True)
        return ffprobe_dur(wav_out) or 0.0
    except Exception as e2:
        print(f"❌ Tüm TTS yolları başarısız, sessizlik üretilecek: {e2}")
        run(["ffmpeg","-y","-f","lavfi","-t","4.0","-i","anullsrc=r=48000:cl=mono", wav_out])
        return 4.0

# -------------------- Video helpers --------------------
def quantize_to_frames(seconds: float, fps: int = TARGET_FPS) -> Tuple[int, float]:
    """Süreyi en yakın kare sayısına yuvarlar; (frames, frames/fps) döner."""
    frames = max(2, int(round(seconds * fps)))
    return frames, frames / float(fps)

def make_segment(src: str, dur_s: float, outp: str):
    """Kaynak klipten tam CFR süreli segment üret (trim'i kareye göre yap)."""
    frames, qdur = quantize_to_frames(dur_s, TARGET_FPS)
    fade = max(0.05, min(0.12, qdur/8.0))
    fade_out_st = max(0.0, qdur - fade)

    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "eq=brightness=0.02:contrast=1.08:saturation=1.1,"
        f"fps={TARGET_FPS},"
        f"setpts=N/{TARGET_FPS}/TB,"
        f"trim=start_frame=0:end_frame={frames},"
        f"fade=t=in:st=0:d={fade:.2f},"
        f"fade=t=out:st={fade_out_st:.2f}:d={fade:.2f}"
    )

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

def draw_capcut_text(seg: str, text: str, color: str, font: str, outp: str, is_hook: bool=False):
    """Altyazıyı segment üzerinde (CFR) çiz — PTS sabit olduğu için kayıp olmaz."""
    wrapped = wrap_mobile_lines(clean_caption_text(text), CAPTION_MAX_LINE, max_lines=5)
    tf = str(pathlib.Path(seg).with_suffix(".caption.txt"))
    pathlib.Path(tf).write_text(wrapped, encoding="utf-8")

    lines = wrapped.split("\n")
    n_lines = max(1, len(lines))
    maxchars = max((len(l) for l in lines), default=1)

    base = 58 if is_hook else 48
    ratio = CAPTION_MAX_LINE / max(1, maxchars)
    fs = int(base * min(1.0, max(0.55, ratio)))
    if n_lines >= 4: fs = int(fs * 0.92)
    if n_lines >= 5: fs = int(fs * 0.88)
    fs = max(28, fs)

    y_pos = "(h*0.58 - text_h/2)" if n_lines >= 4 else "h-h/3-text_h/2"

    col = _ff_color(color)
    font_arg = f":fontfile={_ff_sanitize_font(font)}" if font else ""
    common = f"textfile='{tf}':fontsize={fs}:x=(w-text_w)/2:y={y_pos}:line_spacing=10"

    shadow = f"drawtext={common}{font_arg}:fontcolor=black@0.85:borderw=0"
    box    = f"drawtext={common}{font_arg}:fontcolor=white@0.0:box=1:boxborderw={(22 if is_hook else 18)}:boxcolor=black@0.65"
    main   = f"drawtext={common}{font_arg}:fontcolor={col}:borderw={(5 if is_hook else 4)}:bordercolor=black@0.9"

    vf_advanced = f"{shadow},{box},{main}"
    vf_simple   = f"drawtext={common}{font_arg}:fontcolor=white:borderw=3:bordercolor=black@0.85"

    try:
        run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",seg,"-vf",vf_advanced,
             "-c:v","libx264","-preset","medium","-crf",str(max(16,CRF_VISUAL-3)),
             "-movflags","+faststart", outp])
    except Exception as e:
        print(f"⚠️ drawtext advanced failed, fallback simple: {e}")
        run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",seg,"-vf",vf_simple,
             "-c:v","libx264","-preset","medium","-crf",str(max(16,CRF_VISUAL-2)),
             "-movflags","+faststart", outp])
    finally:
        pathlib.Path(tf).unlink(missing_ok=True)

def pad_video_to_duration(video_in: str, target_sec: float, outp: str):
    """Video kısa ise son kareyi klonla."""
    vdur = ffprobe_dur(video_in)
    if vdur >= target_sec - 0.02:
        pathlib.Path(outp).write_bytes(pathlib.Path(video_in).read_bytes())
        return
    extra = max(0.0, target_sec - vdur)
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", video_in,
        "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={extra:.3f},fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB[v]",
        "-map","[v]",
        "-r", str(TARGET_FPS), "-vsync","cfr",
        "-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),
        "-pix_fmt","yuv420p","-movflags","+faststart",
        outp
    ])

def concat_videos_filter(files: List[str], outp: str):
    """Concat (filter_complex) ile CFR zinciri — PTS drift birikmez."""
    if not files: raise RuntimeError("concat_videos_filter: empty")
    inputs = []; filters = []
    for i, p in enumerate(files):
        inputs += ["-i", p]
        filters.append(f"[{i}:v]fps={TARGET_FPS},settb=AVTB,setpts=N/{TARGET_FPS}/TB[v{i}]")
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

def enforce_video_exact_frames(video_in: str, target_frames: int, outp: str):
    """Final videoyu tam 'target_frames' kareye kes (CFR)."""
    target_frames = max(2, int(target_frames))
    vf = f"fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB,trim=start_frame=0:end_frame={target_frames}"
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", video_in,
        "-vf", vf,
        "-r", str(TARGET_FPS), "-vsync","cfr",
        "-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),
        "-pix_fmt","yuv420p","-movflags","+faststart",
        outp
    ])

# -------------------- Audio concat (lossless, no resample) --------------------
def concat_audios(files: List[str], outp: str):
    """
    Tüm WAV'lar 48kHz/mono/PCM_s16le; concat demuxer + -c copy ile sample-exact birleşir.
    """
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
    """
    Sesi tam (target_frames / FPS) saniyeye kilitle.
    """
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
    """Süreler eşit — mux'ta kopya; encoder delay'i bastır."""
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

# -------------------- Per-scene queries --------------------
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

# -------------------- Pexels (tek odak + tekrar önleme) --------------------
_USED_PEXELS_IDS_RUNTIME = set()

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
            if vid in block or vid in _USED_PEXELS_IDS_RUNTIME:
                continue
            files = v.get("video_files", []) or []
            if not files: continue
            pf = [x for x in files if int(x.get("height",0)) >= int(x.get("width",0)) and int(x.get("height",0)) >= 1080]
            if not pf: continue
            pf.sort(key=lambda x: (abs(int(x.get("height",0))-1440), int(x.get("height",0))*int(x.get("width",0))))
            best = pf[0]
            w = int(best.get("width",0)); h = int(best.get("height",0))
            dur = float(v.get("duration",0))
            dur_bonus = 1.0 if 2.0 <= dur <= 12.0 else 0.0
            tokens = set(re.findall(r"[a-z0-9]+", (v.get("url") or "").lower()))
            qtokens= set(re.findall(r"[a-z0-9]+", query.lower()))
            overlap = len(tokens & qtokens)
            score = overlap*2.0 + dur_bonus + (1.0 if 1080 <= h <= 1920 else 0.0)
            cand.append((score, vid, best.get("link")))
        if not cand: return None, None
        cand.sort(key=lambda x: x[0], reverse=True)
        for _, vid, link in cand:
            if vid not in _USED_PEXELS_IDS_RUNTIME:
                _USED_PEXELS_IDS_RUNTIME.add(vid)
                _blocklist_add_pexels([vid], days=30)
                print(f"   → Pexels pick [{query}] -> id={vid} | {link}")
                return vid, link
        return None, None
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

    # 1) İçerik
    if USE_GEMINI and GEMINI_API_KEY:
        banlist = _recent_topics_for_prompt()
        chosen, last = None, None
        for _ in range(8):
            try:
                ctry, tpc, sents, search_terms, ttl, desc, tags = build_via_gemini(MODE, CHANNEL_NAME, banlist)
                last = (ctry, tpc, sents, search_terms, ttl, desc, tags)
                sig = f"{MODE}|{tpc}|{sents[0] if sents else ''}"
                h = _hash12(sig)
                if not _is_recent(h, window_days=180):
                    _record_recent(h, MODE, tpc)
                    chosen = last; break
                else:
                    banlist.insert(0, tpc)
                    time.sleep(1.0)
            except Exception as e:
                print(f"Gemini error: {str(e)[:160]}")
                time.sleep(0.8)
        if chosen is None:
            if last:
                ctry, tpc, sents, search_terms, ttl, desc, tags = last
            else:
                ctry, tpc = "World", "Amazing Facts"
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
                search_terms = ["cappadocia underground", "ancient tunnels", "turkey caves 4k"]
                ttl = desc = ""; tags = []
    else:
        ctry, tpc = "World", "Amazing Facts"
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
        search_terms = ["cappadocia underground", "ancient tunnels", "turkey caves 4k"]
        ttl = desc = ""; tags = []

    sentences = sents
    print(f"📝 Content: {ctry} | {tpc}")
    print(f"📊 Sentences: {len(sentences)}")

    # 2) TTS
    tmp = tempfile.mkdtemp(prefix="enhanced_shorts_")
    font = font_path()
    wavs, metas = [], []   # metas: [(clean_text, duration)]
    processed_sentences = []

    print("🎤 TTS…")
    for i, s in enumerate(sentences):
        base = normalize_sentence(s)
        processed_sentences.append(base)
        w = str(pathlib.Path(tmp) / f"sent_{i:02d}.wav")
        d = tts_to_wav(base, w)  # 48kHz/mono/pcm_s16le + dynaudnorm + atempo
        wavs.append(w)
        metas.append((base, d))
        print(f"   {i+1}/{len(sentences)}: {d:.2f}s")
    sentences = processed_sentences

    # 3) Pexels — sahne başına net sorgular
    per_scene_queries = build_per_scene_queries(sentences, search_terms or [], MODE, topic=tpc)
    print("🔎 Per-scene queries:")
    for q in per_scene_queries:
        print(f"   • {q}")

    picked = []
    for q in per_scene_queries:
        vid, link = pexels_pick_one(q)
        if vid and link: picked.append((vid, link))
    if not picked:
        raise RuntimeError("Pexels: hiçbir sonuç bulunamadı (per-scene).")

    clips = []
    for idx, (vid, link) in enumerate(picked):
        try:
            f = str(pathlib.Path(tmp) / f"clip_{idx:02d}_{vid}.mp4")
            with requests.get(link, stream=True, timeout=120) as rr:
                rr.raise_for_status()
                with open(f, "wb") as w:
                    for ch in rr.iter_content(8192):
                        w.write(ch)
            if pathlib.Path(f).stat().st_size > 400_000:
                clips.append(f)
        except Exception as e:
            print(f"⚠️ download fail ({vid}): {e}")

    if len(clips) < len(sentences):
        print("⚠️ Yeterli klip yok; mevcut klipleri döndürerek kullanılacak.")

    # 4) Segment + altyazı (her segment CFR + tam süre)
    print("🎬 Segments…")
    segs = []
    for i, (base_text, d) in enumerate(metas):
        base   = str(pathlib.Path(tmp) / f"seg_{i:02d}.mp4")
        make_segment(clips[i % len(clips)], d, base)
        colored = str(pathlib.Path(tmp) / f"segsub_{i:02d}.mp4")
        draw_capcut_text(
            base,
            base_text,
            CAPTION_COLORS[i % len(CAPTION_COLORS)],
            font,
            colored,
            is_hook=(i == 0)
        )
        segs.append(colored)

    # 5) Birleştir (filter_complex concat) + Ses birleştir (lossless)
    print("🎞️ Assemble…")
    vcat = str(pathlib.Path(tmp) / "video_concat.mp4"); concat_videos_filter(segs, vcat)
    acat = str(pathlib.Path(tmp) / "audio_concat.wav"); concat_audios(wavs, acat)

    # 6) Süre & kare kilitleme (kritik)
    adur = ffprobe_dur(acat)
    vdur = ffprobe_dur(vcat)

    # Gerekirse video'yu audio süresine pad et (çok ufak farklar için)
    if vdur + 0.02 < adur:
        vcat_padded = str(pathlib.Path(tmp) / "video_padded.mp4")
        pad_video_to_duration(vcat, adur, vcat_padded)
        vcat = vcat_padded
        vdur = ffprobe_dur(vcat)

    # Audio karesine göre kesin kilit
    a_frames = max(2, int(round(adur * TARGET_FPS)))

    # Video: tam kareye kes
    vcat_exact = str(pathlib.Path(tmp) / "video_exact.mp4")
    enforce_video_exact_frames(vcat, a_frames, vcat_exact)
    vcat = vcat_exact

    # Audio: tam süreye kes (frame->saniye)
    acat_exact = str(pathlib.Path(tmp) / "audio_exact.wav")
    lock_audio_duration(acat, a_frames, acat_exact)
    acat = acat_exact

    # Debug ölçüm
    vdur2 = ffprobe_dur(vcat); adur2 = ffprobe_dur(acat)
    print(f"🔒 Locked A/V: video={vdur2:.3f}s | audio={adur2:.3f}s | fps={TARGET_FPS}")

    # 7) Mux
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^A-Za-z0-9]+', '_', tpc)[:60] or "Short"
    outp = f"{OUT_DIR}/{ctry}_{safe_topic}_{ts}.mp4"

    print("🔄 Mux…")
    mux(vcat, acat, outp)
    final = ffprobe_dur(outp)
    print(f"✅ Saved: {outp} ({final:.2f}s)")

    # 8) Metadata
    def _ok_str(x): return isinstance(x, str) and len(x.strip()) > 0
    if _ok_str(ttl):
        meta = {
            "title": ttl[:95],
            "description": (desc or "")[:4900],
            "tags": (tags[:15] if isinstance(tags, list) else []),
            "privacy": VISIBILITY,
            "defaultLanguage": LANG,
            "defaultAudioLanguage": LANG
        }
    else:
        hook = (sentences[0].rstrip(" .!?") if sentences else f"{ctry} secrets")
        title = f"{hook} — {ctry} Facts"
        description = "• " + "\n• ".join(sentences[:6]) + f"\n\n#{ctry.lower()} #shorts #facts"
        meta = {
            "title": title[:95],
            "description": description[:4900],
            "tags": ["shorts","facts",ctry.lower(),"education","interesting","broll","documentary","learn","trivia","history"],
            "privacy": VISIBILITY,
            "defaultLanguage": LANG,
            "defaultAudioLanguage": LANG
        }

    # 9) Upload (varsa env)
    try:
        print("📤 Uploading to YouTube…")
        vid_id = upload_youtube(outp, meta)
        print(f"🎉 YouTube Video ID: {vid_id}\n🔗 https://youtube.com/watch?v={vid_id}")
    except Exception as e:
        print(f"❌ Upload skipped: {e}")

    # 10) Temizlik
    try:
        shutil.rmtree(tmp)
        print("🧹 Cleaned temp files")
    except:
        pass

if __name__ == "__main__":
    main()
