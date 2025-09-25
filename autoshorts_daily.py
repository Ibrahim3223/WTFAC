# autoshorts_daily.py — Topic-locked Gemini • Multi-clip Pexels • Hard A/V lock • Long SEO desc
# -*- coding: utf-8 -*-
import os, sys, re, json, time, random, datetime, tempfile, pathlib, subprocess, hashlib, math, shutil
from typing import List, Optional, Tuple, Dict

# -------------------- ENV / constants --------------------
VOICE_STYLE    = os.getenv("TTS_STYLE", "narration-professional")
TARGET_MIN_SEC = float(os.getenv("TARGET_MIN_SEC", "22"))
TARGET_MAX_SEC = float(os.getenv("TARGET_MAX_SEC", "42"))

CHANNEL_NAME   = os.getenv("CHANNEL_NAME", "DefaultChannel")
MODE           = os.getenv("MODE", "freeform").strip().lower()  # sadece LOG için
LANG           = os.getenv("LANG", "en")
VISIBILITY     = os.getenv("VISIBILITY", "public")
ROTATION_SEED  = int(os.getenv("ROTATION_SEED", "0"))
OUT_DIR        = "out"; pathlib.Path(OUT_DIR).mkdir(exist_ok=True)

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_GEMINI     = os.getenv("USE_GEMINI", "1") == "1"
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Geçiş (transition) ayarları
TRANSITION_SEC  = float(os.getenv("TRANSITION_SEC", "0.28"))
TRANSITION_LIST = [t.strip() for t in os.getenv(
    "TRANSITION_LIST",
    "fade,dissolve,slideleft,slideright,slideup,slidedown,"
    "wipeleft,wiperight,wipeup,wipedown,smoothleft,smoothright,"
    "circleopen,circleclose,radial,zoom,pixelize,distance,squeezeh,squeezew"
).split(",") if t.strip()]
AUDIO_CROSSFADE = os.getenv("AUDIO_CROSSFADE", "0") == "1"  # varsayılan kapalı

# ---- Channel intent (Topic & Terms) ----
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

# Pexels esnetmeler
PEXELS_PER_PAGE          = int(os.getenv("PEXELS_PER_PAGE", "30"))
PEXELS_MAX_USES_PER_CLIP = int(os.getenv("PEXELS_MAX_USES_PER_CLIP", "1"))  # bir klibi kaç sahnede kullanabiliriz
PEXELS_ALLOW_LANDSCAPE   = os.getenv("PEXELS_ALLOW_LANDSCAPE", "1") == "1"

STATE_FILE = f"state_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json"
GLOBAL_TOPIC_STATE = "state_global_topics.json"

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
def _load_json(path, default):
    try: return json.load(open(path, "r", encoding="utf-8"))
    except: return default

def _save_json(path, data):
    pathlib.Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def _state_load() -> dict:
    st = _load_json(STATE_FILE, {"recent": [], "used_pexels_ids": []})
    return st

def _state_save(st: dict):
    st["recent"] = st.get("recent", [])[-1200:]
    st["used_pexels_ids"] = st.get("used_pexels_ids", [])[-5000:]
    _save_json(STATE_FILE, st)

def _global_topics_load() -> dict:
    return _load_json(GLOBAL_TOPIC_STATE, {"recent_topics": []})

def _global_topics_save(gst: dict):
    gst["recent_topics"] = gst.get("recent_topics", [])[-4000:]
    _save_json(GLOBAL_TOPIC_STATE, gst)

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
    _state_save(st)  # <-- DÜZELTİLDİ (önceki hatalı _save_json(st) yerine)
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

def wrap_mobile_lines(text: str, max_line_length: int = CAPTION_MAX_LINE, max_lines: int = CAPTION_MAX_LINES) -> str:
    text = (text or "").strip()
    if not text:
        return text
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
        lines = []
        buf, L = [], 0
        for w in words:
            add = (1 if buf else 0) + len(w)
            if L + add > width and buf:
                lines.append(" ".join(buf)); buf = [w]; L = len(w)
            else:
                buf.append(w); L += add
        if buf: lines.append(" ".join(buf))
        if len(lines) > k_cap and k_cap < HARD_CAP:
            return greedy(width, HARD_CAP)
        return lines
    lines = greedy(max_line_length, max_lines)
    return "\n".join([ln.strip() for ln in lines if ln.strip()])

# -------------------- TTS --------------------
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

def tts_to_wav(text: str, wav_out: str) -> float:
    import asyncio
    from aiohttp.client_exceptions import WSServerHandshakeError
    text = (text or "").strip()
    if not text:
        run(["ffmpeg","-y","-f","lavfi","-t","1.0","-i","anullsrc=r=48000:cl=mono", wav_out])
        return 1.0
    mp3 = wav_out.replace(".wav", ".mp3")
    rate_env = os.getenv("TTS_RATE", "+12%")
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
                print("⚠️ edge-tts 401 → hızlı fallback TTS"); break
            print(f"⚠️ edge-tts deneme {attempt+1}/2 başarısız: {e}"); time.sleep(0.8)
        except Exception as e:
            print(f"⚠️ edge-tts deneme {attempt+1}/2 başarısız: {e}"); time.sleep(0.8)
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
        print(f"❌ TTS tüm yollar başarısız, sessizlik üretilecek: {e2}")
        run(["ffmpeg","-y","-f","lavfi","-t","4.0","-i","anullsrc=r=48000:cl=mono", wav_out])
        return 4.0

# -------------------- Video helpers --------------------
def quantize_to_frames(seconds: float, fps: int = TARGET_FPS) -> Tuple[int, float]:
    frames = max(2, int(round(seconds * fps)))
    return frames, frames / float(fps)

# Geçişleri sahne çiftlerine göre güvenli ve frame’e kuantize hesapla
def _compute_pairwise_transitions(durs: List[float]) -> List[float]:
    if not durs or len(durs) < 2:
        return []
    per = []
    for i in range(len(durs)-1):
        raw = min(TRANSITION_SEC, 0.45 * min(durs[i], durs[i+1]))
        frames, qdur = quantize_to_frames(max(0.0, raw), TARGET_FPS)
        per.append(qdur)
    return per

# no_start_fade / no_end_fade ile giriş/çıkış efektlerini kontrol et
def make_segment(src: str, dur_s: float, outp: str, no_start_fade: bool = False, no_end_fade: bool = False):
    frames, qdur = quantize_to_frames(dur_s, TARGET_FPS)
    fade = max(0.05, min(0.12, qdur/8.0))
    fade_out_st = max(0.0, qdur - fade)
    vf_parts = [
        "scale=1080:1920:force_original_aspect_ratio=increase",
        "crop=1080:1920",
        "eq=brightness=0.02:contrast=1.08:saturation=1.1",
        f"fps={TARGET_FPS}",
        f"setpts=N/{TARGET_FPS}/TB",
        f"trim=start_frame=0:end_frame={frames}",
    ]
    if not no_start_fade:
        vf_parts.append(f"fade=t=in:st=0:d={fade:.2f}")
    if not no_end_fade:
        vf_parts.append(f"fade=t=out:st={fade_out_st:.2f}:d={fade:.2f}")
    vf = ",".join(vf_parts)
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

def draw_capcut_text(seg: str, text: str, color: str, font: str, outp: str,
                     is_hook: bool=False, start_delay: float=0.0, end_cut: float=0.0):
    wrapped = wrap_mobile_lines(clean_caption_text(text), CAPTION_MAX_LINE, CAPTION_MAX_LINES)
    tf = str(pathlib.Path(seg).with_suffix(".caption.txt"))
    pathlib.Path(tf).write_text(wrapped, encoding="utf-8")
    seg_dur = ffprobe_dur(seg)
    frames = max(2, int(round(seg_dur * TARGET_FPS)))

    # yazıyı sahne içi pencereye kısıtla (crossfade sırasında görünmesin)
    s0 = max(0.0, min(start_delay, max(0.0, seg_dur - 0.01)))
    e0 = max(0.0, seg_dur - max(0.0, end_cut))
    if e0 - s0 < 0.08:
        s0 = 0.0; e0 = seg_dur

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

    col = _ff_color(color)
    font_arg = f":fontfile={_ff_sanitize_font(font)}" if font else ""
    enable = f"between(t\\,{s0:.3f}\\,{e0:.3f})"

    common = f"textfile='{tf}':fontsize={fs}:x=(w-text_w)/2:y={y_pos}:line_spacing=10:enable='{enable}'"
    shadow = f"drawtext={common}{font_arg}:fontcolor=black@0.85:borderw=0"
    box    = f"drawtext={common}{font_arg}:fontcolor=white@0.0:box=1:boxborderw={(22 if is_hook else 18)}:boxcolor=black@0.65"
    main   = f"drawtext={common}{font_arg}:fontcolor={col}:borderw={(5 if is_hook else 4)}:bordercolor=black@0.9"

    vf_overlay = f"{shadow},{box},{main}"
    vf = f"{vf_overlay},fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB,trim=start_frame=0:end_frame={frames}"
    tmp_out = str(pathlib.Path(outp).with_suffix(".tmp.mp4"))
    try:
        run([
            "ffmpeg","-y","-hide_banner","-loglevel","error",
            "-i", seg,
            "-vf", vf,
            "-r", str(TARGET_FPS), "-vsync","cfr",
            "-an",
            "-c:v","libx264","-preset","medium","-crf",str(max(16,CRF_VISUAL-3)),
            "-pix_fmt","yuv420p","-movflags","+faststart",
            tmp_out
        ])
        enforce_video_exact_frames(tmp_out, frames, outp)
    finally:
        pathlib.Path(tf).unlink(missing_ok=True)
        pathlib.Path(tmp_out).unlink(missing_ok=True)

def pad_video_to_duration(video_in: str, target_sec: float, outp: str):
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

# --- xfade ile çeşitli geçişler (çift bazlı süre) ---
def concat_videos_with_transitions(files: List[str], outp: str,
                                   trans_sec: float = TRANSITION_SEC,
                                   per_trans: Optional[List[float]] = None):
    if not files: raise RuntimeError("concat_videos_with_transitions: empty")
    if len(files) == 1:
        return concat_videos_filter(files, outp)

    inputs = []
    for p in files:
        inputs += ["-i", p]
    durs = [max(0.02, ffprobe_dur(p)) for p in files]

    if per_trans is None or len(per_trans) != len(files)-1:
        per_trans = _compute_pairwise_transitions(durs)

    pre = [f"[{i}:v]fps={TARGET_FPS},settb=AVTB,setpts=N/{TARGET_FPS}/TB[v{i}]" for i in range(len(files))]

    chain = []
    cur_label = "[v0]"
    cur_dur = durs[0]
    for i in range(1, len(files)):
        trans = TRANSITION_LIST[(i-1) % max(1, len(TRANSITION_LIST))]
        tsec = max(0.04, min(per_trans[i-1], cur_dur-0.01, durs[i]-0.01))
        offset = max(0.0, cur_dur - tsec)
        next_label = f"[x{i}]"
        chain.append(f"{cur_label}[v{i}]xfade=transition={trans}:duration={tsec:.3f}:offset={offset:.3f}{next_label}")
        cur_label = next_label
        cur_dur = cur_dur + durs[i] - tsec

    filtergraph = ";".join(pre + chain)
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        *inputs,
        "-filter_complex", filtergraph,
        "-map", cur_label,
        "-r", str(TARGET_FPS), "-vsync","cfr",
        "-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),
        "-pix_fmt","yuv420p","-movflags","+faststart",
        outp
    ])

# -------------------- Audio concat (lossless) --------------------
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

# --- İsteğe bağlı: Ses tarafında acrossfade; varsayılan KAPALI ---
def concat_audios_with_transitions(files: List[str], outp: str, trans_sec: float = TRANSITION_SEC):
    if not files: raise RuntimeError("concat_audios_with_transitions: empty")
    if len(files) == 1 or trans_sec <= 0.01:
        return concat_audios(files, outp)

    inputs = []
    for p in files:
        inputs += ["-i", p]

    pre = [f"[{i}:a]aformat=sample_fmts=s16:channel_layouts=mono,aresample=48000,asetpts=N/SR/TB[a{i}]" for i in range(len(files))]

    chain = []
    cur_label = "[a0]"
    for i in range(1, len(files)):
        next_label = f"[ax{i}]"
        chain.append(f"{cur_label}[a{i}]acrossfade=d={trans_sec:.3f}:c1=tri:c2=tri{next_label}")
        cur_label = next_label

    filtergraph = ";".join(pre + chain)
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        *inputs,
        "-filter_complex", filtergraph,
        "-map", cur_label,
        "-ar","48000","-ac","1",
        "-c:a","pcm_s16le",
        outp
    ])

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

# -------------------- Template selection (by TOPIC) --------------------
def _select_template_key(topic: str) -> str:
    t = (topic or "").lower()
    geo_kw = ("country", "geograph", "city", "capital", "border", "population", "continent", "flag")
    if any(k in t for k in geo_kw):
        return "country_facts"
    return "_default"

# -------------------- Gemini (topic-locked) --------------------
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

BANNED_PHRASES = [
    "one clear tip", "see it", "learn it", "plot twist",
    "soap-opera narration", "repeat once", "takeaway action",
    "in 60 seconds", "just the point", "crisp beats"
]

def _content_score(sentences: List[str]) -> float:
    if not sentences: return 0.0
    bad = 0
    for s in sentences:
        low = (s or "").lower()
        if any(bp in low for bp in BANNED_PHRASES): bad += 1
        if len(low.split()) < 5: bad += 0.5
    return max(0.0, 10.0 - (bad * 1.4))

def _gemini_call(prompt: str, model: str) -> dict:
    if not GEMINI_API_KEY: raise RuntimeError("GEMINI_API_KEY missing")
    headers = {"Content-Type":"application/json","x-goog-api-key":GEMINI_API_KEY}
    payload = {"contents":[{"parts":[{"text": prompt}]}], "generationConfig":{"temperature":0.75}}
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

def build_via_gemini(channel_name: str, topic_lock: str, user_terms: List[str], banlist: List[str]) -> Tuple[str,List[str],List[str],str,str,List[str]]:
    tpl_key = _select_template_key(topic_lock)
    template = ENHANCED_GEMINI_TEMPLATES[tpl_key]
    avoid = "\n".join(f"- {b}" for b in banlist[:15]) if banlist else "(none)"
    terms_hint = ", ".join(user_terms[:10]) if user_terms else "(none)"
    guardrails = """
RULES (MANDATORY):
- STAY ON TOPIC exactly as provided.
- No country/geography pivot unless topic is about geography.
- No meta sentences, no headers. Return ONLY JSON, no prose/markdown."""
    prompt = f"""{template}

Channel: {channel_name}
Language: {LANG}
TOPIC (hard lock): {topic_lock}
Seed search terms (use and expand): {terms_hint}
Avoid overlap for 180 days:
{avoid}
{guardrails}
"""
    data = _gemini_call(prompt, GEMINI_MODEL)

    topic   = topic_lock
    sentences = [clean_caption_text(s) for s in (data.get("sentences") or [])]
    sentences = [s for s in sentences if s][:8]
    terms = data.get("search_terms") or []
    if isinstance(terms, str): terms=[terms]
    terms = [t.strip() for t in terms if isinstance(t,str) and t.strip()]
    if user_terms:
        pref = [t for t in user_terms if t not in terms]
        terms = pref + terms
    title = (data.get("title") or "").strip()
    desc  = (data.get("description") or "").strip()
    tags  = [t.strip() for t in (data.get("tags") or []) if isinstance(t,str) and t.strip()]
    return topic, sentences, terms, title, desc, tags

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
    if any(k in t for k in ["bridge","tunnel","arch","span"]):
        s.update(["suspension bridge","cable stayed","stone arch","viaduct","aerial city bridge"])
    if any(k in t for k in ["ocean","coast","tide","wave","storm"]):
        s.update(["ocean waves","coastal storm","rocky coast","lighthouse coast"])
    if any(k in t for k in ["timelapse","growth","melt","cloud"]):
        s.update(["city timelapse","plant growth","melting ice","cloud timelapse"])
    if any(k in t for k in ["mechanism","gears","pulley","cam"]):
        s.update(["macro gears","belt pulley","cam follower","robotic arm macro"])
    return list(s)

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
    for s in sentences:
        s_low = " " + (s or "").lower() + " "
        picked=None

        for ph in phrase_pool:
            if f" {ph} " in s_low:
                picked = ph; break

        if not picked:
            toks = _tok4(s)
            if len(toks) >= 2:
                picked = f"{toks[0]} {toks[1]}"
            elif len(toks) == 1:
                picked = toks[0]

        if (not picked or len(picked) < 4) and fb:
            picked = fb[fb_idx % len(fb)]; fb_idx += 1

        if (not picked or len(picked) < 4) and topic_key_join:
            picked = topic_key_join

        if not picked or picked in ("great","nice","good","bad","things","stuff"):
            picked = "macro detail"

        if len(picked.split()) > 2:
            w = picked.split(); picked = f"{w[-2]} {w[-1]}"

        queries.append(picked)

    return queries

# -------------------- Pexels (multi-pick + variety) --------------------
_USED_PEXELS_IDS_RUNTIME = set()

def _pexels_headers():
    if not PEXELS_API_KEY: raise RuntimeError("PEXELS_API_KEY missing")
    return {"Authorization": PEXELS_API_KEY}

def _pexels_search(query: str, locale: str) -> List[Tuple[int, str, int, int, float]]:
    url = "https://api.pexels.com/videos/search"
    r = requests.get(
        url,
        headers=_pexels_headers(),
        params={"query": query, "per_page": max(10, min(80, PEXELS_PER_PAGE)), "orientation":"portrait", "size":"large", "locale": locale},
        timeout=30
    )
    if r.status_code != 200:
        return []
    data = r.json() or {}
    out=[]
    for v in data.get("videos", []):
        vid = int(v.get("id", 0))
        dur = float(v.get("duration",0.0))
        files = v.get("video_files", []) or []
        if not files: continue
        pf = []
        for x in files:
            w = int(x.get("width",0)); h = int(x.get("height",0))
            if h >= 1080 and (h >= w or PEXELS_ALLOW_LANDSCAPE):
                pf.append((w,h,x.get("link")))
        if not pf: continue
        pf.sort(key=lambda t: (abs(t[1]-1440), t[0]*t[1]))
        w,h,link = pf[0]
        out.append((vid, link, w, h, dur))
    return out

def pexels_pick_many(query: str) -> List[Tuple[int,str]]:
    locale = "tr-TR" if LANG.startswith("tr") else "en-US"
    items = _pexels_search(query, locale)
    if not items and locale == "tr-TR":
        items = _pexels_search(query, "en-US")
    if not items:
        return []
    block = _blocklist_get_pexels()
    cand=[]
    qtokens= set(re.findall(r"[a-z0-9]+", query.lower()))
    for vid, link, w, h, dur in items:
        if vid in block or vid in _USED_PEXELS_IDS_RUNTIME:
            continue
        dur_bonus = 1.0 if 2.0 <= dur <= 12.0 else 0.0
        tokens = set(re.findall(r"[a-z0-9]+", (link or "").lower()))
        overlap = len(tokens & qtokens)
        score = overlap*2.0 + dur_bonus + (1.0 if 1080 <= h else 0.0)
        cand.append((score, vid, link))
    cand.sort(key=lambda x: x[0], reverse=True)
    out=[]
    for _, vid, link in cand:
        if vid not in _USED_PEXELS_IDS_RUNTIME:
            out.append((vid, link))
    return out[:5]  # soru başına en fazla 5 aday

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

# -------------------- Long SEO Description --------------------
def build_long_description(channel: str, topic: str, sentences: List[str], tags: List[str]) -> Tuple[str, str, List[str]]:
    hook = (sentences[0].rstrip(" .!?") if sentences else topic or channel)
    title = (hook[:1].upper() + hook[1:])[:95]
    para = " ".join(sentences)
    explainer = (
        f"{para} "
        f"This short explores “{topic}” with clear, visual steps so you can grasp it at a glance. "
        f"Rewatch to catch tiny details, save for later, and share with someone who’ll enjoy it."
    )
    tagset = []
    base_terms = [w for w in re.findall(r"[A-Za-z]{3,}", (topic or ""))][:5]
    for t in base_terms:
        tagset.append("#" + t.lower())
    tagset += ["#shorts", "#learn", "#visual", "#broll", "#education"]
    if tags:
        for t in tags[:10]:
            tclean = re.sub(r"[^A-Za-z0-9]+","", t).lower()
            if tclean and ("#"+tclean) not in tagset:
                tagset.append("#"+tclean)
    body = (
        f"{explainer}\n\n"
        f"— Key takeaways —\n"
        + "\n".join([f"• {s}" for s in sentences[:8]]) +
        "\n\n— Why it matters —\n"
        f"This topic sticks because it ties a vivid visual to a single idea per scene. "
        f"That’s how your brain remembers faster and better.\n\n"
        f"— Watch next —\n"
        f"Subscribe for more {topic.lower()} in clear, repeatable visuals.\n\n"
        + " ".join(tagset)
    )
    if len(body) > 4900:
        body = body[:4900]
    yt_tags = []
    for h in tagset:
        k = h[1:]
        if k and k not in yt_tags:
            yt_tags.append(k)
        if len(yt_tags) >= 15: break
    return title, body, yt_tags

# --------- İlk Pexels kaynağını ikiye böl (loop efekti) ----------
def split_video_in_half(src: str, out_first_half: str, out_second_half: str) -> bool:
    try:
        dur = ffprobe_dur(src)
        if dur <= 0.6:
            return False
        half = max(0.3, dur / 2.0)
        run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", src, "-t", f"{half:.3f}", "-c","copy", out_first_half])
        run(["ffmpeg","-y","-hide_banner","-loglevel","error","-ss", f"{half:.3f}", "-i", src, "-t", f"{max(0.3, dur-half):.3f}", "-c","copy", out_second_half])
        return pathlib.Path(out_first_half).exists() and pathlib.Path(out_second_half).exists()
    except Exception as e:
        print(f"⚠️ split_video_in_half failed: {e}")
        return False

# -------------------- Main --------------------
def main():
    print(f"==> {CHANNEL_NAME} | MODE={MODE} | topic-first build")
    random.seed(ROTATION_SEED or int(time.time()))

    topic_lock = TOPIC or "Interesting Visual Explainers"
    user_terms = SEARCH_TERMS_ENV

    # 1) İçerik üretim (topic-locked) + kalite kontrol
    attempts = 0
    best = None; best_score = -1.0
    banlist = _recent_topics_for_prompt()
    while attempts < 3:
        attempts += 1
        if USE_GEMINI and GEMINI_API_KEY:
            try:
                tpc, sents, search_terms, ttl, desc, tags = build_via_gemini(CHANNEL_NAME, topic_lock, user_terms, banlist)
            except Exception as e:
                print(f"Gemini error: {str(e)[:200]}")
                tpc = topic_lock; sents=[]; search_terms=user_terms or []
                ttl = ""; desc = ""; tags=[]
        else:
            tpc = topic_lock
            sents = [
                f"{tpc} comes alive in small vivid scenes.",
                "Each beat shows one concrete detail to remember.",
                "The story moves forward without fluff or filler.",
                "You can picture it clearly as you listen.",
                "A tiny contrast locks the idea in memory.",
                "No meta talk—just what matters on screen.",
                "Replay to catch micro-details and patterns."
            ]
            search_terms = user_terms or ["macro detail","timelapse","clean b-roll"]
            ttl = ""; desc=""; tags=[]

        score = _content_score(sents)
        print(f"📝 Content: {tpc} | {len(sents)} lines | score={score:.2f}")
        if score > best_score:
            best = (tpc, sents, search_terms, ttl, desc, tags)
            best_score = score
        if score >= 7.2:
            break
        else:
            print("⚠️ Low content score → rebuilding…")
            banlist = [tpc] + banlist
            time.sleep(0.5)

    tpc, sentences, search_terms, ttl, desc, tags = best
    sig = f"{CHANNEL_NAME}|{tpc}|{sentences[0] if sentences else ''}"
    _record_recent(_hash12(sig), MODE, tpc)

    print(f"📊 Sentences: {len(sentences)}")

    # 2) TTS
    tmp = tempfile.mkdtemp(prefix="enhanced_shorts_")
    font = font_path()
    wavs, metas = [], []
    print("🎤 TTS…")
    for i, s in enumerate(sentences):
        base = normalize_sentence(s)
        w = str(pathlib.Path(tmp) / f"sent_{i:02d}.wav")
        d = tts_to_wav(base, w)
        wavs.append(w); metas.append((base, d))
        print(f"   {i+1}/{len(sentences)}: {d:.2f}s")

    # 3) Pexels — çoklu aday + havuz + tekrar limiti
    per_scene_queries = build_per_scene_queries([m[0] for m in metas], (search_terms or user_terms or []), topic=tpc)
    print("🔎 Per-scene queries:")
    for q in per_scene_queries: print(f"   • {q}")

    pool: List[Tuple[int,str]] = []
    seen_ids=set()
    for q in per_scene_queries:
        picks = pexels_pick_many(q)
        for vid, link in picks:
            if vid not in seen_ids:
                seen_ids.add(vid); pool.append((vid, link))

    # Havuz yetersizse generic fallback
    if len(pool) < len(metas):
        extras = ["macro detail","city timelapse","nature macro","clean interior","close up hands","ocean waves","night skyline","forest path"]
        for q in (user_terms or []) + extras:
            for vid, link in pexels_pick_many(q):
                if vid not in seen_ids:
                    seen_ids.add(vid); pool.append((vid, link))
                if len(pool) >= len(metas)*2:
                    break
            if len(pool) >= len(metas)*2:
                break

    if not pool:
        raise RuntimeError("Pexels: hiç uygun klip bulunamadı.")

    # Klipleri indir ve tekrar limitine göre dağıt
    downloads = {}
    print("⬇️ Download pool…")
    for idx, (vid, link) in enumerate(pool):
        try:
            f = str(pathlib.Path(tmp) / f"pool_{idx:02d}_{vid}.mp4")
            with requests.get(link, stream=True, timeout=120) as rr:
                rr.raise_for_status()
                with open(f, "wb") as w:
                    for ch in rr.iter_content(8192):
                        w.write(ch)
            if pathlib.Path(f).stat().st_size > 300_000:
                downloads[vid] = f
        except Exception as e:
            print(f"⚠️ download fail ({vid}): {e}")

    # Dağıtım
    usage = {vid:0 for vid in downloads.keys()}
    chosen_files=[]
    for i in range(len(metas)):
        picked_path=None
        # öncelik: hiç kullanılmamış
        for vid, p in downloads.items():
            if usage[vid] < PEXELS_MAX_USES_PER_CLIP:
                picked_path = p
                usage[vid] += 1
                break
        if not picked_path:
            # mecburen en az kullanılanı seç
            if not usage: raise RuntimeError("Pexels pool empty after filtering.")
            vid = min(usage.keys(), key=lambda k: usage[k])
            usage[vid] += 1
            picked_path = downloads[vid]
        chosen_files.append(picked_path)

    # LOOP: İlk kaynağı ikiye böl → 2. yarı ilk sahne, 1. yarı son sahne
    if chosen_files:
        src0 = chosen_files[0]
        first_half = str(pathlib.Path(tmp)/"loop_first_half.mp4")
        second_half = str(pathlib.Path(tmp)/"loop_second_half.mp4")
        if split_video_in_half(src0, first_half, second_half):
            chosen_files[0] = second_half
            chosen_files[-1] = first_half
            print("🔁 Loop mode: first clip split → second half at scene #1, first half at LAST scene.")
        else:
            print("ℹ️ Loop split skipped (source too short or split failed).")

    # 4) Segment + altyazı
    print("🎬 Segments…")
    segs = []
    total_scenes = len(metas)

    # TTS süreleri baz alınarak geçiş süreleri (çift bazlı) hesapla
    durs = [m[1] for m in metas]
    per_trans = _compute_pairwise_transitions(durs)

    for i, ((base_text, d), src) in enumerate(zip(metas, chosen_files)):
        base   = str(pathlib.Path(tmp) / f"seg_{i:02d}.mp4")
        # İlk sahnede giriş fade yok; son sahnede bitiş fade yok
        make_segment(src, d, base, no_start_fade=(i == 0), no_end_fade=(i == total_scenes - 1))
        colored = str(pathlib.Path(tmp) / f"segsub_{i:02d}.mp4")

        # altyazıyı crossfade dışında tut
        start_delay = per_trans[i-1] if i > 0 else 0.0
        end_cut     = per_trans[i]   if i < total_scenes-1 else 0.0

        draw_capcut_text(
            base,
            base_text,
            CAPTION_COLORS[i % len(CAPTION_COLORS)],
            font,
            colored,
            is_hook=(i == 0),
            start_delay=start_delay,
            end_cut=end_cut
        )
        segs.append(colored)

    # 5) Birleştir — video & ses (ses fadesiz varsayılan)
    print("🎞️ Assemble…")
    vcat = str(pathlib.Path(tmp) / "video_concat.mp4"); concat_videos_with_transitions(segs, vcat, per_trans=per_trans)

    acat = str(pathlib.Path(tmp) / "audio_concat.wav")
    if AUDIO_CROSSFADE:
        concat_audios_with_transitions(wavs, acat)
    else:
        concat_audios(wavs, acat)

    # 6) Süre & kare kilitleme
    adur = ffprobe_dur(acat); vdur = ffprobe_dur(vcat)
    if vdur + 0.02 < adur:
        vcat_padded = str(pathlib.Path(tmp) / "video_padded.mp4")
        pad_video_to_duration(vcat, adur, vcat_padded)
        vcat = vcat_padded
    a_frames = max(2, int(round(adur * TARGET_FPS)))
    vcat_exact = str(pathlib.Path(tmp) / "video_exact.mp4"); enforce_video_exact_frames(vcat, a_frames, vcat_exact); vcat = vcat_exact
    acat_exact = str(pathlib.Path(tmp) / "audio_exact.wav"); lock_audio_duration(acat, a_frames, acat_exact); acat = acat_exact
    vdur2 = ffprobe_dur(vcat); adur2 = ffprobe_dur(acat)
    print(f"🔒 Locked A/V: video={vdur2:.3f}s | audio={adur2:.3f}s | fps={TARGET_FPS}")

    # 7) Mux
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^A-Za-z0-9]+', '_', tpc)[:60] or "Short"
    outp = f"{OUT_DIR}/{CHANNEL_NAME}_{safe_topic}_{ts}.mp4"
    print("🔄 Mux…")
    mux(vcat, acat, outp)
    final = ffprobe_dur(outp)
    print(f"✅ Saved: {outp} ({final:.2f}s)")

    # 8) Metadata (long SEO)
    title, description, yt_tags = build_long_description(CHANNEL_NAME, tpc, [m[0] for m in metas], tags)

    meta = {
        "title": title,
        "description": description,
        "tags": yt_tags,
        "privacy": VISIBILITY,
        "defaultLanguage": LANG,
        "defaultAudioLanguage": LANG
    }

    # 9) Upload (varsa env)
    try:
        if os.getenv("UPLOAD_TO_YT","1") == "1":
            print("📤 Uploading to YouTube…")
            vid_id = upload_youtube(outp, meta)
            print(f"🎉 YouTube Video ID: {vid_id}\n🔗 https://youtube.com/watch?v={vid_id}")
        else:
            print("⏭️ Upload disabled (UPLOAD_TO_YT != 1)")
    except Exception as e:
        print(f"❌ Upload skipped: {e}")

    # 10) Temizlik
    try:
        shutil.rmtree(tmp); print("🧹 Cleaned temp files")
    except: pass

if __name__ == "__main__":
    main()
