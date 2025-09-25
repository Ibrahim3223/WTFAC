# autoshorts_daily.py — A/V frame-lock: per-scene, per-transition; mirrored audio acrossfade; hard total-frames target
# -*- coding: utf-8 -*-
import os, sys, re, json, time, random, datetime, tempfile, pathlib, subprocess, hashlib, math, shutil
from typing import List, Optional, Tuple, Dict

# -------------------- ENV / constants --------------------
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

# Geçiş (transition) ayarları
TRANSITION_SEC  = float(os.getenv("TRANSITION_SEC", "0.28"))
TRANSITION_LIST = [t.strip() for t in os.getenv(
    "TRANSITION_LIST",
    "fade,dissolve,slideleft,slideright,slideup,slidedown,"
    "wipeleft,wiperight,wipeup,wipedown,smoothleft,smoothright,"
    "circleopen,circleclose,radial,zoom,pixelize,distance,squeezeh,squeezew"
).split(",") if t.strip()]

# 🔴 ÖNEMLİ: A/V eşleşmesi için ses acrossfade varsayılan AÇIK
AUDIO_CROSSFADE = os.getenv("AUDIO_CROSSFADE", "1") == "1"
FINAL_LOUDNORM  = os.getenv("FINAL_LOUDNORM",  "1") == "1"  # final tek geçiş loudness

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

PEXELS_PER_PAGE          = int(os.getenv("PEXELS_PER_PAGE", "30"))
PEXELS_MAX_USES_PER_CLIP = int(os.getenv("PEXELS_MAX_USES_PER_CLIP", "1"))
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
    return _load_json(STATE_FILE, {"recent": [], "used_pexels_ids": []})

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

def _record_recent(h: str, mode: str, topic: str):
    st = _state_load()
    st.setdefault("recent", []).append({"h":h,"mode":mode,"topic":topic,"ts":time.time()})
    _state_save(st)  # ✅ önceki ufak düzeltmemiz
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
    """
    Per-cümle: sadece hız/temiz dönüşüm. Dinamik normalizasyon YOK.
    Final mix’te tek sefer loudnorm/limiter uygulanacak.
    """
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
                "-af", f"atempo={atempo}",
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
            "-af", f"atempo={atempo}",
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

def frames_to_seconds(frames: int, fps: int = TARGET_FPS) -> float:
    return max(0.0, int(frames) / float(fps))

def _compute_pairwise_transitions(quantized_durs: List[float]) -> List[float]:
    if not quantized_durs or len(quantized_durs) < 2:
        return []
    per = []
    for i in range(len(quantized_durs)-1):
        raw = min(TRANSITION_SEC, 0.45 * min(quantized_durs[i], quantized_durs[i+1]))
        frames, qdur = quantize_to_frames(max(0.0, raw), TARGET_FPS)
        per.append(qdur)  # zaten kareye kuantize
    return per

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
    """
    FIX: Eğer kaynak video hedeften kısa ise tpad ile son kareyi klonlayıp pad ediyor,
    ardından kesin olarak end_frame=target_frames'e trimliyor.
    """
    target_frames = max(2, int(target_frames))
    target_sec = frames_to_seconds(target_frames)
    vdur = ffprobe_dur(video_in)
    extra = max(0.0, target_sec - vdur)
    if extra > (1.0 / TARGET_FPS) * 0.5:
        vf = f"fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB,tpad=stop_mode=clone:stop_duration={extra:.6f},trim=start_frame=0:end_frame={target_frames}"
    else:
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

def concat_videos_with_transitions_frames(files: List[str], outp: str,
                                          seg_frames: List[int],
                                          trans_frames: List[int]):
    """
    Xfade’i kare cinsinden kur → toplam kare = sum(seg_frames) - sum(trans_frames).
    """
    n = len(files)
    if n == 0: raise RuntimeError("concat_videos_with_transitions_frames: empty")
    if n == 1:
        # Tek parça: sadece fps/trim uygula
        enforce_video_exact_frames(files[0], seg_frames[0], outp); return

    inputs = []
    for p in files: inputs += ["-i", p]

    # per-input hazırlık (fps + tam kare trim)
    pre = []
    for i in range(n):
        pre.append(f"[{i}:v]fps={TARGET_FPS},settb=AVTB,trim=start_frame=0:end_frame={seg_frames[i]},setpts=N/{TARGET_FPS}/TB[v{i}]")

    chain = []
    cur_label = "[v0]"
    cur_frames = seg_frames[0]
    for i in range(1, n):
        # Güvenli tfr (sahnelerin sınırını aşma)
        tfr_req = trans_frames[i-1] if i-1 < len(trans_frames) else 0
        tfr = max(2, min(tfr_req, cur_frames-2, seg_frames[i]-2))  # ≥2 frame güvenlik
        dur = frames_to_seconds(tfr)
        offset = frames_to_seconds(cur_frames - tfr)
        trans = TRANSITION_LIST[(i-1) % max(1, len(TRANSITION_LIST))]
        next_label = f"[x{i}]"
        chain.append(f"{cur_label}[v{i}]xfade=transition={trans}:duration={dur:.6f}:offset={offset:.6f}{next_label}")
        cur_label = next_label
        cur_frames = cur_frames + seg_frames[i] - tfr

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

# -------------------- Audio helpers --------------------
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

def concat_audios_with_transitions(files: List[str], outp: str, trans_frames: List[int]):
    """
    Her komşu çift için acrossfade süresi = transition_frames[i]/FPS
    Giriş wav’ların hepsi zaten sahne_karelerine trimlenmiş olmalı.
    """
    if not files: raise RuntimeError("concat_audios_with_transitions: empty")
    if len(files) == 1 or sum(trans_frames or []) == 0:
        return concat_audios(files, outp)

    n = len(files)
    inputs = []
    for p in files: inputs += ["-i", p]

    # Aformat & pts hazırlığı
    pre = []
    for i in range(n):
        pre.append(f"[{i}:a]aformat=sample_fmts=s16:channel_layouts=mono,aresample=48000,asetpts=N/SR/TB[a{i}]")

    chain = []
    cur_label = "[a0]"
    for i in range(1, n):
        tfr = trans_frames[i-1] if i-1 < len(trans_frames) else 0
        tfr = max(2, tfr)  # ≥2 frame güvenlik
        d = frames_to_seconds(max(0, tfr))
        next_label = f"[ax{i}]"
        chain.append(f"{cur_label}[a{i}]acrossfade=d={d:.6f}:c1=tri:c2=tri{next_label}")
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
    """
    FIX: Hedef süreden kısaysa 'apad' ile sessiz pad; uzunsa trim.
    Böylece mux sırasında -shortest yüzünden erken kesilme olmaz.
    """
    dur = frames_to_seconds(target_frames)
    adur = ffprobe_dur(audio_in)
    extra = max(0.0, dur - adur)
    if extra > (1.0 / TARGET_FPS) * 0.5:
        af = f"apad=pad_dur={extra:.6f},atrim=end={dur:.6f},asetpts=N/SR/TB"
    else:
        af = f"atrim=end={dur:.6f},asetpts=N/SR/TB"
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", audio_in,
        "-af", af,
        "-ar","48000","-ac","1",
        "-c:a","pcm_s16le",
        outp
    ])

def apply_final_loudness_normalization(audio_in: str, audio_out: str):
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", audio_in,
        "-af", "loudnorm=I=-16:LRA=11:TP=-1.5:dual_mono=true,alimiter=limit=0.95",
        "-ar","48000","-ac","1","-c:a","pcm_s16le",
        audio_out
    ])

def mux(video: str, audio: str, outp: str):
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", video, "-i", audio,
        "-map","0:v:0","-map","1:a:0",
        "-c:v","copy",
        "-c:a","aac","-b:a","256k",
        "-movflags","+faststart",
        # "-shortest",  # ❌ kaldırıldı: A/V zaten aynı uzunlukta kilitleniyor
        "-muxpreload","0","-muxdelay","0",
        "-avoid_negative_ts","make_zero",
        outp
    ])

# -------------------- Template & Gemini (değişmedi / kısaltılmış) --------------------
def _select_template_key(topic: str) -> str:
    t = (topic or "").lower()
    geo_kw = ("country", "geograph", "city", "capital", "border", "population", "continent", "flag")
    if any(k in t for k in geo_kw):
        return "country_facts"
    return "_default"

ENHANCED_GEMINI_TEMPLATES = {
    "_default": """Create a 25–40s YouTube Short.
Return STRICT JSON with keys: topic, sentences (7–8), search_terms (4–10), title, description, tags.

CONTENT RULES:
- Stay laser-focused on the provided TOPIC (no pivoting).
- Coherent, causally linked beats; each sentence advances a single concrete idea.
- No meta-instructions; no headers.
- Sentences must be visually anchorable with stock b-roll.
- 6–12 words per sentence; 7–8 sentences total.""",
    "country_facts": """Create amazing country/city facts.
Return STRICT JSON with keys: topic, sentences (7–8), search_terms, title, description, tags.
Rules:
- Specific facts; visually anchorable; 6–12 words per sentence."""
}

BANNED_PHRASES = [
    "one clear tip","see it","learn it","plot twist","soap-opera narration","repeat once","takeaway action",
    "in 60 seconds","just the point","crisp beats"
]

def _content_score(sentences: List[str]) -> float:
    if not sentences: return 0.0
    bad=0
    for s in sentences:
        low=(s or "").lower()
        if any(bp in low for bp in BANNED_PHRASES): bad+=1
        if len(low.split()) < 5: bad+=0.5
    return max(0.0, 10.0 - (bad*1.4))

def _gemini_call(prompt: str, model: str) -> dict:
    if not GEMINI_API_KEY: raise RuntimeError("GEMINI_API_KEY missing")
    headers={"Content-Type":"application/json","x-goog-api-key":GEMINI_API_KEY}
    payload={"contents":[{"parts":[{"text": prompt}]}], "generationConfig":{"temperature":0.75}}
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r=requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
    data=r.json()
    txt=""
    try: txt=data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception: txt=json.dumps(data)
    m=re.search(r"\{(?:.|\n)*\}", txt)
    if not m: raise RuntimeError("Gemini response parse error (no JSON)")
    raw=re.sub(r"^```json\s*|\s*```$", "", m.group(0).strip(), flags=re.MULTILINE)
    return json.loads(raw)

def clean_caption_text(s: str) -> str:
    t = (s or "").strip()
    t = (t.replace("—","-").replace("–","-").replace("“",'"').replace("”",'"').replace("’","'").replace("`",""))
    t = re.sub(r"\s+"," ", t).strip()
    if t and t[0].islower(): t = t[0].upper()+t[1:]
    return t

def build_via_gemini(channel_name: str, topic_lock: str, user_terms: List[str], banlist: List[str]):
    template = ENHANCED_GEMINI_TEMPLATES[_select_template_key(topic_lock)]
    avoid = "\n".join(f"- {b}" for b in banlist[:15]) if banlist else "(none)"
    terms_hint = ", ".join(user_terms[:10]) if user_terms else "(none)"
    guardrails = "Return ONLY JSON, no prose/markdown."
    prompt = f"""{template}

Channel: {channel_name}
Language: {LANG}
TOPIC (hard lock): {topic_lock}
Seed search terms: {terms_hint}
Avoid overlap:
{avoid}
{guardrails}
"""
    data = _gemini_call(prompt, GEMINI_MODEL)
    tpc = topic_lock
    sentences = [clean_caption_text(s) for s in (data.get("sentences") or [])][:8]
    terms = data.get("search_terms") or []
    if isinstance(terms, str): terms=[terms]
    terms = [t.strip() for t in terms if isinstance(t,str) and t.strip()]
    title = (data.get("title") or "").strip()
    desc  = (data.get("description") or "").strip()
    tags  = [t.strip() for t in (data.get("tags") or []) if isinstance(t,str) and t.strip()]
    return tpc, sentences, terms, title, desc, tags

# -------------------- Query & Pexels (özet/aynı mantık) --------------------
_STOP = set("a an the and or but if while of to in on at from by with for about into over after before between during under above across around through this that these those is are was were be been being have has had do does did can could should would may might will shall you your we our they their he she it its as than then so such very more most many much just also only even still yet".split())
_GENERIC_BAD={"great","good","bad","big","small","old","new","many","more","most","thing","things","stuff"}

def _proper_phrases(texts: List[str]) -> List[str]:
    phrases=[]
    for t in texts:
        for m in re.finditer(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", t or ""):
            phrase = re.sub(r"^(The|A|An)\s+", "", m.group(0))
            ws=[w.lower() for w in phrase.split()]
            for i in range(len(ws)-1): phrases.append(f"{ws[i]} {ws[i+1]}")
    seen=set(); out=[]
    for p in phrases:
        if p not in seen: seen.add(p); out.append(p)
    return out

def _tok4(s: str) -> List[str]:
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", (s or "").lower())
    return [w for w in s.split() if len(w)>=4 and w not in _STOP and w not in _GENERIC_BAD]

def build_per_scene_queries(sentences: List[str], fallback_terms: List[str], topic: Optional[str]=None) -> List[str]:
    topic = (topic or "").strip()
    texts_cap = [topic] + sentences
    phrase_pool = _proper_phrases(texts_cap)
    fb=[]
    for t in (fallback_terms or []):
        t = re.sub(r"[^A-Za-z0-9 ]+"," ", str(t)).strip().lower()
        ws=[w for w in t.split() if w not in _STOP and w not in _GENERIC_BAD]
        if ws: fb.append(" ".join(ws[:2]))
    topic_keys = _tok4(topic)[:2]
    topic_key_join = " ".join(topic_keys) if topic_keys else ""
    queries=[]
    fb_idx=0
    for s in sentences:
        s_low = " "+(s or "").lower()+" "
        picked=None
        for ph in phrase_pool:
            if f" {ph} " in s_low: picked=ph; break
        if not picked:
            toks=_tok4(s)
            picked = f"{toks[0]} {toks[1]}" if len(toks)>=2 else (toks[0] if toks else None)
        if (not picked or len(picked)<4) and fb:
            picked = fb[fb_idx % len(fb)]; fb_idx += 1
        if (not picked or len(picked)<4) and topic_key_join:
            picked = topic_key_join
        if not picked: picked="macro detail"
        if len(picked.split())>2:
            w=picked.split(); picked=f"{w[-2]} {w[-1]}"
        queries.append(picked)
    return queries

_USED_PEXELS_IDS_RUNTIME = set()
def _pexels_headers():
    if not PEXELS_API_KEY: raise RuntimeError("PEXELS_API_KEY missing")
    return {"Authorization": PEXELS_API_KEY}
def _pexels_search(query: str, locale: str):
    url="https://api.pexels.com/videos/search"
    r=requests.get(url, headers=_pexels_headers(), params={
        "query":query, "per_page":30, "orientation":"portrait", "size":"large", "locale":locale
    }, timeout=30)
    if r.status_code!=200: return []
    data=r.json() or {}
    out=[]
    for v in data.get("videos", []):
        vid=int(v.get("id",0)); dur=float(v.get("duration",0.0))
        files=v.get("video_files",[]) or []
        if not files: continue
        pf=[]
        for x in files:
            w=int(x.get("width",0)); h=int(x.get("height",0))
            if h>=1080 and (h>=w or PEXELS_ALLOW_LANDSCAPE):
                pf.append((w,h,x.get("link")))
        if not pf: continue
        pf.sort(key=lambda t: (abs(t[1]-1440), t[0]*t[1]))
        w,h,link=pf[0]
        out.append((vid, link, dur))
    return out
def pexels_pick_many(query: str):
    locale = "tr-TR" if LANG.startswith("tr") else "en-US"
    items = _pexels_search(query, locale) or _pexels_search(query, "en-US")
    if not items: return []
    block=_blocklist_get_pexels()
    qtokens=set(re.findall(r"[a-z0-9]+", query.lower()))
    cand=[]
    for vid, link, dur in items:
        if vid in block or vid in _USED_PEXELS_IDS_RUNTIME: continue
        dur_bonus = 1.0 if 2.0 <= dur <= 12.0 else 0.0
        tokens = set(re.findall(r"[a-z0-9]+", (link or "").lower()))
        overlap = len(tokens & qtokens)
        score = overlap*2.0 + dur_bonus + 1.0
        cand.append((score, vid, link))
    cand.sort(key=lambda x: x[0], reverse=True)
    out=[]
    for _, vid, link in cand:
        if vid not in _USED_PEXELS_IDS_RUNTIME:
            out.append((vid, link))
    return out[:5]

# -------------------- YouTube (değişmedi) --------------------
def yt_service():
    cid  = os.getenv("YT_CLIENT_ID"); csec = os.getenv("YT_CLIENT_SECRET"); rtok = os.getenv("YT_REFRESH_TOKEN")
    if not (cid and csec and rtok):
        raise RuntimeError("Missing YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN")
    creds = Credentials(token=None, refresh_token=rtok, token_uri="https://oauth2.googleapis.com/token",
                        client_id=cid, client_secret=csec, scopes=["https://www.googleapis.com/auth/youtube.upload"])
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds, cache_discovery=False)

def upload_youtube(video_path: str, meta: dict) -> str:
    y = yt_service()
    body = {
        "snippet": {
            "title": meta["title"], "description": meta["description"], "tags": meta.get("tags", []),
            "categoryId": "27", "defaultLanguage": meta.get("defaultLanguage", LANG),
            "defaultAudioLanguage": meta.get("defaultAudioLanguage", LANG)
        },
        "status": {"privacyStatus": meta.get("privacy", VISIBILITY), "selfDeclaredMadeForKids": False}
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    req = y.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = req.execute()
    return resp.get("id","")

# -------------------- Long SEO Description (değişmedi kısaltıldı) --------------------
def build_long_description(channel: str, topic: str, sentences: List[str], tags: List[str]) -> Tuple[str, str, List[str]]:
    hook = (sentences[0].rstrip(" .!?") if sentences else topic or channel)
    title = (hook[:1].upper() + hook[1:])[:95]
    para = " ".join(sentences)
    explainer = f"{para} This short explores “{topic}”."
    tagset = ["#shorts","#education","#visual"]
    body = explainer + "\n\n" + "\n".join([f"• {s}" for s in sentences[:8]]) + "\n\n" + " ".join(tagset)
    yt_tags = ["shorts","education","visual"]
    return title, body[:4900], yt_tags

# --------- İlk Pexels kaynağını ikiye böl (loop efekti) ----------
def split_video_in_half(src: str, out_first_half: str, out_second_half: str) -> bool:
    try:
        dur = ffprobe_dur(src)
        if dur <= 0.6: return False
        half = max(0.3, dur/2.0)
        run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", src, "-t", f"{half:.3f}", "-c","copy", out_first_half])
        run(["ffmpeg","-y","-hide_banner","-loglevel","error","-ss", f"{half:.3f}", "-i", src, "-t", f"{max(0.3,dur-half):.3f}", "-c","copy", out_second_half])
        return pathlib.Path(out_first_half).exists() and pathlib.Path(out_second_half).exists()
    except Exception as e:
        print(f"⚠️ split_video_in_half failed: {e}"); return False

# -------------------- Main --------------------
def main():
    print(f"==> {CHANNEL_NAME} | MODE={MODE} | topic-first build")
    random.seed(ROTATION_SEED or int(time.time()))

    topic_lock = TOPIC or "Interesting Visual Explainers"
    user_terms = SEARCH_TERMS_ENV

    # 1) İçerik
    attempts=0; best=None; best_score=-1.0; banlist=_recent_topics_for_prompt()
    while attempts<3:
        attempts+=1
        try:
            tpc, sents, search_terms, ttl, desc, tags = build_via_gemini(CHANNEL_NAME, topic_lock, user_terms, banlist)
        except Exception as e:
            print(f"Gemini error: {str(e)[:200]}"); tpc=topic_lock; sents=[]; search_terms=user_terms or []; ttl=""; desc=""; tags=[]
        score=_content_score(sents); print(f"📝 Content: {tpc} | {len(sents)} lines | score={score:.2f}")
        if score>best_score: best=(tpc,sents,search_terms,ttl,desc,tags); best_score=score
        if score>=7.2: break
        banlist=[tpc]+banlist; time.sleep(0.4)

    tpc, sentences, search_terms, ttl, desc, tags = best
    sig = f"{CHANNEL_NAME}|{tpc}|{sentences[0] if sentences else ''}"
    _record_recent(_hash12(sig), MODE, tpc)

    # 2) TTS
    tmp = tempfile.mkdtemp(prefix="enhanced_shorts_")
    font = font_path()
    wavs_raw, metas = [], []
    print("🎤 TTS…")
    for i, s in enumerate(sentences):
        base = normalize_sentence(s)
        w = str(pathlib.Path(tmp) / f"sent_raw_{i:02d}.wav")
        d = tts_to_wav(base, w)
        wavs_raw.append(w); metas.append((base, d))
        print(f"   {i+1}/{len(sentences)}: {d:.2f}s")

    # 2.5) Süreleri kareye kuantize et
    scene_frames=[]; scene_qdurs=[]
    for _, d in metas:
        fr, qd = quantize_to_frames(d, TARGET_FPS)
        scene_frames.append(fr); scene_qdurs.append(qd)

    # 2.6) Her sahne sesi → tam kareye kilitle
    print("🔒 Trim audio per-scene to quantized frames…")
    wavs=[]
    for i, w in enumerate(wavs_raw):
        w_exact = str(pathlib.Path(tmp) / f"sent_{i:02d}.wav")
        lock_audio_duration(w, scene_frames[i], w_exact)
        wavs.append(w_exact)

    # 3) Pexels
    per_scene_queries = build_per_scene_queries([m[0] for m in metas], (search_terms or user_terms or []), topic=tpc)
    print("🔎 Per-scene queries:"); [print(f"   • {q}") for q in per_scene_queries]
    pool=[]; seen=set()
    for q in per_scene_queries:
        for vid, link in pexels_pick_many(q):
            if vid not in seen: seen.add(vid); pool.append((vid, link))
    if len(pool) < len(metas):
        extras=["macro detail","city timelapse","nature macro","close up hands","ocean waves","night skyline","forest path"]
        for q in (user_terms or []) + extras:
            for vid, link in pexels_pick_many(q):
                if vid not in seen: seen.add(vid); pool.append((vid, link))
                if len(pool)>=len(metas)*2: break
            if len(pool)>=len(metas)*2: break
    if not pool: raise RuntimeError("Pexels: no suitable clips.")

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
        except Exception as e:
            print(f"⚠️ download fail ({vid}): {e}")

    usage={vid:0 for vid in downloads.keys()}
    chosen_files=[]
    for i in range(len(metas)):
        pick=None
        for vid,p in downloads.items():
            if usage[vid] < PEXELS_MAX_USES_PER_CLIP:
                pick=p; usage[vid]+=1; break
        if not pick:
            if not usage: raise RuntimeError("Pexels pool empty after filtering.")
            vid=min(usage.keys(), key=lambda k: usage[k]); usage[vid]+=1; pick=downloads[vid]
        chosen_files.append(pick)

    if chosen_files:
        src0 = chosen_files[0]
        first_half = str(pathlib.Path(tmp)/"loop_first_half.mp4")
        second_half = str(pathlib.Path(tmp)/"loop_second_half.mp4")
        if split_video_in_half(src0, first_half, second_half):
            chosen_files[0] = second_half
            chosen_files[-1] = first_half
            print("🔁 Loop mode: split first clip → 2nd half first, 1st half last.")
        else:
            print("ℹ️ Loop split skipped.")

    # 4) Segment + altyazı
    print("🎬 Segments…")
    segs=[]
    total_scenes=len(metas)
    per_trans = _compute_pairwise_transitions(scene_qdurs)               # saniye (kuantize)
    trans_frames = [quantize_to_frames(t, TARGET_FPS)[0] for t in per_trans]  # kare cinsinden

    for i, ((base_text,_), src) in enumerate(zip(metas, chosen_files)):
        base   = str(pathlib.Path(tmp) / f"seg_{i:02d}.mp4")
        make_segment(src, scene_qdurs[i], base, no_start_fade=(i==0), no_end_fade=(i==total_scenes-1))
        colored = str(pathlib.Path(tmp) / f"segsub_{i:02d}.mp4")
        start_delay = frames_to_seconds(trans_frames[i-1]) if i>0 else 0.0
        end_cut     = frames_to_seconds(trans_frames[i])   if i<total_scenes-1 else 0.0
        draw_capcut_text(
            base, base_text, CAPTION_COLORS[i % len(CAPTION_COLORS)], font, colored,
            is_hook=(i==0), start_delay=start_delay, end_cut=end_cut
        )
        segs.append(colored)

    # 5) Video concat (kare bazlı xfade) + Ses concat (aynı bindirme)
    print("🎞️ Assemble…")
    vcat = str(pathlib.Path(tmp) / "video_concat.mp4")
    concat_videos_with_transitions_frames(segs, vcat, scene_frames, trans_frames)

    acat = str(pathlib.Path(tmp) / "audio_concat.wav")
    if AUDIO_CROSSFADE:
        concat_audios_with_transitions(wavs, acat, trans_frames)
    else:
        # UYARI: video xfade → süre kısalır. Ses crossfade kapalıysa final eşitleme aşağıda yapılacak.
        concat_audios(wavs, acat)

    # 5.5) Final loudness
    if FINAL_LOUDNORM:
        acat_norm = str(pathlib.Path(tmp) / "audio_loudnorm.wav")
        apply_final_loudness_normalization(acat, acat_norm)
        acat = acat_norm

    # 6) Hedef kare hesabı ve KESİN eşitleme
    target_frames_total = max(2, sum(scene_frames) - sum(trans_frames))
    # Üretimden çıkan gerçek süreleri ölç (güvenlik)
    vdur_real = ffprobe_dur(vcat)
    adur_real = ffprobe_dur(acat)
    vframes_real = int(round(vdur_real * TARGET_FPS))
    aframes_real = int(round(adur_real * TARGET_FPS))
    final_target_frames = max(target_frames_total, vframes_real, aframes_real)  # kısa olanı pad'le

    vcat_exact = str(pathlib.Path(tmp) / "video_exact.mp4"); enforce_video_exact_frames(vcat, final_target_frames, vcat_exact); vcat = vcat_exact
    acat_exact = str(pathlib.Path(tmp) / "audio_exact.wav"); lock_audio_duration(acat, final_target_frames, acat_exact); acat = acat_exact
    print(f"🔒 Locked A/V to exact frames: {final_target_frames} @ {TARGET_FPS}fps (~{frames_to_seconds(final_target_frames):.3f}s)")

    # 7) Mux
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^A-Za-z0-9]+', '_', tpc)[:60] or "Short"
    outp = f"{OUT_DIR}/{CHANNEL_NAME}_{safe_topic}_{ts}.mp4"
    print("🔄 Mux…")
    mux(vcat, acat, outp)
    print(f"✅ Saved: {outp} ({ffprobe_dur(outp):.2f}s)")

    # 8) Meta & Upload (opsiyonel)
    title, description, yt_tags = build_long_description(CHANNEL_NAME, tpc, [m[0] for m in metas], tags)
    meta = {"title": title, "description": description, "tags": yt_tags,
            "privacy": VISIBILITY, "defaultLanguage": LANG, "defaultAudioLanguage": LANG}
    try:
        if os.getenv("UPLOAD_TO_YT","1") == "1":
            print("📤 Uploading to YouTube…")
            vid_id = upload_youtube(outp, meta)
            print(f"🎉 YouTube Video ID: {vid_id}\n🔗 https://youtube.com/watch?v={vid_id}")
        else:
            print("⏭️ Upload disabled")
    except Exception as e:
        print(f"❌ Upload skipped: {e}")

    # 9) Temizlik
    try:
        shutil.rmtree(tmp); print("🧹 Cleaned temp files")
    except: pass

if __name__ == "__main__":
    main()
