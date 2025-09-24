# autoshorts_daily.py — Topic-first script, visual hints, looped first clip, stronger SEO
# -*- coding: utf-8 -*-
import os, sys, re, json, time, random, datetime, tempfile, pathlib, subprocess, hashlib, math, shutil
from typing import List, Optional, Tuple

# -------------------- ENV / constants --------------------
VOICE_STYLE    = os.getenv("TTS_STYLE", "narration-professional")
TARGET_MIN_SEC = float(os.getenv("TARGET_MIN_SEC", "22"))
TARGET_MAX_SEC = float(os.getenv("TARGET_MAX_SEC", "42"))

CHANNEL_NAME   = os.getenv("CHANNEL_NAME", "DefaultChannel")
MODE           = os.getenv("MODE", "freeform").strip().lower()   # sadece fallback, yönlendirme yapmaz
LANG           = os.getenv("LANG", "en")
VISIBILITY     = os.getenv("VISIBILITY", "public")
ROTATION_SEED  = int(os.getenv("ROTATION_SEED", "0"))
UPLOAD_TO_YT   = os.getenv("UPLOAD_TO_YT", "1") != "0"

OUT_DIR        = "out"; pathlib.Path(OUT_DIR).mkdir(exist_ok=True)

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_GEMINI     = os.getenv("USE_GEMINI", "1") == "1"
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ---- Channel intent (topic & search terms) ----
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

# Caption tuneables
CAPTION_MAX_LINE  = int(os.getenv("CAPTION_MAX_LINE",  "28"))
CAPTION_MAX_LINES = int(os.getenv("CAPTION_MAX_LINES", "6"))

STATE_FILE = f"state_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json"
GLOBAL_TOPICS = "state_global_topics.json"  # opsiyonel global tekrar önleme

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
def _load_json(path: str, default):
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except:
        return default

def _state_load() -> dict:
    return _load_json(STATE_FILE, {"recent": [], "used_pexels_ids": []})

def _state_save(st: dict):
    st["recent"] = st.get("recent", [])[-1200:]
    st["used_pexels_ids"] = st.get("used_pexels_ids", [])[-5000:]
    pathlib.Path(STATE_FILE).write_text(json.dumps(st, indent=2), encoding="utf-8")

def _global_topics_load() -> dict:
    return _load_json(GLOBAL_TOPICS, {"topics": []})

def _global_topics_save(d: dict):
    d["topics"] = d.get("topics", [])[-3000:]
    pathlib.Path(GLOBAL_TOPICS).write_text(json.dumps(d, indent=2), encoding="utf-8")

def _hash12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _is_recent(h: str, window_days=365) -> bool:
    now = time.time()
    for r in _state_load().get("recent", []):
        if r.get("h")==h and (now - r.get("ts",0)) < window_days*86400:
            return True
    return False

def _record_recent(h: str, topic: str):
    st = _state_load()
    st.setdefault("recent", []).append({"h":h,"topic":topic,"ts":time.time()})
    _state_save(st)
    g = _global_topics_load()
    if topic and topic not in g.get("topics", []):
        g["topics"].append(topic)
        _global_topics_save(g)

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
    # global pool'u da karıştır
    g = _global_topics_load().get("topics", [])
    for t in reversed(g):
        if t and t not in uniq:
            uniq.append(t)
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
    # yasaklı boş cümleler (meta-speech) → temizle
    BAD = {"one clear tip","see it","learn it","in 60 seconds","plot twist","watch:","instrument notes:","key reading"}
    low = t.lower()
    for b in BAD:
        low = low.replace(b, "")
    t = re.sub(r"\s+", " ", low).strip().capitalize()
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
        lines = []; buf, L = [], 0
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
            if getattr(e, "status", None) == 401 or "401" in str(e): break
            time.sleep(0.8)
        except Exception:
            time.sleep(0.8)
    # Fallback: Google TTS
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
    except Exception:
        run(["ffmpeg","-y","-f","lavfi","-t","4.0","-i","anullsrc=r=48000:cl=mono", wav_out])
        return 4.0

# -------------------- Video helpers --------------------
def quantize_to_frames(seconds: float, fps: int = TARGET_FPS) -> Tuple[int, float]:
    frames = max(2, int(round(seconds * fps)))
    return frames, frames / float(fps)

def make_segment(src: str, dur_s: float, outp: str, start_at: float = 0.0):
    """start_at ile ofsetli segment üret; CFR/PTS/trim sabit."""
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
    # Input-level -ss/-t; sonra yine kare kilidi
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-ss", f"{max(0.0, start_at):.3f}", "-t", f"{qdur:.3f}",
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

def draw_capcut_text(seg: str, text: str, color: str, font: str, outp: str, is_hook: bool=False):
    wrapped = wrap_mobile_lines(clean_caption_text(text), CAPTION_MAX_LINE, CAPTION_MAX_LINES)
    tf = str(pathlib.Path(seg).with_suffix(".caption.txt"))
    pathlib.Path(tf).write_text(wrapped, encoding="utf-8")
    seg_dur = ffprobe_dur(seg)
    frames = max(2, int(round(seg_dur * TARGET_FPS)))
    lines = wrapped.split("\n")
    n_lines = max(1, len(lines))
    maxchars = max((len(l) for l in lines), default=1)
    base = 64 if is_hook else 50
    ratio = CAPTION_MAX_LINE / max(1, maxchars)
    fs = int(base * min(1.0, max(0.50, ratio)))
    if n_lines >= 5: fs = int(fs * 0.92)
    if n_lines >= 6: fs = int(fs * 0.88)
    if n_lines >= 7: fs = int(fs * 0.84)
    if n_lines >= 8: fs = int(fs * 0.80)
    fs = max(22, fs)
    if n_lines >= 6:
        y_pos = "(h*0.55 - text_h/2)"
    elif n_lines >= 4:
        y_pos = "(h*0.58 - text_h/2)"
    else:
        y_pos = "h-h/3-text_h/2"
    col = _ff_color(color)
    font_arg = f":fontfile={_ff_sanitize_font(font)}" if font else ""
    common = f"textfile='{tf}':fontsize={fs}:x=(w-text_w)/2:y={y_pos}:line_spacing=10"
    shadow = f"drawtext={common}{font_arg}:fontcolor=black@0.85:borderw=0"
    box    = f"drawtext={common}{font_arg}:fontcolor=white@0.0:box=1:boxborderw={(22 if is_hook else 18)}:boxcolor=black@0.65"
    main   = f"drawtext={common}{font_arg}:fontcolor={col}:borderw={(5 if is_hook else 4)}:bordercolor=black@0.9"
    vf = f"{shadow},{box},{main},fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB,trim=start_frame=0:end_frame={frames}"
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
        pathlib.Path(outp).write_bytes(pathlib.Path(video_in).read_bytes()); return
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

# -------------------- Gemini (topic-first, with visual_hints & hook) --------------------
ENHANCED_GEMINI_TEMPLATES = {
    "_base": """Create a 25–40s YouTube Short for the given channel.
Return STRICT JSON with keys:
- hook_line (<=12 words, concrete, contains a number or contrast)
- sentences (7–8 short lines, each 6–12 words, no meta-language)
- visual_hints (same length as sentences; each a concrete b-roll idea, 2–4 words)
- search_terms (4–10 terms, concrete visual nouns; no adjectives only)
- title (compelling, 40–70 chars), description (>=1200 chars, informative), tags (8–15)
Rules:
- STAY ON TOPIC: exactly match the provided TOPIC. Do NOT pivot to geography/country facts unless TOPIC asks.
- Avoid generic fillers ('in 60 seconds', 'see it', 'learn it', 'plot twist', 'watch:', 'instrument notes:', 'key reading').
- Every sentence should deliver one concrete idea a viewer can visualize.
- Last sentence should callback to the hook (loop feel).
Return ONLY JSON, no markdown/prose.""",
    "country_facts": """Make surprising but true micro-facts about places/culture, same JSON contract."""
}

def _extract_json_block(txt: str) -> dict:
    """
    Gemini bazen JSON dışına metin/işaret koyabiliyor.
    Bu yardımcı, ilk düzgün { ... } bloğunu denge sayacıyla ayıklar.
    """
    # Kod çitlerini temizle
    t = txt.strip()
    t = re.sub(r"^```json\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^```\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    # Eğer doğrudan JSON ise
    try:
        return json.loads(t)
    except Exception:
        pass

    # İlk dengeli { ... } bloğunu yakala
    start = t.find("{")
    if start == -1:
        raise RuntimeError("Gemini response parse error (no opening brace)")

    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = t[start:i+1]
                try:
                    return json.loads(block)
                except Exception as e:
                    # trailing virgüller vb. min temizlik
                    block2 = re.sub(r",\s*}", "}", block)
                    block2 = re.sub(r",\s*\]", "]", block2)
                    return json.loads(block2)
    raise RuntimeError("Gemini response parse error (unbalanced braces)")

def _gemini_call(prompt: str, model: str) -> dict:
    if not GEMINI_API_KEY: raise RuntimeError("GEMINI_API_KEY missing")
    headers = {"Content-Type":"application/json","x-goog-api-key":GEMINI_API_KEY}
    payload = {"contents":[{"parts":[{"text": prompt}]}], "generationConfig":{"temperature":0.7}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        txt = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        txt = json.dumps(data)
    return _extract_json_block(txt)

def build_via_gemini(topic_lock: str, user_terms: List[str], banlist: List[str]) -> Tuple[str,List[str],List[str],List[str],str,str,List[str]]:
    template = ENHANCED_GEMINI_TEMPLATES["_base"]
    if MODE == "country_facts":
        template = ENHANCED_GEMINI_TEMPLATES["country_facts"]
    avoid = "\n".join(f"- {b}" for b in banlist[:15]) if banlist else "(none)"
    terms_hint = ", ".join(user_terms[:8]) if user_terms else "(none)"
    guard = """
Hard constraints:
- Use the given TOPIC exactly.
- Each 'visual_hints[i]' must be directly filmable b-roll (e.g., 'stone arch drone', 'cathedral interior', 'macro gears', 'stormy coast').
- The 'hook_line' must contain a number or contrast ('X vs Y', 'two ways', '3-second test').
"""
    prompt = f"""{template}

Channel: {CHANNEL_NAME}
Language: {LANG}
TOPIC (hard lock): {topic_lock or "Interesting Shorts"}
Seed search terms (use, expand; do not ignore): {terms_hint}
Avoid recently used topics (180 days):
{avoid}
{guard}
"""
    data = _gemini_call(prompt, GEMINI_MODEL)

    hook      = str(data.get("hook_line") or "").strip()
    sentences = [clean_caption_text(s) for s in (data.get("sentences") or []) if str(s).strip()]
    hints     = [str(x).strip() for x in (data.get("visual_hints") or []) if str(x).strip()]

    # En az 7 satır garanti et
    if len(sentences) < 7:
        base = topic_lock or "Quick visual insight"
        fillers = [
            f"{base}: one clean example you can see.",
            "Anchor it to a visual trigger.",
            "Name the pattern in seven words.",
            "Show one counterexample for contrast.",
            "Repeat once in a new context.",
            "Compress to a memorable cue.",
            "Close by calling back to the start."
        ]
        # hook’u da ekleyelim (videonun 1. satırı olacak)
        sentences = [hook or "Two ways you can spot this fast."] + fillers[:7]
    else:
        # Hook'u başa ekle (zaten üstte de böyle kullanıyoruz)
        sentences = [hook or "Two ways you can spot this fast."] + sentences[:7]

    # Hints boş veya kısa ise topic’e göre güvenli varsayılanlar
    default_hints = []
    tl = (topic_lock or "").lower()
    if any(k in tl for k in ["nature","eco","micro","macro"]):
        default_hints = ["leaf macro","dew drops macro","mushroom gills","insect macro","forest moss","stream pebbles","leaf veins"]
    elif any(k in tl for k in ["space","cosmic","planet","astronomy"]):
        default_hints = ["rocket launch","planet surface","star field","telescope","nebula drift","iss window","milky way"]
    elif any(k in tl for k in ["bridge","tunnel","urban","city"]):
        default_hints = ["city bridge drone","stone arch","metro station","aerial skyline","river pier","traffic timelapse","spiral stairs"]
    else:
        default_hints = ["macro gears","timelapse city","close-up hands","notebook topdown","river ripples","window light","walking feet"]

    if len(hints) < 7:
        need = 7 - len(hints)
        hints = hints + default_hints[:need]
    # Hook + 7 sahne → 8 ipucu lazımsa sonuncuyu tekrar et
    while len(hints) < len(sentences):
        hints.append(hints[-1] if hints else default_hints[0])

    terms     = data.get("search_terms") or []
    if isinstance(terms, str): terms=[terms]
    terms = [t.strip() for t in terms if isinstance(t,str) and t.strip()]
    if user_terms:
        pref = [t for t in user_terms if t not in terms]
        terms = (pref + terms)[:12]

    ttl      = (data.get("title") or "").strip()
    desc     = (data.get("description") or "").strip()
    tags     = [t.strip() for t in (data.get("tags") or []) if isinstance(t,str) and t.strip()]
    return hook, sentences[1:], hints, terms, ttl, desc, tags  # sentences[1:] = hook dışındaki 7 satır

# -------------------- Per-scene queries --------------------
_STOP = set("""
a an the and or but if while of to in on at from by with for about into over after before between during under above across around through
this that these those is are was were be been being have has had do does did can could should would may might will shall
you your we our they their he she it its as than then so such very more most many much just also only even still yet
""".split())
_GENERIC_BAD = {"great","good","bad","big","small","old","new","many","more","most","thing","things","stuff","tip","learn","see","watch","notes","reading"}

def _tok4(s: str) -> List[str]:
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", (s or "").lower())
    return [w for w in s.split() if len(w) >= 4 and w not in _STOP and w not in _GENERIC_BAD]

def build_per_scene_queries(sentences: List[str], hints: List[str], fallback_terms: List[str], topic: str) -> List[str]:
    queries=[]
    fb=[]
    for t in (fallback_terms or []):
        t = re.sub(r"[^A-Za-z0-9 ]+"," ", str(t)).strip().lower()
        if not t: continue
        ws = [w for w in t.split() if w not in _STOP and w not in _GENERIC_BAD]
        if ws:
            fb.append(" ".join(ws[:3]))
    topic_keys = _tok4(topic)[:2]
    topic_key_join = " ".join(topic_keys) if topic_keys else ""

    for i, s in enumerate(sentences):
        picked=None
        # 1) visual hint öncelikli
        if i < len(hints):
            cand = hints[i].lower().strip()
            cand = re.sub(r"[^a-z0-9 ]+"," ", cand)
            if len(cand) >= 4: picked = " ".join(cand.split()[:3])
        # 2) cümleden 2-gram
        if not picked:
            toks = _tok4(s)
            if len(toks) >= 2:
                picked = f"{toks[0]} {toks[1]}"
            elif len(toks) == 1:
                picked = toks[0]
        # 3) SEARCH_TERMS rotasyonu
        if (not picked or len(picked) < 4) and fb:
            picked = fb[i % len(fb)]
        # 4) topic fallback
        if (not picked or len(picked) < 4) and topic_key_join:
            picked = topic_key_join
        # 5) son emniyet
        if not picked:
            picked = "macro detail"
        queries.append(picked)
    return queries

# -------------------- Pexels search (relaxed second pass) --------------------
_USED_PEXELS_IDS_RUNTIME = set()

def _pexels_headers():
    if not PEXELS_API_KEY: raise RuntimeError("PEXELS_API_KEY missing")
    return {"Authorization": PEXELS_API_KEY}

def _pexels_locale(lang: str) -> str:
    return "tr-TR" if lang.startswith("tr") else "en-US"

def _pexels_search(query: str, portrait_only=True, page=1, per_page=30):
    headers = _pexels_headers()
    locale  = _pexels_locale(LANG)
    params = {"query": query, "per_page": per_page, "page": page, "locale": locale}
    if portrait_only:
        params.update({"orientation":"portrait", "size":"large"})
    r = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=30)
    if r.status_code != 200: return {}
    return r.json() or {}

def pexels_pick_one(query: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        # 1. geçiş: portrait
        for portrait in (True, False):
            data = _pexels_search(query, portrait_only=portrait, page=1, per_page=30)
            cand = []
            block = _blocklist_get_pexels()
            for v in data.get("videos", []) or []:
                vid = int(v.get("id", 0))
                if vid in block or vid in _USED_PEXELS_IDS_RUNTIME: continue
                files = v.get("video_files", []) or []
                if not files: continue
                pf = [x for x in files if int(x.get("height",0)) >= int(x.get("width",0))] if portrait else files
                if not pf: continue
                pf.sort(key=lambda x: (abs(int(x.get("height",0))-1440) if portrait else 0, int(x.get("height",0))*int(x.get("width",0))))
                best = pf[0]
                h = int(best.get("height",0)); w = int(best.get("width",0))
                if h < 720: continue
                dur = float(v.get("duration",0))
                dur_bonus = 1.0 if 2.0 <= dur <= 15.0 else 0.0
                tokens = set(re.findall(r"[a-z0-9]+", (v.get("url") or "").lower()))
                qtokens= set(re.findall(r"[a-z0-9]+", query.lower()))
                overlap = len(tokens & qtokens)
                score = overlap*2.0 + dur_bonus + (1.0 if 1080 <= h else 0.0)
                cand.append((score, vid, best.get("link")))
            if cand:
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

# -------------------- Description builder (>=2000 chars SEO) --------------------
def _mk_hashtags(topic: str, terms: List[str]) -> List[str]:
    base = set()
    base.update([re.sub(r"[^a-z0-9]+","", k.lower()) for k in topic.split() if k])
    for t in terms[:10]:
        for k in re.findall(r"[a-z0-9]+", t.lower()):
            base.add(k)
    tags = []
    for k in base:
        if 2 < len(k) < 20:
            tags.append("#"+k[:20])
    # sabitler
    fixed = ["#shorts","#learn","#facts","#howitworks","#visual"]
    for f in fixed:
        if f not in tags: tags.append(f)
    # kanal etiketi
    ch = "#"+re.sub(r"[^a-z0-9]+","", CHANNEL_NAME.lower())
    if ch not in tags: tags.append(ch[:25])
    return tags[:15]

def _desc_boost(topic: str, sentences: List[str], terms: List[str]) -> str:
    # 2000+ karakterlik, alt başlıklı açıklama
    bullets = "\n".join([f"• {s}" for s in sentences])
    tags = " ".join(_mk_hashtags(topic, terms))
    guide = (
f"""{topic} — quick, visual explanation with concrete examples.

WHAT YOU'LL GET
{bullets}

WHY IT MATTERS
• Clear mental models you can use today.
• Visual anchors that help you remember fast.
• Short, specific steps — no fluff.

HOW TO APPLY
• Try one idea right after watching.
• Pause, replay the key beat, then teach it to a friend.
• Save to your ‘Practice’ playlist for spaced repetition.

EXTRA CONTEXT
We aim for a tight 25–40 seconds format: one idea per scene, crisp captions, pattern interrupts every ~2–3s to keep your attention fresh. Visuals are stock b-roll aligned to each beat (macro, drone, timelapse, close-ups) so you can *see* the insight, not just hear it.

CREDITS & FAIR USE
This short uses licensed stock b-roll for educational commentary and transformative explanation.

{tags}
""")
    # 2000+ char pad (bilgi tekrarı değil; uygulama örnekleri)
    if len(guide) < 2000:
        extras = []
        for i, s in enumerate(sentences[:8], 1):
            extras.append(f"- Example {i}: {s} — try to spot it in real life today.")
        guide += "\n\nAPPLY IT IN REAL LIFE\n" + "\n".join(extras)
    return guide[:4900]

# -------------------- Main --------------------
def main():
    print(f"==> {CHANNEL_NAME} | MODE={MODE} | topic-first build")
    random.seed(ROTATION_SEED or int(time.time()))

    topic_lock = TOPIC
    user_terms = SEARCH_TERMS_ENV

    # 1) İçerik
    if USE_GEMINI and GEMINI_API_KEY:
        banlist = _recent_topics_for_prompt()
        chosen, last = None, None
        for _ in range(6):
            try:
                hook, sents, hints, search_terms, ttl, desc, tags = build_via_gemini(topic_lock, user_terms, banlist)
                last = (hook, sents, hints, search_terms, ttl, desc, tags)
                sig = f"{topic_lock}|{sents[0] if sents else ''}"
                h = _hash12(sig)
                if not _is_recent(h, window_days=180):
                    _record_recent(h, topic_lock or "topic")
                    chosen = last; break
                else:
                    banlist.insert(0, topic_lock or "topic")
                    time.sleep(0.5)
            except Exception as e:
                print(f"Gemini error: {str(e)[:160]}"); time.sleep(0.6)
        if chosen is None:
            hook, sents, hints, search_terms, ttl, desc, tags = last if last else (
                "Two ways to spot balance fast.",
                [
                    "Start with a simple contrast you can see.",
                    "Pick one crisp example you meet today.",
                    "Name the trigger so it sticks later.",
                    "Show one counterexample to sharpen edges.",
                    "Repeat the pattern once in a new context.",
                    "Compress it to seven words you can recall.",
                    "Close by hinting back to the opening beat."
                ],
                ["macro gears","stone arch","city bridge drone","office top down","river pier","spiral stairs","close-up hands"],
                user_terms or ["macro detail","timelapse city","drone bridge"],
                "", "", []
            )
    else:
        hook = "Two ways to spot balance fast."
        sents = [
            "Start with a simple contrast you can see.",
            "Pick one crisp example you meet today.",
            "Name the trigger so it sticks later.",
            "Show one counterexample to sharpen edges.",
            "Repeat the pattern once in a new context.",
            "Compress it to seven words you can recall.",
            "Close by hinting back to the opening beat."
        ]
        hints = ["macro gears","stone arch","city bridge drone","office top down","river pier","spiral stairs","close-up hands"]
        search_terms = user_terms or ["macro detail","timelapse city","drone bridge"]
        ttl, desc, tags = "", "", []

    sentences = [hook] + sents  # hook'u 0. sahneye koyuyoruz
    print(f"📝 Content: {topic_lock or 'Topic'} | {len(sentences)} lines (hook+{len(sents)})")

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

    # 3) Pexels — sahne başına net sorgular
    # hints güvenli doldurma (en az len(sentences))
    hints_for_queries = list(hints) if hints else []
    while len(hints_for_queries) < len(sentences):
        hints_for_queries.append(hints_for_queries[-1] if hints_for_queries else "macro detail")

    # build_per_scene_queries imzan şu şekilde olmalı:
    # build_per_scene_queries(sentences, hints_or_terms, fallback_terms, topic)
    per_scene_queries = build_per_scene_queries(
        sentences,
        hints_for_queries,
        (search_terms or user_terms or []),
        topic_lock or "Interesting Shorts",
    )

    print("🔎 Per-scene queries:")
    for q in per_scene_queries:
        print(f"   • {q}")

    picked = []
    for q in per_scene_queries:
        vid, link = pexels_pick_one(q)
        if vid and link:
            picked.append((vid, link))

    if not picked:
        raise RuntimeError("Pexels: no results (per-scene).")

    # indir
    clips = []
    for idx, (vid, link) in enumerate(picked):
        try:
            f = str(pathlib.Path(tmp) / f"clip_{idx:02d}_{vid}.mp4")
            with requests.get(link, stream=True, timeout=120) as rr:
                rr.raise_for_status()
                with open(f, "wb") as w:
                    for ch in rr.iter_content(8192):
                        w.write(ch)
            if pathlib.Path(f).stat().st_size > 300_000:
                clips.append(f)
        except Exception as e:
            print(f"⚠️ download fail ({vid}): {e}")

    if len(clips) < len(sentences):
        print("⚠️ Not enough unique clips; rotating existing ones.")
        # yine de döndür, ama ilk klibi ayrıca loop için kullanacağız

    # 4) Segment + altyazı + LOOP HİLESİ
    print("🎬 Segments…")
    segs = []
    # loop için ilk sahnenin kaynağı
    first_clip_src = clips[0 % len(clips)]
    first_audio_text, first_audio_dur = metas[0]
    first_clip_total = ffprobe_dur(first_clip_src)
    # güvenli ofset (klip yarısı - 0.25s)
    half = max(0.0, first_clip_total/2.0 - 0.25)

    for i, (base_text, d) in enumerate(metas):
        # i==0 → aynı ilk klibin İKİNCİ yarısından başlat
        if i == 0:
            src = first_clip_src
            start_at = half
        # son sahne → aynı ilk klibin İLK yarısından başlat (loop hissi)
        elif i == len(metas)-1:
            src = first_clip_src
            start_at = 0.0
        else:
            src = clips[i % len(clips)]
            start_at = 0.0

        base   = str(pathlib.Path(tmp) / f"seg_{i:02d}.mp4")
        make_segment(src, d, base, start_at=start_at)

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

    # 5) Birleştir
    print("🎞️ Assemble…")
    vcat = str(pathlib.Path(tmp) / "video_concat.mp4"); concat_videos_filter(segs, vcat)
    acat = str(pathlib.Path(tmp) / "audio_concat.wav"); concat_audios(wavs, acat)

    # 6) Süre & kare kilitleme
    adur = ffprobe_dur(acat); vdur = ffprobe_dur(vcat)
    if vdur + 0.02 < adur:
        vcat_padded = str(pathlib.Path(tmp) / "video_padded.mp4")
        pad_video_to_duration(vcat, adur, vcat_padded)
        vcat = vcat_padded
        vdur = ffprobe_dur(vcat)
    a_frames = max(2, int(round(adur * TARGET_FPS)))
    vcat_exact = str(pathlib.Path(tmp) / "video_exact.mp4")
    enforce_video_exact_frames(vcat, a_frames, vcat_exact); vcat = vcat_exact
    acat_exact = str(pathlib.Path(tmp) / "audio_exact.wav")
    lock_audio_duration(acat, a_frames, acat_exact); acat = acat_exact
    vdur2 = ffprobe_dur(vcat); adur2 = ffprobe_dur(acat)
    print(f"🔒 Locked A/V: video={vdur2:.3f}s | audio={adur2:.3f}s | fps={TARGET_FPS}")

    # 7) Mux
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^A-Za-z0-9]+', '_', (TOPIC or "Short"))[:60] or "Short"
    outp = f"{OUT_DIR}/{safe_topic}_{ts}.mp4"
    print("🔄 Mux…")
    mux(vcat, acat, outp)
    final = ffprobe_dur(outp)
    print(f"✅ Saved: {outp} ({final:.2f}s)")

    # 8) Metadata (title/desc strong)
    def _ok(x): return isinstance(x, str) and x.strip()
    if _ok(ttl):
        title = ttl[:95]
    else:
        # sayı/kontrastlı hook’tan başlık
        title = (hook if _ok(hook) else sentences[0])[:95]
    if _ok(desc) and len(desc) >= 1000:
        description = desc[:4900]
    else:
        description = _desc_boost(TOPIC or "Shorts", sentences, search_terms)
    meta = {
        "title": title,
        "description": description,
        "tags": (tags[:15] if tags else _mk_hashtags(TOPIC or "Shorts", search_terms)),
        "privacy": VISIBILITY,
        "defaultLanguage": LANG,
        "defaultAudioLanguage": LANG
    }

    # 9) Upload
    try:
        if UPLOAD_TO_YT:
            print("📤 Uploading to YouTube…")
            vid_id = upload_youtube(outp, meta)
            print(f"🎉 YouTube Video ID: {vid_id}\n🔗 https://youtube.com/watch?v={vid_id}")
        else:
            print("⏭️ Upload skipped by UPLOAD_TO_YT=0")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

    # 10) Temizlik
    try:
        shutil.rmtree(tmp); print("🧹 Cleaned temp files")
    except: pass

if __name__ == "__main__":
    main()


