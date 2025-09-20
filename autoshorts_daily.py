# autoshorts_daily.py - Stabil Versiyon (video-özel search terms, sağlam ffmpeg zinciri)
# -*- coding: utf-8 -*-
import os, sys, re, json, time, uuid, random, datetime, tempfile, pathlib, subprocess, hashlib, math
from typing import List, Optional

VOICE_STYLE = os.getenv("TTS_STYLE", "narration-professional")
TARGET_MIN_SEC = float(os.getenv("TARGET_MIN_SEC", "22"))
TARGET_MAX_SEC = float(os.getenv("TARGET_MAX_SEC", "42"))

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
    _pip("google-api-python-client"); from googleapiclient.discovery import build; from googleapiclient.http import MediaFileUpload
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
except ImportError:
    _pip("google-auth"); from google.oauth2.credentials import Credentials; from google.auth.transport.requests import Request

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
GEMINI_PROMPT  = os.getenv("GEMINI_PROMPT", "").strip() or None

VOICE_OPTIONS = {
    "en": ["en-US-JennyNeural","en-US-JasonNeural","en-US-AriaNeural","en-US-GuyNeural","en-AU-NatashaNeural","en-GB-SoniaNeural","en-CA-LiamNeural","en-US-DavisNeural","en-US-AmberNeural"],
    "tr": ["tr-TR-EmelNeural","tr-TR-AhmetNeural"]
}
VOICE = os.getenv("TTS_VOICE", VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])[0])
VOICE_RATE = os.getenv("TTS_RATE", "+10%")

TARGET_FPS     = 25
CRF_VISUAL     = 22
CAPTION_COLORS = ["#FFD700","#FF6B35","#00F5FF","#32CD32","#FF1493","#1E90FF","#FFA500","#FF69B4"]
CAPTION_MAX_LINE = 22

STATE_FILE = f"state_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json"

# ---- America/New_York günü için tek-kez kilidi ----
try:
    from zoneinfo import ZoneInfo
    TZ_NY = ZoneInfo("America/New_York")
except Exception:
    TZ_NY = None
def _now_et():
    return datetime.datetime.now(TZ_NY) if TZ_NY else datetime.datetime.utcnow()
def _daily_lock_et():
    st = _state_load(); today = _now_et().strftime("%Y-%m-%d")
    if st.get("last_date_et") == today:
        print("🔒 Already ran once today (America/New_York). Skipping."); sys.exit(0)
    st["last_date_et"] = today; _state_save(st)

# ---------------- TTS SSML ----------------
def create_advanced_ssml(text: str, voice: str) -> str:
    optimized = text
    nums = {'1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine','10':'ten',
            '11':'eleven','12':'twelve','13':'thirteen','14':'fourteen','15':'fifteen','16':'sixteen','17':'seventeen',
            '18':'eighteen','19':'nineteen','20':'twenty'}
    for n,w in nums.items(): optimized = re.sub(rf'\b{n}\b', w, optimized)
    optimized = re.sub(r'\.(?=\s)', '.<break time="600ms"/>', optimized)
    optimized = re.sub(r',(?=\s)', ',<break time="350ms"/>', optimized)
    optimized = re.sub(r'\?(?=\s)', '?<break time="700ms"/>', optimized)
    optimized = re.sub(r'!(?=\s)', '!<break time="500ms"/>', optimized)
    ssml = f"""
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  <voice name="{voice}">
    <prosody rate="+10%" pitch="+0Hz" volume="medium">
      {optimized}
    </prosody>
  </voice>
</speak>""".strip()
    return ssml

def tts_to_wav(text: str, wav_out: str) -> float:
    import asyncio
    def _run_ff(args): subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y", *args], check=True)
    def _probe(path: str, default: float = 3.5) -> float:
        try:
            pr = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1", path],
                                capture_output=True, text=True, check=True)
            return float(pr.stdout.strip())
        except Exception:
            return default
    mp3 = wav_out.replace(".wav", ".mp3")
    clean = text[:400] if len(text) > 400 else text
    for abbr, full in {"AI":"Artificial Intelligence","USA":"United States","UK":"United Kingdom","NASA":"NASA","DIY":"Do It Yourself"}.items():
        clean = re.sub(rf'\b{abbr}\b', full, clean)
    available = VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])
    selected = VOICE if VOICE in available else available[0]
    print(f"      🎤 Voice: {selected} | {clean[:30]}...")
    try:
        async def _edge_save_premium():
            ssml_text = create_advanced_ssml(clean, selected)
            comm = edge_tts.Communicate(ssml_text, voice=selected)
            await comm.save(mp3)
        try:
            asyncio.run(_edge_save_premium())
        except RuntimeError:
            nest_asyncio.apply(); asyncio.get_event_loop().run_until_complete(_edge_save_premium())
        # —— Güvenli filtre zinciri: de-esser YOK (sürüm farkı çakıyordu)
        _run_ff([
            "-i", mp3, "-ar","48000","-ac","1","-acodec","pcm_s16le",
            "-af","volume=0.92,highpass=f=75,lowpass=f=15000,dynaudnorm=g=7:f=250:r=0.95,acompressor=threshold=-20dB:ratio=2:attack=5:release=50,equalizer=f=2000:t=h:w=200:g=2,equalizer=f=100:t=h:w=50:g=1",
            wav_out
        ])
        pathlib.Path(mp3).unlink(missing_ok=True)
        d = _probe(wav_out, 3.5); print(f"      ✅ Premium Edge-TTS: {d:.1f}s"); return d
    except Exception as e:
        print(f"      ⚠️ SSML başarısız, basit Edge-TTS: {e}")
        try:
            async def _edge_save_simple():
                comm = edge_tts.Communicate(clean, voice=selected, rate=VOICE_RATE)
                await comm.save(mp3)
            try:
                asyncio.run(_edge_save_simple())
            except RuntimeError:
                nest_asyncio.apply(); asyncio.get_event_loop().run_until_complete(_edge_save_simple())
            _run_ff([
                "-i", mp3, "-ar","44100","-ac","1","-acodec","pcm_s16le",
                "-af","volume=0.9,dynaudnorm=g=5:f=200,acompressor=threshold=-18dB:ratio=2.5:attack=5:release=15",
                wav_out
            ])
            pathlib.Path(mp3).unlink(missing_ok=True)
            d = _probe(wav_out, 3.5); print(f"      ✅ Simple Edge-TTS: {d:.1f}s"); return d
        except Exception as e2:
            print(f"      ⚠️ Edge-TTS ve filtre fallback: {e2}")
            try:
                q = requests.utils.quote(clean.replace('"','').replace("'",""))
                url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={q}&tl={LANG or 'en'}&client=tw-ob&ttsspeed=0.85"
                r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=30); r.raise_for_status()
                open(mp3,"wb").write(r.content)
                _run_ff(["-i", mp3, "-ar","44100","-ac","1","-acodec","pcm_s16le","-af","volume=0.9,dynaudnorm", wav_out])
                pathlib.Path(mp3).unlink(missing_ok=True)
                d = _probe(wav_out, 3.5); print(f"      ✅ Google TTS: {d:.1f}s"); return d
            except Exception as e3:
                print(f"      ❌ Tüm TTS başarısız: {e3}"); _run_ff(["-f","lavfi","-t","4.0","-i","anullsrc=r=44100:cl=mono", wav_out]); return 4.0

# ---------------- video-özel arama terimleri ----------------
_STOPWORDS_EN = {
    "the","a","an","and","or","but","of","in","on","to","for","with","by","from","as","at",
    "is","are","was","were","be","been","being","has","have","had","do","does","did",
    "this","that","these","those","it","its","you","your","we","our","they","their","i","me",
    "today","now","then","there","here","very","more","most","over","under","into","out","about","just","also",
    "once","still","even","literally","completely","nearby","therefore","however"
}
def _normalize_words(text: str, lang: str="en") -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text or ""); words = [w.lower() for w in words]
    if lang.startswith("en"): words = [w for w in words if w not in _STOPWORDS_EN]
    return words
def _top_keywords(sentences: List[str], lang: str="en", k: int=8) -> List[str]:
    from collections import Counter
    weights = Counter()
    for idx, s in enumerate(sentences):
        for token in _normalize_words(s, lang):
            weights[token] += (3 if idx == 0 else 2 if idx == 1 else 1)
    bigrams = Counter()
    for s in sentences:
        w = _normalize_words(s, lang)
        for a, b in zip(w, w[1:]):
            if a != b: bigrams[f"{a} {b}"] += 1
    out = [t for t,_ in weights.most_common(k+4)]
    out = (list(dict.fromkeys([bg for bg,_ in bigrams.most_common(6)] + out)))
    return out[:k+4]
def build_search_terms_per_video(sentences: List[str], topic: str, mode: str, lang: str,
                                 hints: Optional[List[str]] = None, max_terms: int = 8) -> List[str]:
    keywords = _top_keywords(sentences + [topic], lang=lang, k=8)
    domain_boosters = {
        "space_news": ["space","rocket launch","telescope","galaxy","nasa","astronaut"],
        "tech_news": ["technology","lab","robot","ai","microchip","innovation"],
        "history_story": ["ancient","ruins","manuscript","archaeology","museum"],
        "country_facts": ["landmark","city skyline","culture","nature","travel"],
        "animal_facts": ["wildlife","close up","nature","habitat"],
        "movie_secrets": ["film set","behind the scenes","cinema","director"],
        "cricket_women": ["women cricket","stadium","crowd","celebration"],
        "fixit_fast": ["workshop","tools","repair","step by step"]
    }
    boosters = domain_boosters.get(mode, [])
    hints = [re.sub(r"\s+"," ", h.strip().lower()) for h in (hints or []) if isinstance(h,str) and h.strip()]
    candidates: List[str] = []
    for kw in keywords:
        if " " in kw: candidates.append(kw)
        else:
            for bx in boosters[:3] or ["highlight"]: candidates.append(f"{kw} {bx}")
            candidates.append(kw)
    for h in hints[:6]:
        if h not in candidates: candidates.append(h)
    decorated: List[str] = []
    for c in candidates:
        base = c.strip()
        if not base: continue
        if " " not in base and boosters: base = f"{base} {boosters[0]}"
        decorated.extend([f"{base} 4k", f"{base} portrait", f"{base} vertical 4k"])
    unique: List[str] = []
    for t in decorated:
        t = re.sub(r"\s+"," ", t).strip()
        if t not in unique: unique.append(t)
    bigram_first = [t for t in unique if " " in t]
    final = (bigram_first + [t for t in unique if t not in bigram_first])[:max_terms]
    final = [t for t in final if len(t.split()) >= 2] or final[:4]
    return final[:max_terms]

# ---------------- Pexels ----------------
def pexels_download(terms: List[str], need: int, tmp: str) -> List[str]:
    if not PEXELS_API_KEY: raise RuntimeError("PEXELS_API_KEY missing")
    out, seen = [], set(); headers = {"Authorization": PEXELS_API_KEY}
    def _score(meta):
        w,h = meta.get("width",0), meta.get("height",0); dur = meta.get("duration",0); size = meta.get("size",0) or 0
        aspect = (h>0 and abs((9/16)-(w/h))<0.08); dur_bonus = -abs(dur-9)+9
        return (2000*(1 if aspect else 0)) + (w*h/1e4) + (size/2e5) + (120*max(dur_bonus,0))
    for term in terms:
        if len(out) >= need: break
        try:
            r = requests.get("https://api.pexels.com/videos/search", headers=headers,
                             params={"query":term,"per_page":30,"orientation":"portrait","size":"large","min_width":"1080","min_height":"1920"}, timeout=30)
            vids = r.json().get("videos", []) if r.status_code==200 else []
            cand = []
            for v in vids:
                files = v.get("video_files",[]); if not files: continue
                best = max(files, key=lambda x: x.get("width",0)*x.get("height",0))
                if best.get("height",0) < 1080: continue
                url = best.get("link"); if not url or url in seen: continue
                meta = {"width": best.get("width",0),"height": best.get("height",0),"duration": v.get("duration",0),
                        "size": best.get("file_size",0) or 0,"url": url}
                cand.append((_score(meta), meta))
            for _,m in sorted(cand, key=lambda x: x[0], reverse=True):
                if len(out) >= need: break
                url = m["url"]; seen.add(url)
                fpath = str(pathlib.Path(tmp)/f"clip_{len(out):02d}_{uuid.uuid4().hex[:6]}.mp4")
                with requests.get(url, stream=True, timeout=180) as rr:
                    rr.raise_for_status()
                    with open(fpath,"wb") as w:
                        for ch in rr.iter_content(1<<14): w.write(ch)
                ok_size = pathlib.Path(fpath).stat().st_size > 1_200_000
                ok_dur  = 4 <= m["duration"] <= 20
                if ok_size and ok_dur: out.append(fpath)
        except Exception: continue
    if len(out) < max(3, need//2): raise RuntimeError("Yeterli kaliteli Pexels video bulunamadı")
    return out[:need]

# ---------------- Video işleme ----------------
def make_segment(src: str, dur: float, outp: str):
    dur = max(0.8, min(dur, 5.0)); fade = max(0.05, min(0.12, dur/8))
    print(f"      📹 Segment: {dur:.1f}s (max 5s)")
    vf = ("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
          "eq=brightness=0.02:contrast=1.08:saturation=1.1,"
          f"fade=t=in:st=0:d={fade:.2f},fade=t=out:st={max(0.0,dur-fade):.2f}:d={fade:.2f}")
    run(["ffmpeg","-y","-i",src,"-t",f"{dur:.3f}","-vf",vf,"-r","25","-an","-c:v","libx264","-preset","fast","-crf","22","-pix_fmt","yuv420p", outp])

def draw_capcut_text(seg: str, text: str, color: str, font: str, outp: str, is_hook: bool=False):
    wrapped = wrap_mobile_lines(clean_caption_text(text), CAPTION_MAX_LINE); esc = escape_drawtext(wrapped)
    lines = wrapped.count("\n")+1; maxchars = max(len(x) for x in wrapped.split("\n"))
    if is_hook: base_fs, border_w, box_border = (52 if lines >= 3 else 58), 5, 20
    else:       base_fs, border_w, box_border = (42 if lines >= 3 else 48), 4, 16
    if maxchars > 25: base_fs -= 6
    elif maxchars > 20: base_fs -= 3
    y_pos = "h-h/3-text_h/2"; common = f"text='{esc}':fontsize={base_fs}:x=(w-text_w)/2:y={y_pos}:line_spacing=10"
    shadow = f"drawtext={common}:fontcolor=black@0.8:borderw=0"
    box    = f"drawtext={common}:fontcolor=white@0.0:box=1:boxborderw={box_border}:boxcolor=black@0.65"
    main   = f"drawtext={common}:fontcolor={color}:borderw={border_w}:bordercolor=black@0.9"
    if font:
        fp = font.replace(":","\\:").replace(",","\\,").replace("\\","/"); shadow += f":fontfile={fp}"; box += f":fontfile={fp}"; main += f":fontfile={fp}"
    vf = f"{shadow},{box},{main}"
    run(["ffmpeg","-y","-i",seg,"-vf",vf,"-c:v","libx264","-preset","medium","-crf",str(max(16,CRF_VISUAL-3)),"-movflags","+faststart", outp])

def concat_videos(files: List[str], outp: str):
    lst = str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst,"w") as f:
        for p in files: f.write(f"file '{p}'\n")
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",lst,"-c","copy", outp])

def concat_audios(files: List[str], outp: str):
    total = sum(ffprobe_dur(f) for f in files); print(f"📊 Audio toplam süre (FULL): {total:.1f}s")
    lst = str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst,"w") as f:
        for p in files: f.write(f"file '{p}'\n")
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",lst,"-af","volume=0.9,dynaudnorm", outp])

def mux(video: str, audio: str, outp: str):
    try:
        vd, ad = ffprobe_dur(video), ffprobe_dur(audio)
        print(f"🔍 Video: {vd:.1f}s | Audio: {ad:.1f}s")
        if abs(vd - ad) > 1.0:
            print("⚠️ Süre uyumsuzluğu düzeltme...")
            min_dur = min(vd, ad, 45.0)
            tv = video.replace(".mp4","_t.mp4"); ta = audio.replace(".wav","_t.wav")
            run(["ffmpeg","-y","-i",video,"-t",f"{min_dur:.2f}","-c","copy",tv])
            run(["ffmpeg","-y","-i",audio,"-t",f"{min_dur:.2f}","-c","copy",ta])
            video, audio = tv, ta
        run(["ffmpeg","-y","-i",video,"-i",audio,"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","256k","-movflags","+faststart","-shortest","-avoid_negative_ts","make_zero", outp])
        for t in [video, audio]:
            if t.endswith("_t.mp4") or t.endswith("_t.wav"): pathlib.Path(t).unlink(missing_ok=True)
    except Exception as e:
        print(f"⚠️ Mux hatası: {e}"); run(["ffmpeg","-y","-i",video,"-i",audio,"-c","copy","-shortest", outp])

# ---------------- İçerik üretimi ----------------
ENHANCED_SCRIPT_BANK = {
    "Turkey": {
        "topic": "Mind-Blowing Turkey Secrets",
        "sentences": [
            "Turkey has a city that's literally underground.",
            "Derinkuyu goes down 18 floors beneath the earth.",
            "Ancient people carved it from solid volcanic rock.",
            "Twenty thousand people once lived there completely hidden.",
            "The tunnels connect to other underground cities nearby.",
            "They had ventilation systems that still work today.",
            "Each floor had kitchens, chapels, and even wine cellars.",
            "This underground world is older than the Roman Empire.",
            "Which underground mystery fascinates you most?"
        ],
        "search_terms": ["underground city 4k","ancient tunnels portrait","turkey caves 4k","historical ruins","cappadocia underground 4k"]
    }
}

ENHANCED_GEMINI_TEMPLATES = {
    "_default": """Create a 25-40s YouTube Short.
EXACTLY 7-8 sentences. Each 6-12 words. Simple but informative.
Return JSON with: country, topic, sentences, search_terms, title, description, tags."""
}

def _validate_soft(sentences: List[str], description: str) -> None:
    # Sıkı hata yerine uyarı ve otomatik doldurma
    if len(sentences) < 7:
        pad = ["This detail adds important context.", "What do you think about it?"]
        sentences.extend(pad[:8-len(sentences)])
    if len(sentences) > 8:
        del sentences[8:]
    words = len(re.findall(r"\w+", " ".join(sentences)))
    if words < 45:  # önceden 70 idi – fazla katıydı
        # ilk cümleleri genişletmeye çalış
        for i in range(min(3, len(sentences))):
            if len(sentences[i].split()) < 8:
                sentences[i] = sentences[i] + " This part explains why it matters."
        # yine de düşükse bırak; video yine üretilecek

def _hash12(s: str) -> str: return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
def _is_recent(h: str, window_days=365) -> bool:
    now = time.time()
    for r in _state_load().get("recent", []):
        if r.get("h")==h and (now - r.get("ts",0)) < window_days*86400: return True
    return False
def _record_recent(h: str, mode: str, topic: str):
    st = _state_load(); st.setdefault("recent", []).append({"h":h,"mode":mode,"topic":topic,"ts":time.time()}); _state_save(st)
def _recent_topics_for_prompt(limit=20) -> List[str]:
    st = _state_load(); topics = [r.get("topic","") for r in reversed(st.get("recent", [])) if r.get("topic")]
    uniq=[]; 
    for t in topics:
        if t and t not in uniq: uniq.append(t)
        if len(uniq) >= limit: break
    return uniq

def _gemini_call(prompt: str, model: str) -> dict:
    if not GEMINI_API_KEY: raise RuntimeError("GEMINI_API_KEY missing")
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    payload = {"contents":[{"parts":[{"text": prompt}]}], "generationConfig":{"temperature":0.8}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200: raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    txt = data.get("candidates",[{}])[0].get("content",{}).get("parts",[{"text":""}])[0].get("text","")
    m = re.search(r"\{(?:.|\n)*\}", txt)
    if not m: raise RuntimeError("Gemini response parse error (no JSON)")
    raw = re.sub(r"^```json\s*|\s*```$", "", m.group(0).strip(), flags=re.MULTILINE)
    return json.loads(raw)

def build_via_gemini(mode: str, channel_name: str, banlist: List[str], channel_config: dict = {}) -> tuple:
    template = ENHANCED_GEMINI_TEMPLATES.get(mode, ENHANCED_GEMINI_TEMPLATES["_default"])
    avoid = "\n".join(f"- {b}" for b in banlist[:15]) if banlist else "(none)"
    prompt = f"""{template}

Channel: {channel_name}
Language: {LANG}
Avoid recent topics:
{avoid}

Return ONLY valid JSON with keys described above."""
    data = _gemini_call(prompt, GEMINI_MODEL)
    country = str(data.get("country") or "World").strip()
    topic = str(data.get("topic") or "Amazing Facts").strip()
    sentences = [clean_caption_text(s) for s in (data.get("sentences") or []) if s]
    _validate_soft(sentences, data.get("description") or "")
    terms = data.get("search_terms") or []
    if isinstance(terms, str): terms = [terms]
    terms = [t.strip() for t in terms if isinstance(t, str) and t.strip()]
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    tags = [t.strip() for t in (data.get("tags") or []) if isinstance(t,str) and t.strip()]
    print(f"✅ Gemini içerik: {len(sentences)} cümle - {topic}")
    return country, topic, sentences, terms, title, description, tags

# ---------------- main ----------------
def main():
    print(f"==> {CHANNEL_NAME} | MODE={MODE} | Enhanced Version")
    _daily_lock_et()

    if USE_GEMINI and GEMINI_API_KEY:
        banlist = _recent_topics_for_prompt()
        MAX_TRIES = 6; chosen = None; last = None
        for attempt in range(MAX_TRIES):
            try:
                print(f"İçerik üretimi denemesi {attempt+1}/{MAX_TRIES}")
                ctry, tpc, sents, terms, ttl, desc, tags = build_via_gemini(MODE, CHANNEL_NAME, banlist)
                last = (ctry, tpc, sents, terms, ttl, desc, tags)
                sig = f"{MODE}|{tpc}|{sents[0] if sents else ''}"
                h = _hash12(sig)
                if not _is_recent(h, window_days=180):
                    _record_recent(h, MODE, tpc); chosen = last; print(f"✅ Benzersiz içerik: {tpc}"); break
                else:
                    banlist.insert(0, tpc); print("⚠️ Benzer içerik, yeniden dene..."); time.sleep(1.5)
            except Exception as e:
                print(f"⚠️ Gemini denemesi başarısız: {str(e)[:200]}"); time.sleep(1.5)
        if chosen is None:
            if last is not None: print("Son sonucu kullanıyoruz..."); ctry, tpc, sents, terms, ttl, desc, tags = last
            else:
                print("Gelişmiş fallback içeriği..."); fb = ENHANCED_SCRIPT_BANK["Turkey"]
                ctry, tpc, sents, terms = "World", fb["topic"], fb["sentences"], fb["search_terms"]; ttl = desc = ""; tags = []
    else:
        print("Gemini devre dışı, gelişmiş fallback..."); fb = ENHANCED_SCRIPT_BANK["Turkey"]
        ctry, tpc, sents, terms = "World", fb["topic"], fb["sentences"], fb["search_terms"]; ttl = desc = ""; tags = []

    print(f"📝 İçerik: {ctry} | {tpc}"); print(f"📊 Cümle sayısı: {len(sents)}")
    sentences = sents

    search_terms = build_search_terms_per_video(sentences=sentences, topic=tpc, mode=MODE, lang=LANG, hints=terms)
    print("🔎 Search terms (per-video):", ", ".join(search_terms[:8]))

    tmp = tempfile.mkdtemp(prefix="enhanced_shorts_"); font = font_path()
    wavs=[]; metas=[]
    print("🎤 Geliştirilmiş TTS işlemi başlıyor...")
    for i, s in enumerate(sentences):
        print(f"   Cümle {i+1}/{len(sentences)}: {s[:50]}...")
        w = str(pathlib.Path(tmp)/f"sent_{i:02d}.wav"); dur = tts_to_wav(s, w)
        wavs.append(w); metas.append((s,dur)); time.sleep(0.2)

    print("🎬 Yüksek kaliteli video indiriliyor...")
    clips = pexels_download(search_terms, need=len(sentences), tmp=tmp)

    print("✨ Sinematik video segmentleri oluşturuluyor...")
    segs=[]
    for i,(s,d) in enumerate(metas):
        print(f"   Segment {i+1}/{len(metas)}")
        base = str(pathlib.Path(tmp)/f"seg_{i:02d}.mp4"); make_segment(clips[i % len(clips)], d, base)
        colored = str(pathlib.Path(tmp)/f"segsub_{i:02d}.mp4")
        color = CAPTION_COLORS[i % len(CAPTION_COLORS)]
        draw_capcut_text(base, s, color, font, colored, is_hook=(i==0)); segs.append(colored)

    print("🎞️ Final video oluşturuluyor...")
    vcat = str(pathlib.Path(tmp)/"video_concat.mp4"); concat_videos(segs, vcat)
    acat = str(pathlib.Path(tmp)/"audio_concat.wav"); concat_audios(wavs, acat)

    total = ffprobe_dur(acat); print(f"📏 Toplam süre: {total:.1f}s (Hedef: {TARGET_MIN_SEC}-{TARGET_MAX_SEC}s)")
    if total < TARGET_MIN_SEC:
        deficit = TARGET_MIN_SEC - total; extra = min(deficit, 5.0)
        if extra > 0.1:
            print(f"⏱️ {extra:.1f}s sessizlik ekleniyor...")
            padded = str(pathlib.Path(tmp)/"audio_padded.wav")
            run(["ffmpeg","-y","-f","lavfi","-t",f"{extra:.2f}","-i","anullsrc=r=48000:cl=mono","-i",acat,
                 "-filter_complex","[1:a][0:a]concat=n=2:v=0:a=1", padded])
            acat = padded

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^A-Za-z0-9]+','_', tpc)[:60] or "Enhanced_Short"
    outp = f"{OUT_DIR}/{ctry}_{safe_topic}_{ts}.mp4"
    print("🔄 Video ve ses birleştiriliyor..."); mux(vcat, acat, outp)
    final_dur = ffprobe_dur(outp); print(f"✅ Video kaydedildi: {outp} ({final_dur:.1f}s)")

    def _ok(x): return isinstance(x,str) and x.strip()
    if _ok(ttl) and _ok(desc):
        meta = {"title": ttl.strip()[:95],"description": desc.strip()[:4900],"tags": (tags[:15] if isinstance(tags,list) else []),
                "privacy": VISIBILITY,"defaultLanguage": LANG,"defaultAudioLanguage": LANG}
    else:
        hook = (sentences[0].rstrip(" .!?") if sentences else f"{ctry} secrets")
        title = f"🤯 {hook} - {ctry} Facts That Will Blow Your Mind"
        if len(title) > 95: title = f"🤯 Mind-Blowing {ctry} Secrets"
        description = (f"🔥 {tpc} that 99% of people don't know!\n\nIn this video:\n"
                       f"✅ {sentences[0] if sentences else 'Amazing facts'}\n"
                       f"✅ {sentences[1] if len(sentences)>1 else 'Hidden secrets'}\n"
                       f"✅ {sentences[2] if len(sentences)>2 else 'Surprising discoveries'}\n\n"
                       f"🎯 Subscribe to {CHANNEL_NAME}\n💬 Comment your favorite fact!\n\n"
                       f"#shorts #facts #{ctry.lower()}facts #mindblown #education #mystery")
        tags = ["shorts","facts",f"{ctry.lower()}facts","mindblown","viral","education","mystery","secrets","amazing","science","discovery","hidden","truth","shocking","unbelievable"]
        meta = {"title": title[:95],"description": description[:4900],"tags": tags[:15],
                "privacy": VISIBILITY,"defaultLanguage": LANG,"defaultAudioLanguage": LANG}

    print("📤 YouTube'a yükleniyor...")
    try:
        vid_id = upload_youtube(outp, meta)
        print(f"🎉 YouTube Video ID: {vid_id}\n🔗 https://youtube.com/watch?v={vid_id}")
    except Exception as e:
        print(f"❌ YouTube upload hatası: {e}")

    try:
        import shutil; shutil.rmtree(tmp); print("🧹 Geçici dosyalar temizlendi")
    except: pass

# ----- utils -----
def run(cmd, check=True):
    res = subprocess.run(cmd, text=True, capture_output=True)
    if check and res.returncode != 0: raise RuntimeError(res.stderr[:2000])
    return res
def ffprobe_dur(p):
    try: return float(run(["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0", p]).stdout.strip())
    except: return 0.0
def font_path():
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc","C:/Windows/Fonts/arial.ttf"]:
        if pathlib.Path(p).exists(): return p
    return ""
def escape_drawtext(s: str) -> str:
    return (s.replace("\\","\\\\").replace(":", "\\:").replace(",", "\\,").replace("'", "\\'").replace("%","\\%"))
def clean_caption_text(s: str) -> str:
    t = (s or "").strip().replace("'","'").replace("—","-").replace('"',"").replace("`","")
    t = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', t); t = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', t); t = re.sub(r'\s+',' ', t)
    if t and t[0].islower(): t = t[0].upper() + t[1:]
    if len(t) > 80: t = " ".join(t.split()[:12]) + "."
    return t.strip()
def wrap_mobile_lines(text: str, max_line_length: int = CAPTION_MAX_LINE) -> str:
    text = (text or "").strip(); 
    if not text: return text
    words = text.split(); n = len(words); target_lines = 3 if n > 12 else 2
    def chunk(w,k): per = math.ceil(len(w)/k); return [" ".join(w[i*per:(i+1)*per]) for i in range(k) if " ".join(w[i*per:(i+1)*per])]
    chunks = chunk(words, target_lines)
    if any(len(c) > max_line_length for c in chunks) and target_lines==2: chunks = chunk(words, 3)
    if len(chunks)==1 and n>1: mid = n//2; chunks = [" ".join(words[:mid]), " ".join(words[mid:])]
    return "\n".join(chunks[:3])

def _state_load():
    try: return json.load(open(STATE_FILE, "r", encoding="utf-8"))
    except: return {"recent": []}
def _state_save(st):
    st["recent"] = st.get("recent", [])[-1000:]
    pathlib.Path(STATE_FILE).write_text(json.dumps(st, indent=2), encoding="utf-8")

def yt_service():
    cid=os.getenv("YT_CLIENT_ID"); csec=os.getenv("YT_CLIENT_SECRET"); rtok=os.getenv("YT_REFRESH_TOKEN")
    if not (cid and csec and rtok): raise RuntimeError("Missing YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN")
    creds = Credentials(token=None, refresh_token=rtok, token_uri="https://oauth2.googleapis.com/token",
                        client_id=cid, client_secret=csec, scopes=["https://www.googleapis.com/auth/youtube.upload"])
    creds.refresh(Request()); return build("youtube","v3",credentials=creds, cache_discovery=False)
def upload_youtube(video_path: str, meta: dict) -> str:
    y = yt_service()
    body = {"snippet":{"title":meta["title"],"description":meta["description"],"tags":meta.get("tags", []),
                       "categoryId":"27","defaultLanguage":meta.get("defaultLanguage", LANG),"defaultAudioLanguage":meta.get("defaultAudioLanguage", LANG)},
            "status":{"privacyStatus":meta.get("privacy", VISIBILITY),"selfDeclaredMadeForKids":False}}
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    req = y.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = req.execute(); return resp.get("id","")

if __name__ == "__main__":
    main()
