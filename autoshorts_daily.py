# autoshorts_daily.py - İyileştirilmiş Versiyon (multiline captions + single voice + no audio truncation)
# -*- coding: utf-8 -*-
import os, sys, re, json, time, uuid, random, datetime, tempfile, pathlib, subprocess, hashlib, math
from dataclasses import dataclass
from typing import List, Optional

VOICE_STYLE = os.getenv("TTS_STYLE", "narration-professional")
TARGET_MIN_SEC = float(os.getenv("TARGET_MIN_SEC", "20"))  # 20-40 saniye arası
TARGET_MAX_SEC = float(os.getenv("TARGET_MAX_SEC", "40"))  # Kısa ve etkili

# ---------------- deps (auto-install) ----------------
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
except ImportError:
    _pip("google-api-python-client"); from googleapiclient.discovery import build
try:
    from googleapiclient.http import MediaFileUpload
except ImportError:
    _pip("google-api-python-client"); from googleapiclient.http import MediaFileUpload
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
except ImportError:
    _pip("google-auth"); from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

# ---------------- geliştirilmiş config ----------------
CHANNEL_NAME  = os.getenv("CHANNEL_NAME", "DefaultChannel")
MODE          = os.getenv("MODE", "country_facts").strip().lower()
LANG          = os.getenv("LANG", "en")
VISIBILITY    = os.getenv("VISIBILITY", "public")
ROTATION_SEED = int(os.getenv("ROTATION_SEED", "0"))
OUT_DIR = "out"
pathlib.Path(OUT_DIR).mkdir(exist_ok=True)

# APIs / keys
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_GEMINI     = os.getenv("USE_GEMINI", "0") == "1"
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_PROMPT  = os.getenv("GEMINI_PROMPT", "").strip() or None

# Geliştirilmiş TTS seçenekleri - daha doğal sesler
VOICE_OPTIONS = {
    "en": [
        "en-US-JennyNeural", "en-US-JasonNeural", "en-US-AriaNeural", "en-US-GuyNeural",
        "en-AU-NatashaNeural", "en-GB-SoniaNeural", "en-CA-LiamNeural", "en-US-DavisNeural", "en-US-AmberNeural",
    ],
    "tr": ["tr-TR-EmelNeural", "tr-TR-AhmetNeural"]
}

# Video süre ayarları - daha uzun içerik için
TARGET_MIN_SEC = float(os.getenv("TARGET_MIN_SEC", "22"))
TARGET_MAX_SEC = float(os.getenv("TARGET_MAX_SEC", "42"))  # 22-42s arası

VOICE = os.getenv("TTS_VOICE", VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])[0])
VOICE_RATE = os.getenv("TTS_RATE", "+10%")  # Doğal hız

TARGET_FPS     = 25
CRF_VISUAL     = 22
CAPTION_COLORS = ["#FFD700","#FF6B35","#00F5FF","#32CD32","#FF1493","#1E90FF","#FFA500","#FF69B4"]
CAPTION_MAX_LINE = 22

# State management
STATE_FILE = f"state_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json"

# ---- America/New_York günü için tek-kez kilidi ----
try:
    from zoneinfo import ZoneInfo
    TZ_NY = ZoneInfo("America/New_York")
except Exception:
    TZ_NY = None

def _now_et():
    if TZ_NY:
        return datetime.datetime.now(TZ_NY)
    return datetime.datetime.utcnow()

def _daily_lock_et():
    st = _state_load()
    today = _now_et().strftime("%Y-%m-%d")
    if st.get("last_date_et") == today:
        print("🔒 Already ran once today (America/New_York). Skipping.")
        sys.exit(0)
    st["last_date_et"] = today
    _state_save(st)

# ---------------- geliştirilmiş TTS (SSML desteği) ----------------
def create_advanced_ssml(text: str, voice: str) -> str:
    optimized_text = text
    number_map = {'1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine','10':'ten',
                  '11':'eleven','12':'twelve','13':'thirteen','14':'fourteen','15':'fifteen','16':'sixteen','17':'seventeen',
                  '18':'eighteen','19':'nineteen','20':'twenty','30':'thirty','40':'forty','50':'fifty','60':'sixty','70':'seventy',
                  '80':'eighty','90':'ninety','100':'one hundred','1000':'one thousand'}
    for num, word in number_map.items():
        optimized_text = re.sub(rf'\b{num}\b', word, optimized_text)
    optimized_text = re.sub(r'\.(?=\s)', '.<break time="600ms"/>', optimized_text)
    optimized_text = re.sub(r',(?=\s)', ',<break time="350ms"/>', optimized_text)
    optimized_text = re.sub(r'\?(?=\s)', '?<break time="700ms"/>', optimized_text)
    optimized_text = re.sub(r'!(?=\s)', '!<break time="500ms"/>', optimized_text)
    optimized_text = re.sub(r':(?=\s)', ':<break time="400ms"/>', optimized_text)
    optimized_text = re.sub(r'\b(amazing|incredible|shocking|secret|hidden|mystery)\b',
                            r'<emphasis level="moderate">\1</emphasis>', optimized_text)
    optimized_text = re.sub(r'(Did you know|Listen to this|Here\'s how)',
                            r'<prosody rate="slow">\1</prosody>', optimized_text)
    ssml = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
      <voice name="{voice}">
        <prosody rate="+10%" pitch="+0Hz" volume="medium">
          <emphasis level="reduced">{optimized_text}</emphasis>
        </prosody>
      </voice>
    </speak>
    """.strip()
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
    clean_text = text[:400] if len(text) > 400 else text
    abbreviations = {"AI":"Artificial Intelligence","USA":"United States","UK":"United Kingdom","NASA":"NASA","DNA":"D.N.A.","CEO":"C.E.O.","DIY":"Do It Yourself"}
    for abbr, full in abbreviations.items(): clean_text = re.sub(rf'\b{abbr}\b', full, clean_text)
    available = VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])
    selected_voice = VOICE if VOICE in available else available[0]
    print(f"      🎤 Voice: {selected_voice} | {clean_text[:30]}...")
    try:
        async def _edge_save_premium():
            ssml_text = create_advanced_ssml(clean_text, selected_voice)
            comm = edge_tts.Communicate(ssml_text, voice=selected_voice)
            await comm.save(mp3)
        try:
            asyncio.run(_edge_save_premium())
        except RuntimeError:
            nest_asyncio.apply(); loop = asyncio.get_event_loop(); loop.run_until_complete(_edge_save_premium())
        _run_ff(["-i", mp3, "-ar","48000","-ac","1","-acodec","pcm_s16le",
                 "-af","volume=0.92,highpass=f=75,lowpass=f=15000,dynaudnorm=g=7:f=300:r=0.95,acompressor=threshold=-20dB:ratio=2:attack=5:release=50,deesser=i=0.5:m=0.5:f=6000:s=o,equalizer=f=2000:t=h:w=200:g=2,equalizer=f=100:t=h:w=50:g=1",
                 wav_out])
        pathlib.Path(mp3).unlink(missing_ok=True)
        final_duration = _probe(wav_out, 3.5)
        print(f"      ✅ Premium Edge-TTS: {final_duration:.1f}s")
        return final_duration
    except Exception as e:
        print(f"      ⚠️ SSML başarısız, basit Edge-TTS deneniyor: {e}")
        try:
            async def _edge_save_simple():
                comm = edge_tts.Communicate(clean_text, voice=selected_voice, rate=VOICE_RATE)
                await comm.save(mp3)
            try:
                asyncio.run(_edge_save_simple())
            except RuntimeError:
                nest_asyncio.apply(); loop = asyncio.get_event_loop(); loop.run_until_complete(_edge_save_simple())
            _run_ff(["-i", mp3, "-ar","44100","-ac","1","-acodec","pcm_s16le",
                     "-af","volume=0.9,dynaudnorm=g=5:f=250,acompressor=threshold=-18dB:ratio=2.5:attack=5:release=15",
                     wav_out])
            pathlib.Path(mp3).unlink(missing_ok=True)
            final_duration = _probe(wav_out, 3.5)
            print(f"      ✅ Simple Edge-TTS: {final_duration:.1f}s")
            return final_duration
        except Exception as e2:
            print(f"      ⚠️ Edge-TTS başarısız, Google TTS: {e2}")
            try:
                q = requests.utils.quote(clean_text.replace('"','').replace("'",""))
                url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={q}&tl={LANG or 'en'}&client=tw-ob&ttsspeed=0.8"
                headers = {"User-Agent":"Mozilla/5.0"}
                r = requests.get(url, headers=headers, timeout=30); r.raise_for_status()
                open(mp3,"wb").write(r.content)
                _run_ff(["-i", mp3, "-ar","44100","-ac","1","-acodec","pcm_s16le","-af","volume=0.9,dynaudnorm", wav_out])
                pathlib.Path(mp3).unlink(missing_ok=True)
                final_duration = _probe(wav_out, 3.5)
                print(f"      ✅ Google TTS: {final_duration:.1f}s")
                return final_duration
            except Exception as e3:
                print(f"      ❌ Tüm TTS başarısız: {e3}")
                _run_ff(["-f","lavfi","-t","4.0","-i","anullsrc=r=44100:cl=mono", wav_out]); return 4.0

# ---------------- geliştirilmiş Pexels (daha kaliteli videolar) ----------------
def pexels_download(terms: List[str], need: int, tmp: str) -> List[str]:
    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY missing")
    out, seen = [], set(); headers={"Authorization": PEXELS_API_KEY}

    def _score(meta):
        w,h = meta.get("width",0), meta.get("height",0)
        dur = meta.get("duration",0); size = meta.get("size",0) or 0
        aspect = (h>0 and abs((9/16)-(w/h))<0.08)  # 9:16
        dur_bonus = -abs(dur-9)+9   # 9 sn çevresi tepe
        return (2000*(1 if aspect else 0)) + (w*h/1e4) + (size/2e5) + (120*max(dur_bonus,0))

    for term in terms:
        if len(out) >= need: break
        try:
            r = requests.get("https://api.pexels.com/videos/search", headers=headers,
                             params={"query":term,"per_page":30,"orientation":"portrait","size":"large",
                                     "min_width":"1080","min_height":"1920"}, timeout=30)
            vids = r.json().get("videos", []) if r.status_code==200 else []
            cand=[]
            for v in vids:
                files = v.get("video_files",[]); if not files: continue
                best = max(files, key=lambda x: x.get("width",0)*x.get("height",0))
                if best.get("height",0) < 1080: continue
                url = best["link"]; if url in seen: continue
                meta = {"width":best.get("width",0),"height":best.get("height",0),
                        "duration":v.get("duration",0),"size":best.get("file_size",0) or 0,"url":url}
                cand.append((_score(meta), meta))
            for _,m in sorted(cand, key=lambda x: x[0], reverse=True):
                if len(out) >= need: break
                url=m["url"]; seen.add(url)
                f = str(pathlib.Path(tmp)/f"clip_{len(out):02d}_{uuid.uuid4().hex[:6]}.mp4")
                with requests.get(url, stream=True, timeout=180) as rr:
                    rr.raise_for_status()
                    with open(f,"wb") as w:
                        for ch in rr.iter_content(1<<14): w.write(ch)
                ok_size = pathlib.Path(f).stat().st_size > 1_200_000
                ok_dur  = 4 <= m["duration"] <= 20
                if ok_size and ok_dur: out.append(f)
        except Exception:
            continue

    if len(out) < max(3, need//2):
        raise RuntimeError("Yeterli kaliteli Pexels video bulunamadı")
    return out[:need]

# ---------------- geliştirilmiş video işleme ----------------
def make_segment(src: str, dur: float, outp: str):
    dur = max(0.8, min(dur, 5.0))
    fade = max(0.05, min(0.12, dur/8))
    print(f"      📹 Segment: {dur:.1f}s (max 5s)")
    vf = ("scale=1080:1920:force_original_aspect_ratio=increase,"
          "crop=1080:1920,"
          "eq=brightness=0.02:contrast=1.08:saturation=1.1,"
          f"fade=t=in:st=0:d={fade:.2f},"
          f"fade=t=out:st={max(0.0,dur-fade):.2f}:d={fade:.2f}")
    run(["ffmpeg","-y","-i",src,"-t",f"{dur:.3f}","-vf",vf,"-r","25","-an",
         "-c:v","libx264","-preset","fast","-crf","22","-pix_fmt","yuv420p", outp])

def draw_capcut_text(seg: str, text: str, color: str, font: str, outp: str, is_hook: bool=False):
    wrapped = wrap_mobile_lines(clean_caption_text(text), CAPTION_MAX_LINE)
    esc = escape_drawtext(wrapped)
    lines = wrapped.count("\n")+1; maxchars = max(len(x) for x in wrapped.split("\n"))
    if is_hook: base_fs, border_w, box_border = (52 if lines >= 3 else 58), 5, 20
    else:       base_fs, border_w, box_border = (42 if lines >= 3 else 48), 4, 16
    if maxchars > 25: base_fs -= 6
    elif maxchars > 20: base_fs -= 3
    y_pos = "h-h/3-text_h/2"
    common = f"text='{esc}':fontsize={base_fs}:x=(w-text_w)/2:y={y_pos}:line_spacing=10"
    shadow = f"drawtext={common}:fontcolor=black@0.8:borderw=0"
    box    = f"drawtext={common}:fontcolor=white@0.0:box=1:boxborderw={box_border}:boxcolor=black@0.65"
    main   = f"drawtext={common}:fontcolor={color}:borderw={border_w}:bordercolor=black@0.9"
    if font:
        fp = font.replace(":","\\:").replace(",","\\,").replace("\\","/"); shadow += f":fontfile={fp}"; box += f":fontfile={fp}"; main += f":fontfile={fp}"
    vf = f"{shadow},{box},{main}"
    run(["ffmpeg","-y","-i",seg,"-vf",vf,"-c:v","libx264","-preset","medium","-crf",str(max(16,CRF_VISUAL-3)), "-movflags","+faststart", outp])

# ---------------- geliştirilmiş içerik üretimi ----------------
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
    },
    "Japan": {
        "topic": "Japan's Hidden Technological Wonders",
        "sentences": [
            "Japan has robots that are indistinguishable from humans.",
            "These androids can hold conversations and express emotions.",
            "Some work as receptionists in hotels and stores.",
            "They're so realistic that people forget they're robots.",
            "The technology uses artificial muscles and skin.",
            "Each android costs more than a luxury car.",
            "They can learn and adapt to human behavior patterns.",
            "Japan plans to have android caregivers in every home.",
            "Would you trust an android to care for your family?"
        ],
        "search_terms": ["japan robot 4k","android technology","futuristic japan 4k","AI robotics 4k","tokyo tech 4k"]
    }
}

ENHANCED_GEMINI_TEMPLATES = {
    "_default": """Create a detailed 25-40 second YouTube Short.
EXACTLY 7-8 sentences. Each sentence 6-10 words. Simple but informative language.
Return JSON: country, topic, sentences, search_terms, title, description, tags.""",
    "country_facts": """Create amazing country facts with detail.
EXACTLY 7-8 sentences about a specific country:
1. "Did you know [country]..." (surprising hook)
2. Geographic or historical context
3. First amazing detail with numbers
4. Second surprising fact
5. Cultural or natural feature
6. Why it's special globally
7. Impact or modern relevance
8. "Have you been to [country]?"
Search terms: country name, landmarks, culture, travel.""",
    "fixit_fast": """Create detailed repair instructions.
EXACTLY 7-8 sentences for complete repair guidance:
1. "Here's how to fix..." (common problem)
2. "First, gather these tools..." (specific tools)
3. "Step one: disconnect power safely..." (safety first)
4. "Next, locate the damaged part..." (identification)
5. "Remove it by doing this..." (removal process)
6. "Install the replacement like this..." (installation)
7. "Test everything works properly..." (verification)
8. "What repair will you try next?"
Search terms: tools, repair, DIY, workshop, fixing, maintenance.""",
    "history_story": """Create detailed historical stories.
EXACTLY 7-8 sentences about historical events:
1. "Long ago, something incredible happened..."
2. Setting and time period details
3. Main characters or figures involved
4. What actually occurred (first part)
5. The dramatic turning point
6. How it concluded or changed things
7. Why it was forgotten or hidden
8. "What other secrets are hidden?"
Search terms: historical, ancient, ruins, manuscripts, archaeology.""",
    "animal_facts": """Create detailed animal abilities explanation.
EXACTLY 7-8 sentences about one specific animal:
1. "Did you know [animal] can..." (amazing ability)
2. Where this animal lives naturally
3. How this specific ability works
4. Why they evolved this adaptation
5. Comparison to human capabilities
6. Additional surprising behavior or fact
7. Role in their ecosystem
8. "Which animal amazes you most?"
Search terms: specific animal name, wildlife, nature, behavior.""",
    "movie_secrets": """Create detailed movie behind-the-scenes facts.
EXACTLY 7-8 sentences about film secrets:
1. "This famous movie scene..." (specific scene)
2. What the director originally planned
3. What went wrong during filming
4. How the actors improvised
5. Technical challenges they faced
6. How they solved the problem
7. Why it became iconic instead
8. "What's your favorite movie moment?"
Search terms: movie theater, film set, cinema, director, hollywood.""",
    "tech_news": """Create detailed technology breakthrough content.
EXACTLY 7-8 sentences about new tech:
1. "New technology can now..." (capability)
2. How the technology actually works
3. What problems it solves
4. Current testing or development stage
5. Expected timeline for public use
6. Potential impact on daily life
7. What experts are saying about it
8. "Are you excited about this?"
Search terms: technology, innovation, gadgets, AI, research lab.""",
    "space_news": """Create detailed space discovery content.
EXACTLY 7-8 sentences about space:
1. "Scientists just discovered..." (discovery)
2. Where in space they found it
3. How they made the discovery
4. What makes it so special
5. Distance from Earth with context
6. What this means for astronomy
7. Future research plans for it
8. "What space mystery interests you?"
Search terms: space, rocket, planets, telescope, astronaut, NASA.""",
    "quotes": """Create detailed quote explanations.
EXACTLY 7-8 sentences about a famous quote:
1. "Someone once said..." (quote)
2. Who said it and when
3. The situation or context behind it
4. What it literally means
5. The deeper philosophical meaning
6. How to apply it today
7. Why it's still relevant now
8. "How do you interpret this?"
Search terms: books, wisdom, philosophy, thinking, inspiration.""",
    "cricket_women": """Create detailed women's cricket content.
EXACTLY 7-8 sentences about women's cricket:
1. "Women's cricket just achieved..." (recent achievement)
2. Which team or player accomplished it
3. Details about the match or record
4. Historical context of the achievement
5. Challenges overcome to get there
6. Impact on women's cricket globally
7. What this means for future players
8. "Do you follow women's cricket?"
Search terms: women cricket, female athletes, cricket stadium, sports."""
}

# ---- İçerik doğrulama ve tekrar koruması ----
def _validate_content(sentences: List[str], description: str) -> None:
    words = len(re.findall(r"\w+", " ".join(sentences)))
    if not (7 <= len(sentences) <= 8):
        raise ValueError("sentence count must be 7–8")
    if words < 70:
        raise ValueError("too few words (<70)")
    if description and not (400 <= len(description) <= 1200):
        raise ValueError("description must be 400–1200 chars if provided")

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

def _recent_topics_for_prompt(limit=20) -> List[str]:
    st = _state_load()
    topics = [r.get("topic","") for r in reversed(st.get("recent", [])) if r.get("topic")]
    uniq=[]
    for t in topics:
        if t and t not in uniq:
            uniq.append(t)
        if len(uniq) >= limit: break
    return uniq

# ---- Gemini çağrısı ve işleyicisi ----
def _gemini_call(prompt: str, model: str) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing")
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    payload = {"contents":[{"parts":[{"text": prompt}]}], "generationConfig":{"temperature":0.8}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    txt = ""
    try:
        txt = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        txt = json.dumps(data)
    m = re.search(r"\{(?:.|\n)*\}", txt)
    if not m:
        raise RuntimeError("Gemini response parse error (no JSON)")
    raw = m.group(0).strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE)
    return json.loads(raw)

def build_via_gemini(mode: str, channel_name: str, banlist: List[str], channel_config: dict = {}) -> tuple:
    template = ENHANCED_GEMINI_TEMPLATES.get(mode, ENHANCED_GEMINI_TEMPLATES["_default"])
    channel_topic = channel_config.get('topic', '')
    channel_search_terms = channel_config.get('search_terms', [])
    content_focus = channel_config.get('content_focus', '')
    avoid = "\n".join(f"- {b}" for b in banlist[:15]) if banlist else "(none)"
    enhanced_prompt = f"""{template}

Channel: {channel_name}
Theme: {channel_topic}
Content Focus: {content_focus}
Language: {LANG}

AVOID these recent topics:
{avoid}

CHANNEL-SPECIFIC REQUIREMENTS for 25-40 second videos:
- Content MUST match the channel's theme: {channel_topic}
- Follow this focus: {content_focus}
- EXACTLY 7-8 sentences (for proper explanation!)
- Each sentence 6-10 words maximum
- Detailed but simple language, family-friendly
- Provide complete explanation especially for instructional content
- End with engaging question

Return ONLY valid JSON:
{{
  "country": "<location or theme>",
  "topic": "<channel-themed title>",
  "sentences": ["<exactly 7-8 detailed sentences>"],
  "search_terms": ["<4-6 terms matching channel theme>"],
  "title": "<engaging title under 80 chars>",
  "description": "<simple description 500-800 chars>",
  "tags": ["<10 relevant tags>"]
}}
"""
    data = _gemini_call(enhanced_prompt, GEMINI_MODEL)
    country = str(data.get("country") or channel_config.get('topic', 'World')).strip()
    topic = str(data.get("topic") or channel_topic or "Amazing Facts").strip()
    sentences = [clean_caption_text(s) for s in (data.get("sentences") or [])]
    sentences = [s for s in sentences if s]
    if len(sentences) < 7:
        sentences += ["This discovery continues amazing scientists worldwide.", "What do you think about this?"]
    sentences = sentences[:8]
    _validate_content(sentences, data.get("description") or "")
    print(f"✅ Gemini detaylı içerik: {len(sentences)} cümle - {topic}")
    terms = channel_search_terms or data.get("search_terms") or []
    if isinstance(terms, str): terms = [terms]
    terms = [t.strip() for t in terms if isinstance(t, str) and t.strip()]
    if not terms:
        mode_terms = {
            "fixit_fast": ["tools 4k","repair workshop","DIY project","fixing","maintenance"],
            "country_facts": ["world travel 4k","cultural heritage","landmarks","city skyline"],
            "history_story": ["ancient ruins 4k","historical","manuscripts","archaeology"],
            "animal_facts": ["wildlife 4k","animal close up","nature","safari"],
            "movie_secrets": ["movie theater 4k","film set","cinema","director"],
            "space_news": ["space 4k","rocket launch","planets","astronaut"],
            "tech_news": ["technology 4k","innovation","gadgets","research lab"]
        }
        terms = mode_terms.get(mode, ["documentary 4k","education","discovery","science"])
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    tags = [t.strip() for t in (data.get("tags") or []) if isinstance(t,str) and t.strip()]
    return country, topic, sentences, terms, title, description, tags

# ---------------- ana fonksiyon güncellemeleri ----------------
def main():
    print(f"==> {CHANNEL_NAME} | MODE={MODE} | Enhanced Version")

    # Günlük tek-kez kilidi (America/New_York)
    _daily_lock_et()

    # 1) Geliştirilmiş içerik üretimi
    if USE_GEMINI and GEMINI_API_KEY:
        banlist = _recent_topics_for_prompt()
        MAX_TRIES = 8
        chosen = None; last = None
        for attempt in range(MAX_TRIES):
            try:
                print(f"İçerik üretimi denemesi {attempt + 1}/{MAX_TRIES}")
                ctry, tpc, sents, terms, ttl, desc, tags = build_via_gemini(MODE, CHANNEL_NAME, banlist)
                last = (ctry, tpc, sents, terms, ttl, desc, tags)
                sig = f"{MODE}|{tpc}|{sents[0] if sents else ''}"
                h = _hash12(sig)
                if not _is_recent(h, window_days=180):
                    _record_recent(h, MODE, tpc); chosen = last
                    print(f"✅ Benzersiz içerik oluşturuldu: {tpc}"); break
                else:
                    banlist.insert(0, tpc)
                    print("⚠️ Benzer içerik tespit edildi, yeniden deneniyor..."); time.sleep(2); continue
            except Exception as e:
                print(f"⚠️ Gemini denemesi başarısız: {str(e)[:200]}"); time.sleep(3)
        if chosen is None:
            if last is not None:
                print("Son sonucu kullanıyoruz..."); ctry, tpc, sents, terms, ttl, desc, tags = last
            else:
                print("Gelişmiş fallback içeriği..."); fb = ENHANCED_SCRIPT_BANK[random.choice(list(ENHANCED_SCRIPT_BANK.keys()))]
                ctry, tpc, sents, terms = fb["country"], fb["topic"], fb["sentences"], fb["search_terms"]; ttl = desc = ""; tags = []
    else:
        print("Gemini devre dışı, gelişmiş fallback..."); fb = ENHANCED_SCRIPT_BANK[random.choice(list(ENHANCED_SCRIPT_BANK.keys()))]
        ctry, tpc, sents, terms = fb["country"], fb["topic"], fb["sentences"], fb["search_terms"]; ttl = desc = ""; tags = []

    print(f"📝 İçerik: {ctry} | {tpc}"); print(f"📊 Cümle sayısı: {len(sents)}")
    sentences = sents; search_terms = terms

    # 2) Geliştirilmiş TTS işlemi
    tmp = tempfile.mkdtemp(prefix="enhanced_shorts_"); font = font_path()
    wavs = []; metas = []
    print("🎤 Geliştirilmiş TTS işlemi başlıyor...")
    for i, s in enumerate(sentences):
        print(f"   Cümle {i+1}/{len(sentences)}: {s[:50]}...")
        w = str(pathlib.Path(tmp)/f"sent_{i:02d}.wav"); dur = tts_to_wav(s, w)
        wavs.append(w); metas.append((s, dur)); time.sleep(0.3)

    # 3) Geliştirilmiş video indirme
    print("🎬 Yüksek kaliteli video indiriliyor...")
    clips = pexels_download(search_terms, need=len(sentences), tmp=tmp)

    # 4) Geliştirilmiş video segmentleri
    print("✨ Sinematik video segmentleri oluşturuluyor...")
    segs = []
    for i, (s, d) in enumerate(metas):
        print(f"   Segment {i+1}/{len(metas)}")
        base = str(pathlib.Path(tmp) / f"seg_{i:02d}.mp4"); make_segment(clips[i % len(clips)], d, base)
        colored = str(pathlib.Path(tmp) / f"segsub_{i:02d}.mp4")
        color = CAPTION_COLORS[i % len(CAPTION_COLORS)]; draw_capcut_text(base, s, color, font, colored, is_hook=(i == 0))
        segs.append(colored)

    # 5) Final video assembly
    print("🎞️ Final video oluşturuluyor...")
    vcat = str(pathlib.Path(tmp) / "video_concat.mp4"); concat_videos(segs, vcat)
    acat = str(pathlib.Path(tmp) / "audio_concat.wav"); concat_audios(wavs, acat)  # >>> FULL narration

    # 6) Süre bilgisi
    total_dur = ffprobe_dur(acat); print(f"📏 Toplam süre: {total_dur:.1f}s (Hedef: {TARGET_MIN_SEC}-{TARGET_MAX_SEC}s)")
    if total_dur < TARGET_MIN_SEC:
        deficit = TARGET_MIN_SEC - total_dur; extra = min(deficit, 5.0)
        if extra > 0.1:
            print(f"⏱️ {extra:.1f}s sessizlik ekleniyor...")
            padded = str(pathlib.Path(tmp) / "audio_padded.wav")
            run(["ffmpeg","-y","-f","lavfi","-t", f"{extra:.2f}", "-i","anullsrc=r=48000:cl=mono","-i", acat,
                 "-filter_complex","[1:a][0:a]concat=n=2:v=0:a=1", padded])
            acat = padded

    # 7) Final export
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^A-Za-z0-9]+','_', tpc)[:60] or "Enhanced_Short"
    outp = f"{OUT_DIR}/{ctry}_{safe_topic}_{ts}.mp4"
    print("🔄 Video ve ses birleştiriliyor..."); mux(vcat, acat, outp)
    final_dur = ffprobe_dur(outp); print(f"✅ Video kaydedildi: {outp} ({final_dur:.1f}s)")

    # 8) Geliştirilmiş metadata
    def _ok_str(x): return isinstance(x,str) and len(x.strip()) > 0
    if _ok_str(ttl) and _ok_str(desc):
        meta = {"title": ttl.strip()[:95], "description": desc.strip()[:4900], "tags": (tags[:15] if isinstance(tags, list) else []),
                "privacy": VISIBILITY, "defaultLanguage": LANG, "defaultAudioLanguage": LANG}
    else:
        hook = (sentences[0].rstrip(" .!?") if sentences else f"{ctry} secrets")
        title = f"🤯 {hook} - {ctry} Facts That Will Blow Your Mind"
        if len(title) > 95: title = f"🤯 Mind-Blowing {ctry} Secrets"
        description = (f"🔥 {tpc} that 99% of people don't know!\n\nIn this video:\n"
                       f"✅ {sentences[0] if sentences else 'Amazing facts'}\n"
                       f"✅ {sentences[1] if len(sentences) > 1 else 'Hidden secrets'}\n"
                       f"✅ {sentences[2] if len(sentences) > 2 else 'Surprising discoveries'}\n\n"
                       f"🎯 Subscribe to {CHANNEL_NAME} for more mind-blowing content!\n"
                       f"💬 Comment your favorite fact below!\n\n"
                       f"#shorts #facts #{ctry.lower()}facts #mindblown #education #mystery")
        tags = ["shorts","facts",f"{ctry.lower()}facts","mindblown","viral","education","mystery","secrets","amazing","science","discovery","hidden","truth","shocking","unbelievable"]
        meta = {"title": title[:95],"description": description[:4900],"tags": tags[:15],
                "privacy": VISIBILITY,"defaultLanguage": LANG,"defaultAudioLanguage": LANG}

    # 9) YouTube upload
    print("📤 YouTube'a yükleniyor...")
    try:
        vid_id = upload_youtube(outp, meta)
        print(f"🎉 YouTube Video ID: {vid_id}"); print(f"🔗 Video URL: https://youtube.com/watch?v={vid_id}")
    except Exception as e:
        print(f"❌ YouTube upload hatası: {e}")

    # Cleanup
    try:
        import shutil; shutil.rmtree(tmp); print("🧹 Geçici dosyalar temizlendi")
    except: pass

# Diğer fonksiyonlar aynı kalıyor (utils, state management, etc.)
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
              "/System/Library/Fonts/Helvetica.ttc",
              "C:/Windows/Fonts/arial.ttf"]:
        if pathlib.Path(p).exists(): return p
    return ""

def escape_drawtext(s: str) -> str:
    return (s.replace("\\","\\\\").replace(":", "\\:").replace(",", "\\,").replace("'", "\\'").replace("%","\\%"))

def clean_caption_text(s: str) -> str:
    t = (s or "").strip().replace("'","'").replace("—","-").replace('"',"").replace("`","")
    t = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', t); t = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', t); t = re.sub(r'\s+',' ', t)
    t = re.sub(r'\d{2,}', lambda m: "many" if int(m.group()) > 100 else m.group(), t)
    t = re.sub(r'\d+\s*-\s*meter', "massive", t); t = re.sub(r'\d+\s*years?', "decades", t)
    if t and t[0].islower(): t = t[0].upper() + t[1:]
    if len(t) > 80:
        words = t.split(); t = " ".join(words[:12]) + "."
    return t.strip()

def wrap_mobile_lines(text: str, max_line_length: int = CAPTION_MAX_LINE) -> str:
    text = (text or "").strip()
    if not text: return text
    words = text.split(); n = len(words); target_lines = 3 if n > 12 else 2
    def chunk_words(w, k): per = math.ceil(len(w) / k); chunks = [" ".join(w[i*per:(i+1)*per]) for i in range(k)]; return [c for c in chunks if c]
    chunks = chunk_words(words, target_lines)
    if any(len(c) > max_line_length for c in chunks) and target_lines == 2:
        target_lines = 3; chunks = chunk_words(words, target_lines)
    if len(chunks) == 1 and n > 1:
        mid = n // 2; chunks = [" ".join(words[:mid]), " ".join(words[mid:])]
    return "\n".join(chunks[:3])

# State management
def _state_load():
    try: return json.load(open(STATE_FILE, "r", encoding="utf-8"))
    except: return {"recent": []}

def _state_save(st):
    st["recent"] = st.get("recent", [])[-1000:]
    pathlib.Path(STATE_FILE).write_text(json.dumps(st, indent=2), encoding="utf-8")

# YouTube
def yt_service():
    cid  = os.getenv("YT_CLIENT_ID"); csec = os.getenv("YT_CLIENT_SECRET"); rtok = os.getenv("YT_REFRESH_TOKEN")
    if not (cid and csec and rtok): raise RuntimeError("Missing YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN")
    creds = Credentials(token=None, refresh_token=rtok, token_uri="https://oauth2.googleapis.com/token",
                        client_id=cid, client_secret=csec, scopes=["https://www.googleapis.com/auth/youtube.upload"])
    creds.refresh(Request()); return build("youtube", "v3", credentials=creds, cache_discovery=False)

def upload_youtube(video_path: str, meta: dict) -> str:
    y = yt_service()
    body = {
        "snippet": {"title": meta["title"],"description": meta["description"],"tags": meta.get("tags", []),
                    "categoryId":"27","defaultLanguage": meta.get("defaultLanguage", LANG),"defaultAudioLanguage": meta.get("defaultAudioLanguage", LANG)},
        "status": {"privacyStatus": meta.get("privacy", VISIBILITY), "selfDeclaredMadeForKids": False}
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    req = y.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = req.execute(); return resp.get("id","")

if __name__ == "__main__":
    main()
