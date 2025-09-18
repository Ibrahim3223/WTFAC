# autoshorts_daily.py
# -*- coding: utf-8 -*-
import os, sys, re, json, time, uuid, random, datetime, tempfile, pathlib, subprocess, hashlib
from dataclasses import dataclass
from typing import List, Optional

VOICE_STYLE = os.getenv("TTS_STYLE", "narration-professional")  # not used by edge-tts; kept for future SSML
TARGET_MIN_SEC = float(os.getenv("TARGET_MIN_SEC", "22"))
TARGET_MAX_SEC = float(os.getenv("TARGET_MAX_SEC", "28"))

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

# ---------------- config ----------------
CHANNEL_NAME  = os.getenv("CHANNEL_NAME", "DefaultChannel")
MODE          = os.getenv("MODE", "country_facts").strip().lower()
LANG          = os.getenv("LANG", "en")
VISIBILITY    = os.getenv("VISIBILITY", "public")  # public | unlisted | private
ROTATION_SEED = int(os.getenv("ROTATION_SEED", "0"))
OUT_DIR = "out"
pathlib.Path(OUT_DIR).mkdir(exist_ok=True)

# APIs / keys
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_GEMINI     = os.getenv("USE_GEMINI", "0") == "1"
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_PROMPT  = os.getenv("GEMINI_PROMPT", "").strip() or None

VOICE      = os.getenv("TTS_VOICE", "en-US-AriaNeural")
VOICE_RATE = os.getenv("TTS_RATE", "+10%")

TARGET_FPS     = 30
CRF_VISUAL     = 20
CAPTION_COLORS = ["#FFD700","#FF6B35","#00F5FF","#32CD32","#FF1493","#1E90FF"]
CAPTION_MAX_LINE = 25

# Kanal bazlı state (tekrar engelleme)
STATE_FILE = f"state_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json"

# ---------------- utils ----------------
def run(cmd, check=True):
    res = subprocess.run(cmd, text=True, capture_output=True)
    if check and res.returncode != 0:
        raise RuntimeError(res.stderr[:2000])
    return res

def ffprobe_dur(p):
    try:
        return float(run(["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0", p]).stdout.strip())
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
    return (s.replace("\\","\\\\")
             .replace(":", "\\:")
             .replace(",", "\\,")
             .replace("'", "\\'")
             .replace("%","\\%"))

# ---------------- text helpers (satır kırma/fix) ----------------
def clean_caption_text(s: str) -> str:
    t = (s or "").strip().replace("’","'").replace("—","-").replace('"',"").replace("`","")
    t = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', t)
    t = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', t)
    t = re.sub(r'\s+',' ', t)
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    words = t.split()
    if len(words) > 18:
        t = " ".join(words[:18]).rstrip(",.;:") + "…"
    return t.strip()

def wrap_mobile_lines(text: str, max_line_length: int = CAPTION_MAX_LINE) -> str:
    words = text.split()
    W = len(words)
    if W <= 6:
        return text
    lines = 2 if W <= 12 else 3
    per = (W + lines - 1)//lines
    chunks = [" ".join(words[i*per:min(W,(i+1)*per)]) for i in range(lines)]
    chunks = [c for c in chunks if c]
    if chunks and max(len(c) for c in chunks) > max_line_length and len(chunks) < 3:
        lines=3; per=(W+lines-1)//lines
        chunks=[" ".join(words[i*per:min(W,(i+1)*per)]) for i in range(lines)]
        chunks=[c for c in chunks if c]
    return "\n".join(chunks[:3])

# ---------------- repeat guard (365 gün) ----------------
def _state_load():
    try:
        return json.load(open(STATE_FILE, "r", encoding="utf-8"))
    except:
        return {"recent": []}

def _state_save(st):
    st["recent"] = st.get("recent", [])[-1000:]
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

def _recent_topics_for_prompt(limit=20) -> List[str]:
    st = _state_load()
    topics = [r.get("topic","") for r in reversed(st.get("recent", [])) if r.get("topic")]
    uniq=[]
    for t in topics:
        if t and t not in uniq:
            uniq.append(t)
        if len(uniq) >= limit: break
    return uniq

# ---------------- TTS (Edge → Google fallback) ----------------
def tts_to_wav(text: str, wav_out: str) -> float:
    """
    Edge-TTS ile MP3 üret, WAV'a çevir; başarısız olursa Google Translate TTS fallback.
    Sonuna 0.2s sessizlik pad ekler. WAV süresini döndürür.
    """
    import asyncio
    def _run_ff(args):
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y", *args], check=True)

    def _probe(path: str, default: float = 2.5) -> float:
        try:
            pr = subprocess.run(
                ["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1", path],
                capture_output=True, text=True, check=True
            )
            return float(pr.stdout.strip())
        except Exception:
            return default

    mp3 = wav_out.replace(".wav", ".mp3")

    # 1) EDGE-TTS (style parametresi edge-tts'te yok; rate/voice kullan)
    try:
        async def _edge_save():
            comm = edge_tts.Communicate(
                text,
                voice=VOICE,
                rate=VOICE_RATE
            )
            await comm.save(mp3)

        try:
            asyncio.run(_edge_save())
        except RuntimeError:
            # already running loop (e.g., notebooks)
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_edge_save())

        _run_ff(["-i", mp3, "-ar","44100","-ac","1","-acodec","pcm_s16le", wav_out])
        pathlib.Path(mp3).unlink(missing_ok=True)

        # 0.2s pad
        pad = str(pathlib.Path(wav_out).with_suffix(".pad.wav"))
        _run_ff([
            "-f","lavfi","-t","0.20","-i","anullsrc=r=44100:cl=mono",
            "-i", wav_out, "-filter_complex","[1:a][0:a]concat=n=2:v=0:a=1", pad
        ])
        pathlib.Path(wav_out).unlink(missing_ok=True)
        pathlib.Path(pad).rename(wav_out)

        return _probe(wav_out, 2.8)

    except Exception as e:
        print("⚠️ edge-tts failed, falling back to Google TTS:", e)

    # 2) GOOGLE TRANSLATE fallback
    try:
        q = requests.utils.quote(text.replace('"','').replace("'",""))
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={q}&tl={LANG or 'en'}&client=tw-ob&ttsspeed=0.9"
        headers = {"User-Agent":"Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=30); r.raise_for_status()
        open(mp3,"wb").write(r.content)
        _run_ff(["-i", mp3, "-ar","44100","-ac","1","-acodec","pcm_s16le", wav_out])
        pathlib.Path(mp3).unlink(missing_ok=True)

        pad = str(pathlib.Path(wav_out).with_suffix(".pad.wav"))
        _run_ff([
            "-f","lavfi","-t","0.20","-i","anullsrc=r=44100:cl=mono",
            "-i", wav_out, "-filter_complex","[1:a][0:a]concat=n=2:v=0:a=1", pad
        ])
        pathlib.Path(wav_out).unlink(missing_ok=True)
        pathlib.Path(pad).rename(wav_out)

        return _probe(wav_out, 2.8)

    except Exception as e2:
        pathlib.Path(mp3).unlink(missing_ok=True)
        raise RuntimeError(f"TTS failed on both Edge and Google: {e2}")

# ---------------- Pexels ----------------
def pexels_download(terms: List[str], need: int, tmp: str) -> List[str]:
    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY missing")
    out=[]; seen=set(); headers={"Authorization": PEXELS_API_KEY}
    for term in terms:
        if len(out) >= need: break
        try:
            r = requests.get("https://api.pexels.com/videos/search", headers=headers,
                             params={"query":term,"per_page":6,"orientation":"portrait","size":"large"}, timeout=30)
            vids = r.json().get("videos", []) if r.status_code==200 else []
            for v in vids:
                if len(out) >= need: break
                files = v.get("video_files",[])
                if not files: continue
                best = max(files, key=lambda x: x.get("width",0)*x.get("height",0))
                if best.get("height",0) < 720: continue
                url = best["link"]
                if url in seen: continue
                seen.add(url)
                f = str(pathlib.Path(tmp)/f"clip_{len(out):02d}_{uuid.uuid4().hex[:6]}.mp4")
                with requests.get(url, stream=True, timeout=120) as rr:
                    rr.raise_for_status()
                    with open(f,"wb") as w:
                        for ch in rr.iter_content(8192): w.write(ch)
                if pathlib.Path(f).stat().st_size > 400_000:
                    out.append(f)
        except Exception:
            continue
    if len(out) < max(2, need//2):
        raise RuntimeError("Not enough Pexels clips")
    return out

# ---------------- video ops (CapCut tarzı yazı) ----------------
def make_segment(src: str, dur: float, outp: str):
    dur = max(0.6, dur)
    fade = max(0.06, min(0.18, dur/6))
    vf=("scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,eq=brightness=0.02:contrast=1.08:saturation=1.08,"
        f"fade=t=in:st=0:d={fade:.2f},fade=t=out:st={max(0.0,dur-fade):.2f}:d={fade:.2f}")
    run(["ffmpeg","-y","-i",src,"-t",f"{dur:.3f}","-vf",vf,"-r",str(TARGET_FPS),"-an",
         "-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),"-pix_fmt","yuv420p",outp])

def draw_capcut_text(seg: str, text: str, color: str, font: str, outp: str, is_hook: bool=False):
    wrapped = wrap_mobile_lines(clean_caption_text(text), CAPTION_MAX_LINE)
    esc = escape_drawtext(wrapped)
    lines = wrapped.count("\n")+1
    maxchars = max(len(x) for x in wrapped.split("\n"))
    base_fs = 40 if (lines>=3 or maxchars>=26) else (48 if (lines==2 or maxchars>=20) else 54)
    fs = base_fs + (8 if is_hook else 0)
    common = f"text='{esc}':fontsize={fs}:x=(w-text_w)/2:y=(h-text_h)/2:line_spacing=8"
    box = f"drawtext={common}:fontcolor=white@0.0:box=1:boxborderw=18:boxcolor=black@0.55"
    main= f"drawtext={common}:fontcolor={color}:borderw=4:bordercolor=black"
    if font:
        fp = font.replace(":","\\:").replace(",","\\,").replace("\\","/")
        box += f":fontfile={fp}"; main += f":fontfile={fp}"
    vf = box + "," + main
    run(["ffmpeg","-y","-i",seg,"-vf",vf,"-c:v","libx264","-preset","medium","-crf",str(max(18,CRF_VISUAL-2)), outp])

def concat_videos(files: List[str], outp: str):
    lst = str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst,"w") as f:
        for p in files: f.write(f"file '{p}'\n")
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",lst,"-c","copy", outp])

def concat_audios(files: List[str], outp: str):
    lst = str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst,"w") as f:
        for p in files: f.write(f"file '{p}'\n")
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",lst,"-af","volume=0.95,dynaudnorm", outp])

def mux(video: str, audio: str, outp: str):
    run(["ffmpeg","-y","-i",video,"-i",audio,"-map","0:v:0","-map","1:a:0",
         "-c:v","copy","-c:a","aac","-b:a","192k","-movflags","+faststart","-shortest", outp])

# ---------------- local fallback content (country bank) ----------------
SCRIPT_BANK = {
    "Turkey": {
        "topic": "Amazing Turkey Facts",
        "sentences": [
            "Turkey is the only country on two continents at once.",
            "Istanbul has underground tunnels dating back four thousand years.",
            "Turkey produces about seventy percent of the world's hazelnuts.",
            "There are seventeen UNESCO World Heritage sites across the country.",
            "Which Turkish fact surprised you the most?"
        ],
        "search_terms": ["turkey landscape 4k","istanbul cityscape","cappadocia 4k","turkish culture","turkey nature 4k"]
    },
    "Japan": {
        "topic": "Incredible Japan Secrets",
        "sentences": [
            "Japan has vending machines that sell almost everything.",
            "There is an island in Japan populated mainly by friendly rabbits.",
            "Japanese trains are famous for delays under twenty seconds.",
            "Japan hosts the world's oldest continuous monarchy.",
            "Which Japan fact blew your mind the most?"
        ],
        "search_terms": ["japan cityscape 4k","tokyo skyline","japanese culture","japan technology","japan nature 4k"]
    },
    "Iceland": {
        "topic": "Mind Blowing Iceland Facts",
        "sentences": [
            "Iceland runs almost entirely on renewable energy.",
            "There are no mosquitoes anywhere on the island.",
            "Iceland plants roughly three million trees every year.",
            "One in ten Icelanders will publish a book in their lifetime.",
            "Which Iceland fact amazed you the most?"
        ],
        "search_terms": ["iceland landscape 4k","iceland waterfalls","iceland northern lights 4k","reykjavik city","iceland nature 4k"]
    },
    "Norway": {
        "topic": "Fascinating Norway Secrets",
        "sentences": [
            "Norway has the most electric cars per capita in the world.",
            "Prisons in Norway focus primarily on rehabilitation.",
            "The sovereign wealth fund is worth over one point four trillion dollars.",
            "Norwegians drink around nine kilograms of coffee per person each year.",
            "Which Norway fact impressed you the most?"
        ],
        "search_terms": ["norway fjords 4k","oslo city","norwegian nature","norway landscape 4k","northern lights 4k"]
    },
    "Mexico": {
        "topic": "Unexpected Mexico Facts",
        "sentences": [
            "Mexico City is slowly sinking because it was built on a lake bed.",
            "Chocolate was first developed by ancient Mesoamerican cultures.",
            "Mexico has more UNESCO World Heritage sites than any other country in the Americas.",
            "The country introduced the world to corn, tomatoes, and chili peppers.",
            "Which Mexico fact did you find most surprising?"
        ],
        "search_terms": ["mexico city 4k portrait","mexico travel 4k","mexico culture","mexico nature 4k","mexico food street"]
    }
}
ROTATION = list(SCRIPT_BANK.keys())

def pick_country_for_today() -> dict:
    now = datetime.datetime.now(datetime.timezone.utc)
    doy = now.timetuple().tm_yday
    idx = (doy + ROTATION_SEED) % len(ROTATION)
    country = ROTATION[idx]
    return {"country": country, **SCRIPT_BANK[country]}

# ---------------- robust per-mode fallback ----------------
def _today_seeded_rng(extra_seed: int = 0):
    base = datetime.date.today().toordinal()
    return random.Random(base + int(extra_seed or 0))

def fallback_content(mode: str, lang: str, seed: int) -> dict:
    rng = _today_seeded_rng(seed)

    def _mk(country, topic, sentences, terms):
        return {
            "country": country, "topic": topic,
            "sentences": [clean_caption_text(s) for s in sentences][:5],
            "search_terms": terms
        }

    if mode == "animal_facts":
        animals = [
            "Dolphin", "Elephant", "Cheetah", "Owl", "Gorilla",
            "Red Panda", "Humpback Whale", "Sea Turtle", "Wolf", "Koala",
            "Penguin", "Octopus", "Giraffe", "Tiger", "Polar Bear"
        ]
        a = rng.choice(animals)
        s = [
            f"{a}s communicate with distinct calls.",
            f"They have unique adaptations for survival.",
            f"They form social bonds and show learning.",
            f"Their diet changes with habitat and season.",
            f"Have you ever seen a {a} up close?"
        ]
        terms = [f"{a.lower()} 4k portrait", "wildlife 4k", "nature close up", "animal b-roll"]
        return _mk(a, f"{a} Facts", s, terms)

    if mode in ("daily_news","tech_news","space_news","sports_news","cricket_women"):
        topic = {
            "daily_news": "World Briefing",
            "tech_news":  "Tech Briefing",
            "space_news": "Space Briefing",
            "sports_news":"Sports Briefing",
            "cricket_women":"Women’s Cricket Briefing"
        }[mode]
        base = [
            "Top headline today.",
            "Another story shaping the day.",
            "A development you should know.",
            "One more quick update.",
            "Which story matters most to you?"
        ]
        s = [clean_caption_text(x) for x in base][:5]
        terms = ["newsroom 4k","city timelapse portrait","typing closeup 4k","press conference 4k"]
        if mode == "cricket_women":
            terms = ["cricket women 4k","stadium 4k portrait","sports crowd 4k","training 4k"]
        if mode == "space_news":
            terms = ["rocket launch 4k","night sky 4k portrait","mission control 4k","earth from space 4k"]
        if mode == "tech_news":
            terms = ["data center 4k","robotics lab 4k","coding closeup 4k","chip factory 4k"]
        if mode == "sports_news":
            terms = ["stadium 4k portrait","training 4k","scoreboard 4k","crowd 4k"]
        return _mk("World", topic, s, terms)

    if mode in ("quotes","history_story","fame_story","nostalgia_story","kids_story",
                "horror_story","post_apoc","ai_alt","utopic_tech","alt_universe","if_lived_today","taxwise_usa","fixit_fast","country_facts","default"):
        if mode == "country_facts" or mode == "default":
            x = pick_country_for_today()
            return _mk(x["country"], x["topic"], x["sentences"], x["search_terms"])

        if mode == "quotes":
            s = ["'Knowledge is power' means using facts well.","It’s not about knowing everything.","It’s about action on what you know.","What will you do today?"]
            terms = ["book closeup 4k","library 4k portrait","thinking person 4k","city walk 4k"]
            return _mk("Wisdom","Famous Quote Explained",s,terms)
        if mode == "history_story":
            s = ["A hidden tunnel under a marketplace.","Smugglers once used it at night.","Decades later, it guided a rescue.","History still echoes under our feet."]
            terms = ["old city 4k","archive 4k","narrow streets 4k","lantern 4k"]
            return _mk("History","Forgotten Passage",s,terms)
        if mode == "fame_story":
            s = ["Before fame, the craft was everything.","Years of trial shaped the style.","One moment turned practice into legacy.","What discipline are you building today?"]
            terms = ["stage lights 4k","studio 4k","microphone 4k","backstage 4k"]
            return _mk("Story","Behind The Fame",s,terms)
        if mode == "nostalgia_story":
            s = ["Cassette wheels turning slowly.","Fuzzy TV glow on the wall.","Arcade coins in a pocket.","What do you miss most?"]
            terms = ["retro tv 4k","arcade 4k portrait","cassette 4k","old radio 4k"]
            return _mk("Retro","Rewind Feels",s,terms)
        if mode == "kids_story":
            s = ["A tiny seed wanted sunlight.","Clouds moved—and made space.","The seed grew brave and tall.","What will you grow today?"]
            terms = ["cartoon clouds 4k","sunlight 4k","sprout 4k","kid drawing 4k"]
            return _mk("Kids","Sunny Seed",s,terms)
        if mode == "horror_story":
            s = ["Keys rattled on the empty table.","A door sighed without wind.","Then my phone lit up: 'I’m home.'","I live alone."]
            terms = ["dark hallway 4k","flicker light 4k","empty room 4k","rain window 4k"]
            return _mk("Story","Three Quiet Sounds",s,terms)
        if mode == "post_apoc":
            s = ["We traded batteries like gold.","Maps were stories in our heads.","Silence had a weather.","Would you keep a journal?"]
            terms = ["abandoned city 4k","dust road 4k","gas mask 4k portrait","rust 4k"]
            return _mk("Story","After The Siren",s,terms)
        if mode == "ai_alt":
            s = ["The coffee machine knows my order.","It asks about my mood first.","I change it to 'surprise me.'","It smiles on the tiny screen."]
            terms = ["robot 4k","smart home 4k","city tech 4k","typing 4k"]
            return _mk("AI","Coffee Protocol",s,terms)
        if mode == "utopic_tech":
            s = ["Solar roads power free transit.","Clean air is a public utility.","Workweeks focus on learning.","What would you build first?"]
            terms = ["futuristic city 4k","solar panels 4k","clean energy 4k","autonomous bus 4k"]
            return _mk("Future","Better Tomorrow",s,terms)
        if mode == "alt_universe":
            s = ["If pirates coded in Java.","If wizards ran DNS.","If dragons loved patch notes.","Which universe do you ship?"]
            terms = ["fantasy city 4k","computer code 4k","scrolls 4k","portal 4k"]
            return _mk("WhatIf","Universe Jump",s,terms)
        if mode == "if_lived_today":
            s = ["A thinker would start a newsletter.","A conqueror would run logistics.","A painter would launch a channel.","How would you start?"]
            terms = ["city coworking 4k","warehouse 4k","studio lights 4k","typing 4k"]
            return _mk("WhatIf","If They Lived Today",s,terms)
        if mode == "taxwise_usa":
            s = ["Save receipts digitally.","Track deductible categories early.","Understand standard vs itemized.","Education only—no advice."]
            terms = ["office desk 4k","calculator 4k","documents 4k","typing 4k"]
            return _mk("USA","TaxWise USA",s,terms)
        if mode == "fixit_fast":
            s = ["Unplug before touching anything.","Take a phone photo before disassembly.","Label screws by step.","Test safely and reassemble slow."]
            terms = ["toolbox 4k","workbench 4k","hands closeup 4k","screws 4k"]
            return _mk("DIY","Fix It Fast",s,terms)

    x = pick_country_for_today()
    return _mk(x["country"], x["topic"], x["sentences"], x["search_terms"])

# ---------------- Gemini helpers ----------------
GEMINI_TEMPLATES = {
  "_default": """You are a senior YouTube Shorts writer & SEO lead.
Goal: a cohesive, cinematic 22–28s vertical short in ENGLISH.

Write a 6–8 beat micro-script. Each beat = 1 sentence, 8–14 words.
Structure:
1) HOOK (problem, paradox, or bold claim)
2) SETUP (where/when/context)
3) TURN (complication / contrast)
4) DETAIL (sensory or concrete image)
5) REVEAL (insight or fact)
6) PAYOFF (resolution or takeaway)
7) CTA (reflective question)

Return STRICT JSON only:
{
 "country": "<theme or 'World'>",
 "topic": "<specific, unique title seed>",
 "sentences": ["<beat1>", "<beat2>", "..."],
 "search_terms": ["vertical 4k b-roll terms for generic stock"],
 "title": "<<=95, SEO with primary keyphrase, no emojis>",
 "description": "<900–1100 chars, natural, 2–3 keyphrases woven in, no hashtags>",
 "tags": ["10–15 short SEO tags"]
}

Constraints:
- Family-friendly. No medical/financial advice. No copyrighted lyrics/scripts.
- Lines must be matchable with generic stock b-roll (avoid niche proper nouns).
- Never repeat recent topics.
JSON ONLY.""",

  "daily_news": """... (aynı yapı, 6–7 headline beat + neutral tone, newsroom b-roll terms)""",
  "tech_news":  """...""",
  "space_news": """...""",
  "sports_news": """...""",
  "cricket_women": """... include 'women cricket' etc.""",
  "animal_facts": """Pick one animal; 6–7 beats (hook→facts→payoff→question); b-roll generic wildlife.""",
  "country_facts": """Pick one country; 6–7 beats; b-roll: city skyline, culture close-ups, nature 4k.""",
  "quotes": """One quote (public domain / generic). Explain via 6–7 beats; 'Wisdom' theme.""",
  "taxwise_usa": """3–4 tips + examples + disclaimer among 6–7 beats; 'education only'.""",
  "horror_story": """Eerie micro fiction, PG-13, 6–7 beats; abandoned interiors, dust beams, slow zooms.""",
  "history_story": """Surprising anecdote (no sensitive live persons); 6–7 beats.""",
  "fame_story": """Celebrity tropes without real names; 6–7 beats.""",
  "nostalgia_story": """Retro vibes; 6–7 beats.""",
  "kids_story": """Wholesome; 6–7 beats.""",
  "fixit_fast": """3 quick repairs across 6–7 beats; household safe items.""",
  "alt_universe": """What-if fun across 6–7 beats; generic references.""",
  "if_lived_today": """Historical-style archetype without naming real people; 6–7 beats.""",
  "utopic_tech": """Hopeful near-future across 6–7 beats; generic tech b-roll terms.""",
  "ai_alt": """AI among us; witty but safe; 6–7 beats."""
}

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

def _gemini_prompt_for_mode(mode: str, channel_name: str, banlist: List[str], custom: Optional[str]) -> str:
    base = custom if custom else GEMINI_TEMPLATES.get(mode, GEMINI_TEMPLATES["_default"])
    avoid = "\n".join(f"- {b}" for b in banlist[:20]) if banlist else "(none)"
    return f"""{base}

Channel: {channel_name}
Avoid repeating these recent topics/hooks:
{avoid}

Return JSON ONLY. No code fences, no extra text.
"""

def build_via_gemini(mode: str, channel_name: str, banlist: List[str]) -> tuple:
    prompt = _gemini_prompt_for_mode(mode, channel_name, banlist, GEMINI_PROMPT)
    data = _gemini_call(prompt, GEMINI_MODEL)

    country = str(data.get("country") or "World").strip()
    topic   = str(data.get("topic") or "Daily Short").strip()

    sentences = [clean_caption_text(s) for s in (data.get("sentences") or [])]
    sentences = [s for s in sentences if s][:5] or ["This is a short.","Please retry.","Thanks."]

    terms = data.get("search_terms") or []
    terms = [t.strip() for t in terms if t.strip()]
    if not terms:
        terms = ["city skyline 4k","typing closeup","nature 4k","silhouette","timelapse","drone 4k"]

    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    tags = data.get("tags") or []
    tags = [t.strip() for t in tags if t.strip()]
    return country, topic, sentences, terms, title, description, tags

# ---------------- YouTube service ----------------
def yt_service():
    cid  = os.getenv("YT_CLIENT_ID")
    csec = os.getenv("YT_CLIENT_SECRET")
    rtok = os.getenv("YT_REFRESH_TOKEN")
    if not (cid and csec and rtok):
        raise RuntimeError("Missing YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN")
    creds = Credentials(
        token=None,
        refresh_token=rtok,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=cid,
        client_secret=csec,
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds, cache_discovery=False)

def upload_youtube(video_path: str, meta: dict) -> str:
    y = yt_service()
    body = {
        "snippet": {
            "title": meta["title"],
            "description": meta["description"],
            "tags": meta.get("tags", []),
            "categoryId": "27",  # Education
            "defaultLanguage": meta.get("defaultLanguage", LANG),
            "defaultAudioLanguage": meta.get("defaultAudioLanguage", LANG)
        },
        "status": {
            "privacyStatus": meta.get("privacy", VISIBILITY),
            "selfDeclaredMadeForKids": False
        }
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    req = y.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = req.execute()
    return resp.get("id", "")

# ---------------- metadata fallback builder ----------------
def build_metadata_fallback(country: str, topic: str, sentences: list, visibility: str = VISIBILITY, lang: str = LANG):
    hook = (sentences[0].rstrip(" .!?") if sentences else f"{country} facts")
    title = f"{hook} — {country}"
    if len(title) > 95:
        title = f"{topic} — {country}"
    description = (
        f"{topic} — {country}. "
        f"{' '.join(sentences[:4])}\n\n"
        f"More daily shorts like this on {CHANNEL_NAME}."
    )
    tags = list(dict.fromkeys([
        country, topic, "shorts", "education", "facts", "culture", f"{country} facts"
    ]))
    return {
        "title": title[:95],
        "description": description[:4900],
        "tags": tags[:15],
        "privacy": visibility,
        "defaultLanguage": lang,
        "defaultAudioLanguage": lang
    }

# ---------------- main ----------------
def main():
    print(f"==> {CHANNEL_NAME} | MODE={MODE}")

    # 1) İçerik seçimi (Gemini → de-dup; moda uygun fallback)
    if USE_GEMINI and GEMINI_API_KEY:
        banlist = _recent_topics_for_prompt()
        MAX_TRIES = 6
        chosen = None
        last = None
        for _ in range(MAX_TRIES):
            try:
                ctry, tpc, sents, terms, ttl, desc, tags = build_via_gemini(MODE, CHANNEL_NAME, banlist)
                last = (ctry, tpc, sents, terms, ttl, desc, tags)
                sig = f"{MODE}|{tpc}|{sents[0] if sents else ''}"
                h = _hash12(sig)
                if not _is_recent(h, window_days=365):
                    _record_recent(h, MODE, tpc)
                    chosen = last
                    break
                else:
                    banlist.insert(0, tpc)
                    continue
            except Exception as e:
                print("⚠️ Gemini failed once:", str(e)[:200])
                time.sleep(1.0)
        if chosen is None:
            if last is not None:
                print("Gemini returned but all were recent → using last unique-ish result.")
                ctry, tpc, sents, terms, ttl, desc, tags = last
            else:
                print("Gemini unavailable → using mode-specific fallback.")
                fb = fallback_content(MODE, LANG, ROTATION_SEED)
                ctry, tpc, sents, terms = fb["country"], fb["topic"], fb["sentences"], fb["search_terms"]
                ttl = desc = ""; tags = []
    else:
        print("USE_GEMINI disabled or API key missing → using mode-specific fallback.")
        fb = fallback_content(MODE, LANG, ROTATION_SEED)
        ctry, tpc, sents, terms = fb["country"], fb["topic"], fb["sentences"], fb["search_terms"]
        ttl = desc = ""; tags = []

    print(f"   Content: {ctry} | {tpc}")
    sentences    = sents
    search_terms = terms

    # 2) TTS per sentence
    tmp  = tempfile.mkdtemp(prefix="shorts_")
    font = font_path()
    wavs=[]; metas=[]
    for i, s in enumerate(sentences):
        w = str(pathlib.Path(tmp)/f"sent_{i:02d}.wav")
        dur = tts_to_wav(s, w)
        wavs.append(w); metas.append((s,dur))
        time.sleep(0.25)  # rate limit

    # 3) Pexels
    clips = pexels_download(search_terms, need=len(sentences), tmp=tmp)

    # 4) Per sentence segment + text overlay
    segs = []
    for i, (s, d) in enumerate(metas):
        base = str(pathlib.Path(tmp) / f"seg_{i:02d}.mp4")
        make_segment(clips[i % len(clips)], d, base)
        colored = str(pathlib.Path(tmp) / f"segsub_{i:02d}.mp4")
        draw_capcut_text(base, s, CAPTION_COLORS[i % len(CAPTION_COLORS)], font, colored, is_hook=(i == 0))
        segs.append(colored)

    # 5) concat video & audio
    vcat = str(pathlib.Path(tmp) / "video_concat.mp4")
    concat_videos(segs, vcat)

    acat = str(pathlib.Path(tmp) / "audio_concat.wav")
    concat_audios(wavs, acat)

    # 6) toplam süre hedef aralığına çek (audio pad)
    total_dur = ffprobe_dur(acat)
    if total_dur < TARGET_MIN_SEC:
        deficit = min(TARGET_MAX_SEC, TARGET_MIN_SEC) - total_dur
        extra = max(0.0, deficit)
        if extra > 0.05:
            padded = str(pathlib.Path(tmp) / "audio_padded.wav")
            run([
                "ffmpeg","-y",
                "-f","lavfi","-t", f"{extra:.2f}", "-i", "anullsrc=r=44100:cl=mono",
                "-i", acat, "-filter_complex", "[1:a][0:a]concat=n=2:v=0:a=1",
                padded
            ])
            acat = padded

    # 7) mux + save
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^A-Za-z0-9]+','_', tpc)[:60] or "Daily_Short"
    outp = f"{OUT_DIR}/{ctry}_{safe_topic}_{ts}.mp4"
    mux(vcat, acat, outp)
    print("Saved:", outp)

    # 8) metadata (Gemini varsa onu, yoksa fallback)
    def _ok_str(x): return isinstance(x,str) and len(x.strip())>0
    if _ok_str(ttl) and _ok_str(desc):
        meta = {
            "title": ttl.strip()[:95],
            "description": desc.strip()[:4900],
            "tags": (tags[:15] if isinstance(tags, list) else []),
            "privacy": VISIBILITY,
            "defaultLanguage": LANG,
            "defaultAudioLanguage": LANG
        }
    else:
        meta = build_metadata_fallback(ctry, tpc, sentences, visibility=VISIBILITY, lang=LANG)

    # 9) upload
    vid_id = upload_youtube(outp, meta)
    print("YouTube Video ID:", vid_id)

if __name__ == "__main__":
    main()
