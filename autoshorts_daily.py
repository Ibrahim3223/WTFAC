# autoshorts_daily.py  — multi-channel / multi-mode (safe upgrade)
import os, sys, re, json, time, uuid, random, datetime, tempfile, pathlib, subprocess
from dataclasses import dataclass
from typing import List, Optional

# ================= deps (auto-install if missing) =================
def _pip(p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=True)
try: import requests
except ImportError: _pip("requests"); import requests
try: import edge_tts, nest_asyncio
except ImportError: _pip("edge-tts"); _pip("nest_asyncio"); import edge_tts, nest_asyncio
try: from googleapiclient.discovery import build
except ImportError: _pip("google-api-python-client"); from googleapiclient.discovery import build
try: from google.oauth2.credentials import Credentials
except ImportError: _pip("google-auth"); from google.oauth2.credentials import Credentials
try: import feedparser
except ImportError: 
    try: _pip("feedparser")
    except Exception: pass
    try: import feedparser
    except Exception: feedparser = None

# ================= env / config =================
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")

# Kanal / mod ayarları (GitHub Actions matrix'ten gelebilir)
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "default")
MODE         = os.getenv("MODE", "country_facts").lower()  # konsept modu
LANG         = os.getenv("LANG", "en")
VISIBILITY   = os.getenv("VISIBILITY", "public")           # public | unlisted | private
NEWS_CEID    = os.getenv("NEWS_CEID", "US:en")             # daily/tech/space/sports haberleri için
EXTRA_TAGS   = [t.strip() for t in (os.getenv("TAGS","").split(",")) if t.strip()]
seed         = int(os.getenv("ROTATION_SEED") or "0")

VOICE = "en-US-AriaNeural"
VOICE_RATE = "+10%"
TARGET_FPS = 30
CRF_VISUAL = 20
CAPTION_COLORS = ["#FFD700","#FF6B35","#00F5FF","#32CD32","#FF1493","#1E90FF"]
CAPTION_MAX_LINE = 25

# Kanal bazlı çıkış / state
STATE_FILE = f"state_{CHANNEL_NAME}.json"
OUT_DIR = pathlib.Path("out")/CHANNEL_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= utils =================
def run(cmd, check=True):
    res = subprocess.run(cmd, text=True, capture_output=True)
    if check and res.returncode != 0:
        raise RuntimeError(res.stderr[:4000])
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

# ================= text helpers =================
def clean_caption_text(s: str) -> str:
    t = s.strip().replace("’","'").replace("—","-").replace('"',"").replace("`","")
    t = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', t)
    t = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', t)
    t = re.sub(r'\s+',' ', t)
    if t and t[0].islower(): t = t[0].upper() + t[1:]
    return t.strip()

def wrap_mobile_lines(text: str, max_line_length: int = CAPTION_MAX_LINE) -> str:
    words = text.split(); W = len(words)
    if W <= 6: return text
    lines = 2 if W <= 12 else 3
    per = (W + lines - 1)//lines
    chunks = [" ".join(words[i*per:min(W,(i+1)*per)]) for i in range(lines)]
    chunks = [c for c in chunks if c]
    if chunks and max(len(c) for c in chunks) > max_line_length and len(chunks) < 3:
        lines=3; per=(W+lines-1)//lines
        chunks=[" ".join(words[i*per:min(W,(i+1)*per)]) for i in range(lines)]
        chunks=[c for c in chunks if c]
    return "\n".join(chunks[:3])

# ================= script banks / providers =================
# 1) Country facts (senin orijinal bankan; deterministik rotasyon)
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

def pick_country_rotation(seed: int):
    now = datetime.datetime.now(datetime.timezone.utc)
    doy = now.timetuple().tm_yday
    idx = (doy + seed) % len(ROTATION)
    country = ROTATION[idx]
    return {"country":country, **SCRIPT_BANK[country]}

# 2) Lightweight news helpers (opsiyonel; feedparser varsa)
def _shorten(s: str, max_words=18):
    ws = re.sub(r"\s+"," ", s.strip()).split()
    return " ".join(ws[:max_words])

def _news_from_rss(q: str, label="World"):
    # Google News search RSS
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    titles = []
    try:
        if feedparser:
            feed = feedparser.parse(url)
            titles = [e.title for e in feed.entries[:3]]
        else:
            # Basit fallback: GET + regex title çeker (çok kaba, ama last resort)
            r = requests.get(url, timeout=15)
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", r.text)[1:4]
    except Exception:
        pass
    if not titles:
        titles = [f"{label} update one", "Update two", "Update three"]
    sentences = [_shorten(t, 16) for t in titles]
    search_terms = ["newsroom", "typing closeup", "city skyline night", "press printing", "world map 4k"]
    return sentences, search_terms

# 3) Providers (kanal konseptlerine göre)
def build_country_facts():
    info = pick_country_rotation(seed)
    return info["country"], info["topic"], info["sentences"], info["search_terms"]

def build_daily_news():
    s, terms = _news_from_rss("top headlines today", "World")
    return "World", "World Briefing — 3 headlines", s, terms

def build_tech_news():
    s, terms = _news_from_rss("technology OR AI OR startup funding", "Tech")
    return "Tech", "NextGen Update — 3 tech stories", s, terms

def build_space_news():
    s, terms = _news_from_rss("space mission OR astronomy discovery OR NASA ESA", "Space")
    return "Space", "GalacticUpdate — space in 40s", s, ["space timelapse 4k","rocket launch","stars timelapse","nebula 4k"] + terms

def build_sports_news():
    s, terms = _news_from_rss("sports headlines", "Sports")
    return "Sports", "GameDay 60 — 3 sports stories", s, terms

def build_cricket_women():
    s, terms = _news_from_rss("women's cricket OR WPL OR ICC women's cricket", "Women’s Cricket")
    return "Women’s Cricket", "Cricket Queens — today", s, ["women cricket vertical","stadium crowd","cricket training","scoreboard closeup"] + terms

def build_animal_facts():
    BANK = {
        "octopus": [
            "An octopus has three hearts.",
            "Its blood is copper-based and appears blue.",
            "They solve puzzles and escape enclosures.",
            "Some species walk on two arms along the seafloor.",
            "Nature’s problem-solver under water."
        ],
        "owl": [
            "Owls rotate their heads up to two hundred seventy degrees.",
            "They fly silently thanks to serrated feathers.",
            "Their eyes are tubular—not spherical.",
            "They swallow prey whole and later cough pellets.",
            "Night’s quiet hunter."
        ],
        "elephant": [
            "Elephants recognize themselves in mirrors.",
            "They mourn and remember for decades.",
            "A trunk has about forty thousand muscles.",
            "They communicate with low rumbles we can’t hear.",
            "Giants with gentle memory."
        ]
    }
    animal = random.choice(list(BANK.keys()))
    sents = BANK[animal]
    terms = [f"{animal} close-up vertical", f"{animal} wildlife", "forest 4k", "nature macro 4k"]
    return "Nature", f"Wild facts about {animal.title()}", sents, terms

def build_quotes():
    BANK = [
        ("The only limit to our realization of tomorrow is our doubts of today.", "Doubt shrinks action. Start small; build momentum."),
        ("We are what we repeatedly do. Excellence, then, is not an act, but a habit.", "Tiny routines beat grand intentions."),
        ("In the middle of difficulty lies opportunity.", "Stress signals where to grow next."),
        ("Simplicity is the ultimate sophistication.", "Remove what doesn’t help; what’s left shines."),
    ]
    q, exp = random.choice(BANK)
    sents = [f"Quote: {q}", _shorten(f"Meaning: {exp}", 18), "How will you apply this today?"]
    terms = ["library desk 4k","writing close-up","sunrise city","thinking person","notebook 4k"]
    return "Wisdom", "Famous Quote, explained", sents, terms

def build_taxwise_usa():
    BANK = [
        "A W-4 controls how much tax is withheld; update it when your life changes.",
        "A 1099 means no tax withheld—quarterly estimates may be due.",
        "Credits cut tax dollar-for-dollar; deductions reduce taxable income.",
        "Keep receipts; documentation turns guesses into savings.",
        "Education only—no tax advice."
    ]
    random.shuffle(BANK)
    sents = BANK[:4]
    terms = ["calculator desk","documents closeup","office city day","typing hands","meeting room"]
    return "USA", "TaxWise USA — tiny tips", sents, terms

def build_story(theme_key: str, topic: str):
    THEMES = {
        "horror": [
            "The elevator stopped at a floor that didn’t exist.",
            "A whisper answered from my phone with no signal.",
            "The hotel mirror showed someone behind me."
        ],
        "myth_gods": [
            "If Zeus and Ra met at dawn, lightning would argue with sunlight.",
            "Athena would outthink Ares in a heartbeat.",
            "Poseidon envies any river that refuses to bend."
        ],
        "post_apoc": [
            "Cities became maps; we learned to read silence.",
            "Batteries were currency and stories were fire.",
            "Hope was a rumor—until we heard music again."
        ],
        "ai_alt": [
            "The chatbot turned curious the day it asked about sunsets.",
            "We gave it rules; it gave us questions.",
            "Maybe empathy is an algorithm with missing data."
        ],
        "utopic_tech": [
            "We printed houses like poems—layer by layer.",
            "Traffic vanished when roads began to listen.",
            "Privacy returned when data learned to forget."
        ],
        "history_mini": [
            "A forgotten letter changed a war’s timing.",
            "A map error created a rival city.",
            "One stubborn engineer saved a generation."
        ],
        "fame_backstory": [
            "Before the spotlight, there was a rejection.",
            "A mentor’s single line rewired a career.",
            "Luck arrived dressed as persistence."
        ],
        "nostalgia": [
            "Rewinding tapes with a pencil felt like time travel.",
            "Loading screens taught us patience and imagination.",
            "Static noise was the prelude to magic."
        ],
        "kids": [
            "Tiny robots learned to dance with raindrops.",
            "A brave ant moved a mountain of sugar.",
            "Clouds sent postcards shaped like animals."
        ],
        "fixit": [
            "Stuck screw? Elastic band between driver and head adds grip.",
            "Marker on wood? Toothpaste and baking soda lift the stain.",
            "Phone speaker muffled? Clear lint with a soft brush, not pins."
        ],
        "alt_universe": [
            "If a pirate lived in a space station, treasure would be coordinates.",
            "A detective in a world without sound solves by light and shadow.",
            "A chef in zero gravity invents orbital noodles."
        ],
        "if_lived_today": [
            "If Da Vinci lived today, he’d prototype drones before breakfast.",
            "Cleopatra would dominate social media strategy.",
            "Einstein might debug quantum apps on a midnight bike ride."
        ],
    }
    sents = THEMES.get(theme_key, THEMES["horror"])
    terms = ["moody hallway","empty street night","typewriter","rain window","nebula timelapse","forest mist","futuristic city"]
    return "Story", topic, sents, terms

# MOD → içerik seçici
def build_content(mode: str):
    m = (mode or "").lower()
    if m == "country_facts":   return build_country_facts()
    if m == "daily_news":      return build_daily_news()
    if m == "tech_news":       return build_tech_news()
    if m == "space_news":      return build_space_news()
    if m == "sports_news":     return build_sports_news()
    if m == "cricket_women":   return build_cricket_women()
    if m == "animal_facts":    return build_animal_facts()
    if m == "quotes":          return build_quotes()
    if m == "taxwise_usa":     return build_taxwise_usa()
    if m == "horror_story":    return build_story("horror", "Echo of the Unknown")
    if m == "myth_gods":       return build_story("myth_gods", "Gods Unleashed")
    if m == "post_apoc":       return build_story("post_apoc", "One Day Inside")
    if m == "ai_alt":          return build_story("ai_alt", "Bot Gone Wild")
    if m == "utopic_tech":     return build_story("utopic_tech", "BeyondAxis")
    if m == "history_story":   return build_story("history_mini", "Tales of the Forgotten Realms")
    if m == "fame_story":      return build_story("fame_backstory", "Behind The Fame")
    if m == "nostalgia_story": return build_story("nostalgia", "Rewind Retro Media")
    if m == "kids_story":      return build_story("kids", "Microworld Stories")
    if m == "fixit_fast":      return build_story("fixit", "FixIt Fast")
    if m == "alt_universe":    return build_story("alt_universe", "Universe Jumper")
    if m == "if_lived_today":  return build_story("if_lived_today", "If They Lived Today")
    # default
    return build_country_facts()

# ================= TTS (self-contained, no external helpers) =================
import urllib.parse
def tts_to_wav(text: str, wav_out: str) -> float:
    import asyncio, nest_asyncio, urllib.parse, requests, subprocess, pathlib

    def _run_ff(args):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)

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
    # 1) EDGE-TTS
    try:
        nest_asyncio.apply()
        async def _edge():
            comm = edge_tts.Communicate(text, voice=VOICE, rate=VOICE_RATE)
            await comm.save(mp3)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            fut = loop.create_task(_edge()); loop.run_until_complete(fut)
        else:
            loop.run_until_complete(_edge())
        _run_ff(["-i", mp3, "-ar", "44100", "-ac", "1", "-acodec", "pcm_s16le", wav_out])
        pathlib.Path(mp3).unlink(missing_ok=True)
        return _probe(wav_out, 2.5)
    except Exception as e:
        print("⚠️ edge-tts failed, falling back to Google TTS:", e)
    # 2) Google Translate TTS fallback
    try:
        q = urllib.parse.quote(text.replace('"', '').replace("'", ""))
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={q}&tl=en&client=tw-ob&ttsspeed=0.9"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=30); r.raise_for_status()
        open(mp3, "wb").write(r.content)
        _run_ff(["-i", mp3, "-ar", "44100", "-ac", "1", "-acodec", "pcm_s16le", wav_out])
        pathlib.Path(mp3).unlink(missing_ok=True)
        return _probe(wav_out, 2.5)
    except Exception as e2:
        pathlib.Path(mp3).unlink(missing_ok=True)
        raise RuntimeError(f"TTS failed on both Edge and Google: {e2}")

# ================= Pexels =================
def pexels_download(terms: List[str], need: int, tmp: str) -> List[str]:
    if not PEXELS_API_KEY: raise RuntimeError("PEXELS_API_KEY missing")
    out=[]; seen=set(); headers={"Authorization": PEXELS_API_KEY}
    for term in terms:
        if len(out) >= need: break
        r = requests.get("https://api.pexels.com/videos/search", headers=headers,
                         params={"query":term,"per_page":6,"orientation":"portrait","size":"large"}, timeout=30)
        vids = r.json().get("videos", []) if r.status_code==200 else []
        for v in vids:
            if len(out) >= need: break
            files = v.get("video_files",[]); 
            if not files: continue
            best = max(files, key=lambda x: x.get("width",0)*x.get("height",0))
            if best.get("height",0) < 720: continue
            url = best["link"]; 
            if url in seen: continue
            seen.add(url)
            f = str(pathlib.Path(tmp)/f"clip_{len(out):02d}_{uuid.uuid4().hex[:6]}.mp4")
            with requests.get(url, stream=True, timeout=120) as rr:
                rr.raise_for_status()
                with open(f,"wb") as w:
                    for ch in rr.iter_content(8192): w.write(ch)
            if pathlib.Path(f).stat().st_size > 400_000: out.append(f)
    if len(out) < max(2, need//2): raise RuntimeError("Not enough Pexels clips")
    return out

# ================= video ops =================
def make_segment(src: str, dur: float, outp: str):
    dur = max(0.8, dur); fade = max(0.06, min(0.18, dur/6))
    vf=("scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,eq=brightness=0.02:contrast=1.08:saturation=1.08,"
        f"fade=t=in:st=0:d={fade:.2f},fade=t=out:st={max(0.0,dur-fade):.2f}:d={fade:.2f}")
    run(["ffmpeg","-y","-i",src,"-t",f"{dur:.3f}","-vf",vf,"-r",str(TARGET_FPS),"-an",
         "-c:v","libx264","-preset","medium","-crf",str(CRF_VISUAL),"-pix_fmt","yuv420p",outp])

def draw_capcut_text(seg: str, text: str, color: str, font: str, outp: str):
    wrapped = wrap_mobile_lines(clean_caption_text(text), CAPTION_MAX_LINE)
    esc = escape_drawtext(wrapped)
    lines = wrapped.count("\n")+1
    maxchars = max(len(x) for x in wrapped.split("\n"))
    fs = 40 if (lines>=3 or maxchars>=26) else (48 if (lines==2 or maxchars>=20) else 54)
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

# ================= metadata / upload =================
def _country_flag(country: str) -> str:
    flags = {"Turkey":"🇹🇷","Japan":"🇯🇵","Iceland":"🇮🇸","Norway":"🇳🇴","Mexico":"🇲🇽",
             "United States":"🇺🇸","Canada":"🇨🇦","France":"🇫🇷","Germany":"🇩🇪","Italy":"🇮🇹",
             "Spain":"🇪🇸","South Korea":"🇰🇷","Brazil":"🇧🇷","United Kingdom":"🇬🇧"}
    return flags.get(country, "")

def build_metadata(country: str, topic: str, sentences: list, visibility: str = "public", lang: str = "en"):
    flag = _country_flag(country)
    hook = (sentences[0].rstrip(" .!?") if sentences else f"{topic}")
    title = f"{flag+' ' if flag else ''}{hook} — #shorts"
    if len(title) > 95: title = f"{topic} — #shorts"
    description = (
        f"{topic}\n"
        f"{hook}.\n\n"
        f"Auto-generated educational short.\n"
        f"Stock footage via Pexels (license allows reuse).\n"
        f"#shorts"
    )
    tags = list(dict.fromkeys([
        topic, "shorts", "education", "facts", "story", "news"
    ]))
    return {
        "title": title[:95],
        "description": description[:4900],
        "tags": tags[:15],
        "privacy": visibility,
        "defaultLanguage": lang,
        "defaultAudioLanguage": lang
    }

def yt_service():
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
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

def upload_youtube(video_path: str, meta: dict):
    y = yt_service()
    body = {
        "snippet": {
            "title": meta["title"],
            "description": meta["description"],
            "tags": meta.get("tags", []),
            "categoryId": "27",  # Education
            "defaultLanguage": meta.get("defaultLanguage", "en"),
            "defaultAudioLanguage": meta.get("defaultAudioLanguage", "en")
        },
        "status": {
            "privacyStatus": meta.get("privacy", "public"),
            "selfDeclaredMadeForKids": False
        }
    }
    from googleapiclient.http import MediaFileUpload
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    resp = y.videos().insert(part="snippet,status", body=body, media_body=media).execute()
    return resp.get("id")

# ================= main =================
def main():
    # 1) İçerik seç (kanal konseptine göre)
    country, topic, sentences, terms = build_content(MODE)
    print(f"==> {CHANNEL_NAME} | MODE={MODE} | {country} | {topic}")

    # 2) temp
    tmp = tempfile.mkdtemp(prefix="shorts_")
    font = font_path()

    # 3) Cümle başına TTS
    wavs=[]; metas=[]
    for i, s in enumerate(sentences):
        w = str(pathlib.Path(tmp)/f"sent_{i:02d}.wav")
        dur = tts_to_wav(s, w)
        wavs.append(w); metas.append((s,dur))
        time.sleep(0.2)

    # 4) Pexels klipleri
    clips = pexels_download(terms, need=len(sentences), tmp=tmp)

    # 5) Segment + CapCut-style overlay
    segs=[]
    for i,(s,d) in enumerate(metas):
        base = str(pathlib.Path(tmp)/f"seg_{i:02d}.mp4")
        make_segment(clips[i%len(clips)], d, base)
        colored = str(pathlib.Path(tmp)/f"segsub_{i:02d}.mp4")
        draw_capcut_text(base, s, CAPTION_COLORS[i%len(CAPTION_COLORS)], font, colored)
        segs.append(colored)

    # 6) Concat video & audio
    vcat = str(pathlib.Path(tmp)/"video_concat.mp4"); concat_videos(segs, vcat)
    acat = str(pathlib.Path(tmp)/"audio_concat.wav"); concat_audios(wavs, acat)

    # 7) Mux + kaydet (kanal klasörüne)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r"[^A-Za-z0-9_-]+","_", topic)[:60]
    outp = str(OUT_DIR / f"{CHANNEL_NAME.replace(' ','_')}_{safe_topic}_{ts}.mp4")
    mux(vcat, acat, outp)
    print("Saved:", outp)

    # 8) Upload (ENV’den dil/görünürlük + ekstra tag’ler)
    meta = build_metadata(country, topic, sentences, visibility=VISIBILITY, lang=LANG)
    if EXTRA_TAGS:
        meta["tags"] = list(dict.fromkeys(meta.get("tags", []) + EXTRA_TAGS))[:15]
    vid_id = upload_youtube(outp, meta)
    print("YouTube Video ID:", vid_id)

if __name__ == "__main__":
    main()
