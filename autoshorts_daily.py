# autoshorts_daily.py
import os, sys, re, json, time, uuid, random, datetime, tempfile, pathlib, subprocess
from dataclasses import dataclass
from typing import List, Optional
# ---- deps ----
def _pip(p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=True)
try: import requests
except ImportError: _pip("requests"); import requests
try: import edge_tts, nest_asyncio
except ImportError: _pip("edge-tts"); _pip("nest_asyncio"); import edge_tts, nest_asyncio
try: from googleapiclient.discovery import build
except ImportError: _pip("google-api-python-client"); from googleapiclient.discovery import build
try: from google.oauth2.credentials import Credentials
except ImportError: _pip("google-auth"); from google.oauth2.credentials import Credentials

# ---- config ----
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
VOICE = "en-US-AriaNeural"
VOICE_RATE = "+10%"
TARGET_FPS = 30
CRF_VISUAL = 20
CAPTION_COLORS = ["#FFD700","#FF6B35","#00F5FF","#32CD32","#FF1493","#1E90FF"]
CAPTION_MAX_LINE = 25

# ---- utils ----
def run(cmd, check=True):
    res = subprocess.run(cmd, text=True, capture_output=True)
    if check and res.returncode != 0:
        raise RuntimeError(res.stderr[:2000])
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
    # \n'ı KAÇIŞLAMA → çok satır için gerekli
    return (s.replace("\\","\\\\").replace(":", "\\:").replace(",", "\\,").replace("'", "\\'").replace("%","\\%"))

# ---- text helpers (satır kırma/fix) ----
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
    if max(len(c) for c in chunks) > max_line_length and len(chunks) < 3:
        lines=3; per=(W+lines-1)//lines
        chunks=[" ".join(words[i*per:min(W,(i+1)*per)]) for i in range(lines)]
        chunks=[c for c in chunks if c]
    return "\n".join(chunks[:3])

# ---- script bank ----
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

def pick_script_for_today() -> dict:
    # her gün farklı ülke (deterministik): gün-yılına göre döndür
    seed = int(os.getenv("ROTATION_SEED","0"))
    tz = os.getenv("TZ","UTC")
    now = datetime.datetime.now(datetime.timezone.utc)
    doy = now.timetuple().tm_yday
    idx = (doy + seed) % len(ROTATION)
    country = ROTATION[idx]
    return {"country":country, **SCRIPT_BANK[country]}

# ---- TTS (edge → mp3 → wav) ----
import asyncio, nest_asyncio; nest_asyncio.apply()
async def _edge(text, out_mp3): await edge_tts.Communicate(text, voice=VOICE, rate=VOICE_RATE).save(out_mp3)
def tts_to_wav(text: str, out_wav: str) -> float:
    mp3 = out_wav.replace(".wav",".mp3")
    loop = asyncio.get_event_loop()
    if loop.is_running():
        fut = asyncio.ensure_future(_edge(text, mp3)); loop.run_until_complete(fut)
    else:
        loop.run_until_complete(_edge(text, mp3))
    run(["ffmpeg","-y","-i", mp3,"-ar","44100","-ac","1","-af","volume=0.95,highpass=f=80,lowpass=f=8000", out_wav])
    pathlib.Path(mp3).unlink(missing_ok=True)
    return max(0.2, ffprobe_dur(out_wav))

# ---- Pexels ----
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

# ---- video ops ----
def make_segment(src: str, dur: float, outp: str):
    dur = max(0.6, dur); fade = max(0.06, min(0.18, dur/6))
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

# ---- upload to YouTube ----
def yt_service():
    cid = os.getenv("YT_CLIENT_ID"); cs = os.getenv("YT_CLIENT_SECRET"); rt = os.getenv("YT_REFRESH_TOKEN")
    if not (cid and cs and rt): raise RuntimeError("YouTube OAuth secrets missing.")
    creds = Credentials(None, refresh_token=rt, token_uri="https://oauth2.googleapis.com/token",
                        client_id=cid, client_secret=cs, scopes=["https://www.googleapis.com/auth/youtube.upload"])
    return build("youtube","v3",credentials=creds, cache_discovery=False)

def upload_youtube(video_path: str, title: str, description: str, tags: list, privacy="public"):
    y = yt_service()
    body = {
        "snippet": {
            "title": title[:95],
            "description": description[:4900],
            "tags": tags,
            "categoryId": "27"  # Education
        },
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}
    }
    from googleapiclient.http import MediaFileUpload
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    req = y.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = req.execute()
    return resp.get("id")

# ---- main daily pipeline ----
def main():
    pathlib.Path("out").mkdir(exist_ok=True)
    # 1) choose script of the day (no repeat until cycle)
    info = pick_script_for_today()
    country, topic = info["country"], info["topic"]
    sentences = info["sentences"]; terms = info["search_terms"]
    print(f"==> {country} | {topic}")

    # temp dir
    tmp = tempfile.mkdtemp(prefix="shorts_")
    font = font_path()

    # 2) per-sentence TTS + 3) download clips
    wavs=[]; metas=[]
    for i, s in enumerate(sentences):
        w = str(pathlib.Path(tmp)/f"sent_{i:02d}.wav")
        dur = tts_to_wav(s, w)
        wavs.append(w); metas.append((s,dur))
        time.sleep(0.3)
    clips = pexels_download(terms, need=len(sentences), tmp=tmp)

    # 4) build exact-length segments + subtitles
    segs=[]
    for i,(s,d) in enumerate(metas):
        base = str(pathlib.Path(tmp)/f"seg_{i:02d}.mp4")
        make_segment(clips[i%len(clips)], d, base)
        colored = str(pathlib.Path(tmp)/f"segsub_{i:02d}.mp4")
        draw_capcut_text(base, s, CAPTION_COLORS[i%len(CAPTION_COLORS)], font, colored)
        segs.append(colored)

    # 5) concat video & audio
    vcat = str(pathlib.Path(tmp)/"video_concat.mp4"); concat_videos(segs, vcat)
    acat = str(pathlib.Path(tmp)/"audio_concat.wav"); concat_audios(wavs, acat)

    # 6) mux + save
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outp = f"out/{country}_{topic.replace(' ','_')}_{ts}.mp4"
    mux(vcat, acat, outp)
    print("Saved:", outp)

    # 7) upload
    title = f"{topic} — {country} #shorts"
    desc  = (f"Daily country facts — {country}\n"
             f"Auto-generated educational short.\n"
             f"Stock footage via Pexels (license allows reuse).")
    tags  = [country, topic, "facts", "geography", "education", "shorts"]
    vid_id = upload_youtube(outp, title, desc, tags, privacy="public")
    print("YouTube Video ID:", vid_id)

if __name__ == "__main__":
    main()
