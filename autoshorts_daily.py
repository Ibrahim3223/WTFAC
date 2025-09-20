# autoshorts_daily.py - Shorts İçin Final Optimize Versiyon
# -*- coding: utf-8 -*-
import os, sys, re, json, time, uuid, random, datetime, tempfile, pathlib, subprocess, hashlib
from dataclasses import dataclass
from typing import List, Optional

VOICE_STYLE = os.getenv("TTS_STYLE", "narration-professional")
TARGET_MIN_SEC = float(os.getenv("TARGET_MIN_SEC", "20"))
TARGET_MAX_SEC = float(os.getenv("TARGET_MAX_SEC", "40"))

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
VISIBILITY    = os.getenv("VISIBILITY", "public")
ROTATION_SEED = int(os.getenv("ROTATION_SEED", "0"))
OUT_DIR = "out"
pathlib.Path(OUT_DIR).mkdir(exist_ok=True)

# APIs / keys
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_GEMINI     = os.getenv("USE_GEMINI", "0") == "1"
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Shorts için optimize edilmiş TTS seçenekleri
VOICE_OPTIONS = {
    "en": [
        "en-US-JennyNeural",    # En doğal kadın ses
        "en-US-JasonNeural",    # Doğal erkek ses  
        "en-US-AriaNeural",     # Profesyonel kadın
        "en-US-GuyNeural",      # Warm erkek ses
        "en-AU-NatashaNeural",  # Avustralya aksanı
        "en-GB-SoniaNeural",    # İngiliz aksanı
        "en-CA-LiamNeural",     # Kanada aksanı
    ],
    "tr": [
        "tr-TR-EmelNeural",   
        "tr-TR-AhmetNeural",  
    ]
}

VOICE = os.getenv("TTS_VOICE", VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])[0])
TARGET_FPS = 25
CRF_VISUAL = 22
CAPTION_COLORS = ["#FFD700","#FF6B35","#00F5FF","#32CD32","#FF1493","#1E90FF","#FFA500","#FF69B4"]

# SHORTS ÖZEL AYARLAR
MAX_CHARS_PER_LINE = 15  # Çok kısa satırlar
MAX_WORDS_PER_SENTENCE = 6  # Çok kısa cümleler
HOOK_FONT_SIZE = 64  # Hook için büyük font
NORMAL_FONT_SIZE = 52  # Normal cümleler için
OUTRO_FONT_SIZE = 58  # Outro için

# State management
STATE_FILE = f"state_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json"

# ---------------- Shorts TTS (problemsiz) ----------------
def optimize_text_for_speech(text: str) -> str:
    """Shorts için metin optimize"""
    optimized_text = text.strip()
    
    # Kısaltmaları açık hale getir
    abbreviations = {
        "AI": "Artificial Intelligence",
        "USA": "United States", 
        "UK": "United Kingdom",
        "NASA": "NASA",
        "DNA": "D.N.A.",
        "CEO": "C.E.O."
    }
    
    for abbr, full in abbreviations.items():
        optimized_text = re.sub(rf'\b{abbr}\b', full, optimized_text, flags=re.IGNORECASE)
    
    # XML/SSML karakterlerini temizle
    optimized_text = optimized_text.replace('<', '').replace('>', '')
    optimized_text = optimized_text.replace('&', 'and')
    
    return optimized_text

def tts_to_wav(text: str, wav_out: str) -> float:
    """Problemsiz Edge-TTS"""
    import asyncio
    
    def _run_ff(args):
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y", *args], check=True)

    def _probe(path: str, default: float = 3.0) -> float:
        try:
            pr = subprocess.run(
                ["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1", path],
                capture_output=True, text=True, check=True
            )
            return float(pr.stdout.strip())
        except Exception:
            return default

    mp3 = wav_out.replace(".wav", ".mp3")
    clean_text = text[:200] if len(text) > 200 else text
    clean_text = optimize_text_for_speech(clean_text)
    
    # Ses seçimi
    voice_rotation = VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])
    import hashlib
    text_hash = int(hashlib.md5(clean_text.encode()).hexdigest()[:4], 16)
    selected_voice = voice_rotation[text_hash % len(voice_rotation)]
    
    print(f"      🎤 {selected_voice.split('-')[-1]}: {clean_text[:25]}...")

    try:
        async def _edge_save():
            comm = edge_tts.Communicate(clean_text, voice=selected_voice, rate="+5%")
            await comm.save(mp3)

        try:
            asyncio.run(_edge_save())
        except RuntimeError:
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_edge_save())

        _run_ff(["-i", mp3, "-ar", "44100", "-ac", "1", "-acodec", "pcm_s16le",
                 "-af", "volume=0.9,dynaudnorm", wav_out])
        pathlib.Path(mp3).unlink(missing_ok=True)
        
        final_duration = _probe(wav_out, 3.0)
        print(f"      ✅ {final_duration:.1f}s")
        return final_duration

    except Exception as e:
        print(f"      ⚠️ Edge-TTS error, using Google TTS: {e}")
        try:
            q = requests.utils.quote(clean_text.replace('"','').replace("'",""))
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={q}&tl={LANG or 'en'}&client=tw-ob&ttsspeed=0.9"
            headers = {"User-Agent":"Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            open(mp3,"wb").write(r.content)
            
            _run_ff(["-i", mp3, "-ar","44100","-ac","1","-acodec","pcm_s16le", wav_out])
            pathlib.Path(mp3).unlink(missing_ok=True)
            
            return _probe(wav_out, 3.0)
        except Exception:
            _run_ff(["-f","lavfi","-t","3.0","-i","anullsrc=r=44100:cl=mono", wav_out])
            return 3.0

# ---------------- Shorts Video Processing ----------------
def clean_caption_text(s: str) -> str:
    """Shorts için agresif cümle temizleme"""
    t = (s or "").strip().replace("'","'").replace("—","-").replace('"',"").replace("`","")
    t = re.sub(r'\s+',' ', t)
    
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    
    # SHORTS için çok agresif kısaltma - MAX 45 karakter
    if len(t) > 45:
        words = t.split()
        t = " ".join(words[:MAX_WORDS_PER_SENTENCE])
        if not t.endswith('.'):
            t += "."
    
    return t.strip()

def force_consistent_wrapping(text: str) -> str:
    """TÜM altyazılar için tutarlı 2 satır formatı"""
    words = text.split()
    total_words = len(words)
    
    if total_words <= 3:
        return text  # Çok kısa ise tek satır
    
    # ZORLA 2 satır - tutarlılık için
    mid = total_words // 2
    line1 = " ".join(words[:mid])
    line2 = " ".join(words[mid:])
    
    # Satır uzunluğu kontrolü - çok uzunsa kelime azalt
    if len(line1) > MAX_CHARS_PER_LINE:
        line1_words = line1.split()
        line1 = " ".join(line1_words[:3])
    
    if len(line2) > MAX_CHARS_PER_LINE:
        line2_words = line2.split()
        line2 = " ".join(line2_words[:3])
    
    return f"{line1}\n{line2}"

def make_segment_with_timing(src: str, dur: float, outp: str, add_buffer: float = 0.3):
    """Video segmenti - timing bufferlı"""
    # Buffer ekle (altyazı görünür kalması için)
    total_dur = dur + add_buffer
    total_dur = max(1.0, min(total_dur, 6.0))  # 1-6s arası
    
    fade = min(0.1, total_dur/10)
    
    print(f"      📹 Segment: {total_dur:.1f}s (audio: {dur:.1f}s + buffer: {add_buffer:.1f}s)")
    
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "eq=brightness=0.02:contrast=1.05:saturation=1.1,"
        f"fade=t=in:st=0:d={fade:.2f},"
        f"fade=t=out:st={max(0.0,total_dur-fade):.2f}:d={fade:.2f}"
    )
    
    run(["ffmpeg","-y","-i",src,"-t",f"{total_dur:.3f}","-vf",vf,"-r","25","-an",
         "-c:v","libx264","-preset","fast","-crf","22","-pix_fmt","yuv420p", outp])

def draw_shorts_caption(seg: str, text: str, color: str, font: str, outp: str, 
                       caption_type: str = "normal"):
    """Shorts için optimize edilmiş tutarlı altyazı"""
    
    # Metni temizle ve ZORLA tutarlı formatla
    clean_text = clean_caption_text(text)
    wrapped = force_consistent_wrapping(clean_text)
    esc = escape_drawtext(wrapped)
    
    lines = wrapped.count("\n") + 1
    
    # Caption type'a göre font boyutu
    if caption_type == "hook":
        base_fs = HOOK_FONT_SIZE
        border_w = 6
        box_border = 20
        y_pos = "h/2-text_h/2"  # Orta
    elif caption_type == "outro":
        base_fs = OUTRO_FONT_SIZE  
        border_w = 5
        box_border = 18
        y_pos = "h-h/3-text_h/2"  # Alt
    else:  # normal
        base_fs = NORMAL_FONT_SIZE
        border_w = 4
        box_border = 15
        y_pos = "h-h/3-text_h/2"  # Alt
    
    # 2 satır için font boyutu ayarla
    if lines >= 2:
        base_fs = int(base_fs * 0.9)  # %10 küçült
    
    line_spacing = 6
    
    common = f"text='{esc}':fontsize={base_fs}:x=(w-text_w)/2:y={y_pos}:line_spacing={line_spacing}"
    
    # Gölge efekti
    shadow = f"drawtext={common}:fontcolor=black@0.8:borderw=0"
    # Arka plan kutusu
    box = f"drawtext={common}:fontcolor=white@0.0:box=1:boxborderw={box_border}:boxcolor=black@0.7"
    # Ana metin
    main = f"drawtext={common}:fontcolor={color}:borderw={border_w}:bordercolor=black@0.9"
    
    if font:
        fp = font.replace(":","\\:").replace(",","\\,").replace("\\","/")
        shadow += f":fontfile={fp}"
        box += f":fontfile={fp}"
        main += f":fontfile={fp}"
    
    vf = f"{shadow},{box},{main}"
    
    run(["ffmpeg","-y","-i",seg,"-vf",vf,"-c:v","libx264","-preset","medium",
         "-crf","20","-movflags","+faststart", outp])

# ---------------- Shorts Content Generation ----------------
ENHANCED_GEMINI_TEMPLATES = {
    "_default": """Create a 20-35 second YouTube Short with perfect structure.
EXACTLY 6 sentences total:
1. HOOK: "Did you know..." (4-5 words, attention-grabbing)
2-4. CONTENT: Main facts (4-6 words each, simple language)  
5. IMPACT: Why it matters (4-6 words)
6. OUTRO: Engaging question (4-6 words)

Each sentence MUST be 4-6 words maximum for perfect shorts captions.
Return JSON: country, topic, sentences, search_terms, title, description, tags.""",

    "country_facts": """Create amazing country facts for shorts.
EXACTLY 6 sentences:
1. HOOK: "Did you know [country]..." (4-5 words)
2. Geographic context (4-6 words)
3. Amazing fact with number (4-6 words)
4. Cultural detail (4-6 words)
5. Global significance (4-6 words)
6. OUTRO: "Have you visited [country]?" (4-5 words)

Keep each sentence under 6 words for perfect mobile captions.""",

    "fixit_fast": """Create repair instructions for shorts.
EXACTLY 6 sentences:
1. HOOK: "Here's how to fix..." (4-5 words)
2. Safety first step (4-6 words)
3. Main repair action (4-6 words)
4. Key technique (4-6 words)
5. Test result (4-6 words)
6. OUTRO: "Try this repair today!" (4-5 words)

Each sentence 4-6 words maximum."""
}

def build_via_gemini(mode: str, channel_name: str, banlist: List[str]) -> tuple:
    """Shorts için optimize edilmiş Gemini entegrasyonu"""
    
    template = ENHANCED_GEMINI_TEMPLATES.get(mode, ENHANCED_GEMINI_TEMPLATES["_default"])
    avoid = "\n".join(f"- {b}" for b in banlist[:10]) if banlist else "(none)"
    
    enhanced_prompt = f"""{template}

Channel: {channel_name}
Language: {LANG}

AVOID these recent topics:
{avoid}

CRITICAL REQUIREMENTS for Shorts:
- EXACTLY 6 sentences total
- Each sentence 4-6 words MAXIMUM (this is crucial for mobile captions)
- Sentence 1: Hook to grab attention  
- Sentences 2-4: Core content
- Sentence 5: Impact/significance
- Sentence 6: Engaging outro question
- Simple, family-friendly language
- Perfect for 20-35 second videos

Return ONLY valid JSON:
{{
  "country": "<location or theme>",
  "topic": "<engaging title>", 
  "sentences": ["<exactly 6 sentences, 4-6 words each>"],
  "search_terms": ["<4-5 terms>"],
  "title": "<title under 80 chars>",
  "description": "<description 500-800 chars>",
  "tags": ["<10 relevant tags>"]
}}
"""

    try:
        data = _gemini_call(enhanced_prompt, GEMINI_MODEL)
        
        country = str(data.get("country", "World")).strip()
        topic = str(data.get("topic", "Amazing Facts")).strip()
        
        sentences = [clean_caption_text(s) for s in (data.get("sentences") or [])]
        sentences = [s for s in sentences if s and len(s.split()) <= MAX_WORDS_PER_SENTENCE]
        
        # EXACTLY 6 sentences kontrolü
        if len(sentences) != 6:
            print(f"⚠️ Sentence count error: {len(sentences)}, using fallback")
            raise ValueError("Incorrect sentence count")
        
        print(f"✅ Gemini Shorts content: 6 sentences - {topic}")
        
        terms = data.get("search_terms") or []
        if isinstance(terms, str):
            terms = [terms]
        terms = [t.strip() for t in terms if isinstance(t, str) and t.strip()]
        
        if not terms:
            terms = ["portrait 4k", "cinematic", "documentary", "education"]
        
        title = (data.get("title") or "").strip()
        description = (data.get("description") or "").strip()  
        tags = data.get("tags") or []
        
        return country, topic, sentences, terms, title, description, tags
        
    except Exception as e:
        print(f"⚠️ Gemini failed, using structured fallback: {e}")
        
        # Strukturlu fallback (6 cümle)
        return ("World", "Daily Facts", [
            "Did you know this fact?",  # Hook
            "Scientists recently discovered something amazing.",  # Content 1
            "This changes how we think.",  # Content 2
            "Experts are studying it more.",  # Content 3
            "It affects our daily lives.",  # Impact
            "What do you think?"  # Outro
        ], ["science 4k", "discovery", "education"], "", "", [])

# ---------------- Main Function ----------------
def main():
    print(f"==> {CHANNEL_NAME} | MODE={MODE} | Shorts Optimized")

    # 1) Content generation
    if USE_GEMINI and GEMINI_API_KEY:
        banlist = _recent_topics_for_prompt()
        MAX_TRIES = 5
        chosen = None
        
        for attempt in range(MAX_TRIES):
            try:
                print(f"Content generation attempt {attempt + 1}/{MAX_TRIES}")
                ctry, tpc, sents, terms, ttl, desc, tags = build_via_gemini(MODE, CHANNEL_NAME, banlist)
                
                if len(sents) == 6:  # Exactly 6 sentences required
                    sig = f"{MODE}|{tpc}|{sents[0]}"
                    h = _hash12(sig)
                    
                    if not _is_recent(h, window_days=90):
                        _record_recent(h, MODE, tpc)
                        chosen = (ctry, tpc, sents, terms, ttl, desc, tags)
                        print(f"✅ Unique shorts content created: {tpc}")
                        break
                    else:
                        banlist.insert(0, tpc)
                        time.sleep(2)
                        continue
                else:
                    print(f"⚠️ Wrong sentence count: {len(sents)}")
                    continue
                    
            except Exception as e:
                print(f"⚠️ Gemini attempt failed: {str(e)[:100]}")
                time.sleep(2)
        
        if chosen is None:
            print("Using structured fallback...")
            ctry, tpc, sents, terms, ttl, desc, tags = ("World", "Daily Facts", [
                "Did you know this?",
                "Science reveals amazing secrets daily.",
                "Researchers work very hard.",
                "Knowledge keeps growing fast.",
                "This affects our lives.",
                "What interests you most?"
            ], ["science 4k", "discovery"], "", "", [])
    else:
        print("Gemini disabled, using fallback...")
        ctry, tpc, sents, terms, ttl, desc, tags = ("World", "Daily Facts", [
            "Did you know this?",
            "Science reveals amazing secrets.",
            "Researchers work very hard.",
            "Knowledge keeps growing fast.",
            "This affects our lives.",
            "What interests you most?"
        ], ["science 4k", "discovery"], "", "", [])

    print(f"📝 Content: {ctry} | {tpc}")
    print(f"📊 Sentence count: {len(sents)}")
    sentences = sents
    search_terms = terms

    # 2) TTS processing
    tmp = tempfile.mkdtemp(prefix="shorts_")
    font = font_path()
    wavs = []
    metas = []
    
    print("🎤 TTS processing...")
    for i, s in enumerate(sentences):
        print(f"   Sentence {i+1}/{len(sentences)}: {s}")
        w = str(pathlib.Path(tmp)/f"sent_{i:02d}.wav")
        dur = tts_to_wav(s, w)
        wavs.append(w)
        metas.append((s, dur))
        time.sleep(0.2)

    # 3) Video download
    print("🎬 Downloading videos...")
    clips = pexels_download(search_terms, need=len(sentences), tmp=tmp)

    # 4) Video segments with proper timing
    print("✨ Creating video segments with timing...")
    segs = []
    for i, (s, d) in enumerate(metas):
        print(f"   Segment {i+1}/{len(metas)}")
        base = str(pathlib.Path(tmp) / f"seg_{i:02d}.mp4")
        
        # Buffer ekle (son segment için daha fazla)
        buffer = 0.5 if i == len(metas)-1 else 0.3
        make_segment_with_timing(clips[i % len(clips)], d, base, add_buffer=buffer)
        
        # Caption type belirleme
        if i == 0:
            cap_type = "hook"
        elif i == len(metas)-1:
            cap_type = "outro"
        else:
            cap_type = "normal"
        
        colored = str(pathlib.Path(tmp) / f"segsub_{i:02d}.mp4")
        color = CAPTION_COLORS[i % len(CAPTION_COLORS)]
        draw_shorts_caption(base, s, color, font, colored, caption_type=cap_type)
        segs.append(colored)

    # 5) Final assembly
    print("🎞️ Final video assembly...")
    vcat = str(pathlib.Path(tmp) / "video_concat.mp4")
    concat_videos(segs, vcat)

    acat = str(pathlib.Path(tmp) / "audio_concat.wav")
    concat_audios(wavs, acat)

    # 6) Final export
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^A-Za-z0-9]+','_', tpc)[:50] or "Shorts"
    outp = f"{OUT_DIR}/{ctry}_{safe_topic}_{ts}.mp4"
    
    print("🔄 Muxing video and audio...")
    mux(vcat, acat, outp)
    final_dur = ffprobe_dur(outp)
    print(f"✅ Video saved: {outp} ({final_dur:.1f}s)")

    # 7) Metadata
    def _ok_str(x): return isinstance(x,str) and len(x.strip()) > 0
    
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
        hook = sentences[0].rstrip(" .!?") if sentences else f"{ctry} secrets"
        title = f"🤯 {hook} - Mind-Blowing Facts!"
        if len(title) > 95:
            title = f"🤯 Mind-Blowing {ctry} Facts!"
            
        description = (
            f"🔥 {tpc} that will blow your mind!\n\n"
            f"In this short:\n"
            f"✅ {sentences[0] if sentences else 'Amazing hook'}\n"
            f"✅ {sentences[1] if len(sentences) > 1 else 'Incredible facts'}\n"
            f"✅ {sentences[-1] if len(sentences) > 2 else 'Engaging outro'}\n\n"
            f"🎯 Subscribe for more mind-blowing shorts!\n"
            f"💬 Comment your thoughts below!\n\n"
            f"#shorts #facts #mindblown #viral #education"
        )
        
        tags = ["shorts", "facts", "mindblown", "viral", "education", "amazing"]
        
        meta = {
            "title": title[:95],
            "description": description[:4900],
            "tags": tags[:15],
            "privacy": VISIBILITY,
            "defaultLanguage": LANG,
            "defaultAudioLanguage": LANG
        }

    # 8) YouTube upload
    print("📤 Uploading to YouTube...")
    try:
        vid_id = upload_youtube(outp, meta)
        print(f"🎉 YouTube Video ID: {vid_id}")
        print(f"🔗 Video URL: https://youtube.com/watch?v={vid_id}")
    except Exception as e:
        print(f"❌ YouTube upload error: {e}")
        
    # Cleanup
    try:
        import shutil
        shutil.rmtree(tmp)
        print("🧹 Cleanup completed")
    except:
        pass

# ---------------- Pexels Download ----------------
def pexels_download(terms: List[str], need: int, tmp: str) -> List[str]:
    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY missing")
    out=[]; seen=set(); headers={"Authorization": PEXELS_API_KEY}
    
    for term in terms:
        if len(out) >= need: break
        try:
            r = requests.get("https://api.pexels.com/videos/search", headers=headers,
                             params={"query":term,"per_page":8,"orientation":"portrait","size":"large"}, timeout=30)
            vids = r.json().get("videos", []) if r.status_code==200 else []
            
            for v in vids:
                if len(out) >= need: break
                files = v.get("video_files",[])
                if not files: continue
                
                best = max(files, key=lambda x: x.get("width",0)*x.get("height",0))
                url = best["link"]
                if url in seen: continue
                seen.add(url)
                
                f = str(pathlib.Path(tmp)/f"clip_{len(out):02d}.mp4")
                with requests.get(url, stream=True, timeout=60) as rr:
                    rr.raise_for_status()
                    with open(f,"wb") as w:
                        for ch in rr.iter_content(8192): w.write(ch)
                
                if pathlib.Path(f).stat().st_size > 500_000:
                    out.append(f)
        except Exception:
            continue
    
    if len(out) < max(2, need//2):
        raise RuntimeError("Insufficient Pexels videos")
    return out

# ---------------- Utility Functions ----------------
def run(cmd, check=True):
    res = subprocess.run(cmd, text=True, capture_output=True)
    if check and res.returncode != 0:
        raise RuntimeError(res.stderr[:1000])
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

def concat_videos(files: List[str], outp: str):
    lst = str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst,"w") as f:
        for p in files: f.write(f"file '{p}'\n")
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",lst,"-c","copy", outp])

def concat_audios(files: List[str], outp: str):
    lst = str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst,"w") as f:
        for p in files: f.write(f"file '{p}'\n")
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",lst,"-af","volume=0.9", outp])

def mux(video: str, audio: str, outp: str):
    try:
        run(["ffmpeg","-y","-i",video,"-i",audio,"-c:v","copy","-c:a","aac",
             "-shortest","-movflags","+faststart", outp])
    except Exception as e:
        print(f"⚠️ Mux error: {e}")
        run(["ffmpeg","-y","-i",video,"-i",audio,"-c","copy","-shortest",outp])

# State management functions
def _state_load():
    try:
        return json.load(open(STATE_FILE, "r", encoding="utf-8"))
    except:
        return {"recent": []}

def _state_save(st):
    st["recent"] = st.get("recent", [])[-500:]
    pathlib.Path(STATE_FILE).write_text(json.dumps(st, indent=2), encoding="utf-8")

def _hash12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _is_recent(h: str, window_days=90) -> bool:
    now = time.time()
    for r in _state_load().get("recent", []):
        if r.get("h")==h and (now - r.get("ts",0)) < window_days*86400:
            return True
    return False

def _record_recent(h: str, mode: str, topic: str):
    st = _state_load()
    st.setdefault("recent", []).append({"h":h,"mode":mode,"topic":topic,"ts":time.time()})
    _state_save(st)

def _recent_topics_for_prompt(limit=10) -> List[str]:
    st = _state_load()
    topics = [r.get("topic","") for r in reversed(st.get("recent", [])) if r.get("topic")]
    uniq=[]
    for t in topics:
        if t and t not in uniq:
            uniq.append(t)
        if len(uniq) >= limit: break
    return uniq

def _gemini_call(prompt: str, model: str) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing")
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    payload = {"contents":[{"parts":[{"text": prompt}]}], "generationConfig":{"temperature":0.7}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(url, headers=headers, json=payload, timeout=45)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    txt = ""
    try:
        txt = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        txt = json.dumps(data)
    m = re.search(r"\{(?:.|\n)*\}", txt)
    if not m:
        raise RuntimeError("Gemini response parse error")
    raw = m.group(0).strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE)
    return json.loads(raw)

def yt_service():
    cid  = os.getenv("YT_CLIENT_ID")
    csec = os.getenv("YT_CLIENT_SECRET")
    rtok = os.getenv("YT_REFRESH_TOKEN")
    if not (cid and csec and rtok):
        raise RuntimeError("Missing YouTube credentials")
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
            "categoryId": "27",
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

if __name__ == "__main__":
    main()
