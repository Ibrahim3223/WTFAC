# autoshorts_daily.py - İyileştirilmiş Versiyon
# -*- coding: utf-8 -*-
import os, sys, re, json, time, uuid, random, datetime, tempfile, pathlib, subprocess, hashlib
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

# Geliştirilmiş TTS ayarları - daha doğal sesler
VOICE_OPTIONS = {
    "en": [
        "en-US-JennyNeural",  # Çok doğal kadın ses
        "en-US-AriaNeural",   # Profesyonel kadın ses
        "en-US-GuyNeural",    # Doğal erkek ses
        "en-AU-NatashaNeural", # Avustralya aksanı
        "en-GB-SoniaNeural",   # İngiliz aksanı
    ],
    "tr": [
        "tr-TR-EmelNeural",   # Türkçe kadın
        "tr-TR-AhmetNeural",  # Türkçe erkek
    ]
}

VOICE = os.getenv("TTS_VOICE", VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])[0])
VOICE_RATE = os.getenv("TTS_RATE", "+25%")  # Daha hızlı konuşma (20-40s için)
VOICE_PITCH = os.getenv("TTS_PITCH", "+0Hz")  # Pitch kontrolü

TARGET_FPS     = 30
CRF_VISUAL     = 18  # Daha yüksek görsel kalite
CAPTION_COLORS = ["#FFD700","#FF6B35","#00F5FF","#32CD32","#FF1493","#1E90FF","#FFA500","#FF69B4"]
CAPTION_MAX_LINE = 22  # Daha kısa satırlar için

# State management
STATE_FILE = f"state_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json"

# ---------------- geliştirilmiş TTS (SSML desteği) ----------------
def create_ssml(text: str, voice: str, rate: str = "+25%", pitch: str = "+0Hz") -> str:
    """SSML ile daha doğal ama hızlı seslendirme oluştur"""
    # Kısa duraklamalar (20-40s için optimize)
    text = re.sub(r'\.(?=\s)', '.<break time="200ms"/>', text)
    text = re.sub(r',(?=\s)', ',<break time="100ms"/>', text)
    text = re.sub(r'\?(?=\s)', '?<break time="250ms"/>', text)
    text = re.sub(r'!(?=\s)', '!<break time="200ms"/>', text)
    
    # Sayıları daha doğal okutma
    text = re.sub(r'\b(\d+)\b', r'<say-as interpret-as="number">\1</say-as>', text)
    
    ssml = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
        <voice name="{voice}">
            <prosody rate="{rate}" pitch="{pitch}">
                {text}
            </prosody>
        </voice>
    </speak>
    """
    return ssml.strip()

def tts_to_wav(text: str, wav_out: str) -> float:
    """Geliştirilmiş TTS - SSML desteği ve daha doğal ses"""
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

    # SSML ile Edge-TTS
    try:
        # SSML oluştur
        ssml_text = create_ssml(text, VOICE, VOICE_RATE, VOICE_PITCH)
        
        async def _edge_save():
            comm = edge_tts.Communicate(ssml_text, voice=VOICE)
            await comm.save(mp3)

        try:
            asyncio.run(_edge_save())
        except RuntimeError:
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_edge_save())

        # WAV'a çevir ve ses kalitesini artır
        _run_ff([
            "-i", mp3, 
            "-ar", "48000",  # Daha yüksek sample rate
            "-ac", "1", 
            "-acodec", "pcm_s16le",
            "-af", "volume=0.9,highpass=f=80,lowpass=f=12000,dynaudnorm=g=3:f=250",  # Ses filtreleri
            wav_out
        ])
        pathlib.Path(mp3).unlink(missing_ok=True)

        # Daha kısa pad (0.15s)
        pad = str(pathlib.Path(wav_out).with_suffix(".pad.wav"))
        _run_ff([
            "-f","lavfi","-t","0.15","-i","anullsrc=r=48000:cl=mono",
            "-i", wav_out, "-filter_complex","[1:a][0:a]concat=n=2:v=0:a=1", pad
        ])
        pathlib.Path(wav_out).unlink(missing_ok=True)
        pathlib.Path(pad).rename(wav_out)

        return _probe(wav_out, 2.8)

    except Exception as e:
        print("⚠️ SSML TTS başarısız, normal edge-tts deneniyor:", e)
        # Fallback to normal edge-tts
        try:
            async def _edge_save_simple():
                comm = edge_tts.Communicate(text, voice=VOICE, rate=VOICE_RATE)
                await comm.save(mp3)

            try:
                asyncio.run(_edge_save_simple())
            except RuntimeError:
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                loop.run_until_complete(_edge_save_simple())

            _run_ff(["-i", mp3, "-ar","48000","-ac","1","-acodec","pcm_s16le", wav_out])
            pathlib.Path(mp3).unlink(missing_ok=True)
            return _probe(wav_out, 2.8)

        except Exception as e2:
            print("⚠️ Edge-TTS tamamen başarısız, Google TTS fallback:", e2)
            # Google TTS fallback (eski kod)
            q = requests.utils.quote(text.replace('"','').replace("'",""))
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={q}&tl={LANG or 'en'}&client=tw-ob&ttsspeed=0.9"
            headers = {"User-Agent":"Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=30); r.raise_for_status()
            open(mp3,"wb").write(r.content)
            _run_ff(["-i", mp3, "-ar","48000","-ac","1","-acodec","pcm_s16le", wav_out])
            pathlib.Path(mp3).unlink(missing_ok=True)
            return _probe(wav_out, 2.8)

# ---------------- geliştirilmiş Pexels (daha kaliteli videolar) ----------------
def pexels_download(terms: List[str], need: int, tmp: str) -> List[str]:
    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY missing")
    out=[]; seen=set(); headers={"Authorization": PEXELS_API_KEY}
    
    # Daha iyi video kriterleri
    for term in terms:
        if len(out) >= need: break
        try:
            # Önce yüksek kaliteli videolar için ara
            r = requests.get("https://api.pexels.com/videos/search", headers=headers,
                             params={"query":term,"per_page":10,"orientation":"portrait","size":"large",
                                   "min_width":"1080","min_height":"1920"}, timeout=30)
            vids = r.json().get("videos", []) if r.status_code==200 else []
            
            # Video kalitesine göre sırala
            for v in vids:
                if len(out) >= need: break
                files = v.get("video_files",[])
                if not files: continue
                
                # En yüksek kaliteli dosyayı seç
                best = max(files, key=lambda x: x.get("width",0)*x.get("height",0))
                if best.get("height",0) < 1080: continue  # Minimum 1080p
                
                url = best["link"]
                if url in seen: continue
                seen.add(url)
                
                f = str(pathlib.Path(tmp)/f"clip_{len(out):02d}_{uuid.uuid4().hex[:6]}.mp4")
                with requests.get(url, stream=True, timeout=120) as rr:
                    rr.raise_for_status()
                    with open(f,"wb") as w:
                        for ch in rr.iter_content(8192): w.write(ch)
                
                # Minimum dosya boyutu kontrolü (daha kaliteli için artırıldı)
                if pathlib.Path(f).stat().st_size > 800_000:  # 800KB+
                    out.append(f)
        except Exception:
            continue
    
    if len(out) < max(3, need//2):  # Daha fazla video gereksinimi
        raise RuntimeError("Yeterli kaliteli Pexels video bulunamadı")
    return out

# ---------------- geliştirilmiş video işleme ----------------
def make_segment(src: str, dur: float, outp: str):
    """Daha sinematik video segmenti oluştur"""
    dur = max(0.8, dur)
    fade = max(0.08, min(0.22, dur/5))
    
    # Geliştirilmiş video filtreleri
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "eq=brightness=0.03:contrast=1.12:saturation=1.15:gamma=0.95,"  # Daha canlı renkler
        "unsharp=5:5:1.0:5:5:0.3,"  # Keskinlik artışı
        f"fade=t=in:st=0:d={fade:.2f}:alpha=1,"
        f"fade=t=out:st={max(0.0,dur-fade):.2f}:d={fade:.2f}:alpha=1"
    )
    
    run(["ffmpeg","-y","-i",src,"-t",f"{dur:.3f}","-vf",vf,"-r",str(TARGET_FPS),"-an",
         "-c:v","libx264","-preset","slow","-crf",str(CRF_VISUAL),"-pix_fmt","yuv420p",
         "-movflags","+faststart", outp])

def draw_capcut_text(seg: str, text: str, color: str, font: str, outp: str, is_hook: bool=False):
    """Geliştirilmiş metin overlay - CapCut tarzı animasyonlu"""
    wrapped = wrap_mobile_lines(clean_caption_text(text), CAPTION_MAX_LINE)
    esc = escape_drawtext(wrapped)
    lines = wrapped.count("\n")+1
    maxchars = max(len(x) for x in wrapped.split("\n"))
    
    # Hook için daha büyük ve bold
    if is_hook:
        base_fs = 52 if lines >= 3 else 58
        border_w = 5
        box_border = 20
    else:
        base_fs = 42 if lines >= 3 else 48
        border_w = 4
        box_border = 16
    
    # Font boyutunu karakter sayısına göre ayarla
    if maxchars > 25:
        base_fs -= 6
    elif maxchars > 20:
        base_fs -= 3
    
    # Pozisyon hesaplama - ekranın alt 1/3'ünde
    y_pos = "h-h/3-text_h/2"
    
    common = f"text='{esc}':fontsize={base_fs}:x=(w-text_w)/2:y={y_pos}:line_spacing=10"
    
    # Gölge efekti
    shadow = f"drawtext={common}:fontcolor=black@0.8:borderw=0"
    # Ana arka plan kutusu
    box = f"drawtext={common}:fontcolor=white@0.0:box=1:boxborderw={box_border}:boxcolor=black@0.65"
    # Ana metin
    main = f"drawtext={common}:fontcolor={color}:borderw={border_w}:bordercolor=black@0.9"
    
    if font:
        fp = font.replace(":","\\:").replace(",","\\,").replace("\\","/")
        shadow += f":fontfile={fp}"
        box += f":fontfile={fp}"
        main += f":fontfile={fp}"
    
    # Animasyon efekti için offset
    vf = f"{shadow},{box},{main}"
    
    run(["ffmpeg","-y","-i",seg,"-vf",vf,"-c:v","libx264","-preset","medium",
         "-crf",str(max(16,CRF_VISUAL-3)), "-movflags","+faststart", outp])

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
    # Diğer ülkeler için de benzer şekilde genişletilebilir
}

# Geliştirilmiş Gemini promptları
ENHANCED_GEMINI_TEMPLATES = {
    "_default": """You are an elite YouTube Shorts scriptwriter with 50M+ views experience.

GOAL: Create a viral 45-55 second vertical short that will get 1M+ views.

STRUCTURE (8-12 beats, each 4-6 seconds):
1. HOOK: Start with shocking fact/question that stops scrolling
2. SETUP: Brief context/location 
3. DETAIL 1: First fascinating detail with specific numbers/facts
4. TWIST: Unexpected revelation that changes everything
5. DETAIL 2: More mind-blowing specifics
6. DETAIL 3: Additional compelling information
7. CONSEQUENCE: What this means/why it matters
8. PAYOFF: Final surprising fact or connection
9. CTA: Question that drives engagement

WRITING RULES:
- Each sentence: 6-12 words maximum
- Use specific numbers, dates, measurements
- Include sensory details (sounds, colors, textures)
- Create curiosity gaps between sentences
- End sentences with cliffhangers when possible
- Use "power words": secret, hidden, forbidden, ancient, impossible

VIRAL ELEMENTS:
- Start with "Did you know..." or "This is the only..."
- Include at least 3 specific statistics
- Reference something "scientists can't explain"
- Mention time periods (thousands of years, centuries)
- Use superlatives (biggest, oldest, only, most dangerous)

Return ONLY this JSON:
{
  "country": "<location or theme>",
  "topic": "<compelling title with numbers/superlatives>", 
  "sentences": ["<8-12 sentences following structure>"],
  "search_terms": ["<6-8 specific b-roll terms for portrait videos>"],
  "title": "<viral title 60-80 chars with numbers/emojis>",
  "description": "<1200-1500 chars with storytelling, 3+ related keywords>",
  "tags": ["<15 trending hashtag-style tags>"]
}

AVOID: Generic facts everyone knows, medical advice, political content.
FOCUS: Mysterious, ancient, scientific, geographical secrets.
""",

    "country_facts": """Create viral country secrets that 99% of people don't know.

Focus on:
- Hidden underground locations
- Ancient unexplained structures  
- Scientific anomalies
- Forbidden or restricted areas
- Government secrets (declassified)
- Geographic impossibilities
- Archaeological mysteries

Structure: Hook about secret → Location → Multiple shocking details → Why it's hidden → Final revelation → Engagement question

Include specific measurements, dates, and "scientists say" credibility.
""",

    "animal_facts": """Create mind-blowing animal abilities that seem impossible.

Focus on:
- Superhuman abilities (echolocation, magnetism, time perception)
- Evolutionary impossibilities 
- Communication methods
- Survival tactics
- Predator-prey relationships
- Environmental adaptations

Structure: Impossible ability → Specific example → How it works → Comparison to humans → Multiple examples → Scientific explanation → Engagement

Use specific numbers, speeds, distances, measurements.
"""
}

def build_via_gemini(mode: str, channel_name: str, banlist: List[str]) -> tuple:
    """Geliştirilmiş Gemini entegrasyonu"""
    template = ENHANCED_GEMINI_TEMPLATES.get(mode, ENHANCED_GEMINI_TEMPLATES["_default"])
    
    avoid = "\n".join(f"- {b}" for b in banlist[:15]) if banlist else "(none)"
    
    prompt = f"""{template}

Channel: {channel_name}
Language: {LANG}

AVOID these recent topics:
{avoid}

REQUIREMENTS:
- 8-12 sentences (45-55 seconds total)
- Each sentence 6-12 words
- Include 3+ specific numbers/statistics
- Start with shocking hook
- End with engagement question
- Family-friendly content only

Return ONLY valid JSON. No code blocks or extra text.
"""

    try:
        data = _gemini_call(prompt, GEMINI_MODEL)
        
        country = str(data.get("country") or "World").strip()
        topic = str(data.get("topic") or "Amazing Facts").strip()
        
        sentences = [clean_caption_text(s) for s in (data.get("sentences") or [])]
        sentences = [s for s in sentences if s]
        
        # Minimum 8, maksimum 12 cümle
        if len(sentences) < 8:
            sentences.extend([
                "This phenomenon baffles scientists worldwide.",
                "Research continues to unlock these mysteries.", 
                "What other secrets are waiting to be discovered?",
                "Share your thoughts in the comments below."
            ])
        sentences = sentences[:12]  # Maximum 12 sentence
        
        terms = data.get("search_terms") or []
        terms = [t.strip() for t in terms if t.strip()]
        if not terms:
            terms = ["documentary style 4k","cinematic b-roll","nature close up 4k","mysterious location","ancient ruins 4k","scientific research 4k"]
        
        title = (data.get("title") or "").strip()
        description = (data.get("description") or "").strip()
        tags = data.get("tags") or []
        tags = [t.strip() for t in tags if t.strip()]
        
        return country, topic, sentences, terms, title, description, tags
        
    except Exception as e:
        print(f"⚠️ Gemini gelişmiş içerik üretimi başarısız: {e}")
        # Enhanced fallback
        enhanced_fallback = ENHANCED_SCRIPT_BANK.get(
            random.choice(list(ENHANCED_SCRIPT_BANK.keys()))
        )
        return (enhanced_fallback["country"], enhanced_fallback["topic"], 
                enhanced_fallback["sentences"], enhanced_fallback["search_terms"], "", "", [])

# ---------------- ana fonksiyon güncellemeleri ----------------
def main():
    print(f"==> {CHANNEL_NAME} | MODE={MODE} | Enhanced Version")

    # 1) Geliştirilmiş içerik üretimi
    if USE_GEMINI and GEMINI_API_KEY:
        banlist = _recent_topics_for_prompt()
        MAX_TRIES = 8  # Daha fazla deneme
        chosen = None
        last = None
        
        for attempt in range(MAX_TRIES):
            try:
                print(f"İçerik üretimi denemesi {attempt + 1}/{MAX_TRIES}")
                ctry, tpc, sents, terms, ttl, desc, tags = build_via_gemini(MODE, CHANNEL_NAME, banlist)
                last = (ctry, tpc, sents, terms, ttl, desc, tags)
                
                sig = f"{MODE}|{tpc}|{sents[0] if sents else ''}"
                h = _hash12(sig)
                
                if not _is_recent(h, window_days=180):  # 6 aylık tekrar kontrolü
                    _record_recent(h, MODE, tpc)
                    chosen = last
                    print(f"✅ Benzersiz içerik oluşturuldu: {tpc}")
                    break
                else:
                    banlist.insert(0, tpc)
                    print(f"⚠️ Benzer içerik tespit edildi, yeniden deneniyor...")
                    time.sleep(2)  # Rate limiting
                    continue
                    
            except Exception as e:
                print(f"⚠️ Gemini denemesi başarısız: {str(e)[:200]}")
                time.sleep(3)
        
        if chosen is None:
            if last is not None:
                print("Son benzersiz sonucu kullanıyoruz...")
                ctry, tpc, sents, terms, ttl, desc, tags = last
            else:
                print("Gelişmiş fallback içeriği kullanıyoruz...")
                enhanced_key = random.choice(list(ENHANCED_SCRIPT_BANK.keys()))
                fb = ENHANCED_SCRIPT_BANK[enhanced_key]
                ctry, tpc, sents, terms = fb["country"], fb["topic"], fb["sentences"], fb["search_terms"]
                ttl = desc = ""; tags = []
    else:
        print("Gemini devre dışı, gelişmiş fallback kullanıyoruz...")
        enhanced_key = random.choice(list(ENHANCED_SCRIPT_BANK.keys()))
        fb = ENHANCED_SCRIPT_BANK[enhanced_key]
        ctry, tpc, sents, terms = fb["country"], fb["topic"], fb["sentences"], fb["search_terms"]
        ttl = desc = ""; tags = []

    print(f"📝 İçerik: {ctry} | {tpc}")
    print(f"📊 Cümle sayısı: {len(sents)}")
    sentences = sents
    search_terms = terms

    # 2) Geliştirilmiş TTS işlemi
    tmp = tempfile.mkdtemp(prefix="enhanced_shorts_")
    font = font_path()
    wavs = []; metas = []
    
    print("🎤 Geliştirilmiş TTS işlemi başlıyor...")
    for i, s in enumerate(sentences):
        print(f"   Cümle {i+1}/{len(sentences)}: {s[:50]}...")
        w = str(pathlib.Path(tmp)/f"sent_{i:02d}.wav")
        dur = tts_to_wav(s, w)
        wavs.append(w)
        metas.append((s, dur))
        time.sleep(0.3)  # Rate limiting

    # 3) Geliştirilmiş video indirme
    print("🎬 Yüksek kaliteli video indiriliyor...")
    clips = pexels_download(search_terms, need=len(sentences), tmp=tmp)

    # 4) Geliştirilmiş video segmentleri
    print("✨ Sinematik video segmentleri oluşturuluyor...")
    segs = []
    for i, (s, d) in enumerate(metas):
        print(f"   Segment {i+1}/{len(metas)}")
        base = str(pathlib.Path(tmp) / f"seg_{i:02d}.mp4")
        make_segment(clips[i % len(clips)], d, base)
        
        colored = str(pathlib.Path(tmp) / f"segsub_{i:02d}.mp4")
        color = CAPTION_COLORS[i % len(CAPTION_COLORS)]
        draw_capcut_text(base, s, color, font, colored, is_hook=(i == 0))
        segs.append(colored)

    # 5) Final video assembly
    print("🎞️ Final video oluşturuluyor...")
    vcat = str(pathlib.Path(tmp) / "video_concat.mp4")
    concat_videos(segs, vcat)

    acat = str(pathlib.Path(tmp) / "audio_concat.wav")
    concat_audios(wavs, acat)

    # 6) Süre optimizasyonu
    total_dur = ffprobe_dur(acat)
    print(f"📏 Toplam süre: {total_dur:.1f}s (Hedef: {TARGET_MIN_SEC}-{TARGET_MAX_SEC}s)")
    
    if total_dur < TARGET_MIN_SEC:
        deficit = TARGET_MIN_SEC - total_dur
        extra = min(deficit, 5.0)  # Maksimum 5s ekleme
        if extra > 0.1:
            print(f"⏱️ {extra:.1f}s sessizlik ekleniyor...")
            padded = str(pathlib.Path(tmp) / "audio_padded.wav")
            run([
                "ffmpeg","-y",
                "-f","lavfi","-t", f"{extra:.2f}", "-i", "anullsrc=r=48000:cl=mono",
                "-i", acat, "-filter_complex", "[1:a][0:a]concat=n=2:v=0:a=1",
                padded
            ])
            acat = padded

    # 7) Final export
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^A-Za-z0-9]+','_', tpc)[:60] or "Enhanced_Short"
    outp = f"{OUT_DIR}/{ctry}_{safe_topic}_{ts}.mp4"
    
    print("🔄 Video ve ses birleştiriliyor...")
    mux(vcat, acat, outp)
    final_dur = ffprobe_dur(outp)
    print(f"✅ Video kaydedildi: {outp} ({final_dur:.1f}s)")

    # 8) Geliştirilmiş metadata
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
        # Geliştirilmiş fallback metadata
        hook = (sentences[0].rstrip(" .!?") if sentences else f"{ctry} secrets")
        title = f"🤯 {hook} - {ctry} Facts That Will Blow Your Mind"
        if len(title) > 95:
            title = f"🤯 Mind-Blowing {ctry} Secrets"
            
        description = (
            f"🔥 {tpc} that 99% of people don't know!\n\n"
            f"In this video:\n"
            f"✅ {sentences[0] if sentences else 'Amazing facts'}\n"
            f"✅ {sentences[1] if len(sentences) > 1 else 'Hidden secrets'}\n"
            f"✅ {sentences[2] if len(sentences) > 2 else 'Surprising discoveries'}\n\n"
            f"🎯 Subscribe to {CHANNEL_NAME} for more mind-blowing content!\n"
            f"💬 Comment your favorite fact below!\n\n"
            f"#shorts #facts #{ctry.lower()}facts #mindblown #viral #education #mystery"
        )
        
        tags = [
            "shorts", "facts", f"{ctry.lower()}facts", "mindblown", "viral", 
            "education", "mystery", "secrets", "amazing", "science",
            "discovery", "hidden", "truth", "shocking", "unbelievable"
        ]
        
        meta = {
            "title": title[:95],
            "description": description[:4900],
            "tags": tags[:15],
            "privacy": VISIBILITY,
            "defaultLanguage": LANG,
            "defaultAudioLanguage": LANG
        }

    # 9) YouTube upload
    print("📤 YouTube'a yükleniyor...")
    try:
        vid_id = upload_youtube(outp, meta)
        print(f"🎉 YouTube Video ID: {vid_id}")
        print(f"🔗 Video URL: https://youtube.com/watch?v={vid_id}")
    except Exception as e:
        print(f"❌ YouTube upload hatası: {e}")
        
    # Cleanup
    try:
        import shutil
        shutil.rmtree(tmp)
        print("🧹 Geçici dosyalar temizlendi")
    except:
        pass

# Diğer fonksiyonlar aynı kalıyor (utils, state management, etc.)
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

def clean_caption_text(s: str) -> str:
    t = (s or "").strip().replace("'","'").replace("—","-").replace('"',"").replace("`","")
    t = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', t)
    t = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', t)
    t = re.sub(r'\s+',' ', t)
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    words = t.split()
    if len(words) > 20:  # Daha uzun cümleler için artırıldı
        t = " ".join(words[:20]).rstrip(",.;:") + "…"
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

# State management functions
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

# Gemini API call function
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

# Video processing functions
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
         "-c:v","copy","-c:a","aac","-b:a","256k","-movflags","+faststart","-shortest", outp])

# YouTube functions
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

if __name__ == "__main__":
    main()
