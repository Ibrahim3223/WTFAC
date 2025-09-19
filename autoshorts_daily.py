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

# Geliştirilmiş TTS seçenekleri - daha doğal sesler
VOICE_OPTIONS = {
    "en": [
        "en-US-JennyNeural",    # En doğal kadın ses
        "en-US-JasonNeural",    # Doğal erkek ses  
        "en-US-AriaNeural",     # Profesyonel kadın
        "en-US-GuyNeural",      # Warm erkek ses
        "en-AU-NatashaNeural",  # Avustralya aksanı (çok doğal)
        "en-GB-SoniaNeural",    # İngiliz aksanı
        "en-CA-LiamNeural",     # Kanada aksanı
    ],
    "tr": [
        "tr-TR-EmelNeural",   
        "tr-TR-AhmetNeural",  
    ]
}

# TTS ayarları güncellenecek
VOICE = os.getenv("TTS_VOICE", VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])[0])
VOICE_RATE = os.getenv("TTS_RATE", "+25%")  # Hızlı ve temiz konuşma

TARGET_FPS     = 25  # Daha hızlı işlem için düşürüldü
CRF_VISUAL     = 22  # Balanced kalite/hız 
CAPTION_COLORS = ["#FFD700","#FF6B35","#00F5FF","#32CD32","#FF1493","#1E90FF","#FFA500","#FF69B4"]
CAPTION_MAX_LINE = 22

# State management
STATE_FILE = f"state_{re.sub(r'[^A-Za-z0-9]+','_',CHANNEL_NAME)}.json"

# ---------------- geliştirilmiş TTS (SSML desteği) ----------------
# TTS ayarları (SSML kaldırıldı - ses kalitesi için)
def tts_to_wav(text: str, wav_out: str) -> float:
    """İyileştirilmiş TTS - daha doğal ses + MAX 6s kontrol"""
    import asyncio
    
    def _run_ff(args):
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y", *args], check=True)

    def _probe(path: str, default: float = 2.5) -> float:
        try:
            pr = subprocess.run(
                ["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1", path],
                capture_output=True, text=True, check=True
            )
            duration = float(pr.stdout.strip())
            return min(duration, 6.0)  # ABSOLUTE MAX 6 seconds
        except Exception:
            return min(default, 6.0)

    mp3 = wav_out.replace(".wav", ".mp3")

    # Metni daha doğal TTS için optimize et
    clean_text = text[:150] if len(text) > 150 else text  # Max 150 karakter
    # Sayıları kelime olarak değiştir (daha doğal okuma için)
    clean_text = re.sub(r'\b(\d+)\b', lambda m: {
        '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
        '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
        '50': 'fifty', '100': 'hundred', '1000': 'thousand'
    }.get(m.group(), m.group()), clean_text)
    
    # Daha doğal ses seçimi (rotation için)
    voice_rotation = VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])
    import hashlib
    text_hash = int(hashlib.md5(clean_text.encode()).hexdigest()[:4], 16)
    selected_voice = voice_rotation[text_hash % len(voice_rotation)]
    
    print(f"      🎤 Ses: {selected_voice.split('-')[-1]} | {clean_text[:30]}...")

    # Edge-TTS ile doğal konuşma
    try:
        async def _edge_save_natural():
            # Daha doğal hız ve stil
            comm = edge_tts.Communicate(
                clean_text, 
                voice=selected_voice, 
                rate="+20%"  # Biraz daha yavaş (daha doğal)
            )
            await comm.save(mp3)

        try:
            asyncio.run(_edge_save_natural())
        except RuntimeError:
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_edge_save_natural())

        # WAV'a çevir + doğal ses filtreleri
        _run_ff([
            "-i", mp3, 
            "-ar", "44100",  
            "-ac", "1", 
            "-acodec", "pcm_s16le",
            "-af", "volume=0.88,highpass=f=85,lowpass=f=10000,dynaudnorm=g=5:f=200,acompressor=threshold=-18dB:ratio=3:attack=3:release=8",  # Daha doğal ses
            "-t", "6.0",  # ZORLA MAX 6s KES
            wav_out
        ])
        pathlib.Path(mp3).unlink(missing_ok=True)

        final_duration = _probe(wav_out, 3.0)
        print(f"      ✅ TTS: {final_duration:.1f}s")
        return final_duration

    except Exception as e:
        print(f"      ⚠️ Edge-TTS başarısız, basit Google TTS: {e}")
        # Google TTS fallback
        try:
            q = requests.utils.quote(clean_text.replace('"','').replace("'",""))
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={q}&tl={LANG or 'en'}&client=tw-ob&ttsspeed=0.85"
            headers = {"User-Agent":"Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=30); r.raise_for_status()
            open(mp3,"wb").write(r.content)
            
            _run_ff(["-i", mp3, "-ar","44100","-ac","1","-acodec","pcm_s16le",
                     "-af", "volume=0.88,dynaudnorm",
                     "-t", "6.0", wav_out])
            pathlib.Path(mp3).unlink(missing_ok=True)
            
            final_duration = _probe(wav_out, 3.0)
            print(f"      ✅ Google TTS: {final_duration:.1f}s")
            return final_duration

        except Exception as e2:
            print(f"      ❌ Tüm TTS başarısız: {e2}")
            _run_ff(["-f","lavfi","-t","3.0","-i","anullsrc=r=44100:cl=mono", wav_out])
            return 3.0

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
    """Süre kontrollü video segmenti oluştur (ABSOLUTE MAX 5s)"""
    # 20-40s toplam için segment başına ABSOLUTE MAX 5s
    dur = max(0.8, min(dur, 5.0))  # Minimum 0.8s, ABSOLUTE MAX 5s
    fade = max(0.05, min(0.12, dur/8))
    
    print(f"      📹 Segment: {dur:.1f}s (max 5s)")
    
    # Basit video filtreleri (hızlı işlem için)
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "eq=brightness=0.02:contrast=1.08:saturation=1.1,"  # Basit renk düzeltme
        f"fade=t=in:st=0:d={fade:.2f},"
        f"fade=t=out:st={max(0.0,dur-fade):.2f}:d={fade:.2f}"
    )
    
    run(["ffmpeg","-y","-i",src,"-t",f"{dur:.3f}","-vf",vf,"-r","25","-an",
         "-c:v","libx264","-preset","fast","-crf","22","-pix_fmt","yuv420p", outp])

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
    "_default": """Create a simple 20-40 second YouTube Short.
EXACTLY 5 sentences. Each sentence 4-8 words. Simple language, family-friendly.
Return JSON: country, topic, sentences, search_terms, title, description, tags.""",

    "country_facts": """Create amazing country facts.
EXACTLY 5 sentences about a specific country:
1. "Did you know [country]..." (surprising fact)
2. Geographic/cultural context
3. Amazing detail with simple numbers
4. Why it's special/unique
5. "Have you been to [country]?"
Search terms: country name, landmarks, culture, travel.""",

    "history_story": """Create forgotten historical stories.
EXACTLY 5 sentences about historical events:
1. "Long ago, something incredible happened..."
2. Setting and time period
3. What actually occurred
4. Why it was forgotten/hidden
5. "What other secrets are hidden?"
Search terms: historical, ancient, ruins, manuscripts.""",

    "quotes": """Create quote explanations.
EXACTLY 5 sentences about a famous quote:
1. "Someone once said..." (quote)
2. Who said it (or general context)
3. What it really means
4. How to apply it today
5. "How do you interpret this?"
Search terms: books, wisdom, philosophy, thinking.""",

    "taxwise_usa": """Create US tax education content.
EXACTLY 5 sentences about tax tips:
1. "Here's a tax tip..." 
2. Specific deduction or strategy
3. How much you could save
4. Important disclaimer about advice
5. "Always consult a tax professional."
Search terms: calculator, tax documents, office, money.""",

    "horror_story": """Create family-friendly mysterious stories.
EXACTLY 5 sentences for spooky atmosphere:
1. "Something strange happened..."
2. Setting and initial mystery
3. The mysterious discovery
4. Rational explanation or twist
5. "What would you have done?"
Search terms: mysterious, shadows, old house, fog.""",

    "daily_news": """Create global news summary.
EXACTLY 5 sentences about current events:
1. "Today's top story..."
2. What happened where
3. Why it matters globally
4. What experts are saying
5. "What's your take on this?"
Search terms: newsroom, world events, global, breaking.""",

    "space_news": """Create space discoveries content.
EXACTLY 5 sentences about space:
1. "Scientists just discovered..."
2. What they found in space
3. How far away it is
4. What this means for us
5. "What space mystery interests you?"
Search terms: space, rocket, planets, telescope, astronaut.""",

    "alt_universe": """Create alternate universe scenarios.
EXACTLY 5 sentences about different worlds:
1. "Imagine a world where..."
2. One major difference from ours
3. How people live there
4. What daily life looks like
5. "Would you visit this place?"
Search terms: universe, dimensions, cosmic, portal.""",

    "if_lived_today": """Create historical figures in modern times.
EXACTLY 5 sentences using generic archetypes:
1. "If an ancient inventor lived today..."
2. What they would create first
3. How they would use modern tools
4. Their biggest contribution now
5. "What would they invent next?"
Search terms: modern city, technology, innovation, future.""",

    "nostalgia_story": """Create nostalgic memories.
EXACTLY 5 sentences about past decades:
1. "Remember when we used..."
2. Specific old technology or trend
3. How different life was then
4. What we miss about it
5. "What do you miss most?"
Search terms: retro, vintage, old technology, nostalgia.""",

    "animal_facts": """Create amazing animal abilities.
EXACTLY 5 sentences about one specific animal:
1. "Did you know [animal] can..."
2. Specific amazing ability
3. How this ability works
4. Why they evolved this way
5. "Which animal amazes you most?"
Search terms: specific animal name, wildlife, nature, close up.""",

    "movie_secrets": """Create movie behind-the-scenes facts.
EXACTLY 5 sentences about film secrets:
1. "This famous movie scene..."
2. What went wrong during filming
3. How the director fixed it
4. Why it became iconic
5. "What's your favorite movie moment?"
Search terms: movie theater, film set, cinema, director.""",

    "tech_news": """Create technology breakthrough content.
EXACTLY 5 sentences about new tech:
1. "New technology can now..."
2. What it does specifically
3. How it will change life
4. When it will be available
5. "Are you excited about this?"
Search terms: technology, innovation, gadgets, AI, research.""",

    "utopic_tech": """Create positive future technology.
EXACTLY 5 sentences about beneficial tech:
1. "Future technology could..."
2. Specific helpful application
3. How it solves real problems
4. Making life better for everyone
5. "What would you want first?"
Search terms: futuristic, clean energy, helpful robots, smart city.""",

    "fame_story": """Create celebrity success stories using archetypes.
EXACTLY 5 sentences about generic success:
1. "One performer started with nothing..."
2. Their biggest early challenge
3. The breakthrough moment
4. How they stayed humble
5. "What drives your dreams?"
Search terms: stage lights, performance, success, inspiration.""",

    "post_apoc": """Create hopeful post-apocalyptic content.
EXACTLY 5 sentences focusing on resilience:
1. "After everything changed..."
2. How people adapted together
3. New skills they learned
4. Community they built
5. "What skills would you learn?"
Search terms: rebuilding, community, survival skills, hope.""",

    "mythology_battle": """Create mythological encounters.
EXACTLY 5 sentences about mythological scenarios:
1. "Ancient legends tell of..."
2. Epic encounter between mythical beings
3. Their different powers clashing
4. Unexpected outcome or lesson
5. "Which mythology interests you?"
Search terms: mythology, ancient temple, legends, divine.""",

    "kids_story": """Create educational children's content.
EXACTLY 5 sentences for young minds:
1. "Once there was a little..."
2. Simple problem they faced
3. Creative solution they found
4. What they learned
5. "What would you do?"
Search terms: children, learning, imagination, colorful, fun.""",

    "ai_alt": """Create positive AI integration scenarios.
EXACTLY 5 sentences about helpful AI:
1. "Imagine if AI could..."
2. Specific helpful task
3. How it makes life easier
4. People working together with AI
5. "How could AI help you?"
Search terms: helpful robots, AI assistant, automation, smart home.""",

    "ai_future": """Create educational AI development content.
EXACTLY 5 sentences about AI research:
1. "Scientists are developing AI that..."
2. Current research breakthrough
3. Potential positive applications
4. Ethical considerations being studied
5. "What AI development excites you?"
Search terms: AI laboratory, research, neural networks, innovation.""",

    "fixit_fast": """Create quick repair tips.
EXACTLY 5 sentences about simple fixes:
1. "Here's a quick fix for..."
2. Common household problem
3. Simple tool you need
4. Step-by-step solution
5. "What do you fix yourself?"
Search terms: tools, repair, DIY, workshop, fixing.""",

    "sports_news": """Create daily sports updates.
EXACTLY 5 sentences about sports:
1. "Today in sports..."
2. Major game or achievement
3. Record broken or milestone
4. What it means for the sport
5. "Which sport do you follow?"
Search terms: sports stadium, athletes, championship, training.""",

    "cricket_women": """Create women's cricket content.
EXACTLY 5 sentences about women's cricket:
1. "Women's cricket just achieved..."
2. Specific achievement or record
3. Player who made it happen
4. Impact on the sport
5. "Do you follow women's cricket?"
Search terms: women cricket, female athletes, cricket stadium, champions."""
}

def build_via_gemini(mode: str, channel_name: str, banlist: List[str], channel_config: dict = {}) -> tuple:
    """Kanal bazlı Gemini entegrasyonu - 25 kanala özel içerik"""
    
    # Kanal konfigürasyonundan template al
    template = ENHANCED_GEMINI_TEMPLATES.get(mode, ENHANCED_GEMINI_TEMPLATES["_default"])
    
    # Kanal özel topic ve search terms
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

CHANNEL-SPECIFIC REQUIREMENTS:
- Content MUST match the channel's theme: {channel_topic}
- Follow this focus: {content_focus}
- EXACTLY 5 sentences (no more, no less!)
- Each sentence 4-8 words maximum  
- Simple language, family-friendly
- End with engaging question

Return ONLY valid JSON:
{{
  "country": "<location or theme>",
  "topic": "<channel-themed title>", 
  "sentences": ["<exactly 5 channel-themed sentences>"],
  "search_terms": ["<4-6 terms matching channel theme>"],
  "title": "<engaging title under 80 chars>",
  "description": "<simple description 500-800 chars>",
  "tags": ["<10 relevant tags>"]
}}
"""

    try:
        data = _gemini_call(enhanced_prompt, GEMINI_MODEL)
        
        country = str(data.get("country") or channel_config.get('topic', 'World')).strip()
        topic = str(data.get("topic") or channel_topic or "Amazing Facts").strip()
        
        sentences = [clean_caption_text(s) for s in (data.get("sentences") or [])]
        sentences = [s for s in sentences if s]
        
        # STRICT: Sadece 5 cümle
        if len(sentences) < 5:
            generic_endings = [
                "This discovery amazes experts worldwide.",
                "What do you think about this?"
            ]
            sentences.extend(generic_endings)
        sentences = sentences[:5]  # EXACTLY 5
        
        print(f"✅ Gemini kanal temalı içerik: {topic}")
        
        # Search terms - önce channel config, sonra Gemini response
        terms = channel_search_terms or data.get("search_terms") or []
        if isinstance(terms, str):
            terms = [terms]
        terms = [t.strip() for t in terms if isinstance(t, str) and t.strip()]
        
        if not terms:
            # Mode bazlı fallback terms
            mode_terms = {
                "country_facts": ["world travel 4k", "cultural heritage", "landmarks", "city skyline"],
                "history_story": ["ancient ruins 4k", "historical", "manuscripts", "archaeology"],
                "quotes": ["books 4k", "wisdom", "philosophy", "thinking person"],
                "movie_secrets": ["movie theater 4k", "film set", "cinema", "director"],
                "animal_facts": ["wildlife 4k", "animal close up", "nature", "safari"],
                "space_news": ["space 4k", "rocket launch", "planets", "astronaut"],
                "tech_news": ["technology 4k", "innovation", "gadgets", "research lab"]
            }
            terms = mode_terms.get(mode, ["documentary 4k", "education", "discovery", "science"])
        
        title = (data.get("title") or "").strip()
        description = (data.get("description") or "").strip()
        tags = data.get("tags") or []
        tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
        
        return country, topic, sentences, terms, title, description, tags
        
    except Exception as e:
        print(f"⚠️ Gemini başarısız, kanal fallback: {e}")
        
        # Kanal config'inden fallback al
        if channel_config:
            topic = channel_config.get('topic', 'Channel Content')
            terms = channel_config.get('search_terms', ['general 4k', 'education'])
            
            # Mode bazlı fallback sentences
            mode_sentences = {
                "country_facts": [
                    "This country has amazing secrets.",
                    "Hidden facts will surprise you completely.",
                    "Culture and nature blend perfectly here.",
                    "History shaped this unique place.",
                    "Which country fascinates you most?"
                ],
                "animal_facts": [
                    "This animal has incredible abilities.",
                    "Nature designed perfect survival skills.",
                    "These creatures amaze scientists daily.",
                    "Evolution created amazing adaptations.",
                    "Which animal surprises you most?"
                ],
                "movie_secrets": [
                    "Movies hide incredible behind-scenes secrets.",
                    "Directors create magic through clever tricks.",
                    "Famous scenes happened by accident.",
                    "Cinema history contains amazing stories.",
                    "What movie secret amazed you?"
                ]
            }
            
            sentences = mode_sentences.get(mode, [
                "Amazing discoveries happen every day.",
                "Science reveals incredible new facts.",
                "These secrets will surprise you.",
                "Knowledge keeps expanding constantly.",
                "What interests you most about this?"
            ])
            
            return "World", topic, sentences, terms, "", "", []
        
        # Son fallback
        return ("World", "Daily Facts", [
            "Did you know this amazing fact?",
            "Scientists make discoveries every day.",
            "This will definitely surprise you.",
            "Knowledge never stops growing.",
            "What do you think?"
        ], ["science 4k", "discovery", "education", "research"], "", "", [])

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
    """Agresif metin temizleme - MAX 80 karakter per sentence"""
    t = (s or "").strip().replace("'","'").replace("—","-").replace('"',"").replace("`","")
    t = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', t)
    t = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', t)
    t = re.sub(r'\s+',' ', t)
    
    # Sayısal detayları basitleştir (TTS sorunları için)
    t = re.sub(r'\d{2,}', lambda m: "many" if int(m.group()) > 100 else m.group(), t)
    t = re.sub(r'\d+\s*-\s*meter', "massive", t)
    t = re.sub(r'\d+\s*years?', "decades", t)
    
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    
    # AGRESIF uzunluk kısıtlaması
    if len(t) > 80:  # MAX 80 karakter
        words = t.split()
        t = " ".join(words[:12]) + "."  # MAX 12 kelime
    
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
    """Ses dosyalarını birleştir - MAX 40s toplam"""
    # Önce toplam süreyi kontrol et
    total_dur = sum(ffprobe_dur(f) for f in files)
    print(f"📊 Audio toplam süre: {total_dur:.1f}s")
    
    if total_dur > TARGET_MAX_SEC:
        print(f"⚠️ Audio çok uzun ({total_dur:.1f}s), kesiliyor...")
        # Her dosyayı proporsiyone göre kısalt
        ratio = TARGET_MAX_SEC / total_dur
        temp_files = []
        
        for i, f in enumerate(files):
            if total_dur <= TARGET_MAX_SEC:
                temp_files.append(f)
                continue
                
            orig_dur = ffprobe_dur(f)
            new_dur = min(orig_dur * ratio, 6.0)  # Max 6s per segment
            
            if new_dur < 1.0:  # Çok kısa olursa atla
                continue
                
            temp_f = f.replace(".wav", f"_cut_{i}.wav")
            run(["ffmpeg","-y","-i",f,"-t",f"{new_dur:.2f}","-c","copy",temp_f])
            temp_files.append(temp_f)
        
        files = temp_files[:6]  # Max 6 segment
    
    lst = str(pathlib.Path(outp).with_suffix(".txt"))
    with open(lst,"w") as f:
        for p in files[:6]:  # MAX 6 dosya
            f.write(f"file '{p}'\n")
    
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",lst,
         "-af","volume=0.9,dynaudnorm","-t",f"{TARGET_MAX_SEC}",outp])  # ZORLA KES
    
    # Temp dosyaları temizle
    for f in files:
        if "_cut_" in f:
            pathlib.Path(f).unlink(missing_ok=True)

def mux(video: str, audio: str, outp: str):
    """Güvenli video/audio birleştirme - FFmpeg hata önleme"""
    try:
        # Önce süreleri kontrol et
        video_dur = ffprobe_dur(video)
        audio_dur = ffprobe_dur(audio)
        
        print(f"🔍 Video: {video_dur:.1f}s | Audio: {audio_dur:.1f}s")
        
        # Süre uyumsuzluğu varsa düzelt
        if abs(video_dur - audio_dur) > 1.0:
            print("⚠️ Video/Audio süre uyumsuzluğu düzeltiliyor...")
            min_dur = min(video_dur, audio_dur, 45.0)  # Max 45s
            
            # Video'yu kırp
            temp_video = video.replace(".mp4", "_temp.mp4")
            run(["ffmpeg","-y","-i",video,"-t",f"{min_dur:.2f}","-c","copy",temp_video])
            
            # Audio'yu kırp  
            temp_audio = audio.replace(".wav", "_temp.wav")
            run(["ffmpeg","-y","-i",audio,"-t",f"{min_dur:.2f}","-c","copy",temp_audio])
            
            video, audio = temp_video, temp_audio
        
        # Güvenli birleştirme
        run(["ffmpeg","-y","-i",video,"-i",audio,
             "-map","0:v:0","-map","1:a:0",
             "-c:v","copy","-c:a","aac","-b:a","256k",
             "-movflags","+faststart",
             "-shortest",  # Kısa olanı kullan
             "-avoid_negative_ts","make_zero",  # Sync problemi önleme
             outp])
        
        # Temp dosyaları temizle
        for temp_file in [video, audio]:
            if "_temp" in temp_file:
                pathlib.Path(temp_file).unlink(missing_ok=True)
                
    except Exception as e:
        print(f"⚠️ Mux hatası: {e}")
        # Son çare - basit copy
        run(["ffmpeg","-y","-i",video,"-i",audio,"-c","copy","-shortest",outp])

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
