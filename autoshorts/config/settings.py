# -*- coding: utf-8 -*-
"""
Configuration settings from environment variables.
All ENV parsing in one place for easy management.
"""
import os
import re
from typing import Optional

# ==================== Helper Functions ====================

def _env_int(name: str, default: int) -> int:
    """Parse ENV as integer with fallback."""
    s = os.getenv(name)
    if s is None: 
        return default
    s = str(s).strip()
    if s == "" or s.lower() == "none": 
        return default
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))  # Handle "68.0" format
        except Exception:
            return default

def _env_float(name: str, default: float) -> float:
    """Parse ENV as float with fallback."""
    s = os.getenv(name)
    if s is None: 
        return default
    s = str(s).strip()
    if s == "" or s.lower() == "none": 
        return default
    try:
        return float(s)
    except Exception:
        return default

def _env_bool(name: str, default: bool = False) -> bool:
    """Parse ENV as boolean."""
    val = os.getenv(name, "").strip()
    return val == "1" or val.lower() == "true"

def _sanitize_lang(val: Optional[str]) -> str:
    """Extract 2-letter language code."""
    val = (val or "").strip()
    if not val: 
        return "en"
    m = re.match(r"([A-Za-z]{2})", val)
    return (m.group(1).lower() if m else "en")

def _sanitize_privacy(val: Optional[str]) -> str:
    """Validate privacy status."""
    v = (val or "").strip().lower()
    return v if v in {"public", "unlisted", "private"} else "public"

# ==================== Channel & Content ====================

CHANNEL_NAME = os.getenv("CHANNEL_NAME", "DefaultChannel")
MODE = os.getenv("MODE", "freeform").strip().lower()
TOPIC = os.getenv("TOPIC", "").strip()
LANG = _sanitize_lang(os.getenv("VIDEO_LANG") or os.getenv("LANG") or "en")
VISIBILITY = _sanitize_privacy(os.getenv("VISIBILITY"))
ROTATION_SEED = _env_int("ROTATION_SEED", 0)

# ==================== TTS Settings ====================

VOICE = os.getenv("TTS_VOICE", "")  # Will be selected based on LANG
VOICE_STYLE = os.getenv("TTS_STYLE", "narration-professional")
TTS_RATE = os.getenv("TTS_RATE", "+12%")
TTS_SSML = _env_bool("TTS_SSML", False)

# ==================== Video Timing ====================

TARGET_MIN_SEC = _env_float("TARGET_MIN_SEC", 22.0)
TARGET_MAX_SEC = _env_float("TARGET_MAX_SEC", 42.0)
TARGET_FPS = 25

# ==================== Captions ====================

REQUIRE_CAPTIONS = _env_bool("REQUIRE_CAPTIONS", False)
KARAOKE_CAPTIONS = _env_bool("KARAOKE_CAPTIONS", True)
CAPTIONS_UPPER = _env_bool("CAPTIONS_UPPER", True)

KARAOKE_INACTIVE = os.getenv("KARAOKE_INACTIVE", "#FFD700")
KARAOKE_ACTIVE = os.getenv("KARAOKE_ACTIVE", "#3EA6FF")
KARAOKE_OUTLINE = os.getenv("KARAOKE_OUTLINE", "#000000")

KARAOKE_OFFSET_MS = _env_int("KARAOKE_OFFSET_MS", 50)
KARAOKE_SPEED = _env_float("KARAOKE_SPEED", 1.0)
CAPTION_LEAD_MS = _env_int("CAPTION_LEAD_MS", 0)

KARAOKE_EFFECTS = _env_bool("KARAOKE_EFFECTS", True)
EFFECT_STYLE = os.getenv("EFFECT_STYLE", "moderate").lower()
VIDEO_MOTION = _env_bool("VIDEO_MOTION", True)
MOTION_INTENSITY = os.getenv("MOTION_INTENSITY", "subtle").lower()

CAPTION_MAX_LINE = _env_int("CAPTION_MAX_LINE", 28)
CAPTION_MAX_LINES = _env_int("CAPTION_MAX_LINES", 6)

# ==================== Video Quality ====================

CRF_VISUAL = 22

# ==================== Gemini AI ====================

USE_GEMINI = _env_bool("USE_GEMINI", True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# ⭐ FIX: Use correct Gemini model name
# gemini-2.5-flash doesn't exist - using stable model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

GEMINI_PROMPT = os.getenv("GEMINI_PROMPT", "").strip()
GEMINI_TEMP = _env_float("GEMINI_TEMP", 0.85)

# ==================== Quality Thresholds ====================

MIN_QUALITY_SCORE = _env_float("MIN_QUALITY_SCORE", 6.5)
MIN_VIRAL_SCORE = _env_float("MIN_VIRAL_SCORE", 6.0)
MIN_OVERALL_SCORE = _env_float("MIN_OVERALL_SCORE", 7.0)

# ==================== Novelty & Cooldown ====================

NOVELTY_ENFORCE = _env_bool("NOVELTY_ENFORCE", True)
NOVELTY_WINDOW = _env_int("NOVELTY_WINDOW", 40)
NOVELTY_JACCARD_MAX = _env_float("NOVELTY_JACCARD_MAX", 0.55)
NOVELTY_RETRIES = _env_int("NOVELTY_RETRIES", 4)

ENTITY_COOLDOWN_DAYS = _env_int("ENTITY_COOLDOWN_DAYS", 45)
SIM_TH_SCRIPT = _env_float("SIM_TH_SCRIPT", 0.92)
SIM_TH_ENTITY = _env_float("SIM_TH_ENTITY", 0.94)

STATE_DIR = os.getenv("STATE_DIR", ".state")

# ==================== Focus & Entity ====================

HOOK_MAX_WORDS = _env_int("HOOK_MAX_WORDS", 8)
STRICT_ENTITY_FILTER = _env_bool("STRICT_ENTITY_FILTER", True)
ENTITY_VISUAL_MIN = _env_float("ENTITY_VISUAL_MIN", 0.95)
ENTITY_VISUAL_STRICT = _env_bool("ENTITY_VISUAL_STRICT", True)

# ==================== Pexels ====================

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "").strip()
PEXELS_PER_PAGE = _env_int("PEXELS_PER_PAGE", 80)
PEXELS_MAX_USES_PER_CLIP = _env_int("PEXELS_MAX_USES_PER_CLIP", 1)
PEXELS_ALLOW_REUSE = _env_bool("PEXELS_ALLOW_REUSE", False)
PEXELS_ALLOW_LANDSCAPE = _env_bool("PEXELS_ALLOW_LANDSCAPE", True)
PEXELS_MIN_DURATION = _env_int("PEXELS_MIN_DURATION", 3)
PEXELS_MAX_DURATION = _env_int("PEXELS_MAX_DURATION", 13)
PEXELS_MIN_HEIGHT = _env_int("PEXELS_MIN_HEIGHT", 720)
PEXELS_STRICT_VERTICAL = _env_bool("PEXELS_STRICT_VERTICAL", False)
PEXELS_DEEP_SEARCH = _env_bool("PEXELS_DEEP_SEARCH", True)
PEXELS_MAX_PAGES = _env_int("PEXELS_MAX_PAGES", 7)

SCENE_STRATEGY = os.getenv("SCENE_STRATEGY", "topic_only").strip().lower()
SCENE_QUERY_MODE = os.getenv("SCENE_QUERY_MODE", "entity").strip().lower()

# ==================== Pixabay ====================

ALLOW_PIXABAY_FALLBACK = _env_bool("ALLOW_PIXABAY_FALLBACK", True)
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "").strip()

# ==================== SEO & Metadata ====================

SEO_KEYWORD_DENSITY = _env_bool("SEO_KEYWORD_DENSITY", True)
TITLE_POWER_WORDS = _env_bool("TITLE_POWER_WORDS", True)
MAX_DESCRIPTION_LENGTH = _env_int("MAX_DESCRIPTION_LENGTH", 4900)
MAX_TAGS = _env_int("MAX_TAGS", 30)

# ==================== CTA ====================

CTA_ENABLE = _env_bool("CTA_ENABLE", True)
CTA_SHOW_SEC = _env_float("CTA_SHOW_SEC", 2.8)
CTA_MAX_CHARS = _env_int("CTA_MAX_CHARS", 64)
CTA_TEXT_FORCE = os.getenv("CTA_TEXT", "").strip()

# ==================== BGM ====================

BGM_ENABLE = _env_bool("BGM_ENABLE", False)
BGM_DIR = os.getenv("BGM_DIR", "bgm").strip()
BGM_FADE = _env_float("BGM_FADE", 0.8)
BGM_GAIN_DB = _env_float("BGM_GAIN_DB", -11.0)
BGM_DUCK_THRESH = _env_float("BGM_DUCK_THRESH", 0.035)
BGM_DUCK_RATIO = _env_float("BGM_DUCK_RATIO", 10.0)
BGM_DUCK_ATTACK_MS = _env_int("BGM_DUCK_ATTACK_MS", 6)
BGM_DUCK_RELEASE_MS = _env_int("BGM_DUCK_RELEASE_MS", 180)

# Parse BGM_URLS from ENV
_bgm_urls_raw = os.getenv("BGM_URLS", "").strip()
if _bgm_urls_raw:
    try:
        import json
        BGM_URLS = json.loads(_bgm_urls_raw)
        if not isinstance(BGM_URLS, list):
            BGM_URLS = []
    except:
        BGM_URLS = [u.strip() for u in _bgm_urls_raw.split(',') if u.strip()]
else:
    BGM_URLS = []

# ==================== YouTube ====================

UPLOAD_TO_YT = _env_bool("UPLOAD_TO_YT", True)
YT_CLIENT_ID = os.getenv("YT_CLIENT_ID", "").strip()
YT_CLIENT_SECRET = os.getenv("YT_CLIENT_SECRET", "").strip()
YT_REFRESH_TOKEN = os.getenv("YT_REFRESH_TOKEN", "").strip()

# ==================== Output ====================

OUT_DIR = "out"

# ==================== Voice Options ====================

VOICE_OPTIONS = {
    "en": [
        "en-US-JennyNeural", "en-US-JasonNeural", "en-US-AriaNeural", 
        "en-US-GuyNeural", "en-AU-NatashaNeural", "en-GB-SoniaNeural", 
        "en-CA-LiamNeural", "en-US-DavisNeural", "en-US-AmberNeural"
    ],
    "tr": ["tr-TR-EmelNeural", "tr-TR-AhmetNeural"]
}

# Select voice based on language if not specified
if not VOICE:
    available = VOICE_OPTIONS.get(LANG, ["en-US-JennyNeural"])
    VOICE = available[0]
