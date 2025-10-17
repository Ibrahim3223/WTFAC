"""
Settings module - Centralized configuration
Loads settings from environment variables with sensible defaults
"""

import os
import re
from typing import List


def _env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.getenv(key, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _parse_list(s: str) -> List[str]:
    """Parse comma-separated list from string."""
    s = (s or "").strip()
    if not s:
        return []
    # Try JSON first
    try:
        import json
        data = json.loads(s)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    # Fallback to comma-separated
    s = re.sub(r'^[\[\(]|\s*[\]\)]$', '', s)
    parts = re.split(r'\s*,\s*', s)
    return [p.strip().strip('"').strip("'") for p in parts if p.strip()]


# ============================================================
# CHANNEL CONFIGURATION
# ============================================================

CHANNEL_NAME = os.getenv("CHANNEL_NAME", "DefaultChannel")
CHANNEL_TOPIC = os.getenv("TOPIC", "Interesting facts and knowledge")
CONTENT_STYLE = os.getenv("CONTENT_STYLE", "Educational and engaging")

# ============================================================
# API KEYS (from environment)
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")

# YouTube OAuth
YT_CLIENT_ID = os.getenv("YT_CLIENT_ID", "")
YT_CLIENT_SECRET = os.getenv("YT_CLIENT_SECRET", "")
YT_REFRESH_TOKEN = os.getenv("YT_REFRESH_TOKEN", "")

# ============================================================
# GEMINI SETTINGS
# ============================================================

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "flash")  # Default: gemini-2.5-flash (via mapping)
USE_GEMINI = _env_bool("USE_GEMINI", True)
ADDITIONAL_PROMPT_CONTEXT = os.getenv("ADDITIONAL_PROMPT_CONTEXT", "")

# ============================================================
# VIDEO SETTINGS
# ============================================================

TARGET_DURATION = _env_int("TARGET_DURATION", 30)  # seconds
TARGET_MIN_SEC = _env_float("TARGET_MIN_SEC", 25.0)
TARGET_MAX_SEC = _env_float("TARGET_MAX_SEC", 35.0)

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
TARGET_FPS = _env_int("TARGET_FPS", 30)
CRF_VISUAL = _env_int("CRF_VISUAL", 20)

# Video motion effects
VIDEO_MOTION = _env_bool("VIDEO_MOTION", True)  # Enable Ken Burns and motion effects
MOTION_INTENSITY = _env_float("MOTION_INTENSITY", 1.18)  # Zoom intensity for Ken Burns (1.0 = no zoom, 1.2 = 20% zoom)

# ============================================================
# TTS SETTINGS
# ============================================================

TTS_VOICE = os.getenv("TTS_VOICE", "en-US-GuyNeural")
VOICE = TTS_VOICE  # Alias for backward compatibility
TTS_RATE = os.getenv("TTS_RATE", "+0%")
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz")
TTS_STYLE = os.getenv("TTS_STYLE", "narration-professional")

# ============================================================
# PEXELS/PIXABAY SETTINGS
# ============================================================

PEXELS_PER_PAGE = _env_int("PEXELS_PER_PAGE", 40)
PEXELS_MAX_USES_PER_CLIP = _env_int("PEXELS_MAX_USES_PER_CLIP", 1)
PEXELS_ALLOW_REUSE = _env_bool("PEXELS_ALLOW_REUSE", False)
PEXELS_ALLOW_LANDSCAPE = _env_bool("PEXELS_ALLOW_LANDSCAPE", False)
PEXELS_MIN_DURATION = _env_int("PEXELS_MIN_DURATION", 4)
PEXELS_MAX_DURATION = _env_int("PEXELS_MAX_DURATION", 12)
PEXELS_MIN_HEIGHT = _env_int("PEXELS_MIN_HEIGHT", 1440)
PEXELS_STRICT_VERTICAL = _env_bool("PEXELS_STRICT_VERTICAL", True)

ALLOW_PIXABAY_FALLBACK = _env_bool("ALLOW_PIXABAY_FALLBACK", True)

# Entity filtering for video search
STRICT_ENTITY_FILTER = _env_bool("STRICT_ENTITY_FILTER", False)

# ============================================================
# CAPTION SETTINGS
# ============================================================

CAPTION_FONT = os.getenv("CAPTION_FONT", "Arial")
CAPTION_FONT_SIZE = _env_int("CAPTION_FONT_SIZE", 70)
CAPTION_MAX_LINE = _env_int("CAPTION_MAX_LINE", 26)
CAPTION_MAX_LINES = _env_int("CAPTION_MAX_LINES", 5)
CAPTION_POSITION = os.getenv("CAPTION_POSITION", "center")  # top, center, bottom

# Karaoke effect colors
CAPTION_PRIMARY_COLOR = os.getenv("CAPTION_PRIMARY_COLOR", "&H00FFFFFF")  # White
CAPTION_OUTLINE_COLOR = os.getenv("CAPTION_OUTLINE_COLOR", "&H00000000")  # Black
CAPTION_HIGHLIGHT_COLOR = os.getenv("CAPTION_HIGHLIGHT_COLOR", "&H0000FFFF")  # Yellow

# ============================================================
# BGM SETTINGS
# ============================================================

BGM_ENABLE = _env_bool("BGM_ENABLE", True)
BGM_VOLUME_DB = _env_float("BGM_DB", -26.0)
BGM_DUCK_DB = _env_float("BGM_DUCK_DB", -12.0)
BGM_FADE_DURATION = _env_float("BGM_FADE", 0.8)
BGM_DIR = os.getenv("BGM_DIR", "bgm")
BGM_URLS = _parse_list(os.getenv("BGM_URLS", ""))

# ============================================================
# STATE MANAGEMENT
# ============================================================

STATE_DIR = os.getenv("STATE_DIR", "state")
ENTITY_COOLDOWN_DAYS = _env_int("ENTITY_COOLDOWN_DAYS", 30)

# Novelty settings
NOVELTY_ENFORCE = _env_bool("NOVELTY_ENFORCE", True)
NOVELTY_WINDOW = _env_int("NOVELTY_WINDOW", 50)
NOVELTY_JACCARD_MAX = _env_float("NOVELTY_JACCARD_MAX", 0.48)
NOVELTY_RETRIES = _env_int("NOVELTY_RETRIES", 6)

# ============================================================
# QUALITY SETTINGS
# ============================================================

# Note: Lower threshold for more content acceptance
# Quality scores range from 0-10, where:
# - 5.0+ = Good quality
# - 6.5+ = High quality  
# - 8.0+ = Excellent quality
MIN_QUALITY_SCORE = _env_float("MIN_QUALITY_SCORE", 5.0)
MAX_GENERATION_ATTEMPTS = _env_int("MAX_GENERATION_ATTEMPTS", 4)

# ============================================================
# UPLOAD SETTINGS
# ============================================================

UPLOAD_TO_YT = _env_bool("UPLOAD_TO_YT", True)
VISIBILITY = os.getenv("VISIBILITY", "public")  # public, unlisted, private

# ============================================================
# LANGUAGE SETTINGS
# ============================================================

LANG = os.getenv("LANG", "en")

# ============================================================
# OUTPUT SETTINGS
# ============================================================

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "out")

# Create necessary directories
import pathlib
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
if BGM_ENABLE and BGM_DIR:
    pathlib.Path(BGM_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================
# VALIDATION
# ============================================================

def validate_settings():
    """Validate critical settings."""
    errors = []
    
    if not CHANNEL_NAME:
        errors.append("CHANNEL_NAME is required")
    
    if USE_GEMINI and not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is required when USE_GEMINI is enabled")
    
    if not PEXELS_API_KEY and not PIXABAY_API_KEY:
        errors.append("Either PEXELS_API_KEY or PIXABAY_API_KEY is required")
    
    if UPLOAD_TO_YT:
        if not YT_CLIENT_ID or not YT_CLIENT_SECRET or not YT_REFRESH_TOKEN:
            errors.append("YouTube credentials (YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN) are required for upload")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))


# Auto-validate on import (optional - can be disabled)
if os.getenv("VALIDATE_SETTINGS", "1") == "1":
    try:
        validate_settings()
    except ValueError as e:
        import logging
        logging.warning(f"Settings validation warning: {e}")
