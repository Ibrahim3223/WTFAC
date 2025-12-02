"""
Settings module - NEW Pydantic-based configuration (backward compatible).

This module provides backward-compatible access to all settings while using
Pydantic models internally for validation and type safety.
"""

import os
import logging
from typing import List

from .models import AppConfig

logger = logging.getLogger(__name__)

# ============================================================
# Load configuration from environment
# ============================================================

try:
    config = AppConfig()
    logger.info("✅ Configuration loaded successfully")
except Exception as e:
    logger.error(f"❌ Configuration error: {e}")
    raise

# ============================================================
# Backward-compatible exports
# ============================================================

# API Keys
GEMINI_API_KEY = config.api.gemini_api_key
PEXELS_API_KEY = config.api.pexels_api_key
PIXABAY_API_KEY = config.api.pixabay_api_key
YT_CLIENT_ID = config.api.yt_client_id
YT_CLIENT_SECRET = config.api.yt_client_secret
YT_REFRESH_TOKEN = config.api.yt_refresh_token

# Gemini
GEMINI_MODEL = config.api.gemini_model
USE_GEMINI = True  # Always true for now
ADDITIONAL_PROMPT_CONTEXT = os.getenv("ADDITIONAL_PROMPT_CONTEXT", "")

# Channel
CHANNEL_NAME = config.channel.channel_name
CHANNEL_TOPIC = config.channel.topic
CHANNEL_MODE = config.channel.mode
LANG = config.channel.lang
VISIBILITY = config.channel.visibility
CONTENT_STYLE = config.channel.content_style
ROTATION_SEED = config.channel.rotation_seed

# Try to load channel-specific settings from channels.yml
try:
    from .channel_loader import apply_channel_settings
    _channel_settings = apply_channel_settings(CHANNEL_NAME)
    CHANNEL_TOPIC = _channel_settings.get("CHANNEL_TOPIC", CHANNEL_TOPIC)
    CHANNEL_MODE = _channel_settings.get("CHANNEL_MODE", CHANNEL_MODE)
    CHANNEL_SEARCH_TERMS = _channel_settings.get("CHANNEL_SEARCH_TERMS", [])
except Exception as e:
    logger.warning(f"⚠️ Failed to load channel config: {e}")
    CHANNEL_SEARCH_TERMS = []

# Allow ENV override if specified
if os.getenv("TOPIC"):
    CHANNEL_TOPIC = os.getenv("TOPIC")

# Video
TARGET_DURATION = config.video.target_duration
TARGET_MIN_SEC = config.video.target_min_sec
TARGET_MAX_SEC = config.video.target_max_sec
VIDEO_WIDTH = config.video.video_width
VIDEO_HEIGHT = config.video.video_height
TARGET_FPS = config.video.target_fps
CRF_VISUAL = config.video.crf_visual
VIDEO_MOTION = config.video.video_motion
MOTION_INTENSITY = config.video.motion_intensity

# TTS
TTS_VOICE = config.tts.voice
VOICE = TTS_VOICE  # Alias
TTS_RATE = config.tts.rate
TTS_PITCH = config.tts.pitch
TTS_STYLE = config.tts.style
TTS_SSML = config.tts.ssml

# Captions
REQUIRE_CAPTIONS = config.captions.require_captions
KARAOKE_CAPTIONS = config.captions.karaoke_captions
CAPTIONS_UPPER = config.captions.captions_upper
KARAOKE_INACTIVE = config.captions.karaoke_inactive
KARAOKE_ACTIVE = config.captions.karaoke_active
KARAOKE_OUTLINE = config.captions.karaoke_outline
KARAOKE_OFFSET_MS = config.captions.karaoke_offset_ms
KARAOKE_SPEED = config.captions.karaoke_speed
CAPTION_LEAD_MS = config.captions.caption_lead_ms
KARAOKE_EFFECTS = config.captions.karaoke_effects
EFFECT_STYLE = config.captions.effect_style

# Pexels
PEXELS_PER_PAGE = config.pexels.per_page
PEXELS_MAX_USES_PER_CLIP = config.pexels.max_uses_per_clip
PEXELS_ALLOW_REUSE = config.pexels.allow_reuse
PEXELS_ALLOW_LANDSCAPE = config.pexels.allow_landscape
PEXELS_MIN_DURATION = config.pexels.min_duration
PEXELS_MAX_DURATION = config.pexels.max_duration
PEXELS_MIN_HEIGHT = config.pexels.min_height
PEXELS_STRICT_VERTICAL = config.pexels.strict_vertical
PEXELS_MAX_PAGES = config.pexels.max_pages
PEXELS_DEEP_SEARCH = config.pexels.deep_search
ALLOW_PIXABAY_FALLBACK = config.pexels.allow_pixabay_fallback

# Quality
MIN_QUALITY_SCORE = config.quality.min_quality_score
MIN_VIRAL_SCORE = config.quality.min_viral_score
MIN_OVERALL_SCORE = config.quality.min_overall_score

# Novelty
NOVELTY_ENFORCE = config.novelty.enforce
NOVELTY_WINDOW = config.novelty.window
NOVELTY_JACCARD_MAX = config.novelty.jaccard_max
NOVELTY_RETRIES = config.novelty.retries
ENTITY_COOLDOWN_DAYS = config.novelty.entity_cooldown_days
STATE_DIR = config.novelty.state_dir

# BGM
BGM_ENABLE = config.bgm.enable
BGM_DIR = config.bgm.dir
BGM_FADE = config.bgm.fade
BGM_GAIN_DB = config.bgm.gain_db
BGM_DUCK_THRESH = config.bgm.duck_thresh
BGM_DUCK_RATIO = config.bgm.duck_ratio
BGM_DUCK_ATTACK_MS = config.bgm.duck_attack_ms
BGM_DUCK_RELEASE_MS = config.bgm.duck_release_ms

# App
UPLOAD_TO_YT = config.upload_to_yt
MAX_GENERATION_ATTEMPTS = config.max_generation_attempts

# ============================================================
# Legacy compatibility - values from old settings.py
# ============================================================

# These are kept for backward compatibility with existing code
SCENE_STRATEGY = os.getenv("SCENE_STRATEGY", "topic_only")
ENTITY_VISUAL_MIN = float(os.getenv("ENTITY_VISUAL_MIN", "0.95"))
ENTITY_VISUAL_STRICT = os.getenv("ENTITY_VISUAL_STRICT", "1") == "1"
STRICT_ENTITY_FILTER = os.getenv("STRICT_ENTITY_FILTER", "1") == "1"
HOOK_MAX_WORDS = int(os.getenv("HOOK_MAX_WORDS", "8"))
CTA_ENABLE = os.getenv("CTA_ENABLE", "1") == "1"
CTA_SHOW_SEC = float(os.getenv("CTA_SHOW_SEC", "2.8"))
CTA_MAX_CHARS = int(os.getenv("CTA_MAX_CHARS", "64"))
SEO_KEYWORD_DENSITY = os.getenv("SEO_KEYWORD_DENSITY", "1") == "1"
TITLE_POWER_WORDS = os.getenv("TITLE_POWER_WORDS", "1") == "1"
MAX_DESCRIPTION_LENGTH = int(os.getenv("MAX_DESCRIPTION_LENGTH", "4900"))
MAX_TAGS = int(os.getenv("MAX_TAGS", "30"))

# Caption offset (for CaptionRenderer compatibility)
CAPTION_OFFSET = None
try:
    offset_str = os.getenv("KARAOKE_OFFSET_MS", "0")
    CAPTION_OFFSET = int(offset_str) / 1000.0 if offset_str else None
except Exception:
    CAPTION_OFFSET = None

# ============================================================
# Validation helpers
# ============================================================

def validate_api_keys() -> List[str]:
    """
    Validate that required API keys are present.

    Returns:
        List of missing API key names
    """
    missing = []

    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if not PEXELS_API_KEY and not PIXABAY_API_KEY:
        missing.append("PEXELS_API_KEY or PIXABAY_API_KEY")

    if UPLOAD_TO_YT:
        if not YT_CLIENT_ID:
            missing.append("YT_CLIENT_ID")
        if not YT_CLIENT_SECRET:
            missing.append("YT_CLIENT_SECRET")
        if not YT_REFRESH_TOKEN:
            missing.append("YT_REFRESH_TOKEN")

    return missing


def get_config() -> AppConfig:
    """Get the full typed configuration object."""
    return config
