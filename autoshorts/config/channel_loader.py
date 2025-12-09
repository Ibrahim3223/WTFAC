# -*- coding: utf-8 -*-
"""
Channel configuration loader from channels.yml
Loads channel-specific settings like topic, search_terms, voice, etc.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# ============================================================================
# KOKORO VOICE MAPPING BY CHANNEL MODE
# ============================================================================
# Each channel mode gets a voice that matches its content style
# Voices: https://github.com/thewh1teagle/kokoro-onnx (26 voices)
#
# Female American: af_heart, af_bella, af_sarah, af_sky, af_alloy,
#                  af_aoede, af_jessica, af_kore, af_nicole, af_nova, af_river
# Male American:   am_adam, am_michael, am_echo, am_eric, am_fenrir,
#                  am_liam, am_onyx, am_puck, am_santa
# British Female:  bf_alice, bf_emma, bf_isabella, bf_lily
# British Male:    bm_daniel, bm_fable, bm_george, bm_lewis
# ============================================================================

VOICE_BY_MODE = {
    # === Educational / Facts ===
    "country_facts": "af_sarah",       # Professional, clear
    "animal_facts": "af_river",        # Calm, nature-friendly
    "animal_patterns": "af_river",     # Nature documentary
    "urban_facts": "af_sky",           # Clear, technical
    "math_visuals": "af_sky",          # Clear, educational

    # === History / Storytelling ===
    "history_story": "bm_fable",       # British storyteller - PERFECT
    "mythology_battle": "am_fenrir",   # Bold, epic voice
    "object_history": "bm_daniel",     # Authoritative British
    "nostalgia_story": "am_liam",      # Smooth, warm nostalgia
    "fame_story": "af_bella",          # Energetic celebrity stories

    # === Horror / Mystery ===
    "horror_story": "am_echo",         # Resonant, mysterious - CREEPY

    # === News / Current Events ===
    "daily_news": "am_onyx",           # Rich, documentary style
    "tech_news": "af_sky",             # Clear, modern tech
    "space_news": "af_nova",           # Modern, cosmic feel
    "sports_news": "am_michael",       # Strong, energetic sports
    "weather_bits": "am_eric",         # Clear weather reports

    # === Kids / Family ===
    "kids_story": "af_heart",          # Warm, friendly for children

    # === Comedy / Entertainment ===
    "deadpan_comedy": "am_puck",       # Lively, comedic timing
    "pet_caption": "af_jessica",       # Friendly pet narration
    "office_plain": "am_eric",         # Clear corporate decode

    # === Sci-Fi / Future / Tech ===
    "ai_alt": "af_nova",               # Modern AI scenarios
    "ai_future": "af_alloy",           # Dynamic future tech
    "utopic_tech": "af_sky",           # Clear optimistic tech
    "post_apoc": "am_adam",            # Deep, serious survival
    "alt_universe": "am_liam",         # Smooth dimension-hopping

    # === Lifestyle / How-To ===
    "fixit_fast": "af_sarah",          # Professional instructions
    "eco_habits": "af_river",          # Calm eco-friendly
    "travel_spot": "bf_emma",          # Elegant British travel

    # === Quotes / Wisdom ===
    "quotes": "bm_daniel",             # Authoritative quotes
    "taxwise_usa": "am_eric",          # Clear financial advice
    "if_lived_today": "bf_isabella",   # Sophisticated reimagining

    # === Sports ===
    "cricket_women": "af_bella",       # Energetic sports

    # === Film / Media ===
    "movie_secrets": "bm_fable",       # Storyteller for film secrets

    # === Nature / Science ===
    "nature_micro": "af_river",        # Calm nature macro
    "coast_science": "af_aoede",       # Smooth ocean science
    "timelapse_science": "af_nova",    # Modern timelapse narration
    "mechanism_explain": "am_onyx",    # Documentary mechanisms

    # === Default ===
    "general": "af_sarah",             # Professional default
}

DEFAULT_VOICE = "af_sarah"  # Fallback for unknown modes


def load_channels_config(config_path: str = "channels.yml") -> Dict[str, Any]:
    """
    Load channels.yml configuration file.
    
    Args:
        config_path: Path to channels.yml file
        
    Returns:
        Dict with channel configurations
    """
    # Try multiple locations
    possible_paths = [
        config_path,
        Path.cwd() / config_path,
        Path(__file__).parent.parent.parent / config_path,
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"✅ Loaded channels config from: {path}")
                    return config
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {path}: {e}")
                continue
    
    logger.warning(f"⚠️ channels.yml not found, using defaults")
    return {"channels": []}


def find_channel_config(channel_name: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find channel configuration by name.
    
    Args:
        channel_name: Channel name to search for
        config: Loaded channels config
        
    Returns:
        Channel config dict or None
    """
    channels = config.get("channels", [])
    
    # Normalize channel name for comparison
    channel_name_normalized = channel_name.strip().lower()
    
    for channel in channels:
        # Check both 'name' and 'env' fields
        name = channel.get("name", "").strip().lower()
        env = channel.get("env", "").strip().lower()
        
        if name == channel_name_normalized or env == channel_name_normalized:
            logger.info(f"✅ Found config for channel: {channel_name}")
            return channel
    
    logger.warning(f"⚠️ No config found for channel: {channel_name}")
    return None


def apply_channel_settings(channel_name: str) -> Dict[str, Any]:
    """
    Load and apply channel-specific settings.
    
    Args:
        channel_name: Name of the channel to configure
        
    Returns:
        Dict with channel-specific settings
    """
    # Load channels config
    config = load_channels_config()
    
    # Find this channel's config
    channel_config = find_channel_config(channel_name, config)
    
    if not channel_config:
        logger.warning(f"⚠️ Using default settings for: {channel_name}")
        return {
            "CHANNEL_TOPIC": "Interesting facts and knowledge",
            "CHANNEL_MODE": "general",
            "CHANNEL_SEARCH_TERMS": [],
            "CHANNEL_LANG": "en",
            "CHANNEL_VISIBILITY": "public",
            "CHANNEL_VOICE": DEFAULT_VOICE
        }

    # Get mode and voice
    mode = channel_config.get("mode", "general")
    voice = channel_config.get("voice") or VOICE_BY_MODE.get(mode, DEFAULT_VOICE)

    # Extract settings
    settings = {
        "CHANNEL_TOPIC": channel_config.get("topic", "Interesting facts and knowledge"),
        "CHANNEL_MODE": mode,
        "CHANNEL_SEARCH_TERMS": channel_config.get("search_terms", []),
        "CHANNEL_LANG": channel_config.get("lang", "en"),
        "CHANNEL_VISIBILITY": channel_config.get("visibility", "public"),
        "CHANNEL_VOICE": voice
    }

    # Get voice info for logging
    voice_info = get_voice_info(voice)

    logger.info(f"📺 Channel: {channel_name}")
    logger.info(f"🎯 Topic: {settings['CHANNEL_TOPIC'][:80]}...")
    logger.info(f"🔍 Search terms: {len(settings['CHANNEL_SEARCH_TERMS'])} available")
    logger.info(f"🎤 Voice: {voice} ({voice_info['name']} - {voice_info['style']})")

    return settings


def get_channel_search_terms(channel_name: str) -> List[str]:
    """
    Get search terms for a specific channel.

    Args:
        channel_name: Name of the channel

    Returns:
        List of search terms
    """
    config = load_channels_config()
    channel_config = find_channel_config(channel_name, config)

    if channel_config:
        return channel_config.get("search_terms", [])

    return []


def get_channel_voice(channel_name: str) -> str:
    """
    Get Kokoro TTS voice for a channel based on its mode.

    Each channel mode has a voice assigned that matches its content style:
    - Horror → am_echo (mysterious, resonant)
    - History → bm_fable (British storyteller)
    - Kids → af_heart (warm, friendly)
    - News → am_onyx (rich, documentary)
    - etc.

    Args:
        channel_name: Name of the channel

    Returns:
        Kokoro voice ID (e.g., "af_sarah", "bm_fable")
    """
    config = load_channels_config()
    channel_config = find_channel_config(channel_name, config)

    if not channel_config:
        logger.warning(f"⚠️ No config for {channel_name}, using default voice")
        return DEFAULT_VOICE

    # First check if channel has explicit voice override in yml
    explicit_voice = channel_config.get("voice")
    if explicit_voice:
        logger.info(f"🎤 Using explicit voice for {channel_name}: {explicit_voice}")
        return explicit_voice

    # Otherwise, map mode to voice
    mode = channel_config.get("mode", "general")
    voice = VOICE_BY_MODE.get(mode, DEFAULT_VOICE)

    logger.info(f"🎤 Voice for {channel_name} (mode={mode}): {voice}")
    return voice


def get_voice_info(voice_id: str) -> Dict[str, str]:
    """
    Get human-readable info about a Kokoro voice.

    Args:
        voice_id: Kokoro voice ID (e.g., "af_sarah")

    Returns:
        Dict with name, gender, accent, style
    """
    VOICE_INFO = {
        # Female American
        "af_heart": {"name": "Heart", "gender": "female", "accent": "american", "style": "warm"},
        "af_bella": {"name": "Bella", "gender": "female", "accent": "american", "style": "energetic"},
        "af_sarah": {"name": "Sarah", "gender": "female", "accent": "american", "style": "professional"},
        "af_sky": {"name": "Sky", "gender": "female", "accent": "american", "style": "clear"},
        "af_alloy": {"name": "Alloy", "gender": "female", "accent": "american", "style": "dynamic"},
        "af_aoede": {"name": "Aoede", "gender": "female", "accent": "american", "style": "smooth"},
        "af_jessica": {"name": "Jessica", "gender": "female", "accent": "american", "style": "friendly"},
        "af_kore": {"name": "Kore", "gender": "female", "accent": "american", "style": "engaging"},
        "af_nicole": {"name": "Nicole", "gender": "female", "accent": "american", "style": "confident"},
        "af_nova": {"name": "Nova", "gender": "female", "accent": "american", "style": "modern"},
        "af_river": {"name": "River", "gender": "female", "accent": "american", "style": "calm"},
        # Male American
        "am_adam": {"name": "Adam", "gender": "male", "accent": "american", "style": "deep"},
        "am_michael": {"name": "Michael", "gender": "male", "accent": "american", "style": "strong"},
        "am_echo": {"name": "Echo", "gender": "male", "accent": "american", "style": "resonant"},
        "am_eric": {"name": "Eric", "gender": "male", "accent": "american", "style": "clear"},
        "am_fenrir": {"name": "Fenrir", "gender": "male", "accent": "american", "style": "bold"},
        "am_liam": {"name": "Liam", "gender": "male", "accent": "american", "style": "smooth"},
        "am_onyx": {"name": "Onyx", "gender": "male", "accent": "american", "style": "rich"},
        "am_puck": {"name": "Puck", "gender": "male", "accent": "american", "style": "lively"},
        "am_santa": {"name": "Santa", "gender": "male", "accent": "american", "style": "jolly"},
        # British Female
        "bf_alice": {"name": "Alice", "gender": "female", "accent": "british", "style": "classic"},
        "bf_emma": {"name": "Emma", "gender": "female", "accent": "british", "style": "elegant"},
        "bf_isabella": {"name": "Isabella", "gender": "female", "accent": "british", "style": "sophisticated"},
        "bf_lily": {"name": "Lily", "gender": "female", "accent": "british", "style": "gentle"},
        # British Male
        "bm_daniel": {"name": "Daniel", "gender": "male", "accent": "british", "style": "authoritative"},
        "bm_fable": {"name": "Fable", "gender": "male", "accent": "british", "style": "storyteller"},
        "bm_george": {"name": "George", "gender": "male", "accent": "british", "style": "classic"},
        "bm_lewis": {"name": "Lewis", "gender": "male", "accent": "british", "style": "warm"},
    }

    return VOICE_INFO.get(voice_id, {
        "name": voice_id,
        "gender": "unknown",
        "accent": "unknown",
        "style": "neutral"
    })
