# -*- coding: utf-8 -*-
"""
Karaoke ASS subtitle builder - CLEAN PRODUCTION VERSION
Modern caption styles - NO EMOJI, NO WATERMARK injection
Text stays EXACTLY as provided - clean and professional
"""
import random
from typing import List, Dict, Optional, Any, Tuple


# ============================================================================
# VIBRANT CAPTION STYLES - Modern, colorful, eye-catching
# Color format: &H00BBGGRR (BGR, not RGB!)
# ============================================================================

CAPTION_STYLES = {
    # Style 1: CLASSIC YELLOW - Safe and proven
    "classic_yellow": {
        "name": "Classic Yellow",
        "fontname": "Arial Black",
        "fontsize_normal": 60,
        "fontsize_hook": 68,
        "fontsize_emphasis": 66,
        "outline": 7,
        "shadow": "5",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H0000FFFF",    # Yellow (BGR: 00-FF-FF)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Yellow
        "color_secondary": "&H0000DDFF", # Light yellow
        "margin_v": 320
    },
    
    # Style 2: NEON CYAN - Electric energy
    "neon_cyan": {
        "name": "Neon Cyan",
        "fontname": "Arial Black",
        "fontsize_normal": 58,
        "fontsize_hook": 66,
        "fontsize_emphasis": 64,
        "outline": 6,
        "shadow": "4",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H00FFFF00",    # Cyan (BGR: FF-FF-00)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Yellow
        "color_secondary": "&H00FFAA00", # Light cyan
        "margin_v": 320
    },
    
    # Style 3: HOT PINK - Bold and attention-grabbing
    "hot_pink": {
        "name": "Hot Pink",
        "fontname": "Impact",
        "fontsize_normal": 58,
        "fontsize_hook": 66,
        "fontsize_emphasis": 64,
        "outline": 6,
        "shadow": "4",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H00FF1493",    # Deep pink (BGR: FF-14-93)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Yellow
        "color_secondary": "&H00FF69B4", # Light pink
        "margin_v": 320
    },
    
    # Style 4: LIME GREEN - Fresh and vibrant
    "lime_green": {
        "name": "Lime Green",
        "fontname": "Arial Black",
        "fontsize_normal": 60,
        "fontsize_hook": 68,
        "fontsize_emphasis": 66,
        "outline": 7,
        "shadow": "5",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H0000FF00",    # Lime (BGR: 00-FF-00)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Yellow
        "color_secondary": "&H0000DD00", # Light lime
        "margin_v": 330
    },
    
    # Style 5: ORANGE FIRE - Warm and energetic
    "orange_fire": {
        "name": "Orange Fire",
        "fontname": "Impact",
        "fontsize_normal": 60,
        "fontsize_hook": 68,
        "fontsize_emphasis": 66,
        "outline": 7,
        "shadow": "5",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H000099FF",    # Orange (BGR: 00-99-FF)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Yellow
        "color_secondary": "&H0000BBFF", # Light orange
        "margin_v": 330
    },
    
    # Style 6: PURPLE VIBES - Trendy and modern
    "purple_vibes": {
        "name": "Purple Vibes",
        "fontname": "Montserrat Black",
        "fontsize_normal": 58,
        "fontsize_hook": 66,
        "fontsize_emphasis": 64,
        "outline": 6,
        "shadow": "4",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H00FF00FF",    # Magenta (BGR: FF-00-FF)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H00FF00FF",  # Magenta
        "color_secondary": "&H00CC00FF", # Light purple
        "margin_v": 320
    },
    
    # Style 7: TURQUOISE WAVE - Cool and calming
    "turquoise_wave": {
        "name": "Turquoise Wave",
        "fontname": "Arial Black",
        "fontsize_normal": 58,
        "fontsize_hook": 66,
        "fontsize_emphasis": 64,
        "outline": 6,
        "shadow": "4",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H00FFCC00",    # Turquoise (BGR: FF-CC-00)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H00FFFF00",  # Cyan
        "color_secondary": "&H00FFDD00", # Light turquoise
        "margin_v": 320
    },
    
    # Style 8: RED HOT - Intense and dramatic
    "red_hot": {
        "name": "Red Hot",
        "fontname": "Impact",
        "fontsize_normal": 60,
        "fontsize_hook": 68,
        "fontsize_emphasis": 66,
        "outline": 7,
        "shadow": "5",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H000000FF",    # Red (BGR: 00-00-FF)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Yellow
        "color_secondary": "&H000033FF", # Light red
        "margin_v": 330
    }
}

# Dynamic weight distribution for A/B testing
STYLE_WEIGHTS = {
    "classic_yellow": 0.25,  # Most popular/safe (25%)
    "neon_cyan": 0.15,       # Electric energy (15%)
    "hot_pink": 0.12,        # Bold attention (12%)
    "lime_green": 0.12,      # Fresh vibes (12%)
    "orange_fire": 0.12,     # Warm energy (12%)
    "purple_vibes": 0.10,    # Trendy (10%)
    "turquoise_wave": 0.08,  # Cool calm (8%)
    "red_hot": 0.06          # Dramatic (6%)
}

# ============================================================================
# EMPHASIS KEYWORDS - Words that get special treatment
# ============================================================================

EMPHASIS_KEYWORDS = {
    # Extreme emotions
    "NEVER", "ALWAYS", "IMPOSSIBLE", "INSANE", "CRAZY", "SHOCKING",
    "UNBELIEVABLE", "INCREDIBLE", "AMAZING", "STUNNING", "MIND-BLOWING",
    
    # Urgency
    "NOW", "IMMEDIATELY", "INSTANTLY", "URGENT", "BREAKING", "ALERT",
    "STOP", "WAIT", "ATTENTION", "WARNING", "DANGER",
    
    # Exclusivity
    "SECRET", "HIDDEN", "BANNED", "ILLEGAL", "FORBIDDEN", "RARE",
    "EXCLUSIVE", "LIMITED", "FIRST", "LAST", "ONLY", "UNIQUE",
    
    # Superlatives
    "BEST", "WORST", "BIGGEST", "SMALLEST", "FASTEST", "SLOWEST",
    "MOST", "LEAST", "ULTIMATE", "SUPREME", "MAXIMUM",
    
    # Social proof
    "VIRAL", "TRENDING", "POPULAR", "FAMOUS", "EVERYONE", "NOBODY",
    "MILLIONS", "THOUSANDS", "BILLION"
}


def get_random_style() -> str:
    """Get a random caption style based on weights."""
    return random.choices(
        list(STYLE_WEIGHTS.keys()),
        weights=list(STYLE_WEIGHTS.values())
    )[0]


def get_style_info(style_name: str) -> Dict[str, Any]:
    """Get information about a specific style."""
    if style_name in CAPTION_STYLES:
        return CAPTION_STYLES[style_name]
    return CAPTION_STYLES["classic_yellow"]


def list_all_styles() -> List[str]:
    """List all available caption styles."""
    return list(CAPTION_STYLES.keys())


# ============================================================================
# BACKWARD COMPATIBILITY - Legacy function (not used in new system)
# ============================================================================

def build_karaoke_ass(
    text: str,
    seg_dur: float,
    words: List[tuple],
    is_hook: bool = False,
    style_name: Optional[str] = None
) -> str:
    """
    DEPRECATED: Legacy function for backward compatibility.
    New system uses renderer.py's _write_smooth_ass instead.
    
    This function is kept only for imports that haven't been updated yet.
    Returns a minimal placeholder ASS file.
    """
    return """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,60,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,-1,0,0,0,100,100,1,0,1,7,5,2,50,50,330,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,{text}
"""
