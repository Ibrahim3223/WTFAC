# -*- coding: utf-8 -*-
"""
Karaoke ASS subtitle builder - VIBRANT EDITION
Ultra-modern caption styles with gradient effects and smooth animations
"""
import os
import random
from typing import List, Tuple, Dict, Optional, Any


# ============================================================================
# VIBRANT CAPTION STYLES - Modern, colorful, eye-catching
# ============================================================================

CAPTION_STYLES = {
    # Style 1: NEON GRADIENT - Electric blue to cyan
    "neon_gradient": {
        "name": "Neon Gradient",
        "fontname": "Arial Black",
        "fontsize_normal": 58,
        "fontsize_hook": 66,
        "fontsize_emphasis": 64,
        "outline": 6,
        "shadow": "4",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H00FFFF00",    # Cyan (BGR format)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Yellow
        "color_secondary": "&H00FF8800", # Light blue for gradient
        "margin_v_normal": 320,
        "margin_v_hook": 280
    },
    
    # Style 2: FIRE GRADIENT - Orange to yellow
    "fire_gradient": {
        "name": "Fire Gradient",
        "fontname": "Impact",
        "fontsize_normal": 60,
        "fontsize_hook": 68,
        "fontsize_emphasis": 66,
        "outline": 7,
        "shadow": "5",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H0000A5FF",    # Orange (BGR)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Yellow
        "color_secondary": "&H0000D4FF", # Light orange
        "margin_v_normal": 330,
        "margin_v_hook": 290
    },
    
    # Style 3: PURPLE VIBES - Deep purple to magenta
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
        "color_active": "&H00FF00FF",    # Magenta (BGR)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H00FF00FF",  # Magenta
        "color_secondary": "&H00CC00FF", # Light purple
        "margin_v_normal": 320,
        "margin_v_hook": 280
    },
    
    # Style 4: LIME PUNCH - Bright lime green
    "lime_punch": {
        "name": "Lime Punch",
        "fontname": "Arial Black",
        "fontsize_normal": 60,
        "fontsize_hook": 68,
        "fontsize_emphasis": 66,
        "outline": 7,
        "shadow": "5",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H0000FF00",    # Lime green (BGR)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Yellow
        "color_secondary": "&H0000DD00", # Light lime
        "margin_v_normal": 330,
        "margin_v_hook": 290
    },
    
    # Style 5: HOT PINK - Vibrant hot pink
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
        "color_active": "&H00FF1493",    # Hot pink (BGR)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Yellow
        "color_secondary": "&H00FF69B4", # Light pink
        "margin_v_normal": 320,
        "margin_v_hook": 280
    },
    
    # Style 6: CLASSIC YELLOW - Traditional but vibrant
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
        "color_active": "&H0000FFFF",    # Yellow (BGR)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Yellow
        "color_secondary": "&H0000DDFF", # Light yellow
        "margin_v_normal": 330,
        "margin_v_hook": 290
    },
    
    # Style 7: OCEAN WAVE - Turquoise to cyan
    "ocean_wave": {
        "name": "Ocean Wave",
        "fontname": "Montserrat Black",
        "fontsize_normal": 58,
        "fontsize_hook": 66,
        "fontsize_emphasis": 64,
        "outline": 6,
        "shadow": "4",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H00FFCC00",    # Turquoise (BGR)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H00FFFF00",  # Cyan
        "color_secondary": "&H00FFDD00", # Light turquoise
        "margin_v_normal": 320,
        "margin_v_hook": 280
    },
    
    # Style 8: SUNSET GLOW - Red-orange gradient
    "sunset_glow": {
        "name": "Sunset Glow",
        "fontname": "Impact",
        "fontsize_normal": 60,
        "fontsize_hook": 68,
        "fontsize_emphasis": 66,
        "outline": 7,
        "shadow": "5",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H000045FF",    # Red-orange (BGR)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000A5FF",  # Orange
        "color_secondary": "&H000070FF", # Light red-orange
        "margin_v_normal": 330,
        "margin_v_hook": 290
    }
}

# Dynamic weight distribution - favor vibrant styles
STYLE_WEIGHTS = {
    "neon_gradient": 0.15,   # Electric energy
    "fire_gradient": 0.15,   # High energy
    "purple_vibes": 0.12,    # Trendy
    "lime_punch": 0.10,      # Attention-grabbing
    "hot_pink": 0.10,        # Bold
    "classic_yellow": 0.20,  # Most popular/safe
    "ocean_wave": 0.10,      # Calm but vibrant
    "sunset_glow": 0.08      # Dramatic
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

# ============================================================================
# EMOJI INJECTION - Enhanced context-aware emoji placement
# ============================================================================

EMOJI_MAP = {
    # Extreme emotions
    "shocking": "🤯", "insane": "🤯", "crazy": "🤯", "mind": "🤯",
    "unbelievable": "😱", "incredible": "😱", "amazing": "😲", "wow": "😮",
    
    # Fire/Energy
    "fire": "🔥", "hot": "🔥", "lit": "🔥", "energy": "⚡",
    "explosive": "💥", "boom": "💥", "power": "💪",
    
    # Warnings
    "warning": "⚠️", "danger": "⚠️", "careful": "⚠️", "stop": "🛑",
    "alert": "🚨", "urgent": "🚨", "breaking": "📢",
    
    # Secrets/Mystery
    "secret": "🤫", "hidden": "🤫", "mystery": "🔍", "discover": "🔍",
    "forbidden": "🚫", "banned": "🚫", "illegal": "🚫",
    
    # Money/Success
    "money": "💰", "rich": "💰", "expensive": "💰", "cost": "💰",
    "profit": "💸", "cash": "💵", "dollar": "💵",
    "success": "✨", "win": "🏆", "champion": "🏆", "best": "👑",
    
    # Trending/Viral
    "viral": "📈", "trending": "📈", "popular": "📈", "famous": "⭐",
    "views": "👁️", "million": "💯", "billions": "💯",
    "subscribe": "🔔", "follow": "❤️", "like": "👍", "love": "❤️",
    
    # Science/Tech
    "science": "🔬", "research": "🔬", "study": "📚", "learn": "📖",
    "brain": "🧠", "smart": "🧠", "genius": "🧠",
    "technology": "🤖", "robot": "🤖", "ai": "🤖", "future": "🚀",
    "space": "🌌", "universe": "🌌", "galaxy": "🌌",
    
    # Nature/Animals
    "animal": "🐾", "nature": "🌿", "plant": "🌱", "tree": "🌳",
    "ocean": "🌊", "water": "💧", "sea": "🌊", "beach": "🏖️",
    "earth": "🌍", "world": "🌍", "planet": "🪐", "global": "🌍",
    
    # Time/Speed
    "fast": "⚡", "quick": "⚡", "instant": "⚡", "speed": "⚡",
    "rapid": "💨", "turbo": "🏃", "zoom": "💨",
    "slow": "🐌", "wait": "⏱️", "time": "⏰", "clock": "⏰",
    "ancient": "🕰️", "history": "📜", "old": "👴",
    
    # Food/Drink
    "food": "🍔", "eat": "🍕", "delicious": "😋", "yummy": "😋",
    "drink": "🥤", "coffee": "☕", "tea": "🍵",
    
    # Emotions
    "happy": "😊", "sad": "😢", "angry": "😠", "laugh": "😂",
    "cry": "😭", "scared": "😨", "surprised": "😮",
    
    # Actions
    "think": "🤔", "question": "❓", "answer": "✅", "check": "✅",
    "wrong": "❌", "mistake": "❌", "fail": "❌",
    "correct": "✔️", "perfect": "💯", "hundred": "💯"
}


def inject_emojis(text: str, max_emojis: int = 2) -> str:
    """
    Inject contextually appropriate emojis into text.
    Prioritizes first occurrence and limits total emojis.
    """
    words = text.lower().split()
    injected = []
    emoji_count = 0
    
    for word in words:
        # Clean word for matching
        clean_word = word.strip(".,!?;:'\"")
        
        # Check if word has an emoji mapping and we haven't hit limit
        if emoji_count < max_emojis and clean_word in EMOJI_MAP:
            emoji = EMOJI_MAP[clean_word]
            injected.append(f"{word} {emoji}")
            emoji_count += 1
        else:
            injected.append(word)
    
    return " ".join(injected)


def build_karaoke_ass(
    text: str,
    seg_dur: float,
    words: List[Tuple[str, float]],
    is_hook: bool = False,
    style_name: Optional[str] = None
) -> str:
    """
    Build karaoke-style ASS subtitle with vibrant gradient effects.
    
    Args:
        text: Full sentence text
        seg_dur: Segment duration in seconds
        words: List of (word, duration) tuples
        is_hook: Whether this is the hook (first sentence)
        style_name: Specific style to use (or None for random weighted choice)
    
    Returns:
        ASS subtitle string
    """
    from autoshorts.config import settings
    
    # Select caption style
    if style_name and style_name in CAPTION_STYLES:
        style = CAPTION_STYLES[style_name]
    else:
        # Random weighted choice for A/B testing
        style_name = random.choices(
            list(STYLE_WEIGHTS.keys()),
            weights=list(STYLE_WEIGHTS.values())
        )[0]
        style = CAPTION_STYLES[style_name]
    
    # Get style parameters
    fontname = style["fontname"]
    fontsize = style["fontsize_hook"] if is_hook else style["fontsize_normal"]
    fontsize_emphasis = style["fontsize_emphasis"]
    outline = style["outline"]
    shadow = style["shadow"]
    margin_v = style["margin_v_hook"] if is_hook else style["margin_v_normal"]
    
    # Colors
    inactive = style["color_inactive"]
    active = style["color_active"]
    outline_c = style["color_outline"]
    emphasis_c = style["color_emphasis"]
    secondary_c = style.get("color_secondary", active)
    
    # Effects
    use_bounce = style.get("bounce", True) and settings.KARAOKE_EFFECTS
    use_glow = style.get("glow", False) and settings.KARAOKE_EFFECTS
    
    # Inject emojis into text
    text_with_emoji = inject_emojis(text, max_emojis=2)
    
    # Convert words to uppercase
    words_upper = [(w.upper(), d) for w, d in words if w.strip()]
    
    if not words_upper:
        split_words = (text_with_emoji or "…").split()
        each_dur = seg_dur / max(1, len(split_words))
        words_upper = [(w.upper(), each_dur) for w in split_words]
    
    # Convert to centiseconds
    ds = [max(8, int(round(d * 100))) for _, d in words_upper]
    
    # Build smooth effect tags
    shake_tag = ""
    blur_tag = ""
    
    if use_bounce:
        if is_hook:
            # Aggressive bounce for hook with smooth easing
            shake_tag = r"\t(0,50,\1.5,\fscx115\fscy115)\t(50,100,\1.5,\fscx95\fscy95)\t(100,150,\fscx100\fscy100)"
        else:
            # Subtle bounce with smooth easing
            shake_tag = r"\t(0,60,\1.5,\fscx108\fscy108)\t(60,120,\1.5,\fscx98\fscy98)\t(120,180,\fscx100\fscy100)"
    
    if use_glow:
        blur_tag = r"\blur3.5\bord" + str(outline + 1)
    else:
        blur_tag = r"\blur1.5"
    
    # Build karaoke line with emphasis detection and gradient colors
    kline_parts = []
    for i, (word_text, _) in enumerate(words_upper):
        duration_cs = ds[i]
        
        # Check if word should be emphasized
        clean_word = word_text.strip(".,!?;:").upper()
        is_emphasis = clean_word in EMPHASIS_KEYWORDS
        
        if is_emphasis:
            # Special treatment for emphasis words - vibrant glow
            tags = f"\\k{duration_cs}\\fs{fontsize_emphasis}\\c{emphasis_c}\\3c{secondary_c}{shake_tag}{blur_tag}\\t(0,{duration_cs*5},\\fscx105\\fscy105)"
        else:
            # Normal word with smooth color transition
            tags = f"\\k{duration_cs}\\c{active}\\3c{secondary_c}{shake_tag}{blur_tag}"
        
        kline_parts.append(f"{{{tags}}}{word_text}")
    
    kline = " ".join(kline_parts)
    
    # Build ASS file with vibrant styling
    ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Base,{fontname},{fontsize},{inactive},{active},{outline_c},&H7F000000,1,0,0,0,100,100,1,0,1,{outline},{shadow},2,50,50,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,{_ass_time(seg_dur)},Base,,0,0,{margin_v},,{{\\bord{outline}\\shad{shadow}\\blur1.5}}{kline}
"""
    
    return ass


def _ass_time(s: float) -> str:
    """Convert seconds to ASS time format with precision."""
    h = int(s // 3600)
    s -= h * 3600
    m = int(s // 60)
    s -= m * 60
    cs = int(round((s - int(s)) * 100))
    s = int(s)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


# ============================================================================
# HELPER: Get random style for A/B testing
# ============================================================================

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
    return CAPTION_STYLES["classic_yellow"]  # Default to safe choice


def list_all_styles() -> List[str]:
    """List all available caption styles."""
    return list(CAPTION_STYLES.keys())


def get_style_preview(style_name: str) -> str:
    """Get a human-readable preview of a style."""
    if style_name not in CAPTION_STYLES:
        return "Style not found"
    
    style = CAPTION_STYLES[style_name]
    return f"{style['name']} - {style['fontname']} @ {style['fontsize_normal']}px"
