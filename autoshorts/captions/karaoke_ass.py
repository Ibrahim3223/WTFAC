# -*- coding: utf-8 -*-
"""
Karaoke ASS subtitle builder - VIRAL OPTIMIZED
Multiple caption styles with emphasis system and emoji injection
"""
import os
import random
from typing import List, Tuple, Dict, Optional, Any


# ============================================================================
# VIRAL CAPTION STYLES - Based on TikTok/CapCut trending formats
# ============================================================================

# ============================================================================
# CAPCUT-STYLE CAPTION STYLES - Clean, modern, NO effects
# ============================================================================

CAPTION_STYLES = {
    # Style 1: CAPCUT CLASSIC - Yellow/White, thick outline
    "capcut_classic": {
        "name": "CapCut Classic",
        "fontname": "Arial Black",
        "fontsize_normal": 58,
        "fontsize_hook": 64,
        "fontsize_emphasis": 58,  # No size change
        "outline": 6,  # Thick outline
        "shadow": "4",
        "glow": False,
        "bounce": False,  # NO bounce
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H0000FFFF",    # Yellow
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Same as active
        "margin_v_normal": 320,
        "margin_v_hook": 280
    },
    
    # Style 2: CAPCUT NEON - Bright magenta/cyan
    "capcut_neon": {
        "name": "CapCut Neon",
        "fontname": "Impact",
        "fontsize_normal": 60,
        "fontsize_hook": 66,
        "fontsize_emphasis": 60,
        "outline": 7,
        "shadow": "5",
        "glow": False,
        "bounce": False,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H00FF00FF",    # Magenta
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H00FF00FF",
        "margin_v_normal": 330,
        "margin_v_hook": 290
    },
    
    # Style 3: CAPCUT CLEAN - Pure white, minimal
    "capcut_clean": {
        "name": "CapCut Clean",
        "fontname": "Montserrat Black",
        "fontsize_normal": 56,
        "fontsize_hook": 62,
        "fontsize_emphasis": 56,
        "outline": 5,
        "shadow": "3",
        "glow": False,
        "bounce": False,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H00FFFFFF",    # White (no change)
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H00FFFFFF",
        "margin_v_normal": 310,
        "margin_v_hook": 270
    }
}

# Weight distribution for A/B testing
STYLE_WEIGHTS = {
    "capcut_classic": 0.50,  # Most popular
    "capcut_neon": 0.30,     # High energy
    "capcut_clean": 0.20     # Minimal
}

# ============================================================================
# EMPHASIS KEYWORDS - Words that get special treatment
# ============================================================================

EMPHASIS_KEYWORDS = {
    "NEVER", "ALWAYS", "SECRET", "HIDDEN", "SHOCKING", "INSANE",
    "BANNED", "ILLEGAL", "IMPOSSIBLE", "CRAZY", "VIRAL", "BREAKING",
    "URGENT", "WARNING", "STOP", "WAIT", "INSTANTLY", "FOREVER",
    "ONLY", "FIRST", "LAST", "BEST", "WORST", "NOBODY", "EVERYONE"
}

# ============================================================================
# EMOJI INJECTION - Context-aware emoji placement
# ============================================================================

EMOJI_MAP = {
    # Emotion/Reaction
    "shocking": "🤯", "insane": "🤯", "crazy": "🤯", "mind": "🤯",
    "fire": "🔥", "hot": "🔥", "amazing": "🔥", "incredible": "🔥",
    "warning": "⚠️", "danger": "⚠️", "careful": "⚠️", "watch": "👀",
    "secret": "🤫", "hidden": "🤫", "never": "🚫", "banned": "🚫",
    
    # Money/Success
    "money": "💰", "rich": "💰", "expensive": "💰", "cost": "💰",
    "profit": "💸", "success": "✨", "win": "🏆", "best": "🏆",
    
    # Trending/Viral
    "viral": "📈", "trending": "📈", "popular": "📈", "views": "👁️",
    "subscribe": "🔔", "follow": "❤️", "like": "👍", "comment": "💬",
    
    # Science/Tech
    "science": "🔬", "research": "🔬", "study": "📚", "brain": "🧠",
    "technology": "🤖", "robot": "🤖", "ai": "🤖", "future": "🚀",
    
    # Nature/Animals
    "animal": "🐾", "nature": "🌿", "ocean": "🌊", "water": "💧",
    "earth": "🌍", "world": "🌍", "planet": "🪐", "space": "🌌",
    
    # Time/Speed
    "fast": "⚡", "quick": "⚡", "instant": "⚡", "speed": "⚡",
    "slow": "🐌", "wait": "⏱️", "time": "⏰", "ancient": "🕰️"
}


def inject_emojis(text: str) -> str:
    """
    Inject contextually appropriate emojis into text.
    Prioritizes first occurrence of keywords.
    """
    words = text.lower().split()
    injected = []
    emoji_used = False
    
    for word in words:
        # Clean word for matching
        clean_word = word.strip(".,!?;:")
        
        # Check if word has an emoji mapping
        if not emoji_used and clean_word in EMOJI_MAP:
            # Add emoji after the word
            emoji = EMOJI_MAP[clean_word]
            injected.append(f"{word} {emoji}")
            emoji_used = True  # Only one emoji per sentence
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
    Build karaoke-style ASS subtitle with viral optimizations.
    
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
    
    # Effects
    use_bounce = style.get("bounce", True) and settings.KARAOKE_EFFECTS
    use_glow = style.get("glow", False) and settings.KARAOKE_EFFECTS
    
    # Inject emojis into text
    text_with_emoji = inject_emojis(text)
    
    # Convert words to uppercase
    words_upper = [(w.upper(), d) for w, d in words if w.strip()]
    
    if not words_upper:
        split_words = (text_with_emoji or "…").split()
        each_dur = seg_dur / max(1, len(split_words))
        words_upper = [(w.upper(), each_dur) for w in split_words]
    
    # Convert to centiseconds
    ds = [max(8, int(round(d * 100))) for _, d in words_upper]
    
    # Build effect tags
    shake_tag = ""
    blur_tag = ""
    
    if use_bounce:
        if is_hook:
            # More aggressive bounce for hook
            shake_tag = r"\t(0,40,\fscx110\fscy110)\t(40,80,\fscx90\fscy90)\t(80,120,\fscx100\fscy100)"
        else:
            # Subtle bounce for normal text
            shake_tag = r"\t(0,50,\fscx105\fscy105)\t(50,100,\fscx95\fscy95)\t(100,150,\fscx100\fscy100)"
    
    if use_glow:
        blur_tag = r"\blur3"
    else:
        blur_tag = r"\blur1"
    
    # Build karaoke line with emphasis detection
    kline_parts = []
    for i, (word_text, _) in enumerate(words_upper):
        duration_cs = ds[i]
        
        # Check if word should be emphasized
        clean_word = word_text.strip(".,!?;:").upper()
        is_emphasis = clean_word in EMPHASIS_KEYWORDS
        
        if is_emphasis:
            # Special treatment for emphasis words
            tags = f"\\k{duration_cs}\\fs{fontsize_emphasis}\\c{emphasis_c}{shake_tag}{blur_tag}"
        else:
            # Normal word
            tags = f"\\k{duration_cs}{shake_tag}{blur_tag}"
        
        kline_parts.append(f"{{{tags}}}{word_text}")
    
    kline = " ".join(kline_parts)
    
    # Build ASS file
    ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Base,{fontname},{fontsize},{inactive},{active},{outline_c},&H7F000000,1,0,0,0,100,100,0,0,1,{outline},{shadow},2,50,50,{margin_v},0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,{_ass_time(seg_dur)},Base,,0,0,{margin_v},,{{\\bord{outline}\\shad{shadow}}}{kline}
"""
    
    return ass


def _ass_time(s: float) -> str:
    """Convert seconds to ASS time format."""
    h = int(s // 3600)
    s -= h * 3600
    m = int(s // 60)
    s -= m * 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


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
    return CAPTION_STYLES["viral_bold"]  # Default
