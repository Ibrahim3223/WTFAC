# -*- coding: utf-8 -*-
"""Karaoke ASS subtitle builder."""
import os
from typing import List, Tuple

def build_karaoke_ass(
    text: str,
    seg_dur: float,
    words: List[Tuple[str, float]],
    is_hook: bool = False
) -> str:
    """Build karaoke-style ASS subtitle."""
    from autoshorts.config import settings
    
    fontsize = 58 if is_hook else 52
    margin_v = 270 if is_hook else 330
    outline = 4 if is_hook else 3
    
    # Convert words to uppercase
    words_upper = [(w.upper(), d) for w, d in words if w.strip()]
    
    if not words_upper:
        split_words = (text or "…").split()
        each_dur = seg_dur / max(1, len(split_words))
        words_upper = [(w.upper(), each_dur) for w in split_words]
    
    # Convert to centiseconds
    total_cs = int(round(seg_dur * 100))
    ds = [max(8, int(round(d * 100))) for _, d in words_upper]
    
    # Effects
    use_effects = settings.KARAOKE_EFFECTS
    effect_style = settings.EFFECT_STYLE.lower()
    
    shake_tag = ""
    blur_tag = ""
    shadow = "0"
    
    if use_effects:
        if effect_style == "dynamic":
            shake_tag = r"\t(0,40,\fscx108\fscy108)\t(40,80,\fscx92\fscy92)\t(80,120,\fscx100\fscy100)"
            blur_tag = r"\blur4"
            shadow = "3"
        elif effect_style == "subtle":
            blur_tag = r"\blur1"
            shadow = "1"
        else:  # moderate
            shake_tag = r"\t(0,50,\fscx103\fscy103)\t(50,100,\fscx97\fscy97)\t(100,150,\fscx100\fscy100)" if is_hook else ""
            blur_tag = r"\blur2"
            shadow = "2"
    
    # Build karaoke line
    kline_parts = []
    for i, (word_text, _) in enumerate(words_upper):
        duration_cs = ds[i]
        tags = f"\\k{duration_cs}{shake_tag}{blur_tag}"
        kline_parts.append(f"{{{tags}}}{word_text}")
    
    kline = " ".join(kline_parts)
    
    # Color conversion
    def to_ass(c):
        c = c.strip()
        if c.startswith("0x"): c = c[2:]
        if c.startswith("#"): c = c[1:]
        if len(c) == 6: c = "00" + c
        return f"&H00{c[-2:]}{c[-4:-2]}{c[-6:-4]}"
    
    inactive = to_ass(settings.KARAOKE_INACTIVE)
    active = to_ass(settings.KARAOKE_ACTIVE)
    outline_c = to_ass(settings.KARAOKE_OUTLINE)
    
    ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Base,DejaVu Sans,{fontsize},{inactive},{active},{outline_c},&H7F000000,1,0,0,0,100,100,0,0,1,{outline},{shadow},2,50,50,{margin_v},0

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
