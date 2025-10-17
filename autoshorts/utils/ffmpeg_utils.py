# -*- coding: utf-8 -*-
"""
FFmpeg utilities: probe, check filters, run commands.
"""
import re
import subprocess
import pathlib
from typing import Optional

def run(cmd, check=True):
    """Execute command and return result."""
    res = subprocess.run(cmd, text=True, capture_output=True)
    if check and res.returncode != 0:
        raise RuntimeError(res.stderr[:4000])
    return res

def ffprobe_duration(path: str) -> float:
    """Get video/audio duration in seconds."""
    try:
        out = run([
            "ffprobe", "-v", "quiet", "-show_entries", 
            "format=duration", "-of", "csv=p=0", path
        ]).stdout.strip()
        return float(out) if out else 0.0
    except:
        return 0.0

def ffmpeg_has_filter(name: str) -> bool:
    """Check if FFmpeg has a specific filter."""
    try:
        out = run(["ffmpeg", "-hide_banner", "-filters"], check=False).stdout
        return bool(re.search(rf"\b{name}\b", out))
    except Exception:
        return False

def font_path() -> str:
    """Find system font path."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf"
    ]
    for p in candidates:
        if pathlib.Path(p).exists():
            return p
    return ""

def sanitize_font_path(font_path_str: str) -> str:
    """Escape font path for FFmpeg."""
    if not font_path_str: 
        return ""
    return font_path_str.replace(":", r"\:").replace(",", r"\,").replace("\\", "/")

def quantize_to_frames(seconds: float, fps: int = 25) -> tuple[int, float]:
    """Convert seconds to exact frame count and back to seconds."""
    frames = max(2, int(round(seconds * fps)))
    return frames, frames / float(fps)

# Cache filter availability
_HAS_DRAWTEXT = None
_HAS_SUBTITLES = None

def has_drawtext() -> bool:
    """Check if drawtext filter is available (cached)."""
    global _HAS_DRAWTEXT
    if _HAS_DRAWTEXT is None:
        _HAS_DRAWTEXT = ffmpeg_has_filter("drawtext")
    return _HAS_DRAWTEXT

def has_subtitles() -> bool:
    """Check if subtitles filter is available (cached)."""
    global _HAS_SUBTITLES
    if _HAS_SUBTITLES is None:
        _HAS_SUBTITLES = ffmpeg_has_filter("subtitles")
    return _HAS_SUBTITLES
