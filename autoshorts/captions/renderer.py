# -*- coding: utf-8 -*-
"""
Caption rendering - CAPCUT PERFECT SYNC
Simple word-by-word Dialogue events - NO karaoke tags
"""
import os
import pathlib
import logging
from typing import List, Tuple, Optional, Dict, Any

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run, has_subtitles, ffprobe_duration
from autoshorts.captions.karaoke_ass import CAPTION_STYLES, get_random_style, EMPHASIS_KEYWORDS

logger = logging.getLogger(__name__)


class CaptionRenderer:
    """Render captions with CapCut-style perfect sync."""
    
    # CapCut-style parameters
    WORDS_PER_CHUNK = 3      # Max words per caption line
    MIN_WORD_DURATION = 0.15  # Min 150ms per word
    
    def render_captions(
        self,
        video_segments: List[str],
        audio_segments: List[Dict[str, Any]],
        output_dir: str
    ) -> List[str]:
        """Render captions on all video segments."""
        captioned_segments = []
        
        for i, (video_path, audio_segment) in enumerate(zip(video_segments, audio_segments)):
            try:
                text = audio_segment["text"]
                words = audio_segment.get("word_timings", [])
                duration = audio_segment.get("duration", 0)
                sentence_type = audio_segment.get("type", "buildup")
                is_hook = (i == 0 or sentence_type == "hook")
                
                logger.info(f"      Rendering caption {i+1}/{len(video_segments)}: {text[:50]}...")
                
                # Validate and fix word timings
                if words:
                    words = self._validate_timings(words, duration)
                else:
                    logger.warning(f"      ⚠️ No word timings, using fallback")
                    words = self._fallback_timings(text, duration)
                
                captioned_path = self.render(
                    video_path=video_path,
                    text=text,
                    words=words,
                    duration=duration,
                    is_hook=is_hook,
                    sentence_type=sentence_type,
                    temp_dir=output_dir
                )
                
                if captioned_path and os.path.exists(captioned_path):
                    captioned_segments.append(captioned_path)
                else:
                    logger.error(f"      ❌ Caption rendering failed for segment {i+1}")
                    return []
                    
            except Exception as e:
                logger.error(f"      ❌ Error rendering caption {i+1}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return []
        
        logger.info(f"      ✅ Rendered {len(captioned_segments)} captioned segments")
        return captioned_segments
    
    def render(
        self,
        video_path: str,
        text: str,
        words: List[Tuple[str, float]],
        duration: float,
        is_hook: bool = False,
        sentence_type: str = "buildup",
        temp_dir: str = None
    ) -> str:
        """Render captions with perfect sync."""
        try:
            if duration <= 0:
                duration = ffprobe_duration(video_path)
            
            frames = max(2, int(round(duration * settings.TARGET_FPS)))
            output = video_path.replace(".mp4", "_caption.mp4")
            
            if settings.KARAOKE_CAPTIONS and has_subtitles():
                # Create CapCut-style ASS (word-by-word, NO karaoke)
                ass_path = video_path.replace(".mp4", ".ass")
                self._write_capcut_ass(words, duration, sentence_type, ass_path)
                
                tmp_out = output.replace(".mp4", ".tmp.mp4")
                
                try:
                    # Add captions
                    run([
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", video_path,
                        "-vf", f"subtitles='{ass_path}',setsar=1,fps={settings.TARGET_FPS}",
                        "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                        "-an",
                        "-c:v", "libx264", "-preset", "medium",
                        "-crf", str(max(16, settings.CRF_VISUAL - 3)),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        tmp_out
                    ])
                    
                    # Enforce exact frames
                    run([
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", tmp_out,
                        "-vf", f"setsar=1,fps={settings.TARGET_FPS},trim=start_frame=0:end_frame={frames}",
                        "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                        "-an",
                        "-c:v", "libx264", "-preset", "medium",
                        "-crf", str(settings.CRF_VISUAL),
                        "-pix_fmt", "yuv420p",
                        output
                    ])
                    
                finally:
                    pathlib.Path(ass_path).unlink(missing_ok=True)
                    pathlib.Path(tmp_out).unlink(missing_ok=True)
                
                return output
            else:
                logger.warning(f"      Skipping captions for {video_path}")
                return video_path
                
        except Exception as e:
            logger.error(f"      ❌ Error in render(): {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return video_path
    
    # ========================================================================
    # TIMING VALIDATION
    # ========================================================================
    
    def _validate_timings(
        self, 
        word_timings: List[Tuple[str, float]], 
        total_duration: float
    ) -> List[Tuple[str, float]]:
        """Validate and fix word timings."""
        if not word_timings:
            return []
        
        # Check if atempo scaling needed
        sum_raw = sum(d for _, d in word_timings)
        
        if sum_raw > total_duration * 1.05:
            atempo_factor = sum_raw / total_duration
            logger.warning(f"      ⚠️ Applying atempo scale: {atempo_factor:.2f}")
            word_timings = [(word, dur / atempo_factor) for word, dur in word_timings]
            sum_raw = sum(d for _, d in word_timings)
        
        fixed = []
        cumulative = 0.0
        
        for i, (word, dur) in enumerate(word_timings):
            if not word.strip():
                continue
            
            # Clamp duration
            dur = max(self.MIN_WORD_DURATION, min(dur, 2.0))
            
            # Stop if exceeding total
            if cumulative + dur > total_duration + 0.01:
                remaining = max(0.0, total_duration - cumulative)
                remaining_words = len(word_timings) - i
                
                if remaining > 0.1 and remaining_words > 0:
                    dur = remaining / remaining_words
                    if dur >= self.MIN_WORD_DURATION:
                        fixed.append((word, dur))
                        cumulative += dur
                
                logger.warning(f"      ⚠️ TTS cut at '{word}' ({i+1}/{len(word_timings)})")
                break
            
            fixed.append((word, dur))
            cumulative += dur
        
        # Final adjustment
        if fixed:
            total_fixed = sum(d for _, d in fixed)
            diff = total_duration - total_fixed
            
            if abs(diff) > 0.01:
                word, dur = fixed[-1]
                fixed[-1] = (word, max(self.MIN_WORD_DURATION, dur + diff))
                total_fixed = sum(d for _, d in fixed)
            
            diff_ms = abs(total_fixed - total_duration) * 1000
            logger.info(f"      🎯 Sync: {total_fixed:.3f}s / {total_duration:.3f}s (±{diff_ms:.0f}ms)")
        
        return fixed
    
    def _fallback_timings(self, text: str, duration: float) -> List[Tuple[str, float]]:
        """Generate fallback timings."""
        words = [w for w in text.split() if w.strip()]
        if not words:
            return []
        
        per_word = max(self.MIN_WORD_DURATION, duration / len(words))
        result = [(w, per_word) for w in words]
        
        # Adjust last word
        if result:
            total = sum(d for _, d in result)
            diff = duration - total
            if abs(diff) > 0.01:
                word, dur = result[-1]
                result[-1] = (word, max(self.MIN_WORD_DURATION, dur + diff))
        
        return result
    
    # ========================================================================
    # CAPCUT-STYLE ASS WRITER
    # ========================================================================
    
    def _write_capcut_ass(
        self,
        words: List[Tuple[str, float]],
        total_duration: float,
        sentence_type: str,
        output_path: str
    ):
        """
        Write CapCut-style ASS: word-by-word chunks, NO karaoke.
        Each chunk is a separate Dialogue with exact start/end.
        """
        if not words:
            return
        
        # Select style
        style = CAPTION_STYLES[get_random_style()]
        
        is_hook = (sentence_type == "hook")
        fontname = style["fontname"]
        fontsize = style["fontsize_hook"] if is_hook else style["fontsize_normal"]
        outline = style["outline"]
        shadow = style["shadow"]
        margin_v = style["margin_v_hook"] if is_hook else style["margin_v_normal"]
        
        # CapCut colors: vibrant, high contrast
        primary_color = "&H00FFFFFF"   # White text
        outline_color = "&H00000000"   # Black outline
        
        # Build ASS
        ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{fontname},{fontsize},{primary_color},{primary_color},{outline_color},&H00000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,50,50,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        # Group words into chunks (2-3 words)
        max_words = 2 if is_hook else self.WORDS_PER_CHUNK
        
        chunks = []
        current = []
        
        for word, dur in words:
            current.append((word, dur))
            
            # Finalize chunk
            if len(current) >= max_words or word.rstrip().endswith(('.', '!', '?', '…')):
                if current:
                    chunks.append(current)
                    current = []
        
        if current:
            chunks.append(current)
        
        # Write Dialogue events
        cumulative = 0.0
        
        for chunk in chunks:
            chunk_text = " ".join(w.upper() for w, _ in chunk)
            chunk_duration = sum(d for _, d in chunk)
            
            start = cumulative
            end = cumulative + chunk_duration
            
            # Format times (centisecond precision)
            start_str = self._ass_time(start)
            end_str = self._ass_time(end)
            
            # Simple Dialogue - NO effects, perfect sync
            ass += f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{chunk_text}\n"
            
            cumulative = end
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass)
        
        logger.debug(f"      ASS: {len(chunks)} chunks, {cumulative:.3f}s total")
    
    def _ass_time(self, seconds: float) -> str:
        """Format seconds to ASS time (H:MM:SS.CC)"""
        cs = int(round(seconds * 100))
        h = cs // 360000
        cs -= h * 360000
        m = cs // 6000
        cs -= m * 6000
        s = cs // 100
        cs %= 100
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
