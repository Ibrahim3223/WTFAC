# -*- coding: utf-8 -*-
"""
Caption rendering - BULLETPROOF SYNC
No animation drift, EXACT timing, aggressive validation
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
    """Render captions with BULLETPROOF sync (no drift)."""
    
    # Caption parameters
    WORDS_PER_CHUNK = 3
    MIN_WORD_DURATION = 0.08
    TIMING_PRECISION = 0.001
    FADE_DURATION = 0.0  # NO FADE - critical for sync!
    
    def __init__(self, caption_offset: Optional[float] = None):
        """Initialize caption renderer."""
        # Get language from settings
        self.language = getattr(settings, 'LANG', 'en').lower()
        logger.info(f"      🎯 Caption renderer: stable-ts ({self.language.upper()}) - WORD-LEVEL precision")
    
    def render_captions(
        self,
        video_segments: List[str],
        audio_segments: List[Dict[str, Any]],
        output_dir: str
    ) -> List[str]:
        """Render captions with BULLETPROOF sync."""
        from autoshorts.captions.forced_aligner import align_text_to_audio
        
        captioned_segments = []
        
        for i, (video_path, audio_segment) in enumerate(zip(video_segments, audio_segments)):
            try:
                text = audio_segment["text"]
                tts_word_timings = audio_segment.get("word_timings", [])
                duration = audio_segment.get("duration", 0)
                audio_path = audio_segment.get("audio_path")
                sentence_type = audio_segment.get("type", "buildup")
                is_hook = (i == 0 or sentence_type == "hook")
                
                logger.info(f"      Rendering caption {i+1}/{len(video_segments)}: {text[:50]}...")
                
                # Get word timings (with language!)
                if audio_path and os.path.exists(audio_path):
                    words = align_text_to_audio(
                        text=text,
                        audio_path=audio_path,
                        tts_word_timings=tts_word_timings,
                        total_duration=duration,
                        language=self.language
                    )
                    logger.info(f"      ✅ Aligned: {len(words)} words with precise sync ({self.language.upper()})")
                else:
                    words = self._smart_fallback_timings(text, duration)
                
                # CRITICAL: AGGRESSIVE VALIDATION
                words = self._aggressive_validate(words, duration)
                
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
                    logger.info(f"      ✅ Caption {i+1} rendered successfully")
                else:
                    logger.warning(f"      ⚠️ Using uncaptioned video for segment {i+1}")
                    captioned_segments.append(video_path)
                    
            except Exception as e:
                logger.error(f"      ❌ Error rendering caption {i+1}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                captioned_segments.append(video_path)
        
        logger.info(f"      ✅ Rendered {len(captioned_segments)} segments")
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
        """Render captions with EXACT timing."""
        try:
            if duration <= 0:
                duration = ffprobe_duration(video_path)
            
            frames = max(2, int(round(duration * settings.TARGET_FPS)))
            output = video_path.replace(".mp4", "_caption.mp4")
            
            if settings.KARAOKE_CAPTIONS and has_subtitles():
                ass_path = video_path.replace(".mp4", ".ass")
                
                try:
                    self._write_exact_ass(words, duration, sentence_type, ass_path)
                except Exception as e:
                    logger.error(f"      ❌ ASS generation failed: {e}")
                    return video_path
                
                if not os.path.exists(ass_path):
                    logger.error(f"      ❌ ASS file not created")
                    return video_path
                
                tmp_out = output.replace(".mp4", ".tmp.mp4")
                
                try:
                    run([
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", video_path,
                        "-vf", f"subtitles='{ass_path}':force_style='Kerning=1',setsar=1,fps={settings.TARGET_FPS}",
                        "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                        "-an",
                        "-c:v", "libx264", "-preset", "medium",
                        "-crf", str(max(16, settings.CRF_VISUAL - 3)),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        tmp_out
                    ])
                    
                    if not os.path.exists(tmp_out):
                        logger.error(f"      ❌ FFmpeg subtitle burn failed")
                        return video_path
                    
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
                    
                    if not os.path.exists(output):
                        logger.error(f"      ❌ FFmpeg final output failed")
                        return video_path
                    
                finally:
                    pathlib.Path(ass_path).unlink(missing_ok=True)
                    pathlib.Path(tmp_out).unlink(missing_ok=True)
                
                return output
            else:
                return video_path
                
        except Exception as e:
            logger.error(f"      ❌ Error in render(): {e}")
            return video_path
    
    # ========================================================================
    # AGGRESSIVE VALIDATION - Prevents ALL drift
    # ========================================================================
    
    def _aggressive_validate(
        self,
        word_timings: List[Tuple[str, float]],
        total_duration: float
    ) -> List[Tuple[str, float]]:
        """
        AGGRESSIVE validation to prevent segment-internal drift.
        
        This is the CRITICAL fix that prevents sync issues.
        """
        if not word_timings:
            return []
        
        # Clean
        word_timings = [(w.strip(), d) for w, d in word_timings if w.strip()]
        if not word_timings:
            return []
        
        # Enforce min/max bounds
        fixed = []
        for word, dur in word_timings:
            dur = max(self.MIN_WORD_DURATION, min(dur, 3.0))
            fixed.append((word, round(dur, 3)))
        
        # CRITICAL: Check total sum
        current_total = sum(d for _, d in fixed)
        
        # If total exceeds target by >0.5%, scale DOWN
        if current_total > total_duration * 1.005:
            scale = total_duration / current_total
            logger.info(f"      📏 Scaling DOWN: {scale:.3f}x (prevent overflow)")
            fixed = [
                (w, max(self.MIN_WORD_DURATION, round(d * scale, 3)))
                for w, d in fixed
            ]
        
        # EXACT match (±1ms tolerance)
        current_total = sum(d for _, d in fixed)
        diff = total_duration - current_total
        
        if abs(diff) > 0.001:
            # Distribute difference across ALL words proportionally
            for i in range(len(fixed)):
                word, dur = fixed[i]
                weight = dur / current_total if current_total > 0 else 1.0 / len(fixed)
                adjustment = diff * weight
                new_dur = max(self.MIN_WORD_DURATION, round(dur + adjustment, 3))
                fixed[i] = (word, new_dur)
        
        # Final check
        final_total = sum(d for _, d in fixed)
        diff_ms = abs(final_total - total_duration) * 1000
        
        if diff_ms > 1.0:
            # Last resort: adjust last word
            last_word, last_dur = fixed[-1]
            final_diff = total_duration - final_total
            fixed[-1] = (last_word, max(self.MIN_WORD_DURATION, round(last_dur + final_diff, 3)))
        
        # Validation log
        validated_total = sum(d for _, d in fixed)
        diff_ms = abs(validated_total - total_duration) * 1000
        logger.info(f"      ✅ Validated: {validated_total:.3f}s (target: {total_duration:.3f}s, diff: {diff_ms:.1f}ms)")
        
        return fixed
    
    def _smart_fallback_timings(self, text: str, duration: float) -> List[Tuple[str, float]]:
        """Smart fallback timing generation."""
        words = [w for w in text.split() if w.strip()]
        if not words:
            return []
        
        # Character-based weights
        total_chars = sum(len(w) for w in words)
        if total_chars == 0:
            dur_per_word = duration / len(words)
            return [(w, round(dur_per_word, 3)) for w in words]
        
        word_timings = []
        for word in words:
            char_ratio = len(word) / total_chars
            dur = max(self.MIN_WORD_DURATION, duration * char_ratio)
            word_timings.append((word, round(dur, 3)))
        
        return word_timings
    
    # ========================================================================
    # EXACT ASS WRITER - NO ANIMATION, PURE TIMING
    # ========================================================================
    
    def _write_exact_ass(
        self,
        words: List[Tuple[str, float]],
        total_duration: float,
        sentence_type: str,
        output_path: str
    ):
        """
        Write ASS with EXACT timing (no animation drift).
        
        CRITICAL: No scale animations, minimal fade, exact cumulative timing.
        """
        if not words:
            logger.warning("      ⚠️ No words to write to ASS")
            return
        
        try:
            style = CAPTION_STYLES[get_random_style()]
        except Exception:
            style = CAPTION_STYLES["classic_yellow"]
        
        is_hook = (sentence_type == "hook")
        fontname = style.get("fontname", "Arial Black")
        fontsize = style.get("fontsize_hook" if is_hook else "fontsize_normal", 60)
        outline = style.get("outline", 7)
        shadow = style.get("shadow", "5")
        margin_v = style.get("margin_v", 320)
        
        primary_color = style.get("color_active", "&H0000FFFF")
        secondary_color = style.get("color_inactive", "&H00FFFFFF")
        outline_color = style.get("color_outline", "&H00000000")
        
        ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{fontname},{fontsize},{primary_color},{secondary_color},{outline_color},&H00000000,-1,0,0,0,100,100,1,0,1,{outline},{shadow},2,50,50,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        max_words = 2 if is_hook else self.WORDS_PER_CHUNK
        chunks = self._create_chunks(words, max_words)
        
        # CRITICAL: CHUNK-LEVEL VALIDATION (prevents intra-segment drift!)
        chunks = self._validate_chunks(chunks, total_duration)
        
        # CRITICAL: Exact cumulative timing with CHUNK-LEVEL validation
        cumulative_time = 0.0
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_text = " ".join(w.upper() for w, _ in chunk)
            
            # Calculate exact chunk duration from word timings
            chunk_duration = sum(d for _, d in chunk)
            
            # EXACT start/end
            start = cumulative_time
            end = start + chunk_duration
            
            # Don't exceed total (safety check)
            if end > total_duration + 0.001:  # Allow 1ms tolerance
                end = total_duration
                if start >= end:
                    break
            
            # CRITICAL: Frame-align timing for PERFECT FFmpeg sync!
            frame_duration = 1.0 / settings.TARGET_FPS
            start_frame = round(start / frame_duration)
            end_frame = round(end / frame_duration)
            
            # Convert back to seconds (frame-aligned)
            start_aligned = start_frame * frame_duration
            end_aligned = end_frame * frame_duration
            
            # Convert to ASS time strings
            start_str = self._ass_time(start_aligned)
            end_str = self._ass_time(end_aligned)
            
            # CRITICAL: Calculate ACTUAL ASS duration after centisecond + frame rounding
            ass_start = self._ass_to_seconds(start_str)
            ass_end = self._ass_to_seconds(end_str)
            
            # Log frame alignment if significant
            frame_shift_ms = abs(start - start_aligned) * 1000
            if frame_shift_ms > 5.0:
                logger.info(f"      🎬 Chunk {chunk_idx+1} frame-aligned: {frame_shift_ms:.1f}ms shift")
            
            # NO EFFECTS for perfect sync!
            effect_tags = ""
            
            ass += f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{effect_tags}{chunk_text}\n"
            
            # Update cumulative - use ACTUAL ASS end time (frame-aligned!)
            cumulative_time = ass_end
        
        # Validation - cumulative_time now reflects ACTUAL frame-aligned ASS timing
        diff_ms = abs(cumulative_time - total_duration) * 1000
        if diff_ms > 10:
            logger.warning(f"      ⚠️ Final drift: {diff_ms:.1f}ms (frame-aligned)")
        else:
            logger.info(f"      ✅ Frame-perfect sync: {cumulative_time:.3f}s (drift: {diff_ms:.1f}ms)")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass)
    
    def _create_chunks(
        self,
        words: List[Tuple[str, float]],
        max_words: int
    ) -> List[List[Tuple[str, float]]]:
        """Create natural chunks."""
        chunks = []
        current = []
        
        for word, dur in words:
            current.append((word, dur))
            
            should_finalize = False
            
            if word.rstrip().endswith(('.', '!', '?', '…', ':')):
                should_finalize = True
            elif len(current) >= max_words:
                should_finalize = True
            elif ',' in word and len(current) >= 2:
                should_finalize = True
            
            if should_finalize and current:
                chunks.append(current)
                current = []
        
        if current:
            chunks.append(current)
        
        return chunks
    
    def _validate_chunks(
        self,
        chunks: List[List[Tuple[str, float]]],
        total_duration: float
    ) -> List[List[Tuple[str, float]]]:
        """
        CRITICAL: Validate each chunk to prevent intra-segment drift.
        
        This ensures exact timing within each segment by adjusting
        chunk durations to match the total duration exactly.
        """
        if not chunks:
            return chunks
        
        # Calculate current total
        current_total = sum(sum(d for _, d in chunk) for chunk in chunks)
        
        # ALWAYS validate chunks for logging, even if close
        validated_chunks = []
        remaining_duration = total_duration
        
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            
            chunk_dur = sum(d for _, d in chunk)
            
            if is_last:
                # Last chunk gets exactly remaining duration
                target_dur = remaining_duration
            else:
                # Scale proportionally based on current chunk weight
                weight = chunk_dur / current_total if current_total > 0 else 1.0 / len(chunks)
                target_dur = total_duration * weight
            
            # Adjust chunk words to exact target duration
            if abs(chunk_dur - target_dur) > 0.001:
                scale = target_dur / chunk_dur if chunk_dur > 0 else 1.0
                chunk = [(w, max(self.MIN_WORD_DURATION, round(d * scale, 3))) for w, d in chunk]
                actual_dur = sum(d for _, d in chunk)
                
                logger.info(f"      📏 Chunk {i+1}/{len(chunks)}: {actual_dur:.3f}s (target: {target_dur:.3f}s)")
            else:
                logger.info(f"      📏 Chunk {i+1}/{len(chunks)}: {chunk_dur:.3f}s (perfect)")
            
            validated_chunks.append(chunk)
            remaining_duration -= sum(d for _, d in chunk)
        
        return validated_chunks
    
    def _ass_time(self, seconds: float) -> str:
        """
        Format seconds to ASS time with MAXIMUM precision.
        
        Uses millisecond-level calculations to minimize rounding errors.
        """
        # Work in milliseconds for precision
        total_ms = int(round(seconds * 1000))
        
        # Convert to centiseconds (ASS format requirement)
        cs = total_ms // 10
        
        h = cs // 360000
        cs -= h * 360000
        m = cs // 6000
        cs -= m * 6000
        s = cs // 100
        cs %= 100
        
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
    
    def _ass_to_seconds(self, ass_time: str) -> float:
        """
        Convert ASS time string back to seconds.
        
        CRITICAL: This gives us the ACTUAL timing after centisecond rounding,
        allowing us to track cumulative time without drift.
        """
        # Parse "h:mm:ss.cc" format
        parts = ass_time.split(':')
        h = int(parts[0])
        m = int(parts[1])
        s_and_cs = parts[2].split('.')
        s = int(s_and_cs[0])
        cs = int(s_and_cs[1]) if len(s_and_cs) > 1 else 0
        
        # Convert to seconds (centisecond precision)
        total_seconds = h * 3600 + m * 60 + s + cs * 0.01
        
        return total_seconds
