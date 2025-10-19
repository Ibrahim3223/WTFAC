# -*- coding: utf-8 -*-
"""
Caption rendering - PERFECT SYNC + CAPCUT SMOOTHNESS
Ultra-precise timing with smooth transitions
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
    """Render captions with CapCut-perfect sync and smooth transitions."""
    
    # Ultra-precise CapCut-style parameters
    WORDS_PER_CHUNK = 3      # Max words per caption line
    MIN_WORD_DURATION = 0.12  # Min 120ms per word (reduced for faster speech)
    TIMING_PRECISION = 0.001  # 1ms precision
    FADE_DURATION = 0.08      # 80ms smooth fade in/out
    
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
                
                # Ultra-precise timing validation
                if words:
                    words = self._ultra_precise_timing(words, duration)
                else:
                    logger.warning(f"      ⚠️ No word timings, using smart fallback")
                    words = self._smart_fallback_timings(text, duration)
                
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
        """Render captions with perfect sync and smooth transitions."""
        try:
            if duration <= 0:
                duration = ffprobe_duration(video_path)
            
            frames = max(2, int(round(duration * settings.TARGET_FPS)))
            output = video_path.replace(".mp4", "_caption.mp4")
            
            if settings.KARAOKE_CAPTIONS and has_subtitles():
                # Create ultra-smooth ASS with fade transitions
                ass_path = video_path.replace(".mp4", ".ass")
                self._write_smooth_ass(words, duration, sentence_type, ass_path)
                
                tmp_out = output.replace(".mp4", ".tmp.mp4")
                
                try:
                    # Add captions with optimized settings
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
                    
                    # Enforce exact frames with precision
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
    # ULTRA-PRECISE TIMING SYSTEM
    # ========================================================================
    
    def _ultra_precise_timing(
        self, 
        word_timings: List[Tuple[str, float]], 
        total_duration: float
    ) -> List[Tuple[str, float]]:
        """
        Ultra-precise timing validation with 1ms accuracy.
        Ensures perfect sync with TTS audio.
        """
        if not word_timings:
            return []
        
        # Remove empty words
        word_timings = [(w.strip(), d) for w, d in word_timings if w.strip()]
        if not word_timings:
            return []
        
        # Calculate raw sum
        sum_raw = sum(d for _, d in word_timings)
        
        # Check if atempo scaling is needed
        if sum_raw > total_duration * 1.02:  # 2% tolerance
            atempo_factor = sum_raw / total_duration
            logger.warning(f"      ⚠️ Applying atempo scale: {atempo_factor:.3f}")
            word_timings = [(word, round(dur / atempo_factor, 3)) for word, dur in word_timings]
            sum_raw = sum(d for _, d in word_timings)
        
        # Build precise timing list
        fixed = []
        cumulative = 0.0
        
        for i, (word, dur) in enumerate(word_timings):
            # Clamp duration with precision
            dur = max(self.MIN_WORD_DURATION, min(dur, 3.0))
            dur = round(dur, 3)  # 1ms precision
            
            # Check if we're about to exceed total duration
            if cumulative + dur > total_duration + self.TIMING_PRECISION:
                remaining = max(0.0, total_duration - cumulative)
                remaining_words = len(word_timings) - i
                
                if remaining > self.MIN_WORD_DURATION and remaining_words > 0:
                    # Distribute remaining time evenly
                    dur_per_word = round(remaining / remaining_words, 3)
                    
                    if dur_per_word >= self.MIN_WORD_DURATION:
                        for j in range(i, len(word_timings)):
                            w = word_timings[j][0]
                            fixed.append((w, dur_per_word))
                
                logger.debug(f"      ⚠️ TTS timing adjusted at word {i+1}/{len(word_timings)}")
                break
            
            fixed.append((word, dur))
            cumulative += dur
        
        # Final precision adjustment
        if fixed:
            total_fixed = sum(d for _, d in fixed)
            diff = total_duration - total_fixed
            
            # Distribute difference across last few words for smoothness
            if abs(diff) > self.TIMING_PRECISION:
                num_adjust = min(3, len(fixed))  # Adjust last 3 words
                adjustment_per_word = round(diff / num_adjust, 3)
                
                for i in range(len(fixed) - num_adjust, len(fixed)):
                    word, dur = fixed[i]
                    new_dur = max(self.MIN_WORD_DURATION, round(dur + adjustment_per_word, 3))
                    fixed[i] = (word, new_dur)
            
            # Verify final timing
            total_fixed = sum(d for _, d in fixed)
            diff_ms = abs(total_fixed - total_duration) * 1000
            
            if diff_ms <= 5:  # Within 5ms
                logger.info(f"      🎯 Perfect sync: {total_fixed:.3f}s / {total_duration:.3f}s (±{diff_ms:.1f}ms)")
            else:
                logger.warning(f"      ⚠️ Sync accuracy: ±{diff_ms:.1f}ms")
        
        return fixed
    
    def _smart_fallback_timings(self, text: str, duration: float) -> List[Tuple[str, float]]:
        """
        Smart fallback timing generation with realistic speech patterns.
        """
        words = [w for w in text.split() if w.strip()]
        if not words:
            return []
        
        # Estimate realistic durations based on word length
        base_durations = []
        total_chars = sum(len(w) for w in words)
        
        for word in words:
            # Longer words take more time (realistic speech pattern)
            word_ratio = len(word) / total_chars if total_chars > 0 else 1.0 / len(words)
            estimated_dur = duration * word_ratio
            estimated_dur = max(self.MIN_WORD_DURATION, min(estimated_dur, 2.0))
            base_durations.append(estimated_dur)
        
        # Normalize to exact duration
        total_estimated = sum(base_durations)
        scale_factor = duration / total_estimated if total_estimated > 0 else 1.0
        
        result = [(words[i], round(base_durations[i] * scale_factor, 3)) 
                  for i in range(len(words))]
        
        # Final adjustment for precision
        total = sum(d for _, d in result)
        diff = duration - total
        
        if abs(diff) > self.TIMING_PRECISION:
            word, dur = result[-1]
            result[-1] = (word, round(dur + diff, 3))
        
        logger.info(f"      📝 Generated {len(result)} word timings (fallback)")
        return result
    
    # ========================================================================
    # SMOOTH ASS WRITER with FADE TRANSITIONS
    # ========================================================================
    
    def _write_smooth_ass(
        self,
        words: List[Tuple[str, float]],
        total_duration: float,
        sentence_type: str,
        output_path: str
    ):
        """
        Write ultra-smooth ASS with fade transitions.
        Each word appears exactly when spoken, with smooth fade in/out.
        """
        if not words:
            return
        
        # Select vibrant style
        style = CAPTION_STYLES[get_random_style()]
        
        is_hook = (sentence_type == "hook")
        fontname = style["fontname"]
        fontsize = style["fontsize_hook"] if is_hook else style["fontsize_normal"]
        outline = style["outline"]
        shadow = style["shadow"]
        margin_v = style["margin_v_hook"] if is_hook else style["margin_v_normal"]
        
        # Get colors from style
        primary_color = style["color_active"]     # Vibrant color
        secondary_color = style["color_inactive"] # White
        outline_color = style["color_outline"]    # Black
        
        # Build ASS header
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
        
        # Group words into smooth chunks
        max_words = 2 if is_hook else self.WORDS_PER_CHUNK
        
        chunks = self._create_smooth_chunks(words, max_words)
        
        # Write Dialogue events with fade transitions
        cumulative = 0.0
        
        for i, chunk in enumerate(chunks):
            chunk_text = " ".join(w.upper() for w, _ in chunk)
            chunk_duration = sum(d for _, d in chunk)
            
            start = cumulative
            end = cumulative + chunk_duration
            
            # Add fade in/out for smoothness
            fade_in = min(self.FADE_DURATION, chunk_duration * 0.2)
            fade_out = min(self.FADE_DURATION, chunk_duration * 0.2)
            
            # Format times with millisecond precision
            start_str = self._ass_time_precise(start)
            end_str = self._ass_time_precise(end)
            
            # Check for emphasis words
            has_emphasis = any(w.strip(".,!?;:").upper() in EMPHASIS_KEYWORDS 
                             for w, _ in chunk)
            
            # Build smooth transition tags
            if has_emphasis:
                # Emphasis: pop in effect
                effect_tags = f"{{\\fad({int(fade_in*1000)},{int(fade_out*1000)})\\t(0,100,\\fscx110\\fscy110)\\t(100,200,\\fscx100\\fscy100)}}"
            else:
                # Normal: smooth fade
                effect_tags = f"{{\\fad({int(fade_in*1000)},{int(fade_out*1000)})}}"
            
            # Write precise Dialogue event
            ass += f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{effect_tags}{chunk_text}\n"
            
            cumulative = end
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass)
        
        logger.debug(f"      ✨ ASS: {len(chunks)} smooth chunks, {cumulative:.3f}s total")
    
    def _create_smooth_chunks(
        self, 
        words: List[Tuple[str, float]], 
        max_words: int
    ) -> List[List[Tuple[str, float]]]:
        """
        Create smooth chunks with natural word grouping.
        Respects punctuation and speech patterns.
        """
        chunks = []
        current = []
        
        for word, dur in words:
            current.append((word, dur))
            
            # Check if we should finalize this chunk
            should_finalize = False
            
            # Natural break points
            if word.rstrip().endswith(('.', '!', '?', '…', ':')):
                should_finalize = True
            
            # Max words reached
            elif len(current) >= max_words:
                should_finalize = True
            
            # Comma with multiple words
            elif ',' in word and len(current) >= 2:
                should_finalize = True
            
            if should_finalize and current:
                chunks.append(current)
                current = []
        
        # Add remaining words
        if current:
            chunks.append(current)
        
        return chunks
    
    def _ass_time_precise(self, seconds: float) -> str:
        """
        Format seconds to ASS time with millisecond precision.
        Format: H:MM:SS.CC (centiseconds)
        """
        # Round to millisecond precision, then convert to centiseconds
        ms = int(round(seconds * 1000))
        cs = ms // 10  # Convert to centiseconds
        
        h = cs // 360000
        cs -= h * 360000
        m = cs // 6000
        cs -= m * 6000
        s = cs // 100
        cs %= 100
        
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
