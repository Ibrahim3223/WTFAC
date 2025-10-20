# -*- coding: utf-8 -*-
"""
Caption rendering - PRODUCTION READY
Excellent sync with stable-ts word-level timings
CRITICAL: Language-aware alignment!
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
    """Render captions with stable-ts word-level precision."""
    
    # Caption parameters
    WORDS_PER_CHUNK = 3
    MIN_WORD_DURATION = 0.12
    TIMING_PRECISION = 0.001
    FADE_DURATION = 0.08
    
    # Smart offset for TTS timings fine-tuning
    CHUNK_OFFSET = -0.15
    
    def __init__(self, caption_offset: Optional[float] = None):
        """
        Initialize caption renderer.
        
        Args:
            caption_offset: Custom timing offset for fine-tuning (optional)
        """
        if caption_offset is not None:
            self.CHUNK_OFFSET = caption_offset
        
        # Get language from settings (CRITICAL for stable-ts accuracy!)
        self.language = getattr(settings, 'LANG', 'en').lower()
        
        logger.info(f"      🎯 Caption renderer: stable-ts ({self.language.upper()}) - WORD-LEVEL precision")
    
    def render_captions(
        self,
        video_segments: List[str],
        audio_segments: List[Dict[str, Any]],
        output_dir: str
    ) -> List[str]:
        """Render captions on all video segments with stable-ts word-level timing."""
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
                
                # Use stable-ts with CORRECT LANGUAGE!
                if audio_path and os.path.exists(audio_path):
                    words = align_text_to_audio(
                        text=text,
                        audio_path=audio_path,
                        tts_word_timings=tts_word_timings,
                        total_duration=duration,
                        language=self.language  # CRITICAL: Pass correct language!
                    )
                    logger.info(f"      ✅ Aligned: {len(words)} words with precise sync ({self.language.upper()})")
                else:
                    # No audio file - use TTS timings or estimation
                    words = self._ultra_precise_timing(tts_word_timings, duration) if tts_word_timings else self._smart_fallback_timings(text, duration)
                
                captioned_path = self.render(
                    video_path=video_path,
                    text=text,
                    words=words,
                    duration=duration,
                    is_hook=is_hook,
                    sentence_type=sentence_type,
                    temp_dir=output_dir,
                    use_offset=False  # stable-ts timings are already precise
                )
                
                if captioned_path and os.path.exists(captioned_path):
                    captioned_segments.append(captioned_path)
                    logger.info(f"      ✅ Caption {i+1} rendered successfully")
                else:
                    logger.error(f"      ❌ Caption rendering failed for segment {i+1}")
                    # CRITICAL: Don't fail entire batch, use uncaptioned video
                    logger.warning(f"      ⚠️ Using uncaptioned video for segment {i+1}")
                    captioned_segments.append(video_path)
                    
            except Exception as e:
                logger.error(f"      ❌ Error rendering caption {i+1}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # CRITICAL: Don't fail entire batch
                logger.warning(f"      ⚠️ Using uncaptioned video for segment {i+1}")
                captioned_segments.append(video_path)
        
        logger.info(f"      ✅ Rendered {len(captioned_segments)} segments (captioned or fallback)")
        return captioned_segments
    
    def render(
        self,
        video_path: str,
        text: str,
        words: List[Tuple[str, float]],
        duration: float,
        is_hook: bool = False,
        sentence_type: str = "buildup",
        temp_dir: str = None,
        use_offset: bool = False
    ) -> str:
        """Render captions with precise sync."""
        try:
            if duration <= 0:
                duration = ffprobe_duration(video_path)
            
            frames = max(2, int(round(duration * settings.TARGET_FPS)))
            output = video_path.replace(".mp4", "_caption.mp4")
            
            if settings.KARAOKE_CAPTIONS and has_subtitles():
                ass_path = video_path.replace(".mp4", ".ass")
                
                # CRITICAL: Safe ASS generation with try-catch
                try:
                    self._write_smooth_ass(words, duration, sentence_type, ass_path, use_offset)
                except Exception as e:
                    logger.error(f"      ❌ ASS generation failed: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    return video_path  # Return uncaptioned video
                
                if not os.path.exists(ass_path):
                    logger.error(f"      ❌ ASS file not created: {ass_path}")
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
                logger.warning(f"      ⚠️ Skipping captions (karaoke disabled or no subtitle support)")
                return video_path
                
        except Exception as e:
            logger.error(f"      ❌ Error in render(): {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return video_path  # Return uncaptioned video instead of failing
    
    # ========================================================================
    # TIMING VALIDATION (only used as last-resort fallback)
    # ========================================================================
    
    def _ultra_precise_timing(
        self, 
        word_timings: List[Tuple[str, float]], 
        total_duration: float
    ) -> List[Tuple[str, float]]:
        """Ultra-precise timing validation with 1ms accuracy."""
        if not word_timings:
            return []
        
        word_timings = [(w.strip(), d) for w, d in word_timings if w.strip()]
        if not word_timings:
            return []
        
        sum_raw = sum(d for _, d in word_timings)
        
        if sum_raw > total_duration * 1.02:
            atempo_factor = sum_raw / total_duration
            logger.debug(f"      ⚠️ Applying atempo scale: {atempo_factor:.3f}")
            word_timings = [(word, round(dur / atempo_factor, 3)) for word, dur in word_timings]
            sum_raw = sum(d for _, d in word_timings)
        
        fixed = []
        cumulative = 0.0
        
        for i, (word, dur) in enumerate(word_timings):
            dur = max(self.MIN_WORD_DURATION, min(dur, 3.0))
            dur = round(dur, 3)
            
            if cumulative + dur > total_duration + self.TIMING_PRECISION:
                remaining = max(0.0, total_duration - cumulative)
                remaining_words = len(word_timings) - i
                
                if remaining > self.MIN_WORD_DURATION and remaining_words > 0:
                    dur_per_word = round(remaining / remaining_words, 3)
                    
                    if dur_per_word >= self.MIN_WORD_DURATION:
                        for j in range(i, len(word_timings)):
                            w = word_timings[j][0]
                            fixed.append((w, dur_per_word))
                
                break
            
            fixed.append((word, dur))
            cumulative += dur
        
        if fixed:
            total_fixed = sum(d for _, d in fixed)
            diff = total_duration - total_fixed
            
            if abs(diff) > self.TIMING_PRECISION:
                num_adjust = min(3, len(fixed))
                adjustment_per_word = round(diff / num_adjust, 3)
                
                for i in range(len(fixed) - num_adjust, len(fixed)):
                    word, dur = fixed[i]
                    new_dur = max(self.MIN_WORD_DURATION, round(dur + adjustment_per_word, 3))
                    fixed[i] = (word, new_dur)
        
        return fixed
    
    def _smart_fallback_timings(self, text: str, duration: float) -> List[Tuple[str, float]]:
        """Smart fallback timing generation."""
        words = [w for w in text.split() if w.strip()]
        if not words:
            return []
        
        base_durations = []
        total_chars = sum(len(w) for w in words)
        
        for word in words:
            word_ratio = len(word) / total_chars if total_chars > 0 else 1.0 / len(words)
            estimated_dur = duration * word_ratio
            estimated_dur = max(self.MIN_WORD_DURATION, min(estimated_dur, 2.0))
            base_durations.append(estimated_dur)
        
        total_estimated = sum(base_durations)
        scale_factor = duration / total_estimated if total_estimated > 0 else 1.0
        
        result = [(words[i], round(base_durations[i] * scale_factor, 3)) 
                  for i in range(len(words))]
        
        total = sum(d for _, d in result)
        diff = duration - total
        
        if abs(diff) > self.TIMING_PRECISION:
            word, dur = result[-1]
            result[-1] = (word, round(dur + diff, 3))
        
        return result
    
    # ========================================================================
    # SMOOTH ASS WRITER - CLEAN (NO EMOJI/WATERMARK)
    # ========================================================================
    
    def _write_smooth_ass(
        self,
        words: List[Tuple[str, float]],
        total_duration: float,
        sentence_type: str,
        output_path: str,
        use_offset: bool = False
    ):
        """Write ultra-smooth ASS with fade transitions and NO overlap."""
        if not words:
            logger.warning("      ⚠️ No words to write to ASS")
            return
        
        try:
            style = CAPTION_STYLES[get_random_style()]
        except Exception as e:
            logger.warning(f"      ⚠️ Failed to get random style: {e}, using classic_yellow")
            style = CAPTION_STYLES["classic_yellow"]
        
        is_hook = (sentence_type == "hook")
        fontname = style.get("fontname", "Arial Black")
        fontsize = style.get("fontsize_hook" if is_hook else "fontsize_normal", 60)
        outline = style.get("outline", 7)
        shadow = style.get("shadow", "5")
        
        # CRITICAL FIX: Safe margin_v access with fallback
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
        chunks = self._create_smooth_chunks(words, max_words)
        
        cumulative = 0.0
        prev_end = 0.0  # Track previous caption end time
        
        for i, chunk in enumerate(chunks):
            # CLEAN TEXT - NO EMOJI, NO WATERMARK
            chunk_text = " ".join(w.upper() for w, _ in chunk)
            chunk_duration = sum(d for _, d in chunk)
            
            if use_offset:
                offset = self.CHUNK_OFFSET * 1.3 if is_hook else self.CHUNK_OFFSET
                start = max(0.0, cumulative + offset)
            else:
                start = cumulative
            
            # CRITICAL: Ensure NO overlap - wait for previous to finish
            if i > 0 and start < prev_end:
                start = prev_end + 0.05  # 50ms gap minimum
            
            end = start + chunk_duration
            
            if end > total_duration:
                end = total_duration
                if start >= end:
                    break
            
            # Reduce fade duration to prevent overlap
            fade_in = min(self.FADE_DURATION, chunk_duration * 0.15)
            fade_out = min(self.FADE_DURATION, chunk_duration * 0.15)
            
            # Shorten end by fade_out to prevent overlap
            display_end = end - (fade_out * 0.5)
            
            start_str = self._ass_time_precise(start)
            end_str = self._ass_time_precise(display_end)
            
            has_emphasis = any(w.strip(".,!?;:").upper() in EMPHASIS_KEYWORDS 
                             for w, _ in chunk)
            
            if has_emphasis:
                effect_tags = f"{{\\fad({int(fade_in*1000)},{int(fade_out*1000)})\\t(0,120,\\fscx115\\fscy115)\\t(120,240,\\fscx100\\fscy100)}}"
            else:
                effect_tags = f"{{\\fad({int(fade_in*1000)},{int(fade_out*1000)})}}"
            
            ass += f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{effect_tags}{chunk_text}\n"
            
            # Update tracking
            prev_end = display_end
            cumulative += chunk_duration
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass)
        
        logger.debug(f"      ✅ ASS written: {output_path}")
    
    def _create_smooth_chunks(
        self, 
        words: List[Tuple[str, float]], 
        max_words: int
    ) -> List[List[Tuple[str, float]]]:
        """Create smooth chunks with natural word grouping."""
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
    
    def _ass_time_precise(self, seconds: float) -> str:
        """Format seconds to ASS time with millisecond precision."""
        ms = int(round(seconds * 1000))
        cs = ms // 10
        
        h = cs // 360000
        cs -= h * 360000
        m = cs // 6000
        cs -= m * 6000
        s = cs // 100
        cs %= 100
        
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
