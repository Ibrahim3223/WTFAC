# -*- coding: utf-8 -*-
"""
Caption rendering on video - ENHANCED WITH CAPCUT CHUNKING
Word-by-word rendering with 2-3 word chunks for maximum readability
"""
import os
import pathlib
import logging
from typing import List, Tuple, Optional, Dict, Any

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run, has_subtitles, quantize_to_frames, ffprobe_duration
from autoshorts.captions.karaoke_ass import CAPTION_STYLES, get_random_style, EMPHASIS_KEYWORDS

logger = logging.getLogger(__name__)


class CaptionRenderer:
    """Render captions on video with CapCut-style chunking."""
    
    # CapCut-style chunking parameters
    WORDS_PER_CHUNK = 3      # Max 3 words per caption block
    MIN_CHUNK_DURATION = 0.8  # Minimum 0.8s per chunk
    MAX_CHUNK_DURATION = 2.5  # Maximum 2.5s per chunk
    
    def render_captions(
        self,
        video_segments: List[str],
        audio_segments: List[Dict[str, Any]],
        output_dir: str
    ) -> List[str]:
        """
        Render captions on all video segments with CapCut-style chunking.
        
        Args:
            video_segments: List of video file paths
            audio_segments: List of audio segment dicts with text and word_timings
            output_dir: Output directory for captioned videos
            
        Returns:
            List of paths to captioned video segments
        """
        captioned_segments = []
        
        for i, (video_path, audio_segment) in enumerate(zip(video_segments, audio_segments)):
            try:
                text = audio_segment["text"]
                words = audio_segment.get("word_timings", [])
                duration = audio_segment.get("duration", 0)
                sentence_type = audio_segment.get("type", "buildup")
                is_hook = (i == 0 or sentence_type == "hook")
                
                logger.info(f"      Rendering caption {i+1}/{len(video_segments)}: {text[:50]}...")
                
                # CRITICAL: Validate and fix word timings
                if words:
                    words = self._validate_and_fix_timings(words, duration)
                else:
                    logger.warning(f"      ⚠️ No word timings, using fallback")
                    words = self._fallback_word_timings(text, duration)
                
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
        words: Optional[List[Tuple[str, float]]] = None,
        duration: float = 0,
        is_hook: bool = False,
        sentence_type: str = "buildup",
        temp_dir: str = None
    ) -> str:
        """
        Render captions on a single video using ASS karaoke with CapCut-style chunks.
        
        Args:
            video_path: Input video file path
            text: Caption text to render
            words: Word timings for karaoke effect
            duration: Video duration in seconds
            is_hook: Whether this is the hook segment (affects styling)
            sentence_type: Type of sentence (hook/buildup/payoff/cta)
            temp_dir: Temporary directory for intermediate files
            
        Returns:
            Path to output video with captions
        """
        try:
            # Get video duration
            if duration <= 0:
                duration = ffprobe_duration(video_path)
            
            frames = max(2, int(round(duration * settings.TARGET_FPS)))
            output = video_path.replace(".mp4", "_caption.mp4")
            
            if settings.KARAOKE_CAPTIONS and has_subtitles():
                # Create CapCut-style chunked ASS
                chunks = self._create_word_chunks(text, words or [], sentence_type)
                
                if not chunks:
                    logger.warning(f"      ⚠️ No chunks created, skipping captions")
                    return video_path
                
                logger.debug(f"      Created {len(chunks)} caption chunks")
                
                # Write chunked ASS file
                ass_path = video_path.replace(".mp4", ".ass")
                self._write_chunked_ass(chunks, duration, sentence_type, ass_path)
                
                tmp_out = output.replace(".mp4", ".tmp.mp4")
                
                try:
                    # First pass: Add captions (video only)
                    run([
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", video_path,
                        "-vf", f"subtitles='{ass_path}',setsar=1,fps={settings.TARGET_FPS}",
                        "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                        "-an",  # Remove audio (will be added later)
                        "-c:v", "libx264", "-preset", "medium",
                        "-crf", str(max(16, settings.CRF_VISUAL - 3)),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        tmp_out
                    ])
                    
                    # Second pass: Enforce exact frames (video only)
                    run([
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", tmp_out,
                        "-vf", f"setsar=1,fps={settings.TARGET_FPS},trim=start_frame=0:end_frame={frames}",
                        "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                        "-an",  # Still no audio
                        "-c:v", "libx264", "-preset", "medium",
                        "-crf", str(settings.CRF_VISUAL),
                        "-pix_fmt", "yuv420p",
                        output
                    ])
                    
                finally:
                    # Cleanup temporary files
                    pathlib.Path(ass_path).unlink(missing_ok=True)
                    pathlib.Path(tmp_out).unlink(missing_ok=True)
                
                return output
            else:
                # No captions required or subtitles not available
                logger.warning(f"      Skipping captions for {video_path}")
                return video_path
                
        except Exception as e:
            logger.error(f"      ❌ Error in render(): {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return video_path  # Return original if caption rendering fails
    
    # ========================================================================
    # NEW METHODS: CapCut-Style Chunking
    # ========================================================================
    
    def _validate_and_fix_timings(
        self, 
        word_timings: List[Tuple[str, float]], 
        total_duration: float
    ) -> List[Tuple[str, float]]:
        """
        CRITICAL: Validate and fix word timings to match audio duration.
        
        Fixes:
        1. Timings exceed audio duration (TTS cut off)
        2. Last words have zero duration
        3. Unrealistic durations
        
        Args:
            word_timings: List of (word, duration) tuples
            total_duration: Actual audio duration in seconds
            
        Returns:
            Fixed word timings
        """
        if not word_timings:
            return []
        
        fixed_timings = []
        cumulative_time = 0.0
        
        for i, (word, duration) in enumerate(word_timings):
            # Skip empty words
            if not word.strip():
                continue
            
            # Fix: Ensure duration is reasonable
            if duration < 0.1:
                duration = 0.25  # Minimum 250ms per word
            elif duration > 2.0:
                duration = 2.0  # Maximum 2s per word
            
            # Fix: If we're exceeding total duration, truncate
            if cumulative_time + duration > total_duration:
                # TTS was cut off - distribute remaining time
                remaining_time = max(0.1, total_duration - cumulative_time)
                remaining_words = len(word_timings) - i
                
                if remaining_words > 0:
                    duration = remaining_time / remaining_words
                else:
                    duration = remaining_time
                
                # Only add if we have time left
                if duration > 0.05:
                    fixed_timings.append((word, duration))
                    cumulative_time += duration
                
                # Stop processing - TTS cut off here
                logger.warning(f"      ⚠️ TTS cut off at word '{word}' ({i+1}/{len(word_timings)}) - truncating captions")
                break
            
            fixed_timings.append((word, duration))
            cumulative_time += duration
        
        # Log the fix
        total_fixed = sum(d for _, d in fixed_timings)
        if len(fixed_timings) != len(word_timings):
            logger.info(f"      📝 Fixed timings: {len(word_timings)} → {len(fixed_timings)} words")
        
        logger.debug(f"      Timing validation: {total_fixed:.2f}s / {total_duration:.2f}s")
        
        return fixed_timings
    
    def _create_word_chunks(
        self,
        text: str,
        word_timings: List[Tuple[str, float]],
        sentence_type: str
    ) -> List[Dict[str, Any]]:
        """
        Create CapCut-style caption chunks (2-3 words per chunk).
        
        Args:
            text: Full sentence text
            word_timings: List of (word, duration) tuples
            sentence_type: Type of sentence (hook/buildup/payoff/cta)
            
        Returns:
            List of chunks with metadata
        """
        if not word_timings:
            return []
        
        chunks = []
        current_chunk_words = []
        current_chunk_duration = 0.0
        chunk_start_time = 0.0
        
        # Determine words per chunk based on sentence type
        if sentence_type == "hook":
            max_words = 2  # Hook: 2 words max (faster pacing)
        else:
            max_words = self.WORDS_PER_CHUNK  # Normal: 3 words max
        
        cumulative_time = 0.0
        
        for i, (word, duration) in enumerate(word_timings):
            # Add word to current chunk
            current_chunk_words.append((word, duration))
            current_chunk_duration += duration
            
            # Check if we should finalize this chunk
            should_finalize = False
            
            # Rule 1: Max words reached
            if len(current_chunk_words) >= max_words:
                should_finalize = True
            
            # Rule 2: Max duration reached
            elif current_chunk_duration >= self.MAX_CHUNK_DURATION:
                should_finalize = True
            
            # Rule 3: Natural break (punctuation)
            elif word.rstrip().endswith((',', '.', '!', '?', ':', ';', '—', '…')):
                should_finalize = True
            
            # Rule 4: Last word
            elif i == len(word_timings) - 1:
                should_finalize = True
            
            if should_finalize and current_chunk_words:
                # Create chunk
                chunk_text = " ".join(w for w, _ in current_chunk_words)
                
                chunk = {
                    "text": chunk_text.upper(),  # CapCut uses uppercase
                    "words": current_chunk_words,
                    "start_time": chunk_start_time,
                    "duration": current_chunk_duration
                }
                
                chunks.append(chunk)
                
                # Reset for next chunk
                chunk_start_time = cumulative_time + current_chunk_duration
                current_chunk_words = []
                current_chunk_duration = 0.0
            
            cumulative_time += duration
        
        logger.debug(f"      Created {len(chunks)} chunks from {len(word_timings)} words")
        return chunks
    
    def _write_chunked_ass(
        self,
        chunks: List[Dict[str, Any]],
        total_duration: float,
        sentence_type: str,
        output_path: str
    ):
        """
        Write ASS file with chunked captions.
        Each chunk is a separate Dialogue event.
        
        Args:
            chunks: List of caption chunks
            total_duration: Total video duration
            sentence_type: Type of sentence
            output_path: Path to output ASS file
        """
        # Select caption style
        style = CAPTION_STYLES[get_random_style()]
        
        # Get style parameters
        fontname = style["fontname"]
        fontsize = style["fontsize_hook"] if sentence_type == "hook" else style["fontsize_normal"]
        fontsize_emphasis = style["fontsize_emphasis"]
        outline = style["outline"]
        shadow = style["shadow"]
        margin_v = style["margin_v_hook"] if sentence_type == "hook" else style["margin_v_normal"]
        
        # Colors
        inactive = style["color_inactive"]
        active = style["color_active"]
        outline_c = style["color_outline"]
        emphasis_c = style["color_emphasis"]
        
        # Build ASS header
        ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Base,{fontname},{fontsize},{inactive},{active},{outline_c},&H7F000000,1,0,0,0,100,100,0,0,1,{outline},{shadow},2,50,50,{margin_v},0
Style: Emphasis,{fontname},{fontsize_emphasis},{emphasis_c},{emphasis_c},{outline_c},&H7F000000,1,0,0,0,100,100,0,0,1,{outline + 1},{shadow},2,50,50,{margin_v},0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        # Add each chunk as a separate Dialogue event
        for chunk in chunks:
            start = chunk["start_time"]
            end = start + chunk["duration"]
            words = chunk["words"]
            
            # Build karaoke tags for word-by-word reveal
            karaoke_text = ""
            for word, word_dur in words:
                word_upper = word.upper()
                duration_cs = max(8, int(round(word_dur * 100)))  # Centiseconds
                
                # Check if emphasis word
                clean_word = word_upper.strip(".,!?;:—…")
                if clean_word in EMPHASIS_KEYWORDS:
                    # Emphasis style with stronger bounce
                    karaoke_text += f"{{\\k{duration_cs}\\fs{fontsize_emphasis}\\c{emphasis_c}\\t(0,40,\\fscx110\\fscy110)\\t(40,80,\\fscx100\\fscy100)}}{word_upper} "
                else:
                    # Normal style with subtle bounce
                    karaoke_text += f"{{\\k{duration_cs}\\t(0,50,\\fscx105\\fscy105)\\t(50,100,\\fscx100\\fscy100)}}{word_upper} "
            
            karaoke_text = karaoke_text.strip()
            
            # Format times
            start_time = self._format_ass_time(start)
            end_time = self._format_ass_time(end)
            
            # Add dialogue line
            ass_content += f"Dialogue: 0,{start_time},{end_time},Base,,0,0,{margin_v},,{karaoke_text}\n"
        
        # Write file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
        
        logger.debug(f"      Wrote ASS with {len(chunks)} chunks to {output_path}")
    
    def _format_ass_time(self, seconds: float) -> str:
        """Format seconds to ASS time format (0:00:00.00)"""
        h = int(seconds // 3600)
        seconds -= h * 3600
        m = int(seconds // 60)
        seconds -= m * 60
        return f"{h:d}:{m:02d}:{seconds:05.2f}"
    
    def _fallback_word_timings(self, text: str, duration: float) -> List[Tuple[str, float]]:
        """
        Generate fallback word timings if TTS doesn't provide them.
        
        Args:
            text: Full text
            duration: Total duration in seconds
            
        Returns:
            List of (word, duration) tuples
        """
        words = text.split()
        if not words:
            return []
        
        # Distribute duration evenly
        per_word = duration / len(words)
        per_word = max(0.2, min(per_word, 1.0))  # Clamp between 0.2-1.0s
        
        logger.debug(f"      Generated fallback timings: {len(words)} words, {per_word:.2f}s each")
        return [(word, per_word) for word in words]
