# -*- coding: utf-8 -*-
"""Caption rendering on video."""
import os
import pathlib
import logging
from typing import List, Tuple, Optional, Dict, Any

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run, has_subtitles, quantize_to_frames, ffprobe_duration
from .karaoke_ass import build_karaoke_ass

logger = logging.getLogger(__name__)


class CaptionRenderer:
    """Render captions on video."""
    
    def render_captions(
        self,
        video_segments: List[str],
        audio_segments: List[Dict[str, Any]],
        output_dir: str
    ) -> List[str]:
        """
        Render captions on all video segments.
        
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
                words = audio_segment.get("word_timings")
                is_hook = (i == 0)  # First segment is the hook
                
                logger.info(f"      Rendering caption {i+1}/{len(video_segments)}: {text[:50]}...")
                
                captioned_path = self.render(
                    video_path=video_path,
                    text=text,
                    words=words,
                    is_hook=is_hook,
                    temp_dir=output_dir
                )
                
                if captioned_path and os.path.exists(captioned_path):
                    captioned_segments.append(captioned_path)
                else:
                    logger.error(f"      ❌ Caption rendering failed for segment {i+1}")
                    return []
                    
            except Exception as e:
                logger.error(f"      ❌ Error rendering caption {i+1}: {e}")
                return []
        
        logger.info(f"      ✅ Rendered {len(captioned_segments)} captioned segments")
        return captioned_segments
    
    def render(
        self,
        video_path: str,
        text: str,
        words: Optional[List[Tuple[str, float]]] = None,
        is_hook: bool = False,
        temp_dir: str = None
    ) -> str:
        """
        Render captions on a single video using ASS karaoke.
        
        Args:
            video_path: Input video file path
            text: Caption text to render
            words: Optional word timings for karaoke effect
            is_hook: Whether this is the hook segment (affects styling)
            temp_dir: Temporary directory for intermediate files
            
        Returns:
            Path to output video with captions
        """
        try:
            seg_dur = ffprobe_duration(video_path)
            frames = max(2, int(round(seg_dur * settings.TARGET_FPS)))
            
            output = video_path.replace(".mp4", "_caption.mp4")
            
            if settings.KARAOKE_CAPTIONS and has_subtitles():
                # Use ASS karaoke
                ass_txt = build_karaoke_ass(text, seg_dur, words or [], is_hook)
                ass_path = video_path.replace(".mp4", ".ass")
                pathlib.Path(ass_path).write_text(ass_txt, encoding="utf-8")
                
                tmp_out = output.replace(".mp4", ".tmp.mp4")
                
                try:
                    # First pass: Add captions (video only)
                    run([
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", video_path,
                        "-vf", f"subtitles='{ass_path}',setsar=1,fps={settings.TARGET_FPS}",
                        "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                        "-an",  # Remove audio here (will be added later in orchestrator)
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
                        "-an",  # Still no audio - orchestrator will add it
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
                logger.warning(f"Skipping captions for {video_path}")
                return video_path
                
        except Exception as e:
            logger.error(f"Error in render(): {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return video_path  # Return original if caption rendering fails
