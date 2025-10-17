# -*- coding: utf-8 -*-
"""Caption rendering on video."""
import os
import pathlib
from typing import List, Tuple, Optional

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run, has_subtitles, quantize_to_frames, ffprobe_duration
from .karaoke_ass import build_karaoke_ass

class CaptionRenderer:
    """Render captions on video."""
    
    def render(
        self,
        video_path: str,
        text: str,
        words: Optional[List[Tuple[str, float]]] = None,
        is_hook: bool = False,
        temp_dir: str = None
    ) -> str:
        """Render captions using ASS karaoke."""
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
                run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", video_path,
                    "-vf", f"subtitles='{ass_path}',setsar=1,fps={settings.TARGET_FPS}",
                    "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                    "-an", "-c:v", "libx264", "-preset", "medium",
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
                    "-an", "-c:v", "libx264", "-preset", "medium",
                    "-crf", str(settings.CRF_VISUAL),
                    "-pix_fmt", "yuv420p",
                    output
                ])
                
            finally:
                pathlib.Path(ass_path).unlink(missing_ok=True)
                pathlib.Path(tmp_out).unlink(missing_ok=True)
            
            return output
        
        # No captions required
        return video_path
