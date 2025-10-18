# -*- coding: utf-8 -*-
"""Video segment creation with motion effects."""
import os
import random
from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run, quantize_to_frames

class SegmentMaker:
    """Create video segments from sources."""
    
    def create(self, video_src: str, duration: float, temp_dir: str, index: int) -> str:
        """Create segment with optional Ken Burns effect."""
        frames, qdur = quantize_to_frames(duration, settings.TARGET_FPS)
        
        is_image = video_src.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        
        output = os.path.join(temp_dir, f"seg_{index:02d}.mp4")
        
        if is_image:
            # Image: Ken Burns
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-loop", "1", "-i", video_src,
                "-vf",
                f"scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920,"
                f"zoompan=z='min(1.18,1+0.0012*on)':d={frames}:s=1080x1920,"
                f"setsar=1,fps={settings.TARGET_FPS}",
                "-t", f"{qdur:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", str(settings.CRF_VISUAL),
                "-pix_fmt", "yuv420p",
                output
            ])
        else:
            # Video: motion (no audio from stock footage)
            fade = max(0.05, min(0.12, qdur/8.0))
            fade_out_st = max(0.0, qdur - fade)
            
            motion = self._get_motion_filter(qdur)
            
            base_filters = [
                "scale=1080:1920:force_original_aspect_ratio=increase",
                "crop=1080:1920",
            ]
            
            if motion:
                base_filters.append(motion)
            
            base_filters.extend([
                "setsar=1",
                "eq=brightness=0.02:contrast=1.08:saturation=1.1",
                f"fps={settings.TARGET_FPS}",
                f"setpts=N/{settings.TARGET_FPS}/TB",
                f"trim=start_frame=0:end_frame={frames}",
                f"fade=t=in:st=0:d={fade:.2f}",
                f"fade=t=out:st={fade_out_st:.2f}:d={fade:.2f}"
            ])
            
            vf = ",".join(base_filters)
            
            # ✅ DÜZELTME: -an flag'i kaldırıldı (audio orchestrator'da eklenecek)
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_src,
                "-vf", vf,
                "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                "-c:v", "libx264", "-preset", "fast", "-crf", str(settings.CRF_VISUAL),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                output
            ])
        
        return output
    
    def _get_motion_filter(self, duration: float) -> str:
        """Generate motion filter string."""
        if not settings.VIDEO_MOTION or duration < 2.0:
            return ""
        
        # ✅ DÜZELTME: MOTION_INTENSITY her türlü değeri kabul et
        intensity_value = settings.MOTION_INTENSITY
        
        # Intensity'yi kategorize et
        try:
            # Float veya string olabilir
            if isinstance(intensity_value, str):
                intensity = intensity_value.lower()
            elif isinstance(intensity_value, (int, float)):
                # Float değere göre kategori belirle
                if intensity_value <= 1.10:
                    intensity = "low"
                elif intensity_value <= 1.18:
                    intensity = "moderate"
                else:
                    intensity = "dynamic"
            else:
                intensity = "moderate"  # Default
        except Exception:
            intensity = "moderate"  # Safe fallback
        
        # Zoom ranges based on intensity
        zoom_range = (1.0, 1.12)
        speed = 0.001
        
        if intensity in ("moderate", "medium"):
            zoom_range = (1.0, 1.15)
            speed = 0.0015
        elif intensity in ("dynamic", "high", "strong"):
            zoom_range = (1.0, 1.20)
            speed = 0.002
        
        motion_types = ['zoom_in', 'zoom_out', 'pan_right', 'pan_left', 'static']
        weights = [0.35, 0.20, 0.20, 0.20, 0.05]
        m = random.choices(motion_types, weights=weights)[0]
        
        if m == 'zoom_in':
            return (
                f"zoompan=z='min(zoom+{speed}, {zoom_range[1]})'"
                ":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1920"
            )
        elif m == 'zoom_out':
            return (
                f"zoompan=z='max(zoom-{speed}, {zoom_range[0]})'"
                ":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1920"
            )
        elif m in ('pan_right', 'pan_left'):
            sign = "+" if m == "pan_right" else "-"
            return (
                f"zoompan=z='1.08':x='iw/2-(iw/zoom/2){sign}min(iw/zoom-iw*0.02,iw*0.004*on)'"
                ":y='ih/2-(ih/zoom/2)':d=1:s=1080x1920"
            )
        
        return ""
