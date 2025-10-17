# -*- coding: utf-8 -*-
"""Video downloader with validation."""
import re
import os
import pathlib
import requests
from typing import List, Tuple, Dict, Optional

from autoshorts.utils.ffmpeg_utils import run


class VideoDownloader:
    """Download and validate video files."""
    
    def download(
        self, 
        pool: List[Tuple[int, str]], 
        temp_dir: str
    ) -> Dict[int, str]:
        """
        Download videos from pool.
        
        Args:
            pool: List of (video_id, url) tuples
            temp_dir: Temporary directory
            
        Returns:
            Dict of {video_id: local_path}
        """
        downloads = {}
        
        print("⬇️ Downloading videos...")
        
        for idx, (vid, link) in enumerate(pool):
            # Check if URL is a video
            ext = self._get_video_ext(link)
            if not ext:
                continue
            
            try:
                # Check Content-Type
                with requests.get(link, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    
                    ctype = (r.headers.get("Content-Type") or "").lower()
                    if ctype and not ctype.startswith("video/"):
                        continue
                    
                    # Download
                    out_path = os.path.join(temp_dir, f"pool_{idx:02d}_{vid}{ext}")
                    
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)
                
                # Validate
                if os.path.getsize(out_path) > 300_000 and self._has_multiple_frames(out_path):
                    downloads[vid] = out_path
                    
            except Exception as e:
                print(f"   ⚠️ Skip {vid}: {e}")
                continue
        
        print(f"   Downloaded: {len(downloads)} videos")
        return downloads
    
    def _get_video_ext(self, url: str) -> Optional[str]:
        """Extract video extension from URL."""
        m = re.search(r"\.(mp4|mov|m4v|webm)(?:$|\?)", url, re.I)
        return ("." + m.group(1).lower()) if m else None
    
    def _has_multiple_frames(self, path: str) -> bool:
        """Check if video has multiple frames."""
        try:
            out = run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0", path
            ], check=False).stdout.strip()
            
            if out and out.upper() != "N/A":
                return int(out) >= 2
            
            return True  # If can't determine, assume OK
            
        except Exception:
            return True
