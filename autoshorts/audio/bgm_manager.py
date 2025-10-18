# -*- coding: utf-8 -*-
"""Background music management."""
import os
import pathlib
import random
import logging
import requests

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run

logger = logging.getLogger(__name__)


class BGMManager:
    """Manage background music."""
    
    def get_bgm(self, duration: float, output_dir: str) -> str:
        """
        Get a BGM file path for the given duration.
        This is a simple method that orchestrator expects.
        
        Args:
            duration: Duration in seconds
            output_dir: Output directory for BGM file
            
        Returns:
            Path to BGM file or empty string if BGM disabled
        """
        if not settings.BGM_ENABLE:
            logger.info("      BGM disabled in settings")
            return ""
        
        try:
            # Find BGM source
            bgm_src = self._pick_source(output_dir)
            if not bgm_src:
                logger.warning("      ⚠️ No BGM source found")
                return ""
            
            # Loop BGM to duration
            bgm_output = os.path.join(output_dir, "bgm_final.wav")
            self._loop_bgm(bgm_src, duration, bgm_output)
            
            logger.info(f"      ✅ BGM prepared: {bgm_output}")
            return bgm_output
            
        except Exception as e:
            logger.error(f"      ❌ BGM error: {e}")
            return ""
    
    def add_bgm(self, voice_path: str, duration: float, temp_dir: str) -> str:
        """
        Add BGM to voice audio.
        
        Args:
            voice_path: Path to voice audio file
            duration: Duration in seconds
            temp_dir: Temporary directory
            
        Returns:
            Path to mixed audio file
        """
        # Find BGM source
        bgm_src = self._pick_source(temp_dir)
        if not bgm_src:
            return voice_path
        
        # Loop BGM to duration
        bgm_loop = os.path.join(temp_dir, "bgm_loop.wav")
        self._loop_bgm(bgm_src, duration, bgm_loop)
        
        # Mix with sidechain ducking
        output = os.path.join(temp_dir, "audio_with_bgm.wav")
        self._mix_with_duck(voice_path, bgm_loop, output)
        
        return output
    
    def _pick_source(self, temp_dir: str) -> str:
        """Pick BGM source from directory or URLs."""
        # Try local directory
        try:
            p = pathlib.Path(settings.BGM_DIR)
            if p.exists():
                files = list(p.glob("*.mp3")) + list(p.glob("*.wav"))
                if files:
                    chosen = str(random.choice(files))
                    logger.debug(f"      Selected local BGM: {chosen}")
                    return chosen
        except Exception as e:
            logger.debug(f"      Local BGM search failed: {e}")
        
        # Try URLs
        if settings.BGM_URLS:
            for url in settings.BGM_URLS:
                try:
                    ext = ".mp3" if ".mp3" in url.lower() else ".wav"
                    out_path = os.path.join(temp_dir, f"bgm_src{ext}")
                    
                    logger.debug(f"      Downloading BGM from: {url}")
                    with requests.get(url, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        with open(out_path, "wb") as f:
                            for chunk in r.iter_content(8192):
                                f.write(chunk)
                    
                    if os.path.getsize(out_path) > 100_000:
                        logger.debug(f"      Downloaded BGM: {out_path}")
                        return out_path
                except Exception as e:
                    logger.debug(f"      BGM download failed: {e}")
                    continue
        
        return ""
    
    def _loop_bgm(self, src: str, duration: float, output: str):
        """Loop BGM to match duration."""
        fade = settings.BGM_FADE
        endst = max(0.0, duration - fade)
        
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-stream_loop", "-1", "-i", src,
            "-t", f"{duration:.3f}",
            "-af",
            f"loudnorm=I=-21:TP=-2.0:LRA=11,"
            f"afade=t=in:st=0:d={fade:.2f},afade=t=out:st={endst:.2f}:d={fade:.2f},"
            "aresample=48000,pan=mono|c0=0.5*FL+0.5*FR",
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
            output
        ])
    
    def _mix_with_duck(self, voice: str, bgm: str, output: str):
        """Mix voice and BGM with sidechain ducking."""
        gain_db = settings.BGM_GAIN_DB
        thresh = settings.BGM_DUCK_THRESH
        ratio = settings.BGM_DUCK_RATIO
        attack = settings.BGM_DUCK_ATTACK_MS
        release = settings.BGM_DUCK_RELEASE_MS
        
        sc = (
            f"sidechaincompress=threshold={thresh}:ratio={ratio}:"
            f"attack={attack}:release={release}:makeup=1.0"
        )
        
        filter_complex = (
            f"[1:a]volume={gain_db}dB[b];"
            f"[b][0:a]{sc}[duck];"
            f"[0:a][duck]amix=inputs=2:duration=shortest,aresample=48000[mix]"
        )
        
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", voice, "-i", bgm,
            "-filter_complex", filter_complex,
            "-map", "[mix]",
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
            output
        ])
