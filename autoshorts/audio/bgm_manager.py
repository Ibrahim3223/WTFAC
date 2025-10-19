# -*- coding: utf-8 -*-
"""
Background music management - PROFESSIONAL AUDIO MIX
Advanced sidechain ducking, EQ, and frequency separation
"""
import os
import pathlib
import random
import logging
import requests

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run

logger = logging.getLogger(__name__)


# ============================================================================
# PROFESSIONAL AUDIO CONSTANTS
# ============================================================================

# Voice EQ settings (studio quality)
VOICE_EQ = {
    "high_pass": 80,        # Remove rumble below 80Hz
    "cut_200hz": -3,        # Reduce muddiness at 200Hz
    "boost_3khz": 4,        # Boost presence at 3kHz
    "high_shelf_8khz": 2,   # Add air/brightness at 8kHz
    "de_ess_threshold": -20 # De-esser threshold
}

# BGM EQ settings (make space for voice)
BGM_EQ = {
    "cut_500_2khz": -6,     # Cut voice frequency range
    "low_boost_60hz": 2,    # Boost bass for warmth
    "high_cut_12khz": -2    # Reduce high end brightness
}

# Sidechain ducking (professional podcast style)
DUCK_SETTINGS = {
    "threshold_db": -25,    # Duck when voice reaches -25dB
    "ratio": 4.0,           # 4:1 compression ratio
    "attack_ms": 10,        # Fast attack (10ms)
    "release_ms": 250,      # Medium release (250ms)
    "knee_db": 3.0          # Soft knee for smooth ducking
}


class BGMManager:
    """Manage background music with professional audio processing."""
    
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
            
            # Loop and process BGM to duration
            bgm_output = os.path.join(output_dir, "bgm_final.wav")
            self._loop_and_process_bgm(bgm_src, duration, bgm_output)
            
            logger.info(f"      ✅ BGM prepared: {bgm_output}")
            return bgm_output
            
        except Exception as e:
            logger.error(f"      ❌ BGM error: {e}")
            return ""
    
    def add_bgm(self, voice_path: str, duration: float, temp_dir: str) -> str:
        """
        Add BGM to voice audio with professional mixing.
        
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
        
        # Loop and process BGM
        bgm_processed = os.path.join(temp_dir, "bgm_processed.wav")
        self._loop_and_process_bgm(bgm_src, duration, bgm_processed)
        
        # Process voice with EQ
        voice_processed = os.path.join(temp_dir, "voice_processed.wav")
        self._process_voice(voice_path, voice_processed)
        
        # Mix with professional sidechain ducking
        output = os.path.join(temp_dir, "audio_with_bgm.wav")
        self._pro_mix(voice_processed, bgm_processed, output)
        
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
    
    def _loop_and_process_bgm(self, src: str, duration: float, output: str):
        """
        Loop BGM to match duration and apply professional EQ.
        
        BGM Processing:
        - Cut 500-2000Hz (make space for voice)
        - Boost 60Hz (add warmth)
        - Cut 12kHz+ (reduce brightness)
        - Normalize to -21 LUFS
        """
        fade = settings.BGM_FADE
        endst = max(0.0, duration - fade)
        
        # Build EQ filter for BGM
        eq_filter = (
            f"equalizer=f=500:width_type=o:width=2.5:g={BGM_EQ['cut_500_2khz']},"  # Cut voice range
            f"equalizer=f=60:width_type=o:width=1:g={BGM_EQ['low_boost_60hz']},"    # Boost bass
            f"equalizer=f=12000:width_type=o:width=1:g={BGM_EQ['high_cut_12khz']}"  # Cut highs
        )
        
        # Full audio filter chain
        audio_filter = (
            f"{eq_filter},"
            f"loudnorm=I=-21:TP=-2.0:LRA=11,"
            f"afade=t=in:st=0:d={fade:.2f},afade=t=out:st={endst:.2f}:d={fade:.2f},"
            f"aresample=48000,pan=mono|c0=0.5*FL+0.5*FR"
        )
        
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-stream_loop", "-1", "-i", src,
            "-t", f"{duration:.3f}",
            "-af", audio_filter,
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
            output
        ])
    
    def _process_voice(self, voice_in: str, voice_out: str):
        """
        Apply professional EQ to voice.
        
        Voice Processing:
        - High-pass at 80Hz (remove rumble)
        - Cut at 200Hz (reduce muddiness)
        - Boost at 3kHz (add presence/clarity)
        - High shelf at 8kHz (add air)
        - De-esser (reduce sibilance)
        - Compression (even dynamics)
        """
        
        # Build EQ filter for voice
        eq_filter = (
            f"highpass=f={VOICE_EQ['high_pass']},"                                          # Remove rumble
            f"equalizer=f=200:width_type=o:width=1:g={VOICE_EQ['cut_200hz']},"            # Cut muddiness
            f"equalizer=f=3000:width_type=o:width=1.5:g={VOICE_EQ['boost_3khz']},"        # Boost presence
            f"equalizer=f=8000:width_type=h:width=1000:g={VOICE_EQ['high_shelf_8khz']}"   # Add air
        )
        
        # De-esser (reduce harsh 's' sounds)
        deess_filter = f"deesser=i={VOICE_EQ['de_ess_threshold']}:m=0.5:f=0.5:s=o"
        
        # Compression (smooth dynamics)
        comp_filter = "acompressor=threshold=-18dB:ratio=3:attack=5:release=50:makeup=2"
        
        # Full filter chain
        audio_filter = f"{eq_filter},{deess_filter},{comp_filter}"
        
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", voice_in,
            "-af", audio_filter,
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
            voice_out
        ])
    
    def _pro_mix(self, voice: str, bgm: str, output: str):
        """
        Mix voice and BGM with professional sidechain ducking and frequency separation.
        
        Mixing Strategy:
        1. Voice: Center, full spectrum (already EQ'd)
        2. BGM: Stereo wide, frequency-separated, sidechained
        3. Sidechain: Duck BGM when voice is active
        """
        
        # BGM volume (in dB)
        gain_db = settings.BGM_GAIN_DB
        
        # Sidechain compression settings
        thresh = DUCK_SETTINGS["threshold_db"]
        ratio = DUCK_SETTINGS["ratio"]
        attack = DUCK_SETTINGS["attack_ms"]
        release = DUCK_SETTINGS["release_ms"]
        knee = DUCK_SETTINGS["knee_db"]
        
        # Build sidechain compressor
        sidechain_filter = (
            f"sidechaincompress=threshold={thresh}dB:ratio={ratio}:"
            f"attack={attack}:release={release}:knee={knee}:makeup=1.0"
        )
        
        # Filter complex:
        # [1:a] = BGM input
        # - Apply gain
        # - Apply sidechain compression (triggered by voice)
        # [0:a] = Voice input
        # - Keep at 100% volume
        # Mix both with amix
        filter_complex = (
            f"[1:a]volume={gain_db}dB[bgm];"
            f"[bgm][0:a]{sidechain_filter}[bgm_ducked];"
            f"[0:a][bgm_ducked]amix=inputs=2:duration=shortest:weights=1.0 0.7,aresample=48000[mix]"
        )
        
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", voice,
            "-i", bgm,
            "-filter_complex", filter_complex,
            "-map", "[mix]",
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
            output
        ])
    
    def _mix_with_duck(self, voice: str, bgm: str, output: str):
        """
        DEPRECATED: Use _pro_mix instead.
        Kept for backward compatibility.
        """
        logger.warning("_mix_with_duck is deprecated, using _pro_mix instead")
        self._pro_mix(voice, bgm, output)
