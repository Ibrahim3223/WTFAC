# -*- coding: utf-8 -*-
"""
Edge-TTS handler with word timing support - 401 ERROR FIX
Robust retry logic + fallback strategies
"""
import re
import asyncio
import logging
import time
from typing import List, Tuple, Dict, Any

try:
    import edge_tts
    import nest_asyncio
except ImportError:
    raise ImportError("edge-tts and nest_asyncio required: pip install edge-tts nest_asyncio")

try:
    import requests
except ImportError:
    raise ImportError("requests required: pip install requests")

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run, ffprobe_duration

logger = logging.getLogger(__name__)


class TTSHandler:
    """Handle text-to-speech generation with word timing."""
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    
    def __init__(self):
        """Initialize TTS handler."""
        self.voice = settings.VOICE
        self.rate = settings.TTS_RATE
        self.lang = settings.LANG
        
        # Apply nest_asyncio for nested event loops
        nest_asyncio.apply()
        
        logger.info(f"   🎤 TTS initialized: voice={self.voice}, rate={self.rate}")
    
    def synthesize(
        self, 
        text: str, 
        wav_out: str
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Synthesize speech from text with robust retry logic.
        
        Args:
            text: Text to synthesize
            wav_out: Output WAV file path
            
        Returns:
            Tuple of (duration_seconds, word_durations)
            word_durations is list of (word, duration_seconds) AFTER atempo
        """
        text = (text or "").strip()
        if not text:
            self._generate_silence(wav_out, 1.0)
            return 1.0, []
        
        # Get atempo factor
        atempo = self._rate_to_atempo(self.rate)
        
        # Try Edge-TTS with word boundaries (BEST - has word timing!)
        for attempt in range(self.MAX_RETRIES):
            try:
                marks = self._edge_stream_tts(text, wav_out)
                duration = self._apply_atempo(wav_out, atempo)
                
                # Merge marks to words with atempo scaling
                words = self._merge_marks_to_words(text, marks, duration, atempo)
                
                logger.info(f"   ✅ Edge-TTS: {len(words)} words | {duration:.2f}s (atempo={atempo:.2f})")
                return duration, words
                
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"   ⚠️ Edge-TTS attempt {attempt+1} failed: {e}, retrying in {self.RETRY_DELAY}s...")
                    time.sleep(self.RETRY_DELAY)
                    self.RETRY_DELAY *= 1.5  # Exponential backoff
                else:
                    logger.warning(f"   ⚠️ Edge-TTS with marks failed after {self.MAX_RETRIES} attempts: {e}")
        
        # Fallback 1: Edge-TTS without marks (still good quality, no word timing)
        for attempt in range(self.MAX_RETRIES):
            try:
                self._edge_simple(text, wav_out)
                duration = self._apply_atempo(wav_out, atempo)
                words = self._equal_split_words(text, duration)
                
                logger.info(f"   ✅ Edge-TTS (no marks): {len(words)} words | {duration:.2f}s")
                return duration, words
                
            except Exception as e2:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"   ⚠️ Edge-TTS simple attempt {attempt+1} failed: {e2}, retrying...")
                    time.sleep(self.RETRY_DELAY)
                else:
                    logger.warning(f"   ⚠️ Edge-TTS simple failed after {self.MAX_RETRIES} attempts: {e2}")
        
        # Fallback 2: Google TTS (last resort, no word timing)
        try:
            self._google_tts(text, wav_out)
            duration = self._apply_atempo(wav_out, atempo)
            words = self._equal_split_words(text, duration)
            
            logger.info(f"   ✅ Google TTS (fallback): {len(words)} words | {duration:.2f}s")
            return duration, words
            
        except Exception as e3:
            logger.error(f"   ❌ All TTS methods failed: {e3}")
            self._generate_silence(wav_out, 4.0)
            return 4.0, []
    
    def _edge_stream_tts(self, text: str, wav_out: str) -> List[Dict[str, Any]]:
        """Edge-TTS with word boundaries and better error handling."""
        mp3_path = wav_out.replace(".wav", ".mp3")
        marks: List[Dict[str, Any]] = []
        
        async def _run():
            audio = bytearray()
            
            # Create communicate object with explicit parameters
            comm = edge_tts.Communicate(
                text=text, 
                voice=self.voice, 
                rate=self.rate,
                # Add timeout to prevent hanging
                proxy=None
            )
            
            try:
                async for chunk in comm.stream():
                    chunk_type = chunk.get("type")
                    
                    if chunk_type == "audio":
                        audio.extend(chunk.get("data", b""))
                        
                    elif chunk_type == "WordBoundary":
                        offset = float(chunk.get("offset", 0)) / 10_000_000.0
                        duration = float(chunk.get("duration", 0)) / 10_000_000.0
                        marks.append({
                            "t0": offset,
                            "t1": offset + duration,
                            "text": str(chunk.get("text", ""))
                        })
                
                # Save audio
                if not audio:
                    raise RuntimeError("No audio data received from Edge-TTS")
                
                with open(mp3_path, "wb") as f:
                    f.write(bytes(audio))
                    
            except Exception as e:
                # Clean up on error
                import pathlib
                pathlib.Path(mp3_path).unlink(missing_ok=True)
                raise e
        
        # Run with timeout
        try:
            asyncio.run(asyncio.wait_for(_run(), timeout=30.0))
        except asyncio.TimeoutError:
            raise RuntimeError("Edge-TTS timeout after 30 seconds")
        
        return marks
    
    def _edge_simple(self, text: str, wav_out: str):
        """Simple Edge-TTS without word boundaries."""
        mp3_path = wav_out.replace(".wav", ".mp3")
        
        async def _run():
            comm = edge_tts.Communicate(text, voice=self.voice, rate=self.rate)
            await comm.save(mp3_path)
        
        try:
            asyncio.run(asyncio.wait_for(_run(), timeout=30.0))
        except asyncio.TimeoutError:
            raise RuntimeError("Edge-TTS simple timeout after 30 seconds")
    
    def _google_tts(self, text: str, wav_out: str):
        """Google TTS fallback."""
        mp3_path = wav_out.replace(".wav", ".mp3")
        
        q = requests.utils.quote(text.replace('"', '').replace("'", ""))
        lang_code = self.lang or "en"
        url = (
            f"https://translate.google.com/translate_tts?"
            f"ie=UTF-8&q={q}&tl={lang_code}&client=tw-ob&ttsspeed=1.0"
        )
        
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        
        with open(mp3_path, "wb") as f:
            f.write(r.content)
    
    def _apply_atempo(self, wav_out: str, atempo: float) -> float:
        """Convert MP3 to WAV with tempo adjustment."""
        mp3_path = wav_out.replace(".wav", ".mp3")
        
        # Convert with atempo
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", mp3_path,
            "-ar", "48000", "-ac", "1", "-acodec", "pcm_s16le",
            "-af", f"dynaudnorm=g=7:f=250,atempo={atempo:.3f}",
            wav_out
        ])
        
        # Cleanup MP3
        import pathlib
        pathlib.Path(mp3_path).unlink(missing_ok=True)
        
        return ffprobe_duration(wav_out)
    
    def _rate_to_atempo(self, rate_str: str, default: float = 1.10) -> float:
        """Parse rate string to atempo value."""
        try:
            if not rate_str:
                return default
            
            rate_str = rate_str.strip()
            
            # Percentage: "+12%" -> 1.12
            if rate_str.endswith("%"):
                val = float(rate_str.replace("%", ""))
                return max(0.5, min(2.0, 1.0 + val / 100.0))
            
            # Multiplier: "1.2x" -> 1.2
            if rate_str.endswith(("x", "X")):
                return max(0.5, min(2.0, float(rate_str[:-1])))
            
            # Direct float
            v = float(rate_str)
            return max(0.5, min(2.0, v))
            
        except Exception:
            return default
    
    def _merge_marks_to_words(
        self, 
        text: str, 
        marks: List[Dict[str, Any]], 
        total_duration: float,
        atempo: float = 1.0
    ) -> List[Tuple[str, float]]:
        """
        Merge Edge-TTS word boundaries into word durations.
        
        CRITICAL: Marks are in ORIGINAL time (before atempo).
        We must scale them to match actual audio duration!
        
        Args:
            text: Original text
            marks: Word boundary marks from Edge-TTS (ORIGINAL time)
            total_duration: Actual audio duration AFTER atempo
            atempo: Speed multiplier that was applied
            
        Returns:
            List of (word, duration) tuples in ACTUAL time
        """
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        
        if not words:
            return []
        
        out = []
        
        # If we have good marks coverage
        if marks and len(marks) >= len(words) * 0.7:
            N = min(len(words), len(marks))
            
            # Extract raw durations from marks (ORIGINAL time)
            raw_durs = []
            for i in range(N):
                t0 = float(marks[i]["t0"])
                t1 = float(marks[i]["t1"])
                dur = max(0.05, t1 - t0)
                raw_durs.append(dur)
            
            # Calculate total time in ORIGINAL marks
            sum_raw = sum(raw_durs)
            
            # Apply atempo scaling
            # If atempo=1.12, audio is 1.12x faster, so timings are 1/1.12 shorter
            scaled_durs = [dur / atempo for dur in raw_durs]
            sum_scaled = sum(scaled_durs)
            
            # Final adjustment to match exact duration
            if sum_scaled > 0:
                correction_factor = total_duration / sum_scaled
            else:
                correction_factor = 1.0
            
            for i in range(N):
                final_dur = max(0.05, scaled_durs[i] * correction_factor)
                out.append((words[i], final_dur))
            
            # Handle remaining words if any
            if len(words) > N:
                used_time = sum(d for _, d in out)
                remain = max(0.0, total_duration - used_time)
                each = remain / (len(words) - N) if (len(words) - N) > 0 else 0.1
                
                for i in range(N, len(words)):
                    out.append((words[i], max(0.05, each)))
            
            # Final precision adjustment
            current_total = sum(d for _, d in out)
            if abs(current_total - total_duration) > 0.01:
                diff = total_duration - current_total
                if out:
                    last_word, last_dur = out[-1]
                    out[-1] = (last_word, max(0.05, last_dur + diff))
            
            # Debug log
            final_total = sum(d for _, d in out)
            logger.debug(
                f"      Word timing: raw={sum_raw:.2f}s → "
                f"scaled={sum_scaled:.2f}s (atempo={atempo:.2f}) → "
                f"final={final_total:.2f}s (target={total_duration:.2f}s)"
            )
        
        else:
            # Fallback: equal split
            out = self._equal_split_words(text, total_duration)
        
        return out
    
    def _equal_split_words(self, text: str, duration: float) -> List[Tuple[str, float]]:
        """Split duration equally among words."""
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        
        if not words:
            return []
        
        each = max(0.05, duration / len(words))
        out = [(w, each) for w in words]
        
        # Adjust last word to match total exactly
        if out:
            current_sum = sum(d for _, d in out)
            diff = duration - current_sum
            if abs(diff) > 0.01:
                last_word, last_dur = out[-1]
                out[-1] = (last_word, max(0.05, last_dur + diff))
        
        return out
    
    def _generate_silence(self, wav_out: str, duration: float):
        """Generate silent audio."""
        run([
            "ffmpeg", "-y", "-f", "lavfi", 
            "-t", f"{duration:.3f}", 
            "-i", "anullsrc=r=48000:cl=mono", 
            wav_out
        ])
