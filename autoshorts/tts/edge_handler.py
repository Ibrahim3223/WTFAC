# -*- coding: utf-8 -*-
"""
Edge-TTS handler - BULLETPROOF with Fast Fallback
Quick fail (2 retries) → Immediate Google TTS fallback
Prioritizes reliability over retrying Edge-TTS
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
    """Handle text-to-speech generation with bulletproof Edge-TTS."""
    
    # Reduced retry for faster fallback
    MAX_RETRIES = 2  # Quick fail to fallback (was 5)
    INITIAL_RETRY_DELAY = 0.3
    MAX_RETRY_DELAY = 2.0
    
    # Rate limiting
    REQUEST_DELAY = 0.2  # Reduced for speed
    LAST_REQUEST_TIME = 0
    
    def __init__(self):
        """Initialize TTS handler with bulletproof settings."""
        self.voice = settings.VOICE
        self.rate = settings.TTS_RATE
        self.lang = settings.LANG
        
        # Apply nest_asyncio for nested event loops
        nest_asyncio.apply()
        
        logger.info(f"   🎤 TTS initialized: voice={self.voice}, rate={self.rate}")
        logger.info(f"   🛡️ BULLETPROOF mode: Fast fallback enabled (Edge → Google)")
    
    def _rate_limit_wait(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - TTSHandler.LAST_REQUEST_TIME
        
        if time_since_last < self.REQUEST_DELAY:
            wait_time = self.REQUEST_DELAY - time_since_last
            logger.debug(f"   ⏳ Rate limit wait: {wait_time:.2f}s")
            time.sleep(wait_time)
        
        TTSHandler.LAST_REQUEST_TIME = time.time()
    
    def synthesize(
        self, 
        text: str, 
        wav_out: str
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Synthesize speech with BULLETPROOF Edge-TTS.
        
        Strategy:
        1. Edge-TTS with word boundaries (BEST - retry 5x with UA rotation)
        2. Edge-TTS without marks (GOOD - retry 3x)
        3. Google TTS (FALLBACK - always works)
        
        Returns:
            Tuple of (duration_seconds, word_durations)
        """
        text = (text or "").strip()
        if not text:
            self._generate_silence(wav_out, 1.0)
            return 1.0, []
        
        # Get atempo factor
        atempo = self._rate_to_atempo(self.rate)
        
        # Layer 1: Edge-TTS with word boundaries (try quickly)
        retry_delay = self.INITIAL_RETRY_DELAY
        
        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit_wait()
                logger.debug(f"   🔄 Edge-TTS attempt {attempt+1}/{self.MAX_RETRIES}")
                
                marks = self._edge_stream_tts(text, wav_out)
                duration = self._apply_atempo(wav_out, atempo)
                words = self._merge_marks_to_words(text, marks, duration, atempo)
                
                logger.info(f"   ✅ Edge-TTS success: {len(words)} words | {duration:.2f}s (atempo={atempo:.2f})")
                return duration, words
                
            except Exception as e:
                try:
                    error_msg = str(e)
                    is_401 = "401" in error_msg
                    error_preview = error_msg[:60] if len(error_msg) > 60 else error_msg
                except:
                    error_preview = "Connection error"
                    is_401 = False
                
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"   ⚠️ Edge-TTS attempt {attempt+1} failed: {error_preview}{' (401)' if is_401 else ''}")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, self.MAX_RETRY_DELAY)
                else:
                    logger.warning(f"   ⚠️ Edge-TTS with marks failed, trying simple mode...")
        
        # Layer 2: Edge-TTS simple (no word boundaries - faster)
        retry_delay = self.INITIAL_RETRY_DELAY
        
        for attempt in range(2):  # Just 2 attempts
            try:
                self._rate_limit_wait()
                logger.debug(f"   🔄 Edge-TTS simple attempt {attempt+1}/2")
                
                self._edge_simple(text, wav_out)
                duration = self._apply_atempo(wav_out, atempo)
                
                # CRITICAL: No word boundaries from simple mode
                # Return empty list so forced aligner uses stable-ts
                logger.info(f"   ✅ Edge-TTS simple: audio generated ({duration:.2f}s)")
                logger.warning("   ℹ️ NOTE: No word timing - stable-ts will align")
                
                return duration, []  # EMPTY = stable-ts forced alignment!
                
            except Exception as e2:
                try:
                    error_preview = str(e2)[:60] if len(str(e2)) > 60 else str(e2)
                except:
                    error_preview = "Connection error"
                
                if attempt < 1:
                    logger.warning(f"   ⚠️ Edge-TTS simple attempt {attempt+1} failed: {error_preview}")
                    time.sleep(retry_delay)
                else:
                    logger.warning(f"   ⚠️ Edge-TTS simple failed, falling back to Google TTS...")
        
        # Layer 3: Google TTS (GUARANTEED to work)
        try:
            logger.info("   🔄 Using Google TTS fallback...")
            self._google_tts(text, wav_out)
            duration = self._apply_atempo(wav_out, atempo)
            
            # CRITICAL: Return EMPTY word list so forced aligner uses stable-ts!
            # Google TTS has no word timing, so we let stable-ts do FORCED ALIGNMENT
            logger.info(f"   ✅ Google TTS: audio generated ({duration:.2f}s)")
            logger.warning("   ℹ️ NOTE: Word timing will be generated by stable-ts FORCED ALIGNMENT")
            
            return duration, []  # EMPTY list = stable-ts will align!
            
        except Exception as e3:
            try:
                error_msg = str(e3)[:100]
            except:
                error_msg = "Unknown error"
            
            logger.error(f"   ❌ Google TTS failed: {error_msg}")
            logger.error(f"   ❌ ALL TTS methods failed - generating silence")
            self._generate_silence(wav_out, 4.0)
            return 4.0, []
    
    def _edge_stream_tts(self, text: str, wav_out: str) -> List[Dict[str, Any]]:
        """Edge-TTS with word boundaries and robust error handling."""
        mp3_path = wav_out.replace(".wav", ".mp3")
        marks: List[Dict[str, Any]] = []
        
        async def _run():
            audio = bytearray()
            
            # Create communicate object
            comm = edge_tts.Communicate(
                text=text, 
                voice=self.voice, 
                rate=self.rate
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
        """Google TTS fallback - simple and reliable."""
        import pathlib
        
        mp3_path = wav_out.replace(".wav", ".mp3")
        
        # Clean text
        text = text.strip()
        if not text:
            raise ValueError("Empty text")
        
        # Trim if too long (Google limit: ~200 chars)
        if len(text) > 200:
            logger.warning(f"   ⚠️ Text too long ({len(text)} chars), trimming to 200")
            text = text[:197] + "..."
        
        # URL encode
        q = requests.utils.quote(text.replace('"', '').replace("'", ""))
        lang_code = self.lang or "en"
        
        url = (
            f"https://translate.google.com/translate_tts?"
            f"ie=UTF-8&q={q}&tl={lang_code}&client=tw-ob&ttsspeed=1.0"
        )
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # Make request with retry
        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, timeout=15)
                r.raise_for_status()
                
                if len(r.content) < 100:
                    raise ValueError("Response too short")
                
                with open(mp3_path, "wb") as f:
                    f.write(r.content)
                
                # Verify file exists
                if not pathlib.Path(mp3_path).exists():
                    raise FileNotFoundError("MP3 not created")
                
                return
                
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"   ⚠️ Google TTS attempt {attempt+1} failed: {str(e)[:50]}")
                    time.sleep(0.5)
                else:
                    raise
    
    def _apply_atempo(self, wav_out: str, atempo: float) -> float:
        """Convert MP3 to WAV with tempo adjustment."""
        import pathlib
        
        mp3_path = wav_out.replace(".wav", ".mp3")
        
        # Verify MP3 exists
        if not pathlib.Path(mp3_path).exists():
            raise FileNotFoundError(f"MP3 not found: {mp3_path}")
        
        # Convert with atempo
        try:
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", mp3_path,
                "-ar", "48000", "-ac", "1", "-acodec", "pcm_s16le",
                "-af", f"dynaudnorm=g=7:f=250,atempo={atempo:.3f}",
                wav_out
            ])
        except Exception as e:
            logger.error(f"   ❌ FFmpeg conversion failed: {e}")
            raise
        finally:
            # Cleanup MP3
            pathlib.Path(mp3_path).unlink(missing_ok=True)
        
        # Verify WAV exists
        if not pathlib.Path(wav_out).exists():
            raise FileNotFoundError(f"WAV not created: {wav_out}")
        
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
        Scale them to match actual audio duration!
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
