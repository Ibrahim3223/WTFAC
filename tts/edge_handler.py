# -*- coding: utf-8 -*-
"""
Edge-TTS handler with word timing support.
"""
import re
import asyncio
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


class TTSHandler:
    """Handle text-to-speech generation with word timing."""
    
    def __init__(self):
        """Initialize TTS handler."""
        self.voice = settings.VOICE
        self.rate = settings.TTS_RATE
        self.lang = settings.LANG
        
        # Apply nest_asyncio for nested event loops
        nest_asyncio.apply()
    
    def synthesize(
        self, 
        text: str, 
        wav_out: str
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            wav_out: Output WAV file path
            
        Returns:
            Tuple of (duration_seconds, word_durations)
            word_durations is list of (word, duration_seconds)
        """
        text = (text or "").strip()
        if not text:
            self._generate_silence(wav_out, 1.0)
            return 1.0, []
        
        # Try Edge-TTS with word boundaries
        try:
            marks = self._edge_stream_tts(text, wav_out)
            duration = self._apply_atempo(wav_out)
            words = self._merge_marks_to_words(text, marks, duration)
            
            print(f"   TTS: {len(words)} words | {duration:.2f}s")
            return duration, words
            
        except Exception as e:
            print(f"   ⚠️ Edge-TTS failed: {e}")
        
        # Fallback: Edge-TTS without marks
        try:
            self._edge_simple(text, wav_out)
            duration = self._apply_atempo(wav_out)
            words = self._equal_split_words(text, duration)
            
            print(f"   TTS (no marks): {len(words)} words | {duration:.2f}s")
            return duration, words
            
        except Exception as e2:
            print(f"   ⚠️ Edge-TTS simple failed: {e2}")
        
        # Last resort: Google TTS
        try:
            self._google_tts(text, wav_out)
            duration = self._apply_atempo(wav_out)
            words = self._equal_split_words(text, duration)
            
            print(f"   TTS (Google fallback): {len(words)} words | {duration:.2f}s")
            return duration, words
            
        except Exception as e3:
            print(f"   ❌ All TTS methods failed: {e3}")
            self._generate_silence(wav_out, 4.0)
            return 4.0, []
    
    def _edge_stream_tts(self, text: str, wav_out: str) -> List[Dict[str, Any]]:
        """Edge-TTS with word boundaries."""
        mp3_path = wav_out.replace(".wav", ".mp3")
        marks: List[Dict[str, Any]] = []
        
        async def _run():
            audio = bytearray()
            comm = edge_tts.Communicate(text, voice=self.voice, rate=self.rate)
            
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
            
            with open(mp3_path, "wb") as f:
                f.write(bytes(audio))
        
        asyncio.run(_run())
        return marks
    
    def _edge_simple(self, text: str, wav_out: str):
        """Simple Edge-TTS without word boundaries."""
        mp3_path = wav_out.replace(".wav", ".mp3")
        
        async def _run():
            comm = edge_tts.Communicate(text, voice=self.voice, rate=self.rate)
            await comm.save(mp3_path)
        
        asyncio.run(_run())
    
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
    
    def _apply_atempo(self, wav_out: str) -> float:
        """Convert MP3 to WAV with tempo adjustment."""
        mp3_path = wav_out.replace(".wav", ".mp3")
        
        # Parse atempo from rate
        atempo = self._rate_to_atempo(self.rate)
        
        # Convert with atempo
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", mp3_path,
            "-ar", "48000", "-ac", "1", "-acodec", "pcm_s16le",
            "-af", f"dynaudnorm=g=7:f=250,atempo={atempo}",
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
        total: float
    ) -> List[Tuple[str, float]]:
        """Merge Edge-TTS word boundaries into word durations."""
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        
        if not words:
            return []
        
        out = []
        
        # If we have good marks coverage
        if marks and len(marks) >= len(words) * 0.7:
            N = min(len(words), len(marks))
            
            # Raw durations from marks
            raw_durs = [max(0.05, float(marks[i]["t1"] - marks[i]["t0"])) for i in range(N)]
            sum_raw = sum(raw_durs)
            
            # Scale to match actual duration
            scale = (total / sum_raw) if sum_raw > 0 else 1.0
            
            for i in range(N):
                scaled_dur = max(0.08, raw_durs[i] * scale)
                out.append((words[i], scaled_dur))
            
            # Handle remaining words
            if len(words) > N:
                used_time = sum(d for _, d in out)
                remain = max(0.0, total - used_time)
                each = remain / (len(words) - N) if (len(words) - N) > 0 else 0.1
                
                for i in range(N, len(words)):
                    out.append((words[i], max(0.08, each)))
            
            # Final adjustment to match total exactly
            current_total = sum(d for _, d in out)
            if abs(current_total - total) > 0.05:
                diff = total - current_total
                if out:
                    last_word, last_dur = out[-1]
                    out[-1] = (last_word, max(0.08, last_dur + diff))
        
        else:
            # Fallback: equal split
            out = self._equal_split_words(text, total)
        
        return out
    
    def _equal_split_words(self, text: str, duration: float) -> List[Tuple[str, float]]:
        """Split duration equally among words."""
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        
        if not words:
            return []
        
        each = max(0.08, duration / len(words))
        out = [(w, each) for w in words]
        
        # Adjust last word to match total exactly
        if out:
            current_sum = sum(d for _, d in out)
            diff = duration - current_sum
            if abs(diff) > 0.01:
                last_word, last_dur = out[-1]
                out[-1] = (last_word, max(0.08, last_dur + diff))
        
        return out
    
    def _generate_silence(self, wav_out: str, duration: float):
        """Generate silent audio."""
        run([
            "ffmpeg", "-y", "-f", "lavfi", 
            "-t", f"{duration:.3f}", 
            "-i", "anullsrc=r=48000:cl=mono", 
            wav_out
        ])
