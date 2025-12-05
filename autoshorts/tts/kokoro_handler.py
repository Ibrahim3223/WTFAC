# -*- coding: utf-8 -*-
"""
Kokoro TTS Handler - Ultra-realistic voice synthesis for Shorts

Features:
- 26 high-quality voice options
- Automatic model downloading and caching
- 24kHz sample rate for clarity
- Support for fp32, fp16, and int8 precision
- Lazy loading for faster startup
"""
import os
import logging
import tempfile
import requests
from pathlib import Path
from typing import Dict, Any, Tuple, List
import wave

logger = logging.getLogger(__name__)

# Available Kokoro voices (26 total)
KOKORO_VOICES = {
    # Female American (most natural for Shorts)
    "af_heart": "Heart - Warm female voice",
    "af_bella": "Bella - Energetic female voice",
    "af_sarah": "Sarah - Professional female voice (DEFAULT)",
    "af_sky": "Sky - Clear female voice",
    "af_alloy": "Alloy - Dynamic female voice",
    "af_aoede": "Aoede - Smooth female voice",
    "af_jessica": "Jessica - Friendly female voice",
    "af_kore": "Kore - Engaging female voice",
    "af_nicole": "Nicole - Confident female voice",
    "af_nova": "Nova - Modern female voice",
    "af_river": "River - Calm female voice",

    # Male American
    "am_adam": "Adam - Deep male voice",
    "am_michael": "Michael - Strong male voice",
    "am_echo": "Echo - Resonant male voice",
    "am_eric": "Eric - Clear male voice",
    "am_fenrir": "Fenrir - Bold male voice",
    "am_liam": "Liam - Smooth male voice",
    "am_onyx": "Onyx - Rich male voice",
    "am_puck": "Puck - Lively male voice",
    "am_santa": "Santa - Jolly male voice",

    # British Female
    "bf_alice": "Alice - British female voice",
    "bf_emma": "Emma - British female voice",
    "bf_isabella": "Isabella - British female voice",
    "bf_lily": "Lily - British female voice",

    # British Male
    "bm_daniel": "Daniel - British male voice",
    "bm_fable": "Fable - British storyteller voice",
    "bm_george": "George - British male voice",
    "bm_lewis": "Lewis - British male voice",
}

DEFAULT_VOICE = "af_sarah"  # Best for general Shorts content


class KokoroTTS:
    """
    Kokoro TTS using kokoro-onnx package.

    Ultra-realistic voice synthesis with 26 voice options.
    Models are automatically downloaded and cached.
    """

    MODEL_BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
    SAMPLE_RATE = 24000  # High quality audio

    PRECISION_FILES = {
        "fp32": "kokoro-v1.0.onnx",        # Full precision (best quality, slower)
        "fp16": "kokoro-v1.0.fp16.onnx",   # Half precision (good balance)
        "int8": "kokoro-v1.0.int8.onnx"    # Quantized (fastest, smaller)
    }

    def __init__(self, voice: str = DEFAULT_VOICE, precision: str = "fp32"):
        """
        Initialize Kokoro TTS.

        Args:
            voice: Voice ID (e.g., 'af_sarah', 'am_michael')
            precision: Model precision ('fp32', 'fp16', 'int8')
        """
        self.voice = voice if voice in KOKORO_VOICES else DEFAULT_VOICE
        self.precision = precision
        self.kokoro = None  # Lazy loaded
        self.cache_dir = Path.home() / ".cache" / "kokoro"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Kokoro] Initialized: voice={self.voice}, precision={self.precision}")

    def _download_file(self, url: str, dest: Path):
        """Download file with progress logging."""
        if dest.exists():
            logger.info(f"[Kokoro] Model cached: {dest.name}")
            return

        logger.info(f"[Kokoro] Downloading {dest.name}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

                # Log progress every MB
                if total_size > 0 and downloaded % (1024 * 1024) == 0:
                    progress = (downloaded / total_size) * 100
                    logger.info(f"[Kokoro]   {progress:.1f}% - {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB")

        logger.info(f"[Kokoro] Download complete: {dest.name}")

    def _ensure_models(self):
        """Download models if not cached."""
        model_file = self.PRECISION_FILES.get(self.precision, self.PRECISION_FILES["fp32"])

        model_path = self.cache_dir / model_file
        voices_path = self.cache_dir / "voices-v1.0.bin"

        # Download model if needed
        if not model_path.exists():
            model_url = f"{self.MODEL_BASE_URL}/{model_file}"
            self._download_file(model_url, model_path)

        # Download voices if needed
        if not voices_path.exists():
            voices_url = f"{self.MODEL_BASE_URL}/voices-v1.0.bin"
            self._download_file(voices_url, voices_path)

        return str(model_path), str(voices_path)

    def _load_model(self):
        """Lazy load Kokoro model."""
        if self.kokoro is not None:
            return

        try:
            from kokoro_onnx import Kokoro

            model_path, voices_path = self._ensure_models()

            logger.info(f"[Kokoro] Loading model: {Path(model_path).name}")
            self.kokoro = Kokoro(model_path, voices_path)
            logger.info(f"[Kokoro] Model loaded successfully")

        except ImportError as e:
            raise ImportError(
                "Kokoro TTS requires: pip install kokoro-onnx soundfile\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Kokoro model: {e}")

    def generate(self, text: str) -> Dict[str, Any]:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize

        Returns:
            Dict with 'audio' (bytes), 'duration' (float), 'word_timings' (list)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        self._load_model()
        text = text.strip()

        logger.info(f"[Kokoro] Generating: voice={self.voice}, {len(text)} chars")

        # Generate audio
        samples, sample_rate = self.kokoro.create(
            text,
            voice=self.voice,
            speed=1.0,
            lang="en-us"
        )

        # Convert to WAV bytes
        wav_bytes = self._array_to_wav(samples, sample_rate)
        duration = len(samples) / sample_rate

        logger.info(f"[Kokoro] Generated: {duration:.2f}s")

        # Extract word timings using forced alignment
        word_timings = self._extract_word_timings(text, wav_bytes, duration)

        return {
            'audio': wav_bytes,
            'duration': duration,
            'word_timings': word_timings
        }

    def synthesize(self, text: str, wav_out: str) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Synthesize text and save to file.

        Args:
            text: Text to synthesize
            wav_out: Output WAV file path

        Returns:
            (duration, word_timings)
        """
        result = self.generate(text)

        with open(wav_out, 'wb') as f:
            f.write(result['audio'])

        return result['duration'], result['word_timings']

    def _array_to_wav(self, samples, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes."""
        import numpy as np

        # Ensure samples are in int16 format
        if samples.dtype != np.int16:
            # Normalize and convert to int16
            samples = np.clip(samples, -1.0, 1.0)
            samples = (samples * 32767).astype(np.int16)

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Write WAV file
            with wave.open(tmp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.tobytes())

            # Read back as bytes
            with open(tmp_path, 'rb') as f:
                wav_bytes = f.read()

            return wav_bytes
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _extract_word_timings(
        self,
        text: str,
        wav_bytes: bytes,
        duration: float
    ) -> List[Tuple[str, float]]:
        """
        Extract word-level timings using forced alignment.

        Args:
            text: Spoken text
            wav_bytes: Audio WAV bytes
            duration: Audio duration

        Returns:
            List of (word, duration) tuples
        """
        try:
            # Import forced aligner
            from autoshorts.captions.forced_aligner import align_text_to_audio

            # Save audio to temp file for alignment
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name

            try:
                # Extract word timings using forced alignment
                word_timings = align_text_to_audio(
                    text=text,
                    audio_path=tmp_path,
                    tts_word_timings=None,  # Kokoro doesn't provide native timings
                    total_duration=duration,
                    language="en"
                )

                logger.debug(f"[Kokoro] Extracted {len(word_timings)} word timings via forced alignment")
                return word_timings

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.warning(f"[Kokoro] Word timing extraction failed: {e}")
            logger.warning(f"[Kokoro] Falling back to estimation")

            # Fallback: simple estimation
            words = text.split()
            if not words:
                return []

            # Character-based estimation
            total_chars = sum(len(w) for w in words)
            if total_chars == 0:
                return []

            word_timings = []
            for word in words:
                char_ratio = len(word) / total_chars
                word_duration = duration * char_ratio
                word_timings.append((word, word_duration))

            return word_timings

    @classmethod
    def list_voices(cls) -> List[str]:
        """Get list of available voice IDs."""
        return list(KOKORO_VOICES.keys())

    @classmethod
    def get_voice_info(cls, voice: str) -> Dict[str, str]:
        """
        Get voice information.

        Args:
            voice: Voice ID

        Returns:
            Dict with name, gender, accent info
        """
        voice_info = {
            "af_heart": {"name": "Heart", "gender": "female", "accent": "american", "style": "warm"},
            "af_bella": {"name": "Bella", "gender": "female", "accent": "american", "style": "energetic"},
            "af_sarah": {"name": "Sarah", "gender": "female", "accent": "american", "style": "professional"},
            "af_sky": {"name": "Sky", "gender": "female", "accent": "american", "style": "clear"},
            "am_adam": {"name": "Adam", "gender": "male", "accent": "american", "style": "deep"},
            "am_michael": {"name": "Michael", "gender": "male", "accent": "american", "style": "strong"},
            "bf_emma": {"name": "Emma", "gender": "female", "accent": "british", "style": "elegant"},
            "bf_isabella": {"name": "Isabella", "gender": "female", "accent": "british", "style": "sophisticated"},
            "bm_daniel": {"name": "Daniel", "gender": "male", "accent": "british", "style": "authoritative"},
        }
        return voice_info.get(voice, {"name": "Unknown", "gender": "unknown", "accent": "unknown", "style": "neutral"})

    @classmethod
    def get_recommended_voice(cls, style: str = "general") -> str:
        """
        Get recommended voice for content style.

        Args:
            style: Content style ('general', 'educational', 'energetic', 'professional')

        Returns:
            Voice ID
        """
        recommendations = {
            "general": "af_sarah",      # Professional, clear
            "educational": "af_sarah",  # Professional, authoritative
            "energetic": "af_bella",    # Energetic, engaging
            "professional": "af_sky",   # Clear, polished
            "storytelling": "bm_fable", # Narrative style
            "friendly": "af_river",     # Warm, approachable
        }
        return recommendations.get(style, DEFAULT_VOICE)


# Test function
def _test_kokoro():
    """Test Kokoro TTS functionality."""
    print("=" * 60)
    print("KOKORO TTS TEST")
    print("=" * 60)

    # List voices
    print(f"\nAvailable voices: {len(KokoroTTS.list_voices())}")
    for voice_id in KokoroTTS.list_voices()[:5]:
        info = KokoroTTS.get_voice_info(voice_id)
        print(f"  {voice_id}: {info.get('name', 'Unknown')} ({info.get('style', 'neutral')})")

    # Test synthesis
    print("\nTesting synthesis...")
    try:
        tts = KokoroTTS(voice="af_sarah", precision="fp32")
        test_text = "This is a test of Kokoro TTS. It sounds amazing!"

        result = tts.generate(test_text)
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Audio size: {len(result['audio'])} bytes")
        print("  [PASS] Kokoro TTS")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    _test_kokoro()
