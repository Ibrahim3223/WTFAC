# -*- coding: utf-8 -*-
"""
Unified TTS Handler - Multi-Provider Support with Smart Fallback

Providers:
1. Kokoro TTS - Ultra-realistic, 26 voices (best quality)
2. Edge TTS - Fast, reliable, word timings (good balance)

Features:
- Automatic provider fallback
- Configuration-based selection
- Consistent voice throughout video (no mid-video changes)
- Lazy loading for performance
"""
import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)


class UnifiedTTSHandler:
    """
    Unified TTS handler with multi-provider support and smart fallback.

    Provider Priority:
    - If TTS_PROVIDER='kokoro': Use Kokoro only (no fallback)
    - If TTS_PROVIDER='edge': Use Edge only (no fallback)
    - If TTS_PROVIDER='auto': Try Kokoro first, fallback to Edge

    This prevents voice changes mid-video which can be jarring.
    """

    def __init__(self, provider: str = "auto", kokoro_voice: str = "af_sarah", kokoro_precision: str = "fp32"):
        """
        Initialize unified TTS handler.

        Args:
            provider: TTS provider ('kokoro', 'edge', 'auto')
            kokoro_voice: Voice ID for Kokoro (e.g., 'af_sarah')
            kokoro_precision: Model precision for Kokoro ('fp32', 'fp16', 'int8')
        """
        self.provider = provider.lower()
        self.kokoro_voice = kokoro_voice
        self.kokoro_precision = kokoro_precision

        # Lazy-loaded providers
        self._kokoro = None
        self._edge = None
        self._last_word_timings = []

        logger.info(f"[UnifiedTTS] Initialized: provider={self.provider}, voice={kokoro_voice}")

    @property
    def kokoro(self):
        """Lazy load Kokoro TTS."""
        if self._kokoro is None:
            try:
                from autoshorts.tts.kokoro_handler import KokoroTTS
                self._kokoro = KokoroTTS(
                    voice=self.kokoro_voice,
                    precision=self.kokoro_precision
                )
                logger.info("[UnifiedTTS] Kokoro TTS loaded")
            except Exception as e:
                logger.warning(f"[UnifiedTTS] Failed to load Kokoro: {e}")
                self._kokoro = False  # Mark as failed
        return self._kokoro if self._kokoro is not False else None

    @property
    def edge(self):
        """Lazy load Edge TTS."""
        if self._edge is None:
            try:
                from autoshorts.tts.handler import TTSHandler
                self._edge = TTSHandler()
                logger.info("[UnifiedTTS] Edge TTS loaded")
            except Exception as e:
                logger.warning(f"[UnifiedTTS] Failed to load Edge TTS: {e}")
                self._edge = False
        return self._edge if self._edge is not False else None

    def generate(self, text: str) -> Dict[str, Any]:
        """
        Generate TTS audio with automatic provider selection.

        Args:
            text: Text to synthesize

        Returns:
            Dict with 'audio' (bytes), 'duration' (float), 'word_timings' (list)

        Raises:
            RuntimeError: If all providers fail
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        text = text.strip()

        # Get provider order based on configuration
        providers = self._get_provider_order()

        # Try each provider in order
        for provider_name in providers:
            try:
                result = self._generate_with_provider(provider_name, text)
                if result:
                    logger.info(f"[UnifiedTTS] Generated with {provider_name}: {result['duration']:.2f}s")
                    self._last_word_timings = result.get('word_timings', [])
                    return result
            except Exception as e:
                logger.warning(f"[UnifiedTTS] {provider_name} failed: {str(e)[:100]}")
                continue

        # All providers failed
        raise RuntimeError("All TTS providers failed")

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

        # Save to file
        with open(wav_out, 'wb') as f:
            f.write(result['audio'])

        return result['duration'], result.get('word_timings', [])

    def get_word_timings(self) -> List[Tuple[str, float]]:
        """Get word timings from last synthesis."""
        return self._last_word_timings

    def _get_provider_order(self) -> List[str]:
        """
        Get provider priority order based on configuration.

        Returns:
            List of provider names in priority order

        Note: When a specific provider is selected (not 'auto'),
              NO fallback is used to prevent voice changes mid-video.
        """
        if self.provider == 'kokoro':
            return ['kokoro']  # No fallback - consistent voice
        elif self.provider == 'edge':
            return ['edge']    # No fallback - consistent voice
        else:  # 'auto'
            # Smart fallback: Kokoro first (best quality), Edge second
            return ['kokoro', 'edge']

    def _generate_with_provider(self, provider: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Generate with specific provider.

        Args:
            provider: Provider name ('kokoro' or 'edge')
            text: Text to synthesize

        Returns:
            Result dict or None if provider unavailable
        """
        if provider == 'kokoro':
            return self._generate_kokoro(text)
        elif provider == 'edge':
            return self._generate_edge(text)
        return None

    def _generate_kokoro(self, text: str) -> Optional[Dict[str, Any]]:
        """Generate with Kokoro TTS."""
        if self.kokoro is None:
            return None

        try:
            result = self.kokoro.generate(text)
            return result
        except Exception as e:
            logger.error(f"[UnifiedTTS] Kokoro generation failed: {e}")
            raise

    def _generate_edge(self, text: str) -> Optional[Dict[str, Any]]:
        """Generate with Edge TTS."""
        if self.edge is None:
            return None

        try:
            # Edge TTS returns (duration, word_timings)
            # We need to adapt it to our format
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                duration, word_timings = self.edge.synthesize(text, tmp_path)

                with open(tmp_path, 'rb') as f:
                    audio_bytes = f.read()

                return {
                    'audio': audio_bytes,
                    'duration': duration,
                    'word_timings': word_timings
                }
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"[UnifiedTTS] Edge generation failed: {e}")
            raise

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about active provider.

        Returns:
            Dict with provider info
        """
        return {
            "configured_provider": self.provider,
            "kokoro_available": self.kokoro is not None,
            "edge_available": self.edge is not None,
            "kokoro_voice": self.kokoro_voice,
            "kokoro_precision": self.kokoro_precision,
        }


# Backward compatibility wrapper
class TTSHandler(UnifiedTTSHandler):
    """
    Backward compatible TTS handler.

    Maintains the same interface as the original TTSHandler
    but with enhanced multi-provider support.
    """

    def __init__(self, provider: str = "auto"):
        """Initialize with config-based settings."""
        # Try to get settings from config
        try:
            from autoshorts.config import settings
            kokoro_voice = getattr(settings, 'KOKORO_VOICE', 'af_sarah')
            kokoro_precision = getattr(settings, 'KOKORO_PRECISION', 'fp32')
            provider = getattr(settings, 'TTS_PROVIDER', 'auto')
        except:
            kokoro_voice = 'af_sarah'
            kokoro_precision = 'fp32'
            provider = 'auto'

        super().__init__(
            provider=provider,
            kokoro_voice=kokoro_voice,
            kokoro_precision=kokoro_precision
        )


# Test function
def _test_unified_tts():
    """Test unified TTS functionality."""
    print("=" * 60)
    print("UNIFIED TTS TEST")
    print("=" * 60)

    test_text = "This is a test of the unified TTS system. It sounds amazing!"

    # Test with auto provider
    print("\n[1] Testing with auto provider:")
    try:
        tts = UnifiedTTSHandler(provider="auto")
        info = tts.get_provider_info()
        print(f"   Config: {info['configured_provider']}")
        print(f"   Kokoro available: {info['kokoro_available']}")
        print(f"   Edge available: {info['edge_available']}")

        result = tts.generate(test_text)
        print(f"   Generated: {result['duration']:.2f}s")
        print("   [PASS] Auto provider")
    except Exception as e:
        print(f"   [FAIL] {e}")

    # Test with Kokoro only
    print("\n[2] Testing with Kokoro provider:")
    try:
        tts = UnifiedTTSHandler(provider="kokoro", kokoro_voice="af_sarah")
        result = tts.generate(test_text)
        print(f"   Generated: {result['duration']:.2f}s")
        print("   [PASS] Kokoro provider")
    except Exception as e:
        print(f"   [FAIL] {e}")

    # Test with Edge only
    print("\n[3] Testing with Edge provider:")
    try:
        tts = UnifiedTTSHandler(provider="edge")
        result = tts.generate(test_text)
        print(f"   Generated: {result['duration']:.2f}s")
        print(f"   Word timings: {len(result.get('word_timings', []))} words")
        print("   [PASS] Edge provider")
    except Exception as e:
        print(f"   [FAIL] {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    _test_unified_tts()
