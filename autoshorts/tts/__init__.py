# -*- coding: utf-8 -*-
"""
Text-to-speech module with multi-provider support.

Providers:
- Edge TTS (fast, reliable, word timings)
- Kokoro TTS (ultra-realistic, 26 voices)

The unified handler provides automatic fallback and provider selection.
"""

# Import Edge TTS handler (original)
from .edge_handler import TTSHandler as EdgeTTSHandler

# Import new handlers
from .kokoro_handler import KokoroTTS
from .unified_handler import UnifiedTTSHandler, TTSHandler

# Backward compatibility: TTSHandler now uses UnifiedTTSHandler
# But EdgeTTSHandler is still available if needed

__all__ = [
    'TTSHandler',           # Main handler (now unified)
    'UnifiedTTSHandler',    # Explicit unified handler
    'KokoroTTS',            # Kokoro TTS only
    'EdgeTTSHandler',       # Edge TTS only (original)
]
