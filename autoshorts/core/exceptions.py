"""
Custom exception hierarchy for autoshorts.

Provides domain-specific exceptions for better error handling.
"""


class AutoShortsError(Exception):
    """Base exception for all autoshorts errors."""
    pass


class ConfigurationError(AutoShortsError):
    """Configuration or settings error."""
    pass


class ContentGenerationError(AutoShortsError):
    """Error during content generation (Gemini API)."""
    pass


class QualityError(AutoShortsError):
    """Content quality below threshold."""
    pass


class NoveltyError(AutoShortsError):
    """Content not novel enough (duplicate/similar)."""
    pass


class TTSError(AutoShortsError):
    """Text-to-speech generation error."""
    pass


class VideoSourceError(AutoShortsError):
    """Error finding or downloading videos (Pexels/Pixabay)."""
    pass


class VideoProductionError(AutoShortsError):
    """Error during video production (FFmpeg, captions, etc)."""
    pass


class CaptionError(AutoShortsError):
    """Error during caption rendering or alignment."""
    pass


class UploadError(AutoShortsError):
    """Error during YouTube upload."""
    pass


class PipelineError(AutoShortsError):
    """Error in pipeline execution."""
    pass


class StateError(AutoShortsError):
    """Error in state management."""
    pass
