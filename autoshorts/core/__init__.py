"""Core utilities and patterns."""

from .result import Result, Success, Failure
from .retry import retry, retry_with_result, RetryContext
from .container import Container, ServiceLifetime, get_container, reset_container
from .exceptions import (
    AutoShortsError,
    ConfigurationError,
    ContentGenerationError,
    QualityError,
    NoveltyError,
    TTSError,
    VideoSourceError,
    VideoProductionError,
    CaptionError,
    UploadError,
    PipelineError,
    StateError,
)

__all__ = [
    # Result pattern
    "Result",
    "Success",
    "Failure",
    # Retry utilities
    "retry",
    "retry_with_result",
    "RetryContext",
    # Dependency injection
    "Container",
    "ServiceLifetime",
    "get_container",
    "reset_container",
    # Exceptions
    "AutoShortsError",
    "ConfigurationError",
    "ContentGenerationError",
    "QualityError",
    "NoveltyError",
    "TTSError",
    "VideoSourceError",
    "VideoProductionError",
    "CaptionError",
    "UploadError",
    "PipelineError",
    "StateError",
]
