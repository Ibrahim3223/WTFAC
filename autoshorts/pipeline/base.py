"""
Base classes for pipeline pattern implementation.

Each stage of the video generation process is a separate, testable component.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

from ..core import Result


@dataclass
class PipelineContext:
    """
    Shared context passed through pipeline stages.

    Stores intermediate results and configuration for each stage.
    """

    # Channel info
    channel: str = ""
    topic: str = ""

    # Stage outputs
    content: Optional[Dict[str, Any]] = None
    audio_segments: Optional[list] = None
    video_path: Optional[str] = None
    video_id: Optional[str] = None

    # Temp directory (shared across stages)
    temp_dir: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """Set a value in metadata."""
        self.metadata[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from metadata."""
        return self.metadata.get(key, default)


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage:
    - Takes a PipelineContext
    - Performs its operation
    - Updates the context with results
    - Returns Result[PipelineContext, str]
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize stage.

        Args:
            name: Optional custom name for logging
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"pipeline.{self.name}")

    @abstractmethod
    def execute(self, context: PipelineContext) -> Result[PipelineContext, str]:
        """
        Execute this stage.

        Args:
            context: Pipeline context with input data

        Returns:
            Result containing updated context or error message
        """
        pass

    def __call__(self, context: PipelineContext) -> Result[PipelineContext, str]:
        """Allow stage to be called directly."""
        self.logger.info(f"Starting stage: {self.name}")
        result = self.execute(context)

        if result.is_ok():
            self.logger.info(f"✅ Stage completed: {self.name}")
        else:
            self.logger.error(f"❌ Stage failed: {self.name} - {result.unwrap_err()}")

        return result

    def skip_if(self, condition: bool, reason: str = "") -> bool:
        """
        Check if stage should be skipped.

        Args:
            condition: If True, stage will be skipped
            reason: Reason for skipping

        Returns:
            True if should skip
        """
        if condition:
            self.logger.info(f"⏭️ Skipping stage: {self.name} - {reason}")
            return True
        return False
