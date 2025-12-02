"""
Pipeline orchestration - chains stages together.
"""

from typing import List, Optional
import logging

from .base import PipelineStage, PipelineContext
from ..core import Result


class Pipeline:
    """
    Pipeline that chains multiple stages together.

    Example:
        pipeline = Pipeline([
            ContentGenerationStage(),
            TTSStage(),
            VideoProductionStage(),
            UploadStage()
        ])

        context = PipelineContext(channel="TestChannel", topic="Facts")
        result = pipeline.run(context)

        if result.is_ok():
            final_context = result.unwrap()
            print(f"Video ID: {final_context.video_id}")
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        name: str = "Pipeline",
        stop_on_error: bool = True
    ):
        """
        Initialize pipeline.

        Args:
            stages: List of pipeline stages to execute in order
            name: Pipeline name for logging
            stop_on_error: If True, stop on first error; otherwise continue
        """
        self.stages = stages
        self.name = name
        self.stop_on_error = stop_on_error
        self.logger = logging.getLogger(f"pipeline.{name}")

    def run(self, context: PipelineContext) -> Result[PipelineContext, str]:
        """
        Run all stages in sequence.

        Args:
            context: Initial pipeline context

        Returns:
            Result containing final context or error message
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Starting pipeline: {self.name}")
        self.logger.info(f"Stages: {len(self.stages)}")
        self.logger.info("=" * 60)

        current_context = context

        for i, stage in enumerate(self.stages, 1):
            self.logger.info(f"Stage {i}/{len(self.stages)}: {stage.name}")

            result = stage(current_context)

            if result.is_err():
                error = result.unwrap_err()
                self.logger.error(f"Pipeline failed at stage {stage.name}: {error}")

                if self.stop_on_error:
                    return Result.err(f"Stage {stage.name} failed: {error}")
                else:
                    self.logger.warning(f"Continuing despite error in {stage.name}")
            else:
                current_context = result.unwrap()

        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline completed: {self.name}")
        self.logger.info("=" * 60)

        return Result.ok(current_context)

    def add_stage(self, stage: PipelineStage) -> 'Pipeline':
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        return self

    def insert_stage(self, index: int, stage: PipelineStage) -> 'Pipeline':
        """Insert a stage at a specific position."""
        self.stages.insert(index, stage)
        return self


class RetryablePipeline(Pipeline):
    """
    Pipeline with automatic retry logic.

    If any stage fails, the entire pipeline can be retried.
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        name: str = "RetryablePipeline",
        max_attempts: int = 3,
        stop_on_error: bool = True
    ):
        super().__init__(stages, name, stop_on_error)
        self.max_attempts = max_attempts

    def run(self, context: PipelineContext) -> Result[PipelineContext, str]:
        """
        Run pipeline with retry logic.

        Args:
            context: Initial pipeline context

        Returns:
            Result containing final context or error message
        """
        for attempt in range(1, self.max_attempts + 1):
            self.logger.info(f"Pipeline attempt {attempt}/{self.max_attempts}")

            result = super().run(context)

            if result.is_ok():
                return result

            if attempt < self.max_attempts:
                self.logger.warning(
                    f"Attempt {attempt} failed. Retrying... "
                    f"({self.max_attempts - attempt} attempts left)"
                )
            else:
                self.logger.error(
                    f"All {self.max_attempts} attempts failed"
                )

        return Result.err(f"Pipeline failed after {self.max_attempts} attempts")
