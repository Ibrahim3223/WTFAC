"""
Orchestrator - Refactored as lightweight pipeline coordinator.

The new orchestrator is much simpler:
- Uses dependency injection
- Delegates work to pipeline stages
- ~100 lines vs old 570 lines
"""

import os
import tempfile
import shutil
import logging
from typing import Optional

from .core import Container, ServiceLifetime, Result
from .pipeline import Pipeline, RetryablePipeline, PipelineContext
from .pipeline.stages import (
    ContentGenerationStage,
    TTSStage,
    VideoProductionStage,
    UploadStage
)
from .content.gemini_client import GeminiClient
from .content.quality_scorer import QualityScorer
from .tts.edge_handler import TTSHandler
from .video.pexels_client import PexelsClient
from .video.downloader import VideoDownloader
from .video.segment_maker import SegmentMaker
from .captions.renderer import CaptionRenderer
from .audio.bgm_manager import BGMManager
from .upload.youtube_uploader import YouTubeUploader
from .state.novelty_guard import NoveltyGuard
from .state.state_guard import StateGuard
from .config import settings

logger = logging.getLogger(__name__)


class ShortsOrchestrator:
    """
    Main orchestrator for YouTube Shorts pipeline.

    Now uses pipeline pattern with dependency injection for clean,
    testable architecture.
    """

    def __init__(self, container: Optional[Container] = None):
        """
        Initialize orchestrator with dependency injection.

        Args:
            container: DI container with registered services
                       If None, creates default container
        """
        logger.info("=" * 60)
        logger.info("Initializing ShortsOrchestrator (v2.0 - Refactored)")
        logger.info("=" * 60)

        self.channel = settings.CHANNEL_NAME
        self.temp_dir = None

        logger.info(f"📺 Channel: {self.channel}")
        logger.info(f"🎯 Topic: {settings.CHANNEL_TOPIC}")
        logger.info(f"⏱️  Duration: {settings.TARGET_DURATION}s")

        # Setup container
        if container is None:
            container = self._create_default_container()

        self.container = container

        # Build pipeline
        self.pipeline = self._build_pipeline()

        logger.info("=" * 60)
        logger.info(f"🚀 Orchestrator ready: {self.channel}")
        logger.info("=" * 60)

    def _create_default_container(self) -> Container:
        """Create container with all default services."""
        container = Container()

        # Validate API keys
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found!")

        logger.info(f"✅ Gemini API key: {settings.GEMINI_API_KEY[:10]}...")

        if settings.PEXELS_API_KEY:
            logger.info(f"✅ Pexels API key: {settings.PEXELS_API_KEY[:10]}...")
        else:
            logger.warning("⚠️ PEXELS_API_KEY not found")

        # Register all services
        container.register(
            GeminiClient,
            lambda: GeminiClient(
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_MODEL,
                max_retries=3
            ),
            ServiceLifetime.SINGLETON
        )

        container.register(QualityScorer, QualityScorer, ServiceLifetime.SINGLETON)
        container.register(TTSHandler, TTSHandler, ServiceLifetime.SINGLETON)
        container.register(PexelsClient, PexelsClient, ServiceLifetime.SINGLETON)
        container.register(VideoDownloader, VideoDownloader, ServiceLifetime.SINGLETON)
        container.register(SegmentMaker, SegmentMaker, ServiceLifetime.SINGLETON)

        container.register(
            CaptionRenderer,
            lambda: CaptionRenderer(caption_offset=settings.CAPTION_OFFSET),
            ServiceLifetime.SINGLETON
        )

        container.register(BGMManager, BGMManager, ServiceLifetime.SINGLETON)
        container.register(YouTubeUploader, YouTubeUploader, ServiceLifetime.SINGLETON)

        container.register(
            NoveltyGuard,
            lambda: NoveltyGuard(
                state_dir=settings.STATE_DIR,
                window_days=settings.ENTITY_COOLDOWN_DAYS
            ),
            ServiceLifetime.SINGLETON
        )

        container.register(
            StateGuard,
            lambda: StateGuard(channel=self.channel),
            ServiceLifetime.SINGLETON
        )

        logger.info("✅ All services registered in container")

        return container

    def _build_pipeline(self) -> Pipeline:
        """Build the video generation pipeline."""
        stages = [
            ContentGenerationStage(
                gemini=self.container.resolve(GeminiClient),
                quality_scorer=self.container.resolve(QualityScorer),
                novelty_guard=self.container.resolve(NoveltyGuard)
            ),
            TTSStage(
                tts_handler=self.container.resolve(TTSHandler)
            ),
            VideoProductionStage(
                pexels=self.container.resolve(PexelsClient),
                downloader=self.container.resolve(VideoDownloader),
                segment_maker=self.container.resolve(SegmentMaker),
                caption_renderer=self.container.resolve(CaptionRenderer),
                bgm_manager=self.container.resolve(BGMManager)
            ),
            UploadStage(
                uploader=self.container.resolve(YouTubeUploader),
                state_guard=self.container.resolve(StateGuard),
                novelty_guard=self.container.resolve(NoveltyGuard)
            )
        ]

        # Use retryable pipeline for robustness
        return RetryablePipeline(
            stages=stages,
            name="ShortsGeneration",
            max_attempts=settings.MAX_GENERATION_ATTEMPTS
        )

    def run(self) -> Optional[str]:
        """
        Execute the full pipeline.

        Returns:
            Video ID if successful, None otherwise
        """
        self.temp_dir = tempfile.mkdtemp(prefix="shorts_")

        try:
            # Create pipeline context
            context = PipelineContext(
                channel=self.channel,
                topic=settings.CHANNEL_TOPIC,
                temp_dir=self.temp_dir
            )

            # Run pipeline
            result = self.pipeline.run(context)

            if result.is_ok():
                final_context = result.unwrap()

                if final_context.video_id:
                    logger.info(f"✅ Success! Video ID: {final_context.video_id}")
                    return final_context.video_id
                else:
                    logger.info(f"⏭️ Video created: {final_context.video_path}")
                    return None
            else:
                error = result.unwrap_err()
                logger.error(f"❌ Pipeline failed: {error}")
                return None

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("🧹 Cleaned temp files")
            except Exception as e:
                logger.warning(f"⚠️ Cleanup warning: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
