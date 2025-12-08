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
from .tts import TTSHandler  # Now uses UnifiedTTSHandler with Kokoro support
from .video.pexels_client import PexelsClient
from .video.downloader import VideoDownloader
from .video.segment_maker import SegmentMaker
from .captions.renderer import CaptionRenderer
from .audio.bgm_manager import BGMManager
from .upload.youtube_uploader import YouTubeUploader
from .state.novelty_guard import NoveltyGuard
from .state.state_guard import StateGuard
from .config import settings

# TIER 1 VIRAL SYSTEM
from .content.hook_generator import HookGenerator
from .content.emotion_analyzer import EmotionAnalyzer
from .content.viral_patterns import ViralPatternAnalyzer
from .content.retention_patterns import CliffhangerInjector
from .video.color_grader import ColorGrader
from .video.mood_analyzer import MoodAnalyzer
from .captions.caption_animator import CaptionAnimator
from .audio.sfx_manager import SFXManager
from .audio.timing_optimizer import TimingOptimizer
from .thumbnail import ThumbnailGenerator

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

        # Determine which LLM provider to use
        llm_provider = settings.get_active_llm_provider()

        # Validate API keys based on provider
        if llm_provider == "groq":
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found!")
            logger.info(f"✅ Groq API key: {settings.GROQ_API_KEY[:10]}...")
            logger.info(f"🚀 Using Groq LLM (14.4K req/day free tier!)")
        else:
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found!")
            logger.info(f"✅ Gemini API key: {settings.GEMINI_API_KEY[:10]}...")

        if settings.PEXELS_API_KEY:
            logger.info(f"✅ Pexels API key: {settings.PEXELS_API_KEY[:10]}...")
        else:
            logger.warning("⚠️ PEXELS_API_KEY not found")

        # Register LLM client (supports both Gemini and Groq)
        container.register(
            GeminiClient,
            lambda: GeminiClient(
                api_key=settings.GEMINI_API_KEY or "",
                model=settings.GROQ_MODEL if llm_provider == "groq" else settings.GEMINI_MODEL,
                max_retries=3,
                provider=llm_provider,
                groq_api_key=settings.GROQ_API_KEY if llm_provider == "groq" else None
            ),
            ServiceLifetime.SINGLETON
        )

        container.register(QualityScorer, QualityScorer, ServiceLifetime.SINGLETON)

        # Register UnifiedTTSHandler with Kokoro support
        container.register(
            TTSHandler,
            lambda: TTSHandler(
                provider=settings.TTS_PROVIDER,
                kokoro_voice=settings.KOKORO_VOICE
            ),
            ServiceLifetime.SINGLETON
        )

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

        # ============================================
        # TIER 1 VIRAL SYSTEM - Register AI components
        # ============================================
        logger.info("🎯 Registering TIER 1 Viral System components...")

        # Content optimization
        container.register(
            HookGenerator,
            lambda: HookGenerator(gemini_api_key=settings.GEMINI_API_KEY),
            ServiceLifetime.SINGLETON
        )

        container.register(
            EmotionAnalyzer,
            lambda: EmotionAnalyzer(),
            ServiceLifetime.SINGLETON
        )

        container.register(
            ViralPatternAnalyzer,
            lambda: ViralPatternAnalyzer(storage_dir=".viral_patterns"),
            ServiceLifetime.SINGLETON
        )

        container.register(
            CliffhangerInjector,
            lambda: CliffhangerInjector(
                interval_seconds=10.0,
                max_cliffhangers=3,
                avoid_repetition=True
            ),
            ServiceLifetime.SINGLETON
        )

        # Video enhancement
        container.register(
            ColorGrader,
            lambda: ColorGrader(),
            ServiceLifetime.SINGLETON
        )

        container.register(
            MoodAnalyzer,
            lambda: MoodAnalyzer(gemini_api_key=settings.GEMINI_API_KEY),
            ServiceLifetime.SINGLETON
        )

        # Captions & Audio
        container.register(
            CaptionAnimator,
            lambda: CaptionAnimator(),
            ServiceLifetime.SINGLETON
        )

        container.register(
            SFXManager,
            lambda: SFXManager(),
            ServiceLifetime.SINGLETON
        )

        container.register(
            TimingOptimizer,
            lambda: TimingOptimizer(gemini_api_key=settings.GEMINI_API_KEY),
            ServiceLifetime.SINGLETON
        )

        # Thumbnail generation
        container.register(
            ThumbnailGenerator,
            lambda: ThumbnailGenerator(gemini_api_key=settings.GEMINI_API_KEY),
            ServiceLifetime.SINGLETON
        )

        logger.info("✅ TIER 1 Viral System components registered")
        logger.info("✅ All services registered in container")

        return container

    def _build_pipeline(self) -> Pipeline:
        """Build the video generation pipeline with TIER 1 Viral System."""
        stages = [
            ContentGenerationStage(
                gemini=self.container.resolve(GeminiClient),
                quality_scorer=self.container.resolve(QualityScorer),
                novelty_guard=self.container.resolve(NoveltyGuard),
                # TIER 1 components
                hook_generator=self.container.resolve(HookGenerator),
                emotion_analyzer=self.container.resolve(EmotionAnalyzer),
                viral_pattern_analyzer=self.container.resolve(ViralPatternAnalyzer),
                cliffhanger_injector=self.container.resolve(CliffhangerInjector)
            ),
            TTSStage(
                tts_handler=self.container.resolve(TTSHandler)
            ),
            VideoProductionStage(
                pexels=self.container.resolve(PexelsClient),
                downloader=self.container.resolve(VideoDownloader),
                segment_maker=self.container.resolve(SegmentMaker),
                caption_renderer=self.container.resolve(CaptionRenderer),
                bgm_manager=self.container.resolve(BGMManager),
                # TIER 1 components
                color_grader=self.container.resolve(ColorGrader),
                mood_analyzer=self.container.resolve(MoodAnalyzer),
                caption_animator=self.container.resolve(CaptionAnimator),
                sfx_manager=self.container.resolve(SFXManager),
                timing_optimizer=self.container.resolve(TimingOptimizer)
            ),
            UploadStage(
                uploader=self.container.resolve(YouTubeUploader),
                state_guard=self.container.resolve(StateGuard),
                novelty_guard=self.container.resolve(NoveltyGuard),
                thumbnail_generator=self.container.resolve(ThumbnailGenerator)
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
