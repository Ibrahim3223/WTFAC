"""
Content Generation Stage - Generates viral content using Gemini AI.
"""

from typing import Dict, Any

from ..base import PipelineStage, PipelineContext
from ...core import Result, ContentGenerationError, QualityError, NoveltyError
from ...content.gemini_client import GeminiClient
from ...content.quality_scorer import QualityScorer
from ...state.novelty_guard import NoveltyGuard
from ...config import settings


class ContentGenerationStage(PipelineStage):
    """
    Generate content with quality and novelty checks.

    Dependencies:
    - GeminiClient: AI content generation
    - QualityScorer: Content quality evaluation
    - NoveltyGuard: Duplicate prevention

    Updates context with:
    - content: Generated content (hook, script, CTA, metadata)
    """

    def __init__(
        self,
        gemini: GeminiClient,
        quality_scorer: QualityScorer,
        novelty_guard: NoveltyGuard
    ):
        super().__init__("ContentGeneration")
        self.gemini = gemini
        self.quality_scorer = quality_scorer
        self.novelty_guard = novelty_guard

    def execute(self, context: PipelineContext) -> Result[PipelineContext, str]:
        """Generate and validate content."""
        try:
            # Generate content
            self.logger.info("🔮 Calling Gemini API...")

            content = self.gemini.generate(
                topic=context.topic or settings.CHANNEL_TOPIC,
                style=settings.CONTENT_STYLE,
                duration=settings.TARGET_DURATION,
                additional_context=settings.ADDITIONAL_PROMPT_CONTEXT
            )

            self.logger.info("✅ Gemini response received")

            # Quality check
            full_text = " ".join([content.hook, *content.script, content.cta])

            score_result = self.quality_scorer.score(
                sentences=[content.hook] + content.script + [content.cta],
                title=content.metadata.get("title", "")
            )
            score = score_result.get("overall", 0.0)

            self.logger.info(
                f"Quality: {score_result.get('quality', 0):.2f} | "
                f"Viral: {score_result.get('viral', 0):.2f} | "
                f"Retention: {score_result.get('retention', 0):.2f} | "
                f"Overall: {score:.2f}"
            )

            if score < settings.MIN_QUALITY_SCORE:
                return Result.err(
                    f"Quality too low: {score:.2f} < {settings.MIN_QUALITY_SCORE}"
                )

            # Novelty check
            decision = self.novelty_guard.check_novelty(
                channel=context.channel or settings.CHANNEL_NAME,
                title=content.metadata.get("title", ""),
                script=full_text,
                search_term=content.search_queries[0] if content.search_queries else None,
                category=context.topic or settings.CHANNEL_TOPIC,
                lang=settings.LANG
            )

            if not decision.ok:
                return Result.err(f"Content not novel: {decision.reason}")

            # Structure content for next stages
            structured_content = {
                "hook": content.hook,
                "script": content.script,
                "cta": content.cta,
                "search_queries": content.search_queries,
                "main_visual_focus": content.main_visual_focus,
                "metadata": content.metadata,
                "sentences": [content.hook] + content.script + [content.cta],
                "quality_score": score
            }

            # Update context
            context.content = structured_content

            self.logger.info(f"✅ Content: {len(structured_content['sentences'])} sentences")

            return Result.ok(context)

        except Exception as e:
            self.logger.error(f"Content generation error: {e}", exc_info=True)
            return Result.err(f"Content generation failed: {str(e)}")
