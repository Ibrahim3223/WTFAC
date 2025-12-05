"""
Content Generation Stage - Generates viral content using Gemini AI.
"""

from typing import Dict, Any, Optional

from ..base import PipelineStage, PipelineContext
from ...core import Result, ContentGenerationError, QualityError, NoveltyError
from ...content.gemini_client import GeminiClient
from ...content.quality_scorer import QualityScorer
from ...state.novelty_guard import NoveltyGuard
from ...config import settings

# TIER 1 VIRAL SYSTEM
from ...content.hook_generator import HookGenerator, EmotionType as HookEmotionType
from ...content.emotion_analyzer import EmotionAnalyzer, EmotionType as AnalyzerEmotionType
from ...content.viral_patterns import ViralPatternAnalyzer
from ...content.retention_patterns import CliffhangerInjector


# ============================================================================
# EMOTION MAPPING (EmotionAnalyzer → HookGenerator)
# ============================================================================
# EmotionAnalyzer has 16 emotions, HookGenerator only supports 8
# Map unsupported emotions to closest supported ones

EMOTION_MAPPING = {
    # Supported emotions (direct mapping)
    "joy": "joy",
    "fear": "fear",
    "anger": "anger",
    "surprise": "surprise",
    "disgust": "disgust",
    "trust": "trust",
    "anticipation": "anticipation",
    "curiosity": "curiosity",

    # Unsupported emotions (map to closest match)
    "sadness": "trust",           # Comfort/empathy
    "awe": "surprise",             # Wonder feeling
    "fomo": "anticipation",        # Fear of missing → action
    "validation": "trust",         # Confirmation → trust
    "nostalgia": "joy",            # Nostalgic feelings
    "schadenfreude": "surprise",   # Unexpected joy
    "inspiration": "anticipation", # Motivation → action
    "neutral": "curiosity",        # Default safe choice
}


def map_emotion_for_hook(emotion_value: str) -> HookEmotionType:
    """
    Map EmotionAnalyzer emotion to HookGenerator emotion.

    Args:
        emotion_value: Emotion string from EmotionAnalyzer

    Returns:
        HookEmotionType enum value
    """
    mapped = EMOTION_MAPPING.get(emotion_value.lower(), "curiosity")
    return HookEmotionType(mapped)


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
        novelty_guard: NoveltyGuard,
        hook_generator: Optional[HookGenerator] = None,
        emotion_analyzer: Optional[EmotionAnalyzer] = None,
        viral_pattern_analyzer: Optional[ViralPatternAnalyzer] = None,
        cliffhanger_injector: Optional[CliffhangerInjector] = None
    ):
        super().__init__("ContentGeneration")
        self.gemini = gemini
        self.quality_scorer = quality_scorer
        self.novelty_guard = novelty_guard

        # TIER 1 VIRAL SYSTEM (optional for backward compatibility)
        self.hook_generator = hook_generator
        self.emotion_analyzer = emotion_analyzer
        self.viral_pattern_analyzer = viral_pattern_analyzer
        self.cliffhanger_injector = cliffhanger_injector

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

            # TIER 1: Analyze emotions and optimize hook
            if self.emotion_analyzer and self.hook_generator:
                self.logger.info("🎭 Analyzing emotions...")

                # Analyze emotional profile
                full_text = " ".join([content.hook, *content.script, content.cta])
                emotion_profile = self.emotion_analyzer.analyze_text(
                    text=full_text,
                    content_type=settings.CONTENT_STYLE
                )

                self.logger.info(
                    f"Primary emotion: {emotion_profile.primary_emotion.value} "
                    f"(intensity: {emotion_profile.intensity.value})"
                )

                # Generate optimized hooks with A/B testing
                self.logger.info("🎣 Generating viral hooks...")

                # Map emotion (EmotionAnalyzer → HookGenerator)
                target_emotion = map_emotion_for_hook(emotion_profile.primary_emotion.value)
                self.logger.info(
                    f"Emotion mapping: {emotion_profile.primary_emotion.value} → {target_emotion.value}"
                )

                hook_result = self.hook_generator.generate_hooks(
                    topic=context.topic or settings.CHANNEL_TOPIC,
                    content_type=settings.CONTENT_STYLE,
                    target_emotion=target_emotion,
                    keywords=None,  # Auto-extract
                    num_variants=3
                )

                # Use best hook
                best_hook = hook_result.best_variant
                self.logger.info(
                    f"🏆 Best hook: '{best_hook.text[:50]}...' "
                    f"(score: {best_hook.score:.2f})"
                )

                # Replace original hook with AI-optimized one
                content.hook = best_hook.text

                # Store emotion data for later stages
                context.emotion_profile = emotion_profile
                context.hook_variants = hook_result

            # TIER 1: Select viral patterns
            if self.viral_pattern_analyzer:
                self.logger.info("📊 Selecting viral patterns...")

                pattern_matches = self.viral_pattern_analyzer.analyze_content(
                    topic=context.topic or settings.CHANNEL_TOPIC,
                    content_type=settings.CONTENT_STYLE,
                    duration=settings.TARGET_DURATION,  # Seconds
                    keywords=[]  # Auto-extract from content
                )

                if pattern_matches:
                    best_match = pattern_matches[0]
                    self.logger.info(
                        f"🎯 Pattern: {best_match.pattern.description} "
                        f"(match: {best_match.match_score:.2f}, effectiveness: {best_match.pattern.effectiveness_score:.2f})"
                    )

                    # Store pattern for later stages
                    context.viral_pattern = best_match

            # TIER 1: Inject cliffhangers for retention optimization
            if self.cliffhanger_injector:
                self.logger.info("🎯 Injecting retention cliffhangers...")

                # Get current sentences (hook + script + cta)
                current_sentences = [content.hook] + content.script + [content.cta]

                # Determine emotion for cliffhanger selection
                emotion = None
                if hasattr(context, 'emotion_profile') and context.emotion_profile:
                    emotion = context.emotion_profile.primary_emotion.value

                # Inject cliffhangers
                sentences_with_cliffhangers = self.cliffhanger_injector.inject_cliffhangers(
                    sentences=current_sentences,
                    target_duration=settings.TARGET_DURATION,
                    emotion=emotion,
                    content_type=settings.CONTENT_STYLE
                )

                # Update content with new sentences
                # First sentence is hook, last is CTA, rest is script
                content.hook = sentences_with_cliffhangers[0]
                content.cta = sentences_with_cliffhangers[-1]
                content.script = sentences_with_cliffhangers[1:-1]

                cliffhanger_count = len(sentences_with_cliffhangers) - len(current_sentences)
                self.logger.info(
                    f"✅ Injected {cliffhanger_count} cliffhangers "
                    f"({len(current_sentences)} → {len(sentences_with_cliffhangers)} sentences)"
                )

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
