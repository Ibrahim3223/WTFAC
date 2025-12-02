"""
Unit tests for ContentGenerationStage.
"""

import pytest
from autoshorts.pipeline.stages import ContentGenerationStage


@pytest.mark.unit
class TestContentGenerationStage:
    """Test ContentGenerationStage."""

    def test_execute_success(
        self,
        pipeline_context,
        mock_gemini,
        mock_quality_scorer,
        mock_novelty_guard
    ):
        """Test successful content generation."""
        stage = ContentGenerationStage(
            gemini=mock_gemini,
            quality_scorer=mock_quality_scorer,
            novelty_guard=mock_novelty_guard
        )

        result = stage.execute(pipeline_context)

        assert result.is_ok()

        context = result.unwrap()
        assert context.content is not None
        assert "hook" in context.content
        assert "script" in context.content
        assert "cta" in context.content
        assert "sentences" in context.content

        # Verify mocks were called
        mock_gemini.generate.assert_called_once()
        mock_quality_scorer.score.assert_called_once()
        mock_novelty_guard.check_novelty.assert_called_once()

    def test_execute_quality_too_low(
        self,
        pipeline_context,
        mock_gemini,
        mock_quality_scorer,
        mock_novelty_guard
    ):
        """Test content generation fails when quality too low."""
        # Set low quality score
        mock_quality_scorer.score.return_value = {
            "quality": 4.0,
            "viral": 3.0,
            "retention": 5.0,
            "overall": 4.0
        }

        stage = ContentGenerationStage(
            gemini=mock_gemini,
            quality_scorer=mock_quality_scorer,
            novelty_guard=mock_novelty_guard
        )

        result = stage.execute(pipeline_context)

        assert result.is_err()
        assert "Quality too low" in result.unwrap_err()

    def test_execute_not_novel(
        self,
        pipeline_context,
        mock_gemini,
        mock_quality_scorer,
        mock_novelty_guard
    ):
        """Test content generation fails when not novel."""
        from autoshorts.state.novelty_guard import NoveltyDecision

        # Set not novel
        mock_novelty_guard.check_novelty.return_value = NoveltyDecision(
            ok=False,
            reason="Too similar to previous content",
            entity_cooldown_ok=True,
            simhash_ok=False,
            entity_jaccard_ok=True
        )

        stage = ContentGenerationStage(
            gemini=mock_gemini,
            quality_scorer=mock_quality_scorer,
            novelty_guard=mock_novelty_guard
        )

        result = stage.execute(pipeline_context)

        assert result.is_err()
        assert "not novel" in result.unwrap_err()
