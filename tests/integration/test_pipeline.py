"""
Integration tests for pipeline.
"""

import pytest
from autoshorts.pipeline import Pipeline, PipelineContext
from autoshorts.pipeline.stages import ContentGenerationStage, TTSStage


@pytest.mark.integration
class TestPipeline:
    """Test pipeline integration."""

    def test_pipeline_with_two_stages(
        self,
        pipeline_context,
        mock_gemini,
        mock_quality_scorer,
        mock_novelty_guard,
        mock_tts
    ):
        """Test pipeline with content generation and TTS."""
        # Create pipeline
        pipeline = Pipeline([
            ContentGenerationStage(
                gemini=mock_gemini,
                quality_scorer=mock_quality_scorer,
                novelty_guard=mock_novelty_guard
            ),
            TTSStage(
                tts_handler=mock_tts
            )
        ])

        # Run
        result = pipeline.run(pipeline_context)

        # Verify
        assert result.is_ok()

        context = result.unwrap()
        assert context.content is not None
        assert context.audio_segments is not None
        assert len(context.audio_segments) == len(context.content["sentences"])

    def test_pipeline_stops_on_error(
        self,
        pipeline_context,
        mock_gemini,
        mock_quality_scorer,
        mock_novelty_guard,
        mock_tts
    ):
        """Test pipeline stops when stage fails."""
        # Make quality scorer fail
        mock_quality_scorer.score.return_value = {
            "quality": 2.0,
            "viral": 1.0,
            "retention": 3.0,
            "overall": 2.0
        }

        pipeline = Pipeline([
            ContentGenerationStage(
                gemini=mock_gemini,
                quality_scorer=mock_quality_scorer,
                novelty_guard=mock_novelty_guard
            ),
            TTSStage(
                tts_handler=mock_tts
            )
        ], stop_on_error=True)

        result = pipeline.run(pipeline_context)

        # Pipeline should fail
        assert result.is_err()

        # TTS should not have been called
        mock_tts.synthesize.assert_not_called()
