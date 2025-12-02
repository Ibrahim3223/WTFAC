"""
Pytest fixtures and configuration.

Shared fixtures for all tests.
"""

import pytest
import tempfile
import shutil
from unittest.mock import MagicMock
from pathlib import Path

from autoshorts.core import Container, ServiceLifetime, Result
from autoshorts.pipeline import PipelineContext
from autoshorts.content.gemini_client import GeminiClient, ContentResponse
from autoshorts.content.quality_scorer import QualityScorer
from autoshorts.tts.edge_handler import TTSHandler
from autoshorts.video.pexels_client import PexelsClient
from autoshorts.state.novelty_guard import NoveltyGuard, NoveltyDecision


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after test."""
    tmpdir = tempfile.mkdtemp(prefix="test_shorts_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def pipeline_context(temp_dir):
    """Provide a basic pipeline context."""
    return PipelineContext(
        channel="TestChannel",
        topic="Test facts",
        temp_dir=temp_dir
    )


@pytest.fixture
def mock_gemini():
    """Mock Gemini client."""
    mock = MagicMock(spec=GeminiClient)

    # Default response
    mock.generate.return_value = ContentResponse(
        hook="Did you know this amazing fact?",
        script=[
            "Here's the first part of the story.",
            "And here's the second part.",
            "Finally, the conclusion."
        ],
        cta="Follow for more!",
        search_queries=["nature landscape", "mountains"],
        main_visual_focus="mountains",
        metadata={
            "title": "Amazing Test Fact",
            "description": "Test description",
            "tags": ["test", "facts"]
        }
    )

    return mock


@pytest.fixture
def mock_quality_scorer():
    """Mock quality scorer."""
    mock = MagicMock(spec=QualityScorer)

    # Default high-quality score
    mock.score.return_value = {
        "quality": 8.0,
        "viral": 7.5,
        "retention": 8.5,
        "overall": 8.0
    }

    return mock


@pytest.fixture
def mock_novelty_guard():
    """Mock novelty guard."""
    mock = MagicMock(spec=NoveltyGuard)

    # Default: content is novel
    mock.check_novelty.return_value = NoveltyDecision(
        ok=True,
        reason="Content is novel",
        entity_cooldown_ok=True,
        simhash_ok=True,
        entity_jaccard_ok=True
    )

    mock.register_item.return_value = None

    return mock


@pytest.fixture
def mock_tts():
    """Mock TTS handler."""
    mock = MagicMock(spec=TTSHandler)

    # Default: successful TTS
    def mock_synthesize(text, wav_out):
        # Create empty wav file
        Path(wav_out).touch()
        return (2.5, [])  # 2.5 seconds, no word timings

    mock.synthesize.side_effect = mock_synthesize

    return mock


@pytest.fixture
def mock_pexels():
    """Mock Pexels client."""
    mock = MagicMock(spec=PexelsClient)

    # Default: return some videos
    mock.search_simple.return_value = [
        ("video1", "https://example.com/video1.mp4"),
        ("video2", "https://example.com/video2.mp4"),
        ("video3", "https://example.com/video3.mp4"),
    ]

    return mock


@pytest.fixture
def container():
    """Provide a fresh DI container for each test."""
    return Container()


@pytest.fixture
def mock_container(
    mock_gemini,
    mock_quality_scorer,
    mock_novelty_guard,
    mock_tts,
    mock_pexels
):
    """Provide a container with all mocked services."""
    container = Container()

    container.register_instance(GeminiClient, mock_gemini)
    container.register_instance(QualityScorer, mock_quality_scorer)
    container.register_instance(NoveltyGuard, mock_novelty_guard)
    container.register_instance(TTSHandler, mock_tts)
    container.register_instance(PexelsClient, mock_pexels)

    return container
