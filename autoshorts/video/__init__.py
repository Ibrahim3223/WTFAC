# -*- coding: utf-8 -*-
"""Video processing module."""

from .pexels_client import PexelsClient
from .downloader import VideoDownloader
from .segment_maker import SegmentMaker
from .shot_variety import ShotVarietyManager, ShotType, PacingStyle, ShotPlan

# TIER 1 VIRAL SYSTEM - Color Grading
from .color_grader import (
    ColorGrader,
    LUTPreset,
    GradingIntensity,
    LUTDefinition,
    GradingPlan,
    VideoGradingPlan,
    select_lut_simple,
    create_simple_grading_plan,
)
from .mood_analyzer import (
    MoodAnalyzer,
    MoodCategory,
    MoodAnalysis,
    analyze_mood_simple,
    get_lut_for_topic,
)

__all__ = [
    'PexelsClient',
    'VideoDownloader',
    'SegmentMaker',
    'ShotVarietyManager',
    'ShotType',
    'PacingStyle',
    'ShotPlan',
    # TIER 1: Color Grading
    'ColorGrader',
    'LUTPreset',
    'GradingIntensity',
    'LUTDefinition',
    'GradingPlan',
    'VideoGradingPlan',
    'select_lut_simple',
    'create_simple_grading_plan',
    # Mood Analysis
    'MoodAnalyzer',
    'MoodCategory',
    'MoodAnalysis',
    'analyze_mood_simple',
    'get_lut_for_topic',
]
