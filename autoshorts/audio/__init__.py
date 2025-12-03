"""Audio processing module."""

from .bgm_manager import BGMManager

# TIER 1 VIRAL SYSTEM - Sound Effects Layer
from .sfx_manager import (
    SFXManager,
    SFXLibrary,
    SFXCategory,
    SFXIntensity,
    SFXTiming,
    SFXFile,
    SFXPlacement,
    SFXPlan,
    create_sfx_plan_simple,
)
from .timing_optimizer import (
    TimingOptimizer,
    TimingStrategy,
    RhythmStyle,
    TimingAnalysis,
    optimize_sfx_timing,
)

__all__ = [
    'BGMManager',
    # TIER 1: SFX System
    'SFXManager',
    'SFXLibrary',
    'SFXCategory',
    'SFXIntensity',
    'SFXTiming',
    'SFXFile',
    'SFXPlacement',
    'SFXPlan',
    'create_sfx_plan_simple',
    # Timing Optimization
    'TimingOptimizer',
    'TimingStrategy',
    'RhythmStyle',
    'TimingAnalysis',
    'optimize_sfx_timing',
]
