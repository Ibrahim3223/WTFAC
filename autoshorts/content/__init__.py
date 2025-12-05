# -*- coding: utf-8 -*-
"""Content generation module."""

from .gemini_client import GeminiClient
from .quality_scorer import QualityScorer
from .text_utils import *

# TIER 1 VIRAL SYSTEM - AI-Powered Hook Generation
from .hook_generator import (
    HookGenerator,
    HookType,
    EmotionType,
    HookVariant,
    HookGenerationResult,
    generate_unique_hook,
    generate_ab_hooks,
)
from .viral_patterns import (
    ViralPatternAnalyzer,
    ViralPattern,
    PatternType,
    PatternMatch,
    get_viral_patterns_for_content,
    get_best_hook_pattern,
)
from .emotion_analyzer import (
    EmotionAnalyzer,
    EmotionalProfile,
    EmotionSignal,
    EmotionTrigger,
    EmotionIntensity,
    analyze_content_emotion,
    get_recommended_arc,
)
from .retention_patterns import (
    CliffhangerInjector,
    CliffhangerType,
    inject_cliffhangers_simple,
    get_random_cliffhanger,
)

# TIER 2 PROFESSIONAL POLISH - Retention Optimization
from .curiosity_generator import (
    CuriosityGenerator,
    CuriosityType,
    CuriosityGap,
    PatternInterrupt,
    InterruptIntensity,
)
from .story_arc import (
    StoryArcOptimizer,
    StoryArcPlan,
    StoryAct,
    StoryBeat,
    EmotionalTone,
    ProgressMarker,
)

# TIER 3 AI-POWERED VIRAL ENGINE - Idea Generation
from .idea_generator import (
    IdeaGenerator,
    ContentIdea,
    IdeaBatch,
    IdeaSource,
    IdeaUrgency,
)

__all__ = [
    'GeminiClient',
    'QualityScorer',
    # TIER 1: AI Hook System
    'HookGenerator',
    'HookType',
    'EmotionType',
    'HookVariant',
    'HookGenerationResult',
    'generate_unique_hook',
    'generate_ab_hooks',
    # Viral Patterns
    'ViralPatternAnalyzer',
    'ViralPattern',
    'PatternType',
    'PatternMatch',
    'get_viral_patterns_for_content',
    'get_best_hook_pattern',
    # Emotion Analysis
    'EmotionAnalyzer',
    'EmotionalProfile',
    'EmotionSignal',
    'EmotionTrigger',
    'EmotionIntensity',
    'analyze_content_emotion',
    'get_recommended_arc',
    # Cliffhanger System
    'CliffhangerInjector',
    'CliffhangerType',
    'inject_cliffhangers_simple',
    'get_random_cliffhanger',
    # TIER 2: Retention Optimization
    'CuriosityGenerator',
    'CuriosityType',
    'CuriosityGap',
    'PatternInterrupt',
    'InterruptIntensity',
    'StoryArcOptimizer',
    'StoryArcPlan',
    'StoryAct',
    'StoryBeat',
    'EmotionalTone',
    'ProgressMarker',
    # TIER 3: Idea Generation
    'IdeaGenerator',
    'ContentIdea',
    'IdeaBatch',
    'IdeaSource',
    'IdeaUrgency',
]
