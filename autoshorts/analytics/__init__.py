# -*- coding: utf-8 -*-
"""Analytics and viral intelligence module."""

# TIER 3: AI-Powered Viral Engine - Viral Pattern Recognition
from .viral_scraper import (
    ViralScraper,
    VideoMetadata,
    ScrapedVideo,
    NicheType,
)
from .pattern_analyzer import (
    PatternAnalyzer,
    ViralPatternData,
    PatternInsight,
    PatternFeatures,
)
from .viral_predictor import (
    ViralPredictor,
    PredictionResult,
    ViralScore,
    ViralFactors,
)

# TIER 3: AI-Powered Viral Engine - Competitor Analysis
from .competitor_analyzer import (
    CompetitorAnalyzer,
    CompetitorChannel,
    CompetitorInsight,
    CompetitorReport,
    TrendingTopic,
    GapAnalysis,
    PerformanceLevel,
)
from .trend_detector import (
    TrendDetector,
    Trend,
    TrendReport,
    TrendPhase,
    TrendType,
    TrendMetrics,
)

__all__ = [
    # Viral Scraping
    'ViralScraper',
    'VideoMetadata',
    'ScrapedVideo',
    'NicheType',
    # Pattern Analysis
    'PatternAnalyzer',
    'ViralPatternData',
    'PatternInsight',
    'PatternFeatures',
    # Viral Prediction
    'ViralPredictor',
    'PredictionResult',
    'ViralScore',
    'ViralFactors',
    # Competitor Analysis
    'CompetitorAnalyzer',
    'CompetitorChannel',
    'CompetitorInsight',
    'CompetitorReport',
    'TrendingTopic',
    'GapAnalysis',
    'PerformanceLevel',
    # Trend Detection
    'TrendDetector',
    'Trend',
    'TrendReport',
    'TrendPhase',
    'TrendType',
    'TrendMetrics',
]
