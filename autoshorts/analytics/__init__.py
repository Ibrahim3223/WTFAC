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
]
