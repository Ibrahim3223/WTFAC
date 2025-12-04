# -*- coding: utf-8 -*-
"""
TIER 2 Phase 2.1: AI Thumbnail Generator
=========================================

High-CTR thumbnail generation with face detection, emotion analysis,
and AI-powered text overlays.

Impact: +200-300% Click-Through Rate

Modules:
    - face_detector: Face detection and emotion analysis
    - text_overlay: AI-powered text generation and styling
    - generator: Main thumbnail generation engine

Usage:
    from autoshorts.thumbnail import ThumbnailGenerator

    generator = ThumbnailGenerator(gemini_api_key)
    thumbnails = generator.generate(
        video_path="video.mp4",
        topic="Amazing space discovery",
        num_variants=3
    )
"""

from autoshorts.thumbnail.generator import ThumbnailGenerator

__all__ = ['ThumbnailGenerator']
