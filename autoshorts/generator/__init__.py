# -*- coding: utf-8 -*-
"""
Video Generator Module - Complete Automation
=============================================

The final integration layer that combines all TIER 1, 2, 3 systems.
"""

from .auto_video_generator import (
    AutoVideoGenerator,
    VideoConfig,
    GeneratedVideo,
    VideoBatch,
)

__all__ = [
    'AutoVideoGenerator',
    'VideoConfig',
    'GeneratedVideo',
    'VideoBatch',
]
