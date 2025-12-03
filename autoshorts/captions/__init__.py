# -*- coding: utf-8 -*-
"""Captions module."""

from .karaoke_ass import build_karaoke_ass
from .renderer import CaptionRenderer

# TIER 1 VIRAL SYSTEM - Advanced Caption Animations
from .caption_animator import (
    CaptionAnimator,
    AnimationStyle,
    AnimationIntensity,
    StyleDefinition,
    select_animation_style,
    create_animated_subtitle,
)

__all__ = [
    'build_karaoke_ass',
    'CaptionRenderer',
    # TIER 1: Caption Animations
    'CaptionAnimator',
    'AnimationStyle',
    'AnimationIntensity',
    'StyleDefinition',
    'select_animation_style',
    'create_animated_subtitle',
]
