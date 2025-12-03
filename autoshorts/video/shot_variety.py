# -*- coding: utf-8 -*-
"""
Shot Variety & Visual Pacing Manager
- Alternates between wide, medium, close-up shots
- Tracks shot history to avoid repetition
- Matches pacing to content type
- Dynamic scene duration based on context
"""
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ShotType(Enum):
    """Video shot composition types."""
    WIDE = "wide"           # Landscape, cityscape, establishing shots
    MEDIUM = "medium"       # Medium distance, people, objects
    CLOSEUP = "closeup"     # Close-up, details, faces
    AERIAL = "aerial"       # Drone shots, bird's eye view
    ACTION = "action"       # Fast movement, dynamic


class PacingStyle(Enum):
    """Visual pacing styles."""
    SLOW = "slow"           # 5-8 seconds per shot
    MEDIUM = "medium"       # 3-5 seconds per shot
    FAST = "fast"           # 1-3 seconds per shot
    DYNAMIC = "dynamic"     # Variable pacing


# Shot type keywords for search query enhancement
SHOT_TYPE_KEYWORDS = {
    ShotType.WIDE: [
        "landscape", "panorama", "wide angle", "vista", "scenery",
        "establishing shot", "cityscape", "skyline", "horizon",
    ],
    ShotType.MEDIUM: [
        "medium shot", "person", "people", "group", "activity",
        "street level", "waist up", "environmental",
    ],
    ShotType.CLOSEUP: [
        "close up", "closeup", "detail", "macro", "face",
        "hands", "eyes", "texture", "focus",
    ],
    ShotType.AERIAL: [
        "aerial", "drone", "birds eye", "overhead", "top down",
        "flying over", "from above",
    ],
    ShotType.ACTION: [
        "action", "movement", "fast", "dynamic", "motion",
        "running", "chase", "speed", "energy",
    ],
}


# Content-based shot type suggestions
CONTENT_SHOT_SUGGESTIONS = {
    "location": [ShotType.WIDE, ShotType.AERIAL, ShotType.MEDIUM],
    "person": [ShotType.MEDIUM, ShotType.CLOSEUP],
    "object": [ShotType.CLOSEUP, ShotType.MEDIUM],
    "landscape": [ShotType.WIDE, ShotType.AERIAL],
    "action": [ShotType.ACTION, ShotType.MEDIUM],
    "detail": [ShotType.CLOSEUP],
    "overview": [ShotType.WIDE, ShotType.AERIAL],
}


@dataclass
class ShotPlan:
    """Plan for a video shot."""
    shot_type: ShotType
    pacing_style: PacingStyle
    duration_range: tuple  # (min, max) seconds
    search_keywords: List[str]
    priority: int  # 1 (highest) to 5 (lowest)


class ShotVarietyManager:
    """Manages shot variety and visual pacing for videos."""

    def __init__(self, variety_strength: str = "medium"):
        """
        Initialize shot variety manager.

        Args:
            variety_strength: How strongly to enforce variety
                - "low": Minimal variety enforcement
                - "medium": Balanced variety (recommended)
                - "high": Strict variety enforcement
        """
        self.variety_strength = variety_strength
        self.shot_history: List[ShotType] = []
        self.max_history = 8  # Track last 8 shots

        # Variety enforcement rules
        self.variety_rules = {
            "low": {
                "max_consecutive": 4,
                "require_alternation": False,
            },
            "medium": {
                "max_consecutive": 2,
                "require_alternation": True,
            },
            "high": {
                "max_consecutive": 1,
                "require_alternation": True,
            },
        }

    def plan_shot(
        self,
        sentence: str,
        sentence_index: int,
        sentence_type: str,
        total_sentences: int,
        keywords: List[str],
    ) -> ShotPlan:
        """
        Plan shot type and pacing for a sentence.

        Args:
            sentence: Sentence text
            sentence_index: Index in the video (0-based)
            sentence_type: "hook", "content", "cta"
            total_sentences: Total number of sentences
            keywords: Extracted keywords

        Returns:
            ShotPlan with recommendations
        """
        # Determine base shot type from content
        content_type = self._analyze_content_type(sentence, keywords)
        suggested_types = CONTENT_SHOT_SUGGESTIONS.get(
            content_type, [ShotType.MEDIUM, ShotType.WIDE]
        )

        # Apply variety rules to select shot type
        shot_type = self._select_shot_with_variety(suggested_types)

        # Determine pacing based on sentence type and position
        pacing_style = self._determine_pacing(
            sentence_type, sentence_index, total_sentences
        )

        # Calculate duration range
        duration_range = self._get_duration_range(pacing_style)

        # Build search keywords for this shot type
        search_keywords = self._build_shot_keywords(shot_type, keywords)

        # Priority based on position
        priority = 1 if sentence_type == "hook" else 3

        # Track shot in history
        self.shot_history.append(shot_type)
        if len(self.shot_history) > self.max_history:
            self.shot_history.pop(0)

        logger.debug(
            f"Shot plan: type={shot_type.value}, pacing={pacing_style.value}, "
            f"keywords={search_keywords[:3]}"
        )

        return ShotPlan(
            shot_type=shot_type,
            pacing_style=pacing_style,
            duration_range=duration_range,
            search_keywords=search_keywords,
            priority=priority,
        )

    def _analyze_content_type(self, sentence: str, keywords: List[str]) -> str:
        """Analyze sentence to determine content type."""
        sentence_lower = sentence.lower()
        keywords_lower = [k.lower() for k in keywords]

        # Check for location indicators
        location_words = ["city", "mountain", "ocean", "desert", "forest", "country", "place"]
        if any(word in sentence_lower or word in keywords_lower for word in location_words):
            return "location"

        # Check for person indicators
        person_words = ["people", "person", "man", "woman", "they", "he", "she", "who"]
        if any(word in sentence_lower for word in person_words):
            return "person"

        # Check for action indicators
        action_words = ["running", "moving", "jumping", "flying", "racing", "chasing"]
        if any(word in sentence_lower or word in keywords_lower for word in action_words):
            return "action"

        # Check for detail indicators
        detail_words = ["detail", "close", "look", "examine", "see", "tiny", "small"]
        if any(word in sentence_lower for word in detail_words):
            return "detail"

        # Check for overview indicators
        overview_words = ["overall", "entire", "whole", "complete", "all", "total"]
        if any(word in sentence_lower for word in overview_words):
            return "overview"

        # Default
        return "overview" if len(keywords_lower) > 0 else "medium"

    def _select_shot_with_variety(self, suggested_types: List[ShotType]) -> ShotType:
        """Select shot type while enforcing variety rules."""
        if not self.shot_history:
            # First shot - prefer WIDE for establishing
            return ShotType.WIDE if ShotType.WIDE in suggested_types else suggested_types[0]

        rules = self.variety_rules.get(self.variety_strength, self.variety_rules["medium"])
        max_consecutive = rules["max_consecutive"]
        require_alternation = rules["require_alternation"]

        # Check recent history
        recent_shots = self.shot_history[-max_consecutive:]
        last_shot = self.shot_history[-1]

        # If all recent shots are the same, force different
        if len(set(recent_shots)) == 1 and recent_shots[0] == last_shot:
            # Pick a different type
            different_types = [t for t in suggested_types if t != last_shot]
            if different_types:
                return different_types[0]

        # If require_alternation, avoid same as last
        if require_alternation and suggested_types:
            different_types = [t for t in suggested_types if t != last_shot]
            if different_types:
                return different_types[0]

        # Default to first suggested
        return suggested_types[0] if suggested_types else ShotType.MEDIUM

    def _determine_pacing(
        self, sentence_type: str, sentence_index: int, total_sentences: int
    ) -> PacingStyle:
        """Determine pacing style based on context."""
        # Hook should be FAST to grab attention
        if sentence_type == "hook":
            return PacingStyle.FAST

        # CTA should be MEDIUM for clarity
        if sentence_type == "cta":
            return PacingStyle.MEDIUM

        # Content pacing based on position
        # First 20%: FAST (maintain hook energy)
        # Middle 60%: MEDIUM (comfortable viewing)
        # Last 20%: DYNAMIC (build to conclusion)
        position_ratio = sentence_index / max(total_sentences, 1)

        if position_ratio < 0.2:
            return PacingStyle.FAST
        elif position_ratio > 0.8:
            return PacingStyle.DYNAMIC
        else:
            return PacingStyle.MEDIUM

    def _get_duration_range(self, pacing_style: PacingStyle) -> tuple:
        """Get duration range for pacing style."""
        ranges = {
            PacingStyle.FAST: (2.0, 4.0),
            PacingStyle.MEDIUM: (3.0, 6.0),
            PacingStyle.SLOW: (5.0, 8.0),
            PacingStyle.DYNAMIC: (2.0, 6.0),
        }
        return ranges.get(pacing_style, (3.0, 6.0))

    def _build_shot_keywords(
        self, shot_type: ShotType, content_keywords: List[str]
    ) -> List[str]:
        """
        Build search keywords combining shot type and content.

        Content keywords FIRST (for video matching)
        Shot type keywords SECOND (optional framing context)
        """
        keywords = []

        # Add content keywords FIRST (highest priority for search)
        keywords.extend(content_keywords[:3])

        # Add shot type keywords LAST (optional framing hints)
        # These provide visual variety context but don't override content
        shot_keywords = SHOT_TYPE_KEYWORDS.get(shot_type, [])
        keywords.extend(shot_keywords[:2])

        return keywords

    def get_variety_stats(self) -> Dict:
        """Get statistics about shot variety."""
        if not self.shot_history:
            return {"total_shots": 0, "variety_score": 0.0}

        # Calculate variety score (0.0 to 1.0)
        unique_types = len(set(self.shot_history))
        total_types = len(ShotType)
        variety_score = unique_types / total_types

        # Count distribution
        distribution = {}
        for shot_type in self.shot_history:
            distribution[shot_type.value] = distribution.get(shot_type.value, 0) + 1

        return {
            "total_shots": len(self.shot_history),
            "unique_types": unique_types,
            "variety_score": variety_score,
            "distribution": distribution,
        }

    def reset(self):
        """Reset shot history (e.g., for new video)."""
        self.shot_history = []
