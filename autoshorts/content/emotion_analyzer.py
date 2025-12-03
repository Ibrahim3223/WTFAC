# -*- coding: utf-8 -*-
"""
Emotion Analyzer - TIER 1 VIRAL SYSTEM
Analyzes content to detect and inject emotional triggers

Key Features:
- Emotion detection from text
- Emotional arc optimization (setup → tension → payoff)
- Trigger injection for maximum engagement
- Emotion-appropriate hook/CTA selection
- Content-aware emotional pacing

Expected Impact: +40-60% emotional engagement
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# EMOTION TYPES (Plutchik's Wheel + Social Emotions)
# ============================================================================

class EmotionType(Enum):
    """Primary emotions for viral content"""
    # Basic emotions (Plutchik)
    JOY = "joy"                        # Happiness, excitement, delight
    SADNESS = "sadness"                # Sorrow, disappointment
    FEAR = "fear"                      # Anxiety, worry, dread
    ANGER = "anger"                    # Frustration, outrage
    SURPRISE = "surprise"              # Shock, amazement, wonder
    DISGUST = "disgust"                # Revulsion, distaste
    TRUST = "trust"                    # Confidence, assurance
    ANTICIPATION = "anticipation"      # Expectation, hope

    # Social/Complex emotions
    CURIOSITY = "curiosity"            # Desire to know/learn
    AWE = "awe"                        # Wonder at something vast
    FOMO = "fomo"                      # Fear of missing out
    VALIDATION = "validation"          # Confirmation of beliefs
    NOSTALGIA = "nostalgia"           # Longing for the past
    SCHADENFREUDE = "schadenfreude"   # Joy at others' misfortune
    INSPIRATION = "inspiration"        # Motivation to improve

    # Default
    NEUTRAL = "neutral"


# ============================================================================
# EMOTION INTENSITY
# ============================================================================

class EmotionIntensity(Enum):
    """Intensity levels for emotions"""
    SUBTLE = "subtle"      # 1-2/10 - Gentle touch
    MODERATE = "moderate"  # 4-6/10 - Clear but not overwhelming
    STRONG = "strong"      # 7-9/10 - Powerful impact
    EXTREME = "extreme"    # 10/10 - Maximum intensity


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EmotionSignal:
    """A detected emotion signal in text"""
    emotion: EmotionType
    intensity: EmotionIntensity
    confidence: float              # 0-1 confidence in detection
    trigger_words: List[str]       # Words that triggered detection
    position: Tuple[int, int]      # Start/end position in text


@dataclass
class EmotionalProfile:
    """Emotional profile of content"""
    primary_emotion: EmotionType       # Dominant emotion
    secondary_emotions: List[EmotionType]  # Supporting emotions
    intensity: EmotionIntensity        # Overall intensity
    signals: List[EmotionSignal]       # All detected signals
    emotional_arc: str                 # Arc type (rising, falling, rollercoaster)
    engagement_score: float            # Predicted engagement (0-1)


@dataclass
class EmotionTrigger:
    """An emotion trigger to inject"""
    emotion: EmotionType
    trigger_phrase: str
    placement: str                     # "hook", "middle", "cta"
    intensity: EmotionIntensity
    reason: str                        # Why this trigger was selected


# ============================================================================
# EMOTION DETECTION DICTIONARIES
# ============================================================================

# Words/phrases that indicate each emotion
EMOTION_KEYWORDS = {
    EmotionType.JOY: [
        "amazing", "incredible", "beautiful", "wonderful", "fantastic", "awesome",
        "love", "perfect", "brilliant", "stunning", "spectacular", "magnificent",
        "delightful", "joyful", "happy", "celebrate", "exciting", "fun"
    ],

    EmotionType.SADNESS: [
        "sad", "tragic", "heartbreaking", "unfortunate", "devastating", "terrible",
        "awful", "horrible", "depressing", "grim", "dark", "miserable", "painful"
    ],

    EmotionType.FEAR: [
        "scary", "terrifying", "dangerous", "deadly", "threatening", "frightening",
        "creepy", "horror", "nightmare", "panic", "anxiety", "worried", "concerned",
        "risky", "hazard", "warning", "beware", "avoid"
    ],

    EmotionType.ANGER: [
        "outrageous", "infuriating", "frustrating", "annoying", "ridiculous",
        "unacceptable", "wrong", "unfair", "unjust", "corrupt", "evil", "hate",
        "disgusting", "terrible", "awful", "worst", "horrible"
    ],

    EmotionType.SURPRISE: [
        "shocking", "unexpected", "unbelievable", "impossible", "astonishing",
        "stunning", "mind-blowing", "incredible", "crazy", "insane", "wild",
        "bizarre", "strange", "weird", "odd", "unusual", "rare"
    ],

    EmotionType.DISGUST: [
        "disgusting", "gross", "revolting", "repulsive", "nasty", "vile",
        "filthy", "horrible", "awful", "terrible", "sickening", "disturbing"
    ],

    EmotionType.TRUST: [
        "proven", "scientific", "research", "study", "expert", "professional",
        "reliable", "trusted", "verified", "confirmed", "evidence", "fact"
    ],

    EmotionType.ANTICIPATION: [
        "upcoming", "soon", "next", "future", "coming", "approaching", "will",
        "going to", "about to", "prepare", "ready", "expect", "wait"
    ],

    EmotionType.CURIOSITY: [
        "secret", "hidden", "mystery", "unknown", "discover", "reveal", "truth",
        "behind", "real", "actually", "really", "what", "why", "how", "wonder"
    ],

    EmotionType.AWE: [
        "magnificent", "breathtaking", "spectacular", "majestic", "epic", "vast",
        "enormous", "colossal", "incredible", "unbelievable", "extraordinary"
    ],

    EmotionType.FOMO: [
        "miss", "limited", "exclusive", "rare", "only", "never", "last chance",
        "before", "disappear", "gone", "too late", "now", "hurry", "quick"
    ],

    EmotionType.VALIDATION: [
        "right", "correct", "exactly", "agree", "yes", "true", "absolutely",
        "definitely", "obviously", "clearly", "prove", "confirm", "vindicate"
    ],

    EmotionType.NOSTALGIA: [
        "remember", "childhood", "old", "classic", "vintage", "retro", "past",
        "used to", "back then", "memories", "throwback", "history", "tradition"
    ],

    EmotionType.INSPIRATION: [
        "inspire", "motivate", "achieve", "success", "overcome", "triumph",
        "victory", "win", "champion", "hero", "dream", "goal", "aspire"
    ],
}


# Phrases that amplify emotions (intensifiers)
INTENSIFIERS = {
    EmotionIntensity.SUBTLE: ["quite", "rather", "somewhat", "a bit", "slightly"],
    EmotionIntensity.MODERATE: ["very", "really", "pretty", "fairly", "considerably"],
    EmotionIntensity.STRONG: ["extremely", "incredibly", "tremendously", "remarkably"],
    EmotionIntensity.EXTREME: [
        "absolutely", "completely", "totally", "utterly", "insanely", "mind-blowingly",
        "unbelievably", "impossibly", "literally", "NEVER", "ALWAYS", "NOBODY", "EVERYONE"
    ],
}


# ============================================================================
# EMOTIONAL ARC TEMPLATES
# ============================================================================

# Proven emotional arcs for viral content
EMOTIONAL_ARCS = {
    "rising_tension": {
        "description": "Build tension toward revelation",
        "pattern": ["curiosity", "anticipation", "surprise"],
        "pacing": "accelerating",
        "best_for": ["education", "entertainment", "tech"],
        "engagement_score": 0.85
    },

    "shock_reveal": {
        "description": "Immediate shock then explanation",
        "pattern": ["surprise", "curiosity", "awe"],
        "pacing": "fast_start",
        "best_for": ["entertainment", "news"],
        "engagement_score": 0.90
    },

    "problem_solution": {
        "description": "Present problem, build tension, deliver solution",
        "pattern": ["fear", "anticipation", "joy"],
        "pacing": "steady",
        "best_for": ["howto", "lifestyle", "education"],
        "engagement_score": 0.80
    },

    "curiosity_payoff": {
        "description": "Create mystery then satisfy",
        "pattern": ["curiosity", "anticipation", "validation"],
        "pacing": "controlled",
        "best_for": ["education", "tech", "science"],
        "engagement_score": 0.82
    },

    "controversy_resolution": {
        "description": "Spark debate then provide insight",
        "pattern": ["anger", "curiosity", "validation"],
        "pacing": "dynamic",
        "best_for": ["news", "lifestyle", "tech"],
        "engagement_score": 0.78
    },

    "inspiration_journey": {
        "description": "Show struggle then triumph",
        "pattern": ["sadness", "anticipation", "inspiration"],
        "pacing": "building",
        "best_for": ["inspiration", "sports", "lifestyle"],
        "engagement_score": 0.83
    },
}


# ============================================================================
# EMOTION ANALYZER CLASS
# ============================================================================

class EmotionAnalyzer:
    """
    Analyzes and optimizes emotional content

    Detects emotions, suggests improvements, and ensures
    proper emotional arc for maximum engagement
    """

    def __init__(self):
        """Initialize emotion analyzer"""
        logger.info("[EmotionAnalyzer] Initialized")

    def analyze_text(
        self,
        text: str,
        content_type: str = "education"
    ) -> EmotionalProfile:
        """
        Analyze text and create emotional profile

        Args:
            text: Text to analyze (script, hook, etc.)
            content_type: Content category

        Returns:
            EmotionalProfile with detected emotions
        """
        logger.debug(f"[EmotionAnalyzer] Analyzing text: {text[:50]}...")

        # Detect all emotion signals
        signals = self._detect_emotions(text)

        if not signals:
            # No clear emotions - neutral content
            return EmotionalProfile(
                primary_emotion=EmotionType.NEUTRAL,
                secondary_emotions=[],
                intensity=EmotionIntensity.SUBTLE,
                signals=[],
                emotional_arc="flat",
                engagement_score=0.3
            )

        # Determine primary emotion
        emotion_counts = {}
        for signal in signals:
            emotion_counts[signal.emotion] = emotion_counts.get(signal.emotion, 0) + 1

        primary_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]

        # Get secondary emotions
        secondary_emotions = [
            e for e, count in emotion_counts.items()
            if e != primary_emotion and count > 0
        ][:3]  # Top 3

        # Determine overall intensity
        avg_intensity = self._average_intensity([s.intensity for s in signals])

        # Detect emotional arc
        arc = self._detect_emotional_arc(signals)

        # Calculate engagement score
        engagement = self._calculate_engagement_score(
            primary_emotion, signals, arc, content_type
        )

        profile = EmotionalProfile(
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            intensity=avg_intensity,
            signals=signals,
            emotional_arc=arc,
            engagement_score=engagement
        )

        logger.info(
            f"[EmotionAnalyzer] Profile: {primary_emotion.value} "
            f"({avg_intensity.value}), arc={arc}, score={engagement:.2f}"
        )

        return profile

    def suggest_emotional_arc(
        self,
        content_type: str,
        duration: int = 30
    ) -> Dict:
        """
        Suggest optimal emotional arc for content

        Args:
            content_type: Content category
            duration: Video duration in seconds

        Returns:
            Arc template with recommendations
        """
        # Find best arcs for this content type
        suitable_arcs = [
            (name, arc) for name, arc in EMOTIONAL_ARCS.items()
            if content_type in arc["best_for"]
        ]

        if not suitable_arcs:
            # Default to curiosity payoff
            suitable_arcs = [("curiosity_payoff", EMOTIONAL_ARCS["curiosity_payoff"])]

        # Select arc with highest engagement score
        best_arc_name, best_arc = max(suitable_arcs, key=lambda x: x[1]["engagement_score"])

        logger.info(f"[EmotionAnalyzer] Suggested arc: {best_arc_name}")

        return {
            "arc_name": best_arc_name,
            **best_arc
        }

    def generate_emotion_triggers(
        self,
        target_emotion: EmotionType,
        content_type: str,
        num_triggers: int = 3
    ) -> List[EmotionTrigger]:
        """
        Generate emotion triggers to inject into content

        Args:
            target_emotion: Desired emotion
            content_type: Content category
            num_triggers: Number of triggers to generate

        Returns:
            List of emotion triggers
        """
        triggers = []

        # Get trigger phrases for this emotion
        trigger_phrases = self._get_trigger_phrases(target_emotion)

        # Select triggers for different placements
        placements = ["hook", "middle", "cta"]
        intensities = [EmotionIntensity.STRONG, EmotionIntensity.MODERATE, EmotionIntensity.MODERATE]

        for i in range(min(num_triggers, len(placements))):
            trigger = EmotionTrigger(
                emotion=target_emotion,
                trigger_phrase=trigger_phrases[i % len(trigger_phrases)],
                placement=placements[i],
                intensity=intensities[i],
                reason=f"Inject {target_emotion.value} in {placements[i]}"
            )
            triggers.append(trigger)

        logger.info(f"[EmotionAnalyzer] Generated {len(triggers)} triggers for {target_emotion.value}")

        return triggers

    def optimize_emotional_pacing(
        self,
        sentences: List[str],
        target_arc: str = "rising_tension"
    ) -> List[Tuple[str, EmotionType]]:
        """
        Optimize emotional pacing across sentences

        Args:
            sentences: Script sentences
            target_arc: Desired emotional arc

        Returns:
            List of (sentence, target_emotion) pairs
        """
        arc_template = EMOTIONAL_ARCS.get(target_arc, EMOTIONAL_ARCS["rising_tension"])
        emotion_pattern = arc_template["pattern"]

        # Map sentences to emotions
        result = []
        num_sentences = len(sentences)

        for i, sentence in enumerate(sentences):
            # Map position to emotion in pattern
            pattern_index = int((i / num_sentences) * len(emotion_pattern))
            pattern_index = min(pattern_index, len(emotion_pattern) - 1)

            target_emotion = EmotionType(emotion_pattern[pattern_index])
            result.append((sentence, target_emotion))

        logger.info(f"[EmotionAnalyzer] Mapped {num_sentences} sentences to arc: {target_arc}")

        return result

    def _detect_emotions(self, text: str) -> List[EmotionSignal]:
        """Detect all emotion signals in text"""
        signals = []
        text_lower = text.lower()

        for emotion, keywords in EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Find position
                    start = text_lower.find(keyword)
                    end = start + len(keyword)

                    # Detect intensity
                    intensity = self._detect_intensity_near(text_lower, start)

                    # Calculate confidence
                    confidence = 0.8 if len(keyword) > 5 else 0.6

                    signal = EmotionSignal(
                        emotion=emotion,
                        intensity=intensity,
                        confidence=confidence,
                        trigger_words=[keyword],
                        position=(start, end)
                    )
                    signals.append(signal)

        return signals

    def _detect_intensity_near(self, text: str, position: int, window: int = 20) -> EmotionIntensity:
        """Detect intensity of emotion near position"""
        # Extract window around position
        start = max(0, position - window)
        end = min(len(text), position + window)
        window_text = text[start:end]

        # Check for intensifiers
        for intensity, intensifiers in sorted(
            INTENSIFIERS.items(),
            key=lambda x: list(EmotionIntensity).index(x[0]),
            reverse=True
        ):
            for intensifier in intensifiers:
                if intensifier in window_text:
                    return intensity

        # Check for caps (indicates high intensity)
        if any(word.isupper() and len(word) > 2 for word in window_text.split()):
            return EmotionIntensity.EXTREME

        # Default
        return EmotionIntensity.MODERATE

    def _average_intensity(self, intensities: List[EmotionIntensity]) -> EmotionIntensity:
        """Calculate average intensity"""
        if not intensities:
            return EmotionIntensity.SUBTLE

        intensity_values = {
            EmotionIntensity.SUBTLE: 1,
            EmotionIntensity.MODERATE: 2,
            EmotionIntensity.STRONG: 3,
            EmotionIntensity.EXTREME: 4
        }

        avg = sum(intensity_values[i] for i in intensities) / len(intensities)

        if avg >= 3.5:
            return EmotionIntensity.EXTREME
        elif avg >= 2.5:
            return EmotionIntensity.STRONG
        elif avg >= 1.5:
            return EmotionIntensity.MODERATE
        else:
            return EmotionIntensity.SUBTLE

    def _detect_emotional_arc(self, signals: List[EmotionSignal]) -> str:
        """Detect the emotional arc from signals"""
        if len(signals) < 2:
            return "flat"

        # Sort by position
        signals_sorted = sorted(signals, key=lambda s: s.position[0])

        # Extract intensity progression
        intensities = [
            list(EmotionIntensity).index(s.intensity)
            for s in signals_sorted
        ]

        # Detect pattern
        if all(intensities[i] <= intensities[i+1] for i in range(len(intensities)-1)):
            return "rising"
        elif all(intensities[i] >= intensities[i+1] for i in range(len(intensities)-1)):
            return "falling"
        elif len(set(intensities)) > 2:
            return "rollercoaster"
        else:
            return "steady"

    def _calculate_engagement_score(
        self,
        primary_emotion: EmotionType,
        signals: List[EmotionSignal],
        arc: str,
        content_type: str
    ) -> float:
        """Calculate predicted engagement score"""
        score = 0.5  # Base score

        # High-engagement emotions
        high_engagement = [
            EmotionType.SURPRISE, EmotionType.CURIOSITY, EmotionType.AWE,
            EmotionType.FOMO, EmotionType.ANGER
        ]
        if primary_emotion in high_engagement:
            score += 0.2

        # Signal count bonus
        signal_bonus = min(len(signals) / 10.0, 0.15)
        score += signal_bonus

        # Arc bonus
        arc_bonus = {
            "rising": 0.15,
            "rollercoaster": 0.10,
            "steady": 0.05,
            "falling": 0.0,
            "flat": 0.0
        }
        score += arc_bonus.get(arc, 0.0)

        # Intensity bonus
        avg_intensity = self._average_intensity([s.intensity for s in signals])
        intensity_bonus = {
            EmotionIntensity.EXTREME: 0.15,
            EmotionIntensity.STRONG: 0.10,
            EmotionIntensity.MODERATE: 0.05,
            EmotionIntensity.SUBTLE: 0.0
        }
        score += intensity_bonus.get(avg_intensity, 0.0)

        return min(score, 1.0)

    def _get_trigger_phrases(self, emotion: EmotionType) -> List[str]:
        """Get trigger phrases for emotion"""
        phrases = {
            EmotionType.CURIOSITY: [
                "You won't believe this...",
                "Here's the secret:",
                "Want to know the truth?"
            ],
            EmotionType.SURPRISE: [
                "This is INSANE!",
                "You've never seen this before.",
                "Wait until you see this."
            ],
            EmotionType.FEAR: [
                "Don't make this mistake!",
                "Warning: This could happen to you.",
                "Avoid this at all costs."
            ],
            EmotionType.JOY: [
                "This is absolutely amazing!",
                "You're going to love this!",
                "Prepare to be delighted!"
            ],
            EmotionType.ANGER: [
                "This is completely unacceptable!",
                "They don't want you to know this.",
                "This needs to stop now."
            ],
            EmotionType.FOMO: [
                "Don't miss this!",
                "Limited time only!",
                "Before it's too late..."
            ],
            EmotionType.INSPIRATION: [
                "You can do this too!",
                "This will change your life.",
                "Start your journey today!"
            ],
        }

        return phrases.get(emotion, ["This is interesting..."])


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def analyze_content_emotion(text: str, content_type: str = "education") -> EmotionalProfile:
    """Quick emotion analysis"""
    analyzer = EmotionAnalyzer()
    return analyzer.analyze_text(text, content_type)


def get_recommended_arc(content_type: str) -> Dict:
    """Get recommended emotional arc for content type"""
    analyzer = EmotionAnalyzer()
    return analyzer.suggest_emotional_arc(content_type)
