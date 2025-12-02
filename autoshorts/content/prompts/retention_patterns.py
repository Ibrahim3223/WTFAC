# -*- coding: utf-8 -*-
"""
Retention Patterns for YouTube Shorts
Injects mini-cliffhangers to maintain viewer engagement

Research shows:
- Viewers drop off every 10-15 seconds without hook
- Mini-cliffhangers increase retention by 15-25%
- Optimal placement: Every 10 seconds in Shorts
"""
import random
import logging
from typing import List

logger = logging.getLogger(__name__)

# Shorts-optimized cliffhangers (max 5 words, instant impact)
SHORTS_CLIFFHANGERS = [
    "But wait...",
    "Here's the twist.",
    "You won't believe this.",
    "Watch what happens.",
    "But that's not all.",
    "The shocking part?",
    "Here's where it gets crazy.",
    "And then this happened.",
    "Wait for it...",
    "But here's the kicker.",
    "The truth?",
    "Get this.",
    "Now watch closely.",
]


def inject_cliffhangers(
    sentences: List[str],
    target_duration: int = 30,
    interval: int = 10,
    max_cliffhangers: int = 3
) -> List[str]:
    """
    Inject mini-cliffhangers into script for retention.

    Args:
        sentences: List of sentence texts
        target_duration: Video duration in seconds (default: 30)
        interval: Cliffhanger interval in seconds (default: 10)
        max_cliffhangers: Maximum cliffhangers to inject (default: 3)

    Returns:
        List with cliffhangers injected at optimal points

    Example:
        >>> sentences = ["Sentence 1.", "Sentence 2.", "Sentence 3.", ...]
        >>> inject_cliffhangers(sentences, target_duration=30)
        ["Sentence 1.", "But wait...", "Sentence 2.", ...]

    Algorithm:
        - Assumes ~3 seconds per sentence
        - Injects every N sentences (where N*3 ≈ interval)
        - Avoids injecting near the end (last 2 sentences)
        - Respects max_cliffhangers limit
    """
    if len(sentences) < 3:
        logger.debug("Script too short for cliffhangers (< 3 sentences)")
        return sentences  # Too short for cliffhangers

    # Calculate sentences per interval (assume ~3 seconds per sentence)
    seconds_per_sentence = 3
    sentences_per_interval = max(1, interval // seconds_per_sentence)

    # Calculate actual max based on duration
    calculated_max = (target_duration // interval) - 1  # Don't add at end
    actual_max = min(max_cliffhangers, calculated_max)

    logger.info(
        f"Injecting up to {actual_max} cliffhangers "
        f"(every {sentences_per_interval} sentences)"
    )

    result = []
    cliffhanger_count = 0

    for i, sentence in enumerate(sentences):
        result.append(sentence)

        # Check if we should inject cliffhanger
        should_inject = (
            (i + 1) % sentences_per_interval == 0 and  # At interval
            i < len(sentences) - 2 and  # Not near end
            cliffhanger_count < actual_max  # Haven't hit max
        )

        if should_inject:
            cliffhanger = random.choice(SHORTS_CLIFFHANGERS)
            result.append(cliffhanger)
            cliffhanger_count += 1
            logger.debug(f"Injected cliffhanger at position {i+1}: '{cliffhanger}'")

    logger.info(f"Total cliffhangers injected: {cliffhanger_count}")
    return result


def get_random_cliffhanger() -> str:
    """
    Get a random cliffhanger phrase.

    Returns:
        Random cliffhanger string

    Example:
        >>> get_random_cliffhanger()
        "But wait..."
    """
    return random.choice(SHORTS_CLIFFHANGERS)


def get_cliffhanger_for_context(context: str = "general") -> str:
    """
    Get context-appropriate cliffhanger.

    Args:
        context: Content context
            - "reveal": Before revealing information
            - "transition": Between topics
            - "surprise": Before surprising fact
            - "general": Default

    Returns:
        Context-appropriate cliffhanger

    Example:
        >>> get_cliffhanger_for_context("reveal")
        "The truth?"
    """
    context_cliffhangers = {
        "reveal": [
            "The truth?",
            "Here's the shocking part.",
            "Get this.",
        ],
        "transition": [
            "But wait...",
            "Now watch closely.",
            "But that's not all.",
        ],
        "surprise": [
            "You won't believe this.",
            "Here's the twist.",
            "And then this happened.",
        ],
    }

    cliffhangers = context_cliffhangers.get(context, SHORTS_CLIFFHANGERS)
    return random.choice(cliffhangers)


def validate_cliffhanger_placement(
    sentences_with_cliffhangers: List[str],
    original_count: int
) -> bool:
    """
    Validate that cliffhangers are placed correctly.

    Args:
        sentences_with_cliffhangers: Script with cliffhangers injected
        original_count: Original sentence count before injection

    Returns:
        True if placement is valid, False otherwise

    Checks:
        - Cliffhangers not in first position
        - Cliffhangers not in last position
        - No consecutive cliffhangers
    """
    if not sentences_with_cliffhangers:
        return False

    # Check first and last positions
    first = sentences_with_cliffhangers[0]
    last = sentences_with_cliffhangers[-1]

    if first in SHORTS_CLIFFHANGERS or last in SHORTS_CLIFFHANGERS:
        logger.warning("Cliffhanger in first or last position")
        return False

    # Check for consecutive cliffhangers
    for i in range(len(sentences_with_cliffhangers) - 1):
        current = sentences_with_cliffhangers[i]
        next_item = sentences_with_cliffhangers[i + 1]

        if current in SHORTS_CLIFFHANGERS and next_item in SHORTS_CLIFFHANGERS:
            logger.warning("Consecutive cliffhangers detected")
            return False

    return True


# Test function
def _test_cliffhangers():
    """Test cliffhanger injection."""
    print("=" * 60)
    print("RETENTION PATTERN TESTS")
    print("=" * 60)

    sentences = [
        "This tiger is the fastest in the world.",
        "It can run 80 miles per hour.",
        "Scientists studied this for years.",
        "They discovered something shocking.",
        "The tiger uses a special technique.",
        "This changed everything we knew.",
        "The secret is in their muscles.",
        "No other animal has this ability.",
    ]

    print(f"\n✅ Original: {len(sentences)} sentences\n")
    for i, s in enumerate(sentences):
        print(f"  {i+1}. {s}")

    # Test with different durations
    for duration in [30, 45, 60]:
        result = inject_cliffhangers(sentences, target_duration=duration)
        cliffhanger_count = len(result) - len(sentences)

        print(f"\n✅ {duration}s video: {len(result)} items ({cliffhanger_count} cliffhangers)\n")

        for i, s in enumerate(result):
            marker = "⚡" if s in SHORTS_CLIFFHANGERS else "📝"
            print(f"  {marker} {i+1}. {s}")

    # Test validation
    print("\n✅ Validation test:")
    result = inject_cliffhangers(sentences, target_duration=30)
    is_valid = validate_cliffhanger_placement(result, len(sentences))
    print(f"  Valid placement: {is_valid}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    _test_cliffhangers()
