# -*- coding: utf-8 -*-
"""
Viral Hook Patterns for YouTube Shorts
Optimized for 30-60 second videos with maximum CTR impact

Based on analysis of viral Shorts:
- First 3 seconds are CRITICAL
- Cold opens (no meta-talk) perform 40% better
- Pattern-based hooks increase CTR by 20-30%
"""
import random
from typing import List, Dict

# Shorts-optimized hooks (max 10 words, 3-4 seconds speaking time)
SHORTS_HOOK_PATTERNS = {
    "extreme": [
        "This {entity} is impossible.",
        "Nobody expected this {outcome}.",
        "{number} people can't explain this.",
        "Everything about {topic} is wrong.",
        "This breaks every rule.",
        "{entity} did the unthinkable.",
        "Scientists can't explain this {thing}.",
        "This {entity} shouldn't exist.",
        "The truth about {topic} is shocking.",
        "{number} {people} discovered this secret.",
    ],

    "high": [
        "This {thing} broke all records.",
        "{entity} shocked the world.",
        "The truth about {topic}.",
        "This shouldn't be possible.",
        "{entity} has a hidden power.",
        "The secret of {topic}.",
        "{number} people missed this detail.",
        "This {thing} changes everything.",
        "{entity} defied all odds.",
        "Watch what happens to this {thing}.",
    ],

    "medium": [
        "Here's what makes {entity} special.",
        "The secret behind {topic}.",
        "{entity} has an unusual ability.",
        "This changes how we see {topic}.",
        "The story of {entity} is fascinating.",
        "{number} facts about {topic}.",
        "This {thing} is remarkable.",
        "The mystery of {topic}.",
    ],

    "low": [
        "Let's explore {topic}.",
        "{entity} is unique for one reason.",
        "There's something special about {topic}.",
        "The science of {topic}.",
        "Discover the truth about {entity}.",
    ],
}

# Words to AVOID in cold opens (kills retention immediately)
# These violate the "cold open" principle and signal "generic content"
COLD_OPEN_VIOLATIONS = [
    "this video",
    "this short",
    "today we",
    "in this video",
    "in this short",
    "let me show",
    "let me tell",
    "welcome to",
    "hey guys",
    "hey everyone",
    "what's up",
    "before we start",
    "make sure to subscribe",
    "don't forget to like",
    "hit the bell",
    "if you enjoy",
]


def get_shorts_hook(intensity: str = "extreme") -> str:
    """
    Get a random viral hook pattern for Shorts.

    Args:
        intensity: Hook intensity level
            - "extreme": Maximum engagement (recommended for Shorts)
            - "high": Strong engagement
            - "medium": Moderate engagement
            - "low": Soft engagement

    Returns:
        Hook pattern string with placeholders like {entity}, {topic}

    Example:
        >>> get_shorts_hook("extreme")
        "This {entity} is impossible."
    """
    patterns = SHORTS_HOOK_PATTERNS.get(intensity, SHORTS_HOOK_PATTERNS["extreme"])
    return random.choice(patterns)


def validate_cold_open(text: str) -> bool:
    """
    Check if text violates cold open rules.

    Cold open = Starting directly with content, no meta-talk.
    Videos with cold opens have 40% higher retention.

    Args:
        text: Text to validate (usually first sentence)

    Returns:
        True if valid cold open, False if violation detected

    Example:
        >>> validate_cold_open("This tiger did the impossible.")
        True
        >>> validate_cold_open("In this video, we explore tigers.")
        False
    """
    text_lower = text.lower()
    return not any(violation in text_lower for violation in COLD_OPEN_VIOLATIONS)


def get_all_violations(text: str) -> List[str]:
    """
    Get list of all cold open violations in text.

    Useful for debugging why a script was rejected.

    Args:
        text: Text to check

    Returns:
        List of violation phrases found in text

    Example:
        >>> get_all_violations("In this video today we explore...")
        ["in this video", "today we"]
    """
    text_lower = text.lower()
    return [v for v in COLD_OPEN_VIOLATIONS if v in text_lower]


def get_hook_guidelines() -> Dict[str, List[str]]:
    """
    Get hook writing guidelines for Shorts.

    Returns:
        Dictionary with DO's and DON'Ts

    Example:
        >>> guidelines = get_hook_guidelines()
        >>> print(guidelines["do"][0])
        "Start with entity name or number"
    """
    return {
        "do": [
            "Start with entity name or number",
            "Use power words (impossible, shocking, secret)",
            "Create curiosity gap",
            "Keep under 10 words",
            "Start strong, no buildup",
            "Use specific details over generic claims",
        ],
        "dont": [
            "Say 'this video' or 'this short'",
            "Use 'today we' or 'let me show'",
            "Start with greetings",
            "Ask for likes/subscribes upfront",
            "Use meta-talk about the video itself",
            "Start with slow buildup",
        ],
    }


def get_pattern_explanation(pattern: str) -> str:
    """
    Get explanation of why a hook pattern works.

    Args:
        pattern: Hook pattern string

    Returns:
        Explanation string

    Example:
        >>> get_pattern_explanation("This {entity} is impossible.")
        "Creates immediate curiosity by claiming impossibility..."
    """
    explanations = {
        "This {entity} is impossible.": (
            "Creates immediate curiosity by claiming impossibility. "
            "Viewer wants to see what defies expectations."
        ),
        "Nobody expected this {outcome}.": (
            "Implies unexpected twist or surprise. "
            "Creates 'fear of missing out' on shocking reveal."
        ),
        "{number} people can't explain this.": (
            "Uses social proof and mystery. "
            "If many people are puzzled, it must be worth watching."
        ),
        "Everything about {topic} is wrong.": (
            "Challenges existing beliefs. "
            "Viewer wants to know what they got wrong."
        ),
    }

    for key, explanation in explanations.items():
        if pattern.startswith(key.split("{")[0]):
            return explanation

    return "Creates engagement through curiosity and specificity."


# Test function
def _test_hooks():
    """Test hook pattern generation and validation."""
    print("=" * 60)
    print("HOOK PATTERN TESTS")
    print("=" * 60)

    # Test pattern generation
    print("\n✅ Generated hooks:")
    for intensity in ["extreme", "high", "medium"]:
        hook = get_shorts_hook(intensity)
        print(f"  [{intensity.upper():8}] {hook}")

    # Test cold open validation
    print("\n✅ Cold open validation:")

    good_hooks = [
        "This tiger did the impossible.",
        "Nobody expected this outcome.",
        "5 million people discovered this secret.",
    ]

    bad_hooks = [
        "In this video, we explore tigers.",
        "Today we'll learn about space.",
        "Welcome to this short about science.",
    ]

    for hook in good_hooks:
        result = "✓" if validate_cold_open(hook) else "✗"
        print(f"  {result} GOOD | {hook}")

    for hook in bad_hooks:
        result = "✓" if validate_cold_open(hook) else "✗"
        violations = get_all_violations(hook)
        print(f"  {result} BAD  | {hook}")
        if violations:
            print(f"         └─ Violations: {', '.join(violations)}")

    # Test guidelines
    print("\n✅ Hook Guidelines:")
    guidelines = get_hook_guidelines()
    print("  DO:")
    for guideline in guidelines["do"][:3]:
        print(f"    • {guideline}")
    print("  DON'T:")
    for guideline in guidelines["dont"][:3]:
        print(f"    • {guideline}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    _test_hooks()
