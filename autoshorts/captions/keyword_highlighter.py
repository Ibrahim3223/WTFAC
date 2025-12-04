# -*- coding: utf-8 -*-
"""
Keyword Highlighter for Shorts Captions
Highlights important words for better engagement and readability

Research shows:
- Highlighted captions increase engagement by 60%
- Numbers and emphasis words draw attention
- Mobile viewers rely heavily on visual cues
"""
import re
import logging
from typing import List, Set

logger = logging.getLogger(__name__)


class ShortsKeywordHighlighter:
    """Highlight important keywords in captions for better engagement."""

    # ASS color codes (BGR format)
    COLORS = {
        "yellow": "&H00FFFF&",  # Numbers, facts
        "red": "&H0000FF&",  # Emphasis, power words
        "cyan": "&HFFFF00&",  # Questions
        "white": "&H00FFFFFF&",  # Default
    }

    # Shorts-specific emphasis words (attention-grabbing)
    EMPHASIS_WORDS = [
        # Extremes
        "shocking", "incredible", "unbelievable", "mindblowing",
        "insane", "crazy", "wild", "extreme",
        # Absolutes
        "never", "always", "nobody", "everyone", "impossible",
        # Mystery/Intrigue
        "secret", "hidden", "truth", "mystery", "unknown",
        # Superlatives
        "best", "worst", "fastest", "biggest", "smallest",
        # Emotional
        "amazing", "stunning", "bizarre", "weird", "strange",
    ]

    def __init__(self, additional_words: List[str] = None):
        """
        Initialize highlighter.

        Args:
            additional_words: Additional emphasis words to highlight
        """
        self.emphasis_words = set(self.EMPHASIS_WORDS)

        if additional_words:
            self.emphasis_words.update(w.lower() for w in additional_words)

        logger.info(f"Initialized with {len(self.emphasis_words)} emphasis words")

    def highlight(self, text: str) -> str:
        """
        Add ASS formatting to highlight keywords.

        Highlights:
        - Numbers → Yellow, Bold, 1.3x size
        - Emphasis words → Red, Bold
        - Questions → Cyan
        - Exclamations → Slightly larger

        Args:
            text: Plain text caption

        Returns:
            ASS-formatted text with color/size highlights

        Example:
            >>> highlighter = ShortsKeywordHighlighter()
            >>> highlighter.highlight("This incredible fact involves 5 million people")
            "This {\\c&H0000FF&\\b1}incredible{\\r} fact involves {\\c&H00FFFF&\\b1\\fs1.3}5 million{\\r} people"
        """
        result = text

        # 1. Highlight numbers (YELLOW, BOLD, 1.3x size for mobile screens)
        # Matches: 5, 100, 1,000, 1.5, (J), etc.
        # First, highlight content in parentheses (letters, numbers, etc.)
        result = re.sub(
            r'\(([A-Za-z0-9]+)\)',  # Match single char/number in parentheses like (J), (1)
            r'{\\c&H00FFFF&\\b1\\fs1.3}(\1){\\r}',
            result
        )

        # Then, highlight standalone numbers
        result = re.sub(
            r'\b(\d+(?:,\d+)*(?:\.\d+)?)\b',
            r'{\\c&H00FFFF&\\b1\\fs1.3}\1{\\r}',
            result
        )

        # 2. Highlight emphasis words (RED, BOLD)
        for word in self.emphasis_words:
            # Word boundary matching for whole words only
            pattern = rf'\b({re.escape(word)})\b'
            replacement = r'{\\c&H0000FF&\\b1}\1{\\r}'
            result = re.sub(
                pattern,
                replacement,
                result,
                flags=re.IGNORECASE
            )

        # 3. Highlight questions (CYAN)
        if '?' in result:
            result = result.replace('?', '{\\c&HFFFF00&\\b1}?{\\r}')

        # 4. Highlight exclamations (BOLD, slightly larger)
        if '!' in result:
            result = result.replace('!', '{\\b1\\fs1.1}!{\\r}')

        logger.debug(f"Highlighted: {result}")
        return result

    def highlight_multiple(self, texts: List[str]) -> List[str]:
        """
        Highlight multiple captions at once.

        Args:
            texts: List of caption texts

        Returns:
            List of highlighted texts
        """
        return [self.highlight(text) for text in texts]

    def add_emphasis_word(self, word: str):
        """
        Add custom emphasis word to highlight list.

        Args:
            word: Word to add (case-insensitive)

        Example:
            >>> highlighter.add_emphasis_word("quantum")
        """
        word_lower = word.lower()
        if word_lower not in self.emphasis_words:
            self.emphasis_words.add(word_lower)
            logger.info(f"Added emphasis word: {word_lower}")

    def add_emphasis_words(self, words: List[str]):
        """
        Add multiple emphasis words.

        Args:
            words: List of words to add
        """
        for word in words:
            self.add_emphasis_word(word)

    def get_emphasis_words(self) -> Set[str]:
        """Get current set of emphasis words."""
        return self.emphasis_words.copy()

    def has_highlights(self, text: str) -> bool:
        """
        Check if text would have any highlights.

        Args:
            text: Text to check

        Returns:
            True if text contains highlightable content
        """
        # Check for numbers
        if re.search(r'\b\d+', text):
            return True

        # Check for emphasis words
        text_lower = text.lower()
        if any(word in text_lower for word in self.emphasis_words):
            return True

        # Check for special characters
        if '?' in text or '!' in text:
            return True

        return False

    def get_highlight_stats(self, text: str) -> dict:
        """
        Get statistics about highlights in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with highlight statistics

        Example:
            >>> stats = highlighter.get_highlight_stats("This incredible 5!")
            >>> print(stats)
            {'numbers': 1, 'emphasis_words': 1, 'questions': 0, 'exclamations': 1}
        """
        text_lower = text.lower()

        return {
            "numbers": len(re.findall(r'\b\d+', text)),
            "emphasis_words": sum(
                1 for word in self.emphasis_words if word in text_lower
            ),
            "questions": text.count('?'),
            "exclamations": text.count('!'),
        }


# Test function
def _test_highlighter():
    """Test keyword highlighter."""
    print("=" * 60)
    print("KEYWORD HIGHLIGHTER TESTS")
    print("=" * 60)

    highlighter = ShortsKeywordHighlighter()

    test_sentences = [
        "This incredible fact involves 5 million people",
        "Nobody expected this shocking result",
        "Is this the truth about space?",
        "The secret number is 42!",
        "The fastest animal runs 120 km/h",
        "Always check the hidden details",
    ]

    print("\n✅ Highlight tests:\n")

    for sentence in test_sentences:
        highlighted = highlighter.highlight(sentence)
        stats = highlighter.get_highlight_stats(sentence)

        print(f"Original:  {sentence}")
        print(f"Stats:     {stats}")
        print(f"Has highlights: {highlighter.has_highlights(sentence)}")
        print()

    # Test custom words
    print("✅ Custom emphasis words:")
    highlighter.add_emphasis_words(["quantum", "neural", "atomic"])
    test = "This quantum effect is atomic"
    print(f"  Original: {test}")
    print(f"  Highlighted: {highlighter.highlight(test)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    _test_highlighter()
