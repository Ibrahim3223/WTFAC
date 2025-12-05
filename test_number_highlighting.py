# -*- coding: utf-8 -*-
"""Test number highlighting with various formats."""

import sys
sys.path.insert(0, '.')

from autoshorts.captions.keyword_highlighter import ShortsKeywordHighlighter

def test_numbers():
    """Test various number formats."""
    highlighter = ShortsKeywordHighlighter()

    test_cases = [
        "3-MINUTE POWER",
        "GOLDEN RATIO (J)",
        "5-STAR RATING",
        "The number is 100",
        "100-YEAR JOURNEY",
        "Top 10 Facts",
        "Part (1) Reveals",
        "3.14 is PI",
        "1,000,000 views",
        "24-HOUR Challenge",
    ]

    print("="*60)
    print("NUMBER HIGHLIGHTING TEST")
    print("="*60)
    print("\nTest Cases:\n")

    for text in test_cases:
        highlighted = highlighter.highlight(text)

        # Check if numbers are in the output
        has_highlight = '{\\c&H00FFFF&' in highlighted

        print(f"Input:     {text}")
        print(f"Output:    {highlighted}")
        print(f"Status:    {'[OK]' if has_highlight else '[FAIL - No highlight]'}")
        print()

    print("="*60)

if __name__ == "__main__":
    test_numbers()
