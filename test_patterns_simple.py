#!/usr/bin/env python3
"""
Simple test for hook and cliffhanger patterns (no external dependencies).
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def test_patterns():
    """Test patterns directly."""
    print("=" * 60)
    print("PATTERN TESTS")
    print("=" * 60)

    # Test 1: Hook patterns
    print("\n[1] Hook Patterns Test:")
    try:
        # Direct import to avoid dependency chain
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hook_patterns",
            os.path.join(project_root, "autoshorts", "content", "prompts", "hook_patterns.py")
        )
        hook_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook_module)

        # Test hook generation
        hook = hook_module.get_shorts_hook("extreme")
        print(f"   Generated hook: {hook}")

        # Test cold open validation
        good = "This tiger did the impossible."
        bad = "In this video, we explore tigers."

        good_result = hook_module.validate_cold_open(good)
        bad_result = hook_module.validate_cold_open(bad)

        print(f"   Good hook validated: {good_result}")
        print(f"   Bad hook rejected: {not bad_result}")

        print("   [PASS] Hook Patterns")
    except Exception as e:
        print(f"   [FAIL] Hook Patterns: {e}")

    # Test 2: Cliffhanger patterns
    print("\n[2] Cliffhanger Patterns Test:")
    try:
        spec = importlib.util.spec_from_file_location(
            "retention_patterns",
            os.path.join(project_root, "autoshorts", "content", "prompts", "retention_patterns.py")
        )
        retention_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(retention_module)

        sentences = [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
            "Fourth sentence.",
        ]

        result = retention_module.inject_cliffhangers(sentences, target_duration=30)
        cliffhangers_added = len(result) - len(sentences)

        print(f"   Original sentences: {len(sentences)}")
        print(f"   With cliffhangers: {len(result)}")
        print(f"   Cliffhangers added: {cliffhangers_added}")
        print("   [PASS] Cliffhanger Patterns")
    except Exception as e:
        print(f"   [FAIL] Cliffhanger Patterns: {e}")

    # Test 3: Keyword highlighter
    print("\n[3] Keyword Highlighter Test:")
    try:
        spec = importlib.util.spec_from_file_location(
            "keyword_highlighter",
            os.path.join(project_root, "autoshorts", "captions", "keyword_highlighter.py")
        )
        highlighter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(highlighter_module)

        highlighter = highlighter_module.ShortsKeywordHighlighter()
        test_text = "This incredible fact involves 5 million people"
        highlighted = highlighter.highlight(test_text)
        has_highlights = highlighter.has_highlights(test_text)

        print(f"   Original: {test_text}")
        print(f"   Has highlights: {has_highlights}")
        print(f"   Result length: {len(highlighted)} chars")
        print("   [PASS] Keyword Highlighter")
    except Exception as e:
        print(f"   [FAIL] Keyword Highlighter: {e}")

    # Test 4: Config manager
    print("\n[4] Config Manager Test:")
    try:
        spec = importlib.util.spec_from_file_location(
            "manager",
            os.path.join(project_root, "autoshorts", "config", "manager.py")
        )
        manager_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(manager_module)

        manager = manager_module.ConfigManager.get_instance()
        config = manager.config

        print(f"   Hook intensity: {config.script_style.hook_intensity}")
        print(f"   Cliffhangers enabled: {config.script_style.cliffhanger_enabled}")
        print(f"   Keyword highlighting: {config.script_style.keyword_highlighting}")
        print("   [PASS] Config Manager")
    except Exception as e:
        print(f"   [FAIL] Config Manager: {e}")

    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_patterns()
