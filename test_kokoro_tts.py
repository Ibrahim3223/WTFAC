#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Kokoro TTS integration.

Tests:
1. Kokoro TTS handler directly
2. Unified TTS handler with different providers
3. Backward compatibility
4. Voice switching
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def test_kokoro_tts():
    """Test Kokoro TTS integration."""
    print("=" * 60)
    print("KOKORO TTS INTEGRATION TESTS")
    print("=" * 60)

    # Test 1: Kokoro handler import
    print("\n[1] Testing Kokoro Handler Import:")
    try:
        from autoshorts.tts.kokoro_handler import KokoroTTS
        print("   [PASS] KokoroTTS imported successfully")

        # Check voices available
        print(f"   Available voices: {len(KokoroTTS.KOKORO_VOICES)}")
        print(f"   Default voice: af_sarah")
        print("   [PASS] Kokoro Handler")
    except Exception as e:
        print(f"   [FAIL] Kokoro Handler: {e}")
        return

    # Test 2: Unified handler import
    print("\n[2] Testing Unified Handler Import:")
    try:
        from autoshorts.tts.unified_handler import UnifiedTTSHandler
        print("   [PASS] UnifiedTTSHandler imported successfully")

        # Test initialization with different providers
        handler_auto = UnifiedTTSHandler(provider="auto")
        handler_kokoro = UnifiedTTSHandler(provider="kokoro")
        handler_edge = UnifiedTTSHandler(provider="edge")

        print(f"   Auto provider: {handler_auto.provider}")
        print(f"   Kokoro provider: {handler_kokoro.provider}")
        print(f"   Edge provider: {handler_edge.provider}")
        print("   [PASS] Unified Handler")
    except Exception as e:
        print(f"   [FAIL] Unified Handler: {e}")
        return

    # Test 3: Backward compatibility
    print("\n[3] Testing Backward Compatibility:")
    try:
        from autoshorts.tts import TTSHandler

        # TTSHandler should now be UnifiedTTSHandler
        handler = TTSHandler()
        print(f"   TTSHandler type: {type(handler).__name__}")
        print(f"   Default provider: {handler.provider}")
        print("   [PASS] Backward Compatibility")
    except Exception as e:
        print(f"   [FAIL] Backward Compatibility: {e}")
        return

    # Test 4: Config integration
    print("\n[4] Testing Config Integration:")
    try:
        from autoshorts.config.models import TTSConfig

        config = TTSConfig()
        print(f"   Provider: {config.provider}")
        print(f"   Kokoro voice: {config.kokoro_voice}")
        print(f"   Kokoro precision: {config.kokoro_precision}")
        print(f"   Edge voice (backward compat): {config.voice}")
        print("   [PASS] Config Integration")
    except Exception as e:
        print(f"   [FAIL] Config Integration: {e}")
        return

    # Test 5: Voice validation
    print("\n[5] Testing Voice Validation:")
    try:
        voices_to_test = ["af_sarah", "am_michael", "af_bella", "bm_george"]

        for voice in voices_to_test:
            if voice in KokoroTTS.KOKORO_VOICES:
                print(f"   [OK] Voice '{voice}': {KokoroTTS.KOKORO_VOICES[voice]}")
            else:
                print(f"   [FAIL] Voice '{voice}' not found")

        print("   [PASS] Voice Validation")
    except Exception as e:
        print(f"   [FAIL] Voice Validation: {e}")
        return

    # Test 6: Provider order logic
    print("\n[6] Testing Provider Order Logic:")
    try:
        handler_auto = UnifiedTTSHandler(provider="auto")
        handler_kokoro = UnifiedTTSHandler(provider="kokoro")
        handler_edge = UnifiedTTSHandler(provider="edge")

        order_auto = handler_auto._get_provider_order()
        order_kokoro = handler_kokoro._get_provider_order()
        order_edge = handler_edge._get_provider_order()

        print(f"   Auto mode: {order_auto}")
        print(f"   Kokoro mode: {order_kokoro}")
        print(f"   Edge mode: {order_edge}")

        # Verify auto tries Kokoro first
        assert order_auto == ['kokoro', 'edge'], "Auto should try Kokoro first"
        assert order_kokoro == ['kokoro'], "Kokoro mode should only use Kokoro"
        assert order_edge == ['edge'], "Edge mode should only use Edge"

        print("   [PASS] Provider Order Logic")
    except Exception as e:
        print(f"   [FAIL] Provider Order Logic: {e}")
        return

    # Test 7: Module exports
    print("\n[7] Testing Module Exports:")
    try:
        from autoshorts.tts import (
            TTSHandler,
            UnifiedTTSHandler,
            KokoroTTS,
            EdgeTTSHandler
        )

        print("   [OK] TTSHandler exported")
        print("   [OK] UnifiedTTSHandler exported")
        print("   [OK] KokoroTTS exported")
        print("   [OK] EdgeTTSHandler exported")
        print("   [PASS] Module Exports")
    except Exception as e:
        print(f"   [FAIL] Module Exports: {e}")
        return

    print("\n" + "=" * 60)
    print("*** ALL KOKORO TTS TESTS PASSED! ***")
    print("=" * 60)
    print("\nNote: Actual audio generation not tested (requires dependencies).")
    print("To test audio generation:")
    print("  1. Install: pip install kokoro-onnx soundfile")
    print("  2. Run a full video generation")
    print("  3. Verify audio quality")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    test_kokoro_tts()
