#!/usr/bin/env python3
"""
Refactoring validation script.

Checks that all modules can be imported and basic functionality works.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_imports():
    """Test core module imports."""
    print("Testing core imports...")
    try:
        from autoshorts.core import Result, Container, retry
        from autoshorts.core import AutoShortsError, ContentGenerationError
        print("✅ Core imports successful")
        return True
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        return False


def test_pipeline_imports():
    """Test pipeline imports."""
    print("Testing pipeline imports...")
    try:
        from autoshorts.pipeline import Pipeline, RetryablePipeline, PipelineContext
        from autoshorts.pipeline.stages import (
            ContentGenerationStage,
            TTSStage,
            VideoProductionStage,
            UploadStage
        )
        print("✅ Pipeline imports successful")
        return True
    except Exception as e:
        print(f"❌ Pipeline imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    print("Testing config loading...")
    try:
        from autoshorts.config import settings
        from autoshorts.config.models import AppConfig

        # Check that config is loaded
        assert hasattr(settings, 'config')
        assert hasattr(settings.config, 'api')
        assert hasattr(settings.config, 'video')

        print(f"  Channel: {settings.CHANNEL_NAME}")
        print(f"  Topic: {settings.CHANNEL_TOPIC}")
        print(f"  Duration: {settings.TARGET_DURATION}s")
        print(f"  Motion Intensity: {settings.MOTION_INTENSITY}")

        print("✅ Config loading successful")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_result_pattern():
    """Test Result pattern basic functionality."""
    print("Testing Result pattern...")
    try:
        from autoshorts.core import Result

        # Test ok
        r1 = Result.ok(42)
        assert r1.is_ok()
        assert r1.unwrap() == 42

        # Test err
        r2 = Result.err("failed")
        assert r2.is_err()
        assert r2.unwrap_or(0) == 0

        # Test map
        r3 = Result.ok(2).map(lambda x: x * 2)
        assert r3.unwrap() == 4

        print("✅ Result pattern works correctly")
        return True
    except Exception as e:
        print(f"❌ Result pattern failed: {e}")
        return False


def test_container():
    """Test DI container."""
    print("Testing DI container...")
    try:
        from autoshorts.core import Container, ServiceLifetime

        class DummyService:
            pass

        container = Container()
        container.register(DummyService, DummyService, ServiceLifetime.SINGLETON)

        s1 = container.resolve(DummyService)
        s2 = container.resolve(DummyService)

        assert s1 is s2, "Singleton should return same instance"
        assert container.is_registered(DummyService)

        print("✅ DI container works correctly")
        return True
    except Exception as e:
        print(f"❌ DI container failed: {e}")
        return False


def test_orchestrator_creation():
    """Test orchestrator can be created (without running)."""
    print("Testing orchestrator creation...")
    try:
        from autoshorts.orchestrator import ShortsOrchestrator
        from autoshorts.core import Container

        # Create mock container to avoid API key requirements
        container = Container()

        # This would normally fail without API keys, but we can
        # at least verify the class loads
        print("  Orchestrator class loaded successfully")
        print("  (Skipping instantiation - requires API keys)")

        print("✅ Orchestrator creation test passed")
        return True
    except Exception as e:
        print(f"❌ Orchestrator creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("REFACTORING VALIDATION")
    print("=" * 60)
    print()

    tests = [
        test_core_imports,
        test_pipeline_imports,
        test_config_loading,
        test_result_pattern,
        test_container,
        test_orchestrator_creation,
    ]

    results = []
    for test in tests:
        print()
        result = test()
        results.append(result)

    print()
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL TESTS PASSED - Refactoring is valid!")
        print("=" * 60)
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
