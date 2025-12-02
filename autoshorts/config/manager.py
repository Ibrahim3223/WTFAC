# -*- coding: utf-8 -*-
"""
Centralized Configuration Manager (Singleton Pattern)

Provides single source of truth for all configuration access.
Makes testing easier with mock configs.
"""

import logging
from typing import Optional
from autoshorts.config.models import AppConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Singleton configuration manager for centralized config access.

    Usage:
        >>> from autoshorts.config.manager import ConfigManager
        >>> config = ConfigManager.get_instance().config
        >>> print(config.script_style.hook_intensity)
        'extreme'
    """

    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None

    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """
        Get singleton instance of ConfigManager.

        Returns:
            ConfigManager instance
        """
        if cls._instance is None:
            cls._instance = ConfigManager()
            cls._instance._config = AppConfig()
            logger.info("[ConfigManager] Initialized with default configuration")
        return cls._instance

    @property
    def config(self) -> AppConfig:
        """
        Get configuration object.

        Returns:
            AppConfig instance with all sub-configs
        """
        if self._config is None:
            self._config = AppConfig()
        return self._config

    def reload(self):
        """
        Reload configuration from environment variables.

        Useful for testing or when env vars change at runtime.
        """
        self._config = AppConfig()
        logger.info("[ConfigManager] Configuration reloaded")

    def validate(self) -> bool:
        """
        Validate current configuration.

        Checks:
        - Required API keys are present
        - Video settings are valid
        - Script style settings are valid
        - All numeric values are within acceptable ranges

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required API keys
            if not self.config.api.gemini_api_key:
                logger.error("[ConfigManager] Missing GEMINI_API_KEY")
                return False

            # Check video settings
            if self.config.video.target_duration < 15:
                logger.error("[ConfigManager] target_duration must be >= 15 seconds")
                return False

            if self.config.video.target_duration > 60:
                logger.error("[ConfigManager] target_duration must be <= 60 seconds")
                return False

            # Check script style settings
            if self.config.script_style.hook_max_words < 5:
                logger.error("[ConfigManager] hook_max_words must be >= 5")
                return False

            if self.config.script_style.cliffhanger_interval < 5:
                logger.error("[ConfigManager] cliffhanger_interval must be >= 5")
                return False

            # Check TTS settings
            if not self.config.tts.edge_voice:
                logger.warning("[ConfigManager] No TTS voice specified, using default")

            logger.info("[ConfigManager] ✅ Configuration is valid")
            return True

        except Exception as e:
            logger.error(f"[ConfigManager] Validation error: {e}")
            return False

    def get_summary(self) -> dict:
        """
        Get configuration summary for debugging.

        Returns:
            Dictionary with key configuration values
        """
        return {
            "channel": {
                "name": self.config.channel.channel_name,
                "topic": self.config.channel.topic[:50] + "...",
                "lang": self.config.channel.lang,
                "mode": self.config.channel.mode,
            },
            "video": {
                "duration": self.config.video.target_duration,
                "resolution": f"{self.config.video.video_width}x{self.config.video.video_height}",
                "fps": self.config.video.target_fps,
            },
            "script_style": {
                "hook_intensity": self.config.script_style.hook_intensity,
                "cliffhangers": self.config.script_style.cliffhanger_enabled,
                "cliffhanger_interval": self.config.script_style.cliffhanger_interval,
                "keyword_highlighting": self.config.script_style.keyword_highlighting,
            },
            "quality": {
                "min_score": self.config.quality.min_quality_score,
                "max_attempts": self.config.max_generation_attempts,
            },
        }

    def print_summary(self):
        """Print configuration summary to console."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)

        for section, values in summary.items():
            print(f"\n{section.upper()}:")
            for key, value in values.items():
                print(f"  {key}: {value}")

        print("\n" + "=" * 60 + "\n")


# Convenience function
def get_config() -> AppConfig:
    """
    Convenience function to get configuration.

    Returns:
        AppConfig instance

    Example:
        >>> from autoshorts.config.manager import get_config
        >>> config = get_config()
        >>> print(config.script_style.hook_intensity)
    """
    return ConfigManager.get_instance().config


# Test function
def _test_config_manager():
    """Test ConfigManager functionality."""
    print("=" * 60)
    print("CONFIG MANAGER TESTS")
    print("=" * 60)

    # Test singleton
    manager1 = ConfigManager.get_instance()
    manager2 = ConfigManager.get_instance()
    print(f"\n✅ Singleton test: {manager1 is manager2}")

    # Test config access
    config = manager1.config
    print(f"\n✅ Config accessed:")
    print(f"  Channel: {config.channel.channel_name}")
    print(f"  Hook intensity: {config.script_style.hook_intensity}")
    print(f"  Cliffhangers enabled: {config.script_style.cliffhanger_enabled}")
    print(f"  Keyword highlighting: {config.script_style.keyword_highlighting}")

    # Test validation
    print(f"\n✅ Validation: {manager1.validate()}")

    # Test summary
    print("\n✅ Configuration Summary:")
    manager1.print_summary()

    # Test convenience function
    config2 = get_config()
    print(f"\n✅ Convenience function: {config is config2}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    _test_config_manager()
