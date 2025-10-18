# -*- coding: utf-8 -*-
"""
Channel configuration loader from channels.yml
Loads channel-specific settings like topic, search_terms, etc.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def load_channels_config(config_path: str = "channels.yml") -> Dict[str, Any]:
    """
    Load channels.yml configuration file.
    
    Args:
        config_path: Path to channels.yml file
        
    Returns:
        Dict with channel configurations
    """
    # Try multiple locations
    possible_paths = [
        config_path,
        Path.cwd() / config_path,
        Path(__file__).parent.parent.parent / config_path,
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"✅ Loaded channels config from: {path}")
                    return config
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {path}: {e}")
                continue
    
    logger.warning(f"⚠️ channels.yml not found, using defaults")
    return {"channels": []}


def find_channel_config(channel_name: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find channel configuration by name.
    
    Args:
        channel_name: Channel name to search for
        config: Loaded channels config
        
    Returns:
        Channel config dict or None
    """
    channels = config.get("channels", [])
    
    # Normalize channel name for comparison
    channel_name_normalized = channel_name.strip().lower()
    
    for channel in channels:
        # Check both 'name' and 'env' fields
        name = channel.get("name", "").strip().lower()
        env = channel.get("env", "").strip().lower()
        
        if name == channel_name_normalized or env == channel_name_normalized:
            logger.info(f"✅ Found config for channel: {channel_name}")
            return channel
    
    logger.warning(f"⚠️ No config found for channel: {channel_name}")
    return None


def apply_channel_settings(channel_name: str) -> Dict[str, Any]:
    """
    Load and apply channel-specific settings.
    
    Args:
        channel_name: Name of the channel to configure
        
    Returns:
        Dict with channel-specific settings
    """
    # Load channels config
    config = load_channels_config()
    
    # Find this channel's config
    channel_config = find_channel_config(channel_name, config)
    
    if not channel_config:
        logger.warning(f"⚠️ Using default settings for: {channel_name}")
        return {
            "CHANNEL_TOPIC": "Interesting facts and knowledge",
            "CHANNEL_MODE": "general",
            "CHANNEL_SEARCH_TERMS": [],
            "CHANNEL_LANG": "en",
            "CHANNEL_VISIBILITY": "public"
        }
    
    # Extract settings
    settings = {
        "CHANNEL_TOPIC": channel_config.get("topic", "Interesting facts and knowledge"),
        "CHANNEL_MODE": channel_config.get("mode", "general"),
        "CHANNEL_SEARCH_TERMS": channel_config.get("search_terms", []),
        "CHANNEL_LANG": channel_config.get("lang", "en"),
        "CHANNEL_VISIBILITY": channel_config.get("visibility", "public")
    }
    
    logger.info(f"📺 Channel: {channel_name}")
    logger.info(f"🎯 Topic: {settings['CHANNEL_TOPIC'][:80]}...")
    logger.info(f"🔍 Search terms: {len(settings['CHANNEL_SEARCH_TERMS'])} available")
    
    return settings


def get_channel_search_terms(channel_name: str) -> List[str]:
    """
    Get search terms for a specific channel.
    
    Args:
        channel_name: Name of the channel
        
    Returns:
        List of search terms
    """
    config = load_channels_config()
    channel_config = find_channel_config(channel_name, config)
    
    if channel_config:
        return channel_config.get("search_terms", [])
    
    return []
