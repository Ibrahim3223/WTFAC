"""
Prompt templates and patterns for content generation.
"""
from .hook_patterns import (
    get_shorts_hook,
    validate_cold_open,
    get_all_violations,
    SHORTS_HOOK_PATTERNS,
    COLD_OPEN_VIOLATIONS,
)

__all__ = [
    "get_shorts_hook",
    "validate_cold_open",
    "get_all_violations",
    "SHORTS_HOOK_PATTERNS",
    "COLD_OPEN_VIOLATIONS",
]
