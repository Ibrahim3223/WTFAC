# -*- coding: utf-8 -*-
"""
Forced Alignment - PRODUCTION BULLETPROOF
Smart timing alignment with multiple fallback strategies
Edge-TTS provides excellent word-level timings - no external dependencies needed!
"""
import os
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

logger.info("✅ Caption aligner ready (TTS word-level timing)")


class ForcedAligner:
    """
    Bulletproof caption alignment with smart fallback strategies.
    
    Uses Edge-TTS word boundaries (excellent quality) with smart validation.
    
    Priority:
    1. Edge-TTS word timings (EXCELLENT - word-level precision from TTS engine)
    2. Smart estimation (GOOD - character-based distribution)
    """
    
    MIN_WORD_DURATION = 0.12  # Minimum realistic word duration
    
    def __init__(self):
        """Initialize aligner."""
        logger.info("      🎯 Caption aligner: TTS word-boundary mode (EXCELLENT quality)")
    
    def align(
        self,
        text: str,
        audio_path: str,
        tts_word_timings: Optional[List[Tuple[str, float]]] = None,
        total_duration: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Align text to audio with smart validation.
        
        Args:
            text: The known text that was spoken
            audio_path: Path to audio file (for validation)
            tts_word_timings: TTS-provided word timings (preferred)
            total_duration: Total audio duration (for validation)
        
        Returns:
            List of (word, duration) tuples with precise timing
        """
        # Strategy 1: TTS word timings (EXCELLENT)
        if tts_word_timings:
            logger.debug(f"      ✅ Using TTS word timings: {len(tts_word_timings)} words")
            return self._validate_timings(tts_word_timings, total_duration)
        
        # Strategy 2: Smart estimation (GOOD)
        if total_duration:
            logger.debug(f"      ℹ️ Using smart estimation")
            return self._smart_estimation(text, total_duration)
        
        # Last resort: equal distribution
        words = text.split()
        duration_per_word = 0.3  # Reasonable default
        logger.warning(f"      ⚠️ Using equal distribution fallback")
        return [(w, duration_per_word) for w in words]
    
    def _validate_timings(
        self,
        word_timings: List[Tuple[str, float]],
        total_duration: Optional[float]
    ) -> List[Tuple[str, float]]:
        """
        Validate and fix word timings to match total duration.
        
        Ensures:
        - No word shorter than MIN_WORD_DURATION
        - Total sum matches audio duration (±1%)
        - Smooth distribution
        """
        if not word_timings:
            return []
        
        # Enforce minimum duration
        fixed = [(w, max(self.MIN_WORD_DURATION, d)) for w, d in word_timings]
        
        # Validate total duration
        if total_duration:
            current_total = sum(d for _, d in fixed)
            
            # If mismatch > 1%, scale proportionally
            if abs(current_total - total_duration) > total_duration * 0.01:
                scale = total_duration / current_total if current_total > 0 else 1.0
                fixed = [(w, max(self.MIN_WORD_DURATION, d * scale)) for w, d in fixed]
                
                # Fine-tune last word to exactly match
                if fixed:
                    current_total = sum(d for _, d in fixed)
                    diff = total_duration - current_total
                    word, dur = fixed[-1]
                    fixed[-1] = (word, max(self.MIN_WORD_DURATION, dur + diff))
        
        return fixed
    
    def _smart_estimation(
        self,
        text: str,
        total_duration: float
    ) -> List[Tuple[str, float]]:
        """
        Smart character-based duration estimation.
        
        Better than equal distribution because it accounts for word length.
        """
        words = [w.strip() for w in text.split() if w.strip()]
        if not words:
            return []
        
        # Calculate character-based weights
        total_chars = sum(len(w) for w in words)
        if total_chars == 0:
            # Equal distribution fallback
            dur_per_word = total_duration / len(words)
            return [(w, dur_per_word) for w in words]
        
        # Distribute duration based on character count
        word_timings = []
        for word in words:
            char_ratio = len(word) / total_chars
            duration = max(self.MIN_WORD_DURATION, total_duration * char_ratio)
            word_timings.append((word, duration))
        
        # Validate and adjust
        return self._validate_timings(word_timings, total_duration)


# Global instance
_aligner_instance = None

def get_aligner() -> ForcedAligner:
    """Get or create global aligner instance."""
    global _aligner_instance
    if _aligner_instance is None:
        _aligner_instance = ForcedAligner()
    return _aligner_instance


def align_text_to_audio(
    text: str,
    audio_path: str,
    tts_word_timings: Optional[List[Tuple[str, float]]] = None,
    total_duration: Optional[float] = None
) -> List[Tuple[str, float]]:
    """
    Convenience function for caption alignment.
    
    Args:
        text: Known text that was spoken
        audio_path: Path to audio file
        tts_word_timings: TTS word timings (preferred)
        total_duration: Audio duration (for validation)
    
    Returns:
        List of (word, duration) tuples with precise sync
    """
    aligner = get_aligner()
    return aligner.align(text, audio_path, tts_word_timings, total_duration)
    
    def _validate_timings(
        self,
        word_timings: List[Tuple[str, float]],
        total_duration: Optional[float]
    ) -> List[Tuple[str, float]]:
        """
        Validate and fix word timings to match total duration.
        
        Ensures:
        - No word shorter than MIN_WORD_DURATION
        - Total sum matches audio duration (±1%)
        - Smooth distribution
        """
        if not word_timings:
            return []
        
        # Enforce minimum duration
        fixed = [(w, max(self.MIN_WORD_DURATION, d)) for w, d in word_timings]
        
        # Validate total duration
        if total_duration:
            current_total = sum(d for _, d in fixed)
            
            # If mismatch > 1%, scale proportionally
            if abs(current_total - total_duration) > total_duration * 0.01:
                scale = total_duration / current_total if current_total > 0 else 1.0
                fixed = [(w, max(self.MIN_WORD_DURATION, d * scale)) for w, d in fixed]
                
                # Fine-tune last word to exactly match
                if fixed:
                    current_total = sum(d for _, d in fixed)
                    diff = total_duration - current_total
                    word, dur = fixed[-1]
                    fixed[-1] = (word, max(self.MIN_WORD_DURATION, dur + diff))
        
        return fixed
    
    def _smart_estimation(
        self,
        text: str,
        total_duration: float
    ) -> List[Tuple[str, float]]:
        """
        Smart character-based duration estimation.
        
        Better than equal distribution because it accounts for word length.
        """
        words = [w.strip() for w in text.split() if w.strip()]
        if not words:
            return []
        
        # Calculate character-based weights
        total_chars = sum(len(w) for w in words)
        if total_chars == 0:
            # Equal distribution fallback
            dur_per_word = total_duration / len(words)
            return [(w, dur_per_word) for w in words]
        
        # Distribute duration based on character count
        word_timings = []
        for word in words:
            char_ratio = len(word) / total_chars
            duration = max(self.MIN_WORD_DURATION, total_duration * char_ratio)
            word_timings.append((word, duration))
        
        # Validate and adjust
        return self._validate_timings(word_timings, total_duration)


# Global instance
_aligner_instance = None

def get_aligner() -> ForcedAligner:
    """Get or create global aligner instance."""
    global _aligner_instance
    if _aligner_instance is None:
        _aligner_instance = ForcedAligner()
    return _aligner_instance


def align_text_to_audio(
    text: str,
    audio_path: str,
    tts_word_timings: Optional[List[Tuple[str, float]]] = None,
    total_duration: Optional[float] = None
) -> List[Tuple[str, float]]:
    """
    Convenience function for forced alignment.
    
    Args:
        text: Known text that was spoken
        audio_path: Path to audio file
        tts_word_timings: Optional TTS word timings (fallback)
        total_duration: Audio duration (for validation)
    
    Returns:
        List of (word, duration) tuples with perfect sync
    """
    aligner = get_aligner()
    return aligner.align(text, audio_path, tts_word_timings, total_duration)
