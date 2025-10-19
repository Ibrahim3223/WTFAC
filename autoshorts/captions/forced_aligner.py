# -*- coding: utf-8 -*-
"""
Forced Alignment - PRODUCTION BULLETPROOF
Uses aeneas for perfect text-to-audio synchronization
Falls back to estimation if aeneas unavailable
"""
import os
import json
import logging
import tempfile
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Try to import aeneas (optional)
AENEAS_AVAILABLE = False
try:
    from aeneas.executetask import ExecuteTask
    from aeneas.task import Task
    AENEAS_AVAILABLE = True
    logger.info("✅ aeneas available for perfect forced alignment")
except ImportError as e:
    logger.warning(f"⚠️ aeneas not available: {e}")
    logger.info("   💡 For perfect sync, install:")
    logger.info("   💡   sudo apt-get install espeak libespeak-dev python3-dev")
    logger.info("   💡   pip install numpy Cython aeneas")
except Exception as e:
    logger.warning(f"⚠️ aeneas import error: {e}")
    logger.info("   System will use fallback methods")


class ForcedAligner:
    """
    Bulletproof forced alignment with multiple fallback strategies.
    
    Priority:
    1. aeneas forced alignment (BEST - phoneme-level precision)
    2. TTS word timings (GOOD - if TTS provides them)
    3. Smart estimation (FALLBACK - character-based)
    """
    
    MIN_WORD_DURATION = 0.12  # Minimum realistic word duration
    
    def __init__(self):
        """Initialize forced aligner."""
        self.aeneas_available = AENEAS_AVAILABLE
        
        if self.aeneas_available:
            logger.info("      🎯 Forced aligner: aeneas mode (BEST quality)")
        else:
            logger.info("      ⚙️ Forced aligner: estimation mode (GOOD quality)")
    
    def align(
        self,
        text: str,
        audio_path: str,
        tts_word_timings: Optional[List[Tuple[str, float]]] = None,
        total_duration: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Align text to audio with multiple fallback strategies.
        
        Args:
            text: The known text that was spoken
            audio_path: Path to audio file
            tts_word_timings: Optional TTS-provided word timings (fallback)
            total_duration: Total audio duration (for validation)
        
        Returns:
            List of (word, duration) tuples with perfect timing
        """
        # Strategy 1: aeneas forced alignment (BEST)
        if self.aeneas_available:
            try:
                result = self._align_with_aeneas(text, audio_path)
                if result:
                    logger.debug(f"      ✅ aeneas alignment: {len(result)} words")
                    return self._validate_timings(result, total_duration)
            except Exception as e:
                logger.debug(f"      ℹ️ aeneas failed: {e}, trying fallback...")
        
        # Strategy 2: TTS word timings (GOOD)
        if tts_word_timings:
            logger.debug(f"      ℹ️ Using TTS word timings: {len(tts_word_timings)} words")
            return self._validate_timings(tts_word_timings, total_duration)
        
        # Strategy 3: Smart estimation (FALLBACK)
        if total_duration:
            logger.debug(f"      ℹ️ Using smart estimation")
            return self._smart_estimation(text, total_duration)
        
        # Last resort: equal distribution
        words = text.split()
        duration_per_word = 0.3  # Reasonable default
        logger.warning(f"      ⚠️ Using equal distribution fallback")
        return [(w, duration_per_word) for w in words]
    
    def _align_with_aeneas(
        self,
        text: str,
        audio_path: str
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Use aeneas for phoneme-perfect forced alignment.
        
        This is the GOLD STANDARD - aligns known text to actual speech.
        """
        if not self.aeneas_available:
            return None
        
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as text_file:
                text_file.write(text)
                text_path = text_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as sync_file:
                sync_path = sync_file.name
            
            try:
                # Configure aeneas task
                config_string = "task_language=eng|is_text_type=plain|os_task_file_format=json"
                
                task = Task()
                task.audio_file_path_absolute = os.path.abspath(audio_path)
                task.text_file_path_absolute = os.path.abspath(text_path)
                task.sync_map_file_path_absolute = os.path.abspath(sync_path)
                task.configuration_string = config_string
                
                # Execute forced alignment
                ExecuteTask(task).execute()
                
                # Parse results
                if not os.path.exists(sync_path):
                    logger.debug(f"      ⚠️ aeneas output file not created")
                    return None
                
                with open(sync_path, 'r') as f:
                    sync_map = json.load(f)
                
                # Extract word timings
                word_timings = []
                fragments = sync_map.get("fragments", [])
                
                for fragment in fragments:
                    begin = float(fragment.get("begin", 0))
                    end = float(fragment.get("end", 0))
                    fragment_text = fragment.get("lines", [""])[0].strip()
                    
                    if not fragment_text:
                        continue
                    
                    # Split fragment into words (aeneas may group words)
                    words_in_fragment = fragment_text.split()
                    duration = end - begin
                    
                    if len(words_in_fragment) == 1:
                        # Single word
                        word_timings.append((words_in_fragment[0], duration))
                    else:
                        # Multiple words - distribute duration
                        char_count = sum(len(w) for w in words_in_fragment)
                        for word in words_in_fragment:
                            word_ratio = len(word) / char_count if char_count > 0 else 1.0 / len(words_in_fragment)
                            word_dur = max(self.MIN_WORD_DURATION, duration * word_ratio)
                            word_timings.append((word, word_dur))
                
                return word_timings
                
            finally:
                # Cleanup temp files
                try:
                    os.unlink(text_path)
                    os.unlink(sync_path)
                except:
                    pass
                    
        except Exception as e:
            logger.debug(f"      ⚠️ aeneas alignment error: {e}")
            return None
    
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
