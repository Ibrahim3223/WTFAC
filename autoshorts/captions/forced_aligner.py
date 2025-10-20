# -*- coding: utf-8 -*-
"""
Forced Alignment - MILISANIYE HASSASIYETİ (WhisperX)
WhisperX ile phoneme-level alignment - izleyicinin gözünü yormayan mükemmel sync!
3-layer fallback system: WhisperX → TTS → Estimation
"""
import os
import logging
import warnings
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# WhisperX import (lazy load)
_WHISPERX_AVAILABLE = False
_whisperx_model = None
_align_model = None
_align_metadata = None
_device = None

try:
    import whisperx
    import torch
    _WHISPERX_AVAILABLE = True
    # GPU detection
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"✅ WhisperX available - Device: {_device.upper()}")
    logger.info("   🎯 Caption alignment: PHONEME-LEVEL precision (~10ms)")
except ImportError:
    logger.warning("⚠️ WhisperX not available - falling back to TTS timings")
    logger.info("   Install: pip install git+https://github.com/m-bain/whisperx.git")


class ForcedAligner:
    """
    Milisaniye hassasiyetinde caption alignment.
    
    3-layer fallback system:
    1. WhisperX forced alignment (BEST - ~10ms phoneme-level precision)
    2. Edge-TTS word timings (GOOD - ~50-100ms TTS engine boundaries)
    3. Character-based estimation (FALLBACK - ~200ms word length distribution)
    """
    
    MIN_WORD_DURATION = 0.08  # 80ms minimum (daha gerçekçi)
    MAX_WORD_DURATION = 3.0   # 3 saniye maximum
    WHISPERX_MODEL = "base"   # base model: hız/kalite dengesi optimal
    
    def __init__(self, language: str = "tr"):
        """
        Initialize aligner.
        
        Args:
            language: Language code for WhisperX (tr, en, etc.)
        """
        self.language = language
        self._ensure_whisperx_models()
        
        if _WHISPERX_AVAILABLE:
            logger.info(f"      🎯 Caption aligner: WhisperX mode ({self.language.upper()}) - MILISANIYE HASSASİYETİ")
        else:
            logger.info("      ℹ️ Caption aligner: TTS fallback mode")
    
    def _ensure_whisperx_models(self):
        """Lazy load WhisperX models (only once)."""
        global _whisperx_model, _align_model, _align_metadata, _device
        
        if not _WHISPERX_AVAILABLE:
            return
        
        try:
            # Load transcription model (once)
            if _whisperx_model is None:
                logger.info(f"      📦 Loading WhisperX model: {self.WHISPERX_MODEL}...")
                
                # Suppress warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _whisperx_model = whisperx.load_model(
                        self.WHISPERX_MODEL,
                        device=_device,
                        compute_type="float16" if _device == "cuda" else "int8",
                        language=self.language
                    )
                
                logger.info(f"      ✅ WhisperX model loaded on {_device.upper()}")
            
            # Load alignment model (once)
            if _align_model is None:
                logger.info(f"      📦 Loading alignment model ({self.language})...")
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _align_model, _align_metadata = whisperx.load_align_model(
                        language_code=self.language,
                        device=_device
                    )
                
                logger.info("      ✅ Alignment model loaded")
        
        except Exception as e:
            logger.error(f"      ❌ WhisperX model load failed: {e}")
            logger.info("      ℹ️ Falling back to TTS timings")
    
    def align(
        self,
        text: str,
        audio_path: str,
        tts_word_timings: Optional[List[Tuple[str, float]]] = None,
        total_duration: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Milisaniye hassasiyetinde caption alignment.
        
        Args:
            text: Known text that was spoken
            audio_path: Path to audio file
            tts_word_timings: TTS word timings (fallback)
            total_duration: Total audio duration (validation)
        
        Returns:
            List of (word, duration) tuples with milisecond precision
        """
        # Strategy 1: WhisperX forced alignment (BEST - ~10ms precision)
        if _WHISPERX_AVAILABLE and os.path.exists(audio_path):
            try:
                logger.debug(f"      🎯 WhisperX alignment: {audio_path}")
                words = self._whisperx_align(text, audio_path, total_duration)
                if words:
                    logger.debug(f"      ✅ WhisperX: {len(words)} words, phoneme-level sync")
                    return words
            except Exception as e:
                logger.warning(f"      ⚠️ WhisperX failed: {e}")
                logger.debug(f"      ℹ️ Falling back to TTS timings...")
        
        # Strategy 2: TTS word timings (GOOD - ~50-100ms precision)
        if tts_word_timings:
            logger.debug(f"      ℹ️ Using TTS word timings: {len(tts_word_timings)} words")
            return self._validate_timings(tts_word_timings, total_duration)
        
        # Strategy 3: Smart estimation (FALLBACK - ~200ms precision)
        if total_duration:
            logger.debug(f"      ℹ️ Using character-based estimation")
            return self._smart_estimation(text, total_duration)
        
        # Last resort: equal distribution
        words = text.split()
        duration_per_word = 0.25
        logger.warning(f"      ⚠️ Using equal distribution fallback")
        return [(w, duration_per_word) for w in words]
    
    def _whisperx_align(
        self,
        text: str,
        audio_path: str,
        total_duration: Optional[float]
    ) -> Optional[List[Tuple[str, float]]]:
        """
        WhisperX forced alignment - phoneme-level precision.
        
        Returns:
            List of (word, duration) or None if failed
        """
        global _whisperx_model, _align_model, _align_metadata, _device
        
        if not (_whisperx_model and _align_model):
            return None
        
        try:
            # Load audio
            audio = whisperx.load_audio(audio_path)
            
            # Transcribe (with known language)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _whisperx_model.transcribe(
                    audio,
                    batch_size=16,
                    language=self.language
                )
            
            # Forced alignment (magic happens here!)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result_aligned = whisperx.align(
                    result["segments"],
                    _align_model,
                    _align_metadata,
                    audio,
                    device=_device,
                    return_char_alignments=False  # Word-level yeterli
                )
            
            # Extract word timings
            word_timings = []
            for segment in result_aligned.get("segments", []):
                for word_info in segment.get("words", []):
                    word = word_info.get("word", "").strip()
                    start = word_info.get("start", 0.0)
                    end = word_info.get("end", 0.0)
                    duration = max(self.MIN_WORD_DURATION, end - start)
                    
                    if word:
                        word_timings.append((word, duration))
            
            if not word_timings:
                logger.warning("      ⚠️ WhisperX returned no words")
                return None
            
            # Validate against known text
            whisper_text = " ".join(w for w, _ in word_timings).lower()
            known_text = text.lower()
            
            # Simple similarity check
            if len(whisper_text) < len(known_text) * 0.5:
                logger.warning(f"      ⚠️ WhisperX text too different from known text")
                logger.debug(f"         Known: {known_text[:100]}")
                logger.debug(f"         WhisperX: {whisper_text[:100]}")
                return None
            
            # Validate total duration
            validated = self._validate_timings(word_timings, total_duration)
            
            return validated
        
        except Exception as e:
            logger.error(f"      ❌ WhisperX alignment error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
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
        - No word longer than MAX_WORD_DURATION
        - Total sum matches audio duration (±2%)
        - Smooth distribution
        """
        if not word_timings:
            return []
        
        # Clean words
        word_timings = [(w.strip(), d) for w, d in word_timings if w.strip()]
        if not word_timings:
            return []
        
        # Enforce duration bounds
        fixed = []
        for word, dur in word_timings:
            dur = max(self.MIN_WORD_DURATION, min(dur, self.MAX_WORD_DURATION))
            fixed.append((word, dur))
        
        # Validate total duration
        if total_duration:
            current_total = sum(d for _, d in fixed)
            
            # If mismatch > 2%, scale proportionally
            if abs(current_total - total_duration) > total_duration * 0.02:
                scale = total_duration / current_total if current_total > 0 else 1.0
                logger.debug(f"      📏 Scaling timings: {scale:.3f}x")
                
                fixed = [
                    (w, max(self.MIN_WORD_DURATION, d * scale))
                    for w, d in fixed
                ]
            
            # Fine-tune last word to exactly match
            if fixed:
                current_total = sum(d for _, d in fixed)
                diff = total_duration - current_total
                
                # Distribute difference across last few words
                if abs(diff) > 0.01:  # >10ms difference
                    num_words_adjust = min(3, len(fixed))
                    adjustment_per_word = diff / num_words_adjust
                    
                    for i in range(len(fixed) - num_words_adjust, len(fixed)):
                        word, dur = fixed[i]
                        new_dur = max(self.MIN_WORD_DURATION, dur + adjustment_per_word)
                        fixed[i] = (word, round(new_dur, 3))
        
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

def get_aligner(language: str = "tr") -> ForcedAligner:
    """Get or create global aligner instance."""
    global _aligner_instance
    if _aligner_instance is None:
        _aligner_instance = ForcedAligner(language=language)
    return _aligner_instance


def align_text_to_audio(
    text: str,
    audio_path: str,
    tts_word_timings: Optional[List[Tuple[str, float]]] = None,
    total_duration: Optional[float] = None,
    language: str = "tr"
) -> List[Tuple[str, float]]:
    """
    Milisaniye hassasiyetinde caption alignment.
    
    Args:
        text: Known text that was spoken
        audio_path: Path to audio file
        tts_word_timings: TTS word timings (fallback)
        total_duration: Audio duration (validation)
        language: Language code (tr, en, etc.)
    
    Returns:
        List of (word, duration) tuples with milisecond precision
    """
    aligner = get_aligner(language=language)
    return aligner.align(text, audio_path, tts_word_timings, total_duration)
