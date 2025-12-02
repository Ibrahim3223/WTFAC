"""
TTS Stage - Generates audio for all sentences using Edge TTS.
"""

import os
from typing import List, Dict, Any

from ..base import PipelineStage, PipelineContext
from ...core import Result, TTSError
from ...tts.edge_handler import TTSHandler


class TTSStage(PipelineStage):
    """
    Generate TTS audio for all content sentences.

    Dependencies:
    - TTSHandler: Text-to-speech synthesis

    Requires context:
    - content: Content from ContentGenerationStage
    - temp_dir: Temporary directory for audio files

    Updates context with:
    - audio_segments: List of audio segments with timings
    """

    def __init__(self, tts_handler: TTSHandler):
        super().__init__("TTS")
        self.tts = tts_handler

    def execute(self, context: PipelineContext) -> Result[PipelineContext, str]:
        """Generate TTS for all sentences."""
        # Validate prerequisites
        if not context.content:
            return Result.err("No content available. Run ContentGenerationStage first.")

        if not context.temp_dir:
            return Result.err("No temp_dir specified in context.")

        try:
            sentences = context.content["sentences"]
            audio_segments: List[Dict[str, Any]] = []

            for i, sentence in enumerate(sentences, 1):
                self.logger.info(f"Processing sentence {i}/{len(sentences)}")

                audio_file = os.path.join(context.temp_dir, f"sentence_{i}.wav")

                duration, word_timings = self.tts.synthesize(
                    text=sentence,
                    wav_out=audio_file
                )

                if not duration or not os.path.exists(audio_file):
                    return Result.err(f"TTS failed for sentence {i}")

                # Determine sentence type
                if i == 1:
                    sentence_type = "hook"
                elif i == len(sentences):
                    sentence_type = "cta"
                elif i == len(sentences) - 1:
                    sentence_type = "payoff"
                else:
                    sentence_type = "buildup"

                segment = {
                    "text": sentence,
                    "audio_path": audio_file,
                    "duration": duration,
                    "word_timings": word_timings,
                    "type": sentence_type
                }
                audio_segments.append(segment)

            # Update context
            context.audio_segments = audio_segments

            self.logger.info(f"✅ Generated {len(audio_segments)} audio segments")

            return Result.ok(context)

        except Exception as e:
            self.logger.error(f"TTS error: {e}", exc_info=True)
            return Result.err(f"TTS generation failed: {str(e)}")
