"""
Orchestrator - Main pipeline coordinator
Manages the full flow: content → TTS → video → upload
"""

import os
import tempfile
import shutil
import logging
from typing import Optional, Dict, Any, List

from .config import settings
from .content.gemini_client import GeminiClient
from .content.quality_scorer import QualityScorer
from .tts.edge_handler import EdgeTTSHandler
from .video.pexels_client import PexelsClient
from .video.downloader import VideoDownloader
from .video.segment_maker import SegmentMaker
from .captions.renderer import CaptionRenderer
from .audio.bgm_manager import BGMManager
from .upload.youtube_uploader import YouTubeUploader
from .state.novelty_guard import NoveltyGuard
from .state.state_guard import StateGuard

logger = logging.getLogger(__name__)


class ShortsOrchestrator:
    """Main orchestrator for the YouTube Shorts pipeline."""
    
    def __init__(self):
        """Initialize all components with proper API keys."""
        self.channel = settings.CHANNEL_NAME
        self.temp_dir = None
        
        # Get API keys from environment
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please set it in GitHub Secrets or your .env file."
            )
        
        pexels_api_key = os.getenv("PEXELS_API_KEY", "")
        pixabay_api_key = os.getenv("PIXABAY_API_KEY", "")
        
        # Initialize modules with API keys
        logger.info(f"[Gemini] Using model: {settings.GEMINI_MODEL}")
        self.gemini = GeminiClient(
            api_key=gemini_api_key,
            model=settings.GEMINI_MODEL,
            max_retries=3
        )
        
        self.quality_scorer = QualityScorer()
        self.tts = EdgeTTSHandler()
        
        self.pexels = PexelsClient(
            pexels_key=pexels_api_key,
            pixabay_key=pixabay_api_key
        )
        
        self.downloader = VideoDownloader()
        self.segment_maker = SegmentMaker()
        self.caption_renderer = CaptionRenderer()
        self.bgm_manager = BGMManager()
        self.uploader = YouTubeUploader()
        
        # State management
        self.novelty_guard = NoveltyGuard(
            state_dir=settings.STATE_DIR,
            window_days=settings.ENTITY_COOLDOWN_DAYS
        )
        self.state_guard = StateGuard(channel=self.channel)
        
        logger.info(f"🚀 Orchestrator ready: {self.channel}")
    
    def run(self) -> Optional[str]:
        """
        Execute the full pipeline.
        Returns: YouTube video ID or None on failure.
        """
        self.temp_dir = tempfile.mkdtemp(prefix="shorts_")
        
        max_attempts = settings.MAX_GENERATION_ATTEMPTS
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"   Attempt {attempt}/{max_attempts}")
                
                # Phase 1: Content Generation
                logger.info("📝 Phase 1: Content generation...")
                content = self._generate_content()
                if not content:
                    logger.error("❌ Content generation failed")
                    continue
                
                # Phase 2: TTS
                logger.info("🎤 Phase 2: Text-to-speech...")
                audio_segments = self._generate_tts(content)
                if not audio_segments:
                    logger.error("❌ TTS generation failed")
                    continue
                
                # Phase 3: Video
                logger.info("🎬 Phase 3: Video production...")
                video_path = self._produce_video(audio_segments, content)
                if not video_path:
                    logger.error("❌ Video production failed")
                    continue
                
                # Phase 4: Upload
                logger.info("📤 Phase 4: Uploading to YouTube...")
                if settings.UPLOAD_TO_YT:
                    video_id = self._upload(video_path, content)
                    logger.info(f"✅ Success! Video ID: {video_id}")
                    return video_id
                else:
                    logger.info(f"⏭️ Upload skipped. Video saved: {video_path}")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Pipeline failed: {e}")
                if attempt == max_attempts:
                    raise
                logger.info(f"🔄 Retrying... ({attempt}/{max_attempts})")
        
        logger.error(f"❌ All {max_attempts} attempts failed")
        return None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("🧹 Cleaned temp files")
            except Exception as e:
                logger.warning(f"⚠️ Cleanup warning: {e}")
    
    def _generate_content(self) -> Optional[Dict[str, Any]]:
        """
        Generate content with quality checks and novelty guard.
        Returns: Content dict or None on failure.
        """
        try:
            # Generate content using Gemini
            content = self.gemini.generate(
                topic=settings.CHANNEL_TOPIC,
                style=settings.CONTENT_STYLE,
                duration=settings.TARGET_DURATION,
                additional_context=settings.ADDITIONAL_PROMPT_CONTEXT
            )
            
            # Combine all text for quality scoring
            full_text = " ".join([
                content.hook,
                *content.script,
                content.cta
            ])
            
            # Quality check
            score = self.quality_scorer.score_content(full_text)
            logger.info(f"   Quality score: {score:.2f}")
            
            if score < settings.MIN_QUALITY_SCORE:
                logger.warning(f"   ⚠️ Quality too low: {score:.2f} < {settings.MIN_QUALITY_SCORE}")
                return None
            
            # Novelty check
            if not self.novelty_guard.is_novel(full_text):
                logger.warning("   ⚠️ Content not novel enough (too similar to recent videos)")
                return None
            
            # Prepare structured content
            structured_content = {
                "hook": content.hook,
                "script": content.script,
                "cta": content.cta,
                "search_queries": content.search_queries,
                "metadata": content.metadata,
                "sentences": [content.hook] + content.script + [content.cta],
                "quality_score": score
            }
            
            logger.info(f"   ✅ Content generated: {len(structured_content['sentences'])} sentences")
            return structured_content
            
        except Exception as e:
            logger.error(f"   ❌ Content generation error: {e}")
            return None
    
    def _generate_tts(self, content: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Generate TTS for all sentences.
        Returns: List of audio segments or None on failure.
        """
        try:
            sentences = content["sentences"]
            audio_segments = []
            
            for i, sentence in enumerate(sentences, 1):
                logger.info(f"   Processing sentence {i}/{len(sentences)}")
                
                segment = self.tts.generate(
                    text=sentence,
                    output_dir=self.temp_dir,
                    voice=settings.TTS_VOICE,
                    rate=settings.TTS_RATE,
                    pitch=settings.TTS_PITCH
                )
                
                if segment:
                    audio_segments.append(segment)
                else:
                    logger.error(f"   ❌ TTS failed for sentence {i}")
                    return None
            
            logger.info(f"   ✅ Generated {len(audio_segments)} audio segments")
            return audio_segments
            
        except Exception as e:
            logger.error(f"   ❌ TTS error: {e}")
            return None
    
    def _produce_video(
        self,
        audio_segments: List[Dict[str, Any]],
        content: Dict[str, Any]
    ) -> Optional[str]:
        """
        Produce the final video.
        Returns: Path to final video or None on failure.
        """
        try:
            # Step 1: Search and download videos
            logger.info("   🔍 Searching for videos...")
            video_clips = self.pexels.search_videos(
                queries=content["search_queries"],
                min_duration=settings.TARGET_DURATION
            )
            
            if not video_clips:
                logger.error("   ❌ No suitable videos found")
                return None
            
            logger.info(f"   📥 Downloading {len(video_clips)} videos...")
            downloaded = self.downloader.download_videos(
                video_clips,
                output_dir=self.temp_dir
            )
            
            if not downloaded:
                logger.error("   ❌ Video download failed")
                return None
            
            # Step 2: Create video segments
            logger.info("   ✂️ Creating video segments...")
            video_segments = self.segment_maker.create_segments(
                video_files=downloaded,
                audio_segments=audio_segments,
                output_dir=self.temp_dir
            )
            
            if not video_segments:
                logger.error("   ❌ Segment creation failed")
                return None
            
            # Step 3: Add captions
            logger.info("   📝 Adding captions...")
            captioned_segments = self.caption_renderer.render_captions(
                video_segments=video_segments,
                audio_segments=audio_segments,
                output_dir=self.temp_dir
            )
            
            if not captioned_segments:
                logger.error("   ❌ Caption rendering failed")
                return None
            
            # Step 4: Add BGM and finalize
            logger.info("   🎵 Adding background music...")
            bgm_path = self.bgm_manager.get_bgm(
                duration=settings.TARGET_DURATION,
                output_dir=self.temp_dir
            )
            
            # Concatenate all segments
            final_video = os.path.join(self.temp_dir, "final_video.mp4")
            
            # Simple concatenation for now
            concat_list = os.path.join(self.temp_dir, "concat_list.txt")
            with open(concat_list, "w") as f:
                for segment in captioned_segments:
                    f.write(f"file '{segment}'\n")
            
            import subprocess
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                final_video
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"   ❌ FFmpeg error: {result.stderr}")
                return None
            
            if not os.path.exists(final_video):
                logger.error("   ❌ Final video not created")
                return None
            
            logger.info(f"   ✅ Video produced: {final_video}")
            return final_video
            
        except Exception as e:
            logger.error(f"   ❌ Video production error: {e}")
            return None
    
    def _upload(self, video_path: str, content: Dict[str, Any]) -> Optional[str]:
        """
        Upload video to YouTube.
        Returns: Video ID or None on failure.
        """
        try:
            metadata = content["metadata"]
            
            video_id = self.uploader.upload(
                video_path=video_path,
                title=metadata.get("title", "Amazing Short"),
                description=metadata.get("description", ""),
                tags=metadata.get("tags", []),
                category_id="22",
                privacy_status="public"
            )
            
            if video_id:
                # Record in state
                self.state_guard.record_upload(video_id, content)
                logger.info(f"   ✅ Uploaded: https://youtube.com/watch?v={video_id}")
                return video_id
            else:
                logger.error("   ❌ Upload failed")
                return None
                
        except Exception as e:
            logger.error(f"   ❌ Upload error: {e}")
            return None
