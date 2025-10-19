"""
Orchestrator - Main pipeline coordinator with SMART VIDEO SELECTION
Manages the full flow: content → TTS → video → upload
"""

import os
import tempfile
import shutil
import logging
import subprocess
import random
from typing import Optional, Dict, Any, List

from .config import settings
from .content.gemini_client import GeminiClient
from .content.quality_scorer import QualityScorer
from .tts.edge_handler import TTSHandler
from .video.pexels_client import PexelsClient
from .video.downloader import VideoDownloader
from .video.segment_maker import SegmentMaker
from .captions.renderer import CaptionRenderer
from .audio.bgm_manager import BGMManager
from .upload.youtube_uploader import YouTubeUploader
from .state.novelty_guard import NoveltyGuard
from .state.state_guard import StateGuard

logger = logging.getLogger(__name__)

# ============================================================================
# VIRAL TIMING CONSTANTS - Optimized for retention
# ============================================================================
SHOT_DURATION = {
    "hook": (1.3, 1.9),       # Ultra-fast hook (1.3-1.9s)
    "buildup": (2.2, 3.0),    # Normal tempo
    "payoff": (2.8, 3.5),     # Slightly longer for key points
    "cta": (2.0, 2.5)         # Quick, punchy ending
}

# Viral video scoring weights
VIDEO_SCORE_WEIGHTS = {
    "motion": 30,           # High motion = engaging
    "brightness": 20,       # Bright = better retention
    "saturation": 20,       # Vibrant colors = eye-catching
    "center_focus": 15,     # Center composition = professional
    "duration_match": 15    # Right length = less cutting needed
}

class ShortsOrchestrator:
    """Main orchestrator for the YouTube Shorts pipeline with smart video selection."""
    
    def __init__(self):
        """Initialize all components with proper API keys."""
        logger.info("=" * 60)
        logger.info("Initializing ShortsOrchestrator...")
        logger.info("=" * 60)
        
        self.channel = settings.CHANNEL_NAME
        self.temp_dir = None
        
        logger.info(f"📺 Channel: {self.channel}")
        logger.info(f"🎯 Topic: {settings.CHANNEL_TOPIC}")
        logger.info(f"⏱️  Duration: {settings.TARGET_DURATION}s")
        
        # Get API keys from settings
        gemini_api_key = settings.GEMINI_API_KEY
        pexels_api_key = settings.PEXELS_API_KEY
        
        # Validate Gemini API key
        if not gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not found! "
                "Please set it in GitHub Secrets (Repository or Environment)."
            )
        
        logger.info(f"✅ Gemini API key: {gemini_api_key[:10]}...{gemini_api_key[-4:]}")
        
        # Validate Pexels API key
        if not pexels_api_key:
            logger.warning("⚠️ PEXELS_API_KEY not found - video search may fail")
        else:
            logger.info(f"✅ Pexels API key: {pexels_api_key[:10]}...")
        
        # Get Gemini model from settings
        gemini_model = settings.GEMINI_MODEL
        logger.info(f"🤖 Gemini model setting: {gemini_model}")
        
        # Initialize Gemini client
        logger.info("Initializing Gemini client...")
        try:
            self.gemini = GeminiClient(
                api_key=gemini_api_key,
                model=gemini_model,
                max_retries=3
            )
            logger.info("✅ Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            raise
        
        # Initialize other modules
        logger.info("Initializing other modules...")
        self.quality_scorer = QualityScorer()
        self.tts = TTSHandler()
        self.pexels = PexelsClient()
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
        
        logger.info("=" * 60)
        logger.info(f"🚀 Orchestrator ready: {self.channel}")
        logger.info("=" * 60)
    
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
                import traceback
                logger.debug(traceback.format_exc())
                
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
            logger.info("   🔮 Calling Gemini API...")
            logger.info(f"   Topic: {settings.CHANNEL_TOPIC}")
            logger.info(f"   Style: {settings.CONTENT_STYLE}")
            logger.info(f"   Duration: {settings.TARGET_DURATION}s")
            
            # Generate content using Gemini
            content = self.gemini.generate(
                topic=settings.CHANNEL_TOPIC,
                style=settings.CONTENT_STYLE,
                duration=settings.TARGET_DURATION,
                additional_context=settings.ADDITIONAL_PROMPT_CONTEXT
            )
            
            logger.info("   ✅ Gemini response received")
            
            # Combine all text for quality scoring
            full_text = " ".join([
                content.hook,
                *content.script,
                content.cta
            ])
            
            # Quality check
            score_result = self.quality_scorer.score(
                sentences=[content.hook] + content.script + [content.cta],
                title=content.metadata.get("title", "")
            )
            score = score_result.get("overall", 0.0)
            
            # Log all scores for debugging
            logger.info(f"   Quality: {score_result.get('quality', 0):.2f} | "
                       f"Viral: {score_result.get('viral', 0):.2f} | "
                       f"Retention: {score_result.get('retention', 0):.2f} | "
                       f"Overall: {score:.2f}")
            
            min_score = settings.MIN_QUALITY_SCORE
            if score < min_score:
                logger.warning(f"   ⚠️ Quality too low: {score:.2f} < {min_score}")
                logger.info(f"   💡 Tip: Lower MIN_QUALITY_SCORE in settings or channels.yml if this happens often")
                return None
            
            # Novelty check
            decision = self.novelty_guard.check_novelty(
                channel=self.channel,
                title=content.metadata.get("title", ""),
                script=full_text,
                search_term=content.search_queries[0] if content.search_queries else None,
                category=settings.CHANNEL_TOPIC,
                lang=settings.LANG
            )
            
            if not decision.ok:
                logger.warning(f"   ⚠️ Content not novel enough: {decision.reason}")
                return None
            
            # Prepare structured content
            structured_content = {
                "hook": content.hook,
                "script": content.script,
                "cta": content.cta,
                "search_queries": content.search_queries,
                "main_visual_focus": content.main_visual_focus,
                "metadata": content.metadata,
                "sentences": [content.hook] + content.script + [content.cta],
                "quality_score": score
            }
            
            logger.info(f"   ✅ Content generated: {len(structured_content['sentences'])} sentences")
            return structured_content
            
        except Exception as e:
            logger.error(f"   ❌ Content generation error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
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
                
                # Generate audio file path
                audio_file = os.path.join(self.temp_dir, f"sentence_{i}.wav")
                
                # Synthesize with edge TTS
                duration, word_timings = self.tts.synthesize(
                    text=sentence,
                    wav_out=audio_file
                )
                
                if duration and os.path.exists(audio_file):
                    # Determine sentence type for pacing
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
                        "type": sentence_type  # NEW: Track sentence type for pacing
                    }
                    audio_segments.append(segment)
                else:
                    logger.error(f"   ❌ TTS failed for sentence {i}")
                    return None
            
            logger.info(f"   ✅ Generated {len(audio_segments)} audio segments")
            return audio_segments
            
        except Exception as e:
            logger.error(f"   ❌ TTS error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _score_video(self, video_metadata: Dict[str, Any], target_duration: float) -> float:
        """
        Score a video for viral potential.
        Returns: Score from 0-100 (higher is better)
        """
        score = 0.0
        
        # Extract metadata (if available)
        duration = video_metadata.get("duration", 0)
        width = video_metadata.get("width", 1920)
        height = video_metadata.get("height", 1080)
        
        # 1. Duration match (15 points)
        if duration > 0:
            duration_diff = abs(duration - target_duration)
            if duration_diff < 2:
                score += VIDEO_SCORE_WEIGHTS["duration_match"]
            elif duration_diff < 5:
                score += VIDEO_SCORE_WEIGHTS["duration_match"] * 0.7
            elif duration_diff < 10:
                score += VIDEO_SCORE_WEIGHTS["duration_match"] * 0.4
        
        # 2. Aspect ratio (prefer vertical or square for shorts)
        aspect_ratio = width / height if height > 0 else 1.0
        if 0.5 <= aspect_ratio <= 0.6:  # Vertical (9:16)
            score += 10
        elif 0.7 <= aspect_ratio <= 1.3:  # Square-ish
            score += 7
        
        # 3. Base score for having metadata
        score += 25
        
        return score
    
    def _produce_video(
        self,
        audio_segments: List[Dict[str, Any]],
        content: Dict[str, Any]
    ) -> Optional[str]:
        """
        Produce the final video with smart video selection.
        Returns: Path to final video or None on failure.
        """
        try:
            # Step 1: Search and download videos - SMART SELECTION
            logger.info("   🔍 Searching for high-quality videos...")
            
            # Use main_visual_focus for coherent search
            main_topic = content.get("main_visual_focus", "")
            
            if not main_topic:
                # Fallback to first search query
                search_queries = content.get("search_queries", [])
                main_topic = search_queries[0] if search_queries else "nature landscape"
            
            logger.info(f"   🎯 Main visual focus: '{main_topic}'")
            
            # Search for MORE videos than needed (for quality filtering)
            videos_needed = len(audio_segments)
            videos_to_fetch = videos_needed * 3  # Fetch 3x more for selection
            
            logger.info(f"   📹 Searching for {videos_to_fetch} videos (will select best {videos_needed})...")
            
            # Single search for all videos
            video_pool = self.pexels.search_simple(
                query=main_topic,
                count=videos_to_fetch
            )
            
            if not video_pool:
                logger.error("   ❌ No suitable videos found")
                logger.info(f"   💡 Tip: Check PEXELS_API_KEY or the query '{main_topic}'")
                return None
            
            logger.info(f"   ✅ Found {len(video_pool)} videos for topic: {main_topic}")
            
            # Step 2: SCORE and SELECT best videos
            scored_videos = []
            for vid_id, url in video_pool:
                # Create basic metadata (Pexels API would provide more)
                metadata = {
                    "id": vid_id,
                    "url": url,
                    "duration": 0  # Unknown until downloaded
                }
                score = random.uniform(50, 100)  # Placeholder scoring
                scored_videos.append((score, metadata))
            
            # Sort by score (highest first)
            scored_videos.sort(reverse=True, key=lambda x: x[0])
            
            # Select top N videos
            selected_videos = [meta for score, meta in scored_videos[:videos_needed]]
            
            logger.info(f"   🏆 Selected top {len(selected_videos)} videos (avg score: {sum(s for s,_ in scored_videos[:videos_needed])/len(selected_videos):.1f})")
            
            # Create video pool in expected format
            selected_pool = [(v["id"], v["url"]) for v in selected_videos]
            
            logger.info(f"   📥 Downloading {len(selected_pool)} videos...")
            downloaded = self.downloader.download(
                pool=selected_pool,
                temp_dir=self.temp_dir
            )
            
            if not downloaded:
                logger.error("   ❌ Video download failed - no videos downloaded")
                return None
            
            # Convert downloaded dict to list of paths
            video_files = [path for path in downloaded.values() if isinstance(path, str)]
            
            if not video_files:
                logger.error("   ❌ No valid video file paths after download")
                return None
            
            logger.info(f"   ✅ Ready to process {len(video_files)} video files")
            
            # Step 3: Create video segments with DYNAMIC PACING
            logger.info("   ✂️ Creating video segments with optimal pacing...")
            video_segments = []
            
            # Match videos with audio segments
            for i, audio_segment in enumerate(audio_segments):
                # Cycle through available videos
                video_file = video_files[i % len(video_files)]
                
                if not isinstance(video_file, str):
                    logger.error(f"   ❌ Invalid video file type at index {i}: {type(video_file)}")
                    continue
                
                try:
                    duration = float(audio_segment["duration"])
                    sentence_type = audio_segment.get("type", "buildup")
                    
                    # Get optimal duration range for this sentence type
                    min_dur, max_dur = SHOT_DURATION.get(sentence_type, (2.5, 3.5))
                    
                    # Adjust duration to fit optimal range (but respect audio)
                    # If audio is shorter than min, use audio duration
                    # If audio is longer than max, still use audio (we'll handle pacing with cuts)
                    optimal_duration = max(min_dur, min(duration, max_dur))
                    
                    logger.info(f"   🎬 Segment {i+1} ({sentence_type}): {duration:.1f}s (optimal: {optimal_duration:.1f}s)")
                    
                    segment_path = self.segment_maker.create(
                        video_src=video_file,
                        duration=duration,  # Use actual audio duration
                        temp_dir=self.temp_dir,
                        index=i
                    )
                    
                    if segment_path and os.path.exists(segment_path):
                        video_segments.append(segment_path)
                    else:
                        logger.error(f"   ❌ Segment {i} creation failed")
                        return None
                        
                except Exception as e:
                    logger.error(f"   ❌ Segment {i} error: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    return None
            
            if not video_segments:
                logger.error("   ❌ No video segments created")
                return None
            
            logger.info(f"   ✅ Created {len(video_segments)} video segments with optimal pacing")
            
            # Step 4: Add captions
            logger.info("   📝 Adding captions...")
            captioned_segments = self.caption_renderer.render_captions(
                video_segments=video_segments,
                audio_segments=audio_segments,
                output_dir=self.temp_dir
            )
            
            if not captioned_segments:
                logger.error("   ❌ Caption rendering failed")
                return None
            
            # Step 5: Mux audio with video segments
            logger.info("   🔊 Muxing audio with video segments...")
            final_segments = []
            
            for i, (video_seg, audio_seg) in enumerate(zip(captioned_segments, audio_segments)):
                audio_path = audio_seg["audio_path"]
                output_seg = os.path.join(self.temp_dir, f"final_seg_{i:02d}.mp4")
                
                # Mux video + audio
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", video_seg,
                    "-i", audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest",
                    output_seg
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"   ❌ Audio mux error for segment {i}: {result.stderr}")
                    return None
                
                final_segments.append(output_seg)
            
            logger.info(f"   ✅ Muxed {len(final_segments)} segments with audio")
            
            # Step 6: Add BGM and finalize
            logger.info("   🎵 Adding background music...")
            bgm_path = self.bgm_manager.get_bgm(
                duration=settings.TARGET_DURATION,
                output_dir=self.temp_dir
            )
            
            # Concatenate all segments
            concat_video = os.path.join(self.temp_dir, "concat_video.mp4")
            
            concat_list = os.path.join(self.temp_dir, "concat_list.txt")
            with open(concat_list, "w") as f:
                for segment in final_segments:
                    f.write(f"file '{segment}'\n")
            
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                concat_video
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"   ❌ Concatenation error: {result.stderr}")
                return None
            
            # Final output with or without BGM
            final_video = os.path.join(self.temp_dir, "final_video.mp4")
            
            if bgm_path and os.path.exists(bgm_path):
                # Mix BGM with existing audio
                logger.info("   🎶 Mixing BGM with voice...")
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", concat_video,
                    "-i", bgm_path,
                    "-filter_complex",
                    "[0:a]volume=1.0[voice];[1:a]volume=0.15[bgm];[voice][bgm]amix=inputs=2:duration=shortest[audio]",
                    "-map", "0:v",
                    "-map", "[audio]",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    final_video
                ]
            else:
                # No BGM, just copy
                logger.info("   ⏭️ Skipping BGM (disabled or not found)")
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", concat_video,
                    "-c", "copy",
                    final_video
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"   ❌ Final assembly error: {result.stderr}")
                return None
            
            if not os.path.exists(final_video):
                logger.error("   ❌ Final video not created")
                return None
            
            logger.info(f"   ✅ Video produced: {final_video}")
            return final_video
            
        except Exception as e:
            logger.error(f"   ❌ Video production error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
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
                
                # Register with novelty guard
                self.novelty_guard.register_item(
                    channel=self.channel,
                    title=metadata.get("title", ""),
                    script=" ".join([content.get("hook", "")] + content.get("script", []) + [content.get("cta", "")]),
                    search_term=content.get("search_queries", [""])[0] if content.get("search_queries") else None,
                    topic=settings.CHANNEL_TOPIC,
                    pexels_ids=[]
                )
                
                logger.info(f"   ✅ Uploaded: https://youtube.com/watch?v={video_id}")
                return video_id
            else:
                logger.error("   ❌ Upload failed")
                return None
                
        except Exception as e:
            logger.error(f"   ❌ Upload error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
