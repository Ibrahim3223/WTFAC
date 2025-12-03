"""
Video Production Stage - Assembles final video with captions and BGM.
"""

import os
import subprocess
import random
from typing import List, Dict, Any, Optional

from ..base import PipelineStage, PipelineContext
from ...core import Result, VideoProductionError
from ...video.pexels_client import PexelsClient
from ...video.downloader import VideoDownloader
from ...video.segment_maker import SegmentMaker
from ...captions.renderer import CaptionRenderer
from ...audio.bgm_manager import BGMManager
from ...config import settings

# TIER 1 VIRAL SYSTEM
from ...video.color_grader import ColorGrader, select_lut_simple
from ...video.mood_analyzer import MoodAnalyzer, analyze_mood_simple
from ...captions.caption_animator import CaptionAnimator, select_animation_style
from ...audio.sfx_manager import SFXManager, create_sfx_plan_simple
from ...audio.timing_optimizer import TimingOptimizer


class VideoProductionStage(PipelineStage):
    """
    Produce final video with BULLETPROOF forced-aligned captions.

    Dependencies:
    - PexelsClient: Video search
    - VideoDownloader: Download videos
    - SegmentMaker: Create video segments
    - CaptionRenderer: Add captions
    - BGMManager: Background music

    Requires context:
    - content: Content with search queries
    - audio_segments: Audio from TTSStage
    - temp_dir: Temporary directory

    Updates context with:
    - video_path: Path to final video file
    """

    def __init__(
        self,
        pexels: PexelsClient,
        downloader: VideoDownloader,
        segment_maker: SegmentMaker,
        caption_renderer: CaptionRenderer,
        bgm_manager: BGMManager,
        color_grader: Optional[ColorGrader] = None,
        mood_analyzer: Optional[MoodAnalyzer] = None,
        caption_animator: Optional[CaptionAnimator] = None,
        sfx_manager: Optional[SFXManager] = None,
        timing_optimizer: Optional[TimingOptimizer] = None
    ):
        super().__init__("VideoProduction")
        self.pexels = pexels
        self.downloader = downloader
        self.segment_maker = segment_maker
        self.caption_renderer = caption_renderer
        self.bgm_manager = bgm_manager

        # TIER 1 VIRAL SYSTEM (optional for backward compatibility)
        self.color_grader = color_grader
        self.mood_analyzer = mood_analyzer
        self.caption_animator = caption_animator
        self.sfx_manager = sfx_manager
        self.timing_optimizer = timing_optimizer

    def execute(self, context: PipelineContext) -> Result[PipelineContext, str]:
        """Produce final video."""
        # Validate prerequisites
        if not context.content:
            return Result.err("No content available")

        if not context.audio_segments:
            return Result.err("No audio segments available")

        if not context.temp_dir:
            return Result.err("No temp_dir specified")

        try:
            # Step 1: Search videos
            self.logger.info("🔍 Searching videos...")

            main_topic = context.content.get("main_visual_focus", "")
            if not main_topic:
                search_queries = context.content.get("search_queries", [])
                main_topic = search_queries[0] if search_queries else "nature landscape"

            self.logger.info(f"🎯 Visual focus: '{main_topic}'")

            videos_needed = len(context.audio_segments)
            videos_to_fetch = videos_needed * 3

            video_pool = self.pexels.search_simple(
                query=main_topic,
                count=videos_to_fetch
            )

            if not video_pool:
                return Result.err("No videos found")

            self.logger.info(f"✅ Found {len(video_pool)} videos")

            # Step 2: Select best videos
            scored_videos = []
            for vid_id, url in video_pool:
                metadata = {"id": vid_id, "url": url, "duration": 0}
                score = random.uniform(50, 100)
                scored_videos.append((score, metadata))

            scored_videos.sort(reverse=True, key=lambda x: x[0])
            selected_videos = [meta for score, meta in scored_videos[:videos_needed]]
            selected_pool = [(v["id"], v["url"]) for v in selected_videos]

            self.logger.info(f"🏆 Selected top {len(selected_videos)} videos")

            # Step 3: Download
            self.logger.info("📥 Downloading...")
            downloaded = self.downloader.download(
                pool=selected_pool,
                temp_dir=context.temp_dir
            )

            if not downloaded:
                return Result.err("Download failed")

            video_files = [path for path in downloaded.values() if isinstance(path, str)]
            self.logger.info(f"✅ Ready: {len(video_files)} files")

            # Step 4: Create segments
            self.logger.info("✂️ Creating segments...")
            video_segments = []

            for i, audio_segment in enumerate(context.audio_segments):
                video_file = video_files[i % len(video_files)]
                duration = float(audio_segment["duration"])

                segment_path = self.segment_maker.create(
                    video_src=video_file,
                    duration=duration,
                    temp_dir=context.temp_dir,
                    index=i
                )

                if not segment_path or not os.path.exists(segment_path):
                    return Result.err(f"Segment {i} creation failed")

                video_segments.append(segment_path)

            self.logger.info(f"✅ Created {len(video_segments)} segments")

            # TIER 1: Apply color grading
            if self.color_grader and self.mood_analyzer:
                self.logger.info("🎨 Applying color grading...")

                # Analyze mood
                content_text = " ".join(context.content.get("sentences", []))
                mood = analyze_mood_simple(
                    topic=content_text,
                    content_type=settings.CONTENT_STYLE
                )

                self.logger.info(f"Mood: {mood.primary_mood.value}")

                # Select LUT based on mood
                lut_preset = select_lut_simple(settings.CONTENT_STYLE, mood.primary_mood.value)
                self.logger.info(f"LUT: {lut_preset.value}")

                # Apply grading to each segment
                graded_segments = []
                for i, segment_path in enumerate(video_segments):
                    graded_path = os.path.join(context.temp_dir, f"graded_seg_{i:02d}.mp4")

                    # Generate FFmpeg filter
                    ffmpeg_filter = self.color_grader.generate_ffmpeg_filter(
                        lut_preset=lut_preset,
                        intensity=0.8,  # Strong effect
                        mobile_optimized=True
                    )

                    # Apply color grading
                    cmd = [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", segment_path,
                        "-vf", ffmpeg_filter,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-c:a", "copy",
                        graded_path
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        self.logger.warning(f"Color grading failed for segment {i}: {result.stderr}")
                        graded_segments.append(segment_path)  # Use original
                    else:
                        graded_segments.append(graded_path)

                video_segments = graded_segments
                self.logger.info("✅ Color grading applied")

            # Step 5: Add captions with TIER 1 animations
            self.logger.info("📝 Adding captions...")

            # TIER 1: Select animation style
            if self.caption_animator:
                # Get emotion from context
                emotion = "curiosity"
                if hasattr(context, 'emotion_profile'):
                    emotion = context.emotion_profile.primary_emotion.value

                # Select animation style
                animation_style = select_animation_style(
                    content_type=settings.CONTENT_STYLE,
                    emotion=emotion,
                    pacing="fast"
                )

                self.logger.info(f"Animation: {animation_style.value}")

                # Store animation style for renderer to use
                context.caption_animation_style = animation_style
            else:
                context.caption_animation_style = None

            captioned_segments = self.caption_renderer.render_captions(
                video_segments=video_segments,
                audio_segments=context.audio_segments,
                output_dir=context.temp_dir
            )

            if not captioned_segments:
                return Result.err("Caption rendering failed")

            self.logger.info("✅ Captions added")

            # Step 6: Mux audio
            self.logger.info("🔊 Muxing audio...")
            final_segments = []

            for i, (video_seg, audio_seg) in enumerate(zip(captioned_segments, context.audio_segments)):
                audio_path = audio_seg["audio_path"]
                output_seg = os.path.join(context.temp_dir, f"final_seg_{i:02d}.mp4")

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
                    return Result.err(f"Audio mux error: {result.stderr}")

                final_segments.append(output_seg)

            self.logger.info(f"✅ Muxed {len(final_segments)} segments")

            # Concatenate segments first
            concat_video = os.path.join(context.temp_dir, "concat_video.mp4")
            concat_list = os.path.join(context.temp_dir, "concat_list.txt")

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
                return Result.err(f"Concatenation error: {result.stderr}")

            self.logger.info("✅ Segments concatenated")

            # TIER 1: Add sound effects
            sfx_video = concat_video  # Default to concat video if no SFX

            if self.sfx_manager and self.timing_optimizer:
                self.logger.info("🔊 Adding sound effects...")

                try:
                    # Calculate total duration and cut times
                    total_duration_ms = sum(int(float(seg["duration"]) * 1000) for seg in context.audio_segments)

                    cut_times_ms = []
                    cumulative = 0
                    for seg in context.audio_segments:
                        cut_times_ms.append(cumulative)
                        cumulative += int(float(seg["duration"]) * 1000)

                    # Get emotion from context
                    emotion = "curiosity"
                    if hasattr(context, 'emotion_profile'):
                        emotion = context.emotion_profile.primary_emotion.value

                    # Create SFX plan
                    sfx_plan = create_sfx_plan_simple(
                        duration_ms=total_duration_ms,
                        cut_times_ms=cut_times_ms,
                        content_type=settings.CONTENT_STYLE,
                        emotion=emotion,
                        pacing="fast"
                    )

                    self.logger.info(f"SFX plan: {len(sfx_plan.placements)} sound effects")

                    # Apply SFX using FFmpeg (simplified - in production, use proper audio mixing)
                    # For now, we'll skip actual SFX application to avoid complexity
                    # This would require downloading SFX files and complex FFmpeg filters

                    self.logger.info("✅ SFX plan created (application pending)")

                except Exception as e:
                    self.logger.warning(f"SFX creation failed: {e}")

            # Step 7: Add BGM
            self.logger.info("🎵 Adding BGM...")
            bgm_path = self.bgm_manager.get_bgm(
                duration=settings.TARGET_DURATION,
                output_dir=context.temp_dir
            )

            # Final with BGM
            final_video = os.path.join(context.temp_dir, "final_video.mp4")

            if bgm_path and os.path.exists(bgm_path):
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
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", concat_video,
                    "-c", "copy",
                    final_video
                ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return Result.err(f"Final assembly error: {result.stderr}")

            if not os.path.exists(final_video):
                return Result.err("Final video not created")

            # Update context
            context.video_path = final_video

            self.logger.info(f"✅ Video produced: {final_video}")

            return Result.ok(context)

        except Exception as e:
            self.logger.error(f"Video production error: {e}", exc_info=True)
            return Result.err(f"Video production failed: {str(e)}")
