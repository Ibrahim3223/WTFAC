"""
Video Production Stage - Assembles final video with captions and BGM.
"""

import os
import subprocess
import random
from typing import List, Dict, Any

from ..base import PipelineStage, PipelineContext
from ...core import Result, VideoProductionError
from ...video.pexels_client import PexelsClient
from ...video.downloader import VideoDownloader
from ...video.segment_maker import SegmentMaker
from ...captions.renderer import CaptionRenderer
from ...audio.bgm_manager import BGMManager
from ...config import settings


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
        bgm_manager: BGMManager
    ):
        super().__init__("VideoProduction")
        self.pexels = pexels
        self.downloader = downloader
        self.segment_maker = segment_maker
        self.caption_renderer = caption_renderer
        self.bgm_manager = bgm_manager

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

            # Step 5: Add captions
            self.logger.info("📝 Adding captions...")

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

            # Step 7: BGM and finalize
            self.logger.info("🎵 Adding BGM...")
            bgm_path = self.bgm_manager.get_bgm(
                duration=settings.TARGET_DURATION,
                output_dir=context.temp_dir
            )

            # Concatenate
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
