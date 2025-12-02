"""
Upload Stage - Uploads video to YouTube.
"""

from ..base import PipelineStage, PipelineContext
from ...core import Result, UploadError
from ...upload.youtube_uploader import YouTubeUploader
from ...state.state_guard import StateGuard
from ...state.novelty_guard import NoveltyGuard
from ...config import settings


class UploadStage(PipelineStage):
    """
    Upload video to YouTube and record state.

    Dependencies:
    - YouTubeUploader: Upload to YouTube
    - StateGuard: Record upload state
    - NoveltyGuard: Register content

    Requires context:
    - video_path: Path to final video
    - content: Content metadata

    Updates context with:
    - video_id: YouTube video ID
    """

    def __init__(
        self,
        uploader: YouTubeUploader,
        state_guard: StateGuard,
        novelty_guard: NoveltyGuard
    ):
        super().__init__("Upload")
        self.uploader = uploader
        self.state_guard = state_guard
        self.novelty_guard = novelty_guard

    def execute(self, context: PipelineContext) -> Result[PipelineContext, str]:
        """Upload video to YouTube."""
        # Check if upload is enabled
        if not settings.UPLOAD_TO_YT:
            self.logger.info("⏭️ Upload skipped (UPLOAD_TO_YT=False)")
            context.video_id = None
            return Result.ok(context)

        # Validate prerequisites
        if not context.video_path:
            return Result.err("No video_path available")

        if not context.content:
            return Result.err("No content metadata available")

        try:
            metadata = context.content["metadata"]

            # Upload
            self.logger.info("📤 Uploading to YouTube...")

            video_id = self.uploader.upload(
                video_path=context.video_path,
                title=metadata.get("title", "Amazing Short"),
                description=metadata.get("description", ""),
                tags=metadata.get("tags", []),
                category_id="22",
                privacy_status=settings.VISIBILITY
            )

            if not video_id:
                return Result.err("Upload failed - no video ID returned")

            # Record state
            self.state_guard.record_upload(video_id, context.content)

            # Register with novelty guard
            self.novelty_guard.register_item(
                channel=context.channel or settings.CHANNEL_NAME,
                title=metadata.get("title", ""),
                script=" ".join([
                    context.content.get("hook", ""),
                    *context.content.get("script", []),
                    context.content.get("cta", "")
                ]),
                search_term=context.content.get("search_queries", [""])[0] if context.content.get("search_queries") else None,
                topic=context.topic or settings.CHANNEL_TOPIC,
                pexels_ids=[]
            )

            # Update context
            context.video_id = video_id

            self.logger.info(f"✅ Uploaded: https://youtube.com/watch?v={video_id}")

            return Result.ok(context)

        except Exception as e:
            self.logger.error(f"Upload error: {e}", exc_info=True)
            return Result.err(f"Upload failed: {str(e)}")
