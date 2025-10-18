# -*- coding: utf-8 -*-
"""YouTube upload handler."""
import logging
from typing import Dict, Any, List, Optional

from autoshorts.config import settings

logger = logging.getLogger(__name__)


class YouTubeUploader:
    """Handle YouTube API uploads."""
    
    def __init__(self):
        """Initialize with credentials."""
        if not all([settings.YT_CLIENT_ID, settings.YT_CLIENT_SECRET, settings.YT_REFRESH_TOKEN]):
            raise ValueError("YouTube credentials missing")
        
        self.client_id = settings.YT_CLIENT_ID
        self.client_secret = settings.YT_CLIENT_SECRET
        self.refresh_token = settings.YT_REFRESH_TOKEN
    
    def upload(
        self,
        video_path: str,
        title: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        category_id: str = "22",
        privacy_status: str = "public"
    ) -> str:
        """
        Upload video to YouTube.
        
        Args:
            video_path: Path to video file
            title: Video title
            description: Video description
            tags: List of tags
            category_id: YouTube category ID (22 = People & Blogs)
            privacy_status: public, unlisted, or private
            
        Returns:
            Video ID or empty string on failure
        """
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            
            # Refresh credentials
            creds = Credentials(
                token=None,
                refresh_token=self.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=["https://www.googleapis.com/auth/youtube.upload"]
            )
            
            creds.refresh(Request())
            
            # Build YouTube service
            youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
            
            # Prepare body
            body = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags or [],
                    "categoryId": category_id,
                    "defaultLanguage": settings.LANG,
                    "defaultAudioLanguage": settings.LANG
                },
                "status": {
                    "privacyStatus": privacy_status,
                    "selfDeclaredMadeForKids": False
                }
            }
            
            # Upload
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
            
            request = youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media
            )
            
            response = request.execute()
            
            video_id = response.get("id", "")
            logger.info(f"   ✅ Uploaded: https://youtube.com/watch?v={video_id}")
            
            return video_id
            
        except Exception as e:
            logger.error(f"   ❌ Upload failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return ""
