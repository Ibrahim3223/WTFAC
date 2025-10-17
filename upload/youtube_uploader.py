# -*- coding: utf-8 -*-
"""YouTube upload handler."""
from typing import Dict, Any

from autoshorts.config import settings

class YouTubeUploader:
    """Handle YouTube API uploads."""
    
    def __init__(self):
        """Initialize with credentials."""
        if not all([settings.YT_CLIENT_ID, settings.YT_CLIENT_SECRET, settings.YT_REFRESH_TOKEN]):
            raise ValueError("YouTube credentials missing")
        
        self.client_id = settings.YT_CLIENT_ID
        self.client_secret = settings.YT_CLIENT_SECRET
        self.refresh_token = settings.YT_REFRESH_TOKEN
    
    def upload(self, video_path: str, metadata: Dict[str, Any]) -> str:
        """Upload video to YouTube."""
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
                "title": metadata["title"],
                "description": metadata["description"],
                "tags": metadata.get("tags", []),
                "categoryId": "27",
                "defaultLanguage": metadata.get("defaultLanguage", settings.LANG),
                "defaultAudioLanguage": metadata.get("defaultAudioLanguage", settings.LANG)
            },
            "status": {
                "privacyStatus": metadata.get("privacy", settings.VISIBILITY),
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
        
        return response.get("id", "")
