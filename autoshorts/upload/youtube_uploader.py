# -*- coding: utf-8 -*-
"""
YouTube Upload Handler - TOPIC-DRIVEN SEO OPTIMIZATION
Smart category detection from content, no hardcoded modes
"""
import logging
import re
from typing import Dict, Any, List, Optional
from autoshorts.config import settings

logger = logging.getLogger(__name__)


class YouTubeUploader:
    """Smart SEO-optimized YouTube uploader"""
    
    # YouTube category IDs
    CATEGORIES = {
        "education": "27",
        "people_blogs": "22",
        "entertainment": "24",
        "howto_style": "26",
        "science_tech": "28",
        "news_politics": "25",
        "comedy": "23",
        "sports": "17",
        "gaming": "20",
        "travel": "19",
        "pets_animals": "15"
    }
    
    def __init__(self):
        """Initialize with credentials"""
        if not all([settings.YT_CLIENT_ID, settings.YT_CLIENT_SECRET, settings.YT_REFRESH_TOKEN]):
            raise ValueError("YouTube credentials missing")
        
        self.client_id = settings.YT_CLIENT_ID
        self.client_secret = settings.YT_CLIENT_SECRET
        self.refresh_token = settings.YT_REFRESH_TOKEN
        
        logger.info("[YouTube] Uploader initialized")
    
    def upload(
        self,
        video_path: str,
        title: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        category_id: str = "22",
        privacy_status: str = "public",
        topic: Optional[str] = None
    ) -> str:
        """
        Upload video with smart SEO optimization.
        
        Args:
            video_path: Path to video file
            title: Video title (from Gemini)
            description: Video description (from Gemini)
            tags: List of tags
            category_id: Default category (will be auto-detected if topic provided)
            privacy_status: public, unlisted, or private
            topic: Channel topic for smart category detection
            
        Returns:
            Video ID or empty string on failure
        """
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            
            logger.info(f"[YouTube] Preparing: {title[:50]}...")
            
            # Smart optimization
            optimized_title = self._optimize_title(title)
            optimized_description = self._optimize_description(description, tags, title)
            optimized_tags = self._optimize_tags(tags)
            smart_category = self._detect_category(topic, title, description) if topic else category_id
            
            # Debug logging
            logger.info(f"[YouTube] Title length: {len(optimized_title)}")
            logger.info(f"[YouTube] Description length: {len(optimized_description)}")
            logger.info(f"[YouTube] Description preview: {optimized_description[:100]}...")
            
            # Validate before upload
            if not optimized_title or len(optimized_title) < 1:
                raise ValueError("Title is empty after optimization")
            if not optimized_description or len(optimized_description) < 1:
                raise ValueError("Description is empty after optimization")
            if len(optimized_description) > 5000:
                optimized_description = optimized_description[:5000]
                logger.warning(f"[YouTube] Description truncated to 5000 chars")
            
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
            
            # Build service
            youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
            
            # Prepare body
            body = {
                "snippet": {
                    "title": optimized_title,
                    "description": optimized_description,
                    "tags": optimized_tags,
                    "categoryId": smart_category,
                    "defaultLanguage": settings.LANG,
                    "defaultAudioLanguage": settings.LANG
                },
                "status": {
                    "privacyStatus": privacy_status,
                    "selfDeclaredMadeForKids": False,
                    "madeForKids": False
                }
            }
            
            logger.info(f"[YouTube] Title: {optimized_title}")
            logger.info(f"[YouTube] Category: {smart_category}")
            logger.info(f"[YouTube] Tags: {len(optimized_tags)}")
            
            # Upload
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
            
            request = youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media
            )
            
            response = request.execute()
            
            video_id = response.get("id", "")
            video_url = f"https://youtube.com/watch?v={video_id}"
            
            logger.info(f"[YouTube] ✅ Upload successful!")
            logger.info(f"[YouTube] 🔗 {video_url}")
            
            return video_id
            
        except Exception as e:
            logger.error(f"[YouTube] ❌ Upload failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return ""
    
    def _detect_category(self, topic: str, title: str, description: str) -> str:
        """
        Smart category detection from topic/content.
        
        Args:
            topic: Channel topic
            title: Video title
            description: Video description
            
        Returns:
            Best category ID
        """
        # Combine all text for analysis
        text = f"{topic} {title} {description}".lower()
        
        # Category detection patterns
        patterns = {
            "27": [  # Education
                "fact", "learn", "explain", "teach", "science", "history", 
                "geography", "knowledge", "educational", "study", "discover"
            ],
            "24": [  # Entertainment
                "story", "tale", "imagine", "movie", "film", "celebrity",
                "drama", "episode", "character", "plot", "entertainment"
            ],
            "26": [  # Howto & Style
                "how to", "fix", "repair", "diy", "tutorial", "guide",
                "tips", "tricks", "hack", "step by step", "learn to"
            ],
            "25": [  # News & Politics
                "news", "update", "daily", "latest", "breaking", "current",
                "today", "headlines", "briefing", "report"
            ],
            "28": [  # Science & Tech
                "tech", "ai", "robot", "future", "innovation", "gadget",
                "computer", "digital", "space", "research", "science"
            ],
            "23": [  # Comedy
                "funny", "comedy", "humor", "joke", "laugh", "hilarious",
                "satire", "parody", "comedic"
            ],
            "17": [  # Sports
                "sport", "game", "player", "team", "match", "cricket",
                "football", "athlete", "training", "score"
            ],
            "19": [  # Travel & Events
                "travel", "country", "place", "destination", "city",
                "world", "explore", "visit", "journey", "trip"
            ],
            "15": [  # Pets & Animals
                "animal", "pet", "wildlife", "creature", "species",
                "dog", "cat", "bird", "nature documentary"
            ],
            "22": [  # People & Blogs (default)
                "life", "personal", "vlog", "daily", "motivation",
                "inspire", "story", "experience", "blog"
            ]
        }
        
        # Count matches for each category
        scores = {}
        for category_id, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[category_id] = score
        
        # Get best match
        best_category = max(scores.items(), key=lambda x: x[1])
        
        # If no strong match, default to Education or People & Blogs
        if best_category[1] == 0:
            # Check if it's more educational or personal
            if any(word in text for word in ["fact", "learn", "explain", "teach"]):
                return "27"  # Education
            return "22"  # People & Blogs
        
        logger.info(f"[YouTube] Auto-detected category: {best_category[0]} (score: {best_category[1]})")
        return best_category[0]
    
    def _optimize_title(self, title: str) -> str:
        """
        Optimize title for YouTube algorithm.
        
        Strategy:
        - Keep under 70 chars (mobile preview)
        - Front-load keywords
        - Clean excessive punctuation
        - Strategic capitalization
        - YouTube-safe characters only
        """
        if not title:
            return "Untitled Short"
        
        # Sanitize first
        title = self._sanitize_text(title)
        
        # Trim to 70 chars
        if len(title) > 70:
            title = title[:67] + "..."
        
        # Clean excessive punctuation
        title = title.replace("!!", "!").replace("??", "?").replace("...", "...")
        
        # Remove any emojis from title (YouTube API sometimes has issues)
        title = re.sub(r'[^\x00-\x7F\u0080-\uFFFF]+', '', title)
        
        # Smart capitalization
        words = title.split()
        optimized_words = []
        
        for word in words:
            # Preserve ALL CAPS (WAIT, STOP, etc.)
            if word.isupper() and len(word) > 1:
                optimized_words.append(word)
            # Preserve hashtags
            elif word.startswith('#'):
                optimized_words.append(word)
            # Title case for others
            else:
                optimized_words.append(word.capitalize() if word[0].islower() else word)
        
        result = " ".join(optimized_words)
        
        # Final cleanup
        result = result.strip()
        
        return result if result else "Untitled Short"
    
    def _optimize_description(
        self, 
        description: str, 
        tags: Optional[List[str]] = None,
        title: str = ""
    ) -> str:
        """
        Build SEO-rich description with YouTube-safe formatting.
        
        Strategy:
        - First 157 chars = mobile preview (CRITICAL)
        - Natural keyword inclusion
        - Strategic hashtags
        - Engagement CTAs
        - Under 5000 chars total
        - YouTube-safe characters only
        """
        if not description:
            description = f"{title}\n\nWatch this amazing Short!"
        
        # Clean description of problematic characters
        description = self._sanitize_text(description)
        
        lines = []
        
        # Main description (from Gemini)
        lines.append(description.strip())
        
        # Engagement CTAs
        lines.append("")
        lines.append("Drop your thoughts below!")
        lines.append("Like if this surprised you!")
        lines.append("Follow for daily content!")
        
        # Strategic hashtags (top 5 tags)
        if tags:
            hashtags = []
            for tag in tags[:5]:
                # Clean tag for hashtag format
                clean_tag = re.sub(r'[^a-zA-Z0-9]', '', tag)
                if clean_tag and len(clean_tag) > 2:
                    hashtags.append(f"#{clean_tag}")
            
            if hashtags:
                lines.append("")
                lines.append(" ".join(hashtags[:5]))  # Max 5 hashtags
        
        # Universal hashtags
        lines.append("")
        lines.append("#Shorts #Viral #Trending #ForYou")
        
        # Watch time indicator
        lines.append("")
        lines.append("Quick watch: Under 1 minute")
        
        # Attribution
        lines.append("")
        lines.append("Footage: Pexels/Pixabay")
        lines.append("Music: Licensed")
        
        # Join
        full_description = "\n".join(lines)
        
        # Final sanitization
        full_description = self._sanitize_text(full_description)
        
        # Ensure under 5000 char limit
        if len(full_description) > 4900:
            full_description = full_description[:4900]
        
        # Remove any trailing newlines
        full_description = full_description.strip()
        
        return full_description
    
    def _sanitize_text(self, text: str) -> str:
        """
        Remove or replace characters that YouTube doesn't accept.
        
        Args:
            text: Raw text
            
        Returns:
            YouTube-safe text
        """
        if not text:
            return ""
        
        # Remove zero-width characters and other invisible chars
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', text)
        
        # Remove emojis that might cause issues (keep common ones)
        # YouTube accepts most emojis, but some cause encoding issues
        
        # Replace problematic quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Replace em dash and en dash with regular dash
        text = text.replace('—', '-').replace('–', '-')
        
        # Replace ellipsis character with three dots
        text = text.replace('…', '...')
        
        # Remove any null bytes
        text = text.replace('\x00', '')
        
        # Ensure proper line breaks (YouTube accepts \n)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive newlines (max 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove excessive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    def _optimize_tags(self, tags: Optional[List[str]] = None) -> List[str]:
        """
        Optimize tags for discoverability.
        
        Strategy:
        - Mix broad and specific
        - Include variations
        - Prioritize by importance
        - Stay under 500 chars
        """
        if not tags:
            tags = []
        
        optimized = []
        seen = set()
        
        # Deduplicate
        for tag in tags:
            clean = tag.strip().lower()
            if clean and clean not in seen:
                optimized.append(tag.strip())
                seen.add(clean)
        
        # Essential base tags
        base_tags = [
            "shorts",
            "viral",
            "trending",
            "youtube shorts",
            "short video",
            "fyp"
        ]
        
        for base in base_tags:
            if base not in seen:
                optimized.append(base)
                seen.add(base)
        
        # Add current year
        import datetime
        year = str(datetime.datetime.now().year)
        if year not in seen:
            optimized.append(year)
        
        # Limit to 25 tags
        optimized = optimized[:25]
        
        # Verify 500 char limit
        total_chars = sum(len(tag) for tag in optimized)
        while total_chars > 480 and len(optimized) > 5:
            removed = optimized.pop()
            total_chars -= len(removed)
        
        return optimized
    
    def get_upload_stats(self) -> Dict[str, Any]:
        """Get channel statistics"""
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            
            creds = Credentials(
                token=None,
                refresh_token=self.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=["https://www.googleapis.com/auth/youtube.readonly"]
            )
            
            creds.refresh(Request())
            youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
            
            request = youtube.channels().list(
                part="statistics,contentDetails",
                mine=True
            )
            
            response = request.execute()
            
            if "items" in response and response["items"]:
                stats = response["items"][0]["statistics"]
                return {
                    "subscribers": int(stats.get("subscriberCount", 0)),
                    "total_views": int(stats.get("viewCount", 0)),
                    "total_videos": int(stats.get("videoCount", 0))
                }
            
            return {}
            
        except Exception as e:
            logger.warning(f"[YouTube] Could not fetch stats: {e}")
            return {}
