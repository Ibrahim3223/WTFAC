"""
Pydantic configuration models for type-safe settings.

All environment variables are validated and typed here.
"""

from typing import List, Optional, Union
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseSettings):
    """API keys and external service credentials."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Gemini
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="flash", alias="GEMINI_MODEL")

    # Pexels/Pixabay
    pexels_api_key: str = Field(default="", alias="PEXELS_API_KEY")
    pixabay_api_key: str = Field(default="", alias="PIXABAY_API_KEY")

    # YouTube OAuth
    yt_client_id: str = Field(default="", alias="YT_CLIENT_ID")
    yt_client_secret: str = Field(default="", alias="YT_CLIENT_SECRET")
    yt_refresh_token: str = Field(default="", alias="YT_REFRESH_TOKEN")


class ChannelConfig(BaseSettings):
    """Channel-specific configuration."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    channel_name: str = Field(default="DefaultChannel", alias="CHANNEL_NAME")
    topic: str = Field(default="Interesting facts and knowledge", alias="TOPIC")
    lang: str = Field(default="en", alias="LANG")
    mode: str = Field(default="general", alias="MODE")
    visibility: str = Field(default="public", alias="VISIBILITY")
    content_style: str = Field(
        default="Educational and engaging",
        alias="CONTENT_STYLE"
    )
    rotation_seed: int = Field(default=0, alias="ROTATION_SEED")

    @field_validator("visibility")
    @classmethod
    def validate_visibility(cls, v: str) -> str:
        allowed = {"public", "private", "unlisted"}
        if v not in allowed:
            raise ValueError(f"visibility must be one of {allowed}")
        return v


class VideoConfig(BaseSettings):
    """Video production settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Duration
    target_duration: int = Field(default=30, ge=15, le=60, alias="TARGET_DURATION")
    target_min_sec: float = Field(default=25.0, alias="TARGET_MIN_SEC")
    target_max_sec: float = Field(default=35.0, alias="TARGET_MAX_SEC")

    # Resolution
    video_width: int = Field(default=1080, alias="VIDEO_WIDTH")
    video_height: int = Field(default=1920, alias="VIDEO_HEIGHT")
    target_fps: int = Field(default=30, ge=24, le=60, alias="TARGET_FPS")
    crf_visual: int = Field(default=20, ge=0, le=51, alias="CRF_VISUAL")

    # Motion effects
    video_motion: bool = Field(default=True, alias="VIDEO_MOTION")
    motion_intensity: Union[float, str] = Field(
        default=1.18,
        alias="MOTION_INTENSITY"
    )

    @field_validator("motion_intensity", mode="before")
    @classmethod
    def parse_motion_intensity(cls, v) -> float:
        """Parse motion intensity - accepts float or style string."""
        if isinstance(v, (int, float)):
            return float(v)

        # Map style strings to numeric values
        style_map = {
            "subtle": 1.05,
            "moderate": 1.18,
            "dynamic": 1.35
        }

        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in style_map:
                return style_map[v_lower]
            try:
                return float(v)
            except ValueError:
                raise ValueError(
                    f"motion_intensity must be a number or one of {list(style_map.keys())}"
                )

        return 1.18  # default


class TTSConfig(BaseSettings):
    """Text-to-speech settings."""

    model_config = SettingsConfigDict(
        env_prefix="TTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    voice: str = Field(default="en-US-GuyNeural", alias="TTS_VOICE")
    rate: str = Field(default="+0%", alias="TTS_RATE")
    pitch: str = Field(default="+0Hz", alias="TTS_PITCH")
    style: str = Field(default="narration-professional", alias="TTS_STYLE")
    ssml: bool = Field(default=False, alias="TTS_SSML")


class CaptionConfig(BaseSettings):
    """Caption and karaoke settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    require_captions: bool = Field(default=True, alias="REQUIRE_CAPTIONS")
    karaoke_captions: bool = Field(default=True, alias="KARAOKE_CAPTIONS")
    captions_upper: bool = Field(default=True, alias="CAPTIONS_UPPER")

    # Colors
    karaoke_inactive: str = Field(default="#FFD700", alias="KARAOKE_INACTIVE")
    karaoke_active: str = Field(default="#3EA6FF", alias="KARAOKE_ACTIVE")
    karaoke_outline: str = Field(default="#000000", alias="KARAOKE_OUTLINE")

    # Timing
    karaoke_offset_ms: int = Field(default=0, alias="KARAOKE_OFFSET_MS")
    karaoke_speed: float = Field(default=1.0, alias="KARAOKE_SPEED")
    caption_lead_ms: int = Field(default=0, alias="CAPTION_LEAD_MS")

    # Effects
    karaoke_effects: bool = Field(default=True, alias="KARAOKE_EFFECTS")
    effect_style: str = Field(default="moderate", alias="EFFECT_STYLE")

    @field_validator("effect_style")
    @classmethod
    def validate_effect_style(cls, v: str) -> str:
        allowed = {"subtle", "moderate", "dynamic"}
        if v not in allowed:
            raise ValueError(f"effect_style must be one of {allowed}")
        return v


class PexelsConfig(BaseSettings):
    """Pexels/Pixabay video search settings."""

    model_config = SettingsConfigDict(
        env_prefix="PEXELS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    per_page: int = Field(default=40, ge=1, le=80, alias="PEXELS_PER_PAGE")
    max_uses_per_clip: int = Field(default=1, alias="PEXELS_MAX_USES_PER_CLIP")
    allow_reuse: bool = Field(default=False, alias="PEXELS_ALLOW_REUSE")
    allow_landscape: bool = Field(default=False, alias="PEXELS_ALLOW_LANDSCAPE")
    min_duration: int = Field(default=4, alias="PEXELS_MIN_DURATION")
    max_duration: int = Field(default=12, alias="PEXELS_MAX_DURATION")
    min_height: int = Field(default=1440, alias="PEXELS_MIN_HEIGHT")
    strict_vertical: bool = Field(default=True, alias="PEXELS_STRICT_VERTICAL")
    max_pages: int = Field(default=7, alias="PEXELS_MAX_PAGES")
    deep_search: bool = Field(default=True, alias="PEXELS_DEEP_SEARCH")

    allow_pixabay_fallback: bool = Field(
        default=True,
        alias="ALLOW_PIXABAY_FALLBACK"
    )


class QualityConfig(BaseSettings):
    """Content quality thresholds."""

    model_config = SettingsConfigDict(
        env_prefix="MIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    min_quality_score: float = Field(
        default=6.5,
        ge=0.0,
        le=10.0,
        alias="MIN_QUALITY_SCORE"
    )
    min_viral_score: float = Field(
        default=6.0,
        ge=0.0,
        le=10.0,
        alias="MIN_VIRAL_SCORE"
    )
    min_overall_score: float = Field(
        default=7.0,
        ge=0.0,
        le=10.0,
        alias="MIN_OVERALL_SCORE"
    )


class NoveltyConfig(BaseSettings):
    """Novelty and anti-repeat settings."""

    model_config = SettingsConfigDict(
        env_prefix="NOVELTY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    enforce: bool = Field(default=True, alias="NOVELTY_ENFORCE")
    window: int = Field(default=40, ge=1, alias="NOVELTY_WINDOW")
    jaccard_max: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        alias="NOVELTY_JACCARD_MAX"
    )
    retries: int = Field(default=4, ge=1, alias="NOVELTY_RETRIES")

    entity_cooldown_days: int = Field(
        default=45,
        ge=0,
        alias="ENTITY_COOLDOWN_DAYS"
    )

    state_dir: str = Field(default=".state", alias="STATE_DIR")


class BGMConfig(BaseSettings):
    """Background music settings."""

    model_config = SettingsConfigDict(
        env_prefix="BGM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    enable: bool = Field(default=True, alias="BGM_ENABLE")
    dir: str = Field(default="bgm", alias="BGM_DIR")
    fade: float = Field(default=0.8, alias="BGM_FADE")
    gain_db: int = Field(default=-11, alias="BGM_GAIN_DB")
    duck_thresh: float = Field(default=0.035, alias="BGM_DUCK_THRESH")
    duck_ratio: int = Field(default=10, alias="BGM_DUCK_RATIO")
    duck_attack_ms: int = Field(default=6, alias="BGM_DUCK_ATTACK_MS")
    duck_release_ms: int = Field(default=180, alias="BGM_DUCK_RELEASE_MS")


class AppConfig(BaseSettings):
    """Main application configuration combining all sub-configs."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Upload
    upload_to_yt: bool = Field(default=True, alias="UPLOAD_TO_YT")
    max_generation_attempts: int = Field(
        default=3,
        ge=1,
        alias="MAX_GENERATION_ATTEMPTS"
    )

    # Sub-configs are loaded separately to avoid circular dependencies
    # They will be initialized in __init__

    def __init__(self, **kwargs):
        """Initialize with sub-configs from environment."""
        super().__init__(**kwargs)
        # Load sub-configs after parent initialization
        self.api = APIConfig()
        self.channel = ChannelConfig()
        self.video = VideoConfig()
        self.tts = TTSConfig()
        self.captions = CaptionConfig()
        self.pexels = PexelsConfig()
        self.quality = QualityConfig()
        self.novelty = NoveltyConfig()
        self.bgm = BGMConfig()
