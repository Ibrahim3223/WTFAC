# -*- coding: utf-8 -*-
"""
Constants used across the system - VIRAL OPTIMIZED
Timing parameters based on 100M+ view analysis
"""

# ============================================================================
# VIRAL TIMING CONSTANTS - Optimized for 70%+ retention
# ============================================================================

# Video duration sweet spots (seconds)
VIRAL_DURATION = {
    "min": 18,              # Minimum for shorts
    "optimal_min": 22,      # Sweet spot start
    "optimal_max": 28,      # Sweet spot end
    "max": 35,              # Maximum before retention drops
    "hook_critical": 3      # First 3 seconds = critical
}

# Shot/scene durations by content type (seconds)
SHOT_DURATION = {
    "hook": (1.3, 1.9),         # Ultra-fast, pattern interrupt
    "buildup": (2.2, 3.0),      # Normal pace, build tension
    "payoff": (2.8, 3.5),       # Slightly longer for key info
    "cta": (2.0, 2.5),          # Quick, punchy ending
    "transition": 0.25          # Transition effect duration
}

# Caption timing (seconds per word)
CAPTION_TIMING = {
    "word_min": 0.22,           # Minimum time per word
    "word_optimal": 0.35,       # Optimal reading speed
    "word_max": 0.60,           # Maximum before boredom
    "emphasis_multiplier": 1.3  # Emphasis words stay longer
}

# Audio levels (dB)
AUDIO_LEVELS = {
    "voice_target": -18,        # Voice loudness target
    "bgm_active": -30,          # BGM when voice is speaking
    "bgm_solo": -21,            # BGM during silence/intro
    "fade_duration": 1.0,       # Audio fade in/out (seconds)
}

# BGM ducking (professional podcast style)
BGM_DUCK = {
    "threshold": -25,           # Duck when voice reaches -25dB
    "ratio": 4.0,               # 4:1 compression ratio
    "attack_ms": 10,            # Fast attack
    "release_ms": 250,          # Medium release
    "knee_db": 3.0,             # Soft knee
    "makeup_gain": 1.0          # Makeup gain after ducking
}

# ============================================================================
# VISUAL QUALITY SCORING - For smart video selection
# ============================================================================

# Video scoring weights (total = 100)
VIDEO_SCORE_WEIGHTS = {
    "motion": 30,               # High motion = engaging
    "brightness": 20,           # Bright = better retention
    "saturation": 20,           # Vibrant = eye-catching
    "center_focus": 15,         # Center composition = pro
    "duration_match": 15        # Right length = less cutting
}

# Quality thresholds
VIDEO_QUALITY = {
    "min_resolution": (1280, 720),      # Minimum HD
    "optimal_resolution": (1920, 1080),  # Full HD
    "min_bitrate": 2000,                # Minimum bitrate (kbps)
    "min_fps": 24,                      # Minimum frame rate
    "optimal_fps": 30,                  # Optimal frame rate
    "max_dark_ratio": 0.25              # Max 25% dark pixels
}

# ============================================================================
# CONTENT FRESHNESS - Novelty guard thresholds
# ============================================================================

# Similarity thresholds (lower = stricter)
NOVELTY_THRESHOLDS = {
    "simhash_hamming_max": 8,       # Max Hamming distance for simhash
    "entity_jaccard_max": 0.58,     # Max entity overlap (was 0.60)
    "embed_cosine_max": 0.86,       # Max embedding similarity (was 0.88)
    "title_similarity_max": 0.70,   # Max title similarity
}

# Cooldown periods (days)
COOLDOWN_DAYS = {
    "search_term": 30,              # Same search term
    "entity": 45,                   # Same entity (person/place)
    "topic_category": 14,           # Same broad topic
    "visual_id": 60,                # Same Pexels video ID
}

# ============================================================================
# CAPTION STYLING - Modern, viral-optimized
# ============================================================================

# Caption positioning (pixels from bottom)
CAPTION_POSITION = {
    "hook": 270,                # Hook closer to center
    "normal": 330,              # Normal position
    "cta": 350,                 # CTA at bottom
}

# Caption colors (ASS format: &H00BBGGRR)
CAPTION_COLORS = {
    "white": "&H00FFFFFF",
    "yellow": "&H0000FFFF",
    "orange": "&H000099FF",
    "cyan": "&H00FFFF00",
    "magenta": "&H00FF00FF",
    "black": "&H00000000",
    "gray": "&H00AAAAAA",
}

# Caption effects
CAPTION_EFFECTS = {
    "outline_width": 4,         # Outline thickness
    "shadow_offset": 3,         # Shadow distance
    "blur_amount": 2,           # Blur radius
    "glow_intensity": 0.3,      # Glow strength
}

# ============================================================================
# CONTENT GENERATION - Gemini parameters
# ============================================================================

# Hook patterns distribution (A/B test weights)
HOOK_PATTERN_WEIGHTS = {
    "curiosity_gap": 0.35,      # "WAIT—" pattern
    "pattern_interrupt": 0.25,   # "STOP scrolling"
    "shocking_stat": 0.20,      # "97% don't know"
    "story_hook": 0.15,         # "In 1847..."
    "controversial": 0.05       # "Unpopular opinion"
}

# Storytelling angles distribution
STORYTELLING_ANGLES = [
    "historical_origin",
    "scientific_explanation",
    "hidden_secret",
    "future_prediction",
    "comparison",
    "behind_the_scenes",
    "myth_busting",
    "personal_impact",
    "extreme_example"
]

# Content quality thresholds
QUALITY_THRESHOLDS = {
    "min_overall_score": 0.65,  # Minimum quality score
    "min_viral_score": 0.60,    # Minimum viral potential
    "min_retention_score": 0.65, # Minimum retention score
    "max_generic_ratio": 0.20,  # Max 20% generic words
}

# ============================================================================
# GENERIC/BANNED TERMS - Filter out low-quality content
# ============================================================================

GENERIC_SKIP = {
    "country", "countries", "people", "history", "stories", "story", 
    "facts", "fact", "amazing", "weird", "random", "culture", "cultural",
    "animal", "animals", "nature", "wild", "pattern", "patterns", 
    "science", "eco", "habit", "habits", "waste", "tip", "tips", 
    "daily", "news", "world", "today", "minute", "short", "video", 
    "watch", "more", "better", "twist", "comment", "voice", "narration", 
    "hook", "topic", "secret", "secrets", "unknown", "things", "life", 
    "lived", "modern", "time", "times", "explained", "guide", "quick", 
    "fix", "fixes", "color", "colors", "skin", "cells", "cell", 
    "temperature", "light", "lights", "effect", "effects", "land", 
    "nation", "state", "city", "capital", "border", "flag", "heritage",
    "travel", "tourism", "planet", "earth", "place", "region", "area"
}

GENERIC_BAD = {
    "great", "good", "bad", "big", "small", "old", "new", "many", 
    "more", "most", "thing", "things", "stuff", "once", "next", 
    "feature", "features", "precisely", "signal", "signals", 
    "masters", "master", "ways", "way", "track", "tracks", 
    "uncover", "gripping", "limb", "emotion", "emotions"
}

# Banned phrases that reduce retention
BANNED_PHRASES = [
    "one clear tip", "see it", "learn it", "plot twist",
    "soap-opera narration", "repeat once", "takeaway action",
    "in 60 seconds", "just the point", "crisp beats",
    "sum it up", "watch till the end", "mind-blowing fact",
    "you won't believe", "wait for it", "mind blown",
    "this will shock you", "number will surprise",
    "keep watching", "stick around", "coming up",
    "before we start", "let me tell you", "today we'll",
    "in this video", "make sure to", "don't forget to"
]

# ============================================================================
# STOPWORDS - Filter for content analysis
# ============================================================================

STOP_EN = {
    "the", "a", "an", "and", "or", "but", "if", "while", "of", "to", 
    "in", "on", "at", "from", "by", "with", "for", "about", "into", 
    "over", "after", "before", "between", "during", "under", "above", 
    "across", "around", "through", "this", "that", "these", "those", 
    "is", "are", "was", "were", "be", "been", "being", "have", "has", 
    "had", "do", "does", "did", "can", "could", "should", "would", 
    "may", "might", "will", "your", "you", "we", "our", "they", 
    "their", "he", "she", "it", "its", "as", "than", "then", "so", 
    "very", "more", "most", "many", "much", "just", "also", "only", 
    "even", "still", "yet"
}

STOP_TR = {
    "ve", "ya", "ama", "eğer", "iken", "ile", "için", "üzerine", 
    "altında", "üzerinde", "arasında", "boyunca", "sonra", "önce", 
    "altında", "üstünde", "hakkında", "üzerinden", "bu", "şu", "o", 
    "bir", "birisi", "şunlar", "bunlar", "biz", "siz", "onlar", "var", 
    "yok", "çok", "daha", "en", "ise", "çünkü", "gibi", "kadar", 
    "zaten", "sadece", "yine", "hâlâ"
}

# ============================================================================
# COUNTRY/ENTITY LISTS - For entity extraction
# ============================================================================

COUNTRIES = {
    "japan", "usa", "united states", "china", "india", "russia", 
    "germany", "france", "uk", "united kingdom", "turkey", "türkiye",
    "brazil", "argentina", "mexico", "spain", "italy", "canada", 
    "australia", "south korea", "north korea", "sweden", "norway",
    "finland", "denmark", "netherlands", "belgium", "poland", "czech", 
    "greece", "egypt", "saudi arabia", "uae", "iran", "iraq",
    "pakistan", "bangladesh", "indonesia", "malaysia", "singapore", 
    "thailand", "vietnam", "philippines", "south africa",
    "nigeria", "ethiopia", "kenya", "morocco", "algeria", "tunisia", 
    "colombia", "chile", "peru"
}

# ============================================================================
# PERFORMANCE TARGETS - Goals for the system
# ============================================================================

PERFORMANCE_TARGETS = {
    "retention_70_percent": 0.70,   # 70%+ watch through rate
    "ctr_min": 0.05,                # 5%+ click-through rate
    "engagement_rate": 0.08,        # 8%+ engagement (likes/comments/shares)
    "rewatch_rate": 0.15,           # 15%+ rewatch rate
}

# ============================================================================
# A/B TESTING - Experiment configurations
# ============================================================================

AB_TEST_CONFIG = {
    "hook_patterns": {
        "enabled": True,
        "variants": 5,              # Number of hook patterns to test
        "sample_size": 20           # Videos per variant before decision
    },
    "caption_styles": {
        "enabled": True,
        "variants": 3,              # Number of caption styles
        "sample_size": 15
    },
    "bgm_volume": {
        "enabled": False,           # Enable when stable
        "variants": 3,
        "levels": [-30, -27, -24]   # BGM dB levels to test
    }
}

# ============================================================================
# SYSTEM LIMITS - Safety and resource management
# ============================================================================

SYSTEM_LIMITS = {
    "max_generation_attempts": 5,   # Max retries per video
    "max_video_search_results": 50, # Max videos to consider
    "max_download_size_mb": 200,    # Max video file size
    "temp_dir_cleanup_hours": 24,   # Clean temp files older than
    "max_concurrent_channels": 10,  # Max parallel channel processing
}
