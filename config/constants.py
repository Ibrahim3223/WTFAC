# -*- coding: utf-8 -*-
"""
Constants used across the system.
Stopwords, generic terms, country names, etc.
"""

# ==================== Generic Skip Terms ====================

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

# ==================== Stopwords ====================

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

# ==================== Banned Phrases ====================

BANNED_PHRASES = [
    "one clear tip", "see it", "learn it", "plot twist",
    "soap-opera narration", "repeat once", "takeaway action",
    "in 60 seconds", "just the point", "crisp beats",
    "sum it up", "watch till the end", "mind-blowing fact",
    "you won't believe", "wait for it", "mind blown",
    "this will shock you", "number will surprise",
    "keep watching", "stick around", "coming up",
]

# ==================== Caption Colors ====================

CAPTION_COLORS = [
    "0xFFD700", "0xFF6B35", "0x00F5FF", "0x32CD32", 
    "0xFF1493", "0x1E90FF", "0xFFA500", "0xFF69B4"
]
