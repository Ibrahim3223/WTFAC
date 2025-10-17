# -*- coding: utf-8 -*-
"""
Text processing utilities: normalize, clean, tokenize.
"""
import re
from typing import List, Set
from autoshorts.config.constants import GENERIC_SKIP, STOP_EN, STOP_TR

def normalize_sentence(raw: str) -> str:
    """Normalize sentence: whitespace, unicode, punctuation."""
    s = (raw or "").strip()
    s = s.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(re.sub(r"\s+", " ", ln).strip() for ln in s.split("\n"))
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace(""", '"').replace(""", '"').replace("'", "'")
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    return s

def clean_caption_text(s: str) -> str:
    """Clean text for captions: normalize + capitalize."""
    t = (s or "").strip()
    t = t.replace("—", "-").replace("–", "-")
    t = t.replace(""", '"').replace(""", '"')
    t = t.replace("'", "'").replace("`", "")
    t = re.sub(r"\s+", " ", t).strip()
    
    # Capitalize first letter if lowercase
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t

def tokenize_words_loose(s: str) -> List[str]:
    """Loose tokenization for entity extraction."""
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return [w for w in s.split() if len(w) >= 3]

def tokenize_words(s: str) -> List[str]:
    """Tokenize with stopword filtering."""
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return [w for w in s.split() if len(w) >= 3 and w not in STOP_EN]

def trigrams(words: List[str]) -> Set[str]:
    """Create 3-word shingles."""
    if len(words) < 3:
        return set()
    return {" ".join(words[i:i+3]) for i in range(len(words)-2)}

def sentences_fingerprint(sentences: List[str]) -> Set[str]:
    """Create trigram fingerprint from sentences."""
    ws = tokenize_words(" ".join(sentences or []))
    return trigrams(ws)

def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b: 
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return (inter / union) if union else 0.0

def simplify_query(q: str, keep: int = 4) -> str:
    """Simplify query to keep N keywords."""
    q = (q or "").lower()
    q = re.sub(r"[^a-z0-9 ]+", " ", q)
    toks = [t for t in q.split() if t and t not in STOP_EN]
    return " ".join(toks[:keep]) if toks else (q.strip()[:40] if q else "")

def extract_keywords(text: str, lang: str, k: int = 6) -> List[str]:
    """Extract top K keywords from text."""
    from collections import Counter
    
    stopwords = STOP_TR if lang.startswith("tr") else STOP_EN
    
    # Tokenize
    text = re.sub(r"[^A-Za-zçğıöşüÇĞİÖŞÜ0-9 ]+", " ", (text or "")).lower()
    words = [w for w in text.split() 
             if len(w) >= 4 and w not in stopwords and w not in GENERIC_SKIP]
    
    # Count frequencies
    cnt = Counter(words)
    
    # Bigrams
    bigrams = Counter()
    for i in range(len(words)-1):
        bigrams[words[i] + " " + words[i+1]] += 1
    
    # Score: bigrams*2 + unigrams
    scored = []
    for w, c in cnt.items():
        scored.append((c, w))
    for bg, c in bigrams.items():
        scored.append((c*2, bg))
    
    scored.sort(reverse=True)
    
    # Dedup and limit
    out = []
    for _, w in scored:
        if w not in out:
            out.append(w)
        if len(out) >= k: 
            break
    
    return out
