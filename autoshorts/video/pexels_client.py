# -*- coding: utf-8 -*-
"""
Pexels API client for video search.
"""
import re
import random
import logging
from typing import List, Tuple, Set, Optional
import requests

from autoshorts.config import settings

logger = logging.getLogger(__name__)


class PexelsClient:
    """Handle Pexels API interactions."""
    
    def __init__(self):
        """Initialize with API key."""
        if not settings.PEXELS_API_KEY:
            raise ValueError("PEXELS_API_KEY required")
        
        self.api_key = settings.PEXELS_API_KEY
        self.base_url = "https://api.pexels.com/videos"
        self.per_page = settings.PEXELS_PER_PAGE
        self.min_height = settings.PEXELS_MIN_HEIGHT
        self.min_duration = settings.PEXELS_MIN_DURATION
        self.max_duration = settings.PEXELS_MAX_DURATION
        self.strict_vertical = settings.PEXELS_STRICT_VERTICAL
        self.allow_landscape = settings.PEXELS_ALLOW_LANDSCAPE
        
        # Track used clips for session
        self._page_urls = {}  # vid -> url (for scoring)
    
    def search_simple(
        self,
        query: str,
        count: int
    ) -> List[Tuple[int, str]]:
        """
        ✅ YENİ: Simple search - one query, get N videos sequentially.
        Perfect for coherent visual storytelling across all scenes.
        
        Args:
            query: Single search query (1-2 words, e.g. "hoopoe bird")
            count: Number of videos needed
            
        Returns:
            List of (video_id, download_url) tuples in order
        """
        logger.info(f"🔍 Simple search: '{query}' → {count} videos")
        
        results = []
        seen_ids = set()
        
        # Search through pages until we have enough
        page = 1
        max_pages = 5  # Limit to avoid too many API calls
        
        while len(results) < count and page <= max_pages:
            page_results = self._search(query, page=page)
            
            for vid, link, w, h, dur in page_results:
                if vid in seen_ids:
                    continue
                
                seen_ids.add(vid)
                results.append((vid, link))
                
                if len(results) >= count:
                    break
            
            page += 1
        
        # If still not enough, try generic fallback
        if len(results) < count:
            logger.warning(f"   ⚠️ Only found {len(results)}/{count} videos for '{query}'")
            logger.info(f"   🔄 Trying generic fallback...")
            
            # Extract single meaningful word
            words = query.lower().split()
            fallback_query = next((w for w in words if len(w) > 3), "nature")
            
            fallback_results = self._search(fallback_query, page=1)
            
            for vid, link, w, h, dur in fallback_results:
                if vid in seen_ids:
                    continue
                
                seen_ids.add(vid)
                results.append((vid, link))
                
                if len(results) >= count:
                    break
        
        # Final fallback: ultra-generic
        if len(results) < count:
            logger.warning(f"   ⚠️ Still need more, using ultra-generic...")
            generic = ["nature landscape", "city skyline", "ocean waves"]
            
            for g_query in generic:
                g_results = self._search(g_query, page=1)
                
                for vid, link, w, h, dur in g_results:
                    if vid in seen_ids:
                        continue
                    
                    seen_ids.add(vid)
                    results.append((vid, link))
                    
                    if len(results) >= count:
                        break
                
                if len(results) >= count:
                    break
        
        logger.info(f"   ✅ Found {len(results)} videos (requested {count})")
        
        return results
    
    def build_pool(
        self,
        focus: str,
        search_terms: List[str],
        need: int
    ) -> List[Tuple[int, str]]:
        """
        Build video pool using FOCUS-FIRST strategy with smart fallbacks.
        
        Args:
            focus: Main visual keyword
            search_terms: Additional search terms
            need: Number of clips needed
            
        Returns:
            List of (video_id, download_url) tuples
        """
        from autoshorts.content.text_utils import simplify_query
        
        # ✅ ULTRA-GENERIC FALLBACK: Always have a backup plan
        ultra_generic_queries = [
            "nature landscape",
            "city skyline",
            "ocean waves",
            "forest trees",
            "sunset sky",
            "mountains",
            "coffee",
            "technology",
            "people lifestyle",
            "abstract motion"
        ]
        
        # Prepare focus queries
        main_focus = simplify_query(
            focus or (search_terms[0] if search_terms else ultra_generic_queries[0]), 
            keep=2
        )
        
        # Get synonyms
        syns = self._get_synonyms(main_focus, settings.LANG)
        syn_tokens = self._url_tokens(" ".join(syns + [main_focus]))
        
        # Build query list with smart fallbacks
        queries = [main_focus] + syns[:3]
        queries = list(dict.fromkeys(queries))  # dedup
        
        logger.info(f"🎯 FOCUS-FIRST: '{main_focus}' | Synonyms: {syns[:3]}")
        
        pool = []
        target = need * 3
        
        # 1. Try specific queries first
        for q in queries:
            qtokens = self._url_tokens(q)
            merged = []
            
            for page in range(1, 4):
                results = self._search(q, page=page)
                merged.extend(results)
                
                if len(merged) >= target:
                    break
            
            ranked = self._rank_and_dedup(
                merged, 
                qtokens, 
                syn_tokens=syn_tokens,
                strict=False
            )
            
            pool.extend(ranked)
            logger.info(f"   Query '{q}': {len(ranked)} clips")
            
            if len(pool) >= target:
                break
        
        # 2. Try broader terms if needed
        if len(pool) < need:
            logger.info(f"   ⚠️ Need more clips, trying broader terms...")
            
            words = main_focus.split()
            for word in words:
                if len(word) > 3:
                    results = self._search(word, page=1)
                    ranked = self._rank_and_dedup(
                        results,
                        self._url_tokens(word),
                        syn_tokens=syn_tokens,
                        strict=False
                    )
                    pool.extend(ranked)
                    logger.info(f"   Broader term '{word}': {len(ranked)} clips")
                    
                    if len(pool) >= need:
                        break
        
        # 3. Try popular videos if still not enough
        if len(pool) < need:
            logger.info(f"   ⚠️ Still need more, checking popular...")
            
            for page in range(1, 3):
                pop = self._popular(page=page)
                pop_ranked = self._rank_and_dedup(
                    pop,
                    syn_tokens,
                    syn_tokens=syn_tokens,
                    strict=False
                )
                pool.extend(pop_ranked)
                
                if len(pool) >= need:
                    break
        
        # 4. ULTRA-GENERIC FALLBACK
        if len(pool) < need:
            logger.warning(f"   ⚠️ Emergency fallback to ultra-generic queries...")
            
            for generic_q in ultra_generic_queries[:5]:
                results = self._search(generic_q, page=1)
                ranked = self._rank_and_dedup(
                    results,
                    self._url_tokens(generic_q),
                    syn_tokens=set(),
                    strict=False
                )
                pool.extend(ranked)
                logger.info(f"   Generic '{generic_q}': {len(ranked)} clips")
                
                if len(pool) >= need:
                    break
        
        # Dedup final results
        seen = set()
        dedup = []
        for vid, link in pool:
            if vid in seen:
                continue
            seen.add(vid)
            dedup.append((vid, link))
        
        final = dedup[:max(need, need * 2)]
        
        logger.info(f"   ✅ Final pool: {len(final)} clips")
        
        if len(final) < need:
            logger.error(f"   ❌ WARNING: Only found {len(final)} clips, needed {need}")
        
        return final
    
    def _search(
        self, 
        query: str, 
        page: int = 1
    ) -> List[Tuple[int, str, int, int, float]]:
        """Search Pexels videos"""
        url = f"{self.base_url}/search"
        
        params = {
            "query": query,
            "per_page": self.per_page,
            "page": page,
            "size": "large",
            "locale": "en-US" if not settings.LANG.startswith("tr") else "tr-TR"
        }
        
        headers = {"Authorization": self.api_key}
        
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
        except Exception as e:
            logger.warning(f"   ⚠️ Search failed for '{query}': {e}")
            return []
        
        data = r.json() or {}
        results = []
        
        for video in data.get("videos", []):
            vid = int(video.get("id", 0))
            dur = float(video.get("duration", 0.0))
            
            if dur < self.min_duration or dur > self.max_duration:
                continue
            
            page_url = (video.get("url") or "").strip()
            if page_url:
                self._page_urls[vid] = page_url
            
            picks = []
            for vf in video.get("video_files", []):
                w = int(vf.get("width", 0))
                h = int(vf.get("height", 0))
                
                if self._is_vertical_ok(w, h):
                    picks.append((w, h, vf.get("link")))
            
            if not picks:
                continue
            
            picks.sort(key=lambda t: (abs(t[1] - 1600), -(t[0] * t[1])))
            w, h, link = picks[0]
            
            results.append((vid, link, w, h, dur))
        
        return results
    
    def _popular(self, page: int = 1) -> List[Tuple[int, str, int, int, float]]:
        """Get popular videos"""
        url = f"{self.base_url}/popular"
        
        params = {
            "per_page": 40,
            "page": page,
            "locale": "en-US"
        }
        
        headers = {"Authorization": self.api_key}
        
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
        except Exception as e:
            logger.warning(f"   ⚠️ Popular videos fetch failed: {e}")
            return []
        
        data = r.json() or {}
        results = []
        
        for video in data.get("videos", []):
            vid = int(video.get("id", 0))
            dur = float(video.get("duration", 0.0))
            
            if dur < self.min_duration or dur > self.max_duration:
                continue
            
            page_url = (video.get("url") or "").strip()
            if page_url:
                self._page_urls[vid] = page_url
            
            picks = []
            for vf in video.get("video_files", []):
                w = int(vf.get("width", 0))
                h = int(vf.get("height", 0))
                
                if self._is_vertical_ok(w, h):
                    picks.append((w, h, vf.get("link")))
            
            if not picks:
                continue
            
            picks.sort(key=lambda t: (abs(t[1] - 1600), -(t[0] * t[1])))
            w, h, link = picks[0]
            
            results.append((vid, link, w, h, dur))
        
        return results
    
    def _is_vertical_ok(self, w: int, h: int) -> bool:
        """Check if dimensions are acceptable"""
        if self.strict_vertical:
            return h > w and h >= self.min_height
        
        return (h >= self.min_height) and (h >= w or self.allow_landscape)
    
    def _rank_and_dedup(
        self,
        items: List[Tuple[int, str, int, int, float]],
        qtokens: Set[str],
        syn_tokens: Optional[Set[str]] = None,
        strict: bool = False
    ) -> List[Tuple[int, str]]:
        """Rank and deduplicate results"""
        syn_tokens = syn_tokens or set()
        
        candidates = []
        
        for vid, link, w, h, dur in items:
            page_url = self._page_urls.get(vid, "")
            tokens = self._url_tokens(link) | self._url_tokens(page_url)
            
            if strict and syn_tokens and not (tokens & syn_tokens):
                continue
            
            overlap_q = len(tokens & qtokens)
            overlap_syn = len(tokens & syn_tokens) if syn_tokens else 0
            
            score = (
                (overlap_q * 1.0) + 
                (overlap_syn * 2.0) + 
                (1.0 if 2.0 <= dur <= 12.0 else 0.0) + 
                (1.0 if h >= 1440 else 0.0)
            )
            
            candidates.append((score, vid, link))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        seen = set()
        result = []
        
        for _, vid, link in candidates:
            if vid in seen:
                continue
            seen.add(vid)
            result.append((vid, link))
        
        return result
    
    def _url_tokens(self, s: str) -> Set[str]:
        """Extract tokens from URL"""
        return set(re.findall(r"[a-z0-9]+", (s or "").lower()))
    
    def _get_synonyms(self, entity: str, lang: str) -> List[str]:
        """Get synonyms for entity"""
        e = (entity or "").lower().strip()
        if not e:
            return []
        
        base = [e]
        if e.endswith("s") and len(e) > 4:
            base.append(e[:-1])
        
        table_en = {
            "chameleon": ["chameleon", "lizard", "gecko", "reptile"],
            "dolphin": ["dolphin", "marine mammal", "ocean animal"],
            "octopus": ["octopus", "cephalopod", "sea creature"],
            "japan": ["japan", "tokyo", "japanese"],
            "italy": ["italy", "rome", "italian"],
            "eagle": ["eagle", "raptor", "bird of prey"],
            "bridge": ["bridge", "structure"],
            "food": ["food", "meal", "cooking"],
            "nature": ["nature", "landscape", "outdoor"],
            "city": ["city", "urban", "street"],
            "ocean": ["ocean", "sea", "water"],
            "people": ["people", "person", "human"],
        }
        
        for k, vals in table_en.items():
            if k in e or e in k:
                return list(dict.fromkeys(vals + base))
        
        return list(dict.fromkeys(base))
