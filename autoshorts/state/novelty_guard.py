# novelty_guard.py
# ENHANCED: 4-Layer Anti-Repeat System + Sub-Topic Rotation
# - Channel-based 30-60 day cooldown
# - Semantic similarity (simhash + optional embeddings)
# - LRU entity rotation
# - SUB-TOPIC diversity tracking (NEW)
# State: .state/novelty.sqlite3

import os, re, json, math, time, sqlite3, hashlib, random, datetime
from typing import List, Dict, Any, Optional, Tuple, Set

# ============================================================================
# CONSTANTS - Import from constants.py if available
# ============================================================================
try:
    from autoshorts.config.constants import (
        NOVELTY_THRESHOLDS,
        COOLDOWN_DAYS,
        COUNTRIES,
        STOP_EN,
        GENERIC_SKIP
    )
    HAMMING_MAX = NOVELTY_THRESHOLDS["simhash_hamming_max"]
    ENT_JACCARD_MAX = NOVELTY_THRESHOLDS["entity_jaccard_max"]
    EMBED_COS_MAX = NOVELTY_THRESHOLDS["embed_cosine_max"]
    ENTITY_COOLDOWN = COOLDOWN_DAYS["entity"]
except ImportError:
    # Fallback if constants.py not available
    HAMMING_MAX = 8
    ENT_JACCARD_MAX = 0.58
    EMBED_COS_MAX = 0.86
    ENTITY_COOLDOWN = 45
    
    STOP_EN = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", 
        "of", "to", "in", "on", "with", "by", "about", "this", "that", 
        "these", "those", "is", "are", "was", "were", "be", "been", "being"
    }
    
    COUNTRIES = {
        "japan", "usa", "united states", "china", "india", "russia", 
        "germany", "france", "uk", "brazil", "canada", "australia"
    }
    
    GENERIC_SKIP = {"country", "countries", "people", "story", "fact", "facts"}

# ============================================================================
# SUB-TOPIC POOLS - Rotate angles within broad topics (NEW)
# ============================================================================
SUB_TOPIC_POOLS = {
    "country_facts": [
        "geography", "culture", "history", "economy", "technology",
        "food", "traditions", "nature", "urban_life", "innovations"
    ],
    "animal_facts": [
        "behavior", "habitat", "evolution", "hunting", "communication",
        "reproduction", "survival", "migration", "social_structure"
    ],
    "history_story": [
        "battles", "inventions", "discoveries", "leaders", "everyday_life",
        "art", "architecture", "trade", "conflicts", "cultural_exchange"
    ],
    "tech_news": [
        "ai", "robotics", "chips", "software", "hardware", "privacy",
        "security", "networking", "quantum", "biotech"
    ],
    "space_news": [
        "missions", "discoveries", "technology", "planets", "stars",
        "commercial", "research", "telescopes", "astronauts"
    ],
    # Default pool for unknown modes
    "_default": [
        "origin", "science", "culture", "future", "comparison",
        "hidden_aspect", "impact", "controversy", "innovation"
    ]
}

# ============================================================================
# Helper Functions
# ============================================================================

def _now_ts() -> int:
    return int(time.time())

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s\-]", " ", s, flags=re.UNICODE)
    s = _norm_space(s)
    return s

def _tokenize_words(s: str) -> List[str]:
    s = _clean_text(s)
    toks = [t for t in s.split() if t and t not in STOP_EN and not t.isdigit()]
    return toks

def _char_shingles(s: str, k: int = 4) -> List[str]:
    s = _clean_text(s)
    s = s.replace(" ", "_")
    if len(s) < k: return [s]
    return [s[i:i+k] for i in range(0, len(s)-k+1)]

def _md5_64(x: str) -> int:
    h = hashlib.md5(x.encode("utf-8")).hexdigest()
    return int(h, 16) & ((1 << 64) - 1)

def simhash64(text: str, use_char_shingles: bool = True) -> int:
    """Fast semantic similarity via simhash. Hamming<8 = very similar."""
    toks = _tokenize_words(text)
    if use_char_shingles:
        toks += _char_shingles(text, k=4)
    if not toks:
        toks = ["empty"]

    bits = [0]*64
    for t in toks:
        hv = _md5_64(t)
        w = 1.0
        for i in range(64):
            if (hv >> i) & 1:
                bits[i] += w
            else:
                bits[i] -= w
    out = 0
    for i in range(64):
        if bits[i] >= 0: out |= (1 << i)
    return out

def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0: return 0.0
    return inter / union

def extract_entities(text: str) -> Set[str]:
    """Light NER: proper noun patterns + country dictionary."""
    text_clean = _clean_text(text)
    ents: Set[str] = set()

    # Countries
    for c in COUNTRIES:
        if c in text_clean:
            ents.add(c)

    # Multi-word capitalized groups (basic proper noun detection)
    for m in re.finditer(r"\b([a-z][a-z0-9\-]+(?:\s+[a-z0-9\-]+){0,3})\b", text_clean):
        chunk = m.group(1)
        if chunk in STOP_EN or chunk in GENERIC_SKIP:
            continue
        if len(chunk) < 3:
            continue
        ents.add(chunk)

    # Limit entity set size
    if len(ents) > 80:
        ents = set(list(ents)[:80])

    return ents

# ============================================================================
# Embeddings (Optional)
# ============================================================================
_USE_EMBEDS = os.getenv("NOVELTY_EMBEDDINGS", "0") == "1"
_EMBED_MODEL = None

def _maybe_load_model():
    global _EMBED_MODEL
    if not _USE_EMBEDS or _EMBED_MODEL is not None: 
        return
    try:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except:
        _EMBED_MODEL = None

def embed(text: str) -> Optional[List[float]]:
    if not _USE_EMBEDS:
        return None
    _maybe_load_model()
    if _EMBED_MODEL is None:
        return None
    v = _EMBED_MODEL.encode([_norm_space(text)], normalize_embeddings=True)
    return v[0].tolist()

def cos_sim(a: List[float], b: List[float]) -> float:
    return sum(x*y for x,y in zip(a,b))

# ============================================================================
# Decision Class
# ============================================================================

class NoveltyDecision:
    def __init__(self, ok: bool, reason: str, nearest: Optional[Dict[str,Any]] = None, 
                 suggestions: Optional[Dict[str,Any]] = None):
        self.ok = ok
        self.reason = reason
        self.nearest = nearest or {}
        self.suggestions = suggestions or {}

    def __repr__(self):
        return f"NoveltyDecision(ok={self.ok}, reason={self.reason})"

# ============================================================================
# NoveltyGuard Class
# ============================================================================

class NoveltyGuard:
    def __init__(self, state_dir: str = ".state", window_days: int = 30, channel: Optional[str]=None):
        self.state_dir = state_dir
        self.db_path = os.path.join(state_dir, "novelty.sqlite3")
        self.window_days = int(window_days)
        _ensure_dir(state_dir)
        self._init_db()
        self._channel = channel

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        conn = self._conn()
        cur = conn.cursor()
        
        # Create table if not exists
        cur.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            ts INTEGER NOT NULL,
            channel TEXT NOT NULL,
            mode TEXT,
            lang TEXT,
            title TEXT,
            topic TEXT,
            entities TEXT,
            category TEXT,
            search_term TEXT,
            simhash INTEGER,
            content_hash TEXT,
            embed BLOB,
            pexels_ids TEXT,
            sub_topic TEXT
        );
        """)
        
        # Migration: Add sub_topic column if it doesn't exist
        try:
            cur.execute("SELECT sub_topic FROM items LIMIT 1;")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            import logging
            logging.getLogger(__name__).info("[NoveltyGuard] Migrating database: adding sub_topic column")
            cur.execute("ALTER TABLE items ADD COLUMN sub_topic TEXT;")
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_items_channel_ts ON items(channel, ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_items_simhash ON items(simhash);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_items_searchterm_ts ON items(channel, search_term, ts);")
        
        # Create sub_topic index (safe now)
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_items_subtopic ON items(channel, sub_topic, ts);")
        except sqlite3.OperationalError:
            pass  # Index might fail if column was just added
        
        conn.commit()
        conn.close()

    # ========================================================================
    # API: Search/Stats
    # ========================================================================
    
    def recent_items(self, channel: str, days: Optional[int]=None) -> List[Dict[str,Any]]:
        wnd = days if days is not None else self.window_days
        since = _now_ts() - int(wnd*86400)
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT ts, channel, mode, lang, title, topic, entities, category, 
                   search_term, simhash, content_hash, embed, pexels_ids, sub_topic
            FROM items 
            WHERE channel=? AND ts>=? 
            ORDER BY ts DESC;
        """, (channel, since))
        rows = cur.fetchall()
        conn.close()
        
        out = []
        for r in rows:
            (ts, ch, mode, lang, title, topic, entities, category, sterm, 
             sh, chash, emb, pids, subtopic) = r
            out.append({
                "ts": ts, "channel": ch, "mode": mode, "lang": lang, 
                "title": title or "", "topic": topic or "",
                "entities": json.loads(entities or "[]"), 
                "category": category or "", "search_term": sterm or "",
                "simhash": sh or 0, "content_hash": chash or "", 
                "embed": json.loads(emb or "null") if emb else None,
                "pexels_ids": json.loads(pids or "[]"),
                "sub_topic": subtopic or ""
            })
        return out

    def term_recency(self, channel: str) -> Dict[str,int]:
        """Map each search_term to its last-used timestamp."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT search_term, MAX(ts) 
            FROM items 
            WHERE channel=? 
            GROUP BY search_term;
        """, (channel,))
        rows = cur.fetchall()
        conn.close()
        return {(k or ""): (v or 0) for (k,v) in rows if k is not None}
    
    def subtopic_recency(self, channel: str, mode: str) -> Dict[str,int]:
        """Map each sub_topic to its last-used timestamp for a given mode."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT sub_topic, MAX(ts) 
            FROM items 
            WHERE channel=? AND mode=?
            GROUP BY sub_topic;
        """, (channel, mode))
        rows = cur.fetchall()
        conn.close()
        return {(k or ""): (v or 0) for (k,v) in rows if k is not None}

    # ========================================================================
    # API: LRU Term Selection
    # ========================================================================
    
    def pick_search_term(self, channel: str, candidates: List[str]) -> str:
        """Select LRU search term from candidates."""
        now = _now_ts()
        seen = self.term_recency(channel)
        scored = []
        
        for c in candidates:
            c2 = c.strip().lower()
            last = seen.get(c2, 0)
            days_since = (now - last)/86400.0 if last > 0 else 9999.0
            cooldown_ok = days_since >= self.window_days
            score = (1 if cooldown_ok else 0, days_since, random.random())
            scored.append((score, c))
        
        scored.sort(reverse=True)
        return scored[0][1]
    
    def pick_sub_topic(self, channel: str, mode: str) -> str:
        """
        NEW: Pick least-recently-used sub-topic for this channel+mode.
        Returns a sub-topic from SUB_TOPIC_POOLS[mode] or _default.
        """
        pool = SUB_TOPIC_POOLS.get(mode, SUB_TOPIC_POOLS["_default"])
        now = _now_ts()
        seen = self.subtopic_recency(channel, mode)
        
        scored = []
        for subtopic in pool:
            last = seen.get(subtopic, 0)
            days_since = (now - last)/86400.0 if last > 0 else 9999.0
            cooldown_ok = days_since >= 14  # 2-week sub-topic cooldown
            score = (1 if cooldown_ok else 0, days_since, random.random())
            scored.append((score, subtopic))
        
        scored.sort(reverse=True)
        best_subtopic = scored[0][1]
        
        import logging
        logging.getLogger(__name__).info(
            f"[NoveltyGuard] Selected sub-topic for {mode}: {best_subtopic} "
            f"(last used {scored[0][0][1]:.1f} days ago)"
        )
        
        return best_subtopic

    # ========================================================================
    # API: Novelty Check (Enhanced)
    # ========================================================================
    
    def check_novelty(
        self, 
        channel: str, 
        title: str, 
        script: str, 
        search_term: Optional[str] = None,
        category: Optional[str] = None, 
        mode: Optional[str] = None, 
        lang: Optional[str] = None,
        sub_topic: Optional[str] = None  # NEW: Optional sub-topic hint
    ) -> NoveltyDecision:
        """
        Enhanced novelty check with sub-topic awareness.
        """
        title = _norm_space(title or "")
        script = _norm_space(script or "")
        base = (title + " || " + script).strip()
        
        if not base:
            return NoveltyDecision(False, "EMPTY_CONTENT")

        ents = list(sorted(extract_entities(title + " " + script)))
        sh = simhash64(base)
        chash = hashlib.md5(base.encode("utf-8")).hexdigest()
        emb = embed(base)

        # Fetch recent items
        cand = self.recent_items(channel, self.window_days)
        nearest = None
        worst_reason = None

        # Check 1: Same search term in window
        if search_term:
            for row in cand:
                if row.get("search_term", "").strip().lower() == search_term.strip().lower():
                    nearest = row
                    worst_reason = "SAME_SEARCH_TERM_WINDOW"
                    return NoveltyDecision(False, worst_reason, nearest, 
                                         suggestions=self._suggest(channel, search_term, mode))

        # Check 2: Same sub-topic recently (if provided)
        if sub_topic and mode:
            for row in cand:
                if (row.get("sub_topic", "").strip().lower() == sub_topic.strip().lower() 
                    and row.get("mode", "") == mode):
                    # Sub-topic used within last 14 days
                    age_days = (_now_ts() - row.get("ts", 0)) / 86400.0
                    if age_days < 14:
                        nearest = row
                        worst_reason = f"SUB_TOPIC_TOO_RECENT (used {age_days:.1f} days ago)"
                        return NoveltyDecision(False, worst_reason, nearest,
                                             suggestions=self._suggest(channel, search_term, mode))

        # Check 3: Simhash
        for row in cand:
            dist = hamming64(sh, int(row.get("simhash") or 0))
            if dist < HAMMING_MAX:
                nearest = row
                worst_reason = f"SIMHASH_NEAR (hamming={dist})"
                return NoveltyDecision(False, worst_reason, nearest,
                                     suggestions=self._suggest(channel, search_term, mode))

        # Check 4: Entity overlap
        for row in cand:
            ja = jaccard(set(ents), set(row.get("entities") or []))
            if ja > ENT_JACCARD_MAX:
                nearest = row
                worst_reason = f"ENTITY_OVERLAP (jaccard={ja:.2f})"
                return NoveltyDecision(False, worst_reason, nearest,
                                     suggestions=self._suggest(channel, search_term, mode))

        # Check 5: Embeddings (optional)
        if emb:
            for row in cand:
                if row.get("embed"):
                    cs = cos_sim(emb, row["embed"])
                    if cs > EMBED_COS_MAX:
                        nearest = row
                        worst_reason = f"EMBED_SIM (cos={cs:.2f})"
                        return NoveltyDecision(False, worst_reason, nearest,
                                             suggestions=self._suggest(channel, search_term, mode))

        return NoveltyDecision(True, "OK")

    def _suggest(self, channel: str, current_term: Optional[str], mode: Optional[str] = None) -> Dict[str,Any]:
        """Generate alternative suggestions."""
        rec = self.term_recency(channel)
        sorted_terms = sorted(rec.items(), key=lambda kv: kv[1] or 0)
        alt_terms = [t for (t,_) in sorted_terms if (not current_term or t != current_term)]
        
        suggestions = {
            "avoid_terms": [current_term] if current_term else [],
            "alt_terms": alt_terms[:5]
        }
        
        # NEW: Suggest alternative sub-topics
        if mode:
            pool = SUB_TOPIC_POOLS.get(mode, SUB_TOPIC_POOLS["_default"])
            seen = self.subtopic_recency(channel, mode)
            # Sort by least recently used
            sorted_subtopics = sorted(pool, key=lambda st: seen.get(st, 0))
            suggestions["alt_subtopics"] = sorted_subtopics[:3]
        
        return suggestions

    # ========================================================================
    # API: Register Item (Enhanced)
    # ========================================================================
    
    def register_item(
        self, 
        channel: str, 
        title: str, 
        script: str,
        search_term: Optional[str] = None, 
        category: Optional[str] = None,
        mode: Optional[str] = None, 
        lang: Optional[str] = None,
        topic: Optional[str] = None, 
        pexels_ids: Optional[List[str]] = None,
        sub_topic: Optional[str] = None  # NEW
    ) -> None:
        """Register a new item with optional sub-topic tracking."""
        title = _norm_space(title or "")
        script = _norm_space(script or "")
        base = (title + " || " + script).strip()
        ents = list(sorted(extract_entities(title + " " + script)))
        sh = simhash64(base)
        chash = hashlib.md5(base.encode("utf-8")).hexdigest()
        emb = embed(base)
        ts = _now_ts()

        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO items 
            (ts, channel, mode, lang, title, topic, entities, category, 
             search_term, simhash, content_hash, embed, pexels_ids, sub_topic)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?);
        """, (
            ts, channel, mode or "", lang or "", title, topic or "", 
            json.dumps(ents), category or "", 
            (search_term or "").strip().lower() or None, 
            int(sh), chash,
            json.dumps(emb) if emb is not None else None,
            json.dumps(list(pexels_ids or [])),
            (sub_topic or "").strip().lower() or None
        ))
        conn.commit()
        conn.close()

    # ========================================================================
    # API: Pexels Deduplication
    # ========================================================================
    
    def used_pexels(self, channel: str, days: Optional[int]=None) -> Set[str]:
        rows = self.recent_items(channel, days)
        out = set()
        for r in rows:
            for vid in (r.get("pexels_ids") or []):
                out.add(str(vid))
        return out

    def filter_new_pexels(self, channel: str, candidate_ids: List[str], 
                          days: Optional[int]=None) -> List[str]:
        used = self.used_pexels(channel, days)
        return [cid for cid in candidate_ids if str(cid) not in used]
