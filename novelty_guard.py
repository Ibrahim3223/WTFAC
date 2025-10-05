# novelty_guard.py
# 3-Katmanlı Anti-Tekrar Duvarı (kanal bazlı 30g cooldown + semantik benzerlik + LRU rotasyon)
# Kalıcı durum: .state/novelty.sqlite3
# Opsiyonel embeddings: NOVELTY_EMBEDDINGS=1 -> sentence-transformers/all-MiniLM-L6-v2 (isteğe bağlı)

import os, re, json, math, time, sqlite3, hashlib, random, datetime
from typing import List, Dict, Any, Optional, Tuple, Set

# ----------------- Yardımcılar -----------------

_STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","for","of","to","in","on","with","by","about",
    "this","that","these","those","is","are","was","were","be","been","being","it","its","as","at","from",
    "you","your","we","our","they","their","i","me","my","mine","he","she","his","her","them","us",
    "video","short","today","news","daily","facts","fact","story","stories","amazing","weird","random"
}

_COUNTRIES = {
    "japan","usa","united states","china","india","russia","germany","france","uk","united kingdom","turkey","türkiye",
    "brazil","argentina","mexico","spain","italy","canada","australia","south korea","north korea","sweden","norway",
    "finland","denmark","netherlands","belgium","poland","czech","greece","egypt","saudi arabia","uae","iran","iraq",
    "pakistan","bangladesh","indonesia","malaysia","singapore","thailand","vietnam","philippines","south africa",
    "nigeria","ethiopia","kenya","morocco","algeria","tunisia","colombia","chile","peru"
}

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
    toks = [t for t in s.split() if t and t not in _STOPWORDS and not t.isdigit()]
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
    """Hızlı semantik benzerlik. Hamming<8 -> 'çok benzer' sayılır (ayarlanabilir)."""
    toks = _tokenize_words(text)
    if use_char_shingles:
        toks += _char_shingles(text, k=4)
    if not toks:
        toks = ["empty"]

    bits = [0]*64
    for t in toks:
        hv = _md5_64(t)
        w  = 1.0  # ağırlıkları istersen burada arttır
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
    """Hafif NER: özel isim paternleri + ülke sözlüğü + büyük harf dizileri."""
    text_clean = _clean_text(text)
    toks = set(_tokenize_words(text))
    ents: Set[str] = set()

    # ülkeler
    for c in _COUNTRIES:
        if c in text_clean:
            ents.add(c)

    # 2+ kelimelik büyük harfli grupları (başlık) yakalama (basit yaklaşım)
    # Burada 'proper noun' için kısıtlı bir sezgisel
    for m in re.finditer(r"\b([a-z][a-z0-9\-]+(?:\s+[a-z0-9\-]+){0,3})\b", text_clean):
        chunk = m.group(1)
        # aşırı genel kelimeleri dışarıda tut
        if chunk in _STOPWORDS: 
            continue
        if len(chunk) < 3:
            continue
        ents.add(chunk)

    # çok uzun setleri ufalt
    if len(ents) > 80:
        ents = set(list(ents)[:80])

    return ents

# ----------------- Embeddings (opsiyonel) -----------------
_USE_EMBEDS = os.getenv("NOVELTY_EMBEDDINGS", "0") == "1"
_EMBED_MODEL = None
def _maybe_load_model():
    global _EMBED_MODEL
    if not _USE_EMBEDS or _EMBED_MODEL is not None: 
        return
    try:
        # Bu indirme ağır olabilir; istersen cache’le.
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        # Embedding zorunlu değil; sessiz düş.
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
    # normalize edilmiş beklenir
    return sum(x*y for x,y in zip(a,b))

# ----------------- SQLite State -----------------

class NoveltyDecision:
    def __init__(self, ok: bool, reason: str, nearest: Optional[Dict[str,Any]] = None, suggestions: Optional[Dict[str,Any]] = None):
        self.ok = ok
        self.reason = reason
        self.nearest = nearest or {}
        self.suggestions = suggestions or {}

    def __repr__(self):
        return f"NoveltyDecision(ok={self.ok}, reason={self.reason})"

class NoveltyGuard:
    def __init__(self, state_dir: str = ".state", window_days: int = 30, channel: Optional[str]=None):
        self.state_dir = state_dir
        self.db_path = os.path.join(state_dir, "novelty.sqlite3")
        self.window_days = int(window_days)
        _ensure_dir(state_dir)
        self._init_db()
        self._channel = channel  # opsiyonel varsayılan

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            ts INTEGER NOT NULL,
            channel TEXT NOT NULL,
            mode TEXT,
            lang TEXT,
            title TEXT,
            topic TEXT,
            entities TEXT,       -- JSON list[str]
            category TEXT,
            search_term TEXT,
            simhash INTEGER,
            content_hash TEXT,   -- md5 of (title+script)
            embed BLOB,          -- JSON list[float] (optional)
            pexels_ids TEXT      -- JSON list[str] (optional)
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_items_channel_ts ON items(channel, ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_items_simhash ON items(simhash);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_items_searchterm_ts ON items(channel, search_term, ts);")
        conn.commit()
        conn.close()

    # ---------- API: arama/istatistik ----------
    def recent_items(self, channel: str, days: Optional[int]=None) -> List[Dict[str,Any]]:
        wnd = days if days is not None else self.window_days
        since = _now_ts() - int(wnd*86400)
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("SELECT ts,channel,mode,lang,title,topic,entities,category,search_term,simhash,content_hash,embed,pexels_ids FROM items WHERE channel=? AND ts>=? ORDER BY ts DESC;", (channel, since))
        rows = cur.fetchall()
        conn.close()
        out=[]
        for r in rows:
            ts, ch, mode, lang, title, topic, entities, category, sterm, sh, chash, emb, pids = r
            out.append({
                "ts": ts, "channel": ch, "mode": mode, "lang": lang, "title": title or "", "topic": topic or "",
                "entities": json.loads(entities or "[]"), "category": category or "", "search_term": sterm or "",
                "simhash": sh or 0, "content_hash": chash or "", "embed": json.loads(emb or "null") if emb else None,
                "pexels_ids": json.loads(pids or "[]")
            })
        return out

    def term_recency(self, channel: str) -> Dict[str,int]:
        """Her search_term için son kullanıldığı timestamp."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("SELECT search_term, MAX(ts) FROM items WHERE channel=? GROUP BY search_term;", (channel,))
        rows = cur.fetchall()
        conn.close()
        return { (k or ""): (v or 0) for (k,v) in rows if k is not None }

    # ---------- API: LRU term seçimi ----------
    def pick_search_term(self, channel: str, candidates: List[str]) -> str:
        """Son 30 günde kullanılmamışsa önceliklendir; eşitlikte en eski kullanılan seçilir."""
        now = _now_ts()
        seen = self.term_recency(channel)
        scored=[]
        for c in candidates:
            c2 = c.strip().lower()
            last = seen.get(c2, 0)
            days_since = (now - last)/86400.0 if last>0 else 9999.0
            cooldown_ok = days_since >= self.window_days
            # Öncelik: cooldown_ok -> büyük skor; sonra en eski
            score = (1 if cooldown_ok else 0, days_since, random.random())
            scored.append((score, c))
        scored.sort(reverse=True)  # en iyi başa
        return scored[0][1]

    # ---------- API: Yenilik kontrolü ----------
    def check_novelty(self, channel: str, title: str, script: str, search_term: Optional[str]=None,
                      category: Optional[str]=None, mode: Optional[str]=None, lang: Optional[str]=None) -> NoveltyDecision:
        title = _norm_space(title or "")
        script = _norm_space(script or "")
        base = (title + " || " + script).strip()
        if not base:
            return NoveltyDecision(False, "EMPTY_CONTENT")

        ents = list(sorted(extract_entities(title + " " + script)))
        sh   = simhash64(base)
        chash = hashlib.md5(base.encode("utf-8")).hexdigest()
        emb = embed(base)

        # Son 30 gün içi karşılaştırma
        cand = self.recent_items(channel, self.window_days)
        nearest = None
        worst_reason = None

        # Eşikler
        HAMMING_MAX     = int(os.getenv("NOVELTY_SIMHASH_MAX", "8"))     # <8 -> çok benzer
        ENT_JACCARD_MAX = float(os.getenv("NOVELTY_ENTITY_JACCARD", "0.60"))  # >0.60 -> fazla ortak
        EMBED_COS_MAX   = float(os.getenv("NOVELTY_EMBED_COS", "0.88"))  # >0.88 -> fazla benzer

        for row in cand:
            # aynı search_term yakın geçmişte mi?
            if search_term and row.get("search_term"):
                if row["search_term"].strip().lower() == search_term.strip().lower():
                    # aynı terim -> doğrudan fail
                    nearest = row; worst_reason = "SAME_SEARCH_TERM_WINDOW"
                    return NoveltyDecision(False, worst_reason, nearest, suggestions=self._suggest(channel, search_term))

            # simhash
            dist = hamming64(sh, int(row.get("simhash") or 0))
            if dist < HAMMING_MAX:
                nearest = row; worst_reason = f"SIMHASH_NEAR (hamming={dist})"
                return NoveltyDecision(False, worst_reason, nearest, suggestions=self._suggest(channel, search_term))

            # entities
            ja = jaccard(set(ents), set(row.get("entities") or []))
            if ja > ENT_JACCARD_MAX:
                nearest = row; worst_reason = f"ENTITY_OVERLAP (jaccard={ja:.2f})"
                return NoveltyDecision(False, worst_reason, nearest, suggestions=self._suggest(channel, search_term))

            # embeddings (opsiyonel, varsa)
            if emb and row.get("embed"):
                cs = cos_sim(emb, row["embed"])
                if cs > EMBED_COS_MAX:
                    nearest = row; worst_reason = f"EMBED_SIM (cos={cs:.2f})"
                    return NoveltyDecision(False, worst_reason, nearest, suggestions=self._suggest(channel, search_term))

        return NoveltyDecision(True, "OK")

    def _suggest(self, channel: str, current_term: Optional[str]) -> Dict[str,Any]:
        """Alternatif search_term ve kaçınma listesi üret."""
        rec = self.term_recency(channel)
        # En uzun süredir kullanılmayanları öner
        sorted_terms = sorted(rec.items(), key=lambda kv: kv[1] or 0)  # eski -> yeni
        alt_terms = [t for (t,_) in sorted_terms if (not current_term or t != current_term)]
        return {
            "avoid_terms": [current_term] if current_term else [],
            "alt_terms": alt_terms[:5]
        }

    # ---------- API: Kayıt ----------
    def register_item(self, channel: str, title: str, script: str,
                      search_term: Optional[str]=None, category: Optional[str]=None,
                      mode: Optional[str]=None, lang: Optional[str]=None,
                      topic: Optional[str]=None, pexels_ids: Optional[List[str]]=None) -> None:
        title  = _norm_space(title or "")
        script = _norm_space(script or "")
        base   = (title + " || " + script).strip()
        ents   = list(sorted(extract_entities(title + " " + script)))
        sh     = simhash64(base)
        chash  = hashlib.md5(base.encode("utf-8")).hexdigest()
        emb    = embed(base)
        ts     = _now_ts()

        conn = self._conn()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO items (ts,channel,mode,lang,title,topic,entities,category,search_term,simhash,content_hash,embed,pexels_ids)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?);
        """, (
            ts, channel, mode or "", lang or "", title, topic or "", json.dumps(ents), category or "", 
            (search_term or "").strip().lower() or None, int(sh), chash,
            json.dumps(emb) if emb is not None else None,
            json.dumps(list(pexels_ids or []))
        ))
        conn.commit()
        conn.close()

    # ---------- API: Pexels tekrarını engelleme ----------
    def used_pexels(self, channel: str, days: Optional[int]=None) -> Set[str]:
        rows = self.recent_items(channel, days)
        out=set()
        for r in rows:
            for vid in (r.get("pexels_ids") or []):
                out.add(str(vid))
        return out

    def filter_new_pexels(self, channel: str, candidate_ids: List[str], days: Optional[int]=None) -> List[str]:
        used = self.used_pexels(channel, days)
        return [cid for cid in candidate_ids if str(cid) not in used]
