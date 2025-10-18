# -*- coding: utf-8 -*-
# state_guard.py — Cooldown + semantic dedupe + idempotency (channel-aware)
import os, json, hashlib, logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np

STATE_DIR = os.getenv("STATE_DIR", ".state")
USED_ENTITIES_FILE = os.path.join(STATE_DIR, "used_entities.json")   # {channel: {entity: iso}}
EMBEDDINGS_FILE   = os.path.join(STATE_DIR, "embeddings.json")       # {channel: {"vectors":[...]} }
UPLOADS_FILE      = os.path.join(STATE_DIR, "uploads.json")          # [{"channel":..,"content_hash":..,"title":..,"ts":..}]
COOLDOWN_DAYS     = int(os.getenv("ENTITY_COOLDOWN_DAYS", "30"))
SIMILARITY_THRESHOLD_SCRIPT  = float(os.getenv("SIM_TH_SCRIPT", "0.90"))
SIMILARITY_THRESHOLD_ENTITY  = float(os.getenv("SIM_TH_ENTITY", "0.92"))  # isim bazlı benzerlik için biraz daha sıkı

# ---- lazy import: sentence-transformers opsiyonel ----
_model = None
def _get_model():
    global _model
    if _model is not None: return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logging.warning(f"[state_guard] Embedding model yüklenemedi: {e}")
        _model = None
    return _model

def _load_json(path, default):
    try:
        with open(path, "r") as f: return json.load(f)
    except Exception:
        return default

def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=2, ensure_ascii=False)

def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

class StateGuard:
    def __init__(self, channel: str):
        self.channel = channel
        os.makedirs(STATE_DIR, exist_ok=True)
        self.used_entities: Dict[str, Dict[str, str]] = _load_json(USED_ENTITIES_FILE, {})
        self.embeddings: Dict[str, Dict[str, List[List[float]]]] = _load_json(EMBEDDINGS_FILE, {})
        self.uploads: List[Dict[str, Any]] = _load_json(UPLOADS_FILE, [])

        # kanal anahtarlarını hazırla
        self.used_entities.setdefault(channel, {})
        self.embeddings.setdefault(channel, {})
        self.embeddings[channel].setdefault("vectors", [])

        self.model = _get_model()

    # ---------- Cooldown ----------
    def is_on_cooldown(self, entity: str) -> bool:
        rec = self.used_entities.get(self.channel, {}).get(entity)
        if not rec: return False
        try:
            last = datetime.fromisoformat(rec)
            return (datetime.now() - last) < timedelta(days=COOLDOWN_DAYS)
        except Exception:
            return False

    def days_since_used(self, entity: str) -> Optional[int]:
        rec = self.used_entities.get(self.channel, {}).get(entity)
        if not rec: return None
        try:
            last = datetime.fromisoformat(rec)
            return (datetime.now() - last).days
        except Exception:
            return None

    # ---------- Entity semantik benzerlik (alias/synonym yakalar) ----------
    def entity_too_similar(self, candidate: str, prev_entities: Optional[List[str]] = None) -> bool:
        if self.model is None:
            return False  # model yoksa geç
        prev = list((self.used_entities.get(self.channel, {}) or {}).keys()) if prev_entities is None else prev_entities
        if not prev: return False
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            vec_new = self.model.encode([_norm_text(candidate)])
            vec_old = self.model.encode([_norm_text(e) for e in prev])
            sims = cosine_similarity(vec_new, vec_old)[0]
            return float(np.max(sims)) >= SIMILARITY_THRESHOLD_ENTITY
        except Exception as e:
            logging.warning(f"[state_guard] entity similarity hata: {e}")
            return False

    # ---------- Script semantik benzerlik ----------
    def script_semantic_duplicate(self, script_text: str) -> bool:
        if self.model is None:
            return False
        vectors = self.embeddings.get(self.channel, {}).get("vectors", [])
        if not vectors: return False
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            new_vec = self.model.encode([_norm_text(script_text)])
            sims = cosine_similarity(new_vec, np.array(vectors))[0]
            return float(np.max(sims)) >= SIMILARITY_THRESHOLD_SCRIPT
        except Exception as e:
            logging.warning(f"[state_guard] script similarity hata: {e}")
            return False

    # ---------- İçerik hash ----------
    @staticmethod
    def make_content_hash(script_text: str, video_paths: List[str], audio_path: Optional[str]) -> str:
        # Flowith formülünü temel al: normalize(script) + klip dosya adları + audio fingerprint
        # (video URL değil, dosyaAdı/ID kullan; stabil). 参: flowith state.generate_content_hash
        norm = _norm_text(script_text)
        video_ids = sorted([os.path.basename(p or "") for p in (video_paths or [])])
        h = hashlib.sha1()
        h.update(norm.encode("utf-8"))
        h.update(":".join(video_ids).encode("utf-8"))
        if audio_path and os.path.exists(audio_path):
            try:
                with open(audio_path, "rb") as f:
                    h.update(hashlib.sha1(f.read()).hexdigest().encode("utf-8"))
            except Exception:
                pass
        return h.hexdigest()

    def was_uploaded(self, content_hash: str) -> bool:
        for rec in self.uploads:
            if rec.get("channel") == self.channel and rec.get("content_hash") == content_hash:
                return True
        return False

    def record_upload(self, video_id: str, content: Dict[str, Any]) -> None:
        """
        Record a successful upload for tracking and deduplication.
        Called by orchestrator after successful YouTube upload.
        
        Args:
            video_id: YouTube video ID
            content: Content dict with metadata, script, etc.
        """
        try:
            # Extract key information
            title = content.get("metadata", {}).get("title", "")
            script_text = " ".join([
                content.get("hook", ""),
                *content.get("script", []),
                content.get("cta", "")
            ])
            
            # Generate content hash (use first 16 chars to avoid SQLite INTEGER overflow)
            content_hash_full = self.make_content_hash(
                script_text=script_text,
                video_paths=[],
                audio_path=None
            )
            content_hash = content_hash_full[:16]  # ✅ DÜZELTME: Shorten hash to avoid SQLite issues
            
            # Extract main entity from title or first search query
            entity = title
            if not entity and content.get("search_queries"):
                entity = content["search_queries"][0]
            
            # Record in state
            self.mark_uploaded(
                entity=entity,
                script_text=script_text,
                content_hash=content_hash,
                video_path=f"youtube:{video_id}",
                title=title
            )
            
            logging.info(f"[state_guard] Recorded upload: {video_id} - {title}")
            
        except Exception as e:
            logging.error(f"[state_guard] Failed to record upload: {e}")
            import traceback
            logging.debug(traceback.format_exc())

    def mark_uploaded(self, entity: str, script_text: str, content_hash: str,
                      video_path: str, title: str = ""):
        # entities
        self.used_entities[self.channel][entity] = datetime.now().isoformat()
        _save_json(USED_ENTITIES_FILE, self.used_entities)

        # embeddings (script)
        if self.model is not None:
            try:
                vec = self.model.encode(_norm_text(script_text)).tolist()
                self.embeddings[self.channel]["vectors"].append(vec)
                _save_json(EMBEDDINGS_FILE, self.embeddings)
            except Exception as e:
                logging.warning(f"[state_guard] embed save hata: {e}")

        # uploads
        self.uploads.append({
            "channel": self.channel,
            "entity": entity,
            "content_hash": content_hash,
            "title": title,
            "video_path": video_path,
            "ts": datetime.now().isoformat()
        })
        _save_json(UPLOADS_FILE, self.uploads)

    # ---------- Aday seçici yardımcı ----------
    def pick_viable_entity(self, candidates: List[str], banned: List[str]) -> Optional[str]:
        banned_set = {e.strip().lower() for e in (banned or [])}
        prev = list(self.used_entities.get(self.channel, {}).keys())
        best = None
        best_score = -1e9
        for c in candidates:
            c0 = c.strip()
            if not c0: continue
            if c0.lower() in banned_set: continue
            if self.is_on_cooldown(c0): continue
            if self.entity_too_similar(c0, prev): continue
            # skorlama: az kullanılan + rastgele küçük şans
            days = self.days_since_used(c0) or 10_000
            score = days + np.random.rand()*0.1
            if score > best_score:
                best, best_score = c0, score
        return best
