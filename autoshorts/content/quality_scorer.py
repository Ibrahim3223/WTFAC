# -*- coding: utf-8 -*-
"""
Universal content quality scoring.
Works for all topics without topic-specific rules.
"""
import re
from typing import List, Dict


class QualityScorer:
    """Score content quality, viral potential, and retention."""
    
    def score(self, sentences: List[str], title: str = "") -> Dict[str, float]:
        """
        Score content across multiple dimensions.
        
        Args:
            sentences: List of script sentences
            title: Optional video title
            
        Returns:
            Dict with keys: quality, viral, retention, overall
        """
        text_all = (" ".join(sentences) + " " + title).lower()
        
        scores = {
            'quality': 5.0,
            'viral': 5.0,
            'retention': 5.0
        }
        
        # ===== QUALITY SIGNALS =====
        scores['quality'] += self._score_quality(sentences, text_all)
        
        # ===== VIRAL SIGNALS =====
        scores['viral'] += self._score_viral(sentences, text_all)
        
        # ===== RETENTION SIGNALS =====
        scores['retention'] += self._score_retention(sentences, text_all)
        
        # ===== NEGATIVE SIGNALS =====
        penalty = self._score_penalties(text_all)
        scores['quality'] -= penalty
        scores['viral'] -= penalty
        scores['retention'] -= penalty * 0.5
        
        # Normalize to 0-10
        for key in scores:
            scores[key] = max(0.0, min(10.0, scores[key]))
        
        # Calculate overall (weighted)
        scores['overall'] = (
            scores['quality'] * 0.4 + 
            scores['viral'] * 0.35 + 
            scores['retention'] * 0.25
        )
        
        return scores
    
    def _score_quality(self, sentences: List[str], text_all: str) -> float:
        """Score content quality."""
        score = 0.0
        
        # Conciseness (short sentences = clearer)
        avg_words = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        if avg_words <= 12:
            score += 1.5
        elif avg_words > 15:
            score -= 1.0
        
        # Specificity (numbers = concrete)
        num_count = len(re.findall(r'\b\d+\b', text_all))
        score += min(2.0, num_count * 0.5)
        
        # Active voice (action verbs)
        action_verbs = [
            'is', 'does', 'makes', 'shows', 'reveals', 
            'changes', 'breaks', 'creates', 'moves', 
            'stops', 'starts', 'turns'
        ]
        action_count = sum(1 for v in action_verbs if v in text_all)
        score += min(1.5, action_count * 0.3)
        
        return score
    
    def _score_viral(self, sentences: List[str], text_all: str) -> float:
        """Score viral potential."""
        score = 0.0
        
        if not sentences:
            return score
        
        hook = sentences[0].lower()
        
        # Hook strength
        if '?' in hook:
            score += 1.0
        if re.search(r'\b\d+\b', hook):
            score += 0.8
        
        mystery_words = ['secret', 'hidden', 'never', 'nobody', 'why', 'how']
        if any(w in hook for w in mystery_words):
            score += 0.6
        
        # Curiosity gap
        question_marks = text_all.count('?')
        score += min(1.2, question_marks * 0.4)
        
        # Emotional triggers
        triggers = [
            'shocking', 'insane', 'crazy', 'mind', 
            'unbelievable', 'secret', 'hidden'
        ]
        score += sum(0.3 for t in triggers if t in text_all)
        
        # Contrast markers
        contrasts = [
            'but', 'however', 'actually', 'surprisingly', 
            'turns out', 'wait'
        ]
        score += sum(0.25 for c in contrasts if c in text_all)
        
        return score
    
    def _score_retention(self, sentences: List[str], text_all: str) -> float:
        """Score retention potential."""
        score = 0.0
        
        # Pattern interrupts
        interrupts = [
            'wait', 'stop', 'look', 'watch', 
            'check', 'see', 'notice'
        ]
        score += sum(0.4 for i in interrupts if i in text_all)
        
        # Temporal cues (urgency)
        temporal = [
            'now', 'right now', 'immediately', 
            'seconds', 'instantly'
        ]
        score += sum(0.3 for t in temporal if t in text_all)
        
        # Visual references
        visual_refs = [
            'look at', 'watch', 'see', 'notice', 'spot', 'check'
        ]
        score += sum(0.35 for v in visual_refs if v in text_all)
        
        # Callback to hook (narrative closure)
        if len(sentences) >= 2 and sentences[-1] and sentences[0]:
            hook_words = set(sentences[0].lower().split()[:5])
            end_words = set(sentences[-1].lower().split())
            if hook_words & end_words:
                score += 1.0
        
        # Too long penalty
        if any(len(s.split()) > 18 for s in sentences):
            score -= 1.0
        
        return score
    
    def _score_penalties(self, text_all: str) -> float:
        """Calculate penalties for bad patterns."""
        penalty = 0.0
        
        # Generic filler
        bad_words = [
            'interesting', 'amazing', 'great', 'nice', 
            'good', 'cool', 'awesome'
        ]
        penalty += sum(0.5 for b in bad_words if b in text_all)
        
        # Meta references (breaks immersion)
        meta = [
            'this video', 'in this', 'today we', 
            "i'm going", "we're going", 'subscribe', 'like'
        ]
        penalty += sum(0.6 for m in meta if m in text_all)
        
        return penalty
