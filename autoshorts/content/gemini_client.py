# -*- coding: utf-8 -*-
"""
Gemini AI client for content generation.
Handles API calls, prompt templates, and content parsing.
With retry mechanism for reliability.
"""
import re
import json
import time
import requests
from typing import Dict, List, Any, Optional

from autoshorts.config import settings


# ==================== TEMPLATES ====================

ENHANCED_GEMINI_TEMPLATES = {
    "_default": """Create a viral 25-40s YouTube Short that STOPS THE SCROLL.

Return STRICT JSON: topic, focus, sentences (7-8), search_terms (4-10), title, description, tags.

🎯 FOCUS RULE:
- ONE specific, visual, filmable subject
- Must have abundant stock footage
- Physical subjects only (no abstract concepts)

🔥 HOOK FORMULA (Sentence 1 - First 3 Seconds):
Pick the BEST pattern for your topic:

**Pattern A: Number + Shock**
"[Number] seconds/ways/reasons [subject] [unexpected action]"
Examples: "3 seconds is all this takes" | "5 reasons nobody tells you"

**Pattern B: Contradiction**
"[Common belief] is wrong. Here's why"
Examples: "You think you know this. You don't" | "Everything you learned is backwards"

**Pattern C: Question Hook**
"What if [surprising claim]?"
Examples: "What if I told you this changes everything?"

**Pattern D: Challenge**
"Try to [action] before [time/condition]"
Examples: "Try to spot the difference" | "Find the hidden detail"

**Pattern E: POV/Relatability**
"POV: You just [discovered/learned/realized] [X]"
Examples: "POV: You finally understand" | "When you realize"

**Pattern F: Mystery/Gap**
"Nobody knows why [X]... until now"
Examples: "The secret behind" | "What they don't show you"

🧲 RETENTION ARCHITECTURE (Sentences 2-7):

**Early (2-3): Build Intrigue**
- Plant a question you'll answer later
- Use: "But here's the crazy part"
- Introduce contrast/surprise

**Middle (4-5): Pattern Interrupt**
- Break expected flow
- Use: "Wait", "Stop", "Watch closely"
- Visual cue: "Look at [specific element]"

**Late (6-7): Climax**
- Deliver on hook promise
- Peak surprise/payoff
- Use: "And that's not even..."

📝 CTA (Sentence 8 - Last 3 Seconds):

**Comment Bait** (universal):
- "Which one surprised you?"
- "A or B? Comment below"
- "Did you catch it? Drop your answer"
- "Agree or disagree?"

⚡ UNIVERSAL RULES:
- 6-12 words per sentence
- Every sentence = ONE filmable action
- NO: "it's interesting", "you won't believe", "subscribe/like"
- Build: tension → peak → satisfying end
- End HIGH (not fade out)

Language: {lang}
""",

    "country_facts": """Create viral geographic/cultural facts.

[Same structure as default but with]:
- Focus on visual landmarks OR cultural elements
- Hook patterns adapted for places/culture
- Everything else stays universal

Language: {lang}
"""
}


class GeminiClient:
    """Handles all Gemini API interactions with retry mechanism."""
    
    def __init__(self):
        """Initialize client with API key from settings."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.GEMINI_MODEL
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        print(f"[Gemini] Using model: {self.model}")
    
    def generate(
        self, 
        topic: str, 
        mode: str, 
        lang: str,
        user_terms: Optional[List[str]] = None,
        banlist: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate content from Gemini.
        
        Args:
            topic: Main topic for the video
            mode: Content mode (country_facts, etc.)
            lang: Language code
            user_terms: Optional seed search terms
            banlist: Topics to avoid
            
        Returns:
            Dict with: topic, focus, sentences, search_terms, title, description, tags
        """
        # Select template
        template_key = self._select_template(mode)
        template = ENHANCED_GEMINI_TEMPLATES.get(template_key, ENHANCED_GEMINI_TEMPLATES["_default"])
        
        # Build prompt
        prompt = self._build_prompt(
            template=template,
            topic=topic,
            lang=lang,
            user_terms=user_terms or [],
            banlist=banlist or []
        )
        
        # Call API with retry
        raw_response = self._call_api_with_retry(prompt)
        
        # Parse response
        parsed = self._parse_response(raw_response)
        
        # Post-process
        parsed = self._post_process(parsed, topic, user_terms)
        
        return parsed
    
    def _select_template(self, mode: str) -> str:
        """Select appropriate template based on mode."""
        mode_lower = (mode or "").lower()
        
        if any(k in mode_lower for k in ["country", "geograph", "city"]):
            return "country_facts"
        
        return "_default"
    
    def _build_prompt(
        self, 
        template: str, 
        topic: str, 
        lang: str,
        user_terms: List[str],
        banlist: List[str]
    ) -> str:
        """Build complete prompt from template."""
        # Format template
        prompt = template.format(lang=lang)
        
        # Add context
        avoid = "\n".join(f"- {b}" for b in banlist[:15]) if banlist else "(none)"
        terms_hint = ", ".join(user_terms[:10]) if user_terms else "(none)"
        
        extra = ""
        if settings.GEMINI_PROMPT:
            extra = f"\n\nADDITIONAL STYLE:\n{settings.GEMINI_PROMPT}"
        
        guardrails = """
RULES (MANDATORY):
- STAY ON TOPIC exactly as provided.
- Return ONLY JSON, no prose/markdown.
- Keys required: topic, focus, sentences, search_terms, title, description, tags.

🎯 FOCUS SELECTION GUIDE:
- Pick ONE visual subject with abundant stock footage
- Good: "chameleon", "Tokyo tower", "lightning storm", "gears"
- Bad: "innovation", "happiness", "success" (too abstract)
- Must be filmable from multiple angles
"""
        
        full_prompt = f"""{prompt}

Channel: {settings.CHANNEL_NAME}
Language: {lang}
TOPIC (hard lock): {topic}
Seed search terms (use and expand): {terms_hint}
Avoid overlap for 180 days:
{avoid}{extra}
{guardrails}
"""
        return full_prompt
    
    def _call_api_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call API with retry mechanism for transient errors."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return self._call_api(prompt)
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if retryable error
                is_retryable = any(code in error_str for code in [
                    '503', '429', '500', '502', '504',
                    'service unavailable', 'rate limit', 'timeout'
                ])
                
                if is_retryable and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                    print(f"   ⚠️ Gemini API error (attempt {attempt+1}/{max_retries}): {str(e)[:100]}")
                    print(f"   ⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Not retryable or last attempt
                    raise
        
        # All retries failed
        raise last_error
    
    def _call_api(self, prompt: str) -> str:
        """Make API call to Gemini."""
        url = f"{self.base_url}/models/{self.model}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        # Add temperature jitter
        jitter = ((settings.ROTATION_SEED or 0) % 13) * 0.01
        temp = max(0.6, min(1.2, settings.GEMINI_TEMP + (jitter - 0.06)))
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temp}
        }
        
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            
            data = r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text
            
        except requests.exceptions.HTTPError as e:
            # Include response body in error for debugging
            error_detail = ""
            try:
                error_detail = e.response.text[:200]
            except:
                pass
            
            raise RuntimeError(
                f"Gemini API HTTP {e.response.status_code}: {str(e)}\n"
                f"Model: {self.model}\n"
                f"Detail: {error_detail}"
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)[:300]}")
    
    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response."""
        # Extract JSON block
        match = re.search(r"\{(?:.|\n)*\}", raw_text)
        if not match:
            raise RuntimeError("No JSON found in Gemini response")
        
        json_text = match.group(0).strip()
        
        # Remove markdown code blocks
        json_text = re.sub(r"^```json\s*|\s*```$", "", json_text, flags=re.MULTILINE)
        
        # Parse
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from Gemini: {e}")
    
    def _post_process(
        self, 
        data: Dict[str, Any], 
        topic: str,
        user_terms: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Clean and normalize parsed data."""
        from autoshorts.content.text_utils import (
            clean_caption_text,
            simplify_query
        )
        
        # Topic (use original)
        data['topic'] = topic
        
        # Sentences
        sentences = data.get("sentences", [])
        if isinstance(sentences, str):
            sentences = [sentences]
        sentences = [clean_caption_text(s) for s in sentences if s]
        data['sentences'] = sentences[:8]
        
        # Search terms
        terms = data.get("search_terms", [])
        if isinstance(terms, str):
            terms = [terms]
        terms = self._normalize_terms(terms)
        
        # Add user terms
        if user_terms:
            seed = self._normalize_terms(user_terms)
            terms = self._normalize_terms(seed + terms)
        
        data['search_terms'] = terms
        
        # Focus extraction
        focus = (data.get("focus") or "").strip()
        if not focus:
            focus = (data.get("title") or topic or (terms[0] if terms else "")).strip()
        
        focus = simplify_query(focus, keep=2)
        
        # Fallback if focus is generic
        if not focus or focus in ["great", "thing", "concept", "idea", "topic", "story"]:
            focus = (terms[0] if terms else simplify_query(topic, keep=1)) or "macro detail"
        
        data['focus'] = focus
        
        # Title, description, tags
        data['title'] = (data.get("title") or "").strip()
        data['description'] = (data.get("description") or "").strip()
        
        tags = data.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        data['tags'] = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
        
        return data
    
    def _normalize_terms(self, terms: List[str]) -> List[str]:
        """Normalize search terms."""
        BAD = {"great", "nice", "good", "bad", "things", "stuff", 
               "concept", "concepts", "idea", "ideas"}
        
        out = []
        seen = set()
        
        for t in terms or []:
            # Clean
            tt = re.sub(r"[^A-Za-z0-9 ]+", " ", str(t)).strip().lower()
            tt = " ".join([w for w in tt.split() if w and len(w) > 2 and w not in BAD])
            tt = tt[:64]
            
            if not tt:
                continue
            
            if tt not in seen:
                seen.add(tt)
                out.append(tt)
        
        return out[:12]
