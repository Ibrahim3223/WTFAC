"""
Gemini API Client for Content Generation
Uses Google's official genai SDK
"""

import json
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass
class ContentResponse:
    """Structured response from content generation"""
    hook: str
    script: List[str]
    cta: str
    search_queries: List[str]
    main_visual_focus: str  # ✅ YENİ: 1-2 kelimelik ana konu
    metadata: Dict[str, Any]


class GeminiClient:
    """Client for Gemini API interactions using official SDK"""
    
    # Available models - UPDATED!
    MODELS = {
        "flash": "gemini-2.5-flash",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "flash-2.0": "gemini-2.0-flash-exp",
        "flash-thinking": "gemini-2.0-flash-thinking-exp-1219",
        "pro": "gemini-1.5-pro-latest",
        "flash-8b": "gemini-1.5-flash-8b-latest"
    }
    
    def __init__(
        self,
        api_key: str,
        model: str = "flash",
        max_retries: int = 3,
        timeout: int = 60
    ):
        """Initialize Gemini client"""
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        logger.info(f"[Gemini] API key provided: {api_key[:10]}...{api_key[-4:]}")
        
        self.client = genai.Client(api_key=api_key)
        
        if model in self.MODELS:
            self.model = self.MODELS[model]
        else:
            self.model = model
        
        self.max_retries = max_retries
        self.timeout = timeout
        
        logger.info(f"[Gemini] Initialized with model: {self.model}")
    
    def generate(
        self,
        topic: str,
        style: str,
        duration: int,
        additional_context: Optional[str] = None
    ) -> ContentResponse:
        """Generate video content using Gemini"""
        prompt = self._build_prompt(topic, style, duration, additional_context)
        
        try:
            raw_response = self._call_api_with_retry(prompt)
            content = self._parse_response(raw_response)
            
            logger.info("[Gemini] Content generated successfully")
            return content
            
        except Exception as e:
            logger.error(f"[Gemini] Generation failed: {e}")
            raise
    
    def _build_prompt(
        self,
        topic: str,
        style: str,
        duration: int,
        additional_context: Optional[str] = None
    ) -> str:
        """Build the generation prompt"""
        
        words_per_minute = 150
        target_words = int((duration / 60) * words_per_minute)
        
        if duration <= 30:
            hook_length = "1 sentence"
            script_sentences = "3-4"
        elif duration <= 45:
            hook_length = "1-2 sentences"
            script_sentences = "5-6"
        else:
            hook_length = "2 sentences"
            script_sentences = "7-8"
        
        prompt = f"""Create a {duration}-second YouTube Short script about: {topic}

Style: {style}
Target length: Approximately {target_words} words total

CRITICAL REQUIREMENTS:
1. Hook must grab attention in {hook_length}
2. Script must be exactly {script_sentences} clear sentences
3. Each sentence should be one complete thought
4. CTA must be engaging and natural (MAX 10 WORDS!)
5. main_visual_focus must be 1-2 words for stock video search (e.g. "hoopoe bird", "ocean waves")

{additional_context or ''}

MAIN_VISUAL_FOCUS RULES - CRITICAL:
- This is THE SINGLE TOPIC for ALL video footage
- Use GENERIC, COMMON terms that exist in stock video libraries
- NEVER use specific species names unless very common (dog, cat, lion, elephant OK; archerfish, pistol shrimp NOT OK)
- Prefer broader categories: "tropical fish" over "archerfish", "ocean creatures" over "pistol shrimp"
- Think: "What footage is actually available on Pexels?"
- Good examples: "tropical fish", "bird flying", "wildlife close up", "ocean life", "nature forest"
- Bad examples: "archerfish", "hoopoe bird", "pistol shrimp", "pangolin"
- When topic is a rare animal: use its habitat or category instead
  * Archerfish → "tropical fish underwater"
  * Hoopoe → "colorful bird"
  * Pangolin → "wildlife closeup"
  * Pistol shrimp → "ocean creatures"

OUTPUT FORMAT (valid JSON only):
{{
    "hook": "Attention-grabbing opening (1 sentence MAX)",
    "script": [
        "First sentence",
        "Second sentence",
        "Third sentence"
    ],
    "cta": "Call to action (SHORT - max 10 words)",
    "main_visual_focus": "1-2 word visual topic (e.g. 'hoopoe bird', 'pangolin animal', 'ocean waves')",
    "search_queries": [
        "noun phrase 1",
        "noun phrase 2", 
        "noun phrase 3"
    ],
    "metadata": {{
        "title": "Title (max 50 chars)",
        "description": "Description (max 100 chars)",
        "tags": ["tag1", "tag2", "tag3"]
    }}
}}

IMPORTANT: 
- Keep it CONCISE! 
- Return ONLY valid JSON
- NO markdown formatting
- NO explanations
- Complete all fields fully before ending response"""
        
        return prompt
    
    def _call_api_with_retry(self, prompt: str) -> str:
        """Call API with retry logic"""
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_api(prompt)
                
            except Exception as e:
                last_error = e
                logger.warning(f"[Gemini] Attempt {attempt}/{self.max_retries} failed: {e}")
                
                if attempt == self.max_retries:
                    raise last_error
                
                wait_time = 2 ** attempt
                logger.info(f"[Gemini] Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        raise last_error
    
    def _call_api(self, prompt: str) -> str:
        """Make actual API call using official SDK"""
        try:
            logger.info(f"[Gemini] Making API call with model: {self.model}")
            
            config = types.GenerateContentConfig(
                temperature=0.9,
                top_k=40,
                top_p=0.95,
                max_output_tokens=4096,
                safety_settings=[
                    types.SafetySetting(
                        category='HARM_CATEGORY_HATE_SPEECH',
                        threshold='BLOCK_NONE'
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_HARASSMENT',
                        threshold='BLOCK_NONE'
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                        threshold='BLOCK_NONE'
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_DANGEROUS_CONTENT',
                        threshold='BLOCK_NONE'
                    )
                ]
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            
            if response.text:
                logger.info(f"[Gemini] ✅ API call successful")
                return response.text
            
            raise RuntimeError(f"Empty response from Gemini API")
            
        except Exception as e:
            logger.error(f"[Gemini] ❌ API call failed with model {self.model}: {e}")
            raise RuntimeError(f"Gemini API call failed: {e}")
    
    def _parse_response(self, raw_text: str) -> ContentResponse:
        """Parse API response into structured content"""
        try:
            # Clean the response
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            if not cleaned.endswith("}"):
                logger.warning("[Gemini] Response appears incomplete, attempting to fix...")
                cleaned = self._fix_incomplete_json(cleaned)
            
            data = json.loads(cleaned)
            
            # Validate required fields
            required = ["hook", "script", "cta", "search_queries"]
            missing = [f for f in required if f not in data]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")
            
            # ✅ YENİ: main_visual_focus opsiyonel ama tercih edilir
            if "main_visual_focus" not in data:
                data["main_visual_focus"] = data["search_queries"][0] if data["search_queries"] else "nature"
                logger.warning(f"[Gemini] main_visual_focus missing, using fallback: {data['main_visual_focus']}")
            
            # Validate types
            if not isinstance(data["script"], list):
                raise ValueError("script must be a list")
            if not isinstance(data["search_queries"], list):
                raise ValueError("search_queries must be a list")
            
            # Ensure metadata exists
            if "metadata" not in data:
                data["metadata"] = {
                    "title": f"Amazing {data.get('hook', 'Video')[:40]}",
                    "description": "Check out this amazing short!",
                    "tags": ["shorts", "viral", "trending"]
                }
            
            return ContentResponse(
                hook=data["hook"].strip(),
                script=[s.strip() for s in data["script"]],
                cta=data["cta"].strip(),
                search_queries=[q.strip() for q in data["search_queries"]],
                main_visual_focus=data["main_visual_focus"].strip(),  # ✅ YENİ
                metadata=data["metadata"]
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"[Gemini] JSON parse error: {e}")
            logger.error(f"[Gemini] Raw response: {raw_text[:500]}")
            raise RuntimeError(f"Failed to parse Gemini response as JSON: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"[Gemini] Content validation error: {e}")
            raise RuntimeError(f"Invalid content structure: {e}")
    
    def _fix_incomplete_json(self, text: str) -> str:
        """Try to fix incomplete JSON"""
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        
        quotes = text.count('"')
        if quotes % 2 != 0:
            text += '"'
        
        text += "]" * open_brackets
        text += "}" * open_braces
        
        logger.info(f"[Gemini] Fixed JSON by adding {open_braces} braces and {open_brackets} brackets")
        return text
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            logger.info("[Gemini] Testing API connection...")
            
            response = self.client.models.generate_content(
                model=self.model,
                contents="Say 'Hello' in JSON format: {\"message\": \"Hello\"}"
            )
            
            if response.text:
                logger.info("[Gemini] ✅ API connection successful")
                return True
            
            logger.error("[Gemini] ❌ Empty response from API")
            return False
            
        except Exception as e:
            logger.error(f"[Gemini] ❌ API connection failed: {e}")
            return False
