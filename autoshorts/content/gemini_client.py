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
    metadata: Dict[str, Any]


class GeminiClient:
    """Client for Gemini API interactions using official SDK"""
    
    # Available models
    MODELS = {
        "flash": "gemini-2.0-flash-exp",
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
        """
        Initialize Gemini client
        
        Args:
            api_key: Gemini API key
            model: Model to use (flash, pro, flash-8b, flash-thinking)
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        # Initialize the official client
        self.client = genai.Client(api_key=api_key)
        
        # Map short names to full model names
        if model in self.MODELS:
            self.model = self.MODELS[model]
        else:
            # If full model name provided, use it directly
            self.model = model
            
        self.max_retries = max_retries
        self.timeout = timeout
        
        logger.info(f"[Gemini] Using model: {self.model}")
    
    def generate(
        self,
        topic: str,
        style: str,
        duration: int,
        additional_context: Optional[str] = None
    ) -> ContentResponse:
        """
        Generate video content using Gemini
        
        Args:
            topic: Video topic/niche
            style: Content style
            duration: Target duration in seconds
            additional_context: Optional additional instructions
            
        Returns:
            ContentResponse with generated content
        """
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
        
        # Calculate approximate word count (150 words per minute of speech)
        words_per_minute = 150
        target_words = int((duration / 60) * words_per_minute)
        
        # Adjust for different durations
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
4. CTA must be engaging and natural
5. Search queries MUST be VISUAL and GENERIC - things you can SEE in stock videos

{additional_context or ''}

SEARCH QUERY RULES:
- Use CONCRETE, VISUAL nouns (clock, nature, city, people, hands, water)
- NEVER use abstract concepts (time, ideas, concepts, minute, second)
- NEVER use text/numbers that appear in script ("1 minute" ❌, "clock" ✅)
- Each query should return 100+ stock videos on Pexels
- Examples: "sunset ocean", "busy city street", "person working laptop"

OUTPUT FORMAT (valid JSON only):
{{
    "hook": "Attention-grabbing opening line",
    "script": [
        "First sentence of main content",
        "Second sentence with key point",
        "Third sentence with details",
        ...
    ],
    "cta": "Call to action",
    "search_queries": [
        "visual concrete noun 1",
        "visual concrete noun 2", 
        "visual concrete noun 3"
    ],
    "metadata": {{
        "title": "Catchy video title (max 60 chars)",
        "description": "SEO-optimized description",
        "tags": ["tag1", "tag2", "tag3"]
    }}
}}

IMPORTANT: Return ONLY valid JSON. No markdown, no explanations."""
        
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
                
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.info(f"[Gemini] Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        raise last_error
    
    def _call_api(self, prompt: str) -> str:
        """Make actual API call using official SDK"""
        
        try:
            # Configure generation settings
            config = types.GenerateContentConfig(
                temperature=0.9,
                top_k=40,
                top_p=0.95,
                max_output_tokens=2048,
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
            
            # Make the API call
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            
            # Extract text from response
            if response.text:
                return response.text
            
            raise RuntimeError(f"Empty response from Gemini API")
            
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")
    
    def _parse_response(self, raw_text: str) -> ContentResponse:
        """Parse API response into structured content"""
        
        try:
            # Clean the response (remove markdown code blocks if present)
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Parse JSON
            data = json.loads(cleaned)
            
            # Validate required fields
            required = ["hook", "script", "cta", "search_queries"]
            missing = [f for f in required if f not in data]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")
            
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
                metadata=data["metadata"]
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"[Gemini] JSON parse error: {e}")
            logger.error(f"[Gemini] Raw response: {raw_text[:500]}")
            raise RuntimeError(f"Failed to parse Gemini response as JSON: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"[Gemini] Content validation error: {e}")
            raise RuntimeError(f"Invalid content structure: {e}")
    
    def test_connection(self) -> bool:
        """Test API connection and credentials"""
        
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
