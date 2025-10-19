"""
Gemini API Client for Content Generation - VIRAL OPTIMIZED
Uses Google's official genai SDK with viral pattern templates
"""

import json
import time
import logging
import random
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
    main_visual_focus: str
    metadata: Dict[str, Any]


# ============================================================================
# VIRAL HOOK PATTERNS - Based on 100M+ view analysis
# ============================================================================
HOOK_PATTERNS = {
    "curiosity_gap": [
        "WAIT— This {topic} fact changes everything you thought you knew.",
        "You've been doing {topic} wrong your entire life. Here's why:",
        "Scientists JUST discovered the truth about {topic}...",
        "Everyone thinks {topic} works like this. They're all wrong.",
        "The {topic} industry doesn't want you knowing this secret.",
        "This {topic} discovery got banned in 3 countries. Here's what they found:",
    ],
    
    "pattern_interrupt": [
        "STOP scrolling. What you're about to see will blow your mind about {topic}.",
        "Before you skip— this {topic} fact will haunt you forever.",
        "POV: You finally understand why {topic} is so bizarre.",
        "Hold up. Did you know {topic} actually does THIS?",
        "Pause everything. This {topic} secret is insane.",
    ],
    
    "shocking_stat": [
        "97% of people don't know this about {topic}. You're probably one of them.",
        "Only 1 in 500 people have seen what {topic} actually looks like.",
        "This {topic} happens every 3 seconds. Yet nobody talks about it.",
        "Experts say {topic} will change within 5 years. Here's how:",
    ],
    
    "story_hook": [
        "In 1847, someone discovered {topic}. What happened next was insane.",
        "There's a place where {topic} defies all logic. Let me show you.",
        "Imagine if {topic} disappeared tomorrow. This is what would happen:",
        "Someone tried to explain {topic}. They were called crazy. But they were right.",
    ],
    
    "controversial": [
        "Unpopular opinion: Everything you know about {topic} is a lie.",
        "I'm about to ruin {topic} for you forever. You ready?",
        "This {topic} myth needs to die. Here's the real story:",
        "Nobody talks about the dark side of {topic}. Until now.",
    ]
}

# Weight distribution for A/B testing
HOOK_PATTERN_WEIGHTS = {
    "curiosity_gap": 0.35,
    "pattern_interrupt": 0.25,
    "shocking_stat": 0.20,
    "story_hook": 0.15,
    "controversial": 0.05
}

# ============================================================================
# STORYTELLING ANGLES - Rotate to keep content fresh
# ============================================================================
STORYTELLING_ANGLES = [
    "historical_origin",
    "scientific_explanation",
    "hidden_secret",
    "future_prediction",
    "comparison",
    "behind_the_scenes",
    "myth_busting",
    "personal_impact",
    "extreme_example"
]

# ============================================================================
# EMPHASIS KEYWORDS - Auto-capitalize in script
# ============================================================================
EMPHASIS_KEYWORDS = {
    "NEVER", "ALWAYS", "SECRET", "HIDDEN", "SHOCKING", "INSANE",
    "BANNED", "ILLEGAL", "IMPOSSIBLE", "CRAZY", "VIRAL", "BREAKING",
    "URGENT", "WARNING", "STOP", "WAIT", "INSTANTLY", "FOREVER"
}


class GeminiClient:
    """Client for Gemini API interactions with viral optimization"""
    
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
        """Generate video content using Gemini with viral patterns"""
        # Select random hook pattern based on weights
        hook_type = random.choices(
            list(HOOK_PATTERN_WEIGHTS.keys()),
            weights=list(HOOK_PATTERN_WEIGHTS.values())
        )[0]
        
        # Select random storytelling angle
        angle = random.choice(STORYTELLING_ANGLES)
        
        logger.info(f"[Gemini] Using hook pattern: {hook_type}, angle: {angle}")
        
        prompt = self._build_viral_prompt(
            topic, style, duration, hook_type, angle, additional_context
        )
        
        try:
            raw_response = self._call_api_with_retry(prompt)
            content = self._parse_response(raw_response)
            
            # Post-process: Add emphasis to keywords
            content = self._add_emphasis(content)
            
            logger.info("[Gemini] Content generated successfully")
            return content
            
        except Exception as e:
            logger.error(f"[Gemini] Generation failed: {e}")
            raise
    
    def _build_viral_prompt(
        self,
        topic: str,
        style: str,
        duration: int,
        hook_type: str,
        angle: str,
        additional_context: Optional[str] = None
    ) -> str:
        """Build viral-optimized generation prompt"""
        
        words_per_minute = 160  # Slightly faster for better pacing
        target_words = int((duration / 60) * words_per_minute)
        
        # Dynamic sentence counts based on duration
        if duration <= 25:
            hook_sentences = "1 SHORT sentence"
            script_sentences = "2-3"
            cta_words = "5-7"
        elif duration <= 35:
            hook_sentences = "1 sentence"
            script_sentences = "3-4"
            cta_words = "6-8"
        else:
            hook_sentences = "1-2 sentences"
            script_sentences = "4-5"
            cta_words = "7-10"
        
        # Get sample hooks for inspiration
        hook_examples = HOOK_PATTERNS.get(hook_type, HOOK_PATTERNS["curiosity_gap"])
        sample_hooks = random.sample(hook_examples, min(2, len(hook_examples)))
        hook_inspiration = "\n".join([f"- {h.replace('{topic}', topic)}" for h in sample_hooks])
        
        # Angle descriptions
        angle_instructions = {
            "historical_origin": "Focus on WHERE this came from and WHY it started",
            "scientific_explanation": "Explain the science/mechanism in simple terms with concrete examples",
            "hidden_secret": "Reveal something most people don't know or misunderstand",
            "future_prediction": "Show how this will change or evolve in the near future",
            "comparison": "Compare this to something unexpected to create new perspective",
            "behind_the_scenes": "Show what happens behind closed doors or away from public view",
            "myth_busting": "Destroy a common misconception with hard facts",
            "personal_impact": "Connect directly to the viewer's daily life with 'YOU' language",
            "extreme_example": "Use the most extreme/unusual case to make the point memorable"
        }
        
        angle_guide = angle_instructions.get(angle, "Tell the story in an unexpected way")
        
        prompt = f"""Create a VIRAL {duration}-second YouTube Short script about: {topic}

VIRAL REQUIREMENTS (CRITICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. HOOK PATTERN: Use "{hook_type}" style
   Examples for inspiration:
   {hook_inspiration}

2. STORYTELLING ANGLE: {angle}
   → {angle_guide}

3. FIRST 3 SECONDS RULE:
   - Hook must be {hook_sentences} MAX
   - Must create immediate curiosity gap
   - Use concrete numbers, names, or shocking facts
   - NO generic phrases like "today we'll explore" or "let me tell you"
   
4. SCRIPT STRUCTURE ({script_sentences} sentences):
   - Each sentence = ONE complete thought
   - Every sentence should have visual potential (describable in stock footage)
   - Build tension → payoff → resolution
   - Use "YOU" language to connect with viewer
   
5. PACING:
   - Vary sentence length: short (3-5 words) → medium (6-10 words) → short
   - Never repeat similar sentence structures
   - Keep momentum building
   
6. CTA (Call to Action - {cta_words} words MAX):
   - Natural, conversational tone
   - Ask for ONE action only (comment/like/follow)
   - Tie back to the content theme
   
7. VISUAL FOCUS:
   - main_visual_focus must be 2-4 words for stock video search
   - Use GENERIC, COMMON terms that exist in Pexels/Pixabay
   - Think: "What footage is actually available?"
   - Good: "tropical fish", "city skyline", "desert landscape"
   - Bad: "archerfish", "hoopoe bird", "specific building"
   - When topic is rare: use broader category
     * Rare animal → "wildlife closeup" or habitat type
     * Specific person → their field/activity
     * Obscure place → region/landscape type

CONTENT RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Target word count: ~{target_words} words total
- NO filler words: "basically", "actually", "literally", "you know"
- NO clickbait that the content doesn't deliver
- Use POWER WORDS: never, always, only, secret, hidden, instant, shocking
- Include at least ONE number/statistic if relevant
- Style: {style}

{additional_context or ''}

OUTPUT FORMAT (valid JSON only - NO MARKDOWN):
{{
    "hook": "One sentence that stops the scroll (use {hook_type} pattern)",
    "script": [
        "First sentence (builds on hook)",
        "Second sentence (adds detail/evidence)",
        "Third sentence (delivers payoff or next layer)"
    ],
    "cta": "Short, natural call to action ({cta_words} words)",
    "main_visual_focus": "2-4 word search term for ALL video clips",
    "search_queries": [
        "specific scene 1 (noun phrase)",
        "specific scene 2 (noun phrase)",
        "specific scene 3 (noun phrase)"
    ],
    "metadata": {{
        "title": "Attention-grabbing title (40-50 chars, NO CLICKBAIT)",
        "description": "Hook viewer to watch (90-100 chars)",
        "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
    }}
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL:
- Return ONLY valid JSON
- NO markdown code blocks (```json)
- NO explanations before or after
- Complete ALL fields before ending response
- Make it IMPOSSIBLE to scroll past
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        return prompt
    
    def _add_emphasis(self, content: ContentResponse) -> ContentResponse:
        """Add emphasis to power keywords in script"""
        emphasized_script = []
        
        for sentence in content.script:
            words = sentence.split()
            emphasized_words = []
            
            for word in words:
                word_upper = word.upper().strip(".,!?")
                if word_upper in EMPHASIS_KEYWORDS:
                    # Capitalize the keyword
                    emphasized_words.append(word_upper)
                else:
                    emphasized_words.append(word)
            
            emphasized_script.append(" ".join(emphasized_words))
        
        content.script = emphasized_script
        return content
    
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
                temperature=0.95,  # Increased for more creativity
                top_k=50,  # Increased for more variety
                top_p=0.97,  # Slightly higher for better diversity
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
            
            # Handle main_visual_focus
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
                main_visual_focus=data["main_visual_focus"].strip(),
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
