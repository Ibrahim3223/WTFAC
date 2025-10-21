"""
Gemini API Client - TOPIC-DRIVEN VIRAL OPTIMIZATION
No hardcoded modes - AI analyzes topic and adapts everything dynamically
"""

import json
import time
import logging
import random
import re
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
# UNIVERSAL VIRAL HOOK FORMULAS - Work for ANY topic
# ============================================================================
HOOK_FORMULAS = [
    # Curiosity Gap
    "WAIT— This {topic_keyword} fact changes everything you thought you knew.",
    "You've been wrong about {topic_keyword} your entire life. Here's why:",
    "Scientists JUST discovered the truth about {topic_keyword}...",
    "Everyone thinks {topic_keyword} works like this. They're ALL wrong.",
    "This {topic_keyword} secret was hidden for decades. Until now.",
    "97% of people don't know this about {topic_keyword}.",
    
    # Pattern Interrupt
    "STOP scrolling. What you're about to see will blow your mind.",
    "Before you skip— this {topic_keyword} truth is insane.",
    "POV: You finally understand why {topic_keyword} is so bizarre.",
    "Hold up. Did you know {topic_keyword} actually does THIS?",
    "Pause everything. This {topic_keyword} fact is unbelievable.",
    
    # Shocking Revelation
    "In {time_period}, something happened that changed {topic_keyword} forever.",
    "This {topic_keyword} discovery got banned in 3 countries.",
    "Only 1 in 500 people have seen what {topic_keyword} looks like.",
    "Nobody talks about the REAL reason behind {topic_keyword}.",
    
    # Story Hook
    "There's a place where {topic_keyword} defies all logic.",
    "Imagine if {topic_keyword} disappeared tomorrow. This would happen:",
    "Someone tried to explain {topic_keyword}. They were called crazy. But they were right.",
    
    # Controversy
    "Unpopular opinion: Everything you know about {topic_keyword} is a lie.",
    "I'm about to ruin {topic_keyword} for you forever. Ready?",
    "This {topic_keyword} myth needs to die. Here's the truth:",
    
    # Urgent/Breaking
    "This just happened with {topic_keyword}. Nobody's talking about it.",
    "Breaking: {topic_keyword} isn't what you think anymore.",
    
    # Simplification
    "The {topic_keyword} explained in 30 seconds. Watch.",
    "This is what they don't tell you about {topic_keyword}.",
]


# ============================================================================
# UNIVERSAL CTA STRATEGIES
# ============================================================================
CTA_STRATEGIES = {
    "comment_question": [
        "Drop a 💬 if you knew this!",
        "Comment your reaction below!",
        "What's your take? Comment now!",
        "Did this surprise you? Let me know!",
        "Have you experienced this? Share!",
    ],
    
    "engagement_challenge": [
        "Tag someone who needs to see this!",
        "Share this if it blew your mind!",
        "Save this before it's too late!",
        "Screen record and repost!",
        "Send this to your group chat!",
    ],
    
    "follow_tease": [
        "Follow for daily mind-blowing facts!",
        "Part 2 drops tonight—follow!",
        "More coming tomorrow. Don't miss it!",
        "Hit follow—you'll want the next one!",
        "Follow so you don't miss what's next!",
    ],
    
    "controversy_spark": [
        "Agree or disagree? Comment your take!",
        "This is controversial—what do YOU think?",
        "Hot take. Am I wrong? Comment!",
        "Change my mind in the comments!",
        "Unpopular opinion? You decide!",
    ],
    
    "value_anchor": [
        "Bookmark this—you'll need it later!",
        "Save this tip before you forget!",
        "Screenshot this for future reference!",
        "Keep this handy for next time!",
        "Save this—it's a game changer!",
    ],
    
    "continuation": [
        "Wait for the next part—it gets crazier!",
        "This is just the beginning. Follow!",
        "Part 2 is even more shocking!",
        "The twist? That's in tomorrow's video!",
        "You won't BELIEVE what happens next!",
    ],
}


# ============================================================================
# TOPIC ANALYZER - Extract keywords and category
# ============================================================================
class TopicAnalyzer:
    """Smart topic analysis to extract themes and keywords"""
    
    # Category detection patterns
    CATEGORY_PATTERNS = {
        "education": ["facts", "learn", "explain", "teach", "science", "history", "geography", "knowledge"],
        "entertainment": ["story", "funny", "comedy", "movie", "film", "celebrity", "drama", "imagine"],
        "howto": ["how to", "fix", "repair", "diy", "tutorial", "guide", "tips", "tricks", "hack"],
        "news": ["news", "update", "daily", "latest", "breaking", "current", "today", "headlines"],
        "tech": ["tech", "ai", "robot", "future", "innovation", "gadget", "computer", "digital"],
        "lifestyle": ["life", "habit", "wellness", "health", "fitness", "food", "cooking", "home"],
        "travel": ["travel", "country", "place", "destination", "city", "world", "explore", "visit"],
        "sports": ["sport", "game", "player", "team", "match", "cricket", "football", "athlete"],
        "animals": ["animal", "wildlife", "pet", "nature", "creature", "species", "zoo"],
        "inspiration": ["motivation", "inspire", "success", "quote", "wisdom", "life lesson"],
    }
    
    @staticmethod
    def analyze(topic: str) -> Dict[str, Any]:
        """
        Analyze topic and extract key information.
        
        Args:
            topic: Channel topic description
            
        Returns:
            Dict with category, keywords, and emotional_tone
        """
        topic_lower = topic.lower()
        
        # Detect category
        category = "education"  # Default
        max_matches = 0
        
        for cat, patterns in TopicAnalyzer.CATEGORY_PATTERNS.items():
            matches = sum(1 for pattern in patterns if pattern in topic_lower)
            if matches > max_matches:
                max_matches = matches
                category = cat
        
        # Extract main keywords (nouns/important words)
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
                     "of", "with", "by", "from", "about", "that", "this", "these", "those"}
        
        words = re.findall(r'\b[a-z]+\b', topic_lower)
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Get top 3-5 most meaningful keywords
        main_keywords = keywords[:5] if len(keywords) >= 5 else keywords
        
        # Detect emotional tone
        emotional_tone = "neutral"
        if any(word in topic_lower for word in ["horror", "scary", "dark", "creepy", "mystery"]):
            emotional_tone = "suspenseful"
        elif any(word in topic_lower for word in ["fun", "funny", "comedy", "humor", "joke"]):
            emotional_tone = "playful"
        elif any(word in topic_lower for word in ["motivat", "inspir", "success", "achieve"]):
            emotional_tone = "inspirational"
        elif any(word in topic_lower for word in ["breaking", "urgent", "alert", "warning"]):
            emotional_tone = "urgent"
        elif any(word in topic_lower for word in ["cute", "adorable", "sweet", "wholesome"]):
            emotional_tone = "wholesome"
        
        return {
            "category": category,
            "keywords": main_keywords,
            "emotional_tone": emotional_tone,
            "primary_keyword": main_keywords[0] if main_keywords else "content"
        }


class GeminiClient:
    """Topic-driven viral content generator"""
    
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
        
        logger.info(f"[Gemini] API key: {api_key[:10]}...{api_key[-4:]}")
        
        self.client = genai.Client(api_key=api_key)
        self.model = self.MODELS.get(model, model)
        self.max_retries = max_retries
        self.timeout = timeout
        
        logger.info(f"[Gemini] Model: {self.model}")
    
    def generate(
        self,
        topic: str,
        style: str,
        duration: int,
        additional_context: Optional[str] = None
    ) -> ContentResponse:
        """Generate content analyzing topic dynamically"""
        
        # Analyze topic to understand what we're working with
        analysis = TopicAnalyzer.analyze(topic)
        
        logger.info(f"[Gemini] Topic analysis:")
        logger.info(f"  Category: {analysis['category']}")
        logger.info(f"  Keywords: {analysis['keywords']}")
        logger.info(f"  Tone: {analysis['emotional_tone']}")
        
        # Select random hook formula
        hook_formula = random.choice(HOOK_FORMULAS)
        
        # Select random CTA strategy
        cta_strategy = random.choice(list(CTA_STRATEGIES.keys()))
        
        prompt = self._build_smart_prompt(
            topic, style, duration, analysis,
            hook_formula, cta_strategy, additional_context
        )
        
        try:
            raw_response = self._call_api_with_retry(prompt)
            content = self._parse_response(raw_response)
            
            # Enhance metadata with topic analysis
            content = self._enhance_metadata(content, analysis, topic)
            
            logger.info("[Gemini] ✅ Content generated successfully")
            return content
            
        except Exception as e:
            logger.error(f"[Gemini] ❌ Generation failed: {e}")
            raise
    
    def _build_smart_prompt(
        self,
        topic: str,
        style: str,
        duration: int,
        analysis: Dict[str, Any],
        hook_formula: str,
        cta_strategy: str,
        additional_context: Optional[str] = None
    ) -> str:
        """Build intelligent prompt based on topic analysis"""
        
        words_per_minute = 165
        target_words = int((duration / 60) * words_per_minute)
        
        # Dynamic structure
        if duration <= 25:
            structure = "Hook (1 sentence) + Core (2 sentences) + CTA (5-7 words)"
            script_count = 2
        elif duration <= 35:
            structure = "Hook (1 sentence) + Core (3 sentences) + CTA (6-8 words)"
            script_count = 3
        else:
            structure = "Hook (1 sentence) + Core (4-5 sentences) + CTA (7-10 words)"
            script_count = 4
        
        # Get sample CTAs
        cta_examples = random.sample(CTA_STRATEGIES[cta_strategy], min(2, len(CTA_STRATEGIES[cta_strategy])))
        
        # Build topic-specific context
        topic_context = f"""
TOPIC ANALYSIS:
• Category: {analysis['category'].upper()}
• Primary focus: {analysis['primary_keyword']}
• Emotional tone: {analysis['emotional_tone']}
• Key themes: {', '.join(analysis['keywords'][:3])}
"""
        
        prompt = f"""Create a VIRAL {duration}-second YouTube Short about:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 TOPIC: {topic}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{topic_context}

🎯 VIRAL FORMULA:

1. HOOK (First 3 seconds - MAKE IT IMPOSSIBLE TO SCROLL):
   Use this formula: "{hook_formula}"
   
   Replace {{topic_keyword}} with the most attention-grabbing word from the topic.
   Replace {{time_period}} with specific dates if relevant.
   
   Hook MUST:
   ✓ Stop scrolling in 1 second
   ✓ Use POWER WORDS: WAIT, STOP, NEVER, SECRET, SHOCKING
   ✓ Create immediate curiosity gap
   ✓ Include specific number/name/place if possible
   ✗ NO generic phrases ("today we'll explore")
   ✗ NO boring starts

2. SCRIPT ({script_count} sentences - BUILD THE STORY):
   Structure: {structure}
   
   Each sentence MUST:
   ✓ Be ONE complete thought (6-15 words)
   ✓ Have visual potential (can find stock footage)
   ✓ Build: tension → insight → payoff
   ✓ Use "YOU" language to connect
   ✓ Include specific details (numbers, names, facts)
   
   Variety is KEY:
   • Mix short (4-6 words) with medium (8-12 words)
   • Never repeat sentence structures
   • Keep momentum building
   • Each sentence reveals NEW information

3. CTA (Call to Action - {cta_strategy} strategy):
   Examples from this strategy:
   {chr(10).join(f"   • {ex}" for ex in cta_examples)}
   
   Your CTA MUST:
   ✓ Be natural and conversational
   ✓ Tie directly to the content
   ✓ Ask for ONE clear action
   ✗ NOT generic "like and subscribe"

4. VISUAL STRATEGY:
   main_visual_focus: 2-4 words for PRIMARY stock footage
   • Use GENERIC, COMMON terms that exist in Pexels/Pixabay
   • Think: "What footage is ACTUALLY available?"
   • Examples: "ocean waves", "city skyline", "forest path"
   • NOT: "specific landmark", "rare animal", "unique person"
   
   search_queries: {script_count + 1} specific shots
   • Each = noun phrase describing one visual
   • Variety: wide shots, close-ups, action, beauty
   • Must support narrative flow

5. PACING & QUALITY:
   • Target: ~{target_words} words total
   • Style: {style}
   • Emotional tone: {analysis['emotional_tone']}
   • NO filler words: "basically", "actually", "literally"
   • USE power words naturally: never, always, only, secret
   • Include at least ONE statistic/number if relevant

6. SEO OPTIMIZATION:
   title: 
   • 40-60 characters
   • Front-load main keyword
   • Create curiosity gap
   • Format: "[Hook Element] | [Keyword]" OR "[Number] [Topic] [Benefit]"
   
   description:
   • First 100 chars = mobile preview (CRITICAL)
   • Expand on hook, create urgency
   • Natural keyword inclusion
   • End with question or CTA
   
   tags:
   • 5 strategic tags:
     * 1 viral broad term
     * 2-3 niche specific terms  
     * 1 long-tail phrase
   • Based on topic category: {analysis['category']}
   • Include year: 2025

{additional_context or ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (VALID JSON ONLY - NO MARKDOWN):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
    "hook": "One powerful sentence using the formula above",
    "script": [
        "First sentence building on hook",
        "Second sentence adding evidence/detail",
        "Third sentence delivering insight/payoff"
    ],
    "cta": "Natural action call using {cta_strategy} (6-9 words)",
    "main_visual_focus": "2-4 word generic search term",
    "search_queries": [
        "wide establishing shot",
        "medium action shot",
        "close up detail",
        "beauty transition shot"
    ],
    "metadata": {{
        "title": "40-60 char SEO-optimized title with hook element",
        "description": "100 char mobile-optimized description expanding the hook",
        "tags": ["viral_term", "niche_keyword", "specific_phrase", "category_tag", "2025"]
    }}
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Return ONLY valid JSON (no ```json blocks)
✓ Complete ALL fields
✓ Make it IMPOSSIBLE to scroll past
✓ Every word must EARN its place
✓ Optimize for WATCH TIME + ENGAGEMENT
✓ Be SPECIFIC, not generic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        return prompt
    
    def _enhance_metadata(
        self, 
        content: ContentResponse,
        analysis: Dict[str, Any],
        topic: str
    ) -> ContentResponse:
        """Enhance metadata with smart topic analysis"""
        
        # Smart tag selection based on category
        category_tags = {
            "education": ["educational", "learn", "knowledge", "facts", "didyouknow"],
            "entertainment": ["entertainment", "fun", "interesting", "story", "amazing"],
            "howto": ["diy", "howto", "tutorial", "tips", "lifehacks"],
            "news": ["news", "update", "breaking", "latest", "current"],
            "tech": ["tech", "technology", "ai", "innovation", "future"],
            "lifestyle": ["lifestyle", "wellness", "health", "habits", "daily"],
            "travel": ["travel", "explore", "world", "adventure", "destination"],
            "sports": ["sports", "athletic", "game", "player", "highlights"],
            "animals": ["animals", "wildlife", "nature", "pets", "creatures"],
            "inspiration": ["motivation", "inspiration", "success", "mindset", "goals"],
        }
        
        # Get category-specific tags
        category = analysis.get("category", "education")
        specific_tags = category_tags.get(category, ["interesting", "amazing"])
        
        # Build strategic tag list
        strategic_tags = []
        
        # Add Gemini-generated tags (keep them)
        if "tags" in content.metadata:
            strategic_tags.extend(content.metadata["tags"][:5])
        
        # Add viral base tags
        viral_tags = ["shorts", "viral", "trending", "fyp", "youtube shorts"]
        strategic_tags.extend(random.sample(viral_tags, 2))
        
        # Add category-specific tags
        strategic_tags.extend(random.sample(specific_tags, 2))
        
        # Add topic keywords as tags
        for keyword in analysis.get("keywords", [])[:2]:
            if len(keyword) > 3:
                strategic_tags.append(keyword)
        
        # Add year
        strategic_tags.append("2025")
        
        # Deduplicate and limit
        strategic_tags = list(dict.fromkeys(strategic_tags))[:20]
        
        content.metadata["tags"] = strategic_tags
        
        # Ensure description has hashtags
        if "description" in content.metadata:
            desc = content.metadata["description"]
            if "#" not in desc:
                # Add relevant hashtags
                hashtags = [f"#{tag.replace(' ', '')}" for tag in strategic_tags[:3]]
                desc += f"\n\n{' '.join(hashtags)}"
                content.metadata["description"] = desc
        
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
        """Make API call"""
        try:
            config = types.GenerateContentConfig(
                temperature=0.92,
                top_k=60,
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
                logger.info("[Gemini] ✅ API successful")
                return response.text
            
            raise RuntimeError("Empty response")
            
        except Exception as e:
            logger.error(f"[Gemini] ❌ API failed: {e}")
            raise
    
    def _parse_response(self, raw_text: str) -> ContentResponse:
        """Parse response"""
        try:
            # Clean
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Fix incomplete JSON
            if not cleaned.endswith("}"):
                logger.warning("[Gemini] Fixing incomplete JSON...")
                cleaned = self._fix_incomplete_json(cleaned)
            
            data = json.loads(cleaned)
            
            # Validate
            required = ["hook", "script", "cta", "search_queries"]
            missing = [f for f in required if f not in data]
            if missing:
                raise ValueError(f"Missing: {missing}")
            
            # Ensure main_visual_focus
            if "main_visual_focus" not in data:
                data["main_visual_focus"] = data["search_queries"][0] if data["search_queries"] else "nature landscape"
            
            # Validate types
            if not isinstance(data["script"], list):
                raise ValueError("script must be list")
            if not isinstance(data["search_queries"], list):
                raise ValueError("search_queries must be list")
            
            # Ensure metadata
            if "metadata" not in data:
                data["metadata"] = {
                    "title": f"{data['hook'][:50]}...",
                    "description": f"{data['hook']} Watch to learn more!",
                    "tags": ["shorts", "viral", "trending", "amazing", "2025"]
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
            logger.error(f"[Gemini] JSON error: {e}")
            logger.error(f"[Gemini] Raw: {raw_text[:500]}")
            raise RuntimeError(f"Parse failed: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"[Gemini] Validation error: {e}")
            raise RuntimeError(f"Invalid structure: {e}")
    
    def _fix_incomplete_json(self, text: str) -> str:
        """Fix incomplete JSON"""
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        
        quotes = text.count('"')
        if quotes % 2 != 0:
            text += '"'
        
        text += "]" * open_brackets
        text += "}" * open_braces
        
        return text
    
    def test_connection(self) -> bool:
        """Test connection"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents='Say "OK" in JSON: {"status": "OK"}'
            )
            
            if response.text:
                logger.info("[Gemini] ✅ Connection OK")
                return True
            return False
            
        except Exception as e:
            logger.error(f"[Gemini] ❌ Connection failed: {e}")
            return False
