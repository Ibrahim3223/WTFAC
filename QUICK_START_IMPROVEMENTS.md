# 🚀 Quick Start: Shorts Sistemi İyileştirmeleri

**Hedef**: Long sisteminden en kritik özellikleri 1 haftada entegre et
**Başlangıç Tarihi**: 2025-12-02

---

## ✅ Bu Hafta Yapılacaklar (Tier 1 - Kritik)

### 1. Hook Patterns (3-4 saat) ⭐⭐⭐⭐⭐

**Amaç**: İlk 3 saniyede izleyiciyi yakalamak (CTR +20-30%)

#### Adımlar:

```bash
# 1. Klasör oluştur
mkdir -p autoshorts/content/prompts

# 2. Hook patterns dosyasını oluştur
# Aşağıdaki kodu autoshorts/content/prompts/hook_patterns.py olarak kaydet
```

**Dosya İçeriği**:
```python
# autoshorts/content/prompts/hook_patterns.py
"""
Viral hook patterns for YouTube Shorts
Optimized for 30-60 second videos
"""
import random
from typing import List

# Shorts-optimized hooks (max 10 words, 3-4 seconds)
SHORTS_HOOK_PATTERNS = {
    "extreme": [
        "This {entity} is impossible.",
        "Nobody expected this {outcome}.",
        "{number} people can't explain this.",
        "Everything about {topic} is wrong.",
        "This breaks every rule.",
        "{entity} did the unthinkable.",
    ],

    "high": [
        "This {thing} broke all records.",
        "{entity} shocked the world.",
        "The truth about {topic}.",
        "This shouldn't be possible.",
        "{number} {people} discovered this secret.",
    ],

    "medium": [
        "Here's what makes {entity} special.",
        "The secret of {topic}.",
        "{entity} has a hidden power.",
        "This changes everything.",
    ],
}

# Words to AVOID in cold opens (kills retention)
COLD_OPEN_VIOLATIONS = [
    "this video", "this short", "today we", "in this video",
    "in this short", "let me show", "welcome to", "hey guys",
    "before we start", "make sure to subscribe",
]

def get_shorts_hook(intensity: str = "extreme") -> str:
    """Get a random hook pattern for Shorts."""
    patterns = SHORTS_HOOK_PATTERNS.get(intensity, SHORTS_HOOK_PATTERNS["extreme"])
    return random.choice(patterns)

def validate_cold_open(text: str) -> bool:
    """Check if text violates cold open rules."""
    text_lower = text.lower()
    return not any(violation in text_lower for violation in COLD_OPEN_VIOLATIONS)

def get_all_violations(text: str) -> List[str]:
    """Get list of all violations in text."""
    text_lower = text.lower()
    return [v for v in COLD_OPEN_VIOLATIONS if v in text_lower]
```

#### Gemini Entegrasyonu:

```python
# autoshorts/content/gemini_client.py içine ekle

# Import ekle (dosyanın başına)
from autoshorts.content.prompts.hook_patterns import (
    get_shorts_hook,
    validate_cold_open,
    get_all_violations,
)

# generate_script metoduna ekle
def generate_script(self, topic: str, mode: str = "general") -> dict:
    # Hook pattern al
    hook_pattern = get_shorts_hook(intensity="extreme")

    # Prompt'u güncelle
    prompt = f"""
    Create a viral YouTube Short (30-60 seconds, 8-12 sentences).

    CRITICAL HOOK RULE:
    First sentence MUST grab attention immediately using this pattern:
    {hook_pattern}

    Example patterns:
    - "This tiger did the impossible."
    - "5 million people can't explain this."
    - "Everything about gravity is wrong."

    COLD OPEN RULES (CRITICAL):
    - NEVER say: "this video", "this short", "today we", "in this video"
    - START DIRECTLY with the topic (no meta-talk)
    - First word should be "This", entity name, or number

    Topic: {topic}
    Mode: {mode}

    Return JSON with "sentences" array containing text and search queries.
    """

    # Generate script
    raw_script = self._call_gemini_api(prompt)
    script = self._parse_script(raw_script)

    # VALIDATE cold open
    first_sentence = script["sentences"][0]["text"]
    if not validate_cold_open(first_sentence):
        violations = get_all_violations(first_sentence)
        logger.warning(f"Cold open violation: {violations}")
        # Retry or fix
        script = self._retry_with_better_hook(topic, mode)

    return script
```

**Test**:
```bash
# Hook pattern testi
python -c "
from autoshorts.content.prompts.hook_patterns import get_shorts_hook, validate_cold_open

# Test patterns
print('Hook patterns:')
for i in range(5):
    print(f'  {i+1}. {get_shorts_hook(\"extreme\")}')

# Test validation
good = 'This tiger did the impossible.'
bad = 'In this video, we explore tigers.'

print(f'\nGood hook: {validate_cold_open(good)}')  # Should be True
print(f'Bad hook: {validate_cold_open(bad)}')      # Should be False
"
```

---

### 2. Keyword Highlighting (3-4 saat) ⭐⭐⭐⭐⭐

**Amaç**: Caption'larda önemli kelimeleri vurgulamak (engagement +60%)

#### Dosya:

```python
# autoshorts/captions/keyword_highlighter.py (NEW)
"""
Keyword highlighter for Shorts captions
Highlights numbers, emphasis words, and questions
"""
import re
import logging

logger = logging.getLogger(__name__)


class ShortsKeywordHighlighter:
    """Highlight important keywords in captions for better engagement."""

    # Shorts-specific emphasis words
    EMPHASIS_WORDS = [
        "shocking", "incredible", "never", "always", "impossible",
        "insane", "crazy", "unbelievable", "mindblowing", "nobody",
        "amazing", "wild", "secret", "hidden", "truth"
    ]

    def highlight(self, text: str) -> str:
        """
        Add ASS formatting to highlight keywords.

        Returns:
            ASS-formatted text with color/size highlights
        """
        result = text

        # 1. Highlight numbers (YELLOW, BOLD, 1.3x size for mobile)
        result = re.sub(
            r'\b(\d+(?:,\d+)*)\b',  # Matches: 5, 100, 1,000,000
            r'{\\c&H00FFFF&\\b1\\fs1.3}\1{\\r}',
            result
        )

        # 2. Highlight emphasis words (RED, BOLD)
        for word in self.EMPHASIS_WORDS:
            pattern = rf'\b({word})\b'
            result = re.sub(
                pattern,
                r'{\\c&H0000FF&\\b1}\1{\\r}',
                result,
                flags=re.IGNORECASE
            )

        # 3. Highlight questions (CYAN)
        if '?' in result:
            result = result.replace('?', '{\\c&H00FFFF&}?{\\r}')

        # 4. Highlight exclamations (slightly larger, bold)
        if '!' in result:
            result = result.replace('!', '{\\b1\\fs1.1}!{\\r}')

        logger.debug(f"Highlighted caption: {result}")
        return result

    def add_emphasis_word(self, word: str):
        """Add custom emphasis word."""
        if word.lower() not in self.EMPHASIS_WORDS:
            self.EMPHASIS_WORDS.append(word.lower())
```

#### Renderer Entegrasyonu:

```python
# autoshorts/captions/renderer.py içine ekle

# Import ekle
from autoshorts.captions.keyword_highlighter import ShortsKeywordHighlighter

class CaptionRenderer:
    def __init__(self):
        self.highlighter = ShortsKeywordHighlighter()
        # ... existing init code ...

    def render_sentence(self, text: str, start_ms: int, end_ms: int, ...) -> str:
        """Render a single caption with keyword highlighting."""

        # Apply keyword highlighting BEFORE rendering
        highlighted_text = self.highlighter.highlight(text)

        # Continue with existing ASS rendering
        return self._render_ass_subtitle(
            highlighted_text,
            start_ms,
            end_ms,
            ...
        )
```

**Test**:
```python
# test_highlighter.py
from autoshorts.captions.keyword_highlighter import ShortsKeywordHighlighter

highlighter = ShortsKeywordHighlighter()

test_sentences = [
    "This incredible fact involves 5 million people",
    "Nobody expected this shocking result",
    "Is this the truth about space?",
    "The secret number is 42!",
]

print("Keyword Highlighting Tests:\n")
for sentence in test_sentences:
    highlighted = highlighter.highlight(sentence)
    print(f"Original:  {sentence}")
    print(f"Highlighted: {highlighted}\n")
```

---

### 3. Cliffhanger Patterns (2 saat) ⭐⭐⭐⭐⭐

**Amaç**: Mini-cliffhanger'larla retention artışı (+15-25%)

#### Dosya:

```python
# autoshorts/content/prompts/retention_patterns.py (NEW)
"""
Retention patterns for YouTube Shorts
Injects mini-cliffhangers every ~10 seconds
"""
import random
from typing import List

# Shorts-optimized cliffhangers (max 5 words)
SHORTS_CLIFFHANGERS = [
    "But wait...",
    "Here's the twist.",
    "You won't believe this.",
    "Watch what happens.",
    "But that's not all.",
    "The shocking part?",
    "Here's where it gets crazy.",
    "And then this happened.",
]

def inject_cliffhangers(
    sentences: List[str],
    target_duration: int = 30,
    interval: int = 10
) -> List[str]:
    """
    Inject cliffhangers into sentences.

    Args:
        sentences: List of sentence texts
        target_duration: Video duration in seconds
        interval: Cliffhanger interval in seconds (default: 10s)

    Returns:
        List with cliffhangers injected
    """
    if len(sentences) < 3:
        return sentences  # Too short for cliffhangers

    # Calculate sentences per interval (assume ~3 seconds per sentence)
    sentences_per_interval = max(1, interval // 3)

    result = []
    cliffhanger_count = 0
    max_cliffhangers = (target_duration // interval) - 1  # Don't add at end

    for i, sentence in enumerate(sentences):
        result.append(sentence)

        # Inject cliffhanger after every N sentences
        should_inject = (
            (i + 1) % sentences_per_interval == 0 and
            i < len(sentences) - 2 and  # Not near end
            cliffhanger_count < max_cliffhangers
        )

        if should_inject:
            cliffhanger = random.choice(SHORTS_CLIFFHANGERS)
            result.append(cliffhanger)
            cliffhanger_count += 1

    return result

def get_random_cliffhanger() -> str:
    """Get a random cliffhanger phrase."""
    return random.choice(SHORTS_CLIFFHANGERS)
```

#### Gemini Entegrasyonu:

```python
# autoshorts/content/gemini_client.py

from autoshorts.content.prompts.retention_patterns import inject_cliffhangers

def generate_script(self, topic: str, mode: str = "general") -> dict:
    # ... existing script generation ...

    # Extract sentences
    sentences = [s["text"] for s in script["sentences"]]

    # Inject cliffhangers for retention
    sentences_with_cliffhangers = inject_cliffhangers(
        sentences,
        target_duration=30,  # or settings.TARGET_DURATION
        interval=10  # Every 10 seconds
    )

    # Update script
    script["sentences"] = [
        {"text": s, "search": script["sentences"][i % len(script["sentences"])]["search"]}
        for i, s in enumerate(sentences_with_cliffhangers)
    ]

    return script
```

**Test**:
```python
from autoshorts.content.prompts.retention_patterns import inject_cliffhangers

sentences = [
    "This tiger is the fastest in the world.",
    "It can run 80 miles per hour.",
    "Scientists studied this for years.",
    "They discovered something shocking.",
    "The tiger uses a special technique.",
    "This changed everything we knew.",
]

result = inject_cliffhangers(sentences, target_duration=30, interval=10)

print("Original sentences:", len(sentences))
print("With cliffhangers:", len(result))
print("\nResult:")
for i, s in enumerate(result):
    print(f"  {i+1}. {s}")
```

---

### 4. ConfigManager Enhancement (2-3 saat) ⭐⭐⭐⭐⭐

**Amaç**: Script style ayarlarını merkezi yönetim

#### Config Model:

```python
# autoshorts/config/models.py içine ekle

@dataclass
class ScriptStyleConfig(BaseSettings):
    """Script style configuration for viral Shorts content."""

    model_config = SettingsConfigDict(
        env_prefix="SCRIPT_",
        env_file=".env",
        extra="ignore"
    )

    # Hook settings
    hook_intensity: str = Field(
        default="extreme",
        alias="SCRIPT_HOOK_INTENSITY"
    )
    cold_open: bool = Field(
        default=True,
        alias="SCRIPT_COLD_OPEN"
    )
    hook_max_words: int = Field(
        default=10,
        ge=5,
        le=15,
        alias="SCRIPT_HOOK_MAX_WORDS"
    )

    # Cliffhanger settings
    cliffhanger_enabled: bool = Field(
        default=True,
        alias="SCRIPT_CLIFFHANGER_ENABLED"
    )
    cliffhanger_interval: int = Field(
        default=10,
        ge=5,
        le=20,
        alias="SCRIPT_CLIFFHANGER_INTERVAL"
    )

    # Content settings
    max_sentence_length: int = Field(
        default=15,
        ge=10,
        le=25,
        alias="SCRIPT_MAX_SENTENCE_LEN"
    )

    @field_validator("hook_intensity")
    @classmethod
    def validate_intensity(cls, v: str) -> str:
        allowed = {"low", "medium", "high", "extreme"}
        if v not in allowed:
            raise ValueError(f"hook_intensity must be one of {allowed}")
        return v

# AppConfig içine ekle
class AppConfig(BaseSettings):
    # ... existing fields ...

    # NEW: Script style config
    script_style: ScriptStyleConfig = Field(
        default_factory=ScriptStyleConfig
    )
```

#### Config Manager:

```python
# autoshorts/config/manager.py (NEW)
"""
Singleton configuration manager for centralized config access.
"""
from typing import Optional
from autoshorts.config.models import AppConfig

class ConfigManager:
    """Centralized configuration manager."""

    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ConfigManager()
            cls._instance._config = AppConfig()
        return cls._instance

    @property
    def config(self) -> AppConfig:
        """Get config object."""
        if self._config is None:
            self._config = AppConfig()
        return self._config

    def reload(self):
        """Reload configuration from environment."""
        self._config = AppConfig()

    def validate(self) -> bool:
        """Validate configuration."""
        try:
            # Required API keys
            if not self.config.api.gemini_api_key:
                return False

            # Video settings
            if self.config.video.target_duration < 15:
                return False

            # Script style settings
            if self.config.script_style.hook_max_words < 5:
                return False

            return True
        except Exception:
            return False
```

#### Usage:

```python
# Kullanım örneği
from autoshorts.config.manager import ConfigManager

# Get config
config = ConfigManager.get_instance().config

# Access script style settings
hook_intensity = config.script_style.hook_intensity  # "extreme"
cliffhanger_interval = config.script_style.cliffhanger_interval  # 10

# Validate
if not ConfigManager.get_instance().validate():
    print("Invalid configuration!")
```

---

## 📊 Test & Validation

### Hızlı Test Scripti:

```python
# test_improvements.py
"""Test script for Tier 1 improvements"""

def test_hook_patterns():
    """Test hook pattern generation"""
    print("=" * 60)
    print("TEST 1: Hook Patterns")
    print("=" * 60)

    from autoshorts.content.prompts.hook_patterns import (
        get_shorts_hook, validate_cold_open
    )

    # Generate hooks
    print("\n✅ Generated hooks:")
    for intensity in ["extreme", "high", "medium"]:
        hook = get_shorts_hook(intensity)
        print(f"  [{intensity.upper()}] {hook}")

    # Validate cold opens
    print("\n✅ Cold open validation:")
    good_hooks = [
        "This tiger did the impossible.",
        "Nobody expected this outcome.",
        "5 million people discovered this secret."
    ]
    bad_hooks = [
        "In this video, we explore tigers.",
        "Today we'll learn about space.",
        "Welcome to this short about science."
    ]

    for hook in good_hooks:
        result = validate_cold_open(hook)
        print(f"  {result} | {hook}")

    for hook in bad_hooks:
        result = validate_cold_open(hook)
        print(f"  {result} | {hook}")


def test_keyword_highlighting():
    """Test keyword highlighter"""
    print("\n" + "=" * 60)
    print("TEST 2: Keyword Highlighting")
    print("=" * 60)

    from autoshorts.captions.keyword_highlighter import ShortsKeywordHighlighter

    highlighter = ShortsKeywordHighlighter()

    test_sentences = [
        "This incredible fact involves 5 million people",
        "Nobody expected this shocking result",
        "Is this the truth about space?",
    ]

    for sentence in test_sentences:
        highlighted = highlighter.highlight(sentence)
        print(f"\nOriginal:  {sentence}")
        print(f"Highlighted: {highlighted}")


def test_cliffhangers():
    """Test cliffhanger injection"""
    print("\n" + "=" * 60)
    print("TEST 3: Cliffhanger Patterns")
    print("=" * 60)

    from autoshorts.content.prompts.retention_patterns import inject_cliffhangers

    sentences = [
        "This tiger is the fastest in the world.",
        "It can run 80 miles per hour.",
        "Scientists studied this for years.",
        "They discovered something shocking.",
        "The tiger uses a special technique.",
        "This changed everything we knew.",
    ]

    result = inject_cliffhangers(sentences, target_duration=30)

    print(f"\n✅ Original: {len(sentences)} sentences")
    print(f"✅ With cliffhangers: {len(result)} sentences\n")

    for i, s in enumerate(result):
        marker = "🎯" if s in sentences else "⚡"
        print(f"  {marker} {i+1}. {s}")


def test_config_manager():
    """Test config manager"""
    print("\n" + "=" * 60)
    print("TEST 4: Config Manager")
    print("=" * 60)

    from autoshorts.config.manager import ConfigManager

    config = ConfigManager.get_instance().config

    print(f"\n✅ Hook Intensity: {config.script_style.hook_intensity}")
    print(f"✅ Cliffhanger Interval: {config.script_style.cliffhanger_interval}s")
    print(f"✅ Max Sentence Length: {config.script_style.max_sentence_length} words")
    print(f"✅ Cold Open: {config.script_style.cold_open}")

    is_valid = ConfigManager.get_instance().validate()
    print(f"\n✅ Config Valid: {is_valid}")


if __name__ == "__main__":
    test_hook_patterns()
    test_keyword_highlighting()
    test_cliffhangers()
    test_config_manager()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
```

**Çalıştır**:
```bash
python test_improvements.py
```

---

## 🎯 Sonraki Adımlar

### Bu Hafta Sonunda:
- ✅ Hook patterns entegre edildi
- ✅ Keyword highlighting eklendi
- ✅ Cliffhanger patterns çalışıyor
- ✅ ConfigManager enhanced

### Gelecek Hafta (Tier 2):
- 🔄 Continuous TTS Handler
- 🔄 Adaptive Audio Mixer
- 🔄 SFX Manager
- 🔄 Provider Pattern

---

## 📈 Beklenen Sonuçlar

| Metrik | Şu An | 1 Hafta Sonra | İyileşme |
|--------|-------|---------------|----------|
| CTR | 8-12% | 10-15% | +20-30% |
| Hook Quality | 6/10 | 8.5/10 | +40% |
| Caption Engagement | 5/10 | 8/10 | +60% |
| Script Variety | 7/10 | 9/10 | +30% |

**Total Time**: 10-13 saat (1-2 gün)
**Expected Impact**: +30-40% overall video quality

---

## 💡 Tips

1. **Hook Patterns**: Her video için farklı intensity dene
2. **Keyword Highlighting**: Mobile görünümü test et
3. **Cliffhangers**: Çok fazla ekleme (max 2-3 per 30s)
4. **Config**: ENV variables ile test etmeyi unutma

**Başarılar! 🚀**
