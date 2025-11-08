"""
Prompt Compression for Token Reduction

Reduces LLM API costs by compressing prompts while preserving semantic meaning:
- Remove unnecessary whitespace
- Abbreviate common terms
- Remove redundant instructions
- Use token-efficient phrasing
- Dynamic compression based on prompt length

Compression Strategies:
1. Whitespace Normalization - Remove extra spaces, newlines
2. Abbreviation - Replace common terms (e.g., "carbon dioxide" -> "CO2")
3. Redundancy Removal - Remove repeated instructions
4. Template Optimization - Optimize system prompts
5. Token Counting - Measure compression ratio

Performance Targets:
- Token reduction: >20%
- Compression ratio: 20-30%
- Response quality: >95% accuracy (preserve semantic meaning)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available. Install with: pip install tiktoken")


logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """
    Result of prompt compression

    Attributes:
        original_prompt: Original prompt text
        compressed_prompt: Compressed prompt text
        original_tokens: Original token count
        compressed_tokens: Compressed token count
        compression_ratio: Compression ratio (0-1, lower is better)
        token_reduction: Token reduction percentage (0-100)
        cost_savings_usd: Estimated cost savings
    """
    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    token_reduction: float
    cost_savings_usd: float = 0.0

    def __str__(self) -> str:
        return (
            f"CompressionResult(\n"
            f"  original_tokens={self.original_tokens},\n"
            f"  compressed_tokens={self.compressed_tokens},\n"
            f"  token_reduction={self.token_reduction:.1f}%,\n"
            f"  compression_ratio={self.compression_ratio:.3f},\n"
            f"  cost_savings=${self.cost_savings_usd:.4f}\n"
            f")"
        )


# Common abbreviations for climate/carbon domain
ABBREVIATIONS = {
    # Carbon terms
    "carbon dioxide": "CO2",
    "carbon monoxide": "CO",
    "methane": "CH4",
    "nitrous oxide": "N2O",
    "greenhouse gas": "GHG",
    "greenhouse gases": "GHGs",
    "carbon footprint": "carbon fp",
    "emissions": "emis",
    "emission": "emis",

    # Energy terms
    "kilowatt-hour": "kWh",
    "kilowatt-hours": "kWh",
    "kilowatt": "kW",
    "megawatt": "MW",
    "gigawatt": "GW",
    "megawatt-hour": "MWh",
    "british thermal unit": "BTU",
    "british thermal units": "BTUs",
    "coefficient of performance": "COP",

    # Building terms
    "air conditioning": "AC",
    "heating ventilation and air conditioning": "HVAC",
    "heating, ventilation, and air conditioning": "HVAC",
    "insulation": "insul",
    "square feet": "sq ft",
    "square foot": "sq ft",
    "square meter": "sq m",
    "square meters": "sq m",

    # Units
    "kilograms": "kg",
    "kilogram": "kg",
    "tonnes": "t",
    "tonne": "t",
    "metric ton": "t",
    "metric tons": "t",
    "pounds": "lbs",
    "pound": "lb",
    "gallons": "gal",
    "gallon": "gal",

    # Time
    "annually": "per yr",
    "per year": "per yr",
    "monthly": "per mo",
    "per month": "per mo",

    # Common phrases
    "approximately": "~",
    "about": "~",
    "around": "~",
    "estimated": "est.",
    "average": "avg",
    "typical": "typ",
    "standard": "std",
    "efficiency": "eff",
    "residential": "res",
    "commercial": "comm",
    "industrial": "ind",
    "percentage": "%",
}


# Redundant phrases to remove
REDUNDANT_PHRASES = [
    # Politeness (can be removed in system prompts)
    r"\bplease\b",
    r"\bkindly\b",
    r"\bthank you\b",
    r"\bthanks\b",

    # Filler words
    r"\bbasically\b",
    r"\bactually\b",
    r"\bjust\b",
    r"\bsimply\b",
    r"\breally\b",
    r"\bvery\b",
    r"\bquite\b",
    r"\brather\b",

    # Redundant instructions
    r"as (?:accurately|precisely) as possible",
    r"to the best of your (?:ability|knowledge)",
    r"if (?:you can|possible)",
    r"(?:try to|make sure to|be sure to)",
]


# Token-efficient rephrasing rules
REPHRASE_RULES = [
    # Questions
    (r"What is the", "What's the"),
    (r"How do I", "How to"),
    (r"Can you tell me", "Tell me"),
    (r"Could you please", "Please"),
    (r"Would you please", "Please"),
    (r"I would like to know", "What is"),

    # Instructions
    (r"Calculate the total", "Calculate"),
    (r"Provide me with", "Provide"),
    (r"Give me", "Provide"),
    (r"Show me", "Show"),

    # Comparisons
    (r"in comparison to", "vs"),
    (r"compared to", "vs"),
    (r"versus", "vs"),

    # Logic
    (r"in order to", "to"),
    (r"due to the fact that", "because"),
    (r"for the purpose of", "to"),
]


class TokenCounter:
    """
    Count tokens using tiktoken

    Supports multiple models:
    - GPT-4, GPT-3.5: cl100k_base encoding
    - GPT-3: p50k_base encoding
    """

    def __init__(self, model: str = "gpt-4"):
        """
        Initialize token counter

        Args:
            model: Model name for tokenization
        """
        self.model = model

        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                logger.warning(f"Model {model} not found. Using cl100k_base encoding.")
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None
            logger.warning("tiktoken not available. Using character-based estimation.")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Args:
            text: Input text

        Returns:
            Token count
        """
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: estimate ~4 chars per token
            return len(text) // 4

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in chat messages

        Args:
            messages: List of chat messages

        Returns:
            Total token count
        """
        total = 0

        for message in messages:
            # Add tokens per message
            total += 4  # Every message has <im_start>, role, content, <im_end>

            for key, value in message.items():
                total += self.count_tokens(str(value))

        total += 2  # Add tokens for reply priming

        return total


class PromptCompressor:
    """
    Compress prompts to reduce token usage

    Applies multiple compression strategies while preserving semantic meaning.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        compression_threshold: int = 3000,
        aggressive_mode: bool = False,
    ):
        """
        Initialize prompt compressor

        Args:
            model: Model name for token counting
            compression_threshold: Only compress if tokens > threshold
            aggressive_mode: Apply more aggressive compression (may affect quality)
        """
        self.model = model
        self.compression_threshold = compression_threshold
        self.aggressive_mode = aggressive_mode

        # Initialize token counter
        self.token_counter = TokenCounter(model=model)

        logger.info(f"PromptCompressor initialized (model={model}, threshold={compression_threshold})")

    def compress(
        self,
        prompt: str,
        force: bool = False,
        preserve_quality: bool = True,
    ) -> CompressionResult:
        """
        Compress prompt text

        Args:
            prompt: Input prompt
            force: Force compression even if below threshold
            preserve_quality: Prioritize quality over compression ratio

        Returns:
            Compression result with metrics
        """
        # Count original tokens
        original_tokens = self.token_counter.count_tokens(prompt)

        # Check threshold
        if not force and original_tokens < self.compression_threshold:
            logger.debug(f"Prompt below threshold ({original_tokens} < {self.compression_threshold}). Skipping compression.")
            return CompressionResult(
                original_prompt=prompt,
                compressed_prompt=prompt,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                token_reduction=0.0,
            )

        # Apply compression strategies
        compressed = prompt

        # 1. Whitespace normalization
        compressed = self._normalize_whitespace(compressed)

        # 2. Apply abbreviations (only if not preserving quality)
        if not preserve_quality or self.aggressive_mode:
            compressed = self._apply_abbreviations(compressed)

        # 3. Remove redundant phrases
        compressed = self._remove_redundant_phrases(compressed)

        # 4. Apply rephrasing rules
        if not preserve_quality or self.aggressive_mode:
            compressed = self._apply_rephrase_rules(compressed)

        # Count compressed tokens
        compressed_tokens = self.token_counter.count_tokens(compressed)

        # Calculate metrics
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        token_reduction = ((original_tokens - compressed_tokens) / original_tokens * 100) if original_tokens > 0 else 0.0

        # Estimate cost savings (GPT-4: $0.03 per 1K tokens)
        cost_per_token = 0.00003  # $0.03 / 1000
        cost_savings = (original_tokens - compressed_tokens) * cost_per_token

        result = CompressionResult(
            original_prompt=prompt,
            compressed_prompt=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            token_reduction=token_reduction,
            cost_savings_usd=cost_savings,
        )

        logger.info(f"Compression complete: {original_tokens} -> {compressed_tokens} tokens ({token_reduction:.1f}% reduction)")

        return result

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace

        - Remove extra spaces
        - Remove trailing/leading whitespace
        - Collapse multiple newlines
        """
        # Remove leading/trailing whitespace
        text = text.strip()

        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)

        # Collapse multiple newlines (max 2)
        text = re.sub(r'\n\n+', '\n\n', text)

        # Remove spaces before punctuation
        text = re.sub(r' ([.,;:!?])', r'\1', text)

        return text

    def _apply_abbreviations(self, text: str) -> str:
        """
        Apply domain-specific abbreviations

        Replace common terms with abbreviations (e.g., "carbon dioxide" -> "CO2")
        """
        # Case-insensitive replacement
        for full, abbr in ABBREVIATIONS.items():
            # Use word boundaries to avoid partial replacements
            pattern = re.compile(r'\b' + re.escape(full) + r'\b', re.IGNORECASE)
            text = pattern.sub(abbr, text)

        return text

    def _remove_redundant_phrases(self, text: str) -> str:
        """
        Remove redundant phrases

        Remove filler words and unnecessary politeness
        """
        for pattern in REDUNDANT_PHRASES:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Clean up extra spaces after removal
        text = re.sub(r' +', ' ', text)
        text = text.strip()

        return text

    def _apply_rephrase_rules(self, text: str) -> str:
        """
        Apply token-efficient rephrasing

        Replace verbose phrases with concise alternatives
        """
        for pattern, replacement in REPHRASE_RULES:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def compress_messages(
        self,
        messages: List[Dict[str, str]],
        force: bool = False,
        preserve_quality: bool = True,
    ) -> Tuple[List[Dict[str, str]], CompressionResult]:
        """
        Compress chat messages

        Args:
            messages: List of chat messages
            force: Force compression even if below threshold
            preserve_quality: Prioritize quality over compression ratio

        Returns:
            Tuple of (compressed_messages, compression_result)
        """
        # Count original tokens
        original_tokens = self.token_counter.count_messages_tokens(messages)

        # Compress each message
        compressed_messages = []
        for msg in messages:
            compressed_msg = msg.copy()
            if "content" in msg:
                result = self.compress(msg["content"], force=force, preserve_quality=preserve_quality)
                compressed_msg["content"] = result.compressed_prompt
            compressed_messages.append(compressed_msg)

        # Count compressed tokens
        compressed_tokens = self.token_counter.count_messages_tokens(compressed_messages)

        # Calculate metrics
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        token_reduction = ((original_tokens - compressed_tokens) / original_tokens * 100) if original_tokens > 0 else 0.0

        # Estimate cost savings
        cost_per_token = 0.00003
        cost_savings = (original_tokens - compressed_tokens) * cost_per_token

        result = CompressionResult(
            original_prompt=str(messages),
            compressed_prompt=str(compressed_messages),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            token_reduction=token_reduction,
            cost_savings_usd=cost_savings,
        )

        return compressed_messages, result

    def optimize_system_prompt(self, system_prompt: str) -> str:
        """
        Optimize system prompt for token efficiency

        System prompts can be more aggressively compressed since they're
        typically technical instructions.

        Args:
            system_prompt: System prompt text

        Returns:
            Optimized system prompt
        """
        # Apply all compression strategies
        result = self.compress(
            system_prompt,
            force=True,
            preserve_quality=False,
        )

        return result.compressed_prompt


@dataclass
class CompressionMetrics:
    """
    Aggregate compression metrics

    Attributes:
        total_requests: Total compression requests
        total_original_tokens: Total original tokens
        total_compressed_tokens: Total compressed tokens
        total_cost_savings_usd: Total cost savings
    """
    total_requests: int = 0
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    total_cost_savings_usd: float = 0.0

    @property
    def avg_compression_ratio(self) -> float:
        """Average compression ratio"""
        if self.total_original_tokens == 0:
            return 1.0
        return self.total_compressed_tokens / self.total_original_tokens

    @property
    def avg_token_reduction(self) -> float:
        """Average token reduction percentage"""
        if self.total_original_tokens == 0:
            return 0.0
        return (1 - self.avg_compression_ratio) * 100

    def update(self, result: CompressionResult):
        """Update metrics with compression result"""
        self.total_requests += 1
        self.total_original_tokens += result.original_tokens
        self.total_compressed_tokens += result.compressed_tokens
        self.total_cost_savings_usd += result.cost_savings_usd

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_requests": self.total_requests,
            "total_original_tokens": self.total_original_tokens,
            "total_compressed_tokens": self.total_compressed_tokens,
            "avg_compression_ratio": self.avg_compression_ratio,
            "avg_token_reduction_pct": self.avg_token_reduction,
            "total_cost_savings_usd": self.total_cost_savings_usd,
        }


# Global metrics
_global_metrics = CompressionMetrics()


def get_compression_metrics() -> CompressionMetrics:
    """Get global compression metrics"""
    return _global_metrics


if __name__ == "__main__":
    """
    Demo and testing
    """
    print("=" * 80)
    print("GreenLang Prompt Compression Demo")
    print("=" * 80)

    # Initialize compressor
    compressor = PromptCompressor(compression_threshold=0)  # Always compress for demo

    # Test prompts
    test_prompts = [
        # Long, verbose prompt
        """
        Please can you tell me what is the carbon dioxide emissions from natural gas
        heating systems in residential buildings? I would like to know this as accurately
        as possible. Also, could you provide information about how this compares to other
        heating options? Thank you very much.
        """,

        # Technical prompt with abbreviations
        """
        Calculate the greenhouse gas emissions for a building that consumes 10,000
        kilowatt-hours of electricity per year and 500 british thermal units of
        natural gas for heating, ventilation, and air conditioning. The building
        is approximately 2,000 square feet.
        """,

        # Short prompt (should not be compressed much)
        "What is the carbon footprint of natural gas?",
    ]

    print("\n1. Testing prompt compression:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n   Test {i}:")
        print(f"   Original: {prompt.strip()[:80]}...")

        result = compressor.compress(prompt, force=True)
        print(f"   Compressed: {result.compressed_prompt[:80]}...")
        print(f"   {result}")

    # Test message compression
    print("\n2. Testing message compression:")
    messages = [
        {"role": "system", "content": "You are a helpful assistant for climate and carbon calculations. Please provide accurate information."},
        {"role": "user", "content": "What is the carbon dioxide emissions from natural gas heating?"},
    ]

    compressed_messages, result = compressor.compress_messages(messages, force=True)
    print(f"   {result}")

    # Test system prompt optimization
    print("\n3. Testing system prompt optimization:")
    system_prompt = """
    You are a helpful assistant specializing in climate change and carbon emissions.
    Please provide accurate and detailed information about greenhouse gas emissions,
    energy efficiency, and carbon footprint calculations. Try to be as precise as
    possible in your calculations. Thank you for your assistance.
    """

    optimized = compressor.optimize_system_prompt(system_prompt)
    print(f"   Original ({compressor.token_counter.count_tokens(system_prompt)} tokens):")
    print(f"   {system_prompt.strip()}")
    print(f"\n   Optimized ({compressor.token_counter.count_tokens(optimized)} tokens):")
    print(f"   {optimized}")

    print("\n" + "=" * 80)
