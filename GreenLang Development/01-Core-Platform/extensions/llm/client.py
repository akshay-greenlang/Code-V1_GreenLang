"""
LLM Client
==========

Client wrapper for Large Language Model integrations.

Author: AI Team
Created: 2025-11-21
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
import json
from abc import ABC, abstractmethod

from greenlang.llm.config import LLMConfig, ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class LLMMessage:
    """A message in an LLM conversation."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionOptions:
    """Options for LLM completion."""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    n: int = 1  # Number of completions


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: Dict[str, int]  # tokens used
    finish_reason: str
    created_at: datetime
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(
        self,
        messages: List[LLMMessage],
        options: CompletionOptions
    ) -> LLMResponse:
        """Generate completion from messages."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    def complete(
        self,
        messages: List[LLMMessage],
        options: CompletionOptions
    ) -> LLMResponse:
        """Generate mock completion."""
        # Simulate processing time
        time.sleep(0.1)

        # Generate mock response based on last message
        last_message = messages[-1] if messages else LLMMessage("user", "")

        response_content = f"Mock response to: {last_message.content[:50]}..."

        return LLMResponse(
            content=response_content,
            model="mock-model",
            usage={
                "prompt_tokens": sum(len(m.content.split()) for m in messages) * 2,
                "completion_tokens": len(response_content.split()) * 2,
                "total_tokens": (sum(len(m.content.split()) for m in messages) + len(response_content.split())) * 2
            },
            finish_reason="stop",
            created_at=datetime.now(),
            latency_ms=100,
            metadata={"provider": "mock"}
        )

    def is_available(self) -> bool:
        """Mock provider is always available."""
        return True


class LLMClient:
    """
    Client for interacting with Large Language Models.

    Provides a unified interface for different LLM providers with
    built-in retry logic, caching, and monitoring.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM client."""
        self.config = config or LLMConfig()
        self.provider = self._initialize_provider()
        self.request_count = 0
        self.total_tokens = 0
        self.total_latency = 0
        self._cache: Dict[str, LLMResponse] = {}

    def _initialize_provider(self) -> BaseLLMProvider:
        """Initialize the appropriate provider."""
        if self.config.provider == ModelProvider.MOCK:
            return MockLLMProvider()
        elif self.config.provider == ModelProvider.OPENAI:
            # Would initialize OpenAI provider here
            logger.warning("OpenAI provider not implemented, using mock")
            return MockLLMProvider()
        elif self.config.provider == ModelProvider.ANTHROPIC:
            # Would initialize Anthropic provider here
            logger.warning("Anthropic provider not implemented, using mock")
            return MockLLMProvider()
        else:
            return MockLLMProvider()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[CompletionOptions] = None
    ) -> str:
        """
        Generate completion for a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            options: Completion options

        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append(LLMMessage("system", system_prompt))
        messages.append(LLMMessage("user", prompt))

        response = self.complete(messages, options)
        return response.content

    def complete(
        self,
        messages: List[LLMMessage],
        options: Optional[CompletionOptions] = None
    ) -> LLMResponse:
        """
        Generate completion from messages.

        Args:
            messages: Conversation messages
            options: Completion options

        Returns:
            LLM response
        """
        options = options or CompletionOptions()

        # Check cache if enabled
        if self.config.cache_enabled:
            cache_key = self._make_cache_key(messages, options)
            if cache_key in self._cache:
                logger.debug("LLM cache hit")
                return self._cache[cache_key]

        # Make request with retry
        start_time = time.time()
        response = None

        for attempt in range(self.config.max_retries):
            try:
                response = self.provider.complete(messages, options)
                break
            except Exception as e:
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        if response is None:
            raise RuntimeError("Failed to get LLM response")

        # Update metrics
        self.request_count += 1
        self.total_tokens += response.usage.get('total_tokens', 0)
        self.total_latency += response.latency_ms

        # Cache if enabled
        if self.config.cache_enabled:
            self._cache[cache_key] = response

        logger.info(f"LLM completion: {response.usage['total_tokens']} tokens, {response.latency_ms:.0f}ms")

        return response

    def classify(
        self,
        text: str,
        categories: List[str],
        confidence_threshold: float = 0.8
    ) -> Optional[str]:
        """
        Classify text into categories.

        Args:
            text: Text to classify
            categories: List of possible categories
            confidence_threshold: Minimum confidence for classification

        Returns:
            Category or None if below threshold
        """
        prompt = f"""Classify the following text into one of these categories: {', '.join(categories)}

Text: {text}

Return only the category name, nothing else."""

        system_prompt = "You are a classification expert. Return only the category name."

        response = self.generate(prompt, system_prompt, CompletionOptions(temperature=0))

        # Check if response is in categories
        response_clean = response.strip().lower()
        for category in categories:
            if category.lower() == response_clean:
                return category

        logger.warning(f"Classification response '{response}' not in categories")
        return None

    def extract_entities(
        self,
        text: str,
        entity_types: List[str]
    ) -> Dict[str, List[str]]:
        """
        Extract entities from text.

        Args:
            text: Text to analyze
            entity_types: Types of entities to extract

        Returns:
            Dictionary of entity type to list of entities
        """
        prompt = f"""Extract the following types of entities from the text: {', '.join(entity_types)}

Text: {text}

Return as JSON with entity types as keys and lists of entities as values."""

        response = self.generate(
            prompt,
            "You are an entity extraction expert. Return only valid JSON.",
            CompletionOptions(temperature=0)
        )

        try:
            # Parse JSON response
            entities = json.loads(response)
            return entities
        except json.JSONDecodeError:
            logger.error(f"Failed to parse entity extraction response: {response}")
            return {entity_type: [] for entity_type in entity_types}

    def summarize(
        self,
        text: str,
        max_length: int = 200,
        style: str = "concise"
    ) -> str:
        """
        Summarize text.

        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            style: Summary style ('concise', 'detailed', 'bullet')

        Returns:
            Summarized text
        """
        style_instructions = {
            "concise": "Be very concise and direct.",
            "detailed": "Provide a comprehensive summary with key details.",
            "bullet": "Use bullet points for main ideas."
        }

        prompt = f"""Summarize the following text in under {max_length} words.
{style_instructions.get(style, '')}

Text: {text}"""

        return self.generate(
            prompt,
            "You are a summarization expert.",
            CompletionOptions(max_tokens=max_length * 2)
        )

    def match_entity(
        self,
        entity: str,
        candidates: List[str],
        threshold: float = 0.8
    ) -> Optional[str]:
        """
        Match entity to best candidate.

        Args:
            entity: Entity to match
            candidates: List of possible matches
            threshold: Matching threshold

        Returns:
            Best match or None if below threshold
        """
        if not candidates:
            return None

        prompt = f"""Match the entity "{entity}" to the best candidate from this list:
{json.dumps(candidates, indent=2)}

Return only the exact matching candidate string, or "NO_MATCH" if no good match exists."""

        response = self.generate(
            prompt,
            "You are an entity matching expert.",
            CompletionOptions(temperature=0)
        )

        response_clean = response.strip()
        if response_clean == "NO_MATCH":
            return None

        # Verify response is in candidates
        if response_clean in candidates:
            return response_clean

        # Try fuzzy match
        for candidate in candidates:
            if candidate.lower() == response_clean.lower():
                return candidate

        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        avg_tokens = self.total_tokens / self.request_count if self.request_count > 0 else 0

        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "average_tokens": avg_tokens,
            "average_latency_ms": avg_latency,
            "cache_size": len(self._cache),
            "provider": self.config.provider.value,
            "model": self.config.model
        }

    def clear_cache(self) -> None:
        """Clear response cache."""
        self._cache.clear()
        logger.info("LLM cache cleared")

    def _make_cache_key(
        self,
        messages: List[LLMMessage],
        options: CompletionOptions
    ) -> str:
        """Generate cache key for request."""
        key_data = {
            "messages": [(m.role, m.content) for m in messages],
            "options": {
                "temperature": options.temperature,
                "max_tokens": options.max_tokens,
                "top_p": options.top_p
            }
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return str(hash(key_str))