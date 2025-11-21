# -*- coding: utf-8 -*-
"""
AgentIntelligence - Multi-provider LLM integration and prompt management.

This module provides the intelligence layer for GreenLang agents, supporting
multiple LLM providers (Anthropic, OpenAI, Google, Meta) with unified interfaces,
prompt template management, context window optimization, and token tracking.

Example:
    >>> from agent_intelligence import LLMProvider, PromptTemplate
    >>> provider = LLMProvider.create("anthropic", api_key="...")
    >>> response = await provider.generate(prompt="Analyze ESG data", temperature=0.7)
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from pydantic import BaseModel, Field, validator
import tiktoken
import jinja2
from greenlang.determinism import FinancialDecimal

# Configure module logger
logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    META = "meta"
    AZURE_OPENAI = "azure_openai"
    BEDROCK = "bedrock"
    LOCAL = "local"


class ModelCapability(str, Enum):
    """Model capabilities."""

    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    STRUCTURED_OUTPUT = "structured_output"


@dataclass
class ModelConfig:
    """Configuration for specific models."""

    provider: ModelProvider
    model_id: str
    display_name: str
    context_window: int
    max_output_tokens: int
    capabilities: List[ModelCapability]
    cost_per_1k_input_tokens: float  # in USD
    cost_per_1k_output_tokens: float  # in USD
    supports_streaming: bool = True
    supports_json_mode: bool = False
    default_temperature: float = 0.7


# Predefined model configurations
MODEL_REGISTRY = {
    "claude-3-opus": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        context_window=200000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.VISION,
            ModelCapability.STRUCTURED_OUTPUT
        ],
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.075,
        supports_streaming=True,
        default_temperature=0.7
    ),
    "gpt-4-turbo": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4-turbo-preview",
        display_name="GPT-4 Turbo",
        context_window=128000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.STRUCTURED_OUTPUT
        ],
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.03,
        supports_streaming=True,
        supports_json_mode=True,
        default_temperature=0.7
    ),
    "gemini-pro": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-pro",
        display_name="Gemini Pro",
        context_window=32000,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING
        ],
        cost_per_1k_input_tokens=0.00125,
        cost_per_1k_output_tokens=0.00375,
        supports_streaming=True,
        default_temperature=0.7
    )
}


class TokenUsage(BaseModel):
    """Token usage tracking."""

    input_tokens: int = Field(0, ge=0, description="Number of input tokens")
    output_tokens: int = Field(0, ge=0, description="Number of output tokens")
    total_tokens: int = Field(0, ge=0, description="Total tokens used")
    input_cost_usd: float = Field(0.0, ge=0.0, description="Cost of input tokens")
    output_cost_usd: float = Field(0.0, ge=0.0, description="Cost of output tokens")
    total_cost_usd: float = Field(0.0, ge=0.0, description="Total cost in USD")

    def add(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add another TokenUsage to this one."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            input_cost_usd=self.input_cost_usd + other.input_cost_usd,
            output_cost_usd=self.output_cost_usd + other.output_cost_usd,
            total_cost_usd=self.total_cost_usd + other.total_cost_usd
        )


class GenerationRequest(BaseModel):
    """Request for text generation."""

    prompt: str = Field(..., description="Input prompt or messages")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum output tokens")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences")
    system_prompt: Optional[str] = Field(None, description="System prompt for chat models")
    json_mode: bool = Field(False, description="Enable JSON mode if supported")
    stream: bool = Field(False, description="Enable streaming response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Request metadata")


class GenerationResponse(BaseModel):
    """Response from text generation."""

    text: str = Field(..., description="Generated text")
    model_id: str = Field(..., description="Model used for generation")
    usage: TokenUsage = Field(..., description="Token usage and costs")
    finish_reason: str = Field(..., description="Reason for completion")
    generation_time_ms: float = Field(..., ge=0.0, description="Generation time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


class PromptTemplate(BaseModel):
    """Template for prompt construction."""

    name: str = Field(..., description="Template name")
    template: str = Field(..., description="Jinja2 template string")
    variables: List[str] = Field(default_factory=list, description="Template variables")
    system_prompt: Optional[str] = Field(None, description="System prompt template")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Template metadata")
    version: str = Field("1.0.0", description="Template version")
    max_tokens: Optional[int] = Field(None, description="Suggested max tokens")
    temperature: Optional[float] = Field(None, description="Suggested temperature")

    @validator('template')
    def validate_template(cls, v):
        """Validate Jinja2 template syntax."""
        try:
            jinja2.Template(v)
        except jinja2.TemplateError as e:
            raise ValueError(f"Invalid template syntax: {str(e)}")
        return v

    def render(self, **kwargs) -> str:
        """Render template with variables."""
        template = jinja2.Template(self.template)
        return template.render(**kwargs)


class ContextWindow:
    """
    Manages context window optimization and chunking.

    Handles splitting long contexts, managing conversation history,
    and optimizing token usage within model limits.
    """

    def __init__(self, max_tokens: int, reserve_output_tokens: int = 1024):
        """
        Initialize context window manager.

        Args:
            max_tokens: Maximum context window size
            reserve_output_tokens: Tokens reserved for output
        """
        self.max_tokens = max_tokens
        self.reserve_output_tokens = reserve_output_tokens
        self.available_tokens = max_tokens - reserve_output_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoding.encode(text))

    def fits_in_context(self, text: str) -> bool:
        """Check if text fits in available context."""
        return self.count_tokens(text) <= self.available_tokens

    def truncate_to_fit(self, text: str, from_end: bool = True) -> str:
        """
        Truncate text to fit in context window.

        Args:
            text: Text to truncate
            from_end: If True, truncate from end; otherwise from beginning

        Returns:
            Truncated text
        """
        tokens = self._encoding.encode(text)

        if len(tokens) <= self.available_tokens:
            return text

        if from_end:
            truncated_tokens = tokens[:self.available_tokens]
        else:
            truncated_tokens = tokens[-self.available_tokens:]

        return self._encoding.decode(truncated_tokens)

    def chunk_text(self, text: str, chunk_overlap: int = 200) -> List[str]:
        """
        Split text into chunks that fit in context window.

        Args:
            text: Text to chunk
            chunk_overlap: Number of tokens to overlap between chunks

        Returns:
            List of text chunks
        """
        tokens = self._encoding.encode(text)
        chunk_size = self.available_tokens - chunk_overlap
        chunks = []

        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + self.available_tokens]
            chunks.append(self._encoding.decode(chunk_tokens))

        return chunks

    def manage_conversation(self, messages: List[Dict[str, str]],
                          preserve_system: bool = True) -> List[Dict[str, str]]:
        """
        Manage conversation history to fit in context window.

        Args:
            messages: Conversation messages
            preserve_system: Always preserve system message

        Returns:
            Messages that fit in context window
        """
        if not messages:
            return messages

        total_tokens = 0
        result = []

        # Preserve system message if present
        if preserve_system and messages[0].get("role") == "system":
            system_msg = messages[0]
            system_tokens = self.count_tokens(system_msg["content"])
            total_tokens += system_tokens
            result.append(system_msg)
            messages = messages[1:]

        # Add messages from most recent, keeping within limit
        for msg in reversed(messages):
            msg_tokens = self.count_tokens(msg["content"])
            if total_tokens + msg_tokens <= self.available_tokens:
                result.insert(len(result) if not preserve_system else 1, msg)
                total_tokens += msg_tokens
            else:
                break

        return result


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_config: ModelConfig, api_key: Optional[str] = None):
        """
        Initialize LLM provider.

        Args:
            model_config: Model configuration
            api_key: API key for provider
        """
        self.model_config = model_config
        self.api_key = api_key
        self.context_window = ContextWindow(model_config.context_window)
        self._usage_history: List[TokenUsage] = []
        self._total_usage = TokenUsage()
        self._logger = logging.getLogger(f"{__name__}.{model_config.provider}")

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text from prompt.

        Args:
            request: Generation request

        Returns:
            Generation response
        """
        pass

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> TokenUsage:
        """
        Calculate token usage and cost.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            TokenUsage with costs
        """
        input_cost = (input_tokens / 1000) * self.model_config.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.model_config.cost_per_1k_output_tokens

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=input_cost + output_cost
        )

        # Track usage
        self._usage_history.append(usage)
        self._total_usage = self._total_usage.add(usage)

        return usage

    def get_total_usage(self) -> TokenUsage:
        """Get total token usage and costs."""
        return self._total_usage

    def reset_usage(self) -> None:
        """Reset usage tracking."""
        self._usage_history = []
        self._total_usage = TokenUsage()


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Claude."""
        start_time = time.time()

        try:
            # Simulate API call (replace with actual implementation)
            # import anthropic
            # client = anthropic.AsyncAnthropic(api_key=self.api_key)
            # response = await client.messages.create(...)

            # For now, return mock response
            generated_text = f"Mock response for prompt: {request.prompt[:100]}..."
            input_tokens = self.context_window.count_tokens(request.prompt)
            output_tokens = self.context_window.count_tokens(generated_text)

            usage = self.calculate_cost(input_tokens, output_tokens)

            return GenerationResponse(
                text=generated_text,
                model_id=self.model_config.model_id,
                usage=usage,
                finish_reason="stop",
                generation_time_ms=(time.time() - start_time) * 1000,
                metadata={"provider": "anthropic"}
            )

        except Exception as e:
            self._logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings (not supported by Anthropic)."""
        raise NotImplementedError("Anthropic does not support embeddings")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation."""

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using GPT."""
        start_time = time.time()

        try:
            # Simulate API call (replace with actual implementation)
            # import openai
            # client = openai.AsyncOpenAI(api_key=self.api_key)
            # response = await client.chat.completions.create(...)

            # For now, return mock response
            generated_text = f"Mock GPT response for: {request.prompt[:100]}..."
            input_tokens = self.context_window.count_tokens(request.prompt)
            output_tokens = self.context_window.count_tokens(generated_text)

            usage = self.calculate_cost(input_tokens, output_tokens)

            return GenerationResponse(
                text=generated_text,
                model_id=self.model_config.model_id,
                usage=usage,
                finish_reason="stop",
                generation_time_ms=(time.time() - start_time) * 1000,
                metadata={"provider": "openai"}
            )

        except Exception as e:
            self._logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        try:
            # Simulate API call
            # import openai
            # client = openai.AsyncOpenAI(api_key=self.api_key)
            # response = await client.embeddings.create(...)

            # For now, return mock embeddings
            embeddings = [[0.1] * 1536 for _ in texts]  # Mock 1536-dim embeddings
            return embeddings

        except Exception as e:
            self._logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
            raise


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _providers = {
        ModelProvider.ANTHROPIC: AnthropicProvider,
        ModelProvider.OPENAI: OpenAIProvider,
        # Add more providers as implemented
    }

    @classmethod
    def create(cls, provider: Union[str, ModelProvider], model: str,
               api_key: Optional[str] = None) -> BaseLLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider: Provider name or enum
            model: Model identifier
            api_key: API key for provider

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider or model not supported
        """
        if isinstance(provider, str):
            provider = ModelProvider(provider)

        if provider not in cls._providers:
            raise ValueError(f"Provider {provider} not supported")

        if model not in MODEL_REGISTRY:
            raise ValueError(f"Model {model} not found in registry")

        model_config = MODEL_REGISTRY[model]
        provider_class = cls._providers[provider]

        return provider_class(model_config, api_key)


class PromptManager:
    """
    Manages prompt templates and optimization.

    Handles template storage, versioning, and optimization for
    different models and use cases.
    """

    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize prompt manager.

        Args:
            template_dir: Directory containing prompt templates
        """
        self.template_dir = template_dir
        self._templates: Dict[str, PromptTemplate] = {}
        self._template_cache: Dict[str, str] = {}

        if template_dir and template_dir.exists():
            self._load_templates()

    def _load_templates(self) -> None:
        """Load templates from directory."""
        for template_file in self.template_dir.glob("*.json"):
            try:
                with open(template_file, 'r') as f:
                    data = json.load(f)
                    template = PromptTemplate(**data)
                    self._templates[template.name] = template
                    logger.debug(f"Loaded template: {template.name}")
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {str(e)}")

    def register_template(self, template: PromptTemplate) -> None:
        """Register a prompt template."""
        self._templates[template.name] = template
        logger.debug(f"Registered template: {template.name}")

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        return self._templates.get(name)

    def render_template(self, name: str, **variables) -> str:
        """
        Render template with variables.

        Args:
            name: Template name
            **variables: Template variables

        Returns:
            Rendered prompt

        Raises:
            ValueError: If template not found
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template {name} not found")

        # Check cache
        cache_key = f"{name}:{hashlib.md5(str(variables).encode()).hexdigest()}"
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        # Render template
        rendered = template.render(**variables)

        # Cache rendered template
        self._template_cache[cache_key] = rendered

        return rendered

    def optimize_for_model(self, prompt: str, model_config: ModelConfig) -> str:
        """
        Optimize prompt for specific model.

        Args:
            prompt: Original prompt
            model_config: Target model configuration

        Returns:
            Optimized prompt
        """
        # Add model-specific optimizations
        optimized = prompt

        # Add format hints for models supporting JSON mode
        if model_config.supports_json_mode and "json" in prompt.lower():
            optimized = f"{optimized}\n\nRespond with valid JSON only."

        # Add capability hints
        if ModelCapability.FUNCTION_CALLING in model_config.capabilities:
            # Add function calling hints if relevant
            pass

        return optimized

    def create_few_shot_prompt(self, task: str, examples: List[Tuple[str, str]],
                               query: str) -> str:
        """
        Create few-shot learning prompt.

        Args:
            task: Task description
            examples: List of (input, output) examples
            query: Query to process

        Returns:
            Few-shot prompt
        """
        prompt_parts = [f"Task: {task}\n\nExamples:"]

        for i, (input_ex, output_ex) in enumerate(examples, 1):
            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(f"Input: {input_ex}")
            prompt_parts.append(f"Output: {output_ex}")

        prompt_parts.append(f"\nNow process the following:")
        prompt_parts.append(f"Input: {query}")
        prompt_parts.append("Output:")

        return "\n".join(prompt_parts)

    def create_chain_of_thought_prompt(self, task: str, query: str) -> str:
        """
        Create chain-of-thought reasoning prompt.

        Args:
            task: Task description
            query: Query to process

        Returns:
            Chain-of-thought prompt
        """
        return f"""Task: {task}

Query: {query}

Let's think through this step by step:

1. First, I need to understand what is being asked...
2. Next, I should identify the key information...
3. Then, I'll analyze the requirements...
4. Finally, I'll formulate my response...

Step-by-step reasoning:
"""


class IntelligenceOrchestrator:
    """
    Orchestrates multiple LLM providers for optimal performance.

    Handles provider selection, fallback strategies, load balancing,
    and cost optimization across multiple providers.
    """

    def __init__(self):
        """Initialize intelligence orchestrator."""
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._primary_provider: Optional[str] = None
        self._fallback_providers: List[str] = []
        self._usage_limits: Dict[str, float] = {}  # Provider -> max cost USD
        self._logger = logging.getLogger(f"{__name__}.Orchestrator")

    def register_provider(self, name: str, provider: BaseLLMProvider,
                         is_primary: bool = False, max_cost_usd: Optional[float] = None) -> None:
        """
        Register an LLM provider.

        Args:
            name: Provider name
            provider: Provider instance
            is_primary: Set as primary provider
            max_cost_usd: Maximum cost limit for this provider
        """
        self._providers[name] = provider

        if is_primary:
            self._primary_provider = name
        else:
            self._fallback_providers.append(name)

        if max_cost_usd:
            self._usage_limits[name] = max_cost_usd

        self._logger.info(f"Registered provider: {name} (primary={is_primary})")

    async def generate(self, request: GenerationRequest,
                      preferred_provider: Optional[str] = None) -> GenerationResponse:
        """
        Generate text using optimal provider.

        Args:
            request: Generation request
            preferred_provider: Preferred provider to use

        Returns:
            Generation response

        Raises:
            RuntimeError: If no providers available or all fail
        """
        # Select provider
        provider_name = preferred_provider or self._primary_provider
        if not provider_name or provider_name not in self._providers:
            provider_name = self._primary_provider

        if not provider_name:
            raise RuntimeError("No providers registered")

        # Check cost limit
        provider = self._providers[provider_name]
        if provider_name in self._usage_limits:
            current_cost = provider.get_total_usage().total_cost_usd
            if current_cost >= self._usage_limits[provider_name]:
                self._logger.warning(f"Provider {provider_name} exceeded cost limit")
                # Try fallback
                for fallback in self._fallback_providers:
                    if fallback != provider_name:
                        provider_name = fallback
                        provider = self._providers[provider_name]
                        break

        # Try generation with fallback
        last_error = None
        providers_to_try = [provider_name] + [p for p in self._fallback_providers if p != provider_name]

        for pname in providers_to_try:
            try:
                provider = self._providers[pname]
                self._logger.debug(f"Attempting generation with {pname}")
                response = await provider.generate(request)
                return response

            except Exception as e:
                self._logger.error(f"Provider {pname} failed: {str(e)}")
                last_error = e
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    def get_usage_summary(self) -> Dict[str, TokenUsage]:
        """Get usage summary for all providers."""
        return {
            name: provider.get_total_usage()
            for name, provider in self._providers.items()
        }

    def optimize_for_cost(self, request: GenerationRequest) -> str:
        """
        Select most cost-effective provider for request.

        Args:
            request: Generation request

        Returns:
            Optimal provider name
        """
        # Estimate tokens
        estimated_input = len(request.prompt.split()) * 1.3  # Rough estimate
        estimated_output = request.max_tokens or 1000

        min_cost = FinancialDecimal.from_string('inf')
        optimal_provider = self._primary_provider

        for name, provider in self._providers.items():
            # Skip if over limit
            if name in self._usage_limits:
                if provider.get_total_usage().total_cost_usd >= self._usage_limits[name]:
                    continue

            # Calculate estimated cost
            config = provider.model_config
            cost = (estimated_input / 1000 * config.cost_per_1k_input_tokens +
                   estimated_output / 1000 * config.cost_per_1k_output_tokens)

            if cost < min_cost:
                min_cost = cost
                optimal_provider = name

        return optimal_provider


# Example usage
if __name__ == "__main__":
    async def main():
        """Test the intelligence layer."""

        # Create providers
        anthropic_provider = LLMProviderFactory.create(
            provider="anthropic",
            model="claude-3-opus",
            api_key="test_key"
        )

        openai_provider = LLMProviderFactory.create(
            provider="openai",
            model="gpt-4-turbo",
            api_key="test_key"
        )

        # Create orchestrator
        orchestrator = IntelligenceOrchestrator()
        orchestrator.register_provider("claude", anthropic_provider, is_primary=True, max_cost_usd=10.0)
        orchestrator.register_provider("gpt4", openai_provider, is_primary=False, max_cost_usd=5.0)

        # Generate text
        request = GenerationRequest(
            prompt="Analyze the ESG impact of renewable energy investments",
            temperature=0.7,
            max_tokens=500
        )

        response = await orchestrator.generate(request)
        print(f"Generated: {response.text[:200]}...")
        print(f"Cost: ${response.usage.total_cost_usd:.4f}")

        # Test prompt manager
        prompt_manager = PromptManager()

        template = PromptTemplate(
            name="esg_analysis",
            template="Analyze the {{ category }} impact of {{ company }} focusing on {{ metrics | join(', ') }}",
            variables=["category", "company", "metrics"]
        )

        prompt_manager.register_template(template)

        rendered = prompt_manager.render_template(
            "esg_analysis",
            category="environmental",
            company="TechCorp",
            metrics=["carbon emissions", "water usage", "waste management"]
        )

        print(f"\nRendered prompt: {rendered}")

        # Test context window
        context = ContextWindow(max_tokens=4096, reserve_output_tokens=1024)
        long_text = "This is a test. " * 1000

        if not context.fits_in_context(long_text):
            truncated = context.truncate_to_fit(long_text)
            print(f"\nTruncated text from {len(long_text)} to {len(truncated)} chars")

        # Get usage summary
        summary = orchestrator.get_usage_summary()
        for provider, usage in summary.items():
            print(f"\n{provider} usage: {usage.total_tokens} tokens, ${usage.total_cost_usd:.4f}")

    # Run the example
    asyncio.run(main())