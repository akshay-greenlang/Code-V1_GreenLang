# -*- coding: utf-8 -*-
"""
BaseLLMProvider - Abstract base class for all LLM providers.

This module defines the interface that all LLM providers must implement,
ensuring consistency across different provider integrations.

Example:
    >>> class CustomProvider(BaseLLMProvider):
    ...     async def generate(self, request):
    ...         # Implementation
    ...         pass
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from pydantic import BaseModel, Field
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    def __init__(self, message: str, provider: str, retryable: bool = True,
                 original_error: Optional[Exception] = None):
        """
        Initialize provider error.

        Args:
            message: Error message
            provider: Provider name that raised the error
            retryable: Whether this error is retryable
            original_error: Original exception if wrapped
        """
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.original_error = original_error
        self.timestamp = DeterministicClock.utcnow()


class RateLimitError(ProviderError):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, provider: str, retry_after: Optional[float] = None):
        """
        Initialize rate limit error.

        Args:
            message: Error message
            provider: Provider name
            retry_after: Seconds to wait before retrying
        """
        super().__init__(message, provider, retryable=True)
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Exception raised for authentication failures."""

    def __init__(self, message: str, provider: str):
        """
        Initialize authentication error.

        Args:
            message: Error message
            provider: Provider name
        """
        super().__init__(message, provider, retryable=False)


class InvalidRequestError(ProviderError):
    """Exception raised for invalid requests."""

    def __init__(self, message: str, provider: str):
        """
        Initialize invalid request error.

        Args:
            message: Error message
            provider: Provider name
        """
        super().__init__(message, provider, retryable=False)


class ServiceUnavailableError(ProviderError):
    """Exception raised when service is unavailable."""

    def __init__(self, message: str, provider: str):
        """
        Initialize service unavailable error.

        Args:
            message: Error message
            provider: Provider name
        """
        super().__init__(message, provider, retryable=True)


class TokenUsage(BaseModel):
    """Token usage and cost tracking."""

    input_tokens: int = Field(0, ge=0, description="Number of input tokens")
    output_tokens: int = Field(0, ge=0, description="Number of output tokens")
    total_tokens: int = Field(0, ge=0, description="Total tokens used")
    input_cost_usd: float = Field(0.0, ge=0.0, description="Cost of input tokens")
    output_cost_usd: float = Field(0.0, ge=0.0, description="Cost of output tokens")
    total_cost_usd: float = Field(0.0, ge=0.0, description="Total cost in USD")
    cached_tokens: int = Field(0, ge=0, description="Number of cached tokens (if supported)")

    def add(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add another TokenUsage to this one."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            input_cost_usd=self.input_cost_usd + other.input_cost_usd,
            output_cost_usd=self.output_cost_usd + other.output_cost_usd,
            total_cost_usd=self.total_cost_usd + other.total_cost_usd,
            cached_tokens=self.cached_tokens + other.cached_tokens
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
    provider: str = Field(..., description="Provider name")
    usage: TokenUsage = Field(..., description="Token usage and costs")
    finish_reason: str = Field(..., description="Reason for completion")
    generation_time_ms: float = Field(..., ge=0.0, description="Generation time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


class ProviderHealth(BaseModel):
    """Provider health status."""

    is_healthy: bool = Field(..., description="Whether provider is healthy")
    last_check: datetime = Field(..., description="Last health check timestamp")
    consecutive_failures: int = Field(0, ge=0, description="Consecutive failure count")
    last_error: Optional[str] = Field(None, description="Last error message")
    latency_ms: Optional[float] = Field(None, description="Average latency in ms")


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM provider implementations must inherit from this class and
    implement the required abstract methods.

    Attributes:
        provider_name: Name of the provider
        model_id: Model identifier
        api_key: API key for authentication
    """

    def __init__(self, provider_name: str, model_id: str, api_key: str,
                 cost_per_1k_input: float, cost_per_1k_output: float):
        """
        Initialize base LLM provider.

        Args:
            provider_name: Name of the provider
            model_id: Model identifier
            api_key: API key for authentication
            cost_per_1k_input: Cost per 1K input tokens in USD
            cost_per_1k_output: Cost per 1K output tokens in USD
        """
        self.provider_name = provider_name
        self.model_id = model_id
        self.api_key = api_key
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self._logger = logging.getLogger(f"{__name__}.{provider_name}")
        self._total_usage = TokenUsage()
        self._health = ProviderHealth(
            is_healthy=True,
            last_check=DeterministicClock.utcnow(),
            consecutive_failures=0
        )

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text from prompt.

        Args:
            request: Generation request

        Returns:
            Generation response

        Raises:
            ProviderError: If generation fails
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

        Raises:
            ProviderError: If embedding generation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """
        Check provider health.

        Returns:
            Provider health status
        """
        pass

    def calculate_cost(self, input_tokens: int, output_tokens: int,
                      cached_tokens: int = 0) -> TokenUsage:
        """
        Calculate token usage and cost.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached tokens (if supported)

        Returns:
            TokenUsage with costs
        """
        # Cached tokens are typically cheaper (50% discount)
        effective_input_tokens = input_tokens - cached_tokens + (cached_tokens * 0.5)

        input_cost = (effective_input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=input_cost + output_cost,
            cached_tokens=cached_tokens
        )

        # Track cumulative usage
        self._total_usage = self._total_usage.add(usage)

        return usage

    def get_total_usage(self) -> TokenUsage:
        """Get total token usage and costs."""
        return self._total_usage

    def reset_usage(self) -> None:
        """Reset usage tracking."""
        self._total_usage = TokenUsage()

    def get_health(self) -> ProviderHealth:
        """Get current health status."""
        return self._health

    def _update_health(self, success: bool, error: Optional[str] = None,
                      latency_ms: Optional[float] = None) -> None:
        """
        Update provider health status.

        Args:
            success: Whether operation succeeded
            error: Error message if failed
            latency_ms: Operation latency in milliseconds
        """
        if success:
            self._health.is_healthy = True
            self._health.consecutive_failures = 0
            self._health.last_error = None
        else:
            self._health.consecutive_failures += 1
            self._health.last_error = error
            # Mark unhealthy after 3 consecutive failures
            if self._health.consecutive_failures >= 3:
                self._health.is_healthy = False

        self._health.last_check = DeterministicClock.utcnow()
        if latency_ms is not None:
            # Exponential moving average for latency
            if self._health.latency_ms is None:
                self._health.latency_ms = latency_ms
            else:
                self._health.latency_ms = 0.7 * self._health.latency_ms + 0.3 * latency_ms
