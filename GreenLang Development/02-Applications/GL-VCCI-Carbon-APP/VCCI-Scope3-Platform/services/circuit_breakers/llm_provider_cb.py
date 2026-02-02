# -*- coding: utf-8 -*-
"""
Circuit Breaker for LLM Provider API Calls

Protects against failures in LLM services:
- Anthropic Claude API (primary)
- OpenAI GPT-4 API (secondary)

Features:
- Separate circuit breakers for each provider
- Automatic failover between providers
- Token usage tracking
- Rate limit handling
- Prometheus metrics

Author: GreenLang Platform Team
Version: 1.0.0
Date: 2025-11-09
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import time

from greenlang.determinism import DeterministicClock
from greenlang.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    create_circuit_breaker,
)
from greenlang.telemetry import get_logger
from greenlang.cache import get_cache_manager


logger = get_logger(__name__)


# ============================================================================
# LLM PROVIDER CIRCUIT BREAKER
# ============================================================================

class LLMProviderCircuitBreaker:
    """
    Circuit breaker wrapper for LLM provider API calls.

    Manages circuit breakers for multiple LLM providers with automatic
    failover when one provider is unavailable.

    Features:
    - Primary/secondary provider failover
    - Rate limit awareness
    - Token usage tracking
    - Caching for repeated prompts

    Example:
        >>> llm_cb = LLMProviderCircuitBreaker()
        >>> response = llm_cb.generate(
        ...     prompt="Classify this emission...",
        ...     model="claude-3-sonnet",
        ... )
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache = get_cache_manager()

        # Circuit breaker for Anthropic Claude
        self.claude_cb = create_circuit_breaker(
            CircuitBreakerConfig(
                name="llm_provider_claude",
                fail_max=3,  # Lower threshold for expensive API
                timeout_duration=120,  # 2 minutes before retry
                reset_timeout=60,
                fallback_function=self._fallback_claude,
            )
        )

        # Circuit breaker for OpenAI GPT-4
        self.openai_cb = create_circuit_breaker(
            CircuitBreakerConfig(
                name="llm_provider_openai",
                fail_max=3,
                timeout_duration=120,
                reset_timeout=60,
                fallback_function=self._fallback_openai,
            )
        )

        # Track provider preference
        self.primary_provider = "claude"
        self.secondary_provider = "openai"

        self.logger.info("LLM provider circuit breakers initialized")

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        provider: Optional[str] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with circuit breaker protection and failover.

        Args:
            prompt: Text prompt for the LLM
            model: Specific model to use (e.g., "claude-3-sonnet")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            provider: Specific provider to use (claude/openai), or None for auto
            cache_key: Optional cache key for repeated prompts
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response with metadata

        Raises:
            CircuitOpenError: If all providers are unavailable
        """
        # Check cache first if cache_key provided
        if cache_key:
            cached = self._get_cached_response(cache_key)
            if cached:
                self.logger.debug(
                    "Using cached LLM response",
                    extra={"cache_key": cache_key}
                )
                return cached

        # Determine provider
        if provider is None:
            provider = self.primary_provider

        # Try primary provider
        try:
            response = self._call_provider(
                provider=provider,
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            # Cache if cache_key provided
            if cache_key:
                self._cache_response(cache_key, response)

            return response

        except CircuitOpenError as e:
            # Primary failed, try secondary
            if provider == self.primary_provider:
                self.logger.warning(
                    f"Primary LLM provider {provider} circuit open, failing over to {self.secondary_provider}",
                    extra={
                        "primary": provider,
                        "secondary": self.secondary_provider,
                    }
                )

                try:
                    response = self._call_provider(
                        provider=self.secondary_provider,
                        prompt=prompt,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs
                    )

                    if cache_key:
                        self._cache_response(cache_key, response)

                    return response

                except CircuitOpenError:
                    self.logger.error(
                        "All LLM providers unavailable",
                        extra={
                            "primary": self.primary_provider,
                            "secondary": self.secondary_provider,
                        }
                    )
                    raise CircuitOpenError(
                        "All LLM providers are currently unavailable. "
                        "Please try again later."
                    )
            raise

    def _call_provider(
        self,
        provider: str,
        prompt: str,
        model: Optional[str],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Call specific LLM provider with circuit breaker."""
        if provider == "claude":
            return self.claude_cb.call(
                self._call_claude,
                prompt=prompt,
                model=model or "claude-3-sonnet-20240229",
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        elif provider == "openai":
            return self.openai_cb.call(
                self._call_openai,
                prompt=prompt,
                model=model or "gpt-4",
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _call_claude(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call Anthropic Claude API.

        This is a placeholder - actual implementation would use anthropic client.
        """
        self.logger.debug(
            f"Calling Claude API",
            extra={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

        # Placeholder for actual API call
        # In production, this would call:
        # import anthropic
        # client = anthropic.Anthropic()
        # response = client.messages.create(...)

        # Simulate API call
        start_time = time.time()

        # Simulated response
        response = {
            "provider": "claude",
            "model": model,
            "text": f"[Claude response to: {prompt[:50]}...]",
            "tokens_used": 150,
            "latency_ms": int((time.time() - start_time) * 1000),
            "timestamp": DeterministicClock.utcnow().isoformat(),
        }

        return response

    def _call_openai(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call OpenAI GPT API.

        This is a placeholder - actual implementation would use openai client.
        """
        self.logger.debug(
            f"Calling OpenAI API",
            extra={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

        # Placeholder for actual API call
        # In production, this would call:
        # import openai
        # client = openai.OpenAI()
        # response = client.chat.completions.create(...)

        # Simulate API call
        start_time = time.time()

        # Simulated response
        response = {
            "provider": "openai",
            "model": model,
            "text": f"[OpenAI response to: {prompt[:50]}...]",
            "tokens_used": 140,
            "latency_ms": int((time.time() - start_time) * 1000),
            "timestamp": DeterministicClock.utcnow().isoformat(),
        }

        return response

    def _fallback_claude(self, **kwargs) -> Dict[str, Any]:
        """Fallback for Claude API failures - try OpenAI."""
        prompt = kwargs.get("prompt", "")
        self.logger.info(
            "Claude API unavailable, attempting OpenAI fallback",
            extra={"prompt_length": len(prompt)}
        )

        # This will be handled by the main generate() method's failover logic
        # Just raise to trigger failover
        raise CircuitOpenError("Claude circuit is open")

    def _fallback_openai(self, **kwargs) -> Dict[str, Any]:
        """Fallback for OpenAI API failures."""
        self.logger.warning("OpenAI API unavailable")
        raise CircuitOpenError("OpenAI circuit is open")

    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached LLM response if available."""
        full_key = f"llm_response:{cache_key}"
        return self.cache.get(full_key)

    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache LLM response."""
        full_key = f"llm_response:{cache_key}"
        # Cache for 1 hour
        self.cache.set(full_key, response, ttl=3600)

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics for all providers."""
        return {
            "claude": self.claude_cb.get_stats(),
            "openai": self.openai_cb.get_stats(),
            "primary_provider": self.primary_provider,
            "secondary_provider": self.secondary_provider,
        }

    def reset_all(self):
        """Reset all circuit breakers."""
        self.claude_cb.reset()
        self.openai_cb.reset()
        self.logger.info("All LLM provider circuit breakers reset")

    def set_primary_provider(self, provider: str):
        """Set the primary LLM provider."""
        if provider not in ["claude", "openai"]:
            raise ValueError(f"Unsupported provider: {provider}")

        old_primary = self.primary_provider
        self.primary_provider = provider
        self.secondary_provider = "openai" if provider == "claude" else "claude"

        self.logger.info(
            f"Primary LLM provider changed: {old_primary} -> {provider}",
            extra={
                "old_primary": old_primary,
                "new_primary": provider,
                "secondary": self.secondary_provider,
            }
        )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_instance: Optional[LLMProviderCircuitBreaker] = None


def get_llm_provider_cb() -> LLMProviderCircuitBreaker:
    """Get singleton instance of LLM provider circuit breaker."""
    global _instance
    if _instance is None:
        _instance = LLMProviderCircuitBreaker()
    return _instance
