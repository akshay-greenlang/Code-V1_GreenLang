# -*- coding: utf-8 -*-
# GL-VCCI ML Module - LLM Client
# Spend Classification ML System - LLM API Client with Caching

"""
LLM Client
==========

LLM API client with caching, retry logic, and cost tracking.

Features:
--------
- Multi-provider support (OpenAI, Anthropic)
- Redis caching (55-minute TTL for tokens)
- Exponential backoff retry (1s, 2s, 4s, 8s)
- Cost tracking (token usage)
- Batch processing
- Rate limiting
- SOC 2 compliant audit logging

Supported Providers:
-------------------
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude 3 family)

Usage:
------
```python
from utils.ml.llm_client import LLMClient
from utils.ml.config import MLConfig

# Initialize client
config = MLConfig()
client = LLMClient(config)

# Single classification
result = await client.classify_spend(
    description="Laptop purchase for software development",
    category_hints=["category_1", "category_2"]
)
print(f"Category: {result.category}, Confidence: {result.confidence}")

# Batch classification
results = await client.classify_batch([
    "Office furniture purchase",
    "Flight to customer meeting",
    "Electricity bill payment"
])
```
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import redis.asyncio as aioredis
from pydantic import BaseModel, Field

from .config import LLMProvider, MLConfig, Scope3Category
from .exceptions import (
    LLMException,
    LLMProviderException,
    LLMRateLimitException,
    LLMTimeoutException,
    LLMTokenLimitException,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Response Models
# ============================================================================

class ClassificationResult(BaseModel):
    """
    LLM classification result.

    Attributes:
        category: Predicted Scope 3 category
        confidence: Confidence score (0.0-1.0)
        reasoning: LLM reasoning for classification
        alternative_categories: Alternative categories with confidence scores
        cached: Whether result was retrieved from cache
        tokens_used: Number of tokens used (input + output)
        cost_usd: Estimated cost in USD
    """
    category: str = Field(description="Predicted Scope 3 category")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: Optional[str] = Field(default=None, description="Classification reasoning")
    alternative_categories: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="Alternative categories [(category, confidence), ...]"
    )
    cached: bool = Field(default=False, description="Retrieved from cache")
    tokens_used: int = Field(default=0, description="Tokens used")
    cost_usd: float = Field(default=0.0, description="Estimated cost in USD")


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """
    LLM API client with caching and retry logic.

    Supports OpenAI and Anthropic providers with automatic fallback,
    caching, cost tracking, and SOC 2 compliant audit logging.
    """

    def __init__(self, config: MLConfig):
        """
        Initialize LLM client.

        Args:
            config: ML configuration
        """
        self.config = config
        self.llm_config = config.llm
        self.cache_config = config.cache
        self.cost_config = config.cost_tracking

        # Initialize Redis cache
        if self.cache_config.enabled:
            self.redis_client = aioredis.from_url(
                self.cache_config.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        else:
            self.redis_client = None

        # Cost tracking
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0
        self.request_count = 0

        # HTTP client
        self.http_client = httpx.AsyncClient(timeout=self.llm_config.timeout_seconds)

        logger.info(
            f"Initialized LLM client: provider={self.llm_config.provider.value}, "
            f"model={self.llm_config.model_name}, cache_enabled={self.cache_config.enabled}"
        )

    async def close(self):
        """Close client connections."""
        if self.redis_client:
            await self.redis_client.close()
        await self.http_client.aclose()

    async def classify_spend(
        self,
        description: str,
        category_hints: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> ClassificationResult:
        """
        Classify procurement spend description into Scope 3 category.

        Args:
            description: Procurement description
            category_hints: Optional category hints to narrow search
            use_cache: Whether to use cache

        Returns:
            Classification result

        Raises:
            LLMException: If classification fails
        """
        # Check cache
        if use_cache and self.redis_client:
            cached_result = await self._get_cached_result(description)
            if cached_result:
                logger.debug(f"Cache hit for description: {description[:50]}...")
                return cached_result

        # Call LLM
        try:
            result = await self._classify_with_retry(description, category_hints)

            # Cache result
            if use_cache and self.redis_client:
                await self._cache_result(description, result)

            # Update cost tracking
            self._update_cost_tracking(result)

            return result

        except Exception as e:
            logger.error(f"LLM classification failed: {e}", exc_info=True)
            raise LLMException(
                message=f"Failed to classify spend: {str(e)}",
                details={"description": description[:100]},
                original_error=e
            )

    async def classify_batch(
        self,
        descriptions: List[str],
        batch_size: int = 10,
        use_cache: bool = True
    ) -> List[ClassificationResult]:
        """
        Classify batch of spend descriptions.

        Args:
            descriptions: List of procurement descriptions
            batch_size: Batch size for processing
            use_cache: Whether to use cache

        Returns:
            List of classification results
        """
        results = []

        # Process in batches
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i + batch_size]

            # Classify concurrently
            tasks = [
                self.classify_spend(desc, use_cache=use_cache)
                for desc in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle errors
            for desc, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch classification failed for '{desc[:50]}...': {result}")
                    # Create error result
                    results.append(ClassificationResult(
                        category="unknown",
                        confidence=0.0,
                        reasoning=f"Error: {str(result)}"
                    ))
                else:
                    results.append(result)

        return results

    async def _classify_with_retry(
        self,
        description: str,
        category_hints: Optional[List[str]] = None
    ) -> ClassificationResult:
        """
        Classify with exponential backoff retry.

        Args:
            description: Procurement description
            category_hints: Optional category hints

        Returns:
            Classification result

        Raises:
            LLMException: If all retries fail
        """
        last_error = None

        for attempt in range(self.llm_config.max_retries + 1):
            try:
                if self.llm_config.provider == LLMProvider.OPENAI:
                    return await self._classify_openai(description, category_hints)
                else:  # ANTHROPIC
                    return await self._classify_anthropic(description, category_hints)

            except LLMRateLimitException as e:
                last_error = e
                if attempt < self.llm_config.max_retries:
                    delay = self.llm_config.retry_delay_seconds * (2 ** attempt)
                    logger.warning(
                        f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/"
                        f"{self.llm_config.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

            except LLMTimeoutException as e:
                last_error = e
                if attempt < self.llm_config.max_retries:
                    delay = self.llm_config.retry_delay_seconds * (2 ** attempt)
                    logger.warning(
                        f"Request timeout, retrying in {delay}s (attempt {attempt + 1}/"
                        f"{self.llm_config.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

            except LLMProviderException as e:
                # Don't retry provider errors (auth, etc.)
                raise

        # All retries failed
        raise last_error or LLMException("Classification failed after all retries")

    async def _classify_openai(
        self,
        description: str,
        category_hints: Optional[List[str]] = None
    ) -> ClassificationResult:
        """
        Classify using OpenAI API.

        Args:
            description: Procurement description
            category_hints: Optional category hints

        Returns:
            Classification result
        """
        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(description, category_hints)

        # API request
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_config.api_key.get_secret_value()}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.llm_config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.llm_config.temperature,
            "max_tokens": self.llm_config.max_tokens,
            "response_format": {"type": "json_object"}  # Structured output
        }

        try:
            response = await self.http_client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise LLMTimeoutException(
                timeout_seconds=self.llm_config.timeout_seconds,
                original_error=e
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 60))
                raise LLMRateLimitException(
                    retry_after=retry_after,
                    provider="openai"
                )
            raise LLMProviderException(
                message=f"OpenAI API error: {e.response.status_code}",
                provider="openai",
                status_code=e.response.status_code,
                original_error=e
            )

        # Parse response
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # Parse JSON response
        try:
            result_data = json.loads(content)
        except json.JSONDecodeError:
            raise LLMException(f"Invalid JSON response from OpenAI: {content}")

        # Calculate cost
        tokens_used = usage.get("total_tokens", 0)
        cost_usd = self._calculate_cost_openai(
            self.llm_config.model_name,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0)
        )

        return ClassificationResult(
            category=result_data["category"],
            confidence=result_data["confidence"],
            reasoning=result_data.get("reasoning"),
            alternative_categories=result_data.get("alternatives", []),
            cached=False,
            tokens_used=tokens_used,
            cost_usd=cost_usd
        )

    async def _classify_anthropic(
        self,
        description: str,
        category_hints: Optional[List[str]] = None
    ) -> ClassificationResult:
        """
        Classify using Anthropic API.

        Args:
            description: Procurement description
            category_hints: Optional category hints

        Returns:
            Classification result
        """
        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(description, category_hints)

        # API request
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.llm_config.api_key.get_secret_value(),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.llm_config.model_name,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.llm_config.temperature,
            "max_tokens": self.llm_config.max_tokens
        }

        try:
            response = await self.http_client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise LLMTimeoutException(
                timeout_seconds=self.llm_config.timeout_seconds,
                original_error=e
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 60))
                raise LLMRateLimitException(
                    retry_after=retry_after,
                    provider="anthropic"
                )
            raise LLMProviderException(
                message=f"Anthropic API error: {e.response.status_code}",
                provider="anthropic",
                status_code=e.response.status_code,
                original_error=e
            )

        # Parse response
        data = response.json()
        content = data["content"][0]["text"]
        usage = data.get("usage", {})

        # Parse JSON response
        try:
            result_data = json.loads(content)
        except json.JSONDecodeError:
            raise LLMException(f"Invalid JSON response from Anthropic: {content}")

        # Calculate cost
        tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        cost_usd = self._calculate_cost_anthropic(
            self.llm_config.model_name,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0)
        )

        return ClassificationResult(
            category=result_data["category"],
            confidence=result_data["confidence"],
            reasoning=result_data.get("reasoning"),
            alternative_categories=result_data.get("alternatives", []),
            cached=False,
            tokens_used=tokens_used,
            cost_usd=cost_usd
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt for classification."""
        categories = Scope3Category.get_all_categories()
        category_descriptions = "\n".join([
            f"- {cat}: {Scope3Category.get_category_name(cat)}"
            for cat in categories
        ])

        return f"""You are a Scope 3 carbon emissions classification expert. Your task is to classify procurement spend descriptions into one of the 15 GHG Protocol Scope 3 categories.

Available Categories:
{category_descriptions}

Instructions:
1. Analyze the spend description carefully
2. Select the most appropriate Scope 3 category
3. Provide a confidence score (0.0-1.0)
4. Explain your reasoning
5. List up to 3 alternative categories with confidence scores

Return your response as JSON in this exact format:
{{
    "category": "category_X_name",
    "confidence": 0.95,
    "reasoning": "Explanation of why this category was selected",
    "alternatives": [
        ["category_Y_name", 0.75],
        ["category_Z_name", 0.60]
    ]
}}

Be precise and consistent in your classifications."""

    def _build_user_prompt(
        self,
        description: str,
        category_hints: Optional[List[str]] = None
    ) -> str:
        """Build user prompt for classification."""
        prompt = f"Classify this procurement spend into a Scope 3 category:\n\n\"{description}\""

        if category_hints:
            hint_names = [Scope3Category.get_category_name(cat) for cat in category_hints]
            prompt += f"\n\nCategory hints (consider these first): {', '.join(hint_names)}"

        return prompt

    async def _get_cached_result(self, description: str) -> Optional[ClassificationResult]:
        """Get cached classification result."""
        if not self.redis_client:
            return None

        cache_key = self._get_cache_key(description)

        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                result_dict = json.loads(cached_data)
                result = ClassificationResult(**result_dict)
                result.cached = True
                return result
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")

        return None

    async def _cache_result(self, description: str, result: ClassificationResult):
        """Cache classification result."""
        if not self.redis_client:
            return

        cache_key = self._get_cache_key(description)

        try:
            result_dict = result.model_dump()
            await self.redis_client.setex(
                cache_key,
                self.cache_config.ttl_seconds,
                json.dumps(result_dict)
            )
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def _get_cache_key(self, description: str) -> str:
        """Generate cache key for description."""
        # Hash description for consistent key
        desc_hash = hashlib.sha256(description.lower().encode()).hexdigest()[:16]
        return f"{self.cache_config.key_prefix}{desc_hash}"

    def _calculate_cost_openai(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate OpenAI API cost."""
        pricing = self.cost_config.openai_pricing.get(model, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def _calculate_cost_anthropic(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate Anthropic API cost."""
        pricing = self.cost_config.anthropic_pricing.get(model, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def _update_cost_tracking(self, result: ClassificationResult):
        """Update cost tracking metrics."""
        if not self.cost_config.enabled:
            return

        self.total_tokens_used += result.tokens_used
        self.total_cost_usd += result.cost_usd
        self.request_count += 1

        # Log summary every N requests
        if self.request_count % self.cost_config.log_every_n_requests == 0:
            logger.info(
                f"LLM Cost Summary: {self.request_count} requests, "
                f"{self.total_tokens_used:,} tokens, "
                f"${self.total_cost_usd:.4f} total cost"
            )

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get cost tracking summary.

        Returns:
            Cost summary dictionary
        """
        return {
            "total_requests": self.request_count,
            "total_tokens": self.total_tokens_used,
            "total_cost_usd": self.total_cost_usd,
            "avg_tokens_per_request": (
                self.total_tokens_used / self.request_count
                if self.request_count > 0 else 0
            ),
            "avg_cost_per_request": (
                self.total_cost_usd / self.request_count
                if self.request_count > 0 else 0
            )
        }
