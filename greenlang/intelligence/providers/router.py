# -*- coding: utf-8 -*-
"""
Smart Provider Router - Multi-Tier Intelligence Selection

Automatically selects the best available intelligence tier:
- Tier 2 (BYOK): User's API keys (OpenAI, Anthropic) - highest quality
- Tier 1 (Local): Ollama local models - free, private, real LLM
- Tier 0 (Deterministic): Template-based - always works, no dependencies

For open-source developers, this means:
- `pip install greenlang` -> Immediately get Tier 0 intelligence
- Install Ollama -> Automatically upgrade to Tier 1 (real LLM)
- Add API key -> Automatically upgrade to Tier 2 (GPT-4/Claude)

The router handles:
- Automatic tier detection
- Graceful fallback on failures
- Cost tracking across tiers
- Consistent API regardless of backend

Author: GreenLang Intelligence Framework
Date: December 2025
Status: Production Ready - Core Router
"""

from __future__ import annotations
import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Callable

from greenlang.intelligence.providers.base import (
    LLMProvider,
    LLMProviderConfig,
    LLMCapabilities,
)
from greenlang.intelligence.schemas.messages import ChatMessage, Role
from greenlang.intelligence.schemas.tools import ToolDef
from greenlang.intelligence.schemas.responses import (
    ChatResponse,
    Usage,
    FinishReason,
    ProviderInfo,
)
from greenlang.intelligence.runtime.budget import Budget

logger = logging.getLogger(__name__)


class IntelligenceTier(Enum):
    """Intelligence tier levels."""
    TIER_0_DETERMINISTIC = 0  # Always available
    TIER_1_LOCAL = 1          # Ollama/local LLM
    TIER_2_CLOUD = 2          # BYOK (OpenAI, Anthropic)

    @property
    def display_name(self) -> str:
        names = {
            0: "Deterministic (Tier 0)",
            1: "Local LLM (Tier 1)",
            2: "Cloud LLM (Tier 2)",
        }
        return names.get(self.value, "Unknown")


@dataclass
class TierStatus:
    """Status of an intelligence tier."""
    tier: IntelligenceTier
    available: bool
    provider_name: str
    model: str
    reason: str  # Why available/unavailable


@dataclass
class RouterConfig:
    """Configuration for the smart router."""
    # Tier preferences
    prefer_local: bool = False      # Prefer Ollama over cloud even if API keys available
    allow_fallback: bool = True     # Fall back to lower tiers on failure
    min_tier: IntelligenceTier = IntelligenceTier.TIER_0_DETERMINISTIC

    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Timeout for tier detection
    detection_timeout_s: float = 5.0


class SmartProviderRouter(LLMProvider):
    """
    Smart Provider Router - Automatic Tier Selection

    Provides seamless intelligence by automatically selecting the best
    available tier and handling fallbacks gracefully.

    Tier Priority (default):
        1. Tier 2 (BYOK): If OPENAI_API_KEY or ANTHROPIC_API_KEY set
        2. Tier 1 (Local): If Ollama is running with a model
        3. Tier 0 (Deterministic): Always available

    With prefer_local=True:
        1. Tier 1 (Local): If Ollama is running
        2. Tier 2 (BYOK): If API keys set
        3. Tier 0 (Deterministic): Always available

    Usage:
        # Auto-detect best tier
        router = SmartProviderRouter()

        # Use like any other provider
        response = await router.chat(messages, budget=budget)

        # Check which tier was used
        print(f"Used tier: {router.current_tier}")

    The router is the default provider created by create_provider()
    when model="auto".
    """

    def __init__(
        self,
        config: Optional[LLMProviderConfig] = None,
        router_config: Optional[RouterConfig] = None,
    ) -> None:
        """
        Initialize Smart Router.

        Args:
            config: Base provider config
            router_config: Router-specific configuration
        """
        if config is None:
            config = LLMProviderConfig(model="auto", api_key_env="")

        super().__init__(config)

        self.router_config = router_config or RouterConfig()
        self._providers: Dict[IntelligenceTier, LLMProvider] = {}
        self._tier_status: Dict[IntelligenceTier, TierStatus] = {}
        self._current_tier: Optional[IntelligenceTier] = None
        self._initialized = False

        logger.info("Initialized SmartProviderRouter - will auto-detect best tier")

    @property
    def capabilities(self) -> LLMCapabilities:
        """Return capabilities of current tier."""
        if self._current_tier and self._current_tier in self._providers:
            return self._providers[self._current_tier].capabilities

        # Default capabilities (Tier 0)
        return LLMCapabilities(
            function_calling=True,
            json_schema_mode=True,
            max_output_tokens=4096,
            context_window_tokens=8000,
        )

    @property
    def current_tier(self) -> Optional[IntelligenceTier]:
        """Get the currently active tier."""
        return self._current_tier

    @property
    def current_provider(self) -> Optional[LLMProvider]:
        """Get the currently active provider."""
        if self._current_tier:
            return self._providers.get(self._current_tier)
        return None

    async def detect_available_tiers(self) -> Dict[IntelligenceTier, TierStatus]:
        """
        Detect which intelligence tiers are available.

        Returns:
            Dict mapping tiers to their availability status
        """
        status = {}

        # Tier 0: Always available
        status[IntelligenceTier.TIER_0_DETERMINISTIC] = TierStatus(
            tier=IntelligenceTier.TIER_0_DETERMINISTIC,
            available=True,
            provider_name="DeterministicProvider",
            model="greenlang-tier0",
            reason="Built-in, always available"
        )

        # Tier 1: Check Ollama
        try:
            from greenlang.intelligence.providers.ollama import (
                check_ollama_available,
                list_ollama_models,
            )

            ollama_available = await asyncio.wait_for(
                check_ollama_available(self.router_config.ollama_host),
                timeout=self.router_config.detection_timeout_s
            )

            if ollama_available:
                models = await list_ollama_models(self.router_config.ollama_host)
                model = self.router_config.ollama_model

                # Check if preferred model is available
                model_available = any(m.startswith(model.split(":")[0]) for m in models)

                if model_available:
                    status[IntelligenceTier.TIER_1_LOCAL] = TierStatus(
                        tier=IntelligenceTier.TIER_1_LOCAL,
                        available=True,
                        provider_name="OllamaProvider",
                        model=model,
                        reason=f"Ollama running with {len(models)} models"
                    )
                elif models:
                    # Use first available model
                    fallback_model = models[0].split(":")[0]
                    status[IntelligenceTier.TIER_1_LOCAL] = TierStatus(
                        tier=IntelligenceTier.TIER_1_LOCAL,
                        available=True,
                        provider_name="OllamaProvider",
                        model=fallback_model,
                        reason=f"Using available model {fallback_model}"
                    )
                else:
                    status[IntelligenceTier.TIER_1_LOCAL] = TierStatus(
                        tier=IntelligenceTier.TIER_1_LOCAL,
                        available=False,
                        provider_name="OllamaProvider",
                        model="",
                        reason="Ollama running but no models. Run: ollama pull llama3.2"
                    )
            else:
                status[IntelligenceTier.TIER_1_LOCAL] = TierStatus(
                    tier=IntelligenceTier.TIER_1_LOCAL,
                    available=False,
                    provider_name="OllamaProvider",
                    model="",
                    reason="Ollama not running. Install from https://ollama.ai"
                )

        except asyncio.TimeoutError:
            status[IntelligenceTier.TIER_1_LOCAL] = TierStatus(
                tier=IntelligenceTier.TIER_1_LOCAL,
                available=False,
                provider_name="OllamaProvider",
                model="",
                reason="Ollama check timed out"
            )
        except Exception as e:
            status[IntelligenceTier.TIER_1_LOCAL] = TierStatus(
                tier=IntelligenceTier.TIER_1_LOCAL,
                available=False,
                provider_name="OllamaProvider",
                model="",
                reason=f"Ollama check failed: {e}"
            )

        # Tier 2: Check API keys
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

        if openai_key:
            status[IntelligenceTier.TIER_2_CLOUD] = TierStatus(
                tier=IntelligenceTier.TIER_2_CLOUD,
                available=True,
                provider_name="OpenAIProvider",
                model="gpt-4o",
                reason="OPENAI_API_KEY configured"
            )
        elif anthropic_key:
            status[IntelligenceTier.TIER_2_CLOUD] = TierStatus(
                tier=IntelligenceTier.TIER_2_CLOUD,
                available=True,
                provider_name="AnthropicProvider",
                model="claude-3-sonnet-20240229",
                reason="ANTHROPIC_API_KEY configured"
            )
        else:
            status[IntelligenceTier.TIER_2_CLOUD] = TierStatus(
                tier=IntelligenceTier.TIER_2_CLOUD,
                available=False,
                provider_name="",
                model="",
                reason="No API key. Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
            )

        self._tier_status = status
        return status

    def _create_provider(self, tier: IntelligenceTier) -> LLMProvider:
        """Create provider for a specific tier."""
        status = self._tier_status.get(tier)

        if tier == IntelligenceTier.TIER_0_DETERMINISTIC:
            from greenlang.intelligence.providers.deterministic import DeterministicProvider
            config = LLMProviderConfig(model="deterministic", api_key_env="")
            return DeterministicProvider(config)

        elif tier == IntelligenceTier.TIER_1_LOCAL:
            from greenlang.intelligence.providers.ollama import OllamaProvider
            model = status.model if status else self.router_config.ollama_model
            config = LLMProviderConfig(model=f"ollama:{model}", api_key_env="")
            return OllamaProvider(config, host=self.router_config.ollama_host, model=model)

        elif tier == IntelligenceTier.TIER_2_CLOUD:
            if status and "OpenAI" in status.provider_name:
                from greenlang.intelligence.providers.openai import OpenAIProvider
                config = LLMProviderConfig(model="gpt-4o", api_key_env="OPENAI_API_KEY")
                return OpenAIProvider(config)
            else:
                from greenlang.intelligence.providers.anthropic import AnthropicProvider
                config = LLMProviderConfig(
                    model="claude-3-sonnet-20240229",
                    api_key_env="ANTHROPIC_API_KEY"
                )
                return AnthropicProvider(config)

        raise ValueError(f"Unknown tier: {tier}")

    async def initialize(self) -> IntelligenceTier:
        """
        Initialize router and select best tier.

        Returns:
            The selected tier
        """
        if self._initialized:
            return self._current_tier

        # Detect available tiers
        await self.detect_available_tiers()

        # Select best tier based on preferences
        if self.router_config.prefer_local:
            tier_order = [
                IntelligenceTier.TIER_1_LOCAL,
                IntelligenceTier.TIER_2_CLOUD,
                IntelligenceTier.TIER_0_DETERMINISTIC,
            ]
        else:
            tier_order = [
                IntelligenceTier.TIER_2_CLOUD,
                IntelligenceTier.TIER_1_LOCAL,
                IntelligenceTier.TIER_0_DETERMINISTIC,
            ]

        # Find best available tier
        for tier in tier_order:
            if tier.value < self.router_config.min_tier.value:
                continue

            status = self._tier_status.get(tier)
            if status and status.available:
                self._current_tier = tier
                self._providers[tier] = self._create_provider(tier)
                logger.info(
                    f"Selected intelligence tier: {tier.display_name} "
                    f"({status.provider_name}: {status.model})"
                )
                break

        # Always ensure Tier 0 is available for fallback
        if IntelligenceTier.TIER_0_DETERMINISTIC not in self._providers:
            self._providers[IntelligenceTier.TIER_0_DETERMINISTIC] = self._create_provider(
                IntelligenceTier.TIER_0_DETERMINISTIC
            )

        self._initialized = True
        return self._current_tier

    async def chat(
        self,
        messages: List[ChatMessage],
        *,
        tools: Optional[List[ToolDef]] = None,
        json_schema: Optional[Any] = None,
        budget: Budget,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        tool_choice: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ChatResponse:
        """
        Execute chat with automatic tier selection and fallback.

        The router will:
        1. Use the best available tier
        2. Fall back to lower tiers on failure (if enabled)
        3. Always succeed with Tier 0 as last resort
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        # Try tiers with fallback
        tiers_to_try = [self._current_tier]

        if self.router_config.allow_fallback:
            # Add fallback tiers
            for tier in IntelligenceTier:
                if tier not in tiers_to_try and tier.value >= self.router_config.min_tier.value:
                    tiers_to_try.append(tier)

            # Sort by value (lower = more basic = more reliable)
            tiers_to_try.sort(key=lambda t: -t.value if t == self._current_tier else t.value)

        last_error = None

        for tier in tiers_to_try:
            # Ensure provider exists
            if tier not in self._providers:
                status = self._tier_status.get(tier)
                if status and status.available:
                    self._providers[tier] = self._create_provider(tier)
                else:
                    continue

            provider = self._providers[tier]

            try:
                response = await provider.chat(
                    messages=messages,
                    tools=tools,
                    json_schema=json_schema,
                    budget=budget,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    tool_choice=tool_choice,
                    metadata=metadata,
                )

                # Add tier info to response
                if response.provider_info:
                    response.provider_info.extra = response.provider_info.extra or {}
                    response.provider_info.extra["intelligence_tier"] = tier.value
                    response.provider_info.extra["tier_name"] = tier.display_name

                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Tier {tier.display_name} failed: {e}. "
                    f"{'Trying fallback...' if self.router_config.allow_fallback else 'No fallback.'}"
                )
                continue

        # All tiers failed
        raise RuntimeError(
            f"All intelligence tiers failed. Last error: {last_error}"
        )

    def get_tier_report(self) -> str:
        """Generate human-readable tier status report."""
        lines = [
            "GreenLang Intelligence Tier Status",
            "=" * 40,
            ""
        ]

        for tier in IntelligenceTier:
            status = self._tier_status.get(tier)
            if status:
                active = " [ACTIVE]" if tier == self._current_tier else ""
                avail = "Available" if status.available else "Unavailable"
                lines.append(f"{tier.display_name}:{active}")
                lines.append(f"  Status: {avail}")
                if status.available:
                    lines.append(f"  Provider: {status.provider_name}")
                    lines.append(f"  Model: {status.model}")
                lines.append(f"  Info: {status.reason}")
                lines.append("")

        return "\n".join(lines)

    async def close(self):
        """Close all providers."""
        for provider in self._providers.values():
            if hasattr(provider, 'close'):
                await provider.close()


# =============================================================================
# SYNC WRAPPER
# =============================================================================

def detect_tiers_sync() -> Dict[IntelligenceTier, TierStatus]:
    """Synchronously detect available tiers."""
    router = SmartProviderRouter()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(router.detect_available_tiers())


def get_tier_report_sync() -> str:
    """Synchronously generate tier report."""
    router = SmartProviderRouter()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(router.initialize())
    return router.get_tier_report()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def create_smart_router(
    prefer_local: bool = False,
    min_tier: IntelligenceTier = IntelligenceTier.TIER_0_DETERMINISTIC,
) -> SmartProviderRouter:
    """
    Create and initialize a smart router.

    Args:
        prefer_local: Prefer Ollama over cloud APIs
        min_tier: Minimum acceptable tier

    Returns:
        Initialized router
    """
    config = RouterConfig(
        prefer_local=prefer_local,
        min_tier=min_tier,
    )
    router = SmartProviderRouter(router_config=config)
    await router.initialize()
    return router


if __name__ == "__main__":
    """Test smart router."""
    import asyncio

    async def test():
        print("=== Smart Provider Router Test ===\n")

        # Create router
        router = SmartProviderRouter()

        # Initialize and detect tiers
        await router.initialize()

        # Print tier report
        print(router.get_tier_report())

        # Test chat
        messages = [
            ChatMessage(
                role=Role.user,
                content="Explain the carbon footprint of 100 gallons of diesel."
            )
        ]

        print("\nTesting chat...")
        response = await router.chat(messages, budget=Budget(max_usd=1.0))

        print(f"\nResponse (first 300 chars):\n{response.text[:300]}...")
        print(f"\nProvider: {response.provider_info.provider}")
        print(f"Model: {response.provider_info.model}")
        if response.provider_info.extra:
            print(f"Tier: {response.provider_info.extra.get('tier_name', 'unknown')}")
        print(f"Cost: ${response.usage.cost_usd:.6f}")

        await router.close()

    asyncio.run(test())
