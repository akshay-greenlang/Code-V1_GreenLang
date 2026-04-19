# -*- coding: utf-8 -*-
"""
Smart Provider Factory - Multi-Tier Intelligence System

GreenLang's intelligence works for EVERYONE - from first-time open-source
users to enterprise deployments with API keys.

TIER SYSTEM:
- Tier 2 (BYOK): OpenAI/Anthropic with user's API keys - highest quality
- Tier 1 (Local): Ollama local models - free, private, real LLM
- Tier 0 (Deterministic): Template-based - always works, no dependencies

SMART AUTO-DETECTION:
    provider = create_provider()  # Auto-selects best available tier!

    # Result depends on your setup:
    # - Has OPENAI_API_KEY? → Uses GPT-4o (Tier 2)
    # - Has Ollama running? → Uses Llama3.2 locally (Tier 1)
    # - Neither? → Uses DeterministicProvider (Tier 0) - REAL VALUE, not demo!

This enables TRUE zero-config setup with REAL intelligence from day one!

Usage:
    from greenlang.intelligence import create_provider

    # Auto-detect best tier (recommended)
    provider = create_provider()

    # Force specific tier
    provider = create_provider(model="deterministic")  # Tier 0
    provider = create_provider(model="ollama:llama3.2")  # Tier 1
    provider = create_provider(model="gpt-4o")  # Tier 2 (needs API key)

    # Prefer local over cloud
    provider = create_provider(model="auto", prefer_local=True)
"""

from __future__ import annotations
import os
import logging
from typing import Optional

from greenlang.agents.intelligence.providers.base import LLMProvider, LLMProviderConfig
from greenlang.agents.intelligence.config import (
    IntelligenceConfig,
    get_config_from_env,
    resolve_model_alias,
)

logger = logging.getLogger(__name__)


def has_openai_key() -> bool:
    """
    Check if OpenAI API key is available

    Checks OPENAI_API_KEY environment variable.

    Returns:
        True if API key is set and non-empty
    """
    key = os.getenv("OPENAI_API_KEY")
    return key is not None and len(key.strip()) > 0


def has_anthropic_key() -> bool:
    """
    Check if Anthropic API key is available

    Checks ANTHROPIC_API_KEY environment variable.

    Returns:
        True if API key is set and non-empty
    """
    key = os.getenv("ANTHROPIC_API_KEY")
    return key is not None and len(key.strip()) > 0


def has_any_api_key() -> bool:
    """
    Check if any supported API key is available

    Returns:
        True if at least one API key is configured
    """
    return has_openai_key() or has_anthropic_key()


def detect_best_provider() -> str:
    """
    Auto-detect best available provider

    Priority:
    1. OpenAI (if OPENAI_API_KEY set) - most widely available
    2. Anthropic (if ANTHROPIC_API_KEY set) - good alternative
    3. Demo mode (if no keys) - always works

    Returns:
        Model name ("gpt-4o", "claude-3-sonnet-20240229", or "demo")
    """
    if has_openai_key():
        model = "gpt-4o"  # Good balance of cost and quality
        logger.info(f"Auto-detected: OpenAI (using {model})")
        return model

    if has_anthropic_key():
        model = "claude-3-sonnet-20240229"
        logger.info(f"Auto-detected: Anthropic (using {model})")
        return model

    logger.warning(
        "No API keys found (OPENAI_API_KEY or ANTHROPIC_API_KEY). "
        "Using demo mode with FakeProvider. "
        "Responses are pre-recorded and clearly marked as demo. "
        "For production use, please set an API key environment variable."
    )
    return "demo"


def create_demo_provider(config: Optional[IntelligenceConfig] = None) -> LLMProvider:
    """
    Create FakeProvider for demo mode

    Args:
        config: Optional configuration (uses defaults if None)

    Returns:
        FakeProvider instance
    """
    from greenlang.agents.intelligence.providers.fake import FakeProvider

    if config is None:
        config = get_config_from_env()

    provider_config = LLMProviderConfig(
        model="demo",
        api_key_env="",  # Not used by FakeProvider
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )

    logger.info(
        "Creating FakeProvider (demo mode) - No API key required. "
        "Responses are simulated. Set OPENAI_API_KEY or ANTHROPIC_API_KEY for production use."
    )

    return FakeProvider(provider_config)


def create_openai_provider(
    model: str,
    config: Optional[IntelligenceConfig] = None,
    api_key: Optional[str] = None,
) -> LLMProvider:
    """
    Create OpenAI provider

    Args:
        model: Model name (e.g., "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo")
        config: Optional configuration (uses defaults if None)
        api_key: Optional explicit API key (or use OPENAI_API_KEY env var)

    Returns:
        OpenAIProvider instance

    Raises:
        ValueError: If API key not found
        ImportError: If openai package not installed
    """
    from greenlang.agents.intelligence.providers.openai import OpenAIProvider

    if config is None:
        config = get_config_from_env()

    # Validate API key
    if not api_key and not has_openai_key():
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter."
        )

    provider_config = LLMProviderConfig(
        model=model,
        api_key_env="OPENAI_API_KEY",
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )

    logger.info(f"Creating OpenAI provider: model={model}")

    return OpenAIProvider(provider_config)


def create_anthropic_provider(
    model: str,
    config: Optional[IntelligenceConfig] = None,
    api_key: Optional[str] = None,
) -> LLMProvider:
    """
    Create Anthropic provider

    Args:
        model: Model name (e.g., "claude-3-sonnet-20240229", "claude-3-opus-20240229")
        config: Optional configuration (uses defaults if None)
        api_key: Optional explicit API key (or use ANTHROPIC_API_KEY env var)

    Returns:
        AnthropicProvider instance

    Raises:
        ValueError: If API key not found
        ImportError: If anthropic package not installed
    """
    from greenlang.agents.intelligence.providers.anthropic import AnthropicProvider

    if config is None:
        config = get_config_from_env()

    # Validate API key
    if not api_key and not has_anthropic_key():
        raise ValueError(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
        )

    provider_config = LLMProviderConfig(
        model=model,
        api_key_env="ANTHROPIC_API_KEY",
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )

    logger.info(f"Creating Anthropic provider: model={model}")

    return AnthropicProvider(provider_config)


def create_deterministic_provider(config: Optional[IntelligenceConfig] = None) -> LLMProvider:
    """
    Create Tier 0 Deterministic Provider.

    Always available, no dependencies. Provides REAL intelligence
    using templates, rules, and statistical analysis.

    Args:
        config: Optional configuration

    Returns:
        DeterministicProvider instance
    """
    from greenlang.agents.intelligence.providers.deterministic import DeterministicProvider

    if config is None:
        config = get_config_from_env()

    provider_config = LLMProviderConfig(
        model="deterministic",
        api_key_env="",
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )

    logger.info(
        "Creating DeterministicProvider (Tier 0) - "
        "Production-ready intelligence, no dependencies required."
    )

    return DeterministicProvider(provider_config)


def create_ollama_provider(
    model: str = "llama3.2",
    config: Optional[IntelligenceConfig] = None,
    host: Optional[str] = None,
) -> LLMProvider:
    """
    Create Tier 1 Ollama Local LLM Provider.

    Real LLM intelligence running locally - no API key, no cost, full privacy.

    Args:
        model: Ollama model name (default: llama3.2)
        config: Optional configuration
        host: Ollama API host (default: http://localhost:11434)

    Returns:
        OllamaProvider instance
    """
    from greenlang.agents.intelligence.providers.ollama import OllamaProvider

    if config is None:
        config = get_config_from_env()

    # Clean model name
    if model.startswith("ollama:"):
        model = model[7:]

    provider_config = LLMProviderConfig(
        model=f"ollama:{model}",
        api_key_env="",
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )

    ollama_host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    logger.info(
        f"Creating OllamaProvider (Tier 1) - "
        f"Local LLM: {model} at {ollama_host}"
    )

    return OllamaProvider(provider_config, host=ollama_host, model=model)


def create_smart_router(
    prefer_local: bool = False,
    config: Optional[IntelligenceConfig] = None,
) -> LLMProvider:
    """
    Create Smart Provider Router (Recommended).

    Automatically selects best available tier:
    - Tier 2: BYOK (OpenAI/Anthropic) if API keys available
    - Tier 1: Ollama if running locally
    - Tier 0: Deterministic (always available)

    Args:
        prefer_local: Prefer Ollama over cloud even if API keys available
        config: Optional configuration

    Returns:
        SmartProviderRouter instance
    """
    from greenlang.agents.intelligence.providers.router import (
        SmartProviderRouter,
        RouterConfig,
    )

    if config is None:
        config = get_config_from_env()

    router_config = RouterConfig(
        prefer_local=prefer_local,
        detection_timeout_s=config.timeout_s,
    )

    provider_config = LLMProviderConfig(
        model="auto",
        api_key_env="",
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )

    logger.info(
        "Creating SmartProviderRouter - "
        "Will auto-detect best available intelligence tier"
    )

    return SmartProviderRouter(provider_config, router_config=router_config)


def create_provider(
    model: str = "auto",
    api_key: Optional[str] = None,
    config: Optional[IntelligenceConfig] = None,
    prefer_local: bool = False,
) -> LLMProvider:
    """
    Smart provider factory - Multi-Tier Intelligence System.

    Works for EVERYONE - from first-time open-source users to enterprise deployments.

    TIER SYSTEM:
    - "auto" (default): Smart router auto-selects best available tier
    - "deterministic": Tier 0 - Template-based, always works
    - "ollama:*": Tier 1 - Local LLM, free and private
    - "gpt-*" / "claude-*": Tier 2 - Cloud LLM with API keys
    - "demo": Legacy demo mode (FakeProvider)

    This is the main entry point for creating LLM providers in GreenLang.
    It enables TRUE zero-config setup with REAL intelligence from day one!

    Args:
        model: Model selection:
            - "auto" (default): Smart router auto-detects best tier
            - "deterministic": Force Tier 0 (no dependencies)
            - "ollama:llama3.2", "ollama:mistral": Force Tier 1 local LLM
            - "gpt-4o", "claude-3-sonnet": Force Tier 2 cloud LLM
            - "demo": Legacy demo mode
        api_key: Optional explicit API key (for Tier 2)
        config: Optional configuration (uses defaults if None)
        prefer_local: Prefer Ollama over cloud even if API keys available

    Returns:
        LLMProvider instance

    Examples:
        # Auto-detect best tier (recommended)
        provider = create_provider()
        # -> Uses Tier 2 if API key set
        # -> Uses Tier 1 if Ollama running
        # -> Uses Tier 0 otherwise (real intelligence, not demo!)

        # Force deterministic (Tier 0) - always works
        provider = create_provider(model="deterministic")

        # Force local LLM (Tier 1)
        provider = create_provider(model="ollama:llama3.2")

        # Force cloud LLM (Tier 2)
        provider = create_provider(model="gpt-4o")

        # Prefer local over cloud
        provider = create_provider(prefer_local=True)

    Tier Report:
        from greenlang.agents.intelligence.providers.router import get_tier_report_sync
        print(get_tier_report_sync())  # Shows all tier statuses
    """
    if config is None:
        config = get_config_from_env()

    # Override with explicit demo_mode (legacy compatibility)
    if config.demo_mode:
        model = "demo"

    # Resolve aliases
    model = resolve_model_alias(model)

    # Smart router (recommended for "auto")
    if model == "auto":
        return create_smart_router(prefer_local=prefer_local, config=config)

    # Tier 0: Deterministic
    if model == "deterministic":
        return create_deterministic_provider(config)

    # Tier 1: Ollama local LLM
    if model.startswith("ollama"):
        ollama_model = model[7:] if model.startswith("ollama:") else "llama3.2"
        return create_ollama_provider(ollama_model, config)

    # Legacy demo mode (now recommends Tier 0)
    if model == "demo":
        logger.info(
            "Note: 'demo' mode uses FakeProvider with canned responses. "
            "Consider using 'deterministic' for real template-based intelligence."
        )
        return create_demo_provider(config)

    # Tier 2: OpenAI models
    if model.startswith("gpt"):
        return create_openai_provider(model, config, api_key)

    # Tier 2: Anthropic models
    if model.startswith("claude"):
        return create_anthropic_provider(model, config, api_key)

    # Unknown model - try to infer from API keys
    if has_openai_key():
        logger.warning(f"Unknown model '{model}', defaulting to OpenAI with this model name")
        return create_openai_provider(model, config, api_key)

    if has_anthropic_key():
        logger.warning(f"Unknown model '{model}', defaulting to Anthropic with this model name")
        return create_anthropic_provider(model, config, api_key)

    # Fallback to deterministic (NOT demo mode!)
    logger.info(
        f"Unknown model '{model}' and no API keys found. "
        "Using Tier 0 DeterministicProvider for real template-based intelligence. "
        "For LLM intelligence: install Ollama or set API keys."
    )
    return create_deterministic_provider(config)


if __name__ == "__main__":
    """
    Factory examples and tests

    Run with:
        # Without API key (uses demo mode)
        python -m greenlang.intelligence.factory

        # With API key (uses real provider)
        export OPENAI_API_KEY=sk-...
        python -m greenlang.intelligence.factory
    """
    import asyncio
    from greenlang.agents.intelligence.schemas.messages import ChatMessage, Role
    from greenlang.agents.intelligence.runtime.budget import Budget

    async def example_auto_detect():
        """Example: Auto-detect provider"""
        print("\n=== Example: Auto-detect Provider ===")

        # Create provider (auto-detects based on API keys)
        provider = create_provider(model="auto")

        print(f"Provider type: {type(provider).__name__}")
        print(f"Model: {provider.config.model}")
        print(f"Capabilities: function_calling={provider.capabilities.function_calling}, "
              f"json_schema={provider.capabilities.json_schema_mode}")

        # Test chat
        messages = [
            ChatMessage(role=Role.user, content="What's the carbon intensity in Texas?")
        ]

        budget = Budget(max_usd=0.10)

        response = await provider.chat(messages=messages, budget=budget)

        print(f"\nResponse preview: {response.text[:200]}...")
        print(f"Cost: ${response.usage.cost_usd:.6f}")
        print(f"Provider: {response.provider_info.provider}")

    async def example_explicit_demo():
        """Example: Explicit demo mode"""
        print("\n=== Example: Explicit Demo Mode ===")

        # Force demo mode (no API key required)
        provider = create_provider(model="demo")

        print(f"Provider type: {type(provider).__name__}")

        messages = [
            ChatMessage(role=Role.user, content="Calculate emissions for 100 gallons of diesel")
        ]

        budget = Budget(max_usd=0.10)

        response = await provider.chat(messages=messages, budget=budget)

        print(f"\nResponse preview: {response.text[:200]}...")
        print(f"Cost: ${response.usage.cost_usd:.6f}")

    async def example_check_api_keys():
        """Example: Check API key availability"""
        print("\n=== Example: Check API Keys ===")

        print(f"OpenAI key available: {has_openai_key()}")
        print(f"Anthropic key available: {has_anthropic_key()}")
        print(f"Any key available: {has_any_api_key()}")

        if has_any_api_key():
            print("\nAPI keys found! Provider will use real LLM.")
        else:
            print("\nNo API keys found. Provider will use demo mode.")
            print("To use real LLMs:")
            print("  export OPENAI_API_KEY=sk-...")
            print("  or")
            print("  export ANTHROPIC_API_KEY=sk-...")

    async def main():
        """Run all examples"""
        print("GreenLang Intelligence Factory")
        print("=" * 60)

        await example_check_api_keys()
        await example_auto_detect()
        await example_explicit_demo()

        print("\n" + "=" * 60)
        print("Factory examples complete!")

    asyncio.run(main())
