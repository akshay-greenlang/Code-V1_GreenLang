# -*- coding: utf-8 -*-
"""
Provider Factory Integration with Dependency Injection
=======================================================

This module integrates the LLM provider factory with the ServiceContainer,
enabling dependency injection of LLM providers throughout the application.

Benefits:
- Centralized provider configuration
- Easy testing with mock providers
- Singleton pattern for resource efficiency
- Config-driven provider selection

Usage:
    >>> from greenlang.config import get_config, ServiceContainer
    >>> from greenlang.config.providers import register_provider_factory
    >>>
    >>> # Setup DI container
    >>> container = ServiceContainer()
    >>> container.register_singleton(type(get_config()), lambda c: get_config())
    >>> register_provider_factory(container)
    >>>
    >>> # Resolve provider (automatically uses config)
    >>> provider = container.resolve(LLMProvider)
    >>> response = await provider.generate_async("Hello")

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from greenlang.config.container import ServiceContainer
    from greenlang.intelligence.providers.base import LLMProvider
    from greenlang.config.schemas import GreenLangConfig


def create_provider_from_config(config: GreenLangConfig) -> LLMProvider:
    """
    Create LLM provider from GreenLangConfig.

    This is a factory function compatible with ServiceContainer.

    Args:
        config: GreenLangConfig instance with LLM settings

    Returns:
        LLMProvider instance (OpenAI, Anthropic, or Fake)

    Example:
        >>> config = GreenLangConfig(llm={"provider": "openai", "model": "gpt-4"})
        >>> provider = create_provider_from_config(config)
        >>> response = await provider.generate_async("Hello")
    """
    from greenlang.intelligence import create_provider
    from greenlang.intelligence.config import IntelligenceConfig

    # Convert GreenLangConfig to IntelligenceConfig
    intelligence_config = IntelligenceConfig(
        default_model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        timeout_s=config.llm.timeout_seconds,
        max_retries=config.llm.max_retries,
    )

    # Use provider selection from config
    model = config.llm.model

    # Create provider
    provider = create_provider(
        model=model,
        config=intelligence_config,
        # API key from environment (or use config.llm.api_key if set)
    )

    return provider


def register_provider_factory(container: ServiceContainer) -> None:
    """
    Register LLM provider factory in ServiceContainer.

    This registers the provider as a singleton, meaning one instance
    is shared across the application for resource efficiency.

    Args:
        container: ServiceContainer instance

    Example:
        >>> container = ServiceContainer()
        >>> container.register_singleton(type(get_config()), lambda c: get_config())
        >>> register_provider_factory(container)
        >>>
        >>> # Now you can resolve providers anywhere
        >>> provider = container.resolve(LLMProvider)
    """
    from greenlang.intelligence.providers.base import LLMProvider
    from greenlang.config.schemas import GreenLangConfig

    # Register provider factory as singleton
    # It will automatically resolve config dependency
    container.register_singleton(
        LLMProvider,
        lambda c: create_provider_from_config(c.resolve(GreenLangConfig))
    )


def setup_provider_di(container: ServiceContainer, config: GreenLangConfig) -> None:
    """
    Convenience function to setup provider DI in one call.

    This registers both the config and provider factory.

    Args:
        container: ServiceContainer instance
        config: GreenLangConfig instance

    Example:
        >>> from greenlang.config import ServiceContainer, get_config
        >>> from greenlang.config.providers import setup_provider_di
        >>>
        >>> container = ServiceContainer()
        >>> config = get_config()
        >>> setup_provider_di(container, config)
        >>>
        >>> # Ready to use!
        >>> provider = container.resolve(LLMProvider)
    """
    from greenlang.config.schemas import GreenLangConfig

    # Register config as singleton
    container.register_singleton(GreenLangConfig, lambda c: config)

    # Register provider factory
    register_provider_factory(container)


# ==============================================================================
# Example Usage
# ==============================================================================

async def example_di_provider_usage():
    """
    Example showing how to use provider DI.

    This demonstrates the full pattern:
    1. Create container
    2. Setup config and provider
    3. Resolve and use provider
    """
    from greenlang.config import ServiceContainer, get_config
    from greenlang.intelligence.providers.base import LLMProvider

    # 1. Create DI container
    container = ServiceContainer()

    # 2. Setup config and provider
    config = get_config()
    setup_provider_di(container, config)

    # 3. Resolve provider (automatically gets config)
    provider = container.resolve(LLMProvider)

    # 4. Use provider
    response = await provider.generate_async(
        prompt="What is the carbon intensity of California's grid?",
        temperature=0.0,
        max_tokens=500
    )

    print(f"Provider: {provider.__class__.__name__}")
    print(f"Response: {response}")

    return response


async def example_agent_with_di():
    """
    Example showing how agents can use DI for providers.

    This demonstrates injecting providers into agents.
    """
    from greenlang.config import ServiceContainer, get_config
    from greenlang.intelligence.providers.base import LLMProvider
    from greenlang.agents.fuel_agent_ai_async import AsyncFuelAgentAI

    # Setup DI
    container = ServiceContainer()
    config = get_config()
    setup_provider_di(container, config)

    # Create agent with DI-resolved provider
    provider = container.resolve(LLMProvider)

    # Option 1: Pass provider to agent (future enhancement)
    # agent = AsyncFuelAgentAI(config, provider=provider)

    # Option 2: Agent creates provider internally from config (current)
    agent = AsyncFuelAgentAI(config)

    # Use agent
    async with agent:
        result = await agent.run_async({
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms",
            "country": "US"
        })

    print(f"Emissions: {result.data['co2e_emissions_kg']} kg CO2e")

    return result


if __name__ == "__main__":
    import asyncio

    # Run examples
    print("=" * 70)
    print("Example 1: Provider DI Usage")
    print("=" * 70)
    asyncio.run(example_di_provider_usage())

    print("\n" + "=" * 70)
    print("Example 2: Agent with DI")
    print("=" * 70)
    asyncio.run(example_agent_with_di())
