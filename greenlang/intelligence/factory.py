"""
Smart Provider Factory

Auto-detects available API keys and creates appropriate LLM provider:
- If API keys present → use real OpenAI/Anthropic providers
- If NO keys → use FakeProvider with demo responses
- Logs which mode is active for user awareness

This enables zero-config setup: import and use immediately without API keys!

Usage:
    from greenlang.intelligence import create_provider

    # Auto-detect (uses demo mode if no keys)
    provider = create_provider()

    # Or specify model
    provider = create_provider(model="gpt-4o")  # Requires OPENAI_API_KEY

    # Or force demo mode
    provider = create_provider(model="demo")
"""

from __future__ import annotations
import os
import logging
from typing import Optional

from greenlang.intelligence.providers.base import LLMProvider, LLMProviderConfig
from greenlang.intelligence.config import (
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
    from greenlang.intelligence.providers.fake import FakeProvider

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
    from greenlang.intelligence.providers.openai import OpenAIProvider

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
    from greenlang.intelligence.providers.anthropic import AnthropicProvider

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


def create_provider(
    model: str = "auto",
    api_key: Optional[str] = None,
    config: Optional[IntelligenceConfig] = None,
) -> LLMProvider:
    """
    Smart provider factory - works without API keys!

    Auto-detects available API keys and creates appropriate provider:
    - "auto": Auto-detect best available (OpenAI > Anthropic > Demo)
    - "demo": Use FakeProvider (no API key required)
    - "gpt-*": Use OpenAI (requires OPENAI_API_KEY)
    - "claude-*": Use Anthropic (requires ANTHROPIC_API_KEY)

    This is the main entry point for creating LLM providers in GreenLang.
    It enables zero-config setup: works immediately without API keys!

    Args:
        model: Model selection:
            - "auto" (default): Auto-detect best available provider
            - "demo": Force demo mode (no API key required)
            - "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo": OpenAI models
            - "claude-3-sonnet-20240229", "claude-3-opus-20240229": Anthropic models
            - Aliases: "gpt4" -> "gpt-4-turbo", "claude" -> "claude-3-sonnet"
        api_key: Optional explicit API key (or use environment variables)
        config: Optional configuration (uses defaults if None)

    Returns:
        LLMProvider instance (OpenAIProvider, AnthropicProvider, or FakeProvider)

    Raises:
        ValueError: If model requires API key but none found
        ImportError: If required provider package not installed

    Examples:
        # Zero-config: Auto-detect (uses demo mode if no keys)
        provider = create_provider()
        # -> Uses demo mode if no API keys found
        # -> Uses OpenAI if OPENAI_API_KEY set
        # -> Uses Anthropic if ANTHROPIC_API_KEY set

        # Explicit demo mode (no API key required)
        provider = create_provider(model="demo")

        # Specific model (requires API key)
        provider = create_provider(model="gpt-4o")
        # -> Requires OPENAI_API_KEY environment variable

        # With explicit API key
        provider = create_provider(model="gpt-4o", api_key="sk-...")

        # With custom configuration
        from greenlang.intelligence.config import IntelligenceConfig
        config = IntelligenceConfig(timeout_s=30.0, max_retries=5)
        provider = create_provider(model="auto", config=config)

    Usage with ChatSession:
        from greenlang.intelligence import create_provider
        from greenlang.intelligence.runtime.session import ChatSession
        from greenlang.intelligence.schemas.messages import ChatMessage, Role
        from greenlang.intelligence.runtime.budget import Budget

        # Create provider (works without API keys!)
        provider = create_provider()

        # Create session
        session = ChatSession(provider)

        # Chat
        response = await session.chat(
            messages=[
                ChatMessage(
                    role=Role.user,
                    content="What's the carbon intensity in California?"
                )
            ],
            budget=Budget(max_usd=0.50)
        )
        print(response.text)
    """
    if config is None:
        config = get_config_from_env()

    # Override with explicit demo_mode
    if config.demo_mode:
        model = "demo"

    # Resolve aliases
    model = resolve_model_alias(model)

    # Auto-detect mode
    if model == "auto":
        model = detect_best_provider()

    # Demo mode
    if model == "demo":
        return create_demo_provider(config)

    # OpenAI models
    if model.startswith("gpt"):
        return create_openai_provider(model, config, api_key)

    # Anthropic models
    if model.startswith("claude"):
        return create_anthropic_provider(model, config, api_key)

    # Unknown model - try to infer from API keys
    if has_openai_key():
        logger.warning(f"Unknown model '{model}', defaulting to OpenAI with this model name")
        return create_openai_provider(model, config, api_key)

    if has_anthropic_key():
        logger.warning(f"Unknown model '{model}', defaulting to Anthropic with this model name")
        return create_anthropic_provider(model, config, api_key)

    # No API keys - use demo mode
    logger.warning(
        f"Unknown model '{model}' and no API keys found. Using demo mode. "
        "Set OPENAI_API_KEY or ANTHROPIC_API_KEY for production use."
    )
    return create_demo_provider(config)


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
    from greenlang.intelligence.schemas.messages import ChatMessage, Role
    from greenlang.intelligence.runtime.budget import Budget

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
