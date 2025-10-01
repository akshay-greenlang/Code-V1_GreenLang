"""
Intelligence Layer Configuration

Default configuration for GreenLang Intelligence Layer with zero-config setup.
Works out of the box without API keys (uses FakeProvider in demo mode).

Configuration priority:
1. Explicit parameters to create_provider()
2. Environment variables
3. Default configuration (demo mode)

Environment variables:
- OPENAI_API_KEY: OpenAI API key
- ANTHROPIC_API_KEY: Anthropic/Claude API key
- GREENLANG_INTELLIGENCE_MODEL: Preferred model ("auto", "gpt-4", "claude-3", "demo")
- GREENLANG_INTELLIGENCE_TIMEOUT: Request timeout in seconds (default: 60)
- GREENLANG_INTELLIGENCE_MAX_RETRIES: Max retry attempts (default: 3)
"""

from __future__ import annotations
import os
from typing import Optional, Literal
from pydantic import BaseModel, Field


class IntelligenceConfig(BaseModel):
    """
    Configuration for Intelligence Layer

    Zero-config defaults that work without API keys.
    Automatically upgrades to real providers when API keys are available.

    Attributes:
        model: Model selection ("auto", specific model, or "demo")
        api_key: Optional explicit API key (or use env var)
        timeout_s: Request timeout in seconds
        max_retries: Max retry attempts on transient failures
        demo_mode: Force demo mode even if API keys available
    """

    model: str = Field(
        default="auto",
        description="Model selection: 'auto', 'gpt-4', 'gpt-4o', 'claude-3-sonnet', 'demo'"
    )

    api_key: Optional[str] = Field(
        default=None,
        description="Optional explicit API key (or use environment variables)"
    )

    timeout_s: float = Field(
        default=60.0,
        description="Request timeout in seconds"
    )

    max_retries: int = Field(
        default=3,
        description="Max retry attempts on transient failures"
    )

    demo_mode: bool = Field(
        default=False,
        description="Force demo mode even if API keys available"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "model": "auto",
                    "timeout_s": 60.0,
                    "max_retries": 3,
                    "demo_mode": False
                }
            ]
        }


# Default configuration (works without API keys)
DEFAULT_CONFIG = IntelligenceConfig(
    model="auto",
    timeout_s=60.0,
    max_retries=3,
    demo_mode=False,
)


def get_config_from_env() -> IntelligenceConfig:
    """
    Load configuration from environment variables

    Reads:
    - GREENLANG_INTELLIGENCE_MODEL
    - GREENLANG_INTELLIGENCE_TIMEOUT
    - GREENLANG_INTELLIGENCE_MAX_RETRIES

    Returns:
        Configuration with environment overrides applied
    """
    model = os.getenv("GREENLANG_INTELLIGENCE_MODEL", DEFAULT_CONFIG.model)

    timeout_s = float(
        os.getenv("GREENLANG_INTELLIGENCE_TIMEOUT", str(DEFAULT_CONFIG.timeout_s))
    )

    max_retries = int(
        os.getenv("GREENLANG_INTELLIGENCE_MAX_RETRIES", str(DEFAULT_CONFIG.max_retries))
    )

    return IntelligenceConfig(
        model=model,
        timeout_s=timeout_s,
        max_retries=max_retries,
        demo_mode=False,
    )


# Model aliases for convenience
MODEL_ALIASES = {
    # OpenAI
    "gpt-4": "gpt-4-turbo",
    "gpt4": "gpt-4-turbo",
    "openai": "gpt-4-turbo",

    # Anthropic
    "claude": "claude-3-sonnet-20240229",
    "claude-3": "claude-3-sonnet-20240229",
    "anthropic": "claude-3-sonnet-20240229",

    # Demo
    "fake": "demo",
    "test": "demo",
}


def resolve_model_alias(model: str) -> str:
    """
    Resolve model alias to actual model name

    Args:
        model: Model name or alias

    Returns:
        Resolved model name

    Example:
        >>> resolve_model_alias("gpt4")
        "gpt-4-turbo"
        >>> resolve_model_alias("claude")
        "claude-3-sonnet-20240229"
    """
    return MODEL_ALIASES.get(model.lower(), model)


# Recommended models by use case
RECOMMENDED_MODELS = {
    "cost_optimized": "gpt-4o-mini",  # Cheapest, good quality
    "balanced": "gpt-4o",  # Good balance of cost and quality
    "best_quality": "gpt-4-turbo",  # Highest quality, most expensive
    "demo": "demo",  # No API key required
}


def get_recommended_model(use_case: Literal["cost_optimized", "balanced", "best_quality", "demo"] = "balanced") -> str:
    """
    Get recommended model for use case

    Args:
        use_case: One of "cost_optimized", "balanced", "best_quality", "demo"

    Returns:
        Recommended model name

    Example:
        >>> get_recommended_model("cost_optimized")
        "gpt-4o-mini"
        >>> get_recommended_model("demo")
        "demo"
    """
    return RECOMMENDED_MODELS.get(use_case, RECOMMENDED_MODELS["balanced"])


if __name__ == "__main__":
    """
    Configuration examples and tests
    """
    print("GreenLang Intelligence Configuration")
    print("=" * 50)

    # Default config
    print("\n1. Default configuration:")
    print(f"   Model: {DEFAULT_CONFIG.model}")
    print(f"   Timeout: {DEFAULT_CONFIG.timeout_s}s")
    print(f"   Max retries: {DEFAULT_CONFIG.max_retries}")

    # Environment-based config
    print("\n2. Configuration from environment:")
    env_config = get_config_from_env()
    print(f"   Model: {env_config.model}")
    print(f"   Timeout: {env_config.timeout_s}s")
    print(f"   Max retries: {env_config.max_retries}")

    # Model aliases
    print("\n3. Model aliases:")
    for alias, actual in MODEL_ALIASES.items():
        print(f"   {alias:20} -> {actual}")

    # Recommended models
    print("\n4. Recommended models by use case:")
    for use_case, model in RECOMMENDED_MODELS.items():
        print(f"   {use_case:20} -> {model}")

    print("\n" + "=" * 50)
    print("To customize configuration:")
    print("  1. Set environment variables:")
    print("     export GREENLANG_INTELLIGENCE_MODEL=gpt-4o")
    print("     export GREENLANG_INTELLIGENCE_TIMEOUT=30")
    print("  2. Or pass explicit config to create_provider()")
