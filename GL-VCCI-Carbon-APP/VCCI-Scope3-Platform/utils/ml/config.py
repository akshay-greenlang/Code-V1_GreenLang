# GL-VCCI ML Module - Configuration
# Spend Classification ML System - Configuration Models

"""
ML Configuration
================

Pydantic configuration models for the Spend Classification ML system.

Configuration Hierarchy:
-----------------------
MLConfig (root)
├── LLMConfig
│   ├── provider: "openai" | "anthropic"
│   ├── model_name: e.g., "gpt-3.5-turbo", "claude-3-haiku-20240307"
│   ├── temperature: 0.0-1.0
│   ├── max_tokens: int
│   └── timeout_seconds: float
├── ClassificationConfig
│   ├── confidence_threshold: 0.85 (default)
│   ├── use_llm_primary: true
│   ├── use_rules_fallback: true
│   └── require_human_review_threshold: 0.5
├── RulesEngineConfig
│   ├── keyword_matching: true
│   ├── regex_patterns: true
│   └── fuzzy_matching: true
├── CacheConfig
│   ├── enabled: true
│   ├── ttl_seconds: 2592000 (30 days)
│   └── redis_url: str
└── CostTrackingConfig
    ├── enabled: true
    └── log_every_n_requests: 100

Usage:
------
```python
from utils.ml.config import MLConfig, LLMConfig

# Load from environment variables
config = MLConfig()

# Custom configuration
config = MLConfig(
    llm=LLMConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.1
    ),
    classification=ClassificationConfig(
        confidence_threshold=0.85
    )
)

# Access settings
if config.llm.provider == "openai":
    api_key = config.llm.api_key
```
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================================
# Enums
# ============================================================================

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Scope3Category(str, Enum):
    """
    GHG Protocol Scope 3 Categories (15 categories).

    Reference: https://ghgprotocol.org/standards/scope-3-standard
    """
    CATEGORY_1 = "category_1_purchased_goods_services"  # Purchased Goods & Services
    CATEGORY_2 = "category_2_capital_goods"  # Capital Goods
    CATEGORY_3 = "category_3_fuel_energy_related"  # Fuel and Energy Related Activities
    CATEGORY_4 = "category_4_upstream_transportation"  # Upstream Transportation & Distribution
    CATEGORY_5 = "category_5_waste"  # Waste Generated in Operations
    CATEGORY_6 = "category_6_business_travel"  # Business Travel
    CATEGORY_7 = "category_7_employee_commuting"  # Employee Commuting
    CATEGORY_8 = "category_8_upstream_leased_assets"  # Upstream Leased Assets
    CATEGORY_9 = "category_9_downstream_transportation"  # Downstream Transportation & Distribution
    CATEGORY_10 = "category_10_processing_sold_products"  # Processing of Sold Products
    CATEGORY_11 = "category_11_use_sold_products"  # Use of Sold Products
    CATEGORY_12 = "category_12_end_of_life_treatment"  # End-of-Life Treatment of Sold Products
    CATEGORY_13 = "category_13_downstream_leased_assets"  # Downstream Leased Assets
    CATEGORY_14 = "category_14_franchises"  # Franchises
    CATEGORY_15 = "category_15_investments"  # Investments

    @classmethod
    def get_category_name(cls, category: str) -> str:
        """Get human-readable category name."""
        category_names = {
            cls.CATEGORY_1.value: "Purchased Goods & Services",
            cls.CATEGORY_2.value: "Capital Goods",
            cls.CATEGORY_3.value: "Fuel and Energy Related Activities",
            cls.CATEGORY_4.value: "Upstream Transportation & Distribution",
            cls.CATEGORY_5.value: "Waste Generated in Operations",
            cls.CATEGORY_6.value: "Business Travel",
            cls.CATEGORY_7.value: "Employee Commuting",
            cls.CATEGORY_8.value: "Upstream Leased Assets",
            cls.CATEGORY_9.value: "Downstream Transportation & Distribution",
            cls.CATEGORY_10.value: "Processing of Sold Products",
            cls.CATEGORY_11.value: "Use of Sold Products",
            cls.CATEGORY_12.value: "End-of-Life Treatment of Sold Products",
            cls.CATEGORY_13.value: "Downstream Leased Assets",
            cls.CATEGORY_14.value: "Franchises",
            cls.CATEGORY_15.value: "Investments",
        }
        return category_names.get(category, category)

    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Get list of all valid categories."""
        return [c.value for c in cls]


# ============================================================================
# LLM Configuration
# ============================================================================

class LLMConfig(BaseModel):
    """
    LLM provider configuration.

    Attributes:
        provider: LLM provider (openai or anthropic)
        model_name: Model identifier (e.g., "gpt-3.5-turbo", "claude-3-haiku-20240307")
        api_key: API key (loaded from environment)
        temperature: Sampling temperature (0.0-1.0, lower = more deterministic)
        max_tokens: Maximum tokens in response
        timeout_seconds: Request timeout in seconds
        max_retries: Maximum retry attempts
        retry_delay_seconds: Initial retry delay (exponential backoff)
    """
    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider (openai or anthropic)"
    )
    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="Model identifier"
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key (loaded from environment)"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (lower = more deterministic)"
    )
    max_tokens: int = Field(
        default=500,
        gt=0,
        le=4096,
        description="Maximum tokens in response"
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts"
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        gt=0,
        description="Initial retry delay (exponential backoff: 1s, 2s, 4s, 8s)"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str, info) -> str:
        """Validate model name matches provider."""
        provider = info.data.get("provider")
        if provider == LLMProvider.OPENAI:
            valid_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
            if v not in valid_models:
                # Allow any gpt-* model
                if not v.startswith("gpt-"):
                    raise ValueError(f"OpenAI model must be one of {valid_models} or start with 'gpt-'")
        elif provider == LLMProvider.ANTHROPIC:
            # Allow any claude-* model
            if not v.startswith("claude-"):
                raise ValueError("Anthropic model must start with 'claude-'")
        return v


# ============================================================================
# Classification Configuration
# ============================================================================

class ClassificationConfig(BaseModel):
    """
    Classification behavior configuration.

    Attributes:
        confidence_threshold: Minimum confidence for accepting classification (0.0-1.0)
        use_llm_primary: Use LLM as primary classifier
        use_rules_fallback: Use rules engine as fallback when LLM confidence is low
        require_human_review_threshold: Flag for human review below this confidence
        enable_multi_category: Allow multi-category classification
        max_categories: Maximum categories per classification (if multi-category enabled)
    """
    confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for accepting classification"
    )
    use_llm_primary: bool = Field(
        default=True,
        description="Use LLM as primary classifier"
    )
    use_rules_fallback: bool = Field(
        default=True,
        description="Use rules engine as fallback"
    )
    require_human_review_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Flag for human review below this confidence"
    )
    enable_multi_category: bool = Field(
        default=False,
        description="Allow multi-category classification"
    )
    max_categories: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum categories per classification"
    )

    @field_validator("require_human_review_threshold")
    @classmethod
    def validate_review_threshold(cls, v: float, info) -> float:
        """Ensure review threshold <= confidence threshold."""
        confidence_threshold = info.data.get("confidence_threshold", 0.85)
        if v > confidence_threshold:
            raise ValueError(
                f"require_human_review_threshold ({v}) must be <= "
                f"confidence_threshold ({confidence_threshold})"
            )
        return v


# ============================================================================
# Rules Engine Configuration
# ============================================================================

class RulesEngineConfig(BaseModel):
    """
    Rule-based classification configuration.

    Attributes:
        enable_keyword_matching: Enable keyword-based classification
        enable_regex_patterns: Enable regex pattern matching
        enable_fuzzy_matching: Enable fuzzy string matching
        fuzzy_threshold: Minimum fuzzy match score (0.0-1.0)
        keyword_case_sensitive: Case-sensitive keyword matching
    """
    enable_keyword_matching: bool = Field(
        default=True,
        description="Enable keyword-based classification"
    )
    enable_regex_patterns: bool = Field(
        default=True,
        description="Enable regex pattern matching"
    )
    enable_fuzzy_matching: bool = Field(
        default=True,
        description="Enable fuzzy string matching"
    )
    fuzzy_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum fuzzy match score"
    )
    keyword_case_sensitive: bool = Field(
        default=False,
        description="Case-sensitive keyword matching"
    )


# ============================================================================
# Cache Configuration
# ============================================================================

class CacheConfig(BaseModel):
    """
    Classification cache configuration.

    Attributes:
        enabled: Enable caching
        ttl_seconds: Cache TTL in seconds (default: 30 days)
        redis_url: Redis connection URL
        key_prefix: Cache key prefix
    """
    enabled: bool = Field(
        default=True,
        description="Enable classification caching"
    )
    ttl_seconds: int = Field(
        default=2592000,  # 30 days
        gt=0,
        description="Cache TTL in seconds"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    key_prefix: str = Field(
        default="vcci:ml:classification:",
        description="Cache key prefix"
    )


# ============================================================================
# Cost Tracking Configuration
# ============================================================================

class CostTrackingConfig(BaseModel):
    """
    LLM cost tracking configuration.

    Attributes:
        enabled: Enable cost tracking
        log_every_n_requests: Log cost summary every N requests
        openai_pricing: OpenAI pricing per 1K tokens (input, output)
        anthropic_pricing: Anthropic pricing per 1K tokens (input, output)
    """
    enabled: bool = Field(
        default=True,
        description="Enable cost tracking"
    )
    log_every_n_requests: int = Field(
        default=100,
        gt=0,
        description="Log cost summary every N requests"
    )
    openai_pricing: Dict[str, Dict[str, float]] = Field(
        default={
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        },
        description="OpenAI pricing per 1K tokens"
    )
    anthropic_pricing: Dict[str, Dict[str, float]] = Field(
        default={
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        },
        description="Anthropic pricing per 1K tokens"
    )


# ============================================================================
# Main Configuration
# ============================================================================

class MLConfig(BaseSettings):
    """
    Root ML configuration.

    Loads configuration from environment variables and .env file.

    Environment Variables:
    ---------------------
    LLM_PROVIDER: LLM provider (openai or anthropic)
    LLM_MODEL_NAME: Model name
    LLM_API_KEY: API key (required)
    OPENAI_API_KEY: OpenAI API key (alternative)
    ANTHROPIC_API_KEY: Anthropic API key (alternative)
    REDIS_URL: Redis connection URL
    ML_CONFIDENCE_THRESHOLD: Classification confidence threshold

    Usage:
    ------
    ```python
    # Load from environment
    config = MLConfig()

    # Access nested config
    llm_provider = config.llm.provider
    confidence = config.classification.confidence_threshold
    ```
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False
    )

    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration"
    )
    classification: ClassificationConfig = Field(
        default_factory=ClassificationConfig,
        description="Classification configuration"
    )
    rules_engine: RulesEngineConfig = Field(
        default_factory=RulesEngineConfig,
        description="Rules engine configuration"
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache configuration"
    )
    cost_tracking: CostTrackingConfig = Field(
        default_factory=CostTrackingConfig,
        description="Cost tracking configuration"
    )

    @field_validator("llm")
    @classmethod
    def validate_llm_config(cls, v: LLMConfig) -> LLMConfig:
        """Validate LLM configuration has API key."""
        if v.api_key is None:
            # Try to get from environment
            import os
            if v.provider == LLMProvider.OPENAI:
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            else:  # ANTHROPIC
                api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")

            if api_key:
                v.api_key = SecretStr(api_key)
            else:
                raise ValueError(
                    f"API key required for {v.provider.value}. "
                    f"Set {v.provider.value.upper()}_API_KEY or LLM_API_KEY environment variable."
                )
        return v

    def get_valid_categories(self) -> List[str]:
        """Get list of all valid Scope 3 categories."""
        return Scope3Category.get_all_categories()

    def get_category_name(self, category: str) -> str:
        """Get human-readable category name."""
        return Scope3Category.get_category_name(category)


# ============================================================================
# Convenience Functions
# ============================================================================

def load_config(env_file: Optional[str] = None) -> MLConfig:
    """
    Load ML configuration from environment.

    Args:
        env_file: Optional path to .env file

    Returns:
        Loaded configuration

    Example:
        >>> config = load_config(".env.production")
        >>> config.llm.provider
        'openai'
    """
    if env_file:
        return MLConfig(_env_file=env_file)
    return MLConfig()


def get_default_config() -> MLConfig:
    """
    Get default ML configuration.

    Returns:
        Default configuration with sensible defaults

    Example:
        >>> config = get_default_config()
        >>> config.classification.confidence_threshold
        0.85
    """
    return MLConfig()
