"""
Feature Flag Configuration - INFRA-008

Configuration management for the GreenLang feature flag system using Pydantic Settings.
Loads configuration from environment variables with the GL_FF_ prefix, with sensible
defaults for each deployment environment (dev, staging, prod).

Environment detection reads from the GL_ENVIRONMENT environment variable, defaulting
to "dev" if not set.

Classes:
    - FeatureFlagConfig: Main configuration model with all tunable parameters.
    - EnvironmentConfig: Pre-built configuration profiles for dev/staging/prod.

Functions:
    - get_config: Factory function that loads config from environment variables.

Example:
    # Set environment variables:
    #   GL_ENVIRONMENT=prod
    #   GL_FF_REDIS_URL=redis://redis-master.prod:6379/2
    #   GL_FF_POSTGRES_URL=postgresql+asyncpg://user:pass@db:5432/greenlang

    >>> from greenlang.infrastructure.feature_flags.config import get_config
    >>> config = get_config()
    >>> config.environment
    'prod'
    >>> config.l1_cache_ttl_seconds
    30
    >>> config.evaluation_log_sample_rate
    0.01
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment Profiles
# ---------------------------------------------------------------------------


class EnvironmentName(str, Enum):
    """Supported deployment environments."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class EnvironmentProfile:
    """Pre-built configuration profiles for each deployment environment.

    These profiles define the recommended defaults for cache TTLs, sampling
    rates, and operational limits. Values loaded from environment variables
    always override these defaults.
    """

    _PROFILES: dict[str, dict] = {
        "dev": {
            "l1_cache_ttl_seconds": 5,
            "l2_redis_ttl_seconds": 30,
            "max_active_flags": 500,
            "evaluation_log_sample_rate": 1.0,
            "kill_switch_enabled": True,
            "stale_detection_days": 7,
            "audit_log_retention_days": 90,
            "ab_testing_enabled": True,
            "pubsub_enabled": False,
        },
        "staging": {
            "l1_cache_ttl_seconds": 15,
            "l2_redis_ttl_seconds": 120,
            "max_active_flags": 750,
            "evaluation_log_sample_rate": 0.1,
            "kill_switch_enabled": True,
            "stale_detection_days": 14,
            "audit_log_retention_days": 180,
            "ab_testing_enabled": True,
            "pubsub_enabled": True,
        },
        "prod": {
            "l1_cache_ttl_seconds": 30,
            "l2_redis_ttl_seconds": 300,
            "max_active_flags": 1000,
            "evaluation_log_sample_rate": 0.01,
            "kill_switch_enabled": True,
            "stale_detection_days": 30,
            "audit_log_retention_days": 365,
            "ab_testing_enabled": True,
            "pubsub_enabled": True,
        },
    }

    @classmethod
    def get_defaults(cls, environment: str) -> dict:
        """Return the default configuration dict for the given environment.

        Args:
            environment: One of 'dev', 'staging', or 'prod'.

        Returns:
            Dict of default configuration values for the environment.
            Falls back to 'dev' defaults for unknown environments.
        """
        return dict(cls._PROFILES.get(environment, cls._PROFILES["dev"]))


# ---------------------------------------------------------------------------
# Main Configuration Model
# ---------------------------------------------------------------------------


class FeatureFlagConfig(BaseSettings):
    """Configuration for the GreenLang feature flag system.

    All fields can be set via environment variables with the GL_FF_ prefix.
    For example, ``GL_FF_REDIS_URL`` sets ``redis_url``.

    The ``environment`` field is loaded from ``GL_ENVIRONMENT`` (no prefix)
    and determines which EnvironmentProfile defaults apply. Explicit
    environment variable overrides always take precedence over profile defaults.

    Attributes:
        environment: Deployment environment (dev, staging, prod).
        redis_url: Redis connection URL for L2 cache and pub/sub.
        postgres_url: PostgreSQL connection URL for persistent flag storage.
        l1_cache_ttl_seconds: In-process (L1) cache time-to-live in seconds.
        l2_redis_ttl_seconds: Redis (L2) cache time-to-live in seconds.
        max_active_flags: Maximum number of concurrently active flags.
        evaluation_log_sample_rate: Fraction of evaluations to log (0.0-1.0).
        kill_switch_enabled: Whether the global kill switch mechanism is active.
        stale_detection_days: Days of inactivity before a flag is flagged as stale.
        audit_log_retention_days: Days to retain audit log entries.
        ab_testing_enabled: Whether A/B testing / experimentation is enabled.
        pubsub_enabled: Whether Redis pub/sub for flag updates is enabled.
        pubsub_channel: Redis pub/sub channel for flag update broadcasts.
        killswitch_channel: Redis pub/sub channel for kill switch signals.
    """

    model_config = {
        "env_prefix": "GL_FF_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "validate_default": True,
    }

    # -- Environment --------------------------------------------------------

    environment: str = Field(
        default="dev",
        description="Deployment environment. Loaded from GL_ENVIRONMENT.",
    )

    # -- Connection URLs ----------------------------------------------------

    redis_url: str = Field(
        default="redis://localhost:6379/2",
        description="Redis connection URL for L2 cache and pub/sub.",
    )
    postgres_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/greenlang",
        description="PostgreSQL connection URL for persistent flag storage.",
    )

    # -- Cache Configuration ------------------------------------------------

    l1_cache_ttl_seconds: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="In-process L1 cache TTL in seconds.",
    )
    l2_redis_ttl_seconds: int = Field(
        default=300,
        ge=1,
        le=86400,
        description="Redis L2 cache TTL in seconds.",
    )

    # -- Operational Limits -------------------------------------------------

    max_active_flags: int = Field(
        default=1000,
        ge=1,
        le=50000,
        description="Maximum number of concurrently active feature flags.",
    )

    # -- Evaluation Logging -------------------------------------------------

    evaluation_log_sample_rate: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Fraction of flag evaluations to log (0.0 = none, 1.0 = all).",
    )

    # -- Kill Switch --------------------------------------------------------

    kill_switch_enabled: bool = Field(
        default=True,
        description="Whether the global kill switch mechanism is active.",
    )

    # -- Stale Flag Detection -----------------------------------------------

    stale_detection_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days of inactivity before a flag is considered stale.",
    )

    # -- Audit Log ----------------------------------------------------------

    audit_log_retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Days to retain audit log entries for regulatory compliance.",
    )

    # -- A/B Testing --------------------------------------------------------

    ab_testing_enabled: bool = Field(
        default=True,
        description="Whether A/B testing / experimentation features are enabled.",
    )

    # -- Pub/Sub ------------------------------------------------------------

    pubsub_enabled: bool = Field(
        default=True,
        description="Whether Redis pub/sub for real-time flag updates is enabled.",
    )
    pubsub_channel: str = Field(
        default="ff:updates",
        min_length=1,
        max_length=256,
        description="Redis pub/sub channel for flag update broadcasts.",
    )
    killswitch_channel: str = Field(
        default="ff:killswitch",
        min_length=1,
        max_length=256,
        description="Redis pub/sub channel for kill switch signals.",
    )

    # -- Validators ---------------------------------------------------------

    @field_validator("environment", mode="before")
    @classmethod
    def resolve_environment(cls, v: Optional[str]) -> str:
        """Resolve environment from environment variables.

        Resolution order:
        1. GL_ENVIRONMENT (project-wide env var, highest priority)
        2. GL_FF_ENVIRONMENT (pydantic-settings prefix, via the ``v`` param)
        3. Falls back to 'dev' if neither is set.

        GL_ENVIRONMENT takes priority because it is the canonical project-wide
        environment indicator, while GL_FF_ENVIRONMENT is the feature-flag-specific
        override. This ensures that setting GL_ENVIRONMENT=prod correctly
        propagates even when GL_FF_ENVIRONMENT is not explicitly set.
        """
        # Priority 1: GL_ENVIRONMENT (project-wide)
        gl_env = os.environ.get("GL_ENVIRONMENT", "").strip().lower()
        if gl_env:
            return gl_env

        # Priority 2: GL_FF_ENVIRONMENT (via pydantic-settings -> v)
        gl_ff_env = os.environ.get("GL_FF_ENVIRONMENT", "").strip().lower()
        if gl_ff_env:
            return gl_ff_env

        # Priority 3: Explicit constructor argument (non-default)
        if v is not None and str(v).strip():
            return str(v).strip().lower()

        return "dev"

    @field_validator("environment")
    @classmethod
    def validate_environment_name(cls, v: str) -> str:
        """Validate environment is one of the known deployment targets."""
        allowed = {"dev", "development", "staging", "prod", "production", "test"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"Invalid environment '{v}'. Allowed: {sorted(allowed)}"
            )
        # Normalize aliases
        if v_lower == "development":
            return "dev"
        if v_lower == "production":
            return "prod"
        return v_lower

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        """Basic validation that the Redis URL has a valid scheme."""
        v_stripped = v.strip()
        if not v_stripped.startswith(("redis://", "rediss://", "unix://")):
            raise ValueError(
                f"Invalid Redis URL scheme. Expected redis://, rediss://, "
                f"or unix://. Got: {v_stripped[:30]}..."
            )
        return v_stripped

    @field_validator("postgres_url")
    @classmethod
    def validate_postgres_url(cls, v: str) -> str:
        """Basic validation that the PostgreSQL URL has a valid scheme."""
        v_stripped = v.strip()
        valid_schemes = (
            "postgresql://",
            "postgresql+asyncpg://",
            "postgresql+psycopg://",
            "postgres://",
        )
        if not v_stripped.startswith(valid_schemes):
            raise ValueError(
                f"Invalid PostgreSQL URL scheme. Expected one of "
                f"{valid_schemes}. Got: {v_stripped[:30]}..."
            )
        return v_stripped

    @field_validator("evaluation_log_sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: float) -> float:
        """Ensure sample rate is a valid probability."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"evaluation_log_sample_rate must be between 0.0 and 1.0. Got: {v}"
            )
        return v

    # -- Model Validator ----------------------------------------------------

    @model_validator(mode="after")
    def apply_environment_defaults(self) -> "FeatureFlagConfig":
        """Apply environment profile defaults for fields not explicitly set.

        This validator checks which fields were explicitly provided via
        environment variables. For fields that were not explicitly set,
        it applies the defaults from the EnvironmentProfile for the
        detected environment.

        Note: Pydantic Settings makes it non-trivial to distinguish
        between "default" and "explicitly set via env var". This validator
        applies environment-specific defaults only when the current value
        matches the Pydantic field default, meaning an explicit env var
        that happens to match the default will be treated as "not overridden".
        This is an acceptable trade-off for simplicity.
        """
        profile = EnvironmentProfile.get_defaults(self.environment)
        field_defaults = {
            "l1_cache_ttl_seconds": 30,
            "l2_redis_ttl_seconds": 300,
            "max_active_flags": 1000,
            "evaluation_log_sample_rate": 0.01,
            "kill_switch_enabled": True,
            "stale_detection_days": 30,
            "audit_log_retention_days": 365,
            "ab_testing_enabled": True,
            "pubsub_enabled": True,
        }

        for field_name, pydantic_default in field_defaults.items():
            current_value = getattr(self, field_name)
            profile_value = profile.get(field_name)
            if current_value == pydantic_default and profile_value is not None:
                object.__setattr__(self, field_name, profile_value)

        return self


# ---------------------------------------------------------------------------
# Convenience Enum (mirrors EnvironmentProfile keys)
# ---------------------------------------------------------------------------


class EnvironmentConfig(str, Enum):
    """Quick-reference enum mapping environment names to profile keys.

    Use with EnvironmentProfile.get_defaults() to retrieve pre-built
    configuration dictionaries.

    Example:
        >>> defaults = EnvironmentProfile.get_defaults(EnvironmentConfig.PROD.value)
        >>> defaults["evaluation_log_sample_rate"]
        0.01
    """

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


_config_instance: Optional[FeatureFlagConfig] = None


def get_config(force_reload: bool = False) -> FeatureFlagConfig:
    """Load feature flag configuration from environment variables.

    Creates a singleton FeatureFlagConfig instance on first call,
    reading from environment variables with the GL_FF_ prefix.
    Subsequent calls return the cached instance unless force_reload is True.

    Environment detection:
        1. GL_FF_ENVIRONMENT (via pydantic-settings prefix)
        2. GL_ENVIRONMENT (project-wide environment variable)
        3. Falls back to "dev" if neither is set.

    Args:
        force_reload: If True, discard the cached config and reload
            from environment variables.

    Returns:
        The FeatureFlagConfig singleton.

    Example:
        >>> import os
        >>> os.environ["GL_ENVIRONMENT"] = "prod"
        >>> os.environ["GL_FF_REDIS_URL"] = "redis://redis.prod:6379/2"
        >>> config = get_config(force_reload=True)
        >>> config.environment
        'prod'
        >>> config.redis_url
        'redis://redis.prod:6379/2'
    """
    global _config_instance

    if _config_instance is not None and not force_reload:
        return _config_instance

    _config_instance = FeatureFlagConfig()

    logger.info(
        "Feature flag config loaded: environment=%s, "
        "l1_ttl=%ds, l2_ttl=%ds, max_flags=%d, "
        "sample_rate=%.4f, pubsub=%s",
        _config_instance.environment,
        _config_instance.l1_cache_ttl_seconds,
        _config_instance.l2_redis_ttl_seconds,
        _config_instance.max_active_flags,
        _config_instance.evaluation_log_sample_rate,
        _config_instance.pubsub_enabled,
    )

    return _config_instance


def reset_config() -> None:
    """Reset the cached configuration. Used for testing.

    Clears the module-level config singleton so that the next call
    to :func:`get_config` reloads from environment variables.
    """
    global _config_instance
    _config_instance = None
    logger.debug("Feature flag config singleton reset.")
