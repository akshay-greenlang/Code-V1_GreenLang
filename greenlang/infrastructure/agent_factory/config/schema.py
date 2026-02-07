"""
Agent Config Schema - Agent Factory Config (INFRA-010)

Pydantic v2 model for agent configuration with comprehensive validation,
environment-specific defaults, and schema migration support. Each agent
in the GreenLang pipeline is configured via this schema.

Classes:
    - RetryConfigSchema: Nested retry configuration.
    - CircuitBreakerConfigSchema: Nested circuit breaker configuration.
    - ResourceLimitsSchema: Nested resource limits.
    - AgentConfigSchema: Top-level agent configuration model.

Example:
    >>> config = AgentConfigSchema(
    ...     agent_key="intake-agent",
    ...     version=1,
    ...     timeout_seconds=60.0,
    ... )
    >>> validated = config.model_dump()
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema Version
# ---------------------------------------------------------------------------

CURRENT_SCHEMA_VERSION: int = 1
"""Current schema version for migration detection."""


# ---------------------------------------------------------------------------
# Nested Configuration Models
# ---------------------------------------------------------------------------


class RetryConfigSchema(BaseModel):
    """Nested retry configuration within an agent config.

    Attributes:
        max_attempts: Maximum retry attempts (including initial call).
        base_delay_s: Base delay between retries in seconds.
        max_delay_s: Maximum delay cap.
        backoff_multiplier: Exponential backoff multiplier.
        jitter_range_s: Maximum random jitter in seconds.
    """

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=3, ge=1, le=10)
    base_delay_s: float = Field(default=1.0, ge=0.1, le=60.0)
    max_delay_s: float = Field(default=30.0, ge=1.0, le=300.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    jitter_range_s: float = Field(default=0.5, ge=0.0, le=5.0)


class CircuitBreakerConfigSchema(BaseModel):
    """Nested circuit breaker configuration within an agent config.

    Attributes:
        failure_rate_threshold: Fraction of failures to trip (0.0-1.0).
        slow_call_threshold_s: Slow call threshold in seconds.
        wait_in_open_s: Wait time before half-open transition.
        half_open_test_requests: Test requests in half-open state.
        sliding_window_size_s: Sliding window duration in seconds.
    """

    model_config = ConfigDict(extra="forbid")

    failure_rate_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    slow_call_threshold_s: float = Field(default=5.0, ge=0.1, le=120.0)
    wait_in_open_s: float = Field(default=60.0, ge=1.0, le=600.0)
    half_open_test_requests: int = Field(default=3, ge=1, le=20)
    sliding_window_size_s: float = Field(default=60.0, ge=5.0, le=600.0)


class ResourceLimitsSchema(BaseModel):
    """Nested resource limits within an agent config.

    Attributes:
        cpu_limit_cores: Maximum CPU cores.
        memory_limit_mb: Maximum memory in MB.
        max_execution_seconds: Maximum execution time.
        max_concurrent: Maximum concurrent executions (bulkhead).
    """

    model_config = ConfigDict(extra="forbid")

    cpu_limit_cores: float = Field(default=4.0, ge=0.1, le=16.0)
    memory_limit_mb: int = Field(default=2048, ge=128, le=16384)
    max_execution_seconds: float = Field(default=600.0, ge=1.0, le=3600.0)
    max_concurrent: int = Field(default=50, ge=1, le=1000)


# ---------------------------------------------------------------------------
# Environment Defaults
# ---------------------------------------------------------------------------

_ENV_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "dev": {
        "timeout_seconds": 120.0,
        "log_level": "DEBUG",
        "resource_limits": {
            "cpu_limit_cores": 2.0,
            "memory_limit_mb": 1024,
            "max_execution_seconds": 300.0,
            "max_concurrent": 10,
        },
        "retry_config": {
            "max_attempts": 2,
            "base_delay_s": 0.5,
        },
        "circuit_breaker_config": {
            "failure_rate_threshold": 0.7,
            "wait_in_open_s": 10.0,
        },
    },
    "staging": {
        "timeout_seconds": 90.0,
        "log_level": "INFO",
        "resource_limits": {
            "cpu_limit_cores": 4.0,
            "memory_limit_mb": 2048,
            "max_execution_seconds": 600.0,
            "max_concurrent": 25,
        },
    },
    "prod": {
        "timeout_seconds": 60.0,
        "log_level": "WARNING",
        "resource_limits": {
            "cpu_limit_cores": 4.0,
            "memory_limit_mb": 2048,
            "max_execution_seconds": 600.0,
            "max_concurrent": 50,
        },
    },
}


# ---------------------------------------------------------------------------
# Agent Config Schema
# ---------------------------------------------------------------------------


class AgentConfigSchema(BaseModel):
    """Top-level configuration schema for a GreenLang agent.

    Defines all configurable parameters for an agent, including resource
    limits, resilience policies, feature flags, and custom settings.
    Supports versioning for safe schema migrations.

    Attributes:
        agent_key: Unique identifier for the agent.
        version: Configuration version for optimistic concurrency.
        schema_version: Schema version for migration detection.
        enabled: Whether the agent is enabled for execution.
        environment: Deployment environment (dev, staging, prod).
        timeout_seconds: Default execution timeout.
        log_level: Logging level for the agent.
        resource_limits: Resource limits configuration.
        retry_config: Retry policy configuration.
        circuit_breaker_config: Circuit breaker configuration.
        feature_flags: Feature flag overrides for this agent.
        custom_settings: Arbitrary agent-specific settings.
        tags: Labels for filtering and grouping.
        owner: Team or individual responsible for this agent.
        description: Human-readable description of the agent.
        updated_at: Last modification timestamp.
        updated_by: Identity of who last modified the config.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    agent_key: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique agent identifier.",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Configuration version for optimistic concurrency.",
    )
    schema_version: int = Field(
        default=CURRENT_SCHEMA_VERSION,
        ge=1,
        description="Schema version for migration detection.",
    )
    enabled: bool = Field(
        default=True,
        description="Whether the agent is enabled for execution.",
    )
    environment: str = Field(
        default="dev",
        description="Deployment environment.",
    )
    timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="Default execution timeout in seconds.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    resource_limits: ResourceLimitsSchema = Field(
        default_factory=ResourceLimitsSchema,
        description="Resource limits configuration.",
    )
    retry_config: RetryConfigSchema = Field(
        default_factory=RetryConfigSchema,
        description="Retry policy configuration.",
    )
    circuit_breaker_config: CircuitBreakerConfigSchema = Field(
        default_factory=CircuitBreakerConfigSchema,
        description="Circuit breaker configuration.",
    )
    feature_flags: Dict[str, bool] = Field(
        default_factory=dict,
        description="Feature flag overrides for this agent.",
    )
    custom_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary agent-specific settings.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Labels for filtering and grouping.",
    )
    owner: str = Field(
        default="",
        max_length=256,
        description="Team or individual responsible.",
    )
    description: str = Field(
        default="",
        max_length=2048,
        description="Human-readable description.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last modification timestamp (UTC).",
    )
    updated_by: str = Field(
        default="",
        max_length=256,
        description="Identity of the last modifier.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("agent_key")
    @classmethod
    def validate_agent_key(cls, v: str) -> str:
        """Ensure agent_key follows naming convention."""
        import re
        if not re.match(r"^[a-z][a-z0-9._-]{0,127}$", v):
            raise ValueError(
                f"agent_key '{v}' must be lowercase alphanumeric "
                f"with dots, hyphens, or underscores."
            )
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Normalise and validate environment name."""
        v_lower = v.strip().lower()
        allowed = {"dev", "development", "staging", "prod", "production", "test"}
        if v_lower not in allowed:
            raise ValueError(
                f"Invalid environment '{v}'. Allowed: {sorted(allowed)}"
            )
        if v_lower == "development":
            return "dev"
        if v_lower == "production":
            return "prod"
        return v_lower

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a standard Python logging level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.strip().upper()
        if v_upper not in allowed:
            raise ValueError(
                f"Invalid log_level '{v}'. Allowed: {sorted(allowed)}"
            )
        return v_upper

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Strip, lowercase, and deduplicate tags."""
        seen: set[str] = set()
        result: List[str] = []
        for tag in v:
            cleaned = tag.strip().lower()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result

    # ------------------------------------------------------------------
    # Schema Migration
    # ------------------------------------------------------------------

    @classmethod
    def migrate(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration data from older schema versions.

        Applies sequential migrations from the source version to the
        current schema version.

        Args:
            data: Raw configuration data.

        Returns:
            Migrated configuration data.
        """
        source_version = data.get("schema_version", 1)

        if source_version >= CURRENT_SCHEMA_VERSION:
            return data

        migrated = dict(data)

        # Version 0 -> 1: add schema_version and resource_limits
        if source_version < 1:
            migrated.setdefault("schema_version", 1)
            migrated.setdefault("resource_limits", {})
            migrated.setdefault("retry_config", {})
            migrated.setdefault("circuit_breaker_config", {})
            logger.info(
                "Migrated config for '%s' from v%d to v1",
                migrated.get("agent_key", "unknown"), source_version,
            )

        migrated["schema_version"] = CURRENT_SCHEMA_VERSION
        return migrated

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def with_environment_defaults(
        cls,
        agent_key: str,
        environment: Optional[str] = None,
        **overrides: Any,
    ) -> AgentConfigSchema:
        """Create a config with environment-specific defaults.

        Args:
            agent_key: Agent identifier.
            environment: Target environment. Auto-detected if None.
            **overrides: Field overrides.

        Returns:
            AgentConfigSchema with appropriate defaults.
        """
        env = environment or os.environ.get("GL_ENVIRONMENT", "dev").lower()
        defaults = _ENV_DEFAULTS.get(env, _ENV_DEFAULTS["dev"])

        # Merge overrides onto defaults
        merged = {**defaults, **overrides}
        merged["agent_key"] = agent_key
        merged["environment"] = env

        # Handle nested model defaults
        if "resource_limits" in defaults and "resource_limits" not in overrides:
            merged["resource_limits"] = defaults["resource_limits"]
        if "retry_config" in defaults and "retry_config" not in overrides:
            merged["retry_config"] = defaults["retry_config"]
        if "circuit_breaker_config" in defaults and "circuit_breaker_config" not in overrides:
            merged["circuit_breaker_config"] = defaults["circuit_breaker_config"]

        return cls(**merged)


__all__ = [
    "AgentConfigSchema",
    "CircuitBreakerConfigSchema",
    "CURRENT_SCHEMA_VERSION",
    "ResourceLimitsSchema",
    "RetryConfigSchema",
]
