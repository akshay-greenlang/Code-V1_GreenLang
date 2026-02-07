# -*- coding: utf-8 -*-
"""
SOC 2 Type II Preparation Configuration - SEC-009

Configuration management for the GreenLang SOC 2 Type II audit preparation platform.
Loads configuration from environment variables with the GL_SOC2_ prefix, with sensible
defaults for each deployment environment (dev, staging, prod).

Environment detection reads from the GL_ENVIRONMENT environment variable, defaulting
to "dev" if not set.

Classes:
    - SOC2Config: Main configuration model with all tunable parameters.
    - EnvironmentProfile: Pre-built configuration profiles for dev/staging/prod.

Functions:
    - get_config: Factory function that loads config from environment variables.

Example:
    # Set environment variables:
    #   GL_ENVIRONMENT=prod
    #   GL_SOC2_EVIDENCE_BUCKET=s3://greenlang-soc2-evidence-prod
    #   GL_SOC2_POSTGRES_URL=postgresql+asyncpg://user:pass@db:5432/greenlang

    >>> from greenlang.infrastructure.soc2_preparation.config import get_config
    >>> config = get_config()
    >>> config.environment
    'prod'
    >>> config.sla_critical_hours
    4

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
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

    These profiles define the recommended defaults for cache TTLs, SLA targets,
    and operational limits. Values loaded from environment variables always
    override these defaults.
    """

    _PROFILES: dict[str, dict] = {
        "dev": {
            "sla_critical_hours": 8,
            "sla_high_hours": 48,
            "sla_normal_hours": 96,
            "sla_low_hours": 168,
            "assessment_cache_ttl": 600,
            "evidence_retention_days": 90,
            "auto_refresh_interval_seconds": 300,
            "max_concurrent_assessments": 5,
            "enable_automated_evidence": True,
            "audit_log_retention_days": 90,
        },
        "staging": {
            "sla_critical_hours": 4,
            "sla_high_hours": 24,
            "sla_normal_hours": 48,
            "sla_low_hours": 72,
            "assessment_cache_ttl": 1800,
            "evidence_retention_days": 365,
            "auto_refresh_interval_seconds": 600,
            "max_concurrent_assessments": 10,
            "enable_automated_evidence": True,
            "audit_log_retention_days": 365,
        },
        "prod": {
            "sla_critical_hours": 4,
            "sla_high_hours": 24,
            "sla_normal_hours": 48,
            "sla_low_hours": 72,
            "assessment_cache_ttl": 3600,
            "evidence_retention_days": 2555,  # 7 years for compliance
            "auto_refresh_interval_seconds": 900,
            "max_concurrent_assessments": 20,
            "enable_automated_evidence": True,
            "audit_log_retention_days": 2555,  # 7 years
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


class SOC2Config(BaseSettings):
    """Configuration for the GreenLang SOC 2 Type II Preparation Platform.

    All fields can be set via environment variables with the GL_SOC2_ prefix.
    For example, ``GL_SOC2_EVIDENCE_BUCKET`` sets ``evidence_bucket``.

    The ``environment`` field is loaded from ``GL_ENVIRONMENT`` (no prefix)
    and determines which EnvironmentProfile defaults apply. Explicit
    environment variable overrides always take precedence over profile defaults.

    Attributes:
        environment: Deployment environment (dev, staging, prod).
        postgres_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching and pub/sub.
        evidence_bucket: S3 bucket URL for evidence storage.
        audit_period_start: Start date of the SOC 2 audit period.
        audit_period_end: End date of the SOC 2 audit period.
        sla_critical_hours: SLA for critical priority auditor requests.
        sla_high_hours: SLA for high priority auditor requests.
        sla_normal_hours: SLA for normal priority auditor requests.
        sla_low_hours: SLA for low priority auditor requests.
        assessment_cache_ttl: TTL for cached assessment results in seconds.
        evidence_retention_days: Days to retain evidence files.
        max_concurrent_assessments: Maximum concurrent assessment runs.
        enable_automated_evidence: Enable automated evidence collection.
        audit_log_retention_days: Days to retain audit log entries.
        auditor_portal_enabled: Enable the external auditor portal.
        notification_email: Email for compliance notifications.
    """

    model_config = {
        "env_prefix": "GL_SOC2_",
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

    postgres_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/greenlang",
        description="PostgreSQL connection URL for persistent storage.",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/3",
        description="Redis connection URL for caching and pub/sub.",
    )
    evidence_bucket: str = Field(
        default="s3://greenlang-soc2-evidence-dev",
        description="S3 bucket URL for evidence storage.",
    )

    # -- Audit Period -------------------------------------------------------

    audit_period_start: Optional[date] = Field(
        default=None,
        description="Start date of the SOC 2 audit period (YYYY-MM-DD).",
    )
    audit_period_end: Optional[date] = Field(
        default=None,
        description="End date of the SOC 2 audit period (YYYY-MM-DD).",
    )

    # -- SLA Configuration --------------------------------------------------

    sla_critical_hours: int = Field(
        default=4,
        ge=1,
        le=24,
        description="SLA for critical priority auditor requests in hours.",
    )
    sla_high_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="SLA for high priority auditor requests in hours.",
    )
    sla_normal_hours: int = Field(
        default=48,
        ge=1,
        le=336,
        description="SLA for normal priority auditor requests in hours.",
    )
    sla_low_hours: int = Field(
        default=72,
        ge=1,
        le=720,
        description="SLA for low priority auditor requests in hours.",
    )

    # -- Cache Configuration ------------------------------------------------

    assessment_cache_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="TTL for cached assessment results in seconds.",
    )

    # -- Retention Configuration --------------------------------------------

    evidence_retention_days: int = Field(
        default=2555,
        ge=30,
        le=3650,
        description="Days to retain evidence files (7 years for compliance).",
    )
    audit_log_retention_days: int = Field(
        default=2555,
        ge=90,
        le=3650,
        description="Days to retain audit log entries.",
    )

    # -- Operational Limits -------------------------------------------------

    max_concurrent_assessments: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent assessment runs.",
    )
    auto_refresh_interval_seconds: int = Field(
        default=900,
        ge=60,
        le=86400,
        description="Interval for auto-refreshing assessment data in seconds.",
    )
    enable_automated_evidence: bool = Field(
        default=True,
        description="Enable automated evidence collection.",
    )

    # -- Portal Configuration -----------------------------------------------

    auditor_portal_enabled: bool = Field(
        default=False,
        description="Enable the external auditor portal.",
    )
    notification_email: str = Field(
        default="",
        max_length=256,
        description="Email address for compliance notifications.",
    )

    # -- Trust Service Categories -------------------------------------------

    tsc_security_enabled: bool = Field(
        default=True,
        description="Enable Security (Common Criteria) trust service category.",
    )
    tsc_availability_enabled: bool = Field(
        default=False,
        description="Enable Availability trust service category.",
    )
    tsc_confidentiality_enabled: bool = Field(
        default=False,
        description="Enable Confidentiality trust service category.",
    )
    tsc_processing_integrity_enabled: bool = Field(
        default=False,
        description="Enable Processing Integrity trust service category.",
    )
    tsc_privacy_enabled: bool = Field(
        default=False,
        description="Enable Privacy trust service category.",
    )

    # -- Validators ---------------------------------------------------------

    @field_validator("environment", mode="before")
    @classmethod
    def resolve_environment(cls, v: Optional[str]) -> str:
        """Resolve environment from environment variables.

        Resolution order:
        1. GL_ENVIRONMENT (project-wide env var, highest priority)
        2. GL_SOC2_ENVIRONMENT (pydantic-settings prefix, via the ``v`` param)
        3. Falls back to 'dev' if neither is set.
        """
        # Priority 1: GL_ENVIRONMENT (project-wide)
        gl_env = os.environ.get("GL_ENVIRONMENT", "").strip().lower()
        if gl_env:
            return gl_env

        # Priority 2: GL_SOC2_ENVIRONMENT (via pydantic-settings -> v)
        gl_soc2_env = os.environ.get("GL_SOC2_ENVIRONMENT", "").strip().lower()
        if gl_soc2_env:
            return gl_soc2_env

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

    @field_validator("evidence_bucket")
    @classmethod
    def validate_evidence_bucket(cls, v: str) -> str:
        """Validate evidence bucket URL format."""
        v_stripped = v.strip()
        if not v_stripped.startswith(("s3://", "gs://", "az://")):
            raise ValueError(
                f"Invalid evidence bucket URL. Expected s3://, gs://, or az://. "
                f"Got: {v_stripped[:30]}..."
            )
        return v_stripped

    @field_validator("notification_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email validation."""
        v_stripped = v.strip()
        if v_stripped and "@" not in v_stripped:
            raise ValueError(f"Invalid email address: {v_stripped}")
        return v_stripped

    # -- Model Validators ---------------------------------------------------

    @model_validator(mode="after")
    def validate_audit_period(self) -> "SOC2Config":
        """Validate audit period dates are consistent."""
        if self.audit_period_start and self.audit_period_end:
            if self.audit_period_start >= self.audit_period_end:
                raise ValueError(
                    f"audit_period_start ({self.audit_period_start}) must be "
                    f"before audit_period_end ({self.audit_period_end})."
                )
        return self

    @model_validator(mode="after")
    def validate_sla_ordering(self) -> "SOC2Config":
        """Validate SLA hours are in ascending order by priority."""
        if not (
            self.sla_critical_hours
            <= self.sla_high_hours
            <= self.sla_normal_hours
            <= self.sla_low_hours
        ):
            raise ValueError(
                f"SLA hours must be in ascending order: "
                f"critical({self.sla_critical_hours}) <= "
                f"high({self.sla_high_hours}) <= "
                f"normal({self.sla_normal_hours}) <= "
                f"low({self.sla_low_hours})"
            )
        return self

    @model_validator(mode="after")
    def apply_environment_defaults(self) -> "SOC2Config":
        """Apply environment profile defaults for fields not explicitly set."""
        profile = EnvironmentProfile.get_defaults(self.environment)
        field_defaults = {
            "sla_critical_hours": 4,
            "sla_high_hours": 24,
            "sla_normal_hours": 48,
            "sla_low_hours": 72,
            "assessment_cache_ttl": 3600,
            "evidence_retention_days": 2555,
            "auto_refresh_interval_seconds": 900,
            "max_concurrent_assessments": 10,
            "enable_automated_evidence": True,
            "audit_log_retention_days": 2555,
        }

        for field_name, pydantic_default in field_defaults.items():
            current_value = getattr(self, field_name)
            profile_value = profile.get(field_name)
            if current_value == pydantic_default and profile_value is not None:
                object.__setattr__(self, field_name, profile_value)

        return self

    # -- Properties ---------------------------------------------------------

    @property
    def enabled_tsc_categories(self) -> list[str]:
        """Return list of enabled Trust Service Categories."""
        categories = []
        if self.tsc_security_enabled:
            categories.append("security")
        if self.tsc_availability_enabled:
            categories.append("availability")
        if self.tsc_confidentiality_enabled:
            categories.append("confidentiality")
        if self.tsc_processing_integrity_enabled:
            categories.append("processing_integrity")
        if self.tsc_privacy_enabled:
            categories.append("privacy")
        return categories


# ---------------------------------------------------------------------------
# Convenience Enum
# ---------------------------------------------------------------------------


class EnvironmentConfig(str, Enum):
    """Quick-reference enum mapping environment names to profile keys."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


_config_instance: Optional[SOC2Config] = None


def get_config(force_reload: bool = False) -> SOC2Config:
    """Load SOC 2 preparation configuration from environment variables.

    Creates a singleton SOC2Config instance on first call,
    reading from environment variables with the GL_SOC2_ prefix.
    Subsequent calls return the cached instance unless force_reload is True.

    Args:
        force_reload: If True, discard the cached config and reload
            from environment variables.

    Returns:
        The SOC2Config singleton.

    Example:
        >>> import os
        >>> os.environ["GL_ENVIRONMENT"] = "prod"
        >>> os.environ["GL_SOC2_EVIDENCE_BUCKET"] = "s3://greenlang-soc2-evidence"
        >>> config = get_config(force_reload=True)
        >>> config.environment
        'prod'
        >>> config.evidence_bucket
        's3://greenlang-soc2-evidence'
    """
    global _config_instance

    if _config_instance is not None and not force_reload:
        return _config_instance

    _config_instance = SOC2Config()

    logger.info(
        "SOC 2 config loaded: environment=%s, "
        "evidence_bucket=%s, sla_critical=%dh, "
        "cache_ttl=%ds, tsc_categories=%s",
        _config_instance.environment,
        _config_instance.evidence_bucket,
        _config_instance.sla_critical_hours,
        _config_instance.assessment_cache_ttl,
        _config_instance.enabled_tsc_categories,
    )

    return _config_instance


def reset_config() -> None:
    """Reset the cached configuration. Used for testing.

    Clears the module-level config singleton so that the next call
    to :func:`get_config` reloads from environment variables.
    """
    global _config_instance
    _config_instance = None
    logger.debug("SOC 2 config singleton reset.")


__all__ = [
    "SOC2Config",
    "EnvironmentConfig",
    "EnvironmentName",
    "EnvironmentProfile",
    "get_config",
    "reset_config",
]
