# -*- coding: utf-8 -*-
"""
Compliance Automation Configuration - SEC-010 Phase 5

Configuration management for the GreenLang multi-compliance automation system
using Pydantic Settings. Loads configuration from environment variables with
the GL_COMPLIANCE_ prefix, with sensible defaults for each deployment
environment (dev, staging, prod).

Classes:
    - ComplianceConfig: Main configuration model with all tunable parameters.
    - EnvironmentProfile: Pre-built configuration profiles for dev/staging/prod.

Functions:
    - get_config: Factory function that loads config from environment variables.
    - reset_config: Reset the cached configuration (for testing).

Example:
    >>> from greenlang.infrastructure.compliance_automation.config import get_config
    >>> config = get_config()
    >>> config.dsar_sla_days
    30
    >>> config.evidence_collection_interval_hours
    24

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GDPR Article references
GDPR_ARTICLES = {
    "access": "Article 15 - Right of Access",
    "rectification": "Article 16 - Right to Rectification",
    "erasure": "Article 17 - Right to Erasure (Right to be Forgotten)",
    "restriction": "Article 18 - Right to Restriction of Processing",
    "portability": "Article 20 - Right to Data Portability",
    "objection": "Article 21 - Right to Object",
}

# Data retention period defaults (in days)
DEFAULT_RETENTION_PERIODS = {
    "operational": 90,  # 90 days
    "compliance": 365,  # 1 year
    "audit": 2555,  # 7 years (regulatory requirement)
    "financial": 2555,  # 7 years
    "pii": 365,  # 1 year after last activity
    "consent": 2555,  # 7 years (proof of consent)
    "security_logs": 365,  # 1 year
    "backup": 30,  # 30 days
}


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

    These profiles define the recommended defaults for compliance settings,
    collection intervals, and SLA thresholds. Values loaded from environment
    variables always override these defaults.
    """

    _PROFILES: Dict[str, Dict] = {
        "dev": {
            "dsar_sla_days": 30,
            "evidence_collection_interval_hours": 168,  # Weekly
            "compliance_assessment_interval_hours": 168,  # Weekly
            "retention_check_interval_hours": 24,
            "consent_audit_interval_hours": 168,
            "auto_evidence_collection": True,
            "strict_mode": False,
            "notification_enabled": False,
            "pii_scan_sample_rate": 0.1,  # 10% sampling in dev
            "max_parallel_dsar": 5,
        },
        "staging": {
            "dsar_sla_days": 30,
            "evidence_collection_interval_hours": 48,  # Every 2 days
            "compliance_assessment_interval_hours": 48,
            "retention_check_interval_hours": 12,
            "consent_audit_interval_hours": 48,
            "auto_evidence_collection": True,
            "strict_mode": True,
            "notification_enabled": True,
            "pii_scan_sample_rate": 0.5,  # 50% sampling in staging
            "max_parallel_dsar": 10,
        },
        "prod": {
            "dsar_sla_days": 30,  # GDPR requirement
            "evidence_collection_interval_hours": 24,  # Daily
            "compliance_assessment_interval_hours": 24,  # Daily
            "retention_check_interval_hours": 6,
            "consent_audit_interval_hours": 24,
            "auto_evidence_collection": True,
            "strict_mode": True,
            "notification_enabled": True,
            "pii_scan_sample_rate": 1.0,  # Full scan in prod
            "max_parallel_dsar": 20,
        },
    }

    @classmethod
    def get_defaults(cls, environment: str) -> Dict:
        """Return the default configuration dict for the given environment.

        Args:
            environment: One of 'dev', 'staging', or 'prod'.

        Returns:
            Dict of default configuration values for the environment.
            Falls back to 'dev' defaults for unknown environments.
        """
        return dict(cls._PROFILES.get(environment, cls._PROFILES["dev"]))


# ---------------------------------------------------------------------------
# Framework-Specific Settings
# ---------------------------------------------------------------------------


class FrameworkSettings:
    """Default settings for each compliance framework."""

    ISO27001 = {
        "enabled": True,
        "total_controls": 93,  # ISO 27001:2022 Annex A controls
        "domains": [
            "A.5",  # Organizational controls
            "A.6",  # People controls
            "A.7",  # Physical controls
            "A.8",  # Technological controls
        ],
        "assessment_frequency_days": 30,
        "evidence_retention_days": 365,
    }

    GDPR = {
        "enabled": True,
        "dsar_sla_days": 30,  # Art. 12(3) requirement
        "consent_granularity": "purpose",  # per-purpose consent
        "data_minimization": True,
        "right_to_portability_formats": ["json", "csv", "xml"],
    }

    PCI_DSS = {
        "enabled": True,
        "version": "4.0",
        "saq_type": "D",  # Merchant type
        "requirements": 12,  # 12 main requirements
        "encryption_algorithms": ["AES-256-GCM", "ChaCha20-Poly1305"],
        "key_rotation_days": 365,
    }

    CCPA = {
        "enabled": True,
        "response_time_days": 45,  # CCPA requirement
        "verification_required": True,
        "sale_opt_out": True,
    }

    LGPD = {
        "enabled": False,  # Enable when operating in Brazil
        "response_time_days": 15,  # LGPD requirement
    }


# ---------------------------------------------------------------------------
# Main Configuration Model
# ---------------------------------------------------------------------------


class ComplianceConfig(BaseSettings):
    """Configuration for the GreenLang multi-compliance automation system.

    All fields can be set via environment variables with the GL_COMPLIANCE_ prefix.
    For example, ``GL_COMPLIANCE_DSAR_SLA_DAYS`` sets ``dsar_sla_days``.

    Attributes:
        environment: Deployment environment (dev, staging, prod).
        dsar_sla_days: GDPR DSAR response SLA in days (default 30).
        evidence_collection_interval_hours: Hours between evidence collection runs.
        compliance_assessment_interval_hours: Hours between compliance assessments.
        retention_check_interval_hours: Hours between retention enforcement runs.
        consent_audit_interval_hours: Hours between consent audit runs.
        auto_evidence_collection: Whether to automatically collect evidence.
        strict_mode: Whether to enforce strict compliance checks.
        notification_enabled: Whether to send compliance notifications.
        pii_scan_sample_rate: Fraction of data to scan for PII (0.0-1.0).
        max_parallel_dsar: Maximum parallel DSAR processing tasks.
    """

    model_config = {
        "env_prefix": "GL_COMPLIANCE_",
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

    # -- Database URLs ------------------------------------------------------

    postgres_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/greenlang",
        description="PostgreSQL connection URL for compliance data.",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/3",
        description="Redis connection URL for caching and pub/sub.",
    )

    # -- DSAR Settings ------------------------------------------------------

    dsar_sla_days: int = Field(
        default=30,
        ge=1,
        le=90,
        description="GDPR DSAR response SLA in days (legally required: 30).",
    )
    dsar_extension_days: int = Field(
        default=60,
        ge=30,
        le=90,
        description="Extended DSAR deadline when complexity warrants (max 90 total).",
    )
    max_parallel_dsar: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of DSARs to process in parallel.",
    )

    # -- Evidence Collection ------------------------------------------------

    evidence_collection_interval_hours: int = Field(
        default=24,
        ge=1,
        le=720,  # Max 30 days
        description="Hours between automated evidence collection runs.",
    )
    auto_evidence_collection: bool = Field(
        default=True,
        description="Whether to automatically collect compliance evidence.",
    )

    # -- Compliance Assessment ----------------------------------------------

    compliance_assessment_interval_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Hours between compliance assessment runs.",
    )
    strict_mode: bool = Field(
        default=True,
        description="Whether to enforce strict compliance checks.",
    )

    # -- Retention Enforcement ----------------------------------------------

    retention_check_interval_hours: int = Field(
        default=6,
        ge=1,
        le=168,
        description="Hours between data retention enforcement runs.",
    )

    # -- Consent Management -------------------------------------------------

    consent_audit_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours between consent audit runs.",
    )

    # -- PII Scanning -------------------------------------------------------

    pii_scan_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of data to scan for PII (0.0=none, 1.0=all).",
    )

    # -- Notifications ------------------------------------------------------

    notification_enabled: bool = Field(
        default=True,
        description="Whether to send compliance notifications.",
    )
    notification_slack_webhook: Optional[str] = Field(
        default=None,
        description="Slack webhook URL for compliance notifications.",
    )
    notification_email_recipients: List[str] = Field(
        default_factory=lambda: ["compliance@greenlang.io"],
        description="Email recipients for compliance notifications.",
    )

    # -- Framework Enablement -----------------------------------------------

    iso27001_enabled: bool = Field(
        default=True,
        description="Whether ISO 27001 compliance is enabled.",
    )
    gdpr_enabled: bool = Field(
        default=True,
        description="Whether GDPR compliance is enabled.",
    )
    pci_dss_enabled: bool = Field(
        default=True,
        description="Whether PCI-DSS compliance is enabled.",
    )
    ccpa_enabled: bool = Field(
        default=True,
        description="Whether CCPA compliance is enabled.",
    )
    lgpd_enabled: bool = Field(
        default=False,
        description="Whether LGPD (Brazil) compliance is enabled.",
    )

    # -- Data Sources -------------------------------------------------------

    database_schemas: List[str] = Field(
        default_factory=lambda: [
            "public",
            "security",
            "security_ops",
            "greenlang",
        ],
        description="Database schemas to scan for PII.",
    )
    s3_buckets: List[str] = Field(
        default_factory=lambda: [
            "greenlang-raw-data",
            "greenlang-processed-data",
            "greenlang-exports",
        ],
        description="S3 buckets to scan for PII.",
    )
    log_sources: List[str] = Field(
        default_factory=lambda: [
            "application",
            "audit",
            "security",
        ],
        description="Log sources to scan for PII.",
    )

    # -- Retention Periods --------------------------------------------------

    retention_operational_days: int = Field(
        default=90,
        ge=30,
        le=365,
        description="Retention period for operational data.",
    )
    retention_compliance_days: int = Field(
        default=365,
        ge=180,
        le=3650,
        description="Retention period for compliance data.",
    )
    retention_audit_days: int = Field(
        default=2555,
        ge=365,
        le=3650,
        description="Retention period for audit data (7 years).",
    )
    retention_pii_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Retention period for PII after last activity.",
    )

    # -- Validators ---------------------------------------------------------

    @field_validator("environment", mode="before")
    @classmethod
    def resolve_environment(cls, v: Optional[str]) -> str:
        """Resolve environment from environment variables.

        Resolution order:
        1. GL_ENVIRONMENT (project-wide env var, highest priority)
        2. GL_COMPLIANCE_ENVIRONMENT (pydantic-settings prefix)
        3. Falls back to 'dev' if neither is set.
        """
        gl_env = os.environ.get("GL_ENVIRONMENT", "").strip().lower()
        if gl_env:
            return gl_env

        gl_compliance_env = os.environ.get(
            "GL_COMPLIANCE_ENVIRONMENT", ""
        ).strip().lower()
        if gl_compliance_env:
            return gl_compliance_env

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

    @model_validator(mode="after")
    def apply_environment_defaults(self) -> "ComplianceConfig":
        """Apply environment profile defaults for fields not explicitly set."""
        profile = EnvironmentProfile.get_defaults(self.environment)
        field_defaults = {
            "dsar_sla_days": 30,
            "evidence_collection_interval_hours": 24,
            "compliance_assessment_interval_hours": 24,
            "retention_check_interval_hours": 6,
            "consent_audit_interval_hours": 24,
            "auto_evidence_collection": True,
            "strict_mode": True,
            "notification_enabled": True,
            "pii_scan_sample_rate": 1.0,
            "max_parallel_dsar": 20,
        }

        for field_name, pydantic_default in field_defaults.items():
            current_value = getattr(self, field_name)
            profile_value = profile.get(field_name)
            if current_value == pydantic_default and profile_value is not None:
                object.__setattr__(self, field_name, profile_value)

        return self


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

_config_instance: Optional[ComplianceConfig] = None


def get_config(force_reload: bool = False) -> ComplianceConfig:
    """Load compliance configuration from environment variables.

    Creates a singleton ComplianceConfig instance on first call,
    reading from environment variables with the GL_COMPLIANCE_ prefix.
    Subsequent calls return the cached instance unless force_reload is True.

    Args:
        force_reload: If True, discard the cached config and reload
            from environment variables.

    Returns:
        The ComplianceConfig singleton.

    Example:
        >>> import os
        >>> os.environ["GL_ENVIRONMENT"] = "prod"
        >>> os.environ["GL_COMPLIANCE_DSAR_SLA_DAYS"] = "30"
        >>> config = get_config(force_reload=True)
        >>> config.environment
        'prod'
        >>> config.dsar_sla_days
        30
    """
    global _config_instance

    if _config_instance is not None and not force_reload:
        return _config_instance

    _config_instance = ComplianceConfig()

    logger.info(
        "Compliance config loaded: environment=%s, "
        "dsar_sla=%d days, evidence_interval=%d hours, "
        "strict_mode=%s, notifications=%s",
        _config_instance.environment,
        _config_instance.dsar_sla_days,
        _config_instance.evidence_collection_interval_hours,
        _config_instance.strict_mode,
        _config_instance.notification_enabled,
    )

    return _config_instance


def reset_config() -> None:
    """Reset the cached configuration. Used for testing.

    Clears the module-level config singleton so that the next call
    to :func:`get_config` reloads from environment variables.
    """
    global _config_instance
    _config_instance = None
    logger.debug("Compliance config singleton reset.")


__all__ = [
    "ComplianceConfig",
    "EnvironmentProfile",
    "EnvironmentName",
    "FrameworkSettings",
    "GDPR_ARTICLES",
    "DEFAULT_RETENTION_PERIODS",
    "get_config",
    "reset_config",
]
