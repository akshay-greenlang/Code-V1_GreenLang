# -*- coding: utf-8 -*-
"""
Threat Modeling Configuration - SEC-010 Phase 2

Configuration management for the GreenLang threat modeling system using Pydantic Settings.
Loads configuration from environment variables with the GL_TM_ prefix, with sensible
defaults for each deployment environment (dev, staging, prod).

Classes:
    - ThreatModelingConfig: Main configuration model with all tunable parameters.

Functions:
    - get_config: Factory function that loads config from environment variables.
    - reset_config: Reset the cached configuration singleton.

Example:
    >>> from greenlang.infrastructure.threat_modeling.config import get_config
    >>> config = get_config()
    >>> config.risk_threshold_critical
    8.0
    >>> config.stride_weights
    {'S': 1.0, 'T': 1.2, 'R': 0.8, ...}

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Default Threat Patterns
# ---------------------------------------------------------------------------


@dataclass
class DefaultThreatPatterns:
    """Default threat patterns per component type.

    These patterns define common threats that apply to each type of
    system component. They are used by the STRIDE engine when generating
    threats during analysis.
    """

    # Web Application threats
    WEB_APP: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"category": "S", "title": "Session Hijacking", "description": "Attacker steals session tokens to impersonate users"},
        {"category": "S", "title": "Credential Stuffing", "description": "Automated attacks using leaked credentials"},
        {"category": "T", "title": "Cross-Site Scripting (XSS)", "description": "Injection of malicious scripts into web pages"},
        {"category": "T", "title": "Cross-Site Request Forgery (CSRF)", "description": "Unauthorized commands from trusted users"},
        {"category": "R", "title": "Insufficient Logging", "description": "Unable to prove user actions or detect attacks"},
        {"category": "I", "title": "Sensitive Data Exposure", "description": "PII or secrets exposed in responses or logs"},
        {"category": "D", "title": "Resource Exhaustion", "description": "DoS via memory, CPU, or connection exhaustion"},
        {"category": "E", "title": "Broken Access Control", "description": "Bypass of authorization checks"},
    ])

    # API threats
    API: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"category": "S", "title": "API Key Compromise", "description": "Attacker obtains valid API keys"},
        {"category": "S", "title": "JWT Token Forgery", "description": "Creation of fake or modified JWT tokens"},
        {"category": "T", "title": "Injection Attacks", "description": "SQL, NoSQL, or command injection via API parameters"},
        {"category": "T", "title": "Mass Assignment", "description": "Unauthorized modification of object properties"},
        {"category": "R", "title": "Missing Audit Trail", "description": "API calls not logged for compliance"},
        {"category": "I", "title": "Excessive Data Exposure", "description": "API returns more data than needed"},
        {"category": "D", "title": "Rate Limit Bypass", "description": "Circumventing rate limiting controls"},
        {"category": "E", "title": "BOLA/IDOR", "description": "Broken Object Level Authorization"},
    ])

    # Database threats
    DATABASE: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"category": "S", "title": "Database Credential Theft", "description": "Compromise of database connection credentials"},
        {"category": "T", "title": "SQL Injection", "description": "Malicious SQL via application inputs"},
        {"category": "T", "title": "Data Corruption", "description": "Unauthorized modification of records"},
        {"category": "R", "title": "Audit Log Tampering", "description": "Modification or deletion of audit logs"},
        {"category": "I", "title": "Unencrypted Data at Rest", "description": "Sensitive data stored without encryption"},
        {"category": "I", "title": "Backup Exposure", "description": "Database backups accessible without authorization"},
        {"category": "D", "title": "Query Denial of Service", "description": "Resource exhaustion via expensive queries"},
        {"category": "E", "title": "Privilege Escalation", "description": "Gaining DBA privileges from application account"},
    ])

    # Message Queue threats
    MESSAGE_QUEUE: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"category": "S", "title": "Message Spoofing", "description": "Injection of fake messages into queues"},
        {"category": "T", "title": "Message Tampering", "description": "Modification of messages in transit"},
        {"category": "R", "title": "Missing Message Acknowledgment", "description": "Unable to prove message delivery"},
        {"category": "I", "title": "Message Interception", "description": "Eavesdropping on queue contents"},
        {"category": "D", "title": "Queue Flooding", "description": "DoS via overwhelming message volume"},
        {"category": "E", "title": "Queue Administrative Access", "description": "Unauthorized management operations"},
    ])

    # External Service threats
    EXTERNAL_SERVICE: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"category": "S", "title": "API Credential Compromise", "description": "Third-party API keys stolen"},
        {"category": "T", "title": "Man-in-the-Middle", "description": "Interception and modification of API calls"},
        {"category": "R", "title": "Third-Party Audit Gaps", "description": "Unable to audit external service actions"},
        {"category": "I", "title": "Data Leakage to Third Party", "description": "Excessive data sent to external service"},
        {"category": "D", "title": "Dependency on External Availability", "description": "Service unavailable due to third-party outage"},
        {"category": "E", "title": "Overprivileged Integration", "description": "External service has excessive permissions"},
    ])

    # Cache/Redis threats
    CACHE: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"category": "S", "title": "Cache Poisoning", "description": "Injection of malicious data into cache"},
        {"category": "T", "title": "Cache Manipulation", "description": "Unauthorized modification of cached data"},
        {"category": "I", "title": "Sensitive Data in Cache", "description": "Secrets or PII stored in cache without encryption"},
        {"category": "D", "title": "Cache Eviction Attack", "description": "Forcing cache misses to overload backend"},
    ])

    # File Storage threats
    FILE_STORAGE: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"category": "S", "title": "Unauthorized Upload", "description": "Attacker uploads files without authentication"},
        {"category": "T", "title": "File Content Manipulation", "description": "Modification of stored files"},
        {"category": "T", "title": "Malware Upload", "description": "Upload of malicious executable files"},
        {"category": "R", "title": "Missing File Access Logs", "description": "Unable to audit file access"},
        {"category": "I", "title": "Insecure Direct Object Reference", "description": "Guessable URLs exposing private files"},
        {"category": "D", "title": "Storage Exhaustion", "description": "DoS via excessive uploads"},
        {"category": "E", "title": "Path Traversal", "description": "Accessing files outside intended directory"},
    ])

    # Load Balancer threats
    LOAD_BALANCER: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"category": "S", "title": "SSL Stripping", "description": "Downgrade attack removing TLS encryption"},
        {"category": "T", "title": "Header Injection", "description": "Manipulation of HTTP headers"},
        {"category": "I", "title": "Backend IP Exposure", "description": "Leaking internal server addresses"},
        {"category": "D", "title": "DDoS Attack", "description": "Distributed denial of service attack"},
    ])

    # Container/Kubernetes threats
    CONTAINER: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"category": "S", "title": "Container Image Tampering", "description": "Malicious modifications to container images"},
        {"category": "T", "title": "Container Escape", "description": "Breaking out of container isolation"},
        {"category": "I", "title": "Secret Exposure in Environment", "description": "Secrets visible in container environment"},
        {"category": "D", "title": "Resource Limit Bypass", "description": "Container consuming excessive resources"},
        {"category": "E", "title": "Privileged Container Abuse", "description": "Escalation via overprivileged containers"},
    ])


# ---------------------------------------------------------------------------
# Main Configuration Model
# ---------------------------------------------------------------------------


class ThreatModelingConfig(BaseSettings):
    """Configuration for the GreenLang threat modeling system.

    All fields can be set via environment variables with the GL_TM_ prefix.
    For example, ``GL_TM_RISK_THRESHOLD_CRITICAL`` sets ``risk_threshold_critical``.

    Attributes:
        environment: Deployment environment (dev, staging, prod).
        risk_threshold_critical: Score threshold for critical risk (>= this).
        risk_threshold_high: Score threshold for high risk.
        risk_threshold_medium: Score threshold for medium risk.
        risk_threshold_low: Score threshold for low risk (<= this).
        stride_weight_s: Weight for Spoofing threats.
        stride_weight_t: Weight for Tampering threats.
        stride_weight_r: Weight for Repudiation threats.
        stride_weight_i: Weight for Information Disclosure threats.
        stride_weight_d: Weight for Denial of Service threats.
        stride_weight_e: Weight for Elevation of Privilege threats.
        review_required_for_critical: Require review for critical threats.
        auto_generate_mitigations: Auto-suggest mitigations for threats.
        max_threats_per_component: Maximum threats per component.
        threat_model_retention_days: Days to retain threat model history.
        postgres_url: PostgreSQL connection URL for storage.
    """

    model_config = {
        "env_prefix": "GL_TM_",
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

    # -- Risk Score Thresholds ----------------------------------------------

    risk_threshold_critical: float = Field(
        default=8.0,
        ge=0.0,
        le=10.0,
        description="Risk score threshold for CRITICAL severity (>= this value).",
    )
    risk_threshold_high: float = Field(
        default=6.0,
        ge=0.0,
        le=10.0,
        description="Risk score threshold for HIGH severity (>= this, < critical).",
    )
    risk_threshold_medium: float = Field(
        default=4.0,
        ge=0.0,
        le=10.0,
        description="Risk score threshold for MEDIUM severity (>= this, < high).",
    )
    risk_threshold_low: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Risk score threshold for LOW severity (< medium).",
    )

    # -- STRIDE Category Weights --------------------------------------------

    stride_weight_s: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Weight multiplier for Spoofing threats.",
    )
    stride_weight_t: float = Field(
        default=1.2,
        ge=0.1,
        le=3.0,
        description="Weight multiplier for Tampering threats (higher due to data integrity).",
    )
    stride_weight_r: float = Field(
        default=0.8,
        ge=0.1,
        le=3.0,
        description="Weight multiplier for Repudiation threats.",
    )
    stride_weight_i: float = Field(
        default=1.5,
        ge=0.1,
        le=3.0,
        description="Weight multiplier for Information Disclosure threats (high for compliance).",
    )
    stride_weight_d: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Weight multiplier for Denial of Service threats.",
    )
    stride_weight_e: float = Field(
        default=1.3,
        ge=0.1,
        le=3.0,
        description="Weight multiplier for Elevation of Privilege threats.",
    )

    # -- Review Workflow Settings -------------------------------------------

    review_required_for_critical: bool = Field(
        default=True,
        description="Require security team review for critical risk threats.",
    )
    review_required_for_high: bool = Field(
        default=True,
        description="Require security team review for high risk threats.",
    )
    auto_generate_mitigations: bool = Field(
        default=True,
        description="Automatically suggest mitigations based on control mapping.",
    )
    approval_required_for_production: bool = Field(
        default=True,
        description="Require threat model approval before production deployment.",
    )

    # -- Operational Limits -------------------------------------------------

    max_threats_per_component: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of threats to generate per component.",
    )
    max_components_per_model: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum components in a single threat model.",
    )
    max_data_flows_per_model: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum data flows in a single threat model.",
    )

    # -- Retention & Storage ------------------------------------------------

    threat_model_retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Days to retain threat model history for audit.",
    )
    postgres_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/greenlang",
        description="PostgreSQL connection URL for threat model storage.",
    )

    # -- Report Settings ----------------------------------------------------

    include_mitigated_in_report: bool = Field(
        default=True,
        description="Include mitigated threats in generated reports.",
    )
    report_format: str = Field(
        default="json",
        description="Default report format: json, pdf, html.",
    )

    # -- Validators ---------------------------------------------------------

    @field_validator("environment", mode="before")
    @classmethod
    def resolve_environment(cls, v: Optional[str]) -> str:
        """Resolve environment from environment variables."""
        gl_env = os.environ.get("GL_ENVIRONMENT", "").strip().lower()
        if gl_env:
            return gl_env
        gl_tm_env = os.environ.get("GL_TM_ENVIRONMENT", "").strip().lower()
        if gl_tm_env:
            return gl_tm_env
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
            raise ValueError(f"Invalid environment '{v}'. Allowed: {sorted(allowed)}")
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
                f"Invalid PostgreSQL URL scheme. Expected one of {valid_schemes}. Got: {v_stripped[:30]}..."
            )
        return v_stripped

    @field_validator("report_format")
    @classmethod
    def validate_report_format(cls, v: str) -> str:
        """Validate report format is supported."""
        allowed = {"json", "pdf", "html", "markdown"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(f"Invalid report format '{v}'. Allowed: {sorted(allowed)}")
        return v_lower

    @model_validator(mode="after")
    def validate_thresholds(self) -> "ThreatModelingConfig":
        """Ensure risk thresholds are in correct order."""
        if not (
            self.risk_threshold_low
            <= self.risk_threshold_medium
            <= self.risk_threshold_high
            <= self.risk_threshold_critical
        ):
            raise ValueError(
                "Risk thresholds must be in order: low <= medium <= high <= critical"
            )
        return self

    # -- Convenience Properties ---------------------------------------------

    @property
    def stride_weights(self) -> Dict[str, float]:
        """Return STRIDE category weights as a dictionary."""
        return {
            "S": self.stride_weight_s,
            "T": self.stride_weight_t,
            "R": self.stride_weight_r,
            "I": self.stride_weight_i,
            "D": self.stride_weight_d,
            "E": self.stride_weight_e,
        }

    def get_severity_for_score(self, score: float) -> str:
        """Determine severity level for a given risk score.

        Args:
            score: Risk score (0.0-10.0).

        Returns:
            Severity string: critical, high, medium, or low.
        """
        if score >= self.risk_threshold_critical:
            return "critical"
        elif score >= self.risk_threshold_high:
            return "high"
        elif score >= self.risk_threshold_medium:
            return "medium"
        return "low"


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

_config_instance: Optional[ThreatModelingConfig] = None


def get_config(force_reload: bool = False) -> ThreatModelingConfig:
    """Load threat modeling configuration from environment variables.

    Creates a singleton ThreatModelingConfig instance on first call,
    reading from environment variables with the GL_TM_ prefix.
    Subsequent calls return the cached instance unless force_reload is True.

    Args:
        force_reload: If True, discard the cached config and reload
            from environment variables.

    Returns:
        The ThreatModelingConfig singleton.

    Example:
        >>> import os
        >>> os.environ["GL_ENVIRONMENT"] = "prod"
        >>> config = get_config(force_reload=True)
        >>> config.environment
        'prod'
        >>> config.risk_threshold_critical
        8.0
    """
    global _config_instance

    if _config_instance is not None and not force_reload:
        return _config_instance

    _config_instance = ThreatModelingConfig()

    logger.info(
        "Threat modeling config loaded: environment=%s, "
        "critical_threshold=%.1f, high_threshold=%.1f, "
        "auto_mitigations=%s, review_critical=%s",
        _config_instance.environment,
        _config_instance.risk_threshold_critical,
        _config_instance.risk_threshold_high,
        _config_instance.auto_generate_mitigations,
        _config_instance.review_required_for_critical,
    )

    return _config_instance


def reset_config() -> None:
    """Reset the cached configuration. Used for testing.

    Clears the module-level config singleton so that the next call
    to :func:`get_config` reloads from environment variables.
    """
    global _config_instance
    _config_instance = None
    logger.debug("Threat modeling config singleton reset.")


def get_default_threat_patterns() -> DefaultThreatPatterns:
    """Return the default threat patterns for component types.

    Returns:
        DefaultThreatPatterns instance with all predefined patterns.
    """
    return DefaultThreatPatterns()


__all__ = [
    "ThreatModelingConfig",
    "DefaultThreatPatterns",
    "get_config",
    "reset_config",
    "get_default_threat_patterns",
]
