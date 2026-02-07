# -*- coding: utf-8 -*-
"""
WAF Management Configuration - SEC-010

Configuration management for the GreenLang WAF and DDoS protection system
using Pydantic Settings. Loads configuration from environment variables
with the GL_WAF_ prefix.

Environment detection reads from the GL_ENVIRONMENT environment variable,
defaulting to "dev" if not set.

Classes:
    - WAFConfig: Main configuration model with all tunable parameters.
    - EnvironmentProfile: Pre-built configuration profiles for dev/staging/prod.

Functions:
    - get_config: Factory function that loads config from environment variables.

Example:
    # Set environment variables:
    #   GL_ENVIRONMENT=prod
    #   GL_WAF_WEB_ACL_ARN=arn:aws:wafv2:us-east-1:123456789012:regional/webacl/prod-waf/xxx
    #   GL_WAF_SHIELD_ENABLED=true

    >>> from greenlang.infrastructure.waf_management.config import get_config
    >>> config = get_config()
    >>> config.environment
    'prod'
    >>> config.rate_limit_threshold
    2000
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import List, Optional

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

    These profiles define the recommended defaults for rate limits,
    detection thresholds, and operational limits. Values loaded from
    environment variables always override these defaults.
    """

    _PROFILES: dict[str, dict] = {
        "dev": {
            "rate_limit_threshold": 10000,  # More permissive in dev
            "rate_limit_window_seconds": 300,
            "shield_enabled": False,  # No Shield in dev
            "auto_mitigation_enabled": False,  # Manual only
            "anomaly_detection_enabled": True,
            "volumetric_attack_multiplier": 5.0,  # Higher threshold
            "slowloris_connection_limit": 100,
            "credential_stuffing_threshold": 50,
            "bot_detection_enabled": True,
            "alert_on_attack": True,
            "auto_scale_on_attack": False,
            "blocked_countries": [],  # No geo-blocking in dev
            "waf_log_enabled": True,
            "waf_sample_rate": 1.0,  # Log all requests
        },
        "staging": {
            "rate_limit_threshold": 5000,
            "rate_limit_window_seconds": 300,
            "shield_enabled": False,  # Optional in staging
            "auto_mitigation_enabled": True,
            "anomaly_detection_enabled": True,
            "volumetric_attack_multiplier": 4.0,
            "slowloris_connection_limit": 50,
            "credential_stuffing_threshold": 30,
            "bot_detection_enabled": True,
            "alert_on_attack": True,
            "auto_scale_on_attack": True,
            "blocked_countries": [],
            "waf_log_enabled": True,
            "waf_sample_rate": 0.1,  # Sample 10%
        },
        "prod": {
            "rate_limit_threshold": 2000,  # PRD requirement: 2000/5min
            "rate_limit_window_seconds": 300,
            "shield_enabled": True,  # Shield Advanced in prod
            "auto_mitigation_enabled": True,
            "anomaly_detection_enabled": True,
            "volumetric_attack_multiplier": 3.0,  # 3x baseline = attack
            "slowloris_connection_limit": 20,
            "credential_stuffing_threshold": 10,
            "bot_detection_enabled": True,
            "alert_on_attack": True,
            "auto_scale_on_attack": True,
            "blocked_countries": [],  # Configure per customer
            "waf_log_enabled": True,
            "waf_sample_rate": 0.01,  # Sample 1%
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


class WAFConfig(BaseSettings):
    """Configuration for the GreenLang WAF and DDoS protection system.

    All fields can be set via environment variables with the GL_WAF_ prefix.
    For example, ``GL_WAF_WEB_ACL_ARN`` sets ``web_acl_arn``.

    Attributes:
        environment: Deployment environment (dev, staging, prod).
        aws_region: AWS region for WAF and Shield resources.
        web_acl_arn: ARN of the AWS WAF v2 Web ACL.
        web_acl_id: ID of the AWS WAF v2 Web ACL.
        shield_enabled: Whether AWS Shield Advanced is enabled.
        shield_resource_arns: List of resource ARNs to protect with Shield.
        rate_limit_threshold: Requests per window before blocking.
        rate_limit_window_seconds: Rate limit evaluation window.
        blocked_countries: List of ISO 3166-1 alpha-2 country codes to block.
        auto_mitigation_enabled: Whether automatic attack mitigation is enabled.
        anomaly_detection_enabled: Whether real-time anomaly detection is active.
        volumetric_attack_multiplier: Traffic multiplier to trigger attack detection.
        slowloris_connection_limit: Max slow connections before blocking.
        credential_stuffing_threshold: Failed logins per IP before blocking.
        bot_detection_enabled: Whether bot detection is active.
        alert_on_attack: Send alerts when attacks are detected.
        auto_scale_on_attack: Trigger auto-scaling during attacks.
        waf_log_enabled: Whether WAF request logging is enabled.
        waf_log_bucket: S3 bucket for WAF logs.
        waf_sample_rate: Fraction of requests to log (0.0-1.0).
        baseline_update_interval_minutes: How often to recalculate traffic baseline.
        attack_cooldown_minutes: Time after attack before normal operations resume.
    """

    model_config = {
        "env_prefix": "GL_WAF_",
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

    # -- AWS Configuration --------------------------------------------------

    aws_region: str = Field(
        default="us-east-1",
        min_length=5,
        max_length=25,
        description="AWS region for WAF and Shield resources.",
    )

    web_acl_arn: str = Field(
        default="",
        max_length=2048,
        description="ARN of the AWS WAF v2 Web ACL.",
    )

    web_acl_id: str = Field(
        default="",
        max_length=256,
        description="ID of the AWS WAF v2 Web ACL.",
    )

    web_acl_scope: str = Field(
        default="REGIONAL",
        description="WAF scope: REGIONAL (for ALB/API Gateway) or CLOUDFRONT.",
    )

    # -- Shield Advanced Configuration --------------------------------------

    shield_enabled: bool = Field(
        default=False,
        description="Whether AWS Shield Advanced is enabled.",
    )

    shield_resource_arns: List[str] = Field(
        default_factory=list,
        description="List of resource ARNs to protect with Shield Advanced.",
    )

    shield_proactive_engagement: bool = Field(
        default=False,
        description="Enable proactive engagement with AWS Shield Response Team.",
    )

    shield_auto_remediate: bool = Field(
        default=True,
        description="Enable automatic DDoS mitigation via Shield.",
    )

    # -- Rate Limiting Configuration ----------------------------------------

    rate_limit_threshold: int = Field(
        default=2000,
        ge=100,
        le=100000000,
        description="Requests per window before blocking (PRD: 2000/5min).",
    )

    rate_limit_window_seconds: int = Field(
        default=300,  # 5 minutes
        ge=60,
        le=3600,
        description="Rate limit evaluation window in seconds.",
    )

    rate_limit_action: str = Field(
        default="BLOCK",
        description="Action on rate limit: BLOCK, COUNT, CAPTCHA.",
    )

    # -- Geo-Blocking Configuration -----------------------------------------

    blocked_countries: List[str] = Field(
        default_factory=list,
        description="ISO 3166-1 alpha-2 country codes to block globally.",
    )

    geo_block_action: str = Field(
        default="BLOCK",
        description="Action on geo-block: BLOCK, COUNT.",
    )

    # -- Anomaly Detection Configuration ------------------------------------

    auto_mitigation_enabled: bool = Field(
        default=True,
        description="Automatically mitigate detected attacks.",
    )

    anomaly_detection_enabled: bool = Field(
        default=True,
        description="Enable real-time traffic anomaly detection.",
    )

    volumetric_attack_multiplier: float = Field(
        default=3.0,
        ge=1.5,
        le=10.0,
        description="Traffic multiplier vs baseline to trigger attack detection.",
    )

    slowloris_connection_limit: int = Field(
        default=20,
        ge=5,
        le=1000,
        description="Max slow/hanging connections per IP before blocking.",
    )

    credential_stuffing_threshold: int = Field(
        default=10,
        ge=3,
        le=100,
        description="Failed login attempts per IP in 5 minutes before blocking.",
    )

    credential_stuffing_window_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Window for credential stuffing detection.",
    )

    # -- Bot Detection Configuration ----------------------------------------

    bot_detection_enabled: bool = Field(
        default=True,
        description="Enable bot traffic detection and management.",
    )

    bot_detection_mode: str = Field(
        default="TARGETED",
        description="Bot detection mode: COMMON (known bots), TARGETED (aggressive).",
    )

    # -- Alert Configuration ------------------------------------------------

    alert_on_attack: bool = Field(
        default=True,
        description="Send alerts when attacks are detected.",
    )

    alert_sns_topic_arn: str = Field(
        default="",
        max_length=2048,
        description="SNS topic ARN for attack alerts.",
    )

    pagerduty_integration_key: str = Field(
        default="",
        max_length=256,
        description="PagerDuty integration key for critical alerts.",
    )

    slack_webhook_url: str = Field(
        default="",
        max_length=512,
        description="Slack webhook URL for attack notifications.",
    )

    # -- Auto-Scaling Configuration -----------------------------------------

    auto_scale_on_attack: bool = Field(
        default=True,
        description="Trigger auto-scaling during detected attacks.",
    )

    auto_scale_target_group_arn: str = Field(
        default="",
        max_length=2048,
        description="ALB target group ARN for auto-scaling.",
    )

    # -- Logging Configuration ----------------------------------------------

    waf_log_enabled: bool = Field(
        default=True,
        description="Enable WAF request logging.",
    )

    waf_log_bucket: str = Field(
        default="",
        max_length=256,
        description="S3 bucket for WAF logs (must have aws-waf-logs- prefix).",
    )

    waf_log_kinesis_arn: str = Field(
        default="",
        max_length=2048,
        description="Kinesis Firehose ARN for WAF log streaming.",
    )

    waf_sample_rate: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Fraction of requests to log (0.0-1.0).",
    )

    # -- Operational Configuration ------------------------------------------

    baseline_update_interval_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="How often to recalculate traffic baseline.",
    )

    attack_cooldown_minutes: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Time after attack ends before normal operations resume.",
    )

    rule_sync_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="How often to sync WAF rules with AWS.",
    )

    metrics_collection_interval_seconds: int = Field(
        default=10,
        ge=5,
        le=60,
        description="How often to collect WAF metrics.",
    )

    # -- IP Reputation Configuration ----------------------------------------

    ip_reputation_enabled: bool = Field(
        default=True,
        description="Enable IP reputation-based blocking.",
    )

    ip_reputation_sources: List[str] = Field(
        default_factory=lambda: ["aws_threat_intel", "project_honeypot"],
        description="IP reputation data sources.",
    )

    # -- Database Configuration ---------------------------------------------

    database_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/greenlang",
        description="PostgreSQL connection URL for WAF rule storage.",
    )

    redis_url: str = Field(
        default="redis://localhost:6379/3",
        description="Redis connection URL for real-time metrics cache.",
    )

    # -- Validators ---------------------------------------------------------

    @field_validator("environment", mode="before")
    @classmethod
    def resolve_environment(cls, v: Optional[str]) -> str:
        """Resolve environment from environment variables.

        Resolution order:
        1. GL_ENVIRONMENT (project-wide env var, highest priority)
        2. GL_WAF_ENVIRONMENT (pydantic-settings prefix)
        3. Falls back to 'dev' if neither is set.
        """
        gl_env = os.environ.get("GL_ENVIRONMENT", "").strip().lower()
        if gl_env:
            return gl_env

        gl_waf_env = os.environ.get("GL_WAF_ENVIRONMENT", "").strip().lower()
        if gl_waf_env:
            return gl_waf_env

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

    @field_validator("web_acl_scope")
    @classmethod
    def validate_web_acl_scope(cls, v: str) -> str:
        """Validate WAF scope is REGIONAL or CLOUDFRONT."""
        v_upper = v.strip().upper()
        if v_upper not in {"REGIONAL", "CLOUDFRONT"}:
            raise ValueError(
                f"Invalid web_acl_scope '{v}'. Must be REGIONAL or CLOUDFRONT."
            )
        return v_upper

    @field_validator("rate_limit_action", "geo_block_action")
    @classmethod
    def validate_waf_action(cls, v: str) -> str:
        """Validate WAF action is valid."""
        allowed = {"ALLOW", "BLOCK", "COUNT", "CAPTCHA", "CHALLENGE"}
        v_upper = v.strip().upper()
        if v_upper not in allowed:
            raise ValueError(
                f"Invalid action '{v}'. Allowed: {sorted(allowed)}"
            )
        return v_upper

    @field_validator("bot_detection_mode")
    @classmethod
    def validate_bot_detection_mode(cls, v: str) -> str:
        """Validate bot detection mode."""
        allowed = {"COMMON", "TARGETED"}
        v_upper = v.strip().upper()
        if v_upper not in allowed:
            raise ValueError(
                f"Invalid bot_detection_mode '{v}'. Allowed: {sorted(allowed)}"
            )
        return v_upper

    @field_validator("blocked_countries")
    @classmethod
    def validate_country_codes(cls, v: List[str]) -> List[str]:
        """Validate and normalize country codes to uppercase."""
        normalized: List[str] = []
        for code in v:
            if not code.strip():
                continue
            code_upper = code.strip().upper()
            if len(code_upper) != 2 or not code_upper.isalpha():
                raise ValueError(
                    f"Invalid country code '{code}'. Must be ISO 3166-1 alpha-2."
                )
            normalized.append(code_upper)
        return list(set(normalized))

    @field_validator("waf_sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: float) -> float:
        """Ensure sample rate is a valid probability."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"waf_sample_rate must be between 0.0 and 1.0. Got: {v}"
            )
        return v

    # -- Model Validator ----------------------------------------------------

    @model_validator(mode="after")
    def apply_environment_defaults(self) -> "WAFConfig":
        """Apply environment profile defaults for fields not explicitly set.

        This validator applies environment-specific defaults only when
        the current value matches the Pydantic field default.
        """
        profile = EnvironmentProfile.get_defaults(self.environment)
        field_defaults = {
            "rate_limit_threshold": 2000,
            "rate_limit_window_seconds": 300,
            "shield_enabled": False,
            "auto_mitigation_enabled": True,
            "anomaly_detection_enabled": True,
            "volumetric_attack_multiplier": 3.0,
            "slowloris_connection_limit": 20,
            "credential_stuffing_threshold": 10,
            "bot_detection_enabled": True,
            "alert_on_attack": True,
            "auto_scale_on_attack": True,
            "waf_log_enabled": True,
            "waf_sample_rate": 0.01,
        }

        for field_name, pydantic_default in field_defaults.items():
            current_value = getattr(self, field_name)
            profile_value = profile.get(field_name)
            if current_value == pydantic_default and profile_value is not None:
                object.__setattr__(self, field_name, profile_value)

        return self


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


_config_instance: Optional[WAFConfig] = None


def get_config(force_reload: bool = False) -> WAFConfig:
    """Load WAF configuration from environment variables.

    Creates a singleton WAFConfig instance on first call,
    reading from environment variables with the GL_WAF_ prefix.
    Subsequent calls return the cached instance unless force_reload is True.

    Args:
        force_reload: If True, discard the cached config and reload
            from environment variables.

    Returns:
        The WAFConfig singleton.

    Example:
        >>> import os
        >>> os.environ["GL_ENVIRONMENT"] = "prod"
        >>> os.environ["GL_WAF_SHIELD_ENABLED"] = "true"
        >>> config = get_config(force_reload=True)
        >>> config.environment
        'prod'
        >>> config.shield_enabled
        True
    """
    global _config_instance

    if _config_instance is not None and not force_reload:
        return _config_instance

    _config_instance = WAFConfig()

    logger.info(
        "WAF config loaded: environment=%s, rate_limit=%d/%ds, "
        "shield=%s, anomaly_detection=%s, auto_mitigate=%s",
        _config_instance.environment,
        _config_instance.rate_limit_threshold,
        _config_instance.rate_limit_window_seconds,
        _config_instance.shield_enabled,
        _config_instance.anomaly_detection_enabled,
        _config_instance.auto_mitigation_enabled,
    )

    return _config_instance


def reset_config() -> None:
    """Reset the cached configuration. Used for testing.

    Clears the module-level config singleton so that the next call
    to :func:`get_config` reloads from environment variables.
    """
    global _config_instance
    _config_instance = None
    logger.debug("WAF config singleton reset.")


__all__ = [
    "WAFConfig",
    "EnvironmentConfig",
    "EnvironmentName",
    "EnvironmentProfile",
    "get_config",
    "reset_config",
]
