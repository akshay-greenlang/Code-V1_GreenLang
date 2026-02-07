# -*- coding: utf-8 -*-
"""
Security Training Configuration - SEC-010

Configuration management for the GreenLang security training platform using
Pydantic Settings. Loads configuration from environment variables with the
GL_TRAINING_ prefix, with sensible defaults for each deployment environment.

Classes:
    - TrainingConfig: Main configuration model with all tunable parameters.
    - EnvironmentProfile: Pre-built configuration profiles for dev/staging/prod.

Functions:
    - get_config: Factory function that loads config from environment variables.
    - reset_config: Reset the cached configuration singleton.

Example:
    # Set environment variables:
    #   GL_ENVIRONMENT=prod
    #   GL_TRAINING_PASS_SCORE=80
    #   GL_TRAINING_CERTIFICATE_VALIDITY_DAYS=365

    >>> from greenlang.infrastructure.security_training.config import get_config
    >>> config = get_config()
    >>> config.pass_score
    80
    >>> config.certificate_validity_days
    365
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
# Environment Profiles
# ---------------------------------------------------------------------------


class EnvironmentProfile:
    """Pre-built configuration profiles for each deployment environment.

    These profiles define the recommended defaults for training settings,
    phishing campaigns, and compliance thresholds. Values loaded from
    environment variables always override these defaults.
    """

    _PROFILES: Dict[str, Dict] = {
        "dev": {
            "pass_score": 70,
            "certificate_validity_days": 30,
            "max_quiz_attempts": 10,
            "phishing_enabled": False,
            "phishing_cooldown_days": 1,
            "reminder_days_before_due": [1],
            "compliance_threshold": 0.70,
            "stale_training_days": 7,
        },
        "staging": {
            "pass_score": 75,
            "certificate_validity_days": 180,
            "max_quiz_attempts": 5,
            "phishing_enabled": True,
            "phishing_cooldown_days": 7,
            "reminder_days_before_due": [7, 3, 1],
            "compliance_threshold": 0.85,
            "stale_training_days": 30,
        },
        "prod": {
            "pass_score": 80,
            "certificate_validity_days": 365,
            "max_quiz_attempts": 3,
            "phishing_enabled": True,
            "phishing_cooldown_days": 30,
            "reminder_days_before_due": [14, 7, 3, 1],
            "compliance_threshold": 0.95,
            "stale_training_days": 30,
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
# Main Configuration Model
# ---------------------------------------------------------------------------


class TrainingConfig(BaseSettings):
    """Configuration for the GreenLang security training platform.

    All fields can be set via environment variables with the GL_TRAINING_ prefix.
    For example, ``GL_TRAINING_PASS_SCORE`` sets ``pass_score``.

    Attributes:
        environment: Deployment environment (dev, staging, prod).
        postgres_url: PostgreSQL connection URL for training data storage.
        redis_url: Redis connection URL for caching and session management.

        # Course Settings
        pass_score: Minimum passing score for assessments (0-100).
        certificate_validity_days: How long certificates are valid.
        max_quiz_attempts: Maximum quiz attempts per user per course.
        quiz_question_count: Number of questions in each quiz.
        quiz_time_limit_minutes: Time limit for quiz completion.

        # Phishing Campaign Settings
        phishing_enabled: Whether phishing simulations are enabled.
        phishing_cooldown_days: Days between campaigns for same user.
        phishing_auto_enroll_on_click: Auto-enroll clickers in training.
        phishing_sender_domain: Domain for phishing simulation emails.
        phishing_tracking_pixel_enabled: Enable open tracking pixels.

        # Reminder Settings
        reminder_days_before_due: Days before due date to send reminders.
        reminder_email_enabled: Whether to send email reminders.
        reminder_slack_enabled: Whether to send Slack reminders.

        # Compliance Settings
        compliance_threshold: Minimum compliance rate target (0.0-1.0).
        training_cycle_days: Days in a training cycle (e.g., annual).
        stale_training_days: Days until training is considered stale.

        # Security Score Weights
        score_weight_training: Weight for training completion (0.0-1.0).
        score_weight_phishing: Weight for phishing resistance (0.0-1.0).
        score_weight_mfa: Weight for MFA enablement (0.0-1.0).
        score_weight_password: Weight for password hygiene (0.0-1.0).
        score_weight_incidents: Weight for security incidents (0.0-1.0).
    """

    model_config = {
        "env_prefix": "GL_TRAINING_",
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
        description="PostgreSQL connection URL for training data storage.",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/4",
        description="Redis connection URL for caching and sessions.",
    )

    # -- Course Settings ----------------------------------------------------

    pass_score: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Minimum passing score for assessments (0-100).",
    )
    certificate_validity_days: int = Field(
        default=365,
        ge=1,
        le=1825,
        description="Certificate validity period in days.",
    )
    max_quiz_attempts: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Maximum quiz attempts per user per course.",
    )
    quiz_question_count: int = Field(
        default=10,
        ge=5,
        le=50,
        description="Number of questions in each quiz.",
    )
    quiz_time_limit_minutes: int = Field(
        default=30,
        ge=5,
        le=180,
        description="Time limit for quiz completion in minutes.",
    )

    # -- Phishing Campaign Settings -----------------------------------------

    phishing_enabled: bool = Field(
        default=True,
        description="Whether phishing simulations are enabled.",
    )
    phishing_cooldown_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days between phishing campaigns for the same user.",
    )
    phishing_auto_enroll_on_click: bool = Field(
        default=True,
        description="Auto-enroll users who click phishing links in training.",
    )
    phishing_sender_domain: str = Field(
        default="security-training.greenlang.io",
        min_length=3,
        max_length=255,
        description="Domain for phishing simulation emails.",
    )
    phishing_tracking_pixel_enabled: bool = Field(
        default=True,
        description="Enable tracking pixel for email open detection.",
    )
    phishing_click_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Target maximum click rate for phishing campaigns (0.0-1.0).",
    )

    # -- Reminder Settings --------------------------------------------------

    reminder_days_before_due: List[int] = Field(
        default_factory=lambda: [14, 7, 3, 1],
        description="Days before due date to send training reminders.",
    )
    reminder_email_enabled: bool = Field(
        default=True,
        description="Whether to send email reminders for overdue training.",
    )
    reminder_slack_enabled: bool = Field(
        default=True,
        description="Whether to send Slack reminders for overdue training.",
    )

    # -- Compliance Settings ------------------------------------------------

    compliance_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Target compliance rate for training completion (0.0-1.0).",
    )
    training_cycle_days: int = Field(
        default=365,
        ge=30,
        le=730,
        description="Days in a training cycle (e.g., 365 for annual).",
    )
    stale_training_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days until completed training is considered stale.",
    )

    # -- Security Score Weights ---------------------------------------------

    score_weight_training: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight for training completion in security score.",
    )
    score_weight_phishing: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for phishing resistance in security score.",
    )
    score_weight_mfa: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for MFA enablement in security score.",
    )
    score_weight_password: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight for password hygiene in security score.",
    )
    score_weight_incidents: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for security incident history in security score.",
    )

    # -- At-Risk Thresholds -------------------------------------------------

    at_risk_score_threshold: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Security score below which a user is considered at-risk.",
    )

    # -- Validators ---------------------------------------------------------

    @field_validator("environment", mode="before")
    @classmethod
    def resolve_environment(cls, v: Optional[str]) -> str:
        """Resolve environment from environment variables.

        Resolution order:
        1. GL_ENVIRONMENT (project-wide env var, highest priority)
        2. GL_TRAINING_ENVIRONMENT (via the ``v`` param)
        3. Falls back to 'dev' if neither is set.
        """
        gl_env = os.environ.get("GL_ENVIRONMENT", "").strip().lower()
        if gl_env:
            return gl_env

        gl_training_env = os.environ.get("GL_TRAINING_ENVIRONMENT", "").strip().lower()
        if gl_training_env:
            return gl_training_env

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
    def validate_score_weights_sum(self) -> "TrainingConfig":
        """Ensure security score weights sum to approximately 1.0."""
        total = (
            self.score_weight_training
            + self.score_weight_phishing
            + self.score_weight_mfa
            + self.score_weight_password
            + self.score_weight_incidents
        )
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "Security score weights sum to %.2f, expected 1.0. "
                "Scores may not be normalized correctly.",
                total,
            )
        return self

    @model_validator(mode="after")
    def apply_environment_defaults(self) -> "TrainingConfig":
        """Apply environment profile defaults for fields not explicitly set."""
        profile = EnvironmentProfile.get_defaults(self.environment)
        field_defaults = {
            "pass_score": 80,
            "certificate_validity_days": 365,
            "max_quiz_attempts": 3,
            "phishing_enabled": True,
            "phishing_cooldown_days": 30,
            "compliance_threshold": 0.95,
            "stale_training_days": 30,
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


_config_instance: Optional[TrainingConfig] = None


def get_config(force_reload: bool = False) -> TrainingConfig:
    """Load security training configuration from environment variables.

    Creates a singleton TrainingConfig instance on first call,
    reading from environment variables with the GL_TRAINING_ prefix.
    Subsequent calls return the cached instance unless force_reload is True.

    Args:
        force_reload: If True, discard the cached config and reload
            from environment variables.

    Returns:
        The TrainingConfig singleton.

    Example:
        >>> import os
        >>> os.environ["GL_ENVIRONMENT"] = "prod"
        >>> os.environ["GL_TRAINING_PASS_SCORE"] = "85"
        >>> config = get_config(force_reload=True)
        >>> config.environment
        'prod'
        >>> config.pass_score
        85
    """
    global _config_instance

    if _config_instance is not None and not force_reload:
        return _config_instance

    _config_instance = TrainingConfig()

    logger.info(
        "Security training config loaded: environment=%s, "
        "pass_score=%d, cert_validity=%dd, phishing=%s, "
        "compliance_threshold=%.2f",
        _config_instance.environment,
        _config_instance.pass_score,
        _config_instance.certificate_validity_days,
        _config_instance.phishing_enabled,
        _config_instance.compliance_threshold,
    )

    return _config_instance


def reset_config() -> None:
    """Reset the cached configuration. Used for testing.

    Clears the module-level config singleton so that the next call
    to :func:`get_config` reloads from environment variables.
    """
    global _config_instance
    _config_instance = None
    logger.debug("Security training config singleton reset.")


__all__ = [
    "EnvironmentProfile",
    "TrainingConfig",
    "get_config",
    "reset_config",
]
