"""
Logging Configuration - INFRA-009

Pydantic Settings model for the GreenLang structured logging system. All fields
can be set via environment variables with the ``GL_LOG_`` prefix (e.g.
``GL_LOG_LEVEL=DEBUG``). The configuration follows a singleton pattern so that
only one instance exists per process, accessible via :func:`get_config`.

Classes:
    - LoggingConfig: Main configuration model with all tunable parameters.

Functions:
    - get_config: Factory function that returns the singleton config instance.
    - reset_config: Reset the singleton for testing.

Example:
    >>> import os
    >>> os.environ["GL_LOG_LEVEL"] = "DEBUG"
    >>> os.environ["GL_LOG_FORMAT"] = "console"
    >>> from greenlang.infrastructure.logging.config import get_config
    >>> config = get_config(force_reload=True)
    >>> config.level
    'DEBUG'
    >>> config.format
    'console'
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Log level mapping
# ---------------------------------------------------------------------------

_LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


# ---------------------------------------------------------------------------
# Main Configuration Model
# ---------------------------------------------------------------------------


class LoggingConfig(BaseSettings):
    """Configuration for the GreenLang structured logging system.

    All fields can be set via environment variables with the ``GL_LOG_``
    prefix. For example, ``GL_LOG_LEVEL`` sets ``level``, and
    ``GL_LOG_SERVICE_NAME`` sets ``service_name``.

    The ``environment`` field is resolved from ``GL_ENVIRONMENT`` first
    (project-wide convention) and falls back to ``GL_LOG_ENVIRONMENT``.

    Attributes:
        level: Log level threshold (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Output format -- ``json`` for production, ``console`` for
            development with colorized output.
        service_name: Name of the service emitting logs.
        service_version: Semantic version of the running service.
        environment: Deployment environment (development, staging, production).
        enable_redaction: Whether to redact PII/secrets from log output.
        redaction_patterns: Additional regex patterns for custom redaction.
        enable_correlation: Whether to propagate correlation / request IDs.
        correlation_header: HTTP header name for the request correlation ID.
        trace_header: HTTP header name for distributed trace IDs.
        enable_otel: Whether to enable the OpenTelemetry log bridge.
        log_file: Optional filesystem path to write logs to.
        max_message_length: Maximum length of the ``event`` field before
            truncation. Guards against accidentally logging large payloads.
        sample_rate: Fraction of DEBUG-level log messages to emit. 1.0 means
            log everything; 0.1 means sample 10 percent. Only applies to
            DEBUG level; higher levels are never sampled.
        tenant_isolation: Whether to enforce tenant_id in log context.
        async_logging: Whether to use a non-blocking queue handler for log
            I/O so that logging never blocks the event loop.
    """

    model_config = {
        "env_prefix": "GL_LOG_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "validate_default": True,
    }

    # -- Core ---------------------------------------------------------------

    level: str = Field(
        default="INFO",
        description="Log level threshold (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    format: Literal["json", "console"] = Field(
        default="json",
        description="Output format. 'json' for production, 'console' for local dev.",
    )

    # -- Service Metadata ---------------------------------------------------

    service_name: str = Field(
        default="greenlang",
        min_length=1,
        max_length=128,
        description="Name of the emitting service.",
    )
    service_version: str = Field(
        default="1.0.0",
        min_length=1,
        max_length=32,
        description="Semantic version of the running service.",
    )
    environment: str = Field(
        default="development",
        description="Deployment environment (development, staging, production).",
    )

    # -- Redaction -----------------------------------------------------------

    enable_redaction: bool = Field(
        default=True,
        description="Redact PII and secrets from log output.",
    )
    redaction_patterns: list[str] = Field(
        default_factory=list,
        description="Additional regex patterns for custom redaction rules.",
    )

    # -- Correlation / Tracing ----------------------------------------------

    enable_correlation: bool = Field(
        default=True,
        description="Propagate correlation / request IDs across log entries.",
    )
    correlation_header: str = Field(
        default="X-Request-ID",
        min_length=1,
        max_length=128,
        description="HTTP header name for request correlation ID.",
    )
    trace_header: str = Field(
        default="X-Trace-ID",
        min_length=1,
        max_length=128,
        description="HTTP header name for distributed trace ID.",
    )

    # -- OpenTelemetry -------------------------------------------------------

    enable_otel: bool = Field(
        default=False,
        description="Enable OpenTelemetry log bridge for OTLP export.",
    )

    # -- File Output ---------------------------------------------------------

    log_file: Optional[str] = Field(
        default=None,
        description="Optional filesystem path to write log output to.",
    )

    # -- Limits & Sampling ---------------------------------------------------

    max_message_length: int = Field(
        default=10000,
        ge=100,
        le=1_000_000,
        description="Maximum event message length before truncation.",
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of DEBUG-level logs to emit (1.0 = all, 0.1 = 10%%). "
            "Higher levels are never sampled."
        ),
    )

    # -- Tenant & Async ------------------------------------------------------

    tenant_isolation: bool = Field(
        default=True,
        description="Enforce tenant_id presence in log context.",
    )
    async_logging: bool = Field(
        default=True,
        description="Use a non-blocking queue handler for log I/O.",
    )

    # -- Validators ----------------------------------------------------------

    @field_validator("level", mode="before")
    @classmethod
    def normalize_level(cls, v: str) -> str:
        """Normalize the log level to upper-case and validate it.

        Args:
            v: Raw log level string.

        Returns:
            Upper-cased log level string.

        Raises:
            ValueError: If the level is not a recognized Python log level.
        """
        normalized = str(v).strip().upper()
        if normalized not in _LOG_LEVEL_MAP:
            raise ValueError(
                f"Invalid log level '{v}'. "
                f"Allowed: {sorted(_LOG_LEVEL_MAP.keys())}"
            )
        return normalized

    @field_validator("environment", mode="before")
    @classmethod
    def resolve_environment(cls, v: Optional[str]) -> str:
        """Resolve environment from the project-wide GL_ENVIRONMENT variable.

        Resolution order:
        1. ``GL_ENVIRONMENT`` (project-wide, highest priority)
        2. ``GL_LOG_ENVIRONMENT`` (via pydantic-settings prefix, the ``v`` arg)
        3. Falls back to ``development`` if neither is set.

        Args:
            v: Value provided by pydantic-settings or constructor.

        Returns:
            Resolved environment string.
        """
        gl_env = os.environ.get("GL_ENVIRONMENT", "").strip().lower()
        if gl_env:
            return gl_env

        gl_log_env = os.environ.get("GL_LOG_ENVIRONMENT", "").strip().lower()
        if gl_log_env:
            return gl_log_env

        if v is not None and str(v).strip():
            return str(v).strip().lower()

        return "development"

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: float) -> float:
        """Ensure sample_rate is a valid probability.

        Args:
            v: Sample rate value.

        Returns:
            Validated sample rate.

        Raises:
            ValueError: If not in [0.0, 1.0].
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"sample_rate must be between 0.0 and 1.0, got {v}"
            )
        return v

    # -- Public Methods ------------------------------------------------------

    def get_structlog_level(self) -> int:
        """Return the Python logging integer level corresponding to ``self.level``.

        Returns:
            Integer log level (e.g. ``logging.INFO`` which equals 20).

        Example:
            >>> config = LoggingConfig(level="DEBUG")
            >>> config.get_structlog_level()
            10
        """
        return _LOG_LEVEL_MAP.get(self.level, logging.INFO)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[LoggingConfig] = None


def get_config(force_reload: bool = False) -> LoggingConfig:
    """Return the logging configuration singleton.

    Creates a :class:`LoggingConfig` on first call by reading environment
    variables with the ``GL_LOG_`` prefix. Subsequent calls return the
    cached instance unless ``force_reload`` is True.

    Args:
        force_reload: If True, discard the cached config and reload from
            environment variables.

    Returns:
        The :class:`LoggingConfig` singleton.

    Example:
        >>> config = get_config()
        >>> config.level
        'INFO'
    """
    global _config_instance

    if _config_instance is not None and not force_reload:
        return _config_instance

    _config_instance = LoggingConfig()

    logger.info(
        "Logging config loaded: level=%s, format=%s, env=%s, "
        "redaction=%s, correlation=%s, otel=%s, async=%s",
        _config_instance.level,
        _config_instance.format,
        _config_instance.environment,
        _config_instance.enable_redaction,
        _config_instance.enable_correlation,
        _config_instance.enable_otel,
        _config_instance.async_logging,
    )

    return _config_instance


def reset_config() -> None:
    """Reset the cached configuration singleton.

    Used for testing so that a fresh config is loaded on the next
    call to :func:`get_config`.
    """
    global _config_instance
    _config_instance = None
    logger.debug("Logging config singleton reset.")
