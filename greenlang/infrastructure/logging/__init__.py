"""
GreenLang Structured Logging - INFRA-009

Production-grade structured logging framework for the GreenLang Climate OS
platform. Built on structlog with automatic PII/secret redaction, correlation
ID propagation, tenant isolation, and both JSON (production) and colorized
console (development) output formats.

This package replaces the legacy :mod:`greenlang.monitoring.telemetry.logging`
module with a modern, structlog-based implementation that integrates with
Loki, OpenTelemetry, and the GreenLang agent execution pipeline.

Public API:
    Configuration:
        - LoggingConfig: Pydantic Settings model for all logging parameters.
        - configure_logging: One-time setup of structlog + stdlib logging.
        - get_logger: Obtain a structlog BoundLogger by name.

    Middleware:
        - StructuredLoggingMiddleware: FastAPI middleware for request logging.

    Redaction:
        - RedactionProcessor: structlog processor that redacts PII/secrets.
        - SensitiveDataPatterns: Regex constants for sensitive data detection.

    Context:
        - bind_context: Bind key-value pairs into the logging context.
        - clear_context: Clear all bound context variables.
        - get_context: Retrieve the current context as a dict.
        - logging_context: Context manager for scoped context bindings.
        - bind_agent_context: Convenience for agent execution metadata.
        - bind_request_context: Convenience for HTTP request metadata.

    Formatters:
        - JsonFormatter: stdlib Formatter for JSON output.
        - ConsoleFormatter: stdlib Formatter for colorized terminal output.
        - get_formatter: Factory to select formatter by name.

Example:
    >>> from greenlang.infrastructure.logging import (
    ...     configure_logging, get_logger, LoggingConfig,
    ...     bind_context, clear_context,
    ... )
    >>> configure_logging(LoggingConfig(level="DEBUG", format="console"))
    >>> log = get_logger("my_agent")
    >>> bind_context(tenant_id="t-acme", request_id="req-123")
    >>> log.info("emission_calculated", scope="scope1", tonnes_co2e=42.5)
    >>> clear_context()
"""

from __future__ import annotations

from greenlang.infrastructure.logging.config import LoggingConfig, get_config
from greenlang.infrastructure.logging.context import (
    bind_agent_context,
    bind_context,
    bind_request_context,
    clear_context,
    get_context,
    logging_context,
)
from greenlang.infrastructure.logging.formatters import (
    ConsoleFormatter,
    JsonFormatter,
    get_formatter,
)
from greenlang.infrastructure.logging.middleware import StructuredLoggingMiddleware
from greenlang.infrastructure.logging.redaction import (
    RedactionProcessor,
    SensitiveDataPatterns,
)
from greenlang.infrastructure.logging.setup import configure_logging, get_logger

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Configuration & Setup
    "LoggingConfig",
    "get_config",
    "configure_logging",
    "get_logger",
    # Middleware
    "StructuredLoggingMiddleware",
    # Redaction
    "RedactionProcessor",
    "SensitiveDataPatterns",
    # Context
    "bind_context",
    "clear_context",
    "get_context",
    "logging_context",
    "bind_agent_context",
    "bind_request_context",
    # Formatters
    "JsonFormatter",
    "ConsoleFormatter",
    "get_formatter",
]
