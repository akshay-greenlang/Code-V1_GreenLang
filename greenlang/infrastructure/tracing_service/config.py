# -*- coding: utf-8 -*-
"""
TracingConfig - Configuration for GreenLang Distributed Tracing SDK (OBS-003)

Provides a dataclass-based configuration object that reads sensible defaults
from environment variables, following the OpenTelemetry environment variable
conventions (OTEL_*) alongside GreenLang-specific settings (GL_*).

All fields have production-safe defaults so the SDK can be initialised with
zero configuration in development while still being fully tuneable for
staging and production deployments.

Example:
    >>> config = TracingConfig()
    >>> config = TracingConfig(service_name="api-service", sampling_rate=0.25)
    >>> config = TracingConfig.from_env()

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _env_bool(key: str, default: str = "false") -> bool:
    """Read a boolean from an environment variable."""
    return os.getenv(key, default).lower() in ("true", "1", "yes")


def _env_float(key: str, default: str = "1.0") -> float:
    """Read a float from an environment variable."""
    try:
        return float(os.getenv(key, default))
    except (ValueError, TypeError):
        return float(default)


def _env_int(key: str, default: str = "0") -> int:
    """Read an int from an environment variable."""
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return int(default)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TracingConfig:
    """Configuration for the GreenLang distributed tracing SDK.

    Attributes:
        service_name: Logical service name emitted in trace Resource.
        service_version: SemVer version string for the service.
        environment: Deployment environment (dev, staging, prod).
        otlp_endpoint: OTLP collector gRPC endpoint.
        otlp_insecure: Whether to use insecure (plaintext) gRPC channel.
        otlp_timeout: Timeout in seconds for OTLP export RPCs.
        otlp_headers: Additional headers sent with OTLP export calls.
        sampling_rate: Base trace-id ratio sampling rate (0.0-1.0).
        always_sample_errors: Force-sample spans that record exceptions.
        batch_max_queue_size: Maximum spans queued before back-pressure.
        batch_max_export_batch_size: Maximum spans per export batch.
        batch_schedule_delay_ms: Delay between batch exports in ms.
        batch_export_timeout_ms: Timeout for a single batch export in ms.
        instrument_fastapi: Auto-instrument FastAPI.
        instrument_httpx: Auto-instrument httpx.
        instrument_psycopg: Auto-instrument psycopg / psycopg2.
        instrument_redis: Auto-instrument redis-py.
        instrument_celery: Auto-instrument Celery workers.
        instrument_requests: Auto-instrument requests library.
        tenant_header: HTTP header carrying the tenant identifier.
        propagate_baggage: Enable W3C Baggage propagation.
        enrich_spans: Inject GreenLang-specific attributes into spans.
        service_overrides: Per-service sampling rate overrides.
        console_exporter: Also export spans to stdout (debug).
        enabled: Master switch to disable tracing entirely.
    """

    # -- Service identification ------------------------------------------------
    service_name: str = field(
        default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "greenlang"),
    )
    service_version: str = field(
        default_factory=lambda: os.getenv("GL_SERVICE_VERSION", "1.0.0"),
    )
    environment: str = field(
        default_factory=lambda: os.getenv("GL_ENVIRONMENT", "dev"),
    )

    # -- OTLP exporter ---------------------------------------------------------
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"
        ),
    )
    otlp_insecure: bool = field(
        default_factory=lambda: _env_bool("OTEL_EXPORTER_OTLP_INSECURE", "true"),
    )
    otlp_timeout: int = field(
        default_factory=lambda: _env_int("OTEL_EXPORTER_OTLP_TIMEOUT", "10"),
    )
    otlp_headers: Dict[str, str] = field(default_factory=dict)

    # -- Sampling --------------------------------------------------------------
    sampling_rate: float = field(
        default_factory=lambda: _env_float("OTEL_TRACES_SAMPLER_ARG", "1.0"),
    )
    always_sample_errors: bool = True

    # -- Batch span processor --------------------------------------------------
    batch_max_queue_size: int = 2048
    batch_max_export_batch_size: int = 512
    batch_schedule_delay_ms: int = 5000
    batch_export_timeout_ms: int = 30000

    # -- Auto-instrumentation flags --------------------------------------------
    instrument_fastapi: bool = True
    instrument_httpx: bool = True
    instrument_psycopg: bool = True
    instrument_redis: bool = True
    instrument_celery: bool = True
    instrument_requests: bool = True

    # -- GreenLang-specific ----------------------------------------------------
    tenant_header: str = "X-Tenant-ID"
    propagate_baggage: bool = True
    enrich_spans: bool = True
    service_overrides: Dict[str, float] = field(default_factory=dict)

    # -- Debug / lifecycle -----------------------------------------------------
    console_exporter: bool = field(
        default_factory=lambda: _env_bool("OTEL_TRACES_CONSOLE", "false"),
    )
    enabled: bool = field(
        default_factory=lambda: _env_bool("OTEL_TRACES_ENABLED", "true"),
    )

    # -- Factory ---------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "TracingConfig":
        """Create a TracingConfig populated entirely from environment variables.

        Returns:
            A fully initialised TracingConfig.
        """
        return cls()

    def __post_init__(self) -> None:
        """Validate configuration values after initialisation."""
        self.sampling_rate = max(0.0, min(1.0, self.sampling_rate))
        if self.batch_max_export_batch_size > self.batch_max_queue_size:
            self.batch_max_export_batch_size = self.batch_max_queue_size
        if not self.otlp_endpoint:
            self.otlp_endpoint = "http://otel-collector:4317"
        logger.debug(
            "TracingConfig: service=%s env=%s endpoint=%s sampling=%.2f enabled=%s",
            self.service_name,
            self.environment,
            self.otlp_endpoint,
            self.sampling_rate,
            self.enabled,
        )


__all__ = ["TracingConfig"]
