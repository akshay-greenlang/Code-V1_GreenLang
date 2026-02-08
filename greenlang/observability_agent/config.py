# -*- coding: utf-8 -*-
"""
Observability Agent Service Configuration - AGENT-FOUND-010

Centralized configuration for the Observability & Telemetry Agent SDK covering:
- Connection URLs for Prometheus, Grafana, Tempo, Loki, Alertmanager
- Database and Redis connection strings
- Metrics retention and buffer sizing
- Alert evaluation and health check intervals
- SLO evaluation windows and burn rate configuration
- OpenTelemetry and structured logging toggles
- Connection pool sizing

All settings can be overridden via environment variables with the
``GL_OBSERVABILITY_AGENT_`` prefix (e.g. ``GL_OBSERVABILITY_AGENT_PROMETHEUS_URL``).

Example:
    >>> from greenlang.observability_agent.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.prometheus_url, cfg.metrics_retention_hours)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-010 Observability & Telemetry Agent
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_OBSERVABILITY_AGENT_"


# ---------------------------------------------------------------------------
# ObservabilityAgentConfig
# ---------------------------------------------------------------------------


@dataclass
class ObservabilityAgentConfig:
    """Complete configuration for the GreenLang Observability Agent SDK.

    Attributes are grouped by concern: connections, observability backends,
    retention and sizing, evaluation intervals, feature toggles, SLO
    defaults, pool sizing, and logging.

    All attributes can be overridden via environment variables using the
    ``GL_OBSERVABILITY_AGENT_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        prometheus_url: Prometheus server URL for metric queries.
        grafana_url: Grafana server URL for dashboard provisioning.
        tempo_url: Grafana Tempo URL for distributed trace queries.
        loki_url: Grafana Loki URL for log aggregation queries.
        alertmanager_url: Alertmanager URL for alert routing.
        metrics_retention_hours: Hours to retain metric data in memory.
        max_active_spans: Maximum number of concurrently active trace spans.
        log_buffer_size: Maximum number of log entries held in the buffer.
        alert_evaluation_interval_seconds: Seconds between alert rule evaluations.
        health_check_interval_seconds: Seconds between health check probes.
        slo_evaluation_interval_seconds: Seconds between SLO compliance evaluations.
        enable_prometheus_export: Whether to expose Prometheus metric endpoints.
        enable_opentelemetry: Whether to enable OpenTelemetry tracing.
        enable_structured_logging: Whether to emit JSON structured logs.
        default_slo_target: Default SLO target ratio (e.g. 0.999 = 99.9%).
        burn_rate_short_window_minutes: Short burn rate evaluation window.
        burn_rate_long_window_minutes: Long burn rate evaluation window.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        log_level: Logging level for the observability agent service.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Observability backends ----------------------------------------------
    prometheus_url: str = "http://localhost:9090"
    grafana_url: str = "http://localhost:3000"
    tempo_url: str = "http://localhost:3200"
    loki_url: str = "http://localhost:3100"
    alertmanager_url: str = "http://localhost:9093"

    # -- Retention and sizing ------------------------------------------------
    metrics_retention_hours: int = 24
    max_active_spans: int = 10000
    log_buffer_size: int = 10000

    # -- Evaluation intervals ------------------------------------------------
    alert_evaluation_interval_seconds: int = 60
    health_check_interval_seconds: int = 30
    slo_evaluation_interval_seconds: int = 300

    # -- Feature toggles -----------------------------------------------------
    enable_prometheus_export: bool = True
    enable_opentelemetry: bool = True
    enable_structured_logging: bool = True

    # -- SLO defaults --------------------------------------------------------
    default_slo_target: float = 0.999
    burn_rate_short_window_minutes: int = 5
    burn_rate_long_window_minutes: int = 60

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ObservabilityAgentConfig:
        """Build an ObservabilityAgentConfig from environment variables.

        Every field can be overridden via ``GL_OBSERVABILITY_AGENT_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated ObservabilityAgentConfig instance.
        """
        prefix = _ENV_PREFIX

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _env(name)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        def _int(name: str, default: int) -> int:
            val = _env(name)
            if val is None:
                return default
            try:
                return int(val)
            except ValueError:
                logger.warning(
                    "Invalid integer for %s%s=%s, using default %d",
                    prefix, name, val, default,
                )
                return default

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                logger.warning(
                    "Invalid float for %s%s=%s, using default %f",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        config = cls(
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            prometheus_url=_str("PROMETHEUS_URL", cls.prometheus_url),
            grafana_url=_str("GRAFANA_URL", cls.grafana_url),
            tempo_url=_str("TEMPO_URL", cls.tempo_url),
            loki_url=_str("LOKI_URL", cls.loki_url),
            alertmanager_url=_str("ALERTMANAGER_URL", cls.alertmanager_url),
            metrics_retention_hours=_int(
                "METRICS_RETENTION_HOURS", cls.metrics_retention_hours,
            ),
            max_active_spans=_int(
                "MAX_ACTIVE_SPANS", cls.max_active_spans,
            ),
            log_buffer_size=_int(
                "LOG_BUFFER_SIZE", cls.log_buffer_size,
            ),
            alert_evaluation_interval_seconds=_int(
                "ALERT_EVALUATION_INTERVAL_SECONDS",
                cls.alert_evaluation_interval_seconds,
            ),
            health_check_interval_seconds=_int(
                "HEALTH_CHECK_INTERVAL_SECONDS",
                cls.health_check_interval_seconds,
            ),
            slo_evaluation_interval_seconds=_int(
                "SLO_EVALUATION_INTERVAL_SECONDS",
                cls.slo_evaluation_interval_seconds,
            ),
            enable_prometheus_export=_bool(
                "ENABLE_PROMETHEUS_EXPORT", cls.enable_prometheus_export,
            ),
            enable_opentelemetry=_bool(
                "ENABLE_OPENTELEMETRY", cls.enable_opentelemetry,
            ),
            enable_structured_logging=_bool(
                "ENABLE_STRUCTURED_LOGGING", cls.enable_structured_logging,
            ),
            default_slo_target=_float(
                "DEFAULT_SLO_TARGET", cls.default_slo_target,
            ),
            burn_rate_short_window_minutes=_int(
                "BURN_RATE_SHORT_WINDOW_MINUTES",
                cls.burn_rate_short_window_minutes,
            ),
            burn_rate_long_window_minutes=_int(
                "BURN_RATE_LONG_WINDOW_MINUTES",
                cls.burn_rate_long_window_minutes,
            ),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            log_level=_str("LOG_LEVEL", cls.log_level),
        )

        logger.info(
            "ObservabilityAgentConfig loaded: prometheus=%s, grafana=%s, "
            "tempo=%s, loki=%s, alertmanager=%s, retention=%dh, "
            "max_spans=%d, log_buffer=%d, slo_target=%.3f",
            config.prometheus_url,
            config.grafana_url,
            config.tempo_url,
            config.loki_url,
            config.alertmanager_url,
            config.metrics_retention_hours,
            config.max_active_spans,
            config.log_buffer_size,
            config.default_slo_target,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[ObservabilityAgentConfig] = None
_config_lock = threading.Lock()


def get_config() -> ObservabilityAgentConfig:
    """Return the singleton ObservabilityAgentConfig, creating from env if needed.

    Returns:
        ObservabilityAgentConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ObservabilityAgentConfig.from_env()
    return _config_instance


def set_config(config: ObservabilityAgentConfig) -> None:
    """Replace the singleton ObservabilityAgentConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("ObservabilityAgentConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "ObservabilityAgentConfig",
    "get_config",
    "set_config",
    "reset_config",
]
