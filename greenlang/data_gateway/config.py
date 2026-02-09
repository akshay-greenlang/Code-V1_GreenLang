# -*- coding: utf-8 -*-
"""
API Gateway Agent Service Configuration - AGENT-DATA-004: Data Gateway

Centralized configuration for the API Gateway Agent SDK covering:
- Database, Redis, and S3 connection URLs
- Cache defaults (TTL, max size)
- Query constraints (complexity, sources, timeout)
- Connection pool sizing
- Health check interval
- Circuit breaker settings (threshold, timeout)
- Retry policy (max attempts, base delay)
- Batch processing defaults (max size, worker count)
- Data retention policy
- Feature toggles (GraphQL, cache warming)
- Logging level

All settings can be overridden via environment variables with the
``GL_DATA_GATEWAY_`` prefix (e.g. ``GL_DATA_GATEWAY_LOG_LEVEL``).

Example:
    >>> from greenlang.data_gateway.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.cache_default_ttl, cfg.max_query_complexity)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 API Gateway Agent
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

_ENV_PREFIX = "GL_DATA_GATEWAY_"


# ---------------------------------------------------------------------------
# DataGatewayConfig
# ---------------------------------------------------------------------------


@dataclass
class DataGatewayConfig:
    """Complete configuration for the GreenLang API Gateway Agent SDK.

    Attributes are grouped by concern: connections, cache settings,
    query constraints, connection pool, health checks, circuit breaker,
    retry policy, batch processing, data retention, feature toggles,
    and logging.

    All attributes can be overridden via environment variables using the
    ``GL_DATA_GATEWAY_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        s3_bucket_url: S3 bucket URL for document storage.
        log_level: Logging level for the data gateway service.
        cache_default_ttl: Default cache TTL in seconds.
        cache_max_size_mb: Maximum cache size in megabytes.
        max_query_complexity: Maximum allowed query complexity score.
        max_sources_per_query: Maximum data sources per single query.
        query_timeout_seconds: Default query execution timeout in seconds.
        connection_pool_min: Minimum connections in the connection pool.
        connection_pool_max: Maximum connections in the connection pool.
        health_check_interval: Interval between health checks in seconds.
        circuit_breaker_threshold: Consecutive failures before circuit opens.
        circuit_breaker_timeout: Seconds before half-open retry after trip.
        retry_max_attempts: Maximum retry attempts for failed requests.
        retry_base_delay: Base delay in seconds between retries.
        batch_max_size: Maximum number of queries in a batch request.
        batch_worker_count: Number of parallel workers for batch processing.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        retention_days: Number of days to retain query history and logs.
        enable_graphql: Whether to enable the GraphQL query interface.
        enable_cache_warming: Whether to enable proactive cache warming.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Cache settings ------------------------------------------------------
    cache_default_ttl: int = 300
    cache_max_size_mb: int = 256

    # -- Query constraints ---------------------------------------------------
    max_query_complexity: int = 100
    max_sources_per_query: int = 10
    query_timeout_seconds: int = 30

    # -- Connection pool (gateway-to-sources) --------------------------------
    connection_pool_min: int = 2
    connection_pool_max: int = 10

    # -- Health checks -------------------------------------------------------
    health_check_interval: int = 60

    # -- Circuit breaker -----------------------------------------------------
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

    # -- Retry policy --------------------------------------------------------
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0

    # -- Batch processing ----------------------------------------------------
    batch_max_size: int = 50
    batch_worker_count: int = 4

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Data retention ------------------------------------------------------
    retention_days: int = 90

    # -- Feature toggles -----------------------------------------------------
    enable_graphql: bool = False
    enable_cache_warming: bool = True

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DataGatewayConfig:
        """Build a DataGatewayConfig from environment variables.

        Every field can be overridden via ``GL_DATA_GATEWAY_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated DataGatewayConfig instance.
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
                    "Invalid float for %s%s=%s, using default %s",
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
            s3_bucket_url=_str("S3_BUCKET_URL", cls.s3_bucket_url),
            log_level=_str("LOG_LEVEL", cls.log_level),
            cache_default_ttl=_int(
                "CACHE_DEFAULT_TTL", cls.cache_default_ttl,
            ),
            cache_max_size_mb=_int(
                "CACHE_MAX_SIZE_MB", cls.cache_max_size_mb,
            ),
            max_query_complexity=_int(
                "MAX_QUERY_COMPLEXITY", cls.max_query_complexity,
            ),
            max_sources_per_query=_int(
                "MAX_SOURCES_PER_QUERY", cls.max_sources_per_query,
            ),
            query_timeout_seconds=_int(
                "QUERY_TIMEOUT_SECONDS", cls.query_timeout_seconds,
            ),
            connection_pool_min=_int(
                "CONNECTION_POOL_MIN", cls.connection_pool_min,
            ),
            connection_pool_max=_int(
                "CONNECTION_POOL_MAX", cls.connection_pool_max,
            ),
            health_check_interval=_int(
                "HEALTH_CHECK_INTERVAL", cls.health_check_interval,
            ),
            circuit_breaker_threshold=_int(
                "CIRCUIT_BREAKER_THRESHOLD",
                cls.circuit_breaker_threshold,
            ),
            circuit_breaker_timeout=_int(
                "CIRCUIT_BREAKER_TIMEOUT",
                cls.circuit_breaker_timeout,
            ),
            retry_max_attempts=_int(
                "RETRY_MAX_ATTEMPTS", cls.retry_max_attempts,
            ),
            retry_base_delay=_float(
                "RETRY_BASE_DELAY", cls.retry_base_delay,
            ),
            batch_max_size=_int(
                "BATCH_MAX_SIZE", cls.batch_max_size,
            ),
            batch_worker_count=_int(
                "BATCH_WORKER_COUNT", cls.batch_worker_count,
            ),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            retention_days=_int(
                "RETENTION_DAYS", cls.retention_days,
            ),
            enable_graphql=_bool(
                "ENABLE_GRAPHQL", cls.enable_graphql,
            ),
            enable_cache_warming=_bool(
                "ENABLE_CACHE_WARMING", cls.enable_cache_warming,
            ),
        )

        logger.info(
            "DataGatewayConfig loaded: cache_ttl=%ds, cache_max=%dMB, "
            "max_complexity=%d, max_sources=%d, query_timeout=%ds, "
            "pool=%d-%d, health_interval=%ds, cb_threshold=%d, "
            "cb_timeout=%ds, retry=%d/%.1fs, batch=%d/%d workers, "
            "retention=%dd, graphql=%s, cache_warming=%s",
            config.cache_default_ttl,
            config.cache_max_size_mb,
            config.max_query_complexity,
            config.max_sources_per_query,
            config.query_timeout_seconds,
            config.connection_pool_min,
            config.connection_pool_max,
            config.health_check_interval,
            config.circuit_breaker_threshold,
            config.circuit_breaker_timeout,
            config.retry_max_attempts,
            config.retry_base_delay,
            config.batch_max_size,
            config.batch_worker_count,
            config.retention_days,
            config.enable_graphql,
            config.enable_cache_warming,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[DataGatewayConfig] = None
_config_lock = threading.Lock()


def get_config() -> DataGatewayConfig:
    """Return the singleton DataGatewayConfig, creating from env if needed.

    Returns:
        DataGatewayConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = DataGatewayConfig.from_env()
    return _config_instance


def set_config(config: DataGatewayConfig) -> None:
    """Replace the singleton DataGatewayConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("DataGatewayConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "DataGatewayConfig",
    "get_config",
    "set_config",
    "reset_config",
]
