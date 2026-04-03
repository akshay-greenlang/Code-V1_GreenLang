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
    >>> from greenlang.agents.data.data_gateway.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.cache_default_ttl, cfg.max_query_complexity)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 API Gateway Agent
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from greenlang.data_commons.config_base import (
    BaseDataConfig,
    EnvReader,
    create_config_singleton,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_DATA_GATEWAY_"


# ---------------------------------------------------------------------------
# DataGatewayConfig
# ---------------------------------------------------------------------------


@dataclass
class DataGatewayConfig(BaseDataConfig):
    """Configuration for the GreenLang API Gateway Agent SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only gateway-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_DATA_GATEWAY_`` prefix.

    Attributes:
        cache_default_ttl: Default cache TTL in seconds.
        cache_max_size_mb: Maximum cache size in megabytes.
        max_query_complexity: Maximum allowed query complexity score.
        max_sources_per_query: Maximum data sources per single query.
        query_timeout_seconds: Default query execution timeout in seconds.
        connection_pool_min: Minimum connections in the gateway-to-sources pool.
        connection_pool_max: Maximum connections in the gateway-to-sources pool.
        health_check_interval: Interval between health checks in seconds.
        circuit_breaker_threshold: Consecutive failures before circuit opens.
        circuit_breaker_timeout: Seconds before half-open retry after trip.
        retry_max_attempts: Maximum retry attempts for failed requests.
        retry_base_delay: Base delay in seconds between retries.
        batch_max_size: Maximum number of queries in a batch request.
        retention_days: Number of days to retain query history and logs.
        enable_graphql: Whether to enable the GraphQL query interface.
        enable_cache_warming: Whether to enable proactive cache warming.
    """

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

    # -- Batch processing (gateway-specific) ---------------------------------
    batch_max_size: int = 50

    # -- Data retention ------------------------------------------------------
    retention_days: int = 90

    # -- Feature toggles -----------------------------------------------------
    enable_graphql: bool = False
    enable_cache_warming: bool = True

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DataGatewayConfig:
        """Build a DataGatewayConfig from environment variables.

        Every field can be overridden via ``GL_DATA_GATEWAY_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated DataGatewayConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            cache_default_ttl=env.int(
                "CACHE_DEFAULT_TTL", cls.cache_default_ttl,
            ),
            cache_max_size_mb=env.int(
                "CACHE_MAX_SIZE_MB", cls.cache_max_size_mb,
            ),
            max_query_complexity=env.int(
                "MAX_QUERY_COMPLEXITY", cls.max_query_complexity,
            ),
            max_sources_per_query=env.int(
                "MAX_SOURCES_PER_QUERY", cls.max_sources_per_query,
            ),
            query_timeout_seconds=env.int(
                "QUERY_TIMEOUT_SECONDS", cls.query_timeout_seconds,
            ),
            connection_pool_min=env.int(
                "CONNECTION_POOL_MIN", cls.connection_pool_min,
            ),
            connection_pool_max=env.int(
                "CONNECTION_POOL_MAX", cls.connection_pool_max,
            ),
            health_check_interval=env.int(
                "HEALTH_CHECK_INTERVAL", cls.health_check_interval,
            ),
            circuit_breaker_threshold=env.int(
                "CIRCUIT_BREAKER_THRESHOLD",
                cls.circuit_breaker_threshold,
            ),
            circuit_breaker_timeout=env.int(
                "CIRCUIT_BREAKER_TIMEOUT",
                cls.circuit_breaker_timeout,
            ),
            retry_max_attempts=env.int(
                "RETRY_MAX_ATTEMPTS", cls.retry_max_attempts,
            ),
            retry_base_delay=env.float(
                "RETRY_BASE_DELAY", cls.retry_base_delay,
            ),
            batch_max_size=env.int(
                "BATCH_MAX_SIZE", cls.batch_max_size,
            ),
            retention_days=env.int(
                "RETENTION_DAYS", cls.retention_days,
            ),
            enable_graphql=env.bool(
                "ENABLE_GRAPHQL", cls.enable_graphql,
            ),
            enable_cache_warming=env.bool(
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

get_config, set_config, reset_config = create_config_singleton(
    DataGatewayConfig, _ENV_PREFIX,
)

__all__ = [
    "DataGatewayConfig",
    "get_config",
    "set_config",
    "reset_config",
]
