# -*- coding: utf-8 -*-
"""
ERP/Finance Connector Service Configuration - AGENT-DATA-003: ERP Connector

Centralized configuration for the ERP/Finance Connector SDK covering:
- ERP system connection defaults (system type, timeout, max connections)
- Sync settings (batch size, max records, timeout, incremental mode)
- Scope 3 mapping defaults (strategy, auto-classification)
- Emissions calculation toggles (methodology, currency)
- Currency conversion settings
- Batch processing defaults (max connections, worker count)
- Connection pool sizing
- Logging level

All settings can be overridden via environment variables with the
``GL_ERP_CONNECTOR_`` prefix (e.g. ``GL_ERP_CONNECTOR_DEFAULT_ERP_SYSTEM``).

Example:
    >>> from greenlang.erp_connector.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_erp_system, cfg.sync_batch_size)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-003 ERP/Finance Connector
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

_ENV_PREFIX = "GL_ERP_CONNECTOR_"


# ---------------------------------------------------------------------------
# ERPConnectorConfig
# ---------------------------------------------------------------------------


@dataclass
class ERPConnectorConfig:
    """Complete configuration for the GreenLang ERP/Finance Connector SDK.

    Attributes are grouped by concern: connections, ERP defaults,
    sync settings, Scope 3 mapping, emissions calculation, currency
    conversion, batch processing, pool sizing, and logging.

    All attributes can be overridden via environment variables using the
    ``GL_ERP_CONNECTOR_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        s3_bucket_url: S3 bucket URL for document storage.
        default_erp_system: Default ERP system type for new connections.
        connection_timeout_seconds: Default timeout for ERP connections.
        max_connections: Maximum concurrent ERP connections allowed.
        sync_batch_size: Number of records per sync batch.
        sync_max_records: Maximum total records per sync operation.
        sync_timeout_seconds: Timeout in seconds for sync operations.
        enable_incremental_sync: Whether to enable incremental sync mode.
        default_mapping_strategy: Default Scope 3 mapping strategy.
        enable_auto_classification: Whether to auto-classify spend categories.
        default_emission_methodology: Default emission calculation methodology.
        default_currency: Default currency for spend amounts.
        enable_emissions_calculation: Whether to calculate emissions estimates.
        base_currency: Base currency for conversion operations.
        enable_currency_conversion: Whether to enable currency conversion.
        batch_max_connections: Maximum connections in batch processing mode.
        batch_worker_count: Number of parallel workers for batch processing.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        log_level: Logging level for the ERP connector service.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""

    # -- ERP defaults --------------------------------------------------------
    default_erp_system: str = "simulated"
    connection_timeout_seconds: int = 60
    max_connections: int = 10

    # -- Sync settings -------------------------------------------------------
    sync_batch_size: int = 1000
    sync_max_records: int = 100000
    sync_timeout_seconds: int = 300
    enable_incremental_sync: bool = True

    # -- Scope 3 mapping -----------------------------------------------------
    default_mapping_strategy: str = "rule_based"
    enable_auto_classification: bool = True

    # -- Emissions calculation ------------------------------------------------
    default_emission_methodology: str = "eeio"
    default_currency: str = "USD"
    enable_emissions_calculation: bool = True

    # -- Currency conversion --------------------------------------------------
    base_currency: str = "USD"
    enable_currency_conversion: bool = True

    # -- Batch processing ----------------------------------------------------
    batch_max_connections: int = 50
    batch_worker_count: int = 4

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ERPConnectorConfig:
        """Build an ERPConnectorConfig from environment variables.

        Every field can be overridden via ``GL_ERP_CONNECTOR_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.

        Returns:
            Populated ERPConnectorConfig instance.
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

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        config = cls(
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            s3_bucket_url=_str("S3_BUCKET_URL", cls.s3_bucket_url),
            default_erp_system=_str(
                "DEFAULT_ERP_SYSTEM", cls.default_erp_system,
            ),
            connection_timeout_seconds=_int(
                "CONNECTION_TIMEOUT_SECONDS",
                cls.connection_timeout_seconds,
            ),
            max_connections=_int(
                "MAX_CONNECTIONS", cls.max_connections,
            ),
            sync_batch_size=_int(
                "SYNC_BATCH_SIZE", cls.sync_batch_size,
            ),
            sync_max_records=_int(
                "SYNC_MAX_RECORDS", cls.sync_max_records,
            ),
            sync_timeout_seconds=_int(
                "SYNC_TIMEOUT_SECONDS", cls.sync_timeout_seconds,
            ),
            enable_incremental_sync=_bool(
                "ENABLE_INCREMENTAL_SYNC",
                cls.enable_incremental_sync,
            ),
            default_mapping_strategy=_str(
                "DEFAULT_MAPPING_STRATEGY",
                cls.default_mapping_strategy,
            ),
            enable_auto_classification=_bool(
                "ENABLE_AUTO_CLASSIFICATION",
                cls.enable_auto_classification,
            ),
            default_emission_methodology=_str(
                "DEFAULT_EMISSION_METHODOLOGY",
                cls.default_emission_methodology,
            ),
            default_currency=_str(
                "DEFAULT_CURRENCY", cls.default_currency,
            ),
            enable_emissions_calculation=_bool(
                "ENABLE_EMISSIONS_CALCULATION",
                cls.enable_emissions_calculation,
            ),
            base_currency=_str(
                "BASE_CURRENCY", cls.base_currency,
            ),
            enable_currency_conversion=_bool(
                "ENABLE_CURRENCY_CONVERSION",
                cls.enable_currency_conversion,
            ),
            batch_max_connections=_int(
                "BATCH_MAX_CONNECTIONS", cls.batch_max_connections,
            ),
            batch_worker_count=_int(
                "BATCH_WORKER_COUNT", cls.batch_worker_count,
            ),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            log_level=_str("LOG_LEVEL", cls.log_level),
        )

        logger.info(
            "ERPConnectorConfig loaded: erp_system=%s, timeout=%ds, "
            "max_connections=%d, sync_batch=%d, sync_max=%d, "
            "incremental=%s, mapping=%s, methodology=%s, "
            "currency=%s, emissions=%s, workers=%d",
            config.default_erp_system,
            config.connection_timeout_seconds,
            config.max_connections,
            config.sync_batch_size,
            config.sync_max_records,
            config.enable_incremental_sync,
            config.default_mapping_strategy,
            config.default_emission_methodology,
            config.default_currency,
            config.enable_emissions_calculation,
            config.batch_worker_count,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[ERPConnectorConfig] = None
_config_lock = threading.Lock()


def get_config() -> ERPConnectorConfig:
    """Return the singleton ERPConnectorConfig, creating from env if needed.

    Returns:
        ERPConnectorConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ERPConnectorConfig.from_env()
    return _config_instance


def set_config(config: ERPConnectorConfig) -> None:
    """Replace the singleton ERPConnectorConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("ERPConnectorConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "ERPConnectorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
