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
    >>> from greenlang.agents.data.erp_connector.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_erp_system, cfg.sync_batch_size)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-003 ERP/Finance Connector
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

_ENV_PREFIX = "GL_ERP_CONNECTOR_"


# ---------------------------------------------------------------------------
# ERPConnectorConfig
# ---------------------------------------------------------------------------


@dataclass
class ERPConnectorConfig(BaseDataConfig):
    """Configuration for the GreenLang ERP/Finance Connector SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only ERP-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_ERP_CONNECTOR_`` prefix.

    Attributes:
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
    """

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

    # -- Batch processing (ERP-specific) --------------------------------------
    batch_max_connections: int = 50

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ERPConnectorConfig:
        """Build an ERPConnectorConfig from environment variables.

        Every field can be overridden via ``GL_ERP_CONNECTOR_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated ERPConnectorConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            # ERP defaults
            default_erp_system=env.str(
                "DEFAULT_ERP_SYSTEM", cls.default_erp_system,
            ),
            connection_timeout_seconds=env.int(
                "CONNECTION_TIMEOUT_SECONDS",
                cls.connection_timeout_seconds,
            ),
            max_connections=env.int(
                "MAX_CONNECTIONS", cls.max_connections,
            ),
            # Sync settings
            sync_batch_size=env.int(
                "SYNC_BATCH_SIZE", cls.sync_batch_size,
            ),
            sync_max_records=env.int(
                "SYNC_MAX_RECORDS", cls.sync_max_records,
            ),
            sync_timeout_seconds=env.int(
                "SYNC_TIMEOUT_SECONDS", cls.sync_timeout_seconds,
            ),
            enable_incremental_sync=env.bool(
                "ENABLE_INCREMENTAL_SYNC",
                cls.enable_incremental_sync,
            ),
            # Scope 3 mapping
            default_mapping_strategy=env.str(
                "DEFAULT_MAPPING_STRATEGY",
                cls.default_mapping_strategy,
            ),
            enable_auto_classification=env.bool(
                "ENABLE_AUTO_CLASSIFICATION",
                cls.enable_auto_classification,
            ),
            # Emissions calculation
            default_emission_methodology=env.str(
                "DEFAULT_EMISSION_METHODOLOGY",
                cls.default_emission_methodology,
            ),
            default_currency=env.str(
                "DEFAULT_CURRENCY", cls.default_currency,
            ),
            enable_emissions_calculation=env.bool(
                "ENABLE_EMISSIONS_CALCULATION",
                cls.enable_emissions_calculation,
            ),
            # Currency conversion
            base_currency=env.str(
                "BASE_CURRENCY", cls.base_currency,
            ),
            enable_currency_conversion=env.bool(
                "ENABLE_CURRENCY_CONVERSION",
                cls.enable_currency_conversion,
            ),
            # Batch (ERP-specific)
            batch_max_connections=env.int(
                "BATCH_MAX_CONNECTIONS", cls.batch_max_connections,
            ),
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

get_config, set_config, reset_config = create_config_singleton(
    ERPConnectorConfig, _ENV_PREFIX,
)

__all__ = [
    "ERPConnectorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
