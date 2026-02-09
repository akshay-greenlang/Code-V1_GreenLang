# -*- coding: utf-8 -*-
"""
EUDR Traceability Connector Service Configuration - AGENT-DATA-004: EUDR Connector

Centralized configuration for the EUDR Traceability Connector SDK covering:
- Connections: database, Redis, S3 storage
- EUDR defaults: deforestation cutoff date, default risk level
- Plot settings: polygon vertices limit, coordinate precision, hectare threshold
- Risk scoring: country/commodity/supplier/traceability weights, thresholds
- DDS settings: validity period, digital signing toggle
- EU Information System: endpoint URL, sandbox mode, timeout
- Batch processing: max size, worker count
- Connection pool sizing
- Logging level
- Record retention period

All settings can be overridden via environment variables with the
``GL_EUDR_TRACEABILITY_`` prefix (e.g. ``GL_EUDR_TRACEABILITY_HIGH_RISK_THRESHOLD``).

Example:
    >>> from greenlang.eudr_traceability.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.deforestation_cutoff_date, cfg.high_risk_threshold)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 EUDR Traceability Connector
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

_ENV_PREFIX = "GL_EUDR_TRACEABILITY_"


# ---------------------------------------------------------------------------
# EUDRTraceabilityConfig
# ---------------------------------------------------------------------------


@dataclass
class EUDRTraceabilityConfig:
    """Complete configuration for the GreenLang EUDR Traceability Connector SDK.

    Attributes are grouped by concern: connections, EUDR defaults,
    plot settings, risk scoring, DDS settings, EU System integration,
    batch processing, pool sizing, logging, and record retention.

    All attributes can be overridden via environment variables using the
    ``GL_EUDR_TRACEABILITY_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        s3_bucket_url: S3 bucket URL for document storage.
        deforestation_cutoff_date: EUDR deforestation-free cutoff date (ISO format).
        default_risk_level: Default risk level for new assessments.
        max_polygon_vertices: Maximum vertices allowed in plot polygon.
        coordinate_precision: Decimal precision for GPS coordinates.
        require_polygon_above_hectares: Hectare threshold requiring polygon geolocation.
        country_risk_weight: Weight for country risk in composite score.
        commodity_risk_weight: Weight for commodity risk in composite score.
        supplier_risk_weight: Weight for supplier risk in composite score.
        traceability_risk_weight: Weight for traceability risk in composite score.
        high_risk_threshold: Score threshold above which risk is classified as high.
        low_risk_threshold: Score threshold below which risk is classified as low.
        dds_validity_days: Number of days a due diligence statement remains valid.
        enable_digital_signing: Whether to digitally sign due diligence statements.
        eu_system_url: Base URL for the EU Information System API.
        eu_system_sandbox: Whether to use the EU System sandbox environment.
        eu_system_timeout_seconds: Timeout for EU System API calls.
        batch_max_size: Maximum number of records per batch operation.
        batch_worker_count: Number of parallel workers for batch processing.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        log_level: Logging level for the EUDR traceability service.
        retention_years: Number of years to retain records for audit compliance.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""

    # -- EUDR defaults -------------------------------------------------------
    deforestation_cutoff_date: str = "2020-12-31"
    default_risk_level: str = "standard"

    # -- Plot settings -------------------------------------------------------
    max_polygon_vertices: int = 500
    coordinate_precision: int = 6
    require_polygon_above_hectares: float = 4.0

    # -- Risk scoring --------------------------------------------------------
    country_risk_weight: float = 0.30
    commodity_risk_weight: float = 0.20
    supplier_risk_weight: float = 0.25
    traceability_risk_weight: float = 0.25
    high_risk_threshold: float = 70.0
    low_risk_threshold: float = 30.0

    # -- DDS settings --------------------------------------------------------
    dds_validity_days: int = 365
    enable_digital_signing: bool = True

    # -- EU Information System -----------------------------------------------
    eu_system_url: str = ""
    eu_system_sandbox: bool = True
    eu_system_timeout_seconds: int = 60

    # -- Batch processing ----------------------------------------------------
    batch_max_size: int = 1000
    batch_worker_count: int = 4

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Record retention ----------------------------------------------------
    retention_years: int = 5

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> EUDRTraceabilityConfig:
        """Build an EUDRTraceabilityConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_TRACEABILITY_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated EUDRTraceabilityConfig instance.
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
            s3_bucket_url=_str("S3_BUCKET_URL", cls.s3_bucket_url),
            deforestation_cutoff_date=_str(
                "DEFORESTATION_CUTOFF_DATE",
                cls.deforestation_cutoff_date,
            ),
            default_risk_level=_str(
                "DEFAULT_RISK_LEVEL", cls.default_risk_level,
            ),
            max_polygon_vertices=_int(
                "MAX_POLYGON_VERTICES", cls.max_polygon_vertices,
            ),
            coordinate_precision=_int(
                "COORDINATE_PRECISION", cls.coordinate_precision,
            ),
            require_polygon_above_hectares=_float(
                "REQUIRE_POLYGON_ABOVE_HECTARES",
                cls.require_polygon_above_hectares,
            ),
            country_risk_weight=_float(
                "COUNTRY_RISK_WEIGHT", cls.country_risk_weight,
            ),
            commodity_risk_weight=_float(
                "COMMODITY_RISK_WEIGHT", cls.commodity_risk_weight,
            ),
            supplier_risk_weight=_float(
                "SUPPLIER_RISK_WEIGHT", cls.supplier_risk_weight,
            ),
            traceability_risk_weight=_float(
                "TRACEABILITY_RISK_WEIGHT", cls.traceability_risk_weight,
            ),
            high_risk_threshold=_float(
                "HIGH_RISK_THRESHOLD", cls.high_risk_threshold,
            ),
            low_risk_threshold=_float(
                "LOW_RISK_THRESHOLD", cls.low_risk_threshold,
            ),
            dds_validity_days=_int(
                "DDS_VALIDITY_DAYS", cls.dds_validity_days,
            ),
            enable_digital_signing=_bool(
                "ENABLE_DIGITAL_SIGNING",
                cls.enable_digital_signing,
            ),
            eu_system_url=_str(
                "EU_SYSTEM_URL", cls.eu_system_url,
            ),
            eu_system_sandbox=_bool(
                "EU_SYSTEM_SANDBOX", cls.eu_system_sandbox,
            ),
            eu_system_timeout_seconds=_int(
                "EU_SYSTEM_TIMEOUT_SECONDS",
                cls.eu_system_timeout_seconds,
            ),
            batch_max_size=_int(
                "BATCH_MAX_SIZE", cls.batch_max_size,
            ),
            batch_worker_count=_int(
                "BATCH_WORKER_COUNT", cls.batch_worker_count,
            ),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            log_level=_str("LOG_LEVEL", cls.log_level),
            retention_years=_int(
                "RETENTION_YEARS", cls.retention_years,
            ),
        )

        logger.info(
            "EUDRTraceabilityConfig loaded: cutoff=%s, risk_level=%s, "
            "max_vertices=%d, precision=%d, polygon_threshold=%.1fha, "
            "risk_weights=[%.2f,%.2f,%.2f,%.2f], "
            "high_risk=%.1f, low_risk=%.1f, "
            "dds_validity=%dd, signing=%s, sandbox=%s, "
            "batch_size=%d, workers=%d, retention=%dy",
            config.deforestation_cutoff_date,
            config.default_risk_level,
            config.max_polygon_vertices,
            config.coordinate_precision,
            config.require_polygon_above_hectares,
            config.country_risk_weight,
            config.commodity_risk_weight,
            config.supplier_risk_weight,
            config.traceability_risk_weight,
            config.high_risk_threshold,
            config.low_risk_threshold,
            config.dds_validity_days,
            config.enable_digital_signing,
            config.eu_system_sandbox,
            config.batch_max_size,
            config.batch_worker_count,
            config.retention_years,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[EUDRTraceabilityConfig] = None
_config_lock = threading.Lock()


def get_config() -> EUDRTraceabilityConfig:
    """Return the singleton EUDRTraceabilityConfig, creating from env if needed.

    Returns:
        EUDRTraceabilityConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = EUDRTraceabilityConfig.from_env()
    return _config_instance


def set_config(config: EUDRTraceabilityConfig) -> None:
    """Replace the singleton EUDRTraceabilityConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("EUDRTraceabilityConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "EUDRTraceabilityConfig",
    "get_config",
    "set_config",
    "reset_config",
]
