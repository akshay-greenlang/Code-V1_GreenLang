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
    >>> from greenlang.agents.data.eudr_traceability.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.deforestation_cutoff_date, cfg.high_risk_threshold)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 EUDR Traceability Connector
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

_ENV_PREFIX = "GL_EUDR_TRACEABILITY_"


# ---------------------------------------------------------------------------
# EUDRTraceabilityConfig
# ---------------------------------------------------------------------------


@dataclass
class EUDRTraceabilityConfig(BaseDataConfig):
    """Configuration for the GreenLang EUDR Traceability Connector SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only EUDR-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_EUDR_TRACEABILITY_`` prefix.

    Attributes:
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
        retention_years: Number of years to retain records for audit compliance.
    """

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

    # -- Batch processing (EUDR-specific) ------------------------------------
    batch_max_size: int = 1000

    # -- Record retention ----------------------------------------------------
    retention_years: int = 5

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> EUDRTraceabilityConfig:
        """Build an EUDRTraceabilityConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_TRACEABILITY_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated EUDRTraceabilityConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            deforestation_cutoff_date=env.str(
                "DEFORESTATION_CUTOFF_DATE",
                cls.deforestation_cutoff_date,
            ),
            default_risk_level=env.str(
                "DEFAULT_RISK_LEVEL", cls.default_risk_level,
            ),
            max_polygon_vertices=env.int(
                "MAX_POLYGON_VERTICES", cls.max_polygon_vertices,
            ),
            coordinate_precision=env.int(
                "COORDINATE_PRECISION", cls.coordinate_precision,
            ),
            require_polygon_above_hectares=env.float(
                "REQUIRE_POLYGON_ABOVE_HECTARES",
                cls.require_polygon_above_hectares,
            ),
            country_risk_weight=env.float(
                "COUNTRY_RISK_WEIGHT", cls.country_risk_weight,
            ),
            commodity_risk_weight=env.float(
                "COMMODITY_RISK_WEIGHT", cls.commodity_risk_weight,
            ),
            supplier_risk_weight=env.float(
                "SUPPLIER_RISK_WEIGHT", cls.supplier_risk_weight,
            ),
            traceability_risk_weight=env.float(
                "TRACEABILITY_RISK_WEIGHT", cls.traceability_risk_weight,
            ),
            high_risk_threshold=env.float(
                "HIGH_RISK_THRESHOLD", cls.high_risk_threshold,
            ),
            low_risk_threshold=env.float(
                "LOW_RISK_THRESHOLD", cls.low_risk_threshold,
            ),
            dds_validity_days=env.int(
                "DDS_VALIDITY_DAYS", cls.dds_validity_days,
            ),
            enable_digital_signing=env.bool(
                "ENABLE_DIGITAL_SIGNING",
                cls.enable_digital_signing,
            ),
            eu_system_url=env.str(
                "EU_SYSTEM_URL", cls.eu_system_url,
            ),
            eu_system_sandbox=env.bool(
                "EU_SYSTEM_SANDBOX", cls.eu_system_sandbox,
            ),
            eu_system_timeout_seconds=env.int(
                "EU_SYSTEM_TIMEOUT_SECONDS",
                cls.eu_system_timeout_seconds,
            ),
            batch_max_size=env.int(
                "BATCH_MAX_SIZE", cls.batch_max_size,
            ),
            retention_years=env.int(
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

get_config, set_config, reset_config = create_config_singleton(
    EUDRTraceabilityConfig, _ENV_PREFIX,
)

__all__ = [
    "EUDRTraceabilityConfig",
    "get_config",
    "set_config",
    "reset_config",
]
