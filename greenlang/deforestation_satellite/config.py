# -*- coding: utf-8 -*-
"""
Deforestation Satellite Connector Agent Service Configuration - AGENT-DATA-007: GL-DATA-GEO-003

Centralized configuration for the Deforestation Satellite Connector Agent SDK covering:
- Database and Redis connection URLs
- EUDR regulatory cutoff date (2020-12-31)
- Default satellite source and cloud cover thresholds
- NDVI change detection thresholds (clear-cut, degradation, partial loss, regrowth)
- Alert confidence, deduplication radius, and deduplication window
- Baseline assessment sample point count
- Batch processing (batch size, worker count)
- Connection pool sizing
- Cache TTL and data retention policy
- Feature toggle (use_mock for development mode)
- External API keys (GFW, Copernicus)
- Logging level

All settings can be overridden via environment variables with the
``GL_DEFORESTATION_SAT_`` prefix (e.g. ``GL_DEFORESTATION_SAT_DEFAULT_SATELLITE``).

Example:
    >>> from greenlang.deforestation_satellite.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_satellite, cfg.eudr_cutoff_date)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
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

_ENV_PREFIX = "GL_DEFORESTATION_SAT_"


# ---------------------------------------------------------------------------
# DeforestationSatelliteConfig
# ---------------------------------------------------------------------------


@dataclass
class DeforestationSatelliteConfig:
    """Complete configuration for the GreenLang Deforestation Satellite Connector Agent SDK.

    Attributes are grouped by concern: connections, EUDR regulation,
    satellite defaults, NDVI thresholds, alert settings, baseline
    assessment, batch processing, connection pool, caching, data
    retention, feature toggles, external API keys, and logging.

    All attributes can be overridden via environment variables using the
    ``GL_DEFORESTATION_SAT_`` prefix.

    Attributes:
        database_url: PostgreSQL/TimescaleDB connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        log_level: Logging level for the deforestation satellite service.
        eudr_cutoff_date: EU Deforestation Regulation cutoff date (ISO format).
            Land cleared after this date is non-compliant.
        default_satellite: Default satellite source for image acquisition
            (sentinel2, landsat8, landsat9, modis, harmonized).
        max_cloud_cover: Maximum acceptable cloud cover percentage for
            satellite scene selection (0-100).
        ndvi_clearcut_threshold: NDVI delta threshold indicating clear-cut
            deforestation (severe vegetation loss).
        ndvi_degradation_threshold: NDVI delta threshold indicating forest
            degradation (moderate vegetation loss).
        ndvi_partial_loss_threshold: NDVI delta threshold indicating partial
            canopy loss (minor vegetation loss).
        ndvi_regrowth_threshold: NDVI delta threshold indicating vegetation
            regrowth (positive change).
        min_alert_confidence: Minimum confidence level for deforestation
            alerts to process (low, nominal, high).
        alert_dedup_radius_m: Radius in meters for alert deduplication
            spatial clustering.
        alert_dedup_days: Time window in days for alert deduplication.
        baseline_sample_points: Number of sample points for polygon
            baseline assessment (grid sampling).
        batch_size: Number of items to process per batch in pipeline stages.
        worker_count: Number of parallel workers for batch processing.
        cache_ttl_seconds: Time-to-live in seconds for cached results.
        pool_min_size: Minimum database connection pool size.
        pool_max_size: Maximum database connection pool size.
        retention_days: Number of days to retain satellite imagery metadata
            and assessment records (default 730 = 2 years).
        use_mock: Whether to use mock satellite data providers for
            development and testing (parsed from string).
        gfw_api_key: API key for Global Forest Watch (GFW) alert service.
        copernicus_api_key: API key for Copernicus Sentinel Hub service.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- EUDR regulation -----------------------------------------------------
    eudr_cutoff_date: str = "2020-12-31"

    # -- Satellite defaults --------------------------------------------------
    default_satellite: str = "sentinel2"
    max_cloud_cover: int = 30

    # -- NDVI change detection thresholds ------------------------------------
    ndvi_clearcut_threshold: float = -0.3
    ndvi_degradation_threshold: float = -0.15
    ndvi_partial_loss_threshold: float = -0.05
    ndvi_regrowth_threshold: float = 0.1

    # -- Alert settings ------------------------------------------------------
    min_alert_confidence: str = "nominal"
    alert_dedup_radius_m: int = 100
    alert_dedup_days: int = 7

    # -- Baseline assessment -------------------------------------------------
    baseline_sample_points: int = 9

    # -- Batch processing ----------------------------------------------------
    batch_size: int = 50
    worker_count: int = 4

    # -- Cache ---------------------------------------------------------------
    cache_ttl_seconds: int = 3600

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Data retention ------------------------------------------------------
    retention_days: int = 730

    # -- Feature toggles -----------------------------------------------------
    use_mock: bool = True

    # -- External API keys ---------------------------------------------------
    gfw_api_key: str = ""
    copernicus_api_key: str = ""

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DeforestationSatelliteConfig:
        """Build a DeforestationSatelliteConfig from environment variables.

        Every field can be overridden via ``GL_DEFORESTATION_SAT_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated DeforestationSatelliteConfig instance.
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
            log_level=_str("LOG_LEVEL", cls.log_level),
            eudr_cutoff_date=_str(
                "EUDR_CUTOFF_DATE", cls.eudr_cutoff_date,
            ),
            default_satellite=_str(
                "DEFAULT_SATELLITE", cls.default_satellite,
            ),
            max_cloud_cover=_int(
                "MAX_CLOUD_COVER", cls.max_cloud_cover,
            ),
            ndvi_clearcut_threshold=_float(
                "NDVI_CLEARCUT_THRESHOLD", cls.ndvi_clearcut_threshold,
            ),
            ndvi_degradation_threshold=_float(
                "NDVI_DEGRADATION_THRESHOLD", cls.ndvi_degradation_threshold,
            ),
            ndvi_partial_loss_threshold=_float(
                "NDVI_PARTIAL_LOSS_THRESHOLD", cls.ndvi_partial_loss_threshold,
            ),
            ndvi_regrowth_threshold=_float(
                "NDVI_REGROWTH_THRESHOLD", cls.ndvi_regrowth_threshold,
            ),
            min_alert_confidence=_str(
                "MIN_ALERT_CONFIDENCE", cls.min_alert_confidence,
            ),
            alert_dedup_radius_m=_int(
                "ALERT_DEDUP_RADIUS_M", cls.alert_dedup_radius_m,
            ),
            alert_dedup_days=_int(
                "ALERT_DEDUP_DAYS", cls.alert_dedup_days,
            ),
            baseline_sample_points=_int(
                "BASELINE_SAMPLE_POINTS", cls.baseline_sample_points,
            ),
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            worker_count=_int("WORKER_COUNT", cls.worker_count),
            cache_ttl_seconds=_int(
                "CACHE_TTL_SECONDS", cls.cache_ttl_seconds,
            ),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            retention_days=_int(
                "RETENTION_DAYS", cls.retention_days,
            ),
            use_mock=_bool("USE_MOCK", cls.use_mock),
            gfw_api_key=_str("GFW_API_KEY", cls.gfw_api_key),
            copernicus_api_key=_str(
                "COPERNICUS_API_KEY", cls.copernicus_api_key,
            ),
        )

        logger.info(
            "DeforestationSatelliteConfig loaded: satellite=%s, "
            "cloud_cover<=%d%%, eudr_cutoff=%s, "
            "ndvi_thresholds=[%.2f/%.2f/%.2f/%.2f], "
            "alert_confidence=%s, dedup=%dm/%dd, "
            "baseline_pts=%d, batch=%d/%d workers, "
            "cache_ttl=%ds, pool=%d-%d, retention=%dd, "
            "mock=%s, gfw_key=%s, copernicus_key=%s",
            config.default_satellite,
            config.max_cloud_cover,
            config.eudr_cutoff_date,
            config.ndvi_clearcut_threshold,
            config.ndvi_degradation_threshold,
            config.ndvi_partial_loss_threshold,
            config.ndvi_regrowth_threshold,
            config.min_alert_confidence,
            config.alert_dedup_radius_m,
            config.alert_dedup_days,
            config.baseline_sample_points,
            config.batch_size,
            config.worker_count,
            config.cache_ttl_seconds,
            config.pool_min_size,
            config.pool_max_size,
            config.retention_days,
            config.use_mock,
            "***" if config.gfw_api_key else "(unset)",
            "***" if config.copernicus_api_key else "(unset)",
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[DeforestationSatelliteConfig] = None
_config_lock = threading.Lock()


def get_config() -> DeforestationSatelliteConfig:
    """Return the singleton DeforestationSatelliteConfig, creating from env if needed.

    Returns:
        DeforestationSatelliteConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = DeforestationSatelliteConfig.from_env()
    return _config_instance


def set_config(config: DeforestationSatelliteConfig) -> None:
    """Replace the singleton DeforestationSatelliteConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("DeforestationSatelliteConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "DeforestationSatelliteConfig",
    "get_config",
    "set_config",
    "reset_config",
]
