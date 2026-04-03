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
    >>> from greenlang.agents.data.deforestation_satellite.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_satellite, cfg.eudr_cutoff_date)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
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

_ENV_PREFIX = "GL_DEFORESTATION_SAT_"


# ---------------------------------------------------------------------------
# DeforestationSatelliteConfig
# ---------------------------------------------------------------------------


@dataclass
class DeforestationSatelliteConfig(BaseDataConfig):
    """Configuration for the GreenLang Deforestation Satellite Connector Agent SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only satellite-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_DEFORESTATION_SAT_`` prefix.

    Attributes:
        eudr_cutoff_date: EU Deforestation Regulation cutoff date (ISO format).
        default_satellite: Default satellite source for image acquisition.
        max_cloud_cover: Maximum acceptable cloud cover percentage.
        ndvi_clearcut_threshold: NDVI delta threshold indicating clear-cut.
        ndvi_degradation_threshold: NDVI delta threshold indicating degradation.
        ndvi_partial_loss_threshold: NDVI delta threshold indicating partial loss.
        ndvi_regrowth_threshold: NDVI delta threshold indicating regrowth.
        min_alert_confidence: Minimum confidence level for alerts.
        alert_dedup_radius_m: Radius in meters for alert deduplication.
        alert_dedup_days: Time window in days for alert deduplication.
        baseline_sample_points: Number of sample points for baseline assessment.
        batch_size: Number of items to process per batch.
        worker_count: Number of parallel workers for batch processing.
        cache_ttl_seconds: Time-to-live in seconds for cached results.
        retention_days: Number of days to retain satellite imagery metadata.
        use_mock: Whether to use mock satellite data providers.
        gfw_api_key: API key for Global Forest Watch (GFW) alert service.
        copernicus_api_key: API key for Copernicus Sentinel Hub service.
    """

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

    # -- Batch processing (satellite-specific) -------------------------------
    batch_size: int = 50
    worker_count: int = 4

    # -- Cache ---------------------------------------------------------------
    cache_ttl_seconds: int = 3600

    # -- Data retention ------------------------------------------------------
    retention_days: int = 730

    # -- Feature toggles -----------------------------------------------------
    use_mock: bool = True

    # -- External API keys ---------------------------------------------------
    gfw_api_key: str = ""
    copernicus_api_key: str = ""

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DeforestationSatelliteConfig:
        """Build a DeforestationSatelliteConfig from environment variables.

        Every field can be overridden via ``GL_DEFORESTATION_SAT_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated DeforestationSatelliteConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            eudr_cutoff_date=env.str(
                "EUDR_CUTOFF_DATE", cls.eudr_cutoff_date,
            ),
            default_satellite=env.str(
                "DEFAULT_SATELLITE", cls.default_satellite,
            ),
            max_cloud_cover=env.int(
                "MAX_CLOUD_COVER", cls.max_cloud_cover,
            ),
            ndvi_clearcut_threshold=env.float(
                "NDVI_CLEARCUT_THRESHOLD", cls.ndvi_clearcut_threshold,
            ),
            ndvi_degradation_threshold=env.float(
                "NDVI_DEGRADATION_THRESHOLD", cls.ndvi_degradation_threshold,
            ),
            ndvi_partial_loss_threshold=env.float(
                "NDVI_PARTIAL_LOSS_THRESHOLD", cls.ndvi_partial_loss_threshold,
            ),
            ndvi_regrowth_threshold=env.float(
                "NDVI_REGROWTH_THRESHOLD", cls.ndvi_regrowth_threshold,
            ),
            min_alert_confidence=env.str(
                "MIN_ALERT_CONFIDENCE", cls.min_alert_confidence,
            ),
            alert_dedup_radius_m=env.int(
                "ALERT_DEDUP_RADIUS_M", cls.alert_dedup_radius_m,
            ),
            alert_dedup_days=env.int(
                "ALERT_DEDUP_DAYS", cls.alert_dedup_days,
            ),
            baseline_sample_points=env.int(
                "BASELINE_SAMPLE_POINTS", cls.baseline_sample_points,
            ),
            batch_size=env.int("BATCH_SIZE", cls.batch_size),
            worker_count=env.int("WORKER_COUNT", cls.worker_count),
            cache_ttl_seconds=env.int(
                "CACHE_TTL_SECONDS", cls.cache_ttl_seconds,
            ),
            retention_days=env.int(
                "RETENTION_DAYS", cls.retention_days,
            ),
            use_mock=env.bool("USE_MOCK", cls.use_mock),
            gfw_api_key=env.str("GFW_API_KEY", cls.gfw_api_key),
            copernicus_api_key=env.str(
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

get_config, set_config, reset_config = create_config_singleton(
    DeforestationSatelliteConfig, _ENV_PREFIX,
)

__all__ = [
    "DeforestationSatelliteConfig",
    "get_config",
    "set_config",
    "reset_config",
]
