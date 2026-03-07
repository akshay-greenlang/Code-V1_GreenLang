# -*- coding: utf-8 -*-
"""
Satellite Monitoring Configuration - AGENT-EUDR-003

Centralized configuration for the Satellite Monitoring Agent covering:
- Satellite data source API credentials (Sentinel-2, Landsat, GFW)
- EUDR deforestation cutoff date and baseline window parameters
- Cloud cover filtering thresholds (standard and absolute maximum)
- NDVI change detection thresholds (deforestation, degradation, regrowth)
- Minimum change area detection sensitivity
- Multi-source fusion weights (Sentinel-2, Landsat, GFW)
- Monitoring pipeline concurrency and batch size limits
- Cache TTL settings (imagery cache and baseline snapshot cache)
- Analysis level timeouts (quick, standard, deep)
- Alert generation confidence thresholds
- Seasonal adjustment and SAR integration toggles
- Database and cache connection settings
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_SAT_`` prefix (e.g. ``GL_EUDR_SAT_DATABASE_URL``,
``GL_EUDR_SAT_CLOUD_COVER_MAX``).

Environment Variable Reference (GL_EUDR_SAT_ prefix):
    GL_EUDR_SAT_DATABASE_URL                  - PostgreSQL connection URL
    GL_EUDR_SAT_REDIS_URL                     - Redis connection URL
    GL_EUDR_SAT_LOG_LEVEL                     - Logging level (DEBUG/INFO/WARNING/ERROR)
    GL_EUDR_SAT_SENTINEL2_CLIENT_ID           - Copernicus Data Space client ID
    GL_EUDR_SAT_SENTINEL2_CLIENT_SECRET       - Copernicus Data Space client secret
    GL_EUDR_SAT_LANDSAT_API_KEY               - USGS EarthExplorer / Landsat API key
    GL_EUDR_SAT_GFW_API_KEY                   - Global Forest Watch API key
    GL_EUDR_SAT_CUTOFF_DATE                   - EUDR deforestation cutoff date
    GL_EUDR_SAT_BASELINE_WINDOW_DAYS          - Days of imagery for baseline compositing
    GL_EUDR_SAT_CLOUD_COVER_MAX               - Max cloud cover % for scene selection
    GL_EUDR_SAT_CLOUD_COVER_ABSOLUTE_MAX      - Absolute max cloud cover % (never accept above)
    GL_EUDR_SAT_NDVI_DEFORESTATION_THRESHOLD  - NDVI delta for deforestation classification
    GL_EUDR_SAT_NDVI_DEGRADATION_THRESHOLD    - NDVI delta for degradation classification
    GL_EUDR_SAT_REGROWTH_THRESHOLD            - NDVI delta for regrowth classification
    GL_EUDR_SAT_MIN_CHANGE_AREA_HA            - Minimum detectable change area in hectares
    GL_EUDR_SAT_SENTINEL2_WEIGHT              - Fusion weight for Sentinel-2 source
    GL_EUDR_SAT_LANDSAT_WEIGHT                - Fusion weight for Landsat source
    GL_EUDR_SAT_GFW_WEIGHT                    - Fusion weight for GFW alerts source
    GL_EUDR_SAT_MONITORING_MAX_CONCURRENCY    - Max concurrent monitoring operations
    GL_EUDR_SAT_CACHE_TTL_SECONDS             - Cache TTL for imagery metadata
    GL_EUDR_SAT_BASELINE_CACHE_TTL_SECONDS    - Cache TTL for baseline snapshots (90 days)
    GL_EUDR_SAT_QUICK_TIMEOUT_SECONDS         - Timeout for quick analysis level
    GL_EUDR_SAT_STANDARD_TIMEOUT_SECONDS      - Timeout for standard analysis level
    GL_EUDR_SAT_DEEP_TIMEOUT_SECONDS          - Timeout for deep analysis level
    GL_EUDR_SAT_MAX_BATCH_SIZE                - Maximum plots in a single batch analysis
    GL_EUDR_SAT_ALERT_CONFIDENCE_THRESHOLD    - Minimum confidence for alert generation
    GL_EUDR_SAT_SEASONAL_ADJUSTMENT_ENABLED   - Enable seasonal NDVI normalization
    GL_EUDR_SAT_SAR_ENABLED                   - Enable Sentinel-1 SAR backscatter fusion
    GL_EUDR_SAT_ENABLE_PROVENANCE             - Enable SHA-256 provenance chain
    GL_EUDR_SAT_GENESIS_HASH                  - Genesis anchor string for provenance
    GL_EUDR_SAT_ENABLE_METRICS                - Enable Prometheus metrics export
    GL_EUDR_SAT_POOL_SIZE                     - Database connection pool size
    GL_EUDR_SAT_RATE_LIMIT                    - Max API requests per minute

Example:
    >>> from greenlang.agents.eudr.satellite_monitoring.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.cloud_cover_max)
    20.0

    >>> # Override for testing
    >>> from greenlang.agents.eudr.satellite_monitoring.config import (
    ...     set_config, reset_config, SatelliteMonitoringConfig,
    ... )
    >>> set_config(SatelliteMonitoringConfig(cloud_cover_max=30.0))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_SAT_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Default fusion weights (must sum to 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_FUSION_WEIGHTS: Dict[str, float] = {
    "sentinel2": 0.50,
    "landsat": 0.30,
    "gfw": 0.20,
}


# ---------------------------------------------------------------------------
# SatelliteMonitoringConfig
# ---------------------------------------------------------------------------


@dataclass
class SatelliteMonitoringConfig:
    """Complete configuration for the EUDR Satellite Monitoring Agent.

    Attributes are grouped by concern: connections, logging, satellite
    API credentials, cutoff and baseline, cloud cover filtering, NDVI
    change detection thresholds, minimum change area, multi-source
    fusion weights, monitoring pipeline, caching, analysis level
    timeouts, alert generation, seasonal and SAR toggles, provenance
    tracking, and metrics export.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_SAT_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage
            of monitoring results, baselines, and alert history.
        redis_url: Redis connection URL for imagery metadata caching
            and rate limit tracking.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        sentinel2_client_id: Copernicus Data Space Ecosystem (CDSE)
            OAuth2 client ID for Sentinel-2 imagery access.
        sentinel2_client_secret: CDSE OAuth2 client secret for
            Sentinel-2 imagery authentication.
        landsat_api_key: USGS EarthExplorer / Landsat API key for
            Landsat 8/9 scene retrieval.
        gfw_api_key: Global Forest Watch (GFW) API key for accessing
            deforestation alert data (GLAD, RADD).
        cutoff_date: EUDR deforestation cutoff date string in ISO
            format (YYYY-MM-DD). Per Article 2(1), this is
            December 31, 2020.
        baseline_window_days: Number of days of cloud-free imagery
            composited to establish a spectral baseline snapshot
            around the cutoff date.
        cloud_cover_max: Maximum acceptable cloud cover percentage
            for scene selection in standard analysis. Scenes above
            this threshold are deprioritized.
        cloud_cover_absolute_max: Absolute ceiling for cloud cover
            percentage. Scenes above this are never accepted.
        ndvi_deforestation_threshold: NDVI delta threshold (negative)
            for classifying a pixel as deforestation. A value of
            -0.15 means NDVI must decrease by at least 0.15.
        ndvi_degradation_threshold: NDVI delta threshold (negative)
            for classifying a pixel as degradation (less severe than
            deforestation). Value of -0.05.
        regrowth_threshold: NDVI delta threshold (positive) for
            classifying a pixel as showing regrowth. Value of 0.10.
        min_change_area_ha: Minimum contiguous change area in
            hectares to qualify as a reportable event. Below this
            threshold, changes are considered noise.
        sentinel2_weight: Weight of Sentinel-2 source in multi-source
            fusion analysis (0.0-1.0).
        landsat_weight: Weight of Landsat source in multi-source
            fusion analysis (0.0-1.0).
        gfw_weight: Weight of GFW alerts source in multi-source
            fusion analysis (0.0-1.0).
        monitoring_max_concurrency: Maximum number of plots monitored
            concurrently in the monitoring pipeline.
        cache_ttl_seconds: Time-to-live in seconds for cached imagery
            metadata and scene search results.
        baseline_cache_ttl_seconds: Time-to-live in seconds for
            cached baseline snapshots (default 90 days / 7,776,000s).
        quick_timeout_seconds: Maximum time in seconds for QUICK
            analysis level per plot.
        standard_timeout_seconds: Maximum time in seconds for
            STANDARD analysis level per plot.
        deep_timeout_seconds: Maximum time in seconds for DEEP
            analysis level per plot.
        max_batch_size: Maximum number of plots in a single batch
            analysis request.
        alert_confidence_threshold: Minimum confidence score (0.0-1.0)
            required before generating a deforestation alert.
        seasonal_adjustment_enabled: Enable seasonal NDVI
            normalization to reduce false positives from phenological
            cycles (deciduous forests, dry/wet seasons).
        sar_enabled: Enable Sentinel-1 SAR backscatter analysis for
            cloud-gap filling and wet-season monitoring when optical
            imagery is unavailable.
        enable_provenance: Enable SHA-256 provenance chain tracking
            for all satellite monitoring operations.
        genesis_hash: Anchor string for the provenance chain, unique
            to the Satellite Monitoring agent.
        enable_metrics: Enable Prometheus metrics export under the
            ``gl_eudr_sat_`` prefix.
        pool_size: PostgreSQL connection pool size.
        rate_limit: Maximum inbound API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Satellite API credentials -------------------------------------------
    sentinel2_client_id: str = ""
    sentinel2_client_secret: str = ""
    landsat_api_key: str = ""
    gfw_api_key: str = ""

    # -- Cutoff and baseline -------------------------------------------------
    cutoff_date: str = "2020-12-31"
    baseline_window_days: int = 90

    # -- Cloud cover filtering -----------------------------------------------
    cloud_cover_max: float = 20.0
    cloud_cover_absolute_max: float = 50.0

    # -- NDVI change detection thresholds ------------------------------------
    ndvi_deforestation_threshold: float = -0.15
    ndvi_degradation_threshold: float = -0.05
    regrowth_threshold: float = 0.10

    # -- Minimum change area -------------------------------------------------
    min_change_area_ha: float = 0.1

    # -- Multi-source fusion weights (must sum to 1.0) -----------------------
    sentinel2_weight: float = 0.50
    landsat_weight: float = 0.30
    gfw_weight: float = 0.20

    # -- Monitoring pipeline -------------------------------------------------
    monitoring_max_concurrency: int = 50

    # -- Caching -------------------------------------------------------------
    cache_ttl_seconds: int = 86400
    baseline_cache_ttl_seconds: int = 7776000

    # -- Analysis level timeouts ---------------------------------------------
    quick_timeout_seconds: float = 10.0
    standard_timeout_seconds: float = 30.0
    deep_timeout_seconds: float = 120.0

    # -- Batch analysis ------------------------------------------------------
    max_batch_size: int = 10_000

    # -- Alert generation ----------------------------------------------------
    alert_confidence_threshold: float = 0.7

    # -- Seasonal and SAR toggles --------------------------------------------
    seasonal_adjustment_enabled: bool = True
    sar_enabled: bool = True

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-SAT-003-SATELLITE-MONITORING-GENESIS"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 10
    rate_limit: int = 1000

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization.

        Performs range checks on all numeric fields, enumeration checks
        on string fields, fusion weight sum validation, and threshold
        ordering checks. Collects all errors before raising a single
        ValueError with all violations listed.

        Raises:
            ValueError: If any configuration value is outside its valid
                range or violates a constraint.
        """
        errors: list[str] = []

        # -- Logging ---------------------------------------------------------
        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )
        else:
            self.log_level = normalised_log

        # -- Cutoff and baseline ---------------------------------------------
        if self.baseline_window_days <= 0:
            errors.append(
                f"baseline_window_days must be > 0, "
                f"got {self.baseline_window_days}"
            )

        # -- Cloud cover filtering -------------------------------------------
        if not (0.0 <= self.cloud_cover_max <= 100.0):
            errors.append(
                f"cloud_cover_max must be in [0.0, 100.0], "
                f"got {self.cloud_cover_max}"
            )
        if not (0.0 <= self.cloud_cover_absolute_max <= 100.0):
            errors.append(
                f"cloud_cover_absolute_max must be in [0.0, 100.0], "
                f"got {self.cloud_cover_absolute_max}"
            )
        if self.cloud_cover_max > self.cloud_cover_absolute_max:
            errors.append(
                f"cloud_cover_max ({self.cloud_cover_max}) must be "
                f"<= cloud_cover_absolute_max ({self.cloud_cover_absolute_max})"
            )

        # -- NDVI thresholds -------------------------------------------------
        if not (-1.0 <= self.ndvi_deforestation_threshold <= 0.0):
            errors.append(
                f"ndvi_deforestation_threshold must be in [-1.0, 0.0], "
                f"got {self.ndvi_deforestation_threshold}"
            )
        if not (-1.0 <= self.ndvi_degradation_threshold <= 0.0):
            errors.append(
                f"ndvi_degradation_threshold must be in [-1.0, 0.0], "
                f"got {self.ndvi_degradation_threshold}"
            )
        if self.ndvi_deforestation_threshold > self.ndvi_degradation_threshold:
            errors.append(
                f"ndvi_deforestation_threshold ({self.ndvi_deforestation_threshold}) "
                f"must be <= ndvi_degradation_threshold "
                f"({self.ndvi_degradation_threshold})"
            )
        if not (0.0 <= self.regrowth_threshold <= 1.0):
            errors.append(
                f"regrowth_threshold must be in [0.0, 1.0], "
                f"got {self.regrowth_threshold}"
            )

        # -- Minimum change area ---------------------------------------------
        if self.min_change_area_ha < 0:
            errors.append(
                f"min_change_area_ha must be >= 0, "
                f"got {self.min_change_area_ha}"
            )

        # -- Fusion weights --------------------------------------------------
        for name, value in [
            ("sentinel2_weight", self.sentinel2_weight),
            ("landsat_weight", self.landsat_weight),
            ("gfw_weight", self.gfw_weight),
        ]:
            if not (0.0 <= value <= 1.0):
                errors.append(
                    f"{name} must be in [0.0, 1.0], got {value}"
                )
        weight_sum = self.sentinel2_weight + self.landsat_weight + self.gfw_weight
        if abs(weight_sum - 1.0) > 0.001:
            errors.append(
                f"Fusion weights (sentinel2 + landsat + gfw) must sum "
                f"to 1.0, got {weight_sum:.4f}"
            )

        # -- Monitoring pipeline ---------------------------------------------
        if not (1 <= self.monitoring_max_concurrency <= 1000):
            errors.append(
                f"monitoring_max_concurrency must be in [1, 1000], "
                f"got {self.monitoring_max_concurrency}"
            )

        # -- Caching ---------------------------------------------------------
        if self.cache_ttl_seconds <= 0:
            errors.append(
                f"cache_ttl_seconds must be > 0, "
                f"got {self.cache_ttl_seconds}"
            )
        if self.baseline_cache_ttl_seconds <= 0:
            errors.append(
                f"baseline_cache_ttl_seconds must be > 0, "
                f"got {self.baseline_cache_ttl_seconds}"
            )

        # -- Timeouts --------------------------------------------------------
        for name, value in [
            ("quick_timeout_seconds", self.quick_timeout_seconds),
            ("standard_timeout_seconds", self.standard_timeout_seconds),
            ("deep_timeout_seconds", self.deep_timeout_seconds),
        ]:
            if value <= 0:
                errors.append(f"{name} must be > 0, got {value}")

        if self.quick_timeout_seconds >= self.standard_timeout_seconds:
            errors.append(
                f"quick_timeout_seconds ({self.quick_timeout_seconds}) must be "
                f"< standard_timeout_seconds ({self.standard_timeout_seconds})"
            )
        if self.standard_timeout_seconds >= self.deep_timeout_seconds:
            errors.append(
                f"standard_timeout_seconds ({self.standard_timeout_seconds}) must be "
                f"< deep_timeout_seconds ({self.deep_timeout_seconds})"
            )

        # -- Batch analysis --------------------------------------------------
        if not (1 <= self.max_batch_size <= 100_000):
            errors.append(
                f"max_batch_size must be in [1, 100000], "
                f"got {self.max_batch_size}"
            )

        # -- Alert generation ------------------------------------------------
        if not (0.0 <= self.alert_confidence_threshold <= 1.0):
            errors.append(
                f"alert_confidence_threshold must be in [0.0, 1.0], "
                f"got {self.alert_confidence_threshold}"
            )

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        # -- Performance tuning ----------------------------------------------
        if self.pool_size <= 0:
            errors.append(f"pool_size must be > 0, got {self.pool_size}")
        if self.rate_limit <= 0:
            errors.append(f"rate_limit must be > 0, got {self.rate_limit}")

        if errors:
            raise ValueError(
                "SatelliteMonitoringConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "SatelliteMonitoringConfig validated successfully: "
            "cutoff=%s, baseline_window=%dd, "
            "cloud_max=%.1f%%, cloud_abs_max=%.1f%%, "
            "ndvi_deforest=%.2f, ndvi_degrad=%.2f, regrowth=%.2f, "
            "min_area=%.2fha, "
            "weights=S2:%.2f/LS:%.2f/GFW:%.2f, "
            "max_concurrency=%d, cache_ttl=%ds, "
            "alert_conf=%.2f, seasonal=%s, sar=%s, "
            "provenance=%s, metrics=%s",
            self.cutoff_date,
            self.baseline_window_days,
            self.cloud_cover_max,
            self.cloud_cover_absolute_max,
            self.ndvi_deforestation_threshold,
            self.ndvi_degradation_threshold,
            self.regrowth_threshold,
            self.min_change_area_ha,
            self.sentinel2_weight,
            self.landsat_weight,
            self.gfw_weight,
            self.monitoring_max_concurrency,
            self.cache_ttl_seconds,
            self.alert_confidence_threshold,
            self.seasonal_adjustment_enabled,
            self.sar_enabled,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> SatelliteMonitoringConfig:
        """Build a SatelliteMonitoringConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_SAT_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated SatelliteMonitoringConfig instance, validated
            via ``__post_init__``.
        """
        prefix = _ENV_PREFIX

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _env(name)
            if val is None:
                return default
            return val.strip().lower() in ("true", "1", "yes")

        def _int(name: str, default: int) -> int:
            val = _env(name)
            if val is None:
                return default
            try:
                return int(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid integer for %s%s=%r, using default %d",
                    prefix, name, val, default,
                )
                return default

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid float for %s%s=%r, using default %f",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val.strip()

        config = cls(
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Satellite API credentials
            sentinel2_client_id=_str(
                "SENTINEL2_CLIENT_ID", cls.sentinel2_client_id,
            ),
            sentinel2_client_secret=_str(
                "SENTINEL2_CLIENT_SECRET", cls.sentinel2_client_secret,
            ),
            landsat_api_key=_str(
                "LANDSAT_API_KEY", cls.landsat_api_key,
            ),
            gfw_api_key=_str(
                "GFW_API_KEY", cls.gfw_api_key,
            ),
            # Cutoff and baseline
            cutoff_date=_str("CUTOFF_DATE", cls.cutoff_date),
            baseline_window_days=_int(
                "BASELINE_WINDOW_DAYS", cls.baseline_window_days,
            ),
            # Cloud cover filtering
            cloud_cover_max=_float(
                "CLOUD_COVER_MAX", cls.cloud_cover_max,
            ),
            cloud_cover_absolute_max=_float(
                "CLOUD_COVER_ABSOLUTE_MAX",
                cls.cloud_cover_absolute_max,
            ),
            # NDVI thresholds
            ndvi_deforestation_threshold=_float(
                "NDVI_DEFORESTATION_THRESHOLD",
                cls.ndvi_deforestation_threshold,
            ),
            ndvi_degradation_threshold=_float(
                "NDVI_DEGRADATION_THRESHOLD",
                cls.ndvi_degradation_threshold,
            ),
            regrowth_threshold=_float(
                "REGROWTH_THRESHOLD", cls.regrowth_threshold,
            ),
            # Minimum change area
            min_change_area_ha=_float(
                "MIN_CHANGE_AREA_HA", cls.min_change_area_ha,
            ),
            # Fusion weights
            sentinel2_weight=_float(
                "SENTINEL2_WEIGHT", cls.sentinel2_weight,
            ),
            landsat_weight=_float(
                "LANDSAT_WEIGHT", cls.landsat_weight,
            ),
            gfw_weight=_float(
                "GFW_WEIGHT", cls.gfw_weight,
            ),
            # Monitoring pipeline
            monitoring_max_concurrency=_int(
                "MONITORING_MAX_CONCURRENCY",
                cls.monitoring_max_concurrency,
            ),
            # Caching
            cache_ttl_seconds=_int(
                "CACHE_TTL_SECONDS", cls.cache_ttl_seconds,
            ),
            baseline_cache_ttl_seconds=_int(
                "BASELINE_CACHE_TTL_SECONDS",
                cls.baseline_cache_ttl_seconds,
            ),
            # Timeouts
            quick_timeout_seconds=_float(
                "QUICK_TIMEOUT_SECONDS", cls.quick_timeout_seconds,
            ),
            standard_timeout_seconds=_float(
                "STANDARD_TIMEOUT_SECONDS",
                cls.standard_timeout_seconds,
            ),
            deep_timeout_seconds=_float(
                "DEEP_TIMEOUT_SECONDS", cls.deep_timeout_seconds,
            ),
            # Batch analysis
            max_batch_size=_int(
                "MAX_BATCH_SIZE", cls.max_batch_size,
            ),
            # Alert generation
            alert_confidence_threshold=_float(
                "ALERT_CONFIDENCE_THRESHOLD",
                cls.alert_confidence_threshold,
            ),
            # Seasonal and SAR toggles
            seasonal_adjustment_enabled=_bool(
                "SEASONAL_ADJUSTMENT_ENABLED",
                cls.seasonal_adjustment_enabled,
            ),
            sar_enabled=_bool(
                "SAR_ENABLED", cls.sar_enabled,
            ),
            # Provenance
            enable_provenance=_bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Metrics
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
            # Performance tuning
            pool_size=_int("POOL_SIZE", cls.pool_size),
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
        )

        logger.info(
            "SatelliteMonitoringConfig loaded: "
            "cutoff=%s, baseline_window=%dd, "
            "cloud_max=%.1f%%, cloud_abs_max=%.1f%%, "
            "ndvi_deforest=%.2f, ndvi_degrad=%.2f, regrowth=%.2f, "
            "min_area=%.2fha, "
            "weights=S2:%.2f/LS:%.2f/GFW:%.2f, "
            "max_concurrency=%d, "
            "cache_ttl=%ds, baseline_cache_ttl=%ds, "
            "quick_timeout=%.1fs, standard_timeout=%.1fs, deep_timeout=%.1fs, "
            "max_batch=%d, alert_conf=%.2f, "
            "seasonal=%s, sar=%s, "
            "provenance=%s, pool=%d, rate_limit=%d/min, metrics=%s",
            config.cutoff_date,
            config.baseline_window_days,
            config.cloud_cover_max,
            config.cloud_cover_absolute_max,
            config.ndvi_deforestation_threshold,
            config.ndvi_degradation_threshold,
            config.regrowth_threshold,
            config.min_change_area_ha,
            config.sentinel2_weight,
            config.landsat_weight,
            config.gfw_weight,
            config.monitoring_max_concurrency,
            config.cache_ttl_seconds,
            config.baseline_cache_ttl_seconds,
            config.quick_timeout_seconds,
            config.standard_timeout_seconds,
            config.deep_timeout_seconds,
            config.max_batch_size,
            config.alert_confidence_threshold,
            config.seasonal_adjustment_enabled,
            config.sar_enabled,
            config.enable_provenance,
            config.pool_size,
            config.rate_limit,
            config.enable_metrics,
        )
        return config

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def timeout_by_level(self) -> Dict[str, float]:
        """Return analysis timeout mapping by level name.

        Returns:
            Dictionary with keys: quick, standard, deep.
        """
        return {
            "quick": self.quick_timeout_seconds,
            "standard": self.standard_timeout_seconds,
            "deep": self.deep_timeout_seconds,
        }

    @property
    def fusion_weights(self) -> Dict[str, float]:
        """Return fusion weight mapping by source name.

        Returns:
            Dictionary with keys: sentinel2, landsat, gfw.
        """
        return {
            "sentinel2": self.sentinel2_weight,
            "landsat": self.landsat_weight,
            "gfw": self.gfw_weight,
        }

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain Python dictionary.

        Sensitive connection strings (database_url, redis_url) and
        API credentials (sentinel2_client_id, sentinel2_client_secret,
        landsat_api_key, gfw_api_key) are redacted to prevent
        accidental credential leakage in logs, exception tracebacks,
        and monitoring dashboards.

        Returns:
            Dictionary representation with sensitive fields redacted.
        """
        return {
            # Connections (redacted)
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # Logging
            "log_level": self.log_level,
            # Satellite API credentials (redacted)
            "sentinel2_client_id": "***" if self.sentinel2_client_id else "",
            "sentinel2_client_secret": "***" if self.sentinel2_client_secret else "",
            "landsat_api_key": "***" if self.landsat_api_key else "",
            "gfw_api_key": "***" if self.gfw_api_key else "",
            # Cutoff and baseline
            "cutoff_date": self.cutoff_date,
            "baseline_window_days": self.baseline_window_days,
            # Cloud cover filtering
            "cloud_cover_max": self.cloud_cover_max,
            "cloud_cover_absolute_max": self.cloud_cover_absolute_max,
            # NDVI thresholds
            "ndvi_deforestation_threshold": self.ndvi_deforestation_threshold,
            "ndvi_degradation_threshold": self.ndvi_degradation_threshold,
            "regrowth_threshold": self.regrowth_threshold,
            # Minimum change area
            "min_change_area_ha": self.min_change_area_ha,
            # Fusion weights
            "sentinel2_weight": self.sentinel2_weight,
            "landsat_weight": self.landsat_weight,
            "gfw_weight": self.gfw_weight,
            # Monitoring pipeline
            "monitoring_max_concurrency": self.monitoring_max_concurrency,
            # Caching
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "baseline_cache_ttl_seconds": self.baseline_cache_ttl_seconds,
            # Timeouts
            "quick_timeout_seconds": self.quick_timeout_seconds,
            "standard_timeout_seconds": self.standard_timeout_seconds,
            "deep_timeout_seconds": self.deep_timeout_seconds,
            # Batch analysis
            "max_batch_size": self.max_batch_size,
            # Alert generation
            "alert_confidence_threshold": self.alert_confidence_threshold,
            # Seasonal and SAR toggles
            "seasonal_adjustment_enabled": self.seasonal_adjustment_enabled,
            "sar_enabled": self.sar_enabled,
            # Provenance
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # Metrics
            "enable_metrics": self.enable_metrics,
            # Performance tuning
            "pool_size": self.pool_size,
            "rate_limit": self.rate_limit,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Returns:
            String representation with sensitive fields redacted.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"SatelliteMonitoringConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[SatelliteMonitoringConfig] = None
_config_lock = threading.Lock()


def get_config() -> SatelliteMonitoringConfig:
    """Return the singleton SatelliteMonitoringConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_SAT_*`` environment variables.

    Returns:
        SatelliteMonitoringConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.cloud_cover_max
        20.0
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = SatelliteMonitoringConfig.from_env()
    return _config_instance


def set_config(config: SatelliteMonitoringConfig) -> None:
    """Replace the singleton SatelliteMonitoringConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New SatelliteMonitoringConfig to install.

    Example:
        >>> cfg = SatelliteMonitoringConfig(cloud_cover_max=30.0)
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "SatelliteMonitoringConfig replaced programmatically: "
        "cloud_max=%.1f%%, ndvi_deforest=%.2f, max_concurrency=%d",
        config.cloud_cover_max,
        config.ndvi_deforestation_threshold,
        config.monitoring_max_concurrency,
    )


def reset_config() -> None:
    """Reset the singleton SatelliteMonitoringConfig to None.

    The next call to get_config() will re-read GL_EUDR_SAT_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("SatelliteMonitoringConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "SatelliteMonitoringConfig",
    "get_config",
    "set_config",
    "reset_config",
]
