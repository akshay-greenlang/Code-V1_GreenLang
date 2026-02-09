# -*- coding: utf-8 -*-
"""
GIS/Mapping Connector Agent Service Configuration - AGENT-DATA-006: GIS Connector

Centralized configuration for the GIS/Mapping Connector Agent SDK covering:
- Database, Redis, and S3 connection URLs
- Default CRS and coordinate precision
- Feature limits (max features, max file size)
- Spatial analysis defaults (buffer distance, simplification tolerance)
- Geocoding cache TTL
- Batch processing (batch size, worker count)
- Connection pool sizing
- Data retention policy
- Feature toggles (raster support, 3D support)
- Earth geometry constants (radius, polygon limits, minimum area)
- Logging level

All settings can be overridden via environment variables with the
``GL_GIS_CONNECTOR_`` prefix (e.g. ``GL_GIS_CONNECTOR_DEFAULT_CRS``).

Example:
    >>> from greenlang.gis_connector.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_crs, cfg.max_features)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-006 GIS/Mapping Connector Agent
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

_ENV_PREFIX = "GL_GIS_CONNECTOR_"


# ---------------------------------------------------------------------------
# GISConnectorConfig
# ---------------------------------------------------------------------------


@dataclass
class GISConnectorConfig:
    """Complete configuration for the GreenLang GIS/Mapping Connector Agent SDK.

    Attributes are grouped by concern: connections, coordinate reference,
    feature constraints, spatial analysis defaults, geocoding, batch
    processing, connection pool, data retention, feature toggles,
    earth geometry constants, and logging.

    All attributes can be overridden via environment variables using the
    ``GL_GIS_CONNECTOR_`` prefix.

    Attributes:
        database_url: PostgreSQL/PostGIS connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        s3_bucket_url: S3 bucket URL for geospatial file storage.
        log_level: Logging level for the GIS connector service.
        default_crs: Default coordinate reference system (EPSG code).
        max_features: Maximum number of features per layer or query result.
        max_file_size_mb: Maximum geospatial file size in megabytes.
        coordinate_precision: Decimal places for coordinate rounding.
        buffer_distance_default: Default buffer distance in meters for
            spatial buffer operations.
        geocoding_cache_ttl: TTL in seconds for geocoding result cache
            entries (default 24 hours).
        simplification_tolerance: Default geometry simplification tolerance
            in CRS units (degrees for WGS-84).
        batch_size: Number of features to process per batch.
        worker_count: Number of parallel workers for batch processing.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        retention_days: Number of days to retain operation logs and layers.
        enable_raster: Whether to enable raster data support (GeoTIFF, etc.).
        enable_3d: Whether to enable 3D coordinate support (altitude/Z).
        earth_radius_meters: Mean Earth radius in meters used for
            Haversine and spherical calculations.
        max_polygon_vertices: Maximum allowed vertices per polygon geometry.
        min_area_sq_meters: Minimum area in square meters for valid polygons.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Coordinate reference system -----------------------------------------
    default_crs: str = "EPSG:4326"

    # -- Feature constraints -------------------------------------------------
    max_features: int = 100000
    max_file_size_mb: int = 500

    # -- Coordinate precision ------------------------------------------------
    coordinate_precision: int = 6

    # -- Spatial analysis defaults -------------------------------------------
    buffer_distance_default: float = 1000.0
    simplification_tolerance: float = 0.001

    # -- Geocoding -----------------------------------------------------------
    geocoding_cache_ttl: int = 86400

    # -- Batch processing ----------------------------------------------------
    batch_size: int = 1000
    worker_count: int = 4

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Data retention ------------------------------------------------------
    retention_days: int = 365

    # -- Feature toggles -----------------------------------------------------
    enable_raster: bool = False
    enable_3d: bool = False

    # -- Earth geometry constants --------------------------------------------
    earth_radius_meters: float = 6371000.0
    max_polygon_vertices: int = 10000
    min_area_sq_meters: float = 1.0

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> GISConnectorConfig:
        """Build a GISConnectorConfig from environment variables.

        Every field can be overridden via ``GL_GIS_CONNECTOR_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated GISConnectorConfig instance.
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
            default_crs=_str("DEFAULT_CRS", cls.default_crs),
            max_features=_int(
                "MAX_FEATURES", cls.max_features,
            ),
            max_file_size_mb=_int(
                "MAX_FILE_SIZE_MB", cls.max_file_size_mb,
            ),
            coordinate_precision=_int(
                "COORDINATE_PRECISION", cls.coordinate_precision,
            ),
            buffer_distance_default=_float(
                "BUFFER_DISTANCE_DEFAULT", cls.buffer_distance_default,
            ),
            geocoding_cache_ttl=_int(
                "GEOCODING_CACHE_TTL", cls.geocoding_cache_ttl,
            ),
            simplification_tolerance=_float(
                "SIMPLIFICATION_TOLERANCE",
                cls.simplification_tolerance,
            ),
            batch_size=_int(
                "BATCH_SIZE", cls.batch_size,
            ),
            worker_count=_int(
                "WORKER_COUNT", cls.worker_count,
            ),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            retention_days=_int(
                "RETENTION_DAYS", cls.retention_days,
            ),
            enable_raster=_bool(
                "ENABLE_RASTER", cls.enable_raster,
            ),
            enable_3d=_bool(
                "ENABLE_3D", cls.enable_3d,
            ),
            earth_radius_meters=_float(
                "EARTH_RADIUS_METERS", cls.earth_radius_meters,
            ),
            max_polygon_vertices=_int(
                "MAX_POLYGON_VERTICES", cls.max_polygon_vertices,
            ),
            min_area_sq_meters=_float(
                "MIN_AREA_SQ_METERS", cls.min_area_sq_meters,
            ),
        )

        logger.info(
            "GISConnectorConfig loaded: crs=%s, max_features=%d, "
            "max_file=%dMB, precision=%d, buffer=%.1fm, "
            "geocode_ttl=%ds, simplify=%.4f, batch=%d/%d workers, "
            "pool=%d-%d, retention=%dd, raster=%s, 3d=%s, "
            "earth_r=%.0fm, max_vertices=%d, min_area=%.1fm2",
            config.default_crs,
            config.max_features,
            config.max_file_size_mb,
            config.coordinate_precision,
            config.buffer_distance_default,
            config.geocoding_cache_ttl,
            config.simplification_tolerance,
            config.batch_size,
            config.worker_count,
            config.pool_min_size,
            config.pool_max_size,
            config.retention_days,
            config.enable_raster,
            config.enable_3d,
            config.earth_radius_meters,
            config.max_polygon_vertices,
            config.min_area_sq_meters,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[GISConnectorConfig] = None
_config_lock = threading.Lock()


def get_config() -> GISConnectorConfig:
    """Return the singleton GISConnectorConfig, creating from env if needed.

    Returns:
        GISConnectorConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = GISConnectorConfig.from_env()
    return _config_instance


def set_config(config: GISConnectorConfig) -> None:
    """Replace the singleton GISConnectorConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("GISConnectorConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "GISConnectorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
