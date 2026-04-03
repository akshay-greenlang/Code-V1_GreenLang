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
    >>> from greenlang.agents.data.gis_connector.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_crs, cfg.max_features)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-006 GIS/Mapping Connector Agent
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

_ENV_PREFIX = "GL_GIS_CONNECTOR_"


# ---------------------------------------------------------------------------
# GISConnectorConfig
# ---------------------------------------------------------------------------


@dataclass
class GISConnectorConfig(BaseDataConfig):
    """Configuration for the GreenLang GIS/Mapping Connector Agent SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only GIS-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_GIS_CONNECTOR_`` prefix.

    Attributes:
        default_crs: Default coordinate reference system (EPSG code).
        max_features: Maximum number of features per layer or query result.
        max_file_size_mb: Maximum geospatial file size in megabytes.
        coordinate_precision: Decimal places for coordinate rounding.
        buffer_distance_default: Default buffer distance in meters.
        simplification_tolerance: Default geometry simplification tolerance.
        geocoding_cache_ttl: TTL in seconds for geocoding result cache entries.
        batch_size: Number of features to process per batch.
        worker_count: Number of parallel workers for batch processing.
        retention_days: Number of days to retain operation logs and layers.
        enable_raster: Whether to enable raster data support.
        enable_3d: Whether to enable 3D coordinate support.
        earth_radius_meters: Mean Earth radius in meters.
        max_polygon_vertices: Maximum allowed vertices per polygon geometry.
        min_area_sq_meters: Minimum area in square meters for valid polygons.
    """

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

    # -- Batch processing (GIS-specific) -------------------------------------
    batch_size: int = 1000
    worker_count: int = 4

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
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> GISConnectorConfig:
        """Build a GISConnectorConfig from environment variables.

        Every field can be overridden via ``GL_GIS_CONNECTOR_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated GISConnectorConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            default_crs=env.str("DEFAULT_CRS", cls.default_crs),
            max_features=env.int(
                "MAX_FEATURES", cls.max_features,
            ),
            max_file_size_mb=env.int(
                "MAX_FILE_SIZE_MB", cls.max_file_size_mb,
            ),
            coordinate_precision=env.int(
                "COORDINATE_PRECISION", cls.coordinate_precision,
            ),
            buffer_distance_default=env.float(
                "BUFFER_DISTANCE_DEFAULT", cls.buffer_distance_default,
            ),
            geocoding_cache_ttl=env.int(
                "GEOCODING_CACHE_TTL", cls.geocoding_cache_ttl,
            ),
            simplification_tolerance=env.float(
                "SIMPLIFICATION_TOLERANCE",
                cls.simplification_tolerance,
            ),
            batch_size=env.int(
                "BATCH_SIZE", cls.batch_size,
            ),
            worker_count=env.int(
                "WORKER_COUNT", cls.worker_count,
            ),
            retention_days=env.int(
                "RETENTION_DAYS", cls.retention_days,
            ),
            enable_raster=env.bool(
                "ENABLE_RASTER", cls.enable_raster,
            ),
            enable_3d=env.bool(
                "ENABLE_3D", cls.enable_3d,
            ),
            earth_radius_meters=env.float(
                "EARTH_RADIUS_METERS", cls.earth_radius_meters,
            ),
            max_polygon_vertices=env.int(
                "MAX_POLYGON_VERTICES", cls.max_polygon_vertices,
            ),
            min_area_sq_meters=env.float(
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

get_config, set_config, reset_config = create_config_singleton(
    GISConnectorConfig, _ENV_PREFIX,
)

__all__ = [
    "GISConnectorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
