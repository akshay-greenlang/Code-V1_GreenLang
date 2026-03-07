# -*- coding: utf-8 -*-
"""
Plot Boundary Manager Configuration - AGENT-EUDR-006

Centralized configuration for the Plot Boundary Manager Agent covering:
- Canonical and supported coordinate reference systems (CRS)
- EUDR Article 9 polygon vs point area threshold (4.0 hectares)
- Polygon vertex limits and geometry repair settings
- Self-intersection repair, sliver detection, spike removal
- Simplification tolerances and area deviation limits
- Overlap detection thresholds (minor/moderate/major/critical)
- Boundary versioning retention per EUDR Article 31 (5 years)
- Batch processing size and concurrency limits
- R-tree spatial index page size for overlap scanning
- Export format and coordinate precision settings
- Geodetic computation settings (Karney algorithm, WGS84 ellipsoid)
- UTM zone auto-detection and coordinate precision digits
- Polygon area limits (min/max hectares)
- Ring closure and hole containment tolerances
- Split/merge area conservation tolerance
- Database and cache connection settings
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_PBM_`` prefix (e.g. ``GL_EUDR_PBM_DATABASE_URL``,
``GL_EUDR_PBM_AREA_THRESHOLD_HECTARES``, ``GL_EUDR_PBM_CANONICAL_CRS``).

Environment Variable Reference (GL_EUDR_PBM_ prefix):
    GL_EUDR_PBM_DATABASE_URL                    - PostgreSQL connection URL
    GL_EUDR_PBM_REDIS_URL                       - Redis connection URL
    GL_EUDR_PBM_LOG_LEVEL                       - Logging level (DEBUG/INFO/WARNING/ERROR)
    GL_EUDR_PBM_CANONICAL_CRS                   - Canonical CRS (e.g. EPSG:4326)
    GL_EUDR_PBM_AREA_THRESHOLD_HECTARES         - EUDR polygon vs point threshold
    GL_EUDR_PBM_MAX_VERTICES_PER_POLYGON        - Max vertices per polygon
    GL_EUDR_PBM_MIN_VERTICES_POLYGON            - Min vertices for a polygon
    GL_EUDR_PBM_DUPLICATE_VERTEX_TOLERANCE_M    - Duplicate vertex tolerance (metres)
    GL_EUDR_PBM_SELF_INTERSECTION_REPAIR        - Enable self-intersection repair
    GL_EUDR_PBM_SLIVER_ASPECT_RATIO_THRESHOLD   - Sliver aspect ratio threshold
    GL_EUDR_PBM_SPIKE_ANGLE_THRESHOLD_DEG       - Spike angle threshold (degrees)
    GL_EUDR_PBM_SIMPLIFICATION_TOLERANCE        - Simplification tolerance (degrees)
    GL_EUDR_PBM_SIMPLIFICATION_AREA_DEV_MAX     - Max area deviation after simplification
    GL_EUDR_PBM_OVERLAP_MIN_AREA_M2             - Min overlap area to report (sq metres)
    GL_EUDR_PBM_OVERLAP_MINOR_THRESHOLD         - Minor overlap threshold (fraction)
    GL_EUDR_PBM_OVERLAP_MODERATE_THRESHOLD      - Moderate overlap threshold (fraction)
    GL_EUDR_PBM_OVERLAP_MAJOR_THRESHOLD         - Major overlap threshold (fraction)
    GL_EUDR_PBM_VERSION_RETENTION_YEARS         - Version retention (years, Article 31)
    GL_EUDR_PBM_BATCH_MAX_SIZE                  - Maximum batch size
    GL_EUDR_PBM_BATCH_CONCURRENCY               - Batch concurrency level
    GL_EUDR_PBM_R_TREE_PAGE_SIZE                - R-tree spatial index page size
    GL_EUDR_PBM_EXPORT_DEFAULT_PRECISION        - Export coordinate precision (digits)
    GL_EUDR_PBM_KARNEY_ALGORITHM_ENABLED        - Enable Karney geodesic algorithm
    GL_EUDR_PBM_UTM_ZONE_AUTO_DETECT            - Enable UTM zone auto-detection
    GL_EUDR_PBM_COORDINATE_PRECISION_DIGITS     - Internal coordinate precision (digits)
    GL_EUDR_PBM_MAX_POLYGON_AREA_HECTARES       - Maximum polygon area (hectares)
    GL_EUDR_PBM_MIN_POLYGON_AREA_HECTARES       - Minimum polygon area (hectares)
    GL_EUDR_PBM_RING_CLOSURE_TOLERANCE_M        - Ring closure tolerance (metres)
    GL_EUDR_PBM_HOLE_CONTAINMENT_BUFFER_M       - Hole containment buffer (metres)
    GL_EUDR_PBM_SPLIT_MERGE_AREA_TOLERANCE      - Split/merge area tolerance (fraction)
    GL_EUDR_PBM_CACHE_TTL_SECONDS               - General cache TTL (seconds)
    GL_EUDR_PBM_GENESIS_HASH                    - Genesis anchor for provenance chain
    GL_EUDR_PBM_ENABLE_METRICS                  - Enable Prometheus metrics export

Example:
    >>> from greenlang.agents.eudr.plot_boundary.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.canonical_crs)
    EPSG:4326

    >>> # Override for testing
    >>> from greenlang.agents.eudr.plot_boundary.config import (
    ...     set_config, reset_config, PlotBoundaryConfig,
    ... )
    >>> set_config(PlotBoundaryConfig(area_threshold_hectares=5.0))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_PBM_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Supported input CRS codes (50+ EPSG codes)
# ---------------------------------------------------------------------------

_DEFAULT_SUPPORTED_INPUT_CRS: List[str] = [
    # WGS84 geographic
    "EPSG:4326",
    # WGS84 pseudo-Mercator (Web Mercator)
    "EPSG:3857",
    # UTM North zones 1-60
    "EPSG:32601", "EPSG:32602", "EPSG:32603", "EPSG:32604", "EPSG:32605",
    "EPSG:32606", "EPSG:32607", "EPSG:32608", "EPSG:32609", "EPSG:32610",
    "EPSG:32611", "EPSG:32612", "EPSG:32613", "EPSG:32614", "EPSG:32615",
    "EPSG:32616", "EPSG:32617", "EPSG:32618", "EPSG:32619", "EPSG:32620",
    # UTM South zones 1-20
    "EPSG:32701", "EPSG:32702", "EPSG:32703", "EPSG:32704", "EPSG:32705",
    "EPSG:32706", "EPSG:32707", "EPSG:32708", "EPSG:32709", "EPSG:32710",
    "EPSG:32711", "EPSG:32712", "EPSG:32713", "EPSG:32714", "EPSG:32715",
    "EPSG:32716", "EPSG:32717", "EPSG:32718", "EPSG:32719", "EPSG:32720",
    # SIRGAS 2000 (South America)
    "EPSG:4674",
    # SIRGAS 2000 / UTM zones
    "EPSG:31981", "EPSG:31982", "EPSG:31983", "EPSG:31984", "EPSG:31985",
    # ETRS89 (Europe)
    "EPSG:4258",
    # ETRS89 / LAEA Europe
    "EPSG:3035",
    # ETRS89 / UTM zones
    "EPSG:25830", "EPSG:25831", "EPSG:25832", "EPSG:25833", "EPSG:25834",
    # NAD83 (North America)
    "EPSG:4269",
    # GDA2020 (Australia)
    "EPSG:7844",
    # GDA2020 / MGA zones
    "EPSG:7854", "EPSG:7855", "EPSG:7856",
]

# ---------------------------------------------------------------------------
# Default export formats
# ---------------------------------------------------------------------------

_DEFAULT_EXPORT_FORMATS: List[str] = [
    "geojson",
    "kml",
    "wkt",
    "wkb",
    "shapefile",
    "eudr_xml",
    "gpx",
    "gml",
]

# ---------------------------------------------------------------------------
# Valid export formats
# ---------------------------------------------------------------------------

_VALID_EXPORT_FORMATS = frozenset({
    "geojson",
    "kml",
    "wkt",
    "wkb",
    "shapefile",
    "eudr_xml",
    "gpx",
    "gml",
})

# ---------------------------------------------------------------------------
# Valid simplification methods
# ---------------------------------------------------------------------------

_VALID_SIMPLIFICATION_METHODS = frozenset({
    "douglas_peucker",
    "visvalingam_whyatt",
    "topology_preserving",
})

# ---------------------------------------------------------------------------
# EUDR-regulated commodities (Article 1(1))
# ---------------------------------------------------------------------------

_DEFAULT_EUDR_COMMODITIES: List[str] = [
    "cattle",
    "cocoa",
    "coffee",
    "palm_oil",
    "rubber",
    "soya",
    "wood",
]


# ---------------------------------------------------------------------------
# PlotBoundaryConfig
# ---------------------------------------------------------------------------


@dataclass
class PlotBoundaryConfig:
    """Complete configuration for the EUDR Plot Boundary Manager Agent.

    Attributes are grouped by concern: connections, logging, CRS settings,
    geometry constraints, validation and repair settings, simplification,
    overlap detection, versioning, batch processing, spatial indexing,
    export settings, geodetic computation, coordinate precision, polygon
    area limits, ring and hole tolerances, split/merge settings,
    EUDR commodity scope, caching, provenance tracking, and metrics export.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_PBM_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage
            of plot boundaries, versions, and compliance records.
        redis_url: Redis connection URL for boundary caching and
            spatial query acceleration.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        canonical_crs: Canonical coordinate reference system for all
            internal storage and processing. Default is EPSG:4326 (WGS84).
        supported_input_crs: List of EPSG codes accepted for input
            boundaries. Boundaries in these CRS values are automatically
            reprojected to the canonical CRS.
        area_threshold_hectares: EUDR Article 9 threshold for polygon
            requirement. Plots above this area require full polygon
            boundaries; plots below may use a single point with
            geolocation. Default is 4.0 hectares.
        max_vertices_per_polygon: Maximum number of vertices allowed
            in a single polygon boundary. Exceeding this triggers
            automatic simplification. Default is 100000.
        min_vertices_polygon: Minimum number of vertices for a valid
            polygon (exterior ring). Must be at least 4 to form a
            closed triangle. Default is 4.
        duplicate_vertex_tolerance_meters: Distance in metres below
            which consecutive vertices are considered duplicates and
            collapsed. Default is 0.01 metres (1 cm).
        self_intersection_repair_enabled: Whether to automatically
            repair self-intersecting polygons using node insertion and
            ring restructuring. Default is True.
        sliver_aspect_ratio_threshold: Maximum length-to-width ratio
            for a polygon before it is flagged as a sliver. Default
            is 20.0.
        spike_angle_threshold_degrees: Minimum interior angle in
            degrees below which a vertex is flagged as a spike and
            optionally removed. Default is 1.0 degrees.
        simplification_default_tolerance: Default tolerance for polygon
            simplification in degrees (for WGS84 coordinates). Default
            is 0.0001 degrees (approximately 11 metres at the equator).
        simplification_area_deviation_max: Maximum allowed area change
            as a fraction of original area after simplification. Default
            is 0.01 (1%).
        overlap_detection_min_area_m2: Minimum overlap area in square
            metres to be reported. Overlaps below this threshold are
            treated as noise. Default is 100.0 sq metres.
        overlap_minor_threshold: Overlap area fraction threshold for
            MINOR severity classification. Default is 0.01 (1%).
        overlap_moderate_threshold: Overlap area fraction threshold for
            MODERATE severity. Default is 0.10 (10%).
        overlap_major_threshold: Overlap area fraction threshold for
            MAJOR severity. Default is 0.50 (50%).
        version_retention_years: Number of years to retain boundary
            version history, per EUDR Article 31 record-keeping
            requirements. Default is 5 years.
        batch_max_size: Maximum number of boundaries processed in a
            single batch operation. Default is 10000.
        batch_concurrency: Number of concurrent workers for batch
            operations. Default is 4.
        r_tree_page_size: Page size for R-tree spatial index used in
            overlap detection scanning. Default is 50.
        export_default_precision: Number of decimal places for
            coordinate values in exported boundary files. Default
            is 8 digits.
        export_formats: List of supported export format identifiers.
        karney_algorithm_enabled: Whether to use the Karney geodesic
            algorithm for high-precision area and perimeter calculations
            on the WGS84 ellipsoid. Default is True.
        wgs84_semi_major_axis: WGS84 ellipsoid semi-major axis in
            metres. Default is 6378137.0 metres.
        wgs84_flattening: WGS84 ellipsoid flattening. Default is
            1/298.257223563.
        utm_zone_auto_detect: Whether to automatically detect the
            appropriate UTM zone for metric area calculations based
            on the polygon centroid. Default is True.
        coordinate_precision_digits: Number of decimal places for
            internal coordinate representation. Default is 8 digits.
        max_polygon_area_hectares: Maximum allowed polygon area in
            hectares. Boundaries exceeding this are rejected. Default
            is 50000.0 hectares.
        min_polygon_area_hectares: Minimum allowed polygon area in
            hectares. Boundaries below this are flagged. Default
            is 0.0001 hectares (1 sq metre).
        ring_closure_tolerance_meters: Maximum gap in metres between
            the first and last vertices of a ring before the ring is
            considered unclosed. Default is 0.1 metres (10 cm).
        hole_containment_buffer_meters: Buffer distance in metres
            applied when checking that holes are contained within the
            exterior ring. Default is 0.0 metres (strict containment).
        split_merge_area_tolerance: Maximum allowed area discrepancy
            as a fraction of the original area when splitting or
            merging boundaries. Default is 0.001 (0.1%).
        eudr_commodities: List of EUDR-regulated commodity identifiers
            per Article 1(1) of EU 2023/1115.
        batch_size: Alias for batch_max_size for compatibility.
        max_concurrent_jobs: Maximum number of concurrent batch jobs.
        cache_ttl_seconds: Time-to-live in seconds for cached boundary
            query results. Default is 3600 seconds (1 hour).
        genesis_hash: Anchor string for the SHA-256 provenance chain,
            unique to the Plot Boundary Manager agent.
        enable_metrics: Enable Prometheus metrics export under the
            ``gl_eudr_pbm_`` prefix.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- CRS settings --------------------------------------------------------
    canonical_crs: str = "EPSG:4326"
    supported_input_crs: List[str] = field(
        default_factory=lambda: list(_DEFAULT_SUPPORTED_INPUT_CRS)
    )

    # -- EUDR polygon vs point threshold (Article 9) -------------------------
    area_threshold_hectares: float = 4.0

    # -- Polygon vertex limits -----------------------------------------------
    max_vertices_per_polygon: int = 100000
    min_vertices_polygon: int = 4

    # -- Geometry repair settings --------------------------------------------
    duplicate_vertex_tolerance_meters: float = 0.01
    self_intersection_repair_enabled: bool = True
    sliver_aspect_ratio_threshold: float = 20.0
    spike_angle_threshold_degrees: float = 1.0

    # -- Simplification settings ---------------------------------------------
    simplification_default_tolerance: float = 0.0001
    simplification_area_deviation_max: float = 0.01

    # -- Overlap detection thresholds ----------------------------------------
    overlap_detection_min_area_m2: float = 100.0
    overlap_minor_threshold: float = 0.01
    overlap_moderate_threshold: float = 0.10
    overlap_major_threshold: float = 0.50

    # -- Boundary versioning (Article 31) ------------------------------------
    version_retention_years: int = 5

    # -- Batch processing ----------------------------------------------------
    batch_max_size: int = 10000
    batch_concurrency: int = 4

    # -- Spatial indexing ----------------------------------------------------
    r_tree_page_size: int = 50

    # -- Export settings -----------------------------------------------------
    export_default_precision: int = 8
    export_formats: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EXPORT_FORMATS)
    )

    # -- Geodetic computation settings ---------------------------------------
    karney_algorithm_enabled: bool = True
    wgs84_semi_major_axis: float = 6378137.0
    wgs84_flattening: float = 1.0 / 298.257223563
    utm_zone_auto_detect: bool = True

    # -- Coordinate precision ------------------------------------------------
    coordinate_precision_digits: int = 8

    # -- Polygon area limits -------------------------------------------------
    max_polygon_area_hectares: float = 50000.0
    min_polygon_area_hectares: float = 0.0001

    # -- Ring and hole tolerances --------------------------------------------
    ring_closure_tolerance_meters: float = 0.1
    hole_containment_buffer_meters: float = 0.0

    # -- Split/merge settings ------------------------------------------------
    split_merge_area_tolerance: float = 0.001

    # -- EUDR commodity scope ------------------------------------------------
    eudr_commodities: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EUDR_COMMODITIES)
    )

    # -- Performance tuning --------------------------------------------------
    batch_size: int = 10000
    max_concurrent_jobs: int = 10

    # -- Caching -------------------------------------------------------------
    cache_ttl_seconds: int = 3600

    # -- Provenance tracking -------------------------------------------------
    genesis_hash: str = "GL-EUDR-PBM-006-PLOT-BOUNDARY-MANAGER-GENESIS"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization.

        Performs range checks on all numeric fields, enumeration checks
        on string fields, and logical consistency checks. Collects all
        errors before raising a single ValueError with all violations
        listed.

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

        # -- CRS settings ---------------------------------------------------
        if not self.canonical_crs:
            errors.append("canonical_crs must not be empty")

        if not self.supported_input_crs:
            errors.append("supported_input_crs must not be empty")

        if self.canonical_crs not in self.supported_input_crs:
            errors.append(
                f"canonical_crs '{self.canonical_crs}' must be included "
                f"in supported_input_crs"
            )

        # -- EUDR threshold --------------------------------------------------
        if self.area_threshold_hectares <= 0.0:
            errors.append(
                f"area_threshold_hectares must be > 0.0, "
                f"got {self.area_threshold_hectares}"
            )

        # -- Polygon vertex limits -------------------------------------------
        if not (4 <= self.min_vertices_polygon <= 100):
            errors.append(
                f"min_vertices_polygon must be in [4, 100], "
                f"got {self.min_vertices_polygon}"
            )

        if not (100 <= self.max_vertices_per_polygon <= 10_000_000):
            errors.append(
                f"max_vertices_per_polygon must be in [100, 10000000], "
                f"got {self.max_vertices_per_polygon}"
            )

        if self.min_vertices_polygon >= self.max_vertices_per_polygon:
            errors.append(
                f"min_vertices_polygon ({self.min_vertices_polygon}) must be "
                f"< max_vertices_per_polygon ({self.max_vertices_per_polygon})"
            )

        # -- Geometry repair settings ----------------------------------------
        if self.duplicate_vertex_tolerance_meters < 0.0:
            errors.append(
                f"duplicate_vertex_tolerance_meters must be >= 0.0, "
                f"got {self.duplicate_vertex_tolerance_meters}"
            )

        if self.sliver_aspect_ratio_threshold <= 1.0:
            errors.append(
                f"sliver_aspect_ratio_threshold must be > 1.0, "
                f"got {self.sliver_aspect_ratio_threshold}"
            )

        if not (0.0 < self.spike_angle_threshold_degrees <= 90.0):
            errors.append(
                f"spike_angle_threshold_degrees must be in (0.0, 90.0], "
                f"got {self.spike_angle_threshold_degrees}"
            )

        # -- Simplification settings -----------------------------------------
        if self.simplification_default_tolerance <= 0.0:
            errors.append(
                f"simplification_default_tolerance must be > 0.0, "
                f"got {self.simplification_default_tolerance}"
            )

        if not (0.0 < self.simplification_area_deviation_max <= 1.0):
            errors.append(
                f"simplification_area_deviation_max must be in (0.0, 1.0], "
                f"got {self.simplification_area_deviation_max}"
            )

        # -- Overlap detection thresholds ------------------------------------
        if self.overlap_detection_min_area_m2 < 0.0:
            errors.append(
                f"overlap_detection_min_area_m2 must be >= 0.0, "
                f"got {self.overlap_detection_min_area_m2}"
            )

        if not (0.0 < self.overlap_minor_threshold <= 1.0):
            errors.append(
                f"overlap_minor_threshold must be in (0.0, 1.0], "
                f"got {self.overlap_minor_threshold}"
            )

        if not (0.0 < self.overlap_moderate_threshold <= 1.0):
            errors.append(
                f"overlap_moderate_threshold must be in (0.0, 1.0], "
                f"got {self.overlap_moderate_threshold}"
            )

        if not (0.0 < self.overlap_major_threshold <= 1.0):
            errors.append(
                f"overlap_major_threshold must be in (0.0, 1.0], "
                f"got {self.overlap_major_threshold}"
            )

        if self.overlap_minor_threshold >= self.overlap_moderate_threshold:
            errors.append(
                f"overlap_minor_threshold ({self.overlap_minor_threshold}) "
                f"must be < overlap_moderate_threshold "
                f"({self.overlap_moderate_threshold})"
            )

        if self.overlap_moderate_threshold >= self.overlap_major_threshold:
            errors.append(
                f"overlap_moderate_threshold ({self.overlap_moderate_threshold}) "
                f"must be < overlap_major_threshold "
                f"({self.overlap_major_threshold})"
            )

        # -- Boundary versioning ---------------------------------------------
        if not (1 <= self.version_retention_years <= 100):
            errors.append(
                f"version_retention_years must be in [1, 100], "
                f"got {self.version_retention_years}"
            )

        # -- Batch processing ------------------------------------------------
        if not (1 <= self.batch_max_size <= 1_000_000):
            errors.append(
                f"batch_max_size must be in [1, 1000000], "
                f"got {self.batch_max_size}"
            )

        if not (1 <= self.batch_concurrency <= 256):
            errors.append(
                f"batch_concurrency must be in [1, 256], "
                f"got {self.batch_concurrency}"
            )

        # -- Spatial indexing ------------------------------------------------
        if not (1 <= self.r_tree_page_size <= 10000):
            errors.append(
                f"r_tree_page_size must be in [1, 10000], "
                f"got {self.r_tree_page_size}"
            )

        # -- Export settings -------------------------------------------------
        if not (1 <= self.export_default_precision <= 15):
            errors.append(
                f"export_default_precision must be in [1, 15], "
                f"got {self.export_default_precision}"
            )

        if not self.export_formats:
            errors.append("export_formats must not be empty")

        for fmt in self.export_formats:
            if fmt not in _VALID_EXPORT_FORMATS:
                errors.append(
                    f"export_formats contains invalid format '{fmt}'; "
                    f"valid formats are {sorted(_VALID_EXPORT_FORMATS)}"
                )

        # -- Geodetic computation settings -----------------------------------
        if self.wgs84_semi_major_axis <= 0.0:
            errors.append(
                f"wgs84_semi_major_axis must be > 0.0, "
                f"got {self.wgs84_semi_major_axis}"
            )

        if not (0.0 < self.wgs84_flattening < 1.0):
            errors.append(
                f"wgs84_flattening must be in (0.0, 1.0), "
                f"got {self.wgs84_flattening}"
            )

        # -- Coordinate precision --------------------------------------------
        if not (1 <= self.coordinate_precision_digits <= 15):
            errors.append(
                f"coordinate_precision_digits must be in [1, 15], "
                f"got {self.coordinate_precision_digits}"
            )

        # -- Polygon area limits ---------------------------------------------
        if self.max_polygon_area_hectares <= 0.0:
            errors.append(
                f"max_polygon_area_hectares must be > 0.0, "
                f"got {self.max_polygon_area_hectares}"
            )

        if self.min_polygon_area_hectares < 0.0:
            errors.append(
                f"min_polygon_area_hectares must be >= 0.0, "
                f"got {self.min_polygon_area_hectares}"
            )

        if self.min_polygon_area_hectares >= self.max_polygon_area_hectares:
            errors.append(
                f"min_polygon_area_hectares ({self.min_polygon_area_hectares}) "
                f"must be < max_polygon_area_hectares "
                f"({self.max_polygon_area_hectares})"
            )

        # -- Ring and hole tolerances ----------------------------------------
        if self.ring_closure_tolerance_meters < 0.0:
            errors.append(
                f"ring_closure_tolerance_meters must be >= 0.0, "
                f"got {self.ring_closure_tolerance_meters}"
            )

        if self.hole_containment_buffer_meters < 0.0:
            errors.append(
                f"hole_containment_buffer_meters must be >= 0.0, "
                f"got {self.hole_containment_buffer_meters}"
            )

        # -- Split/merge settings --------------------------------------------
        if not (0.0 < self.split_merge_area_tolerance <= 1.0):
            errors.append(
                f"split_merge_area_tolerance must be in (0.0, 1.0], "
                f"got {self.split_merge_area_tolerance}"
            )

        # -- EUDR commodity scope --------------------------------------------
        if not self.eudr_commodities:
            errors.append("eudr_commodities must not be empty")

        # -- Performance tuning ----------------------------------------------
        if not (1 <= self.batch_size <= 1_000_000):
            errors.append(
                f"batch_size must be in [1, 1000000], "
                f"got {self.batch_size}"
            )

        if not (1 <= self.max_concurrent_jobs <= 256):
            errors.append(
                f"max_concurrent_jobs must be in [1, 256], "
                f"got {self.max_concurrent_jobs}"
            )

        # -- Caching ---------------------------------------------------------
        if self.cache_ttl_seconds <= 0:
            errors.append(
                f"cache_ttl_seconds must be > 0, "
                f"got {self.cache_ttl_seconds}"
            )

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        if errors:
            raise ValueError(
                "PlotBoundaryConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "PlotBoundaryConfig validated successfully: "
            "canonical_crs=%s, area_threshold=%.1fha, "
            "max_vertices=%d, min_vertices=%d, "
            "simplification_tol=%.6f, area_dev_max=%.3f, "
            "overlap_thresholds=%.2f/%.2f/%.2f, "
            "version_retention=%dy, batch_max=%d, concurrency=%d, "
            "precision=%d, karney=%s, utm_auto=%s, "
            "cache_ttl=%ds, metrics=%s",
            self.canonical_crs,
            self.area_threshold_hectares,
            self.max_vertices_per_polygon,
            self.min_vertices_polygon,
            self.simplification_default_tolerance,
            self.simplification_area_deviation_max,
            self.overlap_minor_threshold,
            self.overlap_moderate_threshold,
            self.overlap_major_threshold,
            self.version_retention_years,
            self.batch_max_size,
            self.batch_concurrency,
            self.export_default_precision,
            self.karney_algorithm_enabled,
            self.utm_zone_auto_detect,
            self.cache_ttl_seconds,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> PlotBoundaryConfig:
        """Build a PlotBoundaryConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_PBM_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated PlotBoundaryConfig instance, validated
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
            # CRS settings
            canonical_crs=_str("CANONICAL_CRS", cls.canonical_crs),
            # EUDR threshold
            area_threshold_hectares=_float(
                "AREA_THRESHOLD_HECTARES", cls.area_threshold_hectares,
            ),
            # Polygon vertex limits
            max_vertices_per_polygon=_int(
                "MAX_VERTICES_PER_POLYGON", cls.max_vertices_per_polygon,
            ),
            min_vertices_polygon=_int(
                "MIN_VERTICES_POLYGON", cls.min_vertices_polygon,
            ),
            # Geometry repair
            duplicate_vertex_tolerance_meters=_float(
                "DUPLICATE_VERTEX_TOLERANCE_M",
                cls.duplicate_vertex_tolerance_meters,
            ),
            self_intersection_repair_enabled=_bool(
                "SELF_INTERSECTION_REPAIR",
                cls.self_intersection_repair_enabled,
            ),
            sliver_aspect_ratio_threshold=_float(
                "SLIVER_ASPECT_RATIO_THRESHOLD",
                cls.sliver_aspect_ratio_threshold,
            ),
            spike_angle_threshold_degrees=_float(
                "SPIKE_ANGLE_THRESHOLD_DEG",
                cls.spike_angle_threshold_degrees,
            ),
            # Simplification
            simplification_default_tolerance=_float(
                "SIMPLIFICATION_TOLERANCE",
                cls.simplification_default_tolerance,
            ),
            simplification_area_deviation_max=_float(
                "SIMPLIFICATION_AREA_DEV_MAX",
                cls.simplification_area_deviation_max,
            ),
            # Overlap detection
            overlap_detection_min_area_m2=_float(
                "OVERLAP_MIN_AREA_M2",
                cls.overlap_detection_min_area_m2,
            ),
            overlap_minor_threshold=_float(
                "OVERLAP_MINOR_THRESHOLD",
                cls.overlap_minor_threshold,
            ),
            overlap_moderate_threshold=_float(
                "OVERLAP_MODERATE_THRESHOLD",
                cls.overlap_moderate_threshold,
            ),
            overlap_major_threshold=_float(
                "OVERLAP_MAJOR_THRESHOLD",
                cls.overlap_major_threshold,
            ),
            # Versioning
            version_retention_years=_int(
                "VERSION_RETENTION_YEARS",
                cls.version_retention_years,
            ),
            # Batch processing
            batch_max_size=_int("BATCH_MAX_SIZE", cls.batch_max_size),
            batch_concurrency=_int(
                "BATCH_CONCURRENCY", cls.batch_concurrency,
            ),
            # Spatial indexing
            r_tree_page_size=_int(
                "R_TREE_PAGE_SIZE", cls.r_tree_page_size,
            ),
            # Export settings
            export_default_precision=_int(
                "EXPORT_DEFAULT_PRECISION",
                cls.export_default_precision,
            ),
            # Geodetic computation
            karney_algorithm_enabled=_bool(
                "KARNEY_ALGORITHM_ENABLED",
                cls.karney_algorithm_enabled,
            ),
            utm_zone_auto_detect=_bool(
                "UTM_ZONE_AUTO_DETECT",
                cls.utm_zone_auto_detect,
            ),
            # Coordinate precision
            coordinate_precision_digits=_int(
                "COORDINATE_PRECISION_DIGITS",
                cls.coordinate_precision_digits,
            ),
            # Polygon area limits
            max_polygon_area_hectares=_float(
                "MAX_POLYGON_AREA_HECTARES",
                cls.max_polygon_area_hectares,
            ),
            min_polygon_area_hectares=_float(
                "MIN_POLYGON_AREA_HECTARES",
                cls.min_polygon_area_hectares,
            ),
            # Ring and hole tolerances
            ring_closure_tolerance_meters=_float(
                "RING_CLOSURE_TOLERANCE_M",
                cls.ring_closure_tolerance_meters,
            ),
            hole_containment_buffer_meters=_float(
                "HOLE_CONTAINMENT_BUFFER_M",
                cls.hole_containment_buffer_meters,
            ),
            # Split/merge
            split_merge_area_tolerance=_float(
                "SPLIT_MERGE_AREA_TOLERANCE",
                cls.split_merge_area_tolerance,
            ),
            # Performance
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            max_concurrent_jobs=_int(
                "MAX_CONCURRENT_JOBS", cls.max_concurrent_jobs,
            ),
            # Caching
            cache_ttl_seconds=_int(
                "CACHE_TTL_SECONDS", cls.cache_ttl_seconds,
            ),
            # Provenance
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Metrics
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
        )

        logger.info(
            "PlotBoundaryConfig loaded: "
            "canonical_crs=%s, area_threshold=%.1fha, "
            "max_vertices=%d, min_vertices=%d, "
            "simplification_tol=%.6f, area_dev_max=%.3f, "
            "overlap_thresholds=%.2f/%.2f/%.2f, "
            "version_retention=%dy, batch_max=%d, concurrency=%d, "
            "precision=%d, karney=%s, utm_auto=%s, "
            "cache_ttl=%ds, metrics=%s",
            config.canonical_crs,
            config.area_threshold_hectares,
            config.max_vertices_per_polygon,
            config.min_vertices_polygon,
            config.simplification_default_tolerance,
            config.simplification_area_deviation_max,
            config.overlap_minor_threshold,
            config.overlap_moderate_threshold,
            config.overlap_major_threshold,
            config.version_retention_years,
            config.batch_max_size,
            config.batch_concurrency,
            config.export_default_precision,
            config.karney_algorithm_enabled,
            config.utm_zone_auto_detect,
            config.cache_ttl_seconds,
            config.enable_metrics,
        )
        return config

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def wgs84_semi_minor_axis(self) -> float:
        """Return the WGS84 semi-minor axis in metres.

        Computed from the semi-major axis and flattening:
        b = a * (1 - f)

        Returns:
            Semi-minor axis in metres.
        """
        return self.wgs84_semi_major_axis * (1.0 - self.wgs84_flattening)

    @property
    def wgs84_eccentricity_squared(self) -> float:
        """Return the first eccentricity squared of the WGS84 ellipsoid.

        Computed as e^2 = 2f - f^2.

        Returns:
            First eccentricity squared.
        """
        f = self.wgs84_flattening
        return 2.0 * f - f * f

    @property
    def genesis_hash_sha256(self) -> str:
        """Return the SHA-256 hash of the genesis anchor string.

        Returns:
            Hex-encoded SHA-256 digest of the genesis_hash attribute.
        """
        return hashlib.sha256(
            self.genesis_hash.encode("utf-8")
        ).hexdigest()

    @property
    def area_threshold_m2(self) -> float:
        """Return the EUDR area threshold in square metres.

        Returns:
            Area threshold converted from hectares to square metres.
        """
        return self.area_threshold_hectares * 10000.0

    @property
    def max_polygon_area_m2(self) -> float:
        """Return the maximum polygon area in square metres.

        Returns:
            Maximum area converted from hectares to square metres.
        """
        return self.max_polygon_area_hectares * 10000.0

    @property
    def min_polygon_area_m2(self) -> float:
        """Return the minimum polygon area in square metres.

        Returns:
            Minimum area converted from hectares to square metres.
        """
        return self.min_polygon_area_hectares * 10000.0

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain Python dictionary.

        Sensitive connection strings (database_url, redis_url) are
        redacted to prevent accidental credential leakage in logs,
        exception tracebacks, and monitoring dashboards.

        Returns:
            Dictionary representation with sensitive fields redacted.
        """
        return {
            # Connections (redacted)
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # Logging
            "log_level": self.log_level,
            # CRS settings
            "canonical_crs": self.canonical_crs,
            "supported_input_crs_count": len(self.supported_input_crs),
            # EUDR threshold
            "area_threshold_hectares": self.area_threshold_hectares,
            # Polygon vertex limits
            "max_vertices_per_polygon": self.max_vertices_per_polygon,
            "min_vertices_polygon": self.min_vertices_polygon,
            # Geometry repair
            "duplicate_vertex_tolerance_meters": self.duplicate_vertex_tolerance_meters,
            "self_intersection_repair_enabled": self.self_intersection_repair_enabled,
            "sliver_aspect_ratio_threshold": self.sliver_aspect_ratio_threshold,
            "spike_angle_threshold_degrees": self.spike_angle_threshold_degrees,
            # Simplification
            "simplification_default_tolerance": self.simplification_default_tolerance,
            "simplification_area_deviation_max": self.simplification_area_deviation_max,
            # Overlap detection
            "overlap_detection_min_area_m2": self.overlap_detection_min_area_m2,
            "overlap_minor_threshold": self.overlap_minor_threshold,
            "overlap_moderate_threshold": self.overlap_moderate_threshold,
            "overlap_major_threshold": self.overlap_major_threshold,
            # Versioning
            "version_retention_years": self.version_retention_years,
            # Batch processing
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            # Spatial indexing
            "r_tree_page_size": self.r_tree_page_size,
            # Export settings
            "export_default_precision": self.export_default_precision,
            "export_formats": self.export_formats,
            # Geodetic computation
            "karney_algorithm_enabled": self.karney_algorithm_enabled,
            "wgs84_semi_major_axis": self.wgs84_semi_major_axis,
            "wgs84_flattening": self.wgs84_flattening,
            "utm_zone_auto_detect": self.utm_zone_auto_detect,
            # Coordinate precision
            "coordinate_precision_digits": self.coordinate_precision_digits,
            # Polygon area limits
            "max_polygon_area_hectares": self.max_polygon_area_hectares,
            "min_polygon_area_hectares": self.min_polygon_area_hectares,
            # Ring and hole tolerances
            "ring_closure_tolerance_meters": self.ring_closure_tolerance_meters,
            "hole_containment_buffer_meters": self.hole_containment_buffer_meters,
            # Split/merge
            "split_merge_area_tolerance": self.split_merge_area_tolerance,
            # EUDR commodity scope
            "eudr_commodities": self.eudr_commodities,
            # Performance
            "batch_size": self.batch_size,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            # Caching
            "cache_ttl_seconds": self.cache_ttl_seconds,
            # Provenance
            "genesis_hash": self.genesis_hash,
            # Metrics
            "enable_metrics": self.enable_metrics,
        }

    def validate(self) -> bool:
        """Re-validate the current configuration.

        Creates a new instance with the same values to trigger
        ``__post_init__`` validation. Returns True if valid, raises
        ValueError if not.

        Returns:
            True if configuration is valid.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        # Trigger __post_init__ validation by creating a copy
        PlotBoundaryConfig(**{
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        })
        return True

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Returns:
            String representation with sensitive fields redacted.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"PlotBoundaryConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[PlotBoundaryConfig] = None
_config_lock = threading.Lock()


def get_config() -> PlotBoundaryConfig:
    """Return the singleton PlotBoundaryConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_PBM_*`` environment variables.

    Returns:
        PlotBoundaryConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.canonical_crs
        'EPSG:4326'
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = PlotBoundaryConfig.from_env()
    return _config_instance


def set_config(config: PlotBoundaryConfig) -> None:
    """Replace the singleton PlotBoundaryConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New PlotBoundaryConfig to install.

    Example:
        >>> cfg = PlotBoundaryConfig(area_threshold_hectares=5.0)
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "PlotBoundaryConfig replaced programmatically: "
        "canonical_crs=%s, area_threshold=%.1fha, "
        "batch_max_size=%d",
        config.canonical_crs,
        config.area_threshold_hectares,
        config.batch_max_size,
    )


def reset_config() -> None:
    """Reset the singleton PlotBoundaryConfig to None.

    The next call to get_config() will re-read GL_EUDR_PBM_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("PlotBoundaryConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "PlotBoundaryConfig",
    "get_config",
    "set_config",
    "reset_config",
]
