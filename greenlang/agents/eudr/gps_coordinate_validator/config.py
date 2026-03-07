# -*- coding: utf-8 -*-
"""
GPS Coordinate Validator Configuration - AGENT-EUDR-007

Centralized configuration for the GPS Coordinate Validator Agent covering:
- Canonical CRS and coordinate format settings
- Latitude/longitude range bounds
- Precision thresholds for EUDR compliance and quality grading
- Swap detection and auto-correction parameters
- Null Island detection threshold
- Duplicate and near-duplicate distance thresholds
- Spatial plausibility check toggles (ocean, country, commodity, elevation)
- Quality score weights and accuracy tier thresholds
- Batch processing limits and concurrency
- Datum transformation parameters (Helmert, Molodensky, WGS84 ellipsoid)
- Supported coordinate formats and geodetic datums
- Report format and version retention settings
- Database and cache connection settings
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_GCV_`` prefix (e.g. ``GL_EUDR_GCV_DATABASE_URL``,
``GL_EUDR_GCV_MIN_DECIMAL_PLACES_EUDR``).

Environment Variable Reference (GL_EUDR_GCV_ prefix):
    GL_EUDR_GCV_DATABASE_URL                  - PostgreSQL connection URL
    GL_EUDR_GCV_REDIS_URL                     - Redis connection URL
    GL_EUDR_GCV_LOG_LEVEL                     - Logging level
    GL_EUDR_GCV_CANONICAL_CRS                 - Canonical CRS identifier
    GL_EUDR_GCV_CANONICAL_FORMAT              - Canonical coordinate format
    GL_EUDR_GCV_LATITUDE_MIN                  - Minimum valid latitude
    GL_EUDR_GCV_LATITUDE_MAX                  - Maximum valid latitude
    GL_EUDR_GCV_LONGITUDE_MIN                 - Minimum valid longitude
    GL_EUDR_GCV_LONGITUDE_MAX                 - Maximum valid longitude
    GL_EUDR_GCV_MIN_DECIMAL_PLACES_EUDR       - Min decimal places for EUDR
    GL_EUDR_GCV_PRECISION_SURVEY_GRADE_M      - Survey grade threshold (m)
    GL_EUDR_GCV_PRECISION_HIGH_M              - High precision threshold (m)
    GL_EUDR_GCV_PRECISION_MODERATE_M          - Moderate precision threshold (m)
    GL_EUDR_GCV_PRECISION_LOW_M               - Low precision threshold (m)
    GL_EUDR_GCV_SWAP_DETECTION_ENABLED        - Enable swap detection
    GL_EUDR_GCV_SWAP_CONFIDENCE_THRESHOLD     - Swap detection confidence
    GL_EUDR_GCV_AUTO_CORRECTION_ENABLED       - Enable auto-correction
    GL_EUDR_GCV_AUTO_CORRECTION_CONFIDENCE_THRESHOLD - Auto-correction confidence
    GL_EUDR_GCV_NULL_ISLAND_THRESHOLD_DEGREES - Null island threshold (deg)
    GL_EUDR_GCV_DUPLICATE_DISTANCE_THRESHOLD_M - Duplicate distance (m)
    GL_EUDR_GCV_OCEAN_CHECK_ENABLED           - Enable ocean check
    GL_EUDR_GCV_COUNTRY_CHECK_ENABLED         - Enable country check
    GL_EUDR_GCV_COMMODITY_CHECK_ENABLED       - Enable commodity check
    GL_EUDR_GCV_ELEVATION_CHECK_ENABLED       - Enable elevation check
    GL_EUDR_GCV_ELEVATION_TOLERANCE_M         - Elevation tolerance (m)
    GL_EUDR_GCV_QUALITY_WEIGHT_PRECISION      - Quality weight: precision
    GL_EUDR_GCV_QUALITY_WEIGHT_PLAUSIBILITY   - Quality weight: plausibility
    GL_EUDR_GCV_QUALITY_WEIGHT_CONSISTENCY    - Quality weight: consistency
    GL_EUDR_GCV_QUALITY_WEIGHT_SOURCE         - Quality weight: source
    GL_EUDR_GCV_TIER_GOLD_THRESHOLD           - Gold tier threshold
    GL_EUDR_GCV_TIER_SILVER_THRESHOLD         - Silver tier threshold
    GL_EUDR_GCV_TIER_BRONZE_THRESHOLD         - Bronze tier threshold
    GL_EUDR_GCV_BATCH_MAX_SIZE                - Max batch size
    GL_EUDR_GCV_BATCH_CONCURRENCY             - Batch concurrency
    GL_EUDR_GCV_HELMERT_MAX_ITERATIONS        - Helmert max iterations
    GL_EUDR_GCV_HELMERT_CONVERGENCE_TOLERANCE - Helmert convergence tol
    GL_EUDR_GCV_MOLODENSKY_ENABLED            - Enable Molodensky transforms
    GL_EUDR_GCV_WGS84_SEMI_MAJOR_AXIS         - WGS84 semi-major axis (m)
    GL_EUDR_GCV_WGS84_FLATTENING              - WGS84 flattening
    GL_EUDR_GCV_VERSION_RETENTION_YEARS        - Version retention (years)
    GL_EUDR_GCV_ENABLE_PROVENANCE              - Enable provenance tracking
    GL_EUDR_GCV_GENESIS_HASH                   - Genesis hash anchor
    GL_EUDR_GCV_ENABLE_METRICS                 - Enable Prometheus metrics
    GL_EUDR_GCV_POOL_SIZE                      - Database pool size
    GL_EUDR_GCV_RATE_LIMIT                     - Max requests per minute

Example:
    >>> from greenlang.agents.eudr.gps_coordinate_validator.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.min_decimal_places_eudr)
    5

    >>> # Override for testing
    >>> from greenlang.agents.eudr.gps_coordinate_validator.config import (
    ...     set_config, reset_config, GPSCoordinateValidatorConfig,
    ... )
    >>> set_config(GPSCoordinateValidatorConfig(min_decimal_places_eudr=6))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GCV-007)
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_GCV_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Default supported formats
# ---------------------------------------------------------------------------

_DEFAULT_SUPPORTED_FORMATS: List[str] = [
    "decimal_degrees",
    "dms",
    "ddm",
    "utm",
    "mgrs",
    "signed_dd",
    "dd_suffix",
]

# ---------------------------------------------------------------------------
# Default supported datums (30+ datums)
# ---------------------------------------------------------------------------

_DEFAULT_SUPPORTED_DATUMS: List[str] = [
    "wgs84", "nad27", "nad83", "ed50", "etrs89", "osgb36",
    "sirgas_2000", "indian_1975", "arc_1960", "pulkovo_1942",
    "tokyo", "gda94", "gda2020", "nzgd2000", "cape",
    "hermannskogel", "potsdam", "rome_1940", "bessel_1841",
    "kertau_1948", "luzon_1911", "timbalai_1948", "everest_1956",
    "kalianpur_1975", "hong_kong_1980", "south_american_1969",
    "bogota_1975", "campo_inchauspe", "chua_astro",
    "corrego_alegre", "yacare", "zanderij", "adindan",
    "minna", "camacupa", "schwarzeck", "hartebeesthoek94",
]

# ---------------------------------------------------------------------------
# Default report formats
# ---------------------------------------------------------------------------

_DEFAULT_REPORT_FORMATS: List[str] = ["json", "pdf", "csv", "eudr_xml"]

# ---------------------------------------------------------------------------
# Default quality score weights (must sum to 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_QUALITY_WEIGHTS: Dict[str, float] = {
    "precision": 0.25,
    "plausibility": 0.25,
    "consistency": 0.25,
    "source": 0.25,
}


# ---------------------------------------------------------------------------
# GPSCoordinateValidatorConfig
# ---------------------------------------------------------------------------


@dataclass
class GPSCoordinateValidatorConfig:
    """Complete configuration for the EUDR GPS Coordinate Validator Agent.

    Attributes are grouped by concern: connections, logging, canonical
    settings, range bounds, precision thresholds, swap detection,
    auto-correction, null island, duplicates, plausibility checks,
    quality scoring, batch processing, datum transformation, supported
    formats/datums, report settings, provenance tracking, and metrics.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_GCV_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching.
        log_level: Logging verbosity level.
        canonical_crs: Canonical Coordinate Reference System identifier.
        canonical_format: Canonical coordinate output format.
        latitude_min: Minimum valid WGS84 latitude.
        latitude_max: Maximum valid WGS84 latitude.
        longitude_min: Minimum valid WGS84 longitude.
        longitude_max: Maximum valid WGS84 longitude.
        min_decimal_places_eudr: Minimum decimal places for EUDR compliance.
        precision_survey_grade_m: Survey grade precision threshold in meters.
        precision_high_m: High precision threshold in meters.
        precision_moderate_m: Moderate precision threshold in meters.
        precision_low_m: Low precision threshold in meters.
        swap_detection_enabled: Whether to enable lat/lon swap detection.
        swap_confidence_threshold: Minimum confidence for swap detection.
        auto_correction_enabled: Whether to enable auto-correction of errors.
        auto_correction_confidence_threshold: Minimum confidence for auto-correction.
        null_island_threshold_degrees: Distance from (0,0) in degrees to
            flag as potential null island.
        duplicate_distance_threshold_m: Distance threshold for duplicate detection.
        ocean_check_enabled: Whether to check if coordinate is in ocean.
        country_check_enabled: Whether to verify country match.
        commodity_check_enabled: Whether to check commodity plausibility.
        elevation_check_enabled: Whether to check elevation plausibility.
        elevation_tolerance_m: Elevation tolerance in meters.
        quality_weight_precision: Weight for precision component (0.0-1.0).
        quality_weight_plausibility: Weight for plausibility component.
        quality_weight_consistency: Weight for consistency component.
        quality_weight_source: Weight for source quality component.
        tier_gold_threshold: Minimum score for GOLD tier.
        tier_silver_threshold: Minimum score for SILVER tier.
        tier_bronze_threshold: Minimum score for BRONZE tier.
        batch_max_size: Maximum coordinates in a single batch.
        batch_concurrency: Maximum concurrent batch workers.
        helmert_max_iterations: Maximum iterations for Helmert transformation.
        helmert_convergence_tolerance: Convergence tolerance for Helmert.
        molodensky_enabled: Whether to use Molodensky transformations.
        wgs84_semi_major_axis: WGS84 semi-major axis in meters.
        wgs84_flattening: WGS84 flattening value.
        supported_formats: List of supported coordinate format strings.
        supported_datums: List of supported geodetic datum strings.
        report_formats: List of supported report output formats.
        version_retention_years: Years to retain coordinate versions.
        enable_provenance: Enable SHA-256 provenance chain tracking.
        genesis_hash: Genesis anchor string for provenance chain.
        enable_metrics: Enable Prometheus metrics export.
        pool_size: Database connection pool size.
        rate_limit: Maximum API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Canonical settings --------------------------------------------------
    canonical_crs: str = "EPSG:4326"
    canonical_format: str = "decimal_degrees"

    # -- Range bounds --------------------------------------------------------
    latitude_min: float = -90.0
    latitude_max: float = 90.0
    longitude_min: float = -180.0
    longitude_max: float = 180.0

    # -- Precision thresholds ------------------------------------------------
    min_decimal_places_eudr: int = 5
    precision_survey_grade_m: float = 1.0
    precision_high_m: float = 10.0
    precision_moderate_m: float = 100.0
    precision_low_m: float = 1000.0

    # -- Swap detection ------------------------------------------------------
    swap_detection_enabled: bool = True
    swap_confidence_threshold: float = 0.85

    # -- Auto-correction -----------------------------------------------------
    auto_correction_enabled: bool = False
    auto_correction_confidence_threshold: float = 0.95

    # -- Null island ---------------------------------------------------------
    null_island_threshold_degrees: float = 0.01

    # -- Duplicates ----------------------------------------------------------
    duplicate_distance_threshold_m: float = 1.0

    # -- Plausibility checks -------------------------------------------------
    ocean_check_enabled: bool = True
    country_check_enabled: bool = True
    commodity_check_enabled: bool = True
    elevation_check_enabled: bool = True
    elevation_tolerance_m: float = 200.0

    # -- Quality scoring weights (must sum to 1.0) ---------------------------
    quality_weight_precision: float = 0.25
    quality_weight_plausibility: float = 0.25
    quality_weight_consistency: float = 0.25
    quality_weight_source: float = 0.25

    # -- Accuracy tier thresholds --------------------------------------------
    tier_gold_threshold: float = 90.0
    tier_silver_threshold: float = 70.0
    tier_bronze_threshold: float = 50.0

    # -- Batch processing ----------------------------------------------------
    batch_max_size: int = 50_000
    batch_concurrency: int = 8

    # -- Datum transformation ------------------------------------------------
    helmert_max_iterations: int = 10
    helmert_convergence_tolerance: float = 1e-12
    molodensky_enabled: bool = True
    wgs84_semi_major_axis: float = 6378137.0
    wgs84_flattening: float = 1.0 / 298.257223563

    # -- Supported formats and datums ----------------------------------------
    supported_formats: List[str] = field(
        default_factory=lambda: list(_DEFAULT_SUPPORTED_FORMATS)
    )
    supported_datums: List[str] = field(
        default_factory=lambda: list(_DEFAULT_SUPPORTED_DATUMS)
    )

    # -- Report settings -----------------------------------------------------
    report_formats: List[str] = field(
        default_factory=lambda: list(_DEFAULT_REPORT_FORMATS)
    )
    version_retention_years: int = 5

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-GCV-007-GPS-COORDINATE-VALIDATOR-GENESIS"

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
        on string fields, quality weight sum validation, tier ordering,
        and normalization. Collects all errors before raising a single
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

        # -- Range bounds ----------------------------------------------------
        if self.latitude_min < -90.0:
            errors.append(
                f"latitude_min must be >= -90.0, got {self.latitude_min}"
            )
        if self.latitude_max > 90.0:
            errors.append(
                f"latitude_max must be <= 90.0, got {self.latitude_max}"
            )
        if self.latitude_min >= self.latitude_max:
            errors.append(
                f"latitude_min ({self.latitude_min}) must be "
                f"< latitude_max ({self.latitude_max})"
            )
        if self.longitude_min < -180.0:
            errors.append(
                f"longitude_min must be >= -180.0, got {self.longitude_min}"
            )
        if self.longitude_max > 180.0:
            errors.append(
                f"longitude_max must be <= 180.0, got {self.longitude_max}"
            )
        if self.longitude_min >= self.longitude_max:
            errors.append(
                f"longitude_min ({self.longitude_min}) must be "
                f"< longitude_max ({self.longitude_max})"
            )

        # -- Precision thresholds --------------------------------------------
        if not (1 <= self.min_decimal_places_eudr <= 15):
            errors.append(
                f"min_decimal_places_eudr must be in [1, 15], "
                f"got {self.min_decimal_places_eudr}"
            )
        for name, value in [
            ("precision_survey_grade_m", self.precision_survey_grade_m),
            ("precision_high_m", self.precision_high_m),
            ("precision_moderate_m", self.precision_moderate_m),
            ("precision_low_m", self.precision_low_m),
        ]:
            if value <= 0:
                errors.append(f"{name} must be > 0, got {value}")

        if self.precision_survey_grade_m >= self.precision_high_m:
            errors.append(
                f"precision_survey_grade_m ({self.precision_survey_grade_m}) "
                f"must be < precision_high_m ({self.precision_high_m})"
            )
        if self.precision_high_m >= self.precision_moderate_m:
            errors.append(
                f"precision_high_m ({self.precision_high_m}) "
                f"must be < precision_moderate_m ({self.precision_moderate_m})"
            )
        if self.precision_moderate_m >= self.precision_low_m:
            errors.append(
                f"precision_moderate_m ({self.precision_moderate_m}) "
                f"must be < precision_low_m ({self.precision_low_m})"
            )

        # -- Swap detection --------------------------------------------------
        if not (0.0 < self.swap_confidence_threshold <= 1.0):
            errors.append(
                f"swap_confidence_threshold must be in (0, 1], "
                f"got {self.swap_confidence_threshold}"
            )

        # -- Auto-correction -------------------------------------------------
        if not (0.0 < self.auto_correction_confidence_threshold <= 1.0):
            errors.append(
                f"auto_correction_confidence_threshold must be in (0, 1], "
                f"got {self.auto_correction_confidence_threshold}"
            )

        # -- Null island -----------------------------------------------------
        if self.null_island_threshold_degrees < 0:
            errors.append(
                f"null_island_threshold_degrees must be >= 0, "
                f"got {self.null_island_threshold_degrees}"
            )

        # -- Duplicates ------------------------------------------------------
        if self.duplicate_distance_threshold_m < 0:
            errors.append(
                f"duplicate_distance_threshold_m must be >= 0, "
                f"got {self.duplicate_distance_threshold_m}"
            )

        # -- Elevation -------------------------------------------------------
        if self.elevation_tolerance_m < 0:
            errors.append(
                f"elevation_tolerance_m must be >= 0, "
                f"got {self.elevation_tolerance_m}"
            )

        # -- Quality weights -------------------------------------------------
        weights = [
            ("quality_weight_precision", self.quality_weight_precision),
            ("quality_weight_plausibility", self.quality_weight_plausibility),
            ("quality_weight_consistency", self.quality_weight_consistency),
            ("quality_weight_source", self.quality_weight_source),
        ]
        for name, value in weights:
            if not (0.0 <= value <= 1.0):
                errors.append(
                    f"{name} must be in [0.0, 1.0], got {value}"
                )
        weight_sum = sum(v for _, v in weights)
        if abs(weight_sum - 1.0) > 0.001:
            errors.append(
                f"quality weights must sum to 1.0, got {weight_sum:.4f}"
            )

        # -- Tier thresholds -------------------------------------------------
        if not (0.0 <= self.tier_bronze_threshold <= 100.0):
            errors.append(
                f"tier_bronze_threshold must be in [0, 100], "
                f"got {self.tier_bronze_threshold}"
            )
        if not (0.0 <= self.tier_silver_threshold <= 100.0):
            errors.append(
                f"tier_silver_threshold must be in [0, 100], "
                f"got {self.tier_silver_threshold}"
            )
        if not (0.0 <= self.tier_gold_threshold <= 100.0):
            errors.append(
                f"tier_gold_threshold must be in [0, 100], "
                f"got {self.tier_gold_threshold}"
            )
        if self.tier_bronze_threshold >= self.tier_silver_threshold:
            errors.append(
                f"tier_bronze_threshold ({self.tier_bronze_threshold}) "
                f"must be < tier_silver_threshold ({self.tier_silver_threshold})"
            )
        if self.tier_silver_threshold >= self.tier_gold_threshold:
            errors.append(
                f"tier_silver_threshold ({self.tier_silver_threshold}) "
                f"must be < tier_gold_threshold ({self.tier_gold_threshold})"
            )

        # -- Batch processing ------------------------------------------------
        if self.batch_max_size < 1:
            errors.append(
                f"batch_max_size must be >= 1, got {self.batch_max_size}"
            )
        if not (1 <= self.batch_concurrency <= 256):
            errors.append(
                f"batch_concurrency must be in [1, 256], "
                f"got {self.batch_concurrency}"
            )

        # -- Datum transformation --------------------------------------------
        if self.helmert_max_iterations < 1:
            errors.append(
                f"helmert_max_iterations must be >= 1, "
                f"got {self.helmert_max_iterations}"
            )
        if self.helmert_convergence_tolerance <= 0:
            errors.append(
                f"helmert_convergence_tolerance must be > 0, "
                f"got {self.helmert_convergence_tolerance}"
            )
        if self.wgs84_semi_major_axis <= 0:
            errors.append(
                f"wgs84_semi_major_axis must be > 0, "
                f"got {self.wgs84_semi_major_axis}"
            )
        if not (0.0 < self.wgs84_flattening < 1.0):
            errors.append(
                f"wgs84_flattening must be in (0, 1), "
                f"got {self.wgs84_flattening}"
            )

        # -- Supported formats -----------------------------------------------
        if not self.supported_formats:
            errors.append("supported_formats must not be empty")

        # -- Supported datums ------------------------------------------------
        if not self.supported_datums:
            errors.append("supported_datums must not be empty")

        # -- Report formats --------------------------------------------------
        if not self.report_formats:
            errors.append("report_formats must not be empty")

        # -- Version retention -----------------------------------------------
        if self.version_retention_years < 1:
            errors.append(
                f"version_retention_years must be >= 1, "
                f"got {self.version_retention_years}"
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
                "GPSCoordinateValidatorConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "GPSCoordinateValidatorConfig validated successfully: "
            "crs=%s, format=%s, eudr_min_dp=%d, "
            "swap_detect=%s, auto_correct=%s, "
            "batch_max=%d, concurrency=%d, "
            "provenance=%s, metrics=%s",
            self.canonical_crs,
            self.canonical_format,
            self.min_decimal_places_eudr,
            self.swap_detection_enabled,
            self.auto_correction_enabled,
            self.batch_max_size,
            self.batch_concurrency,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> GPSCoordinateValidatorConfig:
        """Build a GPSCoordinateValidatorConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_GCV_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated GPSCoordinateValidatorConfig instance, validated
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
            # Canonical settings
            canonical_crs=_str("CANONICAL_CRS", cls.canonical_crs),
            canonical_format=_str("CANONICAL_FORMAT", cls.canonical_format),
            # Range bounds
            latitude_min=_float("LATITUDE_MIN", cls.latitude_min),
            latitude_max=_float("LATITUDE_MAX", cls.latitude_max),
            longitude_min=_float("LONGITUDE_MIN", cls.longitude_min),
            longitude_max=_float("LONGITUDE_MAX", cls.longitude_max),
            # Precision thresholds
            min_decimal_places_eudr=_int(
                "MIN_DECIMAL_PLACES_EUDR",
                cls.min_decimal_places_eudr,
            ),
            precision_survey_grade_m=_float(
                "PRECISION_SURVEY_GRADE_M",
                cls.precision_survey_grade_m,
            ),
            precision_high_m=_float(
                "PRECISION_HIGH_M", cls.precision_high_m,
            ),
            precision_moderate_m=_float(
                "PRECISION_MODERATE_M", cls.precision_moderate_m,
            ),
            precision_low_m=_float(
                "PRECISION_LOW_M", cls.precision_low_m,
            ),
            # Swap detection
            swap_detection_enabled=_bool(
                "SWAP_DETECTION_ENABLED",
                cls.swap_detection_enabled,
            ),
            swap_confidence_threshold=_float(
                "SWAP_CONFIDENCE_THRESHOLD",
                cls.swap_confidence_threshold,
            ),
            # Auto-correction
            auto_correction_enabled=_bool(
                "AUTO_CORRECTION_ENABLED",
                cls.auto_correction_enabled,
            ),
            auto_correction_confidence_threshold=_float(
                "AUTO_CORRECTION_CONFIDENCE_THRESHOLD",
                cls.auto_correction_confidence_threshold,
            ),
            # Null island
            null_island_threshold_degrees=_float(
                "NULL_ISLAND_THRESHOLD_DEGREES",
                cls.null_island_threshold_degrees,
            ),
            # Duplicates
            duplicate_distance_threshold_m=_float(
                "DUPLICATE_DISTANCE_THRESHOLD_M",
                cls.duplicate_distance_threshold_m,
            ),
            # Plausibility checks
            ocean_check_enabled=_bool(
                "OCEAN_CHECK_ENABLED", cls.ocean_check_enabled,
            ),
            country_check_enabled=_bool(
                "COUNTRY_CHECK_ENABLED", cls.country_check_enabled,
            ),
            commodity_check_enabled=_bool(
                "COMMODITY_CHECK_ENABLED", cls.commodity_check_enabled,
            ),
            elevation_check_enabled=_bool(
                "ELEVATION_CHECK_ENABLED", cls.elevation_check_enabled,
            ),
            elevation_tolerance_m=_float(
                "ELEVATION_TOLERANCE_M", cls.elevation_tolerance_m,
            ),
            # Quality weights
            quality_weight_precision=_float(
                "QUALITY_WEIGHT_PRECISION",
                cls.quality_weight_precision,
            ),
            quality_weight_plausibility=_float(
                "QUALITY_WEIGHT_PLAUSIBILITY",
                cls.quality_weight_plausibility,
            ),
            quality_weight_consistency=_float(
                "QUALITY_WEIGHT_CONSISTENCY",
                cls.quality_weight_consistency,
            ),
            quality_weight_source=_float(
                "QUALITY_WEIGHT_SOURCE",
                cls.quality_weight_source,
            ),
            # Tier thresholds
            tier_gold_threshold=_float(
                "TIER_GOLD_THRESHOLD", cls.tier_gold_threshold,
            ),
            tier_silver_threshold=_float(
                "TIER_SILVER_THRESHOLD", cls.tier_silver_threshold,
            ),
            tier_bronze_threshold=_float(
                "TIER_BRONZE_THRESHOLD", cls.tier_bronze_threshold,
            ),
            # Batch processing
            batch_max_size=_int("BATCH_MAX_SIZE", cls.batch_max_size),
            batch_concurrency=_int(
                "BATCH_CONCURRENCY", cls.batch_concurrency,
            ),
            # Datum transformation
            helmert_max_iterations=_int(
                "HELMERT_MAX_ITERATIONS",
                cls.helmert_max_iterations,
            ),
            helmert_convergence_tolerance=_float(
                "HELMERT_CONVERGENCE_TOLERANCE",
                cls.helmert_convergence_tolerance,
            ),
            molodensky_enabled=_bool(
                "MOLODENSKY_ENABLED", cls.molodensky_enabled,
            ),
            wgs84_semi_major_axis=_float(
                "WGS84_SEMI_MAJOR_AXIS",
                cls.wgs84_semi_major_axis,
            ),
            wgs84_flattening=_float(
                "WGS84_FLATTENING", cls.wgs84_flattening,
            ),
            # Version retention
            version_retention_years=_int(
                "VERSION_RETENTION_YEARS",
                cls.version_retention_years,
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
            "GPSCoordinateValidatorConfig loaded: "
            "crs=%s, format=%s, eudr_min_dp=%d, "
            "swap_detect=%s (conf=%.2f), "
            "auto_correct=%s (conf=%.2f), "
            "null_island=%.4fdeg, dup_dist=%.1fm, "
            "ocean=%s, country=%s, commodity=%s, elevation=%s (tol=%.0fm), "
            "weights=[%.2f,%.2f,%.2f,%.2f], "
            "tiers=[%.0f,%.0f,%.0f], "
            "batch_max=%d, concurrency=%d, "
            "helmert_iter=%d, helmert_tol=%e, molodensky=%s, "
            "formats=%d, datums=%d, reports=%d, retention=%dy, "
            "provenance=%s, pool=%d, rate_limit=%d/min, metrics=%s",
            config.canonical_crs,
            config.canonical_format,
            config.min_decimal_places_eudr,
            config.swap_detection_enabled,
            config.swap_confidence_threshold,
            config.auto_correction_enabled,
            config.auto_correction_confidence_threshold,
            config.null_island_threshold_degrees,
            config.duplicate_distance_threshold_m,
            config.ocean_check_enabled,
            config.country_check_enabled,
            config.commodity_check_enabled,
            config.elevation_check_enabled,
            config.elevation_tolerance_m,
            config.quality_weight_precision,
            config.quality_weight_plausibility,
            config.quality_weight_consistency,
            config.quality_weight_source,
            config.tier_gold_threshold,
            config.tier_silver_threshold,
            config.tier_bronze_threshold,
            config.batch_max_size,
            config.batch_concurrency,
            config.helmert_max_iterations,
            config.helmert_convergence_tolerance,
            config.molodensky_enabled,
            len(config.supported_formats),
            len(config.supported_datums),
            len(config.report_formats),
            config.version_retention_years,
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
    def quality_weights(self) -> Dict[str, float]:
        """Return quality score weights as a dictionary.

        Returns:
            Dictionary with keys: precision, plausibility, consistency, source.
        """
        return {
            "precision": self.quality_weight_precision,
            "plausibility": self.quality_weight_plausibility,
            "consistency": self.quality_weight_consistency,
            "source": self.quality_weight_source,
        }

    @property
    def tier_thresholds(self) -> Dict[str, float]:
        """Return accuracy tier thresholds as a dictionary.

        Returns:
            Dictionary with keys: gold, silver, bronze.
        """
        return {
            "gold": self.tier_gold_threshold,
            "silver": self.tier_silver_threshold,
            "bronze": self.tier_bronze_threshold,
        }

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Re-run post-init validation and return True if valid.

        Returns:
            True if configuration passes all validation checks.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        self.__post_init__()
        return True

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
            # Canonical settings
            "canonical_crs": self.canonical_crs,
            "canonical_format": self.canonical_format,
            # Range bounds
            "latitude_min": self.latitude_min,
            "latitude_max": self.latitude_max,
            "longitude_min": self.longitude_min,
            "longitude_max": self.longitude_max,
            # Precision thresholds
            "min_decimal_places_eudr": self.min_decimal_places_eudr,
            "precision_survey_grade_m": self.precision_survey_grade_m,
            "precision_high_m": self.precision_high_m,
            "precision_moderate_m": self.precision_moderate_m,
            "precision_low_m": self.precision_low_m,
            # Swap detection
            "swap_detection_enabled": self.swap_detection_enabled,
            "swap_confidence_threshold": self.swap_confidence_threshold,
            # Auto-correction
            "auto_correction_enabled": self.auto_correction_enabled,
            "auto_correction_confidence_threshold": (
                self.auto_correction_confidence_threshold
            ),
            # Null island
            "null_island_threshold_degrees": self.null_island_threshold_degrees,
            # Duplicates
            "duplicate_distance_threshold_m": self.duplicate_distance_threshold_m,
            # Plausibility checks
            "ocean_check_enabled": self.ocean_check_enabled,
            "country_check_enabled": self.country_check_enabled,
            "commodity_check_enabled": self.commodity_check_enabled,
            "elevation_check_enabled": self.elevation_check_enabled,
            "elevation_tolerance_m": self.elevation_tolerance_m,
            # Quality weights
            "quality_weight_precision": self.quality_weight_precision,
            "quality_weight_plausibility": self.quality_weight_plausibility,
            "quality_weight_consistency": self.quality_weight_consistency,
            "quality_weight_source": self.quality_weight_source,
            # Tier thresholds
            "tier_gold_threshold": self.tier_gold_threshold,
            "tier_silver_threshold": self.tier_silver_threshold,
            "tier_bronze_threshold": self.tier_bronze_threshold,
            # Batch processing
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            # Datum transformation
            "helmert_max_iterations": self.helmert_max_iterations,
            "helmert_convergence_tolerance": self.helmert_convergence_tolerance,
            "molodensky_enabled": self.molodensky_enabled,
            "wgs84_semi_major_axis": self.wgs84_semi_major_axis,
            "wgs84_flattening": self.wgs84_flattening,
            # Supported formats and datums
            "supported_formats": list(self.supported_formats),
            "supported_datums": list(self.supported_datums),
            # Report settings
            "report_formats": list(self.report_formats),
            "version_retention_years": self.version_retention_years,
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
        return f"GPSCoordinateValidatorConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[GPSCoordinateValidatorConfig] = None
_config_lock = threading.Lock()


def get_config() -> GPSCoordinateValidatorConfig:
    """Return the singleton GPSCoordinateValidatorConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_GCV_*`` environment variables.

    Returns:
        GPSCoordinateValidatorConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.min_decimal_places_eudr
        5
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = GPSCoordinateValidatorConfig.from_env()
    return _config_instance


def set_config(config: GPSCoordinateValidatorConfig) -> None:
    """Replace the singleton GPSCoordinateValidatorConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New GPSCoordinateValidatorConfig to install.

    Example:
        >>> cfg = GPSCoordinateValidatorConfig(min_decimal_places_eudr=6)
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "GPSCoordinateValidatorConfig replaced programmatically: "
        "eudr_min_dp=%d, swap=%s, auto_correct=%s, batch_max=%d",
        config.min_decimal_places_eudr,
        config.swap_detection_enabled,
        config.auto_correction_enabled,
        config.batch_max_size,
    )


def reset_config() -> None:
    """Reset the singleton GPSCoordinateValidatorConfig to None.

    The next call to get_config() will re-read GL_EUDR_GCV_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("GPSCoordinateValidatorConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "GPSCoordinateValidatorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
