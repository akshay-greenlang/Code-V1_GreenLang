# -*- coding: utf-8 -*-
"""
Geolocation Verification Configuration - AGENT-EUDR-002

Centralized configuration for the Geolocation Verification Agent covering:
- Coordinate validation precision thresholds
- Polygon topology verification limits
- Protected area screening settings (WDPA update intervals, buffer zones)
- Deforestation cutoff date and satellite data parameters
- Accuracy scoring weights and quality tier thresholds
- Batch verification pipeline concurrency and timeout settings
- Verification result caching TTL
- Database and cache connection settings
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_GEO_`` prefix (e.g. ``GL_EUDR_GEO_DATABASE_URL``,
``GL_EUDR_GEO_COORDINATE_PRECISION_MIN_DECIMALS``).

Environment Variable Reference (GL_EUDR_GEO_ prefix):
    GL_EUDR_GEO_DATABASE_URL                        - PostgreSQL connection URL
    GL_EUDR_GEO_REDIS_URL                           - Redis connection URL
    GL_EUDR_GEO_LOG_LEVEL                           - Logging level (DEBUG/INFO/WARNING/ERROR)
    GL_EUDR_GEO_COORDINATE_PRECISION_MIN_DECIMALS   - Minimum decimal places for coordinates
    GL_EUDR_GEO_POLYGON_AREA_TOLERANCE_PCT          - Area tolerance percentage (+/- declared)
    GL_EUDR_GEO_MAX_BATCH_CONCURRENCY               - Maximum concurrent batch verifications
    GL_EUDR_GEO_VERIFICATION_CACHE_TTL_SECONDS      - Cache TTL for verification results
    GL_EUDR_GEO_WDPA_UPDATE_INTERVAL_DAYS           - Days between WDPA dataset updates
    GL_EUDR_GEO_DEFORESTATION_CUTOFF_DATE           - EUDR deforestation cutoff date
    GL_EUDR_GEO_SCORE_WEIGHT_PRECISION              - Accuracy score: coordinate precision weight
    GL_EUDR_GEO_SCORE_WEIGHT_POLYGON                - Accuracy score: polygon quality weight
    GL_EUDR_GEO_SCORE_WEIGHT_COUNTRY                - Accuracy score: country match weight
    GL_EUDR_GEO_SCORE_WEIGHT_PROTECTED              - Accuracy score: protected area weight
    GL_EUDR_GEO_SCORE_WEIGHT_DEFORESTATION          - Accuracy score: deforestation weight
    GL_EUDR_GEO_SCORE_WEIGHT_TEMPORAL               - Accuracy score: temporal consistency weight
    GL_EUDR_GEO_QUICK_TIMEOUT_SECONDS               - Timeout for QUICK verification level
    GL_EUDR_GEO_STANDARD_TIMEOUT_SECONDS            - Timeout for STANDARD verification level
    GL_EUDR_GEO_DEEP_TIMEOUT_SECONDS                - Timeout for DEEP verification level
    GL_EUDR_GEO_MAX_POLYGON_VERTICES                - Maximum allowed polygon vertices
    GL_EUDR_GEO_MIN_POLYGON_VERTICES                - Minimum required polygon vertices
    GL_EUDR_GEO_SLIVER_RATIO_THRESHOLD              - Area/perimeter^2 sliver threshold
    GL_EUDR_GEO_SPIKE_ANGLE_THRESHOLD_DEGREES       - Minimum angle for spike detection
    GL_EUDR_GEO_ELEVATION_MAX_M                     - Maximum plausible elevation (meters)
    GL_EUDR_GEO_COUNTRY_BOUNDARY_BUFFER_KM          - Country boundary buffer zone (km)
    GL_EUDR_GEO_DUPLICATE_DISTANCE_THRESHOLD_M      - Duplicate coordinate distance (meters)
    GL_EUDR_GEO_ENABLE_PROVENANCE                   - Enable SHA-256 provenance chain
    GL_EUDR_GEO_GENESIS_HASH                        - Genesis anchor string for provenance
    GL_EUDR_GEO_ENABLE_METRICS                      - Enable Prometheus metrics export
    GL_EUDR_GEO_POOL_SIZE                           - Database connection pool size
    GL_EUDR_GEO_RATE_LIMIT                          - Max API requests per minute

Example:
    >>> from greenlang.agents.eudr.geolocation_verification.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.coordinate_precision_min_decimals)
    5

    >>> # Override for testing
    >>> from greenlang.agents.eudr.geolocation_verification.config import (
    ...     set_config, reset_config, GeolocationVerificationConfig,
    ... )
    >>> set_config(GeolocationVerificationConfig(coordinate_precision_min_decimals=6))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
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

_ENV_PREFIX = "GL_EUDR_GEO_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Default score weights (must sum to 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "precision": 0.20,
    "polygon": 0.20,
    "country": 0.15,
    "protected": 0.15,
    "deforestation": 0.15,
    "temporal": 0.15,
}


# ---------------------------------------------------------------------------
# GeolocationVerificationConfig
# ---------------------------------------------------------------------------


@dataclass
class GeolocationVerificationConfig:
    """Complete configuration for the EUDR Geolocation Verification Agent.

    Attributes are grouped by concern: connections, logging, coordinate
    validation, polygon topology, protected area screening, deforestation
    verification, accuracy scoring, batch pipeline, timeouts, provenance
    tracking, and metrics export.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_GEO_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage
            of verification results, accuracy scores, and compliance
            snapshots.
        redis_url: Redis connection URL for verification result caching
            and rate limit tracking.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        coordinate_precision_min_decimals: Minimum number of decimal
            places required for coordinate precision validation.
            5 decimals gives ~1.1m accuracy at the equator.
        polygon_area_tolerance_pct: Percentage tolerance for comparing
            calculated geodesic area to declared area. A value of 10.0
            means the calculated area must be within +/-10% of declared.
        max_batch_concurrency: Maximum number of plots processed
            concurrently in a batch verification job.
        verification_cache_ttl_seconds: Time-to-live in seconds for
            cached verification results in Redis.
        wdpa_update_interval_days: Interval in days between WDPA
            protected area dataset refreshes.
        deforestation_cutoff_date: EUDR deforestation cutoff date
            string in ISO format (YYYY-MM-DD). Per Article 2(1),
            this is December 31, 2020.
        score_weights: Dictionary of accuracy score component weights.
            Keys: precision, polygon, country, protected, deforestation,
            temporal. Values must sum to 1.0.
        quick_timeout_seconds: Maximum time in seconds for QUICK
            verification level per plot.
        standard_timeout_seconds: Maximum time in seconds for STANDARD
            verification level per plot.
        deep_timeout_seconds: Maximum time in seconds for DEEP
            verification level per plot.
        max_polygon_vertices: Maximum allowed vertices in a polygon
            boundary. Polygons exceeding this are rejected.
        min_polygon_vertices: Minimum required vertices for a valid
            polygon (3 unique + 1 closure = 4).
        sliver_ratio_threshold: Area/perimeter^2 ratio threshold below
            which a polygon is classified as a sliver.
        spike_angle_threshold_degrees: Minimum interior angle in degrees
            below which a vertex is flagged as a spike.
        elevation_max_m: Maximum plausible elevation in meters for
            production plot coordinates.
        country_boundary_buffer_km: Buffer zone in kilometers around
            country boundaries for proximity matching.
        duplicate_distance_threshold_m: Distance in meters below which
            two coordinates are considered duplicates.
        enable_provenance: Enable SHA-256 provenance chain tracking for
            all verification operations.
        genesis_hash: Anchor string for the provenance chain, unique
            to the Geolocation Verification agent.
        enable_metrics: Enable Prometheus metrics export under the
            ``gl_eudr_geo_`` prefix.
        pool_size: PostgreSQL connection pool size.
        rate_limit: Maximum inbound API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Coordinate validation -----------------------------------------------
    coordinate_precision_min_decimals: int = 5
    duplicate_distance_threshold_m: float = 10.0
    elevation_max_m: float = 6000.0
    country_boundary_buffer_km: float = 5.0

    # -- Polygon topology verification ---------------------------------------
    polygon_area_tolerance_pct: float = 10.0
    max_polygon_vertices: int = 100_000
    min_polygon_vertices: int = 4
    sliver_ratio_threshold: float = 0.001
    spike_angle_threshold_degrees: float = 1.0

    # -- Protected area screening --------------------------------------------
    wdpa_update_interval_days: int = 90

    # -- Deforestation verification ------------------------------------------
    deforestation_cutoff_date: str = "2020-12-31"

    # -- Accuracy scoring weights (must sum to 1.0) --------------------------
    score_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_SCORE_WEIGHTS)
    )

    # -- Batch pipeline settings ---------------------------------------------
    max_batch_concurrency: int = 50

    # -- Verification level timeouts -----------------------------------------
    quick_timeout_seconds: float = 5.0
    standard_timeout_seconds: float = 30.0
    deep_timeout_seconds: float = 120.0

    # -- Caching -------------------------------------------------------------
    verification_cache_ttl_seconds: int = 3600

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-GEO-002-GEOLOCATION-VERIFICATION-GENESIS"

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
        on string fields, score weight sum validation, and
        normalization. Collects all errors before raising a single
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

        # -- Coordinate validation -------------------------------------------
        if not (1 <= self.coordinate_precision_min_decimals <= 15):
            errors.append(
                f"coordinate_precision_min_decimals must be in [1, 15], "
                f"got {self.coordinate_precision_min_decimals}"
            )
        if self.duplicate_distance_threshold_m < 0:
            errors.append(
                f"duplicate_distance_threshold_m must be >= 0, "
                f"got {self.duplicate_distance_threshold_m}"
            )
        if not (0.0 < self.elevation_max_m <= 9000.0):
            errors.append(
                f"elevation_max_m must be in (0, 9000], "
                f"got {self.elevation_max_m}"
            )
        if self.country_boundary_buffer_km < 0:
            errors.append(
                f"country_boundary_buffer_km must be >= 0, "
                f"got {self.country_boundary_buffer_km}"
            )

        # -- Polygon topology ------------------------------------------------
        if not (0.0 < self.polygon_area_tolerance_pct <= 100.0):
            errors.append(
                f"polygon_area_tolerance_pct must be in (0, 100], "
                f"got {self.polygon_area_tolerance_pct}"
            )
        if self.max_polygon_vertices < 4:
            errors.append(
                f"max_polygon_vertices must be >= 4, "
                f"got {self.max_polygon_vertices}"
            )
        if not (3 <= self.min_polygon_vertices <= 100):
            errors.append(
                f"min_polygon_vertices must be in [3, 100], "
                f"got {self.min_polygon_vertices}"
            )
        if self.min_polygon_vertices > self.max_polygon_vertices:
            errors.append(
                f"min_polygon_vertices ({self.min_polygon_vertices}) must be "
                f"<= max_polygon_vertices ({self.max_polygon_vertices})"
            )
        if not (0.0 < self.sliver_ratio_threshold <= 1.0):
            errors.append(
                f"sliver_ratio_threshold must be in (0, 1], "
                f"got {self.sliver_ratio_threshold}"
            )
        if not (0.0 < self.spike_angle_threshold_degrees <= 45.0):
            errors.append(
                f"spike_angle_threshold_degrees must be in (0, 45], "
                f"got {self.spike_angle_threshold_degrees}"
            )

        # -- Protected area screening ----------------------------------------
        if self.wdpa_update_interval_days <= 0:
            errors.append(
                f"wdpa_update_interval_days must be > 0, "
                f"got {self.wdpa_update_interval_days}"
            )

        # -- Score weights ---------------------------------------------------
        expected_keys = {
            "precision", "polygon", "country",
            "protected", "deforestation", "temporal",
        }
        actual_keys = set(self.score_weights.keys())
        if actual_keys != expected_keys:
            errors.append(
                f"score_weights must have keys {sorted(expected_keys)}, "
                f"got {sorted(actual_keys)}"
            )
        else:
            for key, value in self.score_weights.items():
                if not (0.0 <= value <= 1.0):
                    errors.append(
                        f"score_weights['{key}'] must be in [0.0, 1.0], "
                        f"got {value}"
                    )
            weight_sum = sum(self.score_weights.values())
            if abs(weight_sum - 1.0) > 0.001:
                errors.append(
                    f"score_weights must sum to 1.0, got {weight_sum:.4f}"
                )

        # -- Batch pipeline --------------------------------------------------
        if not (1 <= self.max_batch_concurrency <= 1000):
            errors.append(
                f"max_batch_concurrency must be in [1, 1000], "
                f"got {self.max_batch_concurrency}"
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

        # -- Caching ---------------------------------------------------------
        if self.verification_cache_ttl_seconds <= 0:
            errors.append(
                f"verification_cache_ttl_seconds must be > 0, "
                f"got {self.verification_cache_ttl_seconds}"
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
                "GeolocationVerificationConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "GeolocationVerificationConfig validated successfully: "
            "min_decimals=%d, area_tolerance=%.1f%%, "
            "max_batch=%d, cache_ttl=%ds, "
            "wdpa_interval=%dd, cutoff=%s, "
            "provenance=%s, metrics=%s",
            self.coordinate_precision_min_decimals,
            self.polygon_area_tolerance_pct,
            self.max_batch_concurrency,
            self.verification_cache_ttl_seconds,
            self.wdpa_update_interval_days,
            self.deforestation_cutoff_date,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> GeolocationVerificationConfig:
        """Build a GeolocationVerificationConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_GEO_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated GeolocationVerificationConfig instance, validated
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

        # Build score weights from env or defaults
        default_weights = dict(_DEFAULT_SCORE_WEIGHTS)
        score_weights = {
            "precision": _float(
                "SCORE_WEIGHT_PRECISION", default_weights["precision"],
            ),
            "polygon": _float(
                "SCORE_WEIGHT_POLYGON", default_weights["polygon"],
            ),
            "country": _float(
                "SCORE_WEIGHT_COUNTRY", default_weights["country"],
            ),
            "protected": _float(
                "SCORE_WEIGHT_PROTECTED", default_weights["protected"],
            ),
            "deforestation": _float(
                "SCORE_WEIGHT_DEFORESTATION",
                default_weights["deforestation"],
            ),
            "temporal": _float(
                "SCORE_WEIGHT_TEMPORAL", default_weights["temporal"],
            ),
        }

        config = cls(
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Coordinate validation
            coordinate_precision_min_decimals=_int(
                "COORDINATE_PRECISION_MIN_DECIMALS",
                cls.coordinate_precision_min_decimals,
            ),
            duplicate_distance_threshold_m=_float(
                "DUPLICATE_DISTANCE_THRESHOLD_M",
                cls.duplicate_distance_threshold_m,
            ),
            elevation_max_m=_float(
                "ELEVATION_MAX_M", cls.elevation_max_m,
            ),
            country_boundary_buffer_km=_float(
                "COUNTRY_BOUNDARY_BUFFER_KM",
                cls.country_boundary_buffer_km,
            ),
            # Polygon topology
            polygon_area_tolerance_pct=_float(
                "POLYGON_AREA_TOLERANCE_PCT",
                cls.polygon_area_tolerance_pct,
            ),
            max_polygon_vertices=_int(
                "MAX_POLYGON_VERTICES", cls.max_polygon_vertices,
            ),
            min_polygon_vertices=_int(
                "MIN_POLYGON_VERTICES", cls.min_polygon_vertices,
            ),
            sliver_ratio_threshold=_float(
                "SLIVER_RATIO_THRESHOLD", cls.sliver_ratio_threshold,
            ),
            spike_angle_threshold_degrees=_float(
                "SPIKE_ANGLE_THRESHOLD_DEGREES",
                cls.spike_angle_threshold_degrees,
            ),
            # Protected area screening
            wdpa_update_interval_days=_int(
                "WDPA_UPDATE_INTERVAL_DAYS",
                cls.wdpa_update_interval_days,
            ),
            # Deforestation verification
            deforestation_cutoff_date=_str(
                "DEFORESTATION_CUTOFF_DATE",
                cls.deforestation_cutoff_date,
            ),
            # Accuracy scoring
            score_weights=score_weights,
            # Batch pipeline
            max_batch_concurrency=_int(
                "MAX_BATCH_CONCURRENCY", cls.max_batch_concurrency,
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
            # Caching
            verification_cache_ttl_seconds=_int(
                "VERIFICATION_CACHE_TTL_SECONDS",
                cls.verification_cache_ttl_seconds,
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
            "GeolocationVerificationConfig loaded: "
            "min_decimals=%d, area_tolerance=%.1f%%, "
            "max_batch=%d, cache_ttl=%ds, "
            "wdpa_interval=%dd, cutoff=%s, "
            "score_weights=%s, "
            "quick_timeout=%.1fs, standard_timeout=%.1fs, deep_timeout=%.1fs, "
            "max_vertices=%d, min_vertices=%d, "
            "sliver=%.4f, spike=%.1fdeg, "
            "elevation_max=%.0fm, buffer=%.1fkm, dup_dist=%.1fm, "
            "provenance=%s, pool=%d, rate_limit=%d/min, metrics=%s",
            config.coordinate_precision_min_decimals,
            config.polygon_area_tolerance_pct,
            config.max_batch_concurrency,
            config.verification_cache_ttl_seconds,
            config.wdpa_update_interval_days,
            config.deforestation_cutoff_date,
            config.score_weights,
            config.quick_timeout_seconds,
            config.standard_timeout_seconds,
            config.deep_timeout_seconds,
            config.max_polygon_vertices,
            config.min_polygon_vertices,
            config.sliver_ratio_threshold,
            config.spike_angle_threshold_degrees,
            config.elevation_max_m,
            config.country_boundary_buffer_km,
            config.duplicate_distance_threshold_m,
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
        """Return verification timeout mapping by level name.

        Returns:
            Dictionary with keys: quick, standard, deep.
        """
        return {
            "quick": self.quick_timeout_seconds,
            "standard": self.standard_timeout_seconds,
            "deep": self.deep_timeout_seconds,
        }

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
            # Coordinate validation
            "coordinate_precision_min_decimals": self.coordinate_precision_min_decimals,
            "duplicate_distance_threshold_m": self.duplicate_distance_threshold_m,
            "elevation_max_m": self.elevation_max_m,
            "country_boundary_buffer_km": self.country_boundary_buffer_km,
            # Polygon topology
            "polygon_area_tolerance_pct": self.polygon_area_tolerance_pct,
            "max_polygon_vertices": self.max_polygon_vertices,
            "min_polygon_vertices": self.min_polygon_vertices,
            "sliver_ratio_threshold": self.sliver_ratio_threshold,
            "spike_angle_threshold_degrees": self.spike_angle_threshold_degrees,
            # Protected area screening
            "wdpa_update_interval_days": self.wdpa_update_interval_days,
            # Deforestation verification
            "deforestation_cutoff_date": self.deforestation_cutoff_date,
            # Accuracy scoring
            "score_weights": dict(self.score_weights),
            # Batch pipeline
            "max_batch_concurrency": self.max_batch_concurrency,
            # Timeouts
            "quick_timeout_seconds": self.quick_timeout_seconds,
            "standard_timeout_seconds": self.standard_timeout_seconds,
            "deep_timeout_seconds": self.deep_timeout_seconds,
            # Caching
            "verification_cache_ttl_seconds": self.verification_cache_ttl_seconds,
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
        return f"GeolocationVerificationConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[GeolocationVerificationConfig] = None
_config_lock = threading.Lock()


def get_config() -> GeolocationVerificationConfig:
    """Return the singleton GeolocationVerificationConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_GEO_*`` environment variables.

    Returns:
        GeolocationVerificationConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.coordinate_precision_min_decimals
        5
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = GeolocationVerificationConfig.from_env()
    return _config_instance


def set_config(config: GeolocationVerificationConfig) -> None:
    """Replace the singleton GeolocationVerificationConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New GeolocationVerificationConfig to install.

    Example:
        >>> cfg = GeolocationVerificationConfig(
        ...     coordinate_precision_min_decimals=6,
        ... )
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "GeolocationVerificationConfig replaced programmatically: "
        "min_decimals=%d, area_tolerance=%.1f%%, max_batch=%d",
        config.coordinate_precision_min_decimals,
        config.polygon_area_tolerance_pct,
        config.max_batch_concurrency,
    )


def reset_config() -> None:
    """Reset the singleton GeolocationVerificationConfig to None.

    The next call to get_config() will re-read GL_EUDR_GEO_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("GeolocationVerificationConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "GeolocationVerificationConfig",
    "get_config",
    "set_config",
    "reset_config",
]
