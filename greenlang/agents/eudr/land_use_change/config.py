# -*- coding: utf-8 -*-
"""
Land Use Change Detector Configuration - AGENT-EUDR-005

Centralized configuration for the Land Use Change Detector Agent covering:
- Land use classification settings (classes, methods, confidence thresholds)
- Sentinel-2 spectral band selection for multi-spectral classification
- Transition detection settings (minimum area, granularity, precision)
- Temporal trajectory settings (depth, types, max time steps)
- EUDR cutoff date verification (cutoff date, search window, bias)
- Conversion risk assessment weights and tier thresholds
- Urban encroachment buffer distances and analysis parameters
- EUDR-regulated commodity list for compliance scoping
- Analysis pipeline concurrency and batch size limits
- Cache TTL settings for classification and transition results
- Database and cache connection settings
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_LUC_`` prefix (e.g. ``GL_EUDR_LUC_DATABASE_URL``,
``GL_EUDR_LUC_NUM_CLASSES``, ``GL_EUDR_LUC_CUTOFF_DATE``).

Environment Variable Reference (GL_EUDR_LUC_ prefix):
    GL_EUDR_LUC_DATABASE_URL                - PostgreSQL connection URL
    GL_EUDR_LUC_REDIS_URL                   - Redis connection URL
    GL_EUDR_LUC_LOG_LEVEL                   - Logging level (DEBUG/INFO/WARNING/ERROR)
    GL_EUDR_LUC_NUM_CLASSES                 - Number of land use classes (1-50)
    GL_EUDR_LUC_DEFAULT_METHOD              - Default classification method
    GL_EUDR_LUC_MIN_CONFIDENCE              - Minimum classification confidence (0-1)
    GL_EUDR_LUC_MIN_TRANSITION_AREA_HA      - Minimum transition area (hectares)
    GL_EUDR_LUC_TRANSITION_DATE_GRANULARITY - Transition date granularity
    GL_EUDR_LUC_DEFORESTATION_PRECISION_TARGET - Deforestation detection precision
    GL_EUDR_LUC_MIN_TEMPORAL_DEPTH_YEARS    - Minimum temporal depth (years)
    GL_EUDR_LUC_MAX_TIME_STEPS              - Maximum time steps in trajectory
    GL_EUDR_LUC_CUTOFF_DATE                 - EUDR cutoff date (ISO format)
    GL_EUDR_LUC_SEARCH_WINDOW_DAYS          - Cutoff search window (days)
    GL_EUDR_LUC_CONSERVATIVE_BIAS           - Conservative bias flag
    GL_EUDR_LUC_DEFAULT_BUFFER_KM           - Default urban buffer (km)
    GL_EUDR_LUC_MAX_BUFFER_KM              - Maximum urban buffer (km)
    GL_EUDR_LUC_BATCH_SIZE                  - Batch processing size
    GL_EUDR_LUC_MAX_CONCURRENT_JOBS         - Maximum concurrent analysis jobs
    GL_EUDR_LUC_CACHE_TTL_SECONDS           - General cache TTL (seconds)
    GL_EUDR_LUC_GENESIS_HASH                - Genesis anchor for provenance chain
    GL_EUDR_LUC_ENABLE_METRICS              - Enable Prometheus metrics export

Example:
    >>> from greenlang.agents.eudr.land_use_change.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.num_classes)
    10

    >>> # Override for testing
    >>> from greenlang.agents.eudr.land_use_change.config import (
    ...     set_config, reset_config, LandUseChangeConfig,
    ... )
    >>> set_config(LandUseChangeConfig(num_classes=15))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_LUC_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Valid classification methods
# ---------------------------------------------------------------------------

_VALID_CLASSIFICATION_METHODS = frozenset({
    "spectral",
    "vegetation_index",
    "phenology",
    "texture",
    "ensemble",
})

# ---------------------------------------------------------------------------
# Valid transition date granularities
# ---------------------------------------------------------------------------

_VALID_GRANULARITIES = frozenset({
    "daily",
    "weekly",
    "monthly",
    "quarterly",
    "annual",
})

# ---------------------------------------------------------------------------
# Valid trajectory types
# ---------------------------------------------------------------------------

_VALID_TRAJECTORY_TYPES = frozenset({
    "stable",
    "abrupt_change",
    "gradual_change",
    "oscillating",
    "recovery",
})

# ---------------------------------------------------------------------------
# Valid risk tiers
# ---------------------------------------------------------------------------

_VALID_RISK_TIERS = frozenset({
    "low",
    "moderate",
    "high",
    "critical",
})

# ---------------------------------------------------------------------------
# Default Sentinel-2 spectral bands for classification
# ---------------------------------------------------------------------------

_DEFAULT_SPECTRAL_BANDS: List[str] = [
    "B2",    # Blue (490 nm)
    "B3",    # Green (560 nm)
    "B4",    # Red (665 nm)
    "B5",    # Vegetation Red Edge 1 (705 nm)
    "B6",    # Vegetation Red Edge 2 (740 nm)
    "B7",    # Vegetation Red Edge 3 (783 nm)
    "B8",    # NIR (842 nm)
    "B8A",   # Narrow NIR (865 nm)
    "B11",   # SWIR 1 (1610 nm)
    "B12",   # SWIR 2 (2190 nm)
]

# ---------------------------------------------------------------------------
# Default classification methods list
# ---------------------------------------------------------------------------

_DEFAULT_CLASSIFICATION_METHODS: List[str] = [
    "spectral",
    "vegetation_index",
    "phenology",
    "texture",
    "ensemble",
]

# ---------------------------------------------------------------------------
# Default trajectory types list
# ---------------------------------------------------------------------------

_DEFAULT_TRAJECTORY_TYPES: List[str] = [
    "stable",
    "abrupt_change",
    "gradual_change",
    "oscillating",
    "recovery",
]

# ---------------------------------------------------------------------------
# Default risk tiers list
# ---------------------------------------------------------------------------

_DEFAULT_RISK_TIERS: List[str] = [
    "low",
    "moderate",
    "high",
    "critical",
]

# ---------------------------------------------------------------------------
# Default risk weights (8 factors)
# ---------------------------------------------------------------------------

_DEFAULT_RISK_WEIGHTS: Dict[str, float] = {
    "transition_magnitude": 0.20,
    "proximity_to_forest": 0.15,
    "historical_deforestation_rate": 0.15,
    "commodity_pressure": 0.12,
    "governance_score": 0.10,
    "protected_area_proximity": 0.10,
    "road_infrastructure_proximity": 0.08,
    "population_density_change": 0.10,
}

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
# LandUseChangeConfig
# ---------------------------------------------------------------------------


@dataclass
class LandUseChangeConfig:
    """Complete configuration for the EUDR Land Use Change Detector Agent.

    Attributes are grouped by concern: connections, logging, classification
    settings, transition detection settings, temporal trajectory settings,
    EUDR cutoff date verification, conversion risk assessment, urban
    encroachment analysis, EUDR commodity scope, performance tuning,
    caching, provenance tracking, and metrics export.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_LUC_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage
            of land use change detection results, transition matrices,
            and compliance reports.
        redis_url: Redis connection URL for classification result caching
            and rate limit tracking.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        num_classes: Number of land use classes for classification.
            Default is 10 corresponding to the LandUseCategory enum
            (forest, shrubland, grassland, cropland, wetland, water,
            urban, bare_soil, snow_ice, other).
        default_method: Default classification method to use when no
            specific method is requested. One of: spectral,
            vegetation_index, phenology, texture, ensemble.
        min_confidence: Minimum classification confidence score (0-1)
            required for a classification to be accepted. Classifications
            below this threshold are flagged for manual review.
        spectral_bands: List of Sentinel-2 spectral band identifiers
            used for multi-spectral land use classification.
        classification_methods: List of supported classification methods
            available for ensemble and single-method classification.
        min_transition_area_ha: Minimum area in hectares for a land use
            transition to be recorded. Transitions below this threshold
            are filtered as noise. Default is 0.1 ha.
        transition_date_granularity: Temporal granularity for transition
            date estimation. One of: daily, weekly, monthly, quarterly,
            annual. Default is monthly.
        deforestation_precision_target: Target precision (0-1) for
            deforestation detection. Higher values reduce false positives
            at the cost of false negatives. Default is 0.90.
        min_temporal_depth_years: Minimum number of years of temporal
            data required for trajectory analysis. Default is 3 years.
        trajectory_types: List of trajectory type labels supported
            by the temporal trajectory analyzer.
        max_time_steps: Maximum number of discrete time steps in a
            temporal trajectory analysis. Default is 60 (monthly
            steps over 5 years).
        cutoff_date: EUDR deforestation cutoff date in ISO format
            (YYYY-MM-DD). Per Article 2(1), this is December 31, 2020.
        search_window_days: Number of days around the cutoff date to
            search for cloud-free imagery for cutoff verification.
            Default is 60 days (30 days before and after).
        conservative_bias: When True, ambiguous cases are classified
            as potential deforestation (higher false positive rate but
            lower risk of missing deforestation). Default is True for
            regulatory compliance.
        risk_tiers: List of risk tier labels for conversion risk
            assessment, ordered from lowest to highest severity.
        risk_weights: Dictionary mapping risk factor names to their
            weights in the composite risk score calculation. Weights
            must sum to 1.0.
        default_buffer_km: Default buffer distance in kilometres around
            urban areas for encroachment analysis. Default is 10.0 km.
        max_buffer_km: Maximum allowed buffer distance in kilometres.
            Default is 50.0 km.
        eudr_commodities: List of EUDR-regulated commodity identifiers
            per Article 1(1) of EU 2023/1115.
        batch_size: Number of parcels processed per batch. Default is
            1000.
        max_concurrent_jobs: Maximum number of concurrent analysis jobs
            in the processing pipeline. Default is 10.
        cache_ttl_seconds: Time-to-live in seconds for cached
            classification and transition results. Default is 3600
            seconds (1 hour).
        genesis_hash: Anchor string for the SHA-256 provenance chain,
            unique to the Land Use Change Detector agent. Auto-hashed
            on initialization.
        enable_metrics: Enable Prometheus metrics export under the
            ``gl_eudr_luc_`` prefix.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Classification settings ---------------------------------------------
    num_classes: int = 10
    default_method: str = "ensemble"
    min_confidence: float = 0.60
    spectral_bands: List[str] = field(
        default_factory=lambda: list(_DEFAULT_SPECTRAL_BANDS)
    )
    classification_methods: List[str] = field(
        default_factory=lambda: list(_DEFAULT_CLASSIFICATION_METHODS)
    )

    # -- Transition detection settings ---------------------------------------
    min_transition_area_ha: float = 0.1
    transition_date_granularity: str = "monthly"
    deforestation_precision_target: float = 0.90

    # -- Temporal trajectory settings ----------------------------------------
    min_temporal_depth_years: int = 3
    trajectory_types: List[str] = field(
        default_factory=lambda: list(_DEFAULT_TRAJECTORY_TYPES)
    )
    max_time_steps: int = 60

    # -- EUDR cutoff date verification ---------------------------------------
    cutoff_date: str = "2020-12-31"
    search_window_days: int = 60
    conservative_bias: bool = True

    # -- Conversion risk assessment ------------------------------------------
    risk_tiers: List[str] = field(
        default_factory=lambda: list(_DEFAULT_RISK_TIERS)
    )
    risk_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_RISK_WEIGHTS)
    )

    # -- Urban encroachment analysis -----------------------------------------
    default_buffer_km: float = 10.0
    max_buffer_km: float = 50.0

    # -- EUDR commodity scope ------------------------------------------------
    eudr_commodities: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EUDR_COMMODITIES)
    )

    # -- Performance tuning --------------------------------------------------
    batch_size: int = 1000
    max_concurrent_jobs: int = 10

    # -- Caching -------------------------------------------------------------
    cache_ttl_seconds: int = 3600

    # -- Provenance tracking -------------------------------------------------
    genesis_hash: str = "GL-EUDR-LUC-005-LAND-USE-CHANGE-DETECTOR-GENESIS"

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

        # -- Classification settings -----------------------------------------
        if not (1 <= self.num_classes <= 50):
            errors.append(
                f"num_classes must be in [1, 50], "
                f"got {self.num_classes}"
            )

        if self.default_method not in _VALID_CLASSIFICATION_METHODS:
            errors.append(
                f"default_method must be one of "
                f"{sorted(_VALID_CLASSIFICATION_METHODS)}, "
                f"got '{self.default_method}'"
            )

        if not (0.0 <= self.min_confidence <= 1.0):
            errors.append(
                f"min_confidence must be in [0.0, 1.0], "
                f"got {self.min_confidence}"
            )

        if not self.spectral_bands:
            errors.append("spectral_bands must not be empty")

        if not self.classification_methods:
            errors.append("classification_methods must not be empty")

        for method in self.classification_methods:
            if method not in _VALID_CLASSIFICATION_METHODS:
                errors.append(
                    f"classification_methods contains invalid method "
                    f"'{method}'; valid methods are "
                    f"{sorted(_VALID_CLASSIFICATION_METHODS)}"
                )

        # -- Transition detection settings -----------------------------------
        if self.min_transition_area_ha < 0.0:
            errors.append(
                f"min_transition_area_ha must be >= 0.0, "
                f"got {self.min_transition_area_ha}"
            )

        if self.transition_date_granularity not in _VALID_GRANULARITIES:
            errors.append(
                f"transition_date_granularity must be one of "
                f"{sorted(_VALID_GRANULARITIES)}, "
                f"got '{self.transition_date_granularity}'"
            )

        if not (0.0 <= self.deforestation_precision_target <= 1.0):
            errors.append(
                f"deforestation_precision_target must be in [0.0, 1.0], "
                f"got {self.deforestation_precision_target}"
            )

        # -- Temporal trajectory settings ------------------------------------
        if self.min_temporal_depth_years <= 0:
            errors.append(
                f"min_temporal_depth_years must be > 0, "
                f"got {self.min_temporal_depth_years}"
            )

        if not self.trajectory_types:
            errors.append("trajectory_types must not be empty")

        for ttype in self.trajectory_types:
            if ttype not in _VALID_TRAJECTORY_TYPES:
                errors.append(
                    f"trajectory_types contains invalid type "
                    f"'{ttype}'; valid types are "
                    f"{sorted(_VALID_TRAJECTORY_TYPES)}"
                )

        if not (1 <= self.max_time_steps <= 1000):
            errors.append(
                f"max_time_steps must be in [1, 1000], "
                f"got {self.max_time_steps}"
            )

        # -- Cutoff date verification ----------------------------------------
        try:
            date.fromisoformat(self.cutoff_date)
        except (ValueError, TypeError):
            errors.append(
                f"cutoff_date must be a valid ISO date (YYYY-MM-DD), "
                f"got '{self.cutoff_date}'"
            )

        if not (1 <= self.search_window_days <= 365):
            errors.append(
                f"search_window_days must be in [1, 365], "
                f"got {self.search_window_days}"
            )

        # -- Risk assessment -------------------------------------------------
        if not self.risk_tiers:
            errors.append("risk_tiers must not be empty")

        for tier in self.risk_tiers:
            if tier not in _VALID_RISK_TIERS:
                errors.append(
                    f"risk_tiers contains invalid tier '{tier}'; "
                    f"valid tiers are {sorted(_VALID_RISK_TIERS)}"
                )

        if not self.risk_weights:
            errors.append("risk_weights must not be empty")
        else:
            weight_sum = sum(self.risk_weights.values())
            if abs(weight_sum - 1.0) > 0.01:
                errors.append(
                    f"risk_weights must sum to 1.0, "
                    f"got {weight_sum:.4f}"
                )
            for wname, wval in self.risk_weights.items():
                if wval < 0.0 or wval > 1.0:
                    errors.append(
                        f"risk_weight '{wname}' must be in [0.0, 1.0], "
                        f"got {wval}"
                    )

        # -- Urban encroachment ----------------------------------------------
        if self.default_buffer_km <= 0.0:
            errors.append(
                f"default_buffer_km must be > 0.0, "
                f"got {self.default_buffer_km}"
            )

        if self.max_buffer_km <= 0.0:
            errors.append(
                f"max_buffer_km must be > 0.0, "
                f"got {self.max_buffer_km}"
            )

        if self.default_buffer_km > self.max_buffer_km:
            errors.append(
                f"default_buffer_km ({self.default_buffer_km}) must be "
                f"<= max_buffer_km ({self.max_buffer_km})"
            )

        # -- EUDR commodity scope --------------------------------------------
        if not self.eudr_commodities:
            errors.append("eudr_commodities must not be empty")

        # -- Performance tuning ----------------------------------------------
        if not (1 <= self.batch_size <= 100_000):
            errors.append(
                f"batch_size must be in [1, 100000], "
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
                "LandUseChangeConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "LandUseChangeConfig validated successfully: "
            "num_classes=%d, default_method=%s, "
            "min_confidence=%.2f, min_transition_area=%.2fha, "
            "granularity=%s, precision_target=%.2f, "
            "temporal_depth=%dy, max_time_steps=%d, "
            "cutoff=%s, search_window=%dd, conservative=%s, "
            "default_buffer=%.1fkm, max_buffer=%.1fkm, "
            "batch_size=%d, concurrent=%d, "
            "cache_ttl=%ds, metrics=%s",
            self.num_classes,
            self.default_method,
            self.min_confidence,
            self.min_transition_area_ha,
            self.transition_date_granularity,
            self.deforestation_precision_target,
            self.min_temporal_depth_years,
            self.max_time_steps,
            self.cutoff_date,
            self.search_window_days,
            self.conservative_bias,
            self.default_buffer_km,
            self.max_buffer_km,
            self.batch_size,
            self.max_concurrent_jobs,
            self.cache_ttl_seconds,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> LandUseChangeConfig:
        """Build a LandUseChangeConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_LUC_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated LandUseChangeConfig instance, validated
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
            # Classification settings
            num_classes=_int("NUM_CLASSES", cls.num_classes),
            default_method=_str("DEFAULT_METHOD", cls.default_method),
            min_confidence=_float("MIN_CONFIDENCE", cls.min_confidence),
            # Transition detection
            min_transition_area_ha=_float(
                "MIN_TRANSITION_AREA_HA", cls.min_transition_area_ha,
            ),
            transition_date_granularity=_str(
                "TRANSITION_DATE_GRANULARITY",
                cls.transition_date_granularity,
            ),
            deforestation_precision_target=_float(
                "DEFORESTATION_PRECISION_TARGET",
                cls.deforestation_precision_target,
            ),
            # Temporal trajectory
            min_temporal_depth_years=_int(
                "MIN_TEMPORAL_DEPTH_YEARS",
                cls.min_temporal_depth_years,
            ),
            max_time_steps=_int(
                "MAX_TIME_STEPS", cls.max_time_steps,
            ),
            # Cutoff date verification
            cutoff_date=_str("CUTOFF_DATE", cls.cutoff_date),
            search_window_days=_int(
                "SEARCH_WINDOW_DAYS", cls.search_window_days,
            ),
            conservative_bias=_bool(
                "CONSERVATIVE_BIAS", cls.conservative_bias,
            ),
            # Urban encroachment
            default_buffer_km=_float(
                "DEFAULT_BUFFER_KM", cls.default_buffer_km,
            ),
            max_buffer_km=_float(
                "MAX_BUFFER_KM", cls.max_buffer_km,
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
            "LandUseChangeConfig loaded: "
            "num_classes=%d, default_method=%s, "
            "min_confidence=%.2f, min_transition_area=%.2fha, "
            "granularity=%s, precision_target=%.2f, "
            "temporal_depth=%dy, max_time_steps=%d, "
            "cutoff=%s, search_window=%dd, conservative=%s, "
            "default_buffer=%.1fkm, max_buffer=%.1fkm, "
            "batch_size=%d, concurrent=%d, "
            "cache_ttl=%ds, metrics=%s",
            config.num_classes,
            config.default_method,
            config.min_confidence,
            config.min_transition_area_ha,
            config.transition_date_granularity,
            config.deforestation_precision_target,
            config.min_temporal_depth_years,
            config.max_time_steps,
            config.cutoff_date,
            config.search_window_days,
            config.conservative_bias,
            config.default_buffer_km,
            config.max_buffer_km,
            config.batch_size,
            config.max_concurrent_jobs,
            config.cache_ttl_seconds,
            config.enable_metrics,
        )
        return config

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def cutoff_date_parsed(self) -> date:
        """Return the cutoff date as a Python date object.

        Returns:
            Parsed EUDR cutoff date.
        """
        return date.fromisoformat(self.cutoff_date)

    @property
    def search_window_start(self) -> date:
        """Return the start of the cutoff search window.

        Returns:
            Date marking the start of the search window, computed
            as cutoff_date - search_window_days / 2.
        """
        from datetime import timedelta
        half_window = self.search_window_days // 2
        return self.cutoff_date_parsed - timedelta(days=half_window)

    @property
    def search_window_end(self) -> date:
        """Return the end of the cutoff search window.

        Returns:
            Date marking the end of the search window, computed
            as cutoff_date + search_window_days / 2.
        """
        from datetime import timedelta
        half_window = self.search_window_days // 2
        return self.cutoff_date_parsed + timedelta(days=half_window)

    @property
    def risk_factor_names(self) -> List[str]:
        """Return sorted list of risk factor names.

        Returns:
            List of risk factor names from risk_weights keys.
        """
        return sorted(self.risk_weights.keys())

    @property
    def genesis_hash_sha256(self) -> str:
        """Return the SHA-256 hash of the genesis anchor string.

        Returns:
            Hex-encoded SHA-256 digest of the genesis_hash attribute.
        """
        return hashlib.sha256(
            self.genesis_hash.encode("utf-8")
        ).hexdigest()

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
            # Classification settings
            "num_classes": self.num_classes,
            "default_method": self.default_method,
            "min_confidence": self.min_confidence,
            "spectral_bands": self.spectral_bands,
            "classification_methods": self.classification_methods,
            # Transition detection
            "min_transition_area_ha": self.min_transition_area_ha,
            "transition_date_granularity": self.transition_date_granularity,
            "deforestation_precision_target": self.deforestation_precision_target,
            # Temporal trajectory
            "min_temporal_depth_years": self.min_temporal_depth_years,
            "trajectory_types": self.trajectory_types,
            "max_time_steps": self.max_time_steps,
            # Cutoff date verification
            "cutoff_date": self.cutoff_date,
            "search_window_days": self.search_window_days,
            "conservative_bias": self.conservative_bias,
            # Risk assessment
            "risk_tiers": self.risk_tiers,
            "risk_weights": self.risk_weights,
            # Urban encroachment
            "default_buffer_km": self.default_buffer_km,
            "max_buffer_km": self.max_buffer_km,
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

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Returns:
            String representation with sensitive fields redacted.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"LandUseChangeConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[LandUseChangeConfig] = None
_config_lock = threading.Lock()


def get_config() -> LandUseChangeConfig:
    """Return the singleton LandUseChangeConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_LUC_*`` environment variables.

    Returns:
        LandUseChangeConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.num_classes
        10
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = LandUseChangeConfig.from_env()
    return _config_instance


def set_config(config: LandUseChangeConfig) -> None:
    """Replace the singleton LandUseChangeConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New LandUseChangeConfig to install.

    Example:
        >>> cfg = LandUseChangeConfig(num_classes=15)
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "LandUseChangeConfig replaced programmatically: "
        "num_classes=%d, min_confidence=%.2f, "
        "batch_size=%d",
        config.num_classes,
        config.min_confidence,
        config.batch_size,
    )


def reset_config() -> None:
    """Reset the singleton LandUseChangeConfig to None.

    The next call to get_config() will re-read GL_EUDR_LUC_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("LandUseChangeConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "LandUseChangeConfig",
    "get_config",
    "set_config",
    "reset_config",
]
