# -*- coding: utf-8 -*-
"""
Forest Cover Analysis Configuration - AGENT-EUDR-004

Centralized configuration for the Forest Cover Analysis Agent covering:
- FAO forest definition thresholds (canopy cover, tree height, minimum area)
- EUDR deforestation cutoff date and baseline composite window
- Degradation detection sensitivity thresholds
- Remote sensing API credentials (NASA GEDI, ESA CCI Biomass)
- Hansen Global Forest Change dataset version selection
- Analysis pipeline concurrency and batch size limits
- Cache TTL settings (general analysis and biomass-specific)
- Minimum confidence thresholds for deforestation-free determination
- Database and cache connection settings
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_FCA_`` prefix (e.g. ``GL_EUDR_FCA_DATABASE_URL``,
``GL_EUDR_FCA_CANOPY_COVER_THRESHOLD``).

Environment Variable Reference (GL_EUDR_FCA_ prefix):
    GL_EUDR_FCA_DATABASE_URL               - PostgreSQL connection URL
    GL_EUDR_FCA_REDIS_URL                  - Redis connection URL
    GL_EUDR_FCA_LOG_LEVEL                  - Logging level (DEBUG/INFO/WARNING/ERROR)
    GL_EUDR_FCA_CANOPY_COVER_THRESHOLD     - FAO forest canopy cover threshold (%)
    GL_EUDR_FCA_TREE_HEIGHT_THRESHOLD      - FAO tree height at maturity threshold (m)
    GL_EUDR_FCA_MIN_FOREST_AREA_HA         - FAO minimum forest area threshold (ha)
    GL_EUDR_FCA_DEGRADATION_THRESHOLD      - Canopy loss % to classify as degradation
    GL_EUDR_FCA_CUTOFF_DATE                - EUDR deforestation cutoff date (ISO)
    GL_EUDR_FCA_BASELINE_WINDOW_YEARS      - Years for baseline composite window
    GL_EUDR_FCA_GEDI_API_KEY               - NASA GEDI L2A/L4A data API key
    GL_EUDR_FCA_ESA_CCI_API_KEY            - ESA CCI Biomass product API key
    GL_EUDR_FCA_HANSEN_GFC_VERSION         - Hansen Global Forest Change version
    GL_EUDR_FCA_MAX_BATCH_SIZE             - Maximum plots in a single batch
    GL_EUDR_FCA_ANALYSIS_CONCURRENCY       - Max concurrent analysis operations
    GL_EUDR_FCA_CACHE_TTL_SECONDS          - General analysis cache TTL
    GL_EUDR_FCA_BIOMASS_CACHE_TTL_SECONDS  - Biomass estimate cache TTL
    GL_EUDR_FCA_CONFIDENCE_MIN             - Minimum confidence for determination
    GL_EUDR_FCA_GENESIS_HASH               - Genesis anchor for provenance chain
    GL_EUDR_FCA_ENABLE_METRICS             - Enable Prometheus metrics export

Example:
    >>> from greenlang.agents.eudr.forest_cover_analysis.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.canopy_cover_threshold)
    10.0

    >>> # Override for testing
    >>> from greenlang.agents.eudr.forest_cover_analysis.config import (
    ...     set_config, reset_config, ForestCoverConfig,
    ... )
    >>> set_config(ForestCoverConfig(canopy_cover_threshold=15.0))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_FCA_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# FAO Forest Definition Reference
# ---------------------------------------------------------------------------
# The Food and Agriculture Organization of the United Nations defines
# "forest" as land spanning more than 0.5 hectares with trees higher
# than 5 metres and a canopy cover of more than 10 per cent, or trees
# able to reach these thresholds in situ. These thresholds are the
# defaults for the Forest Cover Analysis Agent and are referenced in
# EUDR Recital 32 and the EU Observatory guidance.
# ---------------------------------------------------------------------------

_FAO_CANOPY_COVER_THRESHOLD: float = 10.0  # percent
_FAO_TREE_HEIGHT_THRESHOLD: float = 5.0  # metres
_FAO_MIN_FOREST_AREA_HA: float = 0.5  # hectares

# ---------------------------------------------------------------------------
# Default Hansen GFC version
# ---------------------------------------------------------------------------

_DEFAULT_HANSEN_GFC_VERSION: str = "v1.11"


# ---------------------------------------------------------------------------
# ForestCoverConfig
# ---------------------------------------------------------------------------


@dataclass
class ForestCoverConfig:
    """Complete configuration for the EUDR Forest Cover Analysis Agent.

    Attributes are grouped by concern: connections, logging, FAO forest
    definition thresholds, EUDR cutoff and baseline window, degradation
    detection, remote sensing API credentials, Hansen GFC dataset,
    analysis pipeline, caching, confidence thresholds, provenance
    tracking, and metrics export.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_FCA_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage
            of forest cover analysis results, historical reconstructions,
            and compliance reports.
        redis_url: Redis connection URL for analysis result caching
            and rate limit tracking.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        canopy_cover_threshold: FAO forest definition threshold for
            canopy cover as a percentage (0-100). Land with canopy
            cover above this threshold qualifies as forest. Default
            is 10.0% per FAO Global Forest Resources Assessment.
        tree_height_threshold: FAO forest definition threshold for
            tree height at maturity in metres. Trees must exceed this
            height (or be able to reach it in situ) for the land to
            qualify as forest. Default is 5.0m.
        min_forest_area_ha: FAO forest definition threshold for minimum
            contiguous forest area in hectares. Land parcels below
            this size are not classified as forest. Default is 0.5 ha.
        degradation_threshold: Percentage of canopy cover loss (within
            a plot that remains classified as forest) that triggers a
            degradation classification. Default is 30.0%.
        cutoff_date: EUDR deforestation cutoff date in ISO format
            (YYYY-MM-DD). Per Article 2(1), this is December 31, 2020.
        baseline_window_years: Number of years of historical imagery
            composited to establish a forest cover baseline around the
            cutoff date (e.g. 3 means 2018-2020 composite).
        gedi_api_key: NASA GEDI (Global Ecosystem Dynamics
            Investigation) API key for accessing L2A canopy height
            and L4A above-ground biomass data.
        esa_cci_api_key: ESA Climate Change Initiative Biomass product
            API key for accessing global above-ground biomass maps.
        hansen_gfc_version: Version of the Hansen et al. Global Forest
            Change dataset used for tree cover loss detection. Default
            is v1.11 (latest as of 2024).
        max_batch_size: Maximum number of plots allowed in a single
            batch analysis request.
        analysis_concurrency: Maximum number of plots analyzed
            concurrently in the analysis pipeline.
        cache_ttl_seconds: Time-to-live in seconds for cached analysis
            results (canopy density, classification, fragmentation).
        biomass_cache_ttl_seconds: Time-to-live in seconds for cached
            biomass estimates. Longer TTL because biomass data changes
            slowly. Default is 86400 seconds (24 hours).
        confidence_min: Minimum confidence score (0.0-1.0) required
            for a deforestation-free determination to be accepted.
            Analyses below this threshold return INCONCLUSIVE.
        genesis_hash: Anchor string for the SHA-256 provenance chain,
            unique to the Forest Cover Analysis agent. Auto-hashed
            on initialization.
        enable_metrics: Enable Prometheus metrics export under the
            ``gl_eudr_fca_`` prefix.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- FAO forest definition thresholds ------------------------------------
    canopy_cover_threshold: float = _FAO_CANOPY_COVER_THRESHOLD
    tree_height_threshold: float = _FAO_TREE_HEIGHT_THRESHOLD
    min_forest_area_ha: float = _FAO_MIN_FOREST_AREA_HA

    # -- Degradation detection -----------------------------------------------
    degradation_threshold: float = 30.0

    # -- EUDR cutoff and baseline --------------------------------------------
    cutoff_date: str = "2020-12-31"
    baseline_window_years: int = 3

    # -- Remote sensing API credentials --------------------------------------
    gedi_api_key: str = ""
    esa_cci_api_key: str = ""

    # -- Hansen Global Forest Change -----------------------------------------
    hansen_gfc_version: str = _DEFAULT_HANSEN_GFC_VERSION

    # -- Analysis pipeline ---------------------------------------------------
    max_batch_size: int = 5000
    analysis_concurrency: int = 8

    # -- Caching -------------------------------------------------------------
    cache_ttl_seconds: int = 3600
    biomass_cache_ttl_seconds: int = 86400

    # -- Confidence thresholds -----------------------------------------------
    confidence_min: float = 0.6

    # -- Provenance tracking -------------------------------------------------
    genesis_hash: str = "GL-EUDR-FCA-004-FOREST-COVER-ANALYSIS-GENESIS"

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

        # -- FAO forest definition thresholds --------------------------------
        if not (0.0 <= self.canopy_cover_threshold <= 100.0):
            errors.append(
                f"canopy_cover_threshold must be in [0.0, 100.0], "
                f"got {self.canopy_cover_threshold}"
            )

        if self.tree_height_threshold <= 0.0:
            errors.append(
                f"tree_height_threshold must be > 0, "
                f"got {self.tree_height_threshold}"
            )

        if self.min_forest_area_ha <= 0.0:
            errors.append(
                f"min_forest_area_ha must be > 0, "
                f"got {self.min_forest_area_ha}"
            )

        # -- Degradation detection -------------------------------------------
        if not (0.0 <= self.degradation_threshold <= 100.0):
            errors.append(
                f"degradation_threshold must be in [0.0, 100.0], "
                f"got {self.degradation_threshold}"
            )

        # -- Cutoff and baseline ---------------------------------------------
        if self.baseline_window_years <= 0:
            errors.append(
                f"baseline_window_years must be > 0, "
                f"got {self.baseline_window_years}"
            )

        # Validate cutoff_date format
        try:
            date.fromisoformat(self.cutoff_date)
        except (ValueError, TypeError):
            errors.append(
                f"cutoff_date must be a valid ISO date (YYYY-MM-DD), "
                f"got '{self.cutoff_date}'"
            )

        # -- Hansen GFC version ----------------------------------------------
        if not self.hansen_gfc_version:
            errors.append("hansen_gfc_version must not be empty")

        # -- Analysis pipeline -----------------------------------------------
        if not (1 <= self.max_batch_size <= 100_000):
            errors.append(
                f"max_batch_size must be in [1, 100000], "
                f"got {self.max_batch_size}"
            )

        if not (1 <= self.analysis_concurrency <= 256):
            errors.append(
                f"analysis_concurrency must be in [1, 256], "
                f"got {self.analysis_concurrency}"
            )

        # -- Caching ---------------------------------------------------------
        if self.cache_ttl_seconds <= 0:
            errors.append(
                f"cache_ttl_seconds must be > 0, "
                f"got {self.cache_ttl_seconds}"
            )

        if self.biomass_cache_ttl_seconds <= 0:
            errors.append(
                f"biomass_cache_ttl_seconds must be > 0, "
                f"got {self.biomass_cache_ttl_seconds}"
            )

        # -- Confidence thresholds -------------------------------------------
        if not (0.0 <= self.confidence_min <= 1.0):
            errors.append(
                f"confidence_min must be in [0.0, 1.0], "
                f"got {self.confidence_min}"
            )

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        if errors:
            raise ValueError(
                "ForestCoverConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "ForestCoverConfig validated successfully: "
            "canopy_threshold=%.1f%%, height_threshold=%.1fm, "
            "min_area=%.2fha, degradation_threshold=%.1f%%, "
            "cutoff=%s, baseline_window=%dy, "
            "hansen=%s, max_batch=%d, concurrency=%d, "
            "cache_ttl=%ds, biomass_cache_ttl=%ds, "
            "confidence_min=%.2f, metrics=%s",
            self.canopy_cover_threshold,
            self.tree_height_threshold,
            self.min_forest_area_ha,
            self.degradation_threshold,
            self.cutoff_date,
            self.baseline_window_years,
            self.hansen_gfc_version,
            self.max_batch_size,
            self.analysis_concurrency,
            self.cache_ttl_seconds,
            self.biomass_cache_ttl_seconds,
            self.confidence_min,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ForestCoverConfig:
        """Build a ForestCoverConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_FCA_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated ForestCoverConfig instance, validated
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
            # FAO forest definition thresholds
            canopy_cover_threshold=_float(
                "CANOPY_COVER_THRESHOLD", cls.canopy_cover_threshold,
            ),
            tree_height_threshold=_float(
                "TREE_HEIGHT_THRESHOLD", cls.tree_height_threshold,
            ),
            min_forest_area_ha=_float(
                "MIN_FOREST_AREA_HA", cls.min_forest_area_ha,
            ),
            # Degradation detection
            degradation_threshold=_float(
                "DEGRADATION_THRESHOLD", cls.degradation_threshold,
            ),
            # EUDR cutoff and baseline
            cutoff_date=_str("CUTOFF_DATE", cls.cutoff_date),
            baseline_window_years=_int(
                "BASELINE_WINDOW_YEARS", cls.baseline_window_years,
            ),
            # Remote sensing API credentials
            gedi_api_key=_str("GEDI_API_KEY", cls.gedi_api_key),
            esa_cci_api_key=_str("ESA_CCI_API_KEY", cls.esa_cci_api_key),
            # Hansen GFC
            hansen_gfc_version=_str(
                "HANSEN_GFC_VERSION", cls.hansen_gfc_version,
            ),
            # Analysis pipeline
            max_batch_size=_int(
                "MAX_BATCH_SIZE", cls.max_batch_size,
            ),
            analysis_concurrency=_int(
                "ANALYSIS_CONCURRENCY", cls.analysis_concurrency,
            ),
            # Caching
            cache_ttl_seconds=_int(
                "CACHE_TTL_SECONDS", cls.cache_ttl_seconds,
            ),
            biomass_cache_ttl_seconds=_int(
                "BIOMASS_CACHE_TTL_SECONDS",
                cls.biomass_cache_ttl_seconds,
            ),
            # Confidence thresholds
            confidence_min=_float(
                "CONFIDENCE_MIN", cls.confidence_min,
            ),
            # Provenance
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Metrics
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
        )

        logger.info(
            "ForestCoverConfig loaded: "
            "canopy_threshold=%.1f%%, height_threshold=%.1fm, "
            "min_area=%.2fha, degradation_threshold=%.1f%%, "
            "cutoff=%s, baseline_window=%dy, "
            "hansen=%s, max_batch=%d, concurrency=%d, "
            "cache_ttl=%ds, biomass_cache_ttl=%ds, "
            "confidence_min=%.2f, metrics=%s",
            config.canopy_cover_threshold,
            config.tree_height_threshold,
            config.min_forest_area_ha,
            config.degradation_threshold,
            config.cutoff_date,
            config.baseline_window_years,
            config.hansen_gfc_version,
            config.max_batch_size,
            config.analysis_concurrency,
            config.cache_ttl_seconds,
            config.biomass_cache_ttl_seconds,
            config.confidence_min,
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
    def baseline_start_year(self) -> int:
        """Return the start year of the baseline composite window.

        Returns:
            Year marking the start of the baseline window, computed
            as cutoff_year - baseline_window_years + 1.
        """
        cutoff = self.cutoff_date_parsed
        return cutoff.year - self.baseline_window_years + 1

    @property
    def baseline_end_year(self) -> int:
        """Return the end year of the baseline composite window.

        Returns:
            Year marking the end of the baseline window (cutoff year).
        """
        return self.cutoff_date_parsed.year

    @property
    def fao_thresholds(self) -> Dict[str, float]:
        """Return FAO forest definition thresholds as a dictionary.

        Returns:
            Dictionary with keys: canopy_cover_pct, tree_height_m,
            min_area_ha.
        """
        return {
            "canopy_cover_pct": self.canopy_cover_threshold,
            "tree_height_m": self.tree_height_threshold,
            "min_area_ha": self.min_forest_area_ha,
        }

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

        Sensitive connection strings (database_url, redis_url) and
        API credentials (gedi_api_key, esa_cci_api_key) are redacted
        to prevent accidental credential leakage in logs, exception
        tracebacks, and monitoring dashboards.

        Returns:
            Dictionary representation with sensitive fields redacted.
        """
        return {
            # Connections (redacted)
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # Logging
            "log_level": self.log_level,
            # FAO forest definition thresholds
            "canopy_cover_threshold": self.canopy_cover_threshold,
            "tree_height_threshold": self.tree_height_threshold,
            "min_forest_area_ha": self.min_forest_area_ha,
            # Degradation detection
            "degradation_threshold": self.degradation_threshold,
            # EUDR cutoff and baseline
            "cutoff_date": self.cutoff_date,
            "baseline_window_years": self.baseline_window_years,
            # Remote sensing API credentials (redacted)
            "gedi_api_key": "***" if self.gedi_api_key else "",
            "esa_cci_api_key": "***" if self.esa_cci_api_key else "",
            # Hansen GFC
            "hansen_gfc_version": self.hansen_gfc_version,
            # Analysis pipeline
            "max_batch_size": self.max_batch_size,
            "analysis_concurrency": self.analysis_concurrency,
            # Caching
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "biomass_cache_ttl_seconds": self.biomass_cache_ttl_seconds,
            # Confidence thresholds
            "confidence_min": self.confidence_min,
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
        return f"ForestCoverConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[ForestCoverConfig] = None
_config_lock = threading.Lock()


def get_config() -> ForestCoverConfig:
    """Return the singleton ForestCoverConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_FCA_*`` environment variables.

    Returns:
        ForestCoverConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.canopy_cover_threshold
        10.0
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ForestCoverConfig.from_env()
    return _config_instance


def set_config(config: ForestCoverConfig) -> None:
    """Replace the singleton ForestCoverConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New ForestCoverConfig to install.

    Example:
        >>> cfg = ForestCoverConfig(canopy_cover_threshold=15.0)
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "ForestCoverConfig replaced programmatically: "
        "canopy_threshold=%.1f%%, confidence_min=%.2f, "
        "max_batch=%d",
        config.canopy_cover_threshold,
        config.confidence_min,
        config.max_batch_size,
    )


def reset_config() -> None:
    """Reset the singleton ForestCoverConfig to None.

    The next call to get_config() will re-read GL_EUDR_FCA_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("ForestCoverConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "ForestCoverConfig",
    "get_config",
    "set_config",
    "reset_config",
]
