# -*- coding: utf-8 -*-
"""
Time Series Gap Filler Agent Service Configuration - AGENT-DATA-014

Centralized configuration for the Time Series Gap Filler SDK covering:
- Database, cache connection defaults
- Record processing limits (max records, batch size)
- Maximum gap ratio and minimum data point requirements
- Default fill strategy and interpolation method selection
- Seasonal decomposition parameters (period, Holt-Winters smoothing)
- Cross-series correlation threshold and confidence threshold
- Feature toggles for seasonal and cross-series engines
- Worker, pool, cache, rate limiting, and provenance settings

All settings can be overridden via environment variables with the
``GL_TSGF_`` prefix (e.g. ``GL_TSGF_MAX_RECORDS``).

Example:
    >>> from greenlang.time_series_gap_filler.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_strategy, cfg.interpolation_method)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
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

_ENV_PREFIX = "GL_TSGF_"


# ---------------------------------------------------------------------------
# TimeSeriesGapFillerConfig
# ---------------------------------------------------------------------------


@dataclass
class TimeSeriesGapFillerConfig:
    """Complete configuration for the GreenLang Time Series Gap Filler SDK.

    Attributes are grouped by concern: connections, logging, record
    processing, gap tolerance, strategy selection, seasonal decomposition,
    smoothing parameters, quality thresholds, feature toggles, worker pool,
    cache, rate limiting, and provenance.

    All attributes can be overridden via environment variables using the
    ``GL_TSGF_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
            Holds time series data, gap metadata, fill audit records,
            and provenance chains.
        redis_url: Redis connection URL for caching layer.
            Used for fill result caching, rate limiting counters,
            and cross-series correlation caches.
        log_level: Logging level for the time series gap filler service.
            Accepts standard Python logging levels: DEBUG, INFO,
            WARNING, ERROR, CRITICAL.
        batch_size: Default batch size for record processing.
            Controls how many data points are processed in a single
            chunk during batch gap filling operations.
        max_records: Maximum records allowed in a single gap filling job.
            Prevents runaway processing on excessively large datasets.
        max_gap_ratio: Maximum allowable fraction of a series that may
            consist of gaps (0.0 to 1.0). Series exceeding this
            threshold are rejected rather than filled, since too few
            observations remain for reliable interpolation.
        min_data_points: Minimum number of non-null data points required
            before gap filling can be attempted. Series with fewer
            valid observations are rejected to avoid unreliable fills.
        default_strategy: Default strategy selection mode for gap
            filling. ``auto`` uses the strategy selector engine to
            pick the best method per gap; alternatives include
            ``linear``, ``cubic_spline``, ``seasonal``,
            ``moving_average``, ``exponential_smoothing``.
        interpolation_method: Default interpolation method when the
            strategy resolves to numeric interpolation. Supports
            ``linear``, ``cubic``, ``spline``, ``pchip``, ``akima``.
        seasonal_periods: Number of observations per seasonal cycle
            used by the seasonal decomposition engine. Set to 12
            for monthly data, 4 for quarterly, 52 for weekly, etc.
        smoothing_alpha: Holt-Winters exponential smoothing level
            coefficient (alpha). Controls the weight given to the
            most recent observation in the level component.
            Range 0.0 to 1.0.
        smoothing_beta: Holt-Winters exponential smoothing trend
            coefficient (beta). Controls the weight given to the
            most recent observation in the trend component.
            Range 0.0 to 1.0.
        smoothing_gamma: Holt-Winters exponential smoothing seasonal
            coefficient (gamma). Controls the weight given to the
            most recent observation in the seasonal component.
            Range 0.0 to 1.0.
        correlation_threshold: Minimum Pearson correlation coefficient
            (0.0 to 1.0) required between two series for cross-series
            gap filling to be applied. Prevents unreliable fills from
            weakly correlated reference series.
        confidence_threshold: Minimum confidence score (0.0 to 1.0)
            for a gap fill to be accepted. Fills below this threshold
            are flagged as low-confidence and may trigger manual review
            or alternative strategy fallback.
        enable_seasonal: Whether seasonal decomposition based gap
            filling is enabled. When True, the seasonal engine
            decomposes the series into trend, seasonal, and residual
            components for pattern-aware filling.
        enable_cross_series: Whether cross-series correlation based
            gap filling is enabled. When True, correlated reference
            series are used to inform fills in the target series.
        worker_count: Number of parallel workers for batch processing.
            Controls concurrency for multi-series gap filling jobs.
        pool_min_size: Minimum connection pool size for database
            connections. Ensures baseline availability under low load.
        pool_max_size: Maximum connection pool size for database
            connections. Caps resource usage under high concurrency.
        cache_ttl: Cache time-to-live in seconds for gap fill results.
            Prevents redundant re-computation when the same series
            is queried repeatedly within the TTL window.
        rate_limit_rpm: Rate limit in requests per minute for the
            gap filler API. Protects backend resources from excessive
            concurrent fill requests.
        enable_provenance: Whether SHA-256 provenance tracking is
            enabled. When True, every gap fill operation records a
            provenance chain including input hash, method used,
            parameters applied, and output hash for full auditability.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Record processing ---------------------------------------------------
    batch_size: int = 1000
    max_records: int = 100_000

    # -- Gap tolerance -------------------------------------------------------
    max_gap_ratio: float = 0.5
    min_data_points: int = 10

    # -- Strategy selection --------------------------------------------------
    default_strategy: str = "auto"
    interpolation_method: str = "linear"

    # -- Seasonal decomposition ----------------------------------------------
    seasonal_periods: int = 12

    # -- Holt-Winters smoothing parameters -----------------------------------
    smoothing_alpha: float = 0.3
    smoothing_beta: float = 0.1
    smoothing_gamma: float = 0.1

    # -- Quality thresholds --------------------------------------------------
    correlation_threshold: float = 0.7
    confidence_threshold: float = 0.6

    # -- Gap classification ---------------------------------------------------
    short_gap_limit: int = 3
    long_gap_limit: int = 12

    # -- Feature toggles -----------------------------------------------------
    enable_seasonal: bool = True
    enable_cross_series: bool = True

    # -- Worker pool ---------------------------------------------------------
    worker_count: int = 4
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Cache ---------------------------------------------------------------
    cache_ttl: int = 3600

    # -- Rate limiting -------------------------------------------------------
    rate_limit_rpm: int = 120

    # -- Provenance ----------------------------------------------------------
    enable_provenance: bool = True

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> TimeSeriesGapFillerConfig:
        """Build a TimeSeriesGapFillerConfig from environment variables.

        Every field can be overridden via ``GL_TSGF_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated TimeSeriesGapFillerConfig instance.
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
                    "Invalid float for %s%s=%s, using default %f",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        config = cls(
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Record processing
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            max_records=_int("MAX_RECORDS", cls.max_records),
            # Gap tolerance
            max_gap_ratio=_float(
                "MAX_GAP_RATIO", cls.max_gap_ratio,
            ),
            min_data_points=_int(
                "MIN_DATA_POINTS", cls.min_data_points,
            ),
            # Strategy selection
            default_strategy=_str(
                "DEFAULT_STRATEGY", cls.default_strategy,
            ),
            interpolation_method=_str(
                "INTERPOLATION_METHOD", cls.interpolation_method,
            ),
            # Seasonal decomposition
            seasonal_periods=_int(
                "SEASONAL_PERIODS", cls.seasonal_periods,
            ),
            # Holt-Winters smoothing parameters
            smoothing_alpha=_float(
                "SMOOTHING_ALPHA", cls.smoothing_alpha,
            ),
            smoothing_beta=_float(
                "SMOOTHING_BETA", cls.smoothing_beta,
            ),
            smoothing_gamma=_float(
                "SMOOTHING_GAMMA", cls.smoothing_gamma,
            ),
            # Quality thresholds
            correlation_threshold=_float(
                "CORRELATION_THRESHOLD", cls.correlation_threshold,
            ),
            confidence_threshold=_float(
                "CONFIDENCE_THRESHOLD", cls.confidence_threshold,
            ),
            # Gap classification
            short_gap_limit=_int(
                "SHORT_GAP_LIMIT", cls.short_gap_limit,
            ),
            long_gap_limit=_int(
                "LONG_GAP_LIMIT", cls.long_gap_limit,
            ),
            # Feature toggles
            enable_seasonal=_bool(
                "ENABLE_SEASONAL", cls.enable_seasonal,
            ),
            enable_cross_series=_bool(
                "ENABLE_CROSS_SERIES", cls.enable_cross_series,
            ),
            # Worker pool
            worker_count=_int(
                "WORKER_COUNT", cls.worker_count,
            ),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            # Cache
            cache_ttl=_int("CACHE_TTL", cls.cache_ttl),
            # Rate limiting
            rate_limit_rpm=_int(
                "RATE_LIMIT_RPM", cls.rate_limit_rpm,
            ),
            # Provenance
            enable_provenance=_bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
        )

        logger.info(
            "TimeSeriesGapFillerConfig loaded: batch=%d, max_records=%d, "
            "max_gap_ratio=%.2f, min_data_points=%d, "
            "strategy=%s, interpolation=%s, "
            "seasonal_periods=%d, "
            "alpha=%.2f, beta=%.2f, gamma=%.2f, "
            "correlation=%.2f, confidence=%.2f, "
            "seasonal=%s, cross_series=%s, "
            "workers=%d, pool=%d-%d, cache_ttl=%ds, "
            "rate=%d rpm, provenance=%s",
            config.batch_size,
            config.max_records,
            config.max_gap_ratio,
            config.min_data_points,
            config.default_strategy,
            config.interpolation_method,
            config.seasonal_periods,
            config.smoothing_alpha,
            config.smoothing_beta,
            config.smoothing_gamma,
            config.correlation_threshold,
            config.confidence_threshold,
            config.enable_seasonal,
            config.enable_cross_series,
            config.worker_count,
            config.pool_min_size,
            config.pool_max_size,
            config.cache_ttl,
            config.rate_limit_rpm,
            config.enable_provenance,
        )
        return config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate all configuration constraints.

        Raises:
            ValueError: If any constraint is violated.
        """
        errors: list[str] = []

        # Record processing
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.max_records < 1:
            errors.append("max_records must be >= 1")
        if self.batch_size > self.max_records:
            errors.append("batch_size must be <= max_records")

        # Gap tolerance
        if not 0.0 <= self.max_gap_ratio <= 1.0:
            errors.append("max_gap_ratio must be between 0.0 and 1.0")
        if self.min_data_points < 2:
            errors.append("min_data_points must be >= 2")

        # Strategy selection
        valid_strategies = (
            "auto", "linear", "cubic_spline", "pchip", "akima",
            "polynomial", "seasonal", "trend", "cross_series",
            "moving_average", "exponential_smoothing", "calendar_aware",
        )
        if self.default_strategy not in valid_strategies:
            errors.append(
                f"default_strategy must be one of {valid_strategies}, "
                f"got '{self.default_strategy}'"
            )

        valid_interpolations = (
            "linear", "cubic", "spline", "pchip", "akima",
        )
        if self.interpolation_method not in valid_interpolations:
            errors.append(
                f"interpolation_method must be one of "
                f"{valid_interpolations}, "
                f"got '{self.interpolation_method}'"
            )

        # Seasonal decomposition
        if self.seasonal_periods < 2:
            errors.append("seasonal_periods must be >= 2")

        # Smoothing parameters
        if not 0.0 <= self.smoothing_alpha <= 1.0:
            errors.append("smoothing_alpha must be between 0.0 and 1.0")
        if not 0.0 <= self.smoothing_beta <= 1.0:
            errors.append("smoothing_beta must be between 0.0 and 1.0")
        if not 0.0 <= self.smoothing_gamma <= 1.0:
            errors.append("smoothing_gamma must be between 0.0 and 1.0")

        # Quality thresholds
        if not 0.0 <= self.correlation_threshold <= 1.0:
            errors.append(
                "correlation_threshold must be between 0.0 and 1.0"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append(
                "confidence_threshold must be between 0.0 and 1.0"
            )

        # Worker pool
        if self.worker_count < 1:
            errors.append("worker_count must be >= 1")
        if self.pool_min_size < 1:
            errors.append("pool_min_size must be >= 1")
        if self.pool_max_size < self.pool_min_size:
            errors.append("pool_max_size must be >= pool_min_size")

        # Cache
        if self.cache_ttl < 0:
            errors.append("cache_ttl must be >= 0")

        # Rate limiting
        if self.rate_limit_rpm < 1:
            errors.append("rate_limit_rpm must be >= 1")

        # Log level
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if self.log_level.upper() not in valid_levels:
            errors.append(
                f"log_level must be one of {valid_levels}, "
                f"got '{self.log_level}'"
            )

        if errors:
            msg = "; ".join(errors)
            logger.error("TimeSeriesGapFillerConfig validation failed: %s", msg)
            raise ValueError(
                f"TimeSeriesGapFillerConfig validation failed: {msg}"
            )

        logger.info("TimeSeriesGapFillerConfig validated successfully")


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[TimeSeriesGapFillerConfig] = None
_config_lock = threading.Lock()


def get_config() -> TimeSeriesGapFillerConfig:
    """Return the singleton TimeSeriesGapFillerConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path.

    Returns:
        TimeSeriesGapFillerConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = TimeSeriesGapFillerConfig.from_env()
    return _config_instance


def set_config(config: TimeSeriesGapFillerConfig) -> None:
    """Replace the singleton TimeSeriesGapFillerConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("TimeSeriesGapFillerConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "TimeSeriesGapFillerConfig",
    "get_config",
    "set_config",
    "reset_config",
]
