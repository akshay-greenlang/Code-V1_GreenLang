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
    >>> from greenlang.agents.data.time_series_gap_filler.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_strategy, cfg.interpolation_method)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
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

_ENV_PREFIX = "GL_TSGF_"


# ---------------------------------------------------------------------------
# TimeSeriesGapFillerConfig
# ---------------------------------------------------------------------------


@dataclass
class TimeSeriesGapFillerConfig(BaseDataConfig):
    """Configuration for the GreenLang Time Series Gap Filler SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only gap-filler-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_TSGF_`` prefix.

    Attributes:
        batch_size: Default batch size for record processing.
        max_records: Maximum records allowed in a single gap filling job.
        max_gap_ratio: Maximum allowable fraction of a series that may
            consist of gaps (0.0 to 1.0).
        min_data_points: Minimum number of non-null data points required
            before gap filling can be attempted.
        default_strategy: Default strategy selection mode for gap filling.
        interpolation_method: Default interpolation method when the
            strategy resolves to numeric interpolation.
        seasonal_periods: Number of observations per seasonal cycle.
        smoothing_alpha: Holt-Winters level coefficient (alpha).
        smoothing_beta: Holt-Winters trend coefficient (beta).
        smoothing_gamma: Holt-Winters seasonal coefficient (gamma).
        correlation_threshold: Minimum Pearson correlation coefficient
            for cross-series gap filling.
        confidence_threshold: Minimum confidence score for a gap fill
            to be accepted.
        short_gap_limit: Maximum gap length classified as short.
        long_gap_limit: Maximum gap length classified as long.
        enable_seasonal: Whether seasonal decomposition is enabled.
        enable_cross_series: Whether cross-series correlation is enabled.
        worker_count: Number of parallel workers for batch processing.
        cache_ttl: Cache time-to-live in seconds.
        rate_limit_rpm: Rate limit in requests per minute.
        enable_provenance: Whether SHA-256 provenance tracking is enabled.
    """

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

    # -- Cache ---------------------------------------------------------------
    cache_ttl: int = 3600

    # -- Rate limiting -------------------------------------------------------
    rate_limit_rpm: int = 120

    # -- Provenance ----------------------------------------------------------
    enable_provenance: bool = True

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> TimeSeriesGapFillerConfig:
        """Build a TimeSeriesGapFillerConfig from environment variables.

        Every field can be overridden via ``GL_TSGF_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated TimeSeriesGapFillerConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            # Record processing
            batch_size=env.int("BATCH_SIZE", cls.batch_size),
            max_records=env.int("MAX_RECORDS", cls.max_records),
            # Gap tolerance
            max_gap_ratio=env.float(
                "MAX_GAP_RATIO", cls.max_gap_ratio,
            ),
            min_data_points=env.int(
                "MIN_DATA_POINTS", cls.min_data_points,
            ),
            # Strategy selection
            default_strategy=env.str(
                "DEFAULT_STRATEGY", cls.default_strategy,
            ),
            interpolation_method=env.str(
                "INTERPOLATION_METHOD", cls.interpolation_method,
            ),
            # Seasonal decomposition
            seasonal_periods=env.int(
                "SEASONAL_PERIODS", cls.seasonal_periods,
            ),
            # Holt-Winters smoothing parameters
            smoothing_alpha=env.float(
                "SMOOTHING_ALPHA", cls.smoothing_alpha,
            ),
            smoothing_beta=env.float(
                "SMOOTHING_BETA", cls.smoothing_beta,
            ),
            smoothing_gamma=env.float(
                "SMOOTHING_GAMMA", cls.smoothing_gamma,
            ),
            # Quality thresholds
            correlation_threshold=env.float(
                "CORRELATION_THRESHOLD", cls.correlation_threshold,
            ),
            confidence_threshold=env.float(
                "CONFIDENCE_THRESHOLD", cls.confidence_threshold,
            ),
            # Gap classification
            short_gap_limit=env.int(
                "SHORT_GAP_LIMIT", cls.short_gap_limit,
            ),
            long_gap_limit=env.int(
                "LONG_GAP_LIMIT", cls.long_gap_limit,
            ),
            # Feature toggles
            enable_seasonal=env.bool(
                "ENABLE_SEASONAL", cls.enable_seasonal,
            ),
            enable_cross_series=env.bool(
                "ENABLE_CROSS_SERIES", cls.enable_cross_series,
            ),
            # Worker pool
            worker_count=env.int(
                "WORKER_COUNT", cls.worker_count,
            ),
            # Cache
            cache_ttl=env.int("CACHE_TTL", cls.cache_ttl),
            # Rate limiting
            rate_limit_rpm=env.int(
                "RATE_LIMIT_RPM", cls.rate_limit_rpm,
            ),
            # Provenance
            enable_provenance=env.bool(
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

get_config, set_config, reset_config = create_config_singleton(
    TimeSeriesGapFillerConfig, _ENV_PREFIX,
)

__all__ = [
    "TimeSeriesGapFillerConfig",
    "get_config",
    "set_config",
    "reset_config",
]
