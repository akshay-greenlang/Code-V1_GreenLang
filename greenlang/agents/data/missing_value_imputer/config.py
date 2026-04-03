# -*- coding: utf-8 -*-
"""
Missing Value Imputer Agent Service Configuration - AGENT-DATA-012

Centralized configuration for the Missing Value Imputer SDK covering:
- Database, cache, and object storage connection defaults
- Record processing limits (max records, batch size)
- Imputation strategy settings (default strategy, confidence threshold)
- Statistical imputation parameters (mean, median, mode)
- KNN imputation parameters (neighbors, max dataset size)
- MICE imputation parameters (iterations, multiple imputations)
- ML imputation toggles (random forest, gradient boosting)
- Time-series imputation settings (interpolation, seasonal, trend)
- Rule-based imputation settings (lookup tables, regulatory defaults)
- Validation and quality parameters (max missing pct, validation split)
- Worker, pool, cache, rate limiting, and provenance settings

All settings can be overridden via environment variables with the
``GL_MVI_`` prefix (e.g. ``GL_MVI_MAX_RECORDS``).

Example:
    >>> from greenlang.agents.data.missing_value_imputer.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_strategy, cfg.knn_neighbors)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
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

_ENV_PREFIX = "GL_MVI_"


# ---------------------------------------------------------------------------
# MissingValueImputerConfig
# ---------------------------------------------------------------------------


@dataclass
class MissingValueImputerConfig(BaseDataConfig):
    """Configuration for the GreenLang Missing Value Imputer SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only imputer-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_MVI_`` prefix.

    Attributes:
        batch_size: Default batch size for record processing.
        max_records: Maximum records allowed in a single imputation job.
        default_strategy: Default imputation strategy.
        confidence_threshold: Minimum confidence score for an imputed value.
        max_missing_pct: Maximum fraction of missing values before flagging.
        enable_statistical: Whether statistical imputation is enabled.
        knn_neighbors: Number of nearest neighbors for KNN imputation.
        max_knn_dataset_size: Maximum dataset size for KNN imputation.
        mice_iterations: Number of MICE algorithm iterations.
        multiple_imputations: Number of multiple imputations for uncertainty.
        enable_ml_imputation: Whether ML-based imputation is enabled.
        enable_timeseries: Whether time-series imputation is enabled.
        interpolation_method: Default interpolation method for time-series.
        seasonal_period: Seasonal period for time-series decomposition.
        trend_window: Window size for trend estimation.
        enable_rule_based: Whether rule-based imputation is enabled.
        validation_split: Fraction of non-missing data held out for validation.
        worker_count: Number of parallel workers for batch processing.
        cache_ttl: Cache time-to-live in seconds for strategy results.
        rate_limit_rpm: Rate limit in requests per minute.
        rate_limit_burst: Maximum burst size for rate limiting.
        enable_provenance: Whether SHA-256 provenance tracking is enabled.
        provenance_hash_algorithm: Hash algorithm for provenance tracking.
        enable_metrics: Whether Prometheus metrics collection is enabled.
        default_confidence_method: Default method for computing confidence scores.
    """

    # -- Record processing ---------------------------------------------------
    batch_size: int = 1000
    max_records: int = 100_000

    # -- Imputation strategy -------------------------------------------------
    default_strategy: str = "auto"
    confidence_threshold: float = 0.7
    max_missing_pct: float = 0.8

    # -- Statistical parameters ----------------------------------------------
    enable_statistical: bool = True

    # -- KNN parameters ------------------------------------------------------
    knn_neighbors: int = 5
    max_knn_dataset_size: int = 50_000

    # -- MICE parameters -----------------------------------------------------
    mice_iterations: int = 10
    multiple_imputations: int = 5

    # -- ML imputation -------------------------------------------------------
    enable_ml_imputation: bool = True

    # -- Time-series imputation ----------------------------------------------
    enable_timeseries: bool = True
    interpolation_method: str = "linear"
    seasonal_period: int = 12
    trend_window: int = 6

    # -- Rule-based imputation -----------------------------------------------
    enable_rule_based: bool = True

    # -- Validation ----------------------------------------------------------
    validation_split: float = 0.2

    # -- Worker pool ---------------------------------------------------------
    worker_count: int = 4

    # -- Cache ---------------------------------------------------------------
    cache_ttl: int = 3600

    # -- Rate limiting -------------------------------------------------------
    rate_limit_rpm: int = 120
    rate_limit_burst: int = 20

    # -- Provenance ----------------------------------------------------------
    enable_provenance: bool = True
    provenance_hash_algorithm: str = "sha256"

    # -- Metrics -------------------------------------------------------------
    enable_metrics: bool = True

    # -- Confidence ----------------------------------------------------------
    default_confidence_method: str = "ensemble"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> MissingValueImputerConfig:
        """Build a MissingValueImputerConfig from environment variables.

        Every field can be overridden via ``GL_MVI_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated MissingValueImputerConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            # Record processing
            batch_size=env.int("BATCH_SIZE", cls.batch_size),
            max_records=env.int("MAX_RECORDS", cls.max_records),
            # Imputation strategy
            default_strategy=env.str(
                "DEFAULT_STRATEGY", cls.default_strategy,
            ),
            confidence_threshold=env.float(
                "CONFIDENCE_THRESHOLD", cls.confidence_threshold,
            ),
            max_missing_pct=env.float(
                "MAX_MISSING_PCT", cls.max_missing_pct,
            ),
            # Statistical parameters
            enable_statistical=env.bool(
                "ENABLE_STATISTICAL", cls.enable_statistical,
            ),
            # KNN parameters
            knn_neighbors=env.int(
                "KNN_NEIGHBORS", cls.knn_neighbors,
            ),
            max_knn_dataset_size=env.int(
                "MAX_KNN_DATASET_SIZE", cls.max_knn_dataset_size,
            ),
            # MICE parameters
            mice_iterations=env.int(
                "MICE_ITERATIONS", cls.mice_iterations,
            ),
            multiple_imputations=env.int(
                "MULTIPLE_IMPUTATIONS", cls.multiple_imputations,
            ),
            # ML imputation
            enable_ml_imputation=env.bool(
                "ENABLE_ML_IMPUTATION", cls.enable_ml_imputation,
            ),
            # Time-series imputation
            enable_timeseries=env.bool(
                "ENABLE_TIMESERIES", cls.enable_timeseries,
            ),
            interpolation_method=env.str(
                "INTERPOLATION_METHOD", cls.interpolation_method,
            ),
            seasonal_period=env.int(
                "SEASONAL_PERIOD", cls.seasonal_period,
            ),
            trend_window=env.int(
                "TREND_WINDOW", cls.trend_window,
            ),
            # Rule-based imputation
            enable_rule_based=env.bool(
                "ENABLE_RULE_BASED", cls.enable_rule_based,
            ),
            # Validation
            validation_split=env.float(
                "VALIDATION_SPLIT", cls.validation_split,
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
            rate_limit_burst=env.int(
                "RATE_LIMIT_BURST", cls.rate_limit_burst,
            ),
            # Provenance
            enable_provenance=env.bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            provenance_hash_algorithm=env.str(
                "PROVENANCE_HASH_ALGORITHM",
                cls.provenance_hash_algorithm,
            ),
            # Metrics
            enable_metrics=env.bool(
                "ENABLE_METRICS", cls.enable_metrics,
            ),
            # Confidence
            default_confidence_method=env.str(
                "DEFAULT_CONFIDENCE_METHOD",
                cls.default_confidence_method,
            ),
        )

        logger.info(
            "MissingValueImputerConfig loaded: batch=%d, max_records=%d, "
            "strategy=%s, confidence=%.2f, max_missing=%.2f, "
            "statistical=%s, knn_k=%d (max_ds=%d), "
            "mice_iter=%d (m=%d), ml=%s, timeseries=%s "
            "(interp=%s, season=%d, trend=%d), rule_based=%s, "
            "val_split=%.2f, workers=%d, pool=%d-%d, "
            "cache_ttl=%ds, rate=%d rpm (burst=%d), "
            "provenance=%s (algo=%s), metrics=%s, "
            "confidence_method=%s",
            config.batch_size,
            config.max_records,
            config.default_strategy,
            config.confidence_threshold,
            config.max_missing_pct,
            config.enable_statistical,
            config.knn_neighbors,
            config.max_knn_dataset_size,
            config.mice_iterations,
            config.multiple_imputations,
            config.enable_ml_imputation,
            config.enable_timeseries,
            config.interpolation_method,
            config.seasonal_period,
            config.trend_window,
            config.enable_rule_based,
            config.validation_split,
            config.worker_count,
            config.pool_min_size,
            config.pool_max_size,
            config.cache_ttl,
            config.rate_limit_rpm,
            config.rate_limit_burst,
            config.enable_provenance,
            config.provenance_hash_algorithm,
            config.enable_metrics,
            config.default_confidence_method,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

get_config, set_config, reset_config = create_config_singleton(
    MissingValueImputerConfig, _ENV_PREFIX,
)

__all__ = [
    "MissingValueImputerConfig",
    "get_config",
    "set_config",
    "reset_config",
]
