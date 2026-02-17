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
    >>> from greenlang.missing_value_imputer.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_strategy, cfg.knn_neighbors)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
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

_ENV_PREFIX = "GL_MVI_"


# ---------------------------------------------------------------------------
# MissingValueImputerConfig
# ---------------------------------------------------------------------------


@dataclass
class MissingValueImputerConfig:
    """Complete configuration for the GreenLang Missing Value Imputer SDK.

    Attributes are grouped by concern: connections, record processing,
    imputation strategy, statistical parameters, KNN parameters, MICE
    parameters, ML imputation, time-series imputation, rule-based
    imputation, validation, worker pool, cache, rate limiting,
    provenance, and logging.

    All attributes can be overridden via environment variables using the
    ``GL_MVI_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        s3_bucket_url: S3 bucket URL for artifact and report storage.
        log_level: Logging level for the missing value imputer service.
        batch_size: Default batch size for record processing.
        max_records: Maximum records allowed in a single imputation job.
        default_strategy: Default imputation strategy when auto-selecting
            (auto, mean, median, mode, knn, regression, mice,
            random_forest, gradient_boosting, linear_interpolation,
            spline_interpolation, seasonal_decomposition, rule_based,
            lookup_table, regulatory_default).
        knn_neighbors: Number of nearest neighbors for KNN imputation.
        mice_iterations: Number of MICE algorithm iterations.
        confidence_threshold: Minimum confidence score (0.0-1.0) for
            an imputed value to be accepted.
        max_missing_pct: Maximum fraction of missing values in a column
            before it is flagged as too sparse for reliable imputation.
        enable_ml_imputation: Whether ML-based imputation strategies
            (random forest, gradient boosting) are enabled.
        enable_timeseries: Whether time-series imputation strategies
            (interpolation, seasonal decomposition) are enabled.
        multiple_imputations: Number of multiple imputations to generate
            for uncertainty estimation (MICE, Bayesian methods).
        validation_split: Fraction of non-missing data held out for
            imputation validation (0.0-1.0).
        worker_count: Number of parallel workers for batch processing.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        cache_ttl: Cache time-to-live in seconds for strategy results.
        rate_limit_rpm: Rate limit in requests per minute.
        rate_limit_burst: Maximum burst size for rate limiting.
        enable_provenance: Whether SHA-256 provenance tracking is enabled.
        provenance_hash_algorithm: Hash algorithm for provenance tracking
            (sha256, sha384, sha512).
        enable_rule_based: Whether rule-based imputation is enabled.
        enable_statistical: Whether statistical imputation strategies
            (mean, median, mode) are enabled.
        max_knn_dataset_size: Maximum dataset size for KNN imputation
            before switching to approximate nearest neighbors.
        interpolation_method: Default interpolation method for time-series
            imputation (linear, quadratic, cubic, spline, polynomial).
        seasonal_period: Seasonal period for time-series decomposition
            (e.g. 12 for monthly data with annual seasonality).
        trend_window: Window size for trend estimation in time-series
            imputation (number of periods).
        default_confidence_method: Default method for computing imputation
            confidence scores (ensemble, bootstrap, cross_validation,
            distance_weighted, rule_priority).
        enable_metrics: Whether Prometheus metrics collection is enabled.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

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
    pool_min_size: int = 2
    pool_max_size: int = 10

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
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> MissingValueImputerConfig:
        """Build a MissingValueImputerConfig from environment variables.

        Every field can be overridden via ``GL_MVI_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated MissingValueImputerConfig instance.
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
            s3_bucket_url=_str("S3_BUCKET_URL", cls.s3_bucket_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Record processing
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            max_records=_int("MAX_RECORDS", cls.max_records),
            # Imputation strategy
            default_strategy=_str(
                "DEFAULT_STRATEGY", cls.default_strategy,
            ),
            confidence_threshold=_float(
                "CONFIDENCE_THRESHOLD", cls.confidence_threshold,
            ),
            max_missing_pct=_float(
                "MAX_MISSING_PCT", cls.max_missing_pct,
            ),
            # Statistical parameters
            enable_statistical=_bool(
                "ENABLE_STATISTICAL", cls.enable_statistical,
            ),
            # KNN parameters
            knn_neighbors=_int(
                "KNN_NEIGHBORS", cls.knn_neighbors,
            ),
            max_knn_dataset_size=_int(
                "MAX_KNN_DATASET_SIZE", cls.max_knn_dataset_size,
            ),
            # MICE parameters
            mice_iterations=_int(
                "MICE_ITERATIONS", cls.mice_iterations,
            ),
            multiple_imputations=_int(
                "MULTIPLE_IMPUTATIONS", cls.multiple_imputations,
            ),
            # ML imputation
            enable_ml_imputation=_bool(
                "ENABLE_ML_IMPUTATION", cls.enable_ml_imputation,
            ),
            # Time-series imputation
            enable_timeseries=_bool(
                "ENABLE_TIMESERIES", cls.enable_timeseries,
            ),
            interpolation_method=_str(
                "INTERPOLATION_METHOD", cls.interpolation_method,
            ),
            seasonal_period=_int(
                "SEASONAL_PERIOD", cls.seasonal_period,
            ),
            trend_window=_int(
                "TREND_WINDOW", cls.trend_window,
            ),
            # Rule-based imputation
            enable_rule_based=_bool(
                "ENABLE_RULE_BASED", cls.enable_rule_based,
            ),
            # Validation
            validation_split=_float(
                "VALIDATION_SPLIT", cls.validation_split,
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
            rate_limit_burst=_int(
                "RATE_LIMIT_BURST", cls.rate_limit_burst,
            ),
            # Provenance
            enable_provenance=_bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            provenance_hash_algorithm=_str(
                "PROVENANCE_HASH_ALGORITHM",
                cls.provenance_hash_algorithm,
            ),
            # Metrics
            enable_metrics=_bool(
                "ENABLE_METRICS", cls.enable_metrics,
            ),
            # Confidence
            default_confidence_method=_str(
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

_config_instance: Optional[MissingValueImputerConfig] = None
_config_lock = threading.Lock()


def get_config() -> MissingValueImputerConfig:
    """Return the singleton MissingValueImputerConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path.

    Returns:
        MissingValueImputerConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = MissingValueImputerConfig.from_env()
    return _config_instance


def set_config(config: MissingValueImputerConfig) -> None:
    """Replace the singleton MissingValueImputerConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("MissingValueImputerConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "MissingValueImputerConfig",
    "get_config",
    "set_config",
    "reset_config",
]
