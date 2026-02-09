# -*- coding: utf-8 -*-
"""
Data Quality Profiler Service Configuration - AGENT-DATA-010

Centralized configuration for the Data Quality Profiler SDK covering:
- Database, cache, and object storage connection defaults
- Profiling settings (row/column limits, sampling, schema inference)
- Quality dimension weights (completeness, validity, consistency,
  timeliness, uniqueness, accuracy)
- Freshness/timeliness SLA thresholds
- Anomaly detection parameters (IQR, z-score, MAD)
- Rule engine limits (rules per dataset, gate conditions)
- Processing limits (batch size, timeout, cache TTL)
- Connection pool sizing and logging

All settings can be overridden via environment variables with the
``GL_DQ_`` prefix (e.g. ``GL_DQ_MAX_ROWS_PER_PROFILE``).

Example:
    >>> from greenlang.data_quality_profiler.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.max_rows_per_profile, cfg.completeness_weight)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-010 Data Quality Profiler (GL-DATA-X-013)
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

_ENV_PREFIX = "GL_DQ_"


# ---------------------------------------------------------------------------
# DataQualityProfilerConfig
# ---------------------------------------------------------------------------


@dataclass
class DataQualityProfilerConfig:
    """Complete configuration for the GreenLang Data Quality Profiler SDK.

    Attributes are grouped by concern: connections, profiling settings,
    quality dimension weights, freshness/timeliness thresholds, anomaly
    detection parameters, rule engine limits, processing limits,
    connection pool sizing, and logging.

    All attributes can be overridden via environment variables using the
    ``GL_DQ_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        s3_bucket_url: S3 bucket URL for report and artifact storage.
        max_rows_per_profile: Maximum rows to include in a single profile run.
        max_columns_per_profile: Maximum columns to include in a single profile.
        sample_size_for_stats: Number of rows sampled for statistical analysis.
        enable_schema_inference: Whether to infer column data types automatically.
        enable_cardinality_analysis: Whether to compute cardinality per column.
        max_unique_values_tracked: Maximum distinct values tracked per column.
        completeness_weight: Weight for completeness dimension in overall score.
        validity_weight: Weight for validity dimension in overall score.
        consistency_weight: Weight for consistency dimension in overall score.
        timeliness_weight: Weight for timeliness dimension in overall score.
        uniqueness_weight: Weight for uniqueness dimension in overall score.
        accuracy_weight: Weight for accuracy dimension in overall score.
        freshness_excellent_hours: Max age (hours) for EXCELLENT freshness.
        freshness_good_hours: Max age (hours) for GOOD freshness.
        freshness_fair_hours: Max age (hours) for FAIR freshness.
        freshness_poor_hours: Max age (hours) for POOR freshness.
        default_sla_hours: Default SLA deadline in hours for freshness checks.
        default_outlier_method: Default outlier detection method.
        iqr_multiplier: IQR fence multiplier for outlier detection.
        zscore_threshold: Z-score threshold for outlier detection.
        mad_threshold: MAD (Median Absolute Deviation) threshold for outliers.
        min_samples_for_anomaly: Minimum sample size required for anomaly detection.
        max_rules_per_dataset: Maximum quality rules allowed per dataset.
        max_gate_conditions: Maximum conditions per quality gate definition.
        default_gate_threshold: Default pass/fail threshold for quality gates.
        batch_max_datasets: Maximum datasets per batch profiling run.
        processing_timeout_seconds: Timeout in seconds for a single profile run.
        cache_ttl_seconds: Default cache time-to-live in seconds.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        log_level: Logging level for the data quality profiler service.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""

    # -- Profiling settings --------------------------------------------------
    max_rows_per_profile: int = 1_000_000
    max_columns_per_profile: int = 500
    sample_size_for_stats: int = 10_000
    enable_schema_inference: bool = True
    enable_cardinality_analysis: bool = True
    max_unique_values_tracked: int = 1000

    # -- Quality dimension weights -------------------------------------------
    completeness_weight: float = 0.20
    validity_weight: float = 0.20
    consistency_weight: float = 0.20
    timeliness_weight: float = 0.15
    uniqueness_weight: float = 0.15
    accuracy_weight: float = 0.10

    # -- Freshness / timeliness thresholds -----------------------------------
    freshness_excellent_hours: int = 24
    freshness_good_hours: int = 72
    freshness_fair_hours: int = 168
    freshness_poor_hours: int = 720
    default_sla_hours: int = 48

    # -- Anomaly detection ---------------------------------------------------
    default_outlier_method: str = "iqr"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    mad_threshold: float = 3.5
    min_samples_for_anomaly: int = 10

    # -- Rule engine ---------------------------------------------------------
    max_rules_per_dataset: int = 100
    max_gate_conditions: int = 20
    default_gate_threshold: float = 0.70

    # -- Processing ----------------------------------------------------------
    batch_max_datasets: int = 50
    processing_timeout_seconds: int = 300
    cache_ttl_seconds: int = 3600

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DataQualityProfilerConfig:
        """Build a DataQualityProfilerConfig from environment variables.

        Every field can be overridden via ``GL_DQ_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated DataQualityProfilerConfig instance.
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
            # Profiling settings
            max_rows_per_profile=_int(
                "MAX_ROWS_PER_PROFILE", cls.max_rows_per_profile,
            ),
            max_columns_per_profile=_int(
                "MAX_COLUMNS_PER_PROFILE", cls.max_columns_per_profile,
            ),
            sample_size_for_stats=_int(
                "SAMPLE_SIZE_FOR_STATS", cls.sample_size_for_stats,
            ),
            enable_schema_inference=_bool(
                "ENABLE_SCHEMA_INFERENCE", cls.enable_schema_inference,
            ),
            enable_cardinality_analysis=_bool(
                "ENABLE_CARDINALITY_ANALYSIS",
                cls.enable_cardinality_analysis,
            ),
            max_unique_values_tracked=_int(
                "MAX_UNIQUE_VALUES_TRACKED",
                cls.max_unique_values_tracked,
            ),
            # Quality dimension weights
            completeness_weight=_float(
                "COMPLETENESS_WEIGHT", cls.completeness_weight,
            ),
            validity_weight=_float(
                "VALIDITY_WEIGHT", cls.validity_weight,
            ),
            consistency_weight=_float(
                "CONSISTENCY_WEIGHT", cls.consistency_weight,
            ),
            timeliness_weight=_float(
                "TIMELINESS_WEIGHT", cls.timeliness_weight,
            ),
            uniqueness_weight=_float(
                "UNIQUENESS_WEIGHT", cls.uniqueness_weight,
            ),
            accuracy_weight=_float(
                "ACCURACY_WEIGHT", cls.accuracy_weight,
            ),
            # Freshness / timeliness thresholds
            freshness_excellent_hours=_int(
                "FRESHNESS_EXCELLENT_HOURS",
                cls.freshness_excellent_hours,
            ),
            freshness_good_hours=_int(
                "FRESHNESS_GOOD_HOURS", cls.freshness_good_hours,
            ),
            freshness_fair_hours=_int(
                "FRESHNESS_FAIR_HOURS", cls.freshness_fair_hours,
            ),
            freshness_poor_hours=_int(
                "FRESHNESS_POOR_HOURS", cls.freshness_poor_hours,
            ),
            default_sla_hours=_int(
                "DEFAULT_SLA_HOURS", cls.default_sla_hours,
            ),
            # Anomaly detection
            default_outlier_method=_str(
                "DEFAULT_OUTLIER_METHOD", cls.default_outlier_method,
            ),
            iqr_multiplier=_float(
                "IQR_MULTIPLIER", cls.iqr_multiplier,
            ),
            zscore_threshold=_float(
                "ZSCORE_THRESHOLD", cls.zscore_threshold,
            ),
            mad_threshold=_float(
                "MAD_THRESHOLD", cls.mad_threshold,
            ),
            min_samples_for_anomaly=_int(
                "MIN_SAMPLES_FOR_ANOMALY", cls.min_samples_for_anomaly,
            ),
            # Rule engine
            max_rules_per_dataset=_int(
                "MAX_RULES_PER_DATASET", cls.max_rules_per_dataset,
            ),
            max_gate_conditions=_int(
                "MAX_GATE_CONDITIONS", cls.max_gate_conditions,
            ),
            default_gate_threshold=_float(
                "DEFAULT_GATE_THRESHOLD", cls.default_gate_threshold,
            ),
            # Processing
            batch_max_datasets=_int(
                "BATCH_MAX_DATASETS", cls.batch_max_datasets,
            ),
            processing_timeout_seconds=_int(
                "PROCESSING_TIMEOUT_SECONDS",
                cls.processing_timeout_seconds,
            ),
            cache_ttl_seconds=_int(
                "CACHE_TTL_SECONDS", cls.cache_ttl_seconds,
            ),
            # Pool sizing
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
        )

        logger.info(
            "DataQualityProfilerConfig loaded: max_rows=%d, max_cols=%d, "
            "sample=%d, schema_inference=%s, cardinality=%s, "
            "weights=[C=%.2f V=%.2f Co=%.2f T=%.2f U=%.2f A=%.2f], "
            "outlier=%s, iqr=%.1f, zscore=%.1f, mad=%.1f, "
            "max_rules=%d, gate_threshold=%.2f, batch=%d, "
            "timeout=%ds, cache_ttl=%ds, pool=%d-%d",
            config.max_rows_per_profile,
            config.max_columns_per_profile,
            config.sample_size_for_stats,
            config.enable_schema_inference,
            config.enable_cardinality_analysis,
            config.completeness_weight,
            config.validity_weight,
            config.consistency_weight,
            config.timeliness_weight,
            config.uniqueness_weight,
            config.accuracy_weight,
            config.default_outlier_method,
            config.iqr_multiplier,
            config.zscore_threshold,
            config.mad_threshold,
            config.max_rules_per_dataset,
            config.default_gate_threshold,
            config.batch_max_datasets,
            config.processing_timeout_seconds,
            config.cache_ttl_seconds,
            config.pool_min_size,
            config.pool_max_size,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[DataQualityProfilerConfig] = None
_config_lock = threading.Lock()


def get_config() -> DataQualityProfilerConfig:
    """Return the singleton DataQualityProfilerConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path.

    Returns:
        DataQualityProfilerConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = DataQualityProfilerConfig.from_env()
    return _config_instance


def set_config(config: DataQualityProfilerConfig) -> None:
    """Replace the singleton DataQualityProfilerConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("DataQualityProfilerConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "DataQualityProfilerConfig",
    "get_config",
    "set_config",
    "reset_config",
]
