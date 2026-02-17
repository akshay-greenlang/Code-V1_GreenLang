# -*- coding: utf-8 -*-
"""
Outlier Detection Agent Service Configuration - AGENT-DATA-013

Centralized configuration for the Outlier Detection SDK covering:
- Database, cache, and object storage connection defaults
- Record processing limits (max records, batch size)
- Statistical detection parameters (IQR multiplier, z-score threshold)
- MAD and Grubbs test parameters
- Local Outlier Factor and Isolation Forest parameters
- Ensemble detection settings (method, minimum consensus)
- Contextual and temporal detection toggles
- Multivariate detection toggle
- Treatment strategy defaults (default treatment, winsorize pct)
- Worker, pool, cache, rate limiting, and provenance settings

All settings can be overridden via environment variables with the
``GL_OD_`` prefix (e.g. ``GL_OD_MAX_RECORDS``).

Example:
    >>> from greenlang.outlier_detector.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.iqr_multiplier, cfg.zscore_threshold)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
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

_ENV_PREFIX = "GL_OD_"


# ---------------------------------------------------------------------------
# OutlierDetectorConfig
# ---------------------------------------------------------------------------


@dataclass
class OutlierDetectorConfig:
    """Complete configuration for the GreenLang Outlier Detector SDK.

    Attributes are grouped by concern: connections, record processing,
    statistical detection, MAD/Grubbs, LOF/Isolation Forest, ensemble,
    contextual/temporal/multivariate, treatment, worker pool, cache,
    rate limiting, provenance, and metrics.

    All attributes can be overridden via environment variables using the
    ``GL_OD_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        s3_bucket_url: S3 bucket URL for artifact and report storage.
        log_level: Logging level for the outlier detector service.
        batch_size: Default batch size for record processing.
        max_records: Maximum records allowed in a single detection job.
        iqr_multiplier: Multiplier for IQR fence calculation
            (default 1.5, use 3.0 for far outliers).
        zscore_threshold: Absolute z-score threshold for flagging
            outliers (default 3.0).
        mad_threshold: Modified z-score threshold using Median
            Absolute Deviation (default 3.5).
        grubbs_alpha: Significance level for Grubbs test (default 0.05).
        lof_neighbors: Number of neighbors for Local Outlier Factor
            (default 20).
        isolation_trees: Number of trees in Isolation Forest
            (default 100).
        ensemble_method: Method for combining multi-detector scores
            (weighted_average, majority_vote, max_score, mean_score).
        min_consensus: Minimum number of detectors that must flag a
            point before it is considered an outlier (default 2).
        enable_contextual: Whether contextual (group-based) detection
            is enabled (default True).
        enable_temporal: Whether temporal (time-series) detection
            is enabled (default True).
        enable_multivariate: Whether multivariate detection
            (Mahalanobis, Isolation Forest, LOF) is enabled.
        default_treatment: Default treatment strategy for detected
            outliers (flag, cap, winsorize, remove, replace, investigate).
        winsorize_pct: Percentile for winsorization treatment
            (fraction 0.0-0.5, default 0.05).
        worker_count: Number of parallel workers for batch processing.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        cache_ttl: Cache time-to-live in seconds for detection results.
        rate_limit_rpm: Rate limit in requests per minute.
        rate_limit_burst: Maximum burst size for rate limiting.
        enable_provenance: Whether SHA-256 provenance tracking is enabled.
        provenance_hash_algorithm: Hash algorithm for provenance tracking
            (sha256, sha384, sha512).
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

    # -- Statistical detection -----------------------------------------------
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    mad_threshold: float = 3.5
    grubbs_alpha: float = 0.05

    # -- LOF / Isolation Forest ----------------------------------------------
    lof_neighbors: int = 20
    isolation_trees: int = 100

    # -- Ensemble detection --------------------------------------------------
    ensemble_method: str = "weighted_average"
    min_consensus: int = 2

    # -- Contextual / Temporal / Multivariate --------------------------------
    enable_contextual: bool = True
    enable_temporal: bool = True
    enable_multivariate: bool = True

    # -- Treatment -----------------------------------------------------------
    default_treatment: str = "flag"
    winsorize_pct: float = 0.05

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

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> OutlierDetectorConfig:
        """Build an OutlierDetectorConfig from environment variables.

        Every field can be overridden via ``GL_OD_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated OutlierDetectorConfig instance.
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
            # Statistical detection
            iqr_multiplier=_float(
                "IQR_MULTIPLIER", cls.iqr_multiplier,
            ),
            zscore_threshold=_float(
                "ZSCORE_THRESHOLD", cls.zscore_threshold,
            ),
            mad_threshold=_float(
                "MAD_THRESHOLD", cls.mad_threshold,
            ),
            grubbs_alpha=_float(
                "GRUBBS_ALPHA", cls.grubbs_alpha,
            ),
            # LOF / Isolation Forest
            lof_neighbors=_int(
                "LOF_NEIGHBORS", cls.lof_neighbors,
            ),
            isolation_trees=_int(
                "ISOLATION_TREES", cls.isolation_trees,
            ),
            # Ensemble detection
            ensemble_method=_str(
                "ENSEMBLE_METHOD", cls.ensemble_method,
            ),
            min_consensus=_int(
                "MIN_CONSENSUS", cls.min_consensus,
            ),
            # Contextual / Temporal / Multivariate
            enable_contextual=_bool(
                "ENABLE_CONTEXTUAL", cls.enable_contextual,
            ),
            enable_temporal=_bool(
                "ENABLE_TEMPORAL", cls.enable_temporal,
            ),
            enable_multivariate=_bool(
                "ENABLE_MULTIVARIATE", cls.enable_multivariate,
            ),
            # Treatment
            default_treatment=_str(
                "DEFAULT_TREATMENT", cls.default_treatment,
            ),
            winsorize_pct=_float(
                "WINSORIZE_PCT", cls.winsorize_pct,
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
        )

        logger.info(
            "OutlierDetectorConfig loaded: batch=%d, max_records=%d, "
            "iqr_mult=%.2f, zscore=%.2f, mad=%.2f, grubbs_alpha=%.3f, "
            "lof_k=%d, iso_trees=%d, ensemble=%s (min_consensus=%d), "
            "contextual=%s, temporal=%s, multivariate=%s, "
            "treatment=%s, winsorize=%.3f, "
            "workers=%d, pool=%d-%d, cache_ttl=%ds, "
            "rate=%d rpm (burst=%d), provenance=%s (algo=%s), "
            "metrics=%s",
            config.batch_size,
            config.max_records,
            config.iqr_multiplier,
            config.zscore_threshold,
            config.mad_threshold,
            config.grubbs_alpha,
            config.lof_neighbors,
            config.isolation_trees,
            config.ensemble_method,
            config.min_consensus,
            config.enable_contextual,
            config.enable_temporal,
            config.enable_multivariate,
            config.default_treatment,
            config.winsorize_pct,
            config.worker_count,
            config.pool_min_size,
            config.pool_max_size,
            config.cache_ttl,
            config.rate_limit_rpm,
            config.rate_limit_burst,
            config.enable_provenance,
            config.provenance_hash_algorithm,
            config.enable_metrics,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[OutlierDetectorConfig] = None
_config_lock = threading.Lock()


def get_config() -> OutlierDetectorConfig:
    """Return the singleton OutlierDetectorConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path.

    Returns:
        OutlierDetectorConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = OutlierDetectorConfig.from_env()
    return _config_instance


def set_config(config: OutlierDetectorConfig) -> None:
    """Replace the singleton OutlierDetectorConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("OutlierDetectorConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "OutlierDetectorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
