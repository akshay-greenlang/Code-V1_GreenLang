# -*- coding: utf-8 -*-
"""
Duplicate Detection Agent Service Configuration - AGENT-DATA-011

Centralized configuration for the Duplicate Detection SDK covering:
- Database, cache, and object storage connection defaults
- Record processing limits (max records, batch size)
- Fingerprinting settings (algorithm, normalization)
- Blocking strategy parameters (sorted neighborhood, canopy)
- Similarity comparison settings (algorithm, ngram size)
- Match classification thresholds (match, possible, non-match)
- Fellegi-Sunter probabilistic model toggle
- Cluster algorithm and quality settings
- Merge strategy and conflict resolution defaults
- Pipeline processing parameters (checkpoint, timeout, comparisons)
- Cache, connection pool, logging, and metrics settings

All settings can be overridden via environment variables with the
``GL_DD_`` prefix (e.g. ``GL_DD_MAX_RECORDS_PER_JOB``).

Example:
    >>> from greenlang.duplicate_detector.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.match_threshold, cfg.blocking_strategy)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
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

_ENV_PREFIX = "GL_DD_"


# ---------------------------------------------------------------------------
# DuplicateDetectorConfig
# ---------------------------------------------------------------------------


@dataclass
class DuplicateDetectorConfig:
    """Complete configuration for the GreenLang Duplicate Detection SDK.

    Attributes are grouped by concern: connections, record processing,
    fingerprinting, blocking strategy, similarity comparison, match
    classification, probabilistic model, clustering, merge strategy,
    pipeline processing, cache, connection pool sizing, logging,
    and metrics.

    All attributes can be overridden via environment variables using the
    ``GL_DD_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        s3_bucket: S3 bucket name for artifact and report storage.
        max_records_per_job: Maximum records allowed in a single dedup job.
        default_batch_size: Default batch size for record processing.
        fingerprint_algorithm: Algorithm for record fingerprinting
            (sha256, simhash, minhash).
        fingerprint_normalize: Whether to normalize fields before
            fingerprinting (lowercasing, whitespace stripping).
        blocking_strategy: Strategy for candidate pair generation
            (sorted_neighborhood, standard, canopy, none).
        blocking_window_size: Window size for sorted neighborhood blocking.
        blocking_key_size: Number of leading characters for blocking keys.
        canopy_tight_threshold: Tight distance threshold for canopy
            clustering (records within this distance share a canopy).
        canopy_loose_threshold: Loose distance threshold for canopy
            clustering (records within this distance may share a canopy).
        default_similarity_algorithm: Default algorithm for field-level
            similarity comparison (exact, levenshtein, jaro_winkler,
            soundex, ngram, tfidf_cosine, numeric, date).
        ngram_size: Character n-gram size for ngram similarity.
        match_threshold: Overall score threshold at or above which a
            record pair is classified as MATCH (0.0 to 1.0).
        possible_threshold: Overall score threshold at or above which a
            record pair is classified as POSSIBLE (0.0 to 1.0).
        non_match_threshold: Overall score threshold at or below which a
            record pair is classified as NON_MATCH (0.0 to 1.0).
        use_fellegi_sunter: Whether to use the Fellegi-Sunter
            probabilistic record linkage model for classification.
        cluster_algorithm: Algorithm for grouping matched pairs into
            duplicate clusters (union_find, connected_components).
        cluster_min_quality: Minimum quality score for a cluster to be
            accepted as a valid duplicate group (0.0 to 1.0).
        default_merge_strategy: Default strategy for merging duplicate
            records within a cluster (keep_first, keep_latest,
            keep_most_complete, merge_fields, golden_record, custom).
        merge_conflict_resolution: Default field-level conflict
            resolution method when merging records (first, latest,
            most_complete, longest, shortest).
        pipeline_checkpoint_interval: Number of records processed
            between pipeline checkpoint saves.
        pipeline_timeout_seconds: Maximum wall-clock time in seconds
            for a single dedup pipeline run.
        max_comparisons_per_block: Maximum pairwise comparisons allowed
            within a single block before truncation.
        cache_ttl_seconds: Default cache time-to-live in seconds.
        cache_enabled: Whether caching is enabled.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        log_level: Logging level for the duplicate detection service.
        enable_metrics: Whether Prometheus metrics collection is enabled.
        max_field_weights: Maximum number of field weight entries per rule.
        max_rules_per_job: Maximum dedup rules allowed per job.
        comparison_sample_rate: Fraction of comparisons to execute
            (1.0 = all, 0.1 = 10 percent sample).
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""
    s3_bucket: str = ""

    # -- Record processing ---------------------------------------------------
    max_records_per_job: int = 1_000_000
    default_batch_size: int = 10_000

    # -- Fingerprinting ------------------------------------------------------
    fingerprint_algorithm: str = "sha256"
    fingerprint_normalize: bool = True

    # -- Blocking strategy ---------------------------------------------------
    blocking_strategy: str = "sorted_neighborhood"
    blocking_window_size: int = 10
    blocking_key_size: int = 3
    canopy_tight_threshold: float = 0.8
    canopy_loose_threshold: float = 0.4

    # -- Similarity comparison -----------------------------------------------
    default_similarity_algorithm: str = "jaro_winkler"
    ngram_size: int = 3

    # -- Match classification ------------------------------------------------
    match_threshold: float = 0.85
    possible_threshold: float = 0.65
    non_match_threshold: float = 0.40
    use_fellegi_sunter: bool = False

    # -- Clustering ----------------------------------------------------------
    cluster_algorithm: str = "union_find"
    cluster_min_quality: float = 0.5

    # -- Merge strategy ------------------------------------------------------
    default_merge_strategy: str = "keep_most_complete"
    merge_conflict_resolution: str = "most_complete"

    # -- Pipeline processing -------------------------------------------------
    pipeline_checkpoint_interval: int = 1000
    pipeline_timeout_seconds: int = 3600
    max_comparisons_per_block: int = 50_000

    # -- Cache ---------------------------------------------------------------
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = True

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Metrics -------------------------------------------------------------
    enable_metrics: bool = True

    # -- Rule limits ---------------------------------------------------------
    max_field_weights: int = 50
    max_rules_per_job: int = 100

    # -- Sampling ------------------------------------------------------------
    comparison_sample_rate: float = 1.0

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DuplicateDetectorConfig:
        """Build a DuplicateDetectorConfig from environment variables.

        Every field can be overridden via ``GL_DD_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated DuplicateDetectorConfig instance.
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
            s3_bucket=_str("S3_BUCKET", cls.s3_bucket),
            # Record processing
            max_records_per_job=_int(
                "MAX_RECORDS_PER_JOB", cls.max_records_per_job,
            ),
            default_batch_size=_int(
                "DEFAULT_BATCH_SIZE", cls.default_batch_size,
            ),
            # Fingerprinting
            fingerprint_algorithm=_str(
                "FINGERPRINT_ALGORITHM", cls.fingerprint_algorithm,
            ),
            fingerprint_normalize=_bool(
                "FINGERPRINT_NORMALIZE", cls.fingerprint_normalize,
            ),
            # Blocking strategy
            blocking_strategy=_str(
                "BLOCKING_STRATEGY", cls.blocking_strategy,
            ),
            blocking_window_size=_int(
                "BLOCKING_WINDOW_SIZE", cls.blocking_window_size,
            ),
            blocking_key_size=_int(
                "BLOCKING_KEY_SIZE", cls.blocking_key_size,
            ),
            canopy_tight_threshold=_float(
                "CANOPY_TIGHT_THRESHOLD", cls.canopy_tight_threshold,
            ),
            canopy_loose_threshold=_float(
                "CANOPY_LOOSE_THRESHOLD", cls.canopy_loose_threshold,
            ),
            # Similarity comparison
            default_similarity_algorithm=_str(
                "DEFAULT_SIMILARITY_ALGORITHM",
                cls.default_similarity_algorithm,
            ),
            ngram_size=_int(
                "NGRAM_SIZE", cls.ngram_size,
            ),
            # Match classification
            match_threshold=_float(
                "MATCH_THRESHOLD", cls.match_threshold,
            ),
            possible_threshold=_float(
                "POSSIBLE_THRESHOLD", cls.possible_threshold,
            ),
            non_match_threshold=_float(
                "NON_MATCH_THRESHOLD", cls.non_match_threshold,
            ),
            use_fellegi_sunter=_bool(
                "USE_FELLEGI_SUNTER", cls.use_fellegi_sunter,
            ),
            # Clustering
            cluster_algorithm=_str(
                "CLUSTER_ALGORITHM", cls.cluster_algorithm,
            ),
            cluster_min_quality=_float(
                "CLUSTER_MIN_QUALITY", cls.cluster_min_quality,
            ),
            # Merge strategy
            default_merge_strategy=_str(
                "DEFAULT_MERGE_STRATEGY", cls.default_merge_strategy,
            ),
            merge_conflict_resolution=_str(
                "MERGE_CONFLICT_RESOLUTION",
                cls.merge_conflict_resolution,
            ),
            # Pipeline processing
            pipeline_checkpoint_interval=_int(
                "PIPELINE_CHECKPOINT_INTERVAL",
                cls.pipeline_checkpoint_interval,
            ),
            pipeline_timeout_seconds=_int(
                "PIPELINE_TIMEOUT_SECONDS",
                cls.pipeline_timeout_seconds,
            ),
            max_comparisons_per_block=_int(
                "MAX_COMPARISONS_PER_BLOCK",
                cls.max_comparisons_per_block,
            ),
            # Cache
            cache_ttl_seconds=_int(
                "CACHE_TTL_SECONDS", cls.cache_ttl_seconds,
            ),
            cache_enabled=_bool(
                "CACHE_ENABLED", cls.cache_enabled,
            ),
            # Pool sizing
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Metrics
            enable_metrics=_bool(
                "ENABLE_METRICS", cls.enable_metrics,
            ),
            # Rule limits
            max_field_weights=_int(
                "MAX_FIELD_WEIGHTS", cls.max_field_weights,
            ),
            max_rules_per_job=_int(
                "MAX_RULES_PER_JOB", cls.max_rules_per_job,
            ),
            # Sampling
            comparison_sample_rate=_float(
                "COMPARISON_SAMPLE_RATE", cls.comparison_sample_rate,
            ),
        )

        logger.info(
            "DuplicateDetectorConfig loaded: max_records=%d, batch=%d, "
            "fingerprint=%s (normalize=%s), blocking=%s (window=%d, key=%d), "
            "canopy=[tight=%.2f loose=%.2f], similarity=%s (ngram=%d), "
            "thresholds=[match=%.2f possible=%.2f non_match=%.2f], "
            "fellegi_sunter=%s, cluster=%s (min_quality=%.2f), "
            "merge=%s (conflict=%s), checkpoint=%d, timeout=%ds, "
            "max_cmp_block=%d, cache_ttl=%ds (enabled=%s), "
            "pool=%d-%d, metrics=%s, max_weights=%d, max_rules=%d, "
            "sample_rate=%.2f",
            config.max_records_per_job,
            config.default_batch_size,
            config.fingerprint_algorithm,
            config.fingerprint_normalize,
            config.blocking_strategy,
            config.blocking_window_size,
            config.blocking_key_size,
            config.canopy_tight_threshold,
            config.canopy_loose_threshold,
            config.default_similarity_algorithm,
            config.ngram_size,
            config.match_threshold,
            config.possible_threshold,
            config.non_match_threshold,
            config.use_fellegi_sunter,
            config.cluster_algorithm,
            config.cluster_min_quality,
            config.default_merge_strategy,
            config.merge_conflict_resolution,
            config.pipeline_checkpoint_interval,
            config.pipeline_timeout_seconds,
            config.max_comparisons_per_block,
            config.cache_ttl_seconds,
            config.cache_enabled,
            config.pool_min_size,
            config.pool_max_size,
            config.enable_metrics,
            config.max_field_weights,
            config.max_rules_per_job,
            config.comparison_sample_rate,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[DuplicateDetectorConfig] = None
_config_lock = threading.Lock()


def get_config() -> DuplicateDetectorConfig:
    """Return the singleton DuplicateDetectorConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path.

    Returns:
        DuplicateDetectorConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = DuplicateDetectorConfig.from_env()
    return _config_instance


def set_config(config: DuplicateDetectorConfig) -> None:
    """Replace the singleton DuplicateDetectorConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("DuplicateDetectorConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "DuplicateDetectorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
