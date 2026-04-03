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
    >>> from greenlang.agents.data.duplicate_detector.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.match_threshold, cfg.blocking_strategy)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
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

_ENV_PREFIX = "GL_DD_"


# ---------------------------------------------------------------------------
# DuplicateDetectorConfig
# ---------------------------------------------------------------------------


@dataclass
class DuplicateDetectorConfig(BaseDataConfig):
    """Configuration for the GreenLang Duplicate Detection SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only dedup-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_DD_`` prefix.

    Attributes:
        s3_bucket: S3 bucket name for artifact and report storage.
        max_records_per_job: Maximum records allowed in a single dedup job.
        default_batch_size: Default batch size for record processing.
        fingerprint_algorithm: Algorithm for record fingerprinting.
        fingerprint_normalize: Whether to normalize fields before fingerprinting.
        blocking_strategy: Strategy for candidate pair generation.
        blocking_window_size: Window size for sorted neighborhood blocking.
        blocking_key_size: Number of leading characters for blocking keys.
        canopy_tight_threshold: Tight distance threshold for canopy clustering.
        canopy_loose_threshold: Loose distance threshold for canopy clustering.
        default_similarity_algorithm: Default field-level similarity algorithm.
        ngram_size: Character n-gram size for ngram similarity.
        match_threshold: Score threshold for MATCH classification.
        possible_threshold: Score threshold for POSSIBLE classification.
        non_match_threshold: Score threshold for NON_MATCH classification.
        use_fellegi_sunter: Whether to use Fellegi-Sunter probabilistic model.
        cluster_algorithm: Algorithm for grouping matched pairs.
        cluster_min_quality: Minimum quality score for a valid cluster.
        default_merge_strategy: Default strategy for merging duplicates.
        merge_conflict_resolution: Default field-level conflict resolution.
        pipeline_checkpoint_interval: Records between checkpoint saves.
        pipeline_timeout_seconds: Maximum wall-clock time for a pipeline run.
        max_comparisons_per_block: Maximum comparisons per block.
        cache_ttl_seconds: Default cache time-to-live in seconds.
        cache_enabled: Whether caching is enabled.
        enable_metrics: Whether Prometheus metrics collection is enabled.
        max_field_weights: Maximum field weight entries per rule.
        max_rules_per_job: Maximum dedup rules allowed per job.
        comparison_sample_rate: Fraction of comparisons to execute.
    """

    # -- S3 storage (different field name from base) -------------------------
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

    # -- Metrics -------------------------------------------------------------
    enable_metrics: bool = True

    # -- Rule limits ---------------------------------------------------------
    max_field_weights: int = 50
    max_rules_per_job: int = 100

    # -- Sampling ------------------------------------------------------------
    comparison_sample_rate: float = 1.0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DuplicateDetectorConfig:
        """Build a DuplicateDetectorConfig from environment variables.

        Every field can be overridden via ``GL_DD_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated DuplicateDetectorConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            # S3 storage
            s3_bucket=env.str("S3_BUCKET", cls.s3_bucket),
            # Record processing
            max_records_per_job=env.int(
                "MAX_RECORDS_PER_JOB", cls.max_records_per_job,
            ),
            default_batch_size=env.int(
                "DEFAULT_BATCH_SIZE", cls.default_batch_size,
            ),
            # Fingerprinting
            fingerprint_algorithm=env.str(
                "FINGERPRINT_ALGORITHM", cls.fingerprint_algorithm,
            ),
            fingerprint_normalize=env.bool(
                "FINGERPRINT_NORMALIZE", cls.fingerprint_normalize,
            ),
            # Blocking strategy
            blocking_strategy=env.str(
                "BLOCKING_STRATEGY", cls.blocking_strategy,
            ),
            blocking_window_size=env.int(
                "BLOCKING_WINDOW_SIZE", cls.blocking_window_size,
            ),
            blocking_key_size=env.int(
                "BLOCKING_KEY_SIZE", cls.blocking_key_size,
            ),
            canopy_tight_threshold=env.float(
                "CANOPY_TIGHT_THRESHOLD", cls.canopy_tight_threshold,
            ),
            canopy_loose_threshold=env.float(
                "CANOPY_LOOSE_THRESHOLD", cls.canopy_loose_threshold,
            ),
            # Similarity comparison
            default_similarity_algorithm=env.str(
                "DEFAULT_SIMILARITY_ALGORITHM",
                cls.default_similarity_algorithm,
            ),
            ngram_size=env.int(
                "NGRAM_SIZE", cls.ngram_size,
            ),
            # Match classification
            match_threshold=env.float(
                "MATCH_THRESHOLD", cls.match_threshold,
            ),
            possible_threshold=env.float(
                "POSSIBLE_THRESHOLD", cls.possible_threshold,
            ),
            non_match_threshold=env.float(
                "NON_MATCH_THRESHOLD", cls.non_match_threshold,
            ),
            use_fellegi_sunter=env.bool(
                "USE_FELLEGI_SUNTER", cls.use_fellegi_sunter,
            ),
            # Clustering
            cluster_algorithm=env.str(
                "CLUSTER_ALGORITHM", cls.cluster_algorithm,
            ),
            cluster_min_quality=env.float(
                "CLUSTER_MIN_QUALITY", cls.cluster_min_quality,
            ),
            # Merge strategy
            default_merge_strategy=env.str(
                "DEFAULT_MERGE_STRATEGY", cls.default_merge_strategy,
            ),
            merge_conflict_resolution=env.str(
                "MERGE_CONFLICT_RESOLUTION",
                cls.merge_conflict_resolution,
            ),
            # Pipeline processing
            pipeline_checkpoint_interval=env.int(
                "PIPELINE_CHECKPOINT_INTERVAL",
                cls.pipeline_checkpoint_interval,
            ),
            pipeline_timeout_seconds=env.int(
                "PIPELINE_TIMEOUT_SECONDS",
                cls.pipeline_timeout_seconds,
            ),
            max_comparisons_per_block=env.int(
                "MAX_COMPARISONS_PER_BLOCK",
                cls.max_comparisons_per_block,
            ),
            # Cache
            cache_ttl_seconds=env.int(
                "CACHE_TTL_SECONDS", cls.cache_ttl_seconds,
            ),
            cache_enabled=env.bool(
                "CACHE_ENABLED", cls.cache_enabled,
            ),
            # Metrics
            enable_metrics=env.bool(
                "ENABLE_METRICS", cls.enable_metrics,
            ),
            # Rule limits
            max_field_weights=env.int(
                "MAX_FIELD_WEIGHTS", cls.max_field_weights,
            ),
            max_rules_per_job=env.int(
                "MAX_RULES_PER_JOB", cls.max_rules_per_job,
            ),
            # Sampling
            comparison_sample_rate=env.float(
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

get_config, set_config, reset_config = create_config_singleton(
    DuplicateDetectorConfig, _ENV_PREFIX,
)

__all__ = [
    "DuplicateDetectorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
