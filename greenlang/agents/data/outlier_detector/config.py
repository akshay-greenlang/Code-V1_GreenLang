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
    >>> from greenlang.agents.data.outlier_detector.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.iqr_multiplier, cfg.zscore_threshold)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
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

_ENV_PREFIX = "GL_OD_"


# ---------------------------------------------------------------------------
# OutlierDetectorConfig
# ---------------------------------------------------------------------------


@dataclass
class OutlierDetectorConfig(BaseDataConfig):
    """Configuration for the GreenLang Outlier Detector SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only outlier-detector-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_OD_`` prefix.

    Attributes:
        batch_size: Default batch size for record processing.
        max_records: Maximum records allowed in a single detection job.
        iqr_multiplier: Multiplier for IQR fence calculation.
        zscore_threshold: Absolute z-score threshold for flagging outliers.
        mad_threshold: Modified z-score threshold using MAD.
        grubbs_alpha: Significance level for Grubbs test.
        lof_neighbors: Number of neighbors for Local Outlier Factor.
        isolation_trees: Number of trees in Isolation Forest.
        ensemble_method: Method for combining multi-detector scores.
        min_consensus: Minimum detectors that must flag a point.
        enable_contextual: Whether contextual detection is enabled.
        enable_temporal: Whether temporal detection is enabled.
        enable_multivariate: Whether multivariate detection is enabled.
        default_treatment: Default treatment strategy for outliers.
        winsorize_pct: Percentile for winsorization treatment.
        worker_count: Number of parallel workers for batch processing.
        cache_ttl: Cache time-to-live in seconds.
        rate_limit_rpm: Rate limit in requests per minute.
        rate_limit_burst: Maximum burst size for rate limiting.
        enable_provenance: Whether SHA-256 provenance tracking is enabled.
        provenance_hash_algorithm: Hash algorithm for provenance tracking.
        enable_metrics: Whether Prometheus metrics collection is enabled.
    """

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
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> OutlierDetectorConfig:
        """Build an OutlierDetectorConfig from environment variables.

        Every field can be overridden via ``GL_OD_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated OutlierDetectorConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            # Record processing
            batch_size=env.int("BATCH_SIZE", cls.batch_size),
            max_records=env.int("MAX_RECORDS", cls.max_records),
            # Statistical detection
            iqr_multiplier=env.float(
                "IQR_MULTIPLIER", cls.iqr_multiplier,
            ),
            zscore_threshold=env.float(
                "ZSCORE_THRESHOLD", cls.zscore_threshold,
            ),
            mad_threshold=env.float(
                "MAD_THRESHOLD", cls.mad_threshold,
            ),
            grubbs_alpha=env.float(
                "GRUBBS_ALPHA", cls.grubbs_alpha,
            ),
            # LOF / Isolation Forest
            lof_neighbors=env.int(
                "LOF_NEIGHBORS", cls.lof_neighbors,
            ),
            isolation_trees=env.int(
                "ISOLATION_TREES", cls.isolation_trees,
            ),
            # Ensemble detection
            ensemble_method=env.str(
                "ENSEMBLE_METHOD", cls.ensemble_method,
            ),
            min_consensus=env.int(
                "MIN_CONSENSUS", cls.min_consensus,
            ),
            # Contextual / Temporal / Multivariate
            enable_contextual=env.bool(
                "ENABLE_CONTEXTUAL", cls.enable_contextual,
            ),
            enable_temporal=env.bool(
                "ENABLE_TEMPORAL", cls.enable_temporal,
            ),
            enable_multivariate=env.bool(
                "ENABLE_MULTIVARIATE", cls.enable_multivariate,
            ),
            # Treatment
            default_treatment=env.str(
                "DEFAULT_TREATMENT", cls.default_treatment,
            ),
            winsorize_pct=env.float(
                "WINSORIZE_PCT", cls.winsorize_pct,
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

get_config, set_config, reset_config = create_config_singleton(
    OutlierDetectorConfig, _ENV_PREFIX,
)

__all__ = [
    "OutlierDetectorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
