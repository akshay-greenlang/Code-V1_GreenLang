# -*- coding: utf-8 -*-
"""
Cross-Source Reconciliation Agent Service Configuration - AGENT-DATA-015

Centralized configuration for the Cross-Source Reconciliation SDK covering:
- Database, cache connection defaults
- Record processing limits (max records, batch size, max sources)
- Default match threshold and tolerance settings (percentage, absolute)
- Default resolution strategy and source credibility weighting
- Temporal alignment and fuzzy matching feature toggles
- Match candidate limits and golden record generation
- Worker, pool, cache, rate limiting, and provenance settings
- Manual review and discrepancy severity thresholds

All settings can be overridden via environment variables with the
``GL_CSR_`` prefix (e.g. ``GL_CSR_MAX_RECORDS``).

Example:
    >>> from greenlang.agents.data.cross_source_reconciliation.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_match_threshold, cfg.default_resolution_strategy)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
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

_ENV_PREFIX = "GL_CSR_"


# ---------------------------------------------------------------------------
# CrossSourceReconciliationConfig
# ---------------------------------------------------------------------------


@dataclass
class CrossSourceReconciliationConfig(BaseDataConfig):
    """Configuration for the GreenLang Cross-Source Reconciliation SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only reconciliation-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_CSR_`` prefix.

    Attributes:
        batch_size: Default batch size for record processing.
        max_records: Maximum records allowed in a single reconciliation job.
        max_sources: Maximum number of data sources per reconciliation job.
        default_match_threshold: Minimum similarity score for a match.
        default_tolerance_pct: Default percentage tolerance for numeric comparisons.
        default_tolerance_abs: Default absolute tolerance for numeric comparisons.
        default_resolution_strategy: Strategy for resolving discrepancies.
        source_credibility_weight: Weight applied to source credibility scores.
        temporal_alignment_enabled: Whether temporal alignment is enabled.
        fuzzy_matching_enabled: Whether fuzzy string matching is enabled.
        max_match_candidates: Maximum candidate records per source record.
        enable_golden_records: Whether golden record generation is enabled.
        max_workers: Number of parallel workers for batch processing.
        pool_size: Connection pool size for database connections.
        cache_ttl: Cache time-to-live in seconds.
        rate_limit: Rate limit in requests per minute.
        enable_provenance: Whether SHA-256 provenance tracking is enabled.
        manual_review_threshold: Confidence score below which manual review is triggered.
        critical_discrepancy_pct: Percentage threshold for critical discrepancies.
        high_discrepancy_pct: Percentage threshold for high discrepancies.
        medium_discrepancy_pct: Percentage threshold for medium discrepancies.
        genesis_hash: Seed string for the SHA-256 provenance chain.
    """

    # -- Record processing ---------------------------------------------------
    batch_size: int = 1000
    max_records: int = 100_000
    max_sources: int = 20

    # -- Matching thresholds -------------------------------------------------
    default_match_threshold: float = 0.85
    default_tolerance_pct: float = 5.0
    default_tolerance_abs: float = 0.01

    # -- Resolution strategy -------------------------------------------------
    default_resolution_strategy: str = "priority_wins"

    # -- Source credibility --------------------------------------------------
    source_credibility_weight: float = 0.4

    # -- Feature toggles -----------------------------------------------------
    temporal_alignment_enabled: bool = True
    fuzzy_matching_enabled: bool = True

    # -- Match candidates ----------------------------------------------------
    max_match_candidates: int = 100

    # -- Golden records ------------------------------------------------------
    enable_golden_records: bool = True

    # -- Worker pool ---------------------------------------------------------
    max_workers: int = 4
    pool_size: int = 5

    # -- Cache ---------------------------------------------------------------
    cache_ttl: int = 3600

    # -- Rate limiting -------------------------------------------------------
    rate_limit: int = 100

    # -- Provenance ----------------------------------------------------------
    enable_provenance: bool = True

    # -- Manual review -------------------------------------------------------
    manual_review_threshold: float = 0.6

    # -- Discrepancy severity thresholds ------------------------------------
    critical_discrepancy_pct: float = 50.0
    high_discrepancy_pct: float = 25.0
    medium_discrepancy_pct: float = 10.0

    # -- Genesis hash --------------------------------------------------------
    genesis_hash: str = "greenlang-cross-source-reconciliation-genesis"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> CrossSourceReconciliationConfig:
        """Build a CrossSourceReconciliationConfig from environment variables.

        Every field can be overridden via ``GL_CSR_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated CrossSourceReconciliationConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            # Record processing
            batch_size=env.int("BATCH_SIZE", cls.batch_size),
            max_records=env.int("MAX_RECORDS", cls.max_records),
            max_sources=env.int("MAX_SOURCES", cls.max_sources),
            # Matching thresholds
            default_match_threshold=env.float(
                "DEFAULT_MATCH_THRESHOLD", cls.default_match_threshold,
            ),
            default_tolerance_pct=env.float(
                "DEFAULT_TOLERANCE_PCT", cls.default_tolerance_pct,
            ),
            default_tolerance_abs=env.float(
                "DEFAULT_TOLERANCE_ABS", cls.default_tolerance_abs,
            ),
            # Resolution strategy
            default_resolution_strategy=env.str(
                "DEFAULT_RESOLUTION_STRATEGY",
                cls.default_resolution_strategy,
            ),
            # Source credibility
            source_credibility_weight=env.float(
                "SOURCE_CREDIBILITY_WEIGHT",
                cls.source_credibility_weight,
            ),
            # Feature toggles
            temporal_alignment_enabled=env.bool(
                "TEMPORAL_ALIGNMENT_ENABLED",
                cls.temporal_alignment_enabled,
            ),
            fuzzy_matching_enabled=env.bool(
                "FUZZY_MATCHING_ENABLED",
                cls.fuzzy_matching_enabled,
            ),
            # Match candidates
            max_match_candidates=env.int(
                "MAX_MATCH_CANDIDATES",
                cls.max_match_candidates,
            ),
            # Golden records
            enable_golden_records=env.bool(
                "ENABLE_GOLDEN_RECORDS",
                cls.enable_golden_records,
            ),
            # Worker pool
            max_workers=env.int("MAX_WORKERS", cls.max_workers),
            pool_size=env.int("POOL_SIZE", cls.pool_size),
            # Cache
            cache_ttl=env.int("CACHE_TTL", cls.cache_ttl),
            # Rate limiting
            rate_limit=env.int("RATE_LIMIT", cls.rate_limit),
            # Provenance
            enable_provenance=env.bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            # Manual review
            manual_review_threshold=env.float(
                "MANUAL_REVIEW_THRESHOLD",
                cls.manual_review_threshold,
            ),
            # Discrepancy severity thresholds
            critical_discrepancy_pct=env.float(
                "CRITICAL_DISCREPANCY_PCT",
                cls.critical_discrepancy_pct,
            ),
            high_discrepancy_pct=env.float(
                "HIGH_DISCREPANCY_PCT",
                cls.high_discrepancy_pct,
            ),
            medium_discrepancy_pct=env.float(
                "MEDIUM_DISCREPANCY_PCT",
                cls.medium_discrepancy_pct,
            ),
            # Genesis hash
            genesis_hash=env.str("GENESIS_HASH", cls.genesis_hash),
        )

        logger.info(
            "CrossSourceReconciliationConfig loaded: batch=%d, "
            "max_records=%d, max_sources=%d, "
            "match_threshold=%.2f, tolerance_pct=%.2f, "
            "tolerance_abs=%.4f, "
            "resolution=%s, credibility_weight=%.2f, "
            "temporal_align=%s, fuzzy=%s, "
            "max_candidates=%d, golden_records=%s, "
            "workers=%d, pool=%d, cache_ttl=%ds, "
            "rate=%d rpm, provenance=%s, "
            "manual_review=%.2f, "
            "critical=%.1f%%, high=%.1f%%, medium=%.1f%%",
            config.batch_size,
            config.max_records,
            config.max_sources,
            config.default_match_threshold,
            config.default_tolerance_pct,
            config.default_tolerance_abs,
            config.default_resolution_strategy,
            config.source_credibility_weight,
            config.temporal_alignment_enabled,
            config.fuzzy_matching_enabled,
            config.max_match_candidates,
            config.enable_golden_records,
            config.max_workers,
            config.pool_size,
            config.cache_ttl,
            config.rate_limit,
            config.enable_provenance,
            config.manual_review_threshold,
            config.critical_discrepancy_pct,
            config.high_discrepancy_pct,
            config.medium_discrepancy_pct,
        )
        return config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate all configuration constraints after initialization.

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
        if self.max_sources < 2:
            errors.append("max_sources must be >= 2")

        # Matching thresholds
        if not 0.0 <= self.default_match_threshold <= 1.0:
            errors.append(
                "default_match_threshold must be between 0.0 and 1.0"
            )
        if self.default_tolerance_pct < 0.0:
            errors.append("default_tolerance_pct must be >= 0.0")
        if self.default_tolerance_abs < 0.0:
            errors.append("default_tolerance_abs must be >= 0.0")

        # Resolution strategy
        valid_strategies = (
            "priority_wins", "most_recent_wins", "average",
            "median", "manual_review", "consensus",
        )
        if self.default_resolution_strategy not in valid_strategies:
            errors.append(
                f"default_resolution_strategy must be one of "
                f"{valid_strategies}, "
                f"got '{self.default_resolution_strategy}'"
            )

        # Source credibility
        if not 0.0 <= self.source_credibility_weight <= 1.0:
            errors.append(
                "source_credibility_weight must be between 0.0 and 1.0"
            )

        # Match candidates
        if self.max_match_candidates < 1:
            errors.append("max_match_candidates must be >= 1")

        # Worker pool
        if self.max_workers < 1:
            errors.append("max_workers must be >= 1")
        if self.pool_size < 1:
            errors.append("pool_size must be >= 1")

        # Cache
        if self.cache_ttl < 0:
            errors.append("cache_ttl must be >= 0")

        # Rate limiting
        if self.rate_limit < 1:
            errors.append("rate_limit must be >= 1")

        # Manual review threshold
        if not 0.0 <= self.manual_review_threshold <= 1.0:
            errors.append(
                "manual_review_threshold must be between 0.0 and 1.0"
            )
        if self.manual_review_threshold > self.default_match_threshold:
            errors.append(
                "manual_review_threshold must be <= "
                "default_match_threshold"
            )

        # Discrepancy severity thresholds (must be ordered)
        if self.medium_discrepancy_pct < 0.0:
            errors.append("medium_discrepancy_pct must be >= 0.0")
        if self.high_discrepancy_pct < 0.0:
            errors.append("high_discrepancy_pct must be >= 0.0")
        if self.critical_discrepancy_pct < 0.0:
            errors.append("critical_discrepancy_pct must be >= 0.0")
        if self.medium_discrepancy_pct >= self.high_discrepancy_pct:
            errors.append(
                "medium_discrepancy_pct must be < high_discrepancy_pct"
            )
        if self.high_discrepancy_pct >= self.critical_discrepancy_pct:
            errors.append(
                "high_discrepancy_pct must be < critical_discrepancy_pct"
            )

        # Log level
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if self.log_level.upper() not in valid_levels:
            errors.append(
                f"log_level must be one of {valid_levels}, "
                f"got '{self.log_level}'"
            )

        # Genesis hash
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        if errors:
            msg = "; ".join(errors)
            logger.error(
                "CrossSourceReconciliationConfig validation failed: %s", msg,
            )
            raise ValueError(
                f"CrossSourceReconciliationConfig validation failed: {msg}"
            )

        logger.debug("CrossSourceReconciliationConfig validated successfully")


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

get_config, set_config, reset_config = create_config_singleton(
    CrossSourceReconciliationConfig, _ENV_PREFIX,
)

__all__ = [
    "CrossSourceReconciliationConfig",
    "get_config",
    "set_config",
    "reset_config",
]
