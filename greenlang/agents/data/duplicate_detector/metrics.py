# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-011: Duplicate Detection Agent

12 Prometheus metrics for duplicate detection service monitoring with
graceful fallback when prometheus_client is not installed.

Metrics:
    1.  gl_dd_jobs_processed_total (Counter, labels: status)
    2.  gl_dd_records_fingerprinted_total (Counter, labels: algorithm)
    3.  gl_dd_blocks_created_total (Counter, labels: strategy)
    4.  gl_dd_comparisons_performed_total (Counter, labels: algorithm)
    5.  gl_dd_matches_found_total (Counter, labels: classification)
    6.  gl_dd_clusters_formed_total (Counter, labels: algorithm)
    7.  gl_dd_merges_completed_total (Counter, labels: strategy)
    8.  gl_dd_merge_conflicts_total (Counter, labels: resolution)
    9.  gl_dd_processing_duration_seconds (Histogram, labels: operation)
    10. gl_dd_similarity_score (Histogram, labels: algorithm)
    11. gl_dd_active_jobs (Gauge)
    12. gl_dd_processing_errors_total (Counter, labels: error_type)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; duplicate detection metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Dedup jobs processed by status
    dd_jobs_processed_total = Counter(
        "gl_dd_jobs_processed_total",
        "Total deduplication jobs processed",
        labelnames=["status"],
    )

    # 2. Records fingerprinted by algorithm
    dd_records_fingerprinted_total = Counter(
        "gl_dd_records_fingerprinted_total",
        "Total records fingerprinted",
        labelnames=["algorithm"],
    )

    # 3. Blocks created by strategy
    dd_blocks_created_total = Counter(
        "gl_dd_blocks_created_total",
        "Total blocking partitions created",
        labelnames=["strategy"],
    )

    # 4. Comparisons performed by algorithm
    dd_comparisons_performed_total = Counter(
        "gl_dd_comparisons_performed_total",
        "Total pairwise comparisons performed",
        labelnames=["algorithm"],
    )

    # 5. Matches found by classification
    dd_matches_found_total = Counter(
        "gl_dd_matches_found_total",
        "Total duplicate matches found",
        labelnames=["classification"],
    )

    # 6. Clusters formed by algorithm
    dd_clusters_formed_total = Counter(
        "gl_dd_clusters_formed_total",
        "Total duplicate clusters formed",
        labelnames=["algorithm"],
    )

    # 7. Merges completed by strategy
    dd_merges_completed_total = Counter(
        "gl_dd_merges_completed_total",
        "Total record merges completed",
        labelnames=["strategy"],
    )

    # 8. Merge conflicts by resolution method
    dd_merge_conflicts_total = Counter(
        "gl_dd_merge_conflicts_total",
        "Total merge conflicts encountered",
        labelnames=["resolution"],
    )

    # 9. Processing duration histogram by operation type
    dd_processing_duration_seconds = Histogram(
        "gl_dd_processing_duration_seconds",
        "Duplicate detection processing duration in seconds",
        labelnames=["operation"],
        buckets=(
            0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
            5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
        ),
    )

    # 10. Similarity score distribution by algorithm
    dd_similarity_score = Histogram(
        "gl_dd_similarity_score",
        "Similarity score distribution",
        labelnames=["algorithm"],
        buckets=(
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
        ),
    )

    # 11. Currently active dedup jobs gauge
    dd_active_jobs = Gauge(
        "gl_dd_active_jobs",
        "Number of currently active deduplication jobs",
    )

    # 12. Processing errors by error type
    dd_processing_errors_total = Counter(
        "gl_dd_processing_errors_total",
        "Total processing errors encountered",
        labelnames=["error_type"],
    )

else:
    # No-op placeholders
    dd_jobs_processed_total = None  # type: ignore[assignment]
    dd_records_fingerprinted_total = None  # type: ignore[assignment]
    dd_blocks_created_total = None  # type: ignore[assignment]
    dd_comparisons_performed_total = None  # type: ignore[assignment]
    dd_matches_found_total = None  # type: ignore[assignment]
    dd_clusters_formed_total = None  # type: ignore[assignment]
    dd_merges_completed_total = None  # type: ignore[assignment]
    dd_merge_conflicts_total = None  # type: ignore[assignment]
    dd_processing_duration_seconds = None  # type: ignore[assignment]
    dd_similarity_score = None  # type: ignore[assignment]
    dd_active_jobs = None  # type: ignore[assignment]
    dd_processing_errors_total = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def inc_jobs(status: str) -> None:
    """Record a dedup job processed event.

    Args:
        status: Job status (completed, failed, cancelled, timeout).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dd_jobs_processed_total.labels(
        status=status,
    ).inc()


def inc_fingerprints(algorithm: str, count: int = 1) -> None:
    """Record records fingerprinted.

    Args:
        algorithm: Fingerprinting algorithm (sha256, simhash, minhash).
        count: Number of records fingerprinted.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dd_records_fingerprinted_total.labels(
        algorithm=algorithm,
    ).inc(count)


def inc_blocks(strategy: str, count: int = 1) -> None:
    """Record blocking partitions created.

    Args:
        strategy: Blocking strategy (sorted_neighborhood, standard, canopy, none).
        count: Number of blocks created.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dd_blocks_created_total.labels(
        strategy=strategy,
    ).inc(count)


def inc_comparisons(algorithm: str, count: int = 1) -> None:
    """Record pairwise comparisons performed.

    Args:
        algorithm: Comparison algorithm (exact, levenshtein, jaro_winkler,
            soundex, ngram, tfidf_cosine, numeric, date).
        count: Number of comparisons performed.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dd_comparisons_performed_total.labels(
        algorithm=algorithm,
    ).inc(count)


def inc_matches(classification: str, count: int = 1) -> None:
    """Record duplicate matches found.

    Args:
        classification: Match classification (match, possible, non_match).
        count: Number of matches found.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dd_matches_found_total.labels(
        classification=classification,
    ).inc(count)


def inc_clusters(algorithm: str, count: int = 1) -> None:
    """Record duplicate clusters formed.

    Args:
        algorithm: Clustering algorithm (union_find, connected_components).
        count: Number of clusters formed.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dd_clusters_formed_total.labels(
        algorithm=algorithm,
    ).inc(count)


def inc_merges(strategy: str, count: int = 1) -> None:
    """Record record merges completed.

    Args:
        strategy: Merge strategy (keep_first, keep_latest, keep_most_complete,
            merge_fields, golden_record, custom).
        count: Number of merges completed.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dd_merges_completed_total.labels(
        strategy=strategy,
    ).inc(count)


def inc_conflicts(resolution: str, count: int = 1) -> None:
    """Record merge conflicts encountered.

    Args:
        resolution: Conflict resolution method (first, latest, most_complete,
            longest, shortest).
        count: Number of conflicts encountered.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dd_merge_conflicts_total.labels(
        resolution=resolution,
    ).inc(count)


def observe_duration(operation: str, duration: float) -> None:
    """Record processing duration for an operation.

    Args:
        operation: Operation type (fingerprint, block, compare, classify,
            cluster, merge, pipeline, job).
        duration: Duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dd_processing_duration_seconds.labels(
        operation=operation,
    ).observe(duration)


def observe_similarity(algorithm: str, score: float) -> None:
    """Record a similarity score observation.

    Args:
        algorithm: Similarity algorithm used.
        score: Similarity score (0.0 - 1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dd_similarity_score.labels(
        algorithm=algorithm,
    ).observe(score)


def set_active_jobs(count: int) -> None:
    """Set the active jobs gauge to an absolute value.

    Args:
        count: Number of currently active jobs.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dd_active_jobs.set(count)


def inc_errors(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data,
            integration, comparison, merge, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dd_processing_errors_total.labels(
        error_type=error_type,
    ).inc()


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "dd_jobs_processed_total",
    "dd_records_fingerprinted_total",
    "dd_blocks_created_total",
    "dd_comparisons_performed_total",
    "dd_matches_found_total",
    "dd_clusters_formed_total",
    "dd_merges_completed_total",
    "dd_merge_conflicts_total",
    "dd_processing_duration_seconds",
    "dd_similarity_score",
    "dd_active_jobs",
    "dd_processing_errors_total",
    # Helper functions
    "inc_jobs",
    "inc_fingerprints",
    "inc_blocks",
    "inc_comparisons",
    "inc_matches",
    "inc_clusters",
    "inc_merges",
    "inc_conflicts",
    "observe_duration",
    "observe_similarity",
    "set_active_jobs",
    "inc_errors",
]
