# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-011: Duplicate Detection Agent

12 Prometheus metrics for duplicate detection service monitoring with
graceful fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_dd_operations_total (Counter, labels: type, tenant_id)
    2.  gl_dd_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_dd_validation_errors_total (Counter, labels: severity, type)
    4.  gl_dd_batch_jobs_total (Counter, labels: status)
    5.  gl_dd_active_jobs (Gauge)
    6.  gl_dd_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_dd_jobs_processed_total (Counter, labels: status)
    8.  gl_dd_records_fingerprinted_total (Counter, labels: algorithm)
    9.  gl_dd_blocks_created_total (Counter, labels: strategy)
    10. gl_dd_comparisons_performed_total (Counter, labels: algorithm)
    11. gl_dd_matches_found_total (Counter, labels: classification)
    12. gl_dd_similarity_score (Histogram, labels: algorithm)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.metrics import (
    CONFIDENCE_BUCKETS,
    PROMETHEUS_AVAILABLE,
    MetricsFactory,
)

# ---------------------------------------------------------------------------
# Standard metrics (6 of 12) via factory
# ---------------------------------------------------------------------------

m = MetricsFactory(
    "gl_dd",
    "Duplicate Detector",
    duration_buckets=(
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
        5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    ),
)

# Backward-compat alias
dd_active_jobs = m.active_jobs

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

dd_jobs_processed_total = m.create_custom_counter(
    "jobs_processed_total",
    "Total deduplication jobs processed",
    labelnames=["status"],
)

dd_records_fingerprinted_total = m.create_custom_counter(
    "records_fingerprinted_total",
    "Total records fingerprinted",
    labelnames=["algorithm"],
)

dd_blocks_created_total = m.create_custom_counter(
    "blocks_created_total",
    "Total blocking partitions created",
    labelnames=["strategy"],
)

dd_comparisons_performed_total = m.create_custom_counter(
    "comparisons_performed_total",
    "Total pairwise comparisons performed",
    labelnames=["algorithm"],
)

dd_matches_found_total = m.create_custom_counter(
    "matches_found_total",
    "Total duplicate matches found",
    labelnames=["classification"],
)

dd_clusters_formed_total = m.create_custom_counter(
    "clusters_formed_total",
    "Total duplicate clusters formed",
    labelnames=["algorithm"],
)

dd_merges_completed_total = m.create_custom_counter(
    "merges_completed_total",
    "Total record merges completed",
    labelnames=["strategy"],
)

dd_merge_conflicts_total = m.create_custom_counter(
    "merge_conflicts_total",
    "Total merge conflicts encountered",
    labelnames=["resolution"],
)

dd_processing_duration_seconds = m.create_custom_histogram(
    "processing_duration_seconds_detail",
    "Duplicate detection processing duration in seconds",
    buckets=(
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
        5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    ),
    labelnames=["operation"],
)

dd_similarity_score = m.create_custom_histogram(
    "similarity_score",
    "Similarity score distribution",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    labelnames=["algorithm"],
)

dd_processing_errors_total = m.create_custom_counter(
    "processing_errors_total",
    "Total processing errors encountered",
    labelnames=["error_type"],
)


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def inc_jobs(status: str) -> None:
    """Record a dedup job processed event.

    Args:
        status: Job status (completed, failed, cancelled, timeout).
    """
    m.safe_inc(dd_jobs_processed_total, 1, status=status)


def inc_fingerprints(algorithm: str, count: int = 1) -> None:
    """Record records fingerprinted.

    Args:
        algorithm: Fingerprinting algorithm (sha256, simhash, minhash).
        count: Number of records fingerprinted.
    """
    m.safe_inc(dd_records_fingerprinted_total, count, algorithm=algorithm)


def inc_blocks(strategy: str, count: int = 1) -> None:
    """Record blocking partitions created.

    Args:
        strategy: Blocking strategy (sorted_neighborhood, standard, canopy, none).
        count: Number of blocks created.
    """
    m.safe_inc(dd_blocks_created_total, count, strategy=strategy)


def inc_comparisons(algorithm: str, count: int = 1) -> None:
    """Record pairwise comparisons performed.

    Args:
        algorithm: Comparison algorithm (exact, levenshtein, jaro_winkler,
            soundex, ngram, tfidf_cosine, numeric, date).
        count: Number of comparisons performed.
    """
    m.safe_inc(dd_comparisons_performed_total, count, algorithm=algorithm)


def inc_matches(classification: str, count: int = 1) -> None:
    """Record duplicate matches found.

    Args:
        classification: Match classification (match, possible, non_match).
        count: Number of matches found.
    """
    m.safe_inc(dd_matches_found_total, count, classification=classification)


def inc_clusters(algorithm: str, count: int = 1) -> None:
    """Record duplicate clusters formed.

    Args:
        algorithm: Clustering algorithm (union_find, connected_components).
        count: Number of clusters formed.
    """
    m.safe_inc(dd_clusters_formed_total, count, algorithm=algorithm)


def inc_merges(strategy: str, count: int = 1) -> None:
    """Record record merges completed.

    Args:
        strategy: Merge strategy (keep_first, keep_latest, keep_most_complete,
            merge_fields, golden_record, custom).
        count: Number of merges completed.
    """
    m.safe_inc(dd_merges_completed_total, count, strategy=strategy)


def inc_conflicts(resolution: str, count: int = 1) -> None:
    """Record merge conflicts encountered.

    Args:
        resolution: Conflict resolution method (first, latest, most_complete,
            longest, shortest).
        count: Number of conflicts encountered.
    """
    m.safe_inc(dd_merge_conflicts_total, count, resolution=resolution)


def observe_duration(operation: str, duration: float) -> None:
    """Record processing duration for an operation.

    Args:
        operation: Operation type (fingerprint, block, compare, classify,
            cluster, merge, pipeline, job).
        duration: Duration in seconds.
    """
    m.safe_observe(dd_processing_duration_seconds, duration, operation=operation)


def observe_similarity(algorithm: str, score: float) -> None:
    """Record a similarity score observation.

    Args:
        algorithm: Similarity algorithm used.
        score: Similarity score (0.0 - 1.0).
    """
    m.safe_observe(dd_similarity_score, score, algorithm=algorithm)


def set_active_jobs(count: int) -> None:
    """Set the active jobs gauge to an absolute value.

    Args:
        count: Number of currently active jobs.
    """
    m.safe_set(dd_active_jobs, count)


def inc_errors(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data,
            integration, comparison, merge, unknown).
    """
    m.safe_inc(dd_processing_errors_total, 1, error_type=error_type)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
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
