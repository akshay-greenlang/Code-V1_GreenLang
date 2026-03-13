# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-038: Reference Number Generator

40+ Prometheus metrics for reference number generator service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_rng_`` prefix for consistent
identification in Prometheus queries, Grafana dashboards, and alerting
rules across the GreenLang platform.

Metrics (40+ per PRD Section 7.6):
    Counters (12):
        1.  gl_eudr_rng_references_generated_total         - References generated [member_state, mode]
        2.  gl_eudr_rng_references_validated_total          - References validated [result]
        3.  gl_eudr_rng_collisions_detected_total           - Collisions detected [member_state]
        4.  gl_eudr_rng_batches_completed_total             - Batch generations completed [status]
        5.  gl_eudr_rng_references_revoked_total            - References revoked [reason]
        6.  gl_eudr_rng_references_transferred_total        - References transferred [reason]
        7.  gl_eudr_rng_validations_failed_total            - Validation failures [check_type]
        8.  gl_eudr_rng_references_expired_total            - References expired [member_state]
        9.  gl_eudr_rng_sequence_overflows_total            - Sequence overflows [strategy]
        10. gl_eudr_rng_idempotent_hits_total               - Idempotent cache hits
        11. gl_eudr_rng_lock_acquisitions_total             - Distributed lock acquisitions [outcome]
        12. gl_eudr_rng_api_errors_total                    - API errors [operation]

    Histograms (10):
        13. gl_eudr_rng_generation_duration_seconds         - Generation latency [mode]
        14. gl_eudr_rng_validation_duration_seconds         - Validation latency
        15. gl_eudr_rng_batch_generation_duration_seconds   - Batch generation latency
        16. gl_eudr_rng_checksum_computation_duration_seconds - Checksum computation latency [algorithm]
        17. gl_eudr_rng_collision_detection_duration_seconds - Collision detection latency
        18. gl_eudr_rng_sequence_increment_duration_seconds - Sequence increment latency
        19. gl_eudr_rng_lock_acquisition_duration_seconds   - Lock acquisition latency
        20. gl_eudr_rng_lifecycle_transition_duration_seconds - Lifecycle transition latency [transition]
        21. gl_eudr_rng_verification_duration_seconds       - Verification latency
        22. gl_eudr_rng_batch_size_distribution             - Batch size distribution

    Gauges (18):
        23. gl_eudr_rng_active_references                   - Currently active references
        24. gl_eudr_rng_available_sequences                 - Available sequence slots
        25. gl_eudr_rng_sequence_utilization_percent        - Sequence utilization [operator, member_state]
        26. gl_eudr_rng_pending_batches                     - Pending batch requests
        27. gl_eudr_rng_references_expiring_30d             - References expiring within 30 days
        28. gl_eudr_rng_reserved_references                 - Reserved (not yet active) references
        29. gl_eudr_rng_used_references                     - References marked as used
        30. gl_eudr_rng_revoked_references                  - Revoked references
        31. gl_eudr_rng_expired_references                  - Expired references
        32. gl_eudr_rng_total_generated_lifetime            - Total generated since start
        33. gl_eudr_rng_collisions_pending_resolution       - Unresolved collisions
        34. gl_eudr_rng_active_locks                        - Currently held distributed locks
        35. gl_eudr_rng_bloom_filter_size                   - Bloom filter entries
        36. gl_eudr_rng_idempotency_cache_size              - Idempotency cache entries
        37. gl_eudr_rng_db_pool_active                      - Active DB pool connections
        38. gl_eudr_rng_db_pool_idle                        - Idle DB pool connections
        39. gl_eudr_rng_uptime_seconds                      - Service uptime
        40. gl_eudr_rng_last_generation_timestamp           - Last successful generation epoch

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-038 (GL-EUDR-RNG-038)
Status: Production Ready
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus_client not available; metrics disabled")


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

if _PROMETHEUS_AVAILABLE:
    # Counters (12)
    _REFERENCES_GENERATED = Counter(
        "gl_eudr_rng_references_generated_total",
        "Reference numbers generated",
        ["member_state", "mode"],
    )
    _REFERENCES_VALIDATED = Counter(
        "gl_eudr_rng_references_validated_total",
        "Reference numbers validated",
        ["result"],
    )
    _COLLISIONS_DETECTED = Counter(
        "gl_eudr_rng_collisions_detected_total",
        "Reference number collisions detected",
        ["member_state"],
    )
    _BATCHES_COMPLETED = Counter(
        "gl_eudr_rng_batches_completed_total",
        "Batch generation requests completed",
        ["status"],
    )
    _REFERENCES_REVOKED = Counter(
        "gl_eudr_rng_references_revoked_total",
        "Reference numbers revoked",
        ["reason"],
    )
    _REFERENCES_TRANSFERRED = Counter(
        "gl_eudr_rng_references_transferred_total",
        "Reference numbers transferred",
        ["reason"],
    )
    _VALIDATIONS_FAILED = Counter(
        "gl_eudr_rng_validations_failed_total",
        "Reference number validation failures",
        ["check_type"],
    )
    _REFERENCES_EXPIRED = Counter(
        "gl_eudr_rng_references_expired_total",
        "Reference numbers expired",
        ["member_state"],
    )
    _SEQUENCE_OVERFLOWS = Counter(
        "gl_eudr_rng_sequence_overflows_total",
        "Sequence counter overflows",
        ["strategy"],
    )
    _IDEMPOTENT_HITS = Counter(
        "gl_eudr_rng_idempotent_hits_total",
        "Idempotent cache hits (repeat request served from cache)",
    )
    _LOCK_ACQUISITIONS = Counter(
        "gl_eudr_rng_lock_acquisitions_total",
        "Distributed lock acquisition attempts",
        ["outcome"],
    )
    _API_ERRORS = Counter(
        "gl_eudr_rng_api_errors_total",
        "API errors by operation type",
        ["operation"],
    )

    # Histograms (10)
    _GENERATION_DURATION = Histogram(
        "gl_eudr_rng_generation_duration_seconds",
        "Reference number generation latency",
        ["mode"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    )
    _VALIDATION_DURATION = Histogram(
        "gl_eudr_rng_validation_duration_seconds",
        "Reference number validation latency",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    )
    _BATCH_GENERATION_DURATION = Histogram(
        "gl_eudr_rng_batch_generation_duration_seconds",
        "Batch reference number generation latency",
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    _CHECKSUM_DURATION = Histogram(
        "gl_eudr_rng_checksum_computation_duration_seconds",
        "Checksum computation latency",
        ["algorithm"],
        buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05),
    )
    _COLLISION_DETECTION_DURATION = Histogram(
        "gl_eudr_rng_collision_detection_duration_seconds",
        "Collision detection latency",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
    )
    _SEQUENCE_INCREMENT_DURATION = Histogram(
        "gl_eudr_rng_sequence_increment_duration_seconds",
        "Sequence counter atomic increment latency",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    )
    _LOCK_ACQUISITION_DURATION = Histogram(
        "gl_eudr_rng_lock_acquisition_duration_seconds",
        "Distributed lock acquisition latency",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
    )
    _LIFECYCLE_TRANSITION_DURATION = Histogram(
        "gl_eudr_rng_lifecycle_transition_duration_seconds",
        "Lifecycle state transition latency",
        ["transition"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    )
    _VERIFICATION_DURATION = Histogram(
        "gl_eudr_rng_verification_duration_seconds",
        "Reference number verification latency",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    )
    _BATCH_SIZE_DISTRIBUTION = Histogram(
        "gl_eudr_rng_batch_size_distribution",
        "Distribution of batch request sizes",
        buckets=(1, 5, 10, 50, 100, 500, 1000, 5000, 10000),
    )

    # Gauges (18)
    _ACTIVE_REFERENCES = Gauge(
        "gl_eudr_rng_active_references",
        "Currently active reference numbers",
    )
    _AVAILABLE_SEQUENCES = Gauge(
        "gl_eudr_rng_available_sequences",
        "Available sequence counter slots",
    )
    _SEQUENCE_UTILIZATION = Gauge(
        "gl_eudr_rng_sequence_utilization_percent",
        "Sequence utilization percentage",
        ["operator", "member_state"],
    )
    _PENDING_BATCHES = Gauge(
        "gl_eudr_rng_pending_batches",
        "Pending batch generation requests",
    )
    _REFERENCES_EXPIRING_30D = Gauge(
        "gl_eudr_rng_references_expiring_30d",
        "References expiring within 30 days",
    )
    _RESERVED_REFERENCES = Gauge(
        "gl_eudr_rng_reserved_references",
        "Reserved but not yet active references",
    )
    _USED_REFERENCES = Gauge(
        "gl_eudr_rng_used_references",
        "References marked as used",
    )
    _REVOKED_REFERENCES = Gauge(
        "gl_eudr_rng_revoked_references",
        "Revoked reference numbers",
    )
    _EXPIRED_REFERENCES = Gauge(
        "gl_eudr_rng_expired_references",
        "Expired reference numbers",
    )
    _TOTAL_GENERATED_LIFETIME = Gauge(
        "gl_eudr_rng_total_generated_lifetime",
        "Total references generated since service start",
    )
    _COLLISIONS_PENDING = Gauge(
        "gl_eudr_rng_collisions_pending_resolution",
        "Unresolved reference number collisions",
    )
    _ACTIVE_LOCKS = Gauge(
        "gl_eudr_rng_active_locks",
        "Currently held distributed locks",
    )
    _BLOOM_FILTER_SIZE = Gauge(
        "gl_eudr_rng_bloom_filter_size",
        "Number of entries in bloom filter",
    )
    _IDEMPOTENCY_CACHE_SIZE = Gauge(
        "gl_eudr_rng_idempotency_cache_size",
        "Number of entries in idempotency cache",
    )
    _DB_POOL_ACTIVE = Gauge(
        "gl_eudr_rng_db_pool_active",
        "Active database pool connections",
    )
    _DB_POOL_IDLE = Gauge(
        "gl_eudr_rng_db_pool_idle",
        "Idle database pool connections",
    )
    _UPTIME_SECONDS = Gauge(
        "gl_eudr_rng_uptime_seconds",
        "Service uptime in seconds",
    )
    _LAST_GENERATION_TIMESTAMP = Gauge(
        "gl_eudr_rng_last_generation_timestamp",
        "Epoch timestamp of last successful generation",
    )


# ---------------------------------------------------------------------------
# Helper Functions - Counters
# ---------------------------------------------------------------------------


def record_reference_generated(member_state: str, mode: str = "single") -> None:
    """Record a reference number generation metric."""
    if _PROMETHEUS_AVAILABLE:
        _REFERENCES_GENERATED.labels(member_state=member_state, mode=mode).inc()


def record_reference_validated(result: str) -> None:
    """Record a reference number validation metric."""
    if _PROMETHEUS_AVAILABLE:
        _REFERENCES_VALIDATED.labels(result=result).inc()


def record_collision_detected(member_state: str) -> None:
    """Record a collision detection metric."""
    if _PROMETHEUS_AVAILABLE:
        _COLLISIONS_DETECTED.labels(member_state=member_state).inc()


def record_batch_completed(status: str) -> None:
    """Record a batch completion metric."""
    if _PROMETHEUS_AVAILABLE:
        _BATCHES_COMPLETED.labels(status=status).inc()


def record_reference_revoked(reason: str) -> None:
    """Record a reference revocation metric."""
    if _PROMETHEUS_AVAILABLE:
        _REFERENCES_REVOKED.labels(reason=reason).inc()


def record_reference_transferred(reason: str) -> None:
    """Record a reference transfer metric."""
    if _PROMETHEUS_AVAILABLE:
        _REFERENCES_TRANSFERRED.labels(reason=reason).inc()


def record_validation_failed(check_type: str) -> None:
    """Record a validation failure metric."""
    if _PROMETHEUS_AVAILABLE:
        _VALIDATIONS_FAILED.labels(check_type=check_type).inc()


def record_reference_expired(member_state: str) -> None:
    """Record a reference expiration metric."""
    if _PROMETHEUS_AVAILABLE:
        _REFERENCES_EXPIRED.labels(member_state=member_state).inc()


def record_sequence_overflow(strategy: str) -> None:
    """Record a sequence overflow metric."""
    if _PROMETHEUS_AVAILABLE:
        _SEQUENCE_OVERFLOWS.labels(strategy=strategy).inc()


def record_idempotent_hit() -> None:
    """Record an idempotent cache hit metric."""
    if _PROMETHEUS_AVAILABLE:
        _IDEMPOTENT_HITS.inc()


def record_lock_acquisition(outcome: str) -> None:
    """Record a distributed lock acquisition metric."""
    if _PROMETHEUS_AVAILABLE:
        _LOCK_ACQUISITIONS.labels(outcome=outcome).inc()


def record_api_error(operation: str) -> None:
    """Record an API error metric."""
    if _PROMETHEUS_AVAILABLE:
        _API_ERRORS.labels(operation=operation).inc()


# ---------------------------------------------------------------------------
# Helper Functions - Histograms
# ---------------------------------------------------------------------------


def observe_generation_duration(mode: str, duration: float) -> None:
    """Observe reference number generation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _GENERATION_DURATION.labels(mode=mode).observe(duration)


def observe_validation_duration(duration: float) -> None:
    """Observe reference number validation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _VALIDATION_DURATION.observe(duration)


def observe_batch_generation_duration(duration: float) -> None:
    """Observe batch generation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _BATCH_GENERATION_DURATION.observe(duration)


def observe_checksum_duration(algorithm: str, duration: float) -> None:
    """Observe checksum computation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _CHECKSUM_DURATION.labels(algorithm=algorithm).observe(duration)


def observe_collision_detection_duration(duration: float) -> None:
    """Observe collision detection duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _COLLISION_DETECTION_DURATION.observe(duration)


def observe_sequence_increment_duration(duration: float) -> None:
    """Observe sequence increment duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _SEQUENCE_INCREMENT_DURATION.observe(duration)


def observe_lock_acquisition_duration(duration: float) -> None:
    """Observe distributed lock acquisition duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _LOCK_ACQUISITION_DURATION.observe(duration)


def observe_lifecycle_transition_duration(transition: str, duration: float) -> None:
    """Observe lifecycle state transition duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _LIFECYCLE_TRANSITION_DURATION.labels(transition=transition).observe(duration)


def observe_verification_duration(duration: float) -> None:
    """Observe reference number verification duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _VERIFICATION_DURATION.observe(duration)


def observe_batch_size(size: int) -> None:
    """Observe batch request size."""
    if _PROMETHEUS_AVAILABLE:
        _BATCH_SIZE_DISTRIBUTION.observe(size)


# ---------------------------------------------------------------------------
# Helper Functions - Gauges
# ---------------------------------------------------------------------------


def set_active_references(count: int) -> None:
    """Set gauge of active reference numbers."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_REFERENCES.set(count)


def set_available_sequences(count: int) -> None:
    """Set gauge of available sequence slots."""
    if _PROMETHEUS_AVAILABLE:
        _AVAILABLE_SEQUENCES.set(count)


def set_sequence_utilization(operator: str, member_state: str, percent: float) -> None:
    """Set gauge of sequence utilization percentage."""
    if _PROMETHEUS_AVAILABLE:
        _SEQUENCE_UTILIZATION.labels(
            operator=operator, member_state=member_state
        ).set(percent)


def set_pending_batches(count: int) -> None:
    """Set gauge of pending batch requests."""
    if _PROMETHEUS_AVAILABLE:
        _PENDING_BATCHES.set(count)


def set_references_expiring_30d(count: int) -> None:
    """Set gauge of references expiring within 30 days."""
    if _PROMETHEUS_AVAILABLE:
        _REFERENCES_EXPIRING_30D.set(count)


def set_reserved_references(count: int) -> None:
    """Set gauge of reserved references."""
    if _PROMETHEUS_AVAILABLE:
        _RESERVED_REFERENCES.set(count)


def set_used_references(count: int) -> None:
    """Set gauge of used references."""
    if _PROMETHEUS_AVAILABLE:
        _USED_REFERENCES.set(count)


def set_revoked_references(count: int) -> None:
    """Set gauge of revoked references."""
    if _PROMETHEUS_AVAILABLE:
        _REVOKED_REFERENCES.set(count)


def set_expired_references(count: int) -> None:
    """Set gauge of expired references."""
    if _PROMETHEUS_AVAILABLE:
        _EXPIRED_REFERENCES.set(count)


def set_total_generated_lifetime(count: int) -> None:
    """Set gauge of total generated references since start."""
    if _PROMETHEUS_AVAILABLE:
        _TOTAL_GENERATED_LIFETIME.set(count)


def set_collisions_pending(count: int) -> None:
    """Set gauge of unresolved collisions."""
    if _PROMETHEUS_AVAILABLE:
        _COLLISIONS_PENDING.set(count)


def set_active_locks(count: int) -> None:
    """Set gauge of currently held distributed locks."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_LOCKS.set(count)


def set_bloom_filter_size(count: int) -> None:
    """Set gauge of bloom filter entries."""
    if _PROMETHEUS_AVAILABLE:
        _BLOOM_FILTER_SIZE.set(count)


def set_idempotency_cache_size(count: int) -> None:
    """Set gauge of idempotency cache entries."""
    if _PROMETHEUS_AVAILABLE:
        _IDEMPOTENCY_CACHE_SIZE.set(count)


def set_db_pool_active(count: int) -> None:
    """Set gauge of active DB pool connections."""
    if _PROMETHEUS_AVAILABLE:
        _DB_POOL_ACTIVE.set(count)


def set_db_pool_idle(count: int) -> None:
    """Set gauge of idle DB pool connections."""
    if _PROMETHEUS_AVAILABLE:
        _DB_POOL_IDLE.set(count)


def set_uptime_seconds(seconds: float) -> None:
    """Set gauge of service uptime."""
    if _PROMETHEUS_AVAILABLE:
        _UPTIME_SECONDS.set(seconds)


def set_last_generation_timestamp(epoch: float) -> None:
    """Set gauge of last generation timestamp."""
    if _PROMETHEUS_AVAILABLE:
        _LAST_GENERATION_TIMESTAMP.set(epoch)
