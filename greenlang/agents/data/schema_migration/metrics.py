# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-017: Schema Migration Agent

12 Prometheus metrics for schema migration service monitoring with
graceful fallback when prometheus_client is not installed.

Metrics:
    1.  gl_sm_schemas_registered_total       (Counter,   labels: schema_type, namespace)
    2.  gl_sm_versions_created_total         (Counter,   labels: bump_type)
    3.  gl_sm_changes_detected_total         (Counter,   labels: change_type, severity)
    4.  gl_sm_compatibility_checks_total     (Counter,   labels: result)
    5.  gl_sm_migrations_planned_total       (Counter,   labels: status)
    6.  gl_sm_migrations_executed_total      (Counter,   labels: status)
    7.  gl_sm_rollbacks_total                (Counter,   labels: rollback_type, status)
    8.  gl_sm_drift_detected_total           (Counter,   labels: drift_type, severity)
    9.  gl_sm_migration_duration_seconds     (Histogram, buckets: migration-scale)
    10. gl_sm_records_migrated               (Histogram, buckets: record-count-scale)
    11. gl_sm_processing_duration_seconds    (Histogram, labels: operation, buckets: sub-second)
    12. gl_sm_active_migrations              (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
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
        "prometheus_client not installed; schema migration metrics disabled"
    )

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Schemas registered by schema type and namespace
    sm_schemas_registered_total = Counter(
        "gl_sm_schemas_registered_total",
        "Total schemas registered in the migration registry",
        labelnames=["schema_type", "namespace"],
    )

    # 2. Schema versions created by semantic version bump type
    sm_versions_created_total = Counter(
        "gl_sm_versions_created_total",
        "Total schema versions created",
        labelnames=["bump_type"],
    )

    # 3. Schema changes detected by change type and severity
    sm_changes_detected_total = Counter(
        "gl_sm_changes_detected_total",
        "Total schema changes detected during comparison",
        labelnames=["change_type", "severity"],
    )

    # 4. Schema compatibility checks by result
    sm_compatibility_checks_total = Counter(
        "gl_sm_compatibility_checks_total",
        "Total schema compatibility checks performed",
        labelnames=["result"],
    )

    # 5. Migration plans created by planning status
    sm_migrations_planned_total = Counter(
        "gl_sm_migrations_planned_total",
        "Total migration plans created",
        labelnames=["status"],
    )

    # 6. Migrations executed by execution status
    sm_migrations_executed_total = Counter(
        "gl_sm_migrations_executed_total",
        "Total schema migrations executed",
        labelnames=["status"],
    )

    # 7. Rollbacks initiated by rollback type and outcome status
    sm_rollbacks_total = Counter(
        "gl_sm_rollbacks_total",
        "Total schema migration rollbacks initiated",
        labelnames=["rollback_type", "status"],
    )

    # 8. Schema drift events detected by drift type and severity
    sm_drift_detected_total = Counter(
        "gl_sm_drift_detected_total",
        "Total schema drift events detected",
        labelnames=["drift_type", "severity"],
    )

    # 9. End-to-end migration duration for full migration operations
    sm_migration_duration_seconds = Histogram(
        "gl_sm_migration_duration_seconds",
        "End-to-end schema migration duration in seconds",
        buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600),
    )

    # 10. Records migrated per migration execution
    sm_records_migrated = Histogram(
        "gl_sm_records_migrated",
        "Number of records migrated per migration execution",
        buckets=(10, 100, 1000, 5000, 10000, 50000, 100000),
    )

    # 11. Processing duration for individual engine operations
    sm_processing_duration_seconds = Histogram(
        "gl_sm_processing_duration_seconds",
        "Schema migration engine operation processing duration in seconds",
        labelnames=["operation"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10),
    )

    # 12. Active concurrent migrations gauge
    sm_active_migrations = Gauge(
        "gl_sm_active_migrations",
        "Number of schema migration operations currently in progress",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    sm_schemas_registered_total = None      # type: ignore[assignment]
    sm_versions_created_total = None        # type: ignore[assignment]
    sm_changes_detected_total = None        # type: ignore[assignment]
    sm_compatibility_checks_total = None    # type: ignore[assignment]
    sm_migrations_planned_total = None      # type: ignore[assignment]
    sm_migrations_executed_total = None     # type: ignore[assignment]
    sm_rollbacks_total = None               # type: ignore[assignment]
    sm_drift_detected_total = None          # type: ignore[assignment]
    sm_migration_duration_seconds = None    # type: ignore[assignment]
    sm_records_migrated = None              # type: ignore[assignment]
    sm_processing_duration_seconds = None   # type: ignore[assignment]
    sm_active_migrations = None             # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_schema_registered(schema_type: str, namespace: str) -> None:
    """Record a schema registration event.

    Args:
        schema_type: Schema type (json_schema, avro, protobuf, sql_ddl, custom).
        namespace: Logical namespace or service the schema belongs to
            (e.g. ``"emissions"``, ``"suppliers"``, ``"global"``).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sm_schemas_registered_total.labels(
        schema_type=schema_type,
        namespace=namespace,
    ).inc()


def record_version_created(bump_type: str) -> None:
    """Record a schema version creation event.

    Args:
        bump_type: Semantic version bump type (major, minor, patch).
            - ``"major"`` – backward-incompatible change.
            - ``"minor"`` – backward-compatible addition.
            - ``"patch"`` – backward-compatible fix.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sm_versions_created_total.labels(
        bump_type=bump_type,
    ).inc()


def record_change_detected(change_type: str, severity: str) -> None:
    """Record a schema change detection event.

    Args:
        change_type: Type of schema change detected
            (field_added, field_removed, field_renamed, type_changed,
            constraint_added, constraint_removed, index_added,
            index_removed, table_added, table_removed, column_modified).
        severity: Change severity
            (breaking, non_breaking, informational).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sm_changes_detected_total.labels(
        change_type=change_type,
        severity=severity,
    ).inc()


def record_compatibility_check(result: str) -> None:
    """Record a schema compatibility check result.

    Args:
        result: Compatibility check outcome
            (compatible, incompatible, warning, error).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sm_compatibility_checks_total.labels(
        result=result,
    ).inc()


def record_migration_planned(status: str) -> None:
    """Record a migration plan creation event.

    Args:
        status: Planning outcome status
            (success, failed, skipped, partial).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sm_migrations_planned_total.labels(
        status=status,
    ).inc()


def record_migration_executed(status: str) -> None:
    """Record a migration execution event.

    Args:
        status: Execution outcome status
            (success, failed, partial, rolled_back, skipped).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sm_migrations_executed_total.labels(
        status=status,
    ).inc()


def record_rollback(rollback_type: str, status: str) -> None:
    """Record a migration rollback event.

    Args:
        rollback_type: Type of rollback initiated
            (automatic, manual, emergency, scheduled).
        status: Rollback outcome status
            (success, failed, partial).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sm_rollbacks_total.labels(
        rollback_type=rollback_type,
        status=status,
    ).inc()


def record_drift_detected(drift_type: str, severity: str) -> None:
    """Record a schema drift detection event.

    Args:
        drift_type: Category of drift detected
            (structural, data_type, constraint, index, default_value,
            nullability, enum_value, foreign_key).
        severity: Drift severity level
            (critical, high, medium, low, informational).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sm_drift_detected_total.labels(
        drift_type=drift_type,
        severity=severity,
    ).inc()


def observe_migration_duration(seconds: float) -> None:
    """Record the end-to-end duration of a complete migration operation.

    Args:
        seconds: Total migration wall-clock time in seconds.
            Buckets: 1, 5, 10, 30, 60, 300, 600, 1800, 3600.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sm_migration_duration_seconds.observe(seconds)


def observe_records_migrated(count: int) -> None:
    """Record the number of records transformed in a migration execution.

    Args:
        count: Total records migrated.
            Buckets: 10, 100, 1000, 5000, 10000, 50000, 100000.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sm_records_migrated.observe(float(count))


def observe_processing_duration(operation: str, seconds: float) -> None:
    """Record processing duration for a single engine operation.

    Args:
        operation: Operation type label
            (schema_register, version_create, change_detect,
            compatibility_check, plan_create, migration_execute,
            rollback, drift_detect, export, validate).
        seconds: Operation duration in seconds.
            Buckets: 0.01, 0.05, 0.1, 0.5, 1, 5, 10.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sm_processing_duration_seconds.labels(
        operation=operation,
    ).observe(seconds)


def set_active_migrations(count: int) -> None:
    """Set the gauge for currently active migration operations.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        count: Number of migrations currently in progress.
            Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sm_active_migrations.set(count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "sm_schemas_registered_total",
    "sm_versions_created_total",
    "sm_changes_detected_total",
    "sm_compatibility_checks_total",
    "sm_migrations_planned_total",
    "sm_migrations_executed_total",
    "sm_rollbacks_total",
    "sm_drift_detected_total",
    "sm_migration_duration_seconds",
    "sm_records_migrated",
    "sm_processing_duration_seconds",
    "sm_active_migrations",
    # Helper functions
    "record_schema_registered",
    "record_version_created",
    "record_change_detected",
    "record_compatibility_check",
    "record_migration_planned",
    "record_migration_executed",
    "record_rollback",
    "record_drift_detected",
    "observe_migration_duration",
    "observe_records_migrated",
    "observe_processing_duration",
    "set_active_migrations",
]
