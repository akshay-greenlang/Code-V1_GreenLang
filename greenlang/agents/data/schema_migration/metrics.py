# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-017: Schema Migration Agent

12 Prometheus metrics for schema migration service monitoring with
graceful fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_sm_operations_total (Counter, labels: type, tenant_id)
    2.  gl_sm_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_sm_validation_errors_total (Counter, labels: severity, type)
    4.  gl_sm_batch_jobs_total (Counter, labels: status)
    5.  gl_sm_active_jobs (Gauge)
    6.  gl_sm_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_sm_schemas_registered_total (Counter, labels: schema_type, namespace)
    8.  gl_sm_versions_created_total (Counter, labels: bump_type)
    9.  gl_sm_changes_detected_total (Counter, labels: change_type, severity)
    10. gl_sm_migration_duration_seconds (Histogram, buckets: migration-scale)
    11. gl_sm_records_migrated (Histogram, buckets: record-count-scale)
    12. gl_sm_active_migrations (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.metrics import (
    PROMETHEUS_AVAILABLE,
    MetricsFactory,
)

# ---------------------------------------------------------------------------
# Standard metrics (6 of 12) via factory
# ---------------------------------------------------------------------------

m = MetricsFactory(
    "gl_sm",
    "Schema Migration",
    duration_buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10),
)

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

sm_schemas_registered_total = m.create_custom_counter(
    "schemas_registered_total",
    "Total schemas registered in the migration registry",
    labelnames=["schema_type", "namespace"],
)

sm_versions_created_total = m.create_custom_counter(
    "versions_created_total",
    "Total schema versions created",
    labelnames=["bump_type"],
)

sm_changes_detected_total = m.create_custom_counter(
    "changes_detected_total",
    "Total schema changes detected during comparison",
    labelnames=["change_type", "severity"],
)

sm_compatibility_checks_total = m.create_custom_counter(
    "compatibility_checks_total",
    "Total schema compatibility checks performed",
    labelnames=["result"],
)

sm_migrations_planned_total = m.create_custom_counter(
    "migrations_planned_total",
    "Total migration plans created",
    labelnames=["status"],
)

sm_migrations_executed_total = m.create_custom_counter(
    "migrations_executed_total",
    "Total schema migrations executed",
    labelnames=["status"],
)

sm_rollbacks_total = m.create_custom_counter(
    "rollbacks_total",
    "Total schema migration rollbacks initiated",
    labelnames=["rollback_type", "status"],
)

sm_drift_detected_total = m.create_custom_counter(
    "drift_detected_total",
    "Total schema drift events detected",
    labelnames=["drift_type", "severity"],
)

sm_migration_duration_seconds = m.create_custom_histogram(
    "migration_duration_seconds",
    "End-to-end schema migration duration in seconds",
    buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600),
)

sm_records_migrated = m.create_custom_histogram(
    "records_migrated",
    "Number of records migrated per migration execution",
    buckets=(10, 100, 1000, 5000, 10000, 50000, 100000),
)

sm_processing_duration_seconds = m.create_custom_histogram(
    "processing_duration_seconds_detail",
    "Schema migration engine operation processing duration in seconds",
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10),
    labelnames=["operation"],
)

sm_active_migrations = m.create_custom_gauge(
    "active_migrations",
    "Number of schema migration operations currently in progress",
)


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
    m.safe_inc(
        sm_schemas_registered_total, 1,
        schema_type=schema_type, namespace=namespace,
    )


def record_version_created(bump_type: str) -> None:
    """Record a schema version creation event.

    Args:
        bump_type: Semantic version bump type (major, minor, patch).
    """
    m.safe_inc(sm_versions_created_total, 1, bump_type=bump_type)


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
    m.safe_inc(
        sm_changes_detected_total, 1,
        change_type=change_type, severity=severity,
    )


def record_compatibility_check(result: str) -> None:
    """Record a schema compatibility check result.

    Args:
        result: Compatibility check outcome
            (compatible, incompatible, warning, error).
    """
    m.safe_inc(sm_compatibility_checks_total, 1, result=result)


def record_migration_planned(status: str) -> None:
    """Record a migration plan creation event.

    Args:
        status: Planning outcome status
            (success, failed, skipped, partial).
    """
    m.safe_inc(sm_migrations_planned_total, 1, status=status)


def record_migration_executed(status: str) -> None:
    """Record a migration execution event.

    Args:
        status: Execution outcome status
            (success, failed, partial, rolled_back, skipped).
    """
    m.safe_inc(sm_migrations_executed_total, 1, status=status)


def record_rollback(rollback_type: str, status: str) -> None:
    """Record a migration rollback event.

    Args:
        rollback_type: Type of rollback initiated
            (automatic, manual, emergency, scheduled).
        status: Rollback outcome status
            (success, failed, partial).
    """
    m.safe_inc(sm_rollbacks_total, 1, rollback_type=rollback_type, status=status)


def record_drift_detected(drift_type: str, severity: str) -> None:
    """Record a schema drift detection event.

    Args:
        drift_type: Category of drift detected
            (structural, data_type, constraint, index, default_value,
            nullability, enum_value, foreign_key).
        severity: Drift severity level
            (critical, high, medium, low, informational).
    """
    m.safe_inc(sm_drift_detected_total, 1, drift_type=drift_type, severity=severity)


def observe_migration_duration(seconds: float) -> None:
    """Record the end-to-end duration of a complete migration operation.

    Args:
        seconds: Total migration wall-clock time in seconds.
    """
    m.safe_observe(sm_migration_duration_seconds, seconds)


def observe_records_migrated(count: int) -> None:
    """Record the number of records transformed in a migration execution.

    Args:
        count: Total records migrated.
    """
    m.safe_observe(sm_records_migrated, float(count))


def observe_processing_duration(operation: str, seconds: float) -> None:
    """Record processing duration for a single engine operation.

    Args:
        operation: Operation type label
            (schema_register, version_create, change_detect,
            compatibility_check, plan_create, migration_execute,
            rollback, drift_detect, export, validate).
        seconds: Operation duration in seconds.
    """
    m.safe_observe(sm_processing_duration_seconds, seconds, operation=operation)


def set_active_migrations(count: int) -> None:
    """Set the gauge for currently active migration operations.

    Args:
        count: Number of migrations currently in progress.
    """
    m.safe_set(sm_active_migrations, count)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
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
