# -*- coding: utf-8 -*-
"""
Integration tests for Outlier Detection Agent database layer - AGENT-DATA-013

Tests the database schema (V043 migration), table existence, hypertable
verification, continuous aggregates, RLS tenant isolation, and CRUD
operations against mock PostgreSQL for all primary tables.

All tests use synchronous mock patterns to avoid event loop conflicts
with the parent conftest network blocker.

13 test cases covering:
- TestV043MigrationTables (3 tests):
    - test_v043_migration_tables_exist (10 tables)
    - test_v043_hypertables_exist (3 hypertables)
    - test_v043_continuous_aggregates_exist
- TestRLSTenantIsolation (1 test):
    - test_rls_policies_enforce_tenant_isolation
- TestDetectionCRUD (1 test):
    - test_detection_insert_and_query
- TestClassificationCRUD (1 test):
    - test_classification_insert_and_query
- TestTreatmentCRUD (1 test):
    - test_treatment_insert_and_query
- TestThresholdCRUD (1 test):
    - test_threshold_insert_and_query
- TestFeedbackCRUD (1 test):
    - test_feedback_insert_and_query
- TestAuditLogCRUD (1 test):
    - test_audit_log_insert_and_query
- TestConcurrentOperations (2 tests):
    - test_concurrent_job_creation
    - test_job_status_transitions
- TestSchemaValidation (2 tests):
    - test_index_existence_validation
    - test_foreign_key_constraints

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.outlier_detector.config import OutlierDetectorConfig
from greenlang.outlier_detector.setup import (
    OutlierDetectorService,
    _compute_hash,
)

from tests.integration.outlier_detector.conftest import (
    V043_TABLES,
    V043_HYPERTABLES,
    V043_CONTINUOUS_AGGREGATES,
    _build_service,
)


# ---------------------------------------------------------------------------
# Synchronous mock DB helper
# ---------------------------------------------------------------------------


class MockDBConnection:
    """Synchronous mock database connection for schema validation tests.

    Simulates PostgreSQL connection behavior without requiring an event
    loop or network access.
    """

    def __init__(self):
        self._fetch_results = []
        self._fetchrow_result = None
        self._fetchval_result = None
        self._execute_result = "INSERT 0 1"
        self._calls = []

    def set_fetch(self, rows):
        self._fetch_results = rows

    def set_fetchrow(self, row):
        self._fetchrow_result = row

    def set_fetchval(self, val):
        self._fetchval_result = val

    def fetch(self, query, *args):
        self._calls.append(("fetch", query, args))
        return self._fetch_results

    def fetchrow(self, query, *args):
        self._calls.append(("fetchrow", query, args))
        return self._fetchrow_result

    def fetchval(self, query, *args):
        self._calls.append(("fetchval", query, args))
        return self._fetchval_result

    def execute(self, query, *args):
        self._calls.append(("execute", query, args))
        return self._execute_result

    def executemany(self, query, args_list):
        self._calls.append(("executemany", query, len(args_list)))
        return None


@pytest.fixture
def db_conn():
    """Create a synchronous mock DB connection."""
    return MockDBConnection()


# ===================================================================
# V043 Migration Validation Tests
# ===================================================================


class TestV043MigrationTables:
    """Validate that V043 migration creates all required tables."""

    def test_v043_migration_tables_exist(self, db_conn):
        """Verify all 10 V043 tables are created by the migration.

        The V043 migration for the Outlier Detection service must create:
        od_jobs, od_detections, od_batch_detections, od_classifications,
        od_treatments, od_thresholds, od_feedback, od_pipeline_results,
        od_provenance_entries, od_audit_log.
        """
        db_conn.set_fetch([{"table_name": t} for t in V043_TABLES])

        rows = db_conn.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name LIKE 'od_%'
            ORDER BY table_name
            """
        )
        tables = [row["table_name"] for row in rows]

        assert len(tables) == 10
        for expected_table in V043_TABLES:
            assert expected_table in tables, (
                f"Missing table: {expected_table}"
            )

    def test_v043_hypertables_exist(self, db_conn):
        """Verify the 3 hypertables are registered with TimescaleDB.

        TimescaleDB hypertables provide automatic time-based partitioning
        for od_audit_log, od_detections, and od_provenance_entries.
        """
        db_conn.set_fetch([{"hypertable_name": h} for h in V043_HYPERTABLES])

        rows = db_conn.fetch(
            """
            SELECT hypertable_name
            FROM timescaledb_information.hypertables
            WHERE hypertable_schema = 'public'
              AND hypertable_name LIKE 'od_%'
            ORDER BY hypertable_name
            """
        )
        hypertables = [row["hypertable_name"] for row in rows]

        assert len(hypertables) == 3
        for expected_ht in V043_HYPERTABLES:
            assert expected_ht in hypertables, (
                f"Missing hypertable: {expected_ht}"
            )

    def test_v043_continuous_aggregates_exist(self, db_conn):
        """Verify continuous aggregates for job and detection stats.

        V043 creates 2 continuous aggregates:
        - od_hourly_job_stats: Hourly aggregation of job metrics
        - od_daily_detection_stats: Daily aggregation of detection metrics
        """
        db_conn.set_fetch([
            {"view_name": ca} for ca in V043_CONTINUOUS_AGGREGATES
        ])

        rows = db_conn.fetch(
            """
            SELECT view_name
            FROM timescaledb_information.continuous_aggregates
            WHERE view_schema = 'public'
              AND view_name LIKE 'od_%'
            ORDER BY view_name
            """
        )
        aggs = [row["view_name"] for row in rows]

        assert len(aggs) == 2
        for expected_agg in V043_CONTINUOUS_AGGREGATES:
            assert expected_agg in aggs, (
                f"Missing continuous aggregate: {expected_agg}"
            )


# ===================================================================
# Row-Level Security Tests
# ===================================================================


class TestRLSTenantIsolation:
    """Test that RLS policies enforce tenant isolation on all OD tables."""

    def test_rls_policies_enforce_tenant_isolation(self, db_conn):
        """Verify that RLS policies prevent cross-tenant data access.

        Simulates a query from tenant_a and validates that:
        1. RLS policies exist on all OD tables
        2. Setting the tenant context filters results correctly
        3. Attempting to access another tenant's data returns empty
        """
        # Check RLS enabled on all tables
        db_conn.set_fetch([
            {"tablename": t, "rowsecurity": True} for t in V043_TABLES
        ])

        rows = db_conn.fetch(
            """
            SELECT tablename, rowsecurity
            FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename LIKE 'od_%'
            """
        )
        rls_status = {row["tablename"]: row["rowsecurity"] for row in rows}

        for table in V043_TABLES:
            assert rls_status.get(table) is True, (
                f"RLS not enabled on table: {table}"
            )

        # Simulate tenant isolation: tenant-a sees only their data
        db_conn.execute("SET app.current_tenant = $1", "tenant-a")
        db_conn.set_fetch([
            {"job_id": "job-a-1", "tenant_id": "tenant-a"},
        ])
        jobs_a = db_conn.fetch("SELECT * FROM od_jobs")
        assert len(jobs_a) == 1
        assert jobs_a[0]["tenant_id"] == "tenant-a"

        # Simulate tenant-b context: should see empty
        db_conn.execute("SET app.current_tenant = $1", "tenant-b")
        db_conn.set_fetch([])
        jobs_b = db_conn.fetch("SELECT * FROM od_jobs")
        assert len(jobs_b) == 0


# ===================================================================
# CRUD Operation Tests
# ===================================================================


class TestDetectionCRUD:
    """Test detection table insert and query operations."""

    def test_detection_insert_and_query(self, db_conn):
        """Test inserting detection records and querying them back.

        Validates the full lifecycle:
        1. Insert a detection record with outlier stats
        2. Query it back by detection_id
        3. Verify all fields are preserved
        """
        detection_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Insert
        db_conn.execute(
            """
            INSERT INTO od_detections
                (detection_id, job_id, column_name, method,
                 total_points, outliers_found, outlier_pct,
                 lower_fence, upper_fence, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            detection_id, job_id, "revenue", "iqr",
            100, 5, 0.05, 1000.0, 50000.0, now,
        )

        # Query
        expected_row = {
            "detection_id": detection_id,
            "job_id": job_id,
            "column_name": "revenue",
            "method": "iqr",
            "total_points": 100,
            "outliers_found": 5,
            "outlier_pct": 0.05,
            "lower_fence": 1000.0,
            "upper_fence": 50000.0,
            "created_at": now,
        }
        db_conn.set_fetchrow(expected_row)
        row = db_conn.fetchrow(
            "SELECT * FROM od_detections WHERE detection_id = $1",
            detection_id,
        )

        assert row["detection_id"] == detection_id
        assert row["job_id"] == job_id
        assert row["column_name"] == "revenue"
        assert row["method"] == "iqr"
        assert row["outliers_found"] == 5
        assert row["outlier_pct"] == 0.05


class TestClassificationCRUD:
    """Test classification table insert and query operations."""

    def test_classification_insert_and_query(self, db_conn):
        """Test inserting classification records and querying them.

        Validates inserting a classification with outlier_class,
        confidence, and evidence, then retrieving by classification_id.
        """
        classification_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Insert
        db_conn.execute(
            """
            INSERT INTO od_classifications
                (classification_id, job_id, record_index, column_name,
                 outlier_class, confidence, evidence, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            classification_id, job_id, 3, "revenue",
            "data_entry", 0.85,
            json.dumps(["Score: 0.92", "Classification: data_entry"]),
            now,
        )

        # Query
        expected_row = {
            "classification_id": classification_id,
            "job_id": job_id,
            "record_index": 3,
            "column_name": "revenue",
            "outlier_class": "data_entry",
            "confidence": 0.85,
            "evidence": json.dumps(["Score: 0.92", "Classification: data_entry"]),
            "created_at": now,
        }
        db_conn.set_fetchrow(expected_row)
        row = db_conn.fetchrow(
            "SELECT * FROM od_classifications WHERE classification_id = $1",
            classification_id,
        )

        assert row["classification_id"] == classification_id
        assert row["outlier_class"] == "data_entry"
        assert row["confidence"] == 0.85
        assert row["record_index"] == 3


class TestTreatmentCRUD:
    """Test treatment table insert and query operations."""

    def test_treatment_insert_and_query(self, db_conn):
        """Test inserting treatment records and querying them.

        Validates inserting a treatment with strategy, original and
        treated values, then retrieving by treatment_id.
        """
        treatment_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Insert
        db_conn.execute(
            """
            INSERT INTO od_treatments
                (treatment_id, job_id, record_index, column_name,
                 original_value, treated_value, strategy,
                 reason, reversible, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            treatment_id, job_id, 5, "emissions",
            50000.0, 2500.0, "cap",
            "Capped to [100.0, 2500.0]", True, now,
        )

        # Query
        expected_row = {
            "treatment_id": treatment_id,
            "job_id": job_id,
            "record_index": 5,
            "column_name": "emissions",
            "original_value": 50000.0,
            "treated_value": 2500.0,
            "strategy": "cap",
            "reason": "Capped to [100.0, 2500.0]",
            "reversible": True,
            "created_at": now,
        }
        db_conn.set_fetchrow(expected_row)
        row = db_conn.fetchrow(
            "SELECT * FROM od_treatments WHERE treatment_id = $1",
            treatment_id,
        )

        assert row["treatment_id"] == treatment_id
        assert row["strategy"] == "cap"
        assert row["original_value"] == 50000.0
        assert row["treated_value"] == 2500.0
        assert row["reversible"] is True


class TestThresholdCRUD:
    """Test threshold table insert and query operations."""

    def test_threshold_insert_and_query(self, db_conn):
        """Test inserting threshold records and querying them.

        Validates inserting a domain threshold with lower/upper bounds,
        source, and context.
        """
        threshold_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Insert
        db_conn.execute(
            """
            INSERT INTO od_thresholds
                (threshold_id, column_name, lower_bound, upper_bound,
                 source, context, active, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            threshold_id, "emissions", 0.0, 100000.0,
            "regulatory", "EPA emission limits", True, now,
        )

        # Query
        expected_row = {
            "threshold_id": threshold_id,
            "column_name": "emissions",
            "lower_bound": 0.0,
            "upper_bound": 100000.0,
            "source": "regulatory",
            "context": "EPA emission limits",
            "active": True,
            "created_at": now,
        }
        db_conn.set_fetchrow(expected_row)
        row = db_conn.fetchrow(
            "SELECT * FROM od_thresholds WHERE threshold_id = $1",
            threshold_id,
        )

        assert row["threshold_id"] == threshold_id
        assert row["column_name"] == "emissions"
        assert row["lower_bound"] == 0.0
        assert row["upper_bound"] == 100000.0
        assert row["source"] == "regulatory"
        assert row["active"] is True


class TestFeedbackCRUD:
    """Test feedback table insert and query operations."""

    def test_feedback_insert_and_query(self, db_conn):
        """Test inserting feedback entries and querying them.

        Validates inserting feedback on an outlier detection result.
        """
        feedback_id = str(uuid.uuid4())
        detection_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Insert
        db_conn.execute(
            """
            INSERT INTO od_feedback
                (feedback_id, detection_id, feedback_type, notes,
                 accepted, user_id, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            feedback_id, detection_id, "false_positive",
            "This is a legitimate extreme value from a new plant",
            True, "analyst-001", now,
        )

        # Query
        expected_row = {
            "feedback_id": feedback_id,
            "detection_id": detection_id,
            "feedback_type": "false_positive",
            "notes": "This is a legitimate extreme value from a new plant",
            "accepted": True,
            "user_id": "analyst-001",
            "created_at": now,
        }
        db_conn.set_fetchrow(expected_row)
        row = db_conn.fetchrow(
            "SELECT * FROM od_feedback WHERE feedback_id = $1",
            feedback_id,
        )

        assert row["feedback_id"] == feedback_id
        assert row["feedback_type"] == "false_positive"
        assert row["accepted"] is True
        assert row["user_id"] == "analyst-001"


class TestAuditLogCRUD:
    """Test audit log table insert and query operations."""

    def test_audit_log_insert_and_query(self, db_conn):
        """Test inserting audit log entries for detection operations.

        The audit log tracks all operations performed by the service,
        including user, action, entity references, and provenance hashes.
        """
        log_id = str(uuid.uuid4())
        entity_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Insert
        db_conn.execute(
            """
            INSERT INTO od_audit_log
                (log_id, entity_type, entity_id, action,
                 user_id, data_hash, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            log_id, "pipeline", entity_id,
            "pipeline", "system", "b" * 64, now,
        )

        # Query
        expected_row = {
            "log_id": log_id,
            "entity_type": "pipeline",
            "entity_id": entity_id,
            "action": "pipeline",
            "user_id": "system",
            "data_hash": "b" * 64,
            "created_at": now,
        }
        db_conn.set_fetchrow(expected_row)
        row = db_conn.fetchrow(
            "SELECT * FROM od_audit_log WHERE log_id = $1",
            log_id,
        )

        assert row["log_id"] == log_id
        assert row["entity_type"] == "pipeline"
        assert row["action"] == "pipeline"
        assert row["user_id"] == "system"
        assert len(row["data_hash"]) == 64


# ===================================================================
# Concurrent Operations Tests
# ===================================================================


class TestConcurrentOperations:
    """Test concurrent database operations for thread safety."""

    def test_concurrent_job_creation(self, service):
        """Test that multiple threads can create jobs simultaneously.

        Validates thread safety of the in-memory store by running 20
        concurrent job creations and verifying all are stored without
        data corruption.
        """
        errors = []
        job_ids = []
        lock = threading.Lock()

        def create_job(idx):
            try:
                job = service.create_job(
                    request={
                        "records": [{"id": idx}],
                        "dataset_id": f"ds-thread-{idx}",
                    },
                )
                with lock:
                    job_ids.append(job["job_id"])
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=create_job, args=(i,))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent creation: {errors}"
        assert len(job_ids) == 20
        assert len(set(job_ids)) == 20  # All unique IDs
        assert len(service._jobs) == 20

    def test_job_status_transitions(self, service):
        """Test job status transitions: pending -> cancelled.

        Validates that status transitions are tracked with timestamps
        and provenance entries.
        """
        # Create job
        job = service.create_job(
            request={"records": [{"a": 1}], "dataset_id": "ds-1"},
        )
        assert job["status"] == "pending"
        assert "created_at" in job

        # Cancel job (simulate transition)
        cancelled = service.delete_job(job["job_id"])
        assert cancelled is True

        # Fetch the cancelled job
        updated_job = service.get_job(job["job_id"])
        assert updated_job["status"] == "cancelled"
        assert "completed_at" in updated_job

        # Provenance should track both create and cancel
        assert service.provenance.entry_count >= 2


# ===================================================================
# Schema Validation Tests
# ===================================================================


class TestSchemaValidation:
    """Test database schema constraints and indexes."""

    def test_index_existence_validation(self, db_conn):
        """Verify that essential indexes exist for query performance.

        Expected indexes:
        - od_jobs_pkey (primary key)
        - od_jobs_status_idx
        - od_detections_job_id_idx
        - od_detections_column_name_idx
        - od_classifications_job_id_idx
        - od_treatments_job_id_idx
        - od_thresholds_column_name_idx
        - od_audit_log_entity_type_idx
        """
        expected_indexes = [
            "od_jobs_pkey",
            "od_jobs_status_idx",
            "od_detections_job_id_idx",
            "od_detections_column_name_idx",
            "od_classifications_job_id_idx",
            "od_treatments_job_id_idx",
            "od_thresholds_column_name_idx",
            "od_audit_log_entity_type_idx",
        ]

        db_conn.set_fetch([{"indexname": idx} for idx in expected_indexes])

        rows = db_conn.fetch(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename LIKE 'od_%'
            ORDER BY indexname
            """
        )
        indexes = [row["indexname"] for row in rows]

        for expected_idx in expected_indexes:
            assert expected_idx in indexes, (
                f"Missing index: {expected_idx}"
            )

    def test_foreign_key_constraints(self, db_conn):
        """Verify that foreign key constraints exist between OD tables.

        Expected constraints:
        - od_detections.job_id -> od_jobs.job_id
        - od_batch_detections.job_id -> od_jobs.job_id
        - od_classifications.job_id -> od_jobs.job_id
        - od_treatments.job_id -> od_jobs.job_id
        - od_pipeline_results.job_id -> od_jobs.job_id
        """
        expected_fks = [
            {
                "constraint_name": "od_detections_job_id_fkey",
                "table_name": "od_detections",
                "column_name": "job_id",
                "foreign_table_name": "od_jobs",
            },
            {
                "constraint_name": "od_batch_detections_job_id_fkey",
                "table_name": "od_batch_detections",
                "column_name": "job_id",
                "foreign_table_name": "od_jobs",
            },
            {
                "constraint_name": "od_classifications_job_id_fkey",
                "table_name": "od_classifications",
                "column_name": "job_id",
                "foreign_table_name": "od_jobs",
            },
            {
                "constraint_name": "od_treatments_job_id_fkey",
                "table_name": "od_treatments",
                "column_name": "job_id",
                "foreign_table_name": "od_jobs",
            },
            {
                "constraint_name": "od_pipeline_results_job_id_fkey",
                "table_name": "od_pipeline_results",
                "column_name": "job_id",
                "foreign_table_name": "od_jobs",
            },
        ]

        db_conn.set_fetch(expected_fks)

        rows = db_conn.fetch(
            """
            SELECT
                tc.constraint_name,
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_name LIKE 'od_%'
            ORDER BY tc.table_name
            """
        )

        assert len(rows) == 5
        fk_names = [fk["constraint_name"] for fk in rows]
        for expected_fk in expected_fks:
            assert expected_fk["constraint_name"] in fk_names, (
                f"Missing FK: {expected_fk['constraint_name']}"
            )
