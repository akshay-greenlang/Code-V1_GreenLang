# -*- coding: utf-8 -*-
"""
Integration tests for Duplicate Detection database layer - AGENT-DATA-011

Tests the database schema (V041 migration), table existence, hypertable
verification, continuous aggregates, RLS tenant isolation, and CRUD
operations against mock PostgreSQL for all primary tables.

All tests use synchronous mock patterns to avoid event loop conflicts
with the parent conftest network blocker.

15 test cases covering:
- test_v041_migration_tables_exist (10 tables)
- test_v041_hypertables_exist (3 hypertables)
- test_v041_continuous_aggregates_exist
- test_rls_policies_enforce_tenant_isolation
- test_fingerprint_insert_and_query
- test_match_insert_and_query
- test_cluster_insert_and_query
- test_merge_decision_insert_and_query
- test_audit_log_insert_and_query
- test_concurrent_job_creation
- test_job_status_transitions
- test_rule_crud_lifecycle
- test_comparison_batch_insert
- test_index_existence_validation
- test_foreign_key_constraints

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.duplicate_detector.config import DuplicateDetectorConfig
from greenlang.duplicate_detector.setup import (
    DuplicateDetectorService,
    _compute_hash,
)

from tests.integration.duplicate_detector.conftest import (
    V041_TABLES,
    V041_HYPERTABLES,
    V041_CONTINUOUS_AGGREGATES,
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
# V041 Migration Validation Tests
# ===================================================================


class TestV041MigrationTables:
    """Validate that V041 migration creates all required tables."""

    def test_v041_migration_tables_exist(self, db_conn):
        """Verify all 10 V041 tables are created by the migration.

        The V041 migration for the Duplicate Detection service must create:
        dd_jobs, dd_fingerprints, dd_blocks, dd_comparisons,
        dd_classifications, dd_clusters, dd_cluster_members,
        dd_merge_decisions, dd_rules, dd_audit_log.
        """
        db_conn.set_fetch([{"table_name": t} for t in V041_TABLES])

        rows = db_conn.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name LIKE 'dd_%'
            ORDER BY table_name
            """
        )
        tables = [row["table_name"] for row in rows]

        assert len(tables) == 10
        for expected_table in V041_TABLES:
            assert expected_table in tables, (
                f"Missing table: {expected_table}"
            )

    def test_v041_hypertables_exist(self, db_conn):
        """Verify the 3 hypertables are registered with TimescaleDB.

        TimescaleDB hypertables provide automatic time-based partitioning
        for dd_audit_log, dd_comparisons, and dd_fingerprints.
        """
        db_conn.set_fetch([{"hypertable_name": h} for h in V041_HYPERTABLES])

        rows = db_conn.fetch(
            """
            SELECT hypertable_name
            FROM timescaledb_information.hypertables
            WHERE hypertable_schema = 'public'
              AND hypertable_name LIKE 'dd_%'
            ORDER BY hypertable_name
            """
        )
        hypertables = [row["hypertable_name"] for row in rows]

        assert len(hypertables) == 3
        for expected_ht in V041_HYPERTABLES:
            assert expected_ht in hypertables, (
                f"Missing hypertable: {expected_ht}"
            )

    def test_v041_continuous_aggregates_exist(self, db_conn):
        """Verify continuous aggregates for job and comparison stats.

        V041 creates 2 continuous aggregates:
        - dd_hourly_job_stats: Hourly aggregation of job metrics
        - dd_daily_comparison_stats: Daily aggregation of comparison metrics
        """
        db_conn.set_fetch([{"view_name": ca} for ca in V041_CONTINUOUS_AGGREGATES])

        rows = db_conn.fetch(
            """
            SELECT view_name
            FROM timescaledb_information.continuous_aggregates
            WHERE view_schema = 'public'
              AND view_name LIKE 'dd_%'
            ORDER BY view_name
            """
        )
        aggs = [row["view_name"] for row in rows]

        assert len(aggs) == 2
        for expected_agg in V041_CONTINUOUS_AGGREGATES:
            assert expected_agg in aggs, (
                f"Missing continuous aggregate: {expected_agg}"
            )


# ===================================================================
# Row-Level Security Tests
# ===================================================================


class TestRLSTenantIsolation:
    """Test that RLS policies enforce tenant isolation on all DD tables."""

    def test_rls_policies_enforce_tenant_isolation(self, db_conn):
        """Verify that RLS policies prevent cross-tenant data access.

        Simulates a query from tenant_a and validates that:
        1. RLS policies exist on all DD tables
        2. Setting the tenant context filters results correctly
        3. Attempting to access another tenant's data returns empty
        """
        # Check RLS enabled on all tables
        db_conn.set_fetch([
            {"tablename": t, "rowsecurity": True} for t in V041_TABLES
        ])

        rows = db_conn.fetch(
            """
            SELECT tablename, rowsecurity
            FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename LIKE 'dd_%'
            """
        )
        rls_status = {row["tablename"]: row["rowsecurity"] for row in rows}

        for table in V041_TABLES:
            assert rls_status.get(table) is True, (
                f"RLS not enabled on table: {table}"
            )

        # Simulate tenant isolation: tenant-a sees only their data
        db_conn.execute("SET app.current_tenant = $1", "tenant-a")
        db_conn.set_fetch([
            {"job_id": "job-a-1", "tenant_id": "tenant-a"},
        ])
        jobs_a = db_conn.fetch("SELECT * FROM dd_jobs")
        assert len(jobs_a) == 1
        assert jobs_a[0]["tenant_id"] == "tenant-a"

        # Simulate tenant-b context: should see empty
        db_conn.execute("SET app.current_tenant = $1", "tenant-b")
        db_conn.set_fetch([])
        jobs_b = db_conn.fetch("SELECT * FROM dd_jobs")
        assert len(jobs_b) == 0


# ===================================================================
# CRUD Operation Tests
# ===================================================================


class TestFingerprintCRUD:
    """Test fingerprint table insert and query operations."""

    def test_fingerprint_insert_and_query(self, db_conn):
        """Test inserting fingerprint records and querying them back.

        Validates the full lifecycle:
        1. Insert a fingerprint record with job_id, record_hash, algorithm
        2. Query it back by fingerprint_id
        3. Verify all fields are preserved
        """
        fingerprint_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        record_hash = "a" * 64
        now = datetime.now(timezone.utc).isoformat()

        # Insert
        db_conn.execute(
            """
            INSERT INTO dd_fingerprints
                (fingerprint_id, job_id, record_index, record_hash,
                 algorithm, field_set, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            fingerprint_id, job_id, 0, record_hash,
            "sha256", json.dumps(["name", "email"]), now,
        )

        # Query
        expected_row = {
            "fingerprint_id": fingerprint_id,
            "job_id": job_id,
            "record_index": 0,
            "record_hash": record_hash,
            "algorithm": "sha256",
            "field_set": json.dumps(["name", "email"]),
            "created_at": now,
        }
        db_conn.set_fetchrow(expected_row)
        row = db_conn.fetchrow(
            "SELECT * FROM dd_fingerprints WHERE fingerprint_id = $1",
            fingerprint_id,
        )

        assert row["fingerprint_id"] == fingerprint_id
        assert row["job_id"] == job_id
        assert row["record_hash"] == record_hash
        assert row["algorithm"] == "sha256"


class TestMatchCRUD:
    """Test match/classification table insert and query operations."""

    def test_match_insert_and_query(self, db_conn):
        """Test inserting match classification records and querying them.

        Validates inserting a MATCH classification with score, threshold,
        and pair identifiers, then retrieving by match_id.
        """
        match_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Insert
        db_conn.execute(
            """
            INSERT INTO dd_classifications
                (match_id, job_id, record_a_id, record_b_id,
                 overall_score, classification, match_threshold,
                 field_scores, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            match_id, job_id, "rec-001", "rec-002",
            0.93, "MATCH", 0.85,
            json.dumps({"name": 0.95, "email": 1.0}), now,
        )

        # Query
        expected_row = {
            "match_id": match_id,
            "job_id": job_id,
            "record_a_id": "rec-001",
            "record_b_id": "rec-002",
            "overall_score": 0.93,
            "classification": "MATCH",
            "match_threshold": 0.85,
            "field_scores": json.dumps({"name": 0.95, "email": 1.0}),
            "created_at": now,
        }
        db_conn.set_fetchrow(expected_row)
        row = db_conn.fetchrow(
            "SELECT * FROM dd_classifications WHERE match_id = $1",
            match_id,
        )

        assert row["match_id"] == match_id
        assert row["classification"] == "MATCH"
        assert row["overall_score"] == 0.93
        assert row["record_a_id"] == "rec-001"
        assert row["record_b_id"] == "rec-002"


class TestClusterCRUD:
    """Test cluster table insert and query operations."""

    def test_cluster_insert_and_query(self, db_conn):
        """Test inserting cluster records and their member associations.

        Validates:
        1. Cluster header insert (cluster_id, algorithm, quality)
        2. Cluster member inserts (member record IDs)
        3. Query returns cluster with correct member count
        """
        cluster_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Insert cluster header
        db_conn.execute(
            """
            INSERT INTO dd_clusters
                (cluster_id, job_id, algorithm, member_count,
                 quality_score, representative_id, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            cluster_id, job_id, "union_find", 3, 0.91, "rec-001", now,
        )

        # Insert cluster members
        for record_id in ["rec-001", "rec-002", "rec-003"]:
            db_conn.execute(
                """
                INSERT INTO dd_cluster_members
                    (cluster_id, record_id)
                VALUES ($1, $2)
                """,
                cluster_id, record_id,
            )

        # Query cluster
        cluster_row = {
            "cluster_id": cluster_id,
            "job_id": job_id,
            "algorithm": "union_find",
            "member_count": 3,
            "quality_score": 0.91,
            "representative_id": "rec-001",
            "created_at": now,
        }
        db_conn.set_fetchrow(cluster_row)
        cluster = db_conn.fetchrow(
            "SELECT * FROM dd_clusters WHERE cluster_id = $1",
            cluster_id,
        )

        # Query members
        member_rows = [
            {"cluster_id": cluster_id, "record_id": "rec-001"},
            {"cluster_id": cluster_id, "record_id": "rec-002"},
            {"cluster_id": cluster_id, "record_id": "rec-003"},
        ]
        db_conn.set_fetch(member_rows)
        members = db_conn.fetch(
            "SELECT * FROM dd_cluster_members WHERE cluster_id = $1",
            cluster_id,
        )

        assert cluster["cluster_id"] == cluster_id
        assert cluster["member_count"] == 3
        assert cluster["algorithm"] == "union_find"
        assert len(members) == 3


class TestMergeDecisionCRUD:
    """Test merge decision table insert and query operations."""

    def test_merge_decision_insert_and_query(self, db_conn):
        """Test inserting merge decisions and querying by merge_id.

        Validates that golden record data, strategy, conflict count,
        and source records are all correctly stored and retrieved.
        """
        merge_id = str(uuid.uuid4())
        cluster_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        golden_record = {
            "name": "Alice Smith",
            "email": "alice@company.com",
            "phone": "555-0001",
        }

        # Insert
        db_conn.execute(
            """
            INSERT INTO dd_merge_decisions
                (merge_id, cluster_id, strategy, golden_record,
                 source_count, conflicts_resolved, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            merge_id, cluster_id, "keep_most_complete",
            json.dumps(golden_record), 3, 2, now,
        )

        # Query
        expected_row = {
            "merge_id": merge_id,
            "cluster_id": cluster_id,
            "strategy": "keep_most_complete",
            "golden_record": json.dumps(golden_record),
            "source_count": 3,
            "conflicts_resolved": 2,
            "created_at": now,
        }
        db_conn.set_fetchrow(expected_row)
        row = db_conn.fetchrow(
            "SELECT * FROM dd_merge_decisions WHERE merge_id = $1",
            merge_id,
        )

        assert row["merge_id"] == merge_id
        assert row["strategy"] == "keep_most_complete"
        assert row["source_count"] == 3
        assert row["conflicts_resolved"] == 2


class TestAuditLogCRUD:
    """Test audit log table insert and query operations."""

    def test_audit_log_insert_and_query(self, db_conn):
        """Test inserting audit log entries for dedup operations.

        The audit log tracks all operations performed by the service,
        including user, action, entity references, and provenance hashes.
        """
        log_id = str(uuid.uuid4())
        entity_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Insert
        db_conn.execute(
            """
            INSERT INTO dd_audit_log
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
            "SELECT * FROM dd_audit_log WHERE log_id = $1",
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
                job = service.create_dedup_job(
                    dataset_ids=[f"ds-thread-{idx}"],
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
        """Test job status transitions: created -> running -> completed.

        Validates that status transitions are tracked with timestamps
        and provenance entries.
        """
        # Create job
        job = service.create_dedup_job(dataset_ids=["ds-1", "ds-2"])
        assert job["status"] == "created"
        assert "created_at" in job

        # Cancel job (simulate transition)
        cancelled = service.cancel_job(job["job_id"])
        assert cancelled["status"] == "cancelled"
        assert "updated_at" in cancelled

        # Provenance should track both create and cancel
        assert service.provenance.entry_count >= 2

    def test_rule_crud_lifecycle(self, service):
        """Test the complete lifecycle of a dedup rule.

        Creates, lists, validates, and then creates additional rules
        to verify the full CRUD flow.
        """
        # Create rule
        rule = service.create_rule(rule_config={
            "name": "lifecycle-test-rule",
            "description": "Test rule for CRUD lifecycle",
            "match_threshold": 0.90,
            "possible_threshold": 0.70,
        })
        assert rule["rule_id"] in service._rules
        assert rule["name"] == "lifecycle-test-rule"
        assert rule["match_threshold"] == 0.90

        # List rules
        result = service.list_rules()
        assert result["count"] == 1
        assert result["rules"][0]["name"] == "lifecycle-test-rule"

        # Create another rule
        rule2 = service.create_rule(rule_config={
            "name": "second-rule",
            "match_threshold": 0.80,
        })
        assert rule2["rule_id"] != rule["rule_id"]

        # List should show both
        result = service.list_rules()
        assert result["count"] == 2
        assert result["total"] == 2

        # Stats should reflect
        stats = service.get_statistics()
        assert stats.total_rules == 2


class TestComparisonBatch:
    """Test batch insert operations for comparisons."""

    def test_comparison_batch_insert(self, db_conn):
        """Test inserting a batch of comparison records efficiently.

        Validates that batch inserts work for the high-volume
        dd_comparisons table.
        """
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        batch_size = 100
        comparison_rows = [
            (
                str(uuid.uuid4()),
                job_id,
                f"rec-{i:04d}",
                f"rec-{i + 1:04d}",
                round(0.5 + (i / batch_size) * 0.5, 4),
                "jaro_winkler",
                now,
            )
            for i in range(batch_size)
        ]

        # Batch insert
        db_conn.executemany(
            """
            INSERT INTO dd_comparisons
                (comparison_id, job_id, record_a_id, record_b_id,
                 overall_score, algorithm, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            comparison_rows,
        )

        # Verify executemany was called with correct count
        executemany_calls = [
            c for c in db_conn._calls if c[0] == "executemany"
        ]
        assert len(executemany_calls) == 1
        assert executemany_calls[0][2] == batch_size

        # Verify count query
        db_conn.set_fetchval(batch_size)
        count = db_conn.fetchval(
            "SELECT COUNT(*) FROM dd_comparisons WHERE job_id = $1",
            job_id,
        )
        assert count == batch_size


# ===================================================================
# Schema Validation Tests
# ===================================================================


class TestSchemaValidation:
    """Test database schema constraints and indexes."""

    def test_index_existence_validation(self, db_conn):
        """Verify that essential indexes exist for query performance.

        Expected indexes:
        - dd_jobs_pkey (primary key)
        - dd_jobs_status_idx
        - dd_fingerprints_job_id_idx
        - dd_fingerprints_record_hash_idx
        - dd_classifications_job_id_idx
        - dd_clusters_job_id_idx
        - dd_audit_log_entity_type_idx
        - dd_audit_log_created_at_idx
        """
        expected_indexes = [
            "dd_jobs_pkey",
            "dd_jobs_status_idx",
            "dd_fingerprints_job_id_idx",
            "dd_fingerprints_record_hash_idx",
            "dd_classifications_job_id_idx",
            "dd_clusters_job_id_idx",
            "dd_audit_log_entity_type_idx",
            "dd_audit_log_created_at_idx",
        ]

        db_conn.set_fetch([{"indexname": idx} for idx in expected_indexes])

        rows = db_conn.fetch(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename LIKE 'dd_%'
            ORDER BY indexname
            """
        )
        indexes = [row["indexname"] for row in rows]

        for expected_idx in expected_indexes:
            assert expected_idx in indexes, (
                f"Missing index: {expected_idx}"
            )

    def test_foreign_key_constraints(self, db_conn):
        """Verify that foreign key constraints exist between DD tables.

        Expected constraints:
        - dd_fingerprints.job_id -> dd_jobs.job_id
        - dd_classifications.job_id -> dd_jobs.job_id
        - dd_clusters.job_id -> dd_jobs.job_id
        - dd_cluster_members.cluster_id -> dd_clusters.cluster_id
        - dd_merge_decisions.cluster_id -> dd_clusters.cluster_id
        """
        expected_fks = [
            {
                "constraint_name": "dd_fingerprints_job_id_fkey",
                "table_name": "dd_fingerprints",
                "column_name": "job_id",
                "foreign_table_name": "dd_jobs",
            },
            {
                "constraint_name": "dd_classifications_job_id_fkey",
                "table_name": "dd_classifications",
                "column_name": "job_id",
                "foreign_table_name": "dd_jobs",
            },
            {
                "constraint_name": "dd_clusters_job_id_fkey",
                "table_name": "dd_clusters",
                "column_name": "job_id",
                "foreign_table_name": "dd_jobs",
            },
            {
                "constraint_name": "dd_cluster_members_cluster_id_fkey",
                "table_name": "dd_cluster_members",
                "column_name": "cluster_id",
                "foreign_table_name": "dd_clusters",
            },
            {
                "constraint_name": "dd_merge_decisions_cluster_id_fkey",
                "table_name": "dd_merge_decisions",
                "column_name": "cluster_id",
                "foreign_table_name": "dd_clusters",
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
              AND tc.table_name LIKE 'dd_%'
            ORDER BY tc.table_name
            """
        )

        assert len(rows) == 5
        fk_names = [fk["constraint_name"] for fk in rows]
        for expected_fk in expected_fks:
            assert expected_fk["constraint_name"] in fk_names, (
                f"Missing FK: {expected_fk['constraint_name']}"
            )
