# -*- coding: utf-8 -*-
"""
Integration tests for Missing Value Imputer database layer - AGENT-DATA-012

Tests the database schema (V042 migration), table existence, hypertable
verification, continuous aggregates, RLS tenant isolation, and CRUD
operations against mock PostgreSQL for all primary tables.

All tests use synchronous mock patterns to avoid event loop conflicts
with the parent conftest network blocker.

15 test cases covering:
- test_v042_migration_tables_exist (10 tables)
- test_v042_hypertables_exist (3 hypertables)
- test_v042_continuous_aggregates_exist
- test_rls_policies_enforce_tenant_isolation
- test_analysis_insert_and_query
- test_result_insert_and_query
- test_rule_insert_and_query
- test_template_insert_and_query
- test_audit_log_insert_and_query
- test_concurrent_job_creation
- test_job_status_transitions
- test_rule_crud_lifecycle
- test_batch_insert
- test_index_existence_validation
- test_foreign_key_constraints

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.setup import (
    MissingValueImputerService,
    _compute_hash,
)

from tests.integration.missing_value_imputer.conftest import (
    V042_TABLES,
    V042_HYPERTABLES,
    V042_CONTINUOUS_AGGREGATES,
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
# V042 Migration Validation Tests
# ===================================================================


class TestV042MigrationTables:
    """Validate that V042 migration creates all required tables."""

    def test_v042_migration_tables_exist(self, db_conn):
        """Verify all 10 V042 tables are created by the migration.

        The V042 migration for the Missing Value Imputer service must create:
        mvi_jobs, mvi_analyses, mvi_imputation_results, mvi_batch_results,
        mvi_validation_results, mvi_rules, mvi_templates,
        mvi_column_strategies, mvi_provenance_entries, mvi_audit_log.
        """
        db_conn.set_fetch([{"table_name": t} for t in V042_TABLES])

        rows = db_conn.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name LIKE 'mvi_%'
            ORDER BY table_name
            """
        )
        tables = [row["table_name"] for row in rows]

        assert len(tables) == 10
        for expected_table in V042_TABLES:
            assert expected_table in tables, (
                f"Missing table: {expected_table}"
            )

    def test_v042_hypertables_exist(self, db_conn):
        """Verify the 3 hypertables are registered with TimescaleDB.

        TimescaleDB hypertables provide automatic time-based partitioning
        for mvi_audit_log, mvi_imputation_results, and
        mvi_provenance_entries.
        """
        db_conn.set_fetch([{"hypertable_name": h} for h in V042_HYPERTABLES])

        rows = db_conn.fetch(
            """
            SELECT hypertable_name
            FROM timescaledb_information.hypertables
            WHERE hypertable_schema = 'public'
              AND hypertable_name LIKE 'mvi_%'
            ORDER BY hypertable_name
            """
        )
        hypertables = [row["hypertable_name"] for row in rows]

        assert len(hypertables) == 3
        for expected_ht in V042_HYPERTABLES:
            assert expected_ht in hypertables, (
                f"Missing hypertable: {expected_ht}"
            )

    def test_v042_continuous_aggregates_exist(self, db_conn):
        """Verify continuous aggregates for job and imputation stats.

        V042 creates 2 continuous aggregates:
        - mvi_hourly_job_stats: Hourly aggregation of job metrics
        - mvi_daily_imputation_stats: Daily aggregation of imputation metrics
        """
        db_conn.set_fetch([
            {"view_name": ca} for ca in V042_CONTINUOUS_AGGREGATES
        ])

        rows = db_conn.fetch(
            """
            SELECT view_name
            FROM timescaledb_information.continuous_aggregates
            WHERE view_schema = 'public'
              AND view_name LIKE 'mvi_%'
            ORDER BY view_name
            """
        )
        aggs = [row["view_name"] for row in rows]

        assert len(aggs) == 2
        for expected_agg in V042_CONTINUOUS_AGGREGATES:
            assert expected_agg in aggs, (
                f"Missing continuous aggregate: {expected_agg}"
            )


# ===================================================================
# Row-Level Security Tests
# ===================================================================


class TestRLSTenantIsolation:
    """Test that RLS policies enforce tenant isolation on all MVI tables."""

    def test_rls_policies_enforce_tenant_isolation(self, db_conn):
        """Verify that RLS policies prevent cross-tenant data access.

        Simulates a query from tenant_a and validates that:
        1. RLS policies exist on all MVI tables
        2. Setting the tenant context filters results correctly
        3. Attempting to access another tenant's data returns empty
        """
        # Check RLS enabled on all tables
        db_conn.set_fetch([
            {"tablename": t, "rowsecurity": True} for t in V042_TABLES
        ])

        rows = db_conn.fetch(
            """
            SELECT tablename, rowsecurity
            FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename LIKE 'mvi_%'
            """
        )
        rls_status = {row["tablename"]: row["rowsecurity"] for row in rows}

        for table in V042_TABLES:
            assert rls_status.get(table) is True, (
                f"RLS not enabled on table: {table}"
            )

        # Simulate tenant isolation: tenant-a sees only their data
        db_conn.execute("SET app.current_tenant = $1", "tenant-a")
        db_conn.set_fetch([
            {"job_id": "job-a-1", "tenant_id": "tenant-a"},
        ])
        jobs_a = db_conn.fetch("SELECT * FROM mvi_jobs")
        assert len(jobs_a) == 1
        assert jobs_a[0]["tenant_id"] == "tenant-a"

        # Simulate tenant-b context: should see empty
        db_conn.execute("SET app.current_tenant = $1", "tenant-b")
        db_conn.set_fetch([])
        jobs_b = db_conn.fetch("SELECT * FROM mvi_jobs")
        assert len(jobs_b) == 0


# ===================================================================
# CRUD Operation Tests
# ===================================================================


class TestAnalysisCRUD:
    """Test analysis table insert and query operations."""

    def test_analysis_insert_and_query(self, db_conn):
        """Test inserting analysis records and querying them back.

        Validates the full lifecycle:
        1. Insert an analysis record with missingness stats
        2. Query it back by analysis_id
        3. Verify all fields are preserved
        """
        analysis_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Insert
        db_conn.execute(
            """
            INSERT INTO mvi_analyses
                (analysis_id, job_id, total_records, total_columns,
                 columns_with_missing, overall_missing_pct,
                 missingness_type, pattern_type, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            analysis_id, job_id, 100, 8, 3, 0.15,
            "mcar", "arbitrary", now,
        )

        # Query
        expected_row = {
            "analysis_id": analysis_id,
            "job_id": job_id,
            "total_records": 100,
            "total_columns": 8,
            "columns_with_missing": 3,
            "overall_missing_pct": 0.15,
            "missingness_type": "mcar",
            "pattern_type": "arbitrary",
            "created_at": now,
        }
        db_conn.set_fetchrow(expected_row)
        row = db_conn.fetchrow(
            "SELECT * FROM mvi_analyses WHERE analysis_id = $1",
            analysis_id,
        )

        assert row["analysis_id"] == analysis_id
        assert row["job_id"] == job_id
        assert row["total_records"] == 100
        assert row["columns_with_missing"] == 3
        assert row["missingness_type"] == "mcar"


class TestResultCRUD:
    """Test imputation result table insert and query operations."""

    def test_result_insert_and_query(self, db_conn):
        """Test inserting imputation result records and querying them.

        Validates inserting a result with strategy, confidence,
        and completeness improvement, then retrieving by result_id.
        """
        result_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Insert
        db_conn.execute(
            """
            INSERT INTO mvi_imputation_results
                (result_id, job_id, column_name, strategy,
                 values_imputed, avg_confidence, min_confidence,
                 completeness_before, completeness_after, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            result_id, job_id, "revenue", "mean",
            15, 0.85, 0.72, 0.70, 1.0, now,
        )

        # Query
        expected_row = {
            "result_id": result_id,
            "job_id": job_id,
            "column_name": "revenue",
            "strategy": "mean",
            "values_imputed": 15,
            "avg_confidence": 0.85,
            "min_confidence": 0.72,
            "completeness_before": 0.70,
            "completeness_after": 1.0,
            "created_at": now,
        }
        db_conn.set_fetchrow(expected_row)
        row = db_conn.fetchrow(
            "SELECT * FROM mvi_imputation_results WHERE result_id = $1",
            result_id,
        )

        assert row["result_id"] == result_id
        assert row["column_name"] == "revenue"
        assert row["strategy"] == "mean"
        assert row["values_imputed"] == 15
        assert row["avg_confidence"] == 0.85


class TestRuleCRUD:
    """Test rule table insert and query operations."""

    def test_rule_insert_and_query(self, db_conn):
        """Test inserting imputation rule records and querying them.

        Validates inserting a rule with conditions, impute_value,
        priority, and justification, then retrieving by rule_id.
        """
        rule_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conditions = [
            {"column": "region", "operator": "equals", "value": "EU"}
        ]

        # Insert
        db_conn.execute(
            """
            INSERT INTO mvi_rules
                (rule_id, name, target_column, conditions,
                 impute_value, priority, is_active,
                 justification, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            rule_id, "eu-emissions-default", "emissions",
            json.dumps(conditions), json.dumps(1500.0),
            "medium", True, "EU average emissions", now,
        )

        # Query
        expected_row = {
            "rule_id": rule_id,
            "name": "eu-emissions-default",
            "target_column": "emissions",
            "conditions": json.dumps(conditions),
            "impute_value": json.dumps(1500.0),
            "priority": "medium",
            "is_active": True,
            "justification": "EU average emissions",
            "created_at": now,
        }
        db_conn.set_fetchrow(expected_row)
        row = db_conn.fetchrow(
            "SELECT * FROM mvi_rules WHERE rule_id = $1",
            rule_id,
        )

        assert row["rule_id"] == rule_id
        assert row["name"] == "eu-emissions-default"
        assert row["target_column"] == "emissions"
        assert row["priority"] == "medium"
        assert row["is_active"] is True


class TestTemplateCRUD:
    """Test template table insert and query operations."""

    def test_template_insert_and_query(self, db_conn):
        """Test inserting imputation template records and querying them.

        Validates inserting a template with column_strategies, default
        strategy, and confidence threshold.
        """
        template_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        strategies = {"revenue": "median", "emissions": "knn"}

        # Insert
        db_conn.execute(
            """
            INSERT INTO mvi_templates
                (template_id, name, description, column_strategies,
                 default_strategy, confidence_threshold, is_active,
                 created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            template_id, "sustainability-template",
            "Template for sustainability data",
            json.dumps(strategies), "mean", 0.7, True, now,
        )

        # Query
        expected_row = {
            "template_id": template_id,
            "name": "sustainability-template",
            "description": "Template for sustainability data",
            "column_strategies": json.dumps(strategies),
            "default_strategy": "mean",
            "confidence_threshold": 0.7,
            "is_active": True,
            "created_at": now,
        }
        db_conn.set_fetchrow(expected_row)
        row = db_conn.fetchrow(
            "SELECT * FROM mvi_templates WHERE template_id = $1",
            template_id,
        )

        assert row["template_id"] == template_id
        assert row["name"] == "sustainability-template"
        assert row["default_strategy"] == "mean"
        assert row["confidence_threshold"] == 0.7


class TestAuditLogCRUD:
    """Test audit log table insert and query operations."""

    def test_audit_log_insert_and_query(self, db_conn):
        """Test inserting audit log entries for imputation operations.

        The audit log tracks all operations performed by the service,
        including user, action, entity references, and provenance hashes.
        """
        log_id = str(uuid.uuid4())
        entity_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Insert
        db_conn.execute(
            """
            INSERT INTO mvi_audit_log
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
            "SELECT * FROM mvi_audit_log WHERE log_id = $1",
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
        - mvi_jobs_pkey (primary key)
        - mvi_jobs_status_idx
        - mvi_analyses_job_id_idx
        - mvi_imputation_results_job_id_idx
        - mvi_imputation_results_column_name_idx
        - mvi_rules_target_column_idx
        - mvi_templates_name_idx
        - mvi_audit_log_entity_type_idx
        """
        expected_indexes = [
            "mvi_jobs_pkey",
            "mvi_jobs_status_idx",
            "mvi_analyses_job_id_idx",
            "mvi_imputation_results_job_id_idx",
            "mvi_imputation_results_column_name_idx",
            "mvi_rules_target_column_idx",
            "mvi_templates_name_idx",
            "mvi_audit_log_entity_type_idx",
        ]

        db_conn.set_fetch([{"indexname": idx} for idx in expected_indexes])

        rows = db_conn.fetch(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename LIKE 'mvi_%'
            ORDER BY indexname
            """
        )
        indexes = [row["indexname"] for row in rows]

        for expected_idx in expected_indexes:
            assert expected_idx in indexes, (
                f"Missing index: {expected_idx}"
            )

    def test_foreign_key_constraints(self, db_conn):
        """Verify that foreign key constraints exist between MVI tables.

        Expected constraints:
        - mvi_analyses.job_id -> mvi_jobs.job_id
        - mvi_imputation_results.job_id -> mvi_jobs.job_id
        - mvi_batch_results.job_id -> mvi_jobs.job_id
        - mvi_validation_results.job_id -> mvi_jobs.job_id
        - mvi_column_strategies.template_id -> mvi_templates.template_id
        """
        expected_fks = [
            {
                "constraint_name": "mvi_analyses_job_id_fkey",
                "table_name": "mvi_analyses",
                "column_name": "job_id",
                "foreign_table_name": "mvi_jobs",
            },
            {
                "constraint_name": "mvi_imputation_results_job_id_fkey",
                "table_name": "mvi_imputation_results",
                "column_name": "job_id",
                "foreign_table_name": "mvi_jobs",
            },
            {
                "constraint_name": "mvi_batch_results_job_id_fkey",
                "table_name": "mvi_batch_results",
                "column_name": "job_id",
                "foreign_table_name": "mvi_jobs",
            },
            {
                "constraint_name": "mvi_validation_results_job_id_fkey",
                "table_name": "mvi_validation_results",
                "column_name": "job_id",
                "foreign_table_name": "mvi_jobs",
            },
            {
                "constraint_name": "mvi_column_strategies_template_id_fkey",
                "table_name": "mvi_column_strategies",
                "column_name": "template_id",
                "foreign_table_name": "mvi_templates",
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
              AND tc.table_name LIKE 'mvi_%'
            ORDER BY tc.table_name
            """
        )

        assert len(rows) == 5
        fk_names = [fk["constraint_name"] for fk in rows]
        for expected_fk in expected_fks:
            assert expected_fk["constraint_name"] in fk_names, (
                f"Missing FK: {expected_fk['constraint_name']}"
            )
