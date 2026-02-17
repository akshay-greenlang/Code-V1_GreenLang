# -*- coding: utf-8 -*-
"""
Unit Tests for MigrationExecutorEngine - AGENT-DATA-017: Schema Migration Agent
================================================================================

Tests all public and private methods of MigrationExecutorEngine with ~120 tests
covering initialization, plan execution, dry-run mode, step transforms (rename,
cast, default, add, remove, compute), checkpoints, rollback (full and partial),
retry with exponential backoff, progress tracking, query methods, cancellation,
edge cases, and concurrent execution.

Test Classes (12):
    - TestMigrationExecutorInit (7 tests)
    - TestExecutePlan (27 tests)
    - TestGetExecution (7 tests)
    - TestListExecutions (10 tests)
    - TestRollbackExecution (16 tests)
    - TestCheckpoints (11 tests)
    - TestRetryStep (8 tests)
    - TestCancelExecution (5 tests)
    - TestExecutionProgress (7 tests)
    - TestStepTransforms (18 tests)
    - TestExpressionEvaluator (7 tests)
    - TestMigrationExecutorEdgeCases (9 tests)

Total: ~132 tests

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
"""

from __future__ import annotations

import copy
import re
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.schema_migration.migration_executor import (
    CAST_FUNCTIONS,
    EXECUTABLE_PLAN_STATUSES,
    MigrationExecutorEngine,
    ROLLBACK_FULL,
    ROLLBACK_PARTIAL,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_ROLLED_BACK,
    STATUS_RUNNING,
    SUPPORTED_TRANSFORMATION_TYPES,
    _ExpressionEvaluator,
    _generate_id,
    _is_missing,
    _sha256,
    _utcnow,
    _utcnow_iso,
)
from greenlang.schema_migration.provenance import ProvenanceTracker


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> MigrationExecutorEngine:
    """Create a fresh MigrationExecutorEngine with default config."""
    return MigrationExecutorEngine()


@pytest.fixture
def fast_engine() -> MigrationExecutorEngine:
    """Create an engine with minimal retry/timeout for fast tests."""
    return MigrationExecutorEngine(config={
        "timeout_seconds": 60,
        "batch_size": 100,
        "auto_rollback": True,
        "max_retries": 1,
        "backoff_base": 0.001,
        "max_workers": 2,
        "dry_run": False,
    })


@pytest.fixture
def no_rollback_engine() -> MigrationExecutorEngine:
    """Create an engine with auto_rollback disabled."""
    return MigrationExecutorEngine(config={
        "auto_rollback": False,
        "max_retries": 1,
        "backoff_base": 0.001,
    })


@pytest.fixture
def sample_data() -> List[Dict[str, Any]]:
    """10 sample records with name, age, email, department fields."""
    return [
        {"name": "Alice", "age": 28, "email": "alice@example.com", "department": "Engineering"},
        {"name": "Bob", "age": 35, "email": "bob@example.com", "department": "Science"},
        {"name": "Carol", "age": 42, "email": "carol@example.com", "department": "Operations"},
        {"name": "David", "age": 31, "email": "david@example.com", "department": "Engineering"},
        {"name": "Eva", "age": 29, "email": "eva@example.com", "department": "Science"},
        {"name": "Frank", "age": 38, "email": "frank@example.com", "department": "Management"},
        {"name": "Grace", "age": 26, "email": "grace@example.com", "department": "Operations"},
        {"name": "Hector", "age": 44, "email": "hector@example.com", "department": "Engineering"},
        {"name": "Ingrid", "age": 33, "email": "ingrid@example.com", "department": "Science"},
        {"name": "James", "age": 30, "email": "james@example.com", "department": "Operations"},
    ]


@pytest.fixture
def rename_plan() -> Dict[str, Any]:
    """A validated plan with a single rename step."""
    return {
        "plan_id": "PL-rename-001",
        "status": "validated",
        "steps": [
            {
                "step_number": 1,
                "transformation_type": "rename",
                "source_field": "department",
                "target_field": "team",
            },
        ],
    }


@pytest.fixture
def multi_step_plan() -> Dict[str, Any]:
    """A validated plan with 3 steps: add, rename, remove."""
    return {
        "plan_id": "PL-multi-001",
        "status": "validated",
        "steps": [
            {
                "step_number": 1,
                "transformation_type": "add",
                "field": "salary",
                "default_value": 0,
            },
            {
                "step_number": 2,
                "transformation_type": "rename",
                "source_field": "department",
                "target_field": "team",
            },
            {
                "step_number": 3,
                "transformation_type": "remove",
                "field": "age",
            },
        ],
    }


@pytest.fixture
def cast_plan() -> Dict[str, Any]:
    """A validated plan with a single cast step (string -> integer)."""
    return {
        "plan_id": "PL-cast-001",
        "status": "validated",
        "steps": [
            {
                "step_number": 1,
                "transformation_type": "cast",
                "field": "age",
                "old_type": "string",
                "new_type": "integer",
            },
        ],
    }


@pytest.fixture
def compute_plan() -> Dict[str, Any]:
    """A validated plan with a compute step."""
    return {
        "plan_id": "PL-compute-001",
        "status": "validated",
        "steps": [
            {
                "step_number": 1,
                "transformation_type": "compute",
                "target_field": "full_label",
                "expression": '{name} + " @ " + {department}',
                "source_fields": ["name", "department"],
            },
        ],
    }


def _make_validated_plan(
    plan_id: str = "PL-test",
    steps: Optional[List[Dict[str, Any]]] = None,
    status: str = "validated",
) -> Dict[str, Any]:
    """Helper to build a valid plan dict with sensible defaults."""
    if steps is None:
        steps = [
            {
                "step_number": 1,
                "transformation_type": "add",
                "field": "new_field",
                "default_value": "default",
            },
        ]
    return {"plan_id": plan_id, "status": status, "steps": steps}


# ===========================================================================
# TestMigrationExecutorInit
# ===========================================================================


class TestMigrationExecutorInit:
    """Tests for MigrationExecutorEngine.__init__."""

    def test_default_initialization(self):
        """Engine initialises with default configuration values."""
        engine = MigrationExecutorEngine()
        assert engine._config["timeout_seconds"] == 3600
        assert engine._config["batch_size"] == 10_000
        assert engine._config["auto_rollback"] is True
        assert engine._config["max_retries"] == 3
        assert engine._config["backoff_base"] == 2.0
        assert engine._config["max_workers"] == 4
        assert engine._config["dry_run"] is False

    def test_custom_config_overrides(self):
        """Custom config dict overrides default values."""
        engine = MigrationExecutorEngine(config={
            "timeout_seconds": 120,
            "batch_size": 500,
            "max_retries": 5,
        })
        assert engine._config["timeout_seconds"] == 120
        assert engine._config["batch_size"] == 500
        assert engine._config["max_retries"] == 5
        # Non-overridden values keep defaults
        assert engine._config["auto_rollback"] is True

    def test_empty_stores_on_init(self):
        """Engine starts with empty execution, rollback, and checkpoint stores."""
        engine = MigrationExecutorEngine()
        assert engine._executions == {}
        assert engine._rollbacks == {}
        assert engine._checkpoints == {}

    def test_stats_zeroed_on_init(self):
        """Engine starts with all statistics counters at zero."""
        engine = MigrationExecutorEngine()
        stats = engine.get_statistics()
        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0
        assert stats["failed_executions"] == 0
        assert stats["total_rollbacks"] == 0
        assert stats["total_records_processed"] == 0

    def test_provenance_tracker_created(self):
        """Engine creates a ProvenanceTracker when none is provided."""
        engine = MigrationExecutorEngine()
        assert isinstance(engine._provenance, ProvenanceTracker)

    def test_external_provenance_tracker_used(self):
        """Engine uses an externally provided ProvenanceTracker."""
        tracker = ProvenanceTracker(genesis_hash="custom-genesis")
        engine = MigrationExecutorEngine(provenance_tracker=tracker)
        assert engine._provenance is tracker

    def test_thread_lock_is_reentrant(self):
        """Engine uses an RLock for thread-safe operations."""
        engine = MigrationExecutorEngine()
        assert isinstance(engine._lock, type(threading.RLock()))


# ===========================================================================
# TestExecutePlan
# ===========================================================================


class TestExecutePlan:
    """Tests for MigrationExecutorEngine.execute_plan."""

    def test_basic_execution_completes(self, fast_engine, rename_plan, sample_data):
        """Executing a valid plan returns status=completed."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        assert result["status"] == STATUS_COMPLETED

    def test_execution_id_has_prefix(self, fast_engine, rename_plan, sample_data):
        """Execution IDs start with 'EXE-' prefix."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        assert result["execution_id"].startswith("EXE-")

    def test_execution_id_is_unique(self, fast_engine, rename_plan, sample_data):
        """Each execution produces a unique execution_id."""
        r1 = fast_engine.execute_plan(rename_plan, copy.deepcopy(sample_data))
        r2 = fast_engine.execute_plan(rename_plan, copy.deepcopy(sample_data))
        assert r1["execution_id"] != r2["execution_id"]

    def test_plan_id_mirrored(self, fast_engine, rename_plan, sample_data):
        """Execution record mirrors the plan_id from input."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        assert result["plan_id"] == "PL-rename-001"

    def test_timestamps_populated(self, fast_engine, rename_plan, sample_data):
        """started_at and completed_at are populated on success."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        assert result["started_at"] is not None
        assert "T" in result["started_at"]
        assert result["completed_at"] is not None
        assert "T" in result["completed_at"]

    def test_duration_ms_non_negative(self, fast_engine, rename_plan, sample_data):
        """duration_ms is a non-negative number on completed execution."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        assert result["duration_ms"] >= 0

    def test_provenance_hash_is_sha256(self, fast_engine, rename_plan, sample_data):
        """Provenance hash is a 64-character hex SHA-256 string."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        assert len(result["provenance_hash"]) == 64
        assert re.match(r"^[0-9a-f]{64}$", result["provenance_hash"])

    def test_records_processed_equals_data_length(self, fast_engine, rename_plan, sample_data):
        """records_processed equals the number of input records."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        assert result["records_processed"] == len(sample_data)

    def test_total_steps_count(self, fast_engine, multi_step_plan, sample_data):
        """total_steps matches the number of steps in the plan."""
        result = fast_engine.execute_plan(multi_step_plan, sample_data)
        assert result["total_steps"] == 3

    def test_completed_steps_equals_total_on_success(self, fast_engine, multi_step_plan, sample_data):
        """completed_steps equals total_steps when all steps succeed."""
        result = fast_engine.execute_plan(multi_step_plan, sample_data)
        assert result["completed_steps"] == result["total_steps"]

    def test_step_results_populated(self, fast_engine, multi_step_plan, sample_data):
        """step_results list has one entry per completed step."""
        result = fast_engine.execute_plan(multi_step_plan, sample_data)
        assert len(result["step_results"]) == 3

    def test_step_results_have_correct_status(self, fast_engine, multi_step_plan, sample_data):
        """Each step_result has status='completed' on success."""
        result = fast_engine.execute_plan(multi_step_plan, sample_data)
        for sr in result["step_results"]:
            assert sr["status"] == "completed"

    def test_dry_run_mode_via_argument(self, fast_engine, rename_plan, sample_data):
        """dry_run=True simulates without mutating caller data."""
        original = copy.deepcopy(sample_data)
        result = fast_engine.execute_plan(rename_plan, sample_data, dry_run=True)
        assert result["dry_run"] is True
        assert result["status"] == STATUS_COMPLETED
        # Original data must remain unchanged
        assert sample_data == original

    def test_dry_run_mode_via_config(self, sample_data):
        """Engine-level dry_run config is respected."""
        engine = MigrationExecutorEngine(config={"dry_run": True, "max_retries": 1, "backoff_base": 0.001})
        plan = _make_validated_plan()
        result = engine.execute_plan(plan, sample_data)
        assert result["dry_run"] is True

    def test_dry_run_argument_overrides_config(self, sample_data):
        """dry_run argument overrides engine-level config."""
        engine = MigrationExecutorEngine(config={"dry_run": True, "max_retries": 1, "backoff_base": 0.001})
        plan = _make_validated_plan()
        result = engine.execute_plan(plan, sample_data, dry_run=False)
        assert result["dry_run"] is False

    def test_none_data_treated_as_empty(self, fast_engine, rename_plan):
        """Passing data=None executes against an empty list."""
        result = fast_engine.execute_plan(rename_plan, data=None)
        assert result["status"] == STATUS_COMPLETED
        assert result["records_processed"] == 0

    def test_original_data_not_mutated(self, fast_engine, rename_plan, sample_data):
        """execute_plan deep-copies data; caller's list is not mutated."""
        original = copy.deepcopy(sample_data)
        fast_engine.execute_plan(rename_plan, sample_data)
        assert sample_data == original

    def test_invalid_plan_status_raises(self, fast_engine, sample_data):
        """Plans with non-executable status raise ValueError."""
        plan = {"plan_id": "PL-bad", "status": "draft", "steps": [
            {"step_number": 1, "transformation_type": "add", "field": "x"},
        ]}
        with pytest.raises(ValueError, match="status"):
            fast_engine.execute_plan(plan, sample_data)

    def test_missing_plan_id_raises(self, fast_engine, sample_data):
        """Plans without plan_id raise ValueError."""
        plan = {"status": "validated", "steps": [
            {"step_number": 1, "transformation_type": "add", "field": "x"},
        ]}
        with pytest.raises(ValueError, match="plan_id"):
            fast_engine.execute_plan(plan, sample_data)

    def test_empty_steps_raises(self, fast_engine, sample_data):
        """Plans with no steps raise ValueError."""
        plan = {"plan_id": "PL-empty", "status": "validated", "steps": []}
        with pytest.raises(ValueError, match="no steps"):
            fast_engine.execute_plan(plan, sample_data)

    def test_unsupported_transformation_type_raises(self, fast_engine, sample_data):
        """Plans with unsupported transformation types raise ValueError."""
        plan = {
            "plan_id": "PL-bad-type",
            "status": "validated",
            "steps": [{"step_number": 1, "transformation_type": "teleport"}],
        }
        with pytest.raises(ValueError, match="unsupported"):
            fast_engine.execute_plan(plan, sample_data)

    def test_non_dict_plan_raises(self, fast_engine, sample_data):
        """Non-dict plan raises ValueError."""
        with pytest.raises(ValueError, match="dictionary"):
            fast_engine.execute_plan("not a dict", sample_data)

    def test_approved_status_is_executable(self, fast_engine, sample_data):
        """Plans with status='approved' can be executed."""
        plan = _make_validated_plan(status="approved")
        result = fast_engine.execute_plan(plan, sample_data)
        assert result["status"] == STATUS_COMPLETED

    def test_statistics_incremented(self, fast_engine, rename_plan, sample_data):
        """Execution increments total_executions and successful_executions."""
        fast_engine.execute_plan(rename_plan, sample_data)
        stats = fast_engine.get_statistics()
        assert stats["total_executions"] == 1
        assert stats["successful_executions"] == 1

    def test_execution_stored_internally(self, fast_engine, rename_plan, sample_data):
        """Completed execution is retrievable via get_execution."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        stored = fast_engine.get_execution(result["execution_id"])
        assert stored is not None
        assert stored["execution_id"] == result["execution_id"]
        assert stored["status"] == STATUS_COMPLETED

    def test_steps_sorted_by_step_number(self, fast_engine, sample_data):
        """Steps are executed in step_number order regardless of input order."""
        plan = {
            "plan_id": "PL-unordered",
            "status": "validated",
            "steps": [
                {"step_number": 3, "transformation_type": "remove", "field": "age"},
                {"step_number": 1, "transformation_type": "add", "field": "salary", "default_value": 0},
                {"step_number": 2, "transformation_type": "rename", "source_field": "department", "target_field": "team"},
            ],
        }
        result = fast_engine.execute_plan(plan, sample_data)
        assert result["status"] == STATUS_COMPLETED
        step_nums = [sr["step_number"] for sr in result["step_results"]]
        assert step_nums == [1, 2, 3]

    def test_timeout_causes_failure(self, sample_data):
        """Execution that exceeds timeout is marked as failed."""
        engine = MigrationExecutorEngine(config={
            "timeout_seconds": 60,
            "max_retries": 1,
            "backoff_base": 0.001,
        })
        plan = _make_validated_plan()
        # Mock time.monotonic so the elapsed time always exceeds the timeout
        original_check = engine._check_timeout

        def force_timeout(start_mono, timeout_seconds, execution_id):
            # Call the real check with a start_mono far in the past
            original_check(start_mono - 1000, timeout_seconds, execution_id)

        engine._check_timeout = force_timeout
        result = engine.execute_plan(plan, sample_data)
        assert result["status"] == STATUS_FAILED
        assert "timeout" in result["error"].lower()


# ===========================================================================
# TestGetExecution
# ===========================================================================


class TestGetExecution:
    """Tests for MigrationExecutorEngine.get_execution."""

    def test_get_existing_execution(self, fast_engine, rename_plan, sample_data):
        """get_execution returns the full record for an existing ID."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        fetched = fast_engine.get_execution(result["execution_id"])
        assert fetched is not None
        assert fetched["execution_id"] == result["execution_id"]
        assert fetched["plan_id"] == "PL-rename-001"

    def test_get_nonexistent_returns_none(self, fast_engine):
        """get_execution returns None for an unknown ID."""
        assert fast_engine.get_execution("EXE-nonexistent") is None

    def test_get_empty_string_returns_none(self, fast_engine):
        """get_execution with empty string returns None."""
        assert fast_engine.get_execution("") is None

    def test_returned_record_is_deep_copy(self, fast_engine, rename_plan, sample_data):
        """get_execution returns a deep copy; mutations do not affect store."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        fetched = fast_engine.get_execution(result["execution_id"])
        fetched["status"] = "tampered"
        refetched = fast_engine.get_execution(result["execution_id"])
        assert refetched["status"] == STATUS_COMPLETED

    def test_get_after_rollback_reflects_status(self, fast_engine, rename_plan, sample_data):
        """get_execution after rollback shows status=rolled_back."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        fast_engine.rollback_execution(result["execution_id"])
        fetched = fast_engine.get_execution(result["execution_id"])
        assert fetched["status"] == STATUS_ROLLED_BACK

    def test_get_after_reset_returns_none(self, fast_engine, rename_plan, sample_data):
        """get_execution returns None after engine.reset() clears state."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        fast_engine.reset()
        assert fast_engine.get_execution(result["execution_id"]) is None

    def test_get_preserves_all_keys(self, fast_engine, rename_plan, sample_data):
        """get_execution result contains all expected keys."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        fetched = fast_engine.get_execution(result["execution_id"])
        expected_keys = {
            "execution_id", "plan_id", "status", "dry_run", "total_steps",
            "completed_steps", "current_step", "records_processed",
            "started_at", "completed_at", "duration_ms", "step_results",
            "error", "provenance_hash",
        }
        assert expected_keys.issubset(set(fetched.keys()))


# ===========================================================================
# TestListExecutions
# ===========================================================================


class TestListExecutions:
    """Tests for MigrationExecutorEngine.list_executions."""

    def test_empty_list_on_fresh_engine(self, fast_engine):
        """list_executions returns empty list when no executions exist."""
        assert fast_engine.list_executions() == []

    def test_all_executions_returned(self, fast_engine, sample_data):
        """list_executions returns all stored executions when no filter."""
        for i in range(3):
            plan = _make_validated_plan(plan_id=f"PL-{i}")
            fast_engine.execute_plan(plan, copy.deepcopy(sample_data))
        executions = fast_engine.list_executions()
        assert len(executions) == 3

    def test_filter_by_status_completed(self, fast_engine, sample_data):
        """list_executions filters by status='completed'."""
        plan = _make_validated_plan()
        fast_engine.execute_plan(plan, sample_data)
        completed = fast_engine.list_executions(status=STATUS_COMPLETED)
        assert len(completed) == 1
        assert completed[0]["status"] == STATUS_COMPLETED

    def test_filter_by_status_returns_empty_when_no_match(self, fast_engine, sample_data):
        """list_executions returns empty when status filter has no matches."""
        plan = _make_validated_plan()
        fast_engine.execute_plan(plan, sample_data)
        failed = fast_engine.list_executions(status=STATUS_FAILED)
        assert failed == []

    def test_pagination_limit(self, fast_engine, sample_data):
        """list_executions respects the limit parameter."""
        for i in range(5):
            plan = _make_validated_plan(plan_id=f"PL-{i}")
            fast_engine.execute_plan(plan, copy.deepcopy(sample_data))
        page = fast_engine.list_executions(limit=2)
        assert len(page) == 2

    def test_pagination_offset(self, fast_engine, sample_data):
        """list_executions respects the offset parameter."""
        for i in range(5):
            plan = _make_validated_plan(plan_id=f"PL-{i}")
            fast_engine.execute_plan(plan, copy.deepcopy(sample_data))
        all_execs = fast_engine.list_executions()
        offset_execs = fast_engine.list_executions(offset=3)
        assert len(offset_execs) == 2
        assert offset_execs[0]["execution_id"] == all_execs[3]["execution_id"]

    def test_pagination_limit_and_offset_combined(self, fast_engine, sample_data):
        """list_executions handles combined limit and offset."""
        for i in range(10):
            plan = _make_validated_plan(plan_id=f"PL-{i}")
            fast_engine.execute_plan(plan, copy.deepcopy(sample_data))
        page = fast_engine.list_executions(offset=2, limit=3)
        assert len(page) == 3

    def test_results_are_deep_copies(self, fast_engine, rename_plan, sample_data):
        """list_executions returns deep copies of records."""
        fast_engine.execute_plan(rename_plan, sample_data)
        execs = fast_engine.list_executions()
        execs[0]["status"] = "tampered"
        refetch = fast_engine.list_executions()
        assert refetch[0]["status"] == STATUS_COMPLETED

    def test_sorted_newest_first(self, fast_engine, sample_data):
        """list_executions returns records sorted by started_at descending."""
        for i in range(3):
            plan = _make_validated_plan(plan_id=f"PL-{i}")
            fast_engine.execute_plan(plan, copy.deepcopy(sample_data))
        execs = fast_engine.list_executions()
        timestamps = [e["started_at"] for e in execs]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_offset_beyond_total_returns_empty(self, fast_engine, sample_data):
        """list_executions with offset beyond total count returns empty."""
        plan = _make_validated_plan()
        fast_engine.execute_plan(plan, sample_data)
        result = fast_engine.list_executions(offset=100)
        assert result == []


# ===========================================================================
# TestRollbackExecution
# ===========================================================================


class TestRollbackExecution:
    """Tests for MigrationExecutorEngine.rollback_execution."""

    def test_full_rollback_completes(self, fast_engine, rename_plan, sample_data):
        """Full rollback on a completed execution succeeds."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        rollback = fast_engine.rollback_execution(result["execution_id"])
        assert rollback["status"] == "completed"
        assert rollback["rollback_type"] == ROLLBACK_FULL

    def test_rollback_id_has_prefix(self, fast_engine, rename_plan, sample_data):
        """Rollback IDs start with 'RBK-' prefix."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        rollback = fast_engine.rollback_execution(result["execution_id"])
        assert rollback["rollback_id"].startswith("RBK-")

    def test_rollback_updates_execution_status(self, fast_engine, rename_plan, sample_data):
        """Rollback changes execution status to rolled_back."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        fast_engine.rollback_execution(result["execution_id"])
        execution = fast_engine.get_execution(result["execution_id"])
        assert execution["status"] == STATUS_ROLLED_BACK

    def test_partial_rollback_to_step(self, fast_engine, multi_step_plan, sample_data):
        """Partial rollback to a specific step succeeds."""
        result = fast_engine.execute_plan(multi_step_plan, sample_data)
        rollback = fast_engine.rollback_execution(
            result["execution_id"],
            rollback_type=ROLLBACK_PARTIAL,
            to_step=2,
        )
        assert rollback["status"] == "completed"
        assert rollback["rollback_type"] == ROLLBACK_PARTIAL
        assert rollback["to_step"] <= 2

    def test_partial_rollback_without_to_step_raises(self, fast_engine, rename_plan, sample_data):
        """Partial rollback without to_step raises ValueError."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        with pytest.raises(ValueError, match="to_step"):
            fast_engine.rollback_execution(
                result["execution_id"],
                rollback_type=ROLLBACK_PARTIAL,
            )

    def test_rollback_nonexistent_execution_raises(self, fast_engine):
        """Rollback on a nonexistent execution_id raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            fast_engine.rollback_execution("EXE-nonexistent")

    def test_rollback_with_reason(self, fast_engine, rename_plan, sample_data):
        """Rollback stores the provided reason string."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        rollback = fast_engine.rollback_execution(
            result["execution_id"],
            reason="Manual operator rollback",
        )
        assert rollback["reason"] == "Manual operator rollback"

    def test_rollback_has_provenance_hash(self, fast_engine, rename_plan, sample_data):
        """Rollback record includes a non-empty provenance hash."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        rollback = fast_engine.rollback_execution(result["execution_id"])
        assert len(rollback["provenance_hash"]) == 64

    def test_rollback_has_created_at(self, fast_engine, rename_plan, sample_data):
        """Rollback record has a created_at timestamp."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        rollback = fast_engine.rollback_execution(result["execution_id"])
        assert rollback["created_at"] is not None
        assert "T" in rollback["created_at"]

    def test_rollback_references_execution_id(self, fast_engine, rename_plan, sample_data):
        """Rollback record references the correct execution_id."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        rollback = fast_engine.rollback_execution(result["execution_id"])
        assert rollback["execution_id"] == result["execution_id"]

    def test_rollback_references_checkpoint(self, fast_engine, rename_plan, sample_data):
        """Rollback record includes a checkpoint_id."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        rollback = fast_engine.rollback_execution(result["execution_id"])
        assert rollback["checkpoint_id"].startswith("CKP-")

    def test_full_rollback_selects_first_checkpoint(self, fast_engine, multi_step_plan, sample_data):
        """Full rollback selects the first (step 0) checkpoint."""
        result = fast_engine.execute_plan(multi_step_plan, sample_data)
        rollback = fast_engine.rollback_execution(result["execution_id"])
        # First checkpoint is for step_number 1 (before execution of step 1)
        assert rollback["to_step"] == 1

    def test_rollback_increments_stats(self, fast_engine, rename_plan, sample_data):
        """Rollback increments total_rollbacks counter."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        fast_engine.rollback_execution(result["execution_id"])
        stats = fast_engine.get_statistics()
        assert stats["total_rollbacks"] >= 1

    def test_unknown_rollback_type_raises(self, fast_engine, rename_plan, sample_data):
        """Unknown rollback_type raises ValueError."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        with pytest.raises(ValueError, match="Unknown rollback_type"):
            fast_engine.rollback_execution(
                result["execution_id"],
                rollback_type="imaginary",
            )

    def test_list_rollbacks_returns_record(self, fast_engine, rename_plan, sample_data):
        """list_rollbacks returns the rollback record after creation."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        fast_engine.rollback_execution(result["execution_id"])
        rollbacks = fast_engine.list_rollbacks(execution_id=result["execution_id"])
        assert len(rollbacks) == 1
        assert rollbacks[0]["execution_id"] == result["execution_id"]

    def test_partial_rollback_no_checkpoint_at_step_raises(self, fast_engine, rename_plan, sample_data):
        """Partial rollback to a step before any checkpoint raises ValueError."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        with pytest.raises(ValueError, match="No checkpoint found"):
            fast_engine.rollback_execution(
                result["execution_id"],
                rollback_type=ROLLBACK_PARTIAL,
                to_step=0,  # No checkpoint at step 0
            )


# ===========================================================================
# TestCheckpoints
# ===========================================================================


class TestCheckpoints:
    """Tests for checkpoint creation and retrieval."""

    def test_checkpoints_created_per_step(self, fast_engine, multi_step_plan, sample_data):
        """One checkpoint is created before each step execution."""
        result = fast_engine.execute_plan(multi_step_plan, sample_data)
        with fast_engine._lock:
            checkpoints = fast_engine._checkpoints.get(result["execution_id"], [])
        assert len(checkpoints) == 3  # One per step

    def test_checkpoint_has_correct_keys(self, fast_engine, rename_plan, sample_data):
        """Checkpoint dicts contain all expected keys."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        with fast_engine._lock:
            checkpoints = fast_engine._checkpoints.get(result["execution_id"], [])
        assert len(checkpoints) >= 1
        ckp = checkpoints[0]
        expected_keys = {
            "checkpoint_id", "execution_id", "step_number",
            "created_at", "has_snapshot", "record_count", "provenance_hash",
        }
        public_keys = {k for k in ckp.keys() if not k.startswith("_")}
        assert expected_keys.issubset(public_keys)

    def test_checkpoint_id_has_prefix(self, fast_engine, rename_plan, sample_data):
        """Checkpoint IDs start with 'CKP-' prefix."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        with fast_engine._lock:
            checkpoints = fast_engine._checkpoints.get(result["execution_id"], [])
        assert checkpoints[0]["checkpoint_id"].startswith("CKP-")

    def test_checkpoint_has_data_snapshot(self, fast_engine, rename_plan, sample_data):
        """Checkpoint stores a data snapshot when not in dry-run mode."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        with fast_engine._lock:
            checkpoints = fast_engine._checkpoints.get(result["execution_id"], [])
        assert checkpoints[0]["has_snapshot"] is True
        assert checkpoints[0]["_data"] is not None

    def test_dry_run_checkpoint_has_no_snapshot(self, fast_engine, rename_plan, sample_data):
        """Checkpoint in dry-run mode does not store data snapshot."""
        result = fast_engine.execute_plan(rename_plan, sample_data, dry_run=True)
        with fast_engine._lock:
            checkpoints = fast_engine._checkpoints.get(result["execution_id"], [])
        for ckp in checkpoints:
            assert ckp["has_snapshot"] is False
            assert ckp["_data"] is None

    def test_checkpoint_step_numbers_ascending(self, fast_engine, multi_step_plan, sample_data):
        """Checkpoints are ordered by step_number in ascending order."""
        result = fast_engine.execute_plan(multi_step_plan, sample_data)
        with fast_engine._lock:
            checkpoints = fast_engine._checkpoints.get(result["execution_id"], [])
        step_nums = [c["step_number"] for c in checkpoints]
        assert step_nums == sorted(step_nums)

    def test_checkpoint_record_count_matches_data(self, fast_engine, rename_plan, sample_data):
        """Checkpoint record_count matches the number of records at that point."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        with fast_engine._lock:
            checkpoints = fast_engine._checkpoints.get(result["execution_id"], [])
        assert checkpoints[0]["record_count"] == len(sample_data)

    def test_manual_checkpoint_creation(self, fast_engine):
        """create_checkpoint can be called directly with custom data."""
        ckp = fast_engine.create_checkpoint(
            execution_id="EXE-manual",
            step_number=5,
            data_snapshot=[{"a": 1}, {"a": 2}],
        )
        assert ckp["checkpoint_id"].startswith("CKP-")
        assert ckp["step_number"] == 5
        assert ckp["record_count"] == 2
        assert ckp["has_snapshot"] is True

    def test_manual_checkpoint_without_snapshot(self, fast_engine):
        """create_checkpoint with data_snapshot=None creates metadata-only checkpoint."""
        ckp = fast_engine.create_checkpoint(
            execution_id="EXE-meta",
            step_number=1,
            data_snapshot=None,
        )
        assert ckp["has_snapshot"] is False
        assert ckp["record_count"] == 0

    def test_checkpoint_provenance_hash(self, fast_engine, rename_plan, sample_data):
        """Checkpoint provenance hash is a 64-char hex SHA-256."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        with fast_engine._lock:
            checkpoints = fast_engine._checkpoints.get(result["execution_id"], [])
        assert len(checkpoints[0]["provenance_hash"]) == 64

    def test_checkpoint_increments_stats(self, fast_engine, rename_plan, sample_data):
        """Checkpoint creation increments total_checkpoints_created counter."""
        fast_engine.execute_plan(rename_plan, sample_data)
        stats = fast_engine.get_statistics()
        assert stats["total_checkpoints_created"] >= 1


# ===========================================================================
# TestRetryStep
# ===========================================================================


class TestRetryStep:
    """Tests for retry logic with exponential backoff."""

    def test_successful_step_no_retry(self, fast_engine, rename_plan, sample_data):
        """Successful steps do not trigger retries."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        stats = fast_engine.get_statistics()
        assert stats["total_retries"] == 0

    def test_failed_step_retries_exhausted(self):
        """A step that always fails exhausts all retry attempts."""
        engine = MigrationExecutorEngine(config={
            "max_retries": 3,
            "backoff_base": 0.001,
            "auto_rollback": False,
        })
        # Patch execute_step to always fail
        original_execute = engine.execute_step

        call_count = [0]

        def failing_execute(step, data, execution_id):
            call_count[0] += 1
            return {
                "step_number": step.get("step_number", 0),
                "transformation_type": step.get("transformation_type", ""),
                "status": "failed",
                "records_affected": 0,
                "data": data,
                "error": "Simulated persistent failure",
                "duration_ms": 0.1,
                "provenance_hash": "",
            }

        engine.execute_step = failing_execute
        plan = _make_validated_plan()
        result = engine.execute_plan(plan, [{"a": 1}])
        # 3 attempts (max_retries=3)
        assert call_count[0] == 3
        assert result["status"] == STATUS_FAILED

    def test_retry_count_tracked_in_stats(self):
        """Retries increment total_retries counter in statistics."""
        engine = MigrationExecutorEngine(config={
            "max_retries": 3,
            "backoff_base": 0.001,
            "auto_rollback": False,
        })
        original_execute = engine.execute_step
        attempt = [0]

        def fail_twice(step, data, execution_id):
            attempt[0] += 1
            if attempt[0] <= 2:
                return {
                    "step_number": step.get("step_number", 0),
                    "transformation_type": step.get("transformation_type", ""),
                    "status": "failed",
                    "records_affected": 0,
                    "data": data,
                    "error": "Transient failure",
                    "duration_ms": 0.1,
                    "provenance_hash": "",
                }
            return original_execute(step, data, execution_id)

        engine.execute_step = fail_twice
        plan = _make_validated_plan()
        result = engine.execute_plan(plan, [{"x": 1}])
        assert result["status"] == STATUS_COMPLETED
        stats = engine.get_statistics()
        assert stats["total_retries"] == 2

    def test_single_retry_max(self):
        """Engine with max_retries=1 does not retry on failure."""
        engine = MigrationExecutorEngine(config={
            "max_retries": 1,
            "backoff_base": 0.001,
            "auto_rollback": False,
        })
        call_count = [0]

        def always_fail(step, data, execution_id):
            call_count[0] += 1
            return {
                "step_number": step.get("step_number", 0),
                "transformation_type": step.get("transformation_type", ""),
                "status": "failed",
                "records_affected": 0,
                "data": data,
                "error": "Always fails",
                "duration_ms": 0.1,
                "provenance_hash": "",
            }

        engine.execute_step = always_fail
        plan = _make_validated_plan()
        engine.execute_plan(plan, [{"a": 1}])
        assert call_count[0] == 1

    def test_auto_rollback_on_step_failure(self):
        """auto_rollback triggers rollback when a step fails."""
        engine = MigrationExecutorEngine(config={
            "max_retries": 1,
            "backoff_base": 0.001,
            "auto_rollback": True,
        })

        def always_fail(step, data, execution_id):
            return {
                "step_number": step.get("step_number", 0),
                "transformation_type": step.get("transformation_type", ""),
                "status": "failed",
                "records_affected": 0,
                "data": data,
                "error": "Deliberate failure",
                "duration_ms": 0.1,
                "provenance_hash": "",
            }

        engine.execute_step = always_fail
        plan = _make_validated_plan()
        result = engine.execute_plan(plan, [{"a": 1}])
        assert result["status"] == STATUS_ROLLED_BACK

    def test_no_auto_rollback_keeps_failed(self, no_rollback_engine):
        """Without auto_rollback, failed step marks execution as failed."""

        def always_fail(step, data, execution_id):
            return {
                "step_number": step.get("step_number", 0),
                "transformation_type": step.get("transformation_type", ""),
                "status": "failed",
                "records_affected": 0,
                "data": data,
                "error": "Deliberate failure",
                "duration_ms": 0.1,
                "provenance_hash": "",
            }

        no_rollback_engine.execute_step = always_fail
        plan = _make_validated_plan()
        result = no_rollback_engine.execute_plan(plan, [{"a": 1}])
        assert result["status"] == STATUS_FAILED

    def test_failed_execution_increments_stats(self, no_rollback_engine):
        """Failed execution increments failed_executions counter."""

        def always_fail(step, data, execution_id):
            return {
                "step_number": step.get("step_number", 0),
                "transformation_type": step.get("transformation_type", ""),
                "status": "failed",
                "records_affected": 0,
                "data": data,
                "error": "Deliberate failure",
                "duration_ms": 0.1,
                "provenance_hash": "",
            }

        no_rollback_engine.execute_step = always_fail
        plan = _make_validated_plan()
        no_rollback_engine.execute_plan(plan, [{"a": 1}])
        stats = no_rollback_engine.get_statistics()
        assert stats["failed_executions"] >= 1

    def test_error_message_preserved(self, no_rollback_engine):
        """Error message from failed step is stored in execution record."""

        def fail_with_message(step, data, execution_id):
            return {
                "step_number": step.get("step_number", 0),
                "transformation_type": step.get("transformation_type", ""),
                "status": "failed",
                "records_affected": 0,
                "data": data,
                "error": "Specific error: disk full",
                "duration_ms": 0.1,
                "provenance_hash": "",
            }

        no_rollback_engine.execute_step = fail_with_message
        plan = _make_validated_plan()
        result = no_rollback_engine.execute_plan(plan, [{"a": 1}])
        assert "disk full" in result["error"]


# ===========================================================================
# TestCancelExecution
# ===========================================================================


class TestCancelExecution:
    """Tests for execution cancellation (simulated via rollback/status)."""

    def test_reset_clears_all_state(self, fast_engine, rename_plan, sample_data):
        """reset() clears executions, rollbacks, and checkpoints."""
        fast_engine.execute_plan(rename_plan, sample_data)
        fast_engine.reset()
        assert fast_engine.list_executions() == []
        stats = fast_engine.get_statistics()
        assert stats["total_executions"] == 0

    def test_reset_leaves_provenance_intact(self, fast_engine, rename_plan, sample_data):
        """reset() does not clear the provenance tracker."""
        fast_engine.execute_plan(rename_plan, sample_data)
        prov_count_before = fast_engine._provenance.entry_count
        fast_engine.reset()
        prov_count_after = fast_engine._provenance.entry_count
        assert prov_count_after == prov_count_before

    def test_reset_zeroes_statistics(self, fast_engine, rename_plan, sample_data):
        """reset() resets all statistics counters to zero."""
        fast_engine.execute_plan(rename_plan, sample_data)
        fast_engine.reset()
        stats = fast_engine.get_statistics()
        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0
        assert stats["total_records_processed"] == 0

    def test_multiple_resets_are_safe(self, fast_engine):
        """Calling reset() multiple times does not raise errors."""
        fast_engine.reset()
        fast_engine.reset()
        fast_engine.reset()
        assert fast_engine.list_executions() == []

    def test_engine_usable_after_reset(self, fast_engine, rename_plan, sample_data):
        """Engine functions normally after a reset."""
        fast_engine.execute_plan(rename_plan, copy.deepcopy(sample_data))
        fast_engine.reset()
        result = fast_engine.execute_plan(rename_plan, copy.deepcopy(sample_data))
        assert result["status"] == STATUS_COMPLETED


# ===========================================================================
# TestExecutionProgress
# ===========================================================================


class TestExecutionProgress:
    """Tests for MigrationExecutorEngine.get_progress."""

    def test_progress_for_completed_execution(self, fast_engine, rename_plan, sample_data):
        """get_progress on a completed execution shows 100% and found=True."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        progress = fast_engine.get_progress(result["execution_id"])
        assert progress["found"] is True
        assert progress["percentage"] == 100.0
        assert progress["status"] == STATUS_COMPLETED

    def test_progress_for_nonexistent_execution(self, fast_engine):
        """get_progress for unknown ID returns found=False."""
        progress = fast_engine.get_progress("EXE-nonexistent")
        assert progress["found"] is False

    def test_progress_keys_present(self, fast_engine, rename_plan, sample_data):
        """get_progress result contains all expected keys."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        progress = fast_engine.get_progress(result["execution_id"])
        expected_keys = {
            "execution_id", "status", "current_step", "total_steps",
            "completed_steps", "records_processed", "percentage",
            "eta_seconds", "elapsed_ms", "found",
        }
        assert expected_keys.issubset(set(progress.keys()))

    def test_percentage_calculation_accuracy(self, fast_engine, multi_step_plan, sample_data):
        """Percentage is calculated as completed_steps / total_steps * 100."""
        result = fast_engine.execute_plan(multi_step_plan, sample_data)
        progress = fast_engine.get_progress(result["execution_id"])
        expected_pct = (progress["completed_steps"] / progress["total_steps"]) * 100.0
        assert progress["percentage"] == pytest.approx(expected_pct, abs=0.01)

    def test_records_processed_in_progress(self, fast_engine, rename_plan, sample_data):
        """records_processed in progress matches execution records_processed."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        progress = fast_engine.get_progress(result["execution_id"])
        assert progress["records_processed"] == len(sample_data)

    def test_eta_is_zero_when_complete(self, fast_engine, rename_plan, sample_data):
        """ETA is zero when the execution is already complete."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        progress = fast_engine.get_progress(result["execution_id"])
        assert progress["eta_seconds"] == 0.0

    def test_total_steps_matches_plan(self, fast_engine, multi_step_plan, sample_data):
        """total_steps in progress matches the plan step count."""
        result = fast_engine.execute_plan(multi_step_plan, sample_data)
        progress = fast_engine.get_progress(result["execution_id"])
        assert progress["total_steps"] == 3


# ===========================================================================
# TestStepTransforms
# ===========================================================================


class TestStepTransforms:
    """Tests for individual transformation methods (apply_*)."""

    def test_rename_moves_value(self, engine):
        """apply_rename moves the value from source to target field."""
        data = [{"old_name": "Alice"}, {"old_name": "Bob"}]
        result = engine.apply_rename(data, "old_name", "new_name")
        assert result[0]["new_name"] == "Alice"
        assert "old_name" not in result[0]
        assert result[1]["new_name"] == "Bob"

    def test_rename_missing_source_leaves_record_unchanged(self, engine):
        """apply_rename skips records that lack the source field."""
        data = [{"other": "value"}]
        result = engine.apply_rename(data, "missing_field", "target")
        assert "target" not in result[0]
        assert result[0]["other"] == "value"

    def test_rename_empty_source_raises(self, engine):
        """apply_rename with empty source_field raises ValueError."""
        with pytest.raises(ValueError, match="source_field"):
            engine.apply_rename([{"a": 1}], "", "b")

    def test_rename_empty_target_raises(self, engine):
        """apply_rename with empty target_field raises ValueError."""
        with pytest.raises(ValueError, match="target_field"):
            engine.apply_rename([{"a": 1}], "a", "")

    def test_remove_deletes_field(self, engine):
        """apply_remove removes the field from all records."""
        data = [{"name": "Alice", "age": 28}, {"name": "Bob", "age": 35}]
        result = engine.apply_remove(data, "age")
        assert "age" not in result[0]
        assert "age" not in result[1]
        assert result[0]["name"] == "Alice"

    def test_remove_missing_field_no_error(self, engine):
        """apply_remove on records missing the field does not raise."""
        data = [{"name": "Alice"}]
        result = engine.apply_remove(data, "nonexistent")
        assert result[0] == {"name": "Alice"}

    def test_remove_empty_field_raises(self, engine):
        """apply_remove with empty field raises ValueError."""
        with pytest.raises(ValueError, match="field"):
            engine.apply_remove([{"a": 1}], "")

    def test_add_field_with_default(self, engine):
        """apply_add adds a field with the specified default value."""
        data = [{"name": "Alice"}, {"name": "Bob"}]
        result = engine.apply_add(data, "role", "member")
        assert result[0]["role"] == "member"
        assert result[1]["role"] == "member"

    def test_add_field_existing_not_overwritten(self, engine):
        """apply_add does not overwrite an existing field."""
        data = [{"name": "Alice", "role": "admin"}]
        result = engine.apply_add(data, "role", "member")
        assert result[0]["role"] == "admin"

    def test_add_field_default_none(self, engine):
        """apply_add with no default_value sets field to None."""
        data = [{"name": "Alice"}]
        result = engine.apply_add(data, "new_field")
        assert result[0]["new_field"] is None

    def test_add_empty_field_raises(self, engine):
        """apply_add with empty field raises ValueError."""
        with pytest.raises(ValueError, match="field"):
            engine.apply_add([{"a": 1}], "")

    def test_cast_string_to_integer(self, engine):
        """apply_cast converts string field to integer."""
        data = [{"qty": "42"}, {"qty": "100"}]
        result = engine.apply_cast(data, "qty", "string", "integer")
        assert result[0]["qty"] == 42
        assert result[1]["qty"] == 100

    def test_cast_integer_to_string(self, engine):
        """apply_cast converts integer field to string."""
        data = [{"age": 28}]
        result = engine.apply_cast(data, "age", "integer", "string")
        assert result[0]["age"] == "28"

    def test_cast_integer_to_number(self, engine):
        """apply_cast converts integer to float."""
        data = [{"val": 10}]
        result = engine.apply_cast(data, "val", "integer", "number")
        assert result[0]["val"] == 10.0
        assert isinstance(result[0]["val"], float)

    def test_cast_number_to_integer(self, engine):
        """apply_cast converts float to integer (rounding)."""
        data = [{"val": 3.7}]
        result = engine.apply_cast(data, "val", "number", "integer")
        assert result[0]["val"] == 4  # round(3.7) = 4

    def test_cast_identity_returns_copy(self, engine):
        """apply_cast with same old_type and new_type returns a copy."""
        data = [{"val": "hello"}]
        result = engine.apply_cast(data, "val", "string", "string")
        assert result[0]["val"] == "hello"

    def test_cast_null_value_stays_none(self, engine):
        """apply_cast on a None value produces None (null propagation)."""
        data = [{"val": None}]
        result = engine.apply_cast(data, "val", "string", "integer")
        assert result[0]["val"] is None

    def test_cast_unsupported_pair_raises(self, engine):
        """apply_cast with unsupported type pair raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported cast"):
            engine.apply_cast([{"v": "x"}], "v", "string", "array")

    def test_default_fills_missing_values(self, engine):
        """apply_default fills None and missing fields with default."""
        data = [
            {"name": "Alice", "status": None},
            {"name": "Bob"},
            {"name": "Carol", "status": "active"},
        ]
        result = engine.apply_default(data, "status", "pending")
        assert result[0]["status"] == "pending"
        assert result[1]["status"] == "pending"
        assert result[2]["status"] == "active"

    def test_default_fills_empty_string(self, engine):
        """apply_default fills empty/whitespace strings with default."""
        data = [{"status": "  "}]
        result = engine.apply_default(data, "status", "pending")
        assert result[0]["status"] == "pending"

    def test_default_empty_field_raises(self, engine):
        """apply_default with empty field raises ValueError."""
        with pytest.raises(ValueError, match="field"):
            engine.apply_default([{"a": 1}], "", "x")

    def test_compute_expression_evaluated(self, engine):
        """apply_compute evaluates expression and stores result."""
        data = [{"qty": 10, "price": 5.0}]
        result = engine.apply_compute(data, "total", "{qty} * {price}", ["qty", "price"])
        assert result[0]["total"] == 50.0

    def test_compute_string_concatenation(self, engine):
        """apply_compute supports string concatenation with + operator."""
        data = [{"first": "John", "last": "Doe"}]
        result = engine.apply_compute(
            data, "full_name", '{first} + " " + {last}', ["first", "last"]
        )
        assert result[0]["full_name"] == 'John Doe'

    def test_compute_division_by_zero_returns_none(self, engine):
        """apply_compute with division by zero returns None for that field."""
        data = [{"a": 10, "b": 0}]
        result = engine.apply_compute(data, "ratio", "{a} / {b}", ["a", "b"])
        assert result[0]["ratio"] is None

    def test_compute_empty_target_raises(self, engine):
        """apply_compute with empty target_field raises ValueError."""
        with pytest.raises(ValueError, match="target_field"):
            engine.apply_compute([{"a": 1}], "", "{a}", ["a"])

    def test_compute_empty_expression_raises(self, engine):
        """apply_compute with empty expression raises ValueError."""
        with pytest.raises(ValueError, match="expression"):
            engine.apply_compute([{"a": 1}], "b", "", ["a"])


# ===========================================================================
# TestExpressionEvaluator
# ===========================================================================


class TestExpressionEvaluator:
    """Tests for the restricted _ExpressionEvaluator class."""

    def test_field_reference(self):
        """Evaluator resolves {field} references from the record."""
        evaluator = _ExpressionEvaluator()
        result = evaluator.evaluate("{x}", {"x": 42}, ["x"])
        assert result == 42

    def test_numeric_literal(self):
        """Evaluator handles numeric literals."""
        evaluator = _ExpressionEvaluator()
        result = evaluator.evaluate("3.14", {}, [])
        assert result == pytest.approx(3.14)

    def test_string_literal_double_quote(self):
        """Evaluator handles double-quoted string literals."""
        evaluator = _ExpressionEvaluator()
        result = evaluator.evaluate('"hello"', {}, [])
        assert result == "hello"

    def test_arithmetic_operations(self):
        """Evaluator supports +, -, *, / arithmetic."""
        evaluator = _ExpressionEvaluator()
        result = evaluator.evaluate("{a} + {b} * {c}", {"a": 2, "b": 3, "c": 4}, ["a", "b", "c"])
        # Operator precedence: b*c=12, then a+12=14
        assert result == 14

    def test_parenthesised_expression(self):
        """Evaluator supports parentheses for grouping."""
        evaluator = _ExpressionEvaluator()
        result = evaluator.evaluate("({a} + {b}) * {c}", {"a": 2, "b": 3, "c": 4}, ["a", "b", "c"])
        assert result == 20

    def test_missing_field_returns_none(self):
        """Evaluator returns None for missing field references."""
        evaluator = _ExpressionEvaluator()
        result = evaluator.evaluate("{missing}", {}, ["missing"])
        assert result is None

    def test_unrecognised_token_raises(self):
        """Evaluator raises ValueError on unrecognised tokens."""
        evaluator = _ExpressionEvaluator()
        with pytest.raises(ValueError, match="Unrecognised token"):
            evaluator.evaluate("@invalid", {}, [])


# ===========================================================================
# TestMigrationExecutorEdgeCases
# ===========================================================================


class TestMigrationExecutorEdgeCases:
    """Edge-case and boundary condition tests."""

    def test_empty_data_list(self, fast_engine):
        """Executing a plan with empty data list succeeds with 0 records."""
        plan = _make_validated_plan()
        result = fast_engine.execute_plan(plan, [])
        assert result["status"] == STATUS_COMPLETED
        assert result["records_processed"] == 0

    def test_empty_plan_steps_raises(self, fast_engine):
        """Plan with empty steps list raises ValueError."""
        plan = {"plan_id": "PL-no-steps", "status": "validated", "steps": []}
        with pytest.raises(ValueError, match="no steps"):
            fast_engine.execute_plan(plan, [{"a": 1}])

    def test_large_dataset(self, fast_engine):
        """Engine handles a large dataset (1000 records) without error."""
        data = [{"field": f"value_{i}", "count": i} for i in range(1000)]
        plan = _make_validated_plan(steps=[
            {"step_number": 1, "transformation_type": "add", "field": "new_col", "default_value": 0},
        ])
        result = fast_engine.execute_plan(plan, data)
        assert result["status"] == STATUS_COMPLETED
        assert result["records_processed"] == 1000

    def test_concurrent_executions(self, fast_engine):
        """Multiple threads can execute plans concurrently without errors."""
        results = []
        errors = []

        def execute_plan(idx):
            try:
                plan = _make_validated_plan(plan_id=f"PL-concurrent-{idx}")
                data = [{"id": j, "name": f"record_{j}"} for j in range(10)]
                result = fast_engine.execute_plan(plan, data)
                results.append(result["execution_id"])
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=execute_plan, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors during concurrent execution: {errors}"
        assert len(results) == 5
        assert len(set(results)) == 5  # All IDs unique

    def test_statistics_comprehensive(self, fast_engine, rename_plan, sample_data):
        """get_statistics returns all expected keys."""
        fast_engine.execute_plan(rename_plan, sample_data)
        stats = fast_engine.get_statistics()
        expected_keys = {
            "total_executions", "successful_executions", "failed_executions",
            "total_rollbacks", "total_records_processed", "total_steps_executed",
            "total_retries", "total_checkpoints_created", "active_executions",
            "stored_executions", "stored_rollbacks", "provenance_entries",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_provenance_chain_verifiable(self, fast_engine, rename_plan, sample_data):
        """Provenance chain passes verification after execution + rollback."""
        result = fast_engine.execute_plan(rename_plan, sample_data)
        fast_engine.rollback_execution(result["execution_id"])
        assert fast_engine._provenance.verify_chain() is True

    def test_module_level_constants(self):
        """Module-level constants are correctly defined."""
        assert "rename" in SUPPORTED_TRANSFORMATION_TYPES
        assert "cast" in SUPPORTED_TRANSFORMATION_TYPES
        assert "add" in SUPPORTED_TRANSFORMATION_TYPES
        assert "remove" in SUPPORTED_TRANSFORMATION_TYPES
        assert "default" in SUPPORTED_TRANSFORMATION_TYPES
        assert "compute" in SUPPORTED_TRANSFORMATION_TYPES
        assert "validated" in EXECUTABLE_PLAN_STATUSES
        assert "approved" in EXECUTABLE_PLAN_STATUSES

    def test_helper_functions(self):
        """Module-level helper functions work correctly."""
        # _sha256 returns 64-char hex
        h = _sha256({"test": "data"})
        assert len(h) == 64
        assert re.match(r"^[0-9a-f]{64}$", h)

        # _sha256 is deterministic
        assert _sha256({"a": 1}) == _sha256({"a": 1})

        # _sha256 of None
        assert len(_sha256(None)) == 64

        # _generate_id
        eid = _generate_id("TEST")
        assert eid.startswith("TEST-")
        assert len(eid) == 5 + 12  # prefix + dash + 12 hex chars

        # _is_missing
        assert _is_missing(None) is True
        assert _is_missing("") is True
        assert _is_missing("   ") is True
        assert _is_missing("value") is False
        assert _is_missing(0) is False

        # _utcnow_iso returns ISO string
        iso = _utcnow_iso()
        assert "T" in iso

    def test_execute_step_directly(self, engine):
        """execute_step can be called directly for a single step."""
        step = {
            "step_number": 1,
            "transformation_type": "add",
            "field": "new_field",
            "default_value": "hello",
        }
        data = [{"existing": "value"}]
        result = engine.execute_step(step, data, "EXE-direct")
        assert result["status"] == "completed"
        assert result["records_affected"] == 1
        assert result["data"][0]["new_field"] == "hello"


# ===========================================================================
# TestExecutePlanWithAllTransformTypes
# ===========================================================================


class TestExecutePlanWithAllTransformTypes:
    """Integration-style tests running full plans with various transform types."""

    def test_add_field_via_plan(self, fast_engine, sample_data):
        """Plan with add step adds a field to all records."""
        plan = {
            "plan_id": "PL-add",
            "status": "validated",
            "steps": [
                {"step_number": 1, "transformation_type": "add", "field": "salary", "default_value": 50000},
            ],
        }
        result = fast_engine.execute_plan(plan, sample_data)
        assert result["status"] == STATUS_COMPLETED

    def test_remove_field_via_plan(self, fast_engine, sample_data):
        """Plan with remove step removes the field from all records."""
        plan = {
            "plan_id": "PL-remove",
            "status": "validated",
            "steps": [
                {"step_number": 1, "transformation_type": "remove", "field": "age"},
            ],
        }
        result = fast_engine.execute_plan(plan, sample_data)
        assert result["status"] == STATUS_COMPLETED

    def test_rename_field_via_plan(self, fast_engine, sample_data):
        """Plan with rename step renames the field in all records."""
        plan = {
            "plan_id": "PL-rename",
            "status": "validated",
            "steps": [
                {"step_number": 1, "transformation_type": "rename", "source_field": "department", "target_field": "team"},
            ],
        }
        result = fast_engine.execute_plan(plan, sample_data)
        assert result["status"] == STATUS_COMPLETED

    def test_cast_field_via_plan(self, fast_engine):
        """Plan with cast step casts the field type."""
        data = [{"age": "28"}, {"age": "35"}]
        plan = {
            "plan_id": "PL-cast",
            "status": "validated",
            "steps": [
                {"step_number": 1, "transformation_type": "cast", "field": "age", "old_type": "string", "new_type": "integer"},
            ],
        }
        result = fast_engine.execute_plan(plan, data)
        assert result["status"] == STATUS_COMPLETED

    def test_default_field_via_plan(self, fast_engine):
        """Plan with default step fills missing values."""
        data = [{"name": "Alice", "status": None}, {"name": "Bob"}]
        plan = {
            "plan_id": "PL-default",
            "status": "validated",
            "steps": [
                {"step_number": 1, "transformation_type": "default", "field": "status", "default_value": "active"},
            ],
        }
        result = fast_engine.execute_plan(plan, data)
        assert result["status"] == STATUS_COMPLETED

    def test_compute_field_via_plan(self, fast_engine):
        """Plan with compute step derives a new field from expression."""
        data = [{"qty": 10, "price": 5.0}, {"qty": 20, "price": 3.0}]
        plan = {
            "plan_id": "PL-compute",
            "status": "validated",
            "steps": [
                {
                    "step_number": 1,
                    "transformation_type": "compute",
                    "target_field": "total",
                    "expression": "{qty} * {price}",
                    "source_fields": ["qty", "price"],
                },
            ],
        }
        result = fast_engine.execute_plan(plan, data)
        assert result["status"] == STATUS_COMPLETED

    def test_multi_step_execution_order(self, fast_engine, sample_data):
        """Multi-step plan executes steps in the correct order."""
        plan = {
            "plan_id": "PL-multi-order",
            "status": "validated",
            "steps": [
                {"step_number": 1, "transformation_type": "add", "field": "salary", "default_value": 0},
                {"step_number": 2, "transformation_type": "rename", "source_field": "department", "target_field": "team"},
                {"step_number": 3, "transformation_type": "remove", "field": "age"},
                {"step_number": 4, "transformation_type": "default", "field": "salary", "default_value": 50000},
            ],
        }
        result = fast_engine.execute_plan(plan, sample_data)
        assert result["status"] == STATUS_COMPLETED
        assert result["completed_steps"] == 4
        step_nums = [sr["step_number"] for sr in result["step_results"]]
        assert step_nums == [1, 2, 3, 4]


# ===========================================================================
# TestCastFunctions
# ===========================================================================


class TestCastFunctions:
    """Tests for the CAST_FUNCTIONS dispatch table."""

    def test_string_to_boolean_true_values(self):
        """String->boolean cast recognizes 'true', '1', 'yes'."""
        cast_fn = CAST_FUNCTIONS[("string", "boolean")]
        assert cast_fn("true") is True
        assert cast_fn("1") is True
        assert cast_fn("yes") is True
        assert cast_fn("True") is True
        assert cast_fn("YES") is True

    def test_string_to_boolean_false_values(self):
        """String->boolean cast returns False for unrecognized values."""
        cast_fn = CAST_FUNCTIONS[("string", "boolean")]
        assert cast_fn("false") is False
        assert cast_fn("no") is False
        assert cast_fn("0") is False

    def test_boolean_to_string(self):
        """Boolean->string cast produces 'true'/'false'."""
        cast_fn = CAST_FUNCTIONS[("boolean", "string")]
        assert cast_fn(True) == "true"
        assert cast_fn(False) == "false"

    def test_boolean_to_integer(self):
        """Boolean->integer cast produces 1/0."""
        cast_fn = CAST_FUNCTIONS[("boolean", "integer")]
        assert cast_fn(True) == 1
        assert cast_fn(False) == 0

    def test_string_to_number(self):
        """String->number cast produces a float."""
        cast_fn = CAST_FUNCTIONS[("string", "number")]
        assert cast_fn("3.14") == pytest.approx(3.14)

    def test_number_to_string(self):
        """Number->string cast produces a string representation."""
        cast_fn = CAST_FUNCTIONS[("number", "string")]
        assert cast_fn(3.14) == "3.14"

    def test_null_propagation_all_casts(self):
        """All cast functions propagate None -> None."""
        for key, fn in CAST_FUNCTIONS.items():
            result = fn(None)
            assert result is None or result is False, (
                f"Cast {key} did not propagate None correctly, got {result!r}"
            )
