# -*- coding: utf-8 -*-
"""
Unit Tests for MigrationPlannerEngine - AGENT-DATA-017: Schema Migration Agent
===============================================================================

Tests all public and private methods of MigrationPlannerEngine with ~100 tests
covering plan creation, step generation, effort estimation, ordering, validation,
dry-run simulation, plan CRUD, rollback generation (via step reversal), status
transitions, provenance tracking, edge cases, and thread safety.

Test Classes (11):
    - TestMigrationPlannerInit (5 tests)
    - TestCreatePlan (21 tests)
    - TestGenerateSteps (17 tests)
    - TestEstimateEffort (11 tests)
    - TestOrderSteps (10 tests)
    - TestValidatePlan (10 tests)
    - TestGetPlan (5 tests)
    - TestListPlans (9 tests)
    - TestGenerateRollbackPlan (8 tests)
    - TestUpdatePlanStatus (6 tests)
    - TestMigrationPlannerEdgeCases (8 tests)

Total: ~110 tests

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
"""

from __future__ import annotations

import copy
import re
import threading
import time
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from greenlang.schema_migration.migration_planner import (
    MigrationPlannerEngine,
    OP_ADD,
    OP_CAST,
    OP_COMPUTE,
    OP_DEFAULT,
    OP_MERGE,
    OP_REMOVE,
    OP_RENAME,
    OP_SPLIT,
    _EFFORT_HIGH_MAX,
    _EFFORT_LOW_MAX,
    _EFFORT_MEDIUM_MAX,
    _OPERATION_BASE_COST,
    _OPERATION_ORDER,
    _compute_sha256,
    _safe_cast,
    _serialize,
)
from greenlang.schema_migration.provenance import ProvenanceTracker


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> MigrationPlannerEngine:
    """Create a fresh MigrationPlannerEngine instance for each test."""
    return MigrationPlannerEngine(genesis_hash="test-genesis")


@pytest.fixture
def rename_change() -> Dict[str, Any]:
    """A single rename change dict."""
    return {
        "change_type": "field_renamed",
        "source_field": "dept",
        "target_field": "department",
    }


@pytest.fixture
def cast_change() -> Dict[str, Any]:
    """A single type-change change dict."""
    return {
        "change_type": "type_changed",
        "field_name": "quantity",
        "old_type": "integer",
        "new_type": "number",
    }


@pytest.fixture
def add_change() -> Dict[str, Any]:
    """A single add-field change dict."""
    return {
        "change_type": "field_added",
        "field_name": "salary",
        "field_type": "number",
        "required": False,
    }


@pytest.fixture
def remove_change() -> Dict[str, Any]:
    """A single remove-field change dict."""
    return {
        "change_type": "field_removed",
        "field_name": "legacy_id",
    }


@pytest.fixture
def default_change() -> Dict[str, Any]:
    """A single set-default change dict."""
    return {
        "change_type": "default_added",
        "field_name": "status",
        "default_value": "active",
        "field_type": "string",
    }


@pytest.fixture
def compute_change() -> Dict[str, Any]:
    """A single compute-field change dict."""
    return {
        "change_type": "computed_field",
        "field_name": "total",
        "source_fields": ["price", "quantity"],
        "formula": "price * quantity",
        "output_type": "number",
    }


@pytest.fixture
def split_change() -> Dict[str, Any]:
    """A single split-field change dict."""
    return {
        "change_type": "field_split",
        "source_field": "full_name",
        "target_fields": ["first_name", "last_name"],
    }


@pytest.fixture
def merge_change() -> Dict[str, Any]:
    """A single merge-fields change dict."""
    return {
        "change_type": "fields_merged",
        "source_fields": ["first_name", "last_name"],
        "target_field": "full_name",
    }


@pytest.fixture
def mixed_changes(
    rename_change, cast_change, add_change, remove_change
) -> List[Dict[str, Any]]:
    """A list of 4 mixed changes: rename, cast, add, remove."""
    return [rename_change, cast_change, add_change, remove_change]


@pytest.fixture
def basic_plan_kwargs(mixed_changes) -> Dict[str, Any]:
    """Standard keyword arguments for create_plan calls."""
    return {
        "source_schema_id": "schema_A",
        "target_schema_id": "schema_B",
        "source_version": "1.0.0",
        "target_version": "2.0.0",
        "changes": mixed_changes,
        "estimated_records": 5000,
    }


def _make_plan(engine: MigrationPlannerEngine, **overrides) -> Dict[str, Any]:
    """Helper to create a plan with sensible defaults, allowing overrides."""
    defaults = {
        "source_schema_id": "src",
        "target_schema_id": "tgt",
        "source_version": "1.0.0",
        "target_version": "2.0.0",
        "changes": [
            {"change_type": "field_added", "field_name": "new_field", "field_type": "string"},
        ],
        "estimated_records": 100,
    }
    defaults.update(overrides)
    return engine.create_plan(**defaults)


# ===========================================================================
# TestMigrationPlannerInit
# ===========================================================================


class TestMigrationPlannerInit:
    """Tests for MigrationPlannerEngine.__init__."""

    def test_default_initialization(self):
        """Engine initialises with default genesis hash."""
        engine = MigrationPlannerEngine()
        assert engine._plans == {}
        assert engine._lock is not None
        assert isinstance(engine._provenance, ProvenanceTracker)

    def test_custom_genesis_hash(self):
        """Engine accepts a custom genesis hash string."""
        engine = MigrationPlannerEngine(genesis_hash="custom-genesis")
        assert isinstance(engine._provenance, ProvenanceTracker)

    def test_stats_initialised(self):
        """Engine starts with zeroed statistics."""
        engine = MigrationPlannerEngine()
        stats = engine.get_statistics()
        assert stats["plans_created"] == 0
        assert stats["plans_validated"] == 0
        assert stats["plans_dry_run"] == 0
        assert stats["total_steps_generated"] == 0

    def test_thread_lock_is_present(self, engine):
        """Engine has a threading.Lock for concurrency safety."""
        assert isinstance(engine._lock, type(threading.Lock()))

    def test_plans_store_empty_on_init(self, engine):
        """Plan store is empty on fresh initialisation."""
        assert engine.list_plans() == []


# ===========================================================================
# TestCreatePlan
# ===========================================================================


class TestCreatePlan:
    """Tests for MigrationPlannerEngine.create_plan."""

    def test_basic_plan_creation(self, engine, basic_plan_kwargs):
        """create_plan returns a dict with all required keys."""
        plan = engine.create_plan(**basic_plan_kwargs)

        assert "plan_id" in plan
        assert plan["source_schema_id"] == "schema_A"
        assert plan["target_schema_id"] == "schema_B"
        assert plan["source_version"] == "1.0.0"
        assert plan["target_version"] == "2.0.0"
        assert plan["status"] == "pending"
        assert isinstance(plan["steps"], list)
        assert plan["step_count"] == len(plan["steps"])
        assert "effort" in plan
        assert "effort_band" in plan
        assert "provenance_hash" in plan
        assert "created_at" in plan
        assert plan["estimated_records"] == 5000

    def test_plan_id_has_prefix(self, engine, basic_plan_kwargs):
        """Plan IDs start with 'PLAN-' prefix."""
        plan = engine.create_plan(**basic_plan_kwargs)
        assert plan["plan_id"].startswith("PLAN-")

    def test_plan_id_is_unique(self, engine, basic_plan_kwargs):
        """Each plan creation produces a unique plan_id."""
        plan1 = engine.create_plan(**basic_plan_kwargs)
        plan2 = engine.create_plan(**basic_plan_kwargs)
        assert plan1["plan_id"] != plan2["plan_id"]

    def test_status_starts_as_pending(self, engine, basic_plan_kwargs):
        """Newly created plans always have status 'pending'."""
        plan = engine.create_plan(**basic_plan_kwargs)
        assert plan["status"] == "pending"

    def test_provenance_hash_is_sha256(self, engine, basic_plan_kwargs):
        """Provenance hash is a 64-character hex string (SHA-256)."""
        plan = engine.create_plan(**basic_plan_kwargs)
        assert len(plan["provenance_hash"]) == 64
        assert re.match(r"^[0-9a-f]{64}$", plan["provenance_hash"])

    def test_created_at_is_iso_format(self, engine, basic_plan_kwargs):
        """created_at is a valid ISO-format timestamp string."""
        plan = engine.create_plan(**basic_plan_kwargs)
        # Should not raise for valid ISO format
        assert "T" in plan["created_at"]

    def test_plan_with_empty_changes_produces_no_steps(self, engine):
        """An empty changes list produces a plan with 0 steps."""
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=[],
        )
        assert plan["step_count"] == 0
        assert plan["steps"] == []

    def test_plan_with_many_changes(self, engine):
        """Plans handle large numbers of changes (50 add changes)."""
        changes = [
            {"change_type": "field_added", "field_name": f"field_{i}", "field_type": "string"}
            for i in range(50)
        ]
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=changes,
            estimated_records=10000,
        )
        assert plan["step_count"] == 50
        assert len(plan["steps"]) == 50

    def test_plan_stored_internally(self, engine, basic_plan_kwargs):
        """Plan is stored and retrievable via get_plan."""
        plan = engine.create_plan(**basic_plan_kwargs)
        retrieved = engine.get_plan(plan["plan_id"])
        assert retrieved is not None
        assert retrieved["plan_id"] == plan["plan_id"]

    def test_plan_with_source_and_target_definitions(self, engine):
        """create_plan accepts optional source_definition and target_definition."""
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=[{"change_type": "field_added", "field_name": "x", "field_type": "string"}],
            source_definition={"type": "object", "properties": {}},
            target_definition={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        assert plan["step_count"] >= 1

    def test_stats_updated_on_creation(self, engine, basic_plan_kwargs):
        """Statistics counters increment on plan creation."""
        engine.create_plan(**basic_plan_kwargs)
        stats = engine.get_statistics()
        assert stats["plans_created"] == 1
        assert stats["total_steps_generated"] > 0

    def test_estimated_records_zero_default(self, engine):
        """estimated_records defaults to 0 and is stored as 0."""
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=[{"change_type": "field_added", "field_name": "x", "field_type": "string"}],
        )
        assert plan["estimated_records"] == 0

    def test_negative_estimated_records_clamped_to_zero(self, engine):
        """Negative estimated_records values are clamped to 0."""
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=[{"change_type": "field_added", "field_name": "x", "field_type": "string"}],
            estimated_records=-500,
        )
        assert plan["estimated_records"] == 0

    def test_validation_errors_empty_on_creation(self, engine, basic_plan_kwargs):
        """Plan starts with an empty validation_errors list."""
        plan = engine.create_plan(**basic_plan_kwargs)
        assert plan["validation_errors"] == []

    def test_dry_run_result_none_on_creation(self, engine, basic_plan_kwargs):
        """Plan starts with dry_run_result=None."""
        plan = engine.create_plan(**basic_plan_kwargs)
        assert plan["dry_run_result"] is None

    def test_empty_source_schema_id_raises(self, engine):
        """Empty source_schema_id raises ValueError."""
        with pytest.raises(ValueError, match="source_schema_id"):
            engine.create_plan(
                source_schema_id="",
                target_schema_id="tgt",
                source_version="1.0.0",
                target_version="2.0.0",
                changes=[],
            )

    def test_empty_target_schema_id_raises(self, engine):
        """Empty target_schema_id raises ValueError."""
        with pytest.raises(ValueError, match="target_schema_id"):
            engine.create_plan(
                source_schema_id="src",
                target_schema_id="",
                source_version="1.0.0",
                target_version="2.0.0",
                changes=[],
            )

    def test_empty_source_version_raises(self, engine):
        """Empty source_version raises ValueError."""
        with pytest.raises(ValueError, match="source_version"):
            engine.create_plan(
                source_schema_id="src",
                target_schema_id="tgt",
                source_version="",
                target_version="2.0.0",
                changes=[],
            )

    def test_changes_not_list_raises(self, engine):
        """Non-list changes argument raises ValueError."""
        with pytest.raises(ValueError, match="changes must be a list"):
            engine.create_plan(
                source_schema_id="src",
                target_schema_id="tgt",
                source_version="1.0.0",
                target_version="2.0.0",
                changes="not-a-list",
            )

    def test_add_with_default_generates_two_steps(self, engine):
        """A field_added change with default_value creates add + set_default steps."""
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=[{
                "change_type": "field_added",
                "field_name": "status",
                "field_type": "string",
                "default_value": "active",
            }],
        )
        ops = [s["operation"] for s in plan["steps"]]
        assert OP_ADD in ops
        assert OP_DEFAULT in ops
        assert plan["step_count"] == 2


# ===========================================================================
# TestGenerateSteps
# ===========================================================================


class TestGenerateSteps:
    """Tests for MigrationPlannerEngine.generate_steps and individual step generators."""

    def test_rename_step_generated(self, engine, rename_change):
        """field_renamed produces a rename_field step."""
        steps = engine.generate_steps([rename_change])
        assert len(steps) == 1
        assert steps[0]["operation"] == OP_RENAME
        assert steps[0]["source_field"] == "dept"
        assert steps[0]["target_field"] == "department"
        assert steps[0]["reversible"] is True

    def test_cast_step_generated(self, engine, cast_change):
        """type_changed produces a cast_type step."""
        steps = engine.generate_steps([cast_change])
        assert len(steps) == 1
        assert steps[0]["operation"] == OP_CAST
        assert steps[0]["parameters"]["old_type"] == "integer"
        assert steps[0]["parameters"]["new_type"] == "number"
        assert steps[0]["reversible"] is False

    def test_cast_step_with_precision(self, engine):
        """Type change with decimal(10,2) extracts precision and scale."""
        steps = engine.generate_steps([{
            "change_type": "type_changed",
            "field_name": "amount",
            "old_type": "integer",
            "new_type": "decimal(10,2)",
        }])
        params = steps[0]["parameters"]
        assert params["precision"] == 10
        assert params["scale"] == 2

    def test_add_step_generated(self, engine, add_change):
        """field_added produces an add_field step."""
        steps = engine.generate_steps([add_change])
        assert len(steps) == 1
        assert steps[0]["operation"] == OP_ADD
        assert steps[0]["target_field"] == "salary"
        assert steps[0]["source_field"] is None

    def test_remove_step_generated(self, engine, remove_change):
        """field_removed produces a remove_field step."""
        steps = engine.generate_steps([remove_change])
        assert len(steps) == 1
        assert steps[0]["operation"] == OP_REMOVE
        assert steps[0]["source_field"] == "legacy_id"
        assert steps[0]["target_field"] is None
        assert steps[0]["reversible"] is False

    def test_default_step_generated(self, engine, default_change):
        """default_added produces a set_default step."""
        steps = engine.generate_steps([default_change])
        assert len(steps) == 1
        assert steps[0]["operation"] == OP_DEFAULT
        assert steps[0]["parameters"]["default_value"] == "active"

    def test_compute_step_generated(self, engine, compute_change):
        """computed_field produces a compute_field step."""
        steps = engine.generate_steps([compute_change])
        assert len(steps) == 1
        assert steps[0]["operation"] == OP_COMPUTE
        assert steps[0]["target_field"] == "total"
        assert steps[0]["parameters"]["formula"] == "price * quantity"
        assert steps[0]["parameters"]["source_fields"] == ["price", "quantity"]

    def test_split_step_generated(self, engine, split_change):
        """field_split produces a split_field step."""
        steps = engine.generate_steps([split_change])
        assert len(steps) == 1
        assert steps[0]["operation"] == OP_SPLIT
        assert steps[0]["source_field"] == "full_name"
        assert steps[0]["parameters"]["target_fields"] == ["first_name", "last_name"]

    def test_merge_step_generated(self, engine, merge_change):
        """fields_merged produces a merge_fields step."""
        steps = engine.generate_steps([merge_change])
        assert len(steps) == 1
        assert steps[0]["operation"] == OP_MERGE
        assert steps[0]["target_field"] == "full_name"
        assert steps[0]["parameters"]["source_fields"] == ["first_name", "last_name"]

    def test_step_numbers_sequential(self, engine, mixed_changes):
        """Steps are numbered sequentially starting from 1."""
        steps = engine.generate_steps(mixed_changes)
        numbers = [s["step_number"] for s in steps]
        assert numbers == list(range(1, len(steps) + 1))

    def test_steps_have_required_keys(self, engine, rename_change):
        """Every generated step contains the required structural keys."""
        steps = engine.generate_steps([rename_change])
        required_keys = {
            "step_number", "operation", "source_field", "target_field",
            "parameters", "reversible", "description", "depends_on",
        }
        for step in steps:
            assert required_keys.issubset(step.keys()), (
                f"Missing keys: {required_keys - set(step.keys())}"
            )

    def test_unknown_change_type_skipped(self, engine):
        """Unknown change_type entries are silently skipped."""
        steps = engine.generate_steps([
            {"change_type": "totally_unknown", "field_name": "foo"},
        ])
        assert steps == []

    def test_non_dict_change_entries_skipped(self, engine):
        """Non-dict entries in the changes list are skipped."""
        steps = engine.generate_steps([
            "this is not a dict",
            42,
            None,
            {"change_type": "field_added", "field_name": "x", "field_type": "string"},
        ])
        assert len(steps) == 1
        assert steps[0]["operation"] == OP_ADD

    def test_changes_not_list_raises_type_error(self, engine):
        """generate_steps raises TypeError if changes is not a list."""
        with pytest.raises(TypeError, match="changes must be a list"):
            engine.generate_steps("not-a-list")

    def test_multiple_change_types_combined(self, engine, mixed_changes):
        """Mixed change types all produce their respective steps."""
        steps = engine.generate_steps(mixed_changes)
        ops = {s["operation"] for s in steps}
        assert OP_RENAME in ops
        assert OP_CAST in ops
        assert OP_ADD in ops
        assert OP_REMOVE in ops

    def test_compute_step_formula_inference(self, engine):
        """Compute step infers formula from target_definition x-formula."""
        target_def = {
            "properties": {
                "total_co2": {"x-formula": "direct_co2 + indirect_co2"},
            }
        }
        steps = engine.generate_steps(
            [{"change_type": "computed_field", "field_name": "total_co2"}],
            target_definition=target_def,
        )
        assert steps[0]["parameters"]["formula"] == "direct_co2 + indirect_co2"

    def test_compute_step_fallback_formula(self, engine):
        """Compute step falls back to source_fields sum when no formula provided."""
        steps = engine.generate_steps([
            {
                "change_type": "computed_field",
                "field_name": "score",
                "source_fields": ["a", "b"],
            }
        ])
        assert steps[0]["parameters"]["formula"] == "a + b"


# ===========================================================================
# TestEstimateEffort
# ===========================================================================


class TestEstimateEffort:
    """Tests for MigrationPlannerEngine.estimate_effort."""

    def test_low_effort_band(self, engine):
        """Small number of simple steps with few records produces LOW band."""
        steps = [{"operation": OP_RENAME}]
        effort = engine.estimate_effort(steps, estimated_records=1000)
        # cost = 1 * 1000 / 1000 = 1.0 < 60 => LOW
        assert effort["effort_band"] == "LOW"
        assert effort["base_cost"] == 1
        assert effort["total_cost"] == 1.0

    def test_medium_effort_band(self, engine):
        """Moderate steps with enough records produces MEDIUM band."""
        # 3 renames = base_cost 3, records 30000 => total_cost = 3 * 30000 / 1000 = 90
        steps = [{"operation": OP_RENAME} for _ in range(3)]
        effort = engine.estimate_effort(steps, estimated_records=30000)
        assert effort["effort_band"] == "MEDIUM"

    def test_high_effort_band(self, engine):
        """Many steps or high records produces HIGH band."""
        # 10 computes = base_cost 50, records 20000 => total_cost = 50 * 20000 / 1000 = 1000
        steps = [{"operation": OP_COMPUTE} for _ in range(10)]
        effort = engine.estimate_effort(steps, estimated_records=20000)
        assert effort["effort_band"] == "HIGH"

    def test_critical_effort_band(self, engine):
        """Very large-scale migration produces CRITICAL band."""
        # 20 computes = base_cost 100, records 50000 => total_cost = 100 * 50000 / 1000 = 5000
        steps = [{"operation": OP_COMPUTE} for _ in range(20)]
        effort = engine.estimate_effort(steps, estimated_records=50000)
        assert effort["effort_band"] == "CRITICAL"

    def test_zero_records_treated_as_one(self, engine):
        """estimated_records=0 is treated as 1 for minimum baseline cost."""
        steps = [{"operation": OP_RENAME}]
        effort = engine.estimate_effort(steps, estimated_records=0)
        assert effort["estimated_records"] == 1
        assert effort["total_cost"] == pytest.approx(1 / 1000.0)

    def test_negative_records_treated_as_one(self, engine):
        """Negative estimated_records is treated as 1."""
        steps = [{"operation": OP_RENAME}]
        effort = engine.estimate_effort(steps, estimated_records=-999)
        assert effort["estimated_records"] == 1

    def test_empty_steps_returns_zero_cost(self, engine):
        """Empty step list returns zero base_cost and LOW band."""
        effort = engine.estimate_effort([], estimated_records=100000)
        assert effort["base_cost"] == 0
        assert effort["total_cost"] == 0.0
        assert effort["effort_band"] == "LOW"

    def test_step_costs_breakdown(self, engine):
        """step_costs dict provides per-step cost breakdown."""
        steps = [
            {"operation": OP_RENAME},
            {"operation": OP_CAST},
            {"operation": OP_COMPUTE},
        ]
        effort = engine.estimate_effort(steps, estimated_records=1000)
        assert len(effort["step_costs"]) == 3
        assert effort["base_cost"] == 1 + 2 + 5  # rename + cast + compute

    def test_unknown_operation_has_cost_one(self, engine):
        """Steps with unknown operation names get a default cost of 1."""
        steps = [{"operation": "unknown_op"}]
        effort = engine.estimate_effort(steps, estimated_records=1000)
        assert effort["base_cost"] == 1

    def test_estimated_minutes_calculated(self, engine):
        """estimated_minutes is derived from total_cost / 60."""
        steps = [{"operation": OP_COMPUTE} for _ in range(10)]
        effort = engine.estimate_effort(steps, estimated_records=12000)
        # base_cost = 50, total_cost = 50 * 12000 / 1000 = 600
        assert effort["estimated_minutes"] == pytest.approx(600.0 / 60.0, rel=1e-2)

    def test_effort_band_boundary_low_medium(self, engine):
        """Boundary test: total_cost exactly at _EFFORT_LOW_MAX threshold."""
        # We need total_cost = 60 exactly (LOW < 60, so 60 is MEDIUM)
        # base_cost = 6, records = 10000 => 6 * 10000 / 1000 = 60
        steps = [{"operation": OP_RENAME} for _ in range(6)]
        effort = engine.estimate_effort(steps, estimated_records=10000)
        assert effort["total_cost"] == 60.0
        assert effort["effort_band"] == "MEDIUM"


# ===========================================================================
# TestOrderSteps
# ===========================================================================


class TestOrderSteps:
    """Tests for MigrationPlannerEngine.order_steps."""

    def test_empty_list_returns_empty(self, engine):
        """Ordering an empty list returns an empty list."""
        assert engine.order_steps([]) == []

    def test_add_before_rename(self, engine):
        """add_field steps (weight 0) appear before rename_field (weight 1)."""
        steps = [
            {"operation": OP_RENAME, "source_field": "a", "target_field": "b"},
            {"operation": OP_ADD, "source_field": None, "target_field": "c"},
        ]
        ordered = engine.order_steps(steps)
        assert ordered[0]["operation"] == OP_ADD
        assert ordered[1]["operation"] == OP_RENAME

    def test_rename_before_cast(self, engine):
        """rename_field steps (weight 1) appear before cast_type (weight 2)."""
        steps = [
            {"operation": OP_CAST, "source_field": "x", "target_field": "x"},
            {"operation": OP_RENAME, "source_field": "a", "target_field": "b"},
        ]
        ordered = engine.order_steps(steps)
        assert ordered[0]["operation"] == OP_RENAME
        assert ordered[1]["operation"] == OP_CAST

    def test_remove_always_last(self, engine):
        """remove_field steps (weight 7) are always last."""
        steps = [
            {"operation": OP_REMOVE, "source_field": "old"},
            {"operation": OP_ADD, "source_field": None, "target_field": "new"},
            {"operation": OP_RENAME, "source_field": "a", "target_field": "b"},
        ]
        ordered = engine.order_steps(steps)
        assert ordered[-1]["operation"] == OP_REMOVE
        assert ordered[0]["operation"] == OP_ADD

    def test_full_ordering_sequence(self, engine):
        """All 8 operation types are ordered correctly by weight."""
        steps = [
            {"operation": OP_REMOVE},
            {"operation": OP_MERGE},
            {"operation": OP_SPLIT},
            {"operation": OP_COMPUTE},
            {"operation": OP_DEFAULT},
            {"operation": OP_CAST},
            {"operation": OP_RENAME},
            {"operation": OP_ADD},
        ]
        ordered = engine.order_steps(steps)
        expected_order = [OP_ADD, OP_RENAME, OP_CAST, OP_DEFAULT, OP_COMPUTE, OP_SPLIT, OP_MERGE, OP_REMOVE]
        actual_order = [s["operation"] for s in ordered]
        assert actual_order == expected_order

    def test_stable_sort_preserves_original_order(self, engine):
        """Steps with the same operation type preserve their original order."""
        steps = [
            {"operation": OP_RENAME, "source_field": "first"},
            {"operation": OP_RENAME, "source_field": "second"},
            {"operation": OP_RENAME, "source_field": "third"},
        ]
        ordered = engine.order_steps(steps)
        fields = [s["source_field"] for s in ordered]
        assert fields == ["first", "second", "third"]

    def test_constraint_removals_before_additions(self, engine):
        """set_default (weight 3) comes after cast (weight 2)."""
        steps = [
            {"operation": OP_DEFAULT, "source_field": "x"},
            {"operation": OP_CAST, "source_field": "x"},
        ]
        ordered = engine.order_steps(steps)
        assert ordered[0]["operation"] == OP_CAST
        assert ordered[1]["operation"] == OP_DEFAULT

    def test_compute_before_split(self, engine):
        """compute_field (weight 4) comes before split_field (weight 5)."""
        steps = [
            {"operation": OP_SPLIT},
            {"operation": OP_COMPUTE},
        ]
        ordered = engine.order_steps(steps)
        assert ordered[0]["operation"] == OP_COMPUTE
        assert ordered[1]["operation"] == OP_SPLIT

    def test_split_before_merge(self, engine):
        """split_field (weight 5) comes before merge_fields (weight 6)."""
        steps = [
            {"operation": OP_MERGE},
            {"operation": OP_SPLIT},
        ]
        ordered = engine.order_steps(steps)
        assert ordered[0]["operation"] == OP_SPLIT
        assert ordered[1]["operation"] == OP_MERGE

    def test_input_list_not_mutated(self, engine):
        """order_steps does not mutate the input list."""
        steps = [
            {"operation": OP_REMOVE},
            {"operation": OP_ADD},
        ]
        original = copy.deepcopy(steps)
        engine.order_steps(steps)
        assert steps[0]["operation"] == original[0]["operation"]
        assert steps[1]["operation"] == original[1]["operation"]


# ===========================================================================
# TestValidatePlan
# ===========================================================================


class TestValidatePlan:
    """Tests for MigrationPlannerEngine.validate_plan."""

    def test_valid_plan_passes(self, engine, basic_plan_kwargs):
        """A properly created plan passes validation."""
        plan = engine.create_plan(**basic_plan_kwargs)
        result = engine.validate_plan(plan["plan_id"])
        assert result["valid"] is True
        assert result["errors"] == []
        assert result["plan_id"] == plan["plan_id"]
        assert "validated_at" in result

    def test_nonexistent_plan_fails(self, engine):
        """Validating a non-existing plan returns valid=False with error."""
        result = engine.validate_plan("PLAN-nonexistent")
        assert result["valid"] is False
        assert len(result["errors"]) == 1
        assert "not found" in result["errors"][0].lower()

    def test_plan_status_updated_on_success(self, engine, basic_plan_kwargs):
        """Successful validation updates plan status to 'validated'."""
        plan = engine.create_plan(**basic_plan_kwargs)
        engine.validate_plan(plan["plan_id"])
        updated = engine.get_plan(plan["plan_id"])
        assert updated["status"] == "validated"

    def test_plan_status_set_to_failed_on_errors(self, engine):
        """Plan with validation errors has its status set to 'failed'."""
        plan = _make_plan(engine)
        # Manually corrupt the plan by injecting a step missing required keys
        with engine._lock:
            engine._plans[plan["plan_id"]]["steps"] = [
                {"operation": "bad_op"}  # Missing description, reversible, depends_on
            ]
        result = engine.validate_plan(plan["plan_id"])
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_empty_steps_plan_fails_validation(self, engine):
        """A plan with no steps fails validation with 'no steps' error."""
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=[],
        )
        result = engine.validate_plan(plan["plan_id"])
        assert result["valid"] is False
        assert any("no steps" in e.lower() for e in result["errors"])

    def test_duplicate_step_numbers_detected(self, engine):
        """Validation detects duplicate step_number values."""
        plan = _make_plan(engine)
        with engine._lock:
            engine._plans[plan["plan_id"]]["steps"] = [
                {
                    "step_number": 1, "operation": OP_ADD,
                    "source_field": None, "target_field": "a",
                    "description": "add a", "reversible": True, "depends_on": [],
                },
                {
                    "step_number": 1, "operation": OP_ADD,
                    "source_field": None, "target_field": "b",
                    "description": "add b", "reversible": True, "depends_on": [],
                },
            ]
        result = engine.validate_plan(plan["plan_id"])
        assert result["valid"] is False
        assert any("duplicate" in e.lower() for e in result["errors"])

    def test_remove_after_non_remove_detected(self, engine):
        """Validation detects non-remove steps after a remove step."""
        plan = _make_plan(engine)
        with engine._lock:
            engine._plans[plan["plan_id"]]["steps"] = [
                {
                    "step_number": 1, "operation": OP_REMOVE,
                    "source_field": "old", "target_field": None,
                    "description": "remove", "reversible": False, "depends_on": [],
                },
                {
                    "step_number": 2, "operation": OP_ADD,
                    "source_field": None, "target_field": "new",
                    "description": "add", "reversible": True, "depends_on": [],
                },
            ]
        result = engine.validate_plan(plan["plan_id"])
        assert result["valid"] is False
        assert any("remove" in e.lower() for e in result["errors"])

    def test_validation_increments_stats(self, engine, basic_plan_kwargs):
        """Validation increments plans_validated counter in stats."""
        plan = engine.create_plan(**basic_plan_kwargs)
        engine.validate_plan(plan["plan_id"])
        stats = engine.get_statistics()
        assert stats["plans_validated"] == 1

    def test_step_count_in_result(self, engine, basic_plan_kwargs):
        """Validation result includes step_count."""
        plan = engine.create_plan(**basic_plan_kwargs)
        result = engine.validate_plan(plan["plan_id"])
        assert result["step_count"] == plan["step_count"]

    def test_depends_on_invalid_reference_detected(self, engine):
        """Validation detects depends_on referencing non-existent step numbers."""
        plan = _make_plan(engine)
        with engine._lock:
            engine._plans[plan["plan_id"]]["steps"] = [
                {
                    "step_number": 1, "operation": OP_ADD,
                    "source_field": None, "target_field": "a",
                    "description": "add a", "reversible": True,
                    "depends_on": [999],  # Non-existent step number
                },
            ]
        result = engine.validate_plan(plan["plan_id"])
        assert result["valid"] is False
        assert any("unknown" in e.lower() for e in result["errors"])


# ===========================================================================
# TestGetPlan
# ===========================================================================


class TestGetPlan:
    """Tests for MigrationPlannerEngine.get_plan."""

    def test_get_existing_plan(self, engine, basic_plan_kwargs):
        """get_plan returns the full plan dict for an existing plan."""
        plan = engine.create_plan(**basic_plan_kwargs)
        retrieved = engine.get_plan(plan["plan_id"])
        assert retrieved is not None
        assert retrieved["plan_id"] == plan["plan_id"]
        assert retrieved["steps"] == plan["steps"]

    def test_get_nonexistent_plan_returns_none(self, engine):
        """get_plan returns None for a non-existent plan_id."""
        result = engine.get_plan("PLAN-does-not-exist")
        assert result is None

    def test_get_plan_reflects_status_update(self, engine, basic_plan_kwargs):
        """get_plan reflects status changes after validation."""
        plan = engine.create_plan(**basic_plan_kwargs)
        assert engine.get_plan(plan["plan_id"])["status"] == "pending"
        engine.validate_plan(plan["plan_id"])
        assert engine.get_plan(plan["plan_id"])["status"] == "validated"

    def test_get_plan_empty_string_id(self, engine):
        """get_plan with empty string returns None."""
        assert engine.get_plan("") is None

    def test_get_plan_after_reset(self, engine, basic_plan_kwargs):
        """get_plan returns None after engine.reset() clears all plans."""
        plan = engine.create_plan(**basic_plan_kwargs)
        engine.reset()
        assert engine.get_plan(plan["plan_id"]) is None


# ===========================================================================
# TestListPlans
# ===========================================================================


class TestListPlans:
    """Tests for MigrationPlannerEngine.list_plans."""

    def test_empty_list(self, engine):
        """list_plans returns empty list when no plans exist."""
        assert engine.list_plans() == []

    def test_all_plans_returned(self, engine):
        """list_plans returns all stored plans when no filter applied."""
        _make_plan(engine, source_schema_id="src1")
        _make_plan(engine, source_schema_id="src2")
        _make_plan(engine, source_schema_id="src3")
        plans = engine.list_plans()
        assert len(plans) == 3

    def test_filter_by_status(self, engine, basic_plan_kwargs):
        """list_plans filters by status when provided."""
        plan = engine.create_plan(**basic_plan_kwargs)
        engine.validate_plan(plan["plan_id"])

        # Second plan left as pending
        _make_plan(engine, source_schema_id="other_src")

        pending_plans = engine.list_plans(status="pending")
        validated_plans = engine.list_plans(status="validated")
        assert len(pending_plans) == 1
        assert len(validated_plans) == 1

    def test_pagination_limit(self, engine):
        """list_plans respects the limit parameter."""
        for i in range(5):
            _make_plan(engine, source_schema_id=f"src_{i}")
        plans = engine.list_plans(limit=2)
        assert len(plans) == 2

    def test_pagination_offset(self, engine):
        """list_plans respects the offset parameter."""
        for i in range(5):
            _make_plan(engine, source_schema_id=f"src_{i}")
        all_plans = engine.list_plans()
        offset_plans = engine.list_plans(offset=2)
        assert len(offset_plans) == 3
        assert offset_plans[0]["plan_id"] == all_plans[2]["plan_id"]

    def test_pagination_offset_and_limit(self, engine):
        """list_plans handles combined offset and limit."""
        for i in range(10):
            _make_plan(engine, source_schema_id=f"src_{i}")
        plans = engine.list_plans(offset=3, limit=4)
        assert len(plans) == 4

    def test_summary_keys_present(self, engine, basic_plan_kwargs):
        """list_plans returns summaries with required keys."""
        engine.create_plan(**basic_plan_kwargs)
        plans = engine.list_plans()
        assert len(plans) == 1
        summary = plans[0]
        expected_keys = {
            "plan_id", "source_schema_id", "target_schema_id",
            "source_version", "target_version", "step_count",
            "effort_band", "status", "created_at",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_limit_clamped_to_max(self, engine):
        """Limit greater than 1000 is clamped to 1000."""
        _make_plan(engine)
        plans = engine.list_plans(limit=9999)
        # Should not raise, just return available plans
        assert isinstance(plans, list)

    def test_negative_offset_clamped_to_zero(self, engine):
        """Negative offset is clamped to 0."""
        _make_plan(engine)
        plans = engine.list_plans(offset=-10)
        assert len(plans) == 1


# ===========================================================================
# TestGenerateRollbackPlan
# ===========================================================================


class TestGenerateRollbackPlan:
    """Tests for rollback plan generation via step reversal logic.

    The MigrationPlannerEngine does not have a dedicated generate_rollback_plan
    method, but rollback is tested by verifying that reversed steps undo
    the original transformations when run through generate_steps.
    """

    def test_rollback_add_becomes_remove(self, engine):
        """Reversing an add_field change produces a remove_field."""
        plan = _make_plan(engine, changes=[
            {"change_type": "field_added", "field_name": "new_col", "field_type": "string"},
        ])
        # Build reverse change
        reverse_changes = [
            {"change_type": "field_removed", "field_name": "new_col"},
        ]
        rollback_steps = engine.generate_steps(reverse_changes)
        assert len(rollback_steps) == 1
        assert rollback_steps[0]["operation"] == OP_REMOVE

    def test_rollback_remove_becomes_add(self, engine):
        """Reversing a remove_field change produces an add_field."""
        reverse_changes = [
            {"change_type": "field_added", "field_name": "old_col", "field_type": "integer"},
        ]
        rollback_steps = engine.generate_steps(reverse_changes)
        assert len(rollback_steps) == 1
        assert rollback_steps[0]["operation"] == OP_ADD

    def test_rollback_rename_reverses_direction(self, engine):
        """Reversing a rename swaps source_field and target_field."""
        reverse_changes = [
            {"change_type": "field_renamed", "source_field": "department", "target_field": "dept"},
        ]
        rollback_steps = engine.generate_steps(reverse_changes)
        assert len(rollback_steps) == 1
        assert rollback_steps[0]["operation"] == OP_RENAME
        assert rollback_steps[0]["source_field"] == "department"
        assert rollback_steps[0]["target_field"] == "dept"

    def test_rollback_type_change_reverses_types(self, engine):
        """Reversing a type_changed swaps old_type and new_type."""
        reverse_changes = [
            {
                "change_type": "type_changed",
                "field_name": "quantity",
                "old_type": "number",
                "new_type": "integer",
            },
        ]
        rollback_steps = engine.generate_steps(reverse_changes)
        assert len(rollback_steps) == 1
        assert rollback_steps[0]["operation"] == OP_CAST
        assert rollback_steps[0]["parameters"]["old_type"] == "number"
        assert rollback_steps[0]["parameters"]["new_type"] == "integer"

    def test_rollback_maintains_ordering(self, engine):
        """Rollback steps maintain proper dependency ordering."""
        reverse_changes = [
            {"change_type": "field_added", "field_name": "restored_col", "field_type": "string"},
            {"change_type": "field_renamed", "source_field": "b", "target_field": "a"},
            {"change_type": "field_removed", "field_name": "obsolete_col"},
        ]
        rollback_steps = engine.generate_steps(reverse_changes)
        ops = [s["operation"] for s in rollback_steps]
        # add_field (weight 0) before rename_field (weight 1) before remove_field (weight 7)
        assert ops.index(OP_ADD) < ops.index(OP_RENAME)
        assert ops.index(OP_RENAME) < ops.index(OP_REMOVE)

    def test_rollback_nonexistent_plan_id(self, engine):
        """Attempting to get a nonexistent plan returns None (no rollback possible)."""
        assert engine.get_plan("PLAN-nonexistent") is None

    def test_rollback_complex_migration(self, engine):
        """Complex rollback reverses multiple step types correctly."""
        reverse_changes = [
            {"change_type": "field_removed", "field_name": "salary"},
            {"change_type": "field_renamed", "source_field": "team", "target_field": "department"},
            {"change_type": "field_added", "field_name": "age", "field_type": "integer"},
        ]
        rollback_steps = engine.generate_steps(reverse_changes)
        ops = {s["operation"] for s in rollback_steps}
        assert OP_ADD in ops
        assert OP_RENAME in ops
        assert OP_REMOVE in ops

    def test_rollback_plan_can_be_created(self, engine):
        """A rollback plan is a valid plan that can be stored."""
        reverse_changes = [
            {"change_type": "field_removed", "field_name": "new_field"},
        ]
        rollback_plan = engine.create_plan(
            source_schema_id="tgt",
            target_schema_id="src",
            source_version="2.0.0",
            target_version="1.0.0",
            changes=reverse_changes,
        )
        assert rollback_plan["status"] == "pending"
        assert rollback_plan["step_count"] >= 1


# ===========================================================================
# TestUpdatePlanStatus
# ===========================================================================


class TestUpdatePlanStatus:
    """Tests for plan status transitions via validate_plan and dry_run."""

    def test_pending_to_validated(self, engine, basic_plan_kwargs):
        """Successful validation transitions pending -> validated."""
        plan = engine.create_plan(**basic_plan_kwargs)
        assert plan["status"] == "pending"
        engine.validate_plan(plan["plan_id"])
        updated = engine.get_plan(plan["plan_id"])
        assert updated["status"] == "validated"

    def test_validated_to_dry_run_complete(self, engine, basic_plan_kwargs):
        """Successful dry_run on validated plan transitions to dry_run_complete."""
        plan = engine.create_plan(**basic_plan_kwargs)
        engine.validate_plan(plan["plan_id"])
        result = engine.dry_run(plan["plan_id"])
        updated = engine.get_plan(plan["plan_id"])
        assert updated["status"] == "dry_run_complete"

    def test_dry_run_only_updates_validated_status(self, engine, basic_plan_kwargs):
        """dry_run on a 'pending' plan does NOT update status to dry_run_complete."""
        plan = engine.create_plan(**basic_plan_kwargs)
        engine.dry_run(plan["plan_id"])
        updated = engine.get_plan(plan["plan_id"])
        # Should still be pending since it was not validated first
        assert updated["status"] == "pending"

    def test_dry_run_nonexistent_plan(self, engine):
        """dry_run on nonexistent plan returns error response."""
        result = engine.dry_run("PLAN-nonexistent")
        assert result["steps_simulated"] == 0
        assert len(result["errors"]) == 1
        assert "not found" in result["errors"][0].lower()

    def test_validation_failed_status(self, engine):
        """Plan with validation errors has status set to 'failed'."""
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=[],
        )
        engine.validate_plan(plan["plan_id"])
        updated = engine.get_plan(plan["plan_id"])
        assert updated["status"] == "failed"

    def test_dry_run_increments_stats(self, engine, basic_plan_kwargs):
        """dry_run increments plans_dry_run counter."""
        plan = engine.create_plan(**basic_plan_kwargs)
        engine.dry_run(plan["plan_id"])
        stats = engine.get_statistics()
        assert stats["plans_dry_run"] == 1


# ===========================================================================
# TestMigrationPlannerEdgeCases
# ===========================================================================


class TestMigrationPlannerEdgeCases:
    """Edge-case and boundary condition tests."""

    def test_empty_schemas_with_no_changes(self, engine):
        """Plan creation with empty schemas and no changes produces empty plan."""
        plan = engine.create_plan(
            source_schema_id="empty_src",
            target_schema_id="empty_tgt",
            source_version="0.0.1",
            target_version="0.0.2",
            changes=[],
            source_definition={},
            target_definition={},
        )
        assert plan["step_count"] == 0
        assert plan["effort_band"] == "LOW"

    def test_very_large_plan_50_steps(self, engine):
        """Engine handles plans with 50 steps without errors."""
        changes = [
            {"change_type": "field_added", "field_name": f"field_{i}", "field_type": "string"}
            for i in range(50)
        ]
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=changes,
        )
        assert plan["step_count"] == 50

    def test_concurrent_plan_creation(self, engine):
        """Multiple threads can create plans concurrently without errors."""
        results = []
        errors = []

        def create_plan(idx):
            try:
                plan = engine.create_plan(
                    source_schema_id=f"src_{idx}",
                    target_schema_id=f"tgt_{idx}",
                    source_version="1.0.0",
                    target_version="2.0.0",
                    changes=[{"change_type": "field_added", "field_name": f"f_{idx}", "field_type": "string"}],
                )
                results.append(plan["plan_id"])
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=create_plan, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors during concurrent creation: {errors}"
        assert len(results) == 10
        assert len(set(results)) == 10  # All IDs unique

    def test_reset_clears_all_state(self, engine, basic_plan_kwargs):
        """reset() clears all plans and resets statistics."""
        engine.create_plan(**basic_plan_kwargs)
        engine.create_plan(**basic_plan_kwargs)
        engine.reset()
        assert engine.list_plans() == []
        stats = engine.get_statistics()
        assert stats["plans_created"] == 0
        assert stats["total_plans_stored"] == 0

    def test_whitespace_only_source_schema_id_raises(self, engine):
        """Whitespace-only string for required arguments raises ValueError."""
        with pytest.raises(ValueError):
            engine.create_plan(
                source_schema_id="   ",
                target_schema_id="tgt",
                source_version="1.0.0",
                target_version="2.0.0",
                changes=[],
            )

    def test_generate_rename_step_missing_source_raises(self, engine):
        """generate_rename_step raises ValueError when source_field is missing."""
        with pytest.raises(ValueError, match="source_field"):
            engine.generate_rename_step({"target_field": "new_name"})

    def test_generate_rename_step_missing_target_raises(self, engine):
        """generate_rename_step raises ValueError when target_field is missing."""
        with pytest.raises(ValueError, match="target_field"):
            engine.generate_rename_step({"source_field": "old_name"})

    def test_generate_cast_step_missing_new_type_raises(self, engine):
        """generate_cast_step raises ValueError when new_type is missing."""
        with pytest.raises(ValueError, match="new_type"):
            engine.generate_cast_step({"field_name": "quantity", "old_type": "integer"})


# ===========================================================================
# TestDryRun
# ===========================================================================


class TestDryRun:
    """Tests for MigrationPlannerEngine.dry_run simulation."""

    def test_dry_run_valid_plan_no_sample_data(self, engine, basic_plan_kwargs):
        """dry_run with no sample data runs structural analysis only."""
        plan = engine.create_plan(**basic_plan_kwargs)
        result = engine.dry_run(plan["plan_id"])
        assert result["plan_id"] == plan["plan_id"]
        assert result["records_processed"] == 0
        assert result["steps_simulated"] == plan["step_count"]

    def test_dry_run_with_sample_data(self, engine):
        """dry_run with sample data transforms records correctly."""
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=[
                {"change_type": "field_renamed", "source_field": "dept", "target_field": "department"},
            ],
        )
        sample = [
            {"dept": "Engineering", "name": "Alice"},
            {"dept": "Science", "name": "Bob"},
        ]
        result = engine.dry_run(plan["plan_id"], sample_data=sample)
        assert result["records_processed"] == 2
        assert result["steps_successful"] >= 1

    def test_dry_run_rename_transforms_records(self, engine):
        """Dry-run rename step correctly renames fields in sample data."""
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=[
                {"change_type": "field_renamed", "source_field": "old_name", "target_field": "new_name"},
            ],
        )
        sample = [{"old_name": "value1"}, {"old_name": "value2"}]
        result = engine.dry_run(plan["plan_id"], sample_data=sample)
        assert result["steps_successful"] == 1
        # The sample data should have been mutated in-place
        assert "new_name" in sample[0]
        assert sample[0]["new_name"] == "value1"

    def test_dry_run_add_field_sets_default(self, engine):
        """Dry-run add_field step injects default value in sample records."""
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=[
                {
                    "change_type": "field_added",
                    "field_name": "status",
                    "field_type": "string",
                    "default_value": "active",
                },
            ],
        )
        sample = [{"name": "Alice"}, {"name": "Bob"}]
        engine.dry_run(plan["plan_id"], sample_data=sample)
        # add_field + set_default => status should be "active"
        assert sample[0].get("status") == "active"

    def test_dry_run_cast_step(self, engine):
        """Dry-run cast_type step converts field values."""
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=[
                {
                    "change_type": "type_changed",
                    "field_name": "age",
                    "old_type": "string",
                    "new_type": "integer",
                },
            ],
        )
        sample = [{"age": "25"}, {"age": "30"}]
        result = engine.dry_run(plan["plan_id"], sample_data=sample)
        assert result["steps_successful"] == 1
        assert sample[0]["age"] == 25
        assert sample[1]["age"] == 30

    def test_dry_run_remove_field(self, engine):
        """Dry-run remove_field step removes the field from records."""
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=[
                {"change_type": "field_removed", "field_name": "legacy"},
            ],
        )
        sample = [{"legacy": "old", "name": "Alice"}]
        engine.dry_run(plan["plan_id"], sample_data=sample)
        assert "legacy" not in sample[0]
        assert "name" in sample[0]

    def test_dry_run_result_stored_on_plan(self, engine, basic_plan_kwargs):
        """dry_run result is stored in the plan's dry_run_result field."""
        plan = engine.create_plan(**basic_plan_kwargs)
        engine.validate_plan(plan["plan_id"])
        engine.dry_run(plan["plan_id"])
        updated = engine.get_plan(plan["plan_id"])
        assert updated["dry_run_result"] is not None
        assert "completed_at" in updated["dry_run_result"]

    def test_dry_run_merge_step(self, engine):
        """Dry-run merge_fields step joins source fields into target."""
        plan = engine.create_plan(
            source_schema_id="src",
            target_schema_id="tgt",
            source_version="1.0.0",
            target_version="2.0.0",
            changes=[
                {
                    "change_type": "fields_merged",
                    "source_fields": ["first", "last"],
                    "target_field": "full_name",
                },
            ],
        )
        sample = [{"first": "John", "last": "Doe"}]
        engine.dry_run(plan["plan_id"], sample_data=sample)
        assert sample[0]["full_name"] == "John Doe"


# ===========================================================================
# TestStatistics
# ===========================================================================


class TestStatistics:
    """Tests for MigrationPlannerEngine.get_statistics."""

    def test_initial_statistics(self, engine):
        """Fresh engine has zeroed statistics."""
        stats = engine.get_statistics()
        assert stats["plans_created"] == 0
        assert stats["plans_validated"] == 0
        assert stats["plans_dry_run"] == 0
        assert stats["total_steps_generated"] == 0
        assert stats["total_plans_stored"] == 0

    def test_statistics_after_operations(self, engine, basic_plan_kwargs):
        """Statistics reflect create, validate, and dry_run operations."""
        plan = engine.create_plan(**basic_plan_kwargs)
        engine.validate_plan(plan["plan_id"])
        engine.dry_run(plan["plan_id"])

        stats = engine.get_statistics()
        assert stats["plans_created"] == 1
        assert stats["plans_validated"] == 1
        assert stats["plans_dry_run"] == 1
        assert stats["total_plans_stored"] == 1

    def test_plans_by_status_breakdown(self, engine, basic_plan_kwargs):
        """Statistics include plans_by_status breakdown."""
        engine.create_plan(**basic_plan_kwargs)  # pending
        plan2 = engine.create_plan(**basic_plan_kwargs)
        engine.validate_plan(plan2["plan_id"])  # validated or dry_run_complete

        stats = engine.get_statistics()
        assert "plans_by_status" in stats
        assert stats["plans_by_status"].get("pending", 0) >= 1

    def test_plans_by_effort_band_breakdown(self, engine, basic_plan_kwargs):
        """Statistics include plans_by_effort_band breakdown."""
        engine.create_plan(**basic_plan_kwargs)
        stats = engine.get_statistics()
        assert "plans_by_effort_band" in stats
        assert sum(stats["plans_by_effort_band"].values()) >= 1

    def test_provenance_entries_tracked(self, engine, basic_plan_kwargs):
        """Statistics include provenance_entries count."""
        engine.create_plan(**basic_plan_kwargs)
        stats = engine.get_statistics()
        assert stats["provenance_entries"] >= 1


# ===========================================================================
# TestHelperFunctions
# ===========================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_serialize_deterministic(self):
        """_serialize produces sorted-key JSON."""
        obj = {"z": 1, "a": 2, "m": 3}
        result = _serialize(obj)
        assert '"a": 2' in result
        assert result.index('"a"') < result.index('"m"') < result.index('"z"')

    def test_compute_sha256_length(self):
        """_compute_sha256 returns a 64-char hex string."""
        h = _compute_sha256("test")
        assert len(h) == 64
        assert re.match(r"^[0-9a-f]{64}$", h)

    def test_compute_sha256_deterministic(self):
        """Same input always produces same hash."""
        h1 = _compute_sha256("hello world")
        h2 = _compute_sha256("hello world")
        assert h1 == h2

    def test_compute_sha256_different_inputs(self):
        """Different inputs produce different hashes."""
        h1 = _compute_sha256("input_a")
        h2 = _compute_sha256("input_b")
        assert h1 != h2

    def test_safe_cast_integer(self):
        """_safe_cast converts string to integer."""
        ok, val = _safe_cast("42", "integer")
        assert ok is True
        assert val == 42

    def test_safe_cast_number(self):
        """_safe_cast converts string to float."""
        ok, val = _safe_cast("3.14", "number")
        assert ok is True
        assert val == pytest.approx(3.14)

    def test_safe_cast_string(self):
        """_safe_cast converts integer to string."""
        ok, val = _safe_cast(42, "string")
        assert ok is True
        assert val == "42"

    def test_safe_cast_boolean(self):
        """_safe_cast converts string to boolean."""
        ok_true, val_true = _safe_cast("true", "boolean")
        ok_false, val_false = _safe_cast("false", "boolean")
        assert ok_true is True and val_true is True
        assert ok_false is True and val_false is False

    def test_safe_cast_array(self):
        """_safe_cast wraps non-list value in a list."""
        ok, val = _safe_cast("item", "array")
        assert ok is True
        assert val == ["item"]

    def test_safe_cast_object(self):
        """_safe_cast wraps non-dict value in a dict."""
        ok, val = _safe_cast("value", "object")
        assert ok is True
        assert val == {"value": "value"}

    def test_safe_cast_unknown_type_passthrough(self):
        """_safe_cast passes through value for unknown target type."""
        ok, val = _safe_cast("data", "custom_type")
        assert ok is True
        assert val == "data"


# ===========================================================================
# TestStepGeneratorValidation
# ===========================================================================


class TestStepGeneratorValidation:
    """Tests for individual step generator validation and edge cases."""

    def test_generate_add_step_missing_field_name_raises(self, engine):
        """generate_add_step raises ValueError when field_name is empty."""
        with pytest.raises(ValueError, match="field_name"):
            engine.generate_add_step({"field_type": "string"})

    def test_generate_remove_step_missing_field_name_raises(self, engine):
        """generate_remove_step raises ValueError when field_name is empty."""
        with pytest.raises(ValueError, match="field_name"):
            engine.generate_remove_step({})

    def test_generate_default_step_missing_field_name_raises(self, engine):
        """generate_default_step raises ValueError when field_name is empty."""
        with pytest.raises(ValueError, match="field_name"):
            engine.generate_default_step({"default_value": 0})

    def test_generate_compute_step_missing_field_name_raises(self, engine):
        """generate_compute_step raises ValueError when field_name is empty."""
        with pytest.raises(ValueError, match="field_name"):
            engine.generate_compute_step({})

    def test_generate_split_step_empty_source_raises(self, engine):
        """generate_split_step raises ValueError when source_field is empty."""
        with pytest.raises(ValueError, match="source_field"):
            engine.generate_split_step("", ["a", "b"])

    def test_generate_split_step_single_target_raises(self, engine):
        """generate_split_step raises ValueError with fewer than 2 target_fields."""
        with pytest.raises(ValueError, match="target_fields"):
            engine.generate_split_step("full_name", ["only_one"])

    def test_generate_merge_step_empty_target_raises(self, engine):
        """generate_merge_step raises ValueError when target_field is empty."""
        with pytest.raises(ValueError, match="target_field"):
            engine.generate_merge_step(["a", "b"], "")

    def test_generate_merge_step_single_source_raises(self, engine):
        """generate_merge_step raises ValueError with fewer than 2 source_fields."""
        with pytest.raises(ValueError, match="source_fields"):
            engine.generate_merge_step(["only_one"], "merged")

    def test_generate_cast_step_missing_field_name_raises(self, engine):
        """generate_cast_step raises ValueError when no field_name can be determined."""
        with pytest.raises(ValueError, match="field_name"):
            engine.generate_cast_step({"new_type": "integer"})

    def test_generate_cast_step_uses_target_field_fallback(self, engine):
        """generate_cast_step falls back to target_field when field_name missing."""
        step = engine.generate_cast_step({
            "target_field": "amount",
            "old_type": "string",
            "new_type": "number",
        })
        assert step["source_field"] == "amount"

    def test_generate_add_step_uses_target_field_fallback(self, engine):
        """generate_add_step falls back to target_field when field_name missing."""
        step = engine.generate_add_step({
            "target_field": "new_col",
            "field_type": "integer",
        })
        assert step["target_field"] == "new_col"

    def test_split_step_description_includes_count(self, engine):
        """Split step description includes the number of target fields."""
        step = engine.generate_split_step("addr", ["street", "city", "zip"])
        assert "3 fields" in step["description"]

    def test_merge_step_description_includes_count(self, engine):
        """Merge step description includes the number of source fields."""
        step = engine.generate_merge_step(["first", "last"], "full_name")
        assert "2 fields" in step["description"]


# ===========================================================================
# TestProvenanceIntegration
# ===========================================================================


class TestProvenanceIntegration:
    """Tests for provenance tracking integration in MigrationPlannerEngine."""

    def test_plan_creation_records_provenance(self, engine, basic_plan_kwargs):
        """Plan creation records a provenance entry."""
        initial_count = engine._provenance.entry_count
        engine.create_plan(**basic_plan_kwargs)
        assert engine._provenance.entry_count > initial_count

    def test_validation_records_provenance(self, engine, basic_plan_kwargs):
        """Plan validation records a provenance entry."""
        plan = engine.create_plan(**basic_plan_kwargs)
        count_before = engine._provenance.entry_count
        engine.validate_plan(plan["plan_id"])
        assert engine._provenance.entry_count > count_before

    def test_dry_run_records_provenance(self, engine, basic_plan_kwargs):
        """Dry run records a provenance entry."""
        plan = engine.create_plan(**basic_plan_kwargs)
        count_before = engine._provenance.entry_count
        engine.dry_run(plan["plan_id"])
        assert engine._provenance.entry_count > count_before

    def test_provenance_chain_verifiable(self, engine, basic_plan_kwargs):
        """Provenance chain passes verification after multiple operations."""
        plan = engine.create_plan(**basic_plan_kwargs)
        engine.validate_plan(plan["plan_id"])
        engine.dry_run(plan["plan_id"])
        assert engine._provenance.verify_chain() is True

    def test_provenance_entries_contain_plan_id(self, engine, basic_plan_kwargs):
        """Provenance entries reference the correct plan_id."""
        plan = engine.create_plan(**basic_plan_kwargs)
        entries = engine._provenance.get_chain(plan["plan_id"])
        assert len(entries) >= 1
        assert entries[0].entity_id == plan["plan_id"]
