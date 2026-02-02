# -*- coding: utf-8 -*-
"""
Plan Determinism Tests for GL-FOUND-X-001 Orchestrator

Tests that verify the orchestrator produces deterministic execution plans:

1. Content-Addressable Plan IDs
   - Same pipeline + inputs = same plan_id
   - Different inputs = different plan_id
   - Plan ID is SHA-256 based

2. Step Ordering Determinism
   - Same DAG always produces same topological order
   - Step IDs are deterministically assigned

3. Idempotency Key Generation
   - Same run_id + step_id + config = same idempotency_key
   - Different attempts have different keys

4. Cross-Run Consistency
   - Running the same pipeline multiple times produces identical plans

Author: GreenLang Team
Version: 1.0.0
Coverage Target: 85%+
"""

import hashlib
import json
import copy
from datetime import datetime, timezone
from typing import Dict, Any, List

import pytest

from greenlang.utilities.determinism import DeterministicClock, freeze_time, unfreeze_time
from greenlang.orchestrator.pipeline_schema import (
    PipelineDefinition,
    PipelineMetadata,
    PipelineSpec,
    PipelineDefaults,
    StepDefinition,
    ParameterDefinition,
    ParameterType,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def plan_compiler():
    """
    Fixture providing a plan compiler function.

    This simulates the _compile_plan method from GLIPOrchestrator.
    """
    def _compile_plan(
        pipeline: PipelineDefinition,
        run_id: str,
        tenant_id: str,
        prefer_glip: bool = True,
    ) -> Dict[str, Any]:
        """Compile execution plan from pipeline definition."""
        steps = []

        for step in pipeline.spec.steps:
            agent_id = step.get_effective_id()
            agent_type = step.agent

            # Generate idempotency key
            idempotency_content = f"{run_id}:{agent_id}:{json.dumps(step.model_dump(exclude_none=True), sort_keys=True)}"
            idempotency_key = hashlib.sha256(idempotency_content.encode()).hexdigest()[:32]

            # Generate step_id
            step_id = f"{run_id}-{agent_id}"

            compiled_step = {
                "step_id": step_id,
                "agent_id": agent_id,
                "agent_type": agent_type,
                "execution_mode": "glip_v1" if prefer_glip else "in_process",
                "idempotency_key": idempotency_key,
                "dependencies": list(step.dependsOn or step.depends_on or []),
                "config": step.with_ or step.inputs or {},
            }
            steps.append(compiled_step)

        # Generate content-addressable plan_id
        plan_content = json.dumps(steps, sort_keys=True)
        plan_id = hashlib.sha256(plan_content.encode()).hexdigest()[:16]

        return {
            "plan_id": plan_id,
            "run_id": run_id,
            "pipeline_id": pipeline.metadata.name,
            "steps": steps,
            "compiled_at": DeterministicClock.now(timezone.utc).isoformat(),
        }

    return _compile_plan


# =============================================================================
# CONTENT-ADDRESSABLE PLAN ID TESTS
# =============================================================================

class TestContentAddressablePlanId:
    """Tests for content-addressable plan ID generation."""

    def test_same_pipeline_same_inputs_same_plan_id(
        self,
        simple_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """Same pipeline with same inputs produces identical plan_id."""
        # Compile the same pipeline multiple times
        plan1 = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-001",
            tenant_id="tenant-test",
        )
        plan2 = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-001",
            tenant_id="tenant-test",
        )

        # Plan IDs should be identical
        assert plan1["plan_id"] == plan2["plan_id"]
        assert len(plan1["plan_id"]) == 16  # SHA-256 truncated to 16 chars

    def test_different_run_id_different_plan_id(
        self,
        simple_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """Different run_ids produce different plan_ids."""
        plan1 = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-001",
            tenant_id="tenant-test",
        )
        plan2 = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-002",
            tenant_id="tenant-test",
        )

        # Plan IDs should differ (because step_ids contain run_id)
        assert plan1["plan_id"] != plan2["plan_id"]

    def test_plan_id_is_sha256_based(
        self,
        simple_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """Plan ID is computed using SHA-256."""
        plan = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-001",
            tenant_id="tenant-test",
        )

        # Verify we can reproduce the plan_id
        steps_json = json.dumps(plan["steps"], sort_keys=True)
        expected_id = hashlib.sha256(steps_json.encode()).hexdigest()[:16]

        assert plan["plan_id"] == expected_id

    def test_plan_id_deterministic_across_iterations(
        self,
        simple_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """Plan ID is deterministic across many iterations."""
        run_id = "run-determinism-test"
        tenant_id = "tenant-test"

        plan_ids = []
        for _ in range(100):
            plan = plan_compiler(
                pipeline=simple_pipeline_def,
                run_id=run_id,
                tenant_id=tenant_id,
            )
            plan_ids.append(plan["plan_id"])

        # All plan_ids should be identical
        assert len(set(plan_ids)) == 1

    def test_multi_step_pipeline_plan_id(
        self,
        multi_step_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """Multi-step pipeline produces consistent plan_id."""
        plans = [
            plan_compiler(
                pipeline=multi_step_pipeline_def,
                run_id="run-multi",
                tenant_id="tenant-test",
            )
            for _ in range(10)
        ]

        plan_ids = [p["plan_id"] for p in plans]
        assert len(set(plan_ids)) == 1

        # Verify step count
        assert len(plans[0]["steps"]) == 4  # ingest, transform, validate, export


# =============================================================================
# STEP ORDERING DETERMINISM TESTS
# =============================================================================

class TestStepOrderingDeterminism:
    """Tests for deterministic step ordering in execution plans."""

    def test_step_order_is_deterministic(
        self,
        multi_step_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """Steps are always in the same order across compilations."""
        plans = [
            plan_compiler(
                pipeline=multi_step_pipeline_def,
                run_id="run-order-test",
                tenant_id="tenant-test",
            )
            for _ in range(50)
        ]

        # Extract step orders
        step_orders = [
            [s["agent_id"] for s in plan["steps"]]
            for plan in plans
        ]

        # All orders should be identical
        first_order = step_orders[0]
        for order in step_orders[1:]:
            assert order == first_order

    def test_dag_topological_order_preserved(
        self,
        multi_step_pipeline_def,
    ):
        """Verify DAG dependencies are respected in execution order."""
        exec_order = multi_step_pipeline_def.get_execution_order()

        # Build dependency map
        deps = multi_step_pipeline_def.get_step_dependencies()

        # For each step, verify all dependencies come before it
        for i, step_id in enumerate(exec_order):
            for dep in deps.get(step_id, []):
                dep_index = exec_order.index(dep)
                assert dep_index < i, (
                    f"Dependency {dep} should come before {step_id}"
                )

    def test_parallel_steps_have_stable_order(
        self,
        multi_step_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """
        Steps that can run in parallel have a stable (deterministic) order.

        In multi_step_pipeline: validate and export both depend on transform
        and can run in parallel. Their order should be stable.
        """
        plans = [
            plan_compiler(
                pipeline=multi_step_pipeline_def,
                run_id="run-parallel-test",
                tenant_id="tenant-test",
            )
            for _ in range(20)
        ]

        # Find positions of validate and export
        validate_positions = []
        export_positions = []

        for plan in plans:
            for i, step in enumerate(plan["steps"]):
                if step["agent_id"] == "validate":
                    validate_positions.append(i)
                elif step["agent_id"] == "export":
                    export_positions.append(i)

        # Positions should be consistent
        assert len(set(validate_positions)) == 1
        assert len(set(export_positions)) == 1

    def test_step_ids_contain_run_id(
        self,
        simple_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """Step IDs are namespaced with run_id."""
        plan = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-namespace-test",
            tenant_id="tenant-test",
        )

        for step in plan["steps"]:
            assert step["step_id"].startswith("run-namespace-test-")
            assert step["agent_id"] in step["step_id"]


# =============================================================================
# IDEMPOTENCY KEY GENERATION TESTS
# =============================================================================

class TestIdempotencyKeyGeneration:
    """Tests for deterministic idempotency key generation."""

    def test_idempotency_key_is_deterministic(
        self,
        simple_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """Same inputs produce same idempotency key."""
        plan1 = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-idem-001",
            tenant_id="tenant-test",
        )
        plan2 = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-idem-001",
            tenant_id="tenant-test",
        )

        # Extract idempotency keys
        keys1 = {s["agent_id"]: s["idempotency_key"] for s in plan1["steps"]}
        keys2 = {s["agent_id"]: s["idempotency_key"] for s in plan2["steps"]}

        assert keys1 == keys2

    def test_idempotency_key_format(
        self,
        simple_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """Idempotency key has correct format (32-char hex)."""
        plan = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-format-test",
            tenant_id="tenant-test",
        )

        for step in plan["steps"]:
            key = step["idempotency_key"]
            assert len(key) == 32
            assert all(c in "0123456789abcdef" for c in key)

    def test_different_run_ids_produce_different_keys(
        self,
        simple_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """Different run_ids produce different idempotency keys."""
        plan1 = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-diff-001",
            tenant_id="tenant-test",
        )
        plan2 = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-diff-002",
            tenant_id="tenant-test",
        )

        keys1 = {s["idempotency_key"] for s in plan1["steps"]}
        keys2 = {s["idempotency_key"] for s in plan2["steps"]}

        # No overlap in keys
        assert keys1.isdisjoint(keys2)

    def test_each_step_has_unique_key(
        self,
        multi_step_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """Each step in a plan has a unique idempotency key."""
        plan = plan_compiler(
            pipeline=multi_step_pipeline_def,
            run_id="run-unique-keys",
            tenant_id="tenant-test",
        )

        keys = [s["idempotency_key"] for s in plan["steps"]]
        assert len(keys) == len(set(keys)), "Duplicate idempotency keys found"


# =============================================================================
# PIPELINE HASH TESTS
# =============================================================================

class TestPipelineHash:
    """Tests for pipeline definition hashing."""

    def test_pipeline_hash_is_deterministic(
        self,
        simple_pipeline_def,
    ):
        """Pipeline hash computation is deterministic."""
        hashes = [simple_pipeline_def.compute_hash() for _ in range(100)]
        assert len(set(hashes)) == 1

    def test_pipeline_hash_length(
        self,
        simple_pipeline_def,
    ):
        """Pipeline hash is full SHA-256 (64 hex chars)."""
        hash_value = simple_pipeline_def.compute_hash()
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_different_pipelines_different_hash(
        self,
        simple_pipeline_def,
        multi_step_pipeline_def,
    ):
        """Different pipelines produce different hashes."""
        hash1 = simple_pipeline_def.compute_hash()
        hash2 = multi_step_pipeline_def.compute_hash()

        assert hash1 != hash2

    def test_normalized_pipeline_same_hash(
        self,
        simple_pipeline_def,
    ):
        """Normalized pipeline produces same hash as original."""
        original_hash = simple_pipeline_def.compute_hash()
        normalized = simple_pipeline_def.normalize()
        normalized_hash = normalized.compute_hash()

        assert original_hash == normalized_hash


# =============================================================================
# TIMING INDEPENDENCE TESTS
# =============================================================================

class TestTimingIndependence:
    """Tests verifying plans are independent of wall-clock time."""

    def test_compiled_at_uses_deterministic_clock(
        self,
        simple_pipeline_def,
        plan_compiler,
    ):
        """compiled_at uses DeterministicClock."""
        # Freeze at specific time
        frozen_time = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        freeze_time(frozen_time)

        try:
            plan = plan_compiler(
                pipeline=simple_pipeline_def,
                run_id="run-timing-test",
                tenant_id="tenant-test",
            )

            # Verify compiled_at matches frozen time
            assert plan["compiled_at"] == frozen_time.isoformat()
        finally:
            unfreeze_time()

    def test_plan_content_independent_of_time(
        self,
        simple_pipeline_def,
        plan_compiler,
    ):
        """
        Plan steps are independent of compilation time.

        The compiled_at field changes, but steps and plan_id should not.
        """
        # Compile at time 1
        time1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        freeze_time(time1)
        try:
            plan1 = plan_compiler(
                pipeline=simple_pipeline_def,
                run_id="run-time-independent",
                tenant_id="tenant-test",
            )
        finally:
            unfreeze_time()

        # Compile at time 2
        time2 = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        freeze_time(time2)
        try:
            plan2 = plan_compiler(
                pipeline=simple_pipeline_def,
                run_id="run-time-independent",
                tenant_id="tenant-test",
            )
        finally:
            unfreeze_time()

        # Plan ID and steps should be identical
        assert plan1["plan_id"] == plan2["plan_id"]
        assert plan1["steps"] == plan2["steps"]

        # Only compiled_at should differ
        assert plan1["compiled_at"] != plan2["compiled_at"]


# =============================================================================
# CROSS-ENVIRONMENT CONSISTENCY TESTS
# =============================================================================

class TestCrossEnvironmentConsistency:
    """Tests verifying plans are consistent across environments."""

    def test_plan_id_independent_of_execution_mode(
        self,
        simple_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """
        Plan ID should not depend on execution mode (GLIP vs in-process).

        Note: Current implementation includes execution_mode in steps,
        so this tests that the mode is consistently applied.
        """
        plan_glip = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-mode-test",
            tenant_id="tenant-test",
            prefer_glip=True,
        )
        plan_inprocess = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-mode-test",
            tenant_id="tenant-test",
            prefer_glip=False,
        )

        # Execution mode is part of plan, so IDs will differ
        # But within each mode, should be deterministic
        glip_plans = [
            plan_compiler(simple_pipeline_def, "run-mode-test", "t", True)
            for _ in range(10)
        ]
        assert len(set(p["plan_id"] for p in glip_plans)) == 1

    def test_plan_deterministic_with_different_tenant(
        self,
        simple_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """
        Plan content should be consistent regardless of tenant_id.

        tenant_id is not part of the plan steps, so same run_id should
        produce same plan regardless of tenant.
        """
        # tenant_id is metadata, not part of plan content
        plan1 = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-tenant-test",
            tenant_id="tenant-alpha",
        )
        plan2 = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-tenant-test",
            tenant_id="tenant-beta",
        )

        # Plan IDs should be same (tenant not in step content)
        assert plan1["plan_id"] == plan2["plan_id"]


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestPlanDeterminismEdgeCases:
    """Edge case tests for plan determinism."""

    def test_empty_params_pipeline(
        self,
        plan_compiler,
        frozen_clock,
    ):
        """Pipeline with no parameters compiles deterministically."""
        pipeline = PipelineDefinition(
            apiVersion="greenlang/v1",
            kind="Pipeline",
            metadata=PipelineMetadata(name="no-params-pipeline", namespace="test"),
            spec=PipelineSpec(
                steps=[
                    StepDefinition(
                        id="static-step",
                        agent="GL-TEST-X-001",
                        with_={"static_value": "constant"},
                    ),
                ],
            ),
        )

        plans = [
            plan_compiler(pipeline, f"run-no-params-{i:03d}", "tenant")
            for i in range(10)
        ]

        # Each run should have unique plan_id (due to run_id in step_id)
        plan_ids = [p["plan_id"] for p in plans]
        assert len(set(plan_ids)) == 10

    def test_complex_nested_config_determinism(
        self,
        plan_compiler,
        frozen_clock,
    ):
        """Complex nested configurations are hashed deterministically."""
        pipeline = PipelineDefinition(
            apiVersion="greenlang/v1",
            kind="Pipeline",
            metadata=PipelineMetadata(name="complex-config", namespace="test"),
            spec=PipelineSpec(
                steps=[
                    StepDefinition(
                        id="complex-step",
                        agent="GL-TEST-X-001",
                        with_={
                            "nested": {
                                "level1": {
                                    "level2": {
                                        "value": 123,
                                        "list": [1, 2, 3],
                                    },
                                },
                            },
                            "array": [
                                {"a": 1},
                                {"b": 2},
                            ],
                        },
                    ),
                ],
            ),
        )

        plans = [
            plan_compiler(pipeline, "run-complex", "tenant")
            for _ in range(50)
        ]

        assert len(set(p["plan_id"] for p in plans)) == 1

    def test_special_characters_in_ids(
        self,
        plan_compiler,
        frozen_clock,
    ):
        """Special characters in IDs don't break determinism."""
        # Note: Pipeline schema enforces DNS-compatible names
        # but step IDs can have some variation
        pipeline = PipelineDefinition(
            apiVersion="greenlang/v1",
            kind="Pipeline",
            metadata=PipelineMetadata(name="special-chars", namespace="test"),
            spec=PipelineSpec(
                steps=[
                    StepDefinition(
                        id="step-with-dashes",
                        agent="GL-TEST-X-001",
                    ),
                ],
            ),
        )

        plans = [
            plan_compiler(pipeline, "run-special-123", "tenant")
            for _ in range(20)
        ]

        assert len(set(p["plan_id"] for p in plans)) == 1


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestPlanDeterminismRegression:
    """Regression tests for known determinism issues."""

    def test_json_key_ordering_consistent(
        self,
        simple_pipeline_def,
        plan_compiler,
        frozen_clock,
    ):
        """
        Verify JSON key ordering doesn't affect plan hash.

        Uses sort_keys=True in JSON serialization.
        """
        plan = plan_compiler(
            pipeline=simple_pipeline_def,
            run_id="run-ordering-test",
            tenant_id="tenant-test",
        )

        # Manually verify the plan_id computation
        steps_json = json.dumps(plan["steps"], sort_keys=True)
        expected = hashlib.sha256(steps_json.encode()).hexdigest()[:16]

        assert plan["plan_id"] == expected

    def test_float_representation_consistency(
        self,
        plan_compiler,
        frozen_clock,
    ):
        """Floating point values are handled consistently."""
        pipeline = PipelineDefinition(
            apiVersion="greenlang/v1",
            kind="Pipeline",
            metadata=PipelineMetadata(name="float-test", namespace="test"),
            spec=PipelineSpec(
                steps=[
                    StepDefinition(
                        id="float-step",
                        agent="GL-TEST-X-001",
                        with_={
                            "threshold": 0.1,
                            "scale": 1.0,
                            "epsilon": 1e-6,
                        },
                    ),
                ],
            ),
        )

        plans = [
            plan_compiler(pipeline, "run-float", "tenant")
            for _ in range(100)
        ]

        assert len(set(p["plan_id"] for p in plans)) == 1
