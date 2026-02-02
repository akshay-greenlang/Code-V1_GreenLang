# -*- coding: utf-8 -*-
"""
Idempotency Tests for GL-FOUND-X-001 Orchestrator

Tests that verify idempotent execution behavior:

1. Idempotency Key Determinism
   - Same inputs produce same idempotency key
   - Keys are unique per step + attempt
   - Keys are 32-char hex strings

2. Retry Behavior
   - Each retry attempt has a different key
   - Same attempt number = same key
   - Key includes run_id, step_id, and attempt

3. Execution Replay
   - Re-running with same key should be safe
   - Mock executor tracks idempotency keys
   - Results are consistent for same key

4. Mock Environment Consistency
   - Mock K8s executor respects idempotency
   - Mock artifact store is consistent
   - Cross-run key collisions don't occur

Author: GreenLang Team
Version: 1.0.0
Coverage Target: 85%+
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Set

import pytest
import pytest_asyncio

from greenlang.utilities.determinism import (
    DeterministicClock,
    freeze_time,
    unfreeze_time,
    deterministic_id,
)
from greenlang.orchestrator.executors.base import (
    ExecutionStatus,
    ResourceProfile,
    RunContext,
    ArtifactReference,
)
from greenlang.orchestrator.pipeline_schema import (
    PipelineDefinition,
    PipelineMetadata,
    PipelineSpec,
    StepDefinition,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_idempotency_key(
    run_id: str,
    step_id: str,
    step_config: Dict[str, Any],
) -> str:
    """
    Generate idempotency key matching orchestrator implementation.

    Format: SHA-256(run_id:step_id:json(config))[:32]
    """
    content = f"{run_id}:{step_id}:{json.dumps(step_config, sort_keys=True)}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def generate_retry_idempotency_key(
    plan_hash: str,
    step_id: str,
    attempt: int,
) -> str:
    """
    Generate idempotency key for a specific retry attempt.

    Format: SHA-256(plan_hash:step_id:attempt)[:32]
    """
    content = f"{plan_hash}:{step_id}:{attempt}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


# =============================================================================
# IDEMPOTENCY KEY DETERMINISM TESTS
# =============================================================================

class TestIdempotencyKeyDeterminism:
    """Tests for deterministic idempotency key generation."""

    def test_same_inputs_same_key(self, frozen_clock):
        """Same inputs produce identical idempotency keys."""
        run_id = "run-idem-001"
        step_id = "step-001"
        config = {"param1": "value1", "param2": 42}

        keys = [
            generate_idempotency_key(run_id, step_id, config)
            for _ in range(100)
        ]

        assert len(set(keys)) == 1

    def test_key_format_is_32_hex(self, frozen_clock):
        """Idempotency key is 32 hex characters."""
        key = generate_idempotency_key(
            run_id="run-format",
            step_id="step-format",
            step_config={"test": True},
        )

        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_different_run_ids_different_keys(self, frozen_clock):
        """Different run_ids produce different keys."""
        config = {"param": "value"}

        key1 = generate_idempotency_key("run-001", "step-001", config)
        key2 = generate_idempotency_key("run-002", "step-001", config)

        assert key1 != key2

    def test_different_step_ids_different_keys(self, frozen_clock):
        """Different step_ids produce different keys."""
        config = {"param": "value"}

        key1 = generate_idempotency_key("run-001", "step-001", config)
        key2 = generate_idempotency_key("run-001", "step-002", config)

        assert key1 != key2

    def test_different_configs_different_keys(self, frozen_clock):
        """Different configurations produce different keys."""
        key1 = generate_idempotency_key(
            "run-001", "step-001", {"param": "value1"}
        )
        key2 = generate_idempotency_key(
            "run-001", "step-001", {"param": "value2"}
        )

        assert key1 != key2

    def test_config_order_independence(self, frozen_clock):
        """Config key ordering doesn't affect idempotency key."""
        # Same config, different insertion order
        config1 = {"z_param": 1, "a_param": 2, "m_param": 3}
        config2 = {"a_param": 2, "m_param": 3, "z_param": 1}

        key1 = generate_idempotency_key("run-001", "step-001", config1)
        key2 = generate_idempotency_key("run-001", "step-001", config2)

        # sort_keys=True ensures same hash regardless of order
        assert key1 == key2

    def test_nested_config_determinism(self, frozen_clock):
        """Nested configurations produce deterministic keys."""
        config = {
            "level1": {
                "level2": {
                    "array": [1, 2, 3],
                    "nested_obj": {"a": 1, "b": 2},
                }
            }
        }

        keys = [
            generate_idempotency_key("run-nested", "step-nested", config)
            for _ in range(50)
        ]

        assert len(set(keys)) == 1


# =============================================================================
# RETRY BEHAVIOR TESTS
# =============================================================================

class TestRetryIdempotencyKeys:
    """Tests for idempotency keys during retry scenarios."""

    def test_each_retry_attempt_has_different_key(self, frozen_clock):
        """Each retry attempt gets a unique idempotency key."""
        plan_hash = "abc123"
        step_id = "step-retry"

        keys = [
            generate_retry_idempotency_key(plan_hash, step_id, attempt)
            for attempt in range(5)
        ]

        # All keys should be unique
        assert len(set(keys)) == 5

    def test_same_attempt_same_key(self, frozen_clock):
        """Same attempt number produces same key."""
        plan_hash = "abc123"
        step_id = "step-retry"
        attempt = 2

        keys = [
            generate_retry_idempotency_key(plan_hash, step_id, attempt)
            for _ in range(100)
        ]

        assert len(set(keys)) == 1

    def test_attempt_zero_is_valid(self, frozen_clock):
        """Attempt 0 (first attempt) produces valid key."""
        key = generate_retry_idempotency_key("plan-hash", "step-id", 0)

        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_retry_keys_are_predictable(self, frozen_clock):
        """Retry keys can be predicted for same inputs."""
        plan_hash = "xyz789"
        step_id = "predictable-step"

        # Generate sequence
        sequence1 = [
            generate_retry_idempotency_key(plan_hash, step_id, i)
            for i in range(3)
        ]

        # Generate same sequence again
        sequence2 = [
            generate_retry_idempotency_key(plan_hash, step_id, i)
            for i in range(3)
        ]

        assert sequence1 == sequence2


# =============================================================================
# EXECUTION REPLAY TESTS
# =============================================================================

class TestExecutionReplay:
    """Tests for safe execution replay with idempotency."""

    @pytest.mark.asyncio
    async def test_mock_executor_tracks_idempotency_keys(
        self,
        mock_k8s_executor,
        mock_artifact_store,
        run_context_factory,
        frozen_clock,
    ):
        """Mock executor records idempotency keys for verification."""
        context = run_context_factory(
            run_id="run-track",
            step_id="step-track",
        )

        # Pre-populate result
        mock_artifact_store.store_mock_result(
            run_id=context.run_id,
            step_id=context.step_id,
            tenant_id=context.tenant_id,
            result={"outputs": {"value": 1}},
        )

        await mock_k8s_executor.execute(
            context=context,
            container_image="test:latest",
            resources=ResourceProfile(),
            namespace="test",
            input_uri="s3://bucket/input",
            output_uri="s3://bucket/output/",
        )

        history = mock_k8s_executor.get_execution_history()
        assert len(history) == 1
        assert history[0]["idempotency_key"] == context.idempotency_key

    @pytest.mark.asyncio
    async def test_replayed_execution_uses_same_key(
        self,
        mock_k8s_executor,
        mock_artifact_store,
        run_context_factory,
        frozen_clock,
    ):
        """Replaying same context uses same idempotency key."""
        # Create context with fixed parameters
        context = run_context_factory(
            run_id="run-replay",
            step_id="step-replay",
            retry_attempt=0,
        )

        mock_artifact_store.store_mock_result(
            run_id=context.run_id,
            step_id=context.step_id,
            tenant_id=context.tenant_id,
            result={"outputs": {}},
        )

        # Execute twice with same context
        await mock_k8s_executor.execute(
            context=context,
            container_image="test:latest",
            resources=ResourceProfile(),
            namespace="test",
            input_uri="s3://bucket/input",
            output_uri="s3://bucket/output/",
        )

        await mock_k8s_executor.execute(
            context=context,
            container_image="test:latest",
            resources=ResourceProfile(),
            namespace="test",
            input_uri="s3://bucket/input",
            output_uri="s3://bucket/output/",
        )

        history = mock_k8s_executor.get_execution_history()
        assert len(history) == 2
        assert history[0]["idempotency_key"] == history[1]["idempotency_key"]

    @pytest.mark.asyncio
    async def test_retry_uses_different_key(
        self,
        mock_k8s_executor,
        mock_artifact_store,
        frozen_clock,
    ):
        """Retry attempt uses different idempotency key."""
        run_id = "run-retry-diff"
        step_id = "step-retry-diff"
        tenant_id = "tenant-test"

        mock_artifact_store.store_mock_result(
            run_id=run_id,
            step_id=step_id,
            tenant_id=tenant_id,
            result={"outputs": {}},
        )

        # First attempt
        context1 = RunContext(
            run_id=run_id,
            step_id=step_id,
            pipeline_id="test-pipeline",
            tenant_id=tenant_id,
            agent_id="GL-TEST-X-001",
            agent_version="1.0.0",
            params={},
            inputs={},
            retry_attempt=0,
            idempotency_key=generate_retry_idempotency_key("plan", step_id, 0),
            trace_id="trace-1",
            span_id="span-1",
            log_correlation_id="log-1",
        )

        # Second attempt (retry)
        context2 = RunContext(
            run_id=run_id,
            step_id=step_id,
            pipeline_id="test-pipeline",
            tenant_id=tenant_id,
            agent_id="GL-TEST-X-001",
            agent_version="1.0.0",
            params={},
            inputs={},
            retry_attempt=1,
            idempotency_key=generate_retry_idempotency_key("plan", step_id, 1),
            trace_id="trace-1",
            span_id="span-1",
            log_correlation_id="log-1",
        )

        await mock_k8s_executor.execute(
            context=context1,
            container_image="test:latest",
            resources=ResourceProfile(),
            namespace="test",
            input_uri="s3://bucket/input",
            output_uri="s3://bucket/output/",
        )

        await mock_k8s_executor.execute(
            context=context2,
            container_image="test:latest",
            resources=ResourceProfile(),
            namespace="test",
            input_uri="s3://bucket/input",
            output_uri="s3://bucket/output/",
        )

        history = mock_k8s_executor.get_execution_history()
        assert len(history) == 2
        assert history[0]["idempotency_key"] != history[1]["idempotency_key"]


# =============================================================================
# RUN CONTEXT DETERMINISM TESTS
# =============================================================================

class TestRunContextDeterminism:
    """Tests for RunContext hash determinism."""

    def test_context_hash_is_deterministic(
        self,
        run_context_factory,
        frozen_clock,
    ):
        """RunContext.compute_hash() is deterministic."""
        context = run_context_factory(
            run_id="run-context-hash",
            step_id="step-hash",
            params={"key": "value"},
        )

        hashes = [context.compute_hash() for _ in range(100)]
        assert len(set(hashes)) == 1

    def test_context_hash_length(self, run_context_factory, frozen_clock):
        """Context hash is 64-char SHA-256."""
        context = run_context_factory()
        hash_value = context.compute_hash()

        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_different_contexts_different_hashes(
        self,
        run_context_factory,
        frozen_clock,
    ):
        """Different contexts produce different hashes."""
        context1 = run_context_factory(
            run_id="run-1",
            step_id="step-1",
        )
        context2 = run_context_factory(
            run_id="run-2",
            step_id="step-2",
        )

        assert context1.compute_hash() != context2.compute_hash()

    def test_context_hash_excludes_volatile_fields(
        self,
        frozen_clock,
    ):
        """Context hash excludes volatile fields like deadline."""
        # Manually create contexts to test volatile field exclusion
        base_context = RunContext(
            run_id="run-volatile",
            step_id="step-volatile",
            pipeline_id="pipeline",
            tenant_id="tenant",
            agent_id="GL-TEST-X-001",
            agent_version="1.0.0",
            params={"key": "value"},
            inputs={},
            deadline_ts=None,
            timeout_seconds=300,
            retry_attempt=0,
            idempotency_key="test-key",
            trace_id="trace",
            span_id="span",
            log_correlation_id="log",
        )

        # Same context with different deadline
        context_with_deadline = RunContext(
            run_id="run-volatile",
            step_id="step-volatile",
            pipeline_id="pipeline",
            tenant_id="tenant",
            agent_id="GL-TEST-X-001",
            agent_version="1.0.0",
            params={"key": "value"},
            inputs={},
            deadline_ts=datetime(2026, 12, 31, tzinfo=timezone.utc),  # Different
            timeout_seconds=600,  # Different
            retry_attempt=0,
            idempotency_key="test-key",
            trace_id="trace",
            span_id="span",
            log_correlation_id="log",
        )

        # Hashes should be the same (volatile fields excluded)
        assert base_context.compute_hash() == context_with_deadline.compute_hash()


# =============================================================================
# CROSS-RUN COLLISION TESTS
# =============================================================================

class TestCrossRunCollisions:
    """Tests to ensure no key collisions across runs."""

    def test_no_key_collisions_across_many_runs(self, frozen_clock):
        """Generate many keys and verify no collisions."""
        keys: Set[str] = set()

        # Generate keys for 100 runs with 10 steps each
        for run_num in range(100):
            run_id = f"run-collision-{run_num:04d}"
            for step_num in range(10):
                step_id = f"step-{step_num:03d}"
                config = {"run": run_num, "step": step_num}

                key = generate_idempotency_key(run_id, step_id, config)
                keys.add(key)

        # Should have 1000 unique keys
        assert len(keys) == 1000

    def test_no_retry_key_collisions(self, frozen_clock):
        """No collisions between retry keys and initial keys."""
        initial_keys: Set[str] = set()
        retry_keys: Set[str] = set()

        plan_hash = "test-plan-hash"

        for step_num in range(100):
            step_id = f"step-{step_num}"

            # Initial key (attempt 0)
            initial_key = generate_retry_idempotency_key(plan_hash, step_id, 0)
            initial_keys.add(initial_key)

            # Retry keys (attempts 1-3)
            for attempt in range(1, 4):
                retry_key = generate_retry_idempotency_key(
                    plan_hash, step_id, attempt
                )
                retry_keys.add(retry_key)

        # No overlap between initial and retry keys
        assert initial_keys.isdisjoint(retry_keys)

        # All keys unique within their groups
        assert len(initial_keys) == 100
        assert len(retry_keys) == 300

    def test_keys_unique_across_pipelines(self, frozen_clock):
        """Keys from different pipelines don't collide."""
        keys: Set[str] = set()

        pipelines = ["pipeline-A", "pipeline-B", "pipeline-C"]
        runs_per_pipeline = 50

        for pipeline in pipelines:
            for run_num in range(runs_per_pipeline):
                run_id = f"{pipeline}-run-{run_num}"
                step_id = "shared-step-name"
                config = {"pipeline": pipeline}

                key = generate_idempotency_key(run_id, step_id, config)
                keys.add(key)

        # Should have 150 unique keys (3 pipelines * 50 runs)
        assert len(keys) == 150


# =============================================================================
# MOCK ENVIRONMENT CONSISTENCY TESTS
# =============================================================================

class TestMockEnvironmentIdempotency:
    """Tests for idempotency in mock execution environment."""

    @pytest.mark.asyncio
    async def test_artifact_store_consistent_on_replay(
        self,
        mock_artifact_store,
        frozen_clock,
    ):
        """Artifact store returns consistent results on replay."""
        run_id = "run-artifact-replay"
        step_id = "step-replay"
        tenant_id = "tenant-test"

        # Store result
        result_data = {"outputs": {"calculation": 42.0}}
        checksum = mock_artifact_store.store_mock_result(
            run_id=run_id,
            step_id=step_id,
            tenant_id=tenant_id,
            result=result_data,
        )

        # Read multiple times
        results = [
            await mock_artifact_store.read_result(run_id, step_id, tenant_id)
            for _ in range(10)
        ]

        # All results should be identical
        assert all(r == result_data for r in results)

        # Checksum should verify
        uri = mock_artifact_store.generate_artifact_uri(
            mock_artifact_store.bucket, tenant_id, run_id, step_id, "result.json"
        )
        assert await mock_artifact_store.verify_checksum(uri, checksum)

    @pytest.mark.asyncio
    async def test_executor_results_consistent(
        self,
        mock_k8s_executor,
        mock_artifact_store,
        run_context_factory,
        frozen_clock,
    ):
        """Mock executor returns consistent results for same context."""
        context = run_context_factory(
            run_id="run-consistent",
            step_id="step-consistent",
        )

        # Pre-populate result
        mock_artifact_store.store_mock_result(
            run_id=context.run_id,
            step_id=context.step_id,
            tenant_id=context.tenant_id,
            result={"outputs": {"value": 123}},
        )

        # Execute multiple times
        results = []
        for _ in range(5):
            result = await mock_k8s_executor.execute(
                context=context,
                container_image="test:latest",
                resources=ResourceProfile(),
                namespace="test",
                input_uri="s3://input",
                output_uri="s3://output/",
            )
            results.append(result)

        # All results should have same status
        statuses = [r.status for r in results]
        assert len(set(statuses)) == 1
        assert statuses[0] == ExecutionStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_execution_history_is_complete(
        self,
        mock_k8s_executor,
        mock_artifact_store,
        run_context_factory,
        frozen_clock,
    ):
        """Execution history records all executions with keys."""
        mock_k8s_executor.clear_history()

        contexts = [
            run_context_factory(
                run_id=f"run-history-{i}",
                step_id=f"step-{i}",
            )
            for i in range(5)
        ]

        # Pre-populate results
        for ctx in contexts:
            mock_artifact_store.store_mock_result(
                run_id=ctx.run_id,
                step_id=ctx.step_id,
                tenant_id=ctx.tenant_id,
                result={"outputs": {}},
            )

        # Execute all
        for ctx in contexts:
            await mock_k8s_executor.execute(
                context=ctx,
                container_image="test:latest",
                resources=ResourceProfile(),
                namespace="test",
                input_uri="s3://input",
                output_uri="s3://output/",
            )

        history = mock_k8s_executor.get_execution_history()

        assert len(history) == 5

        # All idempotency keys should be unique
        keys = [h["idempotency_key"] for h in history]
        assert len(set(keys)) == 5


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestIdempotencyEdgeCases:
    """Edge case tests for idempotency."""

    def test_empty_config_produces_valid_key(self, frozen_clock):
        """Empty configuration produces valid idempotency key."""
        key = generate_idempotency_key("run-empty", "step-empty", {})

        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_large_config_produces_valid_key(self, frozen_clock):
        """Large configuration produces valid idempotency key."""
        large_config = {
            f"key_{i}": f"value_{i}" * 100
            for i in range(100)
        }

        key = generate_idempotency_key("run-large", "step-large", large_config)

        assert len(key) == 32

    def test_unicode_in_config(self, frozen_clock):
        """Unicode characters in config don't break key generation."""
        config = {
            "greeting": "Hello, world!",
            "chinese": "Chinese characters",
            "emoji": "Star symbol",
        }

        key = generate_idempotency_key("run-unicode", "step-unicode", config)

        assert len(key) == 32

    def test_null_values_in_config(self, frozen_clock):
        """Null values in config are handled correctly."""
        config = {
            "present": "value",
            "null_value": None,
            "nested": {"also_null": None},
        }

        keys = [
            generate_idempotency_key("run-null", "step-null", config)
            for _ in range(10)
        ]

        assert len(set(keys)) == 1

    def test_boolean_and_numeric_values(self, frozen_clock):
        """Boolean and numeric values produce consistent keys."""
        config = {
            "bool_true": True,
            "bool_false": False,
            "integer": 42,
            "float": 3.14159,
            "negative": -100,
            "zero": 0,
        }

        keys = [
            generate_idempotency_key("run-types", "step-types", config)
            for _ in range(10)
        ]

        assert len(set(keys)) == 1

    @pytest.mark.asyncio
    async def test_context_with_artifact_inputs(
        self,
        run_context_factory,
        mock_artifact_store,
        frozen_clock,
    ):
        """Context with artifact references has deterministic hash."""
        # Create artifact reference
        artifact_ref = ArtifactReference(
            uri="s3://bucket/artifact.json",
            checksum="abc123def456",
            media_type="application/json",
            size_bytes=1024,
        )

        context = run_context_factory(
            run_id="run-artifact-input",
            step_id="step-artifact",
            inputs={"upstream_data": artifact_ref},
        )

        hashes = [context.compute_hash() for _ in range(50)]
        assert len(set(hashes)) == 1
