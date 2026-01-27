# -*- coding: utf-8 -*-
"""
P1 Feature Tests for GL-FOUND-X-001 Orchestrator

Tests for:
1. Approval Gates - Human approval workflow for sensitive steps
2. Dynamic Fan-Out - Parallel execution of dynamically generated items
3. Concurrency Controls - Rate limiting and parallel execution limits

Author: GreenLang Team
Version: 1.0.0
Coverage Target: 85%+
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from greenlang.utilities.determinism import (
    DeterministicClock,
    freeze_time,
    unfreeze_time,
)
from greenlang.orchestrator.glip_orchestrator import (
    ApprovalStatus,
    ApprovalRequest,
    ConcurrencyConfig,
    FanOutSpec,
    GLIPExecutionMode,
    GLIPOrchestrator,
    GLIPOrchestratorConfig,
    GLIPRunConfig,
)
from greenlang.agents.base import AgentResult


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def frozen_clock():
    """Freeze time for deterministic tests."""
    freeze_time(datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc))
    yield DeterministicClock.now()
    unfreeze_time()


@pytest.fixture
def orchestrator_config():
    """Create test orchestrator configuration."""
    return GLIPOrchestratorConfig(
        k8s_namespace="test-namespace",
        s3_bucket="test-bucket",
        max_parallel_steps=5,
    )


@pytest.fixture
def run_config():
    """Create test run configuration."""
    return GLIPRunConfig(
        pipeline_id="test-pipeline",
        tenant_id="tenant-001",
        user_id="user-001",
        enable_approval_gates=True,
        approval_timeout_seconds=3600,
        max_fan_out_items=50,
        concurrency=ConcurrencyConfig(
            max_concurrent_runs=10,
            max_concurrent_steps_per_run=5,
            rate_limit_requests_per_minute=100,
        ),
    )


@pytest_asyncio.fixture
async def orchestrator(orchestrator_config):
    """Create test orchestrator instance."""
    orch = GLIPOrchestrator(orchestrator_config)
    # Don't initialize backends for unit tests
    yield orch


# =============================================================================
# APPROVAL GATES TESTS
# =============================================================================

class TestApprovalGates:
    """Tests for human approval gate functionality."""

    @pytest.mark.asyncio
    async def test_request_approval_creates_pending_request(
        self,
        orchestrator,
        run_config,
        frozen_clock,
    ):
        """Requesting approval creates a pending approval request."""
        run_id = "run-approval-001"
        step_id = "step-sensitive"
        reason = "This step modifies production data"

        request = await orchestrator.request_approval(
            run_id=run_id,
            step_id=step_id,
            reason=reason,
            run_config=run_config,
        )

        assert request.status == ApprovalStatus.PENDING
        assert request.run_id == run_id
        assert request.step_id == step_id
        assert request.reason == reason
        assert request.tenant_id == run_config.tenant_id
        assert request.request_id.startswith("approval-")

    @pytest.mark.asyncio
    async def test_request_approval_sets_expiration(
        self,
        orchestrator,
        run_config,
        frozen_clock,
    ):
        """Approval request has correct expiration time."""
        request = await orchestrator.request_approval(
            run_id="run-expire",
            step_id="step-expire",
            reason="Test expiration",
            run_config=run_config,
        )

        expected_expiration = frozen_clock + timedelta(
            seconds=run_config.approval_timeout_seconds
        )
        assert request.expires_at == expected_expiration

    @pytest.mark.asyncio
    async def test_process_approval_approves_request(
        self,
        orchestrator,
        run_config,
        frozen_clock,
    ):
        """Processing approval with approved=True marks request as approved."""
        request = await orchestrator.request_approval(
            run_id="run-approve",
            step_id="step-approve",
            reason="Test approval",
            run_config=run_config,
        )

        updated = await orchestrator.process_approval(
            request_id=request.request_id,
            approved=True,
            approved_by="admin@greenlang.io",
            comment="Looks good!",
        )

        assert updated.status == ApprovalStatus.APPROVED
        assert updated.approved_by == "admin@greenlang.io"
        assert updated.approved_at is not None
        assert updated.metadata.get("approval_comment") == "Looks good!"

    @pytest.mark.asyncio
    async def test_process_approval_rejects_request(
        self,
        orchestrator,
        run_config,
        frozen_clock,
    ):
        """Processing approval with approved=False marks request as rejected."""
        request = await orchestrator.request_approval(
            run_id="run-reject",
            step_id="step-reject",
            reason="Test rejection",
            run_config=run_config,
        )

        updated = await orchestrator.process_approval(
            request_id=request.request_id,
            approved=False,
            approved_by="security@greenlang.io",
            comment="Not authorized for this operation",
        )

        assert updated.status == ApprovalStatus.REJECTED
        assert updated.approved_by == "security@greenlang.io"

    @pytest.mark.asyncio
    async def test_process_approval_invalid_request_raises(
        self,
        orchestrator,
        frozen_clock,
    ):
        """Processing non-existent approval raises ValueError."""
        with pytest.raises(ValueError, match="Approval request not found"):
            await orchestrator.process_approval(
                request_id="invalid-request-id",
                approved=True,
                approved_by="admin@greenlang.io",
            )

    @pytest.mark.asyncio
    async def test_wait_for_approval_returns_true_on_approve(
        self,
        orchestrator,
        run_config,
        frozen_clock,
    ):
        """wait_for_approval returns True when approved."""
        request = await orchestrator.request_approval(
            run_id="run-wait-approve",
            step_id="step-wait",
            reason="Test wait",
            run_config=run_config,
        )

        # Approve immediately
        request.status = ApprovalStatus.APPROVED

        result = await orchestrator.wait_for_approval(
            request=request,
            poll_interval_seconds=0.01,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_approval_returns_false_on_reject(
        self,
        orchestrator,
        run_config,
        frozen_clock,
    ):
        """wait_for_approval returns False when rejected."""
        request = await orchestrator.request_approval(
            run_id="run-wait-reject",
            step_id="step-wait",
            reason="Test wait reject",
            run_config=run_config,
        )

        # Reject immediately
        request.status = ApprovalStatus.REJECTED

        result = await orchestrator.wait_for_approval(
            request=request,
            poll_interval_seconds=0.01,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_approval_with_metadata(
        self,
        orchestrator,
        run_config,
        frozen_clock,
    ):
        """Approval request can include custom metadata."""
        metadata = {
            "affected_records": 1000,
            "data_classification": "sensitive",
            "requires_two_person_approval": True,
        }

        request = await orchestrator.request_approval(
            run_id="run-metadata",
            step_id="step-metadata",
            reason="Test with metadata",
            run_config=run_config,
            metadata=metadata,
        )

        assert request.metadata["affected_records"] == 1000
        assert request.metadata["data_classification"] == "sensitive"
        assert request.metadata["requires_two_person_approval"] is True


# =============================================================================
# DYNAMIC FAN-OUT TESTS
# =============================================================================

class TestDynamicFanOut:
    """Tests for dynamic fan-out execution."""

    @pytest.fixture
    def fan_out_spec(self):
        """Create test fan-out specification."""
        return FanOutSpec(
            source_field="items",
            item_param_name="item",
            max_parallel=3,
            continue_on_failure=False,
            aggregate_results=True,
        )

    @pytest.fixture
    def mock_execution_context(self):
        """Create mock execution context."""
        context = MagicMock()
        context.execution_id = "run-fanout"
        context.pipeline = MagicMock()
        context.pipeline.pipeline_id = "test-pipeline"
        return context

    @pytest.fixture
    def mock_node(self):
        """Create mock agent node."""
        node = MagicMock()
        node.agent_id = "processor"
        node.agent_type = "GL-PROCESS-X-001"
        return node

    @pytest.mark.asyncio
    async def test_fan_out_processes_all_items(
        self,
        orchestrator,
        run_config,
        fan_out_spec,
        mock_execution_context,
        mock_node,
        frozen_clock,
    ):
        """Fan-out processes all items in the source field."""
        inputs = {"items": [1, 2, 3, 4, 5]}
        step = {
            "step_id": "step-fanout",
            "execution_mode": GLIPExecutionMode.IN_PROCESS.value,
            "idempotency_key": "key-123",
        }

        # Mock _execute_in_process to return success
        async def mock_execute(*args, **kwargs):
            return AgentResult(success=True, data={"processed": True})

        orchestrator._execute_in_process = mock_execute

        result = await orchestrator.execute_fan_out(
            context=mock_execution_context,
            node=mock_node,
            step=step,
            inputs=inputs,
            run_config=run_config,
            fan_out_spec=fan_out_spec,
        )

        assert result.success is True
        assert result.data["total_items"] == 5
        assert result.data["successful_items"] == 5
        assert result.data["failed_items"] == 0

    @pytest.mark.asyncio
    async def test_fan_out_respects_max_parallel(
        self,
        orchestrator,
        run_config,
        mock_execution_context,
        mock_node,
        frozen_clock,
    ):
        """Fan-out respects max_parallel configuration."""
        fan_out_spec = FanOutSpec(
            source_field="items",
            max_parallel=2,  # Only 2 at a time
        )

        inputs = {"items": list(range(10))}
        step = {
            "step_id": "step-parallel",
            "execution_mode": GLIPExecutionMode.IN_PROCESS.value,
            "idempotency_key": "key-parallel",
        }

        execution_times = []

        async def mock_execute(*args, **kwargs):
            execution_times.append(DeterministicClock.now())
            await asyncio.sleep(0.01)  # Simulate work
            return AgentResult(success=True, data={})

        orchestrator._execute_in_process = mock_execute

        await orchestrator.execute_fan_out(
            context=mock_execution_context,
            node=mock_node,
            step=step,
            inputs=inputs,
            run_config=run_config,
            fan_out_spec=fan_out_spec,
        )

        # Should have been called 10 times
        assert len(execution_times) == 10

    @pytest.mark.asyncio
    async def test_fan_out_stops_on_failure_by_default(
        self,
        orchestrator,
        run_config,
        fan_out_spec,
        mock_execution_context,
        mock_node,
        frozen_clock,
    ):
        """Fan-out stops on first failure when continue_on_failure=False."""
        fan_out_spec.continue_on_failure = False

        inputs = {"items": [1, 2, 3]}
        step = {
            "step_id": "step-fail",
            "execution_mode": GLIPExecutionMode.IN_PROCESS.value,
            "idempotency_key": "key-fail",
        }

        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return AgentResult(success=False, error="Simulated failure")
            return AgentResult(success=True, data={})

        orchestrator._execute_in_process = mock_execute

        result = await orchestrator.execute_fan_out(
            context=mock_execution_context,
            node=mock_node,
            step=step,
            inputs=inputs,
            run_config=run_config,
            fan_out_spec=fan_out_spec,
        )

        assert result.success is False
        assert "Simulated failure" in result.error

    @pytest.mark.asyncio
    async def test_fan_out_continues_on_failure_when_configured(
        self,
        orchestrator,
        run_config,
        mock_execution_context,
        mock_node,
        frozen_clock,
    ):
        """Fan-out continues after failure when continue_on_failure=True."""
        fan_out_spec = FanOutSpec(
            source_field="items",
            max_parallel=5,
            continue_on_failure=True,
        )

        inputs = {"items": [1, 2, 3, 4, 5]}
        step = {
            "step_id": "step-continue",
            "execution_mode": GLIPExecutionMode.IN_PROCESS.value,
            "idempotency_key": "key-continue",
        }

        async def mock_execute(context, node, step, inputs):
            item = inputs.get("item")
            if item == 3:  # Fail on item 3
                return AgentResult(success=False, error="Item 3 failed")
            return AgentResult(success=True, data={"item": item})

        orchestrator._execute_in_process = mock_execute

        result = await orchestrator.execute_fan_out(
            context=mock_execution_context,
            node=mock_node,
            step=step,
            inputs=inputs,
            run_config=run_config,
            fan_out_spec=fan_out_spec,
        )

        # Should still succeed overall
        assert result.success is True
        assert result.data["successful_items"] == 4
        assert result.data["failed_items"] == 1
        assert len(result.data["errors"]) == 1

    @pytest.mark.asyncio
    async def test_fan_out_rejects_exceeding_limit(
        self,
        orchestrator,
        run_config,
        fan_out_spec,
        mock_execution_context,
        mock_node,
        frozen_clock,
    ):
        """Fan-out rejects requests exceeding max_fan_out_items."""
        run_config.max_fan_out_items = 5

        # Try to fan out 10 items
        inputs = {"items": list(range(10))}
        step = {
            "step_id": "step-exceed",
            "execution_mode": GLIPExecutionMode.IN_PROCESS.value,
            "idempotency_key": "key-exceed",
        }

        result = await orchestrator.execute_fan_out(
            context=mock_execution_context,
            node=mock_node,
            step=step,
            inputs=inputs,
            run_config=run_config,
            fan_out_spec=fan_out_spec,
        )

        assert result.success is False
        assert "exceeds limit" in result.error


# =============================================================================
# CONCURRENCY CONTROLS TESTS
# =============================================================================

class TestConcurrencyControls:
    """Tests for concurrency control functionality."""

    @pytest.fixture
    def concurrency_config(self):
        """Create test concurrency configuration."""
        return ConcurrencyConfig(
            max_concurrent_runs=5,
            max_concurrent_steps_per_run=3,
            rate_limit_requests_per_minute=100,
            agent_concurrency_limits={"GL-EXPENSIVE-X-001": 2},
            queue_timeout_seconds=10,
            enable_fair_scheduling=True,
        )

    @pytest.mark.asyncio
    async def test_acquire_slot_succeeds_when_available(
        self,
        orchestrator,
        concurrency_config,
        frozen_clock,
    ):
        """Can acquire concurrency slot when capacity available."""
        result = await orchestrator.acquire_concurrency_slot(
            run_id="run-slot-001",
            agent_id="GL-TEST-X-001",
            tenant_id="tenant-001",
            concurrency_config=concurrency_config,
        )

        assert result is True
        assert "run-slot-001" in orchestrator._concurrency_slots

    @pytest.mark.asyncio
    async def test_acquire_slot_fails_at_capacity(
        self,
        orchestrator,
        concurrency_config,
        frozen_clock,
    ):
        """Cannot acquire slot when at max capacity."""
        # Fill up active runs
        for i in range(5):
            orchestrator._active_runs[f"run-{i}"] = {
                "config": GLIPRunConfig(
                    pipeline_id=f"pipeline-{i}",
                    tenant_id=f"tenant-{i}",
                ),
                "plan": {"steps": []},
            }

        result = await orchestrator.acquire_concurrency_slot(
            run_id="run-overflow",
            agent_id="GL-TEST-X-001",
            tenant_id="tenant-new",
            concurrency_config=concurrency_config,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_slot_respects_agent_limit(
        self,
        orchestrator,
        concurrency_config,
        frozen_clock,
    ):
        """Respects per-agent concurrency limits."""
        expensive_agent = "GL-EXPENSIVE-X-001"

        # Add runs using the expensive agent
        for i in range(2):
            orchestrator._active_runs[f"run-expensive-{i}"] = {
                "config": GLIPRunConfig(
                    pipeline_id=f"pipeline-{i}",
                    tenant_id="tenant-001",
                ),
                "plan": {"steps": [{"agent_type": expensive_agent}]},
            }

        # Try to add another run with the expensive agent
        result = await orchestrator.acquire_concurrency_slot(
            run_id="run-expensive-overflow",
            agent_id=expensive_agent,
            tenant_id="tenant-001",
            concurrency_config=concurrency_config,
        )

        assert result is False

    def test_release_slot_removes_tracking(
        self,
        orchestrator,
        frozen_clock,
    ):
        """Releasing slot removes tracking entry."""
        orchestrator._concurrency_slots["run-release"] = {
            "agent_id": "GL-TEST-X-001",
            "tenant_id": "tenant-001",
            "acquired_at": frozen_clock,
        }

        orchestrator.release_concurrency_slot("run-release")

        assert "run-release" not in orchestrator._concurrency_slots

    @pytest.mark.asyncio
    async def test_wait_for_slot_succeeds_when_available(
        self,
        orchestrator,
        concurrency_config,
        frozen_clock,
    ):
        """wait_for_concurrency_slot succeeds when slot available."""
        result = await orchestrator.wait_for_concurrency_slot(
            run_id="run-wait-slot",
            agent_id="GL-TEST-X-001",
            tenant_id="tenant-001",
            concurrency_config=concurrency_config,
            poll_interval_seconds=0.01,
        )

        assert result is True

    def test_get_concurrency_stats_returns_metrics(
        self,
        orchestrator,
        frozen_clock,
    ):
        """get_concurrency_stats returns correct metrics."""
        # Setup some active runs
        orchestrator._active_runs["run-1"] = {
            "config": GLIPRunConfig(
                pipeline_id="pipeline-1",
                tenant_id="tenant-A",
            ),
            "plan": {"steps": [{"agent_type": "GL-AGENT-1"}]},
        }
        orchestrator._active_runs["run-2"] = {
            "config": GLIPRunConfig(
                pipeline_id="pipeline-2",
                tenant_id="tenant-A",
            ),
            "plan": {"steps": [{"agent_type": "GL-AGENT-2"}]},
        }
        orchestrator._active_runs["run-3"] = {
            "config": GLIPRunConfig(
                pipeline_id="pipeline-3",
                tenant_id="tenant-B",
            ),
            "plan": {"steps": [{"agent_type": "GL-AGENT-1"}]},
        }

        stats = orchestrator.get_concurrency_stats()

        assert stats["active_runs"] == 3
        assert stats["runs_by_tenant"]["tenant-A"] == 2
        assert stats["runs_by_tenant"]["tenant-B"] == 1
        assert stats["runs_by_agent"]["GL-AGENT-1"] == 2
        assert stats["runs_by_agent"]["GL-AGENT-2"] == 1

    @pytest.mark.asyncio
    async def test_fair_scheduling_limits_tenant_burst(
        self,
        orchestrator,
        concurrency_config,
        frozen_clock,
    ):
        """Fair scheduling prevents one tenant from monopolizing resources."""
        # Fill up with runs from tenant-greedy
        for i in range(4):
            orchestrator._active_runs[f"run-greedy-{i}"] = {
                "config": GLIPRunConfig(
                    pipeline_id=f"pipeline-{i}",
                    tenant_id="tenant-greedy",
                ),
                "plan": {"steps": []},
            }

        # Try to add another run for the same tenant
        result = await orchestrator.acquire_concurrency_slot(
            run_id="run-greedy-5",
            agent_id="GL-TEST-X-001",
            tenant_id="tenant-greedy",
            concurrency_config=concurrency_config,
        )

        # Should be rejected due to fair scheduling
        assert result is False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestP1FeaturesIntegration:
    """Integration tests combining P1 features."""

    @pytest.mark.asyncio
    async def test_approval_gate_before_fan_out(
        self,
        orchestrator,
        run_config,
        frozen_clock,
    ):
        """Approval gate can protect fan-out operations."""
        # Request approval for a fan-out operation
        request = await orchestrator.request_approval(
            run_id="run-approve-fanout",
            step_id="step-bulk-delete",
            reason="About to delete 1000 records across 50 partitions",
            run_config=run_config,
            metadata={
                "operation": "bulk_delete",
                "affected_records": 1000,
                "partitions": 50,
            },
        )

        assert request.status == ApprovalStatus.PENDING
        assert request.metadata["operation"] == "bulk_delete"

        # Approve
        await orchestrator.process_approval(
            request_id=request.request_id,
            approved=True,
            approved_by="ops-lead@greenlang.io",
        )

        assert request.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_concurrency_with_fan_out(
        self,
        orchestrator,
        frozen_clock,
    ):
        """Concurrency controls apply to fan-out items."""
        concurrency_config = ConcurrencyConfig(
            max_concurrent_runs=100,
            max_concurrent_steps_per_run=5,  # Limit parallel items
        )

        # Acquire slot
        result = await orchestrator.acquire_concurrency_slot(
            run_id="run-fanout-concurrent",
            agent_id="GL-PROCESSOR-X-001",
            tenant_id="tenant-001",
            concurrency_config=concurrency_config,
        )

        assert result is True

        # Verify stats
        orchestrator._active_runs["run-fanout-concurrent"] = {
            "config": GLIPRunConfig(
                pipeline_id="pipeline-fanout",
                tenant_id="tenant-001",
            ),
            "plan": {"steps": [{"agent_type": "GL-PROCESSOR-X-001"}]},
        }

        stats = orchestrator.get_concurrency_stats()
        assert stats["active_runs"] == 1


# =============================================================================
# EDGE CASES
# =============================================================================

class TestP1EdgeCases:
    """Edge case tests for P1 features."""

    @pytest.mark.asyncio
    async def test_empty_fan_out_source(
        self,
        orchestrator,
        run_config,
        frozen_clock,
    ):
        """Fan-out with empty source completes successfully."""
        fan_out_spec = FanOutSpec(
            source_field="items",
            max_parallel=5,
        )

        context = MagicMock()
        context.execution_id = "run-empty"
        node = MagicMock()
        step = {
            "step_id": "step-empty",
            "execution_mode": GLIPExecutionMode.IN_PROCESS.value,
            "idempotency_key": "key-empty",
        }

        result = await orchestrator.execute_fan_out(
            context=context,
            node=node,
            step=step,
            inputs={"items": []},
            run_config=run_config,
            fan_out_spec=fan_out_spec,
        )

        assert result.success is True
        assert result.data["total_items"] == 0

    @pytest.mark.asyncio
    async def test_approval_with_no_expiration(
        self,
        orchestrator,
        frozen_clock,
    ):
        """Approval without timeout has no expiration."""
        run_config = GLIPRunConfig(
            pipeline_id="test",
            tenant_id="tenant",
            approval_timeout_seconds=0,  # No timeout
        )

        request = await orchestrator.request_approval(
            run_id="run-no-expire",
            step_id="step-no-expire",
            reason="Test no expiration",
            run_config=run_config,
        )

        assert request.expires_at is None

    def test_concurrency_config_defaults(self):
        """ConcurrencyConfig has sensible defaults."""
        config = ConcurrencyConfig()

        assert config.max_concurrent_runs == 100
        assert config.max_concurrent_steps_per_run == 10
        assert config.rate_limit_requests_per_minute == 1000
        assert config.queue_timeout_seconds == 300
        assert config.enable_fair_scheduling is True
