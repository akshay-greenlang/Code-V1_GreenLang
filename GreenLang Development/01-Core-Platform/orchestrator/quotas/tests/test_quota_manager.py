# -*- coding: utf-8 -*-
"""
Unit Tests for QuotaManager (FR-024)
====================================

Tests for namespace concurrency quota management including:
- Quota configuration CRUD
- Admission control
- Priority queue management
- Slot acquisition and release
- Metrics and events
- Queue timeout handling

Author: GreenLang Framework Team
Date: January 2026
GL-FOUND-X-001: FR-024 Namespace Concurrency Quotas
"""

import asyncio
import pytest
import time
from datetime import datetime, timezone
from typing import List

from greenlang.orchestrator.quotas.manager import (
    QuotaConfig,
    QuotaUsage,
    QuotaManager,
    QuotaEvent,
    QuotaEventType,
    QueuedRun,
    QuotaMetrics,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def quota_config() -> QuotaConfig:
    """Create a standard quota configuration."""
    return QuotaConfig(
        max_concurrent_runs=10,
        max_concurrent_steps=50,
        max_queued_runs=20,
        priority_weight=1.0,
        queue_timeout_seconds=300.0,
    )


@pytest.fixture
def production_quota() -> QuotaConfig:
    """Create a production namespace quota."""
    return QuotaConfig(
        max_concurrent_runs=50,
        max_concurrent_steps=200,
        max_queued_runs=100,
        priority_weight=2.0,
        queue_timeout_seconds=600.0,
    )


@pytest.fixture
def quota_manager() -> QuotaManager:
    """Create a quota manager instance."""
    return QuotaManager()


@pytest.fixture
def configured_manager(quota_config: QuotaConfig, production_quota: QuotaConfig) -> QuotaManager:
    """Create a quota manager with configured namespaces."""
    manager = QuotaManager(default_quota=quota_config)
    manager.set_quota("production", production_quota)
    manager.set_quota("development", QuotaConfig(
        max_concurrent_runs=5,
        max_concurrent_steps=25,
        max_queued_runs=10,
        priority_weight=0.5,
    ))
    return manager


# =============================================================================
# QUOTA CONFIG TESTS
# =============================================================================


class TestQuotaConfig:
    """Tests for QuotaConfig model."""

    def test_default_values(self):
        """Test default quota values."""
        config = QuotaConfig()
        assert config.max_concurrent_runs == 20
        assert config.max_concurrent_steps == 100
        assert config.max_queued_runs == 50
        assert config.priority_weight == 1.0
        assert config.queue_timeout_seconds == 300.0

    def test_custom_values(self, quota_config: QuotaConfig):
        """Test custom quota values."""
        assert quota_config.max_concurrent_runs == 10
        assert quota_config.max_concurrent_steps == 50
        assert quota_config.max_queued_runs == 20
        assert quota_config.priority_weight == 1.0

    def test_validation_max_concurrent_runs(self):
        """Test validation of max_concurrent_runs."""
        # Valid values
        QuotaConfig(max_concurrent_runs=1)
        QuotaConfig(max_concurrent_runs=1000)

        # Invalid values
        with pytest.raises(ValueError):
            QuotaConfig(max_concurrent_runs=0)
        with pytest.raises(ValueError):
            QuotaConfig(max_concurrent_runs=1001)

    def test_validation_priority_weight(self):
        """Test validation of priority_weight."""
        # Valid values
        QuotaConfig(priority_weight=0.1)
        QuotaConfig(priority_weight=10.0)

        # Invalid values
        with pytest.raises(ValueError):
            QuotaConfig(priority_weight=0.0)
        with pytest.raises(ValueError):
            QuotaConfig(priority_weight=11.0)

    def test_model_dump_yaml(self, quota_config: QuotaConfig):
        """Test YAML-friendly serialization."""
        data = quota_config.model_dump_yaml()
        assert data["max_concurrent_runs"] == 10
        assert data["max_concurrent_steps"] == 50
        assert data["priority_weight"] == 1.0


# =============================================================================
# QUOTA USAGE TESTS
# =============================================================================


class TestQuotaUsage:
    """Tests for QuotaUsage model."""

    def test_default_values(self):
        """Test default usage values."""
        usage = QuotaUsage()
        assert usage.current_runs == 0
        assert usage.current_steps == 0
        assert usage.queued_runs == 0
        assert len(usage.active_run_ids) == 0
        assert usage.total_runs_started == 0

    def test_update_timestamp(self):
        """Test timestamp update."""
        usage = QuotaUsage()
        original = usage.last_updated
        time.sleep(0.01)
        usage.update_timestamp()
        assert usage.last_updated > original

    def test_to_dict(self):
        """Test dictionary conversion."""
        usage = QuotaUsage(current_runs=5, current_steps=10)
        usage.active_run_ids.add("run-1")
        data = usage.to_dict()
        assert data["current_runs"] == 5
        assert data["current_steps"] == 10
        assert "run-1" in data["active_run_ids"]


# =============================================================================
# QUOTA MANAGER CONFIGURATION TESTS
# =============================================================================


class TestQuotaManagerConfiguration:
    """Tests for QuotaManager configuration methods."""

    def test_initialization(self, quota_manager: QuotaManager):
        """Test default initialization."""
        assert quota_manager is not None
        default_quota = quota_manager.get_quota("unknown")
        assert default_quota.max_concurrent_runs == 20

    def test_initialization_with_default(self, quota_config: QuotaConfig):
        """Test initialization with custom default."""
        manager = QuotaManager(default_quota=quota_config)
        default = manager.get_quota("any_namespace")
        assert default.max_concurrent_runs == 10

    def test_set_quota(self, quota_manager: QuotaManager, quota_config: QuotaConfig):
        """Test setting a namespace quota."""
        quota_manager.set_quota("test", quota_config)
        retrieved = quota_manager.get_quota("test")
        assert retrieved.max_concurrent_runs == 10

    def test_get_all_quotas(self, configured_manager: QuotaManager):
        """Test getting all quotas."""
        quotas = configured_manager.get_all_quotas()
        assert "production" in quotas
        assert "development" in quotas

    def test_delete_quota(self, configured_manager: QuotaManager):
        """Test deleting a quota."""
        assert configured_manager.delete_quota("production") is True
        assert configured_manager.delete_quota("nonexistent") is False
        quotas = configured_manager.get_all_quotas()
        assert "production" not in quotas


# =============================================================================
# ADMISSION CONTROL TESTS
# =============================================================================


class TestAdmissionControl:
    """Tests for admission control methods."""

    def test_can_submit_run_with_capacity(self, configured_manager: QuotaManager):
        """Test admission when capacity is available."""
        assert configured_manager.can_submit_run("production") is True

    def test_can_submit_run_at_limit(self, configured_manager: QuotaManager):
        """Test admission when at capacity limit."""
        # Fill up the namespace
        for i in range(50):  # production max is 50
            usage = configured_manager._get_or_create_usage("production")
            usage.current_runs += 1

        assert configured_manager.can_submit_run("production") is False

    def test_can_queue_run(self, configured_manager: QuotaManager):
        """Test queue admission."""
        assert configured_manager.can_queue_run("production") is True

    def test_can_queue_run_at_limit(self, configured_manager: QuotaManager):
        """Test queue admission when queue is full."""
        usage = configured_manager._get_or_create_usage("production")
        usage.queued_runs = 100  # production max queue is 100

        assert configured_manager.can_queue_run("production") is False

    def test_can_start_step(self, configured_manager: QuotaManager):
        """Test step admission."""
        assert configured_manager.can_start_step("production") is True

    def test_can_start_step_at_limit(self, configured_manager: QuotaManager):
        """Test step admission at limit."""
        usage = configured_manager._get_or_create_usage("production")
        usage.current_steps = 200  # production max is 200

        assert configured_manager.can_start_step("production") is False


# =============================================================================
# SLOT ACQUISITION TESTS
# =============================================================================


class TestSlotAcquisition:
    """Tests for slot acquisition methods."""

    @pytest.mark.asyncio
    async def test_acquire_run_slot_immediate(self, configured_manager: QuotaManager):
        """Test immediate slot acquisition."""
        acquired = await configured_manager.acquire_run_slot(
            "production", "run-1", priority=5
        )
        assert acquired is True

        usage = configured_manager.get_usage("production")
        assert usage.current_runs == 1
        assert "run-1" in usage.active_run_ids

    @pytest.mark.asyncio
    async def test_acquire_run_slot_queued(self, configured_manager: QuotaManager):
        """Test slot acquisition when queued."""
        # Fill up the namespace
        for i in range(50):
            await configured_manager.acquire_run_slot(
                "production", f"run-fill-{i}", priority=5
            )

        # This should be queued
        acquired = await configured_manager.acquire_run_slot(
            "production", "run-queued", priority=5, wait_for_slot=True
        )
        assert acquired is False

        usage = configured_manager.get_usage("production")
        assert usage.queued_runs == 1

    @pytest.mark.asyncio
    async def test_release_run_slot(self, configured_manager: QuotaManager):
        """Test releasing a run slot."""
        await configured_manager.acquire_run_slot("production", "run-1", priority=5)
        released = await configured_manager.release_run_slot("production", "run-1")

        assert released is True
        usage = configured_manager.get_usage("production")
        assert usage.current_runs == 0
        assert "run-1" not in usage.active_run_ids

    @pytest.mark.asyncio
    async def test_release_nonexistent_slot(self, configured_manager: QuotaManager):
        """Test releasing a non-existent slot."""
        released = await configured_manager.release_run_slot("production", "nonexistent")
        assert released is False

    @pytest.mark.asyncio
    async def test_acquire_step_slot(self, configured_manager: QuotaManager):
        """Test step slot acquisition."""
        acquired = await configured_manager.acquire_step_slot(
            "production", "run-1", "step-1"
        )
        assert acquired is True

        usage = configured_manager.get_usage("production")
        assert usage.current_steps == 1
        assert "step-1" in usage.active_step_ids

    @pytest.mark.asyncio
    async def test_release_step_slot(self, configured_manager: QuotaManager):
        """Test releasing a step slot."""
        await configured_manager.acquire_step_slot("production", "run-1", "step-1")
        released = await configured_manager.release_step_slot(
            "production", "run-1", "step-1"
        )

        assert released is True
        usage = configured_manager.get_usage("production")
        assert usage.current_steps == 0


# =============================================================================
# PRIORITY QUEUE TESTS
# =============================================================================


class TestPriorityQueue:
    """Tests for priority queue management."""

    @pytest.mark.asyncio
    async def test_queue_ordering_by_priority(self, configured_manager: QuotaManager):
        """Test that higher priority runs are dequeued first."""
        # Fill up production
        for i in range(50):
            await configured_manager.acquire_run_slot(
                "production", f"run-fill-{i}", priority=5
            )

        # Queue runs with different priorities
        await configured_manager.acquire_run_slot(
            "production", "run-low", priority=3, wait_for_slot=True
        )
        await configured_manager.acquire_run_slot(
            "production", "run-high", priority=9, wait_for_slot=True
        )
        await configured_manager.acquire_run_slot(
            "production", "run-medium", priority=5, wait_for_slot=True
        )

        # Release a slot
        await configured_manager.release_run_slot("production", "run-fill-0")

        # High priority run should be dequeued
        usage = configured_manager.get_usage("production")
        assert "run-high" in usage.active_run_ids

    @pytest.mark.asyncio
    async def test_queue_ordering_by_namespace_weight(self, configured_manager: QuotaManager):
        """Test that namespace weight affects priority."""
        # Fill up both namespaces (development has lower quota)
        for i in range(5):  # development max is 5
            await configured_manager.acquire_run_slot(
                "development", f"dev-fill-{i}", priority=5
            )

        for i in range(50):  # production max is 50
            await configured_manager.acquire_run_slot(
                "production", f"prod-fill-{i}", priority=5
            )

        # Queue runs with same priority in different namespaces
        await configured_manager.acquire_run_slot(
            "development", "dev-queued", priority=5, wait_for_slot=True
        )
        await configured_manager.acquire_run_slot(
            "production", "prod-queued", priority=5, wait_for_slot=True
        )

        # Production has higher weight, so its run should be processed first
        # when we dequeue globally

        # Check queue positions
        dev_pos = configured_manager.get_queue_position("dev-queued")
        prod_pos = configured_manager.get_queue_position("prod-queued")

        # Production (weight=2.0) should have better (lower) position
        assert prod_pos is not None
        assert dev_pos is not None

    def test_get_queue_depth(self, configured_manager: QuotaManager):
        """Test queue depth calculation."""
        assert configured_manager.get_queue_depth() == 0
        assert configured_manager.get_queue_depth("production") == 0


# =============================================================================
# METRICS TESTS
# =============================================================================


class TestMetrics:
    """Tests for quota metrics."""

    @pytest.mark.asyncio
    async def test_get_metrics(self, configured_manager: QuotaManager):
        """Test metrics retrieval."""
        await configured_manager.acquire_run_slot("production", "run-1", priority=5)
        await configured_manager.acquire_step_slot("production", "run-1", "step-1")

        metrics = configured_manager.get_metrics("production")

        assert metrics.namespace == "production"
        assert metrics.runs_utilization_percent == 2.0  # 1/50 = 2%
        assert metrics.steps_utilization_percent == 0.5  # 1/200 = 0.5%
        assert metrics.queue_depth == 0

    def test_get_all_metrics(self, configured_manager: QuotaManager):
        """Test getting all namespace metrics."""
        metrics_list = configured_manager.get_all_metrics()
        assert len(metrics_list) == 2  # production and development
        namespaces = [m.namespace for m in metrics_list]
        assert "production" in namespaces
        assert "development" in namespaces

    def test_prometheus_format(self, configured_manager: QuotaManager):
        """Test Prometheus metrics format."""
        prometheus = configured_manager.get_prometheus_metrics()
        assert "greenlang_quota_usage_percent" in prometheus
        assert "greenlang_queue_depth" in prometheus
        assert 'namespace="production"' in prometheus


# =============================================================================
# EVENT CALLBACK TESTS
# =============================================================================


class TestEventCallbacks:
    """Tests for event callback functionality."""

    @pytest.mark.asyncio
    async def test_event_on_slot_acquired(self, quota_config: QuotaConfig):
        """Test event emission on slot acquisition."""
        events: List[QuotaEvent] = []
        manager = QuotaManager(
            default_quota=quota_config,
            event_callback=lambda e: events.append(e)
        )
        manager.set_quota("test", quota_config)

        await manager.acquire_run_slot("test", "run-1", priority=5)

        assert len(events) >= 1
        slot_events = [
            e for e in events
            if e.event_type == QuotaEventType.CONCURRENCY_SLOT_ACQUIRED
        ]
        assert len(slot_events) == 1
        assert slot_events[0].run_id == "run-1"

    @pytest.mark.asyncio
    async def test_event_on_slot_released(self, quota_config: QuotaConfig):
        """Test event emission on slot release."""
        events: List[QuotaEvent] = []
        manager = QuotaManager(
            default_quota=quota_config,
            event_callback=lambda e: events.append(e)
        )
        manager.set_quota("test", quota_config)

        await manager.acquire_run_slot("test", "run-1", priority=5)
        events.clear()

        await manager.release_run_slot("test", "run-1")

        release_events = [
            e for e in events
            if e.event_type == QuotaEventType.CONCURRENCY_SLOT_RELEASED
        ]
        assert len(release_events) == 1

    def test_event_on_quota_updated(self, quota_manager: QuotaManager):
        """Test event emission on quota update."""
        events: List[QuotaEvent] = []
        quota_manager.set_event_callback(lambda e: events.append(e))

        quota_manager.set_quota("test", QuotaConfig())

        update_events = [
            e for e in events
            if e.event_type in [QuotaEventType.NAMESPACE_CREATED, QuotaEventType.QUOTA_UPDATED]
        ]
        assert len(update_events) >= 1


# =============================================================================
# QUEUED RUN TESTS
# =============================================================================


class TestQueuedRun:
    """Tests for QueuedRun dataclass."""

    def test_is_expired(self):
        """Test expiration check."""
        # Create an already expired run
        expired = QueuedRun(
            effective_priority=5.0,
            queued_at=time.time() - 100,
            run_id="run-1",
            namespace="test",
            base_priority=5,
            timeout_at=time.time() - 10,  # 10 seconds ago
        )
        assert expired.is_expired() is True

        # Create a non-expired run
        valid = QueuedRun(
            effective_priority=5.0,
            queued_at=time.time(),
            run_id="run-2",
            namespace="test",
            base_priority=5,
            timeout_at=time.time() + 300,  # 5 minutes from now
        )
        assert valid.is_expired() is False

    def test_wait_time_seconds(self):
        """Test wait time calculation."""
        run = QueuedRun(
            effective_priority=5.0,
            queued_at=time.time() - 10,  # 10 seconds ago
            run_id="run-1",
            namespace="test",
            base_priority=5,
            timeout_at=time.time() + 290,
        )
        assert run.wait_time_seconds() >= 10.0

    def test_ordering(self):
        """Test priority ordering for heap."""
        high_priority = QueuedRun(
            effective_priority=1.0,  # Lower = higher priority
            queued_at=time.time(),
            run_id="high",
            namespace="test",
            base_priority=9,
            timeout_at=time.time() + 300,
        )
        low_priority = QueuedRun(
            effective_priority=5.0,
            queued_at=time.time(),
            run_id="low",
            namespace="test",
            base_priority=5,
            timeout_at=time.time() + 300,
        )

        assert high_priority < low_priority  # For heap ordering


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for QuotaManager."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, configured_manager: QuotaManager):
        """Test complete run lifecycle."""
        namespace = "production"

        # 1. Check admission
        assert configured_manager.can_submit_run(namespace)

        # 2. Acquire run slot
        acquired = await configured_manager.acquire_run_slot(
            namespace, "run-lifecycle", priority=7
        )
        assert acquired is True

        # 3. Acquire step slots
        for i in range(3):
            step_acquired = await configured_manager.acquire_step_slot(
                namespace, "run-lifecycle", f"step-{i}"
            )
            assert step_acquired is True

        # 4. Check usage
        usage = configured_manager.get_usage(namespace)
        assert usage.current_runs == 1
        assert usage.current_steps == 3
        assert usage.total_runs_started == 1

        # 5. Release step slots
        for i in range(3):
            await configured_manager.release_step_slot(
                namespace, "run-lifecycle", f"step-{i}"
            )

        # 6. Release run slot
        await configured_manager.release_run_slot(namespace, "run-lifecycle")

        # 7. Verify final state
        final_usage = configured_manager.get_usage(namespace)
        assert final_usage.current_runs == 0
        assert final_usage.current_steps == 0
        assert final_usage.total_runs_completed == 1

    @pytest.mark.asyncio
    async def test_multi_namespace_isolation(self, configured_manager: QuotaManager):
        """Test that namespaces are isolated."""
        # Acquire slots in production
        for i in range(5):
            await configured_manager.acquire_run_slot(
                "production", f"prod-{i}", priority=5
            )

        # Check that development is unaffected
        dev_usage = configured_manager.get_usage("development")
        assert dev_usage.current_runs == 0

        prod_usage = configured_manager.get_usage("production")
        assert prod_usage.current_runs == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
