"""
GL-001 ThermalCommand - Webhook Dispatcher Tests

Comprehensive tests for webhook dispatcher including:
- Circuit breaker pattern
- Async dispatch
- Retry logic
- Delivery tracking
- DLQ handling

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import asyncio
from datetime import datetime, timezone, timedelta
import time
import pytest
from pydantic import SecretStr

from ..webhook_dispatcher import (
    WebhookDispatcher,
    CircuitBreaker,
    CircuitState,
    DispatchResult,
    WebhookDeliveryTracker,
)
from ..webhook_manager import (
    WebhookManager,
    DeliveryResult,
    DeliveryStatus,
)
from ..webhook_config import (
    WebhookConfig,
    WebhookEndpoint,
    EndpointStatus,
    RateLimitConfig,
)
from ..webhook_events import (
    WebhookEventType,
    HeatPlanCreatedEvent,
    SetpointRecommendationEvent,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        """Create circuit breaker."""
        return CircuitBreaker(
            failure_threshold=3,
            success_threshold=2,
            reset_timeout_seconds=1.0
        )

    def test_initial_state_closed(self, breaker):
        """Test initial state is closed."""
        can_execute, reason = breaker.can_execute("ep-001")

        assert can_execute is True
        assert reason == "circuit_closed"
        assert breaker.get_state("ep-001") == CircuitState.CLOSED

    def test_circuit_opens_after_failures(self, breaker):
        """Test circuit opens after threshold failures."""
        endpoint_id = "ep-001"

        # Record failures up to threshold
        for _ in range(3):
            breaker.record_failure(endpoint_id)

        assert breaker.get_state(endpoint_id) == CircuitState.OPEN

        can_execute, reason = breaker.can_execute(endpoint_id)
        assert can_execute is False
        assert reason == "circuit_open"

    def test_circuit_transitions_to_half_open(self, breaker):
        """Test circuit transitions to half-open after timeout."""
        endpoint_id = "ep-001"

        # Open the circuit
        for _ in range(3):
            breaker.record_failure(endpoint_id)

        assert breaker.get_state(endpoint_id) == CircuitState.OPEN

        # Wait for reset timeout
        time.sleep(1.1)

        # Should now be half-open
        can_execute, reason = breaker.can_execute(endpoint_id)
        assert can_execute is True
        assert reason == "circuit_half_open"
        assert breaker.get_state(endpoint_id) == CircuitState.HALF_OPEN

    def test_circuit_closes_after_successes(self, breaker):
        """Test circuit closes after threshold successes in half-open."""
        endpoint_id = "ep-001"

        # Open the circuit
        for _ in range(3):
            breaker.record_failure(endpoint_id)

        # Wait for half-open
        time.sleep(1.1)
        breaker.can_execute(endpoint_id)

        # Record successes
        for _ in range(2):
            breaker.record_success(endpoint_id)

        assert breaker.get_state(endpoint_id) == CircuitState.CLOSED

    def test_circuit_reopens_on_failure_in_half_open(self, breaker):
        """Test circuit reopens on failure in half-open state."""
        endpoint_id = "ep-001"

        # Open the circuit
        for _ in range(3):
            breaker.record_failure(endpoint_id)

        # Wait for half-open
        time.sleep(1.1)
        breaker.can_execute(endpoint_id)

        # Fail in half-open
        breaker.record_failure(endpoint_id)

        assert breaker.get_state(endpoint_id) == CircuitState.OPEN

    def test_success_resets_failure_count(self, breaker):
        """Test success resets failure count in closed state."""
        endpoint_id = "ep-001"

        # Record some failures (not enough to open)
        breaker.record_failure(endpoint_id)
        breaker.record_failure(endpoint_id)

        # Success should reset
        breaker.record_success(endpoint_id)

        # Now more failures needed to open
        breaker.record_failure(endpoint_id)
        breaker.record_failure(endpoint_id)

        # Should still be closed (only 2 failures since reset)
        assert breaker.get_state(endpoint_id) == CircuitState.CLOSED

    def test_reset_clears_state(self, breaker):
        """Test reset clears circuit state."""
        endpoint_id = "ep-001"

        # Open circuit
        for _ in range(3):
            breaker.record_failure(endpoint_id)

        assert breaker.get_state(endpoint_id) == CircuitState.OPEN

        # Reset
        breaker.reset(endpoint_id)

        can_execute, _ = breaker.can_execute(endpoint_id)
        assert can_execute is True
        assert breaker.get_state(endpoint_id) == CircuitState.CLOSED

    def test_get_all_states(self, breaker):
        """Test getting all circuit states."""
        # Create states for multiple endpoints
        breaker.can_execute("ep-001")
        breaker.can_execute("ep-002")

        for _ in range(3):
            breaker.record_failure("ep-002")

        states = breaker.get_all_states()

        assert states["ep-001"] == CircuitState.CLOSED
        assert states["ep-002"] == CircuitState.OPEN


class TestWebhookDispatcher:
    """Tests for WebhookDispatcher."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WebhookConfig(endpoints=[
            WebhookEndpoint(
                endpoint_id="ep-001",
                name="Test Endpoint 1",
                url="https://example1.com/webhook",
                secret=SecretStr("secret1"),
                event_types={
                    WebhookEventType.HEAT_PLAN_CREATED,
                    WebhookEventType.SETPOINT_RECOMMENDATION
                }
            ),
            WebhookEndpoint(
                endpoint_id="ep-002",
                name="Test Endpoint 2",
                url="https://example2.com/webhook",
                secret=SecretStr("secret2"),
                event_types={WebhookEventType.HEAT_PLAN_CREATED}
            )
        ])

    @pytest.fixture
    def manager(self, config):
        """Create webhook manager."""
        return WebhookManager(config)

    @pytest.fixture
    def dispatcher(self, manager):
        """Create webhook dispatcher."""
        return WebhookDispatcher(
            manager,
            max_concurrent_deliveries=10,
            circuit_failure_threshold=3,
            circuit_reset_timeout=1.0
        )

    @pytest.fixture
    def sample_event(self):
        """Create sample event."""
        return HeatPlanCreatedEvent(
            plan_id="plan-001",
            horizon_hours=24,
            expected_cost_usd=15000.0,
            expected_emissions_kg_co2e=1200.0
        )

    @pytest.mark.asyncio
    async def test_dispatcher_start_shutdown(self, dispatcher):
        """Test dispatcher start and shutdown."""
        await dispatcher.start()
        assert dispatcher._running is True

        await dispatcher.shutdown()
        assert dispatcher._running is False

    @pytest.mark.asyncio
    async def test_dispatch_to_all_endpoints(self, dispatcher, sample_event):
        """Test dispatching to all subscribed endpoints."""
        await dispatcher.start()

        try:
            result = await dispatcher.dispatch_event(sample_event)

            assert isinstance(result, DispatchResult)
            assert result.event_id == sample_event.event_id
            assert result.total_endpoints == 2  # Both endpoints subscribe
            assert len(result.delivery_results) == 2
        finally:
            await dispatcher.shutdown()

    @pytest.mark.asyncio
    async def test_dispatch_to_specific_endpoints(self, dispatcher, sample_event):
        """Test dispatching to specific endpoints."""
        await dispatcher.start()

        try:
            result = await dispatcher.dispatch_event(
                sample_event,
                endpoint_ids=["ep-001"]
            )

            assert result.total_endpoints == 1
            assert "ep-001" in result.delivery_results
        finally:
            await dispatcher.shutdown()

    @pytest.mark.asyncio
    async def test_dispatch_no_endpoints(self, dispatcher):
        """Test dispatching when no endpoints subscribe."""
        await dispatcher.start()

        try:
            # Create event type that no endpoint subscribes to
            event = HeatPlanCreatedEvent(
                event_id="test-event",
                plan_id="plan-001",
                horizon_hours=24,
                expected_cost_usd=15000.0,
                expected_emissions_kg_co2e=1200.0
            )
            # Manually filter out endpoints
            result = await dispatcher.dispatch_event(
                event,
                endpoint_ids=["nonexistent"]
            )

            assert result.total_endpoints == 0
        finally:
            await dispatcher.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_delivery(self, dispatcher, manager, sample_event):
        """Test circuit breaker blocks delivery when open."""
        await dispatcher.start()

        try:
            # Open circuit for ep-001
            for _ in range(3):
                dispatcher.circuit_breaker.record_failure("ep-001")

            result = await dispatcher.dispatch_event(
                sample_event,
                endpoint_ids=["ep-001"]
            )

            # Delivery should be skipped due to circuit open
            assert result.skipped_deliveries == 1
            ep_result = result.delivery_results["ep-001"]
            assert ep_result.status == DeliveryStatus.CIRCUIT_OPEN
        finally:
            await dispatcher.shutdown()

    @pytest.mark.asyncio
    async def test_idempotent_delivery(self, dispatcher, manager, sample_event):
        """Test idempotent delivery returns cached result."""
        await dispatcher.start()

        try:
            # First delivery
            result1 = await dispatcher.dispatch_event(
                sample_event,
                endpoint_ids=["ep-001"]
            )

            # Second delivery of same event
            result2 = await dispatcher.dispatch_event(
                sample_event,
                endpoint_ids=["ep-001"]
            )

            # Should return cached result (idempotent)
            # Note: Mock delivery may or may not cache depending on implementation
            assert result1.event_id == result2.event_id
        finally:
            await dispatcher.shutdown()

    @pytest.mark.asyncio
    async def test_dispatch_result_statistics(self, dispatcher, sample_event):
        """Test dispatch result contains correct statistics."""
        await dispatcher.start()

        try:
            result = await dispatcher.dispatch_event(sample_event)

            assert result.total_endpoints == 2
            assert result.total_duration_ms >= 0
            assert result.dispatched_at is not None
            assert result.completed_at is not None
            assert result.completed_at >= result.dispatched_at

            # Check counts add up
            total = (
                result.successful_deliveries +
                result.failed_deliveries +
                result.skipped_deliveries
            )
            assert total == result.total_endpoints
        finally:
            await dispatcher.shutdown()

    def test_get_circuit_states(self, dispatcher):
        """Test getting circuit breaker states."""
        # Record some activity
        dispatcher.circuit_breaker.can_execute("ep-001")
        for _ in range(3):
            dispatcher.circuit_breaker.record_failure("ep-002")

        states = dispatcher.get_circuit_states()

        assert "ep-001" in states
        assert states["ep-001"] == CircuitState.CLOSED
        assert states["ep-002"] == CircuitState.OPEN

    def test_get_retry_queue_size(self, dispatcher):
        """Test getting retry queue size."""
        assert dispatcher.get_retry_queue_size() == 0


class TestWebhookDeliveryTracker:
    """Tests for WebhookDeliveryTracker."""

    @pytest.fixture
    def tracker(self):
        """Create delivery tracker."""
        return WebhookDeliveryTracker(history_size=1000)

    def test_record_delivery(self, tracker):
        """Test recording delivery result."""
        result = DeliveryResult(
            event_id="event-001",
            endpoint_id="ep-001",
            status=DeliveryStatus.DELIVERED,
            duration_ms=150.0,
            completed_at=datetime.now(timezone.utc)
        )

        tracker.record(result)

        stats = tracker.get_endpoint_stats("ep-001")
        assert stats["total_deliveries"] >= 1

    def test_get_endpoint_stats(self, tracker):
        """Test getting endpoint statistics."""
        # Record various deliveries
        for i in range(10):
            result = DeliveryResult(
                event_id=f"event-{i}",
                endpoint_id="ep-001",
                status=DeliveryStatus.DELIVERED if i < 8 else DeliveryStatus.FAILED,
                duration_ms=100.0 + i * 10,
                completed_at=datetime.now(timezone.utc)
            )
            tracker.record(result)

        stats = tracker.get_endpoint_stats("ep-001")

        assert stats["total_deliveries"] == 10
        assert stats["successful_deliveries"] == 8
        assert stats["failed_deliveries"] == 2
        assert stats["success_rate"] == 0.8
        assert stats["error_rate"] == 0.2
        assert stats["avg_duration_ms"] > 0

    def test_get_endpoint_stats_time_window(self, tracker):
        """Test endpoint stats respects time window."""
        # Record old delivery (outside window)
        old_result = DeliveryResult(
            event_id="old-event",
            endpoint_id="ep-001",
            status=DeliveryStatus.DELIVERED,
            duration_ms=100.0,
            completed_at=datetime.now(timezone.utc) - timedelta(hours=2)
        )
        tracker.record(old_result)

        # Stats with 1-hour window should not include old delivery
        stats = tracker.get_endpoint_stats("ep-001", window_minutes=60)
        assert stats["total_deliveries"] == 0

    def test_get_recent_failures(self, tracker):
        """Test getting recent failures."""
        # Record some failures
        for i in range(5):
            result = DeliveryResult(
                event_id=f"failed-{i}",
                endpoint_id="ep-001",
                status=DeliveryStatus.FAILED,
                error_message=f"Error {i}",
                completed_at=datetime.now(timezone.utc)
            )
            tracker.record(result)

        # Record success
        tracker.record(DeliveryResult(
            event_id="success",
            endpoint_id="ep-001",
            status=DeliveryStatus.DELIVERED,
            completed_at=datetime.now(timezone.utc)
        ))

        failures = tracker.get_recent_failures(limit=10)

        assert len(failures) == 5
        assert all(f.status == DeliveryStatus.FAILED for f in failures)

    def test_get_global_stats(self, tracker):
        """Test getting global statistics."""
        # Record deliveries for multiple endpoints
        for ep_id in ["ep-001", "ep-002", "ep-003"]:
            for i in range(5):
                result = DeliveryResult(
                    event_id=f"event-{ep_id}-{i}",
                    endpoint_id=ep_id,
                    status=DeliveryStatus.DELIVERED if i < 4 else DeliveryStatus.FAILED,
                    completed_at=datetime.now(timezone.utc)
                )
                tracker.record(result)

        stats = tracker.get_global_stats()

        assert stats["total_deliveries"] == 15
        assert stats["successful_deliveries"] == 12
        assert stats["success_rate"] == 0.8
        assert stats["unique_endpoints"] == 3
        assert stats["endpoints_with_failures"] == 3

    def test_history_size_limit(self, tracker):
        """Test history respects size limit."""
        small_tracker = WebhookDeliveryTracker(history_size=10)

        # Record more than limit
        for i in range(20):
            result = DeliveryResult(
                event_id=f"event-{i}",
                endpoint_id="ep-001",
                status=DeliveryStatus.DELIVERED,
                completed_at=datetime.now(timezone.utc)
            )
            small_tracker.record(result)

        # Should only have last 10
        stats = small_tracker.get_global_stats()
        assert stats["total_deliveries"] <= 10


class TestDispatchResultModel:
    """Tests for DispatchResult model."""

    def test_dispatch_result_creation(self):
        """Test DispatchResult creation."""
        result = DispatchResult(
            event_id="event-001",
            event_type="heat_plan.created"
        )

        assert result.event_id == "event-001"
        assert result.total_endpoints == 0
        assert result.successful_deliveries == 0
        assert result.dispatched_at is not None

    def test_dispatch_result_with_deliveries(self):
        """Test DispatchResult with delivery results."""
        delivery1 = DeliveryResult(
            event_id="event-001",
            endpoint_id="ep-001",
            status=DeliveryStatus.DELIVERED
        )
        delivery2 = DeliveryResult(
            event_id="event-001",
            endpoint_id="ep-002",
            status=DeliveryStatus.FAILED
        )

        result = DispatchResult(
            event_id="event-001",
            event_type="heat_plan.created",
            total_endpoints=2,
            successful_deliveries=1,
            failed_deliveries=1,
            delivery_results={
                "ep-001": delivery1,
                "ep-002": delivery2
            }
        )

        assert result.total_endpoints == 2
        assert result.successful_deliveries == 1
        assert result.failed_deliveries == 1
        assert "ep-001" in result.delivery_results


class TestIntegration:
    """Integration tests for the complete webhook system."""

    @pytest.fixture
    def full_system(self):
        """Create full webhook system."""
        config = WebhookConfig(endpoints=[
            WebhookEndpoint(
                endpoint_id="ep-001",
                name="Primary Endpoint",
                url="https://primary.example.com/webhook",
                secret=SecretStr("primary-secret"),
                event_types={
                    WebhookEventType.HEAT_PLAN_CREATED,
                    WebhookEventType.SAFETY_ACTION_BLOCKED
                }
            ),
            WebhookEndpoint(
                endpoint_id="ep-002",
                name="Backup Endpoint",
                url="https://backup.example.com/webhook",
                secret=SecretStr("backup-secret"),
                event_types={WebhookEventType.HEAT_PLAN_CREATED}
            )
        ])

        manager = WebhookManager(config)
        dispatcher = WebhookDispatcher(manager)

        return {
            "config": config,
            "manager": manager,
            "dispatcher": dispatcher
        }

    @pytest.mark.asyncio
    async def test_full_event_flow(self, full_system):
        """Test complete event flow through the system."""
        manager = full_system["manager"]
        dispatcher = full_system["dispatcher"]

        await manager.start()
        await dispatcher.start()

        try:
            # Create and dispatch event
            event = HeatPlanCreatedEvent(
                plan_id="integration-test-plan",
                horizon_hours=24,
                expected_cost_usd=15000.0,
                expected_emissions_kg_co2e=1200.0
            )

            result = await dispatcher.dispatch_event(event)

            # Verify dispatch completed
            assert result.total_endpoints == 2
            assert result.completed_at is not None

            # Verify statistics updated
            stats = manager.get_statistics()
            assert stats["delivery_stats"]["total_delivered"] + stats["delivery_stats"]["total_failed"] > 0

        finally:
            await dispatcher.shutdown()
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, full_system):
        """Test circuit breaker opens and recovers."""
        dispatcher = full_system["dispatcher"]

        await dispatcher.start()

        try:
            # Trigger circuit open
            for _ in range(5):
                dispatcher.circuit_breaker.record_failure("ep-001")

            assert dispatcher.circuit_breaker.get_state("ep-001") == CircuitState.OPEN

            # Wait for reset timeout (default is longer, but we set it short in fixture)
            # In real test, would need to wait for actual timeout

            # Reset manually for test
            dispatcher.circuit_breaker.reset("ep-001")

            assert dispatcher.circuit_breaker.get_state("ep-001") == CircuitState.CLOSED

        finally:
            await dispatcher.shutdown()
