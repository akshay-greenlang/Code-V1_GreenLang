# -*- coding: utf-8 -*-
"""
Test suite for MessageBus implementation.

Tests cover:
- Event publishing and subscription
- Wildcard patterns
- Priority queuing
- Request-reply pattern
- Error handling and retries
- Dead letter queue
- Metrics collection

Author: GreenLang Framework Team
Date: December 2025
"""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock

from greenlang.core.messaging import (
    InMemoryMessageBus,
    MessageBusConfig,
    create_event,
    StandardEvents,
    EventPriority,
)


@pytest_asyncio.fixture
async def message_bus():
    """Create and start a message bus for testing."""
    bus = InMemoryMessageBus()
    await bus.start()
    yield bus
    await bus.close()


@pytest_asyncio.fixture
async def configured_bus():
    """Create message bus with custom configuration."""
    config = MessageBusConfig(
        max_queue_size=100,
        max_retries=2,
        retry_delay_seconds=0.1,
        enable_dead_letter=True,
    )
    bus = InMemoryMessageBus(config)
    await bus.start()
    yield bus
    await bus.close()


class TestBasicPublishSubscribe:
    """Test basic publish/subscribe functionality."""

    @pytest.mark.asyncio
    async def test_simple_publish_subscribe(self, message_bus):
        """Test that subscribed handler receives published event."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        # Subscribe
        await message_bus.subscribe(
            StandardEvents.AGENT_STARTED, handler, "test-subscriber"
        )

        # Publish event
        event = create_event(
            event_type=StandardEvents.AGENT_STARTED,
            source_agent="GL-001",
            payload={"status": "ready"},
        )
        await message_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Verify
        assert len(received_events) == 1
        assert received_events[0].event_type == StandardEvents.AGENT_STARTED
        assert received_events[0].payload["status"] == "ready"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, message_bus):
        """Test multiple subscribers receive the same event."""
        received_by_sub1 = []
        received_by_sub2 = []

        async def handler1(event):
            received_by_sub1.append(event)

        async def handler2(event):
            received_by_sub2.append(event)

        # Subscribe both handlers
        await message_bus.subscribe(
            StandardEvents.CALCULATION_COMPLETED, handler1, "subscriber-1"
        )
        await message_bus.subscribe(
            StandardEvents.CALCULATION_COMPLETED, handler2, "subscriber-2"
        )

        # Publish event
        event = create_event(
            event_type=StandardEvents.CALCULATION_COMPLETED,
            source_agent="GL-001",
            payload={"result": 42.5},
        )
        await message_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Both handlers should receive the event
        assert len(received_by_sub1) == 1
        assert len(received_by_sub2) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, message_bus):
        """Test unsubscribing from events."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        # Subscribe and then unsubscribe
        await message_bus.subscribe(
            StandardEvents.AGENT_ERROR, handler, "test-subscriber"
        )
        await message_bus.unsubscribe(StandardEvents.AGENT_ERROR, "test-subscriber")

        # Publish event
        event = create_event(
            event_type=StandardEvents.AGENT_ERROR,
            source_agent="GL-001",
            payload={"error": "test error"},
        )
        await message_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Handler should not receive event
        assert len(received_events) == 0


class TestWildcardPatterns:
    """Test wildcard pattern matching."""

    @pytest.mark.asyncio
    async def test_single_level_wildcard(self, message_bus):
        """Test single-level wildcard (*) pattern."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        # Subscribe to agent.* pattern
        await message_bus.subscribe("agent.*", handler, "wildcard-subscriber")

        # Publish matching events
        event1 = create_event(
            event_type=StandardEvents.AGENT_STARTED,
            source_agent="GL-001",
            payload={},
        )
        event2 = create_event(
            event_type=StandardEvents.AGENT_STOPPED,
            source_agent="GL-001",
            payload={},
        )
        event3 = create_event(
            event_type=StandardEvents.CALCULATION_COMPLETED,  # Should not match
            source_agent="GL-001",
            payload={},
        )

        await message_bus.publish(event1)
        await message_bus.publish(event2)
        await message_bus.publish(event3)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Should receive only agent.* events
        assert len(received_events) == 2
        assert all(e.event_type.startswith("agent.") for e in received_events)

    @pytest.mark.asyncio
    async def test_multi_level_wildcard(self, message_bus):
        """Test multi-level wildcard (**) pattern."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        # Subscribe to orchestration.** pattern
        await message_bus.subscribe(
            "orchestration.**", handler, "multi-wildcard-subscriber"
        )

        # Publish matching events
        event1 = create_event(
            event_type=StandardEvents.TASK_ASSIGNED,
            source_agent="orchestrator",
            payload={},
        )
        event2 = create_event(
            event_type=StandardEvents.WORKFLOW_STARTED,
            source_agent="orchestrator",
            payload={},
        )
        event3 = create_event(
            event_type=StandardEvents.AGENT_STARTED,  # Should not match
            source_agent="GL-001",
            payload={},
        )

        await message_bus.publish(event1)
        await message_bus.publish(event2)
        await message_bus.publish(event3)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Should receive only orchestration.** events
        assert len(received_events) == 2
        assert all(e.event_type.startswith("orchestration.") for e in received_events)

    @pytest.mark.asyncio
    async def test_wildcard_all_events(self, message_bus):
        """Test subscribing to all events with * pattern."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        # Subscribe to all events
        await message_bus.subscribe("*", handler, "all-events-subscriber")

        # Publish various events
        events = [
            create_event(
                event_type=StandardEvents.AGENT_STARTED,
                source_agent="GL-001",
                payload={},
            ),
            create_event(
                event_type=StandardEvents.CALCULATION_COMPLETED,
                source_agent="GL-002",
                payload={},
            ),
            create_event(
                event_type=StandardEvents.SAFETY_ALERT,
                source_agent="GL-003",
                payload={},
            ),
        ]

        for event in events:
            await message_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Should receive all events
        assert len(received_events) == 3


class TestPriorityQueuing:
    """Test priority-based event processing."""

    @pytest.mark.asyncio
    async def test_priority_ordering(self, message_bus):
        """Test that higher priority events are processed first."""
        received_events = []
        processing_lock = asyncio.Lock()

        async def slow_handler(event):
            async with processing_lock:
                await asyncio.sleep(0.05)  # Simulate slow processing
                received_events.append(event)

        # Subscribe
        await message_bus.subscribe("test.event", slow_handler, "priority-test")

        # Publish events with different priorities
        low_event = create_event(
            event_type="test.event",
            source_agent="test",
            payload={"priority": "low"},
            priority=EventPriority.LOW,
        )
        high_event = create_event(
            event_type="test.event",
            source_agent="test",
            payload={"priority": "high"},
            priority=EventPriority.HIGH,
        )
        critical_event = create_event(
            event_type="test.event",
            source_agent="test",
            payload={"priority": "critical"},
            priority=EventPriority.CRITICAL,
        )

        # Publish in reverse priority order
        await message_bus.publish(low_event)
        await message_bus.publish(high_event)
        await message_bus.publish(critical_event)

        # Wait for all events to be processed
        await asyncio.sleep(0.5)

        # Critical should be processed first, then high, then low
        assert len(received_events) == 3
        assert received_events[0].payload["priority"] == "critical"
        assert received_events[1].payload["priority"] == "high"
        assert received_events[2].payload["priority"] == "low"


class TestRequestReply:
    """Test request-reply pattern."""

    @pytest.mark.asyncio
    async def test_successful_request_reply(self, message_bus):
        """Test successful request-reply communication."""

        async def request_handler(event):
            # Simulate processing and send reply
            reply = create_event(
                event_type="test.reply",
                source_agent="responder",
                payload={"result": event.payload["value"] * 2},
                correlation_id=event.event_id,  # Link to request
            )
            await message_bus.publish(reply)

        # Subscribe to request events
        await message_bus.subscribe("test.request", request_handler, "responder")

        # Send request
        request = create_event(
            event_type="test.request",
            source_agent="requester",
            payload={"value": 21},
        )

        # Use request-reply pattern
        reply = await message_bus.request_reply(request, timeout_seconds=1.0)

        # Verify reply
        assert reply is not None
        assert reply.payload["result"] == 42

    @pytest.mark.asyncio
    async def test_request_timeout(self, message_bus):
        """Test request timeout when no reply is received."""
        # Send request without any responder
        request = create_event(
            event_type="test.no.responder",
            source_agent="requester",
            payload={"value": 42},
        )

        # Should timeout
        reply = await message_bus.request_reply(request, timeout_seconds=0.2)
        assert reply is None

        # Check metrics
        metrics = message_bus.get_metrics()
        assert metrics.requests_timeout == 1


class TestErrorHandling:
    """Test error handling and retry logic."""

    @pytest.mark.asyncio
    async def test_handler_retry_on_failure(self, configured_bus):
        """Test that failed handlers are retried."""
        attempt_count = 0
        received_events = []

        async def failing_handler(event):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Simulated failure")
            received_events.append(event)

        # Subscribe
        await configured_bus.subscribe(
            "test.retry", failing_handler, "retry-test"
        )

        # Publish event
        event = create_event(
            event_type="test.retry",
            source_agent="test",
            payload={"test": "retry"},
        )
        await configured_bus.publish(event)

        # Wait for retries
        await asyncio.sleep(0.5)

        # Should have retried and succeeded
        assert attempt_count == 3
        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_dead_letter_queue(self, configured_bus):
        """Test that persistently failing events go to dead letter queue."""

        async def always_failing_handler(event):
            raise ValueError("Always fails")

        # Subscribe
        await configured_bus.subscribe(
            "test.deadletter", always_failing_handler, "failing-handler"
        )

        # Publish event
        event = create_event(
            event_type="test.deadletter",
            source_agent="test",
            payload={"test": "deadletter"},
        )
        await configured_bus.publish(event)

        # Wait for retries to exhaust
        await asyncio.sleep(0.5)

        # Event should be in dead letter queue
        dlq = configured_bus.get_dead_letter_queue()
        assert len(dlq) == 1
        assert dlq[0].event_id == event.event_id

    @pytest.mark.asyncio
    async def test_replay_dead_letter(self, configured_bus):
        """Test replaying events from dead letter queue."""
        received_events = []
        should_fail = True

        async def conditional_handler(event):
            if should_fail:
                raise ValueError("Fail on first try")
            received_events.append(event)

        # Subscribe
        await configured_bus.subscribe(
            "test.replay", conditional_handler, "replay-handler"
        )

        # Publish event (will fail)
        event = create_event(
            event_type="test.replay",
            source_agent="test",
            payload={"test": "replay"},
        )
        await configured_bus.publish(event)

        # Wait for failure
        await asyncio.sleep(0.5)

        # Should be in dead letter queue
        dlq = configured_bus.get_dead_letter_queue()
        assert len(dlq) == 1

        # Fix the handler and replay
        should_fail = False
        success = await configured_bus.replay_dead_letter(event.event_id)
        assert success

        # Wait for replay
        await asyncio.sleep(0.2)

        # Should now be processed
        assert len(received_events) == 1


class TestMetrics:
    """Test metrics collection."""

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, message_bus):
        """Test that metrics are tracked correctly."""
        received_count = 0

        async def counter_handler(event):
            nonlocal received_count
            received_count += 1

        # Subscribe
        await message_bus.subscribe("test.metrics", counter_handler, "metrics-test")

        # Publish multiple events
        for i in range(5):
            event = create_event(
                event_type="test.metrics",
                source_agent="test",
                payload={"count": i},
            )
            await message_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.3)

        # Check metrics
        metrics = message_bus.get_metrics()
        assert metrics.events_published == 5
        assert metrics.events_delivered == 5
        assert metrics.events_failed == 0
        assert metrics.active_subscriptions == 1

    @pytest.mark.asyncio
    async def test_delivery_time_tracking(self, message_bus):
        """Test that delivery time is tracked."""

        async def handler(event):
            await asyncio.sleep(0.01)  # Simulate some processing

        # Subscribe and publish
        await message_bus.subscribe("test.timing", handler, "timing-test")

        event = create_event(
            event_type="test.timing",
            source_agent="test",
            payload={},
        )
        await message_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Check that delivery time is recorded
        metrics = message_bus.get_metrics()
        assert metrics.avg_delivery_time_ms > 0


class TestSubscriptionManagement:
    """Test subscription management."""

    @pytest.mark.asyncio
    async def test_get_subscriptions(self, message_bus):
        """Test retrieving active subscriptions."""

        async def handler1(event):
            pass

        async def handler2(event):
            pass

        # Subscribe multiple handlers
        await message_bus.subscribe("test.sub1", handler1, "sub-1")
        await message_bus.subscribe("test.sub1", handler2, "sub-2")
        await message_bus.subscribe("test.sub2", handler1, "sub-3")

        # Get all subscriptions
        subs = message_bus.get_subscriptions()
        assert "test.sub1" in subs
        assert "test.sub2" in subs
        assert len(subs["test.sub1"]) == 2
        assert len(subs["test.sub2"]) == 1

    @pytest.mark.asyncio
    async def test_max_handlers_per_topic(self):
        """Test maximum handlers per topic limit."""
        config = MessageBusConfig(max_handlers_per_topic=2)
        bus = InMemoryMessageBus(config)
        await bus.start()

        async def handler(event):
            pass

        # Add maximum allowed handlers
        await bus.subscribe("test.limit", handler, "sub-1")
        await bus.subscribe("test.limit", handler, "sub-2")

        # Third handler should raise ValueError
        with pytest.raises(ValueError, match="Maximum handlers"):
            await bus.subscribe("test.limit", handler, "sub-3")

        await bus.close()


class TestFilterFunction:
    """Test event filtering."""

    @pytest.mark.asyncio
    async def test_filter_function(self, message_bus):
        """Test that filter function correctly filters events."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        # Filter: only high priority events
        def high_priority_filter(event):
            return event.priority == EventPriority.HIGH

        # Subscribe with filter
        await message_bus.subscribe(
            "test.filter",
            handler,
            "filter-test",
            filter_fn=high_priority_filter,
        )

        # Publish events with different priorities
        high_event = create_event(
            event_type="test.filter",
            source_agent="test",
            payload={"id": 1},
            priority=EventPriority.HIGH,
        )
        low_event = create_event(
            event_type="test.filter",
            source_agent="test",
            payload={"id": 2},
            priority=EventPriority.LOW,
        )

        await message_bus.publish(high_event)
        await message_bus.publish(low_event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Should only receive high priority event
        assert len(received_events) == 1
        assert received_events[0].payload["id"] == 1
