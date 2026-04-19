"""
Unit tests for SSE streaming manager and event types.

Test coverage:
- Event creation and SSE format serialization
- Agent status updates
- Job progress tracking
- Alarm notifications
- Metrics streaming
- Client subscription management
- Event history replay
- Heartbeat functionality
- Stream cleanup and lifecycle
"""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from greenlang.infrastructure.api.sse_streaming import (
    SSEStreamManager,
    SSEStreamConfig,
    SSEStreamEvent,
    EventTypeEnum,
    AgentStatusEnum,
    AlarmSeverityEnum,
    AgentStatusUpdate,
    CalculationProgress,
    AlarmUpdate,
    MetricsUpdate,
    EventStream,
    StreamSubscription,
)


class TestSSEStreamEvent:
    """Test SSE event model and serialization."""

    def test_event_creation(self):
        """Test creating SSE event."""
        event = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"status": "RUNNING"},
            channel="agent:001"
        )

        assert event.event_type == EventTypeEnum.AGENT_STATUS
        assert event.data == {"status": "RUNNING"}
        assert event.channel == "agent:001"
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_event_sse_format(self):
        """Test SSE wire format generation."""
        event = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"status": "RUNNING", "progress": 50},
            channel="agent:001",
            event_id="test-id-123"
        )

        sse_format = event.to_sse_format()

        # Check required fields
        assert "id: test-id-123" in sse_format
        assert "event: agent.status" in sse_format
        assert "data:" in sse_format
        assert "retry: 3000" in sse_format
        # Should end with double newline
        assert sse_format.endswith("\n\n")

    def test_event_hash_calculation(self):
        """Test SHA-256 hash calculation."""
        event = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"status": "RUNNING"},
            channel="agent:001"
        )

        hash1 = event.calculate_hash()
        assert len(hash1) == 64  # SHA-256 hex is 64 chars
        assert hash1.isalnum()

        # Same event should have same hash
        hash2 = event.calculate_hash()
        assert hash1 == hash2

    def test_event_json_serialization(self):
        """Test JSON data serialization in SSE format."""
        data = {"status": "RUNNING", "details": {"cpu": 85.5, "mem": 72.3}}
        event = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data=data,
            channel="agent:001"
        )

        sse_format = event.to_sse_format()
        # Should contain JSON data
        assert "RUNNING" in sse_format
        assert "85.5" in sse_format


class TestEventModels:
    """Test domain event models."""

    def test_agent_status_update(self):
        """Test agent status update model."""
        update = AgentStatusUpdate(
            agent_id="agent:001",
            status=AgentStatusEnum.RUNNING,
            progress_percent=50,
            current_step="calculating_emissions"
        )

        assert update.agent_id == "agent:001"
        assert update.status == AgentStatusEnum.RUNNING
        assert update.progress_percent == 50

    def test_agent_status_validation(self):
        """Test agent status progress validation."""
        # Valid
        update = AgentStatusUpdate(
            agent_id="agent:001",
            status=AgentStatusEnum.RUNNING,
            progress_percent=100
        )
        assert update.progress_percent == 100

        # Invalid - over 100
        with pytest.raises(ValueError):
            AgentStatusUpdate(
                agent_id="agent:001",
                status=AgentStatusEnum.RUNNING,
                progress_percent=101
            )

    def test_calculation_progress(self):
        """Test job progress model."""
        progress = CalculationProgress(
            job_id="job:123",
            agent_id="agent:001",
            progress_percent=75,
            current_step="thermal_analysis",
            step_number=3,
            total_steps=4,
            elapsed_seconds=120.5,
            estimated_total_seconds=160.0
        )

        assert progress.job_id == "job:123"
        assert progress.progress_percent == 75
        assert progress.step_number == 3

    def test_alarm_update(self):
        """Test alarm update model."""
        alarm = AlarmUpdate(
            alarm_id="alarm:001",
            agent_id="agent:001",
            severity=AlarmSeverityEnum.CRITICAL,
            message="Pressure exceeded maximum",
            parameter="pressure",
            current_value=9.5,
            threshold_value=9.0
        )

        assert alarm.severity == AlarmSeverityEnum.CRITICAL
        assert alarm.current_value == 9.5

    def test_metrics_update(self):
        """Test metrics update model."""
        metrics = MetricsUpdate(
            source_id="agent:001",
            metrics={"temperature": 150.5, "pressure": 8.5, "efficiency": 0.92},
            unit="SI"
        )

        assert metrics.source_id == "agent:001"
        assert metrics.metrics["temperature"] == 150.5
        assert len(metrics.metrics) == 3


class TestEventStream:
    """Test event stream channel."""

    @pytest.mark.asyncio
    async def test_stream_subscribe(self):
        """Test client subscription."""
        stream = EventStream("test-channel")

        queue, replay = await stream.subscribe("client:001")

        assert isinstance(queue, asyncio.Queue)
        assert replay == []
        assert stream.subscriber_count == 1

    @pytest.mark.asyncio
    async def test_stream_broadcast(self):
        """Test event broadcasting."""
        stream = EventStream("test-channel")

        queue1, _ = await stream.subscribe("client:001")
        queue2, _ = await stream.subscribe("client:002")

        event = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"status": "RUNNING"},
            channel="test-channel"
        )

        delivered = await stream.broadcast(event)

        assert delivered == 2
        assert queue1.qsize() == 1
        assert queue2.qsize() == 1

    @pytest.mark.asyncio
    async def test_stream_event_history(self):
        """Test event history tracking."""
        stream = EventStream("test-channel", max_history=3)

        event1 = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"n": 1},
            channel="test"
        )
        event2 = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"n": 2},
            channel="test"
        )
        event3 = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"n": 3},
            channel="test"
        )
        event4 = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"n": 4},
            channel="test"
        )

        await stream.broadcast(event1)
        await stream.broadcast(event2)
        await stream.broadcast(event3)
        await stream.broadcast(event4)

        # Should keep only last 3
        assert len(stream._event_history) == 3
        assert stream._event_history[0].data["n"] == 2

    @pytest.mark.asyncio
    async def test_stream_event_replay(self):
        """Test event replay on reconnection."""
        stream = EventStream("test-channel")

        # Broadcast events
        event1 = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"n": 1},
            channel="test",
            event_id="id-1"
        )
        event2 = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"n": 2},
            channel="test",
            event_id="id-2"
        )
        event3 = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"n": 3},
            channel="test",
            event_id="id-3"
        )

        await stream.broadcast(event1)
        await stream.broadcast(event2)
        await stream.broadcast(event3)

        # Reconnect after event1
        queue, replay = await stream.subscribe("client:002", last_event_id="id-1")

        # Should replay events 2 and 3
        assert len(replay) == 2
        assert replay[0].event_id == "id-2"
        assert replay[1].event_id == "id-3"

    @pytest.mark.asyncio
    async def test_stream_unsubscribe(self):
        """Test client unsubscribe."""
        stream = EventStream("test-channel")

        queue, _ = await stream.subscribe("client:001")
        assert stream.subscriber_count == 1

        await stream.unsubscribe("client:001")
        assert stream.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_stream_queue_full_handling(self):
        """Test handling of full client queue."""
        stream = EventStream("test-channel", max_queue_size=1)

        queue, _ = await stream.subscribe("client:001")

        event1 = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"n": 1},
            channel="test"
        )
        event2 = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"n": 2},
            channel="test"
        )

        # First event succeeds
        delivered = await stream.broadcast(event1)
        assert delivered == 1

        # Second event fails (queue full)
        delivered = await stream.broadcast(event2)
        assert delivered == 0


class TestSSEStreamManager:
    """Test SSE stream manager."""

    def test_manager_initialization(self):
        """Test manager creation."""
        config = SSEStreamConfig(
            heartbeat_interval_seconds=30,
            max_clients_per_stream=100
        )
        manager = SSEStreamManager(config)

        assert manager.config == config
        assert len(manager._streams) == 0
        assert len(manager._subscriptions) == 0

    def test_manager_without_fastapi(self):
        """Test manager requires FastAPI."""
        with patch.dict('sys.modules', {'fastapi': None}):
            with pytest.raises(ImportError):
                SSEStreamManager()

    @pytest.mark.asyncio
    async def test_send_agent_status(self):
        """Test sending agent status update."""
        manager = SSEStreamManager()
        await manager.start()

        count = await manager.send_agent_status(
            agent_id="agent:001",
            status=AgentStatusEnum.RUNNING,
            progress_percent=50,
            current_step="analysis"
        )

        # Should have created streams for agent and global
        assert "agent:agent:001" in manager._streams
        assert "global" in manager._streams

        await manager.stop()

    @pytest.mark.asyncio
    async def test_send_job_progress(self):
        """Test sending job progress."""
        manager = SSEStreamManager()
        await manager.start()

        count = await manager.send_job_progress(
            job_id="job:123",
            agent_id="agent:001",
            progress_percent=75,
            current_step="thermal_analysis",
            step_number=3,
            total_steps=4
        )

        # Should have created streams for job, agent, and global
        assert "job:job:123" in manager._streams

        await manager.stop()

    @pytest.mark.asyncio
    async def test_send_alarm_update(self):
        """Test sending alarm update."""
        manager = SSEStreamManager()
        await manager.start()

        count = await manager.send_alarm_update(
            alarm_id="alarm:001",
            agent_id="agent:001",
            severity=AlarmSeverityEnum.CRITICAL,
            message="Pressure exceeded"
        )

        assert "agent:agent:001" in manager._streams

        await manager.stop()

    @pytest.mark.asyncio
    async def test_send_metrics(self):
        """Test sending metrics update."""
        manager = SSEStreamManager()
        await manager.start()

        count = await manager.send_metrics(
            source_id="agent:001",
            metrics={"temperature": 150.5, "pressure": 8.5}
        )

        assert "agent:agent:001" in manager._streams

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_connected_clients(self):
        """Test querying connected clients."""
        manager = SSEStreamManager()

        # Add some mock subscriptions
        manager._subscriptions["client:001"] = StreamSubscription(
            client_id="client:001",
            subscribed_channels={"agent:001"},
            remote_addr="127.0.0.1"
        )

        clients = manager.get_connected_clients()

        assert len(clients) == 1
        assert clients[0]["client_id"] == "client:001"
        assert "agent:001" in clients[0]["channels"]

    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """Test getting manager statistics."""
        manager = SSEStreamManager()
        await manager.start()

        await manager.send_agent_status("agent:001", AgentStatusEnum.RUNNING)

        stats = manager.get_statistics()

        assert "total_clients" in stats
        assert "total_streams" in stats
        assert "active_streams" in stats
        assert "config" in stats
        assert stats["config"]["heartbeat_interval_seconds"] == 30

        await manager.stop()

    @pytest.mark.asyncio
    async def test_close_stream(self):
        """Test closing a client stream."""
        manager = SSEStreamManager()

        manager._subscriptions["client:001"] = StreamSubscription(
            client_id="client:001",
            subscribed_channels={"agent:001"}
        )

        result = manager.close_stream("client:001")
        assert result is True

        result = manager.close_stream("client:999")
        assert result is False

    @pytest.mark.asyncio
    async def test_manager_lifecycle(self):
        """Test manager start and stop."""
        manager = SSEStreamManager()

        assert manager._cleanup_task is None
        assert manager._shutdown is False

        await manager.start()
        assert manager._cleanup_task is not None
        assert manager._shutdown is False

        await manager.stop()
        assert manager._shutdown is True

    @pytest.mark.asyncio
    async def test_cleanup_stale_subscriptions(self):
        """Test cleanup of stale subscriptions."""
        config = SSEStreamConfig(client_timeout_seconds=1)
        manager = SSEStreamManager(config)

        # Create old subscription
        old_time = datetime.utcnow()
        old_sub = StreamSubscription(
            client_id="client:old",
            subscribed_channels={"agent:001"},
            connected_at=old_time,
            last_heartbeat=old_time
        )

        # Create recent subscription
        manager._subscriptions["client:old"] = old_sub
        manager._subscriptions["client:new"] = StreamSubscription(
            client_id="client:new",
            subscribed_channels={"agent:001"}
        )

        # Manually trigger cleanup
        await manager._cleanup_loop.__self__._cleanup_loop()


class TestSSEIntegration:
    """Integration tests for SSE streaming."""

    @pytest.mark.asyncio
    async def test_full_event_flow(self):
        """Test complete event flow from send to receive."""
        manager = SSEStreamManager()
        await manager.start()

        # Send status update
        await manager.send_agent_status(
            agent_id="agent:001",
            status=AgentStatusEnum.RUNNING,
            progress_percent=25
        )

        # Send progress update
        await manager.send_job_progress(
            job_id="job:001",
            agent_id="agent:001",
            progress_percent=50,
            current_step="analysis"
        )

        # Send alarm
        await manager.send_alarm_update(
            alarm_id="alarm:001",
            agent_id="agent:001",
            severity=AlarmSeverityEnum.WARNING,
            message="Threshold approaching"
        )

        # Send metrics
        await manager.send_metrics(
            source_id="agent:001",
            metrics={"temp": 150.5, "pressure": 8.5}
        )

        # Check streams were created
        assert "agent:agent:001" in manager._streams
        assert "job:job:001" in manager._streams
        assert "global" in manager._streams

        await manager.stop()

    @pytest.mark.asyncio
    async def test_event_format_correctness(self):
        """Test that events are formatted correctly for SSE."""
        manager = SSEStreamManager()

        # Create an event
        event = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data={"status": "RUNNING", "progress": 50},
            channel="agent:001"
        )

        sse_text = event.to_sse_format()

        # Parse and verify
        lines = sse_text.strip().split("\n")
        assert any("id:" in line for line in lines)
        assert any("event:" in line for line in lines)
        assert any("data:" in line for line in lines)


class TestStreamConfiguration:
    """Test configuration options."""

    def test_custom_config(self):
        """Test custom configuration."""
        config = SSEStreamConfig(
            heartbeat_interval_seconds=60,
            client_timeout_seconds=600,
            max_clients_per_stream=500,
            max_queue_size=1000,
            api_prefix="/custom/sse"
        )

        assert config.heartbeat_interval_seconds == 60
        assert config.client_timeout_seconds == 600
        assert config.max_clients_per_stream == 500

    def test_default_config(self):
        """Test default configuration."""
        config = SSEStreamConfig()

        assert config.heartbeat_interval_seconds == 30
        assert config.client_timeout_seconds == 300
        assert config.max_clients_per_stream == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
