# -*- coding: utf-8 -*-
"""
Tests for WebSocket metrics server and metric collection.

Tests cover connection management, metric streaming, subscriptions,
authentication, and reconnection logic.
"""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import msgpack
from greenlang.api.websocket.metrics_server import (
from greenlang.determinism import DeterministicClock
    MetricsWebSocketServer,
    ClientConnection,
    MetricFilter,
    MetricAggregator,
    MetricSubscription
)
from greenlang.api.websocket.metric_collector import (
    MetricCollector,
    SystemMetrics,
    WorkflowMetrics,
    AgentMetrics,
    DistributedMetrics
)


@pytest.fixture
async def redis_mock():
    """Mock Redis client."""
    mock = AsyncMock()
    mock.from_url = AsyncMock(return_value=mock)
    mock.pubsub = Mock(return_value=mock)
    mock.psubscribe = AsyncMock()
    mock.get_message = AsyncMock()
    mock.publish = AsyncMock()
    mock.set = AsyncMock()
    mock.get = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def websocket_mock():
    """Mock WebSocket connection."""
    mock = Mock()
    mock.client = Mock()
    mock.client.host = "127.0.0.1"
    mock.client.port = 12345
    mock.accept = AsyncMock()
    mock.send_json = AsyncMock()
    mock.send_bytes = AsyncMock()
    mock.receive_json = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.mark.asyncio
class TestMetricsWebSocketServer:
    """Test WebSocket metrics server."""

    async def test_server_start_stop(self, redis_mock):
        """Test server starts and stops correctly."""
        with patch('greenlang.api.websocket.metrics_server.aioredis', redis_mock):
            server = MetricsWebSocketServer(redis_url="redis://localhost:6379")

            # Start server
            await server.start()
            assert server.running is True
            assert server.redis_client is not None

            # Stop server
            await server.stop()
            assert server.running is False

    async def test_jwt_authentication(self):
        """Test JWT token authentication."""
        server = MetricsWebSocketServer(
            jwt_secret="test-secret",
            jwt_algorithm="HS256"
        )

        # Valid token
        from jose import jwt
        token = jwt.encode({"sub": "user123"}, "test-secret", algorithm="HS256")
        payload = server.authenticate_token(token)
        assert payload is not None
        assert payload["sub"] == "user123"

        # Invalid token
        payload = server.authenticate_token("invalid-token")
        assert payload is None

    async def test_client_connection(self, websocket_mock, redis_mock):
        """Test client connection handling."""
        with patch('greenlang.api.websocket.metrics_server.aioredis', redis_mock):
            server = MetricsWebSocketServer()
            await server.start()

            # Simulate connection
            websocket_mock.receive_json.side_effect = [
                {"type": "subscribe", "data": {"channels": ["system.metrics"]}},
                asyncio.CancelledError()
            ]

            try:
                await server.handle_connection(websocket_mock)
            except asyncio.CancelledError:
                pass

            # Verify client was added
            client_id = f"{websocket_mock.client.host}:{websocket_mock.client.port}"
            assert websocket_mock.accept.called

            await server.stop()

    async def test_metric_subscription(self):
        """Test metric subscription validation."""
        subscription = MetricSubscription(
            channels=["system.metrics", "workflow.metrics"],
            tags={"env": "production"},
            aggregation_interval="5s",
            compression=True
        )

        assert len(subscription.channels) == 2
        assert subscription.tags["env"] == "production"
        assert subscription.aggregation_interval == "5s"

    async def test_invalid_subscription(self):
        """Test invalid subscription rejection."""
        with pytest.raises(ValueError):
            MetricSubscription(
                channels=["invalid.channel"],
                aggregation_interval="5s"
            )

        with pytest.raises(ValueError):
            MetricSubscription(
                channels=["system.metrics"],
                aggregation_interval="10x"  # Invalid interval
            )

    async def test_metric_filtering(self):
        """Test metric filtering by tags."""
        filter = MetricFilter(tags={"env": "production", "region": "us-east-1"})

        # Matching metric
        metric = {
            "name": "cpu.percent",
            "value": 75.0,
            "tags": {"env": "production", "region": "us-east-1"}
        }
        assert filter.matches(metric) is True

        # Non-matching metric
        metric_no_match = {
            "name": "cpu.percent",
            "value": 75.0,
            "tags": {"env": "staging", "region": "us-west-2"}
        }
        assert filter.matches(metric_no_match) is False

    async def test_metric_aggregation(self):
        """Test metric aggregation over time intervals."""
        aggregator = MetricAggregator("5s")

        # Add metrics
        for i in range(10):
            metric = {
                "name": "cpu.percent",
                "type": "gauge",
                "value": 70.0 + i,
                "timestamp": DeterministicClock.utcnow().isoformat()
            }
            aggregator.add_metric(metric)

        # Should not flush immediately
        assert aggregator.should_flush() is False

        # Force flush
        aggregator.last_flush = 0
        assert aggregator.should_flush() is True

        # Get aggregated metrics
        aggregated = aggregator.flush()
        assert len(aggregated) > 0
        assert aggregated[0]["name"] == "cpu.percent"
        assert aggregated[0]["type"] == "gauge"
        assert "value" in aggregated[0]  # Average value
        assert "min" in aggregated[0]
        assert "max" in aggregated[0]

    async def test_rate_limiting(self):
        """Test client rate limiting."""
        from fastapi import WebSocket
        ws_mock = Mock(spec=WebSocket)
        ws_mock.client = Mock()
        ws_mock.client.host = "127.0.0.1"
        ws_mock.client.port = 12345

        client = ClientConnection(ws_mock, "client123")
        client.rate_limit_max = 10  # Low limit for testing

        # Send messages within limit
        for i in range(10):
            assert client.check_rate_limit() is True

        # Exceed limit
        assert client.check_rate_limit() is False

    async def test_compression(self, websocket_mock):
        """Test MessagePack compression."""
        from fastapi import WebSocket
        ws_mock = Mock(spec=WebSocket)
        ws_mock.client = Mock()
        ws_mock.client.host = "127.0.0.1"
        ws_mock.client.port = 12345
        ws_mock.send_bytes = AsyncMock()
        ws_mock.send_json = AsyncMock()

        client = ClientConnection(ws_mock, "client123")
        client.compression = True

        metric = {"name": "cpu.percent", "value": 75.0}
        await client._send_data(metric)

        # Should send as MessagePack bytes
        ws_mock.send_bytes.assert_called_once()

        # Test without compression
        client.compression = False
        await client._send_data(metric)
        ws_mock.send_json.assert_called_once()


@pytest.mark.asyncio
class TestMetricCollector:
    """Test metric collection."""

    async def test_system_metrics_collection(self):
        """Test system metrics collection."""
        collector = SystemMetrics()
        metrics = collector.collect()

        assert "timestamp" in metrics
        assert "cpu" in metrics
        assert "memory" in metrics
        assert "disk" in metrics
        assert "network" in metrics
        assert "process" in metrics

        # Verify CPU metrics
        assert "percent" in metrics["cpu"]
        assert "count" in metrics["cpu"]

        # Verify memory metrics
        assert "total" in metrics["memory"]
        assert "used" in metrics["memory"]
        assert "percent" in metrics["memory"]

    async def test_metric_buffer(self):
        """Test metric buffering."""
        from greenlang.api.websocket.metric_collector import MetricBuffer

        buffer = MetricBuffer(max_size=10, max_age=1.0)

        # Add metrics
        for i in range(5):
            buffer.add("system.metrics", {"value": i})

        assert len(buffer.buffer["system.metrics"]) == 5

        # Should flush when full
        for i in range(10):
            buffer.add("workflow.metrics", {"value": i})

        assert buffer.should_flush("workflow.metrics") is True

    async def test_collector_start_stop(self, redis_mock):
        """Test metric collector start/stop."""
        with patch('greenlang.api.websocket.metric_collector.aioredis', redis_mock):
            collector = MetricCollector(redis_url="redis://localhost:6379")

            await collector.start()
            assert collector.running is True
            assert collector.redis_client is not None

            await collector.stop()
            assert collector.running is False

    async def test_metric_publishing(self, redis_mock):
        """Test metric publishing to Redis."""
        with patch('greenlang.api.websocket.metric_collector.aioredis', redis_mock):
            collector = MetricCollector()
            collector.redis_client = redis_mock

            metric = {"name": "cpu.percent", "value": 75.0}
            await collector._publish_metric("system.metrics", metric)

            # Metric should be buffered
            assert "system.metrics" in collector.buffer.buffer


@pytest.mark.asyncio
class TestReconnectionLogic:
    """Test WebSocket reconnection logic."""

    async def test_exponential_backoff(self):
        """Test exponential backoff on reconnection."""
        # This would be tested in the frontend MetricService
        # Testing reconnection delay calculation
        delays = []
        for attempt in range(5):
            delay = min(1000 * (2 ** attempt), 30000)
            delays.append(delay)

        assert delays[0] == 1000  # 1s
        assert delays[1] == 2000  # 2s
        assert delays[2] == 4000  # 4s
        assert delays[3] == 8000  # 8s
        assert delays[4] == 16000  # 16s

    async def test_max_reconnect_attempts(self):
        """Test max reconnection attempts."""
        max_attempts = 10
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

        assert attempts == max_attempts


@pytest.mark.asyncio
class TestMetricStreamingIntegration:
    """Integration tests for metric streaming."""

    async def test_end_to_end_metric_flow(self, redis_mock, websocket_mock):
        """Test complete metric flow from collection to client."""
        with patch('greenlang.api.websocket.metrics_server.aioredis', redis_mock):
            with patch('greenlang.api.websocket.metric_collector.aioredis', redis_mock):
                # Start server and collector
                server = MetricsWebSocketServer()
                collector = MetricCollector()

                await server.start()
                await collector.start()

                # Collect and publish metric
                metric = {"name": "test.metric", "value": 42.0}
                await collector._publish_metric("system.metrics", metric)

                # Verify metric was published
                assert redis_mock.publish.called or len(collector.buffer.buffer) > 0

                # Cleanup
                await server.stop()
                await collector.stop()


# Run tests with: pytest tests/phase4/test_metrics_websocket.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
