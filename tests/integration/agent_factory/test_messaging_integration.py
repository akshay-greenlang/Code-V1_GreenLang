# -*- coding: utf-8 -*-
"""
Integration tests for the Agent Factory Messaging Protocol.

Tests end-to-end message delivery patterns, serialization roundtrips,
acknowledgment flows, TTL expiration, dead letter routing,
correlation propagation, and channel lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from tests.unit.agent_factory.test_messaging_protocol import (
    Channel,
    MessageEnvelope,
    MessageRouter,
    RoutingPattern,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def router() -> MessageRouter:
    return MessageRouter()


# ============================================================================
# Tests
# ============================================================================


class TestMessagingIntegration:
    """Integration tests for the messaging system."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_point_to_point_delivery(
        self, router: MessageRouter
    ) -> None:
        """Point-to-point message is delivered to exact destination."""
        received: List[MessageEnvelope] = []
        router.register_agent("calc-agent", AsyncMock(side_effect=lambda m: received.append(m)))
        router.register_agent("report-agent", AsyncMock())

        msg = MessageEnvelope(
            source="intake",
            destination="calc-agent",
            payload={"emissions_kg": 1000},
            pattern=RoutingPattern.POINT_TO_POINT,
        )
        await router.route(msg)
        assert len(received) == 1
        assert received[0].payload["emissions_kg"] == 1000

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pub_sub_multiple_subscribers(
        self, router: MessageRouter
    ) -> None:
        """Pub/sub delivers to all subscribers on the channel."""
        ch = router.create_channel("emission-events")
        counts = {"a": 0, "b": 0, "c": 0}

        async def handler_a(m: MessageEnvelope) -> None:
            counts["a"] += 1

        async def handler_b(m: MessageEnvelope) -> None:
            counts["b"] += 1

        async def handler_c(m: MessageEnvelope) -> None:
            counts["c"] += 1

        ch.subscribe("sub-a", handler_a)
        ch.subscribe("sub-b", handler_b)
        ch.subscribe("sub-c", handler_c)

        for i in range(3):
            msg = MessageEnvelope(
                source="producer",
                topic="emission-events",
                payload={"batch": i},
                pattern=RoutingPattern.PUB_SUB,
            )
            await router.route(msg)

        assert counts["a"] == 3
        assert counts["b"] == 3
        assert counts["c"] == 3

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_broadcast_to_all_agents(
        self, router: MessageRouter
    ) -> None:
        """Broadcast reaches every registered agent."""
        agent_received: Dict[str, int] = {}

        for name in ["agent-a", "agent-b", "agent-c"]:
            agent_received[name] = 0

            async def handler(m: MessageEnvelope, n=name) -> None:
                agent_received[n] += 1

            router.register_agent(name, handler)

        msg = MessageEnvelope(
            source="system",
            payload={"command": "reload-config"},
            pattern=RoutingPattern.BROADCAST,
        )
        await router.route(msg)

        for name in agent_received:
            assert agent_received[name] == 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_request_response_pattern(
        self, router: MessageRouter
    ) -> None:
        """Request-response delivers with correlation ID."""
        responses: List[MessageEnvelope] = []

        async def responder(m: MessageEnvelope) -> None:
            responses.append(m)

        router.register_agent("calc-agent", responder)

        req = MessageEnvelope(
            source="api-gateway",
            destination="calc-agent",
            payload={"query": "scope1_total"},
            pattern=RoutingPattern.REQUEST_RESPONSE,
            reply_to="api-gateway",
            correlation_id="req-abc",
        )
        result = await router.route(req)
        assert result is True
        assert len(responses) == 1
        assert responses[0].correlation_id == "req-abc"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_message_acknowledgment_flow(
        self, router: MessageRouter
    ) -> None:
        """Acknowledgment resolves the waiting future."""
        msg = MessageEnvelope(source="a", destination="b")

        async def ack_in_background():
            await asyncio.sleep(0.02)
            await router.acknowledge(msg.message_id)

        task = asyncio.create_task(ack_in_background())
        result = await router.wait_for_ack(msg.message_id, timeout=1.0)
        assert result is True
        await task

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_message_ttl_expiration(
        self, router: MessageRouter
    ) -> None:
        """Expired messages are rejected and moved to DLQ."""
        router.register_agent("target", AsyncMock())
        msg = MessageEnvelope(
            source="a",
            destination="target",
            ttl_seconds=0.0,
            created_at=time.time() - 100,
        )
        result = await router.route(msg)
        assert result is False
        assert len(router.dlq) == 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_dead_letter_routing(
        self, router: MessageRouter
    ) -> None:
        """Unroutable messages accumulate in the DLQ."""
        for i in range(3):
            msg = MessageEnvelope(
                source="a",
                destination=f"unknown-{i}",
                pattern=RoutingPattern.POINT_TO_POINT,
            )
            await router.route(msg)

        assert len(router.dlq) == 3

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_correlation_id_propagation(
        self, router: MessageRouter
    ) -> None:
        """Correlation IDs are preserved through routing."""
        forwarded: List[str] = []

        async def handler(m: MessageEnvelope) -> None:
            forwarded.append(m.correlation_id or "")

        router.register_agent("downstream", handler)

        msg = MessageEnvelope(
            source="upstream",
            destination="downstream",
            correlation_id="corr-xyz-123",
            pattern=RoutingPattern.POINT_TO_POINT,
        )
        await router.route(msg)
        assert forwarded == ["corr-xyz-123"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_serialization_roundtrip(self) -> None:
        """JSON serialization roundtrip preserves all fields."""
        original = MessageEnvelope(
            source="agent-a",
            destination="agent-b",
            topic="test-topic",
            pattern=RoutingPattern.PUB_SUB,
            payload={"key": "value", "nested": {"a": 1}},
            headers={"X-Custom": "header-val"},
            correlation_id="corr-001",
            reply_to="agent-a",
            ttl_seconds=60.0,
        )
        json_str = original.to_json()
        restored = MessageEnvelope.from_json(json_str)

        assert restored.source == original.source
        assert restored.destination == original.destination
        assert restored.topic == original.topic
        assert restored.pattern == original.pattern
        assert restored.payload == original.payload
        assert restored.headers == original.headers
        assert restored.correlation_id == original.correlation_id
        assert restored.reply_to == original.reply_to

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_channel_lifecycle(
        self, router: MessageRouter
    ) -> None:
        """Channels can be created, subscribed to, and unsubscribed."""
        ch = router.create_channel("lifecycle-test")
        assert ch.subscriber_count == 0

        ch.subscribe("sub-1", AsyncMock())
        ch.subscribe("sub-2", AsyncMock())
        assert ch.subscriber_count == 2

        ch.unsubscribe("sub-1")
        assert ch.subscriber_count == 1

        ch.unsubscribe("sub-2")
        assert ch.subscriber_count == 0
