# -*- coding: utf-8 -*-
"""
GreenLang Agent Factory - Inter-Agent Messaging (INFRA-010).

Dual-transport messaging module for inter-agent communication:

- **Durable channels** (Redis Streams): at-least-once delivery with
  consumer groups, acknowledgment tracking, and dead-letter queues.
  Used for agent-to-agent work requests, task delegation, and
  pipeline steps.
- **Ephemeral channels** (Redis Pub/Sub): best-effort, fire-and-forget.
  Used for config change notifications, telemetry signals, health
  broadcasts, and metric events.

Sub-modules:
    protocol        - Message envelope, types, and delivery status.
    serialization   - JSON and MessagePack codecs.
    acknowledgment  - Acknowledgment tracking and DLQ management.
    router          - Dual-transport message routing.
    channels        - Named channel registry.

Quick start:
    >>> from greenlang.infrastructure.agent_factory.messaging import (
    ...     MessageEnvelope,
    ...     MessageRouter,
    ...     ChannelManager,
    ...     ChannelType,
    ... )
    >>> router = MessageRouter(redis_client)
    >>> await router.initialize()
    >>> receipt = await router.send_durable(
    ...     MessageEnvelope.command("intake", "carbon", {"scope": 1})
    ... )

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

# -- Protocol ----------------------------------------------------------------
from greenlang.infrastructure.agent_factory.messaging.protocol import (
    ChannelType,
    DeliveryStatus,
    MessageEnvelope,
    MessageReceipt,
    MessageType,
    PendingMessage,
)

# -- Serialization -----------------------------------------------------------
from greenlang.infrastructure.agent_factory.messaging.serialization import (
    MessageSerializer,
    SerializationFormat,
)

# -- Acknowledgment ----------------------------------------------------------
from greenlang.infrastructure.agent_factory.messaging.acknowledgment import (
    AckTrackerConfig,
    AckTrackerMetrics,
    AcknowledgmentTracker,
)

# -- Router ------------------------------------------------------------------
from greenlang.infrastructure.agent_factory.messaging.router import (
    DurableTransport,
    EphemeralTransport,
    MessageHandler,
    MessageRouter,
    RouterMetrics,
)

# -- Channels ----------------------------------------------------------------
from greenlang.infrastructure.agent_factory.messaging.channels import (
    Channel,
    ChannelConfig,
    ChannelManager,
)


__all__ = [
    # Protocol
    "ChannelType",
    "DeliveryStatus",
    "MessageEnvelope",
    "MessageReceipt",
    "MessageType",
    "PendingMessage",
    # Serialization
    "MessageSerializer",
    "SerializationFormat",
    # Acknowledgment
    "AckTrackerConfig",
    "AckTrackerMetrics",
    "AcknowledgmentTracker",
    # Router
    "DurableTransport",
    "EphemeralTransport",
    "MessageHandler",
    "MessageRouter",
    "RouterMetrics",
    # Channels
    "Channel",
    "ChannelConfig",
    "ChannelManager",
]
