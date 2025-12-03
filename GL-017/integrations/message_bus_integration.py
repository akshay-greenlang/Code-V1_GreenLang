"""
GL-017 CONDENSYNC Message Bus Integration Module

Inter-agent communication for condenser optimization agent coordination,
event publishing, and alert broadcasting.

Events:
- condenser_performance_update: Real-time performance metrics
- cleaning_required: Tube cleaning notification
- optimization_complete: Optimization cycle completion
- alarm_triggered: Condenser alarm events

Author: GreenLang AI Platform
Version: 1.0.0
"""

import asyncio
import logging
import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class MessageBusError(Exception):
    """Base exception for message bus integration."""
    pass


class MessageBusConnectionError(MessageBusError):
    """Raised when message bus connection fails."""
    pass


class MessagePublishError(MessageBusError):
    """Raised when message publishing fails."""
    pass


class MessageSubscribeError(MessageBusError):
    """Raised when subscription fails."""
    pass


# =============================================================================
# Enums
# =============================================================================

class EventType(Enum):
    """Event types for condenser agent."""
    CONDENSER_PERFORMANCE_UPDATE = "condenser_performance_update"
    CLEANING_REQUIRED = "cleaning_required"
    OPTIMIZATION_COMPLETE = "optimization_complete"
    ALARM_TRIGGERED = "alarm_triggered"
    INTERLOCK_TRIPPED = "interlock_tripped"
    MODE_CHANGE = "mode_change"
    SETPOINT_CHANGE = "setpoint_change"
    BASELINE_DEVIATION = "baseline_deviation"
    COOLING_TOWER_UPDATE = "cooling_tower_update"
    MAINTENANCE_DUE = "maintenance_due"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AgentType(Enum):
    """GreenLang agent types."""
    CONDENSER_OPTIMIZATION = "GL-017-CONDENSYNC"
    BOILER_EFFICIENCY = "GL-001-THERMOSYNC"
    STEAM_TURBINE = "GL-002-TURBOSYNC"
    FEEDWATER = "GL-003-AQUASYNC"
    COOLING_TOWER = "GL-004-COOLSYNC"
    STEAM_TRAP = "GL-005-TRAPSYNC"
    SUPERVISOR = "GL-SUP-001"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


# =============================================================================
# Data Models
# =============================================================================

class MessageBusConfig(BaseModel):
    """Configuration for message bus integration."""

    broker_url: str = Field(
        default="amqp://localhost:5672",
        description="Message broker URL"
    )
    exchange_name: str = Field(
        default="greenlang.events",
        description="Exchange name for events"
    )
    queue_name: str = Field(
        default="gl-017-condensync",
        description="Queue name for this agent"
    )
    redis_url: Optional[str] = Field(
        default="redis://localhost:6379",
        description="Redis URL for state synchronization"
    )
    connection_timeout: float = Field(
        default=10.0,
        description="Connection timeout in seconds"
    )
    heartbeat_interval: int = Field(
        default=60,
        description="Heartbeat interval in seconds"
    )
    prefetch_count: int = Field(
        default=10,
        description="Prefetch count for message consumption"
    )
    agent_id: str = Field(
        default="GL-017-CONDENSYNC",
        description="Agent identifier"
    )
    enable_persistence: bool = Field(
        default=True,
        description="Enable message persistence"
    )


@dataclass
class CondenserEvent:
    """Condenser-specific event data."""

    event_id: str
    event_type: EventType
    timestamp: datetime
    source_agent: str

    # Event data
    data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source_agent": self.source_agent,
            "data": self.data,
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CondenserEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_agent=data["source_agent"],
            data=data.get("data", {}),
            priority=MessagePriority(data.get("priority", 2)),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            tags=data.get("tags", [])
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class AgentMessage:
    """Inter-agent message."""

    message_id: str
    source_agent: str
    target_agent: Optional[str]  # None for broadcast
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reply_to: Optional[str] = None
    expires_at: Optional[datetime] = None
    priority: MessagePriority = MessagePriority.NORMAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "reply_to": self.reply_to,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "priority": self.priority.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            source_agent=data["source_agent"],
            target_agent=data.get("target_agent"),
            message_type=data["message_type"],
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            reply_to=data.get("reply_to"),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at") else None
            ),
            priority=MessagePriority(data.get("priority", 2))
        )


# =============================================================================
# Message Bus Integration Class
# =============================================================================

class MessageBusIntegration:
    """
    Inter-agent communication via message bus.

    Provides:
    - Event publishing (condenser_performance_update, cleaning_required, etc.)
    - Agent coordination with steam system agents
    - Alert broadcasting
    - State synchronization
    """

    def __init__(self, config: MessageBusConfig):
        """
        Initialize message bus integration.

        Args:
            config: Message bus configuration
        """
        self.config = config

        self._amqp_connection = None
        self._redis_client = None
        self._connected = False

        # Subscriptions
        self._event_handlers: Dict[EventType, List[Callable]] = {}
        self._message_handlers: Dict[str, Callable] = {}

        # Pending responses for request/reply pattern
        self._pending_responses: Dict[str, asyncio.Future] = {}

        # Consumer task
        self._consumer_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "events_published": 0,
            "events_received": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "last_event_published": None,
            "last_event_received": None
        }

        logger.info(
            f"Message Bus Integration initialized for {config.agent_id}"
        )

    @property
    def is_connected(self) -> bool:
        """Check if connected to message bus."""
        return self._connected

    async def connect(self) -> None:
        """
        Establish connection to message bus.

        Raises:
            MessageBusConnectionError: If connection fails
        """
        logger.info(f"Connecting to message bus at {self.config.broker_url}")

        try:
            await self._connect_amqp()
            await self._connect_redis()
            await self._setup_exchanges_and_queues()

            # Start consumer
            self._consumer_task = asyncio.create_task(self._consume_messages())

            self._connected = True
            logger.info("Successfully connected to message bus")

            # Announce agent online
            await self._announce_online()

        except Exception as e:
            logger.error(f"Failed to connect to message bus: {e}")
            raise MessageBusConnectionError(f"Connection failed: {e}")

    async def _connect_amqp(self) -> None:
        """Connect to AMQP broker."""
        # Simulated connection
        self._amqp_connection = {
            "url": self.config.broker_url,
            "connected": False,
            "channel": None
        }

        await asyncio.sleep(0.1)
        self._amqp_connection["connected"] = True
        self._amqp_connection["channel"] = {"id": 1}

    async def _connect_redis(self) -> None:
        """Connect to Redis for state sync."""
        if self.config.redis_url:
            self._redis_client = {
                "url": self.config.redis_url,
                "connected": True
            }

    async def _setup_exchanges_and_queues(self) -> None:
        """Setup AMQP exchanges and queues."""
        # In production: declare exchanges and queues
        logger.debug(f"Setting up exchange: {self.config.exchange_name}")
        logger.debug(f"Setting up queue: {self.config.queue_name}")

    async def _announce_online(self) -> None:
        """Announce agent coming online."""
        event = CondenserEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.MODE_CHANGE,
            timestamp=datetime.utcnow(),
            source_agent=self.config.agent_id,
            data={
                "status": "online",
                "capabilities": [
                    "condenser_optimization",
                    "performance_monitoring",
                    "tube_cleaning_prediction"
                ]
            },
            tags=["agent_status"]
        )

        await self.publish_event(event)

    async def disconnect(self) -> None:
        """Disconnect from message bus."""
        logger.info("Disconnecting from message bus")

        # Announce offline
        if self._connected:
            try:
                event = CondenserEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType.MODE_CHANGE,
                    timestamp=datetime.utcnow(),
                    source_agent=self.config.agent_id,
                    data={"status": "offline"},
                    tags=["agent_status"]
                )
                await self.publish_event(event)
            except Exception:
                pass

        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

        if self._amqp_connection:
            self._amqp_connection["connected"] = False
            self._amqp_connection = None

        if self._redis_client:
            self._redis_client = None

        self._connected = False
        logger.info("Disconnected from message bus")

    async def publish_event(self, event: CondenserEvent) -> None:
        """
        Publish an event to the message bus.

        Args:
            event: Event to publish

        Raises:
            MessagePublishError: If publishing fails
        """
        if not self._connected:
            raise MessagePublishError("Not connected to message bus")

        try:
            # Simulated publish
            routing_key = f"events.{event.event_type.value}"

            logger.debug(
                f"Publishing event {event.event_id} "
                f"type={event.event_type.value} "
                f"routing_key={routing_key}"
            )

            await asyncio.sleep(0.01)  # Simulate network delay

            self._stats["events_published"] += 1
            self._stats["last_event_published"] = datetime.utcnow()

            logger.info(
                f"Published event: {event.event_type.value} "
                f"(id={event.event_id[:8]})"
            )

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Failed to publish event: {e}")
            raise MessagePublishError(f"Publish failed: {e}")

    async def publish_performance_update(
        self,
        vacuum_pressure: float,
        ttd: float,
        cleanliness_factor: float,
        duty_kw: float,
        cw_flow_rate: float,
        cw_inlet_temp: float,
        cw_outlet_temp: float
    ) -> None:
        """
        Publish condenser performance update event.

        Args:
            vacuum_pressure: Vacuum pressure (kPa abs)
            ttd: Terminal temperature difference (degC)
            cleanliness_factor: Cleanliness factor (0-1)
            duty_kw: Heat duty (kW)
            cw_flow_rate: Cooling water flow rate (m3/h)
            cw_inlet_temp: CW inlet temperature (degC)
            cw_outlet_temp: CW outlet temperature (degC)
        """
        event = CondenserEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.CONDENSER_PERFORMANCE_UPDATE,
            timestamp=datetime.utcnow(),
            source_agent=self.config.agent_id,
            data={
                "vacuum_pressure_kpa": vacuum_pressure,
                "ttd_degc": ttd,
                "cleanliness_factor": cleanliness_factor,
                "duty_kw": duty_kw,
                "cw_flow_rate_m3h": cw_flow_rate,
                "cw_inlet_temp_degc": cw_inlet_temp,
                "cw_outlet_temp_degc": cw_outlet_temp,
                "range_degc": cw_outlet_temp - cw_inlet_temp
            },
            tags=["performance", "condenser"]
        )

        await self.publish_event(event)

    async def publish_cleaning_required(
        self,
        cleanliness_factor: float,
        ttd: float,
        urgency_score: float,
        estimated_loss_kw: float,
        recommended_date: Optional[datetime] = None
    ) -> None:
        """
        Publish cleaning required notification.

        Args:
            cleanliness_factor: Current cleanliness factor
            ttd: Terminal temperature difference
            urgency_score: Urgency score (0-100)
            estimated_loss_kw: Estimated performance loss (kW)
            recommended_date: Recommended cleaning date
        """
        alert_level = AlertLevel.INFO
        if urgency_score >= 80:
            alert_level = AlertLevel.ALARM
        elif urgency_score >= 60:
            alert_level = AlertLevel.WARNING

        event = CondenserEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.CLEANING_REQUIRED,
            timestamp=datetime.utcnow(),
            source_agent=self.config.agent_id,
            data={
                "cleanliness_factor": cleanliness_factor,
                "ttd_degc": ttd,
                "urgency_score": urgency_score,
                "estimated_loss_kw": estimated_loss_kw,
                "recommended_date": (
                    recommended_date.isoformat() if recommended_date else None
                ),
                "alert_level": alert_level.value
            },
            priority=(
                MessagePriority.URGENT if urgency_score >= 80
                else MessagePriority.HIGH if urgency_score >= 60
                else MessagePriority.NORMAL
            ),
            tags=["maintenance", "cleaning", "condenser"]
        )

        await self.publish_event(event)

    async def publish_optimization_complete(
        self,
        optimization_id: str,
        actions_taken: List[Dict[str, Any]],
        performance_improvement: float,
        energy_savings_kw: float
    ) -> None:
        """
        Publish optimization cycle completion event.

        Args:
            optimization_id: Optimization cycle identifier
            actions_taken: List of actions taken
            performance_improvement: Performance improvement (%)
            energy_savings_kw: Energy savings achieved (kW)
        """
        event = CondenserEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.OPTIMIZATION_COMPLETE,
            timestamp=datetime.utcnow(),
            source_agent=self.config.agent_id,
            data={
                "optimization_id": optimization_id,
                "actions_taken": actions_taken,
                "performance_improvement_percent": performance_improvement,
                "energy_savings_kw": energy_savings_kw
            },
            tags=["optimization", "condenser"]
        )

        await self.publish_event(event)

    async def publish_alert(
        self,
        alert_level: AlertLevel,
        title: str,
        description: str,
        related_tags: Optional[List[str]] = None,
        recommended_action: Optional[str] = None
    ) -> None:
        """
        Broadcast an alert.

        Args:
            alert_level: Alert severity level
            title: Alert title
            description: Alert description
            related_tags: Related process tags
            recommended_action: Recommended operator action
        """
        priority_map = {
            AlertLevel.INFO: MessagePriority.LOW,
            AlertLevel.WARNING: MessagePriority.NORMAL,
            AlertLevel.ALARM: MessagePriority.HIGH,
            AlertLevel.CRITICAL: MessagePriority.URGENT,
            AlertLevel.EMERGENCY: MessagePriority.CRITICAL
        }

        event = CondenserEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ALARM_TRIGGERED,
            timestamp=datetime.utcnow(),
            source_agent=self.config.agent_id,
            data={
                "alert_level": alert_level.value,
                "title": title,
                "description": description,
                "related_tags": related_tags or [],
                "recommended_action": recommended_action
            },
            priority=priority_map.get(alert_level, MessagePriority.NORMAL),
            tags=["alert", alert_level.value]
        )

        await self.publish_event(event)

    async def send_message(
        self,
        target_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        wait_for_reply: bool = False,
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """
        Send a message to another agent.

        Args:
            target_agent: Target agent identifier
            message_type: Message type
            payload: Message payload
            wait_for_reply: Wait for reply
            timeout: Reply timeout in seconds

        Returns:
            Reply message if wait_for_reply is True
        """
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            source_agent=self.config.agent_id,
            target_agent=target_agent,
            message_type=message_type,
            payload=payload,
            reply_to=self.config.queue_name if wait_for_reply else None
        )

        # Simulated send
        logger.debug(
            f"Sending message {message.message_id[:8]} "
            f"to {target_agent} type={message_type}"
        )

        await asyncio.sleep(0.01)

        self._stats["messages_sent"] += 1

        if wait_for_reply:
            future = asyncio.Future()
            self._pending_responses[message.message_id] = future

            try:
                reply = await asyncio.wait_for(future, timeout=timeout)
                return reply
            except asyncio.TimeoutError:
                del self._pending_responses[message.message_id]
                raise MessageBusError(f"Reply timeout for message {message.message_id}")

        return None

    async def request_cooling_tower_status(self) -> Dict[str, Any]:
        """Request status from cooling tower agent."""
        reply = await self.send_message(
            target_agent=AgentType.COOLING_TOWER.value,
            message_type="status_request",
            payload={"requested_data": ["cell_status", "basin_temp", "approach"]},
            wait_for_reply=True,
            timeout=10.0
        )

        if reply:
            return reply.payload
        return {}

    def subscribe_to_event(
        self,
        event_type: EventType,
        handler: Callable[[CondenserEvent], Awaitable[None]]
    ) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to
            handler: Async handler function
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []

        self._event_handlers[event_type].append(handler)
        logger.debug(f"Subscribed to event type: {event_type.value}")

    def unsubscribe_from_event(
        self,
        event_type: EventType,
        handler: Callable[[CondenserEvent], Awaitable[None]]
    ) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._event_handlers:
            if handler in self._event_handlers[event_type]:
                self._event_handlers[event_type].remove(handler)

    async def _consume_messages(self) -> None:
        """Consume messages from queue."""
        while self._connected:
            try:
                # Simulated message consumption
                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error consuming messages: {e}")
                self._stats["errors"] += 1

    async def _handle_incoming_event(self, event: CondenserEvent) -> None:
        """Handle incoming event."""
        self._stats["events_received"] += 1
        self._stats["last_event_received"] = datetime.utcnow()

        handlers = self._event_handlers.get(event.event_type, [])

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get cached state for an agent from Redis."""
        if not self._redis_client:
            return None

        # Simulated Redis get
        return {"agent_id": agent_id, "status": "online", "last_seen": datetime.utcnow().isoformat()}

    async def set_agent_state(self, state: Dict[str, Any]) -> None:
        """Set agent state in Redis."""
        if not self._redis_client:
            return

        state["agent_id"] = self.config.agent_id
        state["updated_at"] = datetime.utcnow().isoformat()

        # Simulated Redis set
        logger.debug(f"Updated agent state in Redis")

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "subscribed_events": list(self._event_handlers.keys()),
            "pending_responses": len(self._pending_responses)
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "connected": self._connected,
            "timestamp": datetime.utcnow().isoformat()
        }

        if not self._connected:
            health["status"] = "unhealthy"
            health["reason"] = "Not connected to message bus"
            return health

        # Check AMQP connection
        if not self._amqp_connection or not self._amqp_connection.get("connected"):
            health["status"] = "unhealthy"
            health["reason"] = "AMQP connection lost"
            return health

        # Check for errors
        if self._stats["errors"] > 10:
            health["status"] = "degraded"
            health["error_count"] = self._stats["errors"]

        return health
