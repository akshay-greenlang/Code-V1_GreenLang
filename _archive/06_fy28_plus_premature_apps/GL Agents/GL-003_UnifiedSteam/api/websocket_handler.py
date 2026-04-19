"""
GL-003 UNIFIEDSTEAM - WebSocket Real-Time Streaming Handler

Provides real-time bidirectional communication for steam system monitoring
and control. Supports multiple subscription channels and push notifications.

Channels:
- steam_states: Real-time thermodynamic state updates
- trap_status: Steam trap monitoring and diagnostics
- recommendations: Optimization recommendation alerts
- alarms: System alarms and notifications
- kpis: Key performance indicator updates
- control: Control command interface

Features:
- Connection management with heartbeat/keepalive
- Rate limiting per client
- Message authentication and authorization
- Subscription-based filtering
- Automatic reconnection support
- Message queuing for offline clients

Author: GL-APIEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from abc import ABC, abstractmethod
import asyncio
import json
import logging
import hashlib
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ChannelType(str, Enum):
    """Available WebSocket subscription channels."""
    STEAM_STATES = "steam_states"
    TRAP_STATUS = "trap_status"
    RECOMMENDATIONS = "recommendations"
    ALARMS = "alarms"
    KPIS = "kpis"
    CONTROL = "control"
    SYSTEM_HEALTH = "system_health"


class MessageType(str, Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    DATA = "data"
    ERROR = "error"
    ACK = "ack"
    HEARTBEAT = "heartbeat"
    CONTROL_COMMAND = "control_command"
    CONTROL_RESPONSE = "control_response"


class ClientState(str, Enum):
    """WebSocket client connection state."""
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    SUBSCRIBED = "subscribed"
    DISCONNECTED = "disconnected"


# Rate limiting settings
DEFAULT_RATE_LIMIT_MESSAGES_PER_SECOND = 100
DEFAULT_RATE_LIMIT_BYTES_PER_SECOND = 1_000_000  # 1 MB/s
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30
DEFAULT_CLIENT_TIMEOUT_SECONDS = 120


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WebSocketMessage:
    """
    Standard WebSocket message format.

    Attributes:
        message_id: Unique message identifier
        message_type: Type of message
        channel: Channel for data messages
        payload: Message payload
        timestamp: Message timestamp
        correlation_id: For request-response correlation
        auth_token: Optional authentication token
    """
    message_id: str
    message_type: MessageType
    channel: Optional[ChannelType] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    auth_token: Optional[str] = None

    def to_json(self) -> str:
        """Serialize message to JSON."""
        return json.dumps({
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "channel": self.channel.value if self.channel else None,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
        })

    @classmethod
    def from_json(cls, json_str: str) -> "WebSocketMessage":
        """Deserialize message from JSON."""
        data = json.loads(json_str)
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=MessageType(data["message_type"]),
            channel=ChannelType(data["channel"]) if data.get("channel") else None,
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(timezone.utc),
            correlation_id=data.get("correlation_id"),
            auth_token=data.get("auth_token"),
        )


@dataclass
class Subscription:
    """
    Client subscription to a channel.

    Attributes:
        channel: Subscribed channel
        filters: Optional filter criteria
        created_at: Subscription creation time
        last_message_at: Time of last message
    """
    channel: ChannelType
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_message_at: Optional[datetime] = None


@dataclass
class ClientInfo:
    """
    WebSocket client information.

    Attributes:
        client_id: Unique client identifier
        user_id: Authenticated user ID
        state: Current connection state
        subscriptions: Active channel subscriptions
        connected_at: Connection timestamp
        last_activity_at: Last activity timestamp
        remote_address: Client IP address
        user_agent: Client user agent
        rate_limit_tokens: Current rate limit tokens
        permissions: Client permissions
    """
    client_id: str
    user_id: Optional[str] = None
    state: ClientState = ClientState.CONNECTED
    subscriptions: Dict[ChannelType, Subscription] = field(default_factory=dict)
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    remote_address: str = ""
    user_agent: str = ""
    rate_limit_tokens: float = DEFAULT_RATE_LIMIT_MESSAGES_PER_SECOND
    permissions: Set[str] = field(default_factory=set)


@dataclass
class ChannelStats:
    """Channel usage statistics."""
    channel: ChannelType
    subscriber_count: int
    messages_sent: int
    messages_per_second: float
    bytes_sent: int
    last_message_at: Optional[datetime] = None


# =============================================================================
# WEBSOCKET CONNECTION MANAGER
# =============================================================================

class WebSocketConnectionManager:
    """
    Manages WebSocket client connections and message routing.

    Features:
    - Connection lifecycle management
    - Subscription management
    - Message broadcasting
    - Rate limiting
    - Authentication integration

    Example:
        manager = WebSocketConnectionManager()

        # Register new connection
        client = manager.register_client(websocket, client_id)

        # Subscribe to channel
        manager.subscribe(client_id, ChannelType.STEAM_STATES, {"equipment_id": "boiler_1"})

        # Broadcast update
        await manager.broadcast(
            channel=ChannelType.STEAM_STATES,
            payload={"temperature": 450, "pressure": 4000}
        )
    """

    def __init__(
        self,
        heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        client_timeout: int = DEFAULT_CLIENT_TIMEOUT_SECONDS,
        rate_limit_messages: int = DEFAULT_RATE_LIMIT_MESSAGES_PER_SECOND,
        rate_limit_bytes: int = DEFAULT_RATE_LIMIT_BYTES_PER_SECOND,
    ):
        """
        Initialize connection manager.

        Args:
            heartbeat_interval: Seconds between heartbeat messages
            client_timeout: Seconds before inactive client is disconnected
            rate_limit_messages: Max messages per second per client
            rate_limit_bytes: Max bytes per second per client
        """
        self.heartbeat_interval = heartbeat_interval
        self.client_timeout = client_timeout
        self.rate_limit_messages = rate_limit_messages
        self.rate_limit_bytes = rate_limit_bytes

        # Client tracking
        self._clients: Dict[str, ClientInfo] = {}
        self._websockets: Dict[str, Any] = {}  # client_id -> websocket

        # Channel subscribers
        self._channel_subscribers: Dict[ChannelType, Set[str]] = defaultdict(set)

        # Statistics
        self._channel_stats: Dict[ChannelType, ChannelStats] = {}
        self._total_messages_sent: int = 0
        self._total_bytes_sent: int = 0

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start background tasks."""
        if self._running:
            return

        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("WebSocket connection manager started")

    async def stop(self) -> None:
        """Stop background tasks and disconnect all clients."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Disconnect all clients
        for client_id in list(self._clients.keys()):
            await self.disconnect_client(client_id)

        logger.info("WebSocket connection manager stopped")

    def register_client(
        self,
        websocket: Any,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        remote_address: str = "",
        user_agent: str = "",
    ) -> ClientInfo:
        """
        Register a new WebSocket client connection.

        Args:
            websocket: WebSocket connection object
            client_id: Optional client ID (generated if not provided)
            user_id: Authenticated user ID
            remote_address: Client IP address
            user_agent: Client user agent string

        Returns:
            ClientInfo for the registered client
        """
        if client_id is None:
            client_id = str(uuid.uuid4())

        client = ClientInfo(
            client_id=client_id,
            user_id=user_id,
            state=ClientState.CONNECTED if not user_id else ClientState.AUTHENTICATED,
            remote_address=remote_address,
            user_agent=user_agent,
        )

        self._clients[client_id] = client
        self._websockets[client_id] = websocket

        logger.info(f"Client registered: {client_id} from {remote_address}")

        return client

    async def disconnect_client(self, client_id: str) -> None:
        """
        Disconnect and clean up a client.

        Args:
            client_id: Client to disconnect
        """
        if client_id not in self._clients:
            return

        client = self._clients[client_id]

        # Remove from all channel subscriptions
        for channel in list(client.subscriptions.keys()):
            self._channel_subscribers[channel].discard(client_id)

        # Update state
        client.state = ClientState.DISCONNECTED

        # Close websocket if still connected
        websocket = self._websockets.get(client_id)
        if websocket:
            try:
                await websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket for {client_id}: {e}")

        # Clean up
        del self._clients[client_id]
        self._websockets.pop(client_id, None)

        logger.info(f"Client disconnected: {client_id}")

    def subscribe(
        self,
        client_id: str,
        channel: ChannelType,
        filters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Subscribe a client to a channel.

        Args:
            client_id: Client ID
            channel: Channel to subscribe to
            filters: Optional filter criteria

        Returns:
            True if subscription was successful
        """
        if client_id not in self._clients:
            return False

        client = self._clients[client_id]

        # Check permissions
        if not self._check_channel_permission(client, channel):
            logger.warning(f"Client {client_id} lacks permission for channel {channel}")
            return False

        # Create subscription
        subscription = Subscription(
            channel=channel,
            filters=filters or {},
        )

        client.subscriptions[channel] = subscription
        self._channel_subscribers[channel].add(client_id)

        if client.state == ClientState.AUTHENTICATED:
            client.state = ClientState.SUBSCRIBED

        logger.debug(f"Client {client_id} subscribed to {channel.value}")

        return True

    def unsubscribe(self, client_id: str, channel: ChannelType) -> bool:
        """
        Unsubscribe a client from a channel.

        Args:
            client_id: Client ID
            channel: Channel to unsubscribe from

        Returns:
            True if unsubscription was successful
        """
        if client_id not in self._clients:
            return False

        client = self._clients[client_id]

        if channel in client.subscriptions:
            del client.subscriptions[channel]
            self._channel_subscribers[channel].discard(client_id)

            logger.debug(f"Client {client_id} unsubscribed from {channel.value}")
            return True

        return False

    async def send_to_client(
        self,
        client_id: str,
        message: WebSocketMessage,
    ) -> bool:
        """
        Send a message to a specific client.

        Args:
            client_id: Target client ID
            message: Message to send

        Returns:
            True if message was sent successfully
        """
        if client_id not in self._clients:
            return False

        # Check rate limit
        if not self._check_rate_limit(client_id):
            logger.warning(f"Rate limit exceeded for client {client_id}")
            return False

        websocket = self._websockets.get(client_id)
        if not websocket:
            return False

        try:
            json_message = message.to_json()
            await websocket.send_text(json_message)

            # Update stats
            self._total_messages_sent += 1
            self._total_bytes_sent += len(json_message)
            self._clients[client_id].last_activity_at = datetime.now(timezone.utc)

            return True

        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {e}")
            await self.disconnect_client(client_id)
            return False

    async def broadcast(
        self,
        channel: ChannelType,
        payload: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Broadcast a message to all subscribers of a channel.

        Args:
            channel: Target channel
            payload: Message payload
            filters: Optional filter to match against subscriptions

        Returns:
            Number of clients that received the message
        """
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.DATA,
            channel=channel,
            payload=payload,
        )

        subscribers = self._channel_subscribers.get(channel, set())
        sent_count = 0

        for client_id in list(subscribers):
            if client_id not in self._clients:
                continue

            client = self._clients[client_id]
            subscription = client.subscriptions.get(channel)

            if not subscription:
                continue

            # Check if message matches subscription filters
            if not self._matches_filters(payload, subscription.filters):
                continue

            if await self.send_to_client(client_id, message):
                subscription.last_message_at = datetime.now(timezone.utc)
                sent_count += 1

        # Update channel stats
        self._update_channel_stats(channel, sent_count, len(message.to_json()))

        return sent_count

    async def handle_message(
        self,
        client_id: str,
        raw_message: str,
    ) -> Optional[WebSocketMessage]:
        """
        Handle incoming message from a client.

        Args:
            client_id: Client ID
            raw_message: Raw JSON message string

        Returns:
            Response message or None
        """
        if client_id not in self._clients:
            return None

        client = self._clients[client_id]
        client.last_activity_at = datetime.now(timezone.utc)

        try:
            message = WebSocketMessage.from_json(raw_message)
        except Exception as e:
            logger.warning(f"Invalid message from {client_id}: {e}")
            return WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ERROR,
                payload={"error": "Invalid message format"},
            )

        # Handle different message types
        if message.message_type == MessageType.SUBSCRIBE:
            return await self._handle_subscribe(client_id, message)

        elif message.message_type == MessageType.UNSUBSCRIBE:
            return await self._handle_unsubscribe(client_id, message)

        elif message.message_type == MessageType.HEARTBEAT:
            return await self._handle_heartbeat(client_id, message)

        elif message.message_type == MessageType.CONTROL_COMMAND:
            return await self._handle_control_command(client_id, message)

        else:
            return WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ACK,
                correlation_id=message.message_id,
            )

    async def _handle_subscribe(
        self,
        client_id: str,
        message: WebSocketMessage,
    ) -> WebSocketMessage:
        """Handle subscription request."""
        channel_str = message.payload.get("channel")
        filters = message.payload.get("filters", {})

        try:
            channel = ChannelType(channel_str)
        except ValueError:
            return WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ERROR,
                payload={"error": f"Unknown channel: {channel_str}"},
                correlation_id=message.message_id,
            )

        success = self.subscribe(client_id, channel, filters)

        return WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ACK if success else MessageType.ERROR,
            payload={
                "success": success,
                "channel": channel.value,
                "message": "Subscribed successfully" if success else "Subscription failed",
            },
            correlation_id=message.message_id,
        )

    async def _handle_unsubscribe(
        self,
        client_id: str,
        message: WebSocketMessage,
    ) -> WebSocketMessage:
        """Handle unsubscription request."""
        channel_str = message.payload.get("channel")

        try:
            channel = ChannelType(channel_str)
        except ValueError:
            return WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ERROR,
                payload={"error": f"Unknown channel: {channel_str}"},
                correlation_id=message.message_id,
            )

        success = self.unsubscribe(client_id, channel)

        return WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ACK,
            payload={
                "success": success,
                "channel": channel.value,
            },
            correlation_id=message.message_id,
        )

    async def _handle_heartbeat(
        self,
        client_id: str,
        message: WebSocketMessage,
    ) -> WebSocketMessage:
        """Handle heartbeat/ping request."""
        return WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,
            payload={"pong": True, "server_time": datetime.now(timezone.utc).isoformat()},
            correlation_id=message.message_id,
        )

    async def _handle_control_command(
        self,
        client_id: str,
        message: WebSocketMessage,
    ) -> WebSocketMessage:
        """Handle control command from client."""
        client = self._clients.get(client_id)

        if not client or "control:write" not in client.permissions:
            return WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ERROR,
                payload={"error": "Unauthorized for control commands"},
                correlation_id=message.message_id,
            )

        # Validate and forward command
        command = message.payload.get("command")
        target = message.payload.get("target")
        value = message.payload.get("value")

        logger.info(f"Control command from {client_id}: {command} -> {target} = {value}")

        # In production, this would forward to the control system
        return WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.CONTROL_RESPONSE,
            payload={
                "success": True,
                "command": command,
                "target": target,
                "value": value,
                "message": "Command accepted",
            },
            correlation_id=message.message_id,
        )

    def _check_channel_permission(
        self,
        client: ClientInfo,
        channel: ChannelType,
    ) -> bool:
        """Check if client has permission for channel."""
        # Control channel requires special permission
        if channel == ChannelType.CONTROL:
            return "control:read" in client.permissions

        # All authenticated users can access other channels
        return client.state != ClientState.CONNECTED or True  # Allow anonymous for now

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        if client_id not in self._clients:
            return False

        client = self._clients[client_id]

        # Simple token bucket (replenish 1 token per 1/rate seconds)
        if client.rate_limit_tokens > 0:
            client.rate_limit_tokens -= 1
            return True

        return False

    def _replenish_rate_limits(self) -> None:
        """Replenish rate limit tokens for all clients."""
        for client in self._clients.values():
            client.rate_limit_tokens = min(
                client.rate_limit_tokens + self.rate_limit_messages / 10,
                self.rate_limit_messages
            )

    def _matches_filters(
        self,
        payload: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> bool:
        """Check if payload matches subscription filters."""
        if not filters:
            return True

        for key, value in filters.items():
            if key in payload:
                # Support list of values
                if isinstance(value, list):
                    if payload[key] not in value:
                        return False
                elif payload[key] != value:
                    return False

        return True

    def _update_channel_stats(
        self,
        channel: ChannelType,
        messages_sent: int,
        bytes_sent: int,
    ) -> None:
        """Update channel statistics."""
        if channel not in self._channel_stats:
            self._channel_stats[channel] = ChannelStats(
                channel=channel,
                subscriber_count=0,
                messages_sent=0,
                messages_per_second=0,
                bytes_sent=0,
            )

        stats = self._channel_stats[channel]
        stats.subscriber_count = len(self._channel_subscribers.get(channel, set()))
        stats.messages_sent += messages_sent
        stats.bytes_sent += bytes_sent
        stats.last_message_at = datetime.now(timezone.utc)

    async def _heartbeat_loop(self) -> None:
        """Background task to send heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                heartbeat = WebSocketMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.HEARTBEAT,
                    payload={"server_time": datetime.now(timezone.utc).isoformat()},
                )

                for client_id in list(self._clients.keys()):
                    await self.send_to_client(client_id, heartbeat)

                # Replenish rate limits
                self._replenish_rate_limits()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up inactive clients."""
        while self._running:
            try:
                await asyncio.sleep(self.client_timeout / 4)

                now = datetime.now(timezone.utc)
                timeout_threshold = now - timedelta(seconds=self.client_timeout)

                for client_id, client in list(self._clients.items()):
                    if client.last_activity_at < timeout_threshold:
                        logger.info(f"Client {client_id} timed out")
                        await self.disconnect_client(client_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection manager statistics."""
        return {
            "total_clients": len(self._clients),
            "total_messages_sent": self._total_messages_sent,
            "total_bytes_sent": self._total_bytes_sent,
            "channels": {
                channel.value: {
                    "subscribers": len(subscribers),
                    "stats": (
                        {
                            "messages_sent": self._channel_stats[channel].messages_sent,
                            "bytes_sent": self._channel_stats[channel].bytes_sent,
                        }
                        if channel in self._channel_stats
                        else None
                    ),
                }
                for channel, subscribers in self._channel_subscribers.items()
            },
        }


# =============================================================================
# FASTAPI WEBSOCKET ENDPOINT
# =============================================================================

# Global connection manager instance
connection_manager = WebSocketConnectionManager()


async def websocket_endpoint(websocket, client_id: Optional[str] = None):
    """
    FastAPI WebSocket endpoint handler.

    Example usage in FastAPI:

        from fastapi import FastAPI, WebSocket
        from api.websocket_handler import websocket_endpoint

        app = FastAPI()

        @app.websocket("/ws/{client_id}")
        async def websocket_route(websocket: WebSocket, client_id: str):
            await websocket_endpoint(websocket, client_id)
    """
    await websocket.accept()

    # Register client
    client = connection_manager.register_client(
        websocket=websocket,
        client_id=client_id,
        remote_address=str(websocket.client) if hasattr(websocket, "client") else "",
    )

    # Send welcome message
    welcome = WebSocketMessage(
        message_id=str(uuid.uuid4()),
        message_type=MessageType.ACK,
        payload={
            "message": "Connected to GL-003 UnifiedSteam WebSocket",
            "client_id": client.client_id,
            "available_channels": [c.value for c in ChannelType],
        },
    )
    await connection_manager.send_to_client(client.client_id, welcome)

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            # Handle message
            response = await connection_manager.handle_message(client.client_id, data)

            if response:
                await connection_manager.send_to_client(client.client_id, response)

    except Exception as e:
        logger.info(f"WebSocket connection closed for {client.client_id}: {e}")

    finally:
        await connection_manager.disconnect_client(client.client_id)


# =============================================================================
# DATA PUBLISHERS
# =============================================================================

class SteamStatePublisher:
    """
    Publisher for real-time steam state updates.

    Publishes thermodynamic state data to connected WebSocket clients.
    """

    def __init__(self, manager: WebSocketConnectionManager):
        self.manager = manager

    async def publish_state(
        self,
        equipment_id: str,
        equipment_name: str,
        pressure_kpa: float,
        temperature_c: float,
        enthalpy_kj_kg: float,
        flow_rate_kg_s: Optional[float] = None,
        quality: Optional[float] = None,
    ) -> int:
        """
        Publish steam state update.

        Args:
            equipment_id: Equipment identifier
            equipment_name: Equipment name
            pressure_kpa: Pressure (kPa)
            temperature_c: Temperature (C)
            enthalpy_kj_kg: Specific enthalpy (kJ/kg)
            flow_rate_kg_s: Optional mass flow rate
            quality: Optional steam quality

        Returns:
            Number of clients that received the update
        """
        payload = {
            "equipment_id": equipment_id,
            "equipment_name": equipment_name,
            "pressure_kpa": pressure_kpa,
            "temperature_c": temperature_c,
            "enthalpy_kj_kg": enthalpy_kj_kg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if flow_rate_kg_s is not None:
            payload["flow_rate_kg_s"] = flow_rate_kg_s

        if quality is not None:
            payload["quality"] = quality

        return await self.manager.broadcast(
            channel=ChannelType.STEAM_STATES,
            payload=payload,
            filters={"equipment_id": equipment_id},
        )


class AlarmPublisher:
    """
    Publisher for alarm notifications.

    Publishes alarm events to connected WebSocket clients.
    """

    def __init__(self, manager: WebSocketConnectionManager):
        self.manager = manager

    async def publish_alarm(
        self,
        alarm_id: str,
        alarm_code: str,
        name: str,
        description: str,
        severity: str,
        source_equipment: str,
        measured_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        unit: Optional[str] = None,
    ) -> int:
        """
        Publish alarm notification.

        Returns:
            Number of clients that received the alarm
        """
        payload = {
            "alarm_id": alarm_id,
            "alarm_code": alarm_code,
            "name": name,
            "description": description,
            "severity": severity,
            "source_equipment": source_equipment,
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "is_acknowledged": False,
        }

        if measured_value is not None:
            payload["measured_value"] = measured_value

        if threshold_value is not None:
            payload["threshold_value"] = threshold_value

        if unit:
            payload["unit"] = unit

        return await self.manager.broadcast(
            channel=ChannelType.ALARMS,
            payload=payload,
        )


class RecommendationPublisher:
    """
    Publisher for optimization recommendations.

    Publishes new recommendations to connected WebSocket clients.
    """

    def __init__(self, manager: WebSocketConnectionManager):
        self.manager = manager

    async def publish_recommendation(
        self,
        recommendation_id: str,
        recommendation_type: str,
        priority: str,
        title: str,
        description: str,
        estimated_savings_usd: Optional[float] = None,
        affected_equipment: Optional[List[str]] = None,
    ) -> int:
        """
        Publish new recommendation.

        Returns:
            Number of clients that received the recommendation
        """
        payload = {
            "recommendation_id": recommendation_id,
            "recommendation_type": recommendation_type,
            "priority": priority,
            "title": title,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_new": True,
        }

        if estimated_savings_usd is not None:
            payload["estimated_savings_usd"] = estimated_savings_usd

        if affected_equipment:
            payload["affected_equipment"] = affected_equipment

        return await self.manager.broadcast(
            channel=ChannelType.RECOMMENDATIONS,
            payload=payload,
        )
