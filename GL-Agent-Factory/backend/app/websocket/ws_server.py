"""
WebSocket Server for GreenLang

Real-time communication infrastructure providing:
- Metrics streaming
- Agent execution progress updates
- Live calculation results
- Connection management with heartbeat/keepalive
- Room-based subscriptions (per tenant, per agent)

Example:
    >>> from app.websocket import create_websocket_server
    >>> ws_server = create_websocket_server(redis_client)
    >>> app.include_router(ws_server.router)
    >>>
    >>> # In your agent execution code:
    >>> await ws_server.broadcast_to_room(
    ...     room=f"execution:{execution_id}",
    ...     message={"type": "progress", "percent": 50}
    ... )
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class MessageType(str, Enum):
    """WebSocket message types."""

    # Connection management
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    HEARTBEAT = "heartbeat"
    PONG = "pong"

    # Subscriptions
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"

    # Agent execution
    EXECUTION_STARTED = "execution.started"
    EXECUTION_PROGRESS = "execution.progress"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    EXECUTION_LOG = "execution.log"

    # Metrics
    METRICS_UPDATE = "metrics.update"
    METRICS_BATCH = "metrics.batch"

    # Calculations
    CALCULATION_RESULT = "calculation.result"
    CALCULATION_INTERMEDIATE = "calculation.intermediate"

    # Alerts
    ALERT_TRIGGERED = "alert.triggered"
    ALERT_RESOLVED = "alert.resolved"

    # Errors
    ERROR = "error"

    # Custom
    CUSTOM = "custom"


class ConnectionState(str, Enum):
    """WebSocket connection states."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"


# =============================================================================
# Models
# =============================================================================


class WebSocketMessage(BaseModel):
    """Standard WebSocket message format."""

    type: MessageType
    data: Optional[Dict[str, Any]] = None
    room: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    class Config:
        use_enum_values = True


class ExecutionProgressMessage(BaseModel):
    """Execution progress update message."""

    execution_id: str
    agent_id: str
    status: str
    progress_percent: int = Field(ge=0, le=100)
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    current_step_index: Optional[int] = None
    elapsed_seconds: float = 0
    estimated_remaining_seconds: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None


class MetricsUpdateMessage(BaseModel):
    """Real-time metrics update message."""

    tenant_id: str
    agent_id: Optional[str] = None
    metrics: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CalculationResultMessage(BaseModel):
    """Live calculation result message."""

    execution_id: str
    result_type: str  # "final", "intermediate", "partial"
    value: Any
    unit: Optional[str] = None
    confidence: Optional[float] = None
    methodology: Optional[str] = None
    source_hash: Optional[str] = None


# =============================================================================
# Connection and Room Management
# =============================================================================


@dataclass
class Connection:
    """Represents a single WebSocket connection."""

    connection_id: str
    websocket: WebSocket
    tenant_id: str
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    rooms: Set[str] = field(default_factory=set)
    state: ConnectionState = ConnectionState.CONNECTING
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_alive(self, timeout_seconds: int = 60) -> bool:
        """Check if connection is still alive based on last heartbeat."""
        elapsed = (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds()
        return elapsed < timeout_seconds


@dataclass
class Room:
    """Represents a subscription room."""

    room_id: str
    room_type: str  # "tenant", "agent", "execution", "metrics"
    connections: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def connection_count(self) -> int:
        return len(self.connections)


class ConnectionManager:
    """
    Manages WebSocket connections and rooms.

    Features:
    - Connection lifecycle management
    - Room-based subscriptions
    - Heartbeat monitoring
    - Message broadcasting
    - Connection cleanup
    """

    def __init__(
        self,
        heartbeat_interval: int = 30,
        heartbeat_timeout: int = 60,
        max_connections_per_tenant: int = 100,
        redis_client: Optional[Any] = None,
    ):
        """
        Initialize the connection manager.

        Args:
            heartbeat_interval: Seconds between heartbeat pings
            heartbeat_timeout: Seconds before connection considered dead
            max_connections_per_tenant: Maximum connections per tenant
            redis_client: Optional Redis client for distributed messaging
        """
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.max_connections_per_tenant = max_connections_per_tenant
        self.redis = redis_client

        # Connection storage
        self._connections: Dict[str, Connection] = {}
        self._rooms: Dict[str, Room] = {}
        self._tenant_connections: Dict[str, Set[str]] = {}

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._pubsub_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background tasks."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        if self.redis:
            self._pubsub_task = asyncio.create_task(self._redis_pubsub_loop())

        logger.info("WebSocket connection manager started")

    async def stop(self) -> None:
        """Stop background tasks and close all connections."""
        # Cancel background tasks
        for task in [self._heartbeat_task, self._cleanup_task, self._pubsub_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close all connections
        for connection in list(self._connections.values()):
            await self.disconnect(connection.connection_id)

        logger.info("WebSocket connection manager stopped")

    async def connect(
        self,
        websocket: WebSocket,
        tenant_id: str,
        user_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
    ) -> Connection:
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            tenant_id: Tenant ID for the connection
            user_id: Optional user ID
            api_key_id: Optional API key ID

        Returns:
            The registered Connection object

        Raises:
            HTTPException: If connection limit exceeded
        """
        # Check connection limit
        tenant_connections = self._tenant_connections.get(tenant_id, set())
        if len(tenant_connections) >= self.max_connections_per_tenant:
            await websocket.close(code=1008, reason="Connection limit exceeded")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="WebSocket connection limit exceeded for tenant",
            )

        # Accept the connection
        await websocket.accept()

        # Create connection record
        connection_id = str(uuid.uuid4())
        connection = Connection(
            connection_id=connection_id,
            websocket=websocket,
            tenant_id=tenant_id,
            user_id=user_id,
            api_key_id=api_key_id,
            state=ConnectionState.CONNECTED,
        )

        # Register connection
        self._connections[connection_id] = connection

        # Track by tenant
        if tenant_id not in self._tenant_connections:
            self._tenant_connections[tenant_id] = set()
        self._tenant_connections[tenant_id].add(connection_id)

        # Auto-subscribe to tenant room
        await self.subscribe_to_room(connection_id, f"tenant:{tenant_id}")

        # Send connection confirmation
        await self.send_to_connection(
            connection_id,
            WebSocketMessage(
                type=MessageType.CONNECT,
                data={
                    "connection_id": connection_id,
                    "tenant_id": tenant_id,
                    "heartbeat_interval": self.heartbeat_interval,
                },
            ),
        )

        logger.info(f"WebSocket connected: {connection_id} (tenant: {tenant_id})")

        return connection

    async def disconnect(self, connection_id: str) -> None:
        """
        Disconnect and cleanup a WebSocket connection.

        Args:
            connection_id: The connection ID to disconnect
        """
        connection = self._connections.get(connection_id)
        if not connection:
            return

        # Remove from all rooms
        for room_id in list(connection.rooms):
            await self.unsubscribe_from_room(connection_id, room_id)

        # Remove from tenant tracking
        if connection.tenant_id in self._tenant_connections:
            self._tenant_connections[connection.tenant_id].discard(connection_id)
            if not self._tenant_connections[connection.tenant_id]:
                del self._tenant_connections[connection.tenant_id]

        # Close WebSocket if still open
        if connection.websocket.client_state == WebSocketState.CONNECTED:
            try:
                await connection.websocket.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")

        # Remove connection
        del self._connections[connection_id]

        logger.info(f"WebSocket disconnected: {connection_id}")

    async def subscribe_to_room(self, connection_id: str, room_id: str) -> None:
        """
        Subscribe a connection to a room.

        Args:
            connection_id: The connection ID
            room_id: The room ID to subscribe to
        """
        connection = self._connections.get(connection_id)
        if not connection:
            return

        # Create room if doesn't exist
        if room_id not in self._rooms:
            room_type = room_id.split(":")[0] if ":" in room_id else "custom"
            self._rooms[room_id] = Room(room_id=room_id, room_type=room_type)

        # Add connection to room
        self._rooms[room_id].connections.add(connection_id)
        connection.rooms.add(room_id)

        # Send confirmation
        await self.send_to_connection(
            connection_id,
            WebSocketMessage(
                type=MessageType.SUBSCRIBED,
                data={"room": room_id},
            ),
        )

        logger.debug(f"Connection {connection_id} subscribed to room {room_id}")

    async def unsubscribe_from_room(self, connection_id: str, room_id: str) -> None:
        """
        Unsubscribe a connection from a room.

        Args:
            connection_id: The connection ID
            room_id: The room ID to unsubscribe from
        """
        connection = self._connections.get(connection_id)
        if connection:
            connection.rooms.discard(room_id)

        room = self._rooms.get(room_id)
        if room:
            room.connections.discard(connection_id)

            # Clean up empty rooms (except tenant rooms)
            if not room.connections and not room_id.startswith("tenant:"):
                del self._rooms[room_id]

        logger.debug(f"Connection {connection_id} unsubscribed from room {room_id}")

    async def send_to_connection(
        self,
        connection_id: str,
        message: WebSocketMessage,
    ) -> bool:
        """
        Send a message to a specific connection.

        Args:
            connection_id: The connection ID
            message: The message to send

        Returns:
            True if message was sent, False otherwise
        """
        connection = self._connections.get(connection_id)
        if not connection:
            return False

        try:
            await connection.websocket.send_json(message.dict())
            connection.message_count += 1
            return True
        except Exception as e:
            logger.warning(f"Failed to send message to {connection_id}: {e}")
            # Mark for cleanup
            await self.disconnect(connection_id)
            return False

    async def broadcast_to_room(
        self,
        room_id: str,
        message: WebSocketMessage,
        exclude: Optional[Set[str]] = None,
    ) -> int:
        """
        Broadcast a message to all connections in a room.

        Args:
            room_id: The room ID
            message: The message to broadcast
            exclude: Optional set of connection IDs to exclude

        Returns:
            Number of connections that received the message
        """
        room = self._rooms.get(room_id)
        if not room:
            return 0

        exclude = exclude or set()
        sent_count = 0

        for connection_id in list(room.connections):
            if connection_id in exclude:
                continue

            if await self.send_to_connection(connection_id, message):
                sent_count += 1

        # Also publish to Redis for distributed messaging
        if self.redis:
            await self._publish_to_redis(room_id, message)

        return sent_count

    async def broadcast_to_tenant(
        self,
        tenant_id: str,
        message: WebSocketMessage,
    ) -> int:
        """
        Broadcast a message to all connections for a tenant.

        Args:
            tenant_id: The tenant ID
            message: The message to broadcast

        Returns:
            Number of connections that received the message
        """
        return await self.broadcast_to_room(f"tenant:{tenant_id}", message)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat pings to all connections."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                heartbeat_message = WebSocketMessage(
                    type=MessageType.HEARTBEAT,
                    data={"timestamp": datetime.now(timezone.utc).isoformat()},
                )

                for connection_id in list(self._connections.keys()):
                    await self.send_to_connection(connection_id, heartbeat_message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up dead connections."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_timeout)

                dead_connections = [
                    conn_id
                    for conn_id, conn in self._connections.items()
                    if not conn.is_alive(self.heartbeat_timeout)
                ]

                for connection_id in dead_connections:
                    logger.info(f"Cleaning up dead connection: {connection_id}")
                    await self.disconnect(connection_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _redis_pubsub_loop(self) -> None:
        """Listen for messages from Redis pub/sub for distributed messaging."""
        if not self.redis:
            return

        try:
            pubsub = self.redis.pubsub()
            await pubsub.psubscribe("ws:room:*")

            async for message in pubsub.listen():
                if message["type"] == "pmessage":
                    room_id = message["channel"].decode().replace("ws:room:", "")
                    data = json.loads(message["data"])
                    ws_message = WebSocketMessage(**data)

                    # Broadcast locally
                    room = self._rooms.get(room_id)
                    if room:
                        for connection_id in list(room.connections):
                            await self.send_to_connection(connection_id, ws_message)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Redis pubsub loop error: {e}")

    async def _publish_to_redis(
        self,
        room_id: str,
        message: WebSocketMessage,
    ) -> None:
        """Publish a message to Redis for distributed broadcasting."""
        if not self.redis:
            return

        try:
            channel = f"ws:room:{room_id}"
            await self.redis.publish(channel, json.dumps(message.dict()))
        except Exception as e:
            logger.error(f"Redis publish error: {e}")

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self._connections)

    def get_room_count(self) -> int:
        """Get total number of active rooms."""
        return len(self._rooms)

    def get_tenant_connection_count(self, tenant_id: str) -> int:
        """Get number of connections for a specific tenant."""
        return len(self._tenant_connections.get(tenant_id, set()))

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get detailed connection statistics."""
        return {
            "total_connections": len(self._connections),
            "total_rooms": len(self._rooms),
            "connections_by_tenant": {
                tenant_id: len(conn_ids)
                for tenant_id, conn_ids in self._tenant_connections.items()
            },
            "rooms": {
                room_id: room.connection_count
                for room_id, room in self._rooms.items()
            },
        }


# =============================================================================
# WebSocket Server
# =============================================================================


class WebSocketServer:
    """
    WebSocket server for GreenLang real-time communication.

    Features:
    - Real-time metrics streaming
    - Agent execution progress updates
    - Live calculation results
    - Connection management with heartbeat
    - Room-based subscriptions
    - Redis pub/sub for distributed messaging

    Example:
        >>> ws_server = WebSocketServer(redis_client)
        >>> await ws_server.start()
        >>> app.include_router(ws_server.router)
        >>>
        >>> # Send execution progress
        >>> await ws_server.send_execution_progress(
        ...     execution_id="exec-123",
        ...     agent_id="carbon/calculator",
        ...     progress_percent=50,
        ...     status="running",
        ... )
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        heartbeat_interval: int = 30,
        heartbeat_timeout: int = 60,
        max_connections_per_tenant: int = 100,
    ):
        """
        Initialize the WebSocket server.

        Args:
            redis_client: Optional Redis client for distributed messaging
            heartbeat_interval: Seconds between heartbeat pings
            heartbeat_timeout: Seconds before connection considered dead
            max_connections_per_tenant: Maximum connections per tenant
        """
        self.connection_manager = ConnectionManager(
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            max_connections_per_tenant=max_connections_per_tenant,
            redis_client=redis_client,
        )

        self.router = APIRouter(prefix="/ws", tags=["WebSocket"])
        self._setup_routes()

        # Message handlers
        self._handlers: Dict[MessageType, Callable] = {}
        self._register_default_handlers()

    def _setup_routes(self) -> None:
        """Set up WebSocket routes."""

        @self.router.websocket("/connect")
        async def websocket_endpoint(
            websocket: WebSocket,
            token: Optional[str] = Query(None, description="JWT or API key"),
            tenant_id: Optional[str] = Query(None, description="Tenant ID"),
        ):
            """
            Main WebSocket connection endpoint.

            Connect with: ws://host/ws/connect?token=<api_key>&tenant_id=<tenant>

            Message format:
            ```json
            {
                "type": "subscribe",
                "data": {"room": "execution:exec-123"}
            }
            ```
            """
            # Authenticate
            auth_result = await self._authenticate(token, tenant_id)
            if not auth_result:
                await websocket.close(code=1008, reason="Authentication failed")
                return

            tenant_id, user_id = auth_result

            # Connect
            try:
                connection = await self.connection_manager.connect(
                    websocket=websocket,
                    tenant_id=tenant_id,
                    user_id=user_id,
                )

                # Handle messages
                await self._handle_connection(connection)

            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
            finally:
                if connection:
                    await self.connection_manager.disconnect(connection.connection_id)

        @self.router.get("/stats")
        async def get_websocket_stats() -> Dict[str, Any]:
            """Get WebSocket server statistics."""
            return self.connection_manager.get_connection_stats()

    async def _authenticate(
        self,
        token: Optional[str],
        tenant_id: Optional[str],
    ) -> Optional[tuple[str, Optional[str]]]:
        """
        Authenticate WebSocket connection.

        Returns:
            Tuple of (tenant_id, user_id) if authenticated, None otherwise
        """
        # TODO: Implement proper JWT/API key validation
        # For development, accept any token with a tenant_id

        if tenant_id:
            return tenant_id, None

        if token and token.startswith("gl_"):
            # Extract tenant from token (placeholder)
            return "default_tenant", None

        # Development mode: allow unauthenticated connections
        return "dev_tenant", None

    async def _handle_connection(self, connection: Connection) -> None:
        """Handle incoming messages from a connection."""
        while True:
            try:
                # Receive message
                data = await connection.websocket.receive_json()

                # Parse message
                try:
                    message = WebSocketMessage(**data)
                except Exception as e:
                    await self.connection_manager.send_to_connection(
                        connection.connection_id,
                        WebSocketMessage(
                            type=MessageType.ERROR,
                            data={"error": f"Invalid message format: {e}"},
                        ),
                    )
                    continue

                # Update heartbeat
                connection.last_heartbeat = datetime.now(timezone.utc)

                # Handle message
                await self._handle_message(connection, message)

            except WebSocketDisconnect:
                raise
            except json.JSONDecodeError:
                await self.connection_manager.send_to_connection(
                    connection.connection_id,
                    WebSocketMessage(
                        type=MessageType.ERROR,
                        data={"error": "Invalid JSON"},
                    ),
                )
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                await self.connection_manager.send_to_connection(
                    connection.connection_id,
                    WebSocketMessage(
                        type=MessageType.ERROR,
                        data={"error": str(e)},
                    ),
                )

    async def _handle_message(
        self,
        connection: Connection,
        message: WebSocketMessage,
    ) -> None:
        """Process an incoming message."""
        handler = self._handlers.get(message.type)

        if handler:
            await handler(connection, message)
        else:
            logger.warning(f"No handler for message type: {message.type}")

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""

        @self.on_message(MessageType.HEARTBEAT)
        async def handle_heartbeat(connection: Connection, message: WebSocketMessage):
            # Respond with pong
            await self.connection_manager.send_to_connection(
                connection.connection_id,
                WebSocketMessage(
                    type=MessageType.PONG,
                    data={"timestamp": datetime.now(timezone.utc).isoformat()},
                ),
            )

        @self.on_message(MessageType.SUBSCRIBE)
        async def handle_subscribe(connection: Connection, message: WebSocketMessage):
            room = message.data.get("room") if message.data else None
            if room:
                # Validate room access
                if await self._can_access_room(connection, room):
                    await self.connection_manager.subscribe_to_room(
                        connection.connection_id, room
                    )
                else:
                    await self.connection_manager.send_to_connection(
                        connection.connection_id,
                        WebSocketMessage(
                            type=MessageType.ERROR,
                            data={"error": f"Access denied to room: {room}"},
                        ),
                    )

        @self.on_message(MessageType.UNSUBSCRIBE)
        async def handle_unsubscribe(connection: Connection, message: WebSocketMessage):
            room = message.data.get("room") if message.data else None
            if room:
                await self.connection_manager.unsubscribe_from_room(
                    connection.connection_id, room
                )

    async def _can_access_room(self, connection: Connection, room: str) -> bool:
        """Check if a connection can access a room."""
        # Tenant rooms - must match connection's tenant
        if room.startswith("tenant:"):
            room_tenant = room.split(":", 1)[1]
            return room_tenant == connection.tenant_id

        # Execution rooms - must belong to connection's tenant
        if room.startswith("execution:"):
            # TODO: Validate execution belongs to tenant
            return True

        # Agent rooms - must have access to agent
        if room.startswith("agent:"):
            # TODO: Validate agent access
            return True

        # Metrics rooms
        if room.startswith("metrics:"):
            return True

        return False

    def on_message(
        self,
        message_type: MessageType,
    ) -> Callable:
        """
        Decorator to register a message handler.

        Example:
            >>> @ws_server.on_message(MessageType.CUSTOM)
            ... async def handle_custom(connection, message):
            ...     print(f"Custom message: {message}")
        """
        def decorator(
            func: Callable[[Connection, WebSocketMessage], Coroutine[Any, Any, None]]
        ) -> Callable:
            self._handlers[message_type] = func
            return func
        return decorator

    async def start(self) -> None:
        """Start the WebSocket server."""
        await self.connection_manager.start()
        logger.info("WebSocket server started")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        await self.connection_manager.stop()
        logger.info("WebSocket server stopped")

    # =========================================================================
    # Convenience methods for sending specific message types
    # =========================================================================

    async def send_execution_progress(
        self,
        execution_id: str,
        agent_id: str,
        progress_percent: int,
        status: str,
        current_step: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Send execution progress update.

        Args:
            execution_id: Execution ID
            agent_id: Agent ID
            progress_percent: Progress percentage (0-100)
            status: Current status
            current_step: Current step description
            metrics: Optional metrics data

        Returns:
            Number of connections notified
        """
        message = WebSocketMessage(
            type=MessageType.EXECUTION_PROGRESS,
            data={
                "execution_id": execution_id,
                "agent_id": agent_id,
                "progress_percent": progress_percent,
                "status": status,
                "current_step": current_step,
                "metrics": metrics,
            },
            room=f"execution:{execution_id}",
        )

        return await self.connection_manager.broadcast_to_room(
            f"execution:{execution_id}",
            message,
        )

    async def send_execution_completed(
        self,
        execution_id: str,
        agent_id: str,
        result: Dict[str, Any],
        duration_seconds: float,
    ) -> int:
        """
        Send execution completed notification.

        Args:
            execution_id: Execution ID
            agent_id: Agent ID
            result: Execution result
            duration_seconds: Total execution duration

        Returns:
            Number of connections notified
        """
        message = WebSocketMessage(
            type=MessageType.EXECUTION_COMPLETED,
            data={
                "execution_id": execution_id,
                "agent_id": agent_id,
                "result": result,
                "duration_seconds": duration_seconds,
            },
            room=f"execution:{execution_id}",
        )

        return await self.connection_manager.broadcast_to_room(
            f"execution:{execution_id}",
            message,
        )

    async def send_execution_failed(
        self,
        execution_id: str,
        agent_id: str,
        error: str,
        error_code: str,
    ) -> int:
        """
        Send execution failed notification.

        Args:
            execution_id: Execution ID
            agent_id: Agent ID
            error: Error message
            error_code: Error code

        Returns:
            Number of connections notified
        """
        message = WebSocketMessage(
            type=MessageType.EXECUTION_FAILED,
            data={
                "execution_id": execution_id,
                "agent_id": agent_id,
                "error": error,
                "error_code": error_code,
            },
            room=f"execution:{execution_id}",
        )

        return await self.connection_manager.broadcast_to_room(
            f"execution:{execution_id}",
            message,
        )

    async def send_metrics_update(
        self,
        tenant_id: str,
        metrics: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> int:
        """
        Send real-time metrics update.

        Args:
            tenant_id: Tenant ID
            metrics: Metrics data
            agent_id: Optional agent ID for agent-specific metrics

        Returns:
            Number of connections notified
        """
        room = f"agent:{agent_id}" if agent_id else f"tenant:{tenant_id}"

        message = WebSocketMessage(
            type=MessageType.METRICS_UPDATE,
            data={
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "metrics": metrics,
            },
            room=room,
        )

        return await self.connection_manager.broadcast_to_room(room, message)

    async def send_calculation_result(
        self,
        execution_id: str,
        result_type: str,
        value: Any,
        unit: Optional[str] = None,
        methodology: Optional[str] = None,
    ) -> int:
        """
        Send live calculation result.

        Args:
            execution_id: Execution ID
            result_type: Result type (final, intermediate, partial)
            value: Calculated value
            unit: Optional unit
            methodology: Optional methodology reference

        Returns:
            Number of connections notified
        """
        message = WebSocketMessage(
            type=MessageType.CALCULATION_RESULT,
            data={
                "execution_id": execution_id,
                "result_type": result_type,
                "value": value,
                "unit": unit,
                "methodology": methodology,
            },
            room=f"execution:{execution_id}",
        )

        return await self.connection_manager.broadcast_to_room(
            f"execution:{execution_id}",
            message,
        )

    async def send_alert(
        self,
        tenant_id: str,
        alert_type: str,
        severity: str,
        message_text: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Send alert notification.

        Args:
            tenant_id: Tenant ID
            alert_type: Type of alert
            severity: Alert severity (info, warning, error, critical)
            message_text: Alert message
            data: Optional additional data

        Returns:
            Number of connections notified
        """
        message = WebSocketMessage(
            type=MessageType.ALERT_TRIGGERED,
            data={
                "tenant_id": tenant_id,
                "alert_type": alert_type,
                "severity": severity,
                "message": message_text,
                "data": data,
            },
            room=f"tenant:{tenant_id}",
        )

        return await self.connection_manager.broadcast_to_tenant(tenant_id, message)


# =============================================================================
# Factory Function
# =============================================================================


def create_websocket_server(
    redis_client: Optional[Any] = None,
    heartbeat_interval: int = 30,
    heartbeat_timeout: int = 60,
    max_connections_per_tenant: int = 100,
) -> WebSocketServer:
    """
    Create a configured WebSocket server.

    Args:
        redis_client: Optional Redis client for distributed messaging
        heartbeat_interval: Seconds between heartbeat pings
        heartbeat_timeout: Seconds before connection considered dead
        max_connections_per_tenant: Maximum connections per tenant

    Returns:
        Configured WebSocketServer instance

    Example:
        >>> ws_server = create_websocket_server(redis_client)
        >>> await ws_server.start()
        >>> app.include_router(ws_server.router)
    """
    return WebSocketServer(
        redis_client=redis_client,
        heartbeat_interval=heartbeat_interval,
        heartbeat_timeout=heartbeat_timeout,
        max_connections_per_tenant=max_connections_per_tenant,
    )
