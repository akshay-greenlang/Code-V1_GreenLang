"""
Server-Sent Events (SSE) Manager for GreenLang

This module provides SSE streaming capabilities for
real-time data updates to web clients.

Features:
- Multi-channel streaming
- Client management
- Heartbeat/keepalive
- Reconnection support
- Event filtering
- Backpressure handling

Example:
    >>> manager = SSEManager(config)
    >>> async for event in manager.subscribe("emissions"):
    ...     print(event)
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, Request
    from fastapi.responses import StreamingResponse
    from starlette.responses import Response
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object
    Request = None
    StreamingResponse = None

logger = logging.getLogger(__name__)


class SSEEventType(str, Enum):
    """SSE event types."""
    MESSAGE = "message"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    RECONNECT = "reconnect"


@dataclass
class SSEManagerConfig:
    """Configuration for SSE manager."""
    prefix: str = "/sse"
    heartbeat_interval_seconds: int = 30
    client_timeout_seconds: int = 300
    max_clients_per_channel: int = 1000
    max_queue_size: int = 100
    enable_heartbeat: bool = True
    enable_reconnect_id: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


class SSEEvent(BaseModel):
    """SSE event model."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: SSEEventType = Field(default=SSEEventType.MESSAGE)
    data: Any = Field(..., description="Event data")
    channel: str = Field(..., description="Event channel")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retry_ms: Optional[int] = Field(default=None, description="Retry interval")

    def to_sse_format(self) -> str:
        """Convert to SSE wire format."""
        lines = []

        # Event ID
        lines.append(f"id: {self.event_id}")

        # Event type
        if self.event_type != SSEEventType.MESSAGE:
            lines.append(f"event: {self.event_type.value}")

        # Data (can be multi-line)
        data_str = json.dumps(self.data) if isinstance(self.data, (dict, list)) else str(self.data)
        for line in data_str.split("\n"):
            lines.append(f"data: {line}")

        # Retry
        if self.retry_ms:
            lines.append(f"retry: {self.retry_ms}")

        # End with double newline
        lines.append("")
        lines.append("")

        return "\n".join(lines)


class SSEClient(BaseModel):
    """SSE client connection."""
    client_id: str = Field(default_factory=lambda: str(uuid4()))
    channels: Set[str] = Field(default_factory=set)
    connected_at: datetime = Field(default_factory=datetime.utcnow)
    last_event_id: Optional[str] = Field(default=None)
    user_agent: Optional[str] = Field(default=None)
    remote_addr: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True


class SSEChannel:
    """
    SSE channel for broadcasting events.

    Manages subscriptions and event distribution
    for a single channel.
    """

    def __init__(self, name: str, max_queue_size: int = 100):
        """
        Initialize SSE channel.

        Args:
            name: Channel name
            max_queue_size: Maximum queue size per client
        """
        self.name = name
        self.max_queue_size = max_queue_size
        self._clients: Dict[str, asyncio.Queue] = {}
        self._filters: Dict[str, Callable] = {}
        self._event_history: List[SSEEvent] = []
        self._max_history = 100

    async def subscribe(
        self,
        client_id: str,
        last_event_id: Optional[str] = None
    ) -> asyncio.Queue:
        """
        Subscribe a client to the channel.

        Args:
            client_id: Client identifier
            last_event_id: Last received event ID for replay

        Returns:
            Event queue for the client
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._clients[client_id] = queue

        # Replay missed events if last_event_id provided
        if last_event_id:
            await self._replay_events(queue, last_event_id)

        logger.debug(f"Client {client_id} subscribed to channel {self.name}")
        return queue

    async def unsubscribe(self, client_id: str) -> None:
        """
        Unsubscribe a client from the channel.

        Args:
            client_id: Client identifier
        """
        if client_id in self._clients:
            del self._clients[client_id]
            logger.debug(f"Client {client_id} unsubscribed from channel {self.name}")

    async def broadcast(self, event: SSEEvent) -> int:
        """
        Broadcast event to all subscribers.

        Args:
            event: Event to broadcast

        Returns:
            Number of clients that received the event
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        delivered = 0
        disconnected = []

        for client_id, queue in self._clients.items():
            # Check filter
            filter_fn = self._filters.get(client_id)
            if filter_fn and not filter_fn(event):
                continue

            try:
                # Non-blocking put
                queue.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                # Queue full - client is slow
                logger.warning(f"Queue full for client {client_id}")
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {e}")
                disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            await self.unsubscribe(client_id)

        return delivered

    async def _replay_events(
        self,
        queue: asyncio.Queue,
        last_event_id: str
    ) -> None:
        """Replay events after last_event_id."""
        replay = False
        for event in self._event_history:
            if replay:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    break
            elif event.event_id == last_event_id:
                replay = True

    def add_filter(
        self,
        client_id: str,
        filter_fn: Callable[[SSEEvent], bool]
    ) -> None:
        """
        Add a filter for a client.

        Args:
            client_id: Client identifier
            filter_fn: Filter function
        """
        self._filters[client_id] = filter_fn

    def remove_filter(self, client_id: str) -> None:
        """Remove filter for a client."""
        self._filters.pop(client_id, None)

    @property
    def subscriber_count(self) -> int:
        """Get number of subscribers."""
        return len(self._clients)


class SSEManager:
    """
    Server-Sent Events manager.

    Manages SSE channels, client connections, and
    event broadcasting.

    Attributes:
        config: Manager configuration
        router: FastAPI router
        channels: Active channels

    Example:
        >>> manager = SSEManager()
        >>> # Register with FastAPI
        >>> app.include_router(manager.router)
        >>> # Publish events
        >>> await manager.publish("emissions", {"co2": 150.5})
    """

    def __init__(self, config: Optional[SSEManagerConfig] = None):
        """
        Initialize SSE manager.

        Args:
            config: Manager configuration
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for SSE support. "
                "Install with: pip install fastapi"
            )

        self.config = config or SSEManagerConfig()
        self.router = APIRouter(prefix=self.config.prefix)
        self._channels: Dict[str, SSEChannel] = {}
        self._clients: Dict[str, SSEClient] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Register routes
        self._setup_routes()

        logger.info("SSEManager initialized")

    def _setup_routes(self) -> None:
        """Set up SSE routes."""
        @self.router.get(
            "/{channel}",
            summary="Subscribe to SSE channel"
        )
        async def subscribe(
            channel: str,
            request: Request
        ) -> StreamingResponse:
            return await self._handle_subscription(channel, request)

        @self.router.get(
            "/",
            summary="List available channels"
        )
        async def list_channels() -> Dict[str, Any]:
            return {
                "channels": [
                    {
                        "name": name,
                        "subscribers": ch.subscriber_count
                    }
                    for name, ch in self._channels.items()
                ]
            }

    async def _handle_subscription(
        self,
        channel: str,
        request: Request
    ) -> StreamingResponse:
        """Handle SSE subscription request."""
        # Get or create channel
        if channel not in self._channels:
            self._channels[channel] = SSEChannel(
                channel,
                max_queue_size=self.config.max_queue_size
            )

        sse_channel = self._channels[channel]

        # Check max clients
        if sse_channel.subscriber_count >= self.config.max_clients_per_channel:
            return Response(status_code=503, content="Too many clients")

        # Create client
        client = SSEClient(
            channels={channel},
            last_event_id=request.headers.get("Last-Event-ID"),
            user_agent=request.headers.get("User-Agent"),
            remote_addr=request.client.host if request.client else None,
        )
        self._clients[client.client_id] = client

        # Subscribe to channel
        queue = await sse_channel.subscribe(
            client.client_id,
            client.last_event_id
        )

        # Create streaming response
        return StreamingResponse(
            self._event_generator(client.client_id, channel, queue),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    async def _event_generator(
        self,
        client_id: str,
        channel: str,
        queue: asyncio.Queue
    ) -> AsyncGenerator[str, None]:
        """Generate SSE events for a client."""
        try:
            while not self._shutdown:
                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=self.config.heartbeat_interval_seconds
                    )
                    yield event.to_sse_format()

                except asyncio.TimeoutError:
                    # Send heartbeat
                    if self.config.enable_heartbeat:
                        heartbeat = SSEEvent(
                            event_type=SSEEventType.HEARTBEAT,
                            data={"timestamp": datetime.utcnow().isoformat()},
                            channel=channel
                        )
                        yield heartbeat.to_sse_format()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SSE generator error: {e}")
        finally:
            # Cleanup
            await self._cleanup_client(client_id, channel)

    async def _cleanup_client(self, client_id: str, channel: str) -> None:
        """Clean up client connection."""
        if channel in self._channels:
            await self._channels[channel].unsubscribe(client_id)

        self._clients.pop(client_id, None)
        logger.debug(f"Cleaned up client {client_id}")

    def get_channel(self, name: str) -> SSEChannel:
        """
        Get or create a channel.

        Args:
            name: Channel name

        Returns:
            SSE channel
        """
        if name not in self._channels:
            self._channels[name] = SSEChannel(
                name,
                max_queue_size=self.config.max_queue_size
            )
        return self._channels[name]

    async def publish(
        self,
        channel: str,
        data: Any,
        event_type: SSEEventType = SSEEventType.MESSAGE
    ) -> int:
        """
        Publish an event to a channel.

        Args:
            channel: Target channel
            data: Event data
            event_type: Event type

        Returns:
            Number of clients that received the event
        """
        if channel not in self._channels:
            logger.debug(f"Channel {channel} has no subscribers")
            return 0

        event = SSEEvent(
            event_type=event_type,
            data=data,
            channel=channel
        )

        return await self._channels[channel].broadcast(event)

    async def publish_to_client(
        self,
        client_id: str,
        data: Any,
        event_type: SSEEventType = SSEEventType.MESSAGE
    ) -> bool:
        """
        Publish an event to a specific client.

        Args:
            client_id: Target client
            data: Event data
            event_type: Event type

        Returns:
            True if delivered
        """
        client = self._clients.get(client_id)
        if not client:
            return False

        for channel_name in client.channels:
            channel = self._channels.get(channel_name)
            if channel and client_id in channel._clients:
                event = SSEEvent(
                    event_type=event_type,
                    data=data,
                    channel=channel_name
                )
                try:
                    channel._clients[client_id].put_nowait(event)
                    return True
                except asyncio.QueueFull:
                    return False

        return False

    async def broadcast_all(
        self,
        data: Any,
        event_type: SSEEventType = SSEEventType.MESSAGE
    ) -> Dict[str, int]:
        """
        Broadcast to all channels.

        Args:
            data: Event data
            event_type: Event type

        Returns:
            Dictionary of channel to delivery count
        """
        results = {}
        for channel_name in self._channels:
            count = await self.publish(channel_name, data, event_type)
            results[channel_name] = count
        return results

    def get_client(self, client_id: str) -> Optional[SSEClient]:
        """Get client by ID."""
        return self._clients.get(client_id)

    def list_clients(self, channel: Optional[str] = None) -> List[SSEClient]:
        """
        List connected clients.

        Args:
            channel: Filter by channel

        Returns:
            List of clients
        """
        if channel:
            return [
                c for c in self._clients.values()
                if channel in c.channels
            ]
        return list(self._clients.values())

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get SSE manager statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_clients": len(self._clients),
            "total_channels": len(self._channels),
            "channels": {
                name: ch.subscriber_count
                for name, ch in self._channels.items()
            }
        }

    async def start(self) -> None:
        """Start background tasks."""
        self._shutdown = False

        if self.config.enable_heartbeat:
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop()
            )

        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop()
        )

        logger.info("SSE manager started")

    async def stop(self) -> None:
        """Stop background tasks."""
        self._shutdown = True

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("SSE manager stopped")

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.heartbeat_interval_seconds)
                # Heartbeats are sent in event generator on timeout
            except asyncio.CancelledError:
                break

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.utcnow()
                stale_clients = []

                for client_id, client in self._clients.items():
                    age = (now - client.connected_at).total_seconds()
                    if age > self.config.client_timeout_seconds:
                        stale_clients.append(client_id)

                for client_id in stale_clients:
                    client = self._clients.get(client_id)
                    if client:
                        for channel in client.channels:
                            await self._cleanup_client(client_id, channel)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
