"""
Server-Sent Events (SSE) Streaming Manager for GreenLang Process Heat Agents

This module provides enterprise-grade SSE streaming capabilities for real-time
Process Heat agent monitoring, calculation progress tracking, alarm updates,
and live metrics streaming.

Features:
    - Real-time agent status updates (RUNNING, IDLE, ERROR, COMPLETED)
    - Job progress tracking (0-100% with detailed step information)
    - Alarm state change notifications with severity levels
    - Live metrics streaming (temperature, pressure, efficiency, etc.)
    - Automatic heartbeat every 30s to maintain connection
    - Last-Event-ID support for client reconnection
    - Client filtering by event type subscription
    - Connection pooling with per-client limits
    - Backpressure handling for slow clients
    - Event history replay capability

Example:
    >>> from greenlang.infrastructure.api.sse_streaming import SSEStreamManager
    >>> manager = SSEStreamManager()
    >>> await app.app_context.sse_manager.start()
    >>> # Agent status update
    >>> await manager.send_agent_status("agent-001", "RUNNING", {"progress": 50})
    >>> # Job progress notification
    >>> await manager.send_job_progress("job-123", 75, "Calculating emissions")
    >>> # Alarm update
    >>> await manager.send_alarm_update("alarm-001", "CRITICAL", "Pressure exceeded")
    >>> # Metrics streaming
    >>> await manager.send_metrics("metrics", {"temp": 150.5, "pressure": 8.5})
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, validator

try:
    from fastapi import APIRouter, Request, Query, Path
    from fastapi.responses import StreamingResponse, Response
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object
    Request = None
    StreamingResponse = None
    Response = None

logger = logging.getLogger(__name__)


class EventTypeEnum(str, Enum):
    """Process Heat agent event types."""
    AGENT_STATUS = "agent.status"
    CALCULATION_PROGRESS = "calculation.progress"
    ALARM_UPDATE = "alarm.update"
    METRICS_UPDATE = "metrics.update"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    JOB_COMPLETE = "job.complete"


class AgentStatusEnum(str, Enum):
    """Agent execution status."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


class AlarmSeverityEnum(str, Enum):
    """Alarm severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    ALERT = "ALERT"


@dataclass
class SSEStreamConfig:
    """Configuration for SSE stream manager."""
    heartbeat_interval_seconds: int = 30
    client_timeout_seconds: int = 300
    max_clients_per_stream: int = 1000
    max_queue_size: int = 500
    max_event_history: int = 100
    enable_heartbeat: bool = True
    enable_compression: bool = False
    api_prefix: str = "/api/v1/stream"


class AgentStatusUpdate(BaseModel):
    """Agent status update event."""
    agent_id: str = Field(..., description="Agent identifier")
    status: AgentStatusEnum = Field(..., description="Current agent status")
    progress_percent: int = Field(default=0, ge=0, le=100, description="Progress percentage")
    current_step: Optional[str] = Field(None, description="Current processing step")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @validator('progress_percent')
    def validate_progress(cls, v):
        """Validate progress is 0-100."""
        if not 0 <= v <= 100:
            raise ValueError("Progress must be between 0 and 100")
        return v


class CalculationProgress(BaseModel):
    """Job/calculation progress event."""
    job_id: str = Field(..., description="Job identifier")
    agent_id: str = Field(..., description="Agent executing job")
    progress_percent: int = Field(default=0, ge=0, le=100, description="Progress 0-100")
    current_step: str = Field(..., description="Current step name")
    step_number: int = Field(..., ge=1, description="Step number")
    total_steps: int = Field(..., ge=1, description="Total steps")
    elapsed_seconds: float = Field(default=0, ge=0, description="Elapsed time")
    estimated_total_seconds: Optional[float] = Field(None, description="ETA total seconds")
    details: Dict[str, Any] = Field(default_factory=dict, description="Step details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @validator('progress_percent')
    def validate_progress(cls, v):
        """Validate progress is consistent with steps."""
        if not 0 <= v <= 100:
            raise ValueError("Progress must be between 0 and 100")
        return v


class AlarmUpdate(BaseModel):
    """Alarm state change event."""
    alarm_id: str = Field(..., description="Alarm identifier")
    agent_id: str = Field(..., description="Source agent")
    severity: AlarmSeverityEnum = Field(..., description="Alarm severity")
    message: str = Field(..., description="Alarm message")
    parameter: Optional[str] = Field(None, description="Parameter name")
    current_value: Optional[float] = Field(None, description="Current parameter value")
    threshold_value: Optional[float] = Field(None, description="Alarm threshold")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional info")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsUpdate(BaseModel):
    """Live metrics update event."""
    source_id: str = Field(..., description="Source (agent or device ID)")
    metrics: Dict[str, float] = Field(..., description="Metric name to value mapping")
    unit: Optional[str] = Field(None, description="Metrics unit")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SSEStreamEvent(BaseModel):
    """SSE event envelope."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventTypeEnum = Field(..., description="Event type")
    data: Any = Field(..., description="Event data")
    channel: str = Field(..., description="Channel name")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retry_ms: Optional[int] = Field(default=3000, description="Retry interval")

    def to_sse_format(self) -> str:
        """Convert to SSE wire format (RFC 6202)."""
        lines = []

        # Event ID for reconnection support
        lines.append(f"id: {self.event_id}")

        # Event type
        lines.append(f"event: {self.event_type.value}")

        # Data (JSON encoded, multiline safe)
        if isinstance(self.data, (dict, list)):
            data_str = json.dumps(self.data)
        else:
            data_str = str(self.data)

        for line in data_str.split("\n"):
            lines.append(f"data: {line}")

        # Retry interval
        if self.retry_ms:
            lines.append(f"retry: {self.retry_ms}")

        # End with double newline per SSE spec
        lines.append("")
        lines.append("")

        return "\n".join(lines)

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash for event provenance."""
        content = f"{self.event_id}{self.event_type.value}{json.dumps(self.data, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()


class StreamSubscription(BaseModel):
    """Client stream subscription."""
    client_id: str = Field(default_factory=lambda: str(uuid4()))
    subscribed_channels: Set[str] = Field(default_factory=set)
    event_type_filter: Set[EventTypeEnum] = Field(default_factory=set)
    connected_at: datetime = Field(default_factory=datetime.utcnow)
    last_event_id: Optional[str] = Field(None)
    user_agent: Optional[str] = Field(None)
    remote_addr: Optional[str] = Field(None)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True


class EventStream:
    """Event stream for a specific channel."""

    def __init__(self, name: str, max_queue_size: int = 500, max_history: int = 100):
        """Initialize event stream."""
        self.name = name
        self.max_queue_size = max_queue_size
        self.max_history = max_history
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self._event_filters: Dict[str, Callable[[SSEStreamEvent], bool]] = {}
        self._event_history: List[SSEStreamEvent] = []

    async def subscribe(
        self,
        client_id: str,
        last_event_id: Optional[str] = None,
        event_types: Optional[Set[EventTypeEnum]] = None
    ) -> Tuple[asyncio.Queue, List[SSEStreamEvent]]:
        """
        Subscribe client to stream.

        Args:
            client_id: Client identifier
            last_event_id: Last received event ID for replay
            event_types: Filter by event types

        Returns:
            Queue and list of missed events for replay
        """
        queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._subscribers[client_id] = queue

        # Set up event type filter if provided
        if event_types:
            self._event_filters[client_id] = lambda e: e.event_type in event_types

        # Prepare replay events
        replay_events = []
        if last_event_id:
            found = False
            for event in self._event_history:
                if found:
                    replay_events.append(event)
                elif event.event_id == last_event_id:
                    found = True

        logger.info(f"Client {client_id} subscribed to {self.name}, replaying {len(replay_events)} events")
        return queue, replay_events

    async def unsubscribe(self, client_id: str) -> None:
        """Unsubscribe client from stream."""
        self._subscribers.pop(client_id, None)
        self._event_filters.pop(client_id, None)
        logger.debug(f"Client {client_id} unsubscribed from {self.name}")

    async def broadcast(self, event: SSEStreamEvent) -> int:
        """
        Broadcast event to all subscribers.

        Args:
            event: Event to broadcast

        Returns:
            Number of subscribers that received event
        """
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self.max_history:
            self._event_history = self._event_history[-self.max_history:]

        delivered = 0
        disconnected = []

        for client_id, queue in self._subscribers.items():
            # Check filter
            filter_fn = self._event_filters.get(client_id)
            if filter_fn and not filter_fn(event):
                continue

            try:
                queue.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                logger.warning(f"Queue full for client {client_id} on {self.name}")
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {e}")
                disconnected.append(client_id)

        # Clean up dead subscribers
        for client_id in disconnected:
            await self.unsubscribe(client_id)

        return delivered

    @property
    def subscriber_count(self) -> int:
        """Get number of active subscribers."""
        return len(self._subscribers)


class SSEStreamManager:
    """
    Server-Sent Events Stream Manager for Process Heat Agents.

    Manages real-time event streaming with support for agent status,
    calculation progress, alarms, and metrics.

    Attributes:
        config: Stream manager configuration
        router: FastAPI router for SSE endpoints
        streams: Active event streams by channel

    Example:
        >>> manager = SSEStreamManager()
        >>> await manager.start()
        >>> await manager.send_agent_status("agent-001", AgentStatusEnum.RUNNING, {"progress": 50})
    """

    def __init__(self, config: Optional[SSEStreamConfig] = None):
        """Initialize SSE stream manager."""
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI required for SSE support. Install: pip install fastapi starlette"
            )

        self.config = config or SSEStreamConfig()
        self.router = APIRouter(prefix=self.config.api_prefix, tags=["sse"])
        self._streams: Dict[str, EventStream] = {}
        self._subscriptions: Dict[str, StreamSubscription] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Setup FastAPI routes
        self._setup_routes()
        logger.info("SSEStreamManager initialized with config: heartbeat=%ds, timeout=%ds",
                   self.config.heartbeat_interval_seconds,
                   self.config.client_timeout_seconds)

    def _setup_routes(self) -> None:
        """Set up FastAPI SSE routes."""
        @self.router.get("/events", summary="Subscribe to global events stream")
        async def stream_events(
            request: Request,
            event_types: Optional[str] = Query(None, description="Comma-separated event types"),
        ) -> StreamingResponse:
            """Main SSE endpoint for all events."""
            types = set()
            if event_types:
                for et in event_types.split(","):
                    try:
                        types.add(EventTypeEnum(et))
                    except ValueError:
                        pass
            return await self._handle_stream_request("global", request, types)

        @self.router.get("/agents/{agent_id}", summary="Subscribe to agent-specific stream")
        async def stream_agent(
            request: Request,
            agent_id: str = Path(..., description="Agent ID"),
        ) -> StreamingResponse:
            """Agent-specific SSE endpoint."""
            return await self._handle_stream_request(f"agent:{agent_id}", request)

        @self.router.get("/jobs/{job_id}", summary="Subscribe to job progress stream")
        async def stream_job(
            request: Request,
            job_id: str = Path(..., description="Job ID"),
        ) -> StreamingResponse:
            """Job progress SSE endpoint."""
            return await self._handle_stream_request(f"job:{job_id}", request, {EventTypeEnum.CALCULATION_PROGRESS})

        @self.router.get("/status", summary="Get stream manager status")
        async def get_status() -> Dict[str, Any]:
            """Get streaming manager statistics."""
            return self.get_statistics()

    async def _handle_stream_request(
        self,
        channel: str,
        request: Request,
        event_types: Optional[Set[EventTypeEnum]] = None
    ) -> StreamingResponse:
        """Handle SSE subscription request."""
        # Get or create stream
        if channel not in self._streams:
            self._streams[channel] = EventStream(
                channel,
                max_queue_size=self.config.max_queue_size,
                max_history=self.config.max_event_history
            )

        stream = self._streams[channel]

        # Enforce connection limit
        if stream.subscriber_count >= self.config.max_clients_per_stream:
            return Response(status_code=503, content="Stream at capacity")

        # Create subscription
        last_event_id = request.headers.get("Last-Event-ID")
        sub = StreamSubscription(
            subscribed_channels={channel},
            event_type_filter=event_types or set(),
            last_event_id=last_event_id,
            user_agent=request.headers.get("User-Agent"),
            remote_addr=request.client.host if request.client else None,
        )
        self._subscriptions[sub.client_id] = sub

        # Subscribe and get replay
        queue, replay_events = await stream.subscribe(
            sub.client_id,
            last_event_id,
            event_types
        )

        return StreamingResponse(
            self._event_generator(sub.client_id, channel, queue, replay_events),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            }
        )

    async def _event_generator(
        self,
        client_id: str,
        channel: str,
        queue: asyncio.Queue,
        replay_events: List[SSEStreamEvent]
    ) -> AsyncGenerator[str, None]:
        """Generate SSE events for a client."""
        try:
            # Send replay events first
            for event in replay_events:
                yield event.to_sse_format()

            # Main event loop
            while not self._shutdown:
                try:
                    # Wait for event with heartbeat timeout
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=self.config.heartbeat_interval_seconds
                    )
                    yield event.to_sse_format()

                except asyncio.TimeoutError:
                    # Send heartbeat
                    if self.config.enable_heartbeat:
                        heartbeat = SSEStreamEvent(
                            event_type=EventTypeEnum.HEARTBEAT,
                            data={"timestamp": datetime.utcnow().isoformat()},
                            channel=channel
                        )
                        yield heartbeat.to_sse_format()

                        # Update subscription heartbeat
                        if client_id in self._subscriptions:
                            self._subscriptions[client_id].last_heartbeat = datetime.utcnow()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Event generator error for client {client_id}: {e}")
        finally:
            # Cleanup
            await self._cleanup_subscription(client_id, channel)

    async def _cleanup_subscription(self, client_id: str, channel: str) -> None:
        """Clean up subscription."""
        if channel in self._streams:
            await self._streams[channel].unsubscribe(client_id)

        self._subscriptions.pop(client_id, None)
        logger.debug(f"Cleaned up subscription for client {client_id} on {channel}")

    # =====================================================================
    # Public API for sending events
    # =====================================================================

    async def send_agent_status(
        self,
        agent_id: str,
        status: AgentStatusEnum,
        metadata: Optional[Dict[str, Any]] = None,
        progress_percent: int = 0,
        current_step: Optional[str] = None
    ) -> int:
        """
        Send agent status update.

        Args:
            agent_id: Agent identifier
            status: Agent status
            metadata: Additional metadata
            progress_percent: Progress 0-100
            current_step: Current processing step

        Returns:
            Number of clients that received event
        """
        update = AgentStatusUpdate(
            agent_id=agent_id,
            status=status,
            progress_percent=progress_percent,
            current_step=current_step,
            metadata=metadata or {}
        )

        event = SSEStreamEvent(
            event_type=EventTypeEnum.AGENT_STATUS,
            data=update.dict(),
            channel=f"agent:{agent_id}"
        )

        # Broadcast to both agent-specific and global channels
        count = 0
        count += await self._broadcast_event(f"agent:{agent_id}", event)
        count += await self._broadcast_event("global", event)
        return count

    async def send_job_progress(
        self,
        job_id: str,
        agent_id: str,
        progress_percent: int,
        current_step: str,
        step_number: int = 1,
        total_steps: int = 1,
        elapsed_seconds: float = 0,
        estimated_total_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Send job progress update.

        Args:
            job_id: Job identifier
            agent_id: Agent executing job
            progress_percent: Progress 0-100
            current_step: Step name
            step_number: Current step number
            total_steps: Total steps
            elapsed_seconds: Elapsed time
            estimated_total_seconds: ETA
            details: Step-specific details

        Returns:
            Number of clients that received event
        """
        progress = CalculationProgress(
            job_id=job_id,
            agent_id=agent_id,
            progress_percent=progress_percent,
            current_step=current_step,
            step_number=step_number,
            total_steps=total_steps,
            elapsed_seconds=elapsed_seconds,
            estimated_total_seconds=estimated_total_seconds,
            details=details or {}
        )

        event = SSEStreamEvent(
            event_type=EventTypeEnum.CALCULATION_PROGRESS,
            data=progress.dict(),
            channel=f"job:{job_id}"
        )

        # Broadcast to job-specific, agent, and global channels
        count = 0
        count += await self._broadcast_event(f"job:{job_id}", event)
        count += await self._broadcast_event(f"agent:{agent_id}", event)
        count += await self._broadcast_event("global", event)
        return count

    async def send_alarm_update(
        self,
        alarm_id: str,
        agent_id: str,
        severity: AlarmSeverityEnum,
        message: str,
        parameter: Optional[str] = None,
        current_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Send alarm update.

        Args:
            alarm_id: Alarm identifier
            agent_id: Source agent
            severity: Alarm severity
            message: Alarm message
            parameter: Parameter name
            current_value: Current value
            threshold_value: Threshold value
            metadata: Additional metadata

        Returns:
            Number of clients that received event
        """
        alarm = AlarmUpdate(
            alarm_id=alarm_id,
            agent_id=agent_id,
            severity=severity,
            message=message,
            parameter=parameter,
            current_value=current_value,
            threshold_value=threshold_value,
            metadata=metadata or {}
        )

        event = SSEStreamEvent(
            event_type=EventTypeEnum.ALARM_UPDATE,
            data=alarm.dict(),
            channel=f"agent:{agent_id}"
        )

        # Broadcast to agent and global channels
        count = 0
        count += await self._broadcast_event(f"agent:{agent_id}", event)
        count += await self._broadcast_event("global", event)
        return count

    async def send_metrics(
        self,
        source_id: str,
        metrics: Dict[str, float],
        unit: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Send live metrics update.

        Args:
            source_id: Source ID (agent or device)
            metrics: Metric name to value mapping
            unit: Optional unit of measurement
            metadata: Additional metadata

        Returns:
            Number of clients that received event
        """
        metrics_update = MetricsUpdate(
            source_id=source_id,
            metrics=metrics,
            unit=unit,
            metadata=metadata or {}
        )

        event = SSEStreamEvent(
            event_type=EventTypeEnum.METRICS_UPDATE,
            data=metrics_update.dict(),
            channel=f"agent:{source_id}"
        )

        count = 0
        count += await self._broadcast_event(f"agent:{source_id}", event)
        count += await self._broadcast_event("global", event)
        return count

    async def _broadcast_event(self, channel: str, event: SSEStreamEvent) -> int:
        """Broadcast event to a channel (internal)."""
        if channel not in self._streams:
            return 0
        return await self._streams[channel].broadcast(event)

    # =====================================================================
    # Connection management
    # =====================================================================

    async def start(self) -> None:
        """Start background tasks."""
        self._shutdown = False
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SSEStreamManager started")

    async def stop(self) -> None:
        """Stop background tasks."""
        self._shutdown = True
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("SSEStreamManager stopped")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up stale subscriptions."""
        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.utcnow()
                stale = []

                for client_id, sub in self._subscriptions.items():
                    age = (now - sub.connected_at).total_seconds()
                    heartbeat_age = (now - sub.last_heartbeat).total_seconds()

                    # Remove if too old or no heartbeat for timeout period
                    if age > self.config.client_timeout_seconds or \
                       heartbeat_age > (self.config.client_timeout_seconds * 2):
                        stale.append(client_id)

                for client_id in stale:
                    sub = self._subscriptions.get(client_id)
                    if sub:
                        for channel in sub.subscribed_channels:
                            await self._cleanup_subscription(client_id, channel)
                        logger.info(f"Cleaned up stale subscription {client_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    # =====================================================================
    # Query and monitoring
    # =====================================================================

    def get_connected_clients(self, channel: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of connected clients.

        Args:
            channel: Filter by channel

        Returns:
            List of client info
        """
        clients = []
        for client_id, sub in self._subscriptions.items():
            if channel and channel not in sub.subscribed_channels:
                continue

            clients.append({
                "client_id": client_id,
                "channels": list(sub.subscribed_channels),
                "connected_at": sub.connected_at.isoformat(),
                "last_heartbeat": sub.last_heartbeat.isoformat(),
                "user_agent": sub.user_agent,
                "remote_addr": sub.remote_addr
            })

        return clients

    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming manager statistics."""
        return {
            "total_clients": len(self._subscriptions),
            "total_streams": len(self._streams),
            "active_streams": {
                name: {
                    "subscribers": stream.subscriber_count,
                    "event_history_size": len(stream._event_history)
                }
                for name, stream in self._streams.items()
            },
            "config": {
                "heartbeat_interval_seconds": self.config.heartbeat_interval_seconds,
                "client_timeout_seconds": self.config.client_timeout_seconds,
                "max_queue_size": self.config.max_queue_size
            }
        }

    def close_stream(self, client_id: str) -> bool:
        """
        Close a client stream.

        Args:
            client_id: Client identifier

        Returns:
            True if client was found and removed
        """
        sub = self._subscriptions.get(client_id)
        if not sub:
            return False

        # Remove from all streams
        for channel in list(sub.subscribed_channels):
            asyncio.create_task(self._cleanup_subscription(client_id, channel))

        return True
