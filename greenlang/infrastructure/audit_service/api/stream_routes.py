# -*- coding: utf-8 -*-
"""
Audit Event Streaming REST API Routes - SEC-005

FastAPI router for real-time audit event streaming via WebSocket:

    WebSocket /api/v1/audit/events/stream - Real-time event subscription

Features:
- Redis pub/sub subscription to gl:audit:events channel
- Event filtering by type, tenant, severity
- 30-second heartbeat for connection health
- JWT authentication for secure access

Author: GreenLang Framework Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore[misc, assignment]
    Query = None  # type: ignore[assignment]
    WebSocket = None  # type: ignore[assignment]
    WebSocketDisconnect = Exception  # type: ignore[misc, assignment]
    status = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    Field = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDIS_CHANNEL = "gl:audit:events"
HEARTBEAT_INTERVAL = 30  # seconds
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY = 1  # seconds


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class StreamFilterRequest(BaseModel):
        """Filter configuration for event streaming."""

        event_types: Optional[List[str]] = Field(
            None, description="Event types to filter"
        )
        tenant_id: Optional[str] = Field(
            None, description="Filter by tenant/organization ID"
        )
        severity_min: Optional[str] = Field(
            None, description="Minimum severity level: debug, info, warning, error, critical"
        )
        categories: Optional[List[str]] = Field(
            None, description="Event categories to filter"
        )

    class StreamMessage(BaseModel):
        """WebSocket stream message."""

        type: str = Field(..., description="Message type: event, heartbeat, error, subscribed")
        data: Optional[Dict[str, Any]] = Field(None, description="Message payload")
        timestamp: float = Field(default_factory=time.time, description="Message timestamp")


# ---------------------------------------------------------------------------
# WebSocket Connection Manager
# ---------------------------------------------------------------------------


class AuditStreamManager:
    """Manages WebSocket connections and Redis pub/sub for audit event streaming."""

    def __init__(self):
        """Initialize the stream manager."""
        self._connections: Dict[str, WebSocket] = {}
        self._filters: Dict[str, StreamFilterRequest] = {}
        self._redis_client: Optional[Any] = None
        self._pubsub: Optional[Any] = None
        self._running = False
        self._listener_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def start(self, redis_url: str = "redis://localhost:6379") -> None:
        """Start the stream manager and Redis connection.

        Args:
            redis_url: Redis connection URL.
        """
        if self._running:
            return

        try:
            import redis.asyncio as aioredis

            self._redis_client = await aioredis.from_url(
                redis_url, decode_responses=True
            )
            self._pubsub = self._redis_client.pubsub()
            await self._pubsub.subscribe(REDIS_CHANNEL)

            self._running = True
            self._listener_task = asyncio.create_task(self._listen_to_redis())
            self._heartbeat_task = asyncio.create_task(self._send_heartbeats())

            logger.info("AuditStreamManager started, subscribed to %s", REDIS_CHANNEL)

        except ImportError:
            logger.warning("redis.asyncio not available, streaming disabled")
        except Exception as exc:
            logger.error("Failed to start AuditStreamManager: %s", exc)

    async def stop(self) -> None:
        """Stop the stream manager and close connections."""
        self._running = False

        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all WebSocket connections
        for conn_id, websocket in list(self._connections.items()):
            try:
                await websocket.close()
            except Exception:
                pass

        self._connections.clear()
        self._filters.clear()

        # Close Redis connection
        if self._pubsub:
            await self._pubsub.unsubscribe(REDIS_CHANNEL)
            await self._pubsub.close()

        if self._redis_client:
            await self._redis_client.close()

        logger.info("AuditStreamManager stopped")

    async def register(
        self,
        connection_id: str,
        websocket: WebSocket,
        filters: Optional[StreamFilterRequest] = None,
    ) -> None:
        """Register a new WebSocket connection.

        Args:
            connection_id: Unique connection identifier.
            websocket: WebSocket connection.
            filters: Optional event filters.
        """
        self._connections[connection_id] = websocket
        self._filters[connection_id] = filters or StreamFilterRequest()
        logger.info("Registered WebSocket connection: %s", connection_id)

    async def unregister(self, connection_id: str) -> None:
        """Unregister a WebSocket connection.

        Args:
            connection_id: Connection identifier.
        """
        self._connections.pop(connection_id, None)
        self._filters.pop(connection_id, None)
        logger.info("Unregistered WebSocket connection: %s", connection_id)

    async def update_filters(
        self, connection_id: str, filters: StreamFilterRequest
    ) -> None:
        """Update filters for a connection.

        Args:
            connection_id: Connection identifier.
            filters: New filter configuration.
        """
        if connection_id in self._connections:
            self._filters[connection_id] = filters

    def _matches_filter(
        self, event: Dict[str, Any], filters: StreamFilterRequest
    ) -> bool:
        """Check if an event matches the filter criteria.

        Args:
            event: Audit event data.
            filters: Filter configuration.

        Returns:
            True if event matches filters.
        """
        # Event type filter
        if filters.event_types:
            if event.get("event_type") not in filters.event_types:
                return False

        # Tenant filter
        if filters.tenant_id:
            if event.get("organization_id") != filters.tenant_id:
                return False

        # Category filter
        if filters.categories:
            if event.get("category") not in filters.categories:
                return False

        # Severity filter (minimum level)
        if filters.severity_min:
            severity_order = [
                "debug", "info", "notice", "warning",
                "error", "critical", "alert", "emergency"
            ]
            try:
                min_idx = severity_order.index(filters.severity_min.lower())
                event_severity = event.get("severity", "info").lower()
                event_idx = severity_order.index(event_severity)
                if event_idx < min_idx:
                    return False
            except ValueError:
                pass

        return True

    async def _listen_to_redis(self) -> None:
        """Listen for messages from Redis pub/sub and broadcast to connections."""
        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )

                if message and message["type"] == "message":
                    try:
                        event_data = json.loads(message["data"])
                        await self._broadcast_event(event_data)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in Redis message")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error listening to Redis: %s", exc)
                await asyncio.sleep(RECONNECT_DELAY)

    async def _broadcast_event(self, event: Dict[str, Any]) -> None:
        """Broadcast an event to all matching connections.

        Args:
            event: Audit event data.
        """
        disconnected: List[str] = []

        for conn_id, websocket in self._connections.items():
            filters = self._filters.get(conn_id, StreamFilterRequest())

            if not self._matches_filter(event, filters):
                continue

            try:
                message = StreamMessage(
                    type="event",
                    data=event,
                    timestamp=time.time(),
                )
                await websocket.send_json(message.model_dump())

            except Exception as exc:
                logger.warning("Failed to send to %s: %s", conn_id, exc)
                disconnected.append(conn_id)

        # Clean up disconnected clients
        for conn_id in disconnected:
            await self.unregister(conn_id)

    async def _send_heartbeats(self) -> None:
        """Send periodic heartbeats to all connections."""
        while self._running:
            await asyncio.sleep(HEARTBEAT_INTERVAL)

            disconnected: List[str] = []

            for conn_id, websocket in self._connections.items():
                try:
                    message = StreamMessage(
                        type="heartbeat",
                        data={"ping": True},
                        timestamp=time.time(),
                    )
                    await websocket.send_json(message.model_dump())

                except Exception as exc:
                    logger.warning("Heartbeat failed for %s: %s", conn_id, exc)
                    disconnected.append(conn_id)

            # Clean up disconnected clients
            for conn_id in disconnected:
                await self.unregister(conn_id)


# Global stream manager instance
_stream_manager: Optional[AuditStreamManager] = None


def get_stream_manager() -> AuditStreamManager:
    """Get or create the global AuditStreamManager.

    Returns:
        The AuditStreamManager singleton.
    """
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = AuditStreamManager()
    return _stream_manager


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    from greenlang.infrastructure.audit_service.models import (
        EventCategory,
        SeverityLevel,
    )

    stream_router = APIRouter(
        prefix="/api/v1/audit/events",
        tags=["Audit Streaming"],
    )

    @stream_router.websocket("/stream")
    async def audit_event_stream(
        websocket: WebSocket,
        token: Optional[str] = Query(None, description="JWT authentication token"),
        event_types: Optional[str] = Query(
            None, description="Comma-separated event types to filter"
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant/organization ID"
        ),
        severity_min: Optional[str] = Query(
            None, description="Minimum severity level"
        ),
        categories: Optional[str] = Query(
            None, description="Comma-separated categories to filter"
        ),
    ) -> None:
        """WebSocket endpoint for real-time audit event streaming.

        Establishes a persistent WebSocket connection for receiving audit
        events in real-time. Events are filtered based on query parameters.

        Message Types:
        - event: Audit event matching filters
        - heartbeat: Connection health check (every 30s)
        - error: Error notification
        - subscribed: Subscription confirmation

        Args:
            websocket: WebSocket connection.
            token: Optional JWT token for authentication.
            event_types: Comma-separated event types to filter.
            tenant_id: Tenant/organization filter.
            severity_min: Minimum severity level.
            categories: Comma-separated categories to filter.
        """
        # Authenticate if token provided
        user_id: Optional[str] = None
        if token:
            try:
                from jose import jwt

                # Get JWT secret from environment or config
                import os
                jwt_secret = os.getenv("JWT_SECRET", "your-secret-key")

                payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
                user_id = payload.get("sub")

            except Exception as exc:
                logger.warning("WebSocket authentication failed: %s", exc)
                await websocket.close(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="Invalid authentication token",
                )
                return

        # Accept the connection
        await websocket.accept()

        # Generate connection ID
        connection_id = f"{websocket.client.host}:{websocket.client.port}"
        if user_id:
            connection_id = f"{user_id}:{connection_id}"

        # Parse filters
        filters = StreamFilterRequest(
            event_types=event_types.split(",") if event_types else None,
            tenant_id=tenant_id,
            severity_min=severity_min,
            categories=categories.split(",") if categories else None,
        )

        # Get stream manager
        manager = get_stream_manager()

        # Ensure manager is started
        if not manager._running:
            import os
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            await manager.start(redis_url)

        # Register connection
        await manager.register(connection_id, websocket, filters)

        # Send subscription confirmation
        try:
            confirmation = StreamMessage(
                type="subscribed",
                data={
                    "connection_id": connection_id,
                    "filters": filters.model_dump(),
                },
                timestamp=time.time(),
            )
            await websocket.send_json(confirmation.model_dump())

        except Exception as exc:
            logger.error("Failed to send subscription confirmation: %s", exc)
            await manager.unregister(connection_id)
            return

        # Handle incoming messages
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    message_type = data.get("type")

                    if message_type == "ping":
                        # Respond to client ping
                        pong = StreamMessage(
                            type="pong",
                            data={"received": data.get("timestamp")},
                            timestamp=time.time(),
                        )
                        await websocket.send_json(pong.model_dump())

                    elif message_type == "update_filters":
                        # Update subscription filters
                        new_filters = StreamFilterRequest(**data.get("filters", {}))
                        await manager.update_filters(connection_id, new_filters)

                        ack = StreamMessage(
                            type="filters_updated",
                            data={"filters": new_filters.model_dump()},
                            timestamp=time.time(),
                        )
                        await websocket.send_json(ack.model_dump())

                    elif message_type == "pong":
                        # Client heartbeat response, no action needed
                        pass

                except Exception as exc:
                    if isinstance(exc, WebSocketDisconnect):
                        raise
                    logger.warning("Error processing WebSocket message: %s", exc)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: %s", connection_id)

        except Exception as exc:
            logger.error("WebSocket error: %s", exc)

        finally:
            await manager.unregister(connection_id)

else:
    stream_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - stream_router is None")


__all__ = ["stream_router", "get_stream_manager", "AuditStreamManager"]
