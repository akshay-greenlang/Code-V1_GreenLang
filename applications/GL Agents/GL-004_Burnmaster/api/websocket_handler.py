"""
GL-004 BURNMASTER WebSocket Handler

WebSocket endpoints for real-time updates including status streaming,
recommendations, and alerts with connection lifecycle management.
"""

from fastapi import WebSocket, WebSocketDisconnect, Depends, Query, status
from fastapi.routing import APIRouter
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import logging
import uuid
import random

from .api_auth import get_current_user_from_token, User
from .api_schemas import (
    BurnerMetrics, BurnerState, OperatingMode,
    AlertSeverity, AlertStatus,
    RecommendationPriority, RecommendationStatus
)
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["WebSocket"])


# ============================================================================
# Connection Manager
# ============================================================================

class ConnectionState(str, Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTED = "disconnected"


class WebSocketConnection:
    """Represents a single WebSocket connection."""

    def __init__(
        self,
        websocket: WebSocket,
        connection_id: str,
        unit_id: str,
        channel: str
    ):
        self.websocket = websocket
        self.connection_id = connection_id
        self.unit_id = unit_id
        self.channel = channel
        self.state = ConnectionState.CONNECTING
        self.user: Optional[User] = None
        self.connected_at = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        self.message_count = 0

    async def send_json(self, data: dict):
        """Send JSON data to the client."""
        try:
            await self.websocket.send_json(data)
            self.message_count += 1
        except Exception as e:
            logger.error(f"Error sending to {self.connection_id}: {e}")
            raise

    async def send_heartbeat(self):
        """Send heartbeat message."""
        await self.send_json({
            "type": "heartbeat",
            "timestamp": datetime.utcnow().isoformat()
        })
        self.last_heartbeat = datetime.utcnow()


class ConnectionManager:
    """
    Manages WebSocket connections for all units and channels.

    Handles:
    - Connection lifecycle (connect, authenticate, disconnect)
    - Channel subscription (status, recommendations, alerts)
    - Broadcasting messages to subscribers
    - Heartbeat management
    """

    def __init__(self):
        # Connections organized by unit_id -> channel -> connection_id
        self._connections: Dict[str, Dict[str, Dict[str, WebSocketConnection]]] = {}
        # All connections by connection_id
        self._all_connections: Dict[str, WebSocketConnection] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        unit_id: str,
        channel: str,
        user: Optional[User] = None
    ) -> WebSocketConnection:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: WebSocket instance
            unit_id: Unit identifier to subscribe to
            channel: Channel type (status, recommendations, alerts)
            user: Authenticated user (optional)

        Returns:
            WebSocketConnection object
        """
        await websocket.accept()

        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(
            websocket=websocket,
            connection_id=connection_id,
            unit_id=unit_id,
            channel=channel
        )
        connection.state = ConnectionState.CONNECTED
        connection.user = user

        if user:
            connection.state = ConnectionState.AUTHENTICATED

        async with self._lock:
            # Initialize nested dicts if needed
            if unit_id not in self._connections:
                self._connections[unit_id] = {}
            if channel not in self._connections[unit_id]:
                self._connections[unit_id][channel] = {}

            # Check connection limit
            channel_connections = len(self._connections[unit_id][channel])
            if channel_connections >= settings.websocket.max_connections_per_unit:
                await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
                raise Exception("Connection limit exceeded for unit")

            self._connections[unit_id][channel][connection_id] = connection
            self._all_connections[connection_id] = connection

        logger.info(
            f"WebSocket connected: {connection_id} to {unit_id}/{channel} "
            f"(user: {user.email if user else 'anonymous'})"
        )

        # Send welcome message
        await connection.send_json({
            "type": "connected",
            "connection_id": connection_id,
            "unit_id": unit_id,
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat()
        })

        return connection

    async def disconnect(self, connection: WebSocketConnection):
        """
        Remove a WebSocket connection.

        Args:
            connection: Connection to remove
        """
        async with self._lock:
            unit_id = connection.unit_id
            channel = connection.channel
            connection_id = connection.connection_id

            if (unit_id in self._connections and
                channel in self._connections[unit_id] and
                connection_id in self._connections[unit_id][channel]):
                del self._connections[unit_id][channel][connection_id]

                # Clean up empty dicts
                if not self._connections[unit_id][channel]:
                    del self._connections[unit_id][channel]
                if not self._connections[unit_id]:
                    del self._connections[unit_id]

            if connection_id in self._all_connections:
                del self._all_connections[connection_id]

        connection.state = ConnectionState.DISCONNECTED
        logger.info(f"WebSocket disconnected: {connection.connection_id}")

    async def broadcast_to_unit(
        self,
        unit_id: str,
        channel: str,
        message: dict
    ):
        """
        Broadcast message to all connections for a unit/channel.

        Args:
            unit_id: Target unit
            channel: Target channel
            message: Message to broadcast
        """
        async with self._lock:
            if (unit_id not in self._connections or
                channel not in self._connections[unit_id]):
                return

            connections = list(self._connections[unit_id][channel].values())

        disconnected = []
        for conn in connections:
            try:
                await conn.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to {conn.connection_id}: {e}")
                disconnected.append(conn)

        # Clean up disconnected connections
        for conn in disconnected:
            await self.disconnect(conn)

    async def broadcast_to_all(self, channel: str, message: dict):
        """Broadcast message to all connections on a channel."""
        async with self._lock:
            connections = [
                conn for unit_conns in self._connections.values()
                for ch_conns in unit_conns.values()
                for conn in ch_conns.values()
                if conn.channel == channel
            ]

        for conn in connections:
            try:
                await conn.send_json(message)
            except Exception:
                pass

    async def send_heartbeats(self):
        """Send heartbeat to all connections."""
        async with self._lock:
            connections = list(self._all_connections.values())

        for conn in connections:
            try:
                await conn.send_heartbeat()
            except Exception:
                pass

    def get_connection_count(self, unit_id: Optional[str] = None) -> int:
        """Get number of active connections."""
        if unit_id:
            return sum(
                len(ch_conns)
                for ch_conns in self._connections.get(unit_id, {}).values()
            )
        return len(self._all_connections)


# Global connection manager
manager = ConnectionManager()


# ============================================================================
# Data Generators
# ============================================================================

def generate_status_update(unit_id: str) -> dict:
    """Generate a status update message."""
    return {
        "type": "status_update",
        "unit_id": unit_id,
        "data": {
            "state": "running",
            "mode": "normal",
            "metrics": {
                "firing_rate": 75.5 + random.uniform(-5, 5),
                "fuel_flow_rate": 120.0 + random.uniform(-10, 10),
                "air_flow_rate": 1500.0 + random.uniform(-50, 50),
                "combustion_air_temp": 35.0 + random.uniform(-2, 2),
                "flue_gas_temp": 180.0 + random.uniform(-10, 10),
                "oxygen_level": 3.5 + random.uniform(-0.5, 0.5),
                "co_level": 15.0 + random.uniform(-5, 5),
                "nox_level": 45.0 + random.uniform(-5, 5),
                "efficiency": 94.2 + random.uniform(-1, 1),
                "heat_output": 12.5 + random.uniform(-0.5, 0.5)
            },
            "uptime_hours": 1250.5,
            "active_alerts_count": 2
        },
        "timestamp": datetime.utcnow().isoformat()
    }


def generate_recommendation(unit_id: str) -> dict:
    """Generate a recommendation message."""
    return {
        "type": "new_recommendation",
        "unit_id": unit_id,
        "data": {
            "recommendation_id": f"rec-{uuid.uuid4().hex[:8]}",
            "title": "Optimize Air-Fuel Ratio",
            "description": "Analysis indicates potential for efficiency improvement",
            "priority": "high",
            "status": "pending",
            "category": "efficiency",
            "impact": {
                "efficiency_improvement": 2.3,
                "emissions_reduction": 1.5,
                "cost_savings": 450.0,
                "confidence_level": 0.92
            },
            "valid_until": (datetime.utcnow() + timedelta(hours=24)).isoformat()
        },
        "timestamp": datetime.utcnow().isoformat()
    }


def generate_alert(unit_id: str) -> dict:
    """Generate an alert message."""
    return {
        "type": "alert",
        "unit_id": unit_id,
        "data": {
            "alert_id": f"alert-{uuid.uuid4().hex[:8]}",
            "severity": "warning",
            "status": "active",
            "title": "High Flue Gas Temperature",
            "description": "Flue gas temperature exceeds optimal range",
            "source": "temperature_monitor",
            "metric_name": "flue_gas_temp",
            "metric_value": 195.0,
            "threshold": 190.0,
            "recommended_action": "Check heat exchanger efficiency"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Background Tasks
# ============================================================================

async def heartbeat_task():
    """Background task to send heartbeats."""
    while True:
        await asyncio.sleep(settings.websocket.heartbeat_interval_seconds)
        await manager.send_heartbeats()


async def status_broadcast_task():
    """Background task to broadcast status updates."""
    units = ["burner-001", "burner-002"]
    while True:
        await asyncio.sleep(5)  # Update every 5 seconds
        for unit_id in units:
            message = generate_status_update(unit_id)
            await manager.broadcast_to_unit(unit_id, "status", message)


async def start_background_tasks():
    """Start all background tasks."""
    asyncio.create_task(heartbeat_task())
    asyncio.create_task(status_broadcast_task())


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@router.websocket("/ws/units/{unit_id}/status")
async def websocket_status(
    websocket: WebSocket,
    unit_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time status updates.

    Streams burner metrics and state changes at regular intervals.

    Args:
        websocket: WebSocket connection
        unit_id: Unit to subscribe to
        token: Optional JWT token for authentication
    """
    # Authenticate if token provided
    user = None
    if token:
        try:
            from .api_auth import decode_token
            token_data = decode_token(token)
            user = User(
                id=token_data.user_id,
                email=token_data.email,
                full_name="",
                roles=token_data.roles,
                tenant_id=token_data.tenant_id
            )
        except Exception as e:
            logger.warning(f"WebSocket auth failed: {e}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    try:
        connection = await manager.connect(websocket, unit_id, "status", user)

        while True:
            try:
                # Wait for client messages (ping/pong, commands)
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=settings.websocket.heartbeat_interval_seconds * 2
                )

                # Handle client messages
                if data.get("type") == "ping":
                    await connection.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif data.get("type") == "subscribe":
                    # Handle additional subscriptions if needed
                    pass

            except asyncio.TimeoutError:
                # No message received, send status update
                await connection.send_json(generate_status_update(unit_id))

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from {unit_id}/status")
    except Exception as e:
        logger.error(f"WebSocket error for {unit_id}/status: {e}")
    finally:
        if 'connection' in locals():
            await manager.disconnect(connection)


@router.websocket("/ws/units/{unit_id}/recommendations")
async def websocket_recommendations(
    websocket: WebSocket,
    unit_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time recommendations.

    Streams new optimization recommendations as they are generated.

    Args:
        websocket: WebSocket connection
        unit_id: Unit to subscribe to
        token: Optional JWT token for authentication
    """
    user = None
    if token:
        try:
            from .api_auth import decode_token
            token_data = decode_token(token)
            user = User(
                id=token_data.user_id,
                email=token_data.email,
                full_name="",
                roles=token_data.roles,
                tenant_id=token_data.tenant_id
            )
        except Exception:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    try:
        connection = await manager.connect(websocket, unit_id, "recommendations", user)

        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30
                )

                if data.get("type") == "ping":
                    await connection.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })

            except asyncio.TimeoutError:
                # Periodically send new recommendations (simulated)
                if random.random() < 0.1:  # 10% chance each check
                    await connection.send_json(generate_recommendation(unit_id))

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from {unit_id}/recommendations")
    except Exception as e:
        logger.error(f"WebSocket error for {unit_id}/recommendations: {e}")
    finally:
        if 'connection' in locals():
            await manager.disconnect(connection)


@router.websocket("/ws/units/{unit_id}/alerts")
async def websocket_alerts(
    websocket: WebSocket,
    unit_id: str,
    min_severity: Optional[str] = Query(None),
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time alerts.

    Streams alerts as they are triggered, with optional severity filtering.

    Args:
        websocket: WebSocket connection
        unit_id: Unit to subscribe to
        min_severity: Minimum severity level to receive
        token: Optional JWT token for authentication
    """
    user = None
    if token:
        try:
            from .api_auth import decode_token
            token_data = decode_token(token)
            user = User(
                id=token_data.user_id,
                email=token_data.email,
                full_name="",
                roles=token_data.roles,
                tenant_id=token_data.tenant_id
            )
        except Exception:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    try:
        connection = await manager.connect(websocket, unit_id, "alerts", user)

        # Send subscription confirmation with filters
        await connection.send_json({
            "type": "subscribed",
            "channel": "alerts",
            "filters": {
                "min_severity": min_severity
            },
            "timestamp": datetime.utcnow().isoformat()
        })

        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30
                )

                if data.get("type") == "ping":
                    await connection.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif data.get("type") == "acknowledge":
                    # Handle alert acknowledgement via WebSocket
                    alert_id = data.get("alert_id")
                    await connection.send_json({
                        "type": "acknowledged",
                        "alert_id": alert_id,
                        "timestamp": datetime.utcnow().isoformat()
                    })

            except asyncio.TimeoutError:
                # Periodically send alerts (simulated)
                if random.random() < 0.05:  # 5% chance each check
                    alert = generate_alert(unit_id)

                    # Apply severity filter
                    if min_severity:
                        severity_order = ["info", "warning", "error", "critical"]
                        alert_severity = alert["data"]["severity"]
                        if severity_order.index(alert_severity) < severity_order.index(min_severity):
                            continue

                    await connection.send_json(alert)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from {unit_id}/alerts")
    except Exception as e:
        logger.error(f"WebSocket error for {unit_id}/alerts: {e}")
    finally:
        if 'connection' in locals():
            await manager.disconnect(connection)


# ============================================================================
# Utility Functions
# ============================================================================

async def notify_status_change(unit_id: str, status_data: dict):
    """Notify all subscribers of a status change."""
    message = {
        "type": "status_change",
        "unit_id": unit_id,
        "data": status_data,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast_to_unit(unit_id, "status", message)


async def notify_new_recommendation(unit_id: str, recommendation_data: dict):
    """Notify all subscribers of a new recommendation."""
    message = {
        "type": "new_recommendation",
        "unit_id": unit_id,
        "data": recommendation_data,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast_to_unit(unit_id, "recommendations", message)


async def notify_alert(unit_id: str, alert_data: dict):
    """Notify all subscribers of an alert."""
    message = {
        "type": "alert",
        "unit_id": unit_id,
        "data": alert_data,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast_to_unit(unit_id, "alerts", message)
