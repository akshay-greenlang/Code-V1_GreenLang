# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - WebSocket Real-Time Streaming Handler

Provides WebSocket endpoint for real-time combustion data streaming:
- 10Hz combustion state updates (fuel flow, air flow, temperatures, O2, emissions)
- 1Hz stability metrics broadcast
- Control action notifications
- Connection health monitoring with ping/pong

Security:
- JWT token authentication via query params
- Rate limiting per client (max 5 connections)
- Graceful disconnection handling

Performance:
- Async concurrent stream management
- Prometheus metrics integration
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect, status
import jwt

from .config import settings
from monitoring.metrics import metrics_collector
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class StreamType(str, Enum):
    """WebSocket stream message types"""
    COMBUSTION_STATE = "combustion_state"
    STABILITY_METRICS = "stability_metrics"
    CONTROL_ACTION = "control_action"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    SUBSCRIPTION_ACK = "subscription_ack"


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: StreamType
    timestamp: str
    data: Dict[str, Any]
    sequence: int = 0

    def to_json(self) -> str:
        """Serialize message to JSON string"""
        return json.dumps({
            "type": self.type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "sequence": self.sequence
        })


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection"""
    websocket: WebSocket
    client_id: str
    user_id: str
    connected_at: datetime
    last_activity: datetime
    subscriptions: Set[StreamType] = field(default_factory=lambda: {
        StreamType.COMBUSTION_STATE,
        StreamType.STABILITY_METRICS,
        StreamType.CONTROL_ACTION
    })
    message_count: int = 0
    sequence: int = 0


class WebSocketConnectionManager:
    """
    Manages WebSocket connections for real-time combustion data streaming.

    Features:
    - Connection lifecycle management
    - Rate limiting (max connections per client)
    - Authentication validation
    - Multi-stream broadcasting
    - Health monitoring with ping/pong
    """

    MAX_CONNECTIONS_PER_CLIENT = 5
    PING_INTERVAL_SECONDS = 30
    PONG_TIMEOUT_SECONDS = 10
    STATE_STREAM_INTERVAL_MS = 100  # 10Hz
    STABILITY_STREAM_INTERVAL_MS = 1000  # 1Hz

    def __init__(self):
        """Initialize the connection manager"""
        # Active connections by connection ID
        self.active_connections: Dict[str, ConnectionInfo] = {}

        # Connection count per client (for rate limiting)
        self.connections_per_client: Dict[str, int] = defaultdict(int)

        # Global sequence counter for message ordering
        self._global_sequence: int = 0

        # Streaming tasks
        self._state_stream_task: Optional[asyncio.Task] = None
        self._stability_stream_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Reference to agent (set by main.py)
        self.agent = None

        # Running flag
        self._running = False

        logger.info("WebSocket connection manager initialized")

    def set_agent(self, agent) -> None:
        """Set reference to combustion control agent"""
        self.agent = agent

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token for WebSocket authentication.

        Args:
            token: JWT token from query params

        Returns:
            Decoded token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM]
            )

            # Verify token hasn't expired
            exp = payload.get('exp')
            if not exp or datetime.fromtimestamp(exp) < datetime.utcnow():
                logger.warning("WebSocket token has expired")
                return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("WebSocket token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid WebSocket token: {e}")
            return None
        except Exception as e:
            logger.error(f"WebSocket token verification failed: {e}")
            return None

    async def connect(
        self,
        websocket: WebSocket,
        token: str
    ) -> Optional[ConnectionInfo]:
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
            token: JWT authentication token

        Returns:
            ConnectionInfo if successful, None if rejected
        """
        # Verify token
        payload = await self.verify_token(token)
        if not payload:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            metrics_collector.api_request_counter.labels(
                agent="GL-005",
                method="WS",
                endpoint="/ws/stream",
                status="auth_failed"
            ).inc()
            return None

        client_id = payload.get('sub', 'unknown')
        user_id = payload.get('user_id', client_id)

        # Check rate limit (max connections per client)
        async with self._lock:
            if self.connections_per_client[client_id] >= self.MAX_CONNECTIONS_PER_CLIENT:
                logger.warning(
                    f"Connection rejected: Client {client_id} exceeded max connections "
                    f"({self.MAX_CONNECTIONS_PER_CLIENT})"
                )
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                metrics_collector.api_request_counter.labels(
                    agent="GL-005",
                    method="WS",
                    endpoint="/ws/stream",
                    status="rate_limited"
                ).inc()
                return None

            # Accept connection
            await websocket.accept()

            # Generate connection ID
            connection_id = f"{client_id}_{int(time.time() * 1000)}"

            # Create connection info
            now = datetime.utcnow()
            conn_info = ConnectionInfo(
                websocket=websocket,
                client_id=client_id,
                user_id=user_id,
                connected_at=now,
                last_activity=now
            )

            # Register connection
            self.active_connections[connection_id] = conn_info
            self.connections_per_client[client_id] += 1

        # Update metrics
        metrics_collector.active_connections.labels(agent="GL-005").set(
            len(self.active_connections)
        )
        metrics_collector.api_request_counter.labels(
            agent="GL-005",
            method="WS",
            endpoint="/ws/stream",
            status="connected"
        ).inc()

        logger.info(
            f"WebSocket connected: {connection_id} "
            f"(client: {client_id}, total: {len(self.active_connections)})"
        )

        # Send connection acknowledgment
        await self._send_message(
            conn_info,
            StreamType.CONNECTED,
            {
                "connection_id": connection_id,
                "client_id": client_id,
                "subscriptions": [s.value for s in conn_info.subscriptions],
                "streams": {
                    "combustion_state": f"{1000 / self.STATE_STREAM_INTERVAL_MS}Hz",
                    "stability_metrics": f"{1000 / self.STABILITY_STREAM_INTERVAL_MS}Hz"
                }
            }
        )

        return conn_info

    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Disconnect and unregister a WebSocket connection.

        Args:
            websocket: WebSocket to disconnect
        """
        async with self._lock:
            # Find connection by websocket
            connection_id = None
            conn_info = None

            for cid, info in self.active_connections.items():
                if info.websocket == websocket:
                    connection_id = cid
                    conn_info = info
                    break

            if connection_id and conn_info:
                # Decrement client connection count
                self.connections_per_client[conn_info.client_id] -= 1
                if self.connections_per_client[conn_info.client_id] <= 0:
                    del self.connections_per_client[conn_info.client_id]

                # Remove connection
                del self.active_connections[connection_id]

                logger.info(
                    f"WebSocket disconnected: {connection_id} "
                    f"(total: {len(self.active_connections)})"
                )

        # Update metrics
        metrics_collector.active_connections.labels(agent="GL-005").set(
            len(self.active_connections)
        )
        metrics_collector.api_request_counter.labels(
            agent="GL-005",
            method="WS",
            endpoint="/ws/stream",
            status="disconnected"
        ).inc()

    async def _send_message(
        self,
        conn_info: ConnectionInfo,
        msg_type: StreamType,
        data: Dict[str, Any]
    ) -> bool:
        """
        Send a message to a specific connection.

        Args:
            conn_info: Connection to send to
            msg_type: Message type
            data: Message data

        Returns:
            True if sent successfully
        """
        try:
            self._global_sequence += 1
            conn_info.sequence = self._global_sequence
            conn_info.message_count += 1

            message = WebSocketMessage(
                type=msg_type,
                timestamp=DeterministicClock.utcnow().isoformat(),
                data=data,
                sequence=self._global_sequence
            )

            await conn_info.websocket.send_text(message.to_json())
            conn_info.last_activity = datetime.utcnow()

            return True

        except Exception as e:
            logger.debug(f"Failed to send message: {e}")
            return False

    async def broadcast(
        self,
        msg_type: StreamType,
        data: Dict[str, Any]
    ) -> int:
        """
        Broadcast a message to all connected clients subscribed to the stream type.

        Args:
            msg_type: Message type
            data: Message data

        Returns:
            Number of clients that received the message
        """
        if not self.active_connections:
            return 0

        sent_count = 0
        failed_connections: List[str] = []

        for connection_id, conn_info in self.active_connections.items():
            # Check if client is subscribed to this stream type
            if msg_type not in conn_info.subscriptions:
                continue

            try:
                success = await self._send_message(conn_info, msg_type, data)
                if success:
                    sent_count += 1
                else:
                    failed_connections.append(connection_id)
            except Exception as e:
                logger.debug(f"Broadcast failed for {connection_id}: {e}")
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            if connection_id in self.active_connections:
                conn_info = self.active_connections[connection_id]
                await self.disconnect(conn_info.websocket)

        return sent_count

    async def broadcast_combustion_state(self) -> int:
        """
        Broadcast current combustion state to all connected clients.

        Returns:
            Number of clients that received the state
        """
        if not self.agent or not self.agent.current_state:
            return 0

        state = self.agent.current_state
        data = {
            "fuel_flow": state.fuel_flow,
            "air_flow": state.air_flow,
            "air_fuel_ratio": state.air_fuel_ratio,
            "flame_temperature": state.flame_temperature,
            "furnace_temperature": state.furnace_temperature,
            "flue_gas_temperature": state.flue_gas_temperature,
            "ambient_temperature": state.ambient_temperature,
            "fuel_pressure": state.fuel_pressure,
            "air_pressure": state.air_pressure,
            "furnace_pressure": state.furnace_pressure,
            "o2_percent": state.o2_percent,
            "co_ppm": state.co_ppm,
            "co2_percent": state.co2_percent,
            "nox_ppm": state.nox_ppm,
            "heat_output_kw": state.heat_output_kw,
            "thermal_efficiency": state.thermal_efficiency,
            "excess_air_percent": state.excess_air_percent,
            "state_timestamp": state.timestamp.isoformat()
        }

        return await self.broadcast(StreamType.COMBUSTION_STATE, data)

    async def broadcast_stability_metrics(self) -> int:
        """
        Broadcast latest stability metrics to all connected clients.

        Returns:
            Number of clients that received the metrics
        """
        if not self.agent or not self.agent.stability_history:
            return 0

        stability = self.agent.stability_history[-1]
        data = {
            "heat_output_stability_index": stability.heat_output_stability_index,
            "heat_output_variance": stability.heat_output_variance,
            "heat_output_cv": stability.heat_output_cv,
            "furnace_temp_stability": stability.furnace_temp_stability,
            "flame_temp_stability": stability.flame_temp_stability,
            "o2_stability": stability.o2_stability,
            "co_stability": stability.co_stability,
            "oscillation_detected": stability.oscillation_detected,
            "oscillation_frequency_hz": stability.oscillation_frequency_hz,
            "oscillation_amplitude": stability.oscillation_amplitude,
            "overall_stability_score": stability.overall_stability_score,
            "stability_rating": stability.stability_rating,
            "metrics_timestamp": stability.timestamp.isoformat()
        }

        return await self.broadcast(StreamType.STABILITY_METRICS, data)

    async def broadcast_control_action(self, action) -> int:
        """
        Broadcast control action notification to all connected clients.

        Args:
            action: ControlAction that was taken

        Returns:
            Number of clients that received the notification
        """
        data = {
            "action_id": action.action_id,
            "fuel_flow_setpoint": action.fuel_flow_setpoint,
            "fuel_flow_delta": action.fuel_flow_delta,
            "air_flow_setpoint": action.air_flow_setpoint,
            "air_flow_delta": action.air_flow_delta,
            "fuel_control_mode": action.fuel_control_mode,
            "air_control_mode": action.air_control_mode,
            "o2_trim_enabled": action.o2_trim_enabled,
            "fuel_valve_position": action.fuel_valve_position,
            "air_damper_position": action.air_damper_position,
            "safety_override": action.safety_override,
            "interlock_satisfied": action.interlock_satisfied,
            "hash": action.hash,
            "action_timestamp": action.timestamp.isoformat()
        }

        return await self.broadcast(StreamType.CONTROL_ACTION, data)

    async def handle_client_message(
        self,
        websocket: WebSocket,
        message: str
    ) -> None:
        """
        Handle incoming message from a client.

        Supports:
        - ping/pong for health checks
        - subscription management

        Args:
            websocket: WebSocket that sent the message
            message: Raw message string
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type", "").lower()

            # Find connection info
            conn_info = None
            for info in self.active_connections.values():
                if info.websocket == websocket:
                    conn_info = info
                    break

            if not conn_info:
                return

            conn_info.last_activity = datetime.utcnow()

            if msg_type == "ping":
                # Respond with pong
                await self._send_message(
                    conn_info,
                    StreamType.PONG,
                    {"ping_timestamp": data.get("timestamp")}
                )

            elif msg_type == "subscribe":
                # Handle subscription request
                streams = data.get("streams", [])
                for stream in streams:
                    try:
                        stream_type = StreamType(stream)
                        conn_info.subscriptions.add(stream_type)
                    except ValueError:
                        pass

                await self._send_message(
                    conn_info,
                    StreamType.SUBSCRIPTION_ACK,
                    {"subscriptions": [s.value for s in conn_info.subscriptions]}
                )

            elif msg_type == "unsubscribe":
                # Handle unsubscription request
                streams = data.get("streams", [])
                for stream in streams:
                    try:
                        stream_type = StreamType(stream)
                        conn_info.subscriptions.discard(stream_type)
                    except ValueError:
                        pass

                await self._send_message(
                    conn_info,
                    StreamType.SUBSCRIPTION_ACK,
                    {"subscriptions": [s.value for s in conn_info.subscriptions]}
                )

        except json.JSONDecodeError:
            logger.debug("Received invalid JSON from client")
        except Exception as e:
            logger.debug(f"Error handling client message: {e}")

    async def _state_stream_loop(self) -> None:
        """Background task for streaming combustion state at 10Hz"""
        logger.info("Starting combustion state stream (10Hz)")

        while self._running:
            try:
                if self.active_connections:
                    await self.broadcast_combustion_state()

                await asyncio.sleep(self.STATE_STREAM_INTERVAL_MS / 1000.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"State stream error: {e}")
                await asyncio.sleep(0.1)

        logger.info("Combustion state stream stopped")

    async def _stability_stream_loop(self) -> None:
        """Background task for streaming stability metrics at 1Hz"""
        logger.info("Starting stability metrics stream (1Hz)")

        while self._running:
            try:
                if self.active_connections:
                    await self.broadcast_stability_metrics()

                await asyncio.sleep(self.STABILITY_STREAM_INTERVAL_MS / 1000.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Stability stream error: {e}")
                await asyncio.sleep(0.1)

        logger.info("Stability metrics stream stopped")

    async def _health_monitor_loop(self) -> None:
        """Background task for connection health monitoring"""
        logger.info(f"Starting health monitor (ping every {self.PING_INTERVAL_SECONDS}s)")

        while self._running:
            try:
                await asyncio.sleep(self.PING_INTERVAL_SECONDS)

                if not self.active_connections:
                    continue

                # Send ping to all connections
                stale_connections: List[str] = []
                current_time = datetime.utcnow()

                for connection_id, conn_info in self.active_connections.items():
                    # Check if connection is stale (no activity)
                    inactive_seconds = (current_time - conn_info.last_activity).total_seconds()

                    if inactive_seconds > self.PING_INTERVAL_SECONDS + self.PONG_TIMEOUT_SECONDS:
                        logger.info(
                            f"Connection {connection_id} timed out "
                            f"(inactive {inactive_seconds:.0f}s)"
                        )
                        stale_connections.append(connection_id)
                        continue

                    # Send ping
                    await self._send_message(
                        conn_info,
                        StreamType.PING,
                        {"server_time": current_time.isoformat()}
                    )

                # Disconnect stale connections
                for connection_id in stale_connections:
                    if connection_id in self.active_connections:
                        conn_info = self.active_connections[connection_id]
                        try:
                            await conn_info.websocket.close(
                                code=status.WS_1000_NORMAL_CLOSURE
                            )
                        except Exception:
                            pass
                        await self.disconnect(conn_info.websocket)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Health monitor error: {e}")

        logger.info("Health monitor stopped")

    async def start_streaming(self) -> None:
        """Start all background streaming tasks"""
        if self._running:
            return

        self._running = True

        # Start background tasks
        self._state_stream_task = asyncio.create_task(self._state_stream_loop())
        self._stability_stream_task = asyncio.create_task(self._stability_stream_loop())
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())

        logger.info("WebSocket streaming started")

    async def stop_streaming(self) -> None:
        """Stop all background streaming tasks and disconnect clients"""
        self._running = False

        # Cancel tasks
        if self._state_stream_task:
            self._state_stream_task.cancel()
            try:
                await self._state_stream_task
            except asyncio.CancelledError:
                pass

        if self._stability_stream_task:
            self._stability_stream_task.cancel()
            try:
                await self._stability_stream_task
            except asyncio.CancelledError:
                pass

        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        # Disconnect all clients
        for conn_info in list(self.active_connections.values()):
            try:
                await self._send_message(
                    conn_info,
                    StreamType.DISCONNECTED,
                    {"reason": "server_shutdown"}
                )
                await conn_info.websocket.close(code=status.WS_1001_GOING_AWAY)
            except Exception:
                pass

        self.active_connections.clear()
        self.connections_per_client.clear()

        logger.info("WebSocket streaming stopped")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections"""
        return {
            "total_connections": len(self.active_connections),
            "unique_clients": len(self.connections_per_client),
            "max_connections_per_client": self.MAX_CONNECTIONS_PER_CLIENT,
            "connections_by_client": dict(self.connections_per_client),
            "state_stream_hz": 1000 / self.STATE_STREAM_INTERVAL_MS,
            "stability_stream_hz": 1000 / self.STABILITY_STREAM_INTERVAL_MS,
            "global_sequence": self._global_sequence,
            "is_streaming": self._running
        }


# Global connection manager instance
ws_manager = WebSocketConnectionManager()


__all__ = [
    'ws_manager',
    'WebSocketConnectionManager',
    'StreamType',
    'WebSocketMessage',
    'ConnectionInfo'
]
