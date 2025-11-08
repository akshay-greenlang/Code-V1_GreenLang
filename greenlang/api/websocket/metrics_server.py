"""
WebSocket server for real-time metric streaming.

This module implements a FastAPI WebSocket server that streams metrics
from Redis pub/sub to connected clients with authentication, filtering,
compression, and rate limiting.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import msgpack
import redis.asyncio as aioredis
from fastapi import WebSocket, WebSocketDisconnect, status
from jose import JWTError, jwt
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)


class MetricSubscription(BaseModel):
    """Model for metric subscription requests."""

    channels: List[str] = Field(
        ...,
        description="List of metric channels to subscribe to",
        example=["system.metrics", "workflow.metrics"]
    )
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Tag filters for metrics"
    )
    aggregation_interval: Optional[str] = Field(
        "1s",
        description="Aggregation interval (1s, 5s, 1m, 5m, 1h)"
    )
    compression: bool = Field(
        True,
        description="Enable MessagePack compression"
    )

    @validator("aggregation_interval")
    def validate_interval(cls, v):
        """Validate aggregation interval."""
        valid_intervals = ["1s", "5s", "1m", "5m", "1h"]
        if v not in valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of: {valid_intervals}")
        return v

    @validator("channels")
    def validate_channels(cls, v):
        """Validate channel names."""
        valid_channels = [
            "system.metrics",
            "workflow.metrics",
            "agent.metrics",
            "distributed.metrics"
        ]
        for channel in v:
            if not any(channel.startswith(vc.split(".")[0]) for vc in valid_channels):
                raise ValueError(f"Invalid channel: {channel}")
        return v


class MetricFilter:
    """Filter metrics based on tags and patterns."""

    def __init__(self, tags: Optional[Dict[str, str]] = None):
        """Initialize metric filter.

        Args:
            tags: Tag filters to apply
        """
        self.tags = tags or {}

    def matches(self, metric: Dict[str, Any]) -> bool:
        """Check if metric matches filter criteria.

        Args:
            metric: Metric data to check

        Returns:
            True if metric matches, False otherwise
        """
        if not self.tags:
            return True

        metric_tags = metric.get("tags", {})
        for key, value in self.tags.items():
            if metric_tags.get(key) != value:
                return False

        return True


class MetricAggregator:
    """Aggregate metrics over time intervals."""

    def __init__(self, interval: str):
        """Initialize metric aggregator.

        Args:
            interval: Aggregation interval (1s, 5s, 1m, 5m, 1h)
        """
        self.interval = interval
        self.interval_seconds = self._parse_interval(interval)
        self.buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.last_flush = time.time()

    def _parse_interval(self, interval: str) -> int:
        """Parse interval string to seconds.

        Args:
            interval: Interval string (e.g., "5s", "1m")

        Returns:
            Interval in seconds
        """
        unit = interval[-1]
        value = int(interval[:-1])

        multipliers = {"s": 1, "m": 60, "h": 3600}
        return value * multipliers.get(unit, 1)

    def add_metric(self, metric: Dict[str, Any]) -> None:
        """Add metric to aggregation bucket.

        Args:
            metric: Metric data to add
        """
        metric_name = metric.get("name", "unknown")
        self.buckets[metric_name].append(metric)

    def should_flush(self) -> bool:
        """Check if it's time to flush aggregated metrics.

        Returns:
            True if should flush, False otherwise
        """
        return (time.time() - self.last_flush) >= self.interval_seconds

    def flush(self) -> List[Dict[str, Any]]:
        """Flush and aggregate metrics.

        Returns:
            List of aggregated metrics
        """
        aggregated = []

        for metric_name, metrics in self.buckets.items():
            if not metrics:
                continue

            # Aggregate based on metric type
            first_metric = metrics[0]
            metric_type = first_metric.get("type", "gauge")

            if metric_type == "counter":
                # Sum counters
                total = sum(m.get("value", 0) for m in metrics)
                aggregated.append({
                    "name": metric_name,
                    "type": "counter",
                    "value": total,
                    "timestamp": datetime.utcnow().isoformat(),
                    "tags": first_metric.get("tags", {}),
                    "count": len(metrics)
                })
            elif metric_type == "gauge":
                # Average gauges
                avg_value = sum(m.get("value", 0) for m in metrics) / len(metrics)
                min_value = min(m.get("value", 0) for m in metrics)
                max_value = max(m.get("value", 0) for m in metrics)

                aggregated.append({
                    "name": metric_name,
                    "type": "gauge",
                    "value": avg_value,
                    "min": min_value,
                    "max": max_value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "tags": first_metric.get("tags", {}),
                    "count": len(metrics)
                })
            elif metric_type == "histogram":
                # Merge histogram buckets
                all_values = []
                for m in metrics:
                    all_values.extend(m.get("values", []))

                all_values.sort()
                aggregated.append({
                    "name": metric_name,
                    "type": "histogram",
                    "values": all_values,
                    "timestamp": datetime.utcnow().isoformat(),
                    "tags": first_metric.get("tags", {}),
                    "count": len(all_values)
                })

        # Clear buckets
        self.buckets.clear()
        self.last_flush = time.time()

        return aggregated


class ClientConnection:
    """Represents a connected WebSocket client."""

    def __init__(
        self,
        websocket: WebSocket,
        client_id: str,
        user_id: Optional[str] = None
    ):
        """Initialize client connection.

        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
            user_id: Authenticated user ID
        """
        self.websocket = websocket
        self.client_id = client_id
        self.user_id = user_id
        self.subscriptions: Set[str] = set()
        self.filter = MetricFilter()
        self.aggregator: Optional[MetricAggregator] = None
        self.compression = True
        self.last_message_time = time.time()
        self.message_count = 0
        self.rate_limit_window = 60  # seconds
        self.rate_limit_max = 1000  # messages per window

    def subscribe(self, subscription: MetricSubscription) -> None:
        """Subscribe to metric channels.

        Args:
            subscription: Subscription configuration
        """
        self.subscriptions.update(subscription.channels)
        self.filter = MetricFilter(subscription.tags)
        self.compression = subscription.compression

        if subscription.aggregation_interval:
            self.aggregator = MetricAggregator(subscription.aggregation_interval)

    def unsubscribe(self, channels: List[str]) -> None:
        """Unsubscribe from metric channels.

        Args:
            channels: Channels to unsubscribe from
        """
        self.subscriptions.difference_update(channels)

    def check_rate_limit(self) -> bool:
        """Check if client is within rate limits.

        Returns:
            True if within limits, False if exceeded
        """
        current_time = time.time()

        # Reset counter if window has passed
        if (current_time - self.last_message_time) > self.rate_limit_window:
            self.message_count = 0
            self.last_message_time = current_time

        self.message_count += 1
        return self.message_count <= self.rate_limit_max

    async def send_metric(self, metric: Dict[str, Any]) -> bool:
        """Send metric to client.

        Args:
            metric: Metric data to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.filter.matches(metric):
            return False

        if not self.check_rate_limit():
            logger.warning(f"Rate limit exceeded for client {self.client_id}")
            return False

        try:
            if self.aggregator:
                self.aggregator.add_metric(metric)

                # Send aggregated metrics if interval passed
                if self.aggregator.should_flush():
                    aggregated = self.aggregator.flush()
                    for agg_metric in aggregated:
                        await self._send_data(agg_metric)
            else:
                await self._send_data(metric)

            return True
        except Exception as e:
            logger.error(f"Error sending metric to client {self.client_id}: {e}")
            return False

    async def _send_data(self, data: Dict[str, Any]) -> None:
        """Send data to WebSocket client.

        Args:
            data: Data to send
        """
        if self.compression:
            # Use MessagePack for compression
            packed_data = msgpack.packb(data, use_bin_type=True)
            await self.websocket.send_bytes(packed_data)
        else:
            # Send as JSON
            await self.websocket.send_json(data)

    async def send_error(self, error: str) -> None:
        """Send error message to client.

        Args:
            error: Error message
        """
        await self._send_data({"type": "error", "message": error})

    async def send_heartbeat(self) -> None:
        """Send heartbeat ping to client."""
        await self._send_data({"type": "ping", "timestamp": time.time()})


class MetricsWebSocketServer:
    """WebSocket server for real-time metric streaming."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        jwt_secret: str = "your-secret-key",
        jwt_algorithm: str = "HS256",
        heartbeat_interval: int = 30
    ):
        """Initialize WebSocket server.

        Args:
            redis_url: Redis connection URL
            jwt_secret: JWT secret key for authentication
            jwt_algorithm: JWT algorithm
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.redis_url = redis_url
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.heartbeat_interval = heartbeat_interval

        self.clients: Dict[str, ClientConnection] = {}
        self.redis_client: Optional[aioredis.Redis] = None
        self.pubsub: Optional[aioredis.client.PubSub] = None
        self.running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start WebSocket server and Redis connection."""
        logger.info("Starting metrics WebSocket server")

        # Connect to Redis
        self.redis_client = await aioredis.from_url(
            self.redis_url,
            decode_responses=False
        )
        self.pubsub = self.redis_client.pubsub()

        # Subscribe to all metric channels
        await self.pubsub.psubscribe("*.metrics")

        self.running = True

        # Start background tasks
        self._tasks.append(asyncio.create_task(self._process_redis_messages()))
        self._tasks.append(asyncio.create_task(self._send_heartbeats()))

        logger.info("Metrics WebSocket server started")

    async def stop(self) -> None:
        """Stop WebSocket server gracefully."""
        logger.info("Stopping metrics WebSocket server")

        self.running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Close all client connections
        for client in list(self.clients.values()):
            try:
                await client.websocket.close()
            except Exception as e:
                logger.error(f"Error closing client connection: {e}")

        self.clients.clear()

        # Close Redis connection
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()

        if self.redis_client:
            await self.redis_client.close()

        logger.info("Metrics WebSocket server stopped")

    def authenticate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Authenticate JWT token.

        Args:
            token: JWT token

        Returns:
            Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            return payload
        except JWTError as e:
            logger.warning(f"JWT authentication failed: {e}")
            return None

    async def handle_connection(
        self,
        websocket: WebSocket,
        token: Optional[str] = None
    ) -> None:
        """Handle new WebSocket connection.

        Args:
            websocket: WebSocket connection
            token: Optional JWT token for authentication
        """
        # Authenticate
        user_id = None
        if token:
            payload = self.authenticate_token(token)
            if not payload:
                await websocket.close(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="Invalid authentication token"
                )
                return
            user_id = payload.get("sub")

        await websocket.accept()

        # Create client connection
        client_id = f"{websocket.client.host}:{websocket.client.port}"
        client = ClientConnection(websocket, client_id, user_id)
        self.clients[client_id] = client

        logger.info(f"Client {client_id} connected (user: {user_id})")

        try:
            # Send welcome message
            await client._send_data({
                "type": "welcome",
                "client_id": client_id,
                "timestamp": time.time()
            })

            # Handle client messages
            await self._handle_client_messages(client)
        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Remove client
            if client_id in self.clients:
                del self.clients[client_id]

    async def _handle_client_messages(self, client: ClientConnection) -> None:
        """Handle messages from client.

        Args:
            client: Client connection
        """
        while True:
            try:
                # Receive message
                data = await client.websocket.receive_json()
                message_type = data.get("type")

                if message_type == "subscribe":
                    # Subscribe to channels
                    try:
                        subscription = MetricSubscription(**data.get("data", {}))
                        client.subscribe(subscription)
                        await client._send_data({
                            "type": "subscribed",
                            "channels": list(client.subscriptions)
                        })
                    except Exception as e:
                        await client.send_error(f"Subscription error: {e}")

                elif message_type == "unsubscribe":
                    # Unsubscribe from channels
                    channels = data.get("channels", [])
                    client.unsubscribe(channels)
                    await client._send_data({
                        "type": "unsubscribed",
                        "channels": channels
                    })

                elif message_type == "pong":
                    # Heartbeat response
                    pass

                elif message_type == "get_history":
                    # Request historical metrics
                    await self._send_historical_metrics(client, data.get("data", {}))

                else:
                    await client.send_error(f"Unknown message type: {message_type}")

            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.error(f"Error processing client message: {e}")
                await client.send_error(f"Processing error: {e}")

    async def _process_redis_messages(self) -> None:
        """Process messages from Redis pub/sub."""
        while self.running:
            try:
                message = await self.pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )

                if message and message["type"] == "pmessage":
                    channel = message["channel"].decode("utf-8")
                    data = message["data"]

                    # Decode metric data
                    try:
                        metric = msgpack.unpackb(data, raw=False)
                    except Exception:
                        metric = json.loads(data)

                    # Send to subscribed clients
                    await self._broadcast_metric(channel, metric)

            except Exception as e:
                logger.error(f"Error processing Redis message: {e}")
                await asyncio.sleep(1)

    async def _broadcast_metric(self, channel: str, metric: Dict[str, Any]) -> None:
        """Broadcast metric to subscribed clients.

        Args:
            channel: Channel name
            metric: Metric data
        """
        # Add channel to metric
        metric["channel"] = channel

        # Send to all subscribed clients
        disconnected_clients = []
        for client_id, client in self.clients.items():
            # Check if client is subscribed to this channel
            if not any(channel.startswith(sub.split("*")[0]) for sub in client.subscriptions):
                continue

            try:
                success = await client.send_metric(metric)
                if not success:
                    disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error sending metric to client {client_id}: {e}")
                disconnected_clients.append(client_id)

        # Remove disconnected clients
        for client_id in disconnected_clients:
            if client_id in self.clients:
                del self.clients[client_id]

    async def _send_heartbeats(self) -> None:
        """Send periodic heartbeats to all clients."""
        while self.running:
            await asyncio.sleep(self.heartbeat_interval)

            disconnected_clients = []
            for client_id, client in self.clients.items():
                try:
                    await client.send_heartbeat()
                except Exception as e:
                    logger.error(f"Error sending heartbeat to client {client_id}: {e}")
                    disconnected_clients.append(client_id)

            # Remove disconnected clients
            for client_id in disconnected_clients:
                if client_id in self.clients:
                    del self.clients[client_id]

    async def _send_historical_metrics(
        self,
        client: ClientConnection,
        request: Dict[str, Any]
    ) -> None:
        """Send historical metrics to client.

        Args:
            client: Client connection
            request: Historical metric request
        """
        try:
            channel = request.get("channel")
            start_time = request.get("start_time")
            end_time = request.get("end_time", datetime.utcnow().isoformat())

            if not channel or not start_time:
                await client.send_error("Missing required fields: channel, start_time")
                return

            # Query historical metrics from Redis
            # This would typically use Redis TimeSeries or similar
            # For now, send a placeholder response
            await client._send_data({
                "type": "history",
                "channel": channel,
                "metrics": [],
                "start_time": start_time,
                "end_time": end_time
            })

        except Exception as e:
            logger.error(f"Error fetching historical metrics: {e}")
            await client.send_error(f"Historical query error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics.

        Returns:
            Server statistics
        """
        total_subscriptions = sum(
            len(client.subscriptions) for client in self.clients.values()
        )

        return {
            "connected_clients": len(self.clients),
            "total_subscriptions": total_subscriptions,
            "uptime": time.time() - self._start_time if hasattr(self, "_start_time") else 0
        }
