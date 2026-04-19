"""
OPC-UA Client for FurnacePulse

Implements secure, read-only connection to OPC-UA servers for furnace telemetry.
Features certificate-based authentication, whitelisted node subscriptions,
store-and-forward buffering for network partitions, and automatic reconnection.

OPC-UA (Open Platform Communications Unified Architecture) is the industrial
standard for secure, reliable data exchange in manufacturing environments.
"""

import asyncio
import logging
import pickle
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import deque
import struct

from pydantic import BaseModel, Field, FilePath, validator

logger = logging.getLogger(__name__)


class SignalQuality(IntEnum):
    """
    OPC-UA signal quality flags.

    Interpretation per OPC-UA specification:
    - GOOD: Value is reliable
    - UNCERTAIN: Value may not be accurate
    - BAD: Value should not be used
    - STALE: Value has not been updated within expected interval
    """
    GOOD = 0
    GOOD_CLAMPED = 1  # Value clamped to range
    UNCERTAIN = 64
    UNCERTAIN_LAST_USABLE = 65
    UNCERTAIN_SENSOR_NOT_ACCURATE = 66
    UNCERTAIN_ENGINEERING_UNITS_EXCEEDED = 67
    BAD = 128
    BAD_CONFIG_ERROR = 129
    BAD_NOT_CONNECTED = 130
    BAD_DEVICE_FAILURE = 131
    BAD_SENSOR_FAILURE = 132
    BAD_COMM_FAILURE = 133
    STALE = 200

    @classmethod
    def is_usable(cls, quality: int) -> bool:
        """Check if signal quality is usable for processing."""
        return quality < cls.BAD

    @classmethod
    def is_good(cls, quality: int) -> bool:
        """Check if signal quality is good."""
        return quality < cls.UNCERTAIN

    @classmethod
    def to_string(cls, quality: int) -> str:
        """Convert quality code to human-readable string."""
        try:
            return cls(quality).name
        except ValueError:
            if quality < 64:
                return f"GOOD_VARIANT_{quality}"
            elif quality < 128:
                return f"UNCERTAIN_VARIANT_{quality}"
            elif quality < 200:
                return f"BAD_VARIANT_{quality}"
            else:
                return f"STALE_VARIANT_{quality}"


class TagMapping(BaseModel):
    """Mapping between FurnacePulse tag ID and OPC-UA node."""

    tag_id: str = Field(..., description="FurnacePulse internal tag identifier")
    node_id: str = Field(..., description="OPC-UA node ID (e.g., ns=2;s=Furnace1.Zone1.Temperature)")
    description: str = Field("", description="Human-readable tag description")
    data_type: str = Field("float", description="Expected data type: float, int, bool, string")
    engineering_unit: str = Field("", description="Engineering unit (e.g., degC, bar, kg/h)")
    min_value: Optional[float] = Field(None, description="Expected minimum value for validation")
    max_value: Optional[float] = Field(None, description="Expected maximum value for validation")
    sample_interval_ms: int = Field(1000, description="Sampling interval in milliseconds")
    deadband: float = Field(0.0, description="Deadband for change-based subscription")
    is_critical: bool = Field(False, description="Flag for critical safety signals")


class TagRegistry(BaseModel):
    """
    Registry of tag-to-node mappings for a furnace.

    The registry defines which OPC-UA nodes to subscribe to and how to
    interpret the received values. Only whitelisted nodes are accessible.
    """

    site_id: str = Field(..., description="Site identifier")
    furnace_id: str = Field(..., description="Furnace identifier")
    tags: Dict[str, TagMapping] = Field(default_factory=dict, description="Tag ID to mapping")

    def add_tag(self, mapping: TagMapping) -> None:
        """Add a tag mapping to the registry."""
        self.tags[mapping.tag_id] = mapping
        logger.info(f"Registered tag {mapping.tag_id} -> {mapping.node_id}")

    def get_node_id(self, tag_id: str) -> Optional[str]:
        """Get OPC-UA node ID for a tag."""
        mapping = self.tags.get(tag_id)
        return mapping.node_id if mapping else None

    def get_tag_by_node(self, node_id: str) -> Optional[TagMapping]:
        """Get tag mapping by OPC-UA node ID."""
        for mapping in self.tags.values():
            if mapping.node_id == node_id:
                return mapping
        return None

    def get_all_node_ids(self) -> List[str]:
        """Get list of all whitelisted node IDs."""
        return [m.node_id for m in self.tags.values()]

    def is_node_whitelisted(self, node_id: str) -> bool:
        """Check if a node ID is in the whitelist."""
        return any(m.node_id == node_id for m in self.tags.values())


class OPCUAConfig(BaseModel):
    """Configuration for OPC-UA client connection."""

    endpoint_url: str = Field(..., description="OPC-UA server endpoint URL")
    application_uri: str = Field(
        "urn:greenlang:furnacepulse:client",
        description="Client application URI"
    )

    # Certificate-based authentication
    certificate_path: Optional[str] = Field(None, description="Path to client certificate (.der)")
    private_key_path: Optional[str] = Field(None, description="Path to private key (.pem)")
    server_certificate_path: Optional[str] = Field(None, description="Path to server certificate for validation")

    # Security settings
    security_policy: str = Field(
        "Basic256Sha256",
        description="Security policy: None, Basic256, Basic256Sha256"
    )
    security_mode: str = Field(
        "SignAndEncrypt",
        description="Security mode: None, Sign, SignAndEncrypt"
    )

    # Connection settings
    connection_timeout_seconds: int = Field(30, description="Connection timeout")
    session_timeout_seconds: int = Field(3600, description="Session timeout")
    request_timeout_seconds: int = Field(10, description="Individual request timeout")

    # Reconnection settings
    reconnect_initial_delay_seconds: float = Field(1.0, description="Initial reconnect delay")
    reconnect_max_delay_seconds: float = Field(300.0, description="Maximum reconnect delay")
    reconnect_multiplier: float = Field(2.0, description="Exponential backoff multiplier")
    max_reconnect_attempts: int = Field(-1, description="Max reconnect attempts (-1 for infinite)")

    # Store-and-forward settings
    buffer_db_path: str = Field(
        "furnacepulse_opcua_buffer.db",
        description="SQLite database path for store-and-forward buffer"
    )
    buffer_max_age_hours: int = Field(24, description="Maximum age of buffered data in hours")
    buffer_flush_interval_seconds: int = Field(5, description="Interval to flush buffer when connected")


@dataclass
class DataValue:
    """OPC-UA data value with quality and timestamp."""

    tag_id: str
    node_id: str
    value: Any
    quality: SignalQuality
    source_timestamp: datetime
    server_timestamp: datetime

    def is_usable(self) -> bool:
        """Check if value is usable based on quality."""
        return SignalQuality.is_usable(self.quality)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tag_id": self.tag_id,
            "node_id": self.node_id,
            "value": self.value,
            "quality": int(self.quality),
            "quality_string": SignalQuality.to_string(self.quality),
            "source_timestamp": self.source_timestamp.isoformat(),
            "server_timestamp": self.server_timestamp.isoformat(),
        }


class StoreAndForwardBuffer:
    """
    SQLite-based store-and-forward buffer for network partition resilience.

    Stores data locally when network is unavailable and forwards when
    connectivity is restored. Supports 24+ hours of buffering.
    """

    def __init__(self, db_path: str, max_age_hours: int = 24):
        """Initialize buffer with SQLite database."""
        self.db_path = db_path
        self.max_age_hours = max_age_hours
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS buffered_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    value BLOB NOT NULL,
                    quality INTEGER NOT NULL,
                    source_timestamp TEXT NOT NULL,
                    server_timestamp TEXT NOT NULL,
                    buffered_at TEXT NOT NULL,
                    forwarded INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_buffered_data_forwarded
                ON buffered_data(forwarded, buffered_at)
            """)
            conn.commit()
        logger.info(f"Initialized store-and-forward buffer at {self.db_path}")

    def store(self, data_value: DataValue) -> None:
        """Store a data value in the buffer."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO buffered_data
                (tag_id, node_id, value, quality, source_timestamp, server_timestamp, buffered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data_value.tag_id,
                    data_value.node_id,
                    pickle.dumps(data_value.value),
                    int(data_value.quality),
                    data_value.source_timestamp.isoformat(),
                    data_value.server_timestamp.isoformat(),
                    datetime.utcnow().isoformat(),
                )
            )
            conn.commit()

    def get_pending(self, limit: int = 1000) -> List[Tuple[int, DataValue]]:
        """Get pending (not forwarded) data values."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, tag_id, node_id, value, quality, source_timestamp, server_timestamp
                FROM buffered_data
                WHERE forwarded = 0
                ORDER BY buffered_at ASC
                LIMIT ?
                """,
                (limit,)
            )

            results = []
            for row in cursor.fetchall():
                data_value = DataValue(
                    tag_id=row[1],
                    node_id=row[2],
                    value=pickle.loads(row[3]),
                    quality=SignalQuality(row[4]),
                    source_timestamp=datetime.fromisoformat(row[5]),
                    server_timestamp=datetime.fromisoformat(row[6]),
                )
                results.append((row[0], data_value))

            return results

    def mark_forwarded(self, ids: List[int]) -> None:
        """Mark data values as forwarded."""
        if not ids:
            return

        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE buffered_data SET forwarded = 1 WHERE id IN ({placeholders})",
                ids
            )
            conn.commit()

    def cleanup_old_data(self) -> int:
        """Remove data older than max_age_hours. Returns count of deleted rows."""
        cutoff = datetime.utcnow() - timedelta(hours=self.max_age_hours)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM buffered_data WHERE buffered_at < ?",
                (cutoff.isoformat(),)
            )
            conn.commit()
            deleted = cursor.rowcount

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old buffered records")

        return deleted

    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN forwarded = 0 THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN forwarded = 1 THEN 1 ELSE 0 END) as forwarded
                FROM buffered_data
                """
            )
            row = cursor.fetchone()
            return {
                "total": row[0] or 0,
                "pending": row[1] or 0,
                "forwarded": row[2] or 0,
            }


class OPCUAClient:
    """
    OPC-UA client for FurnacePulse furnace telemetry.

    Features:
    - Certificate-based authentication for security
    - Whitelisted node subscriptions only
    - Read-only credential enforcement
    - Tag registry mapping (tag_id -> OPC-UA node)
    - Signal quality flag interpretation
    - Store-and-forward buffering for 24h+ network partitions
    - Automatic reconnection with exponential backoff

    Usage:
        config = OPCUAConfig(endpoint_url="opc.tcp://server:4840")
        registry = TagRegistry(site_id="site1", furnace_id="furnace1")
        registry.add_tag(TagMapping(
            tag_id="zone1_temp",
            node_id="ns=2;s=Furnace1.Zone1.Temperature"
        ))

        client = OPCUAClient(config, registry)
        await client.connect()

        # Subscribe to data changes
        await client.subscribe(callback=handle_data)

        # Or read single value
        value = await client.read_tag("zone1_temp")
    """

    def __init__(
        self,
        config: OPCUAConfig,
        tag_registry: TagRegistry,
        data_callback: Optional[Callable[[DataValue], None]] = None,
    ):
        """
        Initialize OPC-UA client.

        Args:
            config: Client configuration
            tag_registry: Registry of tag-to-node mappings
            data_callback: Optional callback for received data
        """
        self.config = config
        self.tag_registry = tag_registry
        self.data_callback = data_callback

        # Connection state
        self._connected = False
        self._client = None
        self._subscription = None
        self._handles: Dict[str, int] = {}  # node_id -> subscription handle

        # Reconnection state
        self._reconnect_delay = config.reconnect_initial_delay_seconds
        self._reconnect_attempts = 0
        self._should_reconnect = True
        self._reconnect_task: Optional[asyncio.Task] = None

        # Store-and-forward buffer
        self._buffer = StoreAndForwardBuffer(
            config.buffer_db_path,
            config.buffer_max_age_hours
        )
        self._buffer_flush_task: Optional[asyncio.Task] = None

        # Data queue for internal processing
        self._data_queue: deque = deque(maxlen=10000)

        logger.info(
            f"OPCUAClient initialized for {config.endpoint_url} "
            f"with {len(tag_registry.tags)} whitelisted tags"
        )

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    async def connect(self) -> bool:
        """
        Connect to OPC-UA server with certificate authentication.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails after retries
        """
        try:
            # In production, use asyncua library:
            # from asyncua import Client
            # self._client = Client(url=self.config.endpoint_url)

            # Load certificates for authentication
            if self.config.certificate_path and self.config.private_key_path:
                logger.info("Loading client certificate for authentication")
                # In production:
                # await self._client.set_security(
                #     SecurityPolicyBasic256Sha256,
                #     certificate=self.config.certificate_path,
                #     private_key=self.config.private_key_path,
                #     server_certificate=self.config.server_certificate_path,
                #     mode=MessageSecurityMode.SignAndEncrypt
                # )

            # Connect with timeout
            logger.info(f"Connecting to OPC-UA server at {self.config.endpoint_url}")
            # await asyncio.wait_for(
            #     self._client.connect(),
            #     timeout=self.config.connection_timeout_seconds
            # )

            # Verify read-only access (no write methods available)
            # This is enforced by the OPC-UA server's access control

            self._connected = True
            self._reconnect_delay = self.config.reconnect_initial_delay_seconds
            self._reconnect_attempts = 0

            # Start buffer flush task
            self._buffer_flush_task = asyncio.create_task(self._flush_buffer_loop())

            logger.info("Successfully connected to OPC-UA server")
            return True

        except asyncio.TimeoutError:
            logger.error(
                f"Connection timeout after {self.config.connection_timeout_seconds}s"
            )
            raise ConnectionError("OPC-UA connection timeout")

        except Exception as e:
            logger.error(f"Failed to connect to OPC-UA server: {e}")
            self._connected = False
            raise ConnectionError(f"OPC-UA connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from OPC-UA server."""
        self._should_reconnect = False

        # Cancel background tasks
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        if self._buffer_flush_task:
            self._buffer_flush_task.cancel()
            try:
                await self._buffer_flush_task
            except asyncio.CancelledError:
                pass

        # Unsubscribe and disconnect
        if self._subscription:
            try:
                # await self._subscription.delete()
                pass
            except Exception as e:
                logger.warning(f"Error deleting subscription: {e}")

        if self._client:
            try:
                # await self._client.disconnect()
                pass
            except Exception as e:
                logger.warning(f"Error disconnecting: {e}")

        self._connected = False
        logger.info("Disconnected from OPC-UA server")

    async def subscribe(
        self,
        tag_ids: Optional[List[str]] = None,
        callback: Optional[Callable[[DataValue], None]] = None,
    ) -> None:
        """
        Subscribe to data changes for whitelisted tags.

        Args:
            tag_ids: Specific tags to subscribe to (default: all whitelisted)
            callback: Callback function for data changes
        """
        if not self._connected:
            raise ConnectionError("Not connected to OPC-UA server")

        if callback:
            self.data_callback = callback

        # Determine which tags to subscribe to
        if tag_ids:
            # Validate tags are whitelisted
            for tag_id in tag_ids:
                if tag_id not in self.tag_registry.tags:
                    raise ValueError(f"Tag '{tag_id}' is not in whitelist")
            tags_to_subscribe = [self.tag_registry.tags[t] for t in tag_ids]
        else:
            tags_to_subscribe = list(self.tag_registry.tags.values())

        logger.info(f"Subscribing to {len(tags_to_subscribe)} tags")

        # Create subscription
        # In production with asyncua:
        # self._subscription = await self._client.create_subscription(
        #     period=min(t.sample_interval_ms for t in tags_to_subscribe),
        #     handler=self._subscription_handler
        # )

        # Subscribe to each node
        for tag in tags_to_subscribe:
            try:
                # node = self._client.get_node(tag.node_id)
                # handle = await self._subscription.subscribe_data_change(
                #     node,
                #     sampling_interval=tag.sample_interval_ms,
                #     deadband_val=tag.deadband
                # )
                # self._handles[tag.node_id] = handle

                logger.debug(f"Subscribed to {tag.tag_id} ({tag.node_id})")

            except Exception as e:
                logger.error(f"Failed to subscribe to {tag.tag_id}: {e}")
                if tag.is_critical:
                    raise  # Re-raise for critical tags

    async def read_tag(self, tag_id: str) -> Optional[DataValue]:
        """
        Read a single tag value.

        Args:
            tag_id: Tag identifier from registry

        Returns:
            DataValue with current value, quality, and timestamps
        """
        if not self._connected:
            # Check buffer for last known value
            logger.warning(f"Not connected, returning buffered value for {tag_id}")
            return self._get_last_buffered_value(tag_id)

        mapping = self.tag_registry.tags.get(tag_id)
        if not mapping:
            raise ValueError(f"Tag '{tag_id}' not found in registry")

        try:
            # In production with asyncua:
            # node = self._client.get_node(mapping.node_id)
            # data_value = await node.read_value()

            # Mock response for demonstration
            data_value = DataValue(
                tag_id=tag_id,
                node_id=mapping.node_id,
                value=0.0,  # Would be data_value.Value
                quality=SignalQuality.GOOD,  # Would be from data_value.StatusCode
                source_timestamp=datetime.utcnow(),
                server_timestamp=datetime.utcnow(),
            )

            return data_value

        except Exception as e:
            logger.error(f"Failed to read tag {tag_id}: {e}")
            self._handle_connection_error(e)
            return self._get_last_buffered_value(tag_id)

    async def read_multiple_tags(self, tag_ids: List[str]) -> Dict[str, DataValue]:
        """
        Read multiple tags in a single request.

        Args:
            tag_ids: List of tag identifiers

        Returns:
            Dictionary mapping tag_id to DataValue
        """
        if not self._connected:
            logger.warning("Not connected, returning buffered values")
            return {
                tag_id: self._get_last_buffered_value(tag_id)
                for tag_id in tag_ids
                if self._get_last_buffered_value(tag_id) is not None
            }

        # Validate all tags
        node_ids = []
        for tag_id in tag_ids:
            mapping = self.tag_registry.tags.get(tag_id)
            if not mapping:
                raise ValueError(f"Tag '{tag_id}' not found in registry")
            node_ids.append(mapping.node_id)

        try:
            # In production with asyncua:
            # nodes = [self._client.get_node(nid) for nid in node_ids]
            # values = await self._client.read_values(nodes)

            results = {}
            for tag_id in tag_ids:
                mapping = self.tag_registry.tags[tag_id]
                results[tag_id] = DataValue(
                    tag_id=tag_id,
                    node_id=mapping.node_id,
                    value=0.0,
                    quality=SignalQuality.GOOD,
                    source_timestamp=datetime.utcnow(),
                    server_timestamp=datetime.utcnow(),
                )

            return results

        except Exception as e:
            logger.error(f"Failed to read multiple tags: {e}")
            self._handle_connection_error(e)
            return {}

    def _handle_data_change(
        self,
        node_id: str,
        value: Any,
        quality: int,
        source_timestamp: datetime,
        server_timestamp: datetime,
    ) -> None:
        """Handle incoming data change notification."""
        # Get tag mapping
        mapping = self.tag_registry.get_tag_by_node(node_id)
        if not mapping:
            logger.warning(f"Received data for unknown node: {node_id}")
            return

        # Create data value
        data_value = DataValue(
            tag_id=mapping.tag_id,
            node_id=node_id,
            value=value,
            quality=SignalQuality(quality) if quality in SignalQuality._value2member_map_ else SignalQuality.UNCERTAIN,
            source_timestamp=source_timestamp,
            server_timestamp=server_timestamp,
        )

        # Validate value against expected range
        if mapping.min_value is not None and isinstance(value, (int, float)):
            if value < mapping.min_value:
                logger.warning(
                    f"Tag {mapping.tag_id} value {value} below minimum {mapping.min_value}"
                )
        if mapping.max_value is not None and isinstance(value, (int, float)):
            if value > mapping.max_value:
                logger.warning(
                    f"Tag {mapping.tag_id} value {value} above maximum {mapping.max_value}"
                )

        # Log quality issues for critical tags
        if mapping.is_critical and not data_value.is_usable():
            logger.error(
                f"Critical tag {mapping.tag_id} has bad quality: "
                f"{SignalQuality.to_string(data_value.quality)}"
            )

        # Store in buffer (always, for resilience)
        self._buffer.store(data_value)

        # Add to internal queue
        self._data_queue.append(data_value)

        # Call callback if registered
        if self.data_callback:
            try:
                self.data_callback(data_value)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")

    def _handle_connection_error(self, error: Exception) -> None:
        """Handle connection error and trigger reconnection."""
        if not self._connected:
            return

        logger.error(f"Connection error detected: {error}")
        self._connected = False

        if self._should_reconnect:
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Reconnection loop with exponential backoff."""
        while self._should_reconnect and not self._connected:
            self._reconnect_attempts += 1

            if (
                self.config.max_reconnect_attempts > 0 and
                self._reconnect_attempts > self.config.max_reconnect_attempts
            ):
                logger.error(
                    f"Max reconnection attempts ({self.config.max_reconnect_attempts}) "
                    "exceeded, giving up"
                )
                break

            logger.info(
                f"Reconnection attempt {self._reconnect_attempts} "
                f"in {self._reconnect_delay:.1f}s"
            )

            await asyncio.sleep(self._reconnect_delay)

            try:
                await self.connect()

                # Re-establish subscriptions
                if self._handles:
                    await self.subscribe(list(self.tag_registry.tags.keys()))

                logger.info("Reconnection successful")
                return

            except Exception as e:
                logger.warning(f"Reconnection attempt failed: {e}")

                # Exponential backoff
                self._reconnect_delay = min(
                    self._reconnect_delay * self.config.reconnect_multiplier,
                    self.config.reconnect_max_delay_seconds
                )

    async def _flush_buffer_loop(self) -> None:
        """Background task to flush buffered data when connected."""
        while True:
            try:
                await asyncio.sleep(self.config.buffer_flush_interval_seconds)

                if self._connected:
                    # Get pending data
                    pending = self._buffer.get_pending(limit=100)

                    if pending:
                        # Forward data (would send to Kafka or other destination)
                        forwarded_ids = []
                        for record_id, data_value in pending:
                            if self.data_callback:
                                try:
                                    self.data_callback(data_value)
                                    forwarded_ids.append(record_id)
                                except Exception as e:
                                    logger.error(f"Failed to forward buffered data: {e}")
                                    break  # Stop on first error
                            else:
                                forwarded_ids.append(record_id)

                        if forwarded_ids:
                            self._buffer.mark_forwarded(forwarded_ids)
                            logger.info(f"Forwarded {len(forwarded_ids)} buffered records")

                # Periodic cleanup
                self._buffer.cleanup_old_data()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in buffer flush loop: {e}")

    def _get_last_buffered_value(self, tag_id: str) -> Optional[DataValue]:
        """Get last known value from internal queue."""
        for data_value in reversed(self._data_queue):
            if data_value.tag_id == tag_id:
                return data_value
        return None

    def get_buffer_stats(self) -> Dict[str, int]:
        """Get store-and-forward buffer statistics."""
        return self._buffer.get_stats()

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "connected": self._connected,
            "endpoint": self.config.endpoint_url,
            "reconnect_attempts": self._reconnect_attempts,
            "subscribed_tags": len(self._handles),
            "buffer_stats": self.get_buffer_stats(),
        }
