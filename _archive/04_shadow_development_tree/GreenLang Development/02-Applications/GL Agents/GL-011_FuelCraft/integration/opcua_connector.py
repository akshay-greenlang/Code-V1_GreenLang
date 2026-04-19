"""
GL-011 FUELCRAFT - OPC-UA Connector

Industrial OPC-UA connectivity for fuel system telemetry:
- TankLevel - Fuel tank level sensors
- Temperature - Storage temperature sensors
- FlowRate - Fuel flow meters
- Density - Fuel density measurements

Features:
- Secure authentication (certificate, username/password)
- Multiple security policies (Basic256Sha256, etc.)
- Automatic reconnection with exponential backoff
- Subscription-based data change notifications
- Tag mapping configuration
- Connection health monitoring
- Circuit breaker per IEC 61511

Tag Mapping:
- ns=2;s=FuelSystem.Tank.{tank_id}.Level
- ns=2;s=FuelSystem.Tank.{tank_id}.Temperature
- ns=2;s=FuelSystem.Tank.{tank_id}.FlowRate
- ns=2;s=FuelSystem.Tank.{tank_id}.Density
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import asyncio
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class SecurityPolicy(Enum):
    """OPC-UA security policies."""
    NONE = "None"
    BASIC128RSA15 = "Basic128Rsa15"
    BASIC256 = "Basic256"
    BASIC256SHA256 = "Basic256Sha256"
    AES128_SHA256_RSAOAEP = "Aes128_Sha256_RsaOaep"


class MessageSecurityMode(Enum):
    """OPC-UA message security modes."""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class DataQuality(Enum):
    """OPC-UA data quality indicators."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"


class ConnectionState(Enum):
    """OPC-UA connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class StatusCode(Enum):
    """OPC-UA status codes (subset)."""
    GOOD = 0x00000000
    UNCERTAIN = 0x40000000
    BAD = 0x80000000
    BAD_NODE_ID_UNKNOWN = 0x80340000
    BAD_CONNECTION_CLOSED = 0x80AC0000
    BAD_TIMEOUT = 0x800A0000


# =============================================================================
# Configuration
# =============================================================================

class OPCUAConfig(BaseModel):
    """OPC-UA connection configuration."""
    endpoint_url: str = Field(..., description="OPC-UA server endpoint")
    security_policy: SecurityPolicy = Field(SecurityPolicy.BASIC256SHA256)
    security_mode: MessageSecurityMode = Field(MessageSecurityMode.SIGN_AND_ENCRYPT)

    # Authentication
    username: Optional[str] = Field(None)
    password: Optional[str] = Field(None)  # From vault
    certificate_path: Optional[str] = Field(None)
    private_key_path: Optional[str] = Field(None)

    # Application identity
    application_uri: str = Field("urn:greenlang:gl011:fuelcraft")
    application_name: str = Field("GL-011 FUELCRAFT Fuel Optimizer")

    # Connection settings
    timeout_ms: int = Field(10000, description="Connection timeout")
    session_timeout_ms: int = Field(3600000, description="Session timeout (1 hour)")
    keepalive_interval_ms: int = Field(60000, description="Keepalive interval")

    # Reconnection settings
    auto_reconnect: bool = Field(True)
    max_reconnect_attempts: int = Field(10)
    reconnect_delay_ms: int = Field(5000)
    reconnect_backoff_factor: float = Field(1.5)
    max_reconnect_delay_ms: int = Field(300000)

    # Subscription defaults
    default_publishing_interval_ms: int = Field(1000)
    default_sampling_interval_ms: int = Field(500)
    default_queue_size: int = Field(10)


# =============================================================================
# Tag Models
# =============================================================================

@dataclass
class TelemetryReading:
    """Telemetry reading from OPC-UA server."""
    tag_id: str
    value: Any
    quality: DataQuality
    source_timestamp: datetime
    server_timestamp: datetime
    unit: Optional[str] = None

    def is_good(self) -> bool:
        """Check if reading quality is good."""
        return self.quality == DataQuality.GOOD

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag_id": self.tag_id,
            "value": self.value,
            "quality": self.quality.value,
            "source_timestamp": self.source_timestamp.isoformat(),
            "server_timestamp": self.server_timestamp.isoformat(),
            "unit": self.unit,
        }


class TankLevelTag(BaseModel):
    """Tank level tag configuration."""
    tank_id: str = Field(..., description="Tank identifier")
    node_id: str = Field(..., description="OPC-UA node ID")
    unit: str = Field("percent", description="Level unit")
    min_value: float = Field(0.0, description="Minimum level")
    max_value: float = Field(100.0, description="Maximum level")
    capacity_mmbtu: float = Field(..., description="Tank capacity in MMBtu")
    fuel_type: str = Field(..., description="Fuel type in tank")


class TemperatureTag(BaseModel):
    """Temperature tag configuration."""
    tank_id: str
    node_id: str
    unit: str = Field("celsius", description="Temperature unit")
    min_safe: float = Field(-40.0, description="Minimum safe temperature")
    max_safe: float = Field(60.0, description="Maximum safe temperature")


class FlowRateTag(BaseModel):
    """Flow rate tag configuration."""
    tank_id: str
    node_id: str
    unit: str = Field("mmbtu_per_hour", description="Flow rate unit")
    direction: str = Field("in", description="Flow direction: in, out, bidirectional")


class DensityTag(BaseModel):
    """Density tag configuration."""
    tank_id: str
    node_id: str
    unit: str = Field("kg_per_m3", description="Density unit")
    reference_temperature_c: float = Field(15.0, description="Reference temperature")


# =============================================================================
# Tag Mapping Configuration
# =============================================================================

class TagMappingConfig(BaseModel):
    """Configuration for OPC-UA tag mapping."""
    site_id: str = Field(..., description="Site identifier")
    namespace_index: int = Field(2, description="Default namespace index")

    # Tag patterns
    tank_level_pattern: str = Field(
        "ns={ns};s=FuelSystem.Tank.{tank_id}.Level",
        description="Tank level tag pattern"
    )
    temperature_pattern: str = Field(
        "ns={ns};s=FuelSystem.Tank.{tank_id}.Temperature",
        description="Temperature tag pattern"
    )
    flow_rate_pattern: str = Field(
        "ns={ns};s=FuelSystem.Tank.{tank_id}.FlowRate",
        description="Flow rate tag pattern"
    )
    density_pattern: str = Field(
        "ns={ns};s=FuelSystem.Tank.{tank_id}.Density",
        description="Density tag pattern"
    )

    # Tank configurations
    tanks: List[Dict[str, Any]] = Field(default=[], description="Tank configurations")

    def get_node_id(self, pattern: str, tank_id: str) -> str:
        """Generate node ID from pattern."""
        return pattern.format(ns=self.namespace_index, tank_id=tank_id)


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class OPCUACircuitBreaker:
    """Circuit breaker for OPC-UA connections per IEC 61511."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_ms: int = 30000,
        on_state_change: Optional[Callable[[CircuitBreakerState], None]] = None,
    ):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.reset_timeout_ms = reset_timeout_ms
        self._on_state_change = on_state_change

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._success_count_half_open = 0

    @property
    def state(self) -> CircuitBreakerState:
        """Get current state."""
        import time

        if self._state == CircuitBreakerState.OPEN and self._last_failure_time:
            elapsed = (time.time() - self._last_failure_time) * 1000
            if elapsed >= self.reset_timeout_ms:
                self._set_state(CircuitBreakerState.HALF_OPEN)

        return self._state

    def _set_state(self, new_state: CircuitBreakerState) -> None:
        """Set circuit breaker state."""
        old_state = self._state
        self._state = new_state

        if old_state != new_state:
            logger.info(f"OPC-UA circuit breaker: {old_state.value} -> {new_state.value}")
            if self._on_state_change:
                self._on_state_change(new_state)

    def allow_operation(self) -> bool:
        """Check if operation should be allowed."""
        current_state = self.state
        return current_state != CircuitBreakerState.OPEN

    def record_success(self) -> None:
        """Record successful operation."""
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._success_count_half_open += 1
            if self._success_count_half_open >= 3:
                self._set_state(CircuitBreakerState.CLOSED)
                self._failure_count = 0
                self._success_count_half_open = 0
        elif self._state == CircuitBreakerState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record failed operation."""
        import time

        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitBreakerState.HALF_OPEN:
            self._set_state(CircuitBreakerState.OPEN)
            self._success_count_half_open = 0
        elif self._state == CircuitBreakerState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._set_state(CircuitBreakerState.OPEN)


# =============================================================================
# OPC-UA Connector
# =============================================================================

# Type alias for data change callback
DataChangeCallback = Callable[[str, TelemetryReading], None]


class FuelCraftOPCUAConnector:
    """
    OPC-UA connector for fuel system telemetry.

    Provides secure, reliable connectivity to industrial control systems
    for real-time fuel inventory monitoring.

    Features:
    - Automatic reconnection with exponential backoff
    - Subscription-based data change notifications
    - Tag mapping configuration
    - Certificate-based security
    - Circuit breaker protection

    Example:
        config = OPCUAConfig(
            endpoint_url="opc.tcp://plc.site.company.com:4840",
            security_policy=SecurityPolicy.BASIC256SHA256,
            username="fuelcraft_service",
        )

        connector = FuelCraftOPCUAConnector(config)
        await connector.connect()

        # Subscribe to tank level tags
        await connector.subscribe_tank_levels(
            tank_ids=["TANK-001", "TANK-002"],
            callback=on_level_change,
        )
    """

    def __init__(
        self,
        config: OPCUAConfig,
        tag_mapping: Optional[TagMappingConfig] = None,
        vault_client: Optional[Any] = None,
        on_state_change: Optional[Callable[[ConnectionState], None]] = None,
    ) -> None:
        """
        Initialize OPC-UA connector.

        Args:
            config: Connection configuration
            tag_mapping: Tag mapping configuration
            vault_client: Optional vault client for credentials
            on_state_change: Callback for connection state changes
        """
        self.config = config
        self.tag_mapping = tag_mapping or TagMappingConfig(site_id="default")
        self.vault_client = vault_client
        self._on_state_change = on_state_change

        # Retrieve credentials from vault
        if vault_client and config.username:
            try:
                self.config.password = vault_client.get_secret(
                    f"opcua/{config.endpoint_url}/password"
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve credentials: {e}")

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._client = None
        self._session = None

        # Circuit breaker
        self._circuit_breaker = OPCUACircuitBreaker(
            failure_threshold=5,
            reset_timeout_ms=30000,
        )

        # Subscriptions
        self._subscriptions: Dict[str, Any] = {}
        self._callbacks: Dict[str, DataChangeCallback] = {}

        # Reconnection
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0
        self._stop_reconnect = False

        # Statistics
        self._stats = {
            "connects": 0,
            "disconnects": 0,
            "reconnects": 0,
            "reads": 0,
            "subscriptions": 0,
            "notifications": 0,
            "errors": 0,
        }

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info(f"FuelCraft OPC-UA connector initialized for {config.endpoint_url}")

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    def _set_state(self, new_state: ConnectionState) -> None:
        """Update connection state."""
        old_state = self._state
        self._state = new_state

        if old_state != new_state:
            logger.info(f"OPC-UA state: {old_state.value} -> {new_state.value}")
            if self._on_state_change:
                try:
                    self._on_state_change(new_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")

    async def connect(self) -> bool:
        """
        Connect to OPC-UA server.

        Returns:
            True if connection successful
        """
        # Check circuit breaker
        if not self._circuit_breaker.allow_operation():
            logger.warning("Circuit breaker open, connection rejected")
            return False

        async with self._lock:
            if self._state == ConnectionState.CONNECTED:
                return True

            self._set_state(ConnectionState.CONNECTING)
            self._stop_reconnect = False

            try:
                # In production, use asyncua:
                # from asyncua import Client
                # self._client = Client(self.config.endpoint_url)
                # await self._configure_security()
                # await self._client.connect()

                # Mock successful connection
                self._client = self._create_mock_client()

                self._set_state(ConnectionState.CONNECTED)
                self._stats["connects"] += 1
                self._reconnect_attempts = 0
                self._circuit_breaker.record_success()

                logger.info(f"Connected to OPC-UA server: {self.config.endpoint_url}")
                return True

            except Exception as e:
                self._stats["errors"] += 1
                self._circuit_breaker.record_failure()
                self._set_state(ConnectionState.ERROR)
                logger.error(f"OPC-UA connection failed: {e}")

                if self.config.auto_reconnect:
                    self._start_reconnect()

                return False

    def _create_mock_client(self) -> object:
        """Create mock client for demonstration."""
        class MockClient:
            def __init__(self):
                self.connected = True
            async def disconnect(self):
                self.connected = False
        return MockClient()

    async def disconnect(self) -> None:
        """Disconnect from OPC-UA server."""
        async with self._lock:
            self._stop_reconnect = True

            if self._reconnect_task:
                self._reconnect_task.cancel()
                try:
                    await self._reconnect_task
                except asyncio.CancelledError:
                    pass
                self._reconnect_task = None

            # Unsubscribe all
            self._subscriptions.clear()
            self._callbacks.clear()

            if self._client:
                try:
                    await self._client.disconnect()
                except Exception as e:
                    logger.warning(f"Error during disconnect: {e}")
                self._client = None

            self._set_state(ConnectionState.DISCONNECTED)
            self._stats["disconnects"] += 1
            logger.info("Disconnected from OPC-UA server")

    def _start_reconnect(self) -> None:
        """Start automatic reconnection."""
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Reconnection loop with exponential backoff."""
        self._set_state(ConnectionState.RECONNECTING)

        while not self._stop_reconnect:
            self._reconnect_attempts += 1

            if self._reconnect_attempts > self.config.max_reconnect_attempts:
                logger.error("Max reconnection attempts exceeded")
                self._set_state(ConnectionState.ERROR)
                return

            delay = min(
                self.config.reconnect_delay_ms * (self.config.reconnect_backoff_factor ** (self._reconnect_attempts - 1)),
                self.config.max_reconnect_delay_ms
            )

            logger.info(
                f"Reconnection attempt {self._reconnect_attempts}/{self.config.max_reconnect_attempts} "
                f"in {delay/1000:.1f}s"
            )

            await asyncio.sleep(delay / 1000)

            if self._stop_reconnect:
                return

            try:
                success = await self.connect()
                if success:
                    self._stats["reconnects"] += 1
                    await self._restore_subscriptions()
                    return
            except Exception as e:
                logger.warning(f"Reconnection attempt {self._reconnect_attempts} failed: {e}")

    async def _restore_subscriptions(self) -> None:
        """Restore subscriptions after reconnection."""
        logger.info(f"Restoring {len(self._subscriptions)} subscriptions")
        # Implementation would re-create subscriptions with stored callbacks

    async def subscribe_tank_levels(
        self,
        tank_ids: List[str],
        callback: DataChangeCallback,
        sampling_interval_ms: int = 500,
    ) -> str:
        """
        Subscribe to tank level tags.

        Args:
            tank_ids: List of tank identifiers
            callback: Callback for level changes
            sampling_interval_ms: Sampling interval

        Returns:
            Subscription ID
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to OPC-UA server")

        subscription_id = f"levels-{'-'.join(tank_ids)}"

        node_ids = [
            self.tag_mapping.get_node_id(
                self.tag_mapping.tank_level_pattern, tank_id
            )
            for tank_id in tank_ids
        ]

        # Store callback
        self._callbacks[subscription_id] = callback

        # In production, create actual subscription
        # subscription = await self._client.create_subscription(...)
        # await subscription.subscribe_data_change(nodes)

        self._subscriptions[subscription_id] = {
            "type": "tank_level",
            "tank_ids": tank_ids,
            "node_ids": node_ids,
            "sampling_interval_ms": sampling_interval_ms,
        }

        self._stats["subscriptions"] += 1
        logger.info(f"Subscribed to tank levels: {tank_ids}")

        return subscription_id

    async def subscribe_temperatures(
        self,
        tank_ids: List[str],
        callback: DataChangeCallback,
        sampling_interval_ms: int = 1000,
    ) -> str:
        """Subscribe to temperature tags."""
        if not self.is_connected:
            raise ConnectionError("Not connected")

        subscription_id = f"temps-{'-'.join(tank_ids)}"

        node_ids = [
            self.tag_mapping.get_node_id(
                self.tag_mapping.temperature_pattern, tank_id
            )
            for tank_id in tank_ids
        ]

        self._callbacks[subscription_id] = callback
        self._subscriptions[subscription_id] = {
            "type": "temperature",
            "tank_ids": tank_ids,
            "node_ids": node_ids,
            "sampling_interval_ms": sampling_interval_ms,
        }

        self._stats["subscriptions"] += 1
        logger.info(f"Subscribed to temperatures: {tank_ids}")

        return subscription_id

    async def subscribe_flow_rates(
        self,
        tank_ids: List[str],
        callback: DataChangeCallback,
        sampling_interval_ms: int = 500,
    ) -> str:
        """Subscribe to flow rate tags."""
        if not self.is_connected:
            raise ConnectionError("Not connected")

        subscription_id = f"flows-{'-'.join(tank_ids)}"

        node_ids = [
            self.tag_mapping.get_node_id(
                self.tag_mapping.flow_rate_pattern, tank_id
            )
            for tank_id in tank_ids
        ]

        self._callbacks[subscription_id] = callback
        self._subscriptions[subscription_id] = {
            "type": "flow_rate",
            "tank_ids": tank_ids,
            "node_ids": node_ids,
            "sampling_interval_ms": sampling_interval_ms,
        }

        self._stats["subscriptions"] += 1
        logger.info(f"Subscribed to flow rates: {tank_ids}")

        return subscription_id

    async def read_tank_level(self, tank_id: str) -> TelemetryReading:
        """Read current tank level."""
        if not self.is_connected:
            raise ConnectionError("Not connected")

        node_id = self.tag_mapping.get_node_id(
            self.tag_mapping.tank_level_pattern, tank_id
        )

        return await self._read_tag(node_id, "percent")

    async def read_temperature(self, tank_id: str) -> TelemetryReading:
        """Read current tank temperature."""
        if not self.is_connected:
            raise ConnectionError("Not connected")

        node_id = self.tag_mapping.get_node_id(
            self.tag_mapping.temperature_pattern, tank_id
        )

        return await self._read_tag(node_id, "celsius")

    async def read_all_tank_data(self, tank_id: str) -> Dict[str, TelemetryReading]:
        """Read all telemetry for a tank."""
        if not self.is_connected:
            raise ConnectionError("Not connected")

        return {
            "level": await self.read_tank_level(tank_id),
            "temperature": await self.read_temperature(tank_id),
        }

    async def _read_tag(self, node_id: str, unit: str) -> TelemetryReading:
        """Internal method to read a tag."""
        try:
            # In production:
            # node = self._client.get_node(node_id)
            # data_value = await node.read_data_value()
            # value = data_value.Value.Value
            # ...

            now = datetime.now(timezone.utc)

            # Simulated value for demonstration
            value = self._get_simulated_value(node_id)

            reading = TelemetryReading(
                tag_id=node_id,
                value=value,
                quality=DataQuality.GOOD,
                source_timestamp=now,
                server_timestamp=now,
                unit=unit,
            )

            self._stats["reads"] += 1
            self._circuit_breaker.record_success()

            return reading

        except Exception as e:
            logger.error(f"Error reading tag {node_id}: {e}")
            self._stats["errors"] += 1
            self._circuit_breaker.record_failure()

            return TelemetryReading(
                tag_id=node_id,
                value=None,
                quality=DataQuality.BAD,
                source_timestamp=datetime.now(timezone.utc),
                server_timestamp=datetime.now(timezone.utc),
                unit=unit,
            )

    def _get_simulated_value(self, node_id: str) -> float:
        """Get simulated value for demonstration."""
        import math
        import random

        base = hash(node_id) % 100
        variation = math.sin(datetime.now().timestamp() / 10) * 5

        if "Level" in node_id:
            return 50.0 + base * 0.4 + variation
        elif "Temperature" in node_id:
            return 20.0 + base * 0.1 + variation * 0.2
        elif "FlowRate" in node_id:
            return 100.0 + base * 2 + variation * 5
        elif "Density" in node_id:
            return 800.0 + base * 0.5 + variation * 0.1
        else:
            return float(base) + variation

    async def unsubscribe(self, subscription_id: str) -> None:
        """Remove subscription."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
        if subscription_id in self._callbacks:
            del self._callbacks[subscription_id]
        logger.info(f"Unsubscribed: {subscription_id}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            **self._stats,
            "state": self._state.value,
            "endpoint_url": self.config.endpoint_url,
            "active_subscriptions": len(self._subscriptions),
            "circuit_breaker_state": self._circuit_breaker.state.value,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy" if self.is_connected else "unhealthy",
            "state": self._state.value,
            "circuit_breaker": self._circuit_breaker.state.value,
            "last_read_count": self._stats["reads"],
            "error_count": self._stats["errors"],
        }


def create_fuel_system_tag_list(tank_ids: List[str], namespace: int = 2) -> Dict[str, List[str]]:
    """Create standard fuel system tag list."""
    tags = {
        "levels": [],
        "temperatures": [],
        "flow_rates": [],
        "densities": [],
    }

    for tank_id in tank_ids:
        tags["levels"].append(f"ns={namespace};s=FuelSystem.Tank.{tank_id}.Level")
        tags["temperatures"].append(f"ns={namespace};s=FuelSystem.Tank.{tank_id}.Temperature")
        tags["flow_rates"].append(f"ns={namespace};s=FuelSystem.Tank.{tank_id}.FlowRate")
        tags["densities"].append(f"ns={namespace};s=FuelSystem.Tank.{tank_id}.Density")

    return tags
