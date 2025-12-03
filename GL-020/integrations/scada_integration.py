"""
GL-020 ECONOPULSE - SCADA/DCS Integration Module

Enterprise-grade SCADA and DCS integration providing:
- OPC UA client for modern SCADA systems
- Modbus TCP client for legacy DCS
- Tag subscription for real-time data
- Economizer-specific tag groups
- Historical data retrieval
- Write-back capability for setpoints

Thread-safe with circuit breaker pattern for fault tolerance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import asyncio
import logging
import threading
import struct
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class SCADAProtocol(Enum):
    """Supported SCADA communication protocols."""
    OPC_UA = auto()
    MODBUS_TCP = auto()
    MODBUS_RTU = auto()
    DNP3 = auto()


class TagQuality(Enum):
    """OPC UA tag quality codes."""
    GOOD = "Good"
    GOOD_LOCAL_OVERRIDE = "GoodLocalOverride"
    UNCERTAIN = "Uncertain"
    UNCERTAIN_SENSOR_NOT_ACCURATE = "UncertainSensorNotAccurate"
    BAD = "Bad"
    BAD_NOT_CONNECTED = "BadNotConnected"
    BAD_SENSOR_FAILURE = "BadSensorFailure"
    BAD_COMMUNICATION_FAILURE = "BadCommunicationFailure"


class WriteStatus(Enum):
    """Status of write operations."""
    SUCCESS = "success"
    PENDING = "pending"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TagValue:
    """Represents a single tag value with metadata."""
    tag_name: str
    value: Any
    timestamp: datetime
    quality: TagQuality
    source_timestamp: Optional[datetime] = None
    status_code: int = 0
    unit: str = ""

    def is_good(self) -> bool:
        """Check if tag quality is good."""
        return self.quality in (TagQuality.GOOD, TagQuality.GOOD_LOCAL_OVERRIDE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag_name": self.tag_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "quality": self.quality.value,
            "source_timestamp": (
                self.source_timestamp.isoformat() if self.source_timestamp else None
            ),
            "status_code": self.status_code,
            "unit": self.unit,
        }


@dataclass
class TagSubscription:
    """Configuration for tag subscription."""
    tag_name: str
    node_id: str  # OPC UA node ID or Modbus address
    data_type: str = "Float"
    scan_rate_ms: int = 1000
    deadband: float = 0.0  # Percent change to trigger update
    callback: Optional[Callable[[TagValue], None]] = None
    enabled: bool = True

    def __hash__(self):
        return hash(self.tag_name)


@dataclass
class TagGroup:
    """Group of related tags for batch operations."""
    name: str
    description: str
    tags: List[TagSubscription] = field(default_factory=list)
    enabled: bool = True


@dataclass
class HistoricalDataRequest:
    """Request for historical data retrieval."""
    tag_names: List[str]
    start_time: datetime
    end_time: datetime
    interval_ms: int = 60000  # 1 minute default
    aggregate_type: str = "Average"  # Average, Min, Max, Count, etc.
    max_points: int = 10000


@dataclass
class HistoricalDataResult:
    """Result of historical data retrieval."""
    tag_name: str
    timestamps: List[datetime]
    values: List[float]
    qualities: List[TagQuality]
    aggregate_type: str


@dataclass
class SetpointWriteRequest:
    """Request to write a setpoint value."""
    tag_name: str
    value: Any
    node_id: str
    requires_confirmation: bool = True
    timeout_seconds: float = 10.0
    reason: str = ""
    operator_id: str = ""


@dataclass
class SetpointWriteResult:
    """Result of setpoint write operation."""
    tag_name: str
    status: WriteStatus
    timestamp: datetime
    error_message: Optional[str] = None
    confirmed_value: Optional[Any] = None


@dataclass
class SCADAConfig:
    """SCADA client configuration."""
    protocol: SCADAProtocol
    host: str
    port: int
    application_name: str = "GL-020-ECONOPULSE"
    security_mode: str = "None"  # None, Sign, SignAndEncrypt
    security_policy: str = ""  # Basic256Sha256, etc.
    username: Optional[str] = None
    password: Optional[str] = None  # Retrieved from vault
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    connection_timeout_seconds: float = 10.0
    session_timeout_seconds: float = 3600.0
    max_connections: int = 5


# =============================================================================
# Economizer Tag Groups
# =============================================================================

class EconomizerTagGroups:
    """
    Pre-defined tag groups for economizer monitoring.

    These tag groups follow ISA-95 naming conventions and cover
    all critical economizer measurements.
    """

    @staticmethod
    def feedwater_temperatures() -> TagGroup:
        """Feedwater temperature measurement tags."""
        return TagGroup(
            name="FeedwaterTemperatures",
            description="Economizer feedwater temperature measurements",
            tags=[
                TagSubscription(
                    tag_name="ECON.FW.TEMP.INLET",
                    node_id="ns=2;s=ECON.FeedwaterInletTemp",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.1,
                ),
                TagSubscription(
                    tag_name="ECON.FW.TEMP.OUTLET",
                    node_id="ns=2;s=ECON.FeedwaterOutletTemp",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.1,
                ),
                TagSubscription(
                    tag_name="ECON.FW.TEMP.INTERMEDIATE_1",
                    node_id="ns=2;s=ECON.FeedwaterIntermediateTemp1",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.1,
                ),
                TagSubscription(
                    tag_name="ECON.FW.TEMP.INTERMEDIATE_2",
                    node_id="ns=2;s=ECON.FeedwaterIntermediateTemp2",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.1,
                ),
            ],
        )

    @staticmethod
    def flue_gas_temperatures() -> TagGroup:
        """Flue gas temperature measurement tags."""
        return TagGroup(
            name="FlueGasTemperatures",
            description="Economizer flue gas temperature measurements",
            tags=[
                TagSubscription(
                    tag_name="ECON.FG.TEMP.INLET",
                    node_id="ns=2;s=ECON.FlueGasInletTemp",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.5,
                ),
                TagSubscription(
                    tag_name="ECON.FG.TEMP.OUTLET",
                    node_id="ns=2;s=ECON.FlueGasOutletTemp",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.5,
                ),
                TagSubscription(
                    tag_name="ECON.FG.TEMP.ZONE_A",
                    node_id="ns=2;s=ECON.FlueGasZoneATemp",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.5,
                ),
                TagSubscription(
                    tag_name="ECON.FG.TEMP.ZONE_B",
                    node_id="ns=2;s=ECON.FlueGasZoneBTemp",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.5,
                ),
            ],
        )

    @staticmethod
    def flow_rates() -> TagGroup:
        """Flow rate measurement tags."""
        return TagGroup(
            name="FlowRates",
            description="Economizer flow rate measurements",
            tags=[
                TagSubscription(
                    tag_name="ECON.FW.FLOW",
                    node_id="ns=2;s=ECON.FeedwaterFlow",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.5,
                ),
                TagSubscription(
                    tag_name="ECON.FW.FLOW.TOTALIZER",
                    node_id="ns=2;s=ECON.FeedwaterFlowTotal",
                    data_type="Double",
                    scan_rate_ms=5000,
                    deadband=0.0,
                ),
                TagSubscription(
                    tag_name="ECON.FG.FLOW",
                    node_id="ns=2;s=ECON.FlueGasFlow",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=1.0,
                ),
            ],
        )

    @staticmethod
    def differential_pressures() -> TagGroup:
        """Differential pressure measurement tags."""
        return TagGroup(
            name="DifferentialPressures",
            description="Economizer differential pressure measurements",
            tags=[
                TagSubscription(
                    tag_name="ECON.DP.GASSIDE",
                    node_id="ns=2;s=ECON.DPGasSide",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.01,
                ),
                TagSubscription(
                    tag_name="ECON.DP.WATERSIDE",
                    node_id="ns=2;s=ECON.DPWaterSide",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.01,
                ),
                TagSubscription(
                    tag_name="ECON.DP.TUBE_BANK_1",
                    node_id="ns=2;s=ECON.DPTubeBank1",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.01,
                ),
                TagSubscription(
                    tag_name="ECON.DP.TUBE_BANK_2",
                    node_id="ns=2;s=ECON.DPTubeBank2",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.01,
                ),
            ],
        )

    @staticmethod
    def soot_blower_status() -> TagGroup:
        """Soot blower status tags."""
        return TagGroup(
            name="SootBlowerStatus",
            description="Soot blower system status tags",
            tags=[
                TagSubscription(
                    tag_name="ECON.SB.STATUS",
                    node_id="ns=2;s=ECON.SootBlowerStatus",
                    data_type="Int16",
                    scan_rate_ms=500,
                    deadband=0.0,
                ),
                TagSubscription(
                    tag_name="ECON.SB.CURRENT_ZONE",
                    node_id="ns=2;s=ECON.SootBlowerCurrentZone",
                    data_type="Int16",
                    scan_rate_ms=500,
                    deadband=0.0,
                ),
                TagSubscription(
                    tag_name="ECON.SB.CYCLE_COUNT",
                    node_id="ns=2;s=ECON.SootBlowerCycleCount",
                    data_type="Int32",
                    scan_rate_ms=5000,
                    deadband=0.0,
                ),
                TagSubscription(
                    tag_name="ECON.SB.STEAM_PRESSURE",
                    node_id="ns=2;s=ECON.SootBlowerSteamPressure",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.5,
                ),
                TagSubscription(
                    tag_name="ECON.SB.STEAM_FLOW",
                    node_id="ns=2;s=ECON.SootBlowerSteamFlow",
                    data_type="Float",
                    scan_rate_ms=1000,
                    deadband=0.5,
                ),
            ],
        )

    @staticmethod
    def all_groups() -> List[TagGroup]:
        """Get all economizer tag groups."""
        return [
            EconomizerTagGroups.feedwater_temperatures(),
            EconomizerTagGroups.flue_gas_temperatures(),
            EconomizerTagGroups.flow_rates(),
            EconomizerTagGroups.differential_pressures(),
            EconomizerTagGroups.soot_blower_status(),
        ]


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._last_failure_time:
                    elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitBreakerState.HALF_OPEN
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN

    def is_available(self) -> bool:
        return self.state != CircuitBreakerState.OPEN


# =============================================================================
# SCADA Client
# =============================================================================

class SCADAClient:
    """
    Enterprise SCADA/DCS client supporting OPC UA and Modbus TCP.

    Provides:
    - Tag subscription with callbacks
    - Real-time data polling
    - Historical data retrieval
    - Setpoint write-back with confirmation
    - Connection pooling and retry logic
    - Thread-safe concurrent access
    """

    def __init__(self, config: SCADAConfig, vault_client=None):
        """
        Initialize SCADA client.

        Args:
            config: SCADA connection configuration
            vault_client: Optional vault client for credential retrieval
        """
        self.config = config
        self.vault_client = vault_client

        # Retrieve credentials from vault if available
        if vault_client and config.password is None:
            config.password = vault_client.get_secret(f"scada_{config.host}_password")

        self._connected = False
        self._lock = threading.RLock()
        self._subscriptions: Dict[str, TagSubscription] = {}
        self._tag_values: Dict[str, TagValue] = {}
        self._callbacks: Dict[str, List[Callable[[TagValue], None]]] = defaultdict(list)

        # Circuit breaker for fault tolerance
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
        )

        # Connection pool
        self._connection_pool: List[Any] = []
        self._pool_lock = threading.Lock()

        # Background tasks
        self._polling_task: Optional[asyncio.Task] = None
        self._running = False

        # Protocol-specific client
        self._opc_client = None
        self._modbus_client = None

        logger.info(
            f"Initialized SCADA client for {config.protocol.name} at {config.host}:{config.port}"
        )

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        with self._lock:
            return self._connected

    async def connect(self) -> bool:
        """
        Establish connection to SCADA system.

        Returns:
            True if connection successful, False otherwise.
        """
        if not self._circuit_breaker.is_available():
            logger.warning("SCADA connection blocked by circuit breaker")
            return False

        try:
            if self.config.protocol == SCADAProtocol.OPC_UA:
                await self._connect_opc_ua()
            elif self.config.protocol == SCADAProtocol.MODBUS_TCP:
                await self._connect_modbus_tcp()
            else:
                raise ValueError(f"Unsupported protocol: {self.config.protocol}")

            self._connected = True
            self._circuit_breaker.record_success()
            logger.info(f"Connected to SCADA at {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Failed to connect to SCADA: {e}")
            return False

    async def _connect_opc_ua(self) -> None:
        """Connect to OPC UA server."""
        from asyncua import Client
        from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256

        endpoint = f"opc.tcp://{self.config.host}:{self.config.port}"

        self._opc_client = Client(url=endpoint)
        self._opc_client.application_uri = f"urn:greenlang:{self.config.application_name}"

        # Configure security if required
        if self.config.security_mode != "None":
            await self._opc_client.set_security(
                SecurityPolicyBasic256Sha256,
                certificate=self.config.certificate_path,
                private_key=self.config.private_key_path,
            )

        # Set credentials if provided
        if self.config.username:
            self._opc_client.set_user(self.config.username)
            self._opc_client.set_password(self.config.password or "")

        # Connect with timeout
        await asyncio.wait_for(
            self._opc_client.connect(),
            timeout=self.config.connection_timeout_seconds,
        )

    async def _connect_modbus_tcp(self) -> None:
        """Connect to Modbus TCP server."""
        from pymodbus.client import AsyncModbusTcpClient

        self._modbus_client = AsyncModbusTcpClient(
            host=self.config.host,
            port=self.config.port,
            timeout=self.config.connection_timeout_seconds,
        )

        await self._modbus_client.connect()

    async def disconnect(self) -> None:
        """Disconnect from SCADA system."""
        try:
            self._running = False

            if self._polling_task:
                self._polling_task.cancel()
                try:
                    await self._polling_task
                except asyncio.CancelledError:
                    pass

            if self._opc_client:
                await self._opc_client.disconnect()
                self._opc_client = None

            if self._modbus_client:
                self._modbus_client.close()
                self._modbus_client = None

            self._connected = False
            logger.info("Disconnected from SCADA")

        except Exception as e:
            logger.error(f"Error disconnecting from SCADA: {e}")

    def subscribe(
        self,
        subscription: TagSubscription,
        callback: Optional[Callable[[TagValue], None]] = None,
    ) -> None:
        """
        Subscribe to a tag for real-time updates.

        Args:
            subscription: Tag subscription configuration
            callback: Optional callback function for value updates
        """
        with self._lock:
            self._subscriptions[subscription.tag_name] = subscription

            if callback:
                self._callbacks[subscription.tag_name].append(callback)

            if subscription.callback:
                self._callbacks[subscription.tag_name].append(subscription.callback)

            logger.info(f"Subscribed to tag: {subscription.tag_name}")

    def subscribe_group(
        self,
        tag_group: TagGroup,
        callback: Optional[Callable[[TagValue], None]] = None,
    ) -> None:
        """
        Subscribe to all tags in a tag group.

        Args:
            tag_group: Tag group to subscribe
            callback: Optional callback for all tags in group
        """
        for tag in tag_group.tags:
            self.subscribe(tag, callback)

        logger.info(f"Subscribed to tag group: {tag_group.name} ({len(tag_group.tags)} tags)")

    def unsubscribe(self, tag_name: str) -> None:
        """Unsubscribe from a tag."""
        with self._lock:
            if tag_name in self._subscriptions:
                del self._subscriptions[tag_name]
                self._callbacks.pop(tag_name, None)
                logger.info(f"Unsubscribed from tag: {tag_name}")

    async def start_polling(self) -> None:
        """Start background polling for subscribed tags."""
        self._running = True
        self._polling_task = asyncio.create_task(self._polling_loop())
        logger.info("Started SCADA polling")

    async def stop_polling(self) -> None:
        """Stop background polling."""
        self._running = False
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped SCADA polling")

    async def _polling_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            try:
                # Group subscriptions by scan rate
                scan_groups: Dict[int, List[TagSubscription]] = defaultdict(list)
                for sub in self._subscriptions.values():
                    scan_groups[sub.scan_rate_ms].append(sub)

                # Poll each group
                for scan_rate, subs in scan_groups.items():
                    await self._poll_tags(subs)

                # Wait minimum scan interval
                await asyncio.sleep(0.1)  # 100ms minimum

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(1.0)

    async def _poll_tags(self, subscriptions: List[TagSubscription]) -> None:
        """Poll a list of tags."""
        for sub in subscriptions:
            try:
                value = await self.read_tag(sub.tag_name)

                if value:
                    # Invoke callbacks
                    for callback in self._callbacks.get(sub.tag_name, []):
                        try:
                            callback(value)
                        except Exception as e:
                            logger.error(f"Callback error for {sub.tag_name}: {e}")

            except Exception as e:
                logger.warning(f"Failed to poll tag {sub.tag_name}: {e}")

    async def read_tag(self, tag_name: str) -> Optional[TagValue]:
        """
        Read a single tag value.

        Args:
            tag_name: Name of tag to read

        Returns:
            TagValue or None if read failed
        """
        if not self._circuit_breaker.is_available():
            return None

        subscription = self._subscriptions.get(tag_name)
        if not subscription:
            logger.warning(f"Tag not subscribed: {tag_name}")
            return None

        try:
            if self.config.protocol == SCADAProtocol.OPC_UA:
                value = await self._read_opc_ua_tag(subscription)
            elif self.config.protocol == SCADAProtocol.MODBUS_TCP:
                value = await self._read_modbus_tag(subscription)
            else:
                raise ValueError(f"Unsupported protocol: {self.config.protocol}")

            # Cache value
            with self._lock:
                self._tag_values[tag_name] = value

            self._circuit_breaker.record_success()
            return value

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error reading tag {tag_name}: {e}")
            return None

    async def _read_opc_ua_tag(self, subscription: TagSubscription) -> TagValue:
        """Read tag value from OPC UA server."""
        if not self._opc_client:
            raise ConnectionError("OPC UA client not connected")

        node = self._opc_client.get_node(subscription.node_id)
        data_value = await node.read_data_value()

        # Map OPC UA status to TagQuality
        quality = TagQuality.GOOD
        if not data_value.StatusCode.is_good():
            if data_value.StatusCode.is_bad():
                quality = TagQuality.BAD
            else:
                quality = TagQuality.UNCERTAIN

        return TagValue(
            tag_name=subscription.tag_name,
            value=data_value.Value.Value,
            timestamp=datetime.now(),
            quality=quality,
            source_timestamp=data_value.SourceTimestamp,
            status_code=data_value.StatusCode.value,
        )

    async def _read_modbus_tag(self, subscription: TagSubscription) -> TagValue:
        """Read tag value from Modbus TCP server."""
        if not self._modbus_client:
            raise ConnectionError("Modbus client not connected")

        # Parse node_id as register address (format: "HR:40001" or "IR:30001")
        parts = subscription.node_id.split(':')
        if len(parts) == 2:
            reg_type, address = parts[0], int(parts[1])
        else:
            reg_type, address = "HR", int(subscription.node_id)

        # Read registers based on data type
        if subscription.data_type == "Float":
            count = 2
        elif subscription.data_type == "Double":
            count = 4
        elif subscription.data_type in ("Int16", "UInt16"):
            count = 1
        elif subscription.data_type in ("Int32", "UInt32"):
            count = 2
        else:
            count = 1

        # Read appropriate register type
        if reg_type == "HR":
            result = await self._modbus_client.read_holding_registers(
                address=address,
                count=count,
                slave=1,
            )
        elif reg_type == "IR":
            result = await self._modbus_client.read_input_registers(
                address=address,
                count=count,
                slave=1,
            )
        else:
            raise ValueError(f"Unknown register type: {reg_type}")

        if result.isError():
            raise IOError(f"Modbus read error: {result}")

        # Convert registers to value
        value = self._convert_modbus_registers(
            result.registers,
            subscription.data_type,
        )

        return TagValue(
            tag_name=subscription.tag_name,
            value=value,
            timestamp=datetime.now(),
            quality=TagQuality.GOOD,
        )

    def _convert_modbus_registers(
        self, registers: List[int], data_type: str
    ) -> Any:
        """Convert Modbus registers to typed value."""
        if data_type == "Float":
            raw_bytes = struct.pack('>HH', registers[0], registers[1])
            return struct.unpack('>f', raw_bytes)[0]

        elif data_type == "Double":
            raw_bytes = struct.pack(
                '>HHHH',
                registers[0], registers[1], registers[2], registers[3],
            )
            return struct.unpack('>d', raw_bytes)[0]

        elif data_type == "Int16":
            return struct.unpack('>h', struct.pack('>H', registers[0]))[0]

        elif data_type == "UInt16":
            return registers[0]

        elif data_type == "Int32":
            raw_bytes = struct.pack('>HH', registers[0], registers[1])
            return struct.unpack('>i', raw_bytes)[0]

        elif data_type == "UInt32":
            raw_bytes = struct.pack('>HH', registers[0], registers[1])
            return struct.unpack('>I', raw_bytes)[0]

        else:
            return registers[0]

    async def read_multiple_tags(
        self, tag_names: List[str]
    ) -> Dict[str, TagValue]:
        """
        Read multiple tags in a single batch operation.

        Args:
            tag_names: List of tag names to read

        Returns:
            Dictionary mapping tag names to TagValue objects
        """
        results = {}

        # Use asyncio.gather for parallel reads
        tasks = [self.read_tag(name) for name in tag_names]
        values = await asyncio.gather(*tasks, return_exceptions=True)

        for name, value in zip(tag_names, values):
            if isinstance(value, Exception):
                logger.error(f"Error reading tag {name}: {value}")
                results[name] = TagValue(
                    tag_name=name,
                    value=None,
                    timestamp=datetime.now(),
                    quality=TagQuality.BAD_COMMUNICATION_FAILURE,
                )
            elif value:
                results[name] = value

        return results

    def get_cached_value(self, tag_name: str) -> Optional[TagValue]:
        """Get cached value for a tag."""
        with self._lock:
            return self._tag_values.get(tag_name)

    def get_all_cached_values(self) -> Dict[str, TagValue]:
        """Get all cached tag values."""
        with self._lock:
            return dict(self._tag_values)

    async def write_setpoint(
        self, request: SetpointWriteRequest
    ) -> SetpointWriteResult:
        """
        Write a setpoint value to the SCADA system.

        Args:
            request: Setpoint write request

        Returns:
            SetpointWriteResult with status information
        """
        if not self._circuit_breaker.is_available():
            return SetpointWriteResult(
                tag_name=request.tag_name,
                status=WriteStatus.FAILED,
                timestamp=datetime.now(),
                error_message="Circuit breaker open",
            )

        try:
            if self.config.protocol == SCADAProtocol.OPC_UA:
                result = await self._write_opc_ua_setpoint(request)
            elif self.config.protocol == SCADAProtocol.MODBUS_TCP:
                result = await self._write_modbus_setpoint(request)
            else:
                raise ValueError(f"Unsupported protocol: {self.config.protocol}")

            # Confirm write if required
            if request.requires_confirmation and result.status == WriteStatus.SUCCESS:
                result = await self._confirm_write(request, result)

            self._circuit_breaker.record_success()
            logger.info(
                f"Wrote setpoint {request.tag_name} = {request.value} "
                f"(status: {result.status.value})"
            )
            return result

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error writing setpoint {request.tag_name}: {e}")
            return SetpointWriteResult(
                tag_name=request.tag_name,
                status=WriteStatus.FAILED,
                timestamp=datetime.now(),
                error_message=str(e),
            )

    async def _write_opc_ua_setpoint(
        self, request: SetpointWriteRequest
    ) -> SetpointWriteResult:
        """Write setpoint via OPC UA."""
        if not self._opc_client:
            raise ConnectionError("OPC UA client not connected")

        from asyncua import ua

        node = self._opc_client.get_node(request.node_id)

        # Create data value
        data_value = ua.DataValue(ua.Variant(request.value))

        # Write with timeout
        await asyncio.wait_for(
            node.write_value(data_value),
            timeout=request.timeout_seconds,
        )

        return SetpointWriteResult(
            tag_name=request.tag_name,
            status=WriteStatus.SUCCESS,
            timestamp=datetime.now(),
        )

    async def _write_modbus_setpoint(
        self, request: SetpointWriteRequest
    ) -> SetpointWriteResult:
        """Write setpoint via Modbus TCP."""
        if not self._modbus_client:
            raise ConnectionError("Modbus client not connected")

        # Parse node_id as register address
        parts = request.node_id.split(':')
        if len(parts) == 2:
            address = int(parts[1])
        else:
            address = int(request.node_id)

        # Convert value to registers
        if isinstance(request.value, float):
            raw_bytes = struct.pack('>f', request.value)
            registers = list(struct.unpack('>HH', raw_bytes))
        elif isinstance(request.value, int):
            if -32768 <= request.value <= 32767:
                registers = [request.value & 0xFFFF]
            else:
                raw_bytes = struct.pack('>i', request.value)
                registers = list(struct.unpack('>HH', raw_bytes))
        else:
            registers = [int(request.value)]

        # Write registers
        result = await self._modbus_client.write_registers(
            address=address,
            values=registers,
            slave=1,
        )

        if result.isError():
            return SetpointWriteResult(
                tag_name=request.tag_name,
                status=WriteStatus.FAILED,
                timestamp=datetime.now(),
                error_message=str(result),
            )

        return SetpointWriteResult(
            tag_name=request.tag_name,
            status=WriteStatus.SUCCESS,
            timestamp=datetime.now(),
        )

    async def _confirm_write(
        self,
        request: SetpointWriteRequest,
        result: SetpointWriteResult,
    ) -> SetpointWriteResult:
        """Confirm that written value was applied."""
        await asyncio.sleep(0.5)  # Brief delay for value to propagate

        # Read back the value
        tag_value = await self.read_tag(request.tag_name)

        if tag_value:
            # Check if value matches
            tolerance = abs(request.value * 0.001) if request.value != 0 else 0.001
            if abs(tag_value.value - request.value) <= tolerance:
                result.confirmed_value = tag_value.value
                return result
            else:
                result.status = WriteStatus.FAILED
                result.error_message = (
                    f"Write verification failed: expected {request.value}, "
                    f"got {tag_value.value}"
                )

        return result

    async def get_historical_data(
        self, request: HistoricalDataRequest
    ) -> List[HistoricalDataResult]:
        """
        Retrieve historical data from SCADA historian.

        Args:
            request: Historical data request parameters

        Returns:
            List of HistoricalDataResult objects
        """
        if self.config.protocol != SCADAProtocol.OPC_UA:
            raise NotImplementedError(
                "Historical data retrieval only supported for OPC UA"
            )

        if not self._opc_client:
            raise ConnectionError("OPC UA client not connected")

        results = []

        for tag_name in request.tag_names:
            try:
                subscription = self._subscriptions.get(tag_name)
                if not subscription:
                    continue

                node = self._opc_client.get_node(subscription.node_id)

                # Read historical data
                history = await node.read_raw_history(
                    starttime=request.start_time,
                    endtime=request.end_time,
                    numvalues=request.max_points,
                )

                timestamps = []
                values = []
                qualities = []

                for data_value in history:
                    timestamps.append(data_value.SourceTimestamp or datetime.now())
                    values.append(float(data_value.Value.Value))
                    qualities.append(
                        TagQuality.GOOD
                        if data_value.StatusCode.is_good()
                        else TagQuality.BAD
                    )

                results.append(HistoricalDataResult(
                    tag_name=tag_name,
                    timestamps=timestamps,
                    values=values,
                    qualities=qualities,
                    aggregate_type=request.aggregate_type,
                ))

            except Exception as e:
                logger.error(f"Error retrieving history for {tag_name}: {e}")

        return results

    async def close(self) -> None:
        """Clean up resources and close connections."""
        await self.stop_polling()
        await self.disconnect()

    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status."""
        return {
            "connected": self._connected,
            "protocol": self.config.protocol.name,
            "host": self.config.host,
            "port": self.config.port,
            "circuit_breaker_state": self._circuit_breaker.state.name,
            "subscribed_tags": len(self._subscriptions),
            "cached_values": len(self._tag_values),
            "polling_active": self._running,
        }
