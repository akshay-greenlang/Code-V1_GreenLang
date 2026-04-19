"""
GL-017 CONDENSYNC SCADA Integration Module

OPC-UA client for condenser instrumentation with real-time data acquisition,
setpoint writing, and robust connection management.

Condenser Tag Points:
- CW_INLET_TEMP: Cooling water inlet temperature
- CW_OUTLET_TEMP: Cooling water outlet temperature
- CW_FLOW_RATE: Cooling water flow rate
- CW_PRESSURE: Cooling water pressure
- VACUUM_PRESSURE: Condenser vacuum pressure
- HOTWELL_LEVEL: Hotwell level
- CONDENSATE_FLOW: Condensate flow rate
- CONDENSATE_TEMP: Condensate temperature
- AIR_EJECTOR_STEAM: Air ejector steam flow
- AIR_LEAKAGE_RATE: Air in-leakage rate

Author: GreenLang AI Platform
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class SCADAConnectionError(Exception):
    """Raised when SCADA connection fails."""
    pass


class SCADAReadError(Exception):
    """Raised when reading from SCADA fails."""
    pass


class SCADAWriteError(Exception):
    """Raised when writing to SCADA fails."""
    pass


class SCADAValidationError(Exception):
    """Raised when data validation fails."""
    pass


# =============================================================================
# Enums and Constants
# =============================================================================

class TagQuality(Enum):
    """OPC-UA tag quality indicators."""
    GOOD = "good"
    BAD = "bad"
    UNCERTAIN = "uncertain"
    CONFIG_ERROR = "config_error"
    NOT_CONNECTED = "not_connected"
    DEVICE_FAILURE = "device_failure"
    SENSOR_FAILURE = "sensor_failure"
    LAST_KNOWN = "last_known"
    COMM_FAILURE = "comm_failure"
    OUT_OF_SERVICE = "out_of_service"


class SecurityMode(Enum):
    """OPC-UA security modes."""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class MessageSecurityMode(Enum):
    """OPC-UA message security modes."""
    NONE = 1
    SIGN = 2
    SIGN_AND_ENCRYPT = 3


class ConnectionState(Enum):
    """SCADA connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


# =============================================================================
# Data Models
# =============================================================================

class SCADAConfig(BaseModel):
    """Configuration for SCADA OPC-UA connection."""

    endpoint_url: str = Field(
        ...,
        description="OPC-UA server endpoint URL"
    )
    namespace_index: int = Field(
        default=2,
        description="OPC-UA namespace index for tags"
    )
    security_mode: SecurityMode = Field(
        default=SecurityMode.SIGN_AND_ENCRYPT,
        description="Security mode for OPC-UA connection"
    )
    security_policy: str = Field(
        default="Basic256Sha256",
        description="Security policy URI"
    )
    username: Optional[str] = Field(
        default=None,
        description="Username for authentication"
    )
    password: Optional[str] = Field(
        default=None,
        description="Password for authentication (from vault)"
    )
    certificate_path: Optional[str] = Field(
        default=None,
        description="Path to client certificate"
    )
    private_key_path: Optional[str] = Field(
        default=None,
        description="Path to client private key"
    )
    connection_timeout: float = Field(
        default=10.0,
        description="Connection timeout in seconds"
    )
    request_timeout: float = Field(
        default=5.0,
        description="Request timeout in seconds"
    )
    subscription_interval: int = Field(
        default=1000,
        description="Subscription publishing interval in milliseconds"
    )
    max_reconnect_attempts: int = Field(
        default=5,
        description="Maximum reconnection attempts"
    )
    reconnect_delay_base: float = Field(
        default=1.0,
        description="Base delay for exponential backoff in seconds"
    )
    reconnect_delay_max: float = Field(
        default=60.0,
        description="Maximum reconnection delay in seconds"
    )
    keepalive_interval: int = Field(
        default=5000,
        description="Keepalive interval in milliseconds"
    )
    application_name: str = Field(
        default="GL-017-CONDENSYNC",
        description="Application name for OPC-UA client"
    )

    @validator("endpoint_url")
    def validate_endpoint(cls, v):
        if not v.startswith("opc.tcp://"):
            raise ValueError("Endpoint URL must start with 'opc.tcp://'")
        return v


@dataclass
class CondenserTagMapping:
    """Tag mapping for condenser instrumentation points."""

    # Cooling Water Tags
    CW_INLET_TEMP: str = "Condenser.CoolingWater.InletTemp"
    CW_OUTLET_TEMP: str = "Condenser.CoolingWater.OutletTemp"
    CW_FLOW_RATE: str = "Condenser.CoolingWater.FlowRate"
    CW_PRESSURE: str = "Condenser.CoolingWater.Pressure"
    CW_DIFFERENTIAL_PRESSURE: str = "Condenser.CoolingWater.DifferentialPressure"

    # Vacuum System Tags
    VACUUM_PRESSURE: str = "Condenser.Vacuum.Pressure"
    VACUUM_PRESSURE_A: str = "Condenser.Vacuum.PressureA"
    VACUUM_PRESSURE_B: str = "Condenser.Vacuum.PressureB"
    SATURATION_TEMP: str = "Condenser.Vacuum.SaturationTemp"

    # Hotwell Tags
    HOTWELL_LEVEL: str = "Condenser.Hotwell.Level"
    HOTWELL_TEMP: str = "Condenser.Hotwell.Temperature"
    HOTWELL_LEVEL_SETPOINT: str = "Condenser.Hotwell.LevelSetpoint"

    # Condensate Tags
    CONDENSATE_FLOW: str = "Condenser.Condensate.FlowRate"
    CONDENSATE_TEMP: str = "Condenser.Condensate.Temperature"
    CONDENSATE_PUMP_A_STATUS: str = "Condenser.Condensate.PumpA.Status"
    CONDENSATE_PUMP_B_STATUS: str = "Condenser.Condensate.PumpB.Status"

    # Air Removal Tags
    AIR_EJECTOR_STEAM: str = "Condenser.AirRemoval.EjectorSteamFlow"
    AIR_LEAKAGE_RATE: str = "Condenser.AirRemoval.LeakageRate"
    VACUUM_PUMP_A_STATUS: str = "Condenser.AirRemoval.VacuumPumpA.Status"
    VACUUM_PUMP_B_STATUS: str = "Condenser.AirRemoval.VacuumPumpB.Status"

    # Performance Tags
    TTD: str = "Condenser.Performance.TTD"  # Terminal Temperature Difference
    CLEANLINESS_FACTOR: str = "Condenser.Performance.CleanlinessFactor"
    HEAT_TRANSFER_RATE: str = "Condenser.Performance.HeatTransferRate"
    DUTY: str = "Condenser.Performance.Duty"

    # Setpoint Tags (Writable)
    CW_FLOW_SETPOINT: str = "Condenser.Setpoints.CWFlowSetpoint"
    VACUUM_SETPOINT: str = "Condenser.Setpoints.VacuumSetpoint"
    MAKEUP_VALVE_POSITION: str = "Condenser.Setpoints.MakeupValvePosition"

    def get_all_read_tags(self) -> List[str]:
        """Get all read-only tag names."""
        return [
            self.CW_INLET_TEMP, self.CW_OUTLET_TEMP, self.CW_FLOW_RATE,
            self.CW_PRESSURE, self.CW_DIFFERENTIAL_PRESSURE,
            self.VACUUM_PRESSURE, self.VACUUM_PRESSURE_A, self.VACUUM_PRESSURE_B,
            self.SATURATION_TEMP, self.HOTWELL_LEVEL, self.HOTWELL_TEMP,
            self.CONDENSATE_FLOW, self.CONDENSATE_TEMP,
            self.CONDENSATE_PUMP_A_STATUS, self.CONDENSATE_PUMP_B_STATUS,
            self.AIR_EJECTOR_STEAM, self.AIR_LEAKAGE_RATE,
            self.VACUUM_PUMP_A_STATUS, self.VACUUM_PUMP_B_STATUS,
            self.TTD, self.CLEANLINESS_FACTOR, self.HEAT_TRANSFER_RATE, self.DUTY
        ]

    def get_writable_tags(self) -> List[str]:
        """Get all writable tag names."""
        return [
            self.CW_FLOW_SETPOINT, self.VACUUM_SETPOINT,
            self.MAKEUP_VALVE_POSITION, self.HOTWELL_LEVEL_SETPOINT
        ]


@dataclass
class TagValue:
    """Represents a tag value with metadata."""

    tag_name: str
    value: Any
    quality: TagQuality
    timestamp: datetime
    source_timestamp: Optional[datetime] = None
    status_code: int = 0
    engineering_units: Optional[str] = None

    def is_good(self) -> bool:
        """Check if tag quality is good."""
        return self.quality == TagQuality.GOOD

    def is_stale(self, max_age_seconds: float = 60.0) -> bool:
        """Check if tag value is stale."""
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age > max_age_seconds


@dataclass
class SetpointCommand:
    """Command to write a setpoint value."""

    tag_name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 1
    source: str = "GL-017-CONDENSYNC"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag_name": self.tag_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "command_id": self.command_id,
            "priority": self.priority,
            "source": self.source
        }


class TagValidationRule(BaseModel):
    """Validation rule for a tag."""

    tag_name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    rate_of_change_limit: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = True


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================

class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"
        self._lock = asyncio.Lock()

    async def can_execute(self) -> bool:
        """Check if request can be executed."""
        async with self._lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if self.last_failure_time:
                    elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self.state = "half_open"
                        self.success_count = 0
                        return True
                return False
            else:  # half_open
                return self.success_count < self.half_open_requests

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            if self.state == "half_open":
                self.success_count += 1
                if self.success_count >= self.half_open_requests:
                    self.state = "closed"
                    self.failure_count = 0
            else:
                self.failure_count = 0

    async def record_failure(self) -> None:
        """Record a failed request."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()

            if self.state == "half_open":
                self.state = "open"
            elif self.failure_count >= self.failure_threshold:
                self.state = "open"


# =============================================================================
# SCADA Integration Class
# =============================================================================

class SCADAIntegration:
    """
    OPC-UA client for condenser instrumentation.

    Provides:
    - Secure OPC-UA connection with authentication
    - Real-time data acquisition via subscriptions
    - Setpoint writing with validation
    - Automatic reconnection with exponential backoff
    - Circuit breaker for fault tolerance
    - Data validation and quality checking
    """

    def __init__(self, config: SCADAConfig):
        """
        Initialize SCADA integration.

        Args:
            config: SCADA configuration
        """
        self.config = config
        self.tag_mapping = CondenserTagMapping()

        self._client = None
        self._subscription = None
        self._connection_state = ConnectionState.DISCONNECTED
        self._reconnect_task: Optional[asyncio.Task] = None
        self._subscription_handles: Dict[str, Any] = {}

        # Data storage
        self._current_values: Dict[str, TagValue] = {}
        self._value_callbacks: List[Callable[[TagValue], None]] = []

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker()

        # Validation rules
        self._validation_rules: Dict[str, TagValidationRule] = {}
        self._setup_default_validation_rules()

        # Statistics
        self._stats = {
            "reads_total": 0,
            "reads_success": 0,
            "reads_failed": 0,
            "writes_total": 0,
            "writes_success": 0,
            "writes_failed": 0,
            "reconnections": 0,
            "last_successful_read": None,
            "last_successful_write": None
        }

        logger.info(f"SCADA Integration initialized for endpoint: {config.endpoint_url}")

    def _setup_default_validation_rules(self) -> None:
        """Setup default validation rules for condenser tags."""
        rules = [
            TagValidationRule(
                tag_name=self.tag_mapping.CW_INLET_TEMP,
                min_value=5.0, max_value=45.0,
                rate_of_change_limit=5.0
            ),
            TagValidationRule(
                tag_name=self.tag_mapping.CW_OUTLET_TEMP,
                min_value=10.0, max_value=60.0,
                rate_of_change_limit=5.0
            ),
            TagValidationRule(
                tag_name=self.tag_mapping.CW_FLOW_RATE,
                min_value=0.0, max_value=100000.0,
                rate_of_change_limit=1000.0
            ),
            TagValidationRule(
                tag_name=self.tag_mapping.VACUUM_PRESSURE,
                min_value=0.0, max_value=101.325,
                rate_of_change_limit=10.0
            ),
            TagValidationRule(
                tag_name=self.tag_mapping.HOTWELL_LEVEL,
                min_value=0.0, max_value=100.0,
                rate_of_change_limit=10.0
            ),
            TagValidationRule(
                tag_name=self.tag_mapping.CONDENSATE_FLOW,
                min_value=0.0, max_value=10000.0,
                rate_of_change_limit=500.0
            ),
            TagValidationRule(
                tag_name=self.tag_mapping.CONDENSATE_TEMP,
                min_value=20.0, max_value=100.0,
                rate_of_change_limit=5.0
            ),
            TagValidationRule(
                tag_name=self.tag_mapping.AIR_EJECTOR_STEAM,
                min_value=0.0, max_value=1000.0,
                rate_of_change_limit=100.0
            ),
            TagValidationRule(
                tag_name=self.tag_mapping.AIR_LEAKAGE_RATE,
                min_value=0.0, max_value=100.0,
                rate_of_change_limit=10.0
            ),
        ]

        for rule in rules:
            self._validation_rules[rule.tag_name] = rule

    @property
    def connection_state(self) -> ConnectionState:
        """Get current connection state."""
        return self._connection_state

    @property
    def is_connected(self) -> bool:
        """Check if connected to SCADA."""
        return self._connection_state == ConnectionState.CONNECTED

    async def connect(self) -> None:
        """
        Establish connection to OPC-UA server.

        Raises:
            SCADAConnectionError: If connection fails
        """
        if self._connection_state == ConnectionState.CONNECTED:
            logger.warning("Already connected to SCADA")
            return

        self._connection_state = ConnectionState.CONNECTING
        logger.info(f"Connecting to SCADA at {self.config.endpoint_url}")

        try:
            # Initialize OPC-UA client (simulated for implementation)
            await self._create_client()
            await self._authenticate()
            await self._setup_subscription()

            self._connection_state = ConnectionState.CONNECTED
            logger.info("Successfully connected to SCADA")

        except Exception as e:
            self._connection_state = ConnectionState.ERROR
            logger.error(f"Failed to connect to SCADA: {e}")
            raise SCADAConnectionError(f"Connection failed: {e}")

    async def _create_client(self) -> None:
        """Create OPC-UA client instance."""
        # In production, this would use asyncua or opcua libraries
        # For now, we simulate the client creation
        logger.debug("Creating OPC-UA client")

        # Simulated client configuration
        self._client = {
            "endpoint": self.config.endpoint_url,
            "security_mode": self.config.security_mode.value,
            "security_policy": self.config.security_policy,
            "application_name": self.config.application_name,
            "connected": False
        }

        await asyncio.sleep(0.1)  # Simulate connection delay
        self._client["connected"] = True

    async def _authenticate(self) -> None:
        """Authenticate with OPC-UA server."""
        logger.debug("Authenticating with SCADA")

        if self.config.username and self.config.password:
            # Username/password authentication
            logger.debug("Using username/password authentication")
        elif self.config.certificate_path:
            # Certificate-based authentication
            logger.debug("Using certificate-based authentication")
        else:
            # Anonymous authentication
            logger.debug("Using anonymous authentication")

        await asyncio.sleep(0.05)  # Simulate auth delay

    async def _setup_subscription(self) -> None:
        """Setup OPC-UA subscription for real-time data."""
        logger.debug("Setting up SCADA subscription")

        self._subscription = {
            "id": str(uuid.uuid4()),
            "interval": self.config.subscription_interval,
            "active": True,
            "monitored_items": []
        }

        # Subscribe to all read tags
        for tag in self.tag_mapping.get_all_read_tags():
            await self._add_monitored_item(tag)

    async def _add_monitored_item(self, tag_name: str) -> None:
        """Add a tag to the subscription."""
        if self._subscription:
            self._subscription["monitored_items"].append(tag_name)
            self._subscription_handles[tag_name] = str(uuid.uuid4())
            logger.debug(f"Added monitored item: {tag_name}")

    async def disconnect(self) -> None:
        """Disconnect from OPC-UA server."""
        logger.info("Disconnecting from SCADA")

        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        if self._subscription:
            self._subscription["active"] = False
            self._subscription = None

        if self._client:
            self._client["connected"] = False
            self._client = None

        self._connection_state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from SCADA")

    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        self._connection_state = ConnectionState.RECONNECTING

        for attempt in range(self.config.max_reconnect_attempts):
            delay = min(
                self.config.reconnect_delay_base * (2 ** attempt),
                self.config.reconnect_delay_max
            )

            logger.info(
                f"Reconnection attempt {attempt + 1}/{self.config.max_reconnect_attempts} "
                f"in {delay:.1f}s"
            )

            await asyncio.sleep(delay)

            try:
                await self._create_client()
                await self._authenticate()
                await self._setup_subscription()

                self._connection_state = ConnectionState.CONNECTED
                self._stats["reconnections"] += 1
                logger.info("Successfully reconnected to SCADA")
                return

            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")

        self._connection_state = ConnectionState.ERROR
        logger.error("All reconnection attempts failed")

    async def read_tag(self, tag_name: str) -> TagValue:
        """
        Read a single tag value.

        Args:
            tag_name: Name of the tag to read

        Returns:
            TagValue with current value and metadata

        Raises:
            SCADAReadError: If read fails
        """
        if not await self._circuit_breaker.can_execute():
            raise SCADAReadError("Circuit breaker is open")

        self._stats["reads_total"] += 1

        try:
            if not self.is_connected:
                raise SCADAReadError("Not connected to SCADA")

            # Simulated read operation
            # In production, this would call self._client.read_value(node_id)
            value = await self._simulate_read(tag_name)

            # Validate the value
            validated_value = self._validate_tag_value(value)

            # Store the value
            self._current_values[tag_name] = validated_value

            # Update statistics
            self._stats["reads_success"] += 1
            self._stats["last_successful_read"] = datetime.utcnow()

            await self._circuit_breaker.record_success()

            return validated_value

        except Exception as e:
            self._stats["reads_failed"] += 1
            await self._circuit_breaker.record_failure()
            logger.error(f"Failed to read tag {tag_name}: {e}")
            raise SCADAReadError(f"Read failed: {e}")

    async def read_tags(self, tag_names: List[str]) -> Dict[str, TagValue]:
        """
        Read multiple tag values.

        Args:
            tag_names: List of tag names to read

        Returns:
            Dictionary mapping tag names to TagValues
        """
        results = {}

        # Batch read for efficiency
        read_tasks = [self.read_tag(tag) for tag in tag_names]
        tag_values = await asyncio.gather(*read_tasks, return_exceptions=True)

        for tag_name, result in zip(tag_names, tag_values):
            if isinstance(result, Exception):
                logger.warning(f"Failed to read {tag_name}: {result}")
                results[tag_name] = TagValue(
                    tag_name=tag_name,
                    value=None,
                    quality=TagQuality.BAD,
                    timestamp=datetime.utcnow()
                )
            else:
                results[tag_name] = result

        return results

    async def read_all_condenser_tags(self) -> Dict[str, TagValue]:
        """
        Read all condenser instrumentation tags.

        Returns:
            Dictionary with all tag values
        """
        all_tags = self.tag_mapping.get_all_read_tags()
        return await self.read_tags(all_tags)

    async def write_setpoint(self, command: SetpointCommand) -> bool:
        """
        Write a setpoint value.

        Args:
            command: SetpointCommand with tag name and value

        Returns:
            True if write successful

        Raises:
            SCADAWriteError: If write fails
        """
        if not await self._circuit_breaker.can_execute():
            raise SCADAWriteError("Circuit breaker is open")

        self._stats["writes_total"] += 1

        try:
            if not self.is_connected:
                raise SCADAWriteError("Not connected to SCADA")

            # Validate the setpoint
            if command.tag_name not in self.tag_mapping.get_writable_tags():
                raise SCADAWriteError(f"Tag {command.tag_name} is not writable")

            # Validate value range
            self._validate_setpoint_value(command)

            # Simulated write operation
            # In production, this would call self._client.write_value(node_id, value)
            await self._simulate_write(command)

            # Update statistics
            self._stats["writes_success"] += 1
            self._stats["last_successful_write"] = datetime.utcnow()

            await self._circuit_breaker.record_success()

            logger.info(
                f"Successfully wrote setpoint {command.tag_name}={command.value} "
                f"(command_id={command.command_id})"
            )

            return True

        except Exception as e:
            self._stats["writes_failed"] += 1
            await self._circuit_breaker.record_failure()
            logger.error(f"Failed to write setpoint {command.tag_name}: {e}")
            raise SCADAWriteError(f"Write failed: {e}")

    async def write_setpoints(
        self,
        commands: List[SetpointCommand]
    ) -> Dict[str, bool]:
        """
        Write multiple setpoint values.

        Args:
            commands: List of SetpointCommands

        Returns:
            Dictionary mapping tag names to success status
        """
        results = {}

        for command in commands:
            try:
                success = await self.write_setpoint(command)
                results[command.tag_name] = success
            except Exception as e:
                logger.error(f"Failed to write {command.tag_name}: {e}")
                results[command.tag_name] = False

        return results

    def _validate_tag_value(self, value: TagValue) -> TagValue:
        """Validate a tag value against rules."""
        if value.tag_name in self._validation_rules:
            rule = self._validation_rules[value.tag_name]

            if value.value is not None:
                # Range validation
                if rule.min_value is not None and value.value < rule.min_value:
                    logger.warning(
                        f"Tag {value.tag_name} value {value.value} "
                        f"below minimum {rule.min_value}"
                    )
                    value.quality = TagQuality.UNCERTAIN

                if rule.max_value is not None and value.value > rule.max_value:
                    logger.warning(
                        f"Tag {value.tag_name} value {value.value} "
                        f"above maximum {rule.max_value}"
                    )
                    value.quality = TagQuality.UNCERTAIN

                # Rate of change validation
                if rule.rate_of_change_limit is not None:
                    if value.tag_name in self._current_values:
                        previous = self._current_values[value.tag_name]
                        if previous.value is not None:
                            time_diff = (
                                value.timestamp - previous.timestamp
                            ).total_seconds()
                            if time_diff > 0:
                                rate = abs(value.value - previous.value) / time_diff
                                if rate > rule.rate_of_change_limit:
                                    logger.warning(
                                        f"Tag {value.tag_name} rate of change "
                                        f"{rate:.2f} exceeds limit {rule.rate_of_change_limit}"
                                    )
                                    value.quality = TagQuality.UNCERTAIN

        return value

    def _validate_setpoint_value(self, command: SetpointCommand) -> None:
        """Validate a setpoint value before writing."""
        # Define setpoint limits
        setpoint_limits = {
            self.tag_mapping.CW_FLOW_SETPOINT: (0.0, 100000.0),
            self.tag_mapping.VACUUM_SETPOINT: (0.0, 101.325),
            self.tag_mapping.MAKEUP_VALVE_POSITION: (0.0, 100.0),
            self.tag_mapping.HOTWELL_LEVEL_SETPOINT: (0.0, 100.0),
        }

        if command.tag_name in setpoint_limits:
            min_val, max_val = setpoint_limits[command.tag_name]
            if not min_val <= command.value <= max_val:
                raise SCADAValidationError(
                    f"Setpoint value {command.value} out of range [{min_val}, {max_val}]"
                )

    async def _simulate_read(self, tag_name: str) -> TagValue:
        """Simulate reading a tag value (for testing)."""
        import random

        # Simulated values based on tag type
        simulated_values = {
            self.tag_mapping.CW_INLET_TEMP: random.uniform(20.0, 30.0),
            self.tag_mapping.CW_OUTLET_TEMP: random.uniform(30.0, 40.0),
            self.tag_mapping.CW_FLOW_RATE: random.uniform(40000.0, 60000.0),
            self.tag_mapping.CW_PRESSURE: random.uniform(2.0, 4.0),
            self.tag_mapping.VACUUM_PRESSURE: random.uniform(5.0, 10.0),
            self.tag_mapping.HOTWELL_LEVEL: random.uniform(45.0, 55.0),
            self.tag_mapping.CONDENSATE_FLOW: random.uniform(500.0, 1500.0),
            self.tag_mapping.CONDENSATE_TEMP: random.uniform(35.0, 45.0),
            self.tag_mapping.AIR_EJECTOR_STEAM: random.uniform(100.0, 200.0),
            self.tag_mapping.AIR_LEAKAGE_RATE: random.uniform(1.0, 5.0),
            self.tag_mapping.TTD: random.uniform(2.0, 5.0),
            self.tag_mapping.CLEANLINESS_FACTOR: random.uniform(0.75, 0.95),
        }

        value = simulated_values.get(tag_name, random.uniform(0.0, 100.0))

        await asyncio.sleep(0.01)  # Simulate network delay

        return TagValue(
            tag_name=tag_name,
            value=value,
            quality=TagQuality.GOOD,
            timestamp=datetime.utcnow(),
            source_timestamp=datetime.utcnow()
        )

    async def _simulate_write(self, command: SetpointCommand) -> None:
        """Simulate writing a setpoint value (for testing)."""
        await asyncio.sleep(0.02)  # Simulate network delay

    def register_value_callback(
        self,
        callback: Callable[[TagValue], None]
    ) -> None:
        """
        Register a callback for value updates.

        Args:
            callback: Function to call when values update
        """
        self._value_callbacks.append(callback)

    def unregister_value_callback(
        self,
        callback: Callable[[TagValue], None]
    ) -> None:
        """
        Unregister a value update callback.

        Args:
            callback: Function to remove
        """
        if callback in self._value_callbacks:
            self._value_callbacks.remove(callback)

    def _notify_value_update(self, value: TagValue) -> None:
        """Notify all registered callbacks of a value update."""
        for callback in self._value_callbacks:
            try:
                callback(value)
            except Exception as e:
                logger.error(f"Error in value callback: {e}")

    def get_current_value(self, tag_name: str) -> Optional[TagValue]:
        """Get the current cached value for a tag."""
        return self._current_values.get(tag_name)

    def get_all_current_values(self) -> Dict[str, TagValue]:
        """Get all current cached values."""
        return dict(self._current_values)

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            **self._stats,
            "connection_state": self._connection_state.value,
            "circuit_breaker_state": self._circuit_breaker.state,
            "subscribed_tags": len(self._subscription_handles),
            "cached_values": len(self._current_values)
        }

    def add_validation_rule(self, rule: TagValidationRule) -> None:
        """Add a custom validation rule for a tag."""
        self._validation_rules[rule.tag_name] = rule
        logger.debug(f"Added validation rule for {rule.tag_name}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on SCADA connection.

        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "connection_state": self._connection_state.value,
            "circuit_breaker": self._circuit_breaker.state,
            "timestamp": datetime.utcnow().isoformat()
        }

        if not self.is_connected:
            health["status"] = "unhealthy"
            health["reason"] = "Not connected to SCADA"
        elif self._circuit_breaker.state == "open":
            health["status"] = "degraded"
            health["reason"] = "Circuit breaker open"
        else:
            # Try a test read
            try:
                await self.read_tag(self.tag_mapping.CW_INLET_TEMP)
                health["last_read_success"] = True
            except Exception as e:
                health["status"] = "degraded"
                health["last_read_success"] = False
                health["reason"] = str(e)

        return health


# =============================================================================
# Factory Functions
# =============================================================================

def create_scada_config_from_env() -> SCADAConfig:
    """
    Create SCADA configuration from environment variables.

    Expected environment variables:
    - SCADA_ENDPOINT_URL
    - SCADA_USERNAME
    - SCADA_PASSWORD
    - SCADA_CERTIFICATE_PATH
    - SCADA_PRIVATE_KEY_PATH
    - SCADA_SECURITY_MODE
    - SCADA_NAMESPACE_INDEX
    """
    import os

    return SCADAConfig(
        endpoint_url=os.getenv("SCADA_ENDPOINT_URL", "opc.tcp://localhost:4840"),
        username=os.getenv("SCADA_USERNAME"),
        password=os.getenv("SCADA_PASSWORD"),
        certificate_path=os.getenv("SCADA_CERTIFICATE_PATH"),
        private_key_path=os.getenv("SCADA_PRIVATE_KEY_PATH"),
        security_mode=SecurityMode(
            os.getenv("SCADA_SECURITY_MODE", "SignAndEncrypt")
        ),
        namespace_index=int(os.getenv("SCADA_NAMESPACE_INDEX", "2"))
    )


async def create_and_connect_scada(
    config: Optional[SCADAConfig] = None
) -> SCADAIntegration:
    """
    Create and connect SCADA integration.

    Args:
        config: Optional configuration (uses env if not provided)

    Returns:
        Connected SCADAIntegration instance
    """
    if config is None:
        config = create_scada_config_from_env()

    integration = SCADAIntegration(config)
    await integration.connect()

    return integration
