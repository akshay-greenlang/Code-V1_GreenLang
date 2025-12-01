"""
Condition Monitoring Connector Module for GL-013 PREDICTMAINT (Predictive Maintenance Agent).

Provides integration with industrial condition monitoring systems including
SKF @ptitude, Emerson AMS, and GE Bently Nevada. Supports OPC-UA and Modbus
protocols for real-time vibration data streaming, alarm synchronization,
and trend data retrieval.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import asyncio
import logging
import struct
import time
import uuid
from collections import deque

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    CircuitState,
    ConnectionState,
    ConnectorType,
    DataQualityLevel,
    DataQualityResult,
    HealthCheckResult,
    HealthStatus,
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    ConnectorError,
    ProtocolError,
    RateLimitError,
    TimeoutError,
    ValidationError,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class ConditionMonitoringProvider(str, Enum):
    """Supported condition monitoring system providers."""

    SKF_APTITUDE = "skf_aptitude"
    EMERSON_AMS = "emerson_ams"
    GE_BENTLY_NEVADA = "ge_bently_nevada"
    PRUFTECHNIK = "pruftechnik"
    FLUKE = "fluke"
    VIBRATION_RESEARCH = "vibration_research"
    GENERIC_OPCUA = "generic_opcua"
    GENERIC_MODBUS = "generic_modbus"


class CommunicationProtocol(str, Enum):
    """Communication protocols supported."""

    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    REST_API = "rest_api"
    MQTT = "mqtt"
    PROFINET = "profinet"


class VibrationUnit(str, Enum):
    """Vibration measurement units."""

    MM_S = "mm/s"  # Velocity - millimeters per second
    IN_S = "in/s"  # Velocity - inches per second
    G = "g"  # Acceleration - gravitational units
    M_S2 = "m/s2"  # Acceleration - meters per second squared
    MIL = "mil"  # Displacement - mils (0.001 inch)
    UM = "um"  # Displacement - micrometers


class MeasurementType(str, Enum):
    """Types of condition monitoring measurements."""

    VELOCITY_RMS = "velocity_rms"
    VELOCITY_PEAK = "velocity_peak"
    ACCELERATION_RMS = "acceleration_rms"
    ACCELERATION_PEAK = "acceleration_peak"
    DISPLACEMENT_PP = "displacement_pp"  # Peak-to-peak
    ENVELOPE = "envelope"
    TEMPERATURE = "temperature"
    CURRENT = "current"
    PRESSURE = "pressure"
    FLOW = "flow"
    ULTRASONIC = "ultrasonic"
    OIL_ANALYSIS = "oil_analysis"


class AlarmSeverity(str, Enum):
    """Alarm severity levels (ISO 10816 based)."""

    NORMAL = "normal"  # Zone A - Good
    ALERT = "alert"  # Zone B - Acceptable
    WARNING = "warning"  # Zone C - Unsatisfactory
    DANGER = "danger"  # Zone D - Unacceptable
    CRITICAL = "critical"  # Immediate shutdown required


class AlarmState(str, Enum):
    """Alarm state values."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    CLEARED = "cleared"
    DISABLED = "disabled"


class MeasurementAxis(str, Enum):
    """Measurement axis for vibration."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    AXIAL = "axial"
    RADIAL = "radial"


class MachineState(str, Enum):
    """Machine operational state."""

    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    COASTING = "coasting"
    FAULT = "fault"


class TrendDirection(str, Enum):
    """Trend direction indicators."""

    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"
    ERRATIC = "erratic"


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class OPCUAConfig(BaseModel):
    """OPC-UA connection configuration."""

    model_config = ConfigDict(extra="forbid")

    endpoint_url: str = Field(..., description="OPC-UA server endpoint URL")
    namespace_uri: Optional[str] = Field(default=None, description="Namespace URI")
    security_policy: str = Field(
        default="None",
        description="Security policy (None, Basic256, Basic256Sha256)"
    )
    security_mode: str = Field(
        default="None",
        description="Security mode (None, Sign, SignAndEncrypt)"
    )
    username: Optional[str] = Field(default=None, description="Username for authentication")
    password: Optional[str] = Field(default=None, description="Password for authentication")
    certificate_path: Optional[str] = Field(default=None, description="Client certificate path")
    private_key_path: Optional[str] = Field(default=None, description="Private key path")
    application_uri: str = Field(
        default="urn:gl-013:predictmaint:opcua:client",
        description="Application URI"
    )
    session_timeout_ms: int = Field(
        default=60000,
        ge=1000,
        le=3600000,
        description="Session timeout in milliseconds"
    )
    subscription_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Subscription publish interval"
    )


class ModbusConfig(BaseModel):
    """Modbus connection configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(..., description="Modbus server host")
    port: int = Field(default=502, ge=1, le=65535, description="Modbus server port")
    unit_id: int = Field(default=1, ge=1, le=247, description="Modbus unit ID")
    protocol: CommunicationProtocol = Field(
        default=CommunicationProtocol.MODBUS_TCP,
        description="Modbus protocol variant"
    )

    # RTU settings (serial)
    serial_port: Optional[str] = Field(default=None, description="Serial port for RTU")
    baudrate: int = Field(default=9600, description="Baud rate for RTU")
    parity: str = Field(default="N", pattern="^[NEOSM]$", description="Parity (N/E/O/S/M)")
    stopbits: int = Field(default=1, ge=1, le=2, description="Stop bits")
    bytesize: int = Field(default=8, ge=5, le=8, description="Byte size")

    # Timing
    timeout_seconds: float = Field(default=3.0, ge=0.1, le=30.0, description="Timeout")
    retry_on_empty: bool = Field(default=True, description="Retry on empty response")


class ConditionMonitoringConnectorConfig(BaseConnectorConfig):
    """Configuration for condition monitoring connector."""

    model_config = ConfigDict(extra="forbid")

    # Provider settings
    provider: ConditionMonitoringProvider = Field(..., description="CM system provider")
    protocol: CommunicationProtocol = Field(..., description="Communication protocol")

    # Protocol-specific configuration
    opcua_config: Optional[OPCUAConfig] = Field(default=None, description="OPC-UA configuration")
    modbus_config: Optional[ModbusConfig] = Field(default=None, description="Modbus configuration")

    # REST API configuration (for providers with REST interface)
    api_base_url: Optional[str] = Field(default=None, description="REST API base URL")
    api_key: Optional[str] = Field(default=None, description="API key")
    api_username: Optional[str] = Field(default=None, description="API username")
    api_password: Optional[str] = Field(default=None, description="API password")

    # Data collection settings
    sampling_rate_hz: float = Field(
        default=1.0,
        ge=0.001,
        le=100000.0,
        description="Data sampling rate in Hz"
    )
    buffer_size: int = Field(
        default=1000,
        ge=100,
        le=1000000,
        description="Sample buffer size"
    )
    streaming_enabled: bool = Field(
        default=True,
        description="Enable real-time data streaming"
    )

    # Alarm settings
    alarm_sync_enabled: bool = Field(
        default=True,
        description="Enable alarm synchronization"
    )
    alarm_poll_interval_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=300.0,
        description="Alarm polling interval"
    )

    # Data validation
    validate_readings: bool = Field(
        default=True,
        description="Validate readings against thresholds"
    )
    reject_outliers: bool = Field(
        default=True,
        description="Reject outlier readings"
    )
    outlier_std_dev_threshold: float = Field(
        default=4.0,
        ge=2.0,
        le=10.0,
        description="Standard deviations for outlier detection"
    )

    # Measurement point configuration
    measurement_points: List[str] = Field(
        default_factory=list,
        description="List of measurement point IDs to monitor"
    )

    @field_validator('connector_type', mode='before')
    @classmethod
    def set_connector_type(cls, v):
        return ConnectorType.CONDITION_MONITORING


# =============================================================================
# Pydantic Models - Data Objects
# =============================================================================


class MeasurementPoint(BaseModel):
    """Measurement point configuration."""

    model_config = ConfigDict(extra="allow")

    point_id: str = Field(..., description="Unique measurement point ID")
    point_name: str = Field(..., description="Measurement point name")
    description: Optional[str] = Field(default=None, description="Description")

    # Equipment association
    equipment_id: str = Field(..., description="Associated equipment ID")
    equipment_name: Optional[str] = Field(default=None, description="Equipment name")
    location_description: Optional[str] = Field(default=None, description="Physical location")

    # Measurement configuration
    measurement_type: MeasurementType = Field(..., description="Type of measurement")
    axis: Optional[MeasurementAxis] = Field(default=None, description="Measurement axis")
    unit: str = Field(..., description="Measurement unit")
    sampling_rate_hz: float = Field(default=1.0, description="Sampling rate")

    # Alarm thresholds (ISO 10816 zones)
    alert_threshold: Optional[float] = Field(default=None, description="Alert threshold (Zone B)")
    warning_threshold: Optional[float] = Field(default=None, description="Warning threshold (Zone C)")
    danger_threshold: Optional[float] = Field(default=None, description="Danger threshold (Zone D)")

    # OPC-UA node ID (if applicable)
    opcua_node_id: Optional[str] = Field(default=None, description="OPC-UA node ID")

    # Modbus configuration (if applicable)
    modbus_address: Optional[int] = Field(default=None, description="Modbus register address")
    modbus_data_type: Optional[str] = Field(default=None, description="Modbus data type")
    modbus_scale_factor: float = Field(default=1.0, description="Scale factor for raw value")

    # Status
    enabled: bool = Field(default=True, description="Point enabled")
    online: bool = Field(default=True, description="Point online status")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class VibrationReading(BaseModel):
    """Single vibration reading with full context."""

    model_config = ConfigDict(extra="allow")

    reading_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Reading ID")
    point_id: str = Field(..., description="Measurement point ID")
    equipment_id: str = Field(..., description="Equipment ID")

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Reading timestamp")
    received_at: datetime = Field(default_factory=datetime.utcnow, description="Received timestamp")

    # Measurement values
    value: float = Field(..., description="Primary measurement value")
    unit: str = Field(..., description="Measurement unit")
    measurement_type: MeasurementType = Field(..., description="Measurement type")
    axis: Optional[MeasurementAxis] = Field(default=None, description="Measurement axis")

    # Quality indicators
    quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Data quality score")
    is_valid: bool = Field(default=True, description="Reading validity")
    validation_status: str = Field(default="valid", description="Validation status")

    # Alarm status
    alarm_severity: AlarmSeverity = Field(
        default=AlarmSeverity.NORMAL,
        description="Alarm severity"
    )
    threshold_exceeded: bool = Field(default=False, description="Threshold exceeded")

    # Machine context
    machine_state: Optional[MachineState] = Field(default=None, description="Machine state")
    rpm: Optional[float] = Field(default=None, ge=0, description="Machine RPM")
    load_percent: Optional[float] = Field(default=None, ge=0, le=100, description="Load percentage")

    # Raw data
    raw_value: Optional[float] = Field(default=None, description="Raw sensor value")
    raw_bytes: Optional[bytes] = Field(default=None, description="Raw bytes")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SpectrumData(BaseModel):
    """Vibration spectrum data (FFT)."""

    model_config = ConfigDict(extra="allow")

    spectrum_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Spectrum ID")
    point_id: str = Field(..., description="Measurement point ID")
    equipment_id: str = Field(..., description="Equipment ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")

    # Spectrum parameters
    fmax: float = Field(..., ge=0, description="Maximum frequency (Hz)")
    lines: int = Field(..., ge=100, description="Number of spectral lines")
    resolution: float = Field(..., ge=0, description="Frequency resolution (Hz)")
    window_type: str = Field(default="hanning", description="Window function type")
    averaging_type: str = Field(default="linear", description="Averaging type")
    num_averages: int = Field(default=4, ge=1, description="Number of averages")

    # Spectrum data
    frequencies: List[float] = Field(..., description="Frequency values (Hz)")
    amplitudes: List[float] = Field(..., description="Amplitude values")
    phases: Optional[List[float]] = Field(default=None, description="Phase values (degrees)")

    # Unit information
    amplitude_unit: str = Field(default="mm/s", description="Amplitude unit")
    phase_unit: str = Field(default="degrees", description="Phase unit")

    # Running speed data
    rpm: Optional[float] = Field(default=None, description="Machine RPM during measurement")
    order_frequencies: Optional[Dict[str, float]] = Field(
        default=None,
        description="Order frequencies (1X, 2X, etc.)"
    )

    # Peak detection
    peaks: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Detected peaks [{frequency, amplitude, order}]"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WaveformData(BaseModel):
    """Time-domain waveform data."""

    model_config = ConfigDict(extra="allow")

    waveform_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Waveform ID")
    point_id: str = Field(..., description="Measurement point ID")
    equipment_id: str = Field(..., description="Equipment ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")

    # Waveform parameters
    sample_rate_hz: float = Field(..., ge=0, description="Sample rate (Hz)")
    num_samples: int = Field(..., ge=1, description="Number of samples")
    duration_seconds: float = Field(..., ge=0, description="Duration (seconds)")

    # Waveform data
    samples: List[float] = Field(..., description="Sample values")
    unit: str = Field(default="mm/s", description="Sample unit")

    # Derived values
    rms: Optional[float] = Field(default=None, description="RMS value")
    peak: Optional[float] = Field(default=None, description="Peak value")
    peak_to_peak: Optional[float] = Field(default=None, description="Peak-to-peak value")
    crest_factor: Optional[float] = Field(default=None, description="Crest factor")
    kurtosis: Optional[float] = Field(default=None, description="Kurtosis")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Alarm(BaseModel):
    """Alarm from condition monitoring system."""

    model_config = ConfigDict(extra="allow")

    alarm_id: str = Field(..., description="Unique alarm ID")
    point_id: str = Field(..., description="Measurement point ID")
    equipment_id: str = Field(..., description="Equipment ID")

    # Alarm details
    alarm_type: str = Field(..., description="Alarm type/code")
    severity: AlarmSeverity = Field(..., description="Alarm severity")
    state: AlarmState = Field(default=AlarmState.ACTIVE, description="Alarm state")
    message: str = Field(..., description="Alarm message")

    # Thresholds
    threshold_value: Optional[float] = Field(default=None, description="Threshold that was exceeded")
    actual_value: Optional[float] = Field(default=None, description="Actual value that triggered")
    unit: Optional[str] = Field(default=None, description="Value unit")

    # Timestamps
    triggered_at: datetime = Field(default_factory=datetime.utcnow, description="Triggered timestamp")
    acknowledged_at: Optional[datetime] = Field(default=None, description="Acknowledged timestamp")
    cleared_at: Optional[datetime] = Field(default=None, description="Cleared timestamp")

    # User information
    acknowledged_by: Optional[str] = Field(default=None, description="Acknowledged by user")

    # Source information
    source_system: str = Field(..., description="Source system/provider")
    source_alarm_id: Optional[str] = Field(default=None, description="Original alarm ID from source")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TrendData(BaseModel):
    """Trend data for a measurement point."""

    model_config = ConfigDict(extra="allow")

    point_id: str = Field(..., description="Measurement point ID")
    equipment_id: str = Field(..., description="Equipment ID")
    measurement_type: MeasurementType = Field(..., description="Measurement type")
    unit: str = Field(..., description="Measurement unit")

    # Time range
    start_time: datetime = Field(..., description="Trend start time")
    end_time: datetime = Field(..., description="Trend end time")
    interval_seconds: float = Field(..., description="Data interval in seconds")

    # Trend data points
    timestamps: List[datetime] = Field(..., description="Timestamps")
    values: List[float] = Field(..., description="Measurement values")

    # Statistics
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    avg_value: float = Field(..., description="Average value")
    std_dev: float = Field(..., description="Standard deviation")

    # Trend analysis
    trend_direction: TrendDirection = Field(
        default=TrendDirection.STABLE,
        description="Trend direction"
    )
    trend_slope: Optional[float] = Field(default=None, description="Trend slope (units/day)")
    projected_alarm_date: Optional[datetime] = Field(
        default=None,
        description="Projected date to reach alarm level"
    )

    # Threshold references
    alert_threshold: Optional[float] = Field(default=None, description="Alert threshold")
    warning_threshold: Optional[float] = Field(default=None, description="Warning threshold")
    danger_threshold: Optional[float] = Field(default=None, description="Danger threshold")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RouteData(BaseModel):
    """Route collection data (portable data collector)."""

    model_config = ConfigDict(extra="allow")

    route_id: str = Field(..., description="Route ID")
    route_name: str = Field(..., description="Route name")
    description: Optional[str] = Field(default=None, description="Route description")

    # Collection info
    collector_id: Optional[str] = Field(default=None, description="Collector device ID")
    technician_id: Optional[str] = Field(default=None, description="Technician ID")
    collected_at: datetime = Field(default_factory=datetime.utcnow, description="Collection timestamp")
    uploaded_at: Optional[datetime] = Field(default=None, description="Upload timestamp")

    # Route points
    measurement_points: List[str] = Field(..., description="Measurement point IDs in route")
    readings: List[VibrationReading] = Field(
        default_factory=list,
        description="Collected readings"
    )

    # Completion status
    total_points: int = Field(..., ge=0, description="Total points in route")
    completed_points: int = Field(default=0, ge=0, description="Completed points")
    skipped_points: int = Field(default=0, ge=0, description="Skipped points")
    completion_percentage: float = Field(default=0.0, ge=0, le=100, description="Completion %")

    # Issues
    alarms_detected: int = Field(default=0, ge=0, description="Number of alarms detected")
    issues: List[str] = Field(default_factory=list, description="Issues found during collection")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# OPC-UA Client Wrapper
# =============================================================================


class OPCUAClientWrapper:
    """Wrapper for OPC-UA client operations."""

    def __init__(self, config: OPCUAConfig) -> None:
        """Initialize OPC-UA client wrapper."""
        self._config = config
        self._client = None
        self._session = None
        self._subscriptions: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._connected = False

    async def connect(self) -> None:
        """Establish OPC-UA connection."""
        try:
            # Import asyncua only when needed
            from asyncua import Client
            from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256

            self._client = Client(url=self._config.endpoint_url)

            # Configure security
            if self._config.security_policy != "None":
                await self._client.set_security_string(
                    f"{self._config.security_policy},{self._config.security_mode},"
                    f"{self._config.certificate_path},{self._config.private_key_path}"
                )

            # Set authentication
            if self._config.username and self._config.password:
                self._client.set_user(self._config.username)
                self._client.set_password(self._config.password)

            await self._client.connect()
            self._connected = True
            logger.info(f"OPC-UA connected to {self._config.endpoint_url}")

        except ImportError:
            raise ConfigurationError("asyncua package not installed. Install with: pip install asyncua")
        except Exception as e:
            raise ConnectionError(f"OPC-UA connection failed: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from OPC-UA server."""
        if self._client and self._connected:
            try:
                # Unsubscribe all
                for sub in self._subscriptions.values():
                    await sub.delete()
                self._subscriptions.clear()

                await self._client.disconnect()
            except Exception as e:
                logger.warning(f"Error during OPC-UA disconnect: {e}")
            finally:
                self._connected = False

    async def read_node(self, node_id: str) -> Any:
        """Read value from OPC-UA node."""
        if not self._connected:
            raise ConnectionError("OPC-UA not connected")

        from asyncua import ua

        node = self._client.get_node(node_id)
        value = await node.read_value()
        return value

    async def read_nodes(self, node_ids: List[str]) -> Dict[str, Any]:
        """Read multiple nodes."""
        results = {}
        for node_id in node_ids:
            try:
                results[node_id] = await self.read_node(node_id)
            except Exception as e:
                logger.warning(f"Failed to read node {node_id}: {e}")
                results[node_id] = None
        return results

    async def subscribe(
        self,
        node_ids: List[str],
        callback: Callable[[str, Any, datetime], None],
        interval_ms: Optional[int] = None,
    ) -> str:
        """
        Subscribe to node value changes.

        Returns subscription ID.
        """
        if not self._connected:
            raise ConnectionError("OPC-UA not connected")

        from asyncua import ua

        interval = interval_ms or self._config.subscription_interval_ms

        # Create subscription
        subscription = await self._client.create_subscription(interval, None)

        # Create monitored items
        for node_id in node_ids:
            node = self._client.get_node(node_id)
            await subscription.subscribe_data_change(node, callback)

        sub_id = str(uuid.uuid4())
        self._subscriptions[sub_id] = subscription
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from node changes."""
        if subscription_id in self._subscriptions:
            await self._subscriptions[subscription_id].delete()
            del self._subscriptions[subscription_id]

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# =============================================================================
# Modbus Client Wrapper
# =============================================================================


class ModbusClientWrapper:
    """Wrapper for Modbus client operations."""

    def __init__(self, config: ModbusConfig) -> None:
        """Initialize Modbus client wrapper."""
        self._config = config
        self._client = None
        self._lock = asyncio.Lock()
        self._connected = False

    async def connect(self) -> None:
        """Establish Modbus connection."""
        try:
            if self._config.protocol == CommunicationProtocol.MODBUS_TCP:
                from pymodbus.client import AsyncModbusTcpClient

                self._client = AsyncModbusTcpClient(
                    host=self._config.host,
                    port=self._config.port,
                    timeout=self._config.timeout_seconds,
                )
            else:
                from pymodbus.client import AsyncModbusSerialClient

                self._client = AsyncModbusSerialClient(
                    port=self._config.serial_port,
                    baudrate=self._config.baudrate,
                    parity=self._config.parity,
                    stopbits=self._config.stopbits,
                    bytesize=self._config.bytesize,
                    timeout=self._config.timeout_seconds,
                )

            await self._client.connect()
            self._connected = self._client.connected
            logger.info(f"Modbus connected to {self._config.host}:{self._config.port}")

        except ImportError:
            raise ConfigurationError("pymodbus package not installed. Install with: pip install pymodbus")
        except Exception as e:
            raise ConnectionError(f"Modbus connection failed: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from Modbus server."""
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error during Modbus disconnect: {e}")
            finally:
                self._connected = False

    async def read_holding_registers(
        self,
        address: int,
        count: int = 1,
        unit_id: Optional[int] = None,
    ) -> List[int]:
        """Read holding registers."""
        if not self._connected:
            raise ConnectionError("Modbus not connected")

        async with self._lock:
            unit = unit_id or self._config.unit_id
            response = await self._client.read_holding_registers(address, count, slave=unit)

            if response.isError():
                raise ProtocolError(f"Modbus read error: {response}", protocol="modbus")

            return response.registers

    async def read_input_registers(
        self,
        address: int,
        count: int = 1,
        unit_id: Optional[int] = None,
    ) -> List[int]:
        """Read input registers."""
        if not self._connected:
            raise ConnectionError("Modbus not connected")

        async with self._lock:
            unit = unit_id or self._config.unit_id
            response = await self._client.read_input_registers(address, count, slave=unit)

            if response.isError():
                raise ProtocolError(f"Modbus read error: {response}", protocol="modbus")

            return response.registers

    async def read_float(
        self,
        address: int,
        byte_order: str = ">",  # Big-endian
        word_order: str = ">",
        unit_id: Optional[int] = None,
    ) -> float:
        """Read floating point value from two registers."""
        registers = await self.read_holding_registers(address, 2, unit_id)

        # Pack registers into bytes then unpack as float
        if word_order == ">":
            packed = struct.pack(">HH", registers[0], registers[1])
        else:
            packed = struct.pack("<HH", registers[1], registers[0])

        return struct.unpack(f"{byte_order}f", packed)[0]

    async def write_register(
        self,
        address: int,
        value: int,
        unit_id: Optional[int] = None,
    ) -> None:
        """Write single register."""
        if not self._connected:
            raise ConnectionError("Modbus not connected")

        async with self._lock:
            unit = unit_id or self._config.unit_id
            response = await self._client.write_register(address, value, slave=unit)

            if response.isError():
                raise ProtocolError(f"Modbus write error: {response}", protocol="modbus")

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# =============================================================================
# Data Validator
# =============================================================================


class VibrationDataValidator:
    """Validates vibration readings for data quality."""

    def __init__(
        self,
        reject_outliers: bool = True,
        outlier_std_dev_threshold: float = 4.0,
    ) -> None:
        """Initialize validator."""
        self._reject_outliers = reject_outliers
        self._outlier_threshold = outlier_std_dev_threshold
        self._history: Dict[str, deque] = {}
        self._history_size = 100

    def validate(self, reading: VibrationReading) -> Tuple[bool, str, float]:
        """
        Validate a vibration reading.

        Returns:
            Tuple of (is_valid, status_message, quality_score)
        """
        quality_score = 1.0
        issues = []

        # Check for null/nan values
        if reading.value is None:
            return False, "null_value", 0.0

        import math
        if math.isnan(reading.value) or math.isinf(reading.value):
            return False, "invalid_number", 0.0

        # Check for negative values where inappropriate
        if reading.measurement_type in [
            MeasurementType.VELOCITY_RMS,
            MeasurementType.ACCELERATION_RMS,
        ]:
            if reading.value < 0:
                return False, "negative_rms", 0.0

        # Check for unrealistic values
        if reading.measurement_type == MeasurementType.VELOCITY_RMS:
            if reading.unit == "mm/s" and reading.value > 100:
                quality_score *= 0.5
                issues.append("unusually_high")
        elif reading.measurement_type == MeasurementType.TEMPERATURE:
            if reading.value < -50 or reading.value > 200:
                return False, "temperature_out_of_range", 0.0

        # Check for outliers using historical data
        if self._reject_outliers:
            point_id = reading.point_id
            if point_id not in self._history:
                self._history[point_id] = deque(maxlen=self._history_size)

            history = self._history[point_id]
            if len(history) >= 10:
                import statistics
                mean = statistics.mean(history)
                std_dev = statistics.stdev(history) if len(history) > 1 else 0

                if std_dev > 0:
                    z_score = abs(reading.value - mean) / std_dev
                    if z_score > self._outlier_threshold:
                        quality_score *= 0.3
                        issues.append("statistical_outlier")

            # Add to history
            history.append(reading.value)

        # Determine final status
        if quality_score >= 0.9:
            status = "valid"
        elif quality_score >= 0.5:
            status = f"degraded:{','.join(issues)}"
        else:
            status = f"poor:{','.join(issues)}"

        return quality_score >= 0.5, status, quality_score


# =============================================================================
# Condition Monitoring Connector Implementation
# =============================================================================


class ConditionMonitoringConnector(BaseConnector):
    """
    Condition Monitoring System Connector.

    Provides integration with industrial condition monitoring systems including:
    - SKF @ptitude
    - Emerson AMS
    - GE Bently Nevada

    Features:
    - OPC-UA and Modbus TCP/RTU protocols
    - Real-time vibration data streaming
    - Spectrum and waveform data collection
    - Alarm state synchronization
    - Route data collection
    - Trend data retrieval
    - Data quality validation
    """

    def __init__(self, config: ConditionMonitoringConnectorConfig) -> None:
        """
        Initialize condition monitoring connector.

        Args:
            config: Connector configuration
        """
        super().__init__(config)
        self._cm_config = config

        # Protocol clients
        self._opcua_client: Optional[OPCUAClientWrapper] = None
        self._modbus_client: Optional[ModbusClientWrapper] = None

        # HTTP client for REST API
        self._http_client = None

        # Data validator
        self._validator = VibrationDataValidator(
            reject_outliers=config.reject_outliers,
            outlier_std_dev_threshold=config.outlier_std_dev_threshold,
        )

        # Measurement points cache
        self._measurement_points: Dict[str, MeasurementPoint] = {}

        # Streaming state
        self._streaming_task: Optional[asyncio.Task] = None
        self._streaming_buffer: asyncio.Queue = asyncio.Queue(maxsize=config.buffer_size)
        self._streaming_callbacks: List[Callable[[VibrationReading], None]] = []

        # Alarm state
        self._active_alarms: Dict[str, Alarm] = {}
        self._alarm_poll_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Establish connection to condition monitoring system."""
        self._logger.info(
            f"Connecting to {self._cm_config.provider.value} via {self._cm_config.protocol.value}..."
        )

        if self._cm_config.protocol == CommunicationProtocol.OPC_UA:
            if not self._cm_config.opcua_config:
                raise ConfigurationError("OPC-UA configuration required")

            self._opcua_client = OPCUAClientWrapper(self._cm_config.opcua_config)
            await self._opcua_client.connect()

        elif self._cm_config.protocol in [
            CommunicationProtocol.MODBUS_TCP,
            CommunicationProtocol.MODBUS_RTU,
        ]:
            if not self._cm_config.modbus_config:
                raise ConfigurationError("Modbus configuration required")

            self._modbus_client = ModbusClientWrapper(self._cm_config.modbus_config)
            await self._modbus_client.connect()

        elif self._cm_config.protocol == CommunicationProtocol.REST_API:
            if not self._cm_config.api_base_url:
                raise ConfigurationError("API base URL required")

            import httpx
            self._http_client = httpx.AsyncClient(
                base_url=self._cm_config.api_base_url,
                timeout=httpx.Timeout(self._cm_config.connection_timeout_seconds),
            )

        # Load measurement points
        await self._load_measurement_points()

        # Start alarm polling if enabled
        if self._cm_config.alarm_sync_enabled:
            self._alarm_poll_task = asyncio.create_task(self._alarm_poll_loop())

        self._logger.info(f"Connected to {self._cm_config.provider.value}")

    async def disconnect(self) -> None:
        """Disconnect from condition monitoring system."""
        # Stop streaming
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass

        # Stop alarm polling
        if self._alarm_poll_task:
            self._alarm_poll_task.cancel()
            try:
                await self._alarm_poll_task
            except asyncio.CancelledError:
                pass

        # Disconnect clients
        if self._opcua_client:
            await self._opcua_client.disconnect()
            self._opcua_client = None

        if self._modbus_client:
            await self._modbus_client.disconnect()
            self._modbus_client = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._logger.info("Disconnected from condition monitoring system")

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on connection."""
        start_time = time.time()

        try:
            # Check protocol-specific connection
            if self._cm_config.protocol == CommunicationProtocol.OPC_UA:
                if not self._opcua_client or not self._opcua_client.is_connected:
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        message="OPC-UA client not connected",
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                # Try reading a node
                if self._measurement_points:
                    point = list(self._measurement_points.values())[0]
                    if point.opcua_node_id:
                        await self._opcua_client.read_node(point.opcua_node_id)

            elif self._cm_config.protocol in [
                CommunicationProtocol.MODBUS_TCP,
                CommunicationProtocol.MODBUS_RTU,
            ]:
                if not self._modbus_client or not self._modbus_client.is_connected:
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        message="Modbus client not connected",
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                # Try reading a register
                await self._modbus_client.read_holding_registers(0, 1)

            elif self._cm_config.protocol == CommunicationProtocol.REST_API:
                if not self._http_client:
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        message="HTTP client not initialized",
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                response = await self._http_client.get("/health")
                response.raise_for_status()

            latency_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Condition monitoring connection healthy",
                latency_ms=latency_ms,
                details={
                    "provider": self._cm_config.provider.value,
                    "protocol": self._cm_config.protocol.value,
                    "measurement_points": len(self._measurement_points),
                    "active_alarms": len(self._active_alarms),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def validate_configuration(self) -> bool:
        """Validate connector configuration."""
        # Validate protocol-specific configuration
        if self._cm_config.protocol == CommunicationProtocol.OPC_UA:
            if not self._cm_config.opcua_config:
                raise ConfigurationError("OPC-UA configuration required")
            if not self._cm_config.opcua_config.endpoint_url:
                raise ConfigurationError("OPC-UA endpoint URL required")

        elif self._cm_config.protocol in [
            CommunicationProtocol.MODBUS_TCP,
            CommunicationProtocol.MODBUS_RTU,
        ]:
            if not self._cm_config.modbus_config:
                raise ConfigurationError("Modbus configuration required")

        elif self._cm_config.protocol == CommunicationProtocol.REST_API:
            if not self._cm_config.api_base_url:
                raise ConfigurationError("API base URL required")

        return True

    async def _load_measurement_points(self) -> None:
        """Load measurement point configuration."""
        # In real implementation, this would load from the CM system
        # For now, use configured points
        for point_id in self._cm_config.measurement_points:
            self._measurement_points[point_id] = MeasurementPoint(
                point_id=point_id,
                point_name=f"Point {point_id}",
                equipment_id="unknown",
                measurement_type=MeasurementType.VELOCITY_RMS,
                unit="mm/s",
            )

    # =========================================================================
    # Measurement Point Operations
    # =========================================================================

    async def get_measurement_points(
        self,
        equipment_id: Optional[str] = None,
    ) -> List[MeasurementPoint]:
        """
        Get configured measurement points.

        Args:
            equipment_id: Optional equipment ID filter

        Returns:
            List of measurement points
        """
        points = list(self._measurement_points.values())

        if equipment_id:
            points = [p for p in points if p.equipment_id == equipment_id]

        return points

    async def add_measurement_point(self, point: MeasurementPoint) -> None:
        """
        Add a measurement point to monitoring.

        Args:
            point: Measurement point configuration
        """
        self._measurement_points[point.point_id] = point
        self._logger.info(f"Added measurement point: {point.point_id}")

    async def remove_measurement_point(self, point_id: str) -> None:
        """
        Remove a measurement point from monitoring.

        Args:
            point_id: Point ID to remove
        """
        if point_id in self._measurement_points:
            del self._measurement_points[point_id]
            self._logger.info(f"Removed measurement point: {point_id}")

    # =========================================================================
    # Real-time Data Collection
    # =========================================================================

    async def read_current_values(
        self,
        point_ids: Optional[List[str]] = None,
    ) -> List[VibrationReading]:
        """
        Read current values from measurement points.

        Args:
            point_ids: Optional list of point IDs (reads all if not specified)

        Returns:
            List of current readings
        """
        if point_ids is None:
            point_ids = list(self._measurement_points.keys())

        readings = []
        timestamp = datetime.utcnow()

        for point_id in point_ids:
            if point_id not in self._measurement_points:
                continue

            point = self._measurement_points[point_id]

            try:
                value = await self._read_point_value(point)

                reading = VibrationReading(
                    point_id=point_id,
                    equipment_id=point.equipment_id,
                    timestamp=timestamp,
                    value=value,
                    unit=point.unit,
                    measurement_type=point.measurement_type,
                    axis=point.axis,
                )

                # Validate reading
                if self._cm_config.validate_readings:
                    is_valid, status, quality = self._validator.validate(reading)
                    reading.is_valid = is_valid
                    reading.validation_status = status
                    reading.quality = quality

                # Check alarm thresholds
                reading.alarm_severity = self._check_alarm_thresholds(point, value)
                reading.threshold_exceeded = reading.alarm_severity != AlarmSeverity.NORMAL

                readings.append(reading)

            except Exception as e:
                self._logger.warning(f"Failed to read point {point_id}: {e}")

        return readings

    async def _read_point_value(self, point: MeasurementPoint) -> float:
        """Read value from a single measurement point."""
        if self._cm_config.protocol == CommunicationProtocol.OPC_UA:
            if not point.opcua_node_id:
                raise ConfigurationError(f"No OPC-UA node ID for point {point.point_id}")
            return await self._opcua_client.read_node(point.opcua_node_id)

        elif self._cm_config.protocol in [
            CommunicationProtocol.MODBUS_TCP,
            CommunicationProtocol.MODBUS_RTU,
        ]:
            if point.modbus_address is None:
                raise ConfigurationError(f"No Modbus address for point {point.point_id}")

            raw_value = await self._modbus_client.read_float(point.modbus_address)
            return raw_value * point.modbus_scale_factor

        elif self._cm_config.protocol == CommunicationProtocol.REST_API:
            response = await self._http_client.get(f"/points/{point.point_id}/current")
            response.raise_for_status()
            data = response.json()
            return data.get("value", 0.0)

        raise ConfigurationError(f"Unsupported protocol: {self._cm_config.protocol}")

    def _check_alarm_thresholds(self, point: MeasurementPoint, value: float) -> AlarmSeverity:
        """Check value against alarm thresholds."""
        if point.danger_threshold and value >= point.danger_threshold:
            return AlarmSeverity.DANGER
        elif point.warning_threshold and value >= point.warning_threshold:
            return AlarmSeverity.WARNING
        elif point.alert_threshold and value >= point.alert_threshold:
            return AlarmSeverity.ALERT
        return AlarmSeverity.NORMAL

    # =========================================================================
    # Streaming Data Collection
    # =========================================================================

    async def start_streaming(
        self,
        callback: Optional[Callable[[VibrationReading], None]] = None,
    ) -> None:
        """
        Start real-time data streaming.

        Args:
            callback: Optional callback for each reading
        """
        if not self._cm_config.streaming_enabled:
            raise ConfigurationError("Streaming not enabled")

        if callback:
            self._streaming_callbacks.append(callback)

        if self._streaming_task is None or self._streaming_task.done():
            self._streaming_task = asyncio.create_task(self._streaming_loop())
            self._logger.info("Started data streaming")

    async def stop_streaming(self) -> None:
        """Stop real-time data streaming."""
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
            self._streaming_task = None
            self._logger.info("Stopped data streaming")

    async def _streaming_loop(self) -> None:
        """Background task for continuous data streaming."""
        interval = 1.0 / self._cm_config.sampling_rate_hz

        while True:
            try:
                readings = await self.read_current_values()

                for reading in readings:
                    # Add to buffer
                    try:
                        self._streaming_buffer.put_nowait(reading)
                    except asyncio.QueueFull:
                        # Drop oldest
                        try:
                            self._streaming_buffer.get_nowait()
                            self._streaming_buffer.put_nowait(reading)
                        except asyncio.QueueEmpty:
                            pass

                    # Call callbacks
                    for callback in self._streaming_callbacks:
                        try:
                            callback(reading)
                        except Exception as e:
                            self._logger.warning(f"Streaming callback error: {e}")

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Streaming error: {e}")
                await asyncio.sleep(1.0)  # Back off on error

    async def get_streaming_data(
        self,
        max_items: int = 100,
        timeout: float = 1.0,
    ) -> List[VibrationReading]:
        """
        Get buffered streaming data.

        Args:
            max_items: Maximum items to retrieve
            timeout: Timeout for waiting

        Returns:
            List of buffered readings
        """
        readings = []

        try:
            for _ in range(max_items):
                reading = await asyncio.wait_for(
                    self._streaming_buffer.get(),
                    timeout=timeout,
                )
                readings.append(reading)
        except asyncio.TimeoutError:
            pass

        return readings

    # =========================================================================
    # Spectrum and Waveform Data
    # =========================================================================

    async def collect_spectrum(
        self,
        point_id: str,
        fmax: float = 1000.0,
        lines: int = 800,
        averages: int = 4,
        window_type: str = "hanning",
    ) -> SpectrumData:
        """
        Collect spectrum data from measurement point.

        Args:
            point_id: Measurement point ID
            fmax: Maximum frequency (Hz)
            lines: Number of spectral lines
            averages: Number of averages
            window_type: Window function type

        Returns:
            Spectrum data
        """
        if point_id not in self._measurement_points:
            raise ValidationError(f"Unknown measurement point: {point_id}")

        point = self._measurement_points[point_id]

        if self._cm_config.protocol == CommunicationProtocol.REST_API:
            response = await self._http_client.post(
                f"/points/{point_id}/spectrum",
                json={
                    "fmax": fmax,
                    "lines": lines,
                    "averages": averages,
                    "window": window_type,
                },
            )
            response.raise_for_status()
            data = response.json()

            return SpectrumData(
                point_id=point_id,
                equipment_id=point.equipment_id,
                fmax=fmax,
                lines=lines,
                resolution=fmax / lines,
                window_type=window_type,
                num_averages=averages,
                frequencies=data.get("frequencies", []),
                amplitudes=data.get("amplitudes", []),
                phases=data.get("phases"),
                amplitude_unit=point.unit,
                rpm=data.get("rpm"),
            )

        # For OPC-UA/Modbus, spectrum collection may not be supported
        # or requires specific implementation
        raise NotImplementedError(
            f"Spectrum collection not implemented for {self._cm_config.protocol.value}"
        )

    async def collect_waveform(
        self,
        point_id: str,
        sample_rate_hz: float = 10000.0,
        duration_seconds: float = 1.0,
    ) -> WaveformData:
        """
        Collect time-domain waveform data.

        Args:
            point_id: Measurement point ID
            sample_rate_hz: Sample rate (Hz)
            duration_seconds: Duration (seconds)

        Returns:
            Waveform data
        """
        if point_id not in self._measurement_points:
            raise ValidationError(f"Unknown measurement point: {point_id}")

        point = self._measurement_points[point_id]

        if self._cm_config.protocol == CommunicationProtocol.REST_API:
            response = await self._http_client.post(
                f"/points/{point_id}/waveform",
                json={
                    "sample_rate": sample_rate_hz,
                    "duration": duration_seconds,
                },
            )
            response.raise_for_status()
            data = response.json()

            samples = data.get("samples", [])

            # Calculate derived values
            import statistics
            rms = None
            peak = None
            peak_to_peak = None

            if samples:
                rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5
                peak = max(abs(s) for s in samples)
                peak_to_peak = max(samples) - min(samples)

            return WaveformData(
                point_id=point_id,
                equipment_id=point.equipment_id,
                sample_rate_hz=sample_rate_hz,
                num_samples=len(samples),
                duration_seconds=duration_seconds,
                samples=samples,
                unit=point.unit,
                rms=rms,
                peak=peak,
                peak_to_peak=peak_to_peak,
            )

        raise NotImplementedError(
            f"Waveform collection not implemented for {self._cm_config.protocol.value}"
        )

    # =========================================================================
    # Trend Data
    # =========================================================================

    async def get_trend_data(
        self,
        point_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        interval_seconds: float = 3600.0,
    ) -> TrendData:
        """
        Get historical trend data for measurement point.

        Args:
            point_id: Measurement point ID
            start_time: Start of time range
            end_time: End of time range (default: now)
            interval_seconds: Data interval

        Returns:
            Trend data
        """
        if point_id not in self._measurement_points:
            raise ValidationError(f"Unknown measurement point: {point_id}")

        point = self._measurement_points[point_id]
        end_time = end_time or datetime.utcnow()

        if self._cm_config.protocol == CommunicationProtocol.REST_API:
            response = await self._http_client.get(
                f"/points/{point_id}/trend",
                params={
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "interval": interval_seconds,
                },
            )
            response.raise_for_status()
            data = response.json()

            values = data.get("values", [])
            timestamps = [
                datetime.fromisoformat(ts) for ts in data.get("timestamps", [])
            ]

            # Calculate statistics
            import statistics
            if values:
                min_val = min(values)
                max_val = max(values)
                avg_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                min_val = max_val = avg_val = std_val = 0.0

            # Calculate trend direction
            trend_direction = self._calculate_trend_direction(values)

            return TrendData(
                point_id=point_id,
                equipment_id=point.equipment_id,
                measurement_type=point.measurement_type,
                unit=point.unit,
                start_time=start_time,
                end_time=end_time,
                interval_seconds=interval_seconds,
                timestamps=timestamps,
                values=values,
                min_value=min_val,
                max_value=max_val,
                avg_value=avg_val,
                std_dev=std_val,
                trend_direction=trend_direction,
                alert_threshold=point.alert_threshold,
                warning_threshold=point.warning_threshold,
                danger_threshold=point.danger_threshold,
            )

        # For other protocols, trend may need to be built from current readings
        raise NotImplementedError(
            f"Trend retrieval not implemented for {self._cm_config.protocol.value}"
        )

    def _calculate_trend_direction(self, values: List[float]) -> TrendDirection:
        """Calculate trend direction from values."""
        if len(values) < 10:
            return TrendDirection.STABLE

        # Simple linear regression
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return TrendDirection.STABLE

        slope = numerator / denominator

        # Normalize slope by mean value
        if y_mean > 0:
            normalized_slope = slope / y_mean
        else:
            normalized_slope = slope

        # Classify trend
        if normalized_slope > 0.01:  # > 1% increase per sample
            return TrendDirection.INCREASING
        elif normalized_slope < -0.01:  # > 1% decrease per sample
            return TrendDirection.DECREASING
        else:
            return TrendDirection.STABLE

    # =========================================================================
    # Alarm Operations
    # =========================================================================

    async def get_active_alarms(
        self,
        equipment_id: Optional[str] = None,
        severity: Optional[AlarmSeverity] = None,
    ) -> List[Alarm]:
        """
        Get active alarms.

        Args:
            equipment_id: Optional equipment filter
            severity: Optional severity filter

        Returns:
            List of active alarms
        """
        alarms = list(self._active_alarms.values())

        if equipment_id:
            alarms = [a for a in alarms if a.equipment_id == equipment_id]

        if severity:
            alarms = [a for a in alarms if a.severity == severity]

        return alarms

    async def acknowledge_alarm(self, alarm_id: str, user_id: str) -> Alarm:
        """
        Acknowledge an alarm.

        Args:
            alarm_id: Alarm ID
            user_id: User acknowledging

        Returns:
            Updated alarm
        """
        if alarm_id not in self._active_alarms:
            raise ValidationError(f"Unknown alarm: {alarm_id}")

        alarm = self._active_alarms[alarm_id]

        # Update in source system if REST API
        if self._cm_config.protocol == CommunicationProtocol.REST_API:
            response = await self._http_client.post(
                f"/alarms/{alarm_id}/acknowledge",
                json={"user_id": user_id},
            )
            response.raise_for_status()

        # Update local state
        updated_alarm = Alarm(
            **{
                **alarm.model_dump(),
                "state": AlarmState.ACKNOWLEDGED,
                "acknowledged_at": datetime.utcnow(),
                "acknowledged_by": user_id,
            }
        )
        self._active_alarms[alarm_id] = updated_alarm

        return updated_alarm

    async def _alarm_poll_loop(self) -> None:
        """Background task for polling alarms."""
        while True:
            try:
                await asyncio.sleep(self._cm_config.alarm_poll_interval_seconds)
                await self._sync_alarms()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Alarm poll error: {e}")

    async def _sync_alarms(self) -> None:
        """Synchronize alarms from source system."""
        if self._cm_config.protocol == CommunicationProtocol.REST_API:
            response = await self._http_client.get("/alarms/active")
            response.raise_for_status()
            alarms_data = response.json()

            # Update active alarms
            new_alarms = {}
            for alarm_data in alarms_data.get("alarms", []):
                alarm = Alarm(**alarm_data)
                new_alarms[alarm.alarm_id] = alarm

                # Check for new alarms
                if alarm.alarm_id not in self._active_alarms:
                    self._logger.warning(
                        f"New alarm: {alarm.alarm_id} - {alarm.message} ({alarm.severity.value})"
                    )

            # Check for cleared alarms
            for alarm_id in self._active_alarms:
                if alarm_id not in new_alarms:
                    self._logger.info(f"Alarm cleared: {alarm_id}")

            self._active_alarms = new_alarms

    # =========================================================================
    # Route Collection
    # =========================================================================

    async def get_routes(self) -> List[RouteData]:
        """
        Get available data collection routes.

        Returns:
            List of routes
        """
        if self._cm_config.protocol == CommunicationProtocol.REST_API:
            response = await self._http_client.get("/routes")
            response.raise_for_status()
            data = response.json()

            return [RouteData(**route) for route in data.get("routes", [])]

        return []

    async def get_route_data(
        self,
        route_id: str,
        collection_date: Optional[datetime] = None,
    ) -> RouteData:
        """
        Get route collection data.

        Args:
            route_id: Route ID
            collection_date: Optional specific collection date

        Returns:
            Route data with readings
        """
        if self._cm_config.protocol == CommunicationProtocol.REST_API:
            params = {}
            if collection_date:
                params["date"] = collection_date.isoformat()

            response = await self._http_client.get(
                f"/routes/{route_id}/data",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            return RouteData(**data)

        raise NotImplementedError(
            f"Route data not available for {self._cm_config.protocol.value}"
        )

    # =========================================================================
    # Batch Data Operations
    # =========================================================================

    async def get_equipment_readings(
        self,
        equipment_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> List[VibrationReading]:
        """
        Get all readings for equipment in time range.

        Args:
            equipment_id: Equipment ID
            start_time: Start time
            end_time: End time (default: now)

        Returns:
            List of readings
        """
        end_time = end_time or datetime.utcnow()

        # Get measurement points for equipment
        points = await self.get_measurement_points(equipment_id)

        if not points:
            return []

        if self._cm_config.protocol == CommunicationProtocol.REST_API:
            all_readings = []

            for point in points:
                response = await self._http_client.get(
                    f"/points/{point.point_id}/readings",
                    params={
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                    },
                )
                response.raise_for_status()
                data = response.json()

                for reading_data in data.get("readings", []):
                    reading = VibrationReading(
                        point_id=point.point_id,
                        equipment_id=equipment_id,
                        measurement_type=point.measurement_type,
                        unit=point.unit,
                        **reading_data,
                    )
                    all_readings.append(reading)

            return all_readings

        return []

    async def stream_readings(
        self,
        equipment_id: Optional[str] = None,
    ) -> AsyncGenerator[VibrationReading, None]:
        """
        Async generator for streaming readings.

        Args:
            equipment_id: Optional equipment ID filter

        Yields:
            Vibration readings
        """
        while True:
            try:
                readings = await self.read_current_values()

                for reading in readings:
                    if equipment_id is None or reading.equipment_id == equipment_id:
                        yield reading

                await asyncio.sleep(1.0 / self._cm_config.sampling_rate_hz)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Stream error: {e}")
                await asyncio.sleep(1.0)


# =============================================================================
# Factory Function
# =============================================================================


def create_condition_monitoring_connector(
    provider: ConditionMonitoringProvider,
    protocol: CommunicationProtocol,
    connector_name: str,
    opcua_config: Optional[OPCUAConfig] = None,
    modbus_config: Optional[ModbusConfig] = None,
    api_base_url: Optional[str] = None,
    **kwargs,
) -> ConditionMonitoringConnector:
    """
    Factory function to create condition monitoring connector.

    Args:
        provider: CM system provider
        protocol: Communication protocol
        connector_name: Connector name
        opcua_config: OPC-UA configuration
        modbus_config: Modbus configuration
        api_base_url: REST API base URL
        **kwargs: Additional configuration options

    Returns:
        Configured condition monitoring connector
    """
    config = ConditionMonitoringConnectorConfig(
        connector_name=connector_name,
        connector_type=ConnectorType.CONDITION_MONITORING,
        provider=provider,
        protocol=protocol,
        opcua_config=opcua_config,
        modbus_config=modbus_config,
        api_base_url=api_base_url,
        **kwargs,
    )

    return ConditionMonitoringConnector(config)
