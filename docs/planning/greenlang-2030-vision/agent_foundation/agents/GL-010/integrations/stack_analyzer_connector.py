"""
Stack Gas Analyzer Connector for GL-010 EMISSIONWATCH.

Provides integration with multi-gas stack analyzers including NDIR CO/CO2,
chemiluminescence NOx, UV fluorescence SO2, zirconia O2 sensors, and
opacity monitors for emissions monitoring compliance.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import asyncio
import logging
import time
import uuid

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    ConnectionState,
    ConnectorType,
    HealthCheckResult,
    HealthStatus,
    ConnectorError,
    ConnectionError,
    ConfigurationError,
    ValidationError,
    DataQualityError,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class AnalyzerTechnology(str, Enum):
    """Gas analyzer technology types."""

    NDIR = "ndir"  # Non-Dispersive Infrared
    CHEMILUMINESCENCE = "chemiluminescence"  # NOx detection
    UV_FLUORESCENCE = "uv_fluorescence"  # SO2 detection
    PARAMAGNETIC = "paramagnetic"  # O2 detection
    ZIRCONIA = "zirconia"  # O2 detection (in-situ)
    FTIR = "ftir"  # Fourier Transform Infrared
    UV_ABSORPTION = "uv_absorption"  # Multi-gas
    LASER = "laser"  # Tunable Diode Laser
    ELECTROCHEMICAL = "electrochemical"  # Various gases
    PHOTOACOUSTIC = "photoacoustic"  # Multi-gas
    TRANSMISSOMETER = "transmissometer"  # Opacity
    LIGHT_SCATTERING = "light_scattering"  # Particulate


class AnalyzerVendor(str, Enum):
    """Stack analyzer vendors."""

    THERMO_FISHER = "thermo_fisher"
    TELEDYNE = "teledyne"
    HORIBA = "horiba"
    SIEMENS = "siemens"
    ABB = "abb"
    EMERSON = "emerson"
    YOKOGAWA = "yokogawa"
    SICK = "sick"
    SERVOMEX = "servomex"
    FUJI_ELECTRIC = "fuji_electric"
    AMETEK = "ametek"
    GENERAL_ELECTRIC = "general_electric"
    DURAG = "durag"
    ENVIRO = "enviro"
    GENERIC = "generic"


class GasComponent(str, Enum):
    """Gas components measured."""

    NOX = "nox"
    NO = "no"
    NO2 = "no2"
    SO2 = "so2"
    CO = "co"
    CO2 = "co2"
    O2 = "o2"
    N2O = "n2o"
    HCL = "hcl"
    HF = "hf"
    NH3 = "nh3"
    CH4 = "ch4"
    THC = "thc"  # Total hydrocarbons
    H2S = "h2s"
    MOISTURE = "moisture"
    OPACITY = "opacity"
    PM = "pm"  # Particulate matter


class MeasurementRange(str, Enum):
    """Measurement range settings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class AnalyzerStatus(str, Enum):
    """Analyzer operational status."""

    NORMAL = "normal"
    WARMUP = "warmup"
    STANDBY = "standby"
    CALIBRATING = "calibrating"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    ALARM = "alarm"
    OFFLINE = "offline"


class CalibrationMode(str, Enum):
    """Calibration modes."""

    ZERO = "zero"
    SPAN = "span"
    MID = "mid"
    MULTI_POINT = "multi_point"
    LINEARITY = "linearity"
    CEMS = "cems"  # Full CEMS calibration


class AlarmType(str, Enum):
    """Alarm types."""

    RANGE_HIGH = "range_high"
    RANGE_LOW = "range_low"
    SPAN_DRIFT = "span_drift"
    ZERO_DRIFT = "zero_drift"
    FLOW_FAULT = "flow_fault"
    TEMPERATURE_FAULT = "temperature_fault"
    PRESSURE_FAULT = "pressure_fault"
    COMMUNICATION = "communication"
    CALIBRATION_DUE = "calibration_due"
    MAINTENANCE_DUE = "maintenance_due"
    DETECTOR_FAULT = "detector_fault"
    LAMP_FAULT = "lamp_fault"


# =============================================================================
# Pydantic Models
# =============================================================================


class AnalyzerReading(BaseModel):
    """Single analyzer reading."""

    model_config = ConfigDict(frozen=True)

    reading_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Reading identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Reading timestamp"
    )
    gas_component: GasComponent = Field(..., description="Gas component")
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Measurement unit")

    # Quality indicators
    is_valid: bool = Field(default=True, description="Reading validity")
    quality_code: str = Field(default="OK", description="Quality code")

    # Raw/analog values
    raw_value: Optional[float] = Field(default=None, description="Raw analog value")
    raw_unit: Optional[str] = Field(default=None, description="Raw value unit")

    # Range information
    range_setting: MeasurementRange = Field(
        default=MeasurementRange.AUTO,
        description="Range setting"
    )
    full_scale: Optional[float] = Field(default=None, description="Full scale value")
    percent_of_range: Optional[float] = Field(default=None, ge=0, le=100)


class AnalyzerChannelConfig(BaseModel):
    """Configuration for an analyzer channel."""

    model_config = ConfigDict(frozen=True)

    channel_id: str = Field(..., description="Channel identifier")
    gas_component: GasComponent = Field(..., description="Gas component")
    technology: AnalyzerTechnology = Field(..., description="Detection technology")

    # Measurement ranges
    range_low: float = Field(default=0.0, description="Low range limit")
    range_high: float = Field(..., description="High range limit")
    unit: str = Field(..., description="Measurement unit")

    # Calibration settings
    zero_value: float = Field(default=0.0, description="Zero calibration value")
    span_value: float = Field(..., description="Span calibration value")
    span_gas_concentration: float = Field(..., description="Span gas concentration")

    # Response characteristics
    response_time_t90_seconds: float = Field(
        default=60.0,
        ge=0,
        description="T90 response time"
    )
    detection_limit: Optional[float] = Field(default=None, ge=0)
    repeatability_percent: Optional[float] = Field(default=None, ge=0, le=10)

    # Alarm settings
    alarm_high: Optional[float] = Field(default=None)
    alarm_low: Optional[float] = Field(default=None)
    alarm_deadband: float = Field(default=0.0, ge=0)


class CalibrationResult(BaseModel):
    """Result of a calibration check."""

    model_config = ConfigDict(frozen=True)

    calibration_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Calibration identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Calibration timestamp"
    )
    channel_id: str = Field(..., description="Channel ID")
    gas_component: GasComponent = Field(..., description="Gas component")
    calibration_mode: CalibrationMode = Field(..., description="Calibration mode")

    # Reference values
    reference_value: float = Field(..., description="Reference gas value")
    reference_unit: str = Field(..., description="Reference unit")

    # Measured values
    measured_value: float = Field(..., description="Measured value")
    error_absolute: float = Field(..., description="Absolute error")
    error_percent: float = Field(..., description="Percent error")

    # Pass/fail
    passed: bool = Field(..., description="Calibration passed")
    error_limit_percent: float = Field(..., description="Error limit")

    # Gas information
    cylinder_id: Optional[str] = Field(default=None, description="Gas cylinder ID")
    cylinder_expiration: Optional[datetime] = Field(default=None)

    # Adjustments
    adjustment_made: bool = Field(default=False, description="Adjustment was made")
    previous_factor: Optional[float] = Field(default=None)
    new_factor: Optional[float] = Field(default=None)


class AnalyzerAlarm(BaseModel):
    """Analyzer alarm record."""

    model_config = ConfigDict(frozen=True)

    alarm_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Alarm identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Alarm timestamp"
    )
    channel_id: str = Field(..., description="Channel ID")
    alarm_type: AlarmType = Field(..., description="Alarm type")
    severity: str = Field(default="warning", description="Alarm severity")
    message: str = Field(..., description="Alarm message")
    value: Optional[float] = Field(default=None, description="Triggering value")
    limit: Optional[float] = Field(default=None, description="Alarm limit")
    acknowledged: bool = Field(default=False, description="Alarm acknowledged")
    cleared: bool = Field(default=False, description="Alarm cleared")
    cleared_timestamp: Optional[datetime] = Field(default=None)


class AnalyzerDiagnostics(BaseModel):
    """Analyzer diagnostic information."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    analyzer_id: str = Field(..., description="Analyzer ID")
    status: AnalyzerStatus = Field(..., description="Analyzer status")

    # Operating parameters
    sample_flow_lpm: Optional[float] = Field(default=None, ge=0)
    sample_pressure_mbar: Optional[float] = Field(default=None)
    sample_temperature_c: Optional[float] = Field(default=None)
    cell_temperature_c: Optional[float] = Field(default=None)
    ambient_temperature_c: Optional[float] = Field(default=None)

    # Detector health
    detector_signal: Optional[float] = Field(default=None)
    detector_noise: Optional[float] = Field(default=None)
    lamp_intensity: Optional[float] = Field(default=None)
    lamp_hours: Optional[int] = Field(default=None, ge=0)

    # Calibration status
    last_zero_calibration: Optional[datetime] = Field(default=None)
    last_span_calibration: Optional[datetime] = Field(default=None)
    hours_since_calibration: Optional[float] = Field(default=None, ge=0)

    # Drift tracking
    zero_drift_percent: Optional[float] = Field(default=None)
    span_drift_percent: Optional[float] = Field(default=None)

    # Active alarms
    active_alarms: List[str] = Field(default_factory=list)


class StackAnalyzerConnectorConfig(BaseConnectorConfig):
    """Configuration for stack analyzer connector."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    connector_type: ConnectorType = Field(
        default=ConnectorType.STACK_ANALYZER,
        description="Connector type"
    )

    # Analyzer identification
    analyzer_id: str = Field(..., description="Analyzer identifier")
    analyzer_name: str = Field(..., description="Analyzer name")
    vendor: AnalyzerVendor = Field(..., description="Analyzer vendor")
    model: str = Field(..., description="Analyzer model")
    serial_number: Optional[str] = Field(default=None)

    # Location
    stack_id: str = Field(..., description="Stack identifier")
    unit_id: str = Field(..., description="Unit identifier")
    installation_type: str = Field(
        default="extractive",
        description="Installation type (extractive/in-situ)"
    )

    # Communication settings
    protocol: str = Field(default="modbus_tcp", description="Communication protocol")
    host: Optional[str] = Field(default=None, description="IP address")
    port: int = Field(default=502, ge=1, le=65535, description="Port")
    slave_id: int = Field(default=1, ge=1, le=247, description="Modbus slave ID")

    # Serial settings (for RS-232/485)
    serial_port: Optional[str] = Field(default=None)
    baudrate: int = Field(default=9600)
    parity: str = Field(default="N")

    # Channel configuration
    channels: List[AnalyzerChannelConfig] = Field(
        default_factory=list,
        description="Channel configurations"
    )

    # Sampling settings
    sample_rate_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Sample rate"
    )
    averaging_period_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Averaging period"
    )

    # Calibration settings
    auto_zero_interval_hours: Optional[float] = Field(default=24.0, ge=1)
    auto_span_interval_hours: Optional[float] = Field(default=24.0, ge=1)
    calibration_error_limit_percent: float = Field(
        default=2.5,
        ge=0.1,
        le=10.0,
        description="Calibration error limit"
    )

    # Validation settings
    range_validation_enabled: bool = Field(default=True)
    spike_detection_enabled: bool = Field(default=True)
    spike_threshold_percent: float = Field(default=50.0, ge=10, le=100)


# =============================================================================
# Analyzer Protocol Handlers
# =============================================================================


class AnalyzerProtocolHandler:
    """
    Abstract protocol handler for analyzer communication.

    Concrete implementations handle Modbus, serial, etc.
    """

    def __init__(self, config: StackAnalyzerConnectorConfig) -> None:
        """Initialize protocol handler."""
        self._config = config
        self._connected = False
        self._logger = logging.getLogger(f"analyzer.protocol.{config.analyzer_id}")

    async def connect(self) -> None:
        """Establish connection."""
        raise NotImplementedError

    async def disconnect(self) -> None:
        """Close connection."""
        raise NotImplementedError

    async def read_value(self, channel_id: str) -> float:
        """Read value from channel."""
        raise NotImplementedError

    async def read_all_channels(self) -> Dict[str, float]:
        """Read all channel values."""
        raise NotImplementedError

    async def read_diagnostics(self) -> Dict[str, Any]:
        """Read diagnostic data."""
        raise NotImplementedError

    async def send_command(self, command: str, params: Dict[str, Any]) -> Any:
        """Send command to analyzer."""
        raise NotImplementedError


class ModbusProtocolHandler(AnalyzerProtocolHandler):
    """Modbus protocol handler for analyzers."""

    # Register maps for common analyzers
    REGISTER_MAPS = {
        AnalyzerVendor.THERMO_FISHER: {
            "measurement": {"start": 0, "count": 10},
            "status": {"start": 100, "count": 5},
            "diagnostics": {"start": 200, "count": 20},
        },
        AnalyzerVendor.TELEDYNE: {
            "measurement": {"start": 40001, "count": 20},
            "status": {"start": 40100, "count": 10},
            "diagnostics": {"start": 40200, "count": 30},
        },
        AnalyzerVendor.HORIBA: {
            "measurement": {"start": 0, "count": 16},
            "status": {"start": 50, "count": 8},
            "diagnostics": {"start": 100, "count": 32},
        },
    }

    async def connect(self) -> None:
        """Establish Modbus connection."""
        try:
            self._logger.info(
                f"Connecting to analyzer via Modbus at "
                f"{self._config.host}:{self._config.port}"
            )

            # In production, use pymodbus AsyncModbusTcpClient
            self._connected = True
            self._logger.info("Modbus connection established")

        except Exception as e:
            self._logger.error(f"Modbus connection failed: {e}")
            raise ConnectionError(f"Failed to connect to analyzer: {e}")

    async def disconnect(self) -> None:
        """Close Modbus connection."""
        self._connected = False
        self._logger.info("Modbus connection closed")

    async def read_value(self, channel_id: str) -> float:
        """Read value from specific channel."""
        if not self._connected:
            raise ConnectionError("Not connected to analyzer")

        # Simulated read
        return 0.0

    async def read_all_channels(self) -> Dict[str, float]:
        """Read all channel values."""
        if not self._connected:
            raise ConnectionError("Not connected to analyzer")

        results: Dict[str, float] = {}
        for channel in self._config.channels:
            try:
                value = await self.read_value(channel.channel_id)
                results[channel.channel_id] = value
            except Exception as e:
                self._logger.warning(
                    f"Failed to read channel {channel.channel_id}: {e}"
                )

        return results

    async def read_diagnostics(self) -> Dict[str, Any]:
        """Read diagnostic registers."""
        if not self._connected:
            raise ConnectionError("Not connected to analyzer")

        # Simulated diagnostics
        return {
            "sample_flow": 1.5,
            "sample_pressure": 800,
            "sample_temperature": 25,
            "detector_signal": 95,
        }

    async def send_command(self, command: str, params: Dict[str, Any]) -> Any:
        """Send command via Modbus write."""
        if not self._connected:
            raise ConnectionError("Not connected to analyzer")

        self._logger.info(f"Sending command: {command} with params: {params}")
        return True


class SerialProtocolHandler(AnalyzerProtocolHandler):
    """Serial RS-232/485 protocol handler for analyzers."""

    # Command sets for common analyzers
    COMMAND_SETS = {
        AnalyzerVendor.SERVOMEX: {
            "read_value": "MV?\r\n",
            "read_status": "ST?\r\n",
            "zero_calibration": "CAL Z\r\n",
            "span_calibration": "CAL S\r\n",
        },
        AnalyzerVendor.HORIBA: {
            "read_value": "R\r\n",
            "read_status": "S\r\n",
            "zero_calibration": "ZC\r\n",
            "span_calibration": "SC\r\n",
        },
    }

    async def connect(self) -> None:
        """Establish serial connection."""
        try:
            self._logger.info(
                f"Connecting to analyzer via serial at {self._config.serial_port}"
            )

            # In production, use pyserial-asyncio
            self._connected = True
            self._logger.info("Serial connection established")

        except Exception as e:
            self._logger.error(f"Serial connection failed: {e}")
            raise ConnectionError(f"Failed to connect to analyzer: {e}")

    async def disconnect(self) -> None:
        """Close serial connection."""
        self._connected = False
        self._logger.info("Serial connection closed")

    async def read_value(self, channel_id: str) -> float:
        """Read value via serial command."""
        if not self._connected:
            raise ConnectionError("Not connected to analyzer")

        return 0.0

    async def read_all_channels(self) -> Dict[str, float]:
        """Read all channels via serial."""
        results: Dict[str, float] = {}
        for channel in self._config.channels:
            try:
                value = await self.read_value(channel.channel_id)
                results[channel.channel_id] = value
            except Exception as e:
                self._logger.warning(f"Failed to read channel {channel.channel_id}: {e}")

        return results

    async def read_diagnostics(self) -> Dict[str, Any]:
        """Read diagnostics via serial."""
        return {}

    async def send_command(self, command: str, params: Dict[str, Any]) -> Any:
        """Send serial command."""
        return True


# =============================================================================
# Data Validation
# =============================================================================


class AnalyzerDataValidator:
    """
    Validates analyzer readings for quality assurance.

    Implements range checking, spike detection, and drift monitoring.
    """

    def __init__(self, config: StackAnalyzerConnectorConfig) -> None:
        """
        Initialize validator.

        Args:
            config: Analyzer configuration
        """
        self._config = config
        self._previous_values: Dict[str, float] = {}
        self._logger = logging.getLogger(f"analyzer.validator.{config.analyzer_id}")

    def validate_reading(
        self,
        channel_id: str,
        value: float,
        channel_config: AnalyzerChannelConfig,
    ) -> Tuple[bool, str, float]:
        """
        Validate a single reading.

        Args:
            channel_id: Channel identifier
            value: Measured value
            channel_config: Channel configuration

        Returns:
            Tuple of (is_valid, quality_code, validated_value)
        """
        quality_code = "OK"
        is_valid = True
        validated_value = value

        # Range validation
        if self._config.range_validation_enabled:
            if value < channel_config.range_low:
                quality_code = "BELOW_RANGE"
                is_valid = False
            elif value > channel_config.range_high:
                quality_code = "ABOVE_RANGE"
                is_valid = False

        # Negative value check (for most gases)
        if value < 0 and channel_config.gas_component != GasComponent.O2:
            quality_code = "NEGATIVE"
            is_valid = False
            validated_value = 0.0

        # Spike detection
        if self._config.spike_detection_enabled and channel_id in self._previous_values:
            prev_value = self._previous_values[channel_id]
            if prev_value > 0:
                change_percent = abs(value - prev_value) / prev_value * 100
                if change_percent > self._config.spike_threshold_percent:
                    quality_code = "SPIKE_DETECTED"
                    self._logger.warning(
                        f"Spike detected on {channel_id}: "
                        f"{prev_value} -> {value} ({change_percent:.1f}%)"
                    )

        # Update previous value for next check
        self._previous_values[channel_id] = value

        return is_valid, quality_code, validated_value

    def check_alarm_conditions(
        self,
        channel_id: str,
        value: float,
        channel_config: AnalyzerChannelConfig,
    ) -> List[AnalyzerAlarm]:
        """
        Check for alarm conditions.

        Args:
            channel_id: Channel identifier
            value: Current value
            channel_config: Channel configuration

        Returns:
            List of triggered alarms
        """
        alarms: List[AnalyzerAlarm] = []

        # High alarm
        if channel_config.alarm_high is not None:
            if value > channel_config.alarm_high + channel_config.alarm_deadband:
                alarms.append(AnalyzerAlarm(
                    channel_id=channel_id,
                    alarm_type=AlarmType.RANGE_HIGH,
                    severity="warning",
                    message=f"High alarm: {value} > {channel_config.alarm_high}",
                    value=value,
                    limit=channel_config.alarm_high,
                ))

        # Low alarm
        if channel_config.alarm_low is not None:
            if value < channel_config.alarm_low - channel_config.alarm_deadband:
                alarms.append(AnalyzerAlarm(
                    channel_id=channel_id,
                    alarm_type=AlarmType.RANGE_LOW,
                    severity="warning",
                    message=f"Low alarm: {value} < {channel_config.alarm_low}",
                    value=value,
                    limit=channel_config.alarm_low,
                ))

        return alarms


# =============================================================================
# Calibration Manager
# =============================================================================


class AnalyzerCalibrationManager:
    """
    Manages analyzer calibration operations.

    Tracks calibration history, schedules automatic calibrations,
    and validates calibration results.
    """

    def __init__(self, config: StackAnalyzerConnectorConfig) -> None:
        """
        Initialize calibration manager.

        Args:
            config: Analyzer configuration
        """
        self._config = config
        self._calibration_history: Dict[str, List[CalibrationResult]] = {}
        self._last_zero: Dict[str, datetime] = {}
        self._last_span: Dict[str, datetime] = {}
        self._calibration_factors: Dict[str, Tuple[float, float]] = {}  # (zero, span)
        self._logger = logging.getLogger(f"analyzer.calibration.{config.analyzer_id}")

    def is_calibration_due(
        self,
        channel_id: str,
        calibration_type: CalibrationMode,
    ) -> bool:
        """
        Check if calibration is due.

        Args:
            channel_id: Channel identifier
            calibration_type: Type of calibration

        Returns:
            True if calibration is due
        """
        if calibration_type == CalibrationMode.ZERO:
            last_cal = self._last_zero.get(channel_id)
            interval = self._config.auto_zero_interval_hours
        elif calibration_type == CalibrationMode.SPAN:
            last_cal = self._last_span.get(channel_id)
            interval = self._config.auto_span_interval_hours
        else:
            return False

        if last_cal is None:
            return True

        if interval is None:
            return False

        hours_since = (datetime.utcnow() - last_cal).total_seconds() / 3600
        return hours_since >= interval

    def record_calibration(
        self,
        channel_id: str,
        result: CalibrationResult,
    ) -> None:
        """
        Record calibration result.

        Args:
            channel_id: Channel identifier
            result: Calibration result
        """
        if channel_id not in self._calibration_history:
            self._calibration_history[channel_id] = []

        self._calibration_history[channel_id].append(result)

        # Update last calibration times
        if result.calibration_mode == CalibrationMode.ZERO:
            self._last_zero[channel_id] = result.timestamp
        elif result.calibration_mode == CalibrationMode.SPAN:
            self._last_span[channel_id] = result.timestamp

        # Update calibration factors if adjustment was made
        if result.adjustment_made and result.new_factor is not None:
            if result.calibration_mode == CalibrationMode.ZERO:
                current = self._calibration_factors.get(channel_id, (0, 1))
                self._calibration_factors[channel_id] = (result.new_factor, current[1])
            elif result.calibration_mode == CalibrationMode.SPAN:
                current = self._calibration_factors.get(channel_id, (0, 1))
                self._calibration_factors[channel_id] = (current[0], result.new_factor)

        # Limit history size
        max_entries = 365 * 4
        if len(self._calibration_history[channel_id]) > max_entries:
            self._calibration_history[channel_id] = (
                self._calibration_history[channel_id][-max_entries:]
            )

        self._logger.info(
            f"Recorded {result.calibration_mode.value} calibration for {channel_id}: "
            f"{'PASSED' if result.passed else 'FAILED'}"
        )

    def calculate_calibration_error(
        self,
        reference_value: float,
        measured_value: float,
        channel_config: AnalyzerChannelConfig,
    ) -> Tuple[float, float]:
        """
        Calculate calibration error.

        Args:
            reference_value: Reference gas value
            measured_value: Measured value
            channel_config: Channel configuration

        Returns:
            Tuple of (absolute_error, percent_error)
        """
        absolute_error = measured_value - reference_value

        if reference_value > 0:
            # Percent of reference value
            percent_error = (absolute_error / reference_value) * 100
        else:
            # Percent of span for zero calibration
            span = channel_config.span_value
            percent_error = (absolute_error / span) * 100 if span > 0 else 0

        return absolute_error, percent_error

    def validate_calibration(
        self,
        result: CalibrationResult,
    ) -> bool:
        """
        Validate calibration result meets criteria.

        Args:
            result: Calibration result

        Returns:
            True if calibration passes
        """
        passes = abs(result.error_percent) <= self._config.calibration_error_limit_percent

        if not passes:
            self._logger.warning(
                f"Calibration validation failed: {result.error_percent:.2f}% > "
                f"{self._config.calibration_error_limit_percent}%"
            )

        return passes

    def get_drift_statistics(
        self,
        channel_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Calculate calibration drift statistics.

        Args:
            channel_id: Channel identifier
            days: Number of days to analyze

        Returns:
            Drift statistics
        """
        if channel_id not in self._calibration_history:
            return {}

        history = self._calibration_history[channel_id]
        cutoff = datetime.utcnow() - timedelta(days=days)

        zero_errors = []
        span_errors = []

        for cal in history:
            if cal.timestamp < cutoff:
                continue

            if cal.calibration_mode == CalibrationMode.ZERO:
                zero_errors.append(cal.error_percent)
            elif cal.calibration_mode == CalibrationMode.SPAN:
                span_errors.append(cal.error_percent)

        stats = {
            "period_days": days,
            "zero_calibrations": len(zero_errors),
            "span_calibrations": len(span_errors),
        }

        if zero_errors:
            stats["zero_drift_avg"] = sum(zero_errors) / len(zero_errors)
            stats["zero_drift_max"] = max(abs(e) for e in zero_errors)

        if span_errors:
            stats["span_drift_avg"] = sum(span_errors) / len(span_errors)
            stats["span_drift_max"] = max(abs(e) for e in span_errors)

        return stats


# =============================================================================
# Stack Analyzer Connector
# =============================================================================


class StackAnalyzerConnector(BaseConnector):
    """
    Stack Gas Analyzer Connector.

    Provides comprehensive integration with multi-gas stack analyzers:
    - Multi-technology support (NDIR, chemiluminescence, UV, zirconia)
    - Multi-vendor support (Thermo Fisher, Teledyne, Horiba, Siemens)
    - Real-time data acquisition
    - Automatic calibration management
    - Data quality validation
    - Alarm monitoring

    Features:
    - Multi-gas analysis (NOx, SO2, CO, CO2, O2, opacity)
    - Automatic zero/span checks
    - Drift monitoring and correction
    - EPA Part 75 compatible data quality
    """

    def __init__(self, config: StackAnalyzerConnectorConfig) -> None:
        """
        Initialize stack analyzer connector.

        Args:
            config: Connector configuration
        """
        super().__init__(config)
        self._analyzer_config = config

        # Initialize protocol handler based on configuration
        if config.protocol == "modbus_tcp":
            self._protocol_handler = ModbusProtocolHandler(config)
        else:
            self._protocol_handler = SerialProtocolHandler(config)

        # Initialize components
        self._data_validator = AnalyzerDataValidator(config)
        self._calibration_manager = AnalyzerCalibrationManager(config)

        # Channel configuration lookup
        self._channels: Dict[str, AnalyzerChannelConfig] = {
            ch.channel_id: ch for ch in config.channels
        }

        # Current readings
        self._current_readings: Dict[str, AnalyzerReading] = {}

        # Active alarms
        self._active_alarms: List[AnalyzerAlarm] = []

        # Polling
        self._polling_task: Optional[asyncio.Task] = None
        self._polling_active = False

        # Callbacks
        self._data_callbacks: List[Callable[[Dict[str, AnalyzerReading]], None]] = []
        self._alarm_callbacks: List[Callable[[AnalyzerAlarm], None]] = []

        self._logger = logging.getLogger(f"analyzer.connector.{config.analyzer_id}")

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Establish connection to analyzer.

        Raises:
            ConnectionError: If connection fails
        """
        self._state = ConnectionState.CONNECTING
        self._logger.info(f"Connecting to analyzer {self._analyzer_config.analyzer_name}")

        try:
            await self._protocol_handler.connect()
            self._state = ConnectionState.CONNECTED
            self._logger.info("Analyzer connection established")

            await self._audit_logger.log_operation(
                operation="connect",
                status="success",
                response_summary=f"Connected to {self._analyzer_config.vendor.value} analyzer",
            )

        except Exception as e:
            self._state = ConnectionState.ERROR
            await self._audit_logger.log_operation(
                operation="connect",
                status="failure",
                error_message=str(e),
            )
            raise ConnectionError(
                f"Failed to connect to analyzer: {e}",
                connector_id=self._config.connector_id,
            )

    async def disconnect(self) -> None:
        """Disconnect from analyzer."""
        self._logger.info("Disconnecting from analyzer")

        await self.stop_polling()
        await self._protocol_handler.disconnect()
        self._state = ConnectionState.DISCONNECTED

        await self._audit_logger.log_operation(
            operation="disconnect",
            status="success",
        )

    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check on analyzer.

        Returns:
            Health check result
        """
        start_time = time.time()

        try:
            # Read diagnostics
            diagnostics = await self._protocol_handler.read_diagnostics()

            latency_ms = (time.time() - start_time) * 1000

            # Check calibration status
            cal_due = {}
            for channel_id in self._channels:
                cal_due[channel_id] = {
                    "zero_due": self._calibration_manager.is_calibration_due(
                        channel_id, CalibrationMode.ZERO
                    ),
                    "span_due": self._calibration_manager.is_calibration_due(
                        channel_id, CalibrationMode.SPAN
                    ),
                }

            # Determine status
            status = HealthStatus.HEALTHY
            message = "Analyzer healthy"

            if self._active_alarms:
                status = HealthStatus.DEGRADED
                message = f"Analyzer has {len(self._active_alarms)} active alarms"

            if any(
                v["zero_due"] or v["span_due"]
                for v in cal_due.values()
            ):
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                    message = "Calibration due"

            return HealthCheckResult(
                status=status,
                latency_ms=latency_ms,
                message=message,
                details={
                    "diagnostics": diagnostics,
                    "calibration_status": cal_due,
                    "active_alarms": len(self._active_alarms),
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Health check failed: {e}",
            )

    async def validate_configuration(self) -> bool:
        """
        Validate analyzer configuration.

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        issues: List[str] = []

        if not self._analyzer_config.channels:
            issues.append("At least one channel must be configured")

        if self._analyzer_config.protocol == "modbus_tcp":
            if not self._analyzer_config.host:
                issues.append("Host is required for Modbus TCP")
        elif not self._analyzer_config.serial_port:
            issues.append("Serial port is required")

        for channel in self._analyzer_config.channels:
            if channel.range_high <= channel.range_low:
                issues.append(
                    f"Channel {channel.channel_id}: range_high must be > range_low"
                )

        if issues:
            raise ConfigurationError(
                f"Invalid analyzer configuration: {issues}",
                connector_id=self._config.connector_id,
            )

        return True

    # -------------------------------------------------------------------------
    # Analyzer-Specific Methods
    # -------------------------------------------------------------------------

    async def read_all_channels(self) -> Dict[str, AnalyzerReading]:
        """
        Read all analyzer channels.

        Returns:
            Dictionary of channel ID to reading
        """
        start_time = time.time()

        try:
            raw_values = await self._protocol_handler.read_all_channels()
            readings: Dict[str, AnalyzerReading] = {}

            for channel_id, raw_value in raw_values.items():
                channel_config = self._channels.get(channel_id)
                if not channel_config:
                    continue

                # Validate reading
                is_valid, quality_code, validated_value = self._data_validator.validate_reading(
                    channel_id,
                    raw_value,
                    channel_config,
                )

                # Create reading
                reading = AnalyzerReading(
                    gas_component=channel_config.gas_component,
                    value=validated_value,
                    unit=channel_config.unit,
                    is_valid=is_valid,
                    quality_code=quality_code,
                    raw_value=raw_value,
                    full_scale=channel_config.range_high,
                    percent_of_range=(
                        raw_value / channel_config.range_high * 100
                        if channel_config.range_high > 0 else 0
                    ),
                )

                readings[channel_id] = reading
                self._current_readings[channel_id] = reading

                # Check alarms
                alarms = self._data_validator.check_alarm_conditions(
                    channel_id,
                    validated_value,
                    channel_config,
                )
                for alarm in alarms:
                    await self._handle_alarm(alarm)

            duration_ms = (time.time() - start_time) * 1000
            await self._metrics.record_request(
                success=True,
                latency_ms=duration_ms,
            )

            return readings

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._metrics.record_request(
                success=False,
                latency_ms=duration_ms,
                error=str(e),
            )
            raise

    async def read_channel(self, channel_id: str) -> AnalyzerReading:
        """
        Read a specific channel.

        Args:
            channel_id: Channel identifier

        Returns:
            Channel reading
        """
        channel_config = self._channels.get(channel_id)
        if not channel_config:
            raise ValidationError(
                f"Unknown channel: {channel_id}",
                connector_id=self._config.connector_id,
            )

        raw_value = await self._protocol_handler.read_value(channel_id)

        is_valid, quality_code, validated_value = self._data_validator.validate_reading(
            channel_id,
            raw_value,
            channel_config,
        )

        return AnalyzerReading(
            gas_component=channel_config.gas_component,
            value=validated_value,
            unit=channel_config.unit,
            is_valid=is_valid,
            quality_code=quality_code,
            raw_value=raw_value,
            full_scale=channel_config.range_high,
        )

    async def perform_calibration(
        self,
        channel_id: str,
        calibration_mode: CalibrationMode,
        reference_value: float,
        cylinder_id: Optional[str] = None,
        auto_adjust: bool = False,
    ) -> CalibrationResult:
        """
        Perform calibration check.

        Args:
            channel_id: Channel identifier
            calibration_mode: Calibration mode
            reference_value: Reference gas value
            cylinder_id: Optional gas cylinder ID
            auto_adjust: Whether to auto-adjust on failure

        Returns:
            Calibration result
        """
        channel_config = self._channels.get(channel_id)
        if not channel_config:
            raise ValidationError(f"Unknown channel: {channel_id}")

        self._logger.info(
            f"Performing {calibration_mode.value} calibration on {channel_id}"
        )

        # Read current value with calibration gas applied
        # In practice, would wait for stabilization
        await asyncio.sleep(1)
        measured_value = await self._protocol_handler.read_value(channel_id)

        # Calculate error
        abs_error, pct_error = self._calibration_manager.calculate_calibration_error(
            reference_value,
            measured_value,
            channel_config,
        )

        # Check if passed
        passed = abs(pct_error) <= self._analyzer_config.calibration_error_limit_percent

        # Create result
        result = CalibrationResult(
            channel_id=channel_id,
            gas_component=channel_config.gas_component,
            calibration_mode=calibration_mode,
            reference_value=reference_value,
            reference_unit=channel_config.unit,
            measured_value=measured_value,
            error_absolute=abs_error,
            error_percent=pct_error,
            passed=passed,
            error_limit_percent=self._analyzer_config.calibration_error_limit_percent,
            cylinder_id=cylinder_id,
            adjustment_made=False,
        )

        # Record calibration
        self._calibration_manager.record_calibration(channel_id, result)

        await self._audit_logger.log_operation(
            operation="perform_calibration",
            status="success" if passed else "warning",
            request_data={
                "channel_id": channel_id,
                "mode": calibration_mode.value,
                "reference_value": reference_value,
            },
            response_summary=f"{'PASSED' if passed else 'FAILED'}: {pct_error:.2f}%",
        )

        return result

    async def get_diagnostics(self) -> AnalyzerDiagnostics:
        """
        Get analyzer diagnostic information.

        Returns:
            Diagnostics data
        """
        raw_diagnostics = await self._protocol_handler.read_diagnostics()

        # Build diagnostics model
        diagnostics = AnalyzerDiagnostics(
            analyzer_id=self._analyzer_config.analyzer_id,
            status=AnalyzerStatus.NORMAL if self.is_connected else AnalyzerStatus.OFFLINE,
            sample_flow_lpm=raw_diagnostics.get("sample_flow"),
            sample_pressure_mbar=raw_diagnostics.get("sample_pressure"),
            sample_temperature_c=raw_diagnostics.get("sample_temperature"),
            detector_signal=raw_diagnostics.get("detector_signal"),
            active_alarms=[a.alarm_type.value for a in self._active_alarms],
        )

        return diagnostics

    async def start_polling(
        self,
        callback: Optional[Callable[[Dict[str, AnalyzerReading]], None]] = None,
    ) -> None:
        """
        Start continuous data polling.

        Args:
            callback: Optional callback for new data
        """
        if self._polling_active:
            self._logger.warning("Polling already active")
            return

        if callback:
            self._data_callbacks.append(callback)

        self._polling_active = True
        self._polling_task = asyncio.create_task(self._polling_loop())

        self._logger.info(
            f"Started polling at {self._analyzer_config.sample_rate_seconds}s interval"
        )

    async def stop_polling(self) -> None:
        """Stop data polling."""
        self._polling_active = False

        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass

        self._logger.info("Stopped polling")

    async def _polling_loop(self) -> None:
        """Background polling loop."""
        while self._polling_active:
            try:
                readings = await self.read_all_channels()

                # Notify callbacks
                for callback in self._data_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(readings)
                        else:
                            callback(readings)
                    except Exception as e:
                        self._logger.error(f"Callback error: {e}")

                await asyncio.sleep(self._analyzer_config.sample_rate_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Polling error: {e}")
                await asyncio.sleep(self._analyzer_config.sample_rate_seconds)

    async def _handle_alarm(self, alarm: AnalyzerAlarm) -> None:
        """Handle new alarm."""
        # Check if alarm already active
        for existing in self._active_alarms:
            if (
                existing.channel_id == alarm.channel_id
                and existing.alarm_type == alarm.alarm_type
                and not existing.cleared
            ):
                return  # Already active

        self._active_alarms.append(alarm)
        self._logger.warning(f"Alarm triggered: {alarm.message}")

        # Notify callbacks
        for callback in self._alarm_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alarm)
                else:
                    callback(alarm)
            except Exception as e:
                self._logger.error(f"Alarm callback error: {e}")

    async def acknowledge_alarm(self, alarm_id: str) -> bool:
        """
        Acknowledge an alarm.

        Args:
            alarm_id: Alarm identifier

        Returns:
            True if alarm was acknowledged
        """
        for alarm in self._active_alarms:
            if alarm.alarm_id == alarm_id:
                # Create new alarm with acknowledged flag
                # (Pydantic models are frozen)
                idx = self._active_alarms.index(alarm)
                self._active_alarms[idx] = AnalyzerAlarm(
                    alarm_id=alarm.alarm_id,
                    timestamp=alarm.timestamp,
                    channel_id=alarm.channel_id,
                    alarm_type=alarm.alarm_type,
                    severity=alarm.severity,
                    message=alarm.message,
                    value=alarm.value,
                    limit=alarm.limit,
                    acknowledged=True,
                    cleared=alarm.cleared,
                )
                return True

        return False

    def get_active_alarms(self) -> List[AnalyzerAlarm]:
        """Get list of active alarms."""
        return [a for a in self._active_alarms if not a.cleared]

    def register_data_callback(
        self,
        callback: Callable[[Dict[str, AnalyzerReading]], None],
    ) -> None:
        """Register data callback."""
        self._data_callbacks.append(callback)

    def register_alarm_callback(
        self,
        callback: Callable[[AnalyzerAlarm], None],
    ) -> None:
        """Register alarm callback."""
        self._alarm_callbacks.append(callback)


# =============================================================================
# Factory Function
# =============================================================================


def create_stack_analyzer_connector(
    analyzer_id: str,
    analyzer_name: str,
    vendor: AnalyzerVendor,
    model: str,
    stack_id: str,
    unit_id: str,
    host: str,
    channels: List[AnalyzerChannelConfig],
    **kwargs: Any,
) -> StackAnalyzerConnector:
    """
    Factory function to create stack analyzer connector.

    Args:
        analyzer_id: Analyzer identifier
        analyzer_name: Analyzer name
        vendor: Analyzer vendor
        model: Analyzer model
        stack_id: Stack identifier
        unit_id: Unit identifier
        host: IP address
        channels: Channel configurations
        **kwargs: Additional configuration

    Returns:
        Configured analyzer connector
    """
    config = StackAnalyzerConnectorConfig(
        connector_name=f"StackAnalyzer_{analyzer_id}",
        analyzer_id=analyzer_id,
        analyzer_name=analyzer_name,
        vendor=vendor,
        model=model,
        stack_id=stack_id,
        unit_id=unit_id,
        host=host,
        channels=channels,
        **kwargs,
    )

    return StackAnalyzerConnector(config)
