# -*- coding: utf-8 -*-
"""
Steam Quality Meter Connector for GL-012 STEAMQUAL (SteamQualityController).

Provides integration with steam quality measurement devices including:
- Vortex flow meters
- Orifice plate flow meters
- Ultrasonic flow meters

Supports multiple industrial protocols:
- Modbus TCP/RTU
- OPC-UA
- HART

Features:
- Real-time steam quality parameter reading
- Pressure, temperature, and flow rate measurement
- Meter calibration support
- Data quality validation
- Connection pooling and retry logic

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

import asyncio
import logging
import time
import struct
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, ConfigDict

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    ConnectionState,
    ConnectorType,
    ProtocolType,
    HealthStatus,
    HealthCheckResult,
    ConnectionError,
    ConfigurationError,
    CommunicationError,
    CalibrationError,
    DataQualityError,
    with_retry,
    CircuitState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class MeterType(str, Enum):
    """Types of steam quality meters supported."""

    VORTEX_FLOW = "vortex_flow"
    ORIFICE_PLATE = "orifice_plate"
    ULTRASONIC = "ultrasonic"
    CORIOLIS = "coriolis"
    THERMAL_MASS = "thermal_mass"
    DIFFERENTIAL_PRESSURE = "differential_pressure"


class MeterVendor(str, Enum):
    """Supported meter vendors."""

    EMERSON = "emerson"
    ENDRESS_HAUSER = "endress_hauser"
    YOKOGAWA = "yokogawa"
    ABB = "abb"
    SIEMENS = "siemens"
    HONEYWELL = "honeywell"
    KROHNE = "krohne"


class MeterStatus(str, Enum):
    """Meter operational status."""

    OPERATIONAL = "operational"
    CALIBRATING = "calibrating"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    COMMUNICATION_LOST = "communication_lost"
    OUT_OF_RANGE = "out_of_range"
    LOW_SIGNAL = "low_signal"


class CalibrationStatus(str, Enum):
    """Calibration status enumeration."""

    VALID = "valid"
    EXPIRED = "expired"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    REQUIRED = "required"


class QualityFlag(str, Enum):
    """Data quality flags."""

    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    SUBSTITUTED = "substituted"
    CALCULATED = "calculated"


# =============================================================================
# Data Models
# =============================================================================


class SteamQualityMeterConfig(BaseConnectorConfig):
    """Configuration for steam quality meter connector."""

    model_config = ConfigDict(extra="forbid")

    connector_type: ConnectorType = Field(
        default=ConnectorType.STEAM_QUALITY_METER,
        frozen=True
    )

    # Meter identification
    meter_id: str = Field(
        ...,
        description="Unique meter identifier"
    )
    meter_type: MeterType = Field(
        ...,
        description="Type of flow meter"
    )
    meter_vendor: MeterVendor = Field(
        default=MeterVendor.EMERSON,
        description="Meter vendor"
    )
    meter_model: str = Field(
        default="",
        description="Meter model number"
    )

    # Modbus settings
    slave_address: int = Field(
        default=1,
        ge=1,
        le=247,
        description="Modbus slave address"
    )
    register_base_address: int = Field(
        default=0,
        ge=0,
        description="Base register address"
    )
    byte_order: str = Field(
        default="big",
        pattern="^(big|little)$",
        description="Byte order for register reading"
    )
    word_order: str = Field(
        default="big",
        pattern="^(big|little)$",
        description="Word order for 32-bit values"
    )

    # OPC-UA settings
    opc_namespace: str = Field(
        default="ns=2",
        description="OPC-UA namespace"
    )
    opc_node_id_base: str = Field(
        default="",
        description="OPC-UA base node ID"
    )

    # HART settings
    hart_address: int = Field(
        default=0,
        ge=0,
        le=63,
        description="HART device address"
    )

    # Measurement ranges
    pressure_min_bar: float = Field(
        default=0.0,
        description="Minimum pressure in bar"
    )
    pressure_max_bar: float = Field(
        default=50.0,
        description="Maximum pressure in bar"
    )
    temperature_min_c: float = Field(
        default=0.0,
        description="Minimum temperature in Celsius"
    )
    temperature_max_c: float = Field(
        default=400.0,
        description="Maximum temperature in Celsius"
    )
    flow_min_kg_hr: float = Field(
        default=0.0,
        description="Minimum flow rate in kg/hr"
    )
    flow_max_kg_hr: float = Field(
        default=100000.0,
        description="Maximum flow rate in kg/hr"
    )

    # Quality thresholds
    dryness_fraction_min: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable dryness fraction"
    )
    dryness_fraction_max: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable dryness fraction"
    )

    # Sampling
    sampling_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Data sampling interval"
    )
    averaging_window_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Averaging window for statistics"
    )

    # Calibration
    calibration_due_days: int = Field(
        default=365,
        ge=1,
        description="Days between calibrations"
    )
    last_calibration_date: Optional[datetime] = Field(
        default=None,
        description="Last calibration date"
    )


class SteamQualityReading(BaseModel):
    """Steam quality measurement reading."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Reading timestamp"
    )
    meter_id: str = Field(
        ...,
        description="Meter identifier"
    )

    # Primary quality parameters
    dryness_fraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Steam dryness fraction (0-1, 1=dry saturated)"
    )
    wetness_fraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Steam wetness fraction (0-1)"
    )
    specific_enthalpy_kj_kg: float = Field(
        ...,
        description="Specific enthalpy in kJ/kg"
    )
    specific_entropy_kj_kg_k: Optional[float] = Field(
        default=None,
        description="Specific entropy in kJ/(kg*K)"
    )

    # Process parameters
    pressure_bar: float = Field(
        ...,
        description="Pressure in bar absolute"
    )
    temperature_c: float = Field(
        ...,
        description="Temperature in Celsius"
    )
    saturation_temperature_c: Optional[float] = Field(
        default=None,
        description="Saturation temperature at current pressure"
    )
    superheat_c: Optional[float] = Field(
        default=None,
        description="Degrees of superheat (if superheated)"
    )

    # Flow parameters
    mass_flow_kg_hr: float = Field(
        ...,
        description="Mass flow rate in kg/hr"
    )
    volumetric_flow_m3_hr: Optional[float] = Field(
        default=None,
        description="Volumetric flow rate in m3/hr"
    )
    velocity_m_s: Optional[float] = Field(
        default=None,
        description="Flow velocity in m/s"
    )

    # Density
    density_kg_m3: Optional[float] = Field(
        default=None,
        description="Steam density in kg/m3"
    )

    # Energy flow
    energy_flow_kw: Optional[float] = Field(
        default=None,
        description="Energy flow rate in kW"
    )

    # Quality flags
    quality_flag: QualityFlag = Field(
        default=QualityFlag.GOOD,
        description="Data quality flag"
    )
    quality_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Quality score (0-100)"
    )

    # Raw values for diagnostics
    raw_pressure: Optional[float] = Field(
        default=None,
        description="Raw pressure reading"
    )
    raw_temperature: Optional[float] = Field(
        default=None,
        description="Raw temperature reading"
    )
    raw_flow: Optional[float] = Field(
        default=None,
        description="Raw flow reading"
    )


class PressureReading(BaseModel):
    """Pressure measurement reading."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Reading timestamp"
    )
    meter_id: str = Field(
        ...,
        description="Meter identifier"
    )
    pressure_bar: float = Field(
        ...,
        description="Pressure in bar absolute"
    )
    pressure_bar_gauge: Optional[float] = Field(
        default=None,
        description="Gauge pressure in bar"
    )
    quality_flag: QualityFlag = Field(
        default=QualityFlag.GOOD,
        description="Data quality flag"
    )
    raw_value: Optional[float] = Field(
        default=None,
        description="Raw sensor value"
    )
    unit: str = Field(
        default="bar",
        description="Pressure unit"
    )


class TemperatureReading(BaseModel):
    """Temperature measurement reading."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Reading timestamp"
    )
    meter_id: str = Field(
        ...,
        description="Meter identifier"
    )
    temperature_c: float = Field(
        ...,
        description="Temperature in Celsius"
    )
    temperature_k: Optional[float] = Field(
        default=None,
        description="Temperature in Kelvin"
    )
    quality_flag: QualityFlag = Field(
        default=QualityFlag.GOOD,
        description="Data quality flag"
    )
    raw_value: Optional[float] = Field(
        default=None,
        description="Raw sensor value"
    )
    sensor_type: str = Field(
        default="RTD",
        description="Temperature sensor type (RTD, TC, etc.)"
    )


class FlowReading(BaseModel):
    """Flow measurement reading."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Reading timestamp"
    )
    meter_id: str = Field(
        ...,
        description="Meter identifier"
    )
    mass_flow_kg_hr: float = Field(
        ...,
        description="Mass flow rate in kg/hr"
    )
    volumetric_flow_m3_hr: Optional[float] = Field(
        default=None,
        description="Volumetric flow rate in m3/hr"
    )
    velocity_m_s: Optional[float] = Field(
        default=None,
        description="Flow velocity in m/s"
    )
    totalizer_kg: Optional[float] = Field(
        default=None,
        description="Totalizer reading in kg"
    )
    quality_flag: QualityFlag = Field(
        default=QualityFlag.GOOD,
        description="Data quality flag"
    )
    raw_value: Optional[float] = Field(
        default=None,
        description="Raw flow value"
    )


class MeterStatusReading(BaseModel):
    """Meter status information."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Status timestamp"
    )
    meter_id: str = Field(
        ...,
        description="Meter identifier"
    )
    status: MeterStatus = Field(
        ...,
        description="Current meter status"
    )
    calibration_status: CalibrationStatus = Field(
        ...,
        description="Calibration status"
    )
    last_calibration: Optional[datetime] = Field(
        default=None,
        description="Last calibration timestamp"
    )
    next_calibration_due: Optional[datetime] = Field(
        default=None,
        description="Next calibration due date"
    )
    signal_strength_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Signal strength percentage"
    )
    diagnostics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic information"
    )
    alarms: List[str] = Field(
        default_factory=list,
        description="Active alarms"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Active warnings"
    )


class CalibrationData(BaseModel):
    """Calibration input data."""

    model_config = ConfigDict(extra="forbid")

    reference_pressure_bar: Optional[float] = Field(
        default=None,
        description="Reference pressure for calibration"
    )
    reference_temperature_c: Optional[float] = Field(
        default=None,
        description="Reference temperature for calibration"
    )
    reference_flow_kg_hr: Optional[float] = Field(
        default=None,
        description="Reference flow rate for calibration"
    )
    zero_point: bool = Field(
        default=False,
        description="Perform zero-point calibration"
    )
    span_adjustment: Optional[float] = Field(
        default=None,
        description="Span adjustment factor"
    )
    operator_id: str = Field(
        ...,
        description="Operator performing calibration"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Calibration notes"
    )


class CalibrationResult(BaseModel):
    """Calibration operation result."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Calibration timestamp"
    )
    meter_id: str = Field(
        ...,
        description="Meter identifier"
    )
    success: bool = Field(
        ...,
        description="Whether calibration succeeded"
    )
    calibration_type: str = Field(
        ...,
        description="Type of calibration performed"
    )
    before_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Values before calibration"
    )
    after_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Values after calibration"
    )
    deviation_percent: Optional[float] = Field(
        default=None,
        description="Deviation from reference"
    )
    next_calibration_due: datetime = Field(
        ...,
        description="Next calibration due date"
    )
    certificate_number: Optional[str] = Field(
        default=None,
        description="Calibration certificate number"
    )
    operator_id: str = Field(
        ...,
        description="Operator who performed calibration"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )


# =============================================================================
# Register Maps
# =============================================================================


@dataclass
class ModbusRegisterMap:
    """Modbus register mapping for steam quality meters."""

    # Flow registers
    flow_rate: int = 0  # 32-bit float
    flow_totalizer: int = 2  # 32-bit float

    # Pressure registers
    pressure: int = 4  # 32-bit float

    # Temperature registers
    temperature: int = 6  # 32-bit float

    # Quality registers
    dryness_fraction: int = 8  # 32-bit float
    specific_enthalpy: int = 10  # 32-bit float

    # Density and velocity
    density: int = 12  # 32-bit float
    velocity: int = 14  # 32-bit float

    # Status registers
    status: int = 100  # 16-bit status word
    alarms: int = 101  # 16-bit alarm word

    # Calibration registers
    calibration_status: int = 110
    last_calibration_timestamp: int = 112  # 32-bit epoch


@dataclass
class OPCUANodeMap:
    """OPC-UA node mapping for steam quality meters."""

    flow_rate: str = "FlowRate"
    flow_totalizer: str = "Totalizer"
    pressure: str = "Pressure"
    temperature: str = "Temperature"
    dryness_fraction: str = "DrynessFraction"
    specific_enthalpy: str = "SpecificEnthalpy"
    density: str = "Density"
    velocity: str = "Velocity"
    status: str = "Status"
    alarms: str = "Alarms"


# =============================================================================
# Steam Quality Meter Connector
# =============================================================================


class SteamQualityMeterConnector(BaseConnector):
    """
    Connector for steam quality measurement devices.

    Supports multiple meter types and communication protocols for
    comprehensive steam quality monitoring.

    Features:
    - Multi-protocol support (Modbus TCP, OPC-UA, HART)
    - Real-time steam quality parameter reading
    - Data quality validation and scoring
    - Calibration management
    - Historical data buffering
    - Connection pooling and retry logic
    """

    def __init__(self, config: SteamQualityMeterConfig) -> None:
        """
        Initialize steam quality meter connector.

        Args:
            config: Meter configuration
        """
        super().__init__(config)
        self.meter_config: SteamQualityMeterConfig = config

        # Register maps
        self._modbus_register_map = ModbusRegisterMap()
        self._opcua_node_map = OPCUANodeMap()

        # Data buffers
        self._reading_history: deque = deque(maxlen=10000)
        self._current_reading: Optional[SteamQualityReading] = None

        # Sampling task
        self._sampling_task: Optional[asyncio.Task] = None

        # Protocol handlers
        self._protocol_handlers: Dict[ProtocolType, Callable] = {
            ProtocolType.MODBUS_TCP: self._read_modbus_tcp,
            ProtocolType.MODBUS_RTU: self._read_modbus_rtu,
            ProtocolType.OPC_UA: self._read_opc_ua,
            ProtocolType.HART: self._read_hart,
        }

        # Connection state
        self._modbus_client: Optional[Any] = None
        self._opcua_client: Optional[Any] = None
        self._hart_client: Optional[Any] = None

        self._logger.info(
            f"Initialized SteamQualityMeterConnector for meter {config.meter_id} "
            f"(type={config.meter_type.value}, protocol={config.protocol.value})"
        )

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self, meter_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Establish connection to the steam quality meter.

        Args:
            meter_config: Optional additional configuration

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        if self._state == ConnectionState.CONNECTED:
            self._logger.warning(f"Already connected to meter {self.meter_config.meter_id}")
            return True

        self._state = ConnectionState.CONNECTING

        try:
            # Apply any additional config
            if meter_config:
                for key, value in meter_config.items():
                    if hasattr(self.meter_config, key):
                        setattr(self.meter_config, key, value)

            # Connect based on protocol
            if self.meter_config.protocol == ProtocolType.MODBUS_TCP:
                await self._connect_modbus_tcp()
            elif self.meter_config.protocol == ProtocolType.MODBUS_RTU:
                await self._connect_modbus_rtu()
            elif self.meter_config.protocol == ProtocolType.OPC_UA:
                await self._connect_opc_ua()
            elif self.meter_config.protocol == ProtocolType.HART:
                await self._connect_hart()
            else:
                raise ConfigurationError(
                    f"Unsupported protocol: {self.meter_config.protocol}",
                    connector_id=self._config.connector_id,
                    connector_type=self._config.connector_type,
                )

            self._state = ConnectionState.CONNECTED

            # Start sampling task
            self._sampling_task = asyncio.create_task(self._sampling_loop())

            self._logger.info(
                f"Connected to meter {self.meter_config.meter_id} via "
                f"{self.meter_config.protocol.value}"
            )

            # Log audit entry
            await self._audit_logger.log_operation(
                operation="connect",
                status="success",
                response_summary=f"Connected to {self.meter_config.meter_id}",
            )

            return True

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._logger.error(f"Failed to connect to meter {self.meter_config.meter_id}: {e}")

            await self._audit_logger.log_operation(
                operation="connect",
                status="failure",
                error_message=str(e),
            )

            raise ConnectionError(
                f"Failed to connect to meter: {e}",
                connector_id=self._config.connector_id,
                connector_type=self._config.connector_type,
                details={"meter_id": self.meter_config.meter_id},
            )

    async def _connect_modbus_tcp(self) -> None:
        """Establish Modbus TCP connection."""
        # In production, would use pymodbus library
        # from pymodbus.client import AsyncModbusTcpClient
        # self._modbus_client = AsyncModbusTcpClient(
        #     host=self.meter_config.host,
        #     port=self.meter_config.port,
        #     timeout=self.meter_config.connection_timeout_seconds,
        # )
        # await self._modbus_client.connect()

        # Simulation for development
        self._modbus_client = {
            "type": "modbus_tcp",
            "host": self.meter_config.host,
            "port": self.meter_config.port,
            "connected": True,
            "slave_address": self.meter_config.slave_address,
        }

        self._logger.debug(
            f"Modbus TCP connection established to {self.meter_config.host}:"
            f"{self.meter_config.port}"
        )

    async def _connect_modbus_rtu(self) -> None:
        """Establish Modbus RTU connection."""
        # In production, would use pymodbus library with serial
        self._modbus_client = {
            "type": "modbus_rtu",
            "port": self.meter_config.host,
            "connected": True,
            "slave_address": self.meter_config.slave_address,
        }

        self._logger.debug(f"Modbus RTU connection established on {self.meter_config.host}")

    async def _connect_opc_ua(self) -> None:
        """Establish OPC-UA connection."""
        # In production, would use asyncua library
        # from asyncua import Client as OPCUAClient
        # self._opcua_client = OPCUAClient(
        #     f"opc.tcp://{self.meter_config.host}:{self.meter_config.port}"
        # )
        # await self._opcua_client.connect()

        self._opcua_client = {
            "type": "opc_ua",
            "url": f"opc.tcp://{self.meter_config.host}:{self.meter_config.port}",
            "connected": True,
            "namespace": self.meter_config.opc_namespace,
        }

        self._logger.debug(
            f"OPC-UA connection established to {self.meter_config.host}:"
            f"{self.meter_config.port}"
        )

    async def _connect_hart(self) -> None:
        """Establish HART protocol connection."""
        # HART typically uses serial/modem communication
        self._hart_client = {
            "type": "hart",
            "address": self.meter_config.hart_address,
            "connected": True,
        }

        self._logger.debug(
            f"HART connection established to address {self.meter_config.hart_address}"
        )

    async def disconnect(self) -> None:
        """Disconnect from the steam quality meter."""
        self._logger.info(f"Disconnecting from meter {self.meter_config.meter_id}")

        # Cancel sampling task
        if self._sampling_task:
            self._sampling_task.cancel()
            try:
                await self._sampling_task
            except asyncio.CancelledError:
                pass
            self._sampling_task = None

        # Close protocol-specific connections
        if self._modbus_client:
            # In production: await self._modbus_client.close()
            self._modbus_client = None

        if self._opcua_client:
            # In production: await self._opcua_client.disconnect()
            self._opcua_client = None

        if self._hart_client:
            self._hart_client = None

        self._state = ConnectionState.DISCONNECTED

        await self._audit_logger.log_operation(
            operation="disconnect",
            status="success",
        )

    async def validate_configuration(self) -> bool:
        """
        Validate meter configuration.

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate meter type
        if self.meter_config.meter_type not in MeterType:
            raise ConfigurationError(
                f"Invalid meter type: {self.meter_config.meter_type}",
                connector_id=self._config.connector_id,
            )

        # Validate protocol
        if self.meter_config.protocol not in self._protocol_handlers:
            raise ConfigurationError(
                f"Unsupported protocol: {self.meter_config.protocol}",
                connector_id=self._config.connector_id,
            )

        # Validate ranges
        if self.meter_config.pressure_min_bar >= self.meter_config.pressure_max_bar:
            raise ConfigurationError(
                "Pressure min must be less than max",
                connector_id=self._config.connector_id,
            )

        if self.meter_config.temperature_min_c >= self.meter_config.temperature_max_c:
            raise ConfigurationError(
                "Temperature min must be less than max",
                connector_id=self._config.connector_id,
            )

        if self.meter_config.flow_min_kg_hr >= self.meter_config.flow_max_kg_hr:
            raise ConfigurationError(
                "Flow min must be less than max",
                connector_id=self._config.connector_id,
            )

        self._logger.debug("Configuration validated successfully")
        return True

    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check on the meter connection.

        Returns:
            Health check result
        """
        start_time = time.time()

        try:
            if self._state != ConnectionState.CONNECTED:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Meter not connected",
                    latency_ms=0.0,
                )

            # Try to read a simple value
            status = await self.get_meter_status()
            latency_ms = (time.time() - start_time) * 1000

            if status.status == MeterStatus.OPERATIONAL:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency_ms,
                    message="Meter operational",
                    details={
                        "meter_status": status.status.value,
                        "calibration_status": status.calibration_status.value,
                        "signal_strength": status.signal_strength_percent,
                    },
                )
            elif status.status in (MeterStatus.MAINTENANCE, MeterStatus.CALIBRATING):
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency_ms,
                    message=f"Meter in {status.status.value} mode",
                    details={"meter_status": status.status.value},
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=latency_ms,
                    message=f"Meter status: {status.status.value}",
                    details={
                        "meter_status": status.status.value,
                        "alarms": status.alarms,
                    },
                )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Health check failed: {e}",
            )

    # -------------------------------------------------------------------------
    # Sampling Loop
    # -------------------------------------------------------------------------

    async def _sampling_loop(self) -> None:
        """Continuous sampling loop for meter data."""
        interval = self.meter_config.sampling_interval_seconds

        while self._state == ConnectionState.CONNECTED:
            try:
                reading = await self.read_quality_parameters()
                if reading:
                    self._current_reading = reading
                    self._reading_history.append(reading)

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Sampling error: {e}")
                await asyncio.sleep(interval)

    # -------------------------------------------------------------------------
    # Reading Methods
    # -------------------------------------------------------------------------

    @with_retry(max_retries=3, base_delay=0.5)
    async def read_quality_parameters(self) -> SteamQualityReading:
        """
        Read all steam quality parameters from the meter.

        Returns:
            Complete steam quality reading

        Raises:
            CommunicationError: If reading fails
        """
        if self._state != ConnectionState.CONNECTED:
            raise CommunicationError(
                "Not connected to meter",
                connector_id=self._config.connector_id,
            )

        handler = self._protocol_handlers.get(self.meter_config.protocol)
        if not handler:
            raise ConfigurationError(
                f"No handler for protocol: {self.meter_config.protocol}",
                connector_id=self._config.connector_id,
            )

        reading = await self.execute_with_circuit_breaker(handler)

        # Validate reading quality
        reading = self._validate_reading(reading)

        return reading

    async def _read_modbus_tcp(self) -> SteamQualityReading:
        """Read steam quality parameters via Modbus TCP."""
        # In production, would use actual Modbus reads
        # result = await self._modbus_client.read_holding_registers(
        #     address=self._modbus_register_map.flow_rate,
        #     count=16,
        #     slave=self.meter_config.slave_address,
        # )

        # Simulate realistic readings for development
        import random

        # Base values with realistic variation
        base_pressure = 10.0 + random.uniform(-0.5, 0.5)
        base_temp = 184.0 + random.uniform(-2.0, 2.0)  # Saturation temp ~184C at 10 bar
        base_flow = 5000.0 + random.uniform(-200, 200)

        # Calculate dryness fraction (typically 0.9-1.0 for quality steam)
        dryness = 0.95 + random.uniform(-0.03, 0.03)
        dryness = max(0.0, min(1.0, dryness))

        # Calculate enthalpy based on dryness fraction
        # At 10 bar: hf = 762.8 kJ/kg, hfg = 2015.3 kJ/kg
        hf = 762.8
        hfg = 2015.3
        specific_enthalpy = hf + dryness * hfg

        # Calculate density (approximate for wet steam)
        density = 5.0 + random.uniform(-0.3, 0.3)  # kg/m3 at ~10 bar

        # Calculate velocity
        velocity = 25.0 + random.uniform(-3, 3)  # m/s

        # Energy flow
        energy_flow_kw = (base_flow / 3600) * specific_enthalpy

        return SteamQualityReading(
            timestamp=datetime.utcnow(),
            meter_id=self.meter_config.meter_id,
            dryness_fraction=dryness,
            wetness_fraction=1.0 - dryness,
            specific_enthalpy_kj_kg=specific_enthalpy,
            specific_entropy_kj_kg_k=6.5 + random.uniform(-0.1, 0.1),
            pressure_bar=base_pressure,
            temperature_c=base_temp,
            saturation_temperature_c=183.9,  # Saturation temp at 10 bar
            superheat_c=None if dryness < 1.0 else base_temp - 183.9,
            mass_flow_kg_hr=base_flow,
            volumetric_flow_m3_hr=base_flow / density,
            velocity_m_s=velocity,
            density_kg_m3=density,
            energy_flow_kw=energy_flow_kw,
            quality_flag=QualityFlag.GOOD,
            quality_score=100.0,
            raw_pressure=base_pressure,
            raw_temperature=base_temp,
            raw_flow=base_flow,
        )

    async def _read_modbus_rtu(self) -> SteamQualityReading:
        """Read steam quality parameters via Modbus RTU."""
        # Similar to TCP but over serial
        return await self._read_modbus_tcp()

    async def _read_opc_ua(self) -> SteamQualityReading:
        """Read steam quality parameters via OPC-UA."""
        # In production, would use asyncua library
        # node = self._opcua_client.get_node(
        #     f"{self.meter_config.opc_namespace};s={self._opcua_node_map.flow_rate}"
        # )
        # value = await node.read_value()

        return await self._read_modbus_tcp()

    async def _read_hart(self) -> SteamQualityReading:
        """Read steam quality parameters via HART protocol."""
        # HART communication would use primary variable commands
        return await self._read_modbus_tcp()

    @with_retry(max_retries=3, base_delay=0.5)
    async def read_pressure(self) -> PressureReading:
        """
        Read current pressure from the meter.

        Returns:
            Pressure reading
        """
        if self._state != ConnectionState.CONNECTED:
            raise CommunicationError(
                "Not connected to meter",
                connector_id=self._config.connector_id,
            )

        # Read full parameters and extract pressure
        quality_reading = await self.read_quality_parameters()

        return PressureReading(
            timestamp=quality_reading.timestamp,
            meter_id=self.meter_config.meter_id,
            pressure_bar=quality_reading.pressure_bar,
            pressure_bar_gauge=quality_reading.pressure_bar - 1.01325,  # Gauge = Abs - Atm
            quality_flag=quality_reading.quality_flag,
            raw_value=quality_reading.raw_pressure,
        )

    @with_retry(max_retries=3, base_delay=0.5)
    async def read_temperature(self) -> TemperatureReading:
        """
        Read current temperature from the meter.

        Returns:
            Temperature reading
        """
        if self._state != ConnectionState.CONNECTED:
            raise CommunicationError(
                "Not connected to meter",
                connector_id=self._config.connector_id,
            )

        quality_reading = await self.read_quality_parameters()

        return TemperatureReading(
            timestamp=quality_reading.timestamp,
            meter_id=self.meter_config.meter_id,
            temperature_c=quality_reading.temperature_c,
            temperature_k=quality_reading.temperature_c + 273.15,
            quality_flag=quality_reading.quality_flag,
            raw_value=quality_reading.raw_temperature,
        )

    @with_retry(max_retries=3, base_delay=0.5)
    async def read_flow_rate(self) -> FlowReading:
        """
        Read current flow rate from the meter.

        Returns:
            Flow reading
        """
        if self._state != ConnectionState.CONNECTED:
            raise CommunicationError(
                "Not connected to meter",
                connector_id=self._config.connector_id,
            )

        quality_reading = await self.read_quality_parameters()

        # Get totalizer (would be separate register read in production)
        import random
        totalizer = random.uniform(1000000, 2000000)

        return FlowReading(
            timestamp=quality_reading.timestamp,
            meter_id=self.meter_config.meter_id,
            mass_flow_kg_hr=quality_reading.mass_flow_kg_hr,
            volumetric_flow_m3_hr=quality_reading.volumetric_flow_m3_hr,
            velocity_m_s=quality_reading.velocity_m_s,
            totalizer_kg=totalizer,
            quality_flag=quality_reading.quality_flag,
            raw_value=quality_reading.raw_flow,
        )

    async def get_meter_status(self) -> MeterStatusReading:
        """
        Get current meter status and diagnostics.

        Returns:
            Meter status information
        """
        if self._state != ConnectionState.CONNECTED:
            return MeterStatusReading(
                meter_id=self.meter_config.meter_id,
                status=MeterStatus.COMMUNICATION_LOST,
                calibration_status=CalibrationStatus.REQUIRED,
            )

        # In production, would read status registers
        # Simulate status
        import random

        # Check calibration status
        if self.meter_config.last_calibration_date:
            days_since_cal = (datetime.utcnow() - self.meter_config.last_calibration_date).days
            if days_since_cal > self.meter_config.calibration_due_days:
                cal_status = CalibrationStatus.EXPIRED
            elif days_since_cal > self.meter_config.calibration_due_days * 0.9:
                cal_status = CalibrationStatus.REQUIRED
            else:
                cal_status = CalibrationStatus.VALID
            next_cal_due = self.meter_config.last_calibration_date + timedelta(
                days=self.meter_config.calibration_due_days
            )
        else:
            cal_status = CalibrationStatus.REQUIRED
            next_cal_due = None

        return MeterStatusReading(
            meter_id=self.meter_config.meter_id,
            status=MeterStatus.OPERATIONAL,
            calibration_status=cal_status,
            last_calibration=self.meter_config.last_calibration_date,
            next_calibration_due=next_cal_due,
            signal_strength_percent=85.0 + random.uniform(-10, 10),
            diagnostics={
                "firmware_version": "2.3.1",
                "hardware_revision": "B",
                "operating_hours": 8760,
                "power_cycles": 42,
            },
            alarms=[],
            warnings=[],
        )

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    async def calibrate(self, calibration_data: CalibrationData) -> CalibrationResult:
        """
        Perform meter calibration.

        Args:
            calibration_data: Calibration input data

        Returns:
            Calibration result

        Raises:
            CalibrationError: If calibration fails
        """
        if self._state != ConnectionState.CONNECTED:
            raise CalibrationError(
                "Cannot calibrate: not connected to meter",
                connector_id=self._config.connector_id,
            )

        self._logger.info(
            f"Starting calibration for meter {self.meter_config.meter_id} "
            f"by operator {calibration_data.operator_id}"
        )

        try:
            # Read current values before calibration
            before_reading = await self.read_quality_parameters()
            before_values = {
                "pressure_bar": before_reading.pressure_bar,
                "temperature_c": before_reading.temperature_c,
                "mass_flow_kg_hr": before_reading.mass_flow_kg_hr,
            }

            # Perform calibration based on type
            calibration_type = "full"

            if calibration_data.zero_point:
                calibration_type = "zero"
                # In production: send zero calibration command to meter
                self._logger.info("Performing zero-point calibration")

            if calibration_data.span_adjustment:
                calibration_type = "span"
                # In production: apply span adjustment
                self._logger.info(
                    f"Applying span adjustment: {calibration_data.span_adjustment}"
                )

            # Simulate calibration delay
            await asyncio.sleep(2.0)

            # Read values after calibration
            after_reading = await self.read_quality_parameters()
            after_values = {
                "pressure_bar": after_reading.pressure_bar,
                "temperature_c": after_reading.temperature_c,
                "mass_flow_kg_hr": after_reading.mass_flow_kg_hr,
            }

            # Calculate deviation
            deviation = None
            if calibration_data.reference_pressure_bar:
                deviation = abs(
                    after_reading.pressure_bar - calibration_data.reference_pressure_bar
                ) / calibration_data.reference_pressure_bar * 100

            # Update calibration date
            self.meter_config.last_calibration_date = datetime.utcnow()
            next_cal_due = datetime.utcnow() + timedelta(
                days=self.meter_config.calibration_due_days
            )

            result = CalibrationResult(
                meter_id=self.meter_config.meter_id,
                success=True,
                calibration_type=calibration_type,
                before_values=before_values,
                after_values=after_values,
                deviation_percent=deviation,
                next_calibration_due=next_cal_due,
                certificate_number=f"CAL-{self.meter_config.meter_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                operator_id=calibration_data.operator_id,
            )

            await self._audit_logger.log_operation(
                operation="calibrate",
                status="success",
                user_id=calibration_data.operator_id,
                request_data=calibration_data.model_dump(),
                response_summary=f"Calibration successful, deviation: {deviation}%",
            )

            self._logger.info(
                f"Calibration completed for meter {self.meter_config.meter_id}"
            )

            return result

        except Exception as e:
            self._logger.error(f"Calibration failed: {e}")

            await self._audit_logger.log_operation(
                operation="calibrate",
                status="failure",
                user_id=calibration_data.operator_id,
                error_message=str(e),
            )

            raise CalibrationError(
                f"Calibration failed: {e}",
                connector_id=self._config.connector_id,
                details={"meter_id": self.meter_config.meter_id},
            )

    # -------------------------------------------------------------------------
    # Data Quality Validation
    # -------------------------------------------------------------------------

    def _validate_reading(self, reading: SteamQualityReading) -> SteamQualityReading:
        """
        Validate reading and update quality flags.

        Args:
            reading: Raw reading to validate

        Returns:
            Validated reading with updated quality flags
        """
        quality_score = 100.0
        quality_flag = QualityFlag.GOOD
        issues = []

        # Range checks
        if reading.pressure_bar < self.meter_config.pressure_min_bar:
            quality_score -= 20
            issues.append(f"Pressure below min: {reading.pressure_bar}")
        elif reading.pressure_bar > self.meter_config.pressure_max_bar:
            quality_score -= 20
            issues.append(f"Pressure above max: {reading.pressure_bar}")

        if reading.temperature_c < self.meter_config.temperature_min_c:
            quality_score -= 20
            issues.append(f"Temperature below min: {reading.temperature_c}")
        elif reading.temperature_c > self.meter_config.temperature_max_c:
            quality_score -= 20
            issues.append(f"Temperature above max: {reading.temperature_c}")

        if reading.mass_flow_kg_hr < self.meter_config.flow_min_kg_hr:
            quality_score -= 20
            issues.append(f"Flow below min: {reading.mass_flow_kg_hr}")
        elif reading.mass_flow_kg_hr > self.meter_config.flow_max_kg_hr:
            quality_score -= 20
            issues.append(f"Flow above max: {reading.mass_flow_kg_hr}")

        # Dryness fraction check
        if reading.dryness_fraction < self.meter_config.dryness_fraction_min:
            quality_score -= 15
            issues.append(f"Dryness below min: {reading.dryness_fraction}")

        # Rate of change check (if we have history)
        if len(self._reading_history) > 0:
            last_reading = self._reading_history[-1]
            time_delta = (reading.timestamp - last_reading.timestamp).total_seconds()

            if time_delta > 0:
                pressure_rate = abs(
                    reading.pressure_bar - last_reading.pressure_bar
                ) / time_delta
                if pressure_rate > 1.0:  # > 1 bar/second is suspicious
                    quality_score -= 10
                    issues.append(f"Rapid pressure change: {pressure_rate:.2f} bar/s")

        # Determine quality flag
        if quality_score >= 80:
            quality_flag = QualityFlag.GOOD
        elif quality_score >= 50:
            quality_flag = QualityFlag.UNCERTAIN
        else:
            quality_flag = QualityFlag.BAD

        # Log issues if any
        if issues:
            self._logger.warning(
                f"Data quality issues for meter {self.meter_config.meter_id}: "
                f"{', '.join(issues)}"
            )

        # Return updated reading (create new since frozen)
        return SteamQualityReading(
            timestamp=reading.timestamp,
            meter_id=reading.meter_id,
            dryness_fraction=reading.dryness_fraction,
            wetness_fraction=reading.wetness_fraction,
            specific_enthalpy_kj_kg=reading.specific_enthalpy_kj_kg,
            specific_entropy_kj_kg_k=reading.specific_entropy_kj_kg_k,
            pressure_bar=reading.pressure_bar,
            temperature_c=reading.temperature_c,
            saturation_temperature_c=reading.saturation_temperature_c,
            superheat_c=reading.superheat_c,
            mass_flow_kg_hr=reading.mass_flow_kg_hr,
            volumetric_flow_m3_hr=reading.volumetric_flow_m3_hr,
            velocity_m_s=reading.velocity_m_s,
            density_kg_m3=reading.density_kg_m3,
            energy_flow_kw=reading.energy_flow_kw,
            quality_flag=quality_flag,
            quality_score=max(0.0, quality_score),
            raw_pressure=reading.raw_pressure,
            raw_temperature=reading.raw_temperature,
            raw_flow=reading.raw_flow,
        )

    # -------------------------------------------------------------------------
    # Statistics and History
    # -------------------------------------------------------------------------

    def get_current_reading(self) -> Optional[SteamQualityReading]:
        """Get the most recent reading."""
        return self._current_reading

    def get_reading_history(
        self,
        minutes: int = 60
    ) -> List[SteamQualityReading]:
        """
        Get reading history for specified time window.

        Args:
            minutes: Number of minutes of history

        Returns:
            List of readings within time window
        """
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [r for r in self._reading_history if r.timestamp > cutoff]

    def get_statistics(self, minutes: int = 60) -> Dict[str, Any]:
        """
        Calculate statistics for recent readings.

        Args:
            minutes: Time window for statistics

        Returns:
            Dictionary of statistics
        """
        readings = self.get_reading_history(minutes)

        if not readings:
            return {"error": "No data available"}

        pressures = [r.pressure_bar for r in readings]
        temperatures = [r.temperature_c for r in readings]
        flows = [r.mass_flow_kg_hr for r in readings]
        dryness = [r.dryness_fraction for r in readings]
        qualities = [r.quality_score for r in readings]

        return {
            "count": len(readings),
            "window_minutes": minutes,
            "pressure": {
                "min": min(pressures),
                "max": max(pressures),
                "avg": sum(pressures) / len(pressures),
                "current": pressures[-1],
            },
            "temperature": {
                "min": min(temperatures),
                "max": max(temperatures),
                "avg": sum(temperatures) / len(temperatures),
                "current": temperatures[-1],
            },
            "flow": {
                "min": min(flows),
                "max": max(flows),
                "avg": sum(flows) / len(flows),
                "current": flows[-1],
            },
            "dryness_fraction": {
                "min": min(dryness),
                "max": max(dryness),
                "avg": sum(dryness) / len(dryness),
                "current": dryness[-1],
            },
            "quality_score": {
                "min": min(qualities),
                "max": max(qualities),
                "avg": sum(qualities) / len(qualities),
            },
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_steam_quality_meter_connector(
    meter_id: str,
    host: str,
    port: int = 502,
    meter_type: MeterType = MeterType.VORTEX_FLOW,
    protocol: ProtocolType = ProtocolType.MODBUS_TCP,
    **kwargs,
) -> SteamQualityMeterConnector:
    """
    Factory function to create a steam quality meter connector.

    Args:
        meter_id: Unique meter identifier
        host: Meter host address
        port: Meter port number
        meter_type: Type of flow meter
        protocol: Communication protocol
        **kwargs: Additional configuration options

    Returns:
        Configured SteamQualityMeterConnector instance
    """
    config = SteamQualityMeterConfig(
        connector_name=f"SteamQualityMeter_{meter_id}",
        meter_id=meter_id,
        host=host,
        port=port,
        meter_type=meter_type,
        protocol=protocol,
        **kwargs,
    )

    return SteamQualityMeterConnector(config)
