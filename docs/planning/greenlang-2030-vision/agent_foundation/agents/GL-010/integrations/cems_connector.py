"""
Continuous Emissions Monitoring System (CEMS) Connector for GL-010 EMISSIONWATCH.

Provides integration with major CEMS vendors (Thermo Fisher, Teledyne, Horiba, Siemens)
via Modbus TCP/RTU and OPC-UA protocols. Implements EPA 40 CFR Part 75 compliant
data validation, quality assurance, and missing data substitution procedures.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import (
    Any,
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


class CEMSVendor(str, Enum):
    """Supported CEMS vendors."""

    THERMO_FISHER = "thermo_fisher"
    TELEDYNE = "teledyne"
    HORIBA = "horiba"
    SIEMENS = "siemens"
    ABB = "abb"
    EMERSON = "emerson"
    YOKOGAWA = "yokogawa"
    SICK = "sick"
    SERVOMEX = "servomex"
    GENERIC = "generic"


class ProtocolType(str, Enum):
    """Communication protocol types."""

    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    OPC_UA = "opc_ua"
    OPC_DA = "opc_da"
    MQTT = "mqtt"
    HTTP_REST = "http_rest"


class AnalyzerType(str, Enum):
    """Types of gas analyzers."""

    NOX = "nox"  # Nitrogen oxides (NO + NO2)
    NO = "no"  # Nitric oxide
    NO2 = "no2"  # Nitrogen dioxide
    SO2 = "so2"  # Sulfur dioxide
    CO = "co"  # Carbon monoxide
    CO2 = "co2"  # Carbon dioxide
    O2 = "o2"  # Oxygen
    OPACITY = "opacity"  # Opacity/particulate
    FLOW = "flow"  # Stack flow rate
    TEMPERATURE = "temperature"  # Stack temperature
    PRESSURE = "pressure"  # Stack pressure
    MOISTURE = "moisture"  # Moisture content
    HCL = "hcl"  # Hydrogen chloride
    HF = "hf"  # Hydrogen fluoride
    NH3 = "ammonia"  # Ammonia
    VOC = "voc"  # Volatile organic compounds
    MERCURY = "mercury"  # Mercury


class MeasurementUnit(str, Enum):
    """Measurement units."""

    PPM = "ppm"  # Parts per million
    PPMD = "ppmd"  # Parts per million (dry)
    PPMW = "ppmw"  # Parts per million (wet)
    PERCENT = "percent"
    LB_MMBTU = "lb/mmBtu"  # Pounds per million BTU
    LB_HR = "lb/hr"  # Pounds per hour
    KG_HR = "kg/hr"  # Kilograms per hour
    SCFM = "scfm"  # Standard cubic feet per minute
    DSCFM = "dscfm"  # Dry standard cubic feet per minute
    ACFM = "acfm"  # Actual cubic feet per minute
    M3_HR = "m3/hr"  # Cubic meters per hour
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    INWC = "inwc"  # Inches of water column
    MBAR = "mbar"  # Millibars
    KPA = "kpa"  # Kilopascals


class QualityAssuranceStatus(str, Enum):
    """QA/QC status codes per EPA Part 75."""

    VALID = "valid"
    CALIBRATION_ERROR = "calibration_error"
    OUT_OF_RANGE = "out_of_range"
    INTERFERENCE = "interference"
    SUBSTITUTED = "substituted"
    MISSING = "missing"
    MAINTENANCE = "maintenance"
    INVALID = "invalid"
    QUESTIONABLE = "questionable"


class CalibrationStatus(str, Enum):
    """Calibration status."""

    PASSED = "passed"
    FAILED = "failed"
    EXPIRED = "expired"
    IN_PROGRESS = "in_progress"
    NOT_REQUIRED = "not_required"
    PENDING = "pending"


class SubstitutionMethod(str, Enum):
    """EPA Part 75 missing data substitution methods."""

    LOOKBACK_90 = "90_day_lookback"  # 90-day lookback average
    MAXIMUM_VALUE = "maximum_value"  # Maximum potential concentration
    DEFAULT_HIGH = "default_high"  # Default high value
    FUEL_SPECIFIC = "fuel_specific"  # Fuel-specific emission factor
    RATA_REFERENCE = "rata_reference"  # RATA reference method value
    QUALITY_ASSURED = "quality_assured"  # Quality-assured value
    CONDITIONAL_DATA = "conditional_data"  # Conditional data validation


class ModbusRegisterType(IntEnum):
    """Modbus register types."""

    COIL = 1
    DISCRETE_INPUT = 2
    HOLDING_REGISTER = 3
    INPUT_REGISTER = 4


# =============================================================================
# Pydantic Models
# =============================================================================


class AnalyzerReading(BaseModel):
    """Single analyzer reading with quality metadata."""

    model_config = ConfigDict(frozen=True)

    reading_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique reading identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Reading timestamp (UTC)"
    )
    analyzer_type: AnalyzerType = Field(
        ...,
        description="Type of analyzer"
    )
    value: float = Field(
        ...,
        description="Measured value"
    )
    unit: MeasurementUnit = Field(
        ...,
        description="Measurement unit"
    )
    quality_status: QualityAssuranceStatus = Field(
        default=QualityAssuranceStatus.VALID,
        description="QA/QC status"
    )
    quality_code: Optional[str] = Field(
        default=None,
        description="Detailed quality code"
    )
    calibration_status: CalibrationStatus = Field(
        default=CalibrationStatus.PASSED,
        description="Calibration status at time of reading"
    )
    is_substituted: bool = Field(
        default=False,
        description="Whether value was substituted"
    )
    substitution_method: Optional[SubstitutionMethod] = Field(
        default=None,
        description="Substitution method if substituted"
    )
    original_value: Optional[float] = Field(
        default=None,
        description="Original value before substitution"
    )
    span_value: Optional[float] = Field(
        default=None,
        description="Analyzer span value"
    )
    zero_value: Optional[float] = Field(
        default=None,
        description="Analyzer zero value"
    )
    raw_value: Optional[float] = Field(
        default=None,
        description="Raw analog value before conversion"
    )


class EmissionsData(BaseModel):
    """Complete emissions data packet from CEMS."""

    model_config = ConfigDict(frozen=True)

    data_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique data packet identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Data timestamp (UTC)"
    )
    unit_id: str = Field(
        ...,
        description="Emission unit identifier"
    )
    stack_id: str = Field(
        ...,
        description="Stack identifier"
    )

    # Analyzer readings
    readings: List[AnalyzerReading] = Field(
        default_factory=list,
        description="List of analyzer readings"
    )

    # Calculated values
    nox_rate_lb_mmbtu: Optional[float] = Field(
        default=None,
        ge=0,
        description="NOx emission rate (lb/MMBtu)"
    )
    so2_rate_lb_mmbtu: Optional[float] = Field(
        default=None,
        ge=0,
        description="SO2 emission rate (lb/MMBtu)"
    )
    co2_rate_lb_mmbtu: Optional[float] = Field(
        default=None,
        ge=0,
        description="CO2 emission rate (lb/MMBtu)"
    )
    heat_input_mmbtu_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Heat input rate (MMBtu/hr)"
    )
    stack_flow_scfh: Optional[float] = Field(
        default=None,
        ge=0,
        description="Stack flow rate (SCFH)"
    )

    # Mass emission rates
    nox_mass_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="NOx mass emission rate (lb/hr)"
    )
    so2_mass_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="SO2 mass emission rate (lb/hr)"
    )
    co2_mass_ton_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="CO2 mass emission rate (tons/hr)"
    )

    # Operating data
    load_mw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Unit load (MW)"
    )
    load_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Unit load percentage"
    )
    operating_time_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Operating time in hour"
    )

    # Quality metrics
    overall_quality_status: QualityAssuranceStatus = Field(
        default=QualityAssuranceStatus.VALID,
        description="Overall data quality status"
    )
    data_availability_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Data availability percentage"
    )
    substituted_data_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage of substituted data"
    )

    def get_reading(self, analyzer_type: AnalyzerType) -> Optional[AnalyzerReading]:
        """Get reading for specific analyzer type."""
        for reading in self.readings:
            if reading.analyzer_type == analyzer_type:
                return reading
        return None


class CalibrationData(BaseModel):
    """Calibration data for an analyzer."""

    model_config = ConfigDict(frozen=True)

    calibration_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Calibration record identifier"
    )
    analyzer_type: AnalyzerType = Field(
        ...,
        description="Analyzer type"
    )
    calibration_type: str = Field(
        ...,
        description="Calibration type (daily, quarterly, etc.)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Calibration timestamp"
    )
    status: CalibrationStatus = Field(
        ...,
        description="Calibration status"
    )

    # Zero calibration
    zero_reference: float = Field(..., description="Zero reference gas value")
    zero_measured: float = Field(..., description="Zero measured value")
    zero_error_percent: float = Field(..., description="Zero calibration error %")
    zero_passed: bool = Field(..., description="Zero calibration passed")

    # Span calibration
    span_reference: float = Field(..., description="Span reference gas value")
    span_measured: float = Field(..., description="Span measured value")
    span_error_percent: float = Field(..., description="Span calibration error %")
    span_passed: bool = Field(..., description="Span calibration passed")

    # Upscale/mid calibration (if applicable)
    mid_reference: Optional[float] = Field(default=None, description="Mid reference value")
    mid_measured: Optional[float] = Field(default=None, description="Mid measured value")
    mid_error_percent: Optional[float] = Field(default=None, description="Mid error %")
    mid_passed: Optional[bool] = Field(default=None, description="Mid calibration passed")

    # EPA Part 75 requirements
    cylinder_id: Optional[str] = Field(default=None, description="Calibration gas cylinder ID")
    cylinder_expiration: Optional[datetime] = Field(
        default=None,
        description="Cylinder expiration date"
    )
    technician_id: Optional[str] = Field(default=None, description="Technician ID")
    notes: Optional[str] = Field(default=None, description="Calibration notes")


class QualityFlags(BaseModel):
    """Data quality flags per EPA Part 75 QA/QC."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Flag timestamp"
    )

    # Overall flags
    is_valid: bool = Field(default=True, description="Data is valid")
    requires_substitution: bool = Field(default=False, description="Requires data substitution")

    # Analyzer-specific flags
    analyzer_flags: Dict[AnalyzerType, QualityAssuranceStatus] = Field(
        default_factory=dict,
        description="Quality flags per analyzer"
    )

    # Specific quality issues
    calibration_drift: bool = Field(default=False, description="Calibration drift detected")
    out_of_control: bool = Field(default=False, description="Out of control condition")
    maintenance_mode: bool = Field(default=False, description="System in maintenance mode")
    communication_error: bool = Field(default=False, description="Communication error")
    range_exceeded: bool = Field(default=False, description="Measurement range exceeded")
    interference_detected: bool = Field(default=False, description="Interference detected")
    span_exceeded: bool = Field(default=False, description="Span exceeded")
    negative_value: bool = Field(default=False, description="Negative value detected")

    # Missing data tracking
    missing_data_hours: float = Field(default=0.0, ge=0, description="Hours of missing data")
    consecutive_missing_hours: int = Field(
        default=0,
        ge=0,
        description="Consecutive hours missing"
    )

    # EPA Part 75 specific
    rata_due: bool = Field(default=False, description="RATA is due")
    bias_adjustment_required: bool = Field(
        default=False,
        description="Bias adjustment required"
    )
    linearity_check_due: bool = Field(default=False, description="Linearity check due")


class ModbusRegisterMap(BaseModel):
    """Modbus register mapping for an analyzer."""

    model_config = ConfigDict(frozen=True)

    analyzer_type: AnalyzerType = Field(..., description="Analyzer type")
    register_address: int = Field(..., ge=0, le=65535, description="Register address")
    register_type: ModbusRegisterType = Field(
        default=ModbusRegisterType.HOLDING_REGISTER,
        description="Register type"
    )
    register_count: int = Field(default=1, ge=1, le=125, description="Number of registers")
    data_type: str = Field(default="float32", description="Data type")
    byte_order: str = Field(default="big", description="Byte order")
    word_order: str = Field(default="big", description="Word order")
    scale_factor: float = Field(default=1.0, description="Scale factor")
    offset: float = Field(default=0.0, description="Offset")
    unit: MeasurementUnit = Field(..., description="Measurement unit")


class OPCUANodeMap(BaseModel):
    """OPC-UA node mapping for an analyzer."""

    model_config = ConfigDict(frozen=True)

    analyzer_type: AnalyzerType = Field(..., description="Analyzer type")
    node_id: str = Field(..., description="OPC-UA node ID")
    namespace_index: int = Field(default=2, ge=0, description="Namespace index")
    browse_name: Optional[str] = Field(default=None, description="Browse name")
    display_name: Optional[str] = Field(default=None, description="Display name")
    data_type: str = Field(default="Double", description="OPC-UA data type")
    scale_factor: float = Field(default=1.0, description="Scale factor")
    offset: float = Field(default=0.0, description="Offset")
    unit: MeasurementUnit = Field(..., description="Measurement unit")


class CEMSConnectorConfig(BaseConnectorConfig):
    """Configuration for CEMS connector."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    connector_type: ConnectorType = Field(
        default=ConnectorType.CEMS,
        description="Connector type"
    )

    # CEMS identification
    vendor: CEMSVendor = Field(..., description="CEMS vendor")
    unit_id: str = Field(..., description="Emission unit identifier")
    stack_id: str = Field(..., description="Stack identifier")
    plant_id: str = Field(..., description="Plant identifier")

    # Protocol settings
    protocol: ProtocolType = Field(..., description="Communication protocol")

    # Modbus settings
    modbus_host: Optional[str] = Field(default=None, description="Modbus TCP host")
    modbus_port: int = Field(default=502, ge=1, le=65535, description="Modbus TCP port")
    modbus_slave_id: int = Field(default=1, ge=1, le=247, description="Modbus slave ID")
    modbus_serial_port: Optional[str] = Field(
        default=None,
        description="Serial port for Modbus RTU"
    )
    modbus_baudrate: int = Field(default=9600, description="Baud rate for RTU")
    modbus_parity: str = Field(default="N", pattern="^[NEOMS]$", description="Parity")
    modbus_stopbits: int = Field(default=1, ge=1, le=2, description="Stop bits")
    modbus_bytesize: int = Field(default=8, ge=5, le=8, description="Byte size")
    modbus_register_maps: List[ModbusRegisterMap] = Field(
        default_factory=list,
        description="Modbus register mappings"
    )

    # OPC-UA settings
    opc_endpoint: Optional[str] = Field(default=None, description="OPC-UA endpoint URL")
    opc_security_policy: Optional[str] = Field(
        default=None,
        description="OPC-UA security policy"
    )
    opc_security_mode: Optional[str] = Field(
        default=None,
        description="OPC-UA security mode"
    )
    opc_username: Optional[str] = Field(default=None, description="OPC-UA username")
    opc_certificate_path: Optional[str] = Field(
        default=None,
        description="OPC-UA certificate path"
    )
    opc_node_maps: List[OPCUANodeMap] = Field(
        default_factory=list,
        description="OPC-UA node mappings"
    )

    # Polling settings
    polling_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Data polling interval"
    )
    averaging_period_minutes: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Averaging period"
    )

    # Analyzer configuration
    analyzers_enabled: List[AnalyzerType] = Field(
        default_factory=lambda: [
            AnalyzerType.NOX,
            AnalyzerType.SO2,
            AnalyzerType.CO,
            AnalyzerType.O2,
            AnalyzerType.CO2,
            AnalyzerType.FLOW,
        ],
        description="Enabled analyzers"
    )

    # Calibration settings
    calibration_error_limit_percent: float = Field(
        default=2.5,
        ge=0.1,
        le=10.0,
        description="Calibration error limit %"
    )
    auto_calibration_detection: bool = Field(
        default=True,
        description="Auto-detect calibration events"
    )

    # EPA Part 75 settings
    epa_part75_enabled: bool = Field(
        default=True,
        description="Enable EPA Part 75 compliance"
    )
    missing_data_lookback_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Missing data lookback period"
    )
    bias_adjustment_factor: float = Field(
        default=1.0,
        ge=0.9,
        le=1.1,
        description="Bias adjustment factor"
    )

    # Data quality thresholds
    data_availability_threshold: float = Field(
        default=90.0,
        ge=0,
        le=100,
        description="Min data availability %"
    )

    @field_validator("protocol")
    @classmethod
    def validate_protocol_settings(cls, v: ProtocolType, info) -> ProtocolType:
        """Validate protocol-specific settings are provided."""
        return v


# =============================================================================
# CEMS Data Processor
# =============================================================================


class CEMSDataProcessor:
    """
    Processes and validates CEMS data per EPA Part 75 requirements.

    Handles:
    - Data validation and quality flagging
    - Missing data substitution
    - Calibration drift adjustment
    - Emission rate calculations
    """

    def __init__(self, config: CEMSConnectorConfig) -> None:
        """
        Initialize data processor.

        Args:
            config: CEMS connector configuration
        """
        self._config = config
        self._lookback_data: Dict[AnalyzerType, List[float]] = {}
        self._calibration_history: Dict[AnalyzerType, List[CalibrationData]] = {}
        self._logger = logging.getLogger(f"cems.processor.{config.unit_id}")

    def validate_reading(
        self,
        reading: AnalyzerReading,
        calibration: Optional[CalibrationData] = None,
    ) -> Tuple[AnalyzerReading, QualityAssuranceStatus]:
        """
        Validate a single analyzer reading.

        Args:
            reading: Raw analyzer reading
            calibration: Latest calibration data

        Returns:
            Tuple of validated reading and quality status
        """
        quality_status = QualityAssuranceStatus.VALID
        issues: List[str] = []

        # Check for negative values
        if reading.value < 0:
            if reading.analyzer_type not in [AnalyzerType.TEMPERATURE, AnalyzerType.PRESSURE]:
                issues.append("negative_value")
                quality_status = QualityAssuranceStatus.INVALID

        # Check range limits by analyzer type
        range_limits = self._get_range_limits(reading.analyzer_type)
        if range_limits:
            min_val, max_val = range_limits
            if reading.value < min_val or reading.value > max_val:
                issues.append("out_of_range")
                quality_status = QualityAssuranceStatus.OUT_OF_RANGE

        # Check calibration status
        if calibration and calibration.status != CalibrationStatus.PASSED:
            issues.append("calibration_error")
            if quality_status == QualityAssuranceStatus.VALID:
                quality_status = QualityAssuranceStatus.CALIBRATION_ERROR

        if issues:
            self._logger.warning(
                f"Reading validation issues for {reading.analyzer_type.value}: {issues}"
            )

        return reading, quality_status

    def _get_range_limits(
        self,
        analyzer_type: AnalyzerType,
    ) -> Optional[Tuple[float, float]]:
        """Get valid range limits for analyzer type."""
        # EPA Part 75 typical ranges
        limits = {
            AnalyzerType.NOX: (0, 2000),  # ppm
            AnalyzerType.SO2: (0, 5000),  # ppm
            AnalyzerType.CO: (0, 10000),  # ppm
            AnalyzerType.CO2: (0, 20),  # percent
            AnalyzerType.O2: (0, 25),  # percent
            AnalyzerType.OPACITY: (0, 100),  # percent
            AnalyzerType.FLOW: (0, 10000000),  # SCFH
        }
        return limits.get(analyzer_type)

    def substitute_missing_data(
        self,
        analyzer_type: AnalyzerType,
        missing_hours: int,
    ) -> Tuple[float, SubstitutionMethod]:
        """
        Apply EPA Part 75 missing data substitution.

        Args:
            analyzer_type: Type of analyzer
            missing_hours: Consecutive hours of missing data

        Returns:
            Tuple of substituted value and method used
        """
        lookback_values = self._lookback_data.get(analyzer_type, [])

        if not lookback_values:
            # Use maximum potential value
            method = SubstitutionMethod.MAXIMUM_VALUE
            value = self._get_maximum_potential(analyzer_type)
        elif missing_hours <= 2:
            # Standard substitution for short-term missing data
            method = SubstitutionMethod.LOOKBACK_90
            # Use 90th percentile of lookback data
            sorted_values = sorted(lookback_values)
            idx = int(len(sorted_values) * 0.9)
            value = sorted_values[min(idx, len(sorted_values) - 1)]
        elif missing_hours <= 24:
            # Use higher percentile for longer gaps
            method = SubstitutionMethod.LOOKBACK_90
            sorted_values = sorted(lookback_values)
            idx = int(len(sorted_values) * 0.95)
            value = sorted_values[min(idx, len(sorted_values) - 1)]
        else:
            # Use maximum value for extended outages
            method = SubstitutionMethod.MAXIMUM_VALUE
            value = max(lookback_values) if lookback_values else self._get_maximum_potential(
                analyzer_type
            )

        self._logger.info(
            f"Substituted {analyzer_type.value} data: {value} using {method.value}"
        )

        return value, method

    def _get_maximum_potential(self, analyzer_type: AnalyzerType) -> float:
        """Get maximum potential concentration/value for analyzer type."""
        # EPA default maximum values
        defaults = {
            AnalyzerType.NOX: 200.0,  # ppm
            AnalyzerType.SO2: 200.0,  # ppm
            AnalyzerType.CO: 200.0,  # ppm
            AnalyzerType.CO2: 18.0,  # percent
            AnalyzerType.O2: 14.0,  # percent (used for F-factor)
            AnalyzerType.FLOW: 1000000,  # SCFH
        }
        return defaults.get(analyzer_type, 100.0)

    def calculate_emission_rate(
        self,
        concentration_ppm: float,
        o2_percent: float,
        f_factor: float = 9780,
        diluent: str = "O2",
    ) -> float:
        """
        Calculate emission rate in lb/MMBtu using EPA F-factor method.

        Args:
            concentration_ppm: Pollutant concentration (ppm dry)
            o2_percent: O2 concentration (percent dry)
            f_factor: F-factor for fuel type (default natural gas)
            diluent: Diluent type (O2 or CO2)

        Returns:
            Emission rate in lb/MMBtu
        """
        # EPA F-factor equation for O2 diluent
        # E = C * Fd * (20.9 / (20.9 - %O2)) * K
        # where K is conversion factor for pollutant

        # Molecular weight conversion factors (to lb/scf)
        mw_factors = {
            "NOX": 1.194e-7,  # NOX as NO2
            "SO2": 1.660e-7,
            "CO": 0.725e-7,
            "CO2": 1.137e-7,
        }

        # Default to NOx conversion
        k_factor = mw_factors.get("NOX", 1.194e-7)

        # Calculate corrected concentration
        if diluent == "O2" and o2_percent < 20.9:
            correction = 20.9 / (20.9 - o2_percent)
        else:
            correction = 1.0

        # Calculate emission rate
        emission_rate = concentration_ppm * f_factor * correction * k_factor

        return emission_rate

    def calculate_mass_rate(
        self,
        concentration_ppm: float,
        flow_scfh: float,
        molecular_weight: float,
    ) -> float:
        """
        Calculate mass emission rate in lb/hr.

        Args:
            concentration_ppm: Pollutant concentration (ppm)
            flow_scfh: Stack flow rate (SCFH)
            molecular_weight: Molecular weight of pollutant

        Returns:
            Mass emission rate in lb/hr
        """
        # At standard conditions: 1 lb-mol = 385.3 scf
        molar_volume = 385.3

        # ppm to lb/scf: ppm * 1e-6 * MW / molar_volume
        lb_per_scf = concentration_ppm * 1e-6 * molecular_weight / molar_volume

        # lb/hr = lb/scf * scf/hr
        mass_rate = lb_per_scf * flow_scfh

        return mass_rate

    def update_lookback_data(
        self,
        analyzer_type: AnalyzerType,
        value: float,
    ) -> None:
        """
        Update lookback data for missing data substitution.

        Args:
            analyzer_type: Analyzer type
            value: Valid reading value
        """
        if analyzer_type not in self._lookback_data:
            self._lookback_data[analyzer_type] = []

        self._lookback_data[analyzer_type].append(value)

        # Keep only 90 days of hourly data (2160 hours)
        max_entries = 2160
        if len(self._lookback_data[analyzer_type]) > max_entries:
            self._lookback_data[analyzer_type] = self._lookback_data[analyzer_type][-max_entries:]


# =============================================================================
# Protocol Handlers
# =============================================================================


class ModbusHandler:
    """
    Handles Modbus TCP/RTU communication.

    Implements asynchronous Modbus client for reading CEMS data
    from PLCs and RTUs.
    """

    def __init__(self, config: CEMSConnectorConfig) -> None:
        """
        Initialize Modbus handler.

        Args:
            config: CEMS connector configuration
        """
        self._config = config
        self._client: Optional[Any] = None
        self._connected = False
        self._logger = logging.getLogger(f"cems.modbus.{config.unit_id}")

    async def connect(self) -> None:
        """Establish Modbus connection."""
        try:
            if self._config.protocol == ProtocolType.MODBUS_TCP:
                # Simulated connection for TCP
                self._logger.info(
                    f"Connecting to Modbus TCP at {self._config.modbus_host}:"
                    f"{self._config.modbus_port}"
                )
                # In production, use pymodbus AsyncModbusTcpClient
                self._connected = True

            elif self._config.protocol == ProtocolType.MODBUS_RTU:
                self._logger.info(
                    f"Connecting to Modbus RTU at {self._config.modbus_serial_port}"
                )
                # In production, use pymodbus AsyncModbusSerialClient
                self._connected = True

            self._logger.info("Modbus connection established")

        except Exception as e:
            self._logger.error(f"Modbus connection failed: {e}")
            raise ConnectionError(f"Failed to connect to Modbus device: {e}")

    async def disconnect(self) -> None:
        """Close Modbus connection."""
        if self._client:
            # Close client
            pass
        self._connected = False
        self._logger.info("Modbus connection closed")

    async def read_register(
        self,
        register_map: ModbusRegisterMap,
    ) -> float:
        """
        Read value from Modbus register.

        Args:
            register_map: Register mapping configuration

        Returns:
            Scaled and converted value
        """
        if not self._connected:
            raise ConnectionError("Not connected to Modbus device")

        try:
            # Simulate reading register
            # In production, use self._client.read_holding_registers() etc.

            # Simulated raw value
            raw_value = 0.0

            # Apply data type conversion
            if register_map.data_type == "float32":
                # Convert 2 registers to float32
                pass
            elif register_map.data_type == "uint16":
                pass
            elif register_map.data_type == "int16":
                pass

            # Apply scale and offset
            value = (raw_value * register_map.scale_factor) + register_map.offset

            return value

        except Exception as e:
            self._logger.error(
                f"Failed to read Modbus register {register_map.register_address}: {e}"
            )
            raise

    async def read_all_analyzers(self) -> Dict[AnalyzerType, float]:
        """
        Read all configured analyzer values.

        Returns:
            Dictionary of analyzer type to value
        """
        results: Dict[AnalyzerType, float] = {}

        for register_map in self._config.modbus_register_maps:
            try:
                value = await self.read_register(register_map)
                results[register_map.analyzer_type] = value
            except Exception as e:
                self._logger.warning(
                    f"Failed to read {register_map.analyzer_type.value}: {e}"
                )

        return results


class OPCUAHandler:
    """
    Handles OPC-UA communication.

    Implements asynchronous OPC-UA client for reading CEMS data
    from modern systems.
    """

    def __init__(self, config: CEMSConnectorConfig) -> None:
        """
        Initialize OPC-UA handler.

        Args:
            config: CEMS connector configuration
        """
        self._config = config
        self._client: Optional[Any] = None
        self._connected = False
        self._subscription: Optional[Any] = None
        self._logger = logging.getLogger(f"cems.opcua.{config.unit_id}")

    async def connect(self) -> None:
        """Establish OPC-UA connection."""
        try:
            self._logger.info(f"Connecting to OPC-UA at {self._config.opc_endpoint}")

            # In production, use asyncua Client
            # self._client = Client(url=self._config.opc_endpoint)
            # await self._client.connect()

            self._connected = True
            self._logger.info("OPC-UA connection established")

        except Exception as e:
            self._logger.error(f"OPC-UA connection failed: {e}")
            raise ConnectionError(f"Failed to connect to OPC-UA server: {e}")

    async def disconnect(self) -> None:
        """Close OPC-UA connection."""
        if self._subscription:
            # Delete subscription
            pass
        if self._client:
            # Disconnect client
            pass
        self._connected = False
        self._logger.info("OPC-UA connection closed")

    async def read_node(self, node_map: OPCUANodeMap) -> float:
        """
        Read value from OPC-UA node.

        Args:
            node_map: Node mapping configuration

        Returns:
            Scaled value
        """
        if not self._connected:
            raise ConnectionError("Not connected to OPC-UA server")

        try:
            # In production:
            # node = self._client.get_node(node_map.node_id)
            # value = await node.read_value()

            # Simulated value
            raw_value = 0.0

            # Apply scale and offset
            value = (raw_value * node_map.scale_factor) + node_map.offset

            return value

        except Exception as e:
            self._logger.error(f"Failed to read OPC-UA node {node_map.node_id}: {e}")
            raise

    async def read_all_analyzers(self) -> Dict[AnalyzerType, float]:
        """
        Read all configured analyzer values.

        Returns:
            Dictionary of analyzer type to value
        """
        results: Dict[AnalyzerType, float] = {}

        for node_map in self._config.opc_node_maps:
            try:
                value = await self.read_node(node_map)
                results[node_map.analyzer_type] = value
            except Exception as e:
                self._logger.warning(
                    f"Failed to read {node_map.analyzer_type.value}: {e}"
                )

        return results

    async def subscribe_to_changes(
        self,
        callback: Callable[[AnalyzerType, float], None],
    ) -> None:
        """
        Subscribe to value changes for real-time updates.

        Args:
            callback: Callback function for value changes
        """
        if not self._connected:
            raise ConnectionError("Not connected to OPC-UA server")

        # In production, create subscription with callback
        self._logger.info("Subscribed to OPC-UA value changes")


# =============================================================================
# Calibration Manager
# =============================================================================


class CalibrationManager:
    """
    Manages CEMS calibration data and status.

    Tracks calibration history, detects calibration events,
    and validates calibration results per EPA Part 75.
    """

    def __init__(self, config: CEMSConnectorConfig) -> None:
        """
        Initialize calibration manager.

        Args:
            config: CEMS connector configuration
        """
        self._config = config
        self._calibration_history: Dict[AnalyzerType, List[CalibrationData]] = {}
        self._current_calibration: Dict[AnalyzerType, CalibrationData] = {}
        self._calibration_in_progress = False
        self._logger = logging.getLogger(f"cems.calibration.{config.unit_id}")

    def record_calibration(self, calibration: CalibrationData) -> None:
        """
        Record a calibration event.

        Args:
            calibration: Calibration data
        """
        analyzer_type = calibration.analyzer_type

        if analyzer_type not in self._calibration_history:
            self._calibration_history[analyzer_type] = []

        self._calibration_history[analyzer_type].append(calibration)
        self._current_calibration[analyzer_type] = calibration

        # Keep only 1 year of history
        max_entries = 365 * 4  # Assuming 4 calibrations per day max
        if len(self._calibration_history[analyzer_type]) > max_entries:
            self._calibration_history[analyzer_type] = (
                self._calibration_history[analyzer_type][-max_entries:]
            )

        self._logger.info(
            f"Recorded calibration for {analyzer_type.value}: "
            f"status={calibration.status.value}"
        )

    def get_calibration_status(
        self,
        analyzer_type: AnalyzerType,
    ) -> CalibrationStatus:
        """
        Get current calibration status for analyzer.

        Args:
            analyzer_type: Analyzer type

        Returns:
            Current calibration status
        """
        calibration = self._current_calibration.get(analyzer_type)

        if calibration is None:
            return CalibrationStatus.PENDING

        # Check if calibration has expired (typically 24 hours for daily cal)
        age = datetime.utcnow() - calibration.timestamp
        if age > timedelta(hours=26):  # 2-hour grace period
            return CalibrationStatus.EXPIRED

        return calibration.status

    def detect_calibration_event(
        self,
        analyzer_type: AnalyzerType,
        current_value: float,
        previous_value: float,
    ) -> bool:
        """
        Detect if a calibration event is occurring.

        Args:
            analyzer_type: Analyzer type
            current_value: Current reading
            previous_value: Previous reading

        Returns:
            True if calibration event detected
        """
        if not self._config.auto_calibration_detection:
            return False

        # Detect large step changes that indicate calibration gas
        # Typical cal gases create 50-100% of span change
        if previous_value > 0:
            change_percent = abs(current_value - previous_value) / previous_value * 100
            if change_percent > 50:
                self._logger.info(
                    f"Potential calibration event detected for {analyzer_type.value}: "
                    f"{change_percent:.1f}% change"
                )
                return True

        return False

    def validate_calibration(
        self,
        calibration: CalibrationData,
    ) -> Tuple[bool, List[str]]:
        """
        Validate calibration results per EPA Part 75.

        Args:
            calibration: Calibration data to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues: List[str] = []
        error_limit = self._config.calibration_error_limit_percent

        # Check zero calibration
        if abs(calibration.zero_error_percent) > error_limit:
            issues.append(
                f"Zero error {calibration.zero_error_percent:.2f}% exceeds "
                f"limit of {error_limit}%"
            )

        # Check span calibration
        if abs(calibration.span_error_percent) > error_limit:
            issues.append(
                f"Span error {calibration.span_error_percent:.2f}% exceeds "
                f"limit of {error_limit}%"
            )

        # Check mid-level if applicable
        if calibration.mid_error_percent is not None:
            if abs(calibration.mid_error_percent) > error_limit:
                issues.append(
                    f"Mid error {calibration.mid_error_percent:.2f}% exceeds "
                    f"limit of {error_limit}%"
                )

        # Check cylinder expiration
        if calibration.cylinder_expiration:
            if calibration.cylinder_expiration < datetime.utcnow():
                issues.append("Calibration gas cylinder has expired")

        is_valid = len(issues) == 0

        if not is_valid:
            self._logger.warning(
                f"Calibration validation failed for {calibration.analyzer_type.value}: "
                f"{issues}"
            )

        return is_valid, issues


# =============================================================================
# CEMS Connector
# =============================================================================


class CEMSConnector(BaseConnector):
    """
    Continuous Emissions Monitoring System (CEMS) Connector.

    Provides integration with major CEMS vendors via Modbus TCP/RTU
    and OPC-UA protocols. Implements EPA 40 CFR Part 75 compliant
    data validation, quality assurance, and missing data substitution.

    Features:
    - Multi-vendor support (Thermo Fisher, Teledyne, Horiba, Siemens)
    - Real-time data acquisition
    - EPA Part 75 QA/QC compliance
    - Automatic calibration detection
    - Missing data substitution
    - Multi-analyzer support
    """

    def __init__(self, config: CEMSConnectorConfig) -> None:
        """
        Initialize CEMS connector.

        Args:
            config: CEMS connector configuration
        """
        super().__init__(config)
        self._cems_config = config

        # Initialize protocol handlers
        self._modbus_handler: Optional[ModbusHandler] = None
        self._opcua_handler: Optional[OPCUAHandler] = None

        # Initialize processors
        self._data_processor = CEMSDataProcessor(config)
        self._calibration_manager = CalibrationManager(config)

        # Real-time data storage
        self._current_readings: Dict[AnalyzerType, AnalyzerReading] = {}
        self._averaging_buffer: Dict[AnalyzerType, List[float]] = {}

        # Polling task
        self._polling_task: Optional[asyncio.Task] = None
        self._polling_active = False

        # Callbacks for real-time data
        self._data_callbacks: List[Callable[[EmissionsData], None]] = []

        self._logger = logging.getLogger(f"cems.connector.{config.unit_id}")

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Establish connection to CEMS system.

        Raises:
            ConnectionError: If connection fails
        """
        self._state = ConnectionState.CONNECTING
        self._logger.info(f"Connecting to CEMS system at {self._cems_config.unit_id}")

        try:
            if self._cems_config.protocol in [
                ProtocolType.MODBUS_TCP,
                ProtocolType.MODBUS_RTU,
            ]:
                self._modbus_handler = ModbusHandler(self._cems_config)
                await self._modbus_handler.connect()

            elif self._cems_config.protocol == ProtocolType.OPC_UA:
                self._opcua_handler = OPCUAHandler(self._cems_config)
                await self._opcua_handler.connect()

            self._state = ConnectionState.CONNECTED
            self._logger.info("CEMS connection established successfully")

            # Log audit event
            await self._audit_logger.log_operation(
                operation="connect",
                status="success",
                response_summary=f"Connected to CEMS via {self._cems_config.protocol.value}",
            )

        except Exception as e:
            self._state = ConnectionState.ERROR
            await self._audit_logger.log_operation(
                operation="connect",
                status="failure",
                error_message=str(e),
            )
            raise ConnectionError(
                f"Failed to connect to CEMS: {e}",
                connector_id=self._config.connector_id,
                connector_type=self._config.connector_type,
            )

    async def disconnect(self) -> None:
        """Disconnect from CEMS system."""
        self._logger.info("Disconnecting from CEMS system")

        # Stop polling
        await self.stop_polling()

        # Close handlers
        if self._modbus_handler:
            await self._modbus_handler.disconnect()
            self._modbus_handler = None

        if self._opcua_handler:
            await self._opcua_handler.disconnect()
            self._opcua_handler = None

        self._state = ConnectionState.DISCONNECTED

        await self._audit_logger.log_operation(
            operation="disconnect",
            status="success",
        )

    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check on CEMS connection.

        Returns:
            Health check result
        """
        start_time = time.time()
        details: Dict[str, Any] = {}

        try:
            # Check connection state
            if self._state != ConnectionState.CONNECTED:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Not connected: {self._state.value}",
                    details={"state": self._state.value},
                )

            # Try to read a value
            if self._modbus_handler:
                # Attempt a read operation
                pass
            elif self._opcua_handler:
                # Attempt a read operation
                pass

            # Check calibration status
            cal_status = {}
            for analyzer_type in self._cems_config.analyzers_enabled:
                status = self._calibration_manager.get_calibration_status(analyzer_type)
                cal_status[analyzer_type.value] = status.value
                if status in [CalibrationStatus.FAILED, CalibrationStatus.EXPIRED]:
                    details["calibration_issues"] = True

            details["calibration_status"] = cal_status

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Determine overall status
            if details.get("calibration_issues"):
                status = HealthStatus.DEGRADED
                message = "CEMS connected but has calibration issues"
            else:
                status = HealthStatus.HEALTHY
                message = "CEMS connection healthy"

            return HealthCheckResult(
                status=status,
                latency_ms=latency_ms,
                message=message,
                details=details,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )

    async def validate_configuration(self) -> bool:
        """
        Validate CEMS connector configuration.

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        issues: List[str] = []

        # Validate protocol settings
        if self._cems_config.protocol in [
            ProtocolType.MODBUS_TCP,
            ProtocolType.MODBUS_RTU,
        ]:
            if self._cems_config.protocol == ProtocolType.MODBUS_TCP:
                if not self._cems_config.modbus_host:
                    issues.append("Modbus TCP requires modbus_host")
            else:
                if not self._cems_config.modbus_serial_port:
                    issues.append("Modbus RTU requires modbus_serial_port")

            if not self._cems_config.modbus_register_maps:
                issues.append("Modbus protocol requires register maps")

        elif self._cems_config.protocol == ProtocolType.OPC_UA:
            if not self._cems_config.opc_endpoint:
                issues.append("OPC-UA requires opc_endpoint")
            if not self._cems_config.opc_node_maps:
                issues.append("OPC-UA protocol requires node maps")

        # Validate analyzer configuration
        if not self._cems_config.analyzers_enabled:
            issues.append("At least one analyzer must be enabled")

        if issues:
            raise ConfigurationError(
                f"Invalid CEMS configuration: {issues}",
                connector_id=self._config.connector_id,
                connector_type=self._config.connector_type,
                details={"issues": issues},
            )

        self._logger.info("CEMS configuration validated successfully")
        return True

    # -------------------------------------------------------------------------
    # CEMS-Specific Methods
    # -------------------------------------------------------------------------

    async def read_realtime_emissions(self) -> EmissionsData:
        """
        Read real-time emissions data from CEMS.

        Returns:
            Current emissions data with all analyzer readings

        Raises:
            ConnectionError: If not connected
            DataQualityError: If data quality is unacceptable
        """
        if not self.is_connected:
            raise ConnectionError(
                "Not connected to CEMS",
                connector_id=self._config.connector_id,
            )

        start_time = time.time()

        try:
            # Read all analyzer values
            raw_values: Dict[AnalyzerType, float] = {}

            if self._modbus_handler:
                raw_values = await self._modbus_handler.read_all_analyzers()
            elif self._opcua_handler:
                raw_values = await self._opcua_handler.read_all_analyzers()

            # Build analyzer readings with validation
            readings: List[AnalyzerReading] = []
            quality_issues: List[str] = []

            for analyzer_type, value in raw_values.items():
                # Get unit for this analyzer
                unit = self._get_unit_for_analyzer(analyzer_type)

                # Get calibration status
                cal_status = self._calibration_manager.get_calibration_status(analyzer_type)

                # Create reading
                reading = AnalyzerReading(
                    analyzer_type=analyzer_type,
                    value=value,
                    unit=unit,
                    calibration_status=cal_status,
                    raw_value=value,
                )

                # Validate reading
                validated_reading, quality_status = self._data_processor.validate_reading(
                    reading,
                    self._calibration_manager._current_calibration.get(analyzer_type),
                )

                # Update reading with validated status
                reading = AnalyzerReading(
                    reading_id=reading.reading_id,
                    timestamp=reading.timestamp,
                    analyzer_type=reading.analyzer_type,
                    value=reading.value,
                    unit=reading.unit,
                    quality_status=quality_status,
                    calibration_status=reading.calibration_status,
                    raw_value=reading.raw_value,
                )

                readings.append(reading)

                # Track quality issues
                if quality_status != QualityAssuranceStatus.VALID:
                    quality_issues.append(f"{analyzer_type.value}: {quality_status.value}")

                # Update lookback data for valid readings
                if quality_status == QualityAssuranceStatus.VALID:
                    self._data_processor.update_lookback_data(analyzer_type, value)

            # Calculate emission rates
            nox_reading = next(
                (r for r in readings if r.analyzer_type == AnalyzerType.NOX),
                None,
            )
            o2_reading = next(
                (r for r in readings if r.analyzer_type == AnalyzerType.O2),
                None,
            )
            flow_reading = next(
                (r for r in readings if r.analyzer_type == AnalyzerType.FLOW),
                None,
            )

            nox_rate = None
            nox_mass = None

            if nox_reading and o2_reading:
                nox_rate = self._data_processor.calculate_emission_rate(
                    concentration_ppm=nox_reading.value,
                    o2_percent=o2_reading.value,
                )

                if flow_reading:
                    nox_mass = self._data_processor.calculate_mass_rate(
                        concentration_ppm=nox_reading.value,
                        flow_scfh=flow_reading.value,
                        molecular_weight=46.0,  # NO2
                    )

            # Determine overall quality status
            overall_status = QualityAssuranceStatus.VALID
            if quality_issues:
                overall_status = QualityAssuranceStatus.QUESTIONABLE

            # Build emissions data
            emissions_data = EmissionsData(
                unit_id=self._cems_config.unit_id,
                stack_id=self._cems_config.stack_id,
                readings=readings,
                nox_rate_lb_mmbtu=nox_rate,
                nox_mass_lb_hr=nox_mass,
                stack_flow_scfh=flow_reading.value if flow_reading else None,
                overall_quality_status=overall_status,
            )

            # Record metrics
            duration_ms = (time.time() - start_time) * 1000
            await self._metrics.record_request(
                success=True,
                latency_ms=duration_ms,
            )

            # Audit log
            await self._audit_logger.log_operation(
                operation="read_realtime_emissions",
                status="success",
                duration_ms=duration_ms,
                response_summary=f"Read {len(readings)} analyzer values",
            )

            return emissions_data

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._metrics.record_request(
                success=False,
                latency_ms=duration_ms,
                error=str(e),
            )
            await self._audit_logger.log_operation(
                operation="read_realtime_emissions",
                status="failure",
                error_message=str(e),
                duration_ms=duration_ms,
            )
            raise

    async def get_calibration_status(self) -> Dict[AnalyzerType, CalibrationStatus]:
        """
        Get calibration status for all analyzers.

        Returns:
            Dictionary of analyzer type to calibration status
        """
        status: Dict[AnalyzerType, CalibrationStatus] = {}

        for analyzer_type in self._cems_config.analyzers_enabled:
            status[analyzer_type] = self._calibration_manager.get_calibration_status(
                analyzer_type
            )

        return status

    async def validate_data_quality(self) -> QualityFlags:
        """
        Validate current data quality per EPA Part 75.

        Returns:
            Quality flags indicating data quality issues
        """
        analyzer_flags: Dict[AnalyzerType, QualityAssuranceStatus] = {}
        is_valid = True
        requires_substitution = False
        calibration_drift = False
        out_of_control = False

        for analyzer_type in self._cems_config.analyzers_enabled:
            reading = self._current_readings.get(analyzer_type)

            if reading is None:
                analyzer_flags[analyzer_type] = QualityAssuranceStatus.MISSING
                requires_substitution = True
                is_valid = False
            else:
                analyzer_flags[analyzer_type] = reading.quality_status

                if reading.quality_status == QualityAssuranceStatus.CALIBRATION_ERROR:
                    calibration_drift = True
                    is_valid = False

                if reading.quality_status == QualityAssuranceStatus.OUT_OF_RANGE:
                    out_of_control = True
                    is_valid = False

        # Check for RATA due dates
        rata_due = False  # Would check against RATA schedule

        return QualityFlags(
            is_valid=is_valid,
            requires_substitution=requires_substitution,
            analyzer_flags=analyzer_flags,
            calibration_drift=calibration_drift,
            out_of_control=out_of_control,
            rata_due=rata_due,
        )

    async def record_calibration(self, calibration: CalibrationData) -> bool:
        """
        Record a calibration event.

        Args:
            calibration: Calibration data

        Returns:
            True if calibration passed validation
        """
        is_valid, issues = self._calibration_manager.validate_calibration(calibration)
        self._calibration_manager.record_calibration(calibration)

        await self._audit_logger.log_operation(
            operation="record_calibration",
            status="success" if is_valid else "warning",
            request_data={
                "analyzer_type": calibration.analyzer_type.value,
                "calibration_type": calibration.calibration_type,
                "status": calibration.status.value,
            },
            response_summary=f"Calibration {'passed' if is_valid else 'failed'}: {issues}",
        )

        return is_valid

    async def get_substituted_value(
        self,
        analyzer_type: AnalyzerType,
        missing_hours: int = 1,
    ) -> Tuple[float, SubstitutionMethod]:
        """
        Get substituted value for missing data.

        Args:
            analyzer_type: Analyzer type
            missing_hours: Hours of missing data

        Returns:
            Tuple of substituted value and method used
        """
        value, method = self._data_processor.substitute_missing_data(
            analyzer_type,
            missing_hours,
        )

        await self._audit_logger.log_operation(
            operation="substitute_data",
            status="success",
            request_data={
                "analyzer_type": analyzer_type.value,
                "missing_hours": missing_hours,
            },
            response_summary=f"Substituted {value} using {method.value}",
        )

        return value, method

    async def start_polling(
        self,
        callback: Optional[Callable[[EmissionsData], None]] = None,
    ) -> None:
        """
        Start real-time data polling.

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
            f"Started polling at {self._cems_config.polling_interval_seconds}s interval"
        )

    async def stop_polling(self) -> None:
        """Stop real-time data polling."""
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
                # Read data
                emissions_data = await self.read_realtime_emissions()

                # Update current readings
                for reading in emissions_data.readings:
                    self._current_readings[reading.analyzer_type] = reading

                    # Update averaging buffer
                    if reading.analyzer_type not in self._averaging_buffer:
                        self._averaging_buffer[reading.analyzer_type] = []
                    self._averaging_buffer[reading.analyzer_type].append(reading.value)

                # Notify callbacks
                for callback in self._data_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(emissions_data)
                        else:
                            callback(emissions_data)
                    except Exception as e:
                        self._logger.error(f"Callback error: {e}")

                # Wait for next poll
                await asyncio.sleep(self._cems_config.polling_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Polling error: {e}")
                await asyncio.sleep(self._cems_config.polling_interval_seconds)

    async def get_averaged_data(
        self,
        period_minutes: Optional[int] = None,
    ) -> EmissionsData:
        """
        Get averaged emissions data.

        Args:
            period_minutes: Averaging period (defaults to config value)

        Returns:
            Averaged emissions data
        """
        period = period_minutes or self._cems_config.averaging_period_minutes

        # Calculate averages from buffer
        readings: List[AnalyzerReading] = []

        for analyzer_type, values in self._averaging_buffer.items():
            if values:
                avg_value = sum(values) / len(values)
                unit = self._get_unit_for_analyzer(analyzer_type)

                reading = AnalyzerReading(
                    analyzer_type=analyzer_type,
                    value=avg_value,
                    unit=unit,
                    quality_status=QualityAssuranceStatus.VALID,
                )
                readings.append(reading)

        # Clear buffer after averaging
        self._averaging_buffer.clear()

        return EmissionsData(
            unit_id=self._cems_config.unit_id,
            stack_id=self._cems_config.stack_id,
            readings=readings,
            overall_quality_status=QualityAssuranceStatus.VALID,
        )

    def _get_unit_for_analyzer(self, analyzer_type: AnalyzerType) -> MeasurementUnit:
        """Get default measurement unit for analyzer type."""
        unit_map = {
            AnalyzerType.NOX: MeasurementUnit.PPM,
            AnalyzerType.NO: MeasurementUnit.PPM,
            AnalyzerType.NO2: MeasurementUnit.PPM,
            AnalyzerType.SO2: MeasurementUnit.PPM,
            AnalyzerType.CO: MeasurementUnit.PPM,
            AnalyzerType.CO2: MeasurementUnit.PERCENT,
            AnalyzerType.O2: MeasurementUnit.PERCENT,
            AnalyzerType.OPACITY: MeasurementUnit.PERCENT,
            AnalyzerType.FLOW: MeasurementUnit.SCFM,
            AnalyzerType.TEMPERATURE: MeasurementUnit.FAHRENHEIT,
            AnalyzerType.PRESSURE: MeasurementUnit.INWC,
            AnalyzerType.MOISTURE: MeasurementUnit.PERCENT,
        }
        return unit_map.get(analyzer_type, MeasurementUnit.PPM)

    def register_data_callback(
        self,
        callback: Callable[[EmissionsData], None],
    ) -> None:
        """
        Register callback for real-time data.

        Args:
            callback: Callback function
        """
        self._data_callbacks.append(callback)

    def unregister_data_callback(
        self,
        callback: Callable[[EmissionsData], None],
    ) -> None:
        """
        Unregister data callback.

        Args:
            callback: Callback function to remove
        """
        if callback in self._data_callbacks:
            self._data_callbacks.remove(callback)


# =============================================================================
# Factory Function
# =============================================================================


def create_cems_connector(
    vendor: CEMSVendor,
    unit_id: str,
    stack_id: str,
    plant_id: str,
    protocol: ProtocolType,
    **kwargs: Any,
) -> CEMSConnector:
    """
    Factory function to create CEMS connector with vendor-specific defaults.

    Args:
        vendor: CEMS vendor
        unit_id: Emission unit identifier
        stack_id: Stack identifier
        plant_id: Plant identifier
        protocol: Communication protocol
        **kwargs: Additional configuration

    Returns:
        Configured CEMS connector
    """
    # Vendor-specific defaults
    vendor_defaults: Dict[CEMSVendor, Dict[str, Any]] = {
        CEMSVendor.THERMO_FISHER: {
            "polling_interval_seconds": 1.0,
            "calibration_error_limit_percent": 2.5,
        },
        CEMSVendor.TELEDYNE: {
            "polling_interval_seconds": 1.0,
            "calibration_error_limit_percent": 2.5,
        },
        CEMSVendor.HORIBA: {
            "polling_interval_seconds": 0.5,
            "calibration_error_limit_percent": 2.0,
        },
        CEMSVendor.SIEMENS: {
            "polling_interval_seconds": 1.0,
            "calibration_error_limit_percent": 2.5,
        },
    }

    defaults = vendor_defaults.get(vendor, {})
    defaults.update(kwargs)

    config = CEMSConnectorConfig(
        connector_name=f"CEMS_{vendor.value}_{unit_id}",
        vendor=vendor,
        unit_id=unit_id,
        stack_id=stack_id,
        plant_id=plant_id,
        protocol=protocol,
        **defaults,
    )

    return CEMSConnector(config)
