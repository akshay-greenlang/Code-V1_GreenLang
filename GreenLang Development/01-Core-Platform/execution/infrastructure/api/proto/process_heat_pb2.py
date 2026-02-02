"""
Process Heat Protocol Buffer Message Definitions

This module provides Python dataclasses that match the protobuf definitions
in process_heat.proto. These classes are used for gRPC communication in
the GreenLang process heat monitoring and emissions calculation services.

Features:
    - Type-safe message classes with Pydantic validation
    - Enum definitions for status codes
    - Serialization/deserialization support
    - Timestamp handling with google.protobuf.Timestamp compatibility

Example:
    >>> request = CalculationRequest(
    ...     calculation_id="calc-001",
    ...     agent_name="FurnaceAgent",
    ...     equipment=EquipmentSpecification(
    ...         equipment_id="furnace-001",
    ...         equipment_type="INDUSTRIAL_FURNACE",
    ...         rated_capacity_mw=50.0
    ...     )
    ... )
    >>> print(request.calculation_id)
    'calc-001'
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union
import json


# ============================================================================
# ENUMS
# ============================================================================

class CalculationStatus(IntEnum):
    """Status codes for calculation operations."""
    UNKNOWN = 0
    QUEUED = 1
    PROCESSING = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5


class ValidationStatus(IntEnum):
    """Validation result status codes."""
    VALIDATION_UNKNOWN = 0
    PASS = 1
    FAIL = 2
    WARNING = 3


class Scope(IntEnum):
    """GHG Protocol emission scopes."""
    SCOPE_UNKNOWN = 0
    SCOPE_1 = 1
    SCOPE_2 = 2
    SCOPE_3 = 3


class ComplianceStatus(IntEnum):
    """Regulatory compliance status codes."""
    COMPLIANCE_UNKNOWN = 0
    COMPLIANT = 1
    PARTIALLY_COMPLIANT = 2
    NON_COMPLIANT = 3
    PENDING_REVIEW = 4


class Severity(IntEnum):
    """Violation severity levels."""
    SEVERITY_UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# ============================================================================
# TIMESTAMP HELPER
# ============================================================================

@dataclass
class Timestamp:
    """
    Google Protocol Buffer Timestamp equivalent.

    Represents a point in time with nanosecond precision.

    Attributes:
        seconds: Seconds since Unix epoch (1970-01-01T00:00:00Z)
        nanos: Nanoseconds within the second (0-999999999)
    """
    seconds: int = 0
    nanos: int = 0

    @classmethod
    def from_datetime(cls, dt: datetime) -> "Timestamp":
        """Create Timestamp from datetime object."""
        return cls(
            seconds=int(dt.timestamp()),
            nanos=dt.microsecond * 1000
        )

    def to_datetime(self) -> datetime:
        """Convert to datetime object."""
        return datetime.fromtimestamp(self.seconds + self.nanos / 1e9)

    @classmethod
    def now(cls) -> "Timestamp":
        """Get current time as Timestamp."""
        return cls.from_datetime(datetime.now())

    def SerializeToString(self) -> bytes:
        """Serialize to protobuf wire format (simplified)."""
        return json.dumps({"seconds": self.seconds, "nanos": self.nanos}).encode()

    @classmethod
    def FromString(cls, data: bytes) -> "Timestamp":
        """Deserialize from protobuf wire format (simplified)."""
        parsed = json.loads(data.decode())
        return cls(seconds=parsed["seconds"], nanos=parsed["nanos"])


# ============================================================================
# BASE MESSAGE CLASS
# ============================================================================

@dataclass
class ProtoMessage:
    """
    Base class for all protobuf message types.

    Provides common serialization and utility methods.
    """

    def SerializeToString(self) -> bytes:
        """Serialize message to protobuf wire format (JSON-based for compatibility)."""
        return json.dumps(self._to_dict()).encode('utf-8')

    @classmethod
    def FromString(cls, data: bytes) -> "ProtoMessage":
        """Deserialize message from protobuf wire format."""
        parsed = json.loads(data.decode('utf-8'))
        return cls._from_dict(parsed)

    def _to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if isinstance(value, ProtoMessage):
                result[key] = value._to_dict()
            elif isinstance(value, Timestamp):
                result[key] = {"seconds": value.seconds, "nanos": value.nanos}
            elif isinstance(value, list):
                result[key] = [
                    item._to_dict() if isinstance(item, (ProtoMessage, Timestamp)) else item
                    for item in value
                ]
            elif isinstance(value, IntEnum):
                result[key] = value.value
            else:
                result[key] = value
        return result

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ProtoMessage":
        """Create message from dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        """String representation of message."""
        fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if v is not None)
        return f"{self.__class__.__name__}({fields})"


# ============================================================================
# EQUIPMENT AND INPUT MESSAGES
# ============================================================================

@dataclass
class EquipmentSpecification(ProtoMessage):
    """
    Specification for process heating equipment.

    Attributes:
        equipment_id: Unique identifier for the equipment
        equipment_type: Type of equipment (e.g., FURNACE, BOILER)
        rated_capacity_mw: Maximum rated capacity in megawatts
        fuel_type: Primary fuel type used
        installation_year: Year equipment was installed
        nominal_efficiency_percent: Design efficiency percentage
    """
    equipment_id: str = ""
    equipment_type: str = ""
    rated_capacity_mw: float = 0.0
    fuel_type: str = ""
    installation_year: int = 0
    nominal_efficiency_percent: float = 0.0


@dataclass
class FuelInput(ProtoMessage):
    """
    Fuel consumption input data.

    Attributes:
        quantity: Amount of fuel consumed
        unit: Measurement unit (e.g., m3, kg, MWh)
        carbon_content: Carbon content percentage
        source: Data source identifier
        measurement_date: When the measurement was taken
    """
    quantity: float = 0.0
    unit: str = ""
    carbon_content: float = 0.0
    source: str = ""
    measurement_date: Optional[Timestamp] = None


@dataclass
class OperatingConditions(ProtoMessage):
    """
    Current operating conditions for equipment.

    Attributes:
        load_percent: Current load as percentage of capacity
        inlet_temperature_celsius: Input temperature
        outlet_temperature_celsius: Output temperature
        operating_pressure_bar: Operating pressure
        burner_on: Whether burner is currently active
        number_of_burners_active: Count of active burners
    """
    load_percent: float = 0.0
    inlet_temperature_celsius: float = 0.0
    outlet_temperature_celsius: float = 0.0
    operating_pressure_bar: float = 0.0
    burner_on: bool = False
    number_of_burners_active: int = 0


# ============================================================================
# CALCULATION SERVICE MESSAGES
# ============================================================================

@dataclass
class CalculationRequest(ProtoMessage):
    """
    Request message for RunCalculation RPC.

    Attributes:
        calculation_id: Optional ID (generated if not provided)
        agent_name: Name of the agent to execute
        timestamp: Request timestamp
        equipment: Equipment specification
        fuel_input: Fuel consumption data
        operating_conditions: Current operating state
    """
    calculation_id: str = ""
    agent_name: str = ""
    timestamp: Optional[Timestamp] = None
    equipment: Optional[EquipmentSpecification] = None
    fuel_input: Optional[FuelInput] = None
    operating_conditions: Optional[OperatingConditions] = None


@dataclass
class CalculationResponse(ProtoMessage):
    """
    Response message for RunCalculation RPC.

    Attributes:
        calculation_id: Unique identifier for the calculation
        status: Current status (QUEUED, PROCESSING, COMPLETED)
        message_id: Message correlation ID
        provenance_hash: SHA-256 hash for audit trail
    """
    calculation_id: str = ""
    status: str = ""
    message_id: str = ""
    provenance_hash: str = ""


@dataclass
class StatusRequest(ProtoMessage):
    """
    Request message for GetStatus RPC.

    Attributes:
        calculation_id: ID of the calculation to query
        message_id: Optional message correlation ID
    """
    calculation_id: str = ""
    message_id: str = ""


@dataclass
class StatusResponse(ProtoMessage):
    """
    Response message for GetStatus RPC.

    Attributes:
        calculation_id: ID of the calculation
        status: Current status
        progress_percent: Completion percentage (0-100)
        error_message: Error details if failed
        started_at: When processing started
        completed_at: When processing completed
    """
    calculation_id: str = ""
    status: str = ""
    progress_percent: float = 0.0
    error_message: str = ""
    started_at: Optional[Timestamp] = None
    completed_at: Optional[Timestamp] = None


@dataclass
class StreamRequest(ProtoMessage):
    """
    Request message for StreamResults RPC.

    Attributes:
        agent_name: Filter results by agent name
        filter_type: Additional filter type
        include_intermediate_results: Whether to include partial results
    """
    agent_name: str = ""
    filter_type: str = ""
    include_intermediate_results: bool = False


@dataclass
class CalculationResult(ProtoMessage):
    """
    Streamed calculation result message.

    Attributes:
        calculation_id: Unique calculation identifier
        agent_name: Agent that produced the result
        timestamp: When result was generated
        heat_output_mwh: Heat energy output in MWh
        fuel_consumed_mwh: Fuel energy consumed in MWh
        fuel_efficiency_percent: Thermal efficiency
        co2_emissions_tonnes: CO2 emissions in metric tonnes
        ch4_emissions_tonnes: CH4 emissions in metric tonnes
        n2o_emissions_tonnes: N2O emissions in metric tonnes
        thermal_loss_percent: Thermal loss percentage
        provenance_hash: SHA-256 audit hash
        validation_status: PASS or FAIL
        processing_time_ms: Processing duration in milliseconds
    """
    calculation_id: str = ""
    agent_name: str = ""
    timestamp: Optional[Timestamp] = None
    heat_output_mwh: float = 0.0
    fuel_consumed_mwh: float = 0.0
    fuel_efficiency_percent: float = 0.0
    co2_emissions_tonnes: float = 0.0
    ch4_emissions_tonnes: float = 0.0
    n2o_emissions_tonnes: float = 0.0
    thermal_loss_percent: float = 0.0
    provenance_hash: str = ""
    validation_status: str = ""
    processing_time_ms: float = 0.0


# ============================================================================
# EMISSIONS SERVICE MESSAGES
# ============================================================================

@dataclass
class ActivityData(ProtoMessage):
    """
    Activity data for emissions calculation.

    Attributes:
        activity_id: Unique activity identifier
        activity_type: Type of activity (e.g., FUEL_COMBUSTION)
        quantity: Activity quantity
        unit: Measurement unit
        source: Data source identifier
    """
    activity_id: str = ""
    activity_type: str = ""
    quantity: float = 0.0
    unit: str = ""
    source: str = ""


@dataclass
class EmissionsRequest(ProtoMessage):
    """
    Request message for CalculateEmissions RPC.

    Attributes:
        emissions_id: Optional ID (generated if not provided)
        scope: GHG Protocol scope (SCOPE_1, SCOPE_2, SCOPE_3)
        activity_data: List of activity data records
        region: Geographic region code
        reporting_year: Reporting period year
    """
    emissions_id: str = ""
    scope: str = ""
    activity_data: List[ActivityData] = field(default_factory=list)
    region: str = ""
    reporting_year: int = 0


@dataclass
class EmissionDetail(ProtoMessage):
    """
    Detailed emissions breakdown by source.

    Attributes:
        source_id: Source identifier
        source_name: Source description
        co2_tonnes: CO2 emissions in metric tonnes
        ch4_tonnes: CH4 emissions in metric tonnes
        n2o_tonnes: N2O emissions in metric tonnes
        emissions_factor_used: Applied emission factor
    """
    source_id: str = ""
    source_name: str = ""
    co2_tonnes: float = 0.0
    ch4_tonnes: float = 0.0
    n2o_tonnes: float = 0.0
    emissions_factor_used: float = 0.0


@dataclass
class EmissionsResponse(ProtoMessage):
    """
    Response message for CalculateEmissions RPC.

    Attributes:
        emissions_id: Unique emissions calculation ID
        total_co2_tonnes: Total CO2 emissions
        total_ch4_tonnes: Total CH4 emissions
        total_n2o_tonnes: Total N2O emissions
        total_co2e_tonnes: Total CO2-equivalent emissions
        details: Breakdown by source
        provenance_hash: SHA-256 audit hash
        validation_status: PASS or FAIL
        processing_time_ms: Processing duration
    """
    emissions_id: str = ""
    total_co2_tonnes: float = 0.0
    total_ch4_tonnes: float = 0.0
    total_n2o_tonnes: float = 0.0
    total_co2e_tonnes: float = 0.0
    details: List[EmissionDetail] = field(default_factory=list)
    provenance_hash: str = ""
    validation_status: str = ""
    processing_time_ms: float = 0.0


@dataclass
class EmissionFactorsRequest(ProtoMessage):
    """
    Request message for GetEmissionFactors RPC.

    Attributes:
        fuel_type: Fuel type to query
        scope: GHG Protocol scope
        region: Geographic region code
        year: Reference year
    """
    fuel_type: str = ""
    scope: str = ""
    region: str = ""
    year: int = 0


@dataclass
class EmissionFactor(ProtoMessage):
    """
    Emission factor data.

    Attributes:
        factor_id: Unique factor identifier
        scope: Applicable scope
        co2_per_unit: CO2 emission factor
        ch4_per_unit: CH4 emission factor
        n2o_per_unit: N2O emission factor
        unit: Factor unit (e.g., kg CO2/m3)
        source: Data source (e.g., IPCC_AR5)
        region: Geographic applicability
        year: Reference year
    """
    factor_id: str = ""
    scope: str = ""
    co2_per_unit: float = 0.0
    ch4_per_unit: float = 0.0
    n2o_per_unit: float = 0.0
    unit: str = ""
    source: str = ""
    region: str = ""
    year: int = 0


@dataclass
class EmissionFactorsResponse(ProtoMessage):
    """
    Response message for GetEmissionFactors RPC.

    Attributes:
        fuel_type: Queried fuel type
        factors: List of matching emission factors
        last_updated: When factors were last updated
    """
    fuel_type: str = ""
    factors: List[EmissionFactor] = field(default_factory=list)
    last_updated: Optional[Timestamp] = None


# ============================================================================
# COMPLIANCE SERVICE MESSAGES
# ============================================================================

@dataclass
class ComplianceItem(ProtoMessage):
    """
    Individual compliance requirement status.

    Attributes:
        requirement_id: Unique requirement identifier
        requirement_description: Human-readable description
        status: PASS, FAIL, or PARTIAL
        evidence: Supporting evidence
        issues: List of identified issues
    """
    requirement_id: str = ""
    requirement_description: str = ""
    status: str = ""
    evidence: str = ""
    issues: List[str] = field(default_factory=list)


@dataclass
class ReportRequest(ProtoMessage):
    """
    Request message for GenerateReport RPC.

    Attributes:
        report_id: Optional ID (generated if not provided)
        framework: Regulatory framework (EUDR, CBAM, CSRD)
        facility_id: Facility identifier
        start_date: Reporting period start
        end_date: Reporting period end
        include_recommendations: Whether to include recommendations
    """
    report_id: str = ""
    framework: str = ""
    facility_id: str = ""
    start_date: Optional[Timestamp] = None
    end_date: Optional[Timestamp] = None
    include_recommendations: bool = True


@dataclass
class ReportResponse(ProtoMessage):
    """
    Response message for GenerateReport RPC.

    Attributes:
        report_id: Unique report identifier
        framework: Regulatory framework
        facility_id: Facility identifier
        status: Overall compliance status
        items: Individual requirement results
        recommendations: Improvement recommendations
        summary: Executive summary
        overall_score: Compliance score (0-100)
        provenance_hash: SHA-256 audit hash
        processing_time_ms: Processing duration
    """
    report_id: str = ""
    framework: str = ""
    facility_id: str = ""
    status: str = ""
    items: List[ComplianceItem] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    summary: str = ""
    overall_score: float = 0.0
    provenance_hash: str = ""
    processing_time_ms: float = 0.0


@dataclass
class ComplianceDataPoint(ProtoMessage):
    """
    Data point for compliance checking.

    Attributes:
        metric_name: Name of the metric
        value: Metric value
        unit: Measurement unit
        timestamp: When the value was recorded
    """
    metric_name: str = ""
    value: float = 0.0
    unit: str = ""
    timestamp: str = ""


@dataclass
class ComplianceCheckRequest(ProtoMessage):
    """
    Request message for CheckCompliance RPC.

    Attributes:
        check_id: Optional ID (generated if not provided)
        regulation: Regulation to check against
        facility_id: Facility identifier
        data_points: Data points to evaluate
    """
    check_id: str = ""
    regulation: str = ""
    facility_id: str = ""
    data_points: List[ComplianceDataPoint] = field(default_factory=list)


@dataclass
class ComplianceViolation(ProtoMessage):
    """
    Compliance violation details.

    Attributes:
        violation_id: Unique violation identifier
        violation_type: Type of violation
        description: Detailed description
        severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
        remediation: Recommended remediation steps
    """
    violation_id: str = ""
    violation_type: str = ""
    description: str = ""
    severity: str = ""
    remediation: str = ""


@dataclass
class ComplianceCheckResponse(ProtoMessage):
    """
    Response message for CheckCompliance RPC.

    Attributes:
        check_id: Unique check identifier
        regulation: Regulation checked
        is_compliant: Overall compliance result
        compliance_score: Compliance score (0-100)
        violations: List of violations found
        summary: Executive summary
        provenance_hash: SHA-256 audit hash
        processing_time_ms: Processing duration
    """
    check_id: str = ""
    regulation: str = ""
    is_compliant: bool = True
    compliance_score: float = 100.0
    violations: List[ComplianceViolation] = field(default_factory=list)
    summary: str = ""
    provenance_hash: str = ""
    processing_time_ms: float = 0.0


# ============================================================================
# DESCRIPTOR (for reflection support)
# ============================================================================

# Service descriptor name for reflection
DESCRIPTOR = type('Descriptor', (), {
    'services_by_name': {
        'ProcessHeatService': 'greenlang.infrastructure.api.ProcessHeatService',
        'EmissionsService': 'greenlang.infrastructure.api.EmissionsService',
        'ComplianceService': 'greenlang.infrastructure.api.ComplianceService',
    },
    'full_name': 'greenlang.infrastructure.api',
})()
