"""
GL-009 ThermalIQ - Event Schemas

Avro-compatible schemas for Kafka event streaming.

Event Types:
- AnalysisRequestedEvent: Thermal analysis request submitted
- AnalysisCompletedEvent: Analysis completed with results
- FluidPropertyUpdatedEvent: Fluid property lookup/update
- ExergyCalculatedEvent: Exergy analysis completed
- SankeyGeneratedEvent: Sankey diagram generated
- AlertEvent: System alerts and notifications
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import json
import uuid


# =============================================================================
# Enums
# =============================================================================

class EventType(str, Enum):
    """Event types for ThermalIQ streaming."""
    ANALYSIS_REQUESTED = "analysis_requested"
    ANALYSIS_COMPLETED = "analysis_completed"
    FLUID_PROPERTY_UPDATED = "fluid_property_updated"
    EXERGY_CALCULATED = "exergy_calculated"
    SANKEY_GENERATED = "sankey_generated"
    EFFICIENCY_CALCULATED = "efficiency_calculated"
    RECOMMENDATION_GENERATED = "recommendation_generated"
    ALERT = "alert"
    SENSOR_DATA = "sensor_data"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataQuality(str, Enum):
    """Data quality indicators."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"
    INTERPOLATED = "interpolated"


class FluidPhase(str, Enum):
    """Fluid phase states."""
    LIQUID = "liquid"
    GAS = "gas"
    TWO_PHASE = "two_phase"
    SUPERCRITICAL = "supercritical"


# =============================================================================
# Message Header
# =============================================================================

@dataclass
class MessageHeader:
    """Standard message header for all ThermalIQ events."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.ANALYSIS_REQUESTED
    version: str = "1.0"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "thermaliq"
    correlation_id: Optional[str] = None
    provenance_hash: Optional[str] = None
    tenant_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "event_type": self.event_type.value if isinstance(self.event_type, Enum) else self.event_type,
            "version": self.version,
            "timestamp": self.timestamp,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "provenance_hash": self.provenance_hash,
            "tenant_id": self.tenant_id,
        }


# =============================================================================
# Base Event
# =============================================================================

@dataclass
class BaseEvent:
    """Base event with common fields."""
    header: MessageHeader = field(default_factory=MessageHeader)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["header"] = self.header.to_dict()
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def to_bytes(self) -> bytes:
        """Convert to bytes for Kafka."""
        return self.to_json().encode("utf-8")

    def compute_hash(self) -> str:
        """Compute SHA-256 hash for provenance."""
        data = self.to_json()
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# =============================================================================
# Analysis Events
# =============================================================================

@dataclass
class StreamData:
    """Stream data within events."""
    stream_id: str
    fluid_name: str
    inlet_temperature_C: float
    outlet_temperature_C: float
    pressure_kPa: float
    mass_flow_kg_s: float
    specific_heat_kJ_kgK: Optional[float] = None
    phase: Optional[str] = None


@dataclass
class AnalysisRequestedEvent(BaseEvent):
    """Event for thermal analysis request submission."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    streams: List[Dict[str, Any]] = field(default_factory=list)
    ambient_temperature_C: float = 25.0
    ambient_pressure_kPa: float = 101.325
    analysis_mode: str = "full"
    include_exergy: bool = True
    include_sankey: bool = True
    requester_id: Optional[str] = None

    def __post_init__(self):
        self.header.event_type = EventType.ANALYSIS_REQUESTED


@dataclass
class AnalysisCompletedEvent(BaseEvent):
    """Event for completed thermal analysis."""
    request_id: str = ""
    status: str = "completed"

    # Results
    total_heat_duty_kW: float = 0.0
    total_mass_flow_kg_s: float = 0.0
    first_law_efficiency_percent: float = 0.0
    second_law_efficiency_percent: Optional[float] = None

    # Stream results
    stream_results: List[Dict[str, Any]] = field(default_factory=list)

    # Exergy results (if included)
    exergy_destruction_kW: Optional[float] = None
    exergy_efficiency_percent: Optional[float] = None
    improvement_potential_kW: Optional[float] = None

    # Sankey data (if included)
    sankey_node_count: Optional[int] = None
    sankey_link_count: Optional[int] = None

    # Processing metadata
    computation_hash: str = ""
    processing_time_ms: float = 0.0

    def __post_init__(self):
        self.header.event_type = EventType.ANALYSIS_COMPLETED


# =============================================================================
# Fluid Property Events
# =============================================================================

@dataclass
class FluidPropertyUpdatedEvent(BaseEvent):
    """Event for fluid property lookup/update."""
    fluid_name: str = ""
    temperature_C: float = 0.0
    pressure_kPa: float = 0.0
    phase: str = "liquid"

    # Thermodynamic properties
    density_kg_m3: float = 0.0
    specific_heat_kJ_kgK: float = 0.0
    enthalpy_kJ_kg: float = 0.0
    entropy_kJ_kgK: float = 0.0
    internal_energy_kJ_kg: float = 0.0

    # Transport properties
    viscosity_Pa_s: float = 0.0
    thermal_conductivity_W_mK: float = 0.0
    prandtl_number: float = 0.0

    # Quality indicators
    quality: Optional[float] = None
    data_quality: DataQuality = DataQuality.GOOD
    data_source: str = "CoolProp"

    # Computation
    computation_hash: str = ""

    def __post_init__(self):
        self.header.event_type = EventType.FLUID_PROPERTY_UPDATED


# =============================================================================
# Exergy Events
# =============================================================================

@dataclass
class ExergyComponent:
    """Component exergy breakdown."""
    name: str
    exergy_input_kW: float
    exergy_output_kW: float
    exergy_destruction_kW: float
    exergy_efficiency_percent: float
    irreversibility_kW: float


@dataclass
class ExergyCalculatedEvent(BaseEvent):
    """Event for exergy analysis completion."""
    request_id: str = ""

    # Dead state conditions
    dead_state_temperature_C: float = 25.0
    dead_state_pressure_kPa: float = 101.325

    # Total exergy metrics
    total_exergy_input_kW: float = 0.0
    total_exergy_output_kW: float = 0.0
    total_exergy_destruction_kW: float = 0.0
    exergy_efficiency_percent: float = 0.0

    # Exergy breakdown
    physical_exergy_kW: float = 0.0
    chemical_exergy_kW: Optional[float] = None
    kinetic_exergy_kW: Optional[float] = None
    potential_exergy_kW: Optional[float] = None

    # Component analysis
    components: List[Dict[str, Any]] = field(default_factory=list)

    # Improvement
    improvement_potential_kW: float = 0.0

    # Provenance
    computation_hash: str = ""
    processing_time_ms: float = 0.0

    def __post_init__(self):
        self.header.event_type = EventType.EXERGY_CALCULATED


# =============================================================================
# Sankey Events
# =============================================================================

@dataclass
class SankeyNode:
    """Sankey diagram node."""
    id: str
    name: str
    value: float
    category: str
    color: Optional[str] = None


@dataclass
class SankeyLink:
    """Sankey diagram link."""
    source: str
    target: str
    value: float
    label: Optional[str] = None
    color: Optional[str] = None


@dataclass
class SankeyGeneratedEvent(BaseEvent):
    """Event for Sankey diagram generation."""
    request_id: str = ""
    diagram_type: str = "energy"  # energy or exergy

    # Diagram data
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)

    # Summary
    total_input_kW: float = 0.0
    total_output_kW: float = 0.0
    total_losses_kW: float = 0.0

    # Rendering hints
    layout_direction: str = "left_to_right"
    color_scheme: str = "thermal"

    # Provenance
    computation_hash: str = ""
    processing_time_ms: float = 0.0

    def __post_init__(self):
        self.header.event_type = EventType.SANKEY_GENERATED


# =============================================================================
# Efficiency Events
# =============================================================================

@dataclass
class EfficiencyCalculatedEvent(BaseEvent):
    """Event for efficiency calculation."""
    request_id: str = ""
    method: str = "combined"

    # First law efficiency
    first_law_efficiency_percent: float = 0.0
    energy_input_kW: float = 0.0
    energy_output_kW: float = 0.0
    energy_loss_kW: float = 0.0

    # Second law efficiency
    second_law_efficiency_percent: Optional[float] = None
    exergy_input_kW: Optional[float] = None
    exergy_output_kW: Optional[float] = None
    exergy_destruction_kW: Optional[float] = None

    # Provenance
    computation_hash: str = ""
    processing_time_ms: float = 0.0

    def __post_init__(self):
        self.header.event_type = EventType.EFFICIENCY_CALCULATED


# =============================================================================
# Alert Events
# =============================================================================

@dataclass
class AlertEvent(BaseEvent):
    """Event for system alerts and notifications."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: str = ""
    alert_name: str = ""
    severity: AlertSeverity = AlertSeverity.INFO
    description: str = ""
    recommended_actions: List[str] = field(default_factory=list)

    # Context
    affected_streams: List[str] = field(default_factory=list)
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None

    # Impact
    potential_efficiency_loss_percent: Optional[float] = None
    potential_exergy_loss_kW: Optional[float] = None

    # Acknowledgment
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None

    def __post_init__(self):
        self.header.event_type = EventType.ALERT


# =============================================================================
# Sensor Data Events
# =============================================================================

@dataclass
class SensorDataEvent(BaseEvent):
    """Event for real-time sensor data."""
    sensor_id: str = ""
    stream_id: str = ""
    measurement_type: str = ""  # temperature, pressure, flow, etc.

    # Measurement
    value: float = 0.0
    unit: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Quality
    data_quality: DataQuality = DataQuality.GOOD
    is_interpolated: bool = False
    is_stale: bool = False

    def __post_init__(self):
        self.header.event_type = EventType.SENSOR_DATA


# =============================================================================
# Avro Schemas for Schema Registry
# =============================================================================

AVRO_SCHEMAS = {
    "analysis_requested": {
        "type": "record",
        "name": "AnalysisRequestedEvent",
        "namespace": "com.greenlang.thermaliq",
        "fields": [
            {"name": "message_id", "type": "string"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "version", "type": "string"},
            {"name": "source", "type": "string"},
            {"name": "request_id", "type": "string"},
            {"name": "streams", "type": {"type": "array", "items": {
                "type": "record",
                "name": "StreamData",
                "fields": [
                    {"name": "stream_id", "type": "string"},
                    {"name": "fluid_name", "type": "string"},
                    {"name": "inlet_temperature_C", "type": "double"},
                    {"name": "outlet_temperature_C", "type": "double"},
                    {"name": "pressure_kPa", "type": "double"},
                    {"name": "mass_flow_kg_s", "type": "double"},
                ]
            }}},
            {"name": "ambient_temperature_C", "type": "double"},
            {"name": "analysis_mode", "type": "string"},
            {"name": "include_exergy", "type": "boolean"},
            {"name": "include_sankey", "type": "boolean"},
        ],
    },
    "analysis_completed": {
        "type": "record",
        "name": "AnalysisCompletedEvent",
        "namespace": "com.greenlang.thermaliq",
        "fields": [
            {"name": "message_id", "type": "string"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "request_id", "type": "string"},
            {"name": "status", "type": "string"},
            {"name": "total_heat_duty_kW", "type": "double"},
            {"name": "total_mass_flow_kg_s", "type": "double"},
            {"name": "first_law_efficiency_percent", "type": "double"},
            {"name": "second_law_efficiency_percent", "type": ["null", "double"]},
            {"name": "exergy_destruction_kW", "type": ["null", "double"]},
            {"name": "computation_hash", "type": "string"},
            {"name": "processing_time_ms", "type": "double"},
        ],
    },
    "fluid_property_updated": {
        "type": "record",
        "name": "FluidPropertyUpdatedEvent",
        "namespace": "com.greenlang.thermaliq",
        "fields": [
            {"name": "message_id", "type": "string"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "fluid_name", "type": "string"},
            {"name": "temperature_C", "type": "double"},
            {"name": "pressure_kPa", "type": "double"},
            {"name": "phase", "type": "string"},
            {"name": "density_kg_m3", "type": "double"},
            {"name": "specific_heat_kJ_kgK", "type": "double"},
            {"name": "enthalpy_kJ_kg", "type": "double"},
            {"name": "entropy_kJ_kgK", "type": "double"},
            {"name": "viscosity_Pa_s", "type": "double"},
            {"name": "thermal_conductivity_W_mK", "type": "double"},
            {"name": "prandtl_number", "type": "double"},
            {"name": "data_source", "type": "string"},
            {"name": "computation_hash", "type": "string"},
        ],
    },
    "exergy_calculated": {
        "type": "record",
        "name": "ExergyCalculatedEvent",
        "namespace": "com.greenlang.thermaliq",
        "fields": [
            {"name": "message_id", "type": "string"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "request_id", "type": "string"},
            {"name": "dead_state_temperature_C", "type": "double"},
            {"name": "dead_state_pressure_kPa", "type": "double"},
            {"name": "total_exergy_input_kW", "type": "double"},
            {"name": "total_exergy_output_kW", "type": "double"},
            {"name": "total_exergy_destruction_kW", "type": "double"},
            {"name": "exergy_efficiency_percent", "type": "double"},
            {"name": "physical_exergy_kW", "type": "double"},
            {"name": "improvement_potential_kW", "type": "double"},
            {"name": "computation_hash", "type": "string"},
            {"name": "processing_time_ms", "type": "double"},
        ],
    },
    "sankey_generated": {
        "type": "record",
        "name": "SankeyGeneratedEvent",
        "namespace": "com.greenlang.thermaliq",
        "fields": [
            {"name": "message_id", "type": "string"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "request_id", "type": "string"},
            {"name": "diagram_type", "type": "string"},
            {"name": "node_count", "type": "int"},
            {"name": "link_count", "type": "int"},
            {"name": "total_input_kW", "type": "double"},
            {"name": "total_output_kW", "type": "double"},
            {"name": "total_losses_kW", "type": "double"},
            {"name": "computation_hash", "type": "string"},
            {"name": "processing_time_ms", "type": "double"},
        ],
    },
    "alert": {
        "type": "record",
        "name": "AlertEvent",
        "namespace": "com.greenlang.thermaliq",
        "fields": [
            {"name": "message_id", "type": "string"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "alert_id", "type": "string"},
            {"name": "alert_type", "type": "string"},
            {"name": "alert_name", "type": "string"},
            {"name": "severity", "type": "string"},
            {"name": "description", "type": "string"},
            {"name": "recommended_actions", "type": {"type": "array", "items": "string"}},
            {"name": "affected_streams", "type": {"type": "array", "items": "string"}},
            {"name": "current_value", "type": ["null", "double"]},
            {"name": "threshold_value", "type": ["null", "double"]},
            {"name": "acknowledged", "type": "boolean"},
        ],
    },
}


def get_avro_schema(event_type: str) -> Dict:
    """Get Avro schema for event type."""
    return AVRO_SCHEMAS.get(event_type, {})
