"""
GL-002 FLAMEGUARD - Event Schemas

Avro-compatible schemas for Kafka event streaming.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import hashlib
import json


class EventType(Enum):
    """Event types for streaming."""
    PROCESS_DATA = "process_data"
    OPTIMIZATION = "optimization"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    EMISSIONS = "emissions"
    ALARM = "alarm"
    SETPOINT_CHANGE = "setpoint_change"
    STATE_CHANGE = "state_change"
    CALCULATION = "calculation"


class Severity(Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class BaseEvent:
    """Base event with common fields."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    boiler_id: str
    source: str = "flameguard"
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def compute_hash(self) -> str:
        """Compute event hash for provenance."""
        data = self.to_json()
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class ProcessDataEvent(BaseEvent):
    """Real-time process data event."""
    event_type: EventType = field(default=EventType.PROCESS_DATA, init=False)

    # Pressure
    drum_pressure_psig: float = 0.0
    fuel_pressure_psig: float = 0.0
    air_pressure_in_wc: float = 0.0

    # Level
    drum_level_inches: float = 0.0

    # Flow
    steam_flow_klb_hr: float = 0.0
    feedwater_flow_klb_hr: float = 0.0
    fuel_flow_scfh: float = 0.0
    air_flow_scfm: float = 0.0

    # Temperature
    steam_temperature_f: float = 0.0
    feedwater_temperature_f: float = 0.0
    flue_gas_temperature_f: float = 0.0
    ambient_temperature_f: float = 70.0

    # Combustion
    o2_percent: float = 0.0
    co_ppm: float = 0.0
    nox_ppm: float = 0.0

    # Operating state
    load_percent: float = 0.0
    firing: bool = False

    # Quality indicators
    data_quality: str = "good"
    stale_tags: List[str] = field(default_factory=list)


@dataclass
class OptimizationEvent(BaseEvent):
    """Optimization run event."""
    event_type: EventType = field(default=EventType.OPTIMIZATION, init=False)

    # Mode
    optimization_mode: str = "balanced"

    # Current state
    current_efficiency: float = 0.0
    current_emissions_mtco2e: float = 0.0
    current_cost_usd: float = 0.0

    # Recommendations
    recommended_o2_setpoint: float = 0.0
    recommended_excess_air: float = 0.0
    recommended_load: Optional[float] = None

    # Predictions
    predicted_efficiency: float = 0.0
    predicted_emissions_mtco2e: float = 0.0
    predicted_cost_usd: float = 0.0

    # Improvements
    efficiency_improvement: float = 0.0
    emissions_reduction: float = 0.0
    cost_savings_usd: float = 0.0

    # Metadata
    model_version: str = "1.0"
    calculation_hash: str = ""
    applied: bool = False


@dataclass
class SafetyEvent(BaseEvent):
    """Safety system event."""
    event_type: EventType = field(default=EventType.SAFETY, init=False)

    # Event details
    safety_event_type: str = ""  # trip, alarm, bypass, reset
    severity: Severity = Severity.INFO
    interlock_tag: Optional[str] = None

    # State
    previous_state: str = ""
    new_state: str = ""

    # Trip details
    trip_cause: Optional[str] = None
    trip_value: Optional[float] = None
    trip_setpoint: Optional[float] = None

    # Bypass details
    bypass_reason: Optional[str] = None
    bypass_operator: Optional[str] = None
    bypass_duration_min: Optional[int] = None

    # Flame status
    flame_proven: bool = True
    flame_signal_percent: float = 0.0

    # BMS state
    bms_state: str = ""


@dataclass
class EfficiencyEvent(BaseEvent):
    """Efficiency calculation event."""
    event_type: EventType = field(default=EventType.EFFICIENCY, init=False)

    # Efficiency values
    gross_efficiency_percent: float = 0.0
    net_efficiency_percent: float = 0.0
    fuel_efficiency_percent: float = 0.0

    # Losses
    stack_loss_percent: float = 0.0
    radiation_loss_percent: float = 0.0
    blowdown_loss_percent: float = 0.0
    unaccounted_loss_percent: float = 0.0

    # Heat balance
    heat_input_mmbtu_hr: float = 0.0
    heat_output_mmbtu_hr: float = 0.0

    # Calculation metadata
    calculation_method: str = "indirect"
    standard: str = "ASME PTC 4.1"
    calculation_hash: str = ""

    # Operating conditions
    load_percent: float = 0.0
    o2_percent: float = 0.0
    flue_gas_temp_f: float = 0.0


@dataclass
class EmissionsEvent(BaseEvent):
    """Emissions calculation event."""
    event_type: EventType = field(default=EventType.EMISSIONS, init=False)

    # Stack emissions (lb/hr)
    nox_lb_hr: float = 0.0
    co_lb_hr: float = 0.0
    co2_ton_hr: float = 0.0
    so2_lb_hr: float = 0.0
    pm_lb_hr: float = 0.0
    voc_lb_hr: float = 0.0

    # Concentrations
    nox_ppm: float = 0.0
    co_ppm: float = 0.0
    nox_ppm_corrected: float = 0.0  # @ 3% O2

    # GHG
    ghg_mtco2e_hr: float = 0.0

    # Calculation metadata
    emission_factors_source: str = "EPA"
    calculation_hash: str = ""

    # Operating conditions
    fuel_type: str = "natural_gas"
    fuel_flow_scfh: float = 0.0
    o2_percent: float = 0.0


@dataclass
class AlarmEvent(BaseEvent):
    """Alarm and notification event."""
    event_type: EventType = field(default=EventType.ALARM, init=False)

    # Alarm details
    alarm_tag: str = ""
    alarm_type: str = ""  # high, low, deviation, rate
    severity: Severity = Severity.WARNING
    message: str = ""

    # Values
    current_value: float = 0.0
    limit_value: float = 0.0
    unit: str = ""

    # State
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    # Duration
    active_since: Optional[datetime] = None
    cleared_at: Optional[datetime] = None


@dataclass
class SetpointChangeEvent(BaseEvent):
    """Setpoint change event."""
    event_type: EventType = field(default=EventType.SETPOINT_CHANGE, init=False)

    # Change details
    setpoint_tag: str = ""
    previous_value: float = 0.0
    new_value: float = 0.0
    unit: str = ""

    # Source
    change_source: str = ""  # manual, optimization, cascade
    operator: Optional[str] = None
    reason: Optional[str] = None

    # Approval
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


@dataclass
class StateChangeEvent(BaseEvent):
    """Boiler state change event."""
    event_type: EventType = field(default=EventType.STATE_CHANGE, init=False)

    # State transition
    previous_state: str = ""
    new_state: str = ""
    transition_reason: str = ""

    # Operator
    initiated_by: Optional[str] = None
    is_automatic: bool = False


# Avro schema definitions for Kafka Schema Registry
AVRO_SCHEMAS = {
    "process_data": {
        "type": "record",
        "name": "ProcessDataEvent",
        "namespace": "com.greenlang.flameguard",
        "fields": [
            {"name": "event_id", "type": "string"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "boiler_id", "type": "string"},
            {"name": "source", "type": "string"},
            {"name": "version", "type": "string"},
            {"name": "drum_pressure_psig", "type": "double"},
            {"name": "drum_level_inches", "type": "double"},
            {"name": "steam_flow_klb_hr", "type": "double"},
            {"name": "steam_temperature_f", "type": "double"},
            {"name": "flue_gas_temperature_f", "type": "double"},
            {"name": "o2_percent", "type": "double"},
            {"name": "co_ppm", "type": "double"},
            {"name": "fuel_flow_scfh", "type": "double"},
            {"name": "load_percent", "type": "double"},
            {"name": "firing", "type": "boolean"},
            {"name": "data_quality", "type": "string"},
        ],
    },
    "optimization": {
        "type": "record",
        "name": "OptimizationEvent",
        "namespace": "com.greenlang.flameguard",
        "fields": [
            {"name": "event_id", "type": "string"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "boiler_id", "type": "string"},
            {"name": "optimization_mode", "type": "string"},
            {"name": "current_efficiency", "type": "double"},
            {"name": "predicted_efficiency", "type": "double"},
            {"name": "efficiency_improvement", "type": "double"},
            {"name": "recommended_o2_setpoint", "type": "double"},
            {"name": "applied", "type": "boolean"},
            {"name": "calculation_hash", "type": "string"},
        ],
    },
    "safety": {
        "type": "record",
        "name": "SafetyEvent",
        "namespace": "com.greenlang.flameguard",
        "fields": [
            {"name": "event_id", "type": "string"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "boiler_id", "type": "string"},
            {"name": "safety_event_type", "type": "string"},
            {"name": "severity", "type": "string"},
            {"name": "interlock_tag", "type": ["null", "string"]},
            {"name": "previous_state", "type": "string"},
            {"name": "new_state", "type": "string"},
            {"name": "flame_proven", "type": "boolean"},
            {"name": "bms_state", "type": "string"},
        ],
    },
    "efficiency": {
        "type": "record",
        "name": "EfficiencyEvent",
        "namespace": "com.greenlang.flameguard",
        "fields": [
            {"name": "event_id", "type": "string"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "boiler_id", "type": "string"},
            {"name": "gross_efficiency_percent", "type": "double"},
            {"name": "net_efficiency_percent", "type": "double"},
            {"name": "stack_loss_percent", "type": "double"},
            {"name": "radiation_loss_percent", "type": "double"},
            {"name": "calculation_method", "type": "string"},
            {"name": "standard", "type": "string"},
            {"name": "calculation_hash", "type": "string"},
        ],
    },
    "emissions": {
        "type": "record",
        "name": "EmissionsEvent",
        "namespace": "com.greenlang.flameguard",
        "fields": [
            {"name": "event_id", "type": "string"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "boiler_id", "type": "string"},
            {"name": "nox_lb_hr", "type": "double"},
            {"name": "co_lb_hr", "type": "double"},
            {"name": "co2_ton_hr", "type": "double"},
            {"name": "ghg_mtco2e_hr", "type": "double"},
            {"name": "emission_factors_source", "type": "string"},
            {"name": "calculation_hash", "type": "string"},
        ],
    },
    "alarm": {
        "type": "record",
        "name": "AlarmEvent",
        "namespace": "com.greenlang.flameguard",
        "fields": [
            {"name": "event_id", "type": "string"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "boiler_id", "type": "string"},
            {"name": "alarm_tag", "type": "string"},
            {"name": "alarm_type", "type": "string"},
            {"name": "severity", "type": "string"},
            {"name": "message", "type": "string"},
            {"name": "current_value", "type": "double"},
            {"name": "limit_value", "type": "double"},
            {"name": "unit", "type": "string"},
            {"name": "acknowledged", "type": "boolean"},
        ],
    },
}


def get_avro_schema(event_type: str) -> Dict:
    """Get Avro schema for event type."""
    return AVRO_SCHEMAS.get(event_type, {})
