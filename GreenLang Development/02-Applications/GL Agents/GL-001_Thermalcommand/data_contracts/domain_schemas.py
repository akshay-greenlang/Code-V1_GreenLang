"""
GL-001 ThermalCommand: Canonical Domain Schemas

This module defines the canonical data schemas for all data domains in the
ThermalCommand ProcessHeatOrchestrator system. These schemas provide strict
type validation, unit governance, and data quality enforcement.

Domain Coverage:
- ProcessSensorData: Header pressure/temp, flow, valve positions
- EnergyConsumptionData: Fuel flow, electricity, steam production
- SafetySystemStatus: SIS permissives, trip statuses, bypasses
- ProductionSchedule: Campaigns, unit targets, batch plans
- WeatherForecast: Temperature, humidity, wind, forecast uncertainty
- EnergyPrices: Day-ahead/real-time, fuel prices, tariffs
- EquipmentHealth: Vibration, lube oil, fouling, RUL
- AlarmState: Severity, shelving, acknowledgement

Compliance:
- ISO 50001 Energy Management
- IEC 62443 Industrial Cybersecurity
- ISA-95 Enterprise-Control Integration
- ASME PTC 4.1 Steam Generators

Author: GreenLang Data Integration Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


# =============================================================================
# Enumerations
# =============================================================================

class DataQualityLevel(str, Enum):
    """Data quality classification levels."""
    GOOD = "good"           # >95% complete, validated
    FAIR = "fair"           # 80-95% complete, some gaps
    POOR = "poor"           # <80% complete, significant gaps
    UNKNOWN = "unknown"     # Quality not assessed


class UnitSystem(str, Enum):
    """Unit system for measurements."""
    SI = "SI"               # International System
    IMPERIAL = "imperial"   # US/Imperial units
    CUSTOM = "custom"       # Plant-specific units


class AlarmSeverity(IntEnum):
    """Alarm severity levels per ISA-18.2."""
    DIAGNOSTIC = 0          # Informational only
    LOW = 1                 # Low priority advisory
    MEDIUM = 2              # Attention required
    HIGH = 3                # Immediate attention
    CRITICAL = 4            # Safety critical


class EquipmentStatus(str, Enum):
    """Equipment operational status."""
    RUNNING = "running"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"
    FAULTED = "faulted"
    SHUTDOWN = "shutdown"


class TripStatus(str, Enum):
    """Safety trip status."""
    NORMAL = "normal"
    PRE_ALARM = "pre_alarm"
    ALARM = "alarm"
    TRIPPED = "tripped"
    BYPASSED = "bypassed"


class ForecastConfidence(str, Enum):
    """Forecast confidence levels."""
    HIGH = "high"           # >90% confidence
    MEDIUM = "medium"       # 70-90% confidence
    LOW = "low"             # 50-70% confidence
    VERY_LOW = "very_low"   # <50% confidence


class PriceMarket(str, Enum):
    """Energy price market types."""
    DAY_AHEAD = "day_ahead"
    REAL_TIME = "real_time"
    BALANCING = "balancing"
    ANCILLARY = "ancillary"


class FuelType(str, Enum):
    """Fuel types for energy consumption."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    COAL = "coal"
    BIOMASS = "biomass"
    HYDROGEN = "hydrogen"
    ELECTRICITY = "electricity"


# =============================================================================
# Base Schema Models
# =============================================================================

class ProvenanceInfo(BaseModel):
    """
    Provenance tracking for data lineage and audit compliance.

    All data records include provenance to enable:
    - Full audit trail for regulatory compliance
    - Data lineage tracking for debugging
    - Reproducibility verification
    """
    model_config = ConfigDict(frozen=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this record"
    )
    source_system: str = Field(
        ...,
        description="Originating system (e.g., 'SCADA', 'ERP', 'CMMS')"
    )
    source_tag: Optional[str] = Field(
        default=None,
        description="Original tag/field name in source system"
    )
    timestamp_collected: datetime = Field(
        ...,
        description="UTC timestamp when data was collected"
    )
    timestamp_processed: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when data was processed"
    )
    data_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the record data for integrity verification"
    )
    parent_record_id: Optional[str] = Field(
        default=None,
        description="ID of parent record if derived/transformed"
    )
    transformation_chain: List[str] = Field(
        default_factory=list,
        description="List of transformations applied to this data"
    )

    def compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of data for integrity verification."""
        import json
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()


class DataQualityMetrics(BaseModel):
    """
    Data quality metrics for each record or batch.

    Quality scoring components:
    - Completeness: % of required fields populated
    - Validity: % passing schema/range validation
    - Timeliness: Data freshness score
    - Consistency: Cross-field validation score
    """
    model_config = ConfigDict(frozen=True)

    quality_level: DataQualityLevel = Field(
        default=DataQualityLevel.UNKNOWN,
        description="Overall quality classification"
    )
    completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Completeness score (0-1)"
    )
    validity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Validity score (0-1)"
    )
    timeliness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Timeliness score (0-1)"
    )
    consistency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Consistency score (0-1)"
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall quality score (0-100)"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="List of validation errors"
    )
    validation_warnings: List[str] = Field(
        default_factory=list,
        description="List of validation warnings"
    )


class BaseDataContract(BaseModel):
    """
    Base class for all data contracts providing common fields.

    All domain schemas inherit from this to ensure:
    - Consistent timestamp handling (UTC)
    - Provenance tracking
    - Data quality metrics
    - Version information
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",  # Reject unknown fields for strict contracts
    )

    schema_version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Schema version for compatibility checking"
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of the data record"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        default=None,
        description="Data provenance and lineage information"
    )
    quality: Optional[DataQualityMetrics] = Field(
        default=None,
        description="Data quality metrics"
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc_timestamp(cls, v: Any) -> datetime:
        """Ensure timestamp is in UTC."""
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime) and v.tzinfo is None:
            # Assume UTC if no timezone
            from datetime import timezone
            v = v.replace(tzinfo=timezone.utc)
        return v


# =============================================================================
# Process Sensor Data Schema
# =============================================================================

class SteamHeaderData(BaseModel):
    """Steam header pressure, temperature, and flow data."""
    model_config = ConfigDict(frozen=True)

    header_id: str = Field(
        ...,
        pattern=r"^header[A-Z]$",
        description="Header identifier (e.g., 'headerA', 'headerB')"
    )
    pressure_barg: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description="Header pressure in bar(g)"
    )
    temperature_c: float = Field(
        ...,
        ge=0.0,
        le=600.0,
        description="Header temperature in Celsius"
    )
    flow_total_tph: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Total flow rate in tonnes per hour"
    )
    flow_setpoint_tph: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1000.0,
        description="Flow setpoint in tonnes per hour"
    )
    superheat_c: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=200.0,
        description="Superheat temperature in Celsius"
    )

    @model_validator(mode="after")
    def validate_steam_conditions(self) -> "SteamHeaderData":
        """Validate steam conditions are physically reasonable."""
        # Approximate saturation temperature check
        # Simplified: T_sat (C) ~ 100 + 20*ln(P+1) for rough validation
        import math
        approx_sat_temp = 100 + 20 * math.log(self.pressure_barg + 1)
        if self.temperature_c < approx_sat_temp * 0.9:
            raise ValueError(
                f"Temperature {self.temperature_c}C appears too low for "
                f"pressure {self.pressure_barg} bar(g)"
            )
        return self


class ValvePosition(BaseModel):
    """Control valve position data."""
    model_config = ConfigDict(frozen=True)

    valve_id: str = Field(
        ...,
        description="Unique valve identifier"
    )
    position_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Valve position in percent open"
    )
    setpoint_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Valve setpoint in percent"
    )
    mode: Literal["auto", "manual", "cascade", "remote"] = Field(
        default="auto",
        description="Valve control mode"
    )
    travel_limit_lo: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Low travel limit percent"
    )
    travel_limit_hi: Optional[float] = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="High travel limit percent"
    )


class ProcessSensorData(BaseDataContract):
    """
    Process sensor data from SCADA/DCS systems.

    Contains real-time measurements for:
    - Steam header conditions (pressure, temperature, flow)
    - Control valve positions
    - Process temperatures
    - Flow measurements

    Data Quality Requirements:
    - Update frequency: 1 second (typical), 100ms (fast scan)
    - Latency: <500ms from sensor to schema
    - Completeness: >99% for safety-critical tags
    """

    facility_id: str = Field(
        ...,
        description="Facility identifier"
    )
    area_id: str = Field(
        ...,
        description="Plant area identifier"
    )

    # Steam header data
    steam_headers: Dict[str, SteamHeaderData] = Field(
        default_factory=dict,
        description="Steam header data keyed by header ID"
    )

    # Valve positions
    valve_positions: Dict[str, ValvePosition] = Field(
        default_factory=dict,
        description="Valve position data keyed by valve ID"
    )

    # Process temperatures
    temperatures: Dict[str, float] = Field(
        default_factory=dict,
        description="Process temperatures in Celsius keyed by tag"
    )

    # Flow measurements
    flows: Dict[str, float] = Field(
        default_factory=dict,
        description="Flow rates keyed by tag (units in tag dictionary)"
    )

    # Pressure measurements
    pressures: Dict[str, float] = Field(
        default_factory=dict,
        description="Pressure readings in bar(g) keyed by tag"
    )

    # Level measurements
    levels: Dict[str, float] = Field(
        default_factory=dict,
        description="Level readings in percent keyed by tag"
    )

    # Raw SCADA tags (for unmapped data)
    raw_tags: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw SCADA tag values for unmapped data"
    )

    scan_rate_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="SCADA scan rate in milliseconds"
    )


# =============================================================================
# Energy Consumption Data Schema
# =============================================================================

class FuelConsumption(BaseModel):
    """Fuel consumption record for a single fuel type."""
    model_config = ConfigDict(frozen=True)

    fuel_type: FuelType = Field(
        ...,
        description="Type of fuel"
    )
    flow_rate: float = Field(
        ...,
        ge=0.0,
        description="Fuel flow rate"
    )
    flow_unit: str = Field(
        ...,
        pattern=r"^(kg/h|Nm3/h|t/h|MW)$",
        description="Flow rate unit"
    )
    heating_value_mj_kg: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=60.0,
        description="Lower heating value in MJ/kg"
    )
    energy_rate_mw: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Energy input rate in MW (thermal)"
    )
    co2_factor_kg_mwh: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="CO2 emission factor in kg/MWh"
    )


class BoilerPerformance(BaseModel):
    """Boiler performance and efficiency data."""
    model_config = ConfigDict(frozen=True)

    boiler_id: str = Field(
        ...,
        pattern=r"^B\d+$",
        description="Boiler identifier (e.g., 'B1', 'B2')"
    )
    fuel_flow_kgh: float = Field(
        ...,
        ge=0.0,
        description="Fuel flow rate in kg/h or Nm3/h"
    )
    steam_output_tph: float = Field(
        ...,
        ge=0.0,
        description="Steam output in tonnes per hour"
    )
    max_rate_tph: float = Field(
        ...,
        ge=0.0,
        description="Maximum steam capacity in t/h"
    )
    efficiency_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Boiler efficiency percentage"
    )
    turndown_ratio: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=20.0,
        description="Turndown ratio (max/min load)"
    )
    load_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=110.0,
        description="Current load as percentage of max"
    )
    flue_gas_temp_c: Optional[float] = Field(
        default=None,
        description="Flue gas exit temperature in Celsius"
    )
    excess_air_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=200.0,
        description="Excess air percentage"
    )


class EnergyConsumptionData(BaseDataContract):
    """
    Energy consumption and production data.

    Contains measurements for:
    - Fuel consumption (gas, oil, coal, biomass, hydrogen)
    - Electricity import/export
    - Steam production and consumption
    - Boiler performance metrics

    Standards Compliance:
    - ISO 50001 Energy Management
    - ASME PTC 4.1 for boiler efficiency
    """

    facility_id: str = Field(
        ...,
        description="Facility identifier"
    )
    period_start: datetime = Field(
        ...,
        description="Start of measurement period"
    )
    period_end: datetime = Field(
        ...,
        description="End of measurement period"
    )

    # Fuel consumption by type
    fuel_consumption: List[FuelConsumption] = Field(
        default_factory=list,
        description="Fuel consumption by type"
    )

    # Boiler performance
    boiler_performance: Dict[str, BoilerPerformance] = Field(
        default_factory=dict,
        description="Boiler performance data keyed by boiler ID"
    )

    # Electricity
    electricity_import_mwh: float = Field(
        default=0.0,
        ge=0.0,
        description="Electricity imported in MWh"
    )
    electricity_export_mwh: float = Field(
        default=0.0,
        ge=0.0,
        description="Electricity exported in MWh"
    )
    electricity_net_mwh: Optional[float] = Field(
        default=None,
        description="Net electricity (import - export) in MWh"
    )

    # Steam
    steam_production_tonnes: float = Field(
        default=0.0,
        ge=0.0,
        description="Total steam production in tonnes"
    )
    steam_import_tonnes: float = Field(
        default=0.0,
        ge=0.0,
        description="Steam imported in tonnes"
    )
    steam_export_tonnes: float = Field(
        default=0.0,
        ge=0.0,
        description="Steam exported in tonnes"
    )

    # Heat demand by unit
    unit_heat_demands_mwth: Dict[str, float] = Field(
        default_factory=dict,
        description="Heat demand per unit in MWth"
    )

    # Aggregated metrics
    total_fuel_energy_mwh: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total fuel energy input in MWh"
    )
    total_useful_heat_mwh: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total useful heat output in MWh"
    )
    overall_efficiency_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Overall thermal efficiency percentage"
    )

    @model_validator(mode="after")
    def calculate_net_electricity(self) -> "EnergyConsumptionData":
        """Calculate net electricity if not provided."""
        if self.electricity_net_mwh is None:
            object.__setattr__(
                self,
                "electricity_net_mwh",
                self.electricity_import_mwh - self.electricity_export_mwh
            )
        return self


# =============================================================================
# Safety System Status Schema
# =============================================================================

class SISPermissive(BaseModel):
    """Safety Instrumented System permissive status."""
    model_config = ConfigDict(frozen=True)

    permissive_id: str = Field(
        ...,
        description="Permissive identifier"
    )
    description: str = Field(
        ...,
        description="Permissive description"
    )
    is_enabled: bool = Field(
        ...,
        description="True if permissive is enabled"
    )
    required_for: List[str] = Field(
        default_factory=list,
        description="Operations requiring this permissive"
    )
    last_change: datetime = Field(
        ...,
        description="Last state change timestamp"
    )
    override_active: bool = Field(
        default=False,
        description="True if override is active"
    )
    override_expires: Optional[datetime] = Field(
        default=None,
        description="Override expiration timestamp"
    )
    override_authorization: Optional[str] = Field(
        default=None,
        description="Authorization ID for override"
    )


class TripPoint(BaseModel):
    """Safety trip point status."""
    model_config = ConfigDict(frozen=True)

    trip_id: str = Field(
        ...,
        description="Trip point identifier"
    )
    description: str = Field(
        ...,
        description="Trip point description"
    )
    status: TripStatus = Field(
        ...,
        description="Current trip status"
    )
    setpoint: float = Field(
        ...,
        description="Trip setpoint value"
    )
    setpoint_unit: str = Field(
        ...,
        description="Setpoint engineering unit"
    )
    current_value: float = Field(
        ...,
        description="Current process value"
    )
    deviation_pct: Optional[float] = Field(
        default=None,
        description="Deviation from setpoint percentage"
    )
    trip_count_24h: int = Field(
        default=0,
        ge=0,
        description="Number of trips in last 24 hours"
    )


class BypassRecord(BaseModel):
    """Safety bypass record."""
    model_config = ConfigDict(frozen=True)

    bypass_id: str = Field(
        ...,
        description="Bypass record identifier"
    )
    element_id: str = Field(
        ...,
        description="Bypassed element identifier"
    )
    element_type: Literal["trip", "interlock", "alarm", "permissive"] = Field(
        ...,
        description="Type of bypassed element"
    )
    reason: str = Field(
        ...,
        min_length=10,
        description="Reason for bypass (minimum 10 characters)"
    )
    authorized_by: str = Field(
        ...,
        description="Person authorizing bypass"
    )
    start_time: datetime = Field(
        ...,
        description="Bypass start time"
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Bypass end time (None if active)"
    )
    max_duration_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Maximum bypass duration in hours"
    )
    compensatory_measures: List[str] = Field(
        default_factory=list,
        description="Compensatory measures in place"
    )


class SafetySystemStatus(BaseDataContract):
    """
    Safety Instrumented System (SIS) status data.

    Contains:
    - SIS permissive states
    - Trip point statuses
    - Active bypasses
    - Safety integrity level (SIL) status

    Standards Compliance:
    - IEC 61511 Functional Safety
    - ISA 84 Safety Instrumented Systems
    - OSHA PSM Requirements
    """

    facility_id: str = Field(
        ...,
        description="Facility identifier"
    )
    area_id: str = Field(
        ...,
        description="Plant area identifier"
    )

    # Permissives
    permissives: Dict[str, SISPermissive] = Field(
        default_factory=dict,
        description="SIS permissives keyed by ID"
    )
    dispatch_enabled: bool = Field(
        default=False,
        description="Master dispatch permissive status"
    )

    # Trip points
    trip_points: Dict[str, TripPoint] = Field(
        default_factory=dict,
        description="Trip points keyed by ID"
    )
    trips_in_alarm: int = Field(
        default=0,
        ge=0,
        description="Count of trip points in alarm"
    )

    # Bypasses
    active_bypasses: List[BypassRecord] = Field(
        default_factory=list,
        description="Currently active bypasses"
    )
    bypass_count: int = Field(
        default=0,
        ge=0,
        description="Number of active bypasses"
    )

    # Safety summary
    sil_status: Literal["operational", "degraded", "failed"] = Field(
        default="operational",
        description="Overall SIL status"
    )
    proof_test_due: Optional[datetime] = Field(
        default=None,
        description="Next proof test due date"
    )
    last_trip_event: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last trip event"
    )

    @model_validator(mode="after")
    def update_bypass_count(self) -> "SafetySystemStatus":
        """Ensure bypass count matches active bypasses."""
        actual_count = len(self.active_bypasses)
        if self.bypass_count != actual_count:
            object.__setattr__(self, "bypass_count", actual_count)
        return self


# =============================================================================
# Production Schedule Schema
# =============================================================================

class BatchPlan(BaseModel):
    """Individual batch production plan."""
    model_config = ConfigDict(frozen=True)

    batch_id: str = Field(
        ...,
        description="Unique batch identifier"
    )
    product_code: str = Field(
        ...,
        description="Product code/SKU"
    )
    product_name: str = Field(
        ...,
        description="Product name"
    )
    quantity: float = Field(
        ...,
        gt=0,
        description="Planned quantity"
    )
    quantity_unit: str = Field(
        ...,
        description="Quantity unit (kg, tonnes, units)"
    )
    scheduled_start: datetime = Field(
        ...,
        description="Planned start time"
    )
    scheduled_end: datetime = Field(
        ...,
        description="Planned end time"
    )
    heat_demand_mwth: float = Field(
        ...,
        ge=0.0,
        description="Heat demand for batch in MWth"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Batch priority (1=highest, 10=lowest)"
    )
    unit_id: str = Field(
        ...,
        description="Production unit ID"
    )
    recipe_id: Optional[str] = Field(
        default=None,
        description="Recipe/formulation ID"
    )

    @model_validator(mode="after")
    def validate_schedule(self) -> "BatchPlan":
        """Validate end is after start."""
        if self.scheduled_end <= self.scheduled_start:
            raise ValueError("scheduled_end must be after scheduled_start")
        return self


class UnitTarget(BaseModel):
    """Production unit targets."""
    model_config = ConfigDict(frozen=True)

    unit_id: str = Field(
        ...,
        description="Production unit identifier"
    )
    unit_name: str = Field(
        ...,
        description="Production unit name"
    )
    target_output: float = Field(
        ...,
        ge=0.0,
        description="Target output rate"
    )
    target_unit: str = Field(
        ...,
        description="Output unit (tonnes/h, units/h)"
    )
    heat_demand_mwth: float = Field(
        ...,
        ge=0.0,
        description="Heat demand in MWth"
    )
    steam_demand_tph: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Steam demand in tonnes per hour"
    )
    min_capacity_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Minimum capacity percentage"
    )
    max_capacity_pct: float = Field(
        default=100.0,
        ge=0.0,
        le=110.0,
        description="Maximum capacity percentage"
    )


class Campaign(BaseModel):
    """Production campaign definition."""
    model_config = ConfigDict(frozen=True)

    campaign_id: str = Field(
        ...,
        description="Campaign identifier"
    )
    campaign_name: str = Field(
        ...,
        description="Campaign name"
    )
    product_family: str = Field(
        ...,
        description="Product family/group"
    )
    start_date: datetime = Field(
        ...,
        description="Campaign start date"
    )
    end_date: datetime = Field(
        ...,
        description="Campaign end date"
    )
    total_heat_budget_mwh: float = Field(
        ...,
        ge=0.0,
        description="Total heat budget in MWh"
    )
    batches: List[str] = Field(
        default_factory=list,
        description="List of batch IDs in campaign"
    )


class ProductionSchedule(BaseDataContract):
    """
    Production schedule and demand data.

    Contains:
    - Campaign definitions
    - Unit production targets
    - Batch plans
    - Heat demand forecasts

    Standards Compliance:
    - ISA-95 Batch Control
    - ISA-88 Enterprise-Control Integration
    """

    facility_id: str = Field(
        ...,
        description="Facility identifier"
    )
    schedule_horizon_start: datetime = Field(
        ...,
        description="Schedule horizon start"
    )
    schedule_horizon_end: datetime = Field(
        ...,
        description="Schedule horizon end"
    )

    # Campaigns
    campaigns: List[Campaign] = Field(
        default_factory=list,
        description="Active and planned campaigns"
    )

    # Unit targets
    unit_targets: Dict[str, UnitTarget] = Field(
        default_factory=dict,
        description="Unit targets keyed by unit ID"
    )

    # Batch plans
    batch_plans: List[BatchPlan] = Field(
        default_factory=list,
        description="Scheduled batch plans"
    )

    # Aggregated demands
    total_heat_demand_mwth: float = Field(
        default=0.0,
        ge=0.0,
        description="Total instantaneous heat demand in MWth"
    )
    total_steam_demand_tph: float = Field(
        default=0.0,
        ge=0.0,
        description="Total steam demand in tonnes per hour"
    )

    # Schedule metadata
    schedule_version: str = Field(
        default="1",
        description="Schedule version identifier"
    )
    last_updated_by: Optional[str] = Field(
        default=None,
        description="User who last updated schedule"
    )
    approved_by: Optional[str] = Field(
        default=None,
        description="Approver for schedule"
    )


# =============================================================================
# Weather Forecast Schema
# =============================================================================

class HourlyForecast(BaseModel):
    """Hourly weather forecast data point."""
    model_config = ConfigDict(frozen=True)

    forecast_time: datetime = Field(
        ...,
        description="Forecast valid time"
    )
    temperature_c: float = Field(
        ...,
        ge=-60.0,
        le=60.0,
        description="Temperature in Celsius"
    )
    humidity_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative humidity percentage"
    )
    wind_speed_ms: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Wind speed in m/s"
    )
    wind_direction_deg: float = Field(
        ...,
        ge=0.0,
        le=360.0,
        description="Wind direction in degrees"
    )
    precipitation_mm: float = Field(
        default=0.0,
        ge=0.0,
        description="Precipitation in mm"
    )
    cloud_cover_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Cloud cover percentage"
    )
    solar_radiation_wm2: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1400.0,
        description="Solar radiation in W/m2"
    )
    atmospheric_pressure_hpa: Optional[float] = Field(
        default=None,
        ge=800.0,
        le=1100.0,
        description="Atmospheric pressure in hPa"
    )


class ForecastUncertainty(BaseModel):
    """Forecast uncertainty quantification."""
    model_config = ConfigDict(frozen=True)

    parameter: str = Field(
        ...,
        description="Parameter name (e.g., 'temperature_c')"
    )
    confidence: ForecastConfidence = Field(
        ...,
        description="Confidence level"
    )
    lower_bound: float = Field(
        ...,
        description="Lower bound (10th percentile)"
    )
    upper_bound: float = Field(
        ...,
        description="Upper bound (90th percentile)"
    )
    std_dev: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Standard deviation"
    )


class WeatherForecast(BaseDataContract):
    """
    Weather forecast data for operational planning.

    Contains:
    - Hourly forecasts (temperature, humidity, wind)
    - Forecast uncertainty quantification
    - Impact assessment for operations

    Data Sources:
    - National Weather Service
    - Commercial weather providers
    - On-site weather stations
    """

    facility_id: str = Field(
        ...,
        description="Facility identifier"
    )
    location_lat: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Facility latitude"
    )
    location_lon: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Facility longitude"
    )

    # Forecast data
    forecast_issued: datetime = Field(
        ...,
        description="Forecast issuance time"
    )
    forecast_provider: str = Field(
        ...,
        description="Weather data provider"
    )
    hourly_forecasts: List[HourlyForecast] = Field(
        default_factory=list,
        description="Hourly forecast data"
    )

    # Current conditions
    current_temperature_c: float = Field(
        ...,
        ge=-60.0,
        le=60.0,
        description="Current temperature in Celsius"
    )
    current_humidity_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Current humidity percentage"
    )

    # Uncertainty
    uncertainties: List[ForecastUncertainty] = Field(
        default_factory=list,
        description="Uncertainty quantification for key parameters"
    )

    # Operational impact
    heating_degree_days: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Heating degree days"
    )
    cooling_degree_days: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Cooling degree days"
    )
    ambient_impact_factor: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=1.5,
        description="Ambient temperature efficiency impact factor"
    )


# =============================================================================
# Energy Prices Schema
# =============================================================================

class ElectricityPrice(BaseModel):
    """Electricity price record."""
    model_config = ConfigDict(frozen=True)

    price_time: datetime = Field(
        ...,
        description="Price valid time"
    )
    market: PriceMarket = Field(
        ...,
        description="Market type"
    )
    price_usd_mwh: float = Field(
        ...,
        description="Price in USD/MWh"
    )
    currency: str = Field(
        default="USD",
        description="Currency code"
    )
    zone: Optional[str] = Field(
        default=None,
        description="Pricing zone/node"
    )
    is_forecast: bool = Field(
        default=False,
        description="True if forecasted price"
    )
    confidence_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Forecast confidence percentage"
    )


class FuelPrice(BaseModel):
    """Fuel price record."""
    model_config = ConfigDict(frozen=True)

    fuel_type: FuelType = Field(
        ...,
        description="Fuel type"
    )
    price: float = Field(
        ...,
        ge=0.0,
        description="Price value"
    )
    price_unit: str = Field(
        ...,
        pattern=r"^\$/(MMBtu|therm|MWh|kg|tonne|Nm3|bbl)$",
        description="Price unit"
    )
    effective_date: datetime = Field(
        ...,
        description="Price effective date"
    )
    expiry_date: Optional[datetime] = Field(
        default=None,
        description="Price expiry date"
    )
    contract_id: Optional[str] = Field(
        default=None,
        description="Fuel contract identifier"
    )


class TariffPeriod(BaseModel):
    """Time-of-use tariff period."""
    model_config = ConfigDict(frozen=True)

    period_name: str = Field(
        ...,
        description="Period name (peak, off-peak, shoulder)"
    )
    start_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Start hour (0-23)"
    )
    end_hour: int = Field(
        ...,
        ge=0,
        le=24,
        description="End hour (1-24)"
    )
    rate_usd_kwh: float = Field(
        ...,
        ge=0.0,
        description="Energy rate in USD/kWh"
    )
    demand_charge_usd_kw: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Demand charge in USD/kW"
    )
    days: List[Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]] = Field(
        default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"],
        description="Applicable days"
    )


class EnergyPrices(BaseDataContract):
    """
    Energy price data for optimization.

    Contains:
    - Day-ahead electricity prices
    - Real-time electricity prices
    - Fuel prices by type
    - Tariff structures

    Data Sources:
    - ISO/RTO market data
    - Utility tariffs
    - Fuel contract prices
    """

    facility_id: str = Field(
        ...,
        description="Facility identifier"
    )
    pricing_region: str = Field(
        ...,
        description="Pricing region/ISO"
    )

    # Electricity prices
    day_ahead_prices: List[ElectricityPrice] = Field(
        default_factory=list,
        description="Day-ahead electricity prices"
    )
    real_time_prices: List[ElectricityPrice] = Field(
        default_factory=list,
        description="Real-time electricity prices"
    )
    current_rt_price_usd_mwh: float = Field(
        ...,
        ge=-500.0,  # Can be negative in oversupply
        le=10000.0,
        description="Current real-time price in USD/MWh"
    )

    # Fuel prices
    fuel_prices: List[FuelPrice] = Field(
        default_factory=list,
        description="Fuel prices by type"
    )

    # Tariffs
    tariff_periods: List[TariffPeriod] = Field(
        default_factory=list,
        description="Time-of-use tariff periods"
    )

    # Carbon pricing
    carbon_price_usd_tonne: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Carbon price in USD/tonne CO2"
    )

    # REC/renewable pricing
    rec_price_usd_mwh: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Renewable Energy Certificate price USD/MWh"
    )


# =============================================================================
# Equipment Health Schema
# =============================================================================

class VibrationData(BaseModel):
    """Equipment vibration monitoring data."""
    model_config = ConfigDict(frozen=True)

    measurement_point: str = Field(
        ...,
        description="Vibration measurement point ID"
    )
    velocity_mm_s: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Velocity in mm/s RMS"
    )
    displacement_um: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Displacement in micrometers peak-to-peak"
    )
    acceleration_g: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Acceleration in g RMS"
    )
    temperature_c: Optional[float] = Field(
        default=None,
        description="Bearing temperature in Celsius"
    )
    alarm_level: AlarmSeverity = Field(
        default=AlarmSeverity.DIAGNOSTIC,
        description="Current alarm level"
    )
    trend: Literal["stable", "increasing", "decreasing"] = Field(
        default="stable",
        description="Vibration trend"
    )


class LubeOilAnalysis(BaseModel):
    """Lubricating oil analysis results."""
    model_config = ConfigDict(frozen=True)

    sample_date: datetime = Field(
        ...,
        description="Sample collection date"
    )
    oil_type: str = Field(
        ...,
        description="Oil type/grade"
    )
    viscosity_cst: float = Field(
        ...,
        ge=0.0,
        description="Kinematic viscosity at 40C in cSt"
    )
    water_ppm: float = Field(
        default=0.0,
        ge=0.0,
        description="Water content in ppm"
    )
    particle_count_4um: Optional[int] = Field(
        default=None,
        ge=0,
        description="Particle count >4um per mL"
    )
    particle_count_14um: Optional[int] = Field(
        default=None,
        ge=0,
        description="Particle count >14um per mL"
    )
    iron_ppm: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Iron content in ppm"
    )
    copper_ppm: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Copper content in ppm"
    )
    acid_number_mg_koh_g: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total acid number in mg KOH/g"
    )
    condition: Literal["good", "marginal", "critical"] = Field(
        default="good",
        description="Overall oil condition"
    )


class FoulingIndicator(BaseModel):
    """Heat exchanger fouling indicators."""
    model_config = ConfigDict(frozen=True)

    equipment_id: str = Field(
        ...,
        description="Heat exchanger identifier"
    )
    fouling_factor: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fouling factor (0=clean, 1=severely fouled)"
    )
    ua_value_kw_k: float = Field(
        ...,
        ge=0.0,
        description="Overall heat transfer coefficient * area in kW/K"
    )
    ua_clean_kw_k: float = Field(
        ...,
        ge=0.0,
        description="Clean UA value in kW/K"
    )
    delta_p_increase_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Pressure drop increase percentage"
    )
    effectiveness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current heat exchanger effectiveness"
    )
    cleaning_recommended: bool = Field(
        default=False,
        description="True if cleaning is recommended"
    )


class RemainingUsefulLife(BaseModel):
    """Remaining useful life prediction."""
    model_config = ConfigDict(frozen=True)

    equipment_id: str = Field(
        ...,
        description="Equipment identifier"
    )
    component: str = Field(
        ...,
        description="Component name"
    )
    rul_hours: float = Field(
        ...,
        ge=0.0,
        description="Remaining useful life in hours"
    )
    rul_confidence_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Confidence in RUL estimate"
    )
    failure_mode: str = Field(
        ...,
        description="Predicted failure mode"
    )
    recommended_action: str = Field(
        ...,
        description="Recommended maintenance action"
    )
    action_deadline: Optional[datetime] = Field(
        default=None,
        description="Deadline for recommended action"
    )


class EquipmentHealth(BaseDataContract):
    """
    Equipment health and condition monitoring data.

    Contains:
    - Vibration monitoring data
    - Lubricating oil analysis
    - Fouling indicators
    - Remaining useful life predictions

    Standards Compliance:
    - ISO 17359 Condition Monitoring
    - ISO 10816 Vibration Evaluation
    - IEEE 762 Equipment Reliability
    """

    facility_id: str = Field(
        ...,
        description="Facility identifier"
    )
    equipment_id: str = Field(
        ...,
        description="Equipment identifier"
    )
    equipment_type: str = Field(
        ...,
        description="Equipment type (boiler, pump, turbine, etc.)"
    )
    equipment_status: EquipmentStatus = Field(
        default=EquipmentStatus.RUNNING,
        description="Current equipment status"
    )

    # Overall health score
    health_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall health score (0-1)"
    )
    health_trend: Literal["improving", "stable", "degrading"] = Field(
        default="stable",
        description="Health trend direction"
    )

    # Vibration data
    vibration_data: Dict[str, VibrationData] = Field(
        default_factory=dict,
        description="Vibration data keyed by measurement point"
    )

    # Lube oil analysis
    lube_oil_analysis: Optional[LubeOilAnalysis] = Field(
        default=None,
        description="Latest lube oil analysis"
    )

    # Fouling indicators
    fouling_indicators: List[FoulingIndicator] = Field(
        default_factory=list,
        description="Heat exchanger fouling data"
    )

    # RUL predictions
    rul_predictions: List[RemainingUsefulLife] = Field(
        default_factory=list,
        description="Remaining useful life predictions"
    )

    # CMMS integration
    cmms_asset_id: Optional[str] = Field(
        default=None,
        description="CMMS asset identifier"
    )
    open_work_orders: int = Field(
        default=0,
        ge=0,
        description="Number of open work orders"
    )
    last_pm_date: Optional[datetime] = Field(
        default=None,
        description="Last preventive maintenance date"
    )
    next_pm_date: Optional[datetime] = Field(
        default=None,
        description="Next scheduled PM date"
    )


# =============================================================================
# Alarm State Schema
# =============================================================================

class AlarmRecord(BaseModel):
    """Individual alarm record."""
    model_config = ConfigDict(frozen=True)

    alarm_id: str = Field(
        ...,
        description="Unique alarm identifier"
    )
    tag: str = Field(
        ...,
        description="Associated process tag"
    )
    description: str = Field(
        ...,
        description="Alarm description"
    )
    severity: AlarmSeverity = Field(
        ...,
        description="Alarm severity"
    )
    state: Literal["active", "cleared", "acknowledged", "shelved"] = Field(
        ...,
        description="Current alarm state"
    )
    alarm_time: datetime = Field(
        ...,
        description="Alarm activation time"
    )
    clear_time: Optional[datetime] = Field(
        default=None,
        description="Alarm clear time"
    )
    ack_time: Optional[datetime] = Field(
        default=None,
        description="Acknowledgement time"
    )
    ack_by: Optional[str] = Field(
        default=None,
        description="User who acknowledged"
    )
    shelved_until: Optional[datetime] = Field(
        default=None,
        description="Shelving expiration time"
    )
    shelved_by: Optional[str] = Field(
        default=None,
        description="User who shelved alarm"
    )
    value_at_alarm: Optional[float] = Field(
        default=None,
        description="Process value at alarm time"
    )
    setpoint: Optional[float] = Field(
        default=None,
        description="Alarm setpoint"
    )
    area: str = Field(
        ...,
        description="Plant area"
    )
    equipment_id: Optional[str] = Field(
        default=None,
        description="Associated equipment"
    )
    consequent_actions: List[str] = Field(
        default_factory=list,
        description="Required consequent actions"
    )


class AlarmStatistics(BaseModel):
    """Alarm statistics for performance monitoring."""
    model_config = ConfigDict(frozen=True)

    period_start: datetime = Field(
        ...,
        description="Statistics period start"
    )
    period_end: datetime = Field(
        ...,
        description="Statistics period end"
    )
    total_alarms: int = Field(
        default=0,
        ge=0,
        description="Total alarms in period"
    )
    alarms_per_hour: float = Field(
        default=0.0,
        ge=0.0,
        description="Average alarms per hour"
    )
    mean_time_to_ack_minutes: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Mean time to acknowledge in minutes"
    )
    mean_time_to_clear_minutes: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Mean time to clear in minutes"
    )
    chattering_alarms: int = Field(
        default=0,
        ge=0,
        description="Number of chattering alarms"
    )
    nuisance_alarms: int = Field(
        default=0,
        ge=0,
        description="Number of nuisance alarms"
    )
    stale_alarms: int = Field(
        default=0,
        ge=0,
        description="Number of stale (standing) alarms"
    )


class AlarmState(BaseDataContract):
    """
    Alarm system state and statistics.

    Contains:
    - Active alarms with full details
    - Shelved alarms
    - Alarm statistics
    - ISA-18.2 compliance metrics

    Standards Compliance:
    - ISA-18.2 Alarm Management
    - EEMUA 191 Alarm Systems
    - IEC 62682 Management of Alarm Systems
    """

    facility_id: str = Field(
        ...,
        description="Facility identifier"
    )
    area_id: Optional[str] = Field(
        default=None,
        description="Plant area filter (None for facility-wide)"
    )

    # Active alarms
    active_alarms: List[AlarmRecord] = Field(
        default_factory=list,
        description="Currently active alarms"
    )
    active_count: int = Field(
        default=0,
        ge=0,
        description="Count of active alarms"
    )

    # Severity breakdown
    critical_count: int = Field(
        default=0,
        ge=0,
        description="Critical severity alarm count"
    )
    high_count: int = Field(
        default=0,
        ge=0,
        description="High severity alarm count"
    )
    medium_count: int = Field(
        default=0,
        ge=0,
        description="Medium severity alarm count"
    )
    low_count: int = Field(
        default=0,
        ge=0,
        description="Low severity alarm count"
    )

    # Shelved alarms
    shelved_alarms: List[AlarmRecord] = Field(
        default_factory=list,
        description="Currently shelved alarms"
    )
    shelved_count: int = Field(
        default=0,
        ge=0,
        description="Count of shelved alarms"
    )

    # Unacknowledged alarms
    unacknowledged_count: int = Field(
        default=0,
        ge=0,
        description="Count of unacknowledged alarms"
    )

    # Statistics
    statistics: Optional[AlarmStatistics] = Field(
        default=None,
        description="Alarm statistics for current period"
    )

    # ISA-18.2 KPIs
    alarm_rate_target: float = Field(
        default=6.0,
        ge=0.0,
        description="Target max alarms per operator per 10 minutes"
    )
    alarm_rate_actual: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Actual alarm rate per operator per 10 minutes"
    )
    is_alarm_flood: bool = Field(
        default=False,
        description="True if alarm flood detected"
    )

    @model_validator(mode="after")
    def update_counts(self) -> "AlarmState":
        """Update alarm counts from active alarms list."""
        if self.active_alarms:
            object.__setattr__(self, "active_count", len(self.active_alarms))
            object.__setattr__(
                self, "critical_count",
                sum(1 for a in self.active_alarms if a.severity == AlarmSeverity.CRITICAL)
            )
            object.__setattr__(
                self, "high_count",
                sum(1 for a in self.active_alarms if a.severity == AlarmSeverity.HIGH)
            )
            object.__setattr__(
                self, "medium_count",
                sum(1 for a in self.active_alarms if a.severity == AlarmSeverity.MEDIUM)
            )
            object.__setattr__(
                self, "low_count",
                sum(1 for a in self.active_alarms if a.severity == AlarmSeverity.LOW)
            )
            object.__setattr__(
                self, "unacknowledged_count",
                sum(1 for a in self.active_alarms if a.ack_time is None and a.state == "active")
            )
        return self


# =============================================================================
# Schema Export Registry
# =============================================================================

DOMAIN_SCHEMAS = {
    "ProcessSensorData": ProcessSensorData,
    "EnergyConsumptionData": EnergyConsumptionData,
    "SafetySystemStatus": SafetySystemStatus,
    "ProductionSchedule": ProductionSchedule,
    "WeatherForecast": WeatherForecast,
    "EnergyPrices": EnergyPrices,
    "EquipmentHealth": EquipmentHealth,
    "AlarmState": AlarmState,
}

SUB_SCHEMAS = {
    "SteamHeaderData": SteamHeaderData,
    "ValvePosition": ValvePosition,
    "FuelConsumption": FuelConsumption,
    "BoilerPerformance": BoilerPerformance,
    "SISPermissive": SISPermissive,
    "TripPoint": TripPoint,
    "BypassRecord": BypassRecord,
    "BatchPlan": BatchPlan,
    "UnitTarget": UnitTarget,
    "Campaign": Campaign,
    "HourlyForecast": HourlyForecast,
    "ForecastUncertainty": ForecastUncertainty,
    "ElectricityPrice": ElectricityPrice,
    "FuelPrice": FuelPrice,
    "TariffPeriod": TariffPeriod,
    "VibrationData": VibrationData,
    "LubeOilAnalysis": LubeOilAnalysis,
    "FoulingIndicator": FoulingIndicator,
    "RemainingUsefulLife": RemainingUsefulLife,
    "AlarmRecord": AlarmRecord,
    "AlarmStatistics": AlarmStatistics,
    "ProvenanceInfo": ProvenanceInfo,
    "DataQualityMetrics": DataQualityMetrics,
}

__all__ = [
    # Enums
    "DataQualityLevel",
    "UnitSystem",
    "AlarmSeverity",
    "EquipmentStatus",
    "TripStatus",
    "ForecastConfidence",
    "PriceMarket",
    "FuelType",
    # Base models
    "ProvenanceInfo",
    "DataQualityMetrics",
    "BaseDataContract",
    # Domain schemas
    "ProcessSensorData",
    "EnergyConsumptionData",
    "SafetySystemStatus",
    "ProductionSchedule",
    "WeatherForecast",
    "EnergyPrices",
    "EquipmentHealth",
    "AlarmState",
    # Sub-schemas
    "SteamHeaderData",
    "ValvePosition",
    "FuelConsumption",
    "BoilerPerformance",
    "SISPermissive",
    "TripPoint",
    "BypassRecord",
    "BatchPlan",
    "UnitTarget",
    "Campaign",
    "HourlyForecast",
    "ForecastUncertainty",
    "ElectricityPrice",
    "FuelPrice",
    "TariffPeriod",
    "VibrationData",
    "LubeOilAnalysis",
    "FoulingIndicator",
    "RemainingUsefulLife",
    "AlarmRecord",
    "AlarmStatistics",
    # Registries
    "DOMAIN_SCHEMAS",
    "SUB_SCHEMAS",
]
