"""
GL-016 WATERGUARD Boiler Water Treatment Agent - Schemas Module

This module defines all input/output Pydantic models for the WATERGUARD
Boiler Water Treatment Agent.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
import hashlib

from pydantic import BaseModel, Field, field_validator, model_validator, computed_field

from .config import (
    QualityFlag,
    ComplianceStatus,
    ChemicalType,
    BlowdownMode,
    ConstraintType,
)


class SeverityLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ReasonCode(Enum):
    HIGH_CONDUCTIVITY = "high_conductivity"
    LOW_CONDUCTIVITY = "low_conductivity"
    HIGH_SILICA = "high_silica"
    HIGH_PH = "high_ph"
    LOW_PH = "low_ph"
    HIGH_ALKALINITY = "high_alkalinity"
    LOW_ALKALINITY = "low_alkalinity"
    HIGH_DISSOLVED_O2 = "high_dissolved_o2"
    HIGH_IRON = "high_iron"
    HIGH_COPPER = "high_copper"
    COC_TOO_HIGH = "coc_too_high"
    COC_TOO_LOW = "coc_too_low"
    OPTIMIZE_EFFICIENCY = "optimize_efficiency"
    SAFETY_INTERLOCK = "safety_interlock"
    SCHEDULED = "scheduled"
    MANUAL_REQUEST = "manual_request"


class CoCCalculationMethod(Enum):
    CONDUCTIVITY_RATIO = "conductivity_ratio"
    SILICA_RATIO = "silica_ratio"
    CHLORIDE_RATIO = "chloride_ratio"
    TDS_RATIO = "tds_ratio"
    MASS_BALANCE = "mass_balance"


class DosingMode(Enum):
    CONTINUOUS = "continuous"
    BATCH = "batch"
    SLUG = "slug"
    PROPORTIONAL = "proportional"


class WaterChemistryInput(BaseModel):
    conductivity_us_cm: float = Field(..., ge=0.0, le=20000.0, description="Boiler water conductivity (uS/cm)")
    ph: float = Field(..., ge=0.0, le=14.0, description="Boiler water pH")
    silica_ppm: float = Field(..., ge=0.0, le=1000.0, description="Silica concentration (ppm as SiO2)")
    alkalinity_ppm_caco3: float = Field(..., ge=0.0, le=5000.0, description="Total alkalinity (ppm as CaCO3)")
    dissolved_o2_ppb: Optional[float] = Field(default=None, ge=0.0, le=1000.0, description="Dissolved oxygen (ppb)")
    iron_ppm: Optional[float] = Field(default=None, ge=0.0, le=50.0, description="Iron concentration (ppm)")
    copper_ppm: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Copper concentration (ppm)")
    hardness_ppm: Optional[float] = Field(default=None, ge=0.0, le=500.0, description="Total hardness (ppm as CaCO3)")
    phosphate_ppm: Optional[float] = Field(default=None, ge=0.0, le=500.0, description="Phosphate residual (ppm)")
    sulfite_ppm: Optional[float] = Field(default=None, ge=0.0, le=500.0, description="Sulfite residual (ppm)")
    tds_ppm: Optional[float] = Field(default=None, ge=0.0, le=50000.0, description="Total dissolved solids (ppm)")
    chloride_ppm: Optional[float] = Field(default=None, ge=0.0, le=10000.0, description="Chloride concentration (ppm)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Sample timestamp")
    sample_id: Optional[str] = Field(default=None, description="Sample identifier")
    analyzer_id: Optional[str] = Field(default=None, description="Analyzer identifier")
    quality_flag: QualityFlag = Field(default=QualityFlag.GOOD, description="Data quality indicator")

    @computed_field
    @property
    def data_hash(self) -> str:
        data_str = f"{self.conductivity_us_cm}{self.ph}{self.silica_ppm}{self.alkalinity_ppm_caco3}"
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]


class FeedwaterChemistryInput(BaseModel):
    conductivity_us_cm: float = Field(..., ge=0.0, le=5000.0, description="Feedwater conductivity (uS/cm)")
    ph: float = Field(..., ge=0.0, le=14.0, description="Feedwater pH")
    dissolved_o2_ppb: Optional[float] = Field(default=None, ge=0.0, le=1000.0, description="Dissolved oxygen (ppb)")
    hardness_ppm: Optional[float] = Field(default=None, ge=0.0, le=500.0, description="Total hardness (ppm as CaCO3)")
    silica_ppm: Optional[float] = Field(default=None, ge=0.0, le=200.0, description="Silica (ppm as SiO2)")
    iron_ppm: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Iron (ppm)")
    copper_ppm: Optional[float] = Field(default=None, ge=0.0, le=5.0, description="Copper (ppm)")
    temperature_c: Optional[float] = Field(default=None, ge=0.0, le=200.0, description="Feedwater temperature (C)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Sample timestamp")
    quality_flag: QualityFlag = Field(default=QualityFlag.GOOD, description="Data quality indicator")


class BoilerOperatingInput(BaseModel):
    steam_flow_kg_s: float = Field(..., ge=0.0, le=500.0, description="Steam flow rate (kg/s)")
    pressure_kpa: float = Field(..., ge=100.0, le=25000.0, description="Boiler operating pressure (kPa)")
    feedwater_temp_c: float = Field(..., ge=0.0, le=250.0, description="Feedwater temperature (C)")
    blowdown_flow_kg_s: Optional[float] = Field(default=None, ge=0.0, le=50.0, description="Current blowdown flow (kg/s)")
    load_percent: float = Field(..., ge=0.0, le=110.0, description="Boiler load as percentage of MCR")
    drum_level_percent: Optional[float] = Field(default=None, ge=-50.0, le=150.0, description="Drum level")
    feedwater_flow_kg_s: Optional[float] = Field(default=None, ge=0.0, le=500.0, description="Feedwater flow (kg/s)")
    makeup_water_flow_kg_s: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Makeup water flow (kg/s)")
    condensate_return_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Condensate return ratio")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Data timestamp")
    quality_flag: QualityFlag = Field(default=QualityFlag.GOOD, description="Data quality indicator")


class CyclesOfConcentrationResult(BaseModel):
    coc_value: float = Field(..., ge=1.0, le=50.0, description="Calculated cycles of concentration")
    method: CoCCalculationMethod = Field(..., description="Calculation method used")
    computation_hash: str = Field(..., description="SHA-256 hash for audit trail")
    feedwater_conductivity_us_cm: Optional[float] = Field(default=None, description="Feedwater conductivity used")
    boiler_conductivity_us_cm: Optional[float] = Field(default=None, description="Boiler conductivity used")
    confidence_percent: float = Field(default=95.0, ge=0.0, le=100.0, description="Confidence in calculation")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Calculation timestamp")


class ConstraintDistance(BaseModel):
    parameter: str = Field(..., description="Parameter name")
    current_value: float = Field(..., description="Current measured value")
    limit_value: float = Field(..., description="Limit value")
    distance_percent: float = Field(..., description="Distance to limit as percentage")
    constraint_type: ConstraintType = Field(..., description="Hard or soft constraint")
    time_to_violation_minutes: Optional[float] = Field(default=None, description="Estimated time to violation")


class ComplianceViolation(BaseModel):
    parameter: str = Field(..., description="Parameter in violation")
    current_value: float = Field(..., description="Current value")
    limit_value: float = Field(..., description="Limit exceeded")
    severity: SeverityLevel = Field(..., description="Violation severity")
    constraint_type: ConstraintType = Field(..., description="Hard or soft constraint")
    reason_code: ReasonCode = Field(..., description="Reason code")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Detection timestamp")


class ComplianceWarning(BaseModel):
    parameter: str = Field(..., description="Parameter approaching limit")
    current_value: float = Field(..., description="Current value")
    limit_value: float = Field(..., description="Approaching limit")
    distance_percent: float = Field(..., description="Distance to limit")
    estimated_time_to_violation_min: Optional[float] = Field(default=None, description="Estimated time to violation")
    reason_code: ReasonCode = Field(..., description="Reason code")


class ComplianceStatusResult(BaseModel):
    all_constraints_met: bool = Field(..., description="True if all constraints are met")
    overall_status: ComplianceStatus = Field(..., description="Overall compliance status")
    violations: List[ComplianceViolation] = Field(default_factory=list, description="List of violations")
    warnings: List[ComplianceWarning] = Field(default_factory=list, description="List of warnings")
    distances_to_limits: List[ConstraintDistance] = Field(default_factory=list, description="Distance to each limit")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Assessment timestamp")
    computation_hash: str = Field(default="", description="SHA-256 hash for audit")


class BlowdownRecommendation(BaseModel):
    target_setpoint_percent: float = Field(..., ge=0.0, le=20.0, description="Recommended blowdown rate")
    current_value_percent: float = Field(..., ge=0.0, le=20.0, description="Current blowdown rate")
    reason_code: ReasonCode = Field(..., description="Reason for recommendation")
    constraint_distances: List[ConstraintDistance] = Field(default_factory=list, description="Distances to constraints")
    time_to_violation_minutes: Optional[float] = Field(default=None, description="Time until constraint violation")
    expected_coc_after: float = Field(..., ge=1.0, le=50.0, description="Expected CoC after adjustment")
    energy_impact_kw: Optional[float] = Field(default=None, description="Energy impact of change (kW)")
    confidence_percent: float = Field(default=95.0, ge=0.0, le=100.0, description="Recommendation confidence")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Recommendation timestamp")


class DosingRecommendation(BaseModel):
    chemical_type: ChemicalType = Field(..., description="Chemical to dose")
    rate_ml_min: float = Field(..., ge=0.0, le=10000.0, description="Recommended dosing rate (ml/min)")
    mode: DosingMode = Field(default=DosingMode.CONTINUOUS, description="Dosing mode")
    reason_code: ReasonCode = Field(..., description="Reason for recommendation")
    target_residual_ppm: Optional[float] = Field(default=None, description="Target residual concentration")
    current_residual_ppm: Optional[float] = Field(default=None, description="Current residual")
    duration_minutes: Optional[float] = Field(default=None, description="Dosing duration if batch/slug")
    confidence_percent: float = Field(default=95.0, ge=0.0, le=100.0, description="Recommendation confidence")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Recommendation timestamp")


class CalculationStep(BaseModel):
    step_number: int = Field(..., ge=1, description="Step number in calculation")
    description: str = Field(..., description="Step description")
    formula: str = Field(..., description="Formula used")
    inputs: Dict[str, float] = Field(..., description="Input values")
    output: float = Field(..., description="Output value")
    unit: str = Field(default="", description="Engineering unit")


class ProvenanceRecord(BaseModel):
    calculation_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique calculation ID")
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    output_hash: str = Field(..., description="SHA-256 hash of outputs")
    provenance_hash: str = Field(..., description="Combined provenance hash")
    formula_version: str = Field(default="1.0.0", description="Formula version")
    steps: List[CalculationStep] = Field(default_factory=list, description="Calculation steps")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Calculation timestamp")
    deterministic: bool = Field(default=True, description="Deterministic calculation flag")


class ChemistryState(BaseModel):
    calculated_coc: CyclesOfConcentrationResult = Field(..., description="Calculated CoC")
    compliance_status: ComplianceStatusResult = Field(..., description="Compliance assessment")
    blowdown_recommendation: Optional[BlowdownRecommendation] = Field(default=None, description="Blowdown recommendation")
    dosing_recommendations: List[DosingRecommendation] = Field(default_factory=list, description="Dosing recommendations")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="State timestamp")
    trace_id: str = Field(default_factory=lambda: str(uuid4()), description="Trace ID for debugging")
    provenance: Optional[ProvenanceRecord] = Field(default=None, description="Calculation provenance")


class WaterguardEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique event ID")
    event_type: str = Field(..., description="Event type")
    severity: SeverityLevel = Field(default=SeverityLevel.INFO, description="Event severity")
    system_id: Optional[str] = Field(default=None, description="System identifier")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event payload")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Event timestamp")


class ChemistryEvent(WaterguardEvent):
    parameter: str = Field(..., description="Chemistry parameter")
    value: float = Field(..., description="Parameter value")
    unit: str = Field(default="", description="Engineering unit")
    quality_flag: QualityFlag = Field(default=QualityFlag.GOOD, description="Data quality")


class SafetyEvent(WaterguardEvent):
    safety_function: str = Field(..., description="Safety function triggered")
    action_taken: str = Field(..., description="Action taken")
    interlock_active: bool = Field(default=False, description="Interlock status")


class AnomalyEvent(WaterguardEvent):
    anomaly_type: str = Field(..., description="Type of anomaly")
    affected_sensors: List[str] = Field(default_factory=list, description="Affected sensors")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Anomaly confidence")


class APIResponse(BaseModel):
    success: bool = Field(..., description="Request success status")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    trace_id: str = Field(default_factory=lambda: str(uuid4()), description="Request trace ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Health status")
    agent_id: str = Field(..., description="Agent identifier")
    version: str = Field(..., description="Agent version")
    uptime_seconds: float = Field(..., ge=0.0, description="Agent uptime")
    last_calculation_timestamp: Optional[datetime] = Field(default=None, description="Last calculation time")
    active_violations: int = Field(default=0, ge=0, description="Active violation count")
    kafka_connected: bool = Field(default=False, description="Kafka connection status")
    opcua_connected: bool = Field(default=False, description="OPC-UA connection status")
