# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Schema Definitions

This module defines all Pydantic models for inputs, outputs, and intermediate
data structures used by the Steam Trap Monitoring Agent.

All models include comprehensive validation, documentation, and support for
SHA-256 provenance tracking for audit compliance.

Standards:
    - DOE Steam System Best Practices
    - Spirax Sarco Steam Trap Guidelines
    - ASME B16.34 Valve Ratings
    - ISO 6552 Automatic Steam Traps

Example:
    >>> from greenlang.agents.process_heat.gl_008_steam_trap_monitor.schemas import (
    ...     TrapDiagnosticInput,
    ...     TrapDiagnosticOutput,
    ... )
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class TrapStatus(str, Enum):
    """Steam trap operational status."""
    GOOD = "good"                     # Operating correctly
    LEAKING = "leaking"               # Passing live steam
    FAILED_OPEN = "failed_open"       # Blowing through (failed open)
    FAILED_CLOSED = "failed_closed"   # Plugged/blocked (not passing condensate)
    COLD = "cold"                     # No steam/condensate flow
    FLOODED = "flooded"               # Condensate backing up
    UNKNOWN = "unknown"               # Cannot determine


class DiagnosisConfidence(str, Enum):
    """Confidence level in diagnostic assessment."""
    HIGH = "high"           # >90% confidence
    MEDIUM = "medium"       # 70-90% confidence
    LOW = "low"             # 50-70% confidence
    UNCERTAIN = "uncertain" # <50% confidence


class TrendDirection(str, Enum):
    """Trend direction for trap parameters."""
    STABLE = "stable"
    DEGRADING = "degrading"
    IMPROVING = "improving"
    ERRATIC = "erratic"


class MaintenancePriority(str, Enum):
    """Maintenance action priority levels."""
    EMERGENCY = "emergency"    # Immediate action required
    URGENT = "urgent"          # Within 24 hours
    HIGH = "high"              # Within 48 hours
    MEDIUM = "medium"          # Within 1 week
    LOW = "low"                # Within 1 month
    SCHEDULED = "scheduled"    # Next planned outage


class SurveyStatus(str, Enum):
    """Survey completion status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    OVERDUE = "overdue"


# =============================================================================
# SENSOR READING SCHEMAS
# =============================================================================

class SensorReading(BaseModel):
    """Base sensor reading model."""

    sensor_id: str = Field(..., description="Unique sensor identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )
    quality_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Data quality score (0-1)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional sensor metadata"
    )


class UltrasonicReading(SensorReading):
    """Ultrasonic sensor reading for steam trap diagnostics."""

    decibel_level_db: float = Field(
        ...,
        ge=0,
        le=120,
        description="Ultrasonic noise level in dB (typically 30-40 kHz)"
    )
    frequency_khz: float = Field(
        default=38.0,
        gt=0,
        le=100,
        description="Measurement frequency in kHz"
    )
    peak_amplitude: Optional[float] = Field(
        default=None,
        description="Peak amplitude of ultrasonic signal"
    )
    rms_amplitude: Optional[float] = Field(
        default=None,
        description="RMS amplitude of ultrasonic signal"
    )
    cycling_detected: bool = Field(
        default=False,
        description="Whether cycling pattern was detected"
    )
    cycle_frequency_hz: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cycling frequency if detected"
    )
    continuous_flow_detected: bool = Field(
        default=False,
        description="Whether continuous flow (blow-through) detected"
    )
    background_noise_db: Optional[float] = Field(
        default=None,
        ge=0,
        description="Background noise level for comparison"
    )


class TemperatureReading(SensorReading):
    """Temperature sensor reading for steam trap diagnostics."""

    inlet_temp_f: float = Field(
        ...,
        description="Inlet temperature (upstream of trap) in Fahrenheit"
    )
    outlet_temp_f: float = Field(
        ...,
        description="Outlet temperature (downstream of trap) in Fahrenheit"
    )
    ambient_temp_f: Optional[float] = Field(
        default=None,
        description="Ambient temperature for reference"
    )
    delta_t_f: Optional[float] = Field(
        default=None,
        description="Temperature differential (inlet - outlet)"
    )
    saturation_temp_f: Optional[float] = Field(
        default=None,
        description="Steam saturation temperature at operating pressure"
    )
    subcooling_f: Optional[float] = Field(
        default=None,
        description="Degrees of subcooling at outlet"
    )

    @validator("delta_t_f", always=True)
    def calculate_delta_t(cls, v, values):
        """Calculate delta T if not provided."""
        if v is None and "inlet_temp_f" in values and "outlet_temp_f" in values:
            return values["inlet_temp_f"] - values["outlet_temp_f"]
        return v


class VisualInspectionReading(BaseModel):
    """Visual inspection observation for steam traps."""

    inspector_id: str = Field(..., description="Inspector identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Inspection timestamp"
    )
    visible_steam_discharge: bool = Field(
        default=False,
        description="Steam visible at discharge"
    )
    condensate_visible: bool = Field(
        default=False,
        description="Condensate visible at discharge"
    )
    trap_cycling_observed: bool = Field(
        default=False,
        description="Cycling action observed"
    )
    trap_body_condition: str = Field(
        default="good",
        description="Physical condition: good, corroded, damaged"
    )
    insulation_condition: str = Field(
        default="intact",
        description="Insulation status: intact, damaged, missing"
    )
    leaks_detected: bool = Field(
        default=False,
        description="External leaks observed"
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Inspector notes"
    )


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class TrapInfo(BaseModel):
    """Steam trap identification and specification."""

    trap_id: str = Field(..., description="Unique trap identifier")
    tag_number: Optional[str] = Field(
        default=None,
        description="Plant tag number (e.g., ST-001)"
    )
    trap_type: str = Field(
        ...,
        description="Trap type: float_thermostatic, inverted_bucket, thermostatic, thermodynamic"
    )
    manufacturer: Optional[str] = Field(
        default=None,
        description="Trap manufacturer"
    )
    model: Optional[str] = Field(
        default=None,
        description="Trap model number"
    )
    orifice_size_in: Optional[float] = Field(
        default=None,
        gt=0,
        description="Orifice diameter in inches"
    )
    connection_size_in: float = Field(
        default=0.75,
        gt=0,
        le=4.0,
        description="Connection size in inches"
    )
    pressure_rating_psig: float = Field(
        default=150.0,
        gt=0,
        description="Maximum pressure rating (psig)"
    )
    installation_date: Optional[datetime] = Field(
        default=None,
        description="Installation date"
    )
    last_maintenance_date: Optional[datetime] = Field(
        default=None,
        description="Last maintenance date"
    )
    application: str = Field(
        default="drip_leg",
        description="Application: drip_leg, process, tracer, unit_heater"
    )
    location: Optional[str] = Field(
        default=None,
        description="Physical location description"
    )
    area_code: Optional[str] = Field(
        default=None,
        description="Plant area code for survey routing"
    )
    gps_coordinates: Optional[Tuple[float, float]] = Field(
        default=None,
        description="GPS coordinates (latitude, longitude)"
    )


class TrapDiagnosticInput(BaseModel):
    """
    Input data for steam trap diagnostic analysis.

    This model encapsulates all sensor readings and trap information
    required for comprehensive trap health assessment.

    Attributes:
        request_id: Unique request identifier
        trap_info: Trap identification and specifications
        ultrasonic_readings: Ultrasonic sensor data
        temperature_readings: Temperature sensor data
        visual_inspection: Visual inspection results
        operating_conditions: Current operating parameters

    Example:
        >>> input_data = TrapDiagnosticInput(
        ...     trap_info=TrapInfo(trap_id="ST-001", trap_type="float_thermostatic"),
        ...     ultrasonic_readings=[ultrasonic_reading],
        ...     temperature_readings=[temp_reading],
        ... )
    """

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique diagnostic request identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp"
    )

    # Trap identification
    trap_info: TrapInfo = Field(..., description="Trap identification and specs")

    # Sensor readings
    ultrasonic_readings: List[UltrasonicReading] = Field(
        default_factory=list,
        description="Ultrasonic sensor readings"
    )
    temperature_readings: List[TemperatureReading] = Field(
        default_factory=list,
        description="Temperature readings"
    )
    visual_inspection: Optional[VisualInspectionReading] = Field(
        default=None,
        description="Visual inspection results"
    )

    # Operating conditions
    steam_pressure_psig: float = Field(
        ...,
        gt=0,
        description="Operating steam pressure (psig)"
    )
    back_pressure_psig: float = Field(
        default=0.0,
        ge=0,
        description="Back pressure at discharge (psig)"
    )
    condensate_load_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Expected condensate load (lb/hr)"
    )

    # Historical context
    previous_status: Optional[TrapStatus] = Field(
        default=None,
        description="Previous diagnostic status"
    )
    last_survey_date: Optional[datetime] = Field(
        default=None,
        description="Date of last survey"
    )
    failure_history: List[str] = Field(
        default_factory=list,
        description="Historical failure modes"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrapSurveyInput(BaseModel):
    """Input for trap survey route planning."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )
    plant_id: str = Field(..., description="Plant identifier")
    survey_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Planned survey date"
    )

    # Traps to survey
    trap_ids: List[str] = Field(
        ...,
        min_items=1,
        description="List of trap IDs to include in survey"
    )
    trap_locations: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Trap ID to GPS coordinates mapping"
    )
    trap_areas: Dict[str, str] = Field(
        default_factory=dict,
        description="Trap ID to area code mapping"
    )

    # Constraints
    max_traps_per_route: int = Field(
        default=50,
        gt=0,
        le=200,
        description="Maximum traps per route"
    )
    available_hours: float = Field(
        default=8.0,
        gt=0,
        le=24,
        description="Available survey hours"
    )
    minutes_per_trap: float = Field(
        default=5.0,
        gt=0,
        le=30,
        description="Estimated minutes per trap"
    )

    # Priority weighting
    prioritize_failed: bool = Field(
        default=True,
        description="Prioritize previously failed traps"
    )
    prioritize_high_pressure: bool = Field(
        default=True,
        description="Prioritize high-pressure systems"
    )


class CondensateLoadInput(BaseModel):
    """Input for condensate load calculation."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )

    # Application type
    application: str = Field(
        ...,
        description="Application: drip_leg, heat_exchanger, tracer, unit_heater, process"
    )

    # Steam conditions
    steam_pressure_psig: float = Field(
        ...,
        gt=0,
        description="Operating steam pressure (psig)"
    )
    steam_temperature_f: Optional[float] = Field(
        default=None,
        description="Steam temperature if superheated"
    )

    # Pipe/equipment parameters
    pipe_diameter_in: Optional[float] = Field(
        default=None,
        gt=0,
        description="Pipe diameter for drip legs"
    )
    pipe_length_ft: Optional[float] = Field(
        default=None,
        gt=0,
        description="Pipe length for drip legs"
    )
    pipe_material: str = Field(
        default="carbon_steel",
        description="Pipe material: carbon_steel, stainless, copper"
    )

    # Heat transfer equipment
    heat_transfer_rate_btu_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Heat transfer rate for exchangers"
    )
    equipment_surface_area_ft2: Optional[float] = Field(
        default=None,
        ge=0,
        description="Equipment surface area"
    )

    # Operating conditions
    ambient_temperature_f: float = Field(
        default=70.0,
        description="Ambient temperature"
    )
    insulation_thickness_in: float = Field(
        default=2.0,
        ge=0,
        description="Insulation thickness"
    )
    insulation_type: str = Field(
        default="calcium_silicate",
        description="Insulation type: calcium_silicate, mineral_wool, fiberglass"
    )

    # Load calculation mode
    calculate_startup: bool = Field(
        default=True,
        description="Calculate startup load"
    )
    calculate_operating: bool = Field(
        default=True,
        description="Calculate operating load"
    )
    startup_time_minutes: float = Field(
        default=15.0,
        gt=0,
        description="Startup time period"
    )


# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

class TrapCondition(BaseModel):
    """Detailed trap condition assessment."""

    status: TrapStatus = Field(
        ...,
        description="Overall trap status"
    )
    confidence: DiagnosisConfidence = Field(
        ...,
        description="Diagnosis confidence level"
    )
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Numeric confidence score (0-1)"
    )

    # Failure mode probabilities
    failed_open_probability: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Probability of failed open"
    )
    failed_closed_probability: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Probability of failed closed"
    )
    leaking_probability: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Probability of leaking steam"
    )

    # Diagnostic indicators
    ultrasonic_assessment: Optional[str] = Field(
        default=None,
        description="Ultrasonic diagnostic result"
    )
    temperature_assessment: Optional[str] = Field(
        default=None,
        description="Temperature differential result"
    )
    visual_assessment: Optional[str] = Field(
        default=None,
        description="Visual inspection result"
    )

    # Supporting evidence
    evidence: List[str] = Field(
        default_factory=list,
        description="Evidence supporting diagnosis"
    )
    inconsistencies: List[str] = Field(
        default_factory=list,
        description="Inconsistent observations"
    )


class TrapHealthScore(BaseModel):
    """Trap health scoring model."""

    overall_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall health score (0-100)"
    )
    category: str = Field(
        ...,
        description="Health category: excellent, good, fair, poor, critical"
    )

    # Component scores
    thermal_efficiency_score: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Thermal efficiency score"
    )
    mechanical_condition_score: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Mechanical condition score"
    )
    operational_score: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Operational performance score"
    )

    # Trend
    trend: TrendDirection = Field(
        default=TrendDirection.STABLE,
        description="Health trend direction"
    )
    days_to_critical: Optional[int] = Field(
        default=None,
        ge=0,
        description="Estimated days until critical condition"
    )


class SteamLossEstimate(BaseModel):
    """Steam loss estimation for failed traps."""

    steam_loss_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Estimated steam loss rate (lb/hr)"
    )
    steam_loss_lb_year: float = Field(
        default=0.0,
        ge=0,
        description="Annual steam loss (lb/year)"
    )

    # Energy impact
    energy_loss_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Energy loss (MMBTU/hr)"
    )
    energy_loss_mmbtu_year: float = Field(
        default=0.0,
        ge=0,
        description="Annual energy loss (MMBTU/year)"
    )

    # Cost impact
    cost_per_hour_usd: float = Field(
        default=0.0,
        ge=0,
        description="Hourly cost of steam loss"
    )
    cost_per_year_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual cost of steam loss"
    )

    # Carbon impact
    co2_emissions_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="CO2 emissions (lb/hr)"
    )
    co2_emissions_tons_year: float = Field(
        default=0.0,
        ge=0,
        description="Annual CO2 emissions (tons/year)"
    )

    # Calculation basis
    calculation_method: str = Field(
        default="orifice_flow",
        description="Calculation method used"
    )
    orifice_diameter_in: Optional[float] = Field(
        default=None,
        description="Assumed orifice diameter"
    )
    operating_hours_per_year: int = Field(
        default=8760,
        ge=0,
        description="Operating hours per year"
    )


class MaintenanceRecommendation(BaseModel):
    """Maintenance action recommendation."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique recommendation ID"
    )
    priority: MaintenancePriority = Field(
        ...,
        description="Action priority"
    )
    action: str = Field(
        ...,
        description="Recommended action"
    )
    description: str = Field(
        ...,
        max_length=500,
        description="Detailed description"
    )

    # Timing
    deadline_hours: Optional[float] = Field(
        default=None,
        description="Recommended deadline (hours)"
    )

    # Resource estimates
    estimated_duration_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated repair duration"
    )
    estimated_cost_usd: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated repair cost"
    )

    # Parts
    parts_required: List[str] = Field(
        default_factory=list,
        description="Required parts/materials"
    )

    # Justification
    reason: str = Field(
        ...,
        description="Reason for recommendation"
    )
    potential_savings_usd: Optional[float] = Field(
        default=None,
        ge=0,
        description="Potential annual savings if addressed"
    )


class FailureModeProbability(BaseModel):
    """Failure mode probability with explanation."""

    failure_mode: str = Field(
        ...,
        description="Failure mode: failed_open, failed_closed, leaking"
    )
    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability (0-1)"
    )
    confidence: DiagnosisConfidence = Field(
        ...,
        description="Confidence in assessment"
    )
    indicators: List[str] = Field(
        default_factory=list,
        description="Supporting indicators"
    )
    contradictors: List[str] = Field(
        default_factory=list,
        description="Contradicting indicators"
    )


class TrapDiagnosticOutput(BaseModel):
    """
    Output from steam trap diagnostic analysis.

    This comprehensive output contains trap status, health assessment,
    steam loss estimates, and maintenance recommendations.

    Attributes:
        request_id: Original request identifier
        trap_id: Trap identifier
        condition: Detailed trap condition
        health_score: Health scoring
        steam_loss: Steam loss estimates
        recommendations: Maintenance recommendations

    Example:
        >>> output = agent.process(diagnostic_input)
        >>> print(f"Trap {output.trap_id}: {output.condition.status}")
        >>> if output.steam_loss.cost_per_year_usd > 1000:
        ...     print("High-priority repair needed")
    """

    request_id: str = Field(..., description="Original request ID")
    trap_id: str = Field(..., description="Trap identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )
    status: str = Field(
        default="success",
        description="Analysis status"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing duration in milliseconds"
    )

    # Diagnostic results
    condition: TrapCondition = Field(
        ...,
        description="Trap condition assessment"
    )
    health_score: TrapHealthScore = Field(
        ...,
        description="Health scoring"
    )
    failure_probabilities: List[FailureModeProbability] = Field(
        default_factory=list,
        description="Failure mode probabilities"
    )

    # Steam loss
    steam_loss: SteamLossEstimate = Field(
        default_factory=SteamLossEstimate,
        description="Steam loss estimates"
    )

    # Recommendations
    recommendations: List[MaintenanceRecommendation] = Field(
        default_factory=list,
        description="Maintenance recommendations"
    )

    # Diagnostic details
    diagnostic_methods_used: List[str] = Field(
        default_factory=list,
        description="Diagnostic methods applied"
    )
    sensor_data_quality: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Overall sensor data quality"
    )

    # Compliance
    asme_b16_34_compliant: bool = Field(
        default=True,
        description="ASME B16.34 compliance status"
    )
    pressure_rating_adequate: bool = Field(
        default=True,
        description="Pressure rating adequacy"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CondensateLoadOutput(BaseModel):
    """Output from condensate load calculation."""

    request_id: str = Field(..., description="Original request ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )

    # Calculated loads
    startup_load_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Startup condensate load (lb/hr)"
    )
    operating_load_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Operating condensate load (lb/hr)"
    )
    peak_load_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Peak condensate load (lb/hr)"
    )

    # Safety factor applied
    safety_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="Applied safety factor (DOE recommended)"
    )
    design_load_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Design load with safety factor (lb/hr)"
    )

    # Trap sizing recommendation
    recommended_trap_capacity_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Recommended trap capacity"
    )
    recommended_orifice_size_in: Optional[float] = Field(
        default=None,
        description="Recommended orifice size"
    )
    recommended_trap_types: List[str] = Field(
        default_factory=list,
        description="Suitable trap types for application"
    )

    # Calculation details
    calculation_method: str = Field(
        default="",
        description="Calculation method used"
    )
    formula_reference: str = Field(
        default="",
        description="Engineering reference"
    )

    # Warnings
    warnings: List[str] = Field(
        default_factory=list,
        description="Calculation warnings"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )


class EconomicAnalysisOutput(BaseModel):
    """Output from economic analysis of trap failures."""

    request_id: str = Field(..., description="Original request ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Summary statistics
    total_traps_analyzed: int = Field(
        default=0,
        ge=0,
        description="Total traps analyzed"
    )
    traps_failed: int = Field(
        default=0,
        ge=0,
        description="Number of failed traps"
    )
    failure_rate_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Failure rate percentage"
    )

    # Steam losses
    total_steam_loss_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total steam loss (lb/hr)"
    )
    total_steam_loss_lb_year: float = Field(
        default=0.0,
        ge=0,
        description="Annual steam loss (lb/year)"
    )

    # Financial impact
    total_annual_loss_usd: float = Field(
        default=0.0,
        ge=0,
        description="Total annual loss"
    )
    repair_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Estimated repair cost"
    )
    net_annual_savings_usd: float = Field(
        default=0.0,
        description="Net annual savings if repaired"
    )

    # ROI analysis
    simple_payback_months: Optional[float] = Field(
        default=None,
        ge=0,
        description="Simple payback period"
    )
    roi_pct: Optional[float] = Field(
        default=None,
        description="Return on investment percentage"
    )
    npv_5year_usd: Optional[float] = Field(
        default=None,
        description="5-year NPV at 10% discount"
    )

    # Environmental impact
    total_co2_reduction_tons_year: float = Field(
        default=0.0,
        ge=0,
        description="Potential CO2 reduction"
    )

    # Top offenders
    top_failing_traps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top failing traps by cost"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )


class RouteStop(BaseModel):
    """Single stop in a survey route."""

    sequence: int = Field(..., ge=1, description="Stop sequence number")
    trap_id: str = Field(..., description="Trap identifier")
    location: Optional[str] = Field(default=None, description="Location description")
    area_code: Optional[str] = Field(default=None, description="Area code")
    gps_coordinates: Optional[Tuple[float, float]] = Field(
        default=None,
        description="GPS coordinates"
    )
    estimated_time_minutes: float = Field(
        default=5.0,
        ge=0,
        description="Estimated time at stop"
    )
    priority: str = Field(default="normal", description="Stop priority")
    special_instructions: Optional[str] = Field(
        default=None,
        description="Special instructions for this trap"
    )


class SurveyRouteOutput(BaseModel):
    """Output from survey route optimization."""

    request_id: str = Field(..., description="Original request ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Optimization timestamp"
    )

    # Route summary
    total_routes: int = Field(
        default=0,
        ge=0,
        description="Total number of routes"
    )
    total_traps: int = Field(
        default=0,
        ge=0,
        description="Total traps in survey"
    )
    total_distance_ft: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total walking distance"
    )
    total_time_hours: float = Field(
        default=0.0,
        ge=0,
        description="Total estimated time"
    )

    # Route details
    routes: List[List[RouteStop]] = Field(
        default_factory=list,
        description="Optimized routes"
    )

    # Optimization metrics
    optimization_method: str = Field(
        default="nearest_neighbor",
        description="TSP optimization method used"
    )
    distance_savings_pct: Optional[float] = Field(
        default=None,
        description="Distance savings vs naive ordering"
    )

    # Coverage
    coverage_by_area: Dict[str, int] = Field(
        default_factory=dict,
        description="Trap count by area"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )


class TrapStatusSummary(BaseModel):
    """Summary of trap population status."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Summary timestamp"
    )
    plant_id: str = Field(..., description="Plant identifier")

    # Population counts
    total_traps: int = Field(default=0, ge=0, description="Total traps")
    traps_good: int = Field(default=0, ge=0, description="Good traps")
    traps_failed_open: int = Field(default=0, ge=0, description="Failed open")
    traps_failed_closed: int = Field(default=0, ge=0, description="Failed closed")
    traps_leaking: int = Field(default=0, ge=0, description="Leaking traps")
    traps_unknown: int = Field(default=0, ge=0, description="Unknown status")

    # Rates
    overall_failure_rate_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Overall failure rate"
    )
    failed_open_rate_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Failed open rate"
    )

    # By type
    status_by_trap_type: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Status breakdown by trap type"
    )
    status_by_area: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Status breakdown by area"
    )

    # Economic summary
    total_annual_loss_usd: float = Field(
        default=0.0,
        ge=0,
        description="Total annual steam loss cost"
    )
    total_steam_loss_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total steam loss rate"
    )

    # Survey status
    last_survey_date: Optional[datetime] = Field(
        default=None,
        description="Date of last complete survey"
    )
    survey_status: SurveyStatus = Field(
        default=SurveyStatus.NOT_STARTED,
        description="Current survey status"
    )
    traps_surveyed_this_cycle: int = Field(
        default=0,
        ge=0,
        description="Traps surveyed in current cycle"
    )

    # Recommendations
    priority_repairs_count: int = Field(
        default=0,
        ge=0,
        description="Number of priority repairs needed"
    )
    estimated_repair_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Estimated total repair cost"
    )
    potential_annual_savings_usd: float = Field(
        default=0.0,
        ge=0,
        description="Potential savings if all repaired"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )


# =============================================================================
# UPDATE FORWARD REFERENCES
# =============================================================================

TrapDiagnosticInput.update_forward_refs()
TrapDiagnosticOutput.update_forward_refs()
CondensateLoadOutput.update_forward_refs()
EconomicAnalysisOutput.update_forward_refs()
SurveyRouteOutput.update_forward_refs()
TrapStatusSummary.update_forward_refs()
