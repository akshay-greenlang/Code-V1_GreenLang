"""
GL-017 CONDENSYNC Agent - Schema Definitions

Pydantic models for condenser optimizer inputs, outputs, and results.
All schemas follow HEI Standards for Steam Surface Condensers.

Standards Reference: HEI Standards for Steam Surface Condensers, 12th Edition
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field, validator


class CondenserStatus(Enum):
    """Condenser operating status."""
    OFFLINE = "offline"
    STANDBY = "standby"
    WARMING = "warming"
    NORMAL = "normal"
    DEGRADED = "degraded"
    ALARM = "alarm"
    TRIP = "trip"


class CleaningStatus(Enum):
    """Tube cleaning status."""
    NOT_REQUIRED = "not_required"
    RECOMMENDED = "recommended"
    REQUIRED = "required"
    URGENT = "urgent"
    IN_PROGRESS = "in_progress"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class CondenserInput(BaseModel):
    """Input data for condenser optimization."""

    # Identity
    condenser_id: str = Field(..., description="Condenser identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )

    # Operating status
    operating_status: CondenserStatus = Field(
        default=CondenserStatus.NORMAL,
        description="Current operating status"
    )
    load_pct: float = Field(
        ...,
        ge=0,
        le=120,
        description="Current load percentage"
    )

    # Steam side
    exhaust_steam_flow_lb_hr: float = Field(
        ...,
        gt=0,
        description="Exhaust steam flow (lb/hr)"
    )
    exhaust_steam_pressure_psia: float = Field(
        ...,
        gt=0,
        le=20,
        description="Exhaust steam pressure (psia)"
    )
    exhaust_steam_quality_pct: float = Field(
        default=95.0,
        ge=80,
        le=100,
        description="Exhaust steam quality (%)"
    )
    hotwell_temperature_f: float = Field(
        ...,
        description="Hotwell temperature (F)"
    )
    hotwell_level_pct: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description="Hotwell level (%)"
    )

    # Vacuum
    condenser_vacuum_inhga: float = Field(
        ...,
        gt=0,
        le=10,
        description="Condenser vacuum (inHgA)"
    )
    saturation_temperature_f: float = Field(
        ...,
        description="Saturation temperature at vacuum (F)"
    )

    # Cooling water inlet
    cw_inlet_temperature_f: float = Field(
        ...,
        description="Cooling water inlet temperature (F)"
    )
    cw_inlet_flow_gpm: float = Field(
        ...,
        gt=0,
        description="Cooling water flow (GPM)"
    )
    cw_inlet_pressure_psig: Optional[float] = Field(
        default=None,
        description="Cooling water inlet pressure (psig)"
    )

    # Cooling water outlet
    cw_outlet_temperature_f: float = Field(
        ...,
        description="Cooling water outlet temperature (F)"
    )
    cw_outlet_pressure_psig: Optional[float] = Field(
        default=None,
        description="Cooling water outlet pressure (psig)"
    )

    # Waterbox
    waterbox_dp_psi: Optional[float] = Field(
        default=None,
        ge=0,
        description="Waterbox differential pressure (psi)"
    )

    # Cooling tower (if applicable)
    wet_bulb_temperature_f: Optional[float] = Field(
        default=None,
        description="Wet bulb temperature (F)"
    )
    dry_bulb_temperature_f: Optional[float] = Field(
        default=None,
        description="Dry bulb temperature (F)"
    )
    relative_humidity_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Relative humidity (%)"
    )

    # Cooling tower chemistry
    cw_conductivity_umhos: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cooling water conductivity (umhos/cm)"
    )
    cw_ph: Optional[float] = Field(
        default=None,
        ge=0,
        le=14,
        description="Cooling water pH"
    )
    cw_chlorides_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cooling water chlorides (ppm)"
    )
    cw_silica_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cooling water silica (ppm)"
    )
    makeup_water_flow_gpm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Makeup water flow (GPM)"
    )
    blowdown_flow_gpm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Blowdown flow (GPM)"
    )

    # Vacuum system
    air_ejector_suction_pressure_inhga: Optional[float] = Field(
        default=None,
        description="Air ejector suction pressure (inHgA)"
    )
    air_removal_scfm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Air removal rate (SCFM)"
    )
    motive_steam_flow_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Motive steam flow (lb/hr)"
    )
    motive_steam_pressure_psig: Optional[float] = Field(
        default=None,
        description="Motive steam pressure (psig)"
    )

    # Air ingress indicators
    condensate_dissolved_o2_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Condensate dissolved oxygen (ppb)"
    )
    subcooling_f: Optional[float] = Field(
        default=None,
        description="Subcooling below saturation (F)"
    )

    # Ambient conditions
    barometric_pressure_inhg: float = Field(
        default=29.92,
        ge=28,
        le=32,
        description="Barometric pressure (inHg)"
    )

    class Config:
        use_enum_values = True


class CoolingTowerInput(BaseModel):
    """Input data specific to cooling tower analysis."""

    tower_id: str = Field(..., description="Cooling tower identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )

    # Operating conditions
    fan_speed_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Fan speed (%)"
    )
    fan_power_kw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Fan power (kW)"
    )
    fans_operating: int = Field(
        default=1,
        ge=0,
        description="Number of fans operating"
    )

    # Temperatures
    hot_water_temp_f: float = Field(
        ...,
        description="Hot water temperature (F)"
    )
    cold_water_temp_f: float = Field(
        ...,
        description="Cold water temperature (F)"
    )
    wet_bulb_temp_f: float = Field(
        ...,
        description="Wet bulb temperature (F)"
    )
    dry_bulb_temp_f: Optional[float] = Field(
        default=None,
        description="Dry bulb temperature (F)"
    )

    # Water flow
    circulation_flow_gpm: float = Field(
        ...,
        gt=0,
        description="Circulation flow (GPM)"
    )
    makeup_flow_gpm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Makeup water flow (GPM)"
    )
    blowdown_flow_gpm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Blowdown flow (GPM)"
    )
    evaporation_rate_gpm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Evaporation rate (GPM)"
    )

    # Water chemistry
    makeup_conductivity_umhos: Optional[float] = Field(
        default=None,
        ge=0,
        description="Makeup water conductivity (umhos/cm)"
    )
    tower_conductivity_umhos: Optional[float] = Field(
        default=None,
        ge=0,
        description="Tower water conductivity (umhos/cm)"
    )
    ph: Optional[float] = Field(
        default=None,
        ge=0,
        le=14,
        description="Tower water pH"
    )
    free_chlorine_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Free chlorine residual (ppm)"
    )


class VacuumSystemInput(BaseModel):
    """Input data for vacuum system analysis."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )

    # Vacuum readings
    condenser_vacuum_inhga: float = Field(
        ...,
        gt=0,
        description="Condenser vacuum (inHgA)"
    )
    first_stage_suction_inhga: Optional[float] = Field(
        default=None,
        description="First stage ejector suction (inHgA)"
    )
    second_stage_suction_inhga: Optional[float] = Field(
        default=None,
        description="Second stage ejector suction (inHgA)"
    )

    # Air removal
    air_removal_scfm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Measured air removal (SCFM)"
    )

    # Motive steam
    motive_steam_pressure_psig: float = Field(
        ...,
        description="Motive steam pressure (psig)"
    )
    motive_steam_temperature_f: Optional[float] = Field(
        default=None,
        description="Motive steam temperature (F)"
    )
    motive_steam_flow_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Motive steam flow (lb/hr)"
    )

    # Intercondenser
    intercondenser_outlet_temp_f: Optional[float] = Field(
        default=None,
        description="Intercondenser outlet temperature (F)"
    )
    aftercondenser_outlet_temp_f: Optional[float] = Field(
        default=None,
        description="Aftercondenser outlet temperature (F)"
    )


# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

class CleanlinessResult(BaseModel):
    """HEI cleanliness factor calculation result."""

    cleanliness_factor: float = Field(
        ...,
        ge=0,
        le=1.2,
        description="HEI cleanliness factor"
    )
    design_cleanliness: float = Field(
        ...,
        description="Design cleanliness factor"
    )
    cleanliness_ratio: float = Field(
        ...,
        description="Actual / Design ratio"
    )

    # Heat transfer coefficients
    u_actual_btu_hr_ft2_f: float = Field(
        ...,
        description="Actual overall U (BTU/hr-ft2-F)"
    )
    u_clean_btu_hr_ft2_f: float = Field(
        ...,
        description="Clean tube U (BTU/hr-ft2-F)"
    )
    u_design_btu_hr_ft2_f: float = Field(
        ...,
        description="Design U (BTU/hr-ft2-F)"
    )

    # Fouling
    fouling_factor_hr_ft2_f_btu: float = Field(
        default=0.0,
        description="Calculated fouling factor"
    )
    estimated_fouling_thickness_mils: Optional[float] = Field(
        default=None,
        description="Estimated fouling thickness (mils)"
    )

    # Status
    cleaning_status: CleaningStatus = Field(
        default=CleaningStatus.NOT_REQUIRED,
        description="Cleaning status"
    )
    estimated_days_to_cleaning: Optional[int] = Field(
        default=None,
        description="Estimated days until cleaning needed"
    )

    # Calculation details
    lmtd_f: float = Field(..., description="Log mean temperature difference (F)")
    heat_duty_btu_hr: float = Field(..., description="Heat duty (BTU/hr)")
    surface_area_ft2: float = Field(..., description="Heat transfer surface area (ft2)")

    # Provenance
    calculation_method: str = Field(
        default="HEI_STANDARD",
        description="Calculation method"
    )
    formula_reference: str = Field(
        default="HEI Standards 12th Ed. Section 5",
        description="Standard reference"
    )

    class Config:
        use_enum_values = True


class TubeFoulingResult(BaseModel):
    """Tube fouling analysis result."""

    # Primary metrics
    fouling_detected: bool = Field(
        ...,
        description="Fouling detected"
    )
    fouling_severity: str = Field(
        default="none",
        description="Fouling severity (none, light, moderate, severe)"
    )
    fouling_trend: str = Field(
        default="stable",
        description="Fouling trend (improving, stable, degrading)"
    )

    # Backpressure analysis
    current_backpressure_inhga: float = Field(
        ...,
        description="Current backpressure (inHgA)"
    )
    expected_backpressure_inhga: float = Field(
        ...,
        description="Expected clean backpressure (inHgA)"
    )
    backpressure_penalty_inhg: float = Field(
        ...,
        description="Backpressure penalty (inHg)"
    )
    backpressure_deviation_pct: float = Field(
        ...,
        description="Backpressure deviation (%)"
    )

    # Performance impact
    heat_rate_penalty_btu_kwh: float = Field(
        default=0.0,
        description="Estimated heat rate penalty (BTU/kWh)"
    )
    efficiency_loss_pct: float = Field(
        default=0.0,
        description="Estimated efficiency loss (%)"
    )
    lost_capacity_mw: float = Field(
        default=0.0,
        description="Estimated lost capacity (MW)"
    )
    daily_cost_impact_usd: float = Field(
        default=0.0,
        description="Daily cost impact ($)"
    )

    # Cleaning recommendation
    cleaning_recommended: bool = Field(
        default=False,
        description="Cleaning recommended"
    )
    recommended_cleaning_method: Optional[str] = Field(
        default=None,
        description="Recommended cleaning method"
    )
    estimated_cleaning_benefit_usd: Optional[float] = Field(
        default=None,
        description="Estimated cleaning benefit ($)"
    )


class VacuumSystemResult(BaseModel):
    """Vacuum system analysis result."""

    # Vacuum performance
    vacuum_normal: bool = Field(
        ...,
        description="Vacuum within normal range"
    )
    current_vacuum_inhga: float = Field(
        ...,
        description="Current vacuum (inHgA)"
    )
    expected_vacuum_inhga: float = Field(
        ...,
        description="Expected vacuum (inHgA)"
    )
    vacuum_deviation_inhg: float = Field(
        ...,
        description="Vacuum deviation (inHg)"
    )

    # Air removal capacity
    air_removal_capacity_pct: float = Field(
        default=100.0,
        ge=0,
        description="Air removal capacity (%)"
    )
    estimated_air_ingress_scfm: float = Field(
        default=0.0,
        ge=0,
        description="Estimated air ingress (SCFM)"
    )
    air_ingress_excessive: bool = Field(
        default=False,
        description="Air ingress excessive"
    )

    # Ejector performance
    ejector_efficiency_pct: Optional[float] = Field(
        default=None,
        description="Ejector efficiency (%)"
    )
    motive_steam_consumption_lb_hr: Optional[float] = Field(
        default=None,
        description="Motive steam consumption (lb/hr)"
    )
    motive_steam_specific_lb_scfm: Optional[float] = Field(
        default=None,
        description="Specific steam consumption (lb/SCFM)"
    )

    # Recommendations
    maintenance_required: bool = Field(
        default=False,
        description="Maintenance required"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Recommended action"
    )


class AirIngresResult(BaseModel):
    """Air ingress analysis result."""

    # Detection
    air_ingress_detected: bool = Field(
        ...,
        description="Air ingress detected"
    )
    ingress_severity: str = Field(
        default="none",
        description="Ingress severity (none, minor, moderate, severe)"
    )

    # Quantification
    estimated_air_ingress_scfm: float = Field(
        default=0.0,
        ge=0,
        description="Estimated air ingress (SCFM)"
    )
    subcooling_observed_f: float = Field(
        default=0.0,
        description="Observed subcooling (F)"
    )
    dissolved_o2_ppb: Optional[float] = Field(
        default=None,
        description="Dissolved oxygen (ppb)"
    )

    # Source identification
    probable_leak_locations: List[str] = Field(
        default_factory=list,
        description="Probable leak locations"
    )
    confidence_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Confidence in source identification (%)"
    )

    # Impact assessment
    heat_rate_impact_btu_kwh: float = Field(
        default=0.0,
        description="Heat rate impact (BTU/kWh)"
    )
    dissolved_o2_impact: str = Field(
        default="none",
        description="Impact on feedwater DO"
    )

    # Recommendations
    leak_testing_recommended: bool = Field(
        default=False,
        description="Leak testing recommended"
    )
    recommended_test_method: Optional[str] = Field(
        default=None,
        description="Recommended test method"
    )


class CoolingTowerResult(BaseModel):
    """Cooling tower analysis result."""

    # Performance
    thermal_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=150,
        description="Thermal efficiency (%)"
    )
    approach_f: float = Field(
        ...,
        description="Approach temperature (F)"
    )
    range_f: float = Field(
        ...,
        description="Temperature range (F)"
    )
    liquid_to_gas_ratio: float = Field(
        ...,
        gt=0,
        description="L/G ratio"
    )

    # Water balance
    cycles_of_concentration: float = Field(
        ...,
        ge=1,
        description="Cycles of concentration"
    )
    evaporation_rate_gpm: float = Field(
        ...,
        ge=0,
        description="Evaporation rate (GPM)"
    )
    drift_loss_gpm: float = Field(
        default=0.0,
        ge=0,
        description="Drift loss (GPM)"
    )
    blowdown_rate_gpm: float = Field(
        ...,
        ge=0,
        description="Blowdown rate (GPM)"
    )
    makeup_required_gpm: float = Field(
        ...,
        ge=0,
        description="Required makeup (GPM)"
    )

    # Water chemistry compliance
    chemistry_compliant: bool = Field(
        default=True,
        description="Chemistry within limits"
    )
    chemistry_deviations: List[str] = Field(
        default_factory=list,
        description="Chemistry parameter deviations"
    )
    scaling_potential: str = Field(
        default="low",
        description="Scaling potential (low, moderate, high)"
    )
    corrosion_potential: str = Field(
        default="low",
        description="Corrosion potential (low, moderate, high)"
    )

    # Optimization
    optimal_cycles: float = Field(
        ...,
        ge=1,
        description="Optimal cycles of concentration"
    )
    optimal_blowdown_gpm: float = Field(
        ...,
        ge=0,
        description="Optimal blowdown rate (GPM)"
    )
    water_savings_potential_gpm: float = Field(
        default=0.0,
        description="Potential water savings (GPM)"
    )
    chemical_cost_savings_pct: float = Field(
        default=0.0,
        description="Potential chemical cost savings (%)"
    )


class PerformanceResult(BaseModel):
    """Condenser performance analysis result."""

    # Current performance
    actual_duty_btu_hr: float = Field(
        ...,
        description="Actual heat duty (BTU/hr)"
    )
    design_duty_btu_hr: float = Field(
        ...,
        description="Design heat duty (BTU/hr)"
    )
    duty_ratio_pct: float = Field(
        ...,
        description="Duty ratio (% of design)"
    )

    # Backpressure
    actual_backpressure_inhga: float = Field(
        ...,
        description="Actual backpressure (inHgA)"
    )
    expected_backpressure_inhga: float = Field(
        ...,
        description="Expected backpressure per curve (inHgA)"
    )
    backpressure_deviation_inhg: float = Field(
        ...,
        description="Backpressure deviation (inHg)"
    )
    backpressure_deviation_pct: float = Field(
        ...,
        description="Backpressure deviation (%)"
    )

    # Terminal temperature difference
    ttd_actual_f: float = Field(
        ...,
        description="Actual TTD (F)"
    )
    ttd_design_f: float = Field(
        ...,
        description="Design TTD (F)"
    )
    ttd_deviation_f: float = Field(
        ...,
        description="TTD deviation (F)"
    )

    # Turbine impact
    heat_rate_impact_btu_kwh: float = Field(
        default=0.0,
        description="Heat rate impact (BTU/kWh)"
    )
    capacity_impact_mw: float = Field(
        default=0.0,
        description="Capacity impact (MW)"
    )
    efficiency_impact_pct: float = Field(
        default=0.0,
        description="Efficiency impact (%)"
    )

    # Performance degradation source
    degradation_source: str = Field(
        default="none",
        description="Primary degradation source"
    )
    degradation_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Degradation breakdown by source (%)"
    )


class OptimizationRecommendation(BaseModel):
    """Optimization recommendation."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Recommendation identifier"
    )
    category: str = Field(
        ...,
        description="Category (vacuum, fouling, air_ingress, cooling_tower)"
    )
    priority: AlertSeverity = Field(
        default=AlertSeverity.INFO,
        description="Priority level"
    )
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")

    current_value: Optional[float] = Field(
        default=None,
        description="Current value"
    )
    target_value: Optional[float] = Field(
        default=None,
        description="Target value"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measure"
    )

    estimated_benefit_btu_kwh: Optional[float] = Field(
        default=None,
        description="Heat rate benefit (BTU/kWh)"
    )
    estimated_benefit_mw: Optional[float] = Field(
        default=None,
        description="Capacity benefit (MW)"
    )
    estimated_annual_savings_usd: Optional[float] = Field(
        default=None,
        description="Annual savings ($)"
    )

    implementation_difficulty: str = Field(
        default="low",
        description="Implementation difficulty"
    )
    requires_outage: bool = Field(
        default=False,
        description="Requires unit outage"
    )
    payback_months: Optional[float] = Field(
        default=None,
        description="Estimated payback period (months)"
    )

    class Config:
        use_enum_values = True


class Alert(BaseModel):
    """System alert."""

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Alert identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alert timestamp"
    )
    severity: AlertSeverity = Field(
        ...,
        description="Alert severity"
    )
    category: str = Field(..., description="Alert category")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    value: Optional[float] = Field(default=None, description="Triggering value")
    threshold: Optional[float] = Field(default=None, description="Threshold value")
    unit: Optional[str] = Field(default=None, description="Unit")
    recommended_action: Optional[str] = Field(
        default=None,
        description="Recommended action"
    )
    auto_acknowledge: bool = Field(
        default=False,
        description="Auto-acknowledge when cleared"
    )

    class Config:
        use_enum_values = True


class CondenserOutput(BaseModel):
    """Complete output from condenser optimization."""

    # Identity
    condenser_id: str = Field(..., description="Condenser identifier")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Output timestamp"
    )

    # Status
    status: str = Field(default="success", description="Processing status")
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing time (ms)"
    )

    # Analysis results
    cleanliness: CleanlinessResult = Field(
        ...,
        description="HEI cleanliness analysis"
    )
    tube_fouling: TubeFoulingResult = Field(
        ...,
        description="Tube fouling analysis"
    )
    vacuum_system: VacuumSystemResult = Field(
        ...,
        description="Vacuum system analysis"
    )
    air_ingress: AirIngresResult = Field(
        ...,
        description="Air ingress analysis"
    )
    cooling_tower: Optional[CoolingTowerResult] = Field(
        default=None,
        description="Cooling tower analysis"
    )
    performance: PerformanceResult = Field(
        ...,
        description="Performance analysis"
    )

    # Recommendations
    recommendations: List[OptimizationRecommendation] = Field(
        default_factory=list,
        description="Optimization recommendations"
    )

    # Alerts
    alerts: List[Alert] = Field(
        default_factory=list,
        description="Active alerts"
    )

    # KPIs
    kpis: Dict[str, float] = Field(
        default_factory=dict,
        description="Key performance indicators"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )
    input_hash: Optional[str] = Field(
        default=None,
        description="Input data hash"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
