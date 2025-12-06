"""
GL-020 ECONOPULSE Agent - Schema Definitions

Pydantic models for economizer optimizer inputs, outputs, and results.
All schemas follow ASME PTC 4.3 and ASME PTC 4.1 standards.

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code (applicable methods)
    - ASME PTC 4.1 Steam Generating Units
    - API 560 Fired Heaters for General Refinery Service
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field, validator


class EconomizerStatus(Enum):
    """Economizer operating status."""
    OFFLINE = "offline"
    STANDBY = "standby"
    WARMING = "warming"
    NORMAL = "normal"
    DEGRADED = "degraded"
    STEAMING_RISK = "steaming_risk"
    ALARM = "alarm"
    TRIP = "trip"


class FoulingType(Enum):
    """Fouling classification."""
    GAS_SIDE = "gas_side"
    WATER_SIDE = "water_side"
    COMBINED = "combined"
    NONE = "none"


class FoulingSeverity(Enum):
    """Fouling severity levels."""
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


class CleaningStatus(Enum):
    """Cleaning recommendation status."""
    NOT_REQUIRED = "not_required"
    MONITOR = "monitor"
    RECOMMENDED = "recommended"
    REQUIRED = "required"
    URGENT = "urgent"


class SootBlowingStatus(Enum):
    """Soot blowing status."""
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    BYPASSED = "bypassed"


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class EconomizerInput(BaseModel):
    """Input data for economizer optimization."""

    # Identity
    economizer_id: str = Field(..., description="Economizer identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )

    # Operating status
    operating_status: EconomizerStatus = Field(
        default=EconomizerStatus.NORMAL,
        description="Current operating status"
    )
    load_pct: float = Field(
        ...,
        ge=0,
        le=120,
        description="Current boiler load percentage"
    )

    # Gas side inlet
    gas_inlet_temp_f: float = Field(
        ...,
        ge=200,
        le=1200,
        description="Gas inlet temperature (F)"
    )
    gas_inlet_flow_lb_hr: float = Field(
        ...,
        gt=0,
        description="Gas flow rate (lb/hr)"
    )
    gas_inlet_pressure_in_wc: float = Field(
        default=0.0,
        description="Gas inlet pressure (in. WC gauge)"
    )

    # Gas side outlet
    gas_outlet_temp_f: float = Field(
        ...,
        ge=100,
        le=800,
        description="Gas outlet temperature (F)"
    )
    gas_outlet_pressure_in_wc: float = Field(
        default=0.0,
        description="Gas outlet pressure (in. WC gauge)"
    )

    # Gas composition (for acid dew point)
    flue_gas_o2_pct: float = Field(
        default=3.0,
        ge=0,
        le=21,
        description="Flue gas O2 content (%)"
    )
    flue_gas_co2_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=25,
        description="Flue gas CO2 content (%)"
    )
    flue_gas_so2_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Flue gas SO2 content (ppm)"
    )
    flue_gas_moisture_pct: float = Field(
        default=10.0,
        ge=0,
        le=30,
        description="Flue gas moisture content (%)"
    )

    # Water side inlet
    water_inlet_temp_f: float = Field(
        ...,
        ge=100,
        le=500,
        description="Water inlet temperature (F)"
    )
    water_inlet_flow_lb_hr: float = Field(
        ...,
        gt=0,
        description="Water flow rate (lb/hr)"
    )
    water_inlet_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Water inlet pressure (psig)"
    )

    # Water side outlet
    water_outlet_temp_f: float = Field(
        ...,
        ge=100,
        le=600,
        description="Water outlet temperature (F)"
    )
    water_outlet_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Water outlet pressure (psig)"
    )

    # Saturation conditions (for steaming detection)
    drum_pressure_psig: float = Field(
        default=500.0,
        ge=0,
        le=3000,
        description="Steam drum pressure (psig)"
    )
    saturation_temp_f: Optional[float] = Field(
        default=None,
        description="Saturation temperature at drum pressure (F)"
    )

    # Cold-end metal temperatures (for acid dew point)
    cold_end_metal_temp_f: Optional[float] = Field(
        default=None,
        description="Cold-end metal temperature (F)"
    )
    cold_end_metal_temps_f: Optional[List[float]] = Field(
        default=None,
        description="Multiple cold-end metal temperatures (F)"
    )

    # Pressure drops
    gas_side_dp_in_wc: Optional[float] = Field(
        default=None,
        ge=0,
        description="Gas-side differential pressure (in. WC)"
    )
    water_side_dp_psi: Optional[float] = Field(
        default=None,
        ge=0,
        description="Water-side differential pressure (psi)"
    )

    # Water chemistry
    feedwater_ph: Optional[float] = Field(
        default=None,
        ge=0,
        le=14,
        description="Feedwater pH"
    )
    feedwater_hardness_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Feedwater hardness (ppm as CaCO3)"
    )
    feedwater_iron_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Feedwater iron (ppm)"
    )
    feedwater_silica_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Feedwater silica (ppm)"
    )

    # Soot blower status
    soot_blower_active: bool = Field(
        default=False,
        description="Soot blower currently active"
    )
    last_soot_blow_timestamp: Optional[datetime] = Field(
        default=None,
        description="Last soot blow completion time"
    )

    # Fuel data (for acid dew point)
    fuel_sulfur_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=5,
        description="Fuel sulfur content (%)"
    )
    fuel_type: Optional[str] = Field(
        default=None,
        description="Fuel type identifier"
    )

    # Ambient conditions
    ambient_temp_f: float = Field(
        default=70.0,
        description="Ambient temperature (F)"
    )
    barometric_pressure_inhg: float = Field(
        default=29.92,
        ge=28,
        le=32,
        description="Barometric pressure (inHg)"
    )

    class Config:
        use_enum_values = True

    @validator("gas_side_dp_in_wc", always=True)
    def calculate_gas_dp(cls, v, values):
        """Calculate gas-side DP if not provided."""
        if v is None:
            inlet = values.get("gas_inlet_pressure_in_wc", 0)
            outlet = values.get("gas_outlet_pressure_in_wc", 0)
            return abs(inlet - outlet)
        return v

    @validator("water_side_dp_psi", always=True)
    def calculate_water_dp(cls, v, values):
        """Calculate water-side DP if not provided."""
        if v is None:
            inlet = values.get("water_inlet_pressure_psig", 0)
            outlet = values.get("water_outlet_pressure_psig", 0)
            return abs(inlet - outlet)
        return v


class SootBlowerInput(BaseModel):
    """Input data for soot blower optimization."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )

    # Current fouling state
    gas_side_dp_ratio: float = Field(
        ...,
        ge=0.5,
        le=5.0,
        description="Gas-side DP ratio (actual/design)"
    )
    effectiveness_ratio: float = Field(
        ...,
        ge=0.5,
        le=1.5,
        description="Effectiveness ratio (actual/design)"
    )
    gas_outlet_temp_deviation_f: float = Field(
        default=0.0,
        description="Gas outlet temp deviation from expected (F)"
    )

    # Operating conditions
    boiler_load_pct: float = Field(
        ...,
        ge=0,
        le=120,
        description="Current boiler load (%)"
    )
    steam_available: bool = Field(
        default=True,
        description="Soot blowing steam available"
    )
    steam_pressure_psig: Optional[float] = Field(
        default=None,
        description="Available steam pressure (psig)"
    )

    # Scheduling
    hours_since_last_blow: float = Field(
        default=0.0,
        ge=0,
        description="Hours since last soot blow"
    )
    scheduled_blow_time: Optional[datetime] = Field(
        default=None,
        description="Next scheduled blow time"
    )


# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

class GasSideFoulingResult(BaseModel):
    """Gas-side fouling analysis result."""

    # Detection
    fouling_detected: bool = Field(..., description="Fouling detected")
    fouling_severity: FoulingSeverity = Field(
        default=FoulingSeverity.NONE,
        description="Fouling severity"
    )
    fouling_trend: str = Field(
        default="stable",
        description="Fouling trend (improving, stable, degrading)"
    )

    # Pressure drop analysis
    current_dp_in_wc: float = Field(
        ...,
        description="Current gas-side DP (in. WC)"
    )
    design_dp_in_wc: float = Field(
        ...,
        description="Design gas-side DP (in. WC)"
    )
    corrected_dp_in_wc: float = Field(
        ...,
        description="Flow-corrected DP (in. WC)"
    )
    dp_ratio: float = Field(
        ...,
        description="DP ratio (corrected/design)"
    )
    dp_deviation_pct: float = Field(
        ...,
        description="DP deviation from design (%)"
    )

    # Heat transfer degradation
    u_actual_btu_hr_ft2_f: float = Field(
        ...,
        description="Actual gas-side U (BTU/hr-ft2-F)"
    )
    u_clean_btu_hr_ft2_f: float = Field(
        ...,
        description="Clean condition U (BTU/hr-ft2-F)"
    )
    u_degradation_pct: float = Field(
        ...,
        description="U-value degradation (%)"
    )

    # Fouling estimate
    estimated_fouling_thickness_in: Optional[float] = Field(
        default=None,
        description="Estimated ash/soot layer thickness (inches)"
    )
    fouling_resistance_hr_ft2_f_btu: float = Field(
        default=0.0,
        description="Calculated fouling resistance"
    )

    # Performance impact
    efficiency_loss_pct: float = Field(
        default=0.0,
        description="Efficiency loss due to fouling (%)"
    )
    fuel_waste_pct: float = Field(
        default=0.0,
        description="Fuel waste due to fouling (%)"
    )

    # Recommendations
    cleaning_status: CleaningStatus = Field(
        default=CleaningStatus.NOT_REQUIRED,
        description="Cleaning status"
    )
    soot_blow_recommended: bool = Field(
        default=False,
        description="Soot blowing recommended"
    )
    estimated_hours_to_cleaning: Optional[float] = Field(
        default=None,
        description="Estimated hours until cleaning needed"
    )

    # Provenance
    calculation_method: str = Field(
        default="ASME_PTC_4.3",
        description="Calculation method"
    )

    class Config:
        use_enum_values = True


class WaterSideFoulingResult(BaseModel):
    """Water-side fouling/scaling analysis result."""

    # Detection
    fouling_detected: bool = Field(..., description="Fouling detected")
    fouling_severity: FoulingSeverity = Field(
        default=FoulingSeverity.NONE,
        description="Fouling severity"
    )
    fouling_type: str = Field(
        default="none",
        description="Fouling type (scale, deposit, corrosion)"
    )

    # Pressure drop analysis
    current_dp_psi: float = Field(..., description="Current water-side DP (psi)")
    design_dp_psi: float = Field(..., description="Design water-side DP (psi)")
    corrected_dp_psi: float = Field(
        ...,
        description="Flow-corrected DP (psi)"
    )
    dp_ratio: float = Field(..., description="DP ratio (corrected/design)")

    # Heat transfer
    fouling_factor_hr_ft2_f_btu: float = Field(
        default=0.0,
        description="Calculated fouling factor"
    )
    design_fouling_factor: float = Field(
        ...,
        description="Design fouling factor"
    )
    fouling_factor_ratio: float = Field(
        ...,
        description="Actual/design fouling factor ratio"
    )

    # Scale/deposit estimate
    estimated_scale_thickness_mils: Optional[float] = Field(
        default=None,
        description="Estimated scale thickness (mils)"
    )
    scale_composition: Optional[str] = Field(
        default=None,
        description="Probable scale composition"
    )

    # Water chemistry compliance
    chemistry_compliant: bool = Field(
        default=True,
        description="Water chemistry within limits"
    )
    chemistry_deviations: List[str] = Field(
        default_factory=list,
        description="Chemistry parameter deviations"
    )

    # Recommendations
    cleaning_status: CleaningStatus = Field(
        default=CleaningStatus.NOT_REQUIRED,
        description="Cleaning status"
    )
    recommended_cleaning_method: Optional[str] = Field(
        default=None,
        description="Recommended cleaning method"
    )

    class Config:
        use_enum_values = True


class SootBlowerResult(BaseModel):
    """Soot blower optimization result."""

    # Status
    blowing_recommended: bool = Field(
        ...,
        description="Soot blowing recommended now"
    )
    blowing_status: SootBlowingStatus = Field(
        default=SootBlowingStatus.IDLE,
        description="Current blowing status"
    )

    # Timing
    hours_since_last_blow: float = Field(
        ...,
        description="Hours since last soot blow"
    )
    recommended_next_blow_hours: float = Field(
        ...,
        description="Recommended hours until next blow"
    )
    optimal_blow_interval_hours: float = Field(
        ...,
        description="Calculated optimal blow interval"
    )

    # Trigger analysis
    dp_trigger_active: bool = Field(
        default=False,
        description="DP trigger activated"
    )
    effectiveness_trigger_active: bool = Field(
        default=False,
        description="Effectiveness trigger activated"
    )
    time_trigger_active: bool = Field(
        default=False,
        description="Time trigger activated"
    )
    trigger_reason: str = Field(
        default="",
        description="Primary trigger reason"
    )

    # Steam consumption
    estimated_steam_per_cycle_lb: float = Field(
        ...,
        description="Estimated steam per cycle (lb)"
    )
    steam_savings_vs_fixed_pct: float = Field(
        default=0.0,
        description="Steam savings vs. fixed schedule (%)"
    )
    daily_steam_consumption_lb: float = Field(
        default=0.0,
        description="Estimated daily steam consumption (lb)"
    )

    # Effectiveness tracking
    pre_blow_effectiveness: Optional[float] = Field(
        default=None,
        description="Effectiveness before last blow"
    )
    post_blow_effectiveness: Optional[float] = Field(
        default=None,
        description="Effectiveness after last blow"
    )
    blow_effectiveness_gain: Optional[float] = Field(
        default=None,
        description="Effectiveness gain from last blow"
    )

    # Optimization metrics
    blowing_efficiency_score: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Blowing optimization efficiency (0-1)"
    )

    class Config:
        use_enum_values = True


class AcidDewPointResult(BaseModel):
    """Acid dew point calculation result."""

    # Dew point values
    sulfuric_acid_dew_point_f: float = Field(
        ...,
        description="Calculated H2SO4 dew point (F)"
    )
    water_dew_point_f: float = Field(
        ...,
        description="Water dew point (F)"
    )
    effective_dew_point_f: float = Field(
        ...,
        description="Effective acid dew point (higher of the two) (F)"
    )

    # Current conditions
    min_metal_temp_f: float = Field(
        ...,
        description="Minimum measured cold-end metal temp (F)"
    )
    avg_metal_temp_f: float = Field(
        ...,
        description="Average cold-end metal temp (F)"
    )
    margin_above_dew_point_f: float = Field(
        ...,
        description="Margin above acid dew point (F)"
    )

    # Safety assessment
    corrosion_risk: str = Field(
        default="low",
        description="Corrosion risk level (low, moderate, high, critical)"
    )
    below_dew_point: bool = Field(
        default=False,
        description="Metal temp below acid dew point"
    )
    margin_adequate: bool = Field(
        default=True,
        description="Safety margin adequate"
    )

    # Fuel/combustion inputs
    so3_concentration_ppm: float = Field(
        ...,
        description="Calculated SO3 concentration (ppm)"
    )
    h2o_concentration_pct: float = Field(
        ...,
        description="H2O concentration (%)"
    )
    excess_air_pct: float = Field(
        ...,
        description="Calculated excess air (%)"
    )

    # Recommendations
    min_recommended_metal_temp_f: float = Field(
        ...,
        description="Minimum recommended metal temperature (F)"
    )
    feedwater_temp_adjustment_f: Optional[float] = Field(
        default=None,
        description="Recommended feedwater temp adjustment (F)"
    )
    action_required: bool = Field(
        default=False,
        description="Corrective action required"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Recommended action"
    )

    # Corrosion impact
    estimated_corrosion_rate_mpy: Optional[float] = Field(
        default=None,
        description="Estimated corrosion rate (mils per year)"
    )
    tube_life_impact_pct: Optional[float] = Field(
        default=None,
        description="Estimated tube life impact (%)"
    )

    # Provenance
    calculation_method: str = Field(
        default="VERHOFF_BANCHERO",
        description="Acid dew point calculation method"
    )
    formula_reference: str = Field(
        default="Verhoff & Banchero, Chemical Engineering Progress, 1974",
        description="Formula reference"
    )


class EffectivenessResult(BaseModel):
    """Heat transfer effectiveness calculation result."""

    # Effectiveness values
    current_effectiveness: float = Field(
        ...,
        ge=0,
        le=1.5,
        description="Current heat transfer effectiveness"
    )
    design_effectiveness: float = Field(
        ...,
        ge=0,
        le=1,
        description="Design effectiveness"
    )
    effectiveness_ratio: float = Field(
        ...,
        description="Actual/design effectiveness ratio"
    )
    effectiveness_deviation_pct: float = Field(
        ...,
        description="Effectiveness deviation from design (%)"
    )

    # NTU analysis
    current_ntu: float = Field(
        ...,
        description="Current Number of Transfer Units"
    )
    design_ntu: float = Field(
        ...,
        description="Design NTU"
    )

    # UA values
    current_ua_btu_hr_f: float = Field(
        ...,
        description="Current UA value (BTU/hr-F)"
    )
    design_ua_btu_hr_f: float = Field(
        ...,
        description="Design UA value (BTU/hr-F)"
    )
    clean_ua_btu_hr_f: float = Field(
        ...,
        description="Clean condition UA (BTU/hr-F)"
    )
    ua_degradation_pct: float = Field(
        ...,
        description="UA degradation from design (%)"
    )

    # Heat duty
    actual_duty_btu_hr: float = Field(
        ...,
        description="Actual heat duty (BTU/hr)"
    )
    expected_duty_btu_hr: float = Field(
        ...,
        description="Expected duty at current conditions (BTU/hr)"
    )
    duty_deficit_btu_hr: float = Field(
        ...,
        description="Duty shortfall (BTU/hr)"
    )

    # Temperature analysis
    lmtd_f: float = Field(..., description="Log mean temperature difference (F)")
    approach_temp_f: float = Field(
        ...,
        description="Water outlet approach to gas inlet (F)"
    )
    gas_temp_drop_f: float = Field(
        ...,
        description="Gas temperature drop (F)"
    )
    water_temp_rise_f: float = Field(
        ...,
        description="Water temperature rise (F)"
    )

    # Capacity ratios
    c_min_btu_hr_f: float = Field(
        ...,
        description="Minimum capacity rate (BTU/hr-F)"
    )
    c_max_btu_hr_f: float = Field(
        ...,
        description="Maximum capacity rate (BTU/hr-F)"
    )
    capacity_ratio: float = Field(
        ...,
        description="Capacity ratio (C_min/C_max)"
    )

    # Performance assessment
    performance_status: str = Field(
        default="normal",
        description="Performance status (normal, degraded, critical)"
    )
    primary_degradation_source: str = Field(
        default="none",
        description="Primary source of degradation"
    )

    # Provenance
    calculation_method: str = Field(
        default="NTU_EPSILON",
        description="Calculation method"
    )
    formula_reference: str = Field(
        default="ASME PTC 4.3 / Incropera Heat Transfer",
        description="Standard reference"
    )


class SteamingResult(BaseModel):
    """Steaming economizer detection result."""

    # Steaming risk
    steaming_detected: bool = Field(
        default=False,
        description="Steaming condition detected"
    )
    steaming_risk: str = Field(
        default="low",
        description="Steaming risk level (low, moderate, high, critical)"
    )
    steaming_risk_score: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Steaming risk score (0-100)"
    )

    # Approach temperature
    approach_temp_f: float = Field(
        ...,
        description="Approach to saturation (F)"
    )
    design_approach_f: float = Field(
        ...,
        description="Design approach temperature (F)"
    )
    approach_margin_f: float = Field(
        ...,
        description="Margin above minimum approach (F)"
    )

    # Saturation conditions
    water_outlet_temp_f: float = Field(
        ...,
        description="Water outlet temperature (F)"
    )
    saturation_temp_f: float = Field(
        ...,
        description="Saturation temperature at outlet pressure (F)"
    )
    subcooling_f: float = Field(
        ...,
        description="Subcooling at outlet (F)"
    )

    # Operating conditions
    current_load_pct: float = Field(
        ...,
        description="Current boiler load (%)"
    )
    water_flow_pct: float = Field(
        ...,
        description="Water flow (% of design)"
    )
    low_load_risk: bool = Field(
        default=False,
        description="Low-load steaming risk"
    )

    # Fluctuation analysis
    dp_fluctuation_detected: bool = Field(
        default=False,
        description="DP fluctuation detected (steaming indicator)"
    )
    temp_fluctuation_detected: bool = Field(
        default=False,
        description="Temperature fluctuation detected"
    )

    # Recommendations
    action_required: bool = Field(
        default=False,
        description="Immediate action required"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Recommended action"
    )
    increase_water_flow: bool = Field(
        default=False,
        description="Increase water flow recommended"
    )
    activate_recirculation: bool = Field(
        default=False,
        description="Activate recirculation recommended"
    )
    reduce_heat_input: bool = Field(
        default=False,
        description="Reduce heat input recommended"
    )

    # Min load limit
    min_safe_load_pct: float = Field(
        default=25.0,
        description="Minimum safe load without steaming risk (%)"
    )
    current_min_load_margin_pct: float = Field(
        default=0.0,
        description="Margin above minimum safe load (%)"
    )


class OptimizationRecommendation(BaseModel):
    """Optimization recommendation."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Recommendation identifier"
    )
    category: str = Field(
        ...,
        description="Category (gas_fouling, water_fouling, soot_blowing, acid_dew_point, steaming)"
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

    estimated_efficiency_gain_pct: Optional[float] = Field(
        default=None,
        description="Estimated efficiency gain (%)"
    )
    estimated_fuel_savings_pct: Optional[float] = Field(
        default=None,
        description="Estimated fuel savings (%)"
    )
    estimated_annual_savings_usd: Optional[float] = Field(
        default=None,
        description="Annual savings ($)"
    )

    implementation_difficulty: str = Field(
        default="low",
        description="Implementation difficulty (low, medium, high)"
    )
    requires_outage: bool = Field(
        default=False,
        description="Requires unit outage"
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

    class Config:
        use_enum_values = True


class EconomizerOutput(BaseModel):
    """Complete output from economizer optimization."""

    # Identity
    economizer_id: str = Field(..., description="Economizer identifier")
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
    operating_status: EconomizerStatus = Field(
        default=EconomizerStatus.NORMAL,
        description="Current operating status"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing time (ms)"
    )

    # Analysis results
    gas_side_fouling: GasSideFoulingResult = Field(
        ...,
        description="Gas-side fouling analysis"
    )
    water_side_fouling: WaterSideFoulingResult = Field(
        ...,
        description="Water-side fouling analysis"
    )
    soot_blower: SootBlowerResult = Field(
        ...,
        description="Soot blower optimization"
    )
    acid_dew_point: AcidDewPointResult = Field(
        ...,
        description="Acid dew point analysis"
    )
    effectiveness: EffectivenessResult = Field(
        ...,
        description="Heat transfer effectiveness"
    )
    steaming: SteamingResult = Field(
        ...,
        description="Steaming detection"
    )

    # Overall fouling assessment
    primary_fouling_type: FoulingType = Field(
        default=FoulingType.NONE,
        description="Primary fouling type"
    )
    overall_fouling_severity: FoulingSeverity = Field(
        default=FoulingSeverity.NONE,
        description="Overall fouling severity"
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

    class Config:
        use_enum_values = True
