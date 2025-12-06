"""
GL-018 UnifiedCombustionOptimizer - Schema Definitions

Pydantic models for combustion optimizer inputs, outputs, and analysis results.
Supports ASME PTC 4.1, API 560, and NFPA 85 compliant data structures.

This module provides data schemas for:
    - Combustion system operating data (input)
    - Optimization results and recommendations (output)
    - Flue gas analysis results
    - Burner tuning and flame stability data
    - Emissions monitoring data
    - BMS sequence status
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid

from pydantic import BaseModel, Field, validator

from .config import BMSSequence, ControlMode, EmissionControlTechnology


# =============================================================================
# ENUMS
# =============================================================================


class OperatingStatus(Enum):
    """Combustion equipment operating status."""
    OFFLINE = "offline"
    STANDBY = "standby"
    PRE_PURGE = "pre_purge"
    IGNITION = "ignition"
    LOW_FIRE = "low_fire"
    MODULATING = "modulating"
    HIGH_FIRE = "high_fire"
    POST_PURGE = "post_purge"
    TRIP = "trip"
    LOCKOUT = "lockout"
    MAINTENANCE = "maintenance"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    TRIP = "trip"


class RecommendationPriority(Enum):
    """Recommendation priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    SAFETY = "safety"


# =============================================================================
# INPUT SCHEMAS
# =============================================================================


class BurnerStatus(BaseModel):
    """Individual burner status."""

    burner_id: str = Field(..., description="Burner identifier")
    is_firing: bool = Field(default=False, description="Burner firing status")
    firing_rate_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Current firing rate (%)"
    )
    flame_signal_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Flame signal strength (%)"
    )
    flame_stable: bool = Field(default=False, description="Flame stability status")
    air_register_position_pct: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description="Air register position (%)"
    )
    fuel_valve_position_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Fuel valve position (%)"
    )


class FlueGasReading(BaseModel):
    """Flue gas analyzer reading."""

    o2_pct: float = Field(
        ...,
        ge=0,
        le=21,
        description="O2 concentration (%)"
    )
    co_ppm: float = Field(
        default=0.0,
        ge=0,
        description="CO concentration (ppm)"
    )
    co2_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=25,
        description="CO2 concentration (%)"
    )
    nox_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="NOx concentration (ppm)"
    )
    so2_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="SO2 concentration (ppm)"
    )
    temperature_f: float = Field(
        ...,
        description="Flue gas temperature (F)"
    )
    velocity_fps: Optional[float] = Field(
        default=None,
        ge=0,
        description="Flue gas velocity (ft/s)"
    )
    analyzer_status: str = Field(
        default="normal",
        description="Analyzer status"
    )


class CombustionInput(BaseModel):
    """
    Complete input data for combustion optimization.

    This model captures all operating data needed for combustion
    optimization including fuel, air, flue gas, and equipment status.
    """

    # Identity
    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )

    # Operating status
    operating_status: OperatingStatus = Field(
        default=OperatingStatus.MODULATING,
        description="Current operating status"
    )
    bms_sequence: BMSSequence = Field(
        default=BMSSequence.RUNNING,
        description="Current BMS sequence"
    )
    load_pct: float = Field(
        ...,
        ge=0,
        le=120,
        description="Current load percentage"
    )

    # Fuel data
    fuel_type: str = Field(default="natural_gas", description="Fuel type")
    fuel_flow_rate: float = Field(
        ...,
        ge=0,
        description="Fuel flow rate (lb/hr or SCF/hr)"
    )
    fuel_pressure_psig: Optional[float] = Field(
        default=None,
        description="Fuel pressure (psig)"
    )
    fuel_temperature_f: Optional[float] = Field(
        default=None,
        description="Fuel temperature (F)"
    )
    fuel_hhv: Optional[float] = Field(
        default=None,
        description="Fuel HHV (BTU/lb or BTU/SCF)"
    )

    # Combustion air
    combustion_air_flow_scfm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Combustion air flow (SCFM)"
    )
    combustion_air_temperature_f: float = Field(
        default=77.0,
        description="Combustion air temperature (F)"
    )
    combustion_air_humidity_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Combustion air humidity (%)"
    )
    air_damper_position_pct: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description="Air damper position (%)"
    )

    # Flue gas data
    flue_gas: FlueGasReading = Field(
        ...,
        description="Flue gas analyzer reading"
    )
    draft_in_wc: Optional[float] = Field(
        default=None,
        description="Furnace draft (inches WC)"
    )

    # Burner status
    burners: List[BurnerStatus] = Field(
        default_factory=list,
        description="Individual burner status"
    )

    # Heat output (for efficiency)
    steam_flow_rate_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Steam flow rate (lb/hr)"
    )
    steam_pressure_psig: Optional[float] = Field(
        default=None,
        ge=0,
        description="Steam pressure (psig)"
    )
    steam_temperature_f: Optional[float] = Field(
        default=None,
        description="Steam temperature (F)"
    )
    feedwater_temperature_f: float = Field(
        default=200.0,
        description="Feedwater temperature (F)"
    )
    feedwater_flow_rate_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Feedwater flow rate (lb/hr)"
    )

    # FGR data
    fgr_rate_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=50,
        description="FGR rate (%)"
    )
    fgr_damper_position_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="FGR damper position (%)"
    )

    # SCR data
    scr_inlet_temp_f: Optional[float] = Field(
        default=None,
        description="SCR inlet temperature (F)"
    )
    scr_outlet_nox_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="SCR outlet NOx (ppm)"
    )
    ammonia_flow_rate_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Ammonia injection rate (lb/hr)"
    )

    # Ambient conditions
    ambient_temperature_f: float = Field(
        default=77.0,
        description="Ambient temperature (F)"
    )
    barometric_pressure_psia: float = Field(
        default=14.696,
        description="Barometric pressure (psia)"
    )

    # Blowdown
    blowdown_rate_pct: float = Field(
        default=2.0,
        ge=0,
        le=20,
        description="Blowdown rate (%)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# ANALYSIS RESULT SCHEMAS
# =============================================================================


class FlueGasAnalysis(BaseModel):
    """Flue gas composition and combustion analysis results per API 560."""

    # Primary calculations
    excess_air_pct: float = Field(..., description="Excess air (%)")
    air_fuel_ratio: float = Field(..., description="Actual air-fuel ratio")
    stoichiometric_air_fuel_ratio: float = Field(
        ...,
        description="Stoichiometric air-fuel ratio"
    )

    # Combustion efficiency
    combustion_efficiency_pct: float = Field(
        ...,
        description="Combustion efficiency (%)"
    )
    stack_loss_pct: float = Field(..., description="Dry stack loss (%)")
    moisture_loss_pct: float = Field(
        default=0.0,
        description="Moisture loss (%)"
    )
    co_loss_pct: float = Field(default=0.0, description="CO loss (%)")

    # O2 analysis
    o2_dry_pct: float = Field(..., description="O2 on dry basis (%)")
    o2_wet_pct: float = Field(..., description="O2 on wet basis (%)")
    optimal_o2_pct: float = Field(..., description="Optimal O2 setpoint (%)")
    o2_deviation_pct: float = Field(
        ...,
        description="Deviation from optimal O2 (%)"
    )

    # CO2 analysis
    co2_max_pct: float = Field(
        ...,
        description="Maximum theoretical CO2 (%)"
    )
    co2_actual_pct: Optional[float] = Field(
        default=None,
        description="Actual CO2 (%)"
    )

    # Dewpoints
    water_dew_point_f: float = Field(
        ...,
        description="Water dew point (F)"
    )
    acid_dew_point_f: float = Field(
        ...,
        description="Acid dew point (F)"
    )
    acid_dew_point_margin_f: float = Field(
        ...,
        description="Margin above acid dew point (F)"
    )

    # Optimization recommendations
    adjust_air_fuel: bool = Field(
        default=False,
        description="Air-fuel adjustment recommended"
    )
    adjustment_direction: Optional[str] = Field(
        default=None,
        description="Adjustment direction (increase_air, decrease_air)"
    )
    estimated_improvement_pct: Optional[float] = Field(
        default=None,
        description="Estimated efficiency improvement (%)"
    )

    # Provenance
    formula_reference: str = Field(
        default="API 560, ASME PTC 4.1",
        description="Calculation standard"
    )


class FlameStabilityAnalysis(BaseModel):
    """Flame stability analysis results."""

    # Flame Stability Index
    flame_stability_index: float = Field(
        ...,
        ge=0,
        le=1,
        description="Flame Stability Index (0-1)"
    )
    fsi_status: str = Field(
        default="normal",
        description="FSI status (optimal, normal, warning, alarm)"
    )

    # Flame characteristics
    flame_intensity_avg: float = Field(
        ...,
        ge=0,
        le=100,
        description="Average flame intensity (%)"
    )
    flame_intensity_variance: float = Field(
        default=0.0,
        ge=0,
        description="Flame intensity variance"
    )
    flicker_frequency_hz: Optional[float] = Field(
        default=None,
        description="Detected flicker frequency (Hz)"
    )

    # Per-burner analysis
    burner_flame_status: Dict[str, bool] = Field(
        default_factory=dict,
        description="Flame status by burner ID"
    )
    burner_fsi: Dict[str, float] = Field(
        default_factory=dict,
        description="FSI by burner ID"
    )

    # Stability factors
    combustion_noise_level: Optional[float] = Field(
        default=None,
        description="Combustion noise level (dB)"
    )
    pressure_pulsation_in_wc: Optional[float] = Field(
        default=None,
        description="Pressure pulsation amplitude (in WC)"
    )

    # Recommendations
    tuning_required: bool = Field(
        default=False,
        description="Burner tuning required"
    )
    tuning_recommendations: List[str] = Field(
        default_factory=list,
        description="Specific tuning recommendations"
    )


class BurnerTuningResult(BaseModel):
    """Burner tuning analysis and recommendations."""

    burner_id: str = Field(..., description="Burner identifier")

    # Current settings
    current_air_register_pct: float = Field(
        ...,
        description="Current air register position (%)"
    )
    current_fuel_pressure_psig: Optional[float] = Field(
        default=None,
        description="Current fuel pressure (psig)"
    )

    # Recommended settings
    recommended_air_register_pct: float = Field(
        ...,
        description="Recommended air register position (%)"
    )
    recommended_fuel_pressure_psig: Optional[float] = Field(
        default=None,
        description="Recommended fuel pressure (psig)"
    )

    # Expected results
    expected_o2_change_pct: float = Field(
        default=0.0,
        description="Expected O2 change (%)"
    )
    expected_co_change_ppm: float = Field(
        default=0.0,
        description="Expected CO change (ppm)"
    )
    expected_nox_change_ppm: float = Field(
        default=0.0,
        description="Expected NOx change (ppm)"
    )

    # Adjustment priority
    priority: str = Field(
        default="medium",
        description="Tuning priority"
    )
    confidence_pct: float = Field(
        default=85.0,
        ge=0,
        le=100,
        description="Recommendation confidence (%)"
    )


class EfficiencyResult(BaseModel):
    """Efficiency calculation results per ASME PTC 4.1."""

    # Efficiency values
    gross_efficiency_pct: float = Field(
        ...,
        description="Gross efficiency (%)"
    )
    net_efficiency_pct: float = Field(
        ...,
        description="Net efficiency (%)"
    )
    combustion_efficiency_pct: float = Field(
        ...,
        description="Combustion efficiency (%)"
    )

    # Loss breakdown (ASME PTC 4.1)
    dry_flue_gas_loss_pct: float = Field(
        ...,
        description="Dry flue gas loss (L1) (%)"
    )
    moisture_in_fuel_loss_pct: float = Field(
        default=0.0,
        description="Moisture in fuel loss (L2) (%)"
    )
    moisture_from_h2_loss_pct: float = Field(
        default=0.0,
        description="Moisture from H2 combustion loss (L3) (%)"
    )
    moisture_in_air_loss_pct: float = Field(
        default=0.0,
        description="Moisture in air loss (L4) (%)"
    )
    radiation_loss_pct: float = Field(
        ...,
        description="Radiation and convection loss (L5) (%)"
    )
    blowdown_loss_pct: float = Field(
        default=0.0,
        description="Blowdown loss (L6) (%)"
    )
    unburned_carbon_loss_pct: float = Field(
        default=0.0,
        description="Unburned carbon loss (L7) (%)"
    )
    unburned_hydrogen_loss_pct: float = Field(
        default=0.0,
        description="CO loss / unburned H2 loss (L8) (%)"
    )
    sensible_heat_loss_pct: float = Field(
        default=0.0,
        description="Sensible heat in residue loss (%)"
    )
    other_losses_pct: float = Field(
        default=0.0,
        description="Other unmeasured losses (%)"
    )
    total_losses_pct: float = Field(
        ...,
        description="Total losses (%)"
    )

    # Energy balance
    heat_input_btu_hr: float = Field(..., description="Heat input (BTU/hr)")
    heat_output_btu_hr: float = Field(..., description="Heat output (BTU/hr)")
    heat_loss_btu_hr: float = Field(..., description="Heat loss (BTU/hr)")

    # Additional data
    excess_air_pct: float = Field(..., description="Excess air (%)")
    fuel_consumption_rate: float = Field(
        ...,
        description="Fuel consumption rate"
    )

    # Metadata
    calculation_method: str = Field(
        default="ASME_PTC_4.1_LOSSES",
        description="Calculation method"
    )
    formula_reference: str = Field(
        default="ASME PTC 4.1-2013",
        description="Standard reference"
    )

    # Uncertainty
    uncertainty_lower_pct: Optional[float] = Field(
        default=None,
        description="Lower uncertainty bound"
    )
    uncertainty_upper_pct: Optional[float] = Field(
        default=None,
        description="Upper uncertainty bound"
    )


class EmissionsAnalysis(BaseModel):
    """Emissions analysis and compliance results."""

    # NOx emissions
    nox_ppm_actual: Optional[float] = Field(
        default=None,
        description="Actual NOx (ppm)"
    )
    nox_ppm_corrected: Optional[float] = Field(
        default=None,
        description="NOx corrected to 3% O2 (ppm)"
    )
    nox_lb_mmbtu: Optional[float] = Field(
        default=None,
        description="NOx emission rate (lb/MMBTU)"
    )
    nox_permit_limit_lb_mmbtu: float = Field(
        ...,
        description="NOx permit limit (lb/MMBTU)"
    )
    nox_compliance_pct: Optional[float] = Field(
        default=None,
        description="NOx as % of permit limit"
    )

    # CO emissions
    co_ppm_actual: float = Field(..., description="Actual CO (ppm)")
    co_ppm_corrected: float = Field(
        ...,
        description="CO corrected to 3% O2 (ppm)"
    )
    co_lb_mmbtu: float = Field(
        ...,
        description="CO emission rate (lb/MMBTU)"
    )
    co_permit_limit_lb_mmbtu: float = Field(
        ...,
        description="CO permit limit (lb/MMBTU)"
    )
    co_compliance_pct: float = Field(
        ...,
        description="CO as % of permit limit"
    )

    # CO2 emissions
    co2_lb_mmbtu: float = Field(
        ...,
        description="CO2 emission rate (lb/MMBTU)"
    )
    co2_tons_hr: float = Field(
        ...,
        description="CO2 emissions (tons/hr)"
    )
    co2_annual_tons_projected: float = Field(
        ...,
        description="Projected annual CO2 (tons)"
    )

    # Control system status
    fgr_active: bool = Field(
        default=False,
        description="FGR system active"
    )
    fgr_rate_pct: Optional[float] = Field(
        default=None,
        description="FGR rate (%)"
    )
    scr_active: bool = Field(
        default=False,
        description="SCR system active"
    )
    scr_efficiency_pct: Optional[float] = Field(
        default=None,
        description="SCR NOx removal efficiency (%)"
    )
    ammonia_slip_ppm: Optional[float] = Field(
        default=None,
        description="Ammonia slip (ppm)"
    )

    # Compliance status
    in_compliance: bool = Field(..., description="Overall compliance status")
    compliance_issues: List[str] = Field(
        default_factory=list,
        description="List of compliance issues"
    )

    # Recommendations
    emission_reduction_potential_pct: Optional[float] = Field(
        default=None,
        description="Potential emission reduction (%)"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Emission reduction recommendations"
    )


class BMSStatus(BaseModel):
    """Burner Management System status per NFPA 85."""

    current_sequence: BMSSequence = Field(
        ...,
        description="Current BMS sequence"
    )
    sequence_time_remaining_s: Optional[float] = Field(
        default=None,
        description="Time remaining in current sequence (s)"
    )

    # Safety interlocks
    all_interlocks_satisfied: bool = Field(
        ...,
        description="All safety interlocks satisfied"
    )
    active_interlocks: List[str] = Field(
        default_factory=list,
        description="List of active interlocks"
    )
    tripped_interlocks: List[str] = Field(
        default_factory=list,
        description="List of tripped interlocks"
    )

    # Purge status
    purge_complete: bool = Field(
        default=False,
        description="Purge cycle complete"
    )
    purge_air_flow_verified: bool = Field(
        default=False,
        description="Purge air flow verified"
    )

    # Flame status
    pilot_flame_proven: bool = Field(
        default=False,
        description="Pilot flame proven"
    )
    main_flame_proven: bool = Field(
        default=False,
        description="Main flame proven"
    )
    flame_detector_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Flame detector status by ID"
    )

    # Operational status
    ready_to_fire: bool = Field(
        default=False,
        description="Ready to fire"
    )
    permissive_satisfied: bool = Field(
        default=False,
        description="All permissives satisfied"
    )

    class Config:
        use_enum_values = True


class OptimizationRecommendation(BaseModel):
    """Optimization recommendation with implementation details."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Recommendation identifier"
    )
    category: str = Field(
        ...,
        description="Category (combustion, emissions, efficiency, safety)"
    )
    priority: RecommendationPriority = Field(
        default=RecommendationPriority.MEDIUM,
        description="Priority level"
    )
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")

    # Current vs recommended
    parameter: Optional[str] = Field(
        default=None,
        description="Parameter to adjust"
    )
    current_value: Optional[float] = Field(
        default=None,
        description="Current value"
    )
    recommended_value: Optional[float] = Field(
        default=None,
        description="Recommended value"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Parameter unit"
    )

    # Expected benefits
    estimated_efficiency_gain_pct: Optional[float] = Field(
        default=None,
        description="Estimated efficiency improvement (%)"
    )
    estimated_nox_reduction_pct: Optional[float] = Field(
        default=None,
        description="Estimated NOx reduction (%)"
    )
    estimated_fuel_savings_pct: Optional[float] = Field(
        default=None,
        description="Estimated fuel savings (%)"
    )
    estimated_annual_savings_usd: Optional[float] = Field(
        default=None,
        description="Estimated annual savings ($)"
    )

    # Implementation
    implementation_difficulty: str = Field(
        default="low",
        description="Implementation difficulty (low, medium, high)"
    )
    requires_shutdown: bool = Field(
        default=False,
        description="Requires equipment shutdown"
    )
    auto_implementable: bool = Field(
        default=False,
        description="Can be auto-implemented by optimizer"
    )

    # Confidence
    confidence_pct: float = Field(
        default=85.0,
        ge=0,
        le=100,
        description="Recommendation confidence (%)"
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
    severity: AlertSeverity = Field(..., description="Alert severity")
    category: str = Field(..., description="Alert category")
    message: str = Field(..., description="Alert message")
    parameter: Optional[str] = Field(
        default=None,
        description="Related parameter"
    )
    value: Optional[float] = Field(
        default=None,
        description="Parameter value that triggered alert"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Threshold that was exceeded"
    )
    acknowledged: bool = Field(
        default=False,
        description="Alert acknowledged"
    )
    action_required: bool = Field(
        default=False,
        description="Operator action required"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# OUTPUT SCHEMA
# =============================================================================


class CombustionOutput(BaseModel):
    """
    Complete output from combustion optimization.

    This model provides comprehensive results including efficiency,
    emissions, flame stability, and optimization recommendations.
    """

    # Identity
    equipment_id: str = Field(..., description="Equipment identifier")
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
    efficiency: EfficiencyResult = Field(
        ...,
        description="Efficiency calculation results"
    )
    flue_gas_analysis: FlueGasAnalysis = Field(
        ...,
        description="Flue gas analysis results"
    )
    flame_stability: FlameStabilityAnalysis = Field(
        ...,
        description="Flame stability analysis"
    )
    emissions: EmissionsAnalysis = Field(
        ...,
        description="Emissions analysis results"
    )
    bms_status: BMSStatus = Field(
        ...,
        description="BMS status"
    )

    # Burner tuning
    burner_tuning: List[BurnerTuningResult] = Field(
        default_factory=list,
        description="Per-burner tuning recommendations"
    )

    # Optimization
    recommendations: List[OptimizationRecommendation] = Field(
        default_factory=list,
        description="Optimization recommendations"
    )

    # Control setpoints
    optimal_o2_setpoint_pct: float = Field(
        ...,
        description="Optimal O2 setpoint (%)"
    )
    optimal_excess_air_pct: float = Field(
        ...,
        description="Optimal excess air (%)"
    )
    recommended_air_damper_pct: Optional[float] = Field(
        default=None,
        description="Recommended air damper position (%)"
    )
    recommended_fgr_rate_pct: Optional[float] = Field(
        default=None,
        description="Recommended FGR rate (%)"
    )

    # KPIs
    kpis: Dict[str, float] = Field(
        default_factory=dict,
        description="Key performance indicators"
    )

    # Alerts
    alerts: List[Alert] = Field(
        default_factory=list,
        description="Active alerts"
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
    calculation_chain: List[str] = Field(
        default_factory=list,
        description="Chain of calculation IDs"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        use_enum_values = True
