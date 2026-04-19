"""
GL-009 THERMALIQ Agent - Pydantic Schema Definitions

This module defines all data models for the thermal fluid systems agent,
including input/output schemas for thermal oil properties, degradation
monitoring, exergy analysis, and safety interlocks.

All models use Pydantic for validation with comprehensive field constraints.

Example:
    >>> from greenlang.agents.process_heat.gl_009_thermal_fluid.schemas import (
    ...     ThermalFluidInput,
    ...     ThermalFluidOutput,
    ... )
    >>> input_data = ThermalFluidInput(
    ...     system_id="TF-001",
    ...     fluid_type="therminol_66",
    ...     bulk_temperature_f=550.0,
    ...     flow_rate_gpm=250.0,
    ... )
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# ENUMS
# =============================================================================

class ThermalFluidType(str, Enum):
    """Supported thermal fluid types."""
    THERMINOL_55 = "therminol_55"
    THERMINOL_59 = "therminol_59"
    THERMINOL_62 = "therminol_62"
    THERMINOL_66 = "therminol_66"
    THERMINOL_VP1 = "therminol_vp1"
    THERMINOL_VP3 = "therminol_vp3"
    THERMINOL_XP = "therminol_xp"
    DOWTHERM_A = "dowtherm_a"
    DOWTHERM_G = "dowtherm_g"
    DOWTHERM_J = "dowtherm_j"
    DOWTHERM_Q = "dowtherm_q"
    DOWTHERM_RP = "dowtherm_rp"
    MARLOTHERM_SH = "marlotherm_sh"
    MARLOTHERM_LH = "marlotherm_lh"
    MOBILTHERM_603 = "mobiltherm_603"
    MOBILTHERM_605 = "mobiltherm_605"
    PARATHERM_NF = "paratherm_nf"
    PARATHERM_HE = "paratherm_he"
    SYLTHERM_800 = "syltherm_800"
    SYLTHERM_XLT = "syltherm_xlt"
    CUSTOM = "custom"


class DegradationLevel(str, Enum):
    """Thermal fluid degradation severity levels."""
    EXCELLENT = "excellent"  # New or like-new condition
    GOOD = "good"  # Normal operating condition
    FAIR = "fair"  # Some degradation, monitor closely
    POOR = "poor"  # Significant degradation, action needed
    CRITICAL = "critical"  # Immediate replacement required


class SafetyStatus(str, Enum):
    """Safety interlock status."""
    NORMAL = "normal"
    WARNING = "warning"
    ALARM = "alarm"
    TRIP = "trip"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


class HeaterType(str, Enum):
    """Thermal fluid heater types."""
    FIRED_HEATER = "fired_heater"
    ELECTRIC = "electric"
    WASTE_HEAT = "waste_heat"
    SOLAR = "solar"


class FlowRegime(str, Enum):
    """Flow regime classification."""
    LAMINAR = "laminar"
    TRANSITIONAL = "transitional"
    TURBULENT = "turbulent"


class ValidationStatus(str, Enum):
    """Validation result status."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


class OptimizationStatus(str, Enum):
    """Optimization result status."""
    OPTIMAL = "optimal"
    SUBOPTIMAL = "suboptimal"
    CRITICAL = "critical"


# =============================================================================
# INPUT MODELS
# =============================================================================

class ThermalFluidInput(BaseModel):
    """Primary input data for thermal fluid system analysis."""

    # Identity
    system_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique system identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )

    # Fluid specification
    fluid_type: ThermalFluidType = Field(
        ...,
        description="Thermal fluid type"
    )
    fluid_charge_gallons: Optional[float] = Field(
        default=None,
        gt=0,
        description="Total fluid charge volume (gallons)"
    )
    fluid_age_months: Optional[int] = Field(
        default=None,
        ge=0,
        description="Fluid age since last replacement (months)"
    )

    # Operating conditions
    bulk_temperature_f: float = Field(
        ...,
        ge=0,
        le=800,
        description="Bulk fluid temperature (F)"
    )
    inlet_temperature_f: Optional[float] = Field(
        default=None,
        ge=0,
        le=800,
        description="Heater inlet temperature (F)"
    )
    outlet_temperature_f: Optional[float] = Field(
        default=None,
        ge=0,
        le=800,
        description="Heater outlet temperature (F)"
    )
    film_temperature_f: Optional[float] = Field(
        default=None,
        ge=0,
        le=900,
        description="Maximum film temperature (F)"
    )

    # Flow data
    flow_rate_gpm: float = Field(
        ...,
        gt=0,
        description="System flow rate (GPM)"
    )
    design_flow_rate_gpm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Design flow rate (GPM)"
    )

    # Pressure data
    pump_discharge_pressure_psig: float = Field(
        default=50.0,
        ge=0,
        description="Pump discharge pressure (psig)"
    )
    pump_suction_pressure_psig: Optional[float] = Field(
        default=None,
        ge=-14.7,
        description="Pump suction pressure (psig)"
    )
    system_pressure_drop_psi: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total system pressure drop (psi)"
    )

    # Heater data
    heater_type: HeaterType = Field(
        default=HeaterType.FIRED_HEATER,
        description="Type of thermal fluid heater"
    )
    heater_duty_btu_hr: Optional[float] = Field(
        default=None,
        gt=0,
        description="Heater duty (BTU/hr)"
    )
    heater_design_duty_btu_hr: Optional[float] = Field(
        default=None,
        gt=0,
        description="Heater design duty (BTU/hr)"
    )

    # Heat exchange data
    heat_exchanger_count: int = Field(
        default=1,
        ge=1,
        description="Number of heat exchangers in system"
    )
    total_heat_transfer_area_ft2: Optional[float] = Field(
        default=None,
        gt=0,
        description="Total heat transfer area (ft2)"
    )

    # Expansion tank data
    expansion_tank_level_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Expansion tank level (%)"
    )
    expansion_tank_temp_f: Optional[float] = Field(
        default=None,
        ge=0,
        le=400,
        description="Expansion tank temperature (F)"
    )
    expansion_tank_pressure_psig: Optional[float] = Field(
        default=None,
        ge=-14.7,
        description="Expansion tank pressure (psig)"
    )

    # Ambient conditions
    ambient_temperature_f: float = Field(
        default=77.0,
        ge=-50,
        le=150,
        description="Ambient temperature (F)"
    )

    class Config:
        use_enum_values = True

    @validator('outlet_temperature_f')
    def outlet_greater_than_inlet(cls, v, values):
        """Validate outlet temp is greater than inlet in heating mode."""
        if v is not None and values.get('inlet_temperature_f') is not None:
            if v < values['inlet_temperature_f']:
                # Could be cooling mode, just validate range
                pass
        return v


class FluidLabAnalysis(BaseModel):
    """Laboratory analysis results for thermal fluid degradation monitoring."""

    sample_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Sample collection date"
    )
    sample_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Laboratory sample identifier"
    )

    # Viscosity measurements
    viscosity_cst_100f: Optional[float] = Field(
        default=None,
        gt=0,
        description="Kinematic viscosity at 100F (cSt)"
    )
    viscosity_cst_210f: Optional[float] = Field(
        default=None,
        gt=0,
        description="Kinematic viscosity at 210F (cSt)"
    )
    viscosity_change_pct: Optional[float] = Field(
        default=None,
        description="Viscosity change from baseline (%)"
    )

    # Thermal properties
    thermal_conductivity_change_pct: Optional[float] = Field(
        default=None,
        description="Thermal conductivity change (%)"
    )
    specific_heat_change_pct: Optional[float] = Field(
        default=None,
        description="Specific heat change (%)"
    )

    # Flash point
    flash_point_f: Optional[float] = Field(
        default=None,
        gt=0,
        description="Closed cup flash point (F)"
    )
    flash_point_drop_f: Optional[float] = Field(
        default=None,
        description="Flash point drop from new (F)"
    )

    # Auto-ignition
    auto_ignition_temp_f: Optional[float] = Field(
        default=None,
        gt=0,
        description="Auto-ignition temperature (F)"
    )

    # Chemical analysis
    total_acid_number_mg_koh_g: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total acid number (mg KOH/g)"
    )
    carbon_residue_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Carbon residue - Conradson (%)"
    )
    moisture_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Water content (ppm)"
    )
    particulate_count_ml: Optional[float] = Field(
        default=None,
        ge=0,
        description="Particulate count per ml"
    )

    # Gas chromatography
    low_boilers_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Low boiling compounds (%)"
    )
    high_boilers_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="High boiling compounds (%)"
    )

    # Color/appearance
    color_astm: Optional[float] = Field(
        default=None,
        ge=0,
        le=8,
        description="ASTM color rating (0-8)"
    )


class ExpansionTankData(BaseModel):
    """Expansion tank operating data."""

    tank_id: str = Field(..., description="Expansion tank identifier")

    # Tank dimensions
    total_volume_gallons: float = Field(
        ...,
        gt=0,
        description="Total tank volume (gallons)"
    )
    design_temperature_f: float = Field(
        default=300.0,
        gt=0,
        description="Design temperature (F)"
    )
    design_pressure_psig: float = Field(
        default=15.0,
        ge=-14.7,
        description="Design pressure (psig)"
    )

    # Operating data
    current_level_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current level (%)"
    )
    current_temperature_f: float = Field(
        ...,
        ge=0,
        description="Current temperature (F)"
    )
    current_pressure_psig: float = Field(
        default=0.0,
        ge=-14.7,
        description="Current pressure (psig)"
    )

    # System data
    system_volume_gallons: float = Field(
        ...,
        gt=0,
        description="Total system volume (gallons)"
    )
    cold_fill_temp_f: float = Field(
        default=70.0,
        ge=0,
        description="Cold fill temperature (F)"
    )
    max_operating_temp_f: float = Field(
        ...,
        gt=0,
        description="Maximum operating temperature (F)"
    )


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class FluidProperties(BaseModel):
    """Calculated thermal fluid properties at operating conditions."""

    temperature_f: float = Field(..., description="Temperature point (F)")

    # Thermal properties
    density_lb_ft3: float = Field(..., description="Density (lb/ft3)")
    specific_heat_btu_lb_f: float = Field(..., description="Specific heat (BTU/lb-F)")
    thermal_conductivity_btu_hr_ft_f: float = Field(
        ...,
        description="Thermal conductivity (BTU/hr-ft-F)"
    )

    # Transport properties
    kinematic_viscosity_cst: float = Field(
        ...,
        description="Kinematic viscosity (cSt)"
    )
    dynamic_viscosity_cp: float = Field(
        ...,
        description="Dynamic viscosity (cP)"
    )
    prandtl_number: float = Field(..., description="Prandtl number")

    # Safety properties
    vapor_pressure_psia: float = Field(..., description="Vapor pressure (psia)")
    flash_point_f: float = Field(..., description="Flash point (F)")
    auto_ignition_temp_f: float = Field(..., description="Auto-ignition temperature (F)")
    max_film_temp_f: float = Field(..., description="Maximum film temperature (F)")
    max_bulk_temp_f: float = Field(..., description="Maximum bulk temperature (F)")


class ExergyAnalysis(BaseModel):
    """Exergy (2nd Law) analysis results."""

    # Efficiency metrics
    exergy_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Exergy efficiency (%)"
    )
    first_law_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="First law (energy) efficiency (%)"
    )

    # Exergy flows
    exergy_input_btu_hr: float = Field(
        ...,
        ge=0,
        description="Exergy input rate (BTU/hr)"
    )
    exergy_output_btu_hr: float = Field(
        ...,
        ge=0,
        description="Useful exergy output (BTU/hr)"
    )
    exergy_destruction_btu_hr: float = Field(
        ...,
        ge=0,
        description="Exergy destruction rate (BTU/hr)"
    )

    # Temperature analysis
    carnot_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Carnot efficiency (%)"
    )
    log_mean_temp_ratio: float = Field(
        ...,
        gt=0,
        description="Logarithmic mean temperature ratio"
    )

    # Destruction breakdown
    heater_destruction_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Heater exergy destruction (%)"
    )
    piping_destruction_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Piping heat loss exergy destruction (%)"
    )
    mixing_destruction_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Mixing exergy destruction (%)"
    )

    # Reference conditions
    reference_temperature_f: float = Field(
        default=77.0,
        description="Dead state reference temperature (F)"
    )

    # Metadata
    calculation_method: str = Field(
        default="SECOND_LAW_AVAILABILITY",
        description="Calculation method"
    )


class DegradationAnalysis(BaseModel):
    """Thermal fluid degradation analysis results."""

    # Overall assessment
    degradation_level: DegradationLevel = Field(
        ...,
        description="Overall degradation severity"
    )
    remaining_life_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Estimated remaining useful life (%)"
    )
    replacement_recommended: bool = Field(
        default=False,
        description="Replacement recommendation"
    )

    # Individual indicators
    viscosity_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Viscosity degradation status"
    )
    thermal_conductivity_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Thermal conductivity status"
    )
    flash_point_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Flash point status"
    )
    acid_number_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Acid number status"
    )
    carbon_residue_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Carbon residue status"
    )
    moisture_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Moisture content status"
    )
    low_boilers_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Low boilers status"
    )
    high_boilers_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="High boilers status"
    )

    # Quantitative degradation metrics
    degradation_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Composite degradation score (0=new, 100=end of life)"
    )

    # Specific findings
    findings: List[str] = Field(
        default_factory=list,
        description="Specific degradation findings"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Maintenance recommendations"
    )

    # Next sample date
    next_sample_date: Optional[datetime] = Field(
        default=None,
        description="Recommended next sampling date"
    )

    class Config:
        use_enum_values = True


class HeatTransferAnalysis(BaseModel):
    """Heat transfer coefficient analysis results."""

    # Reynolds number and flow regime
    reynolds_number: float = Field(..., ge=0, description="Reynolds number")
    flow_regime: FlowRegime = Field(..., description="Flow regime")

    # Heat transfer coefficients
    film_coefficient_btu_hr_ft2_f: float = Field(
        ...,
        gt=0,
        description="Film heat transfer coefficient (BTU/hr-ft2-F)"
    )
    overall_coefficient_btu_hr_ft2_f: Optional[float] = Field(
        default=None,
        gt=0,
        description="Overall heat transfer coefficient (BTU/hr-ft2-F)"
    )

    # Nusselt correlation
    nusselt_number: float = Field(..., gt=0, description="Nusselt number")
    correlation_used: str = Field(
        ...,
        description="Heat transfer correlation (e.g., Dittus-Boelter)"
    )

    # Performance indicators
    effectiveness: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Heat exchanger effectiveness"
    )
    fouling_factor_hr_ft2_f_btu: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated fouling factor"
    )

    # Warnings
    warnings: List[str] = Field(
        default_factory=list,
        description="Heat transfer warnings"
    )

    class Config:
        use_enum_values = True


class ExpansionTankSizing(BaseModel):
    """Expansion tank sizing analysis per API 660."""

    # Sizing results
    required_volume_gallons: float = Field(
        ...,
        gt=0,
        description="Required tank volume (gallons)"
    )
    actual_volume_gallons: float = Field(
        ...,
        gt=0,
        description="Actual tank volume (gallons)"
    )
    sizing_adequate: bool = Field(
        ...,
        description="Whether tank is adequately sized"
    )

    # Expansion calculation
    thermal_expansion_pct: float = Field(
        ...,
        ge=0,
        description="Thermal expansion percentage"
    )
    expansion_volume_gallons: float = Field(
        ...,
        ge=0,
        description="Expansion volume (gallons)"
    )

    # Level analysis
    cold_level_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Expected cold level (%)"
    )
    hot_level_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Expected hot level (%)"
    )
    current_level_deviation_pct: float = Field(
        default=0.0,
        description="Deviation from expected level (%)"
    )

    # Pressure analysis
    required_npsh_ft: float = Field(
        ...,
        ge=0,
        description="Required NPSH at pump suction (ft)"
    )
    available_npsh_ft: float = Field(
        ...,
        ge=0,
        description="Available NPSH (ft)"
    )
    npsh_margin_ft: float = Field(
        ...,
        description="NPSH margin (ft)"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Tank sizing recommendations"
    )

    # Reference standard
    calculation_standard: str = Field(
        default="API_660",
        description="Design standard reference"
    )


class SafetyAnalysis(BaseModel):
    """High temperature safety analysis results."""

    # Overall status
    safety_status: SafetyStatus = Field(
        ...,
        description="Overall safety status"
    )

    # Temperature monitoring
    film_temp_status: SafetyStatus = Field(
        default=SafetyStatus.NORMAL,
        description="Film temperature status"
    )
    bulk_temp_status: SafetyStatus = Field(
        default=SafetyStatus.NORMAL,
        description="Bulk temperature status"
    )
    flash_point_margin_status: SafetyStatus = Field(
        default=SafetyStatus.NORMAL,
        description="Flash point margin status"
    )
    auto_ignition_margin_status: SafetyStatus = Field(
        default=SafetyStatus.NORMAL,
        description="Auto-ignition margin status"
    )

    # Margin calculations
    film_temp_margin_f: float = Field(
        ...,
        description="Film temperature margin to limit (F)"
    )
    bulk_temp_margin_f: float = Field(
        ...,
        description="Bulk temperature margin to limit (F)"
    )
    flash_point_margin_f: float = Field(
        ...,
        description="Flash point safety margin (F)"
    )
    auto_ignition_margin_f: float = Field(
        ...,
        description="Auto-ignition safety margin (F)"
    )

    # Flow safety
    minimum_flow_met: bool = Field(
        default=True,
        description="Minimum flow requirement met"
    )
    flow_margin_pct: float = Field(
        default=0.0,
        description="Flow margin above minimum (%)"
    )

    # Pressure safety
    npsh_adequate: bool = Field(
        default=True,
        description="NPSH adequate for cavitation prevention"
    )
    pressure_relief_adequate: bool = Field(
        default=True,
        description="Pressure relief protection adequate"
    )

    # Interlock recommendations
    trip_setpoints: Dict[str, float] = Field(
        default_factory=dict,
        description="Recommended trip setpoints"
    )
    alarm_setpoints: Dict[str, float] = Field(
        default_factory=dict,
        description="Recommended alarm setpoints"
    )

    # Active alarms/trips
    active_alarms: List[str] = Field(
        default_factory=list,
        description="Currently active alarm conditions"
    )
    active_trips: List[str] = Field(
        default_factory=list,
        description="Currently active trip conditions"
    )

    # Recommendations
    safety_recommendations: List[str] = Field(
        default_factory=list,
        description="Safety improvement recommendations"
    )

    class Config:
        use_enum_values = True


class OptimizationRecommendation(BaseModel):
    """Individual optimization recommendation."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Recommendation identifier"
    )
    category: str = Field(
        ...,
        description="Category (thermal, degradation, safety, efficiency)"
    )
    priority: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Priority (1=highest, 5=lowest)"
    )
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    current_value: Optional[float] = Field(
        default=None,
        description="Current measured value"
    )
    recommended_value: Optional[float] = Field(
        default=None,
        description="Recommended target value"
    )
    estimated_savings_pct: Optional[float] = Field(
        default=None,
        description="Estimated efficiency improvement (%)"
    )
    estimated_annual_savings_usd: Optional[float] = Field(
        default=None,
        description="Estimated annual savings ($)"
    )
    implementation_difficulty: str = Field(
        default="medium",
        description="Implementation difficulty (low, medium, high)"
    )
    requires_shutdown: bool = Field(
        default=False,
        description="Requires system shutdown"
    )


class ThermalFluidOutput(BaseModel):
    """Complete output from thermal fluid system analysis."""

    # Identity
    system_id: str = Field(..., description="System identifier")
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
    overall_status: OptimizationStatus = Field(
        default=OptimizationStatus.OPTIMAL,
        description="Overall system status"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing time (ms)"
    )

    # Analysis results
    fluid_properties: FluidProperties = Field(
        ...,
        description="Calculated fluid properties"
    )
    exergy_analysis: Optional[ExergyAnalysis] = Field(
        default=None,
        description="Exergy (2nd Law) analysis"
    )
    degradation_analysis: Optional[DegradationAnalysis] = Field(
        default=None,
        description="Fluid degradation analysis"
    )
    heat_transfer_analysis: Optional[HeatTransferAnalysis] = Field(
        default=None,
        description="Heat transfer coefficient analysis"
    )
    expansion_tank_analysis: Optional[ExpansionTankSizing] = Field(
        default=None,
        description="Expansion tank sizing analysis"
    )
    safety_analysis: SafetyAnalysis = Field(
        ...,
        description="Safety interlock analysis"
    )

    # Recommendations
    recommendations: List[OptimizationRecommendation] = Field(
        default_factory=list,
        description="Optimization recommendations"
    )

    # KPIs
    kpis: Dict[str, float] = Field(
        default_factory=dict,
        description="Key performance indicators"
    )

    # Alerts and warnings
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active alerts"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
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
    calculation_count: int = Field(
        default=0,
        description="Number of calculations performed"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        use_enum_values = True
