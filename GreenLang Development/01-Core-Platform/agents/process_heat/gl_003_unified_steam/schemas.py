"""
GL-003 UNIFIED STEAM SYSTEM OPTIMIZER - Pydantic Data Models

This module provides comprehensive Pydantic data models for steam system
optimization. All models include validation, documentation, and support
for IAPWS-IF97 steam property calculations.

Data Model Categories:
    - Steam property models (pressure, temperature, enthalpy, entropy)
    - Header balance models (flow, pressure, exergy)
    - Quality measurement models (dryness, TDS, conductivity)
    - PRV operation models (Cv, opening, performance)
    - Condensate models (return, flash, recovery)
    - Optimization result models

Example:
    >>> from greenlang.agents.process_heat.gl_003_unified_steam.schemas import (
    ...     SteamProperties,
    ...     HeaderBalanceInput,
    ...     SteamQualityReading,
    ... )
    >>>
    >>> props = SteamProperties(
    ...     pressure_psig=150.0,
    ...     temperature_f=366.0,
    ...     enthalpy_btu_lb=1195.0,
    ... )
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class SteamPhase(Enum):
    """Steam phase states."""
    SUBCOOLED_LIQUID = "subcooled_liquid"
    SATURATED_LIQUID = "saturated_liquid"
    WET_STEAM = "wet_steam"
    SATURATED_VAPOR = "saturated_vapor"
    SUPERHEATED_VAPOR = "superheated_vapor"


class ValidationStatus(Enum):
    """Validation status for calculations."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    UNCHECKED = "unchecked"


class OptimizationStatus(Enum):
    """Optimization operation status."""
    OPTIMAL = "optimal"
    SUBOPTIMAL = "suboptimal"
    CRITICAL = "critical"
    FAILED = "failed"


class TrapStatus(Enum):
    """Steam trap operational status."""
    OPERATING = "operating"
    FAILED_OPEN = "failed_open"
    FAILED_CLOSED = "failed_closed"
    LEAKING = "leaking"
    UNKNOWN = "unknown"


# =============================================================================
# STEAM PROPERTY MODELS
# =============================================================================

class SteamProperties(BaseModel):
    """
    Complete steam thermodynamic properties.

    Based on IAPWS-IF97 formulations with validation.
    """

    # Pressure and temperature
    pressure_psig: float = Field(
        ...,
        ge=-14.696,
        description="Gauge pressure (psig)"
    )
    pressure_psia: Optional[float] = Field(
        default=None,
        description="Absolute pressure (psia)"
    )
    temperature_f: float = Field(
        ...,
        ge=32,
        le=1500,
        description="Temperature (F)"
    )
    saturation_temperature_f: Optional[float] = Field(
        default=None,
        description="Saturation temperature at pressure (F)"
    )

    # Phase and quality
    phase: SteamPhase = Field(
        default=SteamPhase.SATURATED_VAPOR,
        description="Steam phase"
    )
    dryness_fraction: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Steam quality/dryness fraction (0-1)"
    )
    superheat_f: Optional[float] = Field(
        default=None,
        ge=0,
        description="Superheat above saturation (F)"
    )

    # Thermodynamic properties
    enthalpy_btu_lb: float = Field(
        ...,
        description="Specific enthalpy (BTU/lb)"
    )
    entropy_btu_lb_r: Optional[float] = Field(
        default=None,
        description="Specific entropy (BTU/lb-R)"
    )
    specific_volume_ft3_lb: Optional[float] = Field(
        default=None,
        description="Specific volume (ft3/lb)"
    )
    density_lb_ft3: Optional[float] = Field(
        default=None,
        description="Density (lb/ft3)"
    )
    internal_energy_btu_lb: Optional[float] = Field(
        default=None,
        description="Specific internal energy (BTU/lb)"
    )

    # Exergy
    exergy_btu_lb: Optional[float] = Field(
        default=None,
        description="Specific exergy (BTU/lb)"
    )

    @validator('pressure_psia', always=True)
    def calculate_psia(cls, v, values):
        """Calculate absolute pressure from gauge."""
        if v is None and 'pressure_psig' in values:
            return values['pressure_psig'] + 14.696
        return v

    @validator('density_lb_ft3', always=True)
    def calculate_density(cls, v, values):
        """Calculate density from specific volume."""
        if v is None and values.get('specific_volume_ft3_lb'):
            return 1.0 / values['specific_volume_ft3_lb']
        return v

    class Config:
        use_enum_values = True


class SteamFlowMeasurement(BaseModel):
    """Steam flow measurement with uncertainty."""

    flow_rate_lb_hr: float = Field(
        ...,
        ge=0,
        description="Mass flow rate (lb/hr)"
    )
    flow_rate_klb_hr: Optional[float] = Field(
        default=None,
        description="Mass flow rate (klb/hr)"
    )
    volumetric_flow_acfm: Optional[float] = Field(
        default=None,
        description="Actual volumetric flow (ACFM)"
    )
    velocity_ft_s: Optional[float] = Field(
        default=None,
        description="Steam velocity (ft/s)"
    )

    # Measurement quality
    measurement_type: str = Field(
        default="orifice",
        description="Flow measurement method"
    )
    uncertainty_pct: float = Field(
        default=2.0,
        ge=0,
        le=20,
        description="Measurement uncertainty (%)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp"
    )

    @validator('flow_rate_klb_hr', always=True)
    def calculate_klb(cls, v, values):
        """Calculate klb/hr from lb/hr."""
        if v is None and 'flow_rate_lb_hr' in values:
            return values['flow_rate_lb_hr'] / 1000
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# HEADER BALANCE MODELS
# =============================================================================

class HeaderReading(BaseModel):
    """Real-time steam header reading."""

    header_id: str = Field(
        ...,
        description="Header identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )

    # Pressure
    pressure_psig: float = Field(
        ...,
        description="Current pressure (psig)"
    )
    pressure_setpoint_psig: float = Field(
        ...,
        description="Pressure setpoint (psig)"
    )
    pressure_deviation_psi: Optional[float] = Field(
        default=None,
        description="Deviation from setpoint (psi)"
    )

    # Temperature
    temperature_f: float = Field(
        ...,
        description="Current temperature (F)"
    )
    temperature_setpoint_f: Optional[float] = Field(
        default=None,
        description="Temperature setpoint (F)"
    )

    # Flow
    total_supply_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total supply flow (lb/hr)"
    )
    total_demand_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total demand flow (lb/hr)"
    )
    imbalance_lb_hr: Optional[float] = Field(
        default=None,
        description="Supply-demand imbalance (lb/hr)"
    )

    @validator('pressure_deviation_psi', always=True)
    def calculate_pressure_deviation(cls, v, values):
        """Calculate pressure deviation."""
        if v is None:
            if 'pressure_psig' in values and 'pressure_setpoint_psig' in values:
                return values['pressure_psig'] - values['pressure_setpoint_psig']
        return v

    @validator('imbalance_lb_hr', always=True)
    def calculate_imbalance(cls, v, values):
        """Calculate flow imbalance."""
        if v is None:
            supply = values.get('total_supply_lb_hr', 0)
            demand = values.get('total_demand_lb_hr', 0)
            return supply - demand
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HeaderBalanceInput(BaseModel):
    """Input for header balance calculation."""

    # Header identification
    header_id: str = Field(..., description="Header identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Input timestamp"
    )

    # Current conditions
    current_pressure_psig: float = Field(
        ...,
        description="Current header pressure (psig)"
    )
    current_temperature_f: float = Field(
        ...,
        description="Current header temperature (F)"
    )

    # Setpoints
    pressure_setpoint_psig: float = Field(
        ...,
        description="Pressure setpoint (psig)"
    )
    temperature_setpoint_f: Optional[float] = Field(
        default=None,
        description="Temperature setpoint for superheated (F)"
    )

    # Supply sources
    supplies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of supply sources with flows"
    )

    # Demand consumers
    demands: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of demand consumers with flows"
    )

    # Control configuration
    pressure_deadband_psi: float = Field(
        default=2.0,
        ge=0,
        description="Pressure control deadband (psi)"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HeaderBalanceOutput(BaseModel):
    """Output from header balance calculation."""

    header_id: str = Field(..., description="Header identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )

    # Balance status
    status: OptimizationStatus = Field(
        ...,
        description="Balance optimization status"
    )

    # Flow balance
    total_supply_lb_hr: float = Field(
        ...,
        description="Total supply flow (lb/hr)"
    )
    total_demand_lb_hr: float = Field(
        ...,
        description="Total demand flow (lb/hr)"
    )
    imbalance_lb_hr: float = Field(
        ...,
        description="Supply-demand imbalance (lb/hr)"
    )
    imbalance_pct: float = Field(
        ...,
        description="Imbalance as percentage of demand"
    )

    # Pressure status
    pressure_psig: float = Field(..., description="Current pressure (psig)")
    pressure_deviation_psi: float = Field(
        ...,
        description="Deviation from setpoint (psi)"
    )
    pressure_trend: str = Field(
        default="stable",
        description="Pressure trend (rising, falling, stable)"
    )

    # Energy/Exergy analysis
    exergy_supply_btu_hr: Optional[float] = Field(
        default=None,
        description="Total exergy supply rate (BTU/hr)"
    )
    exergy_demand_btu_hr: Optional[float] = Field(
        default=None,
        description="Total exergy demand rate (BTU/hr)"
    )
    exergy_efficiency_pct: Optional[float] = Field(
        default=None,
        description="Header exergy efficiency (%)"
    )

    # Recommendations
    adjustments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recommended supply adjustments"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Calculation time (ms)"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# QUALITY MONITORING MODELS
# =============================================================================

class SteamQualityReading(BaseModel):
    """Steam quality measurement reading per ASME standards."""

    # Identification
    reading_id: str = Field(
        default="",
        description="Reading identifier"
    )
    location_id: str = Field(
        ...,
        description="Sample location identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )

    # Steam conditions
    pressure_psig: float = Field(..., description="Steam pressure (psig)")
    temperature_f: float = Field(..., description="Steam temperature (F)")

    # Dryness/Quality
    dryness_fraction: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Steam dryness fraction (quality)"
    )
    moisture_content_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Moisture content (%)"
    )

    # Water chemistry
    tds_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total Dissolved Solids (ppm)"
    )
    cation_conductivity_us_cm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cation conductivity (microS/cm)"
    )
    specific_conductivity_us_cm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Specific conductivity (microS/cm)"
    )
    ph: Optional[float] = Field(
        default=None,
        ge=0,
        le=14,
        description="pH value"
    )

    # Silica and contaminants
    silica_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Silica content (ppm)"
    )
    sodium_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Sodium content (ppm)"
    )
    dissolved_o2_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Dissolved oxygen (ppb)"
    )
    iron_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Iron content (ppb)"
    )

    @validator('moisture_content_pct', always=True)
    def calculate_moisture(cls, v, values):
        """Calculate moisture from dryness."""
        if v is None and 'dryness_fraction' in values:
            return (1 - values['dryness_fraction']) * 100
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SteamQualityAnalysis(BaseModel):
    """Analysis result for steam quality monitoring."""

    # Source reading
    reading: SteamQualityReading = Field(
        ...,
        description="Source quality reading"
    )

    # Validation status
    overall_status: ValidationStatus = Field(
        ...,
        description="Overall quality status"
    )

    # Parameter statuses
    dryness_status: ValidationStatus = Field(
        default=ValidationStatus.UNCHECKED,
        description="Dryness fraction status"
    )
    tds_status: ValidationStatus = Field(
        default=ValidationStatus.UNCHECKED,
        description="TDS status"
    )
    conductivity_status: ValidationStatus = Field(
        default=ValidationStatus.UNCHECKED,
        description="Cation conductivity status"
    )
    silica_status: ValidationStatus = Field(
        default=ValidationStatus.UNCHECKED,
        description="Silica status"
    )

    # Limits comparison
    limits_exceeded: List[str] = Field(
        default_factory=list,
        description="List of exceeded limits"
    )
    limits_warning: List[str] = Field(
        default_factory=list,
        description="List of warning-level parameters"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Quality improvement recommendations"
    )

    # Carryover estimate
    estimated_carryover_pct: Optional[float] = Field(
        default=None,
        description="Estimated mechanical carryover (%)"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# PRV OPERATION MODELS
# =============================================================================

class PRVOperatingPoint(BaseModel):
    """PRV operating point data."""

    prv_id: str = Field(..., description="PRV identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )

    # Pressures
    inlet_pressure_psig: float = Field(
        ...,
        description="Inlet pressure (psig)"
    )
    outlet_pressure_psig: float = Field(
        ...,
        description="Outlet pressure (psig)"
    )
    pressure_drop_psi: Optional[float] = Field(
        default=None,
        description="Pressure drop across PRV (psi)"
    )

    # Flow
    flow_rate_lb_hr: float = Field(
        ...,
        ge=0,
        description="Flow rate through PRV (lb/hr)"
    )

    # Valve position
    opening_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Valve opening percentage"
    )
    cv_actual: Optional[float] = Field(
        default=None,
        description="Actual Cv at current conditions"
    )

    # Temperatures
    inlet_temperature_f: Optional[float] = Field(
        default=None,
        description="Inlet temperature (F)"
    )
    outlet_temperature_f: Optional[float] = Field(
        default=None,
        description="Outlet temperature (F)"
    )

    @validator('pressure_drop_psi', always=True)
    def calculate_pressure_drop(cls, v, values):
        """Calculate pressure drop."""
        if v is None:
            inlet = values.get('inlet_pressure_psig', 0)
            outlet = values.get('outlet_pressure_psig', 0)
            return inlet - outlet
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PRVSizingInput(BaseModel):
    """Input for PRV sizing calculation per ASME B31.1."""

    prv_id: str = Field(..., description="PRV identifier")

    # Design conditions
    inlet_pressure_psig: float = Field(
        ...,
        gt=0,
        description="Design inlet pressure (psig)"
    )
    outlet_pressure_psig: float = Field(
        ...,
        gt=0,
        description="Design outlet pressure (psig)"
    )
    design_flow_lb_hr: float = Field(
        ...,
        gt=0,
        description="Design flow rate (lb/hr)"
    )

    # Steam conditions
    inlet_temperature_f: Optional[float] = Field(
        default=None,
        description="Inlet steam temperature (F)"
    )
    steam_quality: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Inlet steam quality"
    )

    # Operating range
    min_flow_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Minimum expected flow (lb/hr)"
    )
    max_flow_lb_hr: float = Field(
        ...,
        gt=0,
        description="Maximum expected flow (lb/hr)"
    )

    # Target opening range (per ASME B31.1)
    target_opening_min_pct: float = Field(
        default=50.0,
        ge=20,
        le=80,
        description="Target minimum opening (%)"
    )
    target_opening_max_pct: float = Field(
        default=70.0,
        ge=30,
        le=90,
        description="Target maximum opening (%)"
    )

    @validator('outlet_pressure_psig')
    def outlet_less_than_inlet(cls, v, values):
        """Validate pressure relationship."""
        if 'inlet_pressure_psig' in values:
            if v >= values['inlet_pressure_psig']:
                raise ValueError("Outlet pressure must be < inlet pressure")
        return v


class PRVSizingOutput(BaseModel):
    """Output from PRV sizing calculation per ASME B31.1."""

    prv_id: str = Field(..., description="PRV identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )

    # Calculated Cv
    cv_required: float = Field(
        ...,
        description="Required Cv at design flow"
    )
    cv_recommended: float = Field(
        ...,
        description="Recommended Cv (includes margin)"
    )
    cv_margin_pct: float = Field(
        default=15.0,
        description="Safety margin applied (%)"
    )

    # Opening analysis at design
    opening_at_design_pct: float = Field(
        ...,
        description="Valve opening at design flow (%)"
    )
    opening_at_min_pct: float = Field(
        ...,
        description="Valve opening at minimum flow (%)"
    )
    opening_at_max_pct: float = Field(
        ...,
        description="Valve opening at maximum flow (%)"
    )

    # ASME B31.1 compliance
    meets_opening_targets: bool = Field(
        ...,
        description="Meets 50-70% opening targets"
    )
    opening_target_status: str = Field(
        ...,
        description="Status description"
    )

    # Flow capacity
    max_flow_capacity_lb_hr: float = Field(
        ...,
        description="Maximum flow at 100% opening (lb/hr)"
    )
    rangeability: float = Field(
        ...,
        description="Calculated rangeability"
    )

    # Critical flow check
    is_critical_flow: bool = Field(
        default=False,
        description="Critical (choked) flow condition"
    )
    critical_pressure_ratio: Optional[float] = Field(
        default=None,
        description="Critical pressure ratio if applicable"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Sizing recommendations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 calculation hash"
    )
    formula_reference: str = Field(
        default="ASME B31.1-2020",
        description="Standard reference"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# CONDENSATE MODELS
# =============================================================================

class CondensateReading(BaseModel):
    """Condensate system reading."""

    location_id: str = Field(..., description="Location identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )

    # Flow and temperature
    flow_rate_lb_hr: float = Field(
        ...,
        ge=0,
        description="Condensate flow rate (lb/hr)"
    )
    temperature_f: float = Field(
        ...,
        description="Condensate temperature (F)"
    )
    pressure_psig: float = Field(
        default=0.0,
        description="Condensate pressure (psig)"
    )

    # Quality
    tds_ppm: Optional[float] = Field(
        default=None,
        description="TDS content (ppm)"
    )
    oil_ppm: Optional[float] = Field(
        default=None,
        description="Oil contamination (ppm)"
    )
    iron_ppb: Optional[float] = Field(
        default=None,
        description="Iron content (ppb)"
    )
    ph: Optional[float] = Field(
        default=None,
        description="pH value"
    )

    # Status
    is_contaminated: bool = Field(
        default=False,
        description="Contamination detected flag"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CondensateReturnAnalysis(BaseModel):
    """Analysis of condensate return system."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Return rates
    total_steam_flow_lb_hr: float = Field(
        ...,
        description="Total steam production (lb/hr)"
    )
    condensate_return_lb_hr: float = Field(
        ...,
        description="Condensate return flow (lb/hr)"
    )
    return_rate_pct: float = Field(
        ...,
        description="Return rate percentage"
    )
    target_return_rate_pct: float = Field(
        default=85.0,
        description="Target return rate (%)"
    )

    # Temperature analysis
    avg_return_temperature_f: float = Field(
        ...,
        description="Average return temperature (F)"
    )
    target_return_temperature_f: float = Field(
        default=180.0,
        description="Target return temperature (F)"
    )
    temperature_shortfall_f: Optional[float] = Field(
        default=None,
        description="Temperature below target (F)"
    )

    # Energy analysis
    heat_recovered_btu_hr: float = Field(
        ...,
        description="Heat recovered from condensate (BTU/hr)"
    )
    potential_additional_recovery_btu_hr: float = Field(
        default=0.0,
        description="Additional recoverable heat (BTU/hr)"
    )
    makeup_water_required_lb_hr: float = Field(
        ...,
        description="Makeup water requirement (lb/hr)"
    )

    # Cost analysis
    fuel_savings_usd_hr: Optional[float] = Field(
        default=None,
        description="Fuel savings from return ($/hr)"
    )
    potential_additional_savings_usd_hr: Optional[float] = Field(
        default=None,
        description="Potential additional savings ($/hr)"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Improvement recommendations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 calculation hash"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# FLASH STEAM MODELS
# =============================================================================

class FlashSteamInput(BaseModel):
    """Input for flash steam recovery calculation."""

    condensate_flow_lb_hr: float = Field(
        ...,
        gt=0,
        description="Condensate flow rate (lb/hr)"
    )
    condensate_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Condensate pressure (psig)"
    )
    condensate_temperature_f: Optional[float] = Field(
        default=None,
        description="Condensate temperature (F) - defaults to saturation"
    )
    flash_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Flash vessel pressure (psig)"
    )

    @validator('flash_pressure_psig')
    def flash_less_than_condensate(cls, v, values):
        """Validate pressure relationship."""
        if 'condensate_pressure_psig' in values:
            if v >= values['condensate_pressure_psig']:
                raise ValueError(
                    "Flash pressure must be < condensate pressure"
                )
        return v


class FlashSteamOutput(BaseModel):
    """Output from flash steam recovery calculation."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )

    # Flash calculation results
    flash_fraction_pct: float = Field(
        ...,
        description="Flash steam fraction (%)"
    )
    flash_steam_lb_hr: float = Field(
        ...,
        description="Flash steam generated (lb/hr)"
    )
    residual_condensate_lb_hr: float = Field(
        ...,
        description="Residual condensate (lb/hr)"
    )

    # Thermodynamic details
    condensate_enthalpy_in_btu_lb: float = Field(
        ...,
        description="Inlet condensate enthalpy (BTU/lb)"
    )
    flash_steam_enthalpy_btu_lb: float = Field(
        ...,
        description="Flash steam enthalpy (BTU/lb)"
    )
    residual_enthalpy_btu_lb: float = Field(
        ...,
        description="Residual condensate enthalpy (BTU/lb)"
    )

    # Energy recovery
    energy_recovered_btu_hr: float = Field(
        ...,
        description="Energy recovered as flash steam (BTU/hr)"
    )
    recovery_efficiency_pct: float = Field(
        ...,
        description="Flash recovery efficiency (%)"
    )

    # Economics
    annual_savings_usd: Optional[float] = Field(
        default=None,
        description="Annual savings from flash recovery ($)"
    )

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 calculation hash"
    )
    formula_reference: str = Field(
        default="Thermodynamic flash calculation",
        description="Calculation method reference"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# STEAM TRAP MODELS
# =============================================================================

class SteamTrapReading(BaseModel):
    """Steam trap survey reading."""

    trap_id: str = Field(..., description="Trap identifier")
    location: str = Field(..., description="Physical location")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Survey timestamp"
    )

    # Trap information
    trap_type: str = Field(
        default="thermodynamic",
        description="Trap type"
    )
    size_inches: float = Field(
        default=0.75,
        description="Trap size (inches)"
    )
    design_capacity_lb_hr: float = Field(
        default=0.0,
        description="Design capacity (lb/hr)"
    )

    # Operating conditions
    inlet_pressure_psig: float = Field(
        ...,
        description="Inlet pressure (psig)"
    )
    differential_pressure_psi: Optional[float] = Field(
        default=None,
        description="Differential pressure (psi)"
    )

    # Status
    status: TrapStatus = Field(
        ...,
        description="Trap operational status"
    )
    temperature_f: Optional[float] = Field(
        default=None,
        description="Measured temperature (F)"
    )
    steam_loss_lb_hr: Optional[float] = Field(
        default=None,
        description="Estimated steam loss if failed (lb/hr)"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrapSurveyAnalysis(BaseModel):
    """Steam trap survey analysis results."""

    survey_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Survey date"
    )

    # Survey statistics
    total_traps: int = Field(..., description="Total traps surveyed")
    operating_count: int = Field(
        default=0,
        description="Operating correctly"
    )
    failed_open_count: int = Field(
        default=0,
        description="Failed open (leaking)"
    )
    failed_closed_count: int = Field(
        default=0,
        description="Failed closed (blocked)"
    )
    unknown_count: int = Field(
        default=0,
        description="Unknown status"
    )

    # Failure rates
    failure_rate_pct: float = Field(
        ...,
        description="Overall failure rate (%)"
    )
    failed_open_rate_pct: float = Field(
        default=0.0,
        description="Failed-open rate (%)"
    )

    # Loss estimates
    total_steam_loss_lb_hr: float = Field(
        default=0.0,
        description="Total steam loss (lb/hr)"
    )
    annual_steam_loss_mlb: float = Field(
        default=0.0,
        description="Annual steam loss (Mlb/year)"
    )
    annual_cost_usd: float = Field(
        default=0.0,
        description="Annual loss cost ($)"
    )

    # Recommendations
    priority_repairs: List[str] = Field(
        default_factory=list,
        description="Priority trap repairs"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="General recommendations"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 analysis hash"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# OPTIMIZATION RESULT MODELS
# =============================================================================

class OptimizationRecommendation(BaseModel):
    """Single optimization recommendation."""

    category: str = Field(
        ...,
        description="Recommendation category"
    )
    priority: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Priority (1=highest, 5=lowest)"
    )
    description: str = Field(
        ...,
        description="Recommendation description"
    )
    action: str = Field(
        ...,
        description="Recommended action"
    )

    # Impact estimates
    energy_savings_pct: Optional[float] = Field(
        default=None,
        description="Estimated energy savings (%)"
    )
    cost_savings_usd_year: Optional[float] = Field(
        default=None,
        description="Estimated annual savings ($)"
    )
    implementation_cost_usd: Optional[float] = Field(
        default=None,
        description="Implementation cost ($)"
    )
    payback_months: Optional[float] = Field(
        default=None,
        description="Simple payback (months)"
    )

    # Implementation
    complexity: str = Field(
        default="medium",
        description="Implementation complexity"
    )
    requires_shutdown: bool = Field(
        default=False,
        description="Requires system shutdown"
    )


class UnifiedSteamOptimizerOutput(BaseModel):
    """Complete output from GL-003 Unified Steam System Optimizer."""

    # Identification
    optimizer_id: str = Field(
        default="GL-003-UNIFIED",
        description="Optimizer identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Overall status
    overall_status: OptimizationStatus = Field(
        ...,
        description="Overall system status"
    )
    system_efficiency_pct: float = Field(
        ...,
        description="Overall system efficiency (%)"
    )
    exergy_efficiency_pct: Optional[float] = Field(
        default=None,
        description="System exergy efficiency (%)"
    )

    # Header balances
    header_analyses: List[HeaderBalanceOutput] = Field(
        default_factory=list,
        description="Header balance analyses"
    )

    # Quality analysis
    quality_analyses: List[SteamQualityAnalysis] = Field(
        default_factory=list,
        description="Steam quality analyses"
    )

    # PRV analysis
    prv_analyses: List[PRVSizingOutput] = Field(
        default_factory=list,
        description="PRV performance analyses"
    )

    # Condensate analysis
    condensate_analysis: Optional[CondensateReturnAnalysis] = Field(
        default=None,
        description="Condensate return analysis"
    )

    # Flash steam analysis
    flash_analyses: List[FlashSteamOutput] = Field(
        default_factory=list,
        description="Flash steam analyses"
    )

    # Trap survey
    trap_analysis: Optional[TrapSurveyAnalysis] = Field(
        default=None,
        description="Steam trap analysis"
    )

    # Recommendations
    recommendations: List[OptimizationRecommendation] = Field(
        default_factory=list,
        description="Optimization recommendations"
    )

    # Warnings and alerts
    warnings: List[str] = Field(
        default_factory=list,
        description="System warnings"
    )
    alerts: List[str] = Field(
        default_factory=list,
        description="Critical alerts"
    )

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for complete audit trail"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Total processing time (ms)"
    )
    calculation_count: int = Field(
        default=0,
        description="Number of calculations performed"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
