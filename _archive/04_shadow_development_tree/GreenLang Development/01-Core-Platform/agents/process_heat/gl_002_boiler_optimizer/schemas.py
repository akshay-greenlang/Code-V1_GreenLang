"""
GL-002 BoilerOptimizer Agent - Schema Definitions

Pydantic models for boiler optimizer inputs, outputs, and results.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field, validator


class OperatingStatus(Enum):
    """Boiler operating status."""
    OFFLINE = "offline"
    STANDBY = "standby"
    PURGING = "purging"
    IGNITION = "ignition"
    LOW_FIRE = "low_fire"
    MODULATING = "modulating"
    HIGH_FIRE = "high_fire"
    SHUTDOWN = "shutdown"
    TRIP = "trip"


class BoilerInput(BaseModel):
    """Input data for boiler optimization."""

    # Identity
    boiler_id: str = Field(..., description="Boiler identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )

    # Operating status
    operating_status: OperatingStatus = Field(
        default=OperatingStatus.MODULATING,
        description="Current operating status"
    )
    load_pct: float = Field(
        ...,
        ge=0,
        le=120,
        description="Current load percentage"
    )

    # Fuel
    fuel_type: str = Field(default="natural_gas", description="Fuel type")
    fuel_flow_rate: float = Field(
        ...,
        gt=0,
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
        description="Fuel HHV if known (BTU/lb or BTU/SCF)"
    )

    # Steam
    steam_flow_rate_lb_hr: float = Field(
        ...,
        gt=0,
        description="Steam flow rate (lb/hr)"
    )
    steam_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Steam pressure (psig)"
    )
    steam_temperature_f: Optional[float] = Field(
        default=None,
        description="Steam temperature (F)"
    )

    # Feedwater
    feedwater_flow_rate_lb_hr: float = Field(
        ...,
        gt=0,
        description="Feedwater flow rate (lb/hr)"
    )
    feedwater_temperature_f: float = Field(
        default=200.0,
        description="Feedwater temperature (F)"
    )
    feedwater_pressure_psig: Optional[float] = Field(
        default=None,
        description="Feedwater pressure (psig)"
    )

    # Combustion
    flue_gas_o2_pct: float = Field(
        ...,
        ge=0,
        le=21,
        description="Flue gas O2 (%)"
    )
    flue_gas_co_ppm: float = Field(
        default=0.0,
        ge=0,
        description="Flue gas CO (ppm)"
    )
    flue_gas_nox_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Flue gas NOx (ppm)"
    )
    flue_gas_temperature_f: float = Field(
        ...,
        description="Flue gas temperature (F)"
    )
    combustion_air_temperature_f: float = Field(
        default=77.0,
        description="Combustion air temperature (F)"
    )

    # Blowdown
    blowdown_rate_pct: float = Field(
        default=2.0,
        ge=0,
        le=20,
        description="Blowdown rate (%)"
    )
    blowdown_tds_ppm: Optional[float] = Field(
        default=None,
        description="Blowdown TDS (ppm)"
    )

    # Drum
    drum_level_in: Optional[float] = Field(
        default=None,
        ge=-12,
        le=12,
        description="Drum level (inches)"
    )
    drum_pressure_psig: Optional[float] = Field(
        default=None,
        description="Drum pressure (psig)"
    )

    # Economizer
    economizer_inlet_temp_f: Optional[float] = Field(
        default=None,
        description="Economizer flue gas inlet temp (F)"
    )
    economizer_outlet_temp_f: Optional[float] = Field(
        default=None,
        description="Economizer flue gas outlet temp (F)"
    )
    economizer_water_inlet_temp_f: Optional[float] = Field(
        default=None,
        description="Economizer water inlet temp (F)"
    )
    economizer_water_outlet_temp_f: Optional[float] = Field(
        default=None,
        description="Economizer water outlet temp (F)"
    )

    # Ambient
    ambient_temperature_f: float = Field(
        default=77.0,
        description="Ambient temperature (F)"
    )
    ambient_humidity_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Ambient humidity (%)"
    )
    barometric_pressure_psia: float = Field(
        default=14.696,
        description="Barometric pressure (psia)"
    )

    class Config:
        use_enum_values = True


class EfficiencyResult(BaseModel):
    """Boiler efficiency calculation result."""

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

    # Loss breakdown
    dry_flue_gas_loss_pct: float = Field(
        ...,
        description="Dry flue gas loss (%)"
    )
    moisture_in_fuel_loss_pct: float = Field(
        default=0.0,
        description="Moisture in fuel loss (%)"
    )
    moisture_from_h2_loss_pct: float = Field(
        default=0.0,
        description="Moisture from H2 combustion loss (%)"
    )
    radiation_loss_pct: float = Field(
        ...,
        description="Radiation and convection loss (%)"
    )
    blowdown_loss_pct: float = Field(
        default=0.0,
        description="Blowdown loss (%)"
    )
    unburned_loss_pct: float = Field(
        default=0.0,
        description="Unburned combustibles loss (%)"
    )
    other_losses_pct: float = Field(
        default=0.0,
        description="Other losses (%)"
    )
    total_losses_pct: float = Field(
        ...,
        description="Total losses (%)"
    )

    # Supporting data
    excess_air_pct: float = Field(..., description="Excess air (%)")
    heat_input_btu_hr: float = Field(..., description="Heat input (BTU/hr)")
    heat_output_btu_hr: float = Field(..., description="Heat output (BTU/hr)")

    # Metadata
    calculation_method: str = Field(
        default="ASME_PTC_4.1_LOSSES",
        description="Calculation method used"
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


class OptimizationRecommendation(BaseModel):
    """Optimization recommendation."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Recommendation identifier"
    )
    category: str = Field(
        ...,
        description="Category (combustion, steam, economizer, maintenance)"
    )
    priority: str = Field(
        default="medium",
        description="Priority (low, medium, high, critical)"
    )
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    current_value: Optional[float] = Field(
        default=None,
        description="Current value"
    )
    recommended_value: Optional[float] = Field(
        default=None,
        description="Recommended value"
    )
    estimated_savings_pct: Optional[float] = Field(
        default=None,
        description="Estimated efficiency improvement (%)"
    )
    estimated_savings_btu_hr: Optional[float] = Field(
        default=None,
        description="Estimated savings (BTU/hr)"
    )
    estimated_annual_savings_usd: Optional[float] = Field(
        default=None,
        description="Estimated annual savings ($)"
    )
    implementation_difficulty: str = Field(
        default="low",
        description="Implementation difficulty (low, medium, high)"
    )
    requires_shutdown: bool = Field(
        default=False,
        description="Requires boiler shutdown"
    )


class BoilerOutput(BaseModel):
    """Complete output from boiler optimization."""

    # Identity
    boiler_id: str = Field(..., description="Boiler identifier")
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

    # Efficiency
    efficiency: EfficiencyResult = Field(
        ...,
        description="Efficiency calculation results"
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

    # Alerts
    alerts: List[Dict[str, Any]] = Field(
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

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class CombustionAnalysis(BaseModel):
    """Combustion analysis results."""

    excess_air_pct: float = Field(..., description="Excess air (%)")
    air_fuel_ratio: float = Field(..., description="Air-fuel ratio")
    stoichiometric_air: float = Field(
        ...,
        description="Stoichiometric air requirement"
    )
    combustion_efficiency_pct: float = Field(
        ...,
        description="Combustion efficiency (%)"
    )
    stack_loss_pct: float = Field(..., description="Stack loss (%)")
    co_loss_pct: float = Field(default=0.0, description="CO loss (%)")
    optimal_o2_pct: float = Field(..., description="Optimal O2 setpoint (%)")
    o2_deviation_pct: float = Field(
        ...,
        description="Deviation from optimal O2 (%)"
    )

    # Recommendations
    adjust_air_fuel: bool = Field(
        default=False,
        description="Air-fuel adjustment needed"
    )
    adjustment_direction: Optional[str] = Field(
        default=None,
        description="Adjustment direction (increase_air, decrease_air)"
    )
    estimated_improvement_pct: Optional[float] = Field(
        default=None,
        description="Estimated improvement from adjustment"
    )


class SteamSystemAnalysis(BaseModel):
    """Steam system analysis results."""

    # Steam quality
    steam_quality_pct: Optional[float] = Field(
        default=None,
        description="Steam quality (%)"
    )
    superheat_f: Optional[float] = Field(
        default=None,
        description="Degrees of superheat (F)"
    )

    # Mass balance
    steam_to_feedwater_ratio: float = Field(
        ...,
        description="Steam to feedwater ratio"
    )
    blowdown_rate_actual_pct: float = Field(
        ...,
        description="Actual blowdown rate (%)"
    )
    makeup_rate_pct: float = Field(..., description="Makeup water rate (%)")

    # Energy
    steam_enthalpy_btu_lb: float = Field(
        ...,
        description="Steam enthalpy (BTU/lb)"
    )
    feedwater_enthalpy_btu_lb: float = Field(
        ...,
        description="Feedwater enthalpy (BTU/lb)"
    )
    heat_added_btu_lb: float = Field(
        ...,
        description="Heat added per lb steam (BTU/lb)"
    )

    # Drum level
    drum_level_status: str = Field(
        default="normal",
        description="Drum level status"
    )
    drum_level_deviation_in: Optional[float] = Field(
        default=None,
        description="Drum level deviation (inches)"
    )


class EconomizerAnalysis(BaseModel):
    """Economizer analysis results."""

    enabled: bool = Field(..., description="Economizer enabled")
    duty_btu_hr: float = Field(default=0.0, description="Current duty (BTU/hr)")
    effectiveness: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Current effectiveness"
    )
    design_effectiveness: float = Field(
        default=0.7,
        description="Design effectiveness"
    )
    fouling_factor: Optional[float] = Field(
        default=None,
        description="Estimated fouling factor"
    )
    water_temp_rise_f: float = Field(
        default=0.0,
        description="Water temperature rise (F)"
    )
    flue_gas_temp_drop_f: float = Field(
        default=0.0,
        description="Flue gas temperature drop (F)"
    )
    acid_dew_point_margin_f: Optional[float] = Field(
        default=None,
        description="Margin above acid dew point (F)"
    )
    cleaning_recommended: bool = Field(
        default=False,
        description="Cleaning recommended"
    )
