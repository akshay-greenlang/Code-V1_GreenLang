"""
GL-002 BoilerOptimizer Agent - Configuration Module

Configuration schemas for boiler optimization including combustion,
steam system, and economizer settings.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator


class FuelType(Enum):
    """Supported fuel types."""
    NATURAL_GAS = "natural_gas"
    NO2_FUEL_OIL = "no2_fuel_oil"
    NO6_FUEL_OIL = "no6_fuel_oil"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUB_BITUMINOUS = "coal_sub_bituminous"
    BIOMASS_WOOD = "biomass_wood"
    DUAL_FUEL = "dual_fuel"


class BoilerType(Enum):
    """Boiler types."""
    FIRETUBE = "firetube"
    WATERTUBE = "watertube"
    PACKAGE = "package"
    FIELD_ERECTED = "field_erected"
    HRSG = "hrsg"
    CFB = "cfb"


class ControlMode(Enum):
    """Control modes."""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    CASCADE = "cascade"
    OPTIMIZING = "optimizing"


class CombustionConfig(BaseModel):
    """Combustion system configuration."""

    # Burner settings
    burner_count: int = Field(default=1, ge=1, le=20, description="Number of burners")
    burner_type: str = Field(
        default="low_nox",
        description="Burner type (conventional, low_nox, ultra_low_nox)"
    )
    turndown_ratio: float = Field(
        default=4.0,
        ge=2.0,
        le=10.0,
        description="Burner turndown ratio"
    )

    # Air-fuel ratio targets
    target_excess_air_pct: float = Field(
        default=15.0,
        ge=5.0,
        le=50.0,
        description="Target excess air percentage"
    )
    min_excess_air_pct: float = Field(
        default=10.0,
        ge=3.0,
        le=30.0,
        description="Minimum safe excess air"
    )
    max_excess_air_pct: float = Field(
        default=25.0,
        ge=10.0,
        le=50.0,
        description="Maximum excess air"
    )

    # Flue gas limits
    max_co_ppm: float = Field(default=100.0, ge=0, description="Maximum CO (ppm)")
    max_nox_ppm: float = Field(default=30.0, ge=0, description="Maximum NOx (ppm)")
    max_flue_temp_f: float = Field(
        default=500.0,
        description="Maximum flue gas temperature (F)"
    )

    # Air preheater
    air_preheater_enabled: bool = Field(
        default=False,
        description="Enable air preheater"
    )
    air_preheater_effectiveness: float = Field(
        default=0.7,
        ge=0.3,
        le=0.95,
        description="Air preheater effectiveness"
    )

    # O2 trim control
    o2_trim_enabled: bool = Field(default=True, description="Enable O2 trim control")
    o2_setpoint_pct: float = Field(
        default=3.0,
        ge=1.0,
        le=8.0,
        description="O2 setpoint for trim control"
    )


class SteamConfig(BaseModel):
    """Steam system configuration."""

    # Steam conditions
    design_pressure_psig: float = Field(
        default=150.0,
        ge=0,
        le=2500,
        description="Design steam pressure (psig)"
    )
    design_temperature_f: Optional[float] = Field(
        default=None,
        description="Design temperature for superheated steam (F)"
    )
    steam_flow_capacity_lb_hr: float = Field(
        default=50000.0,
        gt=0,
        description="Steam flow capacity (lb/hr)"
    )

    # Feedwater
    feedwater_temperature_f: float = Field(
        default=227.0,
        ge=100,
        le=500,
        description="Feedwater temperature (F)"
    )
    makeup_water_pct: float = Field(
        default=5.0,
        ge=0,
        le=50,
        description="Makeup water percentage"
    )

    # Blowdown
    tds_limit_ppm: float = Field(
        default=3500.0,
        ge=500,
        le=10000,
        description="TDS limit for blowdown control (ppm)"
    )
    blowdown_rate_pct: float = Field(
        default=3.0,
        ge=0,
        le=20,
        description="Continuous blowdown rate (%)"
    )
    blowdown_heat_recovery: bool = Field(
        default=True,
        description="Enable blowdown heat recovery"
    )

    # Drum level control
    drum_level_setpoint_in: float = Field(
        default=0.0,
        ge=-10,
        le=10,
        description="Drum level setpoint (inches from center)"
    )
    drum_level_control_type: str = Field(
        default="three_element",
        description="Control type (single_element, two_element, three_element)"
    )

    # Deaerator
    deaerator_pressure_psig: float = Field(
        default=5.0,
        ge=0,
        le=30,
        description="Deaerator pressure (psig)"
    )
    deaerator_o2_limit_ppb: float = Field(
        default=7.0,
        ge=0,
        le=100,
        description="Deaerator dissolved O2 limit (ppb)"
    )


class EconomizerConfig(BaseModel):
    """Economizer configuration."""

    enabled: bool = Field(default=True, description="Enable economizer")
    design_duty_btu_hr: float = Field(
        default=5000000.0,
        gt=0,
        description="Design duty (BTU/hr)"
    )
    design_effectiveness: float = Field(
        default=0.7,
        ge=0.3,
        le=0.95,
        description="Design effectiveness"
    )
    tube_material: str = Field(
        default="carbon_steel",
        description="Tube material"
    )
    min_outlet_temp_f: float = Field(
        default=250.0,
        ge=150,
        le=400,
        description="Minimum outlet temperature to prevent condensation (F)"
    )
    acid_dew_point_margin_f: float = Field(
        default=25.0,
        ge=10,
        le=50,
        description="Margin above acid dew point (F)"
    )
    cleaning_interval_hours: int = Field(
        default=2000,
        ge=100,
        le=8760,
        description="Recommended cleaning interval (hours)"
    )


class SafetyConfig(BaseModel):
    """Safety configuration for boiler operation."""

    # Safety system
    sil_level: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Safety Integrity Level"
    )
    esd_integration: bool = Field(default=True, description="ESD integration")

    # Flame supervision
    flame_detector_type: str = Field(
        default="uv_ir",
        description="Flame detector type (uv, ir, uv_ir)"
    )
    flame_failure_response_s: float = Field(
        default=4.0,
        ge=1.0,
        le=10.0,
        description="Flame failure response time (s)"
    )

    # Pressure limits
    low_water_trip_psig: float = Field(
        default=10.0,
        ge=0,
        description="Low water cutoff pressure (psig)"
    )
    high_pressure_trip_psig: float = Field(
        default=175.0,
        gt=0,
        description="High pressure trip (psig)"
    )

    # Interlock delays
    purge_time_s: float = Field(
        default=60.0,
        ge=30,
        le=300,
        description="Pre-purge time (s)"
    )
    post_purge_time_s: float = Field(
        default=30.0,
        ge=15,
        le=120,
        description="Post-purge time (s)"
    )


class BoilerConfig(BaseModel):
    """
    Complete boiler optimizer configuration.

    This configuration defines all parameters for the GL-002
    BoilerOptimizer Agent including combustion, steam, economizer,
    and safety settings.
    """

    # Identity
    boiler_id: str = Field(..., description="Unique boiler identifier")
    name: str = Field(default="", description="Boiler name")
    boiler_type: BoilerType = Field(
        default=BoilerType.WATERTUBE,
        description="Boiler type"
    )
    fuel_type: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Primary fuel type"
    )

    # Capacity
    design_capacity_mmbtu_hr: float = Field(
        default=50.0,
        gt=0,
        description="Design capacity (MMBTU/hr)"
    )
    min_load_pct: float = Field(
        default=25.0,
        ge=0,
        le=100,
        description="Minimum load percentage"
    )
    max_load_pct: float = Field(
        default=110.0,
        ge=100,
        le=120,
        description="Maximum load percentage"
    )

    # Design efficiency
    design_efficiency_pct: float = Field(
        default=82.0,
        ge=50,
        le=100,
        description="Design efficiency (%)"
    )
    guarantee_efficiency_pct: float = Field(
        default=80.0,
        ge=50,
        le=100,
        description="Guarantee efficiency (%)"
    )

    # Sub-configurations
    combustion: CombustionConfig = Field(
        default_factory=CombustionConfig,
        description="Combustion configuration"
    )
    steam: SteamConfig = Field(
        default_factory=SteamConfig,
        description="Steam system configuration"
    )
    economizer: EconomizerConfig = Field(
        default_factory=EconomizerConfig,
        description="Economizer configuration"
    )
    safety: SafetyConfig = Field(
        default_factory=SafetyConfig,
        description="Safety configuration"
    )

    # Control
    control_mode: ControlMode = Field(
        default=ControlMode.OPTIMIZING,
        description="Operating control mode"
    )
    optimization_enabled: bool = Field(
        default=True,
        description="Enable automatic optimization"
    )
    optimization_interval_s: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Optimization interval (seconds)"
    )

    # Data collection
    historian_tag_prefix: str = Field(
        default="",
        description="Historian tag prefix"
    )
    data_collection_interval_s: int = Field(
        default=1,
        ge=1,
        le=60,
        description="Data collection interval (seconds)"
    )

    class Config:
        use_enum_values = True

    @validator("name", always=True)
    def set_default_name(cls, v, values):
        """Set default name from boiler_id."""
        if not v and "boiler_id" in values:
            return f"Boiler {values['boiler_id']}"
        return v

    @validator("guarantee_efficiency_pct")
    def validate_guarantee_efficiency(cls, v, values):
        """Ensure guarantee efficiency is less than design."""
        if "design_efficiency_pct" in values:
            if v > values["design_efficiency_pct"]:
                raise ValueError(
                    "Guarantee efficiency cannot exceed design efficiency"
                )
        return v
