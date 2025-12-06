"""
GL-020 ECONOPULSE Agent - Configuration Module

Configuration schemas for economizer optimization including gas-side fouling,
water-side scaling, soot blower optimization, acid dew point, and steaming
economizer detection settings.

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code (applicable methods)
    - ASME PTC 4.1 Steam Generating Units
    - API 560 Fired Heaters for General Refinery Service
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator


class EconomizerType(Enum):
    """Economizer design types."""
    BARE_TUBE = "bare_tube"
    FINNED_TUBE = "finned_tube"
    EXTENDED_SURFACE = "extended_surface"
    CAST_IRON = "cast_iron"
    CONDENSING = "condensing"
    NON_CONDENSING = "non_condensing"


class EconomizerArrangement(Enum):
    """Economizer flow arrangement."""
    COUNTERFLOW = "counterflow"
    PARALLEL_FLOW = "parallel_flow"
    CROSSFLOW = "crossflow"
    CROSSFLOW_MIXED = "crossflow_mixed"


class TubeMaterial(Enum):
    """Economizer tube materials."""
    CARBON_STEEL = "carbon_steel"
    LOW_ALLOY_STEEL = "low_alloy_steel"
    STAINLESS_304 = "stainless_304"
    STAINLESS_316 = "stainless_316"
    CORTEN = "corten"
    CAST_IRON = "cast_iron"


class FuelType(Enum):
    """Fuel types for acid dew point calculations."""
    NATURAL_GAS = "natural_gas"
    NO2_FUEL_OIL = "no2_fuel_oil"
    NO6_FUEL_OIL = "no6_fuel_oil"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUB_BITUMINOUS = "coal_sub_bituminous"
    BIOMASS = "biomass"
    REFINERY_GAS = "refinery_gas"


class SootBlowerType(Enum):
    """Soot blower types."""
    ROTARY = "rotary"
    RETRACTABLE = "retractable"
    STATIONARY = "stationary"
    ACOUSTIC = "acoustic"
    STEAM = "steam"
    AIR = "air"


class GasSideFoulingConfig(BaseModel):
    """Gas-side fouling detection configuration."""

    # Design values
    design_gas_dp_in_wc: float = Field(
        default=2.0,
        gt=0,
        le=10,
        description="Design gas-side pressure drop (in. WC)"
    )
    design_gas_velocity_fps: float = Field(
        default=50.0,
        gt=0,
        le=100,
        description="Design gas velocity (ft/s)"
    )
    design_heat_transfer_coeff: float = Field(
        default=10.0,
        gt=0,
        description="Design gas-side heat transfer coefficient (BTU/hr-ft2-F)"
    )

    # Fouling thresholds
    dp_warning_ratio: float = Field(
        default=1.3,
        ge=1.1,
        le=2.0,
        description="Pressure drop warning ratio (actual/design)"
    )
    dp_alarm_ratio: float = Field(
        default=1.5,
        ge=1.2,
        le=2.5,
        description="Pressure drop alarm ratio"
    )
    dp_cleaning_trigger_ratio: float = Field(
        default=1.7,
        ge=1.3,
        le=3.0,
        description="Pressure drop cleaning trigger ratio"
    )

    # Heat transfer degradation
    u_degradation_warning_pct: float = Field(
        default=10.0,
        ge=5,
        le=25,
        description="U-value degradation warning (%)"
    )
    u_degradation_alarm_pct: float = Field(
        default=20.0,
        ge=10,
        le=40,
        description="U-value degradation alarm (%)"
    )

    # Trend analysis
    trend_analysis_hours: int = Field(
        default=168,
        ge=24,
        le=720,
        description="Hours of data for trend analysis"
    )
    trend_threshold_pct_per_day: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Fouling rate threshold (% per day)"
    )


class WaterSideFoulingConfig(BaseModel):
    """Water-side fouling/scaling detection configuration."""

    # Design values
    design_water_dp_psi: float = Field(
        default=5.0,
        gt=0,
        le=30,
        description="Design water-side pressure drop (psi)"
    )
    design_water_velocity_fps: float = Field(
        default=6.0,
        gt=0,
        le=15,
        description="Design water velocity (ft/s)"
    )
    design_fouling_factor: float = Field(
        default=0.001,
        ge=0,
        le=0.01,
        description="Design fouling factor (hr-ft2-F/BTU)"
    )

    # Scaling thresholds
    dp_warning_ratio: float = Field(
        default=1.2,
        ge=1.1,
        le=1.5,
        description="Pressure drop warning ratio"
    )
    dp_alarm_ratio: float = Field(
        default=1.4,
        ge=1.2,
        le=2.0,
        description="Pressure drop alarm ratio"
    )

    # Water chemistry limits
    max_hardness_ppm: float = Field(
        default=0.5,
        ge=0,
        le=5,
        description="Maximum feedwater hardness (ppm as CaCO3)"
    )
    max_silica_ppm: float = Field(
        default=0.02,
        ge=0,
        le=0.5,
        description="Maximum feedwater silica (ppm)"
    )
    max_iron_ppm: float = Field(
        default=0.01,
        ge=0,
        le=0.1,
        description="Maximum feedwater iron (ppm)"
    )
    max_copper_ppm: float = Field(
        default=0.005,
        ge=0,
        le=0.05,
        description="Maximum feedwater copper (ppm)"
    )
    target_ph: float = Field(
        default=9.2,
        ge=8.5,
        le=10.5,
        description="Target feedwater pH"
    )
    ph_tolerance: float = Field(
        default=0.3,
        ge=0.1,
        le=0.5,
        description="pH tolerance band"
    )

    # Internal inspection
    inspection_interval_months: int = Field(
        default=24,
        ge=6,
        le=60,
        description="Internal inspection interval (months)"
    )


class SootBlowerConfig(BaseModel):
    """Soot blower optimization configuration."""

    # Soot blower inventory
    num_soot_blowers: int = Field(
        default=4,
        ge=0,
        le=20,
        description="Number of soot blowers"
    )
    blower_type: SootBlowerType = Field(
        default=SootBlowerType.ROTARY,
        description="Soot blower type"
    )

    # Steam consumption
    steam_pressure_psig: float = Field(
        default=200.0,
        ge=50,
        le=600,
        description="Soot blowing steam pressure (psig)"
    )
    steam_flow_per_blower_lb: float = Field(
        default=500.0,
        gt=0,
        le=2000,
        description="Steam consumption per blower cycle (lb)"
    )
    blowing_duration_s: int = Field(
        default=90,
        ge=30,
        le=300,
        description="Blowing duration per cycle (seconds)"
    )

    # Scheduling
    fixed_schedule_enabled: bool = Field(
        default=False,
        description="Use fixed schedule vs. intelligent"
    )
    fixed_interval_hours: float = Field(
        default=8.0,
        ge=1,
        le=24,
        description="Fixed schedule interval (hours)"
    )
    min_interval_hours: float = Field(
        default=2.0,
        ge=0.5,
        le=8,
        description="Minimum interval between blowing (hours)"
    )
    max_interval_hours: float = Field(
        default=12.0,
        ge=4,
        le=48,
        description="Maximum interval between blowing (hours)"
    )

    # Intelligent triggers
    dp_trigger_ratio: float = Field(
        default=1.2,
        ge=1.05,
        le=1.5,
        description="Pressure drop ratio to trigger blowing"
    )
    u_degradation_trigger_pct: float = Field(
        default=5.0,
        ge=2,
        le=15,
        description="U-value degradation to trigger blowing (%)"
    )
    exit_temp_rise_trigger_f: float = Field(
        default=20.0,
        ge=5,
        le=50,
        description="Exit temp rise to trigger blowing (F)"
    )

    # Optimization targets
    target_steam_savings_pct: float = Field(
        default=20.0,
        ge=0,
        le=50,
        description="Target steam savings vs. fixed schedule (%)"
    )
    effectiveness_threshold: float = Field(
        default=0.85,
        ge=0.7,
        le=0.95,
        description="Minimum effectiveness threshold"
    )

    class Config:
        use_enum_values = True


class AcidDewPointConfig(BaseModel):
    """Acid dew point calculation configuration."""

    # Fuel sulfur content
    fuel_type: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Primary fuel type"
    )
    fuel_sulfur_pct: float = Field(
        default=0.001,
        ge=0,
        le=5,
        description="Fuel sulfur content (%)"
    )
    so3_conversion_pct: float = Field(
        default=2.0,
        ge=0.5,
        le=10,
        description="SO2 to SO3 conversion (%)"
    )

    # Moisture content
    flue_gas_moisture_pct: float = Field(
        default=10.0,
        ge=0,
        le=25,
        description="Flue gas moisture content (%)"
    )

    # Safety margins
    acid_dew_point_margin_f: float = Field(
        default=30.0,
        ge=10,
        le=75,
        description="Safety margin above acid dew point (F)"
    )
    min_metal_temp_f: float = Field(
        default=270.0,
        ge=200,
        le=400,
        description="Minimum cold-end metal temperature (F)"
    )

    # Corrosion monitoring
    corrosion_probe_enabled: bool = Field(
        default=False,
        description="Corrosion probe monitoring enabled"
    )
    corrosion_rate_warning_mpy: float = Field(
        default=5.0,
        ge=1,
        le=20,
        description="Corrosion rate warning (mils per year)"
    )
    corrosion_rate_alarm_mpy: float = Field(
        default=10.0,
        ge=5,
        le=50,
        description="Corrosion rate alarm (mils per year)"
    )

    # Temperature monitoring
    cold_end_temp_points: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of cold-end temperature measurement points"
    )

    class Config:
        use_enum_values = True


class EffectivenessConfig(BaseModel):
    """Heat transfer effectiveness configuration."""

    # Design effectiveness
    design_effectiveness: float = Field(
        default=0.80,
        ge=0.5,
        le=0.95,
        description="Design heat transfer effectiveness"
    )
    design_ntu: float = Field(
        default=2.0,
        gt=0,
        le=10,
        description="Design Number of Transfer Units (NTU)"
    )

    # Thresholds
    effectiveness_warning_pct: float = Field(
        default=90.0,
        ge=70,
        le=98,
        description="Effectiveness warning threshold (% of design)"
    )
    effectiveness_alarm_pct: float = Field(
        default=80.0,
        ge=60,
        le=95,
        description="Effectiveness alarm threshold (% of design)"
    )

    # Calculation method
    calculation_method: str = Field(
        default="ntu_epsilon",
        description="Calculation method (ntu_epsilon, lmtd, uvalue)"
    )
    include_radiation_correction: bool = Field(
        default=True,
        description="Include radiation heat transfer correction"
    )

    # Reference conditions
    reference_gas_flow_lb_hr: float = Field(
        default=100000.0,
        gt=0,
        description="Reference gas flow rate (lb/hr)"
    )
    reference_water_flow_lb_hr: float = Field(
        default=80000.0,
        gt=0,
        description="Reference water flow rate (lb/hr)"
    )


class SteamingConfig(BaseModel):
    """Steaming economizer detection configuration."""

    # Design conditions
    design_approach_temp_f: float = Field(
        default=30.0,
        ge=10,
        le=100,
        description="Design approach to saturation (F)"
    )
    design_subcooling_f: float = Field(
        default=20.0,
        ge=5,
        le=50,
        description="Design subcooling at outlet (F)"
    )
    design_outlet_pressure_psig: float = Field(
        default=500.0,
        ge=100,
        le=3000,
        description="Design outlet pressure (psig)"
    )

    # Steaming thresholds
    approach_warning_f: float = Field(
        default=15.0,
        ge=5,
        le=30,
        description="Approach temperature warning (F)"
    )
    approach_alarm_f: float = Field(
        default=10.0,
        ge=2,
        le=20,
        description="Approach temperature alarm (F)"
    )
    approach_trip_f: float = Field(
        default=5.0,
        ge=0,
        le=10,
        description="Approach temperature trip (F)"
    )

    # Steaming detection
    steaming_detection_enabled: bool = Field(
        default=True,
        description="Enable steaming detection"
    )
    dp_fluctuation_threshold_pct: float = Field(
        default=10.0,
        ge=5,
        le=30,
        description="DP fluctuation threshold for steaming (%)"
    )
    temp_fluctuation_threshold_f: float = Field(
        default=5.0,
        ge=2,
        le=20,
        description="Temperature fluctuation threshold (F)"
    )

    # Load-based limits
    steaming_risk_load_pct: float = Field(
        default=30.0,
        ge=10,
        le=50,
        description="Load below which steaming risk increases (%)"
    )
    min_water_flow_pct: float = Field(
        default=25.0,
        ge=10,
        le=50,
        description="Minimum water flow (% of design)"
    )

    # Recirculation
    recirculation_enabled: bool = Field(
        default=False,
        description="Economizer recirculation available"
    )
    recirculation_trigger_approach_f: float = Field(
        default=12.0,
        ge=5,
        le=25,
        description="Approach temp to trigger recirculation (F)"
    )


class EconomizerDesignConfig(BaseModel):
    """Economizer design specifications."""

    # Physical design
    economizer_type: EconomizerType = Field(
        default=EconomizerType.FINNED_TUBE,
        description="Economizer type"
    )
    arrangement: EconomizerArrangement = Field(
        default=EconomizerArrangement.COUNTERFLOW,
        description="Flow arrangement"
    )
    tube_material: TubeMaterial = Field(
        default=TubeMaterial.CARBON_STEEL,
        description="Tube material"
    )

    # Surface area
    total_surface_area_ft2: float = Field(
        default=5000.0,
        gt=0,
        description="Total heat transfer surface area (ft2)"
    )
    bare_tube_area_ft2: float = Field(
        default=1000.0,
        gt=0,
        description="Bare tube surface area (ft2)"
    )
    extended_surface_ratio: float = Field(
        default=5.0,
        ge=1,
        le=15,
        description="Extended surface to bare tube ratio"
    )

    # Tube specifications
    tube_od_in: float = Field(
        default=2.0,
        ge=0.5,
        le=4,
        description="Tube OD (inches)"
    )
    tube_wall_thickness_in: float = Field(
        default=0.12,
        ge=0.05,
        le=0.5,
        description="Tube wall thickness (inches)"
    )
    num_tubes: int = Field(
        default=200,
        gt=0,
        description="Number of tubes"
    )
    tube_length_ft: float = Field(
        default=15.0,
        gt=0,
        le=50,
        description="Tube length (ft)"
    )
    num_passes: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Number of water passes"
    )

    # Fin specifications (if finned)
    fin_height_in: float = Field(
        default=0.75,
        ge=0,
        le=2,
        description="Fin height (inches)"
    )
    fin_pitch_per_in: float = Field(
        default=5.0,
        ge=0,
        le=12,
        description="Fin pitch (fins per inch)"
    )
    fin_thickness_in: float = Field(
        default=0.05,
        ge=0,
        le=0.25,
        description="Fin thickness (inches)"
    )

    class Config:
        use_enum_values = True


class PerformanceBaselineConfig(BaseModel):
    """Performance baseline configuration."""

    # Design point
    design_duty_btu_hr: float = Field(
        default=20_000_000.0,
        gt=0,
        description="Design heat duty (BTU/hr)"
    )
    design_gas_inlet_temp_f: float = Field(
        default=600.0,
        ge=200,
        le=1200,
        description="Design gas inlet temperature (F)"
    )
    design_gas_outlet_temp_f: float = Field(
        default=350.0,
        ge=150,
        le=600,
        description="Design gas outlet temperature (F)"
    )
    design_water_inlet_temp_f: float = Field(
        default=250.0,
        ge=100,
        le=500,
        description="Design water inlet temperature (F)"
    )
    design_water_outlet_temp_f: float = Field(
        default=350.0,
        ge=150,
        le=600,
        description="Design water outlet temperature (F)"
    )

    # Flow rates
    design_gas_flow_lb_hr: float = Field(
        default=100000.0,
        gt=0,
        description="Design gas flow rate (lb/hr)"
    )
    design_water_flow_lb_hr: float = Field(
        default=80000.0,
        gt=0,
        description="Design water flow rate (lb/hr)"
    )

    # Pressure drops
    design_gas_dp_in_wc: float = Field(
        default=2.0,
        gt=0,
        le=10,
        description="Design gas-side pressure drop (in. WC)"
    )
    design_water_dp_psi: float = Field(
        default=5.0,
        gt=0,
        le=30,
        description="Design water-side pressure drop (psi)"
    )

    # Overall coefficient
    design_ua_btu_hr_f: float = Field(
        default=100000.0,
        gt=0,
        description="Design UA value (BTU/hr-F)"
    )
    clean_ua_btu_hr_f: float = Field(
        default=120000.0,
        gt=0,
        description="Clean condition UA value (BTU/hr-F)"
    )


class EconomizerOptimizationConfig(BaseModel):
    """
    Complete economizer optimization configuration.

    This configuration defines all parameters for the GL-020
    ECONOPULSE Agent including gas-side fouling, water-side scaling,
    soot blower optimization, acid dew point, effectiveness,
    and steaming economizer detection settings.

    Standards Reference:
        - ASME PTC 4.3 Air Heater Test Code
        - ASME PTC 4.1 Steam Generating Units
    """

    # Identity
    economizer_id: str = Field(..., description="Unique economizer identifier")
    name: str = Field(default="", description="Economizer name")
    boiler_id: str = Field(default="", description="Associated boiler ID")

    # Design specifications
    design: EconomizerDesignConfig = Field(
        default_factory=EconomizerDesignConfig,
        description="Economizer design configuration"
    )

    # Performance baseline
    baseline: PerformanceBaselineConfig = Field(
        default_factory=PerformanceBaselineConfig,
        description="Performance baseline configuration"
    )

    # Sub-configurations
    gas_side: GasSideFoulingConfig = Field(
        default_factory=GasSideFoulingConfig,
        description="Gas-side fouling configuration"
    )
    water_side: WaterSideFoulingConfig = Field(
        default_factory=WaterSideFoulingConfig,
        description="Water-side fouling configuration"
    )
    soot_blower: SootBlowerConfig = Field(
        default_factory=SootBlowerConfig,
        description="Soot blower configuration"
    )
    acid_dew_point: AcidDewPointConfig = Field(
        default_factory=AcidDewPointConfig,
        description="Acid dew point configuration"
    )
    effectiveness: EffectivenessConfig = Field(
        default_factory=EffectivenessConfig,
        description="Effectiveness configuration"
    )
    steaming: SteamingConfig = Field(
        default_factory=SteamingConfig,
        description="Steaming detection configuration"
    )

    # Control settings
    optimization_enabled: bool = Field(
        default=True,
        description="Enable automatic optimization"
    )
    optimization_interval_s: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Optimization interval (seconds)"
    )

    # Safety settings
    sil_level: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Safety Integrity Level"
    )
    high_water_temp_trip_f: float = Field(
        default=450.0,
        ge=300,
        le=600,
        description="High water temperature trip (F)"
    )
    low_water_flow_trip_pct: float = Field(
        default=20.0,
        ge=10,
        le=50,
        description="Low water flow trip (% of design)"
    )
    high_gas_temp_alarm_f: float = Field(
        default=700.0,
        ge=400,
        le=1000,
        description="High gas inlet temperature alarm (F)"
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
        """Set default name from economizer_id."""
        if not v and "economizer_id" in values:
            return f"Economizer {values['economizer_id']}"
        return v
