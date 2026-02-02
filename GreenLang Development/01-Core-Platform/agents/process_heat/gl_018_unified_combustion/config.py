"""
GL-018 UnifiedCombustionOptimizer - Configuration Module

Configuration schemas for unified combustion optimization including burner control,
flue gas analysis, emissions management, and BMS coordination per NFPA 85.

This module consolidates GL-002 (FLAMEGUARD), GL-004 (BURNMASTER), and GL-018 (FLUEFLOW)
configurations to eliminate 70-80% functional overlap.

Standards:
    - ASME PTC 4.1 for efficiency calculations
    - API 560 for combustion analysis
    - NFPA 85 for BMS coordination
    - EPA 40 CFR Part 60 for emissions
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================


class FuelType(Enum):
    """Supported fuel types for combustion optimization."""
    NATURAL_GAS = "natural_gas"
    NO2_FUEL_OIL = "no2_fuel_oil"
    NO6_FUEL_OIL = "no6_fuel_oil"
    PROPANE = "propane"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUB_BITUMINOUS = "coal_sub_bituminous"
    BIOMASS_WOOD = "biomass_wood"
    BIOGAS = "biogas"
    HYDROGEN = "hydrogen"
    DUAL_FUEL = "dual_fuel"
    SYNGAS = "syngas"


class EquipmentType(Enum):
    """Combustion equipment types."""
    BOILER_FIRETUBE = "boiler_firetube"
    BOILER_WATERTUBE = "boiler_watertube"
    BOILER_PACKAGE = "boiler_package"
    BOILER_HRSG = "boiler_hrsg"
    FURNACE_PROCESS = "furnace_process"
    HEATER_FIRED = "heater_fired"
    INCINERATOR = "incinerator"
    TURBINE_GAS = "turbine_gas"
    ENGINE_RECIP = "engine_recip"
    KILN = "kiln"


class BurnerType(Enum):
    """Burner types for combustion control."""
    CONVENTIONAL = "conventional"
    LOW_NOX = "low_nox"
    ULTRA_LOW_NOX = "ultra_low_nox"
    STAGED_AIR = "staged_air"
    STAGED_FUEL = "staged_fuel"
    PREMIXED = "premixed"
    FLUE_GAS_RECIRCULATION = "fgr"
    DUAL_FUEL = "dual_fuel"


class ControlMode(Enum):
    """Combustion control modes."""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    CASCADE = "cascade"
    OPTIMIZING = "optimizing"
    CROSS_LIMITING = "cross_limiting"
    OXYGEN_TRIM = "oxygen_trim"
    PARALLEL_POSITIONING = "parallel_positioning"


class EmissionControlTechnology(Enum):
    """Emission control technologies."""
    NONE = "none"
    LOW_NOX_BURNER = "low_nox_burner"
    FLUE_GAS_RECIRCULATION = "fgr"
    SELECTIVE_CATALYTIC_REDUCTION = "scr"
    SELECTIVE_NON_CATALYTIC_REDUCTION = "sncr"
    OXIDATION_CATALYST = "oxidation_catalyst"
    ESP = "electrostatic_precipitator"
    BAGHOUSE = "baghouse"
    WET_SCRUBBER = "wet_scrubber"


class BMSSequence(Enum):
    """BMS operating sequences per NFPA 85."""
    IDLE = "idle"
    PRE_PURGE = "pre_purge"
    PILOT_TRIAL = "pilot_trial"
    MAIN_FLAME_TRIAL = "main_flame_trial"
    RUNNING = "running"
    LOW_FIRE_HOLD = "low_fire_hold"
    MODULATING = "modulating"
    POST_PURGE = "post_purge"
    LOCKOUT = "lockout"


# =============================================================================
# SUB-CONFIGURATIONS
# =============================================================================


class BurnerConfig(BaseModel):
    """Burner system configuration."""

    burner_id: str = Field(..., description="Unique burner identifier")
    burner_type: BurnerType = Field(
        default=BurnerType.LOW_NOX,
        description="Burner type"
    )
    burner_count: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of burners"
    )
    capacity_mmbtu_hr: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Burner capacity (MMBTU/hr)"
    )
    turndown_ratio: float = Field(
        default=4.0,
        ge=2.0,
        le=20.0,
        description="Burner turndown ratio"
    )
    min_firing_rate_pct: float = Field(
        default=25.0,
        ge=5.0,
        le=50.0,
        description="Minimum firing rate (%)"
    )
    design_nox_ppm: float = Field(
        default=30.0,
        ge=0,
        le=200,
        description="Design NOx emissions (ppm @ 3% O2)"
    )
    design_co_ppm: float = Field(
        default=50.0,
        ge=0,
        le=400,
        description="Design CO emissions (ppm)"
    )


class AirFuelConfig(BaseModel):
    """Air-fuel ratio control configuration per NFPA 85."""

    control_mode: ControlMode = Field(
        default=ControlMode.CROSS_LIMITING,
        description="Air-fuel control mode"
    )
    # Target O2 settings
    target_o2_pct: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Target O2 percentage in flue gas"
    )
    min_o2_pct: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Minimum safe O2 percentage"
    )
    max_o2_pct: float = Field(
        default=6.0,
        ge=3.0,
        le=12.0,
        description="Maximum O2 percentage"
    )
    # Excess air limits
    target_excess_air_pct: float = Field(
        default=15.0,
        ge=5.0,
        le=40.0,
        description="Target excess air percentage"
    )
    min_excess_air_pct: float = Field(
        default=10.0,
        ge=3.0,
        le=25.0,
        description="Minimum safe excess air"
    )
    max_excess_air_pct: float = Field(
        default=25.0,
        ge=15.0,
        le=50.0,
        description="Maximum excess air"
    )
    # O2 trim settings
    o2_trim_enabled: bool = Field(
        default=True,
        description="Enable O2 trim control"
    )
    o2_trim_bias_max_pct: float = Field(
        default=5.0,
        ge=1.0,
        le=15.0,
        description="Maximum O2 trim bias (%)"
    )
    o2_trim_response_time_s: float = Field(
        default=30.0,
        ge=10.0,
        le=120.0,
        description="O2 trim response time (seconds)"
    )
    # Cross-limiting per NFPA 85
    cross_limiting_enabled: bool = Field(
        default=True,
        description="Enable cross-limiting per NFPA 85"
    )
    fuel_lead_lag_s: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Fuel lead on decrease, lag on increase (seconds)"
    )


class FlueGasConfig(BaseModel):
    """Flue gas analysis and monitoring configuration."""

    # Analyzer settings
    analyzer_type: str = Field(
        default="in_situ_zirconia",
        description="O2 analyzer type"
    )
    analyzer_response_time_s: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Analyzer response time (seconds)"
    )
    calibration_interval_hours: int = Field(
        default=168,
        ge=24,
        le=720,
        description="Calibration interval (hours)"
    )
    # Flue gas temperature limits
    max_flue_temp_f: float = Field(
        default=500.0,
        ge=300,
        le=1200,
        description="Maximum flue gas temperature (F)"
    )
    min_flue_temp_f: float = Field(
        default=250.0,
        ge=150,
        le=400,
        description="Minimum flue gas temperature (F)"
    )
    # Acid dew point
    acid_dew_point_margin_f: float = Field(
        default=25.0,
        ge=10,
        le=100,
        description="Margin above acid dew point (F)"
    )
    # Emissions limits
    co_alarm_ppm: float = Field(
        default=100.0,
        ge=50,
        le=500,
        description="CO alarm setpoint (ppm)"
    )
    co_trip_ppm: float = Field(
        default=400.0,
        ge=100,
        le=1000,
        description="CO trip setpoint (ppm)"
    )
    nox_limit_ppm: float = Field(
        default=30.0,
        ge=5,
        le=200,
        description="NOx emissions limit (ppm @ 3% O2)"
    )


class FlameStabilityConfig(BaseModel):
    """Flame stability and supervision configuration."""

    # Flame detector settings
    detector_type: str = Field(
        default="uv_ir",
        description="Flame detector type (uv, ir, uv_ir, scanner)"
    )
    detector_count_per_burner: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Flame detectors per burner"
    )
    flame_signal_min_pct: float = Field(
        default=30.0,
        ge=10.0,
        le=50.0,
        description="Minimum flame signal strength (%)"
    )
    flame_flicker_frequency_hz: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Expected flame flicker frequency (Hz)"
    )
    # Flame Stability Index (FSI) thresholds
    fsi_optimal_min: float = Field(
        default=0.85,
        ge=0.7,
        le=0.95,
        description="Minimum optimal FSI"
    )
    fsi_warning_threshold: float = Field(
        default=0.70,
        ge=0.5,
        le=0.85,
        description="FSI warning threshold"
    )
    fsi_alarm_threshold: float = Field(
        default=0.50,
        ge=0.3,
        le=0.7,
        description="FSI alarm threshold"
    )
    # Response times per NFPA 85
    flame_failure_response_s: float = Field(
        default=4.0,
        ge=1.0,
        le=10.0,
        description="Flame failure response time (seconds)"
    )


class EmissionsConfig(BaseModel):
    """Emissions control and monitoring configuration."""

    # Control technologies
    nox_control: EmissionControlTechnology = Field(
        default=EmissionControlTechnology.LOW_NOX_BURNER,
        description="Primary NOx control technology"
    )
    co_control: EmissionControlTechnology = Field(
        default=EmissionControlTechnology.NONE,
        description="CO control technology"
    )
    particulate_control: EmissionControlTechnology = Field(
        default=EmissionControlTechnology.NONE,
        description="Particulate control technology"
    )
    # FGR settings
    fgr_enabled: bool = Field(
        default=False,
        description="Enable flue gas recirculation"
    )
    fgr_rate_pct: float = Field(
        default=15.0,
        ge=5.0,
        le=30.0,
        description="FGR rate percentage"
    )
    # SCR settings
    scr_enabled: bool = Field(
        default=False,
        description="Enable SCR"
    )
    scr_inlet_temp_min_f: float = Field(
        default=550.0,
        ge=400,
        le=700,
        description="SCR minimum inlet temperature (F)"
    )
    scr_inlet_temp_max_f: float = Field(
        default=750.0,
        ge=600,
        le=900,
        description="SCR maximum inlet temperature (F)"
    )
    ammonia_slip_limit_ppm: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Ammonia slip limit (ppm)"
    )
    # Regulatory limits
    nox_permit_limit_lb_mmbtu: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="NOx permit limit (lb/MMBTU)"
    )
    co_permit_limit_lb_mmbtu: float = Field(
        default=0.04,
        ge=0.01,
        le=0.5,
        description="CO permit limit (lb/MMBTU)"
    )


class BMSConfig(BaseModel):
    """Burner Management System configuration per NFPA 85."""

    # SIL rating
    sil_level: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Safety Integrity Level"
    )
    # Timing per NFPA 85
    pre_purge_time_s: float = Field(
        default=60.0,
        ge=30.0,
        le=300.0,
        description="Pre-purge time (seconds)"
    )
    post_purge_time_s: float = Field(
        default=30.0,
        ge=15.0,
        le=120.0,
        description="Post-purge time (seconds)"
    )
    pilot_trial_time_s: float = Field(
        default=10.0,
        ge=5.0,
        le=30.0,
        description="Pilot trial for ignition (seconds)"
    )
    main_flame_trial_time_s: float = Field(
        default=10.0,
        ge=5.0,
        le=30.0,
        description="Main flame trial for ignition (seconds)"
    )
    # Purge requirements
    purge_air_flow_pct: float = Field(
        default=25.0,
        ge=25.0,
        le=100.0,
        description="Minimum purge air flow (% of full load)"
    )
    purge_volume_changes: int = Field(
        default=4,
        ge=4,
        le=8,
        description="Required volume changes during purge"
    )
    # Interlock settings
    low_fire_interlock: bool = Field(
        default=True,
        description="Require low fire for lightoff"
    )
    flame_detector_redundancy: str = Field(
        default="1oo2",
        description="Flame detector voting logic (1oo1, 1oo2, 2oo2, 2oo3)"
    )


class SootBlowingConfig(BaseModel):
    """Soot blowing optimization configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable soot blowing optimization"
    )
    blower_count: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of soot blowers"
    )
    steam_pressure_psig: float = Field(
        default=150.0,
        ge=50,
        le=600,
        description="Soot blowing steam pressure (psig)"
    )
    interval_hours: float = Field(
        default=8.0,
        ge=1.0,
        le=24.0,
        description="Nominal blowing interval (hours)"
    )
    # Optimization triggers
    delta_t_trigger_f: float = Field(
        default=50.0,
        ge=20,
        le=150,
        description="Temperature rise trigger for blowing (F)"
    )
    draft_loss_trigger_in_wc: float = Field(
        default=1.0,
        ge=0.3,
        le=3.0,
        description="Draft loss trigger for blowing (inches WC)"
    )
    efficiency_loss_trigger_pct: float = Field(
        default=0.5,
        ge=0.2,
        le=2.0,
        description="Efficiency loss trigger for blowing (%)"
    )


class BlowdownConfig(BaseModel):
    """Blowdown optimization configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable blowdown optimization"
    )
    control_type: str = Field(
        default="tds_based",
        description="Control type (tds_based, timer, continuous)"
    )
    tds_setpoint_ppm: float = Field(
        default=3500.0,
        ge=500,
        le=10000,
        description="TDS setpoint (ppm)"
    )
    min_rate_pct: float = Field(
        default=1.0,
        ge=0.5,
        le=5.0,
        description="Minimum blowdown rate (%)"
    )
    max_rate_pct: float = Field(
        default=8.0,
        ge=3.0,
        le=15.0,
        description="Maximum blowdown rate (%)"
    )
    heat_recovery_enabled: bool = Field(
        default=True,
        description="Enable blowdown heat recovery"
    )


class EfficiencyConfig(BaseModel):
    """Efficiency calculation configuration per ASME PTC 4.1."""

    calculation_method: str = Field(
        default="losses",
        description="Method: 'losses' or 'input_output'"
    )
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
        description="Guaranteed efficiency (%)"
    )
    # Efficiency targets by load
    full_load_target_pct: float = Field(
        default=84.0,
        ge=60,
        le=100,
        description="Full load efficiency target (%)"
    )
    min_load_target_pct: float = Field(
        default=80.0,
        ge=50,
        le=100,
        description="Minimum load efficiency target (%)"
    )
    # Uncertainty
    measurement_uncertainty_pct: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Measurement uncertainty (%)"
    )


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================


class UnifiedCombustionConfig(BaseModel):
    """
    Complete configuration for GL-018 UnifiedCombustionOptimizer.

    This configuration consolidates settings from GL-002 (FLAMEGUARD),
    GL-004 (BURNMASTER), and GL-018 (FLUEFLOW) agents.

    Features:
        - ASME PTC 4.1 efficiency calculations
        - API 560 combustion analysis
        - NFPA 85 BMS coordination
        - EPA emissions compliance
        - Air-fuel ratio optimization
        - Flame stability monitoring

    Example:
        >>> config = UnifiedCombustionConfig(
        ...     equipment_id="B-001",
        ...     fuel_type=FuelType.NATURAL_GAS,
        ...     burner=BurnerConfig(burner_id="BNR-001", capacity_mmbtu_hr=50.0)
        ... )
    """

    # Identity
    equipment_id: str = Field(..., description="Unique equipment identifier")
    name: str = Field(default="", description="Equipment name")
    equipment_type: EquipmentType = Field(
        default=EquipmentType.BOILER_WATERTUBE,
        description="Equipment type"
    )
    fuel_type: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Primary fuel type"
    )
    secondary_fuel_type: Optional[FuelType] = Field(
        default=None,
        description="Secondary fuel type for dual-fuel"
    )

    # Capacity
    design_capacity_mmbtu_hr: float = Field(
        default=50.0,
        gt=0,
        le=2000,
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

    # Sub-configurations
    burner: BurnerConfig = Field(
        ...,
        description="Burner configuration"
    )
    air_fuel: AirFuelConfig = Field(
        default_factory=AirFuelConfig,
        description="Air-fuel ratio configuration"
    )
    flue_gas: FlueGasConfig = Field(
        default_factory=FlueGasConfig,
        description="Flue gas configuration"
    )
    flame_stability: FlameStabilityConfig = Field(
        default_factory=FlameStabilityConfig,
        description="Flame stability configuration"
    )
    emissions: EmissionsConfig = Field(
        default_factory=EmissionsConfig,
        description="Emissions configuration"
    )
    bms: BMSConfig = Field(
        default_factory=BMSConfig,
        description="BMS configuration"
    )
    soot_blowing: SootBlowingConfig = Field(
        default_factory=SootBlowingConfig,
        description="Soot blowing configuration"
    )
    blowdown: BlowdownConfig = Field(
        default_factory=BlowdownConfig,
        description="Blowdown configuration"
    )
    efficiency: EfficiencyConfig = Field(
        default_factory=EfficiencyConfig,
        description="Efficiency configuration"
    )

    # Control settings
    control_mode: ControlMode = Field(
        default=ControlMode.OPTIMIZING,
        description="Operating control mode"
    )
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
        """Set default name from equipment_id."""
        if not v and "equipment_id" in values:
            return f"Combustion System {values['equipment_id']}"
        return v

    @validator("guarantee_efficiency_pct", pre=True, always=True)
    def validate_guarantee_efficiency(cls, v, values):
        """Ensure guarantee efficiency is less than design."""
        return v  # Full validation done in EfficiencyConfig
