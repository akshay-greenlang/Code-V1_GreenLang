"""
GL-002 FLAMEGUARD BoilerEfficiencyOptimizer - Configuration Module

This module defines all configuration schemas for the FLAMEGUARD
BoilerEfficiencyOptimizer including combustion, safety, optimization,
fuel management, emissions, and operational settings.

All configurations use Pydantic for validation with comprehensive defaults
and documentation following ASME PTC 4.1 and NFPA 85 standards.

Standards Compliance:
    - ASME PTC 4.1 (Fired Steam Generators Performance Test Codes)
    - NFPA 85 (Boiler and Combustion Systems Hazards Code)
    - IEC 61511 (Functional Safety - Safety Instrumented Systems)
    - EPA 40 CFR Part 60/63 (Emissions Standards)
    - ISO 50001 (Energy Management Systems)
"""

from datetime import timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
import os

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# ENUMS
# =============================================================================

class BoilerType(Enum):
    """Types of industrial boilers."""
    WATER_TUBE = "water_tube"
    FIRE_TUBE = "fire_tube"
    PACKAGE = "package"
    FLUIDIZED_BED = "fluidized_bed"
    RECOVERY = "recovery"
    HRSG = "hrsg"  # Heat Recovery Steam Generator
    CFBC = "cfbc"  # Circulating Fluidized Bed Combustion


class FuelType(Enum):
    """Types of boiler fuels."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUB_BITUMINOUS = "coal_sub_bituminous"
    COAL_LIGNITE = "coal_lignite"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_AGRICULTURAL = "biomass_agricultural"
    BIOGAS = "biogas"
    HYDROGEN = "hydrogen"
    LPG = "lpg"
    MIXED = "mixed"


class CombustionMode(Enum):
    """Combustion control modes."""
    MANUAL = "manual"
    PARALLEL = "parallel"  # Fuel and air in parallel
    CROSS_LIMITED = "cross_limited"  # Cross-limiting metered control
    O2_TRIM = "o2_trim"
    FULL_METERING = "full_metering"
    AI_OPTIMIZED = "ai_optimized"


class SafetyIntegrityLevel(Enum):
    """IEC 61511 Safety Integrity Levels."""
    NONE = 0
    SIL_1 = 1
    SIL_2 = 2
    SIL_3 = 3
    SIL_4 = 4


class OperatingState(Enum):
    """Boiler operating states per NFPA 85."""
    OFFLINE = "offline"
    COLD_STANDBY = "cold_standby"
    HOT_STANDBY = "hot_standby"
    PURGING = "purging"
    PILOT_LIGHT = "pilot_light"
    LOW_FIRE = "low_fire"
    MODULATING = "modulating"
    HIGH_FIRE = "high_fire"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


class OptimizationObjective(Enum):
    """Optimization objectives."""
    EFFICIENCY = "efficiency"
    EMISSIONS = "emissions"
    COST = "cost"
    COMBINED = "combined"
    LOAD_FOLLOWING = "load_following"


class EmissionStandard(Enum):
    """Applicable emission standards."""
    EPA_MACT = "epa_mact"
    EPA_NSPS = "epa_nsps"
    EPA_NESHAP = "epa_neshap"
    EU_IED = "eu_ied"
    EU_MCP = "eu_mcp"
    CARB = "carb"
    STATE_SPECIFIC = "state_specific"


class ControlMode(Enum):
    """Control loop operating modes."""
    MANUAL = "manual"
    AUTO = "auto"
    CASCADE = "cascade"
    OVERRIDE = "override"


# =============================================================================
# FUEL CONFIGURATION
# =============================================================================

class FuelProperties(BaseModel):
    """Physical and chemical properties of fuel."""

    fuel_type: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Type of fuel"
    )
    higher_heating_value_btu_lb: float = Field(
        default=23875.0,
        ge=1000.0,
        le=60000.0,
        description="Higher Heating Value (HHV) in BTU/lb"
    )
    lower_heating_value_btu_lb: float = Field(
        default=21500.0,
        ge=900.0,
        le=55000.0,
        description="Lower Heating Value (LHV) in BTU/lb"
    )
    carbon_content_percent: float = Field(
        default=75.0,
        ge=0.0,
        le=100.0,
        description="Carbon content by mass percentage"
    )
    hydrogen_content_percent: float = Field(
        default=25.0,
        ge=0.0,
        le=100.0,
        description="Hydrogen content by mass percentage"
    )
    sulfur_content_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Sulfur content by mass percentage"
    )
    nitrogen_content_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Nitrogen content by mass percentage"
    )
    ash_content_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Ash content by mass percentage"
    )
    moisture_content_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=60.0,
        description="Moisture content by mass percentage"
    )
    stoichiometric_air_fuel_ratio: float = Field(
        default=17.2,
        ge=1.0,
        le=50.0,
        description="Stoichiometric air-to-fuel ratio by mass"
    )
    co2_emission_factor_kg_mmbtu: float = Field(
        default=53.06,
        ge=0.0,
        le=120.0,
        description="CO2 emission factor in kg/MMBTU (EPA default for natural gas)"
    )
    density_lb_ft3: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=100.0,
        description="Fuel density at standard conditions (lb/ft³)"
    )


class FuelConfig(BaseModel):
    """Fuel system configuration."""

    primary_fuel: FuelProperties = Field(
        default_factory=FuelProperties,
        description="Primary fuel properties"
    )
    backup_fuel: Optional[FuelProperties] = Field(
        default=None,
        description="Backup fuel properties"
    )
    fuel_blending_enabled: bool = Field(
        default=False,
        description="Enable multi-fuel blending"
    )
    blend_fuels: List[FuelProperties] = Field(
        default_factory=list,
        description="Available fuels for blending"
    )
    max_blend_ratio: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum blend ratio for secondary fuels"
    )
    fuel_pressure_psig: float = Field(
        default=15.0,
        ge=0.1,
        le=500.0,
        description="Fuel supply pressure (psig)"
    )
    fuel_temperature_f: float = Field(
        default=70.0,
        ge=32.0,
        le=500.0,
        description="Fuel supply temperature (°F)"
    )
    fuel_flow_meter_accuracy: float = Field(
        default=0.5,
        ge=0.1,
        le=5.0,
        description="Fuel flow meter accuracy (%)"
    )


# =============================================================================
# COMBUSTION CONFIGURATION
# =============================================================================

class O2TrimConfig(BaseModel):
    """Oxygen trim control configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable O2 trim control"
    )
    target_o2_percent: float = Field(
        default=3.0,
        ge=0.5,
        le=15.0,
        description="Target O2 percentage in flue gas"
    )
    o2_setpoint_curve: Dict[float, float] = Field(
        default_factory=lambda: {
            0.25: 5.0,   # 25% load -> 5.0% O2
            0.50: 3.5,   # 50% load -> 3.5% O2
            0.75: 3.0,   # 75% load -> 3.0% O2
            1.00: 2.5,   # 100% load -> 2.5% O2
        },
        description="O2 setpoint vs load curve (load fraction -> O2%)"
    )
    min_o2_percent: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Minimum allowable O2 percentage"
    )
    max_o2_percent: float = Field(
        default=8.0,
        ge=3.0,
        le=15.0,
        description="Maximum allowable O2 percentage"
    )
    proportional_gain: float = Field(
        default=2.0,
        ge=0.1,
        le=20.0,
        description="O2 trim proportional gain"
    )
    integral_time_s: float = Field(
        default=120.0,
        ge=10.0,
        le=600.0,
        description="O2 trim integral time (seconds)"
    )
    derivative_time_s: float = Field(
        default=0.0,
        ge=0.0,
        le=60.0,
        description="O2 trim derivative time (seconds)"
    )
    output_limit_percent: float = Field(
        default=10.0,
        ge=1.0,
        le=30.0,
        description="Maximum O2 trim output adjustment (%)"
    )
    deadband_percent: float = Field(
        default=0.2,
        ge=0.05,
        le=1.0,
        description="O2 control deadband (%)"
    )


class ExcessAirConfig(BaseModel):
    """Excess air control configuration."""

    design_excess_air_percent: float = Field(
        default=15.0,
        ge=5.0,
        le=100.0,
        description="Design excess air percentage"
    )
    min_excess_air_percent: float = Field(
        default=10.0,
        ge=3.0,
        le=50.0,
        description="Minimum safe excess air percentage"
    )
    max_excess_air_percent: float = Field(
        default=50.0,
        ge=15.0,
        le=200.0,
        description="Maximum allowable excess air percentage"
    )
    excess_air_curve: Dict[float, float] = Field(
        default_factory=lambda: {
            0.25: 40.0,  # 25% load -> 40% excess air
            0.50: 25.0,  # 50% load -> 25% excess air
            0.75: 18.0,  # 75% load -> 18% excess air
            1.00: 15.0,  # 100% load -> 15% excess air
        },
        description="Excess air vs load curve"
    )
    combustion_air_temp_compensation: bool = Field(
        default=True,
        description="Enable air temperature compensation"
    )
    air_temp_reference_f: float = Field(
        default=80.0,
        ge=32.0,
        le=150.0,
        description="Reference air temperature for compensation (°F)"
    )


class COMonitoringConfig(BaseModel):
    """Carbon monoxide monitoring configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable CO monitoring"
    )
    co_limit_ppm: float = Field(
        default=400.0,
        ge=50.0,
        le=2000.0,
        description="CO limit for combustion optimization (ppm)"
    )
    co_alarm_ppm: float = Field(
        default=600.0,
        ge=100.0,
        le=5000.0,
        description="CO alarm threshold (ppm)"
    )
    co_response_gain: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="CO response gain for excess air adjustment"
    )
    averaging_time_s: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="CO averaging time (seconds)"
    )
    breakthrough_threshold_ppm: float = Field(
        default=200.0,
        ge=50.0,
        le=500.0,
        description="CO breakthrough detection threshold (ppm)"
    )


class CombustionConfig(BaseModel):
    """Complete combustion system configuration."""

    control_mode: CombustionMode = Field(
        default=CombustionMode.O2_TRIM,
        description="Combustion control mode"
    )
    o2_trim: O2TrimConfig = Field(
        default_factory=O2TrimConfig,
        description="O2 trim configuration"
    )
    excess_air: ExcessAirConfig = Field(
        default_factory=ExcessAirConfig,
        description="Excess air configuration"
    )
    co_monitoring: COMonitoringConfig = Field(
        default_factory=COMonitoringConfig,
        description="CO monitoring configuration"
    )
    air_fuel_ratio_control: bool = Field(
        default=True,
        description="Enable air-fuel ratio control"
    )
    cross_limiting_enabled: bool = Field(
        default=True,
        description="Enable cross-limiting for fuel/air"
    )
    lead_lag_on_load_increase: str = Field(
        default="air_leads",
        description="Lead-lag strategy on load increase (air_leads/fuel_leads)"
    )
    lead_lag_on_load_decrease: str = Field(
        default="fuel_leads",
        description="Lead-lag strategy on load decrease"
    )
    min_firing_rate_percent: float = Field(
        default=25.0,
        ge=10.0,
        le=50.0,
        description="Minimum firing rate (%)"
    )
    max_firing_rate_percent: float = Field(
        default=100.0,
        ge=50.0,
        le=110.0,
        description="Maximum firing rate (%)"
    )
    load_ramp_rate_percent_min: float = Field(
        default=5.0,
        ge=0.5,
        le=20.0,
        description="Maximum load ramp rate (%/min)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# SAFETY CONFIGURATION
# =============================================================================

class FlameDetectionConfig(BaseModel):
    """Flame detection and monitoring configuration per NFPA 85."""

    scanner_type: str = Field(
        default="UV",
        description="Flame scanner type (UV, IR, combination)"
    )
    flame_failure_response_time_s: float = Field(
        default=4.0,
        ge=1.0,
        le=10.0,
        description="Flame failure response time per NFPA 85 (seconds)"
    )
    pilot_flame_proving_time_s: float = Field(
        default=10.0,
        ge=5.0,
        le=30.0,
        description="Pilot flame proving time (seconds)"
    )
    main_flame_proving_time_s: float = Field(
        default=10.0,
        ge=5.0,
        le=30.0,
        description="Main flame proving time (seconds)"
    )
    min_flame_signal_percent: float = Field(
        default=10.0,
        ge=5.0,
        le=30.0,
        description="Minimum flame signal for stable operation (%)"
    )
    flame_scanner_self_check_enabled: bool = Field(
        default=True,
        description="Enable flame scanner self-check"
    )
    redundant_scanners: bool = Field(
        default=True,
        description="Use redundant flame scanners"
    )
    scanner_voting: str = Field(
        default="2oo3",
        description="Scanner voting logic (1oo1, 1oo2, 2oo2, 2oo3)"
    )


class PurgeConfig(BaseModel):
    """Furnace purge configuration per NFPA 85."""

    pre_purge_time_s: float = Field(
        default=300.0,
        ge=60.0,
        le=900.0,
        description="Pre-purge time - minimum 5 air changes"
    )
    post_purge_time_s: float = Field(
        default=60.0,
        ge=30.0,
        le=300.0,
        description="Post-purge time"
    )
    purge_airflow_percent: float = Field(
        default=25.0,
        ge=15.0,
        le=50.0,
        description="Purge airflow rate as % of full load"
    )
    min_air_changes: int = Field(
        default=5,
        ge=4,
        le=10,
        description="Minimum furnace air changes during purge"
    )
    purge_interlock_enabled: bool = Field(
        default=True,
        description="Enable purge completion interlock"
    )
    forced_draft_fan_proof: bool = Field(
        default=True,
        description="Require FD fan proof during purge"
    )


class SafetyInterlockConfig(BaseModel):
    """Safety interlock configuration."""

    high_steam_pressure_psig: float = Field(
        default=150.0,
        ge=15.0,
        le=3000.0,
        description="High steam pressure trip setpoint (psig)"
    )
    low_water_level_inches: float = Field(
        default=2.0,
        ge=-10.0,
        le=20.0,
        description="Low water level trip setpoint (inches from normal)"
    )
    high_water_level_inches: float = Field(
        default=6.0,
        ge=0.0,
        le=20.0,
        description="High water level trip setpoint (inches from normal)"
    )
    low_fuel_pressure_psig: float = Field(
        default=2.0,
        ge=0.1,
        le=50.0,
        description="Low fuel pressure trip setpoint (psig)"
    )
    high_fuel_pressure_psig: float = Field(
        default=25.0,
        ge=5.0,
        le=500.0,
        description="High fuel pressure trip setpoint (psig)"
    )
    low_combustion_air_pressure_inwc: float = Field(
        default=0.5,
        ge=0.1,
        le=10.0,
        description="Low combustion air pressure trip (in WC)"
    )
    high_flue_gas_temp_f: float = Field(
        default=700.0,
        ge=300.0,
        le=1500.0,
        description="High flue gas temperature trip (°F)"
    )
    low_atomizing_media_pressure_psig: Optional[float] = Field(
        default=None,
        ge=10.0,
        le=200.0,
        description="Low atomizing steam/air pressure trip (psig) - for oil firing"
    )


class SafetyConfig(BaseModel):
    """Complete safety system configuration per NFPA 85 and IEC 61511."""

    sil_level: SafetyIntegrityLevel = Field(
        default=SafetyIntegrityLevel.SIL_2,
        description="Safety Integrity Level per IEC 61511"
    )
    emergency_shutdown_enabled: bool = Field(
        default=True,
        description="Enable Emergency Shutdown (ESD) system"
    )
    burner_management_system: bool = Field(
        default=True,
        description="Require Burner Management System (BMS)"
    )
    flame_detection: FlameDetectionConfig = Field(
        default_factory=FlameDetectionConfig,
        description="Flame detection configuration"
    )
    purge: PurgeConfig = Field(
        default_factory=PurgeConfig,
        description="Furnace purge configuration"
    )
    interlocks: SafetyInterlockConfig = Field(
        default_factory=SafetyInterlockConfig,
        description="Safety interlock setpoints"
    )
    watchdog_timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="Safety system watchdog timeout (ms)"
    )
    heartbeat_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Safety heartbeat interval (ms)"
    )
    max_consecutive_failures: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max consecutive failures before trip"
    )
    trip_reset_requires_authorization: bool = Field(
        default=True,
        description="Require authorization to reset trips"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# EFFICIENCY CONFIGURATION
# =============================================================================

class EfficiencyCalculationConfig(BaseModel):
    """Efficiency calculation configuration per ASME PTC 4.1."""

    calculation_method: str = Field(
        default="indirect_losses",
        description="Efficiency calculation method (direct/indirect_losses)"
    )
    include_blowdown_loss: bool = Field(
        default=True,
        description="Include blowdown heat loss"
    )
    include_radiation_loss: bool = Field(
        default=True,
        description="Include radiation and convection loss"
    )
    include_unburned_carbon_loss: bool = Field(
        default=True,
        description="Include unburned carbon loss (solid fuels)"
    )
    ambient_temperature_f: float = Field(
        default=77.0,
        ge=32.0,
        le=120.0,
        description="Reference ambient temperature (°F)"
    )
    ambient_humidity_percent: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Reference ambient humidity (%)"
    )
    design_efficiency_percent: float = Field(
        default=82.0,
        ge=60.0,
        le=98.0,
        description="Design boiler efficiency (%)"
    )
    min_acceptable_efficiency_percent: float = Field(
        default=75.0,
        ge=50.0,
        le=95.0,
        description="Minimum acceptable efficiency (%)"
    )
    efficiency_alarm_threshold_percent: float = Field(
        default=78.0,
        ge=50.0,
        le=95.0,
        description="Efficiency alarm threshold (%)"
    )


class HeatBalanceConfig(BaseModel):
    """Heat balance calculation configuration."""

    calculate_stack_loss: bool = Field(
        default=True,
        description="Calculate dry flue gas stack loss"
    )
    calculate_moisture_loss: bool = Field(
        default=True,
        description="Calculate moisture in fuel loss"
    )
    calculate_hydrogen_loss: bool = Field(
        default=True,
        description="Calculate hydrogen combustion moisture loss"
    )
    calculate_air_moisture_loss: bool = Field(
        default=True,
        description="Calculate combustion air moisture loss"
    )
    calculate_incomplete_combustion_loss: bool = Field(
        default=True,
        description="Calculate CO/unburned loss"
    )
    radiation_loss_percent: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Fixed radiation loss assumption (%)"
    )
    use_adiabatic_flame_temp: bool = Field(
        default=True,
        description="Calculate adiabatic flame temperature"
    )


# =============================================================================
# EMISSIONS CONFIGURATION
# =============================================================================

class EmissionsConfig(BaseModel):
    """Emissions monitoring and reporting configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable emissions monitoring"
    )
    applicable_standard: EmissionStandard = Field(
        default=EmissionStandard.EPA_MACT,
        description="Applicable emissions standard"
    )
    continuous_emissions_monitoring: bool = Field(
        default=True,
        description="Use Continuous Emissions Monitoring System (CEMS)"
    )
    nox_limit_lb_mmbtu: float = Field(
        default=0.10,
        ge=0.01,
        le=1.0,
        description="NOx emission limit (lb/MMBTU)"
    )
    co_limit_lb_mmbtu: float = Field(
        default=0.08,
        ge=0.01,
        le=0.5,
        description="CO emission limit (lb/MMBTU)"
    )
    so2_limit_lb_mmbtu: float = Field(
        default=0.50,
        ge=0.01,
        le=2.0,
        description="SO2 emission limit (lb/MMBTU)"
    )
    pm_limit_lb_mmbtu: float = Field(
        default=0.03,
        ge=0.001,
        le=0.5,
        description="Particulate matter limit (lb/MMBTU)"
    )
    voc_limit_lb_mmbtu: float = Field(
        default=0.005,
        ge=0.001,
        le=0.1,
        description="VOC emission limit (lb/MMBTU)"
    )
    ghg_reporting_enabled: bool = Field(
        default=True,
        description="Enable GHG Protocol reporting"
    )
    ghg_scope: int = Field(
        default=1,
        ge=1,
        le=3,
        description="GHG emissions scope (1=direct)"
    )
    stack_flow_rate_acfm: Optional[float] = Field(
        default=None,
        ge=1000.0,
        le=1000000.0,
        description="Design stack flow rate (ACFM)"
    )
    stack_temperature_f: Optional[float] = Field(
        default=None,
        ge=200.0,
        le=800.0,
        description="Design stack temperature (°F)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================

class AIOptimizationConfig(BaseModel):
    """AI/ML optimization configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable AI-based optimization"
    )
    model_type: str = Field(
        default="ensemble",
        description="ML model type (ensemble, neural_network, gaussian_process)"
    )
    optimization_objective: OptimizationObjective = Field(
        default=OptimizationObjective.COMBINED,
        description="Primary optimization objective"
    )
    efficiency_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for efficiency in combined objective"
    )
    emissions_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for emissions in combined objective"
    )
    cost_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for cost in combined objective"
    )
    model_update_interval_s: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Model update interval (seconds)"
    )
    exploration_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Exploration rate for reinforcement learning"
    )
    min_data_points_for_training: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Minimum data points for model training"
    )
    prediction_confidence_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=0.99,
        description="Minimum confidence for predictions"
    )
    fallback_to_rule_based: bool = Field(
        default=True,
        description="Fallback to rule-based on low confidence"
    )
    uncertainty_quantification: bool = Field(
        default=True,
        description="Enable uncertainty bounds"
    )
    explainability_enabled: bool = Field(
        default=True,
        description="Enable SHAP/LIME explanations"
    )

    class Config:
        use_enum_values = True

    @root_validator(skip_on_failure=True)
    def validate_weights(cls, values):
        """Validate optimization weights sum to 1.0."""
        eff = values.get('efficiency_weight', 0)
        emis = values.get('emissions_weight', 0)
        cost = values.get('cost_weight', 0)
        total = eff + emis + cost
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Optimization weights must sum to 1.0, got {total}")
        return values


class SetpointOptimizationConfig(BaseModel):
    """Setpoint optimization configuration."""

    optimize_o2_setpoint: bool = Field(
        default=True,
        description="Optimize O2 setpoint based on conditions"
    )
    optimize_steam_pressure_setpoint: bool = Field(
        default=False,
        description="Optimize steam pressure setpoint"
    )
    optimize_feedwater_temperature: bool = Field(
        default=False,
        description="Optimize feedwater temperature"
    )
    setpoint_update_interval_s: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Setpoint update interval (seconds)"
    )
    max_setpoint_change_per_update: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Maximum setpoint change per update (%)"
    )
    require_operator_approval: bool = Field(
        default=False,
        description="Require operator approval for setpoint changes"
    )


class OptimizationConfig(BaseModel):
    """Complete optimization configuration."""

    ai: AIOptimizationConfig = Field(
        default_factory=AIOptimizationConfig,
        description="AI/ML optimization settings"
    )
    setpoints: SetpointOptimizationConfig = Field(
        default_factory=SetpointOptimizationConfig,
        description="Setpoint optimization settings"
    )
    run_in_advisory_mode: bool = Field(
        default=True,
        description="Run in advisory mode (recommendations only)"
    )
    max_optimization_cycles_per_hour: int = Field(
        default=12,
        ge=1,
        le=60,
        description="Maximum optimization cycles per hour"
    )
    stability_window_s: int = Field(
        default=300,
        ge=60,
        le=1800,
        description="Required stable period before optimization (seconds)"
    )
    exclude_during_sootblowing: bool = Field(
        default=True,
        description="Exclude optimization during sootblowing"
    )
    exclude_during_load_changes: bool = Field(
        default=True,
        description="Exclude during rapid load changes"
    )
    load_change_threshold_percent_min: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Load change threshold (%/min)"
    )


# =============================================================================
# INTEGRATION CONFIGURATION
# =============================================================================

class SCADAConfig(BaseModel):
    """SCADA/DCS integration configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable SCADA integration"
    )
    protocol: str = Field(
        default="OPC-UA",
        description="Communication protocol (OPC-UA, Modbus-TCP, DNP3)"
    )
    server_endpoint: str = Field(
        default="opc.tcp://localhost:4840",
        description="SCADA server endpoint"
    )
    namespace_uri: str = Field(
        default="urn:greenlang:flameguard",
        description="OPC-UA namespace URI"
    )
    polling_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Data polling interval (ms)"
    )
    write_enabled: bool = Field(
        default=True,
        description="Enable write-back to SCADA"
    )
    write_requires_confirmation: bool = Field(
        default=True,
        description="Require confirmation for writes"
    )
    connection_timeout_s: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Connection timeout (seconds)"
    )
    retry_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Connection retry count"
    )
    security_policy: str = Field(
        default="Basic256Sha256",
        description="OPC-UA security policy"
    )


class IntegrationConfig(BaseModel):
    """External system integration configuration."""

    scada: SCADAConfig = Field(
        default_factory=SCADAConfig,
        description="SCADA/DCS integration"
    )
    historian_enabled: bool = Field(
        default=True,
        description="Enable process historian integration"
    )
    historian_url: Optional[str] = Field(
        default=None,
        description="Process historian URL"
    )
    erp_enabled: bool = Field(
        default=False,
        description="Enable ERP integration for cost data"
    )
    weather_integration: bool = Field(
        default=False,
        description="Enable weather data integration"
    )
    emissions_cems_integration: bool = Field(
        default=True,
        description="Enable CEMS data integration"
    )


# =============================================================================
# BOILER SPECIFICATIONS
# =============================================================================

class BoilerSpecifications(BaseModel):
    """Boiler equipment specifications."""

    boiler_id: str = Field(
        default="BOILER-001",
        description="Unique boiler identifier"
    )
    boiler_name: str = Field(
        default="Main Steam Boiler",
        description="Boiler name"
    )
    boiler_type: BoilerType = Field(
        default=BoilerType.WATER_TUBE,
        description="Boiler type"
    )
    manufacturer: str = Field(
        default="",
        description="Boiler manufacturer"
    )
    model: str = Field(
        default="",
        description="Boiler model"
    )
    year_installed: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Year installed"
    )
    rated_capacity_klb_hr: float = Field(
        default=100.0,
        ge=1.0,
        le=10000.0,
        description="Rated steam capacity (klb/hr)"
    )
    design_pressure_psig: float = Field(
        default=150.0,
        ge=15.0,
        le=5000.0,
        description="Design pressure (psig)"
    )
    operating_pressure_psig: float = Field(
        default=125.0,
        ge=15.0,
        le=4500.0,
        description="Normal operating pressure (psig)"
    )
    design_temperature_f: float = Field(
        default=366.0,
        ge=200.0,
        le=1200.0,
        description="Design steam temperature (°F)"
    )
    feedwater_temperature_f: float = Field(
        default=227.0,
        ge=60.0,
        le=500.0,
        description="Design feedwater temperature (°F)"
    )
    blowdown_rate_percent: float = Field(
        default=3.0,
        ge=0.5,
        le=10.0,
        description="Continuous blowdown rate (%)"
    )
    turndown_ratio: float = Field(
        default=4.0,
        ge=2.0,
        le=20.0,
        description="Turndown ratio (e.g., 4:1)"
    )
    number_of_burners: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of burners"
    )
    furnace_volume_ft3: Optional[float] = Field(
        default=None,
        ge=10.0,
        le=100000.0,
        description="Furnace volume (ft³)"
    )
    heating_surface_ft2: Optional[float] = Field(
        default=None,
        ge=100.0,
        le=500000.0,
        description="Total heating surface (ft²)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

class MetricsConfig(BaseModel):
    """Prometheus metrics configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    prefix: str = Field(
        default="greenlang_flameguard",
        description="Metrics name prefix"
    )
    port: int = Field(
        default=9091,
        ge=1024,
        le=65535,
        description="Metrics HTTP port"
    )
    push_gateway_url: Optional[str] = Field(
        default=None,
        description="Prometheus push gateway URL"
    )
    collection_interval_s: float = Field(
        default=15.0,
        ge=1.0,
        le=300.0,
        description="Metrics collection interval"
    )
    include_process_metrics: bool = Field(
        default=True,
        description="Include process variable metrics"
    )
    include_optimization_metrics: bool = Field(
        default=True,
        description="Include optimization metrics"
    )
    include_safety_metrics: bool = Field(
        default=True,
        description="Include safety system metrics"
    )
    histogram_buckets: List[float] = Field(
        default=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        description="Histogram bucket boundaries"
    )


class AlertingConfig(BaseModel):
    """Alerting configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable alerting"
    )
    efficiency_drop_alert_percent: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Efficiency drop alert threshold (%)"
    )
    emissions_spike_alert_percent: float = Field(
        default=20.0,
        ge=5.0,
        le=100.0,
        description="Emissions spike alert threshold (%)"
    )
    equipment_degradation_alert: bool = Field(
        default=True,
        description="Alert on equipment degradation"
    )
    optimization_failure_alert: bool = Field(
        default=True,
        description="Alert on optimization failures"
    )
    alert_cooldown_s: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Alert cooldown period (seconds)"
    )


# =============================================================================
# API CONFIGURATION
# =============================================================================

class APIConfig(BaseModel):
    """API configuration."""

    rest_enabled: bool = Field(
        default=True,
        description="Enable REST API"
    )
    rest_port: int = Field(
        default=8002,
        ge=1024,
        le=65535,
        description="REST API port"
    )
    graphql_enabled: bool = Field(
        default=True,
        description="Enable GraphQL API"
    )
    graphql_port: int = Field(
        default=8003,
        ge=1024,
        le=65535,
        description="GraphQL API port"
    )
    grpc_enabled: bool = Field(
        default=True,
        description="Enable gRPC API"
    )
    grpc_port: int = Field(
        default=50052,
        ge=1024,
        le=65535,
        description="gRPC port"
    )
    auth_enabled: bool = Field(
        default=True,
        description="Enable API authentication"
    )
    auth_method: str = Field(
        default="jwt",
        description="Authentication method (jwt, oauth2, api_key)"
    )
    rate_limit_rpm: int = Field(
        default=1000,
        ge=10,
        description="Rate limit (requests per minute)"
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="CORS allowed origins"
    )


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

class FlameguardConfig(BaseModel):
    """
    Complete GL-002 FLAMEGUARD BoilerEfficiencyOptimizer configuration.

    This is the main configuration class aggregating all sub-configurations.

    Example:
        >>> config = FlameguardConfig(
        ...     boiler=BoilerSpecifications(
        ...         boiler_id="BOILER-001",
        ...         rated_capacity_klb_hr=150.0
        ...     ),
        ...     safety=SafetyConfig(sil_level=SafetyIntegrityLevel.SIL_2)
        ... )
        >>> optimizer = BoilerEfficiencyOptimizer(config)
    """

    # Identity
    agent_id: str = Field(
        default_factory=lambda: f"GL-002-{os.getpid()}",
        description="Unique agent identifier"
    )
    name: str = Field(
        default="FLAMEGUARD-Primary",
        description="Human-readable agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Agent version"
    )

    # Environment
    environment: str = Field(
        default="production",
        description="Environment (development, staging, production)"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    # Equipment
    boiler: BoilerSpecifications = Field(
        default_factory=BoilerSpecifications,
        description="Boiler equipment specifications"
    )

    # Fuel
    fuel: FuelConfig = Field(
        default_factory=FuelConfig,
        description="Fuel system configuration"
    )

    # Combustion
    combustion: CombustionConfig = Field(
        default_factory=CombustionConfig,
        description="Combustion control configuration"
    )

    # Safety
    safety: SafetyConfig = Field(
        default_factory=SafetyConfig,
        description="Safety system configuration"
    )

    # Efficiency
    efficiency: EfficiencyCalculationConfig = Field(
        default_factory=EfficiencyCalculationConfig,
        description="Efficiency calculation configuration"
    )
    heat_balance: HeatBalanceConfig = Field(
        default_factory=HeatBalanceConfig,
        description="Heat balance configuration"
    )

    # Emissions
    emissions: EmissionsConfig = Field(
        default_factory=EmissionsConfig,
        description="Emissions monitoring configuration"
    )

    # Optimization
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimization configuration"
    )

    # Integration
    integration: IntegrationConfig = Field(
        default_factory=IntegrationConfig,
        description="External system integration"
    )

    # Monitoring
    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig,
        description="Metrics configuration"
    )
    alerting: AlertingConfig = Field(
        default_factory=AlertingConfig,
        description="Alerting configuration"
    )

    # API
    api: APIConfig = Field(
        default_factory=APIConfig,
        description="API configuration"
    )

    # Operational
    audit_enabled: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )
    deterministic_mode: bool = Field(
        default=True,
        description="Enable deterministic calculations"
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )

    class Config:
        use_enum_values = True

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment is valid."""
        valid_envs = {"development", "staging", "production"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @classmethod
    def from_environment(cls) -> "FlameguardConfig":
        """
        Create configuration from environment variables.

        Environment variables follow the pattern:
        GL_FLAMEGUARD_<SECTION>_<KEY>=value

        Example:
            GL_FLAMEGUARD_SAFETY_SIL_LEVEL=SIL_2
            GL_FLAMEGUARD_API_REST_PORT=8080
        """
        config_dict = {}
        prefix = "GL_FLAMEGUARD_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_path = key[len(prefix):].lower().split("_")
                current = config_dict
                for part in config_path[:-1]:
                    current = current.setdefault(part, {})
                current[config_path[-1]] = value

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FlameguardConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
