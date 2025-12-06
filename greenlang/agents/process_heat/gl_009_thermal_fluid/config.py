"""
GL-009 THERMALIQ Agent - Configuration Schemas

This module defines configuration schemas for the thermal fluid systems agent,
including system configuration, safety limits, alarm setpoints, and
degradation thresholds.

Example:
    >>> from greenlang.agents.process_heat.gl_009_thermal_fluid.config import (
    ...     ThermalFluidConfig,
    ...     create_default_config,
    ... )
    >>> config = create_default_config(system_id="TF-001")
    >>> print(config.fluid_type)
    therminol_66
"""

from typing import Dict, List, Optional, Set
import uuid

from pydantic import BaseModel, Field, validator

from .schemas import ThermalFluidType, HeaterType


# =============================================================================
# SAFETY CONFIGURATION
# =============================================================================

class TemperatureLimits(BaseModel):
    """Temperature safety limits configuration."""

    max_film_temp_f: float = Field(
        default=700.0,
        gt=0,
        description="Maximum allowable film temperature (F)"
    )
    max_bulk_temp_f: float = Field(
        default=650.0,
        gt=0,
        description="Maximum allowable bulk temperature (F)"
    )
    high_bulk_temp_alarm_f: float = Field(
        default=620.0,
        gt=0,
        description="High bulk temperature alarm (F)"
    )
    high_bulk_temp_trip_f: float = Field(
        default=640.0,
        gt=0,
        description="High bulk temperature trip (F)"
    )
    low_bulk_temp_alarm_f: float = Field(
        default=200.0,
        ge=0,
        description="Low bulk temperature alarm (F)"
    )
    min_flash_point_margin_f: float = Field(
        default=50.0,
        gt=0,
        description="Minimum flash point safety margin (F)"
    )
    min_auto_ignition_margin_f: float = Field(
        default=100.0,
        gt=0,
        description="Minimum auto-ignition safety margin (F)"
    )


class FlowLimits(BaseModel):
    """Flow safety limits configuration."""

    min_flow_pct: float = Field(
        default=25.0,
        ge=0,
        le=100,
        description="Minimum flow percentage of design"
    )
    low_flow_alarm_pct: float = Field(
        default=30.0,
        ge=0,
        le=100,
        description="Low flow alarm (%)"
    )
    low_flow_trip_pct: float = Field(
        default=25.0,
        ge=0,
        le=100,
        description="Low flow trip (%)"
    )
    max_velocity_ft_s: float = Field(
        default=12.0,
        gt=0,
        description="Maximum fluid velocity (ft/s)"
    )


class PressureLimits(BaseModel):
    """Pressure safety limits configuration."""

    max_system_pressure_psig: float = Field(
        default=150.0,
        gt=0,
        description="Maximum system pressure (psig)"
    )
    min_pump_suction_pressure_psig: float = Field(
        default=5.0,
        ge=-14.7,
        description="Minimum pump suction pressure (psig)"
    )
    min_npsh_margin_ft: float = Field(
        default=3.0,
        ge=0,
        description="Minimum NPSH margin (ft)"
    )


class SafetyConfig(BaseModel):
    """Complete safety configuration."""

    sil_level: int = Field(
        default=2,
        ge=0,
        le=4,
        description="Safety Integrity Level (0-4)"
    )
    temperature_limits: TemperatureLimits = Field(
        default_factory=TemperatureLimits,
        description="Temperature limits"
    )
    flow_limits: FlowLimits = Field(
        default_factory=FlowLimits,
        description="Flow limits"
    )
    pressure_limits: PressureLimits = Field(
        default_factory=PressureLimits,
        description="Pressure limits"
    )
    emergency_shutdown_enabled: bool = Field(
        default=True,
        description="Enable ESD integration"
    )
    watchdog_timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="Watchdog timeout (ms)"
    )


# =============================================================================
# DEGRADATION CONFIGURATION
# =============================================================================

class DegradationThresholds(BaseModel):
    """Fluid degradation monitoring thresholds."""

    # Viscosity limits (% change from baseline)
    viscosity_warning_pct: float = Field(
        default=10.0,
        gt=0,
        description="Viscosity change warning threshold (%)"
    )
    viscosity_critical_pct: float = Field(
        default=25.0,
        gt=0,
        description="Viscosity change critical threshold (%)"
    )

    # Flash point limits (drop from new)
    flash_point_warning_drop_f: float = Field(
        default=30.0,
        gt=0,
        description="Flash point drop warning (F)"
    )
    flash_point_critical_drop_f: float = Field(
        default=50.0,
        gt=0,
        description="Flash point drop critical (F)"
    )

    # Acid number limits (mg KOH/g)
    acid_number_warning: float = Field(
        default=0.2,
        ge=0,
        description="Acid number warning threshold"
    )
    acid_number_critical: float = Field(
        default=0.5,
        ge=0,
        description="Acid number critical threshold"
    )

    # Carbon residue limits (%)
    carbon_residue_warning_pct: float = Field(
        default=0.5,
        ge=0,
        description="Carbon residue warning (%)"
    )
    carbon_residue_critical_pct: float = Field(
        default=1.0,
        ge=0,
        description="Carbon residue critical (%)"
    )

    # Moisture limits (ppm)
    moisture_warning_ppm: float = Field(
        default=500.0,
        ge=0,
        description="Moisture content warning (ppm)"
    )
    moisture_critical_ppm: float = Field(
        default=1000.0,
        ge=0,
        description="Moisture content critical (ppm)"
    )

    # Low boilers limits (%)
    low_boilers_warning_pct: float = Field(
        default=3.0,
        ge=0,
        description="Low boilers warning (%)"
    )
    low_boilers_critical_pct: float = Field(
        default=10.0,
        ge=0,
        description="Low boilers critical (%)"
    )

    # High boilers limits (%)
    high_boilers_warning_pct: float = Field(
        default=5.0,
        ge=0,
        description="High boilers warning (%)"
    )
    high_boilers_critical_pct: float = Field(
        default=15.0,
        ge=0,
        description="High boilers critical (%)"
    )

    # Sampling interval (months)
    sampling_interval_months: int = Field(
        default=6,
        ge=1,
        le=24,
        description="Recommended sampling interval"
    )
    sampling_interval_critical_months: int = Field(
        default=3,
        ge=1,
        le=12,
        description="Sampling interval when critical"
    )


# =============================================================================
# HEATER CONFIGURATION
# =============================================================================

class HeaterConfig(BaseModel):
    """Thermal fluid heater configuration."""

    heater_id: str = Field(
        default="HT-001",
        description="Heater identifier"
    )
    heater_type: HeaterType = Field(
        default=HeaterType.FIRED_HEATER,
        description="Heater type"
    )
    design_duty_btu_hr: float = Field(
        default=10_000_000.0,
        gt=0,
        description="Design duty (BTU/hr)"
    )
    design_flow_gpm: float = Field(
        default=500.0,
        gt=0,
        description="Design flow rate (GPM)"
    )
    design_delta_t_f: float = Field(
        default=50.0,
        gt=0,
        description="Design temperature rise (F)"
    )
    max_film_temp_f: float = Field(
        default=700.0,
        gt=0,
        description="Maximum film temperature (F)"
    )
    coil_tube_od_in: float = Field(
        default=3.5,
        gt=0,
        description="Heater coil tube OD (inches)"
    )
    coil_tube_id_in: float = Field(
        default=3.068,
        gt=0,
        description="Heater coil tube ID (inches)"
    )
    coil_length_ft: float = Field(
        default=1000.0,
        gt=0,
        description="Total coil length (ft)"
    )
    radiant_heat_flux_btu_hr_ft2: float = Field(
        default=10000.0,
        gt=0,
        description="Radiant section heat flux (BTU/hr-ft2)"
    )
    convection_heat_flux_btu_hr_ft2: float = Field(
        default=5000.0,
        gt=0,
        description="Convection section heat flux (BTU/hr-ft2)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# EXPANSION TANK CONFIGURATION
# =============================================================================

class ExpansionTankConfig(BaseModel):
    """Expansion tank configuration per API 660."""

    tank_id: str = Field(
        default="ET-001",
        description="Expansion tank identifier"
    )
    volume_gallons: float = Field(
        default=1000.0,
        gt=0,
        description="Tank volume (gallons)"
    )
    design_pressure_psig: float = Field(
        default=15.0,
        ge=-14.7,
        description="Design pressure (psig)"
    )
    design_temperature_f: float = Field(
        default=300.0,
        gt=0,
        description="Design temperature (F)"
    )
    inert_gas_blanket: bool = Field(
        default=True,
        description="Nitrogen blanket installed"
    )
    blanket_pressure_psig: float = Field(
        default=2.0,
        ge=0,
        description="Nitrogen blanket pressure (psig)"
    )
    min_level_pct: float = Field(
        default=10.0,
        ge=0,
        le=100,
        description="Minimum level alarm (%)"
    )
    max_level_pct: float = Field(
        default=90.0,
        ge=0,
        le=100,
        description="Maximum level alarm (%)"
    )
    cold_level_target_pct: float = Field(
        default=25.0,
        ge=0,
        le=100,
        description="Target cold level (%)"
    )


# =============================================================================
# PUMP CONFIGURATION
# =============================================================================

class PumpConfig(BaseModel):
    """Thermal fluid pump configuration."""

    pump_id: str = Field(
        default="P-001",
        description="Pump identifier"
    )
    design_flow_gpm: float = Field(
        default=500.0,
        gt=0,
        description="Design flow rate (GPM)"
    )
    design_head_ft: float = Field(
        default=150.0,
        gt=0,
        description="Design head (ft)"
    )
    npsh_required_ft: float = Field(
        default=10.0,
        ge=0,
        description="NPSH required (ft)"
    )
    efficiency_pct: float = Field(
        default=75.0,
        gt=0,
        le=100,
        description="Pump efficiency (%)"
    )
    motor_hp: float = Field(
        default=50.0,
        gt=0,
        description="Motor horsepower"
    )


# =============================================================================
# PIPING CONFIGURATION
# =============================================================================

class PipingConfig(BaseModel):
    """Piping system configuration."""

    pipe_schedule: str = Field(
        default="40",
        description="Pipe schedule"
    )
    main_header_size_in: float = Field(
        default=6.0,
        gt=0,
        description="Main header size (inches)"
    )
    branch_line_size_in: float = Field(
        default=3.0,
        gt=0,
        description="Branch line size (inches)"
    )
    total_pipe_length_ft: float = Field(
        default=500.0,
        gt=0,
        description="Total equivalent pipe length (ft)"
    )
    insulation_thickness_in: float = Field(
        default=2.0,
        ge=0,
        description="Pipe insulation thickness (inches)"
    )
    insulation_conductivity_btu_hr_ft_f: float = Field(
        default=0.025,
        gt=0,
        description="Insulation thermal conductivity"
    )


# =============================================================================
# EXERGY CONFIGURATION
# =============================================================================

class ExergyConfig(BaseModel):
    """Exergy analysis configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable exergy analysis"
    )
    reference_temperature_f: float = Field(
        default=77.0,
        description="Dead state reference temperature (F)"
    )
    reference_pressure_psia: float = Field(
        default=14.696,
        description="Dead state reference pressure (psia)"
    )
    include_chemical_exergy: bool = Field(
        default=False,
        description="Include chemical exergy (advanced)"
    )


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

class ThermalFluidConfig(BaseModel):
    """Complete GL-009 THERMALIQ agent configuration."""

    # Identity
    agent_id: str = Field(
        default_factory=lambda: f"GL-009-{str(uuid.uuid4())[:8]}",
        description="Agent instance identifier"
    )
    system_id: str = Field(
        ...,
        description="Thermal fluid system identifier"
    )
    system_name: str = Field(
        default="Thermal Fluid System",
        description="Human-readable system name"
    )

    # Fluid specification
    fluid_type: ThermalFluidType = Field(
        default=ThermalFluidType.THERMINOL_66,
        description="Thermal fluid type"
    )
    system_volume_gallons: float = Field(
        default=5000.0,
        gt=0,
        description="Total system volume (gallons)"
    )
    design_temperature_f: float = Field(
        default=600.0,
        gt=0,
        description="Design operating temperature (F)"
    )
    design_flow_gpm: float = Field(
        default=500.0,
        gt=0,
        description="Design flow rate (GPM)"
    )

    # Sub-configurations
    safety: SafetyConfig = Field(
        default_factory=SafetyConfig,
        description="Safety configuration"
    )
    degradation: DegradationThresholds = Field(
        default_factory=DegradationThresholds,
        description="Degradation thresholds"
    )
    heater: HeaterConfig = Field(
        default_factory=HeaterConfig,
        description="Heater configuration"
    )
    expansion_tank: ExpansionTankConfig = Field(
        default_factory=ExpansionTankConfig,
        description="Expansion tank configuration"
    )
    pump: PumpConfig = Field(
        default_factory=PumpConfig,
        description="Pump configuration"
    )
    piping: PipingConfig = Field(
        default_factory=PipingConfig,
        description="Piping configuration"
    )
    exergy: ExergyConfig = Field(
        default_factory=ExergyConfig,
        description="Exergy analysis configuration"
    )

    # Processing options
    enable_ml_predictions: bool = Field(
        default=False,
        description="Enable ML-based predictions (non-safety)"
    )
    enable_trending: bool = Field(
        default=True,
        description="Enable historical trending"
    )
    audit_enabled: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )

    # Cost parameters for recommendations
    fuel_cost_usd_mmbtu: float = Field(
        default=8.0,
        gt=0,
        description="Fuel cost ($/MMBTU)"
    )
    electricity_cost_usd_kwh: float = Field(
        default=0.10,
        gt=0,
        description="Electricity cost ($/kWh)"
    )
    fluid_cost_usd_gallon: float = Field(
        default=15.0,
        gt=0,
        description="Thermal fluid cost ($/gallon)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_default_config(
    system_id: str,
    fluid_type: ThermalFluidType = ThermalFluidType.THERMINOL_66,
    design_temperature_f: float = 600.0,
    design_flow_gpm: float = 500.0,
    system_volume_gallons: float = 5000.0,
) -> ThermalFluidConfig:
    """
    Create a default thermal fluid system configuration.

    Args:
        system_id: System identifier
        fluid_type: Thermal fluid type
        design_temperature_f: Design operating temperature (F)
        design_flow_gpm: Design flow rate (GPM)
        system_volume_gallons: System volume (gallons)

    Returns:
        ThermalFluidConfig with defaults for specified fluid

    Example:
        >>> config = create_default_config(
        ...     system_id="TF-001",
        ...     fluid_type=ThermalFluidType.DOWTHERM_A,
        ...     design_temperature_f=700.0,
        ... )
    """
    # Adjust safety limits based on fluid type
    temp_limits = _get_temperature_limits_for_fluid(fluid_type)

    safety = SafetyConfig(
        temperature_limits=temp_limits,
    )

    heater = HeaterConfig(
        design_flow_gpm=design_flow_gpm,
        max_film_temp_f=temp_limits.max_film_temp_f,
    )

    pump = PumpConfig(
        design_flow_gpm=design_flow_gpm,
    )

    return ThermalFluidConfig(
        system_id=system_id,
        fluid_type=fluid_type,
        design_temperature_f=design_temperature_f,
        design_flow_gpm=design_flow_gpm,
        system_volume_gallons=system_volume_gallons,
        safety=safety,
        heater=heater,
        pump=pump,
    )


def _get_temperature_limits_for_fluid(
    fluid_type: ThermalFluidType
) -> TemperatureLimits:
    """Get temperature limits for specific fluid type."""
    # Maximum bulk and film temperatures per manufacturer specifications
    limits_map = {
        ThermalFluidType.THERMINOL_55: TemperatureLimits(
            max_film_temp_f=550.0,
            max_bulk_temp_f=500.0,
            high_bulk_temp_alarm_f=480.0,
            high_bulk_temp_trip_f=495.0,
        ),
        ThermalFluidType.THERMINOL_59: TemperatureLimits(
            max_film_temp_f=600.0,
            max_bulk_temp_f=550.0,
            high_bulk_temp_alarm_f=530.0,
            high_bulk_temp_trip_f=545.0,
        ),
        ThermalFluidType.THERMINOL_62: TemperatureLimits(
            max_film_temp_f=600.0,
            max_bulk_temp_f=550.0,
            high_bulk_temp_alarm_f=530.0,
            high_bulk_temp_trip_f=545.0,
        ),
        ThermalFluidType.THERMINOL_66: TemperatureLimits(
            max_film_temp_f=705.0,
            max_bulk_temp_f=650.0,
            high_bulk_temp_alarm_f=620.0,
            high_bulk_temp_trip_f=640.0,
        ),
        ThermalFluidType.THERMINOL_VP1: TemperatureLimits(
            max_film_temp_f=750.0,
            max_bulk_temp_f=750.0,
            high_bulk_temp_alarm_f=720.0,
            high_bulk_temp_trip_f=740.0,
        ),
        ThermalFluidType.DOWTHERM_A: TemperatureLimits(
            max_film_temp_f=750.0,
            max_bulk_temp_f=750.0,
            high_bulk_temp_alarm_f=720.0,
            high_bulk_temp_trip_f=740.0,
        ),
        ThermalFluidType.DOWTHERM_G: TemperatureLimits(
            max_film_temp_f=700.0,
            max_bulk_temp_f=650.0,
            high_bulk_temp_alarm_f=620.0,
            high_bulk_temp_trip_f=640.0,
        ),
        ThermalFluidType.DOWTHERM_Q: TemperatureLimits(
            max_film_temp_f=650.0,
            max_bulk_temp_f=600.0,
            high_bulk_temp_alarm_f=580.0,
            high_bulk_temp_trip_f=595.0,
        ),
        ThermalFluidType.MARLOTHERM_SH: TemperatureLimits(
            max_film_temp_f=660.0,
            max_bulk_temp_f=610.0,
            high_bulk_temp_alarm_f=590.0,
            high_bulk_temp_trip_f=605.0,
        ),
        ThermalFluidType.SYLTHERM_800: TemperatureLimits(
            max_film_temp_f=780.0,
            max_bulk_temp_f=750.0,
            high_bulk_temp_alarm_f=720.0,
            high_bulk_temp_trip_f=740.0,
        ),
    }

    return limits_map.get(
        fluid_type,
        TemperatureLimits()  # Default limits
    )


def create_high_temperature_config(
    system_id: str,
    design_temperature_f: float = 700.0,
) -> ThermalFluidConfig:
    """
    Create configuration optimized for high-temperature applications.

    Uses Dowtherm A or Therminol VP-1 for temperatures up to 750F.

    Args:
        system_id: System identifier
        design_temperature_f: Design operating temperature (F)

    Returns:
        ThermalFluidConfig for high-temperature service
    """
    if design_temperature_f > 750:
        raise ValueError("Design temperature exceeds maximum for organic fluids")

    fluid_type = ThermalFluidType.DOWTHERM_A
    if design_temperature_f <= 650:
        fluid_type = ThermalFluidType.THERMINOL_66

    config = create_default_config(
        system_id=system_id,
        fluid_type=fluid_type,
        design_temperature_f=design_temperature_f,
    )

    # Increase safety margins for high-temperature service
    config.safety.temperature_limits.min_flash_point_margin_f = 75.0
    config.safety.temperature_limits.min_auto_ignition_margin_f = 125.0

    return config


def create_food_grade_config(
    system_id: str,
    design_temperature_f: float = 500.0,
) -> ThermalFluidConfig:
    """
    Create configuration for food-grade applications.

    Uses food-grade approved fluids with appropriate safety margins.

    Args:
        system_id: System identifier
        design_temperature_f: Design operating temperature (F)

    Returns:
        ThermalFluidConfig for food-grade service
    """
    # Paratherm NF is food-grade
    config = create_default_config(
        system_id=system_id,
        fluid_type=ThermalFluidType.PARATHERM_NF,
        design_temperature_f=design_temperature_f,
    )

    # Enhanced monitoring for food-grade service
    config.degradation.sampling_interval_months = 3
    config.degradation.moisture_warning_ppm = 200.0
    config.degradation.moisture_critical_ppm = 500.0

    return config
