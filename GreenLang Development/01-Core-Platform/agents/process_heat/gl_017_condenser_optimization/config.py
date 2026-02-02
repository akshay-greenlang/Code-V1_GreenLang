"""
GL-017 CONDENSYNC Agent - Configuration Module

Configuration schemas for condenser optimization including cooling tower,
tube fouling, vacuum system, air ingress, and HEI cleanliness settings.

Standards Reference: HEI Standards for Steam Surface Condensers, 12th Edition
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator


class CondenserType(Enum):
    """Surface condenser types per HEI classification."""
    SINGLE_PASS = "single_pass"
    TWO_PASS = "two_pass"
    DIVIDED_WATERBOX = "divided_waterbox"
    SINGLE_SHELL = "single_shell"
    MULTI_SHELL = "multi_shell"
    AIR_COOLED = "air_cooled"


class TubeMaterial(Enum):
    """Condenser tube materials."""
    ADMIRALTY_BRASS = "admiralty_brass"
    ALUMINUM_BRASS = "aluminum_brass"
    COPPER_NICKEL_90_10 = "copper_nickel_90_10"
    COPPER_NICKEL_70_30 = "copper_nickel_70_30"
    STAINLESS_304 = "stainless_304"
    STAINLESS_316 = "stainless_316"
    TITANIUM = "titanium"
    DUPLEX_2205 = "duplex_2205"


class CoolingWaterSource(Enum):
    """Cooling water source types."""
    ONCE_THROUGH_FRESH = "once_through_fresh"
    ONCE_THROUGH_SEAWATER = "once_through_seawater"
    COOLING_TOWER_MECHANICAL = "cooling_tower_mechanical"
    COOLING_TOWER_NATURAL = "cooling_tower_natural"
    HYBRID_COOLING = "hybrid_cooling"
    DRY_COOLING = "dry_cooling"


class VacuumEquipmentType(Enum):
    """Vacuum system equipment types."""
    STEAM_JET_EJECTOR = "steam_jet_ejector"
    LIQUID_RING_PUMP = "liquid_ring_pump"
    DRY_VACUUM_PUMP = "dry_vacuum_pump"
    HYBRID_SYSTEM = "hybrid_system"


class CoolingTowerConfig(BaseModel):
    """Cooling tower configuration."""

    # Tower specifications
    tower_type: str = Field(
        default="mechanical_draft",
        description="Tower type (mechanical_draft, natural_draft, hybrid)"
    )
    design_capacity_gpm: float = Field(
        default=50000.0,
        gt=0,
        description="Design circulation rate (GPM)"
    )
    design_range_f: float = Field(
        default=15.0,
        gt=0,
        le=40,
        description="Design temperature range (F)"
    )
    design_approach_f: float = Field(
        default=8.0,
        gt=0,
        le=20,
        description="Design approach to wet bulb (F)"
    )
    design_wet_bulb_f: float = Field(
        default=78.0,
        description="Design wet bulb temperature (F)"
    )

    # Water chemistry
    target_cycles_concentration: float = Field(
        default=5.0,
        ge=1.5,
        le=10.0,
        description="Target cycles of concentration"
    )
    max_cycles_concentration: float = Field(
        default=7.0,
        ge=2.0,
        le=12.0,
        description="Maximum cycles of concentration"
    )
    min_cycles_concentration: float = Field(
        default=3.0,
        ge=1.5,
        le=6.0,
        description="Minimum cycles of concentration"
    )

    # Chemistry limits
    max_silica_ppm: float = Field(
        default=150.0,
        ge=50,
        le=200,
        description="Maximum silica (ppm as SiO2)"
    )
    max_calcium_ppm: float = Field(
        default=800.0,
        ge=200,
        le=1500,
        description="Maximum calcium hardness (ppm as CaCO3)"
    )
    max_chlorides_ppm: float = Field(
        default=500.0,
        ge=100,
        le=2000,
        description="Maximum chlorides (ppm)"
    )
    max_conductivity_umhos: float = Field(
        default=3000.0,
        ge=500,
        le=5000,
        description="Maximum conductivity (umhos/cm)"
    )
    target_ph: float = Field(
        default=8.0,
        ge=6.5,
        le=9.5,
        description="Target pH"
    )

    # Blowdown
    blowdown_control: str = Field(
        default="conductivity",
        description="Blowdown control method (conductivity, timer, manual)"
    )
    blowdown_setpoint_umhos: float = Field(
        default=2500.0,
        ge=500,
        le=5000,
        description="Blowdown setpoint conductivity (umhos/cm)"
    )

    # Biocide
    biocide_program: str = Field(
        default="oxidizing",
        description="Biocide program (oxidizing, non_oxidizing, combined)"
    )
    free_chlorine_target_ppm: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Free chlorine target (ppm)"
    )


class TubeFoulingConfig(BaseModel):
    """Tube fouling detection configuration."""

    # Design values
    design_cleanliness_factor: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Design cleanliness factor per HEI"
    )
    design_fouling_factor_hr_ft2_f_btu: float = Field(
        default=0.0005,
        ge=0.0,
        le=0.005,
        description="Design fouling factor (hr-ft2-F/BTU)"
    )

    # Tube specifications
    tube_material: TubeMaterial = Field(
        default=TubeMaterial.STAINLESS_316,
        description="Tube material"
    )
    tube_od_in: float = Field(
        default=0.875,
        ge=0.5,
        le=1.5,
        description="Tube OD (inches)"
    )
    tube_gauge: int = Field(
        default=18,
        ge=14,
        le=24,
        description="Tube gauge (BWG)"
    )
    tube_count: int = Field(
        default=15000,
        gt=0,
        description="Number of tubes"
    )
    tube_length_ft: float = Field(
        default=40.0,
        gt=0,
        le=60,
        description="Tube length (ft)"
    )

    # Fouling thresholds
    cleanliness_warning_threshold: float = Field(
        default=0.75,
        ge=0.5,
        le=0.9,
        description="Cleanliness warning threshold"
    )
    cleanliness_alarm_threshold: float = Field(
        default=0.65,
        ge=0.4,
        le=0.8,
        description="Cleanliness alarm threshold"
    )
    cleaning_trigger_threshold: float = Field(
        default=0.60,
        ge=0.3,
        le=0.7,
        description="Cleaning trigger threshold"
    )

    # Backpressure monitoring
    backpressure_baseline_inhg: float = Field(
        default=1.5,
        ge=0.5,
        le=3.0,
        description="Baseline backpressure (inHgA)"
    )
    backpressure_deviation_warning_inhg: float = Field(
        default=0.3,
        ge=0.1,
        le=0.5,
        description="Backpressure deviation warning (inHg)"
    )
    backpressure_deviation_alarm_inhg: float = Field(
        default=0.5,
        ge=0.2,
        le=1.0,
        description="Backpressure deviation alarm (inHg)"
    )

    class Config:
        use_enum_values = True


class VacuumSystemConfig(BaseModel):
    """Vacuum system configuration."""

    # Equipment
    primary_equipment: VacuumEquipmentType = Field(
        default=VacuumEquipmentType.STEAM_JET_EJECTOR,
        description="Primary vacuum equipment type"
    )
    backup_equipment: Optional[VacuumEquipmentType] = Field(
        default=None,
        description="Backup vacuum equipment type"
    )
    ejector_stages: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Number of ejector stages"
    )

    # Design vacuum
    design_vacuum_inhga: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Design vacuum (inHgA)"
    )
    min_vacuum_inhga: float = Field(
        default=2.5,
        ge=1.0,
        le=6.0,
        description="Minimum acceptable vacuum (inHgA)"
    )
    max_vacuum_inhga: float = Field(
        default=0.8,
        ge=0.3,
        le=2.0,
        description="Maximum vacuum (inHgA)"
    )

    # Steam supply (for ejectors)
    motive_steam_pressure_psig: float = Field(
        default=150.0,
        ge=50,
        le=400,
        description="Motive steam pressure (psig)"
    )
    motive_steam_superheat_f: float = Field(
        default=50.0,
        ge=0,
        le=200,
        description="Motive steam superheat (F)"
    )

    # Performance monitoring
    air_removal_capacity_scfm: float = Field(
        default=50.0,
        gt=0,
        description="Design air removal capacity (SCFM)"
    )
    air_removal_warning_pct: float = Field(
        default=80.0,
        ge=50,
        le=95,
        description="Air removal capacity warning (%)"
    )

    # Leak detection
    vacuum_decay_test_enabled: bool = Field(
        default=True,
        description="Enable vacuum decay testing"
    )
    acceptable_decay_rate_inhg_min: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Acceptable vacuum decay rate (inHg/min)"
    )

    class Config:
        use_enum_values = True


class AirIngresConfig(BaseModel):
    """Air ingress detection configuration."""

    # Detection limits
    max_air_ingress_scfm: float = Field(
        default=10.0,
        gt=0,
        le=100,
        description="Maximum acceptable air ingress (SCFM)"
    )
    warning_air_ingress_scfm: float = Field(
        default=5.0,
        gt=0,
        le=50,
        description="Warning air ingress level (SCFM)"
    )

    # Detection methods
    dissolved_oxygen_monitoring: bool = Field(
        default=True,
        description="Enable dissolved oxygen monitoring"
    )
    do_warning_ppb: float = Field(
        default=20.0,
        ge=5,
        le=100,
        description="Dissolved oxygen warning (ppb)"
    )
    do_alarm_ppb: float = Field(
        default=50.0,
        ge=10,
        le=200,
        description="Dissolved oxygen alarm (ppb)"
    )

    # Temperature monitoring
    subcooling_warning_f: float = Field(
        default=3.0,
        ge=1,
        le=10,
        description="Subcooling warning (F)"
    )
    subcooling_alarm_f: float = Field(
        default=5.0,
        ge=2,
        le=15,
        description="Subcooling alarm (F)"
    )

    # Source identification
    tracer_gas_testing: bool = Field(
        default=True,
        description="Enable tracer gas testing capability"
    )
    leak_location_zones: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of leak detection zones"
    )


class CleanlinessConfig(BaseModel):
    """HEI cleanliness factor configuration."""

    # HEI Standards parameters
    hei_edition: str = Field(
        default="12th",
        description="HEI Standards edition"
    )
    heat_transfer_coefficient_method: str = Field(
        default="hei_standard",
        description="U coefficient method (hei_standard, actual_test)"
    )

    # Reference conditions
    reference_inlet_temp_f: float = Field(
        default=70.0,
        description="Reference inlet water temperature (F)"
    )
    reference_velocity_fps: float = Field(
        default=7.0,
        ge=3,
        le=12,
        description="Reference tube velocity (ft/s)"
    )

    # Material correction factors
    tube_material_factor: float = Field(
        default=1.0,
        ge=0.6,
        le=1.1,
        description="Tube material correction factor"
    )
    inlet_water_factor: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Inlet water correction factor"
    )

    # Calculation settings
    include_velocity_correction: bool = Field(
        default=True,
        description="Include velocity correction"
    )
    include_temperature_correction: bool = Field(
        default=True,
        description="Include temperature correction"
    )


class PerformanceConfig(BaseModel):
    """Condenser performance curve configuration."""

    # Design point
    design_duty_btu_hr: float = Field(
        default=500_000_000.0,
        gt=0,
        description="Design duty (BTU/hr)"
    )
    design_steam_flow_lb_hr: float = Field(
        default=500000.0,
        gt=0,
        description="Design steam flow (lb/hr)"
    )
    design_backpressure_inhga: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Design backpressure (inHgA)"
    )
    design_inlet_temp_f: float = Field(
        default=70.0,
        description="Design inlet water temp (F)"
    )
    design_outlet_temp_f: float = Field(
        default=95.0,
        description="Design outlet water temp (F)"
    )
    design_cw_flow_gpm: float = Field(
        default=100000.0,
        gt=0,
        description="Design cooling water flow (GPM)"
    )

    # Performance curves
    curve_points: int = Field(
        default=10,
        ge=5,
        le=20,
        description="Number of performance curve points"
    )
    load_range_min_pct: float = Field(
        default=30.0,
        ge=10,
        le=50,
        description="Minimum load for curve (%)"
    )
    load_range_max_pct: float = Field(
        default=110.0,
        ge=100,
        le=120,
        description="Maximum load for curve (%)"
    )

    # Deviation thresholds
    backpressure_deviation_warning_pct: float = Field(
        default=10.0,
        ge=5,
        le=25,
        description="Backpressure deviation warning (%)"
    )
    backpressure_deviation_alarm_pct: float = Field(
        default=20.0,
        ge=10,
        le=50,
        description="Backpressure deviation alarm (%)"
    )


class CondenserOptimizationConfig(BaseModel):
    """
    Complete condenser optimization configuration.

    This configuration defines all parameters for the GL-017
    CONDENSYNC Agent including cooling tower, tube fouling,
    vacuum system, air ingress, and HEI cleanliness settings.
    """

    # Identity
    condenser_id: str = Field(..., description="Unique condenser identifier")
    name: str = Field(default="", description="Condenser name")
    condenser_type: CondenserType = Field(
        default=CondenserType.TWO_PASS,
        description="Condenser type"
    )
    cooling_source: CoolingWaterSource = Field(
        default=CoolingWaterSource.COOLING_TOWER_MECHANICAL,
        description="Cooling water source"
    )

    # Design specifications
    design_surface_area_ft2: float = Field(
        default=150000.0,
        gt=0,
        description="Design heat transfer surface area (ft2)"
    )
    shell_count: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of shells"
    )
    passes: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Number of passes"
    )

    # Sub-configurations
    cooling_tower: CoolingTowerConfig = Field(
        default_factory=CoolingTowerConfig,
        description="Cooling tower configuration"
    )
    tube_fouling: TubeFoulingConfig = Field(
        default_factory=TubeFoulingConfig,
        description="Tube fouling configuration"
    )
    vacuum_system: VacuumSystemConfig = Field(
        default_factory=VacuumSystemConfig,
        description="Vacuum system configuration"
    )
    air_ingress: AirIngresConfig = Field(
        default_factory=AirIngresConfig,
        description="Air ingress configuration"
    )
    cleanliness: CleanlinessConfig = Field(
        default_factory=CleanlinessConfig,
        description="HEI cleanliness configuration"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance configuration"
    )

    # Control settings
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

    # Safety
    sil_level: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Safety Integrity Level"
    )
    low_vacuum_trip_inhga: float = Field(
        default=5.0,
        ge=2.0,
        le=10.0,
        description="Low vacuum trip point (inHgA)"
    )
    high_hotwell_level_trip_pct: float = Field(
        default=90.0,
        ge=80,
        le=100,
        description="High hotwell level trip (%)"
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
        """Set default name from condenser_id."""
        if not v and "condenser_id" in values:
            return f"Condenser {values['condenser_id']}"
        return v
