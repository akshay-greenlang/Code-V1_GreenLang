# -*- coding: utf-8 -*-
"""
GL-007 FurnaceOptimizer/CoolingTowerOptimizer - Configuration Module

This module defines all configuration schemas for the Furnace and Cooling Tower
Optimizer Agent, including furnace design parameters, cooling tower specifications,
combustion settings, heat transfer parameters, safety limits and interlocks,
SHAP/LIME explainability settings, provenance tracking, and OPC-UA/MQTT
integration configurations.

Configuration follows GreenLang patterns with Pydantic validation and
sensible defaults for industrial furnace and cooling tower applications.

Standards Compliance:
    - NFPA 86: Standard for Ovens and Furnaces
    - ASHRAE 90.1: Energy Standard for Buildings
    - ASHRAE Handbook: HVAC Systems and Equipment
    - API 560: Fired Heaters for General Refinery Service
    - CTI ATC-105: Acceptance Test Code for Water Cooling Towers

Example:
    >>> from greenlang.agents.process_heat.gl_007_furnace_optimizer.config import (
    ...     GL007Config,
    ...     FurnaceOptimizerConfig,
    ...     CoolingTowerConfig,
    ... )
    >>> config = GL007Config.create_default()
    >>> print(config.furnace.furnace_id)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# ENUMS - FURNACE AND COOLING TOWER CLASSIFICATIONS
# =============================================================================


class FurnaceType(str, Enum):
    """
    Types of industrial furnaces per NFPA 86.

    Classification based on heating method and application:
    - DIRECT_FIRED: Direct combustion in furnace chamber
    - INDIRECT_FIRED: Heat transfer through tubes/walls
    - ELECTRIC: Electric resistance or induction heating
    - RADIANT_TUBE: Radiant tube heating systems
    - RECUPERATIVE: Furnaces with heat recovery
    """
    DIRECT_FIRED = "direct_fired"
    INDIRECT_FIRED = "indirect_fired"
    ELECTRIC = "electric"
    RADIANT_TUBE = "radiant_tube"
    RECUPERATIVE = "recuperative"
    REGENERATIVE = "regenerative"
    BOX = "box"
    CONTINUOUS = "continuous"
    BATCH = "batch"


class FuelType(str, Enum):
    """Fuel types for combustion furnaces."""
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    HYDROGEN = "hydrogen"
    BIOGAS = "biogas"
    MIXED = "mixed"


class CoolingTowerType(str, Enum):
    """
    Types of cooling towers per ASHRAE/CTI standards.

    Classification based on airflow mechanism:
    - MECHANICAL_DRAFT: Fan-driven airflow
    - NATURAL_DRAFT: Buoyancy-driven airflow
    - CROSSFLOW: Air enters horizontally
    - COUNTERFLOW: Air flows opposite to water
    """
    MECHANICAL_INDUCED = "mechanical_induced"
    MECHANICAL_FORCED = "mechanical_forced"
    NATURAL_DRAFT = "natural_draft"
    CROSSFLOW = "crossflow"
    COUNTERFLOW = "counterflow"
    HYBRID = "hybrid"


class FillType(str, Enum):
    """Cooling tower fill media types."""
    SPLASH = "splash"
    FILM = "film"
    TRICKLE = "trickle"
    HYBRID = "hybrid"


class ControlMode(str, Enum):
    """Control operating modes."""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    CASCADE = "cascade"
    OPTIMIZING = "optimizing"
    FEEDFORWARD = "feedforward"
    MODEL_PREDICTIVE = "model_predictive"


class SafetyIntegrityLevel(str, Enum):
    """IEC 61508 / ISA 84 Safety Integrity Levels."""
    SIL_1 = "sil_1"
    SIL_2 = "sil_2"
    SIL_3 = "sil_3"
    NON_SIL = "non_sil"


class AlertSeverity(str, Enum):
    """Alert severity levels for monitoring."""
    GOOD = "good"
    ADVISORY = "advisory"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    TRIP = "trip"


class ExplainabilityMethod(str, Enum):
    """Explainability methods for ML models."""
    SHAP = "shap"
    LIME = "lime"
    SHAP_KERNEL = "shap_kernel"
    SHAP_TREE = "shap_tree"
    INTEGRATED_GRADIENTS = "integrated_gradients"


class OPCSecurityPolicy(str, Enum):
    """OPC-UA security policies."""
    NONE = "None"
    BASIC128RSA15 = "Basic128Rsa15"
    BASIC256 = "Basic256"
    BASIC256SHA256 = "Basic256Sha256"


class OPCSecurityMode(str, Enum):
    """OPC-UA security modes."""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class MQTTQoS(int, Enum):
    """MQTT Quality of Service levels."""
    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


# =============================================================================
# COMBUSTION CONFIGURATION
# =============================================================================


class BurnerConfig(BaseModel):
    """Configuration for furnace burner specifications."""

    burner_id: str = Field(
        default="BNR-001",
        description="Burner identifier"
    )
    burner_type: str = Field(
        default="premix",
        description="Burner type (premix, nozzle_mix, raw_gas)"
    )
    capacity_mmbtu_hr: float = Field(
        default=10.0,
        gt=0,
        le=500,
        description="Burner capacity (MMBtu/hr)"
    )
    min_firing_rate_pct: float = Field(
        default=10.0,
        ge=0,
        le=50,
        description="Minimum firing rate (%)"
    )
    max_firing_rate_pct: float = Field(
        default=100.0,
        ge=50,
        le=110,
        description="Maximum firing rate (%)"
    )
    turndown_ratio: float = Field(
        default=10.0,
        ge=2,
        le=50,
        description="Turndown ratio"
    )
    nox_emissions_lb_mmbtu: float = Field(
        default=0.05,
        ge=0,
        le=0.5,
        description="NOx emissions (lb/MMBtu)"
    )
    co_emissions_ppm: float = Field(
        default=50.0,
        ge=0,
        le=500,
        description="CO emissions target (ppm)"
    )


class CombustionConfig(BaseModel):
    """
    Configuration for combustion analysis and optimization.

    Defines fuel properties, air-fuel ratio targets, and emission limits
    per NFPA 86 and environmental regulations.
    """

    # Fuel configuration
    fuel_type: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Primary fuel type"
    )
    fuel_hhv_btu_scf: float = Field(
        default=1020.0,
        gt=0,
        le=3500,
        description="Fuel Higher Heating Value (Btu/scf)"
    )
    fuel_lhv_btu_scf: float = Field(
        default=920.0,
        gt=0,
        le=3200,
        description="Fuel Lower Heating Value (Btu/scf)"
    )
    fuel_specific_gravity: float = Field(
        default=0.60,
        gt=0,
        le=2.0,
        description="Fuel specific gravity relative to air"
    )

    # Natural gas composition (mol%)
    ch4_content_pct: float = Field(
        default=95.0,
        ge=70,
        le=100,
        description="Methane content (%)"
    )
    c2h6_content_pct: float = Field(
        default=2.5,
        ge=0,
        le=15,
        description="Ethane content (%)"
    )
    n2_content_pct: float = Field(
        default=1.0,
        ge=0,
        le=20,
        description="Nitrogen content (%)"
    )
    co2_content_pct: float = Field(
        default=0.5,
        ge=0,
        le=10,
        description="CO2 content (%)"
    )

    # Air-fuel ratio targets
    target_excess_air_pct: float = Field(
        default=15.0,
        ge=5,
        le=50,
        description="Target excess air (%)"
    )
    min_excess_air_pct: float = Field(
        default=10.0,
        ge=3,
        le=30,
        description="Minimum safe excess air (%)"
    )
    max_excess_air_pct: float = Field(
        default=30.0,
        ge=15,
        le=100,
        description="Maximum excess air before efficiency loss (%)"
    )
    target_o2_pct: float = Field(
        default=3.0,
        ge=1.0,
        le=8.0,
        description="Target flue gas O2 (%)"
    )

    # Emission limits
    max_co_ppm: float = Field(
        default=100.0,
        ge=10,
        le=500,
        description="Maximum CO in flue gas (ppm)"
    )
    max_nox_lb_mmbtu: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Maximum NOx emissions (lb/MMBtu)"
    )
    max_sox_lb_mmbtu: float = Field(
        default=0.05,
        ge=0,
        le=0.5,
        description="Maximum SOx emissions (lb/MMBtu)"
    )

    # Burner configuration
    burners: List[BurnerConfig] = Field(
        default_factory=lambda: [BurnerConfig()],
        description="Burner configurations"
    )

    # Combustion air
    combustion_air_temp_f: float = Field(
        default=77.0,
        ge=32,
        le=600,
        description="Combustion air temperature (F)"
    )
    air_preheat_enabled: bool = Field(
        default=False,
        description="Air preheating enabled"
    )
    air_preheat_temp_f: float = Field(
        default=400.0,
        ge=200,
        le=800,
        description="Preheated air temperature (F)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# HEAT TRANSFER CONFIGURATION
# =============================================================================


class HeatTransferConfig(BaseModel):
    """
    Configuration for furnace heat transfer calculations.

    Defines heat transfer parameters, surface areas, and coefficients
    for thermal analysis per ASME PTC 4 and API 560.
    """

    # Heat transfer surfaces
    radiant_surface_area_ft2: float = Field(
        default=500.0,
        gt=0,
        description="Radiant heat transfer area (ft2)"
    )
    convective_surface_area_ft2: float = Field(
        default=1000.0,
        gt=0,
        description="Convective heat transfer area (ft2)"
    )
    total_surface_area_ft2: Optional[float] = Field(
        default=None,
        description="Total heat transfer area (ft2)"
    )

    # Heat transfer coefficients
    radiant_htc_btu_hr_ft2_f: float = Field(
        default=10.0,
        gt=0,
        le=100,
        description="Radiant heat transfer coefficient (Btu/hr-ft2-F)"
    )
    convective_htc_btu_hr_ft2_f: float = Field(
        default=5.0,
        gt=0,
        le=50,
        description="Convective heat transfer coefficient (Btu/hr-ft2-F)"
    )
    overall_htc_btu_hr_ft2_f: Optional[float] = Field(
        default=None,
        description="Overall heat transfer coefficient (Btu/hr-ft2-F)"
    )

    # Wall properties
    wall_thickness_in: float = Field(
        default=9.0,
        gt=0,
        le=24,
        description="Furnace wall thickness (inches)"
    )
    refractory_conductivity_btu_hr_ft_f: float = Field(
        default=0.5,
        gt=0,
        le=5,
        description="Refractory thermal conductivity (Btu/hr-ft-F)"
    )
    insulation_conductivity_btu_hr_ft_f: float = Field(
        default=0.03,
        gt=0,
        le=0.5,
        description="Insulation thermal conductivity (Btu/hr-ft-F)"
    )

    # Heat loss parameters
    wall_loss_pct: float = Field(
        default=2.0,
        ge=0,
        le=10,
        description="Wall heat loss (% of input)"
    )
    opening_loss_pct: float = Field(
        default=1.0,
        ge=0,
        le=5,
        description="Opening heat loss (% of input)"
    )
    other_losses_pct: float = Field(
        default=1.0,
        ge=0,
        le=5,
        description="Other heat losses (% of input)"
    )

    # Fouling factors
    fouling_factor_hr_ft2_f_btu: float = Field(
        default=0.001,
        ge=0,
        le=0.01,
        description="Fouling factor (hr-ft2-F/Btu)"
    )
    fouling_rate_per_month: float = Field(
        default=0.0001,
        ge=0,
        le=0.001,
        description="Fouling rate increase per month"
    )

    @validator("total_surface_area_ft2", always=True)
    def calculate_total_area(cls, v, values):
        """Calculate total surface area."""
        if v is None:
            radiant = values.get("radiant_surface_area_ft2", 0)
            convective = values.get("convective_surface_area_ft2", 0)
            return radiant + convective
        return v


# =============================================================================
# FURNACE OPTIMIZER CONFIGURATION
# =============================================================================


class FurnaceOptimizerConfig(BaseModel):
    """
    Comprehensive furnace optimizer configuration.

    Defines physical characteristics, operating parameters, and design
    limits for industrial furnaces per NFPA 86 and API 560 standards.

    Attributes:
        furnace_id: Unique identifier for the furnace
        furnace_type: Type classification
        design_temp_f: Design operating temperature
        design_duty_mmbtu_hr: Design heat duty
        combustion: Combustion configuration
        heat_transfer: Heat transfer configuration

    Example:
        >>> config = FurnaceOptimizerConfig(
        ...     furnace_id="FUR-001",
        ...     furnace_type=FurnaceType.DIRECT_FIRED,
        ...     design_temp_f=1800.0,
        ...     design_duty_mmbtu_hr=50.0,
        ... )
    """

    # Identification
    furnace_id: str = Field(
        ...,
        description="Unique furnace identifier"
    )
    furnace_tag: str = Field(
        default="",
        description="Plant equipment tag"
    )
    furnace_type: FurnaceType = Field(
        default=FurnaceType.DIRECT_FIRED,
        description="Furnace type classification"
    )
    service: str = Field(
        default="process_heating",
        description="Service designation"
    )

    # Design temperatures
    design_temp_f: float = Field(
        default=1800.0,
        ge=200,
        le=3000,
        description="Design operating temperature (F)"
    )
    min_operating_temp_f: float = Field(
        default=400.0,
        ge=100,
        le=1500,
        description="Minimum operating temperature (F)"
    )
    max_operating_temp_f: float = Field(
        default=2000.0,
        ge=500,
        le=3200,
        description="Maximum operating temperature (F)"
    )
    ambient_temp_f: float = Field(
        default=77.0,
        ge=-40,
        le=130,
        description="Ambient temperature (F)"
    )

    # Heat duty
    design_duty_mmbtu_hr: float = Field(
        default=50.0,
        gt=0,
        le=1000,
        description="Design heat duty (MMBtu/hr)"
    )
    min_duty_mmbtu_hr: float = Field(
        default=5.0,
        ge=0,
        description="Minimum heat duty (MMBtu/hr)"
    )
    max_duty_mmbtu_hr: float = Field(
        default=60.0,
        gt=0,
        description="Maximum heat duty (MMBtu/hr)"
    )

    # Efficiency targets
    design_efficiency_pct: float = Field(
        default=85.0,
        ge=50,
        le=99,
        description="Design thermal efficiency (%)"
    )
    min_acceptable_efficiency_pct: float = Field(
        default=75.0,
        ge=40,
        le=95,
        description="Minimum acceptable efficiency (%)"
    )
    target_efficiency_pct: float = Field(
        default=88.0,
        ge=60,
        le=99,
        description="Target optimized efficiency (%)"
    )

    # Process parameters
    process_fluid: str = Field(
        default="air",
        description="Process fluid type"
    )
    process_flow_rate_scfm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Process flow rate (SCFM)"
    )
    process_inlet_temp_f: float = Field(
        default=77.0,
        ge=-100,
        le=1000,
        description="Process fluid inlet temperature (F)"
    )

    # Tube metal temperature (for fired heaters)
    tmt_design_limit_f: float = Field(
        default=1500.0,
        ge=500,
        le=2000,
        description="Tube metal temperature design limit (F)"
    )
    tmt_alarm_f: float = Field(
        default=1450.0,
        ge=400,
        le=1900,
        description="TMT alarm setpoint (F)"
    )
    tmt_trip_f: float = Field(
        default=1500.0,
        ge=500,
        le=2000,
        description="TMT trip setpoint (F)"
    )

    # Flue gas
    flue_gas_temp_target_f: float = Field(
        default=400.0,
        ge=200,
        le=1000,
        description="Target flue gas exit temperature (F)"
    )
    stack_height_ft: float = Field(
        default=50.0,
        gt=0,
        le=500,
        description="Stack height (ft)"
    )

    # Sub-configurations
    combustion: CombustionConfig = Field(
        default_factory=CombustionConfig,
        description="Combustion configuration"
    )
    heat_transfer: HeatTransferConfig = Field(
        default_factory=HeatTransferConfig,
        description="Heat transfer configuration"
    )

    # Control parameters
    control_mode: ControlMode = Field(
        default=ControlMode.AUTOMATIC,
        description="Control mode"
    )
    temperature_deadband_f: float = Field(
        default=10.0,
        ge=1,
        le=50,
        description="Temperature control deadband (F)"
    )
    ramp_rate_f_per_min: float = Field(
        default=10.0,
        ge=1,
        le=100,
        description="Maximum temperature ramp rate (F/min)"
    )

    # Installation
    installation_date: Optional[datetime] = Field(
        default=None,
        description="Furnace installation date"
    )
    last_inspection_date: Optional[datetime] = Field(
        default=None,
        description="Last inspection date"
    )

    @validator("max_operating_temp_f")
    def validate_max_temp(cls, v, values):
        """Ensure max temp is greater than design temp."""
        if "design_temp_f" in values and v < values["design_temp_f"]:
            raise ValueError("max_operating_temp_f must be >= design_temp_f")
        return v

    @validator("tmt_trip_f")
    def validate_tmt_trip(cls, v, values):
        """Ensure TMT trip is greater than alarm."""
        if "tmt_alarm_f" in values and v < values["tmt_alarm_f"]:
            raise ValueError("tmt_trip_f must be >= tmt_alarm_f")
        return v

    class Config:
        use_enum_values = True


# =============================================================================
# COOLING TOWER CONFIGURATION
# =============================================================================


class CoolingTowerConfig(BaseModel):
    """
    Comprehensive cooling tower configuration.

    Defines physical characteristics, operating parameters, and design
    limits for cooling towers per ASHRAE and CTI standards.

    Attributes:
        tower_id: Unique identifier for the cooling tower
        tower_type: Type classification
        design_wet_bulb_f: Design wet bulb temperature
        design_range_f: Design temperature range
        design_approach_f: Design approach temperature
        design_flow_gpm: Design water flow rate

    Example:
        >>> config = CoolingTowerConfig(
        ...     tower_id="CT-001",
        ...     tower_type=CoolingTowerType.MECHANICAL_INDUCED,
        ...     design_wet_bulb_f=78.0,
        ...     design_range_f=10.0,
        ...     design_approach_f=7.0,
        ...     design_flow_gpm=5000.0,
        ... )
    """

    # Identification
    tower_id: str = Field(
        ...,
        description="Unique cooling tower identifier"
    )
    tower_tag: str = Field(
        default="",
        description="Plant equipment tag"
    )
    tower_type: CoolingTowerType = Field(
        default=CoolingTowerType.MECHANICAL_INDUCED,
        description="Cooling tower type"
    )
    fill_type: FillType = Field(
        default=FillType.FILM,
        description="Fill media type"
    )

    # Design conditions (per CTI ATC-105)
    design_wet_bulb_f: float = Field(
        default=78.0,
        ge=40,
        le=90,
        description="Design wet bulb temperature (F)"
    )
    design_dry_bulb_f: float = Field(
        default=95.0,
        ge=50,
        le=120,
        description="Design dry bulb temperature (F)"
    )
    design_hot_water_temp_f: float = Field(
        default=105.0,
        ge=70,
        le=150,
        description="Design hot water temperature (F)"
    )
    design_cold_water_temp_f: float = Field(
        default=85.0,
        ge=50,
        le=100,
        description="Design cold water temperature (F)"
    )

    # Range and Approach
    design_range_f: float = Field(
        default=10.0,
        ge=5,
        le=40,
        description="Design range (hot - cold) (F)"
    )
    design_approach_f: float = Field(
        default=7.0,
        ge=3,
        le=20,
        description="Design approach (cold - wet bulb) (F)"
    )
    min_approach_f: float = Field(
        default=5.0,
        ge=2,
        le=15,
        description="Minimum achievable approach (F)"
    )

    # Flow parameters
    design_flow_gpm: float = Field(
        default=5000.0,
        gt=0,
        le=500000,
        description="Design water flow rate (GPM)"
    )
    min_flow_gpm: float = Field(
        default=1000.0,
        ge=0,
        description="Minimum water flow (GPM)"
    )
    max_flow_gpm: float = Field(
        default=6000.0,
        gt=0,
        description="Maximum water flow (GPM)"
    )
    design_air_flow_cfm: float = Field(
        default=200000.0,
        gt=0,
        description="Design air flow rate (CFM)"
    )

    # Heat rejection
    design_heat_rejection_mmbtu_hr: float = Field(
        default=25.0,
        gt=0,
        description="Design heat rejection (MMBtu/hr)"
    )
    design_heat_rejection_tons: Optional[float] = Field(
        default=None,
        description="Design heat rejection (tons)"
    )

    # L/G ratio (liquid to gas)
    design_lg_ratio: float = Field(
        default=1.2,
        ge=0.5,
        le=3.0,
        description="Design L/G ratio (lb water/lb air)"
    )
    min_lg_ratio: float = Field(
        default=0.8,
        ge=0.3,
        le=2.0,
        description="Minimum L/G ratio"
    )
    max_lg_ratio: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Maximum L/G ratio"
    )

    # Fan parameters
    num_fans: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of fans"
    )
    fan_motor_hp: float = Field(
        default=100.0,
        gt=0,
        le=1000,
        description="Fan motor horsepower per fan"
    )
    fan_vfd_enabled: bool = Field(
        default=True,
        description="Variable frequency drive enabled"
    )
    min_fan_speed_pct: float = Field(
        default=30.0,
        ge=0,
        le=50,
        description="Minimum fan speed (%)"
    )
    max_fan_speed_pct: float = Field(
        default=100.0,
        ge=80,
        le=110,
        description="Maximum fan speed (%)"
    )

    # Fill characteristics
    fill_height_ft: float = Field(
        default=6.0,
        gt=0,
        le=20,
        description="Fill height (ft)"
    )
    fill_area_ft2: float = Field(
        default=500.0,
        gt=0,
        description="Fill cross-sectional area (ft2)"
    )
    merkel_coefficient: float = Field(
        default=1.5,
        ge=0.5,
        le=4.0,
        description="Merkel coefficient (KaV/L)"
    )

    # Water quality
    cycles_of_concentration: float = Field(
        default=5.0,
        ge=2,
        le=10,
        description="Cycles of concentration"
    )
    blowdown_pct: float = Field(
        default=2.0,
        ge=0.5,
        le=10,
        description="Blowdown percentage (%)"
    )
    makeup_water_temp_f: float = Field(
        default=70.0,
        ge=40,
        le=100,
        description="Makeup water temperature (F)"
    )
    evaporation_rate_pct: float = Field(
        default=1.0,
        ge=0.5,
        le=3.0,
        description="Evaporation rate (% of circulation)"
    )

    # Control parameters
    control_mode: ControlMode = Field(
        default=ControlMode.AUTOMATIC,
        description="Control mode"
    )
    approach_setpoint_f: float = Field(
        default=8.0,
        ge=3,
        le=20,
        description="Approach temperature setpoint (F)"
    )
    cold_water_setpoint_f: float = Field(
        default=85.0,
        ge=50,
        le=100,
        description="Cold water temperature setpoint (F)"
    )
    deadband_f: float = Field(
        default=2.0,
        ge=0.5,
        le=5,
        description="Temperature control deadband (F)"
    )

    @validator("design_heat_rejection_tons", always=True)
    def calculate_tons(cls, v, values):
        """Calculate heat rejection in tons."""
        if v is None and "design_heat_rejection_mmbtu_hr" in values:
            # 1 ton = 12,000 Btu/hr = 0.012 MMBtu/hr
            return values["design_heat_rejection_mmbtu_hr"] / 0.012
        return v

    @validator("design_range_f")
    def validate_range(cls, v, values):
        """Validate range matches hot-cold difference."""
        if "design_hot_water_temp_f" in values and "design_cold_water_temp_f" in values:
            calculated = values["design_hot_water_temp_f"] - values["design_cold_water_temp_f"]
            if abs(v - calculated) > 1.0:
                # Auto-correct based on temps
                return calculated
        return v

    class Config:
        use_enum_values = True


# =============================================================================
# ASHRAE CONFIGURATION
# =============================================================================


class ASHRAEConfig(BaseModel):
    """
    ASHRAE standard compliance configuration.

    Defines parameters for ASHRAE 90.1 energy efficiency and
    ASHRAE Handbook guidelines for cooling tower performance.
    """

    # ASHRAE 90.1 requirements
    ashrae_90_1_compliance: bool = Field(
        default=True,
        description="Enable ASHRAE 90.1 compliance checking"
    )
    climate_zone: str = Field(
        default="4A",
        description="ASHRAE climate zone"
    )
    building_type: str = Field(
        default="industrial",
        description="Building type classification"
    )

    # Cooling tower efficiency per ASHRAE 90.1
    min_tower_efficiency_gpm_hp: float = Field(
        default=42.1,
        ge=20,
        le=100,
        description="Minimum cooling tower efficiency (GPM/hp) per ASHRAE 90.1"
    )
    max_fan_power_bhp_gpm: float = Field(
        default=0.0238,
        ge=0.01,
        le=0.05,
        description="Maximum fan power (bhp/GPM)"
    )

    # Design conditions per ASHRAE Handbook
    handbook_year: int = Field(
        default=2021,
        ge=2010,
        le=2030,
        description="ASHRAE Handbook reference year"
    )
    design_db_01_pct_f: float = Field(
        default=95.0,
        ge=70,
        le=120,
        description="0.1% design dry bulb temperature (F)"
    )
    design_wb_01_pct_f: float = Field(
        default=78.0,
        ge=50,
        le=90,
        description="0.1% design wet bulb temperature (F)"
    )

    # Water treatment per ASHRAE Guideline 12
    guideline_12_compliance: bool = Field(
        default=True,
        description="Enable ASHRAE Guideline 12 compliance"
    )
    max_bacteria_cfu_ml: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Maximum bacteria count (CFU/mL)"
    )
    legionella_testing_required: bool = Field(
        default=True,
        description="Legionella testing required"
    )
    max_legionella_cfu_l: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum Legionella count (CFU/L)"
    )


# =============================================================================
# NFPA 86 SAFETY CONFIGURATION
# =============================================================================


class NFPA86Config(BaseModel):
    """
    NFPA 86 safety compliance configuration.

    Defines safety limits and interlock requirements per
    NFPA 86 Standard for Ovens and Furnaces.
    """

    # NFPA 86 compliance
    nfpa_86_compliance: bool = Field(
        default=True,
        description="Enable NFPA 86 compliance checking"
    )
    nfpa_86_edition: str = Field(
        default="2023",
        description="NFPA 86 edition year"
    )
    furnace_class: str = Field(
        default="A",
        description="Furnace class (A, B, C, D per NFPA 86)"
    )

    # Purge requirements (NFPA 86 Chapter 8)
    purge_required: bool = Field(
        default=True,
        description="Pre-ignition purge required"
    )
    min_purge_time_sec: int = Field(
        default=60,
        ge=15,
        le=600,
        description="Minimum purge time (seconds)"
    )
    min_purge_volume_changes: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Minimum air volume changes during purge"
    )

    # Safety interlocks
    flame_safety_system: bool = Field(
        default=True,
        description="Flame safety system installed"
    )
    flame_detection_type: str = Field(
        default="UV",
        description="Flame detection type (UV, IR, flame_rod)"
    )
    max_flame_failure_response_sec: float = Field(
        default=4.0,
        ge=1,
        le=10,
        description="Maximum flame failure response time (seconds)"
    )

    # Temperature limits
    max_temp_deviation_f: float = Field(
        default=50.0,
        ge=10,
        le=200,
        description="Maximum temperature deviation before alarm (F)"
    )
    high_temp_alarm_f: float = Field(
        default=1850.0,
        ge=500,
        le=3000,
        description="High temperature alarm setpoint (F)"
    )
    high_temp_trip_f: float = Field(
        default=1900.0,
        ge=500,
        le=3100,
        description="High temperature trip setpoint (F)"
    )
    low_temp_alarm_f: float = Field(
        default=300.0,
        ge=100,
        le=1000,
        description="Low temperature alarm setpoint (F)"
    )

    # Pressure/draft limits
    max_furnace_pressure_in_wc: float = Field(
        default=0.5,
        ge=-1,
        le=2,
        description="Maximum furnace pressure (inches WC)"
    )
    min_combustion_air_pressure_in_wc: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Minimum combustion air proving pressure (inches WC)"
    )

    # Explosion relief
    explosion_relief_required: bool = Field(
        default=True,
        description="Explosion relief venting required"
    )
    relief_area_ft2_per_100_ft3: float = Field(
        default=1.0,
        ge=0.5,
        le=2,
        description="Relief area per 100 ft3 of furnace volume"
    )

    # SIL rating
    sil_rating: SafetyIntegrityLevel = Field(
        default=SafetyIntegrityLevel.SIL_2,
        description="Safety Integrity Level rating"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# EXPLAINABILITY CONFIGURATION
# =============================================================================


class SHAPConfig(BaseModel):
    """SHAP explainability configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable SHAP explainability"
    )
    method: str = Field(
        default="kernel",
        description="SHAP method (kernel, tree, deep)"
    )
    n_samples: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of samples for SHAP calculation"
    )
    background_samples: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Background samples for SHAP kernel"
    )
    cache_explanations: bool = Field(
        default=True,
        description="Cache SHAP explanations"
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="SHAP cache TTL (hours)"
    )


class LIMEConfig(BaseModel):
    """LIME explainability configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable LIME explainability"
    )
    num_features: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Number of features in explanation"
    )
    num_samples: int = Field(
        default=5000,
        ge=100,
        le=20000,
        description="Number of samples for LIME"
    )
    kernel_width: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=10.0,
        description="LIME kernel width (None = auto)"
    )
    discretize_continuous: bool = Field(
        default=True,
        description="Discretize continuous features"
    )


class ExplainabilityConfig(BaseModel):
    """Complete explainability configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable explainability features"
    )
    primary_method: ExplainabilityMethod = Field(
        default=ExplainabilityMethod.SHAP,
        description="Primary explainability method"
    )
    shap: SHAPConfig = Field(
        default_factory=SHAPConfig,
        description="SHAP configuration"
    )
    lime: LIMEConfig = Field(
        default_factory=LIMEConfig,
        description="LIME configuration"
    )
    auto_explain_on_anomaly: bool = Field(
        default=True,
        description="Generate explanation on anomaly detection"
    )
    top_features_to_report: int = Field(
        default=10,
        ge=3,
        le=30,
        description="Number of top features to report"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# PROVENANCE CONFIGURATION
# =============================================================================


class ProvenanceConfig(BaseModel):
    """Provenance tracking configuration for audit trails."""

    enabled: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    hash_algorithm: str = Field(
        default="sha256",
        description="Hash algorithm (sha256, sha384, sha512)"
    )
    hash_inputs: bool = Field(
        default=True,
        description="Hash all input data"
    )
    hash_outputs: bool = Field(
        default=True,
        description="Hash all output data"
    )
    hash_intermediate: bool = Field(
        default=False,
        description="Hash intermediate calculation results"
    )
    track_data_sources: bool = Field(
        default=True,
        description="Track all data sources"
    )
    include_timestamps: bool = Field(
        default=True,
        description="Include timestamps in provenance"
    )
    retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Provenance data retention (days)"
    )
    compliance_mode: str = Field(
        default="standard",
        description="Compliance mode (standard, strict, audit)"
    )


# =============================================================================
# INTEGRATION CONFIGURATION
# =============================================================================


class OPCUANodeConfig(BaseModel):
    """Configuration for an OPC-UA node mapping."""

    node_id: str = Field(
        ...,
        description="OPC-UA node ID"
    )
    tag_name: str = Field(
        ...,
        description="Local tag name"
    )
    data_type: str = Field(
        default="Double",
        description="Data type"
    )
    access_level: str = Field(
        default="read",
        description="Access level (read, write, read_write)"
    )
    sampling_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Sampling interval (ms)"
    )


class OPCUAConfig(BaseModel):
    """OPC-UA integration configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable OPC-UA integration"
    )
    endpoint: str = Field(
        default="opc.tcp://localhost:4840/greenlang/",
        description="OPC-UA server endpoint"
    )
    security_policy: OPCSecurityPolicy = Field(
        default=OPCSecurityPolicy.BASIC256SHA256,
        description="OPC-UA security policy"
    )
    security_mode: OPCSecurityMode = Field(
        default=OPCSecurityMode.SIGN_AND_ENCRYPT,
        description="OPC-UA security mode"
    )
    timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=120000,
        description="Connection timeout (ms)"
    )
    nodes: List[OPCUANodeConfig] = Field(
        default_factory=list,
        description="OPC-UA node configurations"
    )

    class Config:
        use_enum_values = True


class MQTTConfig(BaseModel):
    """MQTT integration configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable MQTT integration"
    )
    broker_host: str = Field(
        default="localhost",
        description="MQTT broker hostname"
    )
    broker_port: int = Field(
        default=1883,
        ge=1,
        le=65535,
        description="MQTT broker port"
    )
    client_id: str = Field(
        default="gl007-furnace-optimizer",
        description="MQTT client ID"
    )
    username: Optional[str] = Field(
        default=None,
        description="MQTT username"
    )
    use_tls: bool = Field(
        default=False,
        description="Use TLS encryption"
    )
    qos: MQTTQoS = Field(
        default=MQTTQoS.AT_LEAST_ONCE,
        description="MQTT QoS level"
    )
    topic_prefix: str = Field(
        default="greenlang/furnace",
        description="MQTT topic prefix"
    )
    publish_interval_sec: int = Field(
        default=5,
        ge=1,
        le=3600,
        description="Publish interval (seconds)"
    )

    class Config:
        use_enum_values = True


class IntegrationConfig(BaseModel):
    """Complete integration configuration."""

    opcua: OPCUAConfig = Field(
        default_factory=OPCUAConfig,
        description="OPC-UA configuration"
    )
    mqtt: MQTTConfig = Field(
        default_factory=MQTTConfig,
        description="MQTT configuration"
    )


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================


class GL007Config(BaseModel):
    """
    Master configuration for GL-007 Furnace and Cooling Tower Optimizer.

    This configuration combines all component configurations for
    comprehensive furnace and cooling tower optimization.

    Attributes:
        furnace: Furnace optimizer configuration
        cooling_tower: Cooling tower configuration
        ashrae: ASHRAE compliance configuration
        nfpa86: NFPA 86 safety configuration
        explainability: SHAP/LIME explainability settings
        provenance: Provenance tracking configuration
        integration: OPC-UA/MQTT integration settings

    Example:
        >>> config = GL007Config(
        ...     furnace=FurnaceOptimizerConfig(furnace_id="FUR-001"),
        ...     cooling_tower=CoolingTowerConfig(tower_id="CT-001"),
        ... )

    Standards Compliance:
        - NFPA 86: Standard for Ovens and Furnaces
        - ASHRAE 90.1: Energy Standard for Buildings
        - ASHRAE Handbook: HVAC Systems and Equipment
        - API 560: Fired Heaters for General Refinery Service
        - CTI ATC-105: Acceptance Test Code for Cooling Towers
    """

    # Component configurations
    furnace: Optional[FurnaceOptimizerConfig] = Field(
        default=None,
        description="Furnace optimizer configuration"
    )
    cooling_tower: Optional[CoolingTowerConfig] = Field(
        default=None,
        description="Cooling tower configuration"
    )
    ashrae: ASHRAEConfig = Field(
        default_factory=ASHRAEConfig,
        description="ASHRAE compliance configuration"
    )
    nfpa86: NFPA86Config = Field(
        default_factory=NFPA86Config,
        description="NFPA 86 safety configuration"
    )
    explainability: ExplainabilityConfig = Field(
        default_factory=ExplainabilityConfig,
        description="Explainability configuration"
    )
    provenance: ProvenanceConfig = Field(
        default_factory=ProvenanceConfig,
        description="Provenance tracking configuration"
    )
    integration: IntegrationConfig = Field(
        default_factory=IntegrationConfig,
        description="Integration configuration"
    )

    # Agent identification
    agent_id: str = Field(
        default="GL-007",
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="FurnaceOptimizer/CoolingTowerOptimizer",
        description="Agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Configuration version"
    )

    # Performance settings
    calculation_precision: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Decimal precision for calculations"
    )
    optimization_interval_sec: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Optimization interval (seconds)"
    )
    data_collection_interval_sec: int = Field(
        default=1,
        ge=1,
        le=60,
        description="Data collection interval (seconds)"
    )

    # Data management
    data_retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Operational data retention (days)"
    )
    trend_history_days: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Trend analysis history (days)"
    )

    @root_validator(skip_on_failure=True)
    def validate_at_least_one_equipment(cls, values):
        """Ensure at least one equipment type is configured."""
        furnace = values.get("furnace")
        cooling_tower = values.get("cooling_tower")
        if furnace is None and cooling_tower is None:
            # Create default furnace if neither specified
            values["furnace"] = FurnaceOptimizerConfig(furnace_id="FUR-001")
        return values

    class Config:
        use_enum_values = True
        validate_assignment = True


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_default_config(
    furnace_id: str = "FUR-001",
    tower_id: str = "CT-001",
) -> GL007Config:
    """
    Create a default GL-007 configuration with typical industrial values.

    Args:
        furnace_id: Unique furnace identifier
        tower_id: Unique cooling tower identifier

    Returns:
        GL007Config with typical industrial settings
    """
    return GL007Config(
        furnace=FurnaceOptimizerConfig(
            furnace_id=furnace_id,
            furnace_type=FurnaceType.DIRECT_FIRED,
            design_temp_f=1800.0,
            design_duty_mmbtu_hr=50.0,
            design_efficiency_pct=85.0,
        ),
        cooling_tower=CoolingTowerConfig(
            tower_id=tower_id,
            tower_type=CoolingTowerType.MECHANICAL_INDUCED,
            design_wet_bulb_f=78.0,
            design_range_f=10.0,
            design_approach_f=7.0,
            design_flow_gpm=5000.0,
        ),
    )


def create_high_efficiency_config(
    furnace_id: str = "FUR-001",
    tower_id: str = "CT-001",
) -> GL007Config:
    """
    Create configuration optimized for high efficiency operation.

    Returns:
        GL007Config optimized for high efficiency
    """
    return GL007Config(
        furnace=FurnaceOptimizerConfig(
            furnace_id=furnace_id,
            furnace_type=FurnaceType.RECUPERATIVE,
            design_temp_f=1800.0,
            design_duty_mmbtu_hr=50.0,
            design_efficiency_pct=92.0,
            target_efficiency_pct=94.0,
            combustion=CombustionConfig(
                target_excess_air_pct=10.0,
                target_o2_pct=2.0,
                air_preheat_enabled=True,
                air_preheat_temp_f=500.0,
            ),
        ),
        cooling_tower=CoolingTowerConfig(
            tower_id=tower_id,
            tower_type=CoolingTowerType.COUNTERFLOW,
            design_wet_bulb_f=78.0,
            design_range_f=10.0,
            design_approach_f=5.0,
            design_flow_gpm=5000.0,
            fan_vfd_enabled=True,
        ),
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "FurnaceType",
    "FuelType",
    "CoolingTowerType",
    "FillType",
    "ControlMode",
    "SafetyIntegrityLevel",
    "AlertSeverity",
    "ExplainabilityMethod",
    "OPCSecurityPolicy",
    "OPCSecurityMode",
    "MQTTQoS",
    # Configuration Classes
    "BurnerConfig",
    "CombustionConfig",
    "HeatTransferConfig",
    "FurnaceOptimizerConfig",
    "CoolingTowerConfig",
    "ASHRAEConfig",
    "NFPA86Config",
    "SHAPConfig",
    "LIMEConfig",
    "ExplainabilityConfig",
    "ProvenanceConfig",
    "OPCUANodeConfig",
    "OPCUAConfig",
    "MQTTConfig",
    "IntegrationConfig",
    "GL007Config",
    # Factory Functions
    "create_default_config",
    "create_high_efficiency_config",
]
