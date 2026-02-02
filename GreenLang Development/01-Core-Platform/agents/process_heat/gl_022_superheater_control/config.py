# -*- coding: utf-8 -*-
"""
GL-022 SuperheaterControlAgent - Configuration Module

This module defines all configuration schemas for the Superheater Control Agent,
including superheater design parameters, desuperheater spray control, temperature
control PID/cascade settings, process demand integration, safety limits and
interlocks, SHAP/LIME explainability settings, provenance tracking, and
OPC-UA/Kafka integration configurations.

Configuration follows GreenLang patterns with Pydantic validation and
sensible defaults for industrial superheater applications.

Standards Compliance:
    - ASME Section I: Power Boilers (Superheater Design)
    - ASME B31.1: Power Piping
    - ASME PTC 4.1: Steam Generating Units
    - API 530: Calculation of Heater Tube Thickness
    - API 560: Fired Heaters for General Refinery Service
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - ISA 84: Safety Instrumented Systems (SIL ratings)

Example:
    >>> from greenlang.agents.process_heat.gl_022_superheater_control.config import (
    ...     GL022Config,
    ...     SuperheaterDesignConfig,
    ...     DesuperheaterControlConfig,
    ...     TemperatureControlConfig,
    ... )
    >>> config = GL022Config(
    ...     superheater=SuperheaterDesignConfig(
    ...         superheater_id="SH-001",
    ...         design_outlet_temp_f=950.0,
    ...         design_pressure_psig=600.0,
    ...     ),
    ...     desuperheater=DesuperheaterControlConfig(
    ...         desuperheater_id="DSH-001",
    ...         spray_type=SprayType.WATER_SPRAY,
    ...     ),
    ... )

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# ENUMS - SUPERHEATER AND CONTROL CLASSIFICATIONS
# =============================================================================


class SuperheaterType(str, Enum):
    """
    Types of industrial superheaters.

    Classification based on heat transfer mechanism and configuration:
    - RADIANT: Direct radiant heat transfer from furnace
    - CONVECTION: Convective heat transfer from flue gas
    - RADIANT_CONVECTION: Combined radiant and convection sections
    - PENDANT: Pendant-style tube arrangement
    - HORIZONTAL: Horizontal tube arrangement
    - PLATEN: Platen-style arrangement for high temperature
    """
    RADIANT = "radiant"
    CONVECTION = "convection"
    RADIANT_CONVECTION = "radiant_convection"
    PENDANT = "pendant"
    HORIZONTAL = "horizontal"
    PLATEN = "platen"
    REHEATER = "reheater"
    SECONDARY = "secondary"
    FINAL = "final"


class TubeMaterial(str, Enum):
    """
    Superheater tube materials per ASME Section II.

    Selection based on operating temperature and pressure requirements:
    - Carbon steel: Up to 800F
    - Chrome-moly alloys: Up to 1100F
    - Stainless steels: Up to 1200F+
    """
    # Carbon steel
    SA_178_A = "sa_178_a"  # Carbon steel ERW tube
    SA_192 = "sa_192"  # Seamless carbon steel
    SA_210_A1 = "sa_210_a1"  # Medium carbon steel seamless

    # Chrome-moly alloys
    SA_213_T11 = "sa_213_t11"  # 1.25Cr-0.5Mo (up to 1025F)
    SA_213_T22 = "sa_213_t22"  # 2.25Cr-1Mo (up to 1075F)
    SA_213_T91 = "sa_213_t91"  # 9Cr-1Mo-V (up to 1200F)
    SA_213_T92 = "sa_213_t92"  # 9Cr-2W (up to 1200F)

    # Stainless steels
    SA_213_TP304H = "sa_213_tp304h"  # 18Cr-8Ni
    SA_213_TP321H = "sa_213_tp321h"  # 18Cr-10Ni-Ti
    SA_213_TP347H = "sa_213_tp347h"  # 18Cr-10Ni-Nb

    # Austenitic alloys
    ALLOY_617 = "alloy_617"  # Ni-Cr-Co-Mo
    ALLOY_625 = "alloy_625"  # Ni-Cr-Mo-Nb
    ALLOY_740H = "alloy_740h"  # Advanced Ni-base

    # Other
    CUSTOM = "custom"


class SprayType(str, Enum):
    """
    Types of desuperheating spray systems.

    Selection based on steam conditions and control requirements:
    - WATER_SPRAY: Direct water injection (most common)
    - STEAM_ATOMIZING: Steam-assisted atomization
    - SURFACE_CONTACT: Surface contact cooling
    - VENTURI: Venturi-type mixing
    """
    WATER_SPRAY = "water_spray"
    STEAM_ATOMIZING = "steam_atomizing"
    SURFACE_CONTACT = "surface_contact"
    VENTURI = "venturi"
    SPRAY_RING = "spray_ring"
    VARIABLE_AREA = "variable_area"
    DUAL_NOZZLE = "dual_nozzle"


class ControlMode(str, Enum):
    """Temperature control operating modes."""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    CASCADE = "cascade"
    OPTIMIZING = "optimizing"
    FEEDFORWARD = "feedforward"
    ADAPTIVE = "adaptive"
    MODEL_PREDICTIVE = "model_predictive"


class ControlAction(str, Enum):
    """Controller action type."""
    DIRECT = "direct"  # Increase output increases PV
    REVERSE = "reverse"  # Increase output decreases PV


class SafetyIntegrityLevel(str, Enum):
    """IEC 61508 / ISA 84 Safety Integrity Levels."""
    SIL_1 = "sil_1"  # PFD 0.1 to 0.01
    SIL_2 = "sil_2"  # PFD 0.01 to 0.001
    SIL_3 = "sil_3"  # PFD 0.001 to 0.0001
    SIL_4 = "sil_4"  # PFD 0.0001 to 0.00001
    NON_SIL = "non_sil"


class AlertSeverity(str, Enum):
    """Alert severity levels for superheater monitoring."""
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
    SHAP_DEEP = "shap_deep"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    ATTENTION = "attention"
    COUNTERFACTUAL = "counterfactual"


class OPCSecurityPolicy(str, Enum):
    """OPC-UA security policies."""
    NONE = "None"
    BASIC128RSA15 = "Basic128Rsa15"
    BASIC256 = "Basic256"
    BASIC256SHA256 = "Basic256Sha256"
    AES128_SHA256_RSAOAEP = "Aes128_Sha256_RsaOaep"
    AES256_SHA256_RSAPSS = "Aes256_Sha256_RsaPss"


class OPCSecurityMode(str, Enum):
    """OPC-UA security modes."""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class KafkaSecurityProtocol(str, Enum):
    """Kafka security protocols."""
    PLAINTEXT = "PLAINTEXT"
    SSL = "SSL"
    SASL_PLAINTEXT = "SASL_PLAINTEXT"
    SASL_SSL = "SASL_SSL"


# =============================================================================
# SUPERHEATER DESIGN CONFIGURATION
# =============================================================================


class TubeSpecificationConfig(BaseModel):
    """Superheater tube specifications per ASME standards."""

    tube_material: TubeMaterial = Field(
        default=TubeMaterial.SA_213_T22,
        description="Tube material specification per ASME Section II"
    )
    tube_od_in: float = Field(
        default=2.0,
        gt=0,
        le=6,
        description="Tube outside diameter (inches)"
    )
    tube_wall_thickness_in: float = Field(
        default=0.165,
        gt=0,
        le=1,
        description="Tube wall thickness per API 530 (inches)"
    )
    tube_id_in: Optional[float] = Field(
        default=None,
        gt=0,
        description="Tube inside diameter (auto-calculated if None)"
    )
    tube_length_ft: float = Field(
        default=30.0,
        gt=0,
        le=100,
        description="Effective tube length (feet)"
    )
    num_tubes: int = Field(
        default=100,
        gt=0,
        description="Number of tubes"
    )
    num_passes: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Number of steam passes"
    )
    tube_pitch_in: float = Field(
        default=4.0,
        gt=0,
        description="Tube pitch (inches)"
    )
    tube_arrangement: str = Field(
        default="staggered",
        description="Tube arrangement (inline, staggered)"
    )

    # Temperature limits per material
    max_tube_metal_temp_f: float = Field(
        default=1050.0,
        gt=0,
        le=1400,
        description="Maximum allowable tube metal temperature per ASME (F)"
    )
    design_metal_temp_f: float = Field(
        default=1000.0,
        gt=0,
        le=1400,
        description="Design tube metal temperature (F)"
    )

    # Creep and stress
    design_stress_psi: Optional[float] = Field(
        default=None,
        gt=0,
        description="Design stress per ASME Section I (psi)"
    )
    corrosion_allowance_in: float = Field(
        default=0.05,
        ge=0,
        le=0.25,
        description="Corrosion allowance (inches)"
    )

    @validator("tube_id_in", always=True)
    def calculate_tube_id(cls, v, values):
        """Calculate tube ID from OD and wall thickness."""
        if v is None and "tube_od_in" in values and "tube_wall_thickness_in" in values:
            return values["tube_od_in"] - 2 * values["tube_wall_thickness_in"]
        return v

    class Config:
        use_enum_values = True


class SuperheaterDesignConfig(BaseModel):
    """
    Comprehensive superheater design configuration.

    Defines physical characteristics, operating parameters, and design
    limits for industrial superheaters per ASME Section I standards.

    Attributes:
        superheater_id: Unique identifier for the superheater
        superheater_type: Type classification (radiant, convection, etc.)
        design_outlet_temp_f: Target outlet steam temperature
        design_pressure_psig: Design operating pressure
        tube_spec: Tube material and dimensional specifications

    Temperature Ranges (Typical):
        - Standard industrial: 750-900F
        - High efficiency power: 900-1050F
        - Ultra-supercritical: 1100-1200F+

    Example:
        >>> config = SuperheaterDesignConfig(
        ...     superheater_id="SH-001",
        ...     superheater_type=SuperheaterType.CONVECTION,
        ...     design_outlet_temp_f=950.0,
        ...     design_pressure_psig=600.0,
        ... )
    """

    # Identification
    superheater_id: str = Field(
        ...,
        description="Unique superheater identifier"
    )
    superheater_tag: str = Field(
        default="",
        description="Plant equipment tag (e.g., SH-101)"
    )
    superheater_type: SuperheaterType = Field(
        default=SuperheaterType.CONVECTION,
        description="Superheater type classification"
    )
    service: str = Field(
        default="primary",
        description="Service designation (primary, secondary, reheat)"
    )

    # Design temperatures
    design_inlet_temp_f: float = Field(
        default=500.0,
        ge=200,
        le=1000,
        description="Design inlet steam temperature (F)"
    )
    design_outlet_temp_f: float = Field(
        default=950.0,
        ge=400,
        le=1200,
        description="Design outlet steam temperature (F)"
    )
    min_outlet_temp_f: float = Field(
        default=850.0,
        ge=300,
        le=1100,
        description="Minimum acceptable outlet temperature (F)"
    )
    max_outlet_temp_f: float = Field(
        default=1050.0,
        ge=500,
        le=1250,
        description="Maximum allowable outlet temperature (F)"
    )
    target_superheat_f: Optional[float] = Field(
        default=None,
        ge=0,
        le=500,
        description="Target superheat above saturation (F)"
    )

    # Design pressures per ASME
    design_pressure_psig: float = Field(
        default=600.0,
        ge=15,
        le=4000,
        description="Design operating pressure (psig)"
    )
    max_allowable_working_pressure_psig: float = Field(
        default=700.0,
        ge=15,
        le=4500,
        description="Maximum Allowable Working Pressure per ASME (psig)"
    )
    min_pressure_psig: float = Field(
        default=400.0,
        ge=0,
        description="Minimum operating pressure (psig)"
    )
    pressure_drop_design_psi: float = Field(
        default=30.0,
        ge=0,
        le=100,
        description="Design pressure drop across superheater (psi)"
    )

    # Flow rates
    design_steam_flow_lb_hr: float = Field(
        default=100000.0,
        gt=0,
        description="Design steam flow rate (lb/hr)"
    )
    max_steam_flow_lb_hr: float = Field(
        default=120000.0,
        gt=0,
        description="Maximum steam flow capacity (lb/hr)"
    )
    min_steam_flow_lb_hr: float = Field(
        default=30000.0,
        ge=0,
        description="Minimum steam flow for safe operation (lb/hr)"
    )

    # Heat transfer
    design_duty_mmbtu_hr: float = Field(
        default=50.0,
        gt=0,
        description="Design heat duty (MMBtu/hr)"
    )
    design_heat_flux_btu_hr_ft2: float = Field(
        default=15000.0,
        gt=0,
        le=100000,
        description="Design heat flux (Btu/hr-ft2)"
    )
    max_heat_flux_btu_hr_ft2: float = Field(
        default=25000.0,
        gt=0,
        le=150000,
        description="Maximum allowable heat flux (Btu/hr-ft2)"
    )
    total_surface_area_ft2: float = Field(
        default=5000.0,
        gt=0,
        description="Total heat transfer surface area (ft2)"
    )

    # Flue gas conditions
    flue_gas_inlet_temp_f: float = Field(
        default=1600.0,
        ge=500,
        le=3000,
        description="Flue gas inlet temperature (F)"
    )
    flue_gas_outlet_temp_f: float = Field(
        default=1000.0,
        ge=300,
        le=2000,
        description="Flue gas outlet temperature (F)"
    )
    flue_gas_velocity_fps: float = Field(
        default=60.0,
        gt=0,
        le=150,
        description="Flue gas velocity (ft/s)"
    )

    # Tube specifications
    tube_spec: TubeSpecificationConfig = Field(
        default_factory=TubeSpecificationConfig,
        description="Tube specification configuration"
    )

    # Installation
    installation_date: Optional[datetime] = Field(
        default=None,
        description="Superheater installation date"
    )
    last_inspection_date: Optional[datetime] = Field(
        default=None,
        description="Last tube inspection date"
    )
    expected_lifetime_hours: float = Field(
        default=200000.0,
        gt=0,
        description="Expected superheater lifetime (operating hours)"
    )

    @validator("design_outlet_temp_f")
    def validate_outlet_temp(cls, v, values):
        """Ensure outlet temp is greater than inlet temp."""
        if "design_inlet_temp_f" in values and v <= values["design_inlet_temp_f"]:
            raise ValueError(
                "design_outlet_temp_f must be greater than design_inlet_temp_f"
            )
        return v

    @validator("max_outlet_temp_f")
    def validate_max_outlet_temp(cls, v, values):
        """Ensure max outlet temp is within material limits."""
        if "tube_spec" in values and values["tube_spec"]:
            max_metal = values["tube_spec"].max_tube_metal_temp_f
            if v > max_metal + 50:  # Allow 50F difference between steam and metal
                raise ValueError(
                    f"max_outlet_temp_f ({v}F) exceeds tube metal limit ({max_metal}F)"
                )
        return v

    @validator("max_allowable_working_pressure_psig")
    def validate_mawp(cls, v, values):
        """Ensure MAWP is greater than design pressure."""
        if "design_pressure_psig" in values and v < values["design_pressure_psig"]:
            raise ValueError(
                "MAWP must be >= design_pressure_psig"
            )
        return v

    class Config:
        use_enum_values = True


# =============================================================================
# DESUPERHEATER SPRAY CONTROL CONFIGURATION
# =============================================================================


class SprayWaterQualityConfig(BaseModel):
    """
    Spray water quality requirements per ASME standards.

    Poor spray water quality leads to deposits, erosion, and thermal shock.

    Quality Requirements:
        - TDS < 2 ppm (preferably < 0.5 ppm)
        - Silica < 0.02 ppm
        - Hardness: essentially zero
        - pH: 9.0-9.5 (matching steam cycle)
    """

    max_tds_ppm: float = Field(
        default=2.0,
        ge=0,
        le=10,
        description="Maximum Total Dissolved Solids (ppm)"
    )
    target_tds_ppm: float = Field(
        default=0.5,
        ge=0,
        le=5,
        description="Target TDS level (ppm)"
    )
    max_silica_ppm: float = Field(
        default=0.02,
        ge=0,
        le=0.5,
        description="Maximum silica content (ppm)"
    )
    max_hardness_ppm: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Maximum hardness as CaCO3 (ppm)"
    )
    max_iron_ppb: float = Field(
        default=10.0,
        ge=0,
        le=100,
        description="Maximum iron content (ppb)"
    )
    max_copper_ppb: float = Field(
        default=5.0,
        ge=0,
        le=50,
        description="Maximum copper content (ppb)"
    )
    target_ph_min: float = Field(
        default=9.0,
        ge=7.0,
        le=10.0,
        description="Minimum acceptable pH"
    )
    target_ph_max: float = Field(
        default=9.5,
        ge=7.5,
        le=11.0,
        description="Maximum acceptable pH"
    )
    max_dissolved_o2_ppb: float = Field(
        default=7.0,
        ge=0,
        le=50,
        description="Maximum dissolved oxygen (ppb)"
    )

    # Source specification
    water_source: str = Field(
        default="condensate",
        description="Spray water source (condensate, demin, feedwater)"
    )
    quality_monitoring_enabled: bool = Field(
        default=True,
        description="Enable online water quality monitoring"
    )
    quality_check_interval_s: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Quality check interval (seconds)"
    )


class SprayNozzleConfig(BaseModel):
    """Configuration for desuperheater spray nozzle specifications."""

    nozzle_id: str = Field(
        default="primary",
        description="Nozzle identifier"
    )
    nozzle_type: str = Field(
        default="variable_area",
        description="Nozzle type (fixed, variable_area, spring_loaded)"
    )
    nozzle_count: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Number of spray nozzles"
    )
    orifice_diameter_in: float = Field(
        default=0.25,
        gt=0,
        le=1,
        description="Nozzle orifice diameter (inches)"
    )
    spray_angle_deg: float = Field(
        default=60.0,
        ge=15,
        le=120,
        description="Spray angle (degrees)"
    )
    cv_rated: float = Field(
        default=5.0,
        gt=0,
        description="Rated flow coefficient (Cv)"
    )
    max_flow_gpm: float = Field(
        default=50.0,
        gt=0,
        description="Maximum spray flow (GPM)"
    )
    min_flow_gpm: float = Field(
        default=5.0,
        ge=0,
        description="Minimum spray flow (GPM)"
    )
    droplet_size_microns: float = Field(
        default=100.0,
        gt=0,
        le=500,
        description="Mean droplet size (microns)"
    )
    atomization_type: str = Field(
        default="pressure",
        description="Atomization method (pressure, steam_assist, mechanical)"
    )
    material: str = Field(
        default="stainless_316",
        description="Nozzle material"
    )


class DesuperheaterControlConfig(BaseModel):
    """
    Configuration for desuperheater spray control system.

    Defines spray water system parameters, control limits, and
    performance thresholds for steam attemperation.

    Attributes:
        desuperheater_id: Unique desuperheater identifier
        spray_type: Type of desuperheating system
        spray_water_temp_f: Spray water temperature
        max_spray_rate_lb_hr: Maximum spray water flow
        min_approach_temp_f: Minimum approach to saturation

    Design Considerations:
        - Approach temperature: minimum 20-30F above saturation
        - Spray velocity: typically 50-150 ft/s
        - Pressure differential: min 50 psi above steam pressure

    Example:
        >>> config = DesuperheaterControlConfig(
        ...     desuperheater_id="DSH-001",
        ...     spray_type=SprayType.WATER_SPRAY,
        ...     max_spray_rate_lb_hr=10000.0,
        ... )
    """

    # Identification
    desuperheater_id: str = Field(
        ...,
        description="Unique desuperheater identifier"
    )
    desuperheater_tag: str = Field(
        default="",
        description="Plant equipment tag (e.g., DSH-101)"
    )
    spray_type: SprayType = Field(
        default=SprayType.WATER_SPRAY,
        description="Desuperheating spray type"
    )
    location: str = Field(
        default="interstage",
        description="Location (interstage, outlet, attemperator_station)"
    )

    # Spray water conditions
    spray_water_temp_f: float = Field(
        default=250.0,
        ge=50,
        le=450,
        description="Spray water temperature (F)"
    )
    spray_water_pressure_psig: float = Field(
        default=750.0,
        gt=0,
        description="Spray water supply pressure (psig)"
    )
    min_pressure_differential_psi: float = Field(
        default=50.0,
        ge=20,
        le=200,
        description="Minimum spray water pressure above steam (psi)"
    )

    # Flow capacity
    max_spray_rate_lb_hr: float = Field(
        default=10000.0,
        gt=0,
        description="Maximum spray water rate (lb/hr)"
    )
    min_spray_rate_lb_hr: float = Field(
        default=500.0,
        ge=0,
        description="Minimum controllable spray rate (lb/hr)"
    )
    design_spray_rate_lb_hr: float = Field(
        default=5000.0,
        gt=0,
        description="Design spray rate for normal operation (lb/hr)"
    )
    max_spray_pct_of_steam: float = Field(
        default=10.0,
        ge=0,
        le=20,
        description="Maximum spray as % of steam flow"
    )

    # Temperature control
    min_approach_temp_f: float = Field(
        default=25.0,
        ge=10,
        le=75,
        description="Minimum approach to saturation temperature (F)"
    )
    target_approach_temp_f: float = Field(
        default=50.0,
        ge=20,
        le=100,
        description="Target approach to saturation (F)"
    )

    # Attemperation limits
    max_temp_reduction_f: float = Field(
        default=150.0,
        ge=0,
        le=300,
        description="Maximum temperature reduction per stage (F)"
    )
    max_rate_of_change_f_min: float = Field(
        default=10.0,
        ge=1,
        le=50,
        description="Maximum rate of temperature change (F/min)"
    )

    # Thermal shock protection
    thermal_shock_enabled: bool = Field(
        default=True,
        description="Enable thermal shock protection"
    )
    max_thermal_gradient_f_in: float = Field(
        default=100.0,
        ge=10,
        le=300,
        description="Maximum allowable thermal gradient (F/inch)"
    )
    quench_detection_enabled: bool = Field(
        default=True,
        description="Enable quench condition detection"
    )
    quench_temp_threshold_f: float = Field(
        default=400.0,
        ge=200,
        le=600,
        description="Temperature threshold for quench alarm (F)"
    )

    # Spray water quality
    spray_water_quality: SprayWaterQualityConfig = Field(
        default_factory=SprayWaterQualityConfig,
        description="Spray water quality requirements"
    )

    # Spray nozzle configuration
    spray_nozzle: SprayNozzleConfig = Field(
        default_factory=SprayNozzleConfig,
        description="Spray nozzle configuration"
    )

    # Control valve
    control_valve_cv: float = Field(
        default=50.0,
        gt=0,
        description="Control valve Cv rating"
    )
    control_valve_rangeability: float = Field(
        default=50.0,
        ge=10,
        le=100,
        description="Control valve rangeability"
    )
    valve_type: str = Field(
        default="globe",
        description="Valve type (globe, cage, ball)"
    )
    valve_characteristic: str = Field(
        default="equal_percent",
        description="Valve characteristic (linear, equal_percent, quick_open)"
    )

    # Evaporation distance
    min_evaporation_distance_ft: float = Field(
        default=10.0,
        ge=3,
        le=50,
        description="Minimum straight pipe for evaporation (feet)"
    )
    max_evaporation_time_ms: float = Field(
        default=500.0,
        ge=100,
        le=2000,
        description="Maximum droplet evaporation time (milliseconds)"
    )

    @validator("spray_water_pressure_psig")
    def validate_spray_pressure(cls, v, values):
        """Validate spray water pressure vs steam pressure."""
        # This will be validated at runtime against actual steam pressure
        return v

    class Config:
        use_enum_values = True


# =============================================================================
# TEMPERATURE CONTROL CONFIGURATION
# =============================================================================


class PIDTuningConfig(BaseModel):
    """
    PID controller tuning parameters for temperature control.

    Provides tuning parameters for master and slave controllers
    in cascade temperature control schemes.

    Tuning Guidelines:
        - Superheater temperature: slow process (large time constant)
        - Typical: Kp=1-5, Ti=60-300s, Td=0-60s
        - Use derivative sparingly due to measurement noise
    """

    # Proportional settings
    kp: float = Field(
        default=2.0,
        ge=0.1,
        le=50,
        description="Proportional gain (Kp)"
    )
    proportional_band_pct: Optional[float] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Proportional band (% = 100/Kp)"
    )

    # Integral settings
    ti_seconds: float = Field(
        default=120.0,
        ge=1,
        le=3600,
        description="Integral time (seconds)"
    )
    ki: Optional[float] = Field(
        default=None,
        ge=0,
        description="Integral gain (ki = 1/Ti)"
    )
    integral_windup_limit_pct: float = Field(
        default=100.0,
        ge=0,
        le=200,
        description="Integral windup limit (%)"
    )

    # Derivative settings
    td_seconds: float = Field(
        default=15.0,
        ge=0,
        le=600,
        description="Derivative time (seconds)"
    )
    kd: Optional[float] = Field(
        default=None,
        ge=0,
        description="Derivative gain"
    )
    derivative_filter_coeff: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Derivative filter coefficient (N)"
    )

    # Output limits
    output_min_pct: float = Field(
        default=0.0,
        ge=-100,
        le=100,
        description="Minimum controller output (%)"
    )
    output_max_pct: float = Field(
        default=100.0,
        ge=0,
        le=200,
        description="Maximum controller output (%)"
    )
    rate_limit_pct_s: float = Field(
        default=10.0,
        ge=0,
        le=100,
        description="Output rate limit (%/s)"
    )

    # Control action
    control_action: ControlAction = Field(
        default=ControlAction.REVERSE,
        description="Controller action (direct/reverse)"
    )

    # Dead band
    deadband_f: float = Field(
        default=2.0,
        ge=0,
        le=20,
        description="Control deadband (F)"
    )

    # Setpoint tracking
    setpoint_rate_limit_f_min: float = Field(
        default=5.0,
        ge=0,
        le=30,
        description="Setpoint rate of change limit (F/min)"
    )

    @validator("proportional_band_pct", always=True)
    def calculate_pb(cls, v, values):
        """Calculate proportional band from Kp."""
        if v is None and "kp" in values:
            return 100.0 / values["kp"]
        return v

    @validator("ki", always=True)
    def calculate_ki(cls, v, values):
        """Calculate Ki from Ti."""
        if v is None and "ti_seconds" in values and values["ti_seconds"] > 0:
            return 1.0 / values["ti_seconds"]
        return v

    class Config:
        use_enum_values = True


class CascadeControlConfig(BaseModel):
    """
    Cascade control configuration for superheater temperature.

    Implements primary (temperature) to secondary (spray flow)
    cascade control architecture for optimal temperature regulation.

    Architecture:
        Primary (Master): Steam temperature PID
        Secondary (Slave): Spray flow PID
    """

    enabled: bool = Field(
        default=True,
        description="Enable cascade control"
    )

    # Master (temperature) controller
    master_pid: PIDTuningConfig = Field(
        default_factory=lambda: PIDTuningConfig(
            kp=2.0,
            ti_seconds=180.0,
            td_seconds=30.0,
            deadband_f=3.0,
        ),
        description="Master (temperature) controller tuning"
    )
    master_sample_time_s: float = Field(
        default=5.0,
        ge=1,
        le=60,
        description="Master controller sample time (seconds)"
    )

    # Slave (spray flow) controller
    slave_pid: PIDTuningConfig = Field(
        default_factory=lambda: PIDTuningConfig(
            kp=1.0,
            ti_seconds=30.0,
            td_seconds=5.0,
            deadband_f=0.5,
        ),
        description="Slave (spray flow) controller tuning"
    )
    slave_sample_time_s: float = Field(
        default=1.0,
        ge=0.1,
        le=10,
        description="Slave controller sample time (seconds)"
    )

    # Cascade coordination
    master_output_min_pct: float = Field(
        default=0.0,
        ge=-50,
        le=100,
        description="Master output minimum (slave setpoint min) (%)"
    )
    master_output_max_pct: float = Field(
        default=100.0,
        ge=0,
        le=150,
        description="Master output maximum (slave setpoint max) (%)"
    )
    bumpless_transfer_enabled: bool = Field(
        default=True,
        description="Enable bumpless transfer on mode changes"
    )
    tracking_enabled: bool = Field(
        default=True,
        description="Enable slave-to-master tracking"
    )

    # Auto-tuning
    auto_tune_enabled: bool = Field(
        default=False,
        description="Enable adaptive auto-tuning"
    )
    auto_tune_interval_hours: int = Field(
        default=168,
        ge=24,
        le=720,
        description="Auto-tuning interval (hours)"
    )


class FeedforwardControlConfig(BaseModel):
    """
    Feedforward control configuration for load disturbances.

    Implements feedforward compensation for measurable load
    disturbances to improve temperature regulation.

    Common Feedforward Signals:
        - Steam flow rate changes
        - Fuel firing rate changes
        - Flue gas temperature changes
    """

    enabled: bool = Field(
        default=True,
        description="Enable feedforward control"
    )

    # Steam flow feedforward
    steam_flow_ff_enabled: bool = Field(
        default=True,
        description="Enable steam flow feedforward"
    )
    steam_flow_ff_gain: float = Field(
        default=1.0,
        ge=0,
        le=5,
        description="Steam flow feedforward gain"
    )
    steam_flow_ff_lead_s: float = Field(
        default=30.0,
        ge=0,
        le=120,
        description="Steam flow feedforward lead time (seconds)"
    )
    steam_flow_ff_lag_s: float = Field(
        default=60.0,
        ge=1,
        le=300,
        description="Steam flow feedforward lag time (seconds)"
    )

    # Fuel firing feedforward
    fuel_firing_ff_enabled: bool = Field(
        default=True,
        description="Enable fuel firing rate feedforward"
    )
    fuel_firing_ff_gain: float = Field(
        default=0.8,
        ge=0,
        le=5,
        description="Fuel firing feedforward gain"
    )
    fuel_firing_ff_delay_s: float = Field(
        default=30.0,
        ge=0,
        le=120,
        description="Fuel firing feedforward delay (seconds)"
    )

    # Flue gas temperature feedforward
    flue_gas_ff_enabled: bool = Field(
        default=False,
        description="Enable flue gas temperature feedforward"
    )
    flue_gas_ff_gain: float = Field(
        default=0.5,
        ge=0,
        le=3,
        description="Flue gas temperature feedforward gain"
    )

    # Dynamic compensation
    lead_lag_ratio: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Lead/lag ratio for dynamic compensation"
    )
    feedforward_limit_pct: float = Field(
        default=30.0,
        ge=0,
        le=100,
        description="Maximum feedforward contribution (%)"
    )


class TemperatureControlConfig(BaseModel):
    """
    Complete temperature control configuration for superheater.

    Integrates PID tuning, cascade control, feedforward compensation,
    and adaptive control features for optimal steam temperature regulation.

    Attributes:
        control_mode: Operating control mode
        setpoint_f: Temperature setpoint
        cascade: Cascade control configuration
        feedforward: Feedforward control configuration

    Example:
        >>> config = TemperatureControlConfig(
        ...     control_mode=ControlMode.CASCADE,
        ...     setpoint_f=950.0,
        ...     tolerance_f=5.0,
        ... )
    """

    control_mode: ControlMode = Field(
        default=ControlMode.CASCADE,
        description="Operating control mode"
    )
    enabled: bool = Field(
        default=True,
        description="Enable temperature control"
    )

    # Setpoint configuration
    setpoint_f: float = Field(
        default=950.0,
        ge=400,
        le=1200,
        description="Temperature setpoint (F)"
    )
    setpoint_min_f: float = Field(
        default=800.0,
        ge=300,
        le=1100,
        description="Minimum allowable setpoint (F)"
    )
    setpoint_max_f: float = Field(
        default=1050.0,
        ge=500,
        le=1250,
        description="Maximum allowable setpoint (F)"
    )
    tolerance_f: float = Field(
        default=5.0,
        ge=1,
        le=25,
        description="Temperature tolerance band (F)"
    )

    # Alarm thresholds
    high_temp_warning_f: float = Field(
        default=1000.0,
        ge=500,
        le=1200,
        description="High temperature warning (F)"
    )
    high_temp_alarm_f: float = Field(
        default=1025.0,
        ge=500,
        le=1250,
        description="High temperature alarm (F)"
    )
    high_temp_trip_f: float = Field(
        default=1050.0,
        ge=500,
        le=1300,
        description="High temperature trip (F)"
    )
    low_temp_warning_f: float = Field(
        default=850.0,
        ge=300,
        le=1000,
        description="Low temperature warning (F)"
    )
    low_temp_alarm_f: float = Field(
        default=800.0,
        ge=250,
        le=950,
        description="Low temperature alarm (F)"
    )

    # Rate of change limits (thermal shock prevention)
    max_rate_up_f_min: float = Field(
        default=10.0,
        ge=1,
        le=30,
        description="Maximum heating rate (F/min)"
    )
    max_rate_down_f_min: float = Field(
        default=15.0,
        ge=1,
        le=50,
        description="Maximum cooling rate (F/min)"
    )
    rate_alarm_f_min: float = Field(
        default=20.0,
        ge=5,
        le=50,
        description="Rate of change alarm threshold (F/min)"
    )

    # Control sub-configurations
    cascade: CascadeControlConfig = Field(
        default_factory=CascadeControlConfig,
        description="Cascade control configuration"
    )
    feedforward: FeedforwardControlConfig = Field(
        default_factory=FeedforwardControlConfig,
        description="Feedforward control configuration"
    )

    # Measurement configuration
    measurement_filter_s: float = Field(
        default=5.0,
        ge=0,
        le=60,
        description="Temperature measurement filter time constant (seconds)"
    )
    thermocouple_type: str = Field(
        default="K",
        description="Thermocouple type (K, N, J, R, S)"
    )
    redundant_sensors: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of redundant temperature sensors"
    )
    sensor_voting: str = Field(
        default="median",
        description="Sensor voting logic (median, average, min, max)"
    )

    # Optimization settings
    optimization_enabled: bool = Field(
        default=True,
        description="Enable temperature optimization"
    )
    optimization_interval_s: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Optimization interval (seconds)"
    )

    @validator("setpoint_f")
    def validate_setpoint(cls, v, values):
        """Ensure setpoint is within limits."""
        if "setpoint_min_f" in values and v < values["setpoint_min_f"]:
            raise ValueError("setpoint_f must be >= setpoint_min_f")
        if "setpoint_max_f" in values and v > values["setpoint_max_f"]:
            raise ValueError("setpoint_f must be <= setpoint_max_f")
        return v

    @validator("high_temp_alarm_f")
    def validate_alarm_thresholds(cls, v, values):
        """Ensure alarm thresholds are properly ordered."""
        if "high_temp_warning_f" in values and v <= values["high_temp_warning_f"]:
            raise ValueError("high_temp_alarm_f must be > high_temp_warning_f")
        return v

    class Config:
        use_enum_values = True


# =============================================================================
# PROCESS DEMAND INTEGRATION CONFIGURATION
# =============================================================================


class ProcessDemandConfig(BaseModel):
    """
    Configuration for process demand integration and load following.

    Coordinates superheater operation with downstream process demands
    including turbines, process headers, and plant-wide optimization.

    Attributes:
        load_following_enabled: Enable load following mode
        demand_sources: List of demand signal sources
        anticipatory_control: Enable demand anticipation
    """

    load_following_enabled: bool = Field(
        default=True,
        description="Enable load following mode"
    )

    # Demand signal sources
    demand_sources: List[str] = Field(
        default_factory=lambda: ["turbine", "process_header"],
        description="List of demand signal sources"
    )
    primary_demand_source: str = Field(
        default="turbine",
        description="Primary demand signal source"
    )

    # Load range
    min_load_pct: float = Field(
        default=30.0,
        ge=0,
        le=100,
        description="Minimum operating load (%)"
    )
    max_load_pct: float = Field(
        default=100.0,
        ge=50,
        le=120,
        description="Maximum operating load (%)"
    )
    normal_load_pct: float = Field(
        default=85.0,
        ge=30,
        le=110,
        description="Normal operating load (%)"
    )

    # Load rate limits
    max_load_rate_pct_min: float = Field(
        default=5.0,
        ge=0.5,
        le=20,
        description="Maximum load rate of change (%/min)"
    )
    startup_rate_pct_min: float = Field(
        default=2.0,
        ge=0.1,
        le=10,
        description="Startup load rate (%/min)"
    )
    shutdown_rate_pct_min: float = Field(
        default=3.0,
        ge=0.1,
        le=10,
        description="Shutdown load rate (%/min)"
    )

    # Anticipatory control
    anticipatory_control_enabled: bool = Field(
        default=True,
        description="Enable demand anticipation"
    )
    anticipation_time_s: float = Field(
        default=60.0,
        ge=0,
        le=300,
        description="Demand anticipation time (seconds)"
    )
    demand_prediction_enabled: bool = Field(
        default=False,
        description="Enable ML-based demand prediction"
    )
    prediction_horizon_s: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Demand prediction horizon (seconds)"
    )

    # Turbine coordination
    turbine_coordination_enabled: bool = Field(
        default=True,
        description="Enable turbine inlet condition coordination"
    )
    turbine_inlet_temp_target_f: float = Field(
        default=950.0,
        ge=500,
        le=1200,
        description="Target turbine inlet temperature (F)"
    )
    turbine_temp_tolerance_f: float = Field(
        default=10.0,
        ge=1,
        le=30,
        description="Turbine temperature tolerance (F)"
    )

    # Header pressure coordination
    header_pressure_target_psig: float = Field(
        default=600.0,
        ge=50,
        le=3000,
        description="Target header pressure (psig)"
    )
    header_pressure_tolerance_psi: float = Field(
        default=5.0,
        ge=1,
        le=50,
        description="Header pressure tolerance (psi)"
    )

    # Process constraints
    min_steam_quality: float = Field(
        default=0.995,
        ge=0.95,
        le=1.0,
        description="Minimum required steam quality (dryness)"
    )
    superheat_margin_f: float = Field(
        default=50.0,
        ge=20,
        le=150,
        description="Minimum superheat margin (F)"
    )


# =============================================================================
# SAFETY LIMITS AND INTERLOCKS CONFIGURATION
# =============================================================================


class SafetyLimitsConfig(BaseModel):
    """
    Safety limits configuration per ASME and NFPA standards.

    Defines alarm and trip setpoints for safe superheater operation.
    """

    # High temperature limits
    tube_metal_high_warning_f: float = Field(
        default=1025.0,
        ge=500,
        le=1300,
        description="Tube metal high temperature warning (F)"
    )
    tube_metal_high_alarm_f: float = Field(
        default=1050.0,
        ge=500,
        le=1350,
        description="Tube metal high temperature alarm (F)"
    )
    tube_metal_high_trip_f: float = Field(
        default=1075.0,
        ge=500,
        le=1400,
        description="Tube metal high temperature trip (F)"
    )

    # Steam temperature limits
    steam_outlet_high_trip_f: float = Field(
        default=1050.0,
        ge=500,
        le=1300,
        description="Steam outlet high temperature trip (F)"
    )
    steam_outlet_low_trip_f: float = Field(
        default=700.0,
        ge=300,
        le=900,
        description="Steam outlet low temperature trip (F)"
    )

    # Rate of change limits
    rate_of_change_trip_f_min: float = Field(
        default=25.0,
        ge=10,
        le=50,
        description="Temperature rate of change trip (F/min)"
    )

    # Pressure limits
    high_pressure_warning_psig: float = Field(
        default=650.0,
        ge=100,
        le=4000,
        description="High pressure warning (psig)"
    )
    high_pressure_alarm_psig: float = Field(
        default=680.0,
        ge=100,
        le=4200,
        description="High pressure alarm (psig)"
    )
    high_pressure_trip_psig: float = Field(
        default=700.0,
        ge=100,
        le=4500,
        description="High pressure trip (psig)"
    )
    low_pressure_warning_psig: float = Field(
        default=450.0,
        ge=0,
        le=2000,
        description="Low pressure warning (psig)"
    )
    low_pressure_alarm_psig: float = Field(
        default=400.0,
        ge=0,
        le=1800,
        description="Low pressure alarm (psig)"
    )

    # Flow limits
    low_flow_warning_pct: float = Field(
        default=35.0,
        ge=0,
        le=100,
        description="Low steam flow warning (% of design)"
    )
    low_flow_alarm_pct: float = Field(
        default=30.0,
        ge=0,
        le=100,
        description="Low steam flow alarm (% of design)"
    )
    low_flow_trip_pct: float = Field(
        default=25.0,
        ge=0,
        le=100,
        description="Low steam flow trip (% of design)"
    )

    # Thermal differential limits
    max_thermal_differential_f: float = Field(
        default=75.0,
        ge=20,
        le=200,
        description="Maximum tube-to-tube temperature differential (F)"
    )
    thermal_differential_alarm_f: float = Field(
        default=100.0,
        ge=30,
        le=250,
        description="Thermal differential alarm (F)"
    )


class InterlockConfig(BaseModel):
    """
    Interlock configuration for superheater protection.

    Defines safety interlock logic and timing per NFPA 85 and ISA 84.
    """

    # Interlock enables
    high_temp_interlock_enabled: bool = Field(
        default=True,
        description="Enable high temperature interlock"
    )
    low_flow_interlock_enabled: bool = Field(
        default=True,
        description="Enable low steam flow interlock"
    )
    high_pressure_interlock_enabled: bool = Field(
        default=True,
        description="Enable high pressure interlock"
    )
    rate_of_change_interlock_enabled: bool = Field(
        default=True,
        description="Enable rate of change interlock"
    )
    desuperheater_interlock_enabled: bool = Field(
        default=True,
        description="Enable desuperheater protection interlock"
    )

    # Response times
    high_temp_response_s: float = Field(
        default=5.0,
        ge=1,
        le=30,
        description="High temperature interlock response (seconds)"
    )
    low_flow_response_s: float = Field(
        default=10.0,
        ge=1,
        le=60,
        description="Low flow interlock response (seconds)"
    )
    rate_of_change_response_s: float = Field(
        default=30.0,
        ge=5,
        le=120,
        description="Rate of change interlock response (seconds)"
    )

    # Permissives
    spray_permissive_min_flow_pct: float = Field(
        default=30.0,
        ge=10,
        le=50,
        description="Minimum steam flow for spray permissive (%)"
    )
    spray_permissive_min_temp_f: float = Field(
        default=600.0,
        ge=300,
        le=900,
        description="Minimum steam temp for spray permissive (F)"
    )
    spray_permissive_max_superheat_f: float = Field(
        default=200.0,
        ge=50,
        le=400,
        description="Maximum superheat for spray permissive (F)"
    )

    # Interlock bypass (requires authorization)
    bypass_authorization_required: bool = Field(
        default=True,
        description="Require authorization for interlock bypass"
    )
    max_bypass_duration_hours: float = Field(
        default=8.0,
        ge=1,
        le=24,
        description="Maximum bypass duration (hours)"
    )


class SafetyConfig(BaseModel):
    """
    Complete safety configuration for superheater control.

    Integrates safety limits, interlocks, and SIS requirements
    per ASME, NFPA 85, and ISA 84 standards.

    Attributes:
        sil_rating: Safety Integrity Level
        limits: Safety limit configuration
        interlocks: Interlock configuration
        bms_integration: BMS integration enabled

    Example:
        >>> config = SafetyConfig(
        ...     sil_rating=SafetyIntegrityLevel.SIL_2,
        ...     limits=SafetyLimitsConfig(),
        ... )
    """

    # SIS/SIL configuration
    sil_rating: SafetyIntegrityLevel = Field(
        default=SafetyIntegrityLevel.SIL_2,
        description="Safety Integrity Level rating"
    )
    sis_proof_test_interval_months: int = Field(
        default=12,
        ge=1,
        le=60,
        description="SIS proof test interval (months)"
    )
    pfd_target: float = Field(
        default=0.001,
        gt=0,
        lt=1,
        description="Probability of Failure on Demand target"
    )

    # Sub-configurations
    limits: SafetyLimitsConfig = Field(
        default_factory=SafetyLimitsConfig,
        description="Safety limits configuration"
    )
    interlocks: InterlockConfig = Field(
        default_factory=InterlockConfig,
        description="Interlock configuration"
    )

    # Standards compliance
    asme_section_i_compliance: bool = Field(
        default=True,
        description="Enable ASME Section I compliance"
    )
    nfpa_85_compliance: bool = Field(
        default=True,
        description="Enable NFPA 85 compliance"
    )
    api_560_compliance: bool = Field(
        default=False,
        description="Enable API 560 compliance (fired heaters)"
    )

    # BMS integration
    bms_integration_enabled: bool = Field(
        default=True,
        description="Enable BMS integration"
    )
    bms_trip_on_high_temp: bool = Field(
        default=True,
        description="Trip boiler on superheater high temperature"
    )
    bms_trip_on_low_flow: bool = Field(
        default=True,
        description="Trip boiler on superheater low flow"
    )

    # Redundancy
    sensor_redundancy: str = Field(
        default="2oo3",
        description="Sensor voting logic (1oo1, 1oo2, 2oo3)"
    )
    actuator_redundancy: bool = Field(
        default=False,
        description="Redundant actuators installed"
    )

    # Audit trail
    safety_event_logging: bool = Field(
        default=True,
        description="Enable safety event logging"
    )
    safety_event_retention_days: int = Field(
        default=365,
        ge=30,
        description="Safety event log retention (days)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# SHAP/LIME EXPLAINABILITY CONFIGURATION
# =============================================================================


class SHAPConfig(BaseModel):
    """
    SHAP (SHapley Additive exPlanations) configuration.

    Configures SHAP-based feature importance analysis for
    temperature prediction and control optimization models.
    """

    enabled: bool = Field(
        default=True,
        description="Enable SHAP explainability"
    )
    method: str = Field(
        default="kernel",
        description="SHAP method (kernel, tree, deep, linear)"
    )
    n_samples: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of samples for SHAP calculation"
    )
    feature_perturbation: str = Field(
        default="interventional",
        description="Perturbation method (interventional, observational)"
    )
    interaction_analysis: bool = Field(
        default=False,
        description="Enable SHAP interaction analysis"
    )
    max_interactions: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum feature interactions to analyze"
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
    background_samples: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Background samples for SHAP kernel"
    )


class LIMEConfig(BaseModel):
    """
    LIME (Local Interpretable Model-agnostic Explanations) configuration.

    Configures LIME-based local explanations for individual
    predictions in the superheater control system.
    """

    enabled: bool = Field(
        default=True,
        description="Enable LIME explainability"
    )
    mode: str = Field(
        default="tabular",
        description="LIME mode (tabular, text, image)"
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
    discretizer: str = Field(
        default="quartile",
        description="Discretization method (quartile, decile, entropy)"
    )
    feature_selection: str = Field(
        default="lasso_path",
        description="Feature selection method (lasso_path, forward_selection, auto)"
    )
    model_regressor: Optional[str] = Field(
        default=None,
        description="Local surrogate model (None = ridge regression)"
    )


class ExplainabilityConfig(BaseModel):
    """
    Complete explainability configuration for ML-based control.

    Integrates SHAP and LIME configurations for comprehensive
    model explanation capabilities.

    Attributes:
        enabled: Master enable for explainability
        primary_method: Primary explainability method
        shap: SHAP configuration
        lime: LIME configuration

    Example:
        >>> config = ExplainabilityConfig(
        ...     enabled=True,
        ...     primary_method=ExplainabilityMethod.SHAP,
        ... )
    """

    enabled: bool = Field(
        default=True,
        description="Enable explainability features"
    )
    primary_method: ExplainabilityMethod = Field(
        default=ExplainabilityMethod.SHAP,
        description="Primary explainability method"
    )

    # Method configurations
    shap: SHAPConfig = Field(
        default_factory=SHAPConfig,
        description="SHAP configuration"
    )
    lime: LIMEConfig = Field(
        default_factory=LIMEConfig,
        description="LIME configuration"
    )

    # Explanation generation
    auto_explain_enabled: bool = Field(
        default=True,
        description="Auto-generate explanations for predictions"
    )
    explain_threshold_change_f: float = Field(
        default=10.0,
        ge=1,
        le=50,
        description="Temperature change threshold for explanation (F)"
    )
    explain_on_anomaly: bool = Field(
        default=True,
        description="Generate explanation on anomaly detection"
    )

    # Feature importance
    feature_importance_enabled: bool = Field(
        default=True,
        description="Calculate global feature importance"
    )
    importance_update_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Feature importance update interval (hours)"
    )
    top_features_to_report: int = Field(
        default=10,
        ge=5,
        le=30,
        description="Number of top features to report"
    )

    # Counterfactual explanations
    counterfactual_enabled: bool = Field(
        default=False,
        description="Enable counterfactual explanations"
    )
    counterfactual_method: str = Field(
        default="dice",
        description="Counterfactual method (dice, wachter, growing_spheres)"
    )
    num_counterfactuals: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of counterfactual examples"
    )

    # Attention visualization (for transformer models)
    attention_visualization: bool = Field(
        default=False,
        description="Enable attention visualization"
    )

    # Reporting
    explanation_format: str = Field(
        default="json",
        description="Explanation output format (json, html, text)"
    )
    include_confidence_intervals: bool = Field(
        default=True,
        description="Include confidence intervals in explanations"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# PROVENANCE TRACKING CONFIGURATION
# =============================================================================


class ProvenanceConfig(BaseModel):
    """
    Provenance tracking configuration for audit trail compliance.

    Implements SHA-256 hashing for complete data lineage and
    reproducibility in regulatory environments.

    Features:
        - Input/output hashing for all calculations
        - Configuration version tracking
        - Data source traceability
        - Calculation timestamp recording
    """

    enabled: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )

    # Hashing configuration
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
    hash_configs: bool = Field(
        default=True,
        description="Hash configuration at runtime"
    )
    hash_intermediate: bool = Field(
        default=False,
        description="Hash intermediate calculation results"
    )

    # Data lineage
    track_data_sources: bool = Field(
        default=True,
        description="Track all data sources"
    )
    track_transformations: bool = Field(
        default=True,
        description="Track data transformations"
    )
    track_model_versions: bool = Field(
        default=True,
        description="Track ML model versions"
    )

    # Timestamps
    include_timestamps: bool = Field(
        default=True,
        description="Include timestamps in provenance"
    )
    timestamp_format: str = Field(
        default="iso8601",
        description="Timestamp format (iso8601, unix, rfc3339)"
    )
    timezone: str = Field(
        default="UTC",
        description="Timezone for timestamps"
    )

    # Storage
    store_provenance: bool = Field(
        default=True,
        description="Store provenance records"
    )
    retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Provenance data retention (days)"
    )
    storage_backend: str = Field(
        default="database",
        description="Storage backend (database, file, blockchain)"
    )

    # Verification
    verify_on_read: bool = Field(
        default=False,
        description="Verify provenance on data read"
    )
    alert_on_mismatch: bool = Field(
        default=True,
        description="Alert on provenance mismatch"
    )

    # Regulatory compliance
    compliance_mode: str = Field(
        default="standard",
        description="Compliance mode (standard, strict, audit)"
    )
    digital_signature_enabled: bool = Field(
        default=False,
        description="Enable digital signatures"
    )
    certificate_path: Optional[str] = Field(
        default=None,
        description="Path to signing certificate"
    )


# =============================================================================
# OPC-UA INTEGRATION CONFIGURATION
# =============================================================================


class OPCUANodeConfig(BaseModel):
    """Configuration for an OPC-UA node mapping."""

    node_id: str = Field(
        ...,
        description="OPC-UA node ID (e.g., ns=2;s=Superheater.OutletTemp)"
    )
    tag_name: str = Field(
        ...,
        description="Local tag name"
    )
    data_type: str = Field(
        default="Double",
        description="Data type (Double, Float, Int32, Boolean, String)"
    )
    access_level: str = Field(
        default="read",
        description="Access level (read, write, read_write)"
    )
    sampling_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Sampling interval (milliseconds)"
    )
    dead_band: float = Field(
        default=0.0,
        ge=0,
        description="Dead band for value change detection"
    )
    engineering_units: str = Field(
        default="",
        description="Engineering units"
    )


class OPCUAConfig(BaseModel):
    """
    OPC-UA integration configuration.

    Configures OPC-UA client connection for real-time data
    acquisition from industrial control systems.

    Attributes:
        enabled: Enable OPC-UA integration
        endpoint: OPC-UA server endpoint URL
        security_policy: OPC-UA security policy
        security_mode: OPC-UA security mode
        nodes: List of node configurations

    Example:
        >>> config = OPCUAConfig(
        ...     enabled=True,
        ...     endpoint="opc.tcp://plc.plant.local:4840",
        ...     security_policy=OPCSecurityPolicy.BASIC256SHA256,
        ... )
    """

    enabled: bool = Field(
        default=False,
        description="Enable OPC-UA integration"
    )

    # Connection settings
    endpoint: str = Field(
        default="opc.tcp://localhost:4840/greenlang/",
        description="OPC-UA server endpoint URL"
    )
    namespace_uri: str = Field(
        default="urn:greenlang:superheater",
        description="Namespace URI"
    )
    application_name: str = Field(
        default="GL022-SuperheaterControl",
        description="OPC-UA application name"
    )
    application_uri: str = Field(
        default="urn:greenlang:gl022:superheater",
        description="OPC-UA application URI"
    )

    # Security
    security_policy: OPCSecurityPolicy = Field(
        default=OPCSecurityPolicy.BASIC256SHA256,
        description="OPC-UA security policy"
    )
    security_mode: OPCSecurityMode = Field(
        default=OPCSecurityMode.SIGN_AND_ENCRYPT,
        description="OPC-UA security mode"
    )
    certificate_path: Optional[str] = Field(
        default=None,
        description="Client certificate path"
    )
    private_key_path: Optional[str] = Field(
        default=None,
        description="Private key path"
    )
    username: Optional[str] = Field(
        default=None,
        description="Username for authentication"
    )
    password_secret_name: Optional[str] = Field(
        default=None,
        description="Secret name for password (use secrets manager)"
    )

    # Connection parameters
    timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=120000,
        description="Connection timeout (milliseconds)"
    )
    keepalive_interval_ms: int = Field(
        default=10000,
        ge=1000,
        le=60000,
        description="Keepalive interval (milliseconds)"
    )
    reconnect_interval_ms: int = Field(
        default=5000,
        ge=1000,
        le=60000,
        description="Reconnect interval (milliseconds)"
    )
    max_reconnect_attempts: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum reconnection attempts"
    )

    # Subscription settings
    subscription_enabled: bool = Field(
        default=True,
        description="Enable OPC-UA subscriptions"
    )
    publishing_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Subscription publishing interval (milliseconds)"
    )
    max_notifications_per_publish: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum notifications per publish"
    )
    lifetime_count: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Subscription lifetime count"
    )
    max_keep_alive_count: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum keep alive count"
    )

    # Node configurations
    nodes: List[OPCUANodeConfig] = Field(
        default_factory=lambda: [
            OPCUANodeConfig(
                node_id="ns=2;s=Superheater.OutletTemp",
                tag_name="sh_outlet_temp_f",
                engineering_units="F",
            ),
            OPCUANodeConfig(
                node_id="ns=2;s=Superheater.InletTemp",
                tag_name="sh_inlet_temp_f",
                engineering_units="F",
            ),
            OPCUANodeConfig(
                node_id="ns=2;s=Superheater.SteamFlow",
                tag_name="steam_flow_lb_hr",
                engineering_units="lb/hr",
            ),
            OPCUANodeConfig(
                node_id="ns=2;s=Superheater.Pressure",
                tag_name="steam_pressure_psig",
                engineering_units="psig",
            ),
            OPCUANodeConfig(
                node_id="ns=2;s=Desuperheater.SprayFlow",
                tag_name="spray_flow_lb_hr",
                engineering_units="lb/hr",
            ),
            OPCUANodeConfig(
                node_id="ns=2;s=Desuperheater.ValvePosition",
                tag_name="spray_valve_pct",
                access_level="read_write",
                engineering_units="%",
            ),
        ],
        description="OPC-UA node configurations"
    )

    # Historical data
    historical_access_enabled: bool = Field(
        default=True,
        description="Enable historical data access"
    )
    historical_query_timeout_ms: int = Field(
        default=60000,
        ge=5000,
        le=300000,
        description="Historical query timeout (milliseconds)"
    )

    # Method calls
    method_calls_enabled: bool = Field(
        default=True,
        description="Enable OPC-UA method calls"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# KAFKA INTEGRATION CONFIGURATION
# =============================================================================


class KafkaTopicConfig(BaseModel):
    """Configuration for a Kafka topic."""

    topic_name: str = Field(
        ...,
        description="Kafka topic name"
    )
    direction: str = Field(
        default="consume",
        description="Direction (consume, produce, both)"
    )
    partition_count: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of partitions"
    )
    replication_factor: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Replication factor"
    )
    retention_hours: int = Field(
        default=168,
        ge=1,
        le=8760,
        description="Message retention (hours)"
    )
    key_serializer: str = Field(
        default="string",
        description="Key serializer (string, json, avro)"
    )
    value_serializer: str = Field(
        default="json",
        description="Value serializer (string, json, avro)"
    )


class KafkaConfig(BaseModel):
    """
    Kafka integration configuration.

    Configures Kafka producer/consumer for event streaming
    and real-time data pipeline integration.

    Attributes:
        enabled: Enable Kafka integration
        bootstrap_servers: Kafka broker addresses
        security_protocol: Security protocol
        topics: List of topic configurations

    Example:
        >>> config = KafkaConfig(
        ...     enabled=True,
        ...     bootstrap_servers=["kafka1:9092", "kafka2:9092"],
        ... )
    """

    enabled: bool = Field(
        default=False,
        description="Enable Kafka integration"
    )

    # Connection settings
    bootstrap_servers: List[str] = Field(
        default_factory=lambda: ["localhost:9092"],
        description="Kafka bootstrap server addresses"
    )
    client_id: str = Field(
        default="gl022-superheater-control",
        description="Kafka client ID"
    )
    group_id: str = Field(
        default="gl022-superheater-group",
        description="Consumer group ID"
    )

    # Security
    security_protocol: KafkaSecurityProtocol = Field(
        default=KafkaSecurityProtocol.PLAINTEXT,
        description="Kafka security protocol"
    )
    ssl_ca_location: Optional[str] = Field(
        default=None,
        description="SSL CA certificate location"
    )
    ssl_certificate_location: Optional[str] = Field(
        default=None,
        description="SSL client certificate location"
    )
    ssl_key_location: Optional[str] = Field(
        default=None,
        description="SSL client key location"
    )
    sasl_mechanism: Optional[str] = Field(
        default=None,
        description="SASL mechanism (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512)"
    )
    sasl_username: Optional[str] = Field(
        default=None,
        description="SASL username"
    )
    sasl_password_secret_name: Optional[str] = Field(
        default=None,
        description="Secret name for SASL password"
    )

    # Producer settings
    producer_acks: str = Field(
        default="all",
        description="Producer acknowledgment level (0, 1, all)"
    )
    producer_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Producer retry attempts"
    )
    producer_batch_size: int = Field(
        default=16384,
        ge=0,
        le=1048576,
        description="Producer batch size (bytes)"
    )
    producer_linger_ms: int = Field(
        default=10,
        ge=0,
        le=1000,
        description="Producer linger time (milliseconds)"
    )
    producer_compression_type: str = Field(
        default="gzip",
        description="Compression type (none, gzip, snappy, lz4, zstd)"
    )
    enable_idempotence: bool = Field(
        default=True,
        description="Enable idempotent producer"
    )

    # Consumer settings
    auto_offset_reset: str = Field(
        default="latest",
        description="Auto offset reset (earliest, latest, none)"
    )
    enable_auto_commit: bool = Field(
        default=False,
        description="Enable auto commit"
    )
    max_poll_records: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Maximum records per poll"
    )
    session_timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=60000,
        description="Session timeout (milliseconds)"
    )
    heartbeat_interval_ms: int = Field(
        default=3000,
        ge=500,
        le=10000,
        description="Heartbeat interval (milliseconds)"
    )

    # Topics
    topics: List[KafkaTopicConfig] = Field(
        default_factory=lambda: [
            KafkaTopicConfig(
                topic_name="greenlang.superheater.measurements",
                direction="produce",
            ),
            KafkaTopicConfig(
                topic_name="greenlang.superheater.control",
                direction="produce",
            ),
            KafkaTopicConfig(
                topic_name="greenlang.superheater.alarms",
                direction="produce",
            ),
            KafkaTopicConfig(
                topic_name="greenlang.process.demand",
                direction="consume",
            ),
        ],
        description="Kafka topic configurations"
    )

    # Schema registry
    schema_registry_enabled: bool = Field(
        default=False,
        description="Enable schema registry"
    )
    schema_registry_url: Optional[str] = Field(
        default=None,
        description="Schema registry URL"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# DATA HISTORIAN CONFIGURATION
# =============================================================================


class HistorianConfig(BaseModel):
    """
    Data historian configuration for time-series data storage.

    Supports OSIsoft PI, Aspen InfoPlus.21, Wonderware Historian,
    and InfluxDB backends.
    """

    enabled: bool = Field(
        default=True,
        description="Enable historian integration"
    )
    historian_type: str = Field(
        default="influxdb",
        description="Historian type (pi, ip21, wonderware, influxdb)"
    )
    connection_string: Optional[str] = Field(
        default=None,
        description="Historian connection string"
    )

    # Tag configuration
    tag_prefix: str = Field(
        default="GL022.SH",
        description="Historian tag prefix"
    )

    # Write settings
    write_enabled: bool = Field(
        default=True,
        description="Enable writing to historian"
    )
    write_interval_s: int = Field(
        default=1,
        ge=1,
        le=3600,
        description="Write interval (seconds)"
    )
    compression_enabled: bool = Field(
        default=True,
        description="Enable data compression"
    )
    compression_deviation: float = Field(
        default=0.1,
        ge=0,
        le=10,
        description="Compression deviation threshold"
    )

    # Read settings
    read_enabled: bool = Field(
        default=True,
        description="Enable reading from historian"
    )
    max_query_duration_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Maximum query duration (hours)"
    )


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================


class GL022Config(BaseModel):
    """
    Master configuration for GL-022 SuperheaterControlAgent.

    This configuration combines all component configurations for
    comprehensive superheater temperature control and optimization.

    Attributes:
        superheater: Superheater design configuration
        desuperheater: Desuperheater spray control configuration
        temperature_control: Temperature control configuration
        process_demand: Process demand integration
        safety: Safety limits and interlocks
        explainability: SHAP/LIME explainability settings
        provenance: Provenance tracking configuration
        opcua: OPC-UA integration settings
        kafka: Kafka integration settings
        historian: Data historian configuration

    Example:
        >>> config = GL022Config(
        ...     superheater=SuperheaterDesignConfig(
        ...         superheater_id="SH-001",
        ...         design_outlet_temp_f=950.0,
        ...     ),
        ...     desuperheater=DesuperheaterControlConfig(
        ...         desuperheater_id="DSH-001",
        ...     ),
        ... )

    Standards Compliance:
        - ASME Section I: Power Boilers
        - ASME B31.1: Power Piping
        - ASME PTC 4.1: Steam Generating Units
        - API 530: Heater Tube Thickness
        - NFPA 85: Boiler Hazards Code
        - ISA 84: Safety Instrumented Systems
    """

    # Component configurations
    superheater: SuperheaterDesignConfig = Field(
        ...,
        description="Superheater design configuration"
    )
    desuperheater: DesuperheaterControlConfig = Field(
        ...,
        description="Desuperheater control configuration"
    )
    temperature_control: TemperatureControlConfig = Field(
        default_factory=TemperatureControlConfig,
        description="Temperature control configuration"
    )
    process_demand: ProcessDemandConfig = Field(
        default_factory=ProcessDemandConfig,
        description="Process demand integration"
    )
    safety: SafetyConfig = Field(
        default_factory=SafetyConfig,
        description="Safety configuration"
    )
    explainability: ExplainabilityConfig = Field(
        default_factory=ExplainabilityConfig,
        description="Explainability configuration"
    )
    provenance: ProvenanceConfig = Field(
        default_factory=ProvenanceConfig,
        description="Provenance tracking configuration"
    )
    opcua: OPCUAConfig = Field(
        default_factory=OPCUAConfig,
        description="OPC-UA integration configuration"
    )
    kafka: KafkaConfig = Field(
        default_factory=KafkaConfig,
        description="Kafka integration configuration"
    )
    historian: HistorianConfig = Field(
        default_factory=HistorianConfig,
        description="Data historian configuration"
    )

    # Agent identification
    agent_id: str = Field(
        default="GL-022",
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="SuperheaterControlAgent",
        description="Agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Configuration version"
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
    archive_enabled: bool = Field(
        default=True,
        description="Enable data archiving"
    )

    # Audit settings
    audit_enabled: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    audit_level: str = Field(
        default="standard",
        description="Audit level (minimal, standard, verbose)"
    )

    # Performance settings
    calculation_precision: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Decimal precision for calculations"
    )
    optimization_interval_s: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Optimization interval (seconds)"
    )
    data_collection_interval_s: int = Field(
        default=1,
        ge=1,
        le=60,
        description="Data collection interval (seconds)"
    )

    @root_validator(skip_on_failure=True)
    def validate_config_consistency(cls, values):
        """Validate configuration consistency across components."""
        superheater = values.get("superheater")
        temperature_control = values.get("temperature_control")
        safety = values.get("safety")

        # Sync temperature limits
        if superheater and temperature_control:
            if temperature_control.setpoint_f > superheater.max_outlet_temp_f:
                raise ValueError(
                    f"Temperature setpoint ({temperature_control.setpoint_f}F) "
                    f"exceeds superheater max outlet temp ({superheater.max_outlet_temp_f}F)"
                )

        # Sync safety limits with superheater design
        if superheater and safety and safety.limits:
            if safety.limits.steam_outlet_high_trip_f < superheater.design_outlet_temp_f:
                raise ValueError(
                    "Safety trip point must be above design outlet temperature"
                )

        return values

    class Config:
        use_enum_values = True
        validate_assignment = True


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_default_config(
    superheater_id: str = "SH-001",
    desuperheater_id: str = "DSH-001",
    design_outlet_temp_f: float = 950.0,
    design_pressure_psig: float = 600.0,
) -> GL022Config:
    """
    Create a default GL-022 configuration with typical industrial values.

    Args:
        superheater_id: Unique superheater identifier
        desuperheater_id: Unique desuperheater identifier
        design_outlet_temp_f: Target outlet steam temperature (F)
        design_pressure_psig: Design operating pressure (psig)

    Returns:
        GL022Config with typical industrial settings

    Example:
        >>> config = create_default_config(
        ...     superheater_id="SH-001",
        ...     design_outlet_temp_f=950.0,
        ... )
    """
    return GL022Config(
        superheater=SuperheaterDesignConfig(
            superheater_id=superheater_id,
            superheater_type=SuperheaterType.CONVECTION,
            design_inlet_temp_f=500.0,
            design_outlet_temp_f=design_outlet_temp_f,
            min_outlet_temp_f=design_outlet_temp_f - 100.0,
            max_outlet_temp_f=design_outlet_temp_f + 100.0,
            design_pressure_psig=design_pressure_psig,
            max_allowable_working_pressure_psig=design_pressure_psig * 1.15,
            design_steam_flow_lb_hr=100000.0,
            tube_spec=TubeSpecificationConfig(
                tube_material=TubeMaterial.SA_213_T22,
                max_tube_metal_temp_f=1050.0,
            ),
        ),
        desuperheater=DesuperheaterControlConfig(
            desuperheater_id=desuperheater_id,
            spray_type=SprayType.WATER_SPRAY,
            spray_water_temp_f=250.0,
            spray_water_pressure_psig=design_pressure_psig + 150.0,
            max_spray_rate_lb_hr=10000.0,
            min_approach_temp_f=25.0,
        ),
        temperature_control=TemperatureControlConfig(
            control_mode=ControlMode.CASCADE,
            setpoint_f=design_outlet_temp_f,
            tolerance_f=5.0,
            high_temp_warning_f=design_outlet_temp_f + 50.0,
            high_temp_alarm_f=design_outlet_temp_f + 75.0,
            high_temp_trip_f=design_outlet_temp_f + 100.0,
        ),
    )


def create_high_temperature_config(
    superheater_id: str = "SH-001",
    desuperheater_id: str = "DSH-001",
) -> GL022Config:
    """
    Create configuration for high-temperature (1050F+) superheater application.

    Returns:
        GL022Config optimized for high-temperature operation
    """
    return GL022Config(
        superheater=SuperheaterDesignConfig(
            superheater_id=superheater_id,
            superheater_type=SuperheaterType.RADIANT_CONVECTION,
            design_inlet_temp_f=600.0,
            design_outlet_temp_f=1050.0,
            min_outlet_temp_f=950.0,
            max_outlet_temp_f=1100.0,
            design_pressure_psig=1500.0,
            max_allowable_working_pressure_psig=1725.0,
            design_steam_flow_lb_hr=150000.0,
            tube_spec=TubeSpecificationConfig(
                tube_material=TubeMaterial.SA_213_T91,
                max_tube_metal_temp_f=1200.0,
                tube_od_in=2.5,
                tube_wall_thickness_in=0.2,
            ),
        ),
        desuperheater=DesuperheaterControlConfig(
            desuperheater_id=desuperheater_id,
            spray_type=SprayType.STEAM_ATOMIZING,
            spray_water_temp_f=350.0,
            spray_water_pressure_psig=1700.0,
            max_spray_rate_lb_hr=15000.0,
            min_approach_temp_f=30.0,
            max_rate_of_change_f_min=8.0,  # More conservative for high temp
        ),
        temperature_control=TemperatureControlConfig(
            control_mode=ControlMode.CASCADE,
            setpoint_f=1050.0,
            tolerance_f=8.0,
            high_temp_warning_f=1075.0,
            high_temp_alarm_f=1090.0,
            high_temp_trip_f=1100.0,
            max_rate_up_f_min=8.0,
            max_rate_down_f_min=12.0,
        ),
        safety=SafetyConfig(
            sil_rating=SafetyIntegrityLevel.SIL_2,
            limits=SafetyLimitsConfig(
                tube_metal_high_warning_f=1150.0,
                tube_metal_high_alarm_f=1175.0,
                tube_metal_high_trip_f=1200.0,
                steam_outlet_high_trip_f=1100.0,
            ),
        ),
    )
