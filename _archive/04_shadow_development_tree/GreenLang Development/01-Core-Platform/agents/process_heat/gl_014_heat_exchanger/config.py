# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Configuration Module

This module defines all configuration schemas for the Heat Exchanger Optimization
agent, including exchanger types, TEMA classifications, fouling settings,
cleaning schedules, and safety thresholds.

Configuration follows GreenLang patterns with Pydantic validation and
sensible defaults based on TEMA standards (9th Edition) and ASME PTC 12.5.

References:
    - TEMA Standards 9th Edition (2007)
    - ASME PTC 12.5 Performance Test Code for Heat Exchangers
    - API 660 Shell-and-Tube Heat Exchangers
    - HTRI/HTFS Guidelines for Heat Exchanger Design

Example:
    >>> from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    ...     HeatExchangerConfig
    ... )
    >>> config = HeatExchangerConfig(
    ...     exchanger_id="HX-001",
    ...     exchanger_type=ExchangerType.SHELL_TUBE,
    ...     tema_type="AES"
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class ExchangerType(str, Enum):
    """Types of heat exchangers supported."""
    SHELL_TUBE = "shell_tube"
    PLATE = "plate"
    PLATE_FIN = "plate_fin"
    AIR_COOLED = "air_cooled"
    DOUBLE_PIPE = "double_pipe"
    SPIRAL = "spiral"
    SCRAPED_SURFACE = "scraped_surface"
    REBOILER = "reboiler"
    CONDENSER = "condenser"


class TEMAFrontEnd(str, Enum):
    """TEMA front end head types."""
    A = "A"  # Channel and removable cover
    B = "B"  # Bonnet (integral cover)
    C = "C"  # Channel integral with tubesheet, removable cover
    N = "N"  # Channel integral with tubesheet, non-removable cover
    D = "D"  # Special high pressure closure


class TEMAShell(str, Enum):
    """TEMA shell types."""
    E = "E"  # One pass shell
    F = "F"  # Two pass shell with longitudinal baffle
    G = "G"  # Split flow
    H = "H"  # Double split flow
    J = "J"  # Divided flow
    K = "K"  # Kettle type reboiler
    X = "X"  # Cross flow


class TEMARearEnd(str, Enum):
    """TEMA rear end head types."""
    L = "L"  # Fixed tubesheet like A
    M = "M"  # Fixed tubesheet like B
    N = "N"  # Fixed tubesheet like N
    P = "P"  # Outside packed floating head
    S = "S"  # Floating head with backing device
    T = "T"  # Pull through floating head
    U = "U"  # U-tube bundle
    W = "W"  # Externally sealed floating tubesheet


class TEMAClass(str, Enum):
    """TEMA mechanical standards class."""
    R = "R"  # Generally severe requirements - petroleum and chemical
    C = "C"  # Generally moderate requirements - commercial
    B = "B"  # Chemical service


class FlowArrangement(str, Enum):
    """Flow arrangement types."""
    COUNTER_FLOW = "counter_flow"
    PARALLEL_FLOW = "parallel_flow"
    CROSS_FLOW_MIXED = "cross_flow_mixed"
    CROSS_FLOW_UNMIXED = "cross_flow_unmixed"
    MULTI_PASS = "multi_pass"


class FoulingCategory(str, Enum):
    """Fouling mechanism categories per TEMA Table RGP-T2.4."""
    CRYSTALLIZATION = "crystallization"  # Scaling
    PARTICULATE = "particulate"  # Sedimentation
    BIOLOGICAL = "biological"  # Biofouling
    CHEMICAL = "chemical"  # Corrosion products
    COKING = "coking"  # High temperature deposits
    COMBINED = "combined"


class CleaningMethod(str, Enum):
    """Cleaning methods available."""
    MECHANICAL_BRUSHING = "mechanical_brushing"
    MECHANICAL_RODDING = "mechanical_rodding"
    HIGH_PRESSURE_WATER = "high_pressure_water"
    CHEMICAL_ACID = "chemical_acid"
    CHEMICAL_ALKALINE = "chemical_alkaline"
    CHEMICAL_SOLVENT = "chemical_solvent"
    PIGGING = "pigging"
    STEAM_BLOWING = "steam_blowing"
    ULTRASONIC = "ultrasonic"
    THERMAL_SHOCK = "thermal_shock"


class TubeLayout(str, Enum):
    """Tube layout patterns."""
    TRIANGULAR_30 = "triangular_30"  # 30 degree
    TRIANGULAR_60 = "triangular_60"  # 60 degree rotated
    SQUARE_90 = "square_90"  # 90 degree
    SQUARE_45 = "square_45"  # 45 degree rotated


class TubeMaterial(str, Enum):
    """Common tube materials."""
    CARBON_STEEL = "carbon_steel"
    STAINLESS_304 = "stainless_304"
    STAINLESS_316 = "stainless_316"
    STAINLESS_321 = "stainless_321"
    DUPLEX_2205 = "duplex_2205"
    SUPER_DUPLEX = "super_duplex"
    MONEL_400 = "monel_400"
    INCONEL_600 = "inconel_600"
    INCONEL_625 = "inconel_625"
    HASTELLOY_C276 = "hastelloy_c276"
    TITANIUM_GR2 = "titanium_gr2"
    COPPER_NICKEL_90_10 = "copper_nickel_90_10"
    COPPER_NICKEL_70_30 = "copper_nickel_70_30"
    ADMIRALTY_BRASS = "admiralty_brass"
    ALUMINUM_BRASS = "aluminum_brass"


class FailureMode(str, Enum):
    """Heat exchanger failure modes."""
    TUBE_LEAK = "tube_leak"
    TUBE_RUPTURE = "tube_rupture"
    TUBE_BLOCKAGE = "tube_blockage"
    SHELL_CORROSION = "shell_corrosion"
    BAFFLE_DAMAGE = "baffle_damage"
    TUBESHEET_DAMAGE = "tubesheet_damage"
    GASKET_FAILURE = "gasket_failure"
    EXPANSION_JOINT_FAILURE = "expansion_joint_failure"
    FOULING_CRITICAL = "fouling_critical"
    VIBRATION_DAMAGE = "vibration_damage"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


# =============================================================================
# GEOMETRY CONFIGURATION
# =============================================================================

class TubeGeometryConfig(BaseModel):
    """Tube geometry configuration."""

    outer_diameter_mm: float = Field(
        default=25.4,
        gt=0,
        description="Tube outer diameter (mm) - common: 19.05, 25.4, 31.75"
    )
    wall_thickness_mm: float = Field(
        default=2.11,
        gt=0,
        description="Tube wall thickness (mm) - BWG gauge"
    )
    tube_length_m: float = Field(
        default=6.096,
        gt=0,
        description="Tube length (m) - standard: 3.048, 4.877, 6.096"
    )
    tube_count: int = Field(
        default=100,
        ge=1,
        description="Number of tubes"
    )
    tube_passes: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Number of tube passes"
    )
    tube_pitch_mm: float = Field(
        default=31.75,
        gt=0,
        description="Tube pitch (center to center distance)"
    )
    tube_layout: TubeLayout = Field(
        default=TubeLayout.TRIANGULAR_30,
        description="Tube layout pattern"
    )
    tube_material: TubeMaterial = Field(
        default=TubeMaterial.CARBON_STEEL,
        description="Tube material"
    )

    @property
    def inner_diameter_mm(self) -> float:
        """Calculate tube inner diameter."""
        return self.outer_diameter_mm - 2 * self.wall_thickness_mm

    @property
    def tube_area_m2(self) -> float:
        """Calculate total tube heat transfer area."""
        import math
        return (
            math.pi * self.outer_diameter_mm / 1000
            * self.tube_length_m * self.tube_count
        )

    @property
    def pitch_ratio(self) -> float:
        """Calculate pitch to diameter ratio."""
        return self.tube_pitch_mm / self.outer_diameter_mm


class ShellGeometryConfig(BaseModel):
    """Shell geometry configuration."""

    inner_diameter_mm: float = Field(
        default=610.0,
        gt=0,
        description="Shell inner diameter (mm)"
    )
    shell_passes: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of shell passes"
    )
    baffle_cut_percent: float = Field(
        default=25.0,
        ge=15,
        le=45,
        description="Baffle cut as percentage of shell ID"
    )
    baffle_spacing_mm: float = Field(
        default=300.0,
        gt=0,
        description="Baffle spacing (mm)"
    )
    baffle_count: int = Field(
        default=10,
        ge=0,
        description="Number of baffles"
    )
    sealing_strips: int = Field(
        default=0,
        ge=0,
        description="Number of sealing strip pairs"
    )


class PlateGeometryConfig(BaseModel):
    """Plate heat exchanger geometry configuration."""

    plate_count: int = Field(
        default=50,
        ge=3,
        description="Number of plates"
    )
    plate_length_mm: float = Field(
        default=1000.0,
        gt=0,
        description="Plate length (mm)"
    )
    plate_width_mm: float = Field(
        default=400.0,
        gt=0,
        description="Plate width (mm)"
    )
    plate_spacing_mm: float = Field(
        default=3.0,
        gt=0,
        description="Plate spacing / channel gap (mm)"
    )
    chevron_angle_deg: float = Field(
        default=60.0,
        ge=25,
        le=70,
        description="Chevron angle (degrees)"
    )
    port_diameter_mm: float = Field(
        default=150.0,
        gt=0,
        description="Port diameter (mm)"
    )
    plate_material: TubeMaterial = Field(
        default=TubeMaterial.STAINLESS_316,
        description="Plate material"
    )

    @property
    def heat_transfer_area_m2(self) -> float:
        """Calculate total heat transfer area."""
        return (
            (self.plate_count - 2)
            * self.plate_length_mm / 1000
            * self.plate_width_mm / 1000
        )


class AirCooledGeometryConfig(BaseModel):
    """Air-cooled heat exchanger geometry configuration."""

    bundle_count: int = Field(
        default=2,
        ge=1,
        description="Number of tube bundles"
    )
    fan_count: int = Field(
        default=4,
        ge=1,
        description="Number of fans"
    )
    fan_diameter_m: float = Field(
        default=3.0,
        gt=0,
        description="Fan diameter (m)"
    )
    tubes_per_row: int = Field(
        default=30,
        ge=1,
        description="Tubes per row"
    )
    tube_rows: int = Field(
        default=6,
        ge=1,
        description="Number of tube rows"
    )
    fin_pitch_mm: float = Field(
        default=2.5,
        gt=0,
        description="Fin pitch (mm)"
    )
    fin_height_mm: float = Field(
        default=15.0,
        gt=0,
        description="Fin height (mm)"
    )
    fin_thickness_mm: float = Field(
        default=0.4,
        gt=0,
        description="Fin thickness (mm)"
    )
    induced_draft: bool = Field(
        default=True,
        description="True for induced draft, False for forced draft"
    )


# =============================================================================
# FOULING CONFIGURATION
# =============================================================================

class FoulingConfig(BaseModel):
    """Fouling configuration per TEMA RGP-T2.4."""

    shell_side_fouling_m2kw: float = Field(
        default=0.00017,
        ge=0,
        description="Shell side fouling resistance (m2K/W)"
    )
    tube_side_fouling_m2kw: float = Field(
        default=0.00017,
        ge=0,
        description="Tube side fouling resistance (m2K/W)"
    )
    fouling_category: FoulingCategory = Field(
        default=FoulingCategory.PARTICULATE,
        description="Primary fouling mechanism"
    )
    design_fouling_factor: float = Field(
        default=1.0,
        ge=1.0,
        le=2.0,
        description="Design oversize factor for fouling"
    )
    fouling_rate_m2kw_per_day: float = Field(
        default=0.000001,
        ge=0,
        description="Expected fouling rate (m2K/W per day)"
    )
    asymptotic_fouling_m2kw: Optional[float] = Field(
        default=None,
        description="Asymptotic fouling resistance"
    )
    removal_rate_coefficient: float = Field(
        default=0.1,
        ge=0,
        le=1.0,
        description="Fouling removal rate coefficient"
    )

    # ML prediction settings
    ml_prediction_enabled: bool = Field(
        default=True,
        description="Enable ML fouling prediction"
    )
    prediction_horizon_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Fouling prediction horizon"
    )


class TEMAFoulingFactors(BaseModel):
    """Standard TEMA fouling factors per RGP-T2.4."""

    # Water services (m2K/W)
    cooling_tower_water: float = Field(
        default=0.00035,
        description="Cooling tower water, treated"
    )
    sea_water: float = Field(
        default=0.00017,
        description="Sea water, below 50C"
    )
    boiler_feedwater: float = Field(
        default=0.00009,
        description="Boiler feedwater, treated"
    )
    river_water: float = Field(
        default=0.00035,
        description="River water, minimum treated"
    )

    # Hydrocarbon services (m2K/W)
    fuel_oil: float = Field(
        default=0.00088,
        description="Fuel oil"
    )
    crude_oil_dry: float = Field(
        default=0.00035,
        description="Crude oil, dry"
    )
    crude_oil_wet: float = Field(
        default=0.00053,
        description="Crude oil, wet"
    )
    gas_oil: float = Field(
        default=0.00035,
        description="Gas oil, light ends"
    )
    gasoline: float = Field(
        default=0.00018,
        description="Gasoline"
    )
    naphtha: float = Field(
        default=0.00018,
        description="Naphtha"
    )

    # Process streams (m2K/W)
    steam: float = Field(
        default=0.00009,
        description="Steam, oil-free"
    )
    process_gas: float = Field(
        default=0.00018,
        description="Process gas, clean"
    )
    organic_solvents: float = Field(
        default=0.00018,
        description="Organic solvents"
    )


# =============================================================================
# CLEANING CONFIGURATION
# =============================================================================

class CleaningConfig(BaseModel):
    """Cleaning schedule configuration."""

    preferred_methods: List[CleaningMethod] = Field(
        default=[CleaningMethod.HIGH_PRESSURE_WATER],
        description="Preferred cleaning methods in order"
    )
    minimum_interval_days: int = Field(
        default=30,
        ge=1,
        description="Minimum interval between cleanings"
    )
    maximum_interval_days: int = Field(
        default=365,
        ge=30,
        description="Maximum interval between cleanings"
    )
    effectiveness_threshold: float = Field(
        default=0.70,
        ge=0.5,
        le=1.0,
        description="Minimum effectiveness to trigger cleaning"
    )
    fouling_threshold_m2kw: float = Field(
        default=0.00035,
        ge=0,
        description="Fouling resistance threshold for cleaning"
    )
    cleaning_duration_hours: float = Field(
        default=8.0,
        gt=0,
        description="Expected cleaning duration"
    )
    cleaning_cost_usd: float = Field(
        default=5000.0,
        ge=0,
        description="Estimated cleaning cost"
    )
    production_loss_usd_per_hour: float = Field(
        default=1000.0,
        ge=0,
        description="Production loss during cleaning"
    )

    # Optimization settings
    optimize_schedule: bool = Field(
        default=True,
        description="Enable cleaning schedule optimization"
    )
    target_availability: float = Field(
        default=0.95,
        ge=0.80,
        le=1.0,
        description="Target equipment availability"
    )


# =============================================================================
# TUBE INTEGRITY CONFIGURATION
# =============================================================================

class TubeIntegrityConfig(BaseModel):
    """Tube integrity and inspection configuration."""

    design_life_years: float = Field(
        default=20.0,
        gt=0,
        description="Design life (years)"
    )
    installed_date: Optional[datetime] = Field(
        default=None,
        description="Installation date"
    )
    last_inspection_date: Optional[datetime] = Field(
        default=None,
        description="Last tube inspection date"
    )

    # Thickness monitoring
    minimum_wall_thickness_mm: float = Field(
        default=1.25,
        gt=0,
        description="Minimum allowable wall thickness"
    )
    corrosion_allowance_mm: float = Field(
        default=1.5,
        gt=0,
        description="Corrosion allowance"
    )
    expected_corrosion_rate_mm_year: float = Field(
        default=0.1,
        ge=0,
        description="Expected corrosion rate"
    )

    # Inspection settings
    inspection_interval_months: int = Field(
        default=24,
        ge=6,
        le=120,
        description="Tube inspection interval"
    )
    eddy_current_enabled: bool = Field(
        default=True,
        description="Enable eddy current testing"
    )
    plugging_threshold_percent: float = Field(
        default=10.0,
        ge=0,
        le=100,
        description="Tube plugging threshold before retube"
    )

    # Prediction settings
    predict_tube_failures: bool = Field(
        default=True,
        description="Enable tube failure prediction"
    )
    weibull_beta: Optional[float] = Field(
        default=None,
        description="Weibull shape parameter from history"
    )
    weibull_eta: Optional[float] = Field(
        default=None,
        description="Weibull scale parameter from history"
    )


# =============================================================================
# OPERATING LIMITS CONFIGURATION
# =============================================================================

class OperatingLimitsConfig(BaseModel):
    """Operating limits and alarm thresholds."""

    # Temperature limits (Celsius)
    max_shell_inlet_temp_c: float = Field(
        default=300.0,
        description="Maximum shell inlet temperature"
    )
    max_tube_inlet_temp_c: float = Field(
        default=300.0,
        description="Maximum tube inlet temperature"
    )
    max_tube_wall_temp_c: float = Field(
        default=350.0,
        description="Maximum tube wall temperature"
    )
    min_approach_temp_c: float = Field(
        default=5.0,
        gt=0,
        description="Minimum temperature approach"
    )

    # Pressure limits (barg)
    max_shell_pressure_barg: float = Field(
        default=15.0,
        description="Maximum shell side pressure"
    )
    max_tube_pressure_barg: float = Field(
        default=15.0,
        description="Maximum tube side pressure"
    )
    max_differential_pressure_bar: float = Field(
        default=5.0,
        gt=0,
        description="Maximum shell-tube differential pressure"
    )

    # Flow limits
    max_shell_velocity_m_s: float = Field(
        default=3.0,
        gt=0,
        description="Maximum shell side velocity"
    )
    max_tube_velocity_m_s: float = Field(
        default=3.0,
        gt=0,
        description="Maximum tube side velocity"
    )
    min_tube_velocity_m_s: float = Field(
        default=0.5,
        gt=0,
        description="Minimum tube velocity (prevent fouling)"
    )

    # Pressure drop limits
    max_shell_dp_bar: float = Field(
        default=1.0,
        gt=0,
        description="Maximum shell side pressure drop"
    )
    max_tube_dp_bar: float = Field(
        default=1.0,
        gt=0,
        description="Maximum tube side pressure drop"
    )

    # Effectiveness limits
    min_effectiveness: float = Field(
        default=0.5,
        ge=0,
        le=1.0,
        description="Minimum acceptable effectiveness"
    )
    alarm_effectiveness: float = Field(
        default=0.65,
        ge=0,
        le=1.0,
        description="Effectiveness alarm threshold"
    )


# =============================================================================
# ECONOMICS CONFIGURATION
# =============================================================================

class EconomicsConfig(BaseModel):
    """Economic analysis configuration."""

    energy_cost_usd_per_kwh: float = Field(
        default=0.10,
        ge=0,
        description="Energy cost (USD/kWh)"
    )
    steam_cost_usd_per_ton: float = Field(
        default=30.0,
        ge=0,
        description="Steam cost (USD/metric ton)"
    )
    cooling_water_cost_usd_per_m3: float = Field(
        default=0.50,
        ge=0,
        description="Cooling water cost (USD/m3)"
    )
    replacement_cost_usd: float = Field(
        default=500000.0,
        ge=0,
        description="Exchanger replacement cost"
    )
    retube_cost_usd: float = Field(
        default=150000.0,
        ge=0,
        description="Retubing cost"
    )
    production_value_usd_per_hour: float = Field(
        default=10000.0,
        ge=0,
        description="Production value per hour"
    )
    discount_rate: float = Field(
        default=0.10,
        ge=0,
        le=0.30,
        description="Discount rate for NPV calculations"
    )
    analysis_horizon_years: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Economic analysis horizon"
    )


# =============================================================================
# ML CONFIGURATION
# =============================================================================

class MLConfig(BaseModel):
    """Machine learning configuration for GL-014."""

    enabled: bool = Field(
        default=True,
        description="Enable ML predictions"
    )
    fouling_prediction_enabled: bool = Field(
        default=True,
        description="Enable ML fouling rate prediction"
    )
    tube_failure_prediction_enabled: bool = Field(
        default=True,
        description="Enable ML tube failure prediction"
    )
    cleaning_optimization_enabled: bool = Field(
        default=True,
        description="Enable ML cleaning schedule optimization"
    )
    model_update_interval_days: int = Field(
        default=30,
        ge=7,
        description="Model retraining interval"
    )
    uncertainty_quantification: bool = Field(
        default=True,
        description="Enable uncertainty bounds"
    )
    confidence_threshold: float = Field(
        default=0.80,
        ge=0.50,
        le=0.99,
        description="Minimum confidence for predictions"
    )
    explainability_enabled: bool = Field(
        default=True,
        description="Enable SHAP feature importance"
    )


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

class HeatExchangerConfig(BaseModel):
    """
    Main configuration for GL-014 Heat Exchanger Optimization Agent.

    This configuration encompasses all aspects of heat exchanger monitoring
    and optimization including geometry, fouling, cleaning, tube integrity,
    operating limits, and economics.

    Attributes:
        exchanger_id: Unique exchanger identifier
        exchanger_type: Type of heat exchanger
        tema_type: TEMA designation (e.g., AES, BEM, AEU)
        service_description: Process service description

    Example:
        >>> config = HeatExchangerConfig(
        ...     exchanger_id="E-1001",
        ...     exchanger_type=ExchangerType.SHELL_TUBE,
        ...     tema_type="AES",
        ...     service_description="Crude preheat #1"
        ... )
    """

    # Equipment identification
    exchanger_id: str = Field(..., description="Unique exchanger identifier")
    exchanger_type: ExchangerType = Field(
        ...,
        description="Type of heat exchanger"
    )
    tema_type: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="TEMA type designation (e.g., AES, BEM)"
    )
    tema_class: TEMAClass = Field(
        default=TEMAClass.R,
        description="TEMA mechanical standards class"
    )
    service_description: str = Field(
        default="",
        description="Process service description"
    )
    tag_number: str = Field(
        default="",
        description="Plant tag number (e.g., E-1001)"
    )
    location: str = Field(
        default="",
        description="Physical location / area"
    )

    # Geometry (type-specific)
    tube_geometry: Optional[TubeGeometryConfig] = Field(
        default=None,
        description="Tube geometry for shell-tube exchangers"
    )
    shell_geometry: Optional[ShellGeometryConfig] = Field(
        default=None,
        description="Shell geometry for shell-tube exchangers"
    )
    plate_geometry: Optional[PlateGeometryConfig] = Field(
        default=None,
        description="Plate geometry for plate exchangers"
    )
    air_cooled_geometry: Optional[AirCooledGeometryConfig] = Field(
        default=None,
        description="Geometry for air-cooled exchangers"
    )

    # Flow arrangement
    flow_arrangement: FlowArrangement = Field(
        default=FlowArrangement.COUNTER_FLOW,
        description="Flow arrangement"
    )

    # Design parameters
    design_duty_kw: float = Field(
        default=1000.0,
        gt=0,
        description="Design heat duty (kW)"
    )
    design_u_w_m2k: float = Field(
        default=500.0,
        gt=0,
        description="Design overall U (W/m2K)"
    )
    design_lmtd_c: float = Field(
        default=30.0,
        gt=0,
        description="Design LMTD (Celsius)"
    )
    design_effectiveness: float = Field(
        default=0.75,
        ge=0,
        le=1.0,
        description="Design thermal effectiveness"
    )

    # Stream properties at design
    shell_side_fluid: str = Field(
        default="water",
        description="Shell side fluid"
    )
    tube_side_fluid: str = Field(
        default="water",
        description="Tube side fluid"
    )
    shell_flow_kg_s: float = Field(
        default=10.0,
        gt=0,
        description="Shell side mass flow rate (kg/s)"
    )
    tube_flow_kg_s: float = Field(
        default=10.0,
        gt=0,
        description="Tube side mass flow rate (kg/s)"
    )

    # Configurations
    fouling: FoulingConfig = Field(
        default_factory=FoulingConfig,
        description="Fouling configuration"
    )
    cleaning: CleaningConfig = Field(
        default_factory=CleaningConfig,
        description="Cleaning configuration"
    )
    tube_integrity: TubeIntegrityConfig = Field(
        default_factory=TubeIntegrityConfig,
        description="Tube integrity configuration"
    )
    operating_limits: OperatingLimitsConfig = Field(
        default_factory=OperatingLimitsConfig,
        description="Operating limits"
    )
    economics: EconomicsConfig = Field(
        default_factory=EconomicsConfig,
        description="Economics configuration"
    )
    ml: MLConfig = Field(
        default_factory=MLConfig,
        description="ML configuration"
    )

    # TEMA fouling factors reference
    tema_fouling_factors: TEMAFoulingFactors = Field(
        default_factory=TEMAFoulingFactors,
        description="TEMA standard fouling factors"
    )

    # ASME PTC 12.5 testing
    asme_testing_enabled: bool = Field(
        default=False,
        description="Enable ASME PTC 12.5 compliance testing"
    )

    # Audit and provenance
    audit_enabled: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance"
    )

    class Config:
        use_enum_values = True

    @validator("tema_type")
    def validate_tema_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate TEMA type designation."""
        if v is None:
            return None

        if len(v) != 3:
            raise ValueError("TEMA type must be 3 characters")

        front_ends = set(e.value for e in TEMAFrontEnd)
        shells = set(e.value for e in TEMAShell)
        rear_ends = set(e.value for e in TEMARearEnd)

        if v[0] not in front_ends:
            raise ValueError(f"Invalid TEMA front end: {v[0]}")
        if v[1] not in shells:
            raise ValueError(f"Invalid TEMA shell: {v[1]}")
        if v[2] not in rear_ends:
            raise ValueError(f"Invalid TEMA rear end: {v[2]}")

        return v.upper()

    @validator("tube_geometry", always=True)
    def validate_tube_geometry(cls, v, values):
        """Ensure tube geometry for shell-tube exchangers."""
        exchanger_type = values.get("exchanger_type")
        if exchanger_type == ExchangerType.SHELL_TUBE and v is None:
            return TubeGeometryConfig()
        return v

    @validator("shell_geometry", always=True)
    def validate_shell_geometry(cls, v, values):
        """Ensure shell geometry for shell-tube exchangers."""
        exchanger_type = values.get("exchanger_type")
        if exchanger_type == ExchangerType.SHELL_TUBE and v is None:
            return ShellGeometryConfig()
        return v

    @validator("plate_geometry", always=True)
    def validate_plate_geometry(cls, v, values):
        """Ensure plate geometry for plate exchangers."""
        exchanger_type = values.get("exchanger_type")
        if exchanger_type == ExchangerType.PLATE and v is None:
            return PlateGeometryConfig()
        return v

    @validator("air_cooled_geometry", always=True)
    def validate_air_cooled_geometry(cls, v, values):
        """Ensure air-cooled geometry for air-cooled exchangers."""
        exchanger_type = values.get("exchanger_type")
        if exchanger_type == ExchangerType.AIR_COOLED and v is None:
            return AirCooledGeometryConfig()
        return v
