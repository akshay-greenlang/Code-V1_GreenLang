# -*- coding: utf-8 -*-
"""
GL-023 HeatLoadBalancer Agent - Configuration Module

This module defines comprehensive configuration schemas for the Heat Load Balancer
Agent, which provides optimal load distribution across multiple boilers and furnaces
in industrial process heat systems.

Configuration Categories:
    - Equipment Configuration: Boiler and furnace specifications
    - Efficiency Curve Configuration: Polynomial and piecewise efficiency models
    - Optimization Configuration: MILP solver and economic dispatch settings
    - Fuel Configuration: Multi-fuel pricing and carbon intensity
    - Demand Configuration: Load profiles and demand response
    - Safety Configuration: N+1 redundancy and emergency reserves
    - Integration Configuration: OPC-UA, Kafka, SCADA connectivity
    - Explainability Configuration: SHAP/LIME for optimization decisions

Standards Compliance:
    - ASME CSD-1: Controls and Safety Devices for Automatically Fired Boilers
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - API 560: Fired Heaters for General Refinery Service
    - ISA 84: Safety Instrumented Systems (SIL ratings)
    - IEEE 1815 (DNP3): SCADA Communications
    - OPC-UA Part 4: Services

Example:
    >>> from greenlang.agents.process_heat.gl_023_heat_load_balancer.config import (
    ...     GL023Config,
    ...     BoilerConfig,
    ...     FurnaceConfig,
    ...     MILPConfig,
    ...     create_default_config,
    ... )
    >>>
    >>> config = create_default_config()
    >>> print(f"Fleet size: {len(config.equipment_fleet.boilers)} boilers, "
    ...       f"{len(config.equipment_fleet.furnaces)} furnaces")
    Fleet size: 3 boilers, 2 furnaces

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# ENUMS - EQUIPMENT AND OPTIMIZATION CLASSIFICATIONS
# =============================================================================


class BoilerType(str, Enum):
    """
    Types of industrial boilers supported.

    Classification based on construction and application:
    - FIRETUBE: Hot gases pass through tubes surrounded by water
    - WATERTUBE: Water in tubes, gases flow around
    - PACKAGE: Factory-assembled, shop-tested units
    - FIELD_ERECTED: Large units assembled on-site
    - HRSG: Heat Recovery Steam Generator (combined cycle)
    - CFB: Circulating Fluidized Bed
    - ONCE_THROUGH: Supercritical, no drum
    """
    FIRETUBE = "firetube"
    WATERTUBE = "watertube"
    PACKAGE = "package"
    FIELD_ERECTED = "field_erected"
    HRSG = "hrsg"
    CFB = "cfb"
    ONCE_THROUGH = "once_through"
    ELECTRIC = "electric"


class FurnaceType(str, Enum):
    """
    Types of industrial furnaces supported per API 560.

    Classification based on tube configuration and application:
    - CABIN: Rectangular box with horizontal tubes
    - CYLINDRICAL_VERTICAL: Vertical tubes in cylindrical shell
    - CYLINDRICAL_HORIZONTAL: Horizontal tubes in cylindrical shell
    - BOX: Rectangular with vertical tubes
    - REFORMER: For steam methane reforming
    - PYROLYSIS: For thermal cracking
    - THERMAL_OXIDIZER: For waste gas combustion
    """
    CABIN = "cabin"
    CYLINDRICAL_VERTICAL = "cylindrical_vertical"
    CYLINDRICAL_HORIZONTAL = "cylindrical_horizontal"
    BOX = "box"
    REFORMER = "reformer"
    PYROLYSIS = "pyrolysis"
    THERMAL_OXIDIZER = "thermal_oxidizer"
    PROCESS_HEATER = "process_heater"


class FuelType(str, Enum):
    """
    Supported fuel types for multi-fuel optimization.

    Includes conventional, alternative, and renewable fuels
    with different carbon intensities and availability profiles.
    """
    NATURAL_GAS = "natural_gas"
    NO2_FUEL_OIL = "no2_fuel_oil"
    NO6_FUEL_OIL = "no6_fuel_oil"
    LPG_PROPANE = "lpg_propane"
    LPG_BUTANE = "lpg_butane"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUB_BITUMINOUS = "coal_sub_bituminous"
    REFINERY_FUEL_GAS = "refinery_fuel_gas"
    HYDROGEN = "hydrogen"
    BIOGAS = "biogas"
    RNG = "rng"  # Renewable Natural Gas
    BIOMASS = "biomass"
    WASTE_HEAT = "waste_heat"


class SolverType(str, Enum):
    """
    MILP solver options for optimization.

    Ranges from open-source (CBC, GLPK) to commercial (CPLEX, Gurobi)
    with different performance characteristics.
    """
    CBC = "cbc"  # Open source, good for medium problems
    GLPK = "glpk"  # GNU Linear Programming Kit
    CPLEX = "cplex"  # IBM commercial solver
    GUROBI = "gurobi"  # Commercial, excellent performance
    HIGHS = "highs"  # New open-source solver
    SCIP = "scip"  # Academic solver
    MOSEK = "mosek"  # Commercial, good for convex problems


class OptimizationObjective(str, Enum):
    """
    Optimization objective functions.

    Supports single objective and multi-objective (Pareto) approaches
    for economic and environmental optimization.
    """
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_EMISSIONS = "minimize_emissions"
    MINIMIZE_FUEL_CONSUMPTION = "minimize_fuel_consumption"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    PARETO_COST_EMISSIONS = "pareto_cost_emissions"
    PARETO_MULTI_OBJECTIVE = "pareto_multi_objective"
    WEIGHTED_SUM = "weighted_sum"


class EquipmentStatus(str, Enum):
    """Equipment operational status."""
    AVAILABLE = "available"
    RUNNING = "running"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"
    FAULTED = "faulted"
    OFFLINE = "offline"
    STARTING = "starting"
    STOPPING = "stopping"


class SafetyIntegrityLevel(str, Enum):
    """IEC 61508 / ISA 84 Safety Integrity Levels."""
    SIL_1 = "sil_1"  # PFD 0.1 to 0.01
    SIL_2 = "sil_2"  # PFD 0.01 to 0.001
    SIL_3 = "sil_3"  # PFD 0.001 to 0.0001
    SIL_4 = "sil_4"  # PFD 0.0001 to 0.00001
    NON_SIL = "non_sil"


class ExplainabilityMethod(str, Enum):
    """Explainability methods for optimization decisions."""
    SHAP = "shap"
    SHAP_KERNEL = "shap_kernel"
    SHAP_TREE = "shap_tree"
    LIME = "lime"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    CONSTRAINT_ANALYSIS = "constraint_analysis"
    DUAL_ANALYSIS = "dual_analysis"


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


class AlertSeverity(str, Enum):
    """Alert severity levels for monitoring."""
    GOOD = "good"
    ADVISORY = "advisory"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    TRIP = "trip"


# =============================================================================
# EFFICIENCY CURVE CONFIGURATION
# =============================================================================


class PolynomialEfficiencyConfig(BaseModel):
    """
    Polynomial efficiency curve configuration.

    Efficiency is modeled as:
        eta = a0 + a1*L + a2*L^2 + a3*L^3

    Where L is load fraction (0.0 to 1.0) and eta is efficiency (0.0 to 1.0).

    Typical values for natural gas boilers:
        a0 = 0.65 (efficiency at zero load - theoretical)
        a1 = 0.35 (linear coefficient)
        a2 = -0.20 (quadratic coefficient - efficiency drops at high load)
        a3 = 0.02 (cubic coefficient - minor correction)

    Example:
        >>> config = PolynomialEfficiencyConfig(
        ...     a0=0.70, a1=0.25, a2=-0.15, a3=0.02
        ... )
        >>> # At 80% load: eta = 0.70 + 0.25*0.8 - 0.15*0.64 + 0.02*0.512
        >>> # eta = 0.70 + 0.20 - 0.096 + 0.01 = 0.814 (81.4%)
    """

    a0: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Constant coefficient (intercept)"
    )
    a1: float = Field(
        default=0.25,
        ge=-1.0,
        le=1.0,
        description="Linear coefficient"
    )
    a2: float = Field(
        default=-0.15,
        ge=-1.0,
        le=1.0,
        description="Quadratic coefficient"
    )
    a3: float = Field(
        default=0.02,
        ge=-1.0,
        le=1.0,
        description="Cubic coefficient"
    )

    # Validity range
    min_load_fraction: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum load fraction for valid efficiency"
    )
    max_load_fraction: float = Field(
        default=1.0,
        ge=0.0,
        le=1.2,
        description="Maximum load fraction for valid efficiency"
    )

    # Temperature correction
    ambient_temp_reference_f: float = Field(
        default=60.0,
        ge=-20,
        le=120,
        description="Reference ambient temperature for curve (F)"
    )
    temp_correction_factor: float = Field(
        default=0.0002,
        ge=0.0,
        le=0.001,
        description="Efficiency correction per degree F deviation"
    )

    @validator("max_load_fraction")
    def validate_load_range(cls, v, values):
        """Ensure max load is greater than min load."""
        if "min_load_fraction" in values and v <= values["min_load_fraction"]:
            raise ValueError("max_load_fraction must be > min_load_fraction")
        return v


class EfficiencyDataPoint(BaseModel):
    """Single data point for piecewise linear efficiency curve."""

    load_pct: float = Field(
        ...,
        ge=0,
        le=120,
        description="Load percentage (0-120%)"
    )
    efficiency_pct: float = Field(
        ...,
        ge=50,
        le=100,
        description="Efficiency at this load (%)"
    )


class PiecewiseEfficiencyConfig(BaseModel):
    """
    Piecewise linear efficiency curve from manufacturer data.

    Uses linear interpolation between data points for efficiency
    lookup. Useful when manufacturer provides test data at specific loads.

    Example:
        >>> config = PiecewiseEfficiencyConfig(
        ...     data_points=[
        ...         EfficiencyDataPoint(load_pct=25, efficiency_pct=78.0),
        ...         EfficiencyDataPoint(load_pct=50, efficiency_pct=82.0),
        ...         EfficiencyDataPoint(load_pct=75, efficiency_pct=84.0),
        ...         EfficiencyDataPoint(load_pct=100, efficiency_pct=82.5),
        ...     ]
        ... )
    """

    data_points: List[EfficiencyDataPoint] = Field(
        default_factory=list,
        description="Efficiency data points from manufacturer"
    )
    interpolation_method: str = Field(
        default="linear",
        description="Interpolation method (linear, cubic_spline)"
    )
    extrapolate_below_min: bool = Field(
        default=False,
        description="Allow extrapolation below minimum load"
    )
    extrapolate_above_max: bool = Field(
        default=False,
        description="Allow extrapolation above maximum load"
    )

    @validator("data_points")
    def validate_data_points(cls, v):
        """Ensure data points are sorted by load."""
        if len(v) < 2:
            raise ValueError("At least 2 data points required")
        loads = [p.load_pct for p in v]
        if loads != sorted(loads):
            raise ValueError("Data points must be sorted by load_pct")
        return v


class EfficiencyCurveConfig(BaseModel):
    """
    Complete efficiency curve configuration.

    Supports both polynomial and piecewise linear representations.
    The polynomial model is used by default, with piecewise as override.

    Attributes:
        curve_type: Type of efficiency curve model
        polynomial: Polynomial coefficients if curve_type is polynomial
        piecewise: Data points if curve_type is piecewise
        fuel_correction_factors: Efficiency adjustments by fuel type

    Example:
        >>> config = EfficiencyCurveConfig(
        ...     curve_type="polynomial",
        ...     polynomial=PolynomialEfficiencyConfig(a0=0.72, a1=0.22),
        ... )
    """

    curve_type: str = Field(
        default="polynomial",
        description="Curve type (polynomial, piecewise, lookup_table)"
    )
    polynomial: PolynomialEfficiencyConfig = Field(
        default_factory=PolynomialEfficiencyConfig,
        description="Polynomial efficiency curve configuration"
    )
    piecewise: Optional[PiecewiseEfficiencyConfig] = Field(
        default=None,
        description="Piecewise efficiency curve (overrides polynomial if set)"
    )

    # Fuel-specific corrections
    fuel_correction_factors: Dict[str, float] = Field(
        default_factory=lambda: {
            "natural_gas": 1.0,
            "no2_fuel_oil": 0.98,
            "no6_fuel_oil": 0.96,
            "lpg_propane": 0.99,
            "hydrogen": 1.02,
            "biogas": 0.97,
        },
        description="Efficiency multiplier by fuel type"
    )

    # Degradation
    degradation_rate_pct_year: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="Annual efficiency degradation (%/year)"
    )
    last_tuning_date: Optional[datetime] = Field(
        default=None,
        description="Date of last combustion tuning"
    )

    # Validation range
    min_valid_efficiency_pct: float = Field(
        default=60.0,
        ge=40,
        le=80,
        description="Minimum valid efficiency (%)"
    )
    max_valid_efficiency_pct: float = Field(
        default=98.0,
        ge=85,
        le=100,
        description="Maximum valid efficiency (%)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# EQUIPMENT CONFIGURATION
# =============================================================================


class BoilerConfig(BaseModel):
    """
    Configuration for a single boiler in the fleet.

    Defines physical characteristics, operating limits, efficiency curves,
    and fuel capabilities for a boiler unit per ASME CSD-1 and NFPA 85.

    Attributes:
        boiler_id: Unique boiler identifier
        boiler_type: Type of boiler construction
        capacity_mmbtu_hr: Maximum heat output capacity
        min_load_pct: Minimum stable firing rate
        max_load_pct: Maximum allowable load
        turndown_ratio: Ratio of max to min firing rate
        fuel_types: List of fuels this boiler can fire
        efficiency_curve: Efficiency model configuration

    Example:
        >>> boiler = BoilerConfig(
        ...     boiler_id="BLR-001",
        ...     boiler_type=BoilerType.WATERTUBE,
        ...     capacity_mmbtu_hr=100.0,
        ...     min_load_pct=25.0,
        ...     max_load_pct=100.0,
        ...     turndown_ratio=4.0,
        ...     fuel_types=[FuelType.NATURAL_GAS, FuelType.NO2_FUEL_OIL],
        ... )
    """

    # Identification
    boiler_id: str = Field(
        ...,
        description="Unique boiler identifier"
    )
    boiler_tag: str = Field(
        default="",
        description="Plant equipment tag (e.g., BLR-101)"
    )
    boiler_name: str = Field(
        default="",
        description="Human-readable boiler name"
    )
    boiler_type: BoilerType = Field(
        default=BoilerType.WATERTUBE,
        description="Boiler type classification"
    )

    # Capacity specifications
    capacity_mmbtu_hr: float = Field(
        ...,
        gt=0,
        le=2000,
        description="Maximum heat output capacity (MMBtu/hr)"
    )
    steam_capacity_lb_hr: Optional[float] = Field(
        default=None,
        gt=0,
        description="Steam generation capacity (lb/hr)"
    )
    design_pressure_psig: float = Field(
        default=150.0,
        ge=0,
        le=3000,
        description="Design steam pressure (psig)"
    )
    design_temperature_f: Optional[float] = Field(
        default=None,
        description="Design steam temperature for superheated (F)"
    )

    # Operating load limits
    min_load_pct: float = Field(
        default=25.0,
        ge=0,
        le=100,
        description="Minimum stable load percentage"
    )
    max_load_pct: float = Field(
        default=100.0,
        ge=50,
        le=120,
        description="Maximum allowable load percentage"
    )
    turndown_ratio: float = Field(
        default=4.0,
        ge=1.5,
        le=20.0,
        description="Burner turndown ratio (max/min firing rate)"
    )

    # Fuel capabilities
    fuel_types: List[FuelType] = Field(
        default_factory=lambda: [FuelType.NATURAL_GAS],
        description="Supported fuel types"
    )
    primary_fuel: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Primary/design fuel"
    )
    dual_fuel_capable: bool = Field(
        default=False,
        description="Can fire two fuels simultaneously"
    )
    fuel_switch_time_minutes: int = Field(
        default=15,
        ge=5,
        le=120,
        description="Time to switch between fuels (minutes)"
    )

    # Efficiency
    efficiency_curve: EfficiencyCurveConfig = Field(
        default_factory=EfficiencyCurveConfig,
        description="Efficiency curve configuration"
    )
    design_efficiency_pct: float = Field(
        default=82.0,
        ge=50,
        le=98,
        description="Design point efficiency (%)"
    )

    # Ramp rates per NFPA 85
    ramp_up_rate_pct_min: float = Field(
        default=5.0,
        ge=0.5,
        le=20.0,
        description="Maximum load increase rate (%/min)"
    )
    ramp_down_rate_pct_min: float = Field(
        default=5.0,
        ge=0.5,
        le=20.0,
        description="Maximum load decrease rate (%/min)"
    )
    cold_start_time_minutes: int = Field(
        default=30,
        ge=10,
        le=180,
        description="Cold start time including purge (minutes)"
    )
    warm_start_time_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Warm start time (minutes)"
    )
    hot_start_time_minutes: int = Field(
        default=5,
        ge=2,
        le=30,
        description="Hot start time (minutes)"
    )

    # Operating status
    status: EquipmentStatus = Field(
        default=EquipmentStatus.AVAILABLE,
        description="Current operational status"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Dispatch priority (1=highest, 10=lowest)"
    )
    base_load: bool = Field(
        default=False,
        description="Designate as base load unit"
    )

    # Maintenance
    operating_hours: float = Field(
        default=0.0,
        ge=0,
        description="Total operating hours"
    )
    last_maintenance_date: Optional[datetime] = Field(
        default=None,
        description="Date of last major maintenance"
    )
    next_maintenance_hours: Optional[float] = Field(
        default=None,
        description="Hours until next scheduled maintenance"
    )

    # Cost factors
    variable_om_cost_per_mmbtu: float = Field(
        default=0.50,
        ge=0,
        description="Variable O&M cost ($/MMBtu)"
    )
    startup_cost_usd: float = Field(
        default=500.0,
        ge=0,
        description="Cost per cold start ($)"
    )

    @validator("max_load_pct")
    def validate_load_range(cls, v, values):
        """Ensure max load is greater than min load."""
        if "min_load_pct" in values and v <= values["min_load_pct"]:
            raise ValueError("max_load_pct must be > min_load_pct")
        return v

    @validator("boiler_name", always=True)
    def set_default_name(cls, v, values):
        """Set default name from boiler_id."""
        if not v and "boiler_id" in values:
            return f"Boiler {values['boiler_id']}"
        return v

    @validator("primary_fuel")
    def validate_primary_in_fuel_list(cls, v, values):
        """Ensure primary fuel is in fuel_types list."""
        if "fuel_types" in values and v not in values["fuel_types"]:
            raise ValueError("primary_fuel must be in fuel_types list")
        return v

    class Config:
        use_enum_values = True


class TubeLimitsConfig(BaseModel):
    """
    Furnace tube temperature and stress limits per API 560/530.

    Defines thermal constraints to prevent tube damage and ensure
    safe operation of process heaters and furnaces.
    """

    max_tube_metal_temp_f: float = Field(
        default=1100.0,
        ge=500,
        le=2000,
        description="Maximum allowable tube metal temperature (F)"
    )
    design_tube_metal_temp_f: float = Field(
        default=1000.0,
        ge=400,
        le=1800,
        description="Design tube metal temperature (F)"
    )
    max_heat_flux_btu_hr_ft2: float = Field(
        default=12000.0,
        ge=5000,
        le=50000,
        description="Maximum heat flux (Btu/hr-ft2)"
    )
    min_process_flow_pct: float = Field(
        default=30.0,
        ge=10,
        le=50,
        description="Minimum process flow to prevent overheating (%)"
    )

    # Tube material limits per API 530
    tube_material: str = Field(
        default="9Cr-1Mo",
        description="Tube material specification"
    )
    allowable_stress_psi: float = Field(
        default=10000.0,
        gt=0,
        description="Allowable stress at design temperature (psi)"
    )


class FurnaceConfig(BaseModel):
    """
    Configuration for a single furnace/process heater in the fleet.

    Defines physical characteristics, operating limits, efficiency curves,
    and process constraints for fired heaters per API 560.

    Attributes:
        furnace_id: Unique furnace identifier
        furnace_type: Type of furnace construction
        capacity_mmbtu_hr: Maximum heat release
        min_firing_rate_pct: Minimum stable firing
        max_firing_rate_pct: Maximum allowable firing
        process_type: Process application (heating, reforming, etc.)
        tube_limits: Tube temperature and flux limits

    Example:
        >>> furnace = FurnaceConfig(
        ...     furnace_id="FUR-001",
        ...     furnace_type=FurnaceType.CABIN,
        ...     capacity_mmbtu_hr=150.0,
        ...     min_firing_rate_pct=20.0,
        ...     max_firing_rate_pct=100.0,
        ...     process_type="crude_preheat",
        ... )
    """

    # Identification
    furnace_id: str = Field(
        ...,
        description="Unique furnace identifier"
    )
    furnace_tag: str = Field(
        default="",
        description="Plant equipment tag (e.g., H-101)"
    )
    furnace_name: str = Field(
        default="",
        description="Human-readable furnace name"
    )
    furnace_type: FurnaceType = Field(
        default=FurnaceType.CABIN,
        description="Furnace type classification per API 560"
    )
    process_type: str = Field(
        default="general_heating",
        description="Process application (crude_preheat, reformer, etc.)"
    )

    # Capacity specifications
    capacity_mmbtu_hr: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Maximum heat release (MMBtu/hr)"
    )
    design_duty_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Design heat duty (MMBtu/hr)"
    )
    radiant_section_pct: float = Field(
        default=70.0,
        ge=50,
        le=90,
        description="Radiant section heat absorption (%)"
    )
    convection_section_pct: float = Field(
        default=30.0,
        ge=10,
        le=50,
        description="Convection section heat absorption (%)"
    )

    # Firing rate limits
    min_firing_rate_pct: float = Field(
        default=20.0,
        ge=0,
        le=100,
        description="Minimum stable firing rate (%)"
    )
    max_firing_rate_pct: float = Field(
        default=100.0,
        ge=50,
        le=120,
        description="Maximum allowable firing rate (%)"
    )

    # Fuel capabilities
    fuel_types: List[FuelType] = Field(
        default_factory=lambda: [FuelType.REFINERY_FUEL_GAS, FuelType.NATURAL_GAS],
        description="Supported fuel types"
    )
    primary_fuel: FuelType = Field(
        default=FuelType.REFINERY_FUEL_GAS,
        description="Primary/design fuel"
    )

    # Efficiency
    efficiency_curve: EfficiencyCurveConfig = Field(
        default_factory=EfficiencyCurveConfig,
        description="Efficiency curve configuration"
    )
    design_efficiency_pct: float = Field(
        default=85.0,
        ge=60,
        le=98,
        description="Design point efficiency (%)"
    )

    # Tube limits per API 560/530
    tube_limits: TubeLimitsConfig = Field(
        default_factory=TubeLimitsConfig,
        description="Tube temperature and flux limits"
    )

    # Ramp rates
    ramp_up_rate_pct_min: float = Field(
        default=3.0,
        ge=0.5,
        le=15.0,
        description="Maximum firing increase rate (%/min)"
    )
    ramp_down_rate_pct_min: float = Field(
        default=5.0,
        ge=0.5,
        le=20.0,
        description="Maximum firing decrease rate (%/min)"
    )
    startup_time_minutes: int = Field(
        default=60,
        ge=15,
        le=240,
        description="Startup time including warmup (minutes)"
    )

    # Operating status
    status: EquipmentStatus = Field(
        default=EquipmentStatus.AVAILABLE,
        description="Current operational status"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Dispatch priority (1=highest)"
    )

    # Cost factors
    variable_om_cost_per_mmbtu: float = Field(
        default=0.30,
        ge=0,
        description="Variable O&M cost ($/MMBtu)"
    )
    startup_cost_usd: float = Field(
        default=1000.0,
        ge=0,
        description="Cost per startup ($)"
    )

    @validator("max_firing_rate_pct")
    def validate_firing_range(cls, v, values):
        """Ensure max firing is greater than min."""
        if "min_firing_rate_pct" in values and v <= values["min_firing_rate_pct"]:
            raise ValueError("max_firing_rate_pct must be > min_firing_rate_pct")
        return v

    @validator("furnace_name", always=True)
    def set_default_name(cls, v, values):
        """Set default name from furnace_id."""
        if not v and "furnace_id" in values:
            return f"Furnace {values['furnace_id']}"
        return v

    @validator("design_duty_mmbtu_hr", always=True)
    def set_default_duty(cls, v, values):
        """Set default design duty from capacity."""
        if v == 0.0 and "capacity_mmbtu_hr" in values:
            return values["capacity_mmbtu_hr"] * 0.85
        return v

    class Config:
        use_enum_values = True


class EquipmentFleetConfig(BaseModel):
    """
    Configuration for the complete equipment fleet.

    Aggregates all boilers and furnaces that participate in
    load balancing optimization.

    Attributes:
        boilers: List of boiler configurations
        furnaces: List of furnace configurations
        total_capacity_mmbtu_hr: Auto-calculated total fleet capacity

    Example:
        >>> fleet = EquipmentFleetConfig(
        ...     boilers=[
        ...         BoilerConfig(boiler_id="BLR-001", capacity_mmbtu_hr=100),
        ...         BoilerConfig(boiler_id="BLR-002", capacity_mmbtu_hr=100),
        ...     ],
        ...     furnaces=[
        ...         FurnaceConfig(furnace_id="FUR-001", capacity_mmbtu_hr=150),
        ...     ]
        ... )
    """

    boilers: List[BoilerConfig] = Field(
        default_factory=list,
        description="List of boiler configurations"
    )
    furnaces: List[FurnaceConfig] = Field(
        default_factory=list,
        description="List of furnace configurations"
    )

    # Fleet parameters
    max_simultaneous_starts: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum equipment units to start simultaneously"
    )
    staggered_start_delay_s: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Delay between staggered starts (seconds)"
    )

    # Coordination
    load_sharing_enabled: bool = Field(
        default=True,
        description="Enable automatic load sharing"
    )
    equal_percentage_bias: float = Field(
        default=0.0,
        ge=-10,
        le=10,
        description="Bias toward equal percentage loading (%)"
    )

    @validator("boilers", "furnaces")
    def validate_unique_ids(cls, v):
        """Ensure all equipment IDs are unique."""
        if v:
            ids = [eq.boiler_id if hasattr(eq, "boiler_id") else eq.furnace_id for eq in v]
            if len(ids) != len(set(ids)):
                raise ValueError("Equipment IDs must be unique")
        return v


# =============================================================================
# FUEL CONFIGURATION
# =============================================================================


class FuelConfig(BaseModel):
    """
    Configuration for a single fuel type.

    Defines pricing, carbon intensity, and availability for
    multi-fuel optimization.

    Attributes:
        fuel_type: Fuel type identifier
        price_per_mmbtu: Current fuel price
        carbon_intensity_kg_co2_mmbtu: CO2 emission factor
        availability: Current availability status

    Carbon Intensity Reference (kg CO2/MMBtu):
        - Natural Gas: 53.07
        - No. 2 Fuel Oil: 73.15
        - No. 6 Fuel Oil: 75.10
        - Coal (bituminous): 93.30
        - Propane: 62.87
        - Hydrogen (green): 0.0
        - Biogas: ~0 (biogenic)
    """

    fuel_type: FuelType = Field(
        ...,
        description="Fuel type"
    )
    fuel_name: str = Field(
        default="",
        description="Human-readable fuel name"
    )

    # Pricing
    price_per_mmbtu: float = Field(
        ...,
        ge=0,
        le=200,
        description="Fuel price ($/MMBtu)"
    )
    price_volatility_pct: float = Field(
        default=10.0,
        ge=0,
        le=100,
        description="Expected price volatility (%)"
    )
    price_source: str = Field(
        default="contract",
        description="Price source (spot, contract, index)"
    )

    # Carbon intensity
    carbon_intensity_kg_co2_mmbtu: float = Field(
        ...,
        ge=0,
        le=150,
        description="CO2 emission factor (kg CO2/MMBtu)"
    )
    carbon_intensity_source: str = Field(
        default="epa_ghg",
        description="Source of emission factor (epa_ghg, ipcc, custom)"
    )

    # Availability
    availability: str = Field(
        default="available",
        description="Availability status (available, limited, unavailable)"
    )
    max_supply_rate_mmbtu_hr: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum supply rate if constrained (MMBtu/hr)"
    )

    # Heating value
    hhv_btu_scf: Optional[float] = Field(
        default=None,
        gt=0,
        description="Higher Heating Value for gases (Btu/SCF)"
    )
    hhv_btu_lb: Optional[float] = Field(
        default=None,
        gt=0,
        description="Higher Heating Value for liquids/solids (Btu/lb)"
    )

    @validator("fuel_name", always=True)
    def set_default_name(cls, v, values):
        """Set default name from fuel_type."""
        if not v and "fuel_type" in values:
            return str(values["fuel_type"]).replace("_", " ").title()
        return v

    class Config:
        use_enum_values = True


class FuelBlendConfig(BaseModel):
    """
    Configuration for fuel blending constraints.

    Defines limits on fuel mixing ratios and blend quality
    requirements (e.g., Wobbe Index for gas blending).
    """

    blending_enabled: bool = Field(
        default=False,
        description="Enable fuel blending"
    )

    # Blend ratio limits
    max_fuels_in_blend: int = Field(
        default=2,
        ge=2,
        le=4,
        description="Maximum fuels in a blend"
    )
    min_component_pct: float = Field(
        default=20.0,
        ge=5,
        le=50,
        description="Minimum component percentage in blend"
    )

    # Wobbe Index constraints for gas blending
    min_wobbe_index: float = Field(
        default=1300.0,
        ge=1000,
        le=1500,
        description="Minimum Wobbe Index (Btu/SCF)"
    )
    max_wobbe_index: float = Field(
        default=1400.0,
        ge=1100,
        le=1600,
        description="Maximum Wobbe Index (Btu/SCF)"
    )

    # Blend change rate
    max_blend_change_rate_pct_min: float = Field(
        default=2.0,
        ge=0.5,
        le=10,
        description="Maximum blend ratio change rate (%/min)"
    )


class MultiFuelConfig(BaseModel):
    """
    Complete multi-fuel configuration.

    Aggregates all fuel configurations and switching/blending constraints.

    Attributes:
        fuels: List of available fuel configurations
        blending: Blending constraints configuration
        switching_lockout_minutes: Minimum time between fuel switches

    Example:
        >>> config = MultiFuelConfig(
        ...     fuels=[
        ...         FuelConfig(fuel_type=FuelType.NATURAL_GAS,
        ...                    price_per_mmbtu=4.50,
        ...                    carbon_intensity_kg_co2_mmbtu=53.07),
        ...         FuelConfig(fuel_type=FuelType.NO2_FUEL_OIL,
        ...                    price_per_mmbtu=15.00,
        ...                    carbon_intensity_kg_co2_mmbtu=73.15),
        ...     ],
        ... )
    """

    fuels: List[FuelConfig] = Field(
        default_factory=list,
        description="Available fuel configurations"
    )
    blending: FuelBlendConfig = Field(
        default_factory=FuelBlendConfig,
        description="Blending constraints"
    )

    # Fuel switching
    switching_enabled: bool = Field(
        default=True,
        description="Enable automatic fuel switching"
    )
    switching_lockout_minutes: int = Field(
        default=60,
        ge=15,
        le=480,
        description="Minimum time between fuel switches (minutes)"
    )
    economic_switch_threshold_pct: float = Field(
        default=15.0,
        ge=5,
        le=50,
        description="Minimum price differential to trigger switch (%)"
    )

    # Carbon pricing
    carbon_price_usd_ton: float = Field(
        default=50.0,
        ge=0,
        le=500,
        description="Carbon price for optimization ($/ton CO2)"
    )
    include_carbon_in_optimization: bool = Field(
        default=True,
        description="Include carbon cost in objective function"
    )

    @validator("fuels")
    def validate_unique_fuels(cls, v):
        """Ensure fuel types are unique."""
        if v:
            types = [f.fuel_type for f in v]
            if len(types) != len(set(types)):
                raise ValueError("Fuel types must be unique")
        return v


# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================


class MILPConfig(BaseModel):
    """
    Mixed Integer Linear Programming (MILP) solver configuration.

    Configures the optimization solver for economic dispatch and
    unit commitment problems.

    Attributes:
        solver_type: MILP solver selection
        time_limit_s: Maximum solve time
        gap_tolerance: Optimality gap tolerance
        max_iterations: Maximum solver iterations

    Solver Selection Guide:
        - CBC: Good open-source option, moderate problems
        - GLPK: Simple problems, educational use
        - HiGHS: Fast open-source, good for LP relaxations
        - CPLEX/Gurobi: Large-scale, commercial applications

    Example:
        >>> config = MILPConfig(
        ...     solver_type=SolverType.CBC,
        ...     time_limit_s=60,
        ...     gap_tolerance=0.01,
        ... )
    """

    solver_type: SolverType = Field(
        default=SolverType.CBC,
        description="MILP solver type"
    )

    # Solver parameters
    time_limit_s: int = Field(
        default=60,
        ge=5,
        le=3600,
        description="Maximum solve time (seconds)"
    )
    gap_tolerance: float = Field(
        default=0.01,
        ge=0.0001,
        le=0.1,
        description="Optimality gap tolerance (0.01 = 1%)"
    )
    absolute_gap: float = Field(
        default=100.0,
        ge=0,
        description="Absolute gap tolerance ($)"
    )
    max_iterations: int = Field(
        default=1000000,
        ge=1000,
        le=100000000,
        description="Maximum solver iterations"
    )

    # Solution quality
    feasibility_tolerance: float = Field(
        default=1e-6,
        ge=1e-10,
        le=1e-3,
        description="Constraint feasibility tolerance"
    )
    integrality_tolerance: float = Field(
        default=1e-5,
        ge=1e-10,
        le=1e-3,
        description="Integer variable tolerance"
    )

    # Solver behavior
    presolve_enabled: bool = Field(
        default=True,
        description="Enable solver presolve"
    )
    cuts_enabled: bool = Field(
        default=True,
        description="Enable cutting planes"
    )
    threads: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of solver threads"
    )

    # Warm start
    warm_start_enabled: bool = Field(
        default=True,
        description="Use previous solution as warm start"
    )

    # Logging
    solver_log_enabled: bool = Field(
        default=False,
        description="Enable detailed solver logging"
    )
    solver_log_level: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Solver log verbosity (0-3)"
    )

    class Config:
        use_enum_values = True


class EconomicDispatchConfig(BaseModel):
    """
    Economic dispatch optimization configuration.

    Defines objective function, constraints, and parameters for
    optimal load allocation across equipment fleet.

    Attributes:
        objective: Optimization objective function
        cost_weight: Weight for cost minimization
        emissions_weight: Weight for emissions minimization
        efficiency_weight: Weight for efficiency maximization

    Example:
        >>> config = EconomicDispatchConfig(
        ...     objective=OptimizationObjective.PARETO_COST_EMISSIONS,
        ...     cost_weight=0.6,
        ...     emissions_weight=0.3,
        ...     efficiency_weight=0.1,
        ... )
    """

    objective: OptimizationObjective = Field(
        default=OptimizationObjective.MINIMIZE_COST,
        description="Primary optimization objective"
    )

    # Multi-objective weights (must sum to 1.0)
    cost_weight: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Weight for cost minimization"
    )
    emissions_weight: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Weight for emissions minimization"
    )
    efficiency_weight: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Weight for efficiency maximization"
    )

    # Optimization horizon
    horizon_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Optimization horizon (hours)"
    )
    time_step_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Optimization time step (minutes)"
    )
    rolling_window_hours: int = Field(
        default=1,
        ge=1,
        le=24,
        description="Rolling optimization window (hours)"
    )

    # Load constraints
    enforce_demand_balance: bool = Field(
        default=True,
        description="Strictly enforce demand balance"
    )
    demand_tolerance_pct: float = Field(
        default=1.0,
        ge=0,
        le=10,
        description="Allowable demand imbalance (%)"
    )

    # Unit commitment
    include_startup_costs: bool = Field(
        default=True,
        description="Include startup costs in optimization"
    )
    include_shutdown_costs: bool = Field(
        default=False,
        description="Include shutdown costs"
    )
    commitment_lookahead_hours: int = Field(
        default=4,
        ge=1,
        le=24,
        description="Look-ahead for unit commitment decisions"
    )

    # Reserves
    spinning_reserve_pct: float = Field(
        default=10.0,
        ge=0,
        le=30,
        description="Spinning reserve requirement (%)"
    )
    non_spinning_reserve_pct: float = Field(
        default=5.0,
        ge=0,
        le=20,
        description="Non-spinning reserve requirement (%)"
    )

    @validator("efficiency_weight")
    def validate_weights_sum(cls, v, values):
        """Ensure weights sum to 1.0."""
        total = v
        if "cost_weight" in values:
            total += values["cost_weight"]
        if "emissions_weight" in values:
            total += values["emissions_weight"]
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v

    class Config:
        use_enum_values = True


class ConstraintConfig(BaseModel):
    """
    Operational constraint configuration.

    Defines ramp rates, minimum run/down times, and startup
    restrictions for unit commitment optimization.

    Attributes:
        ramp_rate_limits_enabled: Enforce ramp rate limits
        min_run_time_hours: Minimum time unit must run once started
        min_down_time_hours: Minimum time unit must be off once stopped
        startup_costs_enabled: Include startup costs in optimization
    """

    # Ramp rate constraints
    ramp_rate_limits_enabled: bool = Field(
        default=True,
        description="Enforce equipment ramp rate limits"
    )
    global_ramp_limit_pct_min: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=20,
        description="Global ramp rate limit override (%/min)"
    )

    # Minimum run/down time
    min_run_time_enabled: bool = Field(
        default=True,
        description="Enforce minimum run time"
    )
    default_min_run_time_hours: float = Field(
        default=2.0,
        ge=0.5,
        le=24,
        description="Default minimum run time (hours)"
    )

    min_down_time_enabled: bool = Field(
        default=True,
        description="Enforce minimum down time"
    )
    default_min_down_time_hours: float = Field(
        default=1.0,
        ge=0.5,
        le=24,
        description="Default minimum down time (hours)"
    )

    # Startup constraints
    max_starts_per_day: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Maximum starts per unit per day"
    )
    cold_start_threshold_hours: float = Field(
        default=8.0,
        ge=2,
        le=24,
        description="Hours after shutdown to be considered cold start"
    )

    # Load constraints
    enforce_min_load: bool = Field(
        default=True,
        description="Enforce minimum stable load"
    )
    enforce_max_load: bool = Field(
        default=True,
        description="Enforce maximum load"
    )

    # Fuel constraints
    fuel_availability_check: bool = Field(
        default=True,
        description="Check fuel availability in dispatch"
    )


class OptimizationConfig(BaseModel):
    """
    Complete optimization configuration.

    Combines MILP solver settings, economic dispatch parameters,
    and operational constraints.

    Attributes:
        milp: MILP solver configuration
        dispatch: Economic dispatch configuration
        constraints: Operational constraints
        enabled: Master enable for optimization

    Example:
        >>> config = OptimizationConfig(
        ...     enabled=True,
        ...     milp=MILPConfig(solver_type=SolverType.CBC),
        ...     dispatch=EconomicDispatchConfig(
        ...         objective=OptimizationObjective.MINIMIZE_COST
        ...     ),
        ... )
    """

    enabled: bool = Field(
        default=True,
        description="Enable optimization"
    )

    milp: MILPConfig = Field(
        default_factory=MILPConfig,
        description="MILP solver configuration"
    )
    dispatch: EconomicDispatchConfig = Field(
        default_factory=EconomicDispatchConfig,
        description="Economic dispatch configuration"
    )
    constraints: ConstraintConfig = Field(
        default_factory=ConstraintConfig,
        description="Operational constraints"
    )

    # Execution settings
    execution_interval_s: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Optimization execution interval (seconds)"
    )
    adaptive_interval_enabled: bool = Field(
        default=True,
        description="Adjust interval based on load variability"
    )

    # Solution handling
    auto_implement: bool = Field(
        default=False,
        description="Auto-implement optimization results"
    )
    operator_approval_required: bool = Field(
        default=True,
        description="Require operator approval for changes"
    )
    approval_timeout_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Operator approval timeout (minutes)"
    )

    # Fallback
    fallback_to_heuristic: bool = Field(
        default=True,
        description="Use heuristic if MILP fails"
    )
    heuristic_method: str = Field(
        default="equal_percentage",
        description="Heuristic method (equal_percentage, merit_order)"
    )


# =============================================================================
# DEMAND CONFIGURATION
# =============================================================================


class DemandProfileConfig(BaseModel):
    """
    Demand profile configuration for load forecasting.

    Defines base load, peak load, and forecast parameters for
    demand-side optimization.

    Attributes:
        base_load_mmbtu_hr: Minimum expected demand
        peak_load_mmbtu_hr: Maximum expected demand
        load_forecast_horizon_hr: Forecast look-ahead period
    """

    # Load levels
    base_load_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Base/minimum demand (MMBtu/hr)"
    )
    peak_load_mmbtu_hr: float = Field(
        ...,
        gt=0,
        description="Peak/maximum demand (MMBtu/hr)"
    )
    average_load_mmbtu_hr: Optional[float] = Field(
        default=None,
        gt=0,
        description="Average demand (MMBtu/hr)"
    )

    # Load profile
    profile_type: str = Field(
        default="constant",
        description="Profile type (constant, daily_pattern, forecast)"
    )
    daily_profile_enabled: bool = Field(
        default=False,
        description="Use hourly daily profile"
    )
    hourly_factors: List[float] = Field(
        default_factory=lambda: [1.0] * 24,
        description="24-hour load factors (multiplier on base load)"
    )

    # Forecasting
    load_forecast_enabled: bool = Field(
        default=True,
        description="Enable load forecasting"
    )
    load_forecast_horizon_hr: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Forecast horizon (hours)"
    )
    forecast_update_interval_min: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Forecast update interval (minutes)"
    )

    # Weather correlation
    weather_correlation_enabled: bool = Field(
        default=False,
        description="Include weather in demand forecast"
    )
    heating_degree_day_base_f: float = Field(
        default=65.0,
        ge=50,
        le=75,
        description="Base temperature for HDD calculation (F)"
    )

    @validator("peak_load_mmbtu_hr")
    def validate_peak_vs_base(cls, v, values):
        """Ensure peak load is greater than base load."""
        if "base_load_mmbtu_hr" in values and v <= values["base_load_mmbtu_hr"]:
            raise ValueError("peak_load_mmbtu_hr must be > base_load_mmbtu_hr")
        return v

    @validator("average_load_mmbtu_hr", always=True)
    def set_default_average(cls, v, values):
        """Set default average from base and peak."""
        if v is None:
            base = values.get("base_load_mmbtu_hr", 0)
            peak = values.get("peak_load_mmbtu_hr", 0)
            return (base + peak) / 2
        return v

    @validator("hourly_factors")
    def validate_hourly_factors(cls, v):
        """Ensure 24 hourly factors provided."""
        if len(v) != 24:
            raise ValueError("hourly_factors must have 24 values")
        if any(f < 0 or f > 3.0 for f in v):
            raise ValueError("hourly_factors must be between 0 and 3.0")
        return v


class DemandResponseConfig(BaseModel):
    """
    Demand response configuration.

    Defines curtailment limits and priority loads for
    demand-side management during supply constraints.

    Attributes:
        enabled: Enable demand response capability
        curtailment_limits_pct: Maximum allowable curtailment
        priority_loads: List of priority load identifiers
    """

    enabled: bool = Field(
        default=False,
        description="Enable demand response"
    )

    # Curtailment
    max_curtailment_pct: float = Field(
        default=20.0,
        ge=0,
        le=50,
        description="Maximum demand curtailment (%)"
    )
    curtailment_rate_pct_min: float = Field(
        default=5.0,
        ge=1,
        le=20,
        description="Maximum curtailment rate (%/min)"
    )
    curtailment_duration_limit_hr: float = Field(
        default=4.0,
        ge=0.5,
        le=24,
        description="Maximum curtailment duration (hours)"
    )

    # Priority loads
    priority_loads: List[str] = Field(
        default_factory=list,
        description="List of priority load identifiers (never curtailed)"
    )
    curtailable_loads: List[str] = Field(
        default_factory=list,
        description="List of curtailable load identifiers"
    )

    # Economic parameters
    curtailment_cost_per_mmbtu: float = Field(
        default=100.0,
        ge=0,
        description="Cost of curtailment ($/MMBtu not supplied)"
    )

    # Triggers
    trigger_on_price_spike: bool = Field(
        default=True,
        description="Trigger DR on fuel price spike"
    )
    price_spike_threshold_pct: float = Field(
        default=50.0,
        ge=10,
        le=200,
        description="Price increase to trigger DR (%)"
    )
    trigger_on_capacity_limit: bool = Field(
        default=True,
        description="Trigger DR when approaching capacity limit"
    )
    capacity_trigger_pct: float = Field(
        default=95.0,
        ge=80,
        le=100,
        description="Capacity utilization to trigger DR (%)"
    )


class DemandConfig(BaseModel):
    """
    Complete demand configuration.

    Combines demand profile and demand response settings.

    Example:
        >>> config = DemandConfig(
        ...     profile=DemandProfileConfig(
        ...         base_load_mmbtu_hr=200,
        ...         peak_load_mmbtu_hr=400,
        ...     ),
        ...     demand_response=DemandResponseConfig(enabled=True),
        ... )
    """

    profile: DemandProfileConfig = Field(
        ...,
        description="Demand profile configuration"
    )
    demand_response: DemandResponseConfig = Field(
        default_factory=DemandResponseConfig,
        description="Demand response configuration"
    )

    # Demand monitoring
    monitoring_interval_s: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Demand monitoring interval (seconds)"
    )
    averaging_window_s: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Demand averaging window (seconds)"
    )


# =============================================================================
# SAFETY CONFIGURATION
# =============================================================================


class SafetyLimitsConfig(BaseModel):
    """
    Safety limits for equipment operation.

    Defines hard limits that override optimization to ensure
    safe equipment operation per ASME CSD-1 and NFPA 85.
    """

    # Capacity limits
    max_total_capacity_pct: float = Field(
        default=95.0,
        ge=80,
        le=100,
        description="Maximum total fleet capacity utilization (%)"
    )
    min_running_units: int = Field(
        default=1,
        ge=0,
        description="Minimum number of units that must be running"
    )

    # Redundancy
    n_plus_1_redundancy: bool = Field(
        default=True,
        description="Maintain N+1 redundancy"
    )
    n_plus_redundancy_count: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Number of redundant units to maintain"
    )

    # Reserve capacity
    emergency_reserve_pct: float = Field(
        default=15.0,
        ge=0,
        le=30,
        description="Emergency reserve capacity (%)"
    )
    spinning_reserve_mmbtu_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Spinning reserve capacity (MMBtu/hr)"
    )

    # Load limits
    max_load_rate_change_pct_min: float = Field(
        default=10.0,
        ge=1,
        le=20,
        description="Maximum fleet load change rate (%/min)"
    )


class InterlockConfig(BaseModel):
    """
    Equipment coordination interlock configuration.

    Defines interlocks for safe coordination between
    multiple boilers and furnaces.
    """

    # Equipment interlocks
    shared_fuel_header_interlock: bool = Field(
        default=True,
        description="Interlock for shared fuel header pressure"
    )
    fuel_header_min_pressure_psig: float = Field(
        default=10.0,
        ge=5,
        le=50,
        description="Minimum fuel header pressure (psig)"
    )

    shared_steam_header_interlock: bool = Field(
        default=True,
        description="Interlock for shared steam header"
    )
    steam_header_high_pressure_psig: float = Field(
        default=160.0,
        ge=50,
        le=3000,
        description="High pressure trip setpoint (psig)"
    )
    steam_header_low_pressure_psig: float = Field(
        default=130.0,
        ge=0,
        le=2500,
        description="Low pressure trip setpoint (psig)"
    )

    # Cascade trip
    cascade_trip_enabled: bool = Field(
        default=False,
        description="Trip multiple units on single failure"
    )
    trip_delay_s: float = Field(
        default=5.0,
        ge=1,
        le=30,
        description="Delay before cascade trip (seconds)"
    )

    # Start permissives
    max_simultaneous_starts: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum simultaneous unit starts"
    )
    stagger_start_delay_s: int = Field(
        default=30,
        ge=10,
        le=120,
        description="Delay between staggered starts (seconds)"
    )


class SafetyConfig(BaseModel):
    """
    Complete safety configuration.

    Combines safety limits and interlocks per ASME CSD-1,
    NFPA 85, and ISA 84 requirements.

    Attributes:
        sil_rating: Overall system SIL rating
        limits: Safety limits configuration
        interlocks: Interlock configuration
        asme_csd1_compliance: Enable ASME CSD-1 checks
        nfpa_85_compliance: Enable NFPA 85 checks

    Example:
        >>> config = SafetyConfig(
        ...     sil_rating=SafetyIntegrityLevel.SIL_2,
        ...     limits=SafetyLimitsConfig(n_plus_1_redundancy=True),
        ... )
    """

    sil_rating: SafetyIntegrityLevel = Field(
        default=SafetyIntegrityLevel.SIL_2,
        description="Safety Integrity Level"
    )

    limits: SafetyLimitsConfig = Field(
        default_factory=SafetyLimitsConfig,
        description="Safety limits configuration"
    )
    interlocks: InterlockConfig = Field(
        default_factory=InterlockConfig,
        description="Interlock configuration"
    )

    # Standards compliance
    asme_csd1_compliance: bool = Field(
        default=True,
        description="Enable ASME CSD-1 compliance checks"
    )
    nfpa_85_compliance: bool = Field(
        default=True,
        description="Enable NFPA 85 compliance checks"
    )
    api_560_compliance: bool = Field(
        default=True,
        description="Enable API 560 compliance for furnaces"
    )

    # Audit
    safety_event_logging: bool = Field(
        default=True,
        description="Log all safety-related events"
    )
    proof_test_interval_months: int = Field(
        default=12,
        ge=3,
        le=60,
        description="SIS proof test interval (months)"
    )

    # Emergency shutdown
    emergency_shutdown_enabled: bool = Field(
        default=True,
        description="Enable emergency shutdown capability"
    )
    esd_response_time_s: float = Field(
        default=5.0,
        ge=1,
        le=30,
        description="ESD response time (seconds)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# INTEGRATION CONFIGURATION
# =============================================================================


class OPCUAEndpointConfig(BaseModel):
    """Configuration for a single OPC-UA endpoint."""

    endpoint_id: str = Field(
        ...,
        description="Endpoint identifier"
    )
    endpoint_url: str = Field(
        ...,
        description="OPC-UA endpoint URL"
    )
    namespace_uri: str = Field(
        default="urn:greenlang:heatloadbalancer",
        description="Namespace URI"
    )
    equipment_ids: List[str] = Field(
        default_factory=list,
        description="Equipment IDs served by this endpoint"
    )
    security_policy: OPCSecurityPolicy = Field(
        default=OPCSecurityPolicy.BASIC256SHA256,
        description="Security policy"
    )
    security_mode: OPCSecurityMode = Field(
        default=OPCSecurityMode.SIGN_AND_ENCRYPT,
        description="Security mode"
    )

    class Config:
        use_enum_values = True


class OPCUAConfig(BaseModel):
    """
    OPC-UA integration configuration for multiple equipment.

    Configures OPC-UA client connections for real-time data
    acquisition from industrial control systems.

    Attributes:
        enabled: Enable OPC-UA integration
        endpoints: List of OPC-UA endpoint configurations
        reconnect_interval_ms: Reconnection interval

    Example:
        >>> config = OPCUAConfig(
        ...     enabled=True,
        ...     endpoints=[
        ...         OPCUAEndpointConfig(
        ...             endpoint_id="boilers",
        ...             endpoint_url="opc.tcp://plc1:4840/boilers",
        ...             equipment_ids=["BLR-001", "BLR-002"],
        ...         ),
        ...     ],
        ... )
    """

    enabled: bool = Field(
        default=False,
        description="Enable OPC-UA integration"
    )

    endpoints: List[OPCUAEndpointConfig] = Field(
        default_factory=list,
        description="OPC-UA endpoint configurations"
    )

    # Connection settings
    timeout_ms: int = Field(
        default=30000,
        ge=5000,
        le=120000,
        description="Connection timeout (milliseconds)"
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
    publishing_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Subscription publishing interval (ms)"
    )
    sampling_interval_ms: int = Field(
        default=500,
        ge=50,
        le=30000,
        description="Node sampling interval (ms)"
    )

    # Authentication
    username: Optional[str] = Field(
        default=None,
        description="Username for authentication"
    )
    password_secret_name: Optional[str] = Field(
        default=None,
        description="Secret name for password"
    )
    certificate_path: Optional[str] = Field(
        default=None,
        description="Client certificate path"
    )

    class Config:
        use_enum_values = True


class KafkaTopicConfig(BaseModel):
    """Configuration for a Kafka topic."""

    topic_name: str = Field(
        ...,
        description="Kafka topic name"
    )
    direction: str = Field(
        default="produce",
        description="Direction (produce, consume, both)"
    )
    partition_count: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of partitions"
    )
    retention_hours: int = Field(
        default=168,
        ge=1,
        le=8760,
        description="Message retention (hours)"
    )


class KafkaConfig(BaseModel):
    """
    Kafka integration configuration for event streaming.

    Configures Kafka producer/consumer for load commands,
    feedback, and optimization results.

    Attributes:
        enabled: Enable Kafka integration
        bootstrap_servers: Kafka broker addresses
        topics: Topic configurations for different message types

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

    # Connection
    bootstrap_servers: List[str] = Field(
        default_factory=lambda: ["localhost:9092"],
        description="Kafka broker addresses"
    )
    client_id: str = Field(
        default="gl023-heat-load-balancer",
        description="Kafka client ID"
    )
    group_id: str = Field(
        default="gl023-hlb-group",
        description="Consumer group ID"
    )

    # Security
    security_protocol: KafkaSecurityProtocol = Field(
        default=KafkaSecurityProtocol.PLAINTEXT,
        description="Security protocol"
    )
    sasl_mechanism: Optional[str] = Field(
        default=None,
        description="SASL mechanism"
    )
    sasl_username: Optional[str] = Field(
        default=None,
        description="SASL username"
    )
    sasl_password_secret_name: Optional[str] = Field(
        default=None,
        description="Secret name for SASL password"
    )

    # Topics
    topics: List[KafkaTopicConfig] = Field(
        default_factory=lambda: [
            KafkaTopicConfig(
                topic_name="greenlang.hlb.load_commands",
                direction="produce",
            ),
            KafkaTopicConfig(
                topic_name="greenlang.hlb.equipment_feedback",
                direction="consume",
            ),
            KafkaTopicConfig(
                topic_name="greenlang.hlb.optimization_results",
                direction="produce",
            ),
            KafkaTopicConfig(
                topic_name="greenlang.hlb.demand_forecast",
                direction="consume",
            ),
        ],
        description="Topic configurations"
    )

    # Producer settings
    producer_acks: str = Field(
        default="all",
        description="Producer acknowledgment level"
    )
    producer_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Producer retry attempts"
    )

    # Consumer settings
    auto_offset_reset: str = Field(
        default="latest",
        description="Auto offset reset policy"
    )
    max_poll_records: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Max records per poll"
    )

    class Config:
        use_enum_values = True


class SCADAConfig(BaseModel):
    """
    SCADA integration configuration for multi-equipment coordination.

    Configures communication with plant SCADA/DCS for
    setpoint distribution and status monitoring.

    Attributes:
        enabled: Enable SCADA integration
        protocol: Communication protocol (modbus, dnp3, opc)
        master_station_address: SCADA master station address
    """

    enabled: bool = Field(
        default=False,
        description="Enable SCADA integration"
    )

    # Protocol
    protocol: str = Field(
        default="modbus_tcp",
        description="Protocol (modbus_tcp, dnp3, opc_da)"
    )

    # Connection
    master_station_address: str = Field(
        default="192.168.1.100",
        description="SCADA master station address"
    )
    port: int = Field(
        default=502,
        ge=1,
        le=65535,
        description="Communication port"
    )

    # Modbus settings
    unit_id: int = Field(
        default=1,
        ge=1,
        le=247,
        description="Modbus unit ID"
    )
    register_map_path: Optional[str] = Field(
        default=None,
        description="Path to register map configuration"
    )

    # Timing
    poll_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=30000,
        description="Poll interval (milliseconds)"
    )
    timeout_ms: int = Field(
        default=5000,
        ge=1000,
        le=30000,
        description="Communication timeout (ms)"
    )

    # Setpoint writing
    setpoint_write_enabled: bool = Field(
        default=True,
        description="Enable setpoint writing to SCADA"
    )
    write_confirmation_required: bool = Field(
        default=True,
        description="Require write confirmation"
    )


class IntegrationConfig(BaseModel):
    """
    Complete integration configuration.

    Combines OPC-UA, Kafka, and SCADA configurations.
    """

    opcua: OPCUAConfig = Field(
        default_factory=OPCUAConfig,
        description="OPC-UA configuration"
    )
    kafka: KafkaConfig = Field(
        default_factory=KafkaConfig,
        description="Kafka configuration"
    )
    scada: SCADAConfig = Field(
        default_factory=SCADAConfig,
        description="SCADA configuration"
    )

    # Data historian
    historian_enabled: bool = Field(
        default=True,
        description="Enable historian integration"
    )
    historian_type: str = Field(
        default="influxdb",
        description="Historian type (influxdb, pi, ip21)"
    )
    historian_write_interval_s: int = Field(
        default=1,
        ge=1,
        le=60,
        description="Historian write interval (seconds)"
    )


# =============================================================================
# EXPLAINABILITY CONFIGURATION
# =============================================================================


class SHAPConfig(BaseModel):
    """
    SHAP configuration for optimization decision explanation.

    Explains which factors (load, efficiency, cost, constraints)
    influenced the optimization solution.
    """

    enabled: bool = Field(
        default=True,
        description="Enable SHAP explanations"
    )
    method: str = Field(
        default="kernel",
        description="SHAP method (kernel, linear, sampling)"
    )
    n_samples: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of samples for SHAP"
    )
    max_features: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Maximum features to include"
    )
    cache_explanations: bool = Field(
        default=True,
        description="Cache SHAP explanations"
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Cache TTL (hours)"
    )


class LIMEConfig(BaseModel):
    """
    LIME configuration for local optimization explanations.

    Provides local interpretable explanations for specific
    dispatch decisions.
    """

    enabled: bool = Field(
        default=True,
        description="Enable LIME explanations"
    )
    mode: str = Field(
        default="tabular",
        description="LIME mode"
    )
    num_features: int = Field(
        default=10,
        ge=3,
        le=30,
        description="Number of features in explanation"
    )
    num_samples: int = Field(
        default=5000,
        ge=500,
        le=20000,
        description="Number of samples for LIME"
    )
    discretize_continuous: bool = Field(
        default=True,
        description="Discretize continuous features"
    )


class ExplainabilityConfig(BaseModel):
    """
    Complete explainability configuration.

    Enables transparent decision-making for regulatory compliance
    and operator trust.

    Attributes:
        enabled: Master enable for explainability
        primary_method: Primary explanation method
        shap: SHAP configuration
        lime: LIME configuration
        explain_dispatch_decisions: Generate explanations for dispatch

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
        description="Primary explanation method"
    )

    shap: SHAPConfig = Field(
        default_factory=SHAPConfig,
        description="SHAP configuration"
    )
    lime: LIMEConfig = Field(
        default_factory=LIMEConfig,
        description="LIME configuration"
    )

    # Decision explanations
    explain_dispatch_decisions: bool = Field(
        default=True,
        description="Explain dispatch optimization decisions"
    )
    explain_unit_commitment: bool = Field(
        default=True,
        description="Explain unit commitment decisions"
    )
    explain_constraint_binding: bool = Field(
        default=True,
        description="Explain binding constraints"
    )

    # Sensitivity analysis
    sensitivity_analysis_enabled: bool = Field(
        default=True,
        description="Enable sensitivity analysis"
    )
    dual_variable_analysis: bool = Field(
        default=True,
        description="Report constraint dual variables (shadow prices)"
    )

    # Reporting
    explanation_format: str = Field(
        default="json",
        description="Explanation output format (json, html, text)"
    )
    include_visualizations: bool = Field(
        default=True,
        description="Include visualizations in explanations"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# PROVENANCE CONFIGURATION
# =============================================================================


class ProvenanceConfig(BaseModel):
    """
    Provenance tracking configuration for audit trail compliance.

    Implements SHA-256 hashing for complete data lineage and
    reproducibility per regulatory requirements.

    Attributes:
        enabled: Enable provenance tracking
        hash_algorithm: Hashing algorithm
        track_inputs: Track input data hashes
        track_outputs: Track output data hashes
    """

    enabled: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )

    # Hashing
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
    hash_configurations: bool = Field(
        default=True,
        description="Hash configuration at runtime"
    )

    # Timestamps
    include_timestamps: bool = Field(
        default=True,
        description="Include timestamps in provenance"
    )
    timestamp_format: str = Field(
        default="iso8601",
        description="Timestamp format"
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
        description="Provenance retention (days)"
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


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================


class GL023Config(BaseModel):
    """
    Master configuration for GL-023 HeatLoadBalancer Agent.

    This configuration combines all component configurations for
    optimal heat load distribution across multiple boilers and furnaces.

    Attributes:
        equipment_fleet: Boiler and furnace fleet configuration
        efficiency_curves: Default efficiency curve settings
        fuel: Multi-fuel configuration
        optimization: MILP and dispatch optimization
        demand: Demand profile and response configuration
        safety: Safety limits and interlocks
        integration: OPC-UA, Kafka, SCADA connectivity
        explainability: SHAP/LIME decision explanations
        provenance: Audit trail tracking

    Standards Compliance:
        - ASME CSD-1: Controls and Safety Devices
        - NFPA 85: Boiler and Combustion Systems
        - API 560: Fired Heaters
        - ISA 84: Safety Instrumented Systems

    Example:
        >>> config = GL023Config(
        ...     equipment_fleet=EquipmentFleetConfig(
        ...         boilers=[
        ...             BoilerConfig(boiler_id="BLR-001", capacity_mmbtu_hr=100),
        ...         ],
        ...     ),
        ...     demand=DemandConfig(
        ...         profile=DemandProfileConfig(
        ...             base_load_mmbtu_hr=50,
        ...             peak_load_mmbtu_hr=200,
        ...         ),
        ...     ),
        ... )

    Author: GreenLang Process Heat Team
    Version: 1.0.0
    """

    # Component configurations
    equipment_fleet: EquipmentFleetConfig = Field(
        ...,
        description="Equipment fleet configuration"
    )
    fuel: MultiFuelConfig = Field(
        default_factory=MultiFuelConfig,
        description="Multi-fuel configuration"
    )
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimization configuration"
    )
    demand: DemandConfig = Field(
        ...,
        description="Demand configuration"
    )
    safety: SafetyConfig = Field(
        default_factory=SafetyConfig,
        description="Safety configuration"
    )
    integration: IntegrationConfig = Field(
        default_factory=IntegrationConfig,
        description="Integration configuration"
    )
    explainability: ExplainabilityConfig = Field(
        default_factory=ExplainabilityConfig,
        description="Explainability configuration"
    )
    provenance: ProvenanceConfig = Field(
        default_factory=ProvenanceConfig,
        description="Provenance configuration"
    )

    # Agent identification
    agent_id: str = Field(
        default="GL-023",
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="HeatLoadBalancer",
        description="Agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Configuration version"
    )

    # General settings
    calculation_precision: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Decimal precision for calculations"
    )
    data_collection_interval_s: int = Field(
        default=1,
        ge=1,
        le=60,
        description="Data collection interval (seconds)"
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

    # Data retention
    data_retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Operational data retention (days)"
    )

    @root_validator(skip_on_failure=True)
    def validate_fleet_vs_demand(cls, values):
        """Validate fleet capacity can meet demand."""
        fleet = values.get("equipment_fleet")
        demand = values.get("demand")

        if fleet and demand:
            total_capacity = sum(b.capacity_mmbtu_hr for b in fleet.boilers)
            total_capacity += sum(f.capacity_mmbtu_hr for f in fleet.furnaces)

            peak_demand = demand.profile.peak_load_mmbtu_hr

            if total_capacity < peak_demand:
                raise ValueError(
                    f"Total fleet capacity ({total_capacity:.1f} MMBtu/hr) "
                    f"is less than peak demand ({peak_demand:.1f} MMBtu/hr)"
                )

        return values

    class Config:
        use_enum_values = True
        validate_assignment = True


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_default_config(
    plant_name: str = "Default Plant",
) -> GL023Config:
    """
    Create a default GL-023 configuration for a typical 3-boiler + 2-furnace plant.

    This configuration represents a typical industrial facility with:
    - 3 natural gas fired watertube boilers (100 MMBtu/hr each)
    - 2 process furnaces (150 MMBtu/hr each)
    - Base load of 200 MMBtu/hr, peak of 400 MMBtu/hr
    - Natural gas as primary fuel with fuel oil backup

    Args:
        plant_name: Name for the plant configuration

    Returns:
        GL023Config with typical industrial settings

    Example:
        >>> config = create_default_config("Refinery Steam Plant")
        >>> print(f"Total boiler capacity: {sum(b.capacity_mmbtu_hr for b in config.equipment_fleet.boilers)} MMBtu/hr")
        Total boiler capacity: 300.0 MMBtu/hr
    """
    # Create default efficiency curve
    default_efficiency = EfficiencyCurveConfig(
        curve_type="polynomial",
        polynomial=PolynomialEfficiencyConfig(
            a0=0.70,
            a1=0.25,
            a2=-0.15,
            a3=0.02,
        ),
    )

    # Create boilers
    boilers = [
        BoilerConfig(
            boiler_id="BLR-001",
            boiler_tag="B-101",
            boiler_name="Boiler 1",
            boiler_type=BoilerType.WATERTUBE,
            capacity_mmbtu_hr=100.0,
            steam_capacity_lb_hr=100000.0,
            design_pressure_psig=150.0,
            min_load_pct=25.0,
            max_load_pct=100.0,
            turndown_ratio=4.0,
            fuel_types=[FuelType.NATURAL_GAS, FuelType.NO2_FUEL_OIL],
            primary_fuel=FuelType.NATURAL_GAS,
            efficiency_curve=default_efficiency,
            design_efficiency_pct=82.0,
            priority=1,
            base_load=True,
        ),
        BoilerConfig(
            boiler_id="BLR-002",
            boiler_tag="B-102",
            boiler_name="Boiler 2",
            boiler_type=BoilerType.WATERTUBE,
            capacity_mmbtu_hr=100.0,
            steam_capacity_lb_hr=100000.0,
            design_pressure_psig=150.0,
            min_load_pct=25.0,
            max_load_pct=100.0,
            turndown_ratio=4.0,
            fuel_types=[FuelType.NATURAL_GAS, FuelType.NO2_FUEL_OIL],
            primary_fuel=FuelType.NATURAL_GAS,
            efficiency_curve=default_efficiency,
            design_efficiency_pct=82.0,
            priority=2,
        ),
        BoilerConfig(
            boiler_id="BLR-003",
            boiler_tag="B-103",
            boiler_name="Boiler 3",
            boiler_type=BoilerType.WATERTUBE,
            capacity_mmbtu_hr=100.0,
            steam_capacity_lb_hr=100000.0,
            design_pressure_psig=150.0,
            min_load_pct=25.0,
            max_load_pct=100.0,
            turndown_ratio=4.0,
            fuel_types=[FuelType.NATURAL_GAS],
            primary_fuel=FuelType.NATURAL_GAS,
            efficiency_curve=default_efficiency,
            design_efficiency_pct=82.0,
            priority=3,
        ),
    ]

    # Create furnaces
    furnaces = [
        FurnaceConfig(
            furnace_id="FUR-001",
            furnace_tag="H-101",
            furnace_name="Process Heater 1",
            furnace_type=FurnaceType.CABIN,
            process_type="crude_preheat",
            capacity_mmbtu_hr=150.0,
            design_duty_mmbtu_hr=127.5,
            min_firing_rate_pct=20.0,
            max_firing_rate_pct=100.0,
            fuel_types=[FuelType.REFINERY_FUEL_GAS, FuelType.NATURAL_GAS],
            primary_fuel=FuelType.REFINERY_FUEL_GAS,
            design_efficiency_pct=85.0,
            priority=1,
        ),
        FurnaceConfig(
            furnace_id="FUR-002",
            furnace_tag="H-102",
            furnace_name="Process Heater 2",
            furnace_type=FurnaceType.CABIN,
            process_type="general_heating",
            capacity_mmbtu_hr=150.0,
            design_duty_mmbtu_hr=127.5,
            min_firing_rate_pct=20.0,
            max_firing_rate_pct=100.0,
            fuel_types=[FuelType.REFINERY_FUEL_GAS, FuelType.NATURAL_GAS],
            primary_fuel=FuelType.REFINERY_FUEL_GAS,
            design_efficiency_pct=85.0,
            priority=2,
        ),
    ]

    # Create fuel configurations
    fuels = [
        FuelConfig(
            fuel_type=FuelType.NATURAL_GAS,
            fuel_name="Natural Gas",
            price_per_mmbtu=4.50,
            carbon_intensity_kg_co2_mmbtu=53.07,
            availability="available",
            hhv_btu_scf=1020.0,
        ),
        FuelConfig(
            fuel_type=FuelType.NO2_FUEL_OIL,
            fuel_name="No. 2 Fuel Oil",
            price_per_mmbtu=15.00,
            carbon_intensity_kg_co2_mmbtu=73.15,
            availability="available",
            hhv_btu_lb=19500.0,
        ),
        FuelConfig(
            fuel_type=FuelType.REFINERY_FUEL_GAS,
            fuel_name="Refinery Fuel Gas",
            price_per_mmbtu=2.50,
            carbon_intensity_kg_co2_mmbtu=60.0,
            availability="available",
            hhv_btu_scf=1100.0,
        ),
    ]

    return GL023Config(
        agent_name=f"HeatLoadBalancer - {plant_name}",
        equipment_fleet=EquipmentFleetConfig(
            boilers=boilers,
            furnaces=furnaces,
        ),
        fuel=MultiFuelConfig(
            fuels=fuels,
            carbon_price_usd_ton=50.0,
        ),
        optimization=OptimizationConfig(
            enabled=True,
            milp=MILPConfig(
                solver_type=SolverType.CBC,
                time_limit_s=60,
                gap_tolerance=0.01,
            ),
            dispatch=EconomicDispatchConfig(
                objective=OptimizationObjective.MINIMIZE_COST,
                cost_weight=0.6,
                emissions_weight=0.3,
                efficiency_weight=0.1,
            ),
        ),
        demand=DemandConfig(
            profile=DemandProfileConfig(
                base_load_mmbtu_hr=200.0,
                peak_load_mmbtu_hr=400.0,
            ),
        ),
        safety=SafetyConfig(
            sil_rating=SafetyIntegrityLevel.SIL_2,
            limits=SafetyLimitsConfig(
                n_plus_1_redundancy=True,
                emergency_reserve_pct=15.0,
            ),
        ),
    )


def create_large_plant_config(
    num_boilers: int = 10,
    num_furnaces: int = 5,
    plant_name: str = "Large Plant",
) -> GL023Config:
    """
    Create a configuration for a large plant with 10+ equipment units.

    This configuration represents a large industrial complex with:
    - Configurable number of boilers and furnaces
    - Mixed equipment sizes and types
    - Commercial-grade MILP solver recommendation
    - Enhanced N+2 redundancy

    Args:
        num_boilers: Number of boilers (default 10)
        num_furnaces: Number of furnaces (default 5)
        plant_name: Name for the plant configuration

    Returns:
        GL023Config for large-scale plant

    Example:
        >>> config = create_large_plant_config(num_boilers=12, num_furnaces=6)
        >>> print(f"Fleet size: {len(config.equipment_fleet.boilers)} boilers")
        Fleet size: 12 boilers
    """
    # Create default efficiency curve for boilers
    boiler_efficiency = EfficiencyCurveConfig(
        curve_type="polynomial",
        polynomial=PolynomialEfficiencyConfig(
            a0=0.70,
            a1=0.25,
            a2=-0.15,
            a3=0.02,
        ),
    )

    # Create furnace efficiency curve
    furnace_efficiency = EfficiencyCurveConfig(
        curve_type="polynomial",
        polynomial=PolynomialEfficiencyConfig(
            a0=0.75,
            a1=0.20,
            a2=-0.10,
            a3=0.01,
        ),
    )

    # Generate boilers with varying capacities
    boilers = []
    boiler_capacities = [50, 75, 100, 100, 125, 150, 150, 200, 200, 250]
    for i in range(num_boilers):
        capacity = boiler_capacities[i % len(boiler_capacities)]
        boilers.append(
            BoilerConfig(
                boiler_id=f"BLR-{i+1:03d}",
                boiler_tag=f"B-{100+i+1}",
                boiler_name=f"Boiler {i+1}",
                boiler_type=BoilerType.WATERTUBE if capacity >= 100 else BoilerType.FIRETUBE,
                capacity_mmbtu_hr=float(capacity),
                steam_capacity_lb_hr=float(capacity * 1000),
                design_pressure_psig=150.0 if capacity < 150 else 600.0,
                min_load_pct=25.0,
                max_load_pct=100.0,
                turndown_ratio=4.0 if capacity < 150 else 5.0,
                fuel_types=[FuelType.NATURAL_GAS, FuelType.NO2_FUEL_OIL],
                primary_fuel=FuelType.NATURAL_GAS,
                efficiency_curve=boiler_efficiency,
                design_efficiency_pct=82.0 + (capacity / 100),  # Larger = slightly more efficient
                priority=i + 1,
                base_load=(i < 2),  # First 2 are base load
            )
        )

    # Generate furnaces
    furnaces = []
    furnace_capacities = [100, 150, 150, 200, 250]
    for i in range(num_furnaces):
        capacity = furnace_capacities[i % len(furnace_capacities)]
        furnaces.append(
            FurnaceConfig(
                furnace_id=f"FUR-{i+1:03d}",
                furnace_tag=f"H-{100+i+1}",
                furnace_name=f"Furnace {i+1}",
                furnace_type=FurnaceType.CABIN if i % 2 == 0 else FurnaceType.CYLINDRICAL_VERTICAL,
                process_type="process_heating",
                capacity_mmbtu_hr=float(capacity),
                min_firing_rate_pct=20.0,
                max_firing_rate_pct=100.0,
                fuel_types=[FuelType.REFINERY_FUEL_GAS, FuelType.NATURAL_GAS],
                primary_fuel=FuelType.REFINERY_FUEL_GAS,
                efficiency_curve=furnace_efficiency,
                design_efficiency_pct=85.0,
                priority=i + 1,
            )
        )

    # Calculate total capacity
    total_boiler_capacity = sum(b.capacity_mmbtu_hr for b in boilers)
    total_furnace_capacity = sum(f.capacity_mmbtu_hr for f in furnaces)
    total_capacity = total_boiler_capacity + total_furnace_capacity

    # Create fuel configurations
    fuels = [
        FuelConfig(
            fuel_type=FuelType.NATURAL_GAS,
            fuel_name="Natural Gas",
            price_per_mmbtu=4.50,
            carbon_intensity_kg_co2_mmbtu=53.07,
            availability="available",
        ),
        FuelConfig(
            fuel_type=FuelType.NO2_FUEL_OIL,
            fuel_name="No. 2 Fuel Oil",
            price_per_mmbtu=15.00,
            carbon_intensity_kg_co2_mmbtu=73.15,
            availability="available",
        ),
        FuelConfig(
            fuel_type=FuelType.REFINERY_FUEL_GAS,
            fuel_name="Refinery Fuel Gas",
            price_per_mmbtu=2.50,
            carbon_intensity_kg_co2_mmbtu=60.0,
            availability="available",
        ),
        FuelConfig(
            fuel_type=FuelType.HYDROGEN,
            fuel_name="Hydrogen",
            price_per_mmbtu=20.00,
            carbon_intensity_kg_co2_mmbtu=0.0,
            availability="limited",
        ),
    ]

    return GL023Config(
        agent_name=f"HeatLoadBalancer - {plant_name}",
        equipment_fleet=EquipmentFleetConfig(
            boilers=boilers,
            furnaces=furnaces,
            max_simultaneous_starts=3,
        ),
        fuel=MultiFuelConfig(
            fuels=fuels,
            carbon_price_usd_ton=75.0,  # Higher carbon price for large facilities
        ),
        optimization=OptimizationConfig(
            enabled=True,
            milp=MILPConfig(
                solver_type=SolverType.CBC,  # Recommend CPLEX/Gurobi for production
                time_limit_s=120,  # Longer solve time for larger problem
                gap_tolerance=0.005,  # Tighter gap for better solutions
                threads=8,
            ),
            dispatch=EconomicDispatchConfig(
                objective=OptimizationObjective.PARETO_COST_EMISSIONS,
                cost_weight=0.5,
                emissions_weight=0.4,
                efficiency_weight=0.1,
                horizon_hours=48,  # Longer horizon for large plants
            ),
        ),
        demand=DemandConfig(
            profile=DemandProfileConfig(
                base_load_mmbtu_hr=total_capacity * 0.4,  # 40% base load
                peak_load_mmbtu_hr=total_capacity * 0.85,  # 85% peak
                load_forecast_horizon_hr=48,
            ),
            demand_response=DemandResponseConfig(
                enabled=True,
                max_curtailment_pct=15.0,
            ),
        ),
        safety=SafetyConfig(
            sil_rating=SafetyIntegrityLevel.SIL_2,
            limits=SafetyLimitsConfig(
                n_plus_1_redundancy=True,
                n_plus_redundancy_count=2,  # N+2 for large plant
                emergency_reserve_pct=10.0,
            ),
        ),
    )
