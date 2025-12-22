"""
GL-009 THERMALIQ - Configuration Module

Configuration settings, enums, and constants for the Thermal Fluid Analyzer
agent. All configuration supports deterministic, reproducible operation.

This module defines:
    - ThermalIQConfig: Main configuration class
    - CalculationMode: Enum for analysis modes
    - FluidConfig: Fluid-specific configuration
    - SafetyConfig: Thermal safety constraints
    - ExplainabilityConfig: SHAP/LIME settings
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import os


# =============================================================================
# ENUMS
# =============================================================================

class CalculationMode(Enum):
    """
    Calculation mode for thermal analysis.

    Defines the scope and depth of thermal calculations:
    - EFFICIENCY: Basic thermal efficiency only
    - EXERGY: Full exergy (second-law) analysis
    - FULL_ANALYSIS: Complete analysis with Sankey, recommendations
    """
    EFFICIENCY = "efficiency"
    EXERGY = "exergy"
    FULL_ANALYSIS = "full_analysis"


class FluidPhase(Enum):
    """
    Fluid phase classification.

    Used for property lookup and calculation method selection.
    """
    LIQUID = "liquid"
    GAS = "gas"
    TWO_PHASE = "two_phase"
    SUPERCRITICAL = "supercritical"
    SUBCOOLED = "subcooled"
    SUPERHEATED = "superheated"


class FluidLibraryType(Enum):
    """
    Fluid property library source.

    Determines which library to use for thermophysical properties:
    - COOLPROP: CoolProp library (recommended for accuracy)
    - IAPWS: IAPWS-IF97 for steam/water
    - INTERNAL: Built-in lookup tables
    - CUSTOM: User-defined properties
    """
    COOLPROP = "coolprop"
    IAPWS = "iapws"
    INTERNAL = "internal"
    CUSTOM = "custom"


class ExplainabilityMethod(Enum):
    """
    Explainability method selection.

    Determines which XAI methods to use:
    - SHAP: SHapley Additive exPlanations (global + local)
    - LIME: Local Interpretable Model-agnostic Explanations
    - BOTH: Use both SHAP and LIME
    - NONE: No explainability (faster execution)
    """
    SHAP = "shap"
    LIME = "lime"
    BOTH = "both"
    NONE = "none"


class SankeyOutputFormat(Enum):
    """
    Sankey diagram output format.

    Determines the format of generated Sankey data:
    - PLOTLY: Compatible with Plotly.js
    - D3: Compatible with D3.js
    - ECHARTS: Compatible with Apache ECharts
    - JSON: Generic JSON structure
    """
    PLOTLY = "plotly"
    D3 = "d3"
    ECHARTS = "echarts"
    JSON = "json"


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Reference conditions for exergy calculations
REFERENCE_TEMPERATURE_K = 298.15  # 25 deg C (dead state temperature)
REFERENCE_PRESSURE_KPA = 101.325  # 1 atm

# Common fluid specific heats (kJ/kg-K) at standard conditions
WATER_SPECIFIC_HEAT_KJ_KGK = 4.186
STEAM_SPECIFIC_HEAT_KJ_KGK = 2.010
AIR_SPECIFIC_HEAT_KJ_KGK = 1.005
THERMAL_OIL_SPECIFIC_HEAT_KJ_KGK = 2.100
GLYCOL_50_SPECIFIC_HEAT_KJ_KGK = 3.350

# Stefan-Boltzmann constant for radiation calculations
STEFAN_BOLTZMANN_W_M2K4 = 5.67e-8


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class FluidConfig:
    """
    Configuration for a specific thermal fluid.

    Stores thermophysical properties and lookup parameters
    for a thermal fluid used in calculations.

    Attributes:
        fluid_id: Unique identifier for the fluid
        fluid_name: Human-readable name
        library: Property library to use
        Cp_kJ_kgK: Specific heat capacity (if constant)
        density_kg_m3: Density (if constant)
        viscosity_Pa_s: Dynamic viscosity (if constant)
        conductivity_W_mK: Thermal conductivity (if constant)
        phase: Expected phase at operating conditions
        min_temperature_C: Minimum valid temperature
        max_temperature_C: Maximum valid temperature
        min_pressure_kPa: Minimum valid pressure
        max_pressure_kPa: Maximum valid pressure
    """
    fluid_id: str
    fluid_name: str
    library: FluidLibraryType = FluidLibraryType.INTERNAL

    # Constant properties (used if library lookup not available)
    Cp_kJ_kgK: float = 4.186
    density_kg_m3: float = 1000.0
    viscosity_Pa_s: float = 0.001
    conductivity_W_mK: float = 0.6

    # Phase information
    phase: FluidPhase = FluidPhase.LIQUID

    # Valid operating range
    min_temperature_C: float = -50.0
    max_temperature_C: float = 400.0
    min_pressure_kPa: float = 1.0
    max_pressure_kPa: float = 20000.0

    # Safety limits
    flash_point_C: Optional[float] = None
    autoignition_C: Optional[float] = None
    max_film_temperature_C: Optional[float] = None

    def is_temperature_valid(self, temperature_C: float) -> bool:
        """Check if temperature is within valid range."""
        return self.min_temperature_C <= temperature_C <= self.max_temperature_C

    def is_pressure_valid(self, pressure_kPa: float) -> bool:
        """Check if pressure is within valid range."""
        return self.min_pressure_kPa <= pressure_kPa <= self.max_pressure_kPa


@dataclass
class SafetyConfig:
    """
    Thermal safety configuration and constraints.

    Defines safety limits for thermal operations to prevent
    equipment damage and unsafe conditions.

    Attributes:
        max_temperature_C: Maximum allowable temperature
        min_temperature_C: Minimum allowable temperature
        max_pressure_kPa: Maximum allowable pressure
        max_thermal_stress_rate_C_min: Maximum temperature change rate
        max_exergy_destruction_pct: Warning threshold for exergy loss
        enable_safety_interlocks: Enable automatic safety checks
        fail_safe_mode: Behavior on safety violation
    """
    # Temperature limits
    max_temperature_C: float = 500.0
    min_temperature_C: float = -40.0

    # Pressure limits
    max_pressure_kPa: float = 10000.0
    min_pressure_kPa: float = 10.0

    # Thermal stress limits
    max_thermal_stress_rate_C_min: float = 10.0

    # Exergy thresholds
    max_exergy_destruction_pct: float = 50.0
    min_efficiency_warning_pct: float = 30.0

    # Safety system settings
    enable_safety_interlocks: bool = True
    fail_safe_mode: str = "warn"  # warn, reject, or safe_state

    # Alarm thresholds
    alarm_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high_temperature_C": 450.0,
        "low_temperature_C": -30.0,
        "high_pressure_kPa": 8000.0,
        "low_efficiency_pct": 25.0,
    })


@dataclass
class ExplainabilityConfig:
    """
    Configuration for SHAP/LIME explainability features.

    Controls the explainability methods and their parameters
    for generating interpretable analysis results.

    Attributes:
        method: Explainability method to use
        shap_sample_size: Number of samples for SHAP
        lime_num_features: Number of top features for LIME
        generate_recommendations: Auto-generate recommendations
        max_recommendations: Maximum recommendations to generate
        confidence_threshold: Minimum confidence for recommendations
    """
    method: ExplainabilityMethod = ExplainabilityMethod.BOTH

    # SHAP settings
    shap_enabled: bool = True
    shap_sample_size: int = 100
    shap_background_samples: int = 50

    # LIME settings
    lime_enabled: bool = True
    lime_num_features: int = 10
    lime_num_samples: int = 1000

    # Recommendation settings
    generate_recommendations: bool = True
    max_recommendations: int = 5
    confidence_threshold: float = 0.8

    # Output settings
    include_feature_importance: bool = True
    include_local_explanations: bool = True
    include_counterfactuals: bool = False


@dataclass
class SankeyConfig:
    """
    Configuration for Sankey diagram generation.

    Controls the format and detail level of generated
    Sankey diagram data for energy flow visualization.

    Attributes:
        output_format: Format for Sankey data
        include_losses: Include loss nodes in diagram
        min_flow_threshold_kW: Minimum flow to include
        color_scheme: Color scheme for nodes/links
        node_padding: Padding between nodes
    """
    output_format: SankeyOutputFormat = SankeyOutputFormat.PLOTLY
    include_losses: bool = True
    include_exergy_flows: bool = True

    # Display thresholds
    min_flow_threshold_kW: float = 1.0
    min_flow_threshold_pct: float = 0.1

    # Styling
    color_scheme: str = "thermal"  # thermal, energy, custom
    node_padding: int = 15
    link_opacity: float = 0.5

    # Labels
    show_values: bool = True
    show_percentages: bool = True
    value_format: str = ".1f"  # Python format string


# =============================================================================
# FLUID LIBRARY
# =============================================================================

# Built-in fluid configurations
DEFAULT_FLUID_LIBRARY: Dict[str, FluidConfig] = {
    "water": FluidConfig(
        fluid_id="water",
        fluid_name="Water",
        library=FluidLibraryType.IAPWS,
        Cp_kJ_kgK=4.186,
        density_kg_m3=1000.0,
        viscosity_Pa_s=0.001,
        conductivity_W_mK=0.6,
        phase=FluidPhase.LIQUID,
        min_temperature_C=0.0,
        max_temperature_C=374.0,
    ),
    "steam": FluidConfig(
        fluid_id="steam",
        fluid_name="Steam",
        library=FluidLibraryType.IAPWS,
        Cp_kJ_kgK=2.010,
        density_kg_m3=0.6,
        viscosity_Pa_s=0.000012,
        conductivity_W_mK=0.025,
        phase=FluidPhase.GAS,
        min_temperature_C=100.0,
        max_temperature_C=600.0,
    ),
    "thermal_oil": FluidConfig(
        fluid_id="thermal_oil",
        fluid_name="Thermal Oil (Therminol 66)",
        library=FluidLibraryType.INTERNAL,
        Cp_kJ_kgK=2.100,
        density_kg_m3=850.0,
        viscosity_Pa_s=0.002,
        conductivity_W_mK=0.12,
        phase=FluidPhase.LIQUID,
        min_temperature_C=-40.0,
        max_temperature_C=350.0,
        flash_point_C=170.0,
        max_film_temperature_C=375.0,
    ),
    "air": FluidConfig(
        fluid_id="air",
        fluid_name="Air",
        library=FluidLibraryType.COOLPROP,
        Cp_kJ_kgK=1.005,
        density_kg_m3=1.225,
        viscosity_Pa_s=0.0000181,
        conductivity_W_mK=0.026,
        phase=FluidPhase.GAS,
        min_temperature_C=-100.0,
        max_temperature_C=1000.0,
    ),
    "glycol_50": FluidConfig(
        fluid_id="glycol_50",
        fluid_name="50% Ethylene Glycol",
        library=FluidLibraryType.INTERNAL,
        Cp_kJ_kgK=3.350,
        density_kg_m3=1070.0,
        viscosity_Pa_s=0.003,
        conductivity_W_mK=0.40,
        phase=FluidPhase.LIQUID,
        min_temperature_C=-35.0,
        max_temperature_C=120.0,
    ),
    "r134a": FluidConfig(
        fluid_id="r134a",
        fluid_name="R-134a Refrigerant",
        library=FluidLibraryType.COOLPROP,
        Cp_kJ_kgK=1.43,
        density_kg_m3=1206.0,
        viscosity_Pa_s=0.0002,
        conductivity_W_mK=0.082,
        phase=FluidPhase.LIQUID,
        min_temperature_C=-40.0,
        max_temperature_C=100.0,
    ),
    "ammonia": FluidConfig(
        fluid_id="ammonia",
        fluid_name="Ammonia (NH3)",
        library=FluidLibraryType.COOLPROP,
        Cp_kJ_kgK=4.70,
        density_kg_m3=602.0,
        viscosity_Pa_s=0.00013,
        conductivity_W_mK=0.49,
        phase=FluidPhase.LIQUID,
        min_temperature_C=-77.0,
        max_temperature_C=130.0,
    ),
    "molten_salt": FluidConfig(
        fluid_id="molten_salt",
        fluid_name="Solar Salt (60% NaNO3, 40% KNO3)",
        library=FluidLibraryType.INTERNAL,
        Cp_kJ_kgK=1.53,
        density_kg_m3=1800.0,
        viscosity_Pa_s=0.003,
        conductivity_W_mK=0.52,
        phase=FluidPhase.LIQUID,
        min_temperature_C=220.0,
        max_temperature_C=600.0,
    ),
}


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

@dataclass
class ThermalIQConfig:
    """
    Master configuration for GL-009 THERMALIQ agent.

    All settings required for deterministic, reproducible operation
    of the thermal fluid analyzer system.

    Attributes:
        agent_id: Agent identifier (GL-009)
        agent_name: Agent name (THERMALIQ)
        version: Configuration version
        mode: Default calculation mode
        reference_temperature_K: Dead state temperature for exergy
        reference_pressure_kPa: Reference pressure for exergy
        fluid_library: Available fluid configurations
        safety: Safety configuration
        explainability: Explainability configuration
        sankey: Sankey diagram configuration

    Example:
        >>> config = ThermalIQConfig()
        >>> config.mode = CalculationMode.FULL_ANALYSIS
        >>> orchestrator = ThermalIQOrchestrator(config)
    """
    # Agent identification
    agent_id: str = "GL-009"
    agent_name: str = "THERMALIQ"
    version: str = "1.0.0"

    # Calculation settings
    mode: CalculationMode = CalculationMode.FULL_ANALYSIS

    # Reference conditions for exergy (dead state)
    reference_temperature_K: float = REFERENCE_TEMPERATURE_K
    reference_pressure_kPa: float = REFERENCE_PRESSURE_KPA

    # Fluid library
    fluid_library: Dict[str, FluidConfig] = field(
        default_factory=lambda: DEFAULT_FLUID_LIBRARY.copy()
    )
    default_fluid: str = "water"

    # Sub-configurations
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    sankey: SankeyConfig = field(default_factory=SankeyConfig)

    # Calculation precision
    calculation_precision: int = 6  # Decimal places
    tolerance: float = 1e-6
    max_iterations: int = 100

    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    batch_size: int = 1000

    # Audit and provenance
    enable_audit_logging: bool = True
    enable_provenance_tracking: bool = True
    deterministic_mode: bool = True

    # Integration endpoints
    kafka_brokers: str = field(
        default_factory=lambda: os.getenv("KAFKA_BROKERS", "localhost:9092")
    )
    graphql_port: int = 8080
    metrics_port: int = 9090

    def get_fluid_config(self, fluid_id: str) -> Optional[FluidConfig]:
        """
        Get configuration for a specific fluid.

        Args:
            fluid_id: Fluid identifier

        Returns:
            FluidConfig or None if not found
        """
        return self.fluid_library.get(fluid_id.lower())

    def add_fluid(self, config: FluidConfig) -> None:
        """
        Add a custom fluid to the library.

        Args:
            config: Fluid configuration to add
        """
        self.fluid_library[config.fluid_id.lower()] = config

    def validate(self) -> List[str]:
        """
        Validate configuration settings.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if self.reference_temperature_K <= 0:
            errors.append("Reference temperature must be positive")

        if self.reference_pressure_kPa <= 0:
            errors.append("Reference pressure must be positive")

        if self.calculation_precision < 1 or self.calculation_precision > 15:
            errors.append("Calculation precision must be between 1 and 15")

        if not self.fluid_library:
            errors.append("Fluid library cannot be empty")

        if self.default_fluid not in self.fluid_library:
            errors.append(f"Default fluid '{self.default_fluid}' not in library")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "version": self.version,
            "mode": self.mode.value,
            "reference_temperature_K": self.reference_temperature_K,
            "reference_pressure_kPa": self.reference_pressure_kPa,
            "deterministic_mode": self.deterministic_mode,
            "available_fluids": list(self.fluid_library.keys()),
        }


# Default configuration instance
DEFAULT_CONFIG = ThermalIQConfig()
