"""
GL-009 THERMALIQ - Thermal Fluid Systems Agent

This module provides comprehensive analysis and optimization for thermal fluid
(hot oil) systems including exergy-based efficiency analysis, fluid degradation
monitoring, expansion tank sizing, heat transfer calculations, and high-temperature
safety monitoring.

Key Features:
    - Thermal fluid property database (20+ fluids: Therminol, Dowtherm, Marlotherm)
    - Exergy (2nd Law) efficiency analysis (vs 1st Law for steam systems)
    - Fluid degradation monitoring (viscosity, flash point, TAN, low/high boilers)
    - Expansion tank sizing validation per API 660
    - Heat transfer coefficient calculations (Dittus-Boelter, Gnielinski, etc.)
    - Film temperature monitoring and limits
    - Flash point and auto-ignition safety margins
    - SIL-2 safety interlock recommendations
    - Zero-hallucination deterministic calculations
    - SHA-256 provenance tracking for audit trails

Supported Thermal Fluids:
    - Therminol 55, 59, 62, 66, VP-1, VP-3, XP
    - Dowtherm A, G, J, Q, RP
    - Marlotherm SH, LH
    - Mobiltherm 603, 605
    - Paratherm NF, HE
    - Syltherm 800, XLT

Example:
    >>> from greenlang.agents.process_heat.gl_009_thermal_fluid import (
    ...     ThermalFluidAnalyzer,
    ...     ThermalFluidConfig,
    ...     ThermalFluidInput,
    ...     ThermalFluidType,
    ...     create_default_config,
    ... )
    >>>
    >>> # Create configuration
    >>> config = create_default_config(
    ...     system_id="TF-001",
    ...     fluid_type=ThermalFluidType.THERMINOL_66,
    ...     design_temperature_f=600.0,
    ... )
    >>>
    >>> # Initialize analyzer
    >>> analyzer = ThermalFluidAnalyzer(config)
    >>>
    >>> # Create input data
    >>> input_data = ThermalFluidInput(
    ...     system_id="TF-001",
    ...     fluid_type=ThermalFluidType.THERMINOL_66,
    ...     bulk_temperature_f=580.0,
    ...     flow_rate_gpm=450.0,
    ... )
    >>>
    >>> # Process analysis
    >>> result = analyzer.process(input_data)
    >>>
    >>> # Access results
    >>> print(f"Safety status: {result.safety_analysis.safety_status}")
    >>> print(f"Exergy efficiency: {result.exergy_analysis.exergy_efficiency_pct}%")
    >>> print(f"Film temp margin: {result.safety_analysis.film_temp_margin_f}F")

Module Structure:
    - analyzer.py: Main ThermalFluidAnalyzer class
    - schemas.py: Pydantic data models (input/output)
    - config.py: Configuration schemas
    - fluid_properties.py: Thermal fluid property database
    - exergy.py: Exergy (2nd Law) efficiency calculations
    - degradation.py: Fluid degradation monitoring
    - expansion_tank.py: Expansion tank sizing per API 660
    - heat_transfer.py: Heat transfer coefficient calculations
    - safety.py: High temperature safety interlocks

Score Target: 95+/100
"""

# Version
__version__ = "1.0.0"
__agent_type__ = "GL-009"
__agent_name__ = "THERMALIQ"

# Main analyzer
from .analyzer import ThermalFluidAnalyzer

# Configuration
from .config import (
    ThermalFluidConfig,
    SafetyConfig,
    TemperatureLimits,
    FlowLimits,
    PressureLimits,
    DegradationThresholds,
    HeaterConfig,
    ExpansionTankConfig,
    PumpConfig,
    PipingConfig,
    ExergyConfig,
    create_default_config,
    create_high_temperature_config,
    create_food_grade_config,
)

# Schemas
from .schemas import (
    # Enums
    ThermalFluidType,
    DegradationLevel,
    SafetyStatus,
    HeaterType,
    FlowRegime,
    ValidationStatus,
    OptimizationStatus,
    # Input models
    ThermalFluidInput,
    FluidLabAnalysis,
    ExpansionTankData,
    # Output models
    FluidProperties,
    ExergyAnalysis,
    DegradationAnalysis,
    HeatTransferAnalysis,
    ExpansionTankSizing,
    SafetyAnalysis,
    OptimizationRecommendation,
    ThermalFluidOutput,
)

# Sub-modules
from .fluid_properties import (
    ThermalFluidPropertyDatabase,
    get_fluid_properties,
    compare_fluids,
)

from .exergy import (
    ExergyAnalyzer,
    ExergyDestructionBreakdown,
    calculate_exergy_efficiency,
    calculate_carnot_limit,
)

from .degradation import (
    DegradationMonitor,
    FluidDegradationLimits,
)

from .expansion_tank import (
    ExpansionTankAnalyzer,
    calculate_expansion,
    size_tank,
)

from .heat_transfer import (
    HeatTransferCalculator,
    HeatTransferCorrelation,
    calculate_reynolds,
    get_minimum_velocity,
)

from .safety import (
    SafetyMonitor,
    check_temperature_safety,
)


# Public API
__all__ = [
    # Version info
    "__version__",
    "__agent_type__",
    "__agent_name__",
    # Main analyzer
    "ThermalFluidAnalyzer",
    # Configuration
    "ThermalFluidConfig",
    "SafetyConfig",
    "TemperatureLimits",
    "FlowLimits",
    "PressureLimits",
    "DegradationThresholds",
    "HeaterConfig",
    "ExpansionTankConfig",
    "PumpConfig",
    "PipingConfig",
    "ExergyConfig",
    "create_default_config",
    "create_high_temperature_config",
    "create_food_grade_config",
    # Enums
    "ThermalFluidType",
    "DegradationLevel",
    "SafetyStatus",
    "HeaterType",
    "FlowRegime",
    "ValidationStatus",
    "OptimizationStatus",
    "HeatTransferCorrelation",
    # Input models
    "ThermalFluidInput",
    "FluidLabAnalysis",
    "ExpansionTankData",
    # Output models
    "FluidProperties",
    "ExergyAnalysis",
    "DegradationAnalysis",
    "HeatTransferAnalysis",
    "ExpansionTankSizing",
    "SafetyAnalysis",
    "OptimizationRecommendation",
    "ThermalFluidOutput",
    # Sub-analyzers
    "ThermalFluidPropertyDatabase",
    "ExergyAnalyzer",
    "ExergyDestructionBreakdown",
    "DegradationMonitor",
    "FluidDegradationLimits",
    "ExpansionTankAnalyzer",
    "HeatTransferCalculator",
    "SafetyMonitor",
    # Convenience functions
    "get_fluid_properties",
    "compare_fluids",
    "calculate_exergy_efficiency",
    "calculate_carnot_limit",
    "calculate_expansion",
    "size_tank",
    "calculate_reynolds",
    "get_minimum_velocity",
    "check_temperature_safety",
]
