"""
GreenLang Thermodynamic Library

Zero-Hallucination Engineering Calculations

This module provides deterministic, standards-compliant thermodynamic
calculations for industrial and process engineering applications.

Modules:
    - steam_tables: IAPWS-IF97 steam property calculations
    - combustion: Combustion stoichiometry per EPA Method 19
    - psychrometrics: Air-water vapor properties (ASHRAE)
    - heat_transfer: NTU, LMTD, effectiveness methods
    - pinch_analysis: Automated heat integration
    - exergy: Second law thermodynamic analysis

All calculations provide:
    - Deterministic outputs (same input = same output)
    - Complete provenance tracking (SHA-256 hashes)
    - Standards compliance (ASME, EPA, ASHRAE)
    - Uncertainty quantification where applicable

Author: GreenLang Engineering Team
License: MIT
"""

# Steam Tables (IAPWS-IF97)
from .steam_tables import (
    IAPWSIF97,
    SteamProperties,
    Region,
    steam_pt,
    steam_px,
    saturation_p,
    saturation_t,
)

# Combustion Calculations (EPA Method 19, ASME PTC 4.1)
from .combustion import (
    CombustionCalculator,
    CombustionResult,
    FuelComposition,
    GasFuelComposition,
    combustion_coal,
    combustion_natural_gas,
)

# Psychrometrics (ASHRAE Fundamentals)
from .psychrometrics import (
    PsychrometricCalculator,
    PsychrometricProperties,
    psychrometric_properties,
    saturation_vapor_pressure,
    dew_point,
    wet_bulb,
)

# Heat Transfer
from .heat_transfer import (
    HeatTransferCalculator,
    HeatTransferResult,
    OverallHeatTransferResult,
    HeatExchangerType,
    heat_exchanger_analysis,
    calculate_lmtd,
    calculate_effectiveness,
)

# Pinch Analysis
from .pinch_analysis import (
    PinchAnalyzer,
    PinchResult,
    StreamData,
    AreaTarget,
    pinch_analysis,
    minimum_utilities,
)

# Exergy Analysis
from .exergy import (
    ExergyCalculator,
    ExergyResult,
    ExergyState,
    ExergyDestructionResult,
    ComponentExergyAnalysis,
    STANDARD_CHEMICAL_EXERGIES,
    calculate_physical_exergy,
    exergetic_efficiency,
    heat_to_exergy,
)

# Uncertainty Propagation (GUM ISO/IEC Guide 98-3:2008)
from .uncertainty import (
    GUMUncertaintyCalculator,
    UncertaintyResult,
    UncertaintyContribution,
    MeasuredValue,
    SensitivityCoefficient,
    DistributionType,
    UncertaintyType,
    propagate_uncertainty,
    combine_uncorrelated,
)

__version__ = "1.0.0"

__all__ = [
    # Steam Tables
    "IAPWSIF97",
    "SteamProperties",
    "Region",
    "steam_pt",
    "steam_px",
    "saturation_p",
    "saturation_t",
    # Combustion
    "CombustionCalculator",
    "CombustionResult",
    "FuelComposition",
    "GasFuelComposition",
    "combustion_coal",
    "combustion_natural_gas",
    # Psychrometrics
    "PsychrometricCalculator",
    "PsychrometricProperties",
    "psychrometric_properties",
    "saturation_vapor_pressure",
    "dew_point",
    "wet_bulb",
    # Heat Transfer
    "HeatTransferCalculator",
    "HeatTransferResult",
    "OverallHeatTransferResult",
    "HeatExchangerType",
    "heat_exchanger_analysis",
    "calculate_lmtd",
    "calculate_effectiveness",
    # Pinch Analysis
    "PinchAnalyzer",
    "PinchResult",
    "StreamData",
    "AreaTarget",
    "pinch_analysis",
    "minimum_utilities",
    # Exergy
    "ExergyCalculator",
    "ExergyResult",
    "ExergyState",
    "ExergyDestructionResult",
    "ComponentExergyAnalysis",
    "STANDARD_CHEMICAL_EXERGIES",
    "calculate_physical_exergy",
    "exergetic_efficiency",
    "heat_to_exergy",
    # Uncertainty Propagation
    "GUMUncertaintyCalculator",
    "UncertaintyResult",
    "UncertaintyContribution",
    "MeasuredValue",
    "SensitivityCoefficient",
    "DistributionType",
    "UncertaintyType",
    "propagate_uncertainty",
    "combine_uncorrelated",
]
