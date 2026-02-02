"""
GreenLang Formula Calculation Engine
=====================================

Zero-hallucination calculation engine with full provenance tracking.
All calculations are deterministic, reproducible, and auditable.

This module provides:
- CalculationEngine: Main engine for executing formulas
- FormulaRegistry: Registry of all available formulas
- ValidationEngine: Input validation and constraint checking
- ProvenanceTracker: SHA-256 based audit trails

Guarantees:
- Deterministic: Same input always produces same output
- Reproducible: Full calculation provenance tracking
- Auditable: SHA-256 hash for every calculation
- Zero-Hallucination: No LLM in calculation path

Copyright (c) 2024 GreenLang. All rights reserved.
"""

from .calculation_engine import (
    CalculationEngine,
    CalculationResult,
    FormulaInput,
    FormulaDefinition,
    CalculationStep,
    ProvenanceTracker,
    FormulaRegistry,
    UnitConverter,
    UncertaintyPropagator,
    CalculationCache,
)

from .validation import (
    ValidationEngine,
    ValidationResult,
    ValidationError,
    RangeValidator,
    UnitValidator,
    PhysicalFeasibilityValidator,
    ConstraintValidator,
)

from .thermodynamic_formulas import (
    ThermodynamicFormulas,
    SteamProperties,
    IdealGasCalculations,
    PsychrometricCalculations,
    ExergyAnalysis,
    HeatExchangerAnalysis,
)

from .combustion_formulas import (
    CombustionFormulas,
    StoichiometricCalculations,
    ExcessAirCalculations,
    FlameTemperatureCalculations,
    HeatReleaseCalculations,
    EmissionFactorCalculations,
)

from .efficiency_formulas import (
    EfficiencyFormulas,
    BoilerEfficiency,
    FurnaceEfficiency,
    HeatExchangerEfficiency,
    TurbineEfficiency,
    SystemEfficiency,
)

from .safety_formulas import (
    SafetyFormulas,
    PSVSizing,
    ReliefLoadCalculations,
    FireCaseCalculations,
    PurgeTimeCalculations,
    SIFProbabilityCalculations,
)

from .heat_transfer_formulas import (
    HeatTransferFormulas,
    ConductionCalculations,
    ConvectionCalculations,
    RadiationCalculations,
    HeatExchangerDesign,
    FinEfficiencyCalculations,
)

__version__ = "1.0.0"
__author__ = "GreenLang Engineering Team"

__all__ = [
    # Core Engine
    "CalculationEngine",
    "CalculationResult",
    "FormulaInput",
    "FormulaDefinition",
    "CalculationStep",
    "ProvenanceTracker",
    "FormulaRegistry",
    "UnitConverter",
    "UncertaintyPropagator",
    "CalculationCache",
    # Validation
    "ValidationEngine",
    "ValidationResult",
    "ValidationError",
    "RangeValidator",
    "UnitValidator",
    "PhysicalFeasibilityValidator",
    "ConstraintValidator",
    # Thermodynamic Formulas
    "ThermodynamicFormulas",
    "SteamProperties",
    "IdealGasCalculations",
    "PsychrometricCalculations",
    "ExergyAnalysis",
    "HeatExchangerAnalysis",
    # Combustion Formulas
    "CombustionFormulas",
    "StoichiometricCalculations",
    "ExcessAirCalculations",
    "FlameTemperatureCalculations",
    "HeatReleaseCalculations",
    "EmissionFactorCalculations",
    # Efficiency Formulas
    "EfficiencyFormulas",
    "BoilerEfficiency",
    "FurnaceEfficiency",
    "HeatExchangerEfficiency",
    "TurbineEfficiency",
    "SystemEfficiency",
    # Safety Formulas
    "SafetyFormulas",
    "PSVSizing",
    "ReliefLoadCalculations",
    "FireCaseCalculations",
    "PurgeTimeCalculations",
    "SIFProbabilityCalculations",
    # Heat Transfer Formulas
    "HeatTransferFormulas",
    "ConductionCalculations",
    "ConvectionCalculations",
    "RadiationCalculations",
    "HeatExchangerDesign",
    "FinEfficiencyCalculations",
]
