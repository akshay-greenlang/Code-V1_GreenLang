"""GL-009 THERMALIQ Zero-Hallucination Thermal Efficiency Calculators.

This module provides world-class deterministic calculation engines for
thermal efficiency analysis. All calculators guarantee:

- 100% bit-perfect reproducibility
- Complete provenance tracking with SHA-256 hashing
- Zero LLM involvement in calculation paths
- Full compliance with ASME/ISO standards
- Comprehensive audit trails

Standards Implemented:
    - ASME PTC 4.1: Steam Generating Units
    - ASME PTC 46: Overall Plant Performance
    - ISO 50001: Energy Management Systems
    - IAPWS-IF97: Steam Properties

Modules:
    - first_law_efficiency: Energy balance efficiency (First Law)
    - second_law_efficiency: Exergy/availability efficiency (Second Law)
    - heat_loss_calculator: Comprehensive heat loss analysis
    - fuel_energy_calculator: Fuel energy content calculations
    - steam_energy_calculator: Steam property calculations (IAPWS-IF97)
    - sankey_generator: Energy flow visualization data
    - benchmark_calculator: Industry benchmark comparisons
    - improvement_analyzer: Efficiency improvement opportunities
    - uncertainty_calculator: Measurement uncertainty quantification
    - provenance: Audit trail and hashing utilities

Example:
    >>> from calculators import FirstLawEfficiencyCalculator, FirstLawResult
    >>> calculator = FirstLawEfficiencyCalculator()
    >>> result = calculator.calculate(
    ...     energy_inputs={"natural_gas": 1000.0},
    ...     useful_outputs={"steam": 850.0},
    ...     losses={"flue_gas": 100.0, "radiation": 30.0, "other": 20.0}
    ... )
    >>> print(f"Efficiency: {result.efficiency_percent}%")
    Efficiency: 85.0%

Author: GL-009 THERMALIQ Agent
Version: 1.0.0
License: Proprietary - GreenLang
"""

from typing import List

# First Law Efficiency Calculator
from .first_law_efficiency import (
    FirstLawEfficiencyCalculator,
    FirstLawResult,
    EnergyInput,
    UsefulOutput,
    EnergyLoss,
    EnergyBalanceValidation,
)

# Second Law Efficiency Calculator
from .second_law_efficiency import (
    SecondLawEfficiencyCalculator,
    SecondLawResult,
    ExergyStream,
    ReferenceEnvironment,
    IrreversibilityBreakdown,
)

# Heat Loss Calculator
from .heat_loss_calculator import (
    HeatLossCalculator,
    HeatLossResult,
    RadiationLoss,
    ConvectionLoss,
    ConductionLoss,
    FlueGasLoss,
    UnburnedFuelLoss,
    SurfaceGeometry,
    FlueGasComposition,
)

# Fuel Energy Calculator
from .fuel_energy_calculator import (
    FuelEnergyCalculator,
    FuelEnergyResult,
    FuelComposition,
    CombustionResult,
    FuelType,
    HeatingValueType,
)

# Steam Energy Calculator
from .steam_energy_calculator import (
    SteamEnergyCalculator,
    SteamEnergyResult,
    SteamState,
    SteamQuality,
    FlashSteamResult,
    CondensateReturn,
)

# Sankey Diagram Generator
from .sankey_generator import (
    SankeyGenerator,
    SankeyDiagram,
    SankeyNode,
    SankeyLink,
    NodeType,
    EnergyFlowCategory,
)

# Benchmark Calculator
from .benchmark_calculator import (
    BenchmarkCalculator,
    BenchmarkResult,
    IndustryBenchmark,
    PercentileRanking,
    GapAnalysis,
    ProcessType,
)

# Improvement Analyzer
from .improvement_analyzer import (
    ImprovementAnalyzer,
    ImprovementAnalysis,
    ImprovementOpportunity,
    HeatRecoveryOption,
    ROICalculation,
    PriorityLevel,
)

# Uncertainty Calculator
from .uncertainty_calculator import (
    UncertaintyCalculator,
    UncertaintyResult,
    InstrumentAccuracy,
    MonteCarloResult,
    ConfidenceInterval,
    SensitivityFactor,
)

# Provenance Utilities
from .provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    CalculationStep,
    AuditTrail,
    generate_provenance_hash,
    create_audit_trail,
)

__version__ = "1.0.0"
__author__ = "GL-009 THERMALIQ Agent"
__all__: List[str] = [
    # First Law
    "FirstLawEfficiencyCalculator",
    "FirstLawResult",
    "EnergyInput",
    "UsefulOutput",
    "EnergyLoss",
    "EnergyBalanceValidation",
    # Second Law
    "SecondLawEfficiencyCalculator",
    "SecondLawResult",
    "ExergyStream",
    "ReferenceEnvironment",
    "IrreversibilityBreakdown",
    # Heat Loss
    "HeatLossCalculator",
    "HeatLossResult",
    "RadiationLoss",
    "ConvectionLoss",
    "ConductionLoss",
    "FlueGasLoss",
    "UnburnedFuelLoss",
    "SurfaceGeometry",
    "FlueGasComposition",
    # Fuel Energy
    "FuelEnergyCalculator",
    "FuelEnergyResult",
    "FuelComposition",
    "CombustionResult",
    "FuelType",
    "HeatingValueType",
    # Steam Energy
    "SteamEnergyCalculator",
    "SteamEnergyResult",
    "SteamState",
    "SteamQuality",
    "FlashSteamResult",
    "CondensateReturn",
    # Sankey
    "SankeyGenerator",
    "SankeyDiagram",
    "SankeyNode",
    "SankeyLink",
    "NodeType",
    "EnergyFlowCategory",
    # Benchmark
    "BenchmarkCalculator",
    "BenchmarkResult",
    "IndustryBenchmark",
    "PercentileRanking",
    "GapAnalysis",
    "ProcessType",
    # Improvement
    "ImprovementAnalyzer",
    "ImprovementAnalysis",
    "ImprovementOpportunity",
    "HeatRecoveryOption",
    "ROICalculation",
    "PriorityLevel",
    # Uncertainty
    "UncertaintyCalculator",
    "UncertaintyResult",
    "InstrumentAccuracy",
    "MonteCarloResult",
    "ConfidenceInterval",
    "SensitivityFactor",
    # Provenance
    "ProvenanceTracker",
    "ProvenanceRecord",
    "CalculationStep",
    "AuditTrail",
    "generate_provenance_hash",
    "create_audit_trail",
]
