# -*- coding: utf-8 -*-
"""
GL-004 Calculator Modules

Physics-based, zero-hallucination calculator modules for burner optimization.
All calculations are deterministic and based on established engineering principles.

Module Categories:
- Core Calculators: Stoichiometric, combustion efficiency, flame analysis
- Optimization: Air-fuel ratio, burner tuning, performance
- Emissions: Emissions calculation, prediction, compliance
- Advanced: Multi-burner coordination, atomization analysis

Reference Standards:
- ASME PTC 4: Fired Steam Generators
- API 535: Burners for Fired Heaters
- EPA 40 CFR Part 60/63
- EU Industrial Emissions Directive
"""

# Core calculators
from .stoichiometric_calculator import StoichiometricCalculator
from .combustion_efficiency_calculator import CombustionEfficiencyCalculator
from .flame_analysis_calculator import FlameAnalysisCalculator

# Optimization calculators
from .air_fuel_optimizer import AirFuelOptimizer
from .burner_performance_calculator import BurnerPerformanceCalculator

# Emissions calculators
from .emissions_calculator import EmissionsCalculator

# Advanced calculators
from .burner_tuning_optimizer import (
    BurnerTuningOptimizer,
    BurnerTuningInput,
    BurnerTuningOutput,
    FuelType,
    BurnerType,
    FlameStabilityStatus,
    AtomizationQuality,
    FuelProperties,
    AirFuelRatioResult,
    EmissionTargets,
    TurndownAnalysisResult,
    AirDistributionResult,
    FlameStabilityResult,
    DraftPressureResult,
    AtomizationResult,
    MultiBurnerResult,
    ProvenanceTracker as TuningProvenanceTracker,
    FUEL_PROPERTIES_DB,
)

from .emissions_predictor import (
    EmissionsPredictor,
    EmissionsPredictorInput,
    EmissionsPredictorOutput,
    FuelComposition,
    CombustionConditions,
    NOxPrediction,
    COPrediction,
    UHCPrediction,
    PMPrediction,
    ComplianceResult,
    ComplianceStatus,
    EmissionCreditsResult,
    LoadEmissionCurve,
    RegulatoryLimit,
    RegulatoryStandard,
    EmissionType,
    CombustionMode,
    ProvenanceTracker as EmissionsProvenanceTracker,
    EPA_NSPS_LIMITS,
    EPA_MACT_LIMITS,
    EU_IED_LIMITS,
)

__all__ = [
    # Core calculators
    'StoichiometricCalculator',
    'CombustionEfficiencyCalculator',
    'FlameAnalysisCalculator',

    # Optimization calculators
    'AirFuelOptimizer',
    'BurnerPerformanceCalculator',

    # Emissions calculators
    'EmissionsCalculator',

    # Burner Tuning Optimizer (Advanced)
    'BurnerTuningOptimizer',
    'BurnerTuningInput',
    'BurnerTuningOutput',
    'FuelType',
    'BurnerType',
    'FlameStabilityStatus',
    'AtomizationQuality',
    'FuelProperties',
    'AirFuelRatioResult',
    'EmissionTargets',
    'TurndownAnalysisResult',
    'AirDistributionResult',
    'FlameStabilityResult',
    'DraftPressureResult',
    'AtomizationResult',
    'MultiBurnerResult',
    'TuningProvenanceTracker',
    'FUEL_PROPERTIES_DB',

    # Emissions Predictor (Advanced)
    'EmissionsPredictor',
    'EmissionsPredictorInput',
    'EmissionsPredictorOutput',
    'FuelComposition',
    'CombustionConditions',
    'NOxPrediction',
    'COPrediction',
    'UHCPrediction',
    'PMPrediction',
    'ComplianceResult',
    'ComplianceStatus',
    'EmissionCreditsResult',
    'LoadEmissionCurve',
    'RegulatoryLimit',
    'RegulatoryStandard',
    'EmissionType',
    'CombustionMode',
    'EmissionsProvenanceTracker',
    'EPA_NSPS_LIMITS',
    'EPA_MACT_LIMITS',
    'EU_IED_LIMITS',

    # Legacy exports (for backwards compatibility)
    'EmissionsComplianceCalculator',
    'FuelPropertiesCalculator',
]

# Legacy aliases for backwards compatibility
EmissionsComplianceCalculator = EmissionsPredictor
FuelPropertiesCalculator = BurnerTuningOptimizer
