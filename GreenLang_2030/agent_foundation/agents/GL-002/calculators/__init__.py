"""
GL-002 BoilerEfficiencyOptimizer Calculation Engines

Zero-hallucination calculation engines for boiler efficiency optimization
with 100% deterministic guarantees and complete provenance tracking.

Modules:
1. combustion_efficiency - ASME PTC 4.1 combustion efficiency
2. fuel_optimization - Fuel consumption and cost optimization
3. emissions_calculator - EPA emissions calculations (NOx, CO2, SOx)
4. steam_generation - Steam output quality optimization
5. heat_transfer - Boiler heat transfer efficiency
6. blowdown_optimizer - Optimal blowdown rate calculations
7. economizer_performance - Economizer efficiency calculations
8. provenance - Zero-hallucination provenance tracking

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from .provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    ProvenanceValidator,
    CalculationStep,
    create_calculation_hash
)

from .combustion_efficiency import (
    CombustionEfficiencyCalculator,
    CombustionData,
    CombustionResults
)

from .fuel_optimization import (
    FuelOptimizationCalculator,
    FuelData,
    BoilerOperatingData,
    OptimizationConstraints,
    OptimizationResults
)

from .emissions_calculator import (
    EmissionsCalculator,
    BoilerEmissionData,
    EmissionFactors,
    RegulatoryLimits,
    EmissionResults,
    ComplianceStatus
)

from .steam_generation import (
    SteamGenerationCalculator,
    SteamQualityData,
    SteamProperties,
    SteamGenerationResults
)

from .heat_transfer import (
    HeatTransferCalculator,
    HeatTransferData,
    HeatTransferResults
)

from .blowdown_optimizer import (
    BlowdownOptimizer,
    BlowdownData,
    BlowdownResults
)

from .economizer_performance import (
    EconomizerPerformanceCalculator,
    EconomizerData,
    EconomizerResults
)

__all__ = [
    # Provenance
    'ProvenanceTracker',
    'ProvenanceRecord',
    'ProvenanceValidator',
    'CalculationStep',
    'create_calculation_hash',

    # Combustion Efficiency
    'CombustionEfficiencyCalculator',
    'CombustionData',
    'CombustionResults',

    # Fuel Optimization
    'FuelOptimizationCalculator',
    'FuelData',
    'BoilerOperatingData',
    'OptimizationConstraints',
    'OptimizationResults',

    # Emissions
    'EmissionsCalculator',
    'BoilerEmissionData',
    'EmissionFactors',
    'RegulatoryLimits',
    'EmissionResults',
    'ComplianceStatus',

    # Steam Generation
    'SteamGenerationCalculator',
    'SteamQualityData',
    'SteamProperties',
    'SteamGenerationResults',

    # Heat Transfer
    'HeatTransferCalculator',
    'HeatTransferData',
    'HeatTransferResults',

    # Blowdown Optimization
    'BlowdownOptimizer',
    'BlowdownData',
    'BlowdownResults',

    # Economizer Performance
    'EconomizerPerformanceCalculator',
    'EconomizerData',
    'EconomizerResults',
]

__version__ = '1.0.0'
__author__ = 'GL-CalculatorEngineer'
__description__ = 'Zero-hallucination boiler efficiency calculation engines with 100% deterministic guarantees'
