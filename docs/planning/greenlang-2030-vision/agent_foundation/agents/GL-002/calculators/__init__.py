# -*- coding: utf-8 -*-
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
9. boiler_efficiency_asme - Complete ASME PTC 4 indirect method (13 heat losses)
10. load_allocation_optimizer - Multi-boiler load allocation and dispatch

Author: GL-CalculatorEngineer
Version: 1.1.0
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
    OptimizationConstraints
)

from .emissions_calculator import (
    EmissionsCalculator,
    BoilerEmissionData,
    EmissionFactors,
    RegulatoryLimits
)

from .steam_generation import (
    SteamGenerationCalculator,
    SteamGenerationData
)

from .heat_transfer import (
    HeatTransferCalculator,
    HeatTransferData
)

from .blowdown_optimizer import (
    BlowdownOptimizer,
    BlowdownData
)

from .economizer_performance import (
    EconomizerPerformanceCalculator,
    EconomizerData
)

from .boiler_efficiency_asme import (
    ASMEPTC4Calculator,
    ASMEPTC4Result,
    FuelAnalysis,
    FlueGasAnalysis,
    AmbientConditions,
    BoilerParameters,
    HeatLossBreakdown,
    FuelType,
    calculate_boiler_efficiency_asme_ptc4
)

from .load_allocation_optimizer import (
    LoadAllocationOptimizer,
    LoadAllocationResult,
    BoilerCharacteristics,
    LoadAllocationConstraints,
    BoilerLoadAllocation,
    MarginalCostPoint,
    BoilerStatus,
    OptimizationObjective,
    optimize_multi_boiler_load
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

    # Emissions
    'EmissionsCalculator',
    'BoilerEmissionData',
    'EmissionFactors',
    'RegulatoryLimits',

    # Steam Generation
    'SteamGenerationCalculator',
    'SteamGenerationData',

    # Heat Transfer
    'HeatTransferCalculator',
    'HeatTransferData',

    # Blowdown Optimization
    'BlowdownOptimizer',
    'BlowdownData',

    # Economizer Performance
    'EconomizerPerformanceCalculator',
    'EconomizerData',

    # ASME PTC 4 Boiler Efficiency (Complete)
    'ASMEPTC4Calculator',
    'ASMEPTC4Result',
    'FuelAnalysis',
    'FlueGasAnalysis',
    'AmbientConditions',
    'BoilerParameters',
    'HeatLossBreakdown',
    'FuelType',
    'calculate_boiler_efficiency_asme_ptc4',

    # Load Allocation Optimizer
    'LoadAllocationOptimizer',
    'LoadAllocationResult',
    'BoilerCharacteristics',
    'LoadAllocationConstraints',
    'BoilerLoadAllocation',
    'MarginalCostPoint',
    'BoilerStatus',
    'OptimizationObjective',
    'optimize_multi_boiler_load',
]

__version__ = '1.1.0'
__author__ = 'GL-CalculatorEngineer'
__description__ = 'Zero-hallucination boiler efficiency calculation engines with 100% deterministic guarantees'
