# -*- coding: utf-8 -*-
"""
GL-001 ProcessHeatOrchestrator Calculation Engines

Zero-hallucination calculation engines for process heat operations
with 100% deterministic guarantees and complete provenance tracking.

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

from .thermal_efficiency import (
    ThermalEfficiencyCalculator,
    PlantData
)

from .heat_distribution import (
    HeatDistributionOptimizer,
    HeatDemandNode,
    DistributionPipe,
    HeatSource
)

from .energy_balance import (
    EnergyBalanceValidator,
    EnergyBalanceData,
    EnergyFlow,
    EnergyFlowType
)

from .emissions_compliance import (
    EmissionsComplianceChecker,
    EmissionsData,
    EmissionMeasurement,
    RegulatoryLimit,
    PollutantType,
    ComplianceStatus
)

from .kpi_calculator import (
    KPICalculator,
    OperationalData
)

__all__ = [
    # Provenance
    'ProvenanceTracker',
    'ProvenanceRecord',
    'ProvenanceValidator',
    'CalculationStep',
    'create_calculation_hash',

    # Thermal Efficiency
    'ThermalEfficiencyCalculator',
    'PlantData',

    # Heat Distribution
    'HeatDistributionOptimizer',
    'HeatDemandNode',
    'DistributionPipe',
    'HeatSource',

    # Energy Balance
    'EnergyBalanceValidator',
    'EnergyBalanceData',
    'EnergyFlow',
    'EnergyFlowType',

    # Emissions Compliance
    'EmissionsComplianceChecker',
    'EmissionsData',
    'EmissionMeasurement',
    'RegulatoryLimit',
    'PollutantType',
    'ComplianceStatus',

    # KPI
    'KPICalculator',
    'OperationalData'
]

__version__ = '1.0.0'