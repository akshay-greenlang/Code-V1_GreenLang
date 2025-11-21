# -*- coding: utf-8 -*-
"""
GL-003 Steam System Calculators - Zero Hallucination Suite

Production-quality calculators for steam system analysis with complete
provenance tracking and bit-perfect reproducibility.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: IAPWS-IF97, ASME, ASHRAE, EPA AP-42, GHG Protocol
"""

# Core utilities
from .provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    CalculationStep,
    ProvenanceValidator,
    create_calculation_hash
)

# Steam properties (IAPWS-IF97)
from .steam_properties import (
    SteamPropertiesCalculator,
    SteamProperties
)

# Distribution efficiency
from .distribution_efficiency import (
    DistributionEfficiencyCalculator,
    DistributionResults,
    PipeSegment
)

# Leak detection
from .leak_detection import (
    LeakDetectionCalculator,
    LeakDetectionResult,
    FlowMeasurement
)

# Heat loss calculations
from .heat_loss_calculator import (
    HeatLossCalculator,
    HeatLossResult
)

# Condensate optimization
from .condensate_optimizer import (
    CondensateOptimizer,
    CondensateResults,
    CondensateData
)

# Steam trap analysis
from .steam_trap_analyzer import (
    SteamTrapAnalyzer,
    TrapAnalysisResult,
    SteamTrapData
)

# Pressure and flow analysis
from .pressure_analysis import (
    PressureAnalysisCalculator,
    PressureAnalysisResult,
    PipeFlowData
)

# Emissions calculations
from .emissions_calculator import (
    EmissionsCalculator,
    EmissionsResult,
    FuelConsumptionData
)

# KPI dashboard
from .kpi_calculator import (
    KPICalculator,
    KPIDashboard,
    SystemMetrics
)

__all__ = [
    # Provenance
    'ProvenanceTracker',
    'ProvenanceRecord',
    'CalculationStep',
    'ProvenanceValidator',
    'create_calculation_hash',

    # Steam Properties
    'SteamPropertiesCalculator',
    'SteamProperties',

    # Distribution
    'DistributionEfficiencyCalculator',
    'DistributionResults',
    'PipeSegment',

    # Leak Detection
    'LeakDetectionCalculator',
    'LeakDetectionResult',
    'FlowMeasurement',

    # Heat Loss
    'HeatLossCalculator',
    'HeatLossResult',

    # Condensate
    'CondensateOptimizer',
    'CondensateResults',
    'CondensateData',

    # Steam Traps
    'SteamTrapAnalyzer',
    'TrapAnalysisResult',
    'SteamTrapData',

    # Pressure Analysis
    'PressureAnalysisCalculator',
    'PressureAnalysisResult',
    'PipeFlowData',

    # Emissions
    'EmissionsCalculator',
    'EmissionsResult',
    'FuelConsumptionData',

    # KPIs
    'KPICalculator',
    'KPIDashboard',
    'SystemMetrics',
]

__version__ = '1.0.0'
__author__ = 'GL-CalculatorEngineer'
__standards__ = [
    'IAPWS-IF97',
    'ASME Steam Tables',
    'ASME B31.1',
    'ASHRAE Handbook',
    'EPA AP-42',
    'GHG Protocol',
    'ISO 12241',
    'ISO 5167'
]
