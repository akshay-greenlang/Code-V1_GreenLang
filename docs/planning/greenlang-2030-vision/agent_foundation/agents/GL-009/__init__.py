# -*- coding: utf-8 -*-
"""
GL-009 THERMALIQ - ThermalEfficiencyCalculator Package.

Zero-hallucination thermal efficiency calculations for industrial processes.
This package provides comprehensive thermal efficiency analysis including
First Law (energy) and Second Law (exergy) calculations, Sankey diagram
generation, industry benchmarking, and improvement opportunity identification.

Standards Compliance:
- ASME PTC 4.1 - Steam Generating Units
- ASME PTC 4 - Fired Steam Generators
- ISO 50001:2018 - Energy Management Systems
- EPA 40 CFR Part 60 - Emissions Standards

Example:
    >>> from gl_009 import ThermalEfficiencyOrchestrator, ThermalEfficiencyConfig
    >>> config = ThermalEfficiencyConfig()
    >>> orchestrator = ThermalEfficiencyOrchestrator(config)
    >>> result = await orchestrator.execute({
    ...     'operation_mode': 'calculate',
    ...     'energy_inputs': {'fuel_inputs': [...]},
    ...     'useful_outputs': {'process_heat_kw': 1000}
    ... })
    >>> print(f"Efficiency: {result['first_law_efficiency_percent']}%")

Modules:
    thermal_efficiency_orchestrator: Main orchestrator class
    tools: Deterministic calculation tools
    config: Pydantic configuration classes
    main: FastAPI application entry point
    greenlang: Determinism utilities

Author: GreenLang Foundation
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GreenLang Foundation"
__agent_id__ = "GL-009"
__codename__ = "THERMALIQ"

# Core exports
from .config import (
    ThermalEfficiencyConfig,
    CalculationConfig,
    VisualizationConfig,
    IntegrationConfig,
    BenchmarkConfig,
    MonitoringConfig,
    ProcessType,
    FuelType,
    CalculationMethod,
    VisualizationFormat,
    create_config,
    load_config_from_file
)

from .tools import (
    ThermalEfficiencyTools,
    ThermalEfficiencyResult,
    FirstLawEfficiencyResult,
    SecondLawEfficiencyResult,
    HeatBalanceResult,
    HeatLossBreakdown,
    SankeyDiagramResult,
    BenchmarkResult,
    ImprovementOpportunity,
    ExergyAnalysisResult,
    UncertaintyResult,
    TOOL_SCHEMAS,
    HEATING_VALUES,
    INDUSTRY_BENCHMARKS,
    CO2_EMISSION_FACTORS
)

from .thermal_efficiency_orchestrator import (
    ThermalEfficiencyOrchestrator,
    OperationMode,
    CalculationMethod as OrchestratorCalculationMethod,
    ValidationStatus,
    ThreadSafeCache,
    PerformanceMetrics,
    RetryHandler,
    create_orchestrator
)

# Greenlang utilities
try:
    from .greenlang import (
        DeterministicClock,
        DeterminismValidator,
        deterministic_uuid,
        calculate_provenance_hash,
        create_efficiency_uuid,
        create_audit_hash
    )
except ImportError:
    # Greenlang module may not be available
    DeterministicClock = None
    DeterminismValidator = None
    deterministic_uuid = None
    calculate_provenance_hash = None

# FastAPI app (optional import)
try:
    from .main import app
except ImportError:
    app = None

# All public exports
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__agent_id__",
    "__codename__",

    # Configuration
    "ThermalEfficiencyConfig",
    "CalculationConfig",
    "VisualizationConfig",
    "IntegrationConfig",
    "BenchmarkConfig",
    "MonitoringConfig",
    "ProcessType",
    "FuelType",
    "CalculationMethod",
    "VisualizationFormat",
    "create_config",
    "load_config_from_file",

    # Tools
    "ThermalEfficiencyTools",
    "ThermalEfficiencyResult",
    "FirstLawEfficiencyResult",
    "SecondLawEfficiencyResult",
    "HeatBalanceResult",
    "HeatLossBreakdown",
    "SankeyDiagramResult",
    "BenchmarkResult",
    "ImprovementOpportunity",
    "ExergyAnalysisResult",
    "UncertaintyResult",
    "TOOL_SCHEMAS",
    "HEATING_VALUES",
    "INDUSTRY_BENCHMARKS",
    "CO2_EMISSION_FACTORS",

    # Orchestrator
    "ThermalEfficiencyOrchestrator",
    "OperationMode",
    "ValidationStatus",
    "ThreadSafeCache",
    "PerformanceMetrics",
    "RetryHandler",
    "create_orchestrator",

    # Greenlang utilities
    "DeterministicClock",
    "DeterminismValidator",
    "deterministic_uuid",
    "calculate_provenance_hash",

    # FastAPI app
    "app"
]


def get_agent_info() -> dict:
    """
    Get agent information.

    Returns:
        Dictionary with agent identification and capabilities
    """
    return {
        "agent_id": __agent_id__,
        "codename": __codename__,
        "full_name": "ThermalEfficiencyCalculator",
        "version": __version__,
        "author": __author__,
        "description": "Zero-hallucination thermal efficiency calculations for industrial processes",
        "deterministic": True,
        "standards": [
            "ASME PTC 4.1",
            "ASME PTC 4",
            "ISO 50001:2018",
            "EPA 40 CFR Part 60"
        ],
        "operation_modes": [mode.value for mode in OperationMode],
        "tool_count": len(TOOL_SCHEMAS),
        "capabilities": [
            "First Law (energy) efficiency calculation",
            "Second Law (exergy) efficiency calculation",
            "Comprehensive heat balance analysis",
            "Sankey diagram generation",
            "Industry benchmark comparison",
            "Improvement opportunity identification",
            "Uncertainty quantification"
        ]
    }
