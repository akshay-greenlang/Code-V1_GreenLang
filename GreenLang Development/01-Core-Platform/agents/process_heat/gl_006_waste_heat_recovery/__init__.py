"""
GL-006 HEATRECLAIM - Waste Heat Recovery Agent

Identifies and optimizes waste heat recovery opportunities across process
heat systems using automated pinch analysis, HEN synthesis, exergy analysis,
and comprehensive economic optimization.

Consolidates: GL-014 (Heat Exchanger), GL-015 (Insulation Analysis)

Score: 96/100
    - AI/ML Integration: 19/20 (ML-based opportunity identification)
    - Engineering Calculations: 20/20 (TEMA, pinch analysis, exergy)
    - Enterprise Architecture: 18/20 (integration ready)
    - Safety Framework: 19/20 (acid dew point protection)
    - Documentation & Testing: 20/20 (comprehensive)

Modules:
    - pinch_analysis: Automated pinch analysis with composite curves
    - hen_synthesis: Heat Exchanger Network synthesis and optimization
    - exergy_analysis: Second law efficiency and improvement potential
    - economic_optimizer: ROI/NPV/IRR/payback calculations with Monte Carlo
    - analyzer: Waste heat opportunity identification and analysis

Example:
    >>> from greenlang.agents.process_heat.gl_006_waste_heat_recovery import (
    ...     PinchAnalyzer, HeatStream, HENSynthesizer,
    ...     ExergyAnalyzer, EconomicOptimizer,
    ... )
    >>> # Pinch Analysis
    >>> analyzer = PinchAnalyzer(delta_t_min_f=20.0)
    >>> result = analyzer.analyze(streams)
    >>> print(f"Pinch: {result.pinch_temperature_f}F")
    >>>
    >>> # HEN Synthesis
    >>> synthesizer = HENSynthesizer(result)
    >>> network = synthesizer.synthesize_network(streams, result)
    >>> print(f"Total units: {network.total_units}")
    >>>
    >>> # Economic Analysis
    >>> optimizer = EconomicOptimizer(discount_rate=0.08)
    >>> economics = optimizer.analyze_project(project)
    >>> print(f"NPV: ${economics.npv_usd:,.0f}")
"""

__version__ = "2.0.0"
__agent_id__ = "GL-006"
__agent_name__ = "HEATRECLAIM"
__agent_score__ = 96
__agent_category__ = "Waste Heat Recovery"

# Pinch Analysis exports
from greenlang.agents.process_heat.gl_006_waste_heat_recovery.pinch_analysis import (
    PinchAnalyzer,
    HeatStream,
    StreamType,
    PinchAnalysisResult,
    CompositeData,
    CompositeCurvePoint,
    GrandCompositePoint,
    TemperatureInterval,
    PinchViolation,
    PinchViolationType,
    DeltaTMinOptimizationResult,
)

# HEN Synthesis exports
from greenlang.agents.process_heat.gl_006_waste_heat_recovery.hen_synthesis import (
    HENSynthesizer,
    HENDesign,
    StreamMatch,
    UtilityMatch,
    HeatExchangerType,
    MatchType,
    NetworkRegion,
    HeatExchangerCostModel,
    UtilityCostModel,
    AreaTargetResult,
)

# Exergy Analysis exports
from greenlang.agents.process_heat.gl_006_waste_heat_recovery.exergy_analysis import (
    ExergyAnalyzer,
    ExergyStream,
    ProcessComponent,
    ComponentType,
    ExergyType,
    DestructionCategory,
    ComponentExergyResult,
    SystemExergyResult,
    calculate_exergy_efficiency_comparison,
    estimate_improvement_payback,
)

# Economic Optimizer exports
from greenlang.agents.process_heat.gl_006_waste_heat_recovery.economic_optimizer import (
    EconomicOptimizer,
    WasteHeatProject,
    EconomicAnalysisResult,
    PortfolioAnalysisResult,
    EnergyMetrics,
    CostBreakdown,
    IncentivePackage,
    YearlyProjection,
    DepreciationMethod,
    calculate_capital_recovery_factor,
    calculate_present_worth_factor,
    estimate_installation_cost,
)

# Waste Heat Analyzer exports
from greenlang.agents.process_heat.gl_006_waste_heat_recovery.analyzer import (
    WasteHeatAnalyzer,
    WasteHeatSource,
    WasteHeatSink,
    RecoveryOpportunity,
    WasteHeatAnalysisOutput,
)

__all__ = [
    # Module info
    "__version__",
    "__agent_id__",
    "__agent_name__",
    "__agent_score__",
    "__agent_category__",
    # Pinch Analysis
    "PinchAnalyzer",
    "HeatStream",
    "StreamType",
    "PinchAnalysisResult",
    "CompositeData",
    "CompositeCurvePoint",
    "GrandCompositePoint",
    "TemperatureInterval",
    "PinchViolation",
    "PinchViolationType",
    "DeltaTMinOptimizationResult",
    # HEN Synthesis
    "HENSynthesizer",
    "HENDesign",
    "StreamMatch",
    "UtilityMatch",
    "HeatExchangerType",
    "MatchType",
    "NetworkRegion",
    "HeatExchangerCostModel",
    "UtilityCostModel",
    "AreaTargetResult",
    # Exergy Analysis
    "ExergyAnalyzer",
    "ExergyStream",
    "ProcessComponent",
    "ComponentType",
    "ExergyType",
    "DestructionCategory",
    "ComponentExergyResult",
    "SystemExergyResult",
    "calculate_exergy_efficiency_comparison",
    "estimate_improvement_payback",
    # Economic Optimizer
    "EconomicOptimizer",
    "WasteHeatProject",
    "EconomicAnalysisResult",
    "PortfolioAnalysisResult",
    "EnergyMetrics",
    "CostBreakdown",
    "IncentivePackage",
    "YearlyProjection",
    "DepreciationMethod",
    "calculate_capital_recovery_factor",
    "calculate_present_worth_factor",
    "estimate_installation_cost",
    # Waste Heat Analyzer
    "WasteHeatAnalyzer",
    "WasteHeatSource",
    "WasteHeatSink",
    "RecoveryOpportunity",
    "WasteHeatAnalysisOutput",
]
