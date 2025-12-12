"""
GL-018 UNIFIEDCOMBUSTION - Unified Combustion Optimizer Agent

This module provides the UnifiedCombustionOptimizerAgent for comprehensive
combustion optimization of industrial boilers, furnaces, and ovens.

The agent consolidates GL-002 and GL-004 functionality, providing:
- NFPA 85/86 compliance checking
- O2 trim optimization
- CO optimization
- Excess air control
- Safety interlock verification
- SHAP/LIME-style explainability
- Causal inference for root cause analysis
- Zero-hallucination deterministic calculations
- SHA-256 provenance tracking

Agent ID: GL-018
Agent Name: UNIFIEDCOMBUSTION
Version: 1.0.0
Category: Combustion
Type: Optimizer
Priority: P0
Market Size: $24B

Zero-Hallucination Guarantee:
    All calculations use deterministic physics-based formulas from:
    - ASME PTC 4 Fired Steam Generators
    - EPA Method 19 for combustion efficiency
    - NFPA 85 Boiler and Combustion Systems Hazards Code
    - NFPA 86 Standard for Ovens and Furnaces
    - API 535 Burners for Fired Heaters

    NO ML/LLM is used in any calculation path, ensuring 100% reproducibility
    and complete audit trail compliance.

Example:
    >>> from gl_018_unified_combustion import UnifiedCombustionOptimizerAgent
    >>> from gl_018_unified_combustion import CombustionInput, FlueGasMeasurements
    >>>
    >>> agent = UnifiedCombustionOptimizerAgent()
    >>> input_data = CombustionInput(
    ...     equipment_id="BOILER-001",
    ...     equipment_type="boiler",
    ...     fuel_type="natural_gas",
    ...     fuel_flow_rate=500.0,
    ...     flue_gas=FlueGasMeasurements(
    ...         o2_percent=3.5,
    ...         co_ppm=50,
    ...         nox_ppm=80,
    ...         stack_temperature_c=350
    ...     ),
    ...     air_flow=AirFlowMeasurements(primary_air_flow_m3h=5000)
    ... )
    >>> result = agent.run(input_data)
    >>> print(f"Efficiency: {result.efficiency_metrics.combustion_efficiency_pct}%")
    >>> print(f"Optimal O2: {result.o2_trim.optimal_o2_pct}%")
    >>> print(f"NFPA Status: {result.nfpa_compliance.overall_status}")
"""

from .agent import UnifiedCombustionOptimizerAgent

from .schemas import (
    # Input models
    CombustionInput,
    FuelComposition,
    FlueGasMeasurements,
    FlameMetrics,
    AirFlowMeasurements,
    SafetyInterlockData,
    BurnerStatus,
    # Output models
    CombustionOutput,
    EfficiencyMetrics,
    OptimizationRecommendation,
    O2TrimRecommendation,
    ExcessAirRecommendation,
    EmissionsAnalysis,
    NFPAComplianceResult,
    NFPAViolation,
    SafetyInterlockAssessment,
    ExplainabilityReport,
    FeatureImportance,
    CausalRelationship,
    AttentionVisualization,
    ProvenanceRecord,
    CalculationStep,
    # Enums
    FuelType,
    EquipmentType,
    OptimizationMode,
    ComplianceStatus,
    SafetyInterlockStatus,
    Priority,
    CausalRelationType,
    # Config
    AgentConfig,
)

# Agent metadata
AGENT_ID = "GL-018"
AGENT_NAME = "UNIFIEDCOMBUSTION"
VERSION = "1.0.0"
DESCRIPTION = "Unified Combustion Optimizer Agent for industrial combustion equipment"
CATEGORY = "Combustion"
TYPE = "Optimizer"
PRIORITY = "P0"
MARKET_SIZE = "$24B"

# Supported standards
SUPPORTED_STANDARDS = [
    "NFPA 85",
    "NFPA 86",
    "ASME PTC 4",
    "EPA Method 19",
    "API 535",
]

# Supported fuel types
SUPPORTED_FUELS = [fuel.value for fuel in FuelType]

# Supported equipment types
SUPPORTED_EQUIPMENT = [eq.value for eq in EquipmentType]

__all__ = [
    # Agent class
    "UnifiedCombustionOptimizerAgent",

    # Input models
    "CombustionInput",
    "FuelComposition",
    "FlueGasMeasurements",
    "FlameMetrics",
    "AirFlowMeasurements",
    "SafetyInterlockData",
    "BurnerStatus",

    # Output models
    "CombustionOutput",
    "EfficiencyMetrics",
    "OptimizationRecommendation",
    "O2TrimRecommendation",
    "ExcessAirRecommendation",
    "EmissionsAnalysis",
    "NFPAComplianceResult",
    "NFPAViolation",
    "SafetyInterlockAssessment",
    "ExplainabilityReport",
    "FeatureImportance",
    "CausalRelationship",
    "AttentionVisualization",
    "ProvenanceRecord",
    "CalculationStep",

    # Enums
    "FuelType",
    "EquipmentType",
    "OptimizationMode",
    "ComplianceStatus",
    "SafetyInterlockStatus",
    "Priority",
    "CausalRelationType",

    # Config
    "AgentConfig",

    # Metadata
    "AGENT_ID",
    "AGENT_NAME",
    "VERSION",
    "DESCRIPTION",
    "CATEGORY",
    "TYPE",
    "PRIORITY",
    "MARKET_SIZE",
    "SUPPORTED_STANDARDS",
    "SUPPORTED_FUELS",
    "SUPPORTED_EQUIPMENT",
]
