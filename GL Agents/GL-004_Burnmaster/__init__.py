"""
GL-004 BURNMASTER - Burner Optimization Agent

Burner optimization agent for air-fuel ratio control, flame stability monitoring,
turndown optimization, emissions control (NOx, CO), and combustion efficiency
maximization with SHAP/LIME explainability and real-time advisory capabilities.

Agent Details:
    - Agent ID: GL-004
    - Codename: BURNMASTER
    - Domain: Combustion Optimization
    - Type: Optimizer / Advisory + Control Support
    - Priority: P1
    - Business Value: $5B
    - Target: Q1 2026
    - Status: Consolidated into GL-018 UNIFIEDCOMBUSTION

Key Capabilities:
    - Air-fuel ratio optimization (excess O2 / lambda control)
    - Flame stability optimization
    - Turndown optimization
    - Emissions control (NOx, CO)
    - Advisory and closed-loop modes

Author: GreenLang AI Agent Workforce
Version: 1.0.0
License: Proprietary - GreenLang
"""

__version__ = "1.0.0"
__author__ = "GreenLang AI Agent Workforce"
__agent_id__ = "GL-004"
__codename__ = "BURNMASTER"

from .core import (
    BurnerConfig,
    BurnerProcessData,
    CombustionProperties,
    BurnerSystemOrchestrator,
)

from .combustion import (
    compute_stoichiometric_ratio,
    compute_excess_air,
    compute_lambda_value,
    compute_combustion_efficiency,
    compute_adiabatic_flame_temperature,
    compute_flue_gas_losses,
)

from .calculators import (
    AirFuelRatioCalculator,
    EmissionsCalculator,
    FlameStabilityCalculator,
    TurndownCalculator,
    CombustionKPICalculator,
)

from .optimization import (
    AirFuelOptimizer,
    NOxReductionOptimizer,
    TurndownOptimizer,
    BurnerTuningOptimizer,
    RecommendationEngine,
)

from .explainability import (
    CombustionPhysicsExplainer,
    SHAPExplainer,
    LIMEExplainer,
    ExplanationGenerator,
)

from .control import (
    AirFuelController,
    O2TrimController,
    FlameStabilityController,
    DamperPositionController,
)

from .safety import (
    CombustionSafetyEnvelope,
    FlameoutProtection,
    EmissionsLimitMonitor,
    InterlockManager,
)

from .monitoring import (
    AlertManager,
    HealthMonitor,
    MetricsCollector,
    EmissionsTrendMonitor,
)

from .audit import (
    AuditLogger,
    ProvenanceTracker,
    EvidencePackager,
    CalculationTrace,
)

from .uncertainty import (
    SensorUncertaintyManager,
    UncertaintyPropagator,
    UncertaintyGate,
)

__all__ = [
    "__version__",
    "__author__",
    "__agent_id__",
    "__codename__",
    "BurnerConfig",
    "BurnerProcessData",
    "CombustionProperties",
    "BurnerSystemOrchestrator",
    "compute_stoichiometric_ratio",
    "compute_excess_air",
    "compute_lambda_value",
    "compute_combustion_efficiency",
    "compute_adiabatic_flame_temperature",
    "compute_flue_gas_losses",
    "AirFuelRatioCalculator",
    "EmissionsCalculator",
    "FlameStabilityCalculator",
    "TurndownCalculator",
    "CombustionKPICalculator",
    "AirFuelOptimizer",
    "NOxReductionOptimizer",
    "TurndownOptimizer",
    "BurnerTuningOptimizer",
    "RecommendationEngine",
    "CombustionPhysicsExplainer",
    "SHAPExplainer",
    "LIMEExplainer",
    "ExplanationGenerator",
    "AirFuelController",
    "O2TrimController",
    "FlameStabilityController",
    "DamperPositionController",
    "CombustionSafetyEnvelope",
    "FlameoutProtection",
    "EmissionsLimitMonitor",
    "InterlockManager",
    "AlertManager",
    "HealthMonitor",
    "MetricsCollector",
    "EmissionsTrendMonitor",
    "AuditLogger",
    "ProvenanceTracker",
    "EvidencePackager",
    "CalculationTrace",
    "SensorUncertaintyManager",
    "UncertaintyPropagator",
    "UncertaintyGate",
]
