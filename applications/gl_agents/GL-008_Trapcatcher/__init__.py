# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER

Steam Trap Monitoring and Diagnostic Agent for GreenLang Platform.

This agent provides comprehensive steam trap monitoring including:
- Real-time acoustic and thermal analysis
- Multimodal diagnostic classification
- Energy loss quantification with CO2e
- SHAP-compatible explainability
- Fleet-wide analysis and prioritization

Standards:
- ASME PTC 39: Steam Traps - Performance Test Codes
- DOE Steam System Assessment Protocol
- ISO 7841: Automatic steam traps - Steam loss determination

Zero-Hallucination Guarantee:
All calculations use deterministic engineering formulas.
No LLM or AI inference in any calculation path.
Same inputs always produce identical outputs.

Example:
    >>> from trapcatcher import TrapcatcherAgent, TrapDiagnosticInput
    >>> agent = TrapcatcherAgent()
    >>> result = agent.diagnose_trap(TrapDiagnosticInput(
    ...     trap_id="ST-001",
    ...     acoustic_amplitude_db=75.0,
    ...     inlet_temp_c=185.0,
    ...     outlet_temp_c=180.0,
    ...     pressure_bar_g=10.0
    ... ))
    >>> print(f"Condition: {result.condition}")

Author: GreenLang Team
Date: December 2025
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-008"
__agent_name__ = "TRAPCATCHER"

# Main agent
from .agent import (
    TrapcatcherAgent,
    AgentConfig,
    AgentMode,
    TrapDiagnosticInput,
    DiagnosticOutput,
    FleetSummary,
    AlertLevel,
)

# Core classifier
from .core import (
    TrapStateClassifier,
    ClassificationConfig,
    ClassificationResult,
    TrapCondition,
    ConfidenceLevel,
)

# Calculators
from .calculators import (
    SteamTrapEnergyLossCalculator,
    TrapPopulationAnalyzer,
    AcousticDiagnosticCalculator,
)

# Explainability
from .explainability import (
    DiagnosticExplainer,
    ExplainerConfig,
    ExplanationResult,
)

# Reporting
from .reporting import (
    ClimateIntelligenceReporter,
    ReporterConfig,
    EmissionsReport,
    FleetClimateMetrics,
)

# Optimization
from .optimization import (
    MaintenanceRouteOptimizer,
    OptimizerConfig,
    MaintenanceTask,
    OptimizedRoute,
)

# Diagnostics
from .diagnostics import (
    FailurePredictor,
    PredictorConfig,
    FailurePrediction,
    RiskAssessment,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__agent_id__",
    "__agent_name__",
    # Main agent
    "TrapcatcherAgent",
    "AgentConfig",
    "AgentMode",
    "TrapDiagnosticInput",
    "DiagnosticOutput",
    "FleetSummary",
    "AlertLevel",
    # Core
    "TrapStateClassifier",
    "ClassificationConfig",
    "ClassificationResult",
    "TrapCondition",
    "ConfidenceLevel",
    # Calculators
    "SteamTrapEnergyLossCalculator",
    "TrapPopulationAnalyzer",
    "AcousticDiagnosticCalculator",
    # Explainability
    "DiagnosticExplainer",
    "ExplainerConfig",
    "ExplanationResult",
    # Reporting
    "ClimateIntelligenceReporter",
    "ReporterConfig",
    "EmissionsReport",
    "FleetClimateMetrics",
    # Optimization
    "MaintenanceRouteOptimizer",
    "OptimizerConfig",
    "MaintenanceTask",
    "OptimizedRoute",
    # Diagnostics
    "FailurePredictor",
    "PredictorConfig",
    "FailurePrediction",
    "RiskAssessment",
]
