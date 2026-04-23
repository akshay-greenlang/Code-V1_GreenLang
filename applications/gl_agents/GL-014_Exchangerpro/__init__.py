"""
GL-014_EXCHANGERPRO - Heat Exchanger Optimizer Agent

Industrial heat exchanger performance monitoring, fouling prediction,
and cleaning schedule optimization with zero-hallucination calculations.

This agent provides:
- Deterministic thermal calculations (Q, UA, LMTD, epsilon-NTU, pressure drop)
- ML-based fouling prediction with SHAP/LIME explainability
- Cost-optimized cleaning schedule recommendations
- TEMA-compliant terminology and calculations
- Full provenance tracking with SHA-256 hashes

Version: 1.0.0
Agent ID: GL-014
Category: Thermal/Optimization
Standards: TEMA, ASME

Zero-Hallucination Principle:
    All heat-transfer and pressure-drop calculations are performed by the
    deterministic thermal engine. Natural language generation is used only
    to describe computed results, summarize explanations, and format
    recommendations. The LLM never computes Q, UA, LMTD, NTU, or delta-P.
"""

__version__ = "1.0.0"
__agent_id__ = "GL-014"
__agent_name__ = "EXCHANGERPRO"
__full_name__ = "Heat Exchanger Optimizer"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.orchestrator import ExchangerProOrchestrator
    from .core.config import AgentSettings
    from .core.schemas import (
        ExchangerConfig,
        OperatingState,
        ThermalKPIs,
        FoulingState,
        CleaningRecommendation,
    )

__all__ = [
    "__version__",
    "__agent_id__",
    "__agent_name__",
    "__full_name__",
    "ExchangerProOrchestrator",
    "AgentSettings",
    "ExchangerConfig",
    "OperatingState",
    "ThermalKPIs",
    "FoulingState",
    "CleaningRecommendation",
]


def get_version() -> str:
    """Return agent version."""
    return __version__


def get_agent_info() -> dict:
    """Return agent metadata."""
    return {
        "agent_id": __agent_id__,
        "agent_name": __agent_name__,
        "full_name": __full_name__,
        "version": __version__,
        "category": "thermal_optimization",
        "standards": ["TEMA", "ASME"],
        "capabilities": [
            "thermal_performance_monitoring",
            "fouling_prediction",
            "cleaning_schedule_optimization",
            "explainability_reporting",
            "cmms_integration",
        ],
    }
