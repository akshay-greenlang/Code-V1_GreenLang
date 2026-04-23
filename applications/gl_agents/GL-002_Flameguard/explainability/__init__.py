"""
GL-002 FLAMEGUARD - Explainability Module

SHAP/LIME-based explanations for ML-driven decisions.
Provides transparent, physics-grounded explanations for all
optimization recommendations following GreenLang zero-hallucination principles.

This module includes:
- DecisionExplainer: SHAP/LIME-based explanations for combustion decisions
- OptimizationExplainer: Detailed optimization recommendation explanations
- ExplanationAuditLogger: Audit logging for explainability decisions
- AuditedDecisionExplainer: Wrapper with automatic audit logging
"""

from .decision_explainer import (
    DecisionExplainer,
    DecisionExplanation,
    FeatureContribution,
    LIMEExplanation,
    PhysicsGrounding,
    CounterfactualExplanation,
    ExplanationType,
    FeatureCategory,
    ImpactDirection,
    EFFICIENCY_SENSITIVITIES,
    REFERENCE_VALUES,
)
from .optimization_explainer import (
    OptimizationExplainer,
    OptimizationExplanation,
    OptimizationFactor,
    TradeoffAnalysis,
)
from .explanation_audit import (
    ExplanationAuditLogger,
    ExplanationAuditEntry,
    ExplanationAuditEventType,
    AuditedDecisionExplainer,
)

__all__ = [
    # Decision Explainer
    "DecisionExplainer",
    "DecisionExplanation",
    "FeatureContribution",
    "LIMEExplanation",
    "PhysicsGrounding",
    "CounterfactualExplanation",
    "ExplanationType",
    "FeatureCategory",
    "ImpactDirection",
    "EFFICIENCY_SENSITIVITIES",
    "REFERENCE_VALUES",
    # Optimization Explainer
    "OptimizationExplainer",
    "OptimizationExplanation",
    "OptimizationFactor",
    "TradeoffAnalysis",
    # Explanation Audit
    "ExplanationAuditLogger",
    "ExplanationAuditEntry",
    "ExplanationAuditEventType",
    "AuditedDecisionExplainer",
]
