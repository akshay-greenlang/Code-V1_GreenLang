"""
GL-003 UNIFIEDSTEAM - Explainability Module

Provides multi-level explanations for steam system optimization recommendations:
- Physics-based explanations (thermodynamic traces, constraint analysis)
- ML-based explanations (SHAP global/local, LIME single-event)
- Human-readable explanations (engineering terms, operator briefings)
"""

from .physics_explainer import (
    PhysicsExplainer,
    PhysicsTrace,
    PropertyExplanation,
    ActiveConstraint,
    BalanceExplanation,
    ThermodynamicState,
)
from .shap_explainer import (
    SHAPExplainer,
    FeatureImportance,
    LocalExplanation,
    AssetExplanation,
    SHAPVisualization,
)
from .lime_explainer import (
    LIMEExplainer,
    LIMEExplanation,
    AnomalyExplanation,
    Counterfactual,
)
from .explanation_generator import (
    ExplanationGenerator,
    UserExplanation,
    EngineeringExplanation,
    OperatorBriefing,
)
from .explainability_payload import (
    ExplainabilityPayload,
    PrimaryDriver,
    SupportingEvidence,
    VerificationStep,
)

__all__ = [
    # Physics explainer
    "PhysicsExplainer",
    "PhysicsTrace",
    "PropertyExplanation",
    "ActiveConstraint",
    "BalanceExplanation",
    "ThermodynamicState",
    # SHAP explainer
    "SHAPExplainer",
    "FeatureImportance",
    "LocalExplanation",
    "AssetExplanation",
    "SHAPVisualization",
    # LIME explainer
    "LIMEExplainer",
    "LIMEExplanation",
    "AnomalyExplanation",
    "Counterfactual",
    # Explanation generator
    "ExplanationGenerator",
    "UserExplanation",
    "EngineeringExplanation",
    "OperatorBriefing",
    # Payload
    "ExplainabilityPayload",
    "PrimaryDriver",
    "SupportingEvidence",
    "VerificationStep",
]
