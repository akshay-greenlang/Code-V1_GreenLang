# -*- coding: utf-8 -*-
"""
GL-004 Burnmaster - Explainability Module

This module provides comprehensive explainability for combustion optimization,
including physics-based explanations, SHAP/LIME model explanations, constraint
analysis, and recommendation explanations.

Key Components:
    - PhysicsExplainer: Physics-based explanations (stoichiometry, emissions, stability)
    - SHAPExplainer: SHAP-based model explanations
    - LIMEExplainer: LIME-based local explanations with counterfactuals
    - ConstraintExplainer: Optimization constraint analysis
    - RecommendationExplainer: Recommendation impact and comparison
    - ExplanationGenerator: Comprehensive explanation orchestration

Design Principles:
    - Dual Audience: Explanations for both operators (plain language) and engineers (technical)
    - Confidence Levels: All explanations include confidence scores and uncertainty bounds
    - Serializable: All outputs are Pydantic models for logging and UI display
    - Provenance: SHA-256 hashes for complete audit trails
    - Zero Hallucination: Physics-based where possible, model-based with uncertainty

Example:
    >>> from explainability import ExplanationGenerator, ExplanationContext
    >>> generator = ExplanationGenerator()
    >>> context = ExplanationContext(
    ...     context_id="ctx-001",
    ...     boiler_id="boiler-1",
    ...     current_state={"o2_percent": 3.5},
    ...     optimization_result={"status": "optimal"}
    ... )
    >>> explanation = generator.generate_full_explanation(context)
    >>> print(explanation.operator_summary)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

# Pydantic schemas (data models)
from .explainability_payload import (
    # Enums
    ExplanationType,
    AudienceLevel,
    ConfidenceLevel,
    ConstraintStatus,
    ImpactDirection,
    # Base models
    BaseExplanation,
    FeatureContribution,
    UncertaintyBounds,
    # Physics explanations
    StoichiometryExplanation,
    EfficiencyExplanation,
    EmissionExplanation,
    StabilityExplanation,
    PhysicsExplanation,
    # SHAP explanations
    SHAPValues,
    SHAPExplanation,
    # LIME explanations
    LIMEExplanation,
    ConsistencyReport,
    CounterfactualExplanation,
    # Constraint explanations
    ConstraintViolation,
    ConstraintExplanation,
    ViolationExplanation,
    MarginExplanation,
    RelaxationSuggestion,
    # Recommendation explanations
    Recommendation,
    ImpactPrediction,
    RecommendationExplanation,
    ComparisonTable,
    # Comprehensive explanations
    ExplanationContext,
    ComprehensiveExplanation,
    OperatorExplanation,
    EngineerExplanation,
)

# Physics explainer
from .physics_explainer import (
    PhysicsExplainer,
    PhysicsExplainerConfig,
)

# SHAP explainer
from .shap_explainer import (
    SHAPExplainer,
    SHAPExplainerConfig,
)

# LIME explainer
from .lime_explainer import (
    LIMEExplainer,
    LIMEExplainerConfig,
)

# Constraint explainer
from .constraint_explainer import (
    ConstraintExplainer,
    ConstraintExplainerConfig,
)

# Recommendation explainer
from .recommendation_explainer import (
    RecommendationExplainer,
    RecommendationExplainerConfig,
)

# Explanation generator
from .explanation_generator import (
    ExplanationGenerator,
    ExplanationGeneratorConfig,
)


__all__ = [
    # Enums
    "ExplanationType",
    "AudienceLevel",
    "ConfidenceLevel",
    "ConstraintStatus",
    "ImpactDirection",
    # Base models
    "BaseExplanation",
    "FeatureContribution",
    "UncertaintyBounds",
    # Physics explanations
    "StoichiometryExplanation",
    "EfficiencyExplanation",
    "EmissionExplanation",
    "StabilityExplanation",
    "PhysicsExplanation",
    # SHAP explanations
    "SHAPValues",
    "SHAPExplanation",
    # LIME explanations
    "LIMEExplanation",
    "ConsistencyReport",
    "CounterfactualExplanation",
    # Constraint explanations
    "ConstraintViolation",
    "ConstraintExplanation",
    "ViolationExplanation",
    "MarginExplanation",
    "RelaxationSuggestion",
    # Recommendation explanations
    "Recommendation",
    "ImpactPrediction",
    "RecommendationExplanation",
    "ComparisonTable",
    # Comprehensive explanations
    "ExplanationContext",
    "ComprehensiveExplanation",
    "OperatorExplanation",
    "EngineerExplanation",
    # Explainer classes
    "PhysicsExplainer",
    "PhysicsExplainerConfig",
    "SHAPExplainer",
    "SHAPExplainerConfig",
    "LIMEExplainer",
    "LIMEExplainerConfig",
    "ConstraintExplainer",
    "ConstraintExplainerConfig",
    "RecommendationExplainer",
    "RecommendationExplainerConfig",
    "ExplanationGenerator",
    "ExplanationGeneratorConfig",
]

__version__ = "1.0.0"
__author__ = "GreenLang Process Heat Team"
