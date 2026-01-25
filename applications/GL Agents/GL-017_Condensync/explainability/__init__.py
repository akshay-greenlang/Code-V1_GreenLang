# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Explainability Module

Comprehensive explainability package for condenser optimization providing:
- Physics-based diagnostic explanations
- SHAP-based feature importance
- LIME local interpretable explanations
- Physics narrative generation for multiple audiences

Zero-Hallucination Guarantee:
All explanations derived from deterministic calculations.
No LLM or AI inference in explanation generation.
Complete provenance tracking with SHA-256 hashes.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from .diagnostic_explainer import (
    CondenserDiagnosticExplainer,
    ExplainerConfig,
    DiagnosticExplanation,
    FeatureContribution,
    CounterfactualExplanation,
    ConstraintVisualization,
    EvidenceChain,
    ContributionDirection,
    ExplanationStyle,
    EvidenceStrength,
    ConstraintType,
    PhysicsParameter,
    FEATURE_BASELINES,
    PHYSICS_EXPLANATIONS,
)

from .shap_explainer import (
    CondenserSHAPExplainer,
    SHAPExplainerConfig,
    SHAPExplanation,
    FeatureImportance,
    GlobalFeatureImportance,
    InteractionEffect,
    GlobalImportanceTrend,
    ExplainerType,
    ImportanceType,
    CONDENSER_FEATURES as SHAP_CONDENSER_FEATURES,
    SHAP_AVAILABLE,
)

from .lime_explainer import (
    CondenserLIMEExplainer,
    LIMEExplainerConfig,
    LIMEExplanation,
    LIMEFeatureWeight,
    DiscretizationMethod,
    LocalModelType,
    CONDENSER_FEATURES as LIME_CONDENSER_FEATURES,
    LIME_AVAILABLE,
)

from .physics_narrative import (
    PhysicsNarrativeGenerator,
    PhysicsExplanation,
    NarrativeReport,
    AudienceType,
    PhysicsMechanism,
    ImpactMetric,
    TrendDirection,
    PHYSICS_EQUATIONS,
    DRIVER_PHYSICS_MAP,
    NARRATIVE_TEMPLATES,
)


# Agent identification
AGENT_ID = "GL-017"
AGENT_NAME = "Condensync"
MODULE_VERSION = "1.0.0"


__all__ = [
    # Diagnostic Explainer
    "CondenserDiagnosticExplainer",
    "ExplainerConfig",
    "DiagnosticExplanation",
    "FeatureContribution",
    "CounterfactualExplanation",
    "ConstraintVisualization",
    "EvidenceChain",
    "ContributionDirection",
    "ExplanationStyle",
    "EvidenceStrength",
    "ConstraintType",
    "PhysicsParameter",
    "FEATURE_BASELINES",
    "PHYSICS_EXPLANATIONS",

    # SHAP Explainer
    "CondenserSHAPExplainer",
    "SHAPExplainerConfig",
    "SHAPExplanation",
    "FeatureImportance",
    "GlobalFeatureImportance",
    "InteractionEffect",
    "GlobalImportanceTrend",
    "ExplainerType",
    "ImportanceType",
    "SHAP_CONDENSER_FEATURES",
    "SHAP_AVAILABLE",

    # LIME Explainer
    "CondenserLIMEExplainer",
    "LIMEExplainerConfig",
    "LIMEExplanation",
    "LIMEFeatureWeight",
    "DiscretizationMethod",
    "LocalModelType",
    "LIME_CONDENSER_FEATURES",
    "LIME_AVAILABLE",

    # Physics Narrative
    "PhysicsNarrativeGenerator",
    "PhysicsExplanation",
    "NarrativeReport",
    "AudienceType",
    "PhysicsMechanism",
    "ImpactMetric",
    "TrendDirection",
    "PHYSICS_EQUATIONS",
    "DRIVER_PHYSICS_MAP",
    "NARRATIVE_TEMPLATES",

    # Module info
    "AGENT_ID",
    "AGENT_NAME",
    "MODULE_VERSION",
]
