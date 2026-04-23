"""
GL-012 STEAMQUAL - Explainability Module

Provides multi-level explanations for steam quality estimates and control
recommendations, ensuring all outputs are traceable to data and assumptions
per the playbook requirement.

This module implements:
- SHAP-based explanations for quality estimates (dryness fraction predictions)
- Root cause analysis for quality events (high drum level, separator flooding,
  PRV condensation, trap failure)
- Recommendation explanations with expected impact and alternatives
- Complete audit trail with input/output hashes, model versions, and timestamps

Standards Compliance:
    - ASME PTC 19.11 Steam Quality
    - IAPWS-IF97 Steam Tables
    - GreenLang Zero-Hallucination Principle

Zero-Hallucination Guarantee:
    All explanations are grounded in physics-based calculations and
    deterministic rules. No LLM inference is used for numerical results.
    All explanations traceable to source data and standard references.

Example:
    >>> from explainability import SHAPQualityExplainer, RootCauseAnalyzer
    >>> from explainability import RecommendationExplainer, ExplanationAuditTrail
    >>>
    >>> # Explain quality estimate
    >>> explainer = SHAPQualityExplainer(agent_id="GL-012")
    >>> explanation = explainer.explain_dryness_estimate(estimate)
    >>> print(explanation.summary)
    >>>
    >>> # Analyze root cause
    >>> analyzer = RootCauseAnalyzer(agent_id="GL-012")
    >>> analysis = analyzer.analyze_event(quality_event)
    >>> print(analysis.primary_cause)

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

# SHAP Quality Explainer exports
from .shap_quality_explainer import (
    # Enums
    QualityFeatureCategory,
    ModelType,
    ImpactDirection,
    # Data classes
    FeatureContribution,
    LocalExplanation,
    GlobalFeatureImportance,
    CounterfactualExplanation,
    # Main classes
    SHAPQualityExplainer,
    QualityExplanation,
    # Factory
    create_quality_explainer,
)

# Root Cause Analyzer exports
from .root_cause_analyzer import (
    # Enums
    RootCauseCategory,
    CausalChainType,
    EventSeverity,
    # Data classes
    CausalFactor,
    CausalTemplate,
    CorrelationResult,
    TimelineEvent,
    # Main classes
    RootCauseAnalyzer,
    RootCauseAnalysis,
    # Constants
    QUALITY_EVENT_TEMPLATES,
)

# Recommendation Explainer exports
from .recommendation_explainer import (
    # Enums
    ActionType,
    ActionPriority,
    ConfidenceLevel,
    # Data classes
    ExpectedImpact,
    AlternativeAction,
    ActionRationale,
    # Main classes
    RecommendationExplainer,
    RecommendationExplanation,
    ControlRecommendation,
)

# Audit Trail exports
from .audit_trail import (
    # Enums
    AuditEventType,
    # Data classes
    AuditEntry,
    ModelVersion,
    ConfigVersion,
    # Main classes
    ExplanationAuditTrail,
    AuditExporter,
    # Functions
    compute_provenance_hash,
)

# LIME Quality Explainer exports
from .lime_quality_explainer import (
    # Enums
    FeatureSelectionMethod,
    KernelType,
    ConsistencyStatus,
    ExplanationMode,
    # Models
    LIMEConfig,
    FeatureExplanation,
    LIMEExplanation,
    # Main class
    LIMEQualityExplainer,
    # Factory
    create_lime_explainer,
    # Constants
    QUALITY_FEATURE_METADATA,
)


# Module version
__version__ = "1.0.0"

# Explainability module metadata
__standards__ = [
    "ASME PTC 19.11",
    "IAPWS-IF97",
]


__all__ = [
    # Version
    "__version__",
    "__standards__",

    # SHAP Quality Explainer
    "QualityFeatureCategory",
    "ModelType",
    "ImpactDirection",
    "FeatureContribution",
    "LocalExplanation",
    "GlobalFeatureImportance",
    "CounterfactualExplanation",
    "SHAPQualityExplainer",
    "QualityExplanation",
    "create_quality_explainer",

    # Root Cause Analyzer
    "RootCauseCategory",
    "CausalChainType",
    "EventSeverity",
    "CausalFactor",
    "CausalTemplate",
    "CorrelationResult",
    "TimelineEvent",
    "RootCauseAnalyzer",
    "RootCauseAnalysis",
    "QUALITY_EVENT_TEMPLATES",

    # Recommendation Explainer
    "ActionType",
    "ActionPriority",
    "ConfidenceLevel",
    "ExpectedImpact",
    "AlternativeAction",
    "ActionRationale",
    "RecommendationExplainer",
    "RecommendationExplanation",
    "ControlRecommendation",

    # Audit Trail
    "AuditEventType",
    "AuditEntry",
    "ModelVersion",
    "ConfigVersion",
    "ExplanationAuditTrail",
    "AuditExporter",
    "compute_provenance_hash",

    # LIME Quality Explainer
    "FeatureSelectionMethod",
    "KernelType",
    "ConsistencyStatus",
    "ExplanationMode",
    "LIMEConfig",
    "FeatureExplanation",
    "LIMEExplanation",
    "LIMEQualityExplainer",
    "create_lime_explainer",
    "QUALITY_FEATURE_METADATA",
]


# Convenience aliases for backwards compatibility
QualityExplainer = SHAPQualityExplainer
