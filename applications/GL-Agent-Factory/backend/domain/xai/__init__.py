# -*- coding: utf-8 -*-
"""
GreenLang Explainability (XAI) Module
=====================================

Provides explainability and uncertainty quantification for AI agents
to ensure transparency and trust in regulatory compliance applications.

Components:
- explanation_engine: Core explanation generation
- uncertainty: Uncertainty quantification and propagation
- feature_importance: SHAP/LIME-style feature attribution
- decision_trace: Decision audit trail
- confidence: Confidence scoring and calibration
"""

from .explanation_engine import (
    ExplanationEngine,
    Explanation,
    ExplanationType,
    ExplanationLevel,
    get_explanation_engine,
)
from .uncertainty import (
    UncertaintyQuantifier,
    UncertaintyType,
    UncertaintyBound,
    PropagatedUncertainty,
)
from .feature_importance import (
    FeatureImportance,
    FeatureAttribution,
    SHAPExplainer,
    calculate_feature_importance,
)
from .decision_trace import (
    DecisionTrace,
    DecisionStep,
    DecisionNode,
    trace_decision,
)
from .confidence import (
    ConfidenceScore,
    ConfidenceLevel,
    CalibrationCurve,
    calculate_confidence,
)

__all__ = [
    # Explanation Engine
    "ExplanationEngine",
    "Explanation",
    "ExplanationType",
    "ExplanationLevel",
    "get_explanation_engine",
    # Uncertainty
    "UncertaintyQuantifier",
    "UncertaintyType",
    "UncertaintyBound",
    "PropagatedUncertainty",
    # Feature Importance
    "FeatureImportance",
    "FeatureAttribution",
    "SHAPExplainer",
    "calculate_feature_importance",
    # Decision Trace
    "DecisionTrace",
    "DecisionStep",
    "DecisionNode",
    "trace_decision",
    # Confidence
    "ConfidenceScore",
    "ConfidenceLevel",
    "CalibrationCurve",
    "calculate_confidence",
]

__version__ = "1.0.0"
