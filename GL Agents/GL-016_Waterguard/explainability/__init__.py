"""
GL-016 Waterguard Explainability Module

This module provides SHAP and LIME-based explainability for Waterguard's
ML-driven water treatment recommendations. All explanations are derived
from structured data using deterministic methods - NO generative AI.

Key Components:
- WaterguardSHAPExplainer: SHAP-based feature importance and local explanations
- WaterguardLIMEExplainer: LIME-based local interpretable explanations
- ExplainabilityService: Unified orchestration of explanation methods
- NarrativeGenerator: Template-based human-readable explanation generation
- ExplainabilityReportGenerator: Daily reports and trend analysis
- ExplanationDriftDetector: Detection of explanation drift and instability

Usage:
    from explainability import ExplainabilityService

    service = ExplainabilityService(model, config)
    explanation = service.generate_recommendation_explanation(
        recommendation=recommendation,
        model=model,
        inputs=input_data
    )
"""

from .explanation_schemas import (
    FeatureContribution,
    LocalExplanation,
    GlobalExplanation,
    ExplanationStabilityMetrics,
    ExplanationPayload,
    ExplanationMethod,
    FeatureDirection,
    RecommendationType,
)

from .shap_explainer import (
    WaterguardSHAPExplainer,
    SHAPExplanation,
    SHAPSummaryStatistics,
)

from .lime_explainer import (
    WaterguardLIMEExplainer,
    LIMEExplanation,
)

from .explainability_service import (
    ExplainabilityService,
    ExplainabilityConfig,
)

from .narrative_generator import (
    NarrativeGenerator,
    NarrativeConfig,
    NarrativeTemplate,
)

from .report_generator import (
    ExplainabilityReportGenerator,
    DailyExplainabilityReport,
    FeatureImportanceTrend,
    ExplanationConsistencyMetrics,
)

from .drift_detector import (
    ExplanationDriftDetector,
    DriftDetectionResult,
    DriftAlert,
    DriftSeverity,
)

__version__ = "1.0.0"
__author__ = "GreenLang GL-016 Waterguard Team"

__all__ = [
    # Schemas
    "FeatureContribution",
    "LocalExplanation",
    "GlobalExplanation",
    "ExplanationStabilityMetrics",
    "ExplanationPayload",
    "ExplanationMethod",
    "FeatureDirection",
    "RecommendationType",
    # SHAP
    "WaterguardSHAPExplainer",
    "SHAPExplanation",
    "SHAPSummaryStatistics",
    # LIME
    "WaterguardLIMEExplainer",
    "LIMEExplanation",
    # Service
    "ExplainabilityService",
    "ExplainabilityConfig",
    # Narrative
    "NarrativeGenerator",
    "NarrativeConfig",
    "NarrativeTemplate",
    # Reports
    "ExplainabilityReportGenerator",
    "DailyExplainabilityReport",
    "FeatureImportanceTrend",
    "ExplanationConsistencyMetrics",
    # Drift Detection
    "ExplanationDriftDetector",
    "DriftDetectionResult",
    "DriftAlert",
    "DriftSeverity",
]
