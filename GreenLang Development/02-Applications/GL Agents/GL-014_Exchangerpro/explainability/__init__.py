# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Explainability Module

Provides comprehensive ML explainability for heat exchanger fouling analysis
with zero-hallucination guarantees. All numeric values are derived from
deterministic calculations (SHAP, LIME, causal inference), not LLM-generated
estimates.

This module ensures:
1. Transparency - All predictions are fully explainable
2. Stability - Explanations are consistent for similar operating points
3. Engineering Alignment - ML features map to physical concepts
4. Provenance - Complete audit trail with SHA-256 hashes
5. Zero-Hallucination - No LLM-generated numeric calculations

Components:
-----------
- FoulingSHAPExplainer: SHAP-based feature attribution
- FoulingLIMEExplainer: Local interpretable explanations with counterfactuals
- FoulingCausalAnalyzer: Causal inference and root cause analysis
- EngineeringRationaleGenerator: Operator-friendly engineering explanations
- ExplainabilityReportGenerator: PDF/HTML/JSON report generation

Example:
--------
    >>> from gl014_exchangerpro.explainability import (
    ...     FoulingSHAPExplainer,
    ...     ExplainabilityReportGenerator,
    ...     PredictionType,
    ... )
    >>>
    >>> # Generate SHAP explanation
    >>> shap_explainer = FoulingSHAPExplainer()
    >>> local_exp = shap_explainer.explain_prediction(
    ...     model=fouling_model,
    ...     features=input_features,
    ...     feature_names=feature_names,
    ...     prediction_type=PredictionType.FOULING_FACTOR,
    ...     exchanger_id="HX-001"
    ... )
    >>>
    >>> # Generate full report
    >>> report_gen = ExplainabilityReportGenerator()
    >>> report = report_gen.generate_report(
    ...     model=fouling_model,
    ...     features=input_features,
    ...     feature_names=feature_names,
    ...     exchanger_id="HX-001"
    ... )
    >>> html_report = report_gen.to_html(report)

Author: GreenLang AI Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GreenLang AI Team"

# Schemas
from .explanation_schemas import (
    # Enums
    ExplanationType,
    PredictionType,
    FoulingMechanism,
    ConfidenceLevel,
    FeatureCategory,
    # Core models
    FeatureImportance,
    FeatureContribution,
    ConfidenceBounds,
    UncertaintyEstimate,
    LocalExplanation,
    GlobalExplanation,
    CausalRelationship,
    RootCauseAnalysis,
    CounterfactualExplanation,
    EngineeringRationale,
    ExplanationStabilityMetrics,
    FoulingExplainabilityReport,
    DashboardExplanationData,
    # Type aliases
    FeatureContributions,
    FeatureImportanceList,
    CausalRelationships,
)

# SHAP Explainer
from .shap_explainer import (
    FoulingSHAPExplainer,
    SHAPConfig,
    SHAPResult,
    FOULING_FEATURE_CATEGORIES,
    # Convenience functions
    explain_fouling_prediction,
    get_global_feature_importance,
    verify_shap_consistency,
)

# LIME Explainer
from .lime_explainer import (
    FoulingLIMEExplainer,
    LIMEConfig,
    LIMEResult,
    # Convenience functions
    explain_fouling_with_lime,
    generate_counterfactual_explanation,
    validate_lime_explanation,
    compare_lime_explanations,
)

# Causal Analyzer
from .causal_analyzer import (
    FoulingCausalAnalyzer,
    FoulingCausalGraph,
    CausalAnalyzerConfig,
    CausalEdge,
    CausalEdgeType,
    CausalEffect,
    CounterfactualScenario,
    CausalAnalysisResult,
    # Convenience functions
    identify_fouling_root_causes,
    compute_intervention_effect,
    rank_root_causes,
)

# Engineering Rationale
from .engineering_rationale import (
    EngineeringRationaleGenerator,
    RationaleCategory,
    SeverityLevel,
    EngineeringObservation,
    ThermalPerformanceMetrics,
    HydraulicPerformanceMetrics,
    # Convenience functions
    generate_engineering_explanation,
    explain_feature_engineering,
)

# Report Generator
from .report_generator import (
    ExplainabilityReportGenerator,
    ReportConfig,
    ReportFormat,
    AudienceType,
    ReportSection,
    # Convenience functions
    generate_quick_report,
    export_report_json,
    export_report_html,
    export_report_markdown,
)


__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Enums
    "ExplanationType",
    "PredictionType",
    "FoulingMechanism",
    "ConfidenceLevel",
    "FeatureCategory",
    "CausalEdgeType",
    "RationaleCategory",
    "SeverityLevel",
    "ReportFormat",
    "AudienceType",

    # Schema models
    "FeatureImportance",
    "FeatureContribution",
    "ConfidenceBounds",
    "UncertaintyEstimate",
    "LocalExplanation",
    "GlobalExplanation",
    "CausalRelationship",
    "RootCauseAnalysis",
    "CounterfactualExplanation",
    "EngineeringRationale",
    "ExplanationStabilityMetrics",
    "FoulingExplainabilityReport",
    "DashboardExplanationData",

    # Type aliases
    "FeatureContributions",
    "FeatureImportanceList",
    "CausalRelationships",

    # SHAP
    "FoulingSHAPExplainer",
    "SHAPConfig",
    "SHAPResult",
    "FOULING_FEATURE_CATEGORIES",
    "explain_fouling_prediction",
    "get_global_feature_importance",
    "verify_shap_consistency",

    # LIME
    "FoulingLIMEExplainer",
    "LIMEConfig",
    "LIMEResult",
    "explain_fouling_with_lime",
    "generate_counterfactual_explanation",
    "validate_lime_explanation",
    "compare_lime_explanations",

    # Causal
    "FoulingCausalAnalyzer",
    "FoulingCausalGraph",
    "CausalAnalyzerConfig",
    "CausalEdge",
    "CausalEffect",
    "CounterfactualScenario",
    "CausalAnalysisResult",
    "identify_fouling_root_causes",
    "compute_intervention_effect",
    "rank_root_causes",

    # Engineering Rationale
    "EngineeringRationaleGenerator",
    "EngineeringObservation",
    "ThermalPerformanceMetrics",
    "HydraulicPerformanceMetrics",
    "generate_engineering_explanation",
    "explain_feature_engineering",

    # Report Generator
    "ExplainabilityReportGenerator",
    "ReportConfig",
    "ReportSection",
    "generate_quick_report",
    "export_report_json",
    "export_report_html",
    "export_report_markdown",
]
