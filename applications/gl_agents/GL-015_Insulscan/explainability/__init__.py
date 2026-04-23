# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Explainability Module

Provides comprehensive ML explainability for insulation scanning and thermal
assessment with zero-hallucination guarantees. All numeric values are derived
from deterministic calculations (SHAP, LIME, causal inference), not LLM-generated
estimates.

This module ensures:
1. Transparency - All predictions are fully explainable
2. Stability - Explanations are consistent for similar operating points
3. Engineering Alignment - ML features map to physical concepts
4. Provenance - Complete audit trail with SHA-256 hashes
5. Zero-Hallucination - No LLM-generated numeric calculations

Components:
-----------
- InsulationSHAPExplainer: SHAP-based feature attribution with batch support
- InsulationLIMEExplainer: Local interpretable explanations with counterfactuals
- InsulationCausalAnalyzer: Causal inference and root cause analysis
- InsulationReportGenerator: PDF/HTML/JSON report generation with ISO 50001

Causal Factors Analyzed:
- Age: Time-based degradation and material fatigue
- Moisture: Water ingress and vapor accumulation
- UV Exposure: Photodegradation of insulation materials
- Mechanical Damage: Physical impacts and compression
- Thermal Cycling: Expansion/contraction stress

Example:
--------
    >>> from gl015_insulscan.explainability import (
    ...     InsulationSHAPExplainer,
    ...     InsulationReportGenerator,
    ...     PredictionType,
    ... )
    >>>
    >>> # Generate SHAP explanation
    >>> shap_explainer = InsulationSHAPExplainer()
    >>> local_exp = shap_explainer.explain_prediction(
    ...     model=condition_model,
    ...     features=input_features,
    ...     feature_names=feature_names,
    ...     prediction_type=PredictionType.CONDITION_SCORE,
    ...     asset_id="INS-001"
    ... )
    >>>
    >>> # Generate full report with ISO 50001 compliance
    >>> report_gen = InsulationReportGenerator()
    >>> report = report_gen.generate_report(
    ...     model=condition_model,
    ...     features=input_features,
    ...     feature_names=feature_names,
    ...     asset_id="INS-001"
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
    DegradationMechanism,
    InsulationType,
    ConfidenceLevel,
    FeatureCategory,
    # Core models
    FeatureImportance,
    FeatureContribution,
    ConfidenceBounds,
    UncertaintyEstimate,
    LocalExplanation,
    GlobalExplanation,
    CausalFactor,
    CausalRelationship,
    RootCauseAnalysis,
    CounterfactualExplanation,
    ThermalImageData,
    HeatLossDiagram,
    RepairRecommendation,
    InsulationExplanation,
    ExplanationStabilityMetrics,
    ISO50001ComplianceData,
    InsulationExplainabilityReport,
    DashboardExplanationData,
    # Type aliases
    FeatureContributions,
    FeatureImportanceList,
    CausalRelationships,
    CausalFactors,
    RepairRecommendations,
)

# SHAP Explainer
from .shap_explainer import (
    InsulationSHAPExplainer,
    SHAPConfig,
    SHAPResult,
    INSULATION_FEATURE_CATEGORIES,
    # Convenience functions
    explain_insulation_prediction,
    get_global_insulation_importance,
    verify_shap_consistency,
)

# LIME Explainer
from .lime_explainer import (
    InsulationLIMEExplainer,
    LIMEConfig,
    LIMEResult,
    SeedManager,
    # Convenience functions
    explain_insulation_with_lime,
    generate_insulation_counterfactual,
    validate_lime_explanation,
    compare_lime_explanations,
)

# Causal Analyzer
from .causal_analyzer import (
    InsulationCausalAnalyzer,
    InsulationCausalGraph,
    CausalAnalyzerConfig,
    CausalEdge,
    CausalEdgeType,
    CausalEffect,
    CounterfactualScenario,
    CausalAnalysisResult,
    # Convenience functions
    identify_insulation_root_causes,
    compute_insulation_intervention_effect,
    rank_insulation_root_causes,
    analyze_what_if_thicker_insulation,
)

# Report Generator
from .report_generator import (
    InsulationReportGenerator,
    ReportConfig,
    ReportFormat,
    AudienceType,
    ReportSection,
    # Convenience functions
    generate_quick_insulation_report,
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
    "DegradationMechanism",
    "InsulationType",
    "ConfidenceLevel",
    "FeatureCategory",
    "CausalEdgeType",
    "ReportFormat",
    "AudienceType",

    # Schema models
    "FeatureImportance",
    "FeatureContribution",
    "ConfidenceBounds",
    "UncertaintyEstimate",
    "LocalExplanation",
    "GlobalExplanation",
    "CausalFactor",
    "CausalRelationship",
    "RootCauseAnalysis",
    "CounterfactualExplanation",
    "ThermalImageData",
    "HeatLossDiagram",
    "RepairRecommendation",
    "InsulationExplanation",
    "ExplanationStabilityMetrics",
    "ISO50001ComplianceData",
    "InsulationExplainabilityReport",
    "DashboardExplanationData",

    # Type aliases
    "FeatureContributions",
    "FeatureImportanceList",
    "CausalRelationships",
    "CausalFactors",
    "RepairRecommendations",

    # SHAP
    "InsulationSHAPExplainer",
    "SHAPConfig",
    "SHAPResult",
    "INSULATION_FEATURE_CATEGORIES",
    "explain_insulation_prediction",
    "get_global_insulation_importance",
    "verify_shap_consistency",

    # LIME
    "InsulationLIMEExplainer",
    "LIMEConfig",
    "LIMEResult",
    "SeedManager",
    "explain_insulation_with_lime",
    "generate_insulation_counterfactual",
    "validate_lime_explanation",
    "compare_lime_explanations",

    # Causal
    "InsulationCausalAnalyzer",
    "InsulationCausalGraph",
    "CausalAnalyzerConfig",
    "CausalEdge",
    "CausalEffect",
    "CounterfactualScenario",
    "CausalAnalysisResult",
    "identify_insulation_root_causes",
    "compute_insulation_intervention_effect",
    "rank_insulation_root_causes",
    "analyze_what_if_thicker_insulation",

    # Report Generator
    "InsulationReportGenerator",
    "ReportConfig",
    "ReportSection",
    "generate_quick_insulation_report",
    "export_report_json",
    "export_report_html",
    "export_report_markdown",
]
