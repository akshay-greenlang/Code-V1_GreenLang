# -*- coding: utf-8 -*-
"""
GL-011 FuelCraft Explainability Module

Provides ML explainability for fuel price forecasting with zero-hallucination
guarantees. Aligned with GreenLang Global AI Standards v2.0.

Features:
- SHAP-based explanations (TreeExplainer, KernelExplainer)
- LIME-based local explanations
- Optimization binding constraint analysis
- Sensitivity and marginal cost analysis
- Template-driven report generation (ZERO free-form narrative)
- Structured artifact storage

Global AI Standards v2.0 Compliance:
- MANDATORY: SHAP TreeExplainer Integration (5 points)
- RECOMMENDED: LIME Explainer (3 points)
- REQUIRED: Engineering Rationale with Citations (4 points)
- REQUIRED: Decision Audit Trail (1 point)

Zero-Hallucination Architecture:
- All numeric values computed deterministically via SHAP/LIME
- Fixed random seeds for reproducibility
- Provenance hashing for complete audit trails
- LLM NEVER used for numeric calculations
- Template-driven reports with data citations only

Usage:
    from explainability import (
        SHAPForecastExplainer,
        LIMEForecastExplainer,
        OptimizationExplainer,
        ReportGenerator,
    )

    # Initialize SHAP explainer
    shap_explainer = SHAPForecastExplainer(
        model=price_model,
        feature_names=feature_names,
        business_labels=BUSINESS_LABELS,
    )

    # Generate explanation
    explanation = shap_explainer.explain(
        features=feature_vector,
        forecast=price_forecast,
    )

    # Generate template-driven report
    generator = ReportGenerator()
    report = generator.generate_forecast_report(
        explanation=explanation,
        bundle_hash=forecast_bundle.bundle_hash,
    )

Author: GreenLang AI Team
Version: 1.0.0
"""

from .shap_explainer import (
    # Data models
    SHAPExplanation,
    FeatureAttribution,
    InteractionEffect,
    SHAPConfig,
    # Explainer classes
    SHAPForecastExplainer,
    TreeExplainerWrapper,
    KernelExplainerWrapper,
    # Utility functions
    verify_shap_additivity,
    aggregate_attributions,
    compute_attribution_hash,
)

from .lime_explainer import (
    # Data models
    LIMEExplanation,
    LocalSurrogateModel,
    LIMEConfig,
    # Explainer classes
    LIMEForecastExplainer,
    TabularExplainerWrapper,
    # Utility functions
    validate_lime_fidelity,
    compare_shap_lime,
    compute_lime_hash,
)

from .optimization_explainer import (
    # Data models
    BindingConstraint,
    ShadowPrice,
    SensitivityResult,
    MarginalCostAnalysis,
    DecisionDriver,
    OptimizationExplanation,
    # Explainer classes
    OptimizationExplainer,
    BindingConstraintAnalyzer,
    SensitivityAnalyzer,
    # Utility functions
    extract_dual_values,
    compute_slack_analysis,
    identify_decision_drivers,
)

from .report_generator import (
    # Data models
    ReportTemplate,
    ReportSection,
    DataCitation,
    ExplainabilityAnnex,
    GeneratedReport,
    ReportConfig,
    # Generator classes
    ReportGenerator,
    TemplateRenderer,
    # Utility functions
    validate_report_citations,
    export_report,
    compute_report_hash,
)

from .lime_fuel_explainer import (
    # Enums
    FuelPropertyType,
    PerturbationType,
    # Data models
    ChemistryConstraint,
    FuelLocalSurrogateModel,
    FuelFeatureContribution,
    LIMEFuelExplanation,
    # Explainer classes
    LIMEFuelExplainer,
)

from .causal_fuel_analyzer import (
    # Enums
    CausalNodeType,
    CausalEdgeType,
    # Data models
    CausalNode,
    CausalEdge,
    CounterfactualResult,
    CausalExplanation,
    # DAG and Analyzer classes
    FuelCausalDAG,
    CausalFuelAnalyzer,
    # Utility functions
    validate_causal_explanation,
)

__version__ = "1.0.0"
__author__ = "GreenLang AI Team"

# Global AI Standards v2.0 Compliance Markers
GLOBAL_AI_STANDARDS_VERSION = "2.0.0"
EXPLAINABILITY_CATEGORY_MAX_POINTS = 15
IMPLEMENTED_CRITERIA = [
    "SHAP TreeExplainer Integration",
    "LIME Explainer",
    "Engineering Rationale with Citations",
    "Decision Audit Trail",
]

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "GLOBAL_AI_STANDARDS_VERSION",
    "EXPLAINABILITY_CATEGORY_MAX_POINTS",
    "IMPLEMENTED_CRITERIA",
    # SHAP
    "SHAPExplanation",
    "FeatureAttribution",
    "InteractionEffect",
    "SHAPConfig",
    "SHAPForecastExplainer",
    "TreeExplainerWrapper",
    "KernelExplainerWrapper",
    "verify_shap_additivity",
    "aggregate_attributions",
    "compute_attribution_hash",
    # LIME
    "LIMEExplanation",
    "LocalSurrogateModel",
    "LIMEConfig",
    "LIMEForecastExplainer",
    "TabularExplainerWrapper",
    "validate_lime_fidelity",
    "compare_shap_lime",
    "compute_lime_hash",
    # Optimization
    "BindingConstraint",
    "ShadowPrice",
    "SensitivityResult",
    "MarginalCostAnalysis",
    "DecisionDriver",
    "OptimizationExplanation",
    "OptimizationExplainer",
    "BindingConstraintAnalyzer",
    "SensitivityAnalyzer",
    "extract_dual_values",
    "compute_slack_analysis",
    "identify_decision_drivers",
    # Reports
    "ReportTemplate",
    "ReportSection",
    "DataCitation",
    "ExplainabilityAnnex",
    "GeneratedReport",
    "ReportConfig",
    "ReportGenerator",
    "TemplateRenderer",
    "validate_report_citations",
    "export_report",
    "compute_report_hash",
    # LIME Fuel Explainer
    "FuelPropertyType",
    "PerturbationType",
    "ChemistryConstraint",
    "FuelLocalSurrogateModel",
    "FuelFeatureContribution",
    "LIMEFuelExplanation",
    "LIMEFuelExplainer",
    # Causal Fuel Analyzer
    "CausalNodeType",
    "CausalEdgeType",
    "CausalNode",
    "CausalEdge",
    "CounterfactualResult",
    "CausalExplanation",
    "FuelCausalDAG",
    "CausalFuelAnalyzer",
    "validate_causal_explanation",
]
