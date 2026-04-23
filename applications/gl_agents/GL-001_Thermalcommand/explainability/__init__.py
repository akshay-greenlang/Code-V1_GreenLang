# -*- coding: utf-8 -*-
"""
GL-001 ThermalCommand Explainability Module

Provides ML explainability with zero-hallucination guarantees for:
- SHAP-based explanations (TreeExplainer, KernelExplainer)
- LIME-based explanations (LimeTabularExplainer)
- Optimization decision explanations (binding constraints, shadow prices)
- Counterfactual generation
- Report generation for dashboards and audits

Zero-Hallucination Guarantees:
- All numeric values are computed deterministically via SHAP/LIME algorithms
- Fixed random seeds ensure reproducibility
- Provenance hashing for complete audit trails
- LLM is NEVER used for numeric calculations
- 80%+ confidence threshold for all outputs

Usage:
    from explainability import ExplainabilityService, SHAPExplainer, LIMEExplainer
    from explainability import ReportGenerator, PredictionType

    # Initialize service
    service = ExplainabilityService(
        training_data=X_train,
        feature_names=feature_names
    )
    service.set_model(model, model_type="tree")

    # Explain demand forecast
    report = service.explain_demand_forecast(
        forecast_input=X_test[0],
        predict_fn=model.predict
    )

    # Generate dashboard data
    generator = ReportGenerator()
    dashboard_data = generator.generate_dashboard_data(report)

Author: GreenLang AI Team
Version: 1.0.0
"""

# Schemas and data models
from .explanation_schemas import (
    # Enums
    ExplanationType,
    PredictionType,
    ConfidenceLevel,
    # Core data structures
    FeatureContribution,
    ConfidenceBounds,
    UncertaintyRange,
    Counterfactual,
    # Explanation types
    SHAPExplanation,
    LIMEExplanation,
    DecisionExplanation,
    # Reports
    ExplanationReport,
    BatchExplanationSummary,
    DashboardExplanationData,
    # Type aliases
    FeatureContributions,
    CounterfactualList,
)

# SHAP explainer
from .shap_explainer import (
    SHAPExplainer,
    TreeSHAPExplainer,
    KernelSHAPExplainer,
    SHAPConfig,
    verify_shap_consistency,
    aggregate_shap_explanations,
)

# LIME explainer
from .lime_explainer import (
    LIMEExplainer,
    TabularLIMEExplainer,
    LIMEConfig,
    aggregate_lime_explanations,
    compare_lime_explanations,
    validate_lime_explanation,
)

# Explainability service
from .explainability_service import (
    ExplainabilityService,
    ExplanationMethod,
    OptimizationContext,
    ServiceConfig,
)

# Report generator
from .report_generator import (
    ReportGenerator,
    ReportConfig,
    ReportMetadata,
    generate_quick_report,
    export_report_json,
)

__version__ = "1.0.0"
__author__ = "GreenLang AI Team"

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Enums
    "ExplanationType",
    "PredictionType",
    "ConfidenceLevel",
    "ExplanationMethod",
    # Data structures
    "FeatureContribution",
    "ConfidenceBounds",
    "UncertaintyRange",
    "Counterfactual",
    # Explanation types
    "SHAPExplanation",
    "LIMEExplanation",
    "DecisionExplanation",
    # Reports
    "ExplanationReport",
    "BatchExplanationSummary",
    "DashboardExplanationData",
    # Type aliases
    "FeatureContributions",
    "CounterfactualList",
    # SHAP
    "SHAPExplainer",
    "TreeSHAPExplainer",
    "KernelSHAPExplainer",
    "SHAPConfig",
    "verify_shap_consistency",
    "aggregate_shap_explanations",
    # LIME
    "LIMEExplainer",
    "TabularLIMEExplainer",
    "LIMEConfig",
    "aggregate_lime_explanations",
    "compare_lime_explanations",
    "validate_lime_explanation",
    # Service
    "ExplainabilityService",
    "OptimizationContext",
    "ServiceConfig",
    # Reports
    "ReportGenerator",
    "ReportConfig",
    "ReportMetadata",
    "generate_quick_report",
    "export_report_json",
]
