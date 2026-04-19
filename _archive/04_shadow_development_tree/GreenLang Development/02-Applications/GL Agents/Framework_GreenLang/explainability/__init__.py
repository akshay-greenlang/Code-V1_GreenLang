"""
GreenLang Framework - Explainability Module

Comprehensive explainability toolkit for GreenLang agents providing:
- SHAP explanations for tree-based and general ML models
- LIME local interpretable explanations
- Engineering rationale with standard citations
- Causal analysis with DAGs and counterfactuals

All explainability modules follow zero-hallucination principles:
- Numeric values are computed deterministically
- Citations reference authoritative standards
- Provenance hashes enable full auditability

Modules:
- shap_explainer: SHAP-based ML model explanations
- lime_explainer: LIME-based local interpretable explanations
- engineering_rationale: Rule-based engineering explanations with citations
- causal_analysis: Causal inference and root cause analysis
- explanation_schemas: Data models for all explanation types

Example Usage:

    # SHAP Explanations
    from greenlang_framework.explainability import (
        SHAPExplainerService, SHAPConfig, PredictionType
    )

    config = SHAPConfig(random_seed=42, num_samples=100)
    explainer = SHAPExplainerService(config, feature_names=["temp", "pressure"])
    explainer.fit_tree_explainer(xgb_model)
    explanation = explainer.explain_instance(instance, PredictionType.REGRESSION)

    # LIME Explanations
    from greenlang_framework.explainability import (
        LIMEExplainerService, LIMEConfig
    )

    lime_config = LIMEConfig(random_seed=42, num_samples=5000)
    lime_explainer = LIMEExplainerService(X_train, feature_names, lime_config)
    lime_exp = lime_explainer.explain_instance(instance, model.predict, PredictionType.REGRESSION)

    # Engineering Rationale
    from greenlang_framework.explainability import (
        EngineeringRationaleGenerator, CalculationType
    )

    generator = EngineeringRationaleGenerator()
    rationale = generator.generate_rationale(
        CalculationType.COMBUSTION,
        inputs={"fuel_flow": 100},
        outputs={"efficiency": 0.85}
    )

    # Causal Analysis
    from greenlang_framework.explainability import CausalAnalysisService

    service = CausalAnalysisService()
    service.add_variable("temperature", value=350)
    service.add_variable("efficiency", value=0.85)
    service.add_causal_relationship("temperature", "efficiency", effect_size=0.1)
    result = service.analyze(outcome_variable="efficiency")

Version: 1.0.0
Author: GreenLang AI Team
"""

__version__ = "1.0.0"
__author__ = "GreenLang AI Team"

# Explanation schemas
from .explanation_schemas import (
    # Enums
    PredictionType,
    ExplainerType,
    StandardSource,
    # Data classes
    ConfidenceBounds,
    UncertaintyRange,
    FeatureContribution,
    InteractionEffect,
    SHAPExplanation,
    LIMEExplanation,
    StandardCitation,
    ThermodynamicPrinciple,
    EngineeringRationale,
    CausalNode,
    CausalEdge,
    CounterfactualExplanation,
    RootCauseAnalysis,
    CausalAnalysisResult,
    DashboardExplanationData,
)

# SHAP explainer
from .shap_explainer import (
    SHAPConfig,
    SHAPExplainerService,
    TreeSHAPExplainer,
    KernelSHAPExplainer,
    verify_shap_consistency,
    aggregate_shap_explanations,
    compare_explanations as compare_shap_explanations,
    SHAP_AVAILABLE,
)

# LIME explainer
from .lime_explainer import (
    LIMEConfig,
    LIMEExplainerService,
    TabularLIMEExplainer,
    ClassificationLIMEExplainer,
    aggregate_lime_explanations,
    compare_lime_explanations,
    compute_explanation_stability,
    LIME_AVAILABLE,
)

# Engineering rationale
from .engineering_rationale import (
    CalculationType,
    EngineeringRationaleGenerator,
    RationaleConfig,
    STANDARD_CITATIONS,
    THERMODYNAMIC_PRINCIPLES,
    get_all_standard_sources,
    get_citations_by_source,
    format_principle_as_markdown,
    format_rationale_as_markdown,
)

# Causal analysis
from .causal_analysis import (
    NodeType,
    EdgeType,
    DeviationType,
    CausalGraphConfig,
    CausalGraph,
    CausalAnalysisService,
    InterventionRecommendation,
    build_thermal_system_dag,
    format_causal_analysis_report,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Enums
    "PredictionType",
    "ExplainerType",
    "StandardSource",
    "CalculationType",
    "NodeType",
    "EdgeType",
    "DeviationType",

    # Configuration classes
    "SHAPConfig",
    "LIMEConfig",
    "RationaleConfig",
    "CausalGraphConfig",

    # Data schemas
    "ConfidenceBounds",
    "UncertaintyRange",
    "FeatureContribution",
    "InteractionEffect",
    "SHAPExplanation",
    "LIMEExplanation",
    "StandardCitation",
    "ThermodynamicPrinciple",
    "EngineeringRationale",
    "CausalNode",
    "CausalEdge",
    "CounterfactualExplanation",
    "RootCauseAnalysis",
    "CausalAnalysisResult",
    "DashboardExplanationData",
    "InterventionRecommendation",

    # SHAP explainer
    "SHAPExplainerService",
    "TreeSHAPExplainer",
    "KernelSHAPExplainer",
    "verify_shap_consistency",
    "aggregate_shap_explanations",
    "compare_shap_explanations",
    "SHAP_AVAILABLE",

    # LIME explainer
    "LIMEExplainerService",
    "TabularLIMEExplainer",
    "ClassificationLIMEExplainer",
    "aggregate_lime_explanations",
    "compare_lime_explanations",
    "compute_explanation_stability",
    "LIME_AVAILABLE",

    # Engineering rationale
    "EngineeringRationaleGenerator",
    "STANDARD_CITATIONS",
    "THERMODYNAMIC_PRINCIPLES",
    "get_all_standard_sources",
    "get_citations_by_source",
    "format_principle_as_markdown",
    "format_rationale_as_markdown",

    # Causal analysis
    "CausalGraph",
    "CausalAnalysisService",
    "build_thermal_system_dag",
    "format_causal_analysis_report",
]
