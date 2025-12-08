# -*- coding: utf-8 -*-
"""
GreenLang AI/ML Explainability Framework
=========================================

Provides SHAP, LIME, and causal inference for zero-hallucination AI agents.
Target: Raise AI/ML score from 73.5 to 95+/100

This package provides comprehensive explainability capabilities for GreenLang
ML models, including:

- SHAP: SHapley Additive exPlanations for global/local feature attribution
- LIME: Local Interpretable Model-agnostic Explanations for local interpretability
- Causal Inference: DoWhy-based causal effect estimation and counterfactuals
- Explanation Generator: Human-readable narrative generation for stakeholders
- ExplainabilityLayer: Unified interface for all explainability methods

Modules:
    core: Unified explainability framework with all components
    shap_explainer: Standalone SHAP implementation
    lime_explainer: Standalone LIME implementation
    causal_inference: DoWhy-based causal inference
    explanation_generator: Human-readable explanation generation

Example:
    >>> from greenlang.ml.explainability import ExplainabilityLayer, ExplanationType
    >>> layer = ExplainabilityLayer(model, training_data=X_train)
    >>> result = layer.explain(X_test, method=ExplanationType.SHAP)
    >>> print(result.human_readable)
    >>> print(f"Provenance: {result.provenance_hash}")

Example (Async):
    >>> result = await layer.explain_async(X_test, method="lime")

Example (Agent Integration):
    >>> class MyAgent(BaseAgent, ExplainableAgentMixin):
    ...     def __init__(self, model):
    ...         super().__init__()
    ...         self.setup_explainability(model)
    ...
    >>> agent = MyAgent(trained_model)
    >>> explanation = agent.explain_prediction(X_sample)

Author: GreenLang Team
Version: 1.0.0
"""

# Import from unified core module
from greenlang.ml.explainability.core import (
    # Enums
    ExplanationType,
    ExplainerType,
    AudienceLevel,
    ConfidenceLevel,
    # Dataclasses
    FeatureContribution,
    ExplanationResult,
    CausalEffectResult,
    CounterfactualResult,
    # Configs
    ExplainerConfig,
    SHAPExplainerConfig,
    LIMEExplainerConfig,
    CausalExplainerConfig,
    ExplanationGeneratorConfig,
    # Base
    BaseExplainer,
    # Unified Explainers (from core)
    SHAPExplainer as UnifiedSHAPExplainer,
    LIMEExplainer as UnifiedLIMEExplainer,
    CausalExplainer,
    # Generator
    ExplanationGenerator as UnifiedExplanationGenerator,
    # Unified Layer
    ExplainabilityLayer,
    # Mixin
    ExplainableAgentMixin,
    # Factory
    create_explainability_layer,
)

# Import from standalone modules for backward compatibility
from greenlang.ml.explainability.shap_explainer import (
    SHAPExplainer,
    SHAPExplainerConfig as StandaloneSHAPConfig,
    SHAPResult,
    ExplainerType as SHAPExplainerType,
)
from greenlang.ml.explainability.lime_explainer import (
    LIMEExplainer,
    LIMEExplainerConfig as StandaloneLIMEConfig,
    LIMEResult,
    LIMEBatchResult,
    LIMEMode,
    KernelType,
)
from greenlang.ml.explainability.causal_inference import (
    CausalInference,
    CausalInferenceConfig,
    CausalEffectResult as StandaloneCausalEffectResult,
    CounterfactualResult as StandaloneCounterfactualResult,
    IdentificationMethod,
    EstimationMethod,
    RefutationMethod,
)
from greenlang.ml.explainability.explanation_generator import (
    ExplanationGenerator,
    ExplanationGeneratorConfig as StandaloneGeneratorConfig,
    Explanation,
    FeatureExplanation,
    AudienceLevel as StandaloneAudienceLevel,
    ExplanationType as StandaloneExplanationType,
)

# Import from Natural Language Explainer module
from greenlang.ml.explainability.natural_language_explainer import (
    # Main class
    NaturalLanguageExplainer,
    # Enums
    Audience,
    OutputFormat,
    DecisionType,
    # Data model
    ExplanationOutput,
    # Factory
    create_natural_language_explainer,
)

# Import from Process Heat SHAP integration module
from greenlang.ml.explainability.process_heat_shap import (
    # Enums
    ProcessHeatAgentType,
    SHAPExplainerMode,
    ConfidenceLevel as ProcessHeatConfidenceLevel,
    IndustrialDomain,
    # Base config
    ProcessHeatSHAPConfigBase,
    # Agent-specific configs
    GL001ThermalCommandSHAPConfig,
    GL003UnifiedSteamSHAPConfig,
    GL006HeatReclaimSHAPConfig,
    GL010EmissionsGuardianSHAPConfig,
    GL013PredictMaintSHAPConfig,
    GL018UnifiedCombustionSHAPConfig,
    # Results
    ProcessHeatSHAPResult,
    ProcessHeatBatchSHAPResult,
    # Cache
    SHAPCache,
    # Main explainer
    ProcessHeatSHAPExplainer,
    # Factory
    create_process_heat_explainer,
    # Feature mappings
    INDUSTRIAL_FEATURE_NAMES,
)

# Import from Attention Visualizer module
from greenlang.ml.explainability.attention_visualizer import (
    # Main class
    AttentionVisualizer,
    # Data models
    AttentionWeights,
    AttentionSummary,
    # Enums
    VisualizationType,
    ExportFormat,
)

# Import from Dashboard module (TASK-030)
from greenlang.ml.explainability.dashboard import (
    # Enums
    ChartType,
    ExportFormat as DashboardExportFormat,
    TimeRange,
    DashboardViewMode,
    # Data Models
    FeatureContributionData,
    FeatureContributionChart,
    GlobalImportanceChart,
    CounterfactualChange,
    CounterfactualComparisonView,
    PredictionHistoryEntry,
    PredictionHistoryView,
    ModelSummary,
    ExplanationDashboardData,
    DashboardSummary,
    # Visualization Generator
    VisualizationDataGenerator,
    # State Management
    DashboardStateManager,
    dashboard_state,
    get_dashboard_state,
    # Router
    dashboard_router,
)

__version__ = "1.0.0"

__all__ = [
    # === Unified Framework (core.py) ===
    # Enums
    "ExplanationType",
    "ExplainerType",
    "AudienceLevel",
    "ConfidenceLevel",
    # Dataclasses
    "FeatureContribution",
    "ExplanationResult",
    "CausalEffectResult",
    "CounterfactualResult",
    # Configs
    "ExplainerConfig",
    "SHAPExplainerConfig",
    "LIMEExplainerConfig",
    "CausalExplainerConfig",
    "ExplanationGeneratorConfig",
    # Base
    "BaseExplainer",
    # Unified Explainers
    "UnifiedSHAPExplainer",
    "UnifiedLIMEExplainer",
    "CausalExplainer",
    # Generator
    "UnifiedExplanationGenerator",
    # Unified Layer (MAIN INTERFACE)
    "ExplainabilityLayer",
    # Mixin for Agent Integration
    "ExplainableAgentMixin",
    # Factory
    "create_explainability_layer",
    # === Standalone Modules (backward compatibility) ===
    # SHAP
    "SHAPExplainer",
    "StandaloneSHAPConfig",
    "SHAPResult",
    "SHAPExplainerType",
    # LIME
    "LIMEExplainer",
    "StandaloneLIMEConfig",
    "LIMEResult",
    "LIMEBatchResult",
    "LIMEMode",
    "KernelType",
    # Causal Inference
    "CausalInference",
    "CausalInferenceConfig",
    "StandaloneCausalEffectResult",
    "StandaloneCounterfactualResult",
    "IdentificationMethod",
    "EstimationMethod",
    "RefutationMethod",
    # Explanation Generator
    "ExplanationGenerator",
    "StandaloneGeneratorConfig",
    "Explanation",
    "FeatureExplanation",
    "StandaloneAudienceLevel",
    "StandaloneExplanationType",
    # === Natural Language Explainer (natural_language_explainer.py) ===
    "NaturalLanguageExplainer",
    "Audience",
    "OutputFormat",
    "DecisionType",
    "ExplanationOutput",
    "create_natural_language_explainer",
    # === Process Heat SHAP Integration (process_heat_shap.py) ===
    # Enums
    "ProcessHeatAgentType",
    "SHAPExplainerMode",
    "ProcessHeatConfidenceLevel",
    "IndustrialDomain",
    # Base config
    "ProcessHeatSHAPConfigBase",
    # Agent-specific configs
    "GL001ThermalCommandSHAPConfig",
    "GL003UnifiedSteamSHAPConfig",
    "GL006HeatReclaimSHAPConfig",
    "GL010EmissionsGuardianSHAPConfig",
    "GL013PredictMaintSHAPConfig",
    "GL018UnifiedCombustionSHAPConfig",
    # Results
    "ProcessHeatSHAPResult",
    "ProcessHeatBatchSHAPResult",
    # Cache
    "SHAPCache",
    # Main explainer
    "ProcessHeatSHAPExplainer",
    # Factory
    "create_process_heat_explainer",
    # Feature mappings
    "INDUSTRIAL_FEATURE_NAMES",
    # === Attention Visualizer (attention_visualizer.py) ===
    "AttentionVisualizer",
    "AttentionWeights",
    "AttentionSummary",
    "VisualizationType",
    "ExportFormat",
    # === Dashboard (dashboard.py) TASK-030 ===
    # Enums
    "ChartType",
    "DashboardExportFormat",
    "TimeRange",
    "DashboardViewMode",
    # Data Models
    "FeatureContributionData",
    "FeatureContributionChart",
    "GlobalImportanceChart",
    "CounterfactualChange",
    "CounterfactualComparisonView",
    "PredictionHistoryEntry",
    "PredictionHistoryView",
    "ModelSummary",
    "ExplanationDashboardData",
    "DashboardSummary",
    # Visualization Generator
    "VisualizationDataGenerator",
    # State Management
    "DashboardStateManager",
    "dashboard_state",
    "get_dashboard_state",
    # Router
    "dashboard_router",
]