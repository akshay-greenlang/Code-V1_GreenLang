# -*- coding: utf-8 -*-
"""
GL-013 PredictiveMaintenance Explainability Module

Provides ML explainability with zero-hallucination guarantees for predictive maintenance.

Author: GreenLang AI Team
Version: 1.0.0
"""

from .explanation_schemas import (
    ExplanationType, PredictionType, ModalityType, ConfidenceLevel,
    FeatureContribution, ConfidenceBounds, UncertaintyRange,
    AttentionWeight, TemporalSaliencyMap, CausalEdge, RootCauseHypothesis,
    SHAPExplanation, LIMEExplanation, AttentionExplanation, CausalExplanation,
    MaintenanceExplanationReport, DashboardExplanationData,
    FeatureContributions, RootCauseHypotheses, AttentionWeights,
)

from .shap_explainer import (
    SHAPExplainer, TreeSHAPExplainer, KernelSHAPExplainer, SHAPConfig,
    verify_shap_consistency, aggregate_shap_explanations, get_top_drivers,
)

from .lime_explainer import (
    LIMEExplainer, TabularLIMEExplainer, LIMEConfig,
    aggregate_lime_explanations, compare_lime_explanations, validate_lime_explanation,
)

from .attention_visualizer import (
    AttentionVisualizer, AttentionVisualizerConfig,
    extract_attention_weights, compute_temporal_saliency, get_cross_modal_attention,
)

from .causal_analyzer import (
    CausalAnalyzer, CausalAnalyzerConfig, CausalGraph,
    identify_confounders, compute_backdoor_adjustment, rank_root_causes,
)

from .report_generator import (
    ReportGenerator, ReportConfig,
    generate_quick_report, export_report_json, export_report_html,
)

__version__ = "1.0.0"
__author__ = "GreenLang AI Team"

__all__ = [
    "__version__", "__author__",
    "ExplanationType", "PredictionType", "ModalityType", "ConfidenceLevel",
    "FeatureContribution", "ConfidenceBounds", "UncertaintyRange",
    "AttentionWeight", "TemporalSaliencyMap", "CausalEdge", "RootCauseHypothesis",
    "SHAPExplanation", "LIMEExplanation", "AttentionExplanation", "CausalExplanation",
    "MaintenanceExplanationReport", "DashboardExplanationData",
    "FeatureContributions", "RootCauseHypotheses", "AttentionWeights",
    "SHAPExplainer", "TreeSHAPExplainer", "KernelSHAPExplainer", "SHAPConfig",
    "verify_shap_consistency", "aggregate_shap_explanations", "get_top_drivers",
    "LIMEExplainer", "TabularLIMEExplainer", "LIMEConfig",
    "aggregate_lime_explanations", "compare_lime_explanations", "validate_lime_explanation",
    "AttentionVisualizer", "AttentionVisualizerConfig",
    "extract_attention_weights", "compute_temporal_saliency", "get_cross_modal_attention",
    "CausalAnalyzer", "CausalAnalyzerConfig", "CausalGraph",
    "identify_confounders", "compute_backdoor_adjustment", "rank_root_causes",
    "ReportGenerator", "ReportConfig",
    "generate_quick_report", "export_report_json", "export_report_html",
]
