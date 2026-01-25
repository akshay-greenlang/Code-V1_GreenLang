"""
Causal Module for GL-004 BURNMASTER

This module implements causal inference for root cause analysis of combustion issues.
It provides tools for building causal graphs, analyzing root causes, computing
counterfactuals, recommending interventions, and analyzing anomalies and trips.

Components:
    - CombustionCausalGraph: Build and validate causal graphs for combustion systems
    - RootCauseAnalyzer: Analyze deviations and trace causal paths
    - CounterfactualEngine: Compute counterfactual scenarios ("what if" analysis)
    - InterventionRecommender: Recommend and validate interventions
    - CausalAnomalyAnalyzer: Detect and analyze anomaly sources
    - TripCausalAnalyzer: Analyze trip sequences and identify initiators

Example:
    >>> from causal import CombustionCausalGraph, RootCauseAnalyzer
    >>> graph_builder = CombustionCausalGraph()
    >>> graph = graph_builder.build_graph(variables, domain_knowledge)
    >>> analyzer = RootCauseAnalyzer(graph)
    >>> analysis = analyzer.analyze_deviation("efficiency", observed=0.85, expected=0.92)
"""

from causal.causal_graph import (
    CombustionCausalGraph,
    CausalGraph,
    CausalNode,
    CausalEdge,
    GraphValidationResult,
)

from causal.root_cause_analyzer import (
    RootCauseAnalyzer,
    RootCauseAnalysis,
    RankedCause,
    CausalPath,
    RCAReport,
)

from causal.counterfactual_engine import (
    CounterfactualEngine,
    CounterfactualQuery,
    CounterfactualResult,
    Intervention,
    Effect,
    ScenarioComparison,
)

from causal.intervention_recommender import (
    InterventionRecommender,
    InterventionRecommendation,
    ImpactEstimate,
    RankedIntervention,
    SafetyValidation,
)

from causal.anomaly_analyzer import (
    CausalAnomalyAnalyzer,
    Anomaly,
    AnomalySource,
    PropagationPath,
    IsolationResult,
    AnomalyAnalysis,
    CorrectiveAction,
)

from causal.trip_analyzer import (
    TripCausalAnalyzer,
    Event,
    TripEvent,
    TripSequenceAnalysis,
    TripInitiator,
    ContributingFactor,
    PreventionRecommendation,
    TripReport,
)

__all__ = [
    # Causal Graph
    "CombustionCausalGraph",
    "CausalGraph",
    "CausalNode",
    "CausalEdge",
    "GraphValidationResult",
    # Root Cause Analysis
    "RootCauseAnalyzer",
    "RootCauseAnalysis",
    "RankedCause",
    "CausalPath",
    "RCAReport",
    # Counterfactual
    "CounterfactualEngine",
    "CounterfactualQuery",
    "CounterfactualResult",
    "Intervention",
    "Effect",
    "ScenarioComparison",
    # Intervention
    "InterventionRecommender",
    "InterventionRecommendation",
    "ImpactEstimate",
    "RankedIntervention",
    "SafetyValidation",
    # Anomaly
    "CausalAnomalyAnalyzer",
    "Anomaly",
    "AnomalySource",
    "PropagationPath",
    "IsolationResult",
    "AnomalyAnalysis",
    "CorrectiveAction",
    # Trip
    "TripCausalAnalyzer",
    "Event",
    "TripEvent",
    "TripSequenceAnalysis",
    "TripInitiator",
    "ContributingFactor",
    "PreventionRecommendation",
    "TripReport",
]

__version__ = "1.0.0"
__author__ = "GreenLang Team"
