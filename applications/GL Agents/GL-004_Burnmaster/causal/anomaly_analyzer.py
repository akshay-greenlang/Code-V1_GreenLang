"""
CausalAnomalyAnalyzer - Analyzes anomalies using causal graph structure.

This module implements causal-based anomaly analysis to detect anomaly sources,
trace propagation paths, isolate root causes, and recommend corrective actions.

Example:
    >>> analyzer = CausalAnomalyAnalyzer(causal_graph)
    >>> anomaly = Anomaly(variable='CO', value=800, threshold=500)
    >>> source = analyzer.detect_anomaly_source(anomaly, graph)
    >>> print(source.root_cause)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

from causal.causal_graph import CausalGraph, NodeType

logger = logging.getLogger(__name__)


class AnomalySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Anomaly(BaseModel):
    variable: str = Field(..., description="Variable with anomaly")
    value: float = Field(..., description="Observed anomalous value")
    threshold: float = Field(..., description="Normal threshold")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    severity: AnomalySeverity = Field(default=AnomalySeverity.MEDIUM)
    description: str = Field("")

    class Config:
        use_enum_values = True


class AnomalySource(BaseModel):
    anomaly: Anomaly = Field(...)
    root_cause: str = Field(..., description="Most likely root cause variable")
    root_cause_confidence: float = Field(..., ge=0.0, le=1.0)
    contributing_factors: List[str] = Field(default_factory=list)
    propagation_path: List[str] = Field(default_factory=list)
    mechanism: str = Field(..., description="Explanation of how anomaly occurred")
    evidence: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field("")


class PropagationPath(BaseModel):
    source: str = Field(..., description="Source of anomaly")
    affected_nodes: List[str] = Field(default_factory=list)
    propagation_order: List[str] = Field(default_factory=list)
    impact_scores: Dict[str, float] = Field(default_factory=dict)
    expected_delays: Dict[str, float] = Field(default_factory=dict)
    total_impact: float = Field(...)


class IsolationResult(BaseModel):
    symptoms: List[str] = Field(...)
    isolated_cause: str = Field(..., description="Isolated root cause")
    confidence: float = Field(..., ge=0.0, le=1.0)
    ruling_out: List[str] = Field(default_factory=list, description="Variables ruled out")
    evidence_chain: List[str] = Field(default_factory=list)
    isolation_method: str = Field(...)


class AnomalyAnalysis(BaseModel):
    anomalies: List[Anomaly] = Field(...)
    source: AnomalySource = Field(...)
    propagation: PropagationPath = Field(...)
    isolation: IsolationResult = Field(...)
    overall_severity: AnomalySeverity = Field(...)
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field("")

    class Config:
        use_enum_values = True


class CorrectiveAction(BaseModel):
    action_type: str = Field(..., description="Type of corrective action")
    target_variable: str = Field(...)
    recommended_value: Optional[float] = Field(None)
    priority: int = Field(..., ge=1, le=5)
    rationale: str = Field(...)
    expected_effect: str = Field(...)
    safety_notes: List[str] = Field(default_factory=list)
    estimated_time: str = Field(...)


class CausalAnomalyAnalyzer:
    def __init__(self, graph: CausalGraph, historical_data: Optional[Any] = None):
        self.graph = graph
        self._nx_graph = graph.to_networkx()
        self.historical_data = historical_data
        logger.info(f"CausalAnomalyAnalyzer initialized with {len(graph.nodes)} nodes")

    def detect_anomaly_source(self, anomaly: Anomaly, graph: CausalGraph) -> AnomalySource:
        logger.info(f"Detecting source of anomaly in {anomaly.variable}")
        
        if anomaly.variable not in graph.nodes:
            raise ValueError(f"Variable {anomaly.variable} not in causal graph")
        
        # Get all ancestors (potential causes)
        ancestors = graph.get_ancestors(anomaly.variable)
        direct_parents = graph.get_parents(anomaly.variable)
        
        # Score each ancestor as potential root cause
        cause_scores = {}
        for ancestor in ancestors | set(direct_parents):
            score = self._compute_cause_score(ancestor, anomaly.variable, graph)
            cause_scores[ancestor] = score
        
        # Find most likely root cause
        if cause_scores:
            root_cause = max(cause_scores, key=cause_scores.get)
            confidence = cause_scores[root_cause]
        else:
            root_cause = anomaly.variable
            confidence = 0.5
        
        # Trace propagation path
        try:
            path = nx.shortest_path(self._nx_graph, root_cause, anomaly.variable)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = [root_cause, anomaly.variable]
        
        # Identify contributing factors
        contributing = [a for a in ancestors if cause_scores.get(a, 0) > 0.3 and a != root_cause][:3]
        
        mechanism = self._explain_mechanism(root_cause, anomaly.variable, path, graph)
        
        source_data = f"{anomaly.variable}{root_cause}{confidence}"
        provenance_hash = hashlib.sha256(source_data.encode()).hexdigest()
        
        return AnomalySource(
            anomaly=anomaly, root_cause=root_cause, root_cause_confidence=confidence,
            contributing_factors=contributing, propagation_path=path,
            mechanism=mechanism, evidence={"cause_scores": cause_scores},
            provenance_hash=provenance_hash)

    def _compute_cause_score(self, cause: str, effect: str, graph: CausalGraph) -> float:
        try:
            path = nx.shortest_path(self._nx_graph, cause, effect)
            path_length = len(path) - 1
            
            # Compute path strength
            strength = 1.0
            for i in range(len(path) - 1):
                edge = next((e for e in graph.edges if e.source == path[i] and e.target == path[i+1]), None)
                if edge:
                    strength *= edge.weight
            
            # Apply distance decay
            decay = 0.9 ** path_length
            
            # Boost input nodes
            node = graph.nodes.get(cause)
            boost = 1.2 if node and node.node_type == NodeType.INPUT else 1.0
            
            return min(strength * decay * boost, 1.0)
        except nx.NetworkXNoPath:
            return 0.0

    def _explain_mechanism(self, root_cause: str, effect: str, path: List[str], graph: CausalGraph) -> str:
        if len(path) <= 1:
            return f"Direct relationship between {root_cause} and {effect}"
        
        mechanism = f"Anomaly propagates from {root_cause} through: "
        for i in range(len(path) - 1):
            edge = next((e for e in graph.edges if e.source == path[i] and e.target == path[i+1]), None)
            if edge and edge.mechanism:
                mechanism += f"{path[i]} -> {path[i+1]} ({edge.mechanism}), "
            else:
                mechanism += f"{path[i]} -> {path[i+1]}, "
        return mechanism.rstrip(", ")

    def propagate_anomaly_effect(self, source: str, graph: CausalGraph) -> PropagationPath:
        logger.info(f"Propagating anomaly effect from {source}")
        
        if source not in graph.nodes:
            raise ValueError(f"Source {source} not in causal graph")
        
        descendants = graph.get_descendants(source)
        
        # Calculate propagation order using BFS
        try:
            distances = nx.single_source_shortest_path_length(self._nx_graph, source)
        except nx.NetworkXError:
            distances = {source: 0}
        
        propagation_order = sorted(descendants, key=lambda x: distances.get(x, float('inf')))
        
        # Calculate impact scores
        impact_scores = {}
        for desc in descendants:
            try:
                path = nx.shortest_path(self._nx_graph, source, desc)
                impact = 1.0
                for i in range(len(path) - 1):
                    edge = next((e for e in graph.edges if e.source == path[i] and e.target == path[i+1]), None)
                    if edge:
                        impact *= edge.weight
                impact_scores[desc] = impact
            except nx.NetworkXNoPath:
                impact_scores[desc] = 0.0
        
        # Estimate delays (arbitrary time units)
        expected_delays = {node: dist * 10.0 for node, dist in distances.items() if node in descendants}
        
        total_impact = sum(impact_scores.values())
        
        return PropagationPath(
            source=source, affected_nodes=list(descendants),
            propagation_order=propagation_order, impact_scores=impact_scores,
            expected_delays=expected_delays, total_impact=total_impact)

    def isolate_root_cause(self, symptoms: List[str]) -> IsolationResult:
        logger.info(f"Isolating root cause from {len(symptoms)} symptoms")
        
        if not symptoms:
            raise ValueError("At least one symptom required")
        
        # Find common ancestors of all symptoms
        common_ancestors: Optional[Set[str]] = None
        for symptom in symptoms:
            if symptom in self.graph.nodes:
                ancestors = self.graph.get_ancestors(symptom)
                if common_ancestors is None:
                    common_ancestors = ancestors
                else:
                    common_ancestors &= ancestors
        
        if not common_ancestors:
            common_ancestors = set()
        
        # Score candidates
        candidate_scores = {}
        for candidate in common_ancestors:
            score = 0.0
            for symptom in symptoms:
                score += self._compute_cause_score(candidate, symptom, self.graph)
            candidate_scores[candidate] = score / len(symptoms)
        
        # Find best candidate
        if candidate_scores:
            isolated_cause = max(candidate_scores, key=candidate_scores.get)
            confidence = candidate_scores[isolated_cause]
        else:
            isolated_cause = symptoms[0]
            confidence = 0.3
        
        # Rule out less likely candidates
        ruling_out = [c for c in candidate_scores if candidate_scores[c] < confidence * 0.5][:5]
        
        evidence_chain = [f"All symptoms ({symptoms}) share common ancestor: {isolated_cause}"]
        for symptom in symptoms:
            try:
                path = nx.shortest_path(self._nx_graph, isolated_cause, symptom)
                evidence_chain.append(f"Path to {symptom}: {' -> '.join(path)}")
            except nx.NetworkXNoPath:
                pass
        
        return IsolationResult(
            symptoms=symptoms, isolated_cause=isolated_cause, confidence=confidence,
            ruling_out=ruling_out, evidence_chain=evidence_chain,
            isolation_method="common_ancestor_analysis")

    def recommend_corrective_action(self, analysis: AnomalyAnalysis) -> CorrectiveAction:
        logger.info(f"Recommending corrective action for {analysis.source.root_cause}")
        
        root_cause = analysis.source.root_cause
        severity = analysis.overall_severity
        
        # Determine action type based on root cause
        node = self.graph.nodes.get(root_cause)
        if node:
            if node.node_type == NodeType.INPUT:
                action_type = "adjust_setpoint"
            elif node.node_type == NodeType.INTERMEDIATE:
                action_type = "investigate_process"
            else:
                action_type = "monitor_output"
        else:
            action_type = "investigate"
        
        # Determine priority based on severity
        priority_map = {
            AnomalySeverity.LOW: 4,
            AnomalySeverity.MEDIUM: 3,
            AnomalySeverity.HIGH: 2,
            AnomalySeverity.CRITICAL: 1
        }
        priority = priority_map.get(severity, 3)
        
        # Generate recommendation
        if root_cause == "fuel_flow":
            rationale = "Anomaly traced to fuel supply system"
            recommended_value = None  # Requires further analysis
            expected_effect = "Stabilize combustion and reduce downstream anomalies"
            safety_notes = ["Verify fuel pressure before adjustment", "Monitor flame stability"]
        elif root_cause == "air_flow":
            rationale = "Anomaly traced to air supply system"
            recommended_value = None
            expected_effect = "Restore proper air-fuel ratio"
            safety_notes = ["Check damper positions", "Verify fan operation"]
        elif root_cause == "flame_temp":
            rationale = "Anomaly in combustion temperature"
            recommended_value = None
            expected_effect = "Normalize flame conditions"
            safety_notes = ["Review temperature sensor calibration", "Check for air leaks"]
        else:
            rationale = f"Investigate {root_cause} for anomaly resolution"
            recommended_value = None
            expected_effect = "Address root cause of observed symptoms"
            safety_notes = ["Follow standard troubleshooting procedures"]
        
        estimated_time = "5-15 minutes" if priority <= 2 else "15-60 minutes"
        
        return CorrectiveAction(
            action_type=action_type, target_variable=root_cause,
            recommended_value=recommended_value, priority=priority,
            rationale=rationale, expected_effect=expected_effect,
            safety_notes=safety_notes, estimated_time=estimated_time)

    def create_full_analysis(self, anomalies: List[Anomaly]) -> AnomalyAnalysis:
        logger.info(f"Creating full analysis for {len(anomalies)} anomalies")
        
        if not anomalies:
            raise ValueError("At least one anomaly required")
        
        # Use first anomaly for source detection
        primary = anomalies[0]
        source = self.detect_anomaly_source(primary, self.graph)
        propagation = self.propagate_anomaly_effect(source.root_cause, self.graph)
        
        symptoms = [a.variable for a in anomalies]
        isolation = self.isolate_root_cause(symptoms)
        
        # Determine overall severity
        severities = [a.severity for a in anomalies]
        if AnomalySeverity.CRITICAL in severities:
            overall = AnomalySeverity.CRITICAL
        elif AnomalySeverity.HIGH in severities:
            overall = AnomalySeverity.HIGH
        elif AnomalySeverity.MEDIUM in severities:
            overall = AnomalySeverity.MEDIUM
        else:
            overall = AnomalySeverity.LOW
        
        analysis_data = f"{len(anomalies)}{source.root_cause}{overall}"
        provenance_hash = hashlib.sha256(analysis_data.encode()).hexdigest()
        
        return AnomalyAnalysis(
            anomalies=anomalies, source=source, propagation=propagation,
            isolation=isolation, overall_severity=overall,
            provenance_hash=provenance_hash)
