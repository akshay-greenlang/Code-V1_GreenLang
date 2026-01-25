"""
RootCauseAnalyzer - Root cause analysis for combustion deviations.

This module implements root cause analysis using causal graphs to identify
and rank potential causes of observed deviations in combustion systems.

Example:
    >>> analyzer = RootCauseAnalyzer(causal_graph)
    >>> analysis = analyzer.analyze_deviation('efficiency', observed=0.85, expected=0.92)
    >>> print(analysis.ranked_causes)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from causal.causal_graph import CausalGraph, CausalEdge, NodeType

logger = logging.getLogger(__name__)


class DeviationType(str, Enum):
    HIGH = "high"
    LOW = "low"
    VOLATILE = "volatile"
    TREND = "trend"


class RankedCause(BaseModel):
    variable: str = Field(..., description="Variable name")
    rank: int = Field(..., description="Rank (1 = most likely cause)")
    score: float = Field(..., ge=0.0, le=1.0, description="Likelihood score")
    causal_path: List[str] = Field(default_factory=list, description="Path from cause to effect")
    mechanism: str = Field("", description="Explanation of causal mechanism")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Supporting evidence")


class CausalPath(BaseModel):
    source: str = Field(..., description="Starting node")
    target: str = Field(..., description="Ending node")
    path: List[str] = Field(..., description="Nodes in path order")
    total_strength: float = Field(..., description="Product of edge weights")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="Edge details")
    length: int = Field(..., description="Number of edges in path")


class RootCauseAnalysis(BaseModel):
    target: str = Field(..., description="Target variable analyzed")
    observed: float = Field(..., description="Observed value")
    expected: float = Field(..., description="Expected value")
    deviation: float = Field(..., description="Deviation amount")
    deviation_type: DeviationType = Field(..., description="Type of deviation")
    ranked_causes: List[RankedCause] = Field(default_factory=list)
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float = Field(..., description="Processing time")
    provenance_hash: str = Field("", description="SHA-256 hash")

    class Config:
        use_enum_values = True


class RCAReport(BaseModel):
    analysis: RootCauseAnalysis = Field(..., description="The analysis results")
    summary: str = Field(..., description="Executive summary")
    top_causes: List[str] = Field(default_factory=list, description="Top 3 causes")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
    confidence_level: str = Field(..., description="Overall confidence")
    report_timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field("", description="SHA-256 hash")


class RootCauseAnalyzer:
    def __init__(self, graph: CausalGraph):
        self.graph = graph
        self._nx_graph = graph.to_networkx()
        logger.info(f"RootCauseAnalyzer initialized with {len(graph.nodes)} nodes")

    def analyze_deviation(self, target: str, observed: float, expected: float,
                          data: Optional[pd.DataFrame] = None) -> RootCauseAnalysis:
        start_time = datetime.now()
        logger.info(f"Analyzing deviation for {target}: observed={observed}, expected={expected}")
        
        if target not in self.graph.nodes:
            raise ValueError(f"Target variable {target} not in causal graph")
        
        deviation = observed - expected
        deviation_pct = abs(deviation / expected) if expected != 0 else float("inf")
        
        if deviation > 0:
            deviation_type = DeviationType.HIGH
        else:
            deviation_type = DeviationType.LOW
        
        # Get all potential causes (ancestors)
        ancestors = self.graph.get_ancestors(target)
        direct_parents = self.graph.get_parents(target)
        
        # Rank causes by causal strength
        ranked_causes = self._rank_causes_internal(target, ancestors, direct_parents, data)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        analysis_data = f"{target}{observed}{expected}{len(ranked_causes)}"
        provenance_hash = hashlib.sha256(analysis_data.encode()).hexdigest()
        
        analysis = RootCauseAnalysis(
            target=target, observed=observed, expected=expected,
            deviation=deviation, deviation_type=deviation_type,
            ranked_causes=ranked_causes, processing_time_ms=processing_time,
            provenance_hash=provenance_hash)
        
        logger.info(f"Analysis complete: {len(ranked_causes)} potential causes identified")
        return analysis

    def _rank_causes_internal(self, target: str, ancestors: Set[str],
                              direct_parents: List[str], data: Optional[pd.DataFrame]) -> List[RankedCause]:
        causes = []
        
        for parent in direct_parents:
            path = self.trace_causal_path(parent, target)
            strength = self.compute_causal_strength(parent, target, data) if data is not None else path.total_strength
            
            edge = next((e for e in self.graph.edges if e.source == parent and e.target == target), None)
            mechanism = edge.mechanism if edge else "Direct causal relationship"
            
            causes.append({"variable": parent, "score": strength * 1.2,
                           "path": path.path, "mechanism": mechanism,
                           "confidence": edge.confidence if edge else 0.8})
        
        for ancestor in ancestors - set(direct_parents):
            path = self.trace_causal_path(ancestor, target)
            if path.path:
                decay_factor = 0.85 ** (path.length - 1)
                score = path.total_strength * decay_factor
                causes.append({"variable": ancestor, "score": score,
                               "path": path.path, "mechanism": f"Indirect effect via {path.path}",
                               "confidence": 0.7 * decay_factor})
        
        causes.sort(key=lambda x: x["score"], reverse=True)
        
        ranked_causes = []
        for rank, cause in enumerate(causes, 1):
            ranked_causes.append(RankedCause(
                variable=cause["variable"], rank=rank, score=min(cause["score"], 1.0),
                causal_path=cause["path"], mechanism=cause["mechanism"],
                confidence=cause["confidence"], evidence={}))
        
        return ranked_causes

    def rank_causes(self, target: str, data: pd.DataFrame) -> List[RankedCause]:
        logger.info(f"Ranking causes for {target} using {len(data)} observations")
        
        if target not in self.graph.nodes:
            raise ValueError(f"Target variable {target} not in causal graph")
        
        ancestors = self.graph.get_ancestors(target)
        direct_parents = self.graph.get_parents(target)
        
        return self._rank_causes_internal(target, ancestors, direct_parents, data)

    def trace_causal_path(self, from_node: str, to_node: str) -> CausalPath:
        if from_node not in self.graph.nodes or to_node not in self.graph.nodes:
            raise ValueError(f"Node not found in graph")
        
        try:
            path = nx.shortest_path(self._nx_graph, from_node, to_node)
        except nx.NetworkXNoPath:
            return CausalPath(source=from_node, target=to_node, path=[], 
                              total_strength=0.0, edges=[], length=0)
        
        edges_info = []
        total_strength = 1.0
        
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            edge = next((e for e in self.graph.edges if e.source == source and e.target == target), None)
            if edge:
                edges_info.append({"source": source, "target": target,
                                   "weight": edge.weight, "mechanism": edge.mechanism})
                total_strength *= edge.weight
        
        return CausalPath(source=from_node, target=to_node, path=path,
                          total_strength=total_strength, edges=edges_info, length=len(path) - 1)

    def compute_causal_strength(self, cause: str, effect: str, data: pd.DataFrame) -> float:
        if data is None or cause not in data.columns or effect not in data.columns:
            path = self.trace_causal_path(cause, effect)
            return path.total_strength
        
        try:
            correlation = abs(data[cause].corr(data[effect]))
            path = self.trace_causal_path(cause, effect)
            graph_strength = path.total_strength
            combined_strength = 0.6 * correlation + 0.4 * graph_strength
            return min(combined_strength, 1.0)
        except Exception as e:
            logger.warning(f"Could not compute causal strength: {e}")
            return self.trace_causal_path(cause, effect).total_strength

    def generate_rca_report(self, analysis: RootCauseAnalysis) -> RCAReport:
        logger.info(f"Generating RCA report for {analysis.target}")
        
        # Generate summary
        deviation_dir = "higher" if analysis.deviation_type == DeviationType.HIGH else "lower"
        deviation_pct = abs(analysis.deviation / analysis.expected * 100) if analysis.expected != 0 else 0
        
        summary = (f"Root cause analysis for {analysis.target}: Observed value ({analysis.observed:.3f}) "
                   f"is {deviation_pct:.1f}% {deviation_dir} than expected ({analysis.expected:.3f}). "
                   f"Analysis identified {len(analysis.ranked_causes)} potential causes.")
        
        # Top causes
        top_causes = [f"{c.rank}. {c.variable} (score: {c.score:.2f})" 
                      for c in analysis.ranked_causes[:3]]
        
        # Generate recommendations
        recommendations = []
        for cause in analysis.ranked_causes[:3]:
            if "fuel" in cause.variable.lower():
                recommendations.append(f"Investigate fuel system: Check {cause.variable} settings and quality")
            elif "air" in cause.variable.lower():
                recommendations.append(f"Check air supply: Verify {cause.variable} damper positions and fan operation")
            elif "temp" in cause.variable.lower():
                recommendations.append(f"Monitor temperature: Review {cause.variable} sensor readings and calibration")
            else:
                recommendations.append(f"Review {cause.variable}: {cause.mechanism}")
        
        # Confidence level
        if analysis.ranked_causes:
            avg_confidence = sum(c.confidence for c in analysis.ranked_causes[:3]) / min(len(analysis.ranked_causes), 3)
            if avg_confidence >= 0.8:
                confidence_level = "HIGH"
            elif avg_confidence >= 0.6:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"
        else:
            confidence_level = "UNKNOWN"
        
        report_data = f"{analysis.target}{summary}{top_causes}"
        provenance_hash = hashlib.sha256(report_data.encode()).hexdigest()
        
        report = RCAReport(
            analysis=analysis, summary=summary, top_causes=top_causes,
            recommendations=recommendations, confidence_level=confidence_level,
            provenance_hash=provenance_hash)
        
        logger.info(f"RCA report generated with confidence: {confidence_level}")
        return report
