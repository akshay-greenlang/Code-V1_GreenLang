# -*- coding: utf-8 -*-
"""
GL-005 COMBUSENSE - Causal Analysis for Combustion Events

Causal DAG-based analysis for understanding combustion anomaly root causes.
Uses directed acyclic graphs to model causal relationships in combustion systems.

Reference: Pearl's Causal Inference, NFPA 85 Root Cause Analysis
Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations
import hashlib, logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

logger = logging.getLogger(__name__)

class CausalRelationType(str, Enum):
    DIRECT_CAUSE = "direct_cause"
    CONTRIBUTING_FACTOR = "contributing_factor"
    CORRELATION = "correlation"
    CONFOUNDED = "confounded"
    MEDIATED = "mediated"

class CausalStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    UNCERTAIN = "uncertain"

@dataclass
class CausalNode:
    node_id: str
    name: str
    description: str
    node_type: str  # "root_cause", "intermediate", "effect", "confounder"
    current_value: Optional[float] = None
    normal_range: Optional[Tuple[float, float]] = None
    unit: str = ""

    def is_abnormal(self) -> bool:
        if self.current_value is None or self.normal_range is None:
            return False
        return not (self.normal_range[0] <= self.current_value <= self.normal_range[1])

@dataclass
class CausalEdge:
    source_id: str
    target_id: str
    relation_type: CausalRelationType
    strength: CausalStrength
    coefficient: float = 0.0  # Causal effect size
    confidence: float = 0.0  # Confidence in the causal relationship
    mechanism: str = ""  # Description of causal mechanism

@dataclass
class CausalPath:
    path_id: str
    nodes: List[str]
    edges: List[CausalEdge]
    total_effect: float
    confidence: float
    description: str

@dataclass
class RootCauseAnalysis:
    analysis_id: str
    timestamp: datetime
    target_effect: str  # The anomaly/effect being analyzed
    identified_causes: List[Dict[str, Any]]
    causal_paths: List[CausalPath]
    primary_root_cause: Optional[str]
    contributing_factors: List[str]
    confounders: List[str]
    recommendations: List[str]
    confidence_score: float
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            content = f"{self.analysis_id}|{self.target_effect}|{self.primary_root_cause}|{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()

class CombustionCausalDAG:
    """
    Directed Acyclic Graph for combustion causal analysis.

    Models causal relationships in combustion systems:
    - Fuel quality → Combustion efficiency
    - Air-fuel ratio → O2/CO levels
    - Burner condition → Flame stability
    - Control system → Process variables
    """

    VERSION = "1.0.0"

    def __init__(self, agent_id: str = "GL-005"):
        self.agent_id = agent_id
        self._nodes: Dict[str, CausalNode] = {}
        self._edges: List[CausalEdge] = []
        self._adjacency: Dict[str, Set[str]] = {}  # source -> targets
        self._reverse_adjacency: Dict[str, Set[str]] = {}  # target -> sources
        self._initialize_combustion_dag()
        logger.info(f"CombustionCausalDAG initialized for {agent_id}")

    def _initialize_combustion_dag(self):
        """Initialize standard combustion causal relationships."""
        # Define nodes
        nodes = [
            CausalNode("fuel_quality", "Fuel Quality", "Fuel heating value and composition", "root_cause",
                      normal_range=(0.95, 1.05), unit="relative"),
            CausalNode("air_fuel_ratio", "Air-Fuel Ratio", "Combustion air to fuel ratio", "intermediate",
                      normal_range=(1.05, 1.20), unit="ratio"),
            CausalNode("burner_condition", "Burner Condition", "Physical state of burners", "root_cause",
                      normal_range=(0.8, 1.0), unit="score"),
            CausalNode("flame_stability", "Flame Stability", "Flame detection and stability", "intermediate",
                      normal_range=(0.9, 1.0), unit="score"),
            CausalNode("o2_level", "O2 Level", "Oxygen in flue gas", "effect",
                      normal_range=(2.0, 5.0), unit="%"),
            CausalNode("co_level", "CO Level", "Carbon monoxide in flue gas", "effect",
                      normal_range=(0, 100), unit="ppm"),
            CausalNode("nox_level", "NOx Level", "Nitrogen oxides in flue gas", "effect",
                      normal_range=(0, 150), unit="ppm"),
            CausalNode("efficiency", "Combustion Efficiency", "Overall combustion efficiency", "effect",
                      normal_range=(85.0, 98.0), unit="%"),
            CausalNode("temperature", "Flame Temperature", "Combustion temperature", "intermediate",
                      normal_range=(1400, 1800), unit="°C"),
            CausalNode("control_system", "Control System", "DCS/PLC control performance", "root_cause",
                      normal_range=(0.95, 1.0), unit="score"),
        ]
        for node in nodes:
            self.add_node(node)

        # Define causal edges
        edges = [
            CausalEdge("fuel_quality", "air_fuel_ratio", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, 0.7, 0.9, "Fuel composition affects stoichiometric ratio"),
            CausalEdge("fuel_quality", "temperature", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, 0.8, 0.9, "Heating value affects flame temperature"),
            CausalEdge("air_fuel_ratio", "o2_level", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, 0.9, 0.95, "Excess air increases O2"),
            CausalEdge("air_fuel_ratio", "co_level", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, -0.8, 0.9, "Rich mixture increases CO"),
            CausalEdge("air_fuel_ratio", "efficiency", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, 0.6, 0.85, "Optimal ratio maximizes efficiency"),
            CausalEdge("burner_condition", "flame_stability", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, 0.8, 0.9, "Worn burners cause flame instability"),
            CausalEdge("burner_condition", "co_level", CausalRelationType.CONTRIBUTING_FACTOR, CausalStrength.MODERATE, 0.4, 0.7, "Poor atomization increases CO"),
            CausalEdge("flame_stability", "efficiency", CausalRelationType.DIRECT_CAUSE, CausalStrength.MODERATE, 0.5, 0.8, "Unstable flames reduce efficiency"),
            CausalEdge("temperature", "nox_level", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, 0.85, 0.9, "High temp increases thermal NOx"),
            CausalEdge("temperature", "efficiency", CausalRelationType.CONTRIBUTING_FACTOR, CausalStrength.MODERATE, 0.4, 0.75, "Temperature affects heat transfer"),
            CausalEdge("control_system", "air_fuel_ratio", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, 0.7, 0.85, "Control maintains ratio setpoint"),
            CausalEdge("control_system", "flame_stability", CausalRelationType.CONTRIBUTING_FACTOR, CausalStrength.MODERATE, 0.3, 0.7, "Control affects flame modulation"),
        ]
        for edge in edges:
            self.add_edge(edge)

    def add_node(self, node: CausalNode):
        self._nodes[node.node_id] = node
        if node.node_id not in self._adjacency:
            self._adjacency[node.node_id] = set()
        if node.node_id not in self._reverse_adjacency:
            self._reverse_adjacency[node.node_id] = set()

    def add_edge(self, edge: CausalEdge):
        self._edges.append(edge)
        if edge.source_id not in self._adjacency:
            self._adjacency[edge.source_id] = set()
        self._adjacency[edge.source_id].add(edge.target_id)
        if edge.target_id not in self._reverse_adjacency:
            self._reverse_adjacency[edge.target_id] = set()
        self._reverse_adjacency[edge.target_id].add(edge.source_id)

    def update_node_value(self, node_id: str, value: float):
        if node_id in self._nodes:
            self._nodes[node_id].current_value = value

    def get_upstream_causes(self, node_id: str, max_depth: int = 5) -> List[str]:
        """Get all upstream causes of a node (ancestors in DAG)."""
        visited = set()
        causes = []

        def dfs(current: str, depth: int):
            if depth > max_depth or current in visited:
                return
            visited.add(current)
            for parent in self._reverse_adjacency.get(current, set()):
                causes.append(parent)
                dfs(parent, depth + 1)

        dfs(node_id, 0)
        return causes

    def get_causal_paths(self, source: str, target: str) -> List[CausalPath]:
        """Find all causal paths from source to target."""
        paths = []

        def dfs(current: str, path: List[str], edges: List[CausalEdge]):
            if current == target:
                total_effect = 1.0
                confidence = 1.0
                for e in edges:
                    total_effect *= e.coefficient
                    confidence *= e.confidence

                paths.append(CausalPath(
                    path_id=str(uuid.uuid4())[:8],
                    nodes=path.copy(),
                    edges=edges.copy(),
                    total_effect=total_effect,
                    confidence=confidence,
                    description=" → ".join(path)
                ))
                return

            for next_node in self._adjacency.get(current, set()):
                if next_node not in path:
                    edge = next((e for e in self._edges if e.source_id == current and e.target_id == next_node), None)
                    if edge:
                        path.append(next_node)
                        edges.append(edge)
                        dfs(next_node, path, edges)
                        path.pop()
                        edges.pop()

        dfs(source, [source], [])
        return paths

    def analyze_root_cause(self, effect_node_id: str, observations: Dict[str, float]) -> RootCauseAnalysis:
        """Perform root cause analysis for an observed effect."""
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Update node values from observations
        for node_id, value in observations.items():
            self.update_node_value(node_id, value)

        # Find abnormal upstream nodes
        upstream = self.get_upstream_causes(effect_node_id)
        abnormal_causes = []
        for node_id in upstream:
            node = self._nodes.get(node_id)
            if node and node.is_abnormal():
                # Get edge to effect
                paths = self.get_causal_paths(node_id, effect_node_id)
                max_effect = max((p.total_effect for p in paths), default=0)
                abnormal_causes.append({
                    "node_id": node_id,
                    "name": node.name,
                    "current_value": node.current_value,
                    "normal_range": node.normal_range,
                    "causal_effect": max_effect,
                    "paths": len(paths)
                })

        # Sort by causal effect
        abnormal_causes.sort(key=lambda x: abs(x["causal_effect"]), reverse=True)

        # Identify primary root cause
        primary = abnormal_causes[0]["name"] if abnormal_causes else None

        # Contributing factors (excluding primary)
        contributing = [c["name"] for c in abnormal_causes[1:4]]

        # Find confounders
        confounders = [n.name for n in self._nodes.values() if n.node_type == "confounder" and n.is_abnormal()]

        # Generate recommendations
        recommendations = self._generate_recommendations(abnormal_causes, effect_node_id)

        # Calculate confidence
        confidence = min(0.95, 0.5 + 0.1 * len(abnormal_causes)) if abnormal_causes else 0.3

        return RootCauseAnalysis(
            analysis_id=analysis_id,
            timestamp=timestamp,
            target_effect=self._nodes[effect_node_id].name if effect_node_id in self._nodes else effect_node_id,
            identified_causes=abnormal_causes,
            causal_paths=[p for c in abnormal_causes[:3] for p in self.get_causal_paths(c["node_id"], effect_node_id)[:2]],
            primary_root_cause=primary,
            contributing_factors=contributing,
            confounders=confounders,
            recommendations=recommendations,
            confidence_score=confidence
        )

    def _generate_recommendations(self, causes: List[Dict], effect: str) -> List[str]:
        """Generate actionable recommendations based on root causes."""
        recommendations = []

        cause_recommendations = {
            "fuel_quality": "Verify fuel supplier quality, test fuel heating value and composition",
            "air_fuel_ratio": "Check O2 trim calibration, verify combustion air damper operation",
            "burner_condition": "Inspect burner tips for wear, check atomization pattern",
            "flame_stability": "Verify flame scanner calibration, check ignition system",
            "control_system": "Review control loop tuning, check sensor calibration",
            "temperature": "Verify thermocouple accuracy, check for refractory damage",
        }

        for cause in causes[:3]:
            node_id = cause["node_id"]
            if node_id in cause_recommendations:
                recommendations.append(cause_recommendations[node_id])

        if not recommendations:
            recommendations.append("Perform comprehensive combustion system inspection")

        return recommendations

    def get_dag_summary(self) -> Dict[str, Any]:
        """Get summary of the causal DAG structure."""
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "root_causes": [n.name for n in self._nodes.values() if n.node_type == "root_cause"],
            "effects": [n.name for n in self._nodes.values() if n.node_type == "effect"],
            "intermediates": [n.name for n in self._nodes.values() if n.node_type == "intermediate"],
        }

__all__ = ["CausalRelationType", "CausalStrength", "CausalNode", "CausalEdge", "CausalPath",
           "RootCauseAnalysis", "CombustionCausalDAG"]
