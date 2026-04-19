# -*- coding: utf-8 -*-
"""
GL-002 FLAMEGUARD - Causal Analysis for Flame Safety Events

Causal DAG-based analysis for flame safety root cause analysis.

Reference: NFPA 85, FM Global Loss Prevention
Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations
import hashlib, logging, uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class CausalRelationType(str, Enum):
    DIRECT_CAUSE = "direct_cause"
    CONTRIBUTING_FACTOR = "contributing_factor"
    CORRELATION = "correlation"

class CausalStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"

@dataclass
class CausalNode:
    node_id: str
    name: str
    description: str
    node_type: str
    current_value: Optional[float] = None
    normal_range: Optional[Tuple[float, float]] = None

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
    coefficient: float = 0.0
    mechanism: str = ""

@dataclass
class RootCauseAnalysis:
    analysis_id: str
    timestamp: datetime
    target_effect: str
    identified_causes: List[Dict[str, Any]]
    primary_root_cause: Optional[str]
    contributing_factors: List[str]
    recommendations: List[str]
    confidence_score: float
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            content = f"{self.analysis_id}|{self.target_effect}|{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()

class FlameSafetyCausalDAG:
    VERSION = "1.0.0"

    def __init__(self, agent_id: str = "GL-002"):
        self.agent_id = agent_id
        self._nodes: Dict[str, CausalNode] = {}
        self._edges: List[CausalEdge] = []
        self._adjacency: Dict[str, Set[str]] = {}
        self._reverse_adjacency: Dict[str, Set[str]] = {}
        self._initialize_flame_dag()
        logger.info(f"FlameSafetyCausalDAG initialized for {agent_id}")

    def _initialize_flame_dag(self):
        nodes = [
            CausalNode("fuel_supply", "Fuel Supply", "Fuel pressure and quality", "root_cause", normal_range=(5.0, 15.0)),
            CausalNode("ignition_system", "Ignition System", "Spark igniter condition", "root_cause", normal_range=(0.8, 1.0)),
            CausalNode("combustion_air", "Combustion Air", "Air supply and damper", "root_cause", normal_range=(80, 120)),
            CausalNode("scanner_condition", "Scanner Condition", "Flame detector health", "root_cause", normal_range=(0.9, 1.0)),
            CausalNode("burner_condition", "Burner Condition", "Physical burner state", "root_cause", normal_range=(0.8, 1.0)),
            CausalNode("flame_quality", "Flame Quality", "Flame characteristics", "intermediate", normal_range=(0.7, 1.0)),
            CausalNode("flame_stability", "Flame Stability", "Flame stability index", "effect", normal_range=(0.85, 1.0)),
            CausalNode("flame_loss", "Flame Loss", "Loss of flame event", "effect", normal_range=(0.0, 0.1)),
            CausalNode("ignition_failure", "Ignition Failure", "Failed ignition attempt", "effect", normal_range=(0.0, 0.1)),
        ]
        for node in nodes:
            self._nodes[node.node_id] = node
            self._adjacency[node.node_id] = set()
            self._reverse_adjacency[node.node_id] = set()

        edges = [
            CausalEdge("fuel_supply", "flame_quality", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, 0.8, "Fuel pressure affects flame shape"),
            CausalEdge("combustion_air", "flame_quality", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, 0.7, "Air flow affects combustion"),
            CausalEdge("burner_condition", "flame_quality", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, 0.75, "Burner wear affects flame"),
            CausalEdge("flame_quality", "flame_stability", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, 0.85, "Poor quality causes instability"),
            CausalEdge("flame_stability", "flame_loss", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, 0.9, "Instability leads to loss"),
            CausalEdge("ignition_system", "ignition_failure", CausalRelationType.DIRECT_CAUSE, CausalStrength.STRONG, 0.85, "Igniter failure prevents ignition"),
            CausalEdge("fuel_supply", "ignition_failure", CausalRelationType.CONTRIBUTING_FACTOR, CausalStrength.MODERATE, 0.5, "Low pressure affects ignition"),
            CausalEdge("scanner_condition", "flame_loss", CausalRelationType.CONTRIBUTING_FACTOR, CausalStrength.MODERATE, 0.4, "Scanner fault causes false trip"),
        ]
        for edge in edges:
            self._edges.append(edge)
            self._adjacency[edge.source_id].add(edge.target_id)
            self._reverse_adjacency[edge.target_id].add(edge.source_id)

    def update_node_value(self, node_id: str, value: float):
        if node_id in self._nodes:
            self._nodes[node_id].current_value = value

    def get_upstream_causes(self, node_id: str, max_depth: int = 5) -> List[str]:
        visited, causes = set(), []
        def dfs(current: str, depth: int):
            if depth > max_depth or current in visited:
                return
            visited.add(current)
            for parent in self._reverse_adjacency.get(current, set()):
                causes.append(parent)
                dfs(parent, depth + 1)
        dfs(node_id, 0)
        return causes

    def analyze_root_cause(self, effect_node_id: str, observations: Dict[str, float]) -> RootCauseAnalysis:
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        for node_id, value in observations.items():
            self.update_node_value(node_id, value)

        upstream = self.get_upstream_causes(effect_node_id)
        abnormal_causes = []
        for node_id in upstream:
            node = self._nodes.get(node_id)
            if node and node.is_abnormal():
                edge = next((e for e in self._edges if e.source_id == node_id), None)
                abnormal_causes.append({
                    "node_id": node_id, "name": node.name,
                    "current_value": node.current_value, "normal_range": node.normal_range,
                    "causal_effect": edge.coefficient if edge else 0})

        abnormal_causes.sort(key=lambda x: abs(x["causal_effect"]), reverse=True)
        primary = abnormal_causes[0]["name"] if abnormal_causes else None
        contributing = [c["name"] for c in abnormal_causes[1:4]]
        recommendations = self._generate_recommendations(abnormal_causes, effect_node_id)
        confidence = min(0.95, 0.5 + 0.15 * len(abnormal_causes)) if abnormal_causes else 0.3

        return RootCauseAnalysis(
            analysis_id=analysis_id, timestamp=timestamp,
            target_effect=self._nodes[effect_node_id].name if effect_node_id in self._nodes else effect_node_id,
            identified_causes=abnormal_causes, primary_root_cause=primary,
            contributing_factors=contributing, recommendations=recommendations, confidence_score=confidence)

    def _generate_recommendations(self, causes: List[Dict], effect: str) -> List[str]:
        recs = {"fuel_supply": "Check fuel pressure regulator and supply lines",
                "ignition_system": "Inspect spark igniter and transformer",
                "combustion_air": "Verify air damper operation and linkage",
                "scanner_condition": "Clean flame scanner lens, check calibration",
                "burner_condition": "Inspect burner tips and diffuser"}
        recommendations = [recs[c["node_id"]] for c in causes[:3] if c["node_id"] in recs]
        return recommendations or ["Perform comprehensive burner inspection"]

__all__ = ["CausalRelationType", "CausalStrength", "CausalNode", "CausalEdge",
           "RootCauseAnalysis", "FlameSafetyCausalDAG"]
