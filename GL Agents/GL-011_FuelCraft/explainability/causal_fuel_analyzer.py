# -*- coding: utf-8 -*-
# Causal Fuel Analyzer for GL-011 FuelCraft - DAG-based causal analysis
# Author: GreenLang AI Team, Version: 1.0.0

from __future__ import annotations
import hashlib, json, logging, uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class CausalNodeType(str, Enum):
    INPUT = "input"
    INTERMEDIATE = "intermediate"
    OUTPUT = "output"

class CausalEdgeType(str, Enum):
    PHYSICS_BASED = "physics_based"
    LEARNED = "learned"
    REGULATORY = "regulatory"

@dataclass
class CausalNode:
    node_id: str
    name: str
    node_type: CausalNodeType
    description: str = ""
    unit: str = ""
    current_value: Optional[float] = None
    valid_range: Tuple[float, float] = (float("-inf"), float("inf"))

@dataclass
class CausalEdge:
    source_id: str
    target_id: str
    edge_type: CausalEdgeType
    coefficient: float = 1.0
    description: str = ""
    confidence: float = 1.0
    is_active: bool = True

class CounterfactualResult(BaseModel):
    scenario_id: str = Field(...)
    intervention_node: str = Field(...)
    original_value: float = Field(...)
    counterfactual_value: float = Field(...)
    affected_outputs: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    causal_path: List[str] = Field(default_factory=list)
    total_effect: float = Field(0.0)
    provenance_hash: str = Field("")

class CausalExplanation(BaseModel):
    explanation_id: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target_node: str = Field(...)
    direct_causes: Dict[str, float] = Field(default_factory=dict)
    indirect_causes: Dict[str, float] = Field(default_factory=dict)
    total_effect_by_input: Dict[str, float] = Field(default_factory=dict)
    counterfactuals: List[CounterfactualResult] = Field(default_factory=list)
    dag_structure: Dict[str, List[str]] = Field(default_factory=dict)
    provenance_hash: str = Field("")
    def model_post_init(self, ctx):
        if not self.provenance_hash:
            self.provenance_hash = hashlib.sha256(json.dumps({"id": self.explanation_id, "target": self.target_node}, sort_keys=True).encode()).hexdigest()

class FuelCausalDAG:
    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[Tuple[str,str], CausalEdge] = {}
        self._build_fuel_dag()
    def _build_fuel_dag(self):
        for nid, name, unit in [("aromatics_pct", "Aromatics", "%"), ("paraffins_pct", "Paraffins", "%"), ("olefins_pct", "Olefins", "%"), ("ethanol_pct", "Ethanol", "%"), ("sulfur_ppm", "Sulfur", "ppm"), ("biodiesel_pct", "Biodiesel", "%")]:
            self.nodes[nid] = CausalNode(nid, name, CausalNodeType.INPUT, unit=unit, valid_range=(0, 100 if "%" in unit else 500))
        for nid, name in [("combustion_efficiency", "Combustion Efficiency"), ("knock_resistance", "Knock Resistance"), ("ignition_delay", "Ignition Delay"), ("volatility_index", "Volatility Index")]:
            self.nodes[nid] = CausalNode(nid, name, CausalNodeType.INTERMEDIATE)
        for nid, name, unit in [("octane_rating", "Octane Rating", "RON"), ("cetane_number", "Cetane Number", "CN"), ("co2_emissions", "CO2 Emissions", "g/MJ"), ("nox_emissions", "NOx Emissions", "g/MJ")]:
            self.nodes[nid] = CausalNode(nid, name, CausalNodeType.OUTPUT, unit=unit)
        for src, tgt, coef in [("aromatics_pct", "knock_resistance", 0.35), ("paraffins_pct", "knock_resistance", -0.15), ("ethanol_pct", "knock_resistance", 0.45), ("knock_resistance", "octane_rating", 0.9), ("paraffins_pct", "ignition_delay", -0.4), ("aromatics_pct", "ignition_delay", 0.3), ("ignition_delay", "cetane_number", -0.85), ("sulfur_ppm", "co2_emissions", 0.02)]:
            self.edges[(src, tgt)] = CausalEdge(src, tgt, CausalEdgeType.PHYSICS_BASED, coef, confidence=0.95)
        for src, tgt, coef in [("ethanol_pct", "co2_emissions", -0.12), ("biodiesel_pct", "co2_emissions", -0.08), ("aromatics_pct", "nox_emissions", 0.15), ("combustion_efficiency", "co2_emissions", -0.25)]:
            self.edges[(src, tgt)] = CausalEdge(src, tgt, CausalEdgeType.LEARNED, coef, confidence=0.8)
    def get_parents(self, node_id: str) -> List[str]:
        return [src for (src, tgt) in self.edges if tgt == node_id and self.edges[(src,tgt)].is_active]
    def get_children(self, node_id: str) -> List[str]:
        return [tgt for (src, tgt) in self.edges if src == node_id and self.edges[(src,tgt)].is_active]
    def get_ancestors(self, node_id: str, visited: Set[str] = None) -> Set[str]:
        if visited is None: visited = set()
        for p in self.get_parents(node_id):
            if p not in visited: visited.add(p); self.get_ancestors(p, visited)
        return visited
    def compute_total_effect(self, source_id: str, target_id: str, visited: Set[str] = None) -> float:
        if visited is None: visited = set()
        if source_id in visited: return 0.0
        visited.add(source_id)
        if source_id == target_id: return 1.0
        direct = self.edges.get((source_id, target_id))
        effect = direct.coefficient if direct and direct.is_active else 0.0
        for child in self.get_children(source_id):
            if child != target_id:
                edge = self.edges.get((source_id, child))
                if edge and edge.is_active: effect += edge.coefficient * self.compute_total_effect(child, target_id, visited.copy())
        return effect

class CausalFuelAnalyzer:
    def __init__(self, dag: Optional[FuelCausalDAG] = None):
        self.dag = dag or FuelCausalDAG()
        logger.info(f"CausalFuelAnalyzer: {len(self.dag.nodes)} nodes, {len(self.dag.edges)} edges")
    def explain_output(self, output_node: str) -> CausalExplanation:
        if output_node not in self.dag.nodes: raise ValueError(f"Unknown: {output_node}")
        direct = {p: self.dag.edges[(p, output_node)].coefficient for p in self.dag.get_parents(output_node) if (p, output_node) in self.dag.edges}
        ancestors = self.dag.get_ancestors(output_node)
        indirect = {a: round(self.dag.compute_total_effect(a, output_node), 6) for a in ancestors if a not in direct and abs(self.dag.compute_total_effect(a, output_node)) > 0.001}
        inputs = [n for n, node in self.dag.nodes.items() if node.node_type == CausalNodeType.INPUT]
        total = {inp: round(self.dag.compute_total_effect(inp, output_node), 6) for inp in inputs}
        return CausalExplanation(explanation_id=str(uuid.uuid4()), target_node=output_node, direct_causes=direct, indirect_causes=indirect, total_effect_by_input=total, dag_structure={n: self.dag.get_children(n) for n in self.dag.nodes})
    def analyze_counterfactual(self, node: str, orig: float, new: float) -> CounterfactualResult:
        if node not in self.dag.nodes: raise ValueError(f"Unknown: {node}")
        delta = new - orig
        outputs = [n for n, nd in self.dag.nodes.items() if nd.node_type == CausalNodeType.OUTPUT]
        affected = {}
        for out in outputs:
            eff = self.dag.compute_total_effect(node, out)
            if abs(eff) > 0.001:
                orig_v = self.dag.nodes[out].current_value or 0.0
                affected[out] = (round(orig_v, 4), round(orig_v + delta * eff, 4))
        result = CounterfactualResult(scenario_id=str(uuid.uuid4()), intervention_node=node, original_value=orig, counterfactual_value=new, affected_outputs=affected, total_effect=round(sum(abs(n-o) for o,n in affected.values()), 4))
        result.provenance_hash = hashlib.sha256(json.dumps({"s": result.scenario_id, "n": node}).encode()).hexdigest()
        return result
    def analyze_blend_scenario(self, changes: Dict[str, Tuple[float, float]]) -> List[CounterfactualResult]:
        return [self.analyze_counterfactual(n, o, nv) for n, (o, nv) in changes.items() if n in self.dag.nodes and self.dag.nodes[n].node_type == CausalNodeType.INPUT]

def validate_causal_explanation(exp: CausalExplanation) -> Tuple[bool, List[str]]:
    issues = []
    if not exp.target_node: issues.append("No target")
    if not exp.direct_causes and not exp.indirect_causes: issues.append("No causes")
    return len(issues) == 0, issues

__all__ = ["CausalNodeType", "CausalEdgeType", "CausalNode", "CausalEdge", "FuelCausalDAG", "CounterfactualResult", "CausalExplanation", "CausalFuelAnalyzer", "validate_causal_explanation"]
