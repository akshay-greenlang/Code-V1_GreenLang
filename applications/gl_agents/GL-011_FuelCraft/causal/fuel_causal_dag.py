# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT - Formal Causal DAG for Fuel Optimization

Implements directed acyclic graph (DAG) for causal analysis of fuel blend
optimization decisions. Supports counterfactual reasoning and intervention analysis.

Reference: ASTM D4814, EPA 40 CFR Part 80, ISO 22000
Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations
import hashlib, logging, uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Type of causal node in the DAG."""
    ROOT_CAUSE = "root_cause"
    INTERMEDIATE = "intermediate"
    EFFECT = "effect"
    CONFOUNDING = "confounding"
    INSTRUMENTAL = "instrumental"


class EdgeType(str, Enum):
    """Type of causal relationship."""
    DIRECT_CAUSE = "direct_cause"
    CONTRIBUTING_FACTOR = "contributing_factor"
    MODERATING = "moderating"
    MEDIATING = "mediating"


class InterventionType(str, Enum):
    """Type of do-calculus intervention."""
    SET_VALUE = "set_value"
    SHIFT_VALUE = "shift_value"
    RANDOMIZE = "randomize"


@dataclass
class CausalNode:
    """Node in the causal DAG representing a variable."""
    node_id: str
    name: str
    description: str
    node_type: NodeType
    unit: str = ""
    current_value: Optional[float] = None
    normal_range: Optional[Tuple[float, float]] = None
    is_observable: bool = True
    is_manipulable: bool = True

    def is_abnormal(self) -> bool:
        """Check if current value is outside normal range."""
        if self.current_value is None or self.normal_range is None:
            return False
        return not (self.normal_range[0] <= self.current_value <= self.normal_range[1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "description": self.description,
            "node_type": self.node_type.value,
            "unit": self.unit,
            "current_value": self.current_value,
            "normal_range": list(self.normal_range) if self.normal_range else None,
            "is_abnormal": self.is_abnormal(),
        }


@dataclass
class CausalEdge:
    """Directed edge representing causal relationship between nodes."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    coefficient: float = 1.0
    mechanism: str = ""
    confidence: float = 0.9
    reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "coefficient": self.coefficient,
            "mechanism": self.mechanism,
            "confidence": self.confidence,
        }


@dataclass
class CausalEffect:
    """Represents a computed causal effect."""
    cause_node: str
    effect_node: str
    total_effect: float
    direct_effect: float
    indirect_effect: float
    paths: List[List[str]]
    confidence_interval: Tuple[float, float]
    p_value: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause_node": self.cause_node,
            "effect_node": self.effect_node,
            "total_effect": self.total_effect,
            "direct_effect": self.direct_effect,
            "indirect_effect": self.indirect_effect,
            "paths": self.paths,
            "confidence_interval": list(self.confidence_interval),
        }


@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis."""
    result_id: str
    timestamp: datetime
    original_outcome: float
    counterfactual_outcome: float
    intervention: Dict[str, float]
    effect_size: float
    confidence: float
    explanation: str
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            content = f"{self.result_id}|{self.intervention}|{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class GraphValidationResult:
    """Result of DAG validation."""
    is_valid: bool
    is_acyclic: bool
    has_root_nodes: bool
    has_leaf_nodes: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class FuelCausalGraph:
    """
    Causal graph with path analysis and effect computation.

    Provides methods for analyzing causal paths and computing
    direct, indirect, and total causal effects.
    """

    def __init__(self):
        self._nodes: Dict[str, CausalNode] = {}
        self._edges: List[CausalEdge] = {}
        self._adjacency: Dict[str, Set[str]] = {}
        self._reverse_adjacency: Dict[str, Set[str]] = {}

    def add_node(self, node: CausalNode) -> None:
        """Add a node to the graph."""
        self._nodes[node.node_id] = node
        if node.node_id not in self._adjacency:
            self._adjacency[node.node_id] = set()
        if node.node_id not in self._reverse_adjacency:
            self._reverse_adjacency[node.node_id] = set()

    def add_edge(self, edge: CausalEdge) -> None:
        """Add an edge to the graph."""
        self._edges.append(edge)
        self._adjacency[edge.source_id].add(edge.target_id)
        self._reverse_adjacency[edge.target_id].add(edge.source_id)

    def get_parents(self, node_id: str) -> Set[str]:
        """Get parent nodes (direct causes)."""
        return self._reverse_adjacency.get(node_id, set())

    def get_children(self, node_id: str) -> Set[str]:
        """Get child nodes (direct effects)."""
        return self._adjacency.get(node_id, set())

    def find_all_paths(self, source: str, target: str, max_depth: int = 10) -> List[List[str]]:
        """Find all causal paths between two nodes."""
        paths = []

        def dfs(current: str, path: List[str], visited: Set[str]):
            if len(path) > max_depth:
                return
            if current == target:
                paths.append(path.copy())
                return
            for child in self._adjacency.get(current, set()):
                if child not in visited:
                    visited.add(child)
                    path.append(child)
                    dfs(child, path, visited)
                    path.pop()
                    visited.remove(child)

        dfs(source, [source], {source})
        return paths

    def compute_path_effect(self, path: List[str]) -> float:
        """Compute effect along a single path (product of coefficients)."""
        effect = 1.0
        for i in range(len(path) - 1):
            edge = self._get_edge(path[i], path[i+1])
            if edge:
                effect *= edge.coefficient
        return effect

    def _get_edge(self, source: str, target: str) -> Optional[CausalEdge]:
        """Get edge between two nodes."""
        for edge in self._edges:
            if edge.source_id == source and edge.target_id == target:
                return edge
        return None

    def validate(self) -> GraphValidationResult:
        """Validate the causal graph structure."""
        issues = []
        warnings = []

        # Check for cycles
        is_acyclic = self._is_acyclic()
        if not is_acyclic:
            issues.append("Graph contains cycles - not a valid DAG")

        # Check for root nodes
        root_nodes = [n for n in self._nodes if not self._reverse_adjacency.get(n)]
        has_roots = len(root_nodes) > 0
        if not has_roots:
            issues.append("No root nodes found")

        # Check for leaf nodes
        leaf_nodes = [n for n in self._nodes if not self._adjacency.get(n)]
        has_leaves = len(leaf_nodes) > 0

        # Check for isolated nodes
        for node_id in self._nodes:
            if not self._adjacency.get(node_id) and not self._reverse_adjacency.get(node_id):
                warnings.append(f"Isolated node: {node_id}")

        return GraphValidationResult(
            is_valid=is_acyclic and has_roots,
            is_acyclic=is_acyclic,
            has_root_nodes=has_roots,
            has_leaf_nodes=has_leaves,
            issues=issues,
            warnings=warnings,
        )

    def _is_acyclic(self) -> bool:
        """Check if graph is acyclic using topological sort."""
        in_degree = {node: 0 for node in self._nodes}
        for edge in self._edges:
            in_degree[edge.target_id] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        count = 0

        while queue:
            node = queue.pop(0)
            count += 1
            for child in self._adjacency.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return count == len(self._nodes)


class FuelCausalDAG(FuelCausalGraph):
    """
    Formal Causal DAG for Fuel Blend Optimization.

    Implements a directed acyclic graph representing causal relationships
    in fuel optimization decisions. Supports:
    - Root cause analysis for blend quality issues
    - Counterfactual reasoning ("what if we changed X?")
    - Intervention analysis (do-calculus)
    - Path-specific effect decomposition

    Reference: Pearl's do-calculus, ASTM fuel standards
    """

    VERSION = "1.0.0"

    def __init__(self, agent_id: str = "GL-011"):
        super().__init__()
        self.agent_id = agent_id
        self._counterfactuals: Dict[str, CounterfactualResult] = {}
        self._initialize_fuel_dag()
        logger.info(f"FuelCausalDAG initialized for {agent_id}")

    def _initialize_fuel_dag(self) -> None:
        """Initialize the fuel optimization causal DAG."""
        # Root cause nodes (exogenous/controllable)
        root_nodes = [
            CausalNode("crude_quality", "Crude Oil Quality", "API gravity and sulfur content",
                      NodeType.ROOT_CAUSE, "API", normal_range=(25, 45)),
            CausalNode("blend_ratio", "Blend Ratio", "Proportion of fuel components",
                      NodeType.ROOT_CAUSE, "%", normal_range=(0, 100)),
            CausalNode("additive_dosage", "Additive Dosage", "Anti-knock additive concentration",
                      NodeType.ROOT_CAUSE, "ppm", normal_range=(0, 500)),
            CausalNode("storage_temp", "Storage Temperature", "Fuel storage tank temperature",
                      NodeType.ROOT_CAUSE, "°F", normal_range=(40, 100)),
            CausalNode("supply_pressure", "Supply Pressure", "Fuel supply line pressure",
                      NodeType.ROOT_CAUSE, "psig", normal_range=(30, 80)),
        ]

        # Intermediate nodes
        intermediate_nodes = [
            CausalNode("volatility", "Volatility", "Reid vapor pressure",
                      NodeType.INTERMEDIATE, "psi", normal_range=(6, 15)),
            CausalNode("octane_rating", "Octane Rating", "Anti-knock index",
                      NodeType.INTERMEDIATE, "RON", normal_range=(87, 93)),
            CausalNode("sulfur_content", "Sulfur Content", "Total sulfur in blend",
                      NodeType.INTERMEDIATE, "ppm", normal_range=(0, 30)),
            CausalNode("carbon_intensity", "Carbon Intensity", "CO2 emissions factor",
                      NodeType.INTERMEDIATE, "kg CO2/GJ", normal_range=(60, 80)),
        ]

        # Effect nodes (outcomes)
        effect_nodes = [
            CausalNode("blend_cost", "Blend Cost", "Total fuel blend cost",
                      NodeType.EFFECT, "$/gal", normal_range=(1.5, 4.0)),
            CausalNode("emission_compliance", "Emission Compliance", "EPA compliance score",
                      NodeType.EFFECT, "%", normal_range=(95, 100)),
            CausalNode("engine_performance", "Engine Performance", "Power output efficiency",
                      NodeType.EFFECT, "%", normal_range=(90, 100)),
            CausalNode("storage_stability", "Storage Stability", "Fuel degradation resistance",
                      NodeType.EFFECT, "months", normal_range=(3, 12)),
        ]

        # Add all nodes
        for node in root_nodes + intermediate_nodes + effect_nodes:
            self.add_node(node)

        # Define causal edges with coefficients
        edges = [
            # Crude quality effects
            CausalEdge("crude_quality", "sulfur_content", EdgeType.DIRECT_CAUSE,
                      -0.7, "Higher API gravity = lower sulfur", reference="ASTM D4294"),
            CausalEdge("crude_quality", "volatility", EdgeType.DIRECT_CAUSE,
                      0.3, "Lighter crudes increase volatility"),

            # Blend ratio effects
            CausalEdge("blend_ratio", "octane_rating", EdgeType.DIRECT_CAUSE,
                      0.8, "Blend proportions determine octane", reference="ASTM D2699"),
            CausalEdge("blend_ratio", "blend_cost", EdgeType.DIRECT_CAUSE,
                      0.9, "Component costs affect total", reference="Cost model"),
            CausalEdge("blend_ratio", "carbon_intensity", EdgeType.DIRECT_CAUSE,
                      0.6, "Blend composition affects CI"),

            # Additive effects
            CausalEdge("additive_dosage", "octane_rating", EdgeType.CONTRIBUTING_FACTOR,
                      0.4, "Additives boost octane"),
            CausalEdge("additive_dosage", "blend_cost", EdgeType.CONTRIBUTING_FACTOR,
                      0.3, "Additive cost contribution"),

            # Storage effects
            CausalEdge("storage_temp", "volatility", EdgeType.DIRECT_CAUSE,
                      0.5, "Temperature affects vapor pressure"),
            CausalEdge("storage_temp", "storage_stability", EdgeType.DIRECT_CAUSE,
                      -0.6, "Heat accelerates degradation"),

            # Intermediate to outcome edges
            CausalEdge("volatility", "engine_performance", EdgeType.DIRECT_CAUSE,
                      0.4, "Proper volatility aids combustion"),
            CausalEdge("octane_rating", "engine_performance", EdgeType.DIRECT_CAUSE,
                      0.7, "Higher octane = better performance"),
            CausalEdge("sulfur_content", "emission_compliance", EdgeType.DIRECT_CAUSE,
                      -0.8, "Lower sulfur = better compliance", reference="EPA 40 CFR 80"),
            CausalEdge("carbon_intensity", "emission_compliance", EdgeType.CONTRIBUTING_FACTOR,
                      -0.5, "Lower CI improves score"),

            # Supply pressure effects
            CausalEdge("supply_pressure", "engine_performance", EdgeType.DIRECT_CAUSE,
                      0.3, "Adequate pressure for atomization"),
        ]

        for edge in edges:
            self.add_edge(edge)

    def update_node_values(self, observations: Dict[str, float]) -> None:
        """Update node values from observations."""
        for node_id, value in observations.items():
            if node_id in self._nodes:
                self._nodes[node_id].current_value = value

    def analyze_root_causes(self, effect_node: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Identify root causes for an effect with contribution scores.

        Args:
            effect_node: The outcome node to analyze
            threshold: Minimum effect magnitude to include

        Returns:
            List of root causes with their contributions
        """
        if effect_node not in self._nodes:
            raise ValueError(f"Unknown effect node: {effect_node}")

        root_causes = []

        # Find all root nodes with paths to effect
        for node_id, node in self._nodes.items():
            if node.node_type == NodeType.ROOT_CAUSE:
                paths = self.find_all_paths(node_id, effect_node)
                if paths:
                    # Compute total effect
                    total_effect = sum(self.compute_path_effect(path) for path in paths)

                    if abs(total_effect) >= threshold:
                        root_causes.append({
                            "node_id": node_id,
                            "name": node.name,
                            "current_value": node.current_value,
                            "is_abnormal": node.is_abnormal(),
                            "total_effect": total_effect,
                            "num_paths": len(paths),
                            "primary_path": paths[0] if paths else [],
                        })

        # Sort by absolute effect magnitude
        root_causes.sort(key=lambda x: abs(x["total_effect"]), reverse=True)
        return root_causes

    def compute_causal_effect(self, cause: str, effect: str) -> CausalEffect:
        """
        Compute total, direct, and indirect causal effects.

        Args:
            cause: Source node ID
            effect: Target node ID

        Returns:
            CausalEffect with decomposed effects
        """
        # Find all paths
        paths = self.find_all_paths(cause, effect)

        if not paths:
            return CausalEffect(
                cause_node=cause,
                effect_node=effect,
                total_effect=0.0,
                direct_effect=0.0,
                indirect_effect=0.0,
                paths=[],
                confidence_interval=(0.0, 0.0),
            )

        # Direct effect (single edge path)
        direct_effect = 0.0
        indirect_paths = []

        for path in paths:
            path_effect = self.compute_path_effect(path)
            if len(path) == 2:  # Direct path
                direct_effect = path_effect
            else:
                indirect_paths.append((path, path_effect))

        # Indirect effect (sum of multi-hop paths)
        indirect_effect = sum(eff for _, eff in indirect_paths)
        total_effect = direct_effect + indirect_effect

        # Simple confidence interval (could be bootstrapped)
        margin = abs(total_effect) * 0.1
        ci = (total_effect - margin, total_effect + margin)

        return CausalEffect(
            cause_node=cause,
            effect_node=effect,
            total_effect=total_effect,
            direct_effect=direct_effect,
            indirect_effect=indirect_effect,
            paths=paths,
            confidence_interval=ci,
        )

    def counterfactual_analysis(
        self,
        intervention: Dict[str, float],
        outcome_node: str,
        prediction_fn: Optional[Callable] = None
    ) -> CounterfactualResult:
        """
        Perform counterfactual analysis: "What if we set X to value?"

        Args:
            intervention: Dict of {node_id: new_value} to intervene on
            outcome_node: The outcome variable to predict
            prediction_fn: Optional custom prediction function

        Returns:
            CounterfactualResult with original and counterfactual outcomes
        """
        result_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Get original outcome
        original_outcome = self._nodes[outcome_node].current_value or 0.0

        # Compute counterfactual using linear structural equations
        counterfactual = original_outcome

        for node_id, new_value in intervention.items():
            if node_id in self._nodes:
                old_value = self._nodes[node_id].current_value or 0.0
                delta = new_value - old_value

                # Find effect of this intervention on outcome
                effect = self.compute_causal_effect(node_id, outcome_node)
                counterfactual += delta * effect.total_effect

        # Use custom prediction if provided
        if prediction_fn:
            counterfactual = prediction_fn(intervention, outcome_node)

        effect_size = counterfactual - original_outcome

        # Generate explanation
        explanations = []
        for node_id, new_value in intervention.items():
            node = self._nodes.get(node_id)
            if node:
                explanations.append(
                    f"Setting {node.name} to {new_value}{node.unit}"
                )

        explanation = (
            f"If {' and '.join(explanations)}, "
            f"then {self._nodes[outcome_node].name} would change by {effect_size:.3f}"
        )

        result = CounterfactualResult(
            result_id=result_id,
            timestamp=timestamp,
            original_outcome=original_outcome,
            counterfactual_outcome=counterfactual,
            intervention=intervention,
            effect_size=effect_size,
            confidence=0.85,
            explanation=explanation,
        )

        self._counterfactuals[result_id] = result
        logger.info(f"Counterfactual analysis: {explanation}")

        return result

    def get_recommendations(self, target_node: str, target_value: float) -> List[Dict[str, Any]]:
        """
        Get recommendations to achieve target outcome.

        Args:
            target_node: Outcome node to optimize
            target_value: Desired value for the outcome

        Returns:
            List of recommended interventions ranked by effectiveness
        """
        current_value = self._nodes[target_node].current_value or 0.0
        needed_change = target_value - current_value

        recommendations = []

        for node_id, node in self._nodes.items():
            if node.node_type == NodeType.ROOT_CAUSE and node.is_manipulable:
                effect = self.compute_causal_effect(node_id, target_node)

                if abs(effect.total_effect) > 0.01:
                    # Required change in cause to achieve target
                    required_change = needed_change / effect.total_effect
                    new_value = (node.current_value or 0.0) + required_change

                    # Check if within normal range
                    feasible = True
                    if node.normal_range:
                        feasible = node.normal_range[0] <= new_value <= node.normal_range[1]

                    recommendations.append({
                        "node_id": node_id,
                        "name": node.name,
                        "current_value": node.current_value,
                        "recommended_value": new_value,
                        "required_change": required_change,
                        "effect_magnitude": abs(effect.total_effect),
                        "is_feasible": feasible,
                        "unit": node.unit,
                    })

        # Sort by effect magnitude (most effective first)
        recommendations.sort(key=lambda x: x["effect_magnitude"], reverse=True)

        return recommendations

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DAG to dictionary."""
        return {
            "agent_id": self.agent_id,
            "version": self.VERSION,
            "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
            "edges": [e.to_dict() for e in self._edges],
            "validation": self.validate().__dict__,
        }


__all__ = [
    "NodeType",
    "EdgeType",
    "InterventionType",
    "CausalNode",
    "CausalEdge",
    "CausalEffect",
    "CounterfactualResult",
    "GraphValidationResult",
    "FuelCausalGraph",
    "FuelCausalDAG",
]
