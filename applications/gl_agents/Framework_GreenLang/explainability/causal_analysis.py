"""
GreenLang Framework - Causal Analysis Module

Provides causal analysis capabilities for understanding cause-effect
relationships in industrial systems. Uses Directed Acyclic Graphs (DAGs)
for representing causal structures.

Features:
- Directed Acyclic Graph (DAG) construction and validation
- Root cause analysis with confidence scoring
- Counterfactual reasoning (what-if scenarios)
- Intervention recommendations with feasibility assessment
- Causal path tracing
- Do-calculus based causal inference

Theory:
Causal analysis goes beyond correlation to understand actual cause-effect
relationships. This module implements:
1. DAG-based causal models
2. Pearl's do-calculus for interventional reasoning
3. Counterfactual analysis for hypothetical scenarios
4. Root cause identification algorithms

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

import numpy as np

from .explanation_schemas import (
    CausalNode,
    CausalEdge,
    CausalAnalysisResult,
    CounterfactualExplanation,
    RootCauseAnalysis,
)

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in causal graph."""
    VARIABLE = "variable"
    INTERVENTION = "intervention"
    OUTCOME = "outcome"
    CONFOUNDER = "confounder"
    MEDIATOR = "mediator"
    INSTRUMENT = "instrument"


class EdgeType(Enum):
    """Types of edges in causal graph."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    CONFOUNDED = "confounded"
    BIDIRECTIONAL = "bidirectional"


class DeviationType(Enum):
    """Types of outcome deviations."""
    HIGH = "high"
    LOW = "low"
    WITHIN_NORMAL = "within_normal"
    CRITICAL = "critical"


@dataclass
class CausalGraphConfig:
    """Configuration for causal graph analysis."""
    max_path_length: int = 10
    min_effect_threshold: float = 0.01
    confidence_threshold: float = 0.7
    max_counterfactuals: int = 5
    feasibility_threshold: float = 0.5
    enable_caching: bool = True


@dataclass
class InterventionRecommendation:
    """Recommendation for a causal intervention."""
    intervention_id: str
    target_variable: str
    current_value: float
    recommended_value: float
    expected_effect: float
    confidence: float
    feasibility_score: float
    side_effects: List[Dict[str, float]]
    mechanism: str
    priority: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intervention_id": self.intervention_id,
            "target_variable": self.target_variable,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "expected_effect": self.expected_effect,
            "confidence": self.confidence,
            "feasibility_score": self.feasibility_score,
            "side_effects": self.side_effects,
            "mechanism": self.mechanism,
            "priority": self.priority
        }


class CausalGraph:
    """
    Directed Acyclic Graph for causal modeling.

    Represents causal relationships between variables as a DAG.
    Provides methods for graph operations, path finding, and
    causal structure validation.
    """

    def __init__(self):
        """Initialize empty causal graph."""
        self._nodes: Dict[str, CausalNode] = {}
        self._edges: Dict[str, List[CausalEdge]] = defaultdict(list)
        self._reverse_edges: Dict[str, List[CausalEdge]] = defaultdict(list)

    def add_node(
        self,
        node_id: str,
        name: str,
        node_type: str = "variable",
        value: Optional[float] = None,
        unit: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CausalNode:
        """
        Add a node to the causal graph.

        Args:
            node_id: Unique identifier for the node
            name: Human-readable name
            node_type: Type of node (variable, intervention, outcome, etc.)
            value: Current value of the node
            unit: Unit of measurement
            metadata: Additional metadata

        Returns:
            Created CausalNode
        """
        node = CausalNode(
            node_id=node_id,
            name=name,
            node_type=node_type,
            value=value,
            unit=unit,
            parents=[],
            children=[],
            metadata=metadata or {}
        )
        self._nodes[node_id] = node
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        effect_size: float = 0.0,
        confidence: float = 0.95,
        edge_type: str = "direct",
        mechanism: Optional[str] = None
    ) -> CausalEdge:
        """
        Add a directed edge from source to target.

        Args:
            source: Source node ID
            target: Target node ID
            effect_size: Causal effect size
            confidence: Confidence in the edge
            edge_type: Type of causal relationship
            mechanism: Description of causal mechanism

        Returns:
            Created CausalEdge

        Raises:
            ValueError: If nodes don't exist or edge would create cycle
        """
        if source not in self._nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self._nodes:
            raise ValueError(f"Target node '{target}' not found")

        # Check for cycle
        if self._would_create_cycle(source, target):
            raise ValueError(
                f"Edge from '{source}' to '{target}' would create a cycle"
            )

        edge = CausalEdge(
            source=source,
            target=target,
            effect_size=effect_size,
            confidence=confidence,
            edge_type=edge_type,
            mechanism=mechanism
        )

        self._edges[source].append(edge)
        self._reverse_edges[target].append(edge)

        # Update node relationships
        self._nodes[source].children.append(target)
        self._nodes[target].parents.append(source)

        return edge

    def get_node(self, node_id: str) -> Optional[CausalNode]:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def get_nodes(self) -> List[CausalNode]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_edges(self) -> List[CausalEdge]:
        """Get all edges."""
        all_edges = []
        for edges in self._edges.values():
            all_edges.extend(edges)
        return all_edges

    def get_parents(self, node_id: str) -> List[str]:
        """Get parent node IDs."""
        if node_id not in self._nodes:
            return []
        return self._nodes[node_id].parents.copy()

    def get_children(self, node_id: str) -> List[str]:
        """Get child node IDs."""
        if node_id not in self._nodes:
            return []
        return self._nodes[node_id].children.copy()

    def get_ancestors(self, node_id: str) -> Set[str]:
        """Get all ancestor node IDs."""
        ancestors = set()
        queue = deque(self.get_parents(node_id))

        while queue:
            parent = queue.popleft()
            if parent not in ancestors:
                ancestors.add(parent)
                queue.extend(self.get_parents(parent))

        return ancestors

    def get_descendants(self, node_id: str) -> Set[str]:
        """Get all descendant node IDs."""
        descendants = set()
        queue = deque(self.get_children(node_id))

        while queue:
            child = queue.popleft()
            if child not in descendants:
                descendants.add(child)
                queue.extend(self.get_children(child))

        return descendants

    def find_paths(
        self,
        source: str,
        target: str,
        max_length: int = 10
    ) -> List[List[str]]:
        """
        Find all directed paths from source to target.

        Args:
            source: Source node ID
            target: Target node ID
            max_length: Maximum path length

        Returns:
            List of paths (each path is list of node IDs)
        """
        if source not in self._nodes or target not in self._nodes:
            return []

        paths = []
        self._find_paths_dfs(source, target, [source], paths, max_length)
        return paths

    def _find_paths_dfs(
        self,
        current: str,
        target: str,
        path: List[str],
        paths: List[List[str]],
        max_length: int
    ) -> None:
        """DFS helper for path finding."""
        if len(path) > max_length:
            return

        if current == target:
            paths.append(path.copy())
            return

        for child in self.get_children(current):
            if child not in path:  # Avoid cycles
                path.append(child)
                self._find_paths_dfs(child, target, path, paths, max_length)
                path.pop()

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding edge would create a cycle."""
        # If target can reach source, adding edge would create cycle
        return source in self.get_descendants(target)

    def is_valid_dag(self) -> bool:
        """Check if graph is a valid DAG (no cycles)."""
        # Use topological sort - if it fails, there's a cycle
        try:
            self.topological_sort()
            return True
        except ValueError:
            return False

    def topological_sort(self) -> List[str]:
        """
        Perform topological sort of nodes.

        Returns:
            List of node IDs in topological order

        Raises:
            ValueError: If graph contains cycles
        """
        in_degree = {node_id: len(node.parents) for node_id, node in self._nodes.items()}
        queue = deque([n for n, d in in_degree.items() if d == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for child in self.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(result) != len(self._nodes):
            raise ValueError("Graph contains cycles")

        return result

    def get_root_nodes(self) -> List[str]:
        """Get nodes with no parents (root causes)."""
        return [n for n, node in self._nodes.items() if not node.parents]

    def get_leaf_nodes(self) -> List[str]:
        """Get nodes with no children (outcomes)."""
        return [n for n, node in self._nodes.items() if not node.children]

    def compute_total_effect(
        self,
        source: str,
        target: str
    ) -> float:
        """
        Compute total causal effect from source to target.

        Sums effect sizes along all paths.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Total causal effect
        """
        paths = self.find_paths(source, target)
        total_effect = 0.0

        for path in paths:
            path_effect = 1.0
            for i in range(len(path) - 1):
                edge_effect = self._get_edge_effect(path[i], path[i + 1])
                path_effect *= edge_effect
            total_effect += path_effect

        return total_effect

    def _get_edge_effect(self, source: str, target: str) -> float:
        """Get effect size of edge between source and target."""
        for edge in self._edges.get(source, []):
            if edge.target == target:
                return edge.effect_size
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self.get_edges()]
        }


class CausalAnalysisService:
    """
    Causal analysis service for industrial systems.

    Provides comprehensive causal analysis including:
    - Root cause analysis
    - Counterfactual reasoning
    - Intervention recommendations
    - Causal path analysis

    Example:
        >>> service = CausalAnalysisService()
        >>> service.add_variable("temperature", value=350, unit="K")
        >>> service.add_variable("efficiency", value=0.85)
        >>> service.add_causal_relationship("temperature", "efficiency", effect_size=0.1)
        >>> result = service.analyze(outcome_variable="efficiency")
    """

    def __init__(
        self,
        config: Optional[CausalGraphConfig] = None,
        agent_id: str = "GL-FRAMEWORK",
        agent_version: str = "1.0.0"
    ):
        """
        Initialize causal analysis service.

        Args:
            config: Configuration settings
            agent_id: Agent identifier for provenance
            agent_version: Agent version for provenance
        """
        self.config = config or CausalGraphConfig()
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.graph = CausalGraph()

        # Historical baselines for deviation detection
        self._baselines: Dict[str, Dict[str, float]] = {}

        # Cache for analysis results
        self._analysis_cache: Dict[str, CausalAnalysisResult] = {}

        logger.info(
            f"CausalAnalysisService initialized: agent={agent_id}"
        )

    def add_variable(
        self,
        variable_id: str,
        name: Optional[str] = None,
        value: Optional[float] = None,
        unit: Optional[str] = None,
        variable_type: str = "variable",
        metadata: Optional[Dict[str, Any]] = None
    ) -> CausalNode:
        """
        Add a variable to the causal model.

        Args:
            variable_id: Unique identifier
            name: Human-readable name (defaults to variable_id)
            value: Current value
            unit: Unit of measurement
            variable_type: Type of variable
            metadata: Additional metadata

        Returns:
            Created CausalNode
        """
        return self.graph.add_node(
            node_id=variable_id,
            name=name or variable_id,
            node_type=variable_type,
            value=value,
            unit=unit,
            metadata=metadata
        )

    def add_causal_relationship(
        self,
        cause: str,
        effect: str,
        effect_size: float = 0.0,
        confidence: float = 0.95,
        mechanism: Optional[str] = None
    ) -> CausalEdge:
        """
        Add a causal relationship between variables.

        Args:
            cause: Cause variable ID
            effect: Effect variable ID
            effect_size: Magnitude of causal effect
            confidence: Confidence in relationship
            mechanism: Description of causal mechanism

        Returns:
            Created CausalEdge
        """
        return self.graph.add_edge(
            source=cause,
            target=effect,
            effect_size=effect_size,
            confidence=confidence,
            edge_type="direct",
            mechanism=mechanism
        )

    def set_baseline(
        self,
        variable_id: str,
        mean: float,
        std: float,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None
    ) -> None:
        """
        Set baseline statistics for a variable.

        Args:
            variable_id: Variable ID
            mean: Baseline mean value
            std: Baseline standard deviation
            lower_limit: Lower control limit
            upper_limit: Upper control limit
        """
        self._baselines[variable_id] = {
            "mean": mean,
            "std": std,
            "lower_limit": lower_limit or (mean - 3 * std),
            "upper_limit": upper_limit or (mean + 3 * std)
        }

    def analyze(
        self,
        outcome_variable: str,
        current_values: Optional[Dict[str, float]] = None
    ) -> CausalAnalysisResult:
        """
        Perform comprehensive causal analysis.

        Args:
            outcome_variable: Variable to analyze
            current_values: Current values of variables

        Returns:
            CausalAnalysisResult with full analysis
        """
        analysis_id = self._generate_analysis_id(outcome_variable)

        # Update node values if provided
        if current_values:
            for var_id, value in current_values.items():
                node = self.graph.get_node(var_id)
                if node:
                    node.value = value

        # Perform root cause analysis
        root_cause_analysis = self.identify_root_causes(outcome_variable)

        # Generate counterfactuals
        counterfactuals = self.generate_counterfactuals(outcome_variable)

        # Generate intervention recommendations
        interventions = self.recommend_interventions(outcome_variable)

        result = CausalAnalysisResult(
            analysis_id=analysis_id,
            nodes=self.graph.get_nodes(),
            edges=self.graph.get_edges(),
            root_cause_analysis=root_cause_analysis,
            counterfactuals=counterfactuals,
            intervention_recommendations=[i.to_dict() for i in interventions],
            timestamp=datetime.now(timezone.utc)
        )

        if self.config.enable_caching:
            self._analysis_cache[analysis_id] = result

        logger.info(
            f"Causal analysis complete: id={analysis_id[:8]}, "
            f"outcome={outcome_variable}, "
            f"root_causes={len(root_cause_analysis.root_causes) if root_cause_analysis else 0}"
        )

        return result

    def identify_root_causes(
        self,
        outcome_variable: str,
        deviation: Optional[float] = None
    ) -> RootCauseAnalysis:
        """
        Identify root causes of outcome deviation.

        Args:
            outcome_variable: Variable showing deviation
            deviation: Optional deviation magnitude

        Returns:
            RootCauseAnalysis with ranked causes
        """
        analysis_id = self._generate_analysis_id(f"rca_{outcome_variable}")

        outcome_node = self.graph.get_node(outcome_variable)
        if outcome_node is None:
            raise ValueError(f"Outcome variable '{outcome_variable}' not found")

        # Calculate deviation if not provided
        if deviation is None and outcome_variable in self._baselines:
            baseline = self._baselines[outcome_variable]
            current = outcome_node.value or 0.0
            deviation = current - baseline["mean"]

        outcome_deviation = deviation or 0.0

        # Find all root nodes (potential root causes)
        root_nodes = self.graph.get_root_nodes()

        # Find ancestors of outcome that are also ancestors
        ancestors = self.graph.get_ancestors(outcome_variable)

        # Rank root causes by their total effect on outcome
        root_causes = []
        for root in root_nodes:
            if root in ancestors or root == outcome_variable:
                total_effect = self.graph.compute_total_effect(root, outcome_variable)
                if abs(total_effect) >= self.config.min_effect_threshold:
                    root_node = self.graph.get_node(root)
                    root_causes.append({
                        "variable": root,
                        "name": root_node.name if root_node else root,
                        "total_effect": total_effect,
                        "contribution": total_effect * (root_node.value or 0.0) if root_node else 0.0,
                        "current_value": root_node.value if root_node else None,
                        "paths_to_outcome": len(self.graph.find_paths(root, outcome_variable))
                    })

        # Sort by absolute contribution
        root_causes.sort(key=lambda x: abs(x.get("contribution", 0)), reverse=True)

        # Find causal paths
        causal_paths = []
        for cause in root_causes[:5]:  # Top 5 causes
            paths = self.graph.find_paths(cause["variable"], outcome_variable)
            causal_paths.extend(paths)

        # Generate recommendations
        recommendations = self._generate_rca_recommendations(root_causes, outcome_deviation)

        return RootCauseAnalysis(
            analysis_id=analysis_id,
            outcome_variable=outcome_variable,
            outcome_deviation=outcome_deviation,
            root_causes=root_causes,
            causal_paths=causal_paths,
            recommendations=recommendations,
            timestamp=datetime.now(timezone.utc)
        )

    def generate_counterfactuals(
        self,
        outcome_variable: str,
        target_outcome: Optional[float] = None
    ) -> List[CounterfactualExplanation]:
        """
        Generate counterfactual explanations.

        Answers: "What would have to change for the outcome to be different?"

        Args:
            outcome_variable: Variable to explain
            target_outcome: Desired outcome value

        Returns:
            List of counterfactual explanations
        """
        counterfactuals = []
        outcome_node = self.graph.get_node(outcome_variable)

        if outcome_node is None:
            return counterfactuals

        original_outcome = outcome_node.value or 0.0

        # If no target specified, use baseline mean
        if target_outcome is None and outcome_variable in self._baselines:
            target_outcome = self._baselines[outcome_variable]["mean"]
        elif target_outcome is None:
            return counterfactuals

        required_change = target_outcome - original_outcome

        # Find variables that can influence the outcome
        ancestors = self.graph.get_ancestors(outcome_variable)

        for ancestor in list(ancestors)[:self.config.max_counterfactuals]:
            ancestor_node = self.graph.get_node(ancestor)
            if ancestor_node is None or ancestor_node.value is None:
                continue

            # Calculate required intervention
            total_effect = self.graph.compute_total_effect(ancestor, outcome_variable)
            if abs(total_effect) < self.config.min_effect_threshold:
                continue

            required_intervention = required_change / total_effect if total_effect != 0 else 0

            # Check feasibility
            feasibility = self._assess_feasibility(
                ancestor,
                ancestor_node.value,
                ancestor_node.value + required_intervention
            )

            if feasibility >= self.config.feasibility_threshold:
                cf_id = hashlib.sha256(
                    f"{outcome_variable}{ancestor}{target_outcome}".encode()
                ).hexdigest()[:12]

                counterfactuals.append(CounterfactualExplanation(
                    counterfactual_id=cf_id,
                    original_outcome=original_outcome,
                    counterfactual_outcome=target_outcome,
                    interventions={ancestor: ancestor_node.value + required_intervention},
                    effect_size=required_change,
                    confidence=self._get_path_confidence(ancestor, outcome_variable),
                    feasibility_score=feasibility,
                    description=(
                        f"If {ancestor_node.name} were changed from "
                        f"{ancestor_node.value:.2f} to "
                        f"{ancestor_node.value + required_intervention:.2f}, "
                        f"the outcome would change by {required_change:.2f}"
                    )
                ))

        # Sort by feasibility
        counterfactuals.sort(key=lambda x: x.feasibility_score, reverse=True)

        return counterfactuals[:self.config.max_counterfactuals]

    def recommend_interventions(
        self,
        outcome_variable: str,
        optimization_goal: str = "maximize"
    ) -> List[InterventionRecommendation]:
        """
        Recommend interventions to improve outcome.

        Args:
            outcome_variable: Variable to optimize
            optimization_goal: "maximize" or "minimize"

        Returns:
            List of prioritized intervention recommendations
        """
        recommendations = []

        # Find controllable variables (ancestors of outcome)
        ancestors = self.graph.get_ancestors(outcome_variable)

        for ancestor in ancestors:
            node = self.graph.get_node(ancestor)
            if node is None or node.value is None:
                continue

            total_effect = self.graph.compute_total_effect(ancestor, outcome_variable)
            if abs(total_effect) < self.config.min_effect_threshold:
                continue

            # Determine intervention direction
            if optimization_goal == "maximize":
                intervention_direction = 1 if total_effect > 0 else -1
            else:
                intervention_direction = -1 if total_effect > 0 else 1

            # Calculate recommended value (10% change in optimal direction)
            change_amount = abs(node.value * 0.1) * intervention_direction
            recommended_value = node.value + change_amount

            # Check feasibility
            feasibility = self._assess_feasibility(
                ancestor,
                node.value,
                recommended_value
            )

            if feasibility < self.config.feasibility_threshold:
                continue

            # Calculate expected effect
            expected_effect = change_amount * total_effect

            # Find side effects (other descendants)
            side_effects = self._calculate_side_effects(
                ancestor,
                change_amount,
                exclude=[outcome_variable]
            )

            intervention_id = hashlib.sha256(
                f"{ancestor}{recommended_value}".encode()
            ).hexdigest()[:12]

            recommendations.append(InterventionRecommendation(
                intervention_id=intervention_id,
                target_variable=ancestor,
                current_value=node.value,
                recommended_value=recommended_value,
                expected_effect=expected_effect,
                confidence=self._get_path_confidence(ancestor, outcome_variable),
                feasibility_score=feasibility,
                side_effects=side_effects,
                mechanism=self._describe_mechanism(ancestor, outcome_variable),
                priority=0  # Will be set after sorting
            ))

        # Sort by expected effect * feasibility
        recommendations.sort(
            key=lambda x: abs(x.expected_effect) * x.feasibility_score,
            reverse=True
        )

        # Assign priorities
        for i, rec in enumerate(recommendations):
            rec.priority = i + 1

        return recommendations

    def trace_causal_path(
        self,
        source: str,
        target: str
    ) -> Dict[str, Any]:
        """
        Trace causal path from source to target with details.

        Args:
            source: Source variable ID
            target: Target variable ID

        Returns:
            Detailed path information
        """
        paths = self.graph.find_paths(source, target, self.config.max_path_length)

        path_details = []
        for path in paths:
            steps = []
            total_effect = 1.0
            min_confidence = 1.0

            for i in range(len(path) - 1):
                edge = self._get_edge(path[i], path[i + 1])
                if edge:
                    steps.append({
                        "from": path[i],
                        "to": path[i + 1],
                        "effect_size": edge.effect_size,
                        "confidence": edge.confidence,
                        "mechanism": edge.mechanism
                    })
                    total_effect *= edge.effect_size
                    min_confidence = min(min_confidence, edge.confidence)

            path_details.append({
                "path": path,
                "length": len(path) - 1,
                "total_effect": total_effect,
                "min_confidence": min_confidence,
                "steps": steps
            })

        return {
            "source": source,
            "target": target,
            "num_paths": len(paths),
            "paths": path_details,
            "total_effect": sum(p["total_effect"] for p in path_details)
        }

    def simulate_intervention(
        self,
        variable: str,
        new_value: float
    ) -> Dict[str, float]:
        """
        Simulate the effect of an intervention (do-calculus).

        Args:
            variable: Variable to intervene on
            new_value: New value to set

        Returns:
            Dictionary of expected values for all downstream variables
        """
        node = self.graph.get_node(variable)
        if node is None:
            raise ValueError(f"Variable '{variable}' not found")

        old_value = node.value or 0.0
        change = new_value - old_value

        # Get all descendants
        descendants = self.graph.get_descendants(variable)

        # Calculate new values for all descendants
        new_values = {variable: new_value}

        for desc in descendants:
            total_effect = self.graph.compute_total_effect(variable, desc)
            desc_node = self.graph.get_node(desc)
            current = desc_node.value if desc_node and desc_node.value else 0.0
            new_values[desc] = current + (change * total_effect)

        return new_values

    # Private methods

    def _generate_analysis_id(self, context: str) -> str:
        """Generate unique analysis ID."""
        id_data = f"{context}{datetime.now(timezone.utc).isoformat()}{uuid.uuid4()}"
        return hashlib.sha256(id_data.encode()).hexdigest()[:16]

    def _get_edge(self, source: str, target: str) -> Optional[CausalEdge]:
        """Get edge between two nodes."""
        for edge in self.graph._edges.get(source, []):
            if edge.target == target:
                return edge
        return None

    def _get_path_confidence(self, source: str, target: str) -> float:
        """Calculate confidence along causal path."""
        paths = self.graph.find_paths(source, target)
        if not paths:
            return 0.0

        max_confidence = 0.0
        for path in paths:
            path_confidence = 1.0
            for i in range(len(path) - 1):
                edge = self._get_edge(path[i], path[i + 1])
                if edge:
                    path_confidence *= edge.confidence
            max_confidence = max(max_confidence, path_confidence)

        return max_confidence

    def _assess_feasibility(
        self,
        variable: str,
        current_value: float,
        target_value: float
    ) -> float:
        """
        Assess feasibility of changing a variable.

        Returns score from 0 (infeasible) to 1 (highly feasible).
        """
        # Check if within baseline limits
        if variable in self._baselines:
            baseline = self._baselines[variable]
            lower = baseline.get("lower_limit", float("-inf"))
            upper = baseline.get("upper_limit", float("inf"))

            if target_value < lower or target_value > upper:
                return 0.3  # Possible but outside normal range

        # Penalize large changes
        if current_value != 0:
            relative_change = abs((target_value - current_value) / current_value)
            if relative_change > 0.5:  # More than 50% change
                return 0.5
            elif relative_change > 0.2:  # More than 20% change
                return 0.7

        return 1.0

    def _calculate_side_effects(
        self,
        variable: str,
        change: float,
        exclude: List[str]
    ) -> List[Dict[str, float]]:
        """Calculate side effects of intervention on other variables."""
        side_effects = []
        descendants = self.graph.get_descendants(variable)

        for desc in descendants:
            if desc in exclude:
                continue

            total_effect = self.graph.compute_total_effect(variable, desc)
            if abs(total_effect) >= self.config.min_effect_threshold:
                desc_node = self.graph.get_node(desc)
                side_effects.append({
                    "variable": desc,
                    "name": desc_node.name if desc_node else desc,
                    "expected_change": change * total_effect
                })

        return side_effects

    def _describe_mechanism(self, source: str, target: str) -> str:
        """Generate description of causal mechanism."""
        paths = self.graph.find_paths(source, target)
        if not paths:
            return "No causal path found"

        # Use shortest path for description
        shortest_path = min(paths, key=len)

        source_node = self.graph.get_node(source)
        target_node = self.graph.get_node(target)

        source_name = source_node.name if source_node else source
        target_name = target_node.name if target_node else target

        if len(shortest_path) == 2:
            return f"{source_name} directly affects {target_name}"
        else:
            intermediates = [self.graph.get_node(n) for n in shortest_path[1:-1]]
            intermediate_names = [n.name if n else "?" for n in intermediates]
            return (
                f"{source_name} affects {target_name} through "
                f"{' -> '.join(intermediate_names)}"
            )

    def _generate_rca_recommendations(
        self,
        root_causes: List[Dict[str, Any]],
        deviation: float
    ) -> List[str]:
        """Generate recommendations from root cause analysis."""
        recommendations = []

        for i, cause in enumerate(root_causes[:3]):  # Top 3 causes
            variable = cause.get("variable", "")
            name = cause.get("name", variable)
            effect = cause.get("total_effect", 0)

            if effect > 0:
                if deviation > 0:
                    recommendations.append(
                        f"Reduce {name} to decrease outcome"
                    )
                else:
                    recommendations.append(
                        f"Increase {name} to increase outcome"
                    )
            else:
                if deviation > 0:
                    recommendations.append(
                        f"Increase {name} to decrease outcome"
                    )
                else:
                    recommendations.append(
                        f"Reduce {name} to increase outcome"
                    )

        if not recommendations:
            recommendations.append(
                "Review measurement systems and data quality"
            )

        return recommendations


# Utility functions

def build_thermal_system_dag() -> CausalGraph:
    """
    Build a sample causal graph for a thermal system.

    Demonstrates typical causal relationships in industrial
    thermal processes.

    Returns:
        Pre-built CausalGraph
    """
    graph = CausalGraph()

    # Add nodes
    graph.add_node("fuel_flow", "Fuel Flow Rate", "variable", unit="kg/s")
    graph.add_node("air_flow", "Air Flow Rate", "variable", unit="kg/s")
    graph.add_node("combustion_temp", "Combustion Temperature", "variable", unit="K")
    graph.add_node("excess_air", "Excess Air Ratio", "variable")
    graph.add_node("heat_transfer", "Heat Transfer Rate", "variable", unit="kW")
    graph.add_node("steam_production", "Steam Production", "variable", unit="kg/s")
    graph.add_node("efficiency", "Boiler Efficiency", "outcome")
    graph.add_node("emissions", "CO2 Emissions", "outcome", unit="kg/h")

    # Add edges
    graph.add_edge("fuel_flow", "combustion_temp", 0.8, mechanism="Direct heating")
    graph.add_edge("air_flow", "combustion_temp", 0.3, mechanism="Combustion support")
    graph.add_edge("air_flow", "excess_air", 0.9, mechanism="Air-fuel ratio")
    graph.add_edge("fuel_flow", "excess_air", -0.5, mechanism="Air-fuel ratio")
    graph.add_edge("combustion_temp", "heat_transfer", 0.7, mechanism="Radiation and convection")
    graph.add_edge("excess_air", "efficiency", -0.4, mechanism="Stack losses")
    graph.add_edge("heat_transfer", "steam_production", 0.9, mechanism="Water heating")
    graph.add_edge("steam_production", "efficiency", 0.6, mechanism="Useful output")
    graph.add_edge("fuel_flow", "emissions", 0.95, mechanism="Carbon content")
    graph.add_edge("efficiency", "emissions", -0.3, mechanism="Reduced fuel per output")

    return graph


def format_causal_analysis_report(result: CausalAnalysisResult) -> str:
    """
    Format causal analysis result as a text report.

    Args:
        result: Causal analysis result

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "CAUSAL ANALYSIS REPORT",
        "=" * 60,
        f"Analysis ID: {result.analysis_id}",
        f"Timestamp: {result.timestamp.isoformat()}",
        f"Provenance Hash: {result.provenance_hash}",
        "",
        "-" * 40,
        "CAUSAL GRAPH SUMMARY",
        "-" * 40,
        f"Nodes: {len(result.nodes)}",
        f"Edges: {len(result.edges)}",
        ""
    ]

    if result.root_cause_analysis:
        rca = result.root_cause_analysis
        lines.extend([
            "-" * 40,
            "ROOT CAUSE ANALYSIS",
            "-" * 40,
            f"Outcome Variable: {rca.outcome_variable}",
            f"Deviation: {rca.outcome_deviation:.4f}",
            "",
            "Ranked Root Causes:"
        ])

        for i, cause in enumerate(rca.root_causes[:5], 1):
            lines.append(
                f"  {i}. {cause.get('name', cause.get('variable'))}: "
                f"effect={cause.get('total_effect', 0):.3f}, "
                f"contribution={cause.get('contribution', 0):.3f}"
            )

        lines.extend(["", "Recommendations:"])
        for rec in rca.recommendations:
            lines.append(f"  - {rec}")

    if result.counterfactuals:
        lines.extend([
            "",
            "-" * 40,
            "COUNTERFACTUAL EXPLANATIONS",
            "-" * 40
        ])

        for cf in result.counterfactuals[:3]:
            lines.append(f"  {cf.description}")
            lines.append(f"    Feasibility: {cf.feasibility_score:.2f}")

    if result.intervention_recommendations:
        lines.extend([
            "",
            "-" * 40,
            "INTERVENTION RECOMMENDATIONS",
            "-" * 40
        ])

        for rec in result.intervention_recommendations[:5]:
            lines.append(
                f"  #{rec['priority']}. {rec['target_variable']}: "
                f"{rec['current_value']:.2f} -> {rec['recommended_value']:.2f} "
                f"(effect: {rec['expected_effect']:.3f})"
            )

    lines.extend(["", "=" * 60])

    return "\n".join(lines)
