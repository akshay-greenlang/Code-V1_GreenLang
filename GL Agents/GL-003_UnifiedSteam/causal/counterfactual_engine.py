"""
GL-003 UNIFIEDSTEAM - Counterfactual Engine

Performs counterfactual reasoning for steam system analysis:
- What-if scenario evaluation
- Intervention impact prediction
- Bounded uncertainty estimation

Answers: "If we had held header pressure at X, what would have happened to steam quality?"
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import logging
import uuid
import math

from .causal_graph import CausalGraph, CausalNode, CausalEdge, NodeType, RelationshipType

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of interventions in counterfactual reasoning."""
    DO = "do"  # do(X=x) - set X to value x
    OBSERVE = "observe"  # Condition on observation
    SHIFT = "shift"  # Shift distribution by amount
    CLAMP = "clamp"  # Hold at value (no response to parents)


@dataclass
class InterventionScenario:
    """Definition of an intervention scenario."""
    scenario_id: str
    name: str
    description: str

    # Interventions
    interventions: Dict[str, Dict[str, Any]]  # node_id -> {type, value, ...}

    # Time scope
    intervention_time: datetime
    duration_seconds: Optional[float] = None  # None = instantaneous

    # Constraints
    must_satisfy: List[str] = field(default_factory=list)  # Constraints that must hold
    cannot_violate: List[str] = field(default_factory=list)  # Safety constraints

    def to_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "interventions": self.interventions,
            "intervention_time": self.intervention_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "must_satisfy": self.must_satisfy,
            "cannot_violate": self.cannot_violate,
        }


@dataclass
class UncertaintyBounds:
    """Uncertainty bounds for counterfactual prediction."""
    lower_bound: float
    upper_bound: float
    point_estimate: float

    # Confidence level
    confidence_level: float = 0.95  # e.g., 95% CI

    # Distribution info
    distribution_type: str = "normal"  # "normal", "uniform", "empirical"
    std_dev: Optional[float] = None

    # Sources of uncertainty
    uncertainty_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "point_estimate": self.point_estimate,
            "confidence_level": self.confidence_level,
            "distribution_type": self.distribution_type,
            "std_dev": self.std_dev,
            "uncertainty_sources": self.uncertainty_sources,
        }


@dataclass
class NodeState:
    """State of a node in counterfactual world."""
    node_id: str
    factual_value: float
    counterfactual_value: float
    change: float
    change_percent: float
    uncertainty: Optional[UncertaintyBounds] = None

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "factual_value": self.factual_value,
            "counterfactual_value": self.counterfactual_value,
            "change": self.change,
            "change_percent": self.change_percent,
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
        }


@dataclass
class CounterfactualResult:
    """Result of counterfactual computation."""
    result_id: str
    scenario: InterventionScenario
    timestamp: datetime

    # Target outcomes
    target_node_id: str
    target_name: str

    # Results
    factual_value: float
    counterfactual_value: float
    effect_size: float  # counterfactual - factual
    effect_percent: float

    # Uncertainty
    uncertainty: UncertaintyBounds

    # All affected nodes
    affected_nodes: List[NodeState]

    # Causal path
    causal_path: List[str]
    path_strength: float  # Combined strength of causal path

    # Validity
    is_valid: bool = True
    validity_notes: List[str] = field(default_factory=list)

    # Natural language
    explanation: str = ""

    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "scenario": self.scenario.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "target_node_id": self.target_node_id,
            "target_name": self.target_name,
            "factual_value": self.factual_value,
            "counterfactual_value": self.counterfactual_value,
            "effect_size": self.effect_size,
            "effect_percent": self.effect_percent,
            "uncertainty": self.uncertainty.to_dict(),
            "affected_nodes": [n.to_dict() for n in self.affected_nodes],
            "causal_path": self.causal_path,
            "path_strength": self.path_strength,
            "is_valid": self.is_valid,
            "validity_notes": self.validity_notes,
            "explanation": self.explanation,
        }


@dataclass
class WhatIfResult:
    """Result of what-if scenario evaluation."""
    result_id: str
    scenario_name: str
    timestamp: datetime

    # Scenario definition
    what_if_statement: str  # e.g., "If header pressure were 100 psig..."
    intervention_description: str

    # Multiple outcomes
    outcomes: Dict[str, CounterfactualResult]  # node_id -> result

    # Summary metrics
    primary_outcome: str
    primary_effect: float
    net_benefit: Optional[float] = None

    # Feasibility
    is_feasible: bool = True
    feasibility_notes: List[str] = field(default_factory=list)

    # Recommendations
    recommendation: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "scenario_name": self.scenario_name,
            "timestamp": self.timestamp.isoformat(),
            "what_if_statement": self.what_if_statement,
            "intervention_description": self.intervention_description,
            "outcomes": {k: v.to_dict() for k, v in self.outcomes.items()},
            "primary_outcome": self.primary_outcome,
            "primary_effect": self.primary_effect,
            "net_benefit": self.net_benefit,
            "is_feasible": self.is_feasible,
            "feasibility_notes": self.feasibility_notes,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
        }


class CounterfactualEngine:
    """
    Performs counterfactual reasoning using causal graphs.

    Features:
    - Intervention computation (do-calculus)
    - What-if scenario evaluation
    - Uncertainty quantification
    - Bounded predictions

    Key method: compute_counterfactual()
    Answers: "If we had held header pressure at X, what would have happened to steam quality?"
    """

    def __init__(
        self,
        causal_graph: CausalGraph,
        agent_id: str = "GL-003",
    ) -> None:
        self.graph = causal_graph
        self.agent_id = agent_id

        # Structural equation models (simplified linear for demo)
        # In production, these would be learned from data
        self._structural_equations: Dict[str, Callable] = {}

        # Default uncertainty parameters
        self.default_std_multiplier = 0.1  # 10% of value as std dev
        self.confidence_level = 0.95

        # Physical relationships (simplified models)
        self._physical_models = self._initialize_physical_models()

        # Cached results
        self._results: Dict[str, CounterfactualResult] = {}
        self._whatif_results: Dict[str, WhatIfResult] = {}

        logger.info(f"CounterfactualEngine initialized with graph: {causal_graph.graph_id}")

    def _initialize_physical_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize simplified physical models for steam systems."""
        return {
            # Pressure -> Temperature relationship (saturation)
            "pressure_to_saturation_temp": {
                "type": "polynomial",
                "coefficients": [212, 0.38, -0.0001],  # Simplified: T_sat approx 212 + 0.38*P - 0.0001*P^2
                "domain": (0, 1000),  # psig
            },
            # PRV pressure drop -> superheat increase
            "prv_superheat_gain": {
                "type": "linear",
                "slope": 0.5,  # 0.5 F superheat per psi drop (simplified)
                "intercept": 0,
            },
            # Spray water -> temperature reduction
            "spray_cooling": {
                "type": "energy_balance",
                "h_steam": 1200,  # BTU/lb (approximate)
                "h_water": 180,  # BTU/lb
                "target_superheat": 50,  # F
            },
            # Flow -> pressure relationship
            "flow_pressure_drop": {
                "type": "quadratic",
                "k_factor": 0.001,  # delta_P = k * Q^2
            },
        }

    def register_structural_equation(
        self,
        node_id: str,
        equation: Callable[[Dict[str, float]], float],
    ) -> None:
        """
        Register a structural equation for a node.

        Args:
            node_id: Node to register equation for
            equation: Function mapping parent values to node value
        """
        self._structural_equations[node_id] = equation
        logger.debug(f"Registered structural equation for {node_id}")

    def compute_counterfactual(
        self,
        intervention: Dict[str, float],
        current_state: Dict[str, float],
        causal_model: Optional[CausalGraph] = None,
        target_nodes: Optional[List[str]] = None,
    ) -> CounterfactualResult:
        """
        Compute counterfactual outcome given an intervention.

        Implements do-calculus: P(Y | do(X=x))

        Args:
            intervention: Dict of {node_id: intervention_value}
            current_state: Current values of all nodes
            causal_model: Optional override for causal graph
            target_nodes: Specific nodes to compute outcomes for

        Returns:
            CounterfactualResult with predicted outcomes
        """
        result_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        graph = causal_model or self.graph

        # Create intervention scenario
        scenario = InterventionScenario(
            scenario_id=str(uuid.uuid4())[:8],
            name="Counterfactual computation",
            description=f"Intervention on {list(intervention.keys())}",
            interventions={
                k: {"type": InterventionType.DO.value, "value": v}
                for k, v in intervention.items()
            },
            intervention_time=timestamp,
        )

        # Identify intervention nodes
        intervention_nodes = list(intervention.keys())

        # Identify target nodes (default: all descendants)
        if target_nodes is None:
            target_nodes = set()
            for int_node in intervention_nodes:
                descendants = graph.get_descendants(int_node)
                target_nodes.update(d.node_id for d in descendants)
            target_nodes = list(target_nodes)

        if not target_nodes:
            # Return minimal result if no targets
            return self._create_empty_result(
                result_id, scenario, timestamp, intervention_nodes[0]
            )

        # Compute counterfactual values for each target
        affected_nodes = []
        primary_target = target_nodes[0]

        for target_id in target_nodes:
            node = graph.get_node(target_id)
            if not node:
                continue

            # Get factual value
            factual = current_state.get(target_id, 0.0)

            # Compute counterfactual value
            cf_value, uncertainty = self._compute_node_counterfactual(
                target_id, intervention, current_state, graph
            )

            change = cf_value - factual
            change_pct = (change / factual * 100) if factual != 0 else 0

            affected_nodes.append(NodeState(
                node_id=target_id,
                factual_value=factual,
                counterfactual_value=cf_value,
                change=change,
                change_percent=change_pct,
                uncertainty=uncertainty,
            ))

        # Get primary target result
        primary_node = next(
            (n for n in affected_nodes if n.node_id == primary_target),
            affected_nodes[0] if affected_nodes else None
        )

        if not primary_node:
            return self._create_empty_result(
                result_id, scenario, timestamp, primary_target
            )

        # Compute causal path and strength
        causal_path = []
        path_strength = 1.0
        for int_node in intervention_nodes:
            paths = graph.get_path(int_node, primary_target)
            if paths:
                causal_path = paths[0]
                path_strength = self._compute_path_strength(causal_path, graph)
                break

        # Generate explanation
        explanation = self._generate_counterfactual_explanation(
            intervention, primary_node, causal_path
        )

        # Check validity
        is_valid, validity_notes = self._check_validity(
            intervention, primary_node, current_state
        )

        result = CounterfactualResult(
            result_id=result_id,
            scenario=scenario,
            timestamp=timestamp,
            target_node_id=primary_target,
            target_name=graph.get_node(primary_target).name if graph.get_node(primary_target) else primary_target,
            factual_value=primary_node.factual_value,
            counterfactual_value=primary_node.counterfactual_value,
            effect_size=primary_node.change,
            effect_percent=primary_node.change_percent,
            uncertainty=primary_node.uncertainty,
            affected_nodes=affected_nodes,
            causal_path=causal_path,
            path_strength=path_strength,
            is_valid=is_valid,
            validity_notes=validity_notes,
            explanation=explanation,
        )

        self._results[result_id] = result
        logger.info(
            f"Computed counterfactual: {primary_target} = {primary_node.counterfactual_value:.2f} "
            f"(was {primary_node.factual_value:.2f})"
        )

        return result

    def _compute_node_counterfactual(
        self,
        target_id: str,
        intervention: Dict[str, float],
        current_state: Dict[str, float],
        graph: CausalGraph,
    ) -> Tuple[float, UncertaintyBounds]:
        """Compute counterfactual value for a single node."""
        node = graph.get_node(target_id)
        if not node:
            return 0.0, self._default_uncertainty(0.0)

        # If target is directly intervened on
        if target_id in intervention:
            value = intervention[target_id]
            return value, self._default_uncertainty(value, sources=["direct_intervention"])

        # Get current value
        factual = current_state.get(target_id, 0.0)

        # Check if we have a structural equation
        if target_id in self._structural_equations:
            # Build parent values (using interventions where applicable)
            parent_values = {}
            for parent in graph.get_direct_causes(target_id):
                if parent.node_id in intervention:
                    parent_values[parent.node_id] = intervention[parent.node_id]
                else:
                    parent_values[parent.node_id] = current_state.get(parent.node_id, 0.0)

            cf_value = self._structural_equations[target_id](parent_values)
            return cf_value, self._default_uncertainty(cf_value)

        # Use simplified causal propagation
        # Find paths from intervention nodes to target
        total_effect = 0.0
        uncertainty_sources = []

        for int_node, int_value in intervention.items():
            paths = graph.get_path(int_node, target_id)
            if not paths:
                continue

            # Use shortest path
            path = min(paths, key=len)

            # Compute effect along path
            effect = self._propagate_effect(
                path, int_value, current_state.get(int_node, 0.0),
                current_state, graph
            )

            total_effect += effect
            uncertainty_sources.append(f"path_from_{int_node}")

        cf_value = factual + total_effect

        # Build uncertainty
        uncertainty = self._compute_uncertainty(
            factual, cf_value, uncertainty_sources
        )

        return cf_value, uncertainty

    def _propagate_effect(
        self,
        path: List[str],
        intervention_value: float,
        original_value: float,
        current_state: Dict[str, float],
        graph: CausalGraph,
    ) -> float:
        """Propagate intervention effect along causal path."""
        if len(path) < 2:
            return 0.0

        # Initial change at intervention node
        change = intervention_value - original_value

        # Propagate through path
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]

            edge = graph.get_edge_between(source, target)
            if edge:
                # Apply edge strength
                change *= edge.strength

                # Apply direction
                if not edge.is_positive:
                    change *= -1

                # Apply relationship-specific transformation
                change = self._apply_relationship(
                    edge.relationship_type, change, current_state, source, target
                )

        return change

    def _apply_relationship(
        self,
        relationship: RelationshipType,
        change: float,
        state: Dict[str, float],
        source: str,
        target: str,
    ) -> float:
        """Apply relationship-specific transformation to change."""
        # Different relationships have different effect propagation
        attenuation = {
            RelationshipType.CAUSES: 0.9,
            RelationshipType.AFFECTS: 0.7,
            RelationshipType.TRANSFERS_MASS: 1.0,
            RelationshipType.TRANSFERS_ENERGY: 0.95,
            RelationshipType.CONSTRAINS: 0.5,
            RelationshipType.CONTROLS: 0.85,
        }

        factor = attenuation.get(relationship, 0.8)
        return change * factor

    def _compute_path_strength(
        self,
        path: List[str],
        graph: CausalGraph,
    ) -> float:
        """Compute combined strength of causal path."""
        if len(path) < 2:
            return 1.0

        strength = 1.0
        for i in range(len(path) - 1):
            edge = graph.get_edge_between(path[i], path[i + 1])
            if edge:
                strength *= edge.strength * edge.confidence

        return strength

    def _default_uncertainty(
        self,
        value: float,
        sources: Optional[List[str]] = None,
    ) -> UncertaintyBounds:
        """Create default uncertainty bounds."""
        std = abs(value) * self.default_std_multiplier
        # 95% CI for normal distribution
        margin = 1.96 * std

        return UncertaintyBounds(
            lower_bound=value - margin,
            upper_bound=value + margin,
            point_estimate=value,
            confidence_level=self.confidence_level,
            distribution_type="normal",
            std_dev=std,
            uncertainty_sources=sources or ["model_uncertainty"],
        )

    def _compute_uncertainty(
        self,
        factual: float,
        counterfactual: float,
        sources: List[str],
    ) -> UncertaintyBounds:
        """Compute uncertainty for counterfactual prediction."""
        # Base uncertainty from prediction
        base_std = abs(counterfactual) * self.default_std_multiplier

        # Additional uncertainty from multiple sources
        source_factor = 1 + 0.1 * len(sources)
        total_std = base_std * source_factor

        margin = 1.96 * total_std

        return UncertaintyBounds(
            lower_bound=counterfactual - margin,
            upper_bound=counterfactual + margin,
            point_estimate=counterfactual,
            confidence_level=self.confidence_level,
            distribution_type="normal",
            std_dev=total_std,
            uncertainty_sources=sources,
        )

    def _generate_counterfactual_explanation(
        self,
        intervention: Dict[str, float],
        primary_node: NodeState,
        causal_path: List[str],
    ) -> str:
        """Generate natural language explanation of counterfactual."""
        parts = []

        # Intervention description
        int_desc = ", ".join(
            f"{k} = {v:.1f}" for k, v in intervention.items()
        )
        parts.append(f"If {int_desc}:")

        # Effect description
        direction = "increase" if primary_node.change > 0 else "decrease"
        parts.append(
            f"{primary_node.node_id} would {direction} by "
            f"{abs(primary_node.change):.2f} ({abs(primary_node.change_percent):.1f}%)"
        )

        # Path description
        if len(causal_path) > 2:
            parts.append(
                f"through causal chain: {' -> '.join(causal_path[:4])}"
                + ("..." if len(causal_path) > 4 else "")
            )

        # Uncertainty
        if primary_node.uncertainty:
            parts.append(
                f"(95% CI: [{primary_node.uncertainty.lower_bound:.2f}, "
                f"{primary_node.uncertainty.upper_bound:.2f}])"
            )

        return " ".join(parts)

    def _check_validity(
        self,
        intervention: Dict[str, float],
        result: NodeState,
        current_state: Dict[str, float],
    ) -> Tuple[bool, List[str]]:
        """Check validity of counterfactual result."""
        is_valid = True
        notes = []

        # Check for extreme changes
        if abs(result.change_percent) > 100:
            notes.append("Large change (>100%) - verify assumptions")

        # Check intervention bounds
        for node_id, value in intervention.items():
            node = self.graph.get_node(node_id)
            if node and node.normal_range:
                low, high = node.normal_range
                if value < low * 0.5 or value > high * 1.5:
                    notes.append(f"Intervention on {node_id} outside normal range")

        # Check for sign reversal
        if result.factual_value * result.counterfactual_value < 0:
            notes.append("Sign reversal in counterfactual - unusual scenario")

        return is_valid, notes

    def _create_empty_result(
        self,
        result_id: str,
        scenario: InterventionScenario,
        timestamp: datetime,
        target: str,
    ) -> CounterfactualResult:
        """Create empty result when computation not possible."""
        return CounterfactualResult(
            result_id=result_id,
            scenario=scenario,
            timestamp=timestamp,
            target_node_id=target,
            target_name=target,
            factual_value=0.0,
            counterfactual_value=0.0,
            effect_size=0.0,
            effect_percent=0.0,
            uncertainty=self._default_uncertainty(0.0),
            affected_nodes=[],
            causal_path=[],
            path_strength=0.0,
            is_valid=False,
            validity_notes=["Unable to compute counterfactual"],
        )

    def evaluate_what_if(
        self,
        scenario: str,
        causal_graph: Optional[CausalGraph] = None,
        current_state: Optional[Dict[str, float]] = None,
        target_metrics: Optional[List[str]] = None,
    ) -> WhatIfResult:
        """
        Evaluate a what-if scenario in natural language.

        Args:
            scenario: Natural language scenario description
            causal_graph: Optional override for causal graph
            current_state: Current system state
            target_metrics: Metrics to evaluate

        Returns:
            WhatIfResult with multi-outcome evaluation
        """
        result_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        graph = causal_graph or self.graph
        current_state = current_state or {}

        # Parse scenario into interventions
        intervention = self._parse_scenario(scenario)

        if not intervention:
            return self._create_empty_whatif(result_id, scenario, timestamp)

        # Determine target metrics
        if not target_metrics:
            # Default: find key outcome nodes
            target_metrics = self._identify_key_outcomes(intervention, graph)

        # Compute counterfactual for each target
        outcomes = {}
        for target in target_metrics:
            cf_result = self.compute_counterfactual(
                intervention, current_state, graph, [target]
            )
            outcomes[target] = cf_result

        # Determine primary outcome
        if outcomes:
            primary = max(
                outcomes.items(),
                key=lambda x: abs(x[1].effect_percent) if x[1].effect_percent else 0
            )
            primary_outcome = primary[0]
            primary_effect = primary[1].effect_size
        else:
            primary_outcome = ""
            primary_effect = 0.0

        # Calculate net benefit (if cost model available)
        net_benefit = self._calculate_net_benefit(intervention, outcomes)

        # Assess feasibility
        is_feasible, feasibility_notes = self._assess_feasibility(intervention, graph)

        # Generate recommendation
        recommendation = self._generate_whatif_recommendation(
            intervention, outcomes, is_feasible
        )

        # Calculate overall confidence
        confidence = self._calculate_whatif_confidence(outcomes)

        # Format what-if statement
        what_if_statement = self._format_whatif_statement(intervention)

        result = WhatIfResult(
            result_id=result_id,
            scenario_name=scenario[:50],
            timestamp=timestamp,
            what_if_statement=what_if_statement,
            intervention_description=str(intervention),
            outcomes=outcomes,
            primary_outcome=primary_outcome,
            primary_effect=primary_effect,
            net_benefit=net_benefit,
            is_feasible=is_feasible,
            feasibility_notes=feasibility_notes,
            recommendation=recommendation,
            confidence=confidence,
        )

        self._whatif_results[result_id] = result
        logger.info(f"Evaluated what-if scenario: {scenario[:30]}...")

        return result

    def _parse_scenario(self, scenario: str) -> Dict[str, float]:
        """Parse natural language scenario into interventions."""
        intervention = {}
        scenario_lower = scenario.lower()

        # Simple keyword-based parsing
        # In production, would use NLP

        # Pressure scenarios
        if "pressure" in scenario_lower:
            # Extract number
            import re
            numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(?:psig|psi)', scenario_lower)
            if numbers:
                intervention["header_HP_pressure"] = float(numbers[0])

        # Temperature scenarios
        if "temperature" in scenario_lower:
            import re
            numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(?:f|degrees)', scenario_lower)
            if numbers:
                intervention["steam_temperature"] = float(numbers[0])

        # Flow scenarios
        if "flow" in scenario_lower:
            import re
            numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(?:klb|lb)', scenario_lower)
            if numbers:
                intervention["steam_flow"] = float(numbers[0])

        return intervention

    def _identify_key_outcomes(
        self,
        intervention: Dict[str, float],
        graph: CausalGraph,
    ) -> List[str]:
        """Identify key outcome nodes to evaluate."""
        outcomes = set()

        # Standard outcomes
        standard = ["efficiency", "steam_quality", "cost"]
        for std in standard:
            nodes = [n for n in graph._nodes.values() if std in n.name.lower()]
            outcomes.update(n.node_id for n in nodes)

        # Descendants of intervention nodes
        for int_node in intervention.keys():
            descendants = graph.get_descendants(int_node, max_depth=3)
            outcomes.update(d.node_id for d in descendants[:5])

        return list(outcomes)[:10]

    def _calculate_net_benefit(
        self,
        intervention: Dict[str, float],
        outcomes: Dict[str, CounterfactualResult],
    ) -> Optional[float]:
        """Calculate net benefit of intervention (placeholder)."""
        # Would integrate with cost model
        return None

    def _assess_feasibility(
        self,
        intervention: Dict[str, float],
        graph: CausalGraph,
    ) -> Tuple[bool, List[str]]:
        """Assess feasibility of intervention."""
        is_feasible = True
        notes = []

        for node_id, value in intervention.items():
            node = graph.get_node(node_id)

            if node:
                # Check controllability
                if not node.is_controllable:
                    is_feasible = False
                    notes.append(f"{node_id} is not controllable")

                # Check range
                if node.normal_range:
                    low, high = node.normal_range
                    if value < low or value > high:
                        notes.append(f"{node_id} value {value} outside normal range [{low}, {high}]")

        return is_feasible, notes

    def _generate_whatif_recommendation(
        self,
        intervention: Dict[str, float],
        outcomes: Dict[str, CounterfactualResult],
        is_feasible: bool,
    ) -> str:
        """Generate recommendation based on what-if analysis."""
        if not is_feasible:
            return "Intervention not feasible with current system configuration"

        # Check if outcomes are beneficial
        positive_effects = [
            o for o in outcomes.values()
            if o.effect_percent > 0
        ]
        negative_effects = [
            o for o in outcomes.values()
            if o.effect_percent < 0
        ]

        if len(positive_effects) > len(negative_effects):
            return "Intervention shows net positive effects; consider implementation"
        elif len(negative_effects) > len(positive_effects):
            return "Intervention shows net negative effects; not recommended"
        else:
            return "Mixed effects; further analysis recommended"

    def _calculate_whatif_confidence(
        self,
        outcomes: Dict[str, CounterfactualResult],
    ) -> float:
        """Calculate overall confidence in what-if results."""
        if not outcomes:
            return 0.0

        confidences = [
            o.path_strength for o in outcomes.values()
            if o.path_strength > 0
        ]

        if confidences:
            return sum(confidences) / len(confidences)
        return 0.5

    def _format_whatif_statement(self, intervention: Dict[str, float]) -> str:
        """Format intervention as what-if statement."""
        parts = []
        for node_id, value in intervention.items():
            parts.append(f"{node_id} were {value:.1f}")

        return f"If {', '.join(parts)}..."

    def _create_empty_whatif(
        self,
        result_id: str,
        scenario: str,
        timestamp: datetime,
    ) -> WhatIfResult:
        """Create empty what-if result."""
        return WhatIfResult(
            result_id=result_id,
            scenario_name=scenario[:50],
            timestamp=timestamp,
            what_if_statement=f"Unable to parse: {scenario}",
            intervention_description="",
            outcomes={},
            primary_outcome="",
            primary_effect=0.0,
            is_feasible=False,
            feasibility_notes=["Could not parse scenario"],
        )

    def get_result(self, result_id: str) -> Optional[CounterfactualResult]:
        """Get counterfactual result by ID."""
        return self._results.get(result_id)

    def get_whatif_result(self, result_id: str) -> Optional[WhatIfResult]:
        """Get what-if result by ID."""
        return self._whatif_results.get(result_id)
