"""
GL-003 UNIFIEDSTEAM - Intervention Recommender

Recommends feasible interventions based on root cause analysis:
- Generate candidate interventions from root causes
- Evaluate intervention impacts using counterfactual engine
- Rank by feasibility and expected benefit

Feasible operations include:
- Adjust setpoint
- Inspect trap group
- Check spray water temperature
- Adjust PRV setpoint
- Schedule maintenance
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging
import uuid

from .causal_graph import CausalGraph, CausalNode, NodeType
from .root_cause_analyzer import RankedCause, Deviation
from .counterfactual_engine import CounterfactualEngine, CounterfactualResult

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of interventions."""
    SETPOINT_ADJUSTMENT = "setpoint_adjustment"
    EQUIPMENT_INSPECTION = "equipment_inspection"
    VALVE_ADJUSTMENT = "valve_adjustment"
    MAINTENANCE_TASK = "maintenance_task"
    OPERATIONAL_CHANGE = "operational_change"
    CONFIGURATION_CHANGE = "configuration_change"
    PROCESS_ADJUSTMENT = "process_adjustment"
    SAFETY_ACTION = "safety_action"


class Urgency(Enum):
    """Urgency level for intervention."""
    IMMEDIATE = "immediate"  # Within minutes
    URGENT = "urgent"  # Within hours
    SOON = "soon"  # Within shift
    SCHEDULED = "scheduled"  # Can be planned
    OPPORTUNISTIC = "opportunistic"  # When convenient


class FeasibilityRating(Enum):
    """Feasibility rating for intervention."""
    EASY = "easy"  # Routine operation
    MODERATE = "moderate"  # Requires some coordination
    DIFFICULT = "difficult"  # Requires significant effort
    COMPLEX = "complex"  # Requires planning/resources
    INFEASIBLE = "infeasible"  # Not possible currently


@dataclass
class FeasibilityAssessment:
    """Assessment of intervention feasibility."""
    rating: FeasibilityRating
    score: float  # 0-1

    # Factors
    requires_equipment: bool = False
    requires_downtime: bool = False
    requires_personnel: int = 1
    estimated_duration_minutes: int = 15
    estimated_cost: float = 0.0

    # Constraints
    prerequisites: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    safety_considerations: List[str] = field(default_factory=list)

    # Time windows
    can_execute_now: bool = True
    best_time_window: str = "any"

    def to_dict(self) -> Dict:
        return {
            "rating": self.rating.value,
            "score": self.score,
            "requires_equipment": self.requires_equipment,
            "requires_downtime": self.requires_downtime,
            "requires_personnel": self.requires_personnel,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "estimated_cost": self.estimated_cost,
            "prerequisites": self.prerequisites,
            "blockers": self.blockers,
            "safety_considerations": self.safety_considerations,
            "can_execute_now": self.can_execute_now,
            "best_time_window": self.best_time_window,
        }


@dataclass
class ImpactEstimate:
    """Estimated impact of an intervention."""
    impact_id: str
    intervention_id: str
    timestamp: datetime

    # Target metric
    target_metric: str
    current_value: float
    predicted_value: float
    improvement: float
    improvement_percent: float

    # Confidence
    confidence: float
    uncertainty_lower: float
    uncertainty_upper: float

    # Side effects
    side_effects: List[Dict[str, Any]] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

    # Time to effect
    time_to_effect_minutes: int = 5
    effect_duration_hours: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "impact_id": self.impact_id,
            "intervention_id": self.intervention_id,
            "timestamp": self.timestamp.isoformat(),
            "target_metric": self.target_metric,
            "current_value": self.current_value,
            "predicted_value": self.predicted_value,
            "improvement": self.improvement,
            "improvement_percent": self.improvement_percent,
            "confidence": self.confidence,
            "uncertainty_lower": self.uncertainty_lower,
            "uncertainty_upper": self.uncertainty_upper,
            "side_effects": self.side_effects,
            "risk_factors": self.risk_factors,
            "time_to_effect_minutes": self.time_to_effect_minutes,
            "effect_duration_hours": self.effect_duration_hours,
        }


@dataclass
class Intervention:
    """A recommended intervention."""
    intervention_id: str
    timestamp: datetime

    # Type and description
    intervention_type: InterventionType
    name: str
    description: str
    detailed_instructions: str

    # Target
    target_equipment: str
    target_parameter: Optional[str] = None

    # Action details
    action: str
    current_value: Optional[float] = None
    target_value: Optional[float] = None
    unit: str = ""

    # Addressing cause
    addresses_cause: Optional[str] = None
    cause_probability: float = 0.0

    # Priority and urgency
    priority: int = 1  # 1 = highest
    urgency: Urgency = Urgency.SCHEDULED

    # Feasibility
    feasibility: Optional[FeasibilityAssessment] = None

    # Impact
    expected_impact: Optional[ImpactEstimate] = None

    # Verification
    verification_steps: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

    # Safety
    safety_warnings: List[str] = field(default_factory=list)
    lockout_required: bool = False

    def to_dict(self) -> Dict:
        return {
            "intervention_id": self.intervention_id,
            "timestamp": self.timestamp.isoformat(),
            "intervention_type": self.intervention_type.value,
            "name": self.name,
            "description": self.description,
            "detailed_instructions": self.detailed_instructions,
            "target_equipment": self.target_equipment,
            "target_parameter": self.target_parameter,
            "action": self.action,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "unit": self.unit,
            "addresses_cause": self.addresses_cause,
            "cause_probability": self.cause_probability,
            "priority": self.priority,
            "urgency": self.urgency.value,
            "feasibility": self.feasibility.to_dict() if self.feasibility else None,
            "expected_impact": self.expected_impact.to_dict() if self.expected_impact else None,
            "verification_steps": self.verification_steps,
            "success_criteria": self.success_criteria,
            "safety_warnings": self.safety_warnings,
            "lockout_required": self.lockout_required,
        }


@dataclass
class RankedInterventions:
    """Collection of ranked interventions."""
    ranking_id: str
    timestamp: datetime
    deviation_id: str

    # Interventions
    interventions: List[Intervention]
    total_generated: int

    # Top recommendation
    top_recommendation: Optional[Intervention] = None

    # Summary
    summary: str = ""
    implementation_order: List[str] = field(default_factory=list)  # Intervention IDs

    # Constraints applied
    constraints_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "ranking_id": self.ranking_id,
            "timestamp": self.timestamp.isoformat(),
            "deviation_id": self.deviation_id,
            "interventions": [i.to_dict() for i in self.interventions],
            "total_generated": self.total_generated,
            "top_recommendation": self.top_recommendation.to_dict() if self.top_recommendation else None,
            "summary": self.summary,
            "implementation_order": self.implementation_order,
            "constraints_applied": self.constraints_applied,
        }


class InterventionRecommender:
    """
    Recommends feasible interventions based on root cause analysis.

    Features:
    - Generate interventions from identified root causes
    - Evaluate impact using counterfactual engine
    - Rank by feasibility and expected benefit
    - Provide detailed implementation guidance
    """

    def __init__(
        self,
        causal_graph: CausalGraph,
        counterfactual_engine: Optional[CounterfactualEngine] = None,
        agent_id: str = "GL-003",
    ) -> None:
        self.graph = causal_graph
        self.cf_engine = counterfactual_engine or CounterfactualEngine(causal_graph)
        self.agent_id = agent_id

        # Intervention templates
        self._intervention_templates = self._initialize_templates()

        # Feasibility models
        self._feasibility_models = self._initialize_feasibility_models()

        # Operational constraints
        self._operational_constraints: Dict[str, Any] = {}

        # Cached results
        self._interventions: Dict[str, Intervention] = {}
        self._rankings: Dict[str, RankedInterventions] = {}

        logger.info(f"InterventionRecommender initialized with graph: {causal_graph.graph_id}")

    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize intervention templates for common causes."""
        return {
            # Setpoint adjustments
            "adjust_pressure_setpoint": {
                "type": InterventionType.SETPOINT_ADJUSTMENT,
                "name": "Adjust Pressure Setpoint",
                "description": "Modify pressure controller setpoint",
                "instructions": "Navigate to pressure controller, adjust SP to target value, monitor response",
                "verification": ["Verify pressure stabilizes at new SP within 5 minutes"],
                "duration_minutes": 5,
                "personnel": 1,
            },
            "adjust_temperature_setpoint": {
                "type": InterventionType.SETPOINT_ADJUSTMENT,
                "name": "Adjust Temperature Setpoint",
                "description": "Modify temperature controller setpoint",
                "instructions": "Navigate to temperature controller, adjust SP to target value, monitor response",
                "verification": ["Verify temperature stabilizes at new SP within 10 minutes"],
                "duration_minutes": 5,
                "personnel": 1,
            },
            "adjust_prv_setpoint": {
                "type": InterventionType.SETPOINT_ADJUSTMENT,
                "name": "Adjust PRV Setpoint",
                "description": "Modify PRV pilot pressure setpoint",
                "instructions": "Adjust PRV pilot regulator to new setpoint, verify downstream pressure",
                "verification": ["Verify downstream pressure at target", "Check for PRV hunting"],
                "duration_minutes": 15,
                "personnel": 1,
            },

            # Inspections
            "inspect_steam_trap": {
                "type": InterventionType.EQUIPMENT_INSPECTION,
                "name": "Inspect Steam Trap",
                "description": "Perform steam trap inspection with ultrasonic tester",
                "instructions": "Use ultrasonic leak detector to verify trap operation, check inlet/outlet temps",
                "verification": ["Document trap cycling pattern", "Record temperature differential"],
                "duration_minutes": 10,
                "personnel": 1,
            },
            "inspect_trap_group": {
                "type": InterventionType.EQUIPMENT_INSPECTION,
                "name": "Inspect Trap Group",
                "description": "Survey all traps in a process area",
                "instructions": "Systematically test each trap, document findings, tag failed traps",
                "verification": ["Complete trap survey form", "Update trap database"],
                "duration_minutes": 60,
                "personnel": 2,
            },

            # Valve adjustments
            "check_spray_water": {
                "type": InterventionType.VALVE_ADJUSTMENT,
                "name": "Check Spray Water System",
                "description": "Verify spray water valve operation and temperature",
                "instructions": "Check spray water supply temp, verify valve responds to control signal",
                "verification": ["Record spray water temperature", "Test valve stroke"],
                "duration_minutes": 15,
                "personnel": 1,
            },
            "adjust_control_valve": {
                "type": InterventionType.VALVE_ADJUSTMENT,
                "name": "Adjust Control Valve",
                "description": "Modify control valve position or tuning",
                "instructions": "Adjust valve position manually or modify PID tuning",
                "verification": ["Verify process response", "Check for oscillation"],
                "duration_minutes": 20,
                "personnel": 1,
            },

            # Maintenance
            "replace_steam_trap": {
                "type": InterventionType.MAINTENANCE_TASK,
                "name": "Replace Steam Trap",
                "description": "Replace failed steam trap",
                "instructions": "Isolate trap, depressurize, replace with new trap, restore service",
                "verification": ["Leak test after installation", "Verify trap cycles properly"],
                "duration_minutes": 60,
                "personnel": 2,
                "requires_downtime": True,
            },
            "clean_strainer": {
                "type": InterventionType.MAINTENANCE_TASK,
                "name": "Clean Strainer",
                "description": "Clean inlet strainer",
                "instructions": "Isolate strainer, remove screen, clean debris, reinstall",
                "verification": ["Check pressure drop after cleaning"],
                "duration_minutes": 30,
                "personnel": 1,
                "requires_downtime": True,
            },

            # Operational changes
            "adjust_load_distribution": {
                "type": InterventionType.OPERATIONAL_CHANGE,
                "name": "Adjust Load Distribution",
                "description": "Redistribute steam load across boilers",
                "instructions": "Adjust individual boiler firing rates to optimize efficiency",
                "verification": ["Monitor header pressure stability", "Check efficiency metrics"],
                "duration_minutes": 15,
                "personnel": 1,
            },
        }

    def _initialize_feasibility_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize feasibility assessment models."""
        return {
            InterventionType.SETPOINT_ADJUSTMENT: {
                "base_rating": FeasibilityRating.EASY,
                "base_score": 0.9,
                "requires_downtime": False,
            },
            InterventionType.EQUIPMENT_INSPECTION: {
                "base_rating": FeasibilityRating.EASY,
                "base_score": 0.85,
                "requires_downtime": False,
            },
            InterventionType.VALVE_ADJUSTMENT: {
                "base_rating": FeasibilityRating.MODERATE,
                "base_score": 0.75,
                "requires_downtime": False,
            },
            InterventionType.MAINTENANCE_TASK: {
                "base_rating": FeasibilityRating.MODERATE,
                "base_score": 0.6,
                "requires_downtime": True,
            },
            InterventionType.OPERATIONAL_CHANGE: {
                "base_rating": FeasibilityRating.MODERATE,
                "base_score": 0.7,
                "requires_downtime": False,
            },
            InterventionType.CONFIGURATION_CHANGE: {
                "base_rating": FeasibilityRating.DIFFICULT,
                "base_score": 0.5,
                "requires_downtime": False,
            },
            InterventionType.SAFETY_ACTION: {
                "base_rating": FeasibilityRating.COMPLEX,
                "base_score": 0.4,
                "requires_downtime": True,
            },
        }

    def set_operational_constraints(
        self,
        constraints: Dict[str, Any],
    ) -> None:
        """Set operational constraints for intervention generation."""
        self._operational_constraints = constraints
        logger.info(f"Set operational constraints: {list(constraints.keys())}")

    def generate_interventions(
        self,
        root_causes: List[RankedCause],
        constraints: Optional[Dict[str, Any]] = None,
        max_interventions: int = 5,
    ) -> List[Intervention]:
        """
        Generate candidate interventions from root causes.

        Args:
            root_causes: Ranked list of potential causes
            constraints: Operational constraints
            max_interventions: Maximum interventions to generate

        Returns:
            List of candidate interventions
        """
        constraints = constraints or self._operational_constraints
        interventions = []

        for cause in root_causes[:max_interventions]:
            # Determine intervention type based on cause
            intervention = self._generate_intervention_for_cause(cause, constraints)
            if intervention:
                interventions.append(intervention)

        logger.info(f"Generated {len(interventions)} interventions from {len(root_causes)} causes")
        return interventions

    def _generate_intervention_for_cause(
        self,
        cause: RankedCause,
        constraints: Dict[str, Any],
    ) -> Optional[Intervention]:
        """Generate intervention for a specific cause."""
        intervention_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Get node information
        node = self.graph.get_node(cause.cause_node_id)

        # Select intervention template
        template = self._select_template(cause, node)
        if not template:
            return None

        # Build intervention
        intervention = Intervention(
            intervention_id=intervention_id,
            timestamp=timestamp,
            intervention_type=template["type"],
            name=template["name"],
            description=f"{template['description']} to address {cause.cause_name}",
            detailed_instructions=template["instructions"],
            target_equipment=cause.cause_name,
            action=cause.suggested_action or template.get("action", "Adjust"),
            addresses_cause=cause.cause_id,
            cause_probability=cause.probability,
            urgency=self._determine_urgency(cause),
            verification_steps=template.get("verification", []),
        )

        # Assess feasibility
        intervention.feasibility = self._assess_feasibility(
            intervention, constraints
        )

        # Add safety warnings based on node type
        if node and node.node_type in [NodeType.PRV, NodeType.BOILER]:
            intervention.safety_warnings.append(
                "Verify safety interlocks before making changes"
            )

        self._interventions[intervention_id] = intervention
        return intervention

    def _select_template(
        self,
        cause: RankedCause,
        node: Optional[CausalNode],
    ) -> Optional[Dict[str, Any]]:
        """Select appropriate intervention template."""
        cause_name = cause.cause_name.lower()
        cause_node = cause.cause_node_id.lower()

        # Match by cause type
        if "pressure" in cause_name or "pressure" in cause_node:
            return self._intervention_templates.get("adjust_pressure_setpoint")
        elif "temperature" in cause_name or "temp" in cause_node:
            return self._intervention_templates.get("adjust_temperature_setpoint")
        elif "prv" in cause_name or "prv" in cause_node:
            return self._intervention_templates.get("adjust_prv_setpoint")
        elif "trap" in cause_name or "trap" in cause_node:
            return self._intervention_templates.get("inspect_steam_trap")
        elif "spray" in cause_name or "desuperheater" in cause_node:
            return self._intervention_templates.get("check_spray_water")
        elif "valve" in cause_name:
            return self._intervention_templates.get("adjust_control_valve")

        # Default based on node type
        if node:
            if node.is_controllable:
                return self._intervention_templates.get("adjust_pressure_setpoint")
            else:
                return self._intervention_templates.get("inspect_steam_trap")

        return None

    def _determine_urgency(self, cause: RankedCause) -> Urgency:
        """Determine urgency based on cause."""
        if cause.probability > 0.8:
            return Urgency.URGENT
        elif cause.probability > 0.5:
            return Urgency.SOON
        elif cause.probability > 0.3:
            return Urgency.SCHEDULED
        else:
            return Urgency.OPPORTUNISTIC

    def _assess_feasibility(
        self,
        intervention: Intervention,
        constraints: Dict[str, Any],
    ) -> FeasibilityAssessment:
        """Assess feasibility of an intervention."""
        model = self._feasibility_models.get(
            intervention.intervention_type,
            {"base_rating": FeasibilityRating.MODERATE, "base_score": 0.5}
        )

        rating = model["base_rating"]
        score = model["base_score"]

        # Adjust based on constraints
        blockers = []
        prerequisites = []
        safety_considerations = []

        # Check time constraints
        if constraints.get("no_changes_until"):
            if intervention.urgency != Urgency.IMMEDIATE:
                blockers.append(f"Changes frozen until {constraints['no_changes_until']}")
                score *= 0.5

        # Check personnel availability
        template = self._intervention_templates.get(intervention.name.lower().replace(" ", "_"), {})
        required_personnel = template.get("personnel", 1)
        available_personnel = constraints.get("available_personnel", 10)
        if required_personnel > available_personnel:
            blockers.append(f"Insufficient personnel ({required_personnel} needed, {available_personnel} available)")
            score *= 0.3

        # Check downtime requirements
        requires_downtime = model.get("requires_downtime", False)
        if requires_downtime and not constraints.get("downtime_available", True):
            blockers.append("Requires downtime but none scheduled")
            score *= 0.4

        # Safety considerations for certain equipment
        if "prv" in intervention.target_equipment.lower():
            safety_considerations.append("PRV adjustment requires verification of relief capacity")
        if "boiler" in intervention.target_equipment.lower():
            safety_considerations.append("Boiler changes require operator presence")

        # Determine final rating
        if blockers:
            rating = FeasibilityRating.INFEASIBLE if score < 0.3 else FeasibilityRating.DIFFICULT
        elif score < 0.5:
            rating = FeasibilityRating.DIFFICULT
        elif score < 0.7:
            rating = FeasibilityRating.MODERATE

        return FeasibilityAssessment(
            rating=rating,
            score=score,
            requires_equipment=template.get("requires_equipment", False),
            requires_downtime=requires_downtime,
            requires_personnel=required_personnel,
            estimated_duration_minutes=template.get("duration_minutes", 15),
            prerequisites=prerequisites,
            blockers=blockers,
            safety_considerations=safety_considerations,
            can_execute_now=len(blockers) == 0,
        )

    def evaluate_intervention_impact(
        self,
        intervention: Intervention,
        causal_model: Optional[CausalGraph] = None,
        current_state: Optional[Dict[str, float]] = None,
        target_metric: Optional[str] = None,
    ) -> ImpactEstimate:
        """
        Evaluate the expected impact of an intervention.

        Args:
            intervention: The intervention to evaluate
            causal_model: Optional causal graph override
            current_state: Current system state
            target_metric: Metric to evaluate impact on

        Returns:
            ImpactEstimate with predicted effects
        """
        impact_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        current_state = current_state or {}

        # Build intervention for counterfactual
        if intervention.target_parameter and intervention.target_value is not None:
            cf_intervention = {intervention.target_parameter: intervention.target_value}
        else:
            # Use cause node as proxy
            cf_intervention = {
                intervention.addresses_cause or intervention.target_equipment:
                intervention.target_value or 0
            }

        # Get target metric
        if not target_metric:
            # Default to efficiency or first descendant
            target_metric = "efficiency"

        # Compute counterfactual
        cf_result = self.cf_engine.compute_counterfactual(
            cf_intervention, current_state, causal_model
        )

        # Build impact estimate
        impact = ImpactEstimate(
            impact_id=impact_id,
            intervention_id=intervention.intervention_id,
            timestamp=timestamp,
            target_metric=cf_result.target_name,
            current_value=cf_result.factual_value,
            predicted_value=cf_result.counterfactual_value,
            improvement=cf_result.effect_size,
            improvement_percent=cf_result.effect_percent,
            confidence=cf_result.path_strength,
            uncertainty_lower=cf_result.uncertainty.lower_bound,
            uncertainty_upper=cf_result.uncertainty.upper_bound,
            time_to_effect_minutes=self._estimate_time_to_effect(intervention),
        )

        # Identify side effects
        for affected in cf_result.affected_nodes[1:]:  # Skip primary target
            if abs(affected.change_percent) > 5:
                impact.side_effects.append({
                    "metric": affected.node_id,
                    "change": affected.change,
                    "change_percent": affected.change_percent,
                })

        # Identify risk factors
        if not cf_result.is_valid:
            impact.risk_factors.extend(cf_result.validity_notes)

        # Store
        intervention.expected_impact = impact

        return impact

    def _estimate_time_to_effect(self, intervention: Intervention) -> int:
        """Estimate time until intervention takes effect."""
        type_times = {
            InterventionType.SETPOINT_ADJUSTMENT: 5,
            InterventionType.VALVE_ADJUSTMENT: 10,
            InterventionType.EQUIPMENT_INSPECTION: 30,
            InterventionType.MAINTENANCE_TASK: 120,
            InterventionType.OPERATIONAL_CHANGE: 15,
        }
        return type_times.get(intervention.intervention_type, 15)

    def rank_interventions_by_feasibility(
        self,
        interventions: List[Intervention],
        operational_context: Optional[Dict[str, Any]] = None,
    ) -> RankedInterventions:
        """
        Rank interventions by feasibility and expected benefit.

        Args:
            interventions: List of candidate interventions
            operational_context: Current operational context

        Returns:
            RankedInterventions with prioritized list
        """
        ranking_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        context = operational_context or {}

        # Score each intervention
        scored = []
        for intervention in interventions:
            score = self._compute_ranking_score(intervention, context)
            scored.append((intervention, score))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update priorities
        for rank, (intervention, score) in enumerate(scored, 1):
            intervention.priority = rank

        ranked_interventions = [i for i, s in scored]

        # Determine implementation order (considering dependencies)
        implementation_order = self._determine_implementation_order(ranked_interventions)

        # Generate summary
        summary = self._generate_ranking_summary(ranked_interventions)

        result = RankedInterventions(
            ranking_id=ranking_id,
            timestamp=timestamp,
            deviation_id=interventions[0].addresses_cause if interventions else "",
            interventions=ranked_interventions,
            total_generated=len(interventions),
            top_recommendation=ranked_interventions[0] if ranked_interventions else None,
            summary=summary,
            implementation_order=implementation_order,
            constraints_applied=list(context.keys()),
        )

        self._rankings[ranking_id] = result
        logger.info(f"Ranked {len(ranked_interventions)} interventions")

        return result

    def _compute_ranking_score(
        self,
        intervention: Intervention,
        context: Dict[str, Any],
    ) -> float:
        """Compute ranking score for an intervention."""
        score = 0.0

        # Feasibility score (0-40 points)
        if intervention.feasibility:
            score += intervention.feasibility.score * 40
        else:
            score += 20  # Default if not assessed

        # Impact score (0-30 points)
        if intervention.expected_impact:
            impact_score = min(30, abs(intervention.expected_impact.improvement_percent))
            score += impact_score
        else:
            score += 10  # Default if not evaluated

        # Cause probability score (0-20 points)
        score += intervention.cause_probability * 20

        # Urgency bonus (0-10 points)
        urgency_scores = {
            Urgency.IMMEDIATE: 10,
            Urgency.URGENT: 8,
            Urgency.SOON: 5,
            Urgency.SCHEDULED: 2,
            Urgency.OPPORTUNISTIC: 0,
        }
        score += urgency_scores.get(intervention.urgency, 0)

        # Context adjustments
        if context.get("prefer_non_invasive"):
            if intervention.intervention_type == InterventionType.SETPOINT_ADJUSTMENT:
                score += 5
            elif intervention.intervention_type == InterventionType.MAINTENANCE_TASK:
                score -= 5

        if context.get("critical_production"):
            if intervention.feasibility and intervention.feasibility.requires_downtime:
                score -= 20

        return score

    def _determine_implementation_order(
        self,
        interventions: List[Intervention],
    ) -> List[str]:
        """Determine order to implement interventions."""
        # Simple: order by priority, but group inspections before actions
        inspections = [
            i for i in interventions
            if i.intervention_type == InterventionType.EQUIPMENT_INSPECTION
        ]
        actions = [
            i for i in interventions
            if i.intervention_type != InterventionType.EQUIPMENT_INSPECTION
        ]

        order = [i.intervention_id for i in inspections[:2]]  # Max 2 inspections first
        order.extend(i.intervention_id for i in actions)

        return order

    def _generate_ranking_summary(
        self,
        interventions: List[Intervention],
    ) -> str:
        """Generate summary of ranked interventions."""
        if not interventions:
            return "No interventions recommended"

        top = interventions[0]
        summary_parts = [
            f"Top recommendation: {top.name}",
            f"Target: {top.target_equipment}",
        ]

        if top.feasibility:
            summary_parts.append(f"Feasibility: {top.feasibility.rating.value}")

        if top.expected_impact:
            summary_parts.append(
                f"Expected improvement: {top.expected_impact.improvement_percent:.1f}%"
            )

        return " | ".join(summary_parts)

    def get_intervention(self, intervention_id: str) -> Optional[Intervention]:
        """Get intervention by ID."""
        return self._interventions.get(intervention_id)

    def get_ranking(self, ranking_id: str) -> Optional[RankedInterventions]:
        """Get ranking by ID."""
        return self._rankings.get(ranking_id)
