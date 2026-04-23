"""
GL-003 UNIFIEDSTEAM - Explanation Generator

Generates human-readable explanations that combine:
- Physics-based explanations (thermodynamic traces)
- ML-based explanations (SHAP/LIME)
- Engineering terminology mapping

Output formats:
- User explanations (technical users)
- Engineering explanations (operators/engineers)
- Operator briefings (shift handoff)

Maps technical drivers to engineering terms:
  "superheat increased because header pressure dropped and PRV opened"
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging
import uuid

logger = logging.getLogger(__name__)


class AudienceLevel(Enum):
    """Target audience for explanation."""
    TECHNICAL = "technical"  # Data scientists, model developers
    ENGINEERING = "engineering"  # Process engineers
    OPERATOR = "operator"  # Control room operators
    MANAGEMENT = "management"  # Plant management


class RecommendationType(Enum):
    """Types of recommendations."""
    SETPOINT_CHANGE = "setpoint_change"
    MAINTENANCE = "maintenance"
    INSPECTION = "inspection"
    OPERATIONAL = "operational"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"


@dataclass
class UserExplanation:
    """User-facing explanation combining physics and ML traces."""
    explanation_id: str
    timestamp: datetime
    recommendation_id: str
    recommendation_type: RecommendationType

    # Summary
    headline: str
    summary: str

    # Physics explanation
    physics_summary: str
    thermodynamic_path: str
    active_constraints: List[str]

    # ML explanation
    model_summary: str
    key_drivers: List[Dict[str, Any]]  # Driver name, value, impact

    # Combined narrative
    full_narrative: str

    # Supporting data
    confidence_score: float = 0.0
    uncertainty_note: str = ""
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "explanation_id": self.explanation_id,
            "timestamp": self.timestamp.isoformat(),
            "recommendation_id": self.recommendation_id,
            "recommendation_type": self.recommendation_type.value,
            "headline": self.headline,
            "summary": self.summary,
            "physics_summary": self.physics_summary,
            "thermodynamic_path": self.thermodynamic_path,
            "active_constraints": self.active_constraints,
            "model_summary": self.model_summary,
            "key_drivers": self.key_drivers,
            "full_narrative": self.full_narrative,
            "confidence_score": self.confidence_score,
            "uncertainty_note": self.uncertainty_note,
            "supporting_evidence": self.supporting_evidence,
        }


@dataclass
class EngineeringExplanation:
    """Explanation in engineering terms for operators and engineers."""
    explanation_id: str
    timestamp: datetime
    recommendation_id: str

    # Engineering summary
    situation: str  # What is happening
    cause: str  # Why it is happening
    action: str  # What to do
    expected_result: str  # What will happen

    # Engineering terminology
    process_variables: Dict[str, str]  # Variable -> current state description
    equipment_status: Dict[str, str]  # Equipment -> status
    control_loop_status: List[str]  # Active control loops

    # Physical relationships
    cause_effect_chain: List[str]  # Sequence of cause -> effect
    constraint_explanations: Dict[str, str]  # Constraint -> why it matters

    # Actionable items
    immediate_actions: List[str]
    verification_steps: List[str]
    monitoring_points: List[str]

    def to_dict(self) -> Dict:
        return {
            "explanation_id": self.explanation_id,
            "timestamp": self.timestamp.isoformat(),
            "recommendation_id": self.recommendation_id,
            "situation": self.situation,
            "cause": self.cause,
            "action": self.action,
            "expected_result": self.expected_result,
            "process_variables": self.process_variables,
            "equipment_status": self.equipment_status,
            "control_loop_status": self.control_loop_status,
            "cause_effect_chain": self.cause_effect_chain,
            "constraint_explanations": self.constraint_explanations,
            "immediate_actions": self.immediate_actions,
            "verification_steps": self.verification_steps,
            "monitoring_points": self.monitoring_points,
        }


@dataclass
class OperatorBriefing:
    """Operator briefing combining multiple recommendations."""
    briefing_id: str
    timestamp: datetime
    shift: str  # e.g., "Day Shift 2024-01-15"
    prepared_by: str

    # System overview
    system_status: str  # "Normal", "Caution", "Alert"
    key_metrics: Dict[str, float]  # Metric -> current value

    # Active recommendations
    active_recommendations: List[Dict[str, Any]]
    priority_actions: List[str]

    # System summary by area
    area_summaries: Dict[str, str]  # Area -> summary

    # Trend observations
    improving_trends: List[str]
    concerning_trends: List[str]

    # Handoff notes
    items_to_watch: List[str]
    pending_actions: List[str]
    recent_changes: List[str]

    # Confidence and notes
    overall_confidence: float = 0.0
    disclaimers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "briefing_id": self.briefing_id,
            "timestamp": self.timestamp.isoformat(),
            "shift": self.shift,
            "prepared_by": self.prepared_by,
            "system_status": self.system_status,
            "key_metrics": self.key_metrics,
            "active_recommendations": self.active_recommendations,
            "priority_actions": self.priority_actions,
            "area_summaries": self.area_summaries,
            "improving_trends": self.improving_trends,
            "concerning_trends": self.concerning_trends,
            "items_to_watch": self.items_to_watch,
            "pending_actions": self.pending_actions,
            "recent_changes": self.recent_changes,
            "overall_confidence": self.overall_confidence,
            "disclaimers": self.disclaimers,
        }


class ExplanationGenerator:
    """
    Generates human-readable explanations combining physics and ML traces.

    Maps technical drivers to engineering terms:
    - "SHAP contribution 0.25 from pressure_delta" ->
      "Header pressure drop caused superheat increase"

    Supports multiple audience levels:
    - Technical: Full detail with model metrics
    - Engineering: Process-focused with cause-effect
    - Operator: Action-focused with simple language
    """

    def __init__(
        self,
        agent_id: str = "GL-003",
    ) -> None:
        self.agent_id = agent_id

        # Technical to engineering term mapping
        self._term_mappings = self._initialize_term_mappings()

        # Cause-effect templates
        self._cause_effect_templates = self._initialize_cause_effect_templates()

        # Action templates
        self._action_templates = self._initialize_action_templates()

        # Cached explanations
        self._user_explanations: Dict[str, UserExplanation] = {}
        self._engineering_explanations: Dict[str, EngineeringExplanation] = {}
        self._operator_briefings: Dict[str, OperatorBriefing] = {}

        logger.info(f"ExplanationGenerator initialized: {agent_id}")

    def _initialize_term_mappings(self) -> Dict[str, Dict[str, str]]:
        """Map technical terms to engineering terms."""
        return {
            # Feature names to engineering terms
            "features": {
                "header_pressure_psig": "header pressure",
                "steam_temperature_f": "steam temperature",
                "superheat_f": "superheat",
                "steam_flow_klb_hr": "steam flow",
                "spray_water_flow_gpm": "spray water flow",
                "prv_position_pct": "PRV opening",
                "desuperheater_outlet_temp_f": "desuperheater outlet temperature",
                "temp_differential_f": "temperature drop across trap",
                "subcooling_f": "condensate subcooling",
                "inlet_temp_f": "trap inlet temperature",
                "outlet_temp_f": "trap outlet temperature",
                "differential_pressure_psi": "pressure drop",
                "operating_hours": "accumulated operating hours",
                "days_since_inspection": "days since last inspection",
            },
            # Direction terms
            "directions": {
                "positive": "increasing",
                "negative": "decreasing",
                "high": "above setpoint",
                "low": "below setpoint",
            },
            # Equipment names
            "equipment": {
                "prv": "Pressure Reducing Valve (PRV)",
                "desuperheater": "Desuperheater spray station",
                "header_hp": "High-Pressure Steam Header",
                "header_mp": "Medium-Pressure Steam Header",
                "header_lp": "Low-Pressure Steam Header",
                "boiler": "Steam Boiler",
                "turbine": "Steam Turbine",
                "trap": "Steam Trap",
                "condensate": "Condensate Return System",
            },
        }

    def _initialize_cause_effect_templates(self) -> Dict[str, str]:
        """Initialize cause-effect relationship templates."""
        return {
            # Pressure-related
            "pressure_drop_superheat": (
                "Header pressure dropped from {from_pressure:.0f} to {to_pressure:.0f} psig, "
                "causing superheat to increase by {superheat_change:.0f} F "
                "as steam expanded through the PRV"
            ),
            "pressure_rise_flow": (
                "Header pressure rose to {pressure:.0f} psig, "
                "reducing steam flow to downstream users"
            ),
            "prv_opened": (
                "PRV opened to {position:.0f}% as header pressure exceeded setpoint, "
                "causing downstream superheat increase"
            ),
            # Temperature-related
            "superheat_spray": (
                "Superheat increased to {superheat:.0f} F, "
                "requiring {spray_flow:.1f} gpm spray water to achieve target temperature"
            ),
            "temp_high": (
                "Steam temperature at {location} reached {temp:.0f} F, "
                "{effect}"
            ),
            # Flow-related
            "flow_imbalance": (
                "Steam flow imbalance detected: {supply:.1f} klb/hr supply vs "
                "{demand:.1f} klb/hr demand, causing {effect}"
            ),
            # Trap-related
            "trap_blowthrough": (
                "Steam trap at {location} showing reduced subcooling ({subcooling:.1f} F), "
                "indicating possible steam blow-through"
            ),
            "trap_blocked": (
                "Steam trap at {location} showing high temperature differential ({temp_diff:.0f} F), "
                "indicating possible condensate backup"
            ),
        }

    def _initialize_action_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize action templates by recommendation type."""
        return {
            "setpoint_change": {
                "template": "Adjust {parameter} from {current:.1f} to {target:.1f} {unit}",
                "verification": "Verify {parameter} stabilizes at new setpoint within {time} minutes",
            },
            "maintenance": {
                "template": "Schedule maintenance for {equipment}: {action}",
                "verification": "Confirm work order created and scheduled",
            },
            "inspection": {
                "template": "Inspect {equipment} for {issue}",
                "verification": "Document inspection findings and update asset records",
            },
            "operational": {
                "template": "{action} to {objective}",
                "verification": "Monitor {metric} for expected change",
            },
            "safety": {
                "template": "SAFETY: {action} immediately",
                "verification": "Confirm safety system status and log event",
            },
        }

    def generate_recommendation_explanation(
        self,
        recommendation: Dict[str, Any],
        physics_trace: Optional[Dict[str, Any]] = None,
        model_trace: Optional[Dict[str, Any]] = None,
    ) -> UserExplanation:
        """
        Generate complete explanation combining physics and ML traces.

        Args:
            recommendation: The recommendation to explain
            physics_trace: PhysicsTrace data (or dict representation)
            model_trace: SHAP/LIME explanation data

        Returns:
            UserExplanation with combined narrative
        """
        explanation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        recommendation_id = recommendation.get("recommendation_id", str(uuid.uuid4())[:8])

        # Determine recommendation type
        rec_type = self._classify_recommendation(recommendation)

        # Generate headline
        headline = self._generate_headline(recommendation, rec_type)

        # Process physics trace
        physics_summary, thermo_path, constraints = self._process_physics_trace(
            physics_trace
        )

        # Process model trace
        model_summary, key_drivers = self._process_model_trace(model_trace)

        # Generate combined summary
        summary = self._generate_summary(
            recommendation, physics_summary, model_summary
        )

        # Generate full narrative
        full_narrative = self._generate_full_narrative(
            recommendation, rec_type, physics_trace, model_trace
        )

        # Extract confidence
        confidence = recommendation.get(
            "confidence", model_trace.get("confidence", 0.8) if model_trace else 0.8
        )

        # Generate uncertainty note
        uncertainty_note = self._generate_uncertainty_note(confidence, constraints)

        # Gather supporting evidence
        supporting_evidence = self._gather_evidence(
            recommendation, physics_trace, model_trace
        )

        explanation = UserExplanation(
            explanation_id=explanation_id,
            timestamp=timestamp,
            recommendation_id=recommendation_id,
            recommendation_type=rec_type,
            headline=headline,
            summary=summary,
            physics_summary=physics_summary,
            thermodynamic_path=thermo_path,
            active_constraints=constraints,
            model_summary=model_summary,
            key_drivers=key_drivers,
            full_narrative=full_narrative,
            confidence_score=confidence,
            uncertainty_note=uncertainty_note,
            supporting_evidence=supporting_evidence,
        )

        self._user_explanations[explanation_id] = explanation
        logger.info(f"Created user explanation: {explanation_id}")

        return explanation

    def _classify_recommendation(
        self,
        recommendation: Dict[str, Any],
    ) -> RecommendationType:
        """Classify recommendation type."""
        rec_text = str(recommendation).lower()

        if "setpoint" in rec_text or "adjust" in rec_text:
            return RecommendationType.SETPOINT_CHANGE
        elif "maintenance" in rec_text or "repair" in rec_text:
            return RecommendationType.MAINTENANCE
        elif "inspect" in rec_text or "check" in rec_text:
            return RecommendationType.INSPECTION
        elif "safety" in rec_text or "alarm" in rec_text:
            return RecommendationType.SAFETY
        elif "efficiency" in rec_text or "optimize" in rec_text:
            return RecommendationType.EFFICIENCY
        else:
            return RecommendationType.OPERATIONAL

    def _generate_headline(
        self,
        recommendation: Dict[str, Any],
        rec_type: RecommendationType,
    ) -> str:
        """Generate attention-grabbing headline."""
        asset = recommendation.get("affected_asset", "Steam System")
        action = recommendation.get("action", "Action Required")

        headlines = {
            RecommendationType.SETPOINT_CHANGE: f"Setpoint Adjustment Recommended for {asset}",
            RecommendationType.MAINTENANCE: f"Maintenance Required: {asset}",
            RecommendationType.INSPECTION: f"Inspection Needed: {asset}",
            RecommendationType.SAFETY: f"SAFETY: Immediate Action Required - {asset}",
            RecommendationType.EFFICIENCY: f"Efficiency Improvement Opportunity: {asset}",
            RecommendationType.OPERATIONAL: f"Operational Recommendation: {asset}",
        }

        return headlines.get(rec_type, f"Recommendation for {asset}")

    def _process_physics_trace(
        self,
        physics_trace: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, List[str]]:
        """Process physics trace into human-readable components."""
        if not physics_trace:
            return ("No physics trace available", "", [])

        # Extract summary
        summary = physics_trace.get("path_summary", "")
        if not summary:
            states = physics_trace.get("states", [])
            if states:
                summary = f"Thermodynamic path through {len(states)} states"

        # Extract thermodynamic path
        thermo_path = physics_trace.get("technical_narrative", "")
        if not thermo_path:
            transitions = physics_trace.get("transitions", [])
            if transitions:
                path_parts = [
                    t.get("physical_description", "") for t in transitions
                ]
                thermo_path = " -> ".join(filter(None, path_parts))

        # Extract active constraints
        constraints = []
        for constraint in physics_trace.get("active_constraints", []):
            if isinstance(constraint, dict):
                constraints.append(constraint.get("constraint_name", str(constraint)))
            else:
                constraints.append(str(constraint))

        return (summary, thermo_path, constraints)

    def _process_model_trace(
        self,
        model_trace: Optional[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Process model trace into human-readable components."""
        if not model_trace:
            return ("No model explanation available", [])

        # Extract summary
        summary = model_trace.get("summary_text", model_trace.get("explanation_text", ""))

        # Extract key drivers
        key_drivers = []
        contributions = model_trace.get("contributions", model_trace.get("feature_weights", []))

        for contrib in contributions[:5]:
            if isinstance(contrib, dict):
                feature = contrib.get("feature_name", "")
                value = contrib.get("feature_value", 0)
                impact = contrib.get("shap_value", contrib.get("contribution", 0))
                direction = contrib.get("direction", "positive" if impact > 0 else "negative")

                # Map to engineering term
                eng_term = self._term_mappings["features"].get(feature, feature)

                key_drivers.append({
                    "driver": eng_term,
                    "value": value,
                    "impact": impact,
                    "direction": self._term_mappings["directions"].get(direction, direction),
                })

        return (summary, key_drivers)

    def _generate_summary(
        self,
        recommendation: Dict[str, Any],
        physics_summary: str,
        model_summary: str,
    ) -> str:
        """Generate combined summary."""
        parts = []

        action = recommendation.get("action", recommendation.get("description", ""))
        if action:
            parts.append(f"Recommendation: {action}")

        if physics_summary:
            parts.append(f"Physics basis: {physics_summary}")

        if model_summary:
            parts.append(f"Model insight: {model_summary}")

        return " | ".join(parts)

    def _generate_full_narrative(
        self,
        recommendation: Dict[str, Any],
        rec_type: RecommendationType,
        physics_trace: Optional[Dict[str, Any]],
        model_trace: Optional[Dict[str, Any]],
    ) -> str:
        """Generate full narrative explanation."""
        lines = []

        # Introduction
        asset = recommendation.get("affected_asset", "the steam system")
        lines.append(f"Analysis of {asset}:")
        lines.append("")

        # Current situation
        lines.append("Current Situation:")
        if physics_trace:
            drivers = physics_trace.get("primary_drivers", [])
            for driver in drivers[:3]:
                if isinstance(driver, dict):
                    signal = driver.get("signal", "")
                    value = driver.get("value", "")
                    desc = driver.get("description", "")
                    lines.append(f"  - {desc}: {signal} = {value}")

        # Root cause (from model)
        if model_trace:
            lines.append("")
            lines.append("Analysis indicates:")
            top_features = model_trace.get("top_positive_features", [])
            for feature in top_features[:2]:
                eng_term = self._term_mappings["features"].get(feature, feature)
                lines.append(f"  - {eng_term} is contributing to the observed condition")

        # Recommendation
        lines.append("")
        lines.append("Recommended Action:")
        action = recommendation.get("action", recommendation.get("description", "No specific action"))
        lines.append(f"  {action}")

        # Expected outcome
        if recommendation.get("expected_outcome"):
            lines.append("")
            lines.append(f"Expected Outcome: {recommendation['expected_outcome']}")

        return "\n".join(lines)

    def _generate_uncertainty_note(
        self,
        confidence: float,
        constraints: List[str],
    ) -> str:
        """Generate note about uncertainty."""
        if confidence >= 0.9:
            return "High confidence recommendation based on strong evidence."
        elif confidence >= 0.7:
            note = "Moderate confidence recommendation."
            if constraints:
                note += f" Constrained by: {', '.join(constraints[:2])}."
            return note
        else:
            return "Lower confidence recommendation. Consider additional verification before action."

    def _gather_evidence(
        self,
        recommendation: Dict[str, Any],
        physics_trace: Optional[Dict[str, Any]],
        model_trace: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Gather supporting evidence for explanation."""
        evidence = []

        # From recommendation
        if recommendation.get("supporting_data"):
            evidence.append({
                "source": "recommendation",
                "data": recommendation["supporting_data"],
            })

        # From physics trace
        if physics_trace:
            if physics_trace.get("balance_residuals"):
                evidence.append({
                    "source": "physics_balance",
                    "data": physics_trace["balance_residuals"],
                })

        # From model trace
        if model_trace:
            if model_trace.get("local_model_r2"):
                evidence.append({
                    "source": "model_fit",
                    "data": {"r2": model_trace["local_model_r2"]},
                })

        return evidence

    def translate_to_engineering_terms(
        self,
        technical_explanation: Dict[str, Any],
    ) -> EngineeringExplanation:
        """
        Translate technical explanation to engineering terms.

        Maps:
        - "SHAP contribution" -> "influence on prediction"
        - Feature names -> process variable names
        - Model outputs -> operational meaning
        """
        explanation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        recommendation_id = technical_explanation.get("recommendation_id", "")

        # Generate Situation
        situation = self._generate_situation(technical_explanation)

        # Generate Cause
        cause = self._generate_cause(technical_explanation)

        # Generate Action
        action = self._generate_action(technical_explanation)

        # Generate Expected Result
        expected_result = self._generate_expected_result(technical_explanation)

        # Map process variables
        process_variables = self._map_process_variables(technical_explanation)

        # Map equipment status
        equipment_status = self._map_equipment_status(technical_explanation)

        # Control loop status
        control_loops = self._identify_control_loops(technical_explanation)

        # Build cause-effect chain
        cause_effect_chain = self._build_cause_effect_chain(technical_explanation)

        # Constraint explanations
        constraint_explanations = self._explain_constraints(technical_explanation)

        # Generate actions
        immediate_actions = self._generate_immediate_actions(technical_explanation)
        verification_steps = self._generate_verification_steps(technical_explanation)
        monitoring_points = self._generate_monitoring_points(technical_explanation)

        explanation = EngineeringExplanation(
            explanation_id=explanation_id,
            timestamp=timestamp,
            recommendation_id=recommendation_id,
            situation=situation,
            cause=cause,
            action=action,
            expected_result=expected_result,
            process_variables=process_variables,
            equipment_status=equipment_status,
            control_loop_status=control_loops,
            cause_effect_chain=cause_effect_chain,
            constraint_explanations=constraint_explanations,
            immediate_actions=immediate_actions,
            verification_steps=verification_steps,
            monitoring_points=monitoring_points,
        )

        self._engineering_explanations[explanation_id] = explanation
        logger.info(f"Created engineering explanation: {explanation_id}")

        return explanation

    def _generate_situation(self, data: Dict[str, Any]) -> str:
        """Generate situation description in engineering terms."""
        asset = data.get("affected_asset", "Steam system")
        status = data.get("current_status", "requires attention")
        return f"{asset} {status}"

    def _generate_cause(self, data: Dict[str, Any]) -> str:
        """Generate cause description in engineering terms."""
        drivers = data.get("key_drivers", data.get("primary_drivers", []))
        if drivers:
            if isinstance(drivers[0], dict):
                causes = [d.get("driver", str(d)) for d in drivers[:2]]
            else:
                causes = drivers[:2]
            return f"Driven by changes in {', '.join(causes)}"
        return "Root cause analysis in progress"

    def _generate_action(self, data: Dict[str, Any]) -> str:
        """Generate action description."""
        return data.get("action", data.get("description", "Monitor and assess"))

    def _generate_expected_result(self, data: Dict[str, Any]) -> str:
        """Generate expected result description."""
        return data.get("expected_outcome", "Improved system performance")

    def _map_process_variables(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Map process variables to engineering descriptions."""
        variables = {}
        current_values = data.get("current_values", data.get("inputs", {}))

        for var, value in current_values.items():
            eng_name = self._term_mappings["features"].get(var, var)
            if isinstance(value, (int, float)):
                variables[eng_name] = f"{value:.1f}"
            else:
                variables[eng_name] = str(value)

        return variables

    def _map_equipment_status(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Map equipment to status descriptions."""
        status = {}
        equipment = data.get("equipment", data.get("affected_equipment", []))

        for equip in equipment:
            eng_name = self._term_mappings["equipment"].get(equip, equip)
            status[eng_name] = "Active"

        if not status:
            status["Steam System"] = "Operating"

        return status

    def _identify_control_loops(self, data: Dict[str, Any]) -> List[str]:
        """Identify relevant control loops."""
        loops = []

        features = list(data.get("current_values", {}).keys())
        for feature in features:
            if "pressure" in feature.lower():
                loops.append("Pressure control loop")
            elif "temp" in feature.lower():
                loops.append("Temperature control loop")
            elif "flow" in feature.lower():
                loops.append("Flow control loop")

        return list(set(loops))[:3]

    def _build_cause_effect_chain(self, data: Dict[str, Any]) -> List[str]:
        """Build cause-effect chain in engineering terms."""
        chain = []
        drivers = data.get("key_drivers", [])

        if drivers and isinstance(drivers[0], dict):
            for driver in drivers[:3]:
                eng_term = driver.get("driver", "")
                direction = driver.get("direction", "")
                chain.append(f"{eng_term} {direction}")

        return chain

    def _explain_constraints(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Explain active constraints in engineering terms."""
        explanations = {}
        constraints = data.get("active_constraints", [])

        constraint_meanings = {
            "pressure_limit": "Equipment pressure rating must not be exceeded",
            "temperature_limit": "Material temperature limits apply",
            "superheat_minimum": "Minimum superheat prevents condensation in piping",
            "flow_limit": "Pipe/equipment capacity limits flow rate",
        }

        for constraint in constraints:
            if isinstance(constraint, str):
                for key, meaning in constraint_meanings.items():
                    if key in constraint.lower():
                        explanations[constraint] = meaning

        return explanations

    def _generate_immediate_actions(self, data: Dict[str, Any]) -> List[str]:
        """Generate list of immediate actions."""
        actions = []
        recommendation = data.get("action", "")

        if recommendation:
            actions.append(recommendation)

        if data.get("urgency") == "immediate":
            actions.insert(0, "Dispatch operator immediately")

        return actions or ["Monitor current conditions"]

    def _generate_verification_steps(self, data: Dict[str, Any]) -> List[str]:
        """Generate verification steps."""
        return [
            "Verify action taken matches recommendation",
            "Confirm system response within expected time",
            "Document results in operations log",
        ]

    def _generate_monitoring_points(self, data: Dict[str, Any]) -> List[str]:
        """Generate monitoring points."""
        points = []
        variables = data.get("current_values", {})

        for var in list(variables.keys())[:3]:
            eng_name = self._term_mappings["features"].get(var, var)
            points.append(f"Monitor {eng_name}")

        return points or ["Monitor system stability"]

    def generate_operator_briefing(
        self,
        recommendations: List[Dict[str, Any]],
        system_state: Dict[str, Any],
    ) -> OperatorBriefing:
        """
        Generate operator briefing combining multiple recommendations.

        Args:
            recommendations: List of active recommendations
            system_state: Current system state

        Returns:
            OperatorBriefing for shift handoff
        """
        briefing_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Determine shift
        hour = timestamp.hour
        if 6 <= hour < 14:
            shift = f"Day Shift {timestamp.strftime('%Y-%m-%d')}"
        elif 14 <= hour < 22:
            shift = f"Evening Shift {timestamp.strftime('%Y-%m-%d')}"
        else:
            shift = f"Night Shift {timestamp.strftime('%Y-%m-%d')}"

        # Determine system status
        system_status = self._determine_system_status(recommendations, system_state)

        # Extract key metrics
        key_metrics = self._extract_key_metrics(system_state)

        # Process recommendations
        active_recs = []
        priority_actions = []
        for rec in recommendations:
            active_recs.append({
                "id": rec.get("recommendation_id", ""),
                "action": rec.get("action", ""),
                "priority": rec.get("priority", "normal"),
                "asset": rec.get("affected_asset", ""),
            })
            if rec.get("priority") == "high":
                priority_actions.append(rec.get("action", ""))

        # Generate area summaries
        area_summaries = self._generate_area_summaries(system_state)

        # Identify trends
        improving, concerning = self._identify_trends(system_state)

        # Handoff items
        items_to_watch = self._generate_watch_items(recommendations)
        pending_actions = [r.get("action", "") for r in recommendations if r.get("status") == "pending"]
        recent_changes = system_state.get("recent_changes", [])

        # Calculate overall confidence
        if recommendations:
            confidences = [r.get("confidence", 0.8) for r in recommendations]
            overall_confidence = sum(confidences) / len(confidences)
        else:
            overall_confidence = 0.9

        # Standard disclaimers
        disclaimers = [
            "AI-generated recommendations require operator verification",
            "Safety systems remain under manual control",
        ]

        briefing = OperatorBriefing(
            briefing_id=briefing_id,
            timestamp=timestamp,
            shift=shift,
            prepared_by=self.agent_id,
            system_status=system_status,
            key_metrics=key_metrics,
            active_recommendations=active_recs,
            priority_actions=priority_actions,
            area_summaries=area_summaries,
            improving_trends=improving,
            concerning_trends=concerning,
            items_to_watch=items_to_watch,
            pending_actions=pending_actions,
            recent_changes=recent_changes,
            overall_confidence=overall_confidence,
            disclaimers=disclaimers,
        )

        self._operator_briefings[briefing_id] = briefing
        logger.info(f"Created operator briefing: {briefing_id}")

        return briefing

    def _determine_system_status(
        self,
        recommendations: List[Dict[str, Any]],
        system_state: Dict[str, Any],
    ) -> str:
        """Determine overall system status."""
        high_priority = any(r.get("priority") == "high" for r in recommendations)
        safety_issues = any(r.get("type") == "safety" for r in recommendations)

        if safety_issues:
            return "Alert"
        elif high_priority or len(recommendations) > 5:
            return "Caution"
        else:
            return "Normal"

    def _extract_key_metrics(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from system state."""
        key_metrics = {}
        metric_keys = [
            "header_pressure_psig",
            "steam_flow_klb_hr",
            "efficiency_percent",
            "trap_health_index",
        ]

        for key in metric_keys:
            if key in system_state:
                key_metrics[self._term_mappings["features"].get(key, key)] = system_state[key]

        return key_metrics

    def _generate_area_summaries(self, system_state: Dict[str, Any]) -> Dict[str, str]:
        """Generate summaries by system area."""
        return {
            "Steam Generation": system_state.get("boiler_status", "Normal operation"),
            "Distribution": system_state.get("header_status", "Pressures stable"),
            "Condensate Return": system_state.get("condensate_status", "Returns normal"),
        }

    def _identify_trends(
        self,
        system_state: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        """Identify improving and concerning trends."""
        improving = []
        concerning = []

        trends = system_state.get("trends", {})
        for metric, direction in trends.items():
            eng_name = self._term_mappings["features"].get(metric, metric)
            if direction == "improving":
                improving.append(f"{eng_name} improving")
            elif direction == "degrading":
                concerning.append(f"{eng_name} degrading")

        return improving, concerning

    def _generate_watch_items(
        self,
        recommendations: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate items to watch for next shift."""
        items = []
        for rec in recommendations[:5]:
            asset = rec.get("affected_asset", "System")
            items.append(f"Watch {asset} for response to recommendation")
        return items

    def get_user_explanation(
        self,
        explanation_id: str,
    ) -> Optional[UserExplanation]:
        """Get user explanation by ID."""
        return self._user_explanations.get(explanation_id)

    def get_engineering_explanation(
        self,
        explanation_id: str,
    ) -> Optional[EngineeringExplanation]:
        """Get engineering explanation by ID."""
        return self._engineering_explanations.get(explanation_id)

    def get_operator_briefing(
        self,
        briefing_id: str,
    ) -> Optional[OperatorBriefing]:
        """Get operator briefing by ID."""
        return self._operator_briefings.get(briefing_id)
