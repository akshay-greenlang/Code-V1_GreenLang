"""
GL-012 STEAMQUAL - Recommendation Explainer

Explains control recommendations for steam quality management including:
- Why a specific action was recommended
- Expected impact on quality parameters
- Alternative actions considered
- Confidence level and supporting evidence

All explanations are traceable to data and assumptions per playbook requirement.

Reference:
    - ASME PTC 19.11 Steam Quality
    - ISA-95 Enterprise-Control System Integration
    - API 560 Fired Heaters for General Refinery Service

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ActionType(Enum):
    """Types of control actions."""

    SETPOINT_CHANGE = "setpoint_change"      # Adjust a control setpoint
    VALVE_ADJUSTMENT = "valve_adjustment"     # Adjust valve position
    LOAD_CHANGE = "load_change"              # Change operating load
    EQUIPMENT_SWITCH = "equipment_switch"     # Switch to backup equipment
    MAINTENANCE = "maintenance"               # Maintenance action required
    OPERATOR_ALERT = "operator_alert"        # Alert for manual intervention
    NO_ACTION = "no_action"                   # Current state is optimal


class ActionPriority(Enum):
    """Priority levels for recommendations."""

    CRITICAL = "critical"      # Immediate action required
    HIGH = "high"              # Action within 1 hour
    MEDIUM = "medium"          # Action within 1 shift
    LOW = "low"                # Action at next opportunity
    ADVISORY = "advisory"      # Informational only


class ConfidenceLevel(Enum):
    """Confidence levels for recommendations."""

    VERY_HIGH = "very_high"    # > 90% confidence
    HIGH = "high"              # 75-90% confidence
    MEDIUM = "medium"          # 50-75% confidence
    LOW = "low"                # 25-50% confidence
    UNCERTAIN = "uncertain"    # < 25% confidence


class ImpactCategory(Enum):
    """Categories of impact from recommendations."""

    QUALITY = "quality"            # Steam quality impact
    EFFICIENCY = "efficiency"       # Energy efficiency impact
    SAFETY = "safety"              # Safety impact
    RELIABILITY = "reliability"     # Equipment reliability impact
    ENVIRONMENTAL = "environmental" # Environmental impact
    COST = "cost"                  # Operating cost impact


class ConstraintType(Enum):
    """Types of constraints on recommendations."""

    PHYSICAL = "physical"          # Physical limits
    SAFETY = "safety"              # Safety limits
    OPERATIONAL = "operational"    # Operational constraints
    EQUIPMENT = "equipment"        # Equipment limitations
    REGULATORY = "regulatory"      # Regulatory requirements


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExpectedImpact:
    """Expected impact of a recommended action."""

    impact_id: str
    category: ImpactCategory
    parameter: str
    unit: str

    # Current and expected values
    current_value: float
    expected_value: float
    delta: float
    delta_pct: float

    # Confidence
    confidence: ConfidenceLevel
    confidence_value: float  # 0-1

    # Time to realize impact
    time_to_impact_minutes: float

    # Description
    description: str = ""
    physics_basis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "impact_id": self.impact_id,
            "category": self.category.value,
            "parameter": self.parameter,
            "unit": self.unit,
            "current_value": self.current_value,
            "expected_value": self.expected_value,
            "delta": self.delta,
            "delta_pct": self.delta_pct,
            "confidence": self.confidence.value,
            "confidence_value": self.confidence_value,
            "time_to_impact_minutes": self.time_to_impact_minutes,
            "description": self.description,
            "physics_basis": self.physics_basis,
        }


@dataclass
class AlternativeAction:
    """An alternative action that was considered."""

    alternative_id: str
    action_type: ActionType
    description: str

    # Comparison to recommended action
    expected_quality_impact: float
    expected_efficiency_impact: float

    # Why not selected
    rejection_reason: str
    limitations: List[str]

    # If it could become preferred
    conditions_to_prefer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alternative_id": self.alternative_id,
            "action_type": self.action_type.value,
            "description": self.description,
            "expected_quality_impact": self.expected_quality_impact,
            "expected_efficiency_impact": self.expected_efficiency_impact,
            "rejection_reason": self.rejection_reason,
            "limitations": self.limitations,
            "conditions_to_prefer": self.conditions_to_prefer,
        }


@dataclass
class ActionRationale:
    """Detailed rationale for why an action was recommended."""

    rationale_id: str
    primary_reason: str
    supporting_reasons: List[str]

    # Evidence
    sensor_evidence: Dict[str, float]
    threshold_violations: List[str]
    trend_indicators: List[str]

    # Physics grounding
    physics_explanation: str
    reference_standard: str

    # Constraints considered
    active_constraints: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rationale_id": self.rationale_id,
            "primary_reason": self.primary_reason,
            "supporting_reasons": self.supporting_reasons,
            "sensor_evidence": self.sensor_evidence,
            "threshold_violations": self.threshold_violations,
            "trend_indicators": self.trend_indicators,
            "physics_explanation": self.physics_explanation,
            "reference_standard": self.reference_standard,
            "active_constraints": self.active_constraints,
        }


@dataclass
class ControlRecommendation:
    """A control recommendation with full explanation."""

    recommendation_id: str
    timestamp: datetime
    header_id: str

    # Action details
    action_type: ActionType
    priority: ActionPriority
    target_equipment: str
    target_parameter: str

    # Values
    current_value: float
    recommended_value: float
    unit: str

    # Expected impacts
    expected_impacts: List[ExpectedImpact]
    primary_impact: ExpectedImpact

    # Rationale
    rationale: ActionRationale

    # Alternatives
    alternatives_considered: List[AlternativeAction]

    # Confidence
    confidence_level: ConfidenceLevel
    confidence_value: float

    # Implementation
    implementation_steps: List[str] = field(default_factory=list)
    estimated_time_minutes: float = 5.0
    reversibility: str = "reversible"  # "reversible", "partially_reversible", "irreversible"

    # Natural language
    summary: str = ""
    detailed_explanation: str = ""

    # Provenance
    provenance_hash: str = ""
    model_version: str = ""
    config_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "recommendation_id": self.recommendation_id,
            "timestamp": self.timestamp.isoformat(),
            "header_id": self.header_id,
            "action_type": self.action_type.value,
            "priority": self.priority.value,
            "target_equipment": self.target_equipment,
            "target_parameter": self.target_parameter,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "unit": self.unit,
            "expected_impacts": [ei.to_dict() for ei in self.expected_impacts],
            "primary_impact": self.primary_impact.to_dict(),
            "rationale": self.rationale.to_dict(),
            "alternatives_considered": [a.to_dict() for a in self.alternatives_considered],
            "confidence_level": self.confidence_level.value,
            "confidence_value": self.confidence_value,
            "implementation_steps": self.implementation_steps,
            "estimated_time_minutes": self.estimated_time_minutes,
            "reversibility": self.reversibility,
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
            "provenance_hash": self.provenance_hash,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class RecommendationExplanation:
    """Complete explanation for a set of recommendations."""

    explanation_id: str
    timestamp: datetime
    header_id: str

    # Quality context
    current_quality: float
    target_quality: float
    quality_gap: float

    # Recommendations
    recommendations: List[ControlRecommendation]
    primary_recommendation: ControlRecommendation

    # Overall assessment
    total_expected_quality_improvement: float
    implementation_complexity: str  # "simple", "moderate", "complex"
    risk_assessment: str

    # Natural language
    executive_summary: str = ""
    technical_summary: str = ""

    # Provenance
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "explanation_id": self.explanation_id,
            "timestamp": self.timestamp.isoformat(),
            "header_id": self.header_id,
            "current_quality": self.current_quality,
            "target_quality": self.target_quality,
            "quality_gap": self.quality_gap,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "primary_recommendation": self.primary_recommendation.to_dict(),
            "total_expected_quality_improvement": self.total_expected_quality_improvement,
            "implementation_complexity": self.implementation_complexity,
            "risk_assessment": self.risk_assessment,
            "executive_summary": self.executive_summary,
            "technical_summary": self.technical_summary,
            "provenance_hash": self.provenance_hash,
        }


# =============================================================================
# RECOMMENDATION KNOWLEDGE BASE
# =============================================================================

# Standard recommendations for quality issues
QUALITY_RECOMMENDATIONS = {
    "high_drum_level": {
        "action_type": ActionType.SETPOINT_CHANGE,
        "target_parameter": "drum_level_setpoint",
        "direction": "decrease",
        "typical_change": -5.0,  # percentage points
        "unit": "%",
        "physics": (
            "Reducing drum level increases the steam disengaging area, "
            "allowing more time for moisture droplets to separate from steam "
            "before entering the steam outlet. Per API 534, optimal level is "
            "typically 40-60% of visible glass."
        ),
        "reference": "API 534, ASME Section I PG-60",
        "expected_quality_gain": 0.005,  # per 5% level reduction
        "time_to_impact": 10.0,  # minutes
        "alternatives": [
            {
                "action": "Increase continuous blowdown",
                "impact": 0.002,
                "limitation": "Increases energy loss",
            },
            {
                "action": "Reduce feedwater flow manually",
                "impact": 0.005,
                "limitation": "May upset level control",
            },
        ],
    },

    "low_separator_efficiency": {
        "action_type": ActionType.MAINTENANCE,
        "target_parameter": "separator_status",
        "direction": "inspect",
        "typical_change": None,
        "unit": "",
        "physics": (
            "Low separator dP indicates reduced centrifugal separation efficiency. "
            "Possible causes include fouled internals, damaged vanes, or bypass flow. "
            "Effective separation requires adequate pressure drop to create "
            "centrifugal acceleration of water droplets."
        ),
        "reference": "API 560, Separator Design Standards",
        "expected_quality_gain": 0.01,
        "time_to_impact": 120.0,  # After maintenance
        "alternatives": [
            {
                "action": "Reduce steam flow temporarily",
                "impact": 0.005,
                "limitation": "May not meet demand",
            },
            {
                "action": "Bypass to backup separator",
                "impact": 0.008,
                "limitation": "Requires backup equipment",
            },
        ],
    },

    "prv_condensation": {
        "action_type": ActionType.VALVE_ADJUSTMENT,
        "target_parameter": "upstream_trap_operation",
        "direction": "verify",
        "typical_change": None,
        "unit": "",
        "physics": (
            "PRV condensation indicates wet steam at the valve inlet. "
            "Moisture in steam will condense when pressure is reduced because "
            "the flash evaporation cannot fully vaporize all liquid present. "
            "Upstream drip legs and traps must remove condensate before the PRV."
        ),
        "reference": "ASME B31.1, ISA S75.05",
        "expected_quality_gain": 0.015,
        "time_to_impact": 30.0,
        "alternatives": [
            {
                "action": "Install separator upstream of PRV",
                "impact": 0.02,
                "limitation": "Capital investment required",
            },
            {
                "action": "Add drip leg before PRV",
                "impact": 0.01,
                "limitation": "Piping modification",
            },
        ],
    },

    "load_too_low": {
        "action_type": ActionType.LOAD_CHANGE,
        "target_parameter": "boiler_load",
        "direction": "increase",
        "typical_change": 10.0,
        "unit": "%",
        "physics": (
            "At low loads, steam velocity through separators decreases, "
            "reducing centrifugal separation efficiency. Additionally, "
            "relative heat losses increase and boiler dynamics become less stable. "
            "Optimal quality is typically achieved at 70-80% of rated load."
        ),
        "reference": "ASME PTC 4.1, Boiler Performance Standards",
        "expected_quality_gain": 0.003,
        "time_to_impact": 15.0,
        "alternatives": [
            {
                "action": "Take standby boiler offline",
                "impact": 0.005,
                "limitation": "Reduces system redundancy",
            },
            {
                "action": "Accept reduced quality at low load",
                "impact": 0.0,
                "limitation": "Quality remains degraded",
            },
        ],
    },

    "load_too_high": {
        "action_type": ActionType.LOAD_CHANGE,
        "target_parameter": "boiler_load",
        "direction": "decrease",
        "typical_change": -10.0,
        "unit": "%",
        "physics": (
            "At high loads approaching maximum capacity, steam velocities "
            "increase and can cause re-entrainment of separated droplets. "
            "Turbulence in the drum steam space increases carryover risk. "
            "Consider bringing additional capacity online."
        ),
        "reference": "ASME PTC 4.1, Boiler Performance Standards",
        "expected_quality_gain": 0.004,
        "time_to_impact": 15.0,
        "alternatives": [
            {
                "action": "Bring standby boiler online",
                "impact": 0.006,
                "limitation": "Requires available capacity",
            },
            {
                "action": "Reduce steam demand",
                "impact": 0.005,
                "limitation": "May impact production",
            },
        ],
    },
}


# =============================================================================
# RECOMMENDATION EXPLAINER
# =============================================================================

class RecommendationExplainer:
    """
    Explains control recommendations for steam quality management.

    Provides detailed explanations of:
    - Why specific actions are recommended
    - Expected impact on quality and other parameters
    - Alternative actions that were considered
    - Confidence levels based on available data

    Example:
        >>> explainer = RecommendationExplainer(agent_id="GL-012")
        >>> explanation = explainer.explain_recommendation(
        ...     issue_type="high_drum_level",
        ...     current_state={"drum_level_pct": 68, "quality": 0.96},
        ...     header_id="HEADER-001"
        ... )
        >>> print(explanation.summary)

    Attributes:
        agent_id: Agent identifier
        knowledge_base: Dictionary of standard recommendations
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        agent_id: str = "GL-012",
        custom_recommendations: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize RecommendationExplainer.

        Args:
            agent_id: Agent identifier
            custom_recommendations: Optional additional recommendations
        """
        self.agent_id = agent_id
        self.knowledge_base = {**QUALITY_RECOMMENDATIONS}
        if custom_recommendations:
            self.knowledge_base.update(custom_recommendations)

        self._explanations: Dict[str, RecommendationExplanation] = {}

        logger.info(f"RecommendationExplainer initialized: {agent_id}")

    def explain_recommendation(
        self,
        issue_type: str,
        current_state: Dict[str, float],
        header_id: str = "HEADER-001",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> RecommendationExplanation:
        """
        Generate explanation for a quality control recommendation.

        Args:
            issue_type: Type of quality issue (e.g., "high_drum_level")
            current_state: Current sensor readings and quality
            header_id: Steam header identifier
            constraints: Optional operational constraints

        Returns:
            RecommendationExplanation with full details
        """
        timestamp = datetime.now(timezone.utc)
        explanation_id = f"REXP-{timestamp.strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"

        current_quality = current_state.get("quality", current_state.get("dryness_fraction", 0.97))
        target_quality = 0.98  # Target quality

        # Get primary recommendation
        primary_rec = self._generate_recommendation(
            issue_type=issue_type,
            current_state=current_state,
            header_id=header_id,
            constraints=constraints,
        )

        # Generate alternative recommendations
        recommendations = [primary_rec]

        # Check for additional issues
        additional_issues = self._identify_additional_issues(current_state)
        for add_issue in additional_issues[:2]:  # Limit to 2 additional
            if add_issue != issue_type:
                add_rec = self._generate_recommendation(
                    issue_type=add_issue,
                    current_state=current_state,
                    header_id=header_id,
                    constraints=constraints,
                )
                add_rec.priority = ActionPriority.MEDIUM
                recommendations.append(add_rec)

        # Calculate total expected improvement
        total_improvement = sum(
            rec.primary_impact.delta
            for rec in recommendations
            if rec.primary_impact.category == ImpactCategory.QUALITY
        )

        # Assess complexity
        complexity = self._assess_complexity(recommendations)

        # Risk assessment
        risk = self._assess_risk(recommendations, constraints)

        # Build explanation
        explanation = RecommendationExplanation(
            explanation_id=explanation_id,
            timestamp=timestamp,
            header_id=header_id,
            current_quality=current_quality,
            target_quality=target_quality,
            quality_gap=target_quality - current_quality,
            recommendations=recommendations,
            primary_recommendation=primary_rec,
            total_expected_quality_improvement=total_improvement,
            implementation_complexity=complexity,
            risk_assessment=risk,
        )

        # Generate natural language
        self._generate_executive_summary(explanation)
        self._generate_technical_summary(explanation)

        # Calculate provenance
        self._calculate_provenance(explanation, current_state)

        # Cache
        self._explanations[explanation_id] = explanation

        logger.info(
            f"Generated recommendation explanation for {issue_type}: "
            f"{primary_rec.action_type.value}"
        )

        return explanation

    def _generate_recommendation(
        self,
        issue_type: str,
        current_state: Dict[str, float],
        header_id: str,
        constraints: Optional[Dict[str, Any]],
    ) -> ControlRecommendation:
        """Generate a single control recommendation."""
        timestamp = datetime.now(timezone.utc)
        recommendation_id = f"REC-{timestamp.strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"

        # Get knowledge base entry
        kb_entry = self.knowledge_base.get(issue_type, {})

        action_type = kb_entry.get("action_type", ActionType.OPERATOR_ALERT)
        target_param = kb_entry.get("target_parameter", "unknown")
        direction = kb_entry.get("direction", "adjust")
        typical_change = kb_entry.get("typical_change", 0)
        unit = kb_entry.get("unit", "")

        # Determine current and recommended values
        current_value = current_state.get(target_param, 0)
        if typical_change is not None:
            recommended_value = current_value + typical_change
        else:
            recommended_value = current_value  # Maintenance action

        # Calculate expected impacts
        expected_impacts = self._calculate_expected_impacts(
            issue_type, kb_entry, current_state
        )
        primary_impact = expected_impacts[0] if expected_impacts else self._default_impact()

        # Build rationale
        rationale = self._build_rationale(
            issue_type, kb_entry, current_state, constraints
        )

        # Generate alternatives
        alternatives = self._generate_alternatives(kb_entry, current_state)

        # Determine confidence
        confidence_value = self._calculate_confidence(kb_entry, current_state)
        confidence_level = self._value_to_confidence_level(confidence_value)

        # Determine priority
        priority = self._determine_priority(issue_type, current_state)

        # Build implementation steps
        implementation_steps = self._build_implementation_steps(
            action_type, target_param, direction, typical_change
        )

        recommendation = ControlRecommendation(
            recommendation_id=recommendation_id,
            timestamp=timestamp,
            header_id=header_id,
            action_type=action_type,
            priority=priority,
            target_equipment=self._get_target_equipment(target_param),
            target_parameter=target_param,
            current_value=current_value,
            recommended_value=recommended_value,
            unit=unit,
            expected_impacts=expected_impacts,
            primary_impact=primary_impact,
            rationale=rationale,
            alternatives_considered=alternatives,
            confidence_level=confidence_level,
            confidence_value=confidence_value,
            implementation_steps=implementation_steps,
            estimated_time_minutes=kb_entry.get("time_to_impact", 10.0),
            reversibility="reversible" if action_type != ActionType.MAINTENANCE else "partially_reversible",
            model_version=self.VERSION,
        )

        # Generate natural language
        self._generate_recommendation_summary(recommendation, kb_entry)
        self._generate_recommendation_detail(recommendation, kb_entry)

        return recommendation

    def _calculate_expected_impacts(
        self,
        issue_type: str,
        kb_entry: Dict[str, Any],
        current_state: Dict[str, float],
    ) -> List[ExpectedImpact]:
        """Calculate expected impacts of the recommendation."""
        impacts = []

        # Quality impact
        quality_gain = kb_entry.get("expected_quality_gain", 0.005)
        current_quality = current_state.get("quality", 0.97)
        expected_quality = min(1.0, current_quality + quality_gain)

        impacts.append(ExpectedImpact(
            impact_id=str(uuid.uuid4())[:8],
            category=ImpactCategory.QUALITY,
            parameter="dryness_fraction",
            unit="fraction",
            current_value=current_quality,
            expected_value=expected_quality,
            delta=quality_gain,
            delta_pct=(quality_gain / current_quality) * 100 if current_quality > 0 else 0,
            confidence=ConfidenceLevel.HIGH,
            confidence_value=0.8,
            time_to_impact_minutes=kb_entry.get("time_to_impact", 10.0),
            description=f"Expected quality improvement from {issue_type} correction",
            physics_basis=kb_entry.get("physics", ""),
        ))

        # Efficiency impact (typically secondary)
        if issue_type in ["high_drum_level", "load_too_low", "load_too_high"]:
            impacts.append(ExpectedImpact(
                impact_id=str(uuid.uuid4())[:8],
                category=ImpactCategory.EFFICIENCY,
                parameter="boiler_efficiency",
                unit="%",
                current_value=82.0,
                expected_value=82.2,
                delta=0.2,
                delta_pct=0.24,
                confidence=ConfidenceLevel.MEDIUM,
                confidence_value=0.6,
                time_to_impact_minutes=kb_entry.get("time_to_impact", 10.0) * 2,
                description="Minor efficiency improvement from optimized operation",
            ))

        return impacts

    def _default_impact(self) -> ExpectedImpact:
        """Return default impact when none calculated."""
        return ExpectedImpact(
            impact_id=str(uuid.uuid4())[:8],
            category=ImpactCategory.QUALITY,
            parameter="dryness_fraction",
            unit="fraction",
            current_value=0.97,
            expected_value=0.98,
            delta=0.01,
            delta_pct=1.0,
            confidence=ConfidenceLevel.LOW,
            confidence_value=0.4,
            time_to_impact_minutes=30.0,
            description="Expected quality improvement",
        )

    def _build_rationale(
        self,
        issue_type: str,
        kb_entry: Dict[str, Any],
        current_state: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
    ) -> ActionRationale:
        """Build rationale for the recommendation."""
        # Primary reason from knowledge base
        primary_reason = self._get_primary_reason(issue_type, current_state)

        # Supporting reasons
        supporting = []
        if kb_entry.get("physics"):
            supporting.append(f"Physics basis: {kb_entry['physics'][:100]}...")
        if kb_entry.get("reference"):
            supporting.append(f"Per {kb_entry['reference']}")

        # Threshold violations
        violations = self._identify_violations(issue_type, current_state)

        # Trend indicators
        trends = self._identify_trends(current_state)

        # Active constraints
        active_constraints = []
        if constraints:
            for name, value in constraints.items():
                active_constraints.append({
                    "constraint": name,
                    "value": value,
                    "type": "operational",
                })

        return ActionRationale(
            rationale_id=str(uuid.uuid4())[:8],
            primary_reason=primary_reason,
            supporting_reasons=supporting,
            sensor_evidence=current_state,
            threshold_violations=violations,
            trend_indicators=trends,
            physics_explanation=kb_entry.get("physics", ""),
            reference_standard=kb_entry.get("reference", "ASME PTC 19.11"),
            active_constraints=active_constraints,
        )

    def _get_primary_reason(
        self, issue_type: str, current_state: Dict[str, float]
    ) -> str:
        """Get primary reason for recommendation."""
        reasons = {
            "high_drum_level": (
                f"Drum level at {current_state.get('drum_level_pct', 0):.1f}% exceeds "
                f"optimal range (40-60%), increasing carryover risk."
            ),
            "low_separator_efficiency": (
                f"Separator dP at {current_state.get('separator_dp_psi', 0):.1f} psi "
                f"is below minimum for effective separation."
            ),
            "prv_condensation": (
                f"PRV condensation rate of {current_state.get('prv_condensation_rate', 0):.1f} lb/hr "
                f"indicates wet steam at valve inlet."
            ),
            "load_too_low": (
                f"Operating at {current_state.get('load_pct', 0):.0f}% load, "
                f"below optimal range for separator efficiency."
            ),
            "load_too_high": (
                f"Operating at {current_state.get('load_pct', 0):.0f}% load, "
                f"approaching maximum with increased carryover risk."
            ),
        }
        return reasons.get(issue_type, f"Quality issue: {issue_type}")

    def _identify_violations(
        self, issue_type: str, current_state: Dict[str, float]
    ) -> List[str]:
        """Identify threshold violations."""
        violations = []

        if current_state.get("drum_level_pct", 50) > 65:
            violations.append(
                f"Drum level {current_state['drum_level_pct']:.1f}% > 65% HH alarm"
            )
        if current_state.get("separator_dp_psi", 5) < 2:
            violations.append(
                f"Separator dP {current_state['separator_dp_psi']:.1f} psi < 2 psi minimum"
            )
        if current_state.get("quality", 1.0) < 0.95:
            violations.append(
                f"Quality {current_state['quality']:.4f} < 0.95 minimum"
            )

        return violations

    def _identify_trends(self, current_state: Dict[str, float]) -> List[str]:
        """Identify trend indicators from state."""
        trends = []

        # These would normally come from historical analysis
        if current_state.get("drum_level_trend", 0) > 0:
            trends.append("Drum level trending upward")
        if current_state.get("quality_trend", 0) < 0:
            trends.append("Quality degrading over time")

        return trends

    def _generate_alternatives(
        self,
        kb_entry: Dict[str, Any],
        current_state: Dict[str, float],
    ) -> List[AlternativeAction]:
        """Generate alternative actions."""
        alternatives = []

        for alt_info in kb_entry.get("alternatives", []):
            alternatives.append(AlternativeAction(
                alternative_id=str(uuid.uuid4())[:8],
                action_type=ActionType.OPERATOR_ALERT,
                description=alt_info.get("action", ""),
                expected_quality_impact=alt_info.get("impact", 0),
                expected_efficiency_impact=0.0,
                rejection_reason=alt_info.get("limitation", "Less effective"),
                limitations=[alt_info.get("limitation", "")],
                conditions_to_prefer="If primary action not feasible",
            ))

        return alternatives

    def _calculate_confidence(
        self,
        kb_entry: Dict[str, Any],
        current_state: Dict[str, float],
    ) -> float:
        """Calculate confidence in the recommendation."""
        confidence = 0.7  # Base confidence

        # Boost for known pattern
        if kb_entry:
            confidence += 0.15

        # Boost for clear violations
        if current_state.get("drum_level_pct", 50) > 65:
            confidence += 0.05
        if current_state.get("separator_dp_psi", 5) < 2:
            confidence += 0.05

        return min(0.95, confidence)

    def _value_to_confidence_level(self, value: float) -> ConfidenceLevel:
        """Convert confidence value to level."""
        if value >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif value >= 0.75:
            return ConfidenceLevel.HIGH
        elif value >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif value >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN

    def _determine_priority(
        self, issue_type: str, current_state: Dict[str, float]
    ) -> ActionPriority:
        """Determine priority of the recommendation."""
        quality = current_state.get("quality", 0.97)

        if quality < 0.90:
            return ActionPriority.CRITICAL
        elif quality < 0.95:
            return ActionPriority.HIGH
        elif quality < 0.97:
            return ActionPriority.MEDIUM
        else:
            return ActionPriority.LOW

    def _get_target_equipment(self, target_param: str) -> str:
        """Get target equipment from parameter name."""
        equipment_map = {
            "drum_level_setpoint": "Drum Level Controller",
            "separator_status": "Steam Separator",
            "upstream_trap_operation": "Steam Traps",
            "boiler_load": "Boiler Load Controller",
        }
        return equipment_map.get(target_param, "Steam System")

    def _build_implementation_steps(
        self,
        action_type: ActionType,
        target_param: str,
        direction: str,
        change: Optional[float],
    ) -> List[str]:
        """Build implementation steps for the recommendation."""
        steps = []

        if action_type == ActionType.SETPOINT_CHANGE:
            steps = [
                f"Verify current {target_param} value in DCS",
                f"Calculate new setpoint ({direction} by {abs(change or 0):.1f})",
                "Enter new setpoint in control system",
                "Monitor response for 10-15 minutes",
                "Verify quality improvement achieved",
            ]
        elif action_type == ActionType.MAINTENANCE:
            steps = [
                "Create maintenance work order",
                "Schedule equipment isolation",
                "Perform inspection/repair",
                "Return to service and test",
                "Verify normal operation",
            ]
        elif action_type == ActionType.VALVE_ADJUSTMENT:
            steps = [
                "Identify affected valves/traps",
                "Perform field inspection",
                "Adjust or repair as needed",
                "Verify proper operation",
                "Monitor downstream quality",
            ]
        elif action_type == ActionType.LOAD_CHANGE:
            steps = [
                "Coordinate with operations",
                "Adjust load setpoint",
                "Monitor boiler response",
                "Verify quality improvement",
            ]

        return steps

    def _identify_additional_issues(
        self, current_state: Dict[str, float]
    ) -> List[str]:
        """Identify additional issues from current state."""
        issues = []

        if current_state.get("drum_level_pct", 50) > 60:
            issues.append("high_drum_level")
        if current_state.get("separator_dp_psi", 5) < 3:
            issues.append("low_separator_efficiency")
        if current_state.get("prv_condensation_rate", 0) > 10:
            issues.append("prv_condensation")
        if current_state.get("load_pct", 75) < 40:
            issues.append("load_too_low")
        if current_state.get("load_pct", 75) > 90:
            issues.append("load_too_high")

        return issues

    def _assess_complexity(
        self, recommendations: List[ControlRecommendation]
    ) -> str:
        """Assess implementation complexity."""
        maintenance_count = sum(
            1 for r in recommendations
            if r.action_type == ActionType.MAINTENANCE
        )
        total_time = sum(r.estimated_time_minutes for r in recommendations)

        if maintenance_count > 1 or total_time > 120:
            return "complex"
        elif maintenance_count == 1 or total_time > 30:
            return "moderate"
        else:
            return "simple"

    def _assess_risk(
        self,
        recommendations: List[ControlRecommendation],
        constraints: Optional[Dict[str, Any]],
    ) -> str:
        """Assess risk of implementing recommendations."""
        high_priority_count = sum(
            1 for r in recommendations
            if r.priority in [ActionPriority.CRITICAL, ActionPriority.HIGH]
        )

        if high_priority_count > 1:
            return "Elevated risk - multiple high-priority issues require attention"
        elif any(r.reversibility == "irreversible" for r in recommendations):
            return "Moderate risk - some actions are not easily reversible"
        else:
            return "Low risk - actions are reversible and well-understood"

    def _generate_recommendation_summary(
        self,
        recommendation: ControlRecommendation,
        kb_entry: Dict[str, Any],
    ) -> None:
        """Generate summary for a recommendation."""
        action_desc = {
            ActionType.SETPOINT_CHANGE: "Adjust setpoint",
            ActionType.VALVE_ADJUSTMENT: "Verify valve/trap operation",
            ActionType.LOAD_CHANGE: "Adjust operating load",
            ActionType.MAINTENANCE: "Schedule maintenance",
            ActionType.EQUIPMENT_SWITCH: "Switch equipment",
            ActionType.OPERATOR_ALERT: "Operator attention required",
            ActionType.NO_ACTION: "No action required",
        }

        summary_parts = [
            f"RECOMMENDATION: {action_desc.get(recommendation.action_type, 'Action required')}.",
            f"Target: {recommendation.target_equipment} ({recommendation.target_parameter}).",
        ]

        if recommendation.recommended_value != recommendation.current_value:
            summary_parts.append(
                f"Change from {recommendation.current_value:.1f} to "
                f"{recommendation.recommended_value:.1f} {recommendation.unit}."
            )

        summary_parts.append(
            f"Expected quality improvement: +{recommendation.primary_impact.delta:.4f}."
        )
        summary_parts.append(
            f"Confidence: {recommendation.confidence_level.value} "
            f"({recommendation.confidence_value:.0%})."
        )

        recommendation.summary = " ".join(summary_parts)

    def _generate_recommendation_detail(
        self,
        recommendation: ControlRecommendation,
        kb_entry: Dict[str, Any],
    ) -> None:
        """Generate detailed explanation for a recommendation."""
        lines = [
            "RECOMMENDATION DETAIL",
            "=" * 50,
            "",
            f"Action: {recommendation.action_type.value}",
            f"Priority: {recommendation.priority.value}",
            f"Target: {recommendation.target_equipment}",
            f"Parameter: {recommendation.target_parameter}",
            "",
            "CURRENT vs RECOMMENDED:",
            f"  Current: {recommendation.current_value:.2f} {recommendation.unit}",
            f"  Recommended: {recommendation.recommended_value:.2f} {recommendation.unit}",
            "",
            "RATIONALE:",
            f"  {recommendation.rationale.primary_reason}",
            "",
            "PHYSICS BASIS:",
            f"  {recommendation.rationale.physics_explanation}",
            "",
            "EXPECTED IMPACT:",
        ]

        for impact in recommendation.expected_impacts:
            lines.append(
                f"  - {impact.parameter}: {impact.current_value:.4f} -> "
                f"{impact.expected_value:.4f} ({impact.delta:+.4f})"
            )

        lines.extend([
            "",
            "IMPLEMENTATION STEPS:",
        ])
        for i, step in enumerate(recommendation.implementation_steps, 1):
            lines.append(f"  {i}. {step}")

        if recommendation.alternatives_considered:
            lines.extend([
                "",
                "ALTERNATIVES CONSIDERED:",
            ])
            for alt in recommendation.alternatives_considered:
                lines.append(
                    f"  - {alt.description}: Rejected due to {alt.rejection_reason}"
                )

        lines.extend([
            "",
            f"Confidence: {recommendation.confidence_level.value} ({recommendation.confidence_value:.0%})",
            f"Reference: {recommendation.rationale.reference_standard}",
        ])

        recommendation.detailed_explanation = "\n".join(lines)

    def _generate_executive_summary(
        self, explanation: RecommendationExplanation
    ) -> None:
        """Generate executive summary."""
        primary = explanation.primary_recommendation

        summary_parts = [
            f"Steam quality at {explanation.header_id} is "
            f"{explanation.current_quality:.4f} (target: {explanation.target_quality:.4f}).",
        ]

        if explanation.quality_gap > 0:
            summary_parts.append(
                f"Primary recommendation: {primary.summary}"
            )
            summary_parts.append(
                f"Expected total improvement: +{explanation.total_expected_quality_improvement:.4f}."
            )
        else:
            summary_parts.append("Quality meets target. Monitoring recommended.")

        summary_parts.append(
            f"Implementation complexity: {explanation.implementation_complexity}. "
            f"{explanation.risk_assessment}"
        )

        explanation.executive_summary = " ".join(summary_parts)

    def _generate_technical_summary(
        self, explanation: RecommendationExplanation
    ) -> None:
        """Generate technical summary."""
        lines = [
            "TECHNICAL SUMMARY",
            "=" * 50,
            "",
            f"Header: {explanation.header_id}",
            f"Current Quality: {explanation.current_quality:.4f}",
            f"Target Quality: {explanation.target_quality:.4f}",
            f"Gap: {explanation.quality_gap:.4f}",
            "",
            "RECOMMENDATIONS:",
        ]

        for i, rec in enumerate(explanation.recommendations, 1):
            lines.append(
                f"  {i}. [{rec.priority.value.upper()}] {rec.action_type.value}: "
                f"{rec.target_parameter}"
            )

        lines.extend([
            "",
            f"Total Expected Improvement: +{explanation.total_expected_quality_improvement:.4f}",
            f"Implementation Complexity: {explanation.implementation_complexity}",
            f"Risk Assessment: {explanation.risk_assessment}",
        ])

        explanation.technical_summary = "\n".join(lines)

    def _calculate_provenance(
        self,
        explanation: RecommendationExplanation,
        current_state: Dict[str, float],
    ) -> None:
        """Calculate provenance hash."""
        provenance_data = {
            "explanation_id": explanation.explanation_id,
            "timestamp": explanation.timestamp.isoformat(),
            "header_id": explanation.header_id,
            "current_quality": explanation.current_quality,
            "recommendations": [r.recommendation_id for r in explanation.recommendations],
            "agent_id": self.agent_id,
        }

        json_str = json.dumps(provenance_data, sort_keys=True)
        explanation.provenance_hash = hashlib.sha256(json_str.encode()).hexdigest()

    def get_explanation(
        self, explanation_id: str
    ) -> Optional[RecommendationExplanation]:
        """Get explanation by ID."""
        return self._explanations.get(explanation_id)

    def list_known_issues(self) -> List[str]:
        """List known issue types."""
        return list(self.knowledge_base.keys())
