"""
GL-003 UNIFIEDSTEAM - Recommendation Engine

Provides packaging and presentation of optimization recommendations:
- Recommendation packaging with context
- Ranking based on operator preferences
- Action statement generation
- Benefit estimation with uncertainty
- Risk assessment

Every recommendation includes:
- Action: What to do
- Rationale: Why to do it
- Expected benefit: Quantified impact
- Risk: Implementation risks
- Verification plan: How to verify success
- Escalation path: When to escalate
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import time

from pydantic import BaseModel, Field, validator

from .constraints import UncertaintyConstraints

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class RecommendationType(str, Enum):
    """Type of recommendation."""

    SETPOINT_CHANGE = "setpoint_change"
    EQUIPMENT_ACTION = "equipment_action"
    MAINTENANCE = "maintenance"
    OPERATIONAL = "operational"
    ALARM = "alarm"
    INFORMATIONAL = "informational"


class RecommendationPriority(str, Enum):
    """Priority level of recommendation."""

    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Action within 1 hour
    MEDIUM = "medium"  # Action within shift
    LOW = "low"  # Action when convenient
    INFORMATIONAL = "informational"  # No action required


class RiskCategory(str, Enum):
    """Risk category for recommendations."""

    SAFETY = "safety"
    EQUIPMENT = "equipment"
    PROCESS = "process"
    ENVIRONMENTAL = "environmental"
    ECONOMIC = "economic"


class OperatorPreference(str, Enum):
    """Operator preference for ranking."""

    COST_FOCUSED = "cost_focused"
    EFFICIENCY_FOCUSED = "efficiency_focused"
    EMISSION_FOCUSED = "emission_focused"
    RELIABILITY_FOCUSED = "reliability_focused"
    BALANCED = "balanced"


# =============================================================================
# Data Models
# =============================================================================


class BenefitEstimate(BaseModel):
    """Quantified benefit estimate with uncertainty."""

    # Primary metrics
    steam_savings_klb_hr: float = Field(
        default=0.0, description="Steam savings (klb/hr)"
    )
    fuel_savings_mmbtu_hr: float = Field(
        default=0.0, description="Fuel savings (MMBTU/hr)"
    )
    cost_savings_per_hr: float = Field(
        default=0.0, description="Cost savings ($/hr)"
    )
    co2e_reduction_lb_hr: float = Field(
        default=0.0, description="CO2e reduction (lb/hr)"
    )

    # Annualized benefits
    annual_cost_savings: float = Field(
        default=0.0, description="Annual cost savings ($)"
    )
    annual_co2e_reduction_tons: float = Field(
        default=0.0, description="Annual CO2e reduction (tons)"
    )

    # Uncertainty bounds
    confidence: float = Field(
        default=0.90, ge=0, le=1, description="Confidence level"
    )
    lower_bound_cost: float = Field(
        default=0.0, description="Lower bound cost savings ($/hr)"
    )
    upper_bound_cost: float = Field(
        default=0.0, description="Upper bound cost savings ($/hr)"
    )
    uncertainty_percent: float = Field(
        default=10.0, ge=0, description="Uncertainty as % of estimate"
    )

    # Assumptions
    assumptions: List[str] = Field(
        default_factory=list, description="Key assumptions"
    )

    def calculate_bounds(self) -> None:
        """Calculate uncertainty bounds from percentage."""
        self.lower_bound_cost = self.cost_savings_per_hr * (1 - self.uncertainty_percent / 100)
        self.upper_bound_cost = self.cost_savings_per_hr * (1 + self.uncertainty_percent / 100)


class RiskAssessment(BaseModel):
    """Risk assessment for a recommendation."""

    overall_risk_level: str = Field(
        default="low", description="Overall risk level"
    )
    risk_score: float = Field(
        default=0.0, ge=0, le=100, description="Risk score (0-100)"
    )

    # Individual risk factors
    risks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Individual risk factors"
    )

    # Mitigation
    mitigations: List[str] = Field(
        default_factory=list, description="Risk mitigations"
    )

    # Requirements
    requires_operator_confirmation: bool = Field(
        default=False, description="Requires operator confirmation"
    )
    requires_engineering_review: bool = Field(
        default=False, description="Requires engineering review"
    )
    confirmation_reason: str = Field(
        default="", description="Reason for confirmation requirement"
    )


class VerificationPlan(BaseModel):
    """Plan for verifying recommendation success."""

    verification_steps: List[str] = Field(
        default_factory=list, description="Steps to verify success"
    )
    key_metrics: List[str] = Field(
        default_factory=list, description="Key metrics to monitor"
    )
    expected_response_time_min: int = Field(
        default=30, description="Expected time to see response (minutes)"
    )
    success_criteria: List[str] = Field(
        default_factory=list, description="Success criteria"
    )
    rollback_procedure: List[str] = Field(
        default_factory=list, description="Rollback procedure if unsuccessful"
    )


class EscalationPath(BaseModel):
    """Escalation path for recommendations."""

    escalation_levels: List[Dict[str, Any]] = Field(
        default_factory=list, description="Escalation levels"
    )
    auto_escalate_after_min: int = Field(
        default=30, description="Auto-escalate if no response (minutes)"
    )
    escalation_contacts: List[str] = Field(
        default_factory=list, description="Escalation contacts"
    )


class Recommendation(BaseModel):
    """Complete recommendation with all required elements."""

    # Identification
    recommendation_id: str = Field(..., description="Unique recommendation ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    recommendation_type: RecommendationType = Field(
        ..., description="Type of recommendation"
    )
    priority: RecommendationPriority = Field(
        ..., description="Priority level"
    )

    # Action
    action: str = Field(..., description="What action to take")
    action_details: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed action parameters"
    )
    target_equipment: List[str] = Field(
        default_factory=list, description="Target equipment IDs"
    )

    # Rationale
    rationale: str = Field(..., description="Why this action is recommended")
    supporting_data: Dict[str, Any] = Field(
        default_factory=dict, description="Supporting data and analysis"
    )

    # Benefits
    expected_benefit: BenefitEstimate = Field(
        ..., description="Expected benefit estimate"
    )

    # Risk
    risk_assessment: RiskAssessment = Field(
        ..., description="Risk assessment"
    )

    # Verification
    verification_plan: VerificationPlan = Field(
        ..., description="Verification plan"
    )

    # Escalation
    escalation_path: EscalationPath = Field(
        ..., description="Escalation path"
    )

    # Status tracking
    status: str = Field(default="pending", description="Current status")
    created_by: str = Field(default="system", description="Created by")
    acknowledged_by: Optional[str] = Field(
        default=None, description="Acknowledged by"
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None, description="Acknowledgment timestamp"
    )

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    source_optimization: str = Field(
        default="", description="Source optimization that generated this"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class RankedList(BaseModel):
    """Ranked list of recommendations."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list, description="Ranked recommendations"
    )
    ranking_preference: OperatorPreference = Field(
        default=OperatorPreference.BALANCED, description="Ranking preference"
    )
    total_potential_savings_hr: float = Field(
        default=0.0, description="Total potential savings ($/hr)"
    )
    total_co2e_reduction_hr: float = Field(
        default=0.0, description="Total CO2e reduction (lb/hr)"
    )


# =============================================================================
# Recommendation Engine
# =============================================================================


class RecommendationEngine:
    """
    Packages and presents optimization recommendations.

    Every recommendation includes:
    - Action: Clear directive on what to do
    - Rationale: Why this action is recommended
    - Expected benefit: Quantified with uncertainty bounds
    - Risk: Assessment of implementation risks
    - Verification plan: How to verify success
    - Escalation path: When and how to escalate
    """

    # Operating hours for annualization
    DEFAULT_OPERATING_HOURS = 8000

    def __init__(
        self,
        uncertainty_constraints: Optional[UncertaintyConstraints] = None,
        operating_hours: int = DEFAULT_OPERATING_HOURS,
        co2_cost_per_ton: float = 0.0,
    ) -> None:
        """
        Initialize recommendation engine.

        Args:
            uncertainty_constraints: Constraints for uncertainty handling
            operating_hours: Annual operating hours
            co2_cost_per_ton: CO2 cost for emissions valuation ($/ton)
        """
        self.uncertainty_constraints = (
            uncertainty_constraints or UncertaintyConstraints()
        )
        self.operating_hours = operating_hours
        self.co2_cost_per_ton = co2_cost_per_ton

        # Recommendation counter for ID generation
        self._recommendation_counter = 0

        logger.info("RecommendationEngine initialized")

    def package_recommendation(
        self,
        optimization_result: Any,
        context: Dict[str, Any],
    ) -> Recommendation:
        """
        Package an optimization result into a complete recommendation.

        Args:
            optimization_result: Result from an optimizer
            context: Additional context (system state, operator info, etc.)

        Returns:
            Complete Recommendation object
        """
        start_time = time.perf_counter()

        # Generate unique ID
        self._recommendation_counter += 1
        rec_id = f"REC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._recommendation_counter:04d}"

        # Determine recommendation type and priority
        rec_type, priority = self._classify_optimization(optimization_result, context)

        # Generate action statement
        action = self.generate_action_statement(optimization_result, context)

        # Generate rationale
        rationale = self._generate_rationale(optimization_result, context)

        # Compute expected benefit
        benefit = self.compute_expected_benefit(optimization_result, context)

        # Assess risk
        risk = self.assess_implementation_risk(optimization_result, context)

        # Create verification plan
        verification = self._create_verification_plan(optimization_result, context)

        # Create escalation path
        escalation = self._create_escalation_path(priority, context)

        # Get target equipment
        target_equipment = self._extract_target_equipment(optimization_result)

        # Get action details
        action_details = self._extract_action_details(optimization_result)

        # Create recommendation
        recommendation = Recommendation(
            recommendation_id=rec_id,
            recommendation_type=rec_type,
            priority=priority,
            action=action,
            action_details=action_details,
            target_equipment=target_equipment,
            rationale=rationale,
            supporting_data=self._extract_supporting_data(optimization_result),
            expected_benefit=benefit,
            risk_assessment=risk,
            verification_plan=verification,
            escalation_path=escalation,
            source_optimization=type(optimization_result).__name__,
        )

        # Generate provenance hash
        recommendation.provenance_hash = self._generate_provenance_hash(
            recommendation
        )

        computation_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Packaged recommendation {rec_id}: {rec_type.value}, "
            f"priority={priority.value}, benefit=${benefit.cost_savings_per_hr:.2f}/hr "
            f"in {computation_time:.1f}ms"
        )

        return recommendation

    def rank_recommendations(
        self,
        recommendations: List[Recommendation],
        operator_preferences: OperatorPreference = OperatorPreference.BALANCED,
    ) -> RankedList:
        """
        Rank recommendations based on operator preferences.

        Args:
            recommendations: List of recommendations to rank
            operator_preferences: Operator preference for ranking

        Returns:
            RankedList with recommendations in priority order
        """
        if not recommendations:
            return RankedList(
                recommendations=[],
                ranking_preference=operator_preferences,
            )

        # Calculate scores based on preferences
        scored_recs: List[Tuple[Recommendation, float]] = []

        for rec in recommendations:
            score = self._calculate_ranking_score(rec, operator_preferences)
            scored_recs.append((rec, score))

        # Sort by score (higher = higher priority)
        scored_recs.sort(key=lambda x: x[1], reverse=True)

        # Extract ranked recommendations
        ranked = [rec for rec, _ in scored_recs]

        # Calculate totals
        total_savings = sum(
            r.expected_benefit.cost_savings_per_hr for r in ranked
        )
        total_co2 = sum(
            r.expected_benefit.co2e_reduction_lb_hr for r in ranked
        )

        return RankedList(
            recommendations=ranked,
            ranking_preference=operator_preferences,
            total_potential_savings_hr=total_savings,
            total_co2e_reduction_hr=total_co2,
        )

    def _calculate_ranking_score(
        self,
        recommendation: Recommendation,
        preferences: OperatorPreference,
    ) -> float:
        """Calculate ranking score based on preferences."""
        score = 0.0

        # Priority contribution (0-40 points)
        priority_scores = {
            RecommendationPriority.CRITICAL: 40,
            RecommendationPriority.HIGH: 30,
            RecommendationPriority.MEDIUM: 20,
            RecommendationPriority.LOW: 10,
            RecommendationPriority.INFORMATIONAL: 0,
        }
        score += priority_scores.get(recommendation.priority, 10)

        # Benefit contributions based on preference
        benefit = recommendation.expected_benefit

        if preferences == OperatorPreference.COST_FOCUSED:
            # Weight cost savings heavily
            score += min(30, benefit.cost_savings_per_hr / 10)
            score += min(10, benefit.steam_savings_klb_hr * 5)

        elif preferences == OperatorPreference.EFFICIENCY_FOCUSED:
            # Weight efficiency improvements
            score += min(25, benefit.fuel_savings_mmbtu_hr * 10)
            score += min(15, benefit.steam_savings_klb_hr * 10)

        elif preferences == OperatorPreference.EMISSION_FOCUSED:
            # Weight emissions reductions
            score += min(30, benefit.co2e_reduction_lb_hr / 100)
            score += min(10, benefit.fuel_savings_mmbtu_hr * 5)

        elif preferences == OperatorPreference.RELIABILITY_FOCUSED:
            # Weight risk reduction
            risk_score = recommendation.risk_assessment.risk_score
            score += max(0, 30 - risk_score / 3)  # Lower risk = higher score
            score += min(10, benefit.cost_savings_per_hr / 20)

        else:  # BALANCED
            # Equal weights
            score += min(15, benefit.cost_savings_per_hr / 15)
            score += min(10, benefit.steam_savings_klb_hr * 5)
            score += min(10, benefit.co2e_reduction_lb_hr / 200)
            score += max(0, 15 - recommendation.risk_assessment.risk_score / 6)

        # Confidence adjustment
        confidence = benefit.confidence
        score *= (0.5 + 0.5 * confidence)  # Scale by 50-100% based on confidence

        return score

    def generate_action_statement(
        self,
        optimization_result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate clear action statement from optimization result.

        Args:
            optimization_result: Result from an optimizer
            context: Additional context

        Returns:
            Clear, actionable statement
        """
        result_type = type(optimization_result).__name__

        # Handle different result types
        if result_type == "SprayOptimizationResult":
            result = optimization_result
            if abs(result.temp_change_f) > 1:
                direction = "Increase" if result.temp_change_f > 0 else "Decrease"
                return (
                    f"{direction} desuperheater {result.desuperheater_id} "
                    f"temperature setpoint to {result.recommended_temp_setpoint_f:.0f}F "
                    f"({abs(result.temp_change_f):.1f}F change)"
                )
            else:
                return f"Maintain current desuperheater {result.desuperheater_id} setpoint"

        elif result_type == "LoadAllocationResult":
            result = optimization_result
            changes = [
                a for a in result.allocations if a.change_required
            ]
            if changes:
                actions = []
                for alloc in changes[:3]:  # Top 3 changes
                    direction = (
                        "Increase" if alloc.recommended_load_percent > alloc.current_load_percent
                        else "Decrease"
                    )
                    actions.append(
                        f"{direction} {alloc.boiler_id} to {alloc.recommended_load_percent:.0f}%"
                    )
                return "Adjust boiler loads: " + "; ".join(actions)
            else:
                return "Maintain current boiler load allocation"

        elif result_type == "RoutingOptimization":
            result = optimization_result
            if result.recommendations:
                count = len(result.recommendations)
                return (
                    f"Implement {count} condensate routing changes to increase "
                    f"return rate from {result.current_return_rate_percent:.0f}% "
                    f"to {result.optimized_return_rate_percent:.0f}%"
                )
            else:
                return "No condensate routing changes recommended"

        elif result_type == "ReplacementPriority":
            result = optimization_result
            if result.within_budget_tasks > 0:
                return (
                    f"Replace {result.within_budget_tasks} failed steam traps "
                    f"to eliminate {result.total_steam_loss_lb_hr:.0f} lb/hr steam loss"
                )
            else:
                return "No trap replacements within current budget"

        elif result_type == "LossMinimizationResult":
            result = optimization_result
            if result.recommendations:
                return result.recommendations[0]  # Primary recommendation
            else:
                return "No loss reduction actions identified"

        elif result_type == "HeaderOptimization":
            result = optimization_result
            if abs(result.pressure_change_psig) > 1:
                direction = "Increase" if result.pressure_change_psig > 0 else "Decrease"
                return (
                    f"{direction} {result.header_id} header pressure setpoint "
                    f"to {result.recommended_pressure_psig:.0f} psig"
                )
            else:
                return f"Maintain current {result.header_id} header pressure"

        else:
            # Generic fallback
            return f"Review optimization result: {result_type}"

    def compute_expected_benefit(
        self,
        optimization_result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> BenefitEstimate:
        """
        Compute expected benefit with uncertainty.

        Args:
            optimization_result: Result from an optimizer
            context: Additional context

        Returns:
            BenefitEstimate with uncertainty bounds
        """
        benefit = BenefitEstimate()
        result_type = type(optimization_result).__name__

        # Default assumptions
        steam_cost_per_klb = 10.0  # $/klb
        fuel_cost_per_mmbtu = 8.0  # $/MMBTU
        co2_factor = 117.0  # lb CO2/MMBTU natural gas

        if context:
            steam_cost_per_klb = context.get("steam_cost_per_klb", steam_cost_per_klb)
            fuel_cost_per_mmbtu = context.get("fuel_cost_per_mmbtu", fuel_cost_per_mmbtu)

        # Extract benefits based on result type
        if result_type == "SprayOptimizationResult":
            result = optimization_result
            # Spray water reduction = steam savings
            if result.spray_water_reduction_percent > 0:
                spray_reduction_gpm = result.spray_water_reduction_percent / 100 * 10  # Estimate
                benefit.steam_savings_klb_hr = spray_reduction_gpm * 60 * 8.34 / 1000
            benefit.cost_savings_per_hr = result.cost_savings_per_hour
            benefit.confidence = result.confidence
            benefit.uncertainty_percent = (1 - result.confidence) * 100

        elif result_type == "LoadAllocationResult":
            result = optimization_result
            benefit.cost_savings_per_hr = result.improvement_percent / 100 * result.total_cost_per_hr
            benefit.fuel_savings_mmbtu_hr = benefit.cost_savings_per_hr / fuel_cost_per_mmbtu
            benefit.co2e_reduction_lb_hr = benefit.fuel_savings_mmbtu_hr * co2_factor
            benefit.confidence = result.confidence
            benefit.uncertainty_percent = 10.0

        elif result_type == "RoutingOptimization":
            result = optimization_result
            benefit.steam_savings_klb_hr = result.total_recoverable_flow_lb_hr / 1000
            benefit.cost_savings_per_hr = result.total_cost_savings_per_hr
            benefit.confidence = result.confidence
            benefit.uncertainty_percent = 15.0

        elif result_type == "ReplacementPriority":
            result = optimization_result
            benefit.steam_savings_klb_hr = result.total_steam_loss_lb_hr / 1000
            benefit.cost_savings_per_hr = result.total_annual_loss_cost / self.operating_hours
            benefit.confidence = 0.90
            benefit.uncertainty_percent = 10.0

        elif result_type == "LossMinimizationResult":
            result = optimization_result
            benefit.steam_savings_klb_hr = (
                result.current_total_loss_klb_hr - result.optimized_total_loss_klb_hr
            )
            benefit.cost_savings_per_hr = result.total_savings_per_hr
            benefit.confidence = result.confidence
            benefit.uncertainty_percent = 20.0

        elif result_type == "HeaderOptimization":
            result = optimization_result
            benefit.cost_savings_per_hr = result.expected_savings_per_hr
            benefit.confidence = result.confidence
            benefit.uncertainty_percent = 15.0

        # Annualize benefits
        benefit.annual_cost_savings = benefit.cost_savings_per_hr * self.operating_hours
        benefit.annual_co2e_reduction_tons = (
            benefit.co2e_reduction_lb_hr * self.operating_hours / 2000
        )

        # Calculate bounds
        benefit.calculate_bounds()

        # Add assumptions
        benefit.assumptions = [
            f"Steam cost: ${steam_cost_per_klb}/klb",
            f"Fuel cost: ${fuel_cost_per_mmbtu}/MMBTU",
            f"Operating hours: {self.operating_hours}/year",
        ]

        return benefit

    def assess_implementation_risk(
        self,
        optimization_result: Any,
        system_state: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessment:
        """
        Assess implementation risk for a recommendation.

        Args:
            optimization_result: Result from an optimizer
            system_state: Current system state

        Returns:
            RiskAssessment with risk factors and mitigations
        """
        assessment = RiskAssessment()
        risks: List[Dict[str, Any]] = []
        mitigations: List[str] = []

        result_type = type(optimization_result).__name__
        confidence = getattr(optimization_result, 'confidence', 0.90)

        # Check confidence level
        if confidence < self.uncertainty_constraints.min_confidence_for_auto_action:
            risks.append({
                "category": RiskCategory.PROCESS.value,
                "description": f"Low confidence ({confidence:.0%})",
                "severity": "medium",
                "probability": "high",
            })
            mitigations.append("Implement changes gradually with monitoring")
            assessment.requires_operator_confirmation = True
            assessment.confirmation_reason = f"Confidence {confidence:.0%} below auto-action threshold"

        # Type-specific risks
        if result_type == "SprayOptimizationResult":
            result = optimization_result
            if result.approach_to_saturation_f < 30:
                risks.append({
                    "category": RiskCategory.SAFETY.value,
                    "description": "Close to saturation - wet steam risk",
                    "severity": "high",
                    "probability": "medium",
                })
                mitigations.append("Monitor outlet temperature continuously")
                assessment.requires_operator_confirmation = True

            if abs(result.valve_change_pct) > 20:
                risks.append({
                    "category": RiskCategory.PROCESS.value,
                    "description": "Large valve position change",
                    "severity": "medium",
                    "probability": "medium",
                })
                mitigations.append("Implement change in steps")

        elif result_type == "LoadAllocationResult":
            result = optimization_result
            changes = [a for a in result.allocations if a.change_required]
            if len(changes) > 2:
                risks.append({
                    "category": RiskCategory.PROCESS.value,
                    "description": "Multiple simultaneous boiler changes",
                    "severity": "medium",
                    "probability": "medium",
                })
                mitigations.append("Implement changes sequentially, 5 minutes apart")

        elif result_type == "HeaderOptimization":
            result = optimization_result
            if abs(result.pressure_change_psig) > 10:
                risks.append({
                    "category": RiskCategory.PROCESS.value,
                    "description": "Large pressure change",
                    "severity": "medium",
                    "probability": "low",
                })
                mitigations.append("Ramp pressure change over 15+ minutes")
                assessment.requires_operator_confirmation = True

        # Check system state if provided
        if system_state:
            if system_state.get("alarm_count", 0) > 0:
                risks.append({
                    "category": RiskCategory.SAFETY.value,
                    "description": "Active alarms present",
                    "severity": "high",
                    "probability": "medium",
                })
                assessment.requires_operator_confirmation = True
                assessment.confirmation_reason = "Active alarms require review"

            if system_state.get("maintenance_mode", False):
                risks.append({
                    "category": RiskCategory.EQUIPMENT.value,
                    "description": "Equipment in maintenance mode",
                    "severity": "medium",
                    "probability": "high",
                })

        # Calculate overall risk score
        severity_scores = {"high": 30, "medium": 15, "low": 5}
        probability_scores = {"high": 3, "medium": 2, "low": 1}

        total_score = 0
        for risk in risks:
            severity = severity_scores.get(risk.get("severity", "low"), 5)
            probability = probability_scores.get(risk.get("probability", "low"), 1)
            total_score += severity * probability

        assessment.risk_score = min(100, total_score)

        # Determine overall level
        if assessment.risk_score >= 60:
            assessment.overall_risk_level = "high"
            assessment.requires_engineering_review = True
        elif assessment.risk_score >= 30:
            assessment.overall_risk_level = "medium"
        else:
            assessment.overall_risk_level = "low"

        assessment.risks = risks
        assessment.mitigations = mitigations

        return assessment

    def _classify_optimization(
        self,
        optimization_result: Any,
        context: Dict[str, Any],
    ) -> Tuple[RecommendationType, RecommendationPriority]:
        """Classify optimization result into type and priority."""
        result_type = type(optimization_result).__name__

        # Type classification
        if result_type in ("SprayOptimizationResult", "HeaderOptimization", "PRVOptimization"):
            rec_type = RecommendationType.SETPOINT_CHANGE
        elif result_type in ("LoadAllocationResult",):
            rec_type = RecommendationType.EQUIPMENT_ACTION
        elif result_type in ("ReplacementPriority", "InspectionSchedule"):
            rec_type = RecommendationType.MAINTENANCE
        elif result_type in ("RoutingOptimization", "LossMinimizationResult"):
            rec_type = RecommendationType.OPERATIONAL
        else:
            rec_type = RecommendationType.INFORMATIONAL

        # Priority classification
        confidence = getattr(optimization_result, 'confidence', 0.90)
        constraints_ok = getattr(optimization_result, 'constraints_satisfied', True)

        if not constraints_ok:
            priority = RecommendationPriority.CRITICAL
        elif confidence < 0.7:
            priority = RecommendationPriority.LOW
        else:
            # Based on benefit magnitude
            savings = 0
            if hasattr(optimization_result, 'cost_savings_per_hour'):
                savings = optimization_result.cost_savings_per_hour
            elif hasattr(optimization_result, 'cost_savings_per_hr'):
                savings = optimization_result.cost_savings_per_hr
            elif hasattr(optimization_result, 'total_cost_savings_per_hr'):
                savings = optimization_result.total_cost_savings_per_hr

            if savings > 100:
                priority = RecommendationPriority.HIGH
            elif savings > 20:
                priority = RecommendationPriority.MEDIUM
            else:
                priority = RecommendationPriority.LOW

        return rec_type, priority

    def _generate_rationale(
        self,
        optimization_result: Any,
        context: Dict[str, Any],
    ) -> str:
        """Generate rationale for recommendation."""
        result_type = type(optimization_result).__name__
        confidence = getattr(optimization_result, 'confidence', 0.90)

        rationale_parts = []

        if result_type == "SprayOptimizationResult":
            result = optimization_result
            if result.spray_water_reduction_percent > 0:
                rationale_parts.append(
                    f"Reduces spray water usage by {result.spray_water_reduction_percent:.1f}%"
                )
            rationale_parts.append(
                f"Maintains {result.approach_to_saturation_f:.0f}F approach to saturation"
            )

        elif result_type == "LoadAllocationResult":
            result = optimization_result
            rationale_parts.append(
                f"Optimizes {result.optimization_objective} across {len(result.allocations)} boilers"
            )
            if result.improvement_percent > 0:
                rationale_parts.append(
                    f"Expected {result.improvement_percent:.1f}% cost improvement"
                )

        elif result_type == "RoutingOptimization":
            result = optimization_result
            rationale_parts.append(
                f"Increases condensate return rate from "
                f"{result.current_return_rate_percent:.0f}% to "
                f"{result.optimized_return_rate_percent:.0f}%"
            )
            rationale_parts.append(
                f"Recovers {result.total_recoverable_flow_lb_hr:.0f} lb/hr condensate"
            )

        elif result_type == "ReplacementPriority":
            result = optimization_result
            rationale_parts.append(
                f"Eliminates {result.total_steam_loss_lb_hr:.0f} lb/hr steam loss"
            )
            rationale_parts.append(
                f"Annual savings potential: ${result.annual_savings_if_complete:,.0f}"
            )

        # Add confidence statement
        rationale_parts.append(f"Analysis confidence: {confidence:.0%}")

        return ". ".join(rationale_parts) + "."

    def _create_verification_plan(
        self,
        optimization_result: Any,
        context: Dict[str, Any],
    ) -> VerificationPlan:
        """Create verification plan for recommendation."""
        result_type = type(optimization_result).__name__
        plan = VerificationPlan()

        if result_type == "SprayOptimizationResult":
            plan.verification_steps = [
                "Monitor desuperheater outlet temperature",
                "Verify temperature reaches new setpoint within 5 minutes",
                "Check for any wet steam alarms",
                "Confirm downstream temperature stability",
            ]
            plan.key_metrics = [
                "Outlet temperature",
                "Spray valve position",
                "Superheat margin",
            ]
            plan.expected_response_time_min = 10
            plan.success_criteria = [
                "Outlet temperature within +/- 5F of setpoint",
                "No wet steam conditions",
                "Stable operation for 15 minutes",
            ]
            plan.rollback_procedure = [
                "Return temperature setpoint to previous value",
                "Monitor for stabilization",
                "Report issue to engineering if instability persists",
            ]

        elif result_type == "LoadAllocationResult":
            plan.verification_steps = [
                "Implement boiler load changes sequentially",
                "Monitor header pressure stability",
                "Verify total steam production matches demand",
                "Check combustion parameters (O2, CO)",
            ]
            plan.key_metrics = [
                "Boiler loads",
                "Header pressure",
                "Total efficiency",
                "CO2 emissions",
            ]
            plan.expected_response_time_min = 15
            plan.success_criteria = [
                "All boilers at recommended loads",
                "Header pressure stable within +/- 2 psi",
                "Efficiency improved or maintained",
            ]
            plan.rollback_procedure = [
                "Return to previous load allocation",
                "Investigate cause of instability",
            ]

        else:
            # Generic plan
            plan.verification_steps = [
                "Implement recommended change",
                "Monitor key process variables",
                "Verify expected improvement achieved",
            ]
            plan.key_metrics = ["Primary process variable"]
            plan.expected_response_time_min = 30
            plan.success_criteria = ["Expected benefit realized"]
            plan.rollback_procedure = ["Reverse change if unsuccessful"]

        return plan

    def _create_escalation_path(
        self,
        priority: RecommendationPriority,
        context: Dict[str, Any],
    ) -> EscalationPath:
        """Create escalation path for recommendation."""
        path = EscalationPath()

        if priority == RecommendationPriority.CRITICAL:
            path.escalation_levels = [
                {"level": 1, "title": "Shift Supervisor", "timeout_min": 5},
                {"level": 2, "title": "Plant Manager", "timeout_min": 15},
                {"level": 3, "title": "VP Operations", "timeout_min": 30},
            ]
            path.auto_escalate_after_min = 5
        elif priority == RecommendationPriority.HIGH:
            path.escalation_levels = [
                {"level": 1, "title": "Shift Supervisor", "timeout_min": 15},
                {"level": 2, "title": "Plant Manager", "timeout_min": 60},
            ]
            path.auto_escalate_after_min = 15
        else:
            path.escalation_levels = [
                {"level": 1, "title": "Shift Supervisor", "timeout_min": 60},
            ]
            path.auto_escalate_after_min = 60

        # Get contacts from context if available
        path.escalation_contacts = context.get(
            "escalation_contacts",
            ["shift_supervisor@plant.com"]
        )

        return path

    def _extract_target_equipment(
        self,
        optimization_result: Any,
    ) -> List[str]:
        """Extract target equipment IDs from optimization result."""
        result_type = type(optimization_result).__name__

        if result_type == "SprayOptimizationResult":
            return [optimization_result.desuperheater_id]
        elif result_type == "LoadAllocationResult":
            return [a.boiler_id for a in optimization_result.allocations if a.change_required]
        elif result_type == "HeaderOptimization":
            return [optimization_result.header_id]
        elif result_type == "ReplacementPriority":
            return [t.trap_id for t in optimization_result.tasks[:10]]
        else:
            return []

    def _extract_action_details(
        self,
        optimization_result: Any,
    ) -> Dict[str, Any]:
        """Extract detailed action parameters."""
        result_type = type(optimization_result).__name__

        if result_type == "SprayOptimizationResult":
            result = optimization_result
            return {
                "setpoint_type": "temperature",
                "current_value": result.recommended_temp_setpoint_f - result.temp_change_f,
                "new_value": result.recommended_temp_setpoint_f,
                "unit": "F",
            }
        elif result_type == "LoadAllocationResult":
            return {
                "action_type": "load_rebalance",
                "changes": [
                    {
                        "equipment": a.boiler_id,
                        "from": a.current_load_percent,
                        "to": a.recommended_load_percent,
                    }
                    for a in optimization_result.allocations if a.change_required
                ],
            }
        else:
            return {}

    def _extract_supporting_data(
        self,
        optimization_result: Any,
    ) -> Dict[str, Any]:
        """Extract supporting data for the recommendation."""
        result_type = type(optimization_result).__name__
        data = {"result_type": result_type}

        if hasattr(optimization_result, 'timestamp'):
            data["analysis_timestamp"] = optimization_result.timestamp.isoformat()
        if hasattr(optimization_result, 'confidence'):
            data["confidence"] = optimization_result.confidence
        if hasattr(optimization_result, 'provenance_hash'):
            data["provenance_hash"] = optimization_result.provenance_hash

        return data

    def _generate_provenance_hash(
        self,
        recommendation: Recommendation,
    ) -> str:
        """Generate SHA-256 provenance hash for recommendation."""
        data = (
            f"{recommendation.recommendation_id}"
            f"{recommendation.action}"
            f"{recommendation.expected_benefit.cost_savings_per_hr}"
            f"{recommendation.timestamp.isoformat()}"
        )
        return hashlib.sha256(data.encode()).hexdigest()
