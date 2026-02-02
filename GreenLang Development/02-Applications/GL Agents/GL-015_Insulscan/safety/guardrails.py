# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Operational Guardrails

Implements safety guardrails for insulation scanning and repair
recommendation systems:

1. Safety Limits for Repair Recommendations:
   - Maximum investment thresholds
   - Minimum ROI requirements
   - Budget constraint enforcement

2. Priority Override Logic:
   - Safety-critical asset prioritization
   - Regulatory compliance requirements
   - Emergency repair escalation

3. Personnel Safety Temperature Limits:
   - Burn prevention thresholds
   - Touch temperature warnings
   - Exposure time limits

4. Plant Safety System Integration:
   - Coordination with maintenance systems
   - Safety work permit considerations
   - Lockout/tagout requirements

Safety Principles:
- Personnel safety is non-negotiable
- Recommendations only, no autonomous actions
- Fail safe on data quality issues
- Full audit trail for all decisions

Standards Reference:
- OSHA 1910.147: Control of Hazardous Energy
- OSHA 29 CFR 1910.132: Personal Protective Equipment
- ASTM C1055: Safe Surface Temperature Limit

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from .exceptions import (
    SafetyLimitExceededError,
    BurnRiskError,
    InvestmentLimitExceededError,
    ViolationContext,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ActionType(str, Enum):
    """Types of actions the system can recommend."""

    RECOMMENDATION = "recommendation"  # Repair/inspection recommendation
    ALERT = "alert"  # Safety alert notification
    INFORMATION = "information"  # Status information
    PRIORITY_OVERRIDE = "priority_override"  # Priority escalation
    BUDGET_REQUEST = "budget_request"  # Budget approval request
    EMERGENCY = "emergency"  # Emergency action required


class GuardrailDecision(str, Enum):
    """Decision outcome from guardrail check."""

    ALLOW = "allow"  # Action is permitted
    ALLOW_WITH_WARNING = "allow_with_warning"  # Permitted but flagged
    REQUIRE_APPROVAL = "require_approval"  # Needs management approval
    REQUIRE_SAFETY_REVIEW = "require_safety_review"  # Safety review needed
    BLOCK = "block"  # Action is blocked
    ESCALATE = "escalate"  # Escalate to higher authority


class SafetyPriority(str, Enum):
    """Priority levels for safety-critical assets."""

    NORMAL = "normal"  # Standard maintenance
    ELEVATED = "elevated"  # Above normal attention
    HIGH = "high"  # Prioritized maintenance
    CRITICAL = "critical"  # Safety-critical asset
    EMERGENCY = "emergency"  # Immediate attention required


class BurnRiskLevel(str, Enum):
    """Personnel burn risk classification."""

    NONE = "none"  # Safe to touch
    LOW = "low"  # Brief contact safe
    MODERATE = "moderate"  # Caution required
    HIGH = "high"  # PPE required
    EXTREME = "extreme"  # No touch zone


class InvestmentCategory(str, Enum):
    """Investment approval categories."""

    ROUTINE = "routine"  # Routine maintenance budget
    MINOR = "minor"  # Minor repairs
    MAJOR = "major"  # Major repairs requiring approval
    CAPITAL = "capital"  # Capital expenditure
    EMERGENCY = "emergency"  # Emergency spending


# =============================================================================
# CONFIGURATION
# =============================================================================


class PersonnelSafetyConfig(BaseModel):
    """
    Configuration for personnel safety limits.

    Based on ASTM C1055 and OSHA burn prevention standards.
    """

    # Touch temperature limits (Celsius)
    safe_touch_temp_C: float = Field(
        default=48.0,
        ge=40.0,
        le=60.0,
        description="Maximum safe continuous touch temperature"
    )
    caution_temp_C: float = Field(
        default=52.0,
        ge=48.0,
        le=65.0,
        description="Temperature requiring caution"
    )
    warning_temp_C: float = Field(
        default=60.0,
        ge=55.0,
        le=75.0,
        description="Temperature requiring PPE"
    )
    danger_temp_C: float = Field(
        default=70.0,
        ge=65.0,
        le=90.0,
        description="Temperature classified as burn hazard"
    )
    extreme_danger_temp_C: float = Field(
        default=82.0,
        ge=75.0,
        le=100.0,
        description="Temperature classified as severe burn hazard"
    )

    # Cold temperature limits
    cold_caution_temp_C: float = Field(
        default=0.0,
        ge=-40.0,
        le=10.0,
        description="Cold temperature requiring caution"
    )
    cold_danger_temp_C: float = Field(
        default=-20.0,
        ge=-60.0,
        le=0.0,
        description="Cold temperature classified as hazard"
    )

    # Exposure time limits (seconds)
    max_contact_time_caution_s: float = Field(
        default=10.0,
        ge=1.0,
        le=30.0,
        description="Max contact time at caution temp"
    )
    max_contact_time_warning_s: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Max contact time at warning temp"
    )


class InvestmentLimitsConfig(BaseModel):
    """
    Configuration for investment approval thresholds.

    Defines budget limits for different approval levels.
    """

    routine_limit_usd: float = Field(
        default=5000.0,
        ge=100.0,
        le=50000.0,
        description="Maximum routine maintenance without approval"
    )
    minor_limit_usd: float = Field(
        default=25000.0,
        ge=5000.0,
        le=100000.0,
        description="Minor repair limit (supervisor approval)"
    )
    major_limit_usd: float = Field(
        default=100000.0,
        ge=25000.0,
        le=500000.0,
        description="Major repair limit (manager approval)"
    )
    capital_threshold_usd: float = Field(
        default=250000.0,
        ge=100000.0,
        le=1000000.0,
        description="Capital expenditure threshold"
    )

    # ROI requirements
    min_roi_percent: float = Field(
        default=15.0,
        ge=0.0,
        le=100.0,
        description="Minimum required ROI percentage"
    )
    max_payback_years: float = Field(
        default=5.0,
        ge=0.5,
        le=20.0,
        description="Maximum acceptable payback period"
    )


class PriorityOverrideConfig(BaseModel):
    """
    Configuration for priority override rules.

    Defines when safety priority overrides normal scheduling.
    """

    # Safety criticality thresholds
    critical_asset_tags: List[str] = Field(
        default_factory=lambda: ["safety", "sil", "critical", "emergency"],
        description="Asset tags that indicate safety-critical equipment"
    )
    regulatory_asset_tags: List[str] = Field(
        default_factory=lambda: ["epa", "osha", "regulatory", "compliance"],
        description="Asset tags indicating regulatory compliance requirements"
    )

    # Heat loss thresholds for priority escalation (W/m2)
    high_heat_loss_threshold_W_m2: float = Field(
        default=500.0,
        ge=100.0,
        le=2000.0,
        description="Heat loss threshold for priority escalation"
    )
    critical_heat_loss_threshold_W_m2: float = Field(
        default=1500.0,
        ge=500.0,
        le=5000.0,
        description="Heat loss threshold for critical priority"
    )

    # Condition score thresholds
    critical_condition_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Condition score threshold for critical priority"
    )
    high_condition_threshold: float = Field(
        default=0.4,
        ge=0.2,
        le=0.7,
        description="Condition score threshold for high priority"
    )


class InsulationGuardrailsConfig(BaseModel):
    """
    Master configuration for insulation operational guardrails.

    Combines all safety and operational limits.
    """

    personnel_safety: PersonnelSafetyConfig = Field(
        default_factory=PersonnelSafetyConfig
    )
    investment_limits: InvestmentLimitsConfig = Field(
        default_factory=InvestmentLimitsConfig
    )
    priority_override: PriorityOverrideConfig = Field(
        default_factory=PriorityOverrideConfig
    )

    # Global settings
    require_safety_review_for_critical: bool = Field(
        default=True,
        description="Require safety review for critical assets"
    )
    log_all_decisions: bool = Field(
        default=True,
        description="Log all guardrail decisions for audit"
    )
    fail_safe_on_missing_data: bool = Field(
        default=True,
        description="Use conservative estimates when data missing"
    )
    max_recommendations_per_asset: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum open recommendations per asset"
    )


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class GuardrailCheckResult:
    """
    Result of a guardrail check.

    Provides complete audit trail for safety decisions.

    Attributes:
        asset_id: Insulation asset identifier
        action_type: Type of action being checked
        decision: Guardrail decision
        reason: Reason for the decision
        safety_priority: Determined safety priority
        burn_risk_level: Personnel burn risk level
        investment_category: Investment approval category
        requires_approval: Whether approval is needed
        approving_authority: Who needs to approve
        blocked_by: Which guardrail blocked (if blocked)
        warnings: List of warnings
        timestamp: When check was performed
        provenance_hash: SHA-256 hash for audit
    """

    asset_id: str
    action_type: ActionType
    decision: GuardrailDecision
    reason: str
    safety_priority: SafetyPriority = SafetyPriority.NORMAL
    burn_risk_level: BurnRiskLevel = BurnRiskLevel.NONE
    investment_category: InvestmentCategory = InvestmentCategory.ROUTINE
    requires_approval: bool = False
    approving_authority: Optional[str] = None
    blocked_by: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.asset_id}|{self.action_type.value}|"
                f"{self.decision.value}|{self.safety_priority.value}|"
                f"{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class RepairRecommendation:
    """
    Repair recommendation data for guardrail evaluation.

    Attributes:
        asset_id: Insulation asset identifier
        estimated_cost_usd: Estimated repair cost
        expected_savings_usd_year: Expected annual savings
        payback_years: Simple payback period
        roi_percent: Return on investment
        current_condition_score: Current condition [0, 1]
        surface_temperature_C: Surface temperature
        heat_loss_W_m2: Current heat loss
        asset_tags: Tags for the asset
        location: Physical location
        priority_requested: Requested priority level
    """

    asset_id: str
    estimated_cost_usd: float = 0.0
    expected_savings_usd_year: float = 0.0
    payback_years: float = 0.0
    roi_percent: float = 0.0
    current_condition_score: float = 0.5
    surface_temperature_C: Optional[float] = None
    heat_loss_W_m2: Optional[float] = None
    asset_tags: List[str] = field(default_factory=list)
    location: str = "unknown"
    priority_requested: SafetyPriority = SafetyPriority.NORMAL


@dataclass
class SafetyAlert:
    """
    Safety alert generated by guardrails.

    Attributes:
        alert_id: Unique alert identifier
        asset_id: Asset that triggered alert
        alert_type: Type of safety concern
        severity: Alert severity
        message: Human-readable message
        recommended_action: What action to take
        requires_immediate_action: If true, immediate attention needed
        timestamp: When alert was generated
    """

    alert_id: str
    asset_id: str
    alert_type: str
    severity: str
    message: str
    recommended_action: str
    requires_immediate_action: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# INSULATION GUARDRAILS
# =============================================================================


class InsulationGuardrails:
    """
    Implements operational safety guardrails for insulation assessments.

    Enforces safety limits and provides recommendations-only output:
    1. Personnel safety temperature limits
    2. Investment threshold enforcement
    3. Priority override for safety-critical assets
    4. Integration with plant safety systems

    Safety Principles:
    - Personnel safety is non-negotiable
    - Recommendations only, no autonomous actions
    - Fail safe on data quality issues
    - Full audit trail for all decisions

    Example:
        >>> config = InsulationGuardrailsConfig()
        >>> guardrails = InsulationGuardrails(config)
        >>>
        >>> # Check repair recommendation
        >>> recommendation = RepairRecommendation(
        ...     asset_id="INS-PIPE-101",
        ...     estimated_cost_usd=50000,
        ...     surface_temperature_C=75.0,
        ... )
        >>> result = guardrails.evaluate_recommendation(recommendation)
        >>>
        >>> if result.decision == GuardrailDecision.BLOCK:
        ...     raise SafetyError(result.reason)

    Author: GL-BackendDeveloper
    Version: 1.0.0
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        config: Optional[InsulationGuardrailsConfig] = None,
        decision_callback: Optional[Callable[[GuardrailCheckResult], None]] = None,
        alert_callback: Optional[Callable[[SafetyAlert], None]] = None,
    ) -> None:
        """
        Initialize operational guardrails.

        Args:
            config: Guardrails configuration
            decision_callback: Optional callback for all decisions
            alert_callback: Optional callback for safety alerts
        """
        self.config = config or InsulationGuardrailsConfig()
        self._lock = threading.RLock()
        self._decision_history: List[GuardrailCheckResult] = []
        self._decision_callbacks: List[Callable[[GuardrailCheckResult], None]] = []
        self._alert_callbacks: List[Callable[[SafetyAlert], None]] = []
        self._active_alerts: Dict[str, SafetyAlert] = {}
        self._blocked_assets: Set[str] = set()

        if decision_callback:
            self._decision_callbacks.append(decision_callback)
        if alert_callback:
            self._alert_callbacks.append(alert_callback)

        logger.info(
            f"InsulationGuardrails initialized: "
            f"safe_touch_temp={self.config.personnel_safety.safe_touch_temp_C}C, "
            f"routine_limit=${self.config.investment_limits.routine_limit_usd:,.0f}"
        )

    # =========================================================================
    # PERSONNEL SAFETY CHECKS
    # =========================================================================

    def check_burn_risk(
        self,
        asset_id: str,
        surface_temp_C: float,
    ) -> Tuple[BurnRiskLevel, str]:
        """
        Check burn risk level for a surface temperature.

        Based on ASTM C1055 and OSHA standards.

        Args:
            asset_id: Asset identifier
            surface_temp_C: Surface temperature in Celsius

        Returns:
            Tuple of (BurnRiskLevel, warning_message)
        """
        safety = self.config.personnel_safety

        if surface_temp_C <= safety.safe_touch_temp_C:
            return BurnRiskLevel.NONE, ""

        elif surface_temp_C <= safety.caution_temp_C:
            return BurnRiskLevel.LOW, f"Surface at {surface_temp_C:.1f}C - limit contact time"

        elif surface_temp_C <= safety.warning_temp_C:
            return BurnRiskLevel.MODERATE, (
                f"Surface at {surface_temp_C:.1f}C - PPE recommended, "
                f"max contact {safety.max_contact_time_caution_s:.0f}s"
            )

        elif surface_temp_C <= safety.danger_temp_C:
            return BurnRiskLevel.HIGH, (
                f"BURN HAZARD: Surface at {surface_temp_C:.1f}C - "
                f"PPE required, max contact {safety.max_contact_time_warning_s:.0f}s"
            )

        else:
            return BurnRiskLevel.EXTREME, (
                f"SEVERE BURN HAZARD: Surface at {surface_temp_C:.1f}C - "
                f"NO TOUCH ZONE - Immediate repair recommended"
            )

    def check_cold_hazard(
        self,
        asset_id: str,
        surface_temp_C: float,
    ) -> Tuple[BurnRiskLevel, str]:
        """
        Check cold hazard for low-temperature surfaces.

        Args:
            asset_id: Asset identifier
            surface_temp_C: Surface temperature in Celsius

        Returns:
            Tuple of (hazard_level, warning_message)
        """
        safety = self.config.personnel_safety

        if surface_temp_C >= safety.cold_caution_temp_C:
            return BurnRiskLevel.NONE, ""

        elif surface_temp_C >= safety.cold_danger_temp_C:
            return BurnRiskLevel.MODERATE, (
                f"Cold hazard: Surface at {surface_temp_C:.1f}C - "
                f"Insulated gloves required"
            )

        else:
            return BurnRiskLevel.HIGH, (
                f"SEVERE COLD HAZARD: Surface at {surface_temp_C:.1f}C - "
                f"Full cold protection PPE required"
            )

    def _generate_burn_hazard_alert(
        self,
        asset_id: str,
        surface_temp_C: float,
        burn_risk: BurnRiskLevel,
        warning_msg: str,
    ) -> None:
        """Generate safety alert for burn hazard."""
        if burn_risk in (BurnRiskLevel.HIGH, BurnRiskLevel.EXTREME):
            alert = SafetyAlert(
                alert_id=f"burn_{asset_id}_{int(datetime.now().timestamp())}",
                asset_id=asset_id,
                alert_type="burn_hazard",
                severity="critical" if burn_risk == BurnRiskLevel.EXTREME else "high",
                message=warning_msg,
                recommended_action="Install warning signage and schedule immediate repair",
                requires_immediate_action=burn_risk == BurnRiskLevel.EXTREME,
            )
            self._emit_alert(alert)

    # =========================================================================
    # INVESTMENT LIMIT CHECKS
    # =========================================================================

    def check_investment_limits(
        self,
        recommendation: RepairRecommendation,
    ) -> Tuple[InvestmentCategory, Optional[str], List[str]]:
        """
        Check investment against approval thresholds.

        Args:
            recommendation: Repair recommendation to evaluate

        Returns:
            Tuple of (category, approving_authority, warnings)
        """
        limits = self.config.investment_limits
        cost = recommendation.estimated_cost_usd
        warnings = []

        # Determine category
        if cost <= limits.routine_limit_usd:
            category = InvestmentCategory.ROUTINE
            authority = None

        elif cost <= limits.minor_limit_usd:
            category = InvestmentCategory.MINOR
            authority = "supervisor"

        elif cost <= limits.major_limit_usd:
            category = InvestmentCategory.MAJOR
            authority = "maintenance_manager"

        elif cost <= limits.capital_threshold_usd:
            category = InvestmentCategory.MAJOR
            authority = "plant_manager"

        else:
            category = InvestmentCategory.CAPITAL
            authority = "capital_committee"

        # Check ROI requirements
        if recommendation.roi_percent < limits.min_roi_percent:
            warnings.append(
                f"ROI {recommendation.roi_percent:.1f}% below minimum "
                f"{limits.min_roi_percent:.1f}%"
            )

        # Check payback period
        if recommendation.payback_years > limits.max_payback_years:
            warnings.append(
                f"Payback {recommendation.payback_years:.1f} years exceeds "
                f"maximum {limits.max_payback_years:.1f} years"
            )

        return category, authority, warnings

    # =========================================================================
    # PRIORITY OVERRIDE LOGIC
    # =========================================================================

    def determine_priority(
        self,
        recommendation: RepairRecommendation,
    ) -> Tuple[SafetyPriority, List[str]]:
        """
        Determine safety priority with override logic.

        Considers:
        - Asset tags (safety-critical, regulatory)
        - Heat loss levels
        - Condition score
        - Surface temperature (burn risk)

        Args:
            recommendation: Repair recommendation

        Returns:
            Tuple of (determined_priority, override_reasons)
        """
        override_config = self.config.priority_override
        priority = recommendation.priority_requested
        reasons = []

        # Check for safety-critical tags
        safety_tags = set(t.lower() for t in recommendation.asset_tags)
        critical_tags = set(t.lower() for t in override_config.critical_asset_tags)
        regulatory_tags = set(t.lower() for t in override_config.regulatory_asset_tags)

        if safety_tags & critical_tags:
            if priority.value < SafetyPriority.CRITICAL.value:
                priority = SafetyPriority.CRITICAL
                matching_tags = safety_tags & critical_tags
                reasons.append(f"Safety-critical tags present: {matching_tags}")

        if safety_tags & regulatory_tags:
            if priority.value < SafetyPriority.HIGH.value:
                priority = SafetyPriority.HIGH
                matching_tags = safety_tags & regulatory_tags
                reasons.append(f"Regulatory compliance required: {matching_tags}")

        # Check heat loss threshold
        if recommendation.heat_loss_W_m2 is not None:
            if recommendation.heat_loss_W_m2 >= override_config.critical_heat_loss_threshold_W_m2:
                if priority.value < SafetyPriority.CRITICAL.value:
                    priority = SafetyPriority.CRITICAL
                    reasons.append(
                        f"Heat loss {recommendation.heat_loss_W_m2:.0f} W/m2 >= "
                        f"critical threshold {override_config.critical_heat_loss_threshold_W_m2}"
                    )
            elif recommendation.heat_loss_W_m2 >= override_config.high_heat_loss_threshold_W_m2:
                if priority.value < SafetyPriority.HIGH.value:
                    priority = SafetyPriority.HIGH
                    reasons.append(
                        f"Heat loss {recommendation.heat_loss_W_m2:.0f} W/m2 >= "
                        f"high threshold {override_config.high_heat_loss_threshold_W_m2}"
                    )

        # Check condition score threshold
        if recommendation.current_condition_score <= override_config.critical_condition_threshold:
            if priority.value < SafetyPriority.CRITICAL.value:
                priority = SafetyPriority.CRITICAL
                reasons.append(
                    f"Condition score {recommendation.current_condition_score:.2f} <= "
                    f"critical threshold {override_config.critical_condition_threshold}"
                )
        elif recommendation.current_condition_score <= override_config.high_condition_threshold:
            if priority.value < SafetyPriority.HIGH.value:
                priority = SafetyPriority.HIGH
                reasons.append(
                    f"Condition score {recommendation.current_condition_score:.2f} <= "
                    f"high threshold {override_config.high_condition_threshold}"
                )

        # Check burn risk - immediate escalation
        if recommendation.surface_temperature_C is not None:
            burn_risk, _ = self.check_burn_risk(
                recommendation.asset_id,
                recommendation.surface_temperature_C,
            )
            if burn_risk == BurnRiskLevel.EXTREME:
                priority = SafetyPriority.EMERGENCY
                reasons.append(
                    f"Extreme burn hazard at {recommendation.surface_temperature_C:.1f}C"
                )
            elif burn_risk == BurnRiskLevel.HIGH:
                if priority.value < SafetyPriority.CRITICAL.value:
                    priority = SafetyPriority.CRITICAL
                    reasons.append(
                        f"Burn hazard at {recommendation.surface_temperature_C:.1f}C"
                    )

        return priority, reasons

    # =========================================================================
    # MAIN EVALUATION METHOD
    # =========================================================================

    def evaluate_recommendation(
        self,
        recommendation: RepairRecommendation,
        bypass_investment_check: bool = False,
    ) -> GuardrailCheckResult:
        """
        Evaluate a repair recommendation against all guardrails.

        Args:
            recommendation: Repair recommendation to evaluate
            bypass_investment_check: Skip investment threshold check

        Returns:
            GuardrailCheckResult with decision and details
        """
        with self._lock:
            warnings = []
            blocked_by = None
            decision = GuardrailDecision.ALLOW
            requires_approval = False
            approving_authority = None

            # 1. Check personnel safety (burn risk)
            burn_risk = BurnRiskLevel.NONE
            if recommendation.surface_temperature_C is not None:
                burn_risk, burn_warning = self.check_burn_risk(
                    recommendation.asset_id,
                    recommendation.surface_temperature_C,
                )
                if burn_warning:
                    warnings.append(burn_warning)

                # Generate alert for high burn risk
                if burn_risk in (BurnRiskLevel.HIGH, BurnRiskLevel.EXTREME):
                    self._generate_burn_hazard_alert(
                        recommendation.asset_id,
                        recommendation.surface_temperature_C,
                        burn_risk,
                        burn_warning,
                    )

            # Check cold hazard
            if recommendation.surface_temperature_C is not None and recommendation.surface_temperature_C < 10:
                cold_risk, cold_warning = self.check_cold_hazard(
                    recommendation.asset_id,
                    recommendation.surface_temperature_C,
                )
                if cold_warning:
                    warnings.append(cold_warning)

            # 2. Check investment limits
            investment_category = InvestmentCategory.ROUTINE
            if not bypass_investment_check:
                investment_category, authority, inv_warnings = self.check_investment_limits(
                    recommendation
                )
                warnings.extend(inv_warnings)

                if authority:
                    requires_approval = True
                    approving_authority = authority

                # Check if investment is allowed
                if investment_category == InvestmentCategory.CAPITAL:
                    decision = GuardrailDecision.REQUIRE_APPROVAL
                elif inv_warnings:
                    decision = GuardrailDecision.ALLOW_WITH_WARNING

            # 3. Determine priority with override logic
            safety_priority, priority_reasons = self.determine_priority(recommendation)

            # Add priority override reasons to warnings if priority was elevated
            if priority_reasons:
                for reason in priority_reasons:
                    warnings.append(f"Priority override: {reason}")

            # 4. Check if safety review is required
            if (
                self.config.require_safety_review_for_critical and
                safety_priority in (SafetyPriority.CRITICAL, SafetyPriority.EMERGENCY)
            ):
                decision = GuardrailDecision.REQUIRE_SAFETY_REVIEW
                requires_approval = True
                if approving_authority is None:
                    approving_authority = "safety_department"

            # 5. Handle emergency priority
            if safety_priority == SafetyPriority.EMERGENCY:
                decision = GuardrailDecision.ESCALATE

            # 6. Check if asset is blocked
            if recommendation.asset_id in self._blocked_assets:
                decision = GuardrailDecision.BLOCK
                blocked_by = "asset_blocked"
                warnings.append("Asset is currently blocked from recommendations")

            # Build result
            reason = self._build_decision_reason(
                recommendation, decision, warnings, priority_reasons
            )

            result = GuardrailCheckResult(
                asset_id=recommendation.asset_id,
                action_type=ActionType.RECOMMENDATION,
                decision=decision,
                reason=reason,
                safety_priority=safety_priority,
                burn_risk_level=burn_risk,
                investment_category=investment_category,
                requires_approval=requires_approval,
                approving_authority=approving_authority,
                blocked_by=blocked_by,
                warnings=warnings,
            )

            # Record decision
            self._record_decision(result)

            return result

    def _build_decision_reason(
        self,
        recommendation: RepairRecommendation,
        decision: GuardrailDecision,
        warnings: List[str],
        priority_reasons: List[str],
    ) -> str:
        """Build human-readable decision reason."""
        parts = [f"Evaluated {recommendation.asset_id}"]

        if decision == GuardrailDecision.ALLOW:
            parts.append("- Recommendation approved")
        elif decision == GuardrailDecision.ALLOW_WITH_WARNING:
            parts.append("- Approved with warnings")
        elif decision == GuardrailDecision.REQUIRE_APPROVAL:
            parts.append(f"- Requires approval for ${recommendation.estimated_cost_usd:,.0f}")
        elif decision == GuardrailDecision.REQUIRE_SAFETY_REVIEW:
            parts.append("- Safety review required due to critical priority")
        elif decision == GuardrailDecision.ESCALATE:
            parts.append("- EMERGENCY: Immediate escalation required")
        elif decision == GuardrailDecision.BLOCK:
            parts.append("- Recommendation blocked")

        if priority_reasons:
            parts.append(f"Priority overrides: {'; '.join(priority_reasons)}")

        if warnings:
            parts.append(f"Warnings: {len(warnings)}")

        return ". ".join(parts)

    # =========================================================================
    # BATCH EVALUATION
    # =========================================================================

    def evaluate_batch(
        self,
        recommendations: List[RepairRecommendation],
    ) -> Dict[str, GuardrailCheckResult]:
        """
        Evaluate multiple recommendations.

        Args:
            recommendations: List of recommendations to evaluate

        Returns:
            Dict mapping asset_id to GuardrailCheckResult
        """
        results = {}
        for rec in recommendations:
            results[rec.asset_id] = self.evaluate_recommendation(rec)
        return results

    # =========================================================================
    # ASSET MANAGEMENT
    # =========================================================================

    def block_asset(self, asset_id: str, reason: str) -> None:
        """Block an asset from receiving recommendations."""
        with self._lock:
            self._blocked_assets.add(asset_id)
            logger.warning(f"Asset {asset_id} blocked: {reason}")

    def unblock_asset(self, asset_id: str) -> bool:
        """Unblock an asset."""
        with self._lock:
            if asset_id in self._blocked_assets:
                self._blocked_assets.remove(asset_id)
                logger.info(f"Asset {asset_id} unblocked")
                return True
            return False

    def get_blocked_assets(self) -> Set[str]:
        """Get set of blocked asset IDs."""
        with self._lock:
            return self._blocked_assets.copy()

    # =========================================================================
    # CALLBACKS AND HISTORY
    # =========================================================================

    def _record_decision(self, result: GuardrailCheckResult) -> None:
        """Record decision for audit trail."""
        self._decision_history.append(result)

        # Trim history if needed
        max_history = 10000
        if len(self._decision_history) > max_history:
            self._decision_history = self._decision_history[-max_history:]

        # Log if configured
        if self.config.log_all_decisions:
            log_level = logging.INFO
            if result.decision == GuardrailDecision.BLOCK:
                log_level = logging.WARNING
            elif result.decision in (
                GuardrailDecision.REQUIRE_SAFETY_REVIEW,
                GuardrailDecision.ESCALATE,
            ):
                log_level = logging.WARNING

            logger.log(
                log_level,
                f"Guardrail decision: asset={result.asset_id}, "
                f"decision={result.decision.value}, "
                f"priority={result.safety_priority.value}"
            )

        # Invoke callbacks
        for callback in self._decision_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Decision callback failed: {e}")

    def _emit_alert(self, alert: SafetyAlert) -> None:
        """Emit a safety alert."""
        self._active_alerts[alert.asset_id] = alert

        logger.warning(
            f"Safety alert: type={alert.alert_type}, "
            f"asset={alert.asset_id}, severity={alert.severity}"
        )

        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def register_decision_callback(
        self,
        callback: Callable[[GuardrailCheckResult], None],
    ) -> None:
        """Register callback for guardrail decisions."""
        if callable(callback):
            self._decision_callbacks.append(callback)

    def register_alert_callback(
        self,
        callback: Callable[[SafetyAlert], None],
    ) -> None:
        """Register callback for safety alerts."""
        if callable(callback):
            self._alert_callbacks.append(callback)

    def get_decision_history(
        self,
        asset_id: Optional[str] = None,
        decision: Optional[GuardrailDecision] = None,
        limit: int = 100,
    ) -> List[GuardrailCheckResult]:
        """
        Get decision history with optional filtering.

        Args:
            asset_id: Filter by asset
            decision: Filter by decision type
            limit: Maximum entries to return

        Returns:
            List of GuardrailCheckResult
        """
        with self._lock:
            results = self._decision_history.copy()

            if asset_id:
                results = [r for r in results if r.asset_id == asset_id]

            if decision:
                results = [r for r in results if r.decision == decision]

            return list(reversed(results[-limit:]))

    def get_active_alerts(self) -> Dict[str, SafetyAlert]:
        """Get currently active safety alerts."""
        with self._lock:
            return self._active_alerts.copy()

    def clear_alert(self, asset_id: str) -> bool:
        """Clear a safety alert for an asset."""
        with self._lock:
            if asset_id in self._active_alerts:
                del self._active_alerts[asset_id]
                logger.info(f"Alert cleared for {asset_id}")
                return True
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get guardrails statistics."""
        with self._lock:
            decisions = self._decision_history

            if not decisions:
                return {
                    "total_decisions": 0,
                    "blocked_assets": len(self._blocked_assets),
                    "active_alerts": len(self._active_alerts),
                }

            decision_counts = {}
            priority_counts = {}

            for d in decisions:
                decision_counts[d.decision.value] = decision_counts.get(d.decision.value, 0) + 1
                priority_counts[d.safety_priority.value] = priority_counts.get(d.safety_priority.value, 0) + 1

            return {
                "total_decisions": len(decisions),
                "decision_distribution": decision_counts,
                "priority_distribution": priority_counts,
                "blocked_assets": len(self._blocked_assets),
                "active_alerts": len(self._active_alerts),
                "config": {
                    "safe_touch_temp_C": self.config.personnel_safety.safe_touch_temp_C,
                    "routine_limit_usd": self.config.investment_limits.routine_limit_usd,
                    "min_roi_percent": self.config.investment_limits.min_roi_percent,
                },
            }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def check_burn_risk_simple(surface_temp_C: float) -> Tuple[str, bool]:
    """
    Quick check if surface temperature poses burn risk.

    Args:
        surface_temp_C: Surface temperature

    Returns:
        Tuple of (risk_level, requires_action)
    """
    if surface_temp_C <= 48.0:
        return "safe", False
    elif surface_temp_C <= 52.0:
        return "caution", False
    elif surface_temp_C <= 60.0:
        return "warning", True
    elif surface_temp_C <= 70.0:
        return "danger", True
    else:
        return "extreme_danger", True


def get_investment_category(cost_usd: float) -> str:
    """
    Quick lookup of investment category.

    Args:
        cost_usd: Estimated cost

    Returns:
        Category name
    """
    if cost_usd <= 5000:
        return "routine"
    elif cost_usd <= 25000:
        return "minor"
    elif cost_usd <= 100000:
        return "major"
    else:
        return "capital"


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ActionType",
    "GuardrailDecision",
    "SafetyPriority",
    "BurnRiskLevel",
    "InvestmentCategory",
    # Config
    "PersonnelSafetyConfig",
    "InvestmentLimitsConfig",
    "PriorityOverrideConfig",
    "InsulationGuardrailsConfig",
    # Data models
    "GuardrailCheckResult",
    "RepairRecommendation",
    "SafetyAlert",
    # Main class
    "InsulationGuardrails",
    # Convenience functions
    "check_burn_risk_simple",
    "get_investment_category",
]
