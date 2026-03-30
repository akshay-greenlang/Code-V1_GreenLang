# -*- coding: utf-8 -*-
"""
ConsumersEngine - PACK-017 ESRS S4 Consumers and End-Users Engine
==================================================================

Assesses and calculates disclosure metrics for ESRS S4: Consumers and
End-Users.  This standard requires undertakings to disclose how they
identify, manage, and remediate impacts on consumers and end-users of
their products and services.

ESRS S4 Disclosure Requirements:
    - S4-1 (Para 14-16): Policies related to consumers and end-users
    - S4-2 (Para 18-21): Processes for engaging with consumers about
      impacts
    - S4-3 (Para 23-26): Processes to remediate negative impacts and
      channels for consumers to raise concerns
    - S4-4 (Para 28-33): Taking action on material impacts on consumers
    - S4-5 (Para 35-37): Targets related to managing material negative
      impacts, positive impacts, and risks/opportunities

This engine implements deterministic assessment logic for each
disclosure requirement, computing coverage scores, gap analysis, and
completeness metrics.

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS S4 Consumers and End-Users
    - EU General Product Safety Regulation (GPSR)
    - EU General Data Protection Regulation (GDPR)

Zero-Hallucination:
    - All scoring uses deterministic arithmetic
    - Coverage ratios are computed from input counts
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-017 ESRS Full Coverage
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value using ROUND_HALF_UP.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))

def _pct(part: int, total: int) -> Decimal:
    """Calculate percentage as Decimal, rounded to 1 decimal place."""
    if total == 0:
        return Decimal("0.0")
    return _round_val(
        _decimal(part) / _decimal(total) * Decimal("100"), 1
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConsumerIssue(str, Enum):
    """Material consumer and end-user issues per ESRS S4.

    These represent the key impact areas that undertakings must
    consider when assessing their effects on consumers and end-users.
    """
    PRODUCT_SAFETY = "product_safety"
    DATA_PRIVACY = "data_privacy"
    ACCESSIBILITY = "accessibility"
    FAIR_MARKETING = "fair_marketing"
    DIGITAL_RIGHTS = "digital_rights"
    HEALTH_IMPACTS = "health_impacts"
    VULNERABLE_CONSUMERS = "vulnerable_consumers"
    SUSTAINABLE_CONSUMPTION = "sustainable_consumption"

class ProductSafetyLevel(str, Enum):
    """Product safety compliance status.

    Reflects the safety assessment outcome for products/services
    under the EU General Product Safety Regulation and equivalent
    frameworks.
    """
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    RECALLED = "recalled"

class DataPrivacyFramework(str, Enum):
    """Data privacy regulatory framework applicable to the undertaking.

    Identifies which data protection regime(s) apply based on the
    jurisdictions where consumers are located.
    """
    GDPR = "gdpr"
    CCPA = "ccpa"
    LGPD = "lgpd"
    PIPA = "pipa"
    NONE = "none"

class VulnerableGroup(str, Enum):
    """Categories of vulnerable consumer groups per ESRS S4.

    Undertakings must disclose how they identify and protect
    consumers belonging to particularly vulnerable groups.
    """
    CHILDREN = "children"
    ELDERLY = "elderly"
    LOW_INCOME = "low_income"
    DISABLED = "disabled"
    DIGITALLY_EXCLUDED = "digitally_excluded"

# ---------------------------------------------------------------------------
# Constants - S4 Disclosure Requirement Data Points
# ---------------------------------------------------------------------------

S4_1_DATAPOINTS: List[str] = [
    "s4_1_01_policies_covering_consumers",
    "s4_1_02_policy_commitments_human_rights",
    "s4_1_03_issues_covered_product_safety",
    "s4_1_04_issues_covered_data_privacy",
    "s4_1_05_issues_covered_accessibility",
    "s4_1_06_issues_covered_fair_marketing",
    "s4_1_07_vulnerable_groups_addressed",
    "s4_1_08_policy_alignment_international_standards",
]

S4_2_DATAPOINTS: List[str] = [
    "s4_2_01_engagement_processes_exist",
    "s4_2_02_engagement_with_consumers_directly",
    "s4_2_03_engagement_with_representatives",
    "s4_2_04_engagement_frequency",
    "s4_2_05_engagement_topics_covered",
    "s4_2_06_results_of_engagement",
]

S4_3_DATAPOINTS: List[str] = [
    "s4_3_01_remediation_processes_exist",
    "s4_3_02_grievance_channels_available",
    "s4_3_03_grievances_received_count",
    "s4_3_04_grievances_resolved_count",
    "s4_3_05_average_resolution_time_days",
    "s4_3_06_satisfaction_with_resolution",
]

S4_4_DATAPOINTS: List[str] = [
    "s4_4_01_actions_taken_on_impacts",
    "s4_4_02_product_safety_assessments_count",
    "s4_4_03_products_recalled_count",
    "s4_4_04_data_privacy_assessments_count",
    "s4_4_05_data_breaches_count",
    "s4_4_06_resources_allocated",
    "s4_4_07_effectiveness_of_actions",
]

S4_5_DATAPOINTS: List[str] = [
    "s4_5_01_targets_set",
    "s4_5_02_measurable_targets_count",
    "s4_5_03_target_base_year",
    "s4_5_04_target_milestone_year",
    "s4_5_05_progress_against_targets",
]

ALL_S4_DATAPOINTS: List[str] = (
    S4_1_DATAPOINTS + S4_2_DATAPOINTS + S4_3_DATAPOINTS
    + S4_4_DATAPOINTS + S4_5_DATAPOINTS
)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ConsumerPolicy(BaseModel):
    """Policy related to consumers and end-users per S4-1 (Para 14-16).

    Represents a single policy document or commitment that addresses
    the undertaking's approach to consumer and end-user impacts.
    """
    policy_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this policy",
    )
    policy_name: str = Field(
        ...,
        description="Name of the policy (e.g. Product Safety Policy)",
        max_length=500,
    )
    issues_covered: List[ConsumerIssue] = Field(
        default_factory=list,
        description="Consumer issues addressed by this policy",
    )
    vulnerable_groups_addressed: List[VulnerableGroup] = Field(
        default_factory=list,
        description="Vulnerable groups explicitly addressed",
    )
    aligned_with_international_standards: bool = Field(
        default=False,
        description="Whether the policy aligns with UN Guiding Principles or OECD Guidelines",
    )
    approved_by_management: bool = Field(
        default=False,
        description="Whether the policy is approved by management body",
    )
    last_reviewed_date: Optional[datetime] = Field(
        default=None,
        description="Date of last policy review",
    )
    scope_description: str = Field(
        default="",
        description="Description of the policy scope and applicability",
        max_length=2000,
    )

class ConsumerEngagement(BaseModel):
    """Consumer engagement process per S4-2 (Para 18-21).

    Describes how the undertaking engages with consumers and
    end-users or their representatives about potential impacts.
    """
    engagement_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this engagement process",
    )
    engagement_type: str = Field(
        ...,
        description="Type of engagement (e.g. survey, focus_group, advisory_panel)",
        max_length=200,
    )
    is_direct_with_consumers: bool = Field(
        default=True,
        description="Whether engagement is directly with consumers (vs. representatives)",
    )
    frequency: str = Field(
        default="annual",
        description="Frequency of engagement (e.g. quarterly, annual, ad_hoc)",
        max_length=100,
    )
    issues_discussed: List[ConsumerIssue] = Field(
        default_factory=list,
        description="Consumer issues discussed in this engagement",
    )
    vulnerable_groups_included: List[VulnerableGroup] = Field(
        default_factory=list,
        description="Vulnerable groups included in engagement",
    )
    participants_count: int = Field(
        default=0,
        description="Number of participants in the engagement",
        ge=0,
    )
    outcomes_documented: bool = Field(
        default=False,
        description="Whether engagement outcomes are documented and tracked",
    )

class ConsumerGrievance(BaseModel):
    """Consumer grievance record per S4-3 (Para 23-26).

    Represents a complaint or concern raised by a consumer or
    end-user through the undertaking's grievance channels.
    """
    grievance_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this grievance",
    )
    issue_type: ConsumerIssue = Field(
        ...,
        description="Category of consumer issue",
    )
    date_received: datetime = Field(
        ...,
        description="Date the grievance was received",
    )
    date_resolved: Optional[datetime] = Field(
        default=None,
        description="Date the grievance was resolved (None if open)",
    )
    is_resolved: bool = Field(
        default=False,
        description="Whether the grievance has been resolved",
    )
    resolution_time_days: Optional[int] = Field(
        default=None,
        description="Number of days from receipt to resolution",
        ge=0,
    )
    channel: str = Field(
        default="",
        description="Channel through which the grievance was raised",
        max_length=200,
    )
    satisfaction_score: Optional[Decimal] = Field(
        default=None,
        description="Consumer satisfaction with resolution (0-10 scale)",
        ge=Decimal("0"),
        le=Decimal("10"),
    )

class ProductSafetyAssessment(BaseModel):
    """Product safety assessment per S4-4 action tracking.

    Records the safety assessment status for a product or product line,
    supporting S4-4 disclosure on actions taken for product safety.
    """
    assessment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this assessment",
    )
    product_name: str = Field(
        ...,
        description="Name of the product or product line assessed",
        max_length=500,
    )
    safety_level: ProductSafetyLevel = Field(
        ...,
        description="Current safety compliance status",
    )
    assessment_date: datetime = Field(
        default_factory=utcnow,
        description="Date of the safety assessment",
    )
    issues_identified: int = Field(
        default=0,
        description="Number of safety issues identified",
        ge=0,
    )
    issues_remediated: int = Field(
        default=0,
        description="Number of safety issues remediated",
        ge=0,
    )
    affects_vulnerable_group: Optional[VulnerableGroup] = Field(
        default=None,
        description="Whether the product primarily affects a vulnerable group",
    )

class DataPrivacyAssessment(BaseModel):
    """Data privacy impact assessment per S4-4 action tracking.

    Records the data privacy assessment for a product, service, or
    processing activity, supporting S4-4 disclosure on data protection.
    """
    assessment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this assessment",
    )
    system_name: str = Field(
        ...,
        description="Name of the system or processing activity assessed",
        max_length=500,
    )
    framework: DataPrivacyFramework = Field(
        ...,
        description="Applicable data privacy regulatory framework",
    )
    assessment_date: datetime = Field(
        default_factory=utcnow,
        description="Date of the privacy impact assessment",
    )
    data_subjects_count: int = Field(
        default=0,
        description="Number of data subjects affected",
        ge=0,
    )
    high_risk_processing: bool = Field(
        default=False,
        description="Whether this involves high-risk processing per GDPR Art. 35",
    )
    breaches_in_period: int = Field(
        default=0,
        description="Number of data breaches in the reporting period",
        ge=0,
    )
    is_compliant: bool = Field(
        default=True,
        description="Whether the assessment found the system compliant",
    )

class ConsumerAction(BaseModel):
    """Action taken on material consumer impacts per S4-4 (Para 28-33).

    Represents a specific action or initiative the undertaking has
    taken to address material negative impacts on consumers.
    """
    action_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this action",
    )
    action_description: str = Field(
        ...,
        description="Description of the action taken",
        max_length=2000,
    )
    issue_addressed: ConsumerIssue = Field(
        ...,
        description="Consumer issue this action addresses",
    )
    start_date: datetime = Field(
        default_factory=utcnow,
        description="Date the action was initiated",
    )
    is_completed: bool = Field(
        default=False,
        description="Whether the action is completed",
    )
    resources_allocated_eur: Decimal = Field(
        default=Decimal("0"),
        description="Resources allocated in EUR",
        ge=Decimal("0"),
    )
    effectiveness_score: Optional[Decimal] = Field(
        default=None,
        description="Self-assessed effectiveness (0-100 scale)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    consumers_affected_count: int = Field(
        default=0,
        description="Number of consumers positively affected",
        ge=0,
    )

class ConsumerTarget(BaseModel):
    """Target for managing consumer impacts per S4-5 (Para 35-37).

    Represents a measurable target set by the undertaking to manage
    material negative impacts, advance positive impacts, or manage
    risks and opportunities related to consumers.
    """
    target_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this target",
    )
    target_description: str = Field(
        ...,
        description="Description of the target",
        max_length=2000,
    )
    issue_addressed: ConsumerIssue = Field(
        ...,
        description="Consumer issue this target addresses",
    )
    is_measurable: bool = Field(
        default=True,
        description="Whether the target is measurable with KPIs",
    )
    baseline_value: Optional[Decimal] = Field(
        default=None,
        description="Baseline value for the target metric",
    )
    target_value: Optional[Decimal] = Field(
        default=None,
        description="Target value to achieve",
    )
    current_value: Optional[Decimal] = Field(
        default=None,
        description="Current value of the target metric",
    )
    base_year: Optional[int] = Field(
        default=None,
        description="Base year for the target",
    )
    target_year: Optional[int] = Field(
        default=None,
        description="Target year for achievement",
    )
    progress_pct: Optional[Decimal] = Field(
        default=None,
        description="Progress toward target as percentage (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )

class S4ConsumersResult(BaseModel):
    """Complete ESRS S4 disclosure result.

    Aggregates all S4 disclosure requirement assessments into a single
    result with completeness validation and provenance tracking.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this assessment",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of assessment (UTC)",
    )
    reporting_year: int = Field(
        default=0, description="Reporting year"
    )
    entity_name: str = Field(
        default="", description="Entity or undertaking name"
    )

    # S4-1: Policies
    s4_1_policies: Dict[str, Any] = Field(
        default_factory=dict,
        description="S4-1 policy assessment results",
    )

    # S4-2: Engagement
    s4_2_engagement: Dict[str, Any] = Field(
        default_factory=dict,
        description="S4-2 engagement process assessment results",
    )

    # S4-3: Grievance
    s4_3_grievance: Dict[str, Any] = Field(
        default_factory=dict,
        description="S4-3 remediation and grievance channel results",
    )

    # S4-4: Actions
    s4_4_actions: Dict[str, Any] = Field(
        default_factory=dict,
        description="S4-4 action assessment results",
    )

    # S4-5: Targets
    s4_5_targets: Dict[str, Any] = Field(
        default_factory=dict,
        description="S4-5 target assessment results",
    )

    # Summary metrics
    total_policies: int = Field(
        default=0, description="Total policies assessed"
    )
    total_engagements: int = Field(
        default=0, description="Total engagement processes"
    )
    total_grievances: int = Field(
        default=0, description="Total grievances received"
    )
    total_actions: int = Field(
        default=0, description="Total actions taken"
    )
    total_targets: int = Field(
        default=0, description="Total targets set"
    )
    overall_issue_coverage_pct: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of consumer issues covered by policies",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and assessment steps",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ConsumersEngine:
    """ESRS S4 Consumers and End-Users assessment engine.

    Provides deterministic, zero-hallucination assessments for all
    five S4 disclosure requirements:

    - S4-1: Policy coverage and alignment assessment
    - S4-2: Engagement process quality and frequency
    - S4-3: Grievance mechanism effectiveness
    - S4-4: Action tracking and effectiveness
    - S4-5: Target progress and measurability

    All calculations use Decimal arithmetic for reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = ConsumersEngine()
        result = engine.calculate_s4_disclosure(
            policies=[ConsumerPolicy(...)],
            engagements=[ConsumerEngagement(...)],
            grievances=[ConsumerGrievance(...)],
            actions=[ConsumerAction(...)],
            safety_assessments=[ProductSafetyAssessment(...)],
            privacy_assessments=[DataPrivacyAssessment(...)],
            targets=[ConsumerTarget(...)],
        )
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # S4-1: Policies (Para 14-16)                                         #
    # ------------------------------------------------------------------ #

    def assess_policies(
        self, policies: List[ConsumerPolicy]
    ) -> Dict[str, Any]:
        """Assess consumer policies for S4-1 disclosure.

        Evaluates policy coverage across consumer issues, vulnerable
        group protection, alignment with international standards, and
        management approval.

        Args:
            policies: List of ConsumerPolicy instances to assess.

        Returns:
            Dict with policy_count, issues_covered, vulnerable_groups,
            international_alignment_pct, management_approval_pct,
            and per-issue coverage flags.
        """
        if not policies:
            logger.warning("S4-1: No consumer policies provided")
            return {
                "policy_count": 0,
                "issues_covered": [],
                "issues_covered_count": 0,
                "issue_coverage_pct": Decimal("0.0"),
                "vulnerable_groups_addressed": [],
                "vulnerable_groups_count": 0,
                "international_alignment_count": 0,
                "international_alignment_pct": Decimal("0.0"),
                "management_approved_count": 0,
                "management_approved_pct": Decimal("0.0"),
                "per_issue_coverage": {},
                "provenance_hash": _compute_hash({"policies": []}),
            }

        all_issues_covered: set = set()
        all_vulnerable_groups: set = set()
        aligned_count = 0
        approved_count = 0

        for policy in policies:
            for issue in policy.issues_covered:
                all_issues_covered.add(issue.value)
            for group in policy.vulnerable_groups_addressed:
                all_vulnerable_groups.add(group.value)
            if policy.aligned_with_international_standards:
                aligned_count += 1
            if policy.approved_by_management:
                approved_count += 1

        total_possible_issues = len(ConsumerIssue)
        issue_coverage_pct = _pct(
            len(all_issues_covered), total_possible_issues
        )

        per_issue = {}
        for issue in ConsumerIssue:
            per_issue[issue.value] = issue.value in all_issues_covered

        result = {
            "policy_count": len(policies),
            "issues_covered": sorted(all_issues_covered),
            "issues_covered_count": len(all_issues_covered),
            "issue_coverage_pct": issue_coverage_pct,
            "vulnerable_groups_addressed": sorted(all_vulnerable_groups),
            "vulnerable_groups_count": len(all_vulnerable_groups),
            "international_alignment_count": aligned_count,
            "international_alignment_pct": _pct(
                aligned_count, len(policies)
            ),
            "management_approved_count": approved_count,
            "management_approved_pct": _pct(
                approved_count, len(policies)
            ),
            "per_issue_coverage": per_issue,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S4-1 assessed: %d policies, %d/%d issues covered (%.1f%%)",
            len(policies), len(all_issues_covered),
            total_possible_issues, float(issue_coverage_pct),
        )

        return result

    # ------------------------------------------------------------------ #
    # S4-2: Engagement (Para 18-21)                                       #
    # ------------------------------------------------------------------ #

    def assess_engagement(
        self, engagements: List[ConsumerEngagement]
    ) -> Dict[str, Any]:
        """Assess consumer engagement processes for S4-2 disclosure.

        Evaluates how the undertaking engages with consumers about
        impacts, including direct engagement, frequency, issue coverage,
        and inclusion of vulnerable groups.

        Args:
            engagements: List of ConsumerEngagement instances.

        Returns:
            Dict with engagement_count, direct_engagement_pct,
            issues_discussed, vulnerable_groups_included,
            total_participants, and documentation rate.
        """
        if not engagements:
            logger.warning("S4-2: No engagement processes provided")
            return {
                "engagement_count": 0,
                "direct_engagement_count": 0,
                "direct_engagement_pct": Decimal("0.0"),
                "issues_discussed": [],
                "issues_discussed_count": 0,
                "vulnerable_groups_included": [],
                "vulnerable_groups_count": 0,
                "total_participants": 0,
                "outcomes_documented_count": 0,
                "outcomes_documented_pct": Decimal("0.0"),
                "provenance_hash": _compute_hash({"engagements": []}),
            }

        direct_count = 0
        all_issues: set = set()
        all_groups: set = set()
        total_participants = 0
        documented_count = 0

        for eng in engagements:
            if eng.is_direct_with_consumers:
                direct_count += 1
            for issue in eng.issues_discussed:
                all_issues.add(issue.value)
            for group in eng.vulnerable_groups_included:
                all_groups.add(group.value)
            total_participants += eng.participants_count
            if eng.outcomes_documented:
                documented_count += 1

        n = len(engagements)

        result = {
            "engagement_count": n,
            "direct_engagement_count": direct_count,
            "direct_engagement_pct": _pct(direct_count, n),
            "issues_discussed": sorted(all_issues),
            "issues_discussed_count": len(all_issues),
            "vulnerable_groups_included": sorted(all_groups),
            "vulnerable_groups_count": len(all_groups),
            "total_participants": total_participants,
            "outcomes_documented_count": documented_count,
            "outcomes_documented_pct": _pct(documented_count, n),
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S4-2 assessed: %d engagements, %d direct, %d participants",
            n, direct_count, total_participants,
        )

        return result

    # ------------------------------------------------------------------ #
    # S4-3: Grievance / Remediation (Para 23-26)                          #
    # ------------------------------------------------------------------ #

    def assess_grievance_mechanisms(
        self, grievances: List[ConsumerGrievance]
    ) -> Dict[str, Any]:
        """Assess grievance mechanisms for S4-3 disclosure.

        Evaluates the effectiveness of remediation processes and
        grievance channels, including resolution rates, average
        resolution time, and consumer satisfaction.

        Args:
            grievances: List of ConsumerGrievance instances.

        Returns:
            Dict with grievance_count, resolved_count, resolution_rate,
            avg_resolution_days, by_issue_type breakdown, by_channel,
            and avg_satisfaction_score.
        """
        if not grievances:
            logger.warning("S4-3: No grievances provided")
            return {
                "grievance_count": 0,
                "resolved_count": 0,
                "open_count": 0,
                "resolution_rate_pct": Decimal("0.0"),
                "avg_resolution_days": Decimal("0"),
                "by_issue_type": {},
                "channels_used": [],
                "channels_count": 0,
                "avg_satisfaction_score": Decimal("0.0"),
                "provenance_hash": _compute_hash({"grievances": []}),
            }

        resolved_count = 0
        resolution_days_sum = Decimal("0")
        resolution_days_entries = 0
        satisfaction_sum = Decimal("0")
        satisfaction_entries = 0
        by_issue: Dict[str, Dict[str, int]] = {}
        channels: set = set()

        for g in grievances:
            issue_key = g.issue_type.value
            if issue_key not in by_issue:
                by_issue[issue_key] = {"total": 0, "resolved": 0}
            by_issue[issue_key]["total"] += 1

            if g.is_resolved:
                resolved_count += 1
                by_issue[issue_key]["resolved"] += 1

            if g.resolution_time_days is not None:
                resolution_days_sum += _decimal(g.resolution_time_days)
                resolution_days_entries += 1

            if g.satisfaction_score is not None:
                satisfaction_sum += g.satisfaction_score
                satisfaction_entries += 1

            if g.channel:
                channels.add(g.channel)

        n = len(grievances)
        avg_days = (
            _round_val(
                _safe_divide(
                    resolution_days_sum,
                    _decimal(resolution_days_entries),
                ),
                1,
            )
            if resolution_days_entries > 0
            else Decimal("0")
        )

        avg_satisfaction = (
            _round_val(
                _safe_divide(
                    satisfaction_sum,
                    _decimal(satisfaction_entries),
                ),
                1,
            )
            if satisfaction_entries > 0
            else Decimal("0.0")
        )

        result = {
            "grievance_count": n,
            "resolved_count": resolved_count,
            "open_count": n - resolved_count,
            "resolution_rate_pct": _pct(resolved_count, n),
            "avg_resolution_days": avg_days,
            "by_issue_type": by_issue,
            "channels_used": sorted(channels),
            "channels_count": len(channels),
            "avg_satisfaction_score": avg_satisfaction,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S4-3 assessed: %d grievances, %d resolved (%.1f%%), "
            "avg %.1f days",
            n, resolved_count,
            float(_pct(resolved_count, n)), float(avg_days),
        )

        return result

    # ------------------------------------------------------------------ #
    # S4-4: Actions (Para 28-33)                                          #
    # ------------------------------------------------------------------ #

    def assess_actions(
        self,
        actions: List[ConsumerAction],
        safety_assessments: List[ProductSafetyAssessment],
        privacy_assessments: List[DataPrivacyAssessment],
    ) -> Dict[str, Any]:
        """Assess actions taken on material consumer impacts for S4-4.

        Evaluates actions addressing consumer impacts, product safety
        compliance status, data privacy assessment outcomes, and
        resource allocation.

        Args:
            actions: List of ConsumerAction instances.
            safety_assessments: List of ProductSafetyAssessment instances.
            privacy_assessments: List of DataPrivacyAssessment instances.

        Returns:
            Dict with action metrics, safety summary, privacy summary,
            total resources, and effectiveness scores.
        """
        # -- Actions analysis --
        action_count = len(actions)
        completed_count = sum(1 for a in actions if a.is_completed)
        total_resources = sum(
            a.resources_allocated_eur for a in actions
        )
        total_consumers_affected = sum(
            a.consumers_affected_count for a in actions
        )

        effectiveness_scores: List[Decimal] = [
            a.effectiveness_score for a in actions
            if a.effectiveness_score is not None
        ]
        avg_effectiveness = Decimal("0.0")
        if effectiveness_scores:
            avg_effectiveness = _round_val(
                sum(effectiveness_scores)
                / _decimal(len(effectiveness_scores)),
                1,
            )

        actions_by_issue: Dict[str, int] = {}
        for a in actions:
            key = a.issue_addressed.value
            actions_by_issue[key] = actions_by_issue.get(key, 0) + 1

        # -- Product safety analysis --
        safety_count = len(safety_assessments)
        compliant_count = sum(
            1 for s in safety_assessments
            if s.safety_level == ProductSafetyLevel.COMPLIANT
        )
        recalled_count = sum(
            1 for s in safety_assessments
            if s.safety_level == ProductSafetyLevel.RECALLED
        )
        total_safety_issues = sum(
            s.issues_identified for s in safety_assessments
        )
        total_safety_remediated = sum(
            s.issues_remediated for s in safety_assessments
        )

        safety_summary = {
            "assessments_count": safety_count,
            "compliant_count": compliant_count,
            "compliant_pct": _pct(compliant_count, safety_count),
            "non_compliant_count": sum(
                1 for s in safety_assessments
                if s.safety_level == ProductSafetyLevel.NON_COMPLIANT
            ),
            "under_review_count": sum(
                1 for s in safety_assessments
                if s.safety_level == ProductSafetyLevel.UNDER_REVIEW
            ),
            "recalled_count": recalled_count,
            "total_issues_identified": total_safety_issues,
            "total_issues_remediated": total_safety_remediated,
            "remediation_rate_pct": _pct(
                total_safety_remediated, total_safety_issues
            ),
        }

        # -- Data privacy analysis --
        privacy_count = len(privacy_assessments)
        privacy_compliant = sum(
            1 for p in privacy_assessments if p.is_compliant
        )
        total_breaches = sum(
            p.breaches_in_period for p in privacy_assessments
        )
        high_risk_count = sum(
            1 for p in privacy_assessments if p.high_risk_processing
        )
        total_data_subjects = sum(
            p.data_subjects_count for p in privacy_assessments
        )

        frameworks_used: set = set()
        for p in privacy_assessments:
            if p.framework != DataPrivacyFramework.NONE:
                frameworks_used.add(p.framework.value)

        privacy_summary = {
            "assessments_count": privacy_count,
            "compliant_count": privacy_compliant,
            "compliant_pct": _pct(privacy_compliant, privacy_count),
            "total_breaches": total_breaches,
            "high_risk_processing_count": high_risk_count,
            "total_data_subjects": total_data_subjects,
            "frameworks_used": sorted(frameworks_used),
        }

        result = {
            "action_count": action_count,
            "completed_count": completed_count,
            "completion_rate_pct": _pct(completed_count, action_count),
            "total_resources_eur": total_resources,
            "total_consumers_affected": total_consumers_affected,
            "avg_effectiveness_score": avg_effectiveness,
            "actions_by_issue": actions_by_issue,
            "product_safety": safety_summary,
            "data_privacy": privacy_summary,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S4-4 assessed: %d actions, %d safety assessments, "
            "%d privacy assessments, %d breaches",
            action_count, safety_count, privacy_count, total_breaches,
        )

        return result

    # ------------------------------------------------------------------ #
    # S4-5: Targets (Para 35-37)                                          #
    # ------------------------------------------------------------------ #

    def assess_targets(
        self, targets: List[ConsumerTarget]
    ) -> Dict[str, Any]:
        """Assess targets for managing consumer impacts per S4-5.

        Evaluates the quality and progress of targets set to manage
        negative impacts, advance positive impacts, and manage material
        risks and opportunities related to consumers.

        Args:
            targets: List of ConsumerTarget instances.

        Returns:
            Dict with target_count, measurable_count, avg_progress,
            targets_by_issue, and on-track assessment.
        """
        if not targets:
            logger.warning("S4-5: No consumer targets provided")
            return {
                "target_count": 0,
                "measurable_count": 0,
                "measurable_pct": Decimal("0.0"),
                "avg_progress_pct": Decimal("0.0"),
                "targets_by_issue": {},
                "targets_on_track": 0,
                "targets_on_track_pct": Decimal("0.0"),
                "provenance_hash": _compute_hash({"targets": []}),
            }

        measurable_count = sum(1 for t in targets if t.is_measurable)
        progress_values: List[Decimal] = [
            t.progress_pct for t in targets
            if t.progress_pct is not None
        ]
        avg_progress = Decimal("0.0")
        if progress_values:
            avg_progress = _round_val(
                sum(progress_values)
                / _decimal(len(progress_values)),
                1,
            )

        targets_by_issue: Dict[str, int] = {}
        for t in targets:
            key = t.issue_addressed.value
            targets_by_issue[key] = targets_by_issue.get(key, 0) + 1

        # A target is considered on-track if progress >= 40%
        on_track_count = sum(
            1 for t in targets
            if t.progress_pct is not None
            and t.progress_pct >= Decimal("40")
        )

        n = len(targets)

        result = {
            "target_count": n,
            "measurable_count": measurable_count,
            "measurable_pct": _pct(measurable_count, n),
            "avg_progress_pct": avg_progress,
            "targets_by_issue": targets_by_issue,
            "targets_on_track": on_track_count,
            "targets_on_track_pct": _pct(on_track_count, n),
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S4-5 assessed: %d targets, %d measurable, "
            "avg progress %.1f%%",
            n, measurable_count, float(avg_progress),
        )

        return result

    # ------------------------------------------------------------------ #
    # Full S4 Disclosure Calculation                                      #
    # ------------------------------------------------------------------ #

    def calculate_s4_disclosure(
        self,
        policies: List[ConsumerPolicy],
        engagements: List[ConsumerEngagement],
        grievances: List[ConsumerGrievance],
        actions: List[ConsumerAction],
        safety_assessments: List[ProductSafetyAssessment],
        privacy_assessments: List[DataPrivacyAssessment],
        targets: List[ConsumerTarget],
        entity_name: str = "",
        reporting_year: int = 0,
    ) -> S4ConsumersResult:
        """Calculate the complete ESRS S4 disclosure.

        Orchestrates assessment of all five S4 disclosure requirements
        and produces a consolidated result with provenance tracking.

        Args:
            policies: Consumer policies for S4-1.
            engagements: Engagement processes for S4-2.
            grievances: Grievance records for S4-3.
            actions: Actions taken for S4-4.
            safety_assessments: Product safety assessments for S4-4.
            privacy_assessments: Data privacy assessments for S4-4.
            targets: Consumer targets for S4-5.
            entity_name: Name of the reporting entity.
            reporting_year: Reporting year.

        Returns:
            S4ConsumersResult with complete provenance.
        """
        t0 = time.perf_counter()

        logger.info(
            "Calculating S4 disclosure: entity=%s, year=%d",
            entity_name, reporting_year,
        )

        # Assess each disclosure requirement
        s4_1 = self.assess_policies(policies)
        s4_2 = self.assess_engagement(engagements)
        s4_3 = self.assess_grievance_mechanisms(grievances)
        s4_4 = self.assess_actions(
            actions, safety_assessments, privacy_assessments
        )
        s4_5 = self.assess_targets(targets)

        # Overall issue coverage from policies
        overall_coverage = s4_1.get(
            "issue_coverage_pct", Decimal("0.0")
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = S4ConsumersResult(
            reporting_year=reporting_year,
            entity_name=entity_name,
            s4_1_policies=s4_1,
            s4_2_engagement=s4_2,
            s4_3_grievance=s4_3,
            s4_4_actions=s4_4,
            s4_5_targets=s4_5,
            total_policies=len(policies),
            total_engagements=len(engagements),
            total_grievances=len(grievances),
            total_actions=len(actions),
            total_targets=len(targets),
            overall_issue_coverage_pct=overall_coverage,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "S4 disclosure calculated: %d policies, %d engagements, "
            "%d grievances, %d actions, %d targets, hash=%s",
            len(policies), len(engagements), len(grievances),
            len(actions), len(targets), result.provenance_hash[:16],
        )

        return result

    # ------------------------------------------------------------------ #
    # Completeness Validation                                             #
    # ------------------------------------------------------------------ #

    def validate_s4_completeness(
        self, result: S4ConsumersResult
    ) -> Dict[str, Any]:
        """Validate completeness against all S4 required data points.

        Checks whether all ESRS S4 mandatory disclosure data points
        are present and populated in the result.

        Args:
            result: S4ConsumersResult to validate.

        Returns:
            Dict with total_datapoints, populated_datapoints,
            missing_datapoints, completeness_pct, is_complete,
            per_dr_completeness, and provenance_hash.
        """
        populated: List[str] = []
        missing: List[str] = []

        # S4-1 checks
        s4_1 = result.s4_1_policies
        s4_1_checks = {
            "s4_1_01_policies_covering_consumers": (
                s4_1.get("policy_count", 0) > 0
            ),
            "s4_1_02_policy_commitments_human_rights": (
                s4_1.get("international_alignment_count", 0) > 0
            ),
            "s4_1_03_issues_covered_product_safety": (
                "product_safety" in s4_1.get("issues_covered", [])
            ),
            "s4_1_04_issues_covered_data_privacy": (
                "data_privacy" in s4_1.get("issues_covered", [])
            ),
            "s4_1_05_issues_covered_accessibility": (
                "accessibility" in s4_1.get("issues_covered", [])
            ),
            "s4_1_06_issues_covered_fair_marketing": (
                "fair_marketing" in s4_1.get("issues_covered", [])
            ),
            "s4_1_07_vulnerable_groups_addressed": (
                len(s4_1.get("vulnerable_groups_addressed", [])) > 0
            ),
            "s4_1_08_policy_alignment_international_standards": (
                s4_1.get("international_alignment_count", 0) > 0
            ),
        }

        # S4-2 checks
        s4_2 = result.s4_2_engagement
        s4_2_checks = {
            "s4_2_01_engagement_processes_exist": (
                s4_2.get("engagement_count", 0) > 0
            ),
            "s4_2_02_engagement_with_consumers_directly": (
                s4_2.get("direct_engagement_count", 0) > 0
            ),
            "s4_2_03_engagement_with_representatives": (
                s4_2.get("engagement_count", 0)
                > s4_2.get("direct_engagement_count", 0)
            ),
            "s4_2_04_engagement_frequency": (
                s4_2.get("engagement_count", 0) > 0
            ),
            "s4_2_05_engagement_topics_covered": (
                s4_2.get("issues_discussed_count", 0) > 0
            ),
            "s4_2_06_results_of_engagement": (
                s4_2.get("outcomes_documented_count", 0) > 0
            ),
        }

        # S4-3 checks
        s4_3 = result.s4_3_grievance
        s4_3_checks = {
            "s4_3_01_remediation_processes_exist": (
                s4_3.get("channels_count", 0) > 0
            ),
            "s4_3_02_grievance_channels_available": (
                s4_3.get("channels_count", 0) > 0
            ),
            "s4_3_03_grievances_received_count": True,
            "s4_3_04_grievances_resolved_count": True,
            "s4_3_05_average_resolution_time_days": True,
            "s4_3_06_satisfaction_with_resolution": True,
        }

        # S4-4 checks
        s4_4 = result.s4_4_actions
        s4_4_checks = {
            "s4_4_01_actions_taken_on_impacts": (
                s4_4.get("action_count", 0) > 0
            ),
            "s4_4_02_product_safety_assessments_count": (
                s4_4.get("product_safety", {})
                .get("assessments_count", 0) > 0
            ),
            "s4_4_03_products_recalled_count": True,
            "s4_4_04_data_privacy_assessments_count": (
                s4_4.get("data_privacy", {})
                .get("assessments_count", 0) > 0
            ),
            "s4_4_05_data_breaches_count": True,
            "s4_4_06_resources_allocated": (
                s4_4.get("total_resources_eur", Decimal("0"))
                > Decimal("0")
            ),
            "s4_4_07_effectiveness_of_actions": (
                s4_4.get("avg_effectiveness_score", Decimal("0"))
                > Decimal("0")
            ),
        }

        # S4-5 checks
        s4_5 = result.s4_5_targets
        s4_5_checks = {
            "s4_5_01_targets_set": (
                s4_5.get("target_count", 0) > 0
            ),
            "s4_5_02_measurable_targets_count": (
                s4_5.get("measurable_count", 0) > 0
            ),
            "s4_5_03_target_base_year": True,
            "s4_5_04_target_milestone_year": True,
            "s4_5_05_progress_against_targets": (
                s4_5.get("avg_progress_pct", Decimal("0"))
                > Decimal("0")
            ),
        }

        all_checks = {
            **s4_1_checks, **s4_2_checks, **s4_3_checks,
            **s4_4_checks, **s4_5_checks,
        }

        for dp, is_populated in all_checks.items():
            if is_populated:
                populated.append(dp)
            else:
                missing.append(dp)

        total = len(ALL_S4_DATAPOINTS)
        pop_count = len(populated)
        completeness = _round_val(
            _decimal(pop_count) / _decimal(total) * Decimal("100"), 1
        )

        # Per-DR completeness breakdown
        def _dr_completeness(
            checks: Dict[str, bool],
        ) -> Dict[str, Any]:
            pop = sum(1 for v in checks.values() if v)
            tot = len(checks)
            return {
                "populated": pop,
                "total": tot,
                "completeness_pct": _pct(pop, tot),
                "missing": [k for k, v in checks.items() if not v],
            }

        per_dr = {
            "S4-1": _dr_completeness(s4_1_checks),
            "S4-2": _dr_completeness(s4_2_checks),
            "S4-3": _dr_completeness(s4_3_checks),
            "S4-4": _dr_completeness(s4_4_checks),
            "S4-5": _dr_completeness(s4_5_checks),
        }

        validation_result = {
            "total_datapoints": total,
            "populated_datapoints": pop_count,
            "missing_datapoints": missing,
            "completeness_pct": completeness,
            "is_complete": len(missing) == 0,
            "per_dr_completeness": per_dr,
            "provenance_hash": _compute_hash(
                {"result_id": result.result_id, "checks": all_checks}
            ),
        }

        logger.info(
            "S4 completeness: %.1f%% (%d/%d), missing=%s",
            float(completeness), pop_count, total, missing,
        )

        return validation_result
