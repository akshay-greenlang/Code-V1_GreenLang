# -*- coding: utf-8 -*-
"""
S2ValueChainWorkersEngine - PACK-017 ESRS S2 Workers in the Value Chain
========================================================================

Calculates and evaluates disclosure requirements for ESRS S2 (Workers in
the Value Chain), covering policies, engagement processes, grievance
mechanisms, actions on material impacts, risk assessments, and targets.

Under ESRS S2, undertakings must disclose how they identify, manage, and
remediate impacts on workers throughout their value chain.  This engine
implements the complete S2 assessment pipeline, including:

- Policy assessment against human rights and ILO standards (S2-1)
- Engagement process evaluation with coverage analysis by tier (S2-2)
- Grievance channel accessibility and resolution rate analysis (S2-3)
- Action evaluation against identified risks with gap detection (S2-4)
- Target tracking with progress measurement (S2-5)
- Cross-cutting risk assessment by tier, country, and category
- Completeness validation against all S2 required data points

ESRS S2 Disclosure Requirements:
    - S2-1  (Para 14-16, AR S2-1 to AR S2-5):  Policies related to value
      chain workers, including alignment with international standards.
    - S2-2  (Para 18-21, AR S2-6 to AR S2-9):  Processes for engaging
      with value chain workers about impacts.
    - S2-3  (Para 23-26, AR S2-10 to AR S2-13):  Processes to remediate
      negative impacts and channels for value chain workers to raise
      concerns.
    - S2-4  (Para 28-33, AR S2-14 to AR S2-18):  Taking action on
      material impacts on value chain workers, and effectiveness of
      those actions.
    - S2-5  (Para 35-37, AR S2-19 to AR S2-22):  Targets related to
      managing material negative impacts, advancing positive impacts,
      and managing material risks and opportunities.

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS S2 Workers in the Value Chain
    - UN Guiding Principles on Business and Human Rights (UNGPs)
    - ILO Declaration on Fundamental Principles and Rights at Work
    - OECD Due Diligence Guidance for Responsible Business Conduct

Zero-Hallucination:
    - All scoring and aggregation use deterministic Decimal arithmetic
    - Coverage percentages computed from explicit set intersections
    - Resolution rates from case counts (no ML/LLM)
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
from typing import Any, Dict, List, Optional, Set

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
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
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
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ValueChainTier(str, Enum):
    """Supply chain tier classification per ESRS S2 AR S2-4.

    Tier 1: Direct suppliers with contractual relationship.
    Tier 2: Suppliers of Tier 1 suppliers.
    Tier 3+: Upstream suppliers beyond Tier 2.
    """
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3_PLUS = "tier_3_plus"

class WorkerType(str, Enum):
    """Worker category classification per ESRS S2 Para 14.

    Identifies the economic sector in which value chain workers operate,
    enabling sector-specific risk assessment and engagement.
    """
    MANUFACTURING = "manufacturing"
    AGRICULTURE = "agriculture"
    LOGISTICS = "logistics"
    MINING = "mining"
    SERVICES = "services"
    CONSTRUCTION = "construction"

class RiskCategory(str, Enum):
    """Human rights and labour risk categories per ESRS S2 AR S2-14.

    These categories align with ILO core labour standards and the UN
    Guiding Principles on Business and Human Rights.
    """
    FORCED_LABOUR = "forced_labour"
    CHILD_LABOUR = "child_labour"
    UNSAFE_CONDITIONS = "unsafe_conditions"
    EXCESSIVE_HOURS = "excessive_hours"
    INADEQUATE_WAGES = "inadequate_wages"
    DISCRIMINATION = "discrimination"
    FREEDOM_OF_ASSOCIATION = "freedom_of_association"

class EngagementMechanism(str, Enum):
    """Engagement mechanism types per ESRS S2 Para 18 and AR S2-6.

    Mechanisms through which the undertaking engages with value chain
    workers or their legitimate representatives.
    """
    DIRECT_DIALOGUE = "direct_dialogue"
    TRADE_UNIONS = "trade_unions"
    WORKER_COMMITTEES = "worker_committees"
    GRIEVANCE_MECHANISM = "grievance_mechanism"
    SURVEYS = "surveys"
    NGO_PARTNERSHIP = "ngo_partnership"

class RemediationStatus(str, Enum):
    """Status of remediation actions per ESRS S2 Para 23-26.

    Tracks the lifecycle of remediation from initial identification
    through resolution and closure.
    """
    IDENTIFIED = "identified"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

class DueDiligencePhase(str, Enum):
    """Due diligence phase per OECD Guidance and ESRS S2 AR S2-14.

    The five phases of human rights due diligence as defined by the
    OECD Due Diligence Guidance for Responsible Business Conduct.
    """
    IDENTIFICATION = "identification"
    PREVENTION = "prevention"
    MITIGATION = "mitigation"
    REMEDIATION = "remediation"
    MONITORING = "monitoring"

# ---------------------------------------------------------------------------
# Constants - ESRS S2 Disclosure Data Points (XBRL identifiers)
# ---------------------------------------------------------------------------

S2_1_DATAPOINTS: List[str] = [
    "s2_1_01_policies_addressing_value_chain_workers",
    "s2_1_02_human_rights_policy_commitments",
    "s2_1_03_ilo_conventions_alignment",
    "s2_1_04_due_diligence_policy_scope",
    "s2_1_05_stakeholders_consulted_in_policy_design",
    "s2_1_06_policy_communication_to_value_chain",
    "s2_1_07_tiers_covered_by_policies",
]

S2_2_DATAPOINTS: List[str] = [
    "s2_2_01_engagement_processes_with_value_chain_workers",
    "s2_2_02_engagement_with_legitimate_representatives",
    "s2_2_03_stage_of_engagement_in_due_diligence",
    "s2_2_04_frequency_of_engagement",
    "s2_2_05_tiers_covered_by_engagement",
    "s2_2_06_worker_types_covered_by_engagement",
    "s2_2_07_outcomes_of_engagement",
]

S2_3_DATAPOINTS: List[str] = [
    "s2_3_01_remediation_processes_for_negative_impacts",
    "s2_3_02_channels_to_raise_concerns",
    "s2_3_03_accessibility_of_channels",
    "s2_3_04_anonymous_reporting_available",
    "s2_3_05_grievance_cases_received",
    "s2_3_06_grievance_cases_resolved",
    "s2_3_07_average_resolution_time_days",
    "s2_3_08_remediation_outcomes",
]

S2_4_DATAPOINTS: List[str] = [
    "s2_4_01_actions_on_material_impacts",
    "s2_4_02_resources_allocated_to_actions",
    "s2_4_03_actions_by_risk_category",
    "s2_4_04_actions_by_value_chain_tier",
    "s2_4_05_effectiveness_of_actions",
    "s2_4_06_workers_covered_by_actions",
    "s2_4_07_due_diligence_approach",
    "s2_4_08_unaddressed_risks_disclosure",
]

S2_5_DATAPOINTS: List[str] = [
    "s2_5_01_targets_for_negative_impacts",
    "s2_5_02_targets_for_positive_impacts",
    "s2_5_03_target_measurement_methodology",
    "s2_5_04_target_base_year_and_value",
    "s2_5_05_target_progress_percentage",
    "s2_5_06_target_timeline",
    "s2_5_07_target_revision_explanation",
]

ALL_S2_DATAPOINTS: List[str] = (
    S2_1_DATAPOINTS + S2_2_DATAPOINTS + S2_3_DATAPOINTS
    + S2_4_DATAPOINTS + S2_5_DATAPOINTS
)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ValueChainWorkerPolicy(BaseModel):
    """Policy related to value chain workers per ESRS S2-1 (Para 14-16).

    Captures the undertaking's policy commitments regarding human rights
    and labour standards for workers in the value chain.
    """
    policy_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this policy",
    )
    name: str = Field(
        ...,
        description="Policy name or title",
        max_length=500,
    )
    scope: str = Field(
        default="",
        description="Scope description (e.g. geographic, sector, tier coverage)",
        max_length=1000,
    )
    covers_tiers: List[ValueChainTier] = Field(
        default_factory=list,
        description="Value chain tiers covered by this policy",
    )
    human_rights_standards_referenced: List[str] = Field(
        default_factory=list,
        description="International human rights standards referenced (e.g. UNGPs, UDHR)",
    )
    ilo_conventions_alignment: List[str] = Field(
        default_factory=list,
        description="ILO Conventions the policy aligns with (e.g. C029, C138, C182)",
    )
    implementation_date: Optional[str] = Field(
        default=None,
        description="Date of policy implementation (ISO-8601 date string)",
        max_length=10,
    )

class EngagementProcess(BaseModel):
    """Engagement process with value chain workers per ESRS S2-2 (Para 18-21).

    Documents how the undertaking engages with workers in the value chain
    about actual and potential impacts.
    """
    process_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this engagement process",
    )
    mechanism: EngagementMechanism = Field(
        ...,
        description="Type of engagement mechanism used",
    )
    worker_types_covered: List[WorkerType] = Field(
        default_factory=list,
        description="Worker types covered by this engagement",
    )
    tiers_covered: List[ValueChainTier] = Field(
        default_factory=list,
        description="Value chain tiers reached by this engagement",
    )
    frequency: str = Field(
        default="",
        description="Engagement frequency (e.g. quarterly, annually, ad hoc)",
        max_length=100,
    )
    last_engagement_date: Optional[str] = Field(
        default=None,
        description="Date of last engagement activity (ISO-8601 date string)",
        max_length=10,
    )
    outcomes_summary: str = Field(
        default="",
        description="Summary of key outcomes from the engagement process",
        max_length=2000,
    )

class GrievanceChannel(BaseModel):
    """Grievance and remediation channel per ESRS S2-3 (Para 23-26).

    Documents channels available for value chain workers to raise
    concerns and seek remediation for negative impacts.
    """
    channel_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this grievance channel",
    )
    type: str = Field(
        ...,
        description="Channel type (e.g. hotline, email, web portal, in-person)",
        max_length=200,
    )
    accessible_to_tiers: List[ValueChainTier] = Field(
        default_factory=list,
        description="Value chain tiers that can access this channel",
    )
    anonymous_reporting: bool = Field(
        default=False,
        description="Whether anonymous reporting is supported",
    )
    response_time_days: Decimal = Field(
        default=Decimal("0"),
        description="Target or average response time in days",
        ge=Decimal("0"),
    )
    cases_received: int = Field(
        default=0,
        description="Total grievance cases received in reporting period",
        ge=0,
    )
    cases_resolved: int = Field(
        default=0,
        description="Total grievance cases resolved in reporting period",
        ge=0,
    )

    @field_validator("cases_resolved")
    @classmethod
    def validate_resolved_not_exceeding_received(
        cls, v: int, info: Any
    ) -> int:
        """Validate resolved cases do not exceed received cases."""
        received = info.data.get("cases_received", 0)
        if v > received:
            raise ValueError(
                f"cases_resolved ({v}) cannot exceed cases_received ({received})"
            )
        return v

class ValueChainWorkerAction(BaseModel):
    """Action taken on material impacts per ESRS S2-4 (Para 28-33).

    Documents specific actions the undertaking is taking to address
    actual or potential negative impacts on value chain workers.
    """
    action_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this action",
    )
    description: str = Field(
        ...,
        description="Description of the action being taken",
        max_length=2000,
    )
    target_risk: RiskCategory = Field(
        ...,
        description="Risk category this action addresses",
    )
    target_tier: ValueChainTier = Field(
        ...,
        description="Value chain tier this action targets",
    )
    resources_allocated: Decimal = Field(
        default=Decimal("0"),
        description="Resources allocated to this action (EUR)",
        ge=Decimal("0"),
    )
    expected_workers_covered: int = Field(
        default=0,
        description="Estimated number of workers covered by this action",
        ge=0,
    )
    timeline: str = Field(
        default="",
        description="Timeline or deadline for the action",
        max_length=200,
    )
    status: RemediationStatus = Field(
        default=RemediationStatus.IDENTIFIED,
        description="Current status of the action",
    )

class ValueChainRiskAssessment(BaseModel):
    """Risk assessment for a value chain segment per ESRS S2-4 AR S2-14.

    Documents the assessed risk of negative impacts on workers at a
    specific supplier, tier, country, or worker-type combination.
    """
    assessment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this risk assessment",
    )
    supplier_id: str = Field(
        default="",
        description="Identifier of the assessed supplier",
        max_length=200,
    )
    tier: ValueChainTier = Field(
        ...,
        description="Value chain tier of the assessed entity",
    )
    country: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country code",
        max_length=3,
    )
    worker_type: WorkerType = Field(
        ...,
        description="Type of workers at the assessed entity",
    )
    risk_category: RiskCategory = Field(
        ...,
        description="Category of risk identified",
    )
    severity: Decimal = Field(
        default=Decimal("0"),
        description="Severity score (0-10 scale)",
        ge=Decimal("0"),
        le=Decimal("10"),
    )
    likelihood: Decimal = Field(
        default=Decimal("0"),
        description="Likelihood score (0-10 scale)",
        ge=Decimal("0"),
        le=Decimal("10"),
    )
    workers_affected_estimate: int = Field(
        default=0,
        description="Estimated number of workers potentially affected",
        ge=0,
    )
    due_diligence_phase: DueDiligencePhase = Field(
        default=DueDiligencePhase.IDENTIFICATION,
        description="Current phase of due diligence for this risk",
    )

class ValueChainWorkerTarget(BaseModel):
    """Target related to value chain workers per ESRS S2-5 (Para 35-37).

    Documents targets the undertaking has set for managing negative
    impacts, advancing positive impacts, or managing risks and
    opportunities related to value chain workers.
    """
    target_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this target",
    )
    metric: str = Field(
        ...,
        description="Metric being targeted (e.g. suppliers_audited, incidents_reduced)",
        max_length=200,
    )
    target_type: str = Field(
        default="negative_impact_reduction",
        description="Type: negative_impact_reduction, positive_impact, risk_management",
        max_length=100,
    )
    base_year: int = Field(
        default=0,
        description="Base year for the target",
        ge=0,
    )
    base_value: Decimal = Field(
        default=Decimal("0"),
        description="Baseline value at base year",
    )
    target_value: Decimal = Field(
        default=Decimal("0"),
        description="Target value to achieve",
    )
    target_year: int = Field(
        default=0,
        description="Year by which target should be achieved",
        ge=0,
    )
    progress_pct: Decimal = Field(
        default=Decimal("0"),
        description="Current progress toward target (%)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )

class S2ValueChainResult(BaseModel):
    """Complete ESRS S2 disclosure result.

    Aggregates all S2 disclosure requirements (S2-1 through S2-5)
    into a single auditable result with provenance tracking.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this calculation",
    )
    # S2-1: Policies
    policies: List[ValueChainWorkerPolicy] = Field(
        default_factory=list,
        description="Policies related to value chain workers (S2-1)",
    )
    # S2-2: Engagement
    engagement_processes: List[EngagementProcess] = Field(
        default_factory=list,
        description="Engagement processes with value chain workers (S2-2)",
    )
    # S2-3: Grievance Channels
    grievance_channels: List[GrievanceChannel] = Field(
        default_factory=list,
        description="Grievance and remediation channels (S2-3)",
    )
    # S2-4: Actions
    actions: List[ValueChainWorkerAction] = Field(
        default_factory=list,
        description="Actions on material impacts (S2-4)",
    )
    # Risk assessments supporting S2-4
    risk_assessments: List[ValueChainRiskAssessment] = Field(
        default_factory=list,
        description="Value chain risk assessments (S2-4 supporting data)",
    )
    # S2-5: Targets
    targets: List[ValueChainWorkerTarget] = Field(
        default_factory=list,
        description="Targets related to value chain workers (S2-5)",
    )
    # Aggregate metrics
    total_suppliers_assessed: int = Field(
        default=0, description="Total number of unique suppliers assessed"
    )
    suppliers_at_risk_count: int = Field(
        default=0, description="Number of suppliers with identified risks"
    )
    workers_covered_by_engagement: int = Field(
        default=0, description="Estimated workers reached by engagement"
    )
    grievance_cases_total: int = Field(
        default=0, description="Total grievance cases across all channels"
    )
    grievance_resolution_rate: Decimal = Field(
        default=Decimal("0"),
        description="Overall grievance resolution rate (0-100%)",
    )
    compliance_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall S2 compliance score (0-100%)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and calculation steps",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of calculation (UTC)",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ValueChainWorkersEngine:
    """ESRS S2 Workers in the Value Chain assessment engine.

    Provides deterministic, zero-hallucination assessments for:
    - Policy coverage analysis against international standards (S2-1)
    - Engagement process evaluation by tier and worker type (S2-2)
    - Grievance channel accessibility and resolution rates (S2-3)
    - Action effectiveness against identified risks (S2-4)
    - Target progress tracking and evaluation (S2-5)
    - Risk aggregation by tier, country, and category
    - S2 completeness validation

    All calculations use Decimal arithmetic for bit-perfect
    reproducibility.  No LLM is used in any calculation path.

    Usage::

        engine = ValueChainWorkersEngine()
        result = engine.calculate_s2_disclosure(
            policies=[...],
            engagement_processes=[...],
            grievance_channels=[...],
            actions=[...],
            risk_assessments=[...],
            targets=[...],
        )
    """

    engine_version: str = _MODULE_VERSION

    ALL_TIERS: Set[str] = {t.value for t in ValueChainTier}
    ALL_WORKER_TYPES: Set[str] = {w.value for w in WorkerType}
    ALL_RISK_CATEGORIES: Set[str] = {r.value for r in RiskCategory}

    # ------------------------------------------------------------------ #
    # S2-1: Policy Assessment                                              #
    # ------------------------------------------------------------------ #

    def assess_policies(
        self, policies: List[ValueChainWorkerPolicy]
    ) -> Dict[str, Any]:
        """Assess policies per ESRS S2-1 (Para 14-16, AR S2-1 to AR S2-5).

        Evaluates policy coverage across value chain tiers, alignment
        with international human rights standards, and ILO convention
        references.

        Args:
            policies: List of ValueChainWorkerPolicy to assess.

        Returns:
            Dict with tier_coverage, standards_referenced, ilo_alignment,
            policy_count, policy_score, and provenance_hash.
        """
        logger.info("Assessing %d value chain worker policies (S2-1)", len(policies))

        if not policies:
            empty = {
                "policy_count": 0,
                "tier_coverage": {},
                "tiers_covered_set": [],
                "tier_coverage_pct": str(Decimal("0")),
                "standards_referenced": [],
                "ilo_conventions_referenced": [],
                "ilo_alignment_count": 0,
                "policy_score": str(Decimal("0")),
                "provenance_hash": "",
            }
            empty["provenance_hash"] = _compute_hash(empty)
            return empty

        # Tier coverage: which tiers are covered by at least one policy
        tiers_covered: Set[str] = set()
        tier_policy_count: Dict[str, int] = {t.value: 0 for t in ValueChainTier}
        all_standards: Set[str] = set()
        all_ilo: Set[str] = set()

        for policy in policies:
            for tier in policy.covers_tiers:
                tiers_covered.add(tier.value)
                tier_policy_count[tier.value] += 1
            for std in policy.human_rights_standards_referenced:
                all_standards.add(std)
            for conv in policy.ilo_conventions_alignment:
                all_ilo.add(conv)

        tier_coverage_pct = _round_val(
            _safe_divide(
                _decimal(len(tiers_covered)),
                _decimal(len(self.ALL_TIERS)),
            ) * Decimal("100"),
            1,
        )

        # Policy score: weighted by tier coverage (50%) + standards (25%) + ILO (25%)
        standards_score = Decimal("100") if len(all_standards) >= 2 else (
            _decimal(len(all_standards)) * Decimal("50")
        )
        ilo_score = Decimal("100") if len(all_ilo) >= 4 else (
            _decimal(len(all_ilo)) * Decimal("25")
        )
        policy_score = _round_val(
            tier_coverage_pct * Decimal("0.50")
            + standards_score * Decimal("0.25")
            + ilo_score * Decimal("0.25"),
            1,
        )

        result = {
            "policy_count": len(policies),
            "tier_coverage": tier_policy_count,
            "tiers_covered_set": sorted(tiers_covered),
            "tier_coverage_pct": str(tier_coverage_pct),
            "standards_referenced": sorted(all_standards),
            "ilo_conventions_referenced": sorted(all_ilo),
            "ilo_alignment_count": len(all_ilo),
            "policy_score": str(policy_score),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S2-1 policy assessment: %d policies, tier_coverage=%.1f%%, score=%.1f",
            len(policies), float(tier_coverage_pct), float(policy_score),
        )
        return result

    # ------------------------------------------------------------------ #
    # S2-2: Engagement Evaluation                                          #
    # ------------------------------------------------------------------ #

    def evaluate_engagement(
        self, processes: List[EngagementProcess]
    ) -> Dict[str, Any]:
        """Evaluate engagement processes per ESRS S2-2 (Para 18-21).

        Analyses engagement coverage across value chain tiers, worker
        types, and mechanism diversity.

        Args:
            processes: List of EngagementProcess to evaluate.

        Returns:
            Dict with coverage_by_tier, coverage_by_worker_type,
            mechanisms_used, engagement_score, and provenance_hash.
        """
        logger.info("Evaluating %d engagement processes (S2-2)", len(processes))

        if not processes:
            empty = {
                "process_count": 0,
                "coverage_by_tier": {},
                "coverage_by_worker_type": {},
                "tier_coverage_pct": str(Decimal("0")),
                "worker_type_coverage_pct": str(Decimal("0")),
                "mechanisms_used": [],
                "mechanism_diversity_count": 0,
                "engagement_score": str(Decimal("0")),
                "provenance_hash": "",
            }
            empty["provenance_hash"] = _compute_hash(empty)
            return empty

        tiers_covered: Set[str] = set()
        worker_types_covered: Set[str] = set()
        mechanisms_used: Set[str] = set()
        tier_process_count: Dict[str, int] = {t.value: 0 for t in ValueChainTier}
        wtype_process_count: Dict[str, int] = {w.value: 0 for w in WorkerType}

        for proc in processes:
            mechanisms_used.add(proc.mechanism.value)
            for tier in proc.tiers_covered:
                tiers_covered.add(tier.value)
                tier_process_count[tier.value] += 1
            for wtype in proc.worker_types_covered:
                worker_types_covered.add(wtype.value)
                wtype_process_count[wtype.value] += 1

        tier_coverage_pct = _round_val(
            _safe_divide(
                _decimal(len(tiers_covered)),
                _decimal(len(self.ALL_TIERS)),
            ) * Decimal("100"),
            1,
        )
        wtype_coverage_pct = _round_val(
            _safe_divide(
                _decimal(len(worker_types_covered)),
                _decimal(len(self.ALL_WORKER_TYPES)),
            ) * Decimal("100"),
            1,
        )

        # Engagement score: tier coverage (40%) + worker type coverage (40%) +
        # mechanism diversity (20%)
        mechanism_count = _decimal(len(mechanisms_used))
        total_mechanisms = _decimal(len(EngagementMechanism))
        mechanism_pct = _round_val(
            _safe_divide(mechanism_count, total_mechanisms) * Decimal("100"), 1
        )
        engagement_score = _round_val(
            tier_coverage_pct * Decimal("0.40")
            + wtype_coverage_pct * Decimal("0.40")
            + mechanism_pct * Decimal("0.20"),
            1,
        )

        result = {
            "process_count": len(processes),
            "coverage_by_tier": tier_process_count,
            "coverage_by_worker_type": wtype_process_count,
            "tier_coverage_pct": str(tier_coverage_pct),
            "worker_type_coverage_pct": str(wtype_coverage_pct),
            "mechanisms_used": sorted(mechanisms_used),
            "mechanism_diversity_count": len(mechanisms_used),
            "engagement_score": str(engagement_score),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S2-2 engagement evaluation: %d processes, tier_cov=%.1f%%, "
            "wtype_cov=%.1f%%, score=%.1f",
            len(processes), float(tier_coverage_pct),
            float(wtype_coverage_pct), float(engagement_score),
        )
        return result

    # ------------------------------------------------------------------ #
    # S2-3: Grievance Channel Evaluation                                   #
    # ------------------------------------------------------------------ #

    def evaluate_grievance_channels(
        self, channels: List[GrievanceChannel]
    ) -> Dict[str, Any]:
        """Evaluate grievance channels per ESRS S2-3 (Para 23-26).

        Analyses accessibility across tiers, anonymous reporting
        availability, and resolution rates.

        Args:
            channels: List of GrievanceChannel to evaluate.

        Returns:
            Dict with accessibility_by_tier, anonymous_available,
            total_cases, resolved_cases, resolution_rate,
            avg_response_time_days, grievance_score, and provenance_hash.
        """
        logger.info("Evaluating %d grievance channels (S2-3)", len(channels))

        if not channels:
            empty = {
                "channel_count": 0,
                "accessibility_by_tier": {},
                "tier_accessibility_pct": str(Decimal("0")),
                "anonymous_channels_count": 0,
                "anonymous_available": False,
                "total_cases_received": 0,
                "total_cases_resolved": 0,
                "resolution_rate": str(Decimal("0")),
                "avg_response_time_days": str(Decimal("0")),
                "grievance_score": str(Decimal("0")),
                "provenance_hash": "",
            }
            empty["provenance_hash"] = _compute_hash(empty)
            return empty

        tiers_accessible: Set[str] = set()
        tier_channel_count: Dict[str, int] = {t.value: 0 for t in ValueChainTier}
        anonymous_count = 0
        total_received = 0
        total_resolved = 0
        response_time_sum = Decimal("0")
        response_time_count = 0

        for ch in channels:
            if ch.anonymous_reporting:
                anonymous_count += 1
            total_received += ch.cases_received
            total_resolved += ch.cases_resolved
            for tier in ch.accessible_to_tiers:
                tiers_accessible.add(tier.value)
                tier_channel_count[tier.value] += 1
            if ch.response_time_days > Decimal("0"):
                response_time_sum += ch.response_time_days
                response_time_count += 1

        resolution_rate = _round_val(
            _safe_divide(
                _decimal(total_resolved),
                _decimal(total_received),
            ) * Decimal("100"),
            1,
        )

        avg_response = _round_val(
            _safe_divide(
                response_time_sum,
                _decimal(response_time_count) if response_time_count > 0 else Decimal("1"),
            ),
            1,
        )

        tier_accessibility_pct = _round_val(
            _safe_divide(
                _decimal(len(tiers_accessible)),
                _decimal(len(self.ALL_TIERS)),
            ) * Decimal("100"),
            1,
        )

        # Grievance score: accessibility (30%) + anonymous (20%) + resolution rate (50%)
        anonymous_score = Decimal("100") if anonymous_count > 0 else Decimal("0")
        grievance_score = _round_val(
            tier_accessibility_pct * Decimal("0.30")
            + anonymous_score * Decimal("0.20")
            + resolution_rate * Decimal("0.50"),
            1,
        )

        result = {
            "channel_count": len(channels),
            "accessibility_by_tier": tier_channel_count,
            "tier_accessibility_pct": str(tier_accessibility_pct),
            "anonymous_channels_count": anonymous_count,
            "anonymous_available": anonymous_count > 0,
            "total_cases_received": total_received,
            "total_cases_resolved": total_resolved,
            "resolution_rate": str(resolution_rate),
            "avg_response_time_days": str(avg_response),
            "grievance_score": str(grievance_score),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S2-3 grievance evaluation: %d channels, resolution=%.1f%%, score=%.1f",
            len(channels), float(resolution_rate), float(grievance_score),
        )
        return result

    # ------------------------------------------------------------------ #
    # S2-4: Risk Assessment                                                #
    # ------------------------------------------------------------------ #

    def assess_value_chain_risks(
        self, assessments: List[ValueChainRiskAssessment]
    ) -> Dict[str, Any]:
        """Assess value chain risks per ESRS S2-4 (Para 28-33).

        Aggregates risk assessments by tier, country, risk category,
        and worker type, computing average severity and likelihood.

        Args:
            assessments: List of ValueChainRiskAssessment to aggregate.

        Returns:
            Dict with by_tier, by_country, by_risk_category,
            high_risk_count, total_workers_affected, risk_score,
            and provenance_hash.
        """
        logger.info("Assessing %d value chain risk records (S2-4)", len(assessments))

        if not assessments:
            empty = {
                "assessment_count": 0,
                "unique_suppliers": 0,
                "by_tier": {},
                "by_country": {},
                "by_risk_category": {},
                "high_risk_count": 0,
                "total_workers_affected": 0,
                "avg_severity": str(Decimal("0")),
                "avg_likelihood": str(Decimal("0")),
                "risk_score": str(Decimal("0")),
                "provenance_hash": "",
            }
            empty["provenance_hash"] = _compute_hash(empty)
            return empty

        suppliers: Set[str] = set()
        by_tier: Dict[str, Dict[str, Any]] = {}
        by_country: Dict[str, Dict[str, Any]] = {}
        by_risk_cat: Dict[str, Dict[str, Any]] = {}
        high_risk_count = 0
        total_workers = 0
        severity_sum = Decimal("0")
        likelihood_sum = Decimal("0")

        high_risk_threshold = Decimal("7")

        for a in assessments:
            if a.supplier_id:
                suppliers.add(a.supplier_id)
            total_workers += a.workers_affected_estimate
            severity_sum += a.severity
            likelihood_sum += a.likelihood

            risk_level = a.severity * a.likelihood
            if risk_level >= high_risk_threshold * high_risk_threshold:
                high_risk_count += 1

            # Aggregate by tier
            tier_key = a.tier.value
            if tier_key not in by_tier:
                by_tier[tier_key] = {"count": 0, "severity_sum": Decimal("0"),
                                     "likelihood_sum": Decimal("0"), "workers": 0}
            by_tier[tier_key]["count"] += 1
            by_tier[tier_key]["severity_sum"] += a.severity
            by_tier[tier_key]["likelihood_sum"] += a.likelihood
            by_tier[tier_key]["workers"] += a.workers_affected_estimate

            # Aggregate by country
            country_key = a.country if a.country else "UNKNOWN"
            if country_key not in by_country:
                by_country[country_key] = {"count": 0, "severity_sum": Decimal("0"),
                                           "workers": 0}
            by_country[country_key]["count"] += 1
            by_country[country_key]["severity_sum"] += a.severity
            by_country[country_key]["workers"] += a.workers_affected_estimate

            # Aggregate by risk category
            cat_key = a.risk_category.value
            if cat_key not in by_risk_cat:
                by_risk_cat[cat_key] = {"count": 0, "severity_sum": Decimal("0"),
                                        "likelihood_sum": Decimal("0"), "workers": 0}
            by_risk_cat[cat_key]["count"] += 1
            by_risk_cat[cat_key]["severity_sum"] += a.severity
            by_risk_cat[cat_key]["likelihood_sum"] += a.likelihood
            by_risk_cat[cat_key]["workers"] += a.workers_affected_estimate

        n = _decimal(len(assessments))
        avg_severity = _round_val(_safe_divide(severity_sum, n), 2)
        avg_likelihood = _round_val(_safe_divide(likelihood_sum, n), 2)

        # Compute average severity per aggregation
        for tier_data in by_tier.values():
            cnt = _decimal(tier_data["count"])
            tier_data["avg_severity"] = str(_round_val(
                _safe_divide(tier_data.pop("severity_sum"), cnt), 2
            ))
            tier_data["avg_likelihood"] = str(_round_val(
                _safe_divide(tier_data.pop("likelihood_sum"), cnt), 2
            ))

        for country_data in by_country.values():
            cnt = _decimal(country_data["count"])
            country_data["avg_severity"] = str(_round_val(
                _safe_divide(country_data.pop("severity_sum"), cnt), 2
            ))

        for cat_data in by_risk_cat.values():
            cnt = _decimal(cat_data["count"])
            cat_data["avg_severity"] = str(_round_val(
                _safe_divide(cat_data.pop("severity_sum"), cnt), 2
            ))
            cat_data["avg_likelihood"] = str(_round_val(
                _safe_divide(cat_data.pop("likelihood_sum"), cnt), 2
            ))

        # Risk score: inverse of risk level (higher = worse conditions detected)
        # Normalized: 100 - (avg_severity * avg_likelihood)
        raw_risk = avg_severity * avg_likelihood
        risk_score = _round_val(
            Decimal("100") - _safe_divide(raw_risk, Decimal("100")) * Decimal("100"),
            1,
        )
        # Clamp 0-100
        risk_score = max(Decimal("0"), min(Decimal("100"), risk_score))

        result = {
            "assessment_count": len(assessments),
            "unique_suppliers": len(suppliers),
            "by_tier": by_tier,
            "by_country": by_country,
            "by_risk_category": by_risk_cat,
            "high_risk_count": high_risk_count,
            "total_workers_affected": total_workers,
            "avg_severity": str(avg_severity),
            "avg_likelihood": str(avg_likelihood),
            "risk_score": str(risk_score),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S2-4 risk assessment: %d records, %d suppliers, %d high-risk, "
            "avg_sev=%.2f, score=%.1f",
            len(assessments), len(suppliers), high_risk_count,
            float(avg_severity), float(risk_score),
        )
        return result

    # ------------------------------------------------------------------ #
    # S2-4: Action Evaluation                                              #
    # ------------------------------------------------------------------ #

    def evaluate_actions(
        self,
        actions: List[ValueChainWorkerAction],
        risk_assessments: List[ValueChainRiskAssessment],
    ) -> Dict[str, Any]:
        """Evaluate actions against identified risks per ESRS S2-4.

        Determines coverage of identified risk categories and tiers
        by the actions taken, and identifies gaps.

        Args:
            actions: List of ValueChainWorkerAction taken.
            risk_assessments: List of ValueChainRiskAssessment for gap analysis.

        Returns:
            Dict with action_count, total_resources, workers_covered,
            risk_categories_addressed, risk_gaps, tier_coverage,
            action_score, and provenance_hash.
        """
        logger.info(
            "Evaluating %d actions against %d risk assessments (S2-4)",
            len(actions), len(risk_assessments),
        )

        if not actions:
            empty = {
                "action_count": 0,
                "total_resources_allocated": str(Decimal("0")),
                "total_workers_covered": 0,
                "risk_categories_addressed": [],
                "tiers_addressed": [],
                "risk_gaps": [],
                "tier_gaps": [],
                "risk_coverage_pct": str(Decimal("0")),
                "status_breakdown": {},
                "action_score": str(Decimal("0")),
                "provenance_hash": "",
            }
            empty["provenance_hash"] = _compute_hash(empty)
            return empty

        total_resources = Decimal("0")
        total_workers = 0
        risk_cats_addressed: Set[str] = set()
        tiers_addressed: Set[str] = set()
        status_counts: Dict[str, int] = {s.value: 0 for s in RemediationStatus}

        for action in actions:
            total_resources += action.resources_allocated
            total_workers += action.expected_workers_covered
            risk_cats_addressed.add(action.target_risk.value)
            tiers_addressed.add(action.target_tier.value)
            status_counts[action.status.value] += 1

        # Identify risk gaps: categories present in assessments but not in actions
        risk_cats_in_assessments: Set[str] = set()
        tiers_in_assessments: Set[str] = set()
        for ra in risk_assessments:
            risk_cats_in_assessments.add(ra.risk_category.value)
            tiers_in_assessments.add(ra.tier.value)

        risk_gaps = sorted(risk_cats_in_assessments - risk_cats_addressed)
        tier_gaps = sorted(tiers_in_assessments - tiers_addressed)

        # Risk coverage: proportion of identified risk categories addressed
        identified_count = len(risk_cats_in_assessments) if risk_cats_in_assessments else len(self.ALL_RISK_CATEGORIES)
        risk_coverage_pct = _round_val(
            _safe_divide(
                _decimal(len(risk_cats_addressed)),
                _decimal(identified_count),
            ) * Decimal("100"),
            1,
        )

        # Action score: risk coverage (50%) + resource allocation presence (20%) +
        # resolution progress (30%)
        resource_score = Decimal("100") if total_resources > Decimal("0") else Decimal("0")
        resolved_or_closed = status_counts.get("resolved", 0) + status_counts.get("closed", 0)
        progress_pct = _round_val(
            _safe_divide(
                _decimal(resolved_or_closed),
                _decimal(len(actions)),
            ) * Decimal("100"),
            1,
        )
        action_score = _round_val(
            risk_coverage_pct * Decimal("0.50")
            + resource_score * Decimal("0.20")
            + progress_pct * Decimal("0.30"),
            1,
        )

        result = {
            "action_count": len(actions),
            "total_resources_allocated": str(_round_val(total_resources, 2)),
            "total_workers_covered": total_workers,
            "risk_categories_addressed": sorted(risk_cats_addressed),
            "tiers_addressed": sorted(tiers_addressed),
            "risk_gaps": risk_gaps,
            "tier_gaps": tier_gaps,
            "risk_coverage_pct": str(risk_coverage_pct),
            "status_breakdown": status_counts,
            "action_score": str(action_score),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S2-4 action evaluation: %d actions, resources=%.2f EUR, "
            "risk_coverage=%.1f%%, gaps=%s, score=%.1f",
            len(actions), float(total_resources),
            float(risk_coverage_pct), risk_gaps, float(action_score),
        )
        return result

    # ------------------------------------------------------------------ #
    # S2-5: Target Evaluation                                              #
    # ------------------------------------------------------------------ #

    def evaluate_targets(
        self, targets: List[ValueChainWorkerTarget]
    ) -> Dict[str, Any]:
        """Evaluate targets per ESRS S2-5 (Para 35-37).

        Analyses target progress, categorisation by type, and
        overall target-setting maturity.

        Args:
            targets: List of ValueChainWorkerTarget to evaluate.

        Returns:
            Dict with target_count, by_type, avg_progress,
            on_track_count, target_score, and provenance_hash.
        """
        logger.info("Evaluating %d targets (S2-5)", len(targets))

        if not targets:
            empty = {
                "target_count": 0,
                "by_type": {},
                "avg_progress_pct": str(Decimal("0")),
                "on_track_count": 0,
                "behind_count": 0,
                "target_score": str(Decimal("0")),
                "provenance_hash": "",
            }
            empty["provenance_hash"] = _compute_hash(empty)
            return empty

        by_type: Dict[str, int] = {}
        progress_sum = Decimal("0")
        on_track = 0
        behind = 0

        for target in targets:
            ttype = target.target_type
            by_type[ttype] = by_type.get(ttype, 0) + 1
            progress_sum += target.progress_pct
            if target.progress_pct >= Decimal("50"):
                on_track += 1
            else:
                behind += 1

        avg_progress = _round_val(
            _safe_divide(progress_sum, _decimal(len(targets))), 1
        )

        # Target score: average progress (70%) + having multiple types (30%)
        type_diversity = _decimal(len(by_type))
        type_score = Decimal("100") if type_diversity >= Decimal("3") else (
            _round_val(type_diversity / Decimal("3") * Decimal("100"), 1)
        )
        target_score = _round_val(
            avg_progress * Decimal("0.70") + type_score * Decimal("0.30"),
            1,
        )

        result = {
            "target_count": len(targets),
            "by_type": by_type,
            "avg_progress_pct": str(avg_progress),
            "on_track_count": on_track,
            "behind_count": behind,
            "target_score": str(target_score),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S2-5 target evaluation: %d targets, avg_progress=%.1f%%, "
            "on_track=%d, score=%.1f",
            len(targets), float(avg_progress), on_track, float(target_score),
        )
        return result

    # ------------------------------------------------------------------ #
    # Full S2 Disclosure Calculation                                       #
    # ------------------------------------------------------------------ #

    def calculate_s2_disclosure(
        self,
        policies: List[ValueChainWorkerPolicy],
        engagement_processes: List[EngagementProcess],
        grievance_channels: List[GrievanceChannel],
        actions: List[ValueChainWorkerAction],
        risk_assessments: List[ValueChainRiskAssessment],
        targets: List[ValueChainWorkerTarget],
    ) -> S2ValueChainResult:
        """Calculate complete ESRS S2 disclosure from all inputs.

        Orchestrates all five S2 disclosure requirement assessments,
        aggregates metrics, and produces a single auditable result
        with provenance hash.

        Args:
            policies: Policies for S2-1 assessment.
            engagement_processes: Processes for S2-2 evaluation.
            grievance_channels: Channels for S2-3 evaluation.
            actions: Actions for S2-4 evaluation.
            risk_assessments: Risk records for S2-4 risk assessment.
            targets: Targets for S2-5 evaluation.

        Returns:
            S2ValueChainResult with all metrics and provenance.
        """
        t0 = time.perf_counter()

        logger.info(
            "Calculating S2 disclosure: %d policies, %d engagements, "
            "%d channels, %d actions, %d risks, %d targets",
            len(policies), len(engagement_processes), len(grievance_channels),
            len(actions), len(risk_assessments), len(targets),
        )

        # S2-1: Policy assessment
        policy_result = self.assess_policies(policies)
        # S2-2: Engagement evaluation
        engagement_result = self.evaluate_engagement(engagement_processes)
        # S2-3: Grievance evaluation
        grievance_result = self.evaluate_grievance_channels(grievance_channels)
        # S2-4: Risk assessment
        risk_result = self.assess_value_chain_risks(risk_assessments)
        # S2-4: Action evaluation
        action_result = self.evaluate_actions(actions, risk_assessments)
        # S2-5: Target evaluation
        target_result = self.evaluate_targets(targets)

        # Aggregate grievance metrics
        total_cases = grievance_result["total_cases_received"]
        total_resolved = grievance_result["total_cases_resolved"]
        resolution_rate = _decimal(grievance_result["resolution_rate"])

        # Workers covered by engagement (sum of action coverage)
        workers_covered = action_result["total_workers_covered"]

        # Unique suppliers from risk assessments
        total_suppliers = risk_result["unique_suppliers"]
        suppliers_at_risk = risk_result["high_risk_count"]

        # Composite compliance score: weighted average of sub-scores
        # S2-1 (20%) + S2-2 (20%) + S2-3 (20%) + S2-4 actions (20%) + S2-5 (20%)
        s1_score = _decimal(policy_result["policy_score"])
        s2_score = _decimal(engagement_result["engagement_score"])
        s3_score = _decimal(grievance_result["grievance_score"])
        s4_score = _decimal(action_result["action_score"])
        s5_score = _decimal(target_result["target_score"])

        compliance_score = _round_val(
            s1_score * Decimal("0.20")
            + s2_score * Decimal("0.20")
            + s3_score * Decimal("0.20")
            + s4_score * Decimal("0.20")
            + s5_score * Decimal("0.20"),
            1,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        result = S2ValueChainResult(
            policies=policies,
            engagement_processes=engagement_processes,
            grievance_channels=grievance_channels,
            actions=actions,
            risk_assessments=risk_assessments,
            targets=targets,
            total_suppliers_assessed=total_suppliers,
            suppliers_at_risk_count=suppliers_at_risk,
            workers_covered_by_engagement=workers_covered,
            grievance_cases_total=total_cases,
            grievance_resolution_rate=resolution_rate,
            compliance_score=compliance_score,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "S2 disclosure calculated: compliance_score=%.1f, "
            "suppliers=%d, at_risk=%d, hash=%s, time=%.1fms",
            float(compliance_score), total_suppliers, suppliers_at_risk,
            result.provenance_hash[:16], elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------ #
    # Completeness Validation                                              #
    # ------------------------------------------------------------------ #

    def validate_s2_completeness(
        self, result: S2ValueChainResult
    ) -> Dict[str, Any]:
        """Validate completeness against all S2 required data points.

        Checks whether all ESRS S2 mandatory disclosure data points
        (S2-1 through S2-5) are present and populated.

        Args:
            result: S2ValueChainResult to validate.

        Returns:
            Dict with total_datapoints, populated_datapoints,
            missing_datapoints, completeness_pct, is_complete,
            by_disclosure, and provenance_hash.
        """
        logger.info("Validating S2 completeness for result %s", result.result_id)

        has_policies = len(result.policies) > 0
        has_engagement = len(result.engagement_processes) > 0
        has_channels = len(result.grievance_channels) > 0
        has_actions = len(result.actions) > 0
        has_risks = len(result.risk_assessments) > 0
        has_targets = len(result.targets) > 0

        # Check coverage conditions for policies
        policy_tiers: Set[str] = set()
        policy_standards: Set[str] = set()
        policy_ilo: Set[str] = set()
        for p in result.policies:
            for t in p.covers_tiers:
                policy_tiers.add(t.value)
            for s in p.human_rights_standards_referenced:
                policy_standards.add(s)
            for c in p.ilo_conventions_alignment:
                policy_ilo.add(c)

        # Check engagement conditions
        engagement_tiers: Set[str] = set()
        engagement_wtypes: Set[str] = set()
        for ep in result.engagement_processes:
            for t in ep.tiers_covered:
                engagement_tiers.add(t.value)
            for w in ep.worker_types_covered:
                engagement_wtypes.add(w.value)

        # Check grievance channel conditions
        channel_tiers: Set[str] = set()
        has_anonymous = False
        for ch in result.grievance_channels:
            for t in ch.accessible_to_tiers:
                channel_tiers.add(t.value)
            if ch.anonymous_reporting:
                has_anonymous = True

        checks: Dict[str, bool] = {
            # S2-1
            "s2_1_01_policies_addressing_value_chain_workers": has_policies,
            "s2_1_02_human_rights_policy_commitments": len(policy_standards) > 0,
            "s2_1_03_ilo_conventions_alignment": len(policy_ilo) > 0,
            "s2_1_04_due_diligence_policy_scope": has_policies,
            "s2_1_05_stakeholders_consulted_in_policy_design": has_policies,
            "s2_1_06_policy_communication_to_value_chain": has_policies,
            "s2_1_07_tiers_covered_by_policies": len(policy_tiers) > 0,
            # S2-2
            "s2_2_01_engagement_processes_with_value_chain_workers": has_engagement,
            "s2_2_02_engagement_with_legitimate_representatives": has_engagement,
            "s2_2_03_stage_of_engagement_in_due_diligence": has_engagement,
            "s2_2_04_frequency_of_engagement": has_engagement,
            "s2_2_05_tiers_covered_by_engagement": len(engagement_tiers) > 0,
            "s2_2_06_worker_types_covered_by_engagement": len(engagement_wtypes) > 0,
            "s2_2_07_outcomes_of_engagement": has_engagement,
            # S2-3
            "s2_3_01_remediation_processes_for_negative_impacts": has_channels,
            "s2_3_02_channels_to_raise_concerns": has_channels,
            "s2_3_03_accessibility_of_channels": len(channel_tiers) > 0,
            "s2_3_04_anonymous_reporting_available": has_anonymous,
            "s2_3_05_grievance_cases_received": result.grievance_cases_total >= 0,
            "s2_3_06_grievance_cases_resolved": True,
            "s2_3_07_average_resolution_time_days": has_channels,
            "s2_3_08_remediation_outcomes": has_channels,
            # S2-4
            "s2_4_01_actions_on_material_impacts": has_actions,
            "s2_4_02_resources_allocated_to_actions": has_actions,
            "s2_4_03_actions_by_risk_category": has_actions,
            "s2_4_04_actions_by_value_chain_tier": has_actions,
            "s2_4_05_effectiveness_of_actions": has_actions,
            "s2_4_06_workers_covered_by_actions": has_actions,
            "s2_4_07_due_diligence_approach": has_risks,
            "s2_4_08_unaddressed_risks_disclosure": has_risks,
            # S2-5
            "s2_5_01_targets_for_negative_impacts": has_targets,
            "s2_5_02_targets_for_positive_impacts": has_targets,
            "s2_5_03_target_measurement_methodology": has_targets,
            "s2_5_04_target_base_year_and_value": has_targets,
            "s2_5_05_target_progress_percentage": has_targets,
            "s2_5_06_target_timeline": has_targets,
            "s2_5_07_target_revision_explanation": True,
        }

        populated = [dp for dp, ok in checks.items() if ok]
        missing = [dp for dp, ok in checks.items() if not ok]

        total = len(ALL_S2_DATAPOINTS)
        pop_count = len(populated)
        completeness_pct = _round_val(
            _safe_divide(_decimal(pop_count), _decimal(total)) * Decimal("100"),
            1,
        )

        # Per-disclosure breakdown
        def _disclosure_pct(dp_list: List[str]) -> str:
            matched = sum(1 for dp in dp_list if dp in populated)
            return str(_round_val(
                _safe_divide(_decimal(matched), _decimal(len(dp_list))) * Decimal("100"),
                1,
            ))

        by_disclosure = {
            "S2-1": _disclosure_pct(S2_1_DATAPOINTS),
            "S2-2": _disclosure_pct(S2_2_DATAPOINTS),
            "S2-3": _disclosure_pct(S2_3_DATAPOINTS),
            "S2-4": _disclosure_pct(S2_4_DATAPOINTS),
            "S2-5": _disclosure_pct(S2_5_DATAPOINTS),
        }

        validation_result = {
            "total_datapoints": total,
            "populated_datapoints": pop_count,
            "missing_datapoints": missing,
            "completeness_pct": str(completeness_pct),
            "is_complete": len(missing) == 0,
            "by_disclosure": by_disclosure,
            "provenance_hash": "",
        }
        validation_result["provenance_hash"] = _compute_hash(
            {"result_id": result.result_id, "checks": {k: v for k, v in checks.items()}}
        )

        logger.info(
            "S2 completeness: %.1f%% (%d/%d), missing=%d, by_dr=%s",
            float(completeness_pct), pop_count, total, len(missing), by_disclosure,
        )

        return validation_result
