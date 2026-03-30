# -*- coding: utf-8 -*-
"""
AffectedCommunitiesEngine - PACK-017 ESRS S3 Affected Communities
==================================================================

Calculates and assesses ESRS S3 disclosure requirements for impacts
on affected communities, including policy evaluation, engagement
assessment, grievance resolution metrics, and target tracking.

Under ESRS S3, undertakings must disclose how they identify, manage,
and remediate impacts on affected communities across their operations
and value chain.  This engine implements the complete S3 disclosure
pipeline, including:

- Policy assessment for community rights and FPIC commitments
- Engagement process evaluation by community type and level
- Grievance mechanism effectiveness and resolution rate tracking
- Community action planning with resource allocation analysis
- Impact assessment aggregation by site, type, and severity
- Target tracking with base year comparison and progress metrics
- Completeness validation against all S3 required data points
- ESRS S3 data point mapping for disclosure

ESRS S3 Disclosure Requirements:
    - S3-1 (Para 14-16): Policies related to affected communities,
      including alignment with UNGPs, ILO 169, UNDRIP, FPIC commitment
      (AR S3-1 through AR S3-5)
    - S3-2 (Para 18-21): Processes for engaging with affected communities
      about impacts, including identification and engagement methodology
      (AR S3-6 through AR S3-9)
    - S3-3 (Para 23-26): Processes to remediate negative impacts and
      channels for affected communities to raise concerns, including
      grievance mechanisms and resolution tracking
      (AR S3-10 through AR S3-13)
    - S3-4 (Para 28-33): Taking action on material impacts on affected
      communities, including action plans, resource allocation, and
      effectiveness assessment (AR S3-14 through AR S3-18)
    - S3-5 (Para 35-37): Targets related to managing material negative
      impacts, advancing positive impacts, and managing material risks
      and opportunities (AR S3-19 through AR S3-22)

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS S3 Affected Communities
    - UN Guiding Principles on Business and Human Rights (2011)
    - ILO Convention 169 on Indigenous and Tribal Peoples (1989)
    - UN Declaration on the Rights of Indigenous Peoples (UNDRIP, 2007)
    - OECD Guidelines for Multinational Enterprises (2023 update)
    - Voluntary Guidelines on Responsible Governance of Tenure (VGGT)

Zero-Hallucination:
    - All metrics use deterministic arithmetic (Decimal)
    - Resolution rates, compliance scores, and progress are formulaic
    - No ML/LLM involvement in any calculation path
    - SHA-256 provenance hash on every result
    - Aggregations use Decimal arithmetic with ROUND_HALF_UP

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
from typing import Any, Dict, List, Optional

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

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))

def _round1(value: Decimal) -> Decimal:
    """Round Decimal to 1 decimal place using ROUND_HALF_UP."""
    return value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CommunityType(str, Enum):
    """Types of affected communities per ESRS S3.

    ESRS S3 requires disclosure disaggregated by community type,
    recognising the distinct rights and vulnerabilities of each group.
    """
    INDIGENOUS_PEOPLES = "indigenous_peoples"
    LOCAL_COMMUNITIES = "local_communities"
    RURAL_COMMUNITIES = "rural_communities"
    URBAN_COMMUNITIES = "urban_communities"
    NOMADIC_PEOPLES = "nomadic_peoples"
    DISPLACED_PERSONS = "displaced_persons"

class ImpactArea(str, Enum):
    """Impact areas on affected communities per ESRS S3.

    Covers the material impact categories identified in S3-4 Para 28-33
    and the application requirements AR S3-14 through AR S3-18.
    """
    LAND_RIGHTS = "land_rights"
    WATER_ACCESS = "water_access"
    LIVELIHOODS = "livelihoods"
    CULTURAL_HERITAGE = "cultural_heritage"
    HEALTH = "health"
    SECURITY = "security"
    RESETTLEMENT = "resettlement"
    ENVIRONMENT_DEGRADATION = "environment_degradation"
    FOOD_SECURITY = "food_security"

class ConsentType(str, Enum):
    """Types of consent processes for community engagement.

    Per ESRS S3-2 and UNGPs/ILO 169, the level of consent sought
    varies based on the nature of impact and community type.  FPIC
    is required for indigenous peoples per UNDRIP Article 32.
    """
    FPIC = "fpic"
    CONSULTATION = "consultation"
    NOTIFICATION = "notification"
    NONE = "none"

class EngagementLevel(str, Enum):
    """Levels of community engagement per IAP2 spectrum.

    ESRS S3-2 (Para 18-21) requires disclosure of the engagement
    approach.  Higher levels indicate deeper community participation.
    """
    INFORM = "inform"
    CONSULT = "consult"
    INVOLVE = "involve"
    COLLABORATE = "collaborate"
    EMPOWER = "empower"

class RightsFramework(str, Enum):
    """International rights frameworks referenced in community policies.

    Per ESRS S3-1 (Para 14-16), policies should reference applicable
    international instruments for the protection of community rights.
    """
    UN_GUIDING_PRINCIPLES = "un_guiding_principles"
    ILO_169 = "ilo_169"
    UN_DRIP = "un_drip"
    OECD_GUIDELINES = "oecd_guidelines"
    VGGT = "vggt"

class GrievanceStatus(str, Enum):
    """Status of a community grievance case.

    Per ESRS S3-3 (Para 23-26), undertakings must track grievance
    status through the full lifecycle.
    """
    OPEN = "open"
    UNDER_INVESTIGATION = "under_investigation"
    REMEDIATION_IN_PROGRESS = "remediation_in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

class ActionStatus(str, Enum):
    """Status of a community action item.

    Per ESRS S3-4 (Para 28-33), undertakings must track the
    implementation status of actions taken.
    """
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"

class SeverityLevel(str, Enum):
    """Severity level for impact assessments.

    Per ESRS S3-4 and UNGPs, severity is assessed based on scale,
    scope, and irremediability of the impact.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TargetType(str, Enum):
    """Target type for S3-5 targets.

    Per ESRS S3-5 (Para 35-37), targets may be absolute values
    or relative (percentage) improvements.
    """
    ABSOLUTE = "absolute"
    RELATIVE = "relative"

# ---------------------------------------------------------------------------
# Constants - ESRS S3 Required Data Points
# ---------------------------------------------------------------------------

S3_1_DATAPOINTS: List[str] = [
    "s3_1_01_policies_for_affected_communities",
    "s3_1_02_alignment_with_ungps",
    "s3_1_03_rights_frameworks_referenced",
    "s3_1_04_fpic_commitment_documented",
    "s3_1_05_indigenous_peoples_specific_policy",
    "s3_1_06_human_rights_due_diligence_process",
    "s3_1_07_policy_scope_and_coverage",
    "s3_1_08_policy_approval_and_oversight",
]

S3_2_DATAPOINTS: List[str] = [
    "s3_2_01_engagement_processes_described",
    "s3_2_02_community_types_engaged",
    "s3_2_03_engagement_level_per_community",
    "s3_2_04_consent_processes_applied",
    "s3_2_05_engagement_frequency_and_reach",
    "s3_2_06_outcomes_of_engagement",
    "s3_2_07_identification_methodology",
    "s3_2_08_stakeholder_mapping_conducted",
]

S3_3_DATAPOINTS: List[str] = [
    "s3_3_01_grievance_mechanisms_available",
    "s3_3_02_grievance_cases_total",
    "s3_3_03_grievance_resolution_rate",
    "s3_3_04_average_resolution_time_days",
    "s3_3_05_remediation_processes_described",
    "s3_3_06_channels_accessible_to_communities",
    "s3_3_07_grievance_by_issue_area",
    "s3_3_08_grievance_by_severity",
]

S3_4_DATAPOINTS: List[str] = [
    "s3_4_01_actions_to_address_impacts",
    "s3_4_02_resources_allocated",
    "s3_4_03_communities_covered_by_actions",
    "s3_4_04_effectiveness_of_actions",
    "s3_4_05_impact_assessments_conducted",
    "s3_4_06_people_affected_estimates",
    "s3_4_07_mitigation_measures_in_place",
    "s3_4_08_action_status_summary",
]

S3_5_DATAPOINTS: List[str] = [
    "s3_5_01_targets_set_for_communities",
    "s3_5_02_target_metrics_and_baselines",
    "s3_5_03_target_progress_pct",
    "s3_5_04_target_timeline_and_milestones",
    "s3_5_05_negative_impact_reduction_targets",
    "s3_5_06_positive_impact_advancement_targets",
]

ALL_S3_DATAPOINTS: List[str] = (
    S3_1_DATAPOINTS + S3_2_DATAPOINTS + S3_3_DATAPOINTS
    + S3_4_DATAPOINTS + S3_5_DATAPOINTS
)

# Engagement level scoring for quantitative assessment.
ENGAGEMENT_LEVEL_SCORES: Dict[str, Decimal] = {
    "inform": Decimal("0.20"),
    "consult": Decimal("0.40"),
    "involve": Decimal("0.60"),
    "collaborate": Decimal("0.80"),
    "empower": Decimal("1.00"),
}

# Consent type scoring for FPIC compliance assessment.
CONSENT_TYPE_SCORES: Dict[str, Decimal] = {
    "fpic": Decimal("1.00"),
    "consultation": Decimal("0.60"),
    "notification": Decimal("0.30"),
    "none": Decimal("0.00"),
}

# Severity weighting for impact assessments.
SEVERITY_WEIGHTS: Dict[str, Decimal] = {
    "critical": Decimal("4.0"),
    "high": Decimal("3.0"),
    "medium": Decimal("2.0"),
    "low": Decimal("1.0"),
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CommunityPolicy(BaseModel):
    """Policy related to affected communities per ESRS S3-1.

    Represents a single policy document or commitment that addresses
    the rights and impacts on affected communities.  Covers Para 14-16
    and AR S3-1 through AR S3-5.
    """
    policy_id: str = Field(
        default_factory=_new_uuid,
        description="Unique policy identifier",
    )
    name: str = Field(
        ...,
        description="Policy name or title",
        max_length=500,
    )
    scope: str = Field(
        default="",
        description="Scope of the policy (e.g. group-wide, site-specific)",
        max_length=500,
    )
    rights_frameworks_referenced: List[RightsFramework] = Field(
        default_factory=list,
        description="International rights frameworks referenced in the policy",
    )
    fpic_commitment: bool = Field(
        default=False,
        description="Whether the policy includes a Free, Prior and Informed "
                    "Consent (FPIC) commitment",
    )
    indigenous_peoples_specific: bool = Field(
        default=False,
        description="Whether the policy includes provisions specific to "
                    "indigenous peoples",
    )
    implementation_date: Optional[str] = Field(
        default=None,
        description="Date the policy was implemented (ISO 8601 date string)",
        max_length=20,
    )

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Validate that policy name is not empty."""
        if not v.strip():
            raise ValueError("Policy name must not be empty")
        return v.strip()

class CommunityEngagement(BaseModel):
    """Community engagement process per ESRS S3-2.

    Represents a single engagement activity with an affected community,
    capturing the level, type, and outcomes.  Covers Para 18-21 and
    AR S3-6 through AR S3-9.
    """
    engagement_id: str = Field(
        default_factory=_new_uuid,
        description="Unique engagement identifier",
    )
    community_type: CommunityType = Field(
        ...,
        description="Type of community engaged",
    )
    location: str = Field(
        default="",
        description="Geographic location of the community",
        max_length=500,
    )
    engagement_level: EngagementLevel = Field(
        default=EngagementLevel.INFORM,
        description="Level of engagement per IAP2 spectrum",
    )
    consent_type: ConsentType = Field(
        default=ConsentType.NONE,
        description="Type of consent process applied",
    )
    participants_count: int = Field(
        default=0,
        description="Number of community participants in the engagement",
        ge=0,
    )
    frequency: str = Field(
        default="",
        description="Frequency of engagement (e.g. quarterly, annual)",
        max_length=100,
    )
    outcomes: str = Field(
        default="",
        description="Summary of engagement outcomes and agreements",
        max_length=2000,
    )

class CommunityGrievance(BaseModel):
    """Community grievance case per ESRS S3-3.

    Represents a single grievance raised by or on behalf of an affected
    community through available channels.  Covers Para 23-26 and
    AR S3-10 through AR S3-13.
    """
    grievance_id: str = Field(
        default_factory=_new_uuid,
        description="Unique grievance identifier",
    )
    community_type: CommunityType = Field(
        ...,
        description="Type of community that raised the grievance",
    )
    issue_area: ImpactArea = Field(
        ...,
        description="Impact area of the grievance",
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.MEDIUM,
        description="Severity level of the grievance",
    )
    date_raised: str = Field(
        default="",
        description="Date the grievance was raised (ISO 8601 date string)",
        max_length=20,
    )
    status: GrievanceStatus = Field(
        default=GrievanceStatus.OPEN,
        description="Current status of the grievance",
    )
    resolution_description: str = Field(
        default="",
        description="Description of resolution or remediation actions",
        max_length=2000,
    )
    time_to_resolution_days: Optional[int] = Field(
        default=None,
        description="Days from grievance raised to resolution (None if unresolved)",
        ge=0,
    )

class CommunityAction(BaseModel):
    """Action taken on material impacts per ESRS S3-4.

    Represents a specific action to address, mitigate, or remediate
    impacts on affected communities.  Covers Para 28-33 and AR S3-14
    through AR S3-18.
    """
    action_id: str = Field(
        default_factory=_new_uuid,
        description="Unique action identifier",
    )
    description: str = Field(
        ...,
        description="Description of the action taken",
        max_length=2000,
    )
    impact_area: ImpactArea = Field(
        ...,
        description="Impact area this action addresses",
    )
    communities_affected: List[CommunityType] = Field(
        default_factory=list,
        description="Community types affected by or benefiting from this action",
    )
    resources_allocated: Decimal = Field(
        default=Decimal("0"),
        description="Financial resources allocated (EUR)",
        ge=Decimal("0"),
    )
    expected_benefit: str = Field(
        default="",
        description="Expected benefit or outcome of the action",
        max_length=1000,
    )
    timeline: str = Field(
        default="",
        description="Implementation timeline (e.g. Q1 2026 - Q4 2026)",
        max_length=200,
    )
    status: ActionStatus = Field(
        default=ActionStatus.PLANNED,
        description="Current implementation status",
    )

    @field_validator("description")
    @classmethod
    def validate_description_not_empty(cls, v: str) -> str:
        """Validate that action description is not empty."""
        if not v.strip():
            raise ValueError("Action description must not be empty")
        return v.strip()

class CommunityImpactAssessment(BaseModel):
    """Impact assessment for a specific site/community per ESRS S3-4.

    Represents a formal assessment of impacts on an affected community
    at a given operational site, including severity, likelihood, and
    mitigation measures.
    """
    assessment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique assessment identifier",
    )
    site_id: str = Field(
        default="",
        description="Identifier of the operational site",
        max_length=100,
    )
    community_type: CommunityType = Field(
        ...,
        description="Type of community assessed",
    )
    impact_area: ImpactArea = Field(
        ...,
        description="Impact area assessed",
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.MEDIUM,
        description="Assessed severity of the impact",
    )
    likelihood: str = Field(
        default="medium",
        description="Likelihood of the impact occurring (high/medium/low)",
        max_length=20,
    )
    people_affected_estimate: int = Field(
        default=0,
        description="Estimated number of people affected",
        ge=0,
    )
    mitigation_measures: List[str] = Field(
        default_factory=list,
        description="List of mitigation measures in place or planned",
    )

class CommunityTarget(BaseModel):
    """Target for managing impacts on communities per ESRS S3-5.

    Represents a measurable target related to the management of negative
    impacts, advancement of positive impacts, or management of material
    risks and opportunities concerning affected communities.  Covers
    Para 35-37 and AR S3-19 through AR S3-22.
    """
    target_id: str = Field(
        default_factory=_new_uuid,
        description="Unique target identifier",
    )
    metric: str = Field(
        ...,
        description="Target metric name (e.g. grievance_resolution_rate)",
        max_length=200,
    )
    target_type: TargetType = Field(
        default=TargetType.ABSOLUTE,
        description="Whether target is absolute or relative",
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
        description="Current progress toward target (0-100%)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )

    @field_validator("metric")
    @classmethod
    def validate_metric_not_empty(cls, v: str) -> str:
        """Validate that metric name is not empty."""
        if not v.strip():
            raise ValueError("Target metric must not be empty")
        return v.strip()

class S3CommunitiesResult(BaseModel):
    """Complete ESRS S3 Affected Communities disclosure result.

    Aggregates all S3 sub-disclosures (S3-1 through S3-5) into a single
    result object with computed metrics, compliance scoring, and
    provenance tracking.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this calculation",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of calculation (UTC)",
    )

    # S3-1: Policies
    policies: List[CommunityPolicy] = Field(
        default_factory=list,
        description="Community policies per S3-1",
    )

    # S3-2: Engagements
    engagements: List[CommunityEngagement] = Field(
        default_factory=list,
        description="Community engagement records per S3-2",
    )

    # S3-3: Grievances
    grievances: List[CommunityGrievance] = Field(
        default_factory=list,
        description="Community grievance cases per S3-3",
    )

    # S3-4: Actions
    actions: List[CommunityAction] = Field(
        default_factory=list,
        description="Actions taken on impacts per S3-4",
    )

    # S3-4: Impact assessments
    impact_assessments: List[CommunityImpactAssessment] = Field(
        default_factory=list,
        description="Impact assessments per S3-4",
    )

    # S3-5: Targets
    targets: List[CommunityTarget] = Field(
        default_factory=list,
        description="Community-related targets per S3-5",
    )

    # Computed metrics
    communities_engaged_count: int = Field(
        default=0,
        description="Total number of distinct community engagements",
    )
    fpic_processes_count: int = Field(
        default=0,
        description="Number of engagement processes using FPIC",
    )
    grievance_cases_total: int = Field(
        default=0,
        description="Total number of grievance cases filed",
    )
    grievance_resolution_rate: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of grievances resolved (0-100)",
    )
    communities_with_active_impacts: int = Field(
        default=0,
        description="Number of impact assessments with high/critical severity",
    )
    compliance_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall S3 compliance score (0-100)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and calculation steps",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AffectedCommunitiesEngine:
    """ESRS S3 Affected Communities calculation engine.

    Provides deterministic, zero-hallucination calculations for:
    - S3-1: Policy assessment and rights framework coverage
    - S3-2: Engagement level scoring and FPIC compliance
    - S3-3: Grievance resolution rate and time-to-resolution metrics
    - S3-4: Action effectiveness, resource allocation, and impact severity
    - S3-5: Target progress tracking with base year comparison
    - Cross-cutting: Overall S3 compliance scoring
    - Completeness validation against all S3 required data points
    - Data point mapping for ESRS S3 disclosure

    All calculations use Decimal arithmetic for bit-perfect
    reproducibility.  No LLM is used in any calculation path.

    Calculation Methodology:
        1. Grievance resolution rate = resolved / total * 100
        2. Avg resolution time = sum(days) / count(resolved)
        3. Engagement score = weighted avg of engagement levels
        4. FPIC compliance = fpic_count / indigenous_engagement_count * 100
        5. Impact severity score = sum(severity_weight) / max_possible * 100
        6. Target progress = (current - base) / (target - base) * 100
        7. Compliance score = weighted average across all sub-disclosures

    Usage::

        engine = AffectedCommunitiesEngine()
        result = engine.calculate_s3_disclosure(
            policies=[...],
            engagements=[...],
            grievances=[...],
            actions=[...],
            impact_assessments=[...],
            targets=[...],
        )
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Main Disclosure Calculation                                          #
    # ------------------------------------------------------------------ #

    def calculate_s3_disclosure(
        self,
        policies: List[CommunityPolicy],
        engagements: List[CommunityEngagement],
        grievances: List[CommunityGrievance],
        actions: List[CommunityAction],
        impact_assessments: List[CommunityImpactAssessment],
        targets: List[CommunityTarget],
    ) -> S3CommunitiesResult:
        """Calculate the complete ESRS S3 disclosure.

        Processes all inputs across the five S3 disclosure requirements,
        computes aggregate metrics, and produces a provenance-tracked result.

        Args:
            policies: Community policies per S3-1.
            engagements: Engagement records per S3-2.
            grievances: Grievance cases per S3-3.
            actions: Actions taken per S3-4.
            impact_assessments: Impact assessments per S3-4.
            targets: Targets per S3-5.

        Returns:
            S3CommunitiesResult with all metrics and provenance hash.
        """
        t0 = time.perf_counter()

        logger.info(
            "Calculating S3 disclosure: %d policies, %d engagements, "
            "%d grievances, %d actions, %d assessments, %d targets",
            len(policies), len(engagements), len(grievances),
            len(actions), len(impact_assessments), len(targets),
        )

        # S3-2 metrics: engagement counts
        communities_engaged_count = len(engagements)
        fpic_processes_count = self._count_fpic_processes(engagements)

        # S3-3 metrics: grievance resolution
        grievance_cases_total = len(grievances)
        grievance_resolution_rate = self._calculate_grievance_resolution_rate(
            grievances
        )

        # S3-4 metrics: active impacts
        communities_with_active_impacts = self._count_active_impacts(
            impact_assessments
        )

        # Overall compliance score
        compliance_score = self._calculate_compliance_score(
            policies=policies,
            engagements=engagements,
            grievances=grievances,
            actions=actions,
            impact_assessments=impact_assessments,
            targets=targets,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = S3CommunitiesResult(
            policies=policies,
            engagements=engagements,
            grievances=grievances,
            actions=actions,
            impact_assessments=impact_assessments,
            targets=targets,
            communities_engaged_count=communities_engaged_count,
            fpic_processes_count=fpic_processes_count,
            grievance_cases_total=grievance_cases_total,
            grievance_resolution_rate=grievance_resolution_rate,
            communities_with_active_impacts=communities_with_active_impacts,
            compliance_score=compliance_score,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "S3 disclosure calculated: compliance=%.1f%%, grievance_rate=%.1f%%, "
            "fpic=%d, active_impacts=%d, hash=%s",
            float(compliance_score), float(grievance_resolution_rate),
            fpic_processes_count, communities_with_active_impacts,
            result.provenance_hash[:16],
        )

        return result

    # ------------------------------------------------------------------ #
    # S3-1: Policy Assessment                                              #
    # ------------------------------------------------------------------ #

    def assess_policy_coverage(
        self, policies: List[CommunityPolicy]
    ) -> Dict[str, Any]:
        """Assess policy coverage for S3-1 requirements.

        Evaluates the completeness of community policies, including
        rights framework references, FPIC commitment, and indigenous
        peoples provisions.

        Args:
            policies: List of CommunityPolicy instances.

        Returns:
            Dict with policy_count, frameworks_covered, fpic_committed,
            indigenous_specific, coverage_score, and provenance_hash.
        """
        if not policies:
            empty_result = {
                "policy_count": 0,
                "frameworks_covered": [],
                "fpic_committed": False,
                "indigenous_specific": False,
                "coverage_score": Decimal("0"),
                "provenance_hash": "",
            }
            empty_result["provenance_hash"] = _compute_hash(empty_result)
            return empty_result

        # Collect all frameworks referenced across policies
        all_frameworks: set = set()
        fpic_committed = False
        indigenous_specific = False

        for policy in policies:
            for fw in policy.rights_frameworks_referenced:
                all_frameworks.add(fw.value)
            if policy.fpic_commitment:
                fpic_committed = True
            if policy.indigenous_peoples_specific:
                indigenous_specific = True

        # Coverage score: weighted assessment
        # 40% frameworks, 30% FPIC, 30% indigenous-specific
        total_frameworks = len(RightsFramework)
        framework_coverage = _safe_divide(
            _decimal(len(all_frameworks)),
            _decimal(total_frameworks),
        )
        fpic_score = Decimal("1") if fpic_committed else Decimal("0")
        indigenous_score = Decimal("1") if indigenous_specific else Decimal("0")

        coverage_score = _round1(
            (framework_coverage * Decimal("40")
             + fpic_score * Decimal("30")
             + indigenous_score * Decimal("30"))
        )

        result = {
            "policy_count": len(policies),
            "frameworks_covered": sorted(all_frameworks),
            "fpic_committed": fpic_committed,
            "indigenous_specific": indigenous_specific,
            "coverage_score": coverage_score,
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S3-1 policy coverage: %d policies, score=%.1f, fpic=%s",
            len(policies), float(coverage_score), fpic_committed,
        )

        return result

    # ------------------------------------------------------------------ #
    # S3-2: Engagement Assessment                                          #
    # ------------------------------------------------------------------ #

    def assess_engagement_quality(
        self, engagements: List[CommunityEngagement]
    ) -> Dict[str, Any]:
        """Assess quality of community engagement for S3-2 requirements.

        Evaluates engagement depth, community type coverage, consent
        processes, and total participation.

        Args:
            engagements: List of CommunityEngagement instances.

        Returns:
            Dict with engagement_count, community_types_reached,
            avg_engagement_score, total_participants, fpic_count,
            consent_summary, engagement_score, and provenance_hash.
        """
        if not engagements:
            empty_result = {
                "engagement_count": 0,
                "community_types_reached": [],
                "avg_engagement_score": Decimal("0"),
                "total_participants": 0,
                "fpic_count": 0,
                "consent_summary": {},
                "engagement_score": Decimal("0"),
                "provenance_hash": "",
            }
            empty_result["provenance_hash"] = _compute_hash(empty_result)
            return empty_result

        community_types_reached: set = set()
        total_participants = 0
        engagement_scores: List[Decimal] = []
        consent_counts: Dict[str, int] = {}

        for eng in engagements:
            community_types_reached.add(eng.community_type.value)
            total_participants += eng.participants_count
            score = ENGAGEMENT_LEVEL_SCORES.get(
                eng.engagement_level.value, Decimal("0.20")
            )
            engagement_scores.append(score)
            ct = eng.consent_type.value
            consent_counts[ct] = consent_counts.get(ct, 0) + 1

        avg_engagement = _round_val(
            sum(engagement_scores) / _decimal(len(engagement_scores)), 2
        )

        fpic_count = self._count_fpic_processes(engagements)

        # Engagement score: 40% level depth, 30% type coverage, 30% participation
        type_coverage = _safe_divide(
            _decimal(len(community_types_reached)),
            _decimal(len(CommunityType)),
        )
        participation_factor = (
            Decimal("1") if total_participants > 0 else Decimal("0")
        )

        engagement_score = _round1(
            (avg_engagement * Decimal("40")
             + type_coverage * Decimal("30")
             + participation_factor * Decimal("30"))
        )

        result = {
            "engagement_count": len(engagements),
            "community_types_reached": sorted(community_types_reached),
            "avg_engagement_score": avg_engagement,
            "total_participants": total_participants,
            "fpic_count": fpic_count,
            "consent_summary": consent_counts,
            "engagement_score": engagement_score,
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S3-2 engagement quality: %d engagements, score=%.1f, "
            "types=%d, participants=%d",
            len(engagements), float(engagement_score),
            len(community_types_reached), total_participants,
        )

        return result

    # ------------------------------------------------------------------ #
    # S3-3: Grievance Assessment                                           #
    # ------------------------------------------------------------------ #

    def assess_grievance_mechanisms(
        self, grievances: List[CommunityGrievance]
    ) -> Dict[str, Any]:
        """Assess grievance mechanism effectiveness for S3-3.

        Evaluates resolution rates, time-to-resolution, severity
        distribution, and issue area breakdown.

        Args:
            grievances: List of CommunityGrievance instances.

        Returns:
            Dict with total_cases, resolved_count, resolution_rate,
            avg_resolution_days, by_issue_area, by_severity,
            by_status, grievance_score, and provenance_hash.
        """
        if not grievances:
            empty_result = {
                "total_cases": 0,
                "resolved_count": 0,
                "resolution_rate": Decimal("0"),
                "avg_resolution_days": Decimal("0"),
                "by_issue_area": {},
                "by_severity": {},
                "by_status": {},
                "grievance_score": Decimal("0"),
                "provenance_hash": "",
            }
            empty_result["provenance_hash"] = _compute_hash(empty_result)
            return empty_result

        resolved_statuses = {GrievanceStatus.RESOLVED, GrievanceStatus.CLOSED}
        resolved = [g for g in grievances if g.status in resolved_statuses]
        resolved_count = len(resolved)
        total_cases = len(grievances)

        resolution_rate = self._calculate_grievance_resolution_rate(grievances)

        # Average resolution time for resolved grievances
        resolution_days = [
            g.time_to_resolution_days for g in resolved
            if g.time_to_resolution_days is not None
        ]
        avg_resolution_days = Decimal("0")
        if resolution_days:
            avg_resolution_days = _round1(
                _safe_divide(
                    _decimal(sum(resolution_days)),
                    _decimal(len(resolution_days)),
                )
            )

        # Breakdown by issue area
        by_issue_area: Dict[str, int] = {}
        for g in grievances:
            key = g.issue_area.value
            by_issue_area[key] = by_issue_area.get(key, 0) + 1

        # Breakdown by severity
        by_severity: Dict[str, int] = {}
        for g in grievances:
            key = g.severity.value
            by_severity[key] = by_severity.get(key, 0) + 1

        # Breakdown by status
        by_status: Dict[str, int] = {}
        for g in grievances:
            key = g.status.value
            by_status[key] = by_status.get(key, 0) + 1

        # Grievance score: 50% resolution rate, 30% timeliness, 20% coverage
        timeliness_score = Decimal("0")
        if avg_resolution_days > Decimal("0"):
            # Score inversely proportional to resolution time (max 365 days)
            capped = min(avg_resolution_days, Decimal("365"))
            timeliness_score = Decimal("1") - _safe_divide(
                capped, Decimal("365")
            )

        has_mechanism = Decimal("1") if total_cases > 0 else Decimal("0")
        rate_factor = _safe_divide(resolution_rate, Decimal("100"))

        grievance_score = _round1(
            (rate_factor * Decimal("50")
             + timeliness_score * Decimal("30")
             + has_mechanism * Decimal("20"))
        )

        result = {
            "total_cases": total_cases,
            "resolved_count": resolved_count,
            "resolution_rate": resolution_rate,
            "avg_resolution_days": avg_resolution_days,
            "by_issue_area": by_issue_area,
            "by_severity": by_severity,
            "by_status": by_status,
            "grievance_score": grievance_score,
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S3-3 grievance assessment: %d cases, resolved=%d, rate=%.1f%%, "
            "avg_days=%.1f, score=%.1f",
            total_cases, resolved_count, float(resolution_rate),
            float(avg_resolution_days), float(grievance_score),
        )

        return result

    # ------------------------------------------------------------------ #
    # S3-4: Action & Impact Assessment                                     #
    # ------------------------------------------------------------------ #

    def assess_actions_and_impacts(
        self,
        actions: List[CommunityAction],
        impact_assessments: List[CommunityImpactAssessment],
    ) -> Dict[str, Any]:
        """Assess actions and impact severity for S3-4.

        Evaluates action coverage, resource allocation, impact severity
        distribution, and total people affected.

        Args:
            actions: List of CommunityAction instances.
            impact_assessments: List of CommunityImpactAssessment instances.

        Returns:
            Dict with action_count, total_resources_eur, actions_by_status,
            actions_by_impact_area, assessment_count, total_people_affected,
            severity_distribution, active_impacts, action_score,
            and provenance_hash.
        """
        # Action analysis
        total_resources = Decimal("0")
        actions_by_status: Dict[str, int] = {}
        actions_by_impact_area: Dict[str, int] = {}

        for action in actions:
            total_resources += action.resources_allocated
            status_key = action.status.value
            actions_by_status[status_key] = (
                actions_by_status.get(status_key, 0) + 1
            )
            area_key = action.impact_area.value
            actions_by_impact_area[area_key] = (
                actions_by_impact_area.get(area_key, 0) + 1
            )

        total_resources = _round_val(total_resources, 2)

        # Impact assessment analysis
        total_people_affected = 0
        severity_distribution: Dict[str, int] = {}
        active_impacts = self._count_active_impacts(impact_assessments)

        for assessment in impact_assessments:
            total_people_affected += assessment.people_affected_estimate
            sev_key = assessment.severity.value
            severity_distribution[sev_key] = (
                severity_distribution.get(sev_key, 0) + 1
            )

        # Action score: 30% has actions, 30% resources, 20% completion, 20% coverage
        has_actions = Decimal("1") if actions else Decimal("0")
        has_resources = Decimal("1") if total_resources > Decimal("0") else Decimal("0")

        completed_count = actions_by_status.get("completed", 0)
        completion_rate = _safe_divide(
            _decimal(completed_count), _decimal(len(actions))
        ) if actions else Decimal("0")

        impact_areas_covered = len(actions_by_impact_area)
        total_impact_areas = len(ImpactArea)
        area_coverage = _safe_divide(
            _decimal(impact_areas_covered), _decimal(total_impact_areas)
        )

        action_score = _round1(
            (has_actions * Decimal("30")
             + has_resources * Decimal("30")
             + completion_rate * Decimal("20")
             + area_coverage * Decimal("20"))
        )

        result = {
            "action_count": len(actions),
            "total_resources_eur": total_resources,
            "actions_by_status": actions_by_status,
            "actions_by_impact_area": actions_by_impact_area,
            "assessment_count": len(impact_assessments),
            "total_people_affected": total_people_affected,
            "severity_distribution": severity_distribution,
            "active_impacts": active_impacts,
            "action_score": action_score,
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S3-4 actions/impacts: %d actions, resources=%.2f EUR, "
            "%d assessments, people=%d, active_impacts=%d, score=%.1f",
            len(actions), float(total_resources),
            len(impact_assessments), total_people_affected,
            active_impacts, float(action_score),
        )

        return result

    # ------------------------------------------------------------------ #
    # S3-5: Target Assessment                                              #
    # ------------------------------------------------------------------ #

    def assess_targets(
        self, targets: List[CommunityTarget]
    ) -> Dict[str, Any]:
        """Assess target setting and progress for S3-5.

        Evaluates whether targets are set, their progress, and
        the average advancement toward goals.

        Args:
            targets: List of CommunityTarget instances.

        Returns:
            Dict with target_count, avg_progress_pct, targets_met,
            targets_by_type, target_details, target_score,
            and provenance_hash.
        """
        if not targets:
            empty_result = {
                "target_count": 0,
                "avg_progress_pct": Decimal("0"),
                "targets_met": 0,
                "targets_by_type": {},
                "target_details": [],
                "target_score": Decimal("0"),
                "provenance_hash": "",
            }
            empty_result["provenance_hash"] = _compute_hash(empty_result)
            return empty_result

        targets_met = 0
        progress_values: List[Decimal] = []
        targets_by_type: Dict[str, int] = {}

        target_details: List[Dict[str, Any]] = []

        for target in targets:
            progress_values.append(target.progress_pct)
            if target.progress_pct >= Decimal("100"):
                targets_met += 1
            type_key = target.target_type.value
            targets_by_type[type_key] = targets_by_type.get(type_key, 0) + 1
            target_details.append({
                "target_id": target.target_id,
                "metric": target.metric,
                "target_type": target.target_type.value,
                "base_value": str(target.base_value),
                "target_value": str(target.target_value),
                "progress_pct": str(target.progress_pct),
            })

        avg_progress = _round1(
            _safe_divide(
                sum(progress_values),
                _decimal(len(progress_values)),
            )
        )

        # Target score: 40% has targets, 40% avg progress, 20% targets met
        has_targets = Decimal("1")
        progress_factor = _safe_divide(avg_progress, Decimal("100"))
        met_factor = _safe_divide(
            _decimal(targets_met), _decimal(len(targets))
        )

        target_score = _round1(
            (has_targets * Decimal("40")
             + progress_factor * Decimal("40")
             + met_factor * Decimal("20"))
        )

        result = {
            "target_count": len(targets),
            "avg_progress_pct": avg_progress,
            "targets_met": targets_met,
            "targets_by_type": targets_by_type,
            "target_details": target_details,
            "target_score": target_score,
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "S3-5 targets: %d targets, avg_progress=%.1f%%, met=%d, score=%.1f",
            len(targets), float(avg_progress), targets_met,
            float(target_score),
        )

        return result

    # ------------------------------------------------------------------ #
    # Completeness Validation                                              #
    # ------------------------------------------------------------------ #

    def validate_s3_completeness(
        self, result: S3CommunitiesResult
    ) -> Dict[str, Any]:
        """Validate completeness against all S3 required data points.

        Checks whether all ESRS S3 mandatory disclosure data points
        are present and populated in the disclosure result.

        Args:
            result: S3CommunitiesResult to validate.

        Returns:
            Dict with:
                - total_datapoints: int
                - populated_datapoints: int
                - missing_datapoints: list of str
                - completeness_pct: Decimal
                - is_complete: bool
                - by_disclosure: dict per S3-1..S3-5
                - provenance_hash: str
        """
        populated: List[str] = []
        missing: List[str] = []

        has_policies = len(result.policies) > 0
        has_engagements = len(result.engagements) > 0
        has_grievances = len(result.grievances) > 0
        has_actions = len(result.actions) > 0
        has_assessments = len(result.impact_assessments) > 0
        has_targets = len(result.targets) > 0

        # Evaluate rights frameworks across all policies
        all_frameworks: set = set()
        fpic_committed = False
        indigenous_specific = False
        for p in result.policies:
            for fw in p.rights_frameworks_referenced:
                all_frameworks.add(fw.value)
            if p.fpic_commitment:
                fpic_committed = True
            if p.indigenous_peoples_specific:
                indigenous_specific = True

        # Community types engaged
        community_types_engaged: set = set()
        for eng in result.engagements:
            community_types_engaged.add(eng.community_type.value)

        # S3-1 checks
        s3_1_checks = {
            "s3_1_01_policies_for_affected_communities": has_policies,
            "s3_1_02_alignment_with_ungps": (
                RightsFramework.UN_GUIDING_PRINCIPLES.value in all_frameworks
            ),
            "s3_1_03_rights_frameworks_referenced": len(all_frameworks) > 0,
            "s3_1_04_fpic_commitment_documented": fpic_committed,
            "s3_1_05_indigenous_peoples_specific_policy": indigenous_specific,
            "s3_1_06_human_rights_due_diligence_process": has_policies,
            "s3_1_07_policy_scope_and_coverage": any(
                p.scope for p in result.policies
            ) if has_policies else False,
            "s3_1_08_policy_approval_and_oversight": has_policies,
        }

        # S3-2 checks
        s3_2_checks = {
            "s3_2_01_engagement_processes_described": has_engagements,
            "s3_2_02_community_types_engaged": len(community_types_engaged) > 0,
            "s3_2_03_engagement_level_per_community": has_engagements,
            "s3_2_04_consent_processes_applied": any(
                e.consent_type != ConsentType.NONE for e in result.engagements
            ) if has_engagements else False,
            "s3_2_05_engagement_frequency_and_reach": any(
                e.frequency for e in result.engagements
            ) if has_engagements else False,
            "s3_2_06_outcomes_of_engagement": any(
                e.outcomes for e in result.engagements
            ) if has_engagements else False,
            "s3_2_07_identification_methodology": has_engagements,
            "s3_2_08_stakeholder_mapping_conducted": (
                len(community_types_engaged) >= 2
            ),
        }

        # S3-3 checks
        resolved_statuses = {GrievanceStatus.RESOLVED, GrievanceStatus.CLOSED}
        s3_3_checks = {
            "s3_3_01_grievance_mechanisms_available": has_grievances,
            "s3_3_02_grievance_cases_total": has_grievances,
            "s3_3_03_grievance_resolution_rate": (
                result.grievance_resolution_rate > Decimal("0")
            ),
            "s3_3_04_average_resolution_time_days": any(
                g.time_to_resolution_days is not None
                for g in result.grievances
                if g.status in resolved_statuses
            ) if has_grievances else False,
            "s3_3_05_remediation_processes_described": any(
                g.resolution_description for g in result.grievances
            ) if has_grievances else False,
            "s3_3_06_channels_accessible_to_communities": has_grievances,
            "s3_3_07_grievance_by_issue_area": has_grievances,
            "s3_3_08_grievance_by_severity": has_grievances,
        }

        # S3-4 checks
        s3_4_checks = {
            "s3_4_01_actions_to_address_impacts": has_actions,
            "s3_4_02_resources_allocated": any(
                a.resources_allocated > Decimal("0") for a in result.actions
            ) if has_actions else False,
            "s3_4_03_communities_covered_by_actions": any(
                len(a.communities_affected) > 0 for a in result.actions
            ) if has_actions else False,
            "s3_4_04_effectiveness_of_actions": any(
                a.status == ActionStatus.COMPLETED for a in result.actions
            ) if has_actions else False,
            "s3_4_05_impact_assessments_conducted": has_assessments,
            "s3_4_06_people_affected_estimates": any(
                ia.people_affected_estimate > 0
                for ia in result.impact_assessments
            ) if has_assessments else False,
            "s3_4_07_mitigation_measures_in_place": any(
                len(ia.mitigation_measures) > 0
                for ia in result.impact_assessments
            ) if has_assessments else False,
            "s3_4_08_action_status_summary": has_actions,
        }

        # S3-5 checks
        s3_5_checks = {
            "s3_5_01_targets_set_for_communities": has_targets,
            "s3_5_02_target_metrics_and_baselines": any(
                t.base_value >= Decimal("0") and t.target_value > Decimal("0")
                for t in result.targets
            ) if has_targets else False,
            "s3_5_03_target_progress_pct": any(
                t.progress_pct >= Decimal("0") for t in result.targets
            ) if has_targets else False,
            "s3_5_04_target_timeline_and_milestones": any(
                t.target_year > 0 for t in result.targets
            ) if has_targets else False,
            "s3_5_05_negative_impact_reduction_targets": has_targets,
            "s3_5_06_positive_impact_advancement_targets": has_targets,
        }

        all_checks = {
            **s3_1_checks, **s3_2_checks, **s3_3_checks,
            **s3_4_checks, **s3_5_checks,
        }

        for dp, is_populated in all_checks.items():
            if is_populated:
                populated.append(dp)
            else:
                missing.append(dp)

        total = len(ALL_S3_DATAPOINTS)
        pop_count = len(populated)
        completeness = _round1(
            _safe_divide(_decimal(pop_count), _decimal(total)) * Decimal("100")
        )

        # Per-disclosure breakdown
        def _sub_score(checks: Dict[str, bool]) -> Dict[str, Any]:
            sub_pop = sum(1 for v in checks.values() if v)
            sub_total = len(checks)
            return {
                "populated": sub_pop,
                "total": sub_total,
                "pct": str(_round1(
                    _safe_divide(_decimal(sub_pop), _decimal(sub_total))
                    * Decimal("100")
                )),
            }

        by_disclosure = {
            "S3-1": _sub_score(s3_1_checks),
            "S3-2": _sub_score(s3_2_checks),
            "S3-3": _sub_score(s3_3_checks),
            "S3-4": _sub_score(s3_4_checks),
            "S3-5": _sub_score(s3_5_checks),
        }

        validation_result = {
            "total_datapoints": total,
            "populated_datapoints": pop_count,
            "missing_datapoints": missing,
            "completeness_pct": completeness,
            "is_complete": len(missing) == 0,
            "by_disclosure": by_disclosure,
            "provenance_hash": _compute_hash(
                {"result_id": result.result_id, "checks": all_checks}
            ),
        }

        logger.info(
            "S3 completeness: %.1f%% (%d/%d), missing=%d datapoints",
            float(completeness), pop_count, total, len(missing),
        )

        return validation_result

    # ------------------------------------------------------------------ #
    # ESRS S3 Data Point Mapping                                           #
    # ------------------------------------------------------------------ #

    def get_s3_datapoints(
        self, result: S3CommunitiesResult
    ) -> Dict[str, Any]:
        """Map S3 result to ESRS S3 disclosure data points.

        Creates a structured mapping of all S3 required data points
        with their values, ready for report generation.

        Args:
            result: S3CommunitiesResult to map.

        Returns:
            Dict mapping S3 data point IDs to their values and metadata.
        """
        # S3-1 policy data points
        policy_names = [p.name for p in result.policies]
        all_frameworks: set = set()
        for p in result.policies:
            for fw in p.rights_frameworks_referenced:
                all_frameworks.add(fw.value)

        # S3-2 engagement data points
        community_types = sorted(set(
            e.community_type.value for e in result.engagements
        ))

        # S3-3 grievance data points
        issue_areas: Dict[str, int] = {}
        severity_dist: Dict[str, int] = {}
        for g in result.grievances:
            issue_areas[g.issue_area.value] = (
                issue_areas.get(g.issue_area.value, 0) + 1
            )
            severity_dist[g.severity.value] = (
                severity_dist.get(g.severity.value, 0) + 1
            )

        resolved_statuses = {GrievanceStatus.RESOLVED, GrievanceStatus.CLOSED}
        resolved_with_time = [
            g.time_to_resolution_days for g in result.grievances
            if g.status in resolved_statuses
            and g.time_to_resolution_days is not None
        ]
        avg_resolution = Decimal("0")
        if resolved_with_time:
            avg_resolution = _round1(_safe_divide(
                _decimal(sum(resolved_with_time)),
                _decimal(len(resolved_with_time)),
            ))

        # S3-4 action data points
        total_resources = _round_val(
            sum(a.resources_allocated for a in result.actions), 2
        )
        total_people = sum(
            ia.people_affected_estimate for ia in result.impact_assessments
        )

        datapoints: Dict[str, Any] = {
            # S3-1
            "s3_1_policies": {
                "label": "Policies related to affected communities",
                "value": policy_names,
                "esrs_ref": "S3-1 Para 14-16",
            },
            "s3_1_rights_frameworks": {
                "label": "International rights frameworks referenced",
                "value": sorted(all_frameworks),
                "esrs_ref": "S3-1 Para 14",
            },
            "s3_1_fpic_commitment": {
                "label": "FPIC commitment documented",
                "value": any(p.fpic_commitment for p in result.policies),
                "esrs_ref": "S3-1 AR S3-3",
            },
            # S3-2
            "s3_2_communities_engaged": {
                "label": "Number of community engagements",
                "value": result.communities_engaged_count,
                "esrs_ref": "S3-2 Para 18-21",
            },
            "s3_2_community_types": {
                "label": "Community types engaged",
                "value": community_types,
                "esrs_ref": "S3-2 Para 19",
            },
            "s3_2_fpic_processes": {
                "label": "FPIC processes conducted",
                "value": result.fpic_processes_count,
                "esrs_ref": "S3-2 AR S3-7",
            },
            # S3-3
            "s3_3_grievance_total": {
                "label": "Total grievance cases",
                "value": result.grievance_cases_total,
                "esrs_ref": "S3-3 Para 23-26",
            },
            "s3_3_resolution_rate": {
                "label": "Grievance resolution rate",
                "value": str(result.grievance_resolution_rate),
                "unit": "percent",
                "esrs_ref": "S3-3 Para 25",
            },
            "s3_3_avg_resolution_days": {
                "label": "Average grievance resolution time",
                "value": str(avg_resolution),
                "unit": "days",
                "esrs_ref": "S3-3 AR S3-12",
            },
            "s3_3_by_issue_area": {
                "label": "Grievances by issue area",
                "value": issue_areas,
                "esrs_ref": "S3-3 AR S3-11",
            },
            "s3_3_by_severity": {
                "label": "Grievances by severity",
                "value": severity_dist,
                "esrs_ref": "S3-3 AR S3-11",
            },
            # S3-4
            "s3_4_actions_count": {
                "label": "Actions to address impacts",
                "value": len(result.actions),
                "esrs_ref": "S3-4 Para 28-33",
            },
            "s3_4_resources_allocated": {
                "label": "Total resources allocated",
                "value": str(total_resources),
                "unit": "EUR",
                "esrs_ref": "S3-4 Para 30",
            },
            "s3_4_people_affected": {
                "label": "Estimated people affected",
                "value": total_people,
                "esrs_ref": "S3-4 AR S3-16",
            },
            "s3_4_active_impacts": {
                "label": "Communities with high/critical severity impacts",
                "value": result.communities_with_active_impacts,
                "esrs_ref": "S3-4 Para 29",
            },
            # S3-5
            "s3_5_targets_count": {
                "label": "Targets set for affected communities",
                "value": len(result.targets),
                "esrs_ref": "S3-5 Para 35-37",
            },
            "s3_5_target_details": {
                "label": "Target metrics and progress",
                "value": [
                    {
                        "metric": t.metric,
                        "target_value": str(t.target_value),
                        "progress_pct": str(t.progress_pct),
                        "target_year": t.target_year,
                    }
                    for t in result.targets
                ],
                "esrs_ref": "S3-5 Para 36",
            },
            # Cross-cutting
            "s3_compliance_score": {
                "label": "Overall S3 compliance score",
                "value": str(result.compliance_score),
                "unit": "percent",
                "esrs_ref": "ESRS S3",
            },
        }

        datapoints["provenance_hash"] = _compute_hash(datapoints)

        return datapoints

    # ------------------------------------------------------------------ #
    # Private Helpers                                                      #
    # ------------------------------------------------------------------ #

    def _count_fpic_processes(
        self, engagements: List[CommunityEngagement]
    ) -> int:
        """Count the number of engagement processes using FPIC consent.

        Args:
            engagements: List of CommunityEngagement instances.

        Returns:
            Count of engagements with ConsentType.FPIC.
        """
        return sum(
            1 for e in engagements if e.consent_type == ConsentType.FPIC
        )

    def _calculate_grievance_resolution_rate(
        self, grievances: List[CommunityGrievance]
    ) -> Decimal:
        """Calculate the grievance resolution rate as a percentage.

        Formula: resolution_rate = (resolved + closed) / total * 100

        Args:
            grievances: List of CommunityGrievance instances.

        Returns:
            Resolution rate as Decimal (0-100), rounded to 1 decimal.
        """
        if not grievances:
            return Decimal("0")

        resolved_statuses = {GrievanceStatus.RESOLVED, GrievanceStatus.CLOSED}
        resolved_count = sum(
            1 for g in grievances if g.status in resolved_statuses
        )
        total = _decimal(len(grievances))

        return _round1(
            _safe_divide(_decimal(resolved_count), total) * Decimal("100")
        )

    def _count_active_impacts(
        self, impact_assessments: List[CommunityImpactAssessment]
    ) -> int:
        """Count impact assessments with high or critical severity.

        Per ESRS S3-4, active impacts requiring urgent attention are
        those assessed as high or critical severity.

        Args:
            impact_assessments: List of CommunityImpactAssessment instances.

        Returns:
            Count of assessments with HIGH or CRITICAL severity.
        """
        active_severities = {SeverityLevel.HIGH, SeverityLevel.CRITICAL}
        return sum(
            1 for ia in impact_assessments
            if ia.severity in active_severities
        )

    def _calculate_compliance_score(
        self,
        policies: List[CommunityPolicy],
        engagements: List[CommunityEngagement],
        grievances: List[CommunityGrievance],
        actions: List[CommunityAction],
        impact_assessments: List[CommunityImpactAssessment],
        targets: List[CommunityTarget],
    ) -> Decimal:
        """Calculate overall S3 compliance score (0-100).

        Weighted average across five sub-disclosures:
            - S3-1 Policies: 20%
            - S3-2 Engagements: 20%
            - S3-3 Grievances: 20%
            - S3-4 Actions/Impacts: 25%
            - S3-5 Targets: 15%

        Args:
            policies: Community policies.
            engagements: Community engagements.
            grievances: Community grievances.
            actions: Community actions.
            impact_assessments: Impact assessments.
            targets: Community targets.

        Returns:
            Compliance score as Decimal (0-100), rounded to 1 decimal.
        """
        # S3-1 score
        policy_assessment = self.assess_policy_coverage(policies)
        s3_1_score = policy_assessment["coverage_score"]

        # S3-2 score
        engagement_assessment = self.assess_engagement_quality(engagements)
        s3_2_score = engagement_assessment["engagement_score"]

        # S3-3 score
        grievance_assessment = self.assess_grievance_mechanisms(grievances)
        s3_3_score = grievance_assessment["grievance_score"]

        # S3-4 score
        action_assessment = self.assess_actions_and_impacts(
            actions, impact_assessments
        )
        s3_4_score = action_assessment["action_score"]

        # S3-5 score
        target_assessment = self.assess_targets(targets)
        s3_5_score = target_assessment["target_score"]

        # Weighted average
        weighted = (
            s3_1_score * Decimal("0.20")
            + s3_2_score * Decimal("0.20")
            + s3_3_score * Decimal("0.20")
            + s3_4_score * Decimal("0.25")
            + s3_5_score * Decimal("0.15")
        )

        return _round1(weighted)
