# -*- coding: utf-8 -*-
"""
GrievanceMechanismEngine - PACK-019 CSDDD Grievance Mechanism Engine
=====================================================================

Assesses the effectiveness and compliance of corporate grievance
mechanisms per Article 12 of the EU Corporate Sustainability Due
Diligence Directive (CSDDD / CS3D).  Article 12 requires companies
to establish and maintain grievance mechanisms that enable stakeholders
to submit complaints when they have legitimate concerns regarding
actual or potential adverse human rights and environmental impacts.

The engine evaluates grievance mechanisms against the eight
effectiveness criteria from the UN Guiding Principles on Business
and Human Rights (UNGPs), Principle 31:

    1. Legitimate - trusted, accountable for fair conduct
    2. Accessible - known to all stakeholder groups, accessible
    3. Predictable - clear and known procedures, timeframes
    4. Equitable - aggrieved parties have reasonable access
    5. Transparent - keeping parties informed of progress
    6. Rights-compatible - outcomes align with human rights
    7. Continuous Learning - used to identify lessons
    8. Based on Engagement & Dialogue - consulting stakeholders

CSDDD Article 12 Requirements:
    - Art 12(1): Companies shall establish a complaints procedure
    - Art 12(2): The procedure shall be legitimate, accessible,
      predictable, equitable, transparent, rights-compatible
    - Art 12(3): Stakeholders may submit complaints regarding
      adverse impacts
    - Art 12(4): Companies shall inform complainants of outcome

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Article 12: Grievance mechanisms
    - UN Guiding Principles on Business and Human Rights (2011)
    - Principle 31: Effectiveness criteria for non-judicial
      grievance mechanisms
    - OECD Guidelines for MNEs, Chapter IV
    - ISO 26000:2010, Clause 7.7.3

Zero-Hallucination:
    - Resolution rates computed as count ratios
    - Response times computed from date arithmetic
    - Accessibility scores use deterministic weighted criteria
    - Effectiveness scores use deterministic formula over criteria
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-019 CSDDD Readiness
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

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))

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

def _pct_dec(part: Decimal, total: Decimal) -> Decimal:
    """Calculate percentage from Decimal values, rounded to 1 dp."""
    if total == Decimal("0"):
        return Decimal("0.0")
    return _round_val(part / total * Decimal("100"), 1)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GrievanceStatus(str, Enum):
    """Status of a grievance case through the resolution lifecycle.

    Tracks the progression of a complaint from initial receipt
    through investigation to final closure, per Art 12(4) CSDDD.
    """
    RECEIVED = "received"
    UNDER_REVIEW = "under_review"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

class StakeholderGroup(str, Enum):
    """Stakeholder groups who may submit grievances per Art 12(3).

    The CSDDD identifies categories of persons entitled to submit
    complaints through the grievance mechanism.
    """
    WORKERS = "workers"
    TRADE_UNIONS = "trade_unions"
    COMMUNITIES = "communities"
    INDIGENOUS_PEOPLES = "indigenous_peoples"
    NGOS = "ngos"
    INVESTORS = "investors"
    CONSUMERS = "consumers"
    REGULATORS = "regulators"

class GrievanceChannel(str, Enum):
    """Channels through which grievances can be submitted.

    Per UNGPs Principle 31, mechanisms must be accessible, which
    requires availability through multiple channels appropriate
    to the stakeholder groups concerned.
    """
    HOTLINE = "hotline"
    EMAIL = "email"
    WEB_PORTAL = "web_portal"
    IN_PERSON = "in_person"
    MOBILE_APP = "mobile_app"
    POSTAL = "postal"
    TRADE_UNION_REP = "trade_union_rep"

class MechanismCriteria(str, Enum):
    """UNGP Principle 31 effectiveness criteria for grievance mechanisms.

    These eight criteria define what makes a non-judicial grievance
    mechanism effective according to the UN Guiding Principles on
    Business and Human Rights.
    """
    LEGITIMATE = "legitimate"
    ACCESSIBLE = "accessible"
    PREDICTABLE = "predictable"
    EQUITABLE = "equitable"
    TRANSPARENT = "transparent"
    RIGHTS_COMPATIBLE = "rights_compatible"
    CONTINUOUS_LEARNING = "continuous_learning"
    BASED_ON_ENGAGEMENT_DIALOGUE = "based_on_engagement_dialogue"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_UNGP_CRITERIA: List[MechanismCriteria] = list(MechanismCriteria)

# Weights for each UNGP criterion when computing the overall
# effectiveness score.  Accessible and Rights-compatible are
# weighted higher as they are core CSDDD Art 12 requirements.
CRITERIA_WEIGHTS: Dict[str, Decimal] = {
    MechanismCriteria.LEGITIMATE.value: Decimal("0.125"),
    MechanismCriteria.ACCESSIBLE.value: Decimal("0.15"),
    MechanismCriteria.PREDICTABLE.value: Decimal("0.125"),
    MechanismCriteria.EQUITABLE.value: Decimal("0.125"),
    MechanismCriteria.TRANSPARENT.value: Decimal("0.125"),
    MechanismCriteria.RIGHTS_COMPATIBLE.value: Decimal("0.15"),
    MechanismCriteria.CONTINUOUS_LEARNING.value: Decimal("0.10"),
    MechanismCriteria.BASED_ON_ENGAGEMENT_DIALOGUE.value: Decimal("0.10"),
}

# Art 12 target response time thresholds (days)
RESPONSE_TARGET_ACKNOWLEDGEMENT_DAYS: int = 5
RESPONSE_TARGET_INITIAL_REVIEW_DAYS: int = 14
RESPONSE_TARGET_RESOLUTION_DAYS: int = 90

# Accessibility sub-criteria
ACCESSIBILITY_SUBCRITERIA: List[str] = [
    "multiple_channels_available",
    "languages_supported",
    "no_cost_to_complainant",
    "anonymous_submission_allowed",
    "disability_accessible",
    "available_to_all_stakeholder_groups",
    "publicised_to_stakeholders",
    "geographically_accessible",
]

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class GrievanceCase(BaseModel):
    """A single grievance case submitted through the mechanism.

    Represents one complaint or concern raised by a stakeholder
    regarding actual or potential adverse human rights or
    environmental impacts, per Art 12(3) CSDDD.
    """
    case_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this grievance case",
    )
    status: GrievanceStatus = Field(
        ...,
        description="Current status of the grievance case",
    )
    submitted_by: str = Field(
        default="",
        description="Name or identifier of the complainant (may be anonymous)",
        max_length=500,
    )
    stakeholder_group: StakeholderGroup = Field(
        ...,
        description="Stakeholder group the complainant belongs to",
    )
    channel: GrievanceChannel = Field(
        default=GrievanceChannel.WEB_PORTAL,
        description="Channel through which the grievance was submitted",
    )
    description: str = Field(
        default="",
        description="Description of the grievance or complaint",
        max_length=5000,
    )
    adverse_impact_ref: str = Field(
        default="",
        description="Reference to the adverse impact this grievance relates to",
        max_length=500,
    )
    category: str = Field(
        default="",
        description="Category of the grievance (human_rights, environment, other)",
        max_length=200,
    )
    resolution: str = Field(
        default="",
        description="Description of the resolution or remedy provided",
        max_length=5000,
    )
    days_to_acknowledge: Optional[int] = Field(
        default=None,
        description="Days from submission to acknowledgement",
        ge=0,
    )
    days_to_resolve: Optional[int] = Field(
        default=None,
        description="Days from submission to resolution",
        ge=0,
    )
    submitted_date: datetime = Field(
        default_factory=utcnow,
        description="Date the grievance was submitted",
    )
    acknowledged_date: Optional[datetime] = Field(
        default=None,
        description="Date the grievance was acknowledged",
    )
    resolved_date: Optional[datetime] = Field(
        default=None,
        description="Date the grievance was resolved",
    )
    is_anonymous: bool = Field(
        default=False,
        description="Whether the grievance was submitted anonymously",
    )
    complainant_satisfied: Optional[bool] = Field(
        default=None,
        description="Whether the complainant was satisfied with the outcome",
    )

class MechanismConfig(BaseModel):
    """Configuration of the grievance mechanism per Art 12 CSDDD.

    Describes the structural properties of the grievance mechanism
    including available channels, languages, and accessibility
    features, used to assess compliance with UNGP criteria.
    """
    mechanism_id: str = Field(
        default_factory=_new_uuid,
        description="Unique mechanism identifier",
    )
    name: str = Field(
        default="Corporate Grievance Mechanism",
        description="Name of the grievance mechanism",
        max_length=500,
    )
    channels_available: List[GrievanceChannel] = Field(
        default_factory=list,
        description="Channels through which grievances can be submitted",
    )
    languages_supported: List[str] = Field(
        default_factory=list,
        description="Languages the mechanism operates in (ISO 639-1 codes)",
    )
    no_cost_to_complainant: bool = Field(
        default=True,
        description="Whether the mechanism is free for complainants",
    )
    anonymous_submission_allowed: bool = Field(
        default=False,
        description="Whether anonymous submissions are accepted",
    )
    disability_accessible: bool = Field(
        default=False,
        description="Whether the mechanism is accessible to persons with disabilities",
    )
    available_to_all_groups: bool = Field(
        default=False,
        description="Whether the mechanism is available to all stakeholder groups",
    )
    publicised_to_stakeholders: bool = Field(
        default=False,
        description="Whether the mechanism is actively publicised to stakeholders",
    )
    geographically_accessible: bool = Field(
        default=False,
        description="Whether the mechanism is accessible across all operating geographies",
    )
    has_written_procedures: bool = Field(
        default=False,
        description="Whether there are written procedures for handling grievances",
    )
    has_defined_timeframes: bool = Field(
        default=False,
        description="Whether timeframes for each stage are defined",
    )
    has_escalation_process: bool = Field(
        default=False,
        description="Whether an escalation process exists",
    )
    has_appeal_mechanism: bool = Field(
        default=False,
        description="Whether complainants can appeal decisions",
    )
    provides_progress_updates: bool = Field(
        default=False,
        description="Whether complainants receive progress updates",
    )
    outcomes_communicated: bool = Field(
        default=False,
        description="Whether outcomes are communicated to complainants",
    )
    independent_oversight: bool = Field(
        default=False,
        description="Whether an independent body oversees the mechanism",
    )
    rights_based_outcomes: bool = Field(
        default=False,
        description="Whether outcomes are assessed for human rights compatibility",
    )
    lessons_learned_process: bool = Field(
        default=False,
        description="Whether there is a process to identify lessons learned",
    )
    stakeholder_input_on_design: bool = Field(
        default=False,
        description="Whether stakeholders were consulted on mechanism design",
    )
    regular_effectiveness_review: bool = Field(
        default=False,
        description="Whether regular effectiveness reviews are conducted",
    )
    target_acknowledgement_days: int = Field(
        default=RESPONSE_TARGET_ACKNOWLEDGEMENT_DAYS,
        description="Target days for acknowledging a grievance",
        ge=1,
    )
    target_resolution_days: int = Field(
        default=RESPONSE_TARGET_RESOLUTION_DAYS,
        description="Target days for resolving a grievance",
        ge=1,
    )

class MechanismAssessment(BaseModel):
    """Assessment of a single UNGP effectiveness criterion.

    Records the evaluation of one of the eight Principle 31 criteria
    with a score, gap identification, and recommendations.
    """
    criteria: MechanismCriteria = Field(
        ...,
        description="UNGP Principle 31 criterion being assessed",
    )
    met: bool = Field(
        default=False,
        description="Whether the criterion is fully met",
    )
    score: Decimal = Field(
        default=Decimal("0"),
        description="Score for this criterion (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Identified gaps for this criterion",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations to address gaps",
    )

class ResolutionStats(BaseModel):
    """Statistical summary of grievance resolution outcomes."""
    total_cases: int = Field(
        default=0, description="Total grievance cases"
    )
    resolved_count: int = Field(
        default=0, description="Number of cases resolved"
    )
    closed_count: int = Field(
        default=0, description="Number of cases closed"
    )
    escalated_count: int = Field(
        default=0, description="Number of cases escalated"
    )
    pending_count: int = Field(
        default=0, description="Number of cases still pending"
    )
    resolution_rate_pct: Decimal = Field(
        default=Decimal("0.0"), description="Resolution rate (%)"
    )
    satisfaction_count: int = Field(
        default=0, description="Number of satisfied complainants"
    )
    satisfaction_rate_pct: Decimal = Field(
        default=Decimal("0.0"), description="Satisfaction rate (%)"
    )
    by_stakeholder_group: Dict[str, int] = Field(
        default_factory=dict, description="Cases by stakeholder group"
    )
    by_channel: Dict[str, int] = Field(
        default_factory=dict, description="Cases by submission channel"
    )
    by_category: Dict[str, int] = Field(
        default_factory=dict, description="Cases by grievance category"
    )
    by_status: Dict[str, int] = Field(
        default_factory=dict, description="Cases by current status"
    )
    anonymous_count: int = Field(
        default=0, description="Number of anonymous grievances"
    )
    anonymous_pct: Decimal = Field(
        default=Decimal("0.0"), description="Anonymous submission rate (%)"
    )

class ResponseTimeStats(BaseModel):
    """Statistical summary of grievance response times."""
    average_days_to_acknowledge: Decimal = Field(
        default=Decimal("0"),
        description="Average days from submission to acknowledgement",
    )
    median_days_to_acknowledge: Decimal = Field(
        default=Decimal("0"),
        description="Median days from submission to acknowledgement",
    )
    average_days_to_resolve: Decimal = Field(
        default=Decimal("0"),
        description="Average days from submission to resolution",
    )
    median_days_to_resolve: Decimal = Field(
        default=Decimal("0"),
        description="Median days from submission to resolution",
    )
    min_days_to_resolve: int = Field(
        default=0, description="Minimum resolution time (days)"
    )
    max_days_to_resolve: int = Field(
        default=0, description="Maximum resolution time (days)"
    )
    within_target_acknowledgement_count: int = Field(
        default=0,
        description="Cases acknowledged within target timeframe",
    )
    within_target_acknowledgement_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage acknowledged within target",
    )
    within_target_resolution_count: int = Field(
        default=0,
        description="Cases resolved within target timeframe",
    )
    within_target_resolution_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage resolved within target",
    )
    cases_with_acknowledgement_data: int = Field(
        default=0, description="Cases with acknowledgement time data"
    )
    cases_with_resolution_data: int = Field(
        default=0, description="Cases with resolution time data"
    )

class GrievanceResult(BaseModel):
    """Complete grievance mechanism assessment result per Art 12 CSDDD.

    Aggregates all case data, mechanism assessments, resolution
    statistics, and response time analysis into a single result
    with provenance tracking.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Unique result identifier"
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version used"
    )
    assessed_at: datetime = Field(
        default_factory=utcnow, description="Timestamp of assessment (UTC)"
    )
    entity_name: str = Field(
        default="", description="Entity or undertaking name"
    )
    reporting_year: int = Field(
        default=0, description="Reporting year"
    )
    cases_count: int = Field(
        default=0, description="Total grievance cases assessed"
    )
    mechanism_assessment: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="UNGP criteria assessment results",
    )
    resolution_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolution statistics",
    )
    response_time_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response time statistics",
    )
    accessibility_score: Decimal = Field(
        default=Decimal("0"),
        description="Accessibility score (0-100)",
    )
    effectiveness_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall effectiveness score (0-100)",
    )
    criteria_met_count: int = Field(
        default=0, description="Number of UNGP criteria fully met"
    )
    criteria_total: int = Field(
        default=len(ALL_UNGP_CRITERIA),
        description="Total UNGP criteria evaluated",
    )
    compliance_gaps: List[str] = Field(
        default_factory=list,
        description="Identified compliance gaps under Art 12",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement",
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

class GrievanceMechanismEngine:
    """CSDDD Article 12 grievance mechanism assessment engine.

    Provides deterministic, zero-hallucination assessments for
    corporate grievance mechanisms against UNGP Principle 31
    effectiveness criteria and CSDDD Art 12 requirements.

    Evaluates:
    - UNGP criteria compliance (8 criteria)
    - Resolution statistics (rates, outcomes, satisfaction)
    - Response time performance (acknowledgement, resolution)
    - Accessibility across stakeholder groups and channels
    - Overall mechanism effectiveness

    All calculations use Decimal arithmetic for reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = GrievanceMechanismEngine()
        result = engine.assess_grievance_mechanism(
            cases=[GrievanceCase(...)],
            mechanism_config=MechanismConfig(...),
        )
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Main Assessment Method                                               #
    # ------------------------------------------------------------------ #

    def assess_grievance_mechanism(
        self,
        cases: List[GrievanceCase],
        mechanism_config: MechanismConfig,
        entity_name: str = "",
        reporting_year: int = 0,
    ) -> GrievanceResult:
        """Perform a complete grievance mechanism assessment.

        Orchestrates evaluation of all UNGP criteria, resolution
        statistics, response time analysis, and accessibility
        assessment to produce a consolidated result.

        Args:
            cases: List of GrievanceCase instances to analyse.
            mechanism_config: Configuration of the grievance mechanism.
            entity_name: Name of the reporting entity.
            reporting_year: Reporting year.

        Returns:
            GrievanceResult with complete assessment and provenance.
        """
        t0 = time.perf_counter()

        logger.info(
            "Assessing grievance mechanism: %d cases, entity=%s, year=%d",
            len(cases), entity_name, reporting_year,
        )

        # Step 1: Assess UNGP criteria
        criteria_assessments = self.assess_mechanism_criteria(mechanism_config)

        # Step 2: Calculate resolution statistics
        resolution_stats = self.calculate_resolution_stats(cases)

        # Step 3: Calculate response time statistics
        response_time_stats = self.calculate_response_times(
            cases, mechanism_config
        )

        # Step 4: Assess accessibility
        accessibility_score = self.assess_accessibility(mechanism_config)

        # Step 5: Calculate overall effectiveness score
        effectiveness_score = self._calculate_effectiveness_score(
            criteria_assessments
        )

        # Step 6: Count criteria met
        criteria_met = sum(
            1 for a in criteria_assessments if a["met"]
        )

        # Step 7: Identify compliance gaps
        compliance_gaps = self._identify_compliance_gaps(
            criteria_assessments, resolution_stats, response_time_stats,
            mechanism_config,
        )

        # Step 8: Generate recommendations
        recommendations = self._generate_recommendations(
            criteria_assessments, resolution_stats, response_time_stats,
            accessibility_score, mechanism_config,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = GrievanceResult(
            entity_name=entity_name,
            reporting_year=reporting_year,
            cases_count=len(cases),
            mechanism_assessment=criteria_assessments,
            resolution_stats=resolution_stats,
            response_time_stats=response_time_stats,
            accessibility_score=accessibility_score,
            effectiveness_score=effectiveness_score,
            criteria_met_count=criteria_met,
            criteria_total=len(ALL_UNGP_CRITERIA),
            compliance_gaps=compliance_gaps,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Grievance mechanism assessed: effectiveness=%.1f%%, "
            "accessibility=%.1f%%, criteria_met=%d/%d, cases=%d, "
            "hash=%s",
            float(effectiveness_score), float(accessibility_score),
            criteria_met, len(ALL_UNGP_CRITERIA), len(cases),
            result.provenance_hash[:16],
        )

        return result

    # ------------------------------------------------------------------ #
    # UNGP Criteria Assessment                                             #
    # ------------------------------------------------------------------ #

    def assess_mechanism_criteria(
        self, config: MechanismConfig
    ) -> List[Dict[str, Any]]:
        """Assess the grievance mechanism against all UNGP Principle 31 criteria.

        Evaluates each of the eight effectiveness criteria individually
        and returns detailed assessment results.

        Args:
            config: MechanismConfig describing the mechanism.

        Returns:
            List of dicts, one per UNGP criterion, with score, met,
            gaps, and recommendations.
        """
        assessments: List[Dict[str, Any]] = []

        # 1. Legitimate
        assessments.append(self._assess_legitimate(config))

        # 2. Accessible
        assessments.append(self._assess_accessible(config))

        # 3. Predictable
        assessments.append(self._assess_predictable(config))

        # 4. Equitable
        assessments.append(self._assess_equitable(config))

        # 5. Transparent
        assessments.append(self._assess_transparent(config))

        # 6. Rights-compatible
        assessments.append(self._assess_rights_compatible(config))

        # 7. Continuous Learning
        assessments.append(self._assess_continuous_learning(config))

        # 8. Based on Engagement & Dialogue
        assessments.append(self._assess_engagement_dialogue(config))

        logger.info(
            "UNGP criteria assessed: %d/%d met",
            sum(1 for a in assessments if a["met"]),
            len(assessments),
        )

        return assessments

    def _assess_legitimate(self, config: MechanismConfig) -> Dict[str, Any]:
        """Assess the Legitimate criterion (UNGP Principle 31).

        A mechanism is legitimate if it is trusted by stakeholder
        groups and is accountable for the fair conduct of processes.
        """
        sub_checks = {
            "independent_oversight": config.independent_oversight,
            "written_procedures": config.has_written_procedures,
            "appeal_mechanism": config.has_appeal_mechanism,
            "stakeholder_trust_design": config.stakeholder_input_on_design,
        }
        met_count = sum(1 for v in sub_checks.values() if v)
        total = len(sub_checks)
        score = _round_val(
            _decimal(met_count) / _decimal(total) * Decimal("100"), 1
        )
        is_met = met_count == total

        gaps = [k for k, v in sub_checks.items() if not v]
        recs: List[str] = []
        if not config.independent_oversight:
            recs.append(
                "Establish independent oversight or advisory body for "
                "the grievance mechanism to enhance legitimacy."
            )
        if not config.has_appeal_mechanism:
            recs.append(
                "Implement an appeal mechanism so complainants can "
                "challenge outcomes they consider unfair."
            )
        if not config.stakeholder_input_on_design:
            recs.append(
                "Consult affected stakeholders on mechanism design "
                "to build trust and ensure relevance."
            )
        if not config.has_written_procedures:
            recs.append(
                "Document written procedures for grievance handling "
                "to ensure accountability and consistency."
            )

        return {
            "criteria": MechanismCriteria.LEGITIMATE.value,
            "met": is_met,
            "score": score,
            "sub_checks": sub_checks,
            "gaps": gaps,
            "recommendations": recs,
        }

    def _assess_accessible(self, config: MechanismConfig) -> Dict[str, Any]:
        """Assess the Accessible criterion (UNGP Principle 31).

        A mechanism is accessible if it is known to all stakeholder
        groups for whose use it is intended, and provides adequate
        assistance for those who face particular barriers to access.
        """
        sub_checks = {
            "multiple_channels": len(config.channels_available) >= 2,
            "languages_supported": len(config.languages_supported) >= 1,
            "no_cost": config.no_cost_to_complainant,
            "anonymous_allowed": config.anonymous_submission_allowed,
            "disability_accessible": config.disability_accessible,
            "all_groups_covered": config.available_to_all_groups,
            "publicised": config.publicised_to_stakeholders,
            "geographic_coverage": config.geographically_accessible,
        }
        met_count = sum(1 for v in sub_checks.values() if v)
        total = len(sub_checks)
        score = _round_val(
            _decimal(met_count) / _decimal(total) * Decimal("100"), 1
        )
        is_met = met_count == total

        gaps = [k for k, v in sub_checks.items() if not v]
        recs: List[str] = []
        if len(config.channels_available) < 2:
            recs.append(
                "Provide at least two submission channels (e.g., "
                "hotline, web portal, email) to ensure accessibility."
            )
        if not config.anonymous_submission_allowed:
            recs.append(
                "Enable anonymous submissions to protect complainants "
                "from retaliation."
            )
        if not config.disability_accessible:
            recs.append(
                "Ensure the mechanism is accessible to persons with "
                "disabilities (WCAG 2.1 compliance for digital channels)."
            )
        if not config.publicised_to_stakeholders:
            recs.append(
                "Actively publicise the mechanism to all relevant "
                "stakeholder groups through appropriate channels."
            )
        if not config.geographically_accessible:
            recs.append(
                "Ensure the mechanism is accessible across all "
                "operating geographies, including supply chain locations."
            )
        if len(config.languages_supported) < 1:
            recs.append(
                "Support at least the local languages of key operating "
                "regions and stakeholder groups."
            )
        if not config.available_to_all_groups:
            recs.append(
                "Extend the mechanism to cover all stakeholder groups "
                "identified under Art 12(3) CSDDD."
            )
        if not config.no_cost_to_complainant:
            recs.append(
                "Ensure the mechanism is free of charge for all "
                "complainants as required by Art 12 CSDDD."
            )

        return {
            "criteria": MechanismCriteria.ACCESSIBLE.value,
            "met": is_met,
            "score": score,
            "sub_checks": sub_checks,
            "gaps": gaps,
            "recommendations": recs,
        }

    def _assess_predictable(self, config: MechanismConfig) -> Dict[str, Any]:
        """Assess the Predictable criterion (UNGP Principle 31).

        A mechanism is predictable if it provides a clear and known
        procedure with an indicative timeframe for each stage and
        clarity on the types of outcomes available.
        """
        sub_checks = {
            "written_procedures": config.has_written_procedures,
            "defined_timeframes": config.has_defined_timeframes,
            "escalation_process": config.has_escalation_process,
        }
        met_count = sum(1 for v in sub_checks.values() if v)
        total = len(sub_checks)
        score = _round_val(
            _decimal(met_count) / _decimal(total) * Decimal("100"), 1
        )
        is_met = met_count == total

        gaps = [k for k, v in sub_checks.items() if not v]
        recs: List[str] = []
        if not config.has_written_procedures:
            recs.append(
                "Develop and publish written procedures that describe "
                "the steps from submission to resolution."
            )
        if not config.has_defined_timeframes:
            recs.append(
                "Define and communicate indicative timeframes for "
                "each stage: acknowledgement, investigation, resolution."
            )
        if not config.has_escalation_process:
            recs.append(
                "Establish a clear escalation process for cases that "
                "cannot be resolved at the first level."
            )

        return {
            "criteria": MechanismCriteria.PREDICTABLE.value,
            "met": is_met,
            "score": score,
            "sub_checks": sub_checks,
            "gaps": gaps,
            "recommendations": recs,
        }

    def _assess_equitable(self, config: MechanismConfig) -> Dict[str, Any]:
        """Assess the Equitable criterion (UNGP Principle 31).

        A mechanism is equitable if aggrieved parties have reasonable
        access to sources of information, advice, and expertise
        necessary to engage in a grievance process on fair terms.
        """
        sub_checks = {
            "no_cost": config.no_cost_to_complainant,
            "appeal_mechanism": config.has_appeal_mechanism,
            "independent_oversight": config.independent_oversight,
            "anonymous_allowed": config.anonymous_submission_allowed,
        }
        met_count = sum(1 for v in sub_checks.values() if v)
        total = len(sub_checks)
        score = _round_val(
            _decimal(met_count) / _decimal(total) * Decimal("100"), 1
        )
        is_met = met_count == total

        gaps = [k for k, v in sub_checks.items() if not v]
        recs: List[str] = []
        if not config.no_cost_to_complainant:
            recs.append(
                "Remove any costs or fees that may prevent aggrieved "
                "parties from accessing the mechanism."
            )
        if not config.has_appeal_mechanism:
            recs.append(
                "Establish an appeal process to ensure equitable "
                "treatment of all complainants."
            )
        if not config.independent_oversight:
            recs.append(
                "Consider independent oversight to ensure decisions "
                "are balanced and fair."
            )

        return {
            "criteria": MechanismCriteria.EQUITABLE.value,
            "met": is_met,
            "score": score,
            "sub_checks": sub_checks,
            "gaps": gaps,
            "recommendations": recs,
        }

    def _assess_transparent(self, config: MechanismConfig) -> Dict[str, Any]:
        """Assess the Transparent criterion (UNGP Principle 31).

        A mechanism is transparent if it keeps parties to a grievance
        informed about its progress and provides sufficient information
        about the mechanism's performance.
        """
        sub_checks = {
            "progress_updates": config.provides_progress_updates,
            "outcomes_communicated": config.outcomes_communicated,
            "publicised": config.publicised_to_stakeholders,
        }
        met_count = sum(1 for v in sub_checks.values() if v)
        total = len(sub_checks)
        score = _round_val(
            _decimal(met_count) / _decimal(total) * Decimal("100"), 1
        )
        is_met = met_count == total

        gaps = [k for k, v in sub_checks.items() if not v]
        recs: List[str] = []
        if not config.provides_progress_updates:
            recs.append(
                "Implement regular progress updates to complainants "
                "at each stage of the grievance process."
            )
        if not config.outcomes_communicated:
            recs.append(
                "Ensure outcomes and reasons for decisions are "
                "communicated clearly to complainants per Art 12(4)."
            )
        if not config.publicised_to_stakeholders:
            recs.append(
                "Publish aggregated statistics on mechanism usage "
                "and outcomes to build stakeholder confidence."
            )

        return {
            "criteria": MechanismCriteria.TRANSPARENT.value,
            "met": is_met,
            "score": score,
            "sub_checks": sub_checks,
            "gaps": gaps,
            "recommendations": recs,
        }

    def _assess_rights_compatible(
        self, config: MechanismConfig
    ) -> Dict[str, Any]:
        """Assess the Rights-compatible criterion (UNGP Principle 31).

        A mechanism is rights-compatible if its outcomes and remedies
        accord with internationally recognised human rights.
        """
        sub_checks = {
            "rights_based_outcomes": config.rights_based_outcomes,
            "independent_oversight": config.independent_oversight,
            "no_cost": config.no_cost_to_complainant,
        }
        met_count = sum(1 for v in sub_checks.values() if v)
        total = len(sub_checks)
        score = _round_val(
            _decimal(met_count) / _decimal(total) * Decimal("100"), 1
        )
        is_met = met_count == total

        gaps = [k for k, v in sub_checks.items() if not v]
        recs: List[str] = []
        if not config.rights_based_outcomes:
            recs.append(
                "Ensure all outcomes are assessed for compatibility "
                "with internationally recognised human rights standards."
            )
        if not config.independent_oversight:
            recs.append(
                "Independent oversight helps ensure remedies are "
                "rights-compatible and not influenced by conflicts."
            )

        return {
            "criteria": MechanismCriteria.RIGHTS_COMPATIBLE.value,
            "met": is_met,
            "score": score,
            "sub_checks": sub_checks,
            "gaps": gaps,
            "recommendations": recs,
        }

    def _assess_continuous_learning(
        self, config: MechanismConfig
    ) -> Dict[str, Any]:
        """Assess the Continuous Learning criterion (UNGP Principle 31).

        A mechanism enables continuous learning if it draws on
        relevant measures to identify lessons for improving the
        mechanism and preventing future harms.
        """
        sub_checks = {
            "lessons_learned_process": config.lessons_learned_process,
            "regular_effectiveness_review": config.regular_effectiveness_review,
        }
        met_count = sum(1 for v in sub_checks.values() if v)
        total = len(sub_checks)
        score = _round_val(
            _decimal(met_count) / _decimal(total) * Decimal("100"), 1
        )
        is_met = met_count == total

        gaps = [k for k, v in sub_checks.items() if not v]
        recs: List[str] = []
        if not config.lessons_learned_process:
            recs.append(
                "Establish a systematic process to identify and apply "
                "lessons learned from grievance cases."
            )
        if not config.regular_effectiveness_review:
            recs.append(
                "Conduct regular (at least annual) reviews of the "
                "mechanism's effectiveness with stakeholder input."
            )

        return {
            "criteria": MechanismCriteria.CONTINUOUS_LEARNING.value,
            "met": is_met,
            "score": score,
            "sub_checks": sub_checks,
            "gaps": gaps,
            "recommendations": recs,
        }

    def _assess_engagement_dialogue(
        self, config: MechanismConfig
    ) -> Dict[str, Any]:
        """Assess the Based on Engagement & Dialogue criterion (UNGP Principle 31).

        A mechanism is based on engagement and dialogue if it consults
        the stakeholder groups for whose use it is intended on its
        design and performance.
        """
        sub_checks = {
            "stakeholder_input_on_design": config.stakeholder_input_on_design,
            "regular_effectiveness_review": config.regular_effectiveness_review,
            "available_to_all_groups": config.available_to_all_groups,
        }
        met_count = sum(1 for v in sub_checks.values() if v)
        total = len(sub_checks)
        score = _round_val(
            _decimal(met_count) / _decimal(total) * Decimal("100"), 1
        )
        is_met = met_count == total

        gaps = [k for k, v in sub_checks.items() if not v]
        recs: List[str] = []
        if not config.stakeholder_input_on_design:
            recs.append(
                "Consult affected stakeholder groups on the design "
                "and ongoing improvement of the grievance mechanism."
            )
        if not config.regular_effectiveness_review:
            recs.append(
                "Include stakeholder representatives in regular "
                "mechanism effectiveness reviews."
            )
        if not config.available_to_all_groups:
            recs.append(
                "Ensure all relevant stakeholder groups identified "
                "under CSDDD can participate in the mechanism."
            )

        return {
            "criteria": MechanismCriteria.BASED_ON_ENGAGEMENT_DIALOGUE.value,
            "met": is_met,
            "score": score,
            "sub_checks": sub_checks,
            "gaps": gaps,
            "recommendations": recs,
        }

    # ------------------------------------------------------------------ #
    # Resolution Statistics                                                #
    # ------------------------------------------------------------------ #

    def calculate_resolution_stats(
        self, cases: List[GrievanceCase]
    ) -> Dict[str, Any]:
        """Calculate resolution statistics from grievance cases.

        Computes resolution rates, satisfaction rates, and breakdowns
        by stakeholder group, channel, category, and status.

        Args:
            cases: List of GrievanceCase instances.

        Returns:
            Dict with resolution statistics and provenance hash.
        """
        if not cases:
            logger.info("No grievance cases to analyse for resolution stats")
            empty_stats = ResolutionStats()
            result = empty_stats.model_dump(mode="json")
            result["provenance_hash"] = _compute_hash(result)
            return result

        total = len(cases)
        resolved = sum(
            1 for c in cases
            if c.status in (GrievanceStatus.RESOLVED, GrievanceStatus.CLOSED)
        )
        closed = sum(
            1 for c in cases if c.status == GrievanceStatus.CLOSED
        )
        escalated = sum(
            1 for c in cases if c.status == GrievanceStatus.ESCALATED
        )
        pending = sum(
            1 for c in cases
            if c.status in (
                GrievanceStatus.RECEIVED,
                GrievanceStatus.UNDER_REVIEW,
                GrievanceStatus.INVESTIGATING,
            )
        )

        # Satisfaction (only for resolved/closed cases with feedback)
        satisfaction_cases = [
            c for c in cases
            if c.complainant_satisfied is not None
            and c.status in (GrievanceStatus.RESOLVED, GrievanceStatus.CLOSED)
        ]
        satisfied_count = sum(
            1 for c in satisfaction_cases if c.complainant_satisfied
        )

        # By stakeholder group
        by_group: Dict[str, int] = {}
        for group in StakeholderGroup:
            count = sum(
                1 for c in cases if c.stakeholder_group == group
            )
            if count > 0:
                by_group[group.value] = count

        # By channel
        by_channel: Dict[str, int] = {}
        for channel in GrievanceChannel:
            count = sum(
                1 for c in cases if c.channel == channel
            )
            if count > 0:
                by_channel[channel.value] = count

        # By category
        by_category: Dict[str, int] = {}
        for case in cases:
            if case.category:
                by_category[case.category] = (
                    by_category.get(case.category, 0) + 1
                )

        # By status
        by_status: Dict[str, int] = {}
        for status in GrievanceStatus:
            count = sum(1 for c in cases if c.status == status)
            if count > 0:
                by_status[status.value] = count

        # Anonymous count
        anonymous = sum(1 for c in cases if c.is_anonymous)

        result = {
            "total_cases": total,
            "resolved_count": resolved,
            "closed_count": closed,
            "escalated_count": escalated,
            "pending_count": pending,
            "resolution_rate_pct": _pct(resolved, total),
            "satisfaction_count": satisfied_count,
            "satisfaction_respondents": len(satisfaction_cases),
            "satisfaction_rate_pct": _pct(
                satisfied_count, len(satisfaction_cases)
            ),
            "by_stakeholder_group": by_group,
            "by_channel": by_channel,
            "by_category": by_category,
            "by_status": by_status,
            "anonymous_count": anonymous,
            "anonymous_pct": _pct(anonymous, total),
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Resolution stats: %d total, %d resolved (%.1f%%), "
            "%d escalated, %d pending",
            total, resolved, float(_pct(resolved, total)),
            escalated, pending,
        )

        return result

    # ------------------------------------------------------------------ #
    # Response Time Statistics                                             #
    # ------------------------------------------------------------------ #

    def calculate_response_times(
        self,
        cases: List[GrievanceCase],
        config: Optional[MechanismConfig] = None,
    ) -> Dict[str, Any]:
        """Calculate response time statistics from grievance cases.

        Analyses acknowledgement and resolution times against
        target timeframes defined in the mechanism configuration.

        Args:
            cases: List of GrievanceCase instances.
            config: Optional MechanismConfig for target timeframes.

        Returns:
            Dict with response time statistics and provenance hash.
        """
        target_ack_days = (
            config.target_acknowledgement_days
            if config else RESPONSE_TARGET_ACKNOWLEDGEMENT_DAYS
        )
        target_res_days = (
            config.target_resolution_days
            if config else RESPONSE_TARGET_RESOLUTION_DAYS
        )

        if not cases:
            logger.info("No grievance cases for response time analysis")
            result: Dict[str, Any] = {
                "average_days_to_acknowledge": Decimal("0"),
                "median_days_to_acknowledge": Decimal("0"),
                "average_days_to_resolve": Decimal("0"),
                "median_days_to_resolve": Decimal("0"),
                "min_days_to_resolve": 0,
                "max_days_to_resolve": 0,
                "within_target_acknowledgement_count": 0,
                "within_target_acknowledgement_pct": Decimal("0.0"),
                "within_target_resolution_count": 0,
                "within_target_resolution_pct": Decimal("0.0"),
                "cases_with_acknowledgement_data": 0,
                "cases_with_resolution_data": 0,
                "target_acknowledgement_days": target_ack_days,
                "target_resolution_days": target_res_days,
                "provenance_hash": _compute_hash({"empty": True}),
            }
            return result

        # Acknowledgement times
        ack_days_list: List[int] = []
        for case in cases:
            if case.days_to_acknowledge is not None:
                ack_days_list.append(case.days_to_acknowledge)
            elif case.acknowledged_date is not None:
                delta = (
                    case.acknowledged_date - case.submitted_date
                ).days
                ack_days_list.append(max(0, delta))

        # Resolution times
        res_days_list: List[int] = []
        for case in cases:
            if case.days_to_resolve is not None:
                res_days_list.append(case.days_to_resolve)
            elif case.resolved_date is not None:
                delta = (case.resolved_date - case.submitted_date).days
                res_days_list.append(max(0, delta))

        # Acknowledgement statistics
        avg_ack = Decimal("0")
        median_ack = Decimal("0")
        within_ack = 0
        if ack_days_list:
            avg_ack = _round_val(
                _decimal(sum(ack_days_list))
                / _decimal(len(ack_days_list)),
                1,
            )
            sorted_ack = sorted(ack_days_list)
            mid = len(sorted_ack) // 2
            if len(sorted_ack) % 2 == 0 and len(sorted_ack) >= 2:
                median_ack = _round_val(
                    (_decimal(sorted_ack[mid - 1]) + _decimal(sorted_ack[mid]))
                    / Decimal("2"),
                    1,
                )
            else:
                median_ack = _decimal(sorted_ack[mid])
            within_ack = sum(
                1 for d in ack_days_list if d <= target_ack_days
            )

        # Resolution statistics
        avg_res = Decimal("0")
        median_res = Decimal("0")
        min_res = 0
        max_res = 0
        within_res = 0
        if res_days_list:
            avg_res = _round_val(
                _decimal(sum(res_days_list))
                / _decimal(len(res_days_list)),
                1,
            )
            sorted_res = sorted(res_days_list)
            mid = len(sorted_res) // 2
            if len(sorted_res) % 2 == 0 and len(sorted_res) >= 2:
                median_res = _round_val(
                    (_decimal(sorted_res[mid - 1]) + _decimal(sorted_res[mid]))
                    / Decimal("2"),
                    1,
                )
            else:
                median_res = _decimal(sorted_res[mid])
            min_res = sorted_res[0]
            max_res = sorted_res[-1]
            within_res = sum(
                1 for d in res_days_list if d <= target_res_days
            )

        result = {
            "average_days_to_acknowledge": avg_ack,
            "median_days_to_acknowledge": median_ack,
            "average_days_to_resolve": avg_res,
            "median_days_to_resolve": median_res,
            "min_days_to_resolve": min_res,
            "max_days_to_resolve": max_res,
            "within_target_acknowledgement_count": within_ack,
            "within_target_acknowledgement_pct": _pct(
                within_ack, len(ack_days_list)
            ),
            "within_target_resolution_count": within_res,
            "within_target_resolution_pct": _pct(
                within_res, len(res_days_list)
            ),
            "cases_with_acknowledgement_data": len(ack_days_list),
            "cases_with_resolution_data": len(res_days_list),
            "target_acknowledgement_days": target_ack_days,
            "target_resolution_days": target_res_days,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Response times: avg ack=%.1f days, avg resolve=%.1f days, "
            "within ack target=%.1f%%, within res target=%.1f%%",
            float(avg_ack), float(avg_res),
            float(_pct(within_ack, len(ack_days_list))),
            float(_pct(within_res, len(res_days_list))),
        )

        return result

    # ------------------------------------------------------------------ #
    # Accessibility Assessment                                             #
    # ------------------------------------------------------------------ #

    def assess_accessibility(
        self, config: MechanismConfig
    ) -> Decimal:
        """Assess overall accessibility of the grievance mechanism.

        Evaluates the mechanism against the accessibility sub-criteria
        and returns a score from 0 to 100.

        Args:
            config: MechanismConfig describing the mechanism.

        Returns:
            Accessibility score as Decimal (0-100).
        """
        checks = {
            "multiple_channels_available": (
                len(config.channels_available) >= 2
            ),
            "languages_supported": len(config.languages_supported) >= 1,
            "no_cost_to_complainant": config.no_cost_to_complainant,
            "anonymous_submission_allowed": (
                config.anonymous_submission_allowed
            ),
            "disability_accessible": config.disability_accessible,
            "available_to_all_stakeholder_groups": (
                config.available_to_all_groups
            ),
            "publicised_to_stakeholders": config.publicised_to_stakeholders,
            "geographically_accessible": config.geographically_accessible,
        }

        met_count = sum(1 for v in checks.values() if v)
        total = len(checks)
        score = _round_val(
            _decimal(met_count) / _decimal(total) * Decimal("100"), 1
        )

        logger.info(
            "Accessibility score: %.1f%% (%d/%d sub-criteria met)",
            float(score), met_count, total,
        )

        return score

    # ------------------------------------------------------------------ #
    # Effectiveness Score Calculation                                       #
    # ------------------------------------------------------------------ #

    def _calculate_effectiveness_score(
        self, criteria_assessments: List[Dict[str, Any]]
    ) -> Decimal:
        """Calculate the overall effectiveness score from criteria assessments.

        Uses weighted average of individual criteria scores based on
        CRITERIA_WEIGHTS.

        Args:
            criteria_assessments: List of criteria assessment dicts.

        Returns:
            Overall effectiveness score as Decimal (0-100).
        """
        if not criteria_assessments:
            return Decimal("0")

        weighted_sum = Decimal("0")
        total_weight = Decimal("0")

        for assessment in criteria_assessments:
            criteria_key = assessment.get("criteria", "")
            weight = CRITERIA_WEIGHTS.get(criteria_key, Decimal("0.125"))
            score = _decimal(assessment.get("score", 0))
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == Decimal("0"):
            return Decimal("0")

        effectiveness = _round_val(
            weighted_sum / total_weight, 1
        )

        return effectiveness

    # ------------------------------------------------------------------ #
    # Compliance Gap Identification                                        #
    # ------------------------------------------------------------------ #

    def _identify_compliance_gaps(
        self,
        criteria_assessments: List[Dict[str, Any]],
        resolution_stats: Dict[str, Any],
        response_time_stats: Dict[str, Any],
        config: MechanismConfig,
    ) -> List[str]:
        """Identify compliance gaps under CSDDD Art 12.

        Args:
            criteria_assessments: UNGP criteria assessment results.
            resolution_stats: Resolution statistics.
            response_time_stats: Response time statistics.
            config: Mechanism configuration.

        Returns:
            List of identified compliance gap descriptions.
        """
        gaps: List[str] = []

        # Check for unmet UNGP criteria
        unmet_criteria = [
            a["criteria"] for a in criteria_assessments if not a["met"]
        ]
        if unmet_criteria:
            gaps.append(
                f"UNGP Principle 31 criteria not fully met: "
                f"{', '.join(unmet_criteria)}"
            )

        # Check resolution rate
        res_rate = resolution_stats.get(
            "resolution_rate_pct", Decimal("0")
        )
        if isinstance(res_rate, (int, float, str)):
            res_rate = _decimal(res_rate)
        if res_rate < Decimal("50") and resolution_stats.get(
            "total_cases", 0
        ) > 0:
            gaps.append(
                f"Resolution rate is {res_rate}%, below the 50% "
                f"minimum expected threshold."
            )

        # Check response times against targets
        avg_ack = response_time_stats.get(
            "average_days_to_acknowledge", Decimal("0")
        )
        if isinstance(avg_ack, (int, float, str)):
            avg_ack = _decimal(avg_ack)
        if avg_ack > _decimal(config.target_acknowledgement_days):
            gaps.append(
                f"Average acknowledgement time ({avg_ack} days) exceeds "
                f"target of {config.target_acknowledgement_days} days."
            )

        avg_res = response_time_stats.get(
            "average_days_to_resolve", Decimal("0")
        )
        if isinstance(avg_res, (int, float, str)):
            avg_res = _decimal(avg_res)
        if avg_res > _decimal(config.target_resolution_days):
            gaps.append(
                f"Average resolution time ({avg_res} days) exceeds "
                f"target of {config.target_resolution_days} days."
            )

        # Check channel coverage
        if len(config.channels_available) < 2:
            gaps.append(
                "Fewer than 2 submission channels available, which "
                "may limit accessibility for stakeholders."
            )

        # Check if anonymous submissions are possible
        if not config.anonymous_submission_allowed:
            gaps.append(
                "Anonymous submissions are not allowed, which may "
                "discourage reporting due to fear of retaliation."
            )

        # Check if outcomes are communicated per Art 12(4)
        if not config.outcomes_communicated:
            gaps.append(
                "Art 12(4) requires companies to inform complainants "
                "of the outcome; this is not currently configured."
            )

        return gaps

    # ------------------------------------------------------------------ #
    # Recommendations Generation                                           #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        criteria_assessments: List[Dict[str, Any]],
        resolution_stats: Dict[str, Any],
        response_time_stats: Dict[str, Any],
        accessibility_score: Decimal,
        config: MechanismConfig,
    ) -> List[str]:
        """Generate recommendations for grievance mechanism improvement.

        Args:
            criteria_assessments: UNGP criteria assessment results.
            resolution_stats: Resolution statistics.
            response_time_stats: Response time statistics.
            accessibility_score: Accessibility score.
            config: Mechanism configuration.

        Returns:
            List of actionable recommendation strings.
        """
        recommendations: List[str] = []

        # Collect recommendations from criteria assessments
        for assessment in criteria_assessments:
            if not assessment.get("met", False):
                for rec in assessment.get("recommendations", []):
                    if rec not in recommendations:
                        recommendations.append(rec)

        # Accessibility recommendations
        if accessibility_score < Decimal("75"):
            recommendations.append(
                "Accessibility score is below 75%. Prioritise improving "
                "channel availability, language support, and geographic "
                "coverage to meet Art 12 requirements."
            )

        # Resolution rate recommendations
        total_cases = resolution_stats.get("total_cases", 0)
        if total_cases > 0:
            res_rate = resolution_stats.get(
                "resolution_rate_pct", Decimal("0")
            )
            if isinstance(res_rate, (int, float, str)):
                res_rate = _decimal(res_rate)
            if res_rate < Decimal("70"):
                recommendations.append(
                    "Resolution rate is below 70%. Review case handling "
                    "capacity and procedures to improve outcomes."
                )

        # Escalation recommendations
        escalated = resolution_stats.get("escalated_count", 0)
        if total_cases > 0 and escalated > 0:
            escalation_rate = _pct(escalated, total_cases)
            if escalation_rate > Decimal("30"):
                recommendations.append(
                    f"Escalation rate is {escalation_rate}%. Investigate "
                    f"root causes to determine if first-level resolution "
                    f"procedures need strengthening."
                )

        # Response time recommendations
        within_ack_pct = response_time_stats.get(
            "within_target_acknowledgement_pct", Decimal("0")
        )
        if isinstance(within_ack_pct, (int, float, str)):
            within_ack_pct = _decimal(within_ack_pct)
        if (
            within_ack_pct < Decimal("90")
            and response_time_stats.get("cases_with_acknowledgement_data", 0)
            > 0
        ):
            recommendations.append(
                "Less than 90% of cases are acknowledged within the "
                "target timeframe. Consider automating acknowledgement "
                "notifications to improve response times."
            )

        # Satisfaction recommendations
        satisfaction_rate = resolution_stats.get(
            "satisfaction_rate_pct", Decimal("0")
        )
        if isinstance(satisfaction_rate, (int, float, str)):
            satisfaction_rate = _decimal(satisfaction_rate)
        satisfaction_respondents = resolution_stats.get(
            "satisfaction_respondents", 0
        )
        if (
            satisfaction_rate < Decimal("60")
            and satisfaction_respondents > 0
        ):
            recommendations.append(
                "Complainant satisfaction is below 60%. Review remedy "
                "quality and communication with affected stakeholders."
            )

        # Cap at reasonable number of recommendations
        if len(recommendations) > 15:
            recommendations = recommendations[:15]

        return recommendations

    # ------------------------------------------------------------------ #
    # Stakeholder Group Coverage Analysis                                  #
    # ------------------------------------------------------------------ #

    def analyse_stakeholder_coverage(
        self, cases: List[GrievanceCase]
    ) -> Dict[str, Any]:
        """Analyse which stakeholder groups are using the mechanism.

        Identifies which groups have submitted grievances and which
        groups may be underserved or unaware of the mechanism.

        Args:
            cases: List of GrievanceCase instances.

        Returns:
            Dict with coverage analysis by stakeholder group.
        """
        all_groups = set(g.value for g in StakeholderGroup)
        groups_with_cases: Dict[str, int] = {}

        for case in cases:
            gv = case.stakeholder_group.value
            groups_with_cases[gv] = groups_with_cases.get(gv, 0) + 1

        covered_groups = set(groups_with_cases.keys())
        uncovered_groups = all_groups - covered_groups

        coverage_pct = _pct(len(covered_groups), len(all_groups))

        result = {
            "total_stakeholder_groups": len(all_groups),
            "groups_with_cases": len(covered_groups),
            "coverage_pct": coverage_pct,
            "cases_by_group": groups_with_cases,
            "groups_not_represented": sorted(uncovered_groups),
            "groups_represented": sorted(covered_groups),
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Stakeholder coverage: %d/%d groups represented (%.1f%%)",
            len(covered_groups), len(all_groups), float(coverage_pct),
        )

        return result

    # ------------------------------------------------------------------ #
    # Trend Analysis (Year-over-Year)                                      #
    # ------------------------------------------------------------------ #

    def compare_periods(
        self,
        current: GrievanceResult,
        previous: GrievanceResult,
    ) -> Dict[str, Any]:
        """Compare grievance mechanism performance across two periods.

        Tracks changes in case volumes, resolution rates, response
        times, and effectiveness scores.

        Args:
            current: Current period result.
            previous: Previous period result.

        Returns:
            Dict with period-over-period changes and provenance.
        """
        cases_change = current.cases_count - previous.cases_count

        current_res_rate = current.resolution_stats.get(
            "resolution_rate_pct", Decimal("0")
        )
        previous_res_rate = previous.resolution_stats.get(
            "resolution_rate_pct", Decimal("0")
        )
        if isinstance(current_res_rate, (int, float, str)):
            current_res_rate = _decimal(current_res_rate)
        if isinstance(previous_res_rate, (int, float, str)):
            previous_res_rate = _decimal(previous_res_rate)

        comparison = {
            "current_period": current.reporting_year,
            "previous_period": previous.reporting_year,
            "cases_change": cases_change,
            "cases_change_pct": (
                _pct_dec(
                    _decimal(abs(cases_change)),
                    _decimal(max(previous.cases_count, 1)),
                )
                if cases_change != 0 else Decimal("0.0")
            ),
            "effectiveness_change_pp": _round_val(
                current.effectiveness_score - previous.effectiveness_score, 1
            ),
            "accessibility_change_pp": _round_val(
                current.accessibility_score - previous.accessibility_score, 1
            ),
            "resolution_rate_change_pp": _round_val(
                current_res_rate - previous_res_rate, 1
            ),
            "criteria_met_change": (
                current.criteria_met_count - previous.criteria_met_count
            ),
            "direction": (
                "improving"
                if current.effectiveness_score > previous.effectiveness_score
                else (
                    "stable"
                    if current.effectiveness_score == previous.effectiveness_score
                    else "declining"
                )
            ),
        }

        comparison["provenance_hash"] = _compute_hash(comparison)

        logger.info(
            "Period comparison: cases %+d, effectiveness %+.1fpp, "
            "direction=%s",
            cases_change,
            float(
                current.effectiveness_score - previous.effectiveness_score
            ),
            comparison["direction"],
        )

        return comparison
