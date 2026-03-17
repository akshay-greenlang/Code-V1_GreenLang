# -*- coding: utf-8 -*-
"""
StakeholderEngagementEngine - PACK-015 Double Materiality Engine 3
====================================================================

Manages stakeholder identification and consultation per ESRS 1 Para
22-23 and the broader double materiality assessment process.

Under ESRS, the double materiality assessment must be informed by
dialogue with affected stakeholders and users of sustainability
statements.  This engine supports the identification, prioritisation,
engagement tracking, and synthesis of stakeholder consultations to
ensure the materiality assessment is grounded in stakeholder
perspectives.

ESRS 1 Stakeholder Engagement Framework:
    - Para 22: The undertaking shall consider the perspectives of its
      stakeholders in its materiality assessment, including affected
      stakeholders and users of sustainability statements.
    - Para 23: Engagement with affected stakeholders is essential to
      identifying and assessing actual and potential negative impacts
      on people and the environment.
    - AR 6: Affected stakeholders include those whose interests are
      affected or could be affected by the undertaking's activities.
    - AR 7: Users of sustainability statements include existing and
      potential investors, lenders, and other creditors.

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS 1 General Requirements, Para 22-23, AR 6-7
    - EFRAG IG 1 Materiality Assessment Implementation Guidance
    - UN Guiding Principles on Business and Human Rights (stakeholder
      engagement expectations)
    - AA1000 Stakeholder Engagement Standard (methodology reference)

Zero-Hallucination:
    - Stakeholder priority mapping uses deterministic grid calculation
    - Coverage percentage is simple division of engaged / total
    - Quality scoring uses weighted criteria with fixed weights
    - Topic frequency is a deterministic count
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-015 Double Materiality
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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator


def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely, returning 0.0 on zero denominator."""
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0


def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.
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


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StakeholderCategory(str, Enum):
    """Stakeholder category per ESRS 1 AR 6 and AR 7.

    Affected stakeholders (AR 6) include those whose interests
    are affected or could be affected.  Users of sustainability
    statements (AR 7) include investors and creditors.
    """
    EMPLOYEES = "employees"
    INVESTORS = "investors"
    CUSTOMERS = "customers"
    SUPPLIERS = "suppliers"
    LOCAL_COMMUNITIES = "local_communities"
    REGULATORS = "regulators"
    NGOS = "ngos"
    INDUSTRY_BODIES = "industry_bodies"
    ACADEMIC_EXPERTS = "academic_experts"
    TRADE_UNIONS = "trade_unions"


class EngagementMethod(str, Enum):
    """Method of stakeholder engagement.

    Methods range from passive (surveys, written submissions) to
    active (workshops, advisory panels).  The quality of engagement
    depends on both the method and the depth of dialogue.
    """
    SURVEY = "survey"
    INTERVIEW = "interview"
    FOCUS_GROUP = "focus_group"
    WORKSHOP = "workshop"
    PUBLIC_CONSULTATION = "public_consultation"
    WRITTEN_SUBMISSION = "written_submission"
    ONGOING_DIALOGUE = "ongoing_dialogue"
    ADVISORY_PANEL = "advisory_panel"


class ConsultationStatus(str, Enum):
    """Status of stakeholder consultation process."""
    NOT_STARTED = "not_started"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VALIDATED = "validated"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Engagement method quality scores (0-1 scale).
# Higher scores indicate deeper, more meaningful engagement.
# Based on AA1000 Stakeholder Engagement Standard levels of engagement.
ENGAGEMENT_METHOD_QUALITY: Dict[str, Decimal] = {
    "survey": Decimal("0.40"),
    "interview": Decimal("0.70"),
    "focus_group": Decimal("0.75"),
    "workshop": Decimal("0.85"),
    "public_consultation": Decimal("0.60"),
    "written_submission": Decimal("0.35"),
    "ongoing_dialogue": Decimal("0.90"),
    "advisory_panel": Decimal("0.95"),
}

# Engagement quality criteria and their weights.
# Used to calculate a composite engagement quality score.
ENGAGEMENT_QUALITY_CRITERIA: Dict[str, Decimal] = {
    "method_quality": Decimal("0.25"),
    "stakeholder_diversity": Decimal("0.20"),
    "topic_coverage": Decimal("0.20"),
    "attendance_depth": Decimal("0.15"),
    "documentation_completeness": Decimal("0.10"),
    "follow_up_actions": Decimal("0.10"),
}

# Sector-to-stakeholder mapping: which stakeholder categories are
# typically most relevant for different NACE sectors.
# This is a heuristic for initial identification, not a binding rule.
SECTOR_STAKEHOLDER_MAP: Dict[str, List[str]] = {
    "agriculture": [
        "employees", "local_communities", "customers", "suppliers",
        "regulators", "ngos", "trade_unions", "academic_experts",
    ],
    "mining": [
        "employees", "local_communities", "regulators", "ngos",
        "investors", "trade_unions", "academic_experts",
    ],
    "manufacturing": [
        "employees", "customers", "suppliers", "local_communities",
        "regulators", "investors", "trade_unions", "industry_bodies",
    ],
    "energy": [
        "employees", "local_communities", "regulators", "investors",
        "customers", "ngos", "academic_experts", "industry_bodies",
    ],
    "construction": [
        "employees", "local_communities", "suppliers", "regulators",
        "investors", "customers", "trade_unions",
    ],
    "retail": [
        "employees", "customers", "suppliers", "investors",
        "regulators", "local_communities", "trade_unions",
    ],
    "transport": [
        "employees", "customers", "regulators", "local_communities",
        "investors", "ngos", "industry_bodies", "trade_unions",
    ],
    "financial_services": [
        "investors", "customers", "employees", "regulators",
        "ngos", "industry_bodies", "academic_experts",
    ],
    "technology": [
        "employees", "customers", "investors", "regulators",
        "academic_experts", "ngos", "industry_bodies",
    ],
    "healthcare": [
        "employees", "customers", "regulators", "local_communities",
        "investors", "ngos", "academic_experts", "trade_unions",
    ],
    "real_estate": [
        "investors", "local_communities", "employees", "customers",
        "regulators", "ngos", "suppliers",
    ],
    "food_beverage": [
        "employees", "customers", "suppliers", "local_communities",
        "regulators", "ngos", "trade_unions", "academic_experts",
    ],
    "chemicals": [
        "employees", "local_communities", "regulators", "ngos",
        "customers", "investors", "academic_experts", "trade_unions",
    ],
    "textiles": [
        "employees", "suppliers", "customers", "ngos",
        "regulators", "investors", "trade_unions", "local_communities",
    ],
    "default": [
        "employees", "investors", "customers", "suppliers",
        "local_communities", "regulators", "ngos",
    ],
}

# Priority matrix labels based on influence x impact grid.
PRIORITY_MATRIX_LABELS: Dict[str, str] = {
    "high_high": "Key Stakeholder: High influence, high impact - prioritise deep engagement",
    "high_low": "Keep Informed: High influence, low impact - regular updates",
    "low_high": "Keep Satisfied: Low influence, high impact - targeted consultation",
    "low_low": "Monitor: Low influence, low impact - periodic check-in",
}

# Influence and impact threshold for high/low classification.
PRIORITY_THRESHOLD: int = 3


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class Stakeholder(BaseModel):
    """A stakeholder identified for the materiality assessment process.

    Represents an individual stakeholder or stakeholder group that
    the undertaking engages with as part of its ESRS double
    materiality assessment.
    """
    id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this stakeholder",
    )
    name: str = Field(
        ...,
        description="Name of the stakeholder or stakeholder group",
        min_length=1,
        max_length=500,
    )
    category: StakeholderCategory = Field(
        ...,
        description="Stakeholder category per ESRS 1 AR 6-7",
    )
    influence_level: int = Field(
        ...,
        description="Level of influence over the undertaking (1-5)",
        ge=1,
        le=5,
    )
    impact_level: int = Field(
        ...,
        description="Level of impact the undertaking has on this stakeholder (1-5)",
        ge=1,
        le=5,
    )
    engagement_method: EngagementMethod = Field(
        default=EngagementMethod.SURVEY,
        description="Primary method of engagement for this stakeholder",
    )
    consultation_status: ConsultationStatus = Field(
        default=ConsultationStatus.NOT_STARTED,
        description="Current status of consultation with this stakeholder",
    )
    is_affected_stakeholder: bool = Field(
        default=True,
        description="True if this is an affected stakeholder per ESRS 1 AR 6",
    )
    is_user_of_statements: bool = Field(
        default=False,
        description="True if this is a user of sustainability statements per AR 7",
    )
    topics_of_interest: List[str] = Field(
        default_factory=list,
        description="ESRS topics this stakeholder is most concerned with",
    )
    contact_info: str = Field(
        default="",
        description="Contact information or reference (not exposed in results)",
        max_length=1000,
    )

    @field_validator("influence_level", "impact_level")
    @classmethod
    def validate_level_range(cls, v: int) -> int:
        """Validate level is within 1-5 range."""
        if v < 1 or v > 5:
            raise ValueError(f"Level must be between 1 and 5, got {v}")
        return v


class ConsultationRecord(BaseModel):
    """Record of a stakeholder consultation event.

    Documents a specific engagement activity including method, date,
    topics discussed, and key findings.  These records form the
    evidence base for the materiality assessment.
    """
    record_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this consultation record",
    )
    stakeholder_id: str = Field(
        ...,
        description="ID of the stakeholder consulted",
        min_length=1,
    )
    date: str = Field(
        ...,
        description="Date of consultation (ISO 8601, e.g. '2025-06-15')",
        min_length=8,
    )
    method: EngagementMethod = Field(
        ...,
        description="Engagement method used for this consultation",
    )
    topics_discussed: List[str] = Field(
        default_factory=list,
        description="Sustainability topics discussed during consultation",
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings or insights from the consultation",
    )
    attendance_count: int = Field(
        default=1,
        description="Number of participants or respondents",
        ge=0,
    )
    duration_minutes: int = Field(
        default=60,
        description="Duration of the consultation in minutes",
        ge=0,
    )
    documentation_available: bool = Field(
        default=True,
        description="Whether documentation (minutes, recordings) is available",
    )
    follow_up_actions: List[str] = Field(
        default_factory=list,
        description="Follow-up actions agreed during consultation",
    )
    quality_notes: str = Field(
        default="",
        description="Notes on the quality of the engagement",
        max_length=5000,
    )


class StakeholderEngagementResult(BaseModel):
    """Result of stakeholder engagement analysis.

    Summarises the overall stakeholder engagement process including
    coverage, quality, topic frequency, priority matrix, and
    recommendations for improving engagement.
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
        default_factory=_utcnow,
        description="Timestamp of calculation (UTC)",
    )

    # --- Coverage Metrics ---
    total_stakeholders: int = Field(
        default=0,
        description="Total number of stakeholders identified",
    )
    engaged_count: int = Field(
        default=0,
        description="Number of stakeholders with completed or in-progress consultation",
    )
    coverage_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Percentage of stakeholders engaged",
    )
    category_coverage: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of stakeholders by category",
    )
    categories_represented: int = Field(
        default=0,
        description="Number of distinct stakeholder categories represented",
    )
    total_categories: int = Field(
        default=len(StakeholderCategory),
        description="Total possible stakeholder categories",
    )

    # --- Consultation Summary ---
    total_consultations: int = Field(
        default=0,
        description="Total number of consultation records",
    )
    total_participants: int = Field(
        default=0,
        description="Total number of participants across all consultations",
    )
    methods_used: List[str] = Field(
        default_factory=list,
        description="Distinct engagement methods used",
    )

    # --- Topic Analysis ---
    topic_frequency: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of times each topic was discussed across consultations",
    )
    top_topics: List[str] = Field(
        default_factory=list,
        description="Most frequently discussed topics (sorted by frequency)",
    )
    topics_not_covered: List[str] = Field(
        default_factory=list,
        description="ESRS topics not discussed in any consultation",
    )

    # --- Priority Matrix ---
    priority_matrix: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Stakeholder names grouped by priority quadrant "
                    "(high_high, high_low, low_high, low_low)",
    )

    # --- Quality ---
    engagement_quality_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Composite engagement quality score (0-1 scale)",
    )
    engagement_quality_grade: str = Field(
        default="",
        description="Quality grade (A/B/C/D/F)",
    )
    avg_method_quality: Decimal = Field(
        default=Decimal("0.000"),
        description="Average quality score of engagement methods used",
    )
    avg_consultation_duration_minutes: float = Field(
        default=0.0,
        description="Average duration of consultations in minutes",
    )
    documentation_rate_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Percentage of consultations with documentation",
    )

    # --- Findings Summary ---
    total_findings: int = Field(
        default=0,
        description="Total number of key findings recorded",
    )
    total_follow_up_actions: int = Field(
        default=0,
        description="Total number of follow-up actions identified",
    )

    # --- Recommendations ---
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving stakeholder engagement",
    )

    # --- Provenance ---
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


class StakeholderEngagementEngine:
    """Stakeholder engagement management engine per ESRS 1 Para 22-23.

    Provides deterministic, zero-hallucination calculations for:
    - Stakeholder identification based on sector mapping
    - Priority matrix mapping (influence vs impact)
    - Consultation record tracking
    - Engagement coverage calculation
    - Engagement quality scoring
    - Topic frequency analysis
    - Synthesis of findings across consultations

    All calculations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Usage::

        engine = StakeholderEngagementEngine()

        stakeholders = [
            Stakeholder(
                name="Works Council",
                category=StakeholderCategory.EMPLOYEES,
                influence_level=4,
                impact_level=5,
                engagement_method=EngagementMethod.WORKSHOP,
            ),
        ]

        records = [
            ConsultationRecord(
                stakeholder_id=stakeholders[0].id,
                date="2025-06-15",
                method=EngagementMethod.WORKSHOP,
                topics_discussed=["s1_own_workforce", "e1_climate"],
                key_findings=["Employees prioritise health and safety"],
                attendance_count=25,
            ),
        ]

        result = engine.synthesize_findings(stakeholders, records)
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Stakeholder Identification                                           #
    # ------------------------------------------------------------------ #

    def identify_stakeholders(
        self,
        sector: str,
        value_chain_stages: Optional[List[str]] = None,
    ) -> List[str]:
        """Identify relevant stakeholder categories for a sector.

        Returns the list of stakeholder categories typically relevant
        for the given sector based on SECTOR_STAKEHOLDER_MAP.  This
        is a starting point for identification; the undertaking must
        refine based on its specific context.

        Args:
            sector: NACE sector key (e.g., "manufacturing", "retail").
                Falls back to "default" if sector not found.
            value_chain_stages: Optional list of value chain stages
                to emphasise.  Not used in current implementation but
                reserved for future weighting.

        Returns:
            List of stakeholder category strings relevant for the sector.
        """
        sector_key = sector.lower().replace(" ", "_")
        categories = SECTOR_STAKEHOLDER_MAP.get(
            sector_key,
            SECTOR_STAKEHOLDER_MAP["default"],
        )
        return list(categories)

    # ------------------------------------------------------------------ #
    # Priority Matrix                                                      #
    # ------------------------------------------------------------------ #

    def map_stakeholder_priority(
        self,
        stakeholders: List[Stakeholder],
    ) -> Dict[str, List[str]]:
        """Map stakeholders to a priority matrix based on influence x impact.

        Creates a 2x2 matrix with quadrants:
        - high_high: High influence (>=3) AND high impact (>=3)
        - high_low: High influence (>=3) AND low impact (<3)
        - low_high: Low influence (<3) AND high impact (>=3)
        - low_low: Low influence (<3) AND low impact (<3)

        Args:
            stakeholders: List of Stakeholder objects.

        Returns:
            Dict with quadrant keys mapping to lists of stakeholder names.
        """
        matrix: Dict[str, List[str]] = {
            "high_high": [],
            "high_low": [],
            "low_high": [],
            "low_low": [],
        }

        for sh in stakeholders:
            high_influence = sh.influence_level >= PRIORITY_THRESHOLD
            high_impact = sh.impact_level >= PRIORITY_THRESHOLD

            if high_influence and high_impact:
                matrix["high_high"].append(sh.name)
            elif high_influence and not high_impact:
                matrix["high_low"].append(sh.name)
            elif not high_influence and high_impact:
                matrix["low_high"].append(sh.name)
            else:
                matrix["low_low"].append(sh.name)

        return matrix

    # ------------------------------------------------------------------ #
    # Consultation Recording                                               #
    # ------------------------------------------------------------------ #

    def record_consultation(
        self,
        stakeholder: Stakeholder,
        date: str,
        method: EngagementMethod,
        topics_discussed: List[str],
        key_findings: List[str],
        attendance_count: int = 1,
        duration_minutes: int = 60,
        documentation_available: bool = True,
        follow_up_actions: Optional[List[str]] = None,
    ) -> ConsultationRecord:
        """Create a new consultation record for a stakeholder.

        Updates the stakeholder's consultation status to IN_PROGRESS
        if it was NOT_STARTED or PLANNED.

        Args:
            stakeholder: The stakeholder being consulted.
            date: Date of consultation (ISO 8601).
            method: Engagement method used.
            topics_discussed: Topics covered in the consultation.
            key_findings: Key findings or insights.
            attendance_count: Number of participants.
            duration_minutes: Duration in minutes.
            documentation_available: Whether documentation exists.
            follow_up_actions: Agreed follow-up actions.

        Returns:
            ConsultationRecord with all fields populated.
        """
        record = ConsultationRecord(
            stakeholder_id=stakeholder.id,
            date=date,
            method=method,
            topics_discussed=topics_discussed,
            key_findings=key_findings,
            attendance_count=attendance_count,
            duration_minutes=duration_minutes,
            documentation_available=documentation_available,
            follow_up_actions=follow_up_actions or [],
        )

        # Update stakeholder status (side effect on mutable model)
        if stakeholder.consultation_status in (
            ConsultationStatus.NOT_STARTED,
            ConsultationStatus.PLANNED,
        ):
            stakeholder.consultation_status = ConsultationStatus.IN_PROGRESS

        return record

    # ------------------------------------------------------------------ #
    # Coverage Calculation                                                 #
    # ------------------------------------------------------------------ #

    def calculate_coverage(
        self,
        stakeholders: List[Stakeholder],
        records: List[ConsultationRecord],
    ) -> Decimal:
        """Calculate the percentage of stakeholders that have been engaged.

        A stakeholder is considered engaged if they have at least one
        consultation record OR their status is COMPLETED or VALIDATED.

        Args:
            stakeholders: List of identified stakeholders.
            records: List of consultation records.

        Returns:
            Coverage percentage as Decimal (0-100 scale, 2 decimal places).
        """
        if not stakeholders:
            return Decimal("0.00")

        # IDs with consultation records
        consulted_ids = set(r.stakeholder_id for r in records)

        # IDs with COMPLETED or VALIDATED status
        completed_ids = set(
            sh.id for sh in stakeholders
            if sh.consultation_status in (
                ConsultationStatus.COMPLETED,
                ConsultationStatus.VALIDATED,
                ConsultationStatus.IN_PROGRESS,
            )
        )

        engaged_ids = consulted_ids | completed_ids
        engaged_count = sum(1 for sh in stakeholders if sh.id in engaged_ids)

        pct = _decimal(engaged_count) / _decimal(len(stakeholders)) * _decimal(100)
        return _round_val(pct, 2)

    # ------------------------------------------------------------------ #
    # Engagement Quality                                                   #
    # ------------------------------------------------------------------ #

    def assess_engagement_quality(
        self,
        stakeholders: List[Stakeholder],
        records: List[ConsultationRecord],
        all_esrs_topics: Optional[List[str]] = None,
    ) -> Decimal:
        """Assess the overall quality of stakeholder engagement.

        Quality is calculated as a weighted composite of:
        - Method quality (average quality score of methods used)
        - Stakeholder diversity (categories represented / total)
        - Topic coverage (topics discussed / total ESRS topics)
        - Attendance depth (average attendance per consultation)
        - Documentation completeness (% of consultations documented)
        - Follow-up actions (has any follow-up been identified)

        Args:
            stakeholders: List of stakeholders.
            records: List of consultation records.
            all_esrs_topics: List of all ESRS topics for coverage calc.
                Defaults to 10 standard topics if not provided.

        Returns:
            Quality score as Decimal (0-1 scale, 3 decimal places).
        """
        if not records:
            return Decimal("0.000")

        if all_esrs_topics is None:
            all_esrs_topics = [
                "e1_climate", "e2_pollution", "e3_water",
                "e4_biodiversity", "e5_circular_economy",
                "s1_own_workforce", "s2_value_chain_workers",
                "s3_affected_communities", "s4_consumers",
                "g1_business_conduct",
            ]

        # 1. Method quality: average of engagement method quality scores
        method_scores = []
        for r in records:
            mq = ENGAGEMENT_METHOD_QUALITY.get(r.method.value, Decimal("0.40"))
            method_scores.append(mq)
        avg_method = (
            sum(method_scores, Decimal("0")) / _decimal(len(method_scores))
        )

        # 2. Stakeholder diversity: categories with engaged stakeholders
        consulted_ids = set(r.stakeholder_id for r in records)
        engaged_categories = set()
        for sh in stakeholders:
            if sh.id in consulted_ids:
                engaged_categories.add(sh.category.value)
        total_possible = len(StakeholderCategory)
        diversity_score = _decimal(len(engaged_categories)) / _decimal(total_possible)

        # 3. Topic coverage: unique topics discussed / total ESRS topics
        all_topics_discussed: set = set()
        for r in records:
            for t in r.topics_discussed:
                all_topics_discussed.add(t)
        topic_coverage = (
            _decimal(len(all_topics_discussed))
            / _decimal(max(len(all_esrs_topics), 1))
        )
        # Cap at 1.0
        if topic_coverage > Decimal("1.0"):
            topic_coverage = Decimal("1.0")

        # 4. Attendance depth: normalised average attendance
        # Use log-scale normalisation: score = min(1, ln(avg+1)/ln(50+1))
        total_attendance = sum(r.attendance_count for r in records)
        avg_attendance = _safe_divide(
            float(total_attendance), float(len(records))
        )
        import math
        attendance_normalised = min(
            1.0, math.log(avg_attendance + 1) / math.log(51)
        )
        attendance_score = _decimal(attendance_normalised)

        # 5. Documentation completeness
        documented = sum(1 for r in records if r.documentation_available)
        doc_score = _decimal(documented) / _decimal(len(records))

        # 6. Follow-up actions
        has_followup = sum(
            1 for r in records if len(r.follow_up_actions) > 0
        )
        followup_score = _decimal(has_followup) / _decimal(len(records))

        # Weighted composite
        weights = ENGAGEMENT_QUALITY_CRITERIA
        quality = (
            avg_method * weights["method_quality"]
            + diversity_score * weights["stakeholder_diversity"]
            + topic_coverage * weights["topic_coverage"]
            + attendance_score * weights["attendance_depth"]
            + doc_score * weights["documentation_completeness"]
            + followup_score * weights["follow_up_actions"]
        )

        return _round_val(quality, 3)

    # ------------------------------------------------------------------ #
    # Synthesis                                                            #
    # ------------------------------------------------------------------ #

    def synthesize_findings(
        self,
        stakeholders: List[Stakeholder],
        records: List[ConsultationRecord],
        all_esrs_topics: Optional[List[str]] = None,
    ) -> StakeholderEngagementResult:
        """Synthesize all stakeholder engagement data into a result.

        This is the main analysis method that computes all engagement
        metrics: coverage, quality, topic frequency, priority matrix,
        and recommendations.

        Args:
            stakeholders: List of identified stakeholders.
            records: List of consultation records.
            all_esrs_topics: Optional list of all ESRS topics.

        Returns:
            StakeholderEngagementResult with complete analysis.

        Raises:
            ValueError: If stakeholders list is empty.
        """
        t0 = time.perf_counter()

        if not stakeholders:
            raise ValueError("At least one Stakeholder is required")

        if all_esrs_topics is None:
            all_esrs_topics = [
                "e1_climate", "e2_pollution", "e3_water",
                "e4_biodiversity", "e5_circular_economy",
                "s1_own_workforce", "s2_value_chain_workers",
                "s3_affected_communities", "s4_consumers",
                "g1_business_conduct",
            ]

        # Coverage
        coverage_pct = self.calculate_coverage(stakeholders, records)
        consulted_ids = set(r.stakeholder_id for r in records)
        completed_ids = set(
            sh.id for sh in stakeholders
            if sh.consultation_status in (
                ConsultationStatus.COMPLETED,
                ConsultationStatus.VALIDATED,
                ConsultationStatus.IN_PROGRESS,
            )
        )
        engaged_ids = consulted_ids | completed_ids
        engaged_count = sum(1 for sh in stakeholders if sh.id in engaged_ids)

        # Category coverage
        category_counts: Dict[str, int] = {}
        for sh in stakeholders:
            cat = sh.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        categories_represented = len(category_counts)

        # Priority matrix
        priority_matrix = self.map_stakeholder_priority(stakeholders)

        # Consultation summary
        total_consultations = len(records)
        total_participants = sum(r.attendance_count for r in records)
        methods_used = list(set(r.method.value for r in records))

        # Topic frequency
        topic_freq: Dict[str, int] = {}
        for r in records:
            for t in r.topics_discussed:
                topic_freq[t] = topic_freq.get(t, 0) + 1

        # Top topics (sorted by frequency descending)
        sorted_topics = sorted(
            topic_freq.items(), key=lambda x: x[1], reverse=True
        )
        top_topics = [t[0] for t in sorted_topics]

        # Topics not covered
        covered_topics = set(topic_freq.keys())
        topics_not_covered = [
            t for t in all_esrs_topics if t not in covered_topics
        ]

        # Quality score
        quality_score = self.assess_engagement_quality(
            stakeholders, records, all_esrs_topics
        )
        quality_grade = self._quality_grade(quality_score)

        # Average method quality
        if records:
            method_scores = [
                ENGAGEMENT_METHOD_QUALITY.get(r.method.value, Decimal("0.40"))
                for r in records
            ]
            avg_method_q = _round_val(
                sum(method_scores, Decimal("0")) / _decimal(len(method_scores)),
                3,
            )
        else:
            avg_method_q = Decimal("0.000")

        # Average consultation duration
        if records:
            avg_duration = _round2(
                _safe_divide(
                    float(sum(r.duration_minutes for r in records)),
                    float(len(records)),
                )
            )
        else:
            avg_duration = 0.0

        # Documentation rate
        if records:
            documented = sum(1 for r in records if r.documentation_available)
            doc_rate = _round_val(
                _decimal(documented) / _decimal(len(records)) * _decimal(100),
                2,
            )
        else:
            doc_rate = Decimal("0.00")

        # Findings and follow-up counts
        total_findings = sum(len(r.key_findings) for r in records)
        total_follow_up = sum(len(r.follow_up_actions) for r in records)

        # Recommendations
        recommendations = self._generate_recommendations(
            stakeholders, records, coverage_pct, quality_score,
            quality_grade, topics_not_covered, categories_represented,
            methods_used,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = StakeholderEngagementResult(
            total_stakeholders=len(stakeholders),
            engaged_count=engaged_count,
            coverage_pct=coverage_pct,
            category_coverage=category_counts,
            categories_represented=categories_represented,
            total_consultations=total_consultations,
            total_participants=total_participants,
            methods_used=methods_used,
            topic_frequency=topic_freq,
            top_topics=top_topics,
            topics_not_covered=topics_not_covered,
            priority_matrix=priority_matrix,
            engagement_quality_score=quality_score,
            engagement_quality_grade=quality_grade,
            avg_method_quality=avg_method_q,
            avg_consultation_duration_minutes=avg_duration,
            documentation_rate_pct=doc_rate,
            total_findings=total_findings,
            total_follow_up_actions=total_follow_up,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Quality Grading                                                      #
    # ------------------------------------------------------------------ #

    def _quality_grade(self, score: Decimal) -> str:
        """Convert quality score to letter grade.

        Grading thresholds:
            A: >= 0.80 (excellent engagement)
            B: >= 0.60 (good engagement)
            C: >= 0.40 (adequate engagement)
            D: >= 0.20 (poor engagement)
            F: < 0.20 (failing engagement)

        Args:
            score: Quality score (0-1 Decimal).

        Returns:
            Letter grade A through F.
        """
        score_float = float(score)
        if score_float >= 0.80:
            return "A"
        elif score_float >= 0.60:
            return "B"
        elif score_float >= 0.40:
            return "C"
        elif score_float >= 0.20:
            return "D"
        else:
            return "F"

    # ------------------------------------------------------------------ #
    # Recommendations                                                      #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        stakeholders: List[Stakeholder],
        records: List[ConsultationRecord],
        coverage_pct: Decimal,
        quality_score: Decimal,
        quality_grade: str,
        topics_not_covered: List[str],
        categories_represented: int,
        methods_used: List[str],
    ) -> List[str]:
        """Generate deterministic recommendations for improving engagement.

        Recommendations are derived from threshold comparisons on
        calculated metrics, not from any LLM or probabilistic model.

        Args:
            stakeholders: Identified stakeholders.
            records: Consultation records.
            coverage_pct: Engagement coverage percentage.
            quality_score: Quality score (0-1).
            quality_grade: Quality grade letter.
            topics_not_covered: ESRS topics not discussed.
            categories_represented: Number of categories with stakeholders.
            methods_used: Distinct engagement methods used.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # R1: Low coverage
        if float(coverage_pct) < 50.0:
            recs.append(
                f"Stakeholder engagement coverage is {coverage_pct}%, "
                f"which is below 50%. Per ESRS 1 Para 22, the undertaking "
                f"shall consider perspectives of its stakeholders. "
                f"Expand engagement to cover more identified stakeholders."
            )

        # R2: Missing categories
        total_possible = len(StakeholderCategory)
        if categories_represented < total_possible // 2:
            recs.append(
                f"Only {categories_represented} of {total_possible} "
                f"stakeholder categories are represented. Consider "
                f"engaging with additional stakeholder groups to ensure "
                f"comprehensive perspective capture."
            )

        # R3: No consultation records
        if not records:
            recs.append(
                "No consultation records found. Per ESRS 1 Para 23, "
                "engagement with affected stakeholders is essential to "
                "identifying and assessing impacts. Begin stakeholder "
                "consultations as a priority."
            )

        # R4: Topics not covered
        if topics_not_covered:
            uncovered = ", ".join(topics_not_covered[:5])
            recs.append(
                f"The following ESRS topics have not been discussed in "
                f"any consultation: {uncovered}. Ensure all potentially "
                f"relevant topics are covered to support a comprehensive "
                f"materiality assessment."
            )

        # R5: Low quality grade
        if quality_grade in ("D", "F"):
            recs.append(
                f"Engagement quality score is {quality_score} "
                f"(grade {quality_grade}). Improve engagement depth by "
                f"using more interactive methods (workshops, advisory "
                f"panels) rather than passive methods (surveys, written "
                f"submissions)."
            )

        # R6: Single method used
        if len(methods_used) <= 1 and records:
            recs.append(
                f"Only one engagement method used ({methods_used[0] if methods_used else 'none'}). "
                f"Diversify methods (interviews, workshops, focus groups) "
                f"to capture richer insights from different stakeholder "
                f"types."
            )

        # R7: Low documentation rate
        if records:
            documented = sum(1 for r in records if r.documentation_available)
            doc_rate = _safe_pct(float(documented), float(len(records)))
            if doc_rate < 80.0:
                recs.append(
                    f"Documentation rate is {_round2(doc_rate)}%. "
                    f"Ensure all consultations are documented with "
                    f"minutes, recordings, or summary notes to support "
                    f"audit trail requirements."
                )

        # R8: No follow-up actions
        if records:
            has_followup = sum(
                1 for r in records if len(r.follow_up_actions) > 0
            )
            if has_followup == 0:
                recs.append(
                    "No follow-up actions recorded from any consultation. "
                    "Document specific actions arising from stakeholder "
                    "feedback to demonstrate responsiveness per ESRS "
                    "disclosure requirements."
                )

        return recs

    # ------------------------------------------------------------------ #
    # Utility: Priority Label                                              #
    # ------------------------------------------------------------------ #

    def get_priority_label(self, quadrant: str) -> str:
        """Return the description for a priority matrix quadrant.

        Args:
            quadrant: One of "high_high", "high_low", "low_high", "low_low".

        Returns:
            Description string.
        """
        return PRIORITY_MATRIX_LABELS.get(quadrant, "Unknown quadrant")

    def get_method_quality_score(self, method: EngagementMethod) -> Decimal:
        """Return the quality score for an engagement method.

        Args:
            method: EngagementMethod enum value.

        Returns:
            Quality score as Decimal (0-1 scale).
        """
        return ENGAGEMENT_METHOD_QUALITY.get(method.value, Decimal("0.40"))
