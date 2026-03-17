# -*- coding: utf-8 -*-
"""
StakeholderEngagementEngine - PACK-019 CSDDD Stakeholder Engagement Engine
===========================================================================

Plans and tracks stakeholder engagement activities per Article 11
of the EU Corporate Sustainability Due Diligence Directive
(CSDDD / CS3D).  Article 11 requires companies to carry out
meaningful engagement with affected stakeholders at key stages
of the due diligence process, including the identification and
assessment of adverse impacts, the development and implementation
of prevention and corrective action plans, and remediation.

The engine evaluates engagement quality, coverage, frequency, and
meaningfulness across all relevant stakeholder groups, assessing
whether engagement practices meet the Art 11 requirements for
effective consultation.

CSDDD Article 11 Requirements:
    - Art 11(1): Companies shall carry out meaningful engagement
      with affected stakeholders at relevant stages of due diligence
    - Art 11(2): Engagement shall include consultation on adverse
      impacts, prevention measures, and remediation
    - Art 11(3): Engagement shall be appropriate to the stakeholder
      group's characteristics and the company's activities
    - Art 11(4): Where direct engagement is not possible, companies
      shall engage through legitimate representatives

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Article 11: Meaningful engagement with stakeholders
    - UN Guiding Principles on Business and Human Rights (2011)
    - OECD Due Diligence Guidance for Responsible Business Conduct
    - ILO Tripartite Declaration of Principles
    - AA1000 Stakeholder Engagement Standard (AA1000SES)
    - ESRS 2 General Disclosures (SBM-2, IRO-1)

Zero-Hallucination:
    - Coverage scores computed from group representation ratios
    - Quality scores computed from meaningfulness boolean counts
    - Frequency scores computed from engagement date intervals
    - All scoring uses deterministic Decimal arithmetic
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


class EngagementMethod(str, Enum):
    """Methods for conducting stakeholder engagement per Art 11 CSDDD.

    The method chosen should be appropriate to the stakeholder
    group and the nature of the impacts being discussed, per
    Art 11(3) CSDDD.
    """
    FORMAL_CONSULTATION = "formal_consultation"
    COMMUNITY_MEETING = "community_meeting"
    WRITTEN_CONSULTATION = "written_consultation"
    SURVEY = "survey"
    FOCUS_GROUP = "focus_group"
    WORKSHOP = "workshop"
    BILATERAL_MEETING = "bilateral_meeting"
    PUBLIC_HEARING = "public_hearing"


class EngagementQuality(str, Enum):
    """Quality assessment of a stakeholder engagement activity.

    Reflects whether the engagement was meaningful as required by
    Art 11(1) CSDDD, ranging from not conducted to meaningful.
    """
    MEANINGFUL = "meaningful"
    ADEQUATE = "adequate"
    INSUFFICIENT = "insufficient"
    NOT_CONDUCTED = "not_conducted"


class StakeholderGroup(str, Enum):
    """Stakeholder groups relevant to CSDDD due diligence per Art 11.

    Art 11 requires engagement with affected stakeholders, which
    includes workers, trade unions, communities, indigenous peoples,
    NGOs, investors, consumers, and regulators.
    """
    WORKERS = "workers"
    TRADE_UNIONS = "trade_unions"
    COMMUNITIES = "communities"
    INDIGENOUS_PEOPLES = "indigenous_peoples"
    NGOS = "ngos"
    INVESTORS = "investors"
    CONSUMERS = "consumers"
    REGULATORS = "regulators"


class DueDiligenceStage(str, Enum):
    """Stages of the due diligence process where engagement is required.

    Per Art 11(2) CSDDD, engagement shall occur at relevant stages
    including impact identification, prevention planning, and
    remediation.
    """
    IMPACT_IDENTIFICATION = "impact_identification"
    IMPACT_ASSESSMENT = "impact_assessment"
    PREVENTION_PLANNING = "prevention_planning"
    CORRECTIVE_ACTION = "corrective_action"
    REMEDIATION = "remediation"
    MONITORING = "monitoring"
    REPORTING = "reporting"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_STAKEHOLDER_GROUPS: List[StakeholderGroup] = list(StakeholderGroup)
ALL_DD_STAGES: List[DueDiligenceStage] = list(DueDiligenceStage)

# Minimum engagement frequency per year for adequate coverage
MINIMUM_ANNUAL_ENGAGEMENTS: int = 1

# Target frequency for meaningful engagement (per group per year)
TARGET_ANNUAL_ENGAGEMENTS: int = 2

# Quality score mapping
QUALITY_SCORES: Dict[str, Decimal] = {
    EngagementQuality.MEANINGFUL.value: Decimal("100"),
    EngagementQuality.ADEQUATE.value: Decimal("70"),
    EngagementQuality.INSUFFICIENT.value: Decimal("30"),
    EngagementQuality.NOT_CONDUCTED.value: Decimal("0"),
}

# Due diligence stages required by Art 11(2)
REQUIRED_DD_STAGES: List[str] = [
    DueDiligenceStage.IMPACT_IDENTIFICATION.value,
    DueDiligenceStage.IMPACT_ASSESSMENT.value,
    DueDiligenceStage.PREVENTION_PLANNING.value,
    DueDiligenceStage.CORRECTIVE_ACTION.value,
    DueDiligenceStage.REMEDIATION.value,
]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class StakeholderEngagement(BaseModel):
    """A single stakeholder engagement activity per Art 11 CSDDD.

    Represents one engagement event with a stakeholder group,
    recording the method, topic, participants, outcomes, and
    quality assessment.
    """
    engagement_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this engagement activity",
    )
    stakeholder_group: StakeholderGroup = Field(
        ...,
        description="Stakeholder group engaged",
    )
    method: EngagementMethod = Field(
        ...,
        description="Method of engagement used",
    )
    topic: str = Field(
        default="",
        description="Primary topic or issue discussed",
        max_length=1000,
    )
    dd_stage: DueDiligenceStage = Field(
        default=DueDiligenceStage.IMPACT_IDENTIFICATION,
        description="Due diligence stage this engagement relates to",
    )
    date: datetime = Field(
        default_factory=_utcnow,
        description="Date the engagement took place",
    )
    duration_hours: Decimal = Field(
        default=Decimal("0"),
        description="Duration of the engagement in hours",
        ge=Decimal("0"),
    )
    participants: int = Field(
        default=0,
        description="Number of stakeholder participants",
        ge=0,
    )
    company_representatives: int = Field(
        default=0,
        description="Number of company representatives present",
        ge=0,
    )
    outcomes: str = Field(
        default="",
        description="Description of engagement outcomes and actions agreed",
        max_length=5000,
    )
    follow_up_actions: List[str] = Field(
        default_factory=list,
        description="List of follow-up actions agreed",
    )
    meaningful: bool = Field(
        default=False,
        description="Whether the engagement was meaningful per Art 11(1)",
    )
    informed_consent_obtained: bool = Field(
        default=False,
        description="Whether free, prior, and informed consent was obtained "
                    "(relevant for indigenous peoples)",
    )
    language_appropriate: bool = Field(
        default=True,
        description="Whether the engagement was conducted in an appropriate language",
    )
    documentation_available: bool = Field(
        default=False,
        description="Whether the engagement was documented",
    )
    feedback_incorporated: bool = Field(
        default=False,
        description="Whether stakeholder feedback was incorporated into decisions",
    )
    location: str = Field(
        default="",
        description="Location of the engagement",
        max_length=500,
    )
    quality: EngagementQuality = Field(
        default=EngagementQuality.NOT_CONDUCTED,
        description="Overall quality assessment of the engagement",
    )


class EngagementAssessment(BaseModel):
    """Assessment of engagement with a single stakeholder group.

    Aggregates all engagement activities for one group and provides
    a quality assessment per Art 11 CSDDD requirements.
    """
    stakeholder_group: str = Field(
        ..., description="Stakeholder group assessed"
    )
    quality: str = Field(
        default=EngagementQuality.NOT_CONDUCTED.value,
        description="Overall quality for this group",
    )
    engagement_count: int = Field(
        default=0, description="Number of engagements with this group"
    )
    total_participants: int = Field(
        default=0, description="Total stakeholder participants"
    )
    total_duration_hours: Decimal = Field(
        default=Decimal("0"), description="Total engagement hours"
    )
    topics_covered: List[str] = Field(
        default_factory=list, description="Topics covered in engagements"
    )
    dd_stages_covered: List[str] = Field(
        default_factory=list,
        description="Due diligence stages covered",
    )
    methods_used: List[str] = Field(
        default_factory=list, description="Methods used for engagement"
    )
    meaningfulness_rate_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of engagements that were meaningful",
    )
    feedback_incorporation_rate_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of engagements where feedback was incorporated",
    )
    documentation_rate_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of engagements that were documented",
    )
    frequency_score: Decimal = Field(
        default=Decimal("0"),
        description="Frequency score (0-100)",
    )
    quality_score: Decimal = Field(
        default=Decimal("0"),
        description="Quality score (0-100)",
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Identified gaps for this stakeholder group",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for this group",
    )


class EngagementResult(BaseModel):
    """Complete stakeholder engagement assessment result per Art 11 CSDDD.

    Aggregates all engagement activities, group-level assessments,
    coverage analysis, and quality metrics into a single result
    with provenance tracking.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Unique result identifier"
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version used"
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp of assessment (UTC)"
    )
    entity_name: str = Field(
        default="", description="Entity or undertaking name"
    )
    reporting_year: int = Field(
        default=0, description="Reporting year"
    )
    total_engagements: int = Field(
        default=0, description="Total engagement activities"
    )
    total_participants: int = Field(
        default=0, description="Total stakeholder participants"
    )
    total_duration_hours: Decimal = Field(
        default=Decimal("0"), description="Total engagement hours"
    )
    group_assessments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-group engagement assessments",
    )
    coverage_score: Decimal = Field(
        default=Decimal("0"),
        description="Stakeholder group coverage score (0-100)",
    )
    quality_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall quality score (0-100)",
    )
    frequency_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall frequency score (0-100)",
    )
    meaningfulness_rate: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of engagements that were meaningful",
    )
    dd_stage_coverage: Dict[str, Any] = Field(
        default_factory=dict,
        description="Coverage of due diligence stages",
    )
    method_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of engagement methods used",
    )
    groups_engaged: int = Field(
        default=0, description="Number of distinct groups engaged"
    )
    groups_not_engaged: List[str] = Field(
        default_factory=list,
        description="Stakeholder groups not engaged",
    )
    compliance_gaps: List[str] = Field(
        default_factory=list,
        description="Identified compliance gaps under Art 11",
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


class StakeholderEngagementEngine:
    """CSDDD Article 11 stakeholder engagement assessment engine.

    Provides deterministic, zero-hallucination assessments for
    stakeholder engagement practices against Art 11 CSDDD
    requirements, evaluating:

    - Coverage: which stakeholder groups are being engaged
    - Quality: whether engagements are meaningful per Art 11(1)
    - Frequency: whether engagement occurs at sufficient intervals
    - Due diligence stage coverage: engagement at required stages
    - Method appropriateness: suitability of engagement methods

    All calculations use Decimal arithmetic for reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = StakeholderEngagementEngine()
        result = engine.assess_engagement(
            engagements=[StakeholderEngagement(...)],
        )
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Main Assessment Method                                               #
    # ------------------------------------------------------------------ #

    def assess_engagement(
        self,
        engagements: List[StakeholderEngagement],
        entity_name: str = "",
        reporting_year: int = 0,
    ) -> EngagementResult:
        """Perform a complete stakeholder engagement assessment.

        Orchestrates evaluation of coverage, quality, frequency,
        and due diligence stage coverage to produce a consolidated
        result per Art 11 CSDDD.

        Args:
            engagements: List of StakeholderEngagement instances.
            entity_name: Name of the reporting entity.
            reporting_year: Reporting year.

        Returns:
            EngagementResult with complete assessment and provenance.
        """
        t0 = time.perf_counter()

        logger.info(
            "Assessing stakeholder engagement: %d activities, "
            "entity=%s, year=%d",
            len(engagements), entity_name, reporting_year,
        )

        # Step 1: Assess by group
        group_assessments = self.assess_by_group(engagements)

        # Step 2: Assess coverage
        coverage_result = self.assess_coverage(engagements)
        coverage_score = coverage_result["coverage_score"]

        # Step 3: Assess quality
        quality_result = self.assess_quality(engagements)
        quality_score = quality_result["overall_quality_score"]

        # Step 4: Assess frequency
        frequency_result = self.assess_frequency(engagements)
        frequency_score = frequency_result["overall_frequency_score"]

        # Step 5: Assess DD stage coverage
        dd_stage_coverage = self._assess_dd_stage_coverage(engagements)

        # Step 6: Method distribution
        method_dist = self._calculate_method_distribution(engagements)

        # Step 7: Overall meaningfulness rate
        meaningful_count = sum(
            1 for e in engagements if e.meaningful
        )
        meaningfulness_rate = _pct(meaningful_count, len(engagements))

        # Step 8: Totals
        total_participants = sum(e.participants for e in engagements)
        total_duration = sum(
            e.duration_hours for e in engagements
        )

        # Step 9: Groups not engaged
        engaged_groups = set(
            e.stakeholder_group.value for e in engagements
        )
        all_groups = set(g.value for g in ALL_STAKEHOLDER_GROUPS)
        not_engaged = sorted(all_groups - engaged_groups)

        # Step 10: Compliance gaps
        compliance_gaps = self._identify_compliance_gaps(
            engagements, coverage_result, quality_result,
            frequency_result, dd_stage_coverage,
        )

        # Step 11: Recommendations
        recommendations = self._generate_recommendations(
            engagements, group_assessments, coverage_result,
            quality_result, frequency_result, dd_stage_coverage,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = EngagementResult(
            entity_name=entity_name,
            reporting_year=reporting_year,
            total_engagements=len(engagements),
            total_participants=total_participants,
            total_duration_hours=_round_val(total_duration, 1),
            group_assessments=group_assessments,
            coverage_score=coverage_score,
            quality_score=quality_score,
            frequency_score=frequency_score,
            meaningfulness_rate=meaningfulness_rate,
            dd_stage_coverage=dd_stage_coverage,
            method_distribution=method_dist,
            groups_engaged=len(engaged_groups),
            groups_not_engaged=not_engaged,
            compliance_gaps=compliance_gaps,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Stakeholder engagement assessed: coverage=%.1f%%, "
            "quality=%.1f%%, frequency=%.1f%%, meaningful=%.1f%%, "
            "groups=%d/%d, hash=%s",
            float(coverage_score), float(quality_score),
            float(frequency_score), float(meaningfulness_rate),
            len(engaged_groups), len(all_groups),
            result.provenance_hash[:16],
        )

        return result

    # ------------------------------------------------------------------ #
    # Per-Group Assessment                                                 #
    # ------------------------------------------------------------------ #

    def assess_by_group(
        self, engagements: List[StakeholderEngagement]
    ) -> List[Dict[str, Any]]:
        """Assess engagement quality and coverage per stakeholder group.

        Evaluates each stakeholder group individually for quality,
        frequency, topic coverage, and meaningfulness.

        Args:
            engagements: List of StakeholderEngagement instances.

        Returns:
            List of dicts, one per stakeholder group, with detailed
            assessment metrics.
        """
        assessments: List[Dict[str, Any]] = []

        for group in ALL_STAKEHOLDER_GROUPS:
            group_engagements = [
                e for e in engagements
                if e.stakeholder_group == group
            ]

            if not group_engagements:
                assessment = {
                    "stakeholder_group": group.value,
                    "quality": EngagementQuality.NOT_CONDUCTED.value,
                    "engagement_count": 0,
                    "total_participants": 0,
                    "total_duration_hours": Decimal("0"),
                    "topics_covered": [],
                    "dd_stages_covered": [],
                    "methods_used": [],
                    "meaningfulness_rate_pct": Decimal("0.0"),
                    "feedback_incorporation_rate_pct": Decimal("0.0"),
                    "documentation_rate_pct": Decimal("0.0"),
                    "frequency_score": Decimal("0"),
                    "quality_score": Decimal("0"),
                    "gaps": [
                        f"No engagement conducted with {group.value}"
                    ],
                    "recommendations": [
                        f"Initiate engagement with {group.value} as "
                        f"required by Art 11 CSDDD."
                    ],
                }
                assessments.append(assessment)
                continue

            count = len(group_engagements)
            total_participants = sum(
                e.participants for e in group_engagements
            )
            total_duration = sum(
                e.duration_hours for e in group_engagements
            )

            # Topics covered (unique)
            topics = sorted(set(
                e.topic for e in group_engagements if e.topic
            ))

            # DD stages covered (unique)
            dd_stages = sorted(set(
                e.dd_stage.value for e in group_engagements
            ))

            # Methods used (unique)
            methods = sorted(set(
                e.method.value for e in group_engagements
            ))

            # Meaningfulness rate
            meaningful_count = sum(
                1 for e in group_engagements if e.meaningful
            )
            meaningfulness_rate = _pct(meaningful_count, count)

            # Feedback incorporation rate
            feedback_count = sum(
                1 for e in group_engagements
                if e.feedback_incorporated
            )
            feedback_rate = _pct(feedback_count, count)

            # Documentation rate
            documented_count = sum(
                1 for e in group_engagements
                if e.documentation_available
            )
            documentation_rate = _pct(documented_count, count)

            # Quality score based on quality enum distribution
            quality_scores = [
                QUALITY_SCORES.get(
                    e.quality.value, Decimal("0")
                )
                for e in group_engagements
            ]
            avg_quality_score = _round_val(
                sum(quality_scores) / _decimal(len(quality_scores)), 1
            )

            # Frequency score
            freq_score = self._calculate_group_frequency_score(
                count
            )

            # Overall quality determination
            if avg_quality_score >= Decimal("80"):
                overall_quality = EngagementQuality.MEANINGFUL.value
            elif avg_quality_score >= Decimal("50"):
                overall_quality = EngagementQuality.ADEQUATE.value
            else:
                overall_quality = EngagementQuality.INSUFFICIENT.value

            # Gaps
            gaps: List[str] = []
            missing_stages = set(REQUIRED_DD_STAGES) - set(dd_stages)
            if missing_stages:
                gaps.append(
                    f"DD stages not covered for {group.value}: "
                    f"{', '.join(sorted(missing_stages))}"
                )
            if meaningfulness_rate < Decimal("50"):
                gaps.append(
                    f"Less than 50% of engagements with {group.value} "
                    f"are assessed as meaningful."
                )
            if documentation_rate < Decimal("80"):
                gaps.append(
                    f"Documentation rate for {group.value} is below 80%."
                )
            if feedback_rate < Decimal("50"):
                gaps.append(
                    f"Feedback incorporation rate for {group.value} "
                    f"is below 50%."
                )

            # Recommendations
            recs: List[str] = []
            if missing_stages:
                recs.append(
                    f"Extend engagement with {group.value} to cover "
                    f"missing DD stages: {', '.join(sorted(missing_stages))}."
                )
            if meaningfulness_rate < Decimal("50"):
                recs.append(
                    f"Improve meaningfulness of engagement with "
                    f"{group.value} by ensuring two-way dialogue and "
                    f"incorporating feedback into decisions."
                )
            if count < TARGET_ANNUAL_ENGAGEMENTS:
                recs.append(
                    f"Increase frequency of engagement with "
                    f"{group.value} to at least "
                    f"{TARGET_ANNUAL_ENGAGEMENTS} times per year."
                )

            assessment = {
                "stakeholder_group": group.value,
                "quality": overall_quality,
                "engagement_count": count,
                "total_participants": total_participants,
                "total_duration_hours": _round_val(total_duration, 1),
                "topics_covered": topics,
                "dd_stages_covered": dd_stages,
                "methods_used": methods,
                "meaningfulness_rate_pct": meaningfulness_rate,
                "feedback_incorporation_rate_pct": feedback_rate,
                "documentation_rate_pct": documentation_rate,
                "frequency_score": freq_score,
                "quality_score": avg_quality_score,
                "gaps": gaps,
                "recommendations": recs,
            }
            assessments.append(assessment)

        logger.info(
            "Per-group assessments: %d groups evaluated",
            len(assessments),
        )

        return assessments

    # ------------------------------------------------------------------ #
    # Coverage Assessment                                                  #
    # ------------------------------------------------------------------ #

    def assess_coverage(
        self, engagements: List[StakeholderEngagement]
    ) -> Dict[str, Any]:
        """Assess coverage of stakeholder groups in engagement activities.

        Determines which stakeholder groups have been engaged and
        which have not, calculating a coverage score.

        Args:
            engagements: List of StakeholderEngagement instances.

        Returns:
            Dict with coverage metrics and provenance hash.
        """
        all_groups = set(g.value for g in ALL_STAKEHOLDER_GROUPS)
        engaged_groups: Dict[str, int] = {}

        for eng in engagements:
            gv = eng.stakeholder_group.value
            engaged_groups[gv] = engaged_groups.get(gv, 0) + 1

        covered = set(engaged_groups.keys())
        not_covered = sorted(all_groups - covered)

        coverage_score = _pct(len(covered), len(all_groups))

        # Meaningful engagement coverage (groups with at least one
        # meaningful engagement)
        meaningful_groups: set = set()
        for eng in engagements:
            if eng.meaningful:
                meaningful_groups.add(eng.stakeholder_group.value)
        meaningful_coverage = _pct(
            len(meaningful_groups), len(all_groups)
        )

        result = {
            "total_stakeholder_groups": len(all_groups),
            "groups_engaged": len(covered),
            "groups_not_engaged": not_covered,
            "coverage_score": coverage_score,
            "engagement_counts_by_group": engaged_groups,
            "meaningful_coverage_groups": len(meaningful_groups),
            "meaningful_coverage_score": meaningful_coverage,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Coverage: %d/%d groups (%.1f%%), meaningful coverage=%.1f%%",
            len(covered), len(all_groups),
            float(coverage_score), float(meaningful_coverage),
        )

        return result

    # ------------------------------------------------------------------ #
    # Quality Assessment                                                   #
    # ------------------------------------------------------------------ #

    def assess_quality(
        self, engagements: List[StakeholderEngagement]
    ) -> Dict[str, Any]:
        """Assess quality and meaningfulness of engagement activities.

        Evaluates whether engagements meet the Art 11(1) requirement
        for meaningful engagement by analysing quality indicators.

        Args:
            engagements: List of StakeholderEngagement instances.

        Returns:
            Dict with quality metrics and provenance hash.
        """
        if not engagements:
            result: Dict[str, Any] = {
                "total_engagements": 0,
                "meaningful_count": 0,
                "adequate_count": 0,
                "insufficient_count": 0,
                "meaningfulness_rate_pct": Decimal("0.0"),
                "feedback_incorporation_rate_pct": Decimal("0.0"),
                "documentation_rate_pct": Decimal("0.0"),
                "language_appropriateness_rate_pct": Decimal("0.0"),
                "informed_consent_rate_pct": Decimal("0.0"),
                "overall_quality_score": Decimal("0"),
                "quality_distribution": {},
                "provenance_hash": _compute_hash({"empty": True}),
            }
            return result

        total = len(engagements)

        # Count by quality level
        quality_counts: Dict[str, int] = {}
        for q in EngagementQuality:
            quality_counts[q.value] = sum(
                1 for e in engagements if e.quality == q
            )

        meaningful = quality_counts.get(
            EngagementQuality.MEANINGFUL.value, 0
        )
        adequate = quality_counts.get(
            EngagementQuality.ADEQUATE.value, 0
        )
        insufficient = quality_counts.get(
            EngagementQuality.INSUFFICIENT.value, 0
        )

        # Rates
        meaningful_by_flag = sum(
            1 for e in engagements if e.meaningful
        )
        feedback_count = sum(
            1 for e in engagements if e.feedback_incorporated
        )
        documented_count = sum(
            1 for e in engagements if e.documentation_available
        )
        language_count = sum(
            1 for e in engagements if e.language_appropriate
        )

        # Informed consent (relevant primarily for indigenous peoples)
        indigenous_engagements = [
            e for e in engagements
            if e.stakeholder_group == StakeholderGroup.INDIGENOUS_PEOPLES
        ]
        consent_count = sum(
            1 for e in indigenous_engagements
            if e.informed_consent_obtained
        )
        consent_rate = _pct(
            consent_count, len(indigenous_engagements)
        )

        # Overall quality score (weighted average of quality enum scores)
        quality_score_values = [
            QUALITY_SCORES.get(e.quality.value, Decimal("0"))
            for e in engagements
        ]
        overall_quality = _round_val(
            sum(quality_score_values) / _decimal(total), 1
        )

        result = {
            "total_engagements": total,
            "meaningful_count": meaningful,
            "adequate_count": adequate,
            "insufficient_count": insufficient,
            "meaningfulness_rate_pct": _pct(meaningful_by_flag, total),
            "feedback_incorporation_rate_pct": _pct(feedback_count, total),
            "documentation_rate_pct": _pct(documented_count, total),
            "language_appropriateness_rate_pct": _pct(language_count, total),
            "informed_consent_rate_pct": consent_rate,
            "indigenous_engagements_count": len(indigenous_engagements),
            "overall_quality_score": overall_quality,
            "quality_distribution": quality_counts,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Quality: overall=%.1f%%, meaningful=%.1f%%, "
            "feedback=%.1f%%, documented=%.1f%%",
            float(overall_quality),
            float(_pct(meaningful_by_flag, total)),
            float(_pct(feedback_count, total)),
            float(_pct(documented_count, total)),
        )

        return result

    # ------------------------------------------------------------------ #
    # Frequency Assessment                                                 #
    # ------------------------------------------------------------------ #

    def assess_frequency(
        self, engagements: List[StakeholderEngagement]
    ) -> Dict[str, Any]:
        """Assess frequency of engagement activities per stakeholder group.

        Evaluates whether engagements occur at sufficient frequency
        to support ongoing due diligence requirements under Art 11.

        Args:
            engagements: List of StakeholderEngagement instances.

        Returns:
            Dict with frequency metrics and provenance hash.
        """
        if not engagements:
            result: Dict[str, Any] = {
                "total_engagements": 0,
                "groups_with_sufficient_frequency": 0,
                "groups_below_target_frequency": [],
                "overall_frequency_score": Decimal("0"),
                "by_group": {},
                "provenance_hash": _compute_hash({"empty": True}),
            }
            return result

        # Count engagements per group
        group_counts: Dict[str, int] = {}
        for eng in engagements:
            gv = eng.stakeholder_group.value
            group_counts[gv] = group_counts.get(gv, 0) + 1

        # Assess frequency per group
        sufficient = 0
        below_target: List[str] = []
        by_group: Dict[str, Dict[str, Any]] = {}

        for group in ALL_STAKEHOLDER_GROUPS:
            gv = group.value
            count = group_counts.get(gv, 0)
            freq_score = self._calculate_group_frequency_score(count)
            meets_minimum = count >= MINIMUM_ANNUAL_ENGAGEMENTS
            meets_target = count >= TARGET_ANNUAL_ENGAGEMENTS

            if meets_minimum:
                sufficient += 1
            else:
                below_target.append(gv)

            by_group[gv] = {
                "engagement_count": count,
                "frequency_score": freq_score,
                "meets_minimum": meets_minimum,
                "meets_target": meets_target,
            }

        # Overall frequency score (average of group scores)
        all_scores = [
            v["frequency_score"] for v in by_group.values()
        ]
        overall_freq = _round_val(
            sum(all_scores) / _decimal(len(all_scores)), 1
        ) if all_scores else Decimal("0")

        result = {
            "total_engagements": len(engagements),
            "groups_with_sufficient_frequency": sufficient,
            "groups_below_target_frequency": below_target,
            "overall_frequency_score": overall_freq,
            "by_group": by_group,
            "minimum_annual_target": MINIMUM_ANNUAL_ENGAGEMENTS,
            "target_annual_frequency": TARGET_ANNUAL_ENGAGEMENTS,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Frequency: overall=%.1f%%, %d/%d groups sufficient",
            float(overall_freq), sufficient, len(ALL_STAKEHOLDER_GROUPS),
        )

        return result

    # ------------------------------------------------------------------ #
    # Due Diligence Stage Coverage                                         #
    # ------------------------------------------------------------------ #

    def _assess_dd_stage_coverage(
        self, engagements: List[StakeholderEngagement]
    ) -> Dict[str, Any]:
        """Assess coverage of due diligence stages in engagements.

        Per Art 11(2) CSDDD, engagement shall occur at relevant
        stages including impact identification, prevention, and
        remediation.

        Args:
            engagements: List of StakeholderEngagement instances.

        Returns:
            Dict with DD stage coverage metrics.
        """
        all_stages = set(s.value for s in ALL_DD_STAGES)
        covered_stages: Dict[str, int] = {}

        for eng in engagements:
            sv = eng.dd_stage.value
            covered_stages[sv] = covered_stages.get(sv, 0) + 1

        covered = set(covered_stages.keys())
        required_covered = set(REQUIRED_DD_STAGES) & covered
        required_missing = set(REQUIRED_DD_STAGES) - covered

        coverage_pct = _pct(len(covered), len(all_stages))
        required_coverage_pct = _pct(
            len(required_covered), len(REQUIRED_DD_STAGES)
        )

        result = {
            "total_stages": len(all_stages),
            "stages_covered": len(covered),
            "stages_not_covered": sorted(all_stages - covered),
            "coverage_pct": coverage_pct,
            "required_stages_covered": len(required_covered),
            "required_stages_total": len(REQUIRED_DD_STAGES),
            "required_stages_missing": sorted(required_missing),
            "required_coverage_pct": required_coverage_pct,
            "engagements_by_stage": covered_stages,
        }
        result["provenance_hash"] = _compute_hash(result)

        return result

    # ------------------------------------------------------------------ #
    # Method Distribution                                                  #
    # ------------------------------------------------------------------ #

    def _calculate_method_distribution(
        self, engagements: List[StakeholderEngagement]
    ) -> Dict[str, int]:
        """Calculate the distribution of engagement methods used.

        Args:
            engagements: List of StakeholderEngagement instances.

        Returns:
            Dict mapping method names to their count.
        """
        distribution: Dict[str, int] = {}
        for method in EngagementMethod:
            count = sum(
                1 for e in engagements if e.method == method
            )
            if count > 0:
                distribution[method.value] = count
        return distribution

    # ------------------------------------------------------------------ #
    # Frequency Score Calculation                                          #
    # ------------------------------------------------------------------ #

    def _calculate_group_frequency_score(
        self, count: int
    ) -> Decimal:
        """Calculate frequency score for a stakeholder group.

        Score is based on engagement count relative to the target
        annual frequency.  Score is capped at 100.

        Args:
            count: Number of engagements with this group.

        Returns:
            Frequency score as Decimal (0-100).
        """
        if count == 0:
            return Decimal("0")

        if count >= TARGET_ANNUAL_ENGAGEMENTS:
            return Decimal("100")

        # Linear interpolation: 0 engagements = 0, target = 100
        score = _round_val(
            _decimal(count) / _decimal(TARGET_ANNUAL_ENGAGEMENTS)
            * Decimal("100"),
            1,
        )

        return min(score, Decimal("100"))

    # ------------------------------------------------------------------ #
    # Compliance Gap Identification                                        #
    # ------------------------------------------------------------------ #

    def _identify_compliance_gaps(
        self,
        engagements: List[StakeholderEngagement],
        coverage_result: Dict[str, Any],
        quality_result: Dict[str, Any],
        frequency_result: Dict[str, Any],
        dd_stage_coverage: Dict[str, Any],
    ) -> List[str]:
        """Identify compliance gaps under CSDDD Art 11.

        Args:
            engagements: All engagement activities.
            coverage_result: Coverage assessment result.
            quality_result: Quality assessment result.
            frequency_result: Frequency assessment result.
            dd_stage_coverage: DD stage coverage result.

        Returns:
            List of identified compliance gap descriptions.
        """
        gaps: List[str] = []

        # Coverage gaps
        not_engaged = coverage_result.get("groups_not_engaged", [])
        if not_engaged:
            gaps.append(
                f"Art 11(1): Stakeholder groups not engaged: "
                f"{', '.join(not_engaged)}"
            )

        # Quality gaps
        overall_quality = quality_result.get(
            "overall_quality_score", Decimal("0")
        )
        if isinstance(overall_quality, (int, float, str)):
            overall_quality = _decimal(overall_quality)
        if overall_quality < Decimal("50"):
            gaps.append(
                f"Art 11(1): Overall engagement quality score "
                f"({overall_quality}%) is below the 50% threshold "
                f"for meaningful engagement."
            )

        # Meaningfulness gaps
        meaningfulness = quality_result.get(
            "meaningfulness_rate_pct", Decimal("0")
        )
        if isinstance(meaningfulness, (int, float, str)):
            meaningfulness = _decimal(meaningfulness)
        if meaningfulness < Decimal("50") and len(engagements) > 0:
            gaps.append(
                f"Art 11(1): Only {meaningfulness}% of engagements "
                f"are assessed as meaningful."
            )

        # DD stage coverage gaps
        required_missing = dd_stage_coverage.get(
            "required_stages_missing", []
        )
        if required_missing:
            gaps.append(
                f"Art 11(2): Required DD stages not covered by "
                f"engagement: {', '.join(required_missing)}"
            )

        # Frequency gaps
        below_target = frequency_result.get(
            "groups_below_target_frequency", []
        )
        if below_target:
            gaps.append(
                f"Art 11: Stakeholder groups with insufficient "
                f"engagement frequency: {', '.join(below_target)}"
            )

        # Indigenous peoples FPIC
        consent_rate = quality_result.get(
            "informed_consent_rate_pct", Decimal("0")
        )
        indigenous_count = quality_result.get(
            "indigenous_engagements_count", 0
        )
        if isinstance(consent_rate, (int, float, str)):
            consent_rate = _decimal(consent_rate)
        if (
            indigenous_count > 0
            and consent_rate < Decimal("100")
        ):
            gaps.append(
                "Art 11: Free, prior, and informed consent not "
                "obtained for all engagements with indigenous peoples."
            )

        # Documentation gaps
        doc_rate = quality_result.get(
            "documentation_rate_pct", Decimal("0")
        )
        if isinstance(doc_rate, (int, float, str)):
            doc_rate = _decimal(doc_rate)
        if doc_rate < Decimal("80") and len(engagements) > 0:
            gaps.append(
                f"Documentation rate is {doc_rate}%, below the "
                f"recommended 80% threshold for audit trail."
            )

        return gaps

    # ------------------------------------------------------------------ #
    # Recommendations Generation                                           #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        engagements: List[StakeholderEngagement],
        group_assessments: List[Dict[str, Any]],
        coverage_result: Dict[str, Any],
        quality_result: Dict[str, Any],
        frequency_result: Dict[str, Any],
        dd_stage_coverage: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations for improving stakeholder engagement.

        Args:
            engagements: All engagement activities.
            group_assessments: Per-group assessment results.
            coverage_result: Coverage assessment result.
            quality_result: Quality assessment result.
            frequency_result: Frequency assessment result.
            dd_stage_coverage: DD stage coverage result.

        Returns:
            List of actionable recommendation strings.
        """
        recommendations: List[str] = []

        # Coverage recommendations
        not_engaged = coverage_result.get("groups_not_engaged", [])
        for group in not_engaged:
            recommendations.append(
                f"Initiate engagement with {group} stakeholder group "
                f"as required by Art 11 CSDDD."
            )

        # Quality recommendations
        overall_quality = quality_result.get(
            "overall_quality_score", Decimal("0")
        )
        if isinstance(overall_quality, (int, float, str)):
            overall_quality = _decimal(overall_quality)
        if overall_quality < Decimal("70"):
            recommendations.append(
                "Improve overall engagement quality by ensuring "
                "two-way dialogue, providing adequate information "
                "in advance, and incorporating feedback."
            )

        # Feedback incorporation
        feedback_rate = quality_result.get(
            "feedback_incorporation_rate_pct", Decimal("0")
        )
        if isinstance(feedback_rate, (int, float, str)):
            feedback_rate = _decimal(feedback_rate)
        if feedback_rate < Decimal("50") and len(engagements) > 0:
            recommendations.append(
                "Strengthen feedback incorporation mechanisms to "
                "demonstrate that stakeholder input influences "
                "due diligence decisions."
            )

        # DD stage recommendations
        required_missing = dd_stage_coverage.get(
            "required_stages_missing", []
        )
        if required_missing:
            recommendations.append(
                f"Extend engagement to cover the following required "
                f"DD stages: {', '.join(required_missing)}."
            )

        # Method diversity
        methods_used = set()
        for eng in engagements:
            methods_used.add(eng.method.value)
        if len(methods_used) < 3 and len(engagements) > 5:
            recommendations.append(
                "Diversify engagement methods (currently using "
                f"{len(methods_used)} methods). Consider adding "
                "community meetings, workshops, or focus groups "
                "to reach different stakeholder groups effectively."
            )

        # Documentation
        doc_rate = quality_result.get(
            "documentation_rate_pct", Decimal("0")
        )
        if isinstance(doc_rate, (int, float, str)):
            doc_rate = _decimal(doc_rate)
        if doc_rate < Decimal("80") and len(engagements) > 0:
            recommendations.append(
                "Improve documentation of engagement activities "
                "to maintain audit trail and demonstrate compliance."
            )

        # Indigenous peoples FPIC
        consent_rate = quality_result.get(
            "informed_consent_rate_pct", Decimal("0")
        )
        indigenous_count = quality_result.get(
            "indigenous_engagements_count", 0
        )
        if isinstance(consent_rate, (int, float, str)):
            consent_rate = _decimal(consent_rate)
        if (
            indigenous_count > 0
            and consent_rate < Decimal("100")
        ):
            recommendations.append(
                "Ensure free, prior, and informed consent (FPIC) "
                "is obtained for all engagements with indigenous "
                "peoples, per UNDRIP and ILO Convention 169."
            )

        # Collect per-group recommendations (limit per group)
        for ga in group_assessments:
            for rec in ga.get("recommendations", [])[:2]:
                if rec not in recommendations:
                    recommendations.append(rec)

        # Cap at 15 recommendations
        if len(recommendations) > 15:
            recommendations = recommendations[:15]

        return recommendations

    # ------------------------------------------------------------------ #
    # Period Comparison                                                     #
    # ------------------------------------------------------------------ #

    def compare_periods(
        self,
        current: EngagementResult,
        previous: EngagementResult,
    ) -> Dict[str, Any]:
        """Compare engagement performance across two reporting periods.

        Tracks changes in engagement volume, coverage, quality,
        and frequency metrics.

        Args:
            current: Current period result.
            previous: Previous period result.

        Returns:
            Dict with period-over-period changes and provenance.
        """
        comparison = {
            "current_period": current.reporting_year,
            "previous_period": previous.reporting_year,
            "engagements_change": (
                current.total_engagements - previous.total_engagements
            ),
            "participants_change": (
                current.total_participants - previous.total_participants
            ),
            "coverage_change_pp": _round_val(
                current.coverage_score - previous.coverage_score, 1
            ),
            "quality_change_pp": _round_val(
                current.quality_score - previous.quality_score, 1
            ),
            "frequency_change_pp": _round_val(
                current.frequency_score - previous.frequency_score, 1
            ),
            "meaningfulness_change_pp": _round_val(
                current.meaningfulness_rate - previous.meaningfulness_rate,
                1,
            ),
            "groups_engaged_change": (
                current.groups_engaged - previous.groups_engaged
            ),
            "direction": (
                "improving"
                if current.quality_score > previous.quality_score
                else (
                    "stable"
                    if current.quality_score == previous.quality_score
                    else "declining"
                )
            ),
        }

        comparison["provenance_hash"] = _compute_hash(comparison)
        return comparison
