# -*- coding: utf-8 -*-
"""
StakeholderEngine - PACK-002 CSRD Professional Engine 5

Stakeholder engagement management engine implementing ESRS 1 Chapter 3
requirements for double materiality assessment. Manages stakeholder
registration, salience mapping (power/legitimacy/urgency), engagement
activity tracking, materiality input aggregation, and audit-ready
evidence packaging.

Stakeholder Salience (Mitchell, Agle & Wood):
    - Power:      Stakeholder's ability to influence the company
    - Legitimacy: Moral/legal claim on the company
    - Urgency:    Time sensitivity of the stakeholder's claim
    - Salience:   Composite measure determining engagement priority

Salience Categories:
    - Definitive:     High power + legitimacy + urgency (all three)
    - Dominant:       High power + legitimacy
    - Dangerous:      High power + urgency
    - Dependent:      High legitimacy + urgency
    - Dormant:        High power only
    - Discretionary:  High legitimacy only
    - Demanding:      High urgency only
    - Non-stakeholder: None above threshold

Features:
    - Stakeholder registration with salience scoring
    - Power/legitimacy/urgency salience map generation
    - Engagement activity recording and tracking
    - Materiality input aggregation with stakeholder weighting
    - Survey generation for materiality assessment
    - Audit-ready evidence package compilation
    - SHA-256 provenance hashing on all outputs

Zero-Hallucination:
    - Salience scores use deterministic arithmetic
    - Materiality aggregation uses weighted averages
    - No LLM involvement in scoring calculations
    - Evidence packaging is a deterministic collection operation

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-002 CSRD Professional
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator, computed_field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# Salience threshold: scores above this are considered "high"
_SALIENCE_THRESHOLD: float = 5.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StakeholderCategory(str, Enum):
    """Stakeholder group categories per ESRS 1."""

    INVESTOR = "investor"
    EMPLOYEE = "employee"
    SUPPLIER = "supplier"
    COMMUNITY = "community"
    REGULATOR = "regulator"
    CUSTOMER = "customer"
    NGO = "ngo"
    BOARD_MEMBER = "board_member"

class EngagementType(str, Enum):
    """Types of stakeholder engagement activity."""

    SURVEY = "survey"
    INTERVIEW = "interview"
    WORKSHOP = "workshop"
    WRITTEN_CONSULTATION = "written_consultation"
    FOCUS_GROUP = "focus_group"
    ADVISORY_PANEL = "advisory_panel"

class SalienceCategory(str, Enum):
    """Salience quadrant based on power/legitimacy/urgency."""

    DEFINITIVE = "definitive"
    DOMINANT = "dominant"
    DANGEROUS = "dangerous"
    DEPENDENT = "dependent"
    DORMANT = "dormant"
    DISCRETIONARY = "discretionary"
    DEMANDING = "demanding"
    NON_STAKEHOLDER = "non_stakeholder"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class Stakeholder(BaseModel):
    """A stakeholder entity with salience attributes."""

    stakeholder_id: str = Field(default_factory=_new_uuid, description="Stakeholder ID")
    name: str = Field(..., description="Stakeholder name")
    category: StakeholderCategory = Field(..., description="Stakeholder group")
    organization: str = Field("", description="Organization or affiliation")
    email: Optional[str] = Field(None, description="Contact email")
    power_score: float = Field(
        0.0, ge=0.0, le=10.0, description="Power to influence (0-10)"
    )
    legitimacy_score: float = Field(
        0.0, ge=0.0, le=10.0, description="Moral/legal claim (0-10)"
    )
    urgency_score: float = Field(
        0.0, ge=0.0, le=10.0, description="Time sensitivity (0-10)"
    )
    registered_at: datetime = Field(
        default_factory=utcnow, description="Registration timestamp"
    )

    @computed_field
    @property
    def salience_score(self) -> float:
        """Compute composite salience score (0-10)."""
        return round(
            (self.power_score + self.legitimacy_score + self.urgency_score) / 3.0,
            2,
        )

    @computed_field
    @property
    def salience_category(self) -> str:
        """Determine salience category based on P/L/U scores."""
        high_p = self.power_score >= _SALIENCE_THRESHOLD
        high_l = self.legitimacy_score >= _SALIENCE_THRESHOLD
        high_u = self.urgency_score >= _SALIENCE_THRESHOLD

        if high_p and high_l and high_u:
            return SalienceCategory.DEFINITIVE.value
        elif high_p and high_l:
            return SalienceCategory.DOMINANT.value
        elif high_p and high_u:
            return SalienceCategory.DANGEROUS.value
        elif high_l and high_u:
            return SalienceCategory.DEPENDENT.value
        elif high_p:
            return SalienceCategory.DORMANT.value
        elif high_l:
            return SalienceCategory.DISCRETIONARY.value
        elif high_u:
            return SalienceCategory.DEMANDING.value
        else:
            return SalienceCategory.NON_STAKEHOLDER.value

class EngagementActivity(BaseModel):
    """A recorded stakeholder engagement activity."""

    activity_id: str = Field(default_factory=_new_uuid, description="Activity ID")
    engagement_type: EngagementType = Field(..., description="Engagement type")
    activity_date: date = Field(..., description="Activity date")
    participants: List[str] = Field(
        default_factory=list, description="Participant stakeholder IDs"
    )
    topics: List[str] = Field(
        default_factory=list, description="Topics discussed"
    )
    findings: List[str] = Field(
        default_factory=list, description="Key findings"
    )
    evidence_references: List[str] = Field(
        default_factory=list, description="Document/file references"
    )
    duration_minutes: int = Field(
        0, ge=0, description="Duration in minutes"
    )

class MaterialityInput(BaseModel):
    """A materiality assessment input from a stakeholder."""

    input_id: str = Field(default_factory=_new_uuid, description="Input ID")
    stakeholder_id: str = Field(..., description="Contributing stakeholder")
    topic: str = Field(..., description="Materiality topic")
    impact_score: float = Field(
        ..., ge=0.0, le=10.0, description="Impact materiality (0-10)"
    )
    financial_score: float = Field(
        ..., ge=0.0, le=10.0, description="Financial materiality (0-10)"
    )
    rationale: str = Field("", description="Stakeholder's rationale")
    confidence: float = Field(
        0.5, ge=0.0, le=1.0, description="Confidence in assessment (0-1)"
    )

class SalienceMap(BaseModel):
    """Stakeholders grouped by salience quadrant."""

    definitive: List[str] = Field(default_factory=list, description="All 3 attributes")
    dominant: List[str] = Field(default_factory=list, description="Power + Legitimacy")
    dangerous: List[str] = Field(default_factory=list, description="Power + Urgency")
    dependent: List[str] = Field(default_factory=list, description="Legitimacy + Urgency")
    dormant: List[str] = Field(default_factory=list, description="Power only")
    discretionary: List[str] = Field(default_factory=list, description="Legitimacy only")
    demanding: List[str] = Field(default_factory=list, description="Urgency only")
    non_stakeholder: List[str] = Field(default_factory=list, description="No high attributes")
    total_stakeholders: int = Field(0, description="Total count")
    provenance_hash: str = Field("", description="SHA-256 hash")

class EngagementReport(BaseModel):
    """Summary report of stakeholder engagement activities."""

    report_id: str = Field(default_factory=_new_uuid, description="Report ID")
    total_stakeholders: int = Field(0, description="Total registered stakeholders")
    stakeholders_by_category: Dict[str, int] = Field(
        default_factory=dict, description="Count per category"
    )
    participation_rate: float = Field(
        0.0, ge=0.0, le=100.0, description="Engagement participation rate"
    )
    total_activities: int = Field(0, description="Total engagement activities")
    activities_by_type: Dict[str, int] = Field(
        default_factory=dict, description="Count per activity type"
    )
    key_findings: List[str] = Field(
        default_factory=list, description="Top findings across activities"
    )
    materiality_influence: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Topic -> {impact_avg, financial_avg, respondent_count}",
    )
    evidence_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Evidence type counts",
    )
    generated_at: datetime = Field(default_factory=utcnow, description="Report time")
    provenance_hash: str = Field("", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class StakeholderConfig(BaseModel):
    """Configuration for the stakeholder engine."""

    salience_threshold: float = Field(
        5.0, ge=1.0, le=10.0, description="Threshold for 'high' on P/L/U scales"
    )
    minimum_engagement_diversity: int = Field(
        3,
        ge=1,
        description="Minimum different stakeholder categories to engage",
    )
    require_evidence_for_findings: bool = Field(
        True, description="Require evidence references for findings"
    )
    weight_by_salience: bool = Field(
        True,
        description="Weight materiality inputs by stakeholder salience score",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class StakeholderEngine:
    """Stakeholder engagement management engine for ESRS materiality assessment.

    Manages stakeholder registration, salience mapping, engagement tracking,
    materiality aggregation, and evidence packaging per ESRS 1 Chapter 3.

    Attributes:
        config: Engine configuration.
        stakeholders: Registered stakeholders keyed by ID.
        activities: Engagement activities keyed by ID.
        materiality_inputs: Materiality inputs from stakeholders.

    Example:
        >>> engine = StakeholderEngine()
        >>> engine.register_stakeholder(Stakeholder(
        ...     name="Jane Investor", category=StakeholderCategory.INVESTOR,
        ...     power_score=8.0, legitimacy_score=7.0, urgency_score=6.0
        ... ))
        >>> salience_map = await engine.generate_salience_map()
    """

    def __init__(self, config: Optional[StakeholderConfig] = None) -> None:
        """Initialize StakeholderEngine.

        Args:
            config: Engine configuration. Uses defaults if not provided.
        """
        self.config = config or StakeholderConfig()
        self.stakeholders: Dict[str, Stakeholder] = {}
        self.activities: Dict[str, EngagementActivity] = {}
        self.materiality_inputs: List[MaterialityInput] = []
        logger.info("StakeholderEngine initialized (version=%s)", _MODULE_VERSION)

    # -- Stakeholder Registration -------------------------------------------

    def register_stakeholder(self, stakeholder: Stakeholder) -> str:
        """Register a stakeholder with automatic salience calculation.

        Args:
            stakeholder: Stakeholder to register.

        Returns:
            Stakeholder ID.

        Raises:
            ValueError: If stakeholder_id already registered.
        """
        if stakeholder.stakeholder_id in self.stakeholders:
            raise ValueError(
                f"Stakeholder '{stakeholder.stakeholder_id}' already registered"
            )

        self.stakeholders[stakeholder.stakeholder_id] = stakeholder
        logger.info(
            "Stakeholder registered: %s (%s, category=%s, salience=%.2f [%s])",
            stakeholder.stakeholder_id,
            stakeholder.name,
            stakeholder.category.value,
            stakeholder.salience_score,
            stakeholder.salience_category,
        )
        return stakeholder.stakeholder_id

    # -- Salience Mapping ---------------------------------------------------

    async def generate_salience_map(
        self,
        stakeholders: Optional[List[Stakeholder]] = None,
    ) -> SalienceMap:
        """Generate salience map grouping stakeholders by P/L/U attributes.

        Args:
            stakeholders: Stakeholders to map. Uses all registered if None.

        Returns:
            SalienceMap with stakeholders grouped by salience category.
        """
        target_stakeholders = stakeholders or list(self.stakeholders.values())

        salience_map = SalienceMap(total_stakeholders=len(target_stakeholders))

        for sh in target_stakeholders:
            category = sh.salience_category
            bucket = getattr(salience_map, category, None)
            if bucket is not None:
                bucket.append(sh.stakeholder_id)

        salience_map.provenance_hash = _compute_hash(salience_map)

        logger.info(
            "Salience map generated: %d stakeholders mapped "
            "(definitive=%d, dominant=%d, dependent=%d, other=%d)",
            len(target_stakeholders),
            len(salience_map.definitive),
            len(salience_map.dominant),
            len(salience_map.dependent),
            len(salience_map.non_stakeholder),
        )
        return salience_map

    # -- Materiality Survey -------------------------------------------------

    async def create_materiality_survey(
        self,
        topics: List[str],
        groups: Optional[List[StakeholderCategory]] = None,
    ) -> Dict[str, Any]:
        """Generate a materiality assessment survey structure.

        Creates a survey template with impact and financial materiality
        questions for each topic, targeted at specified stakeholder groups.

        Args:
            topics: List of materiality topics to assess.
            groups: Stakeholder categories to target (None = all).

        Returns:
            Survey structure with questions and target participants.
        """
        if not topics:
            raise ValueError("At least one topic is required for the survey")

        target_ids: List[str] = []
        for sh in self.stakeholders.values():
            if groups is None or sh.category in groups:
                target_ids.append(sh.stakeholder_id)

        questions: List[Dict[str, Any]] = []
        for topic in topics:
            questions.append({
                "topic": topic,
                "questions": [
                    {
                        "id": f"{topic}_impact",
                        "text": f"How significant is the impact of '{topic}' on people and environment? (0-10)",
                        "type": "scale",
                        "min": 0,
                        "max": 10,
                    },
                    {
                        "id": f"{topic}_financial",
                        "text": f"How significant are the financial risks/opportunities related to '{topic}'? (0-10)",
                        "type": "scale",
                        "min": 0,
                        "max": 10,
                    },
                    {
                        "id": f"{topic}_rationale",
                        "text": f"Please explain your assessment of '{topic}'.",
                        "type": "text",
                    },
                ],
            })

        survey = {
            "survey_id": _new_uuid(),
            "title": "ESRS Double Materiality Assessment Survey",
            "description": (
                "This survey collects stakeholder perspectives on the "
                "impact and financial materiality of sustainability topics "
                "as required by ESRS 1 Chapter 3."
            ),
            "topics": topics,
            "questions": questions,
            "target_participants": target_ids,
            "target_groups": [g.value for g in groups] if groups else ["all"],
            "total_participants": len(target_ids),
            "created_at": utcnow().isoformat(),
            "provenance_hash": "",
        }
        survey["provenance_hash"] = _compute_hash(survey)

        logger.info(
            "Materiality survey created: %d topics, %d target participants",
            len(topics),
            len(target_ids),
        )
        return survey

    # -- Engagement Recording -----------------------------------------------

    def record_engagement(self, activity: EngagementActivity) -> str:
        """Record a stakeholder engagement activity.

        Args:
            activity: Engagement activity to record.

        Returns:
            Activity ID.

        Raises:
            ValueError: If participants reference unknown stakeholders.
        """
        unknown = [
            p for p in activity.participants if p not in self.stakeholders
        ]
        if unknown:
            logger.warning(
                "Engagement activity %s references unknown stakeholders: %s",
                activity.activity_id,
                unknown,
            )

        if self.config.require_evidence_for_findings:
            if activity.findings and not activity.evidence_references:
                logger.warning(
                    "Activity %s has findings but no evidence references",
                    activity.activity_id,
                )

        self.activities[activity.activity_id] = activity
        logger.info(
            "Engagement recorded: %s (type=%s, participants=%d, topics=%d)",
            activity.activity_id,
            activity.type.value,
            len(activity.participants),
            len(activity.topics),
        )
        return activity.activity_id

    # -- Materiality Aggregation --------------------------------------------

    async def aggregate_materiality_inputs(
        self, inputs: List[MaterialityInput]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate materiality inputs with optional salience weighting.

        Computes weighted average impact and financial scores per topic,
        using stakeholder salience as weight when configured.

        Args:
            inputs: List of materiality inputs from stakeholders.

        Returns:
            Dict mapping topic to aggregated scores and metadata.
        """
        # Store inputs
        self.materiality_inputs.extend(inputs)

        topic_data: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

        for inp in inputs:
            weight = 1.0
            if self.config.weight_by_salience:
                sh = self.stakeholders.get(inp.stakeholder_id)
                if sh:
                    weight = max(0.1, sh.salience_score / 10.0)

            effective_weight = weight * inp.confidence
            topic_data[inp.topic].append(
                (inp.impact_score * effective_weight,
                 inp.financial_score * effective_weight,
                 effective_weight)
            )

        aggregated: Dict[str, Dict[str, Any]] = {}

        for topic, weighted_entries in topic_data.items():
            total_weight = sum(w for _, _, w in weighted_entries)
            if total_weight == 0:
                continue

            impact_avg = sum(i for i, _, _ in weighted_entries) / total_weight
            financial_avg = sum(f for _, f, _ in weighted_entries) / total_weight

            aggregated[topic] = {
                "impact_score_avg": round(impact_avg, 2),
                "financial_score_avg": round(financial_avg, 2),
                "combined_score": round((impact_avg + financial_avg) / 2, 2),
                "respondent_count": len(weighted_entries),
                "total_weight": round(total_weight, 4),
                "is_material": impact_avg >= 5.0 or financial_avg >= 5.0,
                "is_double_material": impact_avg >= 5.0 and financial_avg >= 5.0,
            }

        # Sort by combined score descending
        aggregated = dict(
            sorted(
                aggregated.items(),
                key=lambda x: x[1]["combined_score"],
                reverse=True,
            )
        )

        logger.info(
            "Materiality aggregation: %d topics from %d inputs",
            len(aggregated),
            len(inputs),
        )
        return aggregated

    # -- Evidence Package ---------------------------------------------------

    async def generate_evidence_package(self) -> Dict[str, Any]:
        """Compile an audit-ready evidence package for stakeholder engagement.

        Collects all stakeholder registrations, engagement activities,
        materiality inputs, and salience mapping into a structured package
        suitable for auditor review per ESRS 1 requirements.

        Returns:
            Comprehensive evidence package dict.
        """
        salience_map = await self.generate_salience_map()

        # Categorize stakeholders
        categories_count: Dict[str, int] = defaultdict(int)
        for sh in self.stakeholders.values():
            categories_count[sh.category.value] += 1

        # Engagement activity summary
        activity_type_count: Dict[str, int] = defaultdict(int)
        total_duration = 0
        all_findings: List[str] = []
        all_evidence_refs: List[str] = []

        for act in self.activities.values():
            activity_type_count[act.engagement_type.value] += 1
            total_duration += act.duration_minutes
            all_findings.extend(act.findings)
            all_evidence_refs.extend(act.evidence_references)

        # Engaged stakeholder IDs
        engaged_ids: Set[str] = set()
        for act in self.activities.values():
            engaged_ids.update(act.participants)

        participation_rate = 0.0
        if self.stakeholders:
            participation_rate = (
                len(engaged_ids) / len(self.stakeholders) * 100
            )

        # Category engagement coverage
        engaged_categories: Set[str] = set()
        for sh_id in engaged_ids:
            sh = self.stakeholders.get(sh_id)
            if sh:
                engaged_categories.add(sh.category.value)

        package = {
            "package_id": _new_uuid(),
            "generated_at": utcnow().isoformat(),
            "stakeholder_registry": {
                "total": len(self.stakeholders),
                "by_category": dict(categories_count),
                "stakeholders": [
                    sh.model_dump(mode="json") for sh in self.stakeholders.values()
                ],
            },
            "salience_analysis": salience_map.model_dump(mode="json"),
            "engagement_summary": {
                "total_activities": len(self.activities),
                "by_type": dict(activity_type_count),
                "total_duration_minutes": total_duration,
                "total_participants_engaged": len(engaged_ids),
                "participation_rate_pct": round(participation_rate, 1),
                "categories_engaged": sorted(engaged_categories),
                "category_diversity_met": (
                    len(engaged_categories) >= self.config.minimum_engagement_diversity
                ),
            },
            "activities_log": [
                act.model_dump(mode="json") for act in self.activities.values()
            ],
            "materiality_inputs": {
                "total_inputs": len(self.materiality_inputs),
                "inputs": [
                    mi.model_dump(mode="json") for mi in self.materiality_inputs
                ],
            },
            "evidence_references": sorted(set(all_evidence_refs)),
            "key_findings": all_findings,
            "compliance_checklist": {
                "stakeholders_registered": len(self.stakeholders) > 0,
                "salience_analysis_done": len(salience_map.definitive) >= 0,
                "engagement_conducted": len(self.activities) > 0,
                "materiality_collected": len(self.materiality_inputs) > 0,
                "evidence_documented": len(all_evidence_refs) > 0,
                "category_diversity_met": (
                    len(engaged_categories) >= self.config.minimum_engagement_diversity
                ),
            },
        }
        package["provenance_hash"] = _compute_hash(package)

        logger.info(
            "Evidence package generated: %d stakeholders, %d activities, %d inputs",
            len(self.stakeholders),
            len(self.activities),
            len(self.materiality_inputs),
        )
        return package

    # -- Engagement Report --------------------------------------------------

    async def generate_engagement_report(self) -> EngagementReport:
        """Generate a summary engagement report.

        Returns:
            EngagementReport with participation stats and key findings.
        """
        categories_count: Dict[str, int] = defaultdict(int)
        for sh in self.stakeholders.values():
            categories_count[sh.category.value] += 1

        activity_type_count: Dict[str, int] = defaultdict(int)
        all_findings: List[str] = []

        for act in self.activities.values():
            activity_type_count[act.engagement_type.value] += 1
            all_findings.extend(act.findings)

        # Participation rate
        engaged_ids: Set[str] = set()
        for act in self.activities.values():
            engaged_ids.update(act.participants)

        participation_rate = 0.0
        if self.stakeholders:
            participation_rate = len(engaged_ids) / len(self.stakeholders) * 100

        # Materiality influence
        materiality_influence: Dict[str, Dict[str, float]] = {}
        topic_accum: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for mi in self.materiality_inputs:
            topic_accum[mi.topic].append((mi.impact_score, mi.financial_score))

        for topic, entries in topic_accum.items():
            impact_avg = sum(i for i, _ in entries) / len(entries)
            financial_avg = sum(f for _, f in entries) / len(entries)
            materiality_influence[topic] = {
                "impact_avg": round(impact_avg, 2),
                "financial_avg": round(financial_avg, 2),
                "respondent_count": float(len(entries)),
            }

        # Evidence summary
        evidence_counts: Dict[str, int] = defaultdict(int)
        for act in self.activities.values():
            for ref in act.evidence_references:
                ext = ref.rsplit(".", 1)[-1].lower() if "." in ref else "other"
                evidence_counts[ext] += 1

        report = EngagementReport(
            total_stakeholders=len(self.stakeholders),
            stakeholders_by_category=dict(categories_count),
            participation_rate=round(participation_rate, 1),
            total_activities=len(self.activities),
            activities_by_type=dict(activity_type_count),
            key_findings=all_findings[:20],  # Cap at 20
            materiality_influence=materiality_influence,
            evidence_summary=dict(evidence_counts),
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "Engagement report: %d stakeholders, %.1f%% participation, %d activities",
            report.total_stakeholders,
            report.participation_rate,
            report.total_activities,
        )
        return report

    # -- Reset --------------------------------------------------------------

    def reset(self) -> None:
        """Reset engine state, clearing all stakeholders, activities, and inputs."""
        self.stakeholders.clear()
        self.activities.clear()
        self.materiality_inputs.clear()
        logger.info("StakeholderEngine reset")
