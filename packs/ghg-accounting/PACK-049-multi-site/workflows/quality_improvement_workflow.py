# -*- coding: utf-8 -*-
"""
Quality Improvement Workflow
====================================

5-phase workflow for multi-site GHG data quality improvement covering
quality assessment, gap identification, remediation planning,
implementation tracking, and verification within PACK-049 Multi-Site Mgmt.

Phases:
    1. QualityAssess        -- Score each site across 6 quality dimensions
                               (completeness, accuracy, consistency,
                               transparency, timeliness, relevance).
    2. GapIdentify          -- Identify sites below quality threshold per
                               dimension and overall.
    3. RemediationPlan      -- Generate prioritised remediation actions with
                               effort, owner, and timeline estimates.
    4. Implementation       -- Track implementation progress of remediation.
    5. Verification         -- Re-assess quality and verify improvement.

Regulatory Basis:
    GHG Protocol (Ch. 7) -- Data quality management
    ISO 14064-1:2018 (Cl. 6) -- Quality management
    CSRD / ESRS E1 -- Data quality disclosure
    PCAF (2022) -- Data quality scoring framework

Author: GreenLang Team
Version: 49.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class QualityPhase(str, Enum):
    QUALITY_ASSESS = "quality_assess"
    GAP_IDENTIFY = "gap_identify"
    REMEDIATION_PLAN = "remediation_plan"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"

class QualityDimension(str, Enum):
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TRANSPARENCY = "transparency"
    TIMELINESS = "timeliness"
    RELEVANCE = "relevance"

class QualityTier(str, Enum):
    """PCAF-aligned data quality tier (1=best, 5=worst)."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    TIER_5 = "tier_5"

class RemediationPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RemediationStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"

class ImprovementVerdict(str, Enum):
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    DEGRADED = "degraded"

# =============================================================================
# REFERENCE DATA
# =============================================================================

DIMENSION_WEIGHTS: Dict[str, Decimal] = {
    "completeness": Decimal("0.25"),
    "accuracy": Decimal("0.25"),
    "consistency": Decimal("0.15"),
    "transparency": Decimal("0.15"),
    "timeliness": Decimal("0.10"),
    "relevance": Decimal("0.10"),
}

QUALITY_THRESHOLD = Decimal("70")  # Minimum acceptable score per dimension
OVERALL_THRESHOLD = Decimal("75")  # Minimum acceptable overall score

TIER_SCORE_RANGES: Dict[str, Tuple[Decimal, Decimal]] = {
    "tier_1": (Decimal("90"), Decimal("100")),
    "tier_2": (Decimal("75"), Decimal("89.99")),
    "tier_3": (Decimal("60"), Decimal("74.99")),
    "tier_4": (Decimal("40"), Decimal("59.99")),
    "tier_5": (Decimal("0"), Decimal("39.99")),
}

REMEDIATION_EFFORT_MAP: Dict[str, Dict[str, Any]] = {
    "completeness": {"typical_hours": 20, "description": "Fill data gaps, install sub-meters"},
    "accuracy": {"typical_hours": 30, "description": "Calibrate meters, cross-check invoices"},
    "consistency": {"typical_hours": 15, "description": "Standardise units, align methodologies"},
    "transparency": {"typical_hours": 10, "description": "Document sources, add evidence refs"},
    "timeliness": {"typical_hours": 10, "description": "Set up automated data feeds"},
    "relevance": {"typical_hours": 15, "description": "Use site-specific factors, update proxies"},
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    phase_name: str = Field(...)
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class SiteDimensionScore(BaseModel):
    """Score for a single quality dimension at a site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dimension: QualityDimension = Field(...)
    score: Decimal = Field(Decimal("0"), ge=Decimal("0"), le=Decimal("100"))
    max_score: Decimal = Field(Decimal("100"))
    weight: Decimal = Field(Decimal("0"))
    weighted_score: Decimal = Field(Decimal("0"))
    findings: List[str] = Field(default_factory=list)

class SiteQualityAssessment(BaseModel):
    """Quality assessment result for a single site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field(...)
    site_name: str = Field("")
    dimension_scores: List[SiteDimensionScore] = Field(default_factory=list)
    overall_score: Decimal = Field(Decimal("0"))
    quality_tier: QualityTier = Field(QualityTier.TIER_3)
    meets_threshold: bool = Field(False)
    dimensions_below_threshold: List[str] = Field(default_factory=list)

class QualityGap(BaseModel):
    """A quality gap identified at a site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    gap_id: str = Field(default_factory=_new_uuid)
    site_id: str = Field(...)
    site_name: str = Field("")
    dimension: QualityDimension = Field(...)
    current_score: Decimal = Field(Decimal("0"))
    target_score: Decimal = Field(Decimal("70"))
    gap_points: Decimal = Field(Decimal("0"))
    priority: RemediationPriority = Field(RemediationPriority.MEDIUM)
    impact_description: str = Field("")

class RemediationAction(BaseModel):
    """A remediation action for a quality gap."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    action_id: str = Field(default_factory=_new_uuid)
    gap_id: str = Field("")
    site_id: str = Field(...)
    site_name: str = Field("")
    dimension: QualityDimension = Field(...)
    action_description: str = Field("")
    priority: RemediationPriority = Field(RemediationPriority.MEDIUM)
    estimated_hours: int = Field(0)
    assigned_to: str = Field("")
    target_date: str = Field("")
    status: RemediationStatus = Field(RemediationStatus.NOT_STARTED)
    expected_score_improvement: Decimal = Field(Decimal("0"))

class ImplementationProgress(BaseModel):
    """Progress tracking for remediation actions."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field(...)
    total_actions: int = Field(0)
    completed: int = Field(0)
    in_progress: int = Field(0)
    not_started: int = Field(0)
    completion_pct: Decimal = Field(Decimal("0"))

class VerificationResult(BaseModel):
    """Verification result after remediation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field(...)
    site_name: str = Field("")
    previous_score: Decimal = Field(Decimal("0"))
    current_score: Decimal = Field(Decimal("0"))
    improvement: Decimal = Field(Decimal("0"))
    verdict: ImprovementVerdict = Field(ImprovementVerdict.UNCHANGED)
    new_tier: QualityTier = Field(QualityTier.TIER_3)
    provenance_hash: str = Field("")

class QualityImprovementInput(BaseModel):
    """Input for the quality improvement workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    organisation_id: str = Field(...)
    reporting_year: int = Field(...)
    site_quality_data: List[Dict[str, Any]] = Field(default_factory=list)
    remediation_overrides: List[Dict[str, Any]] = Field(default_factory=list)
    prior_assessments: List[Dict[str, Any]] = Field(default_factory=list)
    quality_threshold: Decimal = Field(QUALITY_THRESHOLD)
    overall_threshold: Decimal = Field(OVERALL_THRESHOLD)
    skip_phases: List[str] = Field(default_factory=list)

class QualityImprovementResult(BaseModel):
    """Output from the quality improvement workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    workflow_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    assessments: List[SiteQualityAssessment] = Field(default_factory=list)
    gaps: List[QualityGap] = Field(default_factory=list)
    remediation_actions: List[RemediationAction] = Field(default_factory=list)
    implementation_progress: List[ImplementationProgress] = Field(default_factory=list)
    verifications: List[VerificationResult] = Field(default_factory=list)
    corporate_score: Decimal = Field(Decimal("0"))
    sites_below_threshold: int = Field(0)
    total_gaps: int = Field(0)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")

# =============================================================================
# WORKFLOW CLASS
# =============================================================================

class QualityImprovementWorkflow:
    """
    5-phase quality improvement workflow for multi-site GHG inventories.

    Assesses data quality across 6 dimensions, identifies gaps,
    generates remediation plans, tracks implementation, and verifies
    improvement with full SHA-256 provenance.

    Example:
        >>> wf = QualityImprovementWorkflow()
        >>> inp = QualityImprovementInput(
        ...     organisation_id="ORG-001", reporting_year=2025,
        ...     site_quality_data=[{
        ...         "site_id": "S1", "completeness": 85, "accuracy": 70,
        ...     }],
        ... )
        >>> result = wf.execute(inp)
    """

    PHASE_ORDER: List[QualityPhase] = [
        QualityPhase.QUALITY_ASSESS,
        QualityPhase.GAP_IDENTIFY,
        QualityPhase.REMEDIATION_PLAN,
        QualityPhase.IMPLEMENTATION,
        QualityPhase.VERIFICATION,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._assessments: Dict[str, SiteQualityAssessment] = {}
        self._gaps: List[QualityGap] = []
        self._actions: List[RemediationAction] = []

    def execute(self, input_data: QualityImprovementInput) -> QualityImprovementResult:
        start = utcnow()
        result = QualityImprovementResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING, started_at=start.isoformat(),
        )

        phase_methods = {
            QualityPhase.QUALITY_ASSESS: self._phase_quality_assess,
            QualityPhase.GAP_IDENTIFY: self._phase_gap_identify,
            QualityPhase.REMEDIATION_PLAN: self._phase_remediation_plan,
            QualityPhase.IMPLEMENTATION: self._phase_implementation,
            QualityPhase.VERIFICATION: self._phase_verification,
        }

        for idx, phase in enumerate(self.PHASE_ORDER, 1):
            if phase.value in input_data.skip_phases:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx, status=PhaseStatus.SKIPPED,
                ))
                continue
            phase_start = utcnow()
            try:
                phase_out = phase_methods[phase](input_data, result)
                elapsed = (utcnow() - phase_start).total_seconds()
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                    outputs=phase_out, provenance_hash=_compute_hash(str(phase_out)),
                ))
            except Exception as exc:
                elapsed = (utcnow() - phase_start).total_seconds()
                logger.error("Phase %s failed: %s", phase.value, exc, exc_info=True)
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.FAILED, duration_seconds=elapsed, errors=[str(exc)],
                ))
                result.status = WorkflowStatus.FAILED
                result.errors.append(f"Phase {phase.value}: {exc}")
                break

        if result.status != WorkflowStatus.FAILED:
            result.status = WorkflowStatus.COMPLETED
        end = utcnow()
        result.completed_at = end.isoformat()
        result.duration_seconds = (end - start).total_seconds()
        result.provenance_hash = _compute_hash(
            f"{result.workflow_id}|{result.organisation_id}|"
            f"{float(result.corporate_score)}|{result.completed_at}"
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- QUALITY ASSESS
    # -----------------------------------------------------------------

    def _phase_quality_assess(
        self, input_data: QualityImprovementInput, result: QualityImprovementResult,
    ) -> Dict[str, Any]:
        """Score each site across 6 quality dimensions."""
        logger.info("Phase 1 -- Quality Assess: %d sites", len(input_data.site_quality_data))
        assessments: Dict[str, SiteQualityAssessment] = {}

        for raw in input_data.site_quality_data:
            sid = raw.get("site_id", _new_uuid())
            sname = raw.get("site_name", "")

            dim_scores: List[SiteDimensionScore] = []
            overall = Decimal("0")
            dims_below: List[str] = []

            for dim in QualityDimension:
                raw_score = self._dec(raw.get(dim.value, "50"))
                raw_score = max(Decimal("0"), min(raw_score, Decimal("100")))
                weight = DIMENSION_WEIGHTS.get(dim.value, Decimal("0.10"))
                weighted = (raw_score * weight).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                overall += weighted

                findings: List[str] = []
                if raw_score < input_data.quality_threshold:
                    dims_below.append(dim.value)
                    findings.append(
                        f"{dim.value} score {raw_score} below threshold {input_data.quality_threshold}"
                    )

                dim_scores.append(SiteDimensionScore(
                    dimension=dim, score=raw_score,
                    weight=weight, weighted_score=weighted,
                    findings=findings,
                ))

            overall = overall.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            tier = self._score_to_tier(overall)

            assessment = SiteQualityAssessment(
                site_id=sid, site_name=sname,
                dimension_scores=dim_scores,
                overall_score=overall,
                quality_tier=tier,
                meets_threshold=overall >= input_data.overall_threshold,
                dimensions_below_threshold=dims_below,
            )
            assessments[sid] = assessment

        self._assessments = assessments
        result.assessments = list(assessments.values())

        below_count = sum(1 for a in assessments.values() if not a.meets_threshold)
        result.sites_below_threshold = below_count

        # Corporate score = weighted average of all site scores
        if assessments:
            corp = sum(a.overall_score for a in assessments.values()) / Decimal(str(len(assessments)))
            result.corporate_score = corp.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        tier_dist: Dict[str, int] = {}
        for a in assessments.values():
            tier_dist[a.quality_tier.value] = tier_dist.get(a.quality_tier.value, 0) + 1

        logger.info("Assessed %d sites, %d below threshold, corp score %.1f",
                     len(assessments), below_count, float(result.corporate_score))
        return {
            "sites_assessed": len(assessments),
            "sites_below_threshold": below_count,
            "corporate_score": float(result.corporate_score),
            "tier_distribution": tier_dist,
        }

    def _score_to_tier(self, score: Decimal) -> QualityTier:
        for tier_name, (low, high) in TIER_SCORE_RANGES.items():
            if low <= score <= high:
                return QualityTier(tier_name)
        return QualityTier.TIER_5

    # -----------------------------------------------------------------
    # PHASE 2 -- GAP IDENTIFY
    # -----------------------------------------------------------------

    def _phase_gap_identify(
        self, input_data: QualityImprovementInput, result: QualityImprovementResult,
    ) -> Dict[str, Any]:
        """Identify sites below quality threshold per dimension."""
        logger.info("Phase 2 -- Gap Identify")
        gaps: List[QualityGap] = []

        for sid, assessment in self._assessments.items():
            for ds in assessment.dimension_scores:
                if ds.score < input_data.quality_threshold:
                    gap_points = (input_data.quality_threshold - ds.score).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )

                    if gap_points >= Decimal("30"):
                        priority = RemediationPriority.CRITICAL
                    elif gap_points >= Decimal("20"):
                        priority = RemediationPriority.HIGH
                    elif gap_points >= Decimal("10"):
                        priority = RemediationPriority.MEDIUM
                    else:
                        priority = RemediationPriority.LOW

                    gap = QualityGap(
                        site_id=sid,
                        site_name=assessment.site_name,
                        dimension=ds.dimension,
                        current_score=ds.score,
                        target_score=input_data.quality_threshold,
                        gap_points=gap_points,
                        priority=priority,
                        impact_description=(
                            f"Site {assessment.site_name} scores {ds.score} on "
                            f"{ds.dimension.value}, {gap_points} points below target "
                            f"{input_data.quality_threshold}"
                        ),
                    )
                    gaps.append(gap)

        self._gaps = gaps
        result.gaps = gaps
        result.total_gaps = len(gaps)

        priority_dist: Dict[str, int] = {}
        for g in gaps:
            priority_dist[g.priority.value] = priority_dist.get(g.priority.value, 0) + 1

        logger.info("Identified %d quality gaps", len(gaps))
        return {
            "total_gaps": len(gaps),
            "priority_distribution": priority_dist,
            "sites_with_gaps": len(set(g.site_id for g in gaps)),
        }

    # -----------------------------------------------------------------
    # PHASE 3 -- REMEDIATION PLAN
    # -----------------------------------------------------------------

    def _phase_remediation_plan(
        self, input_data: QualityImprovementInput, result: QualityImprovementResult,
    ) -> Dict[str, Any]:
        """Generate prioritised remediation actions."""
        logger.info("Phase 3 -- Remediation Plan")
        actions: List[RemediationAction] = []

        override_map: Dict[str, Dict[str, Any]] = {}
        for ov in input_data.remediation_overrides:
            key = f"{ov.get('site_id', '')}|{ov.get('dimension', '')}"
            override_map[key] = ov

        for gap in self._gaps:
            override_key = f"{gap.site_id}|{gap.dimension.value}"
            override = override_map.get(override_key, {})

            effort_info = REMEDIATION_EFFORT_MAP.get(gap.dimension.value, {})
            base_hours = effort_info.get("typical_hours", 20)
            base_desc = effort_info.get("description", "Improve data quality")

            # Scale hours by gap size
            scale_factor = float(gap.gap_points) / 20.0
            estimated_hours = max(int(base_hours * max(scale_factor, 0.5)), 4)

            expected_improvement = min(
                gap.gap_points,
                Decimal("25"),  # Max realistic improvement per cycle
            )

            action = RemediationAction(
                gap_id=gap.gap_id,
                site_id=gap.site_id,
                site_name=gap.site_name,
                dimension=gap.dimension,
                action_description=override.get("action_description", base_desc),
                priority=gap.priority,
                estimated_hours=override.get("estimated_hours", estimated_hours),
                assigned_to=override.get("assigned_to", ""),
                target_date=override.get("target_date", ""),
                status=RemediationStatus.NOT_STARTED,
                expected_score_improvement=expected_improvement,
            )
            actions.append(action)

        self._actions = actions
        result.remediation_actions = actions

        total_hours = sum(a.estimated_hours for a in actions)
        logger.info("Generated %d remediation actions, %d total hours",
                     len(actions), total_hours)
        return {
            "actions_generated": len(actions),
            "total_estimated_hours": total_hours,
        }

    # -----------------------------------------------------------------
    # PHASE 4 -- IMPLEMENTATION
    # -----------------------------------------------------------------

    def _phase_implementation(
        self, input_data: QualityImprovementInput, result: QualityImprovementResult,
    ) -> Dict[str, Any]:
        """Track implementation progress of remediation actions."""
        logger.info("Phase 4 -- Implementation")
        progress_by_site: Dict[str, ImplementationProgress] = {}

        for action in self._actions:
            sid = action.site_id
            if sid not in progress_by_site:
                progress_by_site[sid] = ImplementationProgress(site_id=sid)

            prog = progress_by_site[sid]
            prog.total_actions += 1

            if action.status == RemediationStatus.COMPLETED:
                prog.completed += 1
            elif action.status == RemediationStatus.IN_PROGRESS:
                prog.in_progress += 1
            else:
                prog.not_started += 1

        for prog in progress_by_site.values():
            if prog.total_actions > 0:
                prog.completion_pct = (
                    Decimal(str(prog.completed)) / Decimal(str(prog.total_actions)) * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        result.implementation_progress = list(progress_by_site.values())

        total_actions = sum(p.total_actions for p in progress_by_site.values())
        total_completed = sum(p.completed for p in progress_by_site.values())

        logger.info("Implementation: %d/%d actions completed", total_completed, total_actions)
        return {
            "sites_tracked": len(progress_by_site),
            "total_actions": total_actions,
            "completed": total_completed,
            "overall_completion_pct": (
                float(Decimal(str(total_completed)) / Decimal(str(max(total_actions, 1))) * Decimal("100"))
            ),
        }

    # -----------------------------------------------------------------
    # PHASE 5 -- VERIFICATION
    # -----------------------------------------------------------------

    def _phase_verification(
        self, input_data: QualityImprovementInput, result: QualityImprovementResult,
    ) -> Dict[str, Any]:
        """Re-assess quality and verify improvement."""
        logger.info("Phase 5 -- Verification")
        verifications: List[VerificationResult] = []

        prior_lookup: Dict[str, Decimal] = {}
        for rec in input_data.prior_assessments:
            sid = rec.get("site_id", "")
            prior_lookup[sid] = self._dec(rec.get("overall_score", "0"))

        for sid, assessment in self._assessments.items():
            previous = prior_lookup.get(sid, Decimal("0"))
            current = assessment.overall_score
            improvement = (current - previous).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            if improvement > Decimal("2"):
                verdict = ImprovementVerdict.IMPROVED
            elif improvement < Decimal("-2"):
                verdict = ImprovementVerdict.DEGRADED
            else:
                verdict = ImprovementVerdict.UNCHANGED

            prov = _compute_hash(f"{sid}|{float(previous)}|{float(current)}|{verdict.value}")

            verifications.append(VerificationResult(
                site_id=sid,
                site_name=assessment.site_name,
                previous_score=previous,
                current_score=current,
                improvement=improvement,
                verdict=verdict,
                new_tier=assessment.quality_tier,
                provenance_hash=prov,
            ))

        result.verifications = verifications

        improved = sum(1 for v in verifications if v.verdict == ImprovementVerdict.IMPROVED)
        degraded = sum(1 for v in verifications if v.verdict == ImprovementVerdict.DEGRADED)

        logger.info("Verification: %d improved, %d degraded, %d unchanged",
                     improved, degraded, len(verifications) - improved - degraded)
        return {
            "sites_verified": len(verifications),
            "improved": improved,
            "unchanged": len(verifications) - improved - degraded,
            "degraded": degraded,
        }

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    def _dec(self, value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal("0")

__all__ = [
    "QualityImprovementWorkflow",
    "QualityImprovementInput",
    "QualityImprovementResult",
    "QualityPhase",
    "QualityDimension",
    "QualityTier",
    "RemediationPriority",
    "RemediationStatus",
    "ImprovementVerdict",
    "SiteDimensionScore",
    "SiteQualityAssessment",
    "QualityGap",
    "RemediationAction",
    "ImplementationProgress",
    "VerificationResult",
    "PhaseResult",
    "PhaseStatus",
    "WorkflowStatus",
]
