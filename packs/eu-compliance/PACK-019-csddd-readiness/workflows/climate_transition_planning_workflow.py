# -*- coding: utf-8 -*-
"""
CSDDD Climate Transition Planning Workflow
===============================================

4-phase workflow for developing and assessing climate transition plans under
the EU Corporate Sustainability Due Diligence Directive (CSDDD / CS3D).
Covers baseline assessment, target setting, pathway design, and progress
tracking aligned with the Paris Agreement 1.5C objective.

Phases:
    1. BaselineAssessment      -- Assess current emissions and climate posture
    2. TargetSetting           -- Evaluate target alignment with Paris Agreement
    3. PathwayDesign           -- Design decarbonisation pathway with milestones
    4. ProgressTracking        -- Track progress against transition plan targets

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Art. 15: Combating climate change
    - Art. 15(1): Adopt and implement a climate transition plan
    - Art. 15(2): Paris Agreement alignment -- limit to 1.5C
    - Art. 15(3): Time-bound targets for 2030 and five-year intervals to 2050
    - ESRS E1: Climate change disclosures
    - Science Based Targets initiative (SBTi) methodology alignment

Author: GreenLang Team
Version: 19.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class WorkflowPhase(str, Enum):
    """Phases of the climate transition planning workflow."""
    BASELINE_ASSESSMENT = "baseline_assessment"
    TARGET_SETTING = "target_setting"
    PATHWAY_DESIGN = "pathway_design"
    PROGRESS_TRACKING = "progress_tracking"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EmissionScope(str, Enum):
    """GHG emission scope categories."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class TargetType(str, Enum):
    """Types of climate targets per Art. 15(3)."""
    ABSOLUTE = "absolute"           # Absolute emission reduction
    INTENSITY = "intensity"         # Per unit of revenue/output
    NET_ZERO = "net_zero"           # Net-zero by target year
    CARBON_NEUTRAL = "carbon_neutral"


class TargetAlignment(str, Enum):
    """Target alignment with Paris Agreement."""
    ALIGNED_1_5C = "aligned_1_5c"
    ALIGNED_WELL_BELOW_2C = "aligned_well_below_2c"
    ALIGNED_2C = "aligned_2c"
    NOT_ALIGNED = "not_aligned"
    INSUFFICIENT_DATA = "insufficient_data"


class PathwayStatus(str, Enum):
    """Status of the decarbonisation pathway."""
    ON_TRACK = "on_track"
    SLIGHTLY_OFF = "slightly_off"
    SIGNIFICANTLY_OFF = "significantly_off"
    NO_PATHWAY = "no_pathway"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class EmissionData(BaseModel):
    """Emission data record for baseline and tracking."""
    year: int = Field(..., ge=2015, le=2060)
    scope_1_tco2e: float = Field(default=0.0, ge=0.0)
    scope_2_tco2e: float = Field(default=0.0, ge=0.0)
    scope_3_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    revenue_eur: float = Field(default=0.0, ge=0.0, description="Revenue for intensity calc")
    intensity_tco2e_per_meur: float = Field(default=0.0, ge=0.0)
    data_quality: str = Field(default="estimated", description="measured/calculated/estimated")


class ClimateTarget(BaseModel):
    """Climate target per Art. 15(3)."""
    target_id: str = Field(default_factory=lambda: f"ct-{_new_uuid()[:8]}")
    target_name: str = Field(default="")
    target_type: TargetType = Field(default=TargetType.ABSOLUTE)
    target_year: int = Field(default=2030, ge=2025, le=2060)
    base_year: int = Field(default=2019, ge=2010, le=2030)
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Target % reduction from base year")
    scope_coverage: List[EmissionScope] = Field(
        default_factory=lambda: [EmissionScope.SCOPE_1, EmissionScope.SCOPE_2],
        description="Scopes covered by target"
    )
    includes_scope_3: bool = Field(default=False)
    sbti_validated: bool = Field(default=False, description="Validated by SBTi")
    current_progress_pct: float = Field(default=0.0, ge=0.0, le=200.0)


class TransitionAction(BaseModel):
    """Action within the transition plan pathway."""
    action_id: str = Field(default_factory=lambda: f"ta-{_new_uuid()[:8]}")
    action_name: str = Field(default="")
    category: str = Field(default="", description="energy_efficiency, renewables, etc.")
    expected_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    investment_eur: float = Field(default=0.0, ge=0.0)
    start_year: int = Field(default=2026)
    completion_year: int = Field(default=2030)
    status: str = Field(default="planned", description="planned/in_progress/completed")


class ClimateTransitionPlanningInput(BaseModel):
    """Input data model for ClimateTransitionPlanningWorkflow."""
    entity_id: str = Field(default="", description="Reporting entity ID")
    entity_name: str = Field(default="", description="Reporting entity name")
    reporting_year: int = Field(default=2026, ge=2024, le=2050)
    emission_data: List[EmissionData] = Field(
        default_factory=list, description="Historical and current emission data"
    )
    targets: List[ClimateTarget] = Field(
        default_factory=list, description="Climate targets"
    )
    transition_actions: List[TransitionAction] = Field(
        default_factory=list, description="Planned/in-progress transition actions"
    )
    sector: str = Field(default="", description="Company sector for benchmark")
    has_transition_plan: bool = Field(default=False, description="Whether a formal plan exists")
    config: Dict[str, Any] = Field(default_factory=dict)


class ClimateTransitionPlanningResult(BaseModel):
    """Complete result from climate transition planning workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="climate_transition_planning")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    # Baseline
    baseline_year: int = Field(default=2019)
    baseline_total_tco2e: float = Field(default=0.0, ge=0.0)
    current_total_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_since_baseline_pct: float = Field(default=0.0)
    # Targets
    targets_count: int = Field(default=0, ge=0)
    target_alignment: str = Field(default="insufficient_data")
    sbti_validated_targets: int = Field(default=0, ge=0)
    has_2030_target: bool = Field(default=False)
    has_2050_target: bool = Field(default=False)
    # Pathway
    pathway_status: str = Field(default="no_pathway")
    total_planned_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    total_investment_eur: float = Field(default=0.0, ge=0.0)
    actions_count: int = Field(default=0, ge=0)
    # Progress
    overall_progress_score: float = Field(default=0.0, ge=0.0, le=100.0)
    art15_compliance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    progress_by_target: List[Dict[str, Any]] = Field(default_factory=list)
    reporting_year: int = Field(default=2026)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# PARIS-ALIGNED REDUCTION BENCHMARKS
# =============================================================================


PARIS_BENCHMARKS: Dict[str, Dict[str, float]] = {
    # Required cumulative reduction % from 2019 baseline
    "1_5c": {
        "2025": 20.0, "2030": 42.0, "2035": 60.0,
        "2040": 75.0, "2045": 87.0, "2050": 95.0,
    },
    "well_below_2c": {
        "2025": 12.0, "2030": 25.0, "2035": 40.0,
        "2040": 55.0, "2045": 72.0, "2050": 90.0,
    },
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ClimateTransitionPlanningWorkflow:
    """
    4-phase CSDDD climate transition planning workflow.

    Assesses emission baselines, evaluates target alignment with the Paris
    Agreement, designs decarbonisation pathways, and tracks progress per Art. 15.

    Zero-hallucination: all emission calculations and target assessments use
    deterministic arithmetic. No LLM in numeric calculation paths.

    Example:
        >>> wf = ClimateTransitionPlanningWorkflow()
        >>> inp = ClimateTransitionPlanningInput(emission_data=[...], targets=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.art15_compliance_score >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ClimateTransitionPlanningWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._baseline: Optional[EmissionData] = None
        self._current: Optional[EmissionData] = None
        self._target_alignment: str = "insufficient_data"
        self._pathway_status: str = "no_pathway"
        self._progress_score: float = 0.0
        self._art15_score: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.BASELINE_ASSESSMENT.value, "description": "Assess current emissions baseline"},
            {"name": WorkflowPhase.TARGET_SETTING.value, "description": "Evaluate target alignment with Paris"},
            {"name": WorkflowPhase.PATHWAY_DESIGN.value, "description": "Design decarbonisation pathway"},
            {"name": WorkflowPhase.PROGRESS_TRACKING.value, "description": "Track progress against targets"},
        ]

    def validate_inputs(self, input_data: ClimateTransitionPlanningInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.emission_data:
            issues.append("No emission data provided")
        if not input_data.targets:
            issues.append("No climate targets defined")
        return issues

    async def execute(
        self,
        input_data: Optional[ClimateTransitionPlanningInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ClimateTransitionPlanningResult:
        """
        Execute the 4-phase climate transition planning workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            ClimateTransitionPlanningResult with baseline, targets, and progress.
        """
        if input_data is None:
            input_data = ClimateTransitionPlanningInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting climate transition planning workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_baseline_assessment(input_data))
            phase_results.append(await self._phase_target_setting(input_data))
            phase_results.append(await self._phase_pathway_design(input_data))
            phase_results.append(await self._phase_progress_tracking(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Climate transition planning failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        baseline_total = self._baseline.total_tco2e if self._baseline else 0.0
        current_total = self._current.total_tco2e if self._current else 0.0
        reduction_pct = round(
            ((baseline_total - current_total) / baseline_total) * 100, 1
        ) if baseline_total > 0 else 0.0

        targets = input_data.targets
        has_2030 = any(t.target_year == 2030 for t in targets)
        has_2050 = any(t.target_year >= 2050 for t in targets)
        sbti_count = sum(1 for t in targets if t.sbti_validated)

        progress_by_target = [
            {
                "target_id": t.target_id,
                "target_name": t.target_name,
                "target_year": t.target_year,
                "reduction_pct_target": t.reduction_pct,
                "current_progress_pct": t.current_progress_pct,
            }
            for t in targets
        ]

        result = ClimateTransitionPlanningResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            baseline_year=self._baseline.year if self._baseline else 2019,
            baseline_total_tco2e=round(baseline_total, 2),
            current_total_tco2e=round(current_total, 2),
            reduction_since_baseline_pct=reduction_pct,
            targets_count=len(targets),
            target_alignment=self._target_alignment,
            sbti_validated_targets=sbti_count,
            has_2030_target=has_2030,
            has_2050_target=has_2050,
            pathway_status=self._pathway_status,
            total_planned_reduction_tco2e=round(
                sum(a.expected_reduction_tco2e for a in input_data.transition_actions), 2
            ),
            total_investment_eur=round(
                sum(a.investment_eur for a in input_data.transition_actions), 2
            ),
            actions_count=len(input_data.transition_actions),
            overall_progress_score=self._progress_score,
            art15_compliance_score=self._art15_score,
            progress_by_target=progress_by_target,
            reporting_year=input_data.reporting_year,
            executed_at=_utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Climate transition planning %s completed in %.2fs: Art.15=%.1f%%",
            self.workflow_id, elapsed, self._art15_score,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Baseline Assessment
    # -------------------------------------------------------------------------

    async def _phase_baseline_assessment(
        self, input_data: ClimateTransitionPlanningInput,
    ) -> PhaseResult:
        """Assess current emissions and establish baseline."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        emissions = sorted(input_data.emission_data, key=lambda e: e.year)

        # Identify baseline (earliest year) and current (latest year)
        if emissions:
            self._baseline = emissions[0]
            self._current = emissions[-1]

            # Recalculate totals if not set
            for ed in emissions:
                if ed.total_tco2e == 0:
                    ed.total_tco2e = ed.scope_1_tco2e + ed.scope_2_tco2e + ed.scope_3_tco2e
                if ed.revenue_eur > 0 and ed.intensity_tco2e_per_meur == 0:
                    ed.intensity_tco2e_per_meur = round(
                        ed.total_tco2e / (ed.revenue_eur / 1_000_000), 2
                    )

            baseline_total = self._baseline.total_tco2e
            current_total = self._current.total_tco2e
            abs_change = current_total - baseline_total
            pct_change = round(
                (abs_change / baseline_total) * 100, 1
            ) if baseline_total > 0 else 0.0

            outputs["baseline_year"] = self._baseline.year
            outputs["baseline_total_tco2e"] = round(baseline_total, 2)
            outputs["baseline_scope_1"] = round(self._baseline.scope_1_tco2e, 2)
            outputs["baseline_scope_2"] = round(self._baseline.scope_2_tco2e, 2)
            outputs["baseline_scope_3"] = round(self._baseline.scope_3_tco2e, 2)
            outputs["current_year"] = self._current.year
            outputs["current_total_tco2e"] = round(current_total, 2)
            outputs["absolute_change_tco2e"] = round(abs_change, 2)
            outputs["percentage_change"] = pct_change
            outputs["years_of_data"] = len(emissions)
            outputs["scope_3_included"] = any(e.scope_3_tco2e > 0 for e in emissions)

            # Emission trajectory
            trajectory: List[Dict[str, Any]] = []
            for ed in emissions:
                trajectory.append({
                    "year": ed.year,
                    "total_tco2e": round(ed.total_tco2e, 2),
                    "data_quality": ed.data_quality,
                })
            outputs["emission_trajectory"] = trajectory

            if not any(e.scope_3_tco2e > 0 for e in emissions):
                warnings.append("No Scope 3 data -- Art. 15 requires full value chain coverage")
            if len(emissions) < 2:
                warnings.append("Only one year of data -- insufficient for trend analysis")
        else:
            outputs["baseline_year"] = 0
            outputs["years_of_data"] = 0
            warnings.append("No emission data provided -- cannot establish baseline")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 BaselineAssessment: %d years, baseline=%s",
            outputs.get("years_of_data", 0),
            outputs.get("baseline_total_tco2e", "N/A"),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.BASELINE_ASSESSMENT.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Target Setting
    # -------------------------------------------------------------------------

    async def _phase_target_setting(
        self, input_data: ClimateTransitionPlanningInput,
    ) -> PhaseResult:
        """Evaluate climate targets against Paris Agreement benchmarks."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        targets = input_data.targets

        if not targets:
            self._target_alignment = TargetAlignment.INSUFFICIENT_DATA.value
            outputs["targets_count"] = 0
            outputs["alignment"] = self._target_alignment
            warnings.append("No climate targets defined -- Art. 15(3) requires time-bound targets")
            elapsed = (_utcnow() - started).total_seconds()
            return PhaseResult(
                phase_name=WorkflowPhase.TARGET_SETTING.value,
                status=PhaseStatus.COMPLETED,
                duration_seconds=elapsed, outputs=outputs, warnings=warnings,
                provenance_hash=self._hash_dict(outputs),
            )

        # Assess each target against Paris benchmarks
        target_assessments: List[Dict[str, Any]] = []
        most_ambitious_alignment = "not_aligned"

        for target in targets:
            year_str = str(target.target_year)
            benchmark_1_5 = PARIS_BENCHMARKS["1_5c"].get(year_str, 0.0)
            benchmark_wb2 = PARIS_BENCHMARKS["well_below_2c"].get(year_str, 0.0)

            if target.reduction_pct >= benchmark_1_5 and benchmark_1_5 > 0:
                alignment = TargetAlignment.ALIGNED_1_5C.value
            elif target.reduction_pct >= benchmark_wb2 and benchmark_wb2 > 0:
                alignment = TargetAlignment.ALIGNED_WELL_BELOW_2C.value
            elif target.reduction_pct > 0:
                alignment = TargetAlignment.NOT_ALIGNED.value
            else:
                alignment = TargetAlignment.INSUFFICIENT_DATA.value

            # Track most ambitious
            alignment_rank = {
                "aligned_1_5c": 4, "aligned_well_below_2c": 3,
                "aligned_2c": 2, "not_aligned": 1, "insufficient_data": 0,
            }
            if alignment_rank.get(alignment, 0) > alignment_rank.get(most_ambitious_alignment, 0):
                most_ambitious_alignment = alignment

            target_assessments.append({
                "target_id": target.target_id,
                "target_name": target.target_name,
                "target_year": target.target_year,
                "reduction_pct": target.reduction_pct,
                "benchmark_1_5c": benchmark_1_5,
                "benchmark_wb2c": benchmark_wb2,
                "alignment": alignment,
                "sbti_validated": target.sbti_validated,
                "includes_scope_3": target.includes_scope_3,
            })

        self._target_alignment = most_ambitious_alignment

        has_2030 = any(t.target_year == 2030 for t in targets)
        has_2050 = any(t.target_year >= 2050 for t in targets)
        scope_3_targets = sum(1 for t in targets if t.includes_scope_3)

        outputs["targets_count"] = len(targets)
        outputs["target_assessments"] = target_assessments
        outputs["overall_alignment"] = self._target_alignment
        outputs["has_2030_target"] = has_2030
        outputs["has_2050_target"] = has_2050
        outputs["sbti_validated_count"] = sum(1 for t in targets if t.sbti_validated)
        outputs["scope_3_target_count"] = scope_3_targets

        if not has_2030:
            warnings.append("No 2030 target -- Art. 15(3) requires at least 2030 and 5-year intervals")
        if not has_2050:
            warnings.append("No 2050 (or beyond) target -- Art. 15(3) requires targets through 2050")
        if self._target_alignment == "not_aligned":
            warnings.append("Targets not aligned with Paris Agreement 1.5C pathway")
        if scope_3_targets == 0:
            warnings.append("No targets include Scope 3 -- value chain emissions should be covered")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 TargetSetting: %d targets, alignment=%s",
            len(targets), self._target_alignment,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.TARGET_SETTING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Pathway Design
    # -------------------------------------------------------------------------

    async def _phase_pathway_design(
        self, input_data: ClimateTransitionPlanningInput,
    ) -> PhaseResult:
        """Design decarbonisation pathway with milestones."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        actions = input_data.transition_actions
        baseline_total = self._baseline.total_tco2e if self._baseline else 0.0

        if not actions:
            self._pathway_status = PathwayStatus.NO_PATHWAY.value
            outputs["actions_count"] = 0
            outputs["pathway_status"] = self._pathway_status
            warnings.append("No transition actions defined -- pathway design required per Art. 15")
            elapsed = (_utcnow() - started).total_seconds()
            return PhaseResult(
                phase_name=WorkflowPhase.PATHWAY_DESIGN.value,
                status=PhaseStatus.COMPLETED,
                duration_seconds=elapsed, outputs=outputs, warnings=warnings,
                provenance_hash=self._hash_dict(outputs),
            )

        # Aggregate actions
        total_reduction = sum(a.expected_reduction_tco2e for a in actions)
        total_investment = sum(a.investment_eur for a in actions)

        # Group by category
        by_category: Dict[str, float] = {}
        for a in actions:
            cat = a.category if a.category else "other"
            by_category[cat] = by_category.get(cat, 0.0) + a.expected_reduction_tco2e

        # Timeline
        by_year: Dict[int, float] = {}
        for a in actions:
            for year in range(a.start_year, a.completion_year + 1):
                annual_contribution = a.expected_reduction_tco2e / max(
                    (a.completion_year - a.start_year + 1), 1
                )
                by_year[year] = by_year.get(year, 0.0) + annual_contribution

        # Build cumulative pathway
        cumulative = 0.0
        pathway_points: List[Dict[str, Any]] = []
        for year in sorted(by_year.keys()):
            cumulative += by_year[year]
            pct_of_baseline = round(
                (cumulative / baseline_total) * 100, 1
            ) if baseline_total > 0 else 0.0
            pathway_points.append({
                "year": year,
                "annual_reduction_tco2e": round(by_year[year], 2),
                "cumulative_reduction_tco2e": round(cumulative, 2),
                "reduction_pct_of_baseline": pct_of_baseline,
            })

        # Assess adequacy: does the pathway reach the most ambitious target?
        most_ambitious_target_pct = max(
            (t.reduction_pct for t in input_data.targets), default=0.0
        )
        pathway_reduction_pct = round(
            (total_reduction / baseline_total) * 100, 1
        ) if baseline_total > 0 else 0.0

        if pathway_reduction_pct >= most_ambitious_target_pct * 0.9:
            self._pathway_status = PathwayStatus.ON_TRACK.value
        elif pathway_reduction_pct >= most_ambitious_target_pct * 0.6:
            self._pathway_status = PathwayStatus.SLIGHTLY_OFF.value
        else:
            self._pathway_status = PathwayStatus.SIGNIFICANTLY_OFF.value

        # Action status summary
        status_counts: Dict[str, int] = {}
        for a in actions:
            status_counts[a.status] = status_counts.get(a.status, 0) + 1

        outputs["actions_count"] = len(actions)
        outputs["total_planned_reduction_tco2e"] = round(total_reduction, 2)
        outputs["pathway_reduction_pct"] = pathway_reduction_pct
        outputs["total_investment_eur"] = round(total_investment, 2)
        outputs["pathway_status"] = self._pathway_status
        outputs["by_category"] = {k: round(v, 2) for k, v in by_category.items()}
        outputs["pathway_points"] = pathway_points
        outputs["action_status_distribution"] = status_counts
        outputs["cost_per_tco2e"] = round(
            total_investment / total_reduction, 2
        ) if total_reduction > 0 else 0.0

        if self._pathway_status == PathwayStatus.SIGNIFICANTLY_OFF.value:
            warnings.append("Pathway significantly off target -- additional actions needed")
        if pathway_reduction_pct < 42:
            warnings.append("Pathway does not achieve 42% reduction by planned period -- misaligned with 1.5C")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 PathwayDesign: %d actions, %.1f%% reduction, status=%s",
            len(actions), pathway_reduction_pct, self._pathway_status,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.PATHWAY_DESIGN.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Progress Tracking
    # -------------------------------------------------------------------------

    async def _phase_progress_tracking(
        self, input_data: ClimateTransitionPlanningInput,
    ) -> PhaseResult:
        """Track progress against transition plan targets."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        targets = input_data.targets
        baseline_total = self._baseline.total_tco2e if self._baseline else 0.0
        current_total = self._current.total_tco2e if self._current else 0.0

        # Calculate actual progress per target
        target_progress: List[Dict[str, Any]] = []
        for t in targets:
            # Actual reduction from baseline
            actual_reduction_pct = round(
                ((baseline_total - current_total) / baseline_total) * 100, 1
            ) if baseline_total > 0 else 0.0

            # Expected progress (linear interpolation)
            current_year = input_data.reporting_year
            total_years = t.target_year - t.base_year
            years_elapsed = current_year - t.base_year
            expected_progress_pct = round(
                (years_elapsed / total_years) * t.reduction_pct, 1
            ) if total_years > 0 else 0.0

            # On-track assessment
            gap = actual_reduction_pct - expected_progress_pct
            if gap >= -5:
                status = "on_track"
            elif gap >= -15:
                status = "at_risk"
            else:
                status = "off_track"

            target_progress.append({
                "target_id": t.target_id,
                "target_name": t.target_name,
                "target_year": t.target_year,
                "target_reduction_pct": t.reduction_pct,
                "expected_progress_pct": expected_progress_pct,
                "actual_reduction_pct": actual_reduction_pct,
                "gap_pct": round(gap, 1),
                "status": status,
            })

        # Overall progress score
        if target_progress:
            on_track_count = sum(1 for tp in target_progress if tp["status"] == "on_track")
            self._progress_score = round(
                (on_track_count / len(target_progress)) * 100, 1
            )
        else:
            self._progress_score = 0.0

        # Art. 15 compliance score (composite)
        art15_components: List[float] = []
        # Has transition plan
        art15_components.append(100.0 if input_data.has_transition_plan else 0.0)
        # Has 2030 target
        art15_components.append(100.0 if any(t.target_year == 2030 for t in targets) else 0.0)
        # Has 2050 target
        art15_components.append(100.0 if any(t.target_year >= 2050 for t in targets) else 0.0)
        # Paris alignment
        alignment_scores = {
            "aligned_1_5c": 100.0, "aligned_well_below_2c": 75.0,
            "aligned_2c": 50.0, "not_aligned": 25.0, "insufficient_data": 0.0,
        }
        art15_components.append(alignment_scores.get(self._target_alignment, 0.0))
        # Has actions
        art15_components.append(100.0 if input_data.transition_actions else 0.0)
        # Includes scope 3
        art15_components.append(100.0 if any(t.includes_scope_3 for t in targets) else 0.0)

        self._art15_score = round(
            sum(art15_components) / len(art15_components), 1
        ) if art15_components else 0.0

        outputs["target_progress"] = target_progress
        outputs["overall_progress_score"] = self._progress_score
        outputs["art15_compliance_score"] = self._art15_score
        outputs["targets_on_track"] = sum(1 for tp in target_progress if tp["status"] == "on_track")
        outputs["targets_at_risk"] = sum(1 for tp in target_progress if tp["status"] == "at_risk")
        outputs["targets_off_track"] = sum(1 for tp in target_progress if tp["status"] == "off_track")

        if self._art15_score < 50:
            warnings.append(f"Art. 15 compliance score is {self._art15_score}% -- significant gaps remain")
        off_track_targets = [tp for tp in target_progress if tp["status"] == "off_track"]
        if off_track_targets:
            warnings.append(f"{len(off_track_targets)} targets are off track")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ProgressTracking: progress=%.1f%%, Art.15=%.1f%%",
            self._progress_score, self._art15_score,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.PROGRESS_TRACKING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ClimateTransitionPlanningResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
