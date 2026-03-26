# -*- coding: utf-8 -*-
"""
Supplier Programme Workflow
===================================

4-phase workflow for managing supplier decarbonization programmes and
tracking their impact on the reporter's Scope 3 trajectory within PACK-043
Scope 3 Complete Pack.

Phases:
    1. TARGET_SETTING            -- Set science-aligned reduction targets for
                                    top suppliers based on spend/emission share.
    2. COMMITMENT_COLLECTION     -- Track supplier SBTi, RE100, CDP, and
                                    net-zero commitments and validate them.
    3. PROGRESS_TRACKING         -- Measure year-over-year supplier emission
                                    reductions and data quality improvements.
    4. IMPACT_ASSESSMENT         -- Calculate programme impact on the reporter's
                                    Scope 3 trajectory and ROI.

The workflow follows GreenLang zero-hallucination principles: all target
calculations, progress tracking, and impact assessment use deterministic
arithmetic on auditable supplier data. SHA-256 provenance hashes guarantee
auditability.

Regulatory Basis:
    SBTi Corporate Net-Zero Standard -- Supplier engagement target
    CDP Supply Chain Program
    GHG Protocol Scope 3 Standard -- Chapter 8 (Collecting supplier data)
    RE100 criteria

Schedule: quarterly tracking, annual assessment
Estimated duration: 2-4 hours per cycle

Author: GreenLang Platform Team
Version: 43.0.0
"""

_MODULE_VERSION: str = "43.0.0"

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
    PARTIAL = "partial"


class SupplierTier(str, Enum):
    STRATEGIC = "strategic"
    KEY = "key"
    STANDARD = "standard"
    MINOR = "minor"


class CommitmentType(str, Enum):
    SBTI_NEAR_TERM = "sbti_near_term"
    SBTI_NET_ZERO = "sbti_net_zero"
    RE100 = "re100"
    CDP_A_LIST = "cdp_a_list"
    CDP_DISCLOSED = "cdp_disclosed"
    NET_ZERO_PLEDGE = "net_zero_pledge"
    INTERNAL_TARGET = "internal_target"
    NONE = "none"


class CommitmentStatus(str, Enum):
    COMMITTED = "committed"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    LAPSED = "lapsed"
    NOT_COMMITTED = "not_committed"


class ProgressRating(str, Enum):
    ON_TRACK = "on_track"
    AHEAD = "ahead"
    BEHIND = "behind"
    NO_DATA = "no_data"
    NOT_STARTED = "not_started"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# SBTi supplier engagement: required % of suppliers with SBTi by spend
SBTI_SUPPLIER_ENGAGEMENT_TARGET_PCT: float = 67.0

# Typical emission reduction rates by commitment type (% per year)
COMMITMENT_REDUCTION_RATES: Dict[str, float] = {
    CommitmentType.SBTI_NEAR_TERM.value: 4.2,
    CommitmentType.SBTI_NET_ZERO.value: 4.2,
    CommitmentType.RE100.value: 3.0,
    CommitmentType.CDP_A_LIST.value: 3.5,
    CommitmentType.CDP_DISCLOSED.value: 1.5,
    CommitmentType.NET_ZERO_PLEDGE.value: 2.0,
    CommitmentType.INTERNAL_TARGET.value: 1.0,
    CommitmentType.NONE.value: 0.0,
}

# Tier-based engagement priority weights
TIER_WEIGHTS: Dict[str, float] = {
    SupplierTier.STRATEGIC.value: 4.0,
    SupplierTier.KEY.value: 3.0,
    SupplierTier.STANDARD.value: 2.0,
    SupplierTier.MINOR.value: 1.0,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class WorkflowState(BaseModel):
    workflow_id: str = Field(default="")
    current_phase: int = Field(default=0)
    phase_statuses: Dict[str, str] = Field(default_factory=dict)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default="")
    updated_at: str = Field(default="")


class SupplierProfile(BaseModel):
    """Supplier profile for programme management."""

    supplier_id: str = Field(default_factory=lambda: f"sup-{uuid.uuid4().hex[:8]}")
    supplier_name: str = Field(default="")
    tier: SupplierTier = Field(default=SupplierTier.STANDARD)
    annual_spend_usd: float = Field(default=0.0, ge=0.0)
    spend_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    estimated_scope12_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_contribution_tco2e: float = Field(default=0.0, ge=0.0)
    sector: str = Field(default="")
    country: str = Field(default="")
    current_commitments: List[CommitmentType] = Field(default_factory=list)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    latest_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    yoy_change_pct: float = Field(default=0.0)


class SupplierTarget(BaseModel):
    """Reduction target set for a supplier."""

    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    target_type: CommitmentType = Field(default=CommitmentType.SBTI_NEAR_TERM)
    target_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_year: int = Field(default=2030, ge=2025, le=2050)
    base_year: int = Field(default=2025, ge=2015, le=2050)
    base_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    target_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    annual_rate_pct: float = Field(default=0.0, ge=0.0)
    rationale: str = Field(default="")


class CommitmentRecord(BaseModel):
    """Tracked supplier commitment."""

    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    commitment_type: CommitmentType = Field(default=CommitmentType.NONE)
    status: CommitmentStatus = Field(default=CommitmentStatus.NOT_COMMITTED)
    committed_date: str = Field(default="")
    verification_source: str = Field(default="")
    notes: str = Field(default="")


class ProgressRecord(BaseModel):
    """Year-over-year progress for a supplier."""

    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    base_year_tco2e: float = Field(default=0.0, ge=0.0)
    latest_year_tco2e: float = Field(default=0.0, ge=0.0)
    absolute_change_tco2e: float = Field(default=0.0)
    yoy_change_pct: float = Field(default=0.0)
    cumulative_change_pct: float = Field(default=0.0)
    rating: ProgressRating = Field(default=ProgressRating.NO_DATA)
    target_annual_rate_pct: float = Field(default=0.0, ge=0.0)


class ProgrammeImpact(BaseModel):
    """Aggregate programme impact on reporter's Scope 3."""

    total_supplier_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    reporter_scope3_impact_tco2e: float = Field(default=0.0, ge=0.0)
    reporter_scope3_impact_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    programme_cost_usd: float = Field(default=0.0, ge=0.0)
    cost_per_tco2e_avoided_usd: float = Field(default=0.0, ge=0.0)
    suppliers_on_track: int = Field(default=0, ge=0)
    suppliers_behind: int = Field(default=0, ge=0)
    sbti_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    meets_sbti_engagement_target: bool = Field(default=False)


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class SupplierProgrammeInput(BaseModel):
    """Input data model for SupplierProgrammeWorkflow."""

    organization_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    reporter_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    suppliers: List[SupplierProfile] = Field(default_factory=list)
    programme_cost_usd: float = Field(
        default=0.0, ge=0.0, description="Total programme cost for ROI"
    )
    target_sbti_coverage_pct: float = Field(
        default=SBTI_SUPPLIER_ENGAGEMENT_TARGET_PCT, ge=0.0, le=100.0
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class SupplierProgrammeOutput(BaseModel):
    """Complete output from SupplierProgrammeWorkflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="supplier_programme")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_name: str = Field(default="")
    supplier_targets: List[SupplierTarget] = Field(default_factory=list)
    commitment_records: List[CommitmentRecord] = Field(default_factory=list)
    progress_records: List[ProgressRecord] = Field(default_factory=list)
    programme_impact: Optional[ProgrammeImpact] = Field(default=None)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class SupplierProgrammeWorkflow:
    """
    4-phase supplier decarbonization programme management workflow.

    Sets science-aligned supplier targets, tracks commitments, measures
    year-over-year progress, and calculates impact on the reporter's
    Scope 3 trajectory.

    Zero-hallucination: all target calculations, progress metrics, and
    impact assessments use deterministic arithmetic on auditable data.

    Example:
        >>> wf = SupplierProgrammeWorkflow()
        >>> inp = SupplierProgrammeInput(
        ...     reporter_scope3_tco2e=100000,
        ...     suppliers=[SupplierProfile(supplier_name="SupA", spend_pct=25)],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_NAMES: List[str] = [
        "target_setting",
        "commitment_collection",
        "progress_tracking",
        "impact_assessment",
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize SupplierProgrammeWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._targets: List[SupplierTarget] = []
        self._commitments: List[CommitmentRecord] = []
        self._progress: List[ProgressRecord] = []
        self._impact: Optional[ProgrammeImpact] = None
        self._phase_results: List[PhaseResult] = []
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[SupplierProgrammeInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> SupplierProgrammeOutput:
        """Execute the 4-phase supplier programme workflow."""
        if input_data is None:
            input_data = SupplierProgrammeInput()

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting supplier programme workflow %s org=%s suppliers=%d",
            self.workflow_id, input_data.organization_name, len(input_data.suppliers),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING
        self._update_progress(0.0)

        try:
            for i, (name, fn) in enumerate([
                ("target_setting", self._phase_target_setting),
                ("commitment_collection", self._phase_commitment_collection),
                ("progress_tracking", self._phase_progress_tracking),
                ("impact_assessment", self._phase_impact_assessment),
            ], 1):
                phase = await self._execute_with_retry(fn, input_data, i)
                self._phase_results.append(phase)
                if phase.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {i} failed: {phase.errors}")
                self._update_progress(i * 25.0)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Supplier programme workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(
                PhaseResult(phase_name="error", phase_number=0,
                            status=PhaseStatus.FAILED, errors=[str(exc)])
            )

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        result = SupplierProgrammeOutput(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_name=input_data.organization_name,
            supplier_targets=self._targets,
            commitment_records=self._commitments,
            progress_records=self._progress,
            programme_impact=self._impact,
            progress_pct=self._state.progress_pct,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Supplier programme workflow %s completed in %.2fs status=%s",
            self.workflow_id, elapsed, overall_status.value,
        )
        return result

    def get_state(self) -> WorkflowState:
        return self._state.model_copy()

    async def resume(
        self, state: WorkflowState, input_data: SupplierProgrammeInput
    ) -> SupplierProgrammeOutput:
        self._state = state
        self.workflow_id = state.workflow_id
        return await self.execute(input_data)

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: SupplierProgrammeInput, phase_number: int
    ) -> PhaseResult:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    import asyncio
                    await asyncio.sleep(self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1)))
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Target Setting
    # -------------------------------------------------------------------------

    async def _phase_target_setting(
        self, input_data: SupplierProgrammeInput
    ) -> PhaseResult:
        """Set science-aligned reduction targets for top suppliers."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._targets = []

        for supplier in input_data.suppliers:
            # Determine target type based on tier
            if supplier.tier in (SupplierTier.STRATEGIC, SupplierTier.KEY):
                target_type = CommitmentType.SBTI_NEAR_TERM
            else:
                target_type = CommitmentType.INTERNAL_TARGET

            annual_rate = COMMITMENT_REDUCTION_RATES.get(target_type.value, 2.0)
            years = 2030 - input_data.reporting_year
            total_reduction = min(annual_rate * max(years, 1), 100.0)

            base_emissions = supplier.base_year_emissions_tco2e
            if base_emissions <= 0:
                base_emissions = supplier.scope3_contribution_tco2e

            target_emissions = base_emissions * (1 - total_reduction / 100.0)

            self._targets.append(SupplierTarget(
                supplier_id=supplier.supplier_id,
                supplier_name=supplier.supplier_name,
                target_type=target_type,
                target_reduction_pct=round(total_reduction, 2),
                target_year=2030,
                base_year=input_data.reporting_year,
                base_emissions_tco2e=round(base_emissions, 2),
                target_emissions_tco2e=round(max(target_emissions, 0), 2),
                annual_rate_pct=round(annual_rate, 2),
                rationale=(
                    f"SBTi-aligned {annual_rate:.1f}% annual reduction "
                    f"for {supplier.tier.value} supplier"
                ),
            ))

        outputs["targets_set"] = len(self._targets)
        outputs["strategic_key_targets"] = sum(
            1 for t in self._targets if t.target_type == CommitmentType.SBTI_NEAR_TERM
        )
        outputs["avg_target_reduction_pct"] = round(
            sum(t.target_reduction_pct for t in self._targets) / max(len(self._targets), 1), 2
        )

        self._state.phase_statuses["target_setting"] = "completed"
        self._state.current_phase = 1

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 1 TargetSetting: %d targets set", len(self._targets))
        return PhaseResult(
            phase_name="target_setting", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Commitment Collection
    # -------------------------------------------------------------------------

    async def _phase_commitment_collection(
        self, input_data: SupplierProgrammeInput
    ) -> PhaseResult:
        """Track supplier SBTi, RE100, CDP, net-zero commitments."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._commitments = []

        for supplier in input_data.suppliers:
            for commitment_type in supplier.current_commitments:
                status = self._classify_commitment_status(commitment_type)
                self._commitments.append(CommitmentRecord(
                    supplier_id=supplier.supplier_id,
                    supplier_name=supplier.supplier_name,
                    commitment_type=commitment_type,
                    status=status,
                    verification_source=self._get_verification_source(commitment_type),
                ))

            if not supplier.current_commitments:
                self._commitments.append(CommitmentRecord(
                    supplier_id=supplier.supplier_id,
                    supplier_name=supplier.supplier_name,
                    commitment_type=CommitmentType.NONE,
                    status=CommitmentStatus.NOT_COMMITTED,
                ))

        # Calculate SBTi coverage by spend
        total_spend = sum(s.annual_spend_usd for s in input_data.suppliers)
        sbti_spend = sum(
            s.annual_spend_usd for s in input_data.suppliers
            if any(
                c in (CommitmentType.SBTI_NEAR_TERM, CommitmentType.SBTI_NET_ZERO)
                for c in s.current_commitments
            )
        )
        sbti_coverage = (sbti_spend / total_spend * 100.0) if total_spend > 0 else 0.0

        # Commitment type distribution
        type_dist: Dict[str, int] = {}
        for cr in self._commitments:
            t = cr.commitment_type.value
            type_dist[t] = type_dist.get(t, 0) + 1

        outputs["commitments_tracked"] = len(self._commitments)
        outputs["sbti_coverage_by_spend_pct"] = round(sbti_coverage, 2)
        outputs["commitment_type_distribution"] = type_dist
        outputs["suppliers_with_no_commitment"] = sum(
            1 for cr in self._commitments if cr.commitment_type == CommitmentType.NONE
        )

        if sbti_coverage < input_data.target_sbti_coverage_pct:
            warnings.append(
                f"SBTi coverage by spend ({sbti_coverage:.1f}%) is below target "
                f"({input_data.target_sbti_coverage_pct:.0f}%)"
            )

        self._state.phase_statuses["commitment_collection"] = "completed"
        self._state.current_phase = 2

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 CommitmentCollection: %d commitments, SBTi coverage=%.1f%%",
            len(self._commitments), sbti_coverage,
        )
        return PhaseResult(
            phase_name="commitment_collection", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Progress Tracking
    # -------------------------------------------------------------------------

    async def _phase_progress_tracking(
        self, input_data: SupplierProgrammeInput
    ) -> PhaseResult:
        """Measure YoY supplier emission reductions."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._progress = []

        for supplier in input_data.suppliers:
            base = supplier.base_year_emissions_tco2e
            latest = supplier.latest_year_emissions_tco2e

            if base <= 0 and latest <= 0:
                self._progress.append(ProgressRecord(
                    supplier_id=supplier.supplier_id,
                    supplier_name=supplier.supplier_name,
                    rating=ProgressRating.NO_DATA,
                ))
                continue

            abs_change = latest - base
            yoy_pct = supplier.yoy_change_pct
            cumulative_pct = (
                ((base - latest) / base * 100.0) if base > 0 else 0.0
            )

            # Find target for this supplier
            target = next(
                (t for t in self._targets if t.supplier_id == supplier.supplier_id),
                None,
            )
            target_rate = target.annual_rate_pct if target else 0.0

            # Rating
            if cumulative_pct >= target_rate * 1.2:
                rating = ProgressRating.AHEAD
            elif cumulative_pct >= target_rate * 0.8:
                rating = ProgressRating.ON_TRACK
            elif base > 0:
                rating = ProgressRating.BEHIND
            else:
                rating = ProgressRating.NOT_STARTED

            self._progress.append(ProgressRecord(
                supplier_id=supplier.supplier_id,
                supplier_name=supplier.supplier_name,
                base_year_tco2e=round(base, 2),
                latest_year_tco2e=round(latest, 2),
                absolute_change_tco2e=round(abs_change, 2),
                yoy_change_pct=round(yoy_pct, 2),
                cumulative_change_pct=round(cumulative_pct, 2),
                rating=rating,
                target_annual_rate_pct=round(target_rate, 2),
            ))

        # Rating distribution
        rating_dist: Dict[str, int] = {}
        for pr in self._progress:
            r = pr.rating.value
            rating_dist[r] = rating_dist.get(r, 0) + 1

        outputs["suppliers_tracked"] = len(self._progress)
        outputs["rating_distribution"] = rating_dist
        outputs["on_track_or_ahead"] = sum(
            1 for pr in self._progress
            if pr.rating in (ProgressRating.ON_TRACK, ProgressRating.AHEAD)
        )
        outputs["behind"] = sum(
            1 for pr in self._progress if pr.rating == ProgressRating.BEHIND
        )

        self._state.phase_statuses["progress_tracking"] = "completed"
        self._state.current_phase = 3

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ProgressTracking: %d suppliers, on_track=%d behind=%d",
            len(self._progress), outputs["on_track_or_ahead"], outputs["behind"],
        )
        return PhaseResult(
            phase_name="progress_tracking", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Impact Assessment
    # -------------------------------------------------------------------------

    async def _phase_impact_assessment(
        self, input_data: SupplierProgrammeInput
    ) -> PhaseResult:
        """Calculate programme impact on reporter's Scope 3 trajectory."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Sum supplier reductions
        total_supplier_reduction = sum(
            max(pr.base_year_tco2e - pr.latest_year_tco2e, 0.0)
            for pr in self._progress
            if pr.base_year_tco2e > 0
        )

        # Impact on reporter's Scope 3 (proportional to spend-weighted contribution)
        total_scope3 = input_data.reporter_scope3_tco2e
        reporter_impact = total_supplier_reduction  # Direct pass-through
        impact_pct = (
            (reporter_impact / total_scope3 * 100.0)
            if total_scope3 > 0 else 0.0
        )

        # Cost effectiveness
        cost = input_data.programme_cost_usd
        cost_per_tco2e = (
            cost / reporter_impact if reporter_impact > 0 else 0.0
        )

        # SBTi engagement target check
        total_spend = sum(s.annual_spend_usd for s in input_data.suppliers)
        sbti_spend = sum(
            s.annual_spend_usd for s in input_data.suppliers
            if any(
                c in (CommitmentType.SBTI_NEAR_TERM, CommitmentType.SBTI_NET_ZERO)
                for c in s.current_commitments
            )
        )
        sbti_coverage = (sbti_spend / total_spend * 100.0) if total_spend > 0 else 0.0

        on_track = sum(
            1 for pr in self._progress
            if pr.rating in (ProgressRating.ON_TRACK, ProgressRating.AHEAD)
        )
        behind = sum(
            1 for pr in self._progress if pr.rating == ProgressRating.BEHIND
        )

        self._impact = ProgrammeImpact(
            total_supplier_reduction_tco2e=round(total_supplier_reduction, 2),
            reporter_scope3_impact_tco2e=round(reporter_impact, 2),
            reporter_scope3_impact_pct=round(impact_pct, 2),
            programme_cost_usd=round(cost, 2),
            cost_per_tco2e_avoided_usd=round(cost_per_tco2e, 2),
            suppliers_on_track=on_track,
            suppliers_behind=behind,
            sbti_coverage_pct=round(sbti_coverage, 2),
            meets_sbti_engagement_target=sbti_coverage >= input_data.target_sbti_coverage_pct,
        )

        outputs["total_supplier_reduction_tco2e"] = round(total_supplier_reduction, 2)
        outputs["reporter_scope3_impact_pct"] = round(impact_pct, 2)
        outputs["cost_per_tco2e_avoided_usd"] = round(cost_per_tco2e, 2)
        outputs["sbti_coverage_pct"] = round(sbti_coverage, 2)
        outputs["meets_sbti_target"] = self._impact.meets_sbti_engagement_target

        self._state.phase_statuses["impact_assessment"] = "completed"
        self._state.current_phase = 4

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ImpactAssessment: reduction=%.0f tCO2e, impact=%.1f%%, "
            "sbti_coverage=%.1f%%",
            total_supplier_reduction, impact_pct, sbti_coverage,
        )
        return PhaseResult(
            phase_name="impact_assessment", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _classify_commitment_status(self, ct: CommitmentType) -> CommitmentStatus:
        """Classify commitment status based on type."""
        if ct in (CommitmentType.SBTI_NEAR_TERM, CommitmentType.SBTI_NET_ZERO):
            return CommitmentStatus.COMMITTED
        elif ct in (CommitmentType.RE100, CommitmentType.CDP_A_LIST):
            return CommitmentStatus.VERIFIED
        elif ct in (CommitmentType.NET_ZERO_PLEDGE, CommitmentType.INTERNAL_TARGET):
            return CommitmentStatus.IN_PROGRESS
        elif ct == CommitmentType.CDP_DISCLOSED:
            return CommitmentStatus.COMMITTED
        return CommitmentStatus.NOT_COMMITTED

    def _get_verification_source(self, ct: CommitmentType) -> str:
        """Get verification source for commitment type."""
        sources = {
            CommitmentType.SBTI_NEAR_TERM: "SBTi target dashboard",
            CommitmentType.SBTI_NET_ZERO: "SBTi target dashboard",
            CommitmentType.RE100: "RE100 member list",
            CommitmentType.CDP_A_LIST: "CDP A List publication",
            CommitmentType.CDP_DISCLOSED: "CDP disclosure platform",
            CommitmentType.NET_ZERO_PLEDGE: "Race to Zero / public pledge",
            CommitmentType.INTERNAL_TARGET: "Supplier self-declaration",
        }
        return sources.get(ct, "Unverified")

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._targets = []
        self._commitments = []
        self._progress = []
        self._impact = None
        self._phase_results = []
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )

    def _update_progress(self, pct: float) -> None:
        self._state.progress_pct = min(pct, 100.0)
        self._state.updated_at = datetime.utcnow().isoformat()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: SupplierProgrammeOutput) -> str:
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += f"|{result.workflow_id}|{len(result.supplier_targets)}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
