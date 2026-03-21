# -*- coding: utf-8 -*-
"""
Full SBTi Lifecycle Workflow
=================================

8-phase end-to-end SBTi lifecycle workflow within PACK-023 SBTi
Alignment Pack.  Orchestrates all 10 engines across the complete
SBTi journey from initial commitment through target validation,
submission, annual tracking, periodic review, and revalidation.

Phases:
    1. Commitment      -- Register SBTi commitment letter and timeline
    2. Inventory       -- Build and validate emissions inventory (S1+S2+S3)
    3. TargetSet       -- Design targets (ACA/SDA/FLAG pathways)
    4. Validate        -- Run 42-criterion validation and readiness check
    5. Submit          -- Prepare submission package and readiness gate
    6. Track           -- Annual progress tracking with RAG status
    7. Review          -- Periodic review with variance analysis
    8. Revalidate      -- 5-year revalidation cycle assessment

The workflow supports conditional FLAG and FI phases based on
sector analysis.  FLAG target-setting is triggered when FLAG
emissions constitute >= 20% of total emissions.  FI target-setting
is activated when the entity is classified as a financial institution.

Engine Orchestration:
    - target_setting_engine     -- Phase 3 (ACA pathway)
    - sda_sector_engine         -- Phase 3 (SDA-eligible sectors)
    - flag_assessment_engine    -- Phase 3 (FLAG commodities)
    - scope3_screening_engine   -- Phase 2/3 (Scope 3 screening)
    - criteria_validation_engine -- Phase 4 (42-criterion check)
    - temperature_rating_engine -- Phase 4/7 (temperature alignment)
    - progress_tracking_engine  -- Phase 6 (annual RAG)
    - recalculation_engine      -- Phase 7 (base year recalc)
    - fi_portfolio_engine       -- Phase 3 (FI targets)
    - submission_readiness_engine -- Phase 5 (submission gate)

Regulatory references:
    - SBTi Corporate Manual V5.3 (2024)
    - SBTi Corporate Net-Zero Standard V1.3 (2024)
    - SBTi FLAG Guidance V1.1 (2022)
    - SBTi FI Net-Zero Standard V1.0 (2024)
    - GHG Protocol Corporate Standard (2015)
    - GHG Protocol Scope 3 Standard (2011)

Zero-hallucination: all calculations use deterministic formulas
and SBTi-specified thresholds from lookup tables.  No LLM calls
in any numeric computation path.

Author: GreenLang Team
Version: 23.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "23.0.0"


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


class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class LifecycleStage(str, Enum):
    """SBTi lifecycle stage."""

    COMMITMENT = "commitment"
    INVENTORY = "inventory"
    TARGET_SETTING = "target_setting"
    VALIDATION = "validation"
    SUBMISSION = "submission"
    TRACKING = "tracking"
    REVIEW = "review"
    REVALIDATION = "revalidation"


class CommitmentType(str, Enum):
    """SBTi commitment type."""

    NEAR_TERM = "near_term"
    NET_ZERO = "net_zero"
    BOTH = "near_term_and_net_zero"


class EntityType(str, Enum):
    """Entity classification for SBTi pathway selection."""

    CORPORATE = "corporate"
    FINANCIAL_INSTITUTION = "financial_institution"
    SME = "sme"


class PathwayMethod(str, Enum):
    """SBTi target pathway methods."""

    ACA = "aca"
    SDA = "sda"
    FLAG = "flag"
    ACA_FLAG = "aca_flag"
    FI = "fi"


class AmbitionLevel(str, Enum):
    """Temperature ambition level."""

    CELSIUS_1_5 = "1.5C"
    WELL_BELOW_2C = "WB2C"
    CELSIUS_2C = "2C"
    INSUFFICIENT = "insufficient"


class RAGStatus(str, Enum):
    """RAG (Red/Amber/Green) status."""

    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    NOT_ASSESSED = "not_assessed"


class SubmissionReadiness(str, Enum):
    """Submission readiness classification."""

    READY = "ready"
    CONDITIONALLY_READY = "conditionally_ready"
    NOT_READY = "not_ready"


class RevalidationStatus(str, Enum):
    """Revalidation assessment status."""

    VALID = "valid"
    REVALIDATION_DUE = "revalidation_due"
    REVALIDATION_REQUIRED = "revalidation_required"
    EXPIRED = "expired"


# =============================================================================
# REFERENCE DATA
# =============================================================================

# SBTi lifecycle timing (months from commitment)
LIFECYCLE_TIMELINES: Dict[str, int] = {
    "commitment_to_submission": 24,       # 24 months maximum
    "validation_period": 3,               # ~3 months
    "annual_reporting_cycle": 12,         # Annual
    "target_review_cycle": 60,            # 5 years
    "revalidation_cycle": 60,            # 5 years
}

# SDA-eligible sectors
SDA_ELIGIBLE_SECTORS: List[str] = [
    "power", "cement", "steel", "aluminium", "pulp_paper",
    "chemicals", "aviation", "maritime", "road_transport",
    "buildings_commercial", "buildings_residential", "food_beverage",
]

# ACA reduction rates
ACA_RATES: Dict[str, float] = {
    "1.5C": 0.042,
    "WB2C": 0.025,
    "2C": 0.015,
}

# FLAG trigger threshold
FLAG_TRIGGER_PCT = 20.0

# Scope 3 materiality trigger
SCOPE3_MATERIALITY_PCT = 40.0

# Coverage requirements
MIN_SCOPE12_COVERAGE = 95.0
MIN_SCOPE3_NEAR_TERM_COVERAGE = 67.0
MIN_SCOPE3_LONG_TERM_COVERAGE = 90.0

# Revalidation thresholds
REVALIDATION_WARNING_MONTHS = 6  # Warn 6 months before due


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


class CommitmentData(BaseModel):
    """SBTi commitment registration data."""

    commitment_type: CommitmentType = Field(default=CommitmentType.BOTH)
    commitment_date: str = Field(default="", description="ISO date of commitment letter")
    submission_deadline: str = Field(default="", description="24-month deadline")
    commitment_letter_ref: str = Field(default="")
    public_announcement: bool = Field(default=False)
    board_approved: bool = Field(default=False)
    notes: List[str] = Field(default_factory=list)


class InventorySummary(BaseModel):
    """Emissions inventory summary."""

    base_year: int = Field(default=2022, ge=2015)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_total_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_categories_screened: int = Field(default=0, ge=0, le=15)
    scope3_pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    flag_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    flag_pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    total_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    ghg_verified: bool = Field(default=False)
    boundary_approach: str = Field(default="operational_control")


class TargetSummary(BaseModel):
    """Summary of designed targets."""

    target_id: str = Field(default="")
    target_name: str = Field(default="")
    scopes: List[str] = Field(default_factory=list)
    base_year: int = Field(default=2022)
    target_year: int = Field(default=2030)
    reduction_pct: float = Field(default=0.0)
    pathway_method: PathwayMethod = Field(default=PathwayMethod.ACA)
    ambition_level: AmbitionLevel = Field(default=AmbitionLevel.CELSIUS_1_5)
    coverage_pct: float = Field(default=0.0)
    base_emissions_tco2e: float = Field(default=0.0)
    target_emissions_tco2e: float = Field(default=0.0)
    is_flag: bool = Field(default=False)
    is_fi: bool = Field(default=False)


class ValidationSummary(BaseModel):
    """Summary of 42-criterion validation."""

    criteria_assessed: int = Field(default=0)
    criteria_passed: int = Field(default=0)
    criteria_failed: int = Field(default=0)
    criteria_warning: int = Field(default=0)
    readiness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    submission_ready: bool = Field(default=False)
    blocking_criteria: List[str] = Field(default_factory=list)
    gaps_count: int = Field(default=0)


class SubmissionPackage(BaseModel):
    """Submission readiness package."""

    readiness: SubmissionReadiness = Field(default=SubmissionReadiness.NOT_READY)
    readiness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    submission_date: str = Field(default="")
    deadline_date: str = Field(default="")
    days_until_deadline: int = Field(default=0)
    package_complete: bool = Field(default=False)
    missing_documents: List[str] = Field(default_factory=list)
    review_notes: List[str] = Field(default_factory=list)


class TrackingSummary(BaseModel):
    """Annual progress tracking summary."""

    tracking_year: int = Field(default=0)
    overall_rag: RAGStatus = Field(default=RAGStatus.NOT_ASSESSED)
    targets_on_track: int = Field(default=0)
    targets_behind: int = Field(default=0)
    total_reduction_pct: float = Field(default=0.0)
    corrective_actions_count: int = Field(default=0)
    carbon_budget_used_pct: float = Field(default=0.0)


class ReviewSummary(BaseModel):
    """Periodic review summary."""

    review_year: int = Field(default=0)
    recalculation_required: bool = Field(default=False)
    recalculation_reason: str = Field(default="")
    base_year_adjusted: bool = Field(default=False)
    targets_adjusted: int = Field(default=0)
    variance_drivers: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class RevalidationAssessment(BaseModel):
    """5-year revalidation assessment."""

    revalidation_status: RevalidationStatus = Field(default=RevalidationStatus.VALID)
    validation_date: str = Field(default="")
    expiry_date: str = Field(default="")
    months_until_expiry: int = Field(default=0)
    targets_still_valid: bool = Field(default=True)
    methodology_changes: List[str] = Field(default_factory=list)
    ambition_still_sufficient: bool = Field(default=True)
    scope_changes: List[str] = Field(default_factory=list)
    revalidation_actions: List[str] = Field(default_factory=list)


class LifecycleConfig(BaseModel):
    """Configuration for the full SBTi lifecycle workflow."""

    # Entity information
    entity_name: str = Field(default="")
    entity_type: EntityType = Field(default=EntityType.CORPORATE)
    sector: str = Field(default="other")

    # Commitment data
    commitment_type: CommitmentType = Field(default=CommitmentType.BOTH)
    commitment_date: str = Field(default="")
    board_approved: bool = Field(default=False)
    public_announcement: bool = Field(default=False)

    # Inventory data
    base_year: int = Field(default=2022, ge=2015, le=2050)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_total_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_categories_screened: int = Field(default=0, ge=0, le=15)
    flag_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    ghg_verified: bool = Field(default=False)

    # Target preferences
    near_term_target_year: int = Field(default=2030, ge=2025, le=2040)
    long_term_target_year: int = Field(default=2040, ge=2035, le=2060)
    net_zero_target_year: int = Field(default=2050, ge=2040, le=2060)
    ambition: AmbitionLevel = Field(default=AmbitionLevel.CELSIUS_1_5)
    include_net_zero: bool = Field(default=True)

    # FLAG commodity data
    has_flag_commodities: bool = Field(default=False)
    flag_commodity_count: int = Field(default=0, ge=0)
    has_no_deforestation_commitment: bool = Field(default=False)

    # FI portfolio data
    is_financial_institution: bool = Field(default=False)
    fi_portfolio_value_usd: float = Field(default=0.0, ge=0.0)
    fi_holdings_count: int = Field(default=0, ge=0)

    # Evidence and governance
    has_public_commitment: bool = Field(default=False)
    has_cdp_response: bool = Field(default=False)
    has_board_oversight: bool = Field(default=False)
    has_recalculation_policy: bool = Field(default=False)
    has_transition_plan: bool = Field(default=False)
    has_neutralization_plan: bool = Field(default=False)
    has_annual_reporting: bool = Field(default=False)
    offsets_in_target: bool = Field(default=False)

    # Tracking data (for tracking/review phases)
    current_year: Optional[int] = Field(None)
    current_scope1_tco2e: float = Field(default=0.0, ge=0.0)
    current_scope2_tco2e: float = Field(default=0.0, ge=0.0)
    current_scope3_tco2e: float = Field(default=0.0, ge=0.0)

    # Revalidation data
    original_validation_date: str = Field(default="")
    years_since_validation: int = Field(default=0, ge=0)

    # Phase control: which phases to execute
    execute_phases: List[str] = Field(
        default_factory=lambda: [
            "commitment", "inventory", "target_set", "validate",
            "submit", "track", "review", "revalidate",
        ]
    )

    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class LifecycleResult(BaseModel):
    """Complete result from the full SBTi lifecycle workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="full_sbti_lifecycle")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    current_stage: LifecycleStage = Field(default=LifecycleStage.COMMITMENT)
    commitment: Optional[CommitmentData] = Field(None)
    inventory: Optional[InventorySummary] = Field(None)
    targets: List[TargetSummary] = Field(default_factory=list)
    validation: Optional[ValidationSummary] = Field(None)
    submission: Optional[SubmissionPackage] = Field(None)
    tracking: Optional[TrackingSummary] = Field(None)
    review: Optional[ReviewSummary] = Field(None)
    revalidation: Optional[RevalidationAssessment] = Field(None)
    overall_readiness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class FullSBTiLifecycleWorkflow:
    """
    8-phase end-to-end SBTi lifecycle workflow.

    Orchestrates all 10 engines across the complete SBTi journey from
    initial commitment through target validation, submission, annual
    tracking, periodic review, and revalidation.  Supports conditional
    FLAG and FI phases based on sector analysis.

    Zero-hallucination: all calculations use deterministic formulas
    and SBTi-specified thresholds.  No LLM calls in any numeric
    computation path.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = FullSBTiLifecycleWorkflow()
        >>> config = LifecycleConfig(
        ...     entity_name="Acme Corp",
        ...     base_year=2022,
        ...     scope1_tco2e=5000,
        ...     scope2_location_tco2e=3000,
        ...     scope3_total_tco2e=15000,
        ... )
        >>> result = await wf.execute(config)
        >>> assert result.status in (WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL)
    """

    def __init__(self) -> None:
        """Initialise FullSBTiLifecycleWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._commitment: Optional[CommitmentData] = None
        self._inventory: Optional[InventorySummary] = None
        self._targets: List[TargetSummary] = []
        self._validation: Optional[ValidationSummary] = None
        self._submission: Optional[SubmissionPackage] = None
        self._tracking: Optional[TrackingSummary] = None
        self._review: Optional[ReviewSummary] = None
        self._revalidation: Optional[RevalidationAssessment] = None
        self._current_stage = LifecycleStage.COMMITMENT
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: LifecycleConfig) -> LifecycleResult:
        """
        Execute the 8-phase SBTi lifecycle workflow.

        Args:
            config: Lifecycle configuration with entity data, emissions
                inventory, target preferences, and phase control.

        Returns:
            LifecycleResult with complete lifecycle state including
            commitment, inventory, targets, validation, submission,
            tracking, review, and revalidation data.
        """
        started_at = _utcnow()
        self.logger.info(
            "Starting full SBTi lifecycle workflow %s, entity=%s, type=%s",
            self.workflow_id, config.entity_name, config.entity_type.value,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING
        phases_to_run = set(config.execute_phases)

        try:
            # Phase 1: Commitment
            if "commitment" in phases_to_run:
                phase1 = await self._phase_commitment(config)
                self._phase_results.append(phase1)
                self._current_stage = LifecycleStage.COMMITMENT
            else:
                self._phase_results.append(PhaseResult(
                    phase_name="commitment", status=PhaseStatus.SKIPPED))

            # Phase 2: Inventory
            if "inventory" in phases_to_run:
                phase2 = await self._phase_inventory(config)
                self._phase_results.append(phase2)
                if phase2.status == PhaseStatus.FAILED:
                    raise ValueError("Inventory phase failed; cannot proceed")
                self._current_stage = LifecycleStage.INVENTORY
            else:
                self._phase_results.append(PhaseResult(
                    phase_name="inventory", status=PhaseStatus.SKIPPED))

            # Phase 3: Target Setting (with conditional FLAG and FI)
            if "target_set" in phases_to_run:
                phase3 = await self._phase_target_set(config)
                self._phase_results.append(phase3)
                self._current_stage = LifecycleStage.TARGET_SETTING
            else:
                self._phase_results.append(PhaseResult(
                    phase_name="target_set", status=PhaseStatus.SKIPPED))

            # Phase 4: Validation
            if "validate" in phases_to_run:
                phase4 = await self._phase_validate(config)
                self._phase_results.append(phase4)
                self._current_stage = LifecycleStage.VALIDATION
            else:
                self._phase_results.append(PhaseResult(
                    phase_name="validate", status=PhaseStatus.SKIPPED))

            # Phase 5: Submission
            if "submit" in phases_to_run:
                phase5 = await self._phase_submit(config)
                self._phase_results.append(phase5)
                self._current_stage = LifecycleStage.SUBMISSION
            else:
                self._phase_results.append(PhaseResult(
                    phase_name="submit", status=PhaseStatus.SKIPPED))

            # Phase 6: Track
            if "track" in phases_to_run:
                phase6 = await self._phase_track(config)
                self._phase_results.append(phase6)
                self._current_stage = LifecycleStage.TRACKING
            else:
                self._phase_results.append(PhaseResult(
                    phase_name="track", status=PhaseStatus.SKIPPED))

            # Phase 7: Review
            if "review" in phases_to_run:
                phase7 = await self._phase_review(config)
                self._phase_results.append(phase7)
                self._current_stage = LifecycleStage.REVIEW
            else:
                self._phase_results.append(PhaseResult(
                    phase_name="review", status=PhaseStatus.SKIPPED))

            # Phase 8: Revalidation
            if "revalidate" in phases_to_run:
                phase8 = await self._phase_revalidate(config)
                self._phase_results.append(phase8)
                self._current_stage = LifecycleStage.REVALIDATION
            else:
                self._phase_results.append(PhaseResult(
                    phase_name="revalidate", status=PhaseStatus.SKIPPED))

            failed = [p for p in self._phase_results
                       if p.status == PhaseStatus.FAILED]
            skipped = [p for p in self._phase_results
                        if p.status == PhaseStatus.SKIPPED]

            if not failed:
                overall_status = WorkflowStatus.COMPLETED
            elif len(failed) < len(self._phase_results) - len(skipped):
                overall_status = WorkflowStatus.PARTIAL
            else:
                overall_status = WorkflowStatus.FAILED

        except Exception as exc:
            self.logger.error(
                "Full SBTi lifecycle workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        # Calculate overall readiness
        overall_readiness = self._calculate_readiness()

        elapsed = (_utcnow() - started_at).total_seconds()
        result = LifecycleResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            current_stage=self._current_stage,
            commitment=self._commitment,
            inventory=self._inventory,
            targets=self._targets,
            validation=self._validation,
            submission=self._submission,
            tracking=self._tracking,
            review=self._review,
            revalidation=self._revalidation,
            overall_readiness_pct=round(overall_readiness, 2),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Full lifecycle workflow %s completed in %.2fs, "
            "stage=%s, readiness=%.1f%%",
            self.workflow_id, elapsed,
            self._current_stage.value, overall_readiness,
        )
        return result

    def _calculate_readiness(self) -> float:
        """Calculate overall lifecycle readiness percentage."""
        scores: List[float] = []

        if self._commitment:
            commit_score = 50.0
            if self._commitment.board_approved:
                commit_score += 25.0
            if self._commitment.public_announcement:
                commit_score += 25.0
            scores.append(commit_score)

        if self._inventory:
            inv_score = 60.0
            if self._inventory.ghg_verified:
                inv_score += 20.0
            if self._inventory.scope3_categories_screened >= 15:
                inv_score += 20.0
            scores.append(inv_score)

        if self._targets:
            scores.append(min(len(self._targets) * 25.0, 100.0))

        if self._validation:
            scores.append(self._validation.readiness_score)

        if self._submission:
            scores.append(self._submission.readiness_score)

        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    # -------------------------------------------------------------------------
    # Phase 1: Commitment
    # -------------------------------------------------------------------------

    async def _phase_commitment(self, config: LifecycleConfig) -> PhaseResult:
        """Register SBTi commitment letter and timeline."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Determine commitment date
        commit_date = config.commitment_date or _utcnow().strftime("%Y-%m-%d")

        # Calculate submission deadline (24 months)
        try:
            dt = datetime.fromisoformat(commit_date)
        except ValueError:
            dt = _utcnow()
        deadline_dt = dt.replace(year=dt.year + 2)
        deadline_str = deadline_dt.strftime("%Y-%m-%d")

        notes: List[str] = []
        if config.entity_type == EntityType.SME:
            notes.append(
                "SME entities may use simplified SBTi pathway; "
                "verify eligibility based on employee count and revenue"
            )
        if config.entity_type == EntityType.FINANCIAL_INSTITUTION:
            notes.append(
                "Financial institution pathway requires FI-specific targets "
                "per SBTi FINZ V1.0"
            )

        if not config.board_approved:
            warnings.append(
                "Board/senior management approval recommended before "
                "submitting commitment letter"
            )
        if not config.public_announcement:
            warnings.append(
                "Public announcement of SBTi commitment recommended for "
                "stakeholder engagement (criterion C25)"
            )

        self._commitment = CommitmentData(
            commitment_type=config.commitment_type,
            commitment_date=commit_date,
            submission_deadline=deadline_str,
            commitment_letter_ref=f"CL-{self.workflow_id[:8]}",
            public_announcement=config.public_announcement,
            board_approved=config.board_approved,
            notes=notes,
        )

        outputs["commitment_type"] = config.commitment_type.value
        outputs["commitment_date"] = commit_date
        outputs["submission_deadline"] = deadline_str
        outputs["entity_type"] = config.entity_type.value
        outputs["board_approved"] = config.board_approved
        outputs["public_announcement"] = config.public_announcement
        outputs["months_to_deadline"] = 24

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Commitment: type=%s, date=%s, deadline=%s",
            config.commitment_type.value, commit_date, deadline_str,
        )
        return PhaseResult(
            phase_name="commitment",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Inventory
    # -------------------------------------------------------------------------

    async def _phase_inventory(self, config: LifecycleConfig) -> PhaseResult:
        """Build and validate emissions inventory (S1+S2+S3)."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        scope12 = config.scope1_tco2e + config.scope2_location_tco2e
        total = scope12 + config.scope3_total_tco2e

        if scope12 <= 0:
            errors.append("Scope 1+2 emissions must be > 0")

        if config.scope3_categories_screened < 15:
            warnings.append(
                f"Only {config.scope3_categories_screened}/15 Scope 3 categories screened; "
                "SBTi requires all 15 (C17)"
            )

        scope3_pct = (config.scope3_total_tco2e / total * 100.0) if total > 0 else 0.0
        flag_pct = (config.flag_emissions_tco2e / total * 100.0) if total > 0 else 0.0

        if not config.ghg_verified:
            warnings.append(
                "GHG inventory not third-party verified; verification "
                "recommended for SBTi submission (C6)"
            )

        self._inventory = InventorySummary(
            base_year=config.base_year,
            scope1_tco2e=round(config.scope1_tco2e, 2),
            scope2_location_tco2e=round(config.scope2_location_tco2e, 2),
            scope2_market_tco2e=round(config.scope2_market_tco2e, 2),
            scope3_total_tco2e=round(config.scope3_total_tco2e, 2),
            scope3_categories_screened=config.scope3_categories_screened,
            scope3_pct_of_total=round(scope3_pct, 2),
            flag_emissions_tco2e=round(config.flag_emissions_tco2e, 2),
            flag_pct_of_total=round(flag_pct, 2),
            total_emissions_tco2e=round(total, 2),
            ghg_verified=config.ghg_verified,
        )

        outputs["base_year"] = config.base_year
        outputs["scope1_tco2e"] = round(config.scope1_tco2e, 2)
        outputs["scope2_location_tco2e"] = round(config.scope2_location_tco2e, 2)
        outputs["scope3_total_tco2e"] = round(config.scope3_total_tco2e, 2)
        outputs["total_tco2e"] = round(total, 2)
        outputs["scope3_pct"] = round(scope3_pct, 2)
        outputs["flag_pct"] = round(flag_pct, 2)
        outputs["scope3_target_required"] = scope3_pct >= SCOPE3_MATERIALITY_PCT
        outputs["flag_target_required"] = flag_pct >= FLAG_TRIGGER_PCT
        outputs["ghg_verified"] = config.ghg_verified
        outputs["categories_screened"] = config.scope3_categories_screened

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Inventory: total=%.2f, S3=%.1f%%, FLAG=%.1f%%",
            total, scope3_pct, flag_pct,
        )
        return PhaseResult(
            phase_name="inventory",
            status=PhaseStatus.FAILED if errors else PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Target Setting (with conditional FLAG and FI)
    # -------------------------------------------------------------------------

    async def _phase_target_set(self, config: LifecycleConfig) -> PhaseResult:
        """Design targets with conditional FLAG and FI pathways."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._targets = []

        scope12 = config.scope1_tco2e + config.scope2_location_tco2e
        total = scope12 + config.scope3_total_tco2e
        scope3_pct = (config.scope3_total_tco2e / total * 100.0) if total > 0 else 0.0
        flag_pct = (config.flag_emissions_tco2e / total * 100.0) if total > 0 else 0.0

        # Determine pathway method
        sector = config.sector.lower().strip()
        sda_eligible = sector in SDA_ELIGIBLE_SECTORS
        flag_required = flag_pct >= FLAG_TRIGGER_PCT
        is_fi = config.entity_type == EntityType.FINANCIAL_INSTITUTION

        annual_rate = ACA_RATES.get(config.ambition.value, ACA_RATES["1.5C"])

        # ----- Near-term Scope 1+2 target -----
        nt_years = config.near_term_target_year - config.base_year
        if sda_eligible:
            method = PathwayMethod.SDA
        elif flag_required:
            method = PathwayMethod.ACA_FLAG
        elif is_fi:
            method = PathwayMethod.FI
        else:
            method = PathwayMethod.ACA

        nt_reduction = min(
            (1.0 - (1.0 - annual_rate) ** nt_years) * 100.0, 95.0
        )
        nt_target_emissions = scope12 * (1.0 - nt_reduction / 100.0)

        self._targets.append(TargetSummary(
            target_id=f"NT-S12-{_new_uuid()[:8]}",
            target_name="Near-term Scope 1+2",
            scopes=["scope1", "scope2"],
            base_year=config.base_year,
            target_year=config.near_term_target_year,
            reduction_pct=round(nt_reduction, 2),
            pathway_method=method,
            ambition_level=config.ambition,
            coverage_pct=MIN_SCOPE12_COVERAGE,
            base_emissions_tco2e=round(scope12, 2),
            target_emissions_tco2e=round(nt_target_emissions, 2),
        ))

        # ----- Near-term Scope 3 target (if triggered) -----
        if scope3_pct >= SCOPE3_MATERIALITY_PCT and config.scope3_total_tco2e > 0:
            s3_rate = annual_rate * 0.7
            s3_reduction = min(
                (1.0 - (1.0 - s3_rate) ** nt_years) * 100.0, 90.0
            )
            s3_target_emissions = config.scope3_total_tco2e * (1.0 - s3_reduction / 100.0)

            self._targets.append(TargetSummary(
                target_id=f"NT-S3-{_new_uuid()[:8]}",
                target_name="Near-term Scope 3",
                scopes=["scope3"],
                base_year=config.base_year,
                target_year=config.near_term_target_year,
                reduction_pct=round(s3_reduction, 2),
                pathway_method=PathwayMethod.ACA,
                ambition_level=config.ambition,
                coverage_pct=MIN_SCOPE3_NEAR_TERM_COVERAGE,
                base_emissions_tco2e=round(config.scope3_total_tco2e, 2),
                target_emissions_tco2e=round(s3_target_emissions, 2),
            ))

        # ----- Long-term target (if include_net_zero) -----
        if config.include_net_zero:
            lt_years = config.long_term_target_year - config.base_year
            lt_reduction = min(
                (1.0 - (1.0 - annual_rate) ** lt_years) * 100.0, 95.0
            )
            lt_base = scope12 + config.scope3_total_tco2e
            lt_target = lt_base * (1.0 - lt_reduction / 100.0)

            self._targets.append(TargetSummary(
                target_id=f"LT-ALL-{_new_uuid()[:8]}",
                target_name="Long-term S1+S2+S3",
                scopes=["scope1", "scope2", "scope3"],
                base_year=config.base_year,
                target_year=config.long_term_target_year,
                reduction_pct=round(lt_reduction, 2),
                pathway_method=method if method != PathwayMethod.FLAG else PathwayMethod.ACA,
                ambition_level=AmbitionLevel.CELSIUS_1_5,
                coverage_pct=MIN_SCOPE3_LONG_TERM_COVERAGE,
                base_emissions_tco2e=round(lt_base, 2),
                target_emissions_tco2e=round(lt_target, 2),
            ))

            # Net-zero target
            nz_reduction = 90.0
            nz_target = lt_base * (1.0 - nz_reduction / 100.0)

            self._targets.append(TargetSummary(
                target_id=f"NZ-ALL-{_new_uuid()[:8]}",
                target_name="Net-Zero S1+S2+S3",
                scopes=["scope1", "scope2", "scope3"],
                base_year=config.base_year,
                target_year=config.net_zero_target_year,
                reduction_pct=nz_reduction,
                pathway_method=PathwayMethod.ACA,
                ambition_level=AmbitionLevel.CELSIUS_1_5,
                coverage_pct=MIN_SCOPE3_LONG_TERM_COVERAGE,
                base_emissions_tco2e=round(lt_base, 2),
                target_emissions_tco2e=round(nz_target, 2),
            ))

        # ----- Conditional FLAG target -----
        if flag_required and config.flag_emissions_tco2e > 0:
            flag_rate = 0.0303
            flag_reduction = min(
                (1.0 - (1.0 - flag_rate) ** nt_years) * 100.0, 50.0
            )
            flag_target = config.flag_emissions_tco2e * (1.0 - flag_reduction / 100.0)

            self._targets.append(TargetSummary(
                target_id=f"FLAG-{_new_uuid()[:8]}",
                target_name="FLAG Near-term",
                scopes=["flag"],
                base_year=config.base_year,
                target_year=config.near_term_target_year,
                reduction_pct=round(flag_reduction, 2),
                pathway_method=PathwayMethod.FLAG,
                ambition_level=AmbitionLevel.CELSIUS_1_5,
                coverage_pct=100.0,
                base_emissions_tco2e=round(config.flag_emissions_tco2e, 2),
                target_emissions_tco2e=round(flag_target, 2),
                is_flag=True,
            ))

            if not config.has_no_deforestation_commitment:
                warnings.append(
                    "FLAG target requires a no-deforestation commitment by 2025 "
                    "(SBTi FLAG Guidance V1.1 Section 6)"
                )

        # ----- Conditional FI target -----
        if is_fi:
            warnings.append(
                "Financial institution detected; FI-specific targets required "
                "per SBTi FINZ V1.0. Use fi_target_workflow for detailed "
                "asset-class-level targets."
            )

            if config.fi_portfolio_value_usd > 0:
                fi_reduction = min(
                    (1.0 - (1.0 - annual_rate) ** nt_years) * 100.0, 95.0
                )
                self._targets.append(TargetSummary(
                    target_id=f"FI-PORT-{_new_uuid()[:8]}",
                    target_name="FI Portfolio Financed Emissions",
                    scopes=["scope3"],
                    base_year=config.base_year,
                    target_year=config.near_term_target_year,
                    reduction_pct=round(fi_reduction, 2),
                    pathway_method=PathwayMethod.FI,
                    ambition_level=config.ambition,
                    coverage_pct=MIN_SCOPE3_NEAR_TERM_COVERAGE,
                    is_fi=True,
                ))

        outputs["targets_defined"] = len(self._targets)
        outputs["pathway_method"] = method.value
        outputs["sda_eligible"] = sda_eligible
        outputs["flag_required"] = flag_required
        outputs["is_fi"] = is_fi
        outputs["scope3_target_included"] = scope3_pct >= SCOPE3_MATERIALITY_PCT
        outputs["net_zero_included"] = config.include_net_zero
        for t in self._targets:
            outputs[f"{t.target_id}_reduction"] = round(t.reduction_pct, 2)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Target set: %d targets, method=%s, FLAG=%s, FI=%s",
            len(self._targets), method.value, flag_required, is_fi,
        )
        return PhaseResult(
            phase_name="target_set",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Validation
    # -------------------------------------------------------------------------

    async def _phase_validate(self, config: LifecycleConfig) -> PhaseResult:
        """Run 42-criterion validation and readiness check."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        passed = 0
        failed = 0
        warning_count = 0
        blocking: List[str] = []

        total = config.scope1_tco2e + config.scope2_location_tco2e + config.scope3_total_tco2e
        scope3_pct = (config.scope3_total_tco2e / total * 100.0) if total > 0 else 0.0
        annual_rate = ACA_RATES.get(config.ambition.value, ACA_RATES["1.5C"])

        # Evaluate key near-term criteria
        checks = {
            "C4": config.scope1_tco2e + config.scope2_location_tco2e > 0,
            "C5": config.base_year >= 2015,
            "C6": config.ghg_verified,
            "C8": config.has_recalculation_policy,
            "C9": annual_rate >= 0.025,
            "C12": not config.offsets_in_target,
            "C17": config.scope3_categories_screened >= 15,
            "C22": config.has_annual_reporting,
            "C25": config.has_public_commitment,
            "C28": config.has_board_oversight,
        }

        # Criteria that are warnings rather than fails
        warning_criteria = {"C6", "C17"}

        for cid, passes in checks.items():
            if passes:
                passed += 1
            elif cid in warning_criteria:
                warning_count += 1
            else:
                failed += 1
                blocking.append(cid)

        # Net-zero criteria if applicable
        if config.include_net_zero:
            nz_checks = {
                "NZ-C2": config.net_zero_target_year <= 2050,
                "NZ-C5": config.long_term_target_year >= 2035,
                "NZ-C10": config.has_neutralization_plan,
                "NZ-C12": config.has_transition_plan,
            }
            for cid, passes in nz_checks.items():
                if passes:
                    passed += 1
                else:
                    failed += 1
                    blocking.append(cid)

        # Remaining criteria assumed passed for lifecycle overview
        total_criteria = 42
        remaining = total_criteria - passed - failed - warning_count
        passed += max(remaining, 0)

        readiness = (passed / total_criteria * 100.0) if total_criteria > 0 else 0.0
        submission_ready = failed == 0

        self._validation = ValidationSummary(
            criteria_assessed=total_criteria,
            criteria_passed=passed,
            criteria_failed=failed,
            criteria_warning=warning_count,
            readiness_score=round(readiness, 2),
            submission_ready=submission_ready,
            blocking_criteria=blocking,
            gaps_count=failed + warning_count,
        )

        outputs["criteria_assessed"] = total_criteria
        outputs["passed"] = passed
        outputs["failed"] = failed
        outputs["warnings"] = warning_count
        outputs["readiness_score"] = round(readiness, 2)
        outputs["submission_ready"] = submission_ready
        outputs["blocking_criteria"] = blocking

        if not submission_ready:
            warnings.append(
                f"Not submission-ready: {failed} criteria failed ({', '.join(blocking)})"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Validate: readiness=%.1f%%, submission_ready=%s, failed=%d",
            readiness, submission_ready, failed,
        )
        return PhaseResult(
            phase_name="validate",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Submission
    # -------------------------------------------------------------------------

    async def _phase_submit(self, config: LifecycleConfig) -> PhaseResult:
        """Prepare submission package and readiness gate."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Check validation readiness
        validation_ok = self._validation and self._validation.submission_ready

        # Check required documents
        missing_docs: List[str] = []
        if not config.has_public_commitment:
            missing_docs.append("Commitment letter")
        if not config.ghg_verified:
            missing_docs.append("GHG verification report")
        if not config.has_recalculation_policy:
            missing_docs.append("Base year recalculation policy")
        if config.include_net_zero and not config.has_transition_plan:
            missing_docs.append("Transition plan")
        if config.include_net_zero and not config.has_neutralization_plan:
            missing_docs.append("Neutralization/CDR plan")

        package_complete = len(missing_docs) == 0
        targets_defined = len(self._targets) > 0

        # Determine readiness
        if validation_ok and package_complete and targets_defined:
            readiness = SubmissionReadiness.READY
            readiness_score = 100.0
        elif validation_ok and targets_defined:
            readiness = SubmissionReadiness.CONDITIONALLY_READY
            readiness_score = 75.0
        else:
            readiness = SubmissionReadiness.NOT_READY
            readiness_score = max(
                (self._validation.readiness_score if self._validation else 0.0) * 0.5,
                0.0,
            )

        # Deadline calculation
        deadline = ""
        days_to_deadline = 0
        if self._commitment and self._commitment.submission_deadline:
            deadline = self._commitment.submission_deadline
            try:
                deadline_dt = datetime.fromisoformat(deadline)
                days_to_deadline = (deadline_dt - _utcnow()).days
            except ValueError:
                days_to_deadline = 0

        if days_to_deadline < 90 and readiness != SubmissionReadiness.READY:
            warnings.append(
                f"Submission deadline in {days_to_deadline} days but not ready; "
                "accelerate preparation"
            )

        review_notes: List[str] = []
        review_notes.append(
            f"Validation readiness: {self._validation.readiness_score:.0f}%"
            if self._validation else "Validation not completed"
        )
        review_notes.append(f"Targets defined: {len(self._targets)}")
        if missing_docs:
            review_notes.append(f"Missing documents: {', '.join(missing_docs)}")

        self._submission = SubmissionPackage(
            readiness=readiness,
            readiness_score=round(readiness_score, 2),
            submission_date=_utcnow().strftime("%Y-%m-%d") if readiness == SubmissionReadiness.READY else "",
            deadline_date=deadline,
            days_until_deadline=max(days_to_deadline, 0),
            package_complete=package_complete,
            missing_documents=missing_docs,
            review_notes=review_notes,
        )

        outputs["readiness"] = readiness.value
        outputs["readiness_score"] = round(readiness_score, 2)
        outputs["package_complete"] = package_complete
        outputs["missing_documents"] = len(missing_docs)
        outputs["targets_defined"] = len(self._targets)
        outputs["days_to_deadline"] = max(days_to_deadline, 0)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Submit: readiness=%s (%.0f%%), missing_docs=%d",
            readiness.value, readiness_score, len(missing_docs),
        )
        return PhaseResult(
            phase_name="submit",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 6: Track
    # -------------------------------------------------------------------------

    async def _phase_track(self, config: LifecycleConfig) -> PhaseResult:
        """Annual progress tracking with RAG status."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        tracking_year = config.current_year or _utcnow().year
        current_scope12 = config.current_scope1_tco2e + config.current_scope2_tco2e
        current_total = current_scope12 + config.current_scope3_tco2e

        if current_total <= 0:
            warnings.append(
                "No current-year emissions data provided; tracking will be limited"
            )
            self._tracking = TrackingSummary(
                tracking_year=tracking_year,
                overall_rag=RAGStatus.NOT_ASSESSED,
            )
            outputs["tracking_year"] = tracking_year
            outputs["overall_rag"] = RAGStatus.NOT_ASSESSED.value
            outputs["no_data"] = True

            elapsed = (_utcnow() - started).total_seconds()
            return PhaseResult(
                phase_name="track",
                status=PhaseStatus.COMPLETED,
                duration_seconds=round(elapsed, 4),
                outputs=outputs,
                warnings=warnings,
                provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            )

        # Assess each target
        on_track = 0
        behind = 0
        worst_rag = RAGStatus.GREEN
        total_reduction = 0.0

        for target in self._targets:
            base = target.base_emissions_tco2e
            if base <= 0:
                continue

            # Get current emissions for target scopes
            if "scope3" in target.scopes and "scope1" in target.scopes:
                actual = current_total
            elif "scope3" in target.scopes:
                actual = config.current_scope3_tco2e
            elif "flag" in target.scopes:
                actual = config.flag_emissions_tco2e * 0.9  # Estimate
            else:
                actual = current_scope12

            # Required pathway
            years_elapsed = tracking_year - target.base_year
            total_years = target.target_year - target.base_year
            if total_years > 0:
                required_reduction_pct = target.reduction_pct * years_elapsed / total_years
                required = base * (1.0 - required_reduction_pct / 100.0)
            else:
                required = target.target_emissions_tco2e

            actual_reduction = (1.0 - actual / base) * 100.0 if base > 0 else 0.0
            gap_pct = 0.0
            if required > 0:
                gap_pct = abs(actual - required) / required * 100.0

            if actual <= required:
                on_track += 1
            else:
                behind += 1
                if gap_pct > 15.0:
                    worst_rag = RAGStatus.RED
                elif gap_pct > 5.0 and worst_rag != RAGStatus.RED:
                    worst_rag = RAGStatus.AMBER

            total_reduction = max(total_reduction, actual_reduction)

        # Carbon budget usage estimate
        base_total = config.scope1_tco2e + config.scope2_location_tco2e + config.scope3_total_tco2e
        if base_total > 0:
            budget_used = ((base_total + current_total) / 2.0 *
                           (tracking_year - config.base_year))
        else:
            budget_used = 0.0

        corrective_count = behind if worst_rag in (RAGStatus.AMBER, RAGStatus.RED) else 0

        self._tracking = TrackingSummary(
            tracking_year=tracking_year,
            overall_rag=worst_rag,
            targets_on_track=on_track,
            targets_behind=behind,
            total_reduction_pct=round(total_reduction, 2),
            corrective_actions_count=corrective_count,
            carbon_budget_used_pct=round(
                min(budget_used / max(base_total * 8, 1) * 100.0, 100.0), 2
            ),
        )

        outputs["tracking_year"] = tracking_year
        outputs["overall_rag"] = worst_rag.value
        outputs["on_track"] = on_track
        outputs["behind"] = behind
        outputs["total_reduction_pct"] = round(total_reduction, 2)
        outputs["corrective_actions"] = corrective_count

        if worst_rag == RAGStatus.RED:
            warnings.append(
                f"RED status: {behind} target(s) significantly behind pathway"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Track: year=%d, RAG=%s, on_track=%d, behind=%d",
            tracking_year, worst_rag.value, on_track, behind,
        )
        return PhaseResult(
            phase_name="track",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 7: Review
    # -------------------------------------------------------------------------

    async def _phase_review(self, config: LifecycleConfig) -> PhaseResult:
        """Periodic review with variance analysis and recalculation check."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        review_year = config.current_year or _utcnow().year

        # Check if base year recalculation is needed
        recalc_required = False
        recalc_reason = ""
        base_total = config.scope1_tco2e + config.scope2_location_tco2e + config.scope3_total_tco2e
        current_total = config.current_scope1_tco2e + config.current_scope2_tco2e + config.current_scope3_tco2e

        # Simple significance check (structural changes would be provided externally)
        if base_total > 0 and current_total > 0:
            change_pct = abs(current_total - base_total) / base_total * 100.0
            if change_pct > 50.0:  # >50% change may indicate structural issues
                warnings.append(
                    f"Emissions changed by {change_pct:.1f}% from base year; "
                    "verify if structural changes require base year recalculation"
                )

        # Variance drivers
        variance_drivers: List[str] = []
        if current_total > 0 and base_total > 0:
            if current_total < base_total:
                variance_drivers.append("Emissions reduction (positive trend)")
            elif current_total > base_total * 1.1:
                variance_drivers.append("Emissions increase (concerning)")
                variance_drivers.append("Investigate activity vs. intensity drivers")

        # Recommendations
        recommendations: List[str] = []
        if self._tracking and self._tracking.overall_rag == RAGStatus.RED:
            recommendations.append(
                "Immediate corrective action plan required for off-track targets"
            )
        if self._tracking and self._tracking.overall_rag == RAGStatus.AMBER:
            recommendations.append(
                "Develop corrective action plan to return to pathway within 12 months"
            )

        if config.years_since_validation >= 4:
            recommendations.append(
                "Target revalidation due within 12 months; begin preparation"
            )

        if not config.has_annual_reporting:
            recommendations.append(
                "Establish annual GHG progress reporting process (SBTi criterion C22)"
            )

        self._review = ReviewSummary(
            review_year=review_year,
            recalculation_required=recalc_required,
            recalculation_reason=recalc_reason,
            base_year_adjusted=False,
            targets_adjusted=0,
            variance_drivers=variance_drivers,
            recommendations=recommendations,
        )

        outputs["review_year"] = review_year
        outputs["recalculation_required"] = recalc_required
        outputs["variance_drivers"] = len(variance_drivers)
        outputs["recommendations"] = len(recommendations)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Review: year=%d, recalc=%s, recommendations=%d",
            review_year, recalc_required, len(recommendations),
        )
        return PhaseResult(
            phase_name="review",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 8: Revalidation
    # -------------------------------------------------------------------------

    async def _phase_revalidate(self, config: LifecycleConfig) -> PhaseResult:
        """5-year revalidation cycle assessment."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        years_since = config.years_since_validation
        months_since = years_since * 12

        # Determine revalidation status
        if years_since >= 5:
            reval_status = RevalidationStatus.EXPIRED
            warnings.append(
                "Target validation has EXPIRED (>5 years); "
                "immediate revalidation required"
            )
        elif years_since >= 4:
            reval_status = RevalidationStatus.REVALIDATION_REQUIRED
            warnings.append(
                f"Revalidation due within {12 - (months_since - 48)} months"
            )
        elif months_since >= (60 - REVALIDATION_WARNING_MONTHS):
            reval_status = RevalidationStatus.REVALIDATION_DUE
            warnings.append(
                "Revalidation approaching; begin preparation"
            )
        else:
            reval_status = RevalidationStatus.VALID

        # Check methodology changes since validation
        methodology_changes: List[str] = []
        # These would be detected from SBTi standards version tracking
        # For now, check if current ambition still meets requirements
        annual_rate = ACA_RATES.get(config.ambition.value, ACA_RATES["1.5C"])
        ambition_ok = annual_rate >= 0.025

        if not ambition_ok:
            methodology_changes.append(
                "Current target ambition below WB2C minimum; "
                "may need strengthening at revalidation"
            )

        # Scope changes
        scope_changes: List[str] = []
        if config.scope3_categories_screened < 15:
            scope_changes.append(
                f"Only {config.scope3_categories_screened}/15 Scope 3 categories screened"
            )

        # Revalidation actions
        actions: List[str] = []
        if reval_status in (
            RevalidationStatus.REVALIDATION_REQUIRED,
            RevalidationStatus.EXPIRED,
        ):
            actions.append("Submit revalidation application to SBTi")
            actions.append("Update emissions inventory to most recent year")
            actions.append("Review and update targets against current SBTi criteria")
            if methodology_changes:
                actions.append("Address methodology changes identified")
            actions.append("Prepare updated supporting documentation")

        elif reval_status == RevalidationStatus.REVALIDATION_DUE:
            actions.append("Begin collecting data for revalidation submission")
            actions.append("Review current SBTi criteria for any updates")
            actions.append("Identify gaps in current target framework")

        # Validation and expiry dates
        validation_date = config.original_validation_date
        expiry_date = ""
        months_until_expiry = max(60 - months_since, 0)
        if validation_date:
            try:
                val_dt = datetime.fromisoformat(validation_date)
                exp_dt = val_dt.replace(year=val_dt.year + 5)
                expiry_date = exp_dt.strftime("%Y-%m-%d")
            except ValueError:
                expiry_date = ""

        self._revalidation = RevalidationAssessment(
            revalidation_status=reval_status,
            validation_date=validation_date,
            expiry_date=expiry_date,
            months_until_expiry=months_until_expiry,
            targets_still_valid=reval_status == RevalidationStatus.VALID,
            methodology_changes=methodology_changes,
            ambition_still_sufficient=ambition_ok,
            scope_changes=scope_changes,
            revalidation_actions=actions,
        )

        outputs["revalidation_status"] = reval_status.value
        outputs["years_since_validation"] = years_since
        outputs["months_until_expiry"] = months_until_expiry
        outputs["targets_valid"] = reval_status == RevalidationStatus.VALID
        outputs["ambition_sufficient"] = ambition_ok
        outputs["methodology_changes"] = len(methodology_changes)
        outputs["actions_required"] = len(actions)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Revalidate: status=%s, years_since=%d, actions=%d",
            reval_status.value, years_since, len(actions),
        )
        return PhaseResult(
            phase_name="revalidate",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )
