# -*- coding: utf-8 -*-
"""
Maturity Assessment Workflow
===================================

4-phase workflow for assessing and planning Scope 3 data maturity upgrades
within PACK-043 Scope 3 Complete Pack.

Phases:
    1. CURRENT_STATE_SCAN     -- Assess current methodology tier, data quality
                                 rating (DQR), and uncertainty per category
                                 from PACK-042 screening/calculation data.
    2. GAP_ANALYSIS           -- Compare current state to target maturity level,
                                 identify gaps per category and cross-cutting.
    3. UPGRADE_ROADMAP        -- Generate ordered sequence of tier upgrades with
                                 dependency mapping and effort estimates.
    4. ROI_PRIORITIZATION     -- Calculate ROI per upgrade, rank by cost-
                                 effectiveness, allocate across budget envelope.

The workflow follows GreenLang zero-hallucination principles: every score,
gap measurement, and ROI figure is derived from deterministic formulas applied
to auditable input data. SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Scope 3 Standard -- Chapter 7 (Data Quality)
    GHG Protocol Technical Guidance -- Appendix on Data Quality Indicators
    PCAF Standard -- Data Quality Scores (1-5)
    SBTi Corporate Net-Zero Standard -- Scope 3 data quality expectations

Schedule: annually or upon inventory refresh
Estimated duration: 2-4 hours

Author: GreenLang Platform Team
Version: 43.0.0
"""

_MODULE_VERSION: str = "43.0.0"

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

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


class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories (1-15)."""

    CAT_01_PURCHASED_GOODS = "cat_01_purchased_goods_services"
    CAT_02_CAPITAL_GOODS = "cat_02_capital_goods"
    CAT_03_FUEL_ENERGY = "cat_03_fuel_energy_related"
    CAT_04_UPSTREAM_TRANSPORT = "cat_04_upstream_transport"
    CAT_05_WASTE = "cat_05_waste_in_operations"
    CAT_06_BUSINESS_TRAVEL = "cat_06_business_travel"
    CAT_07_COMMUTING = "cat_07_employee_commuting"
    CAT_08_UPSTREAM_LEASED = "cat_08_upstream_leased_assets"
    CAT_09_DOWNSTREAM_TRANSPORT = "cat_09_downstream_transport"
    CAT_10_PROCESSING = "cat_10_processing_sold_products"
    CAT_11_USE_SOLD = "cat_11_use_of_sold_products"
    CAT_12_END_OF_LIFE = "cat_12_end_of_life_treatment"
    CAT_13_DOWNSTREAM_LEASED = "cat_13_downstream_leased_assets"
    CAT_14_FRANCHISES = "cat_14_franchises"
    CAT_15_INVESTMENTS = "cat_15_investments"


class MethodologyTier(str, Enum):
    """Methodology tier for Scope 3 calculation (ascending quality)."""

    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    HYBRID = "hybrid"
    SUPPLIER_SPECIFIC = "supplier_specific"
    VERIFIED = "verified"
    NOT_APPLICABLE = "not_applicable"


class MaturityLevel(str, Enum):
    """Overall Scope 3 maturity level."""

    LEVEL_1_INITIAL = "level_1_initial"
    LEVEL_2_DEVELOPING = "level_2_developing"
    LEVEL_3_DEFINED = "level_3_defined"
    LEVEL_4_MANAGED = "level_4_managed"
    LEVEL_5_OPTIMIZING = "level_5_optimizing"


class GapSeverity(str, Enum):
    """Severity of an identified gap."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class UpgradeEffort(str, Enum):
    """Effort level for a tier upgrade."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Tier numeric values for comparison
TIER_NUMERIC: Dict[str, int] = {
    MethodologyTier.SPEND_BASED.value: 1,
    MethodologyTier.AVERAGE_DATA.value: 2,
    MethodologyTier.HYBRID.value: 3,
    MethodologyTier.SUPPLIER_SPECIFIC.value: 4,
    MethodologyTier.VERIFIED.value: 5,
    MethodologyTier.NOT_APPLICABLE.value: 0,
}

# DQR score ranges per tier (min, max)
TIER_DQR_RANGES: Dict[str, Tuple[float, float]] = {
    MethodologyTier.SPEND_BASED.value: (1.0, 2.0),
    MethodologyTier.AVERAGE_DATA.value: (2.0, 3.0),
    MethodologyTier.HYBRID.value: (2.5, 3.5),
    MethodologyTier.SUPPLIER_SPECIFIC.value: (3.5, 4.5),
    MethodologyTier.VERIFIED.value: (4.5, 5.0),
}

# Typical uncertainty ranges per tier (% of estimate)
TIER_UNCERTAINTY: Dict[str, Tuple[float, float]] = {
    MethodologyTier.SPEND_BASED.value: (50.0, 100.0),
    MethodologyTier.AVERAGE_DATA.value: (30.0, 60.0),
    MethodologyTier.HYBRID.value: (20.0, 40.0),
    MethodologyTier.SUPPLIER_SPECIFIC.value: (10.0, 25.0),
    MethodologyTier.VERIFIED.value: (5.0, 15.0),
}

# Upgrade cost estimates (USD) for one-step tier upgrade per category
UPGRADE_COST_ESTIMATES_USD: Dict[str, float] = {
    "spend_based_to_average_data": 15_000.0,
    "average_data_to_hybrid": 25_000.0,
    "hybrid_to_supplier_specific": 50_000.0,
    "supplier_specific_to_verified": 35_000.0,
}

# Typical duration in weeks for each upgrade step
UPGRADE_DURATION_WEEKS: Dict[str, float] = {
    "spend_based_to_average_data": 4.0,
    "average_data_to_hybrid": 6.0,
    "hybrid_to_supplier_specific": 12.0,
    "supplier_specific_to_verified": 8.0,
}

# Maturity level thresholds (average tier across categories)
MATURITY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    MaturityLevel.LEVEL_1_INITIAL.value: (0.0, 1.5),
    MaturityLevel.LEVEL_2_DEVELOPING.value: (1.5, 2.5),
    MaturityLevel.LEVEL_3_DEFINED.value: (2.5, 3.5),
    MaturityLevel.LEVEL_4_MANAGED.value: (3.5, 4.5),
    MaturityLevel.LEVEL_5_OPTIMIZING.value: (4.5, 5.1),
}

# Human-readable category names
CATEGORY_NAMES: Dict[Scope3Category, str] = {
    Scope3Category.CAT_01_PURCHASED_GOODS: "Purchased Goods & Services",
    Scope3Category.CAT_02_CAPITAL_GOODS: "Capital Goods",
    Scope3Category.CAT_03_FUEL_ENERGY: "Fuel- & Energy-Related Activities",
    Scope3Category.CAT_04_UPSTREAM_TRANSPORT: "Upstream Transportation & Distribution",
    Scope3Category.CAT_05_WASTE: "Waste Generated in Operations",
    Scope3Category.CAT_06_BUSINESS_TRAVEL: "Business Travel",
    Scope3Category.CAT_07_COMMUTING: "Employee Commuting",
    Scope3Category.CAT_08_UPSTREAM_LEASED: "Upstream Leased Assets",
    Scope3Category.CAT_09_DOWNSTREAM_TRANSPORT: "Downstream Transportation & Distribution",
    Scope3Category.CAT_10_PROCESSING: "Processing of Sold Products",
    Scope3Category.CAT_11_USE_SOLD: "Use of Sold Products",
    Scope3Category.CAT_12_END_OF_LIFE: "End-of-Life Treatment of Sold Products",
    Scope3Category.CAT_13_DOWNSTREAM_LEASED: "Downstream Leased Assets",
    Scope3Category.CAT_14_FRANCHISES: "Franchises",
    Scope3Category.CAT_15_INVESTMENTS: "Investments",
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(
        default_factory=dict, description="Phase output data"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class WorkflowState(BaseModel):
    """Persistent state for checkpoint/resume capability."""

    workflow_id: str = Field(default="", description="Unique workflow execution ID")
    current_phase: int = Field(default=0, description="Last completed phase number")
    phase_statuses: Dict[str, str] = Field(
        default_factory=dict, description="Phase name -> status"
    )
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    checkpoint_data: Dict[str, Any] = Field(
        default_factory=dict, description="Serialized intermediate data"
    )
    created_at: str = Field(default="", description="ISO-8601 timestamp")
    updated_at: str = Field(default="", description="ISO-8601 timestamp")


class CategoryMaturityState(BaseModel):
    """Current maturity state for a single Scope 3 category."""

    category: Scope3Category = Field(...)
    category_name: str = Field(default="")
    current_tier: MethodologyTier = Field(default=MethodologyTier.SPEND_BASED)
    dqr_score: float = Field(default=1.0, ge=1.0, le=5.0, description="Data quality rating 1-5")
    uncertainty_pct: float = Field(default=100.0, ge=0.0, le=200.0)
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    is_applicable: bool = Field(default=True)
    data_sources_count: int = Field(default=0, ge=0)
    supplier_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class MaturityGap(BaseModel):
    """Identified gap between current and target maturity."""

    category: Scope3Category = Field(...)
    category_name: str = Field(default="")
    current_tier: MethodologyTier = Field(default=MethodologyTier.SPEND_BASED)
    target_tier: MethodologyTier = Field(default=MethodologyTier.SUPPLIER_SPECIFIC)
    tier_gap: int = Field(default=0, ge=0, description="Number of tiers to upgrade")
    current_dqr: float = Field(default=1.0, ge=1.0, le=5.0)
    target_dqr: float = Field(default=4.0, ge=1.0, le=5.0)
    dqr_gap: float = Field(default=0.0, ge=0.0)
    severity: GapSeverity = Field(default=GapSeverity.MEDIUM)
    uncertainty_reduction_pct: float = Field(default=0.0, ge=0.0)
    description: str = Field(default="")


class UpgradeStep(BaseModel):
    """Single step in an upgrade roadmap."""

    step_id: str = Field(default_factory=lambda: f"step-{uuid.uuid4().hex[:8]}")
    category: Scope3Category = Field(...)
    category_name: str = Field(default="")
    from_tier: MethodologyTier = Field(...)
    to_tier: MethodologyTier = Field(...)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    estimated_duration_weeks: float = Field(default=0.0, ge=0.0)
    effort: UpgradeEffort = Field(default=UpgradeEffort.MEDIUM)
    dependencies: List[str] = Field(
        default_factory=list, description="Step IDs that must complete first"
    )
    prerequisites: List[str] = Field(
        default_factory=list, description="Human-readable prerequisite descriptions"
    )
    expected_dqr_improvement: float = Field(default=0.0, ge=0.0)
    expected_uncertainty_reduction_pct: float = Field(default=0.0, ge=0.0)
    priority_rank: int = Field(default=0, ge=0)


class UpgradeROI(BaseModel):
    """ROI analysis for a single upgrade step."""

    step_id: str = Field(default="")
    category: Scope3Category = Field(...)
    category_name: str = Field(default="")
    upgrade_cost_usd: float = Field(default=0.0, ge=0.0)
    uncertainty_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    cost_per_tco2e_precision: float = Field(
        default=0.0, ge=0.0, description="USD cost per tCO2e of uncertainty reduced"
    )
    roi_score: float = Field(default=0.0, description="Higher is better, 0-100")
    cost_effectiveness_rank: int = Field(default=0, ge=0)
    cumulative_budget_usd: float = Field(default=0.0, ge=0.0)
    within_budget: bool = Field(default=True)


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class MaturityAssessmentInput(BaseModel):
    """Input data model for MaturityAssessmentWorkflow."""

    organization_name: str = Field(default="", description="Organization name")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    category_states: List[CategoryMaturityState] = Field(
        default_factory=list,
        description="Current maturity state per category from PACK-042",
    )
    target_maturity: MaturityLevel = Field(
        default=MaturityLevel.LEVEL_3_DEFINED,
        description="Target overall maturity level",
    )
    target_tier_overrides: Dict[str, str] = Field(
        default_factory=dict,
        description="Category -> override target tier (e.g. cat_01 -> supplier_specific)",
    )
    budget_usd: float = Field(
        default=500_000.0, ge=0.0, description="Total budget for upgrades"
    )
    time_horizon_months: int = Field(
        default=24, ge=6, le=60, description="Time horizon for roadmap"
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("category_states")
    @classmethod
    def validate_category_states(cls, v: List[CategoryMaturityState]) -> List[CategoryMaturityState]:
        """Ensure no duplicate categories."""
        seen: set = set()
        for cs in v:
            if cs.category.value in seen:
                raise ValueError(f"Duplicate category: {cs.category.value}")
            seen.add(cs.category.value)
        return v


class MaturityAssessmentOutput(BaseModel):
    """Complete output from MaturityAssessmentWorkflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="maturity_assessment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_name: str = Field(default="")
    reporting_year: int = Field(default=2025)
    current_maturity: MaturityLevel = Field(default=MaturityLevel.LEVEL_1_INITIAL)
    target_maturity: MaturityLevel = Field(default=MaturityLevel.LEVEL_3_DEFINED)
    current_avg_tier: float = Field(default=1.0)
    current_avg_dqr: float = Field(default=1.0)
    category_states: List[CategoryMaturityState] = Field(default_factory=list)
    gaps: List[MaturityGap] = Field(default_factory=list)
    upgrade_roadmap: List[UpgradeStep] = Field(default_factory=list)
    roi_analysis: List[UpgradeROI] = Field(default_factory=list)
    total_upgrade_cost_usd: float = Field(default=0.0, ge=0.0)
    total_upgrade_duration_weeks: float = Field(default=0.0, ge=0.0)
    budget_usd: float = Field(default=0.0, ge=0.0)
    budget_utilization_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class MaturityAssessmentWorkflow:
    """
    4-phase Scope 3 maturity assessment and upgrade planning workflow.

    Assesses current methodology tier, DQR, and uncertainty per category,
    compares against target maturity, generates an ordered upgrade roadmap
    with dependencies, and prioritizes upgrades by ROI / cost-effectiveness.

    Zero-hallucination: all scores, gaps, costs, and ROI figures are derived
    from deterministic reference data and arithmetic. No LLM calls in any
    numeric path.

    Attributes:
        workflow_id: Unique execution identifier.
        _category_states: Current maturity per category.
        _gaps: Identified maturity gaps.
        _roadmap: Ordered upgrade steps.
        _roi_analysis: ROI per upgrade step.
        _phase_results: Ordered phase outputs.
        _state: Checkpoint/resume state.

    Example:
        >>> wf = MaturityAssessmentWorkflow()
        >>> inp = MaturityAssessmentInput(
        ...     organization_name="Acme Corp",
        ...     category_states=[
        ...         CategoryMaturityState(
        ...             category=Scope3Category.CAT_01_PURCHASED_GOODS,
        ...             current_tier=MethodologyTier.SPEND_BASED,
        ...             dqr_score=1.5,
        ...             emissions_tco2e=50000,
        ...         )
        ...     ],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_NAMES: List[str] = [
        "current_state_scan",
        "gap_analysis",
        "upgrade_roadmap",
        "roi_prioritization",
    ]

    PHASE_WEIGHTS: Dict[str, float] = {
        "current_state_scan": 20.0,
        "gap_analysis": 25.0,
        "upgrade_roadmap": 30.0,
        "roi_prioritization": 25.0,
    }

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize MaturityAssessmentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._category_states: List[CategoryMaturityState] = []
        self._gaps: List[MaturityGap] = []
        self._roadmap: List[UpgradeStep] = []
        self._roi_analysis: List[UpgradeROI] = []
        self._phase_results: List[PhaseResult] = []
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )
        self._current_maturity: MaturityLevel = MaturityLevel.LEVEL_1_INITIAL
        self._avg_tier: float = 1.0
        self._avg_dqr: float = 1.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[MaturityAssessmentInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> MaturityAssessmentOutput:
        """
        Execute the 4-phase maturity assessment workflow.

        Args:
            input_data: Full input model (preferred).
            config: Optional configuration overrides.

        Returns:
            MaturityAssessmentOutput with gaps, roadmap, and ROI analysis.

        Raises:
            ValueError: If required data is missing.
        """
        if input_data is None:
            input_data = MaturityAssessmentInput()

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting maturity assessment workflow %s org=%s categories=%d",
            self.workflow_id,
            input_data.organization_name,
            len(input_data.category_states),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING
        self._update_progress(0.0)

        try:
            # Phase 1: Current State Scan
            phase1 = await self._execute_with_retry(
                self._phase_current_state_scan, input_data, phase_number=1
            )
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 1 failed: {phase1.errors}")
            self._update_progress(20.0)

            # Phase 2: Gap Analysis
            phase2 = await self._execute_with_retry(
                self._phase_gap_analysis, input_data, phase_number=2
            )
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 2 failed: {phase2.errors}")
            self._update_progress(45.0)

            # Phase 3: Upgrade Roadmap
            phase3 = await self._execute_with_retry(
                self._phase_upgrade_roadmap, input_data, phase_number=3
            )
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 3 failed: {phase3.errors}")
            self._update_progress(75.0)

            # Phase 4: ROI Prioritization
            phase4 = await self._execute_with_retry(
                self._phase_roi_prioritization, input_data, phase_number=4
            )
            self._phase_results.append(phase4)
            if phase4.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 4 failed: {phase4.errors}")
            self._update_progress(100.0)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Maturity assessment workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(
                PhaseResult(
                    phase_name="error",
                    phase_number=0,
                    status=PhaseStatus.FAILED,
                    errors=[str(exc)],
                )
            )

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        total_cost = sum(s.estimated_cost_usd for s in self._roadmap)
        total_weeks = self._compute_roadmap_duration()
        budget_util = (
            (total_cost / input_data.budget_usd * 100.0)
            if input_data.budget_usd > 0
            else 0.0
        )

        result = MaturityAssessmentOutput(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_name=input_data.organization_name,
            reporting_year=input_data.reporting_year,
            current_maturity=self._current_maturity,
            target_maturity=input_data.target_maturity,
            current_avg_tier=round(self._avg_tier, 2),
            current_avg_dqr=round(self._avg_dqr, 2),
            category_states=self._category_states,
            gaps=self._gaps,
            upgrade_roadmap=self._roadmap,
            roi_analysis=self._roi_analysis,
            total_upgrade_cost_usd=round(total_cost, 2),
            total_upgrade_duration_weeks=round(total_weeks, 1),
            budget_usd=input_data.budget_usd,
            budget_utilization_pct=round(min(budget_util, 100.0), 1),
            progress_pct=self._state.progress_pct,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Maturity assessment workflow %s completed in %.2fs status=%s "
            "current=%s target=%s gaps=%d steps=%d cost=%.0f",
            self.workflow_id,
            elapsed,
            overall_status.value,
            self._current_maturity.value,
            input_data.target_maturity.value,
            len(self._gaps),
            len(self._roadmap),
            total_cost,
        )
        return result

    def get_state(self) -> WorkflowState:
        """Return current workflow state for checkpoint/resume."""
        return self._state.model_copy()

    async def resume(
        self,
        state: WorkflowState,
        input_data: MaturityAssessmentInput,
    ) -> MaturityAssessmentOutput:
        """Resume workflow from a saved checkpoint state."""
        self._state = state
        self.workflow_id = state.workflow_id
        self.logger.info(
            "Resuming workflow %s from phase %d",
            self.workflow_id,
            state.current_phase,
        )
        return await self.execute(input_data)

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self,
        phase_fn: Any,
        input_data: MaturityAssessmentInput,
        phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number,
                        attempt,
                        self.MAX_RETRIES,
                        exc,
                        delay,
                    )
                    import asyncio

                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Current State Scan
    # -------------------------------------------------------------------------

    async def _phase_current_state_scan(
        self, input_data: MaturityAssessmentInput
    ) -> PhaseResult:
        """Assess current tier, DQR, and uncertainty per category."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._category_states = list(input_data.category_states)

        # Enrich category states with names
        for cs in self._category_states:
            if not cs.category_name:
                cs.category_name = CATEGORY_NAMES.get(cs.category, cs.category.value)

        # Fill missing uncertainty from tier reference data
        for cs in self._category_states:
            if cs.uncertainty_pct == 100.0 and cs.current_tier != MethodologyTier.SPEND_BASED:
                tier_range = TIER_UNCERTAINTY.get(cs.current_tier.value, (50.0, 100.0))
                cs.uncertainty_pct = (tier_range[0] + tier_range[1]) / 2.0

        # Calculate aggregate metrics
        applicable = [cs for cs in self._category_states if cs.is_applicable]
        if not applicable:
            warnings.append("No applicable categories provided; using defaults")
            self._avg_tier = 1.0
            self._avg_dqr = 1.0
        else:
            total_emissions = sum(cs.emissions_tco2e for cs in applicable)
            # Emission-weighted average tier
            if total_emissions > 0:
                self._avg_tier = sum(
                    TIER_NUMERIC.get(cs.current_tier.value, 1) * cs.emissions_tco2e
                    for cs in applicable
                ) / total_emissions
                self._avg_dqr = sum(
                    cs.dqr_score * cs.emissions_tco2e for cs in applicable
                ) / total_emissions
            else:
                self._avg_tier = sum(
                    TIER_NUMERIC.get(cs.current_tier.value, 1) for cs in applicable
                ) / len(applicable)
                self._avg_dqr = sum(cs.dqr_score for cs in applicable) / len(applicable)

            # Compute pct_of_total if missing
            if total_emissions > 0:
                for cs in applicable:
                    cs.pct_of_total = round(
                        (cs.emissions_tco2e / total_emissions) * 100.0, 2
                    )

        # Determine current maturity level
        self._current_maturity = self._tier_to_maturity(self._avg_tier)

        # Tier distribution
        tier_dist: Dict[str, int] = {}
        for cs in applicable:
            t = cs.current_tier.value
            tier_dist[t] = tier_dist.get(t, 0) + 1

        outputs["applicable_categories"] = len(applicable)
        outputs["avg_tier_numeric"] = round(self._avg_tier, 2)
        outputs["avg_dqr_score"] = round(self._avg_dqr, 2)
        outputs["current_maturity"] = self._current_maturity.value
        outputs["tier_distribution"] = tier_dist
        outputs["total_emissions_tco2e"] = round(
            sum(cs.emissions_tco2e for cs in applicable), 2
        )
        outputs["weighted_uncertainty_pct"] = round(
            self._compute_weighted_uncertainty(applicable), 1
        )

        self._state.phase_statuses["current_state_scan"] = "completed"
        self._state.current_phase = 1

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 CurrentStateScan: maturity=%s avg_tier=%.2f avg_dqr=%.2f "
            "categories=%d",
            self._current_maturity.value,
            self._avg_tier,
            self._avg_dqr,
            len(applicable),
        )
        return PhaseResult(
            phase_name="current_state_scan",
            phase_number=1,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Gap Analysis
    # -------------------------------------------------------------------------

    async def _phase_gap_analysis(
        self, input_data: MaturityAssessmentInput
    ) -> PhaseResult:
        """Compare current state to target maturity, identify gaps."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._gaps = []
        target_maturity = input_data.target_maturity
        target_avg_tier = self._maturity_to_min_tier(target_maturity)

        for cs in self._category_states:
            if not cs.is_applicable:
                continue

            # Determine target tier for this category
            override_key = cs.category.value
            if override_key in input_data.target_tier_overrides:
                target_tier_str = input_data.target_tier_overrides[override_key]
                target_tier = MethodologyTier(target_tier_str)
            else:
                target_tier = self._determine_target_tier(cs, target_avg_tier)

            current_num = TIER_NUMERIC.get(cs.current_tier.value, 1)
            target_num = TIER_NUMERIC.get(target_tier.value, 3)
            tier_gap = max(target_num - current_num, 0)

            if tier_gap == 0:
                continue  # Already at or above target

            # DQR gap
            target_dqr_range = TIER_DQR_RANGES.get(target_tier.value, (3.5, 4.5))
            target_dqr = target_dqr_range[0]
            dqr_gap = max(target_dqr - cs.dqr_score, 0.0)

            # Uncertainty reduction
            current_unc = TIER_UNCERTAINTY.get(
                cs.current_tier.value, (50.0, 100.0)
            )
            target_unc = TIER_UNCERTAINTY.get(
                target_tier.value, (10.0, 25.0)
            )
            unc_reduction = max(
                ((current_unc[0] + current_unc[1]) / 2.0)
                - ((target_unc[0] + target_unc[1]) / 2.0),
                0.0,
            )

            # Severity based on tier gap and emission share
            severity = self._assess_gap_severity(tier_gap, cs.pct_of_total)

            gap = MaturityGap(
                category=cs.category,
                category_name=cs.category_name,
                current_tier=cs.current_tier,
                target_tier=target_tier,
                tier_gap=tier_gap,
                current_dqr=cs.dqr_score,
                target_dqr=round(target_dqr, 2),
                dqr_gap=round(dqr_gap, 2),
                severity=severity,
                uncertainty_reduction_pct=round(unc_reduction, 1),
                description=self._format_gap_description(cs, target_tier, tier_gap),
            )
            self._gaps.append(gap)

        # Sort gaps by severity then emission share
        severity_order = {
            GapSeverity.CRITICAL: 0,
            GapSeverity.HIGH: 1,
            GapSeverity.MEDIUM: 2,
            GapSeverity.LOW: 3,
            GapSeverity.NONE: 4,
        }
        self._gaps.sort(
            key=lambda g: (severity_order.get(g.severity, 4), -g.tier_gap)
        )

        outputs["total_gaps"] = len(self._gaps)
        outputs["critical_gaps"] = sum(
            1 for g in self._gaps if g.severity == GapSeverity.CRITICAL
        )
        outputs["high_gaps"] = sum(
            1 for g in self._gaps if g.severity == GapSeverity.HIGH
        )
        outputs["medium_gaps"] = sum(
            1 for g in self._gaps if g.severity == GapSeverity.MEDIUM
        )
        outputs["low_gaps"] = sum(
            1 for g in self._gaps if g.severity == GapSeverity.LOW
        )
        outputs["avg_tier_gap"] = round(
            sum(g.tier_gap for g in self._gaps) / len(self._gaps), 2
        ) if self._gaps else 0.0
        outputs["total_uncertainty_reduction_pct"] = round(
            sum(g.uncertainty_reduction_pct for g in self._gaps), 1
        )

        if not self._gaps:
            warnings.append(
                "No maturity gaps found; all categories meet or exceed target"
            )

        self._state.phase_statuses["gap_analysis"] = "completed"
        self._state.current_phase = 2

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 GapAnalysis: total_gaps=%d critical=%d high=%d",
            len(self._gaps),
            outputs["critical_gaps"],
            outputs["high_gaps"],
        )
        return PhaseResult(
            phase_name="gap_analysis",
            phase_number=2,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Upgrade Roadmap
    # -------------------------------------------------------------------------

    async def _phase_upgrade_roadmap(
        self, input_data: MaturityAssessmentInput
    ) -> PhaseResult:
        """Generate ordered sequence of tier upgrades with dependencies."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._roadmap = []

        for gap in self._gaps:
            steps = self._generate_upgrade_steps(gap)
            self._roadmap.extend(steps)

        # Assign dependencies: within same category, each step depends on previous
        self._assign_step_dependencies()

        # Sort by priority (severity weight * emission share)
        self._roadmap.sort(key=lambda s: s.priority_rank)

        total_cost = sum(s.estimated_cost_usd for s in self._roadmap)
        total_weeks = self._compute_roadmap_duration()

        if total_weeks > input_data.time_horizon_months * 4.33:
            warnings.append(
                f"Roadmap duration ({total_weeks:.0f} weeks) exceeds time horizon "
                f"({input_data.time_horizon_months} months = "
                f"{input_data.time_horizon_months * 4.33:.0f} weeks)"
            )

        outputs["total_steps"] = len(self._roadmap)
        outputs["total_estimated_cost_usd"] = round(total_cost, 2)
        outputs["total_estimated_duration_weeks"] = round(total_weeks, 1)
        outputs["steps_by_effort"] = {
            e.value: sum(1 for s in self._roadmap if s.effort == e)
            for e in UpgradeEffort
        }
        outputs["categories_with_upgrades"] = len(set(
            s.category.value for s in self._roadmap
        ))

        self._state.phase_statuses["upgrade_roadmap"] = "completed"
        self._state.current_phase = 3

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 UpgradeRoadmap: steps=%d cost=%.0f weeks=%.1f",
            len(self._roadmap),
            total_cost,
            total_weeks,
        )
        return PhaseResult(
            phase_name="upgrade_roadmap",
            phase_number=3,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: ROI Prioritization
    # -------------------------------------------------------------------------

    async def _phase_roi_prioritization(
        self, input_data: MaturityAssessmentInput
    ) -> PhaseResult:
        """Calculate ROI per upgrade, rank by cost-effectiveness."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._roi_analysis = []
        budget = input_data.budget_usd

        # Build category emission lookup
        emission_lookup: Dict[str, float] = {
            cs.category.value: cs.emissions_tco2e
            for cs in self._category_states
        }

        for step in self._roadmap:
            cat_emissions = emission_lookup.get(step.category.value, 0.0)

            # Uncertainty reduction in absolute tCO2e terms
            unc_reduction_tco2e = (
                cat_emissions * step.expected_uncertainty_reduction_pct / 100.0
            )

            # Cost per tCO2e of precision gained
            cost_per_tco2e = (
                step.estimated_cost_usd / unc_reduction_tco2e
                if unc_reduction_tco2e > 0
                else float("inf")
            )

            # ROI score: higher is better (normalized 0-100)
            # roi = (uncertainty_reduction_tco2e / cost) * scaling_factor
            if step.estimated_cost_usd > 0 and unc_reduction_tco2e > 0:
                raw_roi = unc_reduction_tco2e / step.estimated_cost_usd
                roi_score = min(raw_roi * 10000.0, 100.0)
            else:
                roi_score = 0.0

            self._roi_analysis.append(
                UpgradeROI(
                    step_id=step.step_id,
                    category=step.category,
                    category_name=step.category_name,
                    upgrade_cost_usd=round(step.estimated_cost_usd, 2),
                    uncertainty_reduction_tco2e=round(unc_reduction_tco2e, 2),
                    cost_per_tco2e_precision=round(
                        cost_per_tco2e if cost_per_tco2e != float("inf") else 0.0, 2
                    ),
                    roi_score=round(roi_score, 2),
                )
            )

        # Rank by ROI score descending
        self._roi_analysis.sort(key=lambda r: r.roi_score, reverse=True)

        # Assign ranks and cumulative budget
        cumulative = 0.0
        for rank, roi in enumerate(self._roi_analysis, 1):
            roi.cost_effectiveness_rank = rank
            cumulative += roi.upgrade_cost_usd
            roi.cumulative_budget_usd = round(cumulative, 2)
            roi.within_budget = cumulative <= budget

        # Budget allocation summary
        within_budget_count = sum(1 for r in self._roi_analysis if r.within_budget)
        allocated_cost = sum(
            r.upgrade_cost_usd for r in self._roi_analysis if r.within_budget
        )

        if budget > 0 and allocated_cost < budget * 0.5:
            warnings.append(
                f"Budget utilization is low ({allocated_cost / budget * 100:.0f}%); "
                f"consider extending scope"
            )
        if budget > 0 and cumulative > budget:
            warnings.append(
                f"Total roadmap cost ({cumulative:.0f} USD) exceeds budget "
                f"({budget:.0f} USD); {len(self._roi_analysis) - within_budget_count} "
                f"upgrades deprioritized"
            )

        outputs["total_roi_items"] = len(self._roi_analysis)
        outputs["within_budget_count"] = within_budget_count
        outputs["total_allocated_usd"] = round(allocated_cost, 2)
        outputs["budget_utilization_pct"] = round(
            (allocated_cost / budget * 100.0) if budget > 0 else 0.0, 1
        )
        outputs["top_3_roi"] = [
            {
                "category": r.category_name,
                "roi_score": r.roi_score,
                "cost_usd": r.upgrade_cost_usd,
                "unc_reduction_tco2e": r.uncertainty_reduction_tco2e,
            }
            for r in self._roi_analysis[:3]
        ]
        outputs["avg_cost_per_tco2e_precision"] = round(
            sum(r.cost_per_tco2e_precision for r in self._roi_analysis if r.cost_per_tco2e_precision > 0)
            / max(sum(1 for r in self._roi_analysis if r.cost_per_tco2e_precision > 0), 1),
            2,
        )

        self._state.phase_statuses["roi_prioritization"] = "completed"
        self._state.current_phase = 4

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ROIPrioritization: items=%d within_budget=%d "
            "allocated=%.0f/%.0f",
            len(self._roi_analysis),
            within_budget_count,
            allocated_cost,
            budget,
        )
        return PhaseResult(
            phase_name="roi_prioritization",
            phase_number=4,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _tier_to_maturity(self, avg_tier: float) -> MaturityLevel:
        """Convert average tier numeric to maturity level."""
        for level, (low, high) in MATURITY_THRESHOLDS.items():
            if low <= avg_tier < high:
                return MaturityLevel(level)
        return MaturityLevel.LEVEL_1_INITIAL

    def _maturity_to_min_tier(self, maturity: MaturityLevel) -> float:
        """Get minimum average tier for a maturity level."""
        thresholds = MATURITY_THRESHOLDS.get(maturity.value, (2.5, 3.5))
        return thresholds[0]

    def _determine_target_tier(
        self, cs: CategoryMaturityState, target_avg_tier: float
    ) -> MethodologyTier:
        """Determine target tier for a category based on its emission share."""
        # Higher emission-share categories need higher tiers
        if cs.pct_of_total >= 10.0:
            return MethodologyTier.SUPPLIER_SPECIFIC
        elif cs.pct_of_total >= 5.0:
            return MethodologyTier.HYBRID
        elif cs.pct_of_total >= 1.0:
            return MethodologyTier.AVERAGE_DATA
        else:
            return MethodologyTier.SPEND_BASED

    def _assess_gap_severity(self, tier_gap: int, pct_of_total: float) -> GapSeverity:
        """Assess severity of a maturity gap."""
        score = tier_gap * (1 + pct_of_total / 10.0)
        if score >= 8.0:
            return GapSeverity.CRITICAL
        elif score >= 5.0:
            return GapSeverity.HIGH
        elif score >= 2.0:
            return GapSeverity.MEDIUM
        elif score > 0:
            return GapSeverity.LOW
        return GapSeverity.NONE

    def _format_gap_description(
        self,
        cs: CategoryMaturityState,
        target_tier: MethodologyTier,
        tier_gap: int,
    ) -> str:
        """Format human-readable gap description."""
        return (
            f"{cs.category_name}: upgrade from {cs.current_tier.value} to "
            f"{target_tier.value} ({tier_gap} tier{'s' if tier_gap != 1 else ''} gap). "
            f"Current DQR={cs.dqr_score:.1f}, emissions={cs.emissions_tco2e:.0f} tCO2e "
            f"({cs.pct_of_total:.1f}% of total)."
        )

    def _generate_upgrade_steps(self, gap: MaturityGap) -> List[UpgradeStep]:
        """Generate step-by-step upgrade path for a gap."""
        steps: List[UpgradeStep] = []
        tier_order = [
            MethodologyTier.SPEND_BASED,
            MethodologyTier.AVERAGE_DATA,
            MethodologyTier.HYBRID,
            MethodologyTier.SUPPLIER_SPECIFIC,
            MethodologyTier.VERIFIED,
        ]

        current_idx = next(
            (i for i, t in enumerate(tier_order) if t == gap.current_tier), 0
        )
        target_idx = next(
            (i for i, t in enumerate(tier_order) if t == gap.target_tier), 0
        )

        for i in range(current_idx, target_idx):
            from_tier = tier_order[i]
            to_tier = tier_order[i + 1]
            upgrade_key = f"{from_tier.value}_to_{to_tier.value}"

            cost = UPGRADE_COST_ESTIMATES_USD.get(upgrade_key, 30_000.0)
            duration = UPGRADE_DURATION_WEEKS.get(upgrade_key, 8.0)

            # Adjust cost by emission share (larger categories cost more)
            category_cs = next(
                (cs for cs in self._category_states if cs.category == gap.category),
                None,
            )
            share_multiplier = 1.0
            if category_cs and category_cs.pct_of_total > 0:
                share_multiplier = max(0.5, min(2.0, category_cs.pct_of_total / 10.0))

            # Uncertainty reduction for this step
            from_unc = TIER_UNCERTAINTY.get(from_tier.value, (50.0, 100.0))
            to_unc = TIER_UNCERTAINTY.get(to_tier.value, (10.0, 25.0))
            unc_reduction = (
                (from_unc[0] + from_unc[1]) / 2.0 - (to_unc[0] + to_unc[1]) / 2.0
            )

            # DQR improvement
            from_dqr = TIER_DQR_RANGES.get(from_tier.value, (1.0, 2.0))
            to_dqr = TIER_DQR_RANGES.get(to_tier.value, (2.0, 3.0))
            dqr_improvement = (to_dqr[0] + to_dqr[1]) / 2.0 - (from_dqr[0] + from_dqr[1]) / 2.0

            effort = self._classify_effort(cost * share_multiplier, duration)

            # Priority rank: lower is higher priority
            priority = self._compute_step_priority(gap, i - current_idx)

            step = UpgradeStep(
                category=gap.category,
                category_name=gap.category_name,
                from_tier=from_tier,
                to_tier=to_tier,
                estimated_cost_usd=round(cost * share_multiplier, 2),
                estimated_duration_weeks=round(duration, 1),
                effort=effort,
                prerequisites=self._get_step_prerequisites(from_tier, to_tier),
                expected_dqr_improvement=round(max(dqr_improvement, 0.0), 2),
                expected_uncertainty_reduction_pct=round(max(unc_reduction, 0.0), 1),
                priority_rank=priority,
            )
            steps.append(step)

        return steps

    def _assign_step_dependencies(self) -> None:
        """Assign dependency IDs between sequential steps per category."""
        by_category: Dict[str, List[UpgradeStep]] = {}
        for step in self._roadmap:
            key = step.category.value
            by_category.setdefault(key, []).append(step)

        for cat_steps in by_category.values():
            for i in range(1, len(cat_steps)):
                cat_steps[i].dependencies.append(cat_steps[i - 1].step_id)

    def _compute_step_priority(self, gap: MaturityGap, step_index: int) -> int:
        """Compute priority rank for a step (lower = higher priority)."""
        severity_weight = {
            GapSeverity.CRITICAL: 1,
            GapSeverity.HIGH: 2,
            GapSeverity.MEDIUM: 3,
            GapSeverity.LOW: 4,
            GapSeverity.NONE: 5,
        }
        base = severity_weight.get(gap.severity, 3)
        return base * 10 + step_index

    def _classify_effort(self, cost: float, weeks: float) -> UpgradeEffort:
        """Classify effort level from cost and duration."""
        if cost > 80_000 or weeks > 10:
            return UpgradeEffort.VERY_HIGH
        elif cost > 40_000 or weeks > 6:
            return UpgradeEffort.HIGH
        elif cost > 15_000 or weeks > 3:
            return UpgradeEffort.MEDIUM
        return UpgradeEffort.LOW

    def _get_step_prerequisites(
        self, from_tier: MethodologyTier, to_tier: MethodologyTier
    ) -> List[str]:
        """Get human-readable prerequisites for a tier upgrade."""
        prereqs: Dict[str, List[str]] = {
            "spend_based_to_average_data": [
                "Identify industry-average emission factor databases",
                "Classify activity data by emission source type",
            ],
            "average_data_to_hybrid": [
                "Engage top suppliers for primary data collection",
                "Establish allocation methodology for mixed data",
            ],
            "hybrid_to_supplier_specific": [
                "Achieve >80% supplier response rate for primary data",
                "Implement supplier data verification process",
                "Establish product-level carbon accounting",
            ],
            "supplier_specific_to_verified": [
                "Select third-party verification body",
                "Prepare verification-ready documentation",
                "Implement ISO 14064-3 compliant QA/QC",
            ],
        }
        key = f"{from_tier.value}_to_{to_tier.value}"
        return prereqs.get(key, ["Review category-specific guidance"])

    def _compute_weighted_uncertainty(
        self, categories: List[CategoryMaturityState]
    ) -> float:
        """Compute emission-weighted average uncertainty."""
        total_e = sum(cs.emissions_tco2e for cs in categories)
        if total_e <= 0:
            return 100.0
        return sum(
            cs.uncertainty_pct * cs.emissions_tco2e for cs in categories
        ) / total_e

    def _compute_roadmap_duration(self) -> float:
        """Compute total roadmap duration accounting for parallelism."""
        if not self._roadmap:
            return 0.0
        # Group by category and sum sequential steps within category
        by_category: Dict[str, float] = {}
        for step in self._roadmap:
            key = step.category.value
            by_category[key] = by_category.get(key, 0.0) + step.estimated_duration_weeks
        # Categories can run in parallel, so duration is the max
        return max(by_category.values()) if by_category else 0.0

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._category_states = []
        self._gaps = []
        self._roadmap = []
        self._roi_analysis = []
        self._phase_results = []
        self._current_maturity = MaturityLevel.LEVEL_1_INITIAL
        self._avg_tier = 1.0
        self._avg_dqr = 1.0
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )

    def _update_progress(self, pct: float) -> None:
        """Update progress percentage in state."""
        self._state.progress_pct = min(pct, 100.0)
        self._state.updated_at = datetime.utcnow().isoformat()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: MaturityAssessmentOutput) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += f"|{result.workflow_id}|{result.current_maturity.value}"
        chain += f"|{result.current_avg_tier}|{result.total_upgrade_cost_usd}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
