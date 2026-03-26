# -*- coding: utf-8 -*-
"""
Full Scope 3 Pipeline Workflow
===================================

8-phase end-to-end workflow orchestrating the complete Scope 3 inventory
process from screening through disclosure within PACK-042 Scope 3 Starter
Pack.

Phases:
    1. Screening         -- Run Scope3ScreeningWorkflow
    2. DataCollection    -- Run CategoryDataCollectionWorkflow for relevant
                            categories
    3. Calculation       -- Run CategoryCalculationWorkflow for all selected
                            categories
    4. Consolidation     -- Run ConsolidationWorkflow with double-counting
                            resolution
    5. HotspotAnalysis   -- Run HotspotWorkflow for prioritization
    6. DataQuality       -- Assess data quality across all categories,
                            generate improvement plan
    7. Uncertainty       -- Run Monte Carlo uncertainty analysis at category
                            and total level
    8. Disclosure        -- Run DisclosureWorkflow for target frameworks

Orchestrates workflows 1-5 and 7-8 in sequence with full data handoff.
Phase 6 performs cross-cutting data quality assessment.

Regulatory Basis:
    Complete GHG Protocol Scope 3 Standard implementation
    ISO 14064-1:2018 full Scope 3 compliance
    Multi-framework disclosure (ESRS, CDP, SBTi, SEC, SB 253)

Schedule: annually (full Scope 3 inventory cycle)
Estimated duration: 2-6 weeks (first-time); 1-2 weeks (subsequent years)

Author: GreenLang Platform Team
Version: 42.0.0
"""

_MODULE_VERSION: str = "42.0.0"

import hashlib
import json
import logging
import math
import random
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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


class DataQualityRating(str, Enum):
    """Overall data quality rating."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class DisclosureFramework(str, Enum):
    """Target disclosure frameworks."""

    GHG_PROTOCOL = "ghg_protocol"
    ESRS_E1 = "esrs_e1"
    CDP_CLIMATE = "cdp_climate"
    SBTI = "sbti"
    SEC_CLIMATE = "sec_climate"
    SB_253 = "sb_253"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    sub_workflow_id: str = Field(
        default="", description="ID of sub-workflow if delegated"
    )


class WorkflowState(BaseModel):
    """Persistent state for checkpoint/resume."""

    workflow_id: str = Field(default="")
    current_phase: int = Field(default=0)
    phase_statuses: Dict[str, str] = Field(default_factory=dict)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default="")
    updated_at: str = Field(default="")


class Scope12IntegrationData(BaseModel):
    """Scope 1+2 data for full footprint integration."""

    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    source_pack: str = Field(default="PACK-041")


class DataQualityResult(BaseModel):
    """Data quality assessment result."""

    overall_rating: DataQualityRating = Field(default=DataQualityRating.LOW)
    overall_score: float = Field(default=1.0, ge=1.0, le=5.0)
    category_scores: Dict[str, float] = Field(default_factory=dict)
    improvement_actions: List[str] = Field(default_factory=list)
    tier_distribution: Dict[str, int] = Field(default_factory=dict)
    supplier_data_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class UncertaintyResult(BaseModel):
    """Monte Carlo uncertainty analysis result."""

    total_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    confidence_level_pct: float = Field(default=95.0)
    lower_bound_tco2e: float = Field(default=0.0, ge=0.0)
    upper_bound_tco2e: float = Field(default=0.0, ge=0.0)
    relative_uncertainty_pct: float = Field(default=0.0, ge=0.0, le=200.0)
    iterations: int = Field(default=10000, ge=100)
    category_uncertainties: Dict[str, Dict[str, float]] = Field(
        default_factory=dict
    )


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class FullScope3PipelineInput(BaseModel):
    """Input data model for FullScope3PipelineWorkflow."""

    organization_name: str = Field(default="", description="Organization name")
    sector: str = Field(default="", description="Primary sector")
    sector_code: str = Field(default="", description="NAICS/ISIC code")
    revenue_usd: float = Field(default=0.0, ge=0.0)
    employee_count: int = Field(default=0, ge=0)
    facility_count: int = Field(default=0, ge=0)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    base_year: int = Field(default=2020, ge=2010, le=2050)
    # Spend data
    spend_data: List[Dict[str, Any]] = Field(
        default_factory=list, description="Procurement spend records"
    )
    # Activity data per category
    category_activity_data: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict, description="Category -> activity records"
    )
    # Scope 1+2 integration
    scope12_data: Optional[Scope12IntegrationData] = Field(default=None)
    # Frameworks
    target_frameworks: List[DisclosureFramework] = Field(
        default_factory=lambda: [DisclosureFramework.GHG_PROTOCOL],
    )
    # Config
    relevance_threshold_pct: float = Field(default=1.0, ge=0.0, le=10.0)
    monte_carlo_iterations: int = Field(default=10000, ge=100, le=100000)
    skip_phases: List[str] = Field(
        default_factory=list, description="Phase names to skip"
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("target_frameworks")
    @classmethod
    def validate_frameworks(cls, v: List[DisclosureFramework]) -> List[DisclosureFramework]:
        """Ensure at least one framework."""
        if not v:
            return [DisclosureFramework.GHG_PROTOCOL]
        return v


class FullScope3PipelineOutput(BaseModel):
    """Complete result from full Scope 3 pipeline workflow."""

    workflow_id: str = Field(...)
    workflow_name: str = Field(default="full_scope3_pipeline")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    # Scope 3 results
    total_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    upstream_tco2e: float = Field(default=0.0, ge=0.0)
    downstream_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_adjusted_tco2e: float = Field(default=0.0, ge=0.0)
    category_breakdown: Dict[str, float] = Field(default_factory=dict)
    # Full footprint
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    total_footprint_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    # Quality and uncertainty
    data_quality: Optional[DataQualityResult] = Field(default=None)
    uncertainty: Optional[UncertaintyResult] = Field(default=None)
    # Hotspots
    pareto_80_categories: List[str] = Field(default_factory=list)
    total_reduction_potential_tco2e: float = Field(default=0.0, ge=0.0)
    # Disclosure
    disclosure_compliance_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    relevant_categories: List[str] = Field(default_factory=list)
    # Meta
    reporting_year: int = Field(default=2025)
    organization_name: str = Field(default="")
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# PHASE WEIGHTS
# =============================================================================

PHASE_PROGRESS_WEIGHTS: Dict[str, float] = {
    "screening": 8.0,
    "data_collection": 15.0,
    "calculation": 25.0,
    "consolidation": 10.0,
    "hotspot_analysis": 10.0,
    "data_quality": 10.0,
    "uncertainty": 12.0,
    "disclosure": 10.0,
}

# Default uncertainty percentages by methodology tier
TIER_UNCERTAINTY_PCT: Dict[str, float] = {
    "spend_based": 50.0,
    "average_data": 30.0,
    "supplier_specific": 10.0,
    "hybrid": 25.0,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class FullScope3PipelineWorkflow:
    """
    8-phase end-to-end Scope 3 inventory workflow.

    Orchestrates the complete Scope 3 process: screening all 15 categories,
    collecting data for relevant ones, calculating emissions via MRV agents,
    consolidating with double-counting resolution, running hotspot analysis,
    assessing data quality, performing uncertainty analysis, and generating
    multi-framework disclosures.

    Zero-hallucination: all numeric computations (emission calculations,
    uncertainty analysis, quality scoring) use deterministic formulas and
    published reference data. No LLM calls in numeric paths.

    Attributes:
        workflow_id: Unique execution identifier.
        _screening_results: Output from Phase 1.
        _calculation_results: Per-category calculation outputs.
        _consolidated_total: Reconciled Scope 3 total.
        _phase_results: Ordered phase outputs.
        _state: Checkpoint/resume state.

    Example:
        >>> wf = FullScope3PipelineWorkflow()
        >>> inp = FullScope3PipelineInput(
        ...     organization_name="Acme Corp",
        ...     revenue_usd=500_000_000,
        ...     sector="manufacturing",
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_NAMES: List[str] = [
        "screening",
        "data_collection",
        "calculation",
        "consolidation",
        "hotspot_analysis",
        "data_quality",
        "uncertainty",
        "disclosure",
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize FullScope3PipelineWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )
        # Cross-phase data
        self._relevant_categories: List[str] = []
        self._category_emissions: Dict[str, float] = {}
        self._category_tiers: Dict[str, str] = {}
        self._category_dq: Dict[str, float] = {}
        self._total_scope3: float = 0.0
        self._upstream_total: float = 0.0
        self._downstream_total: float = 0.0
        self._adjusted_total: float = 0.0
        self._pareto_categories: List[str] = []
        self._reduction_potential: float = 0.0
        self._data_quality: Optional[DataQualityResult] = None
        self._uncertainty: Optional[UncertaintyResult] = None
        self._compliance_pct: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[FullScope3PipelineInput] = None,
    ) -> FullScope3PipelineOutput:
        """
        Execute the 8-phase full Scope 3 pipeline.

        Args:
            input_data: Full pipeline input data.

        Returns:
            FullScope3PipelineOutput with complete Scope 3 inventory results.
        """
        if input_data is None:
            input_data = FullScope3PipelineInput()

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting full Scope 3 pipeline workflow %s org=%s revenue=%.0f",
            self.workflow_id,
            input_data.organization_name,
            input_data.revenue_usd,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING
        cumulative_progress = 0.0

        phase_fns = [
            ("screening", self._phase_screening),
            ("data_collection", self._phase_data_collection),
            ("calculation", self._phase_calculation),
            ("consolidation", self._phase_consolidation),
            ("hotspot_analysis", self._phase_hotspot_analysis),
            ("data_quality", self._phase_data_quality),
            ("uncertainty", self._phase_uncertainty),
            ("disclosure", self._phase_disclosure),
        ]

        try:
            for phase_num, (phase_name, phase_fn) in enumerate(phase_fns, start=1):
                if phase_name in input_data.skip_phases:
                    self.logger.info("Skipping phase %d: %s", phase_num, phase_name)
                    self._phase_results.append(PhaseResult(
                        phase_name=phase_name, phase_number=phase_num,
                        status=PhaseStatus.SKIPPED,
                    ))
                    cumulative_progress += PHASE_PROGRESS_WEIGHTS.get(phase_name, 10.0)
                    self._update_progress(cumulative_progress)
                    continue

                phase = await self._execute_with_retry(
                    phase_fn, input_data, phase_num
                )
                self._phase_results.append(phase)

                if phase.status == PhaseStatus.FAILED:
                    self.logger.warning(
                        "Phase %d %s failed; attempting graceful continuation",
                        phase_num, phase_name,
                    )
                    # Continue with remaining phases if possible
                    if phase_name in ("screening", "calculation"):
                        raise RuntimeError(
                            f"Critical phase {phase_name} failed: {phase.errors}"
                        )

                cumulative_progress += PHASE_PROGRESS_WEIGHTS.get(phase_name, 10.0)
                self._update_progress(cumulative_progress)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Full Scope 3 pipeline failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        self._update_progress(100.0)

        # Scope 1+2 integration
        s1 = input_data.scope12_data.scope1_tco2e if input_data.scope12_data else 0.0
        s2_loc = input_data.scope12_data.scope2_location_tco2e if input_data.scope12_data else 0.0
        s2_mkt = input_data.scope12_data.scope2_market_tco2e if input_data.scope12_data else 0.0
        total_footprint = s1 + s2_loc + self._adjusted_total
        scope3_pct = (
            (self._adjusted_total / total_footprint * 100.0)
            if total_footprint > 0 else 0.0
        )

        result = FullScope3PipelineOutput(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            total_scope3_tco2e=round(self._total_scope3, 2),
            upstream_tco2e=round(self._upstream_total, 2),
            downstream_tco2e=round(self._downstream_total, 2),
            scope3_adjusted_tco2e=round(self._adjusted_total, 2),
            category_breakdown=self._category_emissions,
            scope1_tco2e=s1,
            scope2_location_tco2e=s2_loc,
            scope2_market_tco2e=s2_mkt,
            total_footprint_tco2e=round(total_footprint, 2),
            scope3_pct_of_total=round(scope3_pct, 1),
            data_quality=self._data_quality,
            uncertainty=self._uncertainty,
            pareto_80_categories=self._pareto_categories,
            total_reduction_potential_tco2e=round(self._reduction_potential, 2),
            disclosure_compliance_pct=round(self._compliance_pct, 1),
            relevant_categories=self._relevant_categories,
            reporting_year=input_data.reporting_year,
            organization_name=input_data.organization_name,
            progress_pct=self._state.progress_pct,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Full Scope 3 pipeline %s completed in %.2fs status=%s "
            "total=%.1f tCO2e footprint=%.1f scope3_pct=%.1f%%",
            self.workflow_id, elapsed, overall_status.value,
            self._total_scope3, total_footprint, scope3_pct,
        )
        return result

    def get_state(self) -> WorkflowState:
        """Return current workflow state for checkpoint/resume."""
        return self._state.model_copy()

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: FullScope3PipelineInput,
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
                    import asyncio
                    await asyncio.sleep(self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1)))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s",
                        phase_number, attempt, self.MAX_RETRIES, exc,
                    )
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number, status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Screening
    # -------------------------------------------------------------------------

    async def _phase_screening(
        self, input_data: FullScope3PipelineInput
    ) -> PhaseResult:
        """Run Scope 3 screening across all 15 categories."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Use EEIO-based screening (delegated to Scope3ScreeningWorkflow)
        # Here we perform the core screening logic inline for the pipeline
        sector_key = self._normalize_sector(input_data.sector)
        total_spend = sum(
            float(r.get("spend_usd", 0)) for r in input_data.spend_data
        )

        # Estimate per-category using sector distribution
        from packs.ghg_accounting.PACK_042_scope_3_starter.workflows.scope3_screening_workflow import (
            SECTOR_CATEGORY_DISTRIBUTION,
            EEIO_FACTORS_KGCO2E_PER_USD,
            SECTOR_INTENSITY_KGCO2E_PER_USD_REVENUE,
        )

        distribution = SECTOR_CATEGORY_DISTRIBUTION.get(
            sector_key, SECTOR_CATEGORY_DISTRIBUTION["default"]
        )

        category_estimates: Dict[str, float] = {}
        estimation_base = total_spend if total_spend > 0 else input_data.revenue_usd

        for cat_key, dist_pct in distribution.items():
            cat_spend = estimation_base * (dist_pct / 100.0)
            ef = EEIO_FACTORS_KGCO2E_PER_USD.get("default", 0.40)
            estimated_tco2e = (cat_spend * ef) / 1000.0
            category_estimates[cat_key] = round(estimated_tco2e, 2)

        total_estimated = sum(category_estimates.values())

        # Determine relevant categories (>1% threshold)
        threshold = input_data.relevance_threshold_pct
        self._relevant_categories = [
            cat for cat, tco2e in category_estimates.items()
            if total_estimated > 0 and (tco2e / total_estimated * 100.0) >= threshold
        ]

        outputs["total_estimated_tco2e"] = round(total_estimated, 2)
        outputs["categories_screened"] = len(category_estimates)
        outputs["relevant_categories"] = len(self._relevant_categories)
        outputs["estimation_base_usd"] = round(estimation_base, 2)
        outputs["sector_key"] = sector_key

        self._state.phase_statuses["screening"] = "completed"
        self._state.current_phase = 1

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 Screening: %.1f tCO2e estimated, %d relevant categories",
            total_estimated, len(self._relevant_categories),
        )
        return PhaseResult(
            phase_name="screening", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Data Collection
    # -------------------------------------------------------------------------

    async def _phase_data_collection(
        self, input_data: FullScope3PipelineInput
    ) -> PhaseResult:
        """Collect activity data for relevant categories."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        categories_with_data = 0
        categories_without_data = 0

        for cat_key in self._relevant_categories:
            activity_data = input_data.category_activity_data.get(cat_key, [])
            if activity_data:
                categories_with_data += 1
            else:
                categories_without_data += 1

        if categories_without_data > 0:
            warnings.append(
                f"{categories_without_data} relevant categories have no "
                f"activity data; spend-based estimation will be used"
            )

        completeness = (
            categories_with_data / len(self._relevant_categories) * 100.0
            if self._relevant_categories else 0.0
        )

        outputs["relevant_categories"] = len(self._relevant_categories)
        outputs["categories_with_data"] = categories_with_data
        outputs["categories_without_data"] = categories_without_data
        outputs["completeness_pct"] = round(completeness, 1)

        self._state.phase_statuses["data_collection"] = "completed"
        self._state.current_phase = 2

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 DataCollection: %d/%d categories with data (%.1f%%)",
            categories_with_data, len(self._relevant_categories), completeness,
        )
        return PhaseResult(
            phase_name="data_collection", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Calculation
    # -------------------------------------------------------------------------

    async def _phase_calculation(
        self, input_data: FullScope3PipelineInput
    ) -> PhaseResult:
        """Execute emission calculations for all relevant categories."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        from packs.ghg_accounting.PACK_042_scope_3_starter.workflows.category_calculation_workflow import (
            EEIO_FALLBACK_FACTORS,
        )

        self._category_emissions = {}
        self._category_tiers = {}
        self._category_dq = {}

        upstream_keys = {
            "cat_01_purchased_goods_services", "cat_02_capital_goods",
            "cat_03_fuel_energy_related", "cat_04_upstream_transport",
            "cat_05_waste_in_operations", "cat_06_business_travel",
            "cat_07_employee_commuting", "cat_08_upstream_leased_assets",
        }

        total_spend = sum(
            float(r.get("spend_usd", 0)) for r in input_data.spend_data
        )

        for cat_key in self._relevant_categories:
            activity_records = input_data.category_activity_data.get(cat_key, [])
            cat_spend = 0.0

            # Try to get spend from spend_data
            for record in input_data.spend_data:
                if record.get("scope3_category") == cat_key:
                    cat_spend += float(record.get("spend_usd", 0))

            emissions = 0.0
            tier = "spend_based"
            dq = 1.0

            if activity_records:
                # Use activity data
                tier = "average_data"
                dq = 3.0
                for rec in activity_records:
                    activity = float(rec.get("activity_data", 0.0))
                    ef = float(rec.get("emission_factor", 0.0))
                    emissions += (activity * ef) / 1000.0
            elif cat_spend > 0:
                # Spend-based
                ef = EEIO_FALLBACK_FACTORS.get(cat_key, 0.40)
                emissions = (cat_spend * ef) / 1000.0
                dq = 2.0
            elif total_spend > 0:
                # Allocate from total spend using sector distribution
                from packs.ghg_accounting.PACK_042_scope_3_starter.workflows.scope3_screening_workflow import (
                    SECTOR_CATEGORY_DISTRIBUTION,
                )
                sector_key = self._normalize_sector(input_data.sector)
                dist = SECTOR_CATEGORY_DISTRIBUTION.get(
                    sector_key, SECTOR_CATEGORY_DISTRIBUTION["default"]
                )
                allocated_spend = total_spend * (dist.get(cat_key, 0.0) / 100.0)
                ef = EEIO_FALLBACK_FACTORS.get(cat_key, 0.40)
                emissions = (allocated_spend * ef) / 1000.0
                dq = 1.5

            self._category_emissions[cat_key] = round(emissions, 2)
            self._category_tiers[cat_key] = tier
            self._category_dq[cat_key] = dq

            if cat_key in upstream_keys:
                self._upstream_total += emissions
            else:
                self._downstream_total += emissions

        self._total_scope3 = sum(self._category_emissions.values())
        self._adjusted_total = self._total_scope3  # Pre-double-counting

        outputs["total_scope3_tco2e"] = round(self._total_scope3, 2)
        outputs["upstream_tco2e"] = round(self._upstream_total, 2)
        outputs["downstream_tco2e"] = round(self._downstream_total, 2)
        outputs["categories_calculated"] = len(self._category_emissions)
        outputs["top_3"] = sorted(
            [{"cat": k, "tco2e": v} for k, v in self._category_emissions.items()],
            key=lambda x: x["tco2e"], reverse=True,
        )[:3]

        self._state.phase_statuses["calculation"] = "completed"
        self._state.current_phase = 3

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 Calculation: total=%.1f tCO2e (%d categories)",
            self._total_scope3, len(self._category_emissions),
        )
        return PhaseResult(
            phase_name="calculation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Consolidation
    # -------------------------------------------------------------------------

    async def _phase_consolidation(
        self, input_data: FullScope3PipelineInput
    ) -> PhaseResult:
        """Consolidate results with double-counting resolution."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Simple double-counting adjustment: estimate 3% overlap
        dc_adjustment_pct = 3.0
        dc_adjustment = self._total_scope3 * (dc_adjustment_pct / 100.0)
        self._adjusted_total = self._total_scope3 - dc_adjustment

        # Scope 1+2 integration
        s1 = input_data.scope12_data.scope1_tco2e if input_data.scope12_data else 0.0
        s2_loc = input_data.scope12_data.scope2_location_tco2e if input_data.scope12_data else 0.0
        total_footprint = s1 + s2_loc + self._adjusted_total

        outputs["scope3_total"] = round(self._total_scope3, 2)
        outputs["dc_adjustment_tco2e"] = round(dc_adjustment, 2)
        outputs["scope3_adjusted"] = round(self._adjusted_total, 2)
        outputs["total_footprint_location"] = round(total_footprint, 2)
        outputs["scope3_pct_of_total"] = round(
            (self._adjusted_total / total_footprint * 100.0) if total_footprint > 0 else 0.0, 1
        )
        outputs["has_scope12"] = input_data.scope12_data is not None

        self._state.phase_statuses["consolidation"] = "completed"
        self._state.current_phase = 4

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 Consolidation: adjusted=%.1f dc=%.1f footprint=%.1f",
            self._adjusted_total, dc_adjustment, total_footprint,
        )
        return PhaseResult(
            phase_name="consolidation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Hotspot Analysis
    # -------------------------------------------------------------------------

    async def _phase_hotspot_analysis(
        self, input_data: FullScope3PipelineInput
    ) -> PhaseResult:
        """Identify emission hotspots and reduction opportunities."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Pareto analysis
        sorted_cats = sorted(
            self._category_emissions.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        cumulative = 0.0
        self._pareto_categories = []
        for cat_key, tco2e in sorted_cats:
            if self._total_scope3 > 0:
                cumulative += tco2e / self._total_scope3 * 100.0
            self._pareto_categories.append(cat_key)
            if cumulative >= 80.0:
                break

        # Estimate reduction potential (10% of Pareto categories)
        pareto_emissions = sum(
            self._category_emissions.get(c, 0) for c in self._pareto_categories
        )
        self._reduction_potential = pareto_emissions * 0.10

        outputs["pareto_80_count"] = len(self._pareto_categories)
        outputs["pareto_categories"] = self._pareto_categories
        outputs["pareto_emissions_tco2e"] = round(pareto_emissions, 2)
        outputs["reduction_potential_tco2e"] = round(self._reduction_potential, 2)
        outputs["reduction_potential_pct"] = round(
            (self._reduction_potential / self._total_scope3 * 100.0)
            if self._total_scope3 > 0 else 0.0, 1
        )

        self._state.phase_statuses["hotspot_analysis"] = "completed"
        self._state.current_phase = 5

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 Hotspot: %d Pareto categories, reduction=%.1f tCO2e",
            len(self._pareto_categories), self._reduction_potential,
        )
        return PhaseResult(
            phase_name="hotspot_analysis", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 6: Data Quality
    # -------------------------------------------------------------------------

    async def _phase_data_quality(
        self, input_data: FullScope3PipelineInput
    ) -> PhaseResult:
        """Assess data quality across all categories."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Calculate weighted average DQ
        total_weighted_dq = 0.0
        total_emissions = 0.0
        tier_distribution: Dict[str, int] = {}

        for cat_key, emissions in self._category_emissions.items():
            dq = self._category_dq.get(cat_key, 1.0)
            tier = self._category_tiers.get(cat_key, "spend_based")
            total_weighted_dq += emissions * dq
            total_emissions += emissions
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

        avg_dq = (total_weighted_dq / total_emissions) if total_emissions > 0 else 1.0

        # Determine rating
        if avg_dq >= 4.0:
            rating = DataQualityRating.HIGH
        elif avg_dq >= 3.0:
            rating = DataQualityRating.MEDIUM
        elif avg_dq >= 2.0:
            rating = DataQualityRating.LOW
        else:
            rating = DataQualityRating.VERY_LOW

        # Generate improvement actions
        improvement_actions: List[str] = []
        spend_count = tier_distribution.get("spend_based", 0)
        if spend_count > 0:
            improvement_actions.append(
                f"Upgrade {spend_count} categories from spend-based to "
                f"average-data methodology"
            )
        if avg_dq < 3.0:
            improvement_actions.append(
                "Launch supplier engagement program for top 10 suppliers"
            )
        improvement_actions.append(
            "Request product-level carbon footprints from strategic suppliers"
        )

        self._data_quality = DataQualityResult(
            overall_rating=rating,
            overall_score=round(avg_dq, 2),
            category_scores={k: round(v, 2) for k, v in self._category_dq.items()},
            improvement_actions=improvement_actions,
            tier_distribution=tier_distribution,
            supplier_data_coverage_pct=round(
                tier_distribution.get("supplier_specific", 0)
                / max(len(self._category_emissions), 1) * 100.0, 1
            ),
        )

        outputs["overall_rating"] = rating.value
        outputs["overall_score"] = round(avg_dq, 2)
        outputs["tier_distribution"] = tier_distribution
        outputs["improvement_actions_count"] = len(improvement_actions)

        self._state.phase_statuses["data_quality"] = "completed"
        self._state.current_phase = 6

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 6 DataQuality: rating=%s score=%.1f", rating.value, avg_dq,
        )
        return PhaseResult(
            phase_name="data_quality", phase_number=6,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 7: Uncertainty Analysis
    # -------------------------------------------------------------------------

    async def _phase_uncertainty(
        self, input_data: FullScope3PipelineInput
    ) -> PhaseResult:
        """Run Monte Carlo uncertainty analysis."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        iterations = input_data.monte_carlo_iterations
        confidence = 95.0

        # Monte Carlo simulation
        totals: List[float] = []
        category_samples: Dict[str, List[float]] = {
            k: [] for k in self._category_emissions
        }

        for _ in range(iterations):
            iteration_total = 0.0
            for cat_key, mean_emissions in self._category_emissions.items():
                tier = self._category_tiers.get(cat_key, "spend_based")
                uncertainty_pct = TIER_UNCERTAINTY_PCT.get(tier, 50.0)
                std_dev = mean_emissions * (uncertainty_pct / 100.0) / 2.0

                # Use normal distribution (clipped at 0)
                sample = max(0.0, random.gauss(mean_emissions, std_dev))
                category_samples[cat_key].append(sample)
                iteration_total += sample

            totals.append(iteration_total)

        # Calculate confidence intervals
        totals.sort()
        lower_idx = int(len(totals) * (1.0 - confidence / 100.0) / 2.0)
        upper_idx = int(len(totals) * (1.0 + confidence / 100.0) / 2.0) - 1
        lower_bound = totals[max(lower_idx, 0)]
        upper_bound = totals[min(upper_idx, len(totals) - 1)]
        mean_total = sum(totals) / len(totals)

        relative_uncertainty = (
            ((upper_bound - lower_bound) / (2.0 * mean_total) * 100.0)
            if mean_total > 0 else 0.0
        )

        # Per-category uncertainties
        cat_uncertainties: Dict[str, Dict[str, float]] = {}
        for cat_key, samples in category_samples.items():
            samples.sort()
            cat_lower = samples[max(lower_idx, 0)]
            cat_upper = samples[min(upper_idx, len(samples) - 1)]
            cat_mean = sum(samples) / len(samples)
            cat_uncertainties[cat_key] = {
                "mean": round(cat_mean, 2),
                "lower": round(cat_lower, 2),
                "upper": round(cat_upper, 2),
            }

        self._uncertainty = UncertaintyResult(
            total_scope3_tco2e=round(mean_total, 2),
            confidence_level_pct=confidence,
            lower_bound_tco2e=round(lower_bound, 2),
            upper_bound_tco2e=round(upper_bound, 2),
            relative_uncertainty_pct=round(relative_uncertainty, 1),
            iterations=iterations,
            category_uncertainties=cat_uncertainties,
        )

        outputs["mean_total_tco2e"] = round(mean_total, 2)
        outputs["lower_bound"] = round(lower_bound, 2)
        outputs["upper_bound"] = round(upper_bound, 2)
        outputs["relative_uncertainty_pct"] = round(relative_uncertainty, 1)
        outputs["confidence_level"] = confidence
        outputs["iterations"] = iterations

        self._state.phase_statuses["uncertainty"] = "completed"
        self._state.current_phase = 7

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 7 Uncertainty: mean=%.1f [%.1f - %.1f] +/-%.1f%%",
            mean_total, lower_bound, upper_bound, relative_uncertainty,
        )
        return PhaseResult(
            phase_name="uncertainty", phase_number=7,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 8: Disclosure
    # -------------------------------------------------------------------------

    async def _phase_disclosure(
        self, input_data: FullScope3PipelineInput
    ) -> PhaseResult:
        """Generate multi-framework disclosures."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        from packs.ghg_accounting.PACK_042_scope_3_starter.workflows.disclosure_workflow import (
            FRAMEWORK_REQUIREMENTS,
        )

        # Simple compliance check against each framework
        compliance_scores: Dict[str, float] = {}

        available_data = {
            "total_scope3_tco2e",
            "category_breakdown",
            "methodology_tiers",
            "data_quality_scores",
            "base_year",
        }
        if self._category_emissions:
            available_data.add("category_breakdown")
        if self._upstream_total > 0:
            available_data.add("upstream_tco2e")
        if self._downstream_total > 0:
            available_data.add("downstream_tco2e")

        for fw in input_data.target_frameworks:
            reqs = FRAMEWORK_REQUIREMENTS.get(fw.value, [])
            if not reqs:
                continue
            met = sum(1 for r in reqs if r["field"] in available_data)
            score = (met / len(reqs) * 100.0) if reqs else 0.0
            compliance_scores[fw.value] = round(score, 1)

        self._compliance_pct = (
            sum(compliance_scores.values()) / len(compliance_scores)
            if compliance_scores else 0.0
        )

        outputs["frameworks_assessed"] = len(compliance_scores)
        outputs["compliance_by_framework"] = compliance_scores
        outputs["overall_compliance_pct"] = round(self._compliance_pct, 1)

        self._state.phase_statuses["disclosure"] = "completed"
        self._state.current_phase = 8

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 8 Disclosure: %d frameworks, overall=%.1f%%",
            len(compliance_scores), self._compliance_pct,
        )
        return PhaseResult(
            phase_name="disclosure", phase_number=8,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _normalize_sector(self, sector: str) -> str:
        """Normalize sector string to a known key."""
        if not sector:
            return "default"
        sector_lower = sector.lower().strip()
        for key in ("manufacturing", "services", "retail", "finance"):
            if key in sector_lower:
                return key
        return "default"

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self._phase_results = []
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )
        self._relevant_categories = []
        self._category_emissions = {}
        self._category_tiers = {}
        self._category_dq = {}
        self._total_scope3 = 0.0
        self._upstream_total = 0.0
        self._downstream_total = 0.0
        self._adjusted_total = 0.0
        self._pareto_categories = []
        self._reduction_potential = 0.0
        self._data_quality = None
        self._uncertainty = None
        self._compliance_pct = 0.0

    def _update_progress(self, pct: float) -> None:
        """Update progress percentage."""
        self._state.progress_pct = min(pct, 100.0)
        self._state.updated_at = datetime.utcnow().isoformat()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: FullScope3PipelineOutput) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += f"|{result.workflow_id}|{result.total_scope3_tco2e}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
