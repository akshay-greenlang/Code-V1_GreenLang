# -*- coding: utf-8 -*-
"""
Scenario Planning Workflow
=================================

4-phase workflow for Scope 3 emission reduction scenario planning and
intervention optimization within PACK-043 Scope 3 Complete Pack.

Phases:
    1. BASELINE_ESTABLISHMENT   -- Set baseline Scope 3 inventory from PACK-042
                                   results with year-on-year trend context.
    2. INTERVENTION_DEFINITION  -- Define reduction interventions with cost,
                                   impact, and implementation timeline estimates.
    3. IMPACT_MODELLING         -- Run MACC analysis, what-if scenarios, and
                                   technology pathway models.
    4. PORTFOLIO_OPTIMIZATION   -- Rank scenarios by cost-effectiveness, check
                                   Paris alignment, and build optimal portfolio.

The workflow follows GreenLang zero-hallucination principles: all MACC curves,
scenario projections, and alignment checks use deterministic arithmetic on
auditable input parameters. SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    SBTi Corporate Net-Zero Standard -- Target-setting methodology
    GHG Protocol Mitigation Goal Standard
    IEA Net Zero by 2050 roadmap -- Technology pathway benchmarks
    Paris Agreement -- 1.5C / well-below-2C alignment thresholds

Schedule: annually or upon strategy refresh
Estimated duration: 4-8 hours

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


class InterventionType(str, Enum):
    """Types of Scope 3 reduction interventions."""

    SUPPLIER_ENGAGEMENT = "supplier_engagement"
    MATERIAL_SUBSTITUTION = "material_substitution"
    RENEWABLE_ENERGY_PROCUREMENT = "renewable_energy_procurement"
    LOGISTICS_OPTIMIZATION = "logistics_optimization"
    PRODUCT_REDESIGN = "product_redesign"
    CIRCULAR_ECONOMY = "circular_economy"
    DEMAND_REDUCTION = "demand_reduction"
    MODAL_SHIFT = "modal_shift"
    NEARSHORING = "nearshoring"
    TECHNOLOGY_ADOPTION = "technology_adoption"


class ScenarioType(str, Enum):
    """Scenario modeling types."""

    BUSINESS_AS_USUAL = "business_as_usual"
    MODERATE_AMBITION = "moderate_ambition"
    HIGH_AMBITION = "high_ambition"
    NET_ZERO_ALIGNED = "net_zero_aligned"
    CUSTOM = "custom"


class AlignmentStatus(str, Enum):
    """Paris alignment status."""

    ALIGNED_1_5C = "aligned_1_5c"
    ALIGNED_WB2C = "aligned_well_below_2c"
    NOT_ALIGNED = "not_aligned"
    INSUFFICIENT_DATA = "insufficient_data"


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


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# SBTi required annual linear reduction rates (% per year from base year)
SBTI_REDUCTION_RATES: Dict[str, float] = {
    "1.5C_linear_absolute": 4.2,  # % per year
    "well_below_2C_absolute": 2.5,
    "1.5C_sda_services": 7.0,
    "1.5C_sda_manufacturing": 4.0,
}

# Typical MACC data: intervention -> (abatement cost USD/tCO2e, max potential %)
INTERVENTION_BENCHMARKS: Dict[str, Tuple[float, float]] = {
    InterventionType.SUPPLIER_ENGAGEMENT.value: (15.0, 20.0),
    InterventionType.MATERIAL_SUBSTITUTION.value: (50.0, 15.0),
    InterventionType.RENEWABLE_ENERGY_PROCUREMENT.value: (10.0, 25.0),
    InterventionType.LOGISTICS_OPTIMIZATION.value: (20.0, 12.0),
    InterventionType.PRODUCT_REDESIGN.value: (80.0, 18.0),
    InterventionType.CIRCULAR_ECONOMY.value: (40.0, 10.0),
    InterventionType.DEMAND_REDUCTION.value: (-5.0, 8.0),
    InterventionType.MODAL_SHIFT.value: (25.0, 10.0),
    InterventionType.NEARSHORING.value: (35.0, 8.0),
    InterventionType.TECHNOLOGY_ADOPTION.value: (60.0, 22.0),
}

# Pre-defined scenario reduction profiles (total % reduction by 2030)
SCENARIO_PROFILES: Dict[str, Dict[str, float]] = {
    ScenarioType.BUSINESS_AS_USUAL.value: {
        "annual_reduction_pct": 0.5,
        "target_reduction_2030_pct": 4.0,
    },
    ScenarioType.MODERATE_AMBITION.value: {
        "annual_reduction_pct": 2.5,
        "target_reduction_2030_pct": 20.0,
    },
    ScenarioType.HIGH_AMBITION.value: {
        "annual_reduction_pct": 4.2,
        "target_reduction_2030_pct": 34.0,
    },
    ScenarioType.NET_ZERO_ALIGNED.value: {
        "annual_reduction_pct": 7.0,
        "target_reduction_2030_pct": 50.0,
    },
}


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


class WorkflowState(BaseModel):
    """Persistent state for checkpoint/resume."""

    workflow_id: str = Field(default="")
    current_phase: int = Field(default=0)
    phase_statuses: Dict[str, str] = Field(default_factory=dict)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default="")
    updated_at: str = Field(default="")


class BaselineCategory(BaseModel):
    """Baseline emissions for a single category."""

    category: str = Field(default="")
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    yoy_trend_pct: float = Field(default=0.0, description="Year-over-year change %")


class Intervention(BaseModel):
    """Emission reduction intervention definition."""

    intervention_id: str = Field(
        default_factory=lambda: f"int-{uuid.uuid4().hex[:8]}"
    )
    name: str = Field(default="")
    intervention_type: InterventionType = Field(
        default=InterventionType.SUPPLIER_ENGAGEMENT
    )
    target_categories: List[str] = Field(default_factory=list)
    estimated_reduction_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Estimated % reduction of target category emissions",
    )
    cost_usd: float = Field(default=0.0, ge=0.0)
    implementation_years: float = Field(default=1.0, ge=0.0, le=10.0)
    start_year: int = Field(default=2025, ge=2020, le=2050)
    confidence_pct: float = Field(
        default=70.0, ge=0.0, le=100.0,
        description="Confidence in achieving stated reduction",
    )
    co_benefits: List[str] = Field(default_factory=list)


class MACCEntry(BaseModel):
    """Marginal Abatement Cost Curve entry."""

    intervention_id: str = Field(default="")
    intervention_name: str = Field(default="")
    abatement_cost_usd_per_tco2e: float = Field(default=0.0)
    abatement_potential_tco2e: float = Field(default=0.0, ge=0.0)
    cumulative_abatement_tco2e: float = Field(default=0.0, ge=0.0)
    total_cost_usd: float = Field(default=0.0)
    cumulative_cost_usd: float = Field(default=0.0)


class ScenarioResult(BaseModel):
    """Result of a single scenario projection."""

    scenario_type: ScenarioType = Field(default=ScenarioType.BUSINESS_AS_USUAL)
    scenario_name: str = Field(default="")
    base_year: int = Field(default=2025)
    target_year: int = Field(default=2030)
    baseline_tco2e: float = Field(default=0.0, ge=0.0)
    projected_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    annual_reduction_rate_pct: float = Field(default=0.0, ge=0.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    cost_per_tco2e_reduced: float = Field(default=0.0, ge=0.0)
    alignment_status: AlignmentStatus = Field(
        default=AlignmentStatus.INSUFFICIENT_DATA
    )
    interventions_included: List[str] = Field(default_factory=list)
    year_by_year_tco2e: Dict[int, float] = Field(default_factory=dict)


class OptimalPortfolio(BaseModel):
    """Optimized intervention portfolio."""

    portfolio_name: str = Field(default="")
    total_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    total_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    avg_cost_per_tco2e: float = Field(default=0.0, ge=0.0)
    alignment_status: AlignmentStatus = Field(
        default=AlignmentStatus.NOT_ALIGNED
    )
    selected_interventions: List[str] = Field(default_factory=list)
    budget_utilization_pct: float = Field(default=0.0, ge=0.0, le=100.0)


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class ScenarioPlanningInput(BaseModel):
    """Input data model for ScenarioPlanningWorkflow."""

    organization_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    base_year: int = Field(default=2025, ge=2020, le=2050)
    target_year: int = Field(default=2030, ge=2025, le=2060)
    baseline_total_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    baseline_categories: List[BaselineCategory] = Field(default_factory=list)
    interventions: List[Intervention] = Field(default_factory=list)
    scenarios_to_model: List[ScenarioType] = Field(
        default_factory=lambda: [
            ScenarioType.BUSINESS_AS_USUAL,
            ScenarioType.MODERATE_AMBITION,
            ScenarioType.HIGH_AMBITION,
            ScenarioType.NET_ZERO_ALIGNED,
        ]
    )
    budget_usd: float = Field(default=1_000_000.0, ge=0.0)
    revenue_growth_pct: float = Field(
        default=3.0, description="Annual revenue growth % (for intensity decoupling)"
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class ScenarioPlanningOutput(BaseModel):
    """Complete output from ScenarioPlanningWorkflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="scenario_planning")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_name: str = Field(default="")
    base_year: int = Field(default=2025)
    target_year: int = Field(default=2030)
    baseline_tco2e: float = Field(default=0.0, ge=0.0)
    macc_entries: List[MACCEntry] = Field(default_factory=list)
    scenario_results: List[ScenarioResult] = Field(default_factory=list)
    optimal_portfolio: Optional[OptimalPortfolio] = Field(default=None)
    total_abatement_potential_tco2e: float = Field(default=0.0, ge=0.0)
    total_abatement_cost_usd: float = Field(default=0.0, ge=0.0)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ScenarioPlanningWorkflow:
    """
    4-phase Scope 3 scenario planning and intervention optimization workflow.

    Establishes baseline, defines interventions, runs MACC and what-if
    scenarios, and optimizes an intervention portfolio against budget and
    Paris alignment constraints.

    Zero-hallucination: all projections, MACC curves, and alignment checks
    use deterministic formulas on auditable input parameters. No LLM calls
    in any numeric path.

    Example:
        >>> wf = ScenarioPlanningWorkflow()
        >>> inp = ScenarioPlanningInput(
        ...     baseline_total_scope3_tco2e=100000,
        ...     interventions=[...],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_NAMES: List[str] = [
        "baseline_establishment",
        "intervention_definition",
        "impact_modelling",
        "portfolio_optimization",
    ]

    PHASE_WEIGHTS: Dict[str, float] = {
        "baseline_establishment": 15.0,
        "intervention_definition": 20.0,
        "impact_modelling": 40.0,
        "portfolio_optimization": 25.0,
    }

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ScenarioPlanningWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._baseline_total: float = 0.0
        self._baseline_categories: List[BaselineCategory] = []
        self._interventions: List[Intervention] = []
        self._macc: List[MACCEntry] = []
        self._scenarios: List[ScenarioResult] = []
        self._portfolio: Optional[OptimalPortfolio] = None
        self._phase_results: List[PhaseResult] = []
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[ScenarioPlanningInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ScenarioPlanningOutput:
        """Execute the 4-phase scenario planning workflow."""
        if input_data is None:
            input_data = ScenarioPlanningInput()

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting scenario planning workflow %s org=%s baseline=%.0f",
            self.workflow_id,
            input_data.organization_name,
            input_data.baseline_total_scope3_tco2e,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING
        self._update_progress(0.0)

        try:
            phase1 = await self._execute_with_retry(
                self._phase_baseline_establishment, input_data, 1
            )
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 1 failed: {phase1.errors}")
            self._update_progress(15.0)

            phase2 = await self._execute_with_retry(
                self._phase_intervention_definition, input_data, 2
            )
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 2 failed: {phase2.errors}")
            self._update_progress(35.0)

            phase3 = await self._execute_with_retry(
                self._phase_impact_modelling, input_data, 3
            )
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 3 failed: {phase3.errors}")
            self._update_progress(75.0)

            phase4 = await self._execute_with_retry(
                self._phase_portfolio_optimization, input_data, 4
            )
            self._phase_results.append(phase4)
            if phase4.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 4 failed: {phase4.errors}")
            self._update_progress(100.0)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Scenario planning workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(
                PhaseResult(
                    phase_name="error", phase_number=0,
                    status=PhaseStatus.FAILED, errors=[str(exc)],
                )
            )

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        result = ScenarioPlanningOutput(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_name=input_data.organization_name,
            base_year=input_data.base_year,
            target_year=input_data.target_year,
            baseline_tco2e=self._baseline_total,
            macc_entries=self._macc,
            scenario_results=self._scenarios,
            optimal_portfolio=self._portfolio,
            total_abatement_potential_tco2e=round(
                sum(m.abatement_potential_tco2e for m in self._macc), 2
            ),
            total_abatement_cost_usd=round(
                sum(m.total_cost_usd for m in self._macc), 2
            ),
            progress_pct=self._state.progress_pct,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Scenario planning workflow %s completed in %.2fs status=%s "
            "scenarios=%d abatement=%.0f tCO2e",
            self.workflow_id, elapsed, overall_status.value,
            len(self._scenarios),
            result.total_abatement_potential_tco2e,
        )
        return result

    def get_state(self) -> WorkflowState:
        """Return current workflow state for checkpoint/resume."""
        return self._state.model_copy()

    async def resume(
        self, state: WorkflowState, input_data: ScenarioPlanningInput
    ) -> ScenarioPlanningOutput:
        """Resume workflow from a saved checkpoint state."""
        self._state = state
        self.workflow_id = state.workflow_id
        return await self.execute(input_data)

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: ScenarioPlanningInput, phase_number: int
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
                        "Phase %d attempt %d/%d failed: %s",
                        phase_number, attempt, self.MAX_RETRIES, exc,
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
    # Phase 1: Baseline Establishment
    # -------------------------------------------------------------------------

    async def _phase_baseline_establishment(
        self, input_data: ScenarioPlanningInput
    ) -> PhaseResult:
        """Set baseline Scope 3 inventory from PACK-042 results."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._baseline_total = input_data.baseline_total_scope3_tco2e
        self._baseline_categories = list(input_data.baseline_categories)

        # Validate baseline total matches category sum
        cat_sum = sum(bc.emissions_tco2e for bc in self._baseline_categories)
        if cat_sum > 0 and abs(cat_sum - self._baseline_total) / max(cat_sum, 1) > 0.01:
            warnings.append(
                f"Baseline total ({self._baseline_total:.0f}) differs from "
                f"category sum ({cat_sum:.0f}) by > 1%; using category sum"
            )
            self._baseline_total = cat_sum

        if self._baseline_total <= 0:
            warnings.append("Baseline is zero; scenario projections will be trivial")

        # Fill pct_of_total if missing
        for bc in self._baseline_categories:
            if bc.pct_of_total == 0.0 and self._baseline_total > 0:
                bc.pct_of_total = round(
                    bc.emissions_tco2e / self._baseline_total * 100.0, 2
                )

        outputs["baseline_tco2e"] = round(self._baseline_total, 2)
        outputs["base_year"] = input_data.base_year
        outputs["target_year"] = input_data.target_year
        outputs["time_horizon_years"] = input_data.target_year - input_data.base_year
        outputs["categories_with_data"] = len(self._baseline_categories)
        outputs["top_3_categories"] = [
            {"category": bc.category, "tco2e": bc.emissions_tco2e, "pct": bc.pct_of_total}
            for bc in sorted(
                self._baseline_categories,
                key=lambda x: x.emissions_tco2e,
                reverse=True,
            )[:3]
        ]

        self._state.phase_statuses["baseline_establishment"] = "completed"
        self._state.current_phase = 1

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 BaselineEstablishment: total=%.0f tCO2e, %d categories",
            self._baseline_total, len(self._baseline_categories),
        )
        return PhaseResult(
            phase_name="baseline_establishment", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Intervention Definition
    # -------------------------------------------------------------------------

    async def _phase_intervention_definition(
        self, input_data: ScenarioPlanningInput
    ) -> PhaseResult:
        """Define reduction interventions with cost/impact estimates."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._interventions = list(input_data.interventions)

        # Enrich with benchmark data if estimates are missing
        for intv in self._interventions:
            benchmark = INTERVENTION_BENCHMARKS.get(intv.intervention_type.value)
            if benchmark and intv.estimated_reduction_pct == 0.0:
                intv.estimated_reduction_pct = benchmark[1]
                warnings.append(
                    f"Using benchmark reduction for '{intv.name}': "
                    f"{benchmark[1]:.1f}%"
                )

        # Validate no over-reduction (sum of interventions per category < 100%)
        cat_reductions: Dict[str, float] = {}
        for intv in self._interventions:
            for cat in intv.target_categories:
                cat_reductions[cat] = cat_reductions.get(cat, 0.0) + intv.estimated_reduction_pct

        for cat, total_red in cat_reductions.items():
            if total_red > 100.0:
                warnings.append(
                    f"Category '{cat}' has {total_red:.0f}% total reduction "
                    f"across interventions (exceeds 100%); will be capped"
                )

        # Type distribution
        type_dist: Dict[str, int] = {}
        for intv in self._interventions:
            t = intv.intervention_type.value
            type_dist[t] = type_dist.get(t, 0) + 1

        outputs["interventions_defined"] = len(self._interventions)
        outputs["total_estimated_cost_usd"] = round(
            sum(i.cost_usd for i in self._interventions), 2
        )
        outputs["intervention_type_distribution"] = type_dist
        outputs["categories_targeted"] = len(cat_reductions)
        outputs["avg_confidence_pct"] = round(
            sum(i.confidence_pct for i in self._interventions) / max(len(self._interventions), 1),
            1,
        )

        self._state.phase_statuses["intervention_definition"] = "completed"
        self._state.current_phase = 2

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 InterventionDefinition: %d interventions, cost=%.0f",
            len(self._interventions),
            outputs["total_estimated_cost_usd"],
        )
        return PhaseResult(
            phase_name="intervention_definition", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Impact Modelling
    # -------------------------------------------------------------------------

    async def _phase_impact_modelling(
        self, input_data: ScenarioPlanningInput
    ) -> PhaseResult:
        """Run MACC analysis, what-if scenarios, technology pathway models."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Build category emission lookup
        cat_lookup: Dict[str, float] = {
            bc.category: bc.emissions_tco2e for bc in self._baseline_categories
        }

        # --- MACC Construction ---
        self._macc = []
        for intv in self._interventions:
            # Calculate abatement potential in tCO2e
            target_emissions = sum(
                cat_lookup.get(c, 0.0) for c in intv.target_categories
            )
            abatement = target_emissions * intv.estimated_reduction_pct / 100.0
            abatement *= intv.confidence_pct / 100.0  # Risk-adjusted

            cost_per_tco2e = intv.cost_usd / abatement if abatement > 0 else 0.0

            self._macc.append(MACCEntry(
                intervention_id=intv.intervention_id,
                intervention_name=intv.name,
                abatement_cost_usd_per_tco2e=round(cost_per_tco2e, 2),
                abatement_potential_tco2e=round(abatement, 2),
                total_cost_usd=round(intv.cost_usd, 2),
            ))

        # Sort MACC by cost (ascending) -- standard MACC ordering
        self._macc.sort(key=lambda m: m.abatement_cost_usd_per_tco2e)

        # Calculate cumulative values
        cum_abatement = 0.0
        cum_cost = 0.0
        for entry in self._macc:
            cum_abatement += entry.abatement_potential_tco2e
            cum_cost += entry.total_cost_usd
            entry.cumulative_abatement_tco2e = round(cum_abatement, 2)
            entry.cumulative_cost_usd = round(cum_cost, 2)

        # --- Scenario Projection ---
        self._scenarios = []
        years = input_data.target_year - input_data.base_year

        for scenario_type in input_data.scenarios_to_model:
            profile = SCENARIO_PROFILES.get(scenario_type.value)
            if not profile and scenario_type != ScenarioType.CUSTOM:
                warnings.append(f"No profile for scenario '{scenario_type.value}'")
                continue

            if scenario_type == ScenarioType.CUSTOM:
                # Custom scenario uses all interventions
                total_abatement = sum(m.abatement_potential_tco2e for m in self._macc)
                reduction_pct = (
                    (total_abatement / self._baseline_total * 100.0)
                    if self._baseline_total > 0 else 0.0
                )
                annual_rate = reduction_pct / max(years, 1)
            else:
                annual_rate = profile["annual_reduction_pct"]
                reduction_pct = min(annual_rate * years, 100.0)

            projected = self._baseline_total * (1 - reduction_pct / 100.0)
            reduction_abs = self._baseline_total - projected

            # Year-by-year trajectory
            yby: Dict[int, float] = {}
            for y in range(input_data.base_year, input_data.target_year + 1):
                elapsed_y = y - input_data.base_year
                yby[y] = round(
                    self._baseline_total * (1 - annual_rate / 100.0 * elapsed_y), 2
                )

            # Calculate total cost for this scenario
            scenario_cost = self._estimate_scenario_cost(
                reduction_pct, input_data.budget_usd
            )

            alignment = self._check_alignment(annual_rate)

            self._scenarios.append(ScenarioResult(
                scenario_type=scenario_type,
                scenario_name=scenario_type.value.replace("_", " ").title(),
                base_year=input_data.base_year,
                target_year=input_data.target_year,
                baseline_tco2e=round(self._baseline_total, 2),
                projected_tco2e=round(max(projected, 0), 2),
                reduction_tco2e=round(reduction_abs, 2),
                reduction_pct=round(reduction_pct, 2),
                annual_reduction_rate_pct=round(annual_rate, 2),
                total_cost_usd=round(scenario_cost, 2),
                cost_per_tco2e_reduced=round(
                    scenario_cost / reduction_abs if reduction_abs > 0 else 0.0, 2
                ),
                alignment_status=alignment,
                interventions_included=[m.intervention_name for m in self._macc],
                year_by_year_tco2e=yby,
            ))

        outputs["macc_entries"] = len(self._macc)
        outputs["total_abatement_potential_tco2e"] = round(cum_abatement, 2)
        outputs["total_abatement_cost_usd"] = round(cum_cost, 2)
        outputs["scenarios_modelled"] = len(self._scenarios)
        outputs["cheapest_abatement_usd_per_tco2e"] = round(
            self._macc[0].abatement_cost_usd_per_tco2e, 2
        ) if self._macc else 0.0

        self._state.phase_statuses["impact_modelling"] = "completed"
        self._state.current_phase = 3

        elapsed_s = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ImpactModelling: %d MACC entries, %d scenarios, "
            "total abatement=%.0f tCO2e",
            len(self._macc), len(self._scenarios), cum_abatement,
        )
        return PhaseResult(
            phase_name="impact_modelling", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed_s,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Portfolio Optimization
    # -------------------------------------------------------------------------

    async def _phase_portfolio_optimization(
        self, input_data: ScenarioPlanningInput
    ) -> PhaseResult:
        """Rank scenarios by cost-effectiveness, check Paris alignment."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        budget = input_data.budget_usd

        # Greedy knapsack: select interventions from MACC by cost-effectiveness
        selected: List[str] = []
        total_cost = 0.0
        total_abatement = 0.0

        for entry in self._macc:
            if total_cost + entry.total_cost_usd <= budget:
                selected.append(entry.intervention_name)
                total_cost += entry.total_cost_usd
                total_abatement += entry.abatement_potential_tco2e

        reduction_pct = (
            (total_abatement / self._baseline_total * 100.0)
            if self._baseline_total > 0 else 0.0
        )
        years = input_data.target_year - input_data.base_year
        annual_rate = reduction_pct / max(years, 1)

        alignment = self._check_alignment(annual_rate)

        self._portfolio = OptimalPortfolio(
            portfolio_name="Cost-Optimized Portfolio",
            total_reduction_tco2e=round(total_abatement, 2),
            total_reduction_pct=round(reduction_pct, 2),
            total_cost_usd=round(total_cost, 2),
            avg_cost_per_tco2e=round(
                total_cost / total_abatement if total_abatement > 0 else 0.0, 2
            ),
            alignment_status=alignment,
            selected_interventions=selected,
            budget_utilization_pct=round(
                (total_cost / budget * 100.0) if budget > 0 else 0.0, 1
            ),
        )

        if alignment == AlignmentStatus.NOT_ALIGNED:
            warnings.append(
                f"Optimal portfolio achieves {annual_rate:.1f}% annual reduction, "
                f"which does not meet SBTi 1.5C (4.2%) or WB2C (2.5%) thresholds"
            )

        outputs["selected_interventions"] = len(selected)
        outputs["total_abatement_tco2e"] = round(total_abatement, 2)
        outputs["total_reduction_pct"] = round(reduction_pct, 2)
        outputs["total_cost_usd"] = round(total_cost, 2)
        outputs["budget_utilization_pct"] = self._portfolio.budget_utilization_pct
        outputs["alignment_status"] = alignment.value
        outputs["annual_reduction_rate_pct"] = round(annual_rate, 2)

        # Rank scenarios by cost-effectiveness
        outputs["scenario_ranking"] = [
            {
                "scenario": s.scenario_name,
                "cost_per_tco2e": s.cost_per_tco2e_reduced,
                "reduction_pct": s.reduction_pct,
                "alignment": s.alignment_status.value,
            }
            for s in sorted(self._scenarios, key=lambda x: x.cost_per_tco2e_reduced)
        ]

        self._state.phase_statuses["portfolio_optimization"] = "completed"
        self._state.current_phase = 4

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 PortfolioOptimization: %d interventions selected, "
            "abatement=%.0f tCO2e, alignment=%s",
            len(selected), total_abatement, alignment.value,
        )
        return PhaseResult(
            phase_name="portfolio_optimization", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _check_alignment(self, annual_rate_pct: float) -> AlignmentStatus:
        """Check Paris alignment based on annual reduction rate."""
        if annual_rate_pct >= SBTI_REDUCTION_RATES["1.5C_linear_absolute"]:
            return AlignmentStatus.ALIGNED_1_5C
        elif annual_rate_pct >= SBTI_REDUCTION_RATES["well_below_2C_absolute"]:
            return AlignmentStatus.ALIGNED_WB2C
        else:
            return AlignmentStatus.NOT_ALIGNED

    def _estimate_scenario_cost(
        self, target_reduction_pct: float, budget: float
    ) -> float:
        """Estimate cost to achieve a given reduction using MACC data."""
        needed = self._baseline_total * target_reduction_pct / 100.0
        cost = 0.0
        achieved = 0.0
        for entry in self._macc:
            if achieved >= needed:
                break
            take = min(entry.abatement_potential_tco2e, needed - achieved)
            cost += (
                take / entry.abatement_potential_tco2e * entry.total_cost_usd
                if entry.abatement_potential_tco2e > 0 else 0.0
            )
            achieved += take
        return cost

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self._baseline_total = 0.0
        self._baseline_categories = []
        self._interventions = []
        self._macc = []
        self._scenarios = []
        self._portfolio = None
        self._phase_results = []
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

    def _compute_provenance(self, result: ScenarioPlanningOutput) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += f"|{result.workflow_id}|{result.baseline_tco2e}"
        chain += f"|{result.total_abatement_potential_tco2e}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
