# -*- coding: utf-8 -*-
"""
Category Calculation Workflow
==================================

4-phase workflow for executing emission calculations across selected Scope 3
categories using GreenLang MRV agents within PACK-042 Scope 3 Starter Pack.

Phases:
    1. MethodologySelection    -- Confirm tier per category, validate data
                                  sufficiency for selected tier
    2. AgentRouting            -- Route to correct MRV agent (MRV-014 through
                                  MRV-028) via MRV-029 Category Mapper
    3. CalculationExecution    -- Run per-category calculations with provenance
                                  tracking, support parallel execution
    4. ResultValidation        -- Cross-check results against sector benchmarks,
                                  flag outliers (>2 sigma), validate completeness

The workflow follows GreenLang zero-hallucination principles: all emission
values are computed by deterministic MRV agents with published emission factors.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Value Chain (Scope 3) Standard -- Chapter 8
    GHG Protocol Technical Guidance for Calculating Scope 3 Emissions (2013)
    ISO 14064-1:2018 Clause 5.2.4

Schedule: on-demand (after data collection)
Estimated duration: 1-4 hours per category

Author: GreenLang Platform Team
Version: 42.0.0
"""

_MODULE_VERSION: str = "42.0.0"

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
    """Methodology tier for Scope 3 calculation."""

    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"


class DataSufficiencyLevel(str, Enum):
    """Data sufficiency assessment result."""

    SUFFICIENT = "sufficient"
    PARTIALLY_SUFFICIENT = "partially_sufficient"
    INSUFFICIENT = "insufficient"


class CalculationStatus(str, Enum):
    """Status of a single category calculation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    DOWNGRADED = "downgraded"


class BenchmarkResult(str, Enum):
    """Result of sector benchmark comparison."""

    WITHIN_RANGE = "within_range"
    ABOVE_AVERAGE = "above_average"
    BELOW_AVERAGE = "below_average"
    OUTLIER_HIGH = "outlier_high"
    OUTLIER_LOW = "outlier_low"
    NO_BENCHMARK = "no_benchmark"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
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


class CategoryActivityData(BaseModel):
    """Activity data for a single Scope 3 category."""

    category: Scope3Category = Field(...)
    selected_tier: MethodologyTier = Field(default=MethodologyTier.SPEND_BASED)
    activity_records: List[Dict[str, Any]] = Field(default_factory=list)
    spend_usd: float = Field(default=0.0, ge=0.0)
    data_quality_score: float = Field(default=1.0, ge=1.0, le=5.0)
    reporting_year: int = Field(default=2025)


class DataSufficiencyCheck(BaseModel):
    """Data sufficiency assessment for a category-tier combination."""

    category: Scope3Category = Field(...)
    requested_tier: MethodologyTier = Field(...)
    sufficiency_level: DataSufficiencyLevel = Field(
        default=DataSufficiencyLevel.INSUFFICIENT
    )
    approved_tier: MethodologyTier = Field(default=MethodologyTier.SPEND_BASED)
    mandatory_fields_present: int = Field(default=0, ge=0)
    mandatory_fields_required: int = Field(default=0, ge=0)
    coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    downgrade_reason: str = Field(default="")


class AgentRoutingEntry(BaseModel):
    """Routing entry mapping a category to its MRV agent."""

    category: Scope3Category = Field(...)
    mrv_agent_id: str = Field(default="", description="e.g. MRV-014")
    mrv_agent_name: str = Field(default="")
    approved_tier: MethodologyTier = Field(default=MethodologyTier.SPEND_BASED)
    parallel_group: int = Field(
        default=0, ge=0, description="Parallel execution group"
    )
    estimated_duration_seconds: float = Field(default=0.0, ge=0.0)


class CategoryCalculationResult(BaseModel):
    """Calculation result for a single Scope 3 category."""

    category: Scope3Category = Field(...)
    category_number: int = Field(default=0, ge=0, le=15)
    category_name: str = Field(default="")
    status: CalculationStatus = Field(default=CalculationStatus.PENDING)
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    emissions_co2_tco2e: float = Field(default=0.0, ge=0.0)
    emissions_ch4_tco2e: float = Field(default=0.0, ge=0.0)
    emissions_n2o_tco2e: float = Field(default=0.0, ge=0.0)
    emissions_other_tco2e: float = Field(default=0.0, ge=0.0)
    methodology_tier: MethodologyTier = Field(default=MethodologyTier.SPEND_BASED)
    mrv_agent_id: str = Field(default="")
    data_quality_score: float = Field(default=1.0, ge=1.0, le=5.0)
    uncertainty_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    benchmark_result: BenchmarkResult = Field(default=BenchmarkResult.NO_BENCHMARK)
    calculation_duration_seconds: float = Field(default=0.0, ge=0.0)
    provenance_hash: str = Field(default="")
    notes: str = Field(default="")


class BenchmarkComparison(BaseModel):
    """Benchmark comparison result for a category."""

    category: Scope3Category = Field(...)
    calculated_tco2e: float = Field(default=0.0, ge=0.0)
    sector_mean_tco2e: float = Field(default=0.0, ge=0.0)
    sector_std_tco2e: float = Field(default=0.0, ge=0.0)
    z_score: float = Field(default=0.0)
    result: BenchmarkResult = Field(default=BenchmarkResult.NO_BENCHMARK)


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class CategoryCalculationInput(BaseModel):
    """Input data model for CategoryCalculationWorkflow."""

    category_data: List[CategoryActivityData] = Field(
        default_factory=list, description="Activity data per category"
    )
    methodology_overrides: Dict[str, str] = Field(
        default_factory=dict, description="Category -> forced tier"
    )
    parallel_execution: bool = Field(
        default=True, description="Enable parallel category execution"
    )
    max_parallel: int = Field(
        default=15, ge=1, le=15, description="Max parallel categories"
    )
    sector: str = Field(default="", description="Organization sector for benchmarks")
    employee_count: int = Field(default=0, ge=0)
    revenue_usd: float = Field(default=0.0, ge=0.0)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class CategoryCalculationOutput(BaseModel):
    """Complete result from category calculation workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="category_calculation")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    category_results: List[CategoryCalculationResult] = Field(default_factory=list)
    total_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    upstream_tco2e: float = Field(default=0.0, ge=0.0)
    downstream_tco2e: float = Field(default=0.0, ge=0.0)
    data_sufficiency_checks: List[DataSufficiencyCheck] = Field(default_factory=list)
    agent_routing: List[AgentRoutingEntry] = Field(default_factory=list)
    benchmark_comparisons: List[BenchmarkComparison] = Field(default_factory=list)
    categories_completed: int = Field(default=0, ge=0)
    categories_failed: int = Field(default=0, ge=0)
    categories_downgraded: int = Field(default=0, ge=0)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# MRV agent routing table: Scope 3 category -> MRV agent ID
CATEGORY_TO_MRV_AGENT: Dict[Scope3Category, Tuple[str, str]] = {
    Scope3Category.CAT_01_PURCHASED_GOODS: ("MRV-014", "Scope 3 Category 1 Agent"),
    Scope3Category.CAT_02_CAPITAL_GOODS: ("MRV-015", "Scope 3 Category 2 Agent"),
    Scope3Category.CAT_03_FUEL_ENERGY: ("MRV-016", "Scope 3 Category 3 Agent"),
    Scope3Category.CAT_04_UPSTREAM_TRANSPORT: ("MRV-017", "Scope 3 Category 4 Agent"),
    Scope3Category.CAT_05_WASTE: ("MRV-018", "Scope 3 Category 5 Agent"),
    Scope3Category.CAT_06_BUSINESS_TRAVEL: ("MRV-019", "Scope 3 Category 6 Agent"),
    Scope3Category.CAT_07_COMMUTING: ("MRV-020", "Scope 3 Category 7 Agent"),
    Scope3Category.CAT_08_UPSTREAM_LEASED: ("MRV-021", "Scope 3 Category 8 Agent"),
    Scope3Category.CAT_09_DOWNSTREAM_TRANSPORT: ("MRV-022", "Scope 3 Category 9 Agent"),
    Scope3Category.CAT_10_PROCESSING: ("MRV-023", "Scope 3 Category 10 Agent"),
    Scope3Category.CAT_11_USE_SOLD: ("MRV-024", "Scope 3 Category 11 Agent"),
    Scope3Category.CAT_12_END_OF_LIFE: ("MRV-025", "Scope 3 Category 12 Agent"),
    Scope3Category.CAT_13_DOWNSTREAM_LEASED: ("MRV-026", "Scope 3 Category 13 Agent"),
    Scope3Category.CAT_14_FRANCHISES: ("MRV-027", "Scope 3 Category 14 Agent"),
    Scope3Category.CAT_15_INVESTMENTS: ("MRV-028", "Scope 3 Category 15 Agent"),
}

# Category numbers
CATEGORY_NUMBERS: Dict[Scope3Category, int] = {
    Scope3Category.CAT_01_PURCHASED_GOODS: 1,
    Scope3Category.CAT_02_CAPITAL_GOODS: 2,
    Scope3Category.CAT_03_FUEL_ENERGY: 3,
    Scope3Category.CAT_04_UPSTREAM_TRANSPORT: 4,
    Scope3Category.CAT_05_WASTE: 5,
    Scope3Category.CAT_06_BUSINESS_TRAVEL: 6,
    Scope3Category.CAT_07_COMMUTING: 7,
    Scope3Category.CAT_08_UPSTREAM_LEASED: 8,
    Scope3Category.CAT_09_DOWNSTREAM_TRANSPORT: 9,
    Scope3Category.CAT_10_PROCESSING: 10,
    Scope3Category.CAT_11_USE_SOLD: 11,
    Scope3Category.CAT_12_END_OF_LIFE: 12,
    Scope3Category.CAT_13_DOWNSTREAM_LEASED: 13,
    Scope3Category.CAT_14_FRANCHISES: 14,
    Scope3Category.CAT_15_INVESTMENTS: 15,
}

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

# Upstream categories (1-8) vs downstream (9-15)
UPSTREAM_CATEGORIES = {
    Scope3Category.CAT_01_PURCHASED_GOODS,
    Scope3Category.CAT_02_CAPITAL_GOODS,
    Scope3Category.CAT_03_FUEL_ENERGY,
    Scope3Category.CAT_04_UPSTREAM_TRANSPORT,
    Scope3Category.CAT_05_WASTE,
    Scope3Category.CAT_06_BUSINESS_TRAVEL,
    Scope3Category.CAT_07_COMMUTING,
    Scope3Category.CAT_08_UPSTREAM_LEASED,
}

# Minimum data fields per tier for sufficiency check
MIN_FIELDS_PER_TIER: Dict[str, int] = {
    "spend_based": 1,
    "average_data": 2,
    "supplier_specific": 3,
    "hybrid": 2,
}

# Sector benchmark: mean & std (tCO2e per M USD revenue) per category
SECTOR_BENCHMARKS: Dict[str, Dict[str, Dict[str, float]]] = {
    "manufacturing": {
        "cat_01_purchased_goods_services": {"mean": 450.0, "std": 200.0},
        "cat_02_capital_goods": {"mean": 50.0, "std": 30.0},
        "cat_03_fuel_energy_related": {"mean": 30.0, "std": 15.0},
        "cat_04_upstream_transport": {"mean": 80.0, "std": 40.0},
        "cat_05_waste_in_operations": {"mean": 20.0, "std": 15.0},
        "cat_06_business_travel": {"mean": 15.0, "std": 10.0},
        "cat_07_employee_commuting": {"mean": 20.0, "std": 10.0},
        "cat_11_use_of_sold_products": {"mean": 150.0, "std": 100.0},
    },
    "services": {
        "cat_01_purchased_goods_services": {"mean": 100.0, "std": 50.0},
        "cat_06_business_travel": {"mean": 40.0, "std": 20.0},
        "cat_07_employee_commuting": {"mean": 30.0, "std": 15.0},
    },
    "default": {
        "cat_01_purchased_goods_services": {"mean": 200.0, "std": 100.0},
        "cat_06_business_travel": {"mean": 25.0, "std": 15.0},
        "cat_07_employee_commuting": {"mean": 25.0, "std": 12.0},
    },
}

# Default uncertainty percentages by tier (GHG Protocol guidance)
TIER_UNCERTAINTY_PCT: Dict[str, float] = {
    "spend_based": 50.0,
    "average_data": 30.0,
    "supplier_specific": 10.0,
    "hybrid": 25.0,
}

# Fallback EEIO factors for spend-based calculations (kgCO2e/USD)
EEIO_FALLBACK_FACTORS: Dict[str, float] = {
    "cat_01_purchased_goods_services": 0.40,
    "cat_02_capital_goods": 0.35,
    "cat_03_fuel_energy_related": 0.60,
    "cat_04_upstream_transport": 0.85,
    "cat_05_waste_in_operations": 0.12,
    "cat_06_business_travel": 0.50,
    "cat_07_employee_commuting": 0.30,
    "cat_08_upstream_leased_assets": 0.10,
    "cat_09_downstream_transport": 0.85,
    "cat_10_processing_sold_products": 0.45,
    "cat_11_use_of_sold_products": 0.60,
    "cat_12_end_of_life_treatment": 0.12,
    "cat_13_downstream_leased_assets": 0.10,
    "cat_14_franchises": 0.12,
    "cat_15_investments": 0.05,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class CategoryCalculationWorkflow:
    """
    4-phase Scope 3 category calculation workflow.

    Validates data sufficiency per category, routes to the correct MRV agent
    (MRV-014 through MRV-028), executes calculations with provenance tracking,
    and cross-checks results against sector benchmarks.

    Zero-hallucination: all emission values are computed by deterministic
    formulas or delegated to MRV agents. No LLM calls in numeric paths.

    Attributes:
        workflow_id: Unique execution identifier.
        _sufficiency_checks: Data sufficiency assessments.
        _routing_entries: Agent routing table.
        _category_results: Per-category calculation results.
        _benchmarks: Benchmark comparison results.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = CategoryCalculationWorkflow()
        >>> data = [CategoryActivityData(
        ...     category=Scope3Category.CAT_01_PURCHASED_GOODS,
        ...     selected_tier=MethodologyTier.SPEND_BASED,
        ...     spend_usd=50_000_000,
        ... )]
        >>> inp = CategoryCalculationInput(category_data=data, sector="manufacturing")
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_NAMES: List[str] = [
        "methodology_selection",
        "agent_routing",
        "calculation_execution",
        "result_validation",
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize CategoryCalculationWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._sufficiency_checks: List[DataSufficiencyCheck] = []
        self._routing_entries: List[AgentRoutingEntry] = []
        self._category_results: List[CategoryCalculationResult] = []
        self._benchmarks: List[BenchmarkComparison] = []
        self._phase_results: List[PhaseResult] = []
        self._category_data_map: Dict[str, CategoryActivityData] = {}
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
        input_data: Optional[CategoryCalculationInput] = None,
        category_data: Optional[List[CategoryActivityData]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> CategoryCalculationOutput:
        """
        Execute the 4-phase category calculation workflow.

        Args:
            input_data: Full input model (preferred).
            category_data: Activity data per category (fallback).
            config: Optional configuration overrides.

        Returns:
            CategoryCalculationOutput with per-category emission results.
        """
        if input_data is None:
            input_data = CategoryCalculationInput(
                category_data=category_data or [],
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting category calculation workflow %s categories=%d",
            self.workflow_id, len(input_data.category_data),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        # Build category data lookup
        self._category_data_map = {
            cd.category.value: cd for cd in input_data.category_data
        }

        try:
            phase1 = await self._execute_with_retry(
                self._phase_methodology_selection, input_data, 1
            )
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 1 failed: {phase1.errors}")

            phase2 = await self._execute_with_retry(
                self._phase_agent_routing, input_data, 2
            )
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 2 failed: {phase2.errors}")

            phase3 = await self._execute_with_retry(
                self._phase_calculation_execution, input_data, 3
            )
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 3 failed: {phase3.errors}")

            phase4 = await self._execute_with_retry(
                self._phase_result_validation, input_data, 4
            )
            self._phase_results.append(phase4)
            if phase4.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 4 failed: {phase4.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Category calculation workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        total_scope3 = sum(cr.emissions_tco2e for cr in self._category_results)
        upstream = sum(
            cr.emissions_tco2e for cr in self._category_results
            if cr.category in UPSTREAM_CATEGORIES
        )
        downstream = sum(
            cr.emissions_tco2e for cr in self._category_results
            if cr.category not in UPSTREAM_CATEGORIES
        )
        completed = sum(
            1 for cr in self._category_results
            if cr.status == CalculationStatus.COMPLETED
        )
        failed = sum(
            1 for cr in self._category_results
            if cr.status == CalculationStatus.FAILED
        )
        downgraded = sum(
            1 for cr in self._category_results
            if cr.status == CalculationStatus.DOWNGRADED
        )

        result = CategoryCalculationOutput(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            category_results=self._category_results,
            total_scope3_tco2e=round(total_scope3, 2),
            upstream_tco2e=round(upstream, 2),
            downstream_tco2e=round(downstream, 2),
            data_sufficiency_checks=self._sufficiency_checks,
            agent_routing=self._routing_entries,
            benchmark_comparisons=self._benchmarks,
            categories_completed=completed,
            categories_failed=failed,
            categories_downgraded=downgraded,
            progress_pct=100.0 if overall_status == WorkflowStatus.COMPLETED else 0.0,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Category calculation workflow %s completed in %.2fs "
            "total=%.1f tCO2e upstream=%.1f downstream=%.1f "
            "completed=%d failed=%d downgraded=%d",
            self.workflow_id, elapsed, total_scope3, upstream, downstream,
            completed, failed, downgraded,
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: CategoryCalculationInput,
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
    # Phase 1: Methodology Selection
    # -------------------------------------------------------------------------

    async def _phase_methodology_selection(
        self, input_data: CategoryCalculationInput
    ) -> PhaseResult:
        """Confirm tier per category and validate data sufficiency."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._sufficiency_checks = []

        for cd in input_data.category_data:
            cat_key = cd.category.value
            requested_tier = cd.selected_tier

            # Override if methodology_overrides specified
            override = input_data.methodology_overrides.get(cat_key)
            if override:
                try:
                    requested_tier = MethodologyTier(override)
                except ValueError:
                    warnings.append(
                        f"Invalid tier override '{override}' for {cat_key}"
                    )

            # Check data sufficiency
            sufficiency = self._check_data_sufficiency(cd, requested_tier)
            self._sufficiency_checks.append(sufficiency)

            if sufficiency.sufficiency_level == DataSufficiencyLevel.INSUFFICIENT:
                warnings.append(
                    f"{cat_key}: insufficient data for {requested_tier.value}; "
                    f"downgrading to {sufficiency.approved_tier.value}"
                )

        outputs["categories_assessed"] = len(self._sufficiency_checks)
        outputs["sufficient"] = sum(
            1 for s in self._sufficiency_checks
            if s.sufficiency_level == DataSufficiencyLevel.SUFFICIENT
        )
        outputs["partially_sufficient"] = sum(
            1 for s in self._sufficiency_checks
            if s.sufficiency_level == DataSufficiencyLevel.PARTIALLY_SUFFICIENT
        )
        outputs["insufficient"] = sum(
            1 for s in self._sufficiency_checks
            if s.sufficiency_level == DataSufficiencyLevel.INSUFFICIENT
        )
        outputs["tier_distribution"] = {}
        for s in self._sufficiency_checks:
            tier_val = s.approved_tier.value
            outputs["tier_distribution"][tier_val] = (
                outputs["tier_distribution"].get(tier_val, 0) + 1
            )

        self._state.progress_pct = 15.0

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 MethodologySelection: %d categories, %d sufficient, "
            "%d downgraded",
            len(self._sufficiency_checks), outputs["sufficient"],
            outputs["insufficient"],
        )
        return PhaseResult(
            phase_name="methodology_selection", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _check_data_sufficiency(
        self, data: CategoryActivityData, requested_tier: MethodologyTier
    ) -> DataSufficiencyCheck:
        """Validate data sufficiency for a category-tier combination."""
        min_fields = MIN_FIELDS_PER_TIER.get(requested_tier.value, 1)
        has_spend = data.spend_usd > 0
        has_records = len(data.activity_records) > 0
        total_fields = sum(len(r) for r in data.activity_records)

        # Determine sufficiency
        if requested_tier == MethodologyTier.SPEND_BASED:
            if has_spend or has_records:
                level = DataSufficiencyLevel.SUFFICIENT
                approved = requested_tier
            else:
                level = DataSufficiencyLevel.INSUFFICIENT
                approved = MethodologyTier.SPEND_BASED
        elif requested_tier == MethodologyTier.AVERAGE_DATA:
            if has_records and total_fields >= min_fields * len(data.activity_records):
                level = DataSufficiencyLevel.SUFFICIENT
                approved = requested_tier
            elif has_spend:
                level = DataSufficiencyLevel.PARTIALLY_SUFFICIENT
                approved = MethodologyTier.SPEND_BASED
            else:
                level = DataSufficiencyLevel.INSUFFICIENT
                approved = MethodologyTier.SPEND_BASED
        elif requested_tier == MethodologyTier.SUPPLIER_SPECIFIC:
            if has_records and total_fields >= min_fields * len(data.activity_records):
                level = DataSufficiencyLevel.SUFFICIENT
                approved = requested_tier
            elif has_records:
                level = DataSufficiencyLevel.PARTIALLY_SUFFICIENT
                approved = MethodologyTier.AVERAGE_DATA
            else:
                level = DataSufficiencyLevel.INSUFFICIENT
                approved = MethodologyTier.SPEND_BASED
        else:
            level = DataSufficiencyLevel.SUFFICIENT
            approved = requested_tier

        coverage = 100.0 if level == DataSufficiencyLevel.SUFFICIENT else (
            50.0 if level == DataSufficiencyLevel.PARTIALLY_SUFFICIENT else 0.0
        )

        downgrade_reason = ""
        if approved != requested_tier:
            downgrade_reason = (
                f"Insufficient data for {requested_tier.value}; "
                f"fields present: {total_fields}, minimum required: "
                f"{min_fields * max(len(data.activity_records), 1)}"
            )

        return DataSufficiencyCheck(
            category=data.category,
            requested_tier=requested_tier,
            sufficiency_level=level,
            approved_tier=approved,
            mandatory_fields_present=total_fields,
            mandatory_fields_required=min_fields * max(len(data.activity_records), 1),
            coverage_pct=coverage,
            downgrade_reason=downgrade_reason,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Agent Routing
    # -------------------------------------------------------------------------

    async def _phase_agent_routing(
        self, input_data: CategoryCalculationInput
    ) -> PhaseResult:
        """Route each category to its MRV agent."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._routing_entries = []
        group = 0

        for check in self._sufficiency_checks:
            agent_info = CATEGORY_TO_MRV_AGENT.get(check.category)
            if not agent_info:
                warnings.append(
                    f"No MRV agent mapped for {check.category.value}"
                )
                continue

            agent_id, agent_name = agent_info

            # Assign parallel group (all can run in parallel unless disabled)
            if not input_data.parallel_execution:
                group += 1

            self._routing_entries.append(AgentRoutingEntry(
                category=check.category,
                mrv_agent_id=agent_id,
                mrv_agent_name=agent_name,
                approved_tier=check.approved_tier,
                parallel_group=0 if input_data.parallel_execution else group,
                estimated_duration_seconds=self._estimate_duration(
                    check.category, check.approved_tier
                ),
            ))

        outputs["agents_routed"] = len(self._routing_entries)
        outputs["unique_agents"] = len({r.mrv_agent_id for r in self._routing_entries})
        outputs["parallel_groups"] = (
            max(r.parallel_group for r in self._routing_entries) + 1
            if self._routing_entries else 0
        )
        outputs["routing_table"] = {
            r.category.value: {
                "agent": r.mrv_agent_id,
                "tier": r.approved_tier.value,
                "group": r.parallel_group,
            }
            for r in self._routing_entries
        }

        self._state.progress_pct = 30.0

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 AgentRouting: %d agents routed, %d parallel groups",
            len(self._routing_entries), outputs["parallel_groups"],
        )
        return PhaseResult(
            phase_name="agent_routing", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _estimate_duration(
        self, category: Scope3Category, tier: MethodologyTier
    ) -> float:
        """Estimate calculation duration in seconds."""
        base_durations: Dict[str, float] = {
            "spend_based": 5.0,
            "average_data": 15.0,
            "supplier_specific": 30.0,
            "hybrid": 20.0,
        }
        return base_durations.get(tier.value, 10.0)

    # -------------------------------------------------------------------------
    # Phase 3: Calculation Execution
    # -------------------------------------------------------------------------

    async def _phase_calculation_execution(
        self, input_data: CategoryCalculationInput
    ) -> PhaseResult:
        """Execute per-category calculations via MRV agents."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._category_results = []

        for routing in self._routing_entries:
            calc_start = datetime.utcnow()
            cat_key = routing.category.value
            activity_data = self._category_data_map.get(cat_key)

            try:
                result = self._execute_category_calculation(
                    routing, activity_data, input_data.sector
                )
                calc_elapsed = (datetime.utcnow() - calc_start).total_seconds()
                result.calculation_duration_seconds = calc_elapsed
                self._category_results.append(result)

                self.logger.info(
                    "Category %s calculated: %.2f tCO2e via %s (%s) in %.2fs",
                    cat_key, result.emissions_tco2e, routing.mrv_agent_id,
                    routing.approved_tier.value, calc_elapsed,
                )

            except Exception as exc:
                calc_elapsed = (datetime.utcnow() - calc_start).total_seconds()
                self.logger.error(
                    "Category %s calculation failed: %s", cat_key, exc
                )
                warnings.append(f"Category {cat_key} calculation failed: {exc}")
                self._category_results.append(CategoryCalculationResult(
                    category=routing.category,
                    category_number=CATEGORY_NUMBERS.get(routing.category, 0),
                    category_name=CATEGORY_NAMES.get(routing.category, cat_key),
                    status=CalculationStatus.FAILED,
                    mrv_agent_id=routing.mrv_agent_id,
                    methodology_tier=routing.approved_tier,
                    calculation_duration_seconds=calc_elapsed,
                    notes=f"Calculation failed: {exc}",
                ))

        total = sum(cr.emissions_tco2e for cr in self._category_results)
        completed = sum(
            1 for cr in self._category_results
            if cr.status in (CalculationStatus.COMPLETED, CalculationStatus.DOWNGRADED)
        )

        outputs["total_tco2e"] = round(total, 2)
        outputs["categories_completed"] = completed
        outputs["categories_failed"] = sum(
            1 for cr in self._category_results
            if cr.status == CalculationStatus.FAILED
        )
        outputs["per_category"] = {
            cr.category.value: {
                "tco2e": cr.emissions_tco2e,
                "status": cr.status.value,
                "tier": cr.methodology_tier.value,
            }
            for cr in self._category_results
        }

        self._state.progress_pct = 75.0

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 CalculationExecution: total=%.1f tCO2e, %d/%d completed",
            total, completed, len(self._routing_entries),
        )
        return PhaseResult(
            phase_name="calculation_execution", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _execute_category_calculation(
        self,
        routing: AgentRoutingEntry,
        activity_data: Optional[CategoryActivityData],
        sector: str,
    ) -> CategoryCalculationResult:
        """
        Execute deterministic calculation for a single category.

        Zero-hallucination: uses only EEIO factors and arithmetic.
        In production, this delegates to the actual MRV agent.
        """
        cat = routing.category
        cat_num = CATEGORY_NUMBERS.get(cat, 0)
        cat_name = CATEGORY_NAMES.get(cat, cat.value)
        tier = routing.approved_tier

        # Determine spend or activity-based calculation
        spend = activity_data.spend_usd if activity_data else 0.0
        records = activity_data.activity_records if activity_data else []

        emissions_tco2e = 0.0

        if tier == MethodologyTier.SPEND_BASED:
            # Spend-based: spend * EEIO factor / 1000 (kg -> tonnes)
            ef = EEIO_FALLBACK_FACTORS.get(cat.value, 0.40)
            emissions_tco2e = (spend * ef) / 1000.0

        elif tier in (MethodologyTier.AVERAGE_DATA, MethodologyTier.HYBRID):
            # Average data: sum(activity * emission_factor) per record
            for record in records:
                activity = float(record.get("activity_data", 0.0))
                ef = float(record.get("emission_factor", 0.0))
                emissions_tco2e += (activity * ef) / 1000.0
            # Fallback to spend if no records
            if emissions_tco2e <= 0 and spend > 0:
                ef = EEIO_FALLBACK_FACTORS.get(cat.value, 0.40)
                emissions_tco2e = (spend * ef) / 1000.0

        elif tier == MethodologyTier.SUPPLIER_SPECIFIC:
            # Supplier-specific: sum(activity * supplier_ef) per record
            for record in records:
                activity = float(record.get("activity_data", 0.0))
                ef = float(record.get("supplier_ef", record.get("emission_factor", 0.0)))
                emissions_tco2e += (activity * ef) / 1000.0
            if emissions_tco2e <= 0 and spend > 0:
                ef = EEIO_FALLBACK_FACTORS.get(cat.value, 0.40)
                emissions_tco2e = (spend * ef) / 1000.0

        # Gas breakdown (simplified: assume 95% CO2, 3% CH4, 2% N2O for most)
        emissions_co2 = emissions_tco2e * 0.95
        emissions_ch4 = emissions_tco2e * 0.03
        emissions_n2o = emissions_tco2e * 0.02

        # Data quality
        dq = activity_data.data_quality_score if activity_data else 1.0

        # Uncertainty
        uncertainty = TIER_UNCERTAINTY_PCT.get(tier.value, 50.0)

        # Provenance
        prov_str = f"{cat.value}|{tier.value}|{emissions_tco2e}|{spend}"
        prov_hash = hashlib.sha256(prov_str.encode("utf-8")).hexdigest()

        # Determine status
        sufficiency = next(
            (s for s in self._sufficiency_checks if s.category == cat), None
        )
        status = CalculationStatus.COMPLETED
        if sufficiency and sufficiency.approved_tier != sufficiency.requested_tier:
            status = CalculationStatus.DOWNGRADED

        return CategoryCalculationResult(
            category=cat,
            category_number=cat_num,
            category_name=cat_name,
            status=status,
            emissions_tco2e=round(emissions_tco2e, 2),
            emissions_co2_tco2e=round(emissions_co2, 2),
            emissions_ch4_tco2e=round(emissions_ch4, 2),
            emissions_n2o_tco2e=round(emissions_n2o, 2),
            emissions_other_tco2e=0.0,
            methodology_tier=tier,
            mrv_agent_id=routing.mrv_agent_id,
            data_quality_score=dq,
            uncertainty_pct=uncertainty,
            provenance_hash=prov_hash,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Result Validation
    # -------------------------------------------------------------------------

    async def _phase_result_validation(
        self, input_data: CategoryCalculationInput
    ) -> PhaseResult:
        """Cross-check results against sector benchmarks and flag outliers."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._benchmarks = []
        sector_key = self._normalize_sector(input_data.sector)
        sector_data = SECTOR_BENCHMARKS.get(
            sector_key, SECTOR_BENCHMARKS["default"]
        )

        # Normalize values to per-M-USD-revenue if revenue available
        revenue_m = input_data.revenue_usd / 1_000_000.0 if input_data.revenue_usd > 0 else 1.0

        outlier_count = 0

        for cr in self._category_results:
            if cr.status == CalculationStatus.FAILED:
                continue

            cat_key = cr.category.value
            benchmark = sector_data.get(cat_key)

            if not benchmark:
                self._benchmarks.append(BenchmarkComparison(
                    category=cr.category,
                    calculated_tco2e=cr.emissions_tco2e,
                    result=BenchmarkResult.NO_BENCHMARK,
                ))
                continue

            mean = benchmark["mean"] * revenue_m
            std = benchmark["std"] * revenue_m

            z_score = 0.0
            if std > 0:
                z_score = (cr.emissions_tco2e - mean) / std

            if abs(z_score) > 2.0:
                result = (
                    BenchmarkResult.OUTLIER_HIGH
                    if z_score > 0
                    else BenchmarkResult.OUTLIER_LOW
                )
                outlier_count += 1
                warnings.append(
                    f"{cr.category_name}: emissions {cr.emissions_tco2e:.1f} tCO2e "
                    f"is an outlier (z={z_score:.1f}, mean={mean:.1f})"
                )
                cr.benchmark_result = result
            elif z_score > 1.0:
                result = BenchmarkResult.ABOVE_AVERAGE
                cr.benchmark_result = result
            elif z_score < -1.0:
                result = BenchmarkResult.BELOW_AVERAGE
                cr.benchmark_result = result
            else:
                result = BenchmarkResult.WITHIN_RANGE
                cr.benchmark_result = result

            self._benchmarks.append(BenchmarkComparison(
                category=cr.category,
                calculated_tco2e=cr.emissions_tco2e,
                sector_mean_tco2e=round(mean, 2),
                sector_std_tco2e=round(std, 2),
                z_score=round(z_score, 2),
                result=result,
            ))

        # Completeness check
        completed_cats = sum(
            1 for cr in self._category_results
            if cr.status in (CalculationStatus.COMPLETED, CalculationStatus.DOWNGRADED)
        )
        completeness_pct = (
            completed_cats / len(self._category_results) * 100.0
            if self._category_results else 0.0
        )

        outputs["benchmarks_compared"] = len(self._benchmarks)
        outputs["outliers_detected"] = outlier_count
        outputs["within_range"] = sum(
            1 for b in self._benchmarks if b.result == BenchmarkResult.WITHIN_RANGE
        )
        outputs["completeness_pct"] = round(completeness_pct, 1)
        outputs["sector_key"] = sector_key

        self._state.progress_pct = 100.0

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ResultValidation: %d benchmarks, %d outliers, "
            "%.1f%% completeness",
            len(self._benchmarks), outlier_count, completeness_pct,
        )
        return PhaseResult(
            phase_name="result_validation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _normalize_sector(self, sector: str) -> str:
        """Normalize sector string to benchmark key."""
        if not sector:
            return "default"
        sector_lower = sector.lower().strip()
        for key in ("manufacturing", "services", "retail", "finance"):
            if key in sector_lower:
                return key
        return "default"

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self._sufficiency_checks = []
        self._routing_entries = []
        self._category_results = []
        self._benchmarks = []
        self._phase_results = []
        self._category_data_map = {}
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: CategoryCalculationOutput) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += f"|{result.workflow_id}|{result.total_scope3_tco2e}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
