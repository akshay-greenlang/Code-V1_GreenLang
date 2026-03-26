# -*- coding: utf-8 -*-
"""
Consolidation Workflow
============================

4-phase workflow for consolidating all Scope 3 category results, resolving
double-counting, and integrating with Scope 1+2 data within PACK-042 Scope 3
Starter Pack.

Phases:
    1. CategoryAggregation     -- Sum per-category results into upstream
                                  (Cat 1-8) and downstream (Cat 9-15) subtotals
                                  and total Scope 3
    2. DoubleCountingCheck     -- Run 12-rule double-counting detection engine,
                                  flag overlaps, apply resolution
    3. ScopeIntegration        -- Integrate with Scope 1+2 (PACK-041) if
                                  available for full GHG footprint view
    4. FinalReconciliation     -- Produce reconciled Scope 3 total with
                                  per-category breakdown and full audit trail

The workflow follows GreenLang zero-hallucination principles: all aggregation
and double-counting resolution uses deterministic arithmetic. SHA-256
provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Value Chain (Scope 3) Standard -- Chapter 9
    GHG Protocol Corporate Standard -- Chapter 8 (Accounting for Scope 2)
    ISO 14064-1:2018 Clause 5.3

Schedule: on-demand (after category calculations)
Estimated duration: 1-2 hours

Author: GreenLang Platform Team
Version: 42.0.0
"""

_MODULE_VERSION: str = "42.0.0"

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


class DoubleCountType(str, Enum):
    """Types of double-counting between Scope 3 categories."""

    TRANSPORT_OVERLAP = "transport_overlap"
    FUEL_ENERGY_OVERLAP = "fuel_energy_overlap"
    LEASED_ASSET_OVERLAP = "leased_asset_overlap"
    SCOPE1_2_OVERLAP = "scope1_2_overlap"
    FRANCHISE_LEASED_OVERLAP = "franchise_leased_overlap"
    INVESTMENT_FRANCHISE_OVERLAP = "investment_franchise_overlap"
    UPSTREAM_DOWNSTREAM_OVERLAP = "upstream_downstream_overlap"
    CAPITAL_PURCHASED_OVERLAP = "capital_purchased_overlap"
    WASTE_PROCESSING_OVERLAP = "waste_processing_overlap"
    USE_EOL_OVERLAP = "use_eol_overlap"
    COMMUTING_TRAVEL_OVERLAP = "commuting_travel_overlap"
    SUPPLY_CHAIN_CIRCULAR = "supply_chain_circular"


class ResolutionAction(str, Enum):
    """Action taken to resolve double-counting."""

    DEDUCT = "deduct"
    REALLOCATE = "reallocate"
    FLAG_ONLY = "flag_only"
    EXCLUDE = "exclude"
    NO_ACTION = "no_action"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class CategoryEmission(BaseModel):
    """Emission data for a single Scope 3 category."""

    category: Scope3Category = Field(...)
    category_number: int = Field(default=0, ge=0, le=15)
    category_name: str = Field(default="")
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    methodology_tier: str = Field(default="spend_based")
    data_quality_score: float = Field(default=1.0, ge=1.0, le=5.0)
    uncertainty_pct: float = Field(default=50.0, ge=0.0, le=100.0)


class DoubleCountFlag(BaseModel):
    """Detected double-counting overlap."""

    flag_id: str = Field(
        default_factory=lambda: f"dc-{uuid.uuid4().hex[:8]}"
    )
    overlap_type: DoubleCountType = Field(...)
    category_a: str = Field(default="")
    category_b: str = Field(default="")
    estimated_overlap_tco2e: float = Field(default=0.0, ge=0.0)
    overlap_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    resolution_action: ResolutionAction = Field(default=ResolutionAction.FLAG_ONLY)
    adjustment_tco2e: float = Field(default=0.0)
    description: str = Field(default="")
    rule_id: str = Field(default="")


class Scope12Data(BaseModel):
    """Scope 1 and 2 emission data for full GHG footprint integration."""

    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    reporting_year: int = Field(default=2025)
    source_pack: str = Field(default="PACK-041")
    provenance_hash: str = Field(default="")


class AuditTrailEntry(BaseModel):
    """Audit trail entry for reconciliation transparency."""

    entry_id: str = Field(
        default_factory=lambda: f"audit-{uuid.uuid4().hex[:8]}"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    action: str = Field(default="")
    description: str = Field(default="")
    before_value: float = Field(default=0.0)
    after_value: float = Field(default=0.0)
    affected_category: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class ConsolidationInput(BaseModel):
    """Input data model for ConsolidationWorkflow."""

    category_emissions: List[CategoryEmission] = Field(
        default_factory=list, description="Per-category emission results"
    )
    scope12_data: Optional[Scope12Data] = Field(
        default=None, description="Scope 1+2 data for integration"
    )
    enable_double_counting_resolution: bool = Field(
        default=True, description="Enable automatic double-counting resolution"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    organization_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class ConsolidationOutput(BaseModel):
    """Complete result from consolidation workflow."""

    workflow_id: str = Field(...)
    workflow_name: str = Field(default="consolidation")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    # Scope 3 totals
    scope3_total_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_upstream_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_downstream_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_adjusted_tco2e: float = Field(
        default=0.0, ge=0.0, description="After double-counting resolution"
    )
    # Per-category breakdown
    category_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    # Double counting
    double_count_flags: List[DoubleCountFlag] = Field(default_factory=list)
    total_adjustment_tco2e: float = Field(default=0.0)
    # Full footprint
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    total_footprint_location_tco2e: float = Field(default=0.0, ge=0.0)
    total_footprint_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    # Audit
    audit_trail: List[AuditTrailEntry] = Field(default_factory=list)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# DOUBLE-COUNTING RULES (Zero-Hallucination)
# =============================================================================

# 12 double-counting detection rules based on GHG Protocol Scope 3 guidance
DOUBLE_COUNTING_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "DC-001",
        "type": DoubleCountType.TRANSPORT_OVERLAP,
        "category_a": "cat_04_upstream_transport",
        "category_b": "cat_09_downstream_transport",
        "description": "Transport costs included in purchased goods spend may also appear in Cat 4/9",
        "overlap_estimate_pct": 5.0,
        "resolution": ResolutionAction.FLAG_ONLY,
    },
    {
        "rule_id": "DC-002",
        "type": DoubleCountType.FUEL_ENERGY_OVERLAP,
        "category_a": "cat_03_fuel_energy_related",
        "category_b": "cat_01_purchased_goods_services",
        "description": "Upstream fuel/energy emissions may overlap with purchased goods if energy is embedded",
        "overlap_estimate_pct": 3.0,
        "resolution": ResolutionAction.FLAG_ONLY,
    },
    {
        "rule_id": "DC-003",
        "type": DoubleCountType.LEASED_ASSET_OVERLAP,
        "category_a": "cat_08_upstream_leased_assets",
        "category_b": "cat_13_downstream_leased_assets",
        "description": "Assets may be counted in both upstream and downstream leased categories",
        "overlap_estimate_pct": 10.0,
        "resolution": ResolutionAction.DEDUCT,
    },
    {
        "rule_id": "DC-004",
        "type": DoubleCountType.FRANCHISE_LEASED_OVERLAP,
        "category_a": "cat_14_franchises",
        "category_b": "cat_13_downstream_leased_assets",
        "description": "Franchise locations may also be downstream leased assets",
        "overlap_estimate_pct": 15.0,
        "resolution": ResolutionAction.DEDUCT,
    },
    {
        "rule_id": "DC-005",
        "type": DoubleCountType.INVESTMENT_FRANCHISE_OVERLAP,
        "category_a": "cat_15_investments",
        "category_b": "cat_14_franchises",
        "description": "Investments in franchisees may overlap with franchise category",
        "overlap_estimate_pct": 20.0,
        "resolution": ResolutionAction.DEDUCT,
    },
    {
        "rule_id": "DC-006",
        "type": DoubleCountType.CAPITAL_PURCHASED_OVERLAP,
        "category_a": "cat_02_capital_goods",
        "category_b": "cat_01_purchased_goods_services",
        "description": "Capital goods may be double-counted if included in general procurement spend",
        "overlap_estimate_pct": 8.0,
        "resolution": ResolutionAction.DEDUCT,
    },
    {
        "rule_id": "DC-007",
        "type": DoubleCountType.WASTE_PROCESSING_OVERLAP,
        "category_a": "cat_05_waste_in_operations",
        "category_b": "cat_12_end_of_life_treatment",
        "description": "Waste from operations may overlap with end-of-life treatment of sold products",
        "overlap_estimate_pct": 5.0,
        "resolution": ResolutionAction.FLAG_ONLY,
    },
    {
        "rule_id": "DC-008",
        "type": DoubleCountType.USE_EOL_OVERLAP,
        "category_a": "cat_11_use_of_sold_products",
        "category_b": "cat_12_end_of_life_treatment",
        "description": "Energy used during product lifetime may include end-of-life energy",
        "overlap_estimate_pct": 3.0,
        "resolution": ResolutionAction.FLAG_ONLY,
    },
    {
        "rule_id": "DC-009",
        "type": DoubleCountType.COMMUTING_TRAVEL_OVERLAP,
        "category_a": "cat_07_employee_commuting",
        "category_b": "cat_06_business_travel",
        "description": "Business travel from home office may be counted as commuting",
        "overlap_estimate_pct": 2.0,
        "resolution": ResolutionAction.FLAG_ONLY,
    },
    {
        "rule_id": "DC-010",
        "type": DoubleCountType.SCOPE1_2_OVERLAP,
        "category_a": "cat_03_fuel_energy_related",
        "category_b": "scope_1_2",
        "description": "Cat 3 upstream fuel emissions overlap with Scope 1 direct combustion boundary",
        "overlap_estimate_pct": 0.0,
        "resolution": ResolutionAction.NO_ACTION,
    },
    {
        "rule_id": "DC-011",
        "type": DoubleCountType.UPSTREAM_DOWNSTREAM_OVERLAP,
        "category_a": "cat_04_upstream_transport",
        "category_b": "cat_01_purchased_goods_services",
        "description": "Transport costs embedded in goods purchase price",
        "overlap_estimate_pct": 5.0,
        "resolution": ResolutionAction.FLAG_ONLY,
    },
    {
        "rule_id": "DC-012",
        "type": DoubleCountType.SUPPLY_CHAIN_CIRCULAR,
        "category_a": "cat_01_purchased_goods_services",
        "category_b": "cat_10_processing_sold_products",
        "description": "Circular supply chain: sold product inputs purchased from downstream processor",
        "overlap_estimate_pct": 2.0,
        "resolution": ResolutionAction.FLAG_ONLY,
    },
]

# Category names
CATEGORY_NAMES: Dict[str, str] = {
    "cat_01_purchased_goods_services": "Purchased Goods & Services",
    "cat_02_capital_goods": "Capital Goods",
    "cat_03_fuel_energy_related": "Fuel- & Energy-Related Activities",
    "cat_04_upstream_transport": "Upstream Transportation & Distribution",
    "cat_05_waste_in_operations": "Waste Generated in Operations",
    "cat_06_business_travel": "Business Travel",
    "cat_07_employee_commuting": "Employee Commuting",
    "cat_08_upstream_leased_assets": "Upstream Leased Assets",
    "cat_09_downstream_transport": "Downstream Transportation & Distribution",
    "cat_10_processing_sold_products": "Processing of Sold Products",
    "cat_11_use_of_sold_products": "Use of Sold Products",
    "cat_12_end_of_life_treatment": "End-of-Life Treatment of Sold Products",
    "cat_13_downstream_leased_assets": "Downstream Leased Assets",
    "cat_14_franchises": "Franchises",
    "cat_15_investments": "Investments",
}

UPSTREAM_KEYS = {
    "cat_01_purchased_goods_services",
    "cat_02_capital_goods",
    "cat_03_fuel_energy_related",
    "cat_04_upstream_transport",
    "cat_05_waste_in_operations",
    "cat_06_business_travel",
    "cat_07_employee_commuting",
    "cat_08_upstream_leased_assets",
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ConsolidationWorkflow:
    """
    4-phase consolidation workflow for Scope 3 results.

    Aggregates per-category results into upstream/downstream subtotals, runs
    the 12-rule double-counting detection engine, integrates with Scope 1+2
    data for a full GHG footprint view, and produces a reconciled total with
    complete audit trail.

    Zero-hallucination: all aggregation and adjustment logic uses deterministic
    arithmetic. No LLM calls in numeric paths.

    Attributes:
        workflow_id: Unique execution identifier.
        _emissions_map: Category -> emission data lookup.
        _double_count_flags: Detected double-counting overlaps.
        _audit_trail: Audit trail entries.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = ConsolidationWorkflow()
        >>> emissions = [CategoryEmission(
        ...     category=Scope3Category.CAT_01_PURCHASED_GOODS,
        ...     emissions_tco2e=5000.0,
        ... )]
        >>> inp = ConsolidationInput(category_emissions=emissions)
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ConsolidationWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._emissions_map: Dict[str, CategoryEmission] = {}
        self._double_count_flags: List[DoubleCountFlag] = []
        self._audit_trail: List[AuditTrailEntry] = []
        self._phase_results: List[PhaseResult] = []
        self._upstream_total: float = 0.0
        self._downstream_total: float = 0.0
        self._scope3_total: float = 0.0
        self._scope3_adjusted: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[ConsolidationInput] = None,
        category_emissions: Optional[List[CategoryEmission]] = None,
        scope12_data: Optional[Scope12Data] = None,
    ) -> ConsolidationOutput:
        """
        Execute the 4-phase consolidation workflow.

        Args:
            input_data: Full input model (preferred).
            category_emissions: Category emission results (fallback).
            scope12_data: Scope 1+2 data for integration (fallback).

        Returns:
            ConsolidationOutput with reconciled totals and audit trail.
        """
        if input_data is None:
            input_data = ConsolidationInput(
                category_emissions=category_emissions or [],
                scope12_data=scope12_data,
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting consolidation workflow %s categories=%d",
            self.workflow_id, len(input_data.category_emissions),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._execute_with_retry(
                self._phase_category_aggregation, input_data, 1
            )
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 1 failed: {phase1.errors}")

            phase2 = await self._execute_with_retry(
                self._phase_double_counting_check, input_data, 2
            )
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 2 failed: {phase2.errors}")

            phase3 = await self._execute_with_retry(
                self._phase_scope_integration, input_data, 3
            )
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 3 failed: {phase3.errors}")

            phase4 = await self._execute_with_retry(
                self._phase_final_reconciliation, input_data, 4
            )
            self._phase_results.append(phase4)
            if phase4.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 4 failed: {phase4.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Consolidation workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        # Build category breakdown
        breakdown = []
        for ce in input_data.category_emissions:
            breakdown.append({
                "category": ce.category.value,
                "category_name": ce.category_name or CATEGORY_NAMES.get(ce.category.value, ""),
                "emissions_tco2e": ce.emissions_tco2e,
                "pct_of_scope3": round(
                    (ce.emissions_tco2e / self._scope3_total * 100.0)
                    if self._scope3_total > 0 else 0.0, 2
                ),
                "methodology_tier": ce.methodology_tier,
                "data_quality": ce.data_quality_score,
            })

        # Scope 1+2 values
        s1 = input_data.scope12_data.scope1_tco2e if input_data.scope12_data else 0.0
        s2_loc = input_data.scope12_data.scope2_location_tco2e if input_data.scope12_data else 0.0
        s2_mkt = input_data.scope12_data.scope2_market_tco2e if input_data.scope12_data else 0.0
        total_loc = s1 + s2_loc + self._scope3_adjusted
        total_mkt = s1 + s2_mkt + self._scope3_adjusted
        scope3_pct = (
            (self._scope3_adjusted / total_loc * 100.0) if total_loc > 0 else 0.0
        )

        result = ConsolidationOutput(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            scope3_total_tco2e=round(self._scope3_total, 2),
            scope3_upstream_tco2e=round(self._upstream_total, 2),
            scope3_downstream_tco2e=round(self._downstream_total, 2),
            scope3_adjusted_tco2e=round(self._scope3_adjusted, 2),
            category_breakdown=breakdown,
            double_count_flags=self._double_count_flags,
            total_adjustment_tco2e=round(
                self._scope3_total - self._scope3_adjusted, 2
            ),
            scope1_tco2e=s1,
            scope2_location_tco2e=s2_loc,
            scope2_market_tco2e=s2_mkt,
            total_footprint_location_tco2e=round(total_loc, 2),
            total_footprint_market_tco2e=round(total_mkt, 2),
            scope3_pct_of_total=round(scope3_pct, 1),
            audit_trail=self._audit_trail,
            progress_pct=100.0,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Consolidation workflow %s completed in %.2fs "
            "scope3=%.1f adjusted=%.1f footprint_loc=%.1f scope3_pct=%.1f%%",
            self.workflow_id, elapsed, self._scope3_total,
            self._scope3_adjusted, total_loc, scope3_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: ConsolidationInput, phase_number: int,
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
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Category Aggregation
    # -------------------------------------------------------------------------

    async def _phase_category_aggregation(
        self, input_data: ConsolidationInput
    ) -> PhaseResult:
        """Sum per-category results into upstream/downstream subtotals."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._emissions_map = {
            ce.category.value: ce for ce in input_data.category_emissions
        }

        self._upstream_total = 0.0
        self._downstream_total = 0.0

        for ce in input_data.category_emissions:
            if ce.category.value in UPSTREAM_KEYS:
                self._upstream_total += ce.emissions_tco2e
            else:
                self._downstream_total += ce.emissions_tco2e

        self._scope3_total = self._upstream_total + self._downstream_total
        self._scope3_adjusted = self._scope3_total

        if self._scope3_total <= 0:
            warnings.append("Total Scope 3 emissions are zero; verify input data")

        # Weighted average data quality
        total_weighted_dq = sum(
            ce.emissions_tco2e * ce.data_quality_score
            for ce in input_data.category_emissions
        )
        avg_dq = (
            total_weighted_dq / self._scope3_total
            if self._scope3_total > 0 else 1.0
        )

        self._audit_trail.append(AuditTrailEntry(
            action="category_aggregation",
            description=(
                f"Aggregated {len(input_data.category_emissions)} categories: "
                f"upstream={self._upstream_total:.2f} + "
                f"downstream={self._downstream_total:.2f} = "
                f"total={self._scope3_total:.2f} tCO2e"
            ),
            after_value=self._scope3_total,
            provenance_hash=self._hash_dict({"total": self._scope3_total}),
        ))

        outputs["upstream_tco2e"] = round(self._upstream_total, 2)
        outputs["downstream_tco2e"] = round(self._downstream_total, 2)
        outputs["total_tco2e"] = round(self._scope3_total, 2)
        outputs["upstream_pct"] = round(
            (self._upstream_total / self._scope3_total * 100.0)
            if self._scope3_total > 0 else 0.0, 1
        )
        outputs["downstream_pct"] = round(
            (self._downstream_total / self._scope3_total * 100.0)
            if self._scope3_total > 0 else 0.0, 1
        )
        outputs["categories_counted"] = len(input_data.category_emissions)
        outputs["weighted_avg_data_quality"] = round(avg_dq, 2)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 CategoryAggregation: total=%.1f upstream=%.1f downstream=%.1f",
            self._scope3_total, self._upstream_total, self._downstream_total,
        )
        return PhaseResult(
            phase_name="category_aggregation", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Double-Counting Check
    # -------------------------------------------------------------------------

    async def _phase_double_counting_check(
        self, input_data: ConsolidationInput
    ) -> PhaseResult:
        """Run 12-rule double-counting detection engine."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._double_count_flags = []
        total_adjustment = 0.0

        for rule in DOUBLE_COUNTING_RULES:
            cat_a_key = rule["category_a"]
            cat_b_key = rule["category_b"]

            # Get emissions for both categories
            ce_a = self._emissions_map.get(cat_a_key)
            ce_b = self._emissions_map.get(cat_b_key)

            # Skip if either category has no data
            if not ce_a and cat_a_key != "scope_1_2":
                continue
            if not ce_b and cat_b_key != "scope_1_2":
                continue

            emissions_a = ce_a.emissions_tco2e if ce_a else 0.0
            emissions_b = ce_b.emissions_tco2e if ce_b else 0.0

            # Skip if both are zero
            if emissions_a <= 0 and emissions_b <= 0:
                continue

            overlap_pct = rule["overlap_estimate_pct"]
            smaller = min(emissions_a, emissions_b) if emissions_a > 0 and emissions_b > 0 else 0.0
            estimated_overlap = smaller * (overlap_pct / 100.0)

            resolution = ResolutionAction(rule["resolution"])
            adjustment = 0.0

            if resolution == ResolutionAction.DEDUCT and input_data.enable_double_counting_resolution:
                adjustment = -estimated_overlap
                total_adjustment += adjustment

            flag = DoubleCountFlag(
                overlap_type=DoubleCountType(rule["type"]),
                category_a=cat_a_key,
                category_b=cat_b_key,
                estimated_overlap_tco2e=round(estimated_overlap, 2),
                overlap_pct=overlap_pct,
                resolution_action=resolution,
                adjustment_tco2e=round(adjustment, 2),
                description=rule["description"],
                rule_id=rule["rule_id"],
            )
            self._double_count_flags.append(flag)

            if estimated_overlap > 0:
                self.logger.info(
                    "Double-counting %s: %s vs %s overlap=%.2f tCO2e resolution=%s",
                    rule["rule_id"], cat_a_key, cat_b_key,
                    estimated_overlap, resolution.value,
                )

        # Apply adjustments
        self._scope3_adjusted = self._scope3_total + total_adjustment

        if total_adjustment < 0:
            self._audit_trail.append(AuditTrailEntry(
                action="double_counting_adjustment",
                description=(
                    f"Applied {len(self._double_count_flags)} double-counting rules; "
                    f"total adjustment: {total_adjustment:.2f} tCO2e"
                ),
                before_value=self._scope3_total,
                after_value=self._scope3_adjusted,
                provenance_hash=self._hash_dict({"adjustment": total_adjustment}),
            ))

        flags_with_overlap = [
            f for f in self._double_count_flags if f.estimated_overlap_tco2e > 0
        ]
        flags_deducted = [
            f for f in self._double_count_flags
            if f.resolution_action == ResolutionAction.DEDUCT and f.adjustment_tco2e != 0
        ]

        outputs["rules_evaluated"] = len(DOUBLE_COUNTING_RULES)
        outputs["overlaps_detected"] = len(flags_with_overlap)
        outputs["deductions_applied"] = len(flags_deducted)
        outputs["total_adjustment_tco2e"] = round(total_adjustment, 2)
        outputs["scope3_before"] = round(self._scope3_total, 2)
        outputs["scope3_after"] = round(self._scope3_adjusted, 2)
        outputs["adjustment_pct"] = round(
            abs(total_adjustment) / self._scope3_total * 100.0
            if self._scope3_total > 0 else 0.0, 2
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 DoubleCountingCheck: %d overlaps, adjustment=%.1f tCO2e (%.1f%%)",
            len(flags_with_overlap), total_adjustment,
            outputs["adjustment_pct"],
        )
        return PhaseResult(
            phase_name="double_counting_check", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Scope Integration
    # -------------------------------------------------------------------------

    async def _phase_scope_integration(
        self, input_data: ConsolidationInput
    ) -> PhaseResult:
        """Integrate with Scope 1+2 for full GHG footprint view."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        has_scope12 = input_data.scope12_data is not None

        if has_scope12:
            s12 = input_data.scope12_data
            s1 = s12.scope1_tco2e
            s2_loc = s12.scope2_location_tco2e
            s2_mkt = s12.scope2_market_tco2e

            total_loc = s1 + s2_loc + self._scope3_adjusted
            total_mkt = s1 + s2_mkt + self._scope3_adjusted

            scope3_pct_loc = (
                (self._scope3_adjusted / total_loc * 100.0)
                if total_loc > 0 else 0.0
            )
            scope1_pct = (s1 / total_loc * 100.0) if total_loc > 0 else 0.0
            scope2_pct = (s2_loc / total_loc * 100.0) if total_loc > 0 else 0.0

            outputs["scope1_tco2e"] = s1
            outputs["scope2_location_tco2e"] = s2_loc
            outputs["scope2_market_tco2e"] = s2_mkt
            outputs["scope3_adjusted_tco2e"] = round(self._scope3_adjusted, 2)
            outputs["total_footprint_location_tco2e"] = round(total_loc, 2)
            outputs["total_footprint_market_tco2e"] = round(total_mkt, 2)
            outputs["scope1_pct"] = round(scope1_pct, 1)
            outputs["scope2_pct"] = round(scope2_pct, 1)
            outputs["scope3_pct"] = round(scope3_pct_loc, 1)
            outputs["integration_source"] = s12.source_pack

            self._audit_trail.append(AuditTrailEntry(
                action="scope_integration",
                description=(
                    f"Integrated Scope 1 ({s1:.1f}) + Scope 2 loc ({s2_loc:.1f}) "
                    f"+ Scope 3 ({self._scope3_adjusted:.1f}) = "
                    f"{total_loc:.1f} tCO2e total footprint"
                ),
                after_value=total_loc,
            ))
        else:
            warnings.append(
                "No Scope 1+2 data provided; Scope 3 reported standalone"
            )
            outputs["scope3_adjusted_tco2e"] = round(self._scope3_adjusted, 2)
            outputs["integration_source"] = "none"
            outputs["scope3_pct"] = 100.0

        outputs["has_scope12_data"] = has_scope12

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ScopeIntegration: has_scope12=%s", has_scope12,
        )
        return PhaseResult(
            phase_name="scope_integration", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Final Reconciliation
    # -------------------------------------------------------------------------

    async def _phase_final_reconciliation(
        self, input_data: ConsolidationInput
    ) -> PhaseResult:
        """Produce reconciled total with per-category breakdown and audit trail."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Final provenance chain
        self._audit_trail.append(AuditTrailEntry(
            action="final_reconciliation",
            description=(
                f"Final reconciled Scope 3 total: {self._scope3_adjusted:.2f} tCO2e "
                f"({len(input_data.category_emissions)} categories, "
                f"{len(self._double_count_flags)} DC rules applied)"
            ),
            before_value=self._scope3_total,
            after_value=self._scope3_adjusted,
            provenance_hash=self._hash_dict({
                "total": self._scope3_adjusted,
                "categories": len(input_data.category_emissions),
            }),
        ))

        # Weighted uncertainty
        total_variance = 0.0
        for ce in input_data.category_emissions:
            uncertainty_fraction = ce.uncertainty_pct / 100.0
            total_variance += (ce.emissions_tco2e * uncertainty_fraction) ** 2

        combined_uncertainty_tco2e = total_variance ** 0.5
        combined_uncertainty_pct = (
            (combined_uncertainty_tco2e / self._scope3_adjusted * 100.0)
            if self._scope3_adjusted > 0 else 0.0
        )

        outputs["reconciled_scope3_tco2e"] = round(self._scope3_adjusted, 2)
        outputs["combined_uncertainty_pct"] = round(combined_uncertainty_pct, 1)
        outputs["uncertainty_lower_tco2e"] = round(
            self._scope3_adjusted - combined_uncertainty_tco2e, 2
        )
        outputs["uncertainty_upper_tco2e"] = round(
            self._scope3_adjusted + combined_uncertainty_tco2e, 2
        )
        outputs["audit_trail_entries"] = len(self._audit_trail)
        outputs["reporting_year"] = input_data.reporting_year

        # Category ranking
        sorted_cats = sorted(
            input_data.category_emissions,
            key=lambda x: x.emissions_tco2e,
            reverse=True,
        )
        outputs["top_3_categories"] = [
            {
                "category": ce.category.value,
                "name": ce.category_name or CATEGORY_NAMES.get(ce.category.value, ""),
                "tco2e": ce.emissions_tco2e,
            }
            for ce in sorted_cats[:3]
        ]

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 FinalReconciliation: reconciled=%.1f +/- %.1f%% tCO2e",
            self._scope3_adjusted, combined_uncertainty_pct,
        )
        return PhaseResult(
            phase_name="final_reconciliation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self._emissions_map = {}
        self._double_count_flags = []
        self._audit_trail = []
        self._phase_results = []
        self._upstream_total = 0.0
        self._downstream_total = 0.0
        self._scope3_total = 0.0
        self._scope3_adjusted = 0.0

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: ConsolidationOutput) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += f"|{result.workflow_id}|{result.scope3_adjusted_tco2e}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
