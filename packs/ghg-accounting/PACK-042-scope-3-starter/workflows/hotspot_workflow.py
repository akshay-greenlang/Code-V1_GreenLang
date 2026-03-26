# -*- coding: utf-8 -*-
"""
Hotspot Workflow
=====================

4-phase workflow for identifying emission hotspots and prioritizing reduction
opportunities within PACK-042 Scope 3 Starter Pack.

Phases:
    1. ParetoAnalysis          -- Rank categories by emission contribution,
                                  identify top categories driving 80% of total
    2. MaterialityAssessment   -- Score each hotspot on magnitude (tCO2e),
                                  data quality (DQR), and reduction potential
    3. Benchmarking            -- Compare against sector averages from CDP/
                                  industry databases
    4. ActionPlanning          -- Generate prioritized reduction roadmap with
                                  ROI estimates for tier upgrades and supplier
                                  engagement

The workflow follows GreenLang zero-hallucination principles: all scoring,
ranking, and benchmark comparisons use deterministic formulas. SHA-256
provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Value Chain (Scope 3) Standard -- Chapter 10
    SBTi Corporate Net-Zero Standard v1.1 (Scope 3 target setting)
    CDP Climate Change Questionnaire (C6.7 Scope 3 emissions breakdown)

Schedule: on-demand (after consolidation)
Estimated duration: 2-4 hours

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


class HotspotPriority(str, Enum):
    """Priority level for a hotspot."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionType(str, Enum):
    """Types of reduction actions."""

    TIER_UPGRADE = "tier_upgrade"
    SUPPLIER_ENGAGEMENT = "supplier_engagement"
    PRODUCT_REDESIGN = "product_redesign"
    OPERATIONAL_CHANGE = "operational_change"
    MODAL_SHIFT = "modal_shift"
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_PROCUREMENT = "renewable_procurement"
    CIRCULAR_ECONOMY = "circular_economy"
    POLICY_CHANGE = "policy_change"


class BenchmarkPosition(str, Enum):
    """Position relative to sector benchmark."""

    LEADER = "leader"
    ABOVE_AVERAGE = "above_average"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    LAGGARD = "laggard"
    NO_DATA = "no_data"


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


class CategoryResult(BaseModel):
    """Input category emission result for hotspot analysis."""

    category: Scope3Category = Field(...)
    category_name: str = Field(default="")
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    methodology_tier: str = Field(default="spend_based")
    data_quality_score: float = Field(default=1.0, ge=1.0, le=5.0)
    uncertainty_pct: float = Field(default=50.0, ge=0.0, le=100.0)


class ParetoEntry(BaseModel):
    """Pareto analysis entry for a category."""

    category: Scope3Category = Field(...)
    category_name: str = Field(default="")
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    cumulative_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    rank: int = Field(default=0, ge=0)
    in_pareto_80: bool = Field(default=False)


class MaterialityScore(BaseModel):
    """Materiality scoring for a hotspot category."""

    category: Scope3Category = Field(...)
    magnitude_score: float = Field(
        default=0.0, ge=0.0, le=10.0,
        description="Score based on absolute tCO2e (0-10)",
    )
    data_quality_score: float = Field(
        default=0.0, ge=0.0, le=10.0,
        description="Score based on DQR (0-10; higher = better data)",
    )
    reduction_potential_score: float = Field(
        default=0.0, ge=0.0, le=10.0,
        description="Score based on industry reduction benchmarks (0-10)",
    )
    composite_score: float = Field(
        default=0.0, ge=0.0, le=10.0,
        description="Weighted composite (0-10)",
    )
    priority: HotspotPriority = Field(default=HotspotPriority.LOW)


class BenchmarkEntry(BaseModel):
    """Sector benchmark comparison for a category."""

    category: Scope3Category = Field(...)
    company_intensity: float = Field(default=0.0, ge=0.0)
    sector_median_intensity: float = Field(default=0.0, ge=0.0)
    sector_p25_intensity: float = Field(default=0.0, ge=0.0)
    sector_p75_intensity: float = Field(default=0.0, ge=0.0)
    position: BenchmarkPosition = Field(default=BenchmarkPosition.NO_DATA)
    gap_to_median_pct: float = Field(default=0.0)


class ReductionAction(BaseModel):
    """Prioritized reduction action."""

    action_id: str = Field(
        default_factory=lambda: f"act-{uuid.uuid4().hex[:8]}"
    )
    category: Scope3Category = Field(...)
    action_type: ActionType = Field(...)
    title: str = Field(default="")
    description: str = Field(default="")
    estimated_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    estimated_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    estimated_roi_years: float = Field(default=0.0, ge=0.0)
    effort_level: str = Field(default="medium", description="low|medium|high")
    timeline_months: int = Field(default=12, ge=0)
    priority: HotspotPriority = Field(default=HotspotPriority.MEDIUM)


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class HotspotInput(BaseModel):
    """Input data model for HotspotWorkflow."""

    category_results: List[CategoryResult] = Field(
        default_factory=list, description="Per-category emission results"
    )
    total_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    sector: str = Field(default="", description="Organization sector")
    revenue_usd: float = Field(default=0.0, ge=0.0)
    employee_count: int = Field(default=0, ge=0)
    reporting_year: int = Field(default=2025)
    pareto_threshold_pct: float = Field(
        default=80.0, ge=50.0, le=95.0,
        description="Pareto threshold (default 80%)",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class HotspotOutput(BaseModel):
    """Complete result from hotspot workflow."""

    workflow_id: str = Field(...)
    workflow_name: str = Field(default="hotspot_analysis")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    pareto_entries: List[ParetoEntry] = Field(default_factory=list)
    pareto_80_count: int = Field(default=0, ge=0)
    materiality_scores: List[MaterialityScore] = Field(default_factory=list)
    benchmark_entries: List[BenchmarkEntry] = Field(default_factory=list)
    reduction_actions: List[ReductionAction] = Field(default_factory=list)
    total_reduction_potential_tco2e: float = Field(default=0.0, ge=0.0)
    total_reduction_potential_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Sector benchmark intensities (tCO2e per M USD revenue per category)
# Source: CDP sector averages, publicly available
SECTOR_BENCHMARKS_PER_M_REV: Dict[str, Dict[str, Dict[str, float]]] = {
    "manufacturing": {
        "cat_01_purchased_goods_services": {"p25": 200.0, "median": 450.0, "p75": 800.0},
        "cat_02_capital_goods": {"p25": 20.0, "median": 50.0, "p75": 100.0},
        "cat_04_upstream_transport": {"p25": 30.0, "median": 80.0, "p75": 150.0},
        "cat_06_business_travel": {"p25": 5.0, "median": 15.0, "p75": 30.0},
        "cat_07_employee_commuting": {"p25": 8.0, "median": 20.0, "p75": 40.0},
        "cat_11_use_of_sold_products": {"p25": 50.0, "median": 150.0, "p75": 400.0},
    },
    "services": {
        "cat_01_purchased_goods_services": {"p25": 30.0, "median": 100.0, "p75": 250.0},
        "cat_06_business_travel": {"p25": 10.0, "median": 40.0, "p75": 80.0},
        "cat_07_employee_commuting": {"p25": 10.0, "median": 30.0, "p75": 60.0},
    },
    "default": {
        "cat_01_purchased_goods_services": {"p25": 80.0, "median": 200.0, "p75": 500.0},
        "cat_06_business_travel": {"p25": 8.0, "median": 25.0, "p75": 50.0},
        "cat_07_employee_commuting": {"p25": 8.0, "median": 25.0, "p75": 50.0},
    },
}

# Reduction potential estimates by category (% achievable in 3-5 years)
REDUCTION_POTENTIAL: Dict[str, Dict[str, Any]] = {
    "cat_01_purchased_goods_services": {
        "tier_upgrade_pct": 20.0,
        "supplier_engagement_pct": 15.0,
        "actions": [
            {"type": "supplier_engagement", "title": "Top Supplier Carbon Program",
             "reduction_pct": 15.0, "cost_factor": 50000.0, "timeline_months": 18, "effort": "high"},
            {"type": "tier_upgrade", "title": "Upgrade to Average-Data Methodology",
             "reduction_pct": 0.0, "cost_factor": 20000.0, "timeline_months": 6, "effort": "medium"},
        ],
    },
    "cat_04_upstream_transport": {
        "tier_upgrade_pct": 15.0,
        "supplier_engagement_pct": 10.0,
        "actions": [
            {"type": "modal_shift", "title": "Shift Road to Rail/Sea",
             "reduction_pct": 25.0, "cost_factor": 30000.0, "timeline_months": 12, "effort": "high"},
            {"type": "operational_change", "title": "Optimize Logistics Routes",
             "reduction_pct": 10.0, "cost_factor": 15000.0, "timeline_months": 6, "effort": "medium"},
        ],
    },
    "cat_06_business_travel": {
        "tier_upgrade_pct": 10.0,
        "supplier_engagement_pct": 5.0,
        "actions": [
            {"type": "policy_change", "title": "Travel Policy: Virtual-First",
             "reduction_pct": 30.0, "cost_factor": 5000.0, "timeline_months": 3, "effort": "low"},
            {"type": "operational_change", "title": "Rail over Short-Haul Flights",
             "reduction_pct": 15.0, "cost_factor": 2000.0, "timeline_months": 3, "effort": "low"},
        ],
    },
    "cat_07_employee_commuting": {
        "tier_upgrade_pct": 10.0,
        "supplier_engagement_pct": 0.0,
        "actions": [
            {"type": "policy_change", "title": "Hybrid/Remote Work Policy",
             "reduction_pct": 25.0, "cost_factor": 3000.0, "timeline_months": 3, "effort": "low"},
            {"type": "operational_change", "title": "EV Charging + Bike Infrastructure",
             "reduction_pct": 10.0, "cost_factor": 50000.0, "timeline_months": 12, "effort": "medium"},
        ],
    },
    "cat_11_use_of_sold_products": {
        "tier_upgrade_pct": 15.0,
        "supplier_engagement_pct": 0.0,
        "actions": [
            {"type": "product_redesign", "title": "Energy Efficiency Redesign",
             "reduction_pct": 20.0, "cost_factor": 100000.0, "timeline_months": 24, "effort": "high"},
            {"type": "circular_economy", "title": "Product-as-a-Service Model",
             "reduction_pct": 15.0, "cost_factor": 75000.0, "timeline_months": 18, "effort": "high"},
        ],
    },
    "default": {
        "tier_upgrade_pct": 10.0,
        "supplier_engagement_pct": 5.0,
        "actions": [
            {"type": "tier_upgrade", "title": "Improve Data Quality & Methodology",
             "reduction_pct": 0.0, "cost_factor": 15000.0, "timeline_months": 6, "effort": "medium"},
            {"type": "supplier_engagement", "title": "Supplier Carbon Assessment",
             "reduction_pct": 10.0, "cost_factor": 25000.0, "timeline_months": 12, "effort": "medium"},
        ],
    },
}

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


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class HotspotWorkflow:
    """
    4-phase hotspot analysis workflow for Scope 3 emissions.

    Performs Pareto analysis to identify the categories driving 80% of
    emissions, scores each on materiality (magnitude, data quality, reduction
    potential), benchmarks against sector averages, and generates a prioritized
    reduction action roadmap.

    Zero-hallucination: all scoring and ranking uses deterministic formulas
    and published benchmark data. No LLM calls in numeric paths.

    Example:
        >>> wf = HotspotWorkflow()
        >>> inp = HotspotInput(category_results=[...], total_scope3_tco2e=10000.0)
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    # Materiality weight factors
    WEIGHT_MAGNITUDE: float = 0.5
    WEIGHT_DATA_QUALITY: float = 0.2
    WEIGHT_REDUCTION_POTENTIAL: float = 0.3

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize HotspotWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._pareto_entries: List[ParetoEntry] = []
        self._materiality_scores: List[MaterialityScore] = []
        self._benchmarks: List[BenchmarkEntry] = []
        self._actions: List[ReductionAction] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[HotspotInput] = None,
        category_results: Optional[List[CategoryResult]] = None,
    ) -> HotspotOutput:
        """
        Execute the 4-phase hotspot analysis workflow.

        Args:
            input_data: Full input model (preferred).
            category_results: Category emission results (fallback).

        Returns:
            HotspotOutput with Pareto analysis, materiality, benchmarks, actions.
        """
        if input_data is None:
            results = category_results or []
            total = sum(cr.emissions_tco2e for cr in results)
            input_data = HotspotInput(
                category_results=results,
                total_scope3_tco2e=total,
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting hotspot workflow %s categories=%d total=%.1f",
            self.workflow_id, len(input_data.category_results),
            input_data.total_scope3_tco2e,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        try:
            for phase_num, phase_fn in enumerate(
                [
                    self._phase_pareto_analysis,
                    self._phase_materiality_assessment,
                    self._phase_benchmarking,
                    self._phase_action_planning,
                ],
                start=1,
            ):
                phase = await self._execute_with_retry(phase_fn, input_data, phase_num)
                self._phase_results.append(phase)
                if phase.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {phase_num} failed: {phase.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Hotspot workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        total_reduction = sum(a.estimated_reduction_tco2e for a in self._actions)
        total_reduction_pct = (
            (total_reduction / input_data.total_scope3_tco2e * 100.0)
            if input_data.total_scope3_tco2e > 0 else 0.0
        )

        result = HotspotOutput(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            pareto_entries=self._pareto_entries,
            pareto_80_count=sum(1 for p in self._pareto_entries if p.in_pareto_80),
            materiality_scores=self._materiality_scores,
            benchmark_entries=self._benchmarks,
            reduction_actions=self._actions,
            total_reduction_potential_tco2e=round(total_reduction, 2),
            total_reduction_potential_pct=round(total_reduction_pct, 1),
            progress_pct=100.0,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Hotspot workflow %s completed in %.2fs pareto_80=%d "
            "actions=%d reduction_potential=%.1f%%",
            self.workflow_id, elapsed,
            result.pareto_80_count, len(self._actions),
            total_reduction_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: HotspotInput, phase_number: int,
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
            phase_number=phase_number, status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Pareto Analysis
    # -------------------------------------------------------------------------

    async def _phase_pareto_analysis(self, input_data: HotspotInput) -> PhaseResult:
        """Rank categories and identify top 80% contributors."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total = input_data.total_scope3_tco2e
        if total <= 0:
            total = sum(cr.emissions_tco2e for cr in input_data.category_results)

        sorted_cats = sorted(
            input_data.category_results,
            key=lambda x: x.emissions_tco2e,
            reverse=True,
        )

        self._pareto_entries = []
        cumulative = 0.0
        threshold_reached = False

        for rank, cr in enumerate(sorted_cats, start=1):
            pct = (cr.emissions_tco2e / total * 100.0) if total > 0 else 0.0
            cumulative += pct
            in_pareto = not threshold_reached

            self._pareto_entries.append(ParetoEntry(
                category=cr.category,
                category_name=cr.category_name or CATEGORY_NAMES.get(cr.category.value, ""),
                emissions_tco2e=cr.emissions_tco2e,
                pct_of_total=round(pct, 2),
                cumulative_pct=round(cumulative, 2),
                rank=rank,
                in_pareto_80=in_pareto,
            ))

            if cumulative >= input_data.pareto_threshold_pct and not threshold_reached:
                threshold_reached = True

        pareto_count = sum(1 for p in self._pareto_entries if p.in_pareto_80)

        outputs["total_tco2e"] = round(total, 2)
        outputs["pareto_threshold_pct"] = input_data.pareto_threshold_pct
        outputs["pareto_count"] = pareto_count
        outputs["total_categories"] = len(self._pareto_entries)
        outputs["top_3"] = [
            {"category": p.category_name, "tco2e": p.emissions_tco2e, "pct": p.pct_of_total}
            for p in self._pareto_entries[:3]
        ]

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 ParetoAnalysis: %d/%d categories in Pareto 80%%",
            pareto_count, len(self._pareto_entries),
        )
        return PhaseResult(
            phase_name="pareto_analysis", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Materiality Assessment
    # -------------------------------------------------------------------------

    async def _phase_materiality_assessment(self, input_data: HotspotInput) -> PhaseResult:
        """Score hotspots on magnitude, data quality, and reduction potential."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        max_emissions = max(
            (cr.emissions_tco2e for cr in input_data.category_results), default=1.0
        )

        self._materiality_scores = []

        for cr in input_data.category_results:
            # Magnitude score (0-10): linear scale relative to max
            magnitude = (cr.emissions_tco2e / max_emissions * 10.0) if max_emissions > 0 else 0.0

            # Data quality score (0-10): higher DQR = better data = higher score
            dq = cr.data_quality_score * 2.0  # DQR 1-5 -> 2-10

            # Reduction potential score (0-10): from reference data
            cat_potential = REDUCTION_POTENTIAL.get(
                cr.category.value, REDUCTION_POTENTIAL["default"]
            )
            reduction_pct = cat_potential.get("tier_upgrade_pct", 0.0) + \
                cat_potential.get("supplier_engagement_pct", 0.0)
            reduction_score = min(reduction_pct / 5.0, 10.0)  # 50% -> 10.0

            # Composite score
            composite = (
                self.WEIGHT_MAGNITUDE * magnitude
                + self.WEIGHT_DATA_QUALITY * dq
                + self.WEIGHT_REDUCTION_POTENTIAL * reduction_score
            )

            # Priority classification
            if composite >= 7.0:
                priority = HotspotPriority.CRITICAL
            elif composite >= 5.0:
                priority = HotspotPriority.HIGH
            elif composite >= 3.0:
                priority = HotspotPriority.MEDIUM
            else:
                priority = HotspotPriority.LOW

            self._materiality_scores.append(MaterialityScore(
                category=cr.category,
                magnitude_score=round(magnitude, 2),
                data_quality_score=round(dq, 2),
                reduction_potential_score=round(reduction_score, 2),
                composite_score=round(composite, 2),
                priority=priority,
            ))

        # Sort by composite score descending
        self._materiality_scores.sort(key=lambda x: x.composite_score, reverse=True)

        critical_count = sum(1 for m in self._materiality_scores if m.priority == HotspotPriority.CRITICAL)
        high_count = sum(1 for m in self._materiality_scores if m.priority == HotspotPriority.HIGH)

        outputs["critical_hotspots"] = critical_count
        outputs["high_hotspots"] = high_count
        outputs["total_scored"] = len(self._materiality_scores)
        outputs["top_3_by_composite"] = [
            {"category": m.category.value, "composite": m.composite_score, "priority": m.priority.value}
            for m in self._materiality_scores[:3]
        ]

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 MaterialityAssessment: critical=%d high=%d",
            critical_count, high_count,
        )
        return PhaseResult(
            phase_name="materiality_assessment", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Benchmarking
    # -------------------------------------------------------------------------

    async def _phase_benchmarking(self, input_data: HotspotInput) -> PhaseResult:
        """Compare against sector averages from CDP/industry databases."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        sector_key = self._normalize_sector(input_data.sector)
        sector_data = SECTOR_BENCHMARKS_PER_M_REV.get(
            sector_key, SECTOR_BENCHMARKS_PER_M_REV["default"]
        )
        revenue_m = input_data.revenue_usd / 1_000_000.0 if input_data.revenue_usd > 0 else 1.0

        self._benchmarks = []

        for cr in input_data.category_results:
            cat_key = cr.category.value
            benchmark = sector_data.get(cat_key)

            intensity = cr.emissions_tco2e / revenue_m if revenue_m > 0 else 0.0

            if not benchmark:
                self._benchmarks.append(BenchmarkEntry(
                    category=cr.category,
                    company_intensity=round(intensity, 2),
                    position=BenchmarkPosition.NO_DATA,
                ))
                continue

            median = benchmark["median"]
            p25 = benchmark["p25"]
            p75 = benchmark["p75"]

            if intensity <= p25:
                position = BenchmarkPosition.LEADER
            elif intensity <= median:
                position = BenchmarkPosition.ABOVE_AVERAGE
            elif intensity <= p75:
                position = BenchmarkPosition.BELOW_AVERAGE
            else:
                position = BenchmarkPosition.LAGGARD

            gap = ((intensity - median) / median * 100.0) if median > 0 else 0.0

            self._benchmarks.append(BenchmarkEntry(
                category=cr.category,
                company_intensity=round(intensity, 2),
                sector_median_intensity=round(median, 2),
                sector_p25_intensity=round(p25, 2),
                sector_p75_intensity=round(p75, 2),
                position=position,
                gap_to_median_pct=round(gap, 1),
            ))

        laggards = sum(1 for b in self._benchmarks if b.position == BenchmarkPosition.LAGGARD)
        leaders = sum(1 for b in self._benchmarks if b.position == BenchmarkPosition.LEADER)

        outputs["benchmarks_compared"] = sum(
            1 for b in self._benchmarks if b.position != BenchmarkPosition.NO_DATA
        )
        outputs["leaders"] = leaders
        outputs["laggards"] = laggards
        outputs["sector_key"] = sector_key

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 Benchmarking: %d compared, leaders=%d laggards=%d",
            outputs["benchmarks_compared"], leaders, laggards,
        )
        return PhaseResult(
            phase_name="benchmarking", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Action Planning
    # -------------------------------------------------------------------------

    async def _phase_action_planning(self, input_data: HotspotInput) -> PhaseResult:
        """Generate prioritized reduction roadmap with ROI estimates."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._actions = []

        # Generate actions for Pareto-80 categories
        pareto_cats = {p.category for p in self._pareto_entries if p.in_pareto_80}

        for cr in input_data.category_results:
            if cr.category not in pareto_cats:
                continue

            cat_key = cr.category.value
            potential = REDUCTION_POTENTIAL.get(cat_key, REDUCTION_POTENTIAL["default"])
            cat_materiality = next(
                (m for m in self._materiality_scores if m.category == cr.category),
                None,
            )
            priority = cat_materiality.priority if cat_materiality else HotspotPriority.MEDIUM

            for action_def in potential.get("actions", []):
                reduction_pct = action_def["reduction_pct"]
                reduction_tco2e = cr.emissions_tco2e * (reduction_pct / 100.0)

                # Estimate ROI: cost / (reduction * carbon_price_per_tonne)
                cost = action_def["cost_factor"]
                carbon_price = 50.0  # USD/tCO2e conservative estimate
                annual_savings = reduction_tco2e * carbon_price
                roi_years = cost / annual_savings if annual_savings > 0 else 99.0

                self._actions.append(ReductionAction(
                    category=cr.category,
                    action_type=ActionType(action_def["type"]),
                    title=action_def["title"],
                    description=(
                        f"{action_def['title']} for "
                        f"{CATEGORY_NAMES.get(cat_key, cat_key)}"
                    ),
                    estimated_reduction_tco2e=round(reduction_tco2e, 2),
                    estimated_reduction_pct=round(reduction_pct, 1),
                    estimated_cost_usd=cost,
                    estimated_roi_years=round(roi_years, 1),
                    effort_level=action_def.get("effort", "medium"),
                    timeline_months=action_def.get("timeline_months", 12),
                    priority=priority,
                ))

        # Sort by priority then ROI
        priority_order = {
            HotspotPriority.CRITICAL: 0,
            HotspotPriority.HIGH: 1,
            HotspotPriority.MEDIUM: 2,
            HotspotPriority.LOW: 3,
        }
        self._actions.sort(
            key=lambda a: (priority_order.get(a.priority, 3), a.estimated_roi_years)
        )

        total_reduction = sum(a.estimated_reduction_tco2e for a in self._actions)
        total_cost = sum(a.estimated_cost_usd for a in self._actions)

        outputs["total_actions"] = len(self._actions)
        outputs["total_reduction_tco2e"] = round(total_reduction, 2)
        outputs["total_cost_usd"] = round(total_cost, 2)
        outputs["quick_wins"] = sum(
            1 for a in self._actions if a.effort_level == "low" and a.timeline_months <= 6
        )
        outputs["top_3_actions"] = [
            {"title": a.title, "reduction": a.estimated_reduction_tco2e, "roi_years": a.estimated_roi_years}
            for a in self._actions[:3]
        ]

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ActionPlanning: %d actions, total reduction=%.1f tCO2e, cost=%.0f USD",
            len(self._actions), total_reduction, total_cost,
        )
        return PhaseResult(
            phase_name="action_planning", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _normalize_sector(self, sector: str) -> str:
        """Normalize sector string to benchmark key."""
        if not sector:
            return "default"
        sector_lower = sector.lower().strip()
        for key in ("manufacturing", "services", "retail", "finance"):
            if key in sector_lower:
                return key
        return "default"

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self._pareto_entries = []
        self._materiality_scores = []
        self._benchmarks = []
        self._actions = []
        self._phase_results = []

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: HotspotOutput) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += f"|{result.workflow_id}|{result.total_reduction_potential_tco2e}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
