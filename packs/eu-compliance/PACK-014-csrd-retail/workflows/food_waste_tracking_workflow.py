# -*- coding: utf-8 -*-
"""
Food Waste Tracking Workflow
=================================

4-phase workflow for food waste management and reporting within
PACK-014 CSRD Retail and Consumer Goods Pack.

Phases:
    1. WasteBaseline         -- Establish baseline from historical data
    2. CategoryAnalysis      -- Breakdown by food category and destination
    3. ReductionTargeting    -- Set targets per category, track progress
    4. ProgressReporting     -- Report vs EU 30% target (2030)

Author: GreenLang Team
Version: 14.0.0
"""

import hashlib
import json
import logging
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


class FoodCategory(str, Enum):
    """Food waste categories."""
    BAKERY = "bakery"
    PRODUCE = "produce"
    DAIRY = "dairy"
    MEAT = "meat"
    SEAFOOD = "seafood"
    DELI = "deli"
    FROZEN = "frozen"
    DRY_GOODS = "dry_goods"
    BEVERAGES = "beverages"
    PREPARED_FOODS = "prepared_foods"
    OTHER = "other"


class WasteDestination(str, Enum):
    """Food waste destination in the waste hierarchy."""
    PREVENTION = "prevention"
    REDISTRIBUTION = "redistribution"
    ANIMAL_FEED = "animal_feed"
    COMPOSTING = "composting"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"
    INCINERATION_ENERGY = "incineration_energy"
    LANDFILL = "landfill"
    OTHER = "other"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class WasteRecord(BaseModel):
    """Individual food waste record."""
    record_id: str = Field(default_factory=lambda: f"fw-{uuid.uuid4().hex[:8]}")
    store_id: str = Field(default="")
    period: str = Field(default="", description="YYYY-MM or YYYY-Q#")
    food_category: FoodCategory = Field(default=FoodCategory.OTHER)
    weight_kg: float = Field(default=0.0, ge=0.0, description="Waste weight in kg")
    destination: WasteDestination = Field(default=WasteDestination.LANDFILL)
    value_eur: float = Field(default=0.0, ge=0.0, description="Cost value of wasted food")
    reason: str = Field(default="", description="expired|damaged|overstock|cosmetic|other")


class BaselineData(BaseModel):
    """Historical baseline data for food waste."""
    baseline_year: int = Field(default=2020, ge=2015, le=2050)
    total_waste_tonnes: float = Field(default=0.0, ge=0.0)
    total_food_handled_tonnes: float = Field(default=0.0, ge=0.0)
    waste_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    category_baselines: Dict[str, float] = Field(default_factory=dict)
    revenue_eur: float = Field(default=0.0, ge=0.0)


class CategoryBreakdown(BaseModel):
    """Waste breakdown for a single food category."""
    category: str = Field(...)
    waste_kg: float = Field(default=0.0, ge=0.0)
    share_of_total_pct: float = Field(default=0.0)
    value_lost_eur: float = Field(default=0.0, ge=0.0)
    destination_breakdown: Dict[str, float] = Field(default_factory=dict)
    hierarchy_score: float = Field(default=0.0, ge=0.0, le=100.0)


class ReductionTarget(BaseModel):
    """Reduction target for a food category."""
    category: str = Field(...)
    baseline_kg: float = Field(default=0.0, ge=0.0)
    current_kg: float = Field(default=0.0, ge=0.0)
    target_kg: float = Field(default=0.0, ge=0.0)
    reduction_pct: float = Field(default=0.0)
    on_track: bool = Field(default=False)
    gap_kg: float = Field(default=0.0)


class WasteRecommendation(BaseModel):
    """Recommendation for waste reduction."""
    recommendation_id: str = Field(default_factory=lambda: f"rec-{uuid.uuid4().hex[:6]}")
    category: str = Field(default="")
    action: str = Field(default="")
    expected_reduction_kg: float = Field(default=0.0, ge=0.0)
    expected_savings_eur: float = Field(default=0.0, ge=0.0)
    priority: str = Field(default="medium", description="high|medium|low")
    implementation_months: int = Field(default=6, ge=1)


class FoodWasteInput(BaseModel):
    """Input data model for FoodWasteTrackingWorkflow."""
    waste_records: List[WasteRecord] = Field(default_factory=list)
    baseline_data: Optional[BaselineData] = Field(None)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class FoodWasteResult(BaseModel):
    """Complete result from food waste tracking workflow."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="food_waste_tracking")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    total_waste_tonnes: float = Field(default=0.0, ge=0.0)
    total_value_lost_eur: float = Field(default=0.0, ge=0.0)
    reduction_progress: Dict[str, Any] = Field(default_factory=dict)
    hierarchy_score: float = Field(default=0.0, ge=0.0, le=100.0)
    category_breakdown: List[CategoryBreakdown] = Field(default_factory=list)
    reduction_targets: List[ReductionTarget] = Field(default_factory=list)
    recommendations: List[WasteRecommendation] = Field(default_factory=list)
    eu_2030_target_status: str = Field(default="not_assessed")
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# WASTE HIERARCHY SCORES
# =============================================================================

HIERARCHY_WEIGHTS: Dict[str, float] = {
    "prevention": 100.0,
    "redistribution": 85.0,
    "animal_feed": 70.0,
    "composting": 50.0,
    "anaerobic_digestion": 55.0,
    "incineration_energy": 30.0,
    "landfill": 0.0,
    "other": 10.0,
}

# Average food value EUR per kg by category
FOOD_VALUE_EUR_PER_KG: Dict[str, float] = {
    "bakery": 3.50,
    "produce": 2.20,
    "dairy": 4.10,
    "meat": 8.50,
    "seafood": 12.00,
    "deli": 6.50,
    "frozen": 4.80,
    "dry_goods": 2.80,
    "beverages": 1.50,
    "prepared_foods": 7.20,
    "other": 3.00,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class FoodWasteTrackingWorkflow:
    """
    4-phase food waste management workflow.

    Establishes waste baselines, analyzes by category and destination,
    sets reduction targets, and reports progress against the EU 2030
    30% reduction target.

    Example:
        >>> wf = FoodWasteTrackingWorkflow()
        >>> inp = FoodWasteInput(waste_records=[...])
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize FoodWasteTrackingWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._category_breakdowns: List[CategoryBreakdown] = []
        self._reduction_targets: List[ReductionTarget] = []
        self._recommendations: List[WasteRecommendation] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[FoodWasteInput] = None,
        waste_records: Optional[List[WasteRecord]] = None,
        baseline_data: Optional[BaselineData] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> FoodWasteResult:
        """Execute the 4-phase food waste tracking workflow."""
        if input_data is None:
            input_data = FoodWasteInput(
                waste_records=waste_records or [],
                baseline_data=baseline_data,
                config=config or {},
            )

        started_at = datetime.utcnow()
        self.logger.info("Starting food waste workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase_results.append(await self._phase_waste_baseline(input_data))
            phase_results.append(await self._phase_category_analysis(input_data))
            phase_results.append(await self._phase_reduction_targeting(input_data))
            phase_results.append(await self._phase_progress_reporting(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Food waste workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        total_waste_kg = sum(r.weight_kg for r in input_data.waste_records)
        total_value = sum(r.value_eur for r in input_data.waste_records)
        hierarchy_score = self._calc_overall_hierarchy_score(input_data.waste_records)

        # EU 2030 target assessment
        eu_status = "not_assessed"
        if input_data.baseline_data:
            baseline_tonnes = input_data.baseline_data.total_waste_tonnes
            current_tonnes = total_waste_kg / 1000.0
            if baseline_tonnes > 0:
                reduction_pct = ((baseline_tonnes - current_tonnes) / baseline_tonnes) * 100
                if reduction_pct >= 30:
                    eu_status = "on_track"
                elif reduction_pct >= 15:
                    eu_status = "partially_on_track"
                else:
                    eu_status = "off_track"

        baseline_tonnes = input_data.baseline_data.total_waste_tonnes if input_data.baseline_data else 0.0
        current_tonnes = total_waste_kg / 1000.0
        reduction_pct_val = 0.0
        if baseline_tonnes > 0:
            reduction_pct_val = ((baseline_tonnes - current_tonnes) / baseline_tonnes) * 100

        result = FoodWasteResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=elapsed,
            total_waste_tonnes=round(total_waste_kg / 1000.0, 4),
            total_value_lost_eur=round(total_value, 2),
            reduction_progress={
                "baseline_tonnes": round(baseline_tonnes, 4),
                "current_tonnes": round(current_tonnes, 4),
                "reduction_pct": round(reduction_pct_val, 2),
                "target_pct": 30.0,
                "gap_pct": round(max(30.0 - reduction_pct_val, 0), 2),
            },
            hierarchy_score=round(hierarchy_score, 2),
            category_breakdown=self._category_breakdowns,
            reduction_targets=self._reduction_targets,
            recommendations=self._recommendations,
            eu_2030_target_status=eu_status,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Waste Baseline
    # -------------------------------------------------------------------------

    async def _phase_waste_baseline(self, input_data: FoodWasteInput) -> PhaseResult:
        """Establish baseline from historical data."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        if input_data.baseline_data:
            outputs["baseline_year"] = input_data.baseline_data.baseline_year
            outputs["baseline_waste_tonnes"] = input_data.baseline_data.total_waste_tonnes
            outputs["baseline_waste_rate_pct"] = input_data.baseline_data.waste_rate_pct
        else:
            warnings.append("No baseline data provided; using current data as proxy")
            total_kg = sum(r.weight_kg for r in input_data.waste_records)
            outputs["baseline_year"] = input_data.reporting_year - 1
            outputs["baseline_waste_tonnes"] = round(total_kg / 1000.0, 4)
            outputs["baseline_waste_rate_pct"] = 0.0

        outputs["current_records"] = len(input_data.waste_records)
        outputs["current_waste_kg"] = round(sum(r.weight_kg for r in input_data.waste_records), 2)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 1 WasteBaseline: baseline=%.2f tonnes", outputs.get("baseline_waste_tonnes", 0))
        return PhaseResult(
            phase_name="waste_baseline", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Category Analysis
    # -------------------------------------------------------------------------

    async def _phase_category_analysis(self, input_data: FoodWasteInput) -> PhaseResult:
        """Breakdown waste by food category and destination."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        self._category_breakdowns = []

        category_data: Dict[str, Dict[str, Any]] = {}
        total_waste_kg = sum(r.weight_kg for r in input_data.waste_records)

        for rec in input_data.waste_records:
            cat = rec.food_category.value
            if cat not in category_data:
                category_data[cat] = {"waste_kg": 0.0, "value_eur": 0.0, "destinations": {}}
            category_data[cat]["waste_kg"] += rec.weight_kg
            category_data[cat]["value_eur"] += rec.value_eur
            dest = rec.destination.value
            category_data[cat]["destinations"][dest] = category_data[cat]["destinations"].get(dest, 0.0) + rec.weight_kg

        for cat, data in category_data.items():
            dest_breakdown = data["destinations"]
            total_cat_kg = data["waste_kg"]
            hierarchy_score = self._calc_hierarchy_score(dest_breakdown, total_cat_kg)
            share = (total_cat_kg / total_waste_kg * 100) if total_waste_kg > 0 else 0.0

            self._category_breakdowns.append(CategoryBreakdown(
                category=cat,
                waste_kg=round(data["waste_kg"], 2),
                share_of_total_pct=round(share, 2),
                value_lost_eur=round(data["value_eur"], 2),
                destination_breakdown={k: round(v, 2) for k, v in dest_breakdown.items()},
                hierarchy_score=round(hierarchy_score, 2),
            ))

        self._category_breakdowns.sort(key=lambda c: c.waste_kg, reverse=True)
        outputs["categories_analyzed"] = len(self._category_breakdowns)
        outputs["top_category"] = self._category_breakdowns[0].category if self._category_breakdowns else ""

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 2 CategoryAnalysis: %d categories", len(self._category_breakdowns))
        return PhaseResult(
            phase_name="category_analysis", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Reduction Targeting
    # -------------------------------------------------------------------------

    async def _phase_reduction_targeting(self, input_data: FoodWasteInput) -> PhaseResult:
        """Set targets per category and track progress."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        self._reduction_targets = []
        self._recommendations = []

        baseline_cats = input_data.baseline_data.category_baselines if input_data.baseline_data else {}

        for cb in self._category_breakdowns:
            baseline_kg = baseline_cats.get(cb.category, cb.waste_kg * 1.1)  # 10% increase as fallback baseline
            target_kg = baseline_kg * 0.70  # 30% reduction target
            gap_kg = cb.waste_kg - target_kg
            reduction_pct = ((baseline_kg - cb.waste_kg) / baseline_kg * 100) if baseline_kg > 0 else 0.0
            on_track = cb.waste_kg <= target_kg

            self._reduction_targets.append(ReductionTarget(
                category=cb.category,
                baseline_kg=round(baseline_kg, 2),
                current_kg=round(cb.waste_kg, 2),
                target_kg=round(target_kg, 2),
                reduction_pct=round(reduction_pct, 2),
                on_track=on_track,
                gap_kg=round(max(gap_kg, 0), 2),
            ))

            # Generate recommendations for categories not on track
            if not on_track and gap_kg > 0:
                price_per_kg = FOOD_VALUE_EUR_PER_KG.get(cb.category, 3.0)
                self._recommendations.append(WasteRecommendation(
                    category=cb.category,
                    action=f"Implement dynamic markdown pricing for {cb.category} nearing expiry",
                    expected_reduction_kg=round(gap_kg * 0.4, 2),
                    expected_savings_eur=round(gap_kg * 0.4 * price_per_kg, 2),
                    priority="high" if gap_kg > 1000 else "medium",
                ))
                self._recommendations.append(WasteRecommendation(
                    category=cb.category,
                    action=f"Expand redistribution partnerships for surplus {cb.category}",
                    expected_reduction_kg=round(gap_kg * 0.3, 2),
                    expected_savings_eur=round(gap_kg * 0.3 * price_per_kg * 0.5, 2),
                    priority="high",
                ))

        on_track_count = sum(1 for t in self._reduction_targets if t.on_track)
        outputs["categories_targeted"] = len(self._reduction_targets)
        outputs["on_track_count"] = on_track_count
        outputs["recommendations_generated"] = len(self._recommendations)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 3 ReductionTargeting: %d on track, %d recommendations", on_track_count, len(self._recommendations))
        return PhaseResult(
            phase_name="reduction_targeting", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Progress Reporting
    # -------------------------------------------------------------------------

    async def _phase_progress_reporting(self, input_data: FoodWasteInput) -> PhaseResult:
        """Report progress vs EU 30% reduction target."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}

        total_current_kg = sum(cb.waste_kg for cb in self._category_breakdowns)
        baseline_tonnes = input_data.baseline_data.total_waste_tonnes if input_data.baseline_data else total_current_kg / 1000.0 * 1.1
        current_tonnes = total_current_kg / 1000.0
        reduction_pct = ((baseline_tonnes - current_tonnes) / baseline_tonnes * 100) if baseline_tonnes > 0 else 0.0

        outputs["baseline_tonnes"] = round(baseline_tonnes, 4)
        outputs["current_tonnes"] = round(current_tonnes, 4)
        outputs["reduction_pct"] = round(reduction_pct, 2)
        outputs["target_pct"] = 30.0
        outputs["on_track_for_2030"] = reduction_pct >= 15.0  # halfway check
        outputs["hierarchy_score"] = round(self._calc_overall_hierarchy_score(input_data.waste_records), 2)
        outputs["total_value_lost_eur"] = round(sum(r.value_eur for r in input_data.waste_records), 2)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 4 ProgressReporting: reduction=%.1f%% vs 30%% target", reduction_pct)
        return PhaseResult(
            phase_name="progress_reporting", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _calc_hierarchy_score(self, destinations: Dict[str, float], total_kg: float) -> float:
        """Calculate waste hierarchy score (0-100) for a set of destinations."""
        if total_kg <= 0:
            return 0.0
        weighted_sum = 0.0
        for dest, kg in destinations.items():
            weight = HIERARCHY_WEIGHTS.get(dest, 10.0)
            weighted_sum += kg * weight
        return weighted_sum / total_kg

    def _calc_overall_hierarchy_score(self, records: List[WasteRecord]) -> float:
        """Calculate overall waste hierarchy score."""
        total_kg = sum(r.weight_kg for r in records)
        if total_kg <= 0:
            return 0.0
        weighted = sum(r.weight_kg * HIERARCHY_WEIGHTS.get(r.destination.value, 10.0) for r in records)
        return weighted / total_kg

    def _compute_provenance(self, result: FoodWasteResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
