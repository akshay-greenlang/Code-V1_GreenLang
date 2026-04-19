# -*- coding: utf-8 -*-
"""
Portfolio Benchmark Workflow
===================================

4-phase workflow for portfolio-level energy benchmarking within
PACK-035 Energy Benchmark Pack.

Phases:
    1. PortfolioDataCollection  -- Gather and validate data for all facilities
    2. FacilityBenchmarking     -- Calculate EUI and peer rank per facility
    3. PortfolioAggregation     -- Aggregate metrics, rank facilities, detect outliers
    4. PortfolioReport          -- Generate portfolio report with improvement priorities

Supports 1-1000+ facilities with mixed building types. Provides portfolio-wide
metrics including area-weighted EUI, facility rankings, distribution analysis,
outlier detection, and year-over-year improvement tracking.

The workflow follows GreenLang zero-hallucination principles: all EUI
calculations, aggregations, and rankings use deterministic formulas.
No LLM calls in the numeric computation path.

Schedule: quarterly / annual
Estimated duration: 120 minutes (portfolio-dependent)

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal
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


class BuildingType(str, Enum):
    """Building type classification."""

    OFFICE = "office"
    RETAIL = "retail"
    HOTEL = "hotel"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    WAREHOUSE = "warehouse"
    INDUSTRIAL = "industrial"
    DATA_CENTRE = "data_centre"
    MIXED_USE = "mixed_use"


class AggregationMethod(str, Enum):
    """Portfolio aggregation method."""

    AREA_WEIGHTED = "area_weighted"
    SIMPLE_AVERAGE = "simple_average"
    MEDIAN = "median"
    TOTAL_CONSUMPTION = "total_consumption"


class RankingCriteria(str, Enum):
    """Facility ranking criteria."""

    SITE_EUI = "site_eui"
    SOURCE_EUI = "source_eui"
    ENERGY_STAR_SCORE = "energy_star_score"
    CARBON_INTENSITY = "carbon_intensity"
    YOY_IMPROVEMENT = "yoy_improvement"
    COST_PER_M2 = "cost_per_m2"


class OutlierStatus(str, Enum):
    """Outlier detection status."""

    NORMAL = "normal"
    MILD_OUTLIER = "mild_outlier"
    EXTREME_OUTLIER = "extreme_outlier"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

CIBSE_TYPICAL_EUI: Dict[str, float] = {
    "office": 215.0, "retail": 270.0, "hotel": 305.0, "hospital": 440.0,
    "school": 150.0, "warehouse": 65.0, "industrial": 255.0,
    "data_centre": 510.0, "mixed_use": 230.0,
}

DEFAULT_EMISSION_FACTORS: Dict[str, float] = {
    "electricity": 0.207,
    "natural_gas": 0.183,
    "fuel_oil": 0.267,
    "district_heating": 0.194,
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
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class FacilityRecord(BaseModel):
    """Individual facility data for portfolio benchmarking."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(default="", description="Facility name")
    building_type: BuildingType = Field(default=BuildingType.OFFICE)
    floor_area_m2: float = Field(default=0.0, ge=0.0)
    annual_consumption_kwh: float = Field(default=0.0, ge=0.0)
    annual_cost: float = Field(default=0.0, ge=0.0)
    prior_year_consumption_kwh: float = Field(default=0.0, ge=0.0, description="Prior year for YoY")
    electric_kwh: float = Field(default=0.0, ge=0.0, description="Electric portion")
    fossil_kwh: float = Field(default=0.0, ge=0.0, description="Fossil fuel portion")
    country: str = Field(default="", description="ISO alpha-2")
    climate_zone: str = Field(default="", description="ASHRAE climate zone")
    year_built: int = Field(default=0, ge=0)


class FacilityBenchmarkResult(BaseModel):
    """Benchmark result for a single facility."""

    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    building_type: str = Field(default="")
    floor_area_m2: float = Field(default=0.0, ge=0.0)
    site_eui: float = Field(default=0.0, ge=0.0, description="kWh/m2/yr")
    energy_star_score: int = Field(default=50, ge=1, le=100)
    carbon_intensity: float = Field(default=0.0, ge=0.0, description="kgCO2e/m2/yr")
    cost_per_m2: float = Field(default=0.0, ge=0.0)
    yoy_improvement_pct: float = Field(default=0.0, description="Year-over-year %")
    rank_in_portfolio: int = Field(default=0, ge=0)
    percentile_in_portfolio: float = Field(default=50.0, ge=0.0, le=100.0)
    outlier_status: OutlierStatus = Field(default=OutlierStatus.NORMAL)
    benchmark_vs_typical_pct: float = Field(default=0.0, description="Gap to typical benchmark %")


class PortfolioBenchmarkInput(BaseModel):
    """Input data model for PortfolioBenchmarkWorkflow."""

    portfolio_id: str = Field(default_factory=lambda: f"pf-{uuid.uuid4().hex[:8]}")
    portfolio_name: str = Field(default="", description="Portfolio name")
    facility_list: List[FacilityRecord] = Field(default_factory=list)
    aggregation_method: AggregationMethod = Field(default=AggregationMethod.AREA_WEIGHTED)
    ranking_criteria: RankingCriteria = Field(default=RankingCriteria.SITE_EUI)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    outlier_threshold_iqr: float = Field(default=1.5, ge=1.0, le=3.0, description="IQR multiplier")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facility_list")
    @classmethod
    def validate_facilities(cls, v: List[FacilityRecord]) -> List[FacilityRecord]:
        """Ensure at least one facility is provided."""
        if not v:
            raise ValueError("At least one facility must be provided")
        return v


class PortfolioBenchmarkResult(BaseModel):
    """Complete result from portfolio benchmark workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="portfolio_benchmark")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    portfolio_id: str = Field(default="")
    portfolio_name: str = Field(default="")
    portfolio_metrics: Dict[str, Any] = Field(default_factory=dict)
    facility_rankings: List[Dict[str, Any]] = Field(default_factory=list)
    distribution: Dict[str, Any] = Field(default_factory=dict)
    yoy_improvement: Dict[str, Any] = Field(default_factory=dict)
    outliers: List[Dict[str, Any]] = Field(default_factory=list)
    facility_count: int = Field(default=0, ge=0)
    total_floor_area_m2: float = Field(default=0.0, ge=0.0)
    total_consumption_kwh: float = Field(default=0.0, ge=0.0)
    duration_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class PortfolioBenchmarkWorkflow:
    """
    4-phase portfolio-level energy benchmarking workflow.

    Performs portfolio data collection, per-facility benchmarking,
    portfolio aggregation with outlier detection, and portfolio
    report generation with improvement prioritisation.

    Zero-hallucination: all EUI calculations, statistical aggregations,
    outlier detection (IQR method), and rankings use deterministic
    formulas. No LLM calls in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.
        _facility_results: Per-facility benchmark results.
        _portfolio_metrics: Aggregated portfolio metrics.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = PortfolioBenchmarkWorkflow()
        >>> inp = PortfolioBenchmarkInput(
        ...     facility_list=[FacilityRecord(facility_name="HQ", ...)],
        ... )
        >>> result = wf.run(inp)
        >>> assert result.facility_count > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PortfolioBenchmarkWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._facility_results: List[FacilityBenchmarkResult] = []
        self._portfolio_metrics: Dict[str, Any] = {}
        self._outliers: List[Dict[str, Any]] = []
        self._validated_facilities: List[FacilityRecord] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: PortfolioBenchmarkInput) -> PortfolioBenchmarkResult:
        """
        Execute the 4-phase portfolio benchmark workflow.

        Args:
            input_data: Validated portfolio benchmark input.

        Returns:
            PortfolioBenchmarkResult with portfolio metrics, rankings, distribution.
        """
        t_start = time.perf_counter()
        self.logger.info(
            "Starting portfolio benchmark workflow %s for %d facilities",
            self.workflow_id, len(input_data.facility_list),
        )

        self._phase_results = []
        self._facility_results = []
        self._portfolio_metrics = {}
        self._outliers = []
        self._validated_facilities = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = self._phase_portfolio_data_collection(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_facility_benchmarking(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_portfolio_aggregation(input_data)
            self._phase_results.append(phase3)

            phase4 = self._phase_portfolio_report(input_data)
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Portfolio benchmark workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t_start
        total_area = sum(f.floor_area_m2 for f in self._validated_facilities)
        total_kwh = sum(f.annual_consumption_kwh for f in self._validated_facilities)

        # YoY improvement
        yoy_data = self._calculate_yoy_improvement()

        result = PortfolioBenchmarkResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            portfolio_id=input_data.portfolio_id,
            portfolio_name=input_data.portfolio_name,
            portfolio_metrics=self._portfolio_metrics,
            facility_rankings=[fr.model_dump() for fr in self._facility_results],
            distribution=self._calculate_distribution(),
            yoy_improvement=yoy_data,
            outliers=self._outliers,
            facility_count=len(self._validated_facilities),
            total_floor_area_m2=round(total_area, 2),
            total_consumption_kwh=round(total_kwh, 2),
            duration_seconds=round(elapsed, 4),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Portfolio benchmark workflow %s completed in %.2fs facilities=%d",
            self.workflow_id, elapsed, len(self._validated_facilities),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Portfolio Data Collection
    # -------------------------------------------------------------------------

    def _phase_portfolio_data_collection(
        self, input_data: PortfolioBenchmarkInput
    ) -> PhaseResult:
        """Gather and validate data for all facilities in the portfolio."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        valid = 0
        invalid = 0
        for fac in input_data.facility_list:
            issues: List[str] = []
            if fac.floor_area_m2 <= 0:
                issues.append("zero_floor_area")
            if fac.annual_consumption_kwh <= 0:
                issues.append("zero_consumption")

            if issues:
                invalid += 1
                warnings.append(
                    f"Facility {fac.facility_id}: data issues: {', '.join(issues)}"
                )
            else:
                valid += 1
                self._validated_facilities.append(fac)

        # Building type distribution
        type_counts: Dict[str, int] = {}
        total_area = 0.0
        total_kwh = 0.0
        for fac in self._validated_facilities:
            bt = fac.building_type.value
            type_counts[bt] = type_counts.get(bt, 0) + 1
            total_area += fac.floor_area_m2
            total_kwh += fac.annual_consumption_kwh

        outputs["total_facilities"] = len(input_data.facility_list)
        outputs["valid_facilities"] = valid
        outputs["invalid_facilities"] = invalid
        outputs["building_type_distribution"] = type_counts
        outputs["total_floor_area_m2"] = round(total_area, 2)
        outputs["total_consumption_kwh"] = round(total_kwh, 2)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 1 PortfolioDataCollection: %d valid, %d invalid of %d facilities",
            valid, invalid, len(input_data.facility_list),
        )
        return PhaseResult(
            phase_name="portfolio_data_collection", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Facility Benchmarking
    # -------------------------------------------------------------------------

    def _phase_facility_benchmarking(
        self, input_data: PortfolioBenchmarkInput
    ) -> PhaseResult:
        """Calculate EUI, ENERGY STAR score, and carbon intensity per facility."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for fac in self._validated_facilities:
            result = self._benchmark_single_facility(fac)
            self._facility_results.append(result)

        outputs["facilities_benchmarked"] = len(self._facility_results)
        euis = [fr.site_eui for fr in self._facility_results]
        if euis:
            outputs["eui_range"] = f"{min(euis):.1f}-{max(euis):.1f}"
            outputs["eui_mean"] = round(sum(euis) / len(euis), 2)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 2 FacilityBenchmarking: %d facilities benchmarked",
            len(self._facility_results),
        )
        return PhaseResult(
            phase_name="facility_benchmarking", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _benchmark_single_facility(self, fac: FacilityRecord) -> FacilityBenchmarkResult:
        """Benchmark a single facility (zero-hallucination)."""
        site_eui = fac.annual_consumption_kwh / fac.floor_area_m2 if fac.floor_area_m2 > 0 else 0.0
        cost_per_m2 = fac.annual_cost / fac.floor_area_m2 if fac.floor_area_m2 > 0 else 0.0

        # Carbon intensity
        elec_ef = DEFAULT_EMISSION_FACTORS.get("electricity", 0.207)
        fossil_ef = DEFAULT_EMISSION_FACTORS.get("natural_gas", 0.183)
        total_co2_kg = fac.electric_kwh * elec_ef + fac.fossil_kwh * fossil_ef
        if total_co2_kg == 0 and fac.annual_consumption_kwh > 0:
            total_co2_kg = fac.annual_consumption_kwh * 0.207
        carbon_intensity = total_co2_kg / fac.floor_area_m2 if fac.floor_area_m2 > 0 else 0.0

        # ENERGY STAR score estimation
        typical = CIBSE_TYPICAL_EUI.get(fac.building_type.value, 215.0)
        ratio = site_eui / typical if typical > 0 else 1.0
        if ratio <= 0.4:
            es_score = 95
        elif ratio <= 0.6:
            es_score = 85
        elif ratio <= 0.8:
            es_score = 70
        elif ratio <= 1.0:
            es_score = 50
        elif ratio <= 1.3:
            es_score = 30
        else:
            es_score = max(1, 15)

        # YoY improvement
        yoy = 0.0
        if fac.prior_year_consumption_kwh > 0:
            yoy = (fac.prior_year_consumption_kwh - fac.annual_consumption_kwh) / fac.prior_year_consumption_kwh * 100.0

        # Benchmark gap
        gap_pct = ((site_eui - typical) / typical * 100.0) if typical > 0 else 0.0

        return FacilityBenchmarkResult(
            facility_id=fac.facility_id,
            facility_name=fac.facility_name,
            building_type=fac.building_type.value,
            floor_area_m2=fac.floor_area_m2,
            site_eui=round(site_eui, 2),
            energy_star_score=es_score,
            carbon_intensity=round(carbon_intensity, 4),
            cost_per_m2=round(cost_per_m2, 2),
            yoy_improvement_pct=round(yoy, 2),
            benchmark_vs_typical_pct=round(gap_pct, 2),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Portfolio Aggregation
    # -------------------------------------------------------------------------

    def _phase_portfolio_aggregation(
        self, input_data: PortfolioBenchmarkInput
    ) -> PhaseResult:
        """Aggregate metrics, rank facilities, detect outliers."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not self._facility_results:
            return PhaseResult(
                phase_name="portfolio_aggregation", phase_number=3,
                status=PhaseStatus.SKIPPED, warnings=["No facility results"],
            )

        # Portfolio-level EUI
        total_area = sum(fr.floor_area_m2 for fr in self._facility_results)
        total_kwh = sum(fr.site_eui * fr.floor_area_m2 for fr in self._facility_results)
        total_cost = sum(fr.cost_per_m2 * fr.floor_area_m2 for fr in self._facility_results)
        total_co2 = sum(fr.carbon_intensity * fr.floor_area_m2 for fr in self._facility_results)

        euis = sorted([fr.site_eui for fr in self._facility_results])
        n = len(euis)

        # Aggregation method
        if input_data.aggregation_method == AggregationMethod.AREA_WEIGHTED:
            portfolio_eui = total_kwh / total_area if total_area > 0 else 0.0
        elif input_data.aggregation_method == AggregationMethod.MEDIAN:
            portfolio_eui = euis[n // 2] if n > 0 else 0.0
        elif input_data.aggregation_method == AggregationMethod.SIMPLE_AVERAGE:
            portfolio_eui = sum(euis) / n if n > 0 else 0.0
        else:
            portfolio_eui = total_kwh / total_area if total_area > 0 else 0.0

        # Ranking
        ranking_attr = input_data.ranking_criteria.value
        attr_map = {
            "site_eui": "site_eui",
            "source_eui": "site_eui",
            "energy_star_score": "energy_star_score",
            "carbon_intensity": "carbon_intensity",
            "yoy_improvement": "yoy_improvement_pct",
            "cost_per_m2": "cost_per_m2",
        }
        sort_attr = attr_map.get(ranking_attr, "site_eui")
        reverse_sort = ranking_attr == "energy_star_score" or ranking_attr == "yoy_improvement"
        sorted_results = sorted(
            self._facility_results,
            key=lambda fr: getattr(fr, sort_attr, 0),
            reverse=reverse_sort,
        )
        for idx, fr in enumerate(sorted_results, start=1):
            fr.rank_in_portfolio = idx
            fr.percentile_in_portfolio = round((1.0 - (idx - 1) / max(n, 1)) * 100.0, 1)

        self._facility_results = sorted_results

        # Outlier detection using IQR method (zero-hallucination)
        q1_idx = max(0, n // 4)
        q3_idx = min(n - 1, 3 * n // 4)
        q1 = euis[q1_idx] if euis else 0.0
        q3 = euis[q3_idx] if euis else 0.0
        iqr = q3 - q1
        lower_fence = q1 - input_data.outlier_threshold_iqr * iqr
        upper_fence = q3 + input_data.outlier_threshold_iqr * iqr

        for fr in self._facility_results:
            if fr.site_eui > q3 + 3.0 * iqr:
                fr.outlier_status = OutlierStatus.EXTREME_OUTLIER
                self._outliers.append({
                    "facility_id": fr.facility_id,
                    "facility_name": fr.facility_name,
                    "site_eui": fr.site_eui,
                    "status": "extreme_outlier",
                    "upper_fence": round(upper_fence, 2),
                })
            elif fr.site_eui > upper_fence:
                fr.outlier_status = OutlierStatus.MILD_OUTLIER
                self._outliers.append({
                    "facility_id": fr.facility_id,
                    "facility_name": fr.facility_name,
                    "site_eui": fr.site_eui,
                    "status": "mild_outlier",
                    "upper_fence": round(upper_fence, 2),
                })

        # Portfolio metrics
        self._portfolio_metrics = {
            "portfolio_eui": round(portfolio_eui, 2),
            "aggregation_method": input_data.aggregation_method.value,
            "total_floor_area_m2": round(total_area, 2),
            "total_consumption_kwh": round(sum(f.annual_consumption_kwh for f in self._validated_facilities), 2),
            "total_cost": round(total_cost, 2),
            "total_co2_kg": round(total_co2, 2),
            "portfolio_carbon_intensity": round(total_co2 / total_area, 4) if total_area > 0 else 0.0,
            "portfolio_cost_per_m2": round(total_cost / total_area, 2) if total_area > 0 else 0.0,
            "facility_count": n,
            "outlier_count": len(self._outliers),
            "iqr": round(iqr, 2),
            "q1_eui": round(q1, 2),
            "q3_eui": round(q3, 2),
        }

        outputs.update(self._portfolio_metrics)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 3 PortfolioAggregation: EUI=%.1f outliers=%d ranking=%s",
            portfolio_eui, len(self._outliers), input_data.ranking_criteria.value,
        )
        return PhaseResult(
            phase_name="portfolio_aggregation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Portfolio Report
    # -------------------------------------------------------------------------

    def _phase_portfolio_report(
        self, input_data: PortfolioBenchmarkInput
    ) -> PhaseResult:
        """Generate portfolio report with improvement priorities."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Improvement priorities (highest EUI first, excluding outliers investigated separately)
        priorities: List[Dict[str, Any]] = []
        for fr in self._facility_results:
            if fr.benchmark_vs_typical_pct > 0:
                savings_potential_kwh = (fr.site_eui - fr.site_eui / (1 + fr.benchmark_vs_typical_pct / 100.0)) * fr.floor_area_m2
                priorities.append({
                    "facility_id": fr.facility_id,
                    "facility_name": fr.facility_name,
                    "site_eui": fr.site_eui,
                    "gap_to_typical_pct": fr.benchmark_vs_typical_pct,
                    "savings_potential_kwh": round(savings_potential_kwh, 0),
                    "rank": fr.rank_in_portfolio,
                    "priority": "high" if fr.benchmark_vs_typical_pct > 30 else "medium" if fr.benchmark_vs_typical_pct > 10 else "low",
                })

        priorities.sort(key=lambda p: p["savings_potential_kwh"], reverse=True)

        # Building type summary
        type_summary: Dict[str, Dict[str, Any]] = {}
        for fr in self._facility_results:
            bt = fr.building_type
            if bt not in type_summary:
                type_summary[bt] = {"count": 0, "total_area": 0.0, "total_eui_area": 0.0, "euis": []}
            type_summary[bt]["count"] += 1
            type_summary[bt]["total_area"] += fr.floor_area_m2
            type_summary[bt]["total_eui_area"] += fr.site_eui * fr.floor_area_m2
            type_summary[bt]["euis"].append(fr.site_eui)

        bt_report: List[Dict[str, Any]] = []
        for bt, data in type_summary.items():
            avg_eui = data["total_eui_area"] / data["total_area"] if data["total_area"] > 0 else 0.0
            bt_report.append({
                "building_type": bt,
                "count": data["count"],
                "area_weighted_eui": round(avg_eui, 2),
                "typical_benchmark": CIBSE_TYPICAL_EUI.get(bt, 215.0),
                "gap_pct": round((avg_eui - CIBSE_TYPICAL_EUI.get(bt, 215.0)) / CIBSE_TYPICAL_EUI.get(bt, 215.0) * 100.0, 2) if CIBSE_TYPICAL_EUI.get(bt, 215.0) > 0 else 0.0,
            })

        report = {
            "report_id": f"rpt-{uuid.uuid4().hex[:8]}",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "improvement_priorities": priorities[:20],
            "building_type_summary": bt_report,
            "total_savings_potential_kwh": round(sum(p["savings_potential_kwh"] for p in priorities), 0),
        }

        outputs["report_id"] = report["report_id"]
        outputs["improvement_priorities_count"] = len(priorities)
        outputs["total_savings_potential_kwh"] = report["total_savings_potential_kwh"]

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 4 PortfolioReport: %d priorities, savings=%.0f kWh",
            len(priorities), report["total_savings_potential_kwh"],
        )
        return PhaseResult(
            phase_name="portfolio_report", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Supporting Methods
    # -------------------------------------------------------------------------

    def _calculate_distribution(self) -> Dict[str, Any]:
        """Calculate EUI distribution statistics for the portfolio."""
        if not self._facility_results:
            return {}

        euis = sorted([fr.site_eui for fr in self._facility_results])
        n = len(euis)
        mean = sum(euis) / n
        variance = sum((e - mean) ** 2 for e in euis) / n
        std_dev = math.sqrt(variance)

        return {
            "count": n,
            "mean": round(mean, 2),
            "median": round(euis[n // 2], 2),
            "std_dev": round(std_dev, 2),
            "min": round(euis[0], 2),
            "max": round(euis[-1], 2),
            "p10": round(euis[max(0, int(n * 0.10))], 2),
            "p25": round(euis[max(0, int(n * 0.25))], 2),
            "p75": round(euis[min(n - 1, int(n * 0.75))], 2),
            "p90": round(euis[min(n - 1, int(n * 0.90))], 2),
            "cv_pct": round(std_dev / mean * 100.0, 2) if mean > 0 else 0.0,
        }

    def _calculate_yoy_improvement(self) -> Dict[str, Any]:
        """Calculate year-over-year improvement statistics."""
        yoy_values = [fr.yoy_improvement_pct for fr in self._facility_results if fr.yoy_improvement_pct != 0.0]
        if not yoy_values:
            return {"facilities_with_prior_year": 0}

        improving = sum(1 for v in yoy_values if v > 0)
        worsening = sum(1 for v in yoy_values if v < 0)

        return {
            "facilities_with_prior_year": len(yoy_values),
            "average_improvement_pct": round(sum(yoy_values) / len(yoy_values), 2),
            "improving_count": improving,
            "worsening_count": worsening,
            "stable_count": len(yoy_values) - improving - worsening,
            "best_improvement_pct": round(max(yoy_values), 2),
            "worst_regression_pct": round(min(yoy_values), 2),
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: PortfolioBenchmarkResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
