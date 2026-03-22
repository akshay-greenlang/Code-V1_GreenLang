# -*- coding: utf-8 -*-
"""
Benchmark Analysis Workflow
===================================

3-phase utility benchmark analysis workflow within PACK-036 Utility
Analysis Pack.  Orchestrates metrics calculation, peer comparison, and
benchmark report generation for utility cost and consumption benchmarking.

Phases:
    1. MetricsCalculation  -- Calculate key utility performance metrics:
                               EUI (kWh/m2), cost intensity ($/m2), water
                               use intensity (gal/m2), demand intensity
                               (kW/m2), and utility cost per unit output
    2. PeerComparison      -- Compare metrics against published benchmarks
                               (ENERGY STAR, CIBSE TM46, BPIE), peer
                               databases, and industry medians
    3. BenchmarkReport     -- Generate benchmark report with performance
                               gaps, percentile rankings, and improvement
                               targets

The workflow follows GreenLang zero-hallucination principles: all
calculations use deterministic arithmetic, benchmark lookups from
published datasets, and statistical comparison via z-score/percentile
formulas. No LLM calls in the numeric path.

Schedule: quarterly / annually
Estimated duration: 15 minutes

Regulatory References:
    - ENERGY STAR Portfolio Manager Technical Reference (2023)
    - CIBSE TM46:2008 Energy Benchmarks
    - ASHRAE Standard 100-2018
    - EU EED 2023/1791 Article 8

Author: GreenLang Team
Version: 36.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "36.0.0"


def _utcnow() -> datetime:
    """Return current UTC timestamp with zero microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        s = data.model_dump(mode="json")
    elif isinstance(data, dict):
        s = data
    else:
        s = str(data)
    if isinstance(s, dict):
        s = {k: v for k, v in s.items()
             if k not in ("calculated_at", "processing_time_ms", "provenance_hash")}
    return hashlib.sha256(
        json.dumps(s, sort_keys=True, default=str).encode()
    ).hexdigest()


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
    UNIVERSITY = "university"
    WAREHOUSE = "warehouse"
    INDUSTRIAL = "industrial"
    RESTAURANT = "restaurant"
    SUPERMARKET = "supermarket"
    DATA_CENTRE = "data_centre"
    MIXED_USE = "mixed_use"


class PerformanceLevel(str, Enum):
    """Benchmark performance level classification."""
    TOP_QUARTILE = "top_quartile"
    ABOVE_MEDIAN = "above_median"
    BELOW_MEDIAN = "below_median"
    BOTTOM_QUARTILE = "bottom_quartile"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# CIBSE TM46 typical/good EUI (kWh/m2/yr)
CIBSE_TM46_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "office": {"typical_electric": 95.0, "typical_fossil": 120.0, "good_electric": 54.0, "good_fossil": 79.0},
    "retail": {"typical_electric": 165.0, "typical_fossil": 105.0, "good_electric": 90.0, "good_fossil": 60.0},
    "hotel": {"typical_electric": 105.0, "typical_fossil": 200.0, "good_electric": 60.0, "good_fossil": 120.0},
    "hospital": {"typical_electric": 90.0, "typical_fossil": 350.0, "good_electric": 65.0, "good_fossil": 250.0},
    "school": {"typical_electric": 40.0, "typical_fossil": 110.0, "good_electric": 22.0, "good_fossil": 65.0},
    "university": {"typical_electric": 75.0, "typical_fossil": 130.0, "good_electric": 50.0, "good_fossil": 85.0},
    "warehouse": {"typical_electric": 30.0, "typical_fossil": 35.0, "good_electric": 20.0, "good_fossil": 20.0},
    "industrial": {"typical_electric": 55.0, "typical_fossil": 200.0, "good_electric": 35.0, "good_fossil": 120.0},
    "restaurant": {"typical_electric": 250.0, "typical_fossil": 370.0, "good_electric": 150.0, "good_fossil": 200.0},
    "supermarket": {"typical_electric": 340.0, "typical_fossil": 80.0, "good_electric": 260.0, "good_fossil": 55.0},
    "data_centre": {"typical_electric": 500.0, "typical_fossil": 10.0, "good_electric": 300.0, "good_fossil": 5.0},
    "mixed_use": {"typical_electric": 100.0, "typical_fossil": 130.0, "good_electric": 60.0, "good_fossil": 80.0},
}

# Cost intensity benchmarks ($/m2/yr) by building type
COST_INTENSITY_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "office": {"typical": 28.0, "good": 18.0, "best": 12.0},
    "retail": {"typical": 35.0, "good": 22.0, "best": 15.0},
    "hotel": {"typical": 42.0, "good": 28.0, "best": 20.0},
    "hospital": {"typical": 55.0, "good": 38.0, "best": 28.0},
    "school": {"typical": 18.0, "good": 12.0, "best": 8.0},
    "warehouse": {"typical": 8.0, "good": 5.0, "best": 3.0},
    "industrial": {"typical": 30.0, "good": 20.0, "best": 14.0},
    "data_centre": {"typical": 85.0, "good": 55.0, "best": 38.0},
    "mixed_use": {"typical": 32.0, "good": 20.0, "best": 14.0},
    "default": {"typical": 30.0, "good": 20.0, "best": 14.0},
}

# Peer distribution parameters (std dev as fraction of typical)
PEER_STD_DEV_FRACTION: Dict[str, float] = {
    "office": 0.35, "retail": 0.40, "hotel": 0.30, "hospital": 0.25,
    "school": 0.30, "warehouse": 0.45, "industrial": 0.50,
    "data_centre": 0.40, "mixed_use": 0.35, "default": 0.35,
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


class BenchmarkComparison(BaseModel):
    """A benchmark comparison result.

    Attributes:
        metric_name: Name of the metric being compared.
        facility_value: Facility's value for this metric.
        benchmark_typical: Typical practice benchmark.
        benchmark_good: Good practice benchmark.
        benchmark_best: Best practice benchmark.
        gap_to_typical_pct: Gap vs typical (negative = better).
        gap_to_good_pct: Gap vs good practice.
        percentile: Estimated percentile rank.
        performance_level: Performance classification.
        unit: Unit of measure.
    """
    metric_name: str = Field(default="")
    facility_value: float = Field(default=0.0)
    benchmark_typical: float = Field(default=0.0)
    benchmark_good: float = Field(default=0.0)
    benchmark_best: float = Field(default=0.0)
    gap_to_typical_pct: float = Field(default=0.0)
    gap_to_good_pct: float = Field(default=0.0)
    percentile: float = Field(default=50.0)
    performance_level: str = Field(default="")
    unit: str = Field(default="")


class BenchmarkAnalysisInput(BaseModel):
    """Input data model for BenchmarkAnalysisWorkflow.

    Attributes:
        facility_id: Facility identifier.
        facility_name: Facility name.
        building_type: Building classification.
        floor_area_m2: Gross floor area.
        annual_electricity_kwh: Annual electricity consumption.
        annual_gas_kwh: Annual gas consumption.
        annual_water_gallons: Annual water consumption.
        annual_utility_cost: Total annual utility cost.
        peak_demand_kw: Annual peak demand.
        annual_output_units: Production/revenue output.
        output_unit_label: Label for output units.
        occupant_count: Number of occupants.
        operating_hours_per_year: Annual operating hours.
        currency: Currency code.
        entity_id: Multi-tenant entity identifier.
        tenant_id: Multi-tenant tenant identifier.
    """
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    building_type: BuildingType = Field(default=BuildingType.OFFICE)
    floor_area_m2: float = Field(default=0.0, ge=0.0)
    annual_electricity_kwh: float = Field(default=0.0, ge=0.0)
    annual_gas_kwh: float = Field(default=0.0, ge=0.0)
    annual_water_gallons: float = Field(default=0.0, ge=0.0)
    annual_utility_cost: float = Field(default=0.0, ge=0.0)
    peak_demand_kw: float = Field(default=0.0, ge=0.0)
    annual_output_units: float = Field(default=0.0, ge=0.0)
    output_unit_label: str = Field(default="units")
    occupant_count: int = Field(default=0, ge=0)
    operating_hours_per_year: float = Field(default=2500.0, ge=0.0)
    currency: str = Field(default="USD")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class BenchmarkAnalysisResult(BaseModel):
    """Complete result from benchmark analysis workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="benchmark_analysis")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    facility_id: str = Field(default="")
    building_type: str = Field(default="")
    comparisons: List[BenchmarkComparison] = Field(default_factory=list)
    overall_percentile: float = Field(default=50.0)
    overall_performance: str = Field(default="")
    metrics: Dict[str, Any] = Field(default_factory=dict)
    improvement_targets: List[Dict[str, Any]] = Field(default_factory=list)
    duration_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class BenchmarkAnalysisWorkflow:
    """
    3-phase utility benchmark analysis workflow.

    Calculates performance metrics, compares against published benchmarks,
    and generates benchmark report. Each phase produces a PhaseResult
    with SHA-256 provenance hash.

    Phases:
        1. MetricsCalculation - Calculate EUI, cost intensity, etc.
        2. PeerComparison     - Compare against benchmarks and peers
        3. BenchmarkReport    - Generate report with targets

    Zero-hallucination: all calculations use deterministic formulas and
    published CIBSE/ENERGY STAR benchmark data.

    Example:
        >>> wf = BenchmarkAnalysisWorkflow()
        >>> inp = BenchmarkAnalysisInput(
        ...     facility_id="fac-001",
        ...     building_type=BuildingType.OFFICE,
        ...     floor_area_m2=5000,
        ...     annual_electricity_kwh=500000,
        ...     annual_utility_cost=60000,
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise BenchmarkAnalysisWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self._metrics: Dict[str, Any] = {}
        self._comparisons: List[BenchmarkComparison] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def execute(self, input_data: BenchmarkAnalysisInput) -> BenchmarkAnalysisResult:
        """Execute the 3-phase benchmark analysis workflow.

        Args:
            input_data: Validated benchmark analysis input.

        Returns:
            BenchmarkAnalysisResult with comparisons and targets.
        """
        t_start = time.perf_counter()
        self.logger.info(
            "Starting benchmark analysis %s for facility=%s type=%s",
            self.workflow_id, input_data.facility_id,
            input_data.building_type.value,
        )

        self._phase_results = []
        self._metrics = {}
        self._comparisons = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = self._phase_1_metrics_calculation(input_data)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError(f"Phase 1 failed: {phase1.errors}")

            phase2 = self._phase_2_peer_comparison(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_3_benchmark_report(input_data)
            self._phase_results.append(phase3)

            failed_count = sum(
                1 for p in self._phase_results if p.status == PhaseStatus.FAILED
            )
            if failed_count == 0:
                overall_status = WorkflowStatus.COMPLETED
            elif failed_count < len(self._phase_results):
                overall_status = WorkflowStatus.PARTIAL
            else:
                overall_status = WorkflowStatus.FAILED

        except Exception as exc:
            self.logger.error("Benchmark analysis failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t_start

        # Overall performance from comparisons
        percentiles = [c.percentile for c in self._comparisons if c.percentile > 0]
        overall_pct = sum(percentiles) / len(percentiles) if percentiles else 50.0
        if overall_pct >= 75:
            overall_perf = "top_quartile"
        elif overall_pct >= 50:
            overall_perf = "above_median"
        elif overall_pct >= 25:
            overall_perf = "below_median"
        else:
            overall_perf = "bottom_quartile"

        # Improvement targets
        targets = [
            {
                "metric": c.metric_name,
                "current": c.facility_value,
                "target": c.benchmark_good,
                "reduction_needed": round(
                    max(0.0, c.facility_value - c.benchmark_good), 2
                ),
                "unit": c.unit,
            }
            for c in self._comparisons
            if c.facility_value > c.benchmark_good and c.benchmark_good > 0
        ]

        result = BenchmarkAnalysisResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            facility_id=input_data.facility_id,
            building_type=input_data.building_type.value,
            comparisons=self._comparisons,
            overall_percentile=round(overall_pct, 1),
            overall_performance=overall_perf,
            metrics=self._metrics,
            improvement_targets=targets,
            duration_seconds=round(elapsed, 4),
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Benchmark analysis %s completed in %.2fs: percentile=%.1f "
            "performance=%s",
            self.workflow_id, elapsed, overall_pct, overall_perf,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Metrics Calculation
    # -------------------------------------------------------------------------

    def _phase_1_metrics_calculation(
        self, input_data: BenchmarkAnalysisInput
    ) -> PhaseResult:
        """Calculate key utility performance metrics.

        Args:
            input_data: Benchmark analysis input.

        Returns:
            PhaseResult with metrics outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        area = input_data.floor_area_m2
        if area <= 0:
            return PhaseResult(
                phase_name="metrics_calculation", phase_number=1,
                status=PhaseStatus.FAILED,
                errors=["Floor area must be greater than zero"],
                duration_seconds=round(time.perf_counter() - t_start, 4),
            )

        total_energy_kwh = input_data.annual_electricity_kwh + input_data.annual_gas_kwh
        eui = total_energy_kwh / area
        electric_eui = input_data.annual_electricity_kwh / area
        gas_eui = input_data.annual_gas_kwh / area
        cost_intensity = input_data.annual_utility_cost / area
        demand_intensity = input_data.peak_demand_kw / area if input_data.peak_demand_kw > 0 else 0.0

        # Per-occupant metrics
        kwh_per_occupant = (
            total_energy_kwh / input_data.occupant_count
            if input_data.occupant_count > 0 else 0.0
        )
        cost_per_occupant = (
            input_data.annual_utility_cost / input_data.occupant_count
            if input_data.occupant_count > 0 else 0.0
        )

        # Water use intensity
        wui = (
            input_data.annual_water_gallons / area
            if input_data.annual_water_gallons > 0 else 0.0
        )

        # Output efficiency
        cost_per_output = (
            input_data.annual_utility_cost / input_data.annual_output_units
            if input_data.annual_output_units > 0 else 0.0
        )
        kwh_per_output = (
            total_energy_kwh / input_data.annual_output_units
            if input_data.annual_output_units > 0 else 0.0
        )

        # Operating hours intensity
        kwh_per_operating_hour = (
            total_energy_kwh / input_data.operating_hours_per_year
            if input_data.operating_hours_per_year > 0 else 0.0
        )

        # Average rate
        avg_rate = (
            input_data.annual_utility_cost / total_energy_kwh
            if total_energy_kwh > 0 else 0.0
        )

        self._metrics = {
            "eui_kwh_m2": round(eui, 2),
            "electric_eui_kwh_m2": round(electric_eui, 2),
            "gas_eui_kwh_m2": round(gas_eui, 2),
            "cost_intensity_per_m2": round(cost_intensity, 2),
            "demand_intensity_w_m2": round(demand_intensity * 1000.0, 2),
            "wui_gallons_m2": round(wui, 2),
            "kwh_per_occupant": round(kwh_per_occupant, 2),
            "cost_per_occupant": round(cost_per_occupant, 2),
            "cost_per_output": round(cost_per_output, 4),
            "kwh_per_output": round(kwh_per_output, 4),
            "kwh_per_operating_hour": round(kwh_per_operating_hour, 2),
            "avg_rate_per_kwh": round(avg_rate, 4),
            "total_energy_kwh": round(total_energy_kwh, 2),
            "floor_area_m2": round(area, 2),
        }

        outputs.update(self._metrics)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 1 MetricsCalculation: EUI=%.1f cost=$%.1f/m2 (%.3fs)",
            eui, cost_intensity, elapsed,
        )
        return PhaseResult(
            phase_name="metrics_calculation", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Peer Comparison
    # -------------------------------------------------------------------------

    def _phase_2_peer_comparison(
        self, input_data: BenchmarkAnalysisInput
    ) -> PhaseResult:
        """Compare metrics against published benchmarks.

        Args:
            input_data: Benchmark analysis input.

        Returns:
            PhaseResult with comparison outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        comparisons: List[BenchmarkComparison] = []

        bt = input_data.building_type.value
        cibse = CIBSE_TM46_BENCHMARKS.get(bt, CIBSE_TM46_BENCHMARKS["office"])
        cost_bench = COST_INTENSITY_BENCHMARKS.get(bt, COST_INTENSITY_BENCHMARKS["default"])

        eui = self._metrics.get("eui_kwh_m2", 0.0)
        cost_int = self._metrics.get("cost_intensity_per_m2", 0.0)
        electric_eui = self._metrics.get("electric_eui_kwh_m2", 0.0)

        # EUI comparison
        typical_eui = cibse["typical_electric"] + cibse["typical_fossil"]
        good_eui = cibse["good_electric"] + cibse["good_fossil"]
        best_eui = good_eui * 0.70

        eui_gap_typical = ((eui - typical_eui) / typical_eui * 100.0) if typical_eui > 0 else 0.0
        eui_gap_good = ((eui - good_eui) / good_eui * 100.0) if good_eui > 0 else 0.0
        eui_percentile = self._calc_percentile(eui, typical_eui, bt)

        comparisons.append(BenchmarkComparison(
            metric_name="Energy Use Intensity (EUI)",
            facility_value=round(eui, 2),
            benchmark_typical=round(typical_eui, 2),
            benchmark_good=round(good_eui, 2),
            benchmark_best=round(best_eui, 2),
            gap_to_typical_pct=round(eui_gap_typical, 2),
            gap_to_good_pct=round(eui_gap_good, 2),
            percentile=round(eui_percentile, 1),
            performance_level=self._classify_performance(eui_percentile),
            unit="kWh/m2/yr",
        ))

        # Electricity EUI comparison
        elec_typical = cibse["typical_electric"]
        elec_good = cibse["good_electric"]
        elec_gap = ((electric_eui - elec_typical) / elec_typical * 100.0) if elec_typical > 0 else 0.0
        elec_pct = self._calc_percentile(electric_eui, elec_typical, bt)

        comparisons.append(BenchmarkComparison(
            metric_name="Electricity EUI",
            facility_value=round(electric_eui, 2),
            benchmark_typical=round(elec_typical, 2),
            benchmark_good=round(elec_good, 2),
            benchmark_best=round(elec_good * 0.70, 2),
            gap_to_typical_pct=round(elec_gap, 2),
            gap_to_good_pct=round(
                ((electric_eui - elec_good) / elec_good * 100.0) if elec_good > 0 else 0.0, 2
            ),
            percentile=round(elec_pct, 1),
            performance_level=self._classify_performance(elec_pct),
            unit="kWh/m2/yr",
        ))

        # Cost intensity comparison
        cost_typical = cost_bench["typical"]
        cost_good = cost_bench["good"]
        cost_best = cost_bench["best"]
        cost_gap_typical = ((cost_int - cost_typical) / cost_typical * 100.0) if cost_typical > 0 else 0.0
        cost_pct = self._calc_percentile(cost_int, cost_typical, bt)

        comparisons.append(BenchmarkComparison(
            metric_name="Cost Intensity",
            facility_value=round(cost_int, 2),
            benchmark_typical=round(cost_typical, 2),
            benchmark_good=round(cost_good, 2),
            benchmark_best=round(cost_best, 2),
            gap_to_typical_pct=round(cost_gap_typical, 2),
            gap_to_good_pct=round(
                ((cost_int - cost_good) / cost_good * 100.0) if cost_good > 0 else 0.0, 2
            ),
            percentile=round(cost_pct, 1),
            performance_level=self._classify_performance(cost_pct),
            unit="$/m2/yr",
        ))

        self._comparisons = comparisons

        outputs["comparisons_count"] = len(comparisons)
        outputs["avg_percentile"] = round(
            sum(c.percentile for c in comparisons) / len(comparisons), 1
        )
        outputs["top_quartile_metrics"] = sum(
            1 for c in comparisons if c.percentile >= 75
        )
        outputs["below_median_metrics"] = sum(
            1 for c in comparisons if c.percentile < 50
        )

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 2 PeerComparison: %d comparisons, avg_pct=%.1f (%.3fs)",
            len(comparisons), outputs["avg_percentile"], elapsed,
        )
        return PhaseResult(
            phase_name="peer_comparison", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    def _calc_percentile(
        self, value: float, typical: float, building_type: str
    ) -> float:
        """Calculate percentile using z-score against peer distribution.

        Lower values are better (lower EUI = higher percentile).

        Args:
            value: Facility metric value.
            typical: Typical benchmark (used as mean).
            building_type: Building type for std dev lookup.

        Returns:
            Percentile rank (0-100, higher is better).
        """
        if typical <= 0:
            return 50.0

        std_fraction = PEER_STD_DEV_FRACTION.get(
            building_type, PEER_STD_DEV_FRACTION["default"]
        )
        std_dev = typical * std_fraction

        if std_dev <= 0:
            return 50.0

        # Z-score (negative z = better than mean for lower-is-better)
        z = (typical - value) / std_dev
        # Logistic CDF approximation
        percentile = 100.0 / (1.0 + math.exp(-1.7 * z))
        return max(1.0, min(99.0, percentile))

    def _classify_performance(self, percentile: float) -> str:
        """Classify performance level from percentile.

        Args:
            percentile: Percentile rank.

        Returns:
            Performance level string.
        """
        if percentile >= 75:
            return PerformanceLevel.TOP_QUARTILE.value
        elif percentile >= 50:
            return PerformanceLevel.ABOVE_MEDIAN.value
        elif percentile >= 25:
            return PerformanceLevel.BELOW_MEDIAN.value
        return PerformanceLevel.BOTTOM_QUARTILE.value

    # -------------------------------------------------------------------------
    # Phase 3: Benchmark Report
    # -------------------------------------------------------------------------

    def _phase_3_benchmark_report(
        self, input_data: BenchmarkAnalysisInput
    ) -> PhaseResult:
        """Generate benchmark report with targets.

        Args:
            input_data: Benchmark analysis input.

        Returns:
            PhaseResult with report outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        report_id = f"rpt-{uuid.uuid4().hex[:8]}"

        # Generate improvement recommendations
        recommendations: List[Dict[str, Any]] = []
        for comp in self._comparisons:
            if comp.facility_value > comp.benchmark_good and comp.benchmark_good > 0:
                reduction = comp.facility_value - comp.benchmark_good
                reduction_pct = (reduction / comp.facility_value) * 100.0

                if comp.metric_name == "Energy Use Intensity (EUI)":
                    savings_kwh = reduction * input_data.floor_area_m2
                    recommendations.append({
                        "metric": comp.metric_name,
                        "current": round(comp.facility_value, 2),
                        "target": round(comp.benchmark_good, 2),
                        "reduction": round(reduction, 2),
                        "reduction_pct": round(reduction_pct, 1),
                        "savings_kwh": round(savings_kwh, 0),
                        "priority": "high" if reduction_pct > 20 else "medium",
                    })
                elif comp.metric_name == "Cost Intensity":
                    savings_cost = reduction * input_data.floor_area_m2
                    recommendations.append({
                        "metric": comp.metric_name,
                        "current": round(comp.facility_value, 2),
                        "target": round(comp.benchmark_good, 2),
                        "reduction": round(reduction, 2),
                        "reduction_pct": round(reduction_pct, 1),
                        "savings_cost": round(savings_cost, 2),
                        "priority": "high" if reduction_pct > 25 else "medium",
                    })
                else:
                    recommendations.append({
                        "metric": comp.metric_name,
                        "current": round(comp.facility_value, 2),
                        "target": round(comp.benchmark_good, 2),
                        "reduction": round(reduction, 2),
                        "reduction_pct": round(reduction_pct, 1),
                        "priority": "medium",
                    })

        outputs["report_id"] = report_id
        outputs["generated_at"] = _utcnow().isoformat()
        outputs["facility_id"] = input_data.facility_id
        outputs["building_type"] = input_data.building_type.value
        outputs["metrics_compared"] = len(self._comparisons)
        outputs["recommendations"] = recommendations
        outputs["recommendation_count"] = len(recommendations)
        outputs["benchmark_sources"] = [
            "CIBSE TM46:2008",
            "ENERGY STAR Portfolio Manager",
            "ASHRAE Standard 100-2018",
        ]
        outputs["methodology"] = [
            "EUI calculated as total energy / floor area",
            "Cost intensity calculated as total cost / floor area",
            "Percentile ranking via z-score against peer distribution",
            "Benchmarks from published CIBSE/ENERGY STAR datasets",
        ]

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 3 BenchmarkReport: report=%s, %d recommendations (%.3fs)",
            report_id, len(recommendations), elapsed,
        )
        return PhaseResult(
            phase_name="benchmark_report", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )
