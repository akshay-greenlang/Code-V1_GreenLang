# -*- coding: utf-8 -*-
"""
Performance Gap Workflow
===================================

3-phase workflow for end-use disaggregated performance gap analysis within
PACK-035 Energy Benchmark Pack.

Phases:
    1. BenchmarkEstablishment -- Establish facility EUI and target benchmark
    2. EndUseDisaggregation   -- Disaggregate consumption by end use (HVAC, lighting,
                                 plug loads, process, domestic hot water)
    3. GapAnalysisReport      -- Identify gaps by end use, rank improvement priorities,
                                 quantify savings potential

The workflow follows GreenLang zero-hallucination principles: end-use
disaggregation uses published split ratios (CIBSE Guide F, ASHRAE), gap
calculations are deterministic difference formulas, and savings potential
uses validated benchmark targets. No LLM calls in the numeric path.

Schedule: on-demand / annual
Estimated duration: 30 minutes

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
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


class EndUseCategory(str, Enum):
    """End-use energy category."""

    HEATING = "heating"
    COOLING = "cooling"
    VENTILATION = "ventilation"
    LIGHTING = "lighting"
    PLUG_LOADS = "plug_loads"
    DOMESTIC_HOT_WATER = "domestic_hot_water"
    PROCESS = "process"
    CATERING = "catering"
    LIFTS_ESCALATORS = "lifts_escalators"
    IT_EQUIPMENT = "it_equipment"
    REFRIGERATION = "refrigeration"
    OTHER = "other"


class GapSeverity(str, Enum):
    """Performance gap severity."""

    CRITICAL = "critical"
    SIGNIFICANT = "significant"
    MODERATE = "moderate"
    MINOR = "minor"
    AT_BENCHMARK = "at_benchmark"


class BenchmarkTarget(str, Enum):
    """Benchmark target level."""

    TYPICAL = "typical"
    GOOD_PRACTICE = "good_practice"
    BEST_PRACTICE = "best_practice"
    NZEB = "nearly_zero_energy"
    CUSTOM = "custom"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# End-use split ratios by building type (fraction of total energy)
# Source: CIBSE Guide F (2012), ASHRAE Standard 100-2018
END_USE_SPLITS: Dict[str, Dict[str, float]] = {
    "office": {
        "heating": 0.30, "cooling": 0.15, "ventilation": 0.10, "lighting": 0.22,
        "plug_loads": 0.12, "domestic_hot_water": 0.05, "lifts_escalators": 0.03,
        "it_equipment": 0.03,
    },
    "retail": {
        "heating": 0.22, "cooling": 0.18, "ventilation": 0.08, "lighting": 0.30,
        "plug_loads": 0.05, "domestic_hot_water": 0.02, "refrigeration": 0.12,
        "other": 0.03,
    },
    "hotel": {
        "heating": 0.35, "cooling": 0.10, "ventilation": 0.08, "lighting": 0.15,
        "plug_loads": 0.05, "domestic_hot_water": 0.18, "catering": 0.06,
        "lifts_escalators": 0.03,
    },
    "hospital": {
        "heating": 0.40, "cooling": 0.10, "ventilation": 0.15, "lighting": 0.12,
        "plug_loads": 0.05, "domestic_hot_water": 0.08, "process": 0.05,
        "lifts_escalators": 0.02, "other": 0.03,
    },
    "school": {
        "heating": 0.55, "cooling": 0.05, "ventilation": 0.08, "lighting": 0.18,
        "plug_loads": 0.05, "domestic_hot_water": 0.06, "catering": 0.03,
    },
    "warehouse": {
        "heating": 0.40, "cooling": 0.05, "ventilation": 0.05, "lighting": 0.30,
        "plug_loads": 0.05, "domestic_hot_water": 0.02, "other": 0.13,
    },
    "industrial": {
        "heating": 0.15, "cooling": 0.05, "ventilation": 0.05, "lighting": 0.10,
        "plug_loads": 0.03, "process": 0.55, "domestic_hot_water": 0.02,
        "other": 0.05,
    },
    "data_centre": {
        "cooling": 0.38, "it_equipment": 0.45, "lighting": 0.03,
        "ventilation": 0.05, "plug_loads": 0.02, "other": 0.07,
    },
}

# Good practice end-use EUI targets (kWh/m2/yr) by building type
GOOD_PRACTICE_END_USE_EUI: Dict[str, Dict[str, float]] = {
    "office": {
        "heating": 40.0, "cooling": 18.0, "ventilation": 12.0, "lighting": 14.0,
        "plug_loads": 8.0, "domestic_hot_water": 6.0, "lifts_escalators": 3.0,
        "it_equipment": 4.0,
    },
    "retail": {
        "heating": 30.0, "cooling": 22.0, "ventilation": 10.0, "lighting": 30.0,
        "plug_loads": 5.0, "domestic_hot_water": 2.0, "refrigeration": 15.0,
        "other": 3.0,
    },
    "hotel": {
        "heating": 50.0, "cooling": 15.0, "ventilation": 12.0, "lighting": 18.0,
        "plug_loads": 6.0, "domestic_hot_water": 25.0, "catering": 8.0,
        "lifts_escalators": 4.0,
    },
    "hospital": {
        "heating": 100.0, "cooling": 25.0, "ventilation": 40.0, "lighting": 20.0,
        "plug_loads": 12.0, "domestic_hot_water": 20.0, "process": 15.0,
        "lifts_escalators": 5.0, "other": 8.0,
    },
    "school": {
        "heating": 40.0, "cooling": 4.0, "ventilation": 6.0, "lighting": 10.0,
        "plug_loads": 3.0, "domestic_hot_water": 4.0, "catering": 2.0,
    },
    "warehouse": {
        "heating": 12.0, "cooling": 2.0, "ventilation": 2.0, "lighting": 8.0,
        "plug_loads": 2.0, "domestic_hot_water": 1.0, "other": 4.0,
    },
    "industrial": {
        "heating": 18.0, "cooling": 6.0, "ventilation": 6.0, "lighting": 10.0,
        "plug_loads": 3.0, "process": 60.0, "domestic_hot_water": 2.0,
        "other": 6.0,
    },
    "data_centre": {
        "cooling": 100.0, "it_equipment": 140.0, "lighting": 5.0,
        "ventilation": 12.0, "plug_loads": 3.0, "other": 15.0,
    },
}

DEFAULT_EMISSION_FACTOR = 0.207  # kgCO2e/kWh (UK grid)


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


class SubMeterData(BaseModel):
    """Sub-meter data for end-use disaggregation."""

    end_use: EndUseCategory = Field(default=EndUseCategory.OTHER)
    annual_consumption_kwh: float = Field(default=0.0, ge=0.0)
    data_source: str = Field(default="estimated", description="metered|estimated|calculated")
    confidence_pct: float = Field(default=80.0, ge=0.0, le=100.0)


class PerformanceGapInput(BaseModel):
    """Input data model for PerformanceGapWorkflow."""

    facility_id: str = Field(default="", description="Facility identifier")
    facility_name: str = Field(default="", description="Facility name")
    building_type: BuildingType = Field(default=BuildingType.OFFICE)
    floor_area_m2: float = Field(default=0.0, ge=0.0)
    annual_consumption_kwh: float = Field(default=0.0, ge=0.0, description="Total annual kWh")
    annual_cost: float = Field(default=0.0, ge=0.0, description="Annual energy cost")
    energy_data: List[Dict[str, Any]] = Field(default_factory=list, description="Monthly energy data")
    sub_meter_data: List[SubMeterData] = Field(default_factory=list)
    benchmark_target: BenchmarkTarget = Field(default=BenchmarkTarget.GOOD_PRACTICE)
    custom_target_eui: float = Field(default=0.0, ge=0.0, description="Custom target EUI if CUSTOM")
    emission_factor: float = Field(default=0.207, ge=0.0, description="kgCO2e/kWh")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class EndUseGap(BaseModel):
    """Gap analysis result for a single end use."""

    end_use: str = Field(default="", description="End use category")
    actual_eui: float = Field(default=0.0, ge=0.0, description="Actual kWh/m2/yr")
    target_eui: float = Field(default=0.0, ge=0.0, description="Target kWh/m2/yr")
    gap_eui: float = Field(default=0.0, description="Gap kWh/m2/yr (positive = over-consuming)")
    gap_pct: float = Field(default=0.0, description="Gap %")
    savings_potential_kwh: float = Field(default=0.0, ge=0.0)
    savings_potential_cost: float = Field(default=0.0, ge=0.0)
    co2_reduction_kg: float = Field(default=0.0, ge=0.0)
    severity: GapSeverity = Field(default=GapSeverity.AT_BENCHMARK)
    data_source: str = Field(default="estimated")
    priority_rank: int = Field(default=0, ge=0)


class PerformanceGapResult(BaseModel):
    """Complete result from performance gap workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="performance_gap")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    facility_id: str = Field(default="")
    overall_gap: Dict[str, Any] = Field(default_factory=dict)
    end_use_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    improvement_priorities: List[Dict[str, Any]] = Field(default_factory=list)
    savings_potential: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class PerformanceGapWorkflow:
    """
    3-phase end-use disaggregated performance gap analysis workflow.

    Performs benchmark establishment, end-use disaggregation using
    sub-metering or CIBSE split ratios, and gap analysis with
    savings potential quantification by end use.

    Zero-hallucination: end-use splits from CIBSE Guide F / ASHRAE,
    gap calculations are simple difference formulas, savings potential
    uses published good practice targets. No LLM in numeric path.

    Attributes:
        workflow_id: Unique execution identifier.
        _site_eui: Facility site EUI.
        _end_use_breakdown: End-use consumption breakdown.
        _end_use_gaps: Gap analysis per end use.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = PerformanceGapWorkflow()
        >>> inp = PerformanceGapInput(
        ...     facility_id="fac-001", building_type=BuildingType.OFFICE,
        ...     floor_area_m2=5000, annual_consumption_kwh=1000000,
        ... )
        >>> result = wf.run(inp)
        >>> assert result.overall_gap.get("gap_eui", 0) >= 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PerformanceGapWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._site_eui: float = 0.0
        self._target_eui: float = 0.0
        self._end_use_breakdown: Dict[str, float] = {}
        self._end_use_gaps: List[EndUseGap] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: PerformanceGapInput) -> PerformanceGapResult:
        """
        Execute the 3-phase performance gap workflow.

        Args:
            input_data: Validated performance gap input.

        Returns:
            PerformanceGapResult with overall gap, end-use gaps, and priorities.
        """
        t_start = time.perf_counter()
        self.logger.info(
            "Starting performance gap workflow %s for facility=%s type=%s",
            self.workflow_id, input_data.facility_id, input_data.building_type.value,
        )

        self._phase_results = []
        self._site_eui = 0.0
        self._target_eui = 0.0
        self._end_use_breakdown = {}
        self._end_use_gaps = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = self._phase_benchmark_establishment(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_end_use_disaggregation(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_gap_analysis_report(input_data)
            self._phase_results.append(phase3)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Performance gap workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t_start

        # Overall gap
        overall_gap_eui = self._site_eui - self._target_eui
        overall_gap_pct = (overall_gap_eui / self._target_eui * 100.0) if self._target_eui > 0 else 0.0
        total_savings_kwh = sum(g.savings_potential_kwh for g in self._end_use_gaps)
        total_savings_cost = sum(g.savings_potential_cost for g in self._end_use_gaps)
        total_co2_kg = sum(g.co2_reduction_kg for g in self._end_use_gaps)

        result = PerformanceGapResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            facility_id=input_data.facility_id,
            overall_gap={
                "site_eui": round(self._site_eui, 2),
                "target_eui": round(self._target_eui, 2),
                "gap_eui": round(overall_gap_eui, 2),
                "gap_pct": round(overall_gap_pct, 2),
                "benchmark_target": input_data.benchmark_target.value,
            },
            end_use_gaps=[g.model_dump() for g in self._end_use_gaps],
            improvement_priorities=self._generate_priorities(input_data),
            savings_potential={
                "total_savings_kwh": round(total_savings_kwh, 0),
                "total_savings_cost": round(total_savings_cost, 2),
                "total_co2_reduction_kg": round(total_co2_kg, 2),
                "savings_pct_of_total": round(total_savings_kwh / input_data.annual_consumption_kwh * 100.0, 2) if input_data.annual_consumption_kwh > 0 else 0.0,
            },
            duration_seconds=round(elapsed, 4),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Performance gap workflow %s completed in %.2fs gap=%.1f%%",
            self.workflow_id, elapsed, overall_gap_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Benchmark Establishment
    # -------------------------------------------------------------------------

    def _phase_benchmark_establishment(
        self, input_data: PerformanceGapInput
    ) -> PhaseResult:
        """Establish facility EUI and determine target benchmark."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if input_data.floor_area_m2 <= 0:
            warnings.append("Floor area is zero; EUI will be unreliable")
            self._site_eui = 0.0
        else:
            self._site_eui = input_data.annual_consumption_kwh / input_data.floor_area_m2

        # Determine target EUI
        bt = input_data.building_type.value
        if input_data.benchmark_target == BenchmarkTarget.CUSTOM and input_data.custom_target_eui > 0:
            self._target_eui = input_data.custom_target_eui
        elif input_data.benchmark_target == BenchmarkTarget.GOOD_PRACTICE:
            targets = GOOD_PRACTICE_END_USE_EUI.get(bt, {})
            self._target_eui = sum(targets.values())
        elif input_data.benchmark_target == BenchmarkTarget.BEST_PRACTICE:
            targets = GOOD_PRACTICE_END_USE_EUI.get(bt, {})
            self._target_eui = sum(targets.values()) * 0.70  # 30% below good practice
        elif input_data.benchmark_target == BenchmarkTarget.NZEB:
            targets = GOOD_PRACTICE_END_USE_EUI.get(bt, {})
            self._target_eui = sum(targets.values()) * 0.40  # 60% below good practice
        else:
            # TYPICAL
            splits = END_USE_SPLITS.get(bt, END_USE_SPLITS["office"])
            targets = GOOD_PRACTICE_END_USE_EUI.get(bt, {})
            self._target_eui = sum(targets.values()) * 1.40  # 40% above good practice

        overall_gap = self._site_eui - self._target_eui
        gap_pct = (overall_gap / self._target_eui * 100.0) if self._target_eui > 0 else 0.0

        outputs["site_eui"] = round(self._site_eui, 2)
        outputs["target_eui"] = round(self._target_eui, 2)
        outputs["overall_gap_eui"] = round(overall_gap, 2)
        outputs["overall_gap_pct"] = round(gap_pct, 2)
        outputs["benchmark_target"] = input_data.benchmark_target.value
        outputs["building_type"] = bt

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 1 BenchmarkEstablishment: site=%.1f target=%.1f gap=%.1f%%",
            self._site_eui, self._target_eui, gap_pct,
        )
        return PhaseResult(
            phase_name="benchmark_establishment", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: End-Use Disaggregation
    # -------------------------------------------------------------------------

    def _phase_end_use_disaggregation(
        self, input_data: PerformanceGapInput
    ) -> PhaseResult:
        """Disaggregate consumption by end use using sub-meters or split ratios."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        bt = input_data.building_type.value
        area = input_data.floor_area_m2

        # Check for sub-meter data
        if input_data.sub_meter_data:
            metered_total = sum(sm.annual_consumption_kwh for sm in input_data.sub_meter_data)
            for sm in input_data.sub_meter_data:
                eu = sm.end_use.value
                eui = sm.annual_consumption_kwh / area if area > 0 else 0.0
                self._end_use_breakdown[eu] = round(eui, 2)

            # Check coverage
            coverage_pct = (metered_total / input_data.annual_consumption_kwh * 100.0) if input_data.annual_consumption_kwh > 0 else 0.0
            if coverage_pct < 80:
                warnings.append(
                    f"Sub-meter coverage is {coverage_pct:.0f}%; supplementing with estimates"
                )
                # Fill remaining with split ratios
                remainder_kwh = input_data.annual_consumption_kwh - metered_total
                splits = END_USE_SPLITS.get(bt, END_USE_SPLITS["office"])
                metered_uses = {sm.end_use.value for sm in input_data.sub_meter_data}
                unmetered_splits = {k: v for k, v in splits.items() if k not in metered_uses}
                total_unmetered_weight = sum(unmetered_splits.values())
                if total_unmetered_weight > 0:
                    for eu, weight in unmetered_splits.items():
                        allocated = remainder_kwh * weight / total_unmetered_weight
                        eui = allocated / area if area > 0 else 0.0
                        self._end_use_breakdown[eu] = round(eui, 2)

            outputs["disaggregation_method"] = "sub_metered"
            outputs["coverage_pct"] = round(coverage_pct, 1)
        else:
            # Use published split ratios (zero-hallucination)
            splits = END_USE_SPLITS.get(bt, END_USE_SPLITS["office"])
            for eu, fraction in splits.items():
                eui = self._site_eui * fraction
                self._end_use_breakdown[eu] = round(eui, 2)

            warnings.append("No sub-meter data; using CIBSE Guide F split ratios")
            outputs["disaggregation_method"] = "split_ratios"
            outputs["split_source"] = "CIBSE Guide F / ASHRAE"

        outputs["end_use_count"] = len(self._end_use_breakdown)
        outputs["end_use_eui"] = dict(self._end_use_breakdown)
        outputs["total_disaggregated_eui"] = round(sum(self._end_use_breakdown.values()), 2)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 2 EndUseDisaggregation: %d end uses, total=%.1f kWh/m2",
            len(self._end_use_breakdown), sum(self._end_use_breakdown.values()),
        )
        return PhaseResult(
            phase_name="end_use_disaggregation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Gap Analysis Report
    # -------------------------------------------------------------------------

    def _phase_gap_analysis_report(
        self, input_data: PerformanceGapInput
    ) -> PhaseResult:
        """Identify gaps by end use, rank priorities, quantify savings."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        bt = input_data.building_type.value
        area = input_data.floor_area_m2
        cost_per_kwh = input_data.annual_cost / input_data.annual_consumption_kwh if input_data.annual_consumption_kwh > 0 else 0.15
        ef = input_data.emission_factor

        # Get target end-use EUIs
        target_method = input_data.benchmark_target
        targets = GOOD_PRACTICE_END_USE_EUI.get(bt, {})
        scale_factor = 1.0
        if target_method == BenchmarkTarget.BEST_PRACTICE:
            scale_factor = 0.70
        elif target_method == BenchmarkTarget.NZEB:
            scale_factor = 0.40
        elif target_method == BenchmarkTarget.TYPICAL:
            scale_factor = 1.40

        # Calculate gap per end use
        for eu, actual_eui in self._end_use_breakdown.items():
            target_eui = targets.get(eu, actual_eui) * scale_factor
            gap_eui = actual_eui - target_eui
            gap_pct = (gap_eui / target_eui * 100.0) if target_eui > 0 else 0.0
            savings_kwh = max(0.0, gap_eui * area)
            savings_cost = savings_kwh * cost_per_kwh
            co2_kg = savings_kwh * ef

            # Determine severity
            if gap_pct > 50:
                severity = GapSeverity.CRITICAL
            elif gap_pct > 25:
                severity = GapSeverity.SIGNIFICANT
            elif gap_pct > 10:
                severity = GapSeverity.MODERATE
            elif gap_pct > 0:
                severity = GapSeverity.MINOR
            else:
                severity = GapSeverity.AT_BENCHMARK

            # Determine data source
            metered_uses = {sm.end_use.value for sm in input_data.sub_meter_data}
            data_source = "metered" if eu in metered_uses else "estimated"

            self._end_use_gaps.append(EndUseGap(
                end_use=eu,
                actual_eui=round(actual_eui, 2),
                target_eui=round(target_eui, 2),
                gap_eui=round(gap_eui, 2),
                gap_pct=round(gap_pct, 2),
                savings_potential_kwh=round(savings_kwh, 0),
                savings_potential_cost=round(savings_cost, 2),
                co2_reduction_kg=round(co2_kg, 2),
                severity=severity,
                data_source=data_source,
            ))

        # Rank by savings potential
        self._end_use_gaps.sort(key=lambda g: g.savings_potential_kwh, reverse=True)
        for idx, gap in enumerate(self._end_use_gaps, start=1):
            gap.priority_rank = idx

        # Summary
        critical_count = sum(1 for g in self._end_use_gaps if g.severity == GapSeverity.CRITICAL)
        significant_count = sum(1 for g in self._end_use_gaps if g.severity == GapSeverity.SIGNIFICANT)

        outputs["end_uses_analysed"] = len(self._end_use_gaps)
        outputs["critical_gaps"] = critical_count
        outputs["significant_gaps"] = significant_count
        outputs["total_savings_kwh"] = round(sum(g.savings_potential_kwh for g in self._end_use_gaps), 0)
        outputs["total_savings_cost"] = round(sum(g.savings_potential_cost for g in self._end_use_gaps), 2)
        outputs["top_3_priorities"] = [
            {"end_use": g.end_use, "gap_pct": g.gap_pct, "savings_kwh": g.savings_potential_kwh}
            for g in self._end_use_gaps[:3]
        ]

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 3 GapAnalysisReport: %d gaps, %d critical, savings=%.0f kWh",
            len(self._end_use_gaps), critical_count, outputs["total_savings_kwh"],
        )
        return PhaseResult(
            phase_name="gap_analysis_report", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Supporting Methods
    # -------------------------------------------------------------------------

    def _generate_priorities(self, input_data: PerformanceGapInput) -> List[Dict[str, Any]]:
        """Generate improvement priorities from gap analysis."""
        priorities: List[Dict[str, Any]] = []
        for gap in self._end_use_gaps:
            if gap.savings_potential_kwh <= 0:
                continue
            priorities.append({
                "rank": gap.priority_rank,
                "end_use": gap.end_use,
                "severity": gap.severity.value,
                "gap_pct": gap.gap_pct,
                "savings_kwh": gap.savings_potential_kwh,
                "savings_cost": gap.savings_potential_cost,
                "co2_reduction_kg": gap.co2_reduction_kg,
                "recommendation": self._recommend_action(gap.end_use, gap.gap_pct),
            })
        return priorities

    def _recommend_action(self, end_use: str, gap_pct: float) -> str:
        """Generate deterministic recommendation based on end use and gap severity."""
        recommendations: Dict[str, str] = {
            "heating": "Upgrade heating controls, improve insulation, consider heat pump",
            "cooling": "Optimise chiller staging, improve free cooling, reduce solar gains",
            "ventilation": "Implement demand-controlled ventilation, check fan efficiency",
            "lighting": "LED retrofit, daylight dimming, occupancy sensing controls",
            "plug_loads": "Smart power strips, equipment scheduling, efficiency standards",
            "domestic_hot_water": "Point-of-use heaters, pipe insulation, solar thermal",
            "process": "Process optimisation, waste heat recovery, variable speed drives",
            "it_equipment": "Server virtualisation, hot/cold aisle containment, UPS upgrade",
            "refrigeration": "Door heater controls, EC fan motors, defrost optimisation",
            "catering": "Efficient cooking equipment, ventilation heat recovery",
            "lifts_escalators": "Regenerative drives, standby mode, modernisation",
        }
        base = recommendations.get(end_use, "Investigate and implement best practice measures")
        if gap_pct > 50:
            return f"URGENT: {base}. Gap exceeds 50% of benchmark."
        return base

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: PerformanceGapResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
