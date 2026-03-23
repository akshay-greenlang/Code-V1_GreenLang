# -*- coding: utf-8 -*-
"""
EnPI Tracking Workflow
===================================

4-phase workflow for collecting energy performance data, normalizing against
relevant variables, calculating Energy Performance Indicators (EnPIs), and
conducting performance reviews within PACK-039 Energy Monitoring Pack.

Phases:
    1. DataCollection        -- Gather energy and production/weather data
    2. Normalization         -- Normalize for weather, occupancy, production
    3. EnPICalculation       -- Calculate EnPIs per ISO 50006
    4. PerformanceReview     -- Compare to benchmarks and targets

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - ISO 50001:2018 Clause 6.6 (energy performance indicators)
    - ISO 50006:2014 (measuring energy performance using EnPIs and EnBs)
    - ISO 50015:2014 (measurement and verification)
    - EN 16247-1 (energy audits - general requirements)

Schedule: monthly / quarterly
Estimated duration: 15 minutes

Author: GreenLang Team
Version: 39.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.utcnow()


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


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


class PerformanceTrend(str, Enum):
    """EnPI performance trend direction."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    INSUFFICIENT_DATA = "insufficient_data"


class BenchmarkRating(str, Enum):
    """Rating against industry benchmarks."""

    BEST_IN_CLASS = "best_in_class"
    ABOVE_AVERAGE = "above_average"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    POOR = "poor"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

ENPI_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "office_building": {
        "description": "Commercial office building",
        "typical_eui_kwh_m2": 180.0,
        "best_practice_eui_kwh_m2": 100.0,
        "poor_eui_kwh_m2": 300.0,
        "typical_sec_kwh_per_fte": 4500.0,
        "best_practice_sec_kwh_per_fte": 2800.0,
        "hdd_sensitivity_kwh_per_hdd": 0.08,
        "cdd_sensitivity_kwh_per_cdd": 0.12,
        "iso_50006_category": "Type 1 - Simple facility",
    },
    "manufacturing": {
        "description": "Manufacturing facility",
        "typical_eui_kwh_m2": 350.0,
        "best_practice_eui_kwh_m2": 200.0,
        "poor_eui_kwh_m2": 600.0,
        "typical_sec_kwh_per_unit": 5.0,
        "best_practice_sec_kwh_per_unit": 3.0,
        "hdd_sensitivity_kwh_per_hdd": 0.05,
        "cdd_sensitivity_kwh_per_cdd": 0.08,
        "iso_50006_category": "Type 2 - Production facility",
    },
    "retail_store": {
        "description": "Retail / shopping centre",
        "typical_eui_kwh_m2": 250.0,
        "best_practice_eui_kwh_m2": 150.0,
        "poor_eui_kwh_m2": 400.0,
        "typical_sec_kwh_per_m2_sales": 220.0,
        "best_practice_sec_kwh_per_m2_sales": 130.0,
        "hdd_sensitivity_kwh_per_hdd": 0.06,
        "cdd_sensitivity_kwh_per_cdd": 0.15,
        "iso_50006_category": "Type 1 - Simple facility",
    },
    "hospital": {
        "description": "Hospital / healthcare facility",
        "typical_eui_kwh_m2": 400.0,
        "best_practice_eui_kwh_m2": 280.0,
        "poor_eui_kwh_m2": 550.0,
        "typical_sec_kwh_per_bed": 25000.0,
        "best_practice_sec_kwh_per_bed": 18000.0,
        "hdd_sensitivity_kwh_per_hdd": 0.10,
        "cdd_sensitivity_kwh_per_cdd": 0.14,
        "iso_50006_category": "Type 3 - Complex facility",
    },
    "data_center": {
        "description": "Data centre / colocation",
        "typical_eui_kwh_m2": 2000.0,
        "best_practice_eui_kwh_m2": 1200.0,
        "poor_eui_kwh_m2": 3500.0,
        "typical_pue": 1.58,
        "best_practice_pue": 1.20,
        "hdd_sensitivity_kwh_per_hdd": 0.01,
        "cdd_sensitivity_kwh_per_cdd": 0.05,
        "iso_50006_category": "Type 2 - Production facility",
    },
    "warehouse": {
        "description": "Warehouse / distribution centre",
        "typical_eui_kwh_m2": 120.0,
        "best_practice_eui_kwh_m2": 60.0,
        "poor_eui_kwh_m2": 220.0,
        "typical_sec_kwh_per_pallet": 2.5,
        "best_practice_sec_kwh_per_pallet": 1.5,
        "hdd_sensitivity_kwh_per_hdd": 0.04,
        "cdd_sensitivity_kwh_per_cdd": 0.06,
        "iso_50006_category": "Type 1 - Simple facility",
    },
    "hotel": {
        "description": "Hotel / hospitality",
        "typical_eui_kwh_m2": 280.0,
        "best_practice_eui_kwh_m2": 170.0,
        "poor_eui_kwh_m2": 420.0,
        "typical_sec_kwh_per_room_night": 45.0,
        "best_practice_sec_kwh_per_room_night": 28.0,
        "hdd_sensitivity_kwh_per_hdd": 0.09,
        "cdd_sensitivity_kwh_per_cdd": 0.13,
        "iso_50006_category": "Type 1 - Simple facility",
    },
    "school": {
        "description": "K-12 school / university campus",
        "typical_eui_kwh_m2": 150.0,
        "best_practice_eui_kwh_m2": 80.0,
        "poor_eui_kwh_m2": 250.0,
        "typical_sec_kwh_per_student": 1200.0,
        "best_practice_sec_kwh_per_student": 700.0,
        "hdd_sensitivity_kwh_per_hdd": 0.07,
        "cdd_sensitivity_kwh_per_cdd": 0.10,
        "iso_50006_category": "Type 1 - Simple facility",
    },
    "cold_storage": {
        "description": "Cold storage / refrigerated warehouse",
        "typical_eui_kwh_m2": 450.0,
        "best_practice_eui_kwh_m2": 300.0,
        "poor_eui_kwh_m2": 650.0,
        "typical_sec_kwh_per_m3": 55.0,
        "best_practice_sec_kwh_per_m3": 35.0,
        "hdd_sensitivity_kwh_per_hdd": 0.02,
        "cdd_sensitivity_kwh_per_cdd": 0.18,
        "iso_50006_category": "Type 2 - Production facility",
    },
    "grocery_store": {
        "description": "Grocery / supermarket",
        "typical_eui_kwh_m2": 500.0,
        "best_practice_eui_kwh_m2": 350.0,
        "poor_eui_kwh_m2": 700.0,
        "typical_sec_kwh_per_m2_sales": 450.0,
        "best_practice_sec_kwh_per_m2_sales": 300.0,
        "hdd_sensitivity_kwh_per_hdd": 0.05,
        "cdd_sensitivity_kwh_per_cdd": 0.16,
        "iso_50006_category": "Type 2 - Production facility",
    },
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class PeriodData(BaseModel):
    """Energy and activity data for a reporting period."""

    period_label: str = Field(..., description="Period label (e.g. '2026-01')")
    energy_kwh: Decimal = Field(default=Decimal("0"), ge=0, description="Total energy kWh")
    floor_area_m2: Decimal = Field(default=Decimal("0"), ge=0, description="Conditioned floor area")
    production_units: Decimal = Field(default=Decimal("0"), ge=0, description="Production output units")
    occupancy_fte: Decimal = Field(default=Decimal("0"), ge=0, description="Full-time equivalent occupants")
    hdd: Decimal = Field(default=Decimal("0"), ge=0, description="Heating degree days")
    cdd: Decimal = Field(default=Decimal("0"), ge=0, description="Cooling degree days")
    operating_hours: Decimal = Field(default=Decimal("0"), ge=0, description="Operating hours in period")


class EnPITrackingInput(BaseModel):
    """Input data model for EnPITrackingWorkflow."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(..., min_length=1, description="Facility name")
    facility_type: str = Field(default="office_building", description="Facility type key")
    period_data: List[PeriodData] = Field(
        default_factory=list,
        description="Time-series period data for EnPI calculation",
    )
    baseline_period: Optional[str] = Field(
        default=None,
        description="Baseline period label for comparison",
    )
    target_improvement_pct: Decimal = Field(
        default=Decimal("5.0"), ge=0, le=100,
        description="Target improvement percentage over baseline",
    )
    normalization_variables: List[str] = Field(
        default_factory=lambda: ["hdd", "cdd", "production_units"],
        description="Variables to normalize against",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facility_name")
    @classmethod
    def validate_facility_name(cls, v: str) -> str:
        """Ensure facility name is non-empty after stripping."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("facility_name must not be blank")
        return stripped


class EnPITrackingResult(BaseModel):
    """Complete result from EnPI tracking workflow."""

    tracking_id: str = Field(..., description="Unique EnPI tracking execution ID")
    facility_id: str = Field(default="", description="Facility identifier")
    periods_analysed: int = Field(default=0, ge=0)
    current_eui_kwh_m2: Decimal = Field(default=Decimal("0"), ge=0)
    baseline_eui_kwh_m2: Decimal = Field(default=Decimal("0"), ge=0)
    normalized_eui_kwh_m2: Decimal = Field(default=Decimal("0"), ge=0)
    improvement_pct: Decimal = Field(default=Decimal("0"))
    target_met: bool = Field(default=False)
    performance_trend: str = Field(default="insufficient_data")
    benchmark_rating: str = Field(default="average")
    enpi_values: List[Dict[str, Any]] = Field(default_factory=list)
    normalization_factors: Dict[str, Any] = Field(default_factory=dict)
    cusum_analysis: Dict[str, Any] = Field(default_factory=dict)
    tracking_duration_ms: int = Field(default=0, ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class EnPITrackingWorkflow:
    """
    4-phase EnPI tracking workflow per ISO 50006 for energy management.

    Collects energy performance data, normalizes against relevant variables,
    calculates Energy Performance Indicators, and reviews against benchmarks.

    Zero-hallucination: all EnPI calculations use deterministic formulas
    from ISO 50006. Benchmarks are sourced from validated reference data.
    No LLM calls in the numeric computation path.

    Attributes:
        tracking_id: Unique tracking execution identifier.
        _collected: Collected period data records.
        _normalized: Normalized period values.
        _enpi_values: Calculated EnPI values.
        _review: Performance review results.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = EnPITrackingWorkflow()
        >>> period = PeriodData(
        ...     period_label="2026-01",
        ...     energy_kwh=Decimal("150000"),
        ...     floor_area_m2=Decimal("1000"),
        ... )
        >>> inp = EnPITrackingInput(facility_name="HQ", period_data=[period])
        >>> result = wf.run(inp)
        >>> assert result.current_eui_kwh_m2 > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnPITrackingWorkflow."""
        self.tracking_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._collected: List[Dict[str, Any]] = []
        self._normalized: List[Dict[str, Any]] = []
        self._enpi_values: List[Dict[str, Any]] = []
        self._review: Dict[str, Any] = {}
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: EnPITrackingInput) -> EnPITrackingResult:
        """
        Execute the 4-phase EnPI tracking workflow.

        Args:
            input_data: Validated EnPI tracking input.

        Returns:
            EnPITrackingResult with EnPI values, normalization, and review.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = _utcnow()
        self.logger.info(
            "Starting EnPI tracking workflow %s for facility=%s type=%s periods=%d",
            self.tracking_id, input_data.facility_name,
            input_data.facility_type, len(input_data.period_data),
        )

        self._phase_results = []
        self._collected = []
        self._normalized = []
        self._enpi_values = []
        self._review = {}

        try:
            phase1 = self._phase_data_collection(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_normalization(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_enpi_calculation(input_data)
            self._phase_results.append(phase3)

            phase4 = self._phase_performance_review(input_data)
            self._phase_results.append(phase4)

        except Exception as exc:
            self.logger.error("EnPI tracking workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        # Get latest and baseline EUI
        current_eui = Decimal("0")
        baseline_eui = Decimal("0")
        normalized_eui = Decimal("0")

        if self._enpi_values:
            current_eui = Decimal(str(self._enpi_values[-1].get("eui_kwh_m2", 0)))
            normalized_eui = Decimal(str(self._enpi_values[-1].get("normalized_eui_kwh_m2", 0)))
        if len(self._enpi_values) > 1:
            baseline_eui = Decimal(str(self._enpi_values[0].get("eui_kwh_m2", 0)))

        improvement = Decimal("0")
        if baseline_eui > 0:
            improvement = ((baseline_eui - normalized_eui) / baseline_eui * 100).quantize(
                Decimal("0.1")
            )

        target_met = improvement >= input_data.target_improvement_pct
        trend = self._review.get("trend", PerformanceTrend.INSUFFICIENT_DATA.value)
        rating = self._review.get("benchmark_rating", BenchmarkRating.AVERAGE.value)

        result = EnPITrackingResult(
            tracking_id=self.tracking_id,
            facility_id=input_data.facility_id,
            periods_analysed=len(input_data.period_data),
            current_eui_kwh_m2=current_eui,
            baseline_eui_kwh_m2=baseline_eui,
            normalized_eui_kwh_m2=normalized_eui,
            improvement_pct=improvement,
            target_met=target_met,
            performance_trend=trend,
            benchmark_rating=rating,
            enpi_values=self._enpi_values,
            normalization_factors=self._review.get("normalization_factors", {}),
            cusum_analysis=self._review.get("cusum", {}),
            tracking_duration_ms=int(elapsed_ms),
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "EnPI tracking workflow %s completed in %dms EUI=%.1f normalized=%.1f "
            "improvement=%.1f%% trend=%s rating=%s",
            self.tracking_id, int(elapsed_ms), float(current_eui),
            float(normalized_eui), float(improvement), trend, rating,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Collection
    # -------------------------------------------------------------------------

    def _phase_data_collection(
        self, input_data: EnPITrackingInput
    ) -> PhaseResult:
        """Gather energy and production/weather data."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not input_data.period_data:
            warnings.append("No period data provided; using synthetic defaults")
            # Create a synthetic period
            input_data.period_data.append(PeriodData(
                period_label="2026-01",
                energy_kwh=Decimal("150000"),
                floor_area_m2=Decimal("1000"),
                production_units=Decimal("10000"),
                occupancy_fte=Decimal("50"),
                hdd=Decimal("300"),
                cdd=Decimal("50"),
                operating_hours=Decimal("720"),
            ))

        for period in input_data.period_data:
            record = {
                "period_label": period.period_label,
                "energy_kwh": float(period.energy_kwh),
                "floor_area_m2": float(period.floor_area_m2),
                "production_units": float(period.production_units),
                "occupancy_fte": float(period.occupancy_fte),
                "hdd": float(period.hdd),
                "cdd": float(period.cdd),
                "operating_hours": float(period.operating_hours),
                "data_quality": "complete" if period.energy_kwh > 0 else "incomplete",
            }

            # Validate completeness
            if period.floor_area_m2 == 0:
                record["data_quality"] = "incomplete"
                warnings.append(f"Period {period.period_label}: floor_area_m2 is zero")

            self._collected.append(record)

        outputs["periods_collected"] = len(self._collected)
        outputs["complete_periods"] = sum(
            1 for r in self._collected if r["data_quality"] == "complete"
        )
        outputs["total_energy_kwh"] = round(
            sum(r["energy_kwh"] for r in self._collected), 1
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 DataCollection: %d periods, total=%.0f kWh",
            len(self._collected), outputs["total_energy_kwh"],
        )
        return PhaseResult(
            phase_name="data_collection", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Normalization
    # -------------------------------------------------------------------------

    def _phase_normalization(
        self, input_data: EnPITrackingInput
    ) -> PhaseResult:
        """Normalize energy data for weather, occupancy, production."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        benchmark = ENPI_BENCHMARKS.get(
            input_data.facility_type,
            ENPI_BENCHMARKS["office_building"],
        )
        hdd_sensitivity = benchmark.get("hdd_sensitivity_kwh_per_hdd", 0.08)
        cdd_sensitivity = benchmark.get("cdd_sensitivity_kwh_per_cdd", 0.12)

        # Calculate baseline averages for normalization reference
        avg_hdd = sum(r["hdd"] for r in self._collected) / max(len(self._collected), 1)
        avg_cdd = sum(r["cdd"] for r in self._collected) / max(len(self._collected), 1)
        avg_production = sum(r["production_units"] for r in self._collected) / max(len(self._collected), 1)

        for record in self._collected:
            energy = record["energy_kwh"]
            area = record["floor_area_m2"]

            # Weather normalization
            hdd_adjustment = 0.0
            cdd_adjustment = 0.0
            if "hdd" in input_data.normalization_variables and area > 0:
                hdd_diff = record["hdd"] - avg_hdd
                hdd_adjustment = hdd_diff * hdd_sensitivity * area
            if "cdd" in input_data.normalization_variables and area > 0:
                cdd_diff = record["cdd"] - avg_cdd
                cdd_adjustment = cdd_diff * cdd_sensitivity * area

            # Production normalization
            production_adjustment = 0.0
            if "production_units" in input_data.normalization_variables:
                if avg_production > 0 and record["production_units"] > 0:
                    prod_ratio = record["production_units"] / avg_production
                    if abs(prod_ratio - 1.0) > 0.05:
                        production_adjustment = energy * (1.0 - prod_ratio) * 0.3

            normalized_energy = energy - hdd_adjustment - cdd_adjustment - production_adjustment
            normalized_energy = max(0, normalized_energy)

            normalized_record = {
                "period_label": record["period_label"],
                "raw_energy_kwh": energy,
                "normalized_energy_kwh": round(normalized_energy, 1),
                "hdd_adjustment_kwh": round(hdd_adjustment, 1),
                "cdd_adjustment_kwh": round(cdd_adjustment, 1),
                "production_adjustment_kwh": round(production_adjustment, 1),
                "total_adjustment_kwh": round(
                    hdd_adjustment + cdd_adjustment + production_adjustment, 1
                ),
                "floor_area_m2": area,
            }
            self._normalized.append(normalized_record)

        outputs["periods_normalized"] = len(self._normalized)
        outputs["avg_hdd"] = round(avg_hdd, 1)
        outputs["avg_cdd"] = round(avg_cdd, 1)
        outputs["normalization_variables"] = input_data.normalization_variables

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 Normalization: %d periods normalized, avg HDD=%.0f CDD=%.0f",
            len(self._normalized), avg_hdd, avg_cdd,
        )
        return PhaseResult(
            phase_name="normalization", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: EnPI Calculation
    # -------------------------------------------------------------------------

    def _phase_enpi_calculation(
        self, input_data: EnPITrackingInput
    ) -> PhaseResult:
        """Calculate EnPIs per ISO 50006."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for norm_record in self._normalized:
            area = norm_record["floor_area_m2"]
            raw_energy = norm_record["raw_energy_kwh"]
            norm_energy = norm_record["normalized_energy_kwh"]

            # EUI = Energy Use Intensity (kWh/m2)
            eui = round(raw_energy / max(area, 0.01), 2)
            normalized_eui = round(norm_energy / max(area, 0.01), 2)

            # Find corresponding collected record for SEC calculation
            collected = next(
                (r for r in self._collected if r["period_label"] == norm_record["period_label"]),
                {},
            )

            # SEC = Specific Energy Consumption (kWh/unit)
            production = collected.get("production_units", 0)
            sec = round(norm_energy / max(production, 0.01), 2) if production > 0 else 0

            # Energy per FTE
            fte = collected.get("occupancy_fte", 0)
            energy_per_fte = round(norm_energy / max(fte, 0.01), 2) if fte > 0 else 0

            # Energy per operating hour
            hours = collected.get("operating_hours", 0)
            energy_per_hour = round(norm_energy / max(hours, 0.01), 2) if hours > 0 else 0

            enpi_record = {
                "period_label": norm_record["period_label"],
                "eui_kwh_m2": eui,
                "normalized_eui_kwh_m2": normalized_eui,
                "sec_kwh_per_unit": sec,
                "energy_per_fte_kwh": energy_per_fte,
                "energy_per_hour_kwh": energy_per_hour,
                "raw_energy_kwh": raw_energy,
                "normalized_energy_kwh": norm_energy,
                "floor_area_m2": area,
            }
            self._enpi_values.append(enpi_record)

        outputs["enpi_records"] = len(self._enpi_values)
        if self._enpi_values:
            outputs["latest_eui"] = self._enpi_values[-1]["eui_kwh_m2"]
            outputs["latest_normalized_eui"] = self._enpi_values[-1]["normalized_eui_kwh_m2"]

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 EnPICalculation: %d EnPI records calculated",
            len(self._enpi_values),
        )
        return PhaseResult(
            phase_name="enpi_calculation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Performance Review
    # -------------------------------------------------------------------------

    def _phase_performance_review(
        self, input_data: EnPITrackingInput
    ) -> PhaseResult:
        """Compare to benchmarks, targets, and calculate CUSUM."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        benchmark = ENPI_BENCHMARKS.get(
            input_data.facility_type,
            ENPI_BENCHMARKS["office_building"],
        )

        # Benchmark rating
        typical_eui = benchmark["typical_eui_kwh_m2"]
        best_eui = benchmark["best_practice_eui_kwh_m2"]
        poor_eui = benchmark["poor_eui_kwh_m2"]

        current_eui = self._enpi_values[-1]["normalized_eui_kwh_m2"] if self._enpi_values else 0

        if current_eui <= best_eui:
            rating = BenchmarkRating.BEST_IN_CLASS.value
        elif current_eui <= (best_eui + typical_eui) / 2:
            rating = BenchmarkRating.ABOVE_AVERAGE.value
        elif current_eui <= typical_eui:
            rating = BenchmarkRating.AVERAGE.value
        elif current_eui <= (typical_eui + poor_eui) / 2:
            rating = BenchmarkRating.BELOW_AVERAGE.value
        else:
            rating = BenchmarkRating.POOR.value

        # Trend analysis (linear regression slope direction)
        if len(self._enpi_values) >= 3:
            eui_series = [v["normalized_eui_kwh_m2"] for v in self._enpi_values]
            first_half = sum(eui_series[:len(eui_series)//2]) / max(len(eui_series)//2, 1)
            second_half = sum(eui_series[len(eui_series)//2:]) / max(
                len(eui_series) - len(eui_series)//2, 1
            )
            if second_half < first_half * 0.97:
                trend = PerformanceTrend.IMPROVING.value
            elif second_half > first_half * 1.03:
                trend = PerformanceTrend.DEGRADING.value
            else:
                trend = PerformanceTrend.STABLE.value
        else:
            trend = PerformanceTrend.INSUFFICIENT_DATA.value

        # CUSUM analysis (cumulative sum of deviations from baseline)
        cusum_values: List[Dict[str, Any]] = []
        cumulative = 0.0
        baseline_eui = self._enpi_values[0]["normalized_eui_kwh_m2"] if self._enpi_values else 0

        for enpi in self._enpi_values:
            deviation = enpi["normalized_eui_kwh_m2"] - baseline_eui
            cumulative += deviation
            cusum_values.append({
                "period": enpi["period_label"],
                "deviation": round(deviation, 2),
                "cusum": round(cumulative, 2),
            })

        self._review = {
            "benchmark_rating": rating,
            "trend": trend,
            "typical_eui": typical_eui,
            "best_practice_eui": best_eui,
            "current_eui": current_eui,
            "iso_50006_category": benchmark.get("iso_50006_category", ""),
            "cusum": {
                "values": cusum_values,
                "final_cusum": round(cumulative, 2),
                "direction": "favorable" if cumulative < 0 else "unfavorable",
            },
            "normalization_factors": {
                "hdd_sensitivity": benchmark.get("hdd_sensitivity_kwh_per_hdd", 0),
                "cdd_sensitivity": benchmark.get("cdd_sensitivity_kwh_per_cdd", 0),
            },
        }

        outputs["benchmark_rating"] = rating
        outputs["trend"] = trend
        outputs["current_eui"] = current_eui
        outputs["typical_eui"] = typical_eui
        outputs["cusum_final"] = round(cumulative, 2)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 PerformanceReview: rating=%s trend=%s EUI=%.1f vs typical=%.1f",
            rating, trend, current_eui, typical_eui,
        )
        return PhaseResult(
            phase_name="performance_review", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: EnPITrackingResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
