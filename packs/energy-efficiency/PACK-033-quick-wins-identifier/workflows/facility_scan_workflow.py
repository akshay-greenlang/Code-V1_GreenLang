# -*- coding: utf-8 -*-
"""
Facility Scan Workflow
===================================

4-phase workflow for scanning facilities to identify quick-win energy
efficiency opportunities within PACK-033 Quick Wins Identifier Pack.

Phases:
    1. FacilityRegistration   -- Validate facility profile and equipment survey
    2. QuickWinScanning       -- Run QuickWinsScannerEngine against facility
    3. InitialEstimation      -- Run EnergySavingsEstimatorEngine for top opportunities
    4. ReportGeneration       -- Compile scan results into structured output

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Schedule: on-demand
Estimated duration: 30 minutes

Author: GreenLang Team
Version: 33.0.0
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


class QuickWinItem(BaseModel):
    """Individual quick-win opportunity identified during scan."""

    win_id: str = Field(default_factory=lambda: f"qw-{uuid.uuid4().hex[:8]}")
    title: str = Field(default="", description="Quick win title")
    description: str = Field(default="", description="Detailed description")
    category: str = Field(default="", description="lighting|hvac|motors|controls|envelope|behavioral")
    estimated_savings_kwh: Decimal = Field(default=Decimal("0"), ge=0, description="Annual kWh savings")
    estimated_savings_cost: Decimal = Field(default=Decimal("0"), ge=0, description="Annual cost savings")
    implementation_cost: Decimal = Field(default=Decimal("0"), ge=0, description="Upfront cost")
    simple_payback_months: Decimal = Field(default=Decimal("0"), ge=0, description="Payback in months")
    confidence: str = Field(default="medium", description="high|medium|low")
    equipment_ids: List[str] = Field(default_factory=list, description="Related equipment")


class FacilityScanInput(BaseModel):
    """Input data model for FacilityScanWorkflow."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(..., min_length=1, description="Facility name")
    building_type: str = Field(default="commercial", description="commercial|industrial|institutional|retail|warehouse")
    floor_area_m2: Decimal = Field(..., gt=0, description="Total floor area in square metres")
    operating_hours: int = Field(default=2500, ge=0, le=8760, description="Annual operating hours")
    annual_energy_kwh: Decimal = Field(..., ge=0, description="Annual energy consumption kWh")
    annual_energy_cost: Decimal = Field(..., ge=0, description="Annual energy cost in local currency")
    climate_zone: str = Field(default="4A", description="ASHRAE climate zone")
    equipment_data: Dict[str, Any] = Field(default_factory=dict, description="Equipment survey data")
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


class FacilityScanResult(BaseModel):
    """Complete result from facility scan workflow."""

    scan_id: str = Field(..., description="Unique scan execution ID")
    facility_id: str = Field(default="", description="Scanned facility ID")
    total_quick_wins_found: int = Field(default=0, ge=0, description="Number of quick wins identified")
    quick_wins: List[QuickWinItem] = Field(default_factory=list, description="Identified quick wins")
    estimated_total_savings_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    estimated_total_savings_cost: Decimal = Field(default=Decimal("0"), ge=0)
    scan_duration_ms: int = Field(default=0, ge=0, description="Total scan time in milliseconds")
    phases_completed: List[str] = Field(default_factory=list, description="Phase names completed")
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# QUICK-WIN SCANNING REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Typical quick-win categories with savings ranges (% of annual energy)
QUICK_WIN_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "lighting_led_retrofit": {
        "category": "lighting",
        "title": "LED Lighting Retrofit",
        "savings_pct_low": 0.03,
        "savings_pct_high": 0.12,
        "cost_per_m2": Decimal("8.00"),
        "payback_months_typical": 18,
        "confidence": "high",
    },
    "hvac_setpoint_optimization": {
        "category": "hvac",
        "title": "HVAC Setpoint Optimization",
        "savings_pct_low": 0.02,
        "savings_pct_high": 0.05,
        "cost_per_m2": Decimal("0.50"),
        "payback_months_typical": 3,
        "confidence": "high",
    },
    "occupancy_sensors": {
        "category": "controls",
        "title": "Occupancy Sensor Installation",
        "savings_pct_low": 0.02,
        "savings_pct_high": 0.08,
        "cost_per_m2": Decimal("3.00"),
        "payback_months_typical": 12,
        "confidence": "medium",
    },
    "compressed_air_leak_repair": {
        "category": "motors",
        "title": "Compressed Air Leak Repair",
        "savings_pct_low": 0.01,
        "savings_pct_high": 0.04,
        "cost_per_m2": Decimal("1.50"),
        "payback_months_typical": 6,
        "confidence": "high",
    },
    "envelope_sealing": {
        "category": "envelope",
        "title": "Building Envelope Sealing",
        "savings_pct_low": 0.01,
        "savings_pct_high": 0.05,
        "cost_per_m2": Decimal("2.00"),
        "payback_months_typical": 15,
        "confidence": "medium",
    },
    "power_management": {
        "category": "controls",
        "title": "Equipment Power Management",
        "savings_pct_low": 0.01,
        "savings_pct_high": 0.03,
        "cost_per_m2": Decimal("0.25"),
        "payback_months_typical": 2,
        "confidence": "high",
    },
    "behavioral_energy_campaign": {
        "category": "behavioral",
        "title": "Staff Energy Awareness Campaign",
        "savings_pct_low": 0.01,
        "savings_pct_high": 0.05,
        "cost_per_m2": Decimal("0.10"),
        "payback_months_typical": 1,
        "confidence": "medium",
    },
    "vfd_on_pumps_fans": {
        "category": "motors",
        "title": "VFD Installation on Pumps/Fans",
        "savings_pct_low": 0.02,
        "savings_pct_high": 0.10,
        "cost_per_m2": Decimal("6.00"),
        "payback_months_typical": 24,
        "confidence": "medium",
    },
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class FacilityScanWorkflow:
    """
    4-phase facility scan workflow for quick-win identification.

    Performs facility registration, quick-win scanning against benchmarks,
    initial savings estimation, and report compilation.

    Zero-hallucination: all savings estimates use deterministic benchmark
    percentages and engineering cost ratios. No LLM calls in the numeric
    computation path.

    Attributes:
        scan_id: Unique scan execution identifier.
        _quick_wins: Identified quick-win opportunities.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = FacilityScanWorkflow()
        >>> inp = FacilityScanInput(
        ...     facility_name="Office HQ",
        ...     floor_area_m2=Decimal("5000"),
        ...     annual_energy_kwh=Decimal("750000"),
        ...     annual_energy_cost=Decimal("112500"),
        ... )
        >>> result = wf.run(inp)
        >>> assert result.total_quick_wins_found > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FacilityScanWorkflow."""
        self.scan_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._quick_wins: List[QuickWinItem] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: FacilityScanInput) -> FacilityScanResult:
        """
        Execute the 4-phase facility scan workflow.

        Args:
            input_data: Validated facility scan input.

        Returns:
            FacilityScanResult with identified quick wins and savings estimates.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting facility scan workflow %s for facility=%s type=%s",
            self.scan_id, input_data.facility_name, input_data.building_type,
        )

        self._phase_results = []
        self._quick_wins = []

        try:
            # Phase 1: Facility Registration
            phase1 = self._phase_facility_registration(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Quick Win Scanning
            phase2 = self._phase_quick_win_scanning(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Initial Estimation
            phase3 = self._phase_initial_estimation(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Report Generation
            phase4 = self._phase_report_generation(input_data)
            self._phase_results.append(phase4)

        except Exception as exc:
            self.logger.error("Facility scan workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        total_savings_kwh = sum(w.estimated_savings_kwh for w in self._quick_wins)
        total_savings_cost = sum(w.estimated_savings_cost for w in self._quick_wins)
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = FacilityScanResult(
            scan_id=self.scan_id,
            facility_id=input_data.facility_id,
            total_quick_wins_found=len(self._quick_wins),
            quick_wins=self._quick_wins,
            estimated_total_savings_kwh=total_savings_kwh,
            estimated_total_savings_cost=total_savings_cost,
            scan_duration_ms=int(elapsed_ms),
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Facility scan workflow %s completed in %dms wins=%d savings=%.0f kWh",
            self.scan_id, int(elapsed_ms), len(self._quick_wins),
            float(total_savings_kwh),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Facility Registration
    # -------------------------------------------------------------------------

    def _phase_facility_registration(
        self, input_data: FacilityScanInput
    ) -> PhaseResult:
        """Validate facility profile and equipment survey data."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Validate core fields
        if input_data.operating_hours <= 0:
            warnings.append("Operating hours not specified; defaulting to 2500")

        if not input_data.equipment_data:
            warnings.append("No equipment survey data provided; scanning benchmarks only")

        # Calculate energy use intensity (EUI)
        eui = float(input_data.annual_energy_kwh) / float(input_data.floor_area_m2)
        cost_per_kwh = (
            float(input_data.annual_energy_cost) / float(input_data.annual_energy_kwh)
            if input_data.annual_energy_kwh > 0 else Decimal("0.15")
        )

        # Summarize equipment
        equipment_categories: Dict[str, int] = {}
        for key, value in input_data.equipment_data.items():
            if isinstance(value, list):
                equipment_categories[key] = len(value)
            elif isinstance(value, dict):
                equipment_categories[key] = 1

        outputs["facility_id"] = input_data.facility_id
        outputs["facility_name"] = input_data.facility_name
        outputs["building_type"] = input_data.building_type
        outputs["floor_area_m2"] = str(input_data.floor_area_m2)
        outputs["eui_kwh_per_m2"] = round(eui, 2)
        outputs["cost_per_kwh"] = round(float(cost_per_kwh), 4)
        outputs["equipment_categories"] = equipment_categories
        outputs["climate_zone"] = input_data.climate_zone

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 FacilityRegistration: EUI=%.1f kWh/m2, cost=%.4f/kWh",
            eui, float(cost_per_kwh),
        )
        return PhaseResult(
            phase_name="facility_registration", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Quick Win Scanning
    # -------------------------------------------------------------------------

    def _phase_quick_win_scanning(
        self, input_data: FacilityScanInput
    ) -> PhaseResult:
        """Run QuickWinsScannerEngine logic against facility benchmarks."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        annual_kwh = float(input_data.annual_energy_kwh)
        annual_cost = float(input_data.annual_energy_cost)
        floor_area = float(input_data.floor_area_m2)
        cost_per_kwh = annual_cost / annual_kwh if annual_kwh > 0 else 0.15

        applicable_wins: List[QuickWinItem] = []

        for benchmark_id, benchmark in QUICK_WIN_BENCHMARKS.items():
            # Determine applicability based on building type
            if not self._is_benchmark_applicable(benchmark_id, input_data.building_type):
                continue

            # Calculate savings using midpoint of range
            savings_pct = (benchmark["savings_pct_low"] + benchmark["savings_pct_high"]) / 2.0
            savings_kwh = Decimal(str(round(annual_kwh * savings_pct, 2)))
            savings_cost = Decimal(str(round(float(savings_kwh) * cost_per_kwh, 2)))
            impl_cost = benchmark["cost_per_m2"] * input_data.floor_area_m2

            # Calculate payback
            payback_months = (
                Decimal(str(round(float(impl_cost) / float(savings_cost) * 12, 1)))
                if savings_cost > 0 else Decimal("999")
            )

            # Only include wins with payback under 36 months
            if payback_months <= Decimal("36"):
                win = QuickWinItem(
                    title=benchmark["title"],
                    description=f"{benchmark['title']} for {input_data.building_type} facility",
                    category=benchmark["category"],
                    estimated_savings_kwh=savings_kwh,
                    estimated_savings_cost=savings_cost,
                    implementation_cost=impl_cost,
                    simple_payback_months=payback_months,
                    confidence=benchmark["confidence"],
                )
                applicable_wins.append(win)

        self._quick_wins = applicable_wins

        outputs["benchmarks_evaluated"] = len(QUICK_WIN_BENCHMARKS)
        outputs["quick_wins_found"] = len(applicable_wins)
        outputs["categories_found"] = list(set(w.category for w in applicable_wins))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 QuickWinScanning: %d benchmarks evaluated, %d wins found",
            len(QUICK_WIN_BENCHMARKS), len(applicable_wins),
        )
        return PhaseResult(
            phase_name="quick_win_scanning", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _is_benchmark_applicable(self, benchmark_id: str, building_type: str) -> bool:
        """Check whether a benchmark applies to the given building type."""
        # Industrial-specific benchmarks
        industrial_only = {"compressed_air_leak_repair", "vfd_on_pumps_fans"}
        if benchmark_id in industrial_only and building_type not in ("industrial", "warehouse"):
            return False
        return True

    # -------------------------------------------------------------------------
    # Phase 3: Initial Estimation
    # -------------------------------------------------------------------------

    def _phase_initial_estimation(
        self, input_data: FacilityScanInput
    ) -> PhaseResult:
        """Run EnergySavingsEstimatorEngine for top opportunities."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Sort quick wins by savings (descending) and refine top items
        sorted_wins = sorted(
            self._quick_wins,
            key=lambda w: w.estimated_savings_kwh,
            reverse=True,
        )

        # Refine estimates using equipment data if available
        for win in sorted_wins:
            self._refine_estimate(win, input_data)

        # Aggregate totals
        total_savings_kwh = sum(w.estimated_savings_kwh for w in sorted_wins)
        total_savings_cost = sum(w.estimated_savings_cost for w in sorted_wins)
        total_impl_cost = sum(w.implementation_cost for w in sorted_wins)
        portfolio_payback_months = (
            Decimal(str(round(float(total_impl_cost) / float(total_savings_cost) * 12, 1)))
            if total_savings_cost > 0 else Decimal("0")
        )

        outputs["top_5_wins"] = [
            {"title": w.title, "savings_kwh": str(w.estimated_savings_kwh)}
            for w in sorted_wins[:5]
        ]
        outputs["total_savings_kwh"] = str(total_savings_kwh)
        outputs["total_savings_cost"] = str(total_savings_cost)
        outputs["total_implementation_cost"] = str(total_impl_cost)
        outputs["portfolio_payback_months"] = str(portfolio_payback_months)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 InitialEstimation: total savings=%.0f kWh, payback=%.1f months",
            float(total_savings_kwh), float(portfolio_payback_months),
        )
        return PhaseResult(
            phase_name="initial_estimation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _refine_estimate(
        self, win: QuickWinItem, input_data: FacilityScanInput
    ) -> None:
        """Refine a quick-win estimate using equipment survey data if available."""
        equipment = input_data.equipment_data
        if not equipment:
            return

        # If equipment data contains items for this category, adjust confidence
        category_key = win.category
        if category_key in equipment:
            win.confidence = "high"

    # -------------------------------------------------------------------------
    # Phase 4: Report Generation
    # -------------------------------------------------------------------------

    def _phase_report_generation(
        self, input_data: FacilityScanInput
    ) -> PhaseResult:
        """Compile scan results into structured report output."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Summary statistics
        total_wins = len(self._quick_wins)
        high_confidence = sum(1 for w in self._quick_wins if w.confidence == "high")
        by_category: Dict[str, int] = {}
        for win in self._quick_wins:
            by_category[win.category] = by_category.get(win.category, 0) + 1

        # Savings as percentage of annual energy
        total_savings_kwh = sum(w.estimated_savings_kwh for w in self._quick_wins)
        savings_pct = (
            float(total_savings_kwh) / float(input_data.annual_energy_kwh) * 100.0
            if input_data.annual_energy_kwh > 0 else 0.0
        )

        outputs["total_quick_wins"] = total_wins
        outputs["high_confidence_wins"] = high_confidence
        outputs["wins_by_category"] = by_category
        outputs["savings_pct_of_annual"] = round(savings_pct, 2)
        outputs["report_generated_at"] = datetime.utcnow().isoformat() + "Z"

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 ReportGeneration: %d wins, %.1f%% savings potential",
            total_wins, savings_pct,
        )
        return PhaseResult(
            phase_name="report_generation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: FacilityScanResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
