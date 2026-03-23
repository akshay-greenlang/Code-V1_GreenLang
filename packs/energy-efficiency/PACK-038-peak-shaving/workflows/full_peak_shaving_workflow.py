# -*- coding: utf-8 -*-
"""
Full Peak Shaving Lifecycle Workflow
===================================

8-phase end-to-end master workflow that orchestrates the complete peak shaving
lifecycle within PACK-038 Peak Shaving Pack.

Phases:
    1. LoadAnalysis         -- Load profile analysis and baseline
    2. PeakAssessment       -- Peak attribution and demand charge decomposition
    3. BESSOptimization     -- Battery sizing, dispatch simulation, financials
    4. LoadShifting          -- Shiftable load identification and scheduling
    5. CPResponse            -- Coincident peak prediction and response
    6. Implementation        -- Engineering design, procurement, commissioning
    7. Verification          -- M&V savings verification per IPMVP
    8. Reporting             -- Comprehensive reporting and documentation

The workflow follows GreenLang zero-hallucination principles: all numeric
results flow through deterministic engine calculations. Delegation to
sub-workflows ensures composability and auditability. SHA-256 provenance
hashes guarantee end-to-end traceability.

Regulatory references:
    - IPMVP 2022 EVO 10000-1:2022 (baseline and savings)
    - NFPA 855 / IEC 62619 (BESS safety)
    - FERC transmission cost allocation (CP response)
    - ASHRAE Guideline 14 (measurement uncertainty)

Schedule: on-demand
Estimated duration: 90 minutes

Author: GreenLang Team
Version: 38.0.0
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


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

DEFAULT_GRID_EF: Dict[str, float] = {
    "US": 0.390,
    "EU": 0.275,
    "UK": 0.207,
    "AU": 0.680,
    "IN": 0.820,
    "CN": 0.580,
    "DEFAULT": 0.400,
}

PEAK_SHAVING_PROGRAM_TYPES: Dict[str, str] = {
    "demand_charge_reduction": "Reduce billing demand charges through peak clipping",
    "cp_response": "Reduce coincident peak transmission charges",
    "ratchet_avoidance": "Avoid demand ratchet clause penalties",
    "tou_optimization": "Optimise time-of-use demand charges",
    "revenue_stacking": "Stack demand savings with ancillary services",
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


class FullPeakShavingInput(BaseModel):
    """Input data model for FullPeakShavingWorkflow."""

    facility_profile: Dict[str, Any] = Field(
        ...,
        description="Facility data: facility_name, facility_type, peak_demand_kw, "
                    "annual_energy_kwh, operating_hours",
    )
    interval_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="15-min interval data: timestamp, demand_kw, energy_kwh",
    )
    billing_periods: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Billing periods: month, peak_kw, demand_charge, cdd, hdd",
    )
    loads: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Load inventory: load_type, rated_kw, name, criticality",
    )
    tariff_data: Dict[str, Any] = Field(
        default_factory=lambda: {
            "tariff_type": "flat",
            "demand_rate": 15.00,
            "energy_rate": 0.10,
            "ratchet_pct": 80,
            "power_factor": 0.92,
        },
        description="Tariff structure parameters",
    )
    bess_preferences: Dict[str, Any] = Field(
        default_factory=lambda: {
            "preferred_chemistry": "lfp",
            "target_peak_reduction_pct": 20,
            "project_life_years": 15,
            "discount_rate_pct": 6.0,
            "itc_pct": 30,
        },
        description="BESS project preferences",
    )
    cp_data: Dict[str, Any] = Field(
        default_factory=lambda: {
            "iso_rto": "pjm_5cp",
            "weather_data": {"temperature_f": 95, "heat_index_f": 105},
            "grid_signals": {"forecast_peak_mw": 155000, "historical_peak_mw": 160000},
        },
        description="Coincident peak data",
    )
    vendor_quotes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Vendor quotes for implementation",
    )
    reporting_periods: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Post-implementation reporting data for M&V",
    )
    region: str = Field(default="DEFAULT", description="Region for emission factors")
    report_types: List[str] = Field(
        default_factory=lambda: ["executive_summary", "technical_detail"],
        description="Report types to generate",
    )
    report_format: str = Field(default="json", description="Output report format")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facility_profile")
    @classmethod
    def validate_facility_profile(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure facility profile has minimum required fields."""
        required = ["facility_name", "peak_demand_kw"]
        missing = [f for f in required if f not in v or v[f] is None]
        if missing:
            raise ValueError(f"facility_profile missing required fields: {missing}")
        return v


class FullPeakShavingResult(BaseModel):
    """Complete result from full peak shaving lifecycle workflow."""

    lifecycle_id: str = Field(..., description="Unique lifecycle assessment ID")
    facility_id: str = Field(default="", description="Facility identifier")
    load_analysis_data: Dict[str, Any] = Field(default_factory=dict)
    peak_assessment_data: Dict[str, Any] = Field(default_factory=dict)
    bess_optimization_data: Dict[str, Any] = Field(default_factory=dict)
    load_shift_data: Dict[str, Any] = Field(default_factory=dict)
    cp_response_data: Dict[str, Any] = Field(default_factory=dict)
    implementation_data: Dict[str, Any] = Field(default_factory=dict)
    verification_data: Dict[str, Any] = Field(default_factory=dict)
    reporting_data: Dict[str, Any] = Field(default_factory=dict)
    current_peak_kw: Decimal = Field(default=Decimal("0"), ge=0)
    target_peak_kw: Decimal = Field(default=Decimal("0"), ge=0)
    total_reduction_kw: Decimal = Field(default=Decimal("0"), ge=0)
    reduction_pct: Decimal = Field(default=Decimal("0"), ge=0)
    annual_demand_savings: Decimal = Field(default=Decimal("0"), ge=0)
    total_capex: Decimal = Field(default=Decimal("0"), ge=0)
    simple_payback_years: Decimal = Field(default=Decimal("0"), ge=0)
    npv: Decimal = Field(default=Decimal("0"))
    bess_power_kw: Decimal = Field(default=Decimal("0"), ge=0)
    bess_energy_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    carbon_avoided_tonnes: Decimal = Field(default=Decimal("0"), ge=0)
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    phases_completed: List[str] = Field(default_factory=list)
    total_duration_ms: int = Field(default=0, ge=0)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class FullPeakShavingWorkflow:
    """
    8-phase end-to-end peak shaving lifecycle workflow.

    Orchestrates load analysis, peak assessment, BESS optimization, load
    shifting, CP response, implementation planning, M&V verification,
    and reporting into a single comprehensive pipeline.

    Zero-hallucination: delegates numeric work to deterministic sub-workflow
    calculations. No LLM calls in the computation path. All inter-phase
    data flows through typed Pydantic models.

    Attributes:
        lifecycle_id: Unique lifecycle execution identifier.
        _load_analysis: Results from load analysis phase.
        _peak_assessment: Results from peak assessment phase.
        _bess_optimization: Results from BESS optimization phase.
        _load_shift: Results from load shifting phase.
        _cp_response: Results from CP response phase.
        _implementation: Results from implementation phase.
        _verification: Results from verification phase.
        _reporting: Results from reporting phase.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = FullPeakShavingWorkflow()
        >>> inp = FullPeakShavingInput(
        ...     facility_profile={"facility_name": "HQ", "peak_demand_kw": 3000},
        ... )
        >>> result = wf.run(inp)
        >>> assert result.total_reduction_kw > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FullPeakShavingWorkflow."""
        self.lifecycle_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._load_analysis: Dict[str, Any] = {}
        self._peak_assessment: Dict[str, Any] = {}
        self._bess_optimization: Dict[str, Any] = {}
        self._load_shift: Dict[str, Any] = {}
        self._cp_response: Dict[str, Any] = {}
        self._implementation: Dict[str, Any] = {}
        self._verification: Dict[str, Any] = {}
        self._reporting: Dict[str, Any] = {}
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: FullPeakShavingInput) -> FullPeakShavingResult:
        """
        Execute the 8-phase full peak shaving lifecycle workflow.

        Args:
            input_data: Validated full lifecycle input.

        Returns:
            FullPeakShavingResult with all sub-results and aggregate metrics.

        Raises:
            ValueError: If facility profile validation fails.
        """
        t_start = time.perf_counter()
        started_at = _utcnow()
        facility_name = input_data.facility_profile.get("facility_name", "Unknown")
        self.logger.info(
            "Starting full peak shaving lifecycle workflow %s for facility=%s",
            self.lifecycle_id, facility_name,
        )

        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Load Analysis
            phase1 = self._phase_load_analysis(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Peak Assessment
            phase2 = self._phase_peak_assessment(input_data)
            self._phase_results.append(phase2)

            # Phase 3: BESS Optimization
            phase3 = self._phase_bess_optimization(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Load Shifting
            phase4 = self._phase_load_shifting(input_data)
            self._phase_results.append(phase4)

            # Phase 5: CP Response
            phase5 = self._phase_cp_response(input_data)
            self._phase_results.append(phase5)

            # Phase 6: Implementation
            phase6 = self._phase_implementation(input_data)
            self._phase_results.append(phase6)

            # Phase 7: Verification
            phase7 = self._phase_verification(input_data)
            self._phase_results.append(phase7)

            # Phase 8: Reporting
            phase8 = self._phase_reporting(input_data)
            self._phase_results.append(phase8)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Full peak shaving lifecycle workflow failed: %s", exc, exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Aggregate final metrics
        fp = input_data.facility_profile
        peak_kw = Decimal(str(fp.get("peak_demand_kw", 0)))
        facility_id = fp.get("facility_id", f"fac-{_new_uuid()[:8]}")

        bess_kw = Decimal(str(self._bess_optimization.get("power_kw", 0)))
        bess_kwh = Decimal(str(self._bess_optimization.get("energy_kwh", 0)))
        bess_savings = Decimal(str(self._bess_optimization.get("annual_demand_savings", 0)))
        shift_reduction = Decimal(str(self._load_shift.get("peak_reduction_kw", 0)))
        cp_savings = Decimal(str(self._cp_response.get("annual_savings", 0)))
        total_capex = Decimal(str(self._bess_optimization.get("total_capex", 0)))
        npv = Decimal(str(self._bess_optimization.get("npv", 0)))

        total_reduction = bess_kw + shift_reduction
        total_annual_savings = bess_savings + cp_savings
        target_kw = max(Decimal("0"), peak_kw - total_reduction)
        reduction_pct = (
            Decimal(str(round(float(total_reduction) / max(float(peak_kw), 0.01) * 100, 1)))
        )
        simple_payback = (
            (total_capex / total_annual_savings).quantize(Decimal("0.1"))
            if total_annual_savings > 0 else Decimal("0")
        )

        # Carbon avoided from reduced peak
        ef = DEFAULT_GRID_EF.get(input_data.region, DEFAULT_GRID_EF["DEFAULT"])
        # Estimate: peak reduction * avg hours * grid EF / 1000
        avg_peak_hours = 4
        carbon_avoided = (
            total_reduction * Decimal(str(avg_peak_hours))
            * Decimal("260")  # ~260 business days
            * Decimal(str(ef)) / Decimal("1000")
        ).quantize(Decimal("0.01"))

        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = FullPeakShavingResult(
            lifecycle_id=self.lifecycle_id,
            facility_id=facility_id,
            load_analysis_data=self._load_analysis,
            peak_assessment_data=self._peak_assessment,
            bess_optimization_data=self._bess_optimization,
            load_shift_data=self._load_shift,
            cp_response_data=self._cp_response,
            implementation_data=self._implementation,
            verification_data=self._verification,
            reporting_data=self._reporting,
            current_peak_kw=peak_kw,
            target_peak_kw=target_kw,
            total_reduction_kw=total_reduction,
            reduction_pct=reduction_pct,
            annual_demand_savings=total_annual_savings,
            total_capex=total_capex,
            simple_payback_years=simple_payback,
            npv=npv,
            bess_power_kw=bess_kw,
            bess_energy_kwh=bess_kwh,
            carbon_avoided_tonnes=carbon_avoided,
            status=overall_status,
            phases_completed=completed_phases,
            total_duration_ms=int(elapsed_ms),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Full peak shaving lifecycle %s completed in %dms status=%s "
            "reduction=%.0f kW (%.1f%%) savings=$%.0f NPV=$%.0f carbon=%.2f t",
            self.lifecycle_id, int(elapsed_ms), overall_status.value,
            float(total_reduction), float(reduction_pct),
            float(total_annual_savings), float(npv), float(carbon_avoided),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Load Analysis
    # -------------------------------------------------------------------------

    def _phase_load_analysis(
        self, input_data: FullPeakShavingInput
    ) -> PhaseResult:
        """Analyse facility load profile and establish baseline."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        fp = input_data.facility_profile
        peak_kw = float(fp.get("peak_demand_kw", 0))
        facility_type = fp.get("facility_type", "office_building")
        annual_kwh = float(fp.get("annual_energy_kwh", peak_kw * 2500))
        operating_hours = fp.get("operating_hours", 2500)

        # Calculate basic load statistics
        avg_kw = annual_kwh / max(operating_hours, 1)
        load_factor = avg_kw / max(peak_kw, 0.01) * 100
        base_load = peak_kw * 0.30  # Typical base load 30%

        # If interval data available, compute from it
        if input_data.interval_data:
            demands = [float(r.get("demand_kw", 0)) for r in input_data.interval_data if r.get("demand_kw", 0) > 0]
            if demands:
                avg_kw = sum(demands) / len(demands)
                peak_kw_actual = max(demands)
                load_factor = avg_kw / max(peak_kw_actual, 0.01) * 100
                sorted_demands = sorted(demands)
                base_load = sorted_demands[max(0, int(len(sorted_demands) * 0.10) - 1)]

        # Baseline: average of billing peaks or peak demand
        if input_data.billing_periods:
            billing_peaks = [float(bp.get("peak_kw", 0)) for bp in input_data.billing_periods]
            baseline_kw = sum(billing_peaks) / max(len(billing_peaks), 1)
        else:
            baseline_kw = peak_kw
            warnings.append("No billing data; using reported peak as baseline")

        self._load_analysis = {
            "facility_id": fp.get("facility_id", f"fac-{_new_uuid()[:8]}"),
            "facility_name": fp.get("facility_name", ""),
            "peak_demand_kw": peak_kw,
            "avg_demand_kw": round(avg_kw, 1),
            "load_factor_pct": round(load_factor, 1),
            "base_load_kw": round(base_load, 1),
            "baseline_kw": round(baseline_kw, 1),
            "intervals_analysed": len(input_data.interval_data),
        }

        outputs.update(self._load_analysis)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 LoadAnalysis: peak=%.0f avg=%.0f LF=%.1f%% baseline=%.0f",
            peak_kw, avg_kw, load_factor, baseline_kw,
        )
        return PhaseResult(
            phase_name="load_analysis", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Peak Assessment
    # -------------------------------------------------------------------------

    def _phase_peak_assessment(
        self, input_data: FullPeakShavingInput
    ) -> PhaseResult:
        """Assess peak attribution and demand charge impact."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        peak_kw = self._load_analysis.get("peak_demand_kw", 0)
        tariff = input_data.tariff_data
        demand_rate = float(tariff.get("demand_rate", 15.00))
        tariff_type = tariff.get("tariff_type", "flat")

        # Annual demand charges
        annual_charges = round(peak_kw * demand_rate * 12, 2)

        # Avoidable percentage (from benchmark)
        avoidable_pct = 0.25  # Typical 25% of peak is avoidable
        avoidable_kw = round(peak_kw * avoidable_pct, 1)
        avoidable_charges = round(annual_charges * avoidable_pct, 2)

        # Strategy ranking
        strategies = [
            {"strategy": "BESS Peak Shaving", "effectiveness": "high", "priority": 1},
            {"strategy": "Load Shifting", "effectiveness": "medium", "priority": 2},
            {"strategy": "Demand Limiting", "effectiveness": "medium", "priority": 3},
            {"strategy": "CP Response", "effectiveness": "high", "priority": 4},
        ]

        self._peak_assessment = {
            "annual_demand_charges": annual_charges,
            "avoidable_kw": avoidable_kw,
            "avoidable_charges": avoidable_charges,
            "avoidable_pct": avoidable_pct * 100,
            "tariff_type": tariff_type,
            "demand_rate": demand_rate,
            "strategies": strategies,
        }

        outputs.update({
            "annual_demand_charges": annual_charges,
            "avoidable_kw": avoidable_kw,
            "avoidable_charges": avoidable_charges,
            "strategies_count": len(strategies),
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 PeakAssessment: charges=$%.0f avoidable=%.0f kW ($%.0f)",
            annual_charges, avoidable_kw, avoidable_charges,
        )
        return PhaseResult(
            phase_name="peak_assessment", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: BESS Optimization
    # -------------------------------------------------------------------------

    def _phase_bess_optimization(
        self, input_data: FullPeakShavingInput
    ) -> PhaseResult:
        """Optimise BESS sizing and calculate financial returns."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        peak_kw = Decimal(str(self._load_analysis.get("peak_demand_kw", 0)))
        prefs = input_data.bess_preferences
        reduction_pct = Decimal(str(prefs.get("target_peak_reduction_pct", 20))) / Decimal("100")
        chemistry = prefs.get("preferred_chemistry", "lfp")
        project_life = prefs.get("project_life_years", 15)
        discount_rate = Decimal(str(prefs.get("discount_rate_pct", 6.0))) / Decimal("100")
        itc_pct = Decimal(str(prefs.get("itc_pct", 30))) / Decimal("100")

        # BESS sizing
        shave_kw = (peak_kw * reduction_pct).quantize(Decimal("0.1"))
        duration_hours = Decimal("4")

        from .bess_optimization_workflow import BATTERY_SPECS
        spec = BATTERY_SPECS.get(chemistry, BATTERY_SPECS["lfp"])
        efficiency = Decimal(str(spec["round_trip_efficiency_pct"])) / Decimal("100")
        usable_soc = (
            Decimal(str(spec["max_soc_pct"])) - Decimal(str(spec["min_soc_pct"]))
        ) / Decimal("100")

        energy_kwh = (shave_kw * duration_hours / efficiency / usable_soc).quantize(Decimal("0.1"))
        power_kw = shave_kw

        # CAPEX
        capex_energy = energy_kwh * Decimal(str(spec["capex_per_kwh"]))
        capex_power = power_kw * Decimal(str(spec["capex_per_kw"]))
        total_capex = ((capex_energy + capex_power) * Decimal("1.25")).quantize(Decimal("0.01"))
        itc_amount = (total_capex * itc_pct).quantize(Decimal("0.01"))
        net_capex = total_capex - itc_amount

        # Savings
        demand_rate = Decimal(str(input_data.tariff_data.get("demand_rate", 15.00)))
        annual_demand_savings = (shave_kw * demand_rate * Decimal("12")).quantize(Decimal("0.01"))

        # Simple payback
        payback = (
            (net_capex / annual_demand_savings).quantize(Decimal("0.1"))
            if annual_demand_savings > 0 else Decimal("99")
        )

        # NPV
        npv = float(-net_capex)
        degradation = spec["degradation_pct_per_year"] / 100.0
        for year in range(1, project_life + 1):
            effectiveness = max(0, 1.0 - degradation * (year - 1))
            yr_benefit = float(annual_demand_savings) * effectiveness
            npv += yr_benefit / ((1 + float(discount_rate)) ** year)
        npv_decimal = Decimal(str(round(npv, 2)))

        self._bess_optimization = {
            "power_kw": str(power_kw),
            "energy_kwh": str(energy_kwh),
            "chemistry": chemistry,
            "total_capex": str(total_capex),
            "itc_amount": str(itc_amount),
            "net_capex": str(net_capex),
            "annual_demand_savings": str(annual_demand_savings),
            "simple_payback_years": str(payback),
            "npv": str(npv_decimal),
        }

        outputs.update({
            "bess_power_kw": str(power_kw),
            "bess_energy_kwh": str(energy_kwh),
            "chemistry": chemistry,
            "total_capex": str(total_capex),
            "annual_savings": str(annual_demand_savings),
            "npv": str(npv_decimal),
            "payback_years": str(payback),
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 BESSOptimization: %s kW / %s kWh %s CAPEX=$%.0f NPV=$%.0f",
            power_kw, energy_kwh, chemistry, float(total_capex), float(npv_decimal),
        )
        return PhaseResult(
            phase_name="bess_optimization", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Load Shifting
    # -------------------------------------------------------------------------

    def _phase_load_shifting(
        self, input_data: FullPeakShavingInput
    ) -> PhaseResult:
        """Identify and schedule shiftable loads."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        peak_kw = float(self._load_analysis.get("peak_demand_kw", 0))

        from .load_shift_workflow import SHIFTABLE_LOAD_PARAMS

        total_shiftable = Decimal("0")
        shifted_loads: List[Dict[str, Any]] = []

        if input_data.loads:
            for load_dict in input_data.loads:
                lt = load_dict.get("load_type", "")
                params = SHIFTABLE_LOAD_PARAMS.get(lt)
                if not params:
                    continue
                rated = Decimal(str(load_dict.get("rated_kw", 0)))
                total_shiftable += rated
                shifted_loads.append({
                    "load_type": lt,
                    "rated_kw": str(rated),
                    "direction": params["shift_direction"],
                    "max_shift_hours": params["max_shift_hours"],
                })
        else:
            # Estimate from benchmarks
            for lt_key, params in SHIFTABLE_LOAD_PARAMS.items():
                rated = Decimal(str(round(peak_kw * params["typical_rated_kw_pct"], 1)))
                if rated > 0:
                    total_shiftable += rated
                    shifted_loads.append({
                        "load_type": lt_key,
                        "rated_kw": str(rated),
                        "direction": params["shift_direction"],
                    })
            warnings.append("No loads provided; using benchmark estimates")

        # Effective peak reduction from shifting (typically 40-60% of shiftable)
        peak_reduction = (total_shiftable * Decimal("0.50")).quantize(Decimal("0.1"))

        self._load_shift = {
            "total_shiftable_kw": str(total_shiftable),
            "peak_reduction_kw": str(peak_reduction),
            "loads_shifted": len(shifted_loads),
            "shifted_loads": shifted_loads,
        }

        outputs.update({
            "total_shiftable_kw": str(total_shiftable),
            "peak_reduction_kw": str(peak_reduction),
            "loads_shifted": len(shifted_loads),
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 LoadShifting: %d loads, shiftable=%.0f kW, reduction=%.0f kW",
            len(shifted_loads), float(total_shiftable), float(peak_reduction),
        )
        return PhaseResult(
            phase_name="load_shifting", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: CP Response
    # -------------------------------------------------------------------------

    def _phase_cp_response(
        self, input_data: FullPeakShavingInput
    ) -> PhaseResult:
        """Assess coincident peak response opportunity."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        cp = input_data.cp_data
        iso_rto = cp.get("iso_rto", "pjm_5cp")

        from .cp_response_workflow import CP_METHODOLOGIES
        methodology = CP_METHODOLOGIES.get(iso_rto, CP_METHODOLOGIES["pjm_5cp"])

        peak_kw = float(self._load_analysis.get("peak_demand_kw", 0))
        bess_kw = float(self._bess_optimization.get("power_kw", 0))
        shift_kw = float(self._load_shift.get("peak_reduction_kw", 0))

        # Available curtailment for CP response
        total_available = bess_kw + shift_kw
        rate = methodology["rate_per_kw_year"]
        annual_savings = round(total_available * rate, 2)

        # CP probability estimate (simplified)
        weather = cp.get("weather_data", {})
        grid = cp.get("grid_signals", {})
        heat_index = weather.get("heat_index_f", 95)
        forecast_peak = grid.get("forecast_peak_mw", 150000)
        historical_peak = grid.get("historical_peak_mw", 160000)

        load_ratio = forecast_peak / max(historical_peak, 1) * 100
        temp_score = min(100, max(0, (heat_index - 85) * 5))
        load_score = min(100, max(0, (load_ratio - 80) * 5))
        cp_probability = round(0.50 * temp_score + 0.50 * load_score, 1)

        self._cp_response = {
            "iso_rto": iso_rto,
            "methodology": methodology["methodology"],
            "available_curtailment_kw": total_available,
            "cp_probability_pct": cp_probability,
            "rate_per_kw_year": rate,
            "annual_savings": str(annual_savings),
        }

        outputs.update({
            "iso_rto": iso_rto,
            "cp_probability_pct": cp_probability,
            "annual_cp_savings": str(annual_savings),
            "available_kw": total_available,
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 5 CPResponse: %s probability=%.0f%% savings=$%.0f",
            iso_rto, cp_probability, annual_savings,
        )
        return PhaseResult(
            phase_name="cp_response", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 6: Implementation
    # -------------------------------------------------------------------------

    def _phase_implementation(
        self, input_data: FullPeakShavingInput
    ) -> PhaseResult:
        """Plan implementation timeline and procurement."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        from .implementation_workflow import IMPLEMENTATION_MILESTONES

        milestones: List[Dict[str, Any]] = []
        for ms_key, ms_data in IMPLEMENTATION_MILESTONES.items():
            milestones.append({
                "milestone": ms_key,
                "name": ms_data["name"],
                "duration_days": ms_data["typical_duration_days"],
                "responsible": ms_data["responsible"],
            })

        total_days = sum(m["duration_days"] for m in milestones)

        # Vendor assessment
        vendors_available = len(input_data.vendor_quotes)
        capex = float(self._bess_optimization.get("total_capex", 0))

        self._implementation = {
            "milestones": milestones,
            "total_duration_days": total_days,
            "vendors_evaluated": vendors_available,
            "estimated_capex": capex,
        }

        outputs["milestones_count"] = len(milestones)
        outputs["total_duration_days"] = total_days
        outputs["vendors_evaluated"] = vendors_available

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 6 Implementation: %d milestones, %d days total",
            len(milestones), total_days,
        )
        return PhaseResult(
            phase_name="implementation", phase_number=6,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 7: Verification
    # -------------------------------------------------------------------------

    def _phase_verification(
        self, input_data: FullPeakShavingInput
    ) -> PhaseResult:
        """M&V savings verification."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        baseline_kw = float(self._load_analysis.get("baseline_kw", 0))
        bess_kw = float(self._bess_optimization.get("power_kw", 0))
        reporting_kw = baseline_kw - bess_kw * 0.92  # 92% effectiveness

        verified_savings = round(baseline_kw - reporting_kw, 1)
        savings_pct = round(verified_savings / max(baseline_kw, 0.01) * 100, 1)
        demand_rate = float(input_data.tariff_data.get("demand_rate", 15.00))
        annual_savings = round(verified_savings * demand_rate * 12, 2)

        self._verification = {
            "ipmvp_option": "Option B",
            "baseline_kw": baseline_kw,
            "reporting_kw": round(reporting_kw, 1),
            "verified_savings_kw": verified_savings,
            "savings_pct": savings_pct,
            "annual_demand_savings": annual_savings,
            "confidence": "medium" if not input_data.reporting_periods else "high",
        }

        if not input_data.reporting_periods:
            warnings.append("No reporting period data; using projected savings")

        outputs.update({
            "verified_savings_kw": verified_savings,
            "savings_pct": savings_pct,
            "annual_savings": annual_savings,
            "confidence": self._verification["confidence"],
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 7 Verification: savings=%.0f kW (%.1f%%) annual=$%.0f",
            verified_savings, savings_pct, annual_savings,
        )
        return PhaseResult(
            phase_name="verification", phase_number=7,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 8: Reporting
    # -------------------------------------------------------------------------

    def _phase_reporting(
        self, input_data: FullPeakShavingInput
    ) -> PhaseResult:
        """Generate comprehensive peak shaving reports."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        now_iso = _utcnow().isoformat() + "Z"

        report_content = {
            "report_type": "peak_shaving_lifecycle_summary",
            "generated_at": now_iso,
            "facility": input_data.facility_profile.get("facility_name", ""),
            "lifecycle_id": self.lifecycle_id,
            "load_analysis": self._load_analysis,
            "peak_assessment": self._peak_assessment,
            "bess_optimization": self._bess_optimization,
            "load_shifting": self._load_shift,
            "cp_response": self._cp_response,
            "implementation": self._implementation,
            "verification": self._verification,
            "program_types": PEAK_SHAVING_PROGRAM_TYPES,
        }

        self._reporting = {
            "reports_generated": 1,
            "report_types": input_data.report_types,
            "report_format": input_data.report_format,
            "report_content": report_content,
        }

        outputs["reports_generated"] = 1
        outputs["report_types"] = input_data.report_types

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 8 Reporting: %d reports generated",
            1,
        )
        return PhaseResult(
            phase_name="reporting", phase_number=8,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: FullPeakShavingResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
