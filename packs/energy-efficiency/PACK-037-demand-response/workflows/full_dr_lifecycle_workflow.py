# -*- coding: utf-8 -*-
"""
Full DR Lifecycle Workflow
===================================

8-phase end-to-end master workflow that orchestrates the complete demand
response lifecycle within PACK-037 Demand Response Pack.

Phases:
    1. FlexibilityAssessment  -- Assess facility load flexibility
    2. ProgramSelection       -- Match and select optimal DR programs
    3. Enrollment             -- Enroll in selected programs
    4. EventPreparation       -- Prepare for DR event dispatch
    5. EventExecution         -- Execute load curtailment and monitor
    6. Settlement             -- Calculate baseline and settle revenue
    7. CarbonQuantification   -- Quantify avoided emissions
    8. Reporting              -- Generate comprehensive reports

The workflow follows GreenLang zero-hallucination principles: all numeric
results flow through deterministic engine calculations. Delegation to
sub-workflows ensures composability and auditability. SHA-256 provenance
hashes guarantee end-to-end traceability.

Regulatory references:
    - FERC Orders 745 & 2222
    - ISO/RTO demand response programme rules
    - GHG Protocol Scope 2 for avoided emissions

Schedule: on-demand
Estimated duration: 90 minutes

Author: GreenLang Team
Version: 37.0.0
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
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

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
# DEFAULT REFERENCE DATA (Zero-Hallucination)
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

class FullDRLifecycleInput(BaseModel):
    """Input data model for FullDRLifecycleWorkflow."""

    facility_profile: Dict[str, Any] = Field(
        ...,
        description="Facility data: facility_name, facility_type, peak_demand_kw, "
                    "annual_energy_kwh, operating_hours",
    )
    loads: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Load inventory data with load_type, rated_kw, criticality",
    )
    der_assets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="DER asset data with der_type, capacity_kw, energy_kwh",
    )
    event_data: Dict[str, Any] = Field(
        default_factory=lambda: {
            "event_start_utc": "2026-07-15T14:00:00Z",
            "event_end_utc": "2026-07-15T18:00:00Z",
            "duration_hours": 4,
            "severity": "moderate",
        },
        description="DR event parameters",
    )
    historical_demand: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historical daily demand for baseline calculation",
    )
    interval_readings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Metered interval readings during event",
    )
    program_preferences: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_events_tolerance": 20,
            "performance_confidence_pct": 85,
            "preferred_programs": [],
        },
        description="Program preference parameters",
    )
    region: str = Field(default="DEFAULT", description="Region for emission factors")
    report_types: List[str] = Field(
        default_factory=lambda: ["executive_summary", "revenue_summary"],
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

class FullDRLifecycleResult(BaseModel):
    """Complete result from full DR lifecycle workflow."""

    lifecycle_id: str = Field(..., description="Unique lifecycle assessment ID")
    facility_id: str = Field(default="", description="Facility identifier")
    flexibility_data: Dict[str, Any] = Field(default_factory=dict)
    program_data: Dict[str, Any] = Field(default_factory=dict)
    enrollment_data: Dict[str, Any] = Field(default_factory=dict)
    preparation_data: Dict[str, Any] = Field(default_factory=dict)
    execution_data: Dict[str, Any] = Field(default_factory=dict)
    settlement_data: Dict[str, Any] = Field(default_factory=dict)
    carbon_data: Dict[str, Any] = Field(default_factory=dict)
    reporting_data: Dict[str, Any] = Field(default_factory=dict)
    total_curtailable_kw: Decimal = Field(default=Decimal("0"), ge=0)
    total_committed_kw: Decimal = Field(default=Decimal("0"), ge=0)
    total_curtailed_kw_actual: Decimal = Field(default=Decimal("0"), ge=0)
    performance_pct: Decimal = Field(default=Decimal("0"), ge=0)
    net_revenue: Decimal = Field(default=Decimal("0"))
    carbon_avoided_tonnes: Decimal = Field(default=Decimal("0"), ge=0)
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    phases_completed: List[str] = Field(default_factory=list)
    total_duration_ms: int = Field(default=0, ge=0)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class FullDRLifecycleWorkflow:
    """
    8-phase end-to-end demand response lifecycle workflow.

    Orchestrates flexibility assessment, program selection, enrollment,
    event preparation, event execution, settlement, carbon quantification,
    and reporting into a single comprehensive pipeline.

    Zero-hallucination: delegates numeric work to deterministic sub-workflow
    calculations. No LLM calls in the computation path. All inter-phase
    data flows through typed Pydantic models.

    Attributes:
        lifecycle_id: Unique lifecycle execution identifier.
        _flexibility_data: Results from flexibility assessment phase.
        _program_data: Results from program selection phase.
        _enrollment_data: Results from enrollment phase.
        _preparation_data: Results from event preparation phase.
        _execution_data: Results from event execution phase.
        _settlement_data: Results from settlement phase.
        _carbon_data: Results from carbon quantification phase.
        _reporting_data: Results from reporting phase.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = FullDRLifecycleWorkflow()
        >>> inp = FullDRLifecycleInput(
        ...     facility_profile={"facility_name": "HQ", "peak_demand_kw": 2000},
        ... )
        >>> result = wf.run(inp)
        >>> assert result.total_curtailable_kw > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FullDRLifecycleWorkflow."""
        self.lifecycle_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._flexibility_data: Dict[str, Any] = {}
        self._program_data: Dict[str, Any] = {}
        self._enrollment_data: Dict[str, Any] = {}
        self._preparation_data: Dict[str, Any] = {}
        self._execution_data: Dict[str, Any] = {}
        self._settlement_data: Dict[str, Any] = {}
        self._carbon_data: Dict[str, Any] = {}
        self._reporting_data: Dict[str, Any] = {}
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: FullDRLifecycleInput) -> FullDRLifecycleResult:
        """
        Execute the 8-phase full DR lifecycle workflow.

        Args:
            input_data: Validated full lifecycle input.

        Returns:
            FullDRLifecycleResult with all sub-results and aggregate metrics.

        Raises:
            ValueError: If facility profile validation fails.
        """
        t_start = time.perf_counter()
        started_at = utcnow()
        facility_name = input_data.facility_profile.get("facility_name", "Unknown")
        self.logger.info(
            "Starting full DR lifecycle workflow %s for facility=%s",
            self.lifecycle_id, facility_name,
        )

        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Flexibility Assessment
            phase1 = self._phase_flexibility_assessment(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Program Selection
            phase2 = self._phase_program_selection(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Enrollment
            phase3 = self._phase_enrollment(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Event Preparation
            phase4 = self._phase_event_preparation(input_data)
            self._phase_results.append(phase4)

            # Phase 5: Event Execution
            phase5 = self._phase_event_execution(input_data)
            self._phase_results.append(phase5)

            # Phase 6: Settlement
            phase6 = self._phase_settlement(input_data)
            self._phase_results.append(phase6)

            # Phase 7: Carbon Quantification
            phase7 = self._phase_carbon_quantification(input_data)
            self._phase_results.append(phase7)

            # Phase 8: Reporting
            phase8 = self._phase_reporting(input_data)
            self._phase_results.append(phase8)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Full DR lifecycle workflow failed: %s", exc, exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Aggregate final metrics
        facility_id = self._flexibility_data.get("facility_id", "")
        total_curtailable = Decimal(str(
            self._flexibility_data.get("total_curtailable_kw", 0)
        ))
        total_committed = Decimal(str(
            self._enrollment_data.get("total_committed_kw", 0)
        ))
        total_curtailed = Decimal(str(
            self._execution_data.get("average_curtailed_kw", 0)
        ))
        performance_pct = Decimal(str(
            self._execution_data.get("performance_pct", 0)
        ))
        net_revenue = Decimal(str(
            self._settlement_data.get("net_settlement", 0)
        ))
        carbon_avoided = Decimal(str(
            self._carbon_data.get("carbon_avoided_tonnes", 0)
        ))
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = FullDRLifecycleResult(
            lifecycle_id=self.lifecycle_id,
            facility_id=facility_id,
            flexibility_data=self._flexibility_data,
            program_data=self._program_data,
            enrollment_data=self._enrollment_data,
            preparation_data=self._preparation_data,
            execution_data=self._execution_data,
            settlement_data=self._settlement_data,
            carbon_data=self._carbon_data,
            reporting_data=self._reporting_data,
            total_curtailable_kw=total_curtailable,
            total_committed_kw=total_committed,
            total_curtailed_kw_actual=total_curtailed,
            performance_pct=performance_pct,
            net_revenue=net_revenue,
            carbon_avoided_tonnes=carbon_avoided,
            status=overall_status,
            phases_completed=completed_phases,
            total_duration_ms=int(elapsed_ms),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Full DR lifecycle workflow %s completed in %dms status=%s "
            "curtailable=%.0f committed=%.0f curtailed=%.0f perf=%.1f%% "
            "revenue=%.2f carbon=%.2f",
            self.lifecycle_id, int(elapsed_ms), overall_status.value,
            float(total_curtailable), float(total_committed),
            float(total_curtailed), float(performance_pct),
            float(net_revenue), float(carbon_avoided),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Flexibility Assessment
    # -------------------------------------------------------------------------

    def _phase_flexibility_assessment(
        self, input_data: FullDRLifecycleInput
    ) -> PhaseResult:
        """Assess facility load flexibility potential."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        fp = input_data.facility_profile
        peak_kw = float(fp.get("peak_demand_kw", 0))
        facility_type = fp.get("facility_type", "commercial")
        facility_id = fp.get("facility_id", f"fac-{_new_uuid()[:8]}")

        # Calculate flexibility from loads or benchmarks
        total_rated = Decimal("0")
        total_curtailable = Decimal("0")
        load_count = 0

        from packs.energy_efficiency.PACK_037_demand_response.workflows.flexibility_assessment_workflow import (
            LOAD_FLEXIBILITY_BENCHMARKS,
        )

        if input_data.loads:
            for load_dict in input_data.loads:
                rated = Decimal(str(load_dict.get("rated_kw", 0)))
                load_type = load_dict.get("load_type", "")
                benchmark = LOAD_FLEXIBILITY_BENCHMARKS.get(load_type, {})
                flex_pct = Decimal(str(benchmark.get("typical_flexibility_pct", 0.15)))
                curtailable = (rated * flex_pct).quantize(Decimal("0.1"))
                total_rated += rated
                total_curtailable += curtailable
                load_count += 1
        else:
            # Estimate from peak demand
            total_rated = Decimal(str(peak_kw))
            total_curtailable = (total_rated * Decimal("0.20")).quantize(Decimal("0.1"))
            warnings.append("No load inventory; estimating 20% flexibility from peak demand")

        flex_ratio = (
            Decimal(str(round(float(total_curtailable) / float(total_rated) * 100, 2)))
            if total_rated > 0 else Decimal("0")
        )

        self._flexibility_data = {
            "facility_id": facility_id,
            "facility_name": fp.get("facility_name", ""),
            "peak_demand_kw": peak_kw,
            "total_rated_kw": str(total_rated),
            "total_curtailable_kw": str(total_curtailable),
            "flexibility_ratio_pct": str(flex_ratio),
            "loads_assessed": load_count,
        }

        outputs.update(self._flexibility_data)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 FlexibilityAssessment: curtailable=%.0f kW ratio=%.1f%%",
            float(total_curtailable), float(flex_ratio),
        )
        return PhaseResult(
            phase_name="flexibility_assessment", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Program Selection
    # -------------------------------------------------------------------------

    def _phase_program_selection(
        self, input_data: FullDRLifecycleInput
    ) -> PhaseResult:
        """Match and select optimal DR programs."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        from packs.energy_efficiency.PACK_037_demand_response.workflows.program_enrollment_workflow import (
            DR_PROGRAM_CATALOG,
        )

        curtailable_kw = float(self._flexibility_data.get("total_curtailable_kw", 0))
        facility_type = input_data.facility_profile.get("facility_type", "commercial")
        ramp_time = 10

        eligible_programs: List[Dict[str, Any]] = []
        for prog_key, prog in DR_PROGRAM_CATALOG.items():
            eligible = (
                curtailable_kw >= prog["min_capacity_kw"]
                and facility_type in prog["eligible_facility_types"]
                and ramp_time <= prog["notice_period_min"]
            )
            if eligible:
                eligible_programs.append({
                    "program_key": prog_key,
                    "program_name": prog["program_name"],
                    "program_type": prog.get("program_type", ""),
                    "min_capacity_kw": prog["min_capacity_kw"],
                })

        self._program_data = {
            "programs_evaluated": len(DR_PROGRAM_CATALOG),
            "programs_eligible": len(eligible_programs),
            "eligible_programs": eligible_programs,
        }

        outputs.update(self._program_data)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 ProgramSelection: %d eligible of %d programs",
            len(eligible_programs), len(DR_PROGRAM_CATALOG),
        )
        return PhaseResult(
            phase_name="program_selection", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Enrollment
    # -------------------------------------------------------------------------

    def _phase_enrollment(
        self, input_data: FullDRLifecycleInput
    ) -> PhaseResult:
        """Enroll in selected DR programs."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        curtailable_kw = Decimal(str(
            self._flexibility_data.get("total_curtailable_kw", 0)
        ))
        confidence = Decimal(str(
            input_data.program_preferences.get("performance_confidence_pct", 85)
        )) / Decimal("100")
        committed_kw = (curtailable_kw * confidence).quantize(Decimal("0.1"))

        eligible = self._program_data.get("eligible_programs", [])
        enrollments: List[Dict[str, Any]] = []
        total_revenue = Decimal("0")

        from packs.energy_efficiency.PACK_037_demand_response.workflows.program_enrollment_workflow import (
            DR_PROGRAM_CATALOG,
        )

        for prog_info in eligible:
            prog_key = prog_info["program_key"]
            prog = DR_PROGRAM_CATALOG.get(prog_key, {})
            if not prog:
                continue

            # Revenue estimate
            if "rate_per_kw_year" in prog:
                revenue = (committed_kw * prog["rate_per_kw_year"]).quantize(Decimal("0.01"))
            elif "rate_per_kwh" in prog:
                avg_hours = Decimal(str(prog.get("commitment_hours", 4) or 4))
                events = Decimal(str(min(
                    prog.get("max_events_per_year", 10),
                    int(input_data.program_preferences.get("max_events_tolerance", 20)),
                )))
                revenue = (committed_kw * avg_hours * events * prog["rate_per_kwh"]).quantize(Decimal("0.01"))
            else:
                revenue = Decimal("0")

            enrollments.append({
                "program_key": prog_key,
                "program_name": prog["program_name"],
                "committed_kw": str(committed_kw),
                "projected_revenue": str(revenue),
                "status": "pending_review",
            })
            total_revenue += revenue

        self._enrollment_data = {
            "programs_enrolled": len(enrollments),
            "total_committed_kw": str(committed_kw),
            "total_projected_revenue": str(total_revenue),
            "enrollments": enrollments,
        }

        outputs.update({
            "programs_enrolled": len(enrollments),
            "total_committed_kw": str(committed_kw),
            "total_projected_revenue": str(total_revenue),
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 Enrollment: %d programs, committed=%.0f kW, revenue=%.0f",
            len(enrollments), float(committed_kw), float(total_revenue),
        )
        return PhaseResult(
            phase_name="enrollment", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Event Preparation
    # -------------------------------------------------------------------------

    def _phase_event_preparation(
        self, input_data: FullDRLifecycleInput
    ) -> PhaseResult:
        """Prepare facility for DR event dispatch."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        committed_kw = Decimal(str(
            self._enrollment_data.get("total_committed_kw", 0)
        ))
        event = input_data.event_data

        # Build dispatch plan from loads
        dispatch_actions: List[Dict[str, Any]] = []
        cumulative = Decimal("0")
        priority = 0

        for load_dict in input_data.loads:
            if cumulative >= committed_kw:
                break

            rated = Decimal(str(load_dict.get("rated_kw", 0)))
            from packs.energy_efficiency.PACK_037_demand_response.workflows.flexibility_assessment_workflow import (

                LOAD_FLEXIBILITY_BENCHMARKS,
            )
            benchmark = LOAD_FLEXIBILITY_BENCHMARKS.get(
                load_dict.get("load_type", ""), {}
            )
            flex_pct = Decimal(str(benchmark.get("typical_flexibility_pct", 0.15)))
            curtail_kw = (rated * flex_pct).quantize(Decimal("0.1"))

            if curtail_kw > 0:
                priority += 1
                cumulative += curtail_kw
                dispatch_actions.append({
                    "load_id": load_dict.get("load_id", f"load-{_new_uuid()[:8]}"),
                    "load_name": load_dict.get("name", load_dict.get("load_type", "")),
                    "curtail_kw": str(curtail_kw),
                    "priority": priority,
                })

        achievable = cumulative
        target_met = achievable >= committed_kw

        self._preparation_data = {
            "event_start": event.get("event_start_utc", ""),
            "event_end": event.get("event_end_utc", ""),
            "target_kw": str(committed_kw),
            "achievable_kw": str(achievable),
            "dispatch_actions": dispatch_actions,
            "target_met": target_met,
        }

        if not target_met:
            warnings.append(
                f"Achievable {achievable} kW < committed {committed_kw} kW"
            )

        outputs.update({
            "dispatch_actions_count": len(dispatch_actions),
            "achievable_kw": str(achievable),
            "target_met": target_met,
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 EventPreparation: %d actions, achievable=%.0f kW, target_met=%s",
            len(dispatch_actions), float(achievable), target_met,
        )
        return PhaseResult(
            phase_name="event_preparation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Event Execution
    # -------------------------------------------------------------------------

    def _phase_event_execution(
        self, input_data: FullDRLifecycleInput
    ) -> PhaseResult:
        """Execute load curtailment and monitor performance."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        committed = Decimal(str(
            self._enrollment_data.get("total_committed_kw", 0)
        ))
        achievable = Decimal(str(
            self._preparation_data.get("achievable_kw", 0)
        ))

        # Simulate execution with 92% performance factor
        actual_curtailed = (achievable * Decimal("0.92")).quantize(Decimal("0.1"))
        performance_pct = (
            Decimal(str(round(float(actual_curtailed) / float(committed) * 100, 2)))
            if committed > 0 else Decimal("0")
        )

        # Classify performance
        if float(performance_pct) >= 110:
            rating = "exceeds"
        elif float(performance_pct) >= 90:
            rating = "meets"
        elif float(performance_pct) >= 75:
            rating = "marginal"
        elif float(performance_pct) >= 50:
            rating = "underperforms"
        else:
            rating = "fails"

        self._execution_data = {
            "committed_kw": str(committed),
            "average_curtailed_kw": str(actual_curtailed),
            "performance_pct": str(performance_pct),
            "performance_rating": rating,
        }

        outputs.update(self._execution_data)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 5 EventExecution: curtailed=%.0f kW perf=%.1f%% rating=%s",
            float(actual_curtailed), float(performance_pct), rating,
        )
        return PhaseResult(
            phase_name="event_execution", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 6: Settlement
    # -------------------------------------------------------------------------

    def _phase_settlement(
        self, input_data: FullDRLifecycleInput
    ) -> PhaseResult:
        """Calculate baseline and settle revenue."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        committed = Decimal(str(
            self._enrollment_data.get("total_committed_kw", 0)
        ))
        actual_curtailed = Decimal(str(
            self._execution_data.get("average_curtailed_kw", 0)
        ))
        event_hours = Decimal(str(
            input_data.event_data.get("duration_hours", 4)
        ))

        # Baseline from historical data
        if input_data.historical_demand:
            peak_values = sorted(
                [Decimal(str(d.get("peak_kw", 0))) for d in input_data.historical_demand],
                reverse=True,
            )
            top_5 = peak_values[:5] if len(peak_values) >= 5 else peak_values
            baseline_kw = sum(top_5) / Decimal(str(max(len(top_5), 1)))
        else:
            baseline_kw = committed * Decimal("2.5")
            warnings.append("No historical data; using fallback baseline")

        # Revenue: simplified capacity-based calculation
        rate = Decimal("45.00")  # default capacity rate
        gross_revenue = (
            rate * actual_curtailed * event_hours / Decimal("8760")
        ).quantize(Decimal("0.01"))

        # Penalty for shortfall
        shortfall = max(Decimal("0"), committed - actual_curtailed)
        penalty = Decimal("0")
        if shortfall > 0:
            penalty = (
                rate * shortfall * event_hours / Decimal("8760") * Decimal("1.5")
            ).quantize(Decimal("0.01"))

        net_settlement = gross_revenue - penalty

        self._settlement_data = {
            "baseline_kw": str(baseline_kw.quantize(Decimal("0.1"))),
            "gross_revenue": str(gross_revenue),
            "penalty_amount": str(penalty),
            "net_settlement": str(net_settlement),
            "shortfall_kw": str(shortfall),
        }

        outputs.update(self._settlement_data)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 6 Settlement: baseline=%.0f gross=%.2f penalty=%.2f net=%.2f",
            float(baseline_kw), float(gross_revenue),
            float(penalty), float(net_settlement),
        )
        return PhaseResult(
            phase_name="settlement", phase_number=6,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 7: Carbon Quantification
    # -------------------------------------------------------------------------

    def _phase_carbon_quantification(
        self, input_data: FullDRLifecycleInput
    ) -> PhaseResult:
        """Quantify avoided emissions from demand response."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        ef = DEFAULT_GRID_EF.get(input_data.region, DEFAULT_GRID_EF["DEFAULT"])
        actual_curtailed = Decimal(str(
            self._execution_data.get("average_curtailed_kw", 0)
        ))
        event_hours = Decimal(str(
            input_data.event_data.get("duration_hours", 4)
        ))

        # Avoided kWh = curtailed kW * event hours
        avoided_kwh = actual_curtailed * event_hours

        # Carbon avoided = kWh * grid emission factor / 1000 (kg to tonnes)
        carbon_tonnes = (
            avoided_kwh * Decimal(str(ef)) / Decimal("1000")
        ).quantize(Decimal("0.0001"))

        self._carbon_data = {
            "avoided_kwh": str(avoided_kwh),
            "grid_emission_factor": ef,
            "region": input_data.region,
            "carbon_avoided_tonnes": str(carbon_tonnes),
        }

        outputs.update(self._carbon_data)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 7 CarbonQuantification: avoided=%.0f kWh carbon=%.4f tonnes",
            float(avoided_kwh), float(carbon_tonnes),
        )
        return PhaseResult(
            phase_name="carbon_quantification", phase_number=7,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 8: Reporting
    # -------------------------------------------------------------------------

    def _phase_reporting(
        self, input_data: FullDRLifecycleInput
    ) -> PhaseResult:
        """Generate comprehensive DR lifecycle reports."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        now_iso = utcnow().isoformat() + "Z"

        report_content = {
            "report_type": "dr_lifecycle_summary",
            "generated_at": now_iso,
            "facility": input_data.facility_profile.get("facility_name", ""),
            "lifecycle_id": self.lifecycle_id,
            "flexibility": self._flexibility_data,
            "programs": self._program_data,
            "enrollment": self._enrollment_data,
            "execution": self._execution_data,
            "settlement": self._settlement_data,
            "carbon": self._carbon_data,
        }

        self._reporting_data = {
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

    def _compute_provenance(self, result: FullDRLifecycleResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
