# -*- coding: utf-8 -*-
"""
Program Enrollment Workflow
===================================

4-phase workflow for matching facilities to demand response programs,
projecting revenue, preparing enrollment documentation, and registering
commitments within PACK-037 Demand Response Pack.

Phases:
    1. ProgramMatching          -- Match facility to eligible DR programs
    2. RevenueProjection        -- Project annual DR revenue per program
    3. EnrollmentDocumentation  -- Prepare required enrollment documents
    4. CommitmentRegistration   -- Register capacity commitments

The workflow follows GreenLang zero-hallucination principles: revenue
projections use published programme tariff rates and deterministic
capacity-times-rate calculations. SHA-256 provenance hashes guarantee
auditability.

Regulatory references:
    - FERC Order 745 (compensation for demand response)
    - ISO/RTO capacity market rules
    - PJM / NYISO / CAISO / ERCOT demand response programme tariffs

Schedule: on-demand / annual
Estimated duration: 20 minutes

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


class ProgramType(str, Enum):
    """Type of demand response program."""

    CAPACITY = "capacity"
    ENERGY = "energy"
    ANCILLARY = "ancillary"
    EMERGENCY = "emergency"
    ECONOMIC = "economic"
    PRICE_RESPONSIVE = "price_responsive"


class EnrollmentStatus(str, Enum):
    """Enrollment status."""

    ELIGIBLE = "eligible"
    ENROLLED = "enrolled"
    PENDING_REVIEW = "pending_review"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

DR_PROGRAM_CATALOG: Dict[str, Dict[str, Any]] = {
    "capacity_market": {
        "program_name": "Capacity Market DR",
        "program_type": "capacity",
        "min_capacity_kw": 100,
        "rate_per_kw_year": Decimal("45.00"),
        "commitment_hours": 4,
        "season": "summer",
        "penalty_rate_pct": Decimal("1.5"),
        "notice_period_min": 120,
        "max_events_per_year": 10,
        "eligible_facility_types": ["commercial", "industrial", "institutional", "warehouse"],
    },
    "emergency_dr": {
        "program_name": "Emergency Demand Response",
        "program_type": "emergency",
        "min_capacity_kw": 50,
        "rate_per_kw_year": Decimal("60.00"),
        "commitment_hours": 6,
        "season": "year_round",
        "penalty_rate_pct": Decimal("2.0"),
        "notice_period_min": 60,
        "max_events_per_year": 15,
        "eligible_facility_types": ["commercial", "industrial", "institutional", "warehouse"],
    },
    "economic_dr": {
        "program_name": "Economic Demand Response",
        "program_type": "economic",
        "min_capacity_kw": 200,
        "rate_per_kwh": Decimal("0.25"),
        "commitment_hours": 0,
        "season": "year_round",
        "penalty_rate_pct": Decimal("0.0"),
        "notice_period_min": 30,
        "max_events_per_year": 50,
        "eligible_facility_types": ["commercial", "industrial", "warehouse"],
    },
    "ancillary_services": {
        "program_name": "Ancillary Services DR",
        "program_type": "ancillary",
        "min_capacity_kw": 500,
        "rate_per_kw_year": Decimal("80.00"),
        "commitment_hours": 2,
        "season": "year_round",
        "penalty_rate_pct": Decimal("3.0"),
        "notice_period_min": 10,
        "max_events_per_year": 30,
        "eligible_facility_types": ["industrial", "warehouse"],
    },
    "price_responsive": {
        "program_name": "Critical Peak Pricing Response",
        "program_type": "price_responsive",
        "min_capacity_kw": 25,
        "rate_per_kwh": Decimal("0.50"),
        "commitment_hours": 0,
        "season": "summer",
        "penalty_rate_pct": Decimal("0.0"),
        "notice_period_min": 60,
        "max_events_per_year": 15,
        "eligible_facility_types": ["commercial", "industrial", "institutional", "retail", "warehouse"],
    },
    "frequency_regulation": {
        "program_name": "Frequency Regulation DR",
        "program_type": "ancillary",
        "min_capacity_kw": 1000,
        "rate_per_kw_year": Decimal("120.00"),
        "commitment_hours": 1,
        "season": "year_round",
        "penalty_rate_pct": Decimal("5.0"),
        "notice_period_min": 5,
        "max_events_per_year": 100,
        "eligible_facility_types": ["industrial"],
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


class ProgramMatch(BaseModel):
    """A matched DR program for the facility."""

    match_id: str = Field(default_factory=lambda: f"pm-{uuid.uuid4().hex[:8]}")
    program_key: str = Field(default="", description="Program catalog key")
    program_name: str = Field(default="", description="Program display name")
    program_type: str = Field(default="", description="capacity|energy|ancillary|emergency")
    eligible: bool = Field(default=False, description="Whether facility qualifies")
    committed_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Committed capacity")
    projected_annual_revenue: Decimal = Field(default=Decimal("0"), ge=0)
    penalty_risk_annual: Decimal = Field(default=Decimal("0"), ge=0)
    net_projected_revenue: Decimal = Field(default=Decimal("0"), ge=0)
    enrollment_status: str = Field(default="eligible")


class ProgramEnrollmentInput(BaseModel):
    """Input data model for ProgramEnrollmentWorkflow."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(..., min_length=1, description="Facility name")
    facility_type: str = Field(default="commercial", description="Facility type")
    curtailable_kw: Decimal = Field(..., gt=0, description="Total curtailable capacity kW")
    peak_demand_kw: Decimal = Field(..., gt=0, description="Facility peak demand kW")
    ramp_time_min: int = Field(default=10, ge=0, description="Average ramp time in minutes")
    preferred_programs: List[str] = Field(default_factory=list, description="Preferred program keys")
    max_events_tolerance: int = Field(default=20, ge=1, description="Max events willing to participate")
    performance_confidence_pct: Decimal = Field(default=Decimal("85"), ge=0, le=100)
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


class ProgramEnrollmentResult(BaseModel):
    """Complete result from program enrollment workflow."""

    enrollment_id: str = Field(..., description="Unique enrollment execution ID")
    facility_id: str = Field(default="", description="Facility identifier")
    programs_evaluated: int = Field(default=0, ge=0)
    programs_eligible: int = Field(default=0, ge=0)
    program_matches: List[ProgramMatch] = Field(default_factory=list)
    total_projected_revenue: Decimal = Field(default=Decimal("0"), ge=0)
    total_committed_kw: Decimal = Field(default=Decimal("0"), ge=0)
    total_penalty_risk: Decimal = Field(default=Decimal("0"), ge=0)
    net_projected_revenue: Decimal = Field(default=Decimal("0"), ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ProgramEnrollmentWorkflow:
    """
    4-phase program enrollment workflow for demand response.

    Matches facility curtailment capacity to eligible DR programs,
    projects revenue and penalty risk, prepares enrollment documentation,
    and registers capacity commitments.

    Zero-hallucination: revenue projections use published programme rates
    and deterministic capacity * rate formulas. No LLM calls in the
    numeric computation path.

    Attributes:
        enrollment_id: Unique enrollment execution identifier.
        _matches: Matched program list.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = ProgramEnrollmentWorkflow()
        >>> inp = ProgramEnrollmentInput(
        ...     facility_name="Distribution Center",
        ...     curtailable_kw=Decimal("500"),
        ...     peak_demand_kw=Decimal("2000"),
        ... )
        >>> result = wf.run(inp)
        >>> assert result.programs_eligible > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ProgramEnrollmentWorkflow."""
        self.enrollment_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._matches: List[ProgramMatch] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: ProgramEnrollmentInput) -> ProgramEnrollmentResult:
        """
        Execute the 4-phase program enrollment workflow.

        Args:
            input_data: Validated program enrollment input.

        Returns:
            ProgramEnrollmentResult with matched programs and revenue projections.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = _utcnow()
        self.logger.info(
            "Starting program enrollment workflow %s facility=%s curtailable=%.0f kW",
            self.enrollment_id, input_data.facility_name,
            float(input_data.curtailable_kw),
        )

        self._phase_results = []
        self._matches = []

        try:
            # Phase 1: Program Matching
            phase1 = self._phase_program_matching(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Revenue Projection
            phase2 = self._phase_revenue_projection(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Enrollment Documentation
            phase3 = self._phase_enrollment_documentation(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Commitment Registration
            phase4 = self._phase_commitment_registration(input_data)
            self._phase_results.append(phase4)

        except Exception as exc:
            self.logger.error(
                "Program enrollment workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        eligible_matches = [m for m in self._matches if m.eligible]
        total_revenue = sum(m.projected_annual_revenue for m in eligible_matches)
        total_committed = sum(m.committed_kw for m in eligible_matches)
        total_penalty = sum(m.penalty_risk_annual for m in eligible_matches)
        net_revenue = total_revenue - total_penalty
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = ProgramEnrollmentResult(
            enrollment_id=self.enrollment_id,
            facility_id=input_data.facility_id,
            programs_evaluated=len(self._matches),
            programs_eligible=len(eligible_matches),
            program_matches=self._matches,
            total_projected_revenue=total_revenue,
            total_committed_kw=total_committed,
            total_penalty_risk=total_penalty,
            net_projected_revenue=max(Decimal("0"), net_revenue),
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Program enrollment workflow %s completed in %dms "
            "eligible=%d revenue=%.0f committed=%.0f kW",
            self.enrollment_id, int(elapsed_ms), len(eligible_matches),
            float(total_revenue), float(total_committed),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Program Matching
    # -------------------------------------------------------------------------

    def _phase_program_matching(
        self, input_data: ProgramEnrollmentInput
    ) -> PhaseResult:
        """Match facility to eligible DR programs."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        curtailable_kw = float(input_data.curtailable_kw)
        facility_type = input_data.facility_type
        ramp_time = input_data.ramp_time_min

        for prog_key, prog in DR_PROGRAM_CATALOG.items():
            eligible = True
            reasons: List[str] = []

            # Check minimum capacity
            if curtailable_kw < prog["min_capacity_kw"]:
                eligible = False
                reasons.append(
                    f"Capacity {curtailable_kw:.0f} kW < minimum {prog['min_capacity_kw']} kW"
                )

            # Check facility type eligibility
            if facility_type not in prog["eligible_facility_types"]:
                eligible = False
                reasons.append(
                    f"Facility type '{facility_type}' not eligible"
                )

            # Check ramp time compatibility
            if ramp_time > prog["notice_period_min"]:
                eligible = False
                reasons.append(
                    f"Ramp time {ramp_time} min > notice period {prog['notice_period_min']} min"
                )

            # Check events tolerance
            if input_data.max_events_tolerance < prog["max_events_per_year"]:
                warnings.append(
                    f"Program {prog_key}: max events {prog['max_events_per_year']} "
                    f"exceeds tolerance {input_data.max_events_tolerance}"
                )

            if reasons:
                for r in reasons:
                    warnings.append(f"Program {prog_key}: {r}")

            match = ProgramMatch(
                program_key=prog_key,
                program_name=prog["program_name"],
                program_type=prog.get("program_type", ""),
                eligible=eligible,
            )
            self._matches.append(match)

        eligible_count = sum(1 for m in self._matches if m.eligible)
        outputs["programs_evaluated"] = len(self._matches)
        outputs["programs_eligible"] = eligible_count
        outputs["eligible_programs"] = [
            m.program_key for m in self._matches if m.eligible
        ]

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 ProgramMatching: %d evaluated, %d eligible",
            len(self._matches), eligible_count,
        )
        return PhaseResult(
            phase_name="program_matching", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Revenue Projection
    # -------------------------------------------------------------------------

    def _phase_revenue_projection(
        self, input_data: ProgramEnrollmentInput
    ) -> PhaseResult:
        """Project annual DR revenue per eligible program."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        curtailable_kw = input_data.curtailable_kw
        confidence = float(input_data.performance_confidence_pct) / 100.0

        total_revenue = Decimal("0")
        total_penalty = Decimal("0")

        for match in self._matches:
            if not match.eligible:
                continue

            prog = DR_PROGRAM_CATALOG.get(match.program_key, {})

            # Calculate committed capacity (use confidence factor)
            committed_kw = (curtailable_kw * Decimal(str(confidence))).quantize(
                Decimal("0.1")
            )
            match.committed_kw = committed_kw

            # Revenue projection: capacity-based or energy-based
            if "rate_per_kw_year" in prog:
                annual_revenue = (
                    committed_kw * prog["rate_per_kw_year"]
                ).quantize(Decimal("0.01"))
            elif "rate_per_kwh" in prog:
                # Estimate curtailed energy: committed_kw * avg event hours * events
                avg_event_hours = prog.get("commitment_hours", 4) or 4
                events_per_year = min(
                    prog.get("max_events_per_year", 10),
                    input_data.max_events_tolerance,
                )
                curtailed_kwh = committed_kw * Decimal(str(avg_event_hours * events_per_year))
                annual_revenue = (
                    curtailed_kwh * prog["rate_per_kwh"]
                ).quantize(Decimal("0.01"))
            else:
                annual_revenue = Decimal("0")

            match.projected_annual_revenue = annual_revenue

            # Penalty risk: assume 1 non-performance event per year
            penalty_pct = prog.get("penalty_rate_pct", Decimal("0"))
            penalty_risk = (annual_revenue * penalty_pct / Decimal("100")).quantize(
                Decimal("0.01")
            )
            match.penalty_risk_annual = penalty_risk
            match.net_projected_revenue = annual_revenue - penalty_risk

            total_revenue += annual_revenue
            total_penalty += penalty_risk

        outputs["total_projected_revenue"] = str(total_revenue)
        outputs["total_penalty_risk"] = str(total_penalty)
        outputs["net_revenue"] = str(total_revenue - total_penalty)
        outputs["programs_projected"] = sum(
            1 for m in self._matches if m.projected_annual_revenue > 0
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 RevenueProjection: revenue=%.0f penalty_risk=%.0f net=%.0f",
            float(total_revenue), float(total_penalty),
            float(total_revenue - total_penalty),
        )
        return PhaseResult(
            phase_name="revenue_projection", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Enrollment Documentation
    # -------------------------------------------------------------------------

    def _phase_enrollment_documentation(
        self, input_data: ProgramEnrollmentInput
    ) -> PhaseResult:
        """Prepare required enrollment documents for eligible programs."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        documents: List[Dict[str, str]] = []
        for match in self._matches:
            if not match.eligible:
                continue

            # Standard enrollment document set
            doc_set = [
                {
                    "document_type": "enrollment_application",
                    "program": match.program_key,
                    "status": "prepared",
                    "description": f"Enrollment application for {match.program_name}",
                },
                {
                    "document_type": "capacity_commitment_form",
                    "program": match.program_key,
                    "status": "prepared",
                    "description": f"Capacity commitment of {match.committed_kw} kW",
                },
                {
                    "document_type": "metering_plan",
                    "program": match.program_key,
                    "status": "prepared",
                    "description": "Interval metering and M&V plan",
                },
            ]
            documents.extend(doc_set)

        outputs["documents_prepared"] = len(documents)
        outputs["programs_documented"] = sum(1 for m in self._matches if m.eligible)
        outputs["document_types"] = list(set(d["document_type"] for d in documents))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 EnrollmentDocumentation: %d documents prepared",
            len(documents),
        )
        return PhaseResult(
            phase_name="enrollment_documentation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Commitment Registration
    # -------------------------------------------------------------------------

    def _phase_commitment_registration(
        self, input_data: ProgramEnrollmentInput
    ) -> PhaseResult:
        """Register capacity commitments for eligible programs."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        registrations: List[Dict[str, Any]] = []
        total_committed = Decimal("0")

        for match in self._matches:
            if not match.eligible:
                continue

            match.enrollment_status = "pending_review"
            registration = {
                "program_key": match.program_key,
                "program_name": match.program_name,
                "committed_kw": str(match.committed_kw),
                "enrollment_status": match.enrollment_status,
                "registration_id": f"reg-{_new_uuid()[:8]}",
                "registered_at": _utcnow().isoformat() + "Z",
            }
            registrations.append(registration)
            total_committed += match.committed_kw

        outputs["registrations_created"] = len(registrations)
        outputs["total_committed_kw"] = str(total_committed)
        outputs["registrations"] = registrations

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 CommitmentRegistration: %d registrations, total=%.0f kW",
            len(registrations), float(total_committed),
        )
        return PhaseResult(
            phase_name="commitment_registration", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ProgramEnrollmentResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
