# -*- coding: utf-8 -*-
"""
Implementation Planning Workflow
===================================

4-phase workflow for creating actionable implementation plans from
prioritized quick-win measures within PACK-033 Quick Wins Identifier Pack.

Phases:
    1. Sequencing          -- Determine implementation order from priorities
    2. RebateMatching      -- Run UtilityRebateEngine for incentive matching
    3. BehavioralProgram   -- Run BehavioralChangeEngine if behavioral actions
    4. PlanAssembly        -- Compile implementation plan with timeline/budget

The workflow follows GreenLang zero-hallucination principles: rebate
calculations, sequencing logic, and budget aggregation use deterministic
formulas. SHA-256 provenance hashes guarantee auditability.

Schedule: on-demand (follows prioritization)
Estimated duration: 25 minutes

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


class ImplementationPhaseType(str, Enum):
    """Type of implementation phase."""

    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class CustomerSegment(str, Enum):
    """Utility customer segment for rebate matching."""

    SMALL_COMMERCIAL = "small_commercial"
    LARGE_COMMERCIAL = "large_commercial"
    INDUSTRIAL = "industrial"
    INSTITUTIONAL = "institutional"
    RESIDENTIAL = "residential"


# =============================================================================
# REBATE REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Typical utility rebate programs by measure category and region
REBATE_PROGRAMS: Dict[str, Dict[str, Any]] = {
    "lighting": {
        "program_name": "Commercial Lighting Rebate",
        "rebate_per_kwh_saved": Decimal("0.08"),
        "max_rebate_pct": Decimal("50"),
        "eligible_segments": ["small_commercial", "large_commercial", "institutional"],
    },
    "hvac": {
        "program_name": "HVAC Efficiency Rebate",
        "rebate_per_kwh_saved": Decimal("0.10"),
        "max_rebate_pct": Decimal("40"),
        "eligible_segments": ["small_commercial", "large_commercial", "industrial", "institutional"],
    },
    "motors": {
        "program_name": "Motor & Drive Rebate",
        "rebate_per_kwh_saved": Decimal("0.06"),
        "max_rebate_pct": Decimal("35"),
        "eligible_segments": ["large_commercial", "industrial"],
    },
    "controls": {
        "program_name": "Building Controls Rebate",
        "rebate_per_kwh_saved": Decimal("0.07"),
        "max_rebate_pct": Decimal("45"),
        "eligible_segments": ["small_commercial", "large_commercial", "institutional"],
    },
    "envelope": {
        "program_name": "Building Envelope Rebate",
        "rebate_per_kwh_saved": Decimal("0.05"),
        "max_rebate_pct": Decimal("30"),
        "eligible_segments": ["small_commercial", "large_commercial"],
    },
}

# Behavioral program templates
BEHAVIORAL_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "awareness_campaign": {
        "name": "Energy Awareness Campaign",
        "duration_weeks": 8,
        "target_savings_pct": Decimal("3"),
        "cost_per_employee": Decimal("15"),
        "activities": [
            "Kickoff presentation",
            "Weekly energy tips",
            "Department competitions",
            "Dashboard displays",
            "Recognition program",
        ],
    },
    "energy_champions": {
        "name": "Energy Champions Network",
        "duration_weeks": 12,
        "target_savings_pct": Decimal("5"),
        "cost_per_employee": Decimal("25"),
        "activities": [
            "Champion recruitment",
            "Training workshops",
            "Floor walks",
            "Monthly reporting",
            "Best practice sharing",
        ],
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


class ImplementationPhase(BaseModel):
    """A phase within the implementation plan."""

    phase_type: ImplementationPhaseType = Field(default=ImplementationPhaseType.SHORT_TERM)
    phase_label: str = Field(default="", description="Human-readable label")
    measure_ids: List[str] = Field(default_factory=list, description="Measures in this phase")
    start_month: int = Field(default=0, ge=0, description="Start month offset")
    end_month: int = Field(default=0, ge=0, description="End month offset")
    investment: Decimal = Field(default=Decimal("0"), ge=0)
    expected_savings: Decimal = Field(default=Decimal("0"), ge=0)
    rebates: Decimal = Field(default=Decimal("0"), ge=0)


class RebateMatch(BaseModel):
    """A matched utility rebate for a measure."""

    measure_id: str = Field(default="", description="Measure identifier")
    program_name: str = Field(default="", description="Rebate program name")
    rebate_amount: Decimal = Field(default=Decimal("0"), ge=0)
    rebate_pct_of_cost: Decimal = Field(default=Decimal("0"), ge=0, le=100)


class ImplementationPlanInput(BaseModel):
    """Input data model for ImplementationPlanningWorkflow."""

    prioritization_result_id: str = Field(default="", description="Originating prioritization ID")
    selected_measures: List[str] = Field(
        default_factory=list,
        description="Measure IDs selected for implementation",
    )
    measures_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full measure data dicts with category, costs, savings",
    )
    utility_region: str = Field(default="DEFAULT", description="Utility service territory")
    customer_segment: str = Field(default="large_commercial", description="Customer segment")
    org_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Organization profile (employee_count, budget, etc.)",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class ImplementationPlanResult(BaseModel):
    """Complete result from implementation planning workflow."""

    plan_id: str = Field(..., description="Unique plan ID")
    phases: List[ImplementationPhase] = Field(default_factory=list)
    rebate_matches: List[RebateMatch] = Field(default_factory=list)
    total_investment: Decimal = Field(default=Decimal("0"), ge=0)
    total_rebates: Decimal = Field(default=Decimal("0"), ge=0)
    net_investment: Decimal = Field(default=Decimal("0"), ge=0)
    behavioral_program: Optional[Dict[str, Any]] = Field(default=None)
    timeline_months: int = Field(default=0, ge=0, description="Total plan duration in months")
    total_annual_savings: Decimal = Field(default=Decimal("0"), ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ImplementationPlanningWorkflow:
    """
    4-phase implementation planning workflow for quick-win measures.

    Performs sequencing, utility rebate matching, behavioral program
    design, and plan assembly with timeline and budget projections.

    Zero-hallucination: rebate calculations use published program rates,
    sequencing uses payback-based rules, and budget aggregation is
    deterministic arithmetic. No LLM calls in the numeric computation path.

    Attributes:
        plan_id: Unique plan execution identifier.
        _impl_phases: Implementation phases.
        _rebate_matches: Matched rebates.
        _behavioral_program: Behavioral program if applicable.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = ImplementationPlanningWorkflow()
        >>> inp = ImplementationPlanInput(
        ...     selected_measures=["m-abc123"],
        ...     measures_data=[{"measure_id": "m-abc123", ...}],
        ... )
        >>> result = wf.run(inp)
        >>> assert result.net_investment <= result.total_investment
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ImplementationPlanningWorkflow."""
        self.plan_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._impl_phases: List[ImplementationPhase] = []
        self._rebate_matches: List[RebateMatch] = []
        self._behavioral_program: Optional[Dict[str, Any]] = None
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: ImplementationPlanInput) -> ImplementationPlanResult:
        """
        Execute the 4-phase implementation planning workflow.

        Args:
            input_data: Validated implementation planning input.

        Returns:
            ImplementationPlanResult with phased plan, rebates, and timeline.

        Raises:
            ValueError: If no measures data provided.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting implementation planning workflow %s measures=%d",
            self.plan_id, len(input_data.measures_data),
        )

        self._phase_results = []
        self._impl_phases = []
        self._rebate_matches = []
        self._behavioral_program = None

        try:
            # Phase 1: Sequencing
            phase1 = self._phase_sequencing(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Rebate Matching
            phase2 = self._phase_rebate_matching(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Behavioral Program Design
            phase3 = self._phase_behavioral_program(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Plan Assembly
            phase4 = self._phase_plan_assembly(input_data)
            self._phase_results.append(phase4)

        except Exception as exc:
            self.logger.error(
                "Implementation planning workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        total_investment = sum(p.investment for p in self._impl_phases)
        total_rebates = sum(r.rebate_amount for r in self._rebate_matches)
        net_investment = total_investment - total_rebates
        total_savings = sum(p.expected_savings for p in self._impl_phases)
        timeline_months = max((p.end_month for p in self._impl_phases), default=0)
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = ImplementationPlanResult(
            plan_id=self.plan_id,
            phases=[p.model_copy() for p in self._impl_phases],
            rebate_matches=self._rebate_matches,
            total_investment=total_investment,
            total_rebates=total_rebates,
            net_investment=max(Decimal("0"), net_investment),
            behavioral_program=self._behavioral_program,
            timeline_months=timeline_months,
            total_annual_savings=total_savings,
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Implementation planning workflow %s completed in %.0fms "
            "investment=%.0f rebates=%.0f net=%.0f timeline=%d months",
            self.plan_id, elapsed_ms, float(total_investment),
            float(total_rebates), float(net_investment), timeline_months,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Sequencing
    # -------------------------------------------------------------------------

    def _phase_sequencing(
        self, input_data: ImplementationPlanInput
    ) -> PhaseResult:
        """Determine implementation order based on payback and dependencies."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Classify measures into implementation phases by payback
        immediate: List[Dict[str, Any]] = []
        short_term: List[Dict[str, Any]] = []
        medium_term: List[Dict[str, Any]] = []
        long_term: List[Dict[str, Any]] = []

        for measure in input_data.measures_data:
            measure_id = measure.get("measure_id", "")
            # Filter to selected measures if specified
            if input_data.selected_measures and measure_id not in input_data.selected_measures:
                continue

            payback_months = float(measure.get("simple_payback_months", 12))
            if payback_months <= 3:
                immediate.append(measure)
            elif payback_months <= 6:
                short_term.append(measure)
            elif payback_months <= 18:
                medium_term.append(measure)
            else:
                long_term.append(measure)

        # Build implementation phases
        month_offset = 0
        if immediate:
            phase = self._build_impl_phase(
                ImplementationPhaseType.IMMEDIATE, "Immediate (0-3 months)",
                immediate, month_offset, month_offset + 3,
            )
            self._impl_phases.append(phase)
            month_offset = 3

        if short_term:
            phase = self._build_impl_phase(
                ImplementationPhaseType.SHORT_TERM, "Short-term (3-6 months)",
                short_term, month_offset, month_offset + 3,
            )
            self._impl_phases.append(phase)
            month_offset += 3

        if medium_term:
            phase = self._build_impl_phase(
                ImplementationPhaseType.MEDIUM_TERM, "Medium-term (6-12 months)",
                medium_term, month_offset, month_offset + 6,
            )
            self._impl_phases.append(phase)
            month_offset += 6

        if long_term:
            phase = self._build_impl_phase(
                ImplementationPhaseType.LONG_TERM, "Long-term (12+ months)",
                long_term, month_offset, month_offset + 6,
            )
            self._impl_phases.append(phase)

        outputs["phases_created"] = len(self._impl_phases)
        outputs["immediate_count"] = len(immediate)
        outputs["short_term_count"] = len(short_term)
        outputs["medium_term_count"] = len(medium_term)
        outputs["long_term_count"] = len(long_term)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 Sequencing: %d phases, %d/%d/%d/%d measures",
            len(self._impl_phases), len(immediate), len(short_term),
            len(medium_term), len(long_term),
        )
        return PhaseResult(
            phase_name="sequencing", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _build_impl_phase(
        self,
        phase_type: ImplementationPhaseType,
        label: str,
        measures: List[Dict[str, Any]],
        start_month: int,
        end_month: int,
    ) -> ImplementationPhase:
        """Build an ImplementationPhase from a list of measure dicts."""
        measure_ids = [m.get("measure_id", "") for m in measures]
        investment = Decimal(str(round(
            sum(float(m.get("implementation_cost", 0)) for m in measures), 2
        )))
        savings = Decimal(str(round(
            sum(float(m.get("annual_savings_cost", 0)) for m in measures), 2
        )))

        return ImplementationPhase(
            phase_type=phase_type,
            phase_label=label,
            measure_ids=measure_ids,
            start_month=start_month,
            end_month=end_month,
            investment=investment,
            expected_savings=savings,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Rebate Matching
    # -------------------------------------------------------------------------

    def _phase_rebate_matching(
        self, input_data: ImplementationPlanInput
    ) -> PhaseResult:
        """Run UtilityRebateEngine to match measures to incentive programs."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        segment = input_data.customer_segment
        total_rebate = Decimal("0")

        for measure in input_data.measures_data:
            measure_id = measure.get("measure_id", "")
            if input_data.selected_measures and measure_id not in input_data.selected_measures:
                continue

            category = measure.get("category", "")
            program = REBATE_PROGRAMS.get(category)
            if not program:
                continue

            # Check segment eligibility
            if segment not in program["eligible_segments"]:
                warnings.append(
                    f"Measure {measure_id}: segment '{segment}' not eligible "
                    f"for {program['program_name']}"
                )
                continue

            # Calculate rebate: min(per-kWh rebate, max % of cost)
            savings_kwh = Decimal(str(measure.get("annual_savings_kwh", 0)))
            impl_cost = Decimal(str(measure.get("implementation_cost", 0)))

            rebate_by_kwh = savings_kwh * program["rebate_per_kwh_saved"]
            max_rebate = impl_cost * program["max_rebate_pct"] / Decimal("100")
            rebate = min(rebate_by_kwh, max_rebate)

            if rebate > 0:
                pct = (rebate / impl_cost * Decimal("100")) if impl_cost > 0 else Decimal("0")
                match = RebateMatch(
                    measure_id=measure_id,
                    program_name=program["program_name"],
                    rebate_amount=rebate.quantize(Decimal("0.01")),
                    rebate_pct_of_cost=pct.quantize(Decimal("0.01")),
                )
                self._rebate_matches.append(match)
                total_rebate += rebate

        outputs["rebates_matched"] = len(self._rebate_matches)
        outputs["total_rebate_amount"] = str(total_rebate.quantize(Decimal("0.01")))
        outputs["customer_segment"] = segment

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 RebateMatching: %d matches, total rebates=%.2f",
            len(self._rebate_matches), float(total_rebate),
        )
        return PhaseResult(
            phase_name="rebate_matching", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Behavioral Program Design
    # -------------------------------------------------------------------------

    def _phase_behavioral_program(
        self, input_data: ImplementationPlanInput
    ) -> PhaseResult:
        """Run BehavioralChangeEngine if behavioral actions are in scope."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Check if any selected measures are behavioral
        has_behavioral = any(
            m.get("category") == "behavioral"
            for m in input_data.measures_data
            if not input_data.selected_measures
            or m.get("measure_id") in input_data.selected_measures
        )

        if not has_behavioral:
            outputs["behavioral_program"] = None
            outputs["reason"] = "No behavioral measures in scope"

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self.logger.info("Phase 3 BehavioralProgram: skipped (no behavioral measures)")
            return PhaseResult(
                phase_name="behavioral_program", phase_number=3,
                status=PhaseStatus.SKIPPED, duration_ms=elapsed_ms,
                outputs=outputs, warnings=warnings,
                provenance_hash=self._hash_dict(outputs),
            )

        # Design behavioral program based on org profile
        employee_count = input_data.org_profile.get("employee_count", 100)
        template_key = "energy_champions" if employee_count > 200 else "awareness_campaign"
        template = BEHAVIORAL_TEMPLATES[template_key]

        program_cost = template["cost_per_employee"] * Decimal(str(employee_count))

        self._behavioral_program = {
            "program_name": template["name"],
            "template": template_key,
            "duration_weeks": template["duration_weeks"],
            "target_savings_pct": str(template["target_savings_pct"]),
            "total_cost": str(program_cost.quantize(Decimal("0.01"))),
            "employee_count": employee_count,
            "activities": template["activities"],
        }

        outputs["program_name"] = template["name"]
        outputs["duration_weeks"] = template["duration_weeks"]
        outputs["total_cost"] = str(program_cost.quantize(Decimal("0.01")))
        outputs["target_savings_pct"] = str(template["target_savings_pct"])

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 BehavioralProgram: %s, cost=%.0f, %d employees",
            template["name"], float(program_cost), employee_count,
        )
        return PhaseResult(
            phase_name="behavioral_program", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Plan Assembly
    # -------------------------------------------------------------------------

    def _phase_plan_assembly(
        self, input_data: ImplementationPlanInput
    ) -> PhaseResult:
        """Compile implementation plan with timeline and budget summary."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_investment = sum(p.investment for p in self._impl_phases)
        total_rebates = sum(r.rebate_amount for r in self._rebate_matches)
        total_savings = sum(p.expected_savings for p in self._impl_phases)
        net_investment = total_investment - total_rebates
        timeline = max((p.end_month for p in self._impl_phases), default=0)

        # Apply rebate allocation to phases
        remaining_rebate = total_rebates
        for phase in self._impl_phases:
            phase_share = min(remaining_rebate, phase.investment)
            phase.rebates = phase_share
            remaining_rebate -= phase_share

        # Simple payback on net investment
        portfolio_payback_months = (
            float(net_investment) / float(total_savings) * 12.0
            if total_savings > 0 else 0.0
        )

        outputs["total_investment"] = str(total_investment)
        outputs["total_rebates"] = str(total_rebates)
        outputs["net_investment"] = str(max(Decimal("0"), net_investment))
        outputs["total_annual_savings"] = str(total_savings)
        outputs["timeline_months"] = timeline
        outputs["portfolio_payback_months"] = round(portfolio_payback_months, 1)
        outputs["phases_count"] = len(self._impl_phases)
        outputs["behavioral_included"] = self._behavioral_program is not None

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 PlanAssembly: investment=%.0f rebates=%.0f net=%.0f timeline=%d months",
            float(total_investment), float(total_rebates),
            float(net_investment), timeline,
        )
        return PhaseResult(
            phase_name="plan_assembly", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ImplementationPlanResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
