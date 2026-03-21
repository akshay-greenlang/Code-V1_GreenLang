# -*- coding: utf-8 -*-
"""
Quick Wins Implementation Workflow
========================================

5-phase workflow for implementing SME quick-win carbon reduction
actions within PACK-026 SME Net Zero Pack.  Takes selected quick
wins from identification through vendor research, cost-benefit
analysis, implementation planning, and post-implementation
verification.

Phases:
    1. ActionSelection       -- Select 3-5 quick wins from recommended list
    2. VendorResearch        -- Research implementation vendors/suppliers
    3. CostBenefit           -- Detailed cost-benefit for selected actions
    4. Implementation        -- Implementation timeline + project plan
    5. Verification          -- Post-implementation verification (savings achieved?)

Uses: quick_wins_engine, action_prioritization_engine, cost_benefit_engine.

Zero-hallucination: all calculations deterministic.
SHA-256 provenance hashes for auditability.

Author: GreenLang Team
Version: 26.0.0
Pack: PACK-026 SME Net Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "26.0.0"
_PACK_ID = "PACK-026"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ActionStatus(str, Enum):
    NOT_STARTED = "not_started"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    CANCELLED = "cancelled"


class VendorStatus(str, Enum):
    IDENTIFIED = "identified"
    CONTACTED = "contacted"
    QUOTED = "quoted"
    SELECTED = "selected"
    CONTRACTED = "contracted"


class VerificationResult(str, Enum):
    EXCEEDED = "exceeded"          # Savings better than expected
    MET = "met"                    # Savings as expected (+/-10%)
    PARTIAL = "partial"            # Savings below expected (>50%)
    UNDERPERFORMED = "underperformed"  # Savings significantly below (<50%)
    NOT_VERIFIED = "not_verified"  # Not yet verified


# =============================================================================
# COST-BENEFIT CONSTANTS
# =============================================================================

# Carbon price for social cost of carbon (GBP/tCO2e)
SOCIAL_COST_CARBON_GBP = 85.0

# Discount rate for NPV calculations
DISCOUNT_RATE = 0.05

# Standard project lifetimes by category (years)
PROJECT_LIFETIME_YEARS: Dict[str, int] = {
    "lighting": 10,
    "heating_cooling": 15,
    "energy_efficiency": 10,
    "renewable_energy": 25,
    "transport": 8,
    "behaviour_change": 5,
    "waste_reduction": 5,
    "procurement": 5,
    "digital": 5,
    "water": 10,
}

# Vendor categories by action type
VENDOR_CATEGORIES: Dict[str, List[Dict[str, str]]] = {
    "lighting": [
        {"type": "LED supplier", "description": "LED lighting manufacturers and distributors"},
        {"type": "Electrical contractor", "description": "Licensed electrical installation contractors"},
        {"type": "Energy consultant", "description": "Lighting design and energy audit specialists"},
    ],
    "heating_cooling": [
        {"type": "HVAC contractor", "description": "Heating, ventilation, and AC installers"},
        {"type": "Controls supplier", "description": "Smart thermostat and BMS suppliers"},
        {"type": "Heat pump installer", "description": "MCS-certified heat pump installers"},
    ],
    "energy_efficiency": [
        {"type": "Insulation contractor", "description": "Cavity, loft, and external wall insulation"},
        {"type": "Glazing supplier", "description": "Double/triple glazing window suppliers"},
        {"type": "Energy auditor", "description": "Certified energy assessment providers"},
    ],
    "renewable_energy": [
        {"type": "Solar installer", "description": "MCS-certified solar PV installers"},
        {"type": "Battery supplier", "description": "Energy storage system providers"},
        {"type": "PPA provider", "description": "Power purchase agreement brokers"},
    ],
    "transport": [
        {"type": "EV dealer", "description": "Electric vehicle sales and leasing"},
        {"type": "Charging installer", "description": "EV charge point installers"},
        {"type": "Fleet management", "description": "Fleet electrification consultants"},
    ],
    "waste_reduction": [
        {"type": "Waste management", "description": "Commercial recycling and waste services"},
        {"type": "Circular economy", "description": "Waste-to-resource consultants"},
    ],
    "procurement": [
        {"type": "Sustainability consultant", "description": "Green procurement advisory"},
        {"type": "Supply chain platform", "description": "Sustainable supplier databases"},
    ],
    "behaviour_change": [
        {"type": "Sustainability trainer", "description": "Staff engagement and training"},
        {"type": "Communications agency", "description": "Internal sustainability campaigns"},
    ],
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    mobile_summary: str = Field(default="")


class QuickWinCandidate(BaseModel):
    """A quick win candidate for selection."""

    action_id: str = Field(default="")
    title: str = Field(default="")
    category: str = Field(default="")
    estimated_savings_tco2e: float = Field(default=0.0, ge=0.0)
    estimated_savings_gbp: float = Field(default=0.0, ge=0.0)
    implementation_cost_gbp: float = Field(default=0.0, ge=0.0)
    payback_months: int = Field(default=0, ge=0)
    difficulty: str = Field(default="easy")
    selected: bool = Field(default=False)
    priority: int = Field(default=0, ge=0)


class VendorOption(BaseModel):
    """A potential vendor/supplier for implementation."""

    vendor_type: str = Field(default="")
    description: str = Field(default="")
    search_terms: List[str] = Field(default_factory=list)
    certification_to_look_for: List[str] = Field(default_factory=list)
    estimated_quotes_needed: int = Field(default=3, ge=1)
    status: str = Field(default="identified")
    notes: str = Field(default="")


class CostBenefitDetail(BaseModel):
    """Detailed cost-benefit analysis for an action."""

    action_id: str = Field(default="")
    title: str = Field(default="")
    category: str = Field(default="")
    implementation_cost_gbp: float = Field(default=0.0, ge=0.0)
    annual_energy_savings_gbp: float = Field(default=0.0, ge=0.0)
    annual_carbon_savings_tco2e: float = Field(default=0.0, ge=0.0)
    annual_carbon_value_gbp: float = Field(default=0.0, ge=0.0)
    total_annual_benefit_gbp: float = Field(default=0.0, ge=0.0)
    simple_payback_years: float = Field(default=0.0, ge=0.0)
    npv_gbp: float = Field(default=0.0)
    irr_pct: float = Field(default=0.0)
    project_lifetime_years: int = Field(default=10, ge=1)
    lifetime_savings_gbp: float = Field(default=0.0)
    lifetime_carbon_savings_tco2e: float = Field(default=0.0, ge=0.0)
    roi_pct: float = Field(default=0.0)
    grant_potential_gbp: float = Field(default=0.0, ge=0.0)
    net_cost_after_grants_gbp: float = Field(default=0.0)


class ImplementationMilestone(BaseModel):
    """Implementation timeline milestone."""

    milestone_id: str = Field(default="")
    title: str = Field(default="")
    description: str = Field(default="")
    week_start: int = Field(default=1, ge=1)
    week_end: int = Field(default=1, ge=1)
    responsible: str = Field(default="")
    dependencies: List[str] = Field(default_factory=list)
    status: str = Field(default="not_started")


class ImplementationPlan(BaseModel):
    """Complete implementation plan for an action."""

    action_id: str = Field(default="")
    title: str = Field(default="")
    total_duration_weeks: int = Field(default=4, ge=1)
    milestones: List[ImplementationMilestone] = Field(default_factory=list)
    resources_needed: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """Post-implementation verification report."""

    action_id: str = Field(default="")
    title: str = Field(default="")
    verification_status: str = Field(default="not_verified")
    expected_savings_tco2e: float = Field(default=0.0, ge=0.0)
    actual_savings_tco2e: float = Field(default=0.0, ge=0.0)
    savings_achievement_pct: float = Field(default=0.0, ge=0.0, le=200.0)
    expected_cost_savings_gbp: float = Field(default=0.0, ge=0.0)
    actual_cost_savings_gbp: float = Field(default=0.0, ge=0.0)
    verification_method: str = Field(default="spend_comparison")
    verification_period_months: int = Field(default=3, ge=1)
    notes: List[str] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)


class QuickWinsImplementationConfig(BaseModel):
    """Configuration for quick wins implementation workflow."""

    baseline_tco2e: float = Field(default=0.0, ge=0.0)
    max_selections: int = Field(default=5, ge=1, le=10)
    budget_gbp: float = Field(default=10000, ge=0.0)
    verification_period_months: int = Field(default=3, ge=1, le=12)
    include_vendor_research: bool = Field(default=True)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class QuickWinsImplementationInput(BaseModel):
    """Complete input for quick wins implementation workflow."""

    candidates: List[QuickWinCandidate] = Field(
        default_factory=list, description="Quick win candidates to consider"
    )
    selected_action_ids: List[str] = Field(
        default_factory=list, description="Pre-selected action IDs (if any)"
    )
    config: QuickWinsImplementationConfig = Field(
        default_factory=QuickWinsImplementationConfig,
    )


class QuickWinsImplementationResult(BaseModel):
    """Complete result from quick wins implementation workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="sme_quick_wins_implementation")
    pack_id: str = Field(default="PACK-026")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    selected_actions: List[QuickWinCandidate] = Field(default_factory=list)
    vendor_options: Dict[str, List[VendorOption]] = Field(default_factory=dict)
    cost_benefit_analyses: List[CostBenefitDetail] = Field(default_factory=list)
    implementation_plans: List[ImplementationPlan] = Field(default_factory=list)
    verification_reports: List[VerificationReport] = Field(default_factory=list)
    total_expected_savings_tco2e: float = Field(default=0.0, ge=0.0)
    total_implementation_cost_gbp: float = Field(default=0.0, ge=0.0)
    total_annual_savings_gbp: float = Field(default=0.0, ge=0.0)
    aggregate_npv_gbp: float = Field(default=0.0)
    next_steps: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class QuickWinsImplementationWorkflow:
    """
    5-phase workflow for implementing SME quick-win carbon reduction actions.

    Phase 1: Action Selection (choose 3-5 from recommended list)
    Phase 2: Vendor Research (identify suppliers and contractors)
    Phase 3: Cost-Benefit Analysis (NPV, IRR, payback)
    Phase 4: Implementation Planning (timeline, milestones, resources)
    Phase 5: Verification (post-implementation savings measurement)

    Example:
        >>> wf = QuickWinsImplementationWorkflow()
        >>> inp = QuickWinsImplementationInput(
        ...     candidates=[
        ...         QuickWinCandidate(action_id="led_upgrade", title="LED Lighting", ...),
        ...     ],
        ...     config=QuickWinsImplementationConfig(baseline_tco2e=100.0),
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self) -> None:
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._selected: List[QuickWinCandidate] = []
        self._vendors: Dict[str, List[VendorOption]] = {}
        self._cost_benefits: List[CostBenefitDetail] = []
        self._impl_plans: List[ImplementationPlan] = []
        self._verifications: List[VerificationReport] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: QuickWinsImplementationInput) -> QuickWinsImplementationResult:
        """Execute the 5-phase quick wins implementation workflow."""
        started_at = _utcnow()
        self.logger.info("Starting quick wins implementation %s", self.workflow_id)
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_action_selection(input_data)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError(f"ActionSelection failed: {phase1.errors}")

            phase2 = await self._phase_vendor_research(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_cost_benefit(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_implementation(input_data)
            self._phase_results.append(phase4)

            phase5 = await self._phase_verification(input_data)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Quick wins implementation failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
                mobile_summary="Implementation workflow failed.",
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        total_savings = sum(a.estimated_savings_tco2e for a in self._selected)
        total_cost = sum(a.implementation_cost_gbp for a in self._selected)
        total_annual_gbp = sum(a.estimated_savings_gbp for a in self._selected)
        aggregate_npv = sum(cb.npv_gbp for cb in self._cost_benefits)

        next_steps = self._generate_next_steps()

        result = QuickWinsImplementationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            selected_actions=self._selected,
            vendor_options=self._vendors,
            cost_benefit_analyses=self._cost_benefits,
            implementation_plans=self._impl_plans,
            verification_reports=self._verifications,
            total_expected_savings_tco2e=round(total_savings, 4),
            total_implementation_cost_gbp=round(total_cost, 2),
            total_annual_savings_gbp=round(total_annual_gbp, 2),
            aggregate_npv_gbp=round(aggregate_npv, 2),
            next_steps=next_steps,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Action Selection
    # -------------------------------------------------------------------------

    async def _phase_action_selection(self, inp: QuickWinsImplementationInput) -> PhaseResult:
        """Select 3-5 quick wins based on priority and budget."""
        started = _utcnow()
        warnings: List[str] = []
        errors: List[str] = []
        outputs: Dict[str, Any] = {}

        candidates = inp.candidates
        selected_ids = set(inp.selected_action_ids)
        max_sel = inp.config.max_selections
        budget = inp.config.budget_gbp

        if not candidates:
            errors.append("No quick win candidates provided")
            return PhaseResult(
                phase_name="action_selection", phase_number=1,
                status=PhaseStatus.FAILED, errors=errors,
                mobile_summary="No actions to select.",
            )

        # Mark pre-selected
        for c in candidates:
            if c.action_id in selected_ids:
                c.selected = True

        # If no pre-selection, auto-select by priority and budget
        if not selected_ids:
            sorted_candidates = sorted(
                candidates,
                key=lambda c: (
                    -c.estimated_savings_tco2e,
                    c.implementation_cost_gbp,
                ),
            )
            running_cost = 0.0
            for i, c in enumerate(sorted_candidates):
                if i >= max_sel:
                    break
                if running_cost + c.implementation_cost_gbp <= budget or c.implementation_cost_gbp == 0:
                    c.selected = True
                    c.priority = i + 1
                    running_cost += c.implementation_cost_gbp

        self._selected = [c for c in candidates if c.selected]

        if not self._selected:
            warnings.append("No actions selected within budget; consider increasing budget")
            # Select top action regardless of budget
            if candidates:
                candidates[0].selected = True
                candidates[0].priority = 1
                self._selected = [candidates[0]]

        total_cost = sum(a.implementation_cost_gbp for a in self._selected)
        total_savings = sum(a.estimated_savings_tco2e for a in self._selected)

        outputs["selected_count"] = len(self._selected)
        outputs["total_implementation_cost_gbp"] = round(total_cost, 2)
        outputs["total_expected_savings_tco2e"] = round(total_savings, 4)
        outputs["within_budget"] = total_cost <= budget

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="action_selection", phase_number=1,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Selected {len(self._selected)} actions ({total_savings:.1f} tCO2e savings)",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Vendor Research
    # -------------------------------------------------------------------------

    async def _phase_vendor_research(self, inp: QuickWinsImplementationInput) -> PhaseResult:
        """Research vendors and suppliers for selected actions."""
        started = _utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not inp.config.include_vendor_research:
            return PhaseResult(
                phase_name="vendor_research", phase_number=2,
                status=PhaseStatus.SKIPPED,
                completion_pct=100.0,
                mobile_summary="Vendor research skipped.",
            )

        self._vendors = {}
        total_vendors = 0

        for action in self._selected:
            category = action.category
            vendor_templates = VENDOR_CATEGORIES.get(category, [])

            vendors: List[VendorOption] = []
            for vt in vendor_templates:
                # Generate search terms
                search_terms = [
                    f"{vt['type']} near me",
                    f"local {vt['type']}",
                    f"certified {vt['type']}",
                    f"SME {vt['type']} services",
                ]

                # Certification hints
                certs: List[str] = []
                if category in {"renewable_energy", "heating_cooling"}:
                    certs.append("MCS Certified")
                if category in {"lighting", "energy_efficiency"}:
                    certs.append("NICEIC Approved")
                certs.append("ISO 14001")
                certs.append("Carbon Trust accredited")

                vendor = VendorOption(
                    vendor_type=vt["type"],
                    description=vt["description"],
                    search_terms=search_terms,
                    certification_to_look_for=certs,
                    estimated_quotes_needed=3,
                    status=VendorStatus.IDENTIFIED.value,
                )
                vendors.append(vendor)
                total_vendors += 1

            self._vendors[action.action_id] = vendors

        outputs["actions_researched"] = len(self._vendors)
        outputs["total_vendor_types"] = total_vendors

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="vendor_research", phase_number=2,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Identified {total_vendors} vendor types across {len(self._vendors)} actions",
        )

    # -------------------------------------------------------------------------
    # Phase 3: Cost-Benefit Analysis
    # -------------------------------------------------------------------------

    async def _phase_cost_benefit(self, inp: QuickWinsImplementationInput) -> PhaseResult:
        """Detailed cost-benefit analysis with NPV and IRR."""
        started = _utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._cost_benefits = []

        for action in self._selected:
            lifetime = PROJECT_LIFETIME_YEARS.get(action.category, 10)
            impl_cost = action.implementation_cost_gbp
            annual_energy_savings = action.estimated_savings_gbp
            annual_carbon_savings = action.estimated_savings_tco2e
            annual_carbon_value = annual_carbon_savings * SOCIAL_COST_CARBON_GBP
            total_annual_benefit = annual_energy_savings + annual_carbon_value

            # Simple payback
            payback = impl_cost / max(annual_energy_savings, 0.01) if annual_energy_savings > 0 else 99.0
            payback = min(payback, 99.0)

            # NPV calculation
            npv = -impl_cost
            for year in range(1, lifetime + 1):
                npv += total_annual_benefit / ((1 + DISCOUNT_RATE) ** year)

            # Lifetime savings
            lifetime_savings = annual_energy_savings * lifetime
            lifetime_carbon = annual_carbon_savings * lifetime

            # ROI
            roi = ((lifetime_savings - impl_cost) / max(impl_cost, 1)) * 100 if impl_cost > 0 else 0

            # IRR approximation (simplified Newton's method)
            irr = self._estimate_irr(impl_cost, total_annual_benefit, lifetime)

            # Grant potential (estimate 30-50% for eligible projects)
            grant_pct = 0.40 if impl_cost > 1000 else 0.0
            grant_potential = impl_cost * grant_pct
            net_cost = impl_cost - grant_potential

            cb = CostBenefitDetail(
                action_id=action.action_id,
                title=action.title,
                category=action.category,
                implementation_cost_gbp=round(impl_cost, 2),
                annual_energy_savings_gbp=round(annual_energy_savings, 2),
                annual_carbon_savings_tco2e=round(annual_carbon_savings, 4),
                annual_carbon_value_gbp=round(annual_carbon_value, 2),
                total_annual_benefit_gbp=round(total_annual_benefit, 2),
                simple_payback_years=round(payback, 1),
                npv_gbp=round(npv, 2),
                irr_pct=round(irr, 1),
                project_lifetime_years=lifetime,
                lifetime_savings_gbp=round(lifetime_savings, 2),
                lifetime_carbon_savings_tco2e=round(lifetime_carbon, 4),
                roi_pct=round(roi, 1),
                grant_potential_gbp=round(grant_potential, 2),
                net_cost_after_grants_gbp=round(net_cost, 2),
            )
            self._cost_benefits.append(cb)

        total_npv = sum(cb.npv_gbp for cb in self._cost_benefits)
        positive_npv = sum(1 for cb in self._cost_benefits if cb.npv_gbp > 0)

        outputs["analyses_completed"] = len(self._cost_benefits)
        outputs["total_npv_gbp"] = round(total_npv, 2)
        outputs["positive_npv_count"] = positive_npv
        outputs["avg_payback_years"] = round(
            sum(cb.simple_payback_years for cb in self._cost_benefits)
            / max(len(self._cost_benefits), 1), 1,
        )

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="cost_benefit", phase_number=3,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"NPV: GBP {total_npv:,.0f} ({positive_npv}/{len(self._cost_benefits)} positive)",
        )

    def _estimate_irr(
        self, initial_cost: float, annual_benefit: float, lifetime: int,
    ) -> float:
        """Estimate IRR using bisection method."""
        if initial_cost <= 0 or annual_benefit <= 0:
            return 0.0

        low, high = -0.5, 5.0
        for _ in range(50):
            mid = (low + high) / 2
            npv = -initial_cost
            for yr in range(1, lifetime + 1):
                npv += annual_benefit / ((1 + mid) ** yr)
            if abs(npv) < 0.01:
                return mid * 100
            if npv > 0:
                low = mid
            else:
                high = mid
        return ((low + high) / 2) * 100

    # -------------------------------------------------------------------------
    # Phase 4: Implementation Planning
    # -------------------------------------------------------------------------

    async def _phase_implementation(self, inp: QuickWinsImplementationInput) -> PhaseResult:
        """Generate implementation timelines and project plans."""
        started = _utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._impl_plans = []

        for action in self._selected:
            plan = self._generate_implementation_plan(action)
            self._impl_plans.append(plan)

        total_weeks = max(
            (p.total_duration_weeks for p in self._impl_plans),
            default=0,
        )

        outputs["plans_created"] = len(self._impl_plans)
        outputs["total_duration_weeks"] = total_weeks
        outputs["total_milestones"] = sum(len(p.milestones) for p in self._impl_plans)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="implementation", phase_number=4,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"{len(self._impl_plans)} plans created (~{total_weeks} weeks total)",
        )

    def _generate_implementation_plan(self, action: QuickWinCandidate) -> ImplementationPlan:
        """Generate a standard implementation plan for an action."""
        difficulty = action.difficulty
        base_weeks = {"easy": 4, "medium": 8, "hard": 16}.get(difficulty, 8)

        milestones: List[ImplementationMilestone] = [
            ImplementationMilestone(
                milestone_id=f"{action.action_id}_m1",
                title="Planning and approval",
                description="Secure budget approval and define project scope",
                week_start=1,
                week_end=max(base_weeks // 4, 1),
                responsible="Project lead",
            ),
            ImplementationMilestone(
                milestone_id=f"{action.action_id}_m2",
                title="Vendor selection",
                description="Obtain quotes, compare options, select vendor",
                week_start=max(base_weeks // 4, 1) + 1,
                week_end=max(base_weeks // 2, 2),
                responsible="Procurement",
                dependencies=[f"{action.action_id}_m1"],
            ),
            ImplementationMilestone(
                milestone_id=f"{action.action_id}_m3",
                title="Implementation",
                description="Execute the installation or change",
                week_start=max(base_weeks // 2, 2) + 1,
                week_end=max(base_weeks * 3 // 4, 3),
                responsible="Vendor / Operations",
                dependencies=[f"{action.action_id}_m2"],
            ),
            ImplementationMilestone(
                milestone_id=f"{action.action_id}_m4",
                title="Testing and commissioning",
                description="Verify installation, test operation, sign off",
                week_start=max(base_weeks * 3 // 4, 3) + 1,
                week_end=base_weeks,
                responsible="Operations manager",
                dependencies=[f"{action.action_id}_m3"],
            ),
        ]

        resources = [
            "Budget approval",
            f"Vendor/contractor for {action.category}",
            "Staff time for coordination",
        ]
        if action.implementation_cost_gbp > 5000:
            resources.append("Grant application (if available)")

        risks = [
            "Vendor availability delays",
            "Actual costs exceeding estimates",
            "Operational disruption during installation",
        ]

        success_criteria = [
            f"Installation completed within {base_weeks} weeks",
            f"Energy savings of {action.estimated_savings_tco2e:.1f} tCO2e/year verified",
            f"Cost savings of GBP {action.estimated_savings_gbp:.0f}/year achieved",
        ]

        return ImplementationPlan(
            action_id=action.action_id,
            title=action.title,
            total_duration_weeks=base_weeks,
            milestones=milestones,
            resources_needed=resources,
            risks=risks,
            success_criteria=success_criteria,
        )

    # -------------------------------------------------------------------------
    # Phase 5: Verification
    # -------------------------------------------------------------------------

    async def _phase_verification(self, inp: QuickWinsImplementationInput) -> PhaseResult:
        """Set up post-implementation verification framework."""
        started = _utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._verifications = []
        period = inp.config.verification_period_months

        for action in self._selected:
            report = VerificationReport(
                action_id=action.action_id,
                title=action.title,
                verification_status=VerificationResult.NOT_VERIFIED.value,
                expected_savings_tco2e=action.estimated_savings_tco2e,
                actual_savings_tco2e=0.0,
                savings_achievement_pct=0.0,
                expected_cost_savings_gbp=action.estimated_savings_gbp,
                actual_cost_savings_gbp=0.0,
                verification_method="spend_comparison",
                verification_period_months=period,
                notes=[
                    f"Verification to begin {period} months after implementation",
                    "Compare energy bills before/after implementation",
                    "Account for seasonal variations and occupancy changes",
                ],
                lessons_learned=[],
            )
            self._verifications.append(report)

        outputs["verification_frameworks"] = len(self._verifications)
        outputs["verification_period_months"] = period
        outputs["total_expected_savings_tco2e"] = round(
            sum(v.expected_savings_tco2e for v in self._verifications), 4,
        )

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="verification", phase_number=5,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Verification set for {len(self._verifications)} actions ({period} month period)",
        )

    # -------------------------------------------------------------------------
    # Next Steps
    # -------------------------------------------------------------------------

    def _generate_next_steps(self) -> List[str]:
        steps: List[str] = []

        if self._selected:
            steps.append(
                f"Begin implementation of {self._selected[0].title} (highest priority)."
            )

        if self._vendors:
            steps.append("Contact identified vendors and obtain 3 quotes per action.")

        positive_npv = [cb for cb in self._cost_benefits if cb.npv_gbp > 0]
        if positive_npv:
            steps.append(
                f"{len(positive_npv)} actions have positive NPV - prioritise these."
            )

        grant_eligible = [
            cb for cb in self._cost_benefits if cb.grant_potential_gbp > 0
        ]
        if grant_eligible:
            total_grant = sum(cb.grant_potential_gbp for cb in grant_eligible)
            steps.append(
                f"Apply for grants - potential GBP {total_grant:,.0f} in funding."
            )

        steps.append(
            "Schedule verification reviews after implementation completion."
        )
        steps.append(
            "Update quarterly review with implementation progress."
        )

        return steps
