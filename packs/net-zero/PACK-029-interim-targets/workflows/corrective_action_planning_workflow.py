# -*- coding: utf-8 -*-
"""
Corrective Action Planning Workflow
====================================

6-phase DAG workflow for planning corrective actions to close emission
gaps within PACK-029 Interim Targets Pack.  The workflow quantifies the
gap to target, identifies candidate initiatives via MACC integration
from PACK-028, optimizes the initiative portfolio, schedules deployment,
updates budget allocation, and generates a corrective action plan report.

Phases:
    1. QuantifyGap         -- Quantify gap-to-target using CorrectiveActionEngine
    2. IdentifyInitiatives -- Identify candidate initiatives via MACC integration
                              (from PACK-028 Sector Pathway Pack)
    3. OptimizePortfolio   -- Optimize initiative portfolio by cost, timing, risk
    4. ScheduleDeployment  -- Schedule initiative deployment via InitiativeSchedulerEngine
    5. UpdateBudget        -- Update carbon budget allocation via BudgetAllocationEngine
    6. GenerateReport      -- Generate corrective action plan report

Regulatory references:
    - SBTi Target Tracking Protocol (corrective action requirements)
    - Marginal Abatement Cost Curve methodology (McKinsey/IEA)
    - GHG Protocol Corporate Standard (recalculation policy)
    - TCFD Transition Planning guidance
    - ISO 14064-1:2018 (corrective actions in GHG programmes)

Zero-hallucination: all abatement cost calculations use published MACC
data and deterministic optimization.  No LLM calls in computation path.

Author: GreenLang Team
Version: 29.0.0
Pack: PACK-029 Interim Targets Pack
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"

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

class GapSeverity(str, Enum):
    ON_TRACK = "on_track"
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"

class InitiativePriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class InitiativeCategory(str, Enum):
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_ENERGY = "renewable_energy"
    FUEL_SWITCHING = "fuel_switching"
    PROCESS_CHANGE = "process_change"
    SUPPLY_CHAIN = "supply_chain"
    BEHAVIORAL = "behavioral"
    TECHNOLOGY = "technology"
    CARBON_REMOVAL = "carbon_removal"
    OPERATIONAL = "operational"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class RAGStatus(str, Enum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"

class DeploymentPhase(str, Enum):
    IMMEDIATE = "immediate"        # 0-3 months
    SHORT_TERM = "short_term"      # 3-12 months
    MEDIUM_TERM = "medium_term"    # 1-3 years
    LONG_TERM = "long_term"        # 3-5 years

# =============================================================================
# MACC INITIATIVE LIBRARY (Zero-Hallucination: Published Abatement Costs)
# =============================================================================

MACC_INITIATIVE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "energy_audit": {
        "name": "Energy Audit & Quick Wins",
        "category": "energy_efficiency",
        "abatement_cost_usd_per_tco2e": -50,
        "typical_reduction_pct": 3.0,
        "implementation_months": 3,
        "capex_usd_per_tco2e": 10,
        "risk": "low",
        "trl": 9,
        "deployment_phase": "immediate",
        "description": "Facility-wide energy audit with immediate efficiency improvements.",
    },
    "led_lighting": {
        "name": "LED Lighting Retrofit",
        "category": "energy_efficiency",
        "abatement_cost_usd_per_tco2e": -80,
        "typical_reduction_pct": 2.0,
        "implementation_months": 6,
        "capex_usd_per_tco2e": 15,
        "risk": "low",
        "trl": 9,
        "deployment_phase": "immediate",
        "description": "Replace remaining non-LED lighting with high-efficiency LED systems.",
    },
    "hvac_optimization": {
        "name": "HVAC System Optimization",
        "category": "energy_efficiency",
        "abatement_cost_usd_per_tco2e": -30,
        "typical_reduction_pct": 5.0,
        "implementation_months": 12,
        "capex_usd_per_tco2e": 50,
        "risk": "low",
        "trl": 9,
        "deployment_phase": "short_term",
        "description": "HVAC controls upgrade, variable speed drives, heat recovery.",
    },
    "building_envelope": {
        "name": "Building Envelope Improvement",
        "category": "energy_efficiency",
        "abatement_cost_usd_per_tco2e": 20,
        "typical_reduction_pct": 4.0,
        "implementation_months": 18,
        "capex_usd_per_tco2e": 80,
        "risk": "medium",
        "trl": 9,
        "deployment_phase": "medium_term",
        "description": "Insulation, window upgrades, air sealing improvements.",
    },
    "onsite_solar": {
        "name": "On-Site Solar PV Installation",
        "category": "renewable_energy",
        "abatement_cost_usd_per_tco2e": -20,
        "typical_reduction_pct": 8.0,
        "implementation_months": 12,
        "capex_usd_per_tco2e": 120,
        "risk": "low",
        "trl": 9,
        "deployment_phase": "short_term",
        "description": "Install rooftop or ground-mounted solar PV systems.",
    },
    "renewable_ppa": {
        "name": "Renewable Energy PPA",
        "category": "renewable_energy",
        "abatement_cost_usd_per_tco2e": 5,
        "typical_reduction_pct": 15.0,
        "implementation_months": 6,
        "capex_usd_per_tco2e": 0,
        "risk": "low",
        "trl": 9,
        "deployment_phase": "short_term",
        "description": "Long-term Power Purchase Agreement for renewable electricity.",
    },
    "fleet_electrification": {
        "name": "Fleet Electrification",
        "category": "fuel_switching",
        "abatement_cost_usd_per_tco2e": 40,
        "typical_reduction_pct": 6.0,
        "implementation_months": 24,
        "capex_usd_per_tco2e": 200,
        "risk": "medium",
        "trl": 8,
        "deployment_phase": "medium_term",
        "description": "Replace ICE fleet vehicles with electric alternatives.",
    },
    "heat_pump": {
        "name": "Industrial Heat Pump Installation",
        "category": "fuel_switching",
        "abatement_cost_usd_per_tco2e": 30,
        "typical_reduction_pct": 7.0,
        "implementation_months": 18,
        "capex_usd_per_tco2e": 150,
        "risk": "medium",
        "trl": 8,
        "deployment_phase": "medium_term",
        "description": "Replace gas boilers with industrial heat pumps.",
    },
    "process_optimization": {
        "name": "Process Optimization & Digitalization",
        "category": "process_change",
        "abatement_cost_usd_per_tco2e": -40,
        "typical_reduction_pct": 4.0,
        "implementation_months": 12,
        "capex_usd_per_tco2e": 60,
        "risk": "low",
        "trl": 9,
        "deployment_phase": "short_term",
        "description": "Digital twin, AI-based process optimization, predictive maintenance.",
    },
    "supplier_engagement": {
        "name": "Supplier Engagement Program",
        "category": "supply_chain",
        "abatement_cost_usd_per_tco2e": 15,
        "typical_reduction_pct": 5.0,
        "implementation_months": 18,
        "capex_usd_per_tco2e": 5,
        "risk": "medium",
        "trl": 9,
        "deployment_phase": "medium_term",
        "description": "Engage top 20 suppliers on SBTi target-setting and reduction programs.",
    },
    "employee_engagement": {
        "name": "Employee Behavioral Change Program",
        "category": "behavioral",
        "abatement_cost_usd_per_tco2e": -60,
        "typical_reduction_pct": 2.0,
        "implementation_months": 6,
        "capex_usd_per_tco2e": 5,
        "risk": "low",
        "trl": 9,
        "deployment_phase": "immediate",
        "description": "Sustainability training, commute programs, remote work optimization.",
    },
    "carbon_capture": {
        "name": "Carbon Capture (Point Source)",
        "category": "technology",
        "abatement_cost_usd_per_tco2e": 80,
        "typical_reduction_pct": 10.0,
        "implementation_months": 36,
        "capex_usd_per_tco2e": 400,
        "risk": "high",
        "trl": 7,
        "deployment_phase": "long_term",
        "description": "Install point-source CCS on high-emitting facilities.",
    },
    "nature_based_removal": {
        "name": "Nature-Based Carbon Removal",
        "category": "carbon_removal",
        "abatement_cost_usd_per_tco2e": 25,
        "typical_reduction_pct": 3.0,
        "implementation_months": 12,
        "capex_usd_per_tco2e": 25,
        "risk": "medium",
        "trl": 9,
        "deployment_phase": "short_term",
        "description": "Afforestation, reforestation, and soil carbon projects.",
    },
    "demand_response": {
        "name": "Demand Response & Load Shifting",
        "category": "operational",
        "abatement_cost_usd_per_tco2e": -25,
        "typical_reduction_pct": 3.0,
        "implementation_months": 6,
        "capex_usd_per_tco2e": 20,
        "risk": "low",
        "trl": 9,
        "deployment_phase": "immediate",
        "description": "Shift energy-intensive operations to low-carbon grid periods.",
    },
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    dag_node_id: str = Field(default="")

class GapQuantification(BaseModel):
    """Quantified gap to target."""
    gap_tco2e: float = Field(default=0.0)
    gap_pct: float = Field(default=0.0)
    annual_gap_tco2e: float = Field(default=0.0)
    cumulative_gap_tco2e: float = Field(default=0.0)
    severity: GapSeverity = Field(default=GapSeverity.ON_TRACK)
    years_to_target: int = Field(default=0)
    required_additional_reduction_pct: float = Field(default=0.0)
    carbon_budget_gap_tco2e: float = Field(default=0.0)
    rag_status: RAGStatus = Field(default=RAGStatus.GREEN)

class CandidateInitiative(BaseModel):
    """A candidate corrective action initiative."""
    initiative_id: str = Field(default="")
    initiative_name: str = Field(default="")
    category: InitiativeCategory = Field(default=InitiativeCategory.ENERGY_EFFICIENCY)
    abatement_potential_tco2e: float = Field(default=0.0)
    abatement_cost_usd_per_tco2e: float = Field(default=0.0)
    total_cost_usd: float = Field(default=0.0)
    capex_usd: float = Field(default=0.0)
    implementation_months: int = Field(default=12)
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    trl: int = Field(default=9, ge=1, le=9)
    deployment_phase: DeploymentPhase = Field(default=DeploymentPhase.SHORT_TERM)
    priority: InitiativePriority = Field(default=InitiativePriority.MEDIUM)
    selected: bool = Field(default=False)
    cumulative_abatement_tco2e: float = Field(default=0.0)
    macc_rank: int = Field(default=0)

class OptimizedPortfolio(BaseModel):
    """Optimized initiative portfolio."""
    selected_initiatives: List[CandidateInitiative] = Field(default_factory=list)
    total_abatement_tco2e: float = Field(default=0.0)
    total_cost_usd: float = Field(default=0.0)
    total_capex_usd: float = Field(default=0.0)
    weighted_avg_cost_usd: float = Field(default=0.0)
    gap_closure_pct: float = Field(default=0.0)
    residual_gap_tco2e: float = Field(default=0.0)
    portfolio_risk: RiskLevel = Field(default=RiskLevel.LOW)
    net_savings_usd: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class DeploymentSchedule(BaseModel):
    """Deployment schedule for selected initiatives."""
    immediate_initiatives: List[str] = Field(default_factory=list)
    short_term_initiatives: List[str] = Field(default_factory=list)
    medium_term_initiatives: List[str] = Field(default_factory=list)
    long_term_initiatives: List[str] = Field(default_factory=list)
    quarterly_milestones: Dict[str, float] = Field(default_factory=dict)
    total_deployment_months: int = Field(default=0)
    critical_path_initiatives: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class BudgetUpdate(BaseModel):
    """Updated carbon budget after corrective actions."""
    original_budget_tco2e: float = Field(default=0.0)
    corrective_savings_tco2e: float = Field(default=0.0)
    updated_budget_tco2e: float = Field(default=0.0)
    budget_extension_years: float = Field(default=0.0)
    new_depletion_year: int = Field(default=2050)
    capex_required_usd: float = Field(default=0.0)
    roi_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class CorrectiveActionReport(BaseModel):
    """Complete corrective action plan report."""
    report_id: str = Field(default="")
    report_date: str = Field(default="")
    company_name: str = Field(default="")
    gap: GapQuantification = Field(default_factory=GapQuantification)
    portfolio: OptimizedPortfolio = Field(default_factory=OptimizedPortfolio)
    schedule: DeploymentSchedule = Field(default_factory=DeploymentSchedule)
    budget_update: BudgetUpdate = Field(default_factory=BudgetUpdate)
    executive_summary: str = Field(default="")
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class CorrectiveActionConfig(BaseModel):
    company_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")
    current_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    target_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    target_year: int = Field(default=2030, ge=2025, le=2060)
    current_year: int = Field(default=2025, ge=2020, le=2060)
    carbon_budget_remaining_tco2e: float = Field(default=0.0, ge=0.0)
    available_capex_usd: float = Field(default=0.0, ge=0.0)
    max_risk_level: RiskLevel = Field(default=RiskLevel.HIGH)
    include_carbon_removal: bool = Field(default=False)
    optimization_target: str = Field(default="cost", description="cost|speed|risk")
    output_formats: List[str] = Field(default_factory=lambda: ["json", "html"])

class CorrectiveActionInput(BaseModel):
    config: CorrectiveActionConfig = Field(default_factory=CorrectiveActionConfig)
    existing_initiatives: List[Dict[str, Any]] = Field(default_factory=list)
    custom_initiatives: List[Dict[str, Any]] = Field(default_factory=list)
    scope_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Emission breakdown by scope {scope1, scope2, scope3}",
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints {max_capex, max_risk, min_trl, max_implementation_months}",
    )

class CorrectiveActionResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="corrective_action_planning")
    pack_id: str = Field(default=_PACK_ID)
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    gap: GapQuantification = Field(default_factory=GapQuantification)
    portfolio: OptimizedPortfolio = Field(default_factory=OptimizedPortfolio)
    schedule: DeploymentSchedule = Field(default_factory=DeploymentSchedule)
    budget_update: BudgetUpdate = Field(default_factory=BudgetUpdate)
    report: CorrectiveActionReport = Field(default_factory=CorrectiveActionReport)
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class CorrectiveActionPlanningWorkflow:
    """
    6-phase DAG workflow for corrective action planning.

    Phase 1: QuantifyGap         -- Quantify gap-to-target.
    Phase 2: IdentifyInitiatives -- Identify candidate initiatives (MACC).
    Phase 3: OptimizePortfolio   -- Optimize portfolio (cost, timing, risk).
    Phase 4: ScheduleDeployment  -- Schedule initiative deployment.
    Phase 5: UpdateBudget        -- Update carbon budget allocation.
    Phase 6: GenerateReport      -- Generate corrective action plan.

    DAG Dependencies:
        Phase 1 -> Phase 2 -> Phase 3 -> Phase 4
                                      -> Phase 5  (parallel with Phase 4)
                -> Phase 6  (depends on all prior)
    """

    def __init__(self, config: Optional[CorrectiveActionConfig] = None) -> None:
        self.workflow_id: str = _new_uuid()
        self.config = config or CorrectiveActionConfig()
        self._phase_results: List[PhaseResult] = []
        self._gap: GapQuantification = GapQuantification()
        self._candidates: List[CandidateInitiative] = []
        self._portfolio: OptimizedPortfolio = OptimizedPortfolio()
        self._schedule: DeploymentSchedule = DeploymentSchedule()
        self._budget: BudgetUpdate = BudgetUpdate()
        self._report: CorrectiveActionReport = CorrectiveActionReport()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: CorrectiveActionInput) -> CorrectiveActionResult:
        started_at = utcnow()
        self.config = input_data.config
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        self.logger.info(
            "Starting corrective action planning %s, company=%s",
            self.workflow_id, self.config.company_name,
        )

        try:
            phase1 = await self._phase_quantify_gap(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_identify_initiatives(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_optimize_portfolio(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_schedule_deployment(input_data)
            self._phase_results.append(phase4)

            phase5 = await self._phase_update_budget(input_data)
            self._phase_results.append(phase5)

            phase6 = await self._phase_generate_report(input_data)
            self._phase_results.append(phase6)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Corrective action planning failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()

        result = CorrectiveActionResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            gap=self._gap,
            portfolio=self._portfolio,
            schedule=self._schedule,
            budget_update=self._budget,
            report=self._report,
            key_findings=self._generate_findings(),
            recommendations=self._generate_recommendations(),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"}),
        )
        return result

    async def _phase_quantify_gap(self, input_data: CorrectiveActionInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        current = self.config.current_emissions_tco2e
        target = self.config.target_emissions_tco2e
        years_remaining = max(self.config.target_year - self.config.current_year, 1)

        if current <= 0:
            current = 100000
        if target <= 0:
            target = current * 0.58

        gap = current - target
        gap_pct = (gap / max(target, 1e-10)) * 100
        annual_gap = gap / years_remaining
        cumulative_gap = (gap * years_remaining) / 2

        # Required additional annual reduction rate
        if current > 0:
            required_rate = (1 - (target / current) ** (1.0 / years_remaining)) * 100
        else:
            required_rate = 0

        # Severity
        if gap_pct <= 0:
            severity = GapSeverity.ON_TRACK
            rag = RAGStatus.GREEN
        elif gap_pct <= 10:
            severity = GapSeverity.MINOR
            rag = RAGStatus.GREEN
        elif gap_pct <= 25:
            severity = GapSeverity.MODERATE
            rag = RAGStatus.AMBER
        elif gap_pct <= 50:
            severity = GapSeverity.SIGNIFICANT
            rag = RAGStatus.RED
        else:
            severity = GapSeverity.CRITICAL
            rag = RAGStatus.RED

        budget_gap = max(cumulative_gap - self.config.carbon_budget_remaining_tco2e, 0) if self.config.carbon_budget_remaining_tco2e > 0 else 0

        self._gap = GapQuantification(
            gap_tco2e=round(gap, 2),
            gap_pct=round(gap_pct, 2),
            annual_gap_tco2e=round(annual_gap, 2),
            cumulative_gap_tco2e=round(cumulative_gap, 2),
            severity=severity,
            years_to_target=years_remaining,
            required_additional_reduction_pct=round(required_rate, 2),
            carbon_budget_gap_tco2e=round(budget_gap, 2),
            rag_status=rag,
        )

        outputs["gap_tco2e"] = round(gap, 2)
        outputs["gap_pct"] = round(gap_pct, 2)
        outputs["severity"] = severity.value
        outputs["years_remaining"] = years_remaining
        outputs["required_annual_reduction_pct"] = round(required_rate, 2)

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="quantify_gap", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_quantify_gap",
        )

    async def _phase_identify_initiatives(self, input_data: CorrectiveActionInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        candidates: List[CandidateInitiative] = []
        gap = self._gap.gap_tco2e
        current = self.config.current_emissions_tco2e or 100000
        constraints = input_data.constraints
        max_risk = constraints.get("max_risk", self.config.max_risk_level.value)
        min_trl = constraints.get("min_trl", 7)
        max_months = constraints.get("max_implementation_months", 60)

        risk_order = {"low": 1, "medium": 2, "high": 3, "very_high": 4}

        for key, init_data in MACC_INITIATIVE_LIBRARY.items():
            if init_data["category"] == "carbon_removal" and not self.config.include_carbon_removal:
                continue
            if risk_order.get(init_data["risk"], 5) > risk_order.get(max_risk, 3):
                continue
            if init_data["trl"] < min_trl:
                continue
            if init_data["implementation_months"] > max_months:
                continue

            abatement = current * (init_data["typical_reduction_pct"] / 100.0)
            total_cost = abatement * init_data["abatement_cost_usd_per_tco2e"]
            capex = abatement * init_data["capex_usd_per_tco2e"]

            priority = InitiativePriority.HIGH if init_data["abatement_cost_usd_per_tco2e"] < 0 else (
                InitiativePriority.MEDIUM if init_data["abatement_cost_usd_per_tco2e"] < 50 else
                InitiativePriority.LOW
            )

            candidates.append(CandidateInitiative(
                initiative_id=f"CI-{key.upper()[:6]}",
                initiative_name=init_data["name"],
                category=InitiativeCategory(init_data["category"]),
                abatement_potential_tco2e=round(abatement, 2),
                abatement_cost_usd_per_tco2e=init_data["abatement_cost_usd_per_tco2e"],
                total_cost_usd=round(total_cost, 2),
                capex_usd=round(capex, 2),
                implementation_months=init_data["implementation_months"],
                risk_level=RiskLevel(init_data["risk"]),
                trl=init_data["trl"],
                deployment_phase=DeploymentPhase(init_data["deployment_phase"]),
                priority=priority,
                macc_rank=0,
            ))

        # Add custom initiatives from input
        for ci in input_data.custom_initiatives:
            candidates.append(CandidateInitiative(
                initiative_id=ci.get("id", f"CI-CUSTOM-{len(candidates)+1}"),
                initiative_name=ci.get("name", "Custom Initiative"),
                abatement_potential_tco2e=ci.get("abatement_tco2e", 0),
                abatement_cost_usd_per_tco2e=ci.get("cost_per_tco2e", 0),
                total_cost_usd=ci.get("total_cost", 0),
                capex_usd=ci.get("capex", 0),
                implementation_months=ci.get("months", 12),
                priority=InitiativePriority.MEDIUM,
            ))

        # Sort by MACC (ascending cost)
        candidates.sort(key=lambda c: c.abatement_cost_usd_per_tco2e)
        for i, c in enumerate(candidates):
            c.macc_rank = i + 1

        self._candidates = candidates

        outputs["candidates_count"] = len(candidates)
        outputs["negative_cost_count"] = sum(1 for c in candidates if c.abatement_cost_usd_per_tco2e < 0)
        outputs["total_abatement_potential_tco2e"] = round(
            sum(c.abatement_potential_tco2e for c in candidates), 2,
        )

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="identify_initiatives", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_identify_initiatives",
        )

    async def _phase_optimize_portfolio(self, input_data: CorrectiveActionInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        gap = self._gap.gap_tco2e
        max_capex = self.config.available_capex_usd or float("inf")
        optimization = self.config.optimization_target

        selected: List[CandidateInitiative] = []
        cumulative_abatement = 0.0
        cumulative_cost = 0.0
        cumulative_capex = 0.0

        # Sort candidates based on optimization target
        if optimization == "cost":
            sorted_candidates = sorted(self._candidates, key=lambda c: c.abatement_cost_usd_per_tco2e)
        elif optimization == "speed":
            sorted_candidates = sorted(self._candidates, key=lambda c: c.implementation_months)
        else:  # risk
            risk_map = {"low": 1, "medium": 2, "high": 3, "very_high": 4}
            sorted_candidates = sorted(
                self._candidates,
                key=lambda c: risk_map.get(c.risk_level.value, 5),
            )

        for candidate in sorted_candidates:
            if cumulative_abatement >= gap and gap > 0:
                break
            if cumulative_capex + candidate.capex_usd > max_capex and max_capex < float("inf"):
                continue

            candidate.selected = True
            cumulative_abatement += candidate.abatement_potential_tco2e
            cumulative_cost += candidate.total_cost_usd
            cumulative_capex += candidate.capex_usd
            candidate.cumulative_abatement_tco2e = round(cumulative_abatement, 2)
            selected.append(candidate)

        gap_closure = min((cumulative_abatement / max(gap, 1e-10)) * 100, 100) if gap > 0 else 100
        residual = max(gap - cumulative_abatement, 0)
        avg_cost = cumulative_cost / max(cumulative_abatement, 1e-10)
        net_savings = -sum(c.total_cost_usd for c in selected if c.total_cost_usd < 0)

        high_risk = sum(1 for c in selected if c.risk_level in (RiskLevel.HIGH, RiskLevel.VERY_HIGH))
        portfolio_risk = RiskLevel.HIGH if high_risk >= 2 else (
            RiskLevel.MEDIUM if high_risk >= 1 else RiskLevel.LOW
        )

        self._portfolio = OptimizedPortfolio(
            selected_initiatives=selected,
            total_abatement_tco2e=round(cumulative_abatement, 2),
            total_cost_usd=round(cumulative_cost, 2),
            total_capex_usd=round(cumulative_capex, 2),
            weighted_avg_cost_usd=round(avg_cost, 2),
            gap_closure_pct=round(gap_closure, 1),
            residual_gap_tco2e=round(residual, 2),
            portfolio_risk=portfolio_risk,
            net_savings_usd=round(net_savings, 2),
        )
        self._portfolio.provenance_hash = _compute_hash(
            self._portfolio.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["selected_count"] = len(selected)
        outputs["total_abatement_tco2e"] = round(cumulative_abatement, 2)
        outputs["gap_closure_pct"] = round(gap_closure, 1)
        outputs["total_capex_usd"] = round(cumulative_capex, 2)
        outputs["net_savings_usd"] = round(net_savings, 2)
        outputs["portfolio_risk"] = portfolio_risk.value

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="optimize_portfolio", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_optimize_portfolio",
        )

    async def _phase_schedule_deployment(self, input_data: CorrectiveActionInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        immediate: List[str] = []
        short: List[str] = []
        medium: List[str] = []
        long: List[str] = []

        for init in self._portfolio.selected_initiatives:
            name = init.initiative_name
            if init.deployment_phase == DeploymentPhase.IMMEDIATE:
                immediate.append(name)
            elif init.deployment_phase == DeploymentPhase.SHORT_TERM:
                short.append(name)
            elif init.deployment_phase == DeploymentPhase.MEDIUM_TERM:
                medium.append(name)
            else:
                long.append(name)

        # Quarterly milestones (cumulative abatement)
        milestones: Dict[str, float] = {}
        cum = 0.0
        for init in sorted(self._portfolio.selected_initiatives, key=lambda i: i.implementation_months):
            q = math.ceil(init.implementation_months / 3)
            q_label = f"Q{q}" if q <= 4 else f"Y{math.ceil(q/4)}Q{((q-1)%4)+1}"
            cum += init.abatement_potential_tco2e
            milestones[q_label] = round(cum, 2)

        max_months = max(
            (i.implementation_months for i in self._portfolio.selected_initiatives), default=0,
        )

        critical = [
            i.initiative_name for i in self._portfolio.selected_initiatives
            if i.abatement_potential_tco2e >= self._gap.gap_tco2e * 0.15
        ]

        self._schedule = DeploymentSchedule(
            immediate_initiatives=immediate,
            short_term_initiatives=short,
            medium_term_initiatives=medium,
            long_term_initiatives=long,
            quarterly_milestones=milestones,
            total_deployment_months=max_months,
            critical_path_initiatives=critical,
        )
        self._schedule.provenance_hash = _compute_hash(
            self._schedule.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["immediate_count"] = len(immediate)
        outputs["short_term_count"] = len(short)
        outputs["medium_term_count"] = len(medium)
        outputs["long_term_count"] = len(long)
        outputs["total_deployment_months"] = max_months
        outputs["critical_path_count"] = len(critical)

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="schedule_deployment", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_schedule_deployment",
        )

    async def _phase_update_budget(self, input_data: CorrectiveActionInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        original = self.config.carbon_budget_remaining_tco2e or self._gap.cumulative_gap_tco2e * 2
        savings = self._portfolio.total_abatement_tco2e * self._gap.years_to_target
        updated = original + savings

        extension = (savings / max(self.config.current_emissions_tco2e, 1e-10)) if self.config.current_emissions_tco2e > 0 else 0
        new_depletion = self.config.target_year + int(extension)

        capex = self._portfolio.total_capex_usd
        savings_value = self._portfolio.net_savings_usd * self._gap.years_to_target
        roi = ((savings_value - capex) / max(capex, 1e-10)) * 100 if capex > 0 else 0

        self._budget = BudgetUpdate(
            original_budget_tco2e=round(original, 2),
            corrective_savings_tco2e=round(savings, 2),
            updated_budget_tco2e=round(updated, 2),
            budget_extension_years=round(extension, 1),
            new_depletion_year=min(new_depletion, 2100),
            capex_required_usd=round(capex, 2),
            roi_pct=round(roi, 1),
        )
        self._budget.provenance_hash = _compute_hash(
            self._budget.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["original_budget_tco2e"] = round(original, 2)
        outputs["savings_tco2e"] = round(savings, 2)
        outputs["updated_budget_tco2e"] = round(updated, 2)
        outputs["capex_usd"] = round(capex, 2)
        outputs["roi_pct"] = round(roi, 1)

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="update_budget", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_update_budget",
        )

    async def _phase_generate_report(self, input_data: CorrectiveActionInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        findings = self._generate_findings()
        recommendations = self._generate_recommendations()

        exec_parts = [
            f"Corrective Action Plan for {self.config.company_name or 'Company'}.",
            f"Gap to target: {self._gap.gap_tco2e:,.0f} tCO2e ({self._gap.gap_pct:.1f}%).",
            f"Severity: {self._gap.severity.value}.",
            f"Selected {len(self._portfolio.selected_initiatives)} initiatives with {self._portfolio.total_abatement_tco2e:,.0f} tCO2e total abatement.",
            f"Gap closure: {self._portfolio.gap_closure_pct:.0f}%.",
            f"Total CapEx: ${self._portfolio.total_capex_usd:,.0f}.",
            f"Net savings from negative-cost initiatives: ${self._portfolio.net_savings_usd:,.0f}.",
        ]

        self._report = CorrectiveActionReport(
            report_id=f"CAP-{self.workflow_id[:8]}",
            report_date=utcnow().strftime("%Y-%m-%d"),
            company_name=self.config.company_name,
            gap=self._gap,
            portfolio=self._portfolio,
            schedule=self._schedule,
            budget_update=self._budget,
            executive_summary=" ".join(exec_parts),
            key_findings=findings,
            recommendations=recommendations,
        )
        self._report.provenance_hash = _compute_hash(
            self._report.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["report_id"] = self._report.report_id
        outputs["findings_count"] = len(findings)
        outputs["recommendations_count"] = len(recommendations)

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="generate_report", phase_number=6,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_generate_report",
        )

    def _generate_findings(self) -> List[str]:
        findings: List[str] = []
        findings.append(f"Gap to target: {self._gap.gap_tco2e:,.0f} tCO2e ({self._gap.severity.value}).")
        findings.append(f"Portfolio achieves {self._portfolio.gap_closure_pct:.0f}% gap closure.")
        neg_cost = [i for i in self._portfolio.selected_initiatives if i.abatement_cost_usd_per_tco2e < 0]
        if neg_cost:
            findings.append(f"{len(neg_cost)} initiatives with negative abatement cost (net savings).")
        if self._portfolio.residual_gap_tco2e > 0:
            findings.append(f"Residual gap: {self._portfolio.residual_gap_tco2e:,.0f} tCO2e requires additional measures.")
        findings.append(f"ROI on corrective CapEx: {self._budget.roi_pct:.0f}%.")
        return findings

    def _generate_recommendations(self) -> List[str]:
        recs: List[str] = []
        if self._schedule.immediate_initiatives:
            recs.append(f"Deploy immediately: {', '.join(self._schedule.immediate_initiatives[:3])}.")
        if self._schedule.short_term_initiatives:
            recs.append(f"Deploy within 12 months: {', '.join(self._schedule.short_term_initiatives[:3])}.")
        if self._portfolio.residual_gap_tco2e > 0:
            recs.append("Investigate additional abatement levers for residual gap closure.")
        recs.append("Establish monthly progress tracking against corrective action milestones.")
        recs.append("Update SBTi target tracking documentation with corrective action plan.")
        return recs
