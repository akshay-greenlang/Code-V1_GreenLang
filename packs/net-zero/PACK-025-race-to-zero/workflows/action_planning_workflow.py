# -*- coding: utf-8 -*-
"""
Action Planning Workflow
=============================

6-phase workflow for end-to-end climate action plan development within
PACK-025 Race to Zero Pack.  Covers sector pathway selection, marginal
abatement cost curve (MACC) analysis, partnership identification, action
prioritization, budget allocation, and plan documentation.

Phases:
    1. SectorPathwaySelection   -- Identify and select sector-specific pathways
    2. MACCAnalysis             -- Build marginal abatement cost curve for actions
    3. PartnershipIdentification-- Identify collaboration and partnership opportunities
    4. ActionPrioritization     -- Prioritize actions by impact, cost, and feasibility
    5. BudgetAllocation         -- Allocate budget across prioritized actions
    6. PlanDocumentation        -- Generate complete 10-section climate action plan

Regulatory references:
    - Race to Zero Interpretation Guide (June 2022 update)
    - HLEG "Integrity Matters" Report (November 2022)
    - IEA Net Zero by 2050 Roadmap (2021, updated 2023)
    - IPCC AR6 WG3 -- Mitigation Pathways (2022)
    - SBTi Corporate Net-Zero Standard V1.3 (2024)
    - Mission Possible Partnership (Hard-to-abate sectors, 2022)

Zero-hallucination: all abatement cost calculations, pathway benchmarks,
and budget allocations use deterministic formulas and reference tables.
No LLM calls in the numeric computation path.

Author: GreenLang Team
Version: 25.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "25.0.0"

ProgressCallback = Callable[[str, float, str], Coroutine[Any, Any, None]]

def _new_uuid() -> str:
    return uuid.uuid4().hex

def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()

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
    CANCELLED = "cancelled"

class ActionPlanPhase(str, Enum):
    SECTOR_PATHWAY_SELECTION = "sector_pathway_selection"
    MACC_ANALYSIS = "macc_analysis"
    PARTNERSHIP_IDENTIFICATION = "partnership_identification"
    ACTION_PRIORITIZATION = "action_prioritization"
    BUDGET_ALLOCATION = "budget_allocation"
    PLAN_DOCUMENTATION = "plan_documentation"

class ActionCategory(str, Enum):
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_ENERGY = "renewable_energy"
    ELECTRIFICATION = "electrification"
    FUEL_SWITCHING = "fuel_switching"
    PROCESS_OPTIMIZATION = "process_optimization"
    SUPPLY_CHAIN = "supply_chain"
    TRANSPORT = "transport"
    BUILDINGS = "buildings"
    WASTE_REDUCTION = "waste_reduction"
    CARBON_REMOVAL = "carbon_removal"
    BEHAVIORAL_CHANGE = "behavioral_change"
    POLICY_ADVOCACY = "policy_advocacy"

class FeasibilityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

class TimeHorizon(str, Enum):
    IMMEDIATE = "immediate"       # 0-1 year
    SHORT_TERM = "short_term"     # 1-3 years
    MEDIUM_TERM = "medium_term"   # 3-5 years
    LONG_TERM = "long_term"       # 5-10 years

# =============================================================================
# REFERENCE DATA
# =============================================================================

# Sector pathway benchmarks (IEA NZE / IPCC AR6 / TPI aligned)
SECTOR_PATHWAYS: Dict[str, Dict[str, Any]] = {
    "power_generation": {
        "name": "Power Generation",
        "source": "IEA NZE 2050",
        "milestones": {
            2025: "Coal phase-down begins, 60% RE share in new capacity",
            2030: "Unabated coal phase-out in advanced economies, 80% RE",
            2035: "All power sector emissions halved vs 2020",
            2040: "Net-zero power sector in advanced economies",
            2050: "Global net-zero power sector",
        },
        "intensity_target_2030": 0.138,  # tCO2/MWh
        "reduction_rate_annual": 7.0,
    },
    "steel": {
        "name": "Steel & Iron",
        "source": "MPP / IEA NZE",
        "milestones": {
            2025: "Near-zero steel pilots operational",
            2030: "30% reduction in steel emissions intensity",
            2035: "50% of primary production via DRI-H2/EAF",
            2040: "70% near-zero steel capacity",
            2050: "Net-zero steel production",
        },
        "intensity_target_2030": 1.4,  # tCO2/t steel
        "reduction_rate_annual": 4.5,
    },
    "cement": {
        "name": "Cement & Concrete",
        "source": "MPP / IEA NZE",
        "milestones": {
            2025: "Clinker ratio reduction to 0.65",
            2030: "40% reduction in cement emissions",
            2035: "CCS deployment at scale",
            2040: "50%+ of cement carbon captured",
            2050: "Net-zero cement production",
        },
        "intensity_target_2030": 0.43,  # tCO2/t cement
        "reduction_rate_annual": 3.5,
    },
    "transport_road": {
        "name": "Road Transport",
        "source": "IEA NZE",
        "milestones": {
            2025: "20% EV share in new car sales globally",
            2030: "60% EV share new sales; no new ICE cars from 2035",
            2035: "100% EV new car sales in advanced economies",
            2040: "Majority of fleet electrified",
            2050: "Net-zero road transport",
        },
        "intensity_target_2030": 0.08,  # tCO2/vehicle/yr
        "reduction_rate_annual": 5.0,
    },
    "buildings": {
        "name": "Buildings",
        "source": "IEA NZE",
        "milestones": {
            2025: "All new buildings zero-carbon-ready",
            2030: "50% of heating from heat pumps (new sales)",
            2035: "No new fossil fuel boilers sold",
            2040: "All buildings in advanced economies retrofitted",
            2050: "Net-zero buildings sector",
        },
        "intensity_target_2030": 0.012,  # tCO2/m2/yr
        "reduction_rate_annual": 4.0,
    },
    "financial_services": {
        "name": "Financial Services",
        "source": "GFANZ / PCAF",
        "milestones": {
            2025: "Financed emissions baseline set",
            2030: "Portfolio aligned to 1.5C pathway",
            2035: "50% reduction in financed emissions",
            2040: "Phase out financing of unabated fossil fuels",
            2050: "Net-zero financed emissions",
        },
        "intensity_target_2030": 0.0,
        "reduction_rate_annual": 5.0,
    },
    "general_services": {
        "name": "General / Service Sector",
        "source": "SBTi ACA",
        "milestones": {
            2025: "100% renewable electricity procurement",
            2030: "42% absolute reduction (1.5C ACA)",
            2035: "60% absolute reduction",
            2040: "80% absolute reduction",
            2050: "Net-zero with residual neutralization",
        },
        "intensity_target_2030": 0.0,
        "reduction_rate_annual": 4.2,
    },
}

# Standard abatement actions with typical costs and impact
STANDARD_ABATEMENT_ACTIONS: List[Dict[str, Any]] = [
    {
        "id": "ACT-01", "name": "LED lighting retrofit",
        "category": "energy_efficiency", "abatement_tco2e_per_unit": 2.5,
        "cost_usd_per_tco2e": -50.0, "feasibility": "high",
        "payback_years": 2, "time_horizon": "immediate",
    },
    {
        "id": "ACT-02", "name": "HVAC optimization",
        "category": "energy_efficiency", "abatement_tco2e_per_unit": 5.0,
        "cost_usd_per_tco2e": -30.0, "feasibility": "high",
        "payback_years": 3, "time_horizon": "short_term",
    },
    {
        "id": "ACT-03", "name": "On-site solar PV installation",
        "category": "renewable_energy", "abatement_tco2e_per_unit": 50.0,
        "cost_usd_per_tco2e": 20.0, "feasibility": "high",
        "payback_years": 7, "time_horizon": "short_term",
    },
    {
        "id": "ACT-04", "name": "Renewable electricity PPA",
        "category": "renewable_energy", "abatement_tco2e_per_unit": 200.0,
        "cost_usd_per_tco2e": 5.0, "feasibility": "high",
        "payback_years": 1, "time_horizon": "immediate",
    },
    {
        "id": "ACT-05", "name": "Fleet electrification",
        "category": "electrification", "abatement_tco2e_per_unit": 15.0,
        "cost_usd_per_tco2e": 80.0, "feasibility": "medium",
        "payback_years": 8, "time_horizon": "medium_term",
    },
    {
        "id": "ACT-06", "name": "Heat pump installation",
        "category": "electrification", "abatement_tco2e_per_unit": 8.0,
        "cost_usd_per_tco2e": 60.0, "feasibility": "medium",
        "payback_years": 6, "time_horizon": "short_term",
    },
    {
        "id": "ACT-07", "name": "Supplier engagement program",
        "category": "supply_chain", "abatement_tco2e_per_unit": 100.0,
        "cost_usd_per_tco2e": 15.0, "feasibility": "medium",
        "payback_years": 3, "time_horizon": "medium_term",
    },
    {
        "id": "ACT-08", "name": "Waste reduction and recycling",
        "category": "waste_reduction", "abatement_tco2e_per_unit": 10.0,
        "cost_usd_per_tco2e": -20.0, "feasibility": "high",
        "payback_years": 1, "time_horizon": "immediate",
    },
    {
        "id": "ACT-09", "name": "Business travel reduction policy",
        "category": "behavioral_change", "abatement_tco2e_per_unit": 20.0,
        "cost_usd_per_tco2e": -100.0, "feasibility": "high",
        "payback_years": 0, "time_horizon": "immediate",
    },
    {
        "id": "ACT-10", "name": "Employee commuting program",
        "category": "transport", "abatement_tco2e_per_unit": 5.0,
        "cost_usd_per_tco2e": 10.0, "feasibility": "high",
        "payback_years": 2, "time_horizon": "short_term",
    },
    {
        "id": "ACT-11", "name": "Process efficiency improvements",
        "category": "process_optimization", "abatement_tco2e_per_unit": 30.0,
        "cost_usd_per_tco2e": 25.0, "feasibility": "medium",
        "payback_years": 4, "time_horizon": "medium_term",
    },
    {
        "id": "ACT-12", "name": "Carbon removal credits (nature-based)",
        "category": "carbon_removal", "abatement_tco2e_per_unit": 50.0,
        "cost_usd_per_tco2e": 25.0, "feasibility": "high",
        "payback_years": 0, "time_horizon": "short_term",
    },
]

# Action plan 10-section structure per Interpretation Guide
ACTION_PLAN_SECTIONS: List[Dict[str, str]] = [
    {"id": "S01", "title": "Emissions Profile", "description": "Current GHG inventory by scope"},
    {"id": "S02", "title": "Baseline & Target", "description": "Base year, interim and long-term targets"},
    {"id": "S03", "title": "Sector Pathway Alignment", "description": "Sector-specific pathway and benchmarks"},
    {"id": "S04", "title": "Decarbonization Levers", "description": "Identified abatement actions with tCO2e impact"},
    {"id": "S05", "title": "Marginal Abatement Cost Curve", "description": "MACC analysis with cost-effectiveness ranking"},
    {"id": "S06", "title": "Implementation Timeline", "description": "Phased implementation with milestones"},
    {"id": "S07", "title": "Resource Allocation", "description": "Budget, personnel, and technology resources"},
    {"id": "S08", "title": "Partnership & Collaboration", "description": "Partnership framework and joint actions"},
    {"id": "S09", "title": "Monitoring & Reporting", "description": "KPIs, reporting cadence, verification"},
    {"id": "S10", "title": "Risk Management", "description": "Implementation risks and mitigation strategies"},
]

# Phase dependencies DAG
PHASE_DEPENDENCIES: Dict[ActionPlanPhase, List[ActionPlanPhase]] = {
    ActionPlanPhase.SECTOR_PATHWAY_SELECTION: [],
    ActionPlanPhase.MACC_ANALYSIS: [ActionPlanPhase.SECTOR_PATHWAY_SELECTION],
    ActionPlanPhase.PARTNERSHIP_IDENTIFICATION: [ActionPlanPhase.SECTOR_PATHWAY_SELECTION],
    ActionPlanPhase.ACTION_PRIORITIZATION: [
        ActionPlanPhase.MACC_ANALYSIS,
        ActionPlanPhase.PARTNERSHIP_IDENTIFICATION,
    ],
    ActionPlanPhase.BUDGET_ALLOCATION: [ActionPlanPhase.ACTION_PRIORITIZATION],
    ActionPlanPhase.PLAN_DOCUMENTATION: [ActionPlanPhase.BUDGET_ALLOCATION],
}

PHASE_EXECUTION_ORDER: List[ActionPlanPhase] = [
    ActionPlanPhase.SECTOR_PATHWAY_SELECTION,
    ActionPlanPhase.MACC_ANALYSIS,
    ActionPlanPhase.PARTNERSHIP_IDENTIFICATION,
    ActionPlanPhase.ACTION_PRIORITIZATION,
    ActionPlanPhase.BUDGET_ALLOCATION,
    ActionPlanPhase.PLAN_DOCUMENTATION,
]

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    phase: ActionPlanPhase = Field(...)
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class AbatementAction(BaseModel):
    action_id: str = Field(default="")
    name: str = Field(default="")
    category: ActionCategory = Field(default=ActionCategory.ENERGY_EFFICIENCY)
    abatement_tco2e: float = Field(default=0.0, ge=0.0)
    cost_usd: float = Field(default=0.0)
    cost_per_tco2e: float = Field(default=0.0)
    feasibility: FeasibilityLevel = Field(default=FeasibilityLevel.MEDIUM)
    payback_years: float = Field(default=0.0, ge=0.0)
    time_horizon: TimeHorizon = Field(default=TimeHorizon.SHORT_TERM)
    priority_rank: int = Field(default=0, ge=0)
    budget_allocated_usd: float = Field(default=0.0, ge=0.0)
    selected: bool = Field(default=False)

class PartnerOpportunity(BaseModel):
    partner_name: str = Field(default="")
    partner_type: str = Field(default="")
    collaboration_area: str = Field(default="")
    potential_abatement_tco2e: float = Field(default=0.0, ge=0.0)
    engagement_level: str = Field(default="exploratory")

class ActionPlanDocument(BaseModel):
    plan_id: str = Field(default="")
    org_name: str = Field(default="")
    version: str = Field(default="1.0")
    sections_completed: int = Field(default=0)
    sections_total: int = Field(default=10)
    total_abatement_tco2e: float = Field(default=0.0)
    total_budget_usd: float = Field(default=0.0)
    actions_count: int = Field(default=0)
    timeline_years: int = Field(default=5)
    sector_pathway: str = Field(default="")
    meets_r2z_requirements: bool = Field(default=False)

class ActionPlanningConfig(BaseModel):
    pack_id: str = Field(default="PACK-025")
    pack_version: str = Field(default="1.0.0")
    org_name: str = Field(default="")
    sector: str = Field(default="general_services")
    actor_type: str = Field(default="corporate")
    reporting_year: int = Field(default=2025, ge=2015, le=2050)
    base_year: int = Field(default=2019, ge=2015, le=2050)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    target_reduction_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    total_budget_usd: float = Field(default=0.0, ge=0.0)
    planning_horizon_years: int = Field(default=5, ge=1, le=30)
    partner_initiative: str = Field(default="sbti")
    enable_provenance: bool = Field(default=True)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class ActionPlanningResult(BaseModel):
    execution_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-025")
    workflow_name: str = Field(default="action_planning")
    org_name: str = Field(default="")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    total_duration_ms: float = Field(default=0.0)
    phases_completed: List[str] = Field(default_factory=list)
    phases_skipped: List[str] = Field(default_factory=list)
    phase_results: Dict[str, PhaseResult] = Field(default_factory=dict)
    sector_pathway: str = Field(default="")
    abatement_actions: List[AbatementAction] = Field(default_factory=list)
    partnerships: List[PartnerOpportunity] = Field(default_factory=list)
    plan_document: Optional[ActionPlanDocument] = Field(None)
    total_abatement_tco2e: float = Field(default=0.0)
    total_cost_usd: float = Field(default=0.0)
    cost_effectiveness: float = Field(default=0.0)
    total_records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class ActionPlanningWorkflow:
    """
    6-phase action planning workflow for PACK-025 Race to Zero Pack.

    Develops a comprehensive climate action plan meeting Race to Zero
    publication requirements through sector pathway selection, MACC
    analysis, partnership identification, action prioritization,
    budget allocation, and 10-section plan documentation.

    Engines used:
        - sector_pathway_engine (pathway selection)
        - action_plan_engine (plan generation and validation)
        - partnership_scoring_engine (partnership identification)

    Attributes:
        config: Workflow configuration.
    """

    def __init__(
        self,
        config: Optional[ActionPlanningConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.config = config or ActionPlanningConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._results: Dict[str, ActionPlanningResult] = {}
        self._cancelled: Set[str] = set()
        self._progress_callback = progress_callback

    async def execute(
        self, input_data: Optional[Dict[str, Any]] = None,
    ) -> ActionPlanningResult:
        """Execute the 6-phase action planning workflow."""
        input_data = input_data or {}
        result = ActionPlanningResult(
            org_name=self.config.org_name,
            status=WorkflowStatus.RUNNING,
            started_at=utcnow(),
        )
        self._results[result.execution_id] = result
        start_time = time.monotonic()
        phases = PHASE_EXECUTION_ORDER
        total_phases = len(phases)

        self.logger.info(
            "Starting action planning: execution_id=%s, org=%s, sector=%s",
            result.execution_id, self.config.org_name, self.config.sector,
        )

        shared_context: Dict[str, Any] = dict(input_data)
        shared_context["org_name"] = self.config.org_name
        shared_context["sector"] = self.config.sector
        shared_context["reporting_year"] = self.config.reporting_year
        shared_context["total_emissions"] = (
            self.config.scope1_tco2e + self.config.scope2_tco2e + self.config.scope3_tco2e
        )
        shared_context["target_reduction_pct"] = self.config.target_reduction_pct
        shared_context["total_budget_usd"] = self.config.total_budget_usd

        try:
            for phase_idx, phase in enumerate(phases):
                if result.execution_id in self._cancelled:
                    result.status = WorkflowStatus.CANCELLED
                    break

                if not self._dependencies_met(phase, result):
                    pr = PhaseResult(phase=phase, status=PhaseStatus.FAILED, errors=["Dependencies not met"])
                    result.phase_results[phase.value] = pr
                    result.status = WorkflowStatus.FAILED
                    result.errors.append(f"Phase '{phase.value}' dependencies not met")
                    break

                progress_pct = (phase_idx / total_phases) * 100.0
                if self._progress_callback:
                    await self._progress_callback(phase.value, progress_pct, f"Executing {phase.value}")

                pr = await self._execute_phase(phase, shared_context)
                result.phase_results[phase.value] = pr

                if pr.status == PhaseStatus.FAILED:
                    result.status = WorkflowStatus.PARTIAL
                    result.errors.append(f"Phase '{phase.value}' failed")

                result.phases_completed.append(phase.value)
                result.total_records_processed += pr.records_processed
                shared_context[phase.value] = pr.outputs

            if result.status == WorkflowStatus.RUNNING:
                result.status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Action planning failed: %s", exc, exc_info=True)
            result.status = WorkflowStatus.FAILED
            result.errors.append(str(exc))

        finally:
            result.completed_at = utcnow()
            result.total_duration_ms = (time.monotonic() - start_time) * 1000
            result.sector_pathway = shared_context.get(
                "sector_pathway_selection", {}
            ).get("selected_pathway", "")
            result.abatement_actions = self._extract_actions(shared_context)
            result.partnerships = self._extract_partnerships(shared_context)
            result.plan_document = self._extract_plan(shared_context)
            result.total_abatement_tco2e = shared_context.get(
                "action_prioritization", {}
            ).get("total_selected_abatement_tco2e", 0.0)
            result.total_cost_usd = shared_context.get(
                "budget_allocation", {}
            ).get("total_allocated_usd", 0.0)
            result.cost_effectiveness = round(
                result.total_cost_usd / max(result.total_abatement_tco2e, 1.0), 2
            )
            result.quality_score = self._compute_quality(result)
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    result.model_dump_json(exclude={"provenance_hash"})
                )

        self.logger.info(
            "Action planning %s: status=%s, abatement=%.0f tCO2e, cost=$%.0f",
            result.execution_id, result.status.value,
            result.total_abatement_tco2e, result.total_cost_usd,
        )
        return result

    def cancel(self, execution_id: str) -> Dict[str, Any]:
        self._cancelled.add(execution_id)
        return {"cancelled": True, "execution_id": execution_id}

    # -------------------------------------------------------------------------
    # Phase Execution
    # -------------------------------------------------------------------------

    async def _execute_phase(self, phase: ActionPlanPhase, context: Dict[str, Any]) -> PhaseResult:
        started = utcnow()
        start_time = time.monotonic()
        handler = self._get_phase_handler(phase)
        try:
            outputs, warnings, errors, records = await handler(context)
            status = PhaseStatus.FAILED if errors else PhaseStatus.COMPLETED
        except Exception as exc:
            outputs, warnings, errors, records = {}, [], [str(exc)], 0
            status = PhaseStatus.FAILED
        elapsed_ms = (time.monotonic() - start_time) * 1000
        return PhaseResult(
            phase=phase, status=status, started_at=started, completed_at=utcnow(),
            duration_ms=round(elapsed_ms, 2), records_processed=records,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs) if self.config.enable_provenance else "",
        )

    def _get_phase_handler(self, phase: ActionPlanPhase):
        return {
            ActionPlanPhase.SECTOR_PATHWAY_SELECTION: self._handle_sector_pathway,
            ActionPlanPhase.MACC_ANALYSIS: self._handle_macc_analysis,
            ActionPlanPhase.PARTNERSHIP_IDENTIFICATION: self._handle_partnership_id,
            ActionPlanPhase.ACTION_PRIORITIZATION: self._handle_action_prioritization,
            ActionPlanPhase.BUDGET_ALLOCATION: self._handle_budget_allocation,
            ActionPlanPhase.PLAN_DOCUMENTATION: self._handle_plan_documentation,
        }[phase]

    # -------------------------------------------------------------------------
    # Phase 1: Sector Pathway Selection
    # -------------------------------------------------------------------------

    async def _handle_sector_pathway(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        sector = self.config.sector
        pathway = SECTOR_PATHWAYS.get(sector)

        if not pathway:
            # Default to general services
            pathway = SECTOR_PATHWAYS["general_services"]
            warnings.append(
                f"Sector '{sector}' not found in pathway database. "
                "Using 'general_services' as default."
            )
            sector = "general_services"

        total_emissions = ctx.get("total_emissions", 0)
        target_2030 = total_emissions * (1 - self.config.target_reduction_pct / 100.0)

        # Benchmark comparison
        entity_reduction_rate = self.config.target_reduction_pct / max(2030 - self.config.base_year, 1)
        pathway_rate = pathway.get("reduction_rate_annual", 4.2)
        gap_to_benchmark = round(pathway_rate - entity_reduction_rate, 2)

        outputs["selected_pathway"] = sector
        outputs["pathway_name"] = pathway["name"]
        outputs["pathway_source"] = pathway["source"]
        outputs["milestones"] = pathway["milestones"]
        outputs["pathway_reduction_rate_annual"] = pathway_rate
        outputs["entity_reduction_rate_annual"] = round(entity_reduction_rate, 2)
        outputs["gap_to_benchmark_pct"] = gap_to_benchmark
        outputs["aligned_to_pathway"] = gap_to_benchmark <= 0
        outputs["target_2030_tco2e"] = round(target_2030, 2)
        outputs["total_baseline_tco2e"] = round(total_emissions, 2)

        if gap_to_benchmark > 0:
            warnings.append(
                f"Entity reduction rate ({entity_reduction_rate:.1f}%/yr) is below "
                f"sector benchmark ({pathway_rate:.1f}%/yr). Gap: {gap_to_benchmark:.1f}%."
            )

        records = 1
        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Phase 2: MACC Analysis
    # -------------------------------------------------------------------------

    async def _handle_macc_analysis(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        total_emissions = ctx.get("total_emissions", 0)
        target_reduction = total_emissions * (self.config.target_reduction_pct / 100.0)

        # Scale standard actions to entity size
        macc_actions: List[Dict[str, Any]] = []
        scale_factor = max(total_emissions / 10000.0, 0.1)  # Normalize to 10k tCO2e base

        for act in STANDARD_ABATEMENT_ACTIONS:
            scaled_abatement = act["abatement_tco2e_per_unit"] * scale_factor
            cost = scaled_abatement * act["cost_usd_per_tco2e"]

            macc_action = {
                "action_id": act["id"],
                "name": act["name"],
                "category": act["category"],
                "abatement_tco2e": round(scaled_abatement, 2),
                "cost_usd": round(cost, 2),
                "cost_per_tco2e": act["cost_usd_per_tco2e"],
                "feasibility": act["feasibility"],
                "payback_years": act["payback_years"],
                "time_horizon": act["time_horizon"],
                "is_negative_cost": act["cost_usd_per_tco2e"] < 0,
            }
            macc_actions.append(macc_action)
            records += 1

        # Sort by cost per tCO2e (cheapest/most profitable first) for MACC
        macc_actions.sort(key=lambda a: a["cost_per_tco2e"])

        # Calculate cumulative abatement
        cumulative = 0.0
        for act in macc_actions:
            cumulative += act["abatement_tco2e"]
            act["cumulative_abatement_tco2e"] = round(cumulative, 2)

        total_abatement = sum(a["abatement_tco2e"] for a in macc_actions)
        total_cost = sum(a["cost_usd"] for a in macc_actions)
        negative_cost_actions = sum(1 for a in macc_actions if a["is_negative_cost"])
        negative_cost_abatement = sum(
            a["abatement_tco2e"] for a in macc_actions if a["is_negative_cost"]
        )

        outputs["macc_actions"] = macc_actions
        outputs["actions_count"] = len(macc_actions)
        outputs["total_abatement_potential_tco2e"] = round(total_abatement, 2)
        outputs["total_cost_usd"] = round(total_cost, 2)
        outputs["avg_cost_per_tco2e"] = round(total_cost / max(total_abatement, 1), 2)
        outputs["negative_cost_actions"] = negative_cost_actions
        outputs["negative_cost_abatement_tco2e"] = round(negative_cost_abatement, 2)
        outputs["target_reduction_tco2e"] = round(target_reduction, 2)
        outputs["abatement_gap_tco2e"] = round(max(target_reduction - total_abatement, 0), 2)
        outputs["target_achievable"] = total_abatement >= target_reduction

        if total_abatement < target_reduction:
            warnings.append(
                f"Total abatement potential ({total_abatement:.0f} tCO2e) is below "
                f"target reduction ({target_reduction:.0f} tCO2e). "
                f"Gap: {target_reduction - total_abatement:.0f} tCO2e."
            )

        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Phase 3: Partnership Identification
    # -------------------------------------------------------------------------

    async def _handle_partnership_id(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        # Standard partnership opportunities based on actor type and sector
        partnerships: List[Dict[str, Any]] = [
            {
                "partner_name": "Science Based Targets initiative",
                "partner_type": "standard_setter",
                "collaboration_area": "Target validation and methodology support",
                "potential_abatement_tco2e": 0,
                "engagement_level": "active",
            },
            {
                "partner_name": "RE100",
                "partner_type": "initiative",
                "collaboration_area": "100% renewable electricity procurement",
                "potential_abatement_tco2e": self.config.scope2_tco2e * 0.5,
                "engagement_level": "exploratory",
            },
            {
                "partner_name": "CDP Supply Chain Program",
                "partner_type": "disclosure",
                "collaboration_area": "Supplier engagement and Scope 3 reduction",
                "potential_abatement_tco2e": self.config.scope3_tco2e * 0.1,
                "engagement_level": "exploratory",
            },
            {
                "partner_name": "Industry Peers Consortium",
                "partner_type": "industry_group",
                "collaboration_area": "Joint R&D and knowledge sharing on sector decarbonization",
                "potential_abatement_tco2e": 0,
                "engagement_level": "exploratory",
            },
            {
                "partner_name": "Local Government Climate Alliance",
                "partner_type": "public_sector",
                "collaboration_area": "Grid decarbonization and infrastructure support",
                "potential_abatement_tco2e": self.config.scope2_tco2e * 0.2,
                "engagement_level": "exploratory",
            },
        ]

        for p in partnerships:
            p["potential_abatement_tco2e"] = round(p["potential_abatement_tco2e"], 2)
            records += 1

        total_partnership_abatement = sum(p["potential_abatement_tco2e"] for p in partnerships)

        outputs["partnerships"] = partnerships
        outputs["partnerships_count"] = len(partnerships)
        outputs["total_partnership_abatement_tco2e"] = round(total_partnership_abatement, 2)
        outputs["active_partnerships"] = sum(1 for p in partnerships if p["engagement_level"] == "active")

        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Phase 4: Action Prioritization
    # -------------------------------------------------------------------------

    async def _handle_action_prioritization(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        macc_data = ctx.get("macc_analysis", {})
        macc_actions = macc_data.get("macc_actions", [])
        target_reduction = macc_data.get("target_reduction_tco2e", 0)

        # Priority scoring: lower cost + higher feasibility + shorter payback = higher priority
        feasibility_scores = {"high": 3, "medium": 2, "low": 1, "very_low": 0}
        horizon_scores = {"immediate": 4, "short_term": 3, "medium_term": 2, "long_term": 1}

        for act in macc_actions:
            # Composite priority score (higher = better)
            cost_score = max(0, 100 - act["cost_per_tco2e"])  # Negative cost = bonus
            feasibility_score = feasibility_scores.get(act["feasibility"], 1) * 25
            horizon_score = horizon_scores.get(act["time_horizon"], 1) * 15
            payback_score = max(0, 50 - act["payback_years"] * 5)

            act["priority_score"] = round(cost_score + feasibility_score + horizon_score + payback_score, 1)

        # Sort by priority score (highest first)
        macc_actions.sort(key=lambda a: -a["priority_score"])

        # Select actions to meet target
        selected_abatement = 0.0
        selected_count = 0
        for i, act in enumerate(macc_actions):
            act["priority_rank"] = i + 1
            if selected_abatement < target_reduction:
                act["selected"] = True
                selected_abatement += act["abatement_tco2e"]
                selected_count += 1
            else:
                act["selected"] = False
            records += 1

        outputs["prioritized_actions"] = macc_actions
        outputs["total_actions"] = len(macc_actions)
        outputs["selected_actions"] = selected_count
        outputs["total_selected_abatement_tco2e"] = round(selected_abatement, 2)
        outputs["target_reduction_tco2e"] = round(target_reduction, 2)
        outputs["target_met"] = selected_abatement >= target_reduction
        outputs["coverage_pct"] = round(
            (selected_abatement / max(target_reduction, 1)) * 100.0, 1
        )

        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Phase 5: Budget Allocation
    # -------------------------------------------------------------------------

    async def _handle_budget_allocation(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        prioritized = ctx.get("action_prioritization", {}).get("prioritized_actions", [])
        total_budget = self.config.total_budget_usd
        remaining_budget = total_budget

        allocations: List[Dict[str, Any]] = []
        total_allocated = 0.0
        total_abatement = 0.0

        for act in prioritized:
            if not act.get("selected", False):
                continue

            cost = abs(act["cost_usd"])
            # Negative cost actions generate savings, allocate minimal budget
            if act["cost_usd"] < 0:
                allocated = min(cost * 0.1, remaining_budget) if remaining_budget > 0 else 0
            else:
                allocated = min(cost, remaining_budget) if remaining_budget > 0 else 0

            allocation = {
                "action_id": act["action_id"],
                "name": act["name"],
                "budget_allocated_usd": round(allocated, 2),
                "cost_usd": round(act["cost_usd"], 2),
                "abatement_tco2e": act["abatement_tco2e"],
                "funding_gap_usd": round(max(cost - allocated, 0), 2) if act["cost_usd"] > 0 else 0,
                "time_horizon": act["time_horizon"],
            }
            allocations.append(allocation)
            total_allocated += allocated
            total_abatement += act["abatement_tco2e"]
            remaining_budget -= allocated
            records += 1

        # Category breakdown
        category_allocation: Dict[str, float] = {}
        for alloc in allocations:
            act_ref = next((a for a in prioritized if a["action_id"] == alloc["action_id"]), {})
            cat = act_ref.get("category", "other")
            category_allocation[cat] = category_allocation.get(cat, 0) + alloc["budget_allocated_usd"]

        outputs["allocations"] = allocations
        outputs["total_allocated_usd"] = round(total_allocated, 2)
        outputs["total_budget_usd"] = total_budget
        outputs["remaining_budget_usd"] = round(remaining_budget, 2)
        outputs["budget_utilization_pct"] = round(
            (total_allocated / max(total_budget, 1)) * 100.0, 1
        )
        outputs["total_abatement_funded_tco2e"] = round(total_abatement, 2)
        outputs["cost_per_tco2e_avg"] = round(
            total_allocated / max(total_abatement, 1), 2
        )
        outputs["category_allocation"] = category_allocation
        outputs["funding_gaps_count"] = sum(1 for a in allocations if a.get("funding_gap_usd", 0) > 0)

        if remaining_budget < 0:
            warnings.append("Budget exceeded allocated amount")

        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Phase 6: Plan Documentation
    # -------------------------------------------------------------------------

    async def _handle_plan_documentation(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        pathway = ctx.get("sector_pathway_selection", {})
        macc = ctx.get("macc_analysis", {})
        partners = ctx.get("partnership_identification", {})
        prioritized = ctx.get("action_prioritization", {})
        budget = ctx.get("budget_allocation", {})

        # Validate all 10 sections can be populated
        sections_status: Dict[str, bool] = {}
        for section in ACTION_PLAN_SECTIONS:
            sid = section["id"]
            # Check if we have sufficient data for each section
            if sid == "S01":
                sections_status[sid] = ctx.get("total_emissions", 0) > 0
            elif sid == "S02":
                sections_status[sid] = bool(pathway)
            elif sid == "S03":
                sections_status[sid] = bool(pathway.get("selected_pathway"))
            elif sid == "S04":
                sections_status[sid] = len(macc.get("macc_actions", [])) > 0
            elif sid == "S05":
                sections_status[sid] = len(macc.get("macc_actions", [])) > 0
            elif sid == "S06":
                sections_status[sid] = len(prioritized.get("prioritized_actions", [])) > 0
            elif sid == "S07":
                sections_status[sid] = bool(budget.get("allocations"))
            elif sid == "S08":
                sections_status[sid] = len(partners.get("partnerships", [])) > 0
            elif sid == "S09":
                sections_status[sid] = True  # Always populated
            elif sid == "S10":
                sections_status[sid] = True  # Always populated
            else:
                sections_status[sid] = True

        sections_completed = sum(1 for v in sections_status.values() if v)
        total_sections = len(ACTION_PLAN_SECTIONS)

        plan_id = f"AP-{self.config.reporting_year}-{_new_uuid()[:8].upper()}"

        meets_r2z = (
            sections_completed == total_sections
            and prioritized.get("target_met", False)
        )

        outputs["plan_id"] = plan_id
        outputs["org_name"] = self.config.org_name
        outputs["version"] = "1.0"
        outputs["sections"] = [
            {**s, "completed": sections_status.get(s["id"], False)}
            for s in ACTION_PLAN_SECTIONS
        ]
        outputs["sections_completed"] = sections_completed
        outputs["sections_total"] = total_sections
        outputs["completeness_pct"] = round(
            (sections_completed / total_sections) * 100.0, 1
        )
        outputs["total_abatement_tco2e"] = prioritized.get("total_selected_abatement_tco2e", 0)
        outputs["total_budget_usd"] = budget.get("total_allocated_usd", 0)
        outputs["actions_count"] = prioritized.get("selected_actions", 0)
        outputs["timeline_years"] = self.config.planning_horizon_years
        outputs["sector_pathway"] = pathway.get("selected_pathway", "")
        outputs["meets_r2z_requirements"] = meets_r2z

        if not meets_r2z:
            incomplete = [
                s["title"] for s in ACTION_PLAN_SECTIONS
                if not sections_status.get(s["id"], False)
            ]
            if incomplete:
                warnings.append(f"Incomplete sections: {', '.join(incomplete)}")

        records = 1
        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _dependencies_met(self, phase: ActionPlanPhase, result: ActionPlanningResult) -> bool:
        deps = PHASE_DEPENDENCIES.get(phase, [])
        for dep in deps:
            dep_result = result.phase_results.get(dep.value)
            if not dep_result or dep_result.status not in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED):
                return False
        return True

    def _compute_quality(self, result: ActionPlanningResult) -> float:
        total = len(PHASE_EXECUTION_ORDER)
        completed = len(result.phases_completed)
        return round((completed / max(total, 1)) * 100.0, 1)

    def _extract_actions(self, ctx: Dict[str, Any]) -> List[AbatementAction]:
        data = ctx.get("action_prioritization", {}).get("prioritized_actions", [])
        return [
            AbatementAction(
                action_id=a.get("action_id", ""),
                name=a.get("name", ""),
                category=ActionCategory(a.get("category", "energy_efficiency")),
                abatement_tco2e=a.get("abatement_tco2e", 0.0),
                cost_usd=a.get("cost_usd", 0.0),
                cost_per_tco2e=a.get("cost_per_tco2e", 0.0),
                feasibility=FeasibilityLevel(a.get("feasibility", "medium")),
                payback_years=a.get("payback_years", 0),
                time_horizon=TimeHorizon(a.get("time_horizon", "short_term")),
                priority_rank=a.get("priority_rank", 0),
                selected=a.get("selected", False),
            )
            for a in data
        ]

    def _extract_partnerships(self, ctx: Dict[str, Any]) -> List[PartnerOpportunity]:
        data = ctx.get("partnership_identification", {}).get("partnerships", [])
        return [
            PartnerOpportunity(
                partner_name=p.get("partner_name", ""),
                partner_type=p.get("partner_type", ""),
                collaboration_area=p.get("collaboration_area", ""),
                potential_abatement_tco2e=p.get("potential_abatement_tco2e", 0.0),
                engagement_level=p.get("engagement_level", "exploratory"),
            )
            for p in data
        ]

    def _extract_plan(self, ctx: Dict[str, Any]) -> Optional[ActionPlanDocument]:
        data = ctx.get("plan_documentation", {})
        if not data:
            return None
        return ActionPlanDocument(
            plan_id=data.get("plan_id", ""),
            org_name=data.get("org_name", ""),
            version=data.get("version", "1.0"),
            sections_completed=data.get("sections_completed", 0),
            sections_total=data.get("sections_total", 10),
            total_abatement_tco2e=data.get("total_abatement_tco2e", 0.0),
            total_budget_usd=data.get("total_budget_usd", 0.0),
            actions_count=data.get("actions_count", 0),
            timeline_years=data.get("timeline_years", 5),
            sector_pathway=data.get("sector_pathway", ""),
            meets_r2z_requirements=data.get("meets_r2z_requirements", False),
        )
