# -*- coding: utf-8 -*-
"""
Transition Plan Workflow
==============================

6-phase workflow for climate transition plan development per ESRS E1-1.
Implements baseline assessment, lever identification, action planning,
gap analysis, scenario validation, and report generation.

Phases:
    1. BaselineAssessment     -- Establish current emissions baseline
    2. LeverIdentification    -- Identify decarbonization levers
    3. ActionPlanning         -- Define actions, timelines, and investments
    4. GapAnalysis            -- Assess gap between targets and planned actions
    5. ScenarioValidation     -- Validate against 1.5C / 2C scenarios
    6. ReportGeneration       -- Produce E1-1 disclosure data

Author: GreenLang Team
Version: 16.0.0
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


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class WorkflowPhase(str, Enum):
    """Phases of the transition plan workflow."""
    BASELINE_ASSESSMENT = "baseline_assessment"
    LEVER_IDENTIFICATION = "lever_identification"
    ACTION_PLANNING = "action_planning"
    GAP_ANALYSIS = "gap_analysis"
    SCENARIO_VALIDATION = "scenario_validation"
    REPORT_GENERATION = "report_generation"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class LeverType(str, Enum):
    """Decarbonization lever types."""
    ENERGY_EFFICIENCY = "energy_efficiency"
    FUEL_SWITCHING = "fuel_switching"
    ELECTRIFICATION = "electrification"
    RENEWABLE_PROCUREMENT = "renewable_procurement"
    PROCESS_CHANGE = "process_change"
    SUPPLY_CHAIN = "supply_chain"
    CARBON_CAPTURE = "carbon_capture"
    PRODUCT_REDESIGN = "product_redesign"
    CIRCULAR_ECONOMY = "circular_economy"
    BEHAVIORAL_CHANGE = "behavioral_change"


class ScenarioType(str, Enum):
    """Climate scenario types."""
    IEA_NZE_2050 = "iea_nze_2050"
    IEA_SDS = "iea_sds"
    IEA_APS = "iea_aps"
    IPCC_SSP1_1_9 = "ipcc_ssp1_1.9"
    IPCC_SSP1_2_6 = "ipcc_ssp1_2.6"
    SBTI_1_5C = "sbti_1.5c"
    SBTI_WB2C = "sbti_wb2c"
    CUSTOM = "custom"


class ActionStatus(str, Enum):
    """Transition action status."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class DecarbonizationLever(BaseModel):
    """A decarbonization lever for the transition plan."""
    lever_id: str = Field(default_factory=lambda: f"lv-{_new_uuid()[:8]}")
    name: str = Field(..., description="Lever name")
    lever_type: LeverType = Field(..., description="Lever category")
    description: str = Field(default="")
    estimated_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    estimated_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    scope_applicability: List[str] = Field(default_factory=list)
    capex_required_eur: float = Field(default=0.0, ge=0.0)
    opex_impact_eur: float = Field(default=0.0)
    implementation_start_year: int = Field(default=2025)
    implementation_end_year: int = Field(default=2030)
    confidence_level: str = Field(default="medium")
    feasibility_score: float = Field(default=0.0, ge=0.0, le=5.0)


class TransitionAction(BaseModel):
    """A specific action within the transition plan."""
    action_id: str = Field(default_factory=lambda: f"ta-{_new_uuid()[:8]}")
    name: str = Field(..., description="Action name")
    description: str = Field(default="")
    lever_id: str = Field(default="", description="Associated lever")
    status: ActionStatus = Field(default=ActionStatus.PLANNED)
    target_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    capex_eur: float = Field(default=0.0, ge=0.0)
    start_year: int = Field(default=2025)
    end_year: int = Field(default=2030)
    responsible_unit: str = Field(default="")
    is_taxonomy_aligned: bool = Field(default=False)
    locked_in_emissions_tco2e: float = Field(default=0.0, ge=0.0)


class GapAnalysisItem(BaseModel):
    """Gap analysis between target and planned reductions."""
    scope: str = Field(default="")
    target_reduction_tco2e: float = Field(default=0.0)
    planned_reduction_tco2e: float = Field(default=0.0)
    gap_tco2e: float = Field(default=0.0)
    gap_pct: float = Field(default=0.0)
    gap_status: str = Field(default="on_track")


class TransitionPlanInput(BaseModel):
    """Input data model for TransitionPlanWorkflow."""
    baseline_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    baseline_year: int = Field(default=2019, ge=1990, le=2050)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    target_year: int = Field(default=2030, ge=2025, le=2060)
    net_zero_year: int = Field(default=2050, ge=2030, le=2070)
    target_reduction_pct: float = Field(default=42.0, ge=0.0, le=100.0)
    scope_1_baseline_tco2e: float = Field(default=0.0, ge=0.0)
    scope_2_baseline_tco2e: float = Field(default=0.0, ge=0.0)
    scope_3_baseline_tco2e: float = Field(default=0.0, ge=0.0)
    current_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    levers: List[DecarbonizationLever] = Field(default_factory=list)
    actions: List[TransitionAction] = Field(default_factory=list)
    scenario: ScenarioType = Field(default=ScenarioType.SBTI_1_5C)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class TransitionPlanResult(BaseModel):
    """Complete result from transition plan workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="transition_plan")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    baseline_emissions_tco2e: float = Field(default=0.0)
    current_emissions_tco2e: float = Field(default=0.0)
    target_emissions_tco2e: float = Field(default=0.0)
    planned_reduction_tco2e: float = Field(default=0.0)
    total_gap_tco2e: float = Field(default=0.0)
    gap_analysis: List[GapAnalysisItem] = Field(default_factory=list)
    levers: List[DecarbonizationLever] = Field(default_factory=list)
    actions: List[TransitionAction] = Field(default_factory=list)
    total_capex_eur: float = Field(default=0.0)
    locked_in_emissions_tco2e: float = Field(default=0.0)
    scenario_aligned: bool = Field(default=False)
    scenario_used: str = Field(default="")
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# SCENARIO REDUCTION RATES (annual % reduction required)
# =============================================================================

SCENARIO_ANNUAL_RATES: Dict[str, float] = {
    "iea_nze_2050": 7.6,
    "iea_sds": 5.0,
    "iea_aps": 3.5,
    "ipcc_ssp1_1.9": 7.0,
    "ipcc_ssp1_2.6": 4.2,
    "sbti_1.5c": 4.2,
    "sbti_wb2c": 2.5,
    "custom": 0.0,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class TransitionPlanWorkflow:
    """
    6-phase transition plan development workflow for ESRS E1-1.

    Implements climate transition plan creation with baseline assessment,
    decarbonization lever identification, action planning with CAPEX,
    gap analysis against targets, scenario validation (1.5C/2C), and
    disclosure-ready report generation.

    Zero-hallucination: all gap calculations and scenario comparisons
    use deterministic arithmetic.

    Example:
        >>> wf = TransitionPlanWorkflow()
        >>> inp = TransitionPlanInput(baseline_emissions_tco2e=10000)
        >>> result = await wf.execute(inp)
        >>> assert result.total_gap_tco2e >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize TransitionPlanWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._levers: List[DecarbonizationLever] = []
        self._actions: List[TransitionAction] = []
        self._gap_items: List[GapAnalysisItem] = []
        self._scenario_aligned: bool = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.BASELINE_ASSESSMENT.value, "description": "Establish current emissions baseline"},
            {"name": WorkflowPhase.LEVER_IDENTIFICATION.value, "description": "Identify decarbonization levers"},
            {"name": WorkflowPhase.ACTION_PLANNING.value, "description": "Define actions, timelines, investments"},
            {"name": WorkflowPhase.GAP_ANALYSIS.value, "description": "Assess gap between targets and planned actions"},
            {"name": WorkflowPhase.SCENARIO_VALIDATION.value, "description": "Validate against climate scenarios"},
            {"name": WorkflowPhase.REPORT_GENERATION.value, "description": "Produce E1-1 disclosure data"},
        ]

    def validate_inputs(self, input_data: TransitionPlanInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if input_data.baseline_emissions_tco2e <= 0:
            issues.append("Baseline emissions must be positive")
        if input_data.target_year <= input_data.baseline_year:
            issues.append("Target year must be after baseline year")
        if input_data.net_zero_year < input_data.target_year:
            issues.append("Net-zero year must be at or after target year")
        return issues

    async def execute(
        self,
        input_data: Optional[TransitionPlanInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TransitionPlanResult:
        """
        Execute the 6-phase transition plan workflow.

        Args:
            input_data: Full input model (preferred).
            config: Configuration overrides.

        Returns:
            TransitionPlanResult with gap analysis, levers, and scenario alignment.
        """
        if input_data is None:
            input_data = TransitionPlanInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting transition plan workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_baseline_assessment(input_data))
            phases_done += 1
            phase_results.append(await self._phase_lever_identification(input_data))
            phases_done += 1
            phase_results.append(await self._phase_action_planning(input_data))
            phases_done += 1
            phase_results.append(await self._phase_gap_analysis(input_data))
            phases_done += 1
            phase_results.append(await self._phase_scenario_validation(input_data))
            phases_done += 1
            phase_results.append(await self._phase_report_generation(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Transition plan workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        target_emissions = input_data.baseline_emissions_tco2e * (1 - input_data.target_reduction_pct / 100)
        planned_reduction = sum(lv.estimated_reduction_tco2e for lv in self._levers)
        total_gap = max(0, (input_data.baseline_emissions_tco2e - target_emissions) - planned_reduction)
        total_capex = sum(a.capex_eur for a in self._actions)
        locked_in = sum(a.locked_in_emissions_tco2e for a in self._actions)

        result = TransitionPlanResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            baseline_emissions_tco2e=input_data.baseline_emissions_tco2e,
            current_emissions_tco2e=input_data.current_emissions_tco2e,
            target_emissions_tco2e=round(target_emissions, 2),
            planned_reduction_tco2e=round(planned_reduction, 2),
            total_gap_tco2e=round(total_gap, 2),
            gap_analysis=self._gap_items,
            levers=self._levers,
            actions=self._actions,
            total_capex_eur=round(total_capex, 2),
            locked_in_emissions_tco2e=round(locked_in, 2),
            scenario_aligned=self._scenario_aligned,
            scenario_used=input_data.scenario.value,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Transition plan %s completed in %.2fs: gap=%.2f tCO2e, CAPEX=%.0f EUR",
            self.workflow_id, elapsed, total_gap, total_capex,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Baseline Assessment
    # -------------------------------------------------------------------------

    async def _phase_baseline_assessment(
        self, input_data: TransitionPlanInput,
    ) -> PhaseResult:
        """Establish the emissions baseline for transition planning."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        baseline = input_data.baseline_emissions_tco2e
        current = input_data.current_emissions_tco2e or baseline
        change_pct = round(((current - baseline) / baseline * 100) if baseline > 0 else 0.0, 2)

        outputs["baseline_emissions_tco2e"] = baseline
        outputs["current_emissions_tco2e"] = current
        outputs["change_since_baseline_pct"] = change_pct
        outputs["baseline_year"] = input_data.baseline_year
        outputs["scope_1_baseline"] = input_data.scope_1_baseline_tco2e
        outputs["scope_2_baseline"] = input_data.scope_2_baseline_tco2e
        outputs["scope_3_baseline"] = input_data.scope_3_baseline_tco2e

        scope_sum = (
            input_data.scope_1_baseline_tco2e
            + input_data.scope_2_baseline_tco2e
            + input_data.scope_3_baseline_tco2e
        )
        if scope_sum > 0 and abs(scope_sum - baseline) > baseline * 0.01:
            warnings.append(
                f"Scope totals ({scope_sum:.0f}) differ from baseline total "
                f"({baseline:.0f}) by more than 1%"
            )

        if current > baseline:
            warnings.append(
                f"Current emissions ({current:.0f}) exceed baseline ({baseline:.0f})"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 BaselineAssessment: baseline=%.0f, current=%.0f, change=%.1f%%",
            baseline, current, change_pct,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.BASELINE_ASSESSMENT.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Lever Identification
    # -------------------------------------------------------------------------

    async def _phase_lever_identification(
        self, input_data: TransitionPlanInput,
    ) -> PhaseResult:
        """Identify and catalog decarbonization levers."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._levers = list(input_data.levers)

        type_counts: Dict[str, int] = {}
        total_potential = 0.0
        for lever in self._levers:
            type_counts[lever.lever_type.value] = type_counts.get(lever.lever_type.value, 0) + 1
            total_potential += lever.estimated_reduction_tco2e

        outputs["levers_identified"] = len(self._levers)
        outputs["lever_type_distribution"] = type_counts
        outputs["total_reduction_potential_tco2e"] = round(total_potential, 2)
        outputs["total_capex_required_eur"] = round(
            sum(lv.capex_required_eur for lv in self._levers), 2
        )

        if not self._levers:
            warnings.append("No decarbonization levers provided")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 LeverIdentification: %d levers, %.0f tCO2e reduction potential",
            len(self._levers), total_potential,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.LEVER_IDENTIFICATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Action Planning
    # -------------------------------------------------------------------------

    async def _phase_action_planning(
        self, input_data: TransitionPlanInput,
    ) -> PhaseResult:
        """Define specific actions with timelines and investments."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._actions = list(input_data.actions)

        status_counts: Dict[str, int] = {}
        for action in self._actions:
            status_counts[action.status.value] = status_counts.get(action.status.value, 0) + 1

        total_capex = sum(a.capex_eur for a in self._actions)
        total_target_reduction = sum(a.target_reduction_tco2e for a in self._actions)
        taxonomy_aligned_count = sum(1 for a in self._actions if a.is_taxonomy_aligned)

        outputs["actions_defined"] = len(self._actions)
        outputs["status_distribution"] = status_counts
        outputs["total_capex_eur"] = round(total_capex, 2)
        outputs["total_target_reduction_tco2e"] = round(total_target_reduction, 2)
        outputs["taxonomy_aligned_count"] = taxonomy_aligned_count
        outputs["taxonomy_aligned_pct"] = round(
            (taxonomy_aligned_count / len(self._actions) * 100)
            if self._actions else 0.0, 1
        )

        if not self._actions:
            warnings.append("No transition actions defined")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ActionPlanning: %d actions, CAPEX=%.0f EUR",
            len(self._actions), total_capex,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.ACTION_PLANNING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Gap Analysis
    # -------------------------------------------------------------------------

    async def _phase_gap_analysis(
        self, input_data: TransitionPlanInput,
    ) -> PhaseResult:
        """Assess gap between reduction targets and planned actions."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._gap_items = []

        target_emissions = input_data.baseline_emissions_tco2e * (
            1 - input_data.target_reduction_pct / 100
        )
        required_reduction = input_data.baseline_emissions_tco2e - target_emissions
        planned_reduction = sum(lv.estimated_reduction_tco2e for lv in self._levers)

        overall_gap = max(0, required_reduction - planned_reduction)
        overall_gap_pct = round(
            (overall_gap / required_reduction * 100) if required_reduction > 0 else 0.0, 2
        )

        # Per-scope gap analysis
        for scope_label, scope_baseline in [
            ("scope_1", input_data.scope_1_baseline_tco2e),
            ("scope_2", input_data.scope_2_baseline_tco2e),
            ("scope_3", input_data.scope_3_baseline_tco2e),
        ]:
            if scope_baseline <= 0:
                continue
            scope_target = scope_baseline * (1 - input_data.target_reduction_pct / 100)
            scope_required = scope_baseline - scope_target
            scope_planned = sum(
                lv.estimated_reduction_tco2e for lv in self._levers
                if scope_label in [s.lower() for s in lv.scope_applicability]
            )
            scope_gap = max(0, scope_required - scope_planned)
            gap_status = "on_track" if scope_gap == 0 else (
                "minor_gap" if scope_gap < scope_required * 0.2 else "significant_gap"
            )

            self._gap_items.append(GapAnalysisItem(
                scope=scope_label,
                target_reduction_tco2e=round(scope_required, 2),
                planned_reduction_tco2e=round(scope_planned, 2),
                gap_tco2e=round(scope_gap, 2),
                gap_pct=round(
                    (scope_gap / scope_required * 100) if scope_required > 0 else 0.0, 2
                ),
                gap_status=gap_status,
            ))

        outputs["required_reduction_tco2e"] = round(required_reduction, 2)
        outputs["planned_reduction_tco2e"] = round(planned_reduction, 2)
        outputs["overall_gap_tco2e"] = round(overall_gap, 2)
        outputs["overall_gap_pct"] = overall_gap_pct
        outputs["scope_gaps"] = len(self._gap_items)

        if overall_gap > 0:
            warnings.append(
                f"Transition plan has a {overall_gap:.0f} tCO2e gap "
                f"({overall_gap_pct:.1f}% of required reduction)"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 GapAnalysis: gap=%.0f tCO2e (%.1f%%)",
            overall_gap, overall_gap_pct,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.GAP_ANALYSIS.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Scenario Validation
    # -------------------------------------------------------------------------

    async def _phase_scenario_validation(
        self, input_data: TransitionPlanInput,
    ) -> PhaseResult:
        """Validate the transition plan against climate scenarios."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        scenario_rate = SCENARIO_ANNUAL_RATES.get(input_data.scenario.value, 0.0)
        years_to_target = max(1, input_data.target_year - input_data.baseline_year)
        required_annual_reduction = (
            input_data.target_reduction_pct / years_to_target
        ) if years_to_target > 0 else 0.0

        self._scenario_aligned = required_annual_reduction >= scenario_rate

        outputs["scenario"] = input_data.scenario.value
        outputs["scenario_annual_rate_pct"] = scenario_rate
        outputs["plan_annual_rate_pct"] = round(required_annual_reduction, 2)
        outputs["scenario_aligned"] = self._scenario_aligned
        outputs["target_year"] = input_data.target_year
        outputs["net_zero_year"] = input_data.net_zero_year

        if not self._scenario_aligned:
            warnings.append(
                f"Plan annual reduction rate ({required_annual_reduction:.1f}%) is below "
                f"{input_data.scenario.value} requirement ({scenario_rate}%/yr)"
            )

        # Check net-zero feasibility
        if input_data.net_zero_year > 2050 and input_data.scenario in (
            ScenarioType.IEA_NZE_2050, ScenarioType.SBTI_1_5C
        ):
            warnings.append(
                f"Net-zero year ({input_data.net_zero_year}) is after 2050, "
                f"inconsistent with {input_data.scenario.value} scenario"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 ScenarioValidation: %s aligned=%s, rate=%.1f%%/yr",
            input_data.scenario.value, self._scenario_aligned, required_annual_reduction,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.SCENARIO_VALIDATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 6: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(
        self, input_data: TransitionPlanInput,
    ) -> PhaseResult:
        """Generate E1-1 disclosure-ready output."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        target_emissions = input_data.baseline_emissions_tco2e * (
            1 - input_data.target_reduction_pct / 100
        )

        outputs["e1_1_disclosure"] = {
            "has_transition_plan": True,
            "baseline_year": input_data.baseline_year,
            "baseline_emissions_tco2e": input_data.baseline_emissions_tco2e,
            "target_year": input_data.target_year,
            "target_emissions_tco2e": round(target_emissions, 2),
            "target_reduction_pct": input_data.target_reduction_pct,
            "net_zero_year": input_data.net_zero_year,
            "scenario_used": input_data.scenario.value,
            "scenario_aligned": self._scenario_aligned,
            "decarbonization_levers_count": len(self._levers),
            "total_capex_eur": round(sum(a.capex_eur for a in self._actions), 2),
            "locked_in_emissions_tco2e": round(
                sum(a.locked_in_emissions_tco2e for a in self._actions), 2
            ),
            "reporting_year": input_data.reporting_year,
        }

        outputs["report_ready"] = True

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 6 ReportGeneration: E1-1 disclosure ready")
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_GENERATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: TransitionPlanResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
