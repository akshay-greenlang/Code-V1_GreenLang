# -*- coding: utf-8 -*-
"""
Decarbonization Roadmap Workflow
=================================

Five-phase workflow for building a science-based decarbonization roadmap
for manufacturing facilities, including technology evaluation, investment
planning, and SBTi alignment.

Phases:
    1. BaselineAssessment - Establish emissions baseline and trajectory
    2. TechnologyEvaluation - Evaluate abatement technology options
    3. TargetSetting - Set SBTi-aligned reduction targets
    4. InvestmentPlanning - Build investment plan with cost-benefit analysis
    5. MonitoringSetup - Define KPIs and monitoring framework

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC time with timezone info."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


class PhaseStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=_utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        self.phase_states[phase_name] = status


class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
#  Input / Result
# ---------------------------------------------------------------------------

class TechnologyOption(BaseModel):
    """Abatement technology option."""
    technology_id: str = Field(...)
    technology_name: str = Field(default="")
    category: str = Field(default="", description="energy_efficiency, fuel_switch, electrification, ccus, process_change")
    abatement_potential_tco2e: float = Field(default=0.0, ge=0.0)
    capex_eur: float = Field(default=0.0, ge=0.0)
    annual_opex_eur: float = Field(default=0.0)
    payback_years: float = Field(default=0.0, ge=0.0)
    maturity: str = Field(default="commercial", description="commercial, pilot, rd")
    applicable_facilities: List[str] = Field(default_factory=list)


class DecarbonizationInput(BaseModel):
    """Input for decarbonization roadmap workflow."""
    organization_id: str = Field(...)
    baseline_year: int = Field(..., ge=2015, le=2100)
    target_year: int = Field(..., ge=2025, le=2100)
    target_reduction_pct: float = Field(..., gt=0.0, le=100.0)
    baseline_emissions: Dict[str, float] = Field(
        default_factory=dict,
        description="Scope -> tCO2e baseline values",
    )
    facility_data: List[Dict[str, Any]] = Field(default_factory=list)
    technology_options: List[TechnologyOption] = Field(default_factory=list)
    carbon_price_eur_per_tco2e: float = Field(default=80.0, ge=0.0)
    annual_budget_eur: float = Field(default=0.0, ge=0.0)
    sbti_pathway: str = Field(default="1.5C", description="1.5C or well_below_2C")
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("target_year")
    @classmethod
    def validate_target_after_baseline(cls, v: int, info: Any) -> int:
        """Ensure target year is after baseline year."""
        baseline = info.data.get("baseline_year", 2020)
        if v <= baseline:
            raise ValueError("Target year must be after baseline year")
        return v


class DecarbonizationResult(WorkflowResult):
    """Result from the decarbonization roadmap workflow."""
    baseline: Dict[str, Any] = Field(default_factory=dict)
    target: Dict[str, Any] = Field(default_factory=dict)
    pathway_options: List[Dict[str, Any]] = Field(default_factory=list)
    investment_required: float = Field(default=0.0)
    annual_milestones: List[Dict[str, Any]] = Field(default_factory=list)
    sbti_alignment: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Phase 1: Baseline Assessment
# ---------------------------------------------------------------------------

class BaselineAssessmentPhase:
    """Establish emissions baseline and business-as-usual trajectory."""

    PHASE_NAME = "baseline_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Calculate baseline emissions and BAU projection."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            baseline_em = config.get("baseline_emissions", {})
            baseline_year = config.get("baseline_year", 2020)
            target_year = config.get("target_year", 2030)

            scope1 = baseline_em.get("scope1", 0.0)
            scope2 = baseline_em.get("scope2", 0.0)
            scope3 = baseline_em.get("scope3", 0.0)
            total_baseline = scope1 + scope2 + scope3

            outputs["baseline_year"] = baseline_year
            outputs["scope1_baseline"] = scope1
            outputs["scope2_baseline"] = scope2
            outputs["scope3_baseline"] = scope3
            outputs["total_baseline"] = total_baseline

            # BAU projection (assume 2% annual growth)
            years = target_year - baseline_year
            growth_rate = 0.02
            bau_projection = []
            for yr in range(years + 1):
                year_val = baseline_year + yr
                bau_em = total_baseline * ((1 + growth_rate) ** yr)
                bau_projection.append({
                    "year": year_val,
                    "bau_emissions_tco2e": round(bau_em, 2),
                })
            outputs["bau_projection"] = bau_projection
            outputs["bau_target_year_emissions"] = round(
                total_baseline * ((1 + growth_rate) ** years), 2
            )

            if total_baseline == 0:
                errors.append("Baseline emissions are zero; cannot build roadmap")

            status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED

        except Exception as exc:
            logger.error("BaselineAssessment failed: %s", exc, exc_info=True)
            errors.append(str(exc))
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# ---------------------------------------------------------------------------
#  Phase 2: Technology Evaluation
# ---------------------------------------------------------------------------

class TechnologyEvaluationPhase:
    """Evaluate and rank abatement technology options."""

    PHASE_NAME = "technology_evaluation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Rank technologies by marginal abatement cost (MAC)."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            techs = config.get("technology_options", [])
            carbon_price = config.get("carbon_price_eur_per_tco2e", 80.0)

            evaluated: List[Dict[str, Any]] = []
            total_abatement = 0.0
            total_capex = 0.0

            for t in techs:
                abatement = t.get("abatement_potential_tco2e", 0.0)
                capex = t.get("capex_eur", 0.0)
                opex = t.get("annual_opex_eur", 0.0)
                payback = t.get("payback_years", 0.0)

                # Marginal abatement cost (simplified)
                if abatement > 0:
                    mac = (capex / max(payback, 1) + opex) / abatement
                else:
                    mac = float("inf")

                carbon_savings_eur = abatement * carbon_price
                net_benefit = carbon_savings_eur - (capex / max(payback, 1) + opex)

                total_abatement += abatement
                total_capex += capex

                evaluated.append({
                    "technology_id": t.get("technology_id", ""),
                    "technology_name": t.get("technology_name", ""),
                    "category": t.get("category", ""),
                    "abatement_tco2e": abatement,
                    "capex_eur": capex,
                    "mac_eur_per_tco2e": round(mac, 2) if mac != float("inf") else 0.0,
                    "net_annual_benefit_eur": round(net_benefit, 2),
                    "payback_years": payback,
                    "maturity": t.get("maturity", "commercial"),
                    "recommended": mac < carbon_price * 1.5,
                })

            # Sort by MAC ascending
            evaluated.sort(key=lambda x: x["mac_eur_per_tco2e"])

            outputs["evaluated_technologies"] = evaluated
            outputs["total_abatement_potential"] = round(total_abatement, 2)
            outputs["total_capex_required"] = round(total_capex, 2)
            outputs["recommended_count"] = sum(1 for e in evaluated if e["recommended"])
            outputs["carbon_price_reference"] = carbon_price

            if not techs:
                warnings.append("No technology options provided")

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("TechnologyEvaluation failed: %s", exc, exc_info=True)
            errors.append(str(exc))
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# ---------------------------------------------------------------------------
#  Phase 3: Target Setting
# ---------------------------------------------------------------------------

class TargetSettingPhase:
    """Set SBTi-aligned reduction targets and pathway."""

    PHASE_NAME = "target_setting"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Define near-term and long-term reduction targets."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            baseline = context.get_phase_output("baseline_assessment")
            tech_eval = context.get_phase_output("technology_evaluation")

            total_baseline = baseline.get("total_baseline", 0.0)
            target_pct = config.get("target_reduction_pct", 42.0)
            target_year = config.get("target_year", 2030)
            baseline_year = config.get("baseline_year", 2020)
            pathway = config.get("sbti_pathway", "1.5C")

            target_emissions = total_baseline * (1.0 - target_pct / 100.0)
            required_reduction = total_baseline - target_emissions

            # SBTi alignment check
            years_span = target_year - baseline_year
            annual_reduction_rate = target_pct / max(years_span, 1)

            # SBTi 1.5C requires 4.2% per year for Scope 1+2
            sbti_minimum_rate = 4.2 if pathway == "1.5C" else 2.5
            sbti_aligned = annual_reduction_rate >= sbti_minimum_rate

            total_abatement = tech_eval.get("total_abatement_potential", 0.0)
            tech_gap = max(0, required_reduction - total_abatement)

            outputs["target"] = {
                "target_year": target_year,
                "target_reduction_pct": target_pct,
                "target_emissions_tco2e": round(target_emissions, 2),
                "required_reduction_tco2e": round(required_reduction, 2),
                "annual_reduction_rate_pct": round(annual_reduction_rate, 2),
            }
            outputs["sbti_alignment"] = {
                "pathway": pathway,
                "minimum_annual_rate": sbti_minimum_rate,
                "actual_annual_rate": round(annual_reduction_rate, 2),
                "aligned": sbti_aligned,
                "gap_pct": round(max(0, sbti_minimum_rate - annual_reduction_rate), 2),
            }
            outputs["technology_coverage"] = {
                "total_abatement_available": round(total_abatement, 2),
                "reduction_needed": round(required_reduction, 2),
                "technology_gap_tco2e": round(tech_gap, 2),
                "coverage_pct": round(
                    min(total_abatement / max(required_reduction, 0.001) * 100, 100), 1
                ),
            }

            if not sbti_aligned:
                warnings.append(
                    f"Annual reduction rate ({annual_reduction_rate:.1f}%) below "
                    f"SBTi {pathway} minimum ({sbti_minimum_rate}%)"
                )
            if tech_gap > 0:
                warnings.append(
                    f"Technology gap of {tech_gap:.0f} tCO2e; additional measures needed"
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("TargetSetting failed: %s", exc, exc_info=True)
            errors.append(str(exc))
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# ---------------------------------------------------------------------------
#  Phase 4: Investment Planning
# ---------------------------------------------------------------------------

class InvestmentPlanningPhase:
    """Build investment plan with cost-benefit analysis."""

    PHASE_NAME = "investment_planning"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Sequence technology investments across the roadmap timeline."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            baseline = context.get_phase_output("baseline_assessment")
            tech_eval = context.get_phase_output("technology_evaluation")
            targets = context.get_phase_output("target_setting")

            baseline_year = config.get("baseline_year", 2020)
            target_year = config.get("target_year", 2030)
            annual_budget = config.get("annual_budget_eur", 0.0)
            carbon_price = config.get("carbon_price_eur_per_tco2e", 80.0)
            total_baseline = baseline.get("total_baseline", 0.0)

            techs = tech_eval.get("evaluated_technologies", [])
            recommended = [t for t in techs if t.get("recommended", False)]

            # Sequence by MAC (cheapest first)
            years_span = target_year - baseline_year
            total_investment = sum(t.get("capex_eur", 0.0) for t in recommended)
            total_abatement = sum(t.get("abatement_tco2e", 0.0) for t in recommended)
            total_carbon_savings = total_abatement * carbon_price

            # Build annual milestones
            milestones: List[Dict[str, Any]] = []
            remaining = total_baseline
            cumulative_capex = 0.0
            tech_idx = 0

            for yr in range(years_span):
                year_val = baseline_year + yr + 1
                year_abatement = 0.0
                year_capex = 0.0

                # Deploy technologies that fit within annual budget
                while tech_idx < len(recommended):
                    tech = recommended[tech_idx]
                    capex = tech.get("capex_eur", 0.0)
                    if annual_budget > 0 and year_capex + capex > annual_budget:
                        break
                    year_abatement += tech.get("abatement_tco2e", 0.0)
                    year_capex += capex
                    tech_idx += 1

                remaining = max(0, remaining - year_abatement)
                cumulative_capex += year_capex

                milestones.append({
                    "year": year_val,
                    "expected_emissions_tco2e": round(remaining, 2),
                    "annual_abatement_tco2e": round(year_abatement, 2),
                    "annual_capex_eur": round(year_capex, 2),
                    "cumulative_capex_eur": round(cumulative_capex, 2),
                    "reduction_from_baseline_pct": round(
                        (1 - remaining / max(total_baseline, 0.001)) * 100, 2
                    ),
                })

            outputs["investment_required"] = round(total_investment, 2)
            outputs["total_carbon_savings_eur"] = round(total_carbon_savings, 2)
            outputs["roi_pct"] = round(
                (total_carbon_savings - total_investment) / max(total_investment, 0.001) * 100, 1
            )
            outputs["annual_milestones"] = milestones
            outputs["technologies_deployed"] = tech_idx

            if annual_budget > 0 and total_investment > annual_budget * years_span:
                warnings.append("Total investment exceeds cumulative budget over roadmap period")

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("InvestmentPlanning failed: %s", exc, exc_info=True)
            errors.append(str(exc))
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# ---------------------------------------------------------------------------
#  Phase 5: Monitoring Setup
# ---------------------------------------------------------------------------

class MonitoringSetupPhase:
    """Define KPIs and monitoring framework for decarbonization tracking."""

    PHASE_NAME = "monitoring_setup"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Configure monitoring KPIs and review cadence."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            targets = context.get_phase_output("target_setting")
            target_data = targets.get("target", {})

            kpis = [
                {"kpi": "Total GHG Emissions (tCO2e)", "frequency": "quarterly", "target": target_data.get("target_emissions_tco2e", 0.0)},
                {"kpi": "Emission Intensity (tCO2e/unit)", "frequency": "quarterly", "target": "year-on-year reduction"},
                {"kpi": "Energy Intensity (MWh/unit)", "frequency": "monthly", "target": "year-on-year reduction"},
                {"kpi": "Renewable Energy Share (%)", "frequency": "quarterly", "target": "year-on-year increase"},
                {"kpi": "Technology Deployment Progress (%)", "frequency": "quarterly", "target": "per roadmap schedule"},
                {"kpi": "CAPEX Utilization (%)", "frequency": "quarterly", "target": ">90% of annual budget"},
                {"kpi": "SBTi Alignment Status", "frequency": "annually", "target": "aligned"},
                {"kpi": "Carbon Price Exposure (EUR)", "frequency": "quarterly", "target": "within risk appetite"},
            ]

            review_cadence = {
                "operational_review": "monthly",
                "management_review": "quarterly",
                "board_review": "semi-annually",
                "sbti_progress_report": "annually",
                "external_assurance": "annually",
            }

            outputs["kpis"] = kpis
            outputs["review_cadence"] = review_cadence
            outputs["monitoring_framework"] = "GreenLang PACK-013 Decarbonization Monitor"
            outputs["dashboard_metrics"] = [k["kpi"] for k in kpis]

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("MonitoringSetup failed: %s", exc, exc_info=True)
            errors.append(str(exc))
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# ---------------------------------------------------------------------------
#  Workflow Orchestrator
# ---------------------------------------------------------------------------

class DecarbonizationRoadmapWorkflow:
    """
    Five-phase decarbonization roadmap workflow.

    Builds a science-based decarbonization pathway for manufacturing
    organizations with technology evaluation, investment planning,
    and SBTi alignment assessment.
    """

    WORKFLOW_NAME = "decarbonization_roadmap"

    PHASE_ORDER = [
        "baseline_assessment", "technology_evaluation",
        "target_setting", "investment_planning", "monitoring_setup",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize DecarbonizationRoadmapWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "baseline_assessment": BaselineAssessmentPhase(),
            "technology_evaluation": TechnologyEvaluationPhase(),
            "target_setting": TargetSettingPhase(),
            "investment_planning": InvestmentPlanningPhase(),
            "monitoring_setup": MonitoringSetupPhase(),
        }

    async def run(self, input_data: DecarbonizationInput) -> DecarbonizationResult:
        """Execute the complete 5-phase decarbonization roadmap workflow."""
        started_at = _utcnow()
        logger.info("Starting decarbonization roadmap workflow %s", self.workflow_id)
        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=input_data.model_dump(),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                ))
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            self._notify_progress(phase_name, f"Starting: {phase_name}", idx / len(self.PHASE_ORDER))
            context.mark_phase(phase_name, PhaseStatus.RUNNING)
            try:
                result = await self._phases[phase_name].execute(context)
                completed_phases.append(result)
                if result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, result.status)
                    if phase_name == "baseline_assessment":
                        overall_status = WorkflowStatus.FAILED
                        break
                context.errors.extend(result.errors)
                context.warnings.extend(result.warnings)
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.FAILED,
                    started_at=_utcnow(), errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                ))
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = _utcnow()
        duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        return DecarbonizationResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            baseline=summary.get("baseline", {}),
            target=summary.get("target", {}),
            pathway_options=summary.get("pathway_options", []),
            investment_required=summary.get("investment_required", 0.0),
            annual_milestones=summary.get("annual_milestones", []),
            sbti_alignment=summary.get("sbti_alignment", {}),
        )

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build summary from phase outputs."""
        bl = context.get_phase_output("baseline_assessment")
        tech = context.get_phase_output("technology_evaluation")
        tgt = context.get_phase_output("target_setting")
        inv = context.get_phase_output("investment_planning")
        return {
            "baseline": {
                "year": bl.get("baseline_year"),
                "total_tco2e": bl.get("total_baseline", 0.0),
            },
            "target": tgt.get("target", {}),
            "pathway_options": tech.get("evaluated_technologies", []),
            "investment_required": inv.get("investment_required", 0.0),
            "annual_milestones": inv.get("annual_milestones", []),
            "sbti_alignment": tgt.get("sbti_alignment", {}),
        }

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        """Send progress notification via callback if configured."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
