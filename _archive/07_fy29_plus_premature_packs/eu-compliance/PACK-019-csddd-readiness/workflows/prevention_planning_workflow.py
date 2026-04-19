# -*- coding: utf-8 -*-
"""
CSDDD Prevention Planning Workflow
===============================================

4-phase workflow for designing and planning prevention and mitigation measures
under the EU Corporate Sustainability Due Diligence Directive (CSDDD / CS3D).
Covers measure design, resource allocation, implementation timelines, and
effectiveness metrics (KPIs).

Phases:
    1. MeasureDesign           -- Design prevention/mitigation measures per impact
    2. ResourceAllocation      -- Allocate budget, staff, and resources
    3. ImplementationTimeline  -- Build phased implementation plan
    4. EffectivenessMetrics    -- Define KPIs to measure plan effectiveness

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Art. 7: Preventing potential adverse impacts
    - Art. 7(2)(a): Prevention action plan with timelines
    - Art. 7(2)(b): Contractual assurances from business partners
    - Art. 7(2)(d): Financial or other support for SMEs
    - Art. 7(4): Appropriate measures where prevention is not immediately possible
    - Art. 8: Bringing actual adverse impacts to an end
    - Art. 8(3): Corrective action plan
    - Art. 8(6): Disengagement as a last resort

Author: GreenLang Team
Version: 19.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class WorkflowPhase(str, Enum):
    """Phases of the prevention planning workflow."""
    MEASURE_DESIGN = "measure_design"
    RESOURCE_ALLOCATION = "resource_allocation"
    IMPLEMENTATION_TIMELINE = "implementation_timeline"
    EFFECTIVENESS_METRICS = "effectiveness_metrics"

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

class MeasureType(str, Enum):
    """Type of prevention/mitigation measure per Art. 7-8."""
    PREVENTION = "prevention"
    MITIGATION = "mitigation"
    CORRECTIVE = "corrective"
    CONTRACTUAL = "contractual"
    CAPACITY_BUILDING = "capacity_building"
    DISENGAGEMENT = "disengagement"

class MeasureStatus(str, Enum):
    """Implementation status of a measure."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    DELAYED = "delayed"

class TimelinePhase(str, Enum):
    """Implementation timeline phases."""
    IMMEDIATE = "immediate"       # 0-3 months
    SHORT_TERM = "short_term"     # 3-12 months
    MEDIUM_TERM = "medium_term"   # 1-3 years
    LONG_TERM = "long_term"       # 3+ years

class ResourceCategory(str, Enum):
    """Resource allocation categories."""
    BUDGET = "budget"
    PERSONNEL = "personnel"
    TECHNOLOGY = "technology"
    TRAINING = "training"
    EXTERNAL_CONSULTING = "external_consulting"

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

class PrioritizedImpact(BaseModel):
    """Impact from upstream assessment that needs prevention measures."""
    impact_id: str = Field(..., description="Impact ID from assessment")
    impact_name: str = Field(default="", description="Description of impact")
    category: str = Field(default="", description="human_rights or environment")
    severity_level: str = Field(default="medium", description="critical/high/medium/low")
    priority_score: float = Field(default=0.0, ge=0.0, le=100.0)
    value_chain_stage: str = Field(default="")
    country_code: str = Field(default="")

class ExistingMeasure(BaseModel):
    """Existing prevention/mitigation measure already in place."""
    measure_id: str = Field(default_factory=lambda: f"em-{_new_uuid()[:8]}")
    measure_name: str = Field(default="")
    measure_type: MeasureType = Field(default=MeasureType.PREVENTION)
    related_impact_ids: List[str] = Field(default_factory=list)
    status: MeasureStatus = Field(default=MeasureStatus.IMPLEMENTED)
    effectiveness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    annual_cost_eur: float = Field(default=0.0, ge=0.0)

class PreventionMeasure(BaseModel):
    """Designed prevention or mitigation measure."""
    measure_id: str = Field(default_factory=lambda: f"pm-{_new_uuid()[:8]}")
    measure_name: str = Field(default="")
    description: str = Field(default="")
    measure_type: MeasureType = Field(default=MeasureType.PREVENTION)
    related_impact_ids: List[str] = Field(default_factory=list)
    csddd_article: str = Field(default="art_7", description="Primary article reference")
    timeline_phase: TimelinePhase = Field(default=TimelinePhase.SHORT_TERM)
    estimated_cost_eur: float = Field(default=0.0, ge=0.0)
    responsible_department: str = Field(default="")
    kpi_name: str = Field(default="")
    kpi_target: str = Field(default="")
    status: MeasureStatus = Field(default=MeasureStatus.PLANNED)

class PreventionPlanningInput(BaseModel):
    """Input data model for PreventionPlanningWorkflow."""
    entity_id: str = Field(default="", description="Reporting entity ID")
    entity_name: str = Field(default="", description="Reporting entity name")
    reporting_year: int = Field(default=2026, ge=2024, le=2050)
    prioritized_impacts: List[PrioritizedImpact] = Field(
        default_factory=list, description="Impacts from impact assessment"
    )
    existing_measures: List[ExistingMeasure] = Field(
        default_factory=list, description="Already-implemented measures"
    )
    total_budget_eur: float = Field(
        default=0.0, ge=0.0, description="Available budget for prevention plan"
    )
    available_fte: float = Field(
        default=0.0, ge=0.0, description="Available full-time equivalents"
    )
    config: Dict[str, Any] = Field(default_factory=dict)

class ResourcePlan(BaseModel):
    """Allocated resources for the prevention plan."""
    total_budget_eur: float = Field(default=0.0, ge=0.0)
    allocated_budget_eur: float = Field(default=0.0, ge=0.0)
    remaining_budget_eur: float = Field(default=0.0, ge=0.0)
    budget_utilization_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    allocated_fte: float = Field(default=0.0, ge=0.0)
    by_category: Dict[str, float] = Field(default_factory=dict)
    by_timeline: Dict[str, float] = Field(default_factory=dict)

class PreventionPlanningResult(BaseModel):
    """Complete result from prevention planning workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="prevention_planning")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    # Measures
    new_measures: List[PreventionMeasure] = Field(default_factory=list)
    total_measures_designed: int = Field(default=0, ge=0)
    prevention_count: int = Field(default=0, ge=0)
    mitigation_count: int = Field(default=0, ge=0)
    corrective_count: int = Field(default=0, ge=0)
    contractual_count: int = Field(default=0, ge=0)
    # Resources
    resource_plan: ResourcePlan = Field(default_factory=ResourcePlan)
    total_estimated_cost_eur: float = Field(default=0.0, ge=0.0)
    # Timeline
    immediate_actions: int = Field(default=0, ge=0)
    short_term_actions: int = Field(default=0, ge=0)
    medium_term_actions: int = Field(default=0, ge=0)
    long_term_actions: int = Field(default=0, ge=0)
    # KPIs
    kpis_defined: int = Field(default=0, ge=0)
    kpi_summary: List[Dict[str, Any]] = Field(default_factory=list)
    impacts_covered: int = Field(default=0, ge=0)
    impacts_uncovered: int = Field(default=0, ge=0)
    reporting_year: int = Field(default=2026)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class PreventionPlanningWorkflow:
    """
    4-phase CSDDD prevention planning workflow.

    Designs prevention and mitigation measures, allocates resources, builds
    implementation timelines, and defines effectiveness KPIs per Art. 7-8.

    Zero-hallucination: all cost/resource calculations use deterministic
    arithmetic. No LLM in numeric calculation paths.

    Example:
        >>> wf = PreventionPlanningWorkflow()
        >>> inp = PreventionPlanningInput(prioritized_impacts=[...], total_budget_eur=500_000)
        >>> result = await wf.execute(inp)
        >>> assert result.total_measures_designed > 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize PreventionPlanningWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._measures: List[PreventionMeasure] = []
        self._resource_plan: ResourcePlan = ResourcePlan()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.MEASURE_DESIGN.value, "description": "Design prevention/mitigation measures"},
            {"name": WorkflowPhase.RESOURCE_ALLOCATION.value, "description": "Allocate budget and staff"},
            {"name": WorkflowPhase.IMPLEMENTATION_TIMELINE.value, "description": "Build implementation plan"},
            {"name": WorkflowPhase.EFFECTIVENESS_METRICS.value, "description": "Define effectiveness KPIs"},
        ]

    def validate_inputs(self, input_data: PreventionPlanningInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.prioritized_impacts:
            issues.append("No prioritized impacts provided")
        if input_data.total_budget_eur <= 0:
            issues.append("Budget not specified or zero")
        return issues

    async def execute(
        self,
        input_data: Optional[PreventionPlanningInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> PreventionPlanningResult:
        """
        Execute the 4-phase prevention planning workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            PreventionPlanningResult with measures, resources, and KPIs.
        """
        if input_data is None:
            input_data = PreventionPlanningInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting prevention planning workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_measure_design(input_data))
            phase_results.append(await self._phase_resource_allocation(input_data))
            phase_results.append(await self._phase_implementation_timeline(input_data))
            phase_results.append(await self._phase_effectiveness_metrics(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Prevention planning failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        # Count measure types
        type_counts: Dict[str, int] = {}
        for m in self._measures:
            type_counts[m.measure_type.value] = type_counts.get(m.measure_type.value, 0) + 1

        # Timeline counts
        timeline_counts: Dict[str, int] = {}
        for m in self._measures:
            timeline_counts[m.timeline_phase.value] = timeline_counts.get(m.timeline_phase.value, 0) + 1

        # Coverage
        covered_impact_ids = set()
        for m in self._measures:
            covered_impact_ids.update(m.related_impact_ids)
        all_impact_ids = set(i.impact_id for i in input_data.prioritized_impacts)

        # KPI summary
        kpi_summary = [
            {"measure_id": m.measure_id, "kpi_name": m.kpi_name, "kpi_target": m.kpi_target}
            for m in self._measures if m.kpi_name
        ]

        result = PreventionPlanningResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            new_measures=self._measures,
            total_measures_designed=len(self._measures),
            prevention_count=type_counts.get("prevention", 0),
            mitigation_count=type_counts.get("mitigation", 0),
            corrective_count=type_counts.get("corrective", 0),
            contractual_count=type_counts.get("contractual", 0),
            resource_plan=self._resource_plan,
            total_estimated_cost_eur=round(sum(m.estimated_cost_eur for m in self._measures), 2),
            immediate_actions=timeline_counts.get("immediate", 0),
            short_term_actions=timeline_counts.get("short_term", 0),
            medium_term_actions=timeline_counts.get("medium_term", 0),
            long_term_actions=timeline_counts.get("long_term", 0),
            kpis_defined=len(kpi_summary),
            kpi_summary=kpi_summary,
            impacts_covered=len(covered_impact_ids & all_impact_ids),
            impacts_uncovered=len(all_impact_ids - covered_impact_ids),
            reporting_year=input_data.reporting_year,
            executed_at=utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Prevention planning %s completed in %.2fs: %d measures, %.0f EUR",
            self.workflow_id, elapsed, len(self._measures),
            result.total_estimated_cost_eur,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Measure Design
    # -------------------------------------------------------------------------

    async def _phase_measure_design(
        self, input_data: PreventionPlanningInput,
    ) -> PhaseResult:
        """Design prevention/mitigation measures for each prioritized impact."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._measures = []

        impacts = input_data.prioritized_impacts
        existing_impact_ids = set()
        for em in input_data.existing_measures:
            existing_impact_ids.update(em.related_impact_ids)

        for imp in impacts:
            # Skip if fully covered by existing effective measures
            if imp.impact_id in existing_impact_ids:
                matching = [
                    em for em in input_data.existing_measures
                    if imp.impact_id in em.related_impact_ids
                    and em.effectiveness_pct >= 80
                ]
                if matching:
                    continue

            # Determine measure type based on severity
            if imp.severity_level == "critical":
                measure_type = MeasureType.CORRECTIVE
                article = "art_8"
                cost_factor = 1.5
            elif imp.severity_level == "high":
                measure_type = MeasureType.PREVENTION
                article = "art_7"
                cost_factor = 1.2
            elif imp.severity_level == "medium":
                measure_type = MeasureType.MITIGATION
                article = "art_7"
                cost_factor = 1.0
            else:
                measure_type = MeasureType.PREVENTION
                article = "art_7"
                cost_factor = 0.7

            # Determine timeline
            if imp.severity_level in ("critical", "high"):
                timeline = TimelinePhase.IMMEDIATE
            elif imp.severity_level == "medium":
                timeline = TimelinePhase.SHORT_TERM
            else:
                timeline = TimelinePhase.MEDIUM_TERM

            # Estimate cost as proportion of budget based on priority
            base_cost = input_data.total_budget_eur * 0.05  # 5% per measure baseline
            estimated_cost = round(base_cost * cost_factor, 2)

            # Always add a contractual assurance measure for supply chain impacts
            self._measures.append(PreventionMeasure(
                measure_name=f"Prevention measure for: {imp.impact_name}",
                description=f"Targeted {measure_type.value} for {imp.category} impact in {imp.value_chain_stage}",
                measure_type=measure_type,
                related_impact_ids=[imp.impact_id],
                csddd_article=article,
                timeline_phase=timeline,
                estimated_cost_eur=estimated_cost,
                responsible_department="sustainability",
            ))

            # Add contractual assurance for supply chain impacts
            if imp.value_chain_stage and imp.value_chain_stage != "own_operations":
                self._measures.append(PreventionMeasure(
                    measure_name=f"Contractual assurance for: {imp.impact_name}",
                    description=f"Contractual cascade per Art. 7(2)(b) for {imp.value_chain_stage}",
                    measure_type=MeasureType.CONTRACTUAL,
                    related_impact_ids=[imp.impact_id],
                    csddd_article="art_7",
                    timeline_phase=TimelinePhase.SHORT_TERM,
                    estimated_cost_eur=round(base_cost * 0.3, 2),
                    responsible_department="procurement",
                ))

        outputs["measures_designed"] = len(self._measures)
        outputs["impacts_addressed"] = len(set(
            iid for m in self._measures for iid in m.related_impact_ids
        ))
        outputs["impacts_total"] = len(impacts)
        outputs["already_covered_by_existing"] = len(impacts) - outputs["impacts_addressed"]
        outputs["by_type"] = {
            mt.value: sum(1 for m in self._measures if m.measure_type == mt)
            for mt in MeasureType if sum(1 for m in self._measures if m.measure_type == mt) > 0
        }

        if not self._measures:
            warnings.append("No new measures designed -- all impacts may be covered by existing measures")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 MeasureDesign: %d measures for %d impacts",
            len(self._measures), outputs["impacts_addressed"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.MEASURE_DESIGN.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Resource Allocation
    # -------------------------------------------------------------------------

    async def _phase_resource_allocation(
        self, input_data: PreventionPlanningInput,
    ) -> PhaseResult:
        """Allocate budget and personnel to prevention measures."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        total_budget = input_data.total_budget_eur
        total_fte = input_data.available_fte
        total_cost = sum(m.estimated_cost_eur for m in self._measures)

        # Budget allocation by category
        by_category: Dict[str, float] = {}
        for m in self._measures:
            cat_key = m.measure_type.value
            by_category[cat_key] = by_category.get(cat_key, 0.0) + m.estimated_cost_eur

        # Budget allocation by timeline
        by_timeline: Dict[str, float] = {}
        for m in self._measures:
            tl_key = m.timeline_phase.value
            by_timeline[tl_key] = by_timeline.get(tl_key, 0.0) + m.estimated_cost_eur

        # Check budget sufficiency
        budget_gap = total_cost - total_budget
        utilization = round(
            (min(total_cost, total_budget) / total_budget) * 100, 1
        ) if total_budget > 0 else 0.0

        # FTE allocation (proportional to cost)
        allocated_fte = min(total_fte, len(self._measures) * 0.2)  # ~0.2 FTE per measure

        self._resource_plan = ResourcePlan(
            total_budget_eur=round(total_budget, 2),
            allocated_budget_eur=round(min(total_cost, total_budget), 2),
            remaining_budget_eur=round(max(0, total_budget - total_cost), 2),
            budget_utilization_pct=utilization,
            allocated_fte=round(allocated_fte, 1),
            by_category={k: round(v, 2) for k, v in by_category.items()},
            by_timeline={k: round(v, 2) for k, v in by_timeline.items()},
        )

        outputs["total_budget_eur"] = round(total_budget, 2)
        outputs["total_estimated_cost_eur"] = round(total_cost, 2)
        outputs["budget_gap_eur"] = round(max(0, budget_gap), 2)
        outputs["budget_utilization_pct"] = utilization
        outputs["allocated_fte"] = round(allocated_fte, 1)
        outputs["cost_by_category"] = self._resource_plan.by_category
        outputs["cost_by_timeline"] = self._resource_plan.by_timeline

        if budget_gap > 0:
            warnings.append(
                f"Budget shortfall of EUR {budget_gap:,.2f} -- measures may need reprioritization"
            )
        if total_fte < len(self._measures) * 0.1:
            warnings.append("Insufficient FTE capacity for planned measures")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 ResourceAllocation: %.0f EUR allocated, %.1f FTE",
            self._resource_plan.allocated_budget_eur, allocated_fte,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.RESOURCE_ALLOCATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Implementation Timeline
    # -------------------------------------------------------------------------

    async def _phase_implementation_timeline(
        self, input_data: PreventionPlanningInput,
    ) -> PhaseResult:
        """Build phased implementation plan with milestones."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Group measures by timeline phase
        phases: Dict[str, List[PreventionMeasure]] = {}
        for m in self._measures:
            phases.setdefault(m.timeline_phase.value, []).append(m)

        timeline_plan: List[Dict[str, Any]] = []

        phase_labels = {
            TimelinePhase.IMMEDIATE.value: {"label": "Immediate (0-3 months)", "months": 3},
            TimelinePhase.SHORT_TERM.value: {"label": "Short-term (3-12 months)", "months": 12},
            TimelinePhase.MEDIUM_TERM.value: {"label": "Medium-term (1-3 years)", "months": 36},
            TimelinePhase.LONG_TERM.value: {"label": "Long-term (3+ years)", "months": 60},
        }

        for phase_key in [tp.value for tp in TimelinePhase]:
            measures_in_phase = phases.get(phase_key, [])
            info = phase_labels.get(phase_key, {"label": phase_key, "months": 0})
            phase_cost = sum(m.estimated_cost_eur for m in measures_in_phase)

            timeline_plan.append({
                "phase": phase_key,
                "label": info["label"],
                "duration_months": info["months"],
                "measures_count": len(measures_in_phase),
                "estimated_cost_eur": round(phase_cost, 2),
                "measure_names": [m.measure_name for m in measures_in_phase],
                "milestones": [
                    f"Complete: {m.measure_name}" for m in measures_in_phase
                ],
            })

        outputs["timeline_phases"] = len([tp for tp in timeline_plan if tp["measures_count"] > 0])
        outputs["timeline_plan"] = timeline_plan
        outputs["total_implementation_months"] = max(
            (tp["duration_months"] for tp in timeline_plan if tp["measures_count"] > 0),
            default=0,
        )
        outputs["immediate_measures"] = len(phases.get("immediate", []))
        outputs["total_milestones"] = sum(len(tp["milestones"]) for tp in timeline_plan)

        if not phases.get("immediate") and any(
            i.severity_level in ("critical", "high") for i in input_data.prioritized_impacts
        ):
            warnings.append("No immediate actions planned despite critical/high severity impacts")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ImplementationTimeline: %d phases, %d milestones",
            outputs["timeline_phases"], outputs["total_milestones"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.IMPLEMENTATION_TIMELINE.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Effectiveness Metrics
    # -------------------------------------------------------------------------

    async def _phase_effectiveness_metrics(
        self, input_data: PreventionPlanningInput,
    ) -> PhaseResult:
        """Define KPIs to measure prevention plan effectiveness."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Auto-generate KPIs for each measure based on type
        kpi_templates: Dict[str, Dict[str, str]] = {
            MeasureType.PREVENTION.value: {
                "kpi_name": "Impact occurrence rate reduction",
                "kpi_target": "Reduce occurrence by 50% within timeline phase",
            },
            MeasureType.MITIGATION.value: {
                "kpi_name": "Severity score reduction",
                "kpi_target": "Reduce severity score by 30% within 12 months",
            },
            MeasureType.CORRECTIVE.value: {
                "kpi_name": "Remediation completion rate",
                "kpi_target": "100% corrective actions completed within 6 months",
            },
            MeasureType.CONTRACTUAL.value: {
                "kpi_name": "Supplier compliance rate",
                "kpi_target": "90% of suppliers sign contractual assurances within 12 months",
            },
            MeasureType.CAPACITY_BUILDING.value: {
                "kpi_name": "Training completion rate",
                "kpi_target": "100% of target staff trained within 6 months",
            },
            MeasureType.DISENGAGEMENT.value: {
                "kpi_name": "Disengagement completion",
                "kpi_target": "Complete disengagement and alternative sourcing within timeline",
            },
        }

        for measure in self._measures:
            template = kpi_templates.get(
                measure.measure_type.value,
                {"kpi_name": "Measure completion", "kpi_target": "100% completed on time"},
            )
            measure.kpi_name = template["kpi_name"]
            measure.kpi_target = template["kpi_target"]

        kpis_defined = sum(1 for m in self._measures if m.kpi_name)

        # Aggregate KPI categories
        kpi_categories: Dict[str, int] = {}
        for m in self._measures:
            if m.kpi_name:
                kpi_categories[m.kpi_name] = kpi_categories.get(m.kpi_name, 0) + 1

        outputs["kpis_defined"] = kpis_defined
        outputs["kpi_categories"] = kpi_categories
        outputs["measures_without_kpis"] = len(self._measures) - kpis_defined
        outputs["monitoring_frequency"] = "quarterly"
        outputs["review_cycle"] = "annual"

        if kpis_defined < len(self._measures):
            warnings.append(
                f"{len(self._measures) - kpis_defined} measures without KPIs -- effectiveness tracking at risk"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 EffectivenessMetrics: %d KPIs defined across %d categories",
            kpis_defined, len(kpi_categories),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.EFFECTIVENESS_METRICS.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: PreventionPlanningResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
