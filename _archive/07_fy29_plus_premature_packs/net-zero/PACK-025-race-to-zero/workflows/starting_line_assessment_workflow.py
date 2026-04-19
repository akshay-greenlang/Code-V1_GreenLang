# -*- coding: utf-8 -*-
"""
Starting Line Assessment Workflow
======================================

4-phase workflow for comprehensive Starting Line Criteria compliance
assessment within PACK-025 Race to Zero Pack.  Evaluates the four
Starting Line pillars (Pledge, Plan, Proceed, Publish) with 20
sub-criteria, identifies gaps, generates remediation plans, and
produces compliance certification.

Phases:
    1. StartingLineCriteriaCheck   -- Evaluate all 20 sub-criteria across 4 pillars
    2. GapAnalysis                 -- Identify gaps with severity and effort estimates
    3. RemediationPlan             -- Generate prioritized remediation actions
    4. ComplianceCertification     -- Produce compliance report and certificate

Regulatory references:
    - Race to Zero Interpretation Guide (June 2022 update)
    - HLEG "Integrity Matters" Report (November 2022)
    - SBTi Corporate Net-Zero Standard V1.3 (2024)
    - GHG Protocol Corporate Standard (2015)
    - GHG Protocol Scope 3 Standard (2011)

Zero-hallucination: all criteria assessments use deterministic rule
evaluation against the 20 sub-criteria checklist.  No LLM calls in
the assessment path.

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
from greenlang.schemas.enums import ComplianceStatus

logger = logging.getLogger(__name__)

_MODULE_VERSION = "25.0.0"

ProgressCallback = Callable[[str, float, str], Coroutine[Any, Any, None]]

# =============================================================================
# HELPERS
# =============================================================================

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

class StartingLinePhase(str, Enum):
    CRITERIA_CHECK = "starting_line_criteria_check"
    GAP_ANALYSIS = "gap_analysis"
    REMEDIATION_PLAN = "remediation_plan"
    COMPLIANCE_CERTIFICATION = "compliance_certification"

class CriterionStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"

class GapSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RemediationPriority(str, Enum):
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"

class StartingLinePillar(str, Enum):
    PLEDGE = "pledge"
    PLAN = "plan"
    PROCEED = "proceed"
    PUBLISH = "publish"

# =============================================================================
# REFERENCE DATA (Zero-Hallucination Lookups)
# =============================================================================

# Starting Line 20 sub-criteria per Interpretation Guide (June 2022)
STARTING_LINE_CRITERIA: List[Dict[str, Any]] = [
    # PLEDGE pillar (SL-P1 to SL-P5)
    {
        "id": "SL-P1", "pillar": "pledge", "name": "Net-zero target",
        "description": "Commit to net zero by 2050 at latest, covering all scopes",
        "evidence_required": "Board-approved net-zero target statement",
        "effort_hours": 20, "priority": "immediate",
    },
    {
        "id": "SL-P2", "pillar": "pledge", "name": "Interim target",
        "description": "Set interim target for 2030 reflecting ~50% absolute reduction",
        "evidence_required": "Quantified 2030 interim target with methodology",
        "effort_hours": 40, "priority": "immediate",
    },
    {
        "id": "SL-P3", "pillar": "pledge", "name": "Science-based methodology",
        "description": "Target uses recognized science-based methodology (SBTi, IEA, IPCC)",
        "evidence_required": "Methodology documentation referencing SBTi/IEA/IPCC",
        "effort_hours": 30, "priority": "immediate",
    },
    {
        "id": "SL-P4", "pillar": "pledge", "name": "Fair share",
        "description": "Target represents a 'fair share' of global effort (equity consideration)",
        "evidence_required": "Fair share assessment with equity methodology",
        "effort_hours": 20, "priority": "short_term",
    },
    {
        "id": "SL-P5", "pillar": "pledge", "name": "Scope coverage",
        "description": "Covers Scope 1, 2, and material Scope 3 (or community-wide for cities)",
        "evidence_required": "Scope coverage documentation with Scope 3 screening",
        "effort_hours": 40, "priority": "immediate",
    },
    # PLAN pillar (SL-A1 to SL-A5)
    {
        "id": "SL-A1", "pillar": "plan", "name": "Action plan published",
        "description": "Climate action plan published within 12 months of joining",
        "evidence_required": "Published climate action plan document",
        "effort_hours": 80, "priority": "short_term",
    },
    {
        "id": "SL-A2", "pillar": "plan", "name": "Quantified actions",
        "description": "Plan includes specific, quantified decarbonization actions",
        "evidence_required": "Action list with tCO2e abatement estimates per action",
        "effort_hours": 60, "priority": "short_term",
    },
    {
        "id": "SL-A3", "pillar": "plan", "name": "Timeline and milestones",
        "description": "Actions have defined timelines and measurable milestones",
        "evidence_required": "Gantt chart or timeline with milestones per action",
        "effort_hours": 30, "priority": "short_term",
    },
    {
        "id": "SL-A4", "pillar": "plan", "name": "Resource allocation",
        "description": "Plan specifies resources (financial, human, technical) for implementation",
        "evidence_required": "Budget allocation and resource plan per action",
        "effort_hours": 40, "priority": "short_term",
    },
    {
        "id": "SL-A5", "pillar": "plan", "name": "Sector alignment",
        "description": "Actions aligned with relevant sector pathway(s)",
        "evidence_required": "Sector pathway reference with alignment analysis",
        "effort_hours": 30, "priority": "medium_term",
    },
    # PROCEED pillar (SL-R1 to SL-R5)
    {
        "id": "SL-R1", "pillar": "proceed", "name": "Immediate action",
        "description": "Demonstrable action taken (not just planned) within first year",
        "evidence_required": "Evidence of implemented actions with start dates",
        "effort_hours": 20, "priority": "immediate",
    },
    {
        "id": "SL-R2", "pillar": "proceed", "name": "Emission reductions",
        "description": "Evidence of actual emission reductions or genuine reduction trajectory",
        "evidence_required": "Year-over-year emission comparison showing reductions",
        "effort_hours": 40, "priority": "short_term",
    },
    {
        "id": "SL-R3", "pillar": "proceed", "name": "Investment commitment",
        "description": "Financial resources allocated to decarbonization actions",
        "evidence_required": "CapEx/OpEx allocation records for climate actions",
        "effort_hours": 20, "priority": "short_term",
    },
    {
        "id": "SL-R4", "pillar": "proceed", "name": "Governance integration",
        "description": "Climate targets integrated into corporate/organizational governance",
        "evidence_required": "Board/leadership mandate, KPIs, incentive structure",
        "effort_hours": 30, "priority": "medium_term",
    },
    {
        "id": "SL-R5", "pillar": "proceed", "name": "No contradictory action",
        "description": "No actions contradicting climate commitment (e.g., new fossil fuel)",
        "evidence_required": "Fossil fuel policy, investment screening documentation",
        "effort_hours": 20, "priority": "immediate",
    },
    # PUBLISH pillar (SL-D1 to SL-D5)
    {
        "id": "SL-D1", "pillar": "publish", "name": "Annual reporting",
        "description": "Annual progress reported through partner initiative channels",
        "evidence_required": "Annual report submission confirmation from partner",
        "effort_hours": 40, "priority": "short_term",
    },
    {
        "id": "SL-D2", "pillar": "publish", "name": "Emissions disclosure",
        "description": "GHG emissions disclosed publicly (Scope 1, 2, material Scope 3)",
        "evidence_required": "Published GHG inventory report or CDP response",
        "effort_hours": 40, "priority": "short_term",
    },
    {
        "id": "SL-D3", "pillar": "publish", "name": "Target progress",
        "description": "Progress against targets reported with quantitative metrics",
        "evidence_required": "Quantitative progress metrics against interim target",
        "effort_hours": 20, "priority": "short_term",
    },
    {
        "id": "SL-D4", "pillar": "publish", "name": "Plan updates",
        "description": "Action plan updated and re-published annually",
        "evidence_required": "Updated action plan with version control",
        "effort_hours": 30, "priority": "medium_term",
    },
    {
        "id": "SL-D5", "pillar": "publish", "name": "Transparency",
        "description": "Methodology, assumptions, and limitations transparently documented",
        "evidence_required": "Methodology notes in published reports",
        "effort_hours": 20, "priority": "medium_term",
    },
]

# Pillar compliance thresholds
PILLAR_PASS_THRESHOLD = 0.80  # 80% of sub-criteria must pass within pillar
OVERALL_PASS_THRESHOLD = 0.75  # 75% of all criteria must pass for overall compliance

# Phase dependencies DAG
PHASE_DEPENDENCIES: Dict[StartingLinePhase, List[StartingLinePhase]] = {
    StartingLinePhase.CRITERIA_CHECK: [],
    StartingLinePhase.GAP_ANALYSIS: [StartingLinePhase.CRITERIA_CHECK],
    StartingLinePhase.REMEDIATION_PLAN: [StartingLinePhase.GAP_ANALYSIS],
    StartingLinePhase.COMPLIANCE_CERTIFICATION: [StartingLinePhase.REMEDIATION_PLAN],
}

PHASE_EXECUTION_ORDER: List[StartingLinePhase] = [
    StartingLinePhase.CRITERIA_CHECK,
    StartingLinePhase.GAP_ANALYSIS,
    StartingLinePhase.REMEDIATION_PLAN,
    StartingLinePhase.COMPLIANCE_CERTIFICATION,
]

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    phase: StartingLinePhase = Field(...)
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class CriterionResult(BaseModel):
    criterion_id: str = Field(default="")
    pillar: str = Field(default="")
    name: str = Field(default="")
    status: CriterionStatus = Field(default=CriterionStatus.FAIL)
    evidence_provided: bool = Field(default=False)
    evidence_reference: str = Field(default="")
    notes: str = Field(default="")

class GapItem(BaseModel):
    gap_id: str = Field(default="")
    criterion_id: str = Field(default="")
    pillar: str = Field(default="")
    description: str = Field(default="")
    severity: GapSeverity = Field(default=GapSeverity.MEDIUM)
    effort_hours: int = Field(default=0)
    evidence_needed: str = Field(default="")

class RemediationAction(BaseModel):
    action_id: str = Field(default="")
    gap_id: str = Field(default="")
    criterion_id: str = Field(default="")
    action: str = Field(default="")
    priority: RemediationPriority = Field(default=RemediationPriority.SHORT_TERM)
    effort_hours: int = Field(default=0)
    deadline_months: int = Field(default=3)
    responsible_role: str = Field(default="")
    dependencies: List[str] = Field(default_factory=list)

class ComplianceCertificate(BaseModel):
    certificate_id: str = Field(default="")
    org_name: str = Field(default="")
    assessment_date: str = Field(default="")
    overall_status: ComplianceStatus = Field(default=ComplianceStatus.NON_COMPLIANT)
    pillar_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    criteria_passed: int = Field(default=0)
    criteria_total: int = Field(default=20)
    compliance_score: float = Field(default=0.0)
    gaps_remaining: int = Field(default=0)
    remediation_timeline_months: int = Field(default=0)
    next_review_date: str = Field(default="")

class StartingLineConfig(BaseModel):
    pack_id: str = Field(default="PACK-025")
    pack_version: str = Field(default="1.0.0")
    org_name: str = Field(default="")
    actor_type: str = Field(default="corporate")
    reporting_year: int = Field(default=2025, ge=2015, le=2050)
    base_year: int = Field(default=2019, ge=2015, le=2050)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    # Evidence flags for each criterion
    has_net_zero_target: bool = Field(default=False)
    has_interim_target: bool = Field(default=False)
    has_science_based_method: bool = Field(default=False)
    has_fair_share_assessment: bool = Field(default=False)
    has_scope_coverage: bool = Field(default=False)
    has_action_plan: bool = Field(default=False)
    has_quantified_actions: bool = Field(default=False)
    has_timeline: bool = Field(default=False)
    has_resource_allocation: bool = Field(default=False)
    has_sector_alignment: bool = Field(default=False)
    has_immediate_action: bool = Field(default=False)
    has_emission_reductions: bool = Field(default=False)
    has_investment_commitment: bool = Field(default=False)
    has_governance_integration: bool = Field(default=False)
    has_no_contradictory_action: bool = Field(default=True)
    has_annual_reporting: bool = Field(default=False)
    has_emissions_disclosure: bool = Field(default=False)
    has_target_progress: bool = Field(default=False)
    has_plan_updates: bool = Field(default=False)
    has_transparency: bool = Field(default=False)
    target_reduction_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    join_date: str = Field(default="")
    enable_provenance: bool = Field(default=True)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class StartingLineResult(BaseModel):
    execution_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-025")
    workflow_name: str = Field(default="starting_line_assessment")
    org_name: str = Field(default="")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    total_duration_ms: float = Field(default=0.0)
    phases_completed: List[str] = Field(default_factory=list)
    phases_skipped: List[str] = Field(default_factory=list)
    phase_results: Dict[str, PhaseResult] = Field(default_factory=dict)
    criteria_results: List[CriterionResult] = Field(default_factory=list)
    gaps: List[GapItem] = Field(default_factory=list)
    remediation_actions: List[RemediationAction] = Field(default_factory=list)
    certificate: Optional[ComplianceCertificate] = Field(None)
    overall_compliance: ComplianceStatus = Field(default=ComplianceStatus.NON_COMPLIANT)
    compliance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    total_records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class StartingLineAssessmentWorkflow:
    """
    4-phase Starting Line assessment workflow for PACK-025 Race to Zero Pack.

    Evaluates compliance with the four Starting Line Criteria (Pledge,
    Plan, Proceed, Publish) with 20 sub-criteria from the Interpretation
    Guide.  Produces gap analysis, remediation plan, and compliance
    certification.

    Engines used:
        - starting_line_engine (criteria check)
        - action_plan_engine (plan quality validation)

    Attributes:
        config: Workflow configuration.
    """

    def __init__(
        self,
        config: Optional[StartingLineConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.config = config or StartingLineConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._results: Dict[str, StartingLineResult] = {}
        self._cancelled: Set[str] = set()
        self._progress_callback = progress_callback

    async def execute(
        self,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> StartingLineResult:
        """Execute the 4-phase Starting Line assessment workflow."""
        input_data = input_data or {}

        result = StartingLineResult(
            org_name=self.config.org_name,
            status=WorkflowStatus.RUNNING,
            started_at=utcnow(),
        )
        self._results[result.execution_id] = result

        start_time = time.monotonic()
        phases = PHASE_EXECUTION_ORDER
        total_phases = len(phases)

        self.logger.info(
            "Starting Starting Line assessment: execution_id=%s, org=%s",
            result.execution_id, self.config.org_name,
        )

        shared_context: Dict[str, Any] = dict(input_data)
        shared_context["org_name"] = self.config.org_name
        shared_context["actor_type"] = self.config.actor_type
        shared_context["reporting_year"] = self.config.reporting_year

        try:
            for phase_idx, phase in enumerate(phases):
                if result.execution_id in self._cancelled:
                    result.status = WorkflowStatus.CANCELLED
                    result.errors.append("Assessment cancelled by user")
                    break

                if not self._dependencies_met(phase, result):
                    phase_result = PhaseResult(
                        phase=phase, status=PhaseStatus.FAILED,
                        errors=["Dependencies not met"],
                    )
                    result.phase_results[phase.value] = phase_result
                    result.status = WorkflowStatus.FAILED
                    result.errors.append(f"Phase '{phase.value}' dependencies not met")
                    break

                progress_pct = (phase_idx / total_phases) * 100.0
                if self._progress_callback:
                    await self._progress_callback(
                        phase.value, progress_pct, f"Executing {phase.value}"
                    )

                phase_result = await self._execute_phase(phase, shared_context)
                result.phase_results[phase.value] = phase_result

                if phase_result.status == PhaseStatus.FAILED:
                    result.status = WorkflowStatus.PARTIAL
                    result.errors.append(f"Phase '{phase.value}' failed")
                    # Continue to remaining phases even if one fails

                result.phases_completed.append(phase.value)
                result.total_records_processed += phase_result.records_processed
                shared_context[phase.value] = phase_result.outputs

            if result.status == WorkflowStatus.RUNNING:
                result.status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Starting Line assessment failed: %s", exc, exc_info=True)
            result.status = WorkflowStatus.FAILED
            result.errors.append(str(exc))

        finally:
            result.completed_at = utcnow()
            result.total_duration_ms = (time.monotonic() - start_time) * 1000
            result.criteria_results = self._extract_criteria(shared_context)
            result.gaps = self._extract_gaps(shared_context)
            result.remediation_actions = self._extract_remediations(shared_context)
            result.certificate = self._extract_certificate(shared_context)
            result.overall_compliance = ComplianceStatus(
                shared_context.get("compliance_certification", {}).get(
                    "overall_status", "non_compliant"
                )
            )
            result.compliance_score = shared_context.get(
                "compliance_certification", {}
            ).get("compliance_score", 0.0)
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    result.model_dump_json(exclude={"provenance_hash"})
                )

        self.logger.info(
            "Starting Line assessment %s: status=%s, score=%.1f%%",
            result.execution_id, result.status.value, result.compliance_score,
        )
        return result

    def cancel(self, execution_id: str) -> Dict[str, Any]:
        self._cancelled.add(execution_id)
        return {"cancelled": True, "execution_id": execution_id}

    def get_result(self, execution_id: str) -> Optional[StartingLineResult]:
        return self._results.get(execution_id)

    # -------------------------------------------------------------------------
    # Phase Execution
    # -------------------------------------------------------------------------

    async def _execute_phase(
        self, phase: StartingLinePhase, context: Dict[str, Any]
    ) -> PhaseResult:
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

    def _get_phase_handler(self, phase: StartingLinePhase):
        return {
            StartingLinePhase.CRITERIA_CHECK: self._handle_criteria_check,
            StartingLinePhase.GAP_ANALYSIS: self._handle_gap_analysis,
            StartingLinePhase.REMEDIATION_PLAN: self._handle_remediation_plan,
            StartingLinePhase.COMPLIANCE_CERTIFICATION: self._handle_compliance_certification,
        }[phase]

    # -------------------------------------------------------------------------
    # Phase 1: Starting Line Criteria Check
    # -------------------------------------------------------------------------

    async def _handle_criteria_check(self, ctx: Dict[str, Any]) -> tuple:
        """Evaluate all 20 sub-criteria across the 4 Starting Line pillars."""
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        # Map config evidence flags to criterion IDs
        evidence_map: Dict[str, bool] = {
            "SL-P1": self.config.has_net_zero_target,
            "SL-P2": self.config.has_interim_target,
            "SL-P3": self.config.has_science_based_method,
            "SL-P4": self.config.has_fair_share_assessment,
            "SL-P5": self.config.has_scope_coverage,
            "SL-A1": self.config.has_action_plan,
            "SL-A2": self.config.has_quantified_actions,
            "SL-A3": self.config.has_timeline,
            "SL-A4": self.config.has_resource_allocation,
            "SL-A5": self.config.has_sector_alignment,
            "SL-R1": self.config.has_immediate_action,
            "SL-R2": self.config.has_emission_reductions,
            "SL-R3": self.config.has_investment_commitment,
            "SL-R4": self.config.has_governance_integration,
            "SL-R5": self.config.has_no_contradictory_action,
            "SL-D1": self.config.has_annual_reporting,
            "SL-D2": self.config.has_emissions_disclosure,
            "SL-D3": self.config.has_target_progress,
            "SL-D4": self.config.has_plan_updates,
            "SL-D5": self.config.has_transparency,
        }

        criteria_results: List[Dict[str, Any]] = []
        pillar_scores: Dict[str, Dict[str, int]] = {
            "pledge": {"passed": 0, "total": 0},
            "plan": {"passed": 0, "total": 0},
            "proceed": {"passed": 0, "total": 0},
            "publish": {"passed": 0, "total": 0},
        }

        total_passed = 0
        total_criteria = len(STARTING_LINE_CRITERIA)

        for criterion in STARTING_LINE_CRITERIA:
            cid = criterion["id"]
            pillar = criterion["pillar"]
            has_evidence = evidence_map.get(cid, False)

            # Additional validation for specific criteria
            status = CriterionStatus.PASS.value if has_evidence else CriterionStatus.FAIL.value

            # Special validation for SL-P2 (interim target ambition)
            if cid == "SL-P2" and has_evidence:
                if self.config.target_reduction_pct < 42.0:
                    status = CriterionStatus.PARTIAL.value
                    warnings.append(
                        f"Interim target ({self.config.target_reduction_pct:.0f}%) "
                        "below IPCC minimum (42%)"
                    )

            # Special validation for SL-P5 (scope coverage)
            if cid == "SL-P5" and has_evidence:
                if self.config.scope3_coverage_pct < 67.0:
                    status = CriterionStatus.PARTIAL.value
                    warnings.append(
                        f"Scope 3 coverage ({self.config.scope3_coverage_pct:.1f}%) "
                        "below 67% threshold"
                    )

            # Special validation for SL-R2 (emission reductions)
            if cid == "SL-R2" and has_evidence:
                total = self.config.scope1_tco2e + self.config.scope2_tco2e + self.config.scope3_tco2e
                if total <= 0:
                    status = CriterionStatus.PARTIAL.value
                    warnings.append("Emissions data needed to verify reduction trajectory")

            criterion_result = {
                "criterion_id": cid,
                "pillar": pillar,
                "name": criterion["name"],
                "description": criterion["description"],
                "status": status,
                "evidence_provided": has_evidence,
                "evidence_required": criterion["evidence_required"],
            }
            criteria_results.append(criterion_result)

            pillar_scores[pillar]["total"] += 1
            if status in (CriterionStatus.PASS.value, CriterionStatus.PARTIAL.value):
                pillar_scores[pillar]["passed"] += 1
            if status == CriterionStatus.PASS.value:
                total_passed += 1

            records += 1

        # Calculate pillar compliance
        pillar_compliance: Dict[str, Dict[str, Any]] = {}
        for pillar, scores in pillar_scores.items():
            pct = (scores["passed"] / max(scores["total"], 1)) * 100.0
            compliant = pct >= (PILLAR_PASS_THRESHOLD * 100.0)
            pillar_compliance[pillar] = {
                "passed": scores["passed"],
                "total": scores["total"],
                "compliance_pct": round(pct, 1),
                "compliant": compliant,
            }

        overall_pct = (total_passed / max(total_criteria, 1)) * 100.0

        outputs["criteria_results"] = criteria_results
        outputs["pillar_compliance"] = pillar_compliance
        outputs["total_passed"] = total_passed
        outputs["total_criteria"] = total_criteria
        outputs["overall_compliance_pct"] = round(overall_pct, 1)
        outputs["pillars_compliant"] = sum(
            1 for p in pillar_compliance.values() if p["compliant"]
        )
        outputs["pillars_total"] = 4

        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Phase 2: Gap Analysis
    # -------------------------------------------------------------------------

    async def _handle_gap_analysis(self, ctx: Dict[str, Any]) -> tuple:
        """Identify gaps with severity and effort estimates."""
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        criteria_data = ctx.get("starting_line_criteria_check", {})
        criteria_results = criteria_data.get("criteria_results", [])

        gaps: List[Dict[str, Any]] = []
        total_effort_hours = 0

        for cr in criteria_results:
            if cr["status"] in (CriterionStatus.FAIL.value, CriterionStatus.PARTIAL.value):
                cid = cr["criterion_id"]
                pillar = cr["pillar"]

                # Find the reference criterion for effort estimate
                ref_criterion = next(
                    (c for c in STARTING_LINE_CRITERIA if c["id"] == cid), None
                )
                effort = ref_criterion.get("effort_hours", 20) if ref_criterion else 20

                # Determine severity based on pillar and status
                if pillar == "pledge":
                    severity = GapSeverity.CRITICAL.value
                elif pillar == "proceed" and cid == "SL-R5":
                    severity = GapSeverity.CRITICAL.value
                elif cr["status"] == CriterionStatus.FAIL.value:
                    severity = GapSeverity.HIGH.value
                else:
                    severity = GapSeverity.MEDIUM.value

                gap = {
                    "gap_id": _new_uuid()[:12],
                    "criterion_id": cid,
                    "pillar": pillar,
                    "name": cr["name"],
                    "description": f"Gap in {cr['name']}: {cr.get('description', '')}",
                    "severity": severity,
                    "effort_hours": effort,
                    "evidence_needed": cr.get("evidence_required", ""),
                    "status": cr["status"],
                }
                gaps.append(gap)
                total_effort_hours += effort
                records += 1

        # Sort gaps by severity (critical first)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps.sort(key=lambda g: severity_order.get(g["severity"], 99))

        # Summary statistics
        severity_counts = {}
        for gap in gaps:
            sev = gap["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        pillar_gaps: Dict[str, int] = {}
        for gap in gaps:
            p = gap["pillar"]
            pillar_gaps[p] = pillar_gaps.get(p, 0) + 1

        outputs["gaps"] = gaps
        outputs["gaps_count"] = len(gaps)
        outputs["total_effort_hours"] = total_effort_hours
        outputs["estimated_remediation_weeks"] = round(total_effort_hours / 40.0, 1)
        outputs["severity_distribution"] = severity_counts
        outputs["pillar_gap_counts"] = pillar_gaps
        outputs["critical_gaps_count"] = severity_counts.get("critical", 0)
        outputs["has_blocking_gaps"] = severity_counts.get("critical", 0) > 0

        if severity_counts.get("critical", 0) > 0:
            warnings.append(
                f"{severity_counts['critical']} critical gap(s) found -- "
                "immediate remediation required"
            )

        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Phase 3: Remediation Plan
    # -------------------------------------------------------------------------

    async def _handle_remediation_plan(self, ctx: Dict[str, Any]) -> tuple:
        """Generate prioritized remediation actions for identified gaps."""
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        gap_data = ctx.get("gap_analysis", {})
        gaps = gap_data.get("gaps", [])

        remediation_actions: List[Dict[str, Any]] = []

        # Priority mapping from severity
        severity_to_priority = {
            "critical": RemediationPriority.IMMEDIATE.value,
            "high": RemediationPriority.SHORT_TERM.value,
            "medium": RemediationPriority.MEDIUM_TERM.value,
            "low": RemediationPriority.LONG_TERM.value,
        }

        # Deadline mapping (months)
        priority_to_deadline = {
            "immediate": 1,
            "short_term": 3,
            "medium_term": 6,
            "long_term": 12,
        }

        # Role mapping by pillar
        pillar_to_role = {
            "pledge": "Chief Sustainability Officer",
            "plan": "Climate Action Lead",
            "proceed": "Operations Manager",
            "publish": "Reporting Manager",
        }

        for gap in gaps:
            priority = severity_to_priority.get(gap["severity"], "medium_term")
            deadline = priority_to_deadline.get(priority, 6)

            # Generate remediation action text
            ref_criterion = next(
                (c for c in STARTING_LINE_CRITERIA if c["id"] == gap["criterion_id"]),
                None,
            )

            action_text = f"Address gap in '{gap['name']}': "
            if gap["status"] == CriterionStatus.FAIL.value:
                action_text += f"Develop and provide evidence for {gap.get('evidence_needed', 'required documentation')}"
            else:
                action_text += f"Strengthen existing evidence to fully meet criterion requirements"

            action = {
                "action_id": _new_uuid()[:12],
                "gap_id": gap["gap_id"],
                "criterion_id": gap["criterion_id"],
                "pillar": gap["pillar"],
                "action": action_text,
                "priority": priority,
                "effort_hours": gap["effort_hours"],
                "deadline_months": deadline,
                "responsible_role": pillar_to_role.get(gap["pillar"], "Sustainability Team"),
                "dependencies": [],
            }
            remediation_actions.append(action)
            records += 1

        # Sort by priority
        priority_order = {"immediate": 0, "short_term": 1, "medium_term": 2, "long_term": 3}
        remediation_actions.sort(key=lambda a: priority_order.get(a["priority"], 99))

        # Add dependencies: plan actions depend on pledge actions
        pledge_actions = [a for a in remediation_actions if a["pillar"] == "pledge"]
        for action in remediation_actions:
            if action["pillar"] == "plan" and pledge_actions:
                action["dependencies"] = [pledge_actions[0]["action_id"]]

        # Timeline summary
        timeline = {
            "immediate_actions": sum(1 for a in remediation_actions if a["priority"] == "immediate"),
            "short_term_actions": sum(1 for a in remediation_actions if a["priority"] == "short_term"),
            "medium_term_actions": sum(1 for a in remediation_actions if a["priority"] == "medium_term"),
            "long_term_actions": sum(1 for a in remediation_actions if a["priority"] == "long_term"),
            "total_effort_hours": sum(a["effort_hours"] for a in remediation_actions),
            "max_deadline_months": max(
                (a["deadline_months"] for a in remediation_actions), default=0
            ),
        }

        outputs["remediation_actions"] = remediation_actions
        outputs["actions_count"] = len(remediation_actions)
        outputs["timeline"] = timeline
        outputs["total_effort_hours"] = timeline["total_effort_hours"]
        outputs["estimated_completion_months"] = timeline["max_deadline_months"]

        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Phase 4: Compliance Certification
    # -------------------------------------------------------------------------

    async def _handle_compliance_certification(self, ctx: Dict[str, Any]) -> tuple:
        """Produce compliance report and certificate."""
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        criteria_data = ctx.get("starting_line_criteria_check", {})
        gap_data = ctx.get("gap_analysis", {})
        remediation_data = ctx.get("remediation_plan", {})

        total_passed = criteria_data.get("total_passed", 0)
        total_criteria = criteria_data.get("total_criteria", 20)
        pillar_compliance = criteria_data.get("pillar_compliance", {})
        gaps_count = gap_data.get("gaps_count", 0)
        critical_gaps = gap_data.get("critical_gaps_count", 0)

        compliance_score = (total_passed / max(total_criteria, 1)) * 100.0

        # Determine overall status
        all_pillars_compliant = all(
            p.get("compliant", False) for p in pillar_compliance.values()
        )

        if compliance_score >= 100.0 and gaps_count == 0:
            overall_status = ComplianceStatus.COMPLIANT.value
        elif compliance_score >= (OVERALL_PASS_THRESHOLD * 100.0) and critical_gaps == 0:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT.value
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT.value

        # Generate certificate
        from datetime import timedelta
        assessment_date = utcnow().strftime("%Y-%m-%d")
        remediation_months = remediation_data.get("estimated_completion_months", 0)
        next_review = (
            utcnow() + timedelta(days=remediation_months * 30)
        ).strftime("%Y-%m-%d") if remediation_months > 0 else (
            utcnow() + timedelta(days=365)
        ).strftime("%Y-%m-%d")

        certificate = {
            "certificate_id": f"SL-{self.config.reporting_year}-{_new_uuid()[:8].upper()}",
            "org_name": self.config.org_name,
            "assessment_date": assessment_date,
            "overall_status": overall_status,
            "pillar_results": pillar_compliance,
            "criteria_passed": total_passed,
            "criteria_total": total_criteria,
            "compliance_score": round(compliance_score, 1),
            "gaps_remaining": gaps_count,
            "remediation_timeline_months": remediation_months,
            "next_review_date": next_review,
        }

        outputs.update(certificate)
        outputs["all_pillars_compliant"] = all_pillars_compliant
        outputs["recommendation"] = (
            "Full compliance achieved. Ready for Race to Zero participation."
            if overall_status == ComplianceStatus.COMPLIANT.value
            else f"Remediation required for {gaps_count} gap(s). "
                 f"Estimated completion: {remediation_months} months."
        )

        records = 1
        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _dependencies_met(self, phase: StartingLinePhase, result: StartingLineResult) -> bool:
        deps = PHASE_DEPENDENCIES.get(phase, [])
        for dep in deps:
            dep_result = result.phase_results.get(dep.value)
            if not dep_result or dep_result.status not in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED):
                return False
        return True

    def _extract_criteria(self, ctx: Dict[str, Any]) -> List[CriterionResult]:
        data = ctx.get("starting_line_criteria_check", {}).get("criteria_results", [])
        return [
            CriterionResult(
                criterion_id=cr.get("criterion_id", ""),
                pillar=cr.get("pillar", ""),
                name=cr.get("name", ""),
                status=CriterionStatus(cr.get("status", "fail")),
                evidence_provided=cr.get("evidence_provided", False),
            )
            for cr in data
        ]

    def _extract_gaps(self, ctx: Dict[str, Any]) -> List[GapItem]:
        data = ctx.get("gap_analysis", {}).get("gaps", [])
        return [
            GapItem(
                gap_id=g.get("gap_id", ""),
                criterion_id=g.get("criterion_id", ""),
                pillar=g.get("pillar", ""),
                description=g.get("description", ""),
                severity=GapSeverity(g.get("severity", "medium")),
                effort_hours=g.get("effort_hours", 0),
                evidence_needed=g.get("evidence_needed", ""),
            )
            for g in data
        ]

    def _extract_remediations(self, ctx: Dict[str, Any]) -> List[RemediationAction]:
        data = ctx.get("remediation_plan", {}).get("remediation_actions", [])
        return [
            RemediationAction(
                action_id=a.get("action_id", ""),
                gap_id=a.get("gap_id", ""),
                criterion_id=a.get("criterion_id", ""),
                action=a.get("action", ""),
                priority=RemediationPriority(a.get("priority", "short_term")),
                effort_hours=a.get("effort_hours", 0),
                deadline_months=a.get("deadline_months", 3),
                responsible_role=a.get("responsible_role", ""),
                dependencies=a.get("dependencies", []),
            )
            for a in data
        ]

    def _extract_certificate(self, ctx: Dict[str, Any]) -> Optional[ComplianceCertificate]:
        data = ctx.get("compliance_certification", {})
        if not data:
            return None
        return ComplianceCertificate(
            certificate_id=data.get("certificate_id", ""),
            org_name=data.get("org_name", ""),
            assessment_date=data.get("assessment_date", ""),
            overall_status=ComplianceStatus(data.get("overall_status", "non_compliant")),
            pillar_results=data.get("pillar_results", {}),
            criteria_passed=data.get("criteria_passed", 0),
            criteria_total=data.get("criteria_total", 20),
            compliance_score=data.get("compliance_score", 0.0),
            gaps_remaining=data.get("gaps_remaining", 0),
            remediation_timeline_months=data.get("remediation_timeline_months", 0),
            next_review_date=data.get("next_review_date", ""),
        )
