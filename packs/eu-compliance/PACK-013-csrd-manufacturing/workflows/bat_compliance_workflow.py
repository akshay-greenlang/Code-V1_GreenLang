# -*- coding: utf-8 -*-
"""
BAT Compliance Workflow
=======================

Four-phase workflow for assessing Best Available Techniques (BAT) compliance
under the Industrial Emissions Directive (IED) / BREF documents.

Phases:
    1. BREFIdentification - Identify applicable BREF documents and BAT-AELs
    2. PerformanceAssessment - Compare measured parameters against BAT-AELs
    3. GapAnalysis - Identify non-compliant parameters and risk exposure
    4. TransformationPlanning - Build remediation plan with timeline and cost

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

from pydantic import BaseModel, Field

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

class MeasuredParameter(BaseModel):
    """A measured environmental parameter against BAT-AEL."""
    parameter_id: str = Field(...)
    parameter_name: str = Field(default="", description="e.g. NOx, SOx, PM, VOC")
    measured_value: float = Field(..., description="Measured value")
    unit: str = Field(default="mg/Nm3")
    bat_ael_lower: float = Field(default=0.0, description="BAT-AEL lower bound")
    bat_ael_upper: float = Field(default=0.0, description="BAT-AEL upper bound")
    bref_reference: str = Field(default="", description="BREF document reference")
    facility_id: str = Field(default="")
    measurement_date: str = Field(default="")


class BREFDocument(BaseModel):
    """Reference to an applicable BREF document."""
    bref_id: str = Field(...)
    bref_name: str = Field(default="")
    sector: str = Field(default="")
    publication_year: int = Field(default=2020)
    applicable: bool = Field(default=True)


class BATComplianceInput(BaseModel):
    """Input for BAT compliance workflow."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2000, le=2100)
    facility_data: List[Dict[str, Any]] = Field(default_factory=list)
    measured_parameters: List[MeasuredParameter] = Field(default_factory=list)
    applicable_brefs: List[BREFDocument] = Field(default_factory=list)
    permit_conditions: List[Dict[str, Any]] = Field(default_factory=list)
    investment_budget_eur: float = Field(default=0.0, ge=0.0)
    skip_phases: List[str] = Field(default_factory=list)


class BATComplianceResult(WorkflowResult):
    """Result from the BAT compliance workflow."""
    compliance_status: str = Field(default="UNKNOWN", description="COMPLIANT, PARTIAL, NON_COMPLIANT")
    parameter_results: List[Dict[str, Any]] = Field(default_factory=list)
    transformation_plan: List[Dict[str, Any]] = Field(default_factory=list)
    penalty_risk: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Phase 1: BREF Identification
# ---------------------------------------------------------------------------

class BREFIdentificationPhase:
    """Identify applicable BREF documents and BAT-AELs for the facility."""

    PHASE_NAME = "bref_identification"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Map facilities to applicable BREF documents."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            brefs = config.get("applicable_brefs", [])
            facilities = config.get("facility_data", [])
            params = config.get("measured_parameters", [])

            applicable = [b for b in brefs if b.get("applicable", True)]
            outputs["total_brefs"] = len(brefs)
            outputs["applicable_brefs"] = len(applicable)
            outputs["bref_list"] = [
                {"bref_id": b.get("bref_id"), "bref_name": b.get("bref_name"),
                 "sector": b.get("sector")}
                for b in applicable
            ]
            outputs["facility_count"] = len(facilities)
            outputs["parameter_count"] = len(params)

            # Unique parameters tracked
            param_names = list({p.get("parameter_name", "") for p in params if p.get("parameter_name")})
            outputs["monitored_parameters"] = param_names

            if not applicable:
                warnings.append("No applicable BREF documents identified")
            if not params:
                errors.append("No measured parameters provided for assessment")

            status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED

        except Exception as exc:
            logger.error("BREFIdentification failed: %s", exc, exc_info=True)
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
#  Phase 2: Performance Assessment
# ---------------------------------------------------------------------------

class PerformanceAssessmentPhase:
    """Compare measured parameters against BAT-AEL ranges."""

    PHASE_NAME = "performance_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Evaluate each parameter against its BAT-AEL bounds."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            params = config.get("measured_parameters", [])

            results: List[Dict[str, Any]] = []
            compliant_count = 0
            non_compliant_count = 0
            marginal_count = 0

            for p in params:
                measured = p.get("measured_value", 0.0)
                lower = p.get("bat_ael_lower", 0.0)
                upper = p.get("bat_ael_upper", 0.0)

                if upper > 0 and measured <= upper:
                    if measured <= lower:
                        status_str = "BEST_PRACTICE"
                    else:
                        status_str = "COMPLIANT"
                    compliant_count += 1
                elif upper > 0:
                    exceedance_pct = round((measured - upper) / max(upper, 0.001) * 100, 2)
                    if exceedance_pct <= 10:
                        status_str = "MARGINAL"
                        marginal_count += 1
                    else:
                        status_str = "NON_COMPLIANT"
                        non_compliant_count += 1
                else:
                    status_str = "NO_AEL"
                    warnings.append(f"No BAT-AEL for {p.get('parameter_name', '')}")

                results.append({
                    "parameter_id": p.get("parameter_id", ""),
                    "parameter_name": p.get("parameter_name", ""),
                    "measured_value": measured,
                    "unit": p.get("unit", ""),
                    "bat_ael_lower": lower,
                    "bat_ael_upper": upper,
                    "status": status_str,
                    "exceedance_pct": round(
                        max(0, (measured - upper) / max(upper, 0.001) * 100), 2
                    ) if upper > 0 else 0.0,
                    "facility_id": p.get("facility_id", ""),
                    "bref_reference": p.get("bref_reference", ""),
                })

            outputs["parameter_results"] = results
            outputs["compliant_count"] = compliant_count
            outputs["non_compliant_count"] = non_compliant_count
            outputs["marginal_count"] = marginal_count
            outputs["total_assessed"] = len(results)

            total = len(results)
            outputs["compliance_rate_pct"] = round(
                compliant_count / max(total, 1) * 100, 2
            )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("PerformanceAssessment failed: %s", exc, exc_info=True)
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
#  Phase 3: Gap Analysis
# ---------------------------------------------------------------------------

class GapAnalysisPhase:
    """Identify non-compliant parameters and assess risk exposure."""

    PHASE_NAME = "gap_analysis"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Analyze compliance gaps and estimate penalty risk."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            perf = context.get_phase_output("performance_assessment")
            results = perf.get("parameter_results", [])
            non_compliant = perf.get("non_compliant_count", 0)
            marginal = perf.get("marginal_count", 0)

            gaps = [r for r in results if r.get("status") in ("NON_COMPLIANT", "MARGINAL")]

            # Penalty risk estimate (simplified)
            high_risk = [g for g in gaps if g.get("exceedance_pct", 0) > 50]
            medium_risk = [g for g in gaps if 10 < g.get("exceedance_pct", 0) <= 50]
            low_risk = [g for g in gaps if g.get("exceedance_pct", 0) <= 10 and g.get("status") == "MARGINAL"]

            # Estimated penalty per parameter (simplified EUR)
            estimated_penalty = (
                len(high_risk) * 100_000 + len(medium_risk) * 50_000 + len(low_risk) * 10_000
            )

            outputs["total_gaps"] = len(gaps)
            outputs["high_risk_gaps"] = len(high_risk)
            outputs["medium_risk_gaps"] = len(medium_risk)
            outputs["low_risk_gaps"] = len(low_risk)
            outputs["gap_details"] = gaps
            outputs["penalty_risk"] = {
                "estimated_total_eur": estimated_penalty,
                "high_risk_count": len(high_risk),
                "medium_risk_count": len(medium_risk),
                "low_risk_count": len(low_risk),
                "risk_level": "HIGH" if high_risk else ("MEDIUM" if medium_risk else "LOW"),
            }

            # Overall compliance status
            if non_compliant == 0 and marginal == 0:
                outputs["compliance_status"] = "COMPLIANT"
            elif non_compliant == 0:
                outputs["compliance_status"] = "PARTIAL"
            else:
                outputs["compliance_status"] = "NON_COMPLIANT"

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("GapAnalysis failed: %s", exc, exc_info=True)
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
#  Phase 4: Transformation Planning
# ---------------------------------------------------------------------------

class TransformationPlanningPhase:
    """Build a remediation plan with timeline and investment requirements."""

    PHASE_NAME = "transformation_planning"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Generate transformation plan for non-compliant parameters."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            gap_out = context.get_phase_output("gap_analysis")
            gaps = gap_out.get("gap_details", [])
            budget = config.get("investment_budget_eur", 0.0)

            actions: List[Dict[str, Any]] = []
            total_investment = 0.0

            for idx, gap in enumerate(gaps):
                exceedance = gap.get("exceedance_pct", 0.0)
                param_name = gap.get("parameter_name", "")

                # Estimate investment based on exceedance severity
                if exceedance > 50:
                    est_cost = 500_000.0
                    timeline_months = 24
                    priority = "HIGH"
                elif exceedance > 20:
                    est_cost = 200_000.0
                    timeline_months = 12
                    priority = "MEDIUM"
                else:
                    est_cost = 50_000.0
                    timeline_months = 6
                    priority = "LOW"

                total_investment += est_cost
                actions.append({
                    "action_id": f"ACT-{idx + 1:03d}",
                    "parameter_name": param_name,
                    "facility_id": gap.get("facility_id", ""),
                    "current_value": gap.get("measured_value", 0.0),
                    "target_value": gap.get("bat_ael_upper", 0.0),
                    "exceedance_pct": exceedance,
                    "priority": priority,
                    "estimated_cost_eur": est_cost,
                    "timeline_months": timeline_months,
                    "status": "PLANNED",
                })

            outputs["transformation_plan"] = actions
            outputs["total_investment_required"] = total_investment
            outputs["budget_available"] = budget
            outputs["budget_gap"] = max(0, total_investment - budget)
            outputs["actions_count"] = len(actions)

            high_priority = [a for a in actions if a["priority"] == "HIGH"]
            outputs["high_priority_count"] = len(high_priority)

            if total_investment > budget and budget > 0:
                warnings.append(
                    f"Investment required (EUR {total_investment:,.0f}) exceeds "
                    f"budget (EUR {budget:,.0f})"
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("TransformationPlanning failed: %s", exc, exc_info=True)
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

class BATComplianceWorkflow:
    """
    Four-phase BAT compliance assessment workflow.

    Evaluates facility performance against BAT-AEL ranges from BREF
    documents under the Industrial Emissions Directive, identifies
    compliance gaps, and generates a transformation plan.
    """

    WORKFLOW_NAME = "bat_compliance"

    PHASE_ORDER = [
        "bref_identification", "performance_assessment",
        "gap_analysis", "transformation_planning",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize BATComplianceWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "bref_identification": BREFIdentificationPhase(),
            "performance_assessment": PerformanceAssessmentPhase(),
            "gap_analysis": GapAnalysisPhase(),
            "transformation_planning": TransformationPlanningPhase(),
        }

    async def run(self, input_data: BATComplianceInput) -> BATComplianceResult:
        """Execute the complete 4-phase BAT compliance workflow."""
        started_at = _utcnow()
        logger.info("Starting BAT compliance workflow %s", self.workflow_id)
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
                    if phase_name == "bref_identification":
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

        return BATComplianceResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            compliance_status=summary.get("compliance_status", "UNKNOWN"),
            parameter_results=summary.get("parameter_results", []),
            transformation_plan=summary.get("transformation_plan", []),
            penalty_risk=summary.get("penalty_risk", {}),
        )

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build summary from phase outputs."""
        perf = context.get_phase_output("performance_assessment")
        gap = context.get_phase_output("gap_analysis")
        plan = context.get_phase_output("transformation_planning")
        return {
            "compliance_status": gap.get("compliance_status", "UNKNOWN"),
            "parameter_results": perf.get("parameter_results", []),
            "transformation_plan": plan.get("transformation_plan", []),
            "penalty_risk": gap.get("penalty_risk", {}),
        }

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        """Send progress notification via callback if configured."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
