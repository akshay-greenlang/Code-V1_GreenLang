# -*- coding: utf-8 -*-
"""
Regulatory Compliance Workflow
===============================

Three-phase workflow for multi-regulation compliance assessment across
CSRD, IED, ETS, CBAM, and other EU environmental regulations applicable
to manufacturing organizations.

Phases:
    1. RegulationMapping - Map applicable regulations to the organization
    2. ComplianceAssessment - Evaluate compliance status per regulation
    3. ActionPlanning - Generate gap remediation timeline and action items

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

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
    execution_timestamp: datetime = Field(default_factory=utcnow)
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

class RegulationRecord(BaseModel):
    """An applicable regulation with its requirements."""
    regulation_id: str = Field(...)
    regulation_name: str = Field(default="")
    category: str = Field(default="environmental", description="environmental, social, governance")
    jurisdiction: str = Field(default="EU")
    requirements: List[str] = Field(default_factory=list)
    deadline: str = Field(default="", description="Compliance deadline YYYY-MM-DD")
    penalty_range_eur: str = Field(default="", description="e.g. 10K-1M")
    current_status: str = Field(default="unknown", description="compliant, partial, non_compliant, unknown")

class RegulatoryComplianceInput(BaseModel):
    """Input for regulatory compliance workflow."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2000, le=2100)
    company_data: Dict[str, Any] = Field(default_factory=dict)
    facility_data: List[Dict[str, Any]] = Field(default_factory=list)
    applicable_regulations: List[RegulationRecord] = Field(default_factory=list)
    existing_certifications: List[str] = Field(default_factory=list)
    skip_phases: List[str] = Field(default_factory=list)

class RegulatoryComplianceResult(WorkflowResult):
    """Result from the regulatory compliance workflow."""
    regulation_status: Dict[str, str] = Field(default_factory=dict)
    gaps: List[Dict[str, Any]] = Field(default_factory=list)
    action_items: List[Dict[str, Any]] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)

# ---------------------------------------------------------------------------
#  Phase 1: Regulation Mapping
# ---------------------------------------------------------------------------

class RegulationMappingPhase:
    """Map applicable regulations to the organization's operations."""

    PHASE_NAME = "regulation_mapping"

    # Default manufacturing-relevant EU regulations
    DEFAULT_REGULATIONS = [
        {"id": "CSRD", "name": "Corporate Sustainability Reporting Directive", "category": "reporting"},
        {"id": "EU_ETS", "name": "EU Emissions Trading System", "category": "emissions"},
        {"id": "IED", "name": "Industrial Emissions Directive", "category": "emissions"},
        {"id": "CBAM", "name": "Carbon Border Adjustment Mechanism", "category": "trade"},
        {"id": "EU_TAXONOMY", "name": "EU Taxonomy Regulation", "category": "finance"},
        {"id": "REACH", "name": "Registration, Evaluation of Chemicals", "category": "chemicals"},
        {"id": "WFD", "name": "Waste Framework Directive", "category": "waste"},
        {"id": "EED", "name": "Energy Efficiency Directive", "category": "energy"},
    ]

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Identify and map all applicable regulations."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            explicit_regs = config.get("applicable_regulations", [])
            facilities = config.get("facility_data", [])

            # Use explicit regulations if provided, otherwise use defaults
            if explicit_regs:
                regulations = explicit_regs
            else:
                regulations = [
                    {"regulation_id": r["id"], "regulation_name": r["name"],
                     "category": r["category"], "current_status": "unknown"}
                    for r in self.DEFAULT_REGULATIONS
                ]
                warnings.append("Using default regulation set; provide explicit list for accuracy")

            outputs["total_regulations"] = len(regulations)
            outputs["regulations"] = regulations

            # Group by category
            by_category: Dict[str, int] = {}
            for r in regulations:
                cat = r.get("category", "other")
                by_category[cat] = by_category.get(cat, 0) + 1
            outputs["by_category"] = by_category

            outputs["facility_count"] = len(facilities)
            outputs["certifications"] = config.get("existing_certifications", [])

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("RegulationMapping failed: %s", exc, exc_info=True)
            errors.append(str(exc))
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 2: Compliance Assessment
# ---------------------------------------------------------------------------

class ComplianceAssessmentPhase:
    """Evaluate compliance status per regulation."""

    PHASE_NAME = "compliance_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Assess current compliance level for each mapped regulation."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            mapping = context.get_phase_output("regulation_mapping")
            regulations = mapping.get("regulations", [])
            certifications = mapping.get("certifications", [])

            regulation_status: Dict[str, str] = {}
            gaps: List[Dict[str, Any]] = []
            compliant_count = 0
            partial_count = 0
            non_compliant_count = 0

            for reg in regulations:
                reg_id = reg.get("regulation_id", reg.get("id", ""))
                reg_name = reg.get("regulation_name", reg.get("name", ""))
                current = reg.get("current_status", "unknown")
                requirements = reg.get("requirements", [])

                # Check if any certification covers this regulation
                cert_coverage = any(
                    cert.upper() in reg_id.upper() or reg_id.upper() in cert.upper()
                    for cert in certifications
                )
                if cert_coverage and current == "unknown":
                    current = "partial"

                regulation_status[reg_id] = current

                if current == "compliant":
                    compliant_count += 1
                elif current == "partial":
                    partial_count += 1
                    gaps.append({
                        "regulation_id": reg_id,
                        "regulation_name": reg_name,
                        "status": "partial",
                        "gap_description": f"Partial compliance with {reg_name}",
                        "requirements_met": len(requirements) // 2,
                        "requirements_total": len(requirements),
                        "severity": "MEDIUM",
                    })
                else:
                    non_compliant_count += 1
                    gaps.append({
                        "regulation_id": reg_id,
                        "regulation_name": reg_name,
                        "status": current,
                        "gap_description": f"Compliance status unknown/non-compliant for {reg_name}",
                        "requirements_met": 0,
                        "requirements_total": len(requirements),
                        "severity": "HIGH" if current == "non_compliant" else "MEDIUM",
                    })

            total = len(regulations)
            outputs["regulation_status"] = regulation_status
            outputs["gaps"] = gaps
            outputs["compliant_count"] = compliant_count
            outputs["partial_count"] = partial_count
            outputs["non_compliant_count"] = non_compliant_count
            outputs["overall_compliance_pct"] = round(
                compliant_count / max(total, 1) * 100, 1
            )

            if non_compliant_count > 0:
                warnings.append(f"{non_compliant_count} regulation(s) are non-compliant or unknown")

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("ComplianceAssessment failed: %s", exc, exc_info=True)
            errors.append(str(exc))
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 3: Action Planning
# ---------------------------------------------------------------------------

class ActionPlanningPhase:
    """Generate gap remediation timeline and action items."""

    PHASE_NAME = "action_planning"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Create prioritized action plan for compliance gaps."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            assessment = context.get_phase_output("compliance_assessment")
            gaps = assessment.get("gaps", [])

            action_items: List[Dict[str, Any]] = []
            timeline_entries: List[Dict[str, Any]] = []

            for idx, gap in enumerate(gaps):
                severity = gap.get("severity", "MEDIUM")
                reg_id = gap.get("regulation_id", "")
                reg_name = gap.get("regulation_name", "")

                if severity == "HIGH":
                    priority = 1
                    deadline_months = 3
                elif severity == "MEDIUM":
                    priority = 2
                    deadline_months = 6
                else:
                    priority = 3
                    deadline_months = 12

                action = {
                    "action_id": f"REG-ACT-{idx + 1:03d}",
                    "regulation_id": reg_id,
                    "regulation_name": reg_name,
                    "priority": priority,
                    "severity": severity,
                    "description": f"Achieve compliance with {reg_name}",
                    "steps": [
                        f"Conduct detailed gap assessment for {reg_id}",
                        f"Develop compliance plan for {reg_id}",
                        f"Implement required controls/processes",
                        f"Conduct internal audit and verification",
                    ],
                    "deadline_months": deadline_months,
                    "responsible": "Compliance Team",
                    "status": "PLANNED",
                }
                action_items.append(action)

                timeline_entries.append({
                    "regulation_id": reg_id,
                    "phase": "remediation",
                    "start_month": 1,
                    "end_month": deadline_months,
                    "priority": priority,
                })

            # Sort by priority
            action_items.sort(key=lambda x: x["priority"])
            timeline_entries.sort(key=lambda x: x["priority"])

            outputs["action_items"] = action_items
            outputs["timeline"] = timeline_entries
            outputs["total_actions"] = len(action_items)
            outputs["high_priority_count"] = sum(1 for a in action_items if a["priority"] == 1)
            outputs["estimated_completion_months"] = max(
                (a["deadline_months"] for a in action_items), default=0
            )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("ActionPlanning failed: %s", exc, exc_info=True)
            errors.append(str(exc))
            status = PhaseStatus.FAILED

        completed_at = utcnow()
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

class RegulatoryComplianceWorkflow:
    """
    Three-phase multi-regulation compliance assessment workflow.

    Maps applicable EU regulations to manufacturing operations,
    evaluates current compliance status, and generates a prioritized
    action plan for gap remediation.
    """

    WORKFLOW_NAME = "regulatory_compliance"

    PHASE_ORDER = [
        "regulation_mapping", "compliance_assessment", "action_planning",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize RegulatoryComplianceWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "regulation_mapping": RegulationMappingPhase(),
            "compliance_assessment": ComplianceAssessmentPhase(),
            "action_planning": ActionPlanningPhase(),
        }

    async def run(self, input_data: RegulatoryComplianceInput) -> RegulatoryComplianceResult:
        """Execute the complete 3-phase regulatory compliance workflow."""
        started_at = utcnow()
        logger.info("Starting regulatory compliance workflow %s", self.workflow_id)
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
                context.errors.extend(result.errors)
                context.warnings.extend(result.warnings)
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.FAILED,
                    started_at=utcnow(), errors=[str(exc)],
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

        completed_at = utcnow()
        duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        return RegulatoryComplianceResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            regulation_status=summary.get("regulation_status", {}),
            gaps=summary.get("gaps", []),
            action_items=summary.get("action_items", []),
            timeline=summary.get("timeline", []),
        )

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build summary from phase outputs."""
        assess = context.get_phase_output("compliance_assessment")
        plan = context.get_phase_output("action_planning")
        return {
            "regulation_status": assess.get("regulation_status", {}),
            "gaps": assess.get("gaps", []),
            "action_items": plan.get("action_items", []),
            "timeline": plan.get("timeline", []),
            "overall_compliance_pct": assess.get("overall_compliance_pct", 0.0),
        }

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        """Send progress notification via callback if configured."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
