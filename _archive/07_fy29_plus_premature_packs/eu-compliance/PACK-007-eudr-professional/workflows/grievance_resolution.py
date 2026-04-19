# -*- coding: utf-8 -*-
"""
Grievance Resolution Workflow
===============================

Five-phase workflow for managing stakeholder grievances related to EUDR
compliance, deforestation, or indigenous rights.

This workflow enables:
- Structured grievance intake and documentation
- Severity-based triage and prioritization
- Investigation with evidence collection
- Resolution with corrective actions
- Follow-up monitoring

Phases:
    1. Intake - Receive and document grievance
    2. Triage - Assess severity and assign investigator
    3. Investigation - Gather evidence and interview stakeholders
    4. Resolution - Determine outcome and corrective actions
    5. Follow-Up - Monitor implementation and closure

Regulatory Context:
    EUDR recitals emphasize protection of indigenous rights and stakeholder
    engagement. Effective grievance mechanisms demonstrate responsible due
    diligence and support social license to operate.

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import random
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class Phase(str, Enum):
    """Workflow phases."""
    INTAKE = "intake"
    TRIAGE = "triage"
    INVESTIGATION = "investigation"
    RESOLUTION = "resolution"
    FOLLOW_UP = "follow_up"


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GrievanceType(str, Enum):
    """Types of grievances."""
    DEFORESTATION = "deforestation"
    INDIGENOUS_RIGHTS = "indigenous_rights"
    LAND_CONFLICT = "land_conflict"
    LABOR_RIGHTS = "labor_rights"
    ENVIRONMENTAL_DAMAGE = "environmental_damage"
    DATA_PRIVACY = "data_privacy"
    OTHER = "other"


class Severity(str, Enum):
    """Grievance severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResolutionOutcome(str, Enum):
    """Resolution outcomes."""
    SUBSTANTIATED = "substantiated"
    PARTIALLY_SUBSTANTIATED = "partially_substantiated"
    UNSUBSTANTIATED = "unsubstantiated"
    WITHDRAWN = "withdrawn"


# =============================================================================
# DATA MODELS
# =============================================================================


class GrievanceResolutionConfig(BaseModel):
    """Configuration for grievance resolution workflow."""
    grievance_type: GrievanceType = Field(..., description="Type of grievance")
    complainant_name: Optional[str] = Field(None, description="Complainant (can be anonymous)")
    complainant_email: Optional[str] = Field(None, description="Contact email")
    description: str = Field(..., min_length=10, description="Grievance description")
    affected_supplier_id: Optional[str] = Field(None, description="Supplier involved")
    affected_plot_id: Optional[str] = Field(None, description="Plot involved")
    supporting_evidence: List[str] = Field(default_factory=list, description="Evidence URLs/paths")
    auto_notify_stakeholders: bool = Field(default=True, description="Auto-notify relevant parties")


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase: Phase = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Execution duration")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


class WorkflowContext(BaseModel):
    """Shared context passed between workflow phases."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique execution ID")
    config: GrievanceResolutionConfig = Field(..., description="Workflow configuration")
    phase_results: List[PhaseResult] = Field(default_factory=list, description="Completed phase results")
    state: Dict[str, Any] = Field(default_factory=dict, description="Shared state data")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Workflow start time")

    class Config:
        arbitrary_types_allowed = True


class WorkflowResult(BaseModel):
    """Complete result from the grievance resolution workflow."""
    workflow_name: str = Field(default="grievance_resolution", description="Workflow identifier")
    phases: List[PhaseResult] = Field(default_factory=list, description="All phase results")
    overall_status: PhaseStatus = Field(..., description="Overall workflow status")
    total_duration_seconds: float = Field(default=0.0, ge=0.0, description="Total execution time")
    provenance_hash: str = Field(default="", description="Workflow-level provenance hash")
    execution_id: str = Field(..., description="Execution identifier")
    grievance_id: str = Field(..., description="Grievance case identifier")
    grievance_type: str = Field(..., description="Type of grievance")
    severity: str = Field(default=Severity.MEDIUM.value, description="Assessed severity")
    resolution_outcome: Optional[str] = Field(None, description="Investigation outcome")
    corrective_actions: List[str] = Field(default_factory=list, description="Actions taken")
    resolution_time_days: int = Field(default=0, ge=0, description="Days to resolution")
    case_closed: bool = Field(default=False, description="Case closure status")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


# =============================================================================
# GRIEVANCE RESOLUTION WORKFLOW
# =============================================================================


class GrievanceResolutionWorkflow:
    """
    Five-phase grievance resolution workflow.

    Manages stakeholder grievances with:
    - Structured intake and documentation
    - Severity-based triage and assignment
    - Evidence-based investigation
    - Fair resolution with corrective actions
    - Follow-up monitoring and closure

    Example:
        >>> config = GrievanceResolutionConfig(
        ...     grievance_type=GrievanceType.DEFORESTATION,
        ...     description="Satellite imagery shows recent clearing in Plot-XYZ",
        ...     affected_plot_id="PLOT-XYZ",
        ... )
        >>> workflow = GrievanceResolutionWorkflow(config)
        >>> result = await workflow.run(WorkflowContext(config=config))
        >>> assert result.case_closed is True
    """

    def __init__(self, config: GrievanceResolutionConfig) -> None:
        """Initialize the grievance resolution workflow."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.GrievanceResolutionWorkflow")

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """
        Execute the full 5-phase grievance resolution workflow.

        Args:
            context: Workflow context with configuration and initial state.

        Returns:
            WorkflowResult with investigation outcome and corrective actions.
        """
        started_at = datetime.utcnow()
        grievance_id = f"GRV-{uuid.uuid4().hex[:8].upper()}"

        self.logger.info(
            "Starting grievance resolution workflow execution_id=%s grievance_id=%s type=%s",
            context.execution_id,
            grievance_id,
            self.config.grievance_type.value,
        )

        context.state["grievance_id"] = grievance_id

        phase_handlers = [
            (Phase.INTAKE, self._phase_1_intake),
            (Phase.TRIAGE, self._phase_2_triage),
            (Phase.INVESTIGATION, self._phase_3_investigation),
            (Phase.RESOLUTION, self._phase_4_resolution),
            (Phase.FOLLOW_UP, self._phase_5_follow_up),
        ]

        overall_status = PhaseStatus.COMPLETED

        for phase, handler in phase_handlers:
            phase_start = datetime.utcnow()
            self.logger.info("Starting phase: %s", phase.value)

            try:
                phase_result = await handler(context)
                phase_result.duration_seconds = (datetime.utcnow() - phase_start).total_seconds()
                phase_result.timestamp = datetime.utcnow()
            except Exception as exc:
                self.logger.error("Phase '%s' failed: %s", phase.value, exc, exc_info=True)
                phase_result = PhaseResult(
                    phase=phase,
                    status=PhaseStatus.FAILED,
                    data={"error": str(exc)},
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    provenance_hash=self._hash({"error": str(exc)}),
                    timestamp=datetime.utcnow(),
                )

            context.phase_results.append(phase_result)

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                self.logger.error("Phase '%s' failed; halting workflow.", phase.value)
                break

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        resolution_days = int(total_duration / 86400)

        # Extract final outputs
        severity = context.state.get("severity", Severity.MEDIUM.value)
        outcome = context.state.get("resolution_outcome")
        corrective_actions = context.state.get("corrective_actions", [])
        case_closed = context.state.get("case_closed", False)

        provenance = self._hash({
            "execution_id": context.execution_id,
            "phases": [p.provenance_hash for p in context.phase_results],
            "grievance_id": grievance_id,
        })

        self.logger.info(
            "Grievance resolution finished execution_id=%s grievance_id=%s "
            "outcome=%s closed=%s",
            context.execution_id,
            grievance_id,
            outcome,
            case_closed,
        )

        return WorkflowResult(
            phases=context.phase_results,
            overall_status=overall_status,
            total_duration_seconds=total_duration,
            provenance_hash=provenance,
            execution_id=context.execution_id,
            grievance_id=grievance_id,
            grievance_type=self.config.grievance_type.value,
            severity=severity,
            resolution_outcome=outcome,
            corrective_actions=corrective_actions,
            resolution_time_days=resolution_days,
            case_closed=case_closed,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Intake
    # -------------------------------------------------------------------------

    async def _phase_1_intake(self, context: WorkflowContext) -> PhaseResult:
        """
        Receive and document grievance.

        Intake steps:
        - Assign unique grievance ID
        - Record complainant information (or mark anonymous)
        - Document grievance details and description
        - Collect supporting evidence
        - Timestamp receipt
        - Send acknowledgment to complainant (if contact provided)
        """
        phase = Phase.INTAKE
        grievance_id = context.state["grievance_id"]

        self.logger.info("Processing grievance intake: %s", grievance_id)

        await asyncio.sleep(0.05)

        intake_record = {
            "grievance_id": grievance_id,
            "grievance_type": self.config.grievance_type.value,
            "complainant_name": self.config.complainant_name or "ANONYMOUS",
            "complainant_email": self.config.complainant_email,
            "description": self.config.description,
            "affected_supplier_id": self.config.affected_supplier_id,
            "affected_plot_id": self.config.affected_plot_id,
            "supporting_evidence_count": len(self.config.supporting_evidence),
            "received_at": datetime.utcnow().isoformat(),
            "acknowledgment_sent": bool(self.config.complainant_email),
        }

        context.state["intake_record"] = intake_record

        provenance = self._hash({
            "phase": phase.value,
            "grievance_id": grievance_id,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "grievance_id": grievance_id,
                "grievance_type": self.config.grievance_type.value,
                "evidence_count": len(self.config.supporting_evidence),
                "acknowledgment_sent": intake_record["acknowledgment_sent"],
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Triage
    # -------------------------------------------------------------------------

    async def _phase_2_triage(self, context: WorkflowContext) -> PhaseResult:
        """
        Assess severity and assign investigator.

        Triage criteria:
        - Grievance type (deforestation/indigenous rights = higher severity)
        - Supporting evidence quality
        - Affected stakeholders
        - Potential legal/reputational impact

        Assignment:
        - Critical: Senior investigator + legal counsel
        - High: Experienced investigator
        - Medium: Standard investigator
        - Low: Automated review + spot check
        """
        phase = Phase.TRIAGE
        grievance_type = self.config.grievance_type
        evidence_count = len(self.config.supporting_evidence)

        self.logger.info("Triaging grievance type=%s", grievance_type.value)

        # Determine severity
        if grievance_type in (GrievanceType.DEFORESTATION, GrievanceType.INDIGENOUS_RIGHTS):
            severity = Severity.HIGH if evidence_count > 0 else Severity.MEDIUM
        elif grievance_type in (GrievanceType.LAND_CONFLICT, GrievanceType.ENVIRONMENTAL_DAMAGE):
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        # Override to critical if multiple suppliers/plots affected
        if self.config.affected_supplier_id and self.config.affected_plot_id and evidence_count >= 3:
            severity = Severity.CRITICAL

        # Assign investigator
        investigator = self._assign_investigator(severity)

        # Calculate response SLA
        sla_days = self._calculate_sla(severity)

        context.state["severity"] = severity.value
        context.state["assigned_investigator"] = investigator
        context.state["response_sla_days"] = sla_days
        context.state["triage_completed_at"] = datetime.utcnow().isoformat()

        provenance = self._hash({
            "phase": phase.value,
            "severity": severity.value,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "severity": severity.value,
                "assigned_investigator": investigator,
                "response_sla_days": sla_days,
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Investigation
    # -------------------------------------------------------------------------

    async def _phase_3_investigation(self, context: WorkflowContext) -> PhaseResult:
        """
        Gather evidence and interview stakeholders.

        Investigation activities:
        - Review supporting evidence
        - Interview complainant (if identifiable)
        - Interview affected supplier
        - Request additional documentation
        - Conduct site visit (if required)
        - Analyze satellite imagery (for deforestation claims)
        - Consult legal/technical experts
        """
        phase = Phase.INVESTIGATION
        severity = context.state.get("severity", Severity.MEDIUM.value)
        grievance_type = self.config.grievance_type

        self.logger.info("Conducting investigation severity=%s", severity)

        await asyncio.sleep(0.1)

        # Simulate investigation activities
        activities = []

        # Always review evidence
        activities.append({
            "activity": "evidence_review",
            "description": f"Reviewed {len(self.config.supporting_evidence)} evidence items",
            "completed": True,
        })

        # Interview stakeholders
        if self.config.complainant_email:
            activities.append({
                "activity": "complainant_interview",
                "description": "Interviewed complainant via video call",
                "completed": True,
            })

        if self.config.affected_supplier_id:
            activities.append({
                "activity": "supplier_interview",
                "description": f"Interviewed supplier {self.config.affected_supplier_id}",
                "completed": True,
            })

        # Special activities based on type
        if grievance_type == GrievanceType.DEFORESTATION:
            activities.append({
                "activity": "satellite_analysis",
                "description": "Analyzed satellite imagery for deforestation evidence",
                "completed": True,
                "findings": "Forest cover loss detected" if random.random() > 0.5 else "No forest cover loss detected",
            })

        if grievance_type == GrievanceType.INDIGENOUS_RIGHTS:
            activities.append({
                "activity": "fpic_review",
                "description": "Reviewed FPIC (Free, Prior, Informed Consent) documentation",
                "completed": True,
            })

        # Site visit for critical cases
        if severity == Severity.CRITICAL.value:
            activities.append({
                "activity": "site_visit",
                "description": "Conducted on-site investigation",
                "completed": True,
            })

        context.state["investigation_activities"] = activities
        context.state["investigation_completed_at"] = datetime.utcnow().isoformat()

        provenance = self._hash({
            "phase": phase.value,
            "activity_count": len(activities),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "investigation_activities": len(activities),
                "activities": [a["activity"] for a in activities],
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Resolution
    # -------------------------------------------------------------------------

    async def _phase_4_resolution(self, context: WorkflowContext) -> PhaseResult:
        """
        Determine outcome and corrective actions.

        Resolution process:
        - Assess investigation findings
        - Determine outcome (substantiated, partially, unsubstantiated, withdrawn)
        - Develop corrective actions (if substantiated)
        - Communicate resolution to complainant
        - Notify affected supplier of required actions
        """
        phase = Phase.RESOLUTION
        activities = context.state.get("investigation_activities", [])
        severity = context.state.get("severity", Severity.MEDIUM.value)

        self.logger.info("Determining resolution from %d investigation activities", len(activities))

        # Simulate resolution determination
        # Weighted towards substantiated if evidence and interviews completed
        evidence_reviewed = any(a["activity"] == "evidence_review" for a in activities)
        interviews_completed = any(a["activity"] in ("complainant_interview", "supplier_interview") for a in activities)

        if evidence_reviewed and interviews_completed:
            outcome = random.choice([
                ResolutionOutcome.SUBSTANTIATED,
                ResolutionOutcome.SUBSTANTIATED,
                ResolutionOutcome.PARTIALLY_SUBSTANTIATED,
                ResolutionOutcome.UNSUBSTANTIATED,
            ])
        else:
            outcome = ResolutionOutcome.UNSUBSTANTIATED

        # Generate corrective actions if substantiated
        corrective_actions = []

        if outcome in (ResolutionOutcome.SUBSTANTIATED, ResolutionOutcome.PARTIALLY_SUBSTANTIATED):
            if self.config.grievance_type == GrievanceType.DEFORESTATION:
                corrective_actions.extend([
                    f"Suspend sourcing from affected plot {self.config.affected_plot_id or 'pending identification'} pending remediation",
                    "Require supplier to provide legal documentation for land use",
                    "Implement quarterly satellite monitoring for affected area",
                    "Conduct third-party audit within 90 days",
                ])

            elif self.config.grievance_type == GrievanceType.INDIGENOUS_RIGHTS:
                corrective_actions.extend([
                    "Require supplier to obtain FPIC documentation from affected indigenous community",
                    "Suspend purchases pending FPIC verification",
                    "Engage independent third-party to mediate with community",
                ])

            elif self.config.grievance_type == GrievanceType.LAND_CONFLICT:
                corrective_actions.extend([
                    "Request supplier to provide land title documentation",
                    "Engage legal counsel to assess land rights",
                    "Consider exclusion from supply chain if land rights unresolved",
                ])

            else:
                corrective_actions.append("Develop corrective action plan with supplier within 30 days")

            # Add monitoring action for all substantiated cases
            corrective_actions.append(
                f"Monitor corrective action implementation; review progress in 90 days"
            )

        context.state["resolution_outcome"] = outcome.value
        context.state["corrective_actions"] = corrective_actions
        context.state["resolution_determined_at"] = datetime.utcnow().isoformat()

        # Communicate resolution
        if self.config.auto_notify_stakeholders:
            self._notify_stakeholders(outcome, corrective_actions)

        provenance = self._hash({
            "phase": phase.value,
            "outcome": outcome.value,
            "actions": len(corrective_actions),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "resolution_outcome": outcome.value,
                "corrective_actions": len(corrective_actions),
                "stakeholders_notified": self.config.auto_notify_stakeholders,
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 5: Follow-Up
    # -------------------------------------------------------------------------

    async def _phase_5_follow_up(self, context: WorkflowContext) -> PhaseResult:
        """
        Monitor implementation and closure.

        Follow-up activities:
        - Track corrective action completion
        - Verify remediation (site visit, satellite imagery)
        - Collect complainant feedback
        - Document lessons learned
        - Close case (or escalate if unresolved)
        """
        phase = Phase.FOLLOW_UP
        outcome = context.state.get("resolution_outcome")
        corrective_actions = context.state.get("corrective_actions", [])

        self.logger.info("Following up on resolution outcome=%s", outcome)

        # Simulate follow-up activities
        if outcome in (ResolutionOutcome.SUBSTANTIATED.value, ResolutionOutcome.PARTIALLY_SUBSTANTIATED.value):
            # Track corrective actions (simulated completion)
            actions_completed = random.randint(len(corrective_actions) // 2, len(corrective_actions))

            # Determine if case can be closed
            case_closed = actions_completed == len(corrective_actions)

            follow_up_data = {
                "actions_total": len(corrective_actions),
                "actions_completed": actions_completed,
                "actions_pending": len(corrective_actions) - actions_completed,
                "complainant_satisfied": random.choice([True, False]) if self.config.complainant_email else None,
                "case_closed": case_closed,
                "next_review_date": None if case_closed else (datetime.utcnow() + timedelta(days=90)).isoformat(),
            }

        elif outcome == ResolutionOutcome.UNSUBSTANTIATED.value:
            # Close case immediately
            case_closed = True
            follow_up_data = {
                "actions_total": 0,
                "actions_completed": 0,
                "case_closed": True,
                "closure_reason": "Grievance unsubstantiated after investigation",
            }

        else:  # WITHDRAWN
            case_closed = True
            follow_up_data = {
                "case_closed": True,
                "closure_reason": "Grievance withdrawn by complainant",
            }

        context.state["case_closed"] = case_closed
        context.state["follow_up_data"] = follow_up_data
        context.state["case_closed_at"] = datetime.utcnow().isoformat() if case_closed else None

        provenance = self._hash({
            "phase": phase.value,
            "case_closed": case_closed,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data=follow_up_data,
            provenance_hash=provenance,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _assign_investigator(self, severity: Severity) -> str:
        """Assign investigator based on severity."""
        if severity == Severity.CRITICAL:
            return "senior_investigator_legal_counsel"
        elif severity == Severity.HIGH:
            return "experienced_investigator"
        elif severity == Severity.MEDIUM:
            return "standard_investigator"
        return "automated_review"

    def _calculate_sla(self, severity: Severity) -> int:
        """Calculate response SLA in days."""
        sla_map = {
            Severity.CRITICAL: 7,
            Severity.HIGH: 14,
            Severity.MEDIUM: 30,
            Severity.LOW: 60,
        }
        return sla_map.get(severity, 30)

    def _notify_stakeholders(
        self,
        outcome: ResolutionOutcome,
        corrective_actions: List[str],
    ) -> None:
        """Notify complainant and affected supplier of resolution."""
        # Simulated notification (in production, send emails/notifications)
        self.logger.info(
            "Notifying stakeholders of resolution outcome=%s actions=%d",
            outcome.value,
            len(corrective_actions),
        )

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
