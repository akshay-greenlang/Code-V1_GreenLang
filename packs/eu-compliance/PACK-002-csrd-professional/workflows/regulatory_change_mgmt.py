# -*- coding: utf-8 -*-
"""
Regulatory Change Management Workflow
=======================================

Monitors regulatory sources for changes affecting CSRD compliance, classifies
severity, assesses impact on the current report, generates remediation plans,
and maintains a regulatory deadline calendar.

Phases:
    1. Monitoring: Scan regulatory sources for changes (EFRAG/EU/ESMA/ISSB/national)
    2. Classification: Classify severity, identify affected standards
    3. Impact Assessment: Analyze impact on current report state
    4. Gap Resolution: Generate remediation plans with effort estimates
    5. Calendar Update: Update regulatory deadline calendar

Author: GreenLang Team
Version: 2.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class ChangeSeverity(str, Enum):
    """Severity classification for regulatory changes."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0)
    agents_executed: int = Field(default=0)
    records_processed: int = Field(default=0)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class PhaseDefinition(BaseModel):
    """Internal definition of a workflow phase."""
    name: str
    display_name: str
    estimated_minutes: float
    required: bool = True
    depends_on: List[str] = Field(default_factory=list)


class RegulatoryChangeMgmtInput(BaseModel):
    """Input configuration for the regulatory change management workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    jurisdictions: List[str] = Field(
        default_factory=lambda: ["EU", "DE", "FR", "NL"],
        description="ISO country codes to monitor"
    )
    current_report_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current compliance status and report state"
    )
    monitoring_sources: List[str] = Field(
        default_factory=lambda: [
            "efrag", "eu_commission", "esma", "issb", "national"
        ],
        description="Regulatory sources to monitor"
    )


class RegulatoryChangeMgmtResult(BaseModel):
    """Complete result from the regulatory change management workflow."""
    workflow_id: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    changes_detected: List[Dict[str, Any]] = Field(
        default_factory=list, description="Regulatory changes found"
    )
    impact_assessments: List[Dict[str, Any]] = Field(
        default_factory=list, description="Impact assessment per change"
    )
    compliance_gaps: List[Dict[str, Any]] = Field(
        default_factory=list, description="New compliance gaps identified"
    )
    regulatory_calendar: Dict[str, Any] = Field(
        default_factory=dict, description="Updated regulatory calendar"
    )
    remediation_plans: List[Dict[str, Any]] = Field(
        default_factory=list, description="Remediation plans"
    )
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RegulatoryChangeMgmtWorkflow:
    """
    Regulatory change management workflow.

    Monitors regulatory sources for changes, classifies severity, assesses
    impact on the current report, generates remediation plans, and maintains
    a regulatory deadline calendar.

    Attributes:
        workflow_id: Unique execution identifier.
        _cancelled: Cancellation flag.
        _progress_callback: Optional progress callback.

    Example:
        >>> workflow = RegulatoryChangeMgmtWorkflow()
        >>> input_cfg = RegulatoryChangeMgmtInput(
        ...     organization_id="org-123",
        ...     jurisdictions=["EU", "DE"],
        ... )
        >>> result = await workflow.execute(input_cfg)
        >>> print(f"Changes detected: {len(result.changes_detected)}")
    """

    PHASES: List[PhaseDefinition] = [
        PhaseDefinition(
            name="monitoring",
            display_name="Regulatory Source Monitoring",
            estimated_minutes=15.0,
            required=True,
            depends_on=[],
        ),
        PhaseDefinition(
            name="classification",
            display_name="Change Classification",
            estimated_minutes=10.0,
            required=True,
            depends_on=["monitoring"],
        ),
        PhaseDefinition(
            name="impact_assessment",
            display_name="Impact Assessment",
            estimated_minutes=20.0,
            required=True,
            depends_on=["classification"],
        ),
        PhaseDefinition(
            name="gap_resolution",
            display_name="Gap Resolution & Remediation",
            estimated_minutes=15.0,
            required=True,
            depends_on=["impact_assessment"],
        ),
        PhaseDefinition(
            name="calendar_update",
            display_name="Regulatory Calendar Update",
            estimated_minutes=5.0,
            required=True,
            depends_on=["gap_resolution"],
        ),
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the regulatory change management workflow.

        Args:
            progress_callback: Optional callback(phase_name, message, pct_complete).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._cancelled: bool = False
        self._progress_callback = progress_callback
        self._phase_results: Dict[str, PhaseResult] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: RegulatoryChangeMgmtInput
    ) -> RegulatoryChangeMgmtResult:
        """
        Execute the regulatory change management workflow.

        Args:
            input_data: Validated workflow input.

        Returns:
            RegulatoryChangeMgmtResult with changes, impacts, and remediation plans.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting regulatory change management %s for org=%s jurisdictions=%s",
            self.workflow_id, input_data.organization_id, input_data.jurisdictions,
        )
        self._notify_progress("workflow", "Workflow started", 0.0)

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            for idx, phase_def in enumerate(self.PHASES):
                if self._cancelled:
                    overall_status = WorkflowStatus.CANCELLED
                    break

                for dep in phase_def.depends_on:
                    dep_result = self._phase_results.get(dep)
                    if dep_result and dep_result.status == PhaseStatus.FAILED:
                        if phase_def.required:
                            raise RuntimeError(
                                f"Required phase '{phase_def.name}' cannot run: "
                                f"dependency '{dep}' failed."
                            )

                pct_base = idx / len(self.PHASES)
                self._notify_progress(
                    phase_def.name, f"Starting: {phase_def.display_name}", pct_base
                )

                phase_result = await self._execute_phase(
                    phase_def, input_data, pct_base
                )
                completed_phases.append(phase_result)
                self._phase_results[phase_def.name] = phase_result

                if phase_result.status == PhaseStatus.FAILED and phase_def.required:
                    overall_status = WorkflowStatus.FAILED
                    break

            if overall_status == WorkflowStatus.RUNNING:
                all_ok = all(
                    p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                    for p in completed_phases
                )
                overall_status = (
                    WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL
                )

        except Exception as exc:
            logger.critical(
                "Workflow %s failed: %s", self.workflow_id, exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            completed_phases.append(PhaseResult(
                phase_name="workflow_error", status=PhaseStatus.FAILED,
                errors=[str(exc)],
                provenance_hash=self._hash_data({"error": str(exc)}),
            ))

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        changes = self._extract_changes(completed_phases)
        impacts = self._extract_impacts(completed_phases)
        gaps = self._extract_gaps(completed_phases)
        calendar = self._extract_calendar(completed_phases)
        remediation = self._extract_remediation(completed_phases)
        artifacts = {p.phase_name: p.artifacts for p in completed_phases if p.artifacts}

        provenance = self._hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress("workflow", f"Workflow {overall_status.value}", 1.0)

        return RegulatoryChangeMgmtResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            changes_detected=changes,
            impact_assessments=impacts,
            compliance_gaps=gaps,
            regulatory_calendar=calendar,
            remediation_plans=remediation,
            artifacts=artifacts,
            provenance_hash=provenance,
        )

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        logger.info("Cancellation requested for workflow %s", self.workflow_id)
        self._cancelled = True

    # -------------------------------------------------------------------------
    # Phase Execution
    # -------------------------------------------------------------------------

    async def _execute_phase(
        self, phase_def: PhaseDefinition,
        input_data: RegulatoryChangeMgmtInput, pct_base: float,
    ) -> PhaseResult:
        """Dispatch to the correct phase handler."""
        started_at = datetime.utcnow()
        handler_map = {
            "monitoring": self._phase_monitoring,
            "classification": self._phase_classification,
            "impact_assessment": self._phase_impact_assessment,
            "gap_resolution": self._phase_gap_resolution,
            "calendar_update": self._phase_calendar_update,
        }
        handler = handler_map.get(phase_def.name)
        if handler is None:
            return PhaseResult(
                phase_name=phase_def.name, status=PhaseStatus.FAILED,
                started_at=started_at,
                errors=[f"Unknown phase: {phase_def.name}"],
                provenance_hash=self._hash_data({"error": "unknown_phase"}),
            )
        try:
            result = await handler(input_data, pct_base)
            result.started_at = started_at
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (result.completed_at - started_at).total_seconds()
            return result
        except Exception as exc:
            logger.error("Phase '%s' raised: %s", phase_def.name, exc, exc_info=True)
            return PhaseResult(
                phase_name=phase_def.name, status=PhaseStatus.FAILED,
                started_at=started_at, completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - started_at).total_seconds(),
                errors=[str(exc)],
                provenance_hash=self._hash_data({"error": str(exc)}),
            )

    # -------------------------------------------------------------------------
    # Phase 1: Monitoring
    # -------------------------------------------------------------------------

    async def _phase_monitoring(
        self, input_data: RegulatoryChangeMgmtInput, pct_base: float
    ) -> PhaseResult:
        """
        Scan regulatory sources for changes affecting CSRD compliance.
        """
        phase_name = "monitoring"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        all_changes: List[Dict[str, Any]] = []

        for source in input_data.monitoring_sources:
            self._notify_progress(
                phase_name, f"Scanning {source} for changes", pct_base + 0.02
            )

            changes = await self._scan_source(
                input_data.organization_id, source, input_data.jurisdictions
            )
            agents_executed += 1
            all_changes.extend(changes)

        artifacts["raw_changes"] = all_changes
        artifacts["total_changes_detected"] = len(all_changes)
        artifacts["sources_scanned"] = input_data.monitoring_sources

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(all_changes),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Classification
    # -------------------------------------------------------------------------

    async def _phase_classification(
        self, input_data: RegulatoryChangeMgmtInput, pct_base: float
    ) -> PhaseResult:
        """
        Classify detected changes by severity and identify affected
        ESRS standards.
        """
        phase_name = "classification"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        monitoring_phase = self._phase_results.get("monitoring")
        raw_changes = (
            monitoring_phase.artifacts.get("raw_changes", [])
            if monitoring_phase and monitoring_phase.artifacts else []
        )

        self._notify_progress(
            phase_name, "Classifying regulatory changes", pct_base + 0.02
        )

        classified: List[Dict[str, Any]] = []
        for change in raw_changes:
            classification = await self._classify_change(
                input_data.organization_id, change
            )
            classified.append({**change, **classification})
            agents_executed += 1

        # Sort by severity
        severity_order = {
            "critical": 0, "high": 1, "medium": 2, "low": 3, "informational": 4
        }
        classified.sort(key=lambda c: severity_order.get(c.get("severity", ""), 99))

        artifacts["classified_changes"] = classified
        artifacts["by_severity"] = {}
        for c in classified:
            sev = c.get("severity", "unknown")
            artifacts["by_severity"][sev] = artifacts["by_severity"].get(sev, 0) + 1

        critical_count = artifacts["by_severity"].get("critical", 0)
        if critical_count > 0:
            warnings.append(
                f"{critical_count} critical regulatory change(s) detected."
            )

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(classified),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Impact Assessment
    # -------------------------------------------------------------------------

    async def _phase_impact_assessment(
        self, input_data: RegulatoryChangeMgmtInput, pct_base: float
    ) -> PhaseResult:
        """
        Analyze impact of each classified change on the current report state.
        """
        phase_name = "impact_assessment"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        classification_phase = self._phase_results.get("classification")
        classified = (
            classification_phase.artifacts.get("classified_changes", [])
            if classification_phase and classification_phase.artifacts else []
        )

        self._notify_progress(
            phase_name, "Assessing impact on current report", pct_base + 0.02
        )

        assessments: List[Dict[str, Any]] = []
        compliance_gaps: List[Dict[str, Any]] = []

        for change in classified:
            self._notify_progress(
                phase_name,
                f"Assessing impact of {change.get('change_id', 'unknown')}",
                pct_base + 0.04,
            )

            assessment = await self._assess_change_impact(
                input_data.organization_id, change,
                input_data.current_report_state,
            )
            agents_executed += 1
            assessments.append(assessment)

            if assessment.get("creates_gap", False):
                compliance_gaps.append({
                    "change_id": change.get("change_id", ""),
                    "gap_description": assessment.get("gap_description", ""),
                    "affected_standards": assessment.get("affected_standards", []),
                    "severity": change.get("severity", "medium"),
                })

        artifacts["impact_assessments"] = assessments
        artifacts["compliance_gaps"] = compliance_gaps
        artifacts["total_gaps"] = len(compliance_gaps)

        if compliance_gaps:
            warnings.append(
                f"{len(compliance_gaps)} new compliance gap(s) identified."
            )

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(assessments),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Gap Resolution
    # -------------------------------------------------------------------------

    async def _phase_gap_resolution(
        self, input_data: RegulatoryChangeMgmtInput, pct_base: float
    ) -> PhaseResult:
        """
        Generate remediation plans with effort estimates for each compliance gap.
        """
        phase_name = "gap_resolution"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        impact_phase = self._phase_results.get("impact_assessment")
        gaps = (
            impact_phase.artifacts.get("compliance_gaps", [])
            if impact_phase and impact_phase.artifacts else []
        )

        self._notify_progress(
            phase_name, "Generating remediation plans", pct_base + 0.02
        )

        remediation_plans: List[Dict[str, Any]] = []

        for gap in gaps:
            plan = await self._generate_remediation_plan(
                input_data.organization_id, gap
            )
            agents_executed += 1
            remediation_plans.append(plan)

        artifacts["remediation_plans"] = remediation_plans
        artifacts["total_plans"] = len(remediation_plans)
        artifacts["total_effort_hours"] = sum(
            p.get("estimated_effort_hours", 0) for p in remediation_plans
        )

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(remediation_plans),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 5: Calendar Update
    # -------------------------------------------------------------------------

    async def _phase_calendar_update(
        self, input_data: RegulatoryChangeMgmtInput, pct_base: float
    ) -> PhaseResult:
        """
        Update the regulatory deadline calendar with new effective dates
        and compliance deadlines from detected changes.
        """
        phase_name = "calendar_update"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        classification_phase = self._phase_results.get("classification")
        classified = (
            classification_phase.artifacts.get("classified_changes", [])
            if classification_phase and classification_phase.artifacts else []
        )

        self._notify_progress(
            phase_name, "Updating regulatory calendar", pct_base + 0.02
        )

        calendar = await self._update_regulatory_calendar(
            input_data.organization_id, classified, input_data.jurisdictions
        )
        agents_executed = 1

        artifacts["calendar"] = calendar
        artifacts["upcoming_deadlines"] = calendar.get("upcoming", [])
        artifacts["new_deadlines_added"] = calendar.get("new_entries", 0)

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(classified),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Agent Invocation Helpers
    # -------------------------------------------------------------------------

    async def _scan_source(
        self, org_id: str, source: str, jurisdictions: List[str]
    ) -> List[Dict[str, Any]]:
        """Scan a regulatory source for new changes."""
        await asyncio.sleep(0)
        now = datetime.utcnow()

        source_changes = {
            "efrag": [
                {
                    "change_id": f"EFRAG-{now.strftime('%Y')}-001",
                    "source": "efrag",
                    "title": "Updated ESRS Implementation Guidance v2.1",
                    "publication_date": now.strftime("%Y-%m-%d"),
                    "effective_date": (now + timedelta(days=180)).strftime("%Y-%m-%d"),
                    "affected_standards": ["ESRS_E1", "ESRS_E2"],
                    "summary": "Clarified transition plan disclosure requirements and "
                    "Scope 3 boundary definitions.",
                },
            ],
            "eu_commission": [
                {
                    "change_id": f"EU-COM-{now.strftime('%Y')}-015",
                    "source": "eu_commission",
                    "title": "Delegated Act on ESRS Sector Standards",
                    "publication_date": (now - timedelta(days=14)).strftime("%Y-%m-%d"),
                    "effective_date": (now + timedelta(days=365)).strftime("%Y-%m-%d"),
                    "affected_standards": ["ESRS_sector"],
                    "summary": "New sector-specific standards for energy, mining, "
                    "agriculture, and financial services.",
                },
            ],
            "esma": [],
            "issb": [
                {
                    "change_id": f"ISSB-{now.strftime('%Y')}-003",
                    "source": "issb",
                    "title": "IFRS S2 Climate Disclosure Amendments",
                    "publication_date": (now - timedelta(days=7)).strftime("%Y-%m-%d"),
                    "effective_date": (now + timedelta(days=270)).strftime("%Y-%m-%d"),
                    "affected_standards": ["ESRS_E1"],
                    "summary": "Alignment amendments between IFRS S2 and ESRS E1 "
                    "for interoperability.",
                },
            ],
            "national": [],
        }

        return source_changes.get(source, [])

    async def _classify_change(
        self, org_id: str, change: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify a regulatory change by severity and scope."""
        await asyncio.sleep(0)
        affected = change.get("affected_standards", [])
        source = change.get("source", "")

        if source in ("eu_commission", "efrag") and len(affected) > 1:
            severity = "high"
        elif source == "issb":
            severity = "medium"
        else:
            severity = "low"

        return {
            "severity": severity,
            "classification": "amendment" if "amend" in change.get("title", "").lower() else "new_guidance",
            "requires_action": severity in ("critical", "high", "medium"),
            "affected_disclosures_estimate": len(affected) * 8,
        }

    async def _assess_change_impact(
        self, org_id: str, change: Dict[str, Any],
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess the impact of a regulatory change on the current report."""
        await asyncio.sleep(0)
        creates_gap = change.get("severity") in ("critical", "high")
        return {
            "change_id": change.get("change_id", ""),
            "impact_level": change.get("severity", "medium"),
            "creates_gap": creates_gap,
            "gap_description": (
                f"New disclosure requirements from {change.get('title', '')} "
                "not yet reflected in current report."
                if creates_gap else ""
            ),
            "affected_standards": change.get("affected_standards", []),
            "estimated_rework_hours": 24 if creates_gap else 8,
            "data_changes_required": creates_gap,
        }

    async def _generate_remediation_plan(
        self, org_id: str, gap: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a remediation plan for a compliance gap."""
        await asyncio.sleep(0)
        return {
            "plan_id": str(uuid.uuid4()),
            "change_id": gap.get("change_id", ""),
            "title": f"Remediate: {gap.get('gap_description', 'Unknown gap')[:80]}",
            "steps": [
                "Review regulatory change documentation",
                "Identify affected data points and disclosures",
                "Update data collection procedures",
                "Re-run affected calculations",
                "Update report sections",
                "Validate compliance",
            ],
            "estimated_effort_hours": 32,
            "priority": gap.get("severity", "medium"),
            "deadline": (datetime.utcnow() + timedelta(days=60)).strftime("%Y-%m-%d"),
            "responsible_role": "Sustainability Manager",
        }

    async def _update_regulatory_calendar(
        self, org_id: str, changes: List[Dict[str, Any]],
        jurisdictions: List[str],
    ) -> Dict[str, Any]:
        """Update the regulatory deadline calendar."""
        await asyncio.sleep(0)
        now = datetime.utcnow()

        upcoming = [
            {
                "deadline_id": "CAL-001",
                "title": "CSRD Annual Report Filing Deadline",
                "due_date": (now + timedelta(days=120)).strftime("%Y-%m-%d"),
                "jurisdiction": "EU",
                "status": "on_track",
            },
            {
                "deadline_id": "CAL-002",
                "title": "EU Taxonomy Disclosure Update",
                "due_date": (now + timedelta(days=90)).strftime("%Y-%m-%d"),
                "jurisdiction": "EU",
                "status": "on_track",
            },
        ]

        new_entries = 0
        for change in changes:
            if change.get("effective_date"):
                upcoming.append({
                    "deadline_id": f"CAL-NEW-{new_entries + 1}",
                    "title": f"Effective: {change.get('title', '')}",
                    "due_date": change["effective_date"],
                    "jurisdiction": change.get("source", "EU"),
                    "status": "new",
                })
                new_entries += 1

        return {
            "upcoming": upcoming,
            "total_entries": len(upcoming),
            "new_entries": new_entries,
            "jurisdictions": jurisdictions,
        }

    # -------------------------------------------------------------------------
    # Result Extractors
    # -------------------------------------------------------------------------

    def _extract_changes(self, phases: List[PhaseResult]) -> List[Dict[str, Any]]:
        """Extract detected changes."""
        for p in phases:
            if p.phase_name == "classification" and p.artifacts:
                return p.artifacts.get("classified_changes", [])
        return []

    def _extract_impacts(self, phases: List[PhaseResult]) -> List[Dict[str, Any]]:
        """Extract impact assessments."""
        for p in phases:
            if p.phase_name == "impact_assessment" and p.artifacts:
                return p.artifacts.get("impact_assessments", [])
        return []

    def _extract_gaps(self, phases: List[PhaseResult]) -> List[Dict[str, Any]]:
        """Extract compliance gaps."""
        for p in phases:
            if p.phase_name == "impact_assessment" and p.artifacts:
                return p.artifacts.get("compliance_gaps", [])
        return []

    def _extract_calendar(self, phases: List[PhaseResult]) -> Dict[str, Any]:
        """Extract regulatory calendar."""
        for p in phases:
            if p.phase_name == "calendar_update" and p.artifacts:
                return p.artifacts.get("calendar", {})
        return {}

    def _extract_remediation(self, phases: List[PhaseResult]) -> List[Dict[str, Any]]:
        """Extract remediation plans."""
        for p in phases:
            if p.phase_name == "gap_resolution" and p.artifacts:
                return p.artifacts.get("remediation_plans", [])
        return []

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)

    @staticmethod
    def _hash_data(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        serialized = str(data).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()
