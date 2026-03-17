# -*- coding: utf-8 -*-
"""
Calendar Management Workflow
=================================

Three-phase workflow that populates regulatory deadlines from all four
constituent packs (CSRD, CBAM, EU Taxonomy, EUDR), analyzes cross-regulation
dependencies and critical path, and distributes alerts and notifications
to compliance teams.

Phases:
    1. CalendarPopulation - Populate deadlines from all 4 frameworks
    2. DependencyAnalysis - Identify cross-regulation dependencies and critical path
    3. AlertDistribution - Set up alerts and send notifications

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class RegulationPack(str, Enum):
    """Constituent regulation packs in the bundle."""
    CSRD = "CSRD"
    CBAM = "CBAM"
    EU_TAXONOMY = "EU_TAXONOMY"
    EUDR = "EUDR"


class DeadlinePriority(str, Enum):
    """Priority level for a regulatory deadline."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class DeadlineStatus(str, Enum):
    """Status of a deadline."""
    UPCOMING = "UPCOMING"
    APPROACHING = "APPROACHING"
    DUE_SOON = "DUE_SOON"
    OVERDUE = "OVERDUE"
    COMPLETED = "COMPLETED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AlertChannel(str, Enum):
    """Notification channel for alerts."""
    EMAIL = "EMAIL"
    SLACK = "SLACK"
    TEAMS = "TEAMS"
    DASHBOARD = "DASHBOARD"
    SMS = "SMS"


class DependencyType(str, Enum):
    """Type of cross-regulation dependency."""
    DATA_DEPENDENCY = "DATA_DEPENDENCY"
    SEQUENCE_DEPENDENCY = "SEQUENCE_DEPENDENCY"
    SHARED_DEADLINE = "SHARED_DEADLINE"
    PREREQUISITE = "PREREQUISITE"


# =============================================================================
# REGULATORY CALENDAR DATA
# =============================================================================


REGULATORY_DEADLINES: Dict[str, List[Dict[str, Any]]] = {
    RegulationPack.CSRD.value: [
        {"deadline_id": "CSRD-DM-01", "name": "Double materiality assessment completion", "month": 9, "day": 30, "offset_years": -1, "priority": "HIGH", "description": "Complete double materiality assessment for upcoming reporting year"},
        {"deadline_id": "CSRD-DC-01", "name": "Data collection deadline", "month": 1, "day": 31, "offset_years": 0, "priority": "HIGH", "description": "Close data collection for prior year metrics"},
        {"deadline_id": "CSRD-IR-01", "name": "Internal review deadline", "month": 2, "day": 28, "offset_years": 0, "priority": "MEDIUM", "description": "Complete internal review of sustainability statement"},
        {"deadline_id": "CSRD-EA-01", "name": "External assurance engagement", "month": 3, "day": 15, "offset_years": 0, "priority": "HIGH", "description": "External assurance provider engagement deadline"},
        {"deadline_id": "CSRD-AR-01", "name": "Annual report filing", "month": 4, "day": 30, "offset_years": 0, "priority": "CRITICAL", "description": "File annual report with sustainability statement"},
        {"deadline_id": "CSRD-BP-01", "name": "Board presentation", "month": 3, "day": 31, "offset_years": 0, "priority": "MEDIUM", "description": "Present sustainability results to board"},
    ],
    RegulationPack.CBAM.value: [
        {"deadline_id": "CBAM-Q1-01", "name": "Q1 CBAM report submission", "month": 4, "day": 30, "offset_years": 0, "priority": "HIGH", "description": "Submit Q1 transitional CBAM report"},
        {"deadline_id": "CBAM-Q2-01", "name": "Q2 CBAM report submission", "month": 7, "day": 31, "offset_years": 0, "priority": "HIGH", "description": "Submit Q2 transitional CBAM report"},
        {"deadline_id": "CBAM-Q3-01", "name": "Q3 CBAM report submission", "month": 10, "day": 31, "offset_years": 0, "priority": "HIGH", "description": "Submit Q3 transitional CBAM report"},
        {"deadline_id": "CBAM-Q4-01", "name": "Q4 CBAM report submission", "month": 1, "day": 31, "offset_years": 1, "priority": "HIGH", "description": "Submit Q4 transitional CBAM report"},
        {"deadline_id": "CBAM-AD-01", "name": "Annual CBAM declaration", "month": 5, "day": 31, "offset_years": 1, "priority": "CRITICAL", "description": "Submit annual CBAM declaration"},
        {"deadline_id": "CBAM-CP-01", "name": "Certificate purchase deadline", "month": 5, "day": 15, "offset_years": 1, "priority": "HIGH", "description": "Purchase required CBAM certificates"},
        {"deadline_id": "CBAM-CS-01", "name": "Certificate surrender deadline", "month": 5, "day": 31, "offset_years": 1, "priority": "CRITICAL", "description": "Surrender CBAM certificates"},
        {"deadline_id": "CBAM-SV-01", "name": "Supplier verification deadline", "month": 3, "day": 31, "offset_years": 0, "priority": "MEDIUM", "description": "Complete supplier emissions data verification"},
    ],
    RegulationPack.EU_TAXONOMY.value: [
        {"deadline_id": "TAX-EL-01", "name": "Eligibility screening completion", "month": 11, "day": 30, "offset_years": -1, "priority": "MEDIUM", "description": "Complete taxonomy eligibility screening"},
        {"deadline_id": "TAX-AL-01", "name": "Alignment assessment completion", "month": 1, "day": 31, "offset_years": 0, "priority": "HIGH", "description": "Complete substantial contribution and DNSH assessment"},
        {"deadline_id": "TAX-KP-01", "name": "KPI calculation deadline", "month": 2, "day": 28, "offset_years": 0, "priority": "HIGH", "description": "Calculate revenue/CapEx/OpEx KPIs"},
        {"deadline_id": "TAX-AR-01", "name": "Taxonomy disclosure in annual report", "month": 4, "day": 30, "offset_years": 0, "priority": "CRITICAL", "description": "Include taxonomy disclosures in annual report"},
        {"deadline_id": "TAX-MS-01", "name": "Minimum safeguards assessment", "month": 12, "day": 31, "offset_years": -1, "priority": "MEDIUM", "description": "Complete minimum social safeguards assessment"},
    ],
    RegulationPack.EUDR.value: [
        {"deadline_id": "EUDR-RA-01", "name": "Risk assessment update", "month": 3, "day": 31, "offset_years": 0, "priority": "HIGH", "description": "Update country and supplier risk assessments"},
        {"deadline_id": "EUDR-DD-01", "name": "Due diligence system review", "month": 6, "day": 30, "offset_years": 0, "priority": "MEDIUM", "description": "Annual review of due diligence system"},
        {"deadline_id": "EUDR-AR-01", "name": "Annual compliance review", "month": 12, "day": 31, "offset_years": 0, "priority": "HIGH", "description": "Annual EUDR compliance review and report"},
        {"deadline_id": "EUDR-SC-01", "name": "Supply chain mapping update", "month": 6, "day": 30, "offset_years": 0, "priority": "MEDIUM", "description": "Update supply chain traceability maps"},
        {"deadline_id": "EUDR-MS-01", "name": "Monitoring system check", "month": 9, "day": 30, "offset_years": 0, "priority": "LOW", "description": "Quarterly check of ongoing monitoring system"},
        {"deadline_id": "EUDR-TR-01", "name": "Training completion", "month": 3, "day": 31, "offset_years": 0, "priority": "LOW", "description": "Complete annual EUDR training for staff"},
    ],
}

CROSS_REGULATION_DEPENDENCIES: List[Dict[str, Any]] = [
    {"from_deadline": "CSRD-DC-01", "to_deadline": "TAX-KP-01", "type": "DATA_DEPENDENCY", "description": "Taxonomy KPIs depend on CSRD data collection"},
    {"from_deadline": "CSRD-DC-01", "to_deadline": "CSRD-IR-01", "type": "SEQUENCE_DEPENDENCY", "description": "Internal review follows data collection"},
    {"from_deadline": "CSRD-IR-01", "to_deadline": "CSRD-EA-01", "type": "SEQUENCE_DEPENDENCY", "description": "External assurance follows internal review"},
    {"from_deadline": "CSRD-EA-01", "to_deadline": "CSRD-AR-01", "type": "PREREQUISITE", "description": "Annual report requires assurance completion"},
    {"from_deadline": "TAX-AL-01", "to_deadline": "TAX-KP-01", "type": "SEQUENCE_DEPENDENCY", "description": "KPI calculation follows alignment assessment"},
    {"from_deadline": "TAX-KP-01", "to_deadline": "TAX-AR-01", "type": "PREREQUISITE", "description": "Taxonomy disclosure needs KPIs"},
    {"from_deadline": "CSRD-AR-01", "to_deadline": "TAX-AR-01", "type": "SHARED_DEADLINE", "description": "Both included in same annual report"},
    {"from_deadline": "CBAM-SV-01", "to_deadline": "CBAM-Q1-01", "type": "DATA_DEPENDENCY", "description": "Q1 report depends on verified supplier data"},
    {"from_deadline": "CBAM-CP-01", "to_deadline": "CBAM-CS-01", "type": "SEQUENCE_DEPENDENCY", "description": "Must purchase before surrendering certificates"},
    {"from_deadline": "EUDR-RA-01", "to_deadline": "EUDR-DD-01", "type": "DATA_DEPENDENCY", "description": "Due diligence review uses risk assessment"},
    {"from_deadline": "CBAM-SV-01", "to_deadline": "EUDR-SC-01", "type": "DATA_DEPENDENCY", "description": "EUDR supply chain mapping can use CBAM supplier data"},
]

ALERT_SCHEDULE: List[Dict[str, Any]] = [
    {"days_before": 90, "alert_level": "INFO", "channel": "DASHBOARD"},
    {"days_before": 60, "alert_level": "LOW", "channel": "EMAIL"},
    {"days_before": 30, "alert_level": "MEDIUM", "channel": "EMAIL"},
    {"days_before": 14, "alert_level": "HIGH", "channel": "EMAIL"},
    {"days_before": 7, "alert_level": "HIGH", "channel": "SLACK"},
    {"days_before": 3, "alert_level": "CRITICAL", "channel": "EMAIL"},
    {"days_before": 1, "alert_level": "CRITICAL", "channel": "SMS"},
    {"days_before": 0, "alert_level": "CRITICAL", "channel": "EMAIL"},
]


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
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
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class WorkflowConfig(BaseModel):
    """Configuration for calendar management workflow."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2024, le=2050)
    target_packs: List[RegulationPack] = Field(
        default_factory=lambda: list(RegulationPack)
    )
    reference_date: Optional[str] = Field(
        None,
        description="ISO date string for deadline calculations. Defaults to now."
    )
    completed_deadlines: List[str] = Field(
        default_factory=list,
        description="List of deadline_ids already completed"
    )
    alert_channels: List[AlertChannel] = Field(
        default_factory=lambda: [AlertChannel.EMAIL, AlertChannel.DASHBOARD]
    )
    notification_recipients: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Pack -> list of email/channel recipients"
    )
    skip_phases: List[str] = Field(default_factory=list)


class CalendarManagementResult(WorkflowResult):
    """Result from calendar management workflow."""
    total_deadlines: int = Field(default=0)
    upcoming_deadlines: int = Field(default=0)
    overdue_deadlines: int = Field(default=0)
    dependencies_identified: int = Field(default=0)
    alerts_scheduled: int = Field(default=0)


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class CalendarManagementWorkflow:
    """
    Three-phase calendar management workflow.

    Populates regulatory deadlines, analyzes dependencies, and
    distributes alerts to compliance teams.

    Example:
        >>> wf = CalendarManagementWorkflow()
        >>> config = WorkflowConfig(
        ...     organization_id="org-123",
        ...     reporting_year=2026,
        ... )
        >>> result = wf.execute(config)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "calendar_management"

    PHASE_ORDER = [
        "calendar_population",
        "dependency_analysis",
        "alert_distribution",
    ]

    def __init__(self) -> None:
        """Initialize the calendar management workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._phase_outputs: Dict[str, Dict[str, Any]] = {}

    def execute(self, config: WorkflowConfig) -> CalendarManagementResult:
        """
        Execute the three-phase calendar management workflow.

        Args:
            config: Validated workflow configuration.

        Returns:
            CalendarManagementResult with calendar outcomes.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting calendar management %s for org=%s year=%d",
            self.workflow_id, config.organization_id, config.reporting_year,
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING
        phase_methods = {
            "calendar_population": self._phase_calendar_population,
            "dependency_analysis": self._phase_dependency_analysis,
            "alert_distribution": self._phase_alert_distribution,
        }

        for phase_name in self.PHASE_ORDER:
            if phase_name in config.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                continue

            try:
                phase_result = phase_methods[phase_name](config)
                completed_phases.append(phase_result)
                if phase_result.status == PhaseStatus.COMPLETED:
                    self._phase_outputs[phase_name] = phase_result.outputs
                elif phase_result.status == PhaseStatus.FAILED:
                    overall_status = WorkflowStatus.FAILED
                    break
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = datetime.utcnow()
        summary = self._build_summary()
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        return CalendarManagementResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=(completed_at - started_at).total_seconds(),
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            total_deadlines=summary.get("total_deadlines", 0),
            upcoming_deadlines=summary.get("upcoming_deadlines", 0),
            overdue_deadlines=summary.get("overdue_deadlines", 0),
            dependencies_identified=summary.get("dependencies_identified", 0),
            alerts_scheduled=summary.get("alerts_scheduled", 0),
        )

    # -------------------------------------------------------------------------
    # Phase 1: Calendar Population
    # -------------------------------------------------------------------------

    def _phase_calendar_population(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 1: Populate deadlines from all 4 regulatory frameworks.

        Resolves deadline dates based on reporting year, marks completed
        items, and classifies each by urgency status.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            ref_date = self._parse_reference_date(config.reference_date)
            year = config.reporting_year

            all_deadlines: List[Dict[str, Any]] = []
            per_pack_counts: Dict[str, int] = {}

            for pack in config.target_packs:
                pack_name = pack.value
                pack_deadlines = REGULATORY_DEADLINES.get(pack_name, [])
                per_pack_counts[pack_name] = len(pack_deadlines)

                for dl in pack_deadlines:
                    resolved = self._resolve_deadline(dl, year, ref_date, config)
                    resolved["pack"] = pack_name
                    all_deadlines.append(resolved)

            all_deadlines.sort(key=lambda d: d.get("deadline_date", "9999-12-31"))

            overdue = [d for d in all_deadlines if d["status"] == DeadlineStatus.OVERDUE.value]
            due_soon = [d for d in all_deadlines if d["status"] == DeadlineStatus.DUE_SOON.value]
            approaching = [d for d in all_deadlines if d["status"] == DeadlineStatus.APPROACHING.value]
            upcoming = [d for d in all_deadlines if d["status"] == DeadlineStatus.UPCOMING.value]
            completed = [d for d in all_deadlines if d["status"] == DeadlineStatus.COMPLETED.value]

            outputs["all_deadlines"] = all_deadlines
            outputs["total_deadlines"] = len(all_deadlines)
            outputs["per_pack_counts"] = per_pack_counts
            outputs["overdue_count"] = len(overdue)
            outputs["due_soon_count"] = len(due_soon)
            outputs["approaching_count"] = len(approaching)
            outputs["upcoming_count"] = len(upcoming)
            outputs["completed_count"] = len(completed)
            outputs["reference_date"] = ref_date.isoformat()

            next_deadlines: List[Dict[str, Any]] = []
            active = [d for d in all_deadlines if d["status"] not in (
                DeadlineStatus.COMPLETED.value, DeadlineStatus.NOT_APPLICABLE.value
            )]
            next_deadlines = active[:5] if active else []
            outputs["next_5_deadlines"] = [
                {
                    "deadline_id": d["deadline_id"],
                    "name": d["name"],
                    "pack": d["pack"],
                    "deadline_date": d["deadline_date"],
                    "days_remaining": d["days_remaining"],
                    "priority": d["priority"],
                }
                for d in next_deadlines
            ]

            if overdue:
                warnings.append(
                    f"{len(overdue)} deadlines are overdue: "
                    + ", ".join(d["deadline_id"] for d in overdue[:3])
                )

            logger.info(
                "Calendar populated: %d deadlines, %d overdue, %d due soon",
                len(all_deadlines), len(overdue), len(due_soon),
            )

            status = PhaseStatus.COMPLETED
            records = len(all_deadlines)

        except Exception as exc:
            logger.error("Calendar population failed: %s", exc, exc_info=True)
            errors.append(f"Calendar population failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="calendar_population",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _parse_reference_date(self, ref_date_str: Optional[str]) -> datetime:
        """Parse reference date string or default to now."""
        if ref_date_str:
            try:
                return datetime.fromisoformat(ref_date_str.replace("Z", "+00:00"))
            except ValueError:
                logger.warning("Invalid reference date '%s', using now", ref_date_str)
        return datetime.utcnow()

    def _resolve_deadline(
        self,
        deadline_def: Dict[str, Any],
        reporting_year: int,
        ref_date: datetime,
        config: WorkflowConfig,
    ) -> Dict[str, Any]:
        """Resolve a deadline definition to a concrete dated entry."""
        dl_id = deadline_def["deadline_id"]
        offset = deadline_def.get("offset_years", 0)
        target_year = reporting_year + offset
        month = deadline_def["month"]
        day = deadline_def["day"]

        if month == 2 and day > 28:
            try:
                deadline_date = datetime(target_year, month, day)
            except ValueError:
                deadline_date = datetime(target_year, month, 28)
        else:
            deadline_date = datetime(target_year, month, day)

        days_remaining = (deadline_date - ref_date).days

        if dl_id in config.completed_deadlines:
            dl_status = DeadlineStatus.COMPLETED.value
        elif days_remaining < 0:
            dl_status = DeadlineStatus.OVERDUE.value
        elif days_remaining <= 7:
            dl_status = DeadlineStatus.DUE_SOON.value
        elif days_remaining <= 30:
            dl_status = DeadlineStatus.APPROACHING.value
        else:
            dl_status = DeadlineStatus.UPCOMING.value

        return {
            "deadline_id": dl_id,
            "name": deadline_def["name"],
            "description": deadline_def["description"],
            "deadline_date": deadline_date.strftime("%Y-%m-%d"),
            "days_remaining": days_remaining,
            "priority": deadline_def["priority"],
            "status": dl_status,
            "reporting_year": reporting_year,
            "target_year": target_year,
        }

    # -------------------------------------------------------------------------
    # Phase 2: Dependency Analysis
    # -------------------------------------------------------------------------

    def _phase_dependency_analysis(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 2: Identify cross-regulation dependencies and critical path.

        Analyzes the dependency graph between deadlines to find the
        critical path and identify blockers.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            calendar_out = self._phase_outputs.get("calendar_population", {})
            all_deadlines = calendar_out.get("all_deadlines", [])
            deadline_lookup = {d["deadline_id"]: d for d in all_deadlines}

            active_packs = {p.value for p in config.target_packs}
            relevant_deps: List[Dict[str, Any]] = []
            for dep in CROSS_REGULATION_DEPENDENCIES:
                from_id = dep["from_deadline"]
                to_id = dep["to_deadline"]
                if from_id in deadline_lookup and to_id in deadline_lookup:
                    from_dl = deadline_lookup[from_id]
                    to_dl = deadline_lookup[to_id]
                    enriched_dep = dict(dep)
                    enriched_dep["from_name"] = from_dl["name"]
                    enriched_dep["to_name"] = to_dl["name"]
                    enriched_dep["from_pack"] = from_dl.get("pack", "")
                    enriched_dep["to_pack"] = to_dl.get("pack", "")
                    enriched_dep["from_date"] = from_dl["deadline_date"]
                    enriched_dep["to_date"] = to_dl["deadline_date"]
                    enriched_dep["from_status"] = from_dl["status"]
                    enriched_dep["to_status"] = to_dl["status"]

                    is_cross_pack = enriched_dep["from_pack"] != enriched_dep["to_pack"]
                    enriched_dep["is_cross_regulation"] = is_cross_pack

                    relevant_deps.append(enriched_dep)

            blocker_deps = []
            for dep in relevant_deps:
                from_status = dep["from_status"]
                to_status = dep["to_status"]
                if from_status not in (DeadlineStatus.COMPLETED.value,) and \
                   to_status in (DeadlineStatus.DUE_SOON.value, DeadlineStatus.APPROACHING.value):
                    blocker_deps.append({
                        "blocker": dep["from_deadline"],
                        "blocked": dep["to_deadline"],
                        "blocker_name": dep["from_name"],
                        "blocked_name": dep["to_name"],
                        "dependency_type": dep["type"],
                        "risk": "HIGH" if to_status == DeadlineStatus.DUE_SOON.value else "MEDIUM",
                    })

            critical_path = self._compute_critical_path(
                all_deadlines, relevant_deps, deadline_lookup
            )

            outputs["dependencies"] = relevant_deps
            outputs["dependencies_count"] = len(relevant_deps)
            outputs["cross_regulation_deps"] = sum(
                1 for d in relevant_deps if d.get("is_cross_regulation")
            )
            outputs["blockers"] = blocker_deps
            outputs["blocker_count"] = len(blocker_deps)
            outputs["critical_path"] = critical_path

            if blocker_deps:
                warnings.append(
                    f"{len(blocker_deps)} blocking dependencies found: "
                    + ", ".join(b["blocker"] for b in blocker_deps[:3])
                )

            logger.info(
                "Dependency analysis complete: %d deps, %d blockers, critical path length=%d",
                len(relevant_deps), len(blocker_deps), len(critical_path),
            )

            status = PhaseStatus.COMPLETED
            records = len(relevant_deps)

        except Exception as exc:
            logger.error("Dependency analysis failed: %s", exc, exc_info=True)
            errors.append(f"Dependency analysis failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="dependency_analysis",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _compute_critical_path(
        self,
        deadlines: List[Dict[str, Any]],
        dependencies: List[Dict[str, Any]],
        deadline_lookup: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Compute the critical path through the deadline dependency graph.

        Uses a simplified topological approach: finds the longest chain
        of sequential dependencies ending at the latest deadline.
        """
        graph: Dict[str, List[str]] = {}
        for dep in dependencies:
            from_id = dep["from_deadline"]
            to_id = dep["to_deadline"]
            if from_id not in graph:
                graph[from_id] = []
            graph[from_id].append(to_id)

        all_nodes = set()
        for dep in dependencies:
            all_nodes.add(dep["from_deadline"])
            all_nodes.add(dep["to_deadline"])

        longest_path: List[str] = []

        def dfs(node: str, path: List[str]) -> None:
            nonlocal longest_path
            current_path = path + [node]
            if len(current_path) > len(longest_path):
                longest_path = current_path
            for neighbor in graph.get(node, []):
                if neighbor not in path:
                    dfs(neighbor, current_path)

        targets = set()
        for dep in dependencies:
            targets.add(dep["to_deadline"])
        roots = all_nodes - targets

        for root in roots:
            dfs(root, [])

        if not longest_path and all_nodes:
            longest_path = [sorted(all_nodes, key=lambda n: deadline_lookup.get(n, {}).get("deadline_date", ""))[0]]

        critical_path_entries: List[Dict[str, Any]] = []
        for node_id in longest_path:
            dl = deadline_lookup.get(node_id, {})
            critical_path_entries.append({
                "deadline_id": node_id,
                "name": dl.get("name", ""),
                "pack": dl.get("pack", ""),
                "deadline_date": dl.get("deadline_date", ""),
                "status": dl.get("status", ""),
            })

        return critical_path_entries

    # -------------------------------------------------------------------------
    # Phase 3: Alert Distribution
    # -------------------------------------------------------------------------

    def _phase_alert_distribution(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 3: Set up alerts and send notifications.

        Creates alert schedules for each upcoming deadline and
        distributes immediate notifications for urgent items.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            calendar_out = self._phase_outputs.get("calendar_population", {})
            dep_out = self._phase_outputs.get("dependency_analysis", {})
            all_deadlines = calendar_out.get("all_deadlines", [])
            blockers = dep_out.get("blockers", [])

            active_deadlines = [
                d for d in all_deadlines
                if d["status"] not in (
                    DeadlineStatus.COMPLETED.value,
                    DeadlineStatus.NOT_APPLICABLE.value,
                )
            ]

            scheduled_alerts: List[Dict[str, Any]] = []
            for dl in active_deadlines:
                days_rem = dl["days_remaining"]
                dl_alerts = []

                for alert_def in ALERT_SCHEDULE:
                    alert_days = alert_def["days_before"]
                    alert_channel = alert_def["channel"]

                    if alert_channel not in [c.value for c in config.alert_channels]:
                        continue

                    alert_date_offset = days_rem - alert_days
                    if alert_date_offset >= 0:
                        should_send_now = alert_date_offset == 0
                    else:
                        should_send_now = days_rem <= alert_days

                    pack_name = dl.get("pack", "")
                    recipients = config.notification_recipients.get(
                        pack_name,
                        config.notification_recipients.get("default", []),
                    )

                    alert_entry = {
                        "alert_id": str(uuid.uuid4()),
                        "deadline_id": dl["deadline_id"],
                        "deadline_name": dl["name"],
                        "pack": pack_name,
                        "alert_level": alert_def["alert_level"],
                        "channel": alert_channel,
                        "days_before_deadline": alert_days,
                        "scheduled_send": should_send_now,
                        "recipients": recipients,
                        "message": (
                            f"[{alert_def['alert_level']}] {dl['name']} "
                            f"due in {days_rem} days ({dl['deadline_date']})"
                        ),
                    }
                    dl_alerts.append(alert_entry)

                scheduled_alerts.extend(dl_alerts)

            immediate_alerts = [a for a in scheduled_alerts if a["scheduled_send"]]

            blocker_notifications: List[Dict[str, Any]] = []
            for blocker in blockers:
                blocker_notifications.append({
                    "notification_id": str(uuid.uuid4()),
                    "type": "DEPENDENCY_BLOCKER",
                    "message": (
                        f"Blocker: '{blocker['blocker_name']}' must complete "
                        f"before '{blocker['blocked_name']}' can proceed"
                    ),
                    "risk": blocker["risk"],
                    "channels": [c.value for c in config.alert_channels],
                    "sent_at": datetime.utcnow().isoformat(),
                })

            outputs["scheduled_alerts"] = scheduled_alerts
            outputs["total_alerts_scheduled"] = len(scheduled_alerts)
            outputs["immediate_alerts"] = immediate_alerts
            outputs["immediate_count"] = len(immediate_alerts)
            outputs["blocker_notifications"] = blocker_notifications
            outputs["blocker_notification_count"] = len(blocker_notifications)

            outputs["alert_summary_by_channel"] = {}
            for channel in config.alert_channels:
                channel_alerts = [
                    a for a in scheduled_alerts if a["channel"] == channel.value
                ]
                outputs["alert_summary_by_channel"][channel.value] = len(channel_alerts)

            logger.info(
                "Alert distribution complete: %d scheduled, %d immediate, %d blocker notifications",
                len(scheduled_alerts), len(immediate_alerts), len(blocker_notifications),
            )

            status = PhaseStatus.COMPLETED
            records = len(scheduled_alerts)

        except Exception as exc:
            logger.error("Alert distribution failed: %s", exc, exc_info=True)
            errors.append(f"Alert distribution failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="alert_distribution",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        """Build workflow summary from all phase outputs."""
        calendar = self._phase_outputs.get("calendar_population", {})
        deps = self._phase_outputs.get("dependency_analysis", {})
        alerts = self._phase_outputs.get("alert_distribution", {})

        return {
            "total_deadlines": calendar.get("total_deadlines", 0),
            "upcoming_deadlines": calendar.get("upcoming_count", 0),
            "overdue_deadlines": calendar.get("overdue_count", 0),
            "due_soon_deadlines": calendar.get("due_soon_count", 0),
            "approaching_deadlines": calendar.get("approaching_count", 0),
            "completed_deadlines": calendar.get("completed_count", 0),
            "dependencies_identified": deps.get("dependencies_count", 0),
            "cross_regulation_deps": deps.get("cross_regulation_deps", 0),
            "blockers": deps.get("blocker_count", 0),
            "alerts_scheduled": alerts.get("total_alerts_scheduled", 0),
            "immediate_alerts": alerts.get("immediate_count", 0),
        }


# =============================================================================
# UTILITIES
# =============================================================================


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
