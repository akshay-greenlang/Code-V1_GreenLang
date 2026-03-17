# -*- coding: utf-8 -*-
"""
Remediation Planning Workflow - PACK-018 EU Green Claims Prep
==============================================================

4-phase workflow that triages non-compliant environmental claims, plans
resource allocation for remediation, generates per-claim timelines, and
establishes progress tracking with KPIs and checkpoints.

Phases:
    1. ClaimTriage          -- Categorise non-compliant claims (withdraw/reword/substantiate/replace)
    2. ResourcePlanning     -- Estimate effort, cost, and personnel per action
    3. TimelineGeneration   -- Build per-claim remediation timeline
    4. ProgressTracking     -- Set up KPIs, checkpoints, and monitoring

Reference:
    EU Green Claims Directive (COM/2023/166)
    PACK-018 Solution Pack specification

Author: GreenLang Team
Version: 18.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID-4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hex digest for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Execution status for a single workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RemediationPhase(str, Enum):
    """Remediation planning workflow phase identifiers."""
    CLAIM_TRIAGE = "ClaimTriage"
    RESOURCE_PLANNING = "ResourcePlanning"
    TIMELINE_GENERATION = "TimelineGeneration"
    PROGRESS_TRACKING = "ProgressTracking"


class TriageCategory(str, Enum):
    """Triage category for non-compliant claims."""
    WITHDRAW = "withdraw"
    REWORD = "reword"
    SUBSTANTIATE = "substantiate"
    REPLACE = "replace"


class SeverityLevel(str, Enum):
    """Issue severity classification."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TrackingStatus(str, Enum):
    """Progress tracking status for individual actions."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    OVERDUE = "overdue"


# =============================================================================
# DATA MODELS
# =============================================================================


class RemediationConfig(BaseModel):
    """Configuration for RemediationPlanningWorkflow."""
    working_days_per_month: int = Field(
        default=22, ge=15, le=25,
        description="Working days per month for timeline calculation",
    )
    max_parallel_actions: int = Field(
        default=5, ge=1, le=20,
        description="Maximum actions that can run in parallel",
    )
    default_fte_available: float = Field(
        default=2.0, ge=0.5, le=50.0,
        description="Default full-time equivalents available for remediation",
    )


class RemediationResult(BaseModel):
    """Individual claim remediation plan result."""
    claim_id: str = Field(..., description="Unique claim identifier")
    triage_category: str = Field(..., description="Triage classification")
    severity: str = Field(..., description="Issue severity level")
    actions: List[str] = Field(default_factory=list)
    estimated_effort_days: int = Field(default=0, ge=0)
    estimated_cost_eur: int = Field(default=0, ge=0)
    target_start_month: int = Field(default=1, ge=1)
    target_completion_month: int = Field(default=1, ge=1)
    tracking_status: str = Field(default="not_started")


class WorkflowInput(BaseModel):
    """Input model for RemediationPlanningWorkflow."""
    non_compliant_claims: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of non-compliant claim objects to remediate",
    )
    available_resources: Dict[str, Any] = Field(
        default_factory=dict,
        description="Available resources (budget_eur, fte_available, etc.)",
    )
    entity_name: str = Field(default="", description="Reporting entity name")
    target_months: int = Field(
        default=12, ge=1, le=60,
        description="Target months to complete all remediation",
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    result_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = Field(default=None)


class WorkflowResult(BaseModel):
    """Complete result from RemediationPlanningWorkflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="remediation_planning")
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    phases: List[PhaseResult] = Field(default_factory=list)
    overall_result: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RemediationPlanningWorkflow:
    """
    4-phase remediation planning workflow for EU Green Claims Directive.

    Triages non-compliant claims into action categories, plans resource
    allocation, generates per-claim timelines, and establishes progress
    tracking with KPIs and monitoring checkpoints.

    Zero-hallucination: all triage classification, cost estimation, and
    timeline computation uses deterministic rules and arithmetic. No LLM
    calls in calculation paths.

    Example:
        >>> wf = RemediationPlanningWorkflow()
        >>> result = wf.execute(
        ...     non_compliant_claims=[
        ...         {"id": "C1", "text": "eco-friendly", "severity": "high",
        ...          "issue": "vagueness"},
        ...     ],
        ...     available_resources={"budget_eur": 100000, "fte_available": 3},
        ... )
        >>> assert result["status"] == "completed"
    """

    WORKFLOW_NAME: str = "remediation_planning"

    # Cost estimates per triage category (EUR)
    CATEGORY_COSTS: Dict[str, int] = {
        TriageCategory.WITHDRAW.value: 500,
        TriageCategory.REWORD.value: 3000,
        TriageCategory.SUBSTANTIATE.value: 25000,
        TriageCategory.REPLACE.value: 15000,
    }

    # Duration estimates per triage category (working days)
    CATEGORY_DURATIONS: Dict[str, int] = {
        TriageCategory.WITHDRAW.value: 3,
        TriageCategory.REWORD.value: 10,
        TriageCategory.SUBSTANTIATE.value: 60,
        TriageCategory.REPLACE.value: 30,
    }

    # Triage classification keywords
    WITHDRAW_INDICATORS: set = {
        "fibbing", "false_labels", "fabricated", "fraudulent",
    }
    REWORD_INDICATORS: set = {
        "vagueness", "generic_claim", "misleading", "ambiguous",
    }
    SUBSTANTIATE_INDICATORS: set = {
        "no_proof", "insufficient_evidence", "unverified", "missing_lca",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RemediationPlanningWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self.rem_config = RemediationConfig(**self.config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> dict:
        """
        Execute the 4-phase remediation planning pipeline.

        Keyword Args:
            non_compliant_claims: List of claim dicts with severity and issue info.
            available_resources: Dict of budget/FTE resources.
            target_months: Completion target in months.

        Returns:
            Serialised WorkflowResult dictionary with provenance hash.
        """
        input_data = WorkflowInput(
            non_compliant_claims=kwargs.get("non_compliant_claims", []),
            available_resources=kwargs.get("available_resources", {}),
            entity_name=kwargs.get("entity_name", ""),
            target_months=kwargs.get("target_months", 12),
            config=kwargs.get("config", {}),
        )

        started_at = _utcnow()
        self.logger.info("Starting %s workflow %s -- %d claims",
                         self.WORKFLOW_NAME, self.workflow_id,
                         len(input_data.non_compliant_claims))
        phase_results: List[PhaseResult] = []
        overall_status = PhaseStatus.RUNNING

        try:
            # Phase 1 -- Claim Triage
            phase_results.append(self._run_claim_triage(input_data))

            # Phase 2 -- Resource Planning
            triage_data = phase_results[0].result_data
            phase_results.append(
                self._run_resource_planning(input_data, triage_data)
            )

            # Phase 3 -- Timeline Generation
            resource_data = phase_results[1].result_data
            phase_results.append(
                self._run_timeline_generation(input_data, triage_data, resource_data)
            )

            # Phase 4 -- Progress Tracking
            timeline_data = phase_results[2].result_data
            phase_results.append(
                self._run_progress_tracking(input_data, timeline_data)
            )

            overall_status = PhaseStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Workflow %s failed: %s", self.workflow_id, exc, exc_info=True)
            overall_status = PhaseStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error_capture",
                status=PhaseStatus.FAILED,
                started_at=_utcnow(),
                completed_at=_utcnow(),
                error_message=str(exc),
            ))

        completed_at = _utcnow()

        completed_phases = [p for p in phase_results if p.status == PhaseStatus.COMPLETED]
        overall_result: Dict[str, Any] = {
            "total_claims": len(input_data.non_compliant_claims),
            "phases_completed": len(completed_phases),
            "phases_total": 4,
        }
        if phase_results and phase_results[-1].status == PhaseStatus.COMPLETED:
            overall_result.update(phase_results[-1].result_data)

        result = WorkflowResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            phases=phase_results,
            overall_result=overall_result,
            started_at=started_at,
            completed_at=completed_at,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Workflow %s %s in %.1fs -- %d claims triaged",
            self.workflow_id,
            overall_status.value,
            (completed_at - started_at).total_seconds(),
            len(input_data.non_compliant_claims),
        )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # run_phase dispatcher
    # ------------------------------------------------------------------

    def run_phase(self, phase: RemediationPhase, **kwargs: Any) -> PhaseResult:
        """
        Run a single named phase independently.

        Args:
            phase: The RemediationPhase to execute.
            **kwargs: Phase-specific keyword arguments.

        Returns:
            PhaseResult for the executed phase.
        """
        dispatch: Dict[RemediationPhase, Any] = {
            RemediationPhase.CLAIM_TRIAGE: lambda: self._run_claim_triage(
                WorkflowInput(non_compliant_claims=kwargs.get("non_compliant_claims", []))
            ),
            RemediationPhase.RESOURCE_PLANNING: lambda: self._run_resource_planning(
                WorkflowInput(available_resources=kwargs.get("available_resources", {})),
                kwargs.get("triage_data", {}),
            ),
            RemediationPhase.TIMELINE_GENERATION: lambda: self._run_timeline_generation(
                WorkflowInput(target_months=kwargs.get("target_months", 12)),
                kwargs.get("triage_data", {}),
                kwargs.get("resource_data", {}),
            ),
            RemediationPhase.PROGRESS_TRACKING: lambda: self._run_progress_tracking(
                WorkflowInput(),
                kwargs.get("timeline_data", {}),
            ),
        }
        handler = dispatch.get(phase)
        if handler is None:
            return PhaseResult(
                phase_name=phase.value,
                status=PhaseStatus.FAILED,
                error_message=f"Unknown phase: {phase.value}",
            )
        return handler()

    # ------------------------------------------------------------------
    # Phase 1: Claim Triage
    # ------------------------------------------------------------------

    def _run_claim_triage(self, input_data: WorkflowInput) -> PhaseResult:
        """Categorise non-compliant claims into triage categories."""
        started = _utcnow()
        self.logger.info("Phase 1/4 ClaimTriage -- triaging %d claims",
                         len(input_data.non_compliant_claims))

        triaged: List[Dict[str, Any]] = []
        category_counts: Dict[str, int] = {c.value: 0 for c in TriageCategory}
        severity_counts: Dict[str, int] = {s.value: 0 for s in SeverityLevel}

        for idx, claim in enumerate(input_data.non_compliant_claims):
            claim_id = claim.get("id", f"CLM-{idx:04d}")
            severity = self._normalise_severity(claim.get("severity", "medium"))
            issue = claim.get("issue", "").lower()
            category = self._classify_triage(severity, issue)

            category_counts[category.value] += 1
            severity_counts[severity.value] += 1

            actions = self._get_triage_actions(category)

            triaged.append({
                "claim_id": claim_id,
                "text": claim.get("text", ""),
                "severity": severity.value,
                "issue": issue,
                "triage_category": category.value,
                "actions": actions,
                "action_count": len(actions),
                "estimated_cost_eur": self.CATEGORY_COSTS.get(category.value, 5000),
                "estimated_duration_days": self.CATEGORY_DURATIONS.get(category.value, 10),
            })

        # Sort by severity then cost (most expensive first)
        severity_order = {
            SeverityLevel.CRITICAL.value: 0,
            SeverityLevel.HIGH.value: 1,
            SeverityLevel.MEDIUM.value: 2,
            SeverityLevel.LOW.value: 3,
        }
        triaged.sort(key=lambda x: (severity_order.get(x["severity"], 9), -x["estimated_cost_eur"]))

        result_data: Dict[str, Any] = {
            "triaged_claims": triaged,
            "total_triaged": len(triaged),
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "withdraw_count": category_counts.get(TriageCategory.WITHDRAW.value, 0),
            "reword_count": category_counts.get(TriageCategory.REWORD.value, 0),
            "substantiate_count": category_counts.get(TriageCategory.SUBSTANTIATE.value, 0),
            "replace_count": category_counts.get(TriageCategory.REPLACE.value, 0),
        }

        return PhaseResult(
            phase_name=RemediationPhase.CLAIM_TRIAGE.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 2: Resource Planning
    # ------------------------------------------------------------------

    def _run_resource_planning(
        self, input_data: WorkflowInput, triage_data: Dict[str, Any],
    ) -> PhaseResult:
        """Estimate effort, cost, and personnel requirements."""
        started = _utcnow()
        self.logger.info("Phase 2/4 ResourcePlanning -- estimating resources")

        budget_available = input_data.available_resources.get("budget_eur", 0)
        fte_available = input_data.available_resources.get(
            "fte_available", self.rem_config.default_fte_available,
        )

        total_cost = sum(
            c["estimated_cost_eur"] for c in triage_data.get("triaged_claims", [])
        )
        total_days = sum(
            c["estimated_duration_days"] for c in triage_data.get("triaged_claims", [])
        )

        fte_days_per_year = fte_available * self.rem_config.working_days_per_month * 12
        budget_sufficient = budget_available >= total_cost if budget_available > 0 else False
        fte_sufficient = fte_days_per_year >= total_days

        cost_by_category: Dict[str, int] = {}
        for claim in triage_data.get("triaged_claims", []):
            cat = claim["triage_category"]
            cost_by_category[cat] = cost_by_category.get(cat, 0) + claim["estimated_cost_eur"]

        result_data: Dict[str, Any] = {
            "total_estimated_cost_eur": total_cost,
            "total_estimated_days": total_days,
            "budget_available_eur": budget_available,
            "budget_sufficient": budget_sufficient,
            "budget_gap_eur": max(0, total_cost - budget_available) if budget_available > 0 else total_cost,
            "fte_available": fte_available,
            "fte_days_available_yearly": round(fte_days_per_year, 0),
            "fte_sufficient": fte_sufficient,
            "cost_by_category": cost_by_category,
            "overall_feasible": budget_sufficient and fte_sufficient,
            "external_services_needed": self._count_external_services(triage_data),
        }

        return PhaseResult(
            phase_name=RemediationPhase.RESOURCE_PLANNING.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 3: Timeline Generation
    # ------------------------------------------------------------------

    def _run_timeline_generation(
        self,
        input_data: WorkflowInput,
        triage_data: Dict[str, Any],
        resource_data: Dict[str, Any],
    ) -> PhaseResult:
        """Build per-claim remediation timeline."""
        started = _utcnow()
        self.logger.info("Phase 3/4 TimelineGeneration -- building timelines")

        target_months = input_data.target_months
        working_days_month = self.rem_config.working_days_per_month
        triaged = triage_data.get("triaged_claims", [])

        timeline_items: List[Dict[str, Any]] = []
        current_month = 1

        # Schedule withdrawals first (immediate), then rewords, then
        # substantiate/replace based on severity ordering
        for claim in triaged:
            duration_days = claim["estimated_duration_days"]
            duration_months = max(1, (duration_days + working_days_month - 1) // working_days_month)
            start_month = self._assign_start_month(claim["triage_category"], current_month)
            end_month = min(start_month + duration_months - 1, target_months)

            timeline_items.append({
                "claim_id": claim["claim_id"],
                "triage_category": claim["triage_category"],
                "severity": claim["severity"],
                "start_month": start_month,
                "end_month": end_month,
                "duration_months": duration_months,
                "duration_days": duration_days,
                "actions": claim["actions"],
            })

        milestones: List[Dict[str, Any]] = [
            {
                "month": 1,
                "milestone": "All claim withdrawals initiated",
                "category": TriageCategory.WITHDRAW.value,
            },
            {
                "month": min(3, target_months),
                "milestone": "All claim rewordings completed",
                "category": TriageCategory.REWORD.value,
            },
            {
                "month": min(6, target_months),
                "milestone": "Substantiation evidence collection complete",
                "category": TriageCategory.SUBSTANTIATE.value,
            },
            {
                "month": target_months,
                "milestone": "All remediation actions completed",
                "category": "all",
            },
        ]

        result_data: Dict[str, Any] = {
            "timeline_items": timeline_items,
            "total_items": len(timeline_items),
            "milestones": milestones,
            "target_months": target_months,
            "earliest_start": min(
                (t["start_month"] for t in timeline_items), default=1
            ),
            "latest_end": max(
                (t["end_month"] for t in timeline_items), default=target_months
            ),
        }

        return PhaseResult(
            phase_name=RemediationPhase.TIMELINE_GENERATION.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 4: Progress Tracking
    # ------------------------------------------------------------------

    def _run_progress_tracking(
        self, input_data: WorkflowInput, timeline_data: Dict[str, Any],
    ) -> PhaseResult:
        """Set up KPIs, checkpoints, and progress monitoring."""
        started = _utcnow()
        self.logger.info("Phase 4/4 ProgressTracking -- configuring monitoring")

        total_items = timeline_data.get("total_items", 0)

        kpis: List[Dict[str, Any]] = [
            {
                "kpi_id": "KPI-REM-001",
                "name": "Claims Remediated Rate",
                "description": "Percentage of non-compliant claims with completed remediation",
                "target_value": 100.0,
                "current_value": 0.0,
                "unit": "percent",
                "frequency": "monthly",
            },
            {
                "kpi_id": "KPI-REM-002",
                "name": "Withdrawal Completion Rate",
                "description": "Percentage of withdrawal actions completed on time",
                "target_value": 100.0,
                "current_value": 0.0,
                "unit": "percent",
                "frequency": "weekly",
            },
            {
                "kpi_id": "KPI-REM-003",
                "name": "Budget Utilisation",
                "description": "Percentage of remediation budget spent vs allocated",
                "target_value": 100.0,
                "current_value": 0.0,
                "unit": "percent",
                "frequency": "monthly",
            },
            {
                "kpi_id": "KPI-REM-004",
                "name": "Evidence Completeness",
                "description": "Percentage of substantiation claims with full evidence dossier",
                "target_value": 100.0,
                "current_value": 0.0,
                "unit": "percent",
                "frequency": "monthly",
            },
            {
                "kpi_id": "KPI-REM-005",
                "name": "Milestone Adherence",
                "description": "Percentage of milestones completed on or before target date",
                "target_value": 90.0,
                "current_value": 0.0,
                "unit": "percent",
                "frequency": "monthly",
            },
        ]

        # Generate per-claim tracking entries
        tracking_entries: List[Dict[str, Any]] = []
        for item in timeline_data.get("timeline_items", []):
            tracking_entries.append({
                "claim_id": item["claim_id"],
                "triage_category": item["triage_category"],
                "status": TrackingStatus.NOT_STARTED.value,
                "start_month": item["start_month"],
                "end_month": item["end_month"],
                "completion_pct": 0.0,
                "tracking_id": _new_uuid(),
            })

        checkpoints = timeline_data.get("milestones", [])

        result_data: Dict[str, Any] = {
            "plan_id": _new_uuid(),
            "kpis": kpis,
            "kpi_count": len(kpis),
            "tracking_entries": tracking_entries,
            "tracked_claims": len(tracking_entries),
            "checkpoints": checkpoints,
            "checkpoint_count": len(checkpoints),
            "monitoring_frequency": "monthly",
            "escalation_threshold_pct": 20.0,
            "overall_status": TrackingStatus.NOT_STARTED.value,
        }

        return PhaseResult(
            phase_name=RemediationPhase.PROGRESS_TRACKING.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise_severity(self, severity: str) -> SeverityLevel:
        """Normalise severity string to SeverityLevel enum."""
        severity_lower = severity.lower()
        for level in SeverityLevel:
            if level.value == severity_lower:
                return level
        return SeverityLevel.MEDIUM

    def _classify_triage(self, severity: SeverityLevel, issue: str) -> TriageCategory:
        """Classify a claim into a triage category based on severity and issue."""
        if any(ind in issue for ind in self.WITHDRAW_INDICATORS):
            return TriageCategory.WITHDRAW
        if severity == SeverityLevel.CRITICAL:
            return TriageCategory.WITHDRAW
        if any(ind in issue for ind in self.REWORD_INDICATORS):
            return TriageCategory.REWORD
        if any(ind in issue for ind in self.SUBSTANTIATE_INDICATORS):
            return TriageCategory.SUBSTANTIATE
        if severity == SeverityLevel.HIGH:
            return TriageCategory.SUBSTANTIATE
        return TriageCategory.REPLACE

    def _get_triage_actions(self, category: TriageCategory) -> List[str]:
        """Return ordered action steps for a triage category."""
        actions_map: Dict[TriageCategory, List[str]] = {
            TriageCategory.WITHDRAW: [
                "Immediately cease use of claim in all channels",
                "Notify marketing and legal teams",
                "Remove claim from website, packaging, and advertising",
                "Document withdrawal for audit trail",
            ],
            TriageCategory.REWORD: [
                "Draft alternative claim language with specific metrics",
                "Legal review of proposed rewording",
                "Update claim across all communication channels",
                "Verify rewording meets Article 3 specificity requirements",
            ],
            TriageCategory.SUBSTANTIATE: [
                "Commission lifecycle assessment or scientific study",
                "Compile evidence dossier per Article 5 requirements",
                "Engage accredited third-party verifier",
                "Publish substantiation information alongside claim",
            ],
            TriageCategory.REPLACE: [
                "Identify EU-approved alternative label or claim",
                "Apply for relevant certification scheme",
                "Transition products to new label/claim",
                "Update all marketing materials",
            ],
        }
        return actions_map.get(category, ["Review claim against Directive requirements"])

    def _assign_start_month(self, category: str, current_month: int) -> int:
        """Assign start month based on triage category priority."""
        start_map: Dict[str, int] = {
            TriageCategory.WITHDRAW.value: 1,
            TriageCategory.REWORD.value: 1,
            TriageCategory.SUBSTANTIATE.value: 2,
            TriageCategory.REPLACE.value: 3,
        }
        return start_map.get(category, current_month)

    def _count_external_services(self, triage_data: Dict[str, Any]) -> Dict[str, int]:
        """Count external service requirements from triage data."""
        substantiate_count = sum(
            1 for c in triage_data.get("triaged_claims", [])
            if c["triage_category"] == TriageCategory.SUBSTANTIATE.value
        )
        replace_count = sum(
            1 for c in triage_data.get("triaged_claims", [])
            if c["triage_category"] == TriageCategory.REPLACE.value
        )
        return {
            "lca_providers_needed": substantiate_count,
            "verifiers_needed": substantiate_count,
            "certification_bodies_needed": replace_count,
        }
