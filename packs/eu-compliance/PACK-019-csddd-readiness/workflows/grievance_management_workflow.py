# -*- coding: utf-8 -*-
"""
CSDDD Grievance Management Workflow
===============================================

4-phase workflow for establishing and managing complaints procedures (grievance
mechanisms) under the EU Corporate Sustainability Due Diligence Directive
(CSDDD / CS3D). Covers mechanism design, channel setup, case processing,
and resolution tracking.

Phases:
    1. MechanismDesign         -- Design grievance mechanism per Art. 11 criteria
    2. ChannelSetup            -- Assess and configure reporting channels
    3. CaseProcessing          -- Process and categorize submitted cases
    4. ResolutionTracking      -- Track resolution outcomes and accessibility

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Art. 11: Complaints procedure
    - Art. 11(1): Obligation to establish complaints procedure
    - Art. 11(2): Right to submit complaints (persons, trade unions, CSOs)
    - Art. 11(3): Fair, transparent, accessible procedure
    - Art. 11(4): Inform complainant of outcome; right of appeal
    - Art. 9: Remediation
    - UN Guiding Principles on Business and Human Rights (UNGPs)
      Principle 31: Effectiveness criteria for non-judicial grievance mechanisms

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
    """Phases of the grievance management workflow."""
    MECHANISM_DESIGN = "mechanism_design"
    CHANNEL_SETUP = "channel_setup"
    CASE_PROCESSING = "case_processing"
    RESOLUTION_TRACKING = "resolution_tracking"

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

class ChannelType(str, Enum):
    """Types of grievance reporting channels."""
    HOTLINE = "hotline"
    EMAIL = "email"
    WEB_PORTAL = "web_portal"
    IN_PERSON = "in_person"
    MOBILE_APP = "mobile_app"
    POSTAL = "postal"
    TRADE_UNION = "trade_union"
    THIRD_PARTY = "third_party"

class CaseStatus(str, Enum):
    """Status of a grievance case."""
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    UNDER_REVIEW = "under_review"
    INVESTIGATION = "investigation"
    RESOLVED = "resolved"
    CLOSED = "closed"
    APPEALED = "appealed"
    REJECTED = "rejected"

class CaseCategory(str, Enum):
    """Category of grievance per CSDDD Annexes."""
    HUMAN_RIGHTS = "human_rights"
    ENVIRONMENT = "environment"
    LABOR_RIGHTS = "labor_rights"
    LAND_RIGHTS = "land_rights"
    HEALTH_SAFETY = "health_safety"
    CORRUPTION = "corruption"
    OTHER = "other"

class UNGPCriteria(str, Enum):
    """UNGP Principle 31 effectiveness criteria."""
    LEGITIMATE = "legitimate"
    ACCESSIBLE = "accessible"
    PREDICTABLE = "predictable"
    EQUITABLE = "equitable"
    TRANSPARENT = "transparent"
    RIGHTS_COMPATIBLE = "rights_compatible"
    CONTINUOUS_LEARNING = "continuous_learning"
    DIALOGUE_BASED = "dialogue_based"

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

class MechanismConfig(BaseModel):
    """Configuration for the grievance mechanism."""
    mechanism_id: str = Field(default_factory=lambda: f"mech-{_new_uuid()[:8]}")
    mechanism_name: str = Field(default="", description="Name of mechanism")
    channels: List[ChannelType] = Field(
        default_factory=list, description="Available reporting channels"
    )
    languages_supported: List[str] = Field(
        default_factory=lambda: ["en"], description="ISO 639-1 language codes"
    )
    anonymous_reporting: bool = Field(default=True, description="Allows anonymous complaints")
    whistleblower_protection: bool = Field(default=True)
    acknowledgment_days: int = Field(default=7, ge=1, le=30, description="Days to acknowledge receipt")
    resolution_target_days: int = Field(default=90, ge=1, le=365, description="Target days to resolve")
    appeal_available: bool = Field(default=True, description="Whether appeal process exists")
    eligible_complainants: List[str] = Field(
        default_factory=lambda: ["affected_persons", "trade_unions", "cso_representatives"],
        description="Groups eligible to submit complaints per Art. 11(2)",
    )
    independent_oversight: bool = Field(default=False, description="External oversight body")

class GrievanceCase(BaseModel):
    """Individual grievance case record."""
    case_id: str = Field(default_factory=lambda: f"case-{_new_uuid()[:8]}")
    submitted_date: str = Field(default="", description="ISO date of submission")
    acknowledged_date: str = Field(default="")
    resolved_date: str = Field(default="")
    category: CaseCategory = Field(default=CaseCategory.OTHER)
    status: CaseStatus = Field(default=CaseStatus.SUBMITTED)
    channel_used: ChannelType = Field(default=ChannelType.WEB_PORTAL)
    complainant_group: str = Field(default="", description="Stakeholder group of complainant")
    country_code: str = Field(default="")
    is_anonymous: bool = Field(default=False)
    severity: str = Field(default="medium", description="critical/high/medium/low")
    resolution_outcome: str = Field(default="", description="Resolution description")
    days_to_acknowledge: int = Field(default=0, ge=0)
    days_to_resolve: int = Field(default=0, ge=0)
    appealed: bool = Field(default=False)

class StakeholderGroup(BaseModel):
    """Stakeholder group for accessibility assessment."""
    group_name: str = Field(default="", description="Name of stakeholder group")
    language: str = Field(default="en", description="Primary language")
    literacy_level: str = Field(default="high", description="high/medium/low")
    digital_access: bool = Field(default=True, description="Has digital access")
    location: str = Field(default="", description="Geographic location")
    awareness_of_mechanism: bool = Field(default=False)

class GrievanceManagementInput(BaseModel):
    """Input data model for GrievanceManagementWorkflow."""
    entity_id: str = Field(default="", description="Reporting entity ID")
    entity_name: str = Field(default="", description="Reporting entity name")
    reporting_year: int = Field(default=2026, ge=2024, le=2050)
    mechanism_config: MechanismConfig = Field(
        default_factory=MechanismConfig, description="Mechanism configuration"
    )
    cases: List[GrievanceCase] = Field(
        default_factory=list, description="Grievance cases for analysis"
    )
    stakeholder_groups: List[StakeholderGroup] = Field(
        default_factory=list, description="Stakeholder groups for accessibility"
    )
    config: Dict[str, Any] = Field(default_factory=dict)

class GrievanceManagementResult(BaseModel):
    """Complete result from grievance management workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="grievance_management")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    # Mechanism assessment
    mechanism_readiness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    ungp_compliance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    channels_available: int = Field(default=0, ge=0)
    accessibility_score: float = Field(default=0.0, ge=0.0, le=100.0)
    # Case statistics
    total_cases: int = Field(default=0, ge=0)
    resolved_cases: int = Field(default=0, ge=0)
    pending_cases: int = Field(default=0, ge=0)
    appealed_cases: int = Field(default=0, ge=0)
    avg_days_to_acknowledge: float = Field(default=0.0, ge=0.0)
    avg_days_to_resolve: float = Field(default=0.0, ge=0.0)
    resolution_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    # Breakdown
    cases_by_category: Dict[str, int] = Field(default_factory=dict)
    cases_by_channel: Dict[str, int] = Field(default_factory=dict)
    cases_by_status: Dict[str, int] = Field(default_factory=dict)
    reporting_year: int = Field(default=2026)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class GrievanceManagementWorkflow:
    """
    4-phase CSDDD grievance management workflow.

    Assesses mechanism design against Art. 11 and UNGP Principle 31 criteria,
    evaluates channel accessibility, processes cases, and tracks resolution
    metrics.

    Zero-hallucination: all scores use deterministic formulas.
    No LLM in numeric calculation paths.

    Example:
        >>> wf = GrievanceManagementWorkflow()
        >>> inp = GrievanceManagementInput(
        ...     mechanism_config=MechanismConfig(channels=[ChannelType.HOTLINE, ChannelType.WEB_PORTAL])
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.mechanism_readiness_score >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize GrievanceManagementWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._mechanism_score: float = 0.0
        self._ungp_score: float = 0.0
        self._accessibility_score: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.MECHANISM_DESIGN.value, "description": "Assess mechanism design per Art. 11"},
            {"name": WorkflowPhase.CHANNEL_SETUP.value, "description": "Assess channel accessibility"},
            {"name": WorkflowPhase.CASE_PROCESSING.value, "description": "Process and categorize cases"},
            {"name": WorkflowPhase.RESOLUTION_TRACKING.value, "description": "Track resolution outcomes"},
        ]

    def validate_inputs(self, input_data: GrievanceManagementInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        mc = input_data.mechanism_config
        if not mc.channels:
            issues.append("No reporting channels configured")
        if mc.acknowledgment_days > 14:
            issues.append("Acknowledgment period exceeds 14 days -- may not meet Art. 11(3)")
        return issues

    async def execute(
        self,
        input_data: Optional[GrievanceManagementInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> GrievanceManagementResult:
        """
        Execute the 4-phase grievance management workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            GrievanceManagementResult with scores and case statistics.
        """
        if input_data is None:
            input_data = GrievanceManagementInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting grievance management workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_mechanism_design(input_data))
            phase_results.append(await self._phase_channel_setup(input_data))
            phase_results.append(await self._phase_case_processing(input_data))
            phase_results.append(await self._phase_resolution_tracking(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Grievance management failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        cases = input_data.cases
        resolved = [c for c in cases if c.status in (CaseStatus.RESOLVED, CaseStatus.CLOSED)]
        pending = [c for c in cases if c.status not in (CaseStatus.RESOLVED, CaseStatus.CLOSED, CaseStatus.REJECTED)]
        appealed = [c for c in cases if c.appealed]

        ack_days = [c.days_to_acknowledge for c in cases if c.days_to_acknowledge > 0]
        res_days = [c.days_to_resolve for c in resolved if c.days_to_resolve > 0]

        by_category: Dict[str, int] = {}
        by_channel: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        for c in cases:
            by_category[c.category.value] = by_category.get(c.category.value, 0) + 1
            by_channel[c.channel_used.value] = by_channel.get(c.channel_used.value, 0) + 1
            by_status[c.status.value] = by_status.get(c.status.value, 0) + 1

        result = GrievanceManagementResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            mechanism_readiness_score=self._mechanism_score,
            ungp_compliance_score=self._ungp_score,
            channels_available=len(input_data.mechanism_config.channels),
            accessibility_score=self._accessibility_score,
            total_cases=len(cases),
            resolved_cases=len(resolved),
            pending_cases=len(pending),
            appealed_cases=len(appealed),
            avg_days_to_acknowledge=round(sum(ack_days) / len(ack_days), 1) if ack_days else 0.0,
            avg_days_to_resolve=round(sum(res_days) / len(res_days), 1) if res_days else 0.0,
            resolution_rate_pct=round(
                (len(resolved) / len(cases)) * 100, 1
            ) if cases else 0.0,
            cases_by_category=by_category,
            cases_by_channel=by_channel,
            cases_by_status=by_status,
            reporting_year=input_data.reporting_year,
            executed_at=utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Grievance management %s completed in %.2fs: readiness=%.1f%%, %d cases",
            self.workflow_id, elapsed, self._mechanism_score, len(cases),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Mechanism Design
    # -------------------------------------------------------------------------

    async def _phase_mechanism_design(
        self, input_data: GrievanceManagementInput,
    ) -> PhaseResult:
        """Assess grievance mechanism design against Art. 11 and UNGP P31."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        mc = input_data.mechanism_config

        # Art. 11 compliance checklist
        art11_checks: Dict[str, bool] = {
            "has_channels": len(mc.channels) > 0,
            "anonymous_reporting": mc.anonymous_reporting,
            "whistleblower_protection": mc.whistleblower_protection,
            "timely_acknowledgment": mc.acknowledgment_days <= 14,
            "appeal_available": mc.appeal_available,
            "eligible_complainants_defined": len(mc.eligible_complainants) > 0,
            "multiple_channels": len(mc.channels) >= 2,
            "independent_oversight": mc.independent_oversight,
        }
        art11_score = round(
            (sum(1 for v in art11_checks.values() if v) / len(art11_checks)) * 100, 1
        )

        # UNGP Principle 31 effectiveness criteria assessment
        ungp_checks: Dict[str, bool] = {
            UNGPCriteria.LEGITIMATE.value: mc.independent_oversight or mc.whistleblower_protection,
            UNGPCriteria.ACCESSIBLE.value: len(mc.channels) >= 2 and len(mc.languages_supported) >= 1,
            UNGPCriteria.PREDICTABLE.value: mc.resolution_target_days > 0 and mc.acknowledgment_days > 0,
            UNGPCriteria.EQUITABLE.value: mc.anonymous_reporting and mc.whistleblower_protection,
            UNGPCriteria.TRANSPARENT.value: len(mc.eligible_complainants) > 0,
            UNGPCriteria.RIGHTS_COMPATIBLE.value: "affected_persons" in mc.eligible_complainants,
            UNGPCriteria.CONTINUOUS_LEARNING.value: mc.appeal_available,
            UNGPCriteria.DIALOGUE_BASED.value: "trade_unions" in mc.eligible_complainants or "cso_representatives" in mc.eligible_complainants,
        }
        self._ungp_score = round(
            (sum(1 for v in ungp_checks.values() if v) / len(ungp_checks)) * 100, 1
        )

        # Overall mechanism readiness (60% Art. 11 + 40% UNGP)
        self._mechanism_score = round(0.60 * art11_score + 0.40 * self._ungp_score, 1)

        outputs["art11_compliance_checks"] = art11_checks
        outputs["art11_score"] = art11_score
        outputs["ungp_criteria_checks"] = ungp_checks
        outputs["ungp_score"] = self._ungp_score
        outputs["mechanism_readiness_score"] = self._mechanism_score
        outputs["channels_configured"] = len(mc.channels)
        outputs["languages_supported"] = mc.languages_supported

        if not mc.anonymous_reporting:
            warnings.append("Anonymous reporting not enabled -- may deter vulnerable complainants")
        if not mc.whistleblower_protection:
            warnings.append("No whistleblower protection policy in place")
        if not mc.appeal_available:
            warnings.append("No appeal process -- Art. 11(4) requires right of appeal")
        if not mc.independent_oversight:
            warnings.append("No independent oversight -- consider appointing external monitor")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 MechanismDesign: Art.11=%.1f%%, UNGP=%.1f%%, overall=%.1f%%",
            art11_score, self._ungp_score, self._mechanism_score,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.MECHANISM_DESIGN.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Channel Setup
    # -------------------------------------------------------------------------

    async def _phase_channel_setup(
        self, input_data: GrievanceManagementInput,
    ) -> PhaseResult:
        """Assess channel accessibility for all stakeholder groups."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        mc = input_data.mechanism_config
        groups = input_data.stakeholder_groups

        # Assess accessibility per stakeholder group
        group_scores: List[Dict[str, Any]] = []
        for grp in groups:
            score = 0.0
            factors = []

            # Language coverage
            if grp.language in mc.languages_supported:
                score += 25.0
            else:
                factors.append(f"language_{grp.language}_not_supported")

            # Digital access
            digital_channels = [c for c in mc.channels if c in (
                ChannelType.WEB_PORTAL, ChannelType.EMAIL, ChannelType.MOBILE_APP
            )]
            non_digital_channels = [c for c in mc.channels if c in (
                ChannelType.HOTLINE, ChannelType.IN_PERSON, ChannelType.POSTAL
            )]

            if grp.digital_access and digital_channels:
                score += 25.0
            elif not grp.digital_access and non_digital_channels:
                score += 25.0
            elif not grp.digital_access and not non_digital_channels:
                factors.append("no_non_digital_channels_for_low_access_group")

            # Literacy considerations
            if grp.literacy_level == "low" and ChannelType.HOTLINE in mc.channels:
                score += 25.0
            elif grp.literacy_level in ("medium", "high"):
                score += 25.0
            else:
                factors.append("no_oral_channel_for_low_literacy_group")

            # Awareness
            if grp.awareness_of_mechanism:
                score += 25.0
            else:
                factors.append("group_unaware_of_mechanism")

            group_scores.append({
                "group_name": grp.group_name,
                "accessibility_score": round(score, 1),
                "barriers": factors,
            })

        self._accessibility_score = round(
            sum(gs["accessibility_score"] for gs in group_scores) / len(group_scores), 1
        ) if group_scores else (
            75.0 if len(mc.channels) >= 2 else 50.0
        )

        outputs["stakeholder_groups_assessed"] = len(group_scores)
        outputs["group_accessibility_scores"] = group_scores
        outputs["overall_accessibility_score"] = self._accessibility_score
        outputs["channels_available"] = [c.value for c in mc.channels]
        outputs["has_digital_channels"] = any(
            c in mc.channels for c in (ChannelType.WEB_PORTAL, ChannelType.EMAIL, ChannelType.MOBILE_APP)
        )
        outputs["has_non_digital_channels"] = any(
            c in mc.channels for c in (ChannelType.HOTLINE, ChannelType.IN_PERSON, ChannelType.POSTAL)
        )

        low_access_groups = [gs for gs in group_scores if gs["accessibility_score"] < 50]
        if low_access_groups:
            warnings.append(
                f"{len(low_access_groups)} stakeholder groups have accessibility score below 50%"
            )
        if not groups:
            warnings.append("No stakeholder groups defined for accessibility assessment")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 ChannelSetup: accessibility=%.1f%%, %d groups assessed",
            self._accessibility_score, len(group_scores),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.CHANNEL_SETUP.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Case Processing
    # -------------------------------------------------------------------------

    async def _phase_case_processing(
        self, input_data: GrievanceManagementInput,
    ) -> PhaseResult:
        """Process and categorize submitted grievance cases."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        cases = input_data.cases
        mc = input_data.mechanism_config

        # Case categorization
        by_category: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_channel: Dict[str, int] = {}
        anonymous_count = 0

        for c in cases:
            by_category[c.category.value] = by_category.get(c.category.value, 0) + 1
            by_severity[c.severity] = by_severity.get(c.severity, 0) + 1
            by_channel[c.channel_used.value] = by_channel.get(c.channel_used.value, 0) + 1
            if c.is_anonymous:
                anonymous_count += 1

        # Acknowledgment compliance check
        late_ack = [
            c for c in cases
            if c.days_to_acknowledge > mc.acknowledgment_days and c.days_to_acknowledge > 0
        ]

        # Overdue resolution
        overdue = [
            c for c in cases
            if c.status not in (CaseStatus.RESOLVED, CaseStatus.CLOSED, CaseStatus.REJECTED)
            and c.days_to_resolve > mc.resolution_target_days
        ]

        outputs["total_cases"] = len(cases)
        outputs["by_category"] = by_category
        outputs["by_severity"] = by_severity
        outputs["by_channel"] = by_channel
        outputs["anonymous_submissions"] = anonymous_count
        outputs["anonymous_pct"] = round(
            (anonymous_count / len(cases)) * 100, 1
        ) if cases else 0.0
        outputs["late_acknowledgments"] = len(late_ack)
        outputs["overdue_cases"] = len(overdue)
        outputs["critical_cases"] = by_severity.get("critical", 0)

        if late_ack:
            warnings.append(
                f"{len(late_ack)} cases acknowledged beyond {mc.acknowledgment_days}-day target"
            )
        if overdue:
            warnings.append(
                f"{len(overdue)} cases overdue beyond {mc.resolution_target_days}-day resolution target"
            )
        if by_severity.get("critical", 0) > 0:
            warnings.append(f"{by_severity['critical']} critical cases require escalated attention")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 CaseProcessing: %d cases, %d late ack, %d overdue",
            len(cases), len(late_ack), len(overdue),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.CASE_PROCESSING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Resolution Tracking
    # -------------------------------------------------------------------------

    async def _phase_resolution_tracking(
        self, input_data: GrievanceManagementInput,
    ) -> PhaseResult:
        """Track resolution outcomes and compute resolution metrics."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        cases = input_data.cases
        resolved = [c for c in cases if c.status in (CaseStatus.RESOLVED, CaseStatus.CLOSED)]
        appealed = [c for c in cases if c.appealed]

        # Resolution rate
        resolution_rate = round(
            (len(resolved) / len(cases)) * 100, 1
        ) if cases else 0.0

        # Average resolution time
        res_days_list = [c.days_to_resolve for c in resolved if c.days_to_resolve > 0]
        avg_resolution = round(
            sum(res_days_list) / len(res_days_list), 1
        ) if res_days_list else 0.0

        # Resolution outcomes
        outcomes: Dict[str, int] = {}
        for c in resolved:
            outcome = c.resolution_outcome if c.resolution_outcome else "unspecified"
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

        # Appeal rate
        appeal_rate = round(
            (len(appealed) / len(cases)) * 100, 1
        ) if cases else 0.0

        # Satisfaction proxy: resolved within target and not appealed
        target_days = input_data.mechanism_config.resolution_target_days
        on_time_resolved = [
            c for c in resolved
            if c.days_to_resolve <= target_days and not c.appealed
        ]
        on_time_rate = round(
            (len(on_time_resolved) / len(resolved)) * 100, 1
        ) if resolved else 0.0

        # Resolution by category
        resolution_by_category: Dict[str, float] = {}
        for cat in CaseCategory:
            cat_cases = [c for c in cases if c.category == cat]
            cat_resolved = [c for c in cat_cases if c.status in (CaseStatus.RESOLVED, CaseStatus.CLOSED)]
            if cat_cases:
                resolution_by_category[cat.value] = round(
                    (len(cat_resolved) / len(cat_cases)) * 100, 1
                )

        outputs["resolution_rate_pct"] = resolution_rate
        outputs["avg_resolution_days"] = avg_resolution
        outputs["on_time_resolution_rate_pct"] = on_time_rate
        outputs["appeal_rate_pct"] = appeal_rate
        outputs["appealed_count"] = len(appealed)
        outputs["resolution_outcomes"] = outcomes
        outputs["resolution_by_category"] = resolution_by_category
        outputs["resolved_count"] = len(resolved)
        outputs["total_cases"] = len(cases)

        if resolution_rate < 50 and len(cases) > 0:
            warnings.append(f"Resolution rate is {resolution_rate}% -- below 50% threshold")
        if appeal_rate > 20:
            warnings.append(f"Appeal rate is {appeal_rate}% -- may indicate process issues")
        if avg_resolution > target_days:
            warnings.append(
                f"Average resolution time ({avg_resolution} days) exceeds target ({target_days} days)"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ResolutionTracking: rate=%.1f%%, avg=%.0f days, appeals=%.1f%%",
            resolution_rate, avg_resolution, appeal_rate,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.RESOLUTION_TRACKING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: GrievanceManagementResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
