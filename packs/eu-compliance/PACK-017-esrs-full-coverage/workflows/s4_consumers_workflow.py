# -*- coding: utf-8 -*-
"""
ESRS S4 Consumers and End-Users Workflow
==========================================

5-phase workflow for ESRS S4 Consumers and End-Users disclosure covering
policy review, product safety assessment, data privacy assessment, action
evaluation, and target tracking with full provenance tracking.

Phases:
    1. PolicyReview        -- Review consumer/end-user policies (S4-1)
    2. SafetyAssessment    -- Assess product and service safety (S4-2)
    3. PrivacyAssessment   -- Assess data privacy and security (S4-3)
    4. ActionEvaluation    -- Evaluate actions and remediation (S4-4)
    5. TargetTracking      -- Track targets and progress (S4-5)

ESRS S4 Disclosure Requirements (5 DRs):
    S4-1: Policies related to consumers and end-users
    S4-2: Processes for engaging with consumers and end-users
    S4-3: Processes to remediate negative impacts and channels
    S4-4: Taking action on material impacts
    S4-5: Targets related to managing impacts

Author: GreenLang Team
Version: 17.0.0
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
    """Phases of the S4 consumers workflow."""
    POLICY_REVIEW = "policy_review"
    SAFETY_ASSESSMENT = "safety_assessment"
    PRIVACY_ASSESSMENT = "privacy_assessment"
    ACTION_EVALUATION = "action_evaluation"
    TARGET_TRACKING = "target_tracking"

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

class SafetyRiskLevel(str, Enum):
    """Product/service safety risk level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PrivacyRiskLevel(str, Enum):
    """Data privacy risk level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

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

class SafetyIncident(BaseModel):
    """Product/service safety incident record."""
    incident_id: str = Field(default_factory=lambda: f"si-{_new_uuid()[:8]}")
    product_service: str = Field(default="")
    description: str = Field(default="")
    risk_level: SafetyRiskLevel = Field(default=SafetyRiskLevel.MEDIUM)
    incidents_count: int = Field(default=0, ge=0)
    recalls_count: int = Field(default=0, ge=0)
    injuries_count: int = Field(default=0, ge=0)
    resolved: bool = Field(default=False)

class PrivacyIssue(BaseModel):
    """Data privacy issue record."""
    issue_id: str = Field(default_factory=lambda: f"pi-{_new_uuid()[:8]}")
    description: str = Field(default="")
    risk_level: PrivacyRiskLevel = Field(default=PrivacyRiskLevel.MEDIUM)
    data_subjects_affected: int = Field(default=0, ge=0)
    breach_reported: bool = Field(default=False)
    gdpr_compliant: bool = Field(default=True)
    resolved: bool = Field(default=False)

class S4ConsumersInput(BaseModel):
    """Input data model for S4ConsumersWorkflow."""
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    s4_is_material: bool = Field(default=True, description="Whether S4 is material")
    policies: List[Dict[str, Any]] = Field(
        default_factory=list, description="S4-1 consumer policies"
    )
    engagement_processes: List[Dict[str, Any]] = Field(
        default_factory=list, description="S4-2 engagement processes"
    )
    remediation_channels: List[Dict[str, Any]] = Field(
        default_factory=list, description="S4-3 remediation channels"
    )
    actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="S4-4 actions on impacts"
    )
    safety_incidents: List[SafetyIncident] = Field(
        default_factory=list, description="Product safety incidents"
    )
    privacy_issues: List[PrivacyIssue] = Field(
        default_factory=list, description="Data privacy issues"
    )
    targets: List[Dict[str, Any]] = Field(
        default_factory=list, description="S4-5 targets"
    )
    accessibility_data: Dict[str, Any] = Field(
        default_factory=dict, description="Digital/physical accessibility data"
    )
    config: Dict[str, Any] = Field(default_factory=dict)

class S4ConsumersWorkflowResult(BaseModel):
    """Complete result from S4 consumers workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="s4_consumers")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    s4_is_material: bool = Field(default=True)
    policies_count: int = Field(default=0)
    safety_incidents_count: int = Field(default=0)
    high_critical_safety: int = Field(default=0)
    total_recalls: int = Field(default=0)
    privacy_issues_count: int = Field(default=0)
    data_breaches_count: int = Field(default=0)
    data_subjects_affected: int = Field(default=0)
    targets_count: int = Field(default=0)
    overall_completeness_pct: float = Field(default=0.0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class S4ConsumersWorkflow:
    """
    5-phase ESRS S4 Consumers and End-Users workflow.

    Orchestrates policy review, safety assessment, privacy assessment,
    action evaluation, and target tracking for complete S4 disclosure
    covering S4-1 through S4-5.

    Zero-hallucination: all safety and privacy aggregations use deterministic
    counting. No LLM in numeric assessment paths.

    Example:
        >>> wf = S4ConsumersWorkflow()
        >>> inp = S4ConsumersInput(safety_incidents=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.safety_incidents_count >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize S4ConsumersWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._sub_results: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.POLICY_REVIEW.value, "description": "Review consumer policies"},
            {"name": WorkflowPhase.SAFETY_ASSESSMENT.value, "description": "Assess product safety"},
            {"name": WorkflowPhase.PRIVACY_ASSESSMENT.value, "description": "Assess data privacy"},
            {"name": WorkflowPhase.ACTION_EVALUATION.value, "description": "Evaluate actions and remediation"},
            {"name": WorkflowPhase.TARGET_TRACKING.value, "description": "Track targets and progress"},
        ]

    def validate_inputs(self, input_data: S4ConsumersInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.s4_is_material:
            issues.append("S4 is not material; full disclosure not required")
        if not input_data.policies:
            issues.append("No consumer policies provided")
        return issues

    async def execute(
        self,
        input_data: Optional[S4ConsumersInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> S4ConsumersWorkflowResult:
        """
        Execute the 5-phase S4 consumers workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            S4ConsumersWorkflowResult with safety, privacy, and compliance assessment.
        """
        if input_data is None:
            input_data = S4ConsumersInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting S4 consumers workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_policy_review(input_data))
            phase_results.append(await self._phase_safety_assessment(input_data))
            phase_results.append(await self._phase_privacy_assessment(input_data))
            phase_results.append(await self._phase_action_evaluation(input_data))
            phase_results.append(await self._phase_target_tracking(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("S4 consumers workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        high_safety = sum(
            1 for s in input_data.safety_incidents
            if s.risk_level in (SafetyRiskLevel.HIGH, SafetyRiskLevel.CRITICAL)
        )
        total_recalls = sum(s.recalls_count for s in input_data.safety_incidents)
        breaches = sum(1 for p in input_data.privacy_issues if p.breach_reported)
        total_subjects = sum(p.data_subjects_affected for p in input_data.privacy_issues)
        completeness = self._calculate_completeness(input_data)

        result = S4ConsumersWorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            s4_is_material=input_data.s4_is_material,
            policies_count=len(input_data.policies),
            safety_incidents_count=len(input_data.safety_incidents),
            high_critical_safety=high_safety,
            total_recalls=total_recalls,
            privacy_issues_count=len(input_data.privacy_issues),
            data_breaches_count=breaches,
            data_subjects_affected=total_subjects,
            targets_count=len(input_data.targets),
            overall_completeness_pct=completeness,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "S4 consumers %s completed in %.2fs: %d safety incidents, %d privacy issues",
            self.workflow_id, elapsed, len(input_data.safety_incidents), len(input_data.privacy_issues),
        )
        return result

    async def _phase_policy_review(self, input_data: S4ConsumersInput) -> PhaseResult:
        """Review consumer and end-user policies (S4-1)."""
        started = utcnow()
        outputs: Dict[str, Any] = {"policies_count": len(input_data.policies)}
        warnings: List[str] = []
        outputs["has_product_safety_policy"] = any(p.get("scope") == "product_safety" for p in input_data.policies)
        outputs["has_data_privacy_policy"] = any(p.get("scope") == "data_privacy" for p in input_data.policies)
        outputs["has_accessibility_policy"] = any(p.get("scope") == "accessibility" for p in input_data.policies)
        if not input_data.policies:
            warnings.append("No consumer policies defined (S4-1)")
        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 1 PolicyReview: %d policies", len(input_data.policies))
        return PhaseResult(
            phase_name=WorkflowPhase.POLICY_REVIEW.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    async def _phase_safety_assessment(self, input_data: S4ConsumersInput) -> PhaseResult:
        """Assess product and service safety (S4-2)."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        incidents = input_data.safety_incidents
        outputs["incidents_count"] = len(incidents)
        outputs["by_risk_level"] = {
            level.value: sum(1 for i in incidents if i.risk_level == level) for level in SafetyRiskLevel
        }
        outputs["total_recalls"] = sum(i.recalls_count for i in incidents)
        outputs["total_injuries"] = sum(i.injuries_count for i in incidents)
        outputs["resolved_count"] = sum(1 for i in incidents if i.resolved)
        outputs["unresolved_count"] = sum(1 for i in incidents if not i.resolved)

        critical = [i for i in incidents if i.risk_level == SafetyRiskLevel.CRITICAL]
        if critical:
            warnings.append(f"CRITICAL: {len(critical)} critical safety incidents")
        unresolved = [i for i in incidents if not i.resolved]
        if unresolved:
            warnings.append(f"{len(unresolved)} unresolved safety incidents")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 2 SafetyAssessment: %d incidents, %d recalls",
                         len(incidents), outputs["total_recalls"])
        return PhaseResult(
            phase_name=WorkflowPhase.SAFETY_ASSESSMENT.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    async def _phase_privacy_assessment(self, input_data: S4ConsumersInput) -> PhaseResult:
        """Assess data privacy and security (S4-3)."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        issues = input_data.privacy_issues
        outputs["issues_count"] = len(issues)
        outputs["by_risk_level"] = {
            level.value: sum(1 for i in issues if i.risk_level == level) for level in PrivacyRiskLevel
        }
        outputs["data_breaches"] = sum(1 for i in issues if i.breach_reported)
        outputs["total_data_subjects_affected"] = sum(i.data_subjects_affected for i in issues)
        outputs["gdpr_non_compliant"] = sum(1 for i in issues if not i.gdpr_compliant)
        outputs["resolved_count"] = sum(1 for i in issues if i.resolved)

        breaches = [i for i in issues if i.breach_reported]
        if breaches:
            warnings.append(f"{len(breaches)} data breaches reported")
        non_compliant = [i for i in issues if not i.gdpr_compliant]
        if non_compliant:
            warnings.append(f"CRITICAL: {len(non_compliant)} GDPR non-compliant issues")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 3 PrivacyAssessment: %d issues, %d breaches",
                         len(issues), outputs["data_breaches"])
        return PhaseResult(
            phase_name=WorkflowPhase.PRIVACY_ASSESSMENT.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    async def _phase_action_evaluation(self, input_data: S4ConsumersInput) -> PhaseResult:
        """Evaluate actions and remediation for consumers (S4-4)."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        outputs["actions_count"] = len(input_data.actions)
        outputs["remediation_channels_count"] = len(input_data.remediation_channels)
        outputs["engagement_processes_count"] = len(input_data.engagement_processes)
        outputs["has_accessibility_data"] = bool(input_data.accessibility_data)
        if not input_data.actions:
            warnings.append("No actions on consumer impacts (S4-4)")
        if not input_data.remediation_channels:
            warnings.append("No remediation channels for consumers (S4-3)")
        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 4 ActionEvaluation: %d actions", len(input_data.actions))
        return PhaseResult(
            phase_name=WorkflowPhase.ACTION_EVALUATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    async def _phase_target_tracking(self, input_data: S4ConsumersInput) -> PhaseResult:
        """Track targets and progress for consumers (S4-5)."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        targets = input_data.targets
        outputs["targets_count"] = len(targets)
        outputs["on_track"] = sum(1 for t in targets if t.get("on_track"))
        if not targets:
            warnings.append("No consumer targets defined (S4-5)")
        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 5 TargetTracking: %d targets", len(targets))
        return PhaseResult(
            phase_name=WorkflowPhase.TARGET_TRACKING.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calculate_completeness(self, input_data: S4ConsumersInput) -> float:
        """Calculate overall S4 completeness percentage."""
        scores: List[float] = []
        scores.append(100.0 if input_data.policies else 0.0)
        scores.append(100.0 if input_data.engagement_processes else 0.0)
        scores.append(100.0 if input_data.remediation_channels else 0.0)
        scores.append(100.0 if input_data.actions else 0.0)
        scores.append(100.0 if input_data.targets else 0.0)
        return round(sum(scores) / len(scores), 1) if scores else 0.0

    def _compute_provenance(self, result: S4ConsumersWorkflowResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
