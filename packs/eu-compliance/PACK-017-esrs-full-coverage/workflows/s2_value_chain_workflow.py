# -*- coding: utf-8 -*-
"""
S2 Value Chain Workers Workflow
===============================

5-phase workflow for ESRS S2 Workers in the Value Chain disclosure
covering policy assessment, engagement evaluation, risk assessment, target
tracking, and report generation with full provenance tracking.

Phases:
    1. PolicyAssessment      -- S2-1 policy scope and alignment
    2. EngagementEvaluation  -- S2-2 engagement coverage
    3. RiskAssessment        -- S2-4 value chain due diligence
    4. TargetTracking        -- S2-5 target progress
    5. ReportGeneration      -- Compile full S2 disclosure

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

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


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
    """Phases of the S2 value chain workers workflow."""
    POLICY_ASSESSMENT = "policy_assessment"
    ENGAGEMENT_EVALUATION = "engagement_evaluation"
    RISK_ASSESSMENT = "risk_assessment"
    TARGET_TRACKING = "target_tracking"
    REPORT_GENERATION = "report_generation"


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


class WorkerCategory(str, Enum):
    """Value chain worker category."""
    DIRECT_SUPPLIER = "direct_supplier"
    INDIRECT_SUPPLIER = "indirect_supplier"
    DOWNSTREAM_WORKER = "downstream_worker"
    AGENCY_WORKER = "agency_worker"
    SUBCONTRACTOR = "subcontractor"


class RiskLevel(str, Enum):
    """Human rights risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


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


class SupplierRecord(BaseModel):
    """Supplier assessment record for value chain workers."""
    supplier_id: str = Field(default_factory=lambda: f"sup-{_new_uuid()[:8]}")
    supplier_name: str = Field(default="")
    category: WorkerCategory = Field(default=WorkerCategory.DIRECT_SUPPLIER)
    country: str = Field(default="")
    worker_count_estimate: int = Field(default=0, ge=0)
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    due_diligence_completed: bool = Field(default=False)
    engagement_channel: str = Field(default="")
    issues_identified: List[str] = Field(default_factory=list)


class PolicyRecord(BaseModel):
    """Value chain worker policy record."""
    policy_id: str = Field(default_factory=lambda: f"pol-{_new_uuid()[:8]}")
    policy_name: str = Field(default="")
    covers_human_rights: bool = Field(default=False)
    covers_labour_rights: bool = Field(default=False)
    covers_child_labour: bool = Field(default=False)
    covers_forced_labour: bool = Field(default=False)
    aligned_with_ilo: bool = Field(default=False)
    aligned_with_ungp: bool = Field(default=False)
    scope_covers_value_chain: bool = Field(default=False)


class EngagementRecord(BaseModel):
    """Engagement process record."""
    engagement_id: str = Field(default_factory=lambda: f"eng-{_new_uuid()[:8]}")
    engagement_type: str = Field(default="", description="survey, audit, dialogue, etc.")
    supplier_count_covered: int = Field(default=0, ge=0)
    worker_count_reached: int = Field(default=0, ge=0)
    frequency: str = Field(default="annual")
    findings_count: int = Field(default=0, ge=0)


class ValueChainTarget(BaseModel):
    """Value chain workers target."""
    target_id: str = Field(default_factory=lambda: f"vt-{_new_uuid()[:8]}")
    target_description: str = Field(default="")
    metric: str = Field(default="")
    baseline_year: int = Field(default=2019)
    baseline_value: float = Field(default=0.0)
    target_year: int = Field(default=2030)
    target_value: float = Field(default=0.0)
    current_value: float = Field(default=0.0)


class S2ValueChainInput(BaseModel):
    """Input data model for S2ValueChainWorkflow."""
    supplier_data: List[SupplierRecord] = Field(
        default_factory=list, description="Supplier assessment records"
    )
    policy_data: List[PolicyRecord] = Field(
        default_factory=list, description="Value chain worker policies"
    )
    engagement_data: List[EngagementRecord] = Field(
        default_factory=list, description="Engagement process records"
    )
    targets: List[ValueChainTarget] = Field(
        default_factory=list, description="S2-5 targets"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    incidents: List[Dict[str, Any]] = Field(
        default_factory=list, description="S2-3 remediation incidents"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class S2ValueChainWorkflowResult(BaseModel):
    """Complete result from S2 value chain workers workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="s2_value_chain_workers")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    policies_assessed: int = Field(default=0)
    policy_alignment_pct: float = Field(default=0.0)
    suppliers_assessed: int = Field(default=0)
    high_risk_suppliers: int = Field(default=0)
    engagement_coverage_pct: float = Field(default=0.0)
    due_diligence_coverage_pct: float = Field(default=0.0)
    target_progress: Dict[str, float] = Field(default_factory=dict)
    incidents_count: int = Field(default=0)
    quality_issues: List[str] = Field(default_factory=list)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class S2ValueChainWorkflow:
    """
    5-phase S2 Workers in the Value Chain workflow.

    Implements end-to-end assessment of value chain worker conditions covering
    policy alignment review, engagement process evaluation, due diligence risk
    assessment, target progress tracking, and S2 disclosure report generation.

    Zero-hallucination: all coverage and risk calculations use deterministic
    arithmetic. No LLM in numeric calculation paths.

    Example:
        >>> wf = S2ValueChainWorkflow()
        >>> inp = S2ValueChainInput(supplier_data=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.due_diligence_coverage_pct >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize S2ValueChainWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._suppliers: List[SupplierRecord] = []
        self._target_progress: Dict[str, float] = {}
        self._quality_issues: List[str] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.POLICY_ASSESSMENT.value, "description": "S2-1 policy scope and alignment"},
            {"name": WorkflowPhase.ENGAGEMENT_EVALUATION.value, "description": "S2-2 engagement coverage"},
            {"name": WorkflowPhase.RISK_ASSESSMENT.value, "description": "S2-4 value chain due diligence"},
            {"name": WorkflowPhase.TARGET_TRACKING.value, "description": "S2-5 target progress"},
            {"name": WorkflowPhase.REPORT_GENERATION.value, "description": "Compile full S2 disclosure"},
        ]

    def validate_inputs(self, input_data: S2ValueChainInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.supplier_data:
            issues.append("No supplier data provided")
        if not input_data.policy_data:
            issues.append("No policy data provided for S2-1")
        return issues

    async def execute(
        self,
        input_data: Optional[S2ValueChainInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> S2ValueChainWorkflowResult:
        """
        Execute the 5-phase S2 value chain workers workflow.

        Args:
            input_data: Full input model (preferred).
            config: Configuration overrides.

        Returns:
            S2ValueChainWorkflowResult with policy, engagement, and risk metrics.
        """
        if input_data is None:
            input_data = S2ValueChainInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting S2 value chain workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_policy_assessment(input_data))
            phases_done += 1
            phase_results.append(await self._phase_engagement_evaluation(input_data))
            phases_done += 1
            phase_results.append(await self._phase_risk_assessment(input_data))
            phases_done += 1
            phase_results.append(await self._phase_target_tracking(input_data))
            phases_done += 1
            phase_results.append(await self._phase_report_generation(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("S2 value chain workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        self._suppliers = list(input_data.supplier_data)
        high_risk = sum(1 for s in self._suppliers if s.risk_level in (RiskLevel.HIGH, RiskLevel.VERY_HIGH))
        dd_done = sum(1 for s in self._suppliers if s.due_diligence_completed)
        dd_pct = round((dd_done / len(self._suppliers) * 100) if self._suppliers else 0.0, 2)

        # Policy alignment
        alignment_scores: List[float] = []
        for p in input_data.policy_data:
            score = sum([
                p.covers_human_rights, p.covers_labour_rights,
                p.covers_child_labour, p.covers_forced_labour,
                p.aligned_with_ilo, p.aligned_with_ungp,
                p.scope_covers_value_chain,
            ]) / 7.0 * 100
            alignment_scores.append(score)
        policy_alignment = round(sum(alignment_scores) / len(alignment_scores), 2) if alignment_scores else 0.0

        # Engagement coverage
        total_workers = sum(s.worker_count_estimate for s in self._suppliers)
        engaged_workers = sum(e.worker_count_reached for e in input_data.engagement_data)
        engagement_pct = round((engaged_workers / total_workers * 100) if total_workers > 0 else 0.0, 2)

        result = S2ValueChainWorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            policies_assessed=len(input_data.policy_data),
            policy_alignment_pct=policy_alignment,
            suppliers_assessed=len(self._suppliers),
            high_risk_suppliers=high_risk,
            engagement_coverage_pct=engagement_pct,
            due_diligence_coverage_pct=dd_pct,
            target_progress=self._target_progress,
            incidents_count=len(input_data.incidents),
            quality_issues=self._quality_issues,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "S2 value chain %s completed in %.2fs: %d suppliers, DD=%.1f%%",
            self.workflow_id, elapsed, len(self._suppliers), dd_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Policy Assessment (S2-1)
    # -------------------------------------------------------------------------

    async def _phase_policy_assessment(self, input_data: S2ValueChainInput) -> PhaseResult:
        """Assess value chain worker policies for scope and alignment."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        policies = input_data.policy_data
        outputs["policies_count"] = len(policies)
        outputs["covers_human_rights"] = sum(1 for p in policies if p.covers_human_rights)
        outputs["covers_labour_rights"] = sum(1 for p in policies if p.covers_labour_rights)
        outputs["aligned_ilo"] = sum(1 for p in policies if p.aligned_with_ilo)
        outputs["aligned_ungp"] = sum(1 for p in policies if p.aligned_with_ungp)
        outputs["covers_value_chain"] = sum(1 for p in policies if p.scope_covers_value_chain)

        if not policies:
            warnings.append("No value chain worker policies defined (S2-1)")
        if not any(p.covers_child_labour for p in policies):
            warnings.append("No policy explicitly covers child labour")
        if not any(p.covers_forced_labour for p in policies):
            warnings.append("No policy explicitly covers forced labour")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 1 PolicyAssessment: %d policies", len(policies))
        return PhaseResult(
            phase_name=WorkflowPhase.POLICY_ASSESSMENT.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Engagement Evaluation (S2-2)
    # -------------------------------------------------------------------------

    async def _phase_engagement_evaluation(self, input_data: S2ValueChainInput) -> PhaseResult:
        """Evaluate engagement processes with value chain workers."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        engagements = input_data.engagement_data
        total_suppliers = len(input_data.supplier_data)
        covered_suppliers = sum(e.supplier_count_covered for e in engagements)
        coverage_pct = round((covered_suppliers / total_suppliers * 100) if total_suppliers > 0 else 0.0, 2)

        total_workers = sum(s.worker_count_estimate for s in input_data.supplier_data)
        reached_workers = sum(e.worker_count_reached for e in engagements)
        worker_coverage_pct = round((reached_workers / total_workers * 100) if total_workers > 0 else 0.0, 2)

        outputs["engagements_count"] = len(engagements)
        outputs["supplier_coverage_pct"] = coverage_pct
        outputs["worker_coverage_pct"] = worker_coverage_pct
        outputs["total_findings"] = sum(e.findings_count for e in engagements)

        if not engagements:
            warnings.append("No engagement processes documented (S2-2)")
        if coverage_pct < 50.0 and total_suppliers > 0:
            warnings.append(f"Low supplier engagement coverage: {coverage_pct}%")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 2 EngagementEvaluation: %.1f%% supplier coverage", coverage_pct)
        return PhaseResult(
            phase_name=WorkflowPhase.ENGAGEMENT_EVALUATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Risk Assessment (S2-4)
    # -------------------------------------------------------------------------

    async def _phase_risk_assessment(self, input_data: S2ValueChainInput) -> PhaseResult:
        """Perform value chain due diligence and risk assessment."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        suppliers = input_data.supplier_data
        risk_dist: Dict[str, int] = {}
        for s in suppliers:
            risk_dist[s.risk_level.value] = risk_dist.get(s.risk_level.value, 0) + 1

        dd_completed = sum(1 for s in suppliers if s.due_diligence_completed)
        dd_pct = round((dd_completed / len(suppliers) * 100) if suppliers else 0.0, 2)
        high_risk = [s for s in suppliers if s.risk_level in (RiskLevel.HIGH, RiskLevel.VERY_HIGH)]
        high_risk_no_dd = [s for s in high_risk if not s.due_diligence_completed]

        all_issues: List[str] = []
        for s in suppliers:
            all_issues.extend(s.issues_identified)

        by_country: Dict[str, int] = {}
        for s in suppliers:
            if s.country:
                by_country[s.country] = by_country.get(s.country, 0) + 1

        outputs["suppliers_assessed"] = len(suppliers)
        outputs["risk_distribution"] = risk_dist
        outputs["due_diligence_completed"] = dd_completed
        outputs["due_diligence_pct"] = dd_pct
        outputs["high_risk_count"] = len(high_risk)
        outputs["high_risk_without_dd"] = len(high_risk_no_dd)
        outputs["total_issues_identified"] = len(all_issues)
        outputs["countries_covered"] = len(by_country)

        if high_risk_no_dd:
            warnings.append(f"{len(high_risk_no_dd)} high-risk suppliers without due diligence")
            self._quality_issues.append("CRITICAL: High-risk suppliers lack due diligence")
        if dd_pct < 50.0 and suppliers:
            warnings.append(f"Low due diligence coverage: {dd_pct}%")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 3 RiskAssessment: %d suppliers, %d high risk", len(suppliers), len(high_risk))
        return PhaseResult(
            phase_name=WorkflowPhase.RISK_ASSESSMENT.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Target Tracking (S2-5)
    # -------------------------------------------------------------------------

    async def _phase_target_tracking(self, input_data: S2ValueChainInput) -> PhaseResult:
        """Track progress against value chain worker targets."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._target_progress = {}

        for target in input_data.targets:
            if target.target_value != target.baseline_value:
                change_needed = target.target_value - target.baseline_value
                change_achieved = target.current_value - target.baseline_value
                progress_pct = round(
                    (change_achieved / change_needed * 100) if change_needed != 0 else 100.0, 2
                )
            else:
                progress_pct = 100.0 if target.current_value >= target.target_value else 0.0
            self._target_progress[target.target_id] = progress_pct
            if progress_pct < 0:
                warnings.append(f"Target {target.target_id}: regressing from baseline")

        outputs["targets_assessed"] = len(self._target_progress)
        outputs["target_progress"] = self._target_progress
        outputs["on_track_count"] = sum(1 for p in self._target_progress.values() if p >= 50.0)

        if not input_data.targets:
            warnings.append("No value chain worker targets defined (S2-5)")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 4 TargetTracking: %d targets assessed", len(self._target_progress))
        return PhaseResult(
            phase_name=WorkflowPhase.TARGET_TRACKING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(self, input_data: S2ValueChainInput) -> PhaseResult:
        """Compile full S2 value chain workers disclosure."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        suppliers = input_data.supplier_data
        high_risk = sum(1 for s in suppliers if s.risk_level in (RiskLevel.HIGH, RiskLevel.VERY_HIGH))
        dd_done = sum(1 for s in suppliers if s.due_diligence_completed)

        outputs["s2_disclosure"] = {
            "s2_1_policies_count": len(input_data.policy_data),
            "s2_2_engagements_count": len(input_data.engagement_data),
            "s2_3_incidents_count": len(input_data.incidents),
            "s2_4_suppliers_assessed": len(suppliers),
            "s2_4_high_risk_suppliers": high_risk,
            "s2_4_due_diligence_completed": dd_done,
            "s2_5_targets": self._target_progress,
            "reporting_year": input_data.reporting_year,
            "entity_name": input_data.entity_name,
        }
        outputs["report_ready"] = True

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 5 ReportGeneration: S2 disclosure ready")
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_GENERATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: S2ValueChainWorkflowResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
