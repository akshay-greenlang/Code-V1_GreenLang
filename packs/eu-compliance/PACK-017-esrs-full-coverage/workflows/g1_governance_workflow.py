# -*- coding: utf-8 -*-
"""
ESRS G1 Business Conduct Workflow
====================================

6-phase workflow for ESRS G1 Business Conduct disclosure covering policy
review, supplier management, corruption assessment, political influence
evaluation, payment practices analysis, and report assembly with full
provenance tracking.

Phases:
    1. PolicyReview         -- Review business conduct policies (G1-1)
    2. SupplierManagement   -- Evaluate supplier relationships (G1-2)
    3. CorruptionAssessment -- Assess anti-corruption measures (G1-3, G1-4)
    4. PoliticalInfluence   -- Evaluate political engagement/lobbying (G1-5)
    5. PaymentPractices     -- Analyse payment practices (G1-6)
    6. ReportAssembly       -- Assemble complete G1 disclosure

ESRS G1 Disclosure Requirements (6 DRs):
    G1-1: Business conduct policies and corporate culture
    G1-2: Management of relationships with suppliers
    G1-3: Prevention and detection of corruption and bribery
    G1-4: Incidents of corruption or bribery
    G1-5: Political influence and lobbying activities
    G1-6: Payment practices

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
    """Phases of the G1 business conduct workflow."""
    POLICY_REVIEW = "policy_review"
    SUPPLIER_MANAGEMENT = "supplier_management"
    CORRUPTION_ASSESSMENT = "corruption_assessment"
    POLITICAL_INFLUENCE = "political_influence"
    PAYMENT_PRACTICES = "payment_practices"
    REPORT_ASSEMBLY = "report_assembly"

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

class CorruptionRiskLevel(str, Enum):
    """Anti-corruption risk level."""
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

class CorruptionIncident(BaseModel):
    """Corruption or bribery incident per G1-4."""
    incident_id: str = Field(default_factory=lambda: f"cor-{_new_uuid()[:8]}")
    description: str = Field(default="")
    risk_level: CorruptionRiskLevel = Field(default=CorruptionRiskLevel.MEDIUM)
    occurred_year: int = Field(default=2025)
    legal_proceedings: bool = Field(default=False)
    fines_eur: float = Field(default=0.0, ge=0.0)
    employees_dismissed: int = Field(default=0, ge=0)
    resolved: bool = Field(default=False)

class PoliticalContribution(BaseModel):
    """Political contribution or lobbying record per G1-5."""
    contribution_id: str = Field(default_factory=lambda: f"pol-{_new_uuid()[:8]}")
    recipient: str = Field(default="")
    contribution_type: str = Field(default="", description="monetary, in_kind, lobbying")
    amount_eur: float = Field(default=0.0, ge=0.0)
    topic: str = Field(default="", description="Topic of political engagement")
    registered_lobbyist: bool = Field(default=False)

class PaymentMetrics(BaseModel):
    """Payment practices metrics per G1-6."""
    avg_payment_days: float = Field(default=0.0, ge=0.0, description="Average days to pay suppliers")
    late_payment_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    invoices_paid_within_terms: int = Field(default=0, ge=0)
    invoices_total: int = Field(default=0, ge=0)
    standard_payment_terms_days: int = Field(default=30, ge=0)
    sme_payment_days: float = Field(default=0.0, ge=0.0, description="Avg days for SME suppliers")

class G1GovernanceInput(BaseModel):
    """Input data model for G1GovernanceWorkflow."""
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    g1_is_material: bool = Field(default=True, description="Whether G1 is material")
    policies: List[Dict[str, Any]] = Field(
        default_factory=list, description="G1-1 business conduct policies"
    )
    supplier_data: Dict[str, Any] = Field(
        default_factory=dict, description="G1-2 supplier management data"
    )
    anti_corruption_data: Dict[str, Any] = Field(
        default_factory=dict, description="G1-3 anti-corruption measures"
    )
    corruption_incidents: List[CorruptionIncident] = Field(
        default_factory=list, description="G1-4 corruption incidents"
    )
    political_contributions: List[PoliticalContribution] = Field(
        default_factory=list, description="G1-5 political contributions"
    )
    lobbying_data: Dict[str, Any] = Field(
        default_factory=dict, description="G1-5 lobbying activities"
    )
    payment_metrics: Optional[PaymentMetrics] = Field(
        default=None, description="G1-6 payment practices"
    )
    whistleblower_data: Dict[str, Any] = Field(
        default_factory=dict, description="Whistleblower channel data"
    )
    config: Dict[str, Any] = Field(default_factory=dict)

class G1GovernanceWorkflowResult(BaseModel):
    """Complete result from G1 governance workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="g1_governance")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    g1_is_material: bool = Field(default=True)
    policies_count: int = Field(default=0)
    has_anti_corruption_policy: bool = Field(default=False)
    corruption_incidents_count: int = Field(default=0)
    total_fines_eur: float = Field(default=0.0)
    political_contributions_total_eur: float = Field(default=0.0)
    avg_payment_days: float = Field(default=0.0)
    late_payment_pct: float = Field(default=0.0)
    has_whistleblower_channel: bool = Field(default=False)
    overall_completeness_pct: float = Field(default=0.0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class G1GovernanceWorkflow:
    """
    6-phase ESRS G1 Business Conduct workflow.

    Orchestrates policy review, supplier management, corruption assessment,
    political influence evaluation, payment practices analysis, and report
    assembly for complete G1 disclosure covering G1-1 through G1-6.

    Zero-hallucination: all financial and metric aggregations use deterministic
    arithmetic. No LLM in numeric calculation paths.

    Example:
        >>> wf = G1GovernanceWorkflow()
        >>> inp = G1GovernanceInput(policies=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.overall_completeness_pct >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize G1GovernanceWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._sub_results: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.POLICY_REVIEW.value, "description": "Review business conduct policies"},
            {"name": WorkflowPhase.SUPPLIER_MANAGEMENT.value, "description": "Evaluate supplier relationships"},
            {"name": WorkflowPhase.CORRUPTION_ASSESSMENT.value, "description": "Assess anti-corruption measures"},
            {"name": WorkflowPhase.POLITICAL_INFLUENCE.value, "description": "Evaluate political engagement"},
            {"name": WorkflowPhase.PAYMENT_PRACTICES.value, "description": "Analyse payment practices"},
            {"name": WorkflowPhase.REPORT_ASSEMBLY.value, "description": "Assemble G1 disclosure"},
        ]

    def validate_inputs(self, input_data: G1GovernanceInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.g1_is_material:
            issues.append("G1 is not material; full disclosure not required")
        if not input_data.policies:
            issues.append("No business conduct policies provided")
        return issues

    async def execute(
        self,
        input_data: Optional[G1GovernanceInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> G1GovernanceWorkflowResult:
        """
        Execute the 6-phase G1 business conduct workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            G1GovernanceWorkflowResult with business conduct assessment.
        """
        if input_data is None:
            input_data = G1GovernanceInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting G1 governance workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_policy_review(input_data))
            phase_results.append(await self._phase_supplier_management(input_data))
            phase_results.append(await self._phase_corruption_assessment(input_data))
            phase_results.append(await self._phase_political_influence(input_data))
            phase_results.append(await self._phase_payment_practices(input_data))
            phase_results.append(await self._phase_report_assembly(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("G1 governance workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        total_fines = sum(c.fines_eur for c in input_data.corruption_incidents)
        total_political = sum(p.amount_eur for p in input_data.political_contributions)
        has_ac_policy = any(p.get("type") == "anti_corruption" for p in input_data.policies)
        completeness = self._calculate_completeness(input_data)

        result = G1GovernanceWorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            g1_is_material=input_data.g1_is_material,
            policies_count=len(input_data.policies),
            has_anti_corruption_policy=has_ac_policy,
            corruption_incidents_count=len(input_data.corruption_incidents),
            total_fines_eur=round(total_fines, 2),
            political_contributions_total_eur=round(total_political, 2),
            avg_payment_days=input_data.payment_metrics.avg_payment_days if input_data.payment_metrics else 0.0,
            late_payment_pct=input_data.payment_metrics.late_payment_pct if input_data.payment_metrics else 0.0,
            has_whistleblower_channel=bool(input_data.whistleblower_data),
            overall_completeness_pct=completeness,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "G1 governance %s completed in %.2fs: %d incidents, fines=%.0f EUR",
            self.workflow_id, elapsed, len(input_data.corruption_incidents), total_fines,
        )
        return result

    async def _phase_policy_review(self, input_data: G1GovernanceInput) -> PhaseResult:
        """Review business conduct policies and corporate culture (G1-1)."""
        started = utcnow()
        outputs: Dict[str, Any] = {"policies_count": len(input_data.policies)}
        warnings: List[str] = []
        outputs["has_code_of_conduct"] = any(p.get("type") == "code_of_conduct" for p in input_data.policies)
        outputs["has_anti_corruption_policy"] = any(p.get("type") == "anti_corruption" for p in input_data.policies)
        outputs["has_whistleblower_policy"] = any(p.get("type") == "whistleblower" for p in input_data.policies)
        outputs["has_whistleblower_channel"] = bool(input_data.whistleblower_data)
        if not input_data.policies:
            warnings.append("No business conduct policies defined (G1-1)")
        if not outputs["has_whistleblower_channel"]:
            warnings.append("No whistleblower channel established")
        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 1 PolicyReview: %d policies", len(input_data.policies))
        return PhaseResult(
            phase_name=WorkflowPhase.POLICY_REVIEW.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    async def _phase_supplier_management(self, input_data: G1GovernanceInput) -> PhaseResult:
        """Evaluate supplier relationship management (G1-2)."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        data = input_data.supplier_data
        outputs["has_supplier_data"] = bool(data)
        outputs["total_suppliers"] = data.get("total_suppliers", 0)
        outputs["assessed_suppliers"] = data.get("assessed_suppliers", 0)
        outputs["critical_suppliers"] = data.get("critical_suppliers", 0)
        outputs["terminated_for_violations"] = data.get("terminated_for_violations", 0)
        assessment_pct = round(
            (data.get("assessed_suppliers", 0) / data.get("total_suppliers", 1) * 100)
            if data.get("total_suppliers", 0) > 0 else 0.0, 1
        )
        outputs["supplier_assessment_coverage_pct"] = assessment_pct
        if not data:
            warnings.append("No supplier management data provided (G1-2)")
        if assessment_pct < 50:
            warnings.append(f"Supplier assessment coverage at {assessment_pct}% (below 50%)")
        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 2 SupplierManagement: %d suppliers", outputs["total_suppliers"])
        return PhaseResult(
            phase_name=WorkflowPhase.SUPPLIER_MANAGEMENT.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    async def _phase_corruption_assessment(self, input_data: G1GovernanceInput) -> PhaseResult:
        """Assess anti-corruption measures and incidents (G1-3, G1-4)."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        ac_data = input_data.anti_corruption_data
        incidents = input_data.corruption_incidents
        outputs["has_anti_corruption_program"] = bool(ac_data)
        outputs["training_coverage_pct"] = ac_data.get("training_coverage_pct", 0)
        outputs["risk_assessments_completed"] = ac_data.get("risk_assessments_completed", 0)
        outputs["incidents_count"] = len(incidents)
        outputs["total_fines_eur"] = round(sum(c.fines_eur for c in incidents), 2)
        outputs["legal_proceedings"] = sum(1 for c in incidents if c.legal_proceedings)
        outputs["employees_dismissed"] = sum(c.employees_dismissed for c in incidents)
        outputs["by_risk_level"] = {
            level.value: sum(1 for c in incidents if c.risk_level == level)
            for level in CorruptionRiskLevel
        }
        if incidents:
            warnings.append(f"{len(incidents)} corruption/bribery incidents reported")
        if not ac_data:
            warnings.append("No anti-corruption program data provided (G1-3)")
        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 3 CorruptionAssessment: %d incidents", len(incidents))
        return PhaseResult(
            phase_name=WorkflowPhase.CORRUPTION_ASSESSMENT.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    async def _phase_political_influence(self, input_data: G1GovernanceInput) -> PhaseResult:
        """Evaluate political engagement and lobbying activities (G1-5)."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        contributions = input_data.political_contributions
        outputs["contributions_count"] = len(contributions)
        outputs["total_amount_eur"] = round(sum(c.amount_eur for c in contributions), 2)
        outputs["by_type"] = {}
        for c in contributions:
            ct = c.contribution_type or "unspecified"
            outputs["by_type"][ct] = outputs["by_type"].get(ct, 0) + 1
        outputs["registered_lobbyists"] = sum(1 for c in contributions if c.registered_lobbyist)
        outputs["has_lobbying_data"] = bool(input_data.lobbying_data)
        outputs["lobbying_expenditure_eur"] = input_data.lobbying_data.get("total_expenditure_eur", 0)
        if contributions and not all(c.registered_lobbyist for c in contributions if c.contribution_type == "lobbying"):
            warnings.append("Some lobbying activities not registered")
        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 4 PoliticalInfluence: %d contributions, %.0f EUR",
                         len(contributions), outputs["total_amount_eur"])
        return PhaseResult(
            phase_name=WorkflowPhase.POLITICAL_INFLUENCE.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    async def _phase_payment_practices(self, input_data: G1GovernanceInput) -> PhaseResult:
        """Analyse payment practices (G1-6)."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        pm = input_data.payment_metrics
        if pm:
            outputs["avg_payment_days"] = pm.avg_payment_days
            outputs["late_payment_pct"] = pm.late_payment_pct
            outputs["invoices_paid_within_terms"] = pm.invoices_paid_within_terms
            outputs["invoices_total"] = pm.invoices_total
            outputs["standard_payment_terms_days"] = pm.standard_payment_terms_days
            outputs["sme_payment_days"] = pm.sme_payment_days
            on_time_pct = round(
                (pm.invoices_paid_within_terms / pm.invoices_total * 100)
                if pm.invoices_total > 0 else 0.0, 1
            )
            outputs["on_time_payment_pct"] = on_time_pct
            if pm.avg_payment_days > 60:
                warnings.append(f"Average payment of {pm.avg_payment_days} days exceeds 60-day threshold")
            if pm.late_payment_pct > 20:
                warnings.append(f"Late payment rate of {pm.late_payment_pct}% exceeds 20% threshold")
        else:
            outputs["avg_payment_days"] = 0
            warnings.append("No payment practices data provided (G1-6)")
        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 5 PaymentPractices: avg=%s days",
                         outputs["avg_payment_days"])
        return PhaseResult(
            phase_name=WorkflowPhase.PAYMENT_PRACTICES.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    async def _phase_report_assembly(self, input_data: G1GovernanceInput) -> PhaseResult:
        """Assemble complete G1 disclosure from all phase results."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        dr_available = sum([
            bool(input_data.policies),               # G1-1
            bool(input_data.supplier_data),           # G1-2
            bool(input_data.anti_corruption_data),    # G1-3
            True,                                     # G1-4 (incidents list, always countable)
            bool(input_data.political_contributions) or bool(input_data.lobbying_data),  # G1-5
            input_data.payment_metrics is not None,   # G1-6
        ])
        outputs["drs_with_data"] = dr_available
        outputs["drs_total"] = 6
        outputs["completeness_pct"] = round((dr_available / 6 * 100), 1)
        outputs["disclosure_ready"] = dr_available >= 5
        if dr_available < 6:
            warnings.append(f"{6 - dr_available} G1 disclosure requirements missing data")
        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 6 ReportAssembly: %d/6 DRs with data", dr_available)
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_ASSEMBLY.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calculate_completeness(self, input_data: G1GovernanceInput) -> float:
        """Calculate overall G1 completeness percentage."""
        scores: List[float] = []
        scores.append(100.0 if input_data.policies else 0.0)
        scores.append(100.0 if input_data.supplier_data else 0.0)
        scores.append(100.0 if input_data.anti_corruption_data else 0.0)
        scores.append(100.0)  # G1-4 incidents always countable
        scores.append(100.0 if input_data.political_contributions or input_data.lobbying_data else 0.0)
        scores.append(100.0 if input_data.payment_metrics else 0.0)
        return round(sum(scores) / len(scores), 1) if scores else 0.0

    def _compute_provenance(self, result: G1GovernanceWorkflowResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
