# -*- coding: utf-8 -*-
"""
Audit Preparation Workflow
============================

Four-phase workflow for preparing comprehensive audit packages in response
to competent authority inspections or third-party audits.

This workflow enables:
- Systematic evidence assembly from all data sources
- Gap analysis against regulatory requirements
- Remediation action tracking
- Inspection-ready package generation

Phases:
    1. Evidence Assembly - Collect all DDS, documents, and supporting data
    2. Gap Analysis - Identify missing or incomplete information
    3. Remediation Actions - Address gaps before audit
    4. Inspection Package Generation - Create auditor-ready deliverables

Regulatory Context:
    EUDR Article 13 grants competent authorities inspection powers. Article 14
    requires operators to maintain records for 5 years. This workflow ensures
    operators can respond efficiently to audit requests with complete evidence.

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
    EVIDENCE_ASSEMBLY = "evidence_assembly"
    GAP_ANALYSIS = "gap_analysis"
    REMEDIATION_ACTIONS = "remediation_actions"
    INSPECTION_PACKAGE_GENERATION = "inspection_package_generation"


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EvidenceType(str, Enum):
    """Types of audit evidence."""
    DDS = "dds"
    SUPPLIER_PROFILE = "supplier_profile"
    GEOLOCATION = "geolocation"
    CERTIFICATION = "certification"
    AUDIT_REPORT = "audit_report"
    INVOICE = "invoice"
    RISK_ASSESSMENT = "risk_assessment"
    MITIGATION_PLAN = "mitigation_plan"


class GapSeverity(str, Enum):
    """Severity of compliance gaps."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# DATA MODELS
# =============================================================================


class AuditPreparationConfig(BaseModel):
    """Configuration for audit preparation workflow."""
    audit_start_date: Optional[str] = Field(None, description="Scheduled audit date (YYYY-MM-DD)")
    audit_scope_period_months: int = Field(default=12, ge=1, description="Audit period lookback")
    include_all_suppliers: bool = Field(default=True, description="Include all suppliers in scope")
    evidence_retention_years: int = Field(default=5, ge=1, description="Record retention period")
    operator_id: Optional[str] = Field(None, description="Operator context")


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
    config: AuditPreparationConfig = Field(default_factory=AuditPreparationConfig)
    phase_results: List[PhaseResult] = Field(default_factory=list, description="Completed phase results")
    state: Dict[str, Any] = Field(default_factory=dict, description="Shared state data")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Workflow start time")

    class Config:
        arbitrary_types_allowed = True


class WorkflowResult(BaseModel):
    """Complete result from the audit preparation workflow."""
    workflow_name: str = Field(default="audit_preparation", description="Workflow identifier")
    phases: List[PhaseResult] = Field(default_factory=list, description="All phase results")
    overall_status: PhaseStatus = Field(..., description="Overall workflow status")
    total_duration_seconds: float = Field(default=0.0, ge=0.0, description="Total execution time")
    provenance_hash: str = Field(default="", description="Workflow-level provenance hash")
    execution_id: str = Field(..., description="Execution identifier")
    evidence_items_collected: int = Field(default=0, ge=0, description="Evidence count")
    gaps_identified: int = Field(default=0, ge=0, description="Compliance gaps")
    critical_gaps: int = Field(default=0, ge=0, description="Critical severity gaps")
    remediation_actions_completed: int = Field(default=0, ge=0, description="Actions addressed")
    audit_ready: bool = Field(default=False, description="Ready for inspection")
    package_file_path: Optional[str] = Field(None, description="Inspection package location")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


# =============================================================================
# AUDIT PREPARATION WORKFLOW
# =============================================================================


class AuditPreparationWorkflow:
    """
    Four-phase audit preparation workflow.

    Systematically prepares for competent authority inspections:
    - Evidence collection from all compliance systems
    - Regulatory gap identification and severity classification
    - Remediation action planning and tracking
    - Inspection-ready package generation with provenance

    Example:
        >>> config = AuditPreparationConfig(
        ...     audit_start_date="2026-06-01",
        ...     audit_scope_period_months=12,
        ... )
        >>> workflow = AuditPreparationWorkflow(config)
        >>> result = await workflow.run(WorkflowContext(config=config))
        >>> assert result.audit_ready is True
    """

    def __init__(self, config: Optional[AuditPreparationConfig] = None) -> None:
        """Initialize the audit preparation workflow."""
        self.config = config or AuditPreparationConfig()
        self.logger = logging.getLogger(f"{__name__}.AuditPreparationWorkflow")

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """
        Execute the full 4-phase audit preparation workflow.

        Args:
            context: Workflow context with configuration and initial state.

        Returns:
            WorkflowResult with evidence, gaps, remediation, and package.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting audit preparation workflow execution_id=%s audit_date=%s",
            context.execution_id,
            self.config.audit_start_date or "TBD",
        )

        context.config = self.config

        phase_handlers = [
            (Phase.EVIDENCE_ASSEMBLY, self._phase_1_evidence_assembly),
            (Phase.GAP_ANALYSIS, self._phase_2_gap_analysis),
            (Phase.REMEDIATION_ACTIONS, self._phase_3_remediation_actions),
            (Phase.INSPECTION_PACKAGE_GENERATION, self._phase_4_inspection_package_generation),
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

        # Extract final outputs
        evidence = context.state.get("evidence", [])
        gaps = context.state.get("gaps", [])
        critical_gaps = len([g for g in gaps if g.get("severity") == GapSeverity.CRITICAL.value])
        remediation = context.state.get("remediation_actions", [])
        completed_actions = len([a for a in remediation if a.get("status") == "completed"])
        audit_ready = context.state.get("audit_ready", False)
        package_path = context.state.get("package_file_path")

        provenance = self._hash({
            "execution_id": context.execution_id,
            "phases": [p.provenance_hash for p in context.phase_results],
            "evidence_count": len(evidence),
        })

        self.logger.info(
            "Audit preparation finished execution_id=%s status=%s "
            "evidence=%d gaps=%d audit_ready=%s",
            context.execution_id,
            overall_status.value,
            len(evidence),
            len(gaps),
            audit_ready,
        )

        return WorkflowResult(
            phases=context.phase_results,
            overall_status=overall_status,
            total_duration_seconds=total_duration,
            provenance_hash=provenance,
            execution_id=context.execution_id,
            evidence_items_collected=len(evidence),
            gaps_identified=len(gaps),
            critical_gaps=critical_gaps,
            remediation_actions_completed=completed_actions,
            audit_ready=audit_ready,
            package_file_path=package_path,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Evidence Assembly
    # -------------------------------------------------------------------------

    async def _phase_1_evidence_assembly(self, context: WorkflowContext) -> PhaseResult:
        """
        Collect all DDS, documents, and supporting data.

        Evidence sources:
        - DDS submissions (last N months)
        - Supplier profiles and contracts
        - Geolocation data (plots, coordinates, polygons)
        - Certifications (FSC, PEFC, RSPO, etc.)
        - Third-party audit reports
        - Invoices and shipment documentation
        - Risk assessments and mitigation plans
        """
        phase = Phase.EVIDENCE_ASSEMBLY
        scope_months = self.config.audit_scope_period_months

        self.logger.info("Assembling audit evidence (scope=%d months)", scope_months)

        await asyncio.sleep(0.1)

        # Simulate evidence collection
        evidence = []
        cutoff_date = datetime.utcnow() - timedelta(days=scope_months * 30)

        # DDS submissions
        dds_count = random.randint(10, 100)
        for i in range(dds_count):
            evidence.append({
                "evidence_id": f"EV-DDS-{uuid.uuid4().hex[:8]}",
                "evidence_type": EvidenceType.DDS.value,
                "document_id": f"DDS-{uuid.uuid4().hex[:8]}",
                "submission_date": (cutoff_date + timedelta(days=random.randint(0, scope_months * 30))).isoformat(),
                "reference_number": f"EUIS-{uuid.uuid4().hex[:8].upper()}",
            })

        # Supplier profiles
        supplier_count = random.randint(20, 200)
        for i in range(supplier_count):
            evidence.append({
                "evidence_id": f"EV-SUP-{uuid.uuid4().hex[:8]}",
                "evidence_type": EvidenceType.SUPPLIER_PROFILE.value,
                "supplier_id": f"SUP-{uuid.uuid4().hex[:8]}",
                "country": random.choice(["BR", "ID", "CO", "MY"]),
            })

        # Geolocations
        plot_count = random.randint(100, 1000)
        for i in range(plot_count):
            evidence.append({
                "evidence_id": f"EV-GEO-{uuid.uuid4().hex[:8]}",
                "evidence_type": EvidenceType.GEOLOCATION.value,
                "plot_id": f"PLOT-{uuid.uuid4().hex[:8]}",
                "has_polygon": random.choice([True, False]),
            })

        # Certifications
        cert_count = random.randint(50, 300)
        for i in range(cert_count):
            evidence.append({
                "evidence_id": f"EV-CERT-{uuid.uuid4().hex[:8]}",
                "evidence_type": EvidenceType.CERTIFICATION.value,
                "cert_id": f"CERT-{uuid.uuid4().hex[:8]}",
                "cert_type": random.choice(["FSC", "PEFC", "RSPO", "ISCC"]),
            })

        # Risk assessments
        risk_count = random.randint(10, 50)
        for i in range(risk_count):
            evidence.append({
                "evidence_id": f"EV-RISK-{uuid.uuid4().hex[:8]}",
                "evidence_type": EvidenceType.RISK_ASSESSMENT.value,
                "assessment_date": (cutoff_date + timedelta(days=random.randint(0, scope_months * 30))).isoformat(),
                "composite_score": random.uniform(0, 100),
            })

        context.state["evidence"] = evidence
        context.state["scope_start_date"] = cutoff_date.isoformat()
        context.state["scope_end_date"] = datetime.utcnow().isoformat()

        # Group by type
        by_type = {}
        for ev in evidence:
            ev_type = ev["evidence_type"]
            by_type[ev_type] = by_type.get(ev_type, 0) + 1

        provenance = self._hash({
            "phase": phase.value,
            "evidence_count": len(evidence),
            "scope_months": scope_months,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "evidence_items_collected": len(evidence),
                "by_type": by_type,
                "scope_months": scope_months,
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Gap Analysis
    # -------------------------------------------------------------------------

    async def _phase_2_gap_analysis(self, context: WorkflowContext) -> PhaseResult:
        """
        Identify missing or incomplete information.

        Gap checks:
        - All DDS have valid geolocation (Article 9)
        - All suppliers have risk assessments (Article 8)
        - Certifications are current (not expired)
        - Records are retained for 5 years (Article 14)
        - Mitigation measures documented for high-risk (Article 10)
        """
        phase = Phase.GAP_ANALYSIS
        evidence = context.state.get("evidence", [])

        self.logger.info("Analyzing compliance gaps from %d evidence items", len(evidence))

        gaps = []

        # Gap 1: Missing geolocation for DDS
        dds_evidence = [e for e in evidence if e["evidence_type"] == EvidenceType.DDS.value]
        geo_evidence = [e for e in evidence if e["evidence_type"] == EvidenceType.GEOLOCATION.value]
        if len(dds_evidence) > len(geo_evidence):
            gaps.append({
                "gap_id": f"GAP-{uuid.uuid4().hex[:8]}",
                "gap_type": "missing_geolocation",
                "severity": GapSeverity.CRITICAL.value,
                "description": f"{len(dds_evidence) - len(geo_evidence)} DDS submissions lack geolocation data",
                "article": "Article 9(1)(d)",
                "remediation_required": True,
            })

        # Gap 2: Suppliers without risk assessments
        supplier_evidence = [e for e in evidence if e["evidence_type"] == EvidenceType.SUPPLIER_PROFILE.value]
        risk_evidence = [e for e in evidence if e["evidence_type"] == EvidenceType.RISK_ASSESSMENT.value]
        if len(supplier_evidence) > len(risk_evidence) * 2:
            gaps.append({
                "gap_id": f"GAP-{uuid.uuid4().hex[:8]}",
                "gap_type": "missing_risk_assessment",
                "severity": GapSeverity.HIGH.value,
                "description": f"Risk assessments incomplete for ~{len(supplier_evidence) - len(risk_evidence)} suppliers",
                "article": "Article 8",
                "remediation_required": True,
            })

        # Gap 3: Expired certifications (simulated)
        if random.random() > 0.7:
            gaps.append({
                "gap_id": f"GAP-{uuid.uuid4().hex[:8]}",
                "gap_type": "expired_certifications",
                "severity": GapSeverity.MEDIUM.value,
                "description": f"{random.randint(1, 10)} supplier certifications have expired",
                "article": "Supporting evidence quality",
                "remediation_required": True,
            })

        # Gap 4: Incomplete mitigation plans
        if random.random() > 0.6:
            gaps.append({
                "gap_id": f"GAP-{uuid.uuid4().hex[:8]}",
                "gap_type": "missing_mitigation",
                "severity": GapSeverity.HIGH.value,
                "description": "Mitigation measures not documented for high-risk suppliers",
                "article": "Article 10(1)(d)",
                "remediation_required": True,
            })

        # Gap 5: Documentation retention (simulated check)
        retention_years = self.config.evidence_retention_years
        if random.random() > 0.8:
            gaps.append({
                "gap_id": f"GAP-{uuid.uuid4().hex[:8]}",
                "gap_type": "retention_policy",
                "severity": GapSeverity.LOW.value,
                "description": f"Record retention policy not consistently applied ({retention_years} years required)",
                "article": "Article 14",
                "remediation_required": False,
            })

        context.state["gaps"] = gaps

        # Group by severity
        by_severity = {}
        for gap in gaps:
            sev = gap["severity"]
            by_severity[sev] = by_severity.get(sev, 0) + 1

        provenance = self._hash({
            "phase": phase.value,
            "gap_count": len(gaps),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "gaps_identified": len(gaps),
                "by_severity": by_severity,
                "critical_gaps": by_severity.get(GapSeverity.CRITICAL.value, 0),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Remediation Actions
    # -------------------------------------------------------------------------

    async def _phase_3_remediation_actions(self, context: WorkflowContext) -> PhaseResult:
        """
        Address gaps before audit.

        Remediation actions:
        - Critical gaps: Immediate action required (halt audit if not resolved)
        - High gaps: Resolve before audit or provide explanation
        - Medium/Low gaps: Document as known issues with improvement plan
        """
        phase = Phase.REMEDIATION_ACTIONS
        gaps = context.state.get("gaps", [])

        self.logger.info("Planning remediation for %d gap(s)", len(gaps))

        remediation_actions = []

        for gap in gaps:
            severity = gap["severity"]
            gap_type = gap["gap_type"]

            action = {
                "action_id": f"REM-{uuid.uuid4().hex[:8]}",
                "gap_id": gap["gap_id"],
                "gap_type": gap_type,
                "severity": severity,
                "action_description": self._generate_remediation_action(gap_type, severity),
                "assigned_to": self._assign_remediation_owner(severity),
                "due_date": self._calculate_remediation_due_date(severity),
                "status": "completed" if severity != GapSeverity.CRITICAL.value else random.choice(["completed", "in_progress"]),
            }
            remediation_actions.append(action)

        context.state["remediation_actions"] = remediation_actions

        # Determine if audit-ready
        critical_incomplete = len([
            a for a in remediation_actions
            if a["severity"] == GapSeverity.CRITICAL.value and a["status"] != "completed"
        ])

        audit_ready = critical_incomplete == 0

        context.state["audit_ready"] = audit_ready

        # Count by status
        by_status = {}
        for action in remediation_actions:
            status = action["status"]
            by_status[status] = by_status.get(status, 0) + 1

        provenance = self._hash({
            "phase": phase.value,
            "action_count": len(remediation_actions),
            "audit_ready": audit_ready,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "remediation_actions_planned": len(remediation_actions),
                "by_status": by_status,
                "critical_incomplete": critical_incomplete,
                "audit_ready": audit_ready,
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Inspection Package Generation
    # -------------------------------------------------------------------------

    async def _phase_4_inspection_package_generation(self, context: WorkflowContext) -> PhaseResult:
        """
        Create auditor-ready deliverables.

        Package contents:
        - Executive summary
        - Evidence index (all collected items)
        - Gap analysis report
        - Remediation status
        - DDS register
        - Supplier master list
        - Geolocation register
        - Certification inventory
        - Provenance hashes for all documents
        """
        phase = Phase.INSPECTION_PACKAGE_GENERATION
        evidence = context.state.get("evidence", [])
        gaps = context.state.get("gaps", [])
        remediation = context.state.get("remediation_actions", [])
        audit_ready = context.state.get("audit_ready", False)

        self.logger.info("Generating inspection package")

        # Create package metadata
        package = {
            "package_id": f"PKG-{uuid.uuid4().hex[:8]}",
            "operator_id": self.config.operator_id,
            "audit_start_date": self.config.audit_start_date,
            "scope_period_months": self.config.audit_scope_period_months,
            "generated_at": datetime.utcnow().isoformat(),
            "audit_ready": audit_ready,
            "contents": {
                "executive_summary": self._generate_executive_summary(evidence, gaps, remediation),
                "evidence_index": len(evidence),
                "gap_analysis_report": len(gaps),
                "remediation_status": len(remediation),
                "dds_register": len([e for e in evidence if e["evidence_type"] == EvidenceType.DDS.value]),
                "supplier_master_list": len([e for e in evidence if e["evidence_type"] == EvidenceType.SUPPLIER_PROFILE.value]),
                "geolocation_register": len([e for e in evidence if e["evidence_type"] == EvidenceType.GEOLOCATION.value]),
                "certification_inventory": len([e for e in evidence if e["evidence_type"] == EvidenceType.CERTIFICATION.value]),
            },
            "provenance_hash": self._hash({
                "evidence": len(evidence),
                "gaps": len(gaps),
                "remediation": len(remediation),
            }),
        }

        # Simulate package file generation
        package_file_path = f"/audit_packages/{package['package_id']}.zip"
        context.state["package_file_path"] = package_file_path
        context.state["inspection_package"] = package

        provenance = self._hash({
            "phase": phase.value,
            "package_id": package["package_id"],
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "package_id": package["package_id"],
                "package_file_path": package_file_path,
                "audit_ready": audit_ready,
                "total_documents": sum(package["contents"].values()) if isinstance(package["contents"], dict) else 0,
            },
            provenance_hash=provenance,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _generate_remediation_action(self, gap_type: str, severity: str) -> str:
        """Generate remediation action description."""
        actions = {
            "missing_geolocation": "Request geolocation data from suppliers; validate coordinates",
            "missing_risk_assessment": "Conduct risk assessments for all suppliers missing data",
            "expired_certifications": "Request updated certifications from suppliers",
            "missing_mitigation": "Document mitigation measures for high-risk suppliers",
            "retention_policy": "Implement automated record retention policy",
        }
        return actions.get(gap_type, "Review and address compliance gap")

    def _assign_remediation_owner(self, severity: str) -> str:
        """Assign remediation owner based on severity."""
        if severity == GapSeverity.CRITICAL.value:
            return "compliance_director"
        elif severity == GapSeverity.HIGH.value:
            return "compliance_manager"
        return "compliance_analyst"

    def _calculate_remediation_due_date(self, severity: str) -> str:
        """Calculate remediation due date."""
        if severity == GapSeverity.CRITICAL.value:
            due_date = datetime.utcnow() + timedelta(days=7)
        elif severity == GapSeverity.HIGH.value:
            due_date = datetime.utcnow() + timedelta(days=30)
        else:
            due_date = datetime.utcnow() + timedelta(days=90)
        return due_date.isoformat()

    def _generate_executive_summary(
        self,
        evidence: List[Dict[str, Any]],
        gaps: List[Dict[str, Any]],
        remediation: List[Dict[str, Any]],
    ) -> str:
        """Generate executive summary for inspection package."""
        return (
            f"Audit Preparation Summary\n\n"
            f"Total Evidence Items: {len(evidence)}\n"
            f"Compliance Gaps Identified: {len(gaps)}\n"
            f"Critical Gaps: {len([g for g in gaps if g.get('severity') == GapSeverity.CRITICAL.value])}\n"
            f"Remediation Actions: {len(remediation)}\n"
            f"Actions Completed: {len([a for a in remediation if a.get('status') == 'completed'])}\n\n"
            f"Scope Period: {self.config.audit_scope_period_months} months\n"
            f"Record Retention: {self.config.evidence_retention_years} years\n"
        )

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
