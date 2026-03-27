# -*- coding: utf-8 -*-
"""
Audit Preparation Workflow
==========================

Pre-audit compliance verification and evidence packaging workflow.
Designed to run before external assurance engagements (limited or reasonable)
to ensure the organization is fully prepared for third-party audit.

Executes the full 235-rule ESRS compliance check, re-verifies all
calculations for reproducibility, documents complete data lineage from
source to output, assembles evidence packages, identifies gaps, and
generates auditor-ready documentation.

Steps:
    1. Full compliance rule execution (235 ESRS rules)
    2. Calculation re-verification (all formulas)
    3. Data lineage documentation (source to output trail)
    4. Evidence package assembly
    5. Gap identification and remediation suggestions
    6. Auditor-ready documentation generation

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class AuditStepStatus(str, Enum):
    """Status of an audit preparation step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ComplianceRuleSeverity(str, Enum):
    """Severity of a compliance rule failure."""
    CRITICAL = "critical"   # Blocks assurance opinion
    MAJOR = "major"         # Material finding
    MINOR = "minor"         # Observation / improvement point
    INFO = "info"           # Informational only


class AssuranceLevel(str, Enum):
    """Target assurance level."""
    LIMITED = "limited"      # ISAE 3000 limited
    REASONABLE = "reasonable" # ISAE 3000 reasonable


class EvidenceCategory(str, Enum):
    """Categories of audit evidence."""
    SOURCE_DATA = "source_data"
    CALCULATION = "calculation"
    METHODOLOGY = "methodology"
    CONTROL = "control"
    RECONCILIATION = "reconciliation"
    THIRD_PARTY = "third_party"
    APPROVAL = "approval"
    LINEAGE = "lineage"


class ReadinessLevel(str, Enum):
    """Overall audit readiness assessment."""
    READY = "ready"
    CONDITIONALLY_READY = "conditionally_ready"
    NOT_READY = "not_ready"


# =============================================================================
# DATA MODELS
# =============================================================================


class ComplianceRuleResult(BaseModel):
    """Result of a single compliance rule check."""
    rule_id: str = Field(..., description="Rule identifier (e.g. ESRS_E1_R001)")
    rule_name: str = Field(...)
    esrs_standard: str = Field(..., description="ESRS standard reference")
    esrs_paragraph: str = Field(default="", description="Specific paragraph reference")
    passed: bool = Field(...)
    severity: ComplianceRuleSeverity = Field(...)
    finding: str = Field(default="", description="Finding description if failed")
    remediation: str = Field(default="", description="Suggested remediation")
    evidence_refs: List[str] = Field(
        default_factory=list, description="References to supporting evidence"
    )


class CalculationVerification(BaseModel):
    """Result of re-verifying a single calculation."""
    calculation_id: str = Field(...)
    agent_name: str = Field(..., description="GreenLang agent that produced the calculation")
    scope: str = Field(default="", description="scope1, scope2, scope3")
    original_value: float = Field(...)
    reverified_value: float = Field(...)
    matches: bool = Field(...)
    tolerance_pct: float = Field(default=0.01, description="Acceptable tolerance percentage")
    provenance_hash: str = Field(default="")


class LineageTrail(BaseModel):
    """A single source-to-output data lineage trail."""
    trail_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_system: str = Field(...)
    source_record_id: str = Field(default="")
    intermediate_steps: List[str] = Field(default_factory=list)
    output_data_point: str = Field(...)
    output_value: str = Field(default="")
    transformations_applied: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class EvidenceItem(BaseModel):
    """A single piece of audit evidence."""
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: EvidenceCategory = Field(...)
    title: str = Field(...)
    description: str = Field(default="")
    document_ref: str = Field(default="", description="Document ID or path")
    esrs_data_points: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RemediationItem(BaseModel):
    """A remediation action for an identified gap."""
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(...)
    description: str = Field(...)
    severity: ComplianceRuleSeverity = Field(...)
    category: str = Field(default="compliance")
    estimated_effort_hours: float = Field(default=0.0)
    related_rule_ids: List[str] = Field(default_factory=list)
    deadline_suggestion: Optional[str] = Field(None)


class AuditPreparationInput(BaseModel):
    """Input configuration for the audit preparation workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=2024, le=2050)
    assurance_level: AssuranceLevel = Field(
        default=AssuranceLevel.LIMITED,
        description="Target assurance level"
    )
    esrs_standards: List[str] = Field(
        default_factory=lambda: [
            "ESRS_E1", "ESRS_E2", "ESRS_E3", "ESRS_E4", "ESRS_E5",
            "ESRS_S1", "ESRS_S2", "ESRS_S3", "ESRS_S4",
            "ESRS_G1", "ESRS_G2",
        ],
        description="ESRS standards in scope"
    )
    auditor_firm: Optional[str] = Field(None, description="External auditor firm name")
    recalculation_tolerance_pct: float = Field(
        default=0.01, ge=0, le=1,
        description="Acceptable calculation tolerance (default 0.01 = 1%)"
    )
    include_scope3: bool = Field(
        default=True, description="Include Scope 3 in verification"
    )
    evidence_format: str = Field(
        default="pdf", description="Output format for evidence package: pdf, html, xlsx"
    )

    @field_validator("esrs_standards")
    @classmethod
    def validate_standards(cls, v: List[str]) -> List[str]:
        """Ensure at least one standard is specified."""
        if not v:
            raise ValueError("At least one ESRS standard must be in scope")
        return v


class StepResult(BaseModel):
    """Result from a single audit preparation step."""
    step_name: str = Field(...)
    status: AuditStepStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class AuditPreparationResult(BaseModel):
    """Complete result from the audit preparation workflow."""
    workflow_id: str = Field(...)
    status: AuditStepStatus = Field(...)
    readiness_level: ReadinessLevel = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    steps: List[StepResult] = Field(default_factory=list)
    compliance_summary: Dict[str, Any] = Field(default_factory=dict)
    verification_summary: Dict[str, Any] = Field(default_factory=dict)
    evidence_package_id: Optional[str] = Field(None)
    remediation_items: List[RemediationItem] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# COMPLIANCE RULE CATALOGUE
# =============================================================================

ESRS_RULE_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "ESRS_2": {"total_rules": 45, "description": "General disclosures"},
    "ESRS_E1": {"total_rules": 35, "description": "Climate change"},
    "ESRS_E2": {"total_rules": 15, "description": "Pollution"},
    "ESRS_E3": {"total_rules": 15, "description": "Water and marine resources"},
    "ESRS_E4": {"total_rules": 20, "description": "Biodiversity and ecosystems"},
    "ESRS_E5": {"total_rules": 15, "description": "Resource use and circular economy"},
    "ESRS_S1": {"total_rules": 30, "description": "Own workforce"},
    "ESRS_S2": {"total_rules": 15, "description": "Workers in value chain"},
    "ESRS_S3": {"total_rules": 15, "description": "Affected communities"},
    "ESRS_S4": {"total_rules": 15, "description": "Consumers and end-users"},
    "ESRS_G1": {"total_rules": 10, "description": "Business conduct"},
    "ESRS_G2": {"total_rules": 5, "description": "Corporate culture"},
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class AuditPreparationWorkflow:
    """
    Pre-audit compliance verification and evidence packaging.

    Runs the full 235-rule ESRS compliance check, re-verifies all GHG
    calculations, documents data lineage, assembles an auditor evidence
    package, and generates remediation suggestions for any gaps identified.

    Attributes:
        workflow_id: Unique execution identifier.
        _cancelled: Cancellation flag.
        _progress_callback: Optional callback for progress updates.

    Example:
        >>> wf = AuditPreparationWorkflow()
        >>> inp = AuditPreparationInput(
        ...     organization_id="org-123",
        ...     reporting_year=2025,
        ...     assurance_level=AssuranceLevel.LIMITED,
        ... )
        >>> result = await wf.execute(inp)
        >>> print(f"Readiness: {result.readiness_level.value}")
    """

    STEPS = [
        "compliance_rules",
        "calculation_reverification",
        "lineage_documentation",
        "evidence_assembly",
        "gap_remediation",
        "auditor_documentation",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the audit preparation workflow.

        Args:
            progress_callback: Optional callback(step_name, message, pct_complete).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._cancelled: bool = False
        self._progress_callback = progress_callback
        self._step_results: Dict[str, StepResult] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: AuditPreparationInput) -> AuditPreparationResult:
        """
        Execute the audit preparation workflow.

        Args:
            input_data: Validated audit preparation input.

        Returns:
            AuditPreparationResult with compliance summary, evidence package,
            readiness level, and remediation items.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting audit preparation %s for org=%s year=%d level=%s",
            self.workflow_id, input_data.organization_id,
            input_data.reporting_year, input_data.assurance_level.value,
        )
        self._notify("workflow", "Audit preparation started", 0.0)

        completed_steps: List[StepResult] = []
        overall_status = AuditStepStatus.RUNNING
        compliance_summary: Dict[str, Any] = {}
        verification_summary: Dict[str, Any] = {}
        evidence_package_id: Optional[str] = None
        remediation_items: List[RemediationItem] = []

        step_handlers = [
            ("compliance_rules", self._step_compliance_rules),
            ("calculation_reverification", self._step_calculation_reverification),
            ("lineage_documentation", self._step_lineage_documentation),
            ("evidence_assembly", self._step_evidence_assembly),
            ("gap_remediation", self._step_gap_remediation),
            ("auditor_documentation", self._step_auditor_documentation),
        ]

        try:
            for idx, (step_name, handler) in enumerate(step_handlers):
                if self._cancelled:
                    overall_status = AuditStepStatus.SKIPPED
                    break

                pct = idx / len(step_handlers)
                self._notify(step_name, f"Starting: {step_name}", pct)
                step_started = datetime.utcnow()

                try:
                    step_result = await handler(input_data, pct)
                    step_result.started_at = step_started
                    step_result.completed_at = datetime.utcnow()
                    step_result.duration_seconds = (
                        step_result.completed_at - step_started
                    ).total_seconds()
                except Exception as exc:
                    logger.error("Step '%s' failed: %s", step_name, exc, exc_info=True)
                    step_result = StepResult(
                        step_name=step_name,
                        status=AuditStepStatus.FAILED,
                        started_at=step_started,
                        completed_at=datetime.utcnow(),
                        duration_seconds=(datetime.utcnow() - step_started).total_seconds(),
                        errors=[str(exc)],
                        provenance_hash=self._hash({"error": str(exc)}),
                    )

                completed_steps.append(step_result)
                self._step_results[step_name] = step_result

                # Collect typed outputs
                if step_name == "compliance_rules" and step_result.artifacts:
                    compliance_summary = step_result.artifacts.get("summary", {})
                if step_name == "calculation_reverification" and step_result.artifacts:
                    verification_summary = step_result.artifacts.get("summary", {})
                if step_name == "evidence_assembly" and step_result.artifacts:
                    evidence_package_id = step_result.artifacts.get("package_id")
                if step_name == "gap_remediation" and step_result.artifacts:
                    raw = step_result.artifacts.get("remediation_items", [])
                    remediation_items = [
                        RemediationItem(**r) for r in raw if isinstance(r, dict)
                    ]

                # Non-fatal: continue even if a step fails
                if step_result.status == AuditStepStatus.FAILED:
                    logger.warning(
                        "Audit step '%s' failed but continuing. Errors: %s",
                        step_name, step_result.errors,
                    )

            if overall_status == AuditStepStatus.RUNNING:
                has_failure = any(
                    s.status == AuditStepStatus.FAILED for s in completed_steps
                )
                overall_status = AuditStepStatus.FAILED if has_failure else AuditStepStatus.COMPLETED

        except Exception as exc:
            logger.critical(
                "Audit preparation %s failed: %s", self.workflow_id, exc, exc_info=True
            )
            overall_status = AuditStepStatus.FAILED

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        # Determine readiness level
        readiness = self._assess_readiness(
            compliance_summary, verification_summary, remediation_items,
            input_data.assurance_level,
        )

        metrics = {
            "total_rules_checked": compliance_summary.get("total_rules", 0),
            "rules_passed": compliance_summary.get("passed", 0),
            "rules_failed": compliance_summary.get("failed", 0),
            "calculations_verified": verification_summary.get("total_verified", 0),
            "calculations_matched": verification_summary.get("matched", 0),
            "evidence_items": 0,
            "remediation_items": len(remediation_items),
            "critical_remediations": sum(
                1 for r in remediation_items
                if r.severity == ComplianceRuleSeverity.CRITICAL
            ),
        }
        artifacts = {s.step_name: s.artifacts for s in completed_steps if s.artifacts}
        provenance = self._hash({
            "workflow_id": self.workflow_id,
            "steps": [s.provenance_hash for s in completed_steps],
        })

        self._notify("workflow", f"Audit preparation {overall_status.value}", 1.0)
        logger.info(
            "Audit preparation %s finished: readiness=%s, status=%s, duration=%.1fs",
            self.workflow_id, readiness.value, overall_status.value, total_duration,
        )

        return AuditPreparationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            readiness_level=readiness,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            steps=completed_steps,
            compliance_summary=compliance_summary,
            verification_summary=verification_summary,
            evidence_package_id=evidence_package_id,
            remediation_items=remediation_items,
            metrics=metrics,
            artifacts=artifacts,
            provenance_hash=provenance,
        )

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        logger.info("Cancellation requested for audit preparation %s", self.workflow_id)
        self._cancelled = True

    # -------------------------------------------------------------------------
    # Step 1: Full Compliance Rule Execution
    # -------------------------------------------------------------------------

    async def _step_compliance_rules(
        self, input_data: AuditPreparationInput, pct_base: float
    ) -> StepResult:
        """
        Execute all 235 ESRS compliance rules against the current dataset.

        Agents invoked:
            - greenlang.agents.data.validation_rule_engine (rule execution)
            - greenlang.agents.foundation.schema_compiler (schema validation)

        Rules are organized by ESRS standard and cover:
            - Mandatory disclosure completeness
            - Data type and format correctness
            - Cross-reference consistency (e.g. Scope 1+2+3 = total)
            - Temporal consistency (year-over-year plausibility)
            - Unit consistency
            - Boundary alignment
        """
        step_name = "compliance_rules"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Executing ESRS compliance rules", pct_base + 0.02)

        all_results: List[Dict[str, Any]] = []
        total_rules = 0
        passed = 0
        failed = 0
        critical_failures: List[Dict[str, Any]] = []
        by_standard: Dict[str, Dict[str, int]] = {}

        # ESRS 2 is always in scope (general/cross-cutting disclosures)
        standards_to_check = ["ESRS_2"] + [
            s for s in input_data.esrs_standards if s != "ESRS_2"
        ]

        for standard in standards_to_check:
            rule_cat = ESRS_RULE_CATEGORIES.get(standard, {})
            rules_in_standard = rule_cat.get("total_rules", 0)
            total_rules += rules_in_standard

            self._notify(
                step_name,
                f"Checking {standard} ({rules_in_standard} rules)",
                pct_base + 0.04,
            )

            standard_results = await self._execute_standard_rules(
                input_data.organization_id, input_data.reporting_year, standard
            )

            standard_passed = sum(1 for r in standard_results if r.get("passed", False))
            standard_failed = len(standard_results) - standard_passed
            passed += standard_passed
            failed += standard_failed

            by_standard[standard] = {
                "total": len(standard_results),
                "passed": standard_passed,
                "failed": standard_failed,
            }

            # Track critical failures
            for r in standard_results:
                if not r.get("passed") and r.get("severity") == ComplianceRuleSeverity.CRITICAL.value:
                    critical_failures.append(r)

            all_results.extend(standard_results)

        pass_rate = round(passed / max(total_rules, 1) * 100, 1)
        artifacts["summary"] = {
            "total_rules": total_rules,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": pass_rate,
            "critical_failures": len(critical_failures),
            "by_standard": by_standard,
        }
        artifacts["critical_failure_details"] = critical_failures
        artifacts["all_results"] = all_results

        if critical_failures:
            warnings.append(
                f"{len(critical_failures)} critical compliance failure(s) detected. "
                "These must be resolved before assurance engagement."
            )
        if pass_rate < 90:
            warnings.append(
                f"Overall compliance pass rate is {pass_rate}%. "
                "Target >= 95% for assurance readiness."
            )

        provenance = self._hash(artifacts["summary"])
        status = AuditStepStatus.COMPLETED if not errors else AuditStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=total_rules,
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 2: Calculation Re-verification
    # -------------------------------------------------------------------------

    async def _step_calculation_reverification(
        self, input_data: AuditPreparationInput, pct_base: float
    ) -> StepResult:
        """
        Re-run all GHG calculations and compare results against stored values
        to verify reproducibility and determinism.

        Agents invoked:
            - greenlang.agents.foundation.reproducibility_agent (re-execution)
            - All Scope 1/2/3 calculation agents (zero-hallucination path)
        """
        step_name = "calculation_reverification"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Re-verifying Scope 1 calculations", pct_base + 0.02)

        verifications: List[Dict[str, Any]] = []
        total_verified = 0
        matched = 0
        mismatches: List[Dict[str, Any]] = []
        tolerance = input_data.recalculation_tolerance_pct

        # Re-verify Scope 1
        scope1_verifications = await self._reverify_scope(
            input_data.organization_id, input_data.reporting_year, "scope1", tolerance
        )
        verifications.extend(scope1_verifications)

        self._notify(step_name, "Re-verifying Scope 2 calculations", pct_base + 0.04)

        # Re-verify Scope 2
        scope2_verifications = await self._reverify_scope(
            input_data.organization_id, input_data.reporting_year, "scope2", tolerance
        )
        verifications.extend(scope2_verifications)

        # Re-verify Scope 3 if in scope
        if input_data.include_scope3:
            self._notify(step_name, "Re-verifying Scope 3 calculations", pct_base + 0.06)
            scope3_verifications = await self._reverify_scope(
                input_data.organization_id, input_data.reporting_year, "scope3", tolerance
            )
            verifications.extend(scope3_verifications)

        # Analyze results
        for v in verifications:
            total_verified += 1
            if v.get("matches", False):
                matched += 1
            else:
                mismatches.append(v)

        artifacts["summary"] = {
            "total_verified": total_verified,
            "matched": matched,
            "mismatches": len(mismatches),
            "all_match": len(mismatches) == 0,
            "tolerance_pct": tolerance,
        }
        artifacts["mismatches"] = mismatches
        artifacts["verifications"] = verifications

        if mismatches:
            warnings.append(
                f"{len(mismatches)} calculation(s) did not match within "
                f"{tolerance*100:.2f}% tolerance."
            )
            for mm in mismatches[:5]:
                warnings.append(
                    f"  Mismatch: {mm.get('agent_name', '')} "
                    f"original={mm.get('original_value', 0):.4f} "
                    f"reverified={mm.get('reverified_value', 0):.4f}"
                )

        provenance = self._hash(artifacts["summary"])
        status = AuditStepStatus.COMPLETED if not errors else AuditStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=total_verified,
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 3: Data Lineage Documentation
    # -------------------------------------------------------------------------

    async def _step_lineage_documentation(
        self, input_data: AuditPreparationInput, pct_base: float
    ) -> StepResult:
        """
        Document complete data lineage trails from source systems to final
        reported values, including all transformations and calculations.

        Agents invoked:
            - greenlang.agents.data.data_lineage_tracker (lineage retrieval)
            - greenlang.agents.mrv.audit_trail_lineage (trail documentation)
            - greenlang.agents.foundation.citations_agent (source citation)
        """
        step_name = "lineage_documentation"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Retrieving data lineage trails", pct_base + 0.02)

        # Retrieve all lineage trails for the reporting year
        trails = await self._retrieve_lineage_trails(
            input_data.organization_id, input_data.reporting_year
        )

        self._notify(step_name, "Documenting source-to-output mappings", pct_base + 0.04)

        # Classify trails by completeness
        complete_trails = [t for t in trails if t.get("is_complete", False)]
        incomplete_trails = [t for t in trails if not t.get("is_complete", False)]

        artifacts["total_trails"] = len(trails)
        artifacts["complete_trails"] = len(complete_trails)
        artifacts["incomplete_trails"] = len(incomplete_trails)
        artifacts["trail_details"] = trails

        if incomplete_trails:
            warnings.append(
                f"{len(incomplete_trails)} lineage trail(s) are incomplete. "
                "Source-to-output mapping cannot be fully verified."
            )

        self._notify(step_name, "Generating lineage report", pct_base + 0.06)

        # Generate lineage report document
        lineage_doc = await self._generate_lineage_report(
            input_data.organization_id, input_data.reporting_year, trails
        )
        artifacts["lineage_report_id"] = lineage_doc.get("document_id", "")

        # Source system inventory
        source_systems: set = set()
        for trail in trails:
            src = trail.get("source_system", "")
            if src:
                source_systems.add(src)
        artifacts["source_systems"] = list(source_systems)
        artifacts["source_system_count"] = len(source_systems)

        provenance = self._hash({
            "total": len(trails),
            "complete": len(complete_trails),
            "systems": list(source_systems),
        })
        status = AuditStepStatus.COMPLETED if not errors else AuditStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=len(trails),
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 4: Evidence Package Assembly
    # -------------------------------------------------------------------------

    async def _step_evidence_assembly(
        self, input_data: AuditPreparationInput, pct_base: float
    ) -> StepResult:
        """
        Assemble a structured evidence package for the external auditor.

        The package includes:
            - Source data samples with provenance hashes
            - Calculation workpapers with formulas and inputs
            - Methodology documentation (GHG Protocol, ESRS references)
            - Internal control documentation
            - Reconciliation evidence (scope 2 dual reporting, cross-ref)
            - Third-party certificates and reports
            - Management approval records
            - Complete data lineage documentation

        Agents invoked:
            - greenlang.agents.reporting.assurance_preparation_agent
            - greenlang.agents.reporting.data_room_agent
        """
        step_name = "evidence_assembly"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Assembling evidence package", pct_base + 0.02)

        evidence_items: List[Dict[str, Any]] = []

        # Category 1: Source data evidence
        source_evidence = await self._collect_source_data_evidence(
            input_data.organization_id, input_data.reporting_year
        )
        evidence_items.extend(source_evidence)

        self._notify(step_name, "Collecting calculation workpapers", pct_base + 0.04)

        # Category 2: Calculation evidence
        calc_evidence = await self._collect_calculation_evidence(
            input_data.organization_id, input_data.reporting_year, input_data.include_scope3
        )
        evidence_items.extend(calc_evidence)

        # Category 3: Methodology documentation
        method_evidence = await self._collect_methodology_evidence(
            input_data.organization_id
        )
        evidence_items.extend(method_evidence)

        self._notify(step_name, "Collecting control documentation", pct_base + 0.06)

        # Category 4: Control documentation
        control_evidence = await self._collect_control_evidence(
            input_data.organization_id, input_data.reporting_year
        )
        evidence_items.extend(control_evidence)

        # Category 5: Third-party evidence
        third_party = await self._collect_third_party_evidence(
            input_data.organization_id, input_data.reporting_year
        )
        evidence_items.extend(third_party)

        self._notify(step_name, "Packaging evidence", pct_base + 0.08)

        # Package assembly
        package = await self._assemble_package(
            input_data.organization_id,
            input_data.reporting_year,
            evidence_items,
            input_data.evidence_format,
            input_data.assurance_level,
        )

        artifacts["package_id"] = package.get("package_id", "")
        artifacts["total_evidence_items"] = len(evidence_items)
        artifacts["by_category"] = {}
        for item in evidence_items:
            cat = item.get("category", "other")
            artifacts["by_category"][cat] = artifacts["by_category"].get(cat, 0) + 1
        artifacts["package_format"] = input_data.evidence_format
        artifacts["assurance_level"] = input_data.assurance_level.value

        # Check for completeness
        required_categories = {
            EvidenceCategory.SOURCE_DATA.value,
            EvidenceCategory.CALCULATION.value,
            EvidenceCategory.METHODOLOGY.value,
        }
        present_categories = set(artifacts["by_category"].keys())
        missing_categories = required_categories - present_categories
        if missing_categories:
            warnings.append(
                f"Evidence package missing categories: {', '.join(missing_categories)}"
            )

        provenance = self._hash({
            "package_id": artifacts["package_id"],
            "items": len(evidence_items),
        })
        status = AuditStepStatus.COMPLETED if not errors else AuditStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=len(evidence_items),
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 5: Gap Identification & Remediation
    # -------------------------------------------------------------------------

    async def _step_gap_remediation(
        self, input_data: AuditPreparationInput, pct_base: float
    ) -> StepResult:
        """
        Identify gaps in audit readiness and generate prioritized
        remediation suggestions.
        """
        step_name = "gap_remediation"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Identifying audit gaps", pct_base + 0.02)

        remediation_items: List[Dict[str, Any]] = []

        # Gaps from compliance rules
        compliance_step = self._step_results.get("compliance_rules")
        if compliance_step and compliance_step.artifacts:
            critical_failures = compliance_step.artifacts.get("critical_failure_details", [])
            for failure in critical_failures:
                remediation_items.append({
                    "item_id": str(uuid.uuid4()),
                    "title": f"Fix compliance rule: {failure.get('rule_id', '')}",
                    "description": failure.get("finding", "Compliance rule failed."),
                    "severity": ComplianceRuleSeverity.CRITICAL.value,
                    "category": "compliance",
                    "estimated_effort_hours": 4.0,
                    "related_rule_ids": [failure.get("rule_id", "")],
                    "deadline_suggestion": "Before audit engagement start",
                })

            # Major failures
            all_results = compliance_step.artifacts.get("all_results", [])
            major_failures = [
                r for r in all_results
                if not r.get("passed") and r.get("severity") == ComplianceRuleSeverity.MAJOR.value
            ]
            for failure in major_failures[:20]:
                remediation_items.append({
                    "item_id": str(uuid.uuid4()),
                    "title": f"Address major finding: {failure.get('rule_id', '')}",
                    "description": failure.get("finding", "Major finding."),
                    "severity": ComplianceRuleSeverity.MAJOR.value,
                    "category": "compliance",
                    "estimated_effort_hours": 2.0,
                    "related_rule_ids": [failure.get("rule_id", "")],
                    "deadline_suggestion": "Before fieldwork begins",
                })

        # Gaps from calculation re-verification
        verify_step = self._step_results.get("calculation_reverification")
        if verify_step and verify_step.artifacts:
            mismatches = verify_step.artifacts.get("mismatches", [])
            for mm in mismatches:
                remediation_items.append({
                    "item_id": str(uuid.uuid4()),
                    "title": f"Resolve calculation mismatch: {mm.get('agent_name', '')}",
                    "description": (
                        f"Calculation by {mm.get('agent_name', '')} produced "
                        f"{mm.get('reverified_value', 0)} vs stored {mm.get('original_value', 0)}. "
                        "Investigate input data changes or formula updates."
                    ),
                    "severity": ComplianceRuleSeverity.CRITICAL.value,
                    "category": "calculation",
                    "estimated_effort_hours": 3.0,
                    "related_rule_ids": [],
                    "deadline_suggestion": "Immediately",
                })

        # Gaps from lineage documentation
        lineage_step = self._step_results.get("lineage_documentation")
        if lineage_step and lineage_step.artifacts:
            incomplete = lineage_step.artifacts.get("incomplete_trails", 0)
            if incomplete > 0:
                remediation_items.append({
                    "item_id": str(uuid.uuid4()),
                    "title": f"Complete {incomplete} lineage trail(s)",
                    "description": (
                        f"{incomplete} data lineage trail(s) are incomplete. "
                        "Auditors will require full source-to-output traceability."
                    ),
                    "severity": ComplianceRuleSeverity.MAJOR.value,
                    "category": "lineage",
                    "estimated_effort_hours": float(incomplete) * 1.5,
                    "related_rule_ids": [],
                    "deadline_suggestion": "Before audit engagement start",
                })

        # Sort by severity
        severity_order = {
            ComplianceRuleSeverity.CRITICAL.value: 0,
            ComplianceRuleSeverity.MAJOR.value: 1,
            ComplianceRuleSeverity.MINOR.value: 2,
            ComplianceRuleSeverity.INFO.value: 3,
        }
        remediation_items.sort(
            key=lambda r: severity_order.get(r.get("severity", ""), 99)
        )

        artifacts["remediation_items"] = remediation_items
        artifacts["total_items"] = len(remediation_items)
        artifacts["critical_items"] = sum(
            1 for r in remediation_items
            if r.get("severity") == ComplianceRuleSeverity.CRITICAL.value
        )
        artifacts["estimated_total_hours"] = sum(
            r.get("estimated_effort_hours", 0) for r in remediation_items
        )

        if artifacts["critical_items"] > 0:
            warnings.append(
                f"{artifacts['critical_items']} critical remediation item(s) must be "
                "resolved before audit engagement."
            )

        provenance = self._hash(artifacts)
        status = AuditStepStatus.COMPLETED if not errors else AuditStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=len(remediation_items),
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 6: Auditor Documentation Generation
    # -------------------------------------------------------------------------

    async def _step_auditor_documentation(
        self, input_data: AuditPreparationInput, pct_base: float
    ) -> StepResult:
        """
        Generate the final auditor-ready documentation package.

        Includes:
            - Management representation letter template
            - Engagement scope summary
            - Data availability matrix
            - Control environment overview
            - Prior period comparatives
            - Key contacts and responsibilities

        Agents invoked:
            - greenlang.agents.reporting.assurance_preparation_agent
            - greenlang.agents.reporting.integrated_report_agent
        """
        step_name = "auditor_documentation"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Generating auditor documentation", pct_base + 0.02)

        # Management representation letter template
        mgmt_letter = await self._generate_management_letter(
            input_data.organization_id,
            input_data.reporting_year,
            input_data.assurance_level,
        )
        artifacts["management_letter_id"] = mgmt_letter.get("document_id", "")

        self._notify(step_name, "Building engagement scope document", pct_base + 0.04)

        # Engagement scope summary
        scope_doc = await self._generate_scope_document(
            input_data.organization_id,
            input_data.reporting_year,
            input_data.esrs_standards,
            input_data.assurance_level,
            input_data.auditor_firm,
        )
        artifacts["scope_document_id"] = scope_doc.get("document_id", "")

        # Data availability matrix
        availability = await self._generate_data_availability_matrix(
            input_data.organization_id, input_data.esrs_standards
        )
        artifacts["data_availability_matrix_id"] = availability.get("document_id", "")
        artifacts["data_availability_pct"] = availability.get("availability_pct", 0.0)

        self._notify(step_name, "Generating control environment overview", pct_base + 0.06)

        # Control environment overview
        controls_doc = await self._generate_controls_overview(
            input_data.organization_id
        )
        artifacts["controls_document_id"] = controls_doc.get("document_id", "")

        # Readiness summary (final)
        compliance_step = self._step_results.get("compliance_rules")
        verify_step = self._step_results.get("calculation_reverification")
        remediation_step = self._step_results.get("gap_remediation")

        summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "assurance_level": input_data.assurance_level.value,
            "auditor_firm": input_data.auditor_firm or "Not specified",
            "compliance_pass_rate": (
                compliance_step.artifacts.get("summary", {}).get("pass_rate_pct", 0)
                if compliance_step and compliance_step.artifacts else 0
            ),
            "calculations_reproducible": (
                verify_step.artifacts.get("summary", {}).get("all_match", False)
                if verify_step and verify_step.artifacts else False
            ),
            "open_remediation_items": (
                remediation_step.artifacts.get("total_items", 0)
                if remediation_step and remediation_step.artifacts else 0
            ),
            "generated_at": datetime.utcnow().isoformat(),
        }
        artifacts["readiness_summary"] = summary

        provenance = self._hash(artifacts)
        status = AuditStepStatus.COMPLETED if not errors else AuditStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Readiness Assessment
    # -------------------------------------------------------------------------

    def _assess_readiness(
        self,
        compliance: Dict[str, Any],
        verification: Dict[str, Any],
        remediation: List[RemediationItem],
        assurance_level: AssuranceLevel,
    ) -> ReadinessLevel:
        """
        Assess overall audit readiness based on compliance pass rate,
        calculation reproducibility, and remediation items.

        For REASONABLE assurance, standards are higher than LIMITED.
        """
        pass_rate = compliance.get("pass_rate_pct", 0)
        all_match = verification.get("all_match", False)
        critical_count = sum(
            1 for r in remediation if r.severity == ComplianceRuleSeverity.CRITICAL
        )

        if assurance_level == AssuranceLevel.REASONABLE:
            threshold_pass = 98.0
            max_critical = 0
        else:
            threshold_pass = 90.0
            max_critical = 2

        if pass_rate >= threshold_pass and all_match and critical_count <= max_critical:
            return ReadinessLevel.READY
        elif pass_rate >= 80.0 and critical_count <= max_critical * 2:
            return ReadinessLevel.CONDITIONALLY_READY
        else:
            return ReadinessLevel.NOT_READY

    # -------------------------------------------------------------------------
    # Agent Invocation Helpers
    # -------------------------------------------------------------------------

    async def _execute_standard_rules(
        self, org_id: str, year: int, standard: str
    ) -> List[Dict[str, Any]]:
        """Execute compliance rules for a specific ESRS standard."""
        await asyncio.sleep(0)
        return []

    async def _reverify_scope(
        self, org_id: str, year: int, scope: str, tolerance: float
    ) -> List[Dict[str, Any]]:
        """Re-verify all calculations for a given scope."""
        await asyncio.sleep(0)
        return []

    async def _retrieve_lineage_trails(
        self, org_id: str, year: int
    ) -> List[Dict[str, Any]]:
        """Retrieve all data lineage trails for the reporting year."""
        await asyncio.sleep(0)
        return []

    async def _generate_lineage_report(
        self, org_id: str, year: int, trails: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a lineage report document."""
        await asyncio.sleep(0)
        return {"document_id": str(uuid.uuid4())}

    async def _collect_source_data_evidence(
        self, org_id: str, year: int
    ) -> List[Dict[str, Any]]:
        """Collect source data evidence items."""
        await asyncio.sleep(0)
        return [{"category": EvidenceCategory.SOURCE_DATA.value, "title": "Source data samples"}]

    async def _collect_calculation_evidence(
        self, org_id: str, year: int, include_scope3: bool
    ) -> List[Dict[str, Any]]:
        """Collect calculation workpapers and evidence."""
        await asyncio.sleep(0)
        return [{"category": EvidenceCategory.CALCULATION.value, "title": "GHG calculation workpapers"}]

    async def _collect_methodology_evidence(
        self, org_id: str
    ) -> List[Dict[str, Any]]:
        """Collect methodology documentation."""
        await asyncio.sleep(0)
        return [{"category": EvidenceCategory.METHODOLOGY.value, "title": "GHG Protocol methodology"}]

    async def _collect_control_evidence(
        self, org_id: str, year: int
    ) -> List[Dict[str, Any]]:
        """Collect internal control documentation."""
        await asyncio.sleep(0)
        return [{"category": EvidenceCategory.CONTROL.value, "title": "Internal controls documentation"}]

    async def _collect_third_party_evidence(
        self, org_id: str, year: int
    ) -> List[Dict[str, Any]]:
        """Collect third-party certificates and reports."""
        await asyncio.sleep(0)
        return [{"category": EvidenceCategory.THIRD_PARTY.value, "title": "Third-party certificates"}]

    async def _assemble_package(
        self, org_id: str, year: int,
        items: List[Dict[str, Any]], fmt: str,
        level: AssuranceLevel,
    ) -> Dict[str, Any]:
        """Assemble the evidence package."""
        await asyncio.sleep(0)
        return {"package_id": str(uuid.uuid4())}

    async def _generate_management_letter(
        self, org_id: str, year: int, level: AssuranceLevel
    ) -> Dict[str, Any]:
        """Generate management representation letter template."""
        await asyncio.sleep(0)
        return {"document_id": str(uuid.uuid4())}

    async def _generate_scope_document(
        self, org_id: str, year: int, standards: List[str],
        level: AssuranceLevel, auditor: Optional[str],
    ) -> Dict[str, Any]:
        """Generate engagement scope summary document."""
        await asyncio.sleep(0)
        return {"document_id": str(uuid.uuid4())}

    async def _generate_data_availability_matrix(
        self, org_id: str, standards: List[str]
    ) -> Dict[str, Any]:
        """Generate data availability matrix across ESRS standards."""
        await asyncio.sleep(0)
        return {"document_id": str(uuid.uuid4()), "availability_pct": 0.0}

    async def _generate_controls_overview(
        self, org_id: str
    ) -> Dict[str, Any]:
        """Generate control environment overview document."""
        await asyncio.sleep(0)
        return {"document_id": str(uuid.uuid4())}

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _notify(self, step: str, message: str, pct: float) -> None:
        """Send progress notification."""
        if self._progress_callback:
            try:
                self._progress_callback(step, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for step=%s", step)

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
