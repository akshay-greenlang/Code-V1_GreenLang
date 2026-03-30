# -*- coding: utf-8 -*-
"""
Regulatory Filing Workflow
=============================

6-phase automated regulatory filing workflow for CSRD Enterprise Pack.
Manages the complete filing lifecycle from preparation through post-filing
archival, supporting ESAP, national registries, and EDGAR.

Phases:
    1. Filing Preparation: Assemble report artifacts, resolve filing metadata
    2. Pre-Submission Validation: Validate against registry-specific rules
    3. Internal Approval: Route through approval chain before submission
    4. Submission: Submit ESEF/iXBRL package to target registries
    5. Acknowledgment Tracking: Poll for registry acknowledgment/acceptance
    6. Post-Filing Archive: Archive with complete provenance chain

Author: GreenLang Team
Version: 3.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas.enums import ValidationSeverity

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

class FilingTarget(str, Enum):
    """Regulatory filing target registries."""

    ESAP = "ESAP"
    NATIONAL = "national_registries"
    EDGAR = "EDGAR"

class FilingFormat(str, Enum):
    """Filing output formats."""

    ESEF_IXBRL = "ESEF_iXBRL"
    XBRL = "XBRL"
    INLINE_XBRL = "inline_XBRL"
    PDF = "PDF"
    XHTML = "XHTML"

class SubmissionStatus(str, Enum):
    """Registry submission status."""

    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PROCESSING = "processing"
    ACKNOWLEDGED = "acknowledged"
    ERROR = "error"

class ApprovalStatus(str, Enum):
    """Internal approval status."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration in seconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")

class ValidationError(BaseModel):
    """A single validation error or warning."""

    code: str = Field(..., description="Error code")
    severity: ValidationSeverity = Field(..., description="Error severity")
    message: str = Field(..., description="Error description")
    location: str = Field(default="", description="Location in the filing package")
    rule_reference: str = Field(default="", description="Validation rule reference")

class FilingTargetConfig(BaseModel):
    """Configuration for a specific filing target."""

    target: FilingTarget = Field(..., description="Target registry")
    format: FilingFormat = Field(default=FilingFormat.ESEF_IXBRL)
    jurisdiction: str = Field(default="EU", description="Filing jurisdiction")
    registry_url: str = Field(default="", description="Registry submission endpoint")
    credentials_ref: str = Field(default="", description="Vault reference for credentials")
    taxonomy_version: str = Field(default="2023", description="XBRL taxonomy version")
    language: str = Field(default="en", description="Filing language")
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)

class FilingInput(BaseModel):
    """Input for the regulatory filing workflow."""

    report_id: str = Field(..., description="Report ID to file")
    organization_id: str = Field(..., description="Organization identifier")
    tenant_id: str = Field(default="", description="Tenant isolation ID")
    reporting_year: int = Field(..., ge=2024, le=2050, description="Reporting year")
    filing_targets: List[FilingTargetConfig] = Field(
        ..., min_length=1, description="Target registries for filing"
    )
    approvers: List[str] = Field(
        default_factory=lambda: ["cfo", "legal_counsel"],
        description="Internal approver roles",
    )
    filing_deadline: str = Field(..., description="Filing deadline (YYYY-MM-DD)")
    lei_code: str = Field(default="", description="Legal Entity Identifier")
    isin_code: str = Field(default="", description="ISIN code for listed entities")
    contact_email: str = Field(default="", description="Filing contact email")

    @field_validator("filing_deadline")
    @classmethod
    def validate_iso_date(cls, v: str) -> str:
        """Validate ISO date format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be YYYY-MM-DD format, got: {v}")
        return v

class SubmissionRecord(BaseModel):
    """Record of a single registry submission."""

    submission_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target: str = Field(..., description="Filing target")
    status: SubmissionStatus = Field(default=SubmissionStatus.SUBMITTED)
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = Field(None)
    reference_number: str = Field(default="", description="Registry reference number")
    errors: List[str] = Field(default_factory=list)
    package_hash: str = Field(default="", description="SHA-256 of submitted package")

class FilingResult(BaseModel):
    """Complete result from the regulatory filing workflow."""

    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(default="regulatory_filing")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    report_id: str = Field(default="", description="Source report ID")
    phases: List[PhaseResult] = Field(default_factory=list, description="Per-phase results")
    total_duration_seconds: float = Field(default=0.0)
    submissions: List[SubmissionRecord] = Field(
        default_factory=list, description="Per-target submission records"
    )
    all_accepted: bool = Field(default=False, description="All filings accepted")
    validation_errors_count: int = Field(default=0)
    archive_id: str = Field(default="", description="Post-filing archive ID")
    provenance_hash: str = Field(default="", description="SHA-256 of complete output")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class RegulatoryFilingWorkflow:
    """
    6-phase automated regulatory filing workflow.

    Manages the complete CSRD filing lifecycle from package preparation
    through post-filing archival. Supports ESAP (European Single Access
    Point), national registries, and EDGAR. Generates ESEF/iXBRL packages,
    runs pre-submission validation, routes through approval, submits,
    and tracks acknowledgments.

    Attributes:
        workflow_id: Unique execution identifier.
        config: Optional EnterprisePackConfig.
        _submissions: Submission records per target.
        _validation_errors: Validation errors found.

    Example:
        >>> workflow = RegulatoryFilingWorkflow()
        >>> filing = FilingInput(
        ...     report_id="rpt-001", organization_id="org-001",
        ...     reporting_year=2025, filing_deadline="2026-04-30",
        ...     filing_targets=[FilingTargetConfig(target=FilingTarget.ESAP)],
        ... )
        >>> result = await workflow.execute(filing)
        >>> assert result.all_accepted
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        Initialize the regulatory filing workflow.

        Args:
            config: Optional EnterprisePackConfig.
        """
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._submissions: List[SubmissionRecord] = []
        self._validation_errors: List[ValidationError] = []
        self._context: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, filing_input: FilingInput) -> FilingResult:
        """
        Execute the 6-phase regulatory filing workflow.

        Args:
            filing_input: Validated filing configuration.

        Returns:
            FilingResult with submission records, validation status,
            and archive reference.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting regulatory filing workflow %s for report=%s targets=%d",
            self.workflow_id, filing_input.report_id,
            len(filing_input.filing_targets),
        )

        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Filing Preparation
            p1 = await self._phase_1_filing_preparation(filing_input)
            phase_results.append(p1)
            if p1.status == PhaseStatus.FAILED:
                overall_status = WorkflowStatus.FAILED
                raise RuntimeError("Filing preparation failed")

            # Phase 2: Pre-Submission Validation
            p2 = await self._phase_2_pre_submission_validation(filing_input)
            phase_results.append(p2)
            blocking_errors = [
                e for e in self._validation_errors
                if e.severity == ValidationSeverity.ERROR
            ]
            if blocking_errors:
                overall_status = WorkflowStatus.FAILED
                raise RuntimeError(
                    f"Pre-submission validation failed with {len(blocking_errors)} errors"
                )

            # Phase 3: Internal Approval
            p3 = await self._phase_3_internal_approval(filing_input)
            phase_results.append(p3)
            if p3.status == PhaseStatus.FAILED:
                overall_status = WorkflowStatus.FAILED
                raise RuntimeError("Internal approval not obtained")

            # Phase 4: Submission
            p4 = await self._phase_4_submission(filing_input)
            phase_results.append(p4)

            # Phase 5: Acknowledgment Tracking
            p5 = await self._phase_5_acknowledgment_tracking(filing_input)
            phase_results.append(p5)

            # Phase 6: Post-Filing Archive
            p6 = await self._phase_6_post_filing_archive(filing_input)
            phase_results.append(p6)

            overall_status = WorkflowStatus.COMPLETED

        except RuntimeError:
            if overall_status != WorkflowStatus.FAILED:
                overall_status = WorkflowStatus.FAILED
        except Exception as exc:
            self.logger.critical(
                "Filing workflow %s failed: %s",
                self.workflow_id, str(exc), exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="workflow_error",
                status=PhaseStatus.FAILED,
                errors=[str(exc)],
                provenance_hash=self._hash_data({"error": str(exc)}),
            ))

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        all_accepted = all(
            s.status in (SubmissionStatus.ACCEPTED, SubmissionStatus.ACKNOWLEDGED)
            for s in self._submissions
        ) if self._submissions else False

        provenance = self._hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in phase_results],
            "submissions": [s.submission_id for s in self._submissions],
        })

        self.logger.info(
            "Filing workflow %s finished status=%s submissions=%d "
            "all_accepted=%s in %.1fs",
            self.workflow_id, overall_status.value,
            len(self._submissions), all_accepted, total_duration,
        )

        return FilingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            report_id=filing_input.report_id,
            phases=phase_results,
            total_duration_seconds=total_duration,
            submissions=self._submissions,
            all_accepted=all_accepted,
            validation_errors_count=len(self._validation_errors),
            archive_id=self._context.get("archive_id", ""),
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Filing Preparation
    # -------------------------------------------------------------------------

    async def _phase_1_filing_preparation(
        self, filing: FilingInput
    ) -> PhaseResult:
        """
        Assemble report artifacts and resolve filing metadata.

        Loads the report package, resolves entity metadata (LEI, ISIN),
        generates ESEF/iXBRL package per target requirements, and assembles
        all required filing attachments.

        Steps:
            1. Load report and associated artifacts
            2. Resolve entity metadata (LEI, ISIN, jurisdiction)
            3. Generate ESEF/iXBRL package per target format
            4. Assemble filing attachments
        """
        phase_name = "filing_preparation"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Load report
        report = await self._load_report(filing.report_id)
        outputs["report_loaded"] = report.get("loaded", False)
        outputs["report_sections"] = report.get("sections", 0)
        if not report.get("loaded", False):
            errors.append(f"Report not found: {filing.report_id}")
            return PhaseResult(
                phase_name=phase_name, status=PhaseStatus.FAILED,
                outputs=outputs, errors=errors,
                provenance_hash=self._hash_data(outputs),
            )

        # Step 2: Entity metadata
        metadata = await self._resolve_entity_metadata(filing)
        outputs["lei_code"] = metadata.get("lei", filing.lei_code)
        outputs["isin_code"] = metadata.get("isin", filing.isin_code)
        outputs["jurisdiction"] = metadata.get("jurisdiction", "EU")

        # Step 3: Generate packages per target
        packages: Dict[str, Dict[str, Any]] = {}
        for target_config in filing.filing_targets:
            package = await self._generate_filing_package(
                filing.report_id, target_config, metadata
            )
            packages[target_config.target.value] = package

        outputs["packages_generated"] = len(packages)
        outputs["package_ids"] = {
            t: p.get("package_id", "") for t, p in packages.items()
        }

        self._context["packages"] = packages
        self._context["metadata"] = metadata

        # Step 4: Attachments
        attachments = await self._assemble_attachments(filing, packages)
        outputs["attachments_assembled"] = attachments.get("count", 0)

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name, status=status, duration_seconds=duration,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Pre-Submission Validation
    # -------------------------------------------------------------------------

    async def _phase_2_pre_submission_validation(
        self, filing: FilingInput
    ) -> PhaseResult:
        """
        Validate filing packages against registry-specific rules.

        Runs structural validation (XML/XBRL well-formedness), taxonomic
        validation (correct concepts and dimensions), business rules
        validation, and cross-reference checks.

        Steps:
            1. Structural validation (XML, XBRL schema)
            2. Taxonomic validation (ESEF taxonomy compliance)
            3. Business rule validation (filing-specific rules)
            4. Cross-reference validation (internal consistency)
            5. Generate detailed validation report
        """
        phase_name = "pre_submission_validation"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        packages = self._context.get("packages", {})
        all_validation_errors: List[ValidationError] = []

        for target, package in packages.items():
            # Step 1: Structural validation
            structural = await self._validate_structure(target, package)
            all_validation_errors.extend(structural)

            # Step 2: Taxonomic validation
            taxonomic = await self._validate_taxonomy(target, package)
            all_validation_errors.extend(taxonomic)

            # Step 3: Business rules
            business = await self._validate_business_rules(target, package, filing)
            all_validation_errors.extend(business)

            # Step 4: Cross-reference
            cross_ref = await self._validate_cross_references(target, package)
            all_validation_errors.extend(cross_ref)

        self._validation_errors = all_validation_errors

        # Categorize errors
        error_count = sum(1 for e in all_validation_errors if e.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for e in all_validation_errors if e.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for e in all_validation_errors if e.severity == ValidationSeverity.INFO)

        outputs["total_validations_run"] = len(packages) * 4
        outputs["errors"] = error_count
        outputs["warnings"] = warning_count
        outputs["info"] = info_count
        outputs["validation_passed"] = error_count == 0
        outputs["error_details"] = [
            {"code": e.code, "message": e.message, "location": e.location}
            for e in all_validation_errors if e.severity == ValidationSeverity.ERROR
        ]

        if error_count > 0:
            errors.append(f"Pre-submission validation found {error_count} blocking errors")
        if warning_count > 0:
            warnings.append(f"Pre-submission validation found {warning_count} warnings")

        # Step 5: Validation report
        report = await self._generate_validation_report(all_validation_errors)
        outputs["validation_report_id"] = report.get("report_id", "")

        status = PhaseStatus.COMPLETED if error_count == 0 else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name, status=status, duration_seconds=duration,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Internal Approval
    # -------------------------------------------------------------------------

    async def _phase_3_internal_approval(
        self, filing: FilingInput
    ) -> PhaseResult:
        """
        Route filing through internal approval chain before submission.

        Submits the filing package to each configured approver (CFO, legal,
        compliance) and tracks approval status. All approvers must approve
        before submission proceeds.

        Steps:
            1. Submit to each approver in sequence
            2. Track approval status
            3. Handle rejections with feedback
        """
        phase_name = "internal_approval"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        approvals: Dict[str, Dict[str, Any]] = {}
        all_approved = True

        for approver_role in filing.approvers:
            result = await self._request_approval(approver_role, filing)
            approvals[approver_role] = result

            if result.get("status") == ApprovalStatus.REJECTED.value:
                all_approved = False
                errors.append(
                    f"Filing rejected by {approver_role}: {result.get('feedback', '')}"
                )
            elif result.get("status") == ApprovalStatus.PENDING.value:
                all_approved = False
                warnings.append(f"Approval pending from {approver_role}")

        outputs["approvals"] = approvals
        outputs["all_approved"] = all_approved
        outputs["approvers_count"] = len(filing.approvers)

        status = PhaseStatus.COMPLETED if all_approved else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name, status=status, duration_seconds=duration,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Submission
    # -------------------------------------------------------------------------

    async def _phase_4_submission(
        self, filing: FilingInput
    ) -> PhaseResult:
        """
        Submit ESEF/iXBRL packages to target registries.

        Submits the filing package to each configured registry (ESAP,
        national registries, EDGAR) and records the submission.

        Steps:
            1. For each filing target:
               a. Authenticate with registry
               b. Submit filing package
               c. Record submission ID and timestamp
        """
        phase_name = "submission"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        packages = self._context.get("packages", {})

        for target_config in filing.filing_targets:
            target = target_config.target.value
            package = packages.get(target, {})
            package_hash = self._hash_data(package)

            # Authenticate
            auth = await self._authenticate_registry(target_config)
            if not auth.get("authenticated", False):
                errors.append(f"Authentication failed for {target}")
                continue

            # Submit
            submission = await self._submit_to_registry(
                target_config, package, filing
            )

            record = SubmissionRecord(
                target=target,
                status=SubmissionStatus(submission.get("status", "submitted")),
                reference_number=submission.get("reference", ""),
                package_hash=package_hash,
                errors=submission.get("errors", []),
            )
            self._submissions.append(record)

        outputs["submissions_attempted"] = len(filing.filing_targets)
        outputs["submissions_successful"] = sum(
            1 for s in self._submissions
            if s.status != SubmissionStatus.ERROR
        )
        outputs["submission_ids"] = {
            s.target: s.submission_id for s in self._submissions
        }

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name, status=status, duration_seconds=duration,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Acknowledgment Tracking
    # -------------------------------------------------------------------------

    async def _phase_5_acknowledgment_tracking(
        self, filing: FilingInput
    ) -> PhaseResult:
        """
        Poll registries for filing acknowledgment and acceptance.

        Periodically checks each registry for submission status updates
        until all submissions are acknowledged or a timeout is reached.

        Steps:
            1. Poll each registry for status updates
            2. Update submission records with acknowledgments
            3. Handle rejections with error details
        """
        phase_name = "acknowledgment_tracking"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        max_polls = 5
        for poll_round in range(max_polls):
            all_resolved = True
            for submission in self._submissions:
                if submission.status in (
                    SubmissionStatus.ACCEPTED,
                    SubmissionStatus.ACKNOWLEDGED,
                    SubmissionStatus.REJECTED,
                    SubmissionStatus.ERROR,
                ):
                    continue

                status_update = await self._poll_registry_status(
                    submission.target, submission.submission_id
                )
                new_status = SubmissionStatus(status_update.get("status", "processing"))
                submission.status = new_status
                submission.reference_number = status_update.get(
                    "reference", submission.reference_number
                )

                if new_status == SubmissionStatus.REJECTED:
                    errors.append(
                        f"Filing rejected by {submission.target}: "
                        f"{status_update.get('reason', '')}"
                    )
                elif new_status not in (
                    SubmissionStatus.ACCEPTED, SubmissionStatus.ACKNOWLEDGED
                ):
                    all_resolved = False

            if all_resolved:
                break

        outputs["final_statuses"] = {
            s.target: s.status.value for s in self._submissions
        }
        outputs["all_acknowledged"] = all(
            s.status in (SubmissionStatus.ACCEPTED, SubmissionStatus.ACKNOWLEDGED)
            for s in self._submissions
        )
        outputs["reference_numbers"] = {
            s.target: s.reference_number for s in self._submissions
            if s.reference_number
        }

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name, status=status, duration_seconds=duration,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 6: Post-Filing Archive
    # -------------------------------------------------------------------------

    async def _phase_6_post_filing_archive(
        self, filing: FilingInput
    ) -> PhaseResult:
        """
        Archive filing with complete provenance chain.

        Creates an immutable archive containing the submitted packages,
        validation reports, approval records, submission receipts,
        and complete provenance chain for regulatory retention.

        Steps:
            1. Assemble archive contents
            2. Calculate provenance hash chain
            3. Store archive with retention policy
            4. Generate archive receipt
        """
        phase_name = "post_filing_archive"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Assemble contents
        archive_contents = {
            "report_id": filing.report_id,
            "organization_id": filing.organization_id,
            "reporting_year": filing.reporting_year,
            "filing_date": datetime.utcnow().isoformat(),
            "submissions": [s.model_dump() for s in self._submissions],
            "validation_errors": [e.model_dump() for e in self._validation_errors],
            "packages": list(self._context.get("packages", {}).keys()),
        }

        # Step 2: Provenance chain
        provenance_chain = self._build_provenance_chain(archive_contents)
        outputs["provenance_chain_length"] = len(provenance_chain)
        outputs["root_hash"] = provenance_chain[-1] if provenance_chain else ""

        # Step 3: Store archive
        archive = await self._store_archive(archive_contents, provenance_chain)
        archive_id = archive.get("archive_id", "")
        outputs["archive_id"] = archive_id
        outputs["retention_years"] = archive.get("retention_years", 10)
        outputs["storage_location"] = archive.get("location", "")
        self._context["archive_id"] = archive_id

        # Step 4: Archive receipt
        receipt = await self._generate_archive_receipt(archive_id, filing)
        outputs["receipt_id"] = receipt.get("receipt_id", "")
        outputs["receipt_hash"] = receipt.get("hash", "")

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name, status=status, duration_seconds=duration,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Agent Simulation Stubs
    # -------------------------------------------------------------------------

    async def _load_report(self, report_id: str) -> Dict[str, Any]:
        """Load report and associated artifacts."""
        return {"loaded": True, "sections": 12, "report_id": report_id}

    async def _resolve_entity_metadata(
        self, filing: FilingInput
    ) -> Dict[str, Any]:
        """Resolve entity metadata (LEI, ISIN, jurisdiction)."""
        return {
            "lei": filing.lei_code or "7890ABCDEF1234567890",
            "isin": filing.isin_code or "EU0009876543",
            "jurisdiction": "EU",
        }

    async def _generate_filing_package(
        self, report_id: str, target: FilingTargetConfig,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate ESEF/iXBRL package for a filing target."""
        return {
            "package_id": f"pkg-{uuid.uuid4().hex[:8]}",
            "format": target.format.value,
            "taxonomy": target.taxonomy_version,
            "tags_count": 350,
        }

    async def _assemble_attachments(
        self, filing: FilingInput, packages: Dict
    ) -> Dict[str, Any]:
        """Assemble filing attachments."""
        return {"count": 5}

    async def _validate_structure(
        self, target: str, package: Dict
    ) -> List[ValidationError]:
        """Validate structural well-formedness."""
        return []

    async def _validate_taxonomy(
        self, target: str, package: Dict
    ) -> List[ValidationError]:
        """Validate ESEF taxonomy compliance."""
        return []

    async def _validate_business_rules(
        self, target: str, package: Dict, filing: FilingInput
    ) -> List[ValidationError]:
        """Validate filing-specific business rules."""
        return []

    async def _validate_cross_references(
        self, target: str, package: Dict
    ) -> List[ValidationError]:
        """Validate internal cross-references."""
        return []

    async def _generate_validation_report(
        self, errors: List[ValidationError]
    ) -> Dict[str, Any]:
        """Generate detailed validation report."""
        return {"report_id": f"valrpt-{uuid.uuid4().hex[:8]}"}

    async def _request_approval(
        self, approver_role: str, filing: FilingInput
    ) -> Dict[str, Any]:
        """Request approval from an internal approver."""
        return {
            "status": ApprovalStatus.APPROVED.value,
            "approver": approver_role,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def _authenticate_registry(
        self, target: FilingTargetConfig
    ) -> Dict[str, Any]:
        """Authenticate with a filing registry."""
        return {"authenticated": True, "token_expires": 3600}

    async def _submit_to_registry(
        self, target: FilingTargetConfig, package: Dict,
        filing: FilingInput,
    ) -> Dict[str, Any]:
        """Submit filing package to a registry."""
        return {
            "status": "submitted",
            "reference": f"REF-{target.target.value}-{uuid.uuid4().hex[:8]}",
            "errors": [],
        }

    async def _poll_registry_status(
        self, target: str, submission_id: str
    ) -> Dict[str, Any]:
        """Poll registry for submission status."""
        return {"status": "acknowledged", "reference": f"ACK-{target}-{submission_id[:8]}"}

    def _build_provenance_chain(
        self, contents: Dict[str, Any]
    ) -> List[str]:
        """Build a provenance hash chain from archive contents."""
        chain = []
        current = self._hash_data(contents)
        chain.append(current)
        for submission in self._submissions:
            current = self._hash_data({"prev": current, "sub": submission.submission_id})
            chain.append(current)
        return chain

    async def _store_archive(
        self, contents: Dict, chain: List[str]
    ) -> Dict[str, Any]:
        """Store the filing archive with retention policy."""
        return {
            "archive_id": f"arc-{uuid.uuid4().hex[:8]}",
            "retention_years": 10,
            "location": "s3://gl-enterprise-archives/filings/",
        }

    async def _generate_archive_receipt(
        self, archive_id: str, filing: FilingInput
    ) -> Dict[str, Any]:
        """Generate archive receipt with integrity hash."""
        return {
            "receipt_id": f"rcpt-{uuid.uuid4().hex[:8]}",
            "hash": self._hash_data({"archive": archive_id}),
        }

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _hash_data(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
