# -*- coding: utf-8 -*-
"""
Regulatory Submission Workflow
====================================

4-phase workflow for assembling and submitting regulatory compliance
packages per EU Battery Regulation 2023/1542. Implements documentation
assembly, conformity assessment, submission package generation, and
registry upload preparation.

Phases:
    1. DocumentationAssembly  -- Gather and assemble all required documents
    2. ConformityCheck        -- Verify EU declaration of conformity
    3. SubmissionPackage      -- Generate structured submission package
    4. RegistryUpload         -- Prepare and record registry upload

Regulatory references:
    - EU Regulation 2023/1542 Art. 17 (EU declaration of conformity)
    - EU Regulation 2023/1542 Art. 18 (CE marking)
    - EU Regulation 2023/1542 Art. 77 (battery passport)
    - EU Regulation 2023/1542 Annex IX (EU declaration of conformity content)
    - EU Decision 768/2008/EC (conformity assessment framework)

Author: GreenLang Team
Version: 1.0.0
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

_MODULE_VERSION = "1.0.0"


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
    """Phases of the regulatory submission workflow."""
    DOCUMENTATION_ASSEMBLY = "documentation_assembly"
    CONFORMITY_CHECK = "conformity_check"
    SUBMISSION_PACKAGE = "submission_package"
    REGISTRY_UPLOAD = "registry_upload"


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


class DocumentType(str, Enum):
    """Required regulatory document types."""
    EU_DECLARATION_OF_CONFORMITY = "eu_declaration_of_conformity"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    CARBON_FOOTPRINT_DECLARATION = "carbon_footprint_declaration"
    RECYCLED_CONTENT_DOCUMENTATION = "recycled_content_documentation"
    PERFORMANCE_TEST_REPORT = "performance_test_report"
    DUE_DILIGENCE_REPORT = "due_diligence_report"
    BATTERY_PASSPORT_DATA = "battery_passport_data"
    LABELLING_VERIFICATION = "labelling_verification"
    END_OF_LIFE_PLAN = "end_of_life_plan"
    QUALITY_MANAGEMENT_CERTIFICATE = "quality_management_certificate"
    NOTIFIED_BODY_CERTIFICATE = "notified_body_certificate"
    RISK_ASSESSMENT = "risk_assessment"


class ConformityModule(str, Enum):
    """Conformity assessment modules per Decision 768/2008/EC."""
    MODULE_A = "module_a"
    MODULE_B = "module_b"
    MODULE_C = "module_c"
    MODULE_D = "module_d"
    MODULE_E = "module_e"
    MODULE_F = "module_f"


class SubmissionStatus(str, Enum):
    """Submission package status."""
    DRAFT = "draft"
    ASSEMBLED = "assembled"
    VALIDATED = "validated"
    READY = "ready"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVISION_REQUIRED = "revision_required"


class RegistryType(str, Enum):
    """Regulatory registry types."""
    BATTERY_PASSPORT_REGISTRY = "battery_passport_registry"
    PRODUCER_REGISTER = "producer_register"
    NOTIFIED_BODY_DATABASE = "notified_body_database"
    MARKET_SURVEILLANCE = "market_surveillance"


# =============================================================================
# REQUIRED DOCUMENTS BY CATEGORY
# =============================================================================


REQUIRED_DOCUMENTS: Dict[str, List[str]] = {
    "ev_battery": [
        DocumentType.EU_DECLARATION_OF_CONFORMITY.value,
        DocumentType.TECHNICAL_DOCUMENTATION.value,
        DocumentType.CARBON_FOOTPRINT_DECLARATION.value,
        DocumentType.RECYCLED_CONTENT_DOCUMENTATION.value,
        DocumentType.PERFORMANCE_TEST_REPORT.value,
        DocumentType.DUE_DILIGENCE_REPORT.value,
        DocumentType.BATTERY_PASSPORT_DATA.value,
        DocumentType.LABELLING_VERIFICATION.value,
        DocumentType.END_OF_LIFE_PLAN.value,
        DocumentType.QUALITY_MANAGEMENT_CERTIFICATE.value,
    ],
    "industrial_battery": [
        DocumentType.EU_DECLARATION_OF_CONFORMITY.value,
        DocumentType.TECHNICAL_DOCUMENTATION.value,
        DocumentType.CARBON_FOOTPRINT_DECLARATION.value,
        DocumentType.RECYCLED_CONTENT_DOCUMENTATION.value,
        DocumentType.PERFORMANCE_TEST_REPORT.value,
        DocumentType.DUE_DILIGENCE_REPORT.value,
        DocumentType.BATTERY_PASSPORT_DATA.value,
        DocumentType.LABELLING_VERIFICATION.value,
        DocumentType.END_OF_LIFE_PLAN.value,
    ],
    "lmt_battery": [
        DocumentType.EU_DECLARATION_OF_CONFORMITY.value,
        DocumentType.TECHNICAL_DOCUMENTATION.value,
        DocumentType.CARBON_FOOTPRINT_DECLARATION.value,
        DocumentType.PERFORMANCE_TEST_REPORT.value,
        DocumentType.LABELLING_VERIFICATION.value,
        DocumentType.END_OF_LIFE_PLAN.value,
    ],
    "portable_battery": [
        DocumentType.EU_DECLARATION_OF_CONFORMITY.value,
        DocumentType.TECHNICAL_DOCUMENTATION.value,
        DocumentType.LABELLING_VERIFICATION.value,
        DocumentType.END_OF_LIFE_PLAN.value,
    ],
    "sli_battery": [
        DocumentType.EU_DECLARATION_OF_CONFORMITY.value,
        DocumentType.TECHNICAL_DOCUMENTATION.value,
        DocumentType.LABELLING_VERIFICATION.value,
        DocumentType.END_OF_LIFE_PLAN.value,
    ],
}

# EU Declaration of Conformity required fields (Annex IX)
DOC_REQUIRED_FIELDS: Dict[str, List[str]] = {
    DocumentType.EU_DECLARATION_OF_CONFORMITY.value: [
        "declaration_number",
        "manufacturer_name",
        "manufacturer_address",
        "battery_model",
        "battery_type",
        "applicable_legislation",
        "harmonised_standards",
        "notified_body_details",
        "signatory_name",
        "signatory_position",
        "signature_date",
        "signature_place",
    ],
    DocumentType.TECHNICAL_DOCUMENTATION.value: [
        "general_description",
        "design_drawings",
        "test_reports",
        "standards_applied",
    ],
    DocumentType.CARBON_FOOTPRINT_DECLARATION.value: [
        "total_carbon_footprint_kgco2e",
        "carbon_footprint_per_kwh",
        "performance_class",
        "lifecycle_stages",
    ],
    DocumentType.BATTERY_PASSPORT_DATA.value: [
        "passport_id",
        "unique_identifier",
        "qr_code_reference",
    ],
}


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


class DocumentRecord(BaseModel):
    """A regulatory document record."""
    document_id: str = Field(default_factory=lambda: f"doc-{_new_uuid()[:8]}")
    document_type: str = Field(..., description="Document type classification")
    title: str = Field(default="", description="Document title")
    version: str = Field(default="1.0", description="Document version")
    issued_date: str = Field(default="", description="ISO date of issuance")
    issuer: str = Field(default="", description="Document issuer")
    reference_number: str = Field(default="", description="External reference number")
    fields: Dict[str, Any] = Field(
        default_factory=dict, description="Document data fields"
    )
    file_hash: str = Field(default="", description="SHA-256 of document file")
    status: str = Field(default="draft")


class ConformityCheckResult(BaseModel):
    """Result of a conformity assessment check."""
    check_id: str = Field(default_factory=lambda: f"cc-{_new_uuid()[:8]}")
    document_type: str = Field(default="")
    is_present: bool = Field(default=False)
    is_valid: bool = Field(default=False)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    missing_fields: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)


class SubmissionPackageInfo(BaseModel):
    """Regulatory submission package metadata."""
    package_id: str = Field(default_factory=lambda: f"pkg-{_new_uuid()[:8]}")
    battery_id: str = Field(default="")
    battery_category: str = Field(default="")
    documents_included: int = Field(default=0, ge=0)
    total_documents_required: int = Field(default=0, ge=0)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    package_hash: str = Field(default="")
    status: SubmissionStatus = Field(default=SubmissionStatus.DRAFT)
    created_at: str = Field(default="")


class RegistryUploadRecord(BaseModel):
    """Registry upload tracking record."""
    upload_id: str = Field(default_factory=lambda: f"upl-{_new_uuid()[:8]}")
    registry_type: RegistryType = Field(
        default=RegistryType.BATTERY_PASSPORT_REGISTRY
    )
    registry_name: str = Field(default="")
    package_id: str = Field(default="")
    upload_status: str = Field(default="prepared")
    uploaded_at: str = Field(default="")
    confirmation_id: str = Field(default="")
    response_code: str = Field(default="")


class RegulatorySubmissionInput(BaseModel):
    """Input data model for RegulatorySubmissionWorkflow."""
    battery_id: str = Field(default_factory=lambda: f"bat-{_new_uuid()[:8]}")
    battery_model: str = Field(default="", description="Battery model identifier")
    battery_category: str = Field(default="ev_battery")
    manufacturer_name: str = Field(default="")
    manufacturer_address: str = Field(default="")
    documents: List[DocumentRecord] = Field(default_factory=list)
    conformity_module: ConformityModule = Field(
        default=ConformityModule.MODULE_A,
        description="Conformity assessment module"
    )
    target_registries: List[RegistryType] = Field(
        default_factory=lambda: [RegistryType.BATTERY_PASSPORT_REGISTRY]
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class RegulatorySubmissionResult(BaseModel):
    """Complete result from regulatory submission workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="regulatory_submission")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    battery_id: str = Field(default="")
    submission_package: Optional[SubmissionPackageInfo] = Field(default=None)
    conformity_checks: List[ConformityCheckResult] = Field(default_factory=list)
    registry_uploads: List[RegistryUploadRecord] = Field(default_factory=list)
    documents_submitted: int = Field(default=0, ge=0)
    documents_required: int = Field(default=0, ge=0)
    overall_completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    submission_ready: bool = Field(default=False)
    reporting_year: int = Field(default=2025)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RegulatorySubmissionWorkflow:
    """
    4-phase regulatory submission workflow per EU Battery Regulation.

    Implements end-to-end regulatory submission preparation following
    EU Regulation 2023/1542 Art. 17-18 and Art. 77. Assembles required
    documentation, performs conformity checks, generates a submission
    package with integrity hash, and prepares registry uploads.

    Zero-hallucination: all completeness and conformity checks use
    deterministic field-presence verification. No LLM in validation paths.

    Example:
        >>> wf = RegulatorySubmissionWorkflow()
        >>> inp = RegulatorySubmissionInput(documents=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.submission_ready in (True, False)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize RegulatorySubmissionWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._documents: List[DocumentRecord] = []
        self._conformity_checks: List[ConformityCheckResult] = []
        self._package: Optional[SubmissionPackageInfo] = None
        self._uploads: List[RegistryUploadRecord] = []
        self._submission_ready: bool = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.DOCUMENTATION_ASSEMBLY.value, "description": "Gather and assemble all required documents"},
            {"name": WorkflowPhase.CONFORMITY_CHECK.value, "description": "Verify EU declaration of conformity"},
            {"name": WorkflowPhase.SUBMISSION_PACKAGE.value, "description": "Generate structured submission package"},
            {"name": WorkflowPhase.REGISTRY_UPLOAD.value, "description": "Prepare and record registry upload"},
        ]

    def validate_inputs(self, input_data: RegulatorySubmissionInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.battery_id:
            issues.append("Battery ID is required")
        if not input_data.manufacturer_name:
            issues.append("Manufacturer name is required")
        if not input_data.documents:
            issues.append("No documents provided for submission")
        return issues

    async def execute(
        self,
        input_data: Optional[RegulatorySubmissionInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> RegulatorySubmissionResult:
        """
        Execute the 4-phase regulatory submission workflow.

        Args:
            input_data: Full input model (preferred).
            config: Configuration overrides.

        Returns:
            RegulatorySubmissionResult with conformity checks, package, and
            upload status.
        """
        if input_data is None:
            input_data = RegulatorySubmissionInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting regulatory submission workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_documentation_assembly(input_data))
            phases_done += 1
            phase_results.append(await self._phase_conformity_check(input_data))
            phases_done += 1
            phase_results.append(await self._phase_submission_package(input_data))
            phases_done += 1
            phase_results.append(await self._phase_registry_upload(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error(
                "Regulatory submission workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        required_docs = REQUIRED_DOCUMENTS.get(input_data.battery_category, [])
        completeness = round(
            (self._package.completeness_pct if self._package else 0.0), 1
        )

        result = RegulatorySubmissionResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            battery_id=input_data.battery_id,
            submission_package=self._package,
            conformity_checks=self._conformity_checks,
            registry_uploads=self._uploads,
            documents_submitted=len(self._documents),
            documents_required=len(required_docs),
            overall_completeness_pct=completeness,
            submission_ready=self._submission_ready,
            reporting_year=input_data.reporting_year,
            executed_at=_utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Regulatory submission %s completed in %.2fs: ready=%s, completeness=%.1f%%",
            self.workflow_id, elapsed, self._submission_ready, completeness,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Documentation Assembly
    # -------------------------------------------------------------------------

    async def _phase_documentation_assembly(
        self, input_data: RegulatorySubmissionInput,
    ) -> PhaseResult:
        """Gather and assemble all required regulatory documents."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._documents = list(input_data.documents)

        required = REQUIRED_DOCUMENTS.get(input_data.battery_category, [])
        submitted_types = set(d.document_type for d in self._documents)
        required_set = set(required)

        present = submitted_types & required_set
        missing = required_set - submitted_types
        extra = submitted_types - required_set

        # Compute file hashes for documents without one
        for doc in self._documents:
            if not doc.file_hash and doc.fields:
                doc.file_hash = _compute_hash(
                    json.dumps(doc.fields, sort_keys=True, default=str)
                )

        type_counts: Dict[str, int] = {}
        for doc in self._documents:
            type_counts[doc.document_type] = type_counts.get(doc.document_type, 0) + 1

        outputs["documents_submitted"] = len(self._documents)
        outputs["documents_required"] = len(required)
        outputs["documents_present"] = len(present)
        outputs["documents_missing"] = sorted(missing)
        outputs["documents_extra"] = sorted(extra)
        outputs["type_distribution"] = type_counts
        outputs["battery_category"] = input_data.battery_category

        if missing:
            warnings.append(
                f"Missing required documents: {', '.join(sorted(missing))}"
            )

        if not self._documents:
            warnings.append("No documents provided; submission package will be empty")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 DocumentationAssembly: %d submitted, %d required, %d missing",
            len(self._documents), len(required), len(missing),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.DOCUMENTATION_ASSEMBLY.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Conformity Check
    # -------------------------------------------------------------------------

    async def _phase_conformity_check(
        self, input_data: RegulatorySubmissionInput,
    ) -> PhaseResult:
        """Verify conformity of each document against requirements."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._conformity_checks = []

        required = REQUIRED_DOCUMENTS.get(input_data.battery_category, [])
        doc_map: Dict[str, DocumentRecord] = {
            d.document_type: d for d in self._documents
        }

        for req_type in required:
            doc = doc_map.get(req_type)
            check = self._check_document_conformity(req_type, doc)
            self._conformity_checks.append(check)

        valid_count = sum(1 for c in self._conformity_checks if c.is_valid)
        invalid_count = sum(1 for c in self._conformity_checks if not c.is_valid)
        total_missing_fields = sum(
            len(c.missing_fields) for c in self._conformity_checks
        )

        all_valid = invalid_count == 0

        outputs["conformity_checks_performed"] = len(self._conformity_checks)
        outputs["valid_documents"] = valid_count
        outputs["invalid_documents"] = invalid_count
        outputs["total_missing_fields"] = total_missing_fields
        outputs["all_conformant"] = all_valid
        outputs["conformity_module"] = input_data.conformity_module.value

        if invalid_count > 0:
            invalid_types = [
                c.document_type for c in self._conformity_checks if not c.is_valid
            ]
            warnings.append(
                f"{invalid_count} documents failed conformity check: "
                f"{', '.join(invalid_types)}"
            )

        # EU DoC specific check
        doc_check = next(
            (c for c in self._conformity_checks
             if c.document_type == DocumentType.EU_DECLARATION_OF_CONFORMITY.value),
            None
        )
        if doc_check and not doc_check.is_valid:
            warnings.append(
                "EU Declaration of Conformity is incomplete; "
                f"missing: {', '.join(doc_check.missing_fields)}"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 ConformityCheck: %d valid, %d invalid",
            valid_count, invalid_count,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.CONFORMITY_CHECK.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _check_document_conformity(
        self, doc_type: str, doc: Optional[DocumentRecord],
    ) -> ConformityCheckResult:
        """Check conformity of a single document."""
        if doc is None:
            return ConformityCheckResult(
                document_type=doc_type,
                is_present=False,
                is_valid=False,
                completeness_pct=0.0,
                missing_fields=[doc_type],
                issues=[f"{doc_type}: document not provided"],
            )

        required_fields = DOC_REQUIRED_FIELDS.get(doc_type, [])
        if not required_fields:
            # No specific field requirements defined; mark as valid if present
            return ConformityCheckResult(
                document_type=doc_type,
                is_present=True,
                is_valid=True,
                completeness_pct=100.0,
            )

        present_count = 0
        missing: List[str] = []
        issues: List[str] = []

        for field_name in required_fields:
            value = doc.fields.get(field_name)
            if value is not None and value != "" and value != 0:
                present_count += 1
            else:
                missing.append(field_name)
                issues.append(f"{doc_type}.{field_name}: missing or empty")

        completeness = round(
            (present_count / len(required_fields) * 100)
            if required_fields else 100.0, 1
        )
        is_valid = len(missing) == 0

        return ConformityCheckResult(
            document_type=doc_type,
            is_present=True,
            is_valid=is_valid,
            completeness_pct=completeness,
            missing_fields=missing,
            issues=issues,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Submission Package
    # -------------------------------------------------------------------------

    async def _phase_submission_package(
        self, input_data: RegulatorySubmissionInput,
    ) -> PhaseResult:
        """Generate structured submission package."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        required = REQUIRED_DOCUMENTS.get(input_data.battery_category, [])
        submitted_types = set(d.document_type for d in self._documents)
        docs_included = len(submitted_types & set(required))

        completeness = round(
            (docs_included / len(required) * 100) if required else 0.0, 1
        )

        # Build package manifest
        manifest = {
            "battery_id": input_data.battery_id,
            "battery_model": input_data.battery_model,
            "battery_category": input_data.battery_category,
            "manufacturer": input_data.manufacturer_name,
            "conformity_module": input_data.conformity_module.value,
            "documents": [
                {
                    "type": d.document_type,
                    "title": d.title,
                    "version": d.version,
                    "reference": d.reference_number,
                    "file_hash": d.file_hash,
                }
                for d in self._documents
            ],
            "reporting_year": input_data.reporting_year,
            "created_at": _utcnow().isoformat(),
        }

        manifest_json = json.dumps(manifest, sort_keys=True, default=str)
        package_hash = _compute_hash(manifest_json)

        all_checks_valid = all(c.is_valid for c in self._conformity_checks)
        can_submit = completeness >= 100.0 and all_checks_valid

        pkg_status = (
            SubmissionStatus.READY if can_submit
            else SubmissionStatus.ASSEMBLED if completeness >= 80.0
            else SubmissionStatus.DRAFT
        )

        self._package = SubmissionPackageInfo(
            battery_id=input_data.battery_id,
            battery_category=input_data.battery_category,
            documents_included=docs_included,
            total_documents_required=len(required),
            completeness_pct=completeness,
            package_hash=package_hash,
            status=pkg_status,
            created_at=_utcnow().isoformat(),
        )

        self._submission_ready = can_submit

        outputs["package_id"] = self._package.package_id
        outputs["completeness_pct"] = completeness
        outputs["package_status"] = pkg_status.value
        outputs["package_hash"] = package_hash
        outputs["documents_included"] = docs_included
        outputs["documents_required"] = len(required)
        outputs["submission_ready"] = can_submit

        if not can_submit:
            if completeness < 100.0:
                warnings.append(
                    f"Package incomplete ({completeness}%); "
                    f"cannot submit to registry"
                )
            if not all_checks_valid:
                warnings.append(
                    "Some documents failed conformity checks; "
                    "resolve issues before submission"
                )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 SubmissionPackage: %s, completeness %.1f%%, ready=%s",
            self._package.package_id, completeness, can_submit,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.SUBMISSION_PACKAGE.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Registry Upload
    # -------------------------------------------------------------------------

    async def _phase_registry_upload(
        self, input_data: RegulatorySubmissionInput,
    ) -> PhaseResult:
        """Prepare and record registry upload for each target registry."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._uploads = []

        package_id = self._package.package_id if self._package else ""

        for registry_type in input_data.target_registries:
            can_upload = self._submission_ready

            registry_names: Dict[str, str] = {
                RegistryType.BATTERY_PASSPORT_REGISTRY.value: "EU Battery Passport Registry (ESPR)",
                RegistryType.PRODUCER_REGISTER.value: "National Producer Register",
                RegistryType.NOTIFIED_BODY_DATABASE.value: "NANDO (Notified Bodies Database)",
                RegistryType.MARKET_SURVEILLANCE.value: "ICSMS Market Surveillance",
            }

            upload = RegistryUploadRecord(
                registry_type=registry_type,
                registry_name=registry_names.get(
                    registry_type.value, "Unknown Registry"
                ),
                package_id=package_id,
                upload_status="submitted" if can_upload else "blocked",
                uploaded_at=_utcnow().isoformat() if can_upload else "",
                confirmation_id=f"CONF-{_new_uuid()[:8]}" if can_upload else "",
                response_code="200_ACCEPTED" if can_upload else "400_INCOMPLETE",
            )
            self._uploads.append(upload)

            if not can_upload:
                warnings.append(
                    f"Upload to {upload.registry_name} blocked: "
                    f"submission package not ready"
                )

        upload_summary: Dict[str, str] = {}
        for u in self._uploads:
            upload_summary[u.registry_type.value] = u.upload_status

        outputs["uploads_attempted"] = len(self._uploads)
        outputs["uploads_submitted"] = sum(
            1 for u in self._uploads if u.upload_status == "submitted"
        )
        outputs["uploads_blocked"] = sum(
            1 for u in self._uploads if u.upload_status == "blocked"
        )
        outputs["upload_summary"] = upload_summary
        outputs["package_id"] = package_id

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 RegistryUpload: %d attempted, %d submitted",
            len(self._uploads),
            outputs["uploads_submitted"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.REGISTRY_UPLOAD.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: RegulatorySubmissionResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
