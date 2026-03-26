# -*- coding: utf-8 -*-
"""
DocumentationEngine - PACK-044 Inventory Management Engine 9
==============================================================

Methodology documentation and evidence management engine that tracks
all supporting documentation for a GHG inventory including methodology
descriptions, assumptions, emission factor sources, data quality
assessments, and audit evidence. Produces completeness assessments to
ensure the inventory meets external assurance requirements.

Documentation Completeness Methodology:
    Per-Category Completeness:
        cat_score = sum(doc_present_i * weight_i) for i in REQUIRED_DOCS
        cat_completeness = cat_score / sum(weight_i) * 100

    Overall Completeness:
        overall = sum(cat_completeness_j * emission_weight_j)
            where emission_weight_j = cat_emissions_j / total_emissions

    Required Documentation (GHG Protocol Chapter 8 / ISO 14064-3):
        - Methodology description (weight: 20)
        - Emission factor source and version (weight: 15)
        - Activity data source and collection method (weight: 15)
        - Assumptions and justifications (weight: 15)
        - Calculation procedures (weight: 10)
        - QA/QC procedures applied (weight: 10)
        - Uncertainty assessment (weight: 10)
        - Change log / base year adjustments (weight: 5)

    Assumption Tracking:
        Each assumption has:
            - Unique ID, description, justification
            - Sensitivity (impact on results if assumption changes)
            - Validity period (when assumption should be reviewed)
            - Approval status (draft, reviewed, approved)
            - Evidence references

    Evidence Tracking:
        Each evidence record has:
            - Document type (invoice, meter reading, certificate, etc.)
            - SHA-256 hash of the original document for integrity
            - Chain of custody (who uploaded, when, from where)
            - Retention period (per regulatory requirements)

Regulatory References:
    - GHG Protocol Corporate Standard, Chapter 8 (Reporting)
    - ISO 14064-1:2018, Clause 10 (Documentation requirements)
    - ISO 14064-3:2019, Clause 6 (Verification evidence requirements)
    - CSRD / ESRS E1 AR 47-48 (Documentation and auditability)
    - ISAE 3410 Assurance on GHG Statements, para 49-55

Zero-Hallucination:
    - Completeness scoring uses deterministic weighted summation
    - Document hashing uses SHA-256 for integrity verification
    - All assessment criteria from published standards
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-044 Inventory Management
Engine:  9 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round2(value: Any) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round4(value: Any) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _hash_content(content: str) -> str:
    """Compute SHA-256 hash of string content for integrity.

    Args:
        content: String content to hash.

    Returns:
        SHA-256 hex digest.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DocumentType(str, Enum):
    """Types of methodology documentation.

    METHODOLOGY:       Methodology description document.
    EMISSION_FACTOR:   Emission factor source and reference.
    ACTIVITY_DATA:     Activity data collection method description.
    ASSUMPTION:        Assumption justification document.
    CALCULATION:       Calculation procedure documentation.
    QA_QC:             Quality assurance / quality control record.
    UNCERTAINTY:       Uncertainty assessment document.
    CHANGE_LOG:        Change log and base year adjustment record.
    BOUNDARY:          Organisational or operational boundary description.
    VERIFICATION:      External verification / assurance report.
    """
    METHODOLOGY = "methodology"
    EMISSION_FACTOR = "emission_factor"
    ACTIVITY_DATA = "activity_data"
    ASSUMPTION = "assumption"
    CALCULATION = "calculation"
    QA_QC = "qa_qc"
    UNCERTAINTY = "uncertainty"
    CHANGE_LOG = "change_log"
    BOUNDARY = "boundary"
    VERIFICATION = "verification"


class EvidenceType(str, Enum):
    """Types of supporting evidence.

    INVOICE:           Purchase invoice or receipt.
    METER_READING:     Meter reading record.
    CERTIFICATE:       Certificate (e.g., EAC, REC, GO).
    REPORT:            Third-party report.
    PHOTOGRAPH:        Photographic evidence.
    DATABASE_EXPORT:   Data exported from ERP/BMS.
    SUPPLIER_DATA:     Supplier-provided data sheet.
    GOVERNMENT_RECORD: Government or regulatory record.
    AUDIT_TRAIL:       Internal audit trail or log.
    CALCULATION_SHEET: Spreadsheet or calculation workbook.
    """
    INVOICE = "invoice"
    METER_READING = "meter_reading"
    CERTIFICATE = "certificate"
    REPORT = "report"
    PHOTOGRAPH = "photograph"
    DATABASE_EXPORT = "database_export"
    SUPPLIER_DATA = "supplier_data"
    GOVERNMENT_RECORD = "government_record"
    AUDIT_TRAIL = "audit_trail"
    CALCULATION_SHEET = "calculation_sheet"


class ApprovalStatus(str, Enum):
    """Approval status for assumptions and documents.

    DRAFT:     Document is in draft state.
    REVIEWED:  Document has been peer-reviewed.
    APPROVED:  Document has been formally approved.
    EXPIRED:   Document validity period has expired.
    SUPERSEDED: Document has been replaced by a newer version.
    """
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class SensitivityLevel(str, Enum):
    """Sensitivity level of an assumption.

    HIGH:   Change would materially affect results (>5% impact).
    MEDIUM: Change would moderately affect results (1-5% impact).
    LOW:    Change would have minimal effect (<1% impact).
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AssuranceReadiness(str, Enum):
    """Readiness level for external assurance.

    READY:           All documentation complete and approved.
    MOSTLY_READY:    Minor gaps that can be quickly addressed.
    PARTIALLY_READY: Significant gaps requiring attention.
    NOT_READY:       Major documentation gaps exist.
    """
    READY = "ready"
    MOSTLY_READY = "mostly_ready"
    PARTIALLY_READY = "partially_ready"
    NOT_READY = "not_ready"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Required documentation types and their weights for completeness scoring.
# Based on GHG Protocol Chapter 8 and ISO 14064-3 requirements.
REQUIRED_DOC_WEIGHTS: Dict[str, int] = {
    DocumentType.METHODOLOGY.value: 20,
    DocumentType.EMISSION_FACTOR.value: 15,
    DocumentType.ACTIVITY_DATA.value: 15,
    DocumentType.ASSUMPTION.value: 15,
    DocumentType.CALCULATION.value: 10,
    DocumentType.QA_QC.value: 10,
    DocumentType.UNCERTAINTY.value: 10,
    DocumentType.CHANGE_LOG.value: 5,
}
"""Documentation type weights for completeness calculation (total=100)."""

# Minimum documentation requirements for each assurance level.
ASSURANCE_THRESHOLDS: Dict[str, float] = {
    AssuranceReadiness.READY.value: 90.0,
    AssuranceReadiness.MOSTLY_READY.value: 75.0,
    AssuranceReadiness.PARTIALLY_READY.value: 50.0,
    AssuranceReadiness.NOT_READY.value: 0.0,
}
"""Completeness thresholds for assurance readiness classification."""

# Document retention periods (years) by type per regulatory requirements.
RETENTION_PERIODS: Dict[str, int] = {
    DocumentType.METHODOLOGY.value: 7,
    DocumentType.EMISSION_FACTOR.value: 7,
    DocumentType.ACTIVITY_DATA.value: 7,
    DocumentType.ASSUMPTION.value: 7,
    DocumentType.CALCULATION.value: 7,
    DocumentType.QA_QC.value: 5,
    DocumentType.UNCERTAINTY.value: 5,
    DocumentType.CHANGE_LOG.value: 10,
    DocumentType.BOUNDARY.value: 10,
    DocumentType.VERIFICATION.value: 10,
}
"""Required retention periods in years per document type."""


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class MethodologyDocument(BaseModel):
    """A methodology documentation record.

    Attributes:
        document_id: Unique document identifier.
        document_type: Type of documentation.
        title: Document title.
        description: Document description/summary.
        category_id: Source category this document applies to.
        category_name: Source category name.
        scope: Scope number (1, 2, 3).
        version: Document version.
        author: Document author.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        approval_status: Approval status.
        approved_by: Approver identifier.
        content_hash: SHA-256 hash of document content.
        file_path: Path to the document file.
        retention_years: Required retention period.
        expiry_date: Date when document review is due.
        tags: Searchable tags.
    """
    document_id: str = Field(default_factory=_new_uuid, description="Document ID")
    document_type: DocumentType = Field(..., description="Document type")
    title: str = Field(..., min_length=1, max_length=500, description="Title")
    description: str = Field(default="", description="Description")
    category_id: str = Field(default="", description="Source category ID")
    category_name: str = Field(default="", description="Source category name")
    scope: Optional[int] = Field(default=None, ge=1, le=3, description="Scope")
    version: str = Field(default="1.0", description="Version")
    author: str = Field(default="", description="Author")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Created timestamp"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="Updated timestamp"
    )
    approval_status: ApprovalStatus = Field(
        default=ApprovalStatus.DRAFT, description="Approval status"
    )
    approved_by: Optional[str] = Field(default=None, description="Approver")
    content_hash: str = Field(default="", description="SHA-256 content hash")
    file_path: str = Field(default="", description="File path")
    retention_years: int = Field(default=7, ge=1, le=30, description="Retention years")
    expiry_date: Optional[datetime] = Field(
        default=None, description="Review due date"
    )
    tags: List[str] = Field(default_factory=list, description="Tags")


class Assumption(BaseModel):
    """An inventory assumption tracked for transparency.

    Attributes:
        assumption_id: Unique assumption identifier.
        title: Short title of the assumption.
        description: Full description of the assumption.
        justification: Justification for why this assumption is reasonable.
        category_id: Source category this assumption applies to.
        category_name: Source category name.
        scope: Scope number.
        sensitivity: Impact level if assumption changes.
        impact_description: Description of impact if assumption is wrong.
        valid_from: Start of validity period.
        valid_until: End of validity period (review date).
        approval_status: Approval status.
        approved_by: Approver identifier.
        evidence_refs: References to supporting evidence.
        related_assumptions: IDs of related assumptions.
        review_frequency_months: How often this assumption should be reviewed.
    """
    assumption_id: str = Field(default_factory=_new_uuid, description="Assumption ID")
    title: str = Field(..., min_length=1, max_length=300, description="Title")
    description: str = Field(..., min_length=1, description="Full description")
    justification: str = Field(default="", description="Justification")
    category_id: str = Field(default="", description="Source category ID")
    category_name: str = Field(default="", description="Source category name")
    scope: Optional[int] = Field(default=None, ge=1, le=3, description="Scope")
    sensitivity: SensitivityLevel = Field(
        default=SensitivityLevel.MEDIUM, description="Sensitivity"
    )
    impact_description: str = Field(
        default="", description="Impact if assumption changes"
    )
    valid_from: Optional[datetime] = Field(
        default=None, description="Validity start"
    )
    valid_until: Optional[datetime] = Field(
        default=None, description="Validity end / review date"
    )
    approval_status: ApprovalStatus = Field(
        default=ApprovalStatus.DRAFT, description="Approval status"
    )
    approved_by: Optional[str] = Field(default=None, description="Approver")
    evidence_refs: List[str] = Field(
        default_factory=list, description="Evidence references"
    )
    related_assumptions: List[str] = Field(
        default_factory=list, description="Related assumption IDs"
    )
    review_frequency_months: int = Field(
        default=12, ge=1, le=60, description="Review frequency (months)"
    )


class EvidenceRecord(BaseModel):
    """A piece of supporting evidence for the inventory.

    Attributes:
        evidence_id: Unique evidence identifier.
        evidence_type: Type of evidence.
        title: Evidence title.
        description: Evidence description.
        category_id: Source category this evidence supports.
        category_name: Category name.
        scope: Scope number.
        document_hash: SHA-256 hash of the original document.
        file_path: Path to the evidence file.
        file_size_bytes: File size in bytes.
        mime_type: MIME type of the file.
        uploaded_by: Uploader identifier.
        uploaded_at: Upload timestamp.
        source_system: System from which evidence was obtained.
        chain_of_custody: Chain of custody trail.
        retention_until: Date until which evidence must be retained.
        is_verified: Whether the evidence has been verified.
        verified_by: Verifier identifier.
        verified_at: Verification timestamp.
    """
    evidence_id: str = Field(default_factory=_new_uuid, description="Evidence ID")
    evidence_type: EvidenceType = Field(..., description="Evidence type")
    title: str = Field(..., min_length=1, max_length=500, description="Title")
    description: str = Field(default="", description="Description")
    category_id: str = Field(default="", description="Source category ID")
    category_name: str = Field(default="", description="Category name")
    scope: Optional[int] = Field(default=None, ge=1, le=3, description="Scope")
    document_hash: str = Field(default="", description="SHA-256 document hash")
    file_path: str = Field(default="", description="File path")
    file_size_bytes: int = Field(default=0, ge=0, description="File size (bytes)")
    mime_type: str = Field(default="", description="MIME type")
    uploaded_by: str = Field(default="", description="Uploader")
    uploaded_at: datetime = Field(
        default_factory=_utcnow, description="Upload timestamp"
    )
    source_system: str = Field(default="", description="Source system")
    chain_of_custody: List[str] = Field(
        default_factory=list, description="Chain of custody"
    )
    retention_until: Optional[datetime] = Field(
        default=None, description="Retention date"
    )
    is_verified: bool = Field(default=False, description="Verified flag")
    verified_by: Optional[str] = Field(default=None, description="Verifier")
    verified_at: Optional[datetime] = Field(
        default=None, description="Verification timestamp"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class CategoryDocumentation(BaseModel):
    """Documentation completeness for a single source category.

    Attributes:
        category_id: Source category identifier.
        category_name: Category name.
        scope: Scope number.
        completeness_pct: Documentation completeness percentage.
        documents_present: List of document types present.
        documents_missing: List of document types missing.
        assumptions_count: Number of tracked assumptions.
        evidence_count: Number of evidence records.
        approved_docs_count: Number of approved documents.
        unapproved_docs_count: Number of unapproved documents.
        expired_docs_count: Number of expired documents.
    """
    category_id: str = Field(default="", description="Category ID")
    category_name: str = Field(default="", description="Category name")
    scope: int = Field(default=1, description="Scope")
    completeness_pct: float = Field(default=0.0, description="Completeness %")
    documents_present: List[str] = Field(
        default_factory=list, description="Present document types"
    )
    documents_missing: List[str] = Field(
        default_factory=list, description="Missing document types"
    )
    assumptions_count: int = Field(default=0, description="Assumptions tracked")
    evidence_count: int = Field(default=0, description="Evidence records")
    approved_docs_count: int = Field(default=0, description="Approved docs")
    unapproved_docs_count: int = Field(default=0, description="Unapproved docs")
    expired_docs_count: int = Field(default=0, description="Expired docs")


class DocumentationCompleteness(BaseModel):
    """Overall documentation completeness assessment.

    Attributes:
        overall_completeness_pct: Weighted overall completeness.
        assurance_readiness: Assurance readiness classification.
        total_documents: Total number of documents.
        total_assumptions: Total number of assumptions.
        total_evidence: Total number of evidence records.
        approved_pct: Percentage of documents approved.
        expired_count: Number of expired documents.
        high_sensitivity_assumptions: Count of high-sensitivity assumptions.
        assumptions_due_review: Count of assumptions due for review.
        category_completeness: Per-category completeness details.
        missing_critical_docs: List of critical missing documents.
        recommendations: Improvement recommendations.
    """
    overall_completeness_pct: float = Field(
        default=0.0, description="Overall completeness %"
    )
    assurance_readiness: str = Field(
        default="not_ready", description="Assurance readiness"
    )
    total_documents: int = Field(default=0, description="Total documents")
    total_assumptions: int = Field(default=0, description="Total assumptions")
    total_evidence: int = Field(default=0, description="Total evidence")
    approved_pct: float = Field(default=0.0, description="Approved %")
    expired_count: int = Field(default=0, description="Expired count")
    high_sensitivity_assumptions: int = Field(
        default=0, description="High-sensitivity assumptions"
    )
    assumptions_due_review: int = Field(
        default=0, description="Assumptions due review"
    )
    category_completeness: List[CategoryDocumentation] = Field(
        default_factory=list, description="Per-category completeness"
    )
    missing_critical_docs: List[str] = Field(
        default_factory=list, description="Critical missing docs"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )


class DocumentationResult(BaseModel):
    """Complete documentation assessment result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp (UTC).
        processing_time_ms: Processing time in milliseconds.
        completeness: Overall documentation completeness assessment.
        document_registry: All registered documents.
        assumption_registry: All tracked assumptions.
        evidence_registry: All evidence records.
        integrity_checks: Results of document integrity checks.
        retention_alerts: Documents approaching retention expiry.
        methodology_notes: Methodology notes.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    completeness: Optional[DocumentationCompleteness] = Field(
        default=None, description="Completeness assessment"
    )
    document_registry: List[MethodologyDocument] = Field(
        default_factory=list, description="Document registry"
    )
    assumption_registry: List[Assumption] = Field(
        default_factory=list, description="Assumption registry"
    )
    evidence_registry: List[EvidenceRecord] = Field(
        default_factory=list, description="Evidence registry"
    )
    integrity_checks: List[str] = Field(
        default_factory=list, description="Integrity check results"
    )
    retention_alerts: List[str] = Field(
        default_factory=list, description="Retention alerts"
    )
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology notes"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Model Rebuild
# ---------------------------------------------------------------------------

MethodologyDocument.model_rebuild()
Assumption.model_rebuild()
EvidenceRecord.model_rebuild()
CategoryDocumentation.model_rebuild()
DocumentationCompleteness.model_rebuild()
DocumentationResult.model_rebuild()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DocumentationEngine:
    """GHG inventory documentation and evidence management engine.

    Manages methodology documentation, assumption tracking, and evidence
    records to ensure inventory transparency and assurance-readiness.
    Calculates weighted completeness scores and identifies gaps.

    Features:
        - Methodology document registry with versioning and approval
        - Assumption tracking with sensitivity and validity periods
        - Evidence management with SHA-256 integrity verification
        - Weighted completeness scoring per source category
        - Assurance readiness assessment
        - Retention period monitoring and alerts
        - Document integrity verification

    Guarantees:
        - Deterministic: same inputs produce identical results
        - Reproducible: SHA-256 provenance hash on every result
        - Integrity: document content verified via SHA-256 hashes
        - No LLM: zero hallucination risk in any calculation path

    Usage::

        engine = DocumentationEngine()
        docs = [MethodologyDocument(...)]
        assumptions = [Assumption(...)]
        evidence = [EvidenceRecord(...)]
        result = engine.assess(
            documents=docs,
            assumptions=assumptions,
            evidence=evidence,
            category_ids=["cat_1", "cat_2"],
        )
        print(f"Completeness: {result.completeness.overall_completeness_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the documentation engine.

        Args:
            config: Optional configuration. Supported keys:
                - assurance_target (str): target assurance level
                - retention_alert_months (int): months before expiry to alert
                - review_lookahead_days (int): days to look ahead for reviews
        """
        self._config = config or {}
        self._assurance_target = self._config.get(
            "assurance_target", AssuranceReadiness.READY.value
        )
        self._retention_alert_months = int(
            self._config.get("retention_alert_months", 6)
        )
        self._review_lookahead_days = int(
            self._config.get("review_lookahead_days", 90)
        )
        self._notes: List[str] = []
        logger.info("DocumentationEngine v%s initialised.", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def assess(
        self,
        documents: List[MethodologyDocument],
        assumptions: Optional[List[Assumption]] = None,
        evidence: Optional[List[EvidenceRecord]] = None,
        category_ids: Optional[List[str]] = None,
        category_emissions: Optional[Dict[str, Decimal]] = None,
    ) -> DocumentationResult:
        """Run complete documentation assessment.

        Args:
            documents: All methodology documents.
            assumptions: Optional assumptions registry.
            evidence: Optional evidence records.
            category_ids: Source category IDs to assess. If None, derived
                from documents.
            category_emissions: Optional category_id to emissions mapping
                for weighted scoring.

        Returns:
            DocumentationResult with completeness assessment.
        """
        t0 = time.perf_counter()
        self._notes = [f"Engine version: {self.engine_version}"]
        assumptions = assumptions or []
        evidence = evidence or []

        # Derive category IDs from documents if not provided
        if category_ids is None:
            cat_set: Set[str] = set()
            for doc in documents:
                if doc.category_id:
                    cat_set.add(doc.category_id)
            for asn in assumptions:
                if asn.category_id:
                    cat_set.add(asn.category_id)
            category_ids = sorted(cat_set)

        logger.info(
            "Documentation assessment: %d documents, %d assumptions, "
            "%d evidence records, %d categories",
            len(documents), len(assumptions), len(evidence), len(category_ids),
        )

        # Step 1: Assess per-category completeness
        cat_assessments = self._assess_categories(
            documents, assumptions, evidence, category_ids
        )

        # Step 2: Calculate overall completeness
        completeness = self._calculate_overall_completeness(
            cat_assessments, documents, assumptions, evidence, category_emissions
        )

        # Step 3: Check document integrity
        integrity_checks = self._check_integrity(documents, evidence)

        # Step 4: Check retention periods
        retention_alerts = self._check_retention(documents, evidence)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = DocumentationResult(
            completeness=completeness,
            document_registry=documents,
            assumption_registry=assumptions,
            evidence_registry=evidence,
            integrity_checks=integrity_checks,
            retention_alerts=retention_alerts,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Documentation assessment complete: completeness=%.1f%%, "
            "readiness=%s, docs=%d, hash=%s (%.1f ms)",
            completeness.overall_completeness_pct,
            completeness.assurance_readiness,
            len(documents),
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    def calculate_category_completeness(
        self,
        category_id: str,
        documents: List[MethodologyDocument],
    ) -> float:
        """Calculate documentation completeness for a single category.

        Uses weighted scoring based on REQUIRED_DOC_WEIGHTS. Each
        document type that is present and approved contributes its
        full weight; present but unapproved contributes 70%.

        Args:
            category_id: Source category identifier.
            documents: Documents for this category.

        Returns:
            Completeness percentage (0-100).
        """
        category_docs = [d for d in documents if d.category_id == category_id]
        if not category_docs:
            return 0.0

        total_weight = sum(REQUIRED_DOC_WEIGHTS.values())
        achieved_weight = Decimal("0")

        # Check which document types are present
        present_types: Set[str] = set()
        for doc in category_docs:
            present_types.add(doc.document_type.value)

        for doc_type, weight in REQUIRED_DOC_WEIGHTS.items():
            if doc_type in present_types:
                # Check if any approved version exists
                approved = any(
                    d.approval_status == ApprovalStatus.APPROVED
                    and d.document_type.value == doc_type
                    and d.category_id == category_id
                    for d in category_docs
                )
                if approved:
                    achieved_weight += _decimal(weight)
                else:
                    # Present but not approved: 70% credit
                    achieved_weight += _decimal(weight) * Decimal("0.7")

        return float(_safe_pct(achieved_weight, _decimal(total_weight)))

    def verify_document_integrity(
        self,
        document: MethodologyDocument,
        current_content_hash: str,
    ) -> bool:
        """Verify document integrity by comparing content hashes.

        Args:
            document: Document record with stored hash.
            current_content_hash: Current SHA-256 hash of the content.

        Returns:
            True if hashes match, False if document has been tampered with.
        """
        if not document.content_hash:
            logger.warning(
                "Document '%s' has no stored content hash.", document.title
            )
            return False

        match = document.content_hash == current_content_hash
        if not match:
            logger.warning(
                "Integrity check FAILED for document '%s': "
                "stored=%s, current=%s",
                document.title,
                document.content_hash[:16],
                current_content_hash[:16],
            )
        return match

    def get_assumptions_due_for_review(
        self,
        assumptions: List[Assumption],
        as_of: Optional[datetime] = None,
    ) -> List[Assumption]:
        """Get assumptions that are due for review.

        Args:
            assumptions: All tracked assumptions.
            as_of: Reference date (defaults to now).

        Returns:
            List of assumptions due or overdue for review.
        """
        reference = as_of or _utcnow()
        due: List[Assumption] = []

        for asn in assumptions:
            if asn.valid_until is None:
                continue

            # Check if valid_until is within the lookahead window
            valid_until = asn.valid_until
            if hasattr(valid_until, "tzinfo") and valid_until.tzinfo is None:
                valid_until = valid_until.replace(tzinfo=timezone.utc)

            days_until = (valid_until - reference).days
            if days_until <= self._review_lookahead_days:
                due.append(asn)

        return due

    def register_document(
        self,
        document: MethodologyDocument,
        content: Optional[str] = None,
    ) -> MethodologyDocument:
        """Register a new document with optional content hashing.

        Args:
            document: Document to register.
            content: Optional document content for hash calculation.

        Returns:
            Document with content_hash populated if content provided.
        """
        if content:
            document.content_hash = _hash_content(content)
            logger.info(
                "Document '%s' registered with hash %s",
                document.title, document.content_hash[:16],
            )

        # Set retention based on type
        doc_type_val = document.document_type.value
        if document.retention_years == 7:
            default_retention = RETENTION_PERIODS.get(doc_type_val, 7)
            document.retention_years = default_retention

        return document

    # -------------------------------------------------------------------
    # Private -- Category assessment
    # -------------------------------------------------------------------

    def _assess_categories(
        self,
        documents: List[MethodologyDocument],
        assumptions: List[Assumption],
        evidence: List[EvidenceRecord],
        category_ids: List[str],
    ) -> List[CategoryDocumentation]:
        """Assess documentation completeness per category.

        Args:
            documents: All documents.
            assumptions: All assumptions.
            evidence: All evidence.
            category_ids: Categories to assess.

        Returns:
            List of CategoryDocumentation assessments.
        """
        assessments: List[CategoryDocumentation] = []
        now = _utcnow()

        for cat_id in category_ids:
            cat_docs = [d for d in documents if d.category_id == cat_id]
            cat_asns = [a for a in assumptions if a.category_id == cat_id]
            cat_evid = [e for e in evidence if e.category_id == cat_id]

            # Determine present and missing document types
            present: Set[str] = set()
            for doc in cat_docs:
                present.add(doc.document_type.value)

            missing = [
                dt for dt in REQUIRED_DOC_WEIGHTS.keys()
                if dt not in present
            ]

            # Count approved, unapproved, expired
            approved_count = sum(
                1 for d in cat_docs if d.approval_status == ApprovalStatus.APPROVED
            )
            expired_count = sum(
                1 for d in cat_docs
                if d.expiry_date is not None and d.expiry_date < now
            )
            unapproved_count = len(cat_docs) - approved_count

            completeness = self.calculate_category_completeness(cat_id, documents)

            # Get category name from first document
            cat_name = ""
            scope_val = 1
            for doc in cat_docs:
                if doc.category_name:
                    cat_name = doc.category_name
                    scope_val = doc.scope or 1
                    break
            if not cat_name:
                for asn in cat_asns:
                    if asn.category_name:
                        cat_name = asn.category_name
                        scope_val = asn.scope or 1
                        break

            assessments.append(CategoryDocumentation(
                category_id=cat_id,
                category_name=cat_name,
                scope=scope_val,
                completeness_pct=_round2(completeness),
                documents_present=sorted(present),
                documents_missing=missing,
                assumptions_count=len(cat_asns),
                evidence_count=len(cat_evid),
                approved_docs_count=approved_count,
                unapproved_docs_count=unapproved_count,
                expired_docs_count=expired_count,
            ))

        return assessments

    # -------------------------------------------------------------------
    # Private -- Overall completeness
    # -------------------------------------------------------------------

    def _calculate_overall_completeness(
        self,
        cat_assessments: List[CategoryDocumentation],
        documents: List[MethodologyDocument],
        assumptions: List[Assumption],
        evidence: List[EvidenceRecord],
        category_emissions: Optional[Dict[str, Decimal]] = None,
    ) -> DocumentationCompleteness:
        """Calculate overall documentation completeness.

        Args:
            cat_assessments: Per-category assessments.
            documents: All documents.
            assumptions: All assumptions.
            evidence: All evidence.
            category_emissions: Optional emissions weighting.

        Returns:
            DocumentationCompleteness assessment.
        """
        if not cat_assessments:
            return DocumentationCompleteness()

        # Calculate weighted overall completeness
        if category_emissions:
            total_em = sum(category_emissions.values(), Decimal("0"))
            weighted_sum = Decimal("0")
            for cat in cat_assessments:
                em = category_emissions.get(cat.category_id, Decimal("0"))
                weight = _safe_divide(em, total_em) if total_em > Decimal("0") else (
                    _decimal(1) / _decimal(len(cat_assessments))
                )
                weighted_sum += _decimal(cat.completeness_pct) * weight
            overall = float(weighted_sum)
        else:
            # Equal weighting
            total = sum(c.completeness_pct for c in cat_assessments)
            overall = total / len(cat_assessments) if cat_assessments else 0.0

        # Assurance readiness
        readiness = AssuranceReadiness.NOT_READY.value
        for level, threshold in sorted(
            ASSURANCE_THRESHOLDS.items(), key=lambda x: x[1], reverse=True
        ):
            if overall >= threshold:
                readiness = level
                break

        # Approved percentage
        total_docs = len(documents)
        approved = sum(
            1 for d in documents if d.approval_status == ApprovalStatus.APPROVED
        )
        approved_pct = float(_safe_pct(_decimal(approved), _decimal(total_docs)))

        # Expired documents
        now = _utcnow()
        expired = sum(
            1 for d in documents
            if d.expiry_date is not None and d.expiry_date < now
        )

        # High sensitivity assumptions
        high_sens = sum(
            1 for a in assumptions if a.sensitivity == SensitivityLevel.HIGH
        )

        # Assumptions due for review
        due_review = len(self.get_assumptions_due_for_review(assumptions))

        # Missing critical documents (methodology or EF for any category)
        missing_critical: List[str] = []
        for cat in cat_assessments:
            for doc_type in cat.documents_missing:
                if doc_type in (
                    DocumentType.METHODOLOGY.value,
                    DocumentType.EMISSION_FACTOR.value,
                ):
                    missing_critical.append(
                        f"{cat.category_name}: missing {doc_type}"
                    )

        # Recommendations
        recommendations = self._generate_doc_recommendations(
            cat_assessments, assumptions, overall, readiness
        )

        return DocumentationCompleteness(
            overall_completeness_pct=_round2(overall),
            assurance_readiness=readiness,
            total_documents=total_docs,
            total_assumptions=len(assumptions),
            total_evidence=len(evidence),
            approved_pct=_round2(approved_pct),
            expired_count=expired,
            high_sensitivity_assumptions=high_sens,
            assumptions_due_review=due_review,
            category_completeness=cat_assessments,
            missing_critical_docs=missing_critical,
            recommendations=recommendations,
        )

    # -------------------------------------------------------------------
    # Private -- Integrity checks
    # -------------------------------------------------------------------

    def _check_integrity(
        self,
        documents: List[MethodologyDocument],
        evidence: List[EvidenceRecord],
    ) -> List[str]:
        """Check document and evidence integrity.

        Args:
            documents: Methodology documents.
            evidence: Evidence records.

        Returns:
            List of integrity check result messages.
        """
        checks: List[str] = []

        # Check documents with missing hashes
        no_hash_docs = [d for d in documents if not d.content_hash]
        if no_hash_docs:
            checks.append(
                f"{len(no_hash_docs)} document(s) have no content hash. "
                f"Content integrity cannot be verified."
            )

        no_hash_evidence = [e for e in evidence if not e.document_hash]
        if no_hash_evidence:
            checks.append(
                f"{len(no_hash_evidence)} evidence record(s) have no document hash."
            )

        # Check for unverified evidence
        unverified = [e for e in evidence if not e.is_verified]
        if unverified:
            checks.append(
                f"{len(unverified)} evidence record(s) are not yet verified."
            )

        if not checks:
            checks.append("All integrity checks passed.")

        return checks

    # -------------------------------------------------------------------
    # Private -- Retention checks
    # -------------------------------------------------------------------

    def _check_retention(
        self,
        documents: List[MethodologyDocument],
        evidence: List[EvidenceRecord],
    ) -> List[str]:
        """Check document retention periods for upcoming expiry.

        Args:
            documents: Methodology documents.
            evidence: Evidence records.

        Returns:
            List of retention alert messages.
        """
        alerts: List[str] = []
        now = _utcnow()

        for doc in documents:
            if doc.expiry_date is not None:
                expiry = doc.expiry_date
                if hasattr(expiry, "tzinfo") and expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)

                days_until = (expiry - now).days
                if days_until < 0:
                    alerts.append(
                        f"EXPIRED: Document '{doc.title}' expired "
                        f"{abs(days_until)} days ago."
                    )
                elif days_until <= self._retention_alert_months * 30:
                    alerts.append(
                        f"APPROACHING: Document '{doc.title}' expires in "
                        f"{days_until} days."
                    )

        for ev in evidence:
            if ev.retention_until is not None:
                ret = ev.retention_until
                if hasattr(ret, "tzinfo") and ret.tzinfo is None:
                    ret = ret.replace(tzinfo=timezone.utc)

                days_until = (ret - now).days
                if days_until < 0:
                    alerts.append(
                        f"EXPIRED: Evidence '{ev.title}' retention expired "
                        f"{abs(days_until)} days ago."
                    )
                elif days_until <= self._retention_alert_months * 30:
                    alerts.append(
                        f"APPROACHING: Evidence '{ev.title}' retention expires "
                        f"in {days_until} days."
                    )

        return alerts

    # -------------------------------------------------------------------
    # Private -- Recommendations
    # -------------------------------------------------------------------

    def _generate_doc_recommendations(
        self,
        cat_assessments: List[CategoryDocumentation],
        assumptions: List[Assumption],
        overall_pct: float,
        readiness: str,
    ) -> List[str]:
        """Generate documentation improvement recommendations.

        Args:
            cat_assessments: Per-category assessments.
            assumptions: Assumptions registry.
            overall_pct: Overall completeness percentage.
            readiness: Current assurance readiness.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # Categories with low completeness
        low_cats = [
            c for c in cat_assessments if c.completeness_pct < 50.0
        ]
        if low_cats:
            names = ", ".join(c.category_name or c.category_id for c in low_cats[:5])
            recommendations.append(
                f"{len(low_cats)} category(s) have completeness below 50%: "
                f"{names}. Prioritise methodology and emission factor documentation."
            )

        # Unapproved documents
        total_unapproved = sum(c.unapproved_docs_count for c in cat_assessments)
        if total_unapproved > 0:
            recommendations.append(
                f"{total_unapproved} document(s) are awaiting approval. "
                f"Complete the review and approval process."
            )

        # Expired documents
        total_expired = sum(c.expired_docs_count for c in cat_assessments)
        if total_expired > 0:
            recommendations.append(
                f"{total_expired} document(s) have expired. "
                f"Review and update expired documentation."
            )

        # Assumptions without evidence
        no_evidence_asns = [
            a for a in assumptions if not a.evidence_refs
        ]
        if no_evidence_asns:
            recommendations.append(
                f"{len(no_evidence_asns)} assumption(s) lack supporting evidence. "
                f"Attach evidence references for audit trail."
            )

        # High sensitivity assumptions pending approval
        pending_high = [
            a for a in assumptions
            if a.sensitivity == SensitivityLevel.HIGH
            and a.approval_status != ApprovalStatus.APPROVED
        ]
        if pending_high:
            recommendations.append(
                f"{len(pending_high)} high-sensitivity assumption(s) are not yet "
                f"approved. These should be prioritised for review."
            )

        # Assurance readiness gap
        target_threshold = ASSURANCE_THRESHOLDS.get(
            self._assurance_target, 90.0
        )
        if overall_pct < target_threshold:
            gap = target_threshold - overall_pct
            recommendations.append(
                f"Documentation completeness ({_round2(overall_pct)}%) is "
                f"{_round2(gap)} percentage points below the target for "
                f"'{self._assurance_target}' assurance readiness."
            )

        if not recommendations:
            recommendations.append(
                "Documentation is comprehensive. Maintain current practices "
                "and ensure regular reviews of assumptions and evidence."
            )

        self._notes.append(
            f"Generated {len(recommendations)} documentation recommendations."
        )
        return recommendations
