# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-012 Document Authentication

Pydantic v2 request/response models for all Document Authentication REST API
endpoints. Organized by domain: classification, signature verification, hash
integrity, certificate validation, metadata extraction, fraud detection,
cross-reference verification, reports, batch jobs, and health.

All models use ``ConfigDict(from_attributes=True)`` for ORM compatibility
and ``Field()`` with descriptions for OpenAPI documentation.

All hash and verification fields use deterministic algorithms (SHA-256,
SHA-512) required by EUDR Article 14 and eIDAS Regulation (EU) No 910/2014.

Model Count: 80+ schemas covering 37 endpoints.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-012, Section 7.4
Agent ID: GL-EUDR-DAV-012
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator

from greenlang.schemas import GreenLangBase, utcnow

# =============================================================================
# Helpers
# =============================================================================

def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

# =============================================================================
# Enumerations (API-layer mirrors of domain enums)
# =============================================================================

class DocumentTypeSchema(str, Enum):
    """Type of EUDR supply chain document."""

    COO = "coo"
    PC = "pc"
    BOL = "bol"
    CDE = "cde"
    CDI = "cdi"
    RSPO_CERT = "rspo_cert"
    FSC_CERT = "fsc_cert"
    ISCC_CERT = "iscc_cert"
    FT_CERT = "ft_cert"
    UTZ_CERT = "utz_cert"
    LTR = "ltr"
    LTD = "ltd"
    FMP = "fmp"
    FC = "fc"
    WQC = "wqc"
    DDS_DRAFT = "dds_draft"
    SSD = "ssd"
    IC = "ic"
    TC = "tc"
    WR = "wr"

class ClassificationConfidenceSchema(str, Enum):
    """Confidence level of document classification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class SignatureStandardSchema(str, Enum):
    """Digital signature standard."""

    CADES = "cades"
    PADES = "pades"
    XADES = "xades"
    JADES = "jades"
    QES = "qes"
    PGP = "pgp"
    PKCS7 = "pkcs7"

class SignatureStatusSchema(str, Enum):
    """Verification status of a digital signature."""

    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    NO_SIGNATURE = "no_signature"
    UNKNOWN_SIGNER = "unknown_signer"
    STRIPPED = "stripped"

class HashAlgorithmSchema(str, Enum):
    """Cryptographic hash algorithm for document integrity."""

    SHA256 = "sha256"
    SHA512 = "sha512"
    HMAC_SHA256 = "hmac_sha256"

class CertificateStatusSchema(str, Enum):
    """Validation status of a signing certificate."""

    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SELF_SIGNED = "self_signed"
    WEAK_KEY = "weak_key"
    UNKNOWN = "unknown"

class FraudSeveritySchema(str, Enum):
    """Severity level of a detected fraud pattern."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FraudPatternTypeSchema(str, Enum):
    """Type of fraud pattern detected in document analysis."""

    DUPLICATE_REUSE = "duplicate_reuse"
    QUANTITY_TAMPERING = "quantity_tampering"
    DATE_MANIPULATION = "date_manipulation"
    EXPIRED_CERT = "expired_cert"
    SERIAL_ANOMALY = "serial_anomaly"
    ISSUER_MISMATCH = "issuer_mismatch"
    TEMPLATE_FORGERY = "template_forgery"
    CROSS_DOC_INCONSISTENCY = "cross_doc_inconsistency"
    GEO_IMPOSSIBILITY = "geo_impossibility"
    VELOCITY_ANOMALY = "velocity_anomaly"
    MODIFICATION_ANOMALY = "modification_anomaly"
    ROUND_NUMBER_BIAS = "round_number_bias"
    COPY_PASTE = "copy_paste"
    MISSING_REQUIRED = "missing_required"
    SCOPE_MISMATCH = "scope_mismatch"

class VerificationStatusSchema(str, Enum):
    """Overall status of a document verification process."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class RegistryTypeSchema(str, Enum):
    """Type of external registry for cross-reference verification."""

    FSC = "fsc"
    RSPO = "rspo"
    ISCC = "iscc"
    FAIRTRADE = "fairtrade"
    UTZ_RA = "utz_ra"
    IPPC = "ippc"
    NATIONAL_CUSTOMS = "national_customs"

class ReportFormatSchema(str, Enum):
    """Output format for generated reports."""

    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    EUDR_XML = "eudr_xml"

class AuthenticationResultSchema(str, Enum):
    """Overall authentication result for a document."""

    AUTHENTIC = "authentic"
    SUSPICIOUS = "suspicious"
    FRAUDULENT = "fraudulent"
    INCONCLUSIVE = "inconclusive"

class BatchJobStatusSchema(str, Enum):
    """Status of an async batch job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BatchJobTypeSchema(str, Enum):
    """Types of batch jobs supported by document authentication."""

    CLASSIFY_BATCH = "classify_batch"
    VERIFY_SIGNATURES_BATCH = "verify_signatures_batch"
    DETECT_FRAUD_BATCH = "detect_fraud_batch"
    CROSSREF_BATCH = "crossref_batch"
    REPORT_GENERATION = "report_generation"

class SortOrderSchema(str, Enum):
    """Sort order for list endpoints."""

    ASC = "asc"
    DESC = "desc"

# =============================================================================
# Shared / Common Models
# =============================================================================

class ProvenanceInfo(GreenLangBase):
    """Provenance tracking information for audit trail."""

    provenance_hash: str = Field(
        ..., description="SHA-256 hash of input data"
    )
    created_by: str = Field(..., description="User ID who created the record")
    created_at: datetime = Field(
        default_factory=utcnow, description="Creation timestamp (UTC)"
    )
    source: str = Field(
        default="api", description="Data source (api, import, system)"
    )

    model_config = ConfigDict(from_attributes=True)

class PaginatedMeta(GreenLangBase):
    """Pagination metadata for list responses."""

    total: int = Field(..., ge=0, description="Total number of results")
    limit: int = Field(..., ge=1, description="Maximum results returned")
    offset: int = Field(..., ge=0, description="Results skipped")
    has_more: bool = Field(..., description="Whether more results exist")

    model_config = ConfigDict(from_attributes=True)

class ErrorDetail(GreenLangBase):
    """Individual error detail within an error response."""

    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error description")
    code: Optional[str] = Field(None, description="Error code")

    model_config = ConfigDict(from_attributes=True)

class ErrorResponse(GreenLangBase):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    errors: List[ErrorDetail] = Field(
        default_factory=list, description="List of individual errors"
    )
    request_id: Optional[str] = Field(None, description="Request correlation ID")

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Classification Schemas
# =============================================================================

class ClassifyDocumentSchema(GreenLangBase):
    """Request to classify a single EUDR document by type."""

    document_reference: str = Field(
        ..., min_length=1, max_length=500,
        description="Document reference (file path, S3 key, or URL)",
    )
    operator_id: Optional[str] = Field(
        None, max_length=100,
        description="EUDR operator identifier submitting the document",
    )
    commodity: Optional[str] = Field(
        None, max_length=50,
        description="Expected commodity context for classification hint",
    )
    expected_type: Optional[DocumentTypeSchema] = Field(
        None, description="Expected document type for validation",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "document_reference": "s3://eudr-docs/coo-gh-2026-001.pdf",
                    "operator_id": "OP-GH-001",
                    "commodity": "cocoa",
                }
            ]
        },
    )

class BatchClassifySchema(GreenLangBase):
    """Request to classify multiple documents in batch."""

    documents: List[ClassifyDocumentSchema] = Field(
        ..., min_length=1, max_length=500,
        description="List of documents to classify (max 500)",
    )
    operator_id: Optional[str] = Field(
        None, max_length=100,
        description="Operator performing the batch classification",
    )
    validate_only: bool = Field(
        default=False,
        description="If true, validate without persisting results",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("documents")
    @classmethod
    def validate_batch_size(cls, v: List[ClassifyDocumentSchema]) -> List[ClassifyDocumentSchema]:
        """Ensure batch does not exceed maximum size."""
        if len(v) > 500:
            raise ValueError(f"Maximum 500 documents per batch, got {len(v)}")
        return v

class ClassificationResultSchema(GreenLangBase):
    """Response with document classification result."""

    document_id: str = Field(..., description="Unique document identifier")
    document_reference: str = Field(..., description="Document reference")
    document_type: DocumentTypeSchema = Field(..., description="Classified document type")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence score (0-1)"
    )
    confidence_level: ClassificationConfidenceSchema = Field(
        ..., description="Confidence tier (HIGH/MEDIUM/LOW/UNKNOWN)"
    )
    alternative_types: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative type candidates with confidence scores",
    )
    template_matched: Optional[str] = Field(
        None, description="Template ID matched during classification"
    )
    commodity: Optional[str] = Field(None, description="Detected commodity context")
    operator_id: Optional[str] = Field(None, description="Submitting operator")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Classification timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class BatchClassificationResultSchema(GreenLangBase):
    """Response for batch document classification."""

    total_submitted: int = Field(..., ge=0, description="Total documents submitted")
    total_classified: int = Field(..., ge=0, description="Successfully classified")
    total_failed: int = Field(..., ge=0, description="Failed classifications")
    results: List[ClassificationResultSchema] = Field(
        default_factory=list, description="Classification results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Classification errors"
    )
    validate_only: bool = Field(default=False, description="Validation-only mode")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class TemplateSchema(GreenLangBase):
    """Document classification template for known document layouts."""

    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template display name")
    document_type: DocumentTypeSchema = Field(..., description="Associated document type")
    issuing_authority: Optional[str] = Field(None, description="Issuing authority name")
    country_code: Optional[str] = Field(None, description="ISO 3166-1 alpha-2 country code")
    version: str = Field(default="1.0", description="Template version")
    active: bool = Field(default=True, description="Whether template is active")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Template metadata")
    created_at: datetime = Field(default_factory=utcnow, description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True)

class RegisterTemplateSchema(GreenLangBase):
    """Request to register a new classification template."""

    name: str = Field(
        ..., min_length=1, max_length=200,
        description="Template display name",
    )
    document_type: DocumentTypeSchema = Field(
        ..., description="Document type this template matches",
    )
    issuing_authority: Optional[str] = Field(
        None, max_length=200, description="Issuing authority name",
    )
    country_code: Optional[str] = Field(
        None, max_length=2, description="ISO 3166-1 alpha-2 country code",
    )
    layout_features: Dict[str, Any] = Field(
        default_factory=dict, description="Template layout feature descriptors",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(extra="forbid")

class TemplateListSchema(GreenLangBase):
    """Response listing available classification templates."""

    templates: List[TemplateSchema] = Field(
        default_factory=list, description="Available templates"
    )
    total_count: int = Field(..., ge=0, description="Total templates")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(default_factory=utcnow, description="Response timestamp")

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Signature Schemas
# =============================================================================

class VerifySignatureSchema(GreenLangBase):
    """Request to verify a document's digital signature."""

    document_reference: str = Field(
        ..., min_length=1, max_length=500,
        description="Document reference (file path, S3 key, or URL)",
    )
    expected_standard: Optional[SignatureStandardSchema] = Field(
        None, description="Expected signature standard",
    )
    expected_signer: Optional[str] = Field(
        None, max_length=200,
        description="Expected signer common name for validation",
    )
    check_timestamp: bool = Field(
        default=True, description="Verify signed timestamp per eIDAS",
    )
    check_revocation: bool = Field(
        default=True, description="Check certificate revocation (OCSP/CRL)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "document_reference": "s3://eudr-docs/coo-gh-2026-001.pdf",
                    "expected_standard": "pades",
                    "check_timestamp": True,
                    "check_revocation": True,
                }
            ]
        },
    )

class BatchVerifySignatureSchema(GreenLangBase):
    """Request to verify signatures for multiple documents."""

    documents: List[VerifySignatureSchema] = Field(
        ..., min_length=1, max_length=500,
        description="Documents to verify signatures for (max 500)",
    )
    operator_id: Optional[str] = Field(
        None, max_length=100, description="Operator performing verification",
    )

    model_config = ConfigDict(extra="forbid")

class SignatureResultSchema(GreenLangBase):
    """Response with signature verification result."""

    verification_id: str = Field(..., description="Unique verification identifier")
    document_reference: str = Field(..., description="Document reference")
    signature_status: SignatureStatusSchema = Field(
        ..., description="Signature verification status"
    )
    signature_standard: Optional[SignatureStandardSchema] = Field(
        None, description="Detected signature standard"
    )
    signer_name: Optional[str] = Field(None, description="Signer common name")
    signer_organization: Optional[str] = Field(None, description="Signer organization")
    signing_time: Optional[datetime] = Field(None, description="Timestamp of signing")
    timestamp_valid: Optional[bool] = Field(
        None, description="Whether signed timestamp is valid"
    )
    certificate_status: Optional[CertificateStatusSchema] = Field(
        None, description="Signing certificate status"
    )
    key_size_bits: Optional[int] = Field(None, description="Signing key size in bits")
    algorithm: Optional[str] = Field(None, description="Signature algorithm used")
    issues: List[str] = Field(
        default_factory=list, description="Verification issues found"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Verification timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class BatchSignatureResultSchema(GreenLangBase):
    """Response for batch signature verification."""

    total_submitted: int = Field(..., ge=0, description="Total documents submitted")
    total_verified: int = Field(..., ge=0, description="Successfully verified")
    total_failed: int = Field(..., ge=0, description="Failed verifications")
    results: List[SignatureResultSchema] = Field(
        default_factory=list, description="Verification results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Verification errors"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class SignatureHistorySchema(GreenLangBase):
    """Response with signature verification history for a document."""

    document_id: str = Field(..., description="Document identifier")
    verifications: List[SignatureResultSchema] = Field(
        default_factory=list, description="Historical signature verifications"
    )
    total_count: int = Field(..., ge=0, description="Total verification records")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(default_factory=utcnow, description="Response timestamp")

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Hash Integrity Schemas
# =============================================================================

class ComputeHashSchema(GreenLangBase):
    """Request to compute a cryptographic hash for a document."""

    document_reference: str = Field(
        ..., min_length=1, max_length=500,
        description="Document reference (file path, S3 key, or URL)",
    )
    algorithm: HashAlgorithmSchema = Field(
        default=HashAlgorithmSchema.SHA256,
        description="Hash algorithm to use",
    )
    compute_secondary: bool = Field(
        default=True,
        description="Also compute secondary hash for dual verification",
    )
    register_in_registry: bool = Field(
        default=True,
        description="Register computed hash in the integrity registry",
    )
    dds_id: Optional[str] = Field(
        None, max_length=100,
        description="DDS package ID for Merkle tree inclusion",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "document_reference": "s3://eudr-docs/coo-gh-2026-001.pdf",
                    "algorithm": "sha256",
                    "compute_secondary": True,
                    "register_in_registry": True,
                }
            ]
        },
    )

class VerifyHashSchema(GreenLangBase):
    """Request to verify a document against a stored hash."""

    document_reference: str = Field(
        ..., min_length=1, max_length=500,
        description="Document reference to verify",
    )
    expected_hash: str = Field(
        ..., min_length=32, max_length=256,
        description="Expected hash value to verify against",
    )
    algorithm: HashAlgorithmSchema = Field(
        default=HashAlgorithmSchema.SHA256,
        description="Hash algorithm the expected_hash was computed with",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(extra="forbid")

class HashResultSchema(GreenLangBase):
    """Response with hash computation or verification result."""

    hash_id: str = Field(..., description="Unique hash record identifier")
    document_reference: str = Field(..., description="Document reference")
    primary_hash: str = Field(..., description="Primary hash value")
    primary_algorithm: HashAlgorithmSchema = Field(
        ..., description="Primary hash algorithm used"
    )
    secondary_hash: Optional[str] = Field(
        None, description="Secondary hash value for dual verification"
    )
    secondary_algorithm: Optional[HashAlgorithmSchema] = Field(
        None, description="Secondary hash algorithm used"
    )
    hash_match: Optional[bool] = Field(
        None, description="Whether hash matches expected (for verify)"
    )
    registered: bool = Field(
        default=False, description="Whether hash is in the integrity registry"
    )
    registry_entry_id: Optional[str] = Field(
        None, description="Registry entry ID if registered"
    )
    dds_id: Optional[str] = Field(
        None, description="Associated DDS package ID"
    )
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="File size in bytes"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Computation timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class RegistryLookupSchema(GreenLangBase):
    """Response for hash registry lookup."""

    hash_value: str = Field(..., description="Queried hash value")
    found: bool = Field(..., description="Whether hash exists in registry")
    document_reference: Optional[str] = Field(
        None, description="Associated document reference"
    )
    algorithm: Optional[HashAlgorithmSchema] = Field(
        None, description="Hash algorithm"
    )
    registered_at: Optional[datetime] = Field(
        None, description="Registration timestamp"
    )
    registered_by: Optional[str] = Field(
        None, description="User who registered the hash"
    )
    dds_id: Optional[str] = Field(None, description="Associated DDS package")
    expires_at: Optional[datetime] = Field(
        None, description="Registry entry expiration date"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class MerkleNodeSchema(GreenLangBase):
    """Single node in a Merkle tree."""

    hash_value: str = Field(..., description="Node hash value")
    position: str = Field(..., description="Position (left/right)")
    level: int = Field(..., ge=0, description="Tree level (0 = leaf)")

    model_config = ConfigDict(from_attributes=True)

class MerkleTreeSchema(GreenLangBase):
    """Response with Merkle tree for a DDS package."""

    dds_id: str = Field(..., description="DDS package identifier")
    merkle_root: str = Field(..., description="Merkle root hash")
    algorithm: HashAlgorithmSchema = Field(
        ..., description="Hash algorithm used"
    )
    leaf_count: int = Field(..., ge=0, description="Number of leaf nodes")
    tree_depth: int = Field(..., ge=0, description="Tree depth")
    document_hashes: List[str] = Field(
        default_factory=list, description="Leaf document hashes"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Computation timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Certificate Schemas
# =============================================================================

class ValidateCertificateSchema(GreenLangBase):
    """Request to validate a certificate chain."""

    document_reference: str = Field(
        ..., min_length=1, max_length=500,
        description="Document reference containing the certificate",
    )
    check_ocsp: bool = Field(
        default=True, description="Check OCSP for revocation status",
    )
    check_crl: bool = Field(
        default=True, description="Check CRL for revocation status",
    )
    check_ct_log: bool = Field(
        default=False, description="Check Certificate Transparency logs",
    )
    expected_issuer: Optional[str] = Field(
        None, max_length=200,
        description="Expected certificate issuer for validation",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "document_reference": "s3://eudr-docs/rspo-cert-2026.pdf",
                    "check_ocsp": True,
                    "check_crl": True,
                }
            ]
        },
    )

class CertificateDetailSchema(GreenLangBase):
    """Details of a certificate in the chain."""

    subject: str = Field(..., description="Certificate subject")
    issuer: str = Field(..., description="Certificate issuer")
    serial_number: str = Field(..., description="Certificate serial number")
    not_before: datetime = Field(..., description="Validity start date")
    not_after: datetime = Field(..., description="Validity end date")
    key_type: str = Field(..., description="Key type (RSA/ECDSA)")
    key_size_bits: int = Field(..., description="Key size in bits")
    status: CertificateStatusSchema = Field(..., description="Certificate status")
    is_trusted_ca: bool = Field(
        default=False, description="Whether issuer is a trusted CA"
    )

    model_config = ConfigDict(from_attributes=True)

class CertificateResultSchema(GreenLangBase):
    """Response with certificate chain validation result."""

    validation_id: str = Field(..., description="Unique validation identifier")
    document_reference: str = Field(..., description="Document reference")
    chain_valid: bool = Field(..., description="Whether full chain is valid")
    chain_length: int = Field(..., ge=0, description="Number of certificates in chain")
    certificates: List[CertificateDetailSchema] = Field(
        default_factory=list, description="Certificate chain details"
    )
    root_ca_trusted: bool = Field(
        ..., description="Whether root CA is in trusted store"
    )
    ocsp_status: Optional[str] = Field(None, description="OCSP check result")
    crl_status: Optional[str] = Field(None, description="CRL check result")
    ct_log_found: Optional[bool] = Field(
        None, description="Whether certificate found in CT logs"
    )
    issues: List[str] = Field(
        default_factory=list, description="Validation issues found"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Validation timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class TrustedCASchema(GreenLangBase):
    """Trusted Certificate Authority entry."""

    ca_id: str = Field(..., description="Unique CA identifier")
    name: str = Field(..., description="CA display name")
    subject: str = Field(..., description="CA certificate subject")
    issuer: str = Field(..., description="CA certificate issuer")
    fingerprint_sha256: str = Field(..., description="SHA-256 fingerprint")
    not_before: Optional[datetime] = Field(None, description="Validity start")
    not_after: Optional[datetime] = Field(None, description="Validity end")
    active: bool = Field(default=True, description="Whether CA is active")
    added_by: Optional[str] = Field(None, description="User who added the CA")
    created_at: datetime = Field(
        default_factory=utcnow, description="Creation timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class AddTrustedCASchema(GreenLangBase):
    """Request to add a trusted CA."""

    name: str = Field(
        ..., min_length=1, max_length=200,
        description="CA display name",
    )
    certificate_pem: str = Field(
        ..., min_length=1,
        description="CA certificate in PEM format",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(extra="forbid")

class TrustedCAListSchema(GreenLangBase):
    """Response listing trusted certificate authorities."""

    cas: List[TrustedCASchema] = Field(
        default_factory=list, description="Trusted CAs"
    )
    total_count: int = Field(..., ge=0, description="Total CAs")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Metadata Schemas
# =============================================================================

class ExtractMetadataSchema(GreenLangBase):
    """Request to extract metadata from a document."""

    document_reference: str = Field(
        ..., min_length=1, max_length=500,
        description="Document reference (file path, S3 key, or URL)",
    )
    expected_author: Optional[str] = Field(
        None, max_length=200,
        description="Expected document author for validation",
    )
    validate_dates: bool = Field(
        default=True, description="Validate creation/modification dates",
    )
    flag_missing: bool = Field(
        default=True, description="Flag missing required metadata fields",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "document_reference": "s3://eudr-docs/coo-gh-2026-001.pdf",
                    "validate_dates": True,
                    "flag_missing": True,
                }
            ]
        },
    )

class MetadataFieldSchema(GreenLangBase):
    """Single metadata field extraction result."""

    field_name: str = Field(..., description="Metadata field name")
    value: Optional[str] = Field(None, description="Extracted value")
    present: bool = Field(..., description="Whether field is present in document")
    valid: bool = Field(default=True, description="Whether value passes validation")
    issues: List[str] = Field(
        default_factory=list, description="Validation issues"
    )

    model_config = ConfigDict(from_attributes=True)

class MetadataResultSchema(GreenLangBase):
    """Response with metadata extraction result."""

    document_id: str = Field(..., description="Document identifier")
    document_reference: str = Field(..., description="Document reference")
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    creator: Optional[str] = Field(None, description="Creator application")
    producer: Optional[str] = Field(None, description="PDF producer")
    creation_date: Optional[datetime] = Field(None, description="Creation date")
    modification_date: Optional[datetime] = Field(None, description="Last modification")
    page_count: Optional[int] = Field(None, ge=0, description="Number of pages")
    file_size_bytes: Optional[int] = Field(None, ge=0, description="File size")
    mime_type: Optional[str] = Field(None, description="MIME type")
    fields: List[MetadataFieldSchema] = Field(
        default_factory=list, description="Individual field extraction results"
    )
    missing_required: List[str] = Field(
        default_factory=list, description="Required fields that are missing"
    )
    date_consistency_valid: Optional[bool] = Field(
        None, description="Whether dates are internally consistent"
    )
    author_match: Optional[bool] = Field(
        None, description="Whether author matches expected operator"
    )
    issues: List[str] = Field(
        default_factory=list, description="All metadata issues found"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Extraction timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class ValidateMetadataSchema(GreenLangBase):
    """Request to validate metadata consistency for a document."""

    document_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Document identifier to validate",
    )
    expected_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Expected field values for cross-validation",
    )
    strict_mode: bool = Field(
        default=False,
        description="Whether to apply strict validation rules",
    )

    model_config = ConfigDict(extra="forbid")

class MetadataValidationResultSchema(GreenLangBase):
    """Response with metadata validation result."""

    document_id: str = Field(..., description="Document identifier")
    valid: bool = Field(..., description="Whether metadata passes validation")
    field_results: List[MetadataFieldSchema] = Field(
        default_factory=list, description="Per-field validation results"
    )
    issues: List[str] = Field(
        default_factory=list, description="Validation issues found"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Fraud Detection Schemas
# =============================================================================

class DetectFraudSchema(GreenLangBase):
    """Request to run fraud detection on a document."""

    document_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Document identifier to analyze",
    )
    document_reference: Optional[str] = Field(
        None, max_length=500,
        description="Document reference for content analysis",
    )
    operator_id: Optional[str] = Field(
        None, max_length=100,
        description="Operator who submitted the document",
    )
    patterns_to_check: List[FraudPatternTypeSchema] = Field(
        default_factory=list,
        description="Specific patterns to check (empty = all enabled)",
    )
    cross_reference_docs: List[str] = Field(
        default_factory=list,
        description="Related document IDs for cross-document checks",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "document_id": "doc-001",
                    "operator_id": "OP-GH-001",
                    "patterns_to_check": ["duplicate_reuse", "quantity_tampering"],
                }
            ]
        },
    )

class BatchDetectFraudSchema(GreenLangBase):
    """Request for batch fraud detection."""

    documents: List[DetectFraudSchema] = Field(
        ..., min_length=1, max_length=500,
        description="Documents to analyze (max 500)",
    )
    operator_id: Optional[str] = Field(
        None, max_length=100, description="Operator performing batch analysis",
    )

    model_config = ConfigDict(extra="forbid")

class FraudAlertSchema(GreenLangBase):
    """Single fraud alert detected in a document."""

    alert_id: str = Field(..., description="Unique alert identifier")
    document_id: str = Field(..., description="Affected document")
    pattern_type: FraudPatternTypeSchema = Field(..., description="Pattern type detected")
    severity: FraudSeveritySchema = Field(..., description="Alert severity")
    description: str = Field(..., description="Human-readable alert description")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence (0-1)"
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict, description="Supporting evidence data"
    )
    related_documents: List[str] = Field(
        default_factory=list, description="Related document IDs"
    )
    recommended_action: Optional[str] = Field(
        None, description="Recommended remediation action"
    )
    resolved: bool = Field(default=False, description="Whether alert is resolved")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    resolved_by: Optional[str] = Field(None, description="Resolved by user")
    created_at: datetime = Field(
        default_factory=utcnow, description="Detection timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class FraudDetectionResultSchema(GreenLangBase):
    """Response with fraud detection results for a document."""

    document_id: str = Field(..., description="Document identifier")
    overall_risk: FraudSeveritySchema = Field(
        ..., description="Overall risk assessment"
    )
    risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Composite risk score (0-100)"
    )
    alerts: List[FraudAlertSchema] = Field(
        default_factory=list, description="Individual fraud alerts"
    )
    patterns_checked: int = Field(
        ..., ge=0, description="Number of patterns checked"
    )
    patterns_triggered: int = Field(
        ..., ge=0, description="Number of patterns triggered"
    )
    authentication_result: AuthenticationResultSchema = Field(
        ..., description="Overall authentication result"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Detection timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class BatchFraudResultSchema(GreenLangBase):
    """Response for batch fraud detection."""

    total_submitted: int = Field(..., ge=0, description="Total documents submitted")
    total_analyzed: int = Field(..., ge=0, description="Successfully analyzed")
    total_failed: int = Field(..., ge=0, description="Failed analyses")
    total_alerts: int = Field(..., ge=0, description="Total alerts generated")
    results: List[FraudDetectionResultSchema] = Field(
        default_factory=list, description="Detection results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Analysis errors"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class FraudRuleSchema(GreenLangBase):
    """Fraud detection rule definition."""

    rule_id: str = Field(..., description="Unique rule identifier")
    pattern_type: FraudPatternTypeSchema = Field(..., description="Pattern type")
    name: str = Field(..., description="Rule display name")
    description: str = Field(..., description="Rule description")
    severity: FraudSeveritySchema = Field(..., description="Default severity")
    enabled: bool = Field(default=True, description="Whether rule is active")
    threshold: Optional[float] = Field(None, description="Detection threshold")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Rule-specific parameters"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Rule creation timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class FraudRuleListSchema(GreenLangBase):
    """Response listing active fraud detection rules."""

    rules: List[FraudRuleSchema] = Field(
        default_factory=list, description="Active fraud rules"
    )
    total_count: int = Field(..., ge=0, description="Total rules")
    enabled_count: int = Field(..., ge=0, description="Enabled rules")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class FraudSummarySchema(GreenLangBase):
    """Fraud alert summary for an operator."""

    operator_id: str = Field(..., description="Operator identifier")
    total_alerts: int = Field(..., ge=0, description="Total alerts")
    unresolved_alerts: int = Field(..., ge=0, description="Unresolved alerts")
    critical_count: int = Field(..., ge=0, description="Critical severity count")
    high_count: int = Field(..., ge=0, description="High severity count")
    medium_count: int = Field(..., ge=0, description="Medium severity count")
    low_count: int = Field(..., ge=0, description="Low severity count")
    average_risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Average risk score"
    )
    top_patterns: List[Dict[str, Any]] = Field(
        default_factory=list, description="Most frequent pattern types"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class FraudAlertListSchema(GreenLangBase):
    """Response listing fraud alerts for a document."""

    document_id: str = Field(..., description="Document identifier")
    alerts: List[FraudAlertSchema] = Field(
        default_factory=list, description="Fraud alerts"
    )
    total_count: int = Field(..., ge=0, description="Total alerts")
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Cross-Reference Schemas
# =============================================================================

class CrossRefVerifySchema(GreenLangBase):
    """Request to verify a document against an external registry."""

    document_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Document identifier to verify",
    )
    registry_type: RegistryTypeSchema = Field(
        ..., description="External registry to verify against",
    )
    certificate_number: Optional[str] = Field(
        None, max_length=200,
        description="Certificate or document number for lookup",
    )
    issuer_name: Optional[str] = Field(
        None, max_length=200,
        description="Expected issuer for cross-reference validation",
    )
    use_cache: bool = Field(
        default=True,
        description="Use cached registry results if available",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "document_id": "doc-001",
                    "registry_type": "fsc",
                    "certificate_number": "FSC-C012345",
                    "use_cache": True,
                }
            ]
        },
    )

class BatchCrossRefSchema(GreenLangBase):
    """Request for batch cross-reference verification."""

    verifications: List[CrossRefVerifySchema] = Field(
        ..., min_length=1, max_length=500,
        description="Verifications to perform (max 500)",
    )
    operator_id: Optional[str] = Field(
        None, max_length=100, description="Operator performing verification",
    )

    model_config = ConfigDict(extra="forbid")

class CrossRefResultSchema(GreenLangBase):
    """Response with cross-reference verification result."""

    verification_id: str = Field(..., description="Unique verification identifier")
    document_id: str = Field(..., description="Document identifier")
    registry_type: RegistryTypeSchema = Field(
        ..., description="Registry checked"
    )
    certificate_number: Optional[str] = Field(
        None, description="Certificate number checked"
    )
    registry_found: bool = Field(
        ..., description="Whether entry found in registry"
    )
    registry_status: Optional[str] = Field(
        None, description="Status in external registry (active/revoked/expired)"
    )
    registry_holder: Optional[str] = Field(
        None, description="Certificate holder name from registry"
    )
    registry_scope: Optional[str] = Field(
        None, description="Certificate scope from registry"
    )
    registry_valid_from: Optional[datetime] = Field(
        None, description="Validity start from registry"
    )
    registry_valid_to: Optional[datetime] = Field(
        None, description="Validity end from registry"
    )
    match_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Data match confidence (0-1)"
    )
    discrepancies: List[str] = Field(
        default_factory=list, description="Discrepancies found"
    )
    cached: bool = Field(
        default=False, description="Whether result was served from cache"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Verification timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class BatchCrossRefResultSchema(GreenLangBase):
    """Response for batch cross-reference verification."""

    total_submitted: int = Field(..., ge=0, description="Total verifications submitted")
    total_verified: int = Field(..., ge=0, description="Successfully verified")
    total_failed: int = Field(..., ge=0, description="Failed verifications")
    total_cache_hits: int = Field(..., ge=0, description="Results served from cache")
    results: List[CrossRefResultSchema] = Field(
        default_factory=list, description="Verification results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Verification errors"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class CacheStatsSchema(GreenLangBase):
    """Cross-reference cache statistics."""

    total_entries: int = Field(..., ge=0, description="Total cached entries")
    cache_hit_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Cache hit rate (0-1)"
    )
    entries_by_registry: Dict[str, int] = Field(
        default_factory=dict, description="Cached entries per registry"
    )
    oldest_entry_age_hours: Optional[float] = Field(
        None, description="Age of oldest cache entry in hours"
    )
    cache_ttl_hours: int = Field(..., description="Current cache TTL in hours")
    total_lookups: int = Field(..., ge=0, description="Total cache lookups")
    total_hits: int = Field(..., ge=0, description="Total cache hits")
    total_misses: int = Field(..., ge=0, description="Total cache misses")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Stats collection timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Report Schemas
# =============================================================================

class GenerateReportSchema(GreenLangBase):
    """Request to generate an authentication report."""

    document_ids: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="Document identifiers to include in report",
    )
    report_format: ReportFormatSchema = Field(
        default=ReportFormatSchema.JSON,
        description="Output format for the report",
    )
    include_classification: bool = Field(
        default=True, description="Include classification results"
    )
    include_signatures: bool = Field(
        default=True, description="Include signature verification results"
    )
    include_hashes: bool = Field(
        default=True, description="Include hash integrity results"
    )
    include_certificates: bool = Field(
        default=True, description="Include certificate validation results"
    )
    include_metadata: bool = Field(
        default=True, description="Include metadata extraction results"
    )
    include_fraud: bool = Field(
        default=True, description="Include fraud detection results"
    )
    include_crossref: bool = Field(
        default=True, description="Include cross-reference results"
    )
    operator_id: Optional[str] = Field(
        None, max_length=100, description="Operator generating the report",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "document_ids": ["doc-001", "doc-002", "doc-003"],
                    "report_format": "json",
                    "include_fraud": True,
                }
            ]
        },
    )

class EvidencePackageSchema(GreenLangBase):
    """Request to generate an evidence package for DDS submission."""

    dds_id: str = Field(
        ..., min_length=1, max_length=100,
        description="DDS package identifier",
    )
    document_ids: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="Document identifiers to include",
    )
    include_merkle_proof: bool = Field(
        default=True, description="Include Merkle tree proof"
    )
    include_signature_proofs: bool = Field(
        default=True, description="Include signature verification proofs"
    )
    include_hash_proofs: bool = Field(
        default=True, description="Include hash integrity proofs"
    )
    report_format: ReportFormatSchema = Field(
        default=ReportFormatSchema.PDF,
        description="Output format for the evidence package",
    )
    operator_id: Optional[str] = Field(
        None, max_length=100, description="Operator generating the package",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(extra="forbid")

class ReportResultSchema(GreenLangBase):
    """Response after generating an authentication report."""

    report_id: str = Field(..., description="Unique report identifier")
    report_format: ReportFormatSchema = Field(..., description="Output format")
    document_count: int = Field(
        ..., ge=0, description="Number of documents included"
    )
    overall_result: AuthenticationResultSchema = Field(
        ..., description="Overall authentication assessment"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Report summary statistics"
    )
    dds_id: Optional[str] = Field(None, description="Associated DDS package")
    is_evidence_package: bool = Field(
        default=False, description="Whether this is an evidence package"
    )
    file_reference: Optional[str] = Field(
        None, description="Storage reference for generated file"
    )
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="File size in bytes"
    )
    generated_at: datetime = Field(
        default_factory=utcnow, description="Generation timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None, description="Report expiration date"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class ReportDownloadSchema(GreenLangBase):
    """Response with report download information."""

    report_id: str = Field(..., description="Report identifier")
    report_format: ReportFormatSchema = Field(..., description="Format")
    file_reference: str = Field(
        ..., description="Storage reference for download"
    )
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    download_url: Optional[str] = Field(
        None, description="Pre-signed download URL"
    )
    expires_at: Optional[datetime] = Field(
        None, description="URL expiry time"
    )

    model_config = ConfigDict(from_attributes=True)

class DashboardSchema(GreenLangBase):
    """Authentication dashboard data for an operator."""

    operator_id: str = Field(..., description="Operator identifier")
    total_documents: int = Field(..., ge=0, description="Total documents processed")
    documents_by_type: Dict[str, int] = Field(
        default_factory=dict, description="Document count by type"
    )
    authentication_results: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by result (authentic/suspicious/fraudulent/inconclusive)",
    )
    total_fraud_alerts: int = Field(
        ..., ge=0, description="Total fraud alerts generated"
    )
    unresolved_alerts: int = Field(
        ..., ge=0, description="Unresolved fraud alerts"
    )
    average_risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Average risk score"
    )
    signature_stats: Dict[str, int] = Field(
        default_factory=dict, description="Signature verification status counts"
    )
    hash_integrity_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Hash integrity pass rate (0-1)"
    )
    crossref_match_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Cross-reference match rate (0-1)"
    )
    recent_activity: List[Dict[str, Any]] = Field(
        default_factory=list, description="Recent authentication activities"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Dashboard generation timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Batch Job Schemas
# =============================================================================

class SubmitBatchSchema(GreenLangBase):
    """Request to submit a batch processing job."""

    job_type: BatchJobTypeSchema = Field(
        ..., description="Type of batch job",
    )
    priority: int = Field(
        default=5, ge=1, le=10,
        description="Job priority (1=highest, 10=lowest)",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Job-specific parameters",
    )
    callback_url: Optional[str] = Field(
        None, max_length=500,
        description="Webhook URL for job completion notification",
    )

    model_config = ConfigDict(extra="forbid")

class BatchJobSchema(GreenLangBase):
    """Response for a batch processing job."""

    job_id: str = Field(..., description="Unique job identifier")
    job_type: BatchJobTypeSchema = Field(..., description="Job type")
    status: BatchJobStatusSchema = Field(..., description="Current status")
    priority: int = Field(..., ge=1, le=10, description="Job priority")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Job parameters"
    )
    progress_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage"
    )
    total_items: Optional[int] = Field(
        None, ge=0, description="Total items to process"
    )
    processed_items: Optional[int] = Field(
        None, ge=0, description="Items processed so far"
    )
    failed_items: Optional[int] = Field(
        None, ge=0, description="Items that failed"
    )
    result: Optional[Dict[str, Any]] = Field(
        None, description="Job result data"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    callback_url: Optional[str] = Field(
        None, description="Callback URL"
    )
    submitted_at: datetime = Field(..., description="Submission timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(
        None, description="Completion timestamp"
    )
    cancelled_at: Optional[datetime] = Field(
        None, description="Cancellation timestamp"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)

class BatchJobCancelSchema(GreenLangBase):
    """Response after cancelling a batch job."""

    job_id: str = Field(..., description="Job identifier")
    status: BatchJobStatusSchema = Field(..., description="New status")
    cancelled_at: datetime = Field(..., description="Cancellation timestamp")
    message: str = Field(..., description="Cancellation message")

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Health Schema
# =============================================================================

class HealthSchema(GreenLangBase):
    """Health check response for the document authentication service."""

    service: str = Field(
        default="gl-eudr-dav-012",
        description="Service identifier",
    )
    version: str = Field(
        default="1.0.0",
        description="Service version",
    )
    status: str = Field(
        default="healthy",
        description="Service status (healthy/degraded/unhealthy)",
    )
    database_connected: bool = Field(
        default=True, description="Database connectivity"
    )
    redis_connected: bool = Field(
        default=True, description="Redis connectivity"
    )
    documents_processed: int = Field(
        default=0, ge=0, description="Total documents processed"
    )
    active_fraud_rules: int = Field(
        default=15, ge=0, description="Number of active fraud rules"
    )
    trusted_cas_count: int = Field(
        default=8, ge=0, description="Number of trusted CAs"
    )
    unresolved_alerts: int = Field(
        default=0, ge=0, description="Number of unresolved fraud alerts"
    )
    uptime_seconds: float = Field(
        default=0.0, ge=0.0, description="Service uptime in seconds"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Health check timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Paginated Response Wrapper
# =============================================================================

class PaginatedResponse(GreenLangBase):
    """Generic paginated response wrapper."""

    data: List[Any] = Field(default_factory=list, description="Result items")
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # -- Helpers --
    "_utcnow",
    "_new_id",
    # -- Enumerations --
    "AuthenticationResultSchema",
    "BatchJobStatusSchema",
    "BatchJobTypeSchema",
    "CertificateStatusSchema",
    "ClassificationConfidenceSchema",
    "DocumentTypeSchema",
    "FraudPatternTypeSchema",
    "FraudSeveritySchema",
    "HashAlgorithmSchema",
    "RegistryTypeSchema",
    "ReportFormatSchema",
    "SignatureStandardSchema",
    "SignatureStatusSchema",
    "SortOrderSchema",
    "VerificationStatusSchema",
    # -- Common --
    "ErrorDetail",
    "ErrorResponse",
    "PaginatedMeta",
    "PaginatedResponse",
    "ProvenanceInfo",
    # -- Classification Request --
    "BatchClassifySchema",
    "ClassifyDocumentSchema",
    "RegisterTemplateSchema",
    # -- Classification Response --
    "BatchClassificationResultSchema",
    "ClassificationResultSchema",
    "TemplateListSchema",
    "TemplateSchema",
    # -- Signature Request --
    "BatchVerifySignatureSchema",
    "VerifySignatureSchema",
    # -- Signature Response --
    "BatchSignatureResultSchema",
    "SignatureHistorySchema",
    "SignatureResultSchema",
    # -- Hash Request --
    "ComputeHashSchema",
    "VerifyHashSchema",
    # -- Hash Response --
    "HashResultSchema",
    "MerkleNodeSchema",
    "MerkleTreeSchema",
    "RegistryLookupSchema",
    # -- Certificate Request --
    "AddTrustedCASchema",
    "ValidateCertificateSchema",
    # -- Certificate Response --
    "CertificateDetailSchema",
    "CertificateResultSchema",
    "TrustedCAListSchema",
    "TrustedCASchema",
    # -- Metadata Request --
    "ExtractMetadataSchema",
    "ValidateMetadataSchema",
    # -- Metadata Response --
    "MetadataFieldSchema",
    "MetadataResultSchema",
    "MetadataValidationResultSchema",
    # -- Fraud Request --
    "BatchDetectFraudSchema",
    "DetectFraudSchema",
    # -- Fraud Response --
    "BatchFraudResultSchema",
    "FraudAlertListSchema",
    "FraudAlertSchema",
    "FraudDetectionResultSchema",
    "FraudRuleListSchema",
    "FraudRuleSchema",
    "FraudSummarySchema",
    # -- CrossRef Request --
    "BatchCrossRefSchema",
    "CrossRefVerifySchema",
    # -- CrossRef Response --
    "BatchCrossRefResultSchema",
    "CacheStatsSchema",
    "CrossRefResultSchema",
    # -- Report Request --
    "EvidencePackageSchema",
    "GenerateReportSchema",
    # -- Report Response --
    "DashboardSchema",
    "ReportDownloadSchema",
    "ReportResultSchema",
    # -- Batch --
    "BatchJobCancelSchema",
    "BatchJobSchema",
    "SubmitBatchSchema",
    # -- Health --
    "HealthSchema",
]
