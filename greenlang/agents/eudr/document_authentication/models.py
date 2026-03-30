# -*- coding: utf-8 -*-
"""
Document Authentication Data Models - AGENT-EUDR-012

Pydantic v2 data models for the Document Authentication Agent covering
document classification, digital signature verification, hash integrity
validation, certificate chain validation, metadata extraction, fraud
pattern detection, cross-reference verification against external
registries, and compliance reporting with evidence packages.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all document authentication operations per
EU 2023/1115 Article 14 and eIDAS Regulation (EU) No 910/2014.

Enumerations (15):
    - DocumentType, ClassificationConfidence, SignatureStandard,
      SignatureStatus, HashAlgorithm, CertificateStatus, FraudSeverity,
      FraudPatternType, VerificationStatus, RegistryType, ReportFormat,
      MetadataField, DocumentLanguage, AuthenticationResult, BatchJobStatus

Core Models (9):
    - DocumentRecord, ClassificationResult, SignatureVerificationResult,
      HashRecord, CertificateChainResult, MetadataRecord, FraudAlert,
      CrossRefResult, AuthenticationReport

Request Models (15):
    - ClassifyDocumentRequest, BatchClassifyRequest,
      VerifySignatureRequest, ComputeHashRequest, VerifyHashRequest,
      ValidateCertificateRequest, ExtractMetadataRequest,
      DetectFraudRequest, CrossRefVerifyRequest,
      GenerateReportRequest, BatchVerificationRequest,
      RegisterTemplateRequest, AddTrustedCARequest,
      SearchDocumentsRequest, GetFraudAlertsRequest

Response Models (15):
    - ClassificationResponse, SignatureResponse, HashResponse,
      CertificateResponse, MetadataResponse, FraudDetectionResponse,
      CrossRefResponse, ReportResponse, BatchResponse,
      HealthResponse, DashboardResponse, TemplateResponse,
      TrustedCAResponse, DocumentSearchResponse, FraudAlertListResponse

Compatibility:
    Imports EUDRCommodity from greenlang.agents.data.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector and AGENT-EUDR-011 Mass Balance Calculator.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication (GL-EUDR-DAV-012)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    Field,
    field_validator,
    model_validator,
)
from greenlang.schemas import GreenLangBase, utcnow
from greenlang.schemas.enums import ReportFormat

# ---------------------------------------------------------------------------
# Cross-agent commodity import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.data.eudr_traceability.models import EUDRCommodity
except ImportError:
    EUDRCommodity = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: EUDR deforestation cutoff date (31 December 2020), per Article 2(1).
EUDR_DEFORESTATION_CUTOFF: str = "2020-12-31"

#: Maximum number of documents in a single batch processing job.
MAX_BATCH_SIZE: int = 500

#: EUDR Article 14 data retention in years.
EUDR_RETENTION_YEARS: int = 5

#: Supported hash algorithms for document integrity validation.
SUPPORTED_HASH_ALGORITHMS: List[str] = ["sha256", "sha512", "hmac_sha256"]

#: Supported digital signature standards.
SUPPORTED_SIGNATURE_STANDARDS: List[str] = [
    "cades", "pades", "xades", "jades", "qes", "pgp", "pkcs7",
]

#: Minimum RSA key size in bits (NIST SP 800-57 recommendation).
MIN_RSA_KEY_SIZE: int = 2048

#: Minimum ECDSA curve size in bits (NIST P-256).
MIN_ECDSA_KEY_SIZE: int = 256

# =============================================================================
# Enumerations
# =============================================================================

class DocumentType(str, Enum):
    """Type of EUDR supply chain document.

    Covers the 20 document types commonly encountered in EUDR due
    diligence processes across all seven regulated commodities.

    COO: Certificate of Origin - government-issued document certifying
        the country of origin of goods.
    PC: Phytosanitary Certificate - plant health certificate issued
        by the national plant protection organization (NPPO).
    BOL: Bill of Lading - transport document issued by the carrier.
    CDE: Customs Declaration Export - export customs declaration
        confirming goods have cleared export procedures.
    CDI: Customs Declaration Import - import customs declaration
        confirming goods have cleared import procedures.
    RSPO_CERT: RSPO Sustainability Certificate for palm oil products.
    FSC_CERT: FSC Chain of Custody Certificate for wood products.
    ISCC_CERT: ISCC Sustainability Certificate for biofuels and
        biomass products.
    FT_CERT: Fairtrade Certificate for certified commodities.
    UTZ_CERT: UTZ/Rainforest Alliance Certificate for agricultural
        commodities.
    LTR: Land Title Record - document proving land ownership or
        concession rights in the production country.
    LTD: Land Tenure Document - document establishing land use rights
        for the production area.
    FMP: Forest Management Plan - approved plan for sustainable
        forest management operations.
    FC: Felling Certificate - authorization for timber harvesting
        issued by the national forestry authority.
    WQC: Wood Quality Certificate - certificate attesting to the
        quality and species identification of wood products.
    DDS_DRAFT: Due Diligence Statement Draft - preliminary DDS before
        submission to the EU Information System.
    SSD: Supplier Self-Declaration - self-declaration from the
        supplier regarding deforestation-free production.
    IC: Invoice Commercial - commercial invoice accompanying the
        shipment of goods.
    TC: Transit Certificate - document for goods in transit through
        intermediate countries.
    WR: Weighbridge Receipt - weight measurement certificate at
        the point of loading or unloading.
    """

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

class ClassificationConfidence(str, Enum):
    """Confidence level of document classification.

    HIGH: Classification confidence >= 0.95 (configurable). High
        certainty that the document type is correctly identified.
        No manual review required.
    MEDIUM: Classification confidence >= 0.70 but < 0.95
        (configurable). Moderate certainty; manual review recommended.
    LOW: Classification confidence < 0.70. Low certainty; manual
        review required before proceeding.
    UNKNOWN: Document type could not be determined. Requires manual
        classification by an operator.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class SignatureStandard(str, Enum):
    """Digital signature standard used to sign the document.

    CADES: CMS Advanced Electronic Signatures (ETSI EN 319 122).
        Binary signature format for arbitrary data.
    PADES: PDF Advanced Electronic Signatures (ETSI EN 319 142).
        Embedded signature within PDF documents.
    XADES: XML Advanced Electronic Signatures (ETSI EN 319 132).
        XML-based signature format.
    JADES: JSON Advanced Electronic Signatures (ETSI TS 119 182).
        JSON-based signature format for web applications.
    QES: Qualified Electronic Signature per eIDAS Article 3(12).
        Highest legal standing in the EU.
    PGP: Pretty Good Privacy signature (RFC 4880).
        Commonly used for document exchange.
    PKCS7: PKCS#7 / CMS signature (RFC 5652).
        Legacy signature format still in widespread use.
    """

    CADES = "cades"
    PADES = "pades"
    XADES = "xades"
    JADES = "jades"
    QES = "qes"
    PGP = "pgp"
    PKCS7 = "pkcs7"

class SignatureStatus(str, Enum):
    """Verification status of a digital signature.

    VALID: Signature is cryptographically valid, the signing
        certificate is trusted, and the document has not been modified.
    INVALID: Signature verification failed. The document may have
        been tampered with after signing.
    EXPIRED: The signing certificate has expired. The signature was
        valid at signing time but the certificate is no longer current.
    REVOKED: The signing certificate has been revoked by the issuing
        CA. The signature should not be trusted.
    NO_SIGNATURE: The document does not contain a digital signature.
    UNKNOWN_SIGNER: The signing certificate is not in the trusted
        certificate store. Cannot verify issuer chain.
    STRIPPED: A signature was expected but appears to have been
        removed from the document (signature field present but empty).
    """

    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    NO_SIGNATURE = "no_signature"
    UNKNOWN_SIGNER = "unknown_signer"
    STRIPPED = "stripped"

class HashAlgorithm(str, Enum):
    """Cryptographic hash algorithm for document integrity.

    SHA256: SHA-256 (256-bit). Primary algorithm for EUDR document
        hashing. NIST FIPS 180-4 compliant.
    SHA512: SHA-512 (512-bit). Secondary algorithm for dual-hash
        verification providing defense-in-depth.
    HMAC_SHA256: HMAC-SHA-256 keyed hash. Used when message
        authentication is required in addition to integrity.
    """

    SHA256 = "sha256"
    SHA512 = "sha512"
    HMAC_SHA256 = "hmac_sha256"

class CertificateStatus(str, Enum):
    """Validation status of a signing certificate.

    VALID: Certificate is valid, trusted, and within its validity
        period. Key size meets minimum requirements.
    EXPIRED: Certificate has passed its notAfter date. Was valid
        when issued but is no longer current.
    REVOKED: Certificate has been revoked by the issuing CA via
        OCSP or CRL.
    SELF_SIGNED: Certificate is self-signed (issuer == subject).
        Not trusted unless explicitly allowed in configuration.
    WEAK_KEY: Certificate uses a key size below the configured
        minimum (RSA < 2048, ECDSA < 256).
    UNKNOWN: Certificate status could not be determined. OCSP
        responder or CRL may be unavailable.
    """

    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SELF_SIGNED = "self_signed"
    WEAK_KEY = "weak_key"
    UNKNOWN = "unknown"

class FraudSeverity(str, Enum):
    """Severity level of a detected fraud pattern.

    LOW: Minor anomaly that may be a data quality issue. Logged
        but does not block processing. Example: slightly unusual
        formatting in a certificate number.
    MEDIUM: Moderate anomaly suggesting potential irregularity.
        Flagged for manual review. Example: creation date predates
        the issuing authority's existence.
    HIGH: Significant anomaly strongly suggesting fraud or
        tampering. Processing blocked until manually resolved.
        Example: duplicate certificate number with different content.
    CRITICAL: Confirmed or near-certain fraud indicator. Immediate
        escalation required. Example: forged digital signature or
        revoked signing certificate used after revocation date.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FraudPatternType(str, Enum):
    """Type of fraud pattern detected in document analysis.

    DUPLICATE_REUSE: Same document hash or certificate number used
        across multiple distinct shipments or declarations.
    QUANTITY_TAMPERING: Quantity values in the document deviate
        significantly from cross-referenced records.
    DATE_MANIPULATION: Dates in the document are inconsistent with
        known issuance timelines or cross-referenced records.
    EXPIRED_CERT: Document references a certificate that was expired
        at the time the document claims to have been issued.
    SERIAL_ANOMALY: Certificate or document serial number does not
        follow the expected pattern of the issuing authority.
    ISSUER_MISMATCH: Document claims to be issued by an authority
        that does not match the signing certificate.
    TEMPLATE_FORGERY: Document layout, fonts, or formatting do not
        match the known template of the issuing authority.
    CROSS_DOC_INCONSISTENCY: Data in this document contradicts data
        in other documents for the same shipment or transaction.
    GEO_IMPOSSIBILITY: Geographic data in the document implies a
        physically impossible production or transport scenario.
    VELOCITY_ANOMALY: Issuer produced an unusually high number of
        documents in a short time period.
    MODIFICATION_ANOMALY: Document metadata indicates modifications
        after the purported date of issuance.
    ROUND_NUMBER_BIAS: An unusually high proportion of quantity or
        weight values are suspiciously round numbers.
    COPY_PASTE: Sections of the document appear to be copied from
        another document (text or image duplication detection).
    MISSING_REQUIRED: Required fields or sections expected for this
        document type are missing.
    SCOPE_MISMATCH: The certification scope does not cover the
        commodity or geographic region claimed in the document.
    """

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

class VerificationStatus(str, Enum):
    """Overall status of a document verification process.

    PENDING: Verification request has been received but processing
        has not yet started.
    IN_PROGRESS: Verification is actively being performed across
        one or more verification engines.
    COMPLETED: All verification steps have finished. Results are
        available in the response.
    FAILED: Verification process encountered an unrecoverable error.
        The document could not be fully verified.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class RegistryType(str, Enum):
    """External registry type for cross-reference verification.

    FSC: Forest Stewardship Council certificate database.
        Verifies FSC chain-of-custody certificate validity.
    RSPO: Roundtable on Sustainable Palm Oil member and certificate
        registry. Verifies RSPO certification status.
    ISCC: International Sustainability & Carbon Certification
        registry. Verifies ISCC certificate validity.
    FAIRTRADE: Fairtrade International certification database.
        Verifies Fairtrade certification status.
    UTZ_RA: UTZ / Rainforest Alliance certification database.
        Verifies UTZ/RA certification status.
    IPPC: International Plant Protection Convention ePhyto system.
        Verifies phytosanitary certificate authenticity.
    NATIONAL_CUSTOMS: National customs authority database for
        verifying customs declarations and transit documents.
    """

    FSC = "fsc"
    RSPO = "rspo"
    ISCC = "iscc"
    FAIRTRADE = "fairtrade"
    UTZ_RA = "utz_ra"
    IPPC = "ippc"
    NATIONAL_CUSTOMS = "national_customs"

class MetadataField(str, Enum):
    """Standard metadata fields extracted from documents.

    TITLE: Document title.
    AUTHOR: Document author or creator name.
    CREATOR: Application used to create the document.
    PRODUCER: Application used to produce the PDF/output.
    CREATION_DATE: Date and time the document was originally created.
    MODIFICATION_DATE: Date and time the document was last modified.
    KEYWORDS: Keywords or tags embedded in the document.
    GPS_LAT: GPS latitude extracted from embedded EXIF/XMP data.
    GPS_LON: GPS longitude extracted from embedded EXIF/XMP data.
    """

    TITLE = "title"
    AUTHOR = "author"
    CREATOR = "creator"
    PRODUCER = "producer"
    CREATION_DATE = "creation_date"
    MODIFICATION_DATE = "modification_date"
    KEYWORDS = "keywords"
    GPS_LAT = "gps_lat"
    GPS_LON = "gps_lon"

class DocumentLanguage(str, Enum):
    """Language of the document content.

    EN: English.
    FR: French.
    DE: German.
    ES: Spanish.
    PT: Portuguese.
    ID: Indonesian (Bahasa Indonesia).
    NL: Dutch.
    """

    EN = "en"
    FR = "fr"
    DE = "de"
    ES = "es"
    PT = "pt"
    ID = "id"
    NL = "nl"

class AuthenticationResult(str, Enum):
    """Overall authentication verdict for a document.

    AUTHENTIC: All verification checks passed. The document is
        considered genuine and unmodified.
    SUSPICIOUS: One or more verification checks raised concerns.
        Manual review recommended before accepting.
    FRAUDULENT: One or more verification checks indicate fraud or
        forgery. The document should be rejected.
    INCONCLUSIVE: Verification checks could not reach a definitive
        conclusion due to missing data or external service failures.
    """

    AUTHENTIC = "authentic"
    SUSPICIOUS = "suspicious"
    FRAUDULENT = "fraudulent"
    INCONCLUSIVE = "inconclusive"

class BatchJobStatus(str, Enum):
    """Status of a batch verification job.

    PENDING: Batch job has been created but processing has not
        yet started.
    RUNNING: Batch job is currently processing documents.
    COMPLETED: All documents in the batch have been processed.
    FAILED: Batch job encountered an unrecoverable error.
    CANCELLED: Batch job was cancelled by the operator before
        completion.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# =============================================================================
# Core Models
# =============================================================================

class DocumentRecord(GreenLangBase):
    """A document record tracked by the authentication system.

    Represents a single document submitted for authentication,
    including its classification, integrity hashes, and overall
    authentication result. Each record is immutable once finalized
    and contributes to the SHA-256 provenance chain.

    Attributes:
        document_id: Unique identifier for this document record.
        file_name: Original file name of the uploaded document.
        file_size_bytes: File size in bytes.
        file_hash_sha256: SHA-256 hash of the raw file content.
        file_hash_sha512: Optional SHA-512 hash for dual verification.
        document_type: Classified document type.
        classification_confidence: Confidence level of the classification.
        language: Detected language of the document content.
        commodity: EUDR commodity associated with this document.
        supplier_id: Identifier of the supplier who provided the document.
        shipment_id: Identifier of the associated shipment/transaction.
        authentication_result: Overall authentication verdict.
        verification_status: Current verification processing status.
        fraud_score: Composite fraud risk score (0.0-100.0).
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash for audit trail.
        created_at: UTC timestamp when the record was created.
        updated_at: UTC timestamp when the record was last updated.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this document record",
    )
    file_name: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Original file name of the uploaded document",
    )
    file_size_bytes: int = Field(
        default=0,
        ge=0,
        description="File size in bytes",
    )
    file_hash_sha256: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of the raw file content",
    )
    file_hash_sha512: Optional[str] = Field(
        None,
        min_length=128,
        max_length=128,
        description="SHA-512 hash for dual verification",
    )
    document_type: Optional[DocumentType] = Field(
        None,
        description="Classified document type",
    )
    classification_confidence: ClassificationConfidence = Field(
        default=ClassificationConfidence.UNKNOWN,
        description="Confidence level of the classification",
    )
    language: Optional[DocumentLanguage] = Field(
        None,
        description="Detected language of the document content",
    )
    commodity: Optional[str] = Field(
        None,
        description="EUDR commodity associated with this document",
    )
    supplier_id: Optional[str] = Field(
        None,
        description="Identifier of the supplier who provided the document",
    )
    shipment_id: Optional[str] = Field(
        None,
        description="Identifier of the associated shipment/transaction",
    )
    authentication_result: AuthenticationResult = Field(
        default=AuthenticationResult.INCONCLUSIVE,
        description="Overall authentication verdict",
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.PENDING,
        description="Current verification processing status",
    )
    fraud_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Composite fraud risk score (0.0-100.0)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the record was last updated",
    )

class ClassificationResult(GreenLangBase):
    """Result of document type classification.

    Contains the predicted document type, confidence score, and
    alternative classifications with their respective scores for
    manual review when confidence is below the HIGH threshold.

    Attributes:
        document_id: Reference to the classified document.
        predicted_type: Predicted document type.
        confidence_score: Classification confidence score (0.0-1.0).
        confidence_level: Confidence level category.
        alternative_types: Alternative classifications with scores.
        features_extracted: Features used for classification.
        processing_time_ms: Classification processing time in ms.
        provenance_hash: SHA-256 provenance hash for audit trail.
        classified_at: UTC timestamp of classification.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the classified document",
    )
    predicted_type: DocumentType = Field(
        ...,
        description="Predicted document type",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification confidence score (0.0-1.0)",
    )
    confidence_level: ClassificationConfidence = Field(
        ...,
        description="Confidence level category",
    )
    alternative_types: Dict[str, float] = Field(
        default_factory=dict,
        description="Alternative classifications with scores",
    )
    features_extracted: List[str] = Field(
        default_factory=list,
        description="Features used for classification",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Classification processing time in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )
    classified_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of classification",
    )

class SignatureVerificationResult(GreenLangBase):
    """Result of digital signature verification.

    Contains the verification status, signer identity, signing
    timestamp, and certificate chain details for audit trail purposes.

    Attributes:
        document_id: Reference to the verified document.
        signature_status: Verification status of the signature.
        signature_standard: Digital signature standard detected.
        signer_common_name: Common name of the signer from the cert.
        signer_organization: Organization of the signer.
        signer_country: Country code of the signer.
        signing_timestamp: Timestamp embedded in the signature.
        timestamp_verified: Whether the signing timestamp is verified.
        certificate_serial: Serial number of the signing certificate.
        certificate_issuer: Issuer of the signing certificate.
        certificate_valid_from: Certificate validity start date.
        certificate_valid_to: Certificate validity end date.
        key_algorithm: Key algorithm (RSA, ECDSA, EdDSA).
        key_size_bits: Key size in bits.
        hash_algorithm_used: Hash algorithm used in the signature.
        processing_time_ms: Verification processing time in ms.
        provenance_hash: SHA-256 provenance hash for audit trail.
        verified_at: UTC timestamp of verification.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the verified document",
    )
    signature_status: SignatureStatus = Field(
        ...,
        description="Verification status of the signature",
    )
    signature_standard: Optional[SignatureStandard] = Field(
        None,
        description="Digital signature standard detected",
    )
    signer_common_name: Optional[str] = Field(
        None,
        description="Common name of the signer from the certificate",
    )
    signer_organization: Optional[str] = Field(
        None,
        description="Organization of the signer",
    )
    signer_country: Optional[str] = Field(
        None,
        max_length=3,
        description="Country code of the signer (ISO 3166-1 alpha-2)",
    )
    signing_timestamp: Optional[datetime] = Field(
        None,
        description="Timestamp embedded in the signature",
    )
    timestamp_verified: bool = Field(
        default=False,
        description="Whether the signing timestamp is verified by a TSA",
    )
    certificate_serial: Optional[str] = Field(
        None,
        description="Serial number of the signing certificate",
    )
    certificate_issuer: Optional[str] = Field(
        None,
        description="Issuer of the signing certificate",
    )
    certificate_valid_from: Optional[datetime] = Field(
        None,
        description="Certificate validity start date",
    )
    certificate_valid_to: Optional[datetime] = Field(
        None,
        description="Certificate validity end date",
    )
    key_algorithm: Optional[str] = Field(
        None,
        description="Key algorithm (RSA, ECDSA, EdDSA)",
    )
    key_size_bits: Optional[int] = Field(
        None,
        ge=0,
        description="Key size in bits",
    )
    hash_algorithm_used: Optional[str] = Field(
        None,
        description="Hash algorithm used in the signature",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Verification processing time in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )
    verified_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of verification",
    )

class HashRecord(GreenLangBase):
    """Hash integrity record for a document.

    Stores the primary and secondary hashes along with registry
    lookup results for duplicate detection.

    Attributes:
        hash_id: Unique identifier for this hash record.
        document_id: Reference to the hashed document.
        algorithm: Hash algorithm used.
        hash_value: Computed hash value (hex-encoded).
        secondary_algorithm: Secondary hash algorithm if computed.
        secondary_hash_value: Secondary hash value if computed.
        is_duplicate: Whether an existing document with the same hash
            was found in the registry.
        duplicate_document_id: ID of the duplicate document if found.
        registry_expires_at: When this hash registry entry expires.
        provenance_hash: SHA-256 provenance hash for audit trail.
        computed_at: UTC timestamp when the hash was computed.
    """

    model_config = ConfigDict(from_attributes=True)

    hash_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this hash record",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the hashed document",
    )
    algorithm: HashAlgorithm = Field(
        default=HashAlgorithm.SHA256,
        description="Hash algorithm used",
    )
    hash_value: str = Field(
        ...,
        min_length=1,
        description="Computed hash value (hex-encoded)",
    )
    secondary_algorithm: Optional[HashAlgorithm] = Field(
        None,
        description="Secondary hash algorithm if computed",
    )
    secondary_hash_value: Optional[str] = Field(
        None,
        description="Secondary hash value if computed",
    )
    is_duplicate: bool = Field(
        default=False,
        description="Whether a duplicate hash was found in the registry",
    )
    duplicate_document_id: Optional[str] = Field(
        None,
        description="ID of the duplicate document if found",
    )
    registry_expires_at: Optional[datetime] = Field(
        None,
        description="When this hash registry entry expires",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )
    computed_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the hash was computed",
    )

class CertificateChainResult(GreenLangBase):
    """Result of certificate chain validation.

    Contains the validation status of each certificate in the chain
    from the leaf (signing) certificate up to the root CA, including
    OCSP and CRL check results.

    Attributes:
        document_id: Reference to the document whose cert chain was
            validated.
        chain_valid: Whether the entire certificate chain is valid.
        chain_depth: Number of certificates in the chain.
        leaf_status: Status of the leaf (signing) certificate.
        root_ca_name: Name of the root CA in the chain.
        root_ca_trusted: Whether the root CA is in the trusted store.
        ocsp_checked: Whether OCSP was checked.
        ocsp_status: OCSP response status.
        crl_checked: Whether CRL was checked.
        crl_status: CRL check status.
        ct_log_verified: Whether certificate transparency was verified.
        chain_certificates: Details of each certificate in the chain.
        weak_links: Descriptions of weak links in the chain.
        processing_time_ms: Validation processing time in ms.
        provenance_hash: SHA-256 provenance hash for audit trail.
        validated_at: UTC timestamp of validation.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the validated document",
    )
    chain_valid: bool = Field(
        ...,
        description="Whether the entire certificate chain is valid",
    )
    chain_depth: int = Field(
        default=0,
        ge=0,
        description="Number of certificates in the chain",
    )
    leaf_status: CertificateStatus = Field(
        ...,
        description="Status of the leaf (signing) certificate",
    )
    root_ca_name: Optional[str] = Field(
        None,
        description="Name of the root CA in the chain",
    )
    root_ca_trusted: bool = Field(
        default=False,
        description="Whether the root CA is in the trusted store",
    )
    ocsp_checked: bool = Field(
        default=False,
        description="Whether OCSP was checked",
    )
    ocsp_status: Optional[str] = Field(
        None,
        description="OCSP response status (good, revoked, unknown)",
    )
    crl_checked: bool = Field(
        default=False,
        description="Whether CRL was checked",
    )
    crl_status: Optional[str] = Field(
        None,
        description="CRL check status",
    )
    ct_log_verified: bool = Field(
        default=False,
        description="Whether certificate transparency was verified",
    )
    chain_certificates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Details of each certificate in the chain",
    )
    weak_links: List[str] = Field(
        default_factory=list,
        description="Descriptions of weak links in the chain",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Validation processing time in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )
    validated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of validation",
    )

class MetadataRecord(GreenLangBase):
    """Extracted and analyzed metadata from a document.

    Contains all metadata fields extracted from the document along
    with anomaly flags for fields that appear suspicious or
    inconsistent.

    Attributes:
        document_id: Reference to the document.
        title: Document title.
        author: Document author.
        creator: Application used to create the document.
        producer: Application used to produce the PDF.
        creation_date: Document creation date.
        modification_date: Document last modification date.
        keywords: Keywords embedded in the document.
        gps_lat: GPS latitude from embedded geolocation data.
        gps_lon: GPS longitude from embedded geolocation data.
        page_count: Number of pages in the document.
        file_format: Detected file format (pdf, docx, tiff, etc.).
        raw_metadata: Complete raw metadata dictionary.
        anomalies: List of metadata anomaly descriptions.
        missing_fields: Required metadata fields that are missing.
        creation_date_anomaly: Whether creation date is anomalous.
        author_match: Whether author matches the submitting operator.
        provenance_hash: SHA-256 provenance hash for audit trail.
        extracted_at: UTC timestamp of extraction.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the document",
    )
    title: Optional[str] = Field(
        None,
        description="Document title",
    )
    author: Optional[str] = Field(
        None,
        description="Document author",
    )
    creator: Optional[str] = Field(
        None,
        description="Application used to create the document",
    )
    producer: Optional[str] = Field(
        None,
        description="Application used to produce the PDF",
    )
    creation_date: Optional[datetime] = Field(
        None,
        description="Document creation date",
    )
    modification_date: Optional[datetime] = Field(
        None,
        description="Document last modification date",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords embedded in the document",
    )
    gps_lat: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="GPS latitude from embedded geolocation data",
    )
    gps_lon: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="GPS longitude from embedded geolocation data",
    )
    page_count: int = Field(
        default=0,
        ge=0,
        description="Number of pages in the document",
    )
    file_format: Optional[str] = Field(
        None,
        description="Detected file format (pdf, docx, tiff, etc.)",
    )
    raw_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Complete raw metadata dictionary",
    )
    anomalies: List[str] = Field(
        default_factory=list,
        description="List of metadata anomaly descriptions",
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Required metadata fields that are missing",
    )
    creation_date_anomaly: bool = Field(
        default=False,
        description="Whether creation date is anomalous",
    )
    author_match: Optional[bool] = Field(
        None,
        description="Whether author matches the submitting operator",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )
    extracted_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of extraction",
    )

class FraudAlert(GreenLangBase):
    """A fraud pattern detection alert for a document.

    Represents a single detected fraud pattern with severity,
    confidence, and evidence details for investigation.

    Attributes:
        alert_id: Unique identifier for this fraud alert.
        document_id: Reference to the flagged document.
        pattern_type: Type of fraud pattern detected.
        severity: Severity level of the alert.
        confidence_score: Detection confidence (0.0-1.0).
        description: Human-readable description of the finding.
        evidence: Evidence supporting the fraud detection.
        related_document_ids: IDs of related documents involved.
        recommended_action: Recommended action for the operator.
        resolved: Whether this alert has been resolved.
        resolved_by: Operator who resolved the alert.
        resolved_at: UTC timestamp of resolution.
        resolution_notes: Notes on how the alert was resolved.
        provenance_hash: SHA-256 provenance hash for audit trail.
        detected_at: UTC timestamp of detection.
    """

    model_config = ConfigDict(from_attributes=True)

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this fraud alert",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the flagged document",
    )
    pattern_type: FraudPatternType = Field(
        ...,
        description="Type of fraud pattern detected",
    )
    severity: FraudSeverity = Field(
        ...,
        description="Severity level of the alert",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence (0.0-1.0)",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of the finding",
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Evidence supporting the fraud detection",
    )
    related_document_ids: List[str] = Field(
        default_factory=list,
        description="IDs of related documents involved",
    )
    recommended_action: Optional[str] = Field(
        None,
        description="Recommended action for the operator",
    )
    resolved: bool = Field(
        default=False,
        description="Whether this alert has been resolved",
    )
    resolved_by: Optional[str] = Field(
        None,
        description="Operator who resolved the alert",
    )
    resolved_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp of resolution",
    )
    resolution_notes: Optional[str] = Field(
        None,
        description="Notes on how the alert was resolved",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )
    detected_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of detection",
    )

class CrossRefResult(GreenLangBase):
    """Result of cross-reference verification against an external registry.

    Contains the verification outcome from querying an external
    certification registry (FSC, RSPO, ISCC, etc.) to confirm
    the authenticity and validity of a certificate number.

    Attributes:
        crossref_id: Unique identifier for this cross-reference check.
        document_id: Reference to the verified document.
        registry_type: External registry queried.
        certificate_number: Certificate number looked up.
        registry_found: Whether the certificate was found in the registry.
        registry_status: Status reported by the registry.
        registry_holder_name: Certificate holder name from the registry.
        registry_valid_from: Certificate validity start per the registry.
        registry_valid_to: Certificate validity end per the registry.
        registry_scope: Certification scope from the registry.
        name_match: Whether the holder name matches the document.
        date_match: Whether validity dates match the document.
        scope_match: Whether the certification scope matches.
        discrepancies: List of discrepancies found.
        cached: Whether the result was served from cache.
        processing_time_ms: Cross-reference processing time in ms.
        provenance_hash: SHA-256 provenance hash for audit trail.
        verified_at: UTC timestamp of verification.
    """

    model_config = ConfigDict(from_attributes=True)

    crossref_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this cross-reference check",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the verified document",
    )
    registry_type: RegistryType = Field(
        ...,
        description="External registry queried",
    )
    certificate_number: str = Field(
        ...,
        min_length=1,
        description="Certificate number looked up",
    )
    registry_found: bool = Field(
        ...,
        description="Whether the certificate was found in the registry",
    )
    registry_status: Optional[str] = Field(
        None,
        description="Status reported by the registry",
    )
    registry_holder_name: Optional[str] = Field(
        None,
        description="Certificate holder name from the registry",
    )
    registry_valid_from: Optional[datetime] = Field(
        None,
        description="Certificate validity start per the registry",
    )
    registry_valid_to: Optional[datetime] = Field(
        None,
        description="Certificate validity end per the registry",
    )
    registry_scope: Optional[str] = Field(
        None,
        description="Certification scope from the registry",
    )
    name_match: Optional[bool] = Field(
        None,
        description="Whether the holder name matches the document",
    )
    date_match: Optional[bool] = Field(
        None,
        description="Whether validity dates match the document",
    )
    scope_match: Optional[bool] = Field(
        None,
        description="Whether the certification scope matches",
    )
    discrepancies: List[str] = Field(
        default_factory=list,
        description="List of discrepancies found",
    )
    cached: bool = Field(
        default=False,
        description="Whether the result was served from cache",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Cross-reference processing time in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )
    verified_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of verification",
    )

class AuthenticationReport(GreenLangBase):
    """Comprehensive authentication report for a document.

    Aggregates all verification results (classification, signature,
    hash, certificate chain, metadata, fraud detection, cross-reference)
    into a single compliance report with an overall verdict.

    Attributes:
        report_id: Unique identifier for this report.
        document_id: Reference to the authenticated document.
        authentication_result: Overall authentication verdict.
        classification: Classification result.
        signature: Signature verification result.
        hash_integrity: Hash integrity record.
        certificate_chain: Certificate chain validation result.
        metadata_analysis: Metadata extraction and analysis result.
        fraud_alerts: List of fraud alerts detected.
        cross_references: List of cross-reference results.
        overall_score: Composite authentication score (0.0-100.0).
        report_format: Format of the generated report.
        evidence_package_url: URL to the evidence package if generated.
        report_url: URL to the generated report file.
        generated_by: Identifier of the operator or system that
            generated the report.
        provenance_hash: SHA-256 provenance hash for audit trail.
        generated_at: UTC timestamp of report generation.
        expires_at: UTC timestamp when the report expires.
    """

    model_config = ConfigDict(from_attributes=True)

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this report",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the authenticated document",
    )
    authentication_result: AuthenticationResult = Field(
        ...,
        description="Overall authentication verdict",
    )
    classification: Optional[ClassificationResult] = Field(
        None,
        description="Classification result",
    )
    signature: Optional[SignatureVerificationResult] = Field(
        None,
        description="Signature verification result",
    )
    hash_integrity: Optional[HashRecord] = Field(
        None,
        description="Hash integrity record",
    )
    certificate_chain: Optional[CertificateChainResult] = Field(
        None,
        description="Certificate chain validation result",
    )
    metadata_analysis: Optional[MetadataRecord] = Field(
        None,
        description="Metadata extraction and analysis result",
    )
    fraud_alerts: List[FraudAlert] = Field(
        default_factory=list,
        description="List of fraud alerts detected",
    )
    cross_references: List[CrossRefResult] = Field(
        default_factory=list,
        description="List of cross-reference results",
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Composite authentication score (0.0-100.0)",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Format of the generated report",
    )
    evidence_package_url: Optional[str] = Field(
        None,
        description="URL to the evidence package if generated",
    )
    report_url: Optional[str] = Field(
        None,
        description="URL to the generated report file",
    )
    generated_by: Optional[str] = Field(
        None,
        description="Identifier of the operator or system",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )
    generated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of report generation",
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when the report expires",
    )

# =============================================================================
# Request Models
# =============================================================================

class ClassifyDocumentRequest(GreenLangBase):
    """Request to classify a single document by type.

    Attributes:
        file_name: Original file name of the document.
        file_content_base64: Base64-encoded file content.
        file_hash_sha256: Pre-computed SHA-256 hash of the file.
        commodity: Optional EUDR commodity hint for classification.
        supplier_id: Optional supplier identifier for context.
        operator_id: Identifier of the requesting operator.
    """

    model_config = ConfigDict(from_attributes=True)

    file_name: str = Field(
        ..., min_length=1, max_length=512,
        description="Original file name of the document",
    )
    file_content_base64: str = Field(
        ..., min_length=1,
        description="Base64-encoded file content",
    )
    file_hash_sha256: str = Field(
        ..., min_length=64, max_length=64,
        description="Pre-computed SHA-256 hash of the file",
    )
    commodity: Optional[str] = Field(
        None, description="EUDR commodity hint for classification",
    )
    supplier_id: Optional[str] = Field(
        None, description="Supplier identifier for context",
    )
    operator_id: Optional[str] = Field(
        None, description="Identifier of the requesting operator",
    )

class BatchClassifyRequest(GreenLangBase):
    """Request to classify multiple documents in batch.

    Attributes:
        documents: List of individual classification requests.
        priority: Processing priority (normal, high).
    """

    model_config = ConfigDict(from_attributes=True)

    documents: List[ClassifyDocumentRequest] = Field(
        ..., min_length=1,
        description="List of classification requests",
    )
    priority: str = Field(
        default="normal",
        description="Processing priority (normal, high)",
    )

    @field_validator("documents")
    @classmethod
    def validate_batch_size(
        cls, v: List[ClassifyDocumentRequest],
    ) -> List[ClassifyDocumentRequest]:
        """Validate batch does not exceed MAX_BATCH_SIZE."""
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum {MAX_BATCH_SIZE}"
            )
        return v

class VerifySignatureRequest(GreenLangBase):
    """Request to verify a document's digital signature.

    Attributes:
        document_id: Identifier of the document to verify.
        file_content_base64: Base64-encoded file content.
        expected_signer: Optional expected signer common name.
        require_timestamp: Whether to require a signed timestamp.
        operator_id: Identifier of the requesting operator.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        ..., min_length=1,
        description="Identifier of the document to verify",
    )
    file_content_base64: str = Field(
        ..., min_length=1,
        description="Base64-encoded file content",
    )
    expected_signer: Optional[str] = Field(
        None, description="Expected signer common name",
    )
    require_timestamp: Optional[bool] = Field(
        None, description="Override config require_timestamp setting",
    )
    operator_id: Optional[str] = Field(
        None, description="Identifier of the requesting operator",
    )

class ComputeHashRequest(GreenLangBase):
    """Request to compute integrity hashes for a document.

    Attributes:
        document_id: Identifier of the document.
        file_content_base64: Base64-encoded file content.
        algorithm: Primary hash algorithm to use.
        compute_secondary: Whether to also compute secondary hash.
        check_registry: Whether to check for duplicates in registry.
        operator_id: Identifier of the requesting operator.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        ..., min_length=1,
        description="Identifier of the document",
    )
    file_content_base64: str = Field(
        ..., min_length=1,
        description="Base64-encoded file content",
    )
    algorithm: HashAlgorithm = Field(
        default=HashAlgorithm.SHA256,
        description="Primary hash algorithm to use",
    )
    compute_secondary: bool = Field(
        default=True,
        description="Whether to also compute secondary hash",
    )
    check_registry: bool = Field(
        default=True,
        description="Whether to check for duplicates in registry",
    )
    operator_id: Optional[str] = Field(
        None, description="Identifier of the requesting operator",
    )

class VerifyHashRequest(GreenLangBase):
    """Request to verify a document against a known hash.

    Attributes:
        document_id: Identifier of the document.
        file_content_base64: Base64-encoded file content.
        expected_hash: Expected hash value to verify against.
        algorithm: Hash algorithm of the expected hash.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        ..., min_length=1,
        description="Identifier of the document",
    )
    file_content_base64: str = Field(
        ..., min_length=1,
        description="Base64-encoded file content",
    )
    expected_hash: str = Field(
        ..., min_length=1,
        description="Expected hash value to verify against",
    )
    algorithm: HashAlgorithm = Field(
        default=HashAlgorithm.SHA256,
        description="Hash algorithm of the expected hash",
    )

class ValidateCertificateRequest(GreenLangBase):
    """Request to validate a document's signing certificate chain.

    Attributes:
        document_id: Identifier of the document.
        file_content_base64: Base64-encoded file content.
        check_ocsp: Whether to perform OCSP checking.
        check_crl: Whether to perform CRL checking.
        check_ct: Whether to check certificate transparency logs.
        operator_id: Identifier of the requesting operator.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        ..., min_length=1,
        description="Identifier of the document",
    )
    file_content_base64: str = Field(
        ..., min_length=1,
        description="Base64-encoded file content",
    )
    check_ocsp: Optional[bool] = Field(
        None, description="Override config OCSP setting",
    )
    check_crl: Optional[bool] = Field(
        None, description="Whether to perform CRL checking",
    )
    check_ct: Optional[bool] = Field(
        None, description="Override config CT log setting",
    )
    operator_id: Optional[str] = Field(
        None, description="Identifier of the requesting operator",
    )

class ExtractMetadataRequest(GreenLangBase):
    """Request to extract and analyze document metadata.

    Attributes:
        document_id: Identifier of the document.
        file_content_base64: Base64-encoded file content.
        expected_author: Expected document author for matching.
        upload_date: Date when the document was uploaded for
            creation date tolerance checking.
        operator_id: Identifier of the requesting operator.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        ..., min_length=1,
        description="Identifier of the document",
    )
    file_content_base64: str = Field(
        ..., min_length=1,
        description="Base64-encoded file content",
    )
    expected_author: Optional[str] = Field(
        None, description="Expected document author for matching",
    )
    upload_date: Optional[datetime] = Field(
        None, description="Upload date for creation date tolerance",
    )
    operator_id: Optional[str] = Field(
        None, description="Identifier of the requesting operator",
    )

class DetectFraudRequest(GreenLangBase):
    """Request to run fraud pattern detection on a document.

    Attributes:
        document_id: Identifier of the document to analyze.
        document_type: Classified document type.
        file_content_base64: Base64-encoded file content.
        metadata: Document metadata for pattern analysis.
        related_document_ids: IDs of related documents for cross-check.
        commodity: EUDR commodity for scope validation.
        supplier_id: Supplier identifier for velocity analysis.
        patterns_to_check: Specific patterns to check (all if empty).
        operator_id: Identifier of the requesting operator.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        ..., min_length=1,
        description="Identifier of the document to analyze",
    )
    document_type: Optional[DocumentType] = Field(
        None, description="Classified document type",
    )
    file_content_base64: Optional[str] = Field(
        None, description="Base64-encoded file content",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata for pattern analysis",
    )
    related_document_ids: List[str] = Field(
        default_factory=list,
        description="IDs of related documents for cross-check",
    )
    commodity: Optional[str] = Field(
        None, description="EUDR commodity for scope validation",
    )
    supplier_id: Optional[str] = Field(
        None, description="Supplier identifier for velocity analysis",
    )
    patterns_to_check: List[FraudPatternType] = Field(
        default_factory=list,
        description="Specific patterns to check (all if empty)",
    )
    operator_id: Optional[str] = Field(
        None, description="Identifier of the requesting operator",
    )

class CrossRefVerifyRequest(GreenLangBase):
    """Request to cross-reference a certificate against external registry.

    Attributes:
        document_id: Identifier of the document.
        registry_type: Registry to query.
        certificate_number: Certificate number to look up.
        holder_name: Expected certificate holder name.
        valid_from: Expected validity start date.
        valid_to: Expected validity end date.
        commodity: EUDR commodity for scope matching.
        use_cache: Whether to use cached results.
        operator_id: Identifier of the requesting operator.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        ..., min_length=1,
        description="Identifier of the document",
    )
    registry_type: RegistryType = Field(
        ..., description="Registry to query",
    )
    certificate_number: str = Field(
        ..., min_length=1,
        description="Certificate number to look up",
    )
    holder_name: Optional[str] = Field(
        None, description="Expected certificate holder name",
    )
    valid_from: Optional[datetime] = Field(
        None, description="Expected validity start date",
    )
    valid_to: Optional[datetime] = Field(
        None, description="Expected validity end date",
    )
    commodity: Optional[str] = Field(
        None, description="EUDR commodity for scope matching",
    )
    use_cache: bool = Field(
        default=True, description="Whether to use cached results",
    )
    operator_id: Optional[str] = Field(
        None, description="Identifier of the requesting operator",
    )

class GenerateReportRequest(GreenLangBase):
    """Request to generate a comprehensive authentication report.

    Attributes:
        document_id: Identifier of the document.
        report_format: Desired output format.
        include_evidence_package: Whether to generate evidence package.
        include_classification: Include classification in the report.
        include_signature: Include signature verification in the report.
        include_hash: Include hash integrity in the report.
        include_certificate: Include certificate chain in the report.
        include_metadata: Include metadata analysis in the report.
        include_fraud: Include fraud detection in the report.
        include_crossref: Include cross-reference results in the report.
        operator_id: Identifier of the requesting operator.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        ..., min_length=1,
        description="Identifier of the document",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Desired output format",
    )
    include_evidence_package: bool = Field(
        default=True,
        description="Whether to generate evidence package",
    )
    include_classification: bool = Field(
        default=True, description="Include classification",
    )
    include_signature: bool = Field(
        default=True, description="Include signature verification",
    )
    include_hash: bool = Field(
        default=True, description="Include hash integrity",
    )
    include_certificate: bool = Field(
        default=True, description="Include certificate chain",
    )
    include_metadata: bool = Field(
        default=True, description="Include metadata analysis",
    )
    include_fraud: bool = Field(
        default=True, description="Include fraud detection",
    )
    include_crossref: bool = Field(
        default=True, description="Include cross-reference results",
    )
    operator_id: Optional[str] = Field(
        None, description="Identifier of the requesting operator",
    )

class BatchVerificationRequest(GreenLangBase):
    """Request to verify multiple documents in batch.

    Attributes:
        document_ids: List of document identifiers to verify.
        run_classification: Whether to run classification.
        run_signature: Whether to verify signatures.
        run_hash: Whether to compute and verify hashes.
        run_certificate: Whether to validate certificate chains.
        run_metadata: Whether to extract metadata.
        run_fraud: Whether to run fraud detection.
        run_crossref: Whether to run cross-reference checks.
        priority: Processing priority (normal, high).
        operator_id: Identifier of the requesting operator.
    """

    model_config = ConfigDict(from_attributes=True)

    document_ids: List[str] = Field(
        ..., min_length=1,
        description="List of document identifiers to verify",
    )
    run_classification: bool = Field(
        default=True, description="Whether to run classification",
    )
    run_signature: bool = Field(
        default=True, description="Whether to verify signatures",
    )
    run_hash: bool = Field(
        default=True, description="Whether to compute/verify hashes",
    )
    run_certificate: bool = Field(
        default=True, description="Whether to validate cert chains",
    )
    run_metadata: bool = Field(
        default=True, description="Whether to extract metadata",
    )
    run_fraud: bool = Field(
        default=True, description="Whether to run fraud detection",
    )
    run_crossref: bool = Field(
        default=True, description="Whether to run cross-ref checks",
    )
    priority: str = Field(
        default="normal",
        description="Processing priority (normal, high)",
    )
    operator_id: Optional[str] = Field(
        None, description="Identifier of the requesting operator",
    )

    @field_validator("document_ids")
    @classmethod
    def validate_batch_size(cls, v: List[str]) -> List[str]:
        """Validate batch does not exceed MAX_BATCH_SIZE."""
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum {MAX_BATCH_SIZE}"
            )
        return v

class RegisterTemplateRequest(GreenLangBase):
    """Request to register a known document template for forgery detection.

    Attributes:
        template_name: Human-readable template name.
        document_type: Document type this template applies to.
        issuing_authority: Authority that issues this template.
        template_hash: SHA-256 hash of the template layout.
        template_features: Extracted features of the template.
        valid_from: Date from which this template is valid.
        valid_to: Optional expiry date for the template.
        operator_id: Identifier of the registering operator.
    """

    model_config = ConfigDict(from_attributes=True)

    template_name: str = Field(
        ..., min_length=1, max_length=256,
        description="Human-readable template name",
    )
    document_type: DocumentType = Field(
        ..., description="Document type this template applies to",
    )
    issuing_authority: str = Field(
        ..., min_length=1,
        description="Authority that issues this template",
    )
    template_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the template layout",
    )
    template_features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted features of the template",
    )
    valid_from: datetime = Field(
        ..., description="Date from which this template is valid",
    )
    valid_to: Optional[datetime] = Field(
        None, description="Optional expiry date for the template",
    )
    operator_id: Optional[str] = Field(
        None, description="Identifier of the registering operator",
    )

class AddTrustedCARequest(GreenLangBase):
    """Request to add a trusted certificate authority.

    Attributes:
        ca_name: Common name of the certificate authority.
        ca_certificate_pem: PEM-encoded CA certificate.
        ca_type: Type of CA (root, intermediate).
        valid_from: CA certificate validity start.
        valid_to: CA certificate validity end.
        operator_id: Identifier of the registering operator.
    """

    model_config = ConfigDict(from_attributes=True)

    ca_name: str = Field(
        ..., min_length=1, max_length=256,
        description="Common name of the certificate authority",
    )
    ca_certificate_pem: str = Field(
        ..., min_length=1,
        description="PEM-encoded CA certificate",
    )
    ca_type: str = Field(
        default="root",
        description="Type of CA (root, intermediate)",
    )
    valid_from: Optional[datetime] = Field(
        None, description="CA certificate validity start",
    )
    valid_to: Optional[datetime] = Field(
        None, description="CA certificate validity end",
    )
    operator_id: Optional[str] = Field(
        None, description="Identifier of the registering operator",
    )

class SearchDocumentsRequest(GreenLangBase):
    """Request to search authenticated documents.

    Attributes:
        document_type: Filter by document type.
        authentication_result: Filter by authentication result.
        commodity: Filter by EUDR commodity.
        supplier_id: Filter by supplier.
        date_from: Filter documents created after this date.
        date_to: Filter documents created before this date.
        min_fraud_score: Minimum fraud score filter.
        limit: Maximum number of results to return.
        offset: Pagination offset.
    """

    model_config = ConfigDict(from_attributes=True)

    document_type: Optional[DocumentType] = Field(
        None, description="Filter by document type",
    )
    authentication_result: Optional[AuthenticationResult] = Field(
        None, description="Filter by authentication result",
    )
    commodity: Optional[str] = Field(
        None, description="Filter by EUDR commodity",
    )
    supplier_id: Optional[str] = Field(
        None, description="Filter by supplier",
    )
    date_from: Optional[datetime] = Field(
        None, description="Filter documents created after this date",
    )
    date_to: Optional[datetime] = Field(
        None, description="Filter documents created before this date",
    )
    min_fraud_score: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Minimum fraud score filter",
    )
    limit: int = Field(
        default=50, ge=1, le=1000,
        description="Maximum number of results to return",
    )
    offset: int = Field(
        default=0, ge=0,
        description="Pagination offset",
    )

class GetFraudAlertsRequest(GreenLangBase):
    """Request to retrieve fraud alerts with filtering.

    Attributes:
        document_id: Filter by document.
        severity: Filter by severity level.
        pattern_type: Filter by fraud pattern type.
        resolved: Filter by resolution status.
        date_from: Filter alerts detected after this date.
        date_to: Filter alerts detected before this date.
        limit: Maximum number of results to return.
        offset: Pagination offset.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: Optional[str] = Field(
        None, description="Filter by document",
    )
    severity: Optional[FraudSeverity] = Field(
        None, description="Filter by severity level",
    )
    pattern_type: Optional[FraudPatternType] = Field(
        None, description="Filter by fraud pattern type",
    )
    resolved: Optional[bool] = Field(
        None, description="Filter by resolution status",
    )
    date_from: Optional[datetime] = Field(
        None, description="Filter alerts detected after this date",
    )
    date_to: Optional[datetime] = Field(
        None, description="Filter alerts detected before this date",
    )
    limit: int = Field(
        default=50, ge=1, le=1000,
        description="Maximum number of results to return",
    )
    offset: int = Field(
        default=0, ge=0,
        description="Pagination offset",
    )

# =============================================================================
# Response Models
# =============================================================================

class ClassificationResponse(GreenLangBase):
    """Response for document classification requests.

    Attributes:
        success: Whether the classification succeeded.
        result: Classification result.
        message: Optional message or error description.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(..., description="Whether classification succeeded")
    result: Optional[ClassificationResult] = Field(
        None, description="Classification result",
    )
    message: Optional[str] = Field(
        None, description="Optional message or error description",
    )

class SignatureResponse(GreenLangBase):
    """Response for signature verification requests.

    Attributes:
        success: Whether the verification succeeded.
        result: Signature verification result.
        message: Optional message or error description.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        ..., description="Whether verification succeeded",
    )
    result: Optional[SignatureVerificationResult] = Field(
        None, description="Signature verification result",
    )
    message: Optional[str] = Field(
        None, description="Optional message or error description",
    )

class HashResponse(GreenLangBase):
    """Response for hash computation and verification requests.

    Attributes:
        success: Whether the hash operation succeeded.
        result: Hash record with computed values.
        verified: Whether hash verification matched (for verify requests).
        message: Optional message or error description.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        ..., description="Whether hash operation succeeded",
    )
    result: Optional[HashRecord] = Field(
        None, description="Hash record with computed values",
    )
    verified: Optional[bool] = Field(
        None, description="Whether hash verification matched",
    )
    message: Optional[str] = Field(
        None, description="Optional message or error description",
    )

class CertificateResponse(GreenLangBase):
    """Response for certificate chain validation requests.

    Attributes:
        success: Whether the validation succeeded.
        result: Certificate chain validation result.
        message: Optional message or error description.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        ..., description="Whether validation succeeded",
    )
    result: Optional[CertificateChainResult] = Field(
        None, description="Certificate chain validation result",
    )
    message: Optional[str] = Field(
        None, description="Optional message or error description",
    )

class MetadataResponse(GreenLangBase):
    """Response for metadata extraction requests.

    Attributes:
        success: Whether the extraction succeeded.
        result: Metadata extraction result.
        message: Optional message or error description.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        ..., description="Whether extraction succeeded",
    )
    result: Optional[MetadataRecord] = Field(
        None, description="Metadata extraction result",
    )
    message: Optional[str] = Field(
        None, description="Optional message or error description",
    )

class FraudDetectionResponse(GreenLangBase):
    """Response for fraud pattern detection requests.

    Attributes:
        success: Whether the detection run succeeded.
        alerts: List of fraud alerts detected.
        total_alerts: Total number of alerts.
        highest_severity: Highest severity among detected alerts.
        composite_score: Composite fraud risk score.
        message: Optional message or error description.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        ..., description="Whether detection run succeeded",
    )
    alerts: List[FraudAlert] = Field(
        default_factory=list,
        description="List of fraud alerts detected",
    )
    total_alerts: int = Field(
        default=0, ge=0,
        description="Total number of alerts",
    )
    highest_severity: Optional[FraudSeverity] = Field(
        None, description="Highest severity among detected alerts",
    )
    composite_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Composite fraud risk score",
    )
    message: Optional[str] = Field(
        None, description="Optional message or error description",
    )

class CrossRefResponse(GreenLangBase):
    """Response for cross-reference verification requests.

    Attributes:
        success: Whether the cross-reference check succeeded.
        result: Cross-reference verification result.
        message: Optional message or error description.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        ..., description="Whether cross-reference check succeeded",
    )
    result: Optional[CrossRefResult] = Field(
        None, description="Cross-reference verification result",
    )
    message: Optional[str] = Field(
        None, description="Optional message or error description",
    )

class ReportResponse(GreenLangBase):
    """Response for report generation requests.

    Attributes:
        success: Whether report generation succeeded.
        report: Generated authentication report.
        message: Optional message or error description.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        ..., description="Whether report generation succeeded",
    )
    report: Optional[AuthenticationReport] = Field(
        None, description="Generated authentication report",
    )
    message: Optional[str] = Field(
        None, description="Optional message or error description",
    )

class BatchResponse(GreenLangBase):
    """Response for batch verification requests.

    Attributes:
        success: Whether the batch job was accepted.
        batch_job_id: Identifier of the created batch job.
        status: Current status of the batch job.
        total_documents: Total documents in the batch.
        completed_documents: Documents completed so far.
        failed_documents: Documents that failed verification.
        results: Individual document results (when completed).
        message: Optional message or error description.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        ..., description="Whether batch job was accepted",
    )
    batch_job_id: Optional[str] = Field(
        None, description="Identifier of the created batch job",
    )
    status: BatchJobStatus = Field(
        default=BatchJobStatus.PENDING,
        description="Current status of the batch job",
    )
    total_documents: int = Field(
        default=0, ge=0,
        description="Total documents in the batch",
    )
    completed_documents: int = Field(
        default=0, ge=0,
        description="Documents completed so far",
    )
    failed_documents: int = Field(
        default=0, ge=0,
        description="Documents that failed verification",
    )
    results: List[AuthenticationReport] = Field(
        default_factory=list,
        description="Individual document results (when completed)",
    )
    message: Optional[str] = Field(
        None, description="Optional message or error description",
    )

class HealthResponse(GreenLangBase):
    """Health check response for the document authentication service.

    Attributes:
        status: Service health status (healthy, degraded, unhealthy).
        version: Service version.
        agent_id: Agent identifier.
        database_connected: Whether database is reachable.
        redis_connected: Whether Redis is reachable.
        fsc_api_reachable: Whether FSC API is reachable.
        rspo_api_reachable: Whether RSPO API is reachable.
        iscc_api_reachable: Whether ISCC API is reachable.
        documents_processed: Total documents processed.
        fraud_alerts_active: Number of active (unresolved) fraud alerts.
        uptime_seconds: Service uptime in seconds.
        checked_at: UTC timestamp of health check.
    """

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(
        ..., description="Service health status",
    )
    version: str = Field(
        default=VERSION, description="Service version",
    )
    agent_id: str = Field(
        default="GL-EUDR-DAV-012", description="Agent identifier",
    )
    database_connected: bool = Field(
        default=False, description="Whether database is reachable",
    )
    redis_connected: bool = Field(
        default=False, description="Whether Redis is reachable",
    )
    fsc_api_reachable: bool = Field(
        default=False, description="Whether FSC API is reachable",
    )
    rspo_api_reachable: bool = Field(
        default=False, description="Whether RSPO API is reachable",
    )
    iscc_api_reachable: bool = Field(
        default=False, description="Whether ISCC API is reachable",
    )
    documents_processed: int = Field(
        default=0, ge=0,
        description="Total documents processed",
    )
    fraud_alerts_active: int = Field(
        default=0, ge=0,
        description="Number of active (unresolved) fraud alerts",
    )
    uptime_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Service uptime in seconds",
    )
    checked_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of health check",
    )

class DashboardResponse(GreenLangBase):
    """Dashboard summary response for the document authentication service.

    Attributes:
        total_documents: Total documents in the system.
        documents_authentic: Documents classified as authentic.
        documents_suspicious: Documents classified as suspicious.
        documents_fraudulent: Documents classified as fraudulent.
        documents_inconclusive: Documents with inconclusive results.
        fraud_alerts_total: Total fraud alerts generated.
        fraud_alerts_active: Active (unresolved) fraud alerts.
        fraud_alerts_critical: Critical severity alerts.
        documents_by_type: Document counts by document type.
        documents_by_commodity: Document counts by EUDR commodity.
        average_fraud_score: Average fraud score across all documents.
        average_classification_confidence: Average classification score.
        registries_status: Status of external registry connections.
        generated_at: UTC timestamp of dashboard generation.
    """

    model_config = ConfigDict(from_attributes=True)

    total_documents: int = Field(
        default=0, ge=0, description="Total documents in the system",
    )
    documents_authentic: int = Field(
        default=0, ge=0, description="Documents classified as authentic",
    )
    documents_suspicious: int = Field(
        default=0, ge=0, description="Documents classified as suspicious",
    )
    documents_fraudulent: int = Field(
        default=0, ge=0, description="Documents classified as fraudulent",
    )
    documents_inconclusive: int = Field(
        default=0, ge=0, description="Documents with inconclusive results",
    )
    fraud_alerts_total: int = Field(
        default=0, ge=0, description="Total fraud alerts generated",
    )
    fraud_alerts_active: int = Field(
        default=0, ge=0, description="Active (unresolved) fraud alerts",
    )
    fraud_alerts_critical: int = Field(
        default=0, ge=0, description="Critical severity alerts",
    )
    documents_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Document counts by document type",
    )
    documents_by_commodity: Dict[str, int] = Field(
        default_factory=dict,
        description="Document counts by EUDR commodity",
    )
    average_fraud_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Average fraud score across all documents",
    )
    average_classification_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average classification score",
    )
    registries_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of external registry connections",
    )
    generated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of dashboard generation",
    )

class TemplateResponse(GreenLangBase):
    """Response for template registration requests.

    Attributes:
        success: Whether the registration succeeded.
        template_id: Identifier of the registered template.
        template_name: Name of the registered template.
        message: Optional message or error description.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        ..., description="Whether registration succeeded",
    )
    template_id: Optional[str] = Field(
        None, description="Identifier of the registered template",
    )
    template_name: Optional[str] = Field(
        None, description="Name of the registered template",
    )
    message: Optional[str] = Field(
        None, description="Optional message or error description",
    )

class TrustedCAResponse(GreenLangBase):
    """Response for trusted CA addition requests.

    Attributes:
        success: Whether the addition succeeded.
        ca_id: Identifier of the added CA.
        ca_name: Name of the added CA.
        message: Optional message or error description.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        ..., description="Whether addition succeeded",
    )
    ca_id: Optional[str] = Field(
        None, description="Identifier of the added CA",
    )
    ca_name: Optional[str] = Field(
        None, description="Name of the added CA",
    )
    message: Optional[str] = Field(
        None, description="Optional message or error description",
    )

class DocumentSearchResponse(GreenLangBase):
    """Response for document search requests.

    Attributes:
        success: Whether the search succeeded.
        documents: List of matching document records.
        total_count: Total matching documents (for pagination).
        limit: Requested page size.
        offset: Requested offset.
        message: Optional message or error description.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        ..., description="Whether search succeeded",
    )
    documents: List[DocumentRecord] = Field(
        default_factory=list,
        description="List of matching document records",
    )
    total_count: int = Field(
        default=0, ge=0,
        description="Total matching documents (for pagination)",
    )
    limit: int = Field(
        default=50, ge=1,
        description="Requested page size",
    )
    offset: int = Field(
        default=0, ge=0,
        description="Requested offset",
    )
    message: Optional[str] = Field(
        None, description="Optional message or error description",
    )

class FraudAlertListResponse(GreenLangBase):
    """Response for fraud alert list requests.

    Attributes:
        success: Whether the query succeeded.
        alerts: List of matching fraud alerts.
        total_count: Total matching alerts (for pagination).
        limit: Requested page size.
        offset: Requested offset.
        message: Optional message or error description.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        ..., description="Whether query succeeded",
    )
    alerts: List[FraudAlert] = Field(
        default_factory=list,
        description="List of matching fraud alerts",
    )
    total_count: int = Field(
        default=0, ge=0,
        description="Total matching alerts (for pagination)",
    )
    limit: int = Field(
        default=50, ge=1,
        description="Requested page size",
    )
    offset: int = Field(
        default=0, ge=0,
        description="Requested offset",
    )
    message: Optional[str] = Field(
        None, description="Optional message or error description",
    )
