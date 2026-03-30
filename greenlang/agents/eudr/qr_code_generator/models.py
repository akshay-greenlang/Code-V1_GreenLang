# -*- coding: utf-8 -*-
"""
QR Code Generator Data Models - AGENT-EUDR-014

Pydantic v2 data models for the QR Code Generator Agent covering
QR code generation with configurable version, error correction, and
output format; data payload composition with compression and encryption;
label rendering with EUDR compliance status colour coding; batch code
generation with check digits; verification URL construction with HMAC
signing; anti-counterfeiting via scan velocity monitoring and geo-fencing;
bulk generation job orchestration; QR code lifecycle management
(activation, deactivation, revocation, expiry); scan event recording;
template management; code-to-entity association; and audit logging.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all QR code generation operations per
EU 2023/1115 Article 14.

Enumerations (15):
    - QRCodeVersion, ErrorCorrectionLevel, OutputFormat, ContentType,
      SymbologyType, LabelTemplate, CheckDigitAlgorithm, CodeStatus,
      ScanOutcome, CounterfeitRiskLevel, BulkJobStatus, EUDRCommodity,
      ComplianceStatus, PayloadEncoding, DPILevel

Core Models (12):
    - QRCodeRecord, DataPayload, LabelRecord, BatchCode,
      VerificationURL, SignatureRecord, ScanEvent, BulkJob,
      LifecycleEvent, TemplateDefinition, CodeAssociation,
      AuditLogEntry

Request Models (15):
    - GenerateQRCodeRequest, ComposePayloadRequest,
      RenderLabelRequest, GenerateBatchCodeRequest,
      BuildVerificationURLRequest, SignCodeRequest,
      RecordScanRequest, SubmitBulkJobRequest,
      ActivateCodeRequest, DeactivateCodeRequest,
      RevokeCodeRequest, ReprintCodeRequest,
      SearchCodesRequest, GetScanHistoryRequest,
      ValidateCodeRequest

Response Models (15):
    - QRCodeResponse, PayloadResponse, LabelResponse,
      BatchCodeResponse, VerificationURLResponse,
      SignatureResponse, ScanResponse, BulkJobResponse,
      ActivateResponse, DeactivateResponse,
      RevokeResponse, ReprintResponse,
      SearchResponse, ScanHistoryResponse,
      HealthResponse

Compatibility:
    Imports EUDRCommodity from greenlang.agents.data.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector and AGENT-EUDR-011 Mass Balance Calculator.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from greenlang.schemas import GreenLangBase, utcnow

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Cross-agent commodity import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.data.eudr_traceability.models import (
        EUDRCommodity as _ExternalEUDRCommodity,
    )
except ImportError:
    _ExternalEUDRCommodity = None  # type: ignore[assignment,misc]

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

#: Maximum number of records in a single batch processing job.
MAX_BATCH_SIZE: int = 500

#: EUDR Article 14 data retention in years.
EUDR_RETENTION_YEARS: int = 5

#: Maximum QR code payload bytes (QR v40-H binary capacity).
MAX_QR_PAYLOAD_BYTES: int = 2953

#: Maximum QR code version per ISO/IEC 18004.
MAX_QR_VERSION: int = 40

#: Supported output formats for QR code images.
SUPPORTED_OUTPUT_FORMATS: List[str] = ["png", "svg", "pdf", "zpl", "eps"]

#: Supported content types for QR code payloads.
SUPPORTED_CONTENT_TYPES: List[str] = [
    "full_traceability",
    "compact_verification",
    "consumer_summary",
    "batch_identifier",
    "blockchain_anchor",
]

#: Supported symbology types.
SUPPORTED_SYMBOLOGY_TYPES: List[str] = [
    "qr_code",
    "micro_qr",
    "data_matrix",
    "gs1_digital_link",
]

#: Supported label templates.
SUPPORTED_LABEL_TEMPLATES: List[str] = [
    "product_label",
    "shipping_label",
    "pallet_label",
    "container_label",
    "consumer_label",
]

#: Supported check digit algorithms.
SUPPORTED_CHECK_DIGIT_ALGORITHMS: List[str] = [
    "luhn",
    "iso7064_mod11_10",
    "crc8",
]

#: Default EUDR commodities (EU 2023/1115 Article 1).
DEFAULT_EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

# =============================================================================
# Enumerations
# =============================================================================

class QRCodeVersion(str, Enum):
    """QR code version selection per ISO/IEC 18004.

    AUTO: Automatic version selection based on payload size and
        error correction level. The smallest version that fits the
        data is chosen.
    V1 through V40: Fixed version selection. Higher versions support
        larger data capacity but produce denser codes that may be
        harder to scan on low-resolution printers.
    """

    AUTO = "auto"
    V1 = "1"
    V2 = "2"
    V3 = "3"
    V4 = "4"
    V5 = "5"
    V6 = "6"
    V7 = "7"
    V8 = "8"
    V9 = "9"
    V10 = "10"
    V11 = "11"
    V12 = "12"
    V13 = "13"
    V14 = "14"
    V15 = "15"
    V16 = "16"
    V17 = "17"
    V18 = "18"
    V19 = "19"
    V20 = "20"
    V21 = "21"
    V22 = "22"
    V23 = "23"
    V24 = "24"
    V25 = "25"
    V26 = "26"
    V27 = "27"
    V28 = "28"
    V29 = "29"
    V30 = "30"
    V31 = "31"
    V32 = "32"
    V33 = "33"
    V34 = "34"
    V35 = "35"
    V36 = "36"
    V37 = "37"
    V38 = "38"
    V39 = "39"
    V40 = "40"

class ErrorCorrectionLevel(str, Enum):
    """Error correction level per ISO/IEC 18004.

    L: Low - approximately 7% codeword recovery. Maximizes data
        capacity. Suitable for clean, controlled scanning environments.
    M: Medium - approximately 15% codeword recovery. Default level
        balancing capacity and resilience.
    Q: Quartile - approximately 25% codeword recovery. Good for
        labels exposed to moderate wear or partial obstruction.
    H: High - approximately 30% codeword recovery. Required when
        embedding a centre logo (logo covers ~10% of modules).
    """

    L = "L"
    M = "M"
    Q = "Q"
    H = "H"

class OutputFormat(str, Enum):
    """Output image format for generated QR codes.

    PNG: Raster PNG format. Best for screen display, email, and
        general-purpose use. Supports transparency.
    SVG: Scalable vector format. Best for print production and
        responsive web display. Infinite resolution.
    PDF: Portable Document Format with embedded vector QR code.
        Best for professional print production.
    ZPL: Zebra Programming Language for direct thermal/transfer
        printer output. Used in logistics and warehousing.
    EPS: Encapsulated PostScript for professional prepress and
        desktop publishing workflows.
    """

    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    ZPL = "zpl"
    EPS = "eps"

class ContentType(str, Enum):
    """Payload content type for QR code data.

    FULL_TRACEABILITY: Complete supply chain traceability data
        including operator ID, commodity, origin geolocation,
        DDS reference, certification IDs, and custody chain.
        Largest payload, intended for competent authority scanning.
    COMPACT_VERIFICATION: Compact verification payload containing
        operator ID, DDS reference number, compliance status,
        and HMAC verification token. Default content type.
    CONSUMER_SUMMARY: Consumer-facing summary with product origin
        country, commodity type, deforestation-free status, and
        verification URL. Human-readable when scanned.
    BATCH_IDENTIFIER: Batch-level identifier encoding batch code,
        commodity, quantity, production date, and facility.
        Used for logistics and warehouse management.
    BLOCKCHAIN_ANCHOR: Blockchain anchor reference containing
        transaction hash, chain ID, block number, and Merkle
        proof reference. Links to AGENT-EUDR-013 anchor records.
    """

    FULL_TRACEABILITY = "full_traceability"
    COMPACT_VERIFICATION = "compact_verification"
    CONSUMER_SUMMARY = "consumer_summary"
    BATCH_IDENTIFIER = "batch_identifier"
    BLOCKCHAIN_ANCHOR = "blockchain_anchor"

class SymbologyType(str, Enum):
    """Barcode symbology type for generated codes.

    QR_CODE: Standard ISO/IEC 18004 QR code. Primary symbology
        for EUDR compliance labels.
    MICRO_QR: ISO/IEC 18004 Micro QR code. Compact variant for
        very small labels where space is limited (max 35 numeric
        characters).
    DATA_MATRIX: ISO/IEC 16022 Data Matrix code. Alternative 2D
        symbology preferred in some pharmaceutical and automotive
        supply chains.
    GS1_DIGITAL_LINK: GS1 Digital Link URI encoded in QR format.
        Enables web-resolvable product identification with GTIN
        and additional data attributes.
    """

    QR_CODE = "qr_code"
    MICRO_QR = "micro_qr"
    DATA_MATRIX = "data_matrix"
    GS1_DIGITAL_LINK = "gs1_digital_link"

class LabelTemplate(str, Enum):
    """Pre-designed label template for QR code rendering.

    PRODUCT_LABEL: Individual product label with QR code, product
        name, compliance status indicator, and verification URL.
    SHIPPING_LABEL: Shipping carton label with QR code, batch code,
        destination, weight, and carrier barcode area.
    PALLET_LABEL: Pallet-level label with large QR code, SSCC,
        batch codes, and handling instructions.
    CONTAINER_LABEL: Shipping container label with QR code,
        container number, seal number, and customs reference.
    CONSUMER_LABEL: Consumer-facing label with QR code, origin
        story, deforestation-free badge, and scan instructions.
    """

    PRODUCT_LABEL = "product_label"
    SHIPPING_LABEL = "shipping_label"
    PALLET_LABEL = "pallet_label"
    CONTAINER_LABEL = "container_label"
    CONSUMER_LABEL = "consumer_label"

class CheckDigitAlgorithm(str, Enum):
    """Algorithm for computing batch code check digits.

    LUHN: Luhn algorithm (mod 10). Widely used in credit card
        numbers and IMEI. Detects all single-digit errors and
        most transposition errors.
    ISO7064_MOD11_10: ISO 7064 Mod 11,10 hybrid system. Used in
        ISBN-13 and GS1 identifiers. Detects all single-digit
        and adjacent transposition errors.
    CRC8: CRC-8 cyclic redundancy check. 8-bit check value
        providing higher error detection than Luhn for longer
        codes. Detects burst errors up to 8 bits.
    """

    LUHN = "luhn"
    ISO7064_MOD11_10 = "iso7064_mod11_10"
    CRC8 = "crc8"

class CodeStatus(str, Enum):
    """Lifecycle status of a generated QR code.

    CREATED: QR code generated but not yet activated for scanning.
        Pending label printing or distribution.
    ACTIVE: QR code is live and scannable. Verification requests
        are processed normally.
    DEACTIVATED: QR code temporarily deactivated. Scans return a
        deactivated status but the code can be reactivated.
    REVOKED: QR code permanently revoked due to compliance failure,
        product recall, or fraud detection. Cannot be reactivated.
    EXPIRED: QR code has passed its TTL expiry date. Scans return
        an expired status. Per EUDR Article 14, records are retained
        for 5 years after expiry.
    """

    CREATED = "created"
    ACTIVE = "active"
    DEACTIVATED = "deactivated"
    REVOKED = "revoked"
    EXPIRED = "expired"

class ScanOutcome(str, Enum):
    """Outcome of a QR code scan verification event.

    VERIFIED: Code successfully verified. Compliance data is valid
        and the HMAC token matches.
    COUNTERFEIT_SUSPECTED: Scan triggered counterfeit detection rules
        (velocity threshold exceeded, geo-fence violation, or HMAC
        mismatch).
    EXPIRED_CODE: Code has passed its TTL expiry date and is no
        longer valid for verification.
    REVOKED_CODE: Code has been permanently revoked and is no longer
        valid.
    ERROR: Verification failed due to a system error (database
        timeout, malformed payload, decryption failure).
    """

    VERIFIED = "verified"
    COUNTERFEIT_SUSPECTED = "counterfeit_suspected"
    EXPIRED_CODE = "expired_code"
    REVOKED_CODE = "revoked_code"
    ERROR = "error"

class CounterfeitRiskLevel(str, Enum):
    """Risk level assessment for counterfeit detection.

    LOW: Normal scan pattern. No anomalies detected. Risk score
        below 25th percentile.
    MEDIUM: Slightly elevated scan velocity or minor geo-fence
        proximity. Risk score 25th-75th percentile.
    HIGH: Scan velocity exceeds threshold or geo-fence boundary
        crossed. Risk score 75th-95th percentile. Alerts triggered.
    CRITICAL: Multiple counterfeit indicators present (HMAC mismatch,
        extreme velocity, impossible geo-location). Risk score above
        95th percentile. Immediate investigation required.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class BulkJobStatus(str, Enum):
    """Status of a bulk QR code generation job.

    QUEUED: Job has been accepted and is waiting for worker
        allocation.
    PROCESSING: Job is currently being executed by one or more
        workers. Progress percentage is updated periodically.
    COMPLETED: Job finished successfully. All codes generated and
        output package is available for download.
    FAILED: Job failed due to an unrecoverable error (disk space,
        memory, or validation failure).
    CANCELLED: Job was cancelled by the operator before completion.
        Partially generated codes are discarded.
    """

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EUDRCommodity(str, Enum):
    """EUDR-regulated commodity types per EU 2023/1115 Article 1.

    CATTLE: Cattle and derived products (leather, beef).
    COCOA: Cocoa beans and derived products (chocolate, cocoa butter).
    COFFEE: Coffee beans and derived products (roasted, instant).
    OIL_PALM: Oil palm and derived products (palm oil, palm kernel oil).
    RUBBER: Natural rubber and derived products (tyres, latex).
    SOYA: Soybeans and derived products (soy meal, soy oil).
    WOOD: Wood and derived products (timber, pulp, paper, furniture).
    """

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"

class ComplianceStatus(str, Enum):
    """EUDR compliance status for a product or operator.

    COMPLIANT: Product or operator fully meets EUDR due diligence
        requirements. Deforestation-free and legally produced.
    PENDING: Compliance assessment is in progress. Due diligence
        statement has been submitted but not yet verified.
    NON_COMPLIANT: Product or operator does not meet EUDR
        requirements. Deforestation risk identified or legal
        compliance not demonstrated.
    UNDER_REVIEW: Compliance status is under review by competent
        authority or due to a flagged anomaly.
    """

    COMPLIANT = "compliant"
    PENDING = "pending"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"

class PayloadEncoding(str, Enum):
    """Encoding format for QR code payload data.

    UTF8: Plain UTF-8 text encoding. Human-readable when scanned
        with a standard QR reader.
    BASE64: Base64-encoded binary payload. Used for structured
        data that is not human-readable.
    ZLIB_BASE64: Zlib-compressed then Base64-encoded payload.
        Reduces data size for large payloads while maintaining
        text-safe encoding.
    """

    UTF8 = "utf8"
    BASE64 = "base64"
    ZLIB_BASE64 = "zlib_base64"

class DPILevel(str, Enum):
    """Dots-per-inch resolution presets for QR code output.

    SCREEN_72: 72 DPI for screen display and email attachments.
    DRAFT_150: 150 DPI for draft quality printing and proofing.
    STANDARD_300: 300 DPI for standard quality label printing.
        Default for production labels.
    HIGH_600: 600 DPI for high-resolution printing on small labels
        or when using very high module density (v25+).
    """

    SCREEN_72 = "screen_72"
    DRAFT_150 = "draft_150"
    STANDARD_300 = "standard_300"
    HIGH_600 = "high_600"

# =============================================================================
# Core Models
# =============================================================================

class QRCodeRecord(GreenLangBase):
    """A generated QR code record for EUDR supply chain compliance.

    Represents a single QR code generated for product labelling,
    verification, or traceability purposes under EUDR regulation.

    Attributes:
        code_id: Unique QR code identifier (UUID v4).
        version: QR code version used (auto-selected or fixed).
        error_correction: Error correction level applied.
        symbology: Barcode symbology type.
        output_format: Output image format.
        module_size: Pixel size of each QR module.
        quiet_zone: Quiet zone width in modules.
        dpi: Output resolution in DPI.
        payload_hash: SHA-256 hash of the encoded payload.
        payload_size_bytes: Size of the encoded payload in bytes.
        content_type: Type of data encoded in the QR code.
        encoding: Payload encoding format.
        image_data_hash: SHA-256 hash of the generated image data.
        image_width_px: Width of the output image in pixels.
        image_height_px: Height of the output image in pixels.
        logo_embedded: Whether a centre logo was embedded.
        quality_grade: Achieved ISO/IEC 15416 print quality grade.
        operator_id: EUDR operator who owns this QR code.
        commodity: EUDR-regulated commodity type.
        compliance_status: Current EUDR compliance status.
        dds_reference: Due Diligence Statement reference number.
        batch_code: Associated batch code (if any).
        verification_url: Associated verification URL.
        blockchain_anchor_hash: Blockchain anchor hash (if anchored).
        status: Current lifecycle status of the QR code.
        reprint_count: Number of times this code has been reprinted.
        scan_count: Total number of times this code has been scanned.
        created_at: UTC timestamp when the code was generated.
        activated_at: UTC timestamp when the code was activated.
        deactivated_at: UTC timestamp when the code was deactivated.
        revoked_at: UTC timestamp when the code was revoked.
        expires_at: UTC timestamp when the code expires.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    code_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique QR code identifier (UUID v4)",
    )
    version: str = Field(
        default="auto",
        description="QR code version (auto or 1-40)",
    )
    error_correction: ErrorCorrectionLevel = Field(
        default=ErrorCorrectionLevel.M,
        description="Error correction level applied",
    )
    symbology: SymbologyType = Field(
        default=SymbologyType.QR_CODE,
        description="Barcode symbology type",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG,
        description="Output image format",
    )
    module_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Pixel size of each QR module",
    )
    quiet_zone: int = Field(
        default=4,
        ge=0,
        le=20,
        description="Quiet zone width in modules",
    )
    dpi: int = Field(
        default=300,
        ge=72,
        le=1200,
        description="Output resolution in DPI",
    )
    payload_hash: str = Field(
        ...,
        min_length=64,
        max_length=128,
        description="SHA-256 hash of the encoded payload",
    )
    payload_size_bytes: int = Field(
        ...,
        ge=1,
        description="Size of the encoded payload in bytes",
    )
    content_type: ContentType = Field(
        default=ContentType.COMPACT_VERIFICATION,
        description="Type of data encoded in the QR code",
    )
    encoding: PayloadEncoding = Field(
        default=PayloadEncoding.UTF8,
        description="Payload encoding format",
    )
    image_data_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the generated image data",
    )
    image_width_px: Optional[int] = Field(
        None,
        ge=1,
        description="Width of the output image in pixels",
    )
    image_height_px: Optional[int] = Field(
        None,
        ge=1,
        description="Height of the output image in pixels",
    )
    logo_embedded: bool = Field(
        default=False,
        description="Whether a centre logo was embedded",
    )
    quality_grade: Optional[str] = Field(
        None,
        description="Achieved ISO/IEC 15416 print quality grade",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="EUDR operator identifier",
    )
    commodity: Optional[str] = Field(
        None,
        description="EUDR-regulated commodity type",
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.PENDING,
        description="Current EUDR compliance status",
    )
    dds_reference: Optional[str] = Field(
        None,
        description="Due Diligence Statement reference number",
    )
    batch_code: Optional[str] = Field(
        None,
        description="Associated batch code",
    )
    verification_url: Optional[str] = Field(
        None,
        description="Associated verification URL",
    )
    blockchain_anchor_hash: Optional[str] = Field(
        None,
        description="Blockchain anchor hash if anchored",
    )
    status: CodeStatus = Field(
        default=CodeStatus.CREATED,
        description="Current lifecycle status",
    )
    reprint_count: int = Field(
        default=0,
        ge=0,
        description="Number of reprints",
    )
    scan_count: int = Field(
        default=0,
        ge=0,
        description="Total scan count",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when code was generated",
    )
    activated_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when code was activated",
    )
    deactivated_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when code was deactivated",
    )
    revoked_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when code was revoked",
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when code expires",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash for audit trail",
    )

    @field_validator("payload_hash")
    @classmethod
    def validate_payload_hash_hex(cls, v: str) -> str:
        """Validate payload_hash is a valid hexadecimal string."""
        try:
            int(v, 16)
        except ValueError:
            raise ValueError(
                f"payload_hash must be a valid hexadecimal string, "
                f"got '{v}'"
            )
        return v.lower()

class DataPayload(GreenLangBase):
    """Structured data payload for QR code encoding.

    Attributes:
        payload_id: Unique payload identifier (UUID v4).
        content_type: Type of content encoded.
        encoding: Encoding format applied to the raw data.
        raw_data: Original uncompressed/unencrypted data.
        encoded_data: Final encoded data ready for QR generation.
        compressed: Whether the payload was zlib-compressed.
        encrypted: Whether the payload was AES-256-GCM encrypted.
        compression_ratio: Compression ratio achieved (0.0-1.0).
        payload_version: Schema version of the payload format.
        operator_id: EUDR operator identifier.
        commodity: EUDR-regulated commodity type.
        dds_reference: Due Diligence Statement reference.
        compliance_status: EUDR compliance status.
        origin_country: Country of origin (ISO 3166-1 alpha-2).
        origin_coordinates: Geolocation coordinates (lat, lon).
        certification_ids: List of certification identifiers.
        blockchain_tx_hash: Blockchain transaction hash reference.
        payload_hash: SHA-256 hash of the encoded data.
        size_bytes: Size of the encoded payload in bytes.
        created_at: UTC timestamp when payload was composed.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    payload_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique payload identifier (UUID v4)",
    )
    content_type: ContentType = Field(
        default=ContentType.COMPACT_VERIFICATION,
        description="Type of content encoded",
    )
    encoding: PayloadEncoding = Field(
        default=PayloadEncoding.UTF8,
        description="Encoding format applied",
    )
    raw_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original uncompressed data",
    )
    encoded_data: Optional[str] = Field(
        None,
        description="Final encoded data for QR generation",
    )
    compressed: bool = Field(
        default=False,
        description="Whether payload was zlib-compressed",
    )
    encrypted: bool = Field(
        default=False,
        description="Whether payload was AES-256-GCM encrypted",
    )
    compression_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Compression ratio achieved",
    )
    payload_version: str = Field(
        default="1.0",
        description="Payload schema version",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="EUDR operator identifier",
    )
    commodity: Optional[str] = Field(
        None,
        description="EUDR-regulated commodity type",
    )
    dds_reference: Optional[str] = Field(
        None,
        description="Due Diligence Statement reference",
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.PENDING,
        description="EUDR compliance status",
    )
    origin_country: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Country of origin (ISO 3166-1 alpha-2)",
    )
    origin_coordinates: Optional[Dict[str, float]] = Field(
        None,
        description="Geolocation coordinates {lat, lon}",
    )
    certification_ids: List[str] = Field(
        default_factory=list,
        description="List of certification identifiers",
    )
    blockchain_tx_hash: Optional[str] = Field(
        None,
        description="Blockchain transaction hash reference",
    )
    payload_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the encoded data",
    )
    size_bytes: Optional[int] = Field(
        None,
        ge=0,
        description="Size of the encoded payload in bytes",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when payload was composed",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )

class LabelRecord(GreenLangBase):
    """A rendered label containing a QR code for EUDR compliance.

    Attributes:
        label_id: Unique label identifier (UUID v4).
        code_id: Associated QR code identifier.
        template: Label template used for rendering.
        font: Font family used for text rendering.
        font_size: Font size in points.
        compliance_color_hex: Status colour applied to the label.
        compliance_status: EUDR compliance status displayed.
        bleed_mm: Print bleed margin in millimetres.
        width_mm: Label width in millimetres.
        height_mm: Label height in millimetres.
        output_format: Output image/document format.
        dpi: Output resolution in DPI.
        image_data_hash: SHA-256 hash of the rendered image.
        file_size_bytes: Output file size in bytes.
        operator_id: EUDR operator identifier.
        commodity: EUDR-regulated commodity type.
        product_name: Product name displayed on label.
        batch_code: Batch code displayed on label.
        verification_url: Verification URL displayed on label.
        custom_fields: Additional custom fields rendered on label.
        created_at: UTC timestamp when label was rendered.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    label_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique label identifier (UUID v4)",
    )
    code_id: str = Field(
        ...,
        min_length=1,
        description="Associated QR code identifier",
    )
    template: LabelTemplate = Field(
        default=LabelTemplate.PRODUCT_LABEL,
        description="Label template used",
    )
    font: str = Field(
        default="DejaVuSans",
        description="Font family for text rendering",
    )
    font_size: int = Field(
        default=12,
        ge=4,
        le=72,
        description="Font size in points",
    )
    compliance_color_hex: str = Field(
        default="#2E7D32",
        description="Status colour hex code",
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.PENDING,
        description="EUDR compliance status displayed",
    )
    bleed_mm: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Print bleed margin in mm",
    )
    width_mm: Optional[float] = Field(
        None,
        gt=0,
        description="Label width in mm",
    )
    height_mm: Optional[float] = Field(
        None,
        gt=0,
        description="Label height in mm",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PDF,
        description="Output format",
    )
    dpi: int = Field(
        default=300,
        ge=72,
        le=1200,
        description="Output resolution in DPI",
    )
    image_data_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the rendered image",
    )
    file_size_bytes: Optional[int] = Field(
        None,
        ge=0,
        description="Output file size in bytes",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="EUDR operator identifier",
    )
    commodity: Optional[str] = Field(
        None,
        description="EUDR-regulated commodity type",
    )
    product_name: Optional[str] = Field(
        None,
        description="Product name on label",
    )
    batch_code: Optional[str] = Field(
        None,
        description="Batch code on label",
    )
    verification_url: Optional[str] = Field(
        None,
        description="Verification URL on label",
    )
    custom_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional custom fields",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when label was rendered",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )

class BatchCode(GreenLangBase):
    """A generated batch code for EUDR supply chain tracking.

    Attributes:
        batch_code_id: Unique batch code record identifier (UUID v4).
        code_value: Full batch code string including prefix and check digit.
        prefix: Batch code prefix (operator-commodity-year).
        sequence_number: Numeric sequence portion.
        check_digit: Computed check digit.
        check_digit_algorithm: Algorithm used for check digit.
        operator_id: EUDR operator identifier.
        commodity: EUDR-regulated commodity type.
        year: Production or import year.
        facility_id: Facility or plant identifier.
        quantity: Batch quantity.
        quantity_unit: Unit of measurement for quantity.
        associated_code_ids: List of QR code IDs linked to this batch.
        status: Current status of the batch code.
        created_at: UTC timestamp when batch code was generated.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    batch_code_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch code record identifier (UUID v4)",
    )
    code_value: str = Field(
        ...,
        min_length=1,
        description="Full batch code string",
    )
    prefix: str = Field(
        ...,
        min_length=1,
        description="Batch code prefix",
    )
    sequence_number: int = Field(
        ...,
        ge=0,
        description="Numeric sequence portion",
    )
    check_digit: str = Field(
        ...,
        description="Computed check digit",
    )
    check_digit_algorithm: CheckDigitAlgorithm = Field(
        default=CheckDigitAlgorithm.LUHN,
        description="Algorithm used for check digit",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="EUDR operator identifier",
    )
    commodity: Optional[str] = Field(
        None,
        description="EUDR-regulated commodity type",
    )
    year: int = Field(
        ...,
        ge=2020,
        le=2100,
        description="Production or import year",
    )
    facility_id: Optional[str] = Field(
        None,
        description="Facility or plant identifier",
    )
    quantity: Optional[float] = Field(
        None,
        ge=0,
        description="Batch quantity",
    )
    quantity_unit: Optional[str] = Field(
        None,
        description="Unit of measurement for quantity",
    )
    associated_code_ids: List[str] = Field(
        default_factory=list,
        description="QR code IDs linked to this batch",
    )
    status: CodeStatus = Field(
        default=CodeStatus.CREATED,
        description="Current batch code status",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when batch code was generated",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )

class VerificationURL(GreenLangBase):
    """A verification URL for a QR code with HMAC-signed token.

    Attributes:
        url_id: Unique URL record identifier (UUID v4).
        code_id: Associated QR code identifier.
        full_url: Complete verification URL with token.
        short_url: Shortened URL (if short URL service enabled).
        base_url: Base verification URL.
        token: HMAC-SHA256 verification token.
        hmac_truncated: Truncated HMAC hex string.
        token_created_at: UTC timestamp when token was created.
        token_expires_at: UTC timestamp when token expires.
        operator_id: EUDR operator identifier.
        created_at: UTC timestamp when URL was constructed.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    url_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique URL record identifier (UUID v4)",
    )
    code_id: str = Field(
        ...,
        min_length=1,
        description="Associated QR code identifier",
    )
    full_url: str = Field(
        ...,
        min_length=1,
        description="Complete verification URL with token",
    )
    short_url: Optional[str] = Field(
        None,
        description="Shortened URL",
    )
    base_url: str = Field(
        ...,
        min_length=1,
        description="Base verification URL",
    )
    token: str = Field(
        ...,
        min_length=1,
        description="HMAC-SHA256 verification token",
    )
    hmac_truncated: str = Field(
        ...,
        min_length=4,
        description="Truncated HMAC hex string",
    )
    token_created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when token was created",
    )
    token_expires_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when token expires",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="EUDR operator identifier",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when URL was constructed",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )

class SignatureRecord(GreenLangBase):
    """A digital signature record for QR code authentication.

    Attributes:
        signature_id: Unique signature record identifier (UUID v4).
        code_id: Associated QR code identifier.
        algorithm: Signing algorithm used (e.g. HMAC-SHA256).
        key_id: Identifier of the signing key used.
        signature_value: Hex-encoded signature value.
        signed_data_hash: SHA-256 hash of the data that was signed.
        valid: Whether the signature verification passed.
        verified_at: UTC timestamp of last verification.
        created_at: UTC timestamp when signature was created.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    signature_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique signature record identifier (UUID v4)",
    )
    code_id: str = Field(
        ...,
        min_length=1,
        description="Associated QR code identifier",
    )
    algorithm: str = Field(
        default="HMAC-SHA256",
        description="Signing algorithm used",
    )
    key_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the signing key",
    )
    signature_value: str = Field(
        ...,
        min_length=1,
        description="Hex-encoded signature value",
    )
    signed_data_hash: str = Field(
        ...,
        min_length=64,
        max_length=128,
        description="SHA-256 hash of signed data",
    )
    valid: bool = Field(
        default=True,
        description="Whether signature verification passed",
    )
    verified_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp of last verification",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when signature was created",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )

    @field_validator("signed_data_hash")
    @classmethod
    def validate_signed_data_hash_hex(cls, v: str) -> str:
        """Validate signed_data_hash is a valid hexadecimal string."""
        try:
            int(v, 16)
        except ValueError:
            raise ValueError(
                f"signed_data_hash must be a valid hexadecimal string, "
                f"got '{v}'"
            )
        return v.lower()

class ScanEvent(GreenLangBase):
    """A scan event recorded when a QR code is scanned.

    Attributes:
        scan_id: Unique scan event identifier (UUID v4).
        code_id: Scanned QR code identifier.
        outcome: Scan verification outcome.
        scanner_ip: IP address of the scanner (hashed for privacy).
        scanner_user_agent: User agent string of the scanner.
        scan_latitude: Scan location latitude.
        scan_longitude: Scan location longitude.
        scan_country: Country where scan occurred (ISO 3166-1 alpha-2).
        counterfeit_risk: Assessed counterfeit risk level.
        velocity_scans_per_min: Scan velocity at time of event.
        geo_fence_violated: Whether scan violated a geo-fence.
        hmac_valid: Whether the HMAC token validated successfully.
        response_time_ms: Time to process the scan in milliseconds.
        scanned_at: UTC timestamp when the scan occurred.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    scan_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique scan event identifier (UUID v4)",
    )
    code_id: str = Field(
        ...,
        min_length=1,
        description="Scanned QR code identifier",
    )
    outcome: ScanOutcome = Field(
        ...,
        description="Scan verification outcome",
    )
    scanner_ip: Optional[str] = Field(
        None,
        description="IP address of scanner (hashed for privacy)",
    )
    scanner_user_agent: Optional[str] = Field(
        None,
        description="User agent string of scanner",
    )
    scan_latitude: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Scan location latitude",
    )
    scan_longitude: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Scan location longitude",
    )
    scan_country: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Country of scan (ISO 3166-1 alpha-2)",
    )
    counterfeit_risk: CounterfeitRiskLevel = Field(
        default=CounterfeitRiskLevel.LOW,
        description="Assessed counterfeit risk level",
    )
    velocity_scans_per_min: Optional[int] = Field(
        None,
        ge=0,
        description="Scan velocity at time of event",
    )
    geo_fence_violated: bool = Field(
        default=False,
        description="Whether scan violated a geo-fence",
    )
    hmac_valid: Optional[bool] = Field(
        None,
        description="Whether HMAC token validated",
    )
    response_time_ms: Optional[float] = Field(
        None,
        ge=0,
        description="Processing time in milliseconds",
    )
    scanned_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when scan occurred",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )

class BulkJob(GreenLangBase):
    """A bulk QR code generation job.

    Attributes:
        job_id: Unique bulk job identifier (UUID v4).
        status: Current job status.
        total_codes: Total number of codes to generate.
        completed_codes: Number of codes generated so far.
        failed_codes: Number of codes that failed to generate.
        progress_percent: Completion percentage (0.0-100.0).
        output_format: Output format for generated codes.
        bulk_output_format: Packaging format (zip, tar_gz).
        output_file_hash: SHA-256 hash of the output package.
        output_file_size_bytes: Size of the output package.
        output_file_url: URL for downloading the output package.
        operator_id: EUDR operator who submitted the job.
        commodity: EUDR-regulated commodity type.
        content_type: Payload content type for generated codes.
        error_correction: Error correction level for generated codes.
        worker_count: Number of workers assigned to this job.
        error_message: Error message if job failed.
        started_at: UTC timestamp when processing started.
        completed_at: UTC timestamp when job completed.
        created_at: UTC timestamp when job was submitted.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique bulk job identifier (UUID v4)",
    )
    status: BulkJobStatus = Field(
        default=BulkJobStatus.QUEUED,
        description="Current job status",
    )
    total_codes: int = Field(
        ...,
        ge=1,
        description="Total number of codes to generate",
    )
    completed_codes: int = Field(
        default=0,
        ge=0,
        description="Codes generated so far",
    )
    failed_codes: int = Field(
        default=0,
        ge=0,
        description="Codes that failed to generate",
    )
    progress_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Completion percentage",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG,
        description="Output format for generated codes",
    )
    bulk_output_format: str = Field(
        default="zip",
        description="Packaging format (zip, tar_gz)",
    )
    output_file_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the output package",
    )
    output_file_size_bytes: Optional[int] = Field(
        None,
        ge=0,
        description="Size of the output package",
    )
    output_file_url: Optional[str] = Field(
        None,
        description="URL for downloading output package",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="EUDR operator who submitted the job",
    )
    commodity: Optional[str] = Field(
        None,
        description="EUDR-regulated commodity type",
    )
    content_type: ContentType = Field(
        default=ContentType.COMPACT_VERIFICATION,
        description="Payload content type for generated codes",
    )
    error_correction: ErrorCorrectionLevel = Field(
        default=ErrorCorrectionLevel.M,
        description="Error correction level for generated codes",
    )
    worker_count: int = Field(
        default=4,
        ge=1,
        description="Number of workers assigned",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if job failed",
    )
    started_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when processing started",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when job completed",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when job was submitted",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )

class LifecycleEvent(GreenLangBase):
    """A lifecycle event for a QR code (activation, deactivation, etc.).

    Attributes:
        event_id: Unique event identifier (UUID v4).
        code_id: Associated QR code identifier.
        event_type: Type of lifecycle event.
        previous_status: Status before the event.
        new_status: Status after the event.
        reason: Reason for the lifecycle change.
        performed_by: User or system that performed the change.
        metadata: Additional event metadata.
        created_at: UTC timestamp when event occurred.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier (UUID v4)",
    )
    code_id: str = Field(
        ...,
        min_length=1,
        description="Associated QR code identifier",
    )
    event_type: str = Field(
        ...,
        min_length=1,
        description="Type of lifecycle event",
    )
    previous_status: CodeStatus = Field(
        ...,
        description="Status before the event",
    )
    new_status: CodeStatus = Field(
        ...,
        description="Status after the event",
    )
    reason: Optional[str] = Field(
        None,
        description="Reason for the lifecycle change",
    )
    performed_by: Optional[str] = Field(
        None,
        description="User or system that performed the change",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event metadata",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when event occurred",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )

class TemplateDefinition(GreenLangBase):
    """A label template definition for QR code rendering.

    Attributes:
        template_id: Unique template identifier (UUID v4).
        name: Template name.
        template_type: Label template type.
        description: Human-readable template description.
        width_mm: Template width in millimetres.
        height_mm: Template height in millimetres.
        qr_position_x_mm: QR code X position in mm.
        qr_position_y_mm: QR code Y position in mm.
        qr_size_mm: QR code size in mm.
        fields: List of field definitions for the template.
        version: Template version string.
        is_active: Whether this template is currently active.
        created_at: UTC timestamp when template was created.
        updated_at: UTC timestamp when template was last updated.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    template_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique template identifier (UUID v4)",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Template name",
    )
    template_type: LabelTemplate = Field(
        ...,
        description="Label template type",
    )
    description: Optional[str] = Field(
        None,
        description="Human-readable template description",
    )
    width_mm: float = Field(
        ...,
        gt=0,
        description="Template width in mm",
    )
    height_mm: float = Field(
        ...,
        gt=0,
        description="Template height in mm",
    )
    qr_position_x_mm: float = Field(
        default=0.0,
        ge=0.0,
        description="QR code X position in mm",
    )
    qr_position_y_mm: float = Field(
        default=0.0,
        ge=0.0,
        description="QR code Y position in mm",
    )
    qr_size_mm: float = Field(
        default=20.0,
        gt=0,
        description="QR code size in mm",
    )
    fields: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Field definitions for the template",
    )
    version: str = Field(
        default="1.0",
        description="Template version string",
    )
    is_active: bool = Field(
        default=True,
        description="Whether template is currently active",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when template was created",
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when template was last updated",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )

class CodeAssociation(GreenLangBase):
    """An association between a QR code and an external entity.

    Attributes:
        association_id: Unique association identifier (UUID v4).
        code_id: QR code identifier.
        entity_type: Type of associated entity (product, shipment, batch).
        entity_id: Identifier of the associated entity.
        relationship: Relationship type (primary, secondary, reference).
        metadata: Additional association metadata.
        created_at: UTC timestamp when association was created.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    association_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique association identifier (UUID v4)",
    )
    code_id: str = Field(
        ...,
        min_length=1,
        description="QR code identifier",
    )
    entity_type: str = Field(
        ...,
        min_length=1,
        description="Type of associated entity",
    )
    entity_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of associated entity",
    )
    relationship: str = Field(
        default="primary",
        description="Relationship type",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional association metadata",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when association was created",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )

class AuditLogEntry(GreenLangBase):
    """An audit log entry for QR code generator operations.

    Attributes:
        entry_id: Unique audit entry identifier (UUID v4).
        operation: Operation that was performed.
        entity_type: Type of entity affected.
        entity_id: Identifier of the entity affected.
        operator_id: User or system that performed the operation.
        details: Operation details and parameters.
        result: Operation result (success, failure, partial).
        error_message: Error message if operation failed.
        ip_address: IP address of the requester.
        duration_ms: Operation duration in milliseconds.
        created_at: UTC timestamp when entry was recorded.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    entry_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique audit entry identifier (UUID v4)",
    )
    operation: str = Field(
        ...,
        min_length=1,
        description="Operation performed",
    )
    entity_type: str = Field(
        ...,
        min_length=1,
        description="Type of entity affected",
    )
    entity_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of entity affected",
    )
    operator_id: Optional[str] = Field(
        None,
        description="User or system that performed the operation",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation details and parameters",
    )
    result: str = Field(
        default="success",
        description="Operation result",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if operation failed",
    )
    ip_address: Optional[str] = Field(
        None,
        description="IP address of requester",
    )
    duration_ms: Optional[float] = Field(
        None,
        ge=0,
        description="Operation duration in milliseconds",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when entry was recorded",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )

# =============================================================================
# Request Models
# =============================================================================

class GenerateQRCodeRequest(GreenLangBase):
    """Request to generate a QR code."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    operator_id: str = Field(..., min_length=1, description="EUDR operator ID")
    payload_data: Dict[str, Any] = Field(..., description="Data to encode")
    content_type: Optional[str] = Field(None, description="Payload content type")
    version: Optional[str] = Field(None, description="QR version (auto or 1-40)")
    error_correction: Optional[str] = Field(None, description="Error correction level")
    output_format: Optional[str] = Field(None, description="Output format")
    module_size: Optional[int] = Field(None, ge=1, le=100, description="Module size")
    quiet_zone: Optional[int] = Field(None, ge=0, le=20, description="Quiet zone")
    dpi: Optional[int] = Field(None, ge=72, le=1200, description="DPI")
    embed_logo: Optional[bool] = Field(None, description="Embed centre logo")
    commodity: Optional[str] = Field(None, description="EUDR commodity")
    dds_reference: Optional[str] = Field(None, description="DDS reference number")
    compliance_status: Optional[str] = Field(None, description="Compliance status")

class ComposePayloadRequest(GreenLangBase):
    """Request to compose a data payload for QR encoding."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    operator_id: str = Field(..., min_length=1, description="EUDR operator ID")
    content_type: Optional[str] = Field(None, description="Content type")
    data: Dict[str, Any] = Field(..., description="Raw payload data")
    compress: Optional[bool] = Field(None, description="Enable compression")
    encrypt: Optional[bool] = Field(None, description="Enable encryption")
    commodity: Optional[str] = Field(None, description="EUDR commodity")
    dds_reference: Optional[str] = Field(None, description="DDS reference")
    compliance_status: Optional[str] = Field(None, description="Compliance status")
    origin_country: Optional[str] = Field(None, description="Origin country code")

class RenderLabelRequest(GreenLangBase):
    """Request to render a label with QR code."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., min_length=1, description="QR code ID")
    operator_id: str = Field(..., min_length=1, description="EUDR operator ID")
    template: Optional[str] = Field(None, description="Label template")
    product_name: Optional[str] = Field(None, description="Product name")
    batch_code: Optional[str] = Field(None, description="Batch code")
    custom_fields: Optional[Dict[str, str]] = Field(None, description="Custom fields")
    output_format: Optional[str] = Field(None, description="Output format")
    dpi: Optional[int] = Field(None, ge=72, le=1200, description="DPI")

class GenerateBatchCodeRequest(GreenLangBase):
    """Request to generate a batch code."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    operator_id: str = Field(..., min_length=1, description="EUDR operator ID")
    commodity: str = Field(..., min_length=1, description="EUDR commodity")
    year: int = Field(..., ge=2020, le=2100, description="Production year")
    count: int = Field(default=1, ge=1, le=10000, description="Number of codes")
    facility_id: Optional[str] = Field(None, description="Facility ID")
    prefix_format: Optional[str] = Field(None, description="Prefix format override")
    check_digit_algorithm: Optional[str] = Field(None, description="Check digit algo")

class BuildVerificationURLRequest(GreenLangBase):
    """Request to build a verification URL for a QR code."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., min_length=1, description="QR code ID")
    operator_id: str = Field(..., min_length=1, description="EUDR operator ID")
    base_url: Optional[str] = Field(None, description="Base URL override")
    use_short_url: Optional[bool] = Field(None, description="Generate short URL")
    ttl_years: Optional[int] = Field(None, ge=1, le=25, description="Token TTL years")

class SignCodeRequest(GreenLangBase):
    """Request to sign a QR code with HMAC."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., min_length=1, description="QR code ID")
    data_hash: str = Field(..., min_length=64, description="Data hash to sign")
    key_id: Optional[str] = Field(None, description="Signing key identifier")

class RecordScanRequest(GreenLangBase):
    """Request to record a QR code scan event."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., min_length=1, description="Scanned QR code ID")
    scanner_ip: Optional[str] = Field(None, description="Scanner IP (hashed)")
    scanner_user_agent: Optional[str] = Field(None, description="User agent")
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Latitude")
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0, description="Longitude")
    hmac_token: Optional[str] = Field(None, description="HMAC token from URL")

class SubmitBulkJobRequest(GreenLangBase):
    """Request to submit a bulk QR code generation job."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    operator_id: str = Field(..., min_length=1, description="EUDR operator ID")
    total_codes: int = Field(..., ge=1, description="Number of codes to generate")
    content_type: Optional[str] = Field(None, description="Content type")
    commodity: Optional[str] = Field(None, description="EUDR commodity")
    error_correction: Optional[str] = Field(None, description="Error correction level")
    output_format: Optional[str] = Field(None, description="Output format")
    payload_template: Optional[Dict[str, Any]] = Field(None, description="Payload template")
    worker_count: Optional[int] = Field(None, ge=1, le=64, description="Worker count")

class ActivateCodeRequest(GreenLangBase):
    """Request to activate a QR code for scanning."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., min_length=1, description="QR code ID")
    performed_by: Optional[str] = Field(None, description="User performing activation")
    reason: Optional[str] = Field(None, description="Activation reason")

class DeactivateCodeRequest(GreenLangBase):
    """Request to deactivate a QR code (temporarily)."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., min_length=1, description="QR code ID")
    reason: str = Field(..., min_length=1, description="Deactivation reason")
    performed_by: Optional[str] = Field(None, description="User performing deactivation")

class RevokeCodeRequest(GreenLangBase):
    """Request to permanently revoke a QR code."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., min_length=1, description="QR code ID")
    reason: str = Field(..., min_length=1, description="Revocation reason")
    performed_by: Optional[str] = Field(None, description="User performing revocation")

class ReprintCodeRequest(GreenLangBase):
    """Request to reprint a QR code label."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., min_length=1, description="QR code ID")
    template: Optional[str] = Field(None, description="Label template override")
    output_format: Optional[str] = Field(None, description="Output format override")
    performed_by: Optional[str] = Field(None, description="User requesting reprint")

class SearchCodesRequest(GreenLangBase):
    """Request to search QR codes by criteria."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    operator_id: Optional[str] = Field(None, description="Filter by operator ID")
    commodity: Optional[str] = Field(None, description="Filter by commodity")
    status: Optional[str] = Field(None, description="Filter by status")
    compliance_status: Optional[str] = Field(None, description="Filter by compliance")
    created_after: Optional[datetime] = Field(None, description="Created after date")
    created_before: Optional[datetime] = Field(None, description="Created before date")
    batch_code: Optional[str] = Field(None, description="Filter by batch code")
    limit: int = Field(default=50, ge=1, le=1000, description="Max results")
    offset: int = Field(default=0, ge=0, description="Result offset")

class GetScanHistoryRequest(GreenLangBase):
    """Request to get scan history for a QR code."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., min_length=1, description="QR code ID")
    outcome: Optional[str] = Field(None, description="Filter by scan outcome")
    scanned_after: Optional[datetime] = Field(None, description="Scanned after date")
    scanned_before: Optional[datetime] = Field(None, description="Scanned before date")
    limit: int = Field(default=50, ge=1, le=1000, description="Max results")
    offset: int = Field(default=0, ge=0, description="Result offset")

class ValidateCodeRequest(GreenLangBase):
    """Request to validate a QR code image (decode and verify)."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., min_length=1, description="QR code ID")
    image_data_hash: str = Field(..., min_length=64, description="Image data hash")

# =============================================================================
# Response Models
# =============================================================================

class QRCodeResponse(GreenLangBase):
    """Response after generating a QR code."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., description="Generated QR code ID")
    status: str = Field(default="success", description="Operation status")
    qr_code: Optional[QRCodeRecord] = Field(None, description="QR code record")
    verification_url: Optional[str] = Field(None, description="Verification URL")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: Optional[str] = Field(None, description="Provenance hash")

class PayloadResponse(GreenLangBase):
    """Response after composing a payload."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    payload_id: str = Field(..., description="Payload ID")
    status: str = Field(default="success", description="Operation status")
    payload: Optional[DataPayload] = Field(None, description="Composed payload")
    size_bytes: int = Field(default=0, description="Payload size in bytes")
    compressed: bool = Field(default=False, description="Whether compressed")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: Optional[str] = Field(None, description="Provenance hash")

class LabelResponse(GreenLangBase):
    """Response after rendering a label."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    label_id: str = Field(..., description="Label ID")
    status: str = Field(default="success", description="Operation status")
    label: Optional[LabelRecord] = Field(None, description="Label record")
    file_size_bytes: Optional[int] = Field(None, description="Output file size")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: Optional[str] = Field(None, description="Provenance hash")

class BatchCodeResponse(GreenLangBase):
    """Response after generating batch codes."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    status: str = Field(default="success", description="Operation status")
    batch_codes: List[BatchCode] = Field(default_factory=list, description="Generated codes")
    count: int = Field(default=0, description="Number of codes generated")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: Optional[str] = Field(None, description="Provenance hash")

class VerificationURLResponse(GreenLangBase):
    """Response after building a verification URL."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    url_id: str = Field(..., description="URL record ID")
    status: str = Field(default="success", description="Operation status")
    verification_url: Optional[VerificationURL] = Field(None, description="URL record")
    full_url: str = Field(default="", description="Complete verification URL")
    short_url: Optional[str] = Field(None, description="Shortened URL")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: Optional[str] = Field(None, description="Provenance hash")

class SignatureResponse(GreenLangBase):
    """Response after signing a QR code."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    signature_id: str = Field(..., description="Signature record ID")
    status: str = Field(default="success", description="Operation status")
    signature: Optional[SignatureRecord] = Field(None, description="Signature record")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: Optional[str] = Field(None, description="Provenance hash")

class ScanResponse(GreenLangBase):
    """Response after recording a scan event."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    scan_id: str = Field(..., description="Scan event ID")
    status: str = Field(default="success", description="Operation status")
    outcome: str = Field(default="verified", description="Scan outcome")
    scan_event: Optional[ScanEvent] = Field(None, description="Scan event record")
    counterfeit_risk: str = Field(default="low", description="Counterfeit risk level")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: Optional[str] = Field(None, description="Provenance hash")

class BulkJobResponse(GreenLangBase):
    """Response after submitting or querying a bulk job."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    job_id: str = Field(..., description="Bulk job ID")
    status: str = Field(default="queued", description="Job status")
    bulk_job: Optional[BulkJob] = Field(None, description="Bulk job record")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: Optional[str] = Field(None, description="Provenance hash")

class ActivateResponse(GreenLangBase):
    """Response after activating a QR code."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., description="QR code ID")
    status: str = Field(default="success", description="Operation status")
    previous_status: str = Field(default="created", description="Previous status")
    new_status: str = Field(default="active", description="New status")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: Optional[str] = Field(None, description="Provenance hash")

class DeactivateResponse(GreenLangBase):
    """Response after deactivating a QR code."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., description="QR code ID")
    status: str = Field(default="success", description="Operation status")
    previous_status: str = Field(default="active", description="Previous status")
    new_status: str = Field(default="deactivated", description="New status")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: Optional[str] = Field(None, description="Provenance hash")

class RevokeResponse(GreenLangBase):
    """Response after revoking a QR code."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., description="QR code ID")
    status: str = Field(default="success", description="Operation status")
    previous_status: str = Field(..., description="Previous status")
    new_status: str = Field(default="revoked", description="New status")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: Optional[str] = Field(None, description="Provenance hash")

class ReprintResponse(GreenLangBase):
    """Response after reprinting a QR code label."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., description="QR code ID")
    status: str = Field(default="success", description="Operation status")
    reprint_count: int = Field(default=0, description="Total reprint count")
    label: Optional[LabelRecord] = Field(None, description="Reprinted label record")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: Optional[str] = Field(None, description="Provenance hash")

class SearchResponse(GreenLangBase):
    """Response after searching QR codes."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    status: str = Field(default="success", description="Operation status")
    codes: List[QRCodeRecord] = Field(default_factory=list, description="Matching codes")
    total_count: int = Field(default=0, description="Total matching records")
    limit: int = Field(default=50, description="Result limit used")
    offset: int = Field(default=0, description="Result offset used")
    processing_time_ms: float = Field(default=0.0, description="Processing time")

class ScanHistoryResponse(GreenLangBase):
    """Response after querying scan history."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    code_id: str = Field(..., description="QR code ID")
    status: str = Field(default="success", description="Operation status")
    scans: List[ScanEvent] = Field(default_factory=list, description="Scan events")
    total_count: int = Field(default=0, description="Total scan events")
    limit: int = Field(default=50, description="Result limit used")
    offset: int = Field(default=0, description="Result offset used")
    processing_time_ms: float = Field(default=0.0, description="Processing time")

class HealthResponse(GreenLangBase):
    """Health check response for QR Code Generator Agent."""

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    status: str = Field(default="healthy", description="Service health status")
    agent_id: str = Field(default="GL-EUDR-QRG-014", description="Agent identifier")
    version: str = Field(default=VERSION, description="Service version")
    database_connected: bool = Field(default=False, description="Database connectivity")
    redis_connected: bool = Field(default=False, description="Redis connectivity")
    active_bulk_jobs: int = Field(default=0, description="Active bulk jobs")
    total_codes_generated: int = Field(default=0, description="Total codes generated")
    total_scans_recorded: int = Field(default=0, description="Total scans recorded")
    uptime_seconds: float = Field(default=0.0, description="Service uptime")
    checked_at: datetime = Field(default_factory=utcnow, description="Check timestamp")
