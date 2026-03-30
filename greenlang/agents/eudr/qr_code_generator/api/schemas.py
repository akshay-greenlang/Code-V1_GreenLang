# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-014 QR Code Generator

Pydantic v2 request/response models for all QR Code Generator REST API
endpoints. Organized by domain: QR generation, payload composition,
label rendering, batch codes, verification, anti-counterfeiting,
bulk generation, lifecycle management, and health.

All models use ``ConfigDict(from_attributes=True)`` for ORM compatibility
and ``Field()`` with descriptions for OpenAPI documentation.

All hash fields use deterministic SHA-256 algorithms required by
EUDR Article 14 for five-year audit trail compliance.

Model Count: 80+ schemas covering 37 endpoints.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014, Section 7.4
Agent ID: GL-EUDR-QRG-014
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator

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

class ErrorCorrectionSchema(str, Enum):
    """Error correction level per ISO/IEC 18004."""

    L = "L"
    M = "M"
    Q = "Q"
    H = "H"

class OutputFormatSchema(str, Enum):
    """Output image format for generated QR codes."""

    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    ZPL = "zpl"
    EPS = "eps"

class ContentTypeSchema(str, Enum):
    """Payload content type for QR code data."""

    FULL_TRACEABILITY = "full_traceability"
    COMPACT_VERIFICATION = "compact_verification"
    CONSUMER_SUMMARY = "consumer_summary"
    BATCH_IDENTIFIER = "batch_identifier"
    BLOCKCHAIN_ANCHOR = "blockchain_anchor"

class SymbologyTypeSchema(str, Enum):
    """Barcode symbology type."""

    QR_CODE = "qr_code"
    MICRO_QR = "micro_qr"
    DATA_MATRIX = "data_matrix"
    GS1_DIGITAL_LINK = "gs1_digital_link"

class LabelTemplateSchema(str, Enum):
    """Label template types."""

    PRODUCT_LABEL = "product_label"
    SHIPPING_LABEL = "shipping_label"
    PALLET_LABEL = "pallet_label"
    CONTAINER_LABEL = "container_label"
    CONSUMER_LABEL = "consumer_label"

class CheckDigitAlgorithmSchema(str, Enum):
    """Check digit algorithm for batch codes."""

    LUHN = "luhn"
    ISO7064_MOD11_10 = "iso7064_mod11_10"
    CRC8 = "crc8"

class CodeStatusSchema(str, Enum):
    """Lifecycle status of a generated QR code."""

    CREATED = "created"
    ACTIVE = "active"
    DEACTIVATED = "deactivated"
    REVOKED = "revoked"
    EXPIRED = "expired"

class ScanOutcomeSchema(str, Enum):
    """Outcome of a QR code scan verification event."""

    VERIFIED = "verified"
    COUNTERFEIT_SUSPECTED = "counterfeit_suspected"
    EXPIRED_CODE = "expired_code"
    REVOKED_CODE = "revoked_code"
    ERROR = "error"

class CounterfeitRiskSchema(str, Enum):
    """Risk level for counterfeit detection."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class BulkJobStatusSchema(str, Enum):
    """Status of a bulk QR code generation job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ComplianceStatusSchema(str, Enum):
    """EUDR compliance status."""

    COMPLIANT = "compliant"
    PENDING = "pending"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"

class PayloadEncodingSchema(str, Enum):
    """Encoding format for QR code payload data."""

    UTF8 = "utf8"
    BASE64 = "base64"
    ZLIB_BASE64 = "zlib_base64"

class EUDRCommoditySchema(str, Enum):
    """EUDR-regulated commodity types per EU 2023/1115 Article 1."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"

# =============================================================================
# Shared Models
# =============================================================================

class ProvenanceInfo(GreenLangBase):
    """Provenance tracking metadata for audit trails.

    Attributes:
        provenance_hash: SHA-256 hash of the operation data.
        algorithm: Hash algorithm used (always sha256).
        created_at: Timestamp when the provenance was recorded.
    """

    model_config = ConfigDict(from_attributes=True)

    provenance_hash: str = Field(
        ..., description="SHA-256 hash of the operation data"
    )
    algorithm: str = Field(
        default="sha256", description="Hash algorithm used"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Provenance timestamp"
    )

class PaginatedMeta(GreenLangBase):
    """Pagination metadata for list responses.

    Attributes:
        total: Total number of records matching the query.
        limit: Maximum records per page.
        offset: Number of records skipped.
        has_more: Whether more records exist beyond this page.
    """

    model_config = ConfigDict(from_attributes=True)

    total: int = Field(..., ge=0, description="Total matching records")
    limit: int = Field(..., ge=1, description="Records per page")
    offset: int = Field(..., ge=0, description="Records skipped")
    has_more: bool = Field(..., description="More records exist")

# =============================================================================
# QR Code Generation Schemas
# =============================================================================

class GenerateQRRequest(GreenLangBase):
    """Request to generate a single QR code.

    Attributes:
        operator_id: EUDR operator identifier.
        payload_data: Data to encode in the QR code.
        content_type: Payload content type.
        version: QR version (auto or 1-40).
        error_correction: Error correction level.
        output_format: Output image format.
        symbology: Barcode symbology type.
        module_size: Pixel size of each QR module.
        quiet_zone: Quiet zone width in modules.
        dpi: Output resolution in DPI.
        embed_logo: Whether to embed a centre logo.
        commodity: EUDR-regulated commodity type.
        dds_reference: Due Diligence Statement reference number.
        compliance_status: EUDR compliance status.
    """

    model_config = ConfigDict(from_attributes=True)

    operator_id: str = Field(
        ..., min_length=1, max_length=255,
        description="EUDR operator identifier",
    )
    payload_data: Dict[str, Any] = Field(
        ..., description="Data to encode in the QR code"
    )
    content_type: Optional[ContentTypeSchema] = Field(
        None, description="Payload content type"
    )
    version: Optional[str] = Field(
        None, description="QR version (auto or 1-40)"
    )
    error_correction: Optional[ErrorCorrectionSchema] = Field(
        None, description="Error correction level"
    )
    output_format: Optional[OutputFormatSchema] = Field(
        None, description="Output image format"
    )
    symbology: Optional[SymbologyTypeSchema] = Field(
        None, description="Barcode symbology type"
    )
    module_size: Optional[int] = Field(
        None, ge=1, le=100, description="Module pixel size"
    )
    quiet_zone: Optional[int] = Field(
        None, ge=0, le=20, description="Quiet zone width in modules"
    )
    dpi: Optional[int] = Field(
        None, ge=72, le=1200, description="Output DPI"
    )
    embed_logo: Optional[bool] = Field(
        None, description="Embed centre logo"
    )
    commodity: Optional[EUDRCommoditySchema] = Field(
        None, description="EUDR commodity type"
    )
    dds_reference: Optional[str] = Field(
        None, max_length=255, description="DDS reference number"
    )
    compliance_status: Optional[ComplianceStatusSchema] = Field(
        None, description="EUDR compliance status"
    )

class GenerateDataMatrixRequest(GreenLangBase):
    """Request to generate a Data Matrix code.

    Attributes:
        operator_id: EUDR operator identifier.
        payload_data: Data to encode in the Data Matrix code.
        content_type: Payload content type.
        output_format: Output image format.
        module_size: Module pixel size.
        quiet_zone: Quiet zone width in modules.
        dpi: Output DPI.
        commodity: EUDR commodity type.
        dds_reference: DDS reference number.
    """

    model_config = ConfigDict(from_attributes=True)

    operator_id: str = Field(
        ..., min_length=1, max_length=255,
        description="EUDR operator identifier",
    )
    payload_data: Dict[str, Any] = Field(
        ..., description="Data to encode"
    )
    content_type: Optional[ContentTypeSchema] = Field(
        None, description="Payload content type"
    )
    output_format: Optional[OutputFormatSchema] = Field(
        None, description="Output image format"
    )
    module_size: Optional[int] = Field(
        None, ge=1, le=100, description="Module pixel size"
    )
    quiet_zone: Optional[int] = Field(
        None, ge=0, le=20, description="Quiet zone width"
    )
    dpi: Optional[int] = Field(
        None, ge=72, le=1200, description="Output DPI"
    )
    commodity: Optional[EUDRCommoditySchema] = Field(
        None, description="EUDR commodity type"
    )
    dds_reference: Optional[str] = Field(
        None, max_length=255, description="DDS reference number"
    )

class QRCodeDetailResponse(GreenLangBase):
    """Detailed QR code record response.

    Attributes:
        code_id: Unique QR code identifier.
        version: QR code version used.
        error_correction: Error correction level applied.
        symbology: Barcode symbology type.
        output_format: Output image format.
        module_size: Module pixel size.
        quiet_zone: Quiet zone width.
        dpi: Output DPI.
        payload_hash: SHA-256 hash of the payload.
        payload_size_bytes: Payload size in bytes.
        content_type: Payload content type.
        encoding: Payload encoding format.
        image_width_px: Image width in pixels.
        image_height_px: Image height in pixels.
        logo_embedded: Whether logo was embedded.
        quality_grade: ISO/IEC 15416 quality grade.
        operator_id: EUDR operator identifier.
        commodity: EUDR commodity type.
        compliance_status: EUDR compliance status.
        dds_reference: DDS reference number.
        batch_code: Associated batch code.
        verification_url: Associated verification URL.
        blockchain_anchor_hash: Blockchain anchor hash.
        status: Current lifecycle status.
        reprint_count: Number of reprints.
        scan_count: Total scan count.
        created_at: Generation timestamp.
        activated_at: Activation timestamp.
        expires_at: Expiry timestamp.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(..., description="Unique QR code identifier")
    version: str = Field(default="auto", description="QR code version")
    error_correction: str = Field(default="M", description="Error correction level")
    symbology: str = Field(default="qr_code", description="Barcode symbology")
    output_format: str = Field(default="png", description="Output image format")
    module_size: int = Field(default=10, description="Module pixel size")
    quiet_zone: int = Field(default=4, description="Quiet zone width")
    dpi: int = Field(default=300, description="Output DPI")
    payload_hash: str = Field(..., description="SHA-256 hash of payload")
    payload_size_bytes: int = Field(..., description="Payload size in bytes")
    content_type: str = Field(
        default="compact_verification", description="Content type"
    )
    encoding: str = Field(default="utf8", description="Payload encoding")
    image_width_px: Optional[int] = Field(None, description="Image width")
    image_height_px: Optional[int] = Field(None, description="Image height")
    logo_embedded: bool = Field(default=False, description="Logo embedded")
    quality_grade: Optional[str] = Field(None, description="Quality grade")
    operator_id: str = Field(..., description="EUDR operator ID")
    commodity: Optional[str] = Field(None, description="EUDR commodity")
    compliance_status: str = Field(
        default="pending", description="Compliance status"
    )
    dds_reference: Optional[str] = Field(None, description="DDS reference")
    batch_code: Optional[str] = Field(None, description="Batch code")
    verification_url: Optional[str] = Field(None, description="Verification URL")
    blockchain_anchor_hash: Optional[str] = Field(
        None, description="Blockchain anchor hash"
    )
    status: str = Field(default="created", description="Lifecycle status")
    reprint_count: int = Field(default=0, description="Reprint count")
    scan_count: int = Field(default=0, description="Scan count")
    created_at: datetime = Field(
        default_factory=utcnow, description="Generation timestamp"
    )
    activated_at: Optional[datetime] = Field(
        None, description="Activation timestamp"
    )
    expires_at: Optional[datetime] = Field(None, description="Expiry timestamp")
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class GenerateQRResponse(GreenLangBase):
    """Response after generating a QR code.

    Attributes:
        code_id: Generated QR code identifier.
        status: Operation status.
        qr_code: Detailed QR code record.
        verification_url: Generated verification URL.
        processing_time_ms: Processing time in milliseconds.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(
        default_factory=_new_id, description="Generated QR code ID"
    )
    status: str = Field(default="success", description="Operation status")
    qr_code: Optional[QRCodeDetailResponse] = Field(
        None, description="QR code record"
    )
    verification_url: Optional[str] = Field(
        None, description="Verification URL"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in ms"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class GenerateDataMatrixResponse(GreenLangBase):
    """Response after generating a Data Matrix code.

    Attributes:
        code_id: Generated Data Matrix code identifier.
        status: Operation status.
        symbology: Symbology type (always data_matrix).
        qr_code: Detailed code record.
        processing_time_ms: Processing time in milliseconds.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(
        default_factory=_new_id, description="Generated code ID"
    )
    status: str = Field(default="success", description="Operation status")
    symbology: str = Field(
        default="data_matrix", description="Symbology type"
    )
    qr_code: Optional[QRCodeDetailResponse] = Field(
        None, description="Code record"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in ms"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class QRImageResponse(GreenLangBase):
    """Response for QR code image download.

    Attributes:
        code_id: QR code identifier.
        format: Image format returned.
        content_type: MIME content type.
        file_size_bytes: Image file size.
        image_data_base64: Base64-encoded image data.
        image_hash: SHA-256 hash of image data.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(..., description="QR code identifier")
    format: str = Field(default="png", description="Image format")
    content_type: str = Field(
        default="image/png", description="MIME content type"
    )
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="Image file size"
    )
    image_data_base64: Optional[str] = Field(
        None, description="Base64-encoded image data"
    )
    image_hash: Optional[str] = Field(
        None, description="SHA-256 hash of image data"
    )

# =============================================================================
# Payload Schemas
# =============================================================================

class ComposePayloadRequest(GreenLangBase):
    """Request to compose a data payload for QR encoding.

    Attributes:
        operator_id: EUDR operator identifier.
        content_type: Payload content type.
        data: Raw payload data to compose.
        compress: Enable zlib compression.
        encrypt: Enable AES-256-GCM encryption.
        commodity: EUDR commodity type.
        dds_reference: DDS reference number.
        compliance_status: EUDR compliance status.
        origin_country: Country of origin (ISO 3166-1 alpha-2).
    """

    model_config = ConfigDict(from_attributes=True)

    operator_id: str = Field(
        ..., min_length=1, max_length=255,
        description="EUDR operator identifier",
    )
    content_type: Optional[ContentTypeSchema] = Field(
        None, description="Payload content type"
    )
    data: Dict[str, Any] = Field(
        ..., description="Raw payload data to compose"
    )
    compress: Optional[bool] = Field(None, description="Enable compression")
    encrypt: Optional[bool] = Field(None, description="Enable encryption")
    commodity: Optional[EUDRCommoditySchema] = Field(
        None, description="EUDR commodity type"
    )
    dds_reference: Optional[str] = Field(
        None, max_length=255, description="DDS reference"
    )
    compliance_status: Optional[ComplianceStatusSchema] = Field(
        None, description="Compliance status"
    )
    origin_country: Optional[str] = Field(
        None, min_length=2, max_length=2,
        description="Origin country (ISO 3166-1 alpha-2)",
    )

class ValidatePayloadRequest(GreenLangBase):
    """Request to validate a payload against its schema.

    Attributes:
        content_type: Content type schema to validate against.
        data: Payload data to validate.
        strict: Enable strict validation mode.
    """

    model_config = ConfigDict(from_attributes=True)

    content_type: ContentTypeSchema = Field(
        ..., description="Content type schema to validate against"
    )
    data: Dict[str, Any] = Field(
        ..., description="Payload data to validate"
    )
    strict: bool = Field(
        default=True, description="Enable strict validation"
    )

class ComposePayloadResponse(GreenLangBase):
    """Response after composing a data payload.

    Attributes:
        payload_id: Unique payload identifier.
        status: Operation status.
        content_type: Content type of the payload.
        encoding: Encoding format applied.
        compressed: Whether compression was applied.
        encrypted: Whether encryption was applied.
        compression_ratio: Compression ratio achieved.
        size_bytes: Payload size in bytes.
        payload_hash: SHA-256 hash of the encoded payload.
        payload_version: Payload schema version.
        processing_time_ms: Processing time in milliseconds.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    payload_id: str = Field(
        default_factory=_new_id, description="Unique payload ID"
    )
    status: str = Field(default="success", description="Operation status")
    content_type: str = Field(
        default="compact_verification", description="Content type"
    )
    encoding: str = Field(default="utf8", description="Encoding format")
    compressed: bool = Field(default=False, description="Compression applied")
    encrypted: bool = Field(default=False, description="Encryption applied")
    compression_ratio: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Compression ratio"
    )
    size_bytes: int = Field(default=0, ge=0, description="Payload size")
    payload_hash: Optional[str] = Field(
        None, description="SHA-256 hash of payload"
    )
    payload_version: str = Field(
        default="1.0", description="Payload schema version"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class ValidatePayloadResponse(GreenLangBase):
    """Response after validating a payload.

    Attributes:
        valid: Whether the payload is valid.
        content_type: Content type validated against.
        errors: List of validation errors (if any).
        warnings: List of validation warnings.
        processing_time_ms: Processing time in ms.
    """

    model_config = ConfigDict(from_attributes=True)

    valid: bool = Field(..., description="Whether payload is valid")
    content_type: str = Field(..., description="Content type validated")
    errors: List[str] = Field(
        default_factory=list, description="Validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Validation warnings"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time"
    )

class PayloadDetailResponse(GreenLangBase):
    """Detailed payload record response.

    Attributes:
        payload_id: Unique payload identifier.
        content_type: Content type of the payload.
        encoding: Encoding format.
        compressed: Whether compressed.
        encrypted: Whether encrypted.
        compression_ratio: Compression ratio.
        size_bytes: Payload size.
        payload_hash: SHA-256 hash.
        payload_version: Schema version.
        operator_id: Operator identifier.
        commodity: Commodity type.
        dds_reference: DDS reference.
        compliance_status: Compliance status.
        origin_country: Origin country code.
        created_at: Creation timestamp.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    payload_id: str = Field(..., description="Payload identifier")
    content_type: str = Field(..., description="Content type")
    encoding: str = Field(default="utf8", description="Encoding format")
    compressed: bool = Field(default=False, description="Compressed")
    encrypted: bool = Field(default=False, description="Encrypted")
    compression_ratio: Optional[float] = Field(
        None, description="Compression ratio"
    )
    size_bytes: Optional[int] = Field(None, description="Payload size")
    payload_hash: Optional[str] = Field(None, description="SHA-256 hash")
    payload_version: str = Field(default="1.0", description="Schema version")
    operator_id: str = Field(..., description="Operator identifier")
    commodity: Optional[str] = Field(None, description="Commodity type")
    dds_reference: Optional[str] = Field(None, description="DDS reference")
    compliance_status: str = Field(
        default="pending", description="Compliance status"
    )
    origin_country: Optional[str] = Field(None, description="Origin country")
    created_at: datetime = Field(
        default_factory=utcnow, description="Creation timestamp"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class PayloadSchemaItem(GreenLangBase):
    """A single payload schema definition.

    Attributes:
        content_type: Content type this schema defines.
        description: Human-readable schema description.
        required_fields: List of required field names.
        optional_fields: List of optional field names.
        version: Schema version.
    """

    model_config = ConfigDict(from_attributes=True)

    content_type: str = Field(..., description="Content type")
    description: str = Field(..., description="Schema description")
    required_fields: List[str] = Field(
        default_factory=list, description="Required fields"
    )
    optional_fields: List[str] = Field(
        default_factory=list, description="Optional fields"
    )
    version: str = Field(default="1.0", description="Schema version")

class PayloadSchemasResponse(GreenLangBase):
    """Response listing all available payload schemas.

    Attributes:
        schemas: List of payload schema definitions.
        total: Total number of schemas.
    """

    model_config = ConfigDict(from_attributes=True)

    schemas: List[PayloadSchemaItem] = Field(
        default_factory=list, description="Payload schemas"
    )
    total: int = Field(default=0, ge=0, description="Total schemas")

# =============================================================================
# Label Schemas
# =============================================================================

class GenerateLabelRequest(GreenLangBase):
    """Request to generate a single label with QR code.

    Attributes:
        code_id: QR code identifier to embed in the label.
        operator_id: EUDR operator identifier.
        template: Label template to use.
        product_name: Product name for the label.
        batch_code: Batch code to display.
        custom_fields: Additional custom fields.
        output_format: Output format (png, svg, pdf).
        dpi: Output DPI.
        font: Font family.
        font_size: Font size in points.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(
        ..., min_length=1, max_length=255,
        description="QR code identifier",
    )
    operator_id: str = Field(
        ..., min_length=1, max_length=255,
        description="EUDR operator identifier",
    )
    template: Optional[LabelTemplateSchema] = Field(
        None, description="Label template"
    )
    product_name: Optional[str] = Field(
        None, max_length=500, description="Product name"
    )
    batch_code: Optional[str] = Field(
        None, max_length=255, description="Batch code"
    )
    custom_fields: Optional[Dict[str, str]] = Field(
        None, description="Custom fields"
    )
    output_format: Optional[OutputFormatSchema] = Field(
        None, description="Output format"
    )
    dpi: Optional[int] = Field(
        None, ge=72, le=1200, description="Output DPI"
    )
    font: Optional[str] = Field(
        None, max_length=100, description="Font family"
    )
    font_size: Optional[int] = Field(
        None, ge=4, le=72, description="Font size in points"
    )

class BatchLabelRequest(GreenLangBase):
    """Request to generate labels in batch.

    Attributes:
        labels: List of label generation requests (max 500).
        template: Shared template for all labels.
        output_format: Shared output format.
        dpi: Shared DPI setting.
    """

    model_config = ConfigDict(from_attributes=True)

    labels: List[GenerateLabelRequest] = Field(
        ..., min_length=1, max_length=500,
        description="Label requests (max 500)",
    )
    template: Optional[LabelTemplateSchema] = Field(
        None, description="Shared template for all labels"
    )
    output_format: Optional[OutputFormatSchema] = Field(
        None, description="Shared output format"
    )
    dpi: Optional[int] = Field(
        None, ge=72, le=1200, description="Shared DPI"
    )

class LabelDetailResponse(GreenLangBase):
    """Detailed label record response.

    Attributes:
        label_id: Unique label identifier.
        code_id: Associated QR code identifier.
        template: Template used.
        compliance_status: EUDR compliance status.
        compliance_color_hex: Status colour hex code.
        output_format: Output format.
        dpi: DPI setting.
        width_mm: Label width in mm.
        height_mm: Label height in mm.
        file_size_bytes: Output file size.
        image_data_hash: SHA-256 hash of rendered image.
        operator_id: Operator identifier.
        commodity: Commodity type.
        product_name: Product name.
        batch_code: Batch code.
        created_at: Creation timestamp.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    label_id: str = Field(
        default_factory=_new_id, description="Label identifier"
    )
    code_id: str = Field(..., description="Associated QR code ID")
    template: str = Field(
        default="product_label", description="Template used"
    )
    compliance_status: str = Field(
        default="pending", description="Compliance status"
    )
    compliance_color_hex: str = Field(
        default="#F57F17", description="Status colour"
    )
    output_format: str = Field(default="pdf", description="Output format")
    dpi: int = Field(default=300, description="DPI")
    width_mm: Optional[float] = Field(None, description="Label width mm")
    height_mm: Optional[float] = Field(None, description="Label height mm")
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="File size"
    )
    image_data_hash: Optional[str] = Field(
        None, description="SHA-256 hash of image"
    )
    operator_id: str = Field(..., description="Operator ID")
    commodity: Optional[str] = Field(None, description="Commodity")
    product_name: Optional[str] = Field(None, description="Product name")
    batch_code: Optional[str] = Field(None, description="Batch code")
    created_at: datetime = Field(
        default_factory=utcnow, description="Creation timestamp"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class GenerateLabelResponse(GreenLangBase):
    """Response after generating a label.

    Attributes:
        label_id: Generated label identifier.
        status: Operation status.
        label: Detailed label record.
        file_size_bytes: Output file size.
        processing_time_ms: Processing time in ms.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    label_id: str = Field(
        default_factory=_new_id, description="Generated label ID"
    )
    status: str = Field(default="success", description="Operation status")
    label: Optional[LabelDetailResponse] = Field(
        None, description="Label record"
    )
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="File size"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class BatchLabelResponse(GreenLangBase):
    """Response after batch label generation.

    Attributes:
        status: Operation status.
        labels: List of generated label responses.
        total_generated: Number of labels generated.
        total_failed: Number of labels that failed.
        processing_time_ms: Total processing time.
    """

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(default="success", description="Operation status")
    labels: List[GenerateLabelResponse] = Field(
        default_factory=list, description="Generated labels"
    )
    total_generated: int = Field(
        default=0, ge=0, description="Labels generated"
    )
    total_failed: int = Field(default=0, ge=0, description="Labels failed")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time"
    )

class TemplateDetailResponse(GreenLangBase):
    """Detailed template definition response.

    Attributes:
        template_id: Unique template identifier.
        name: Template name.
        template_type: Label template type.
        description: Template description.
        width_mm: Template width in mm.
        height_mm: Template height in mm.
        qr_size_mm: QR code size in mm.
        version: Template version.
        is_active: Whether template is active.
        created_at: Creation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    template_id: str = Field(
        default_factory=_new_id, description="Template identifier"
    )
    name: str = Field(..., description="Template name")
    template_type: str = Field(..., description="Template type")
    description: Optional[str] = Field(None, description="Description")
    width_mm: float = Field(..., gt=0, description="Width mm")
    height_mm: float = Field(..., gt=0, description="Height mm")
    qr_size_mm: float = Field(default=20.0, gt=0, description="QR size mm")
    version: str = Field(default="1.0", description="Template version")
    is_active: bool = Field(default=True, description="Is active")
    created_at: datetime = Field(
        default_factory=utcnow, description="Creation timestamp"
    )

class TemplateListResponse(GreenLangBase):
    """Response listing all label templates.

    Attributes:
        templates: List of template definitions.
        total: Total number of templates.
    """

    model_config = ConfigDict(from_attributes=True)

    templates: List[TemplateDetailResponse] = Field(
        default_factory=list, description="Templates"
    )
    total: int = Field(default=0, ge=0, description="Total templates")

# =============================================================================
# Batch Code Schemas
# =============================================================================

class GenerateBatchCodesRequest(GreenLangBase):
    """Request to generate batch codes.

    Attributes:
        operator_id: EUDR operator identifier.
        commodity: EUDR commodity type.
        year: Production or import year.
        count: Number of codes to generate.
        facility_id: Facility identifier.
        prefix_format: Prefix format override.
        check_digit_algorithm: Check digit algorithm.
    """

    model_config = ConfigDict(from_attributes=True)

    operator_id: str = Field(
        ..., min_length=1, max_length=255,
        description="EUDR operator identifier",
    )
    commodity: EUDRCommoditySchema = Field(
        ..., description="EUDR commodity type"
    )
    year: int = Field(
        ..., ge=2020, le=2100, description="Production year"
    )
    count: int = Field(
        default=1, ge=1, le=10000, description="Number of codes"
    )
    facility_id: Optional[str] = Field(
        None, max_length=255, description="Facility ID"
    )
    prefix_format: Optional[str] = Field(
        None, max_length=255, description="Prefix format override"
    )
    check_digit_algorithm: Optional[CheckDigitAlgorithmSchema] = Field(
        None, description="Check digit algorithm"
    )

class ReserveCodesRequest(GreenLangBase):
    """Request to reserve a range of batch codes.

    Attributes:
        operator_id: EUDR operator identifier.
        commodity: EUDR commodity type.
        year: Production year.
        count: Number of codes to reserve.
        facility_id: Facility identifier.
    """

    model_config = ConfigDict(from_attributes=True)

    operator_id: str = Field(
        ..., min_length=1, max_length=255,
        description="EUDR operator identifier",
    )
    commodity: EUDRCommoditySchema = Field(
        ..., description="EUDR commodity type"
    )
    year: int = Field(
        ..., ge=2020, le=2100, description="Production year"
    )
    count: int = Field(
        ..., ge=1, le=100000, description="Codes to reserve"
    )
    facility_id: Optional[str] = Field(
        None, max_length=255, description="Facility ID"
    )

class BatchCodeItem(GreenLangBase):
    """A single batch code record.

    Attributes:
        batch_code_id: Unique record identifier.
        code_value: Full batch code string.
        prefix: Batch code prefix.
        sequence_number: Sequence number.
        check_digit: Check digit.
        check_digit_algorithm: Algorithm used.
        operator_id: Operator identifier.
        commodity: Commodity type.
        year: Production year.
        facility_id: Facility identifier.
        status: Current status.
        created_at: Creation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_code_id: str = Field(
        default_factory=_new_id, description="Record identifier"
    )
    code_value: str = Field(..., description="Full batch code")
    prefix: str = Field(..., description="Batch code prefix")
    sequence_number: int = Field(..., ge=0, description="Sequence number")
    check_digit: str = Field(..., description="Check digit")
    check_digit_algorithm: str = Field(
        default="luhn", description="Algorithm used"
    )
    operator_id: str = Field(..., description="Operator ID")
    commodity: Optional[str] = Field(None, description="Commodity")
    year: int = Field(..., ge=2020, description="Production year")
    facility_id: Optional[str] = Field(None, description="Facility ID")
    status: str = Field(default="created", description="Status")
    created_at: datetime = Field(
        default_factory=utcnow, description="Creation timestamp"
    )

class GenerateBatchCodesResponse(GreenLangBase):
    """Response after generating batch codes.

    Attributes:
        status: Operation status.
        batch_codes: Generated batch code records.
        count: Number of codes generated.
        processing_time_ms: Processing time in ms.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(default="success", description="Operation status")
    batch_codes: List[BatchCodeItem] = Field(
        default_factory=list, description="Generated codes"
    )
    count: int = Field(default=0, ge=0, description="Codes generated")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class ReserveCodesResponse(GreenLangBase):
    """Response after reserving batch codes.

    Attributes:
        status: Operation status.
        reserved_count: Number of codes reserved.
        start_sequence: First reserved sequence number.
        end_sequence: Last reserved sequence number.
        prefix: Batch code prefix for the reserved range.
        processing_time_ms: Processing time.
    """

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(default="success", description="Operation status")
    reserved_count: int = Field(
        ..., ge=1, description="Codes reserved"
    )
    start_sequence: int = Field(
        ..., ge=0, description="First sequence number"
    )
    end_sequence: int = Field(
        ..., ge=0, description="Last sequence number"
    )
    prefix: str = Field(..., description="Batch code prefix")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time"
    )

class BatchCodeDetailResponse(GreenLangBase):
    """Detailed batch code response.

    Attributes:
        batch_code: Batch code record.
        associated_code_ids: QR codes linked to this batch.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_code: BatchCodeItem = Field(..., description="Batch code record")
    associated_code_ids: List[str] = Field(
        default_factory=list, description="Linked QR code IDs"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class CodeHierarchyResponse(GreenLangBase):
    """Response showing batch code hierarchy.

    Attributes:
        code_value: Batch code value.
        operator_id: Operator identifier.
        commodity: Commodity type.
        year: Production year.
        parent_codes: Parent batch codes.
        child_codes: Child batch codes.
        associated_qr_codes: Associated QR code identifiers.
        total_associations: Total association count.
    """

    model_config = ConfigDict(from_attributes=True)

    code_value: str = Field(..., description="Batch code value")
    operator_id: str = Field(..., description="Operator ID")
    commodity: Optional[str] = Field(None, description="Commodity")
    year: int = Field(..., ge=2020, description="Production year")
    parent_codes: List[str] = Field(
        default_factory=list, description="Parent batch codes"
    )
    child_codes: List[str] = Field(
        default_factory=list, description="Child batch codes"
    )
    associated_qr_codes: List[str] = Field(
        default_factory=list, description="Associated QR code IDs"
    )
    total_associations: int = Field(
        default=0, ge=0, description="Total associations"
    )

# =============================================================================
# Verification Schemas
# =============================================================================

class BuildURLRequest(GreenLangBase):
    """Request to build a verification URL.

    Attributes:
        code_id: QR code identifier.
        operator_id: EUDR operator identifier.
        base_url: Base URL override.
        use_short_url: Generate short URL.
        ttl_years: Token TTL in years.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(
        ..., min_length=1, max_length=255,
        description="QR code identifier",
    )
    operator_id: str = Field(
        ..., min_length=1, max_length=255,
        description="EUDR operator identifier",
    )
    base_url: Optional[str] = Field(
        None, max_length=2048, description="Base URL override"
    )
    use_short_url: Optional[bool] = Field(
        None, description="Generate short URL"
    )
    ttl_years: Optional[int] = Field(
        None, ge=1, le=25, description="Token TTL years"
    )

class VerifySignatureRequest(GreenLangBase):
    """Request to verify a QR code signature.

    Attributes:
        code_id: QR code identifier.
        signature_value: Hex-encoded signature to verify.
        data_hash: SHA-256 hash of the data that was signed.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(
        ..., min_length=1, max_length=255,
        description="QR code identifier",
    )
    signature_value: str = Field(
        ..., min_length=1, description="Hex-encoded signature"
    )
    data_hash: str = Field(
        ..., min_length=64, max_length=128,
        description="SHA-256 hash of signed data",
    )

    @field_validator("data_hash")
    @classmethod
    def validate_data_hash(cls, v: str) -> str:
        """Validate data_hash is a valid hex string."""
        v = v.strip().lower()
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError("data_hash must be a valid hexadecimal string")
        return v

class OfflineVerifyRequest(GreenLangBase):
    """Request for offline verification check.

    Attributes:
        code_id: QR code identifier.
        hmac_token: HMAC token from the verification URL.
        payload_hash: SHA-256 hash of the scanned payload.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(
        ..., min_length=1, max_length=255,
        description="QR code identifier",
    )
    hmac_token: str = Field(
        ..., min_length=1, description="HMAC token from URL"
    )
    payload_hash: Optional[str] = Field(
        None, min_length=64, max_length=128,
        description="SHA-256 hash of scanned payload",
    )

class BuildURLResponse(GreenLangBase):
    """Response after building a verification URL.

    Attributes:
        url_id: Unique URL record identifier.
        code_id: Associated QR code identifier.
        full_url: Complete verification URL.
        short_url: Shortened URL (if enabled).
        token: HMAC verification token.
        token_expires_at: Token expiry timestamp.
        processing_time_ms: Processing time.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    url_id: str = Field(
        default_factory=_new_id, description="URL record ID"
    )
    code_id: str = Field(..., description="QR code ID")
    full_url: str = Field(..., description="Complete verification URL")
    short_url: Optional[str] = Field(None, description="Shortened URL")
    token: str = Field(..., description="HMAC verification token")
    token_expires_at: Optional[datetime] = Field(
        None, description="Token expiry"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class VerifySignatureResponse(GreenLangBase):
    """Response after verifying a signature.

    Attributes:
        code_id: QR code identifier.
        valid: Whether the signature is valid.
        algorithm: Signing algorithm used.
        key_id: Key identifier used for verification.
        verified_at: Verification timestamp.
        processing_time_ms: Processing time.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(..., description="QR code ID")
    valid: bool = Field(..., description="Signature valid")
    algorithm: str = Field(
        default="HMAC-SHA256", description="Signing algorithm"
    )
    key_id: Optional[str] = Field(None, description="Key identifier")
    verified_at: datetime = Field(
        default_factory=utcnow, description="Verification timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time"
    )

class VerificationStatusResponse(GreenLangBase):
    """Response for verification status lookup.

    Attributes:
        code_id: QR code identifier.
        status: Current code lifecycle status.
        compliance_status: EUDR compliance status.
        verification_url: Verification URL.
        last_verified_at: Last verification timestamp.
        scan_count: Total scan count.
        counterfeit_risk: Current counterfeit risk level.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(..., description="QR code ID")
    status: str = Field(default="active", description="Lifecycle status")
    compliance_status: str = Field(
        default="pending", description="Compliance status"
    )
    verification_url: Optional[str] = Field(
        None, description="Verification URL"
    )
    last_verified_at: Optional[datetime] = Field(
        None, description="Last verification timestamp"
    )
    scan_count: int = Field(default=0, ge=0, description="Scan count")
    counterfeit_risk: str = Field(
        default="low", description="Counterfeit risk"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class OfflineVerifyResponse(GreenLangBase):
    """Response for offline verification.

    Attributes:
        code_id: QR code identifier.
        valid: Whether offline verification passed.
        hmac_valid: Whether the HMAC token is valid.
        status: Code lifecycle status.
        compliance_status: Compliance status.
        verified_at: Verification timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(..., description="QR code ID")
    valid: bool = Field(..., description="Verification passed")
    hmac_valid: bool = Field(..., description="HMAC token valid")
    status: str = Field(default="active", description="Lifecycle status")
    compliance_status: str = Field(
        default="pending", description="Compliance status"
    )
    verified_at: datetime = Field(
        default_factory=utcnow, description="Verification timestamp"
    )

# =============================================================================
# Counterfeit Schemas
# =============================================================================

class CounterfeitCheckRequest(GreenLangBase):
    """Request to check for counterfeiting indicators.

    Attributes:
        code_id: QR code identifier to check.
        scanner_ip: Scanner IP address (hashed).
        latitude: Scan latitude.
        longitude: Scan longitude.
        hmac_token: HMAC token from the scanned URL.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(
        ..., min_length=1, max_length=255,
        description="QR code identifier",
    )
    scanner_ip: Optional[str] = Field(
        None, max_length=255, description="Scanner IP (hashed)"
    )
    latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="Scan latitude"
    )
    longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0, description="Scan longitude"
    )
    hmac_token: Optional[str] = Field(
        None, description="HMAC token from URL"
    )

class CounterfeitCheckResponse(GreenLangBase):
    """Response for a counterfeit check.

    Attributes:
        code_id: QR code identifier.
        risk_level: Assessed risk level.
        risk_score: Numeric risk score (0-100).
        hmac_valid: HMAC token validity.
        velocity_exceeded: Whether scan velocity was exceeded.
        geo_fence_violated: Whether geo-fence was violated.
        alerts: List of triggered alert descriptions.
        checked_at: Check timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(..., description="QR code ID")
    risk_level: str = Field(default="low", description="Risk level")
    risk_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Risk score"
    )
    hmac_valid: Optional[bool] = Field(None, description="HMAC valid")
    velocity_exceeded: bool = Field(
        default=False, description="Velocity exceeded"
    )
    geo_fence_violated: bool = Field(
        default=False, description="Geo-fence violated"
    )
    alerts: List[str] = Field(
        default_factory=list, description="Triggered alerts"
    )
    checked_at: datetime = Field(
        default_factory=utcnow, description="Check timestamp"
    )

class RevokeCodeResponse(GreenLangBase):
    """Response after revoking a counterfeit code.

    Attributes:
        code_id: Revoked code identifier.
        status: New status (revoked).
        previous_status: Previous status.
        reason: Revocation reason.
        revoked_at: Revocation timestamp.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(..., description="Revoked code ID")
    status: str = Field(default="revoked", description="New status")
    previous_status: str = Field(..., description="Previous status")
    reason: str = Field(..., description="Revocation reason")
    revoked_at: datetime = Field(
        default_factory=utcnow, description="Revocation timestamp"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class RevocationListItem(GreenLangBase):
    """A single entry in the revocation list.

    Attributes:
        code_id: Revoked code identifier.
        reason: Revocation reason.
        revoked_at: Revocation timestamp.
        operator_id: Operator who revoked.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(..., description="Revoked code ID")
    reason: str = Field(..., description="Revocation reason")
    revoked_at: datetime = Field(
        default_factory=utcnow, description="Revocation timestamp"
    )
    operator_id: Optional[str] = Field(None, description="Revoking operator")

class RevocationListResponse(GreenLangBase):
    """Response listing all revoked codes.

    Attributes:
        revocations: List of revocation entries.
        total: Total revoked codes.
        pagination: Pagination metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    revocations: List[RevocationListItem] = Field(
        default_factory=list, description="Revocation entries"
    )
    total: int = Field(default=0, ge=0, description="Total revoked")
    pagination: Optional[PaginatedMeta] = Field(
        None, description="Pagination metadata"
    )

class CounterfeitAnalyticsResponse(GreenLangBase):
    """Response for counterfeit analytics.

    Attributes:
        total_checks: Total counterfeit checks performed.
        total_detections: Total counterfeit detections.
        risk_distribution: Count by risk level.
        velocity_violations: Total velocity violations.
        geo_fence_violations: Total geo-fence violations.
        hmac_failures: Total HMAC validation failures.
        top_risky_codes: Most frequently flagged codes.
        period_start: Analytics period start.
        period_end: Analytics period end.
    """

    model_config = ConfigDict(from_attributes=True)

    total_checks: int = Field(default=0, ge=0, description="Total checks")
    total_detections: int = Field(
        default=0, ge=0, description="Total detections"
    )
    risk_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Counts by risk level"
    )
    velocity_violations: int = Field(
        default=0, ge=0, description="Velocity violations"
    )
    geo_fence_violations: int = Field(
        default=0, ge=0, description="Geo-fence violations"
    )
    hmac_failures: int = Field(
        default=0, ge=0, description="HMAC failures"
    )
    top_risky_codes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Most flagged codes"
    )
    period_start: Optional[datetime] = Field(
        None, description="Analytics period start"
    )
    period_end: Optional[datetime] = Field(
        None, description="Analytics period end"
    )

# =============================================================================
# Bulk Generation Schemas
# =============================================================================

class SubmitBulkRequest(GreenLangBase):
    """Request to submit a bulk QR code generation job.

    Attributes:
        operator_id: EUDR operator identifier.
        total_codes: Number of codes to generate.
        content_type: Payload content type.
        commodity: EUDR commodity type.
        error_correction: Error correction level.
        output_format: Output image format.
        payload_template: Template for payload data.
        worker_count: Number of concurrent workers.
    """

    model_config = ConfigDict(from_attributes=True)

    operator_id: str = Field(
        ..., min_length=1, max_length=255,
        description="EUDR operator identifier",
    )
    total_codes: int = Field(
        ..., ge=1, le=100000, description="Codes to generate"
    )
    content_type: Optional[ContentTypeSchema] = Field(
        None, description="Content type"
    )
    commodity: Optional[EUDRCommoditySchema] = Field(
        None, description="EUDR commodity"
    )
    error_correction: Optional[ErrorCorrectionSchema] = Field(
        None, description="Error correction level"
    )
    output_format: Optional[OutputFormatSchema] = Field(
        None, description="Output format"
    )
    payload_template: Optional[Dict[str, Any]] = Field(
        None, description="Payload template"
    )
    worker_count: Optional[int] = Field(
        None, ge=1, le=64, description="Worker count"
    )

class SubmitBulkResponse(GreenLangBase):
    """Response after submitting a bulk generation job.

    Attributes:
        job_id: Bulk job identifier.
        status: Job status (queued).
        total_codes: Total codes to generate.
        operator_id: Operator identifier.
        created_at: Submission timestamp.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(
        default_factory=_new_id, description="Bulk job ID"
    )
    status: str = Field(default="queued", description="Job status")
    total_codes: int = Field(..., ge=1, description="Total codes")
    operator_id: str = Field(..., description="Operator ID")
    created_at: datetime = Field(
        default_factory=utcnow, description="Submission timestamp"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class BulkStatusResponse(GreenLangBase):
    """Response for bulk job status query.

    Attributes:
        job_id: Bulk job identifier.
        status: Current job status.
        total_codes: Total codes to generate.
        completed_codes: Codes generated so far.
        failed_codes: Codes that failed.
        progress_percent: Completion percentage.
        output_format: Output format.
        worker_count: Workers assigned.
        error_message: Error message (if failed).
        started_at: Processing start timestamp.
        completed_at: Completion timestamp.
        created_at: Submission timestamp.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(..., description="Bulk job ID")
    status: str = Field(default="queued", description="Job status")
    total_codes: int = Field(..., ge=1, description="Total codes")
    completed_codes: int = Field(
        default=0, ge=0, description="Completed codes"
    )
    failed_codes: int = Field(default=0, ge=0, description="Failed codes")
    progress_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress %"
    )
    output_format: str = Field(default="png", description="Output format")
    worker_count: int = Field(default=4, ge=1, description="Workers")
    error_message: Optional[str] = Field(None, description="Error message")
    started_at: Optional[datetime] = Field(
        None, description="Start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Completion timestamp"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Submission timestamp"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class BulkDownloadResponse(GreenLangBase):
    """Response for bulk job output download.

    Attributes:
        job_id: Bulk job identifier.
        download_url: Pre-signed download URL.
        expires_in_seconds: URL expiry in seconds.
        file_size_bytes: Output package size.
        file_hash: SHA-256 hash of the output package.
        content_type: MIME content type.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(..., description="Bulk job ID")
    download_url: str = Field(..., description="Download URL")
    expires_in_seconds: int = Field(
        default=3600, ge=60, description="URL expiry seconds"
    )
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="Package size"
    )
    file_hash: Optional[str] = Field(
        None, description="SHA-256 hash of package"
    )
    content_type: str = Field(
        default="application/zip", description="MIME type"
    )

class BulkManifestResponse(GreenLangBase):
    """Response for bulk job manifest download.

    Attributes:
        job_id: Bulk job identifier.
        codes: List of generated code records.
        total_codes: Total codes in manifest.
        generated_at: Manifest generation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(..., description="Bulk job ID")
    codes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Generated code records"
    )
    total_codes: int = Field(default=0, ge=0, description="Total codes")
    generated_at: datetime = Field(
        default_factory=utcnow, description="Manifest generation timestamp"
    )

class BulkCancelResponse(GreenLangBase):
    """Response for bulk job cancellation.

    Attributes:
        job_id: Bulk job identifier.
        status: Updated status (cancelled).
        cancelled_at: Cancellation timestamp.
        message: Cancellation message.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(..., description="Bulk job ID")
    status: str = Field(default="cancelled", description="Job status")
    cancelled_at: datetime = Field(
        default_factory=utcnow, description="Cancellation timestamp"
    )
    message: str = Field(
        default="Bulk job cancelled successfully",
        description="Cancellation message",
    )

# =============================================================================
# Lifecycle Schemas
# =============================================================================

class ActivateRequest(GreenLangBase):
    """Request to activate a QR code.

    Attributes:
        performed_by: User performing activation.
        reason: Activation reason.
    """

    model_config = ConfigDict(from_attributes=True)

    performed_by: Optional[str] = Field(
        None, max_length=255, description="User performing activation"
    )
    reason: Optional[str] = Field(
        None, max_length=2000, description="Activation reason"
    )

class DeactivateRequest(GreenLangBase):
    """Request to temporarily deactivate a QR code.

    Attributes:
        reason: Deactivation reason (required).
        performed_by: User performing deactivation.
    """

    model_config = ConfigDict(from_attributes=True)

    reason: str = Field(
        ..., min_length=1, max_length=2000,
        description="Deactivation reason",
    )
    performed_by: Optional[str] = Field(
        None, max_length=255, description="User performing deactivation"
    )

class RevokeRequest(GreenLangBase):
    """Request to permanently revoke a QR code.

    Attributes:
        reason: Revocation reason (required).
        performed_by: User performing revocation.
    """

    model_config = ConfigDict(from_attributes=True)

    reason: str = Field(
        ..., min_length=1, max_length=2000,
        description="Revocation reason",
    )
    performed_by: Optional[str] = Field(
        None, max_length=255, description="User performing revocation"
    )

class ScanEventRequest(GreenLangBase):
    """Request to record a scan event.

    Attributes:
        code_id: Scanned QR code identifier.
        scanner_ip: Scanner IP address (hashed).
        scanner_user_agent: Scanner user agent string.
        latitude: Scan latitude.
        longitude: Scan longitude.
        hmac_token: HMAC token from URL.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Scanned QR code ID",
    )
    scanner_ip: Optional[str] = Field(
        None, max_length=255, description="Scanner IP (hashed)"
    )
    scanner_user_agent: Optional[str] = Field(
        None, max_length=1000, description="User agent string"
    )
    latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="Scan latitude"
    )
    longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0, description="Scan longitude"
    )
    hmac_token: Optional[str] = Field(
        None, description="HMAC token from URL"
    )

class ActivateResponse(GreenLangBase):
    """Response after activating a QR code.

    Attributes:
        code_id: QR code identifier.
        status: Operation status.
        previous_status: Previous lifecycle status.
        new_status: New lifecycle status (active).
        activated_at: Activation timestamp.
        processing_time_ms: Processing time.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(..., description="QR code ID")
    status: str = Field(default="success", description="Operation status")
    previous_status: str = Field(
        default="created", description="Previous status"
    )
    new_status: str = Field(default="active", description="New status")
    activated_at: datetime = Field(
        default_factory=utcnow, description="Activation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class DeactivateResponse(GreenLangBase):
    """Response after deactivating a QR code.

    Attributes:
        code_id: QR code identifier.
        status: Operation status.
        previous_status: Previous lifecycle status.
        new_status: New lifecycle status (deactivated).
        deactivated_at: Deactivation timestamp.
        reason: Deactivation reason.
        processing_time_ms: Processing time.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(..., description="QR code ID")
    status: str = Field(default="success", description="Operation status")
    previous_status: str = Field(
        default="active", description="Previous status"
    )
    new_status: str = Field(
        default="deactivated", description="New status"
    )
    deactivated_at: datetime = Field(
        default_factory=utcnow, description="Deactivation timestamp"
    )
    reason: str = Field(..., description="Deactivation reason")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class RevokeResponse(GreenLangBase):
    """Response after revoking a QR code.

    Attributes:
        code_id: QR code identifier.
        status: Operation status.
        previous_status: Previous lifecycle status.
        new_status: New lifecycle status (revoked).
        revoked_at: Revocation timestamp.
        reason: Revocation reason.
        processing_time_ms: Processing time.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(..., description="QR code ID")
    status: str = Field(default="success", description="Operation status")
    previous_status: str = Field(..., description="Previous status")
    new_status: str = Field(default="revoked", description="New status")
    revoked_at: datetime = Field(
        default_factory=utcnow, description="Revocation timestamp"
    )
    reason: str = Field(..., description="Revocation reason")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class ScanEventResponse(GreenLangBase):
    """Response after recording a scan event.

    Attributes:
        scan_id: Scan event identifier.
        code_id: Scanned QR code identifier.
        outcome: Scan verification outcome.
        counterfeit_risk: Assessed risk level.
        hmac_valid: Whether HMAC token is valid.
        velocity_scans_per_min: Current scan velocity.
        geo_fence_violated: Whether geo-fence was violated.
        response_time_ms: Processing time.
        scanned_at: Scan timestamp.
        provenance: Provenance tracking.
    """

    model_config = ConfigDict(from_attributes=True)

    scan_id: str = Field(
        default_factory=_new_id, description="Scan event ID"
    )
    code_id: str = Field(..., description="QR code ID")
    outcome: str = Field(default="verified", description="Scan outcome")
    counterfeit_risk: str = Field(
        default="low", description="Risk level"
    )
    hmac_valid: Optional[bool] = Field(None, description="HMAC valid")
    velocity_scans_per_min: Optional[int] = Field(
        None, ge=0, description="Scan velocity"
    )
    geo_fence_violated: bool = Field(
        default=False, description="Geo-fence violated"
    )
    response_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time"
    )
    scanned_at: datetime = Field(
        default_factory=utcnow, description="Scan timestamp"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )

class LifecycleEventItem(GreenLangBase):
    """A single lifecycle event record.

    Attributes:
        event_id: Event identifier.
        code_id: Associated QR code.
        event_type: Type of lifecycle event.
        previous_status: Status before event.
        new_status: Status after event.
        reason: Reason for the change.
        performed_by: User who performed the change.
        created_at: Event timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    event_id: str = Field(
        default_factory=_new_id, description="Event identifier"
    )
    code_id: str = Field(..., description="QR code ID")
    event_type: str = Field(..., description="Event type")
    previous_status: str = Field(..., description="Previous status")
    new_status: str = Field(..., description="New status")
    reason: Optional[str] = Field(None, description="Change reason")
    performed_by: Optional[str] = Field(
        None, description="User who performed the change"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Event timestamp"
    )

class LifecycleHistoryResponse(GreenLangBase):
    """Response for lifecycle history query.

    Attributes:
        code_id: QR code identifier.
        current_status: Current lifecycle status.
        events: List of lifecycle events.
        total_events: Total events in history.
        pagination: Pagination metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    code_id: str = Field(..., description="QR code ID")
    current_status: str = Field(..., description="Current status")
    events: List[LifecycleEventItem] = Field(
        default_factory=list, description="Lifecycle events"
    )
    total_events: int = Field(
        default=0, ge=0, description="Total events"
    )
    pagination: Optional[PaginatedMeta] = Field(
        None, description="Pagination metadata"
    )

# =============================================================================
# Health Schema
# =============================================================================

class HealthComponentSchema(GreenLangBase):
    """Health status for a single service component.

    Attributes:
        name: Component name.
        status: Component health status.
        latency_ms: Component response latency.
        details: Additional health details.
    """

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(..., description="Component name")
    status: str = Field(default="healthy", description="Health status")
    latency_ms: Optional[float] = Field(
        None, ge=0.0, description="Response latency in ms"
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Health details"
    )

class HealthResponse(GreenLangBase):
    """Health check response for the QR Code Generator API.

    Attributes:
        service: Service identifier.
        status: Overall health status.
        version: Service version.
        agent_id: Agent identifier.
        uptime_seconds: Service uptime in seconds.
        active_bulk_jobs: Number of active bulk jobs.
        total_codes_generated: Total QR codes generated.
        components: Component health details.
        checked_at: Health check timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    service: str = Field(
        default="eudr-qr-code-generator",
        description="Service identifier",
    )
    status: str = Field(default="healthy", description="Overall health")
    version: str = Field(default="1.0.0", description="Service version")
    agent_id: str = Field(
        default="GL-EUDR-QRG-014", description="Agent identifier"
    )
    uptime_seconds: float = Field(
        default=0.0, ge=0.0, description="Uptime in seconds"
    )
    active_bulk_jobs: int = Field(
        default=0, ge=0, description="Active bulk jobs"
    )
    total_codes_generated: int = Field(
        default=0, ge=0, description="Total codes generated"
    )
    components: List[HealthComponentSchema] = Field(
        default_factory=lambda: [
            HealthComponentSchema(name="api", status="healthy"),
            HealthComponentSchema(name="database", status="healthy"),
            HealthComponentSchema(name="cache", status="healthy"),
            HealthComponentSchema(name="qr_engine", status="healthy"),
            HealthComponentSchema(name="label_engine", status="healthy"),
            HealthComponentSchema(name="bulk_engine", status="healthy"),
        ],
        description="Component health details",
    )
    checked_at: datetime = Field(
        default_factory=utcnow, description="Health check timestamp"
    )

# =============================================================================
# Error Response
# =============================================================================

class ErrorResponse(GreenLangBase):
    """Standard error response.

    Attributes:
        error: Error type identifier.
        message: Human-readable error message.
        detail: Additional details.
        request_id: Request correlation ID.
    """

    model_config = ConfigDict(from_attributes=True)

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional details")
    request_id: Optional[str] = Field(
        None, description="Request correlation ID"
    )

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enumerations
    "BulkJobStatusSchema",
    "CheckDigitAlgorithmSchema",
    "CodeStatusSchema",
    "ComplianceStatusSchema",
    "ContentTypeSchema",
    "CounterfeitRiskSchema",
    "EUDRCommoditySchema",
    "ErrorCorrectionSchema",
    "LabelTemplateSchema",
    "OutputFormatSchema",
    "PayloadEncodingSchema",
    "ScanOutcomeSchema",
    "SymbologyTypeSchema",
    # Shared
    "HealthComponentSchema",
    "PaginatedMeta",
    "ProvenanceInfo",
    # QR schemas
    "GenerateDataMatrixRequest",
    "GenerateDataMatrixResponse",
    "GenerateQRRequest",
    "GenerateQRResponse",
    "QRCodeDetailResponse",
    "QRImageResponse",
    # Payload schemas
    "ComposePayloadRequest",
    "ComposePayloadResponse",
    "PayloadDetailResponse",
    "PayloadSchemaItem",
    "PayloadSchemasResponse",
    "ValidatePayloadRequest",
    "ValidatePayloadResponse",
    # Label schemas
    "BatchLabelRequest",
    "BatchLabelResponse",
    "GenerateLabelRequest",
    "GenerateLabelResponse",
    "LabelDetailResponse",
    "TemplateDetailResponse",
    "TemplateListResponse",
    # Batch code schemas
    "BatchCodeDetailResponse",
    "BatchCodeItem",
    "CodeHierarchyResponse",
    "GenerateBatchCodesRequest",
    "GenerateBatchCodesResponse",
    "ReserveCodesRequest",
    "ReserveCodesResponse",
    # Verification schemas
    "BuildURLRequest",
    "BuildURLResponse",
    "OfflineVerifyRequest",
    "OfflineVerifyResponse",
    "VerificationStatusResponse",
    "VerifySignatureRequest",
    "VerifySignatureResponse",
    # Counterfeit schemas
    "CounterfeitAnalyticsResponse",
    "CounterfeitCheckRequest",
    "CounterfeitCheckResponse",
    "RevocationListItem",
    "RevocationListResponse",
    "RevokeCodeResponse",
    # Bulk schemas
    "BulkCancelResponse",
    "BulkDownloadResponse",
    "BulkManifestResponse",
    "BulkStatusResponse",
    "SubmitBulkRequest",
    "SubmitBulkResponse",
    # Lifecycle schemas
    "ActivateRequest",
    "ActivateResponse",
    "DeactivateRequest",
    "DeactivateResponse",
    "LifecycleEventItem",
    "LifecycleHistoryResponse",
    "RevokeRequest",
    "RevokeResponse",
    "ScanEventRequest",
    "ScanEventResponse",
    # Health
    "ErrorResponse",
    "HealthResponse",
]
