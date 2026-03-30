# -*- coding: utf-8 -*-
"""
Mobile Data Collector Data Models - AGENT-EUDR-015

Pydantic v2 data models for the Mobile Data Collector Agent covering
offline form collection with local SQLite storage and schema validation;
GPS point and polygon capture with accuracy metadata (HDOP, satellite
count, fix type, augmentation); geotagged photo evidence with EXIF
extraction and SHA-256 integrity hashing; CRDT-based offline data
synchronization with conflict resolution (LWW, set union, manual);
dynamic form templates with conditional logic and multi-language support
(24 EU + 20 local languages); ECDSA P-256 digital signature capture with
timestamp binding and revocation; self-contained data package assembly
with SHA-256 Merkle root integrity; and device fleet management with
sync status, telemetry monitoring, and campaign tracking.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all mobile data collection operations per
EU 2023/1115 Articles 4, 9, and 14.

Enumerations (15):
    - FormStatus, FormType, CaptureAccuracyTier, PhotoType,
      SyncStatus, ConflictResolution, TemplateType, SignatureAlgorithm,
      PackageStatus, DeviceStatus, DevicePlatform, CommodityType,
      ComplianceStatus, FieldType, LanguageCode

Core Models (12):
    - FormSubmission, GPSCapture, PolygonTrace, PhotoEvidence,
      SyncQueueItem, SyncConflict, FormTemplate, DigitalSignature,
      DataPackage, DeviceRegistration, DeviceEvent, AuditLogEntry

Request Models (15):
    - SubmitFormRequest, CaptureGPSRequest, CapturePolygonRequest,
      UploadPhotoRequest, TriggerSyncRequest, ResolveConflictRequest,
      CreateTemplateRequest, UpdateTemplateRequest,
      CaptureSignatureRequest, BuildPackageRequest,
      RegisterDeviceRequest, UpdateDeviceRequest, SearchFormsRequest,
      GetDeviceStatusRequest, ValidateFormRequest

Response Models (15):
    - FormResponse, GPSResponse, PolygonResponse, PhotoResponse,
      SyncResponse, ConflictResponse, TemplateResponse,
      SignatureResponse, PackageResponse, DeviceResponse,
      DeviceStatusResponse, SearchResponse, SyncStatusResponse,
      FleetStatusResponse, HealthResponse

Compatibility:
    Imports EUDRCommodity from greenlang.agents.data.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector and AGENT-EUDR-002 Geolocation Verification.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015 Mobile Data Collector (GL-EUDR-MDC-015)
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

#: Supported photo file formats.
SUPPORTED_PHOTO_FORMATS: List[str] = ["jpeg", "png", "heic"]

#: Supported languages (24 EU + 20 local).
SUPPORTED_LANGUAGES: List[str] = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi",
    "fr", "de", "el", "hu", "ga", "it", "lv", "lt",
    "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv",
    "sw", "tw", "fo", "yo", "ha", "am", "ln", "rw",
    "mg", "wo", "pt-BR", "es-419", "qu", "gn", "fr-GF",
    "id", "ms", "th", "vi", "km",
]

#: WGS84 SRID for GPS coordinates.
WGS84_SRID: int = 4326

#: Default EUDR commodities (EU 2023/1115 Article 1).
DEFAULT_EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

#: Supported EUDR form types.
SUPPORTED_FORM_TYPES: List[str] = [
    "producer_registration",
    "plot_survey",
    "harvest_log",
    "custody_transfer",
    "quality_inspection",
    "smallholder_declaration",
]

#: Supported field types for form templates.
SUPPORTED_FIELD_TYPES: List[str] = [
    "text", "number", "date", "select", "multiselect",
    "gps", "photo", "signature", "checkbox", "textarea",
]

# =============================================================================
# Enumerations
# =============================================================================

class FormStatus(str, Enum):
    """Form submission lifecycle status.

    DRAFT: Form is incomplete and being edited on device. Can be
        modified by the field agent.
    PENDING: Form has passed local validation and is queued for
        synchronization. Cannot be edited.
    SYNCING: Form upload is in progress during a sync session.
    SYNCED: Form has been received and confirmed by the server.
        Immutable -- only corrective amendments via new forms.
    FAILED: Sync failed after maximum retries. Requires manual
        intervention or retry.
    """

    DRAFT = "draft"
    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"

class FormType(str, Enum):
    """EUDR form type classifications per PRD Section 2.2.

    PRODUCER_REGISTRATION: Art. 9(1)(f) supplier identification.
    PLOT_SURVEY: Art. 9(1)(c-d) geolocation capture.
    HARVEST_LOG: Art. 9(1)(a-b,e) production data.
    CUSTODY_TRANSFER: Art. 9(1)(f-g) chain of custody.
    QUALITY_INSPECTION: Art. 10(1) risk assessment data.
    SMALLHOLDER_DECLARATION: Art. 4(2) due diligence.
    """

    PRODUCER_REGISTRATION = "producer_registration"
    PLOT_SURVEY = "plot_survey"
    HARVEST_LOG = "harvest_log"
    CUSTODY_TRANSFER = "custody_transfer"
    QUALITY_INSPECTION = "quality_inspection"
    SMALLHOLDER_DECLARATION = "smallholder_declaration"

class CaptureAccuracyTier(str, Enum):
    """GPS capture accuracy classification per PRD Appendix B.

    EXCELLENT: <1m horizontal accuracy, HDOP <1.0, >=12 satellites.
    GOOD: 1-3m horizontal accuracy, HDOP 1.0-2.0, 8-12 satellites.
    ACCEPTABLE: 3-5m horizontal accuracy, HDOP 2.0-3.0, 6-8 satellites.
    POOR: 5-10m horizontal accuracy, HDOP 3.0-5.0, 4-6 satellites.
    REJECTED: >10m horizontal accuracy, HDOP >5.0, <4 satellites.
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    REJECTED = "rejected"

class PhotoType(str, Enum):
    """Photo evidence category classifications per PRD F3.4.

    PLOT_PHOTO: Landscape view of production plot.
    COMMODITY_PHOTO: Close-up of harvested commodity.
    DOCUMENT_PHOTO: Physical document capture.
    FACILITY_PHOTO: Processing/storage facility.
    TRANSPORT_PHOTO: Vehicle or container.
    IDENTITY_PHOTO: Producer/operator identification.
    """

    PLOT_PHOTO = "plot_photo"
    COMMODITY_PHOTO = "commodity_photo"
    DOCUMENT_PHOTO = "document_photo"
    FACILITY_PHOTO = "facility_photo"
    TRANSPORT_PHOTO = "transport_photo"
    IDENTITY_PHOTO = "identity_photo"

class SyncStatus(str, Enum):
    """Synchronization queue item status.

    QUEUED: Item is in the upload queue awaiting sync.
    IN_PROGRESS: Item upload is currently in progress.
    COMPLETED: Item successfully uploaded and confirmed.
    FAILED: Item upload failed; eligible for retry.
    PERMANENTLY_FAILED: Item exceeded max retries; requires manual
        intervention.
    """

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PERMANENTLY_FAILED = "permanently_failed"

class ConflictResolution(str, Enum):
    """Sync conflict resolution strategy per PRD Appendix C.

    SERVER_WINS: Server version takes precedence for scalar fields.
    CLIENT_WINS: Client (device) version takes precedence.
    MANUAL: Conflict flagged for manual human resolution.
    LWW: Last-Writer-Wins based on vector clock timestamps.
    SET_UNION: Grow-only set union for collection fields (photos, GPS).
    STATE_MACHINE: Higher-precedence status wins (synced > pending > draft).
    """

    SERVER_WINS = "server_wins"
    CLIENT_WINS = "client_wins"
    MANUAL = "manual"
    LWW = "lww"
    SET_UNION = "set_union"
    STATE_MACHINE = "state_machine"

class TemplateType(str, Enum):
    """Form template type classifications.

    BASE: Standard EUDR base template provided by the platform.
    CUSTOM: Operator-specific custom template extending a base.
    INHERITED: Template created via inheritance from a base template.
    """

    BASE = "base"
    CUSTOM = "custom"
    INHERITED = "inherited"

class SignatureAlgorithm(str, Enum):
    """Digital signature algorithm options.

    ECDSA_P256: ECDSA with NIST P-256 curve (secp256r1) and
        deterministic k-value per RFC 6979. Default for EUDR.
    ECDSA_P384: ECDSA with NIST P-384 curve for higher security.
    """

    ECDSA_P256 = "ecdsa_p256"
    ECDSA_P384 = "ecdsa_p384"

class PackageStatus(str, Enum):
    """Data package lifecycle status.

    BUILDING: Package is being assembled incrementally.
    SEALED: Package has been signed and Merkle root computed.
        No further artifacts can be added.
    UPLOADED: Package has been uploaded to the server.
    VERIFIED: Package integrity verified by the server.
    EXPIRED: Package has exceeded its TTL.
    """

    BUILDING = "building"
    SEALED = "sealed"
    UPLOADED = "uploaded"
    VERIFIED = "verified"
    EXPIRED = "expired"

class DeviceStatus(str, Enum):
    """Mobile device status in the fleet.

    ACTIVE: Device has synced within the offline threshold window.
    OFFLINE: Device has not synced within the threshold.
    LOW_BATTERY: Device reported battery below threshold.
    LOW_STORAGE: Device reported storage below threshold.
    DECOMMISSIONED: Device has been retired from active service.
    """

    ACTIVE = "active"
    OFFLINE = "offline"
    LOW_BATTERY = "low_battery"
    LOW_STORAGE = "low_storage"
    DECOMMISSIONED = "decommissioned"

class DevicePlatform(str, Enum):
    """Mobile device operating system platform.

    ANDROID: Android OS (primary target platform).
    IOS: Apple iOS.
    HARMONYOS: Huawei HarmonyOS.
    """

    ANDROID = "android"
    IOS = "ios"
    HARMONYOS = "harmonyos"

class CommodityType(str, Enum):
    """EUDR-regulated commodity types per EU 2023/1115 Article 1.

    Seven commodity categories subject to the EUDR deforestation-free
    requirement and due diligence obligations.
    """

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"

class ComplianceStatus(str, Enum):
    """EUDR compliance assessment status.

    COMPLIANT: Data meets all EUDR requirements.
    NON_COMPLIANT: Data fails one or more EUDR requirements.
    PENDING_REVIEW: Data awaiting compliance review.
    UNDER_INVESTIGATION: Data flagged for competent authority review.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    UNDER_INVESTIGATION = "under_investigation"

class FieldType(str, Enum):
    """Form template field types per PRD F5.3.

    TEXT: Single-line text input.
    NUMBER: Integer or decimal numeric input.
    DATE: Date picker (ISO 8601).
    SELECT: Single-selection dropdown.
    MULTISELECT: Multi-selection dropdown/checkbox group.
    GPS: GPS coordinate capture (point or polygon).
    PHOTO: Photo capture with geotagging.
    SIGNATURE: Digital signature capture.
    CHECKBOX: Boolean checkbox.
    TEXTAREA: Multi-line text input.
    """

    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    SELECT = "select"
    MULTISELECT = "multiselect"
    GPS = "gps"
    PHOTO = "photo"
    SIGNATURE = "signature"
    CHECKBOX = "checkbox"
    TEXTAREA = "textarea"

class LanguageCode(str, Enum):
    """Supported language codes per PRD Appendix D.

    24 EU official languages plus 20 local languages for field
    data collection in commodity-producing countries.
    """

    # EU official languages
    BG = "bg"
    HR = "hr"
    CS = "cs"
    DA = "da"
    NL = "nl"
    EN = "en"
    ET = "et"
    FI = "fi"
    FR = "fr"
    DE = "de"
    EL = "el"
    HU = "hu"
    GA = "ga"
    IT = "it"
    LV = "lv"
    LT = "lt"
    MT = "mt"
    PL = "pl"
    PT = "pt"
    RO = "ro"
    SK = "sk"
    SL = "sl"
    ES = "es"
    SV = "sv"
    # Local languages
    SW = "sw"
    TW = "tw"
    FO = "fo"
    YO = "yo"
    HA = "ha"
    AM = "am"
    LN = "ln"
    RW = "rw"
    MG = "mg"
    WO = "wo"
    PT_BR = "pt-BR"
    ES_419 = "es-419"
    QU = "qu"
    GN = "gn"
    FR_GF = "fr-GF"
    ID = "id"
    MS = "ms"
    TH = "th"
    VI = "vi"
    KM = "km"

# =============================================================================
# Core Models (12)
# =============================================================================

class FormSubmission(GreenLangBase):
    """A single collected form submission from a mobile device.

    Represents a completed or in-progress form (producer registration,
    plot survey, harvest log, custody transfer, quality inspection, or
    smallholder declaration) captured on a mobile device. Links to GPS
    captures, photos, and digital signatures via form_id.

    Attributes:
        form_id: Unique form submission identifier (UUID v4).
        device_id: Device that captured this form.
        operator_id: Field agent who submitted the form.
        form_type: EUDR form type classification.
        template_id: Form template definition used.
        template_version: Semantic version of the template.
        status: Current form lifecycle status.
        data: Form field values as key-value pairs.
        commodity_type: EUDR commodity type if applicable.
        country_code: ISO 3166-1 alpha-2 country code.
        submission_hash: SHA-256 hash of form data at submission.
        local_timestamp: Device-local timestamp at submission.
        server_timestamp: Server-side receipt timestamp.
        gps_capture_ids: Linked GPS capture identifiers.
        photo_ids: Linked photo evidence identifiers.
        signature_ids: Linked digital signature identifiers.
        metadata: Additional form metadata.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    form_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique form submission identifier (UUID v4)",
    )
    device_id: str = Field(
        ..., description="Device that captured this form",
    )
    operator_id: str = Field(
        ..., description="Field agent who submitted the form",
    )
    form_type: FormType = Field(
        ..., description="EUDR form type classification",
    )
    template_id: str = Field(
        ..., description="Form template definition used",
    )
    template_version: str = Field(
        default="1.0.0",
        description="Semantic version of the template",
    )
    status: FormStatus = Field(
        default=FormStatus.DRAFT,
        description="Current form lifecycle status",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Form field values as key-value pairs",
    )
    commodity_type: Optional[CommodityType] = Field(
        None, description="EUDR commodity type if applicable",
    )
    country_code: Optional[str] = Field(
        None, description="ISO 3166-1 alpha-2 country code",
    )
    submission_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of form data at submission",
    )
    local_timestamp: Optional[datetime] = Field(
        None, description="Device-local timestamp at submission",
    )
    server_timestamp: Optional[datetime] = Field(
        None, description="Server-side receipt timestamp",
    )
    gps_capture_ids: List[str] = Field(
        default_factory=list,
        description="Linked GPS capture identifiers",
    )
    photo_ids: List[str] = Field(
        default_factory=list,
        description="Linked photo evidence identifiers",
    )
    signature_ids: List[str] = Field(
        default_factory=list,
        description="Linked digital signature identifiers",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional form metadata",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="Record last update timestamp",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate ISO 3166-1 alpha-2 format (2 uppercase letters)."""
        if v is not None:
            v = v.upper().strip()
            if len(v) != 2 or not v.isalpha():
                raise ValueError(
                    f"country_code must be 2-letter ISO 3166-1 alpha-2, "
                    f"got '{v}'"
                )
        return v

class GPSCapture(GreenLangBase):
    """A single GPS point coordinate capture with accuracy metadata.

    Records a GPS fix from a mobile device sensor, including accuracy
    metadata required for EUDR Article 9(1)(d) compliance: HDOP,
    satellite count, fix type, and augmentation source.

    Attributes:
        capture_id: Unique GPS capture identifier (UUID v4).
        form_id: Associated form submission identifier.
        device_id: Device that captured the GPS fix.
        operator_id: Field agent who triggered the capture.
        latitude: WGS84 latitude in decimal degrees.
        longitude: WGS84 longitude in decimal degrees.
        altitude_m: Altitude above sea level in meters.
        horizontal_accuracy_m: Estimated horizontal accuracy in meters.
        vertical_accuracy_m: Estimated vertical accuracy in meters.
        hdop: Horizontal Dilution of Precision.
        satellite_count: Number of satellites used in fix.
        fix_type: GPS constellation type (GPS, GLONASS, Galileo, etc.).
        augmentation: SBAS augmentation source if available.
        accuracy_tier: Calculated accuracy classification.
        capture_timestamp: Device timestamp at capture.
        srid: Spatial Reference Identifier (4326 for WGS84).
        metadata: Additional capture metadata.
        created_at: Record creation timestamp.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    capture_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique GPS capture identifier (UUID v4)",
    )
    form_id: Optional[str] = Field(
        None, description="Associated form submission identifier",
    )
    device_id: str = Field(
        ..., description="Device that captured the GPS fix",
    )
    operator_id: str = Field(
        ..., description="Field agent who triggered the capture",
    )
    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="WGS84 latitude in decimal degrees",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="WGS84 longitude in decimal degrees",
    )
    altitude_m: Optional[float] = Field(
        None, description="Altitude above sea level in meters",
    )
    horizontal_accuracy_m: float = Field(
        ..., ge=0.0,
        description="Estimated horizontal accuracy in meters",
    )
    vertical_accuracy_m: Optional[float] = Field(
        None, ge=0.0,
        description="Estimated vertical accuracy in meters",
    )
    hdop: float = Field(
        ..., ge=0.0,
        description="Horizontal Dilution of Precision",
    )
    satellite_count: int = Field(
        ..., ge=0,
        description="Number of satellites used in fix",
    )
    fix_type: str = Field(
        default="GPS",
        description="GPS constellation type (GPS, GLONASS, Galileo, BeiDou, combined)",
    )
    augmentation: Optional[str] = Field(
        None,
        description="SBAS augmentation source (WAAS, EGNOS, MSAS, GAGAN)",
    )
    accuracy_tier: Optional[CaptureAccuracyTier] = Field(
        None, description="Calculated accuracy classification",
    )
    capture_timestamp: datetime = Field(
        default_factory=utcnow,
        description="Device timestamp at capture",
    )
    srid: int = Field(
        default=WGS84_SRID,
        description="Spatial Reference Identifier (4326 for WGS84)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional capture metadata",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp",
    )

class PolygonTrace(GreenLangBase):
    """A plot boundary polygon trace with vertex array and area calculation.

    Records the perimeter of a plot of land via walk-around GPS tracing
    for EUDR Article 9(1)(d) compliance (polygon required for plots
    exceeding 4 hectares).

    Attributes:
        polygon_id: Unique polygon trace identifier (UUID v4).
        form_id: Associated form submission identifier.
        device_id: Device used for polygon tracing.
        operator_id: Field agent who performed the trace.
        vertices: List of [latitude, longitude] coordinate pairs.
        vertex_accuracies_m: Per-vertex horizontal accuracy in meters.
        vertex_count: Total number of vertices in the polygon.
        area_ha: Calculated area in hectares (Shoelace + geodesic).
        perimeter_m: Calculated perimeter in meters.
        is_closed: Whether polygon first vertex equals last vertex.
        is_valid: Whether polygon passes self-intersection check.
        capture_start: Timestamp when tracing started.
        capture_end: Timestamp when tracing completed.
        srid: Spatial Reference Identifier.
        metadata: Additional polygon metadata.
        created_at: Record creation timestamp.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    polygon_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique polygon trace identifier (UUID v4)",
    )
    form_id: Optional[str] = Field(
        None, description="Associated form submission identifier",
    )
    device_id: str = Field(
        ..., description="Device used for polygon tracing",
    )
    operator_id: str = Field(
        ..., description="Field agent who performed the trace",
    )
    vertices: List[List[float]] = Field(
        default_factory=list,
        description="List of [latitude, longitude] coordinate pairs",
    )
    vertex_accuracies_m: List[float] = Field(
        default_factory=list,
        description="Per-vertex horizontal accuracy in meters",
    )
    vertex_count: int = Field(
        default=0,
        ge=0,
        description="Total number of vertices in the polygon",
    )
    area_ha: Optional[float] = Field(
        None, ge=0.0,
        description="Calculated area in hectares",
    )
    perimeter_m: Optional[float] = Field(
        None, ge=0.0,
        description="Calculated perimeter in meters",
    )
    is_closed: bool = Field(
        default=False,
        description="Whether polygon first vertex equals last vertex",
    )
    is_valid: bool = Field(
        default=False,
        description="Whether polygon passes self-intersection check",
    )
    capture_start: Optional[datetime] = Field(
        None, description="Timestamp when tracing started",
    )
    capture_end: Optional[datetime] = Field(
        None, description="Timestamp when tracing completed",
    )
    srid: int = Field(
        default=WGS84_SRID,
        description="Spatial Reference Identifier",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional polygon metadata",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp",
    )

class PhotoEvidence(GreenLangBase):
    """A geotagged photo evidence record with integrity hash.

    Records a photo captured in the field with EXIF metadata, GPS
    coordinates, SHA-256 integrity hash, and form submission linkage.

    Attributes:
        photo_id: Unique photo identifier (UUID v4).
        form_id: Associated form submission identifier.
        capture_id: Associated GPS capture identifier.
        device_id: Device that captured the photo.
        operator_id: Field agent who took the photo.
        photo_type: Photo category classification.
        file_name: Original file name on device.
        file_size_bytes: Photo file size in bytes.
        file_format: Image format (jpeg, png, heic).
        width_px: Image width in pixels.
        height_px: Image height in pixels.
        integrity_hash: SHA-256 hash of raw image bytes at capture.
        hash_algorithm: Hash algorithm used (sha256).
        latitude: GPS latitude where photo was taken.
        longitude: GPS longitude where photo was taken.
        exif_timestamp: EXIF timestamp from photo metadata.
        device_timestamp: Device system time at capture.
        compression_quality: JPEG compression quality applied.
        annotation: Optional text annotation.
        sequence_number: Sequence number within batch capture.
        batch_group_id: Batch capture group identifier.
        metadata: Additional photo metadata.
        created_at: Record creation timestamp.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    photo_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique photo identifier (UUID v4)",
    )
    form_id: str = Field(
        ..., description="Associated form submission identifier",
    )
    capture_id: Optional[str] = Field(
        None, description="Associated GPS capture identifier",
    )
    device_id: str = Field(
        ..., description="Device that captured the photo",
    )
    operator_id: str = Field(
        ..., description="Field agent who took the photo",
    )
    photo_type: PhotoType = Field(
        ..., description="Photo category classification",
    )
    file_name: str = Field(
        ..., description="Original file name on device",
    )
    file_size_bytes: int = Field(
        ..., ge=0,
        description="Photo file size in bytes",
    )
    file_format: str = Field(
        default="jpeg",
        description="Image format (jpeg, png, heic)",
    )
    width_px: int = Field(
        ..., ge=1,
        description="Image width in pixels",
    )
    height_px: int = Field(
        ..., ge=1,
        description="Image height in pixels",
    )
    integrity_hash: str = Field(
        ...,
        description="SHA-256 hash of raw image bytes at capture",
    )
    hash_algorithm: str = Field(
        default="sha256",
        description="Hash algorithm used",
    )
    latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0,
        description="GPS latitude where photo was taken",
    )
    longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0,
        description="GPS longitude where photo was taken",
    )
    exif_timestamp: Optional[datetime] = Field(
        None, description="EXIF timestamp from photo metadata",
    )
    device_timestamp: datetime = Field(
        default_factory=utcnow,
        description="Device system time at capture",
    )
    compression_quality: Optional[int] = Field(
        None, ge=1, le=100,
        description="JPEG compression quality applied",
    )
    annotation: Optional[str] = Field(
        None, description="Optional text annotation",
    )
    sequence_number: Optional[int] = Field(
        None, ge=1,
        description="Sequence number within batch capture",
    )
    batch_group_id: Optional[str] = Field(
        None, description="Batch capture group identifier",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional photo metadata",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp",
    )

class SyncQueueItem(GreenLangBase):
    """An item in the offline synchronization upload queue.

    Attributes:
        queue_item_id: Unique queue item identifier (UUID v4).
        device_id: Source device identifier.
        item_type: Type of data being synced (form/gps/photo/signature/package).
        item_id: Identifier of the specific data item.
        priority: Upload priority (1=highest, 5=lowest).
        status: Current sync status.
        retry_count: Number of sync retries attempted.
        next_retry_at: Scheduled time for next retry.
        idempotency_key: UUID for exactly-once delivery guarantee.
        payload_size_bytes: Size of the upload payload in bytes.
        error_message: Last error message if failed.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    queue_item_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique queue item identifier (UUID v4)",
    )
    device_id: str = Field(
        ..., description="Source device identifier",
    )
    item_type: str = Field(
        ...,
        description="Type of data (form, gps, photo, signature, package)",
    )
    item_id: str = Field(
        ..., description="Identifier of the specific data item",
    )
    priority: int = Field(
        default=3, ge=1, le=5,
        description="Upload priority (1=highest, 5=lowest)",
    )
    status: SyncStatus = Field(
        default=SyncStatus.QUEUED,
        description="Current sync status",
    )
    retry_count: int = Field(
        default=0, ge=0,
        description="Number of sync retries attempted",
    )
    next_retry_at: Optional[datetime] = Field(
        None, description="Scheduled time for next retry",
    )
    idempotency_key: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID for exactly-once delivery guarantee",
    )
    payload_size_bytes: int = Field(
        default=0, ge=0,
        description="Size of the upload payload in bytes",
    )
    error_message: Optional[str] = Field(
        None, description="Last error message if failed",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="Record last update timestamp",
    )

class SyncConflict(GreenLangBase):
    """A detected sync conflict between local and server data.

    Attributes:
        conflict_id: Unique conflict identifier (UUID v4).
        device_id: Device that submitted the conflicting data.
        item_type: Type of conflicting data item.
        item_id: Identifier of the conflicting item.
        field_name: Name of the conflicting field.
        local_value: Value from the device (client).
        server_value: Value from the server.
        local_timestamp: Timestamp of the local change.
        server_timestamp: Timestamp of the server change.
        resolution_strategy: Applied or recommended resolution strategy.
        resolved: Whether the conflict has been resolved.
        resolved_value: The value chosen after resolution.
        resolved_by: Operator who resolved the conflict.
        resolved_at: Timestamp of resolution.
        metadata: Additional conflict metadata.
        created_at: Record creation timestamp.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    conflict_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique conflict identifier (UUID v4)",
    )
    device_id: str = Field(
        ..., description="Device that submitted the conflicting data",
    )
    item_type: str = Field(
        ..., description="Type of conflicting data item",
    )
    item_id: str = Field(
        ..., description="Identifier of the conflicting item",
    )
    field_name: str = Field(
        ..., description="Name of the conflicting field",
    )
    local_value: Any = Field(
        None, description="Value from the device (client)",
    )
    server_value: Any = Field(
        None, description="Value from the server",
    )
    local_timestamp: Optional[datetime] = Field(
        None, description="Timestamp of the local change",
    )
    server_timestamp: Optional[datetime] = Field(
        None, description="Timestamp of the server change",
    )
    resolution_strategy: ConflictResolution = Field(
        default=ConflictResolution.SERVER_WINS,
        description="Applied or recommended resolution strategy",
    )
    resolved: bool = Field(
        default=False,
        description="Whether the conflict has been resolved",
    )
    resolved_value: Any = Field(
        None, description="The value chosen after resolution",
    )
    resolved_by: Optional[str] = Field(
        None, description="Operator who resolved the conflict",
    )
    resolved_at: Optional[datetime] = Field(
        None, description="Timestamp of resolution",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional conflict metadata",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp",
    )

class FormTemplate(GreenLangBase):
    """A dynamic form template definition for EUDR data collection.

    Attributes:
        template_id: Unique template identifier (UUID v4).
        name: Human-readable template name.
        form_type: EUDR form type this template implements.
        template_type: Base, custom, or inherited template.
        version: Semantic version string.
        parent_template_id: Parent template for inheritance.
        schema_definition: JSON schema defining form structure.
        fields: List of field definitions with types and validation.
        conditional_logic: List of conditional show/hide/skip rules.
        validation_rules: List of cross-field validation rules.
        language_packs: Language-specific label translations.
        is_active: Whether template is active for deployment.
        metadata: Additional template metadata.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    template_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique template identifier (UUID v4)",
    )
    name: str = Field(
        ..., description="Human-readable template name",
    )
    form_type: FormType = Field(
        ..., description="EUDR form type this template implements",
    )
    template_type: TemplateType = Field(
        default=TemplateType.BASE,
        description="Base, custom, or inherited template",
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version string",
    )
    parent_template_id: Optional[str] = Field(
        None, description="Parent template for inheritance",
    )
    schema_definition: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema defining form structure",
    )
    fields: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of field definitions with types and validation",
    )
    conditional_logic: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of conditional show/hide/skip rules",
    )
    validation_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of cross-field validation rules",
    )
    language_packs: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Language-specific label translations",
    )
    is_active: bool = Field(
        default=True,
        description="Whether template is active for deployment",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional template metadata",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="Record last update timestamp",
    )

class DigitalSignature(GreenLangBase):
    """A digital signature record for custody transfers and declarations.

    Attributes:
        signature_id: Unique signature identifier (UUID v4).
        form_id: Associated form submission identifier.
        signer_name: Name of the signatory.
        signer_role: Role of the signatory.
        signer_device_id: Device used for signing.
        algorithm: Signature algorithm used.
        public_key_fingerprint: Fingerprint of signer public key.
        signature_bytes_hex: DER-encoded signature in hex.
        signed_data_hash: SHA-256 hash of the signed data.
        timestamp_binding: ISO 8601 timestamp included in signed payload.
        visual_signature_svg: SVG touch-path of handwritten signature.
        is_valid: Whether signature passes verification.
        is_revoked: Whether signature has been revoked.
        revocation_reason: Reason for revocation if revoked.
        revoked_at: Timestamp of revocation.
        metadata: Additional signature metadata.
        created_at: Record creation timestamp.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    signature_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique signature identifier (UUID v4)",
    )
    form_id: str = Field(
        ..., description="Associated form submission identifier",
    )
    signer_name: str = Field(
        ..., description="Name of the signatory",
    )
    signer_role: str = Field(
        ...,
        description="Role (producer, collector, inspector, transport_operator, buyer)",
    )
    signer_device_id: str = Field(
        ..., description="Device used for signing",
    )
    algorithm: SignatureAlgorithm = Field(
        default=SignatureAlgorithm.ECDSA_P256,
        description="Signature algorithm used",
    )
    public_key_fingerprint: Optional[str] = Field(
        None, description="Fingerprint of signer public key",
    )
    signature_bytes_hex: Optional[str] = Field(
        None, description="DER-encoded signature in hex",
    )
    signed_data_hash: Optional[str] = Field(
        None, description="SHA-256 hash of the signed data",
    )
    timestamp_binding: Optional[str] = Field(
        None,
        description="ISO 8601 timestamp included in signed payload",
    )
    visual_signature_svg: Optional[str] = Field(
        None,
        description="SVG touch-path of handwritten signature",
    )
    is_valid: bool = Field(
        default=False,
        description="Whether signature passes verification",
    )
    is_revoked: bool = Field(
        default=False,
        description="Whether signature has been revoked",
    )
    revocation_reason: Optional[str] = Field(
        None, description="Reason for revocation if revoked",
    )
    revoked_at: Optional[datetime] = Field(
        None, description="Timestamp of revocation",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional signature metadata",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp",
    )

class DataPackage(GreenLangBase):
    """A self-contained data package with Merkle root integrity.

    Attributes:
        package_id: Unique package identifier (UUID v4).
        device_id: Device that assembled the package.
        operator_id: Operator who built the package.
        status: Package lifecycle status.
        manifest: Package manifest listing all artifacts.
        merkle_root: SHA-256 Merkle root of all artifact hashes.
        merkle_tree: Full Merkle tree structure.
        provenance: Package provenance metadata.
        form_ids: Included form submission identifiers.
        gps_capture_ids: Included GPS capture identifiers.
        photo_ids: Included photo identifiers.
        signature_ids: Included signature identifiers.
        artifact_count: Total number of artifacts in package.
        package_size_bytes: Total package size in bytes.
        compression_format: Compression algorithm used.
        compression_level: Compression level (1-9).
        sealed_at: Timestamp when package was sealed.
        package_signature_hex: ECDSA signature of the manifest.
        export_format: Package export format (zip/tar_gz/json_ld).
        collection_start: Start of the collection date range.
        collection_end: End of the collection date range.
        geographic_extent: GeoJSON bounding box of all captures.
        metadata: Additional package metadata.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    package_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique package identifier (UUID v4)",
    )
    device_id: str = Field(
        ..., description="Device that assembled the package",
    )
    operator_id: str = Field(
        ..., description="Operator who built the package",
    )
    status: PackageStatus = Field(
        default=PackageStatus.BUILDING,
        description="Package lifecycle status",
    )
    manifest: Dict[str, Any] = Field(
        default_factory=dict,
        description="Package manifest listing all artifacts",
    )
    merkle_root: Optional[str] = Field(
        None, description="SHA-256 Merkle root of all artifact hashes",
    )
    merkle_tree: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full Merkle tree structure",
    )
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Package provenance metadata",
    )
    form_ids: List[str] = Field(
        default_factory=list,
        description="Included form submission identifiers",
    )
    gps_capture_ids: List[str] = Field(
        default_factory=list,
        description="Included GPS capture identifiers",
    )
    photo_ids: List[str] = Field(
        default_factory=list,
        description="Included photo identifiers",
    )
    signature_ids: List[str] = Field(
        default_factory=list,
        description="Included signature identifiers",
    )
    artifact_count: int = Field(
        default=0, ge=0,
        description="Total number of artifacts in package",
    )
    package_size_bytes: int = Field(
        default=0, ge=0,
        description="Total package size in bytes",
    )
    compression_format: str = Field(
        default="gzip",
        description="Compression algorithm used",
    )
    compression_level: int = Field(
        default=6, ge=1, le=9,
        description="Compression level (1-9)",
    )
    sealed_at: Optional[datetime] = Field(
        None, description="Timestamp when package was sealed",
    )
    package_signature_hex: Optional[str] = Field(
        None, description="ECDSA signature of the manifest",
    )
    export_format: str = Field(
        default="zip",
        description="Package export format (zip, tar_gz, json_ld)",
    )
    collection_start: Optional[datetime] = Field(
        None, description="Start of the collection date range",
    )
    collection_end: Optional[datetime] = Field(
        None, description="End of the collection date range",
    )
    geographic_extent: Optional[Dict[str, Any]] = Field(
        None,
        description="GeoJSON bounding box of all captures",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional package metadata",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="Record last update timestamp",
    )

class DeviceRegistration(GreenLangBase):
    """A registered mobile data collection device.

    Attributes:
        device_id: Unique device identifier (UUID v4).
        device_model: Device hardware model name.
        platform: Device operating system platform.
        os_version: Operating system version string.
        agent_version: Mobile Data Collector agent version.
        assigned_operator_id: Currently assigned field agent.
        assigned_area: GeoJSON polygon of assigned collection area.
        status: Current device status in the fleet.
        battery_level_pct: Last reported battery level (0-100).
        storage_total_bytes: Total device storage in bytes.
        storage_used_bytes: Used device storage in bytes.
        storage_free_bytes: Free device storage in bytes.
        last_sync_at: Timestamp of last successful sync.
        last_known_latitude: Last known GPS latitude.
        last_known_longitude: Last known GPS longitude.
        last_hdop: Last reported GPS HDOP value.
        last_satellite_count: Last reported satellite count.
        pending_forms: Count of forms awaiting sync.
        pending_photos: Count of photos awaiting sync.
        pending_gps: Count of GPS captures awaiting sync.
        connectivity_type: Last reported connectivity type.
        is_decommissioned: Whether device is decommissioned.
        decommission_reason: Reason for decommissioning.
        registered_at: Device registration timestamp.
        metadata: Additional device metadata.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    device_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique device identifier (UUID v4)",
    )
    device_model: str = Field(
        ..., description="Device hardware model name",
    )
    platform: DevicePlatform = Field(
        default=DevicePlatform.ANDROID,
        description="Device operating system platform",
    )
    os_version: str = Field(
        ..., description="Operating system version string",
    )
    agent_version: str = Field(
        default=VERSION,
        description="Mobile Data Collector agent version",
    )
    assigned_operator_id: Optional[str] = Field(
        None, description="Currently assigned field agent",
    )
    assigned_area: Optional[Dict[str, Any]] = Field(
        None,
        description="GeoJSON polygon of assigned collection area",
    )
    status: DeviceStatus = Field(
        default=DeviceStatus.ACTIVE,
        description="Current device status in the fleet",
    )
    battery_level_pct: Optional[int] = Field(
        None, ge=0, le=100,
        description="Last reported battery level (0-100)",
    )
    storage_total_bytes: Optional[int] = Field(
        None, ge=0,
        description="Total device storage in bytes",
    )
    storage_used_bytes: Optional[int] = Field(
        None, ge=0,
        description="Used device storage in bytes",
    )
    storage_free_bytes: Optional[int] = Field(
        None, ge=0,
        description="Free device storage in bytes",
    )
    last_sync_at: Optional[datetime] = Field(
        None, description="Timestamp of last successful sync",
    )
    last_known_latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0,
        description="Last known GPS latitude",
    )
    last_known_longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0,
        description="Last known GPS longitude",
    )
    last_hdop: Optional[float] = Field(
        None, ge=0.0,
        description="Last reported GPS HDOP value",
    )
    last_satellite_count: Optional[int] = Field(
        None, ge=0,
        description="Last reported satellite count",
    )
    pending_forms: int = Field(
        default=0, ge=0,
        description="Count of forms awaiting sync",
    )
    pending_photos: int = Field(
        default=0, ge=0,
        description="Count of photos awaiting sync",
    )
    pending_gps: int = Field(
        default=0, ge=0,
        description="Count of GPS captures awaiting sync",
    )
    connectivity_type: Optional[str] = Field(
        None,
        description="Last reported connectivity (none/2g/3g/4g/5g/wifi)",
    )
    is_decommissioned: bool = Field(
        default=False,
        description="Whether device is decommissioned",
    )
    decommission_reason: Optional[str] = Field(
        None, description="Reason for decommissioning",
    )
    registered_at: datetime = Field(
        default_factory=utcnow,
        description="Device registration timestamp",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional device metadata",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="Record last update timestamp",
    )

class DeviceEvent(GreenLangBase):
    """A device telemetry event for fleet monitoring.

    Attributes:
        event_id: Unique event identifier (UUID v4).
        device_id: Source device identifier.
        event_type: Telemetry event type.
        battery_level_pct: Battery percentage at event time.
        storage_total_bytes: Total storage at event time.
        storage_used_bytes: Used storage at event time.
        storage_free_bytes: Free storage at event time.
        gps_hdop: GPS HDOP at event time.
        gps_satellites: Satellite count at event time.
        gps_latitude: Latitude at event time.
        gps_longitude: Longitude at event time.
        pending_forms: Forms awaiting sync.
        pending_photos: Photos awaiting sync.
        pending_gps: GPS captures awaiting sync.
        agent_version: Agent version string.
        os_version: OS version string.
        connectivity_type: Connectivity type at event time.
        event_timestamp: Device clock timestamp.
        created_at: Server receipt timestamp.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier (UUID v4)",
    )
    device_id: str = Field(
        ..., description="Source device identifier",
    )
    event_type: str = Field(
        ...,
        description="Telemetry event type (heartbeat, sync_start, sync_complete, "
        "sync_error, low_battery, low_storage, gps_fix_lost, gps_fix_acquired)",
    )
    battery_level_pct: Optional[int] = Field(
        None, ge=0, le=100,
        description="Battery percentage at event time",
    )
    storage_total_bytes: Optional[int] = Field(
        None, ge=0,
        description="Total storage at event time",
    )
    storage_used_bytes: Optional[int] = Field(
        None, ge=0,
        description="Used storage at event time",
    )
    storage_free_bytes: Optional[int] = Field(
        None, ge=0,
        description="Free storage at event time",
    )
    gps_hdop: Optional[float] = Field(
        None, ge=0.0,
        description="GPS HDOP at event time",
    )
    gps_satellites: Optional[int] = Field(
        None, ge=0,
        description="Satellite count at event time",
    )
    gps_latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0,
        description="Latitude at event time",
    )
    gps_longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0,
        description="Longitude at event time",
    )
    pending_forms: Optional[int] = Field(
        None, ge=0,
        description="Forms awaiting sync",
    )
    pending_photos: Optional[int] = Field(
        None, ge=0,
        description="Photos awaiting sync",
    )
    pending_gps: Optional[int] = Field(
        None, ge=0,
        description="GPS captures awaiting sync",
    )
    agent_version: Optional[str] = Field(
        None, description="Agent version string",
    )
    os_version: Optional[str] = Field(
        None, description="OS version string",
    )
    connectivity_type: Optional[str] = Field(
        None,
        description="Connectivity type (none/2g/3g/4g/5g/wifi)",
    )
    event_timestamp: datetime = Field(
        default_factory=utcnow,
        description="Device clock timestamp",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Server receipt timestamp",
    )

class AuditLogEntry(GreenLangBase):
    """An immutable audit trail entry for agent operations.

    Attributes:
        audit_id: Unique audit entry identifier (UUID v4).
        entity_type: Type of entity affected.
        entity_id: Identifier of the affected entity.
        action: Action performed.
        operator_id: Operator who performed the action.
        device_id: Device from which action originated.
        before_state: State before the action.
        after_state: State after the action.
        provenance_hash: SHA-256 chain hash for tamper detection.
        ip_address: Source IP address if available.
        timestamp: Action timestamp.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    audit_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique audit entry identifier (UUID v4)",
    )
    entity_type: str = Field(
        ..., description="Type of entity affected",
    )
    entity_id: str = Field(
        ..., description="Identifier of the affected entity",
    )
    action: str = Field(
        ..., description="Action performed",
    )
    operator_id: Optional[str] = Field(
        None, description="Operator who performed the action",
    )
    device_id: Optional[str] = Field(
        None, description="Device from which action originated",
    )
    before_state: Optional[Dict[str, Any]] = Field(
        None, description="State before the action",
    )
    after_state: Optional[Dict[str, Any]] = Field(
        None, description="State after the action",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 chain hash for tamper detection",
    )
    ip_address: Optional[str] = Field(
        None, description="Source IP address if available",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="Action timestamp",
    )

# =============================================================================
# Request Models (15)
# =============================================================================

class SubmitFormRequest(GreenLangBase):
    """Request to submit a completed form from a device."""

    model_config = ConfigDict(str_strip_whitespace=True)

    device_id: str = Field(..., description="Source device identifier")
    operator_id: str = Field(..., description="Field agent identifier")
    form_type: FormType = Field(..., description="EUDR form type")
    template_id: str = Field(..., description="Template used")
    template_version: str = Field(default="1.0.0", description="Template version")
    data: Dict[str, Any] = Field(..., description="Form field values")
    commodity_type: Optional[CommodityType] = Field(None, description="EUDR commodity")
    country_code: Optional[str] = Field(None, description="ISO 3166-1 alpha-2")
    local_timestamp: Optional[datetime] = Field(None, description="Device timestamp")
    gps_capture_ids: List[str] = Field(default_factory=list, description="Linked GPS IDs")
    photo_ids: List[str] = Field(default_factory=list, description="Linked photo IDs")
    signature_ids: List[str] = Field(default_factory=list, description="Linked signature IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

class CaptureGPSRequest(GreenLangBase):
    """Request to submit a GPS point capture."""

    model_config = ConfigDict(str_strip_whitespace=True)

    device_id: str = Field(..., description="Source device identifier")
    operator_id: str = Field(..., description="Field agent identifier")
    form_id: Optional[str] = Field(None, description="Associated form ID")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="WGS84 latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="WGS84 longitude")
    altitude_m: Optional[float] = Field(None, description="Altitude meters")
    horizontal_accuracy_m: float = Field(..., ge=0.0, description="Accuracy meters")
    hdop: float = Field(..., ge=0.0, description="HDOP")
    satellite_count: int = Field(..., ge=0, description="Satellite count")
    fix_type: str = Field(default="GPS", description="Fix type")
    augmentation: Optional[str] = Field(None, description="SBAS source")
    capture_timestamp: Optional[datetime] = Field(None, description="Capture time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

class CapturePolygonRequest(GreenLangBase):
    """Request to submit a polygon boundary trace."""

    model_config = ConfigDict(str_strip_whitespace=True)

    device_id: str = Field(..., description="Source device identifier")
    operator_id: str = Field(..., description="Field agent identifier")
    form_id: Optional[str] = Field(None, description="Associated form ID")
    vertices: List[List[float]] = Field(..., description="[lat, lon] pairs")
    vertex_accuracies_m: List[float] = Field(default_factory=list, description="Per-vertex accuracy")
    capture_start: Optional[datetime] = Field(None, description="Trace start time")
    capture_end: Optional[datetime] = Field(None, description="Trace end time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

class UploadPhotoRequest(GreenLangBase):
    """Request to upload a photo with metadata."""

    model_config = ConfigDict(str_strip_whitespace=True)

    device_id: str = Field(..., description="Source device identifier")
    operator_id: str = Field(..., description="Field agent identifier")
    form_id: str = Field(..., description="Associated form ID")
    capture_id: Optional[str] = Field(None, description="Associated GPS capture ID")
    photo_type: PhotoType = Field(..., description="Photo category")
    file_name: str = Field(..., description="Original file name")
    file_size_bytes: int = Field(..., ge=0, description="File size bytes")
    file_format: str = Field(default="jpeg", description="Image format")
    width_px: int = Field(..., ge=1, description="Width pixels")
    height_px: int = Field(..., ge=1, description="Height pixels")
    integrity_hash: str = Field(..., description="SHA-256 hash of raw bytes")
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0, description="GPS latitude")
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0, description="GPS longitude")
    exif_timestamp: Optional[datetime] = Field(None, description="EXIF timestamp")
    device_timestamp: Optional[datetime] = Field(None, description="Device time")
    compression_quality: Optional[int] = Field(None, ge=1, le=100, description="Quality")
    annotation: Optional[str] = Field(None, description="Text annotation")
    sequence_number: Optional[int] = Field(None, ge=1, description="Sequence in batch")
    batch_group_id: Optional[str] = Field(None, description="Batch group ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

class TriggerSyncRequest(GreenLangBase):
    """Request to trigger sync for a device."""

    model_config = ConfigDict(str_strip_whitespace=True)

    device_id: str = Field(..., description="Device to sync")
    force: bool = Field(default=False, description="Force immediate sync")
    max_items: Optional[int] = Field(None, ge=1, description="Max items to sync")
    item_types: Optional[List[str]] = Field(None, description="Filter item types")

class ResolveConflictRequest(GreenLangBase):
    """Request to resolve a sync conflict."""

    model_config = ConfigDict(str_strip_whitespace=True)

    conflict_id: str = Field(..., description="Conflict to resolve")
    resolution_strategy: ConflictResolution = Field(..., description="Resolution strategy")
    resolved_value: Any = Field(None, description="Chosen resolved value")
    resolved_by: str = Field(..., description="Operator resolving")
    reason: Optional[str] = Field(None, description="Resolution reason")

class CreateTemplateRequest(GreenLangBase):
    """Request to create a form template."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Template name")
    form_type: FormType = Field(..., description="EUDR form type")
    template_type: TemplateType = Field(default=TemplateType.BASE, description="Template type")
    parent_template_id: Optional[str] = Field(None, description="Parent for inheritance")
    schema_definition: Dict[str, Any] = Field(default_factory=dict, description="JSON schema")
    fields: List[Dict[str, Any]] = Field(default_factory=list, description="Field definitions")
    conditional_logic: List[Dict[str, Any]] = Field(default_factory=list, description="Logic rules")
    validation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Validation rules")
    language_packs: Dict[str, Dict[str, str]] = Field(default_factory=dict, description="Translations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

class UpdateTemplateRequest(GreenLangBase):
    """Request to update a form template."""

    model_config = ConfigDict(str_strip_whitespace=True)

    template_id: str = Field(..., description="Template to update")
    name: Optional[str] = Field(None, description="Updated name")
    schema_definition: Optional[Dict[str, Any]] = Field(None, description="Updated schema")
    fields: Optional[List[Dict[str, Any]]] = Field(None, description="Updated fields")
    conditional_logic: Optional[List[Dict[str, Any]]] = Field(None, description="Updated logic")
    validation_rules: Optional[List[Dict[str, Any]]] = Field(None, description="Updated rules")
    language_packs: Optional[Dict[str, Dict[str, str]]] = Field(None, description="Updated translations")
    is_active: Optional[bool] = Field(None, description="Updated active status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")

class CaptureSignatureRequest(GreenLangBase):
    """Request to submit a captured digital signature."""

    model_config = ConfigDict(str_strip_whitespace=True)

    form_id: str = Field(..., description="Associated form ID")
    signer_name: str = Field(..., description="Signer name")
    signer_role: str = Field(..., description="Signer role")
    signer_device_id: str = Field(..., description="Signing device ID")
    algorithm: SignatureAlgorithm = Field(default=SignatureAlgorithm.ECDSA_P256, description="Algorithm")
    public_key_fingerprint: Optional[str] = Field(None, description="Public key fingerprint")
    signature_bytes_hex: Optional[str] = Field(None, description="DER signature hex")
    signed_data_hash: Optional[str] = Field(None, description="Signed data hash")
    timestamp_binding: Optional[str] = Field(None, description="Timestamp binding ISO 8601")
    visual_signature_svg: Optional[str] = Field(None, description="SVG touch path")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

class BuildPackageRequest(GreenLangBase):
    """Request to build a data package."""

    model_config = ConfigDict(str_strip_whitespace=True)

    device_id: str = Field(..., description="Source device ID")
    operator_id: str = Field(..., description="Operator ID")
    form_ids: List[str] = Field(default_factory=list, description="Forms to include")
    gps_capture_ids: List[str] = Field(default_factory=list, description="GPS captures to include")
    photo_ids: List[str] = Field(default_factory=list, description="Photos to include")
    signature_ids: List[str] = Field(default_factory=list, description="Signatures to include")
    compression_format: str = Field(default="gzip", description="Compression format")
    compression_level: int = Field(default=6, ge=1, le=9, description="Compression level")
    export_format: str = Field(default="zip", description="Export format")
    seal: bool = Field(default=True, description="Seal package after build")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

class RegisterDeviceRequest(GreenLangBase):
    """Request to register a new device in the fleet."""

    model_config = ConfigDict(str_strip_whitespace=True)

    device_model: str = Field(..., description="Device model name")
    platform: DevicePlatform = Field(default=DevicePlatform.ANDROID, description="OS platform")
    os_version: str = Field(..., description="OS version")
    agent_version: str = Field(default=VERSION, description="Agent version")
    assigned_operator_id: Optional[str] = Field(None, description="Assigned operator")
    assigned_area: Optional[Dict[str, Any]] = Field(None, description="Collection area GeoJSON")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

class UpdateDeviceRequest(GreenLangBase):
    """Request to update a device registration."""

    model_config = ConfigDict(str_strip_whitespace=True)

    device_id: str = Field(..., description="Device to update")
    assigned_operator_id: Optional[str] = Field(None, description="New operator")
    assigned_area: Optional[Dict[str, Any]] = Field(None, description="New collection area")
    agent_version: Optional[str] = Field(None, description="Updated agent version")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")

class SearchFormsRequest(GreenLangBase):
    """Request to search form submissions with filters."""

    model_config = ConfigDict(str_strip_whitespace=True)

    form_type: Optional[FormType] = Field(None, description="Filter by form type")
    status: Optional[FormStatus] = Field(None, description="Filter by status")
    device_id: Optional[str] = Field(None, description="Filter by device")
    operator_id: Optional[str] = Field(None, description="Filter by operator")
    commodity_type: Optional[CommodityType] = Field(None, description="Filter by commodity")
    country_code: Optional[str] = Field(None, description="Filter by country")
    date_from: Optional[datetime] = Field(None, description="Start date filter")
    date_to: Optional[datetime] = Field(None, description="End date filter")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=50, ge=1, le=500, description="Page size")

class GetDeviceStatusRequest(GreenLangBase):
    """Request to get device status and telemetry."""

    model_config = ConfigDict(str_strip_whitespace=True)

    device_id: str = Field(..., description="Device to query")
    include_telemetry: bool = Field(default=True, description="Include telemetry history")
    telemetry_limit: int = Field(default=100, ge=1, le=1000, description="Max telemetry events")

class ValidateFormRequest(GreenLangBase):
    """Request to validate a form against its template schema."""

    model_config = ConfigDict(str_strip_whitespace=True)

    form_id: Optional[str] = Field(None, description="Existing form ID to validate")
    template_id: str = Field(..., description="Template to validate against")
    data: Dict[str, Any] = Field(..., description="Form data to validate")
    strictness: str = Field(default="strict", description="Validation strictness")

# =============================================================================
# Response Models (15)
# =============================================================================

class FormResponse(GreenLangBase):
    """Response for form submission operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    form_id: str = Field(..., description="Form submission ID")
    status: FormStatus = Field(..., description="Current form status")
    submission_hash: Optional[str] = Field(None, description="SHA-256 hash")
    provenance_hash: Optional[str] = Field(None, description="Provenance chain hash")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    message: str = Field(default="", description="Status message")
    form: Optional[FormSubmission] = Field(None, description="Full form data")

class GPSResponse(GreenLangBase):
    """Response for GPS capture operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    capture_id: str = Field(..., description="GPS capture ID")
    accuracy_tier: CaptureAccuracyTier = Field(..., description="Accuracy classification")
    provenance_hash: Optional[str] = Field(None, description="Provenance chain hash")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    message: str = Field(default="", description="Status message")
    capture: Optional[GPSCapture] = Field(None, description="Full capture data")

class PolygonResponse(GreenLangBase):
    """Response for polygon trace operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    polygon_id: str = Field(..., description="Polygon trace ID")
    area_ha: Optional[float] = Field(None, description="Calculated area hectares")
    vertex_count: int = Field(default=0, description="Number of vertices")
    is_valid: bool = Field(default=False, description="Geometry validity")
    provenance_hash: Optional[str] = Field(None, description="Provenance chain hash")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    message: str = Field(default="", description="Status message")
    polygon: Optional[PolygonTrace] = Field(None, description="Full polygon data")

class PhotoResponse(GreenLangBase):
    """Response for photo upload operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    photo_id: str = Field(..., description="Photo evidence ID")
    integrity_hash: str = Field(..., description="SHA-256 integrity hash")
    provenance_hash: Optional[str] = Field(None, description="Provenance chain hash")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    message: str = Field(default="", description="Status message")
    photo: Optional[PhotoEvidence] = Field(None, description="Full photo data")

class SyncResponse(GreenLangBase):
    """Response for sync trigger operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    device_id: str = Field(..., description="Device ID")
    items_queued: int = Field(default=0, description="Items queued")
    items_completed: int = Field(default=0, description="Items completed")
    items_failed: int = Field(default=0, description="Items failed")
    conflicts_detected: int = Field(default=0, description="Conflicts found")
    bytes_uploaded: int = Field(default=0, description="Bytes uploaded")
    processing_time_ms: float = Field(default=0.0, description="Sync duration")
    message: str = Field(default="", description="Status message")

class ConflictResponse(GreenLangBase):
    """Response for conflict resolution operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    conflict_id: str = Field(..., description="Conflict ID")
    resolved: bool = Field(default=False, description="Resolution status")
    resolution_strategy: ConflictResolution = Field(..., description="Strategy used")
    resolved_value: Any = Field(None, description="Resolved value")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    message: str = Field(default="", description="Status message")
    conflict: Optional[SyncConflict] = Field(None, description="Full conflict data")

class TemplateResponse(GreenLangBase):
    """Response for template management operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    template_id: str = Field(..., description="Template ID")
    version: str = Field(default="1.0.0", description="Template version")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    message: str = Field(default="", description="Status message")
    template: Optional[FormTemplate] = Field(None, description="Full template data")

class SignatureResponse(GreenLangBase):
    """Response for signature capture/verification operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    signature_id: str = Field(..., description="Signature ID")
    is_valid: bool = Field(default=False, description="Verification result")
    provenance_hash: Optional[str] = Field(None, description="Provenance chain hash")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    message: str = Field(default="", description="Status message")
    signature: Optional[DigitalSignature] = Field(None, description="Full signature data")

class PackageResponse(GreenLangBase):
    """Response for data package operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    package_id: str = Field(..., description="Package ID")
    status: PackageStatus = Field(..., description="Package status")
    merkle_root: Optional[str] = Field(None, description="Merkle root hash")
    artifact_count: int = Field(default=0, description="Artifact count")
    package_size_bytes: int = Field(default=0, description="Package size")
    provenance_hash: Optional[str] = Field(None, description="Provenance chain hash")
    processing_time_ms: float = Field(default=0.0, description="Build duration")
    message: str = Field(default="", description="Status message")
    package: Optional[DataPackage] = Field(None, description="Full package data")

class DeviceResponse(GreenLangBase):
    """Response for device registration/update operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    device_id: str = Field(..., description="Device ID")
    status: DeviceStatus = Field(..., description="Device status")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    message: str = Field(default="", description="Status message")
    device: Optional[DeviceRegistration] = Field(None, description="Full device data")

class DeviceStatusResponse(GreenLangBase):
    """Response for device status query operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    device_id: str = Field(..., description="Device ID")
    status: DeviceStatus = Field(..., description="Current status")
    battery_level_pct: Optional[int] = Field(None, description="Battery level")
    storage_free_bytes: Optional[int] = Field(None, description="Free storage")
    pending_sync_items: int = Field(default=0, description="Pending sync items")
    last_sync_at: Optional[datetime] = Field(None, description="Last sync time")
    telemetry_events: List[DeviceEvent] = Field(default_factory=list, description="Recent telemetry")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    message: str = Field(default="", description="Status message")

class SearchResponse(GreenLangBase):
    """Response for form search operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    forms: List[FormSubmission] = Field(default_factory=list, description="Matching forms")
    total_count: int = Field(default=0, description="Total matches")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=50, description="Page size")
    processing_time_ms: float = Field(default=0.0, description="Query duration")
    message: str = Field(default="", description="Status message")

class SyncStatusResponse(GreenLangBase):
    """Response for sync status query operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    device_id: str = Field(..., description="Device ID")
    pending_items: int = Field(default=0, description="Items pending")
    in_progress_items: int = Field(default=0, description="Items in progress")
    completed_items: int = Field(default=0, description="Items completed")
    failed_items: int = Field(default=0, description="Items failed")
    total_bytes_pending: int = Field(default=0, description="Bytes pending")
    last_sync_at: Optional[datetime] = Field(None, description="Last sync time")
    unresolved_conflicts: int = Field(default=0, description="Unresolved conflicts")
    processing_time_ms: float = Field(default=0.0, description="Query duration")
    message: str = Field(default="", description="Status message")

class FleetStatusResponse(GreenLangBase):
    """Response for fleet dashboard queries."""

    model_config = ConfigDict(str_strip_whitespace=True)

    total_devices: int = Field(default=0, description="Total registered devices")
    active_devices: int = Field(default=0, description="Devices synced recently")
    offline_devices: int = Field(default=0, description="Devices not synced")
    low_battery_devices: int = Field(default=0, description="Low battery devices")
    low_storage_devices: int = Field(default=0, description="Low storage devices")
    decommissioned_devices: int = Field(default=0, description="Decommissioned devices")
    outdated_agent_devices: int = Field(default=0, description="Outdated agent version")
    total_pending_sync_bytes: int = Field(default=0, description="Total pending sync")
    total_pending_forms: int = Field(default=0, description="Total pending forms")
    total_pending_photos: int = Field(default=0, description="Total pending photos")
    processing_time_ms: float = Field(default=0.0, description="Query duration")
    message: str = Field(default="", description="Status message")

class HealthResponse(GreenLangBase):
    """Health check response for the Mobile Data Collector Agent."""

    model_config = ConfigDict(str_strip_whitespace=True)

    status: str = Field(
        default="healthy", description="Service status",
    )
    version: str = Field(
        default=VERSION, description="Agent version",
    )
    agent_id: str = Field(
        default="GL-EUDR-MDC-015",
        description="Agent identifier",
    )
    database_connected: bool = Field(
        default=False, description="Database connectivity",
    )
    redis_connected: bool = Field(
        default=False, description="Redis connectivity",
    )
    active_devices: int = Field(
        default=0, description="Active devices in fleet",
    )
    pending_sync_items: int = Field(
        default=0, description="Total pending sync items",
    )
    unresolved_conflicts: int = Field(
        default=0, description="Unresolved conflicts",
    )
    uptime_seconds: float = Field(
        default=0.0, description="Service uptime in seconds",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="Health check timestamp",
    )
