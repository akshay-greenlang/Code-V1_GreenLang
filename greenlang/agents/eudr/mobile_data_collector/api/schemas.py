# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-015 Mobile Data Collector

Pydantic v2 request/response models for all Mobile Data Collector REST
API endpoints. Organized by domain: forms, GPS, photos, sync, templates,
signatures, packages, devices, and common utilities.

All models use ``ConfigDict(from_attributes=True)`` for ORM compatibility
and ``Field()`` with descriptions for OpenAPI documentation.

All hash fields use deterministic SHA-256 algorithms required by
EUDR Article 14 for five-year audit trail compliance.

Model Count: 70+ schemas covering 70+ endpoints.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015, Section 7.4
Agent ID: GL-EUDR-MDC-015
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# Helpers
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


# =============================================================================
# Enumerations (API-layer mirrors of domain enums)
# =============================================================================


class FormStatusSchema(str, Enum):
    """Form submission lifecycle status."""

    DRAFT = "draft"
    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"


class FormTypeSchema(str, Enum):
    """EUDR form type classifications."""

    PRODUCER_REGISTRATION = "producer_registration"
    PLOT_SURVEY = "plot_survey"
    HARVEST_LOG = "harvest_log"
    CUSTODY_TRANSFER = "custody_transfer"
    QUALITY_INSPECTION = "quality_inspection"
    SMALLHOLDER_DECLARATION = "smallholder_declaration"


class AccuracyTierSchema(str, Enum):
    """GPS capture accuracy classification."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    REJECTED = "rejected"


class PhotoTypeSchema(str, Enum):
    """Photo evidence category."""

    PLOT_PHOTO = "plot_photo"
    COMMODITY_PHOTO = "commodity_photo"
    DOCUMENT_PHOTO = "document_photo"
    FACILITY_PHOTO = "facility_photo"
    TRANSPORT_PHOTO = "transport_photo"
    IDENTITY_PHOTO = "identity_photo"


class SyncStatusSchema(str, Enum):
    """Synchronization queue item status."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PERMANENTLY_FAILED = "permanently_failed"


class ConflictResolutionSchema(str, Enum):
    """Sync conflict resolution strategy."""

    SERVER_WINS = "server_wins"
    CLIENT_WINS = "client_wins"
    MANUAL = "manual"
    LWW = "lww"
    SET_UNION = "set_union"
    STATE_MACHINE = "state_machine"


class TemplateTypeSchema(str, Enum):
    """Form template type."""

    BASE = "base"
    CUSTOM = "custom"
    INHERITED = "inherited"


class TemplateStatusSchema(str, Enum):
    """Form template lifecycle status."""

    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"


class SignatureAlgorithmSchema(str, Enum):
    """Digital signature algorithm."""

    ECDSA_P256 = "ecdsa_p256"
    ECDSA_P384 = "ecdsa_p384"


class PackageStatusSchema(str, Enum):
    """Data package lifecycle status."""

    BUILDING = "building"
    SEALED = "sealed"
    UPLOADED = "uploaded"
    VERIFIED = "verified"
    EXPIRED = "expired"


class DeviceStatusSchema(str, Enum):
    """Mobile device status."""

    ACTIVE = "active"
    OFFLINE = "offline"
    LOW_BATTERY = "low_battery"
    LOW_STORAGE = "low_storage"
    DECOMMISSIONED = "decommissioned"


class DevicePlatformSchema(str, Enum):
    """Mobile device OS platform."""

    ANDROID = "android"
    IOS = "ios"
    HARMONYOS = "harmonyos"


class CommodityTypeSchema(str, Enum):
    """EUDR-regulated commodity types per EU 2023/1115 Article 1."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class ExportFormatSchema(str, Enum):
    """Data package export format."""

    ZIP = "zip"
    TAR_GZ = "tar_gz"
    JSON_LD = "json_ld"


# =============================================================================
# Common Schemas
# =============================================================================


class PaginationSchema(BaseModel):
    """Pagination metadata for list responses.

    Attributes:
        total: Total number of records matching the query.
        page: Current page number.
        page_size: Records per page.
        has_more: Whether more records exist beyond this page.
    """

    model_config = ConfigDict(from_attributes=True)

    total: int = Field(..., ge=0, description="Total matching records")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Records per page")
    has_more: bool = Field(..., description="More records exist")


class ErrorSchema(BaseModel):
    """Standard error response.

    Attributes:
        error: Error type identifier.
        message: Human-readable error message.
        detail: Additional error details.
        request_id: Request correlation ID.
    """

    model_config = ConfigDict(from_attributes=True)

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(
        None, description="Request correlation ID"
    )


class HealthComponentSchema(BaseModel):
    """Health status for a single service component.

    Attributes:
        name: Component name.
        status: Component health status.
        latency_ms: Component response latency in milliseconds.
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


class HealthSchema(BaseModel):
    """Health check response for the Mobile Data Collector API.

    Attributes:
        service: Service identifier.
        status: Overall health status.
        version: Service version.
        agent_id: Agent identifier.
        uptime_seconds: Service uptime in seconds.
        active_devices: Number of active fleet devices.
        pending_sync_items: Total items awaiting sync.
        unresolved_conflicts: Unresolved sync conflicts.
        components: Component health details.
        checked_at: Health check timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    service: str = Field(
        default="eudr-mobile-data-collector",
        description="Service identifier",
    )
    status: str = Field(default="healthy", description="Overall health")
    version: str = Field(default="1.0.0", description="Service version")
    agent_id: str = Field(
        default="GL-EUDR-MDC-015", description="Agent identifier"
    )
    uptime_seconds: float = Field(
        default=0.0, ge=0.0, description="Uptime in seconds"
    )
    active_devices: int = Field(
        default=0, ge=0, description="Active fleet devices"
    )
    pending_sync_items: int = Field(
        default=0, ge=0, description="Total pending sync items"
    )
    unresolved_conflicts: int = Field(
        default=0, ge=0, description="Unresolved sync conflicts"
    )
    components: List[HealthComponentSchema] = Field(
        default_factory=lambda: [
            HealthComponentSchema(name="api", status="healthy"),
            HealthComponentSchema(name="database", status="healthy"),
            HealthComponentSchema(name="cache", status="healthy"),
            HealthComponentSchema(name="sync_engine", status="healthy"),
            HealthComponentSchema(name="fleet_manager", status="healthy"),
        ],
        description="Component health details",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow, description="Health check timestamp"
    )


class SuccessSchema(BaseModel):
    """Standard success response.

    Attributes:
        status: Response status (always 'success').
        message: Success message.
        data: Optional response payload.
    """

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(default="success", description="Response status")
    message: str = Field(default="", description="Success message")
    data: Optional[Any] = Field(None, description="Response payload")


class ProvenanceInfo(BaseModel):
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
        default_factory=_utcnow, description="Provenance timestamp"
    )


# =============================================================================
# Form Schemas
# =============================================================================


class FormSubmitSchema(BaseModel):
    """Request to submit a new form from a device.

    Attributes:
        device_id: Source device identifier.
        operator_id: Field agent identifier.
        form_type: EUDR form type classification.
        template_id: Form template used for data collection.
        template_version: Semantic version of the template.
        data: Form field values as key-value pairs.
        commodity_type: EUDR commodity type if applicable.
        country_code: ISO 3166-1 alpha-2 country code.
        local_timestamp: Device-local timestamp at submission.
        gps_capture_ids: Linked GPS capture identifiers.
        photo_ids: Linked photo evidence identifiers.
        signature_ids: Linked digital signature identifiers.
        metadata: Additional form metadata.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "device_id": "dev-001-abc",
                "operator_id": "op-john-doe",
                "form_type": "plot_survey",
                "template_id": "tpl-plot-survey-v1",
                "template_version": "1.0.0",
                "data": {
                    "producer_name": "Jean Mbeki",
                    "plot_name": "Plot A",
                    "commodity": "cocoa",
                    "area_ha": 2.5,
                },
                "commodity_type": "cocoa",
                "country_code": "CI",
            }
        },
    )

    device_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Source device identifier",
    )
    operator_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Field agent identifier",
    )
    form_type: FormTypeSchema = Field(
        ..., description="EUDR form type classification",
    )
    template_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Form template used for data collection",
    )
    template_version: str = Field(
        default="1.0.0", max_length=20,
        description="Semantic version of the template",
    )
    data: Dict[str, Any] = Field(
        ..., description="Form field values as key-value pairs",
    )
    commodity_type: Optional[CommodityTypeSchema] = Field(
        None, description="EUDR commodity type if applicable",
    )
    country_code: Optional[str] = Field(
        None, min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    local_timestamp: Optional[datetime] = Field(
        None, description="Device-local timestamp at submission",
    )
    gps_capture_ids: List[str] = Field(
        default_factory=list, description="Linked GPS capture identifiers",
    )
    photo_ids: List[str] = Field(
        default_factory=list, description="Linked photo evidence identifiers",
    )
    signature_ids: List[str] = Field(
        default_factory=list,
        description="Linked digital signature identifiers",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional form metadata",
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


class FormUpdateSchema(BaseModel):
    """Request to update a draft form.

    Attributes:
        data: Updated form field values.
        commodity_type: Updated EUDR commodity type.
        country_code: Updated ISO 3166-1 alpha-2 country code.
        gps_capture_ids: Updated linked GPS capture identifiers.
        photo_ids: Updated linked photo evidence identifiers.
        signature_ids: Updated linked digital signature identifiers.
        metadata: Updated additional metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    data: Optional[Dict[str, Any]] = Field(
        None, description="Updated form field values",
    )
    commodity_type: Optional[CommodityTypeSchema] = Field(
        None, description="Updated EUDR commodity type",
    )
    country_code: Optional[str] = Field(
        None, min_length=2, max_length=2,
        description="Updated ISO 3166-1 alpha-2 country code",
    )
    gps_capture_ids: Optional[List[str]] = Field(
        None, description="Updated linked GPS capture identifiers",
    )
    photo_ids: Optional[List[str]] = Field(
        None, description="Updated linked photo evidence identifiers",
    )
    signature_ids: Optional[List[str]] = Field(
        None, description="Updated linked digital signature identifiers",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Updated additional metadata",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate ISO 3166-1 alpha-2 format."""
        if v is not None:
            v = v.upper().strip()
            if len(v) != 2 or not v.isalpha():
                raise ValueError(
                    f"country_code must be 2-letter ISO 3166-1 alpha-2, "
                    f"got '{v}'"
                )
        return v


class FormResponseSchema(BaseModel):
    """Response for form submission operations.

    Attributes:
        form_id: Form submission identifier.
        status: Current form lifecycle status.
        form_type: EUDR form type.
        device_id: Source device identifier.
        operator_id: Field agent identifier.
        template_id: Template used.
        template_version: Template version.
        data: Form field values.
        commodity_type: EUDR commodity type.
        country_code: ISO country code.
        submission_hash: SHA-256 hash of form data.
        gps_capture_ids: Linked GPS captures.
        photo_ids: Linked photos.
        signature_ids: Linked signatures.
        metadata: Additional metadata.
        provenance: Provenance tracking.
        processing_time_ms: Processing duration in milliseconds.
        message: Status message.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    form_id: str = Field(
        default_factory=_new_id, description="Form submission identifier"
    )
    status: str = Field(default="draft", description="Form lifecycle status")
    form_type: str = Field(..., description="EUDR form type")
    device_id: str = Field(..., description="Source device identifier")
    operator_id: str = Field(..., description="Field agent identifier")
    template_id: str = Field(..., description="Template used")
    template_version: str = Field(
        default="1.0.0", description="Template version"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Form field values"
    )
    commodity_type: Optional[str] = Field(
        None, description="EUDR commodity type"
    )
    country_code: Optional[str] = Field(None, description="ISO country code")
    submission_hash: Optional[str] = Field(
        None, description="SHA-256 hash of form data"
    )
    gps_capture_ids: List[str] = Field(
        default_factory=list, description="Linked GPS captures"
    )
    photo_ids: List[str] = Field(
        default_factory=list, description="Linked photos"
    )
    signature_ids: List[str] = Field(
        default_factory=list, description="Linked signatures"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    message: str = Field(default="", description="Status message")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Last update timestamp"
    )


class FormListSchema(BaseModel):
    """Response listing forms with pagination.

    Attributes:
        forms: List of form response records.
        pagination: Pagination metadata.
        processing_time_ms: Query duration in milliseconds.
    """

    model_config = ConfigDict(from_attributes=True)

    forms: List[FormResponseSchema] = Field(
        default_factory=list, description="Form records"
    )
    pagination: PaginationSchema = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Query duration in ms"
    )


class FormValidationSchema(BaseModel):
    """Response for form validation operations.

    Attributes:
        form_id: Form identifier validated.
        template_id: Template validated against.
        is_valid: Whether form passes all validation rules.
        completeness_score: Completeness percentage (0-100).
        errors: List of validation errors.
        warnings: List of validation warnings.
        missing_fields: List of required fields that are missing.
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(from_attributes=True)

    form_id: Optional[str] = Field(
        None, description="Form identifier validated"
    )
    template_id: str = Field(..., description="Template validated against")
    is_valid: bool = Field(
        ..., description="Whether form passes all validation rules"
    )
    completeness_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Completeness percentage (0-100)",
    )
    errors: List[str] = Field(
        default_factory=list, description="Validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Validation warnings"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Required fields that are missing",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )


# =============================================================================
# GPS Schemas
# =============================================================================


class GPSCaptureSchema(BaseModel):
    """Request to record a GPS point capture.

    Attributes:
        device_id: Source device identifier.
        operator_id: Field agent identifier.
        form_id: Associated form submission identifier.
        latitude: WGS84 latitude in decimal degrees.
        longitude: WGS84 longitude in decimal degrees.
        altitude_m: Altitude above sea level in meters.
        horizontal_accuracy_m: Estimated horizontal accuracy in meters.
        vertical_accuracy_m: Estimated vertical accuracy in meters.
        hdop: Horizontal Dilution of Precision.
        satellite_count: Number of satellites used in fix.
        fix_type: GPS constellation type.
        augmentation: SBAS augmentation source.
        capture_timestamp: Device timestamp at capture.
        metadata: Additional capture metadata.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "device_id": "dev-001-abc",
                "operator_id": "op-john-doe",
                "latitude": 5.359952,
                "longitude": -3.974578,
                "horizontal_accuracy_m": 2.1,
                "hdop": 1.2,
                "satellite_count": 10,
                "fix_type": "GPS+GLONASS",
            }
        },
    )

    device_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Source device identifier",
    )
    operator_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Field agent identifier",
    )
    form_id: Optional[str] = Field(
        None, max_length=255,
        description="Associated form submission identifier",
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
        ..., ge=0.0, description="Horizontal Dilution of Precision",
    )
    satellite_count: int = Field(
        ..., ge=0, description="Number of satellites used in fix",
    )
    fix_type: str = Field(
        default="GPS", max_length=50,
        description="GPS constellation type (GPS, GLONASS, Galileo, BeiDou, combined)",
    )
    augmentation: Optional[str] = Field(
        None, max_length=50,
        description="SBAS augmentation source (WAAS, EGNOS, MSAS, GAGAN)",
    )
    capture_timestamp: Optional[datetime] = Field(
        None, description="Device timestamp at capture",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional capture metadata",
    )


class PolygonCaptureSchema(BaseModel):
    """Request to record a polygon boundary trace.

    Attributes:
        device_id: Source device identifier.
        operator_id: Field agent identifier.
        form_id: Associated form submission identifier.
        vertices: List of [latitude, longitude] coordinate pairs.
        vertex_accuracies_m: Per-vertex horizontal accuracy in meters.
        capture_start: Timestamp when tracing started.
        capture_end: Timestamp when tracing completed.
        metadata: Additional polygon metadata.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "device_id": "dev-001-abc",
                "operator_id": "op-john-doe",
                "form_id": "form-abc-123",
                "vertices": [
                    [5.359952, -3.974578],
                    [5.360100, -3.974200],
                    [5.359800, -3.973900],
                    [5.359600, -3.974300],
                ],
                "vertex_accuracies_m": [2.1, 2.5, 3.0, 2.8],
            }
        },
    )

    device_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Source device identifier",
    )
    operator_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Field agent identifier",
    )
    form_id: Optional[str] = Field(
        None, max_length=255,
        description="Associated form submission identifier",
    )
    vertices: List[List[float]] = Field(
        ..., min_length=3,
        description="List of [latitude, longitude] coordinate pairs",
    )
    vertex_accuracies_m: List[float] = Field(
        default_factory=list,
        description="Per-vertex horizontal accuracy in meters",
    )
    capture_start: Optional[datetime] = Field(
        None, description="Timestamp when tracing started",
    )
    capture_end: Optional[datetime] = Field(
        None, description="Timestamp when tracing completed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional polygon metadata",
    )

    @field_validator("vertices")
    @classmethod
    def validate_vertices(
        cls, v: List[List[float]]
    ) -> List[List[float]]:
        """Validate each vertex has exactly 2 coordinates in valid range."""
        for i, vertex in enumerate(v):
            if len(vertex) != 2:
                raise ValueError(
                    f"Vertex {i} must have exactly 2 coordinates "
                    f"[lat, lon], got {len(vertex)}"
                )
            lat, lon = vertex
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(
                    f"Vertex {i} latitude must be in [-90, 90], got {lat}"
                )
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(
                    f"Vertex {i} longitude must be in [-180, 180], got {lon}"
                )
        return v


class GPSResponseSchema(BaseModel):
    """Response for GPS point capture operations.

    Attributes:
        capture_id: GPS capture identifier.
        latitude: WGS84 latitude.
        longitude: WGS84 longitude.
        altitude_m: Altitude in meters.
        horizontal_accuracy_m: Horizontal accuracy in meters.
        hdop: HDOP value.
        satellite_count: Satellite count.
        accuracy_tier: Calculated accuracy classification.
        fix_type: GPS fix type.
        augmentation: SBAS augmentation source.
        form_id: Associated form identifier.
        srid: Spatial Reference Identifier.
        provenance: Provenance tracking.
        processing_time_ms: Processing duration in milliseconds.
        message: Status message.
        created_at: Creation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    capture_id: str = Field(
        default_factory=_new_id, description="GPS capture identifier"
    )
    latitude: float = Field(..., description="WGS84 latitude")
    longitude: float = Field(..., description="WGS84 longitude")
    altitude_m: Optional[float] = Field(None, description="Altitude meters")
    horizontal_accuracy_m: float = Field(
        ..., description="Horizontal accuracy meters"
    )
    hdop: float = Field(..., description="HDOP value")
    satellite_count: int = Field(..., description="Satellite count")
    accuracy_tier: str = Field(
        default="acceptable", description="Accuracy classification"
    )
    fix_type: str = Field(default="GPS", description="GPS fix type")
    augmentation: Optional[str] = Field(
        None, description="SBAS augmentation source"
    )
    form_id: Optional[str] = Field(
        None, description="Associated form identifier"
    )
    srid: int = Field(default=4326, description="Spatial Reference Identifier")
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    message: str = Field(default="", description="Status message")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )


class PolygonResponseSchema(BaseModel):
    """Response for polygon trace operations.

    Attributes:
        polygon_id: Polygon trace identifier.
        vertex_count: Number of vertices.
        area_ha: Calculated area in hectares.
        perimeter_m: Calculated perimeter in meters.
        is_closed: Whether polygon is closed.
        is_valid: Whether polygon passes geometry checks.
        form_id: Associated form identifier.
        srid: Spatial Reference Identifier.
        provenance: Provenance tracking.
        processing_time_ms: Processing duration in milliseconds.
        message: Status message.
        created_at: Creation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    polygon_id: str = Field(
        default_factory=_new_id, description="Polygon trace identifier"
    )
    vertex_count: int = Field(default=0, ge=0, description="Number of vertices")
    area_ha: Optional[float] = Field(
        None, ge=0.0, description="Calculated area hectares"
    )
    perimeter_m: Optional[float] = Field(
        None, ge=0.0, description="Calculated perimeter meters"
    )
    is_closed: bool = Field(
        default=False, description="Whether polygon is closed"
    )
    is_valid: bool = Field(
        default=False, description="Whether polygon passes geometry checks"
    )
    form_id: Optional[str] = Field(
        None, description="Associated form identifier"
    )
    srid: int = Field(default=4326, description="Spatial Reference Identifier")
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    message: str = Field(default="", description="Status message")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )


class AreaResponseSchema(BaseModel):
    """Response for polygon area calculation.

    Attributes:
        area_ha: Calculated area in hectares.
        area_sq_m: Calculated area in square meters.
        perimeter_m: Calculated perimeter in meters.
        vertex_count: Number of vertices used.
        is_valid: Whether the polygon geometry is valid.
        crs: Coordinate reference system used.
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(from_attributes=True)

    area_ha: float = Field(..., ge=0.0, description="Area in hectares")
    area_sq_m: float = Field(..., ge=0.0, description="Area in square meters")
    perimeter_m: float = Field(..., ge=0.0, description="Perimeter in meters")
    vertex_count: int = Field(..., ge=0, description="Vertices used")
    is_valid: bool = Field(..., description="Geometry validity")
    crs: str = Field(default="WGS84", description="CRS used")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )


class GPSValidateSchema(BaseModel):
    """Request to validate GPS coordinates.

    Attributes:
        latitude: WGS84 latitude to validate.
        longitude: WGS84 longitude to validate.
        expected_country_code: Expected country (ISO 3166-1 alpha-2).
        expected_commodity: Expected EUDR commodity context.
    """

    model_config = ConfigDict(from_attributes=True)

    latitude: float = Field(
        ..., ge=-90.0, le=90.0, description="WGS84 latitude"
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0, description="WGS84 longitude"
    )
    expected_country_code: Optional[str] = Field(
        None, min_length=2, max_length=2,
        description="Expected country (ISO 3166-1 alpha-2)",
    )
    expected_commodity: Optional[CommodityTypeSchema] = Field(
        None, description="Expected EUDR commodity context",
    )


class GPSValidateResponseSchema(BaseModel):
    """Response for GPS coordinate validation.

    Attributes:
        is_valid: Whether coordinates are valid.
        latitude: Validated latitude.
        longitude: Validated longitude.
        country_match: Whether coordinates match expected country.
        warnings: Validation warnings.
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(from_attributes=True)

    is_valid: bool = Field(..., description="Whether coordinates are valid")
    latitude: float = Field(..., description="Validated latitude")
    longitude: float = Field(..., description="Validated longitude")
    country_match: Optional[bool] = Field(
        None, description="Whether coordinates match expected country"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Validation warnings"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )


class DistanceRequestSchema(BaseModel):
    """Request to calculate distance between two GPS points.

    Attributes:
        lat1: First point latitude.
        lon1: First point longitude.
        lat2: Second point latitude.
        lon2: Second point longitude.
    """

    model_config = ConfigDict(from_attributes=True)

    lat1: float = Field(..., ge=-90.0, le=90.0, description="First latitude")
    lon1: float = Field(
        ..., ge=-180.0, le=180.0, description="First longitude"
    )
    lat2: float = Field(..., ge=-90.0, le=90.0, description="Second latitude")
    lon2: float = Field(
        ..., ge=-180.0, le=180.0, description="Second longitude"
    )


class DistanceResponseSchema(BaseModel):
    """Response for distance calculation.

    Attributes:
        distance_m: Distance in meters.
        distance_km: Distance in kilometers.
        bearing_degrees: Bearing in degrees from point 1 to point 2.
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(from_attributes=True)

    distance_m: float = Field(..., ge=0.0, description="Distance in meters")
    distance_km: float = Field(
        ..., ge=0.0, description="Distance in kilometers"
    )
    bearing_degrees: Optional[float] = Field(
        None, ge=0.0, le=360.0, description="Bearing degrees"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )


# =============================================================================
# Photo Schemas
# =============================================================================


class PhotoUploadSchema(BaseModel):
    """Request to record photo capture metadata.

    Attributes:
        device_id: Source device identifier.
        operator_id: Field agent identifier.
        form_id: Associated form submission identifier.
        capture_id: Associated GPS capture identifier.
        photo_type: Photo category classification.
        file_name: Original file name on device.
        file_size_bytes: Photo file size in bytes.
        file_format: Image format (jpeg, png, heic).
        width_px: Image width in pixels.
        height_px: Image height in pixels.
        integrity_hash: SHA-256 hash of raw image bytes.
        latitude: GPS latitude where photo was taken.
        longitude: GPS longitude where photo was taken.
        exif_timestamp: EXIF timestamp from photo metadata.
        device_timestamp: Device system time at capture.
        compression_quality: JPEG compression quality applied.
        annotation: Optional text annotation.
        sequence_number: Sequence number within batch capture.
        batch_group_id: Batch capture group identifier.
        metadata: Additional photo metadata.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "device_id": "dev-001-abc",
                "operator_id": "op-john-doe",
                "form_id": "form-abc-123",
                "photo_type": "plot_photo",
                "file_name": "IMG_20260309_143022.jpg",
                "file_size_bytes": 3145728,
                "file_format": "jpeg",
                "width_px": 4032,
                "height_px": 3024,
                "integrity_hash": "a1b2c3d4e5f6...",
                "latitude": 5.359952,
                "longitude": -3.974578,
            }
        },
    )

    device_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Source device identifier",
    )
    operator_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Field agent identifier",
    )
    form_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Associated form submission identifier",
    )
    capture_id: Optional[str] = Field(
        None, max_length=255,
        description="Associated GPS capture identifier",
    )
    photo_type: PhotoTypeSchema = Field(
        ..., description="Photo category classification",
    )
    file_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Original file name on device",
    )
    file_size_bytes: int = Field(
        ..., ge=1, description="Photo file size in bytes",
    )
    file_format: str = Field(
        default="jpeg", max_length=10,
        description="Image format (jpeg, png, heic)",
    )
    width_px: int = Field(..., ge=1, description="Image width in pixels")
    height_px: int = Field(..., ge=1, description="Image height in pixels")
    integrity_hash: str = Field(
        ..., min_length=1,
        description="SHA-256 hash of raw image bytes at capture",
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
    device_timestamp: Optional[datetime] = Field(
        None, description="Device system time at capture",
    )
    compression_quality: Optional[int] = Field(
        None, ge=1, le=100, description="JPEG compression quality applied",
    )
    annotation: Optional[str] = Field(
        None, max_length=2000, description="Optional text annotation",
    )
    sequence_number: Optional[int] = Field(
        None, ge=1, description="Sequence number within batch capture",
    )
    batch_group_id: Optional[str] = Field(
        None, max_length=255, description="Batch capture group identifier",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional photo metadata",
    )


class PhotoResponseSchema(BaseModel):
    """Response for photo capture operations.

    Attributes:
        photo_id: Photo evidence identifier.
        form_id: Associated form identifier.
        photo_type: Photo category.
        file_name: Original file name.
        file_size_bytes: File size in bytes.
        file_format: Image format.
        width_px: Image width.
        height_px: Image height.
        integrity_hash: SHA-256 integrity hash.
        latitude: GPS latitude.
        longitude: GPS longitude.
        annotation: Text annotation.
        provenance: Provenance tracking.
        processing_time_ms: Processing duration in milliseconds.
        message: Status message.
        created_at: Creation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    photo_id: str = Field(
        default_factory=_new_id, description="Photo evidence identifier"
    )
    form_id: str = Field(..., description="Associated form identifier")
    photo_type: str = Field(..., description="Photo category")
    file_name: str = Field(..., description="Original file name")
    file_size_bytes: int = Field(..., ge=0, description="File size bytes")
    file_format: str = Field(default="jpeg", description="Image format")
    width_px: int = Field(..., ge=1, description="Image width pixels")
    height_px: int = Field(..., ge=1, description="Image height pixels")
    integrity_hash: str = Field(..., description="SHA-256 integrity hash")
    latitude: Optional[float] = Field(None, description="GPS latitude")
    longitude: Optional[float] = Field(None, description="GPS longitude")
    annotation: Optional[str] = Field(None, description="Text annotation")
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    message: str = Field(default="", description="Status message")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )


class PhotoListSchema(BaseModel):
    """Response listing photos with pagination.

    Attributes:
        photos: List of photo response records.
        pagination: Pagination metadata.
        processing_time_ms: Query duration in milliseconds.
    """

    model_config = ConfigDict(from_attributes=True)

    photos: List[PhotoResponseSchema] = Field(
        default_factory=list, description="Photo records"
    )
    pagination: PaginationSchema = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Query duration in ms"
    )


class PhotoAnnotationSchema(BaseModel):
    """Request to add an annotation to a photo.

    Attributes:
        annotation: Text annotation to add.
        annotated_by: Operator adding the annotation.
    """

    model_config = ConfigDict(from_attributes=True)

    annotation: str = Field(
        ..., min_length=1, max_length=2000,
        description="Text annotation to add",
    )
    annotated_by: str = Field(
        ..., min_length=1, max_length=255,
        description="Operator adding the annotation",
    )


class GeotagValidationSchema(BaseModel):
    """Request to validate photo geotag proximity.

    Attributes:
        photo_latitude: Photo EXIF latitude.
        photo_longitude: Photo EXIF longitude.
        reference_latitude: Reference point latitude.
        reference_longitude: Reference point longitude.
        max_distance_m: Maximum acceptable distance in meters.
    """

    model_config = ConfigDict(from_attributes=True)

    photo_latitude: float = Field(
        ..., ge=-90.0, le=90.0, description="Photo EXIF latitude"
    )
    photo_longitude: float = Field(
        ..., ge=-180.0, le=180.0, description="Photo EXIF longitude"
    )
    reference_latitude: float = Field(
        ..., ge=-90.0, le=90.0, description="Reference point latitude"
    )
    reference_longitude: float = Field(
        ..., ge=-180.0, le=180.0, description="Reference point longitude"
    )
    max_distance_m: float = Field(
        default=100.0, ge=0.0,
        description="Maximum acceptable distance in meters",
    )


class GeotagValidationResponseSchema(BaseModel):
    """Response for geotag validation.

    Attributes:
        is_valid: Whether geotag is within acceptable distance.
        distance_m: Actual distance in meters.
        max_distance_m: Maximum acceptable distance.
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(from_attributes=True)

    is_valid: bool = Field(
        ..., description="Whether geotag is within acceptable distance"
    )
    distance_m: float = Field(..., ge=0.0, description="Actual distance meters")
    max_distance_m: float = Field(
        ..., ge=0.0, description="Maximum acceptable distance"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )


# =============================================================================
# Sync Schemas
# =============================================================================


class SyncTriggerSchema(BaseModel):
    """Request to trigger a sync session.

    Attributes:
        device_id: Device to sync.
        force: Force immediate sync bypassing interval check.
        max_items: Maximum items to sync in this session.
        item_types: Filter item types to sync.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "device_id": "dev-001-abc",
                "force": False,
                "max_items": 100,
            }
        },
    )

    device_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Device to sync",
    )
    force: bool = Field(
        default=False,
        description="Force immediate sync bypassing interval check",
    )
    max_items: Optional[int] = Field(
        None, ge=1, le=1000, description="Maximum items to sync",
    )
    item_types: Optional[List[str]] = Field(
        None, description="Filter item types (form, gps, photo, signature, package)",
    )


class SyncStatusResponseSchema(BaseModel):
    """Response for sync status query.

    Attributes:
        device_id: Device identifier.
        session_id: Current sync session identifier.
        status: Current sync status.
        pending_items: Items pending upload.
        in_progress_items: Items currently uploading.
        completed_items: Items successfully synced.
        failed_items: Items that failed to sync.
        total_bytes_pending: Total bytes awaiting upload.
        last_sync_at: Timestamp of last successful sync.
        unresolved_conflicts: Number of unresolved conflicts.
        processing_time_ms: Query duration in milliseconds.
        message: Status message.
    """

    model_config = ConfigDict(from_attributes=True)

    device_id: str = Field(..., description="Device identifier")
    session_id: Optional[str] = Field(
        None, description="Current sync session identifier"
    )
    status: str = Field(default="idle", description="Current sync status")
    pending_items: int = Field(default=0, ge=0, description="Items pending")
    in_progress_items: int = Field(
        default=0, ge=0, description="Items in progress"
    )
    completed_items: int = Field(
        default=0, ge=0, description="Items completed"
    )
    failed_items: int = Field(default=0, ge=0, description="Items failed")
    total_bytes_pending: int = Field(
        default=0, ge=0, description="Total bytes pending"
    )
    last_sync_at: Optional[datetime] = Field(
        None, description="Last successful sync"
    )
    unresolved_conflicts: int = Field(
        default=0, ge=0, description="Unresolved conflicts"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Query duration in ms"
    )
    message: str = Field(default="", description="Status message")


class SyncQueueItemSchema(BaseModel):
    """Sync queue item representation.

    Attributes:
        queue_item_id: Queue item identifier.
        device_id: Source device.
        item_type: Data type (form, gps, photo, signature, package).
        item_id: Data item identifier.
        priority: Upload priority (1=highest).
        status: Current sync status.
        retry_count: Number of retries.
        payload_size_bytes: Payload size.
        error_message: Last error message.
        created_at: Creation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    queue_item_id: str = Field(..., description="Queue item identifier")
    device_id: str = Field(..., description="Source device")
    item_type: str = Field(..., description="Data type")
    item_id: str = Field(..., description="Data item identifier")
    priority: int = Field(default=3, ge=1, le=5, description="Priority")
    status: str = Field(default="queued", description="Sync status")
    retry_count: int = Field(default=0, ge=0, description="Retries")
    payload_size_bytes: int = Field(
        default=0, ge=0, description="Payload size"
    )
    error_message: Optional[str] = Field(None, description="Last error")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Created at"
    )


class ConflictListSchema(BaseModel):
    """Response listing unresolved sync conflicts.

    Attributes:
        conflicts: List of conflict records.
        pagination: Pagination metadata.
        processing_time_ms: Query duration in milliseconds.
    """

    model_config = ConfigDict(from_attributes=True)

    conflicts: List[Dict[str, Any]] = Field(
        default_factory=list, description="Conflict records"
    )
    pagination: PaginationSchema = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Query duration in ms"
    )


class ConflictResolutionRequestSchema(BaseModel):
    """Request to resolve a sync conflict.

    Attributes:
        resolution_strategy: Chosen resolution strategy.
        resolved_value: Value to use for resolution.
        resolved_by: Operator resolving the conflict.
        reason: Resolution reason.
    """

    model_config = ConfigDict(from_attributes=True)

    resolution_strategy: ConflictResolutionSchema = Field(
        ..., description="Chosen resolution strategy",
    )
    resolved_value: Any = Field(
        None, description="Value to use for resolution",
    )
    resolved_by: str = Field(
        ..., min_length=1, max_length=255,
        description="Operator resolving the conflict",
    )
    reason: Optional[str] = Field(
        None, max_length=2000, description="Resolution reason",
    )


class ConflictResolutionResponseSchema(BaseModel):
    """Response for conflict resolution.

    Attributes:
        conflict_id: Resolved conflict identifier.
        resolved: Whether conflict was resolved.
        resolution_strategy: Strategy used.
        resolved_value: Chosen value.
        processing_time_ms: Processing duration in milliseconds.
        message: Status message.
    """

    model_config = ConfigDict(from_attributes=True)

    conflict_id: str = Field(..., description="Resolved conflict identifier")
    resolved: bool = Field(default=True, description="Resolution status")
    resolution_strategy: str = Field(..., description="Strategy used")
    resolved_value: Any = Field(None, description="Chosen value")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    message: str = Field(default="", description="Status message")


class SyncHistorySchema(BaseModel):
    """Response for sync session history.

    Attributes:
        sessions: List of sync session records.
        pagination: Pagination metadata.
        processing_time_ms: Query duration in milliseconds.
    """

    model_config = ConfigDict(from_attributes=True)

    sessions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Sync session records"
    )
    pagination: PaginationSchema = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Query duration in ms"
    )


# =============================================================================
# Template Schemas
# =============================================================================


class TemplateCreateSchema(BaseModel):
    """Request to create a form template.

    Attributes:
        name: Human-readable template name.
        form_type: EUDR form type this template implements.
        template_type: Base, custom, or inherited template.
        parent_template_id: Parent template for inheritance.
        schema_definition: JSON schema defining form structure.
        fields: List of field definitions with types and validation.
        conditional_logic: List of conditional show/hide/skip rules.
        validation_rules: List of cross-field validation rules.
        language_packs: Language-specific label translations.
        metadata: Additional template metadata.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "name": "Plot Survey Template v1",
                "form_type": "plot_survey",
                "template_type": "base",
                "fields": [
                    {
                        "name": "producer_name",
                        "type": "text",
                        "required": True,
                        "label": "Producer Name",
                    },
                    {
                        "name": "plot_area",
                        "type": "number",
                        "required": True,
                        "label": "Plot Area (ha)",
                    },
                ],
            }
        },
    )

    name: str = Field(
        ..., min_length=1, max_length=500,
        description="Human-readable template name",
    )
    form_type: FormTypeSchema = Field(
        ..., description="EUDR form type this template implements",
    )
    template_type: TemplateTypeSchema = Field(
        default=TemplateTypeSchema.BASE,
        description="Base, custom, or inherited template",
    )
    parent_template_id: Optional[str] = Field(
        None, max_length=255,
        description="Parent template for inheritance",
    )
    schema_definition: Dict[str, Any] = Field(
        default_factory=dict, description="JSON schema defining form structure",
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
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional template metadata",
    )


class TemplateUpdateSchema(BaseModel):
    """Request to update a draft form template.

    Attributes:
        name: Updated template name.
        schema_definition: Updated JSON schema.
        fields: Updated field definitions.
        conditional_logic: Updated conditional logic rules.
        validation_rules: Updated validation rules.
        language_packs: Updated language translations.
        metadata: Updated metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    name: Optional[str] = Field(
        None, min_length=1, max_length=500,
        description="Updated template name",
    )
    schema_definition: Optional[Dict[str, Any]] = Field(
        None, description="Updated JSON schema",
    )
    fields: Optional[List[Dict[str, Any]]] = Field(
        None, description="Updated field definitions",
    )
    conditional_logic: Optional[List[Dict[str, Any]]] = Field(
        None, description="Updated conditional logic rules",
    )
    validation_rules: Optional[List[Dict[str, Any]]] = Field(
        None, description="Updated validation rules",
    )
    language_packs: Optional[Dict[str, Dict[str, str]]] = Field(
        None, description="Updated language translations",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Updated metadata",
    )


class TemplateResponseSchema(BaseModel):
    """Response for template operations.

    Attributes:
        template_id: Template identifier.
        name: Template name.
        form_type: EUDR form type.
        template_type: Template type.
        status: Template lifecycle status.
        version: Semantic version.
        parent_template_id: Parent template.
        schema_definition: JSON schema.
        fields: Field definitions.
        conditional_logic: Conditional logic rules.
        validation_rules: Validation rules.
        language_packs: Language translations.
        is_active: Whether template is active.
        provenance: Provenance tracking.
        processing_time_ms: Processing duration.
        message: Status message.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    template_id: str = Field(
        default_factory=_new_id, description="Template identifier"
    )
    name: str = Field(..., description="Template name")
    form_type: str = Field(..., description="EUDR form type")
    template_type: str = Field(default="base", description="Template type")
    status: str = Field(default="draft", description="Lifecycle status")
    version: str = Field(default="1.0.0", description="Semantic version")
    parent_template_id: Optional[str] = Field(
        None, description="Parent template"
    )
    schema_definition: Dict[str, Any] = Field(
        default_factory=dict, description="JSON schema"
    )
    fields: List[Dict[str, Any]] = Field(
        default_factory=list, description="Field definitions"
    )
    conditional_logic: List[Dict[str, Any]] = Field(
        default_factory=list, description="Conditional logic rules"
    )
    validation_rules: List[Dict[str, Any]] = Field(
        default_factory=list, description="Validation rules"
    )
    language_packs: Dict[str, Dict[str, str]] = Field(
        default_factory=dict, description="Language translations"
    )
    is_active: bool = Field(default=True, description="Is active")
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    message: str = Field(default="", description="Status message")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Last update timestamp"
    )


class TemplateListSchema(BaseModel):
    """Response listing templates with pagination.

    Attributes:
        templates: List of template records.
        pagination: Pagination metadata.
        processing_time_ms: Query duration in milliseconds.
    """

    model_config = ConfigDict(from_attributes=True)

    templates: List[TemplateResponseSchema] = Field(
        default_factory=list, description="Template records"
    )
    pagination: PaginationSchema = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Query duration in ms"
    )


class TemplateRenderSchema(BaseModel):
    """Request to render a template for a specific language.

    Attributes:
        language_code: ISO 639-1 language code.
        include_schema: Whether to include JSON schema in response.
    """

    model_config = ConfigDict(from_attributes=True)

    language_code: str = Field(
        ..., min_length=2, max_length=10,
        description="ISO 639-1 language code (e.g., en, fr, sw)",
    )
    include_schema: bool = Field(
        default=True, description="Include JSON schema in response",
    )


class TemplateRenderResponseSchema(BaseModel):
    """Response for template rendering.

    Attributes:
        template_id: Template identifier.
        language_code: Rendered language.
        rendered_fields: Fields with localized labels.
        schema: JSON schema if requested.
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(from_attributes=True)

    template_id: str = Field(..., description="Template identifier")
    language_code: str = Field(..., description="Rendered language")
    rendered_fields: List[Dict[str, Any]] = Field(
        default_factory=list, description="Fields with localized labels"
    )
    schema: Optional[Dict[str, Any]] = Field(
        None, description="JSON schema if requested"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )


# =============================================================================
# Signature Schemas
# =============================================================================


class SignatureCreateSchema(BaseModel):
    """Request to create a digital signature.

    Attributes:
        form_id: Associated form submission identifier.
        signer_name: Name of the signatory.
        signer_role: Role of the signatory.
        signer_device_id: Device used for signing.
        algorithm: Signature algorithm.
        public_key_fingerprint: Fingerprint of signer public key.
        signature_bytes_hex: DER-encoded signature in hex.
        signed_data_hash: SHA-256 hash of the signed data.
        timestamp_binding: ISO 8601 timestamp included in signed payload.
        visual_signature_svg: SVG touch-path of handwritten signature.
        metadata: Additional signature metadata.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "form_id": "form-abc-123",
                "signer_name": "Jean Mbeki",
                "signer_role": "producer",
                "signer_device_id": "dev-001-abc",
                "algorithm": "ecdsa_p256",
            }
        },
    )

    form_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Associated form submission identifier",
    )
    signer_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Name of the signatory",
    )
    signer_role: str = Field(
        ..., min_length=1, max_length=100,
        description="Role (producer, collector, inspector, transport_operator, buyer)",
    )
    signer_device_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Device used for signing",
    )
    algorithm: SignatureAlgorithmSchema = Field(
        default=SignatureAlgorithmSchema.ECDSA_P256,
        description="Signature algorithm",
    )
    public_key_fingerprint: Optional[str] = Field(
        None, max_length=255,
        description="Fingerprint of signer public key",
    )
    signature_bytes_hex: Optional[str] = Field(
        None, description="DER-encoded signature in hex",
    )
    signed_data_hash: Optional[str] = Field(
        None, description="SHA-256 hash of the signed data",
    )
    timestamp_binding: Optional[str] = Field(
        None, max_length=50,
        description="ISO 8601 timestamp included in signed payload",
    )
    visual_signature_svg: Optional[str] = Field(
        None, description="SVG touch-path of handwritten signature",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional signature metadata",
    )


class SignatureResponseSchema(BaseModel):
    """Response for signature operations.

    Attributes:
        signature_id: Signature identifier.
        form_id: Associated form.
        signer_name: Signatory name.
        signer_role: Signatory role.
        algorithm: Algorithm used.
        is_valid: Whether signature passes verification.
        is_revoked: Whether signature has been revoked.
        timestamp_binding: Timestamp binding.
        provenance: Provenance tracking.
        processing_time_ms: Processing duration.
        message: Status message.
        created_at: Creation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    signature_id: str = Field(
        default_factory=_new_id, description="Signature identifier"
    )
    form_id: str = Field(..., description="Associated form")
    signer_name: str = Field(..., description="Signatory name")
    signer_role: str = Field(..., description="Signatory role")
    algorithm: str = Field(default="ecdsa_p256", description="Algorithm")
    is_valid: bool = Field(default=False, description="Verification status")
    is_revoked: bool = Field(default=False, description="Revocation status")
    timestamp_binding: Optional[str] = Field(
        None, description="Timestamp binding"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    message: str = Field(default="", description="Status message")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )


class SignatureVerifySchema(BaseModel):
    """Response for signature verification.

    Attributes:
        signature_id: Signature identifier.
        is_valid: Whether signature passes cryptographic verification.
        algorithm: Algorithm used.
        signed_data_hash: Hash of signed data.
        is_revoked: Whether signature is revoked.
        is_expired: Whether signature has expired.
        processing_time_ms: Processing duration.
    """

    model_config = ConfigDict(from_attributes=True)

    signature_id: str = Field(..., description="Signature identifier")
    is_valid: bool = Field(..., description="Cryptographic verification result")
    algorithm: str = Field(default="ecdsa_p256", description="Algorithm used")
    signed_data_hash: Optional[str] = Field(
        None, description="Hash of signed data"
    )
    is_revoked: bool = Field(default=False, description="Revocation status")
    is_expired: bool = Field(default=False, description="Expiration status")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )


class CustodyTransferSchema(BaseModel):
    """Request to create a custody transfer signature.

    Attributes:
        form_id: Custody transfer form identifier.
        from_signer_name: Name of the transferring party.
        from_signer_role: Role of the transferring party.
        to_signer_name: Name of the receiving party.
        to_signer_role: Role of the receiving party.
        device_id: Device used for the transfer.
        commodity_type: EUDR commodity being transferred.
        quantity: Quantity being transferred.
        unit: Unit of measurement.
        metadata: Additional transfer metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    form_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Custody transfer form identifier",
    )
    from_signer_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Transferring party name",
    )
    from_signer_role: str = Field(
        ..., min_length=1, max_length=100,
        description="Transferring party role",
    )
    to_signer_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Receiving party name",
    )
    to_signer_role: str = Field(
        ..., min_length=1, max_length=100,
        description="Receiving party role",
    )
    device_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Device used for the transfer",
    )
    commodity_type: Optional[CommodityTypeSchema] = Field(
        None, description="EUDR commodity being transferred",
    )
    quantity: Optional[float] = Field(
        None, ge=0.0, description="Quantity being transferred",
    )
    unit: Optional[str] = Field(
        None, max_length=50, description="Unit of measurement (kg, tonnes)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional transfer metadata",
    )


class MultiSigSchema(BaseModel):
    """Request to create a multi-signature.

    Attributes:
        form_id: Form requiring multiple signatures.
        required_signers: List of required signer identifiers.
        threshold: Minimum number of signatures required.
        deadline: Deadline for collecting all signatures.
        metadata: Additional multi-sig metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    form_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Form requiring multiple signatures",
    )
    required_signers: List[Dict[str, str]] = Field(
        ..., min_length=2,
        description="List of required signers with name and role",
    )
    threshold: int = Field(
        ..., ge=2, description="Minimum signatures required",
    )
    deadline: Optional[datetime] = Field(
        None, description="Deadline for collecting all signatures",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional multi-sig metadata",
    )


class SignatureListSchema(BaseModel):
    """Response listing signatures with pagination.

    Attributes:
        signatures: List of signature records.
        pagination: Pagination metadata.
        processing_time_ms: Query duration.
    """

    model_config = ConfigDict(from_attributes=True)

    signatures: List[SignatureResponseSchema] = Field(
        default_factory=list, description="Signature records"
    )
    pagination: PaginationSchema = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Query duration in ms"
    )


# =============================================================================
# Package Schemas
# =============================================================================


class PackageCreateSchema(BaseModel):
    """Request to create a new data package.

    Attributes:
        device_id: Source device identifier.
        operator_id: Operator building the package.
        compression_format: Compression algorithm.
        compression_level: Compression level (1-9).
        export_format: Export format (zip, tar_gz, json_ld).
        metadata: Additional package metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    device_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Source device identifier",
    )
    operator_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Operator building the package",
    )
    compression_format: str = Field(
        default="gzip", description="Compression algorithm (gzip, zstd, lz4)",
    )
    compression_level: int = Field(
        default=6, ge=1, le=9, description="Compression level (1-9)",
    )
    export_format: ExportFormatSchema = Field(
        default=ExportFormatSchema.ZIP,
        description="Export format",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional package metadata",
    )


class PackageAddItemSchema(BaseModel):
    """Request to add an item to a package.

    Attributes:
        item_id: Identifier of the item to add.
        item_type: Type of item (form, gps, photo, signature).
    """

    model_config = ConfigDict(from_attributes=True)

    item_id: str = Field(
        ..., min_length=1, max_length=255,
        description="Identifier of the item to add",
    )
    item_type: Optional[str] = Field(
        None, description="Type of item for explicit specification",
    )


class PackageSealSchema(BaseModel):
    """Request to seal a data package.

    Attributes:
        compute_merkle: Whether to compute Merkle root.
        sign_package: Whether to sign the manifest.
    """

    model_config = ConfigDict(from_attributes=True)

    compute_merkle: bool = Field(
        default=True, description="Compute SHA-256 Merkle root",
    )
    sign_package: bool = Field(
        default=True, description="Sign the manifest with ECDSA",
    )


class PackageResponseSchema(BaseModel):
    """Response for data package operations.

    Attributes:
        package_id: Package identifier.
        status: Package lifecycle status.
        device_id: Source device.
        operator_id: Operator.
        form_ids: Included form identifiers.
        gps_capture_ids: Included GPS captures.
        photo_ids: Included photos.
        signature_ids: Included signatures.
        artifact_count: Total artifacts.
        package_size_bytes: Total size.
        merkle_root: SHA-256 Merkle root.
        compression_format: Compression used.
        export_format: Export format.
        sealed_at: Seal timestamp.
        provenance: Provenance tracking.
        processing_time_ms: Processing duration.
        message: Status message.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    package_id: str = Field(
        default_factory=_new_id, description="Package identifier"
    )
    status: str = Field(default="building", description="Lifecycle status")
    device_id: str = Field(..., description="Source device")
    operator_id: str = Field(..., description="Operator")
    form_ids: List[str] = Field(
        default_factory=list, description="Included forms"
    )
    gps_capture_ids: List[str] = Field(
        default_factory=list, description="Included GPS captures"
    )
    photo_ids: List[str] = Field(
        default_factory=list, description="Included photos"
    )
    signature_ids: List[str] = Field(
        default_factory=list, description="Included signatures"
    )
    artifact_count: int = Field(default=0, ge=0, description="Total artifacts")
    package_size_bytes: int = Field(
        default=0, ge=0, description="Total size bytes"
    )
    merkle_root: Optional[str] = Field(
        None, description="SHA-256 Merkle root"
    )
    compression_format: str = Field(
        default="gzip", description="Compression used"
    )
    export_format: str = Field(default="zip", description="Export format")
    sealed_at: Optional[datetime] = Field(None, description="Seal timestamp")
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    message: str = Field(default="", description="Status message")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Last update timestamp"
    )


class PackageExportSchema(BaseModel):
    """Response for package export/download.

    Attributes:
        package_id: Package identifier.
        download_url: Pre-signed download URL.
        expires_in_seconds: URL expiry duration.
        file_size_bytes: Export file size.
        file_hash: SHA-256 hash of export file.
        export_format: Export format used.
        content_type: MIME content type.
    """

    model_config = ConfigDict(from_attributes=True)

    package_id: str = Field(..., description="Package identifier")
    download_url: str = Field(..., description="Pre-signed download URL")
    expires_in_seconds: int = Field(
        default=3600, ge=60, description="URL expiry seconds"
    )
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="Export file size"
    )
    file_hash: Optional[str] = Field(
        None, description="SHA-256 hash of export file"
    )
    export_format: str = Field(default="zip", description="Export format")
    content_type: str = Field(
        default="application/zip", description="MIME content type"
    )


class ManifestSchema(BaseModel):
    """Response for package manifest.

    Attributes:
        package_id: Package identifier.
        artifacts: List of artifact records in the manifest.
        merkle_root: SHA-256 Merkle root hash.
        total_artifacts: Total artifacts in manifest.
        total_size_bytes: Total size of all artifacts.
        sealed_at: When the manifest was sealed.
        processing_time_ms: Processing duration.
    """

    model_config = ConfigDict(from_attributes=True)

    package_id: str = Field(..., description="Package identifier")
    artifacts: List[Dict[str, Any]] = Field(
        default_factory=list, description="Artifact records"
    )
    merkle_root: Optional[str] = Field(
        None, description="SHA-256 Merkle root"
    )
    total_artifacts: int = Field(
        default=0, ge=0, description="Total artifacts"
    )
    total_size_bytes: int = Field(
        default=0, ge=0, description="Total size bytes"
    )
    sealed_at: Optional[datetime] = Field(
        None, description="Manifest seal timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )


class PackageListSchema(BaseModel):
    """Response listing packages with pagination.

    Attributes:
        packages: List of package records.
        pagination: Pagination metadata.
        processing_time_ms: Query duration.
    """

    model_config = ConfigDict(from_attributes=True)

    packages: List[PackageResponseSchema] = Field(
        default_factory=list, description="Package records"
    )
    pagination: PaginationSchema = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Query duration in ms"
    )


class PackageValidateSchema(BaseModel):
    """Response for sealed package validation.

    Attributes:
        package_id: Package identifier.
        is_valid: Whether package passes integrity checks.
        merkle_valid: Whether Merkle root matches.
        signature_valid: Whether package signature is valid.
        artifact_count: Total verified artifacts.
        errors: Validation errors.
        processing_time_ms: Processing duration.
    """

    model_config = ConfigDict(from_attributes=True)

    package_id: str = Field(..., description="Package identifier")
    is_valid: bool = Field(..., description="Integrity check result")
    merkle_valid: bool = Field(..., description="Merkle root matches")
    signature_valid: Optional[bool] = Field(
        None, description="Signature validity"
    )
    artifact_count: int = Field(
        default=0, ge=0, description="Verified artifacts"
    )
    errors: List[str] = Field(
        default_factory=list, description="Validation errors"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )


# =============================================================================
# Device Schemas
# =============================================================================


class DeviceRegisterSchema(BaseModel):
    """Request to register a new device in the fleet.

    Attributes:
        device_model: Device hardware model name.
        platform: Device operating system platform.
        os_version: Operating system version string.
        agent_version: Mobile Data Collector agent version.
        assigned_operator_id: Assigned field agent.
        assigned_area: GeoJSON polygon of assigned collection area.
        metadata: Additional device metadata.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "device_model": "Samsung Galaxy A54",
                "platform": "android",
                "os_version": "14.0",
                "agent_version": "1.0.0",
                "assigned_operator_id": "op-john-doe",
            }
        },
    )

    device_model: str = Field(
        ..., min_length=1, max_length=255,
        description="Device hardware model name",
    )
    platform: DevicePlatformSchema = Field(
        default=DevicePlatformSchema.ANDROID,
        description="Device operating system platform",
    )
    os_version: str = Field(
        ..., min_length=1, max_length=50,
        description="Operating system version string",
    )
    agent_version: str = Field(
        default="1.0.0", max_length=20,
        description="Mobile Data Collector agent version",
    )
    assigned_operator_id: Optional[str] = Field(
        None, max_length=255,
        description="Assigned field agent",
    )
    assigned_area: Optional[Dict[str, Any]] = Field(
        None, description="GeoJSON polygon of assigned collection area",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional device metadata",
    )


class DeviceUpdateSchema(BaseModel):
    """Request to update a device registration.

    Attributes:
        assigned_operator_id: New assigned operator.
        assigned_area: Updated collection area GeoJSON.
        agent_version: Updated agent version.
        metadata: Updated metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    assigned_operator_id: Optional[str] = Field(
        None, max_length=255, description="New assigned operator",
    )
    assigned_area: Optional[Dict[str, Any]] = Field(
        None, description="Updated collection area GeoJSON",
    )
    agent_version: Optional[str] = Field(
        None, max_length=20, description="Updated agent version",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Updated metadata",
    )


class DeviceResponseSchema(BaseModel):
    """Response for device operations.

    Attributes:
        device_id: Device identifier.
        device_model: Device model.
        platform: OS platform.
        os_version: OS version.
        agent_version: Agent version.
        status: Current device status.
        assigned_operator_id: Assigned operator.
        battery_level_pct: Battery level.
        storage_free_bytes: Free storage.
        last_sync_at: Last sync timestamp.
        pending_forms: Pending forms count.
        pending_photos: Pending photos count.
        pending_gps: Pending GPS count.
        connectivity_type: Connectivity type.
        is_decommissioned: Decommission status.
        provenance: Provenance tracking.
        processing_time_ms: Processing duration.
        message: Status message.
        registered_at: Registration timestamp.
        created_at: Creation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    device_id: str = Field(
        default_factory=_new_id, description="Device identifier"
    )
    device_model: str = Field(..., description="Device model")
    platform: str = Field(default="android", description="OS platform")
    os_version: str = Field(..., description="OS version")
    agent_version: str = Field(default="1.0.0", description="Agent version")
    status: str = Field(default="active", description="Device status")
    assigned_operator_id: Optional[str] = Field(
        None, description="Assigned operator"
    )
    battery_level_pct: Optional[int] = Field(
        None, ge=0, le=100, description="Battery level"
    )
    storage_free_bytes: Optional[int] = Field(
        None, ge=0, description="Free storage"
    )
    last_sync_at: Optional[datetime] = Field(
        None, description="Last sync"
    )
    pending_forms: int = Field(default=0, ge=0, description="Pending forms")
    pending_photos: int = Field(default=0, ge=0, description="Pending photos")
    pending_gps: int = Field(default=0, ge=0, description="Pending GPS")
    connectivity_type: Optional[str] = Field(
        None, description="Connectivity type"
    )
    is_decommissioned: bool = Field(
        default=False, description="Decommission status"
    )
    provenance: Optional[ProvenanceInfo] = Field(
        None, description="Provenance tracking"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    message: str = Field(default="", description="Status message")
    registered_at: datetime = Field(
        default_factory=_utcnow, description="Registration timestamp"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )


class FleetStatusSchema(BaseModel):
    """Response for fleet dashboard status.

    Attributes:
        total_devices: Total registered devices.
        active_devices: Devices synced recently.
        offline_devices: Devices not synced within threshold.
        low_battery_devices: Devices with low battery.
        low_storage_devices: Devices with low storage.
        decommissioned_devices: Decommissioned devices.
        outdated_agent_devices: Devices with outdated agent version.
        total_pending_forms: Total pending forms across fleet.
        total_pending_photos: Total pending photos across fleet.
        total_pending_sync_bytes: Total pending sync bytes.
        processing_time_ms: Query duration.
        message: Status message.
    """

    model_config = ConfigDict(from_attributes=True)

    total_devices: int = Field(
        default=0, ge=0, description="Total registered devices"
    )
    active_devices: int = Field(
        default=0, ge=0, description="Active devices"
    )
    offline_devices: int = Field(
        default=0, ge=0, description="Offline devices"
    )
    low_battery_devices: int = Field(
        default=0, ge=0, description="Low battery devices"
    )
    low_storage_devices: int = Field(
        default=0, ge=0, description="Low storage devices"
    )
    decommissioned_devices: int = Field(
        default=0, ge=0, description="Decommissioned devices"
    )
    outdated_agent_devices: int = Field(
        default=0, ge=0, description="Outdated agent devices"
    )
    total_pending_forms: int = Field(
        default=0, ge=0, description="Total pending forms"
    )
    total_pending_photos: int = Field(
        default=0, ge=0, description="Total pending photos"
    )
    total_pending_sync_bytes: int = Field(
        default=0, ge=0, description="Total pending sync bytes"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Query duration in ms"
    )
    message: str = Field(default="", description="Status message")


class HeartbeatSchema(BaseModel):
    """Request to record a device heartbeat.

    Attributes:
        battery_level_pct: Current battery percentage.
        storage_total_bytes: Total storage.
        storage_used_bytes: Used storage.
        storage_free_bytes: Free storage.
        gps_hdop: Current GPS HDOP.
        gps_satellites: Current satellite count.
        gps_latitude: Current latitude.
        gps_longitude: Current longitude.
        pending_forms: Forms awaiting sync.
        pending_photos: Photos awaiting sync.
        pending_gps: GPS captures awaiting sync.
        agent_version: Agent version string.
        os_version: OS version string.
        connectivity_type: Connectivity type.
        event_timestamp: Device clock timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    battery_level_pct: Optional[int] = Field(
        None, ge=0, le=100, description="Battery percentage"
    )
    storage_total_bytes: Optional[int] = Field(
        None, ge=0, description="Total storage"
    )
    storage_used_bytes: Optional[int] = Field(
        None, ge=0, description="Used storage"
    )
    storage_free_bytes: Optional[int] = Field(
        None, ge=0, description="Free storage"
    )
    gps_hdop: Optional[float] = Field(
        None, ge=0.0, description="Current GPS HDOP"
    )
    gps_satellites: Optional[int] = Field(
        None, ge=0, description="Current satellite count"
    )
    gps_latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="Current latitude"
    )
    gps_longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0, description="Current longitude"
    )
    pending_forms: Optional[int] = Field(
        None, ge=0, description="Pending forms"
    )
    pending_photos: Optional[int] = Field(
        None, ge=0, description="Pending photos"
    )
    pending_gps: Optional[int] = Field(
        None, ge=0, description="Pending GPS captures"
    )
    agent_version: Optional[str] = Field(
        None, max_length=20, description="Agent version"
    )
    os_version: Optional[str] = Field(
        None, max_length=50, description="OS version"
    )
    connectivity_type: Optional[str] = Field(
        None, max_length=20,
        description="Connectivity (none/2g/3g/4g/5g/wifi)",
    )
    event_timestamp: Optional[datetime] = Field(
        None, description="Device clock timestamp"
    )


class TelemetrySchema(BaseModel):
    """Request to update device telemetry.

    Attributes:
        event_type: Telemetry event type.
        battery_level_pct: Battery percentage.
        storage_free_bytes: Free storage.
        gps_hdop: GPS HDOP.
        gps_satellites: Satellite count.
        gps_latitude: Latitude.
        gps_longitude: Longitude.
        connectivity_type: Connectivity type.
        error_message: Error details if applicable.
        event_timestamp: Device clock timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    event_type: str = Field(
        ..., min_length=1, max_length=50,
        description="Telemetry event type (heartbeat, sync_start, sync_complete, "
                    "sync_error, low_battery, low_storage, gps_fix_lost)",
    )
    battery_level_pct: Optional[int] = Field(
        None, ge=0, le=100, description="Battery percentage"
    )
    storage_free_bytes: Optional[int] = Field(
        None, ge=0, description="Free storage"
    )
    gps_hdop: Optional[float] = Field(
        None, ge=0.0, description="GPS HDOP"
    )
    gps_satellites: Optional[int] = Field(
        None, ge=0, description="Satellite count"
    )
    gps_latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="Latitude"
    )
    gps_longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0, description="Longitude"
    )
    connectivity_type: Optional[str] = Field(
        None, max_length=20, description="Connectivity type"
    )
    error_message: Optional[str] = Field(
        None, max_length=2000, description="Error details"
    )
    event_timestamp: Optional[datetime] = Field(
        None, description="Device clock timestamp"
    )


class CampaignSchema(BaseModel):
    """Request to create a collection campaign.

    Attributes:
        name: Campaign name.
        description: Campaign description.
        commodity_type: Target EUDR commodity.
        country_code: Target country (ISO 3166-1 alpha-2).
        start_date: Campaign start date.
        end_date: Campaign end date.
        target_forms: Target number of form submissions.
        target_area_ha: Target collection area in hectares.
        assigned_device_ids: Devices assigned to this campaign.
        geographic_bounds: GeoJSON polygon of campaign area.
        metadata: Additional campaign metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(
        ..., min_length=1, max_length=500, description="Campaign name"
    )
    description: Optional[str] = Field(
        None, max_length=2000, description="Campaign description"
    )
    commodity_type: Optional[CommodityTypeSchema] = Field(
        None, description="Target EUDR commodity"
    )
    country_code: Optional[str] = Field(
        None, min_length=2, max_length=2,
        description="Target country (ISO 3166-1 alpha-2)",
    )
    start_date: Optional[datetime] = Field(
        None, description="Campaign start date"
    )
    end_date: Optional[datetime] = Field(
        None, description="Campaign end date"
    )
    target_forms: Optional[int] = Field(
        None, ge=1, description="Target form submissions"
    )
    target_area_ha: Optional[float] = Field(
        None, ge=0.0, description="Target area hectares"
    )
    assigned_device_ids: List[str] = Field(
        default_factory=list, description="Assigned device IDs"
    )
    geographic_bounds: Optional[Dict[str, Any]] = Field(
        None, description="GeoJSON campaign area"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional campaign metadata"
    )


class CampaignResponseSchema(BaseModel):
    """Response for campaign operations.

    Attributes:
        campaign_id: Campaign identifier.
        name: Campaign name.
        status: Campaign status.
        commodity_type: Target commodity.
        country_code: Target country.
        start_date: Start date.
        end_date: End date.
        target_forms: Target submissions.
        completed_forms: Completed submissions.
        assigned_devices: Number of assigned devices.
        progress_percent: Completion percentage.
        processing_time_ms: Processing duration.
        message: Status message.
        created_at: Creation timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    campaign_id: str = Field(
        default_factory=_new_id, description="Campaign identifier"
    )
    name: str = Field(..., description="Campaign name")
    status: str = Field(default="active", description="Campaign status")
    commodity_type: Optional[str] = Field(
        None, description="Target commodity"
    )
    country_code: Optional[str] = Field(None, description="Target country")
    start_date: Optional[datetime] = Field(None, description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date")
    target_forms: Optional[int] = Field(
        None, ge=0, description="Target submissions"
    )
    completed_forms: int = Field(
        default=0, ge=0, description="Completed submissions"
    )
    assigned_devices: int = Field(
        default=0, ge=0, description="Assigned devices"
    )
    progress_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Completion percentage"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    message: str = Field(default="", description="Status message")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )


class DeviceListSchema(BaseModel):
    """Response listing devices with pagination.

    Attributes:
        devices: List of device records.
        pagination: Pagination metadata.
        processing_time_ms: Query duration.
    """

    model_config = ConfigDict(from_attributes=True)

    devices: List[DeviceResponseSchema] = Field(
        default_factory=list, description="Device records"
    )
    pagination: PaginationSchema = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Query duration in ms"
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enumerations
    "AccuracyTierSchema",
    "CommodityTypeSchema",
    "ConflictResolutionSchema",
    "DevicePlatformSchema",
    "DeviceStatusSchema",
    "ExportFormatSchema",
    "FormStatusSchema",
    "FormTypeSchema",
    "PackageStatusSchema",
    "PhotoTypeSchema",
    "SignatureAlgorithmSchema",
    "SyncStatusSchema",
    "TemplateStatusSchema",
    "TemplateTypeSchema",
    # Common
    "ErrorSchema",
    "HealthComponentSchema",
    "HealthSchema",
    "PaginationSchema",
    "ProvenanceInfo",
    "SuccessSchema",
    # Form schemas
    "FormListSchema",
    "FormResponseSchema",
    "FormSubmitSchema",
    "FormUpdateSchema",
    "FormValidationSchema",
    # GPS schemas
    "AreaResponseSchema",
    "DistanceRequestSchema",
    "DistanceResponseSchema",
    "GPSCaptureSchema",
    "GPSResponseSchema",
    "GPSValidateResponseSchema",
    "GPSValidateSchema",
    "PolygonCaptureSchema",
    "PolygonResponseSchema",
    # Photo schemas
    "GeotagValidationResponseSchema",
    "GeotagValidationSchema",
    "PhotoAnnotationSchema",
    "PhotoListSchema",
    "PhotoResponseSchema",
    "PhotoUploadSchema",
    # Sync schemas
    "ConflictListSchema",
    "ConflictResolutionRequestSchema",
    "ConflictResolutionResponseSchema",
    "SyncHistorySchema",
    "SyncQueueItemSchema",
    "SyncStatusResponseSchema",
    "SyncTriggerSchema",
    # Template schemas
    "TemplateCreateSchema",
    "TemplateListSchema",
    "TemplateRenderResponseSchema",
    "TemplateRenderSchema",
    "TemplateResponseSchema",
    "TemplateUpdateSchema",
    # Signature schemas
    "CustodyTransferSchema",
    "MultiSigSchema",
    "SignatureCreateSchema",
    "SignatureListSchema",
    "SignatureResponseSchema",
    "SignatureVerifySchema",
    # Package schemas
    "ManifestSchema",
    "PackageAddItemSchema",
    "PackageCreateSchema",
    "PackageExportSchema",
    "PackageListSchema",
    "PackageResponseSchema",
    "PackageSealSchema",
    "PackageValidateSchema",
    # Device schemas
    "CampaignResponseSchema",
    "CampaignSchema",
    "DeviceListSchema",
    "DeviceRegisterSchema",
    "DeviceResponseSchema",
    "DeviceUpdateSchema",
    "FleetStatusSchema",
    "HeartbeatSchema",
    "TelemetrySchema",
]
