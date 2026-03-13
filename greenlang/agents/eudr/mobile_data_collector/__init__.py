# -*- coding: utf-8 -*-
"""
Mobile Data Collector Agent - AGENT-EUDR-015

Production-grade offline-first mobile data collection platform for
EUDR compliance covering structured form collection with local SQLite
storage and Pydantic schema validation; high-accuracy GPS point and
polygon capture with HDOP, satellite count, and fix type metadata;
geotagged photo evidence with EXIF extraction and SHA-256 integrity
hashing; CRDT-based offline data synchronization with conflict
resolution (last-writer-wins, set union, manual); dynamic form
templates with conditional logic and multi-language support (24 EU
+ 20 local languages); ECDSA P-256 digital signature capture with
timestamp binding and RFC 6979 deterministic k-value; self-contained
data package assembly with SHA-256 Merkle root integrity; and device
fleet management with sync status, telemetry monitoring, and
collection campaign tracking.

This package provides a complete mobile data collection system for
EUDR field-level compliance data capture per EU 2023/1115 Articles
4, 9, 10, 14, 16, and 22:

    Capabilities:
        - Offline-first form engine with 6 EUDR form types (producer
          registration, plot survey, harvest log, custody transfer,
          quality inspection, smallholder declaration), local SQLite
          storage, and queue-based synchronization
        - GPS/geolocation capture with configurable accuracy thresholds
          (default <3m CEP), HDOP validation, satellite count minimums,
          polygon boundary tracing, area calculation (Shoelace with
          geodesic correction), and WGS84 coordinate validation
        - Photo evidence collection with EXIF metadata extraction,
          geotagging, SHA-256 integrity hashing at capture time, JPEG
          compression at 3 quality levels, and batch capture sequencing
        - CRDT-based offline sync with LWW scalar merge, grow-only set
          union, state machine status merge, exponential backoff retries,
          delta compression, idempotency keys, and upload prioritization
        - Dynamic form templates with JSON schema definitions, 14+ field
          types, conditional show/hide/skip logic, cross-field validation,
          multi-language rendering, versioning, and template inheritance
        - ECDSA P-256 digital signatures with timestamp binding, form
          binding, visual SVG touch-path capture, multi-signature
          workflows, offline verification, and revocation support
        - Data package builder with SHA-256 Merkle root integrity,
          artifact manifest, device signing, gzip compression,
          incremental building, and 3 export formats (ZIP/tar.gz/JSON-LD)
        - Device fleet manager tracking sync status, battery, storage,
          GPS quality, agent version, operator assignment, collection
          areas, and fleet dashboard aggregation

    Foundational modules:
        - config: MobileDataCollectorConfig with GL_EUDR_MDC_
          env var support (60+ settings)
        - models: Pydantic v2 data models with 15 enumerations,
          12 core models, 15 request models, and 15 response models
        - provenance: SHA-256 chain-hashed audit trail tracking with
          12 entity types and 12 actions
        - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_mdc_)

PRD: PRD-AGENT-EUDR-015
Agent ID: GL-EUDR-MDC-015
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 14, 16, 22
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.mobile_data_collector import (
    ...     FormSubmission,
    ...     FormType,
    ...     FormStatus,
    ...     CommodityType,
    ...     MobileDataCollectorConfig,
    ...     get_config,
    ... )
    >>> form = FormSubmission(
    ...     device_id="device-001",
    ...     operator_id="agent-001",
    ...     form_type=FormType.HARVEST_LOG,
    ...     template_id="tmpl-001",
    ...     data={"commodity_type": "coffee", "quantity_kg": 500},
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-MDC-015"

# ---- Foundational: config ----
try:
    from greenlang.agents.eudr.mobile_data_collector.config import (
        MobileDataCollectorConfig,
        get_config,
        set_config,
        reset_config,
    )
except ImportError:
    MobileDataCollectorConfig = None  # type: ignore[assignment,misc]
    get_config = None  # type: ignore[assignment]
    set_config = None  # type: ignore[assignment]
    reset_config = None  # type: ignore[assignment]

# ---- Foundational: models ----
try:
    from greenlang.agents.eudr.mobile_data_collector.models import (
        # Constants
        VERSION,
        EUDR_DEFORESTATION_CUTOFF,
        MAX_BATCH_SIZE,
        EUDR_RETENTION_YEARS,
        SUPPORTED_PHOTO_FORMATS,
        SUPPORTED_LANGUAGES,
        WGS84_SRID,
        DEFAULT_EUDR_COMMODITIES,
        SUPPORTED_FORM_TYPES,
        SUPPORTED_FIELD_TYPES,
        # Enumerations
        FormStatus,
        FormType,
        CaptureAccuracyTier,
        PhotoType,
        SyncStatus,
        ConflictResolution,
        TemplateType,
        SignatureAlgorithm,
        PackageStatus,
        DeviceStatus,
        DevicePlatform,
        CommodityType,
        ComplianceStatus,
        FieldType,
        LanguageCode,
        # Core Models
        FormSubmission,
        GPSCapture,
        PolygonTrace,
        PhotoEvidence,
        SyncQueueItem,
        SyncConflict,
        FormTemplate,
        DigitalSignature,
        DataPackage,
        DeviceRegistration,
        DeviceEvent,
        AuditLogEntry,
        # Request Models
        SubmitFormRequest,
        CaptureGPSRequest,
        CapturePolygonRequest,
        UploadPhotoRequest,
        TriggerSyncRequest,
        ResolveConflictRequest,
        CreateTemplateRequest,
        UpdateTemplateRequest,
        CaptureSignatureRequest,
        BuildPackageRequest,
        RegisterDeviceRequest,
        UpdateDeviceRequest,
        SearchFormsRequest,
        GetDeviceStatusRequest,
        ValidateFormRequest,
        # Response Models
        FormResponse,
        GPSResponse,
        PolygonResponse,
        PhotoResponse,
        SyncResponse,
        ConflictResponse,
        TemplateResponse,
        SignatureResponse,
        PackageResponse,
        DeviceResponse,
        DeviceStatusResponse,
        SearchResponse,
        SyncStatusResponse,
        FleetStatusResponse,
        HealthResponse,
    )
except ImportError:
    pass

# ---- Foundational: provenance ----
try:
    from greenlang.agents.eudr.mobile_data_collector.provenance import (
        ProvenanceRecord,
        ProvenanceTracker,
        VALID_ENTITY_TYPES,
        VALID_ACTIONS,
        get_provenance_tracker,
        set_provenance_tracker,
        reset_provenance_tracker,
    )
except ImportError:
    ProvenanceRecord = None  # type: ignore[assignment,misc]
    ProvenanceTracker = None  # type: ignore[assignment,misc]
    VALID_ENTITY_TYPES = frozenset()  # type: ignore[assignment]
    VALID_ACTIONS = frozenset()  # type: ignore[assignment]
    get_provenance_tracker = None  # type: ignore[assignment]
    set_provenance_tracker = None  # type: ignore[assignment]
    reset_provenance_tracker = None  # type: ignore[assignment]

# ---- Foundational: metrics ----
try:
    from greenlang.agents.eudr.mobile_data_collector.metrics import (
        PROMETHEUS_AVAILABLE,
        # Metric objects
        mdc_forms_submitted_total,
        mdc_gps_captures_total,
        mdc_photos_captured_total,
        mdc_syncs_completed_total,
        mdc_sync_conflicts_total,
        mdc_signatures_captured_total,
        mdc_packages_built_total,
        mdc_api_errors_total,
        mdc_form_submission_duration_seconds,
        mdc_gps_capture_duration_seconds,
        mdc_sync_duration_seconds,
        mdc_photo_upload_duration_seconds,
        mdc_package_build_duration_seconds,
        mdc_pending_sync_items,
        mdc_active_devices,
        mdc_offline_devices,
        mdc_storage_used_bytes,
        mdc_pending_uploads,
        # Helper functions
        record_form_submitted,
        record_gps_capture,
        record_photo_captured,
        record_sync_completed,
        record_sync_conflict,
        record_signature_captured,
        record_package_built,
        record_api_error,
        observe_form_submission_duration,
        observe_gps_capture_duration,
        observe_sync_duration,
        observe_photo_upload_duration,
        observe_package_build_duration,
        set_pending_sync_items,
        set_active_devices,
        set_offline_devices,
        set_storage_used_bytes,
        set_pending_uploads,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

# ---- Engine 1: Offline Form Engine ----
try:
    from greenlang.agents.eudr.mobile_data_collector.offline_form_engine import OfflineFormEngine
except ImportError:
    OfflineFormEngine = None  # type: ignore[assignment,misc]

# ---- Engine 2: GPS Capture Engine ----
try:
    from greenlang.agents.eudr.mobile_data_collector.gps_capture_engine import GPSCaptureEngine
except ImportError:
    GPSCaptureEngine = None  # type: ignore[assignment,misc]

# ---- Engine 3: Photo Evidence Collector ----
try:
    from greenlang.agents.eudr.mobile_data_collector.photo_evidence_collector import PhotoEvidenceCollector
except ImportError:
    PhotoEvidenceCollector = None  # type: ignore[assignment,misc]

# ---- Engine 4: Sync Engine ----
try:
    from greenlang.agents.eudr.mobile_data_collector.sync_engine import SyncEngine
except ImportError:
    SyncEngine = None  # type: ignore[assignment,misc]

# ---- Engine 5: Form Template Manager ----
try:
    from greenlang.agents.eudr.mobile_data_collector.form_template_manager import FormTemplateManager
except ImportError:
    FormTemplateManager = None  # type: ignore[assignment,misc]

# ---- Engine 6: Digital Signature Engine ----
try:
    from greenlang.agents.eudr.mobile_data_collector.digital_signature_engine import DigitalSignatureEngine
except ImportError:
    DigitalSignatureEngine = None  # type: ignore[assignment,misc]

# ---- Engine 7: Data Package Builder ----
try:
    from greenlang.agents.eudr.mobile_data_collector.data_package_builder import DataPackageBuilder
except ImportError:
    DataPackageBuilder = None  # type: ignore[assignment,misc]

# ---- Engine 8: Device Fleet Manager ----
try:
    from greenlang.agents.eudr.mobile_data_collector.device_fleet_manager import DeviceFleetManager
except ImportError:
    DeviceFleetManager = None  # type: ignore[assignment,misc]

# ---- Service Facade ----
try:
    from greenlang.agents.eudr.mobile_data_collector.setup import MobileDataCollectorService
except ImportError:
    MobileDataCollectorService = None  # type: ignore[assignment,misc]


__all__ = [
    # -- Version --
    "__version__",
    "__agent_id__",
    "VERSION",
    # -- Config --
    "MobileDataCollectorConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Constants --
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_BATCH_SIZE",
    "EUDR_RETENTION_YEARS",
    "SUPPORTED_PHOTO_FORMATS",
    "SUPPORTED_LANGUAGES",
    "WGS84_SRID",
    "DEFAULT_EUDR_COMMODITIES",
    "SUPPORTED_FORM_TYPES",
    "SUPPORTED_FIELD_TYPES",
    # -- Enumerations --
    "FormStatus",
    "FormType",
    "CaptureAccuracyTier",
    "PhotoType",
    "SyncStatus",
    "ConflictResolution",
    "TemplateType",
    "SignatureAlgorithm",
    "PackageStatus",
    "DeviceStatus",
    "DevicePlatform",
    "CommodityType",
    "ComplianceStatus",
    "FieldType",
    "LanguageCode",
    # -- Core Models --
    "FormSubmission",
    "GPSCapture",
    "PolygonTrace",
    "PhotoEvidence",
    "SyncQueueItem",
    "SyncConflict",
    "FormTemplate",
    "DigitalSignature",
    "DataPackage",
    "DeviceRegistration",
    "DeviceEvent",
    "AuditLogEntry",
    # -- Request Models --
    "SubmitFormRequest",
    "CaptureGPSRequest",
    "CapturePolygonRequest",
    "UploadPhotoRequest",
    "TriggerSyncRequest",
    "ResolveConflictRequest",
    "CreateTemplateRequest",
    "UpdateTemplateRequest",
    "CaptureSignatureRequest",
    "BuildPackageRequest",
    "RegisterDeviceRequest",
    "UpdateDeviceRequest",
    "SearchFormsRequest",
    "GetDeviceStatusRequest",
    "ValidateFormRequest",
    # -- Response Models --
    "FormResponse",
    "GPSResponse",
    "PolygonResponse",
    "PhotoResponse",
    "SyncResponse",
    "ConflictResponse",
    "TemplateResponse",
    "SignatureResponse",
    "PackageResponse",
    "DeviceResponse",
    "DeviceStatusResponse",
    "SearchResponse",
    "SyncStatusResponse",
    "FleetStatusResponse",
    "HealthResponse",
    # -- Provenance --
    "ProvenanceRecord",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "mdc_forms_submitted_total",
    "mdc_gps_captures_total",
    "mdc_photos_captured_total",
    "mdc_syncs_completed_total",
    "mdc_sync_conflicts_total",
    "mdc_signatures_captured_total",
    "mdc_packages_built_total",
    "mdc_api_errors_total",
    "mdc_form_submission_duration_seconds",
    "mdc_gps_capture_duration_seconds",
    "mdc_sync_duration_seconds",
    "mdc_photo_upload_duration_seconds",
    "mdc_package_build_duration_seconds",
    "mdc_pending_sync_items",
    "mdc_active_devices",
    "mdc_offline_devices",
    "mdc_storage_used_bytes",
    "mdc_pending_uploads",
    "record_form_submitted",
    "record_gps_capture",
    "record_photo_captured",
    "record_sync_completed",
    "record_sync_conflict",
    "record_signature_captured",
    "record_package_built",
    "record_api_error",
    "observe_form_submission_duration",
    "observe_gps_capture_duration",
    "observe_sync_duration",
    "observe_photo_upload_duration",
    "observe_package_build_duration",
    "set_pending_sync_items",
    "set_active_devices",
    "set_offline_devices",
    "set_storage_used_bytes",
    "set_pending_uploads",
    # -- Engines --
    "OfflineFormEngine",
    "GPSCaptureEngine",
    "PhotoEvidenceCollector",
    "SyncEngine",
    "FormTemplateManager",
    "DigitalSignatureEngine",
    "DataPackageBuilder",
    "DeviceFleetManager",
    # -- Service Facade --
    "MobileDataCollectorService",
]
