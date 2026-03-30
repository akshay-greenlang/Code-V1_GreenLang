# -*- coding: utf-8 -*-
"""
MobileDataCollectorService - Facade for AGENT-EUDR-015

Unified service facade orchestrating all 8 engines of the Mobile Data
Collector Agent.  Provides a single entry point for form submission,
GPS capture, photo evidence collection, offline synchronization, form
template management, digital signature capture, data package assembly,
and device fleet management.

Engines (8):
    1. OfflineFormEngine         - Offline-first form collection (Feature 1)
    2. GPSCaptureEngine          - GPS point/polygon capture (Feature 2)
    3. PhotoEvidenceCollector    - Geotagged photo evidence (Feature 3)
    4. SyncEngine                - CRDT-based offline sync (Feature 4)
    5. FormTemplateManager       - Dynamic form templates (Feature 5)
    6. DigitalSignatureEngine    - ECDSA P-256 signatures (Feature 6)
    7. DataPackageBuilder        - Merkle-tree data packages (Feature 7)
    8. DeviceFleetManager        - Device fleet management (Feature 8)

Reference Data (3):
    - eudr_form_templates: 6 pre-built EUDR form templates
    - commodity_specifications: 7 EUDR commodity specs
    - language_packs: 30 language translations

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.mobile_data_collector.setup import (
    ...     MobileDataCollectorService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health.status == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-015
Agent ID: GL-EUDR-MDC-015
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 14, 16, 22
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports with graceful fallback
# ---------------------------------------------------------------------------

try:
    from psycopg_pool import AsyncConnectionPool

    PSYCOPG_POOL_AVAILABLE = True
except ImportError:
    AsyncConnectionPool = None  # type: ignore[assignment,misc]
    PSYCOPG_POOL_AVAILABLE = False

try:
    from psycopg import AsyncConnection

    PSYCOPG_AVAILABLE = True
except ImportError:
    AsyncConnection = None  # type: ignore[assignment,misc]
    PSYCOPG_AVAILABLE = False

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False

try:
    from opentelemetry import trace as otel_trace

    OTEL_AVAILABLE = True
except ImportError:
    otel_trace = None  # type: ignore[assignment]
    OTEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Internal imports: config, provenance, metrics
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.mobile_data_collector.config import (
    MobileDataCollectorConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.mobile_data_collector.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from greenlang.agents.eudr.mobile_data_collector.metrics import (
    PROMETHEUS_AVAILABLE,
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

# ---------------------------------------------------------------------------
# Internal imports: models
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.mobile_data_collector.models import (
    VERSION,
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

# ---------------------------------------------------------------------------
# Internal imports: reference data
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.mobile_data_collector.reference_data import (
    ALL_TEMPLATES,
    TEMPLATE_REGISTRY,
    ALL_COMMODITIES,
    VALID_COMMODITY_CODES,
    get_template,
    list_template_names,
    get_template_fields,
    get_required_fields,
    validate_template_data,
    get_commodity,
    is_valid_commodity,
    get_label,
    get_all_labels_for_language,
    list_supported_languages,
)
from greenlang.schemas import utcnow

# ---------------------------------------------------------------------------
# Engine imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.mobile_data_collector.offline_form_engine import OfflineFormEngine
    _OFFLINE_FORM_ENGINE_AVAILABLE = True
except ImportError:
    OfflineFormEngine = None  # type: ignore[assignment,misc]
    _OFFLINE_FORM_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.mobile_data_collector.gps_capture_engine import GPSCaptureEngine
    _GPS_CAPTURE_ENGINE_AVAILABLE = True
except ImportError:
    GPSCaptureEngine = None  # type: ignore[assignment,misc]
    _GPS_CAPTURE_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.mobile_data_collector.photo_evidence_collector import PhotoEvidenceCollector
    _PHOTO_EVIDENCE_COLLECTOR_AVAILABLE = True
except ImportError:
    PhotoEvidenceCollector = None  # type: ignore[assignment,misc]
    _PHOTO_EVIDENCE_COLLECTOR_AVAILABLE = False

try:
    from greenlang.agents.eudr.mobile_data_collector.sync_engine import SyncEngine
    _SYNC_ENGINE_AVAILABLE = True
except ImportError:
    SyncEngine = None  # type: ignore[assignment,misc]
    _SYNC_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.mobile_data_collector.form_template_manager import FormTemplateManager
    _FORM_TEMPLATE_MANAGER_AVAILABLE = True
except ImportError:
    FormTemplateManager = None  # type: ignore[assignment,misc]
    _FORM_TEMPLATE_MANAGER_AVAILABLE = False

try:
    from greenlang.agents.eudr.mobile_data_collector.digital_signature_engine import DigitalSignatureEngine
    _DIGITAL_SIGNATURE_ENGINE_AVAILABLE = True
except ImportError:
    DigitalSignatureEngine = None  # type: ignore[assignment,misc]
    _DIGITAL_SIGNATURE_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.mobile_data_collector.data_package_builder import DataPackageBuilder
    _DATA_PACKAGE_BUILDER_AVAILABLE = True
except ImportError:
    DataPackageBuilder = None  # type: ignore[assignment,misc]
    _DATA_PACKAGE_BUILDER_AVAILABLE = False

try:
    from greenlang.agents.eudr.mobile_data_collector.device_fleet_manager import DeviceFleetManager
    _DEVICE_FLEET_MANAGER_AVAILABLE = True
except ImportError:
    DeviceFleetManager = None  # type: ignore[assignment,misc]
    _DEVICE_FLEET_MANAGER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-MDC-015"
_ENGINE_COUNT = 8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_provenance_hash(*parts: str) -> str:
    """Compute SHA-256 hash over concatenated string parts."""
    combined = "|".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

def _generate_request_id() -> str:
    """Generate a unique request identifier."""
    return f"MDC-{uuid.uuid4().hex[:12]}"

def _elapsed_ms(start: float) -> float:
    """Return elapsed milliseconds since ``start`` (monotonic)."""
    return (time.monotonic() - start) * 1000

# ===========================================================================
# MobileDataCollectorService - Main facade
# ===========================================================================

class MobileDataCollectorService:
    """Facade for the Mobile Data Collector Agent (AGENT-EUDR-015).

    Provides a unified interface to all 8 engines:
        1. OfflineFormEngine         - Offline-first form collection
        2. GPSCaptureEngine          - GPS point/polygon capture
        3. PhotoEvidenceCollector    - Geotagged photo evidence
        4. SyncEngine                - CRDT-based offline sync
        5. FormTemplateManager       - Dynamic form templates
        6. DigitalSignatureEngine    - ECDSA P-256 signatures
        7. DataPackageBuilder        - Merkle-tree data packages
        8. DeviceFleetManager        - Device fleet management

    Thread-safe singleton with double-checked locking.

    Example:
        >>> service = MobileDataCollectorService()
        >>> await service.startup()
        >>> result = await service.submit_form(request)
        >>> await service.shutdown()
    """

    _instance: Optional[MobileDataCollectorService] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize MobileDataCollectorService.

        Loads configuration but does NOT start connections or engines.
        Call ``startup()`` to activate the service.
        """
        self._config: MobileDataCollectorConfig = get_config()

        self._started = False
        self._start_time: Optional[float] = None
        self._config_hash = _compute_provenance_hash(
            self._config.database_url,
            self._config.redis_url,
            str(self._config.min_accuracy_meters),
            str(self._config.max_form_size_kb),
            self._config.genesis_hash,
        )

        # Connection handles
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None

        # Engine instances (lazy initialized)
        self._offline_form_engine: Optional[Any] = None
        self._gps_capture_engine: Optional[Any] = None
        self._photo_evidence_collector: Optional[Any] = None
        self._sync_engine: Optional[Any] = None
        self._form_template_manager: Optional[Any] = None
        self._digital_signature_engine: Optional[Any] = None
        self._data_package_builder: Optional[Any] = None
        self._device_fleet_manager: Optional[Any] = None

        # Reference data (loaded in startup)
        self._ref_templates: Optional[Dict[str, Any]] = None
        self._ref_commodities: Optional[Dict[str, Any]] = None
        self._ref_languages: Optional[List[str]] = None

        # Health check background task
        self._health_task: Optional[asyncio.Task[None]] = None
        self._last_health: Optional[Dict[str, Any]] = None

        # OpenTelemetry tracer
        self._tracer: Optional[Any] = None

        # Provenance tracker
        self._provenance: ProvenanceTracker = get_provenance_tracker()

        # Metrics counters
        self._metrics: Dict[str, int] = {
            "forms_submitted": 0,
            "forms_drafted": 0,
            "gps_captures": 0,
            "polygon_captures": 0,
            "photos_captured": 0,
            "syncs_completed": 0,
            "conflicts_resolved": 0,
            "templates_created": 0,
            "signatures_captured": 0,
            "packages_built": 0,
            "devices_registered": 0,
            "errors": 0,
        }

        logger.info(
            "MobileDataCollectorService created: config_hash=%s, "
            "accuracy=%sm, hdop=%s, version=%s",
            self._config_hash[:12],
            self._config.min_accuracy_meters,
            self._config.hdop_threshold,
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Return whether the service is started and active."""
        return self._started

    @property
    def uptime_seconds(self) -> float:
        """Return seconds since startup, or 0.0 if not started."""
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def config(self) -> MobileDataCollectorConfig:
        """Return the service configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Engine property accessors (lazy initialization)
    # ------------------------------------------------------------------

    @property
    def offline_form_engine(self) -> Any:
        """Return the OfflineFormEngine instance."""
        if self._offline_form_engine is None and _OFFLINE_FORM_ENGINE_AVAILABLE:
            self._offline_form_engine = OfflineFormEngine(self._config)
            logger.info("OfflineFormEngine lazily initialized")
        return self._offline_form_engine

    @property
    def gps_capture_engine(self) -> Any:
        """Return the GPSCaptureEngine instance."""
        if self._gps_capture_engine is None and _GPS_CAPTURE_ENGINE_AVAILABLE:
            self._gps_capture_engine = GPSCaptureEngine(self._config)
            logger.info("GPSCaptureEngine lazily initialized")
        return self._gps_capture_engine

    @property
    def photo_evidence_collector(self) -> Any:
        """Return the PhotoEvidenceCollector instance."""
        if self._photo_evidence_collector is None and _PHOTO_EVIDENCE_COLLECTOR_AVAILABLE:
            self._photo_evidence_collector = PhotoEvidenceCollector(self._config)
            logger.info("PhotoEvidenceCollector lazily initialized")
        return self._photo_evidence_collector

    @property
    def sync_engine(self) -> Any:
        """Return the SyncEngine instance."""
        if self._sync_engine is None and _SYNC_ENGINE_AVAILABLE:
            self._sync_engine = SyncEngine(self._config)
            logger.info("SyncEngine lazily initialized")
        return self._sync_engine

    @property
    def form_template_manager(self) -> Any:
        """Return the FormTemplateManager instance."""
        if self._form_template_manager is None and _FORM_TEMPLATE_MANAGER_AVAILABLE:
            self._form_template_manager = FormTemplateManager(self._config)
            logger.info("FormTemplateManager lazily initialized")
        return self._form_template_manager

    @property
    def digital_signature_engine(self) -> Any:
        """Return the DigitalSignatureEngine instance."""
        if self._digital_signature_engine is None and _DIGITAL_SIGNATURE_ENGINE_AVAILABLE:
            self._digital_signature_engine = DigitalSignatureEngine(self._config)
            logger.info("DigitalSignatureEngine lazily initialized")
        return self._digital_signature_engine

    @property
    def data_package_builder(self) -> Any:
        """Return the DataPackageBuilder instance."""
        if self._data_package_builder is None and _DATA_PACKAGE_BUILDER_AVAILABLE:
            self._data_package_builder = DataPackageBuilder(self._config)
            logger.info("DataPackageBuilder lazily initialized")
        return self._data_package_builder

    @property
    def device_fleet_manager(self) -> Any:
        """Return the DeviceFleetManager instance."""
        if self._device_fleet_manager is None and _DEVICE_FLEET_MANAGER_AVAILABLE:
            self._device_fleet_manager = DeviceFleetManager(self._config)
            logger.info("DeviceFleetManager lazily initialized")
        return self._device_fleet_manager

    # ------------------------------------------------------------------
    # Startup / Shutdown
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Start the service: connect DB, Redis, load reference data.

        Idempotent: safe to call multiple times.
        """
        if self._started:
            logger.debug("MobileDataCollectorService already started")
            return

        start = time.monotonic()
        logger.info("MobileDataCollectorService starting up...")

        self._configure_logging()
        self._init_tracer()
        self._load_reference_data()
        await self._connect_database()
        await self._connect_redis()

        self._started = True
        self._start_time = time.monotonic()
        elapsed = _elapsed_ms(start)

        logger.info(
            "MobileDataCollectorService started in %.1fms: "
            "db=%s, redis=%s, templates=%d, commodities=%d, "
            "languages=%d, config_hash=%s",
            elapsed,
            "connected" if self._db_pool is not None else "skipped",
            "connected" if self._redis is not None else "skipped",
            len(self._ref_templates or {}),
            len(self._ref_commodities or {}),
            len(self._ref_languages or []),
            self._config_hash[:12],
        )

    async def shutdown(self) -> None:
        """Gracefully shut down the service and release all resources.

        Idempotent: safe to call multiple times.
        """
        if not self._started:
            logger.debug("MobileDataCollectorService already stopped")
            return

        logger.info("MobileDataCollectorService shutting down...")
        start = time.monotonic()

        await self._close_redis()
        await self._close_database()

        self._offline_form_engine = None
        self._gps_capture_engine = None
        self._photo_evidence_collector = None
        self._sync_engine = None
        self._form_template_manager = None
        self._digital_signature_engine = None
        self._data_package_builder = None
        self._device_fleet_manager = None

        self._started = False
        elapsed = _elapsed_ms(start)
        logger.info(
            "MobileDataCollectorService shut down in %.1fms", elapsed,
        )

    # ==================================================================
    # FACADE METHODS: Engine 1 - OfflineFormEngine (Forms)
    # ==================================================================

    async def submit_form(self, request: SubmitFormRequest) -> FormResponse:
        """Submit a completed form from a mobile device.

        Args:
            request: Form submission request.

        Returns:
            FormResponse with submission result.
        """
        start = time.monotonic()
        try:
            form = self._build_form_submission(request)
            engine = self.offline_form_engine
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "submit_form", {"form": form},
                )
                if isinstance(result, dict):
                    form = FormSubmission(**result) if "form_id" in result else form

            form.status = FormStatus.PENDING
            submission_hash = _compute_provenance_hash(
                form.form_id, form.device_id,
                form.form_type.value, json.dumps(form.data, sort_keys=True, default=str),
            )
            form.submission_hash = submission_hash

            provenance_hash = self._record_provenance(
                "form_submission", "submit", form.form_id,
                data={"form_type": form.form_type.value, "device_id": form.device_id},
            )

            self._metrics["forms_submitted"] += 1
            commodity_str = form.commodity_type.value if form.commodity_type else "unknown"
            record_form_submitted(form.form_type.value, commodity_str)
            elapsed = _elapsed_ms(start)
            observe_form_submission_duration(elapsed / 1000)

            return FormResponse(
                form_id=form.form_id,
                status=form.status,
                submission_hash=submission_hash,
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed,
                message="Form submitted successfully",
                form=form,
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("submit")
            logger.error("submit_form failed: %s", exc, exc_info=True)
            return FormResponse(
                form_id=request.device_id,
                status=FormStatus.FAILED,
                processing_time_ms=_elapsed_ms(start),
                message=f"Form submission failed: {exc}",
            )

    async def save_draft(self, request: SubmitFormRequest) -> FormResponse:
        """Save a form as a draft on the server.

        Args:
            request: Form data to save as draft.

        Returns:
            FormResponse with draft status.
        """
        start = time.monotonic()
        try:
            form = self._build_form_submission(request)
            form.status = FormStatus.DRAFT

            engine = self.offline_form_engine
            if engine is not None:
                self._safe_engine_call(
                    engine, "save_draft", {"form": form},
                )

            provenance_hash = self._record_provenance(
                "form_submission", "create", form.form_id,
                data={"form_type": form.form_type.value, "status": "draft"},
            )

            self._metrics["forms_drafted"] += 1
            return FormResponse(
                form_id=form.form_id,
                status=FormStatus.DRAFT,
                provenance_hash=provenance_hash,
                processing_time_ms=_elapsed_ms(start),
                message="Draft saved successfully",
                form=form,
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("save_draft")
            logger.error("save_draft failed: %s", exc, exc_info=True)
            return FormResponse(
                form_id="",
                status=FormStatus.FAILED,
                processing_time_ms=_elapsed_ms(start),
                message=f"Draft save failed: {exc}",
            )

    async def get_form(self, form_id: str) -> FormResponse:
        """Retrieve a form submission by ID.

        Args:
            form_id: Form submission identifier.

        Returns:
            FormResponse with form data or error message.
        """
        start = time.monotonic()
        try:
            engine = self.offline_form_engine
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "get_form", {"form_id": form_id},
                )
                if isinstance(result, dict) and "form_id" in result:
                    form = FormSubmission(**result)
                    return FormResponse(
                        form_id=form.form_id,
                        status=form.status,
                        submission_hash=form.submission_hash,
                        processing_time_ms=_elapsed_ms(start),
                        message="Form retrieved",
                        form=form,
                    )

            return FormResponse(
                form_id=form_id,
                status=FormStatus.DRAFT,
                processing_time_ms=_elapsed_ms(start),
                message="Form not found",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("get_form")
            logger.error("get_form failed: %s", exc, exc_info=True)
            return FormResponse(
                form_id=form_id,
                status=FormStatus.FAILED,
                processing_time_ms=_elapsed_ms(start),
                message=f"Form retrieval failed: {exc}",
            )

    async def list_forms(self, request: SearchFormsRequest) -> SearchResponse:
        """Search and list form submissions with filters.

        Args:
            request: Search filters.

        Returns:
            SearchResponse with matching forms.
        """
        start = time.monotonic()
        try:
            engine = self.offline_form_engine
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "search_forms", {
                        "form_type": request.form_type.value if request.form_type else None,
                        "status": request.status.value if request.status else None,
                        "device_id": request.device_id,
                        "operator_id": request.operator_id,
                        "page": request.page,
                        "page_size": request.page_size,
                    },
                )
                if isinstance(result, dict):
                    return SearchResponse(
                        forms=result.get("forms", []),
                        total_count=result.get("total_count", 0),
                        page=request.page,
                        page_size=request.page_size,
                        processing_time_ms=_elapsed_ms(start),
                        message="Search completed",
                    )

            return SearchResponse(
                processing_time_ms=_elapsed_ms(start),
                message="Search completed (no engine)",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("search")
            logger.error("list_forms failed: %s", exc, exc_info=True)
            return SearchResponse(
                processing_time_ms=_elapsed_ms(start),
                message=f"Search failed: {exc}",
            )

    async def validate_form(self, request: ValidateFormRequest) -> FormResponse:
        """Validate form data against a template schema.

        Args:
            request: Validation request with data and template ID.

        Returns:
            FormResponse with validation result.
        """
        start = time.monotonic()
        try:
            # Use reference data for template-based validation
            errors = validate_template_data(request.template_id, request.data)

            engine = self.offline_form_engine
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "validate_form", {
                        "template_id": request.template_id,
                        "data": request.data,
                        "strictness": request.strictness,
                    },
                )
                if isinstance(result, dict):
                    errors.extend(result.get("errors", []))

            status = FormStatus.PENDING if not errors else FormStatus.FAILED
            return FormResponse(
                form_id=request.form_id or "",
                status=status,
                processing_time_ms=_elapsed_ms(start),
                message="Validation passed" if not errors else f"Validation failed: {'; '.join(errors)}",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("validate")
            logger.error("validate_form failed: %s", exc, exc_info=True)
            return FormResponse(
                form_id=request.form_id or "",
                status=FormStatus.FAILED,
                processing_time_ms=_elapsed_ms(start),
                message=f"Validation error: {exc}",
            )

    # ==================================================================
    # FACADE METHODS: Engine 2 - GPSCaptureEngine
    # ==================================================================

    async def capture_point(self, request: CaptureGPSRequest) -> GPSResponse:
        """Capture a GPS point coordinate with accuracy metadata.

        Args:
            request: GPS capture request.

        Returns:
            GPSResponse with capture result.
        """
        start = time.monotonic()
        try:
            capture = GPSCapture(
                form_id=request.form_id,
                device_id=request.device_id,
                operator_id=request.operator_id,
                latitude=request.latitude,
                longitude=request.longitude,
                altitude_m=request.altitude_m,
                horizontal_accuracy_m=request.horizontal_accuracy_m,
                hdop=request.hdop,
                satellite_count=request.satellite_count,
                fix_type=request.fix_type,
                augmentation=request.augmentation,
                capture_timestamp=request.capture_timestamp or utcnow(),
                metadata=request.metadata,
            )
            accuracy_tier = self._classify_accuracy(
                capture.horizontal_accuracy_m, capture.hdop, capture.satellite_count,
            )
            capture.accuracy_tier = accuracy_tier

            engine = self.gps_capture_engine
            if engine is not None:
                self._safe_engine_call(
                    engine, "capture_point", {"capture": capture},
                )

            provenance_hash = self._record_provenance(
                "gps_capture", "capture", capture.capture_id,
                data={
                    "lat": capture.latitude, "lon": capture.longitude,
                    "accuracy": capture.horizontal_accuracy_m,
                },
            )

            self._metrics["gps_captures"] += 1
            record_gps_capture(accuracy_tier.value)
            elapsed = _elapsed_ms(start)
            observe_gps_capture_duration(elapsed / 1000)

            return GPSResponse(
                capture_id=capture.capture_id,
                accuracy_tier=accuracy_tier,
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed,
                message=f"GPS point captured: {accuracy_tier.value}",
                capture=capture,
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("capture")
            logger.error("capture_point failed: %s", exc, exc_info=True)
            return GPSResponse(
                capture_id="",
                accuracy_tier=CaptureAccuracyTier.REJECTED,
                processing_time_ms=_elapsed_ms(start),
                message=f"GPS capture failed: {exc}",
            )

    async def capture_polygon(self, request: CapturePolygonRequest) -> PolygonResponse:
        """Capture a polygon boundary trace.

        Args:
            request: Polygon capture request.

        Returns:
            PolygonResponse with boundary result.
        """
        start = time.monotonic()
        try:
            polygon = PolygonTrace(
                form_id=request.form_id,
                device_id=request.device_id,
                operator_id=request.operator_id,
                vertices=request.vertices,
                vertex_accuracies_m=request.vertex_accuracies_m,
                vertex_count=len(request.vertices),
                capture_start=request.capture_start,
                capture_end=request.capture_end,
                metadata=request.metadata,
            )

            # Check polygon closure
            if len(polygon.vertices) >= 3:
                first = polygon.vertices[0]
                last = polygon.vertices[-1]
                polygon.is_closed = (first == last)
                polygon.is_valid = polygon.is_closed and len(polygon.vertices) >= 4

            # Calculate area via Shoelace approximation
            area_ha = self._calculate_polygon_area(polygon.vertices)
            polygon.area_ha = area_ha

            engine = self.gps_capture_engine
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "capture_polygon", {"polygon": polygon},
                )
                if isinstance(result, dict):
                    if "area_ha" in result:
                        polygon.area_ha = result["area_ha"]
                    if "perimeter_m" in result:
                        polygon.perimeter_m = result["perimeter_m"]

            provenance_hash = self._record_provenance(
                "polygon_trace", "capture", polygon.polygon_id,
                data={"vertex_count": polygon.vertex_count, "area_ha": polygon.area_ha},
            )

            self._metrics["polygon_captures"] += 1
            return PolygonResponse(
                polygon_id=polygon.polygon_id,
                area_ha=polygon.area_ha,
                vertex_count=polygon.vertex_count,
                is_valid=polygon.is_valid,
                provenance_hash=provenance_hash,
                processing_time_ms=_elapsed_ms(start),
                message=f"Polygon captured: {polygon.vertex_count} vertices, {polygon.area_ha or 0:.4f} ha",
                polygon=polygon,
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("capture_polygon")
            logger.error("capture_polygon failed: %s", exc, exc_info=True)
            return PolygonResponse(
                polygon_id="",
                processing_time_ms=_elapsed_ms(start),
                message=f"Polygon capture failed: {exc}",
            )

    async def validate_coordinates(
        self, latitude: float, longitude: float,
    ) -> Dict[str, Any]:
        """Validate WGS84 coordinates are within valid ranges.

        Args:
            latitude: WGS84 latitude (-90 to 90).
            longitude: WGS84 longitude (-180 to 180).

        Returns:
            Validation result dictionary.
        """
        valid = -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0
        return {
            "valid": valid,
            "latitude": latitude,
            "longitude": longitude,
            "message": "Coordinates valid" if valid else "Coordinates out of range",
        }

    async def calculate_area(self, vertices: List[List[float]]) -> Dict[str, Any]:
        """Calculate the area of a polygon in hectares.

        Args:
            vertices: List of [latitude, longitude] pairs.

        Returns:
            Dictionary with area_ha and vertex_count.
        """
        area_ha = self._calculate_polygon_area(vertices)
        return {
            "area_ha": area_ha,
            "vertex_count": len(vertices),
            "is_valid": len(vertices) >= 3,
        }

    # ==================================================================
    # FACADE METHODS: Engine 3 - PhotoEvidenceCollector
    # ==================================================================

    async def capture_photo(self, request: UploadPhotoRequest) -> PhotoResponse:
        """Process a photo upload with metadata and integrity hash.

        Args:
            request: Photo upload request.

        Returns:
            PhotoResponse with photo metadata.
        """
        start = time.monotonic()
        try:
            photo = PhotoEvidence(
                form_id=request.form_id,
                capture_id=request.capture_id,
                device_id=request.device_id,
                operator_id=request.operator_id,
                photo_type=request.photo_type,
                file_name=request.file_name,
                file_size_bytes=request.file_size_bytes,
                file_format=request.file_format,
                width_px=request.width_px,
                height_px=request.height_px,
                integrity_hash=request.integrity_hash,
                latitude=request.latitude,
                longitude=request.longitude,
                exif_timestamp=request.exif_timestamp,
                device_timestamp=request.device_timestamp or utcnow(),
                compression_quality=request.compression_quality,
                annotation=request.annotation,
                sequence_number=request.sequence_number,
                batch_group_id=request.batch_group_id,
                metadata=request.metadata,
            )

            engine = self.photo_evidence_collector
            if engine is not None:
                self._safe_engine_call(
                    engine, "process_photo", {"photo": photo},
                )

            provenance_hash = self._record_provenance(
                "photo_evidence", "capture", photo.photo_id,
                data={
                    "integrity_hash": photo.integrity_hash,
                    "photo_type": photo.photo_type.value,
                },
            )

            self._metrics["photos_captured"] += 1
            record_photo_captured(photo.photo_type.value)
            elapsed = _elapsed_ms(start)
            observe_photo_upload_duration(elapsed / 1000)

            return PhotoResponse(
                photo_id=photo.photo_id,
                integrity_hash=photo.integrity_hash,
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed,
                message="Photo processed successfully",
                photo=photo,
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("upload_photo")
            logger.error("capture_photo failed: %s", exc, exc_info=True)
            return PhotoResponse(
                photo_id="",
                integrity_hash="",
                processing_time_ms=_elapsed_ms(start),
                message=f"Photo processing failed: {exc}",
            )

    async def get_photo(self, photo_id: str) -> PhotoResponse:
        """Retrieve a photo record by ID.

        Args:
            photo_id: Photo evidence identifier.

        Returns:
            PhotoResponse with photo data.
        """
        start = time.monotonic()
        try:
            engine = self.photo_evidence_collector
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "get_photo", {"photo_id": photo_id},
                )
                if isinstance(result, dict) and "photo_id" in result:
                    photo = PhotoEvidence(**result)
                    return PhotoResponse(
                        photo_id=photo.photo_id,
                        integrity_hash=photo.integrity_hash,
                        processing_time_ms=_elapsed_ms(start),
                        message="Photo retrieved",
                        photo=photo,
                    )

            return PhotoResponse(
                photo_id=photo_id,
                integrity_hash="",
                processing_time_ms=_elapsed_ms(start),
                message="Photo not found",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("get_photo failed: %s", exc, exc_info=True)
            return PhotoResponse(
                photo_id=photo_id,
                integrity_hash="",
                processing_time_ms=_elapsed_ms(start),
                message=f"Photo retrieval failed: {exc}",
            )

    async def list_photos(self, form_id: str) -> List[PhotoEvidence]:
        """List all photos associated with a form submission.

        Args:
            form_id: Form submission identifier.

        Returns:
            List of PhotoEvidence records.
        """
        engine = self.photo_evidence_collector
        if engine is not None:
            result = self._safe_engine_call(
                engine, "list_photos", {"form_id": form_id},
            )
            if isinstance(result, list):
                return result
        return []

    async def validate_geotag(
        self, photo_id: str, latitude: float, longitude: float,
    ) -> Dict[str, Any]:
        """Validate a photo's geotag against expected coordinates.

        Args:
            photo_id: Photo identifier.
            latitude: Expected latitude.
            longitude: Expected longitude.

        Returns:
            Validation result dictionary.
        """
        engine = self.photo_evidence_collector
        if engine is not None:
            result = self._safe_engine_call(
                engine, "validate_geotag", {
                    "photo_id": photo_id,
                    "latitude": latitude,
                    "longitude": longitude,
                },
            )
            if isinstance(result, dict):
                return result
        return {"valid": True, "photo_id": photo_id, "message": "Geotag validation skipped (no engine)"}

    # ==================================================================
    # FACADE METHODS: Engine 4 - SyncEngine
    # ==================================================================

    async def start_sync(self, request: TriggerSyncRequest) -> SyncResponse:
        """Trigger a sync session for a device.

        Args:
            request: Sync trigger request.

        Returns:
            SyncResponse with sync result.
        """
        start = time.monotonic()
        try:
            engine = self.sync_engine
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "start_sync", {
                        "device_id": request.device_id,
                        "force": request.force,
                        "max_items": request.max_items,
                        "item_types": request.item_types,
                    },
                )
                if isinstance(result, dict):
                    elapsed = _elapsed_ms(start)
                    observe_sync_duration(elapsed / 1000)
                    self._metrics["syncs_completed"] += 1
                    record_sync_completed()

                    return SyncResponse(
                        device_id=request.device_id,
                        items_queued=result.get("items_queued", 0),
                        items_completed=result.get("items_completed", 0),
                        items_failed=result.get("items_failed", 0),
                        conflicts_detected=result.get("conflicts_detected", 0),
                        bytes_uploaded=result.get("bytes_uploaded", 0),
                        processing_time_ms=elapsed,
                        message="Sync completed",
                    )

            return SyncResponse(
                device_id=request.device_id,
                processing_time_ms=_elapsed_ms(start),
                message="Sync engine not available",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("sync")
            logger.error("start_sync failed: %s", exc, exc_info=True)
            return SyncResponse(
                device_id=request.device_id,
                processing_time_ms=_elapsed_ms(start),
                message=f"Sync failed: {exc}",
            )

    async def get_sync_status(self, device_id: str) -> SyncStatusResponse:
        """Get sync queue status for a device.

        Args:
            device_id: Device identifier.

        Returns:
            SyncStatusResponse with queue metrics.
        """
        start = time.monotonic()
        try:
            engine = self.sync_engine
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "get_status", {"device_id": device_id},
                )
                if isinstance(result, dict):
                    return SyncStatusResponse(
                        device_id=device_id,
                        pending_items=result.get("pending_items", 0),
                        in_progress_items=result.get("in_progress_items", 0),
                        completed_items=result.get("completed_items", 0),
                        failed_items=result.get("failed_items", 0),
                        total_bytes_pending=result.get("total_bytes_pending", 0),
                        last_sync_at=result.get("last_sync_at"),
                        unresolved_conflicts=result.get("unresolved_conflicts", 0),
                        processing_time_ms=_elapsed_ms(start),
                        message="Status retrieved",
                    )

            return SyncStatusResponse(
                device_id=device_id,
                processing_time_ms=_elapsed_ms(start),
                message="Sync engine not available",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("get_sync_status failed: %s", exc, exc_info=True)
            return SyncStatusResponse(
                device_id=device_id,
                processing_time_ms=_elapsed_ms(start),
                message=f"Status query failed: {exc}",
            )

    async def resolve_conflict(self, request: ResolveConflictRequest) -> ConflictResponse:
        """Resolve a sync conflict.

        Args:
            request: Conflict resolution request.

        Returns:
            ConflictResponse with resolution result.
        """
        start = time.monotonic()
        try:
            engine = self.sync_engine
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "resolve_conflict", {
                        "conflict_id": request.conflict_id,
                        "resolution_strategy": request.resolution_strategy.value,
                        "resolved_value": request.resolved_value,
                        "resolved_by": request.resolved_by,
                        "reason": request.reason,
                    },
                )
                if isinstance(result, dict):
                    self._metrics["conflicts_resolved"] += 1
                    record_sync_conflict()

                    self._record_provenance(
                        "sync_conflict", "resolve", request.conflict_id,
                        data={"strategy": request.resolution_strategy.value},
                    )

                    return ConflictResponse(
                        conflict_id=request.conflict_id,
                        resolved=True,
                        resolution_strategy=request.resolution_strategy,
                        resolved_value=request.resolved_value,
                        processing_time_ms=_elapsed_ms(start),
                        message="Conflict resolved",
                    )

            return ConflictResponse(
                conflict_id=request.conflict_id,
                resolved=False,
                resolution_strategy=request.resolution_strategy,
                processing_time_ms=_elapsed_ms(start),
                message="Sync engine not available",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("resolve")
            logger.error("resolve_conflict failed: %s", exc, exc_info=True)
            return ConflictResponse(
                conflict_id=request.conflict_id,
                resolved=False,
                resolution_strategy=request.resolution_strategy,
                processing_time_ms=_elapsed_ms(start),
                message=f"Conflict resolution failed: {exc}",
            )

    async def get_queue_depth(self, device_id: str) -> int:
        """Get the number of pending sync items for a device.

        Args:
            device_id: Device identifier.

        Returns:
            Number of pending sync items.
        """
        status = await self.get_sync_status(device_id)
        return status.pending_items

    # ==================================================================
    # FACADE METHODS: Engine 5 - FormTemplateManager
    # ==================================================================

    async def create_template(self, request: CreateTemplateRequest) -> TemplateResponse:
        """Create a new form template.

        Args:
            request: Template creation request.

        Returns:
            TemplateResponse with created template.
        """
        start = time.monotonic()
        try:
            template = FormTemplate(
                name=request.name,
                form_type=request.form_type,
                template_type=request.template_type,
                parent_template_id=request.parent_template_id,
                schema_definition=request.schema_definition,
                fields=request.fields,
                conditional_logic=request.conditional_logic,
                validation_rules=request.validation_rules,
                language_packs=request.language_packs,
                metadata=request.metadata,
            )

            engine = self.form_template_manager
            if engine is not None:
                self._safe_engine_call(
                    engine, "create_template", {"template": template},
                )

            self._record_provenance(
                "form_template", "create", template.template_id,
                data={"name": template.name, "form_type": template.form_type.value},
            )

            self._metrics["templates_created"] += 1
            return TemplateResponse(
                template_id=template.template_id,
                version=template.version,
                processing_time_ms=_elapsed_ms(start),
                message="Template created",
                template=template,
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("create_template")
            logger.error("create_template failed: %s", exc, exc_info=True)
            return TemplateResponse(
                template_id="",
                processing_time_ms=_elapsed_ms(start),
                message=f"Template creation failed: {exc}",
            )

    async def get_template(self, template_id: str) -> TemplateResponse:
        """Retrieve a form template by ID.

        Args:
            template_id: Template identifier (or form_type for built-in).

        Returns:
            TemplateResponse with template data.
        """
        start = time.monotonic()
        try:
            # Check built-in templates first
            builtin = get_template(template_id)
            if builtin is not None:
                return TemplateResponse(
                    template_id=builtin["template_id"],
                    version=builtin.get("version", "1.0.0"),
                    processing_time_ms=_elapsed_ms(start),
                    message="Built-in template retrieved",
                )

            engine = self.form_template_manager
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "get_template", {"template_id": template_id},
                )
                if isinstance(result, dict) and "template_id" in result:
                    tmpl = FormTemplate(**result)
                    return TemplateResponse(
                        template_id=tmpl.template_id,
                        version=tmpl.version,
                        processing_time_ms=_elapsed_ms(start),
                        message="Template retrieved",
                        template=tmpl,
                    )

            return TemplateResponse(
                template_id=template_id,
                processing_time_ms=_elapsed_ms(start),
                message="Template not found",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("get_template failed: %s", exc, exc_info=True)
            return TemplateResponse(
                template_id=template_id,
                processing_time_ms=_elapsed_ms(start),
                message=f"Template retrieval failed: {exc}",
            )

    async def update_template(self, request: UpdateTemplateRequest) -> TemplateResponse:
        """Update an existing form template.

        Args:
            request: Template update request.

        Returns:
            TemplateResponse with updated template.
        """
        start = time.monotonic()
        try:
            engine = self.form_template_manager
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "update_template", {
                        "template_id": request.template_id,
                        "name": request.name,
                        "fields": request.fields,
                        "language_packs": request.language_packs,
                        "is_active": request.is_active,
                    },
                )
                self._record_provenance(
                    "form_template", "update", request.template_id,
                    data={"name": request.name},
                )
                return TemplateResponse(
                    template_id=request.template_id,
                    processing_time_ms=_elapsed_ms(start),
                    message="Template updated",
                )

            return TemplateResponse(
                template_id=request.template_id,
                processing_time_ms=_elapsed_ms(start),
                message="Template manager not available",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("update_template")
            logger.error("update_template failed: %s", exc, exc_info=True)
            return TemplateResponse(
                template_id=request.template_id,
                processing_time_ms=_elapsed_ms(start),
                message=f"Template update failed: {exc}",
            )

    async def publish_template(self, template_id: str) -> TemplateResponse:
        """Publish a template for deployment to devices.

        Args:
            template_id: Template to publish.

        Returns:
            TemplateResponse with publish result.
        """
        start = time.monotonic()
        try:
            engine = self.form_template_manager
            if engine is not None:
                self._safe_engine_call(
                    engine, "publish_template", {"template_id": template_id},
                )
            self._record_provenance(
                "form_template", "update", template_id,
                data={"action": "publish"},
            )
            return TemplateResponse(
                template_id=template_id,
                processing_time_ms=_elapsed_ms(start),
                message="Template published",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("publish_template failed: %s", exc, exc_info=True)
            return TemplateResponse(
                template_id=template_id,
                processing_time_ms=_elapsed_ms(start),
                message=f"Template publish failed: {exc}",
            )

    async def list_templates(self) -> List[str]:
        """Return list of all available template names (built-in + custom).

        Returns:
            Sorted list of template form type names.
        """
        names = list(list_template_names())
        engine = self.form_template_manager
        if engine is not None:
            result = self._safe_engine_call(engine, "list_templates", {})
            if isinstance(result, list):
                for name in result:
                    if name not in names:
                        names.append(name)
        return sorted(names)

    async def render_template(
        self, template_id: str, language: str = "en",
    ) -> Dict[str, Any]:
        """Render a template with language-specific labels.

        Args:
            template_id: Template identifier or form_type key.
            language: Language code for rendering.

        Returns:
            Rendered template dictionary with translated labels.
        """
        tmpl = get_template(template_id)
        if tmpl is None:
            return {"error": f"Template not found: {template_id}"}

        labels = get_all_labels_for_language(language)
        return {
            "template": tmpl,
            "language": language,
            "labels": labels,
        }

    # ==================================================================
    # FACADE METHODS: Engine 6 - DigitalSignatureEngine
    # ==================================================================

    async def create_signature(self, request: CaptureSignatureRequest) -> SignatureResponse:
        """Capture a digital signature for a form.

        Args:
            request: Signature capture request.

        Returns:
            SignatureResponse with signature data.
        """
        start = time.monotonic()
        try:
            signature = DigitalSignature(
                form_id=request.form_id,
                signer_name=request.signer_name,
                signer_role=request.signer_role,
                signer_device_id=request.signer_device_id,
                algorithm=request.algorithm,
                public_key_fingerprint=request.public_key_fingerprint,
                signature_bytes_hex=request.signature_bytes_hex,
                signed_data_hash=request.signed_data_hash,
                timestamp_binding=request.timestamp_binding,
                visual_signature_svg=request.visual_signature_svg,
                metadata=request.metadata,
            )

            engine = self.digital_signature_engine
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "create_signature", {"signature": signature},
                )
                if isinstance(result, dict):
                    signature.is_valid = result.get("is_valid", False)

            provenance_hash = self._record_provenance(
                "digital_signature", "sign", signature.signature_id,
                data={
                    "form_id": signature.form_id,
                    "signer": signature.signer_name,
                    "algorithm": signature.algorithm.value,
                },
            )

            self._metrics["signatures_captured"] += 1
            record_signature_captured()

            return SignatureResponse(
                signature_id=signature.signature_id,
                is_valid=signature.is_valid,
                provenance_hash=provenance_hash,
                processing_time_ms=_elapsed_ms(start),
                message="Signature captured",
                signature=signature,
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("sign")
            logger.error("create_signature failed: %s", exc, exc_info=True)
            return SignatureResponse(
                signature_id="",
                processing_time_ms=_elapsed_ms(start),
                message=f"Signature capture failed: {exc}",
            )

    async def verify_signature(self, signature_id: str) -> SignatureResponse:
        """Verify a digital signature.

        Args:
            signature_id: Signature identifier.

        Returns:
            SignatureResponse with verification result.
        """
        start = time.monotonic()
        try:
            engine = self.digital_signature_engine
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "verify_signature", {"signature_id": signature_id},
                )
                if isinstance(result, dict):
                    self._record_provenance(
                        "digital_signature", "verify", signature_id,
                        data={"is_valid": result.get("is_valid", False)},
                    )
                    return SignatureResponse(
                        signature_id=signature_id,
                        is_valid=result.get("is_valid", False),
                        processing_time_ms=_elapsed_ms(start),
                        message="Signature verified" if result.get("is_valid") else "Signature invalid",
                    )

            return SignatureResponse(
                signature_id=signature_id,
                is_valid=False,
                processing_time_ms=_elapsed_ms(start),
                message="Signature engine not available",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("verify")
            logger.error("verify_signature failed: %s", exc, exc_info=True)
            return SignatureResponse(
                signature_id=signature_id,
                processing_time_ms=_elapsed_ms(start),
                message=f"Signature verification failed: {exc}",
            )

    async def get_signature(self, signature_id: str) -> SignatureResponse:
        """Retrieve a signature record by ID.

        Args:
            signature_id: Signature identifier.

        Returns:
            SignatureResponse with signature data.
        """
        start = time.monotonic()
        engine = self.digital_signature_engine
        if engine is not None:
            result = self._safe_engine_call(
                engine, "get_signature", {"signature_id": signature_id},
            )
            if isinstance(result, dict) and "signature_id" in result:
                sig = DigitalSignature(**result)
                return SignatureResponse(
                    signature_id=sig.signature_id,
                    is_valid=sig.is_valid,
                    processing_time_ms=_elapsed_ms(start),
                    message="Signature retrieved",
                    signature=sig,
                )
        return SignatureResponse(
            signature_id=signature_id,
            processing_time_ms=_elapsed_ms(start),
            message="Signature not found",
        )

    async def create_custody_signature(
        self, form_id: str, signer_name: str, signer_role: str,
        signer_device_id: str,
    ) -> SignatureResponse:
        """Create a signature specifically for custody transfer forms.

        Args:
            form_id: Custody transfer form ID.
            signer_name: Name of the signatory.
            signer_role: Role (producer, collector, transporter, etc.).
            signer_device_id: Device used for signing.

        Returns:
            SignatureResponse with custody signature.
        """
        request = CaptureSignatureRequest(
            form_id=form_id,
            signer_name=signer_name,
            signer_role=signer_role,
            signer_device_id=signer_device_id,
            metadata={"context": "custody_transfer"},
        )
        return await self.create_signature(request)

    # ==================================================================
    # FACADE METHODS: Engine 7 - DataPackageBuilder
    # ==================================================================

    async def create_package(self, request: BuildPackageRequest) -> PackageResponse:
        """Create a new data package.

        Args:
            request: Package build request.

        Returns:
            PackageResponse with package data.
        """
        start = time.monotonic()
        try:
            package = DataPackage(
                device_id=request.device_id,
                operator_id=request.operator_id,
                form_ids=request.form_ids,
                gps_capture_ids=request.gps_capture_ids,
                photo_ids=request.photo_ids,
                signature_ids=request.signature_ids,
                compression_format=request.compression_format,
                compression_level=request.compression_level,
                export_format=request.export_format,
                metadata=request.metadata,
            )
            package.artifact_count = (
                len(request.form_ids) + len(request.gps_capture_ids)
                + len(request.photo_ids) + len(request.signature_ids)
            )

            engine = self.data_package_builder
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "create_package", {"package": package},
                )
                if isinstance(result, dict):
                    if "merkle_root" in result:
                        package.merkle_root = result["merkle_root"]
                    if "package_size_bytes" in result:
                        package.package_size_bytes = result["package_size_bytes"]

            if request.seal:
                package.status = PackageStatus.SEALED
                package.sealed_at = utcnow()

            provenance_hash = self._record_provenance(
                "data_package", "build", package.package_id,
                data={
                    "artifact_count": package.artifact_count,
                    "export_format": package.export_format,
                },
            )

            self._metrics["packages_built"] += 1
            record_package_built()
            elapsed = _elapsed_ms(start)
            observe_package_build_duration(elapsed / 1000)

            return PackageResponse(
                package_id=package.package_id,
                status=package.status,
                merkle_root=package.merkle_root,
                artifact_count=package.artifact_count,
                package_size_bytes=package.package_size_bytes,
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed,
                message="Package created",
                package=package,
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("build_package")
            logger.error("create_package failed: %s", exc, exc_info=True)
            return PackageResponse(
                package_id="",
                status=PackageStatus.BUILDING,
                processing_time_ms=_elapsed_ms(start),
                message=f"Package creation failed: {exc}",
            )

    async def add_form_to_package(
        self, package_id: str, form_id: str,
    ) -> PackageResponse:
        """Add a form to an existing package.

        Args:
            package_id: Package identifier.
            form_id: Form to add.

        Returns:
            PackageResponse with updated package.
        """
        start = time.monotonic()
        try:
            engine = self.data_package_builder
            if engine is not None:
                self._safe_engine_call(
                    engine, "add_artifact", {
                        "package_id": package_id,
                        "artifact_type": "form",
                        "artifact_id": form_id,
                    },
                )
            return PackageResponse(
                package_id=package_id,
                status=PackageStatus.BUILDING,
                processing_time_ms=_elapsed_ms(start),
                message=f"Form {form_id} added to package",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("add_form_to_package failed: %s", exc, exc_info=True)
            return PackageResponse(
                package_id=package_id,
                status=PackageStatus.BUILDING,
                processing_time_ms=_elapsed_ms(start),
                message=f"Add form failed: {exc}",
            )

    async def seal_package(self, package_id: str) -> PackageResponse:
        """Seal a package and compute Merkle root.

        Args:
            package_id: Package to seal.

        Returns:
            PackageResponse with sealed status.
        """
        start = time.monotonic()
        try:
            engine = self.data_package_builder
            merkle_root = None
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "seal_package", {"package_id": package_id},
                )
                if isinstance(result, dict):
                    merkle_root = result.get("merkle_root")

            self._record_provenance(
                "data_package", "build", package_id,
                data={"action": "seal", "merkle_root": merkle_root},
            )

            return PackageResponse(
                package_id=package_id,
                status=PackageStatus.SEALED,
                merkle_root=merkle_root,
                processing_time_ms=_elapsed_ms(start),
                message="Package sealed",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("seal_package failed: %s", exc, exc_info=True)
            return PackageResponse(
                package_id=package_id,
                status=PackageStatus.BUILDING,
                processing_time_ms=_elapsed_ms(start),
                message=f"Package seal failed: {exc}",
            )

    async def validate_package(self, package_id: str) -> PackageResponse:
        """Validate a sealed package's integrity.

        Args:
            package_id: Package to validate.

        Returns:
            PackageResponse with validation result.
        """
        start = time.monotonic()
        engine = self.data_package_builder
        if engine is not None:
            result = self._safe_engine_call(
                engine, "validate_package", {"package_id": package_id},
            )
            if isinstance(result, dict):
                return PackageResponse(
                    package_id=package_id,
                    status=PackageStatus.VERIFIED if result.get("valid") else PackageStatus.SEALED,
                    merkle_root=result.get("merkle_root"),
                    processing_time_ms=_elapsed_ms(start),
                    message="Package valid" if result.get("valid") else "Package integrity check failed",
                )

        return PackageResponse(
            package_id=package_id,
            status=PackageStatus.SEALED,
            processing_time_ms=_elapsed_ms(start),
            message="Package builder not available",
        )

    async def export_package(
        self, package_id: str, export_format: str = "zip",
    ) -> PackageResponse:
        """Export a sealed package in the specified format.

        Args:
            package_id: Package to export.
            export_format: Export format (zip, tar_gz, json_ld).

        Returns:
            PackageResponse with export data.
        """
        start = time.monotonic()
        engine = self.data_package_builder
        if engine is not None:
            result = self._safe_engine_call(
                engine, "export_package", {
                    "package_id": package_id,
                    "export_format": export_format,
                },
            )
            if isinstance(result, dict):
                return PackageResponse(
                    package_id=package_id,
                    status=PackageStatus.SEALED,
                    processing_time_ms=_elapsed_ms(start),
                    message=f"Package exported as {export_format}",
                    package=None,
                )

        return PackageResponse(
            package_id=package_id,
            status=PackageStatus.SEALED,
            processing_time_ms=_elapsed_ms(start),
            message="Package builder not available",
        )

    # ==================================================================
    # FACADE METHODS: Engine 8 - DeviceFleetManager
    # ==================================================================

    async def register_device(self, request: RegisterDeviceRequest) -> DeviceResponse:
        """Register a new device in the fleet.

        Args:
            request: Device registration request.

        Returns:
            DeviceResponse with registration result.
        """
        start = time.monotonic()
        try:
            device = DeviceRegistration(
                device_model=request.device_model,
                platform=request.platform,
                os_version=request.os_version,
                agent_version=request.agent_version,
                assigned_operator_id=request.assigned_operator_id,
                assigned_area=request.assigned_area,
                metadata=request.metadata,
            )

            engine = self.device_fleet_manager
            if engine is not None:
                self._safe_engine_call(
                    engine, "register_device", {"device": device},
                )

            self._record_provenance(
                "device_registration", "register", device.device_id,
                data={
                    "model": device.device_model,
                    "platform": device.platform.value,
                },
            )

            self._metrics["devices_registered"] += 1
            return DeviceResponse(
                device_id=device.device_id,
                status=DeviceStatus.ACTIVE,
                processing_time_ms=_elapsed_ms(start),
                message="Device registered",
                device=device,
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("register")
            logger.error("register_device failed: %s", exc, exc_info=True)
            return DeviceResponse(
                device_id="",
                status=DeviceStatus.OFFLINE,
                processing_time_ms=_elapsed_ms(start),
                message=f"Device registration failed: {exc}",
            )

    async def get_device(self, device_id: str) -> DeviceResponse:
        """Retrieve a device registration by ID.

        Args:
            device_id: Device identifier.

        Returns:
            DeviceResponse with device data.
        """
        start = time.monotonic()
        engine = self.device_fleet_manager
        if engine is not None:
            result = self._safe_engine_call(
                engine, "get_device", {"device_id": device_id},
            )
            if isinstance(result, dict) and "device_id" in result:
                device = DeviceRegistration(**result)
                return DeviceResponse(
                    device_id=device.device_id,
                    status=device.status,
                    processing_time_ms=_elapsed_ms(start),
                    message="Device retrieved",
                    device=device,
                )

        return DeviceResponse(
            device_id=device_id,
            status=DeviceStatus.OFFLINE,
            processing_time_ms=_elapsed_ms(start),
            message="Device not found",
        )

    async def update_device(self, request: UpdateDeviceRequest) -> DeviceResponse:
        """Update a device registration.

        Args:
            request: Device update request.

        Returns:
            DeviceResponse with update result.
        """
        start = time.monotonic()
        try:
            engine = self.device_fleet_manager
            if engine is not None:
                self._safe_engine_call(
                    engine, "update_device", {
                        "device_id": request.device_id,
                        "assigned_operator_id": request.assigned_operator_id,
                        "assigned_area": request.assigned_area,
                        "agent_version": request.agent_version,
                        "metadata": request.metadata,
                    },
                )
            self._record_provenance(
                "device_registration", "update", request.device_id,
                data={"operator": request.assigned_operator_id},
            )
            return DeviceResponse(
                device_id=request.device_id,
                status=DeviceStatus.ACTIVE,
                processing_time_ms=_elapsed_ms(start),
                message="Device updated",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("update_device")
            logger.error("update_device failed: %s", exc, exc_info=True)
            return DeviceResponse(
                device_id=request.device_id,
                status=DeviceStatus.OFFLINE,
                processing_time_ms=_elapsed_ms(start),
                message=f"Device update failed: {exc}",
            )

    async def get_fleet_status(self) -> FleetStatusResponse:
        """Get fleet-level dashboard status aggregation.

        Returns:
            FleetStatusResponse with fleet metrics.
        """
        start = time.monotonic()
        try:
            engine = self.device_fleet_manager
            if engine is not None:
                result = self._safe_engine_call(
                    engine, "get_fleet_status", {},
                )
                if isinstance(result, dict):
                    resp = FleetStatusResponse(
                        total_devices=result.get("total_devices", 0),
                        active_devices=result.get("active_devices", 0),
                        offline_devices=result.get("offline_devices", 0),
                        low_battery_devices=result.get("low_battery_devices", 0),
                        low_storage_devices=result.get("low_storage_devices", 0),
                        decommissioned_devices=result.get("decommissioned_devices", 0),
                        outdated_agent_devices=result.get("outdated_agent_devices", 0),
                        total_pending_sync_bytes=result.get("total_pending_sync_bytes", 0),
                        total_pending_forms=result.get("total_pending_forms", 0),
                        total_pending_photos=result.get("total_pending_photos", 0),
                        processing_time_ms=_elapsed_ms(start),
                        message="Fleet status retrieved",
                    )
                    set_active_devices(resp.active_devices)
                    set_offline_devices(resp.offline_devices)
                    return resp

            return FleetStatusResponse(
                processing_time_ms=_elapsed_ms(start),
                message="Fleet manager not available",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("get_fleet_status failed: %s", exc, exc_info=True)
            return FleetStatusResponse(
                processing_time_ms=_elapsed_ms(start),
                message=f"Fleet status query failed: {exc}",
            )

    async def record_heartbeat(
        self, device_id: str, event: DeviceEvent,
    ) -> DeviceResponse:
        """Record a device heartbeat telemetry event.

        Args:
            device_id: Device identifier.
            event: Telemetry event data.

        Returns:
            DeviceResponse acknowledging the heartbeat.
        """
        start = time.monotonic()
        try:
            engine = self.device_fleet_manager
            if engine is not None:
                self._safe_engine_call(
                    engine, "record_heartbeat", {
                        "device_id": device_id,
                        "event": event,
                    },
                )
            return DeviceResponse(
                device_id=device_id,
                status=DeviceStatus.ACTIVE,
                processing_time_ms=_elapsed_ms(start),
                message="Heartbeat recorded",
            )
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("record_heartbeat failed: %s", exc, exc_info=True)
            return DeviceResponse(
                device_id=device_id,
                status=DeviceStatus.OFFLINE,
                processing_time_ms=_elapsed_ms(start),
                message=f"Heartbeat failed: {exc}",
            )

    # ==================================================================
    # Cross-cutting methods
    # ==================================================================

    async def build_complete_package(
        self,
        device_id: str,
        operator_id: str,
        form_ids: List[str],
        gps_capture_ids: Optional[List[str]] = None,
        photo_ids: Optional[List[str]] = None,
        signature_ids: Optional[List[str]] = None,
        export_format: str = "zip",
    ) -> PackageResponse:
        """Assemble a complete data package from forms, GPS, photos, and signatures.

        This cross-engine method fetches all related data and builds
        a single sealed package ready for export.

        Args:
            device_id: Device assembling the package.
            operator_id: Operator building the package.
            form_ids: Form submission IDs to include.
            gps_capture_ids: GPS capture IDs (auto-discovered if None).
            photo_ids: Photo IDs (auto-discovered if None).
            signature_ids: Signature IDs (auto-discovered if None).
            export_format: Export format (zip, tar_gz, json_ld).

        Returns:
            PackageResponse with sealed package.
        """
        request = BuildPackageRequest(
            device_id=device_id,
            operator_id=operator_id,
            form_ids=form_ids,
            gps_capture_ids=gps_capture_ids or [],
            photo_ids=photo_ids or [],
            signature_ids=signature_ids or [],
            export_format=export_format,
            seal=True,
        )
        return await self.create_package(request)

    async def get_field_collection_summary(self) -> Dict[str, Any]:
        """Return a summary of all field data collection metrics.

        Returns:
            Dictionary with collection statistics.
        """
        return {
            "agent_id": _AGENT_ID,
            "version": _MODULE_VERSION,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "metrics": dict(self._metrics),
            "available_engines": self._count_available_engines(),
            "reference_data": {
                "templates": len(self._ref_templates or {}),
                "commodities": len(self._ref_commodities or {}),
                "languages": len(self._ref_languages or []),
            },
            "provenance_entries": self._provenance.entry_count,
            "provenance_chain_valid": self._provenance.verify_chain(),
        }

    async def health_check(self) -> HealthResponse:
        """Run a comprehensive health check across all subsystems.

        Returns:
            HealthResponse with status and connectivity info.
        """
        start = time.monotonic()

        db_ok = self._db_pool is not None
        redis_ok = self._redis is not None

        engine_count = self._count_available_engines()
        status = "healthy" if self._started else "starting"
        if engine_count == 0 and self._started:
            status = "degraded"

        return HealthResponse(
            status=status,
            version=VERSION,
            agent_id=_AGENT_ID,
            database_connected=db_ok,
            redis_connected=redis_ok,
            active_devices=self._metrics.get("devices_registered", 0),
            pending_sync_items=0,
            unresolved_conflicts=0,
            uptime_seconds=round(self.uptime_seconds, 1),
            timestamp=utcnow(),
        )

    # ------------------------------------------------------------------
    # Internal: engine call helpers
    # ------------------------------------------------------------------

    def _safe_engine_call(
        self, engine: Any, method_name: str, params: Dict[str, Any],
    ) -> Any:
        """Safely call an engine method with error handling.

        Args:
            engine: Engine instance.
            method_name: Method to call.
            params: Parameters to pass.

        Returns:
            Engine method result or None on error.
        """
        if engine is None:
            logger.debug(
                "Engine not available for method %s", method_name,
            )
            return None
        try:
            method = getattr(engine, method_name, None)
            if method is None:
                logger.warning(
                    "Engine %s has no method %s",
                    type(engine).__name__, method_name,
                )
                return None
            return method(**params)
        except Exception as exc:
            logger.error(
                "Engine call %s.%s failed: %s",
                type(engine).__name__, method_name, exc, exc_info=True,
            )
            return None

    def _record_provenance(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a provenance entry and return the chain hash.

        Args:
            entity_type: Provenance entity type.
            action: Action performed.
            entity_id: Entity identifier.
            data: Optional data to hash.

        Returns:
            SHA-256 chain hash string.
        """
        if not self._config.enable_provenance:
            return ""
        try:
            entry = self._provenance.record(
                entity_type=entity_type,
                action=action,
                entity_id=entity_id,
                data=data,
            )
            return entry.hash_value
        except Exception as exc:
            logger.warning(
                "Provenance recording failed: %s", exc,
            )
            return ""

    # ------------------------------------------------------------------
    # Internal: model builders
    # ------------------------------------------------------------------

    def _build_form_submission(self, request: SubmitFormRequest) -> FormSubmission:
        """Build a FormSubmission model from a SubmitFormRequest.

        Args:
            request: Inbound submission request.

        Returns:
            Populated FormSubmission model.
        """
        return FormSubmission(
            device_id=request.device_id,
            operator_id=request.operator_id,
            form_type=request.form_type,
            template_id=request.template_id,
            template_version=request.template_version,
            data=request.data,
            commodity_type=request.commodity_type,
            country_code=request.country_code,
            local_timestamp=request.local_timestamp,
            server_timestamp=utcnow(),
            gps_capture_ids=request.gps_capture_ids,
            photo_ids=request.photo_ids,
            signature_ids=request.signature_ids,
            metadata=request.metadata,
        )

    # ------------------------------------------------------------------
    # Internal: GPS accuracy classification
    # ------------------------------------------------------------------

    def _classify_accuracy(
        self,
        horizontal_accuracy_m: float,
        hdop: float,
        satellite_count: int,
    ) -> CaptureAccuracyTier:
        """Classify GPS capture accuracy into a tier.

        Args:
            horizontal_accuracy_m: Horizontal accuracy in meters.
            hdop: Horizontal Dilution of Precision.
            satellite_count: Number of satellites used.

        Returns:
            CaptureAccuracyTier classification.
        """
        if horizontal_accuracy_m < 1.0 and hdop < 1.0 and satellite_count >= 12:
            return CaptureAccuracyTier.EXCELLENT
        if horizontal_accuracy_m <= 3.0 and hdop <= 2.0 and satellite_count >= 8:
            return CaptureAccuracyTier.GOOD
        if horizontal_accuracy_m <= 5.0 and hdop <= 3.0 and satellite_count >= 6:
            return CaptureAccuracyTier.ACCEPTABLE
        if horizontal_accuracy_m <= 10.0 and hdop <= 5.0 and satellite_count >= 4:
            return CaptureAccuracyTier.POOR
        return CaptureAccuracyTier.REJECTED

    # ------------------------------------------------------------------
    # Internal: polygon area calculation
    # ------------------------------------------------------------------

    def _calculate_polygon_area(self, vertices: List[List[float]]) -> float:
        """Calculate polygon area in hectares using Shoelace formula.

        Applies a simple geodesic correction factor based on latitude
        for approximate area in hectares.

        Args:
            vertices: List of [latitude, longitude] pairs.

        Returns:
            Area in hectares (approximate).
        """
        import math

        if len(vertices) < 3:
            return 0.0

        # Shoelace formula in degrees, then convert
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][1] * vertices[j][0]
            area -= vertices[j][1] * vertices[i][0]
        area = abs(area) / 2.0

        # Convert from square degrees to hectares
        # 1 degree latitude ~ 111,320 meters
        # 1 degree longitude ~ 111,320 * cos(lat) meters
        avg_lat = sum(v[0] for v in vertices) / n
        lat_m = 111320.0
        lon_m = 111320.0 * math.cos(math.radians(avg_lat))
        area_m2 = area * lat_m * lon_m
        area_ha = area_m2 / 10000.0
        return round(area_ha, 4)

    # ------------------------------------------------------------------
    # Internal: startup helpers
    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        """Configure logging level from config."""
        log_level = getattr(logging, self._config.log_level, logging.INFO)
        logging.getLogger("greenlang.agents.eudr.mobile_data_collector").setLevel(log_level)

    def _init_tracer(self) -> None:
        """Initialize OpenTelemetry tracer if available."""
        if OTEL_AVAILABLE and otel_trace is not None:
            self._tracer = otel_trace.get_tracer(
                _AGENT_ID, _MODULE_VERSION,
            )
            logger.info("OpenTelemetry tracer initialized for %s", _AGENT_ID)

    def _load_reference_data(self) -> None:
        """Load reference data from the reference_data package."""
        self._ref_templates = dict(ALL_TEMPLATES)
        self._ref_commodities = dict(ALL_COMMODITIES)
        self._ref_languages = list_supported_languages()
        logger.info(
            "Reference data loaded: %d templates, %d commodities, %d languages",
            len(self._ref_templates),
            len(self._ref_commodities),
            len(self._ref_languages),
        )

    async def _connect_database(self) -> None:
        """Connect to PostgreSQL if psycopg is available."""
        if not PSYCOPG_POOL_AVAILABLE:
            logger.info("psycopg_pool not available; database skipped")
            return
        try:
            self._db_pool = await AsyncConnectionPool(
                conninfo=self._config.database_url,
                min_size=1,
                max_size=self._config.pool_size,
            ).__aenter__()
            logger.info("Database pool connected: pool_size=%d", self._config.pool_size)
        except Exception as exc:
            logger.warning("Database connection failed (non-fatal): %s", exc)
            self._db_pool = None

    async def _connect_redis(self) -> None:
        """Connect to Redis if redis.asyncio is available."""
        if not REDIS_AVAILABLE or aioredis is None:
            logger.info("redis.asyncio not available; Redis skipped")
            return
        try:
            self._redis = aioredis.from_url(
                self._config.redis_url, decode_responses=True,
            )
            await self._redis.ping()
            logger.info("Redis connected: %s", self._config.redis_url[:30])
        except Exception as exc:
            logger.warning("Redis connection failed (non-fatal): %s", exc)
            self._redis = None

    async def _close_database(self) -> None:
        """Close the database connection pool."""
        if self._db_pool is not None:
            try:
                await self._db_pool.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("Database pool close failed: %s", exc)
            self._db_pool = None

    async def _close_redis(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            try:
                await self._redis.close()
            except Exception as exc:
                logger.warning("Redis close failed: %s", exc)
            self._redis = None

    def _count_available_engines(self) -> int:
        """Count the number of available engine classes."""
        count = 0
        if _OFFLINE_FORM_ENGINE_AVAILABLE:
            count += 1
        if _GPS_CAPTURE_ENGINE_AVAILABLE:
            count += 1
        if _PHOTO_EVIDENCE_COLLECTOR_AVAILABLE:
            count += 1
        if _SYNC_ENGINE_AVAILABLE:
            count += 1
        if _FORM_TEMPLATE_MANAGER_AVAILABLE:
            count += 1
        if _DIGITAL_SIGNATURE_ENGINE_AVAILABLE:
            count += 1
        if _DATA_PACKAGE_BUILDER_AVAILABLE:
            count += 1
        if _DEVICE_FLEET_MANAGER_AVAILABLE:
            count += 1
        return count

# ===========================================================================
# Singleton accessor
# ===========================================================================

_service_lock = threading.Lock()
_service_instance: Optional[MobileDataCollectorService] = None

def get_service() -> MobileDataCollectorService:
    """Return the process-wide singleton MobileDataCollectorService.

    Creates the instance on first call. Thread-safe via double-checked
    locking.

    Returns:
        MobileDataCollectorService singleton.

    Example:
        >>> service = get_service()
        >>> service.is_running
        False
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = MobileDataCollectorService()
                logger.info("MobileDataCollectorService singleton created")
    return _service_instance

def reset_service() -> None:
    """Reset the singleton service instance.

    Intended for testing teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.info("MobileDataCollectorService singleton reset")

# ===========================================================================
# FastAPI lifespan integration
# ===========================================================================

@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for automatic startup/shutdown.

    Usage:
        >>> from fastapi import FastAPI
        >>> from greenlang.agents.eudr.mobile_data_collector.setup import lifespan
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance.

    Yields:
        None (service is started).
    """
    service = get_service()
    await service.startup()
    try:
        yield
    finally:
        await service.shutdown()

# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    "MobileDataCollectorService",
    "get_service",
    "reset_service",
    "lifespan",
]
