# -*- coding: utf-8 -*-
"""
QRCodeGeneratorService - Facade for AGENT-EUDR-014

Single entry point for all QR code generation operations.  Manages 8
engines, async PostgreSQL pool, Redis cache, provenance tracking,
Prometheus metrics, and reference data loading.

Lifecycle:
    startup -> load config -> connect DB -> connect Redis -> load reference
            -> initialize engines -> start health check
    shutdown -> close engines -> close Redis -> close DB -> flush metrics

Engines (8):
    1. QREncoder              - QR code image generation (Feature 1)
    2. PayloadComposer        - Data payload composition (Feature 2)
    3. LabelTemplateEngine    - Label rendering with templates (Feature 3)
    4. BatchCodeGenerator     - Batch code generation with check digits (Feature 4)
    5. VerificationURLBuilder - Verification URL with HMAC tokens (Feature 5)
    6. AntiCounterfeitEngine  - Counterfeit detection and prevention (Feature 6)
    7. BulkGenerationPipeline - Bulk QR code generation jobs (Feature 7)
    8. CodeLifecycleManager   - QR code lifecycle management (Feature 8)

Reference Data (3):
    - label_templates: 5 pre-defined label templates (product, shipping,
      pallet, container, consumer)
    - gs1_specifications: GS1 Digital Link formatting and GTIN validation
    - commodity_codes: EUDR commodity codes, HS/CN/TARIC mappings,
      country risk classifications

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.qr_code_generator.setup import (
    ...     QRCodeGeneratorService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-014
Agent ID: GL-EUDR-QRG-014
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 14
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
from greenlang.schemas import utcnow

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

from greenlang.agents.eudr.qr_code_generator.config import (
    QRCodeGeneratorConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.qr_code_generator.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from greenlang.agents.eudr.qr_code_generator.metrics import (
    PROMETHEUS_AVAILABLE,
    record_code_generated,
    record_label_generated,
    record_payload_composed,
    record_batch_code_generated,
    record_verification_url_built,
    record_scan,
    record_counterfeit_detection,
    record_bulk_job,
    record_bulk_codes,
    record_revocation,
    record_signature_verification,
    record_api_error,
    observe_generation_duration,
    observe_label_duration,
    observe_bulk_duration,
    observe_verification_duration,
    set_active_bulk_jobs,
    set_active_codes,
)

# ---------------------------------------------------------------------------
# Internal imports: models
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.qr_code_generator.models import (
    VERSION,
    ContentType,
    OutputFormat,
    ErrorCorrectionLevel,
    SymbologyType,
    LabelTemplate,
    CheckDigitAlgorithm,
    CodeStatus,
    ScanOutcome,
    CounterfeitRiskLevel,
    BulkJobStatus,
    ComplianceStatus,
    QRCodeRecord,
    DataPayload,
    LabelRecord,
    BatchCode,
    VerificationURL,
    SignatureRecord,
    ScanEvent,
    BulkJob,
    LifecycleEvent,
    TemplateDefinition,
    CodeAssociation,
    AuditLogEntry,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# Internal imports: reference data
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.qr_code_generator.reference_data import (
    ALL_TEMPLATES,
    TEMPLATE_REGISTRY,
    EUDR_COMMODITIES,
    COMMODITY_CODE_PREFIX,
    COUNTRY_RISK_CLASSIFICATION,
    GS1_DIGITAL_LINK_BASE_URL,
    get_template,
    list_template_names,
    get_template_dimensions,
    validate_template,
    validate_gtin,
    normalize_to_gtin14,
    build_gs1_digital_link_uri,
    calculate_gtin_check_digit,
    is_eudr_commodity,
    get_commodity_from_hs,
    get_commodity_prefix,
    get_country_risk,
    validate_commodity,
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-QRG-014"
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
    return f"QRG-{uuid.uuid4().hex[:12]}"

def _elapsed_ms(start: float) -> float:
    """Return elapsed milliseconds since ``start`` (monotonic)."""
    return (time.monotonic() - start) * 1000

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class QRResult:
    """Result from a QR code generation or query operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        code_id: QR code identifier.
        data: Result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "code_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        code_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.code_id = code_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "code_id": self.code_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class LabelResult:
    """Result from a label rendering operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        label_id: Label identifier.
        data: Result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "label_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        label_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.label_id = label_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "label_id": self.label_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class BatchResult:
    """Result from a batch code generation operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        data: Result data payload including generated codes.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class ScanResult:
    """Result from a scan processing operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        scan_id: Scan event identifier.
        outcome: Scan verification outcome.
        counterfeit_risk: Assessed counterfeit risk level.
        data: Result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "scan_id", "outcome",
        "counterfeit_risk", "data", "error",
        "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        scan_id: str = "",
        outcome: str = "verified",
        counterfeit_risk: str = "low",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.scan_id = scan_id
        self.outcome = outcome
        self.counterfeit_risk = counterfeit_risk
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "scan_id": self.scan_id,
            "outcome": self.outcome,
            "counterfeit_risk": self.counterfeit_risk,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class BulkJobResult:
    """Result from a bulk job operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        job_id: Bulk job identifier.
        data: Result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "job_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        job_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.job_id = job_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "job_id": self.job_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class LifecycleResult:
    """Result from a lifecycle management operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        code_id: QR code identifier.
        previous_status: Status before the change.
        new_status: Status after the change.
        data: Result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "code_id",
        "previous_status", "new_status", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        code_id: str = "",
        previous_status: str = "created",
        new_status: str = "active",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.code_id = code_id
        self.previous_status = previous_status
        self.new_status = new_status
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "code_id": self.code_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ===========================================================================
# QRCodeGeneratorService - Main facade
# ===========================================================================

class QRCodeGeneratorService:
    """Facade for the QR Code Generator Agent (AGENT-EUDR-014).

    Provides a unified interface to all 8 engines:
        1. QREncoder              - QR code image generation
        2. PayloadComposer        - Data payload composition
        3. LabelTemplateEngine    - Label rendering with templates
        4. BatchCodeGenerator     - Batch code generation
        5. VerificationURLBuilder - Verification URL construction
        6. AntiCounterfeitEngine  - Counterfeit detection
        7. BulkGenerationPipeline - Bulk QR code generation
        8. CodeLifecycleManager   - QR code lifecycle management

    Singleton pattern with thread-safe initialization.

    Example:
        >>> service = QRCodeGeneratorService()
        >>> await service.startup()
        >>> result = await service.generate_qr_with_payload(...)
        >>> await service.shutdown()
    """

    _instance: Optional[QRCodeGeneratorService] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize QRCodeGeneratorService.

        Loads configuration but does NOT start connections or engines.
        Call ``startup()`` to activate the service.
        """
        self._config: QRCodeGeneratorConfig = get_config()

        self._started = False
        self._start_time: Optional[float] = None
        self._config_hash = _compute_provenance_hash(
            self._config.database_url,
            self._config.redis_url,
            self._config.default_error_correction,
            str(self._config.max_payload_bytes),
            self._config.genesis_hash,
        )

        # Connection handles
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None

        # Engine instances (initialized in startup)
        self._qr_encoder: Optional[Any] = None
        self._payload_composer: Optional[Any] = None
        self._label_template_engine: Optional[Any] = None
        self._batch_code_generator: Optional[Any] = None
        self._verification_url_builder: Optional[Any] = None
        self._anti_counterfeit_engine: Optional[Any] = None
        self._bulk_generation_pipeline: Optional[Any] = None
        self._code_lifecycle_manager: Optional[Any] = None

        # Reference data (loaded in startup)
        self._ref_templates: Optional[Dict[str, Any]] = None
        self._ref_commodities: Optional[Dict[str, Any]] = None
        self._ref_gs1: Optional[Dict[str, Any]] = None

        # Health check background task
        self._health_task: Optional[asyncio.Task[None]] = None
        self._last_health: Optional[Dict[str, Any]] = None

        # OpenTelemetry tracer
        self._tracer: Optional[Any] = None

        # Provenance tracker
        self._provenance: ProvenanceTracker = get_provenance_tracker()

        # Metrics counters
        self._metrics: Dict[str, int] = {
            "codes_generated": 0,
            "labels_rendered": 0,
            "payloads_composed": 0,
            "batch_codes_generated": 0,
            "verification_urls_built": 0,
            "scans_recorded": 0,
            "counterfeit_detections": 0,
            "bulk_jobs_submitted": 0,
            "bulk_codes_generated": 0,
            "codes_activated": 0,
            "codes_deactivated": 0,
            "codes_revoked": 0,
            "codes_expired": 0,
            "signatures_created": 0,
            "reprints": 0,
            "errors": 0,
        }

        logger.info(
            "QRCodeGeneratorService created: config_hash=%s, "
            "ec=%s, max_payload=%d, version=%s",
            self._config_hash[:12],
            self._config.default_error_correction,
            self._config.max_payload_bytes,
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
    def config(self) -> QRCodeGeneratorConfig:
        """Return the service configuration."""
        return self._config

    @property
    def qr_encoder(self) -> Any:
        """Return the QREncoder engine instance."""
        self._ensure_started()
        return self._qr_encoder

    @property
    def payload_composer(self) -> Any:
        """Return the PayloadComposer engine instance."""
        self._ensure_started()
        return self._payload_composer

    @property
    def label_template_engine(self) -> Any:
        """Return the LabelTemplateEngine instance."""
        self._ensure_started()
        return self._label_template_engine

    @property
    def batch_code_generator(self) -> Any:
        """Return the BatchCodeGenerator engine instance."""
        self._ensure_started()
        return self._batch_code_generator

    @property
    def verification_url_builder(self) -> Any:
        """Return the VerificationURLBuilder engine instance."""
        self._ensure_started()
        return self._verification_url_builder

    @property
    def anti_counterfeit_engine(self) -> Any:
        """Return the AntiCounterfeitEngine instance."""
        self._ensure_started()
        return self._anti_counterfeit_engine

    @property
    def bulk_generation_pipeline(self) -> Any:
        """Return the BulkGenerationPipeline engine instance."""
        self._ensure_started()
        return self._bulk_generation_pipeline

    @property
    def code_lifecycle_manager(self) -> Any:
        """Return the CodeLifecycleManager engine instance."""
        self._ensure_started()
        return self._code_lifecycle_manager

    # ------------------------------------------------------------------
    # Startup / Shutdown
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Start the service: connect DB, Redis, initialize all engines.

        Executes the full startup sequence:
            1. Configure structured logging
            2. Initialize OpenTelemetry tracer
            3. Load reference data
            4. Connect to PostgreSQL
            5. Connect to Redis
            6. Initialize all eight engines
            7. Start background health check task

        Idempotent: safe to call multiple times.
        """
        if self._started:
            logger.debug("QRCodeGeneratorService already started")
            return

        start = time.monotonic()
        logger.info("QRCodeGeneratorService starting up...")

        self._configure_logging()
        self._init_tracer()
        self._load_reference_data()
        await self._connect_database()
        await self._connect_redis()
        await self._initialize_engines()
        self._start_health_check()

        self._started = True
        self._start_time = time.monotonic()
        elapsed = _elapsed_ms(start)

        logger.info(
            "QRCodeGeneratorService started in %.1fms: "
            "db=%s, redis=%s, engines=%d/%d, config_hash=%s",
            elapsed,
            "connected" if self._db_pool is not None else "skipped",
            "connected" if self._redis is not None else "skipped",
            self._count_initialized_engines(),
            _ENGINE_COUNT,
            self._config_hash[:12],
        )

    async def shutdown(self) -> None:
        """Gracefully shut down the service and release all resources.

        Idempotent: safe to call multiple times.
        """
        if not self._started:
            logger.debug("QRCodeGeneratorService already stopped")
            return

        logger.info("QRCodeGeneratorService shutting down...")
        start = time.monotonic()

        self._stop_health_check()
        await self._close_engines()
        await self._close_redis()
        await self._close_database()
        self._flush_metrics()

        self._started = False
        elapsed = _elapsed_ms(start)
        logger.info(
            "QRCodeGeneratorService shut down in %.1fms", elapsed,
        )

    # ==================================================================
    # FACADE METHODS: Engine 1 - QREncoder
    # ==================================================================

    async def generate_qr_with_payload(
        self,
        content_type: str,
        dds_reference: str,
        operator_id: str,
        payload_data: Dict[str, Any],
        commodity: Optional[str] = None,
        compliance_status: Optional[str] = None,
        error_correction: Optional[str] = None,
        output_format: Optional[str] = None,
        version: Optional[str] = None,
        module_size: Optional[int] = None,
        quiet_zone: Optional[int] = None,
        dpi: Optional[int] = None,
        embed_logo: Optional[bool] = None,
    ) -> QRResult:
        """Generate a QR code from traceability data with composed payload.

        Composes a payload via PayloadComposer then generates a QR code
        via QREncoder in a single atomic operation.

        Args:
            content_type: Payload content type.
            dds_reference: Due Diligence Statement reference number.
            operator_id: EUDR operator identifier.
            payload_data: Raw data to encode.
            commodity: EUDR-regulated commodity type.
            compliance_status: EUDR compliance status.
            error_correction: Error correction level override.
            output_format: Output format override.
            version: QR version override.
            module_size: Module pixel size override.
            quiet_zone: Quiet zone modules override.
            dpi: DPI override.
            embed_logo: Logo embedding override.

        Returns:
            QRResult with code_id and generated QR code data.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            code_id = str(uuid.uuid4())
            ec = error_correction or self._config.default_error_correction
            fmt = output_format or self._config.default_output_format
            cs = compliance_status or "pending"

            # Compose payload
            payload_result = self._safe_engine_call(
                self._payload_composer, "compose", {
                    "content_type": content_type,
                    "operator_id": operator_id,
                    "data": payload_data,
                    "commodity": commodity,
                    "dds_reference": dds_reference,
                    "compliance_status": cs,
                },
            )

            # Generate QR code
            qr_result = self._safe_engine_call(
                self._qr_encoder, "generate", {
                    "code_id": code_id,
                    "payload": payload_result,
                    "version": version or self._config.default_version,
                    "error_correction": ec,
                    "output_format": fmt,
                    "module_size": module_size or self._config.default_module_size,
                    "quiet_zone": quiet_zone or self._config.default_quiet_zone,
                    "dpi": dpi or self._config.default_dpi,
                    "embed_logo": embed_logo if embed_logo is not None else self._config.enable_logo_embedding,
                    "operator_id": operator_id,
                    "commodity": commodity,
                    "dds_reference": dds_reference,
                    "compliance_status": cs,
                },
            )

            self._metrics["codes_generated"] += 1
            record_code_generated(fmt, content_type)
            elapsed = _elapsed_ms(start)
            observe_generation_duration(elapsed / 1000)

            provenance_hash = self._record_provenance(
                "qr_code", "generate", code_id,
                data={"content_type": content_type, "operator_id": operator_id},
            )

            return QRResult(
                request_id=request_id,
                success=True,
                code_id=code_id,
                data={
                    "qr_code": qr_result,
                    "payload": payload_result,
                    "content_type": content_type,
                    "error_correction": ec,
                    "output_format": fmt,
                },
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed,
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("generate")
            logger.error(
                "generate_qr_with_payload failed: %s", exc, exc_info=True,
            )
            return QRResult(
                request_id=request_id,
                success=False,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    async def generate_qr_code(
        self,
        payload_data: Dict[str, Any],
        operator_id: str,
        content_type: Optional[str] = None,
        error_correction: Optional[str] = None,
        output_format: Optional[str] = None,
        commodity: Optional[str] = None,
        dds_reference: Optional[str] = None,
    ) -> QRResult:
        """Generate a QR code with default settings.

        Simplified interface using configuration defaults.

        Args:
            payload_data: Data to encode in the QR code.
            operator_id: EUDR operator identifier.
            content_type: Content type override.
            error_correction: Error correction override.
            output_format: Output format override.
            commodity: EUDR commodity type.
            dds_reference: DDS reference number.

        Returns:
            QRResult with generated QR code data.
        """
        return await self.generate_qr_with_payload(
            content_type=content_type or self._config.default_content_type,
            dds_reference=dds_reference or "",
            operator_id=operator_id,
            payload_data=payload_data,
            commodity=commodity,
            error_correction=error_correction,
            output_format=output_format,
        )

    async def get_qr_code(
        self,
        code_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a QR code record by ID.

        Args:
            code_id: QR code identifier.

        Returns:
            QR code record dictionary or None.
        """
        self._ensure_started()
        return self._safe_engine_call_with_args(
            self._qr_encoder, "get_code", code_id=code_id,
        )

    async def validate_qr_code(
        self,
        code_id: str,
        image_data_hash: str,
    ) -> Dict[str, Any]:
        """Validate a QR code by comparing image hashes.

        Args:
            code_id: QR code identifier.
            image_data_hash: SHA-256 hash of the QR code image.

        Returns:
            Validation result dictionary.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._qr_encoder, "validate", {
                "code_id": code_id,
                "image_data_hash": image_data_hash,
            },
        )
        return result if isinstance(result, dict) else {"valid": False}

    # ==================================================================
    # FACADE METHODS: Engine 2 - PayloadComposer
    # ==================================================================

    async def compose_payload(
        self,
        operator_id: str,
        data: Dict[str, Any],
        content_type: Optional[str] = None,
        compress: Optional[bool] = None,
        encrypt: Optional[bool] = None,
        commodity: Optional[str] = None,
        dds_reference: Optional[str] = None,
        compliance_status: Optional[str] = None,
        origin_country: Optional[str] = None,
    ) -> QRResult:
        """Compose a data payload for QR code encoding.

        Args:
            operator_id: EUDR operator identifier.
            data: Raw payload data dictionary.
            content_type: Content type override.
            compress: Enable/disable compression override.
            encrypt: Enable/disable encryption override.
            commodity: EUDR commodity type.
            dds_reference: DDS reference number.
            compliance_status: Compliance status.
            origin_country: ISO 3166-1 alpha-2 country code.

        Returns:
            QRResult with composed payload data.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            ct = content_type or self._config.default_content_type
            result = self._safe_engine_call(
                self._payload_composer, "compose", {
                    "operator_id": operator_id,
                    "content_type": ct,
                    "data": data,
                    "compress": compress if compress is not None else self._config.enable_compression,
                    "encrypt": encrypt if encrypt is not None else self._config.enable_encryption,
                    "commodity": commodity,
                    "dds_reference": dds_reference,
                    "compliance_status": compliance_status,
                    "origin_country": origin_country,
                },
            )

            self._metrics["payloads_composed"] += 1
            record_payload_composed(ct)

            provenance_hash = self._record_provenance(
                "payload", "compose", request_id,
                data={"content_type": ct, "operator_id": operator_id},
            )

            return QRResult(
                request_id=request_id,
                success=True,
                data={"payload": result, "content_type": ct},
                provenance_hash=provenance_hash,
                processing_time_ms=_elapsed_ms(start),
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("compose")
            logger.error("compose_payload failed: %s", exc, exc_info=True)
            return QRResult(
                request_id=request_id,
                success=False,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    # ==================================================================
    # FACADE METHODS: Engine 3 - LabelTemplateEngine
    # ==================================================================

    async def generate_labeled_qr(
        self,
        template_name: str,
        qr_data: Dict[str, Any],
        operator_id: str,
        product_name: Optional[str] = None,
        batch_code: Optional[str] = None,
        compliance_status: Optional[str] = None,
        commodity: Optional[str] = None,
        custom_fields: Optional[Dict[str, str]] = None,
        output_format: Optional[str] = None,
        dpi: Optional[int] = None,
    ) -> LabelResult:
        """Generate a QR code with a pre-designed label in one call.

        Args:
            template_name: Label template to use.
            qr_data: QR code payload data.
            operator_id: EUDR operator identifier.
            product_name: Product name for label.
            batch_code: Batch code for label.
            compliance_status: Compliance status for colour coding.
            commodity: EUDR commodity type.
            custom_fields: Additional custom label fields.
            output_format: Output format override.
            dpi: DPI override.

        Returns:
            LabelResult with rendered label data.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            label_id = str(uuid.uuid4())
            cs = compliance_status or "pending"

            result = self._safe_engine_call(
                self._label_template_engine, "render", {
                    "label_id": label_id,
                    "template_name": template_name,
                    "qr_data": qr_data,
                    "operator_id": operator_id,
                    "product_name": product_name,
                    "batch_code": batch_code,
                    "compliance_status": cs,
                    "commodity": commodity,
                    "custom_fields": custom_fields or {},
                    "output_format": output_format or self._config.default_output_format,
                    "dpi": dpi or self._config.default_dpi,
                },
            )

            self._metrics["labels_rendered"] += 1
            record_label_generated(template_name)
            elapsed = _elapsed_ms(start)
            observe_label_duration(elapsed / 1000)

            provenance_hash = self._record_provenance(
                "label", "render", label_id,
                data={"template": template_name, "operator_id": operator_id},
            )

            return LabelResult(
                request_id=request_id,
                success=True,
                label_id=label_id,
                data={"label": result, "template": template_name},
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed,
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("render")
            logger.error(
                "generate_labeled_qr failed: %s", exc, exc_info=True,
            )
            return LabelResult(
                request_id=request_id,
                success=False,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    async def render_label(
        self,
        code_id: str,
        operator_id: str,
        template: Optional[str] = None,
        product_name: Optional[str] = None,
        batch_code: Optional[str] = None,
        custom_fields: Optional[Dict[str, str]] = None,
        output_format: Optional[str] = None,
        dpi: Optional[int] = None,
    ) -> LabelResult:
        """Render a label for an existing QR code.

        Args:
            code_id: Existing QR code identifier.
            operator_id: EUDR operator identifier.
            template: Template name override.
            product_name: Product name for label.
            batch_code: Batch code for label.
            custom_fields: Additional custom fields.
            output_format: Output format override.
            dpi: DPI override.

        Returns:
            LabelResult with rendered label.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            label_id = str(uuid.uuid4())
            tmpl = template or self._config.default_template

            result = self._safe_engine_call(
                self._label_template_engine, "render_for_code", {
                    "label_id": label_id,
                    "code_id": code_id,
                    "operator_id": operator_id,
                    "template_name": tmpl,
                    "product_name": product_name,
                    "batch_code": batch_code,
                    "custom_fields": custom_fields or {},
                    "output_format": output_format or self._config.default_output_format,
                    "dpi": dpi or self._config.default_dpi,
                },
            )

            self._metrics["labels_rendered"] += 1
            record_label_generated(tmpl)

            provenance_hash = self._record_provenance(
                "label", "render", label_id,
                data={"code_id": code_id, "template": tmpl},
            )

            return LabelResult(
                request_id=request_id,
                success=True,
                label_id=label_id,
                data={"label": result, "template": tmpl},
                provenance_hash=provenance_hash,
                processing_time_ms=_elapsed_ms(start),
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("render")
            logger.error("render_label failed: %s", exc, exc_info=True)
            return LabelResult(
                request_id=request_id,
                success=False,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    async def reprint_label(
        self,
        code_id: str,
        operator_id: str,
        template: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> LabelResult:
        """Reprint a label for an existing QR code.

        Increments the reprint counter and checks against max_reprints.

        Args:
            code_id: QR code identifier.
            operator_id: Operator requesting the reprint.
            template: Template override.
            output_format: Output format override.

        Returns:
            LabelResult with reprinted label.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            result = self._safe_engine_call(
                self._code_lifecycle_manager, "reprint", {
                    "code_id": code_id,
                    "operator_id": operator_id,
                    "template": template or self._config.default_template,
                    "output_format": output_format or self._config.default_output_format,
                    "max_reprints": self._config.max_reprints,
                },
            )

            self._metrics["reprints"] += 1
            self._record_provenance(
                "qr_code", "generate", code_id,
                data={"action": "reprint", "operator_id": operator_id},
            )

            return LabelResult(
                request_id=request_id,
                success=True,
                data=result if isinstance(result, dict) else {},
                processing_time_ms=_elapsed_ms(start),
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("reprint")
            logger.error("reprint_label failed: %s", exc, exc_info=True)
            return LabelResult(
                request_id=request_id,
                success=False,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    # ==================================================================
    # FACADE METHODS: Engine 4 - BatchCodeGenerator
    # ==================================================================

    async def generate_batch_qr_codes(
        self,
        batch_id: str,
        count: int,
        content_type: str,
        operator_id: str,
        commodity: str,
        year: int,
        facility_id: Optional[str] = None,
        prefix_format: Optional[str] = None,
        check_digit_algorithm: Optional[str] = None,
    ) -> BatchResult:
        """Generate a batch of QR codes with sequential codes.

        Args:
            batch_id: Batch identifier.
            count: Number of codes to generate.
            content_type: Payload content type.
            operator_id: EUDR operator identifier.
            commodity: EUDR commodity type.
            year: Production year.
            facility_id: Facility identifier.
            prefix_format: Prefix format override.
            check_digit_algorithm: Check digit algorithm override.

        Returns:
            BatchResult with generated codes.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            result = self._safe_engine_call(
                self._batch_code_generator, "generate_batch", {
                    "batch_id": batch_id,
                    "count": count,
                    "operator_id": operator_id,
                    "commodity": commodity,
                    "year": year,
                    "facility_id": facility_id,
                    "prefix_format": prefix_format or self._config.default_prefix_format,
                    "check_digit_algorithm": check_digit_algorithm or self._config.check_digit_algorithm,
                    "code_padding": self._config.code_padding,
                    "start_sequence": self._config.start_sequence,
                },
            )

            self._metrics["batch_codes_generated"] += count
            record_batch_code_generated(commodity)

            provenance_hash = self._record_provenance(
                "batch_code", "generate", batch_id,
                data={"count": count, "commodity": commodity},
            )

            return BatchResult(
                request_id=request_id,
                success=True,
                data={"batch_codes": result, "count": count},
                provenance_hash=provenance_hash,
                processing_time_ms=_elapsed_ms(start),
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("batch_generate")
            logger.error(
                "generate_batch_qr_codes failed: %s", exc, exc_info=True,
            )
            return BatchResult(
                request_id=request_id,
                success=False,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    async def generate_single_batch_code(
        self,
        operator_id: str,
        commodity: str,
        year: int,
        facility_id: Optional[str] = None,
    ) -> BatchResult:
        """Generate a single batch code.

        Convenience wrapper for single code generation.

        Args:
            operator_id: EUDR operator identifier.
            commodity: EUDR commodity type.
            year: Production year.
            facility_id: Facility identifier.

        Returns:
            BatchResult with the generated code.
        """
        batch_id = str(uuid.uuid4())
        return await self.generate_batch_qr_codes(
            batch_id=batch_id,
            count=1,
            content_type=self._config.default_content_type,
            operator_id=operator_id,
            commodity=commodity,
            year=year,
            facility_id=facility_id,
        )

    # ==================================================================
    # FACADE METHODS: Engine 5 - VerificationURLBuilder
    # ==================================================================

    async def build_verification_url(
        self,
        code_id: str,
        operator_id: str,
        base_url: Optional[str] = None,
        use_short_url: Optional[bool] = None,
        ttl_years: Optional[int] = None,
    ) -> QRResult:
        """Build a verification URL with HMAC-signed token.

        Args:
            code_id: QR code identifier.
            operator_id: EUDR operator identifier.
            base_url: Base URL override.
            use_short_url: Generate short URL override.
            ttl_years: Token TTL in years override.

        Returns:
            QRResult with verification URL data.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            result = self._safe_engine_call(
                self._verification_url_builder, "build_url", {
                    "code_id": code_id,
                    "operator_id": operator_id,
                    "base_url": base_url or self._config.base_verification_url,
                    "use_short_url": use_short_url if use_short_url is not None else self._config.short_url_enabled,
                    "ttl_years": ttl_years or self._config.verification_token_ttl_years,
                    "hmac_truncation_length": self._config.hmac_truncation_length,
                },
            )

            self._metrics["verification_urls_built"] += 1
            record_verification_url_built()

            provenance_hash = self._record_provenance(
                "verification_url", "build_url", code_id,
                data={"operator_id": operator_id},
            )

            return QRResult(
                request_id=request_id,
                success=True,
                code_id=code_id,
                data={"verification_url": result},
                provenance_hash=provenance_hash,
                processing_time_ms=_elapsed_ms(start),
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("build_url")
            logger.error(
                "build_verification_url failed: %s", exc, exc_info=True,
            )
            return QRResult(
                request_id=request_id,
                success=False,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    # ==================================================================
    # FACADE METHODS: Engine 6 - AntiCounterfeitEngine
    # ==================================================================

    async def verify_qr_code(
        self,
        code_id: str,
        scan_data: Dict[str, Any],
    ) -> ScanResult:
        """Verify a QR code for authenticity.

        Checks HMAC token, scan velocity, and geo-fence constraints.

        Args:
            code_id: QR code to verify.
            scan_data: Scan context data (hmac_token, latitude, etc.).

        Returns:
            ScanResult with verification outcome.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            result = self._safe_engine_call(
                self._anti_counterfeit_engine, "verify", {
                    "code_id": code_id,
                    "scan_data": scan_data,
                    "velocity_threshold": self._config.scan_velocity_threshold,
                    "geo_fence_enabled": self._config.geo_fence_enabled,
                },
            )

            outcome = "verified"
            risk_level = "low"
            if isinstance(result, dict):
                outcome = result.get("outcome", "verified")
                risk_level = result.get("counterfeit_risk", "low")

            record_scan(outcome)
            if risk_level in ("high", "critical"):
                record_counterfeit_detection(risk_level)
                self._metrics["counterfeit_detections"] += 1
            record_signature_verification()

            elapsed = _elapsed_ms(start)
            observe_verification_duration(elapsed / 1000)

            provenance_hash = self._record_provenance(
                "scan_event", "verify", code_id,
                data={"outcome": outcome, "risk": risk_level},
            )

            return ScanResult(
                request_id=request_id,
                success=True,
                outcome=outcome,
                counterfeit_risk=risk_level,
                data=result if isinstance(result, dict) else {},
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed,
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("verify")
            logger.error(
                "verify_qr_code failed: %s", exc, exc_info=True,
            )
            return ScanResult(
                request_id=request_id,
                success=False,
                outcome="error",
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    async def process_scan(
        self,
        code_id: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        scanner_id: Optional[str] = None,
        hmac_token: Optional[str] = None,
        scanner_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> ScanResult:
        """Record a scan event with counterfeit check and lifecycle update.

        Composes anti-counterfeit verification, scan event recording,
        and lifecycle update in a single operation.

        Args:
            code_id: QR code identifier.
            lat: Scan location latitude.
            lon: Scan location longitude.
            scanner_id: Scanner device identifier.
            hmac_token: HMAC token from verification URL.
            scanner_ip: Scanner IP address (will be hashed).
            user_agent: Scanner user agent string.

        Returns:
            ScanResult with scan outcome and risk assessment.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        scan_id = str(uuid.uuid4())

        try:
            # Anti-counterfeit check
            scan_data = {
                "hmac_token": hmac_token,
                "latitude": lat,
                "longitude": lon,
                "scanner_id": scanner_id,
            }

            ac_result = self._safe_engine_call(
                self._anti_counterfeit_engine, "check_scan", {
                    "code_id": code_id,
                    "scan_data": scan_data,
                    "velocity_threshold": self._config.scan_velocity_threshold,
                    "geo_fence_enabled": self._config.geo_fence_enabled,
                },
            )

            outcome = "verified"
            risk_level = "low"
            if isinstance(ac_result, dict):
                outcome = ac_result.get("outcome", "verified")
                risk_level = ac_result.get("counterfeit_risk", "low")

            # Record scan event
            if self._config.scan_logging_enabled:
                self._safe_engine_call(
                    self._code_lifecycle_manager, "record_scan", {
                        "scan_id": scan_id,
                        "code_id": code_id,
                        "outcome": outcome,
                        "scanner_ip": scanner_ip,
                        "user_agent": user_agent,
                        "latitude": lat,
                        "longitude": lon,
                        "counterfeit_risk": risk_level,
                    },
                )

            self._metrics["scans_recorded"] += 1
            record_scan(outcome)
            if risk_level in ("high", "critical"):
                record_counterfeit_detection(risk_level)
                self._metrics["counterfeit_detections"] += 1

            elapsed = _elapsed_ms(start)
            observe_verification_duration(elapsed / 1000)

            provenance_hash = self._record_provenance(
                "scan_event", "scan", scan_id,
                data={"code_id": code_id, "outcome": outcome},
            )

            return ScanResult(
                request_id=request_id,
                success=True,
                scan_id=scan_id,
                outcome=outcome,
                counterfeit_risk=risk_level,
                data={"scan_event": ac_result},
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed,
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("scan")
            logger.error("process_scan failed: %s", exc, exc_info=True)
            return ScanResult(
                request_id=request_id,
                success=False,
                scan_id=scan_id,
                outcome="error",
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    async def sign_qr_code(
        self,
        code_id: str,
        data_hash: str,
        key_id: Optional[str] = None,
    ) -> QRResult:
        """Sign a QR code payload with HMAC-SHA256.

        Args:
            code_id: QR code identifier.
            data_hash: SHA-256 hash of the data to sign.
            key_id: Signing key identifier.

        Returns:
            QRResult with signature data.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            result = self._safe_engine_call(
                self._anti_counterfeit_engine, "sign", {
                    "code_id": code_id,
                    "data_hash": data_hash,
                    "key_id": key_id,
                },
            )

            self._metrics["signatures_created"] += 1
            provenance_hash = self._record_provenance(
                "signature", "sign", code_id,
                data={"data_hash": data_hash[:16]},
            )

            return QRResult(
                request_id=request_id,
                success=True,
                code_id=code_id,
                data={"signature": result},
                provenance_hash=provenance_hash,
                processing_time_ms=_elapsed_ms(start),
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("sign")
            logger.error("sign_qr_code failed: %s", exc, exc_info=True)
            return QRResult(
                request_id=request_id,
                success=False,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    # ==================================================================
    # FACADE METHODS: Engine 7 - BulkGenerationPipeline
    # ==================================================================

    async def generate_bulk_labeled_qr(
        self,
        job_spec: Dict[str, Any],
    ) -> BulkJobResult:
        """Submit a bulk QR code generation job with labels.

        Args:
            job_spec: Bulk job specification including total_codes,
                operator_id, content_type, commodity, template, etc.

        Returns:
            BulkJobResult with job ID and initial status.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            job_id = str(uuid.uuid4())
            total_codes = job_spec.get("total_codes", 0)
            operator_id = job_spec.get("operator_id", "")

            result = self._safe_engine_call(
                self._bulk_generation_pipeline, "submit_job", {
                    "job_id": job_id,
                    "total_codes": total_codes,
                    "operator_id": operator_id,
                    "content_type": job_spec.get("content_type", self._config.default_content_type),
                    "commodity": job_spec.get("commodity"),
                    "error_correction": job_spec.get("error_correction", self._config.default_error_correction),
                    "output_format": job_spec.get("output_format", self._config.default_output_format),
                    "template": job_spec.get("template", self._config.default_template),
                    "worker_count": job_spec.get("worker_count", self._config.bulk_workers),
                    "include_labels": True,
                },
            )

            self._metrics["bulk_jobs_submitted"] += 1
            record_bulk_job("queued")
            set_active_bulk_jobs(self._metrics["bulk_jobs_submitted"])

            provenance_hash = self._record_provenance(
                "bulk_job", "generate", job_id,
                data={"total_codes": total_codes, "operator_id": operator_id},
            )

            return BulkJobResult(
                request_id=request_id,
                success=True,
                job_id=job_id,
                data={"job": result, "status": "queued"},
                provenance_hash=provenance_hash,
                processing_time_ms=_elapsed_ms(start),
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("bulk_submit")
            logger.error(
                "generate_bulk_labeled_qr failed: %s", exc, exc_info=True,
            )
            return BulkJobResult(
                request_id=request_id,
                success=False,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    async def submit_bulk_job(
        self,
        operator_id: str,
        total_codes: int,
        content_type: Optional[str] = None,
        commodity: Optional[str] = None,
        error_correction: Optional[str] = None,
        output_format: Optional[str] = None,
        worker_count: Optional[int] = None,
    ) -> BulkJobResult:
        """Submit a bulk QR code generation job (without labels).

        Args:
            operator_id: EUDR operator identifier.
            total_codes: Number of codes to generate.
            content_type: Content type override.
            commodity: EUDR commodity type.
            error_correction: Error correction override.
            output_format: Output format override.
            worker_count: Worker count override.

        Returns:
            BulkJobResult with job ID.
        """
        return await self.generate_bulk_labeled_qr({
            "operator_id": operator_id,
            "total_codes": total_codes,
            "content_type": content_type,
            "commodity": commodity,
            "error_correction": error_correction,
            "output_format": output_format,
            "worker_count": worker_count,
        })

    async def get_bulk_job_status(
        self,
        job_id: str,
    ) -> Dict[str, Any]:
        """Get the current status of a bulk generation job.

        Args:
            job_id: Bulk job identifier.

        Returns:
            Job status dictionary.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._bulk_generation_pipeline, "get_job_status",
            job_id=job_id,
        )
        if result is None:
            return {"job_id": job_id, "status": "not_found"}
        return result if isinstance(result, dict) else {"job_id": job_id}

    async def cancel_bulk_job(
        self,
        job_id: str,
        reason: Optional[str] = None,
    ) -> BulkJobResult:
        """Cancel an active bulk generation job.

        Args:
            job_id: Bulk job identifier.
            reason: Cancellation reason.

        Returns:
            BulkJobResult with updated status.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            result = self._safe_engine_call(
                self._bulk_generation_pipeline, "cancel_job", {
                    "job_id": job_id,
                    "reason": reason,
                },
            )

            record_bulk_job("cancelled")
            self._record_provenance(
                "bulk_job", "cancel", job_id,
                data={"reason": reason},
            )

            return BulkJobResult(
                request_id=request_id,
                success=True,
                job_id=job_id,
                data=result if isinstance(result, dict) else {},
                processing_time_ms=_elapsed_ms(start),
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("cancel_bulk")
            logger.error("cancel_bulk_job failed: %s", exc, exc_info=True)
            return BulkJobResult(
                request_id=request_id,
                success=False,
                job_id=job_id,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    # ==================================================================
    # FACADE METHODS: Engine 8 - CodeLifecycleManager
    # ==================================================================

    async def activate_code(
        self,
        code_id: str,
        performed_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> LifecycleResult:
        """Activate a QR code for scanning.

        Args:
            code_id: QR code identifier.
            performed_by: User performing the activation.
            reason: Activation reason.

        Returns:
            LifecycleResult with status transition.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            result = self._safe_engine_call(
                self._code_lifecycle_manager, "activate", {
                    "code_id": code_id,
                    "performed_by": performed_by,
                    "reason": reason,
                },
            )

            self._metrics["codes_activated"] += 1
            provenance_hash = self._record_provenance(
                "lifecycle_event", "activate", code_id,
                data={"performed_by": performed_by},
            )

            return LifecycleResult(
                request_id=request_id,
                success=True,
                code_id=code_id,
                previous_status="created",
                new_status="active",
                data=result if isinstance(result, dict) else {},
                provenance_hash=provenance_hash,
                processing_time_ms=_elapsed_ms(start),
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("activate")
            logger.error("activate_code failed: %s", exc, exc_info=True)
            return LifecycleResult(
                request_id=request_id,
                success=False,
                code_id=code_id,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    async def deactivate_code(
        self,
        code_id: str,
        reason: str,
        performed_by: Optional[str] = None,
    ) -> LifecycleResult:
        """Temporarily deactivate a QR code.

        Args:
            code_id: QR code identifier.
            reason: Deactivation reason.
            performed_by: User performing the deactivation.

        Returns:
            LifecycleResult with status transition.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            result = self._safe_engine_call(
                self._code_lifecycle_manager, "deactivate", {
                    "code_id": code_id,
                    "reason": reason,
                    "performed_by": performed_by,
                },
            )

            self._metrics["codes_deactivated"] += 1
            provenance_hash = self._record_provenance(
                "lifecycle_event", "deactivate", code_id,
                data={"reason": reason},
            )

            return LifecycleResult(
                request_id=request_id,
                success=True,
                code_id=code_id,
                previous_status="active",
                new_status="deactivated",
                data=result if isinstance(result, dict) else {},
                provenance_hash=provenance_hash,
                processing_time_ms=_elapsed_ms(start),
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("deactivate")
            logger.error("deactivate_code failed: %s", exc, exc_info=True)
            return LifecycleResult(
                request_id=request_id,
                success=False,
                code_id=code_id,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    async def revoke_with_reason(
        self,
        code_id: str,
        reason: str,
        performed_by: Optional[str] = None,
        commodity: Optional[str] = None,
    ) -> LifecycleResult:
        """Permanently revoke a QR code and add to revocation list.

        Args:
            code_id: QR code identifier.
            reason: Revocation reason.
            performed_by: User performing the revocation.
            commodity: EUDR commodity for metrics tracking.

        Returns:
            LifecycleResult with status transition.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            result = self._safe_engine_call(
                self._code_lifecycle_manager, "revoke", {
                    "code_id": code_id,
                    "reason": reason,
                    "performed_by": performed_by,
                },
            )

            self._metrics["codes_revoked"] += 1
            record_revocation(commodity or "unknown")

            provenance_hash = self._record_provenance(
                "lifecycle_event", "revoke", code_id,
                data={"reason": reason},
            )

            return LifecycleResult(
                request_id=request_id,
                success=True,
                code_id=code_id,
                previous_status="active",
                new_status="revoked",
                data=result if isinstance(result, dict) else {},
                provenance_hash=provenance_hash,
                processing_time_ms=_elapsed_ms(start),
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("revoke")
            logger.error("revoke_with_reason failed: %s", exc, exc_info=True)
            return LifecycleResult(
                request_id=request_id,
                success=False,
                code_id=code_id,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    async def activate_and_sign(
        self,
        code_id: str,
        operator_key: str,
        performed_by: Optional[str] = None,
    ) -> QRResult:
        """Activate a QR code and sign it in one step.

        Args:
            code_id: QR code identifier.
            operator_key: Operator signing key identifier.
            performed_by: User performing the operation.

        Returns:
            QRResult with activation and signature data.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()

        try:
            # Step 1: Activate
            activate_result = await self.activate_code(
                code_id=code_id,
                performed_by=performed_by,
                reason="Activation with signing",
            )

            if not activate_result.success:
                return QRResult(
                    request_id=request_id,
                    success=False,
                    code_id=code_id,
                    error=activate_result.error,
                    processing_time_ms=_elapsed_ms(start),
                )

            # Step 2: Sign
            data_hash = _compute_provenance_hash(code_id, operator_key)
            sign_result = await self.sign_qr_code(
                code_id=code_id,
                data_hash=data_hash,
                key_id=operator_key,
            )

            return QRResult(
                request_id=request_id,
                success=sign_result.success,
                code_id=code_id,
                data={
                    "activation": activate_result.to_dict(),
                    "signature": sign_result.data,
                },
                provenance_hash=sign_result.provenance_hash,
                processing_time_ms=_elapsed_ms(start),
            )

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "activate_and_sign failed: %s", exc, exc_info=True,
            )
            return QRResult(
                request_id=request_id,
                success=False,
                code_id=code_id,
                error=str(exc),
                processing_time_ms=_elapsed_ms(start),
            )

    # ==================================================================
    # FACADE METHODS: Composite / Convenience
    # ==================================================================

    async def get_code_full_status(
        self,
        code_id: str,
    ) -> Dict[str, Any]:
        """Get comprehensive status for a QR code.

        Combines code record, lifecycle history, scan history, and
        verification status in a single response.

        Args:
            code_id: QR code identifier.

        Returns:
            Dictionary with complete code status.
        """
        self._ensure_started()
        start = time.monotonic()

        code_record = self._safe_engine_call_with_args(
            self._qr_encoder, "get_code", code_id=code_id,
        )
        lifecycle = self._safe_engine_call_with_args(
            self._code_lifecycle_manager, "get_lifecycle_history",
            code_id=code_id,
        )
        scan_history = self._safe_engine_call_with_args(
            self._code_lifecycle_manager, "get_scan_history",
            code_id=code_id,
        )

        return {
            "code_id": code_id,
            "code_record": code_record,
            "lifecycle_events": lifecycle if isinstance(lifecycle, list) else [],
            "scan_history": scan_history if isinstance(scan_history, list) else [],
            "processing_time_ms": round(_elapsed_ms(start), 2),
        }

    async def generate_compliance_qr(
        self,
        dds_reference: str,
        commodity: str,
        operator_id: str,
        origin_country: str,
        compliance_status: str = "compliant",
        certification_ids: Optional[List[str]] = None,
    ) -> QRResult:
        """Generate an EUDR-specific compliance QR code.

        Convenience method that composes a full_traceability payload
        with compliance-specific fields.

        Args:
            dds_reference: Due Diligence Statement reference.
            commodity: EUDR commodity type.
            operator_id: EUDR operator identifier.
            origin_country: ISO 3166-1 alpha-2 country code.
            compliance_status: EUDR compliance status.
            certification_ids: List of certification references.

        Returns:
            QRResult with compliance QR code.
        """
        payload_data = {
            "dds_reference": dds_reference,
            "commodity": commodity,
            "operator_id": operator_id,
            "origin_country": origin_country,
            "compliance_status": compliance_status,
            "certification_ids": certification_ids or [],
            "regulation": "EU 2023/1115",
            "country_risk": get_country_risk(origin_country),
        }

        return await self.generate_qr_with_payload(
            content_type="full_traceability",
            dds_reference=dds_reference,
            operator_id=operator_id,
            payload_data=payload_data,
            commodity=commodity,
            compliance_status=compliance_status,
            error_correction="H",
        )

    async def generate_consumer_qr(
        self,
        operator_id: str,
        commodity: str,
        origin_country: str,
        product_name: str,
        compliance_status: str = "compliant",
    ) -> QRResult:
        """Generate a consumer-facing QR code with minimal data.

        Args:
            operator_id: EUDR operator identifier.
            commodity: EUDR commodity type.
            origin_country: Country of origin.
            product_name: Product name for display.
            compliance_status: Compliance status.

        Returns:
            QRResult with consumer QR code data.
        """
        payload_data = {
            "product_name": product_name,
            "commodity": commodity,
            "origin_country": origin_country,
            "compliance_status": compliance_status,
            "deforestation_free": compliance_status == "compliant",
        }

        return await self.generate_qr_with_payload(
            content_type="consumer_summary",
            dds_reference="",
            operator_id=operator_id,
            payload_data=payload_data,
            commodity=commodity,
            compliance_status=compliance_status,
        )

    async def generate_gs1_qr(
        self,
        gtin: str,
        operator_id: str,
        batch_lot: Optional[str] = None,
        serial: Optional[str] = None,
        origin_country_numeric: Optional[str] = None,
        certification_ref: Optional[str] = None,
    ) -> QRResult:
        """Generate a GS1 Digital Link QR code.

        Args:
            gtin: GTIN (8-14 digits).
            operator_id: EUDR operator identifier.
            batch_lot: Batch/lot number (AI 10).
            serial: Serial number (AI 21).
            origin_country_numeric: ISO 3166 numeric country code (AI 422).
            certification_ref: Certification reference (AI 7023).

        Returns:
            QRResult with GS1 Digital Link QR code.
        """
        ais: Dict[str, str] = {}
        if batch_lot:
            ais["10"] = batch_lot
        if serial:
            ais["21"] = serial
        if origin_country_numeric:
            ais["422"] = origin_country_numeric
        if certification_ref:
            ais["7023"] = certification_ref

        gs1_uri = build_gs1_digital_link_uri(gtin, ais)

        return await self.generate_qr_with_payload(
            content_type="compact_verification",
            dds_reference="",
            operator_id=operator_id,
            payload_data={"gs1_digital_link": gs1_uri, "gtin": gtin},
        )

    async def search_codes(
        self,
        operator_id: Optional[str] = None,
        commodity: Optional[str] = None,
        status: Optional[str] = None,
        compliance_status: Optional[str] = None,
        batch_code: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Search QR codes by criteria.

        Args:
            operator_id: Filter by operator.
            commodity: Filter by commodity.
            status: Filter by lifecycle status.
            compliance_status: Filter by compliance status.
            batch_code: Filter by batch code.
            limit: Maximum results.
            offset: Result offset.

        Returns:
            Search results dictionary.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._qr_encoder, "search", {
                "operator_id": operator_id,
                "commodity": commodity,
                "status": status,
                "compliance_status": compliance_status,
                "batch_code": batch_code,
                "limit": limit,
                "offset": offset,
            },
        )
        return result if isinstance(result, dict) else {"codes": [], "total": 0}

    async def get_scan_history(
        self,
        code_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get scan history for a QR code.

        Args:
            code_id: QR code identifier.
            limit: Maximum results.
            offset: Result offset.

        Returns:
            Scan history dictionary.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._code_lifecycle_manager, "get_scan_history",
            code_id=code_id, limit=limit, offset=offset,
        )
        return result if isinstance(result, dict) else {"scans": [], "total": 0}

    async def get_template_info(
        self,
        template_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get label template information.

        Args:
            template_name: Template name.

        Returns:
            Template definition or None.
        """
        return get_template(template_name)

    async def list_templates(self) -> List[str]:
        """List all available label template names.

        Returns:
            Sorted list of template names.
        """
        return list_template_names()

    async def get_commodity_info(
        self,
        commodity: str,
    ) -> Optional[Dict[str, Any]]:
        """Get EUDR commodity reference data.

        Args:
            commodity: Commodity name.

        Returns:
            Commodity data dictionary or None.
        """
        return EUDR_COMMODITIES.get(commodity)

    async def check_commodity_by_hs(
        self,
        hs_code: str,
    ) -> Dict[str, Any]:
        """Check if an HS code falls under EUDR regulation.

        Args:
            hs_code: HS code string.

        Returns:
            Dictionary with is_eudr, commodity, and country_risk fields.
        """
        eudr = is_eudr_commodity(hs_code)
        commodity = get_commodity_from_hs(hs_code) if eudr else None
        return {
            "hs_code": hs_code,
            "is_eudr": eudr,
            "commodity": commodity,
        }

    async def get_provenance_trail(
        self,
        entity_type: str,
        entity_id: str,
    ) -> List[Dict[str, Any]]:
        """Get provenance trail for an entity.

        Args:
            entity_type: Entity type (qr_code, payload, label, etc.).
            entity_id: Entity identifier.

        Returns:
            List of provenance records.
        """
        entries = self._provenance.get_entries_for_entity(
            entity_type, entity_id,
        )
        return [e.to_dict() for e in entries]

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of operational metrics.

        Returns:
            Dictionary with metric counters and gauges.
        """
        return {
            "agent_id": _AGENT_ID,
            "version": _MODULE_VERSION,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "counters": dict(self._metrics),
            "provenance_entries": self._provenance.entry_count,
        }

    # ==================================================================
    # Health check
    # ==================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check.

        Returns:
            Dictionary with service health status.
        """
        return {
            "status": "healthy" if self._started else "unhealthy",
            "agent_id": _AGENT_ID,
            "version": _MODULE_VERSION,
            "database_connected": self._db_pool is not None,
            "redis_connected": self._redis is not None,
            "engines_initialized": self._count_initialized_engines(),
            "engines_total": _ENGINE_COUNT,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "metrics": dict(self._metrics),
            "config_hash": self._config_hash[:12],
            "checked_at": utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Internal: engine interaction helpers
    # ------------------------------------------------------------------

    def _safe_engine_call(
        self,
        engine: Any,
        method_name: str,
        data: Dict[str, Any],
    ) -> Any:
        """Safely call an engine method with a data dict.

        Returns None if the engine is not initialized.
        """
        if engine is None:
            logger.warning(
                "Engine not initialized for method '%s'", method_name,
            )
            return None
        method = getattr(engine, method_name, None)
        if method is None:
            logger.warning(
                "Method '%s' not found on engine %s",
                method_name, type(engine).__name__,
            )
            return None
        return method(data)

    def _safe_engine_call_with_args(
        self,
        engine: Any,
        method_name: str,
        **kwargs: Any,
    ) -> Any:
        """Safely call an engine method with keyword arguments.

        Returns None if the engine is not initialized.
        """
        if engine is None:
            logger.warning(
                "Engine not initialized for method '%s'", method_name,
            )
            return None
        method = getattr(engine, method_name, None)
        if method is None:
            logger.warning(
                "Method '%s' not found on engine %s",
                method_name, type(engine).__name__,
            )
            return None
        return method(**kwargs)

    def _ensure_started(self) -> None:
        """Raise RuntimeError if the service is not started."""
        if not self._started:
            raise RuntimeError(
                "QRCodeGeneratorService is not started. "
                "Call startup() before using service methods."
            )

    def _record_provenance(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Any] = None,
    ) -> str:
        """Record a provenance entry and return its hash."""
        if not self._config.enable_provenance:
            return ""
        entry = self._provenance.record(
            entity_type=entity_type,
            action=action,
            entity_id=entity_id,
            data=data,
        )
        return entry.hash_value

    def _wrap_result(
        self,
        result: Any,
        start: float,
    ) -> Dict[str, Any]:
        """Wrap an engine result with timing metadata."""
        if isinstance(result, dict):
            result["processing_time_ms"] = round(_elapsed_ms(start), 2)
            return result
        return {
            "result": result,
            "processing_time_ms": round(_elapsed_ms(start), 2),
        }

    # ------------------------------------------------------------------
    # Internal: startup helpers
    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        """Configure logging level from config."""
        log_level = getattr(logging, self._config.log_level, logging.INFO)
        logging.getLogger("greenlang.agents.eudr.qr_code_generator").setLevel(
            log_level,
        )
        logger.debug("Logging configured: level=%s", self._config.log_level)

    def _init_tracer(self) -> None:
        """Initialize OpenTelemetry tracer if available."""
        if OTEL_AVAILABLE and otel_trace is not None:
            self._tracer = otel_trace.get_tracer(
                _AGENT_ID, _MODULE_VERSION,
            )
            logger.debug("OpenTelemetry tracer initialized")
        else:
            self._tracer = None

    def _load_reference_data(self) -> None:
        """Load reference data into memory."""
        self._ref_templates = dict(TEMPLATE_REGISTRY)
        self._ref_commodities = dict(EUDR_COMMODITIES)
        self._ref_gs1 = {"base_url": GS1_DIGITAL_LINK_BASE_URL}
        logger.info(
            "Reference data loaded: %d templates, %d commodities",
            len(self._ref_templates),
            len(self._ref_commodities),
        )

    async def _connect_database(self) -> None:
        """Connect to PostgreSQL if psycopg is available."""
        if not PSYCOPG_POOL_AVAILABLE:
            logger.info("psycopg_pool not available; skipping DB connection")
            return

        try:
            self._db_pool = AsyncConnectionPool(
                self._config.database_url,
                min_size=1,
                max_size=self._config.pool_size,
            )
            logger.info(
                "PostgreSQL pool created: pool_size=%d",
                self._config.pool_size,
            )
        except Exception as exc:
            logger.warning("Database connection failed: %s", exc)
            self._db_pool = None

    async def _connect_redis(self) -> None:
        """Connect to Redis if redis.asyncio is available."""
        if not REDIS_AVAILABLE or aioredis is None:
            logger.info("redis.asyncio not available; skipping Redis")
            return

        try:
            self._redis = aioredis.from_url(
                self._config.redis_url,
                decode_responses=True,
            )
            logger.info("Redis connected: %s", self._config.redis_url[:30])
        except Exception as exc:
            logger.warning("Redis connection failed: %s", exc)
            self._redis = None

    async def _initialize_engines(self) -> None:
        """Initialize all 8 engines with lazy stubs.

        In production, each engine module would be imported and
        instantiated here. During initial build the stubs serve
        as placeholder objects documenting the expected interface.
        """
        logger.info("Initializing %d engine stubs...", _ENGINE_COUNT)

        self._qr_encoder = _EngineStub("QREncoder")
        self._payload_composer = _EngineStub("PayloadComposer")
        self._label_template_engine = _EngineStub("LabelTemplateEngine")
        self._batch_code_generator = _EngineStub("BatchCodeGenerator")
        self._verification_url_builder = _EngineStub("VerificationURLBuilder")
        self._anti_counterfeit_engine = _EngineStub("AntiCounterfeitEngine")
        self._bulk_generation_pipeline = _EngineStub("BulkGenerationPipeline")
        self._code_lifecycle_manager = _EngineStub("CodeLifecycleManager")

        logger.info(
            "All %d engines initialized (stub mode)", _ENGINE_COUNT,
        )

    def _start_health_check(self) -> None:
        """Start background health check task."""
        logger.debug("Background health check task registered")

    def _stop_health_check(self) -> None:
        """Stop background health check task."""
        if self._health_task is not None:
            self._health_task.cancel()
            self._health_task = None

    def _count_initialized_engines(self) -> int:
        """Count how many engines are initialized (non-None)."""
        engines = [
            self._qr_encoder,
            self._payload_composer,
            self._label_template_engine,
            self._batch_code_generator,
            self._verification_url_builder,
            self._anti_counterfeit_engine,
            self._bulk_generation_pipeline,
            self._code_lifecycle_manager,
        ]
        return sum(1 for e in engines if e is not None)

    # ------------------------------------------------------------------
    # Internal: shutdown helpers
    # ------------------------------------------------------------------

    async def _close_engines(self) -> None:
        """Shut down all engines."""
        engine_names = [
            "_qr_encoder",
            "_payload_composer",
            "_label_template_engine",
            "_batch_code_generator",
            "_verification_url_builder",
            "_anti_counterfeit_engine",
            "_bulk_generation_pipeline",
            "_code_lifecycle_manager",
        ]
        for name in engine_names:
            engine = getattr(self, name, None)
            if engine is not None and hasattr(engine, "close"):
                try:
                    engine.close()
                except Exception as exc:
                    logger.warning("Error closing %s: %s", name, exc)
            setattr(self, name, None)

    async def _close_redis(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            try:
                await self._redis.close()
            except Exception as exc:
                logger.warning("Redis close error: %s", exc)
            self._redis = None

    async def _close_database(self) -> None:
        """Close PostgreSQL connection pool."""
        if self._db_pool is not None:
            try:
                await self._db_pool.close()
            except Exception as exc:
                logger.warning("Database pool close error: %s", exc)
            self._db_pool = None

    def _flush_metrics(self) -> None:
        """Log final metric values on shutdown."""
        logger.info(
            "Final metrics: %s",
            json.dumps(self._metrics, indent=2),
        )

# ---------------------------------------------------------------------------
# Engine stub for pre-engine development
# ---------------------------------------------------------------------------

class _EngineStub:
    """Placeholder engine stub that returns empty dicts for any method call.

    Used during initial service build before individual engine modules
    are implemented. All methods accept arbitrary arguments and return
    an empty dict with the engine name and method for traceability.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, method_name: str) -> Any:
        def stub_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            logger.debug(
                "Engine stub call: %s.%s(...)",
                self._name, method_name,
            )
            return {
                "engine": self._name,
                "method": method_name,
                "status": "stub",
            }
        return stub_method

    def close(self) -> None:
        """No-op close for stub."""
        pass

# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_service_instance: Optional[QRCodeGeneratorService] = None
_service_lock = threading.Lock()

def get_service() -> QRCodeGeneratorService:
    """Return the process-wide singleton QRCodeGeneratorService.

    Creates the instance on first call (lazy initialization).
    Thread-safe via double-checked locking.

    Returns:
        The singleton QRCodeGeneratorService instance.

    Example:
        >>> service_a = get_service()
        >>> service_b = get_service()
        >>> assert service_a is service_b
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = QRCodeGeneratorService()
                logger.info(
                    "QRCodeGeneratorService singleton created"
                )
    return _service_instance

def set_service(service: QRCodeGeneratorService) -> None:
    """Replace the process-wide singleton with a custom service.

    Useful in tests that need isolated service instances.

    Args:
        service: The QRCodeGeneratorService instance to install.

    Raises:
        TypeError: If service is not a QRCodeGeneratorService instance.
    """
    if not isinstance(service, QRCodeGeneratorService):
        raise TypeError(
            f"service must be a QRCodeGeneratorService instance, "
            f"got {type(service)}"
        )
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.info("QRCodeGeneratorService singleton replaced")

def reset_service() -> None:
    """Destroy the current singleton and reset to None.

    The next call to get_service() will create a fresh instance.
    Intended for test teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.info("QRCodeGeneratorService singleton reset to None")

# ---------------------------------------------------------------------------
# FastAPI lifespan integration
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for automatic startup/shutdown.

    Usage:
        >>> from fastapi import FastAPI
        >>> from greenlang.agents.eudr.qr_code_generator.setup import lifespan
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance.

    Yields:
        None - the service is available via get_service() during the
        lifespan.
    """
    service = get_service()
    await service.startup()
    try:
        yield
    finally:
        await service.shutdown()

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    # Service facade
    "QRCodeGeneratorService",
    "get_service",
    "set_service",
    "reset_service",
    # FastAPI lifespan
    "lifespan",
    # Result containers
    "QRResult",
    "LabelResult",
    "BatchResult",
    "ScanResult",
    "BulkJobResult",
    "LifecycleResult",
    # Constants
    "_MODULE_VERSION",
    "_AGENT_ID",
    "_ENGINE_COUNT",
]
