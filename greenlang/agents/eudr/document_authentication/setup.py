# -*- coding: utf-8 -*-
"""
DocumentAuthenticationService - Facade for AGENT-EUDR-012

Single entry point for all document authentication operations.  Manages 8
engines, async PostgreSQL pool, Redis cache, OpenTelemetry tracing,
Prometheus metrics.

Lifecycle:
    startup -> load config -> connect DB -> connect Redis -> load reference data
            -> initialize engines -> start health check
    shutdown -> close engines -> close Redis -> close DB -> flush metrics

Engines (8):
    1. DocumentClassifierEngine    - Document type classification (Feature 1)
    2. SignatureVerifierEngine      - Digital signature verification (Feature 2)
    3. HashIntegrityValidator       - SHA-256/512 integrity validation (Feature 3)
    4. CertificateChainValidator   - X.509 chain validation (Feature 4)
    5. MetadataExtractorEngine     - Metadata extraction & analysis (Feature 5)
    6. FraudPatternDetector        - Deterministic fraud detection (Feature 6)
    7. CrossReferenceVerifier      - External registry verification (Feature 7)
    8. ComplianceReporter          - Report & evidence generation (Feature 8)

Reference Data (3):
    - document_templates: Known template specs per type per country (20+)
    - trusted_cas: Trusted CA registry with pinned issuers (25+)
    - fraud_rules: 15 deterministic fraud rules with thresholds

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with ``FastAPI(lifespan=lifespan)``
    for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.document_authentication.setup import (
    ...     DocumentAuthenticationService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-012
Agent ID: GL-EUDR-DAV-012
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 14
Standard: eIDAS Regulation (EU) No 910/2014
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
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

from greenlang.agents.eudr.document_authentication.config import (
    DocumentAuthenticationConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.document_authentication.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from greenlang.agents.eudr.document_authentication.metrics import (
    PROMETHEUS_AVAILABLE,
    record_api_error,
    record_classification,
    record_crossref_query,
    record_document_processed,
    record_duplicate_detected,
    record_fraud_alert,
    record_fraud_critical,
    record_hash_computed,
    record_report_generated,
    record_signature_verified,
    record_tampering_detected,
    observe_classification_duration,
    observe_crossref_duration,
    observe_verification_duration,
    set_active_verifications,
)

# ---------------------------------------------------------------------------
# Internal imports: reference data
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.document_authentication.reference_data import (
    DOCUMENT_TEMPLATES,
    FRAUD_RULES,
    TRUSTED_CAS,
    get_all_rules,
    get_required_documents,
    get_trusted_cas,
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-DAV-012"
_ENGINE_COUNT = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_provenance_hash(*parts: str) -> str:
    """Compute SHA-256 hash over concatenated string parts."""
    combined = "|".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _generate_request_id() -> str:
    """Generate a unique request identifier."""
    return f"DAV-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Result container: HealthResult
# ---------------------------------------------------------------------------


class HealthResult:
    """Health check result container.

    Attributes:
        status: Overall health status (healthy, degraded, unhealthy).
        checks: Individual component check results.
        timestamp: When the health check was performed.
        version: Service version string.
        uptime_seconds: Seconds since service startup.
    """

    __slots__ = ("status", "checks", "timestamp", "version", "uptime_seconds")

    def __init__(
        self,
        status: str = "unhealthy",
        checks: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        version: str = _MODULE_VERSION,
        uptime_seconds: float = 0.0,
    ) -> None:
        self.status = status
        self.checks = checks or {}
        self.timestamp = timestamp or _utcnow()
        self.version = version
        self.uptime_seconds = uptime_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize health status to dictionary for JSON response."""
        return {
            "status": self.status,
            "checks": self.checks,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
        }


# ---------------------------------------------------------------------------
# Result container: ClassificationResult
# ---------------------------------------------------------------------------


class ClassificationResult:
    """Result from a document classification operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        document_id: Document identifier.
        data: Classification result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "document_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        document_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.document_id = document_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "document_id": self.document_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: SignatureResult
# ---------------------------------------------------------------------------


class SignatureResult:
    """Result from a signature verification operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        document_id: Document identifier.
        data: Signature verification result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "document_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        document_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.document_id = document_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "document_id": self.document_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: HashResult
# ---------------------------------------------------------------------------


class HashResult:
    """Result from a hash integrity operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        data: Hash operation result data payload.
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


# ---------------------------------------------------------------------------
# Result container: CertificateResult
# ---------------------------------------------------------------------------


class CertificateResult:
    """Result from a certificate chain validation operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        data: Certificate validation result data payload.
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


# ---------------------------------------------------------------------------
# Result container: MetadataResult
# ---------------------------------------------------------------------------


class MetadataResult:
    """Result from a metadata extraction operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        document_id: Document identifier.
        data: Metadata extraction result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "document_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        document_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.document_id = document_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "document_id": self.document_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: FraudResult
# ---------------------------------------------------------------------------


class FraudResult:
    """Result from a fraud detection operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        document_id: Document identifier.
        data: Fraud detection result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "document_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        document_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.document_id = document_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "document_id": self.document_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: CrossRefResult
# ---------------------------------------------------------------------------


class CrossRefResult:
    """Result from a cross-reference verification operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        data: Cross-reference result data payload.
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


# ---------------------------------------------------------------------------
# Result container: ReportResult
# ---------------------------------------------------------------------------


class ReportResult:
    """Result from a report generation operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        report_id: Report identifier.
        data: Report result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "report_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        report_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.report_id = report_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "report_id": self.report_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: FullVerificationResult
# ---------------------------------------------------------------------------


class FullVerificationResult:
    """Result from a full document verification (all engines).

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        document_id: Document identifier.
        data: Full verification result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "document_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        document_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.document_id = document_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "document_id": self.document_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: BatchResult
# ---------------------------------------------------------------------------


class BatchResult:
    """Result from a batch processing operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        job_id: Batch job identifier.
        data: Batch result data payload.
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


# ---------------------------------------------------------------------------
# Result container: DashboardResult
# ---------------------------------------------------------------------------


class DashboardResult:
    """Result from a dashboard or overview operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        data: Dashboard data payload.
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


# ===========================================================================
# DocumentAuthenticationService - Main facade
# ===========================================================================


class DocumentAuthenticationService:
    """Facade for the Document Authentication Agent (AGENT-EUDR-012).

    Provides a unified interface to all 8 engines:
        1. DocumentClassifierEngine    - Document type classification
        2. SignatureVerifierEngine      - Digital signature verification
        3. HashIntegrityValidator       - SHA-256/512 integrity validation
        4. CertificateChainValidator   - X.509 chain validation
        5. MetadataExtractorEngine     - Metadata extraction & analysis
        6. FraudPatternDetector        - Deterministic fraud detection
        7. CrossReferenceVerifier      - External registry verification
        8. ComplianceReporter          - Report & evidence generation

    Singleton pattern with thread-safe initialization.

    Example:
        >>> service = DocumentAuthenticationService()
        >>> await service.startup()
        >>> result = await service.classify_document({...})
        >>> await service.shutdown()
    """

    _instance: Optional[DocumentAuthenticationService] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize DocumentAuthenticationService.

        Loads configuration but does NOT start connections or engines.
        Call ``startup()`` to activate the service.
        """
        self._config: DocumentAuthenticationConfig = get_config()

        self._started = False
        self._start_time: Optional[float] = None
        self._config_hash = _compute_provenance_hash(
            self._config.database_url,
            self._config.redis_url,
            self._config.hash_algorithm,
            str(self._config.min_confidence_high),
            self._config.genesis_hash,
        )

        # Connection handles
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None

        # Engine instances (initialized in startup)
        self._classifier: Optional[Any] = None
        self._signature_verifier: Optional[Any] = None
        self._hash_validator: Optional[Any] = None
        self._certificate_validator: Optional[Any] = None
        self._metadata_extractor: Optional[Any] = None
        self._fraud_detector: Optional[Any] = None
        self._cross_reference: Optional[Any] = None
        self._reporter: Optional[Any] = None

        # Reference data (loaded in startup)
        self._ref_document_templates: Optional[Dict[str, Any]] = None
        self._ref_trusted_cas: Optional[List[Dict[str, Any]]] = None
        self._ref_fraud_rules: Optional[List[Dict[str, Any]]] = None

        # Health check background task
        self._health_task: Optional[asyncio.Task[None]] = None
        self._last_health: Optional[HealthResult] = None

        # OpenTelemetry tracer
        self._tracer: Optional[Any] = None

        # Metrics counters
        self._metrics: Dict[str, int] = {
            "documents_classified": 0,
            "signatures_verified": 0,
            "hashes_computed": 0,
            "hashes_verified": 0,
            "certificates_validated": 0,
            "metadata_extracted": 0,
            "fraud_detections": 0,
            "cross_references": 0,
            "reports_generated": 0,
            "full_verifications": 0,
            "batch_jobs": 0,
            "errors": 0,
        }

        logger.info(
            "DocumentAuthenticationService created: config_hash=%s, "
            "hash_algorithm=%s, confidence_high=%.2f",
            self._config_hash[:12],
            self._config.hash_algorithm,
            self._config.min_confidence_high,
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
    def config(self) -> DocumentAuthenticationConfig:
        """Return the service configuration."""
        return self._config

    @property
    def classifier(self) -> Any:
        """Return the DocumentClassifierEngine instance."""
        self._ensure_started()
        return self._classifier

    @property
    def signature_verifier(self) -> Any:
        """Return the SignatureVerifierEngine instance."""
        self._ensure_started()
        return self._signature_verifier

    @property
    def hash_validator(self) -> Any:
        """Return the HashIntegrityValidator instance."""
        self._ensure_started()
        return self._hash_validator

    @property
    def certificate_validator(self) -> Any:
        """Return the CertificateChainValidator instance."""
        self._ensure_started()
        return self._certificate_validator

    @property
    def metadata_extractor(self) -> Any:
        """Return the MetadataExtractorEngine instance."""
        self._ensure_started()
        return self._metadata_extractor

    @property
    def fraud_detector(self) -> Any:
        """Return the FraudPatternDetector instance."""
        self._ensure_started()
        return self._fraud_detector

    @property
    def cross_reference(self) -> Any:
        """Return the CrossReferenceVerifier instance."""
        self._ensure_started()
        return self._cross_reference

    @property
    def reporter(self) -> Any:
        """Return the ComplianceReporter instance."""
        self._ensure_started()
        return self._reporter

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
            logger.debug("DocumentAuthenticationService already started")
            return

        start = time.monotonic()
        logger.info("DocumentAuthenticationService starting up...")

        self._configure_logging()
        self._init_tracer()
        self._load_reference_data()
        await self._connect_database()
        await self._connect_redis()
        await self._initialize_engines()
        self._start_health_check()

        self._started = True
        self._start_time = time.monotonic()
        elapsed = (time.monotonic() - start) * 1000

        logger.info(
            "DocumentAuthenticationService started in %.1fms: "
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
            logger.debug("DocumentAuthenticationService already stopped")
            return

        logger.info("DocumentAuthenticationService shutting down...")
        start = time.monotonic()

        self._stop_health_check()
        await self._close_engines()
        await self._close_redis()
        await self._close_database()
        self._flush_metrics()

        self._started = False
        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "DocumentAuthenticationService shut down in %.1fms", elapsed,
        )

    # ==================================================================
    # FACADE METHODS: Engine 1 - DocumentClassifierEngine
    # ==================================================================

    async def classify_document(
        self,
        document_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Classify a document by type.

        Delegates to DocumentClassifierEngine.classify().

        Args:
            document_data: Document data including content, filename,
                and optional country_hint.

        Returns:
            Dictionary with classification result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._classifier, "classify", document_data,
            )
            self._metrics["documents_classified"] += 1
            elapsed = (time.monotonic() - start)
            observe_classification_duration(elapsed)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("classify")
            logger.error("classify_document failed: %s", exc, exc_info=True)
            raise

    async def batch_classify(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Classify multiple documents in a batch.

        Delegates to DocumentClassifierEngine.batch_classify().

        Args:
            documents: List of document data dictionaries.

        Returns:
            Dictionary with batch classification results.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._classifier, "batch_classify",
                {"documents": documents},
            )
            self._metrics["documents_classified"] += len(documents)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("classify")
            logger.error("batch_classify failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # FACADE METHODS: Engine 2 - SignatureVerifierEngine
    # ==================================================================

    async def verify_signature(
        self,
        signature_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify a document's digital signature.

        Delegates to SignatureVerifierEngine.verify().

        Args:
            signature_data: Signature verification data including
                document_content and optional signature_standard.

        Returns:
            Dictionary with signature verification result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._signature_verifier, "verify", signature_data,
            )
            self._metrics["signatures_verified"] += 1
            elapsed = (time.monotonic() - start)
            observe_verification_duration(elapsed)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("verify_signature")
            logger.error(
                "verify_signature failed: %s", exc, exc_info=True,
            )
            raise

    async def batch_verify_signatures(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify signatures for multiple documents in a batch.

        Delegates to SignatureVerifierEngine.batch_verify().

        Args:
            documents: List of document data dictionaries.

        Returns:
            Dictionary with batch signature verification results.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._signature_verifier, "batch_verify",
                {"documents": documents},
            )
            self._metrics["signatures_verified"] += len(documents)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("verify_signature")
            logger.error(
                "batch_verify_signatures failed: %s", exc, exc_info=True,
            )
            raise

    # ==================================================================
    # FACADE METHODS: Engine 3 - HashIntegrityValidator
    # ==================================================================

    async def compute_hash(
        self,
        hash_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute SHA-256/512 hash for a document.

        Delegates to HashIntegrityValidator.compute_hash().

        Args:
            hash_data: Hash computation data including document_content
                and optional algorithm override.

        Returns:
            Dictionary with computed hash values.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._hash_validator, "compute_hash", hash_data,
            )
            self._metrics["hashes_computed"] += 1
            record_hash_computed(self._config.hash_algorithm)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("compute_hash")
            logger.error("compute_hash failed: %s", exc, exc_info=True)
            raise

    async def verify_hash(
        self,
        verify_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify a document against a known hash.

        Delegates to HashIntegrityValidator.verify_hash().

        Args:
            verify_data: Hash verification data including document_content
                and expected_hash.

        Returns:
            Dictionary with verification result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._hash_validator, "verify_hash", verify_data,
            )
            self._metrics["hashes_verified"] += 1
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("verify_hash")
            logger.error("verify_hash failed: %s", exc, exc_info=True)
            raise

    async def lookup_hash(
        self,
        hash_value: str,
    ) -> Dict[str, Any]:
        """Look up a hash in the registry to detect duplicates.

        Delegates to HashIntegrityValidator.lookup_hash().

        Args:
            hash_value: SHA-256 hash to look up.

        Returns:
            Dictionary with lookup result.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._hash_validator, "lookup_hash",
            hash_value=hash_value,
        )
        if result is None:
            return {"status": "engine_unavailable", "found": False}
        return result if isinstance(result, dict) else {"found": False}

    async def build_merkle_tree(
        self,
        document_hashes: List[str],
    ) -> Dict[str, Any]:
        """Build a Merkle tree over a set of document hashes.

        Delegates to HashIntegrityValidator.build_merkle_tree().

        Args:
            document_hashes: List of SHA-256 hashes to include.

        Returns:
            Dictionary with Merkle root and tree structure.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._hash_validator, "build_merkle_tree",
            {"hashes": document_hashes},
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result

    # ==================================================================
    # FACADE METHODS: Engine 4 - CertificateChainValidator
    # ==================================================================

    async def validate_certificate_chain(
        self,
        cert_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate a certificate chain.

        Delegates to CertificateChainValidator.validate_chain().

        Args:
            cert_data: Certificate chain data including leaf_cert_pem
                and optional intermediate chain.

        Returns:
            Dictionary with chain validation result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._certificate_validator, "validate_chain", cert_data,
            )
            self._metrics["certificates_validated"] += 1
            elapsed = (time.monotonic() - start)
            observe_verification_duration(elapsed)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("validate_chain")
            logger.error(
                "validate_certificate_chain failed: %s", exc,
                exc_info=True,
            )
            raise

    async def add_trusted_ca(
        self,
        ca_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add a trusted CA to the certificate chain validator.

        Delegates to CertificateChainValidator.add_trusted_ca().

        Args:
            ca_data: Trusted CA data including ca_name, ca_cert_pem,
                and optional ca_category.

        Returns:
            Dictionary with registration result.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._certificate_validator, "add_trusted_ca", ca_data,
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result

    async def list_trusted_cas(self) -> Dict[str, Any]:
        """List all trusted certificate authorities.

        Delegates to CertificateChainValidator.list_trusted_cas().

        Returns:
            Dictionary with list of trusted CAs.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._certificate_validator, "list_trusted_cas",
        )
        if result is None:
            return {
                "status": "engine_unavailable",
                "trusted_cas": get_trusted_cas(),
            }
        return result

    # ==================================================================
    # FACADE METHODS: Engine 5 - MetadataExtractorEngine
    # ==================================================================

    async def extract_metadata(
        self,
        document_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract metadata from a document.

        Delegates to MetadataExtractorEngine.extract().

        Args:
            document_data: Document data including content and filename.

        Returns:
            Dictionary with extracted metadata.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._metadata_extractor, "extract", document_data,
            )
            self._metrics["metadata_extracted"] += 1
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("extract_metadata")
            logger.error(
                "extract_metadata failed: %s", exc, exc_info=True,
            )
            raise

    async def validate_metadata(
        self,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate extracted metadata for anomalies.

        Delegates to MetadataExtractorEngine.validate().

        Args:
            metadata: Extracted metadata to validate.

        Returns:
            Dictionary with metadata validation result.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._metadata_extractor, "validate", metadata,
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result

    # ==================================================================
    # FACADE METHODS: Engine 6 - FraudPatternDetector
    # ==================================================================

    async def detect_fraud(
        self,
        fraud_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run fraud pattern detection on a document.

        Delegates to FraudPatternDetector.detect().

        Args:
            fraud_data: Fraud detection data including document_id,
                document_type, and extracted fields.

        Returns:
            Dictionary with fraud detection result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._fraud_detector, "detect", fraud_data,
            )
            self._metrics["fraud_detections"] += 1
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("detect_fraud")
            logger.error("detect_fraud failed: %s", exc, exc_info=True)
            raise

    async def batch_detect_fraud(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run fraud detection on multiple documents in a batch.

        Delegates to FraudPatternDetector.batch_detect().

        Args:
            documents: List of document data dictionaries.

        Returns:
            Dictionary with batch fraud detection results.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._fraud_detector, "batch_detect",
                {"documents": documents},
            )
            self._metrics["fraud_detections"] += len(documents)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("detect_fraud")
            logger.error(
                "batch_detect_fraud failed: %s", exc, exc_info=True,
            )
            raise

    async def get_fraud_rules(self) -> Dict[str, Any]:
        """Return all configured fraud detection rules.

        Returns:
            Dictionary with list of fraud rules and their status.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._fraud_detector, "get_rules",
        )
        if result is None:
            return {
                "status": "engine_unavailable",
                "rules": get_all_rules(),
            }
        return result

    # ==================================================================
    # FACADE METHODS: Engine 7 - CrossReferenceVerifier
    # ==================================================================

    async def verify_cross_reference(
        self,
        crossref_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify a document against external registries.

        Delegates to CrossReferenceVerifier.verify().

        Args:
            crossref_data: Cross-reference data including document_type,
                certificate_number, and registry_type.

        Returns:
            Dictionary with cross-reference verification result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._cross_reference, "verify", crossref_data,
            )
            self._metrics["cross_references"] += 1
            elapsed = (time.monotonic() - start)
            observe_crossref_duration(elapsed)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("cross_reference")
            logger.error(
                "verify_cross_reference failed: %s", exc, exc_info=True,
            )
            raise

    async def batch_cross_reference(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify multiple documents against external registries.

        Delegates to CrossReferenceVerifier.batch_verify().

        Args:
            documents: List of cross-reference data dictionaries.

        Returns:
            Dictionary with batch cross-reference results.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._cross_reference, "batch_verify",
                {"documents": documents},
            )
            self._metrics["cross_references"] += len(documents)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("cross_reference")
            logger.error(
                "batch_cross_reference failed: %s", exc, exc_info=True,
            )
            raise

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cross-reference cache statistics.

        Returns:
            Dictionary with cache hit/miss stats.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._cross_reference, "get_cache_stats",
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result

    # ==================================================================
    # FACADE METHODS: Engine 8 - ComplianceReporter
    # ==================================================================

    async def generate_authentication_report(
        self,
        report_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate an authentication report for a document.

        Delegates to ComplianceReporter.generate_report().

        Args:
            report_data: Report generation data including document_id,
                report_format, and optional sections.

        Returns:
            Dictionary with report generation result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._reporter, "generate_report", report_data,
            )
            self._metrics["reports_generated"] += 1
            report_fmt = report_data.get("report_format", "json")
            record_report_generated(report_fmt)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("generate_report")
            logger.error(
                "generate_authentication_report failed: %s",
                exc, exc_info=True,
            )
            raise

    async def generate_evidence_package(
        self,
        package_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a compliance evidence package for a DDS.

        Delegates to ComplianceReporter.generate_evidence_package().

        Args:
            package_data: Evidence package data including dds_id,
                document_ids, and optional format.

        Returns:
            Dictionary with evidence package generation result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._reporter, "generate_evidence_package",
                package_data,
            )
            self._metrics["reports_generated"] += 1
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("generate_report")
            logger.error(
                "generate_evidence_package failed: %s",
                exc, exc_info=True,
            )
            raise

    async def generate_completeness_report(
        self,
        completeness_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a document completeness report for a DDS.

        Delegates to ComplianceReporter.generate_completeness_report().

        Args:
            completeness_data: Completeness check data including
                commodity, submitted_document_types.

        Returns:
            Dictionary with completeness report.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._reporter, "generate_completeness_report",
            completeness_data,
        )
        if result is None:
            commodity = completeness_data.get("commodity", "")
            required = get_required_documents(commodity)
            return {
                "status": "engine_unavailable",
                "required_documents": required,
            }
        return result

    async def get_report(
        self,
        report_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a generated report by ID.

        Delegates to ComplianceReporter.get_report().

        Args:
            report_id: Report identifier.

        Returns:
            Report record or None.
        """
        self._ensure_started()
        return self._safe_engine_call_with_args(
            self._reporter, "get_report",
            report_id=report_id,
        )

    async def download_report(
        self,
        report_id: str,
    ) -> Dict[str, Any]:
        """Download a generated report with content.

        Delegates to ComplianceReporter.download_report().

        Args:
            report_id: Report identifier.

        Returns:
            Dictionary with report content.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._reporter, "download_report",
            report_id=report_id,
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result if isinstance(result, dict) else {"data": result}

    # ==================================================================
    # FACADE METHODS: Cross-engine operations
    # ==================================================================

    async def get_dashboard(self) -> Dict[str, Any]:
        """Get an overview dashboard of authentication activity.

        Aggregates metrics from all engines into a single dashboard
        view suitable for the EUDR compliance dashboard.

        Returns:
            Dictionary with dashboard data.
        """
        self._ensure_started()
        return {
            "agent_id": _AGENT_ID,
            "version": _MODULE_VERSION,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "timestamp": _utcnow().isoformat(),
            "metrics": dict(self._metrics),
            "engines_active": self._count_initialized_engines(),
            "engines_total": _ENGINE_COUNT,
            "config_hash": self._config_hash[:12],
        }

    async def full_document_verification(
        self,
        document_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run all 8 engines on a single document.

        Executes the complete document authentication pipeline:
            1. Classify the document type
            2. Verify the digital signature
            3. Compute and check integrity hash
            4. Validate the certificate chain
            5. Extract and validate metadata
            6. Run fraud pattern detection
            7. Verify against external registries
            8. Generate authentication report

        Args:
            document_data: Complete document data including content,
                filename, optional country_hint, and operator context.

        Returns:
            Dictionary with combined results from all engines.
        """
        self._ensure_started()
        start = time.monotonic()
        request_id = _generate_request_id()
        results: Dict[str, Any] = {"request_id": request_id}

        try:
            # Step 1: Classify
            classification = await self.classify_document(document_data)
            results["classification"] = classification

            # Step 2: Verify signature
            signature = await self.verify_signature(document_data)
            results["signature"] = signature

            # Step 3: Compute hash
            hash_result = await self.compute_hash(document_data)
            results["hash"] = hash_result

            # Step 4: Validate certificate chain
            cert_result = await self.validate_certificate_chain(
                document_data,
            )
            results["certificate"] = cert_result

            # Step 5: Extract metadata
            metadata = await self.extract_metadata(document_data)
            results["metadata"] = metadata

            # Step 6: Detect fraud
            fraud = await self.detect_fraud(document_data)
            results["fraud"] = fraud

            # Step 7: Cross-reference (if applicable)
            crossref = await self.verify_cross_reference(document_data)
            results["cross_reference"] = crossref

            # Step 8: Generate report
            report_data = {
                "document_id": document_data.get("document_id", ""),
                "report_format": document_data.get("report_format", "json"),
                "results": results,
            }
            report = await self.generate_authentication_report(report_data)
            results["report"] = report

            # Compute overall status
            results["overall_status"] = self._compute_overall_status(
                results,
            )

            self._metrics["full_verifications"] += 1
            elapsed_ms = (time.monotonic() - start) * 1000
            results["processing_time_ms"] = round(elapsed_ms, 2)

            # Provenance hash
            results["provenance_hash"] = _compute_provenance_hash(
                request_id,
                json.dumps(results.get("overall_status", {}), default=str),
            )

            return results

        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("batch_verify")
            logger.error(
                "full_document_verification failed: %s",
                exc, exc_info=True,
            )
            raise

    async def batch_full_verification(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run full verification on multiple documents.

        Processes each document through all 8 engines. Respects
        batch_max_size and batch_concurrency from configuration.

        Args:
            documents: List of document data dictionaries.

        Returns:
            Dictionary with batch results including per-document outcomes.
        """
        self._ensure_started()
        start = time.monotonic()
        job_id = f"BATCH-{uuid.uuid4().hex[:8]}"

        max_size = self._config.batch_max_size
        if len(documents) > max_size:
            return {
                "success": False,
                "job_id": job_id,
                "error": (
                    f"Batch size {len(documents)} exceeds maximum "
                    f"{max_size}"
                ),
            }

        set_active_verifications(len(documents))
        results: List[Dict[str, Any]] = []

        try:
            for idx, doc in enumerate(documents):
                doc_result = await self.full_document_verification(doc)
                results.append(doc_result)
                set_active_verifications(len(documents) - idx - 1)

            self._metrics["batch_jobs"] += 1
            elapsed_ms = (time.monotonic() - start) * 1000

            return {
                "success": True,
                "job_id": job_id,
                "total_documents": len(documents),
                "results": results,
                "processing_time_ms": round(elapsed_ms, 2),
                "provenance_hash": _compute_provenance_hash(
                    job_id, str(len(documents)),
                ),
            }

        except Exception as exc:
            self._metrics["errors"] += 1
            set_active_verifications(0)
            logger.error(
                "batch_full_verification failed: %s", exc, exc_info=True,
            )
            raise
        finally:
            set_active_verifications(0)

    # ==================================================================
    # Statistics and health
    # ==================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics and counters.

        Returns:
            Dictionary with service metrics and counters.
        """
        return {
            "agent_id": _AGENT_ID,
            "version": _MODULE_VERSION,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "timestamp": _utcnow().isoformat(),
            "metrics": dict(self._metrics),
            "engines_active": self._count_initialized_engines(),
            "engines_total": _ENGINE_COUNT,
            "reference_data": {
                "document_templates": (
                    len(self._ref_document_templates)
                    if self._ref_document_templates else 0
                ),
                "trusted_cas": (
                    len(self._ref_trusted_cas)
                    if self._ref_trusted_cas else 0
                ),
                "fraud_rules": (
                    len(self._ref_fraud_rules)
                    if self._ref_fraud_rules else 0
                ),
            },
        }

    async def get_health(self) -> Dict[str, Any]:
        """Alias for health_check(). Returns comprehensive health status."""
        return await self.health_check()

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check.

        Checks database, Redis, engine, and reference data health.

        Returns:
            Dictionary with overall status and component checks.
        """
        checks: Dict[str, Any] = {}

        checks["database"] = await self._check_database_health()
        checks["redis"] = await self._check_redis_health()
        checks["engines"] = self._check_engine_health()
        checks["reference_data"] = self._check_reference_data_health()

        # Determine overall status
        statuses = [c.get("status", "unhealthy") for c in checks.values()]
        if all(s == "healthy" for s in statuses):
            overall = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            overall = "unhealthy"
        else:
            overall = "degraded"

        health = HealthResult(
            status=overall,
            checks=checks,
            timestamp=_utcnow(),
            version=_MODULE_VERSION,
            uptime_seconds=self.uptime_seconds,
        )
        self._last_health = health
        return health.to_dict()

    # ------------------------------------------------------------------
    # Internal: Startup helpers
    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        """Configure structured logging for the service."""
        log_level = getattr(
            logging, self._config.log_level.upper(), logging.INFO,
        )
        logging.getLogger(
            "greenlang.agents.eudr.document_authentication"
        ).setLevel(log_level)
        logger.debug("Logging configured: level=%s", self._config.log_level)

    def _init_tracer(self) -> None:
        """Initialize OpenTelemetry tracer if available."""
        if OTEL_AVAILABLE and otel_trace is not None:
            try:
                self._tracer = otel_trace.get_tracer(
                    "greenlang.agents.eudr.document_authentication",
                    _MODULE_VERSION,
                )
                logger.debug("OpenTelemetry tracer initialized")
            except Exception as exc:
                logger.warning("OpenTelemetry init failed: %s", exc)
        else:
            logger.debug("OpenTelemetry not available; tracing disabled")

    def _load_reference_data(self) -> None:
        """Load reference datasets for deterministic validation."""
        try:
            self._ref_document_templates = DOCUMENT_TEMPLATES
            logger.debug(
                "Loaded document templates: %d document types",
                len(self._ref_document_templates)
                if self._ref_document_templates else 0,
            )
        except Exception as exc:
            logger.warning("Failed to load document templates: %s", exc)

        try:
            self._ref_trusted_cas = TRUSTED_CAS
            logger.debug(
                "Loaded trusted CAs: %d entries",
                len(self._ref_trusted_cas)
                if self._ref_trusted_cas else 0,
            )
        except Exception as exc:
            logger.warning("Failed to load trusted CAs: %s", exc)

        try:
            self._ref_fraud_rules = FRAUD_RULES
            logger.debug(
                "Loaded fraud rules: %d rules",
                len(self._ref_fraud_rules)
                if self._ref_fraud_rules else 0,
            )
        except Exception as exc:
            logger.warning("Failed to load fraud rules: %s", exc)

    async def _connect_database(self) -> None:
        """Connect to the PostgreSQL database pool."""
        if not PSYCOPG_POOL_AVAILABLE:
            logger.info(
                "psycopg_pool not available; database connection skipped"
            )
            return

        try:
            self._db_pool = AsyncConnectionPool(
                self._config.database_url,
                min_size=2,
                max_size=self._config.pool_size,
                open=False,
            )
            await self._db_pool.open()
            logger.info(
                "PostgreSQL connection pool opened: pool_size=%d",
                self._config.pool_size,
            )
        except Exception as exc:
            logger.warning(
                "PostgreSQL connection failed (non-fatal): %s", exc,
            )
            self._db_pool = None

    async def _connect_redis(self) -> None:
        """Connect to the Redis cache."""
        if not REDIS_AVAILABLE or aioredis is None:
            logger.info("Redis not available; cache connection skipped")
            return

        try:
            self._redis = aioredis.from_url(
                self._config.redis_url,
                decode_responses=True,
            )
            await self._redis.ping()
            logger.info("Redis connection established")
        except Exception as exc:
            logger.warning(
                "Redis connection failed (non-fatal): %s", exc,
            )
            self._redis = None

    async def _initialize_engines(self) -> None:
        """Initialize all 8 engines with graceful fallback."""
        config = self._config

        # Engine 1: DocumentClassifierEngine
        try:
            from greenlang.agents.eudr.document_authentication.document_classifier import (
                DocumentClassifierEngine,
            )
            self._classifier = DocumentClassifierEngine(config=config)
            logger.debug("Engine 1 initialized: DocumentClassifierEngine")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 1 (DocumentClassifierEngine) init failed: %s", exc,
            )

        # Engine 2: SignatureVerifierEngine
        try:
            from greenlang.agents.eudr.document_authentication.signature_verifier import (
                SignatureVerifierEngine,
            )
            self._signature_verifier = SignatureVerifierEngine(config=config)
            logger.debug("Engine 2 initialized: SignatureVerifierEngine")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 2 (SignatureVerifierEngine) init failed: %s", exc,
            )

        # Engine 3: HashIntegrityValidator
        try:
            from greenlang.agents.eudr.document_authentication.hash_integrity_validator import (
                HashIntegrityValidator,
            )
            self._hash_validator = HashIntegrityValidator(config=config)
            logger.debug("Engine 3 initialized: HashIntegrityValidator")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 3 (HashIntegrityValidator) init failed: %s", exc,
            )

        # Engine 4: CertificateChainValidator
        try:
            from greenlang.agents.eudr.document_authentication.certificate_chain_validator import (
                CertificateChainValidator,
            )
            self._certificate_validator = CertificateChainValidator(
                config=config,
            )
            logger.debug("Engine 4 initialized: CertificateChainValidator")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 4 (CertificateChainValidator) init failed: %s", exc,
            )

        # Engine 5: MetadataExtractorEngine
        try:
            from greenlang.agents.eudr.document_authentication.metadata_extractor import (
                MetadataExtractorEngine,
            )
            self._metadata_extractor = MetadataExtractorEngine(config=config)
            logger.debug("Engine 5 initialized: MetadataExtractorEngine")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 5 (MetadataExtractorEngine) init failed: %s", exc,
            )

        # Engine 6: FraudPatternDetector
        try:
            from greenlang.agents.eudr.document_authentication.fraud_pattern_detector import (
                FraudPatternDetector,
            )
            self._fraud_detector = FraudPatternDetector(config=config)
            logger.debug("Engine 6 initialized: FraudPatternDetector")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 6 (FraudPatternDetector) init failed: %s", exc,
            )

        # Engine 7: CrossReferenceVerifier
        try:
            from greenlang.agents.eudr.document_authentication.cross_reference_verifier import (
                CrossReferenceVerifier,
            )
            self._cross_reference = CrossReferenceVerifier(config=config)
            logger.debug("Engine 7 initialized: CrossReferenceVerifier")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 7 (CrossReferenceVerifier) init failed: %s", exc,
            )

        # Engine 8: ComplianceReporter
        try:
            from greenlang.agents.eudr.document_authentication.compliance_reporter import (
                ComplianceReporter,
            )
            self._reporter = ComplianceReporter(config=config)
            logger.debug("Engine 8 initialized: ComplianceReporter")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 8 (ComplianceReporter) init failed: %s", exc,
            )

        count = self._count_initialized_engines()
        logger.info("Engines initialized: %d/%d", count, _ENGINE_COUNT)

    async def _close_engines(self) -> None:
        """Close all engines and release resources."""
        engine_names = [
            "_classifier",
            "_signature_verifier",
            "_hash_validator",
            "_certificate_validator",
            "_metadata_extractor",
            "_fraud_detector",
            "_cross_reference",
            "_reporter",
        ]
        for name in engine_names:
            engine = getattr(self, name, None)
            if engine is not None and hasattr(engine, "close"):
                try:
                    result = engine.close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.warning("Error closing %s: %s", name, exc)
            setattr(self, name, None)
        logger.debug("All engines closed")

    async def _close_redis(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            try:
                await self._redis.close()
                logger.info("Redis connection closed")
            except Exception as exc:
                logger.warning("Error closing Redis: %s", exc)
            finally:
                self._redis = None

    async def _close_database(self) -> None:
        """Close the PostgreSQL connection pool."""
        if self._db_pool is not None:
            try:
                await self._db_pool.close()
                logger.info("PostgreSQL connection pool closed")
            except Exception as exc:
                logger.warning("Error closing database pool: %s", exc)
            finally:
                self._db_pool = None

    def _flush_metrics(self) -> None:
        """Flush Prometheus metrics."""
        if self._config.enable_metrics:
            logger.debug(
                "Metrics flushed: %s",
                {k: v for k, v in self._metrics.items() if v > 0},
            )

    # ------------------------------------------------------------------
    # Internal: Health checks
    # ------------------------------------------------------------------

    def _start_health_check(self) -> None:
        """Start the background health check task."""
        try:
            loop = asyncio.get_running_loop()
            self._health_task = loop.create_task(
                self._health_check_loop(),
            )
            logger.debug("Health check background task started")
        except RuntimeError:
            logger.debug(
                "No running event loop; health check task not started",
            )

    def _stop_health_check(self) -> None:
        """Cancel the background health check task."""
        if self._health_task is not None:
            self._health_task.cancel()
            self._health_task = None
            logger.debug("Health check background task cancelled")

    async def _health_check_loop(self) -> None:
        """Periodic background health check."""
        while True:
            try:
                await asyncio.sleep(30.0)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Health check loop error: %s", exc)

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity health."""
        if self._db_pool is None:
            return {"status": "degraded", "reason": "no_pool"}
        try:
            start = time.monotonic()
            async with self._db_pool.connection() as conn:
                await conn.execute("SELECT 1")
            latency_ms = (time.monotonic() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            return {"status": "unhealthy", "reason": str(exc)}

    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity health."""
        if self._redis is None:
            return {"status": "degraded", "reason": "not_connected"}
        try:
            start = time.monotonic()
            await self._redis.ping()
            latency_ms = (time.monotonic() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            return {"status": "unhealthy", "reason": str(exc)}

    def _check_engine_health(self) -> Dict[str, Any]:
        """Check engine initialization status."""
        engines = {
            "document_classifier": self._classifier,
            "signature_verifier": self._signature_verifier,
            "hash_integrity_validator": self._hash_validator,
            "certificate_chain_validator": self._certificate_validator,
            "metadata_extractor": self._metadata_extractor,
            "fraud_pattern_detector": self._fraud_detector,
            "cross_reference_verifier": self._cross_reference,
            "compliance_reporter": self._reporter,
        }
        engine_status = {
            name: "initialized" if engine is not None else "not_available"
            for name, engine in engines.items()
        }
        count = self._count_initialized_engines()
        if count == _ENGINE_COUNT:
            status = "healthy"
        elif count > 0:
            status = "degraded"
        else:
            status = "unhealthy"
        return {
            "status": status,
            "initialized_count": count,
            "total_count": _ENGINE_COUNT,
            "engines": engine_status,
        }

    def _check_reference_data_health(self) -> Dict[str, Any]:
        """Check reference data availability."""
        loaded = sum(1 for x in [
            self._ref_document_templates,
            self._ref_trusted_cas,
            self._ref_fraud_rules,
        ] if x is not None)
        return {
            "status": "healthy" if loaded == 3 else "degraded",
            "loaded_datasets": loaded,
            "total_datasets": 3,
        }

    def _count_initialized_engines(self) -> int:
        """Count the number of successfully initialized engines."""
        engines = [
            self._classifier,
            self._signature_verifier,
            self._hash_validator,
            self._certificate_validator,
            self._metadata_extractor,
            self._fraud_detector,
            self._cross_reference,
            self._reporter,
        ]
        return sum(1 for e in engines if e is not None)

    # ------------------------------------------------------------------
    # Internal: Utility helpers
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        """Ensure the service has been started.

        Raises:
            RuntimeError: If the service has not been started.
        """
        if not self._started:
            raise RuntimeError(
                "DocumentAuthenticationService is not started. "
                "Call startup() first."
            )

    def _wrap_result(
        self,
        result: Any,
        start_time: float,
    ) -> Dict[str, Any]:
        """Wrap an engine result with processing time metadata.

        Args:
            result: Engine method result.
            start_time: Monotonic start time.

        Returns:
            Result with processing_time_ms added.
        """
        elapsed_ms = (time.monotonic() - start_time) * 1000
        if isinstance(result, dict):
            result["processing_time_ms"] = round(elapsed_ms, 2)
            return result
        return {
            "data": result,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def _safe_engine_call(
        self,
        engine: Optional[Any],
        method_name: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Safely delegate a call to an engine method.

        If the engine is None or the method does not exist, returns
        None without raising.

        Args:
            engine: Engine instance (may be None).
            method_name: Method to invoke on the engine.
            payload: Optional dictionary payload for the method.

        Returns:
            Engine method result dict, or None on failure.
        """
        if engine is None:
            return None
        try:
            method = getattr(engine, method_name, None)
            if method is None:
                return None
            if payload is not None:
                result = method(payload)
            else:
                result = method()
            if isinstance(result, dict):
                return result
            return None
        except Exception as exc:
            logger.debug(
                "Engine call fallback: %s.%s -> %s",
                type(engine).__name__, method_name, exc,
            )
            return None

    def _safe_engine_call_with_args(
        self,
        engine: Optional[Any],
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Safely delegate a call to an engine method with arguments.

        Args:
            engine: Engine instance (may be None).
            method_name: Method to invoke on the engine.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            Engine method result, or None on failure.
        """
        if engine is None:
            return None
        try:
            method = getattr(engine, method_name, None)
            if method is None:
                return None
            return method(*args, **kwargs)
        except Exception as exc:
            logger.debug(
                "Engine call fallback: %s.%s -> %s",
                type(engine).__name__, method_name, exc,
            )
            return None

    def _compute_overall_status(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute the overall verification status from engine results.

        Args:
            results: Combined results from all engines.

        Returns:
            Dictionary with overall pass/fail status and summary.
        """
        checks = {
            "classification": results.get("classification", {}),
            "signature": results.get("signature", {}),
            "hash": results.get("hash", {}),
            "certificate": results.get("certificate", {}),
            "metadata": results.get("metadata", {}),
            "fraud": results.get("fraud", {}),
            "cross_reference": results.get("cross_reference", {}),
        }

        # Count passed / failed / unavailable
        passed = 0
        failed = 0
        unavailable = 0
        for name, check_result in checks.items():
            if not isinstance(check_result, dict):
                unavailable += 1
                continue
            status = check_result.get("status", "unknown")
            if status == "engine_unavailable":
                unavailable += 1
            elif check_result.get("success", True):
                passed += 1
            else:
                failed += 1

        if failed > 0:
            overall = "FAIL"
        elif unavailable == len(checks):
            overall = "UNKNOWN"
        elif unavailable > 0:
            overall = "PARTIAL"
        else:
            overall = "PASS"

        return {
            "status": overall,
            "passed": passed,
            "failed": failed,
            "unavailable": unavailable,
            "total_checks": len(checks),
        }


# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for the Document Authentication service.

    Automatically starts the service on application startup and shuts it
    down on application shutdown.  The service instance is stored in
    ``app.state.dav_service`` for access from route handlers.

    Usage with FastAPI::

        from fastapi import FastAPI
        from greenlang.agents.eudr.document_authentication.setup import lifespan

        app = FastAPI(lifespan=lifespan)

    Args:
        app: The FastAPI application instance.

    Yields:
        None (service is accessible via ``app.state.dav_service``).
    """
    service = get_service()
    app.state.dav_service = service
    try:
        await service.startup()
        yield
    finally:
        await service.shutdown()


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_service_instance: Optional[DocumentAuthenticationService] = None
_service_lock = threading.Lock()


def get_service() -> DocumentAuthenticationService:
    """Return the singleton DocumentAuthenticationService instance.

    Uses double-checked locking for thread safety.  The instance is
    created on first call.

    Returns:
        DocumentAuthenticationService singleton instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = DocumentAuthenticationService()
    return _service_instance


def set_service(service: DocumentAuthenticationService) -> None:
    """Replace the singleton DocumentAuthenticationService instance.

    Primarily intended for testing and dependency injection.

    Args:
        service: Replacement service instance.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.info("DocumentAuthenticationService singleton replaced")


def reset_service() -> None:
    """Reset the singleton DocumentAuthenticationService to None.

    The next call to ``get_service()`` will create a fresh instance.
    Intended for test teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.debug("DocumentAuthenticationService singleton reset")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Service
    "DocumentAuthenticationService",
    "HealthResult",
    "lifespan",
    "get_service",
    "set_service",
    "reset_service",
    # Result containers
    "ClassificationResult",
    "SignatureResult",
    "HashResult",
    "CertificateResult",
    "MetadataResult",
    "FraudResult",
    "CrossRefResult",
    "ReportResult",
    "FullVerificationResult",
    "BatchResult",
    "DashboardResult",
]
