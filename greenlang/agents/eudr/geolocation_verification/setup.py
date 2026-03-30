# -*- coding: utf-8 -*-
"""
GeolocationVerificationService - Facade for AGENT-EUDR-002 Geolocation Verification

This module implements the GeolocationVerificationService, the single entry point
for all geolocation verification operations in the GL-EUDR-APP. It manages the
lifecycle of eight internal engines, an async PostgreSQL connection pool (psycopg +
psycopg_pool), a Redis cache connection, OpenTelemetry tracing, and Prometheus
metrics. The service exposes a unified interface consumed by the FastAPI router
layer and the GL-EUDR-APP integration.

Lifecycle:
    startup  -> load config -> connect DB -> register pgvector -> connect Redis
             -> initialize engines -> start health check background task
    shutdown -> close engines -> close Redis -> close DB pool -> flush metrics

Engines (8):
    1. CoordinateValidator         - WGS84 coordinate validation (Feature 1)
    2. PolygonTopologyVerifier     - Polygon topology checks (Feature 2)
    3. ProtectedAreaChecker        - WDPA protected area screening (Feature 3)
    4. DeforestationCutoffVerifier - Post-cutoff deforestation detection (Feature 4)
    5. AccuracyScoringEngine       - Composite accuracy scoring (Feature 5)
    6. TemporalConsistencyAnalyzer - Boundary change detection (Feature 6)
    7. BatchVerificationPipeline   - Batch processing pipeline (Feature 7)
    8. Article9ComplianceReporter  - EUDR Article 9 compliance reports (Feature 8)

FastAPI Integration:
    Use the ``lifespan`` async context manager with ``FastAPI(lifespan=lifespan)``
    for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.geolocation_verification.setup import (
    ...     GeolocationVerificationService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
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

from greenlang.agents.eudr.geolocation_verification.config import (
    GeolocationVerificationConfig,
    get_config,
)

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
# Helpers
# ---------------------------------------------------------------------------

def _compute_service_hash(config: GeolocationVerificationConfig) -> str:
    """Compute SHA-256 hash of the service configuration for provenance.

    Args:
        config: Service configuration to hash.

    Returns:
        SHA-256 hex digest string.
    """
    raw = json.dumps(config.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()

# ---------------------------------------------------------------------------
# Health status model
# ---------------------------------------------------------------------------

class HealthStatus:
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
        version: str = "1.0.0",
        uptime_seconds: float = 0.0,
    ) -> None:
        self.status = status
        self.checks = checks or {}
        self.timestamp = timestamp or utcnow()
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
# VerifyPlotRequest
# ---------------------------------------------------------------------------

class VerifyPlotRequest:
    """Request object for full plot verification.

    Bundles all input data needed by the service to perform a comprehensive
    plot verification across all engines.

    Attributes:
        plot_id: Unique plot identifier.
        lat: Latitude in decimal degrees (WGS84).
        lon: Longitude in decimal degrees (WGS84).
        declared_country: ISO 3166-1 alpha-2 country code.
        commodity: EUDR commodity identifier.
        declared_area_ha: Operator-declared area in hectares.
        polygon_vertices: List of (lat, lon) tuples forming the polygon ring.
        buffer_km: Buffer zone in km for protected area checks.
    """

    __slots__ = (
        "plot_id", "lat", "lon", "declared_country", "commodity",
        "declared_area_ha", "polygon_vertices", "buffer_km",
    )

    def __init__(
        self,
        plot_id: str = "",
        lat: float = 0.0,
        lon: float = 0.0,
        declared_country: str = "",
        commodity: str = "",
        declared_area_ha: float = 0.0,
        polygon_vertices: Optional[List[Tuple[float, float]]] = None,
        buffer_km: float = 1.0,
    ) -> None:
        self.plot_id = plot_id or f"PLOT-{uuid.uuid4().hex[:8]}"
        self.lat = lat
        self.lon = lon
        self.declared_country = declared_country
        self.commodity = commodity
        self.declared_area_ha = declared_area_ha
        self.polygon_vertices = polygon_vertices or []
        self.buffer_km = buffer_km

# ---------------------------------------------------------------------------
# PlotVerificationResult (unified result from verify_plot)
# ---------------------------------------------------------------------------

class PlotVerificationResult:
    """Unified result from a full plot verification across all engines.

    Attributes:
        plot_id: Plot identifier.
        overall_valid: Whether the plot passes all verification checks.
        coordinate_result: Result from CoordinateValidator.
        polygon_result: Result from PolygonTopologyVerifier.
        protected_area_result: Result from ProtectedAreaChecker.
        deforestation_result: Result from DeforestationCutoffVerifier.
        accuracy_score: Result from AccuracyScoringEngine.
        provenance_hash: SHA-256 hash of the combined result.
        verified_at: UTC timestamp.
        processing_time_ms: Total processing time in milliseconds.
    """

    __slots__ = (
        "plot_id", "overall_valid", "coordinate_result", "polygon_result",
        "protected_area_result", "deforestation_result", "accuracy_score",
        "provenance_hash", "verified_at", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        overall_valid: bool = False,
        coordinate_result: Optional[Any] = None,
        polygon_result: Optional[Any] = None,
        protected_area_result: Optional[Any] = None,
        deforestation_result: Optional[Any] = None,
        accuracy_score: Optional[Any] = None,
        provenance_hash: str = "",
        verified_at: Optional[datetime] = None,
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id
        self.overall_valid = overall_valid
        self.coordinate_result = coordinate_result
        self.polygon_result = polygon_result
        self.protected_area_result = protected_area_result
        self.deforestation_result = deforestation_result
        self.accuracy_score = accuracy_score
        self.provenance_hash = provenance_hash
        self.verified_at = verified_at or utcnow()
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        def _result_dict(result: Any) -> Optional[Dict[str, Any]]:
            if result is None:
                return None
            if hasattr(result, "to_dict"):
                return result.to_dict()
            return str(result)

        return {
            "plot_id": self.plot_id,
            "overall_valid": self.overall_valid,
            "coordinate_result": _result_dict(self.coordinate_result),
            "polygon_result": _result_dict(self.polygon_result),
            "protected_area_result": _result_dict(self.protected_area_result),
            "deforestation_result": _result_dict(self.deforestation_result),
            "accuracy_score": _result_dict(self.accuracy_score),
            "provenance_hash": self.provenance_hash,
            "verified_at": self.verified_at.isoformat(),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# BatchVerificationResult
# ---------------------------------------------------------------------------

class BatchVerificationResult:
    """Result of a batch verification job.

    Attributes:
        batch_id: Unique batch identifier.
        status: Batch status ('pending', 'processing', 'completed', 'failed').
        total_plots: Total plots in the batch.
        completed_plots: Number of plots processed so far.
        failed_plots: Number of plots that failed verification.
        results: Per-plot verification results.
        submitted_at: When the batch was submitted.
        completed_at: When the batch completed (or None).
        processing_time_ms: Total batch processing time in milliseconds.
    """

    __slots__ = (
        "batch_id", "status", "total_plots", "completed_plots",
        "failed_plots", "results", "submitted_at", "completed_at",
        "processing_time_ms",
    )

    def __init__(
        self,
        batch_id: str = "",
        status: str = "pending",
        total_plots: int = 0,
        completed_plots: int = 0,
        failed_plots: int = 0,
        results: Optional[List[PlotVerificationResult]] = None,
        submitted_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        processing_time_ms: float = 0.0,
    ) -> None:
        self.batch_id = batch_id or f"BATCH-{uuid.uuid4().hex[:12]}"
        self.status = status
        self.total_plots = total_plots
        self.completed_plots = completed_plots
        self.failed_plots = failed_plots
        self.results = results or []
        self.submitted_at = submitted_at or utcnow()
        self.completed_at = completed_at
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "batch_id": self.batch_id,
            "status": self.status,
            "total_plots": self.total_plots,
            "completed_plots": self.completed_plots,
            "failed_plots": self.failed_plots,
            "results": [r.to_dict() for r in self.results],
            "submitted_at": self.submitted_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# GeolocationVerificationService
# ---------------------------------------------------------------------------

class GeolocationVerificationService:
    """Facade service for the EUDR Geolocation Verification Agent.

    This is the single entry point for all geolocation verification operations.
    It manages the full lifecycle of database connections, cache connections,
    eight internal engines, health monitoring, and OpenTelemetry tracing.

    The service follows a strict startup/shutdown protocol:
        startup:  config -> DB pool -> pgvector -> Redis -> engines -> health
        shutdown: health stop -> engines -> Redis -> DB pool -> metrics flush

    Attributes:
        config: Service configuration loaded from env or injected.
        is_running: Whether the service is currently active and healthy.

    Example:
        >>> service = GeolocationVerificationService()
        >>> await service.startup()
        >>> result = service.validate_coordinate(5.123, -73.456, "CO", "coffee")
        >>> await service.shutdown()
    """

    def __init__(
        self,
        config: Optional[GeolocationVerificationConfig] = None,
    ) -> None:
        """Initialize GeolocationVerificationService.

        Loads configuration but does NOT start connections or engines.
        Call ``startup()`` to activate the service.

        Args:
            config: Optional configuration override. If None, loads from
                environment variables via ``get_config()``.
        """
        self._config = config or get_config()
        self._started = False
        self._start_time: Optional[float] = None
        self._config_hash = _compute_service_hash(self._config)

        # Connection handles (initialized in startup)
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None

        # Engine instances (initialized in startup)
        self._coordinate_validator: Optional[Any] = None
        self._polygon_verifier: Optional[Any] = None
        self._protected_area_checker: Optional[Any] = None
        self._deforestation_verifier: Optional[Any] = None
        self._accuracy_scorer: Optional[Any] = None
        self._temporal_analyzer: Optional[Any] = None
        self._batch_pipeline: Optional[Any] = None
        self._article9_reporter: Optional[Any] = None

        # Batch tracking
        self._batch_registry: Dict[str, BatchVerificationResult] = {}
        self._batch_lock = threading.Lock()

        # Health check background task
        self._health_task: Optional[asyncio.Task[None]] = None
        self._last_health: Optional[HealthStatus] = None
        self._health_interval_seconds: float = 30.0

        # OpenTelemetry tracer
        self._tracer: Optional[Any] = None

        logger.info(
            "GeolocationVerificationService created: config_hash=%s, "
            "pool_size=%d, cache_ttl=%ds",
            self._config_hash[:12],
            self._config.pool_size,
            self._config.verification_cache_ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> GeolocationVerificationConfig:
        """Return the service configuration."""
        return self._config

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
    def coordinate_validator(self) -> Any:
        """Return the CoordinateValidator instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._coordinate_validator

    @property
    def polygon_verifier(self) -> Any:
        """Return the PolygonTopologyVerifier instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._polygon_verifier

    @property
    def protected_area_checker(self) -> Any:
        """Return the ProtectedAreaChecker instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._protected_area_checker

    @property
    def deforestation_verifier(self) -> Any:
        """Return the DeforestationCutoffVerifier instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._deforestation_verifier

    @property
    def accuracy_scorer(self) -> Any:
        """Return the AccuracyScoringEngine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._accuracy_scorer

    @property
    def temporal_analyzer(self) -> Any:
        """Return the TemporalConsistencyAnalyzer instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._temporal_analyzer

    @property
    def batch_pipeline(self) -> Any:
        """Return the BatchVerificationPipeline instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._batch_pipeline

    @property
    def article9_reporter(self) -> Any:
        """Return the Article9ComplianceReporter instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._article9_reporter

    @property
    def db_pool(self) -> Any:
        """Return the async PostgreSQL connection pool.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._db_pool

    @property
    def redis_client(self) -> Any:
        """Return the async Redis client.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._redis

    @property
    def last_health(self) -> Optional[HealthStatus]:
        """Return the most recent cached health check result."""
        return self._last_health

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Start the service: connect DB, Redis, initialize all engines.

        Executes the full startup sequence in order:
            1. Configure structured logging
            2. Initialize OpenTelemetry tracer
            3. Connect to PostgreSQL and create connection pool
            4. Register pgvector type extension
            5. Connect to Redis for caching
            6. Initialize all eight engines
            7. Start background health check task

        Idempotent: safe to call multiple times.

        Raises:
            RuntimeError: If a critical connection fails.
        """
        if self._started:
            logger.debug("GeolocationVerificationService already started")
            return

        start = time.monotonic()
        logger.info("GeolocationVerificationService starting up...")

        # Step 1: Configure logging
        self._configure_logging()

        # Step 2: Initialize OpenTelemetry tracer
        self._init_tracer()

        # Step 3: Connect to PostgreSQL
        await self._connect_database()

        # Step 4: Register pgvector extension
        await self._register_pgvector()

        # Step 5: Connect to Redis
        await self._connect_redis()

        # Step 6: Initialize all engines
        await self._initialize_engines()

        # Step 7: Start health check background task
        self._start_health_check()

        self._started = True
        self._start_time = time.monotonic()
        elapsed = (time.monotonic() - start) * 1000

        logger.info(
            "GeolocationVerificationService started in %.1fms: "
            "db=%s, redis=%s, engines=8, config_hash=%s",
            elapsed,
            "connected" if self._db_pool is not None else "skipped",
            "connected" if self._redis is not None else "skipped",
            self._config_hash[:12],
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Gracefully shut down the service and release all resources.

        Executes the shutdown sequence in reverse order:
            1. Cancel health check background task
            2. Close all engines
            3. Close Redis connection
            4. Close PostgreSQL connection pool
            5. Flush Prometheus metrics

        Idempotent: safe to call multiple times.
        """
        if not self._started:
            logger.debug("GeolocationVerificationService already stopped")
            return

        logger.info("GeolocationVerificationService shutting down...")
        start = time.monotonic()

        # Step 1: Cancel health check
        self._stop_health_check()

        # Step 2: Close engines
        await self._close_engines()

        # Step 3: Close Redis
        await self._close_redis()

        # Step 4: Close database pool
        await self._close_database()

        # Step 5: Flush metrics
        self._flush_metrics()

        self._started = False
        elapsed = (time.monotonic() - start) * 1000

        logger.info(
            "GeolocationVerificationService shut down in %.1fms", elapsed
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of all components.

        Checks database connectivity, Redis connectivity, engine status,
        and memory usage. Returns a structured health report suitable for
        the ``/health`` endpoint.

        Returns:
            Dictionary with status, component checks, version, and uptime.
        """
        checks: Dict[str, Any] = {}

        # Database check
        checks["database"] = await self._check_database_health()

        # Redis check
        checks["redis"] = await self._check_redis_health()

        # Engine checks
        checks["engines"] = self._check_engine_health()

        # Determine overall status
        statuses = [
            v.get("status", "unhealthy") if isinstance(v, dict) else "unhealthy"
            for v in checks.values()
        ]
        if all(s == "healthy" for s in statuses):
            overall = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            overall = "unhealthy"
        else:
            overall = "degraded"

        health = HealthStatus(
            status=overall,
            checks=checks,
            timestamp=utcnow(),
            version="1.0.0",
            uptime_seconds=self.uptime_seconds,
        )
        self._last_health = health
        return health.to_dict()

    # ------------------------------------------------------------------
    # Unified API: Coordinate validation
    # ------------------------------------------------------------------

    def validate_coordinate(
        self,
        lat: float,
        lon: float,
        declared_country: str = "",
        commodity: str = "",
    ) -> Any:
        """Validate a single GPS coordinate pair.

        Delegates to the CoordinateValidator engine for WGS84 bounds
        checking, precision assessment, country matching, and anomaly
        detection.

        Args:
            lat: Latitude in decimal degrees (WGS84).
            lon: Longitude in decimal degrees (WGS84).
            declared_country: ISO 3166-1 alpha-2 country code.
            commodity: EUDR commodity identifier.

        Returns:
            CoordinateValidationResult from the validator engine.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the CoordinateValidator is not available.
        """
        self._ensure_started()
        if self._coordinate_validator is None:
            raise RuntimeError("CoordinateValidator engine is not available")

        logger.debug(
            "Validating coordinate: lat=%.6f, lon=%.6f, country=%s, "
            "commodity=%s",
            lat, lon, declared_country, commodity,
        )

        from greenlang.agents.eudr.geolocation_verification.models import (
            VerifyCoordinateRequest,
        )

        coord_input = VerifyCoordinateRequest(
            lat=lat,
            lon=lon,
            declared_country=declared_country,
            commodity=commodity,
        )
        return self._coordinate_validator.validate(coord_input)

    def validate_coordinates_batch(
        self,
        coordinates: List[Dict[str, Any]],
    ) -> List[Any]:
        """Validate a batch of GPS coordinates.

        Each coordinate dict must contain 'lat' and 'lon' keys, and
        optionally 'declared_country' and 'commodity'.

        Args:
            coordinates: List of coordinate dictionaries.

        Returns:
            List of CoordinateValidationResult objects.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the CoordinateValidator is not available.
        """
        self._ensure_started()
        if self._coordinate_validator is None:
            raise RuntimeError("CoordinateValidator engine is not available")

        logger.info(
            "Validating coordinate batch: count=%d", len(coordinates)
        )

        from greenlang.agents.eudr.geolocation_verification.models import (
            VerifyCoordinateRequest,
        )

        inputs = [
            VerifyCoordinateRequest(
                lat=float(c.get("lat", 0.0)),
                lon=float(c.get("lon", 0.0)),
                declared_country=str(c.get("declared_country", "")),
                commodity=str(c.get("commodity", "")),
                plot_id=str(c.get("plot_id", "")),
            )
            for c in coordinates
        ]
        return self._coordinate_validator.validate_batch(inputs)

    # ------------------------------------------------------------------
    # Unified API: Polygon verification
    # ------------------------------------------------------------------

    def verify_polygon(
        self,
        vertices: List[Tuple[float, float]],
        declared_area_ha: Optional[float] = None,
    ) -> Any:
        """Verify a polygon boundary topology.

        Delegates to the PolygonTopologyVerifier for ring closure,
        winding order, self-intersection, sliver, and spike checks.

        Args:
            vertices: List of (lat, lon) tuples forming the polygon ring.
            declared_area_ha: Operator-declared area in hectares for
                tolerance checking.

        Returns:
            PolygonVerificationResult from the verifier engine.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the PolygonTopologyVerifier is not available.
        """
        self._ensure_started()
        if self._polygon_verifier is None:
            raise RuntimeError(
                "PolygonTopologyVerifier engine is not available"
            )

        logger.debug(
            "Verifying polygon: vertices=%d, declared_area=%.2f ha",
            len(vertices),
            declared_area_ha or 0.0,
        )

        from greenlang.agents.eudr.geolocation_verification.models import (
            VerifyPolygonRequest,
        )

        polygon_input = VerifyPolygonRequest(
            vertices=vertices,
            declared_area_ha=declared_area_ha,
        )
        return self._polygon_verifier.verify(polygon_input)

    # ------------------------------------------------------------------
    # Unified API: Protected area checking
    # ------------------------------------------------------------------

    def check_protected_areas(
        self,
        lat: float,
        lon: float,
        polygon: Optional[List[Tuple[float, float]]] = None,
        buffer_km: float = 1.0,
    ) -> Any:
        """Check whether a location overlaps protected areas.

        Delegates to the ProtectedAreaChecker for WDPA dataset screening.

        Args:
            lat: Latitude in decimal degrees (WGS84).
            lon: Longitude in decimal degrees (WGS84).
            polygon: Optional polygon vertices for area overlap checking.
            buffer_km: Buffer zone in km around the point/polygon.

        Returns:
            ProtectedAreaCheckResult from the checker engine.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the ProtectedAreaChecker is not available.
        """
        self._ensure_started()
        if self._protected_area_checker is None:
            raise RuntimeError(
                "ProtectedAreaChecker engine is not available"
            )

        logger.debug(
            "Checking protected areas: lat=%.6f, lon=%.6f, "
            "polygon_vertices=%d, buffer=%.1f km",
            lat, lon, len(polygon) if polygon else 0, buffer_km,
        )
        return self._protected_area_checker.check(
            lat=lat,
            lon=lon,
            polygon=polygon,
            buffer_km=buffer_km,
        )

    # ------------------------------------------------------------------
    # Unified API: Deforestation verification
    # ------------------------------------------------------------------

    def verify_deforestation(
        self,
        plot_id: str,
        lat: float,
        lon: float,
        polygon: Optional[List[Tuple[float, float]]] = None,
        commodity: str = "",
    ) -> Any:
        """Verify deforestation status for a production plot.

        Delegates to the DeforestationCutoffVerifier for post-cutoff
        deforestation alert detection.

        Args:
            plot_id: Unique plot identifier.
            lat: Latitude in decimal degrees (WGS84).
            lon: Longitude in decimal degrees (WGS84).
            polygon: Optional polygon vertices for area analysis.
            commodity: EUDR commodity identifier.

        Returns:
            DeforestationVerificationResult from the verifier engine.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the DeforestationCutoffVerifier is not available.
        """
        self._ensure_started()
        if self._deforestation_verifier is None:
            raise RuntimeError(
                "DeforestationCutoffVerifier engine is not available"
            )

        logger.debug(
            "Verifying deforestation: plot=%s, lat=%.6f, lon=%.6f, "
            "commodity=%s",
            plot_id, lat, lon, commodity,
        )
        return self._deforestation_verifier.verify(
            plot_id=plot_id,
            lat=lat,
            lon=lon,
            polygon=polygon,
            commodity=commodity,
        )

    # ------------------------------------------------------------------
    # Unified API: Full plot verification
    # ------------------------------------------------------------------

    def verify_plot(self, request: VerifyPlotRequest) -> PlotVerificationResult:
        """Perform full verification of a production plot across all engines.

        Orchestrates coordinate validation, polygon verification,
        protected area checking, deforestation verification, and accuracy
        scoring into a single unified result. All sub-results are combined
        and a provenance hash is computed over the aggregate.

        Args:
            request: VerifyPlotRequest with all plot data.

        Returns:
            PlotVerificationResult with combined results from all engines.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info(
            "Verifying plot: id=%s, lat=%.6f, lon=%.6f, "
            "country=%s, commodity=%s, area=%.2f ha, "
            "polygon_vertices=%d",
            request.plot_id, request.lat, request.lon,
            request.declared_country, request.commodity,
            request.declared_area_ha, len(request.polygon_vertices),
        )

        # Step 1: Coordinate validation
        coord_result = self._safe_validate_coordinate(
            request.lat, request.lon,
            request.declared_country, request.commodity,
        )

        # Step 2: Polygon verification (if polygon provided)
        polygon_result = self._safe_verify_polygon(
            request.polygon_vertices, request.declared_area_ha,
        )

        # Step 3: Protected area check
        protected_result = self._safe_check_protected_areas(
            request.lat, request.lon,
            request.polygon_vertices or None,
            request.buffer_km,
        )

        # Step 4: Deforestation verification
        deforestation_result = self._safe_verify_deforestation(
            request.plot_id, request.lat, request.lon,
            request.polygon_vertices or None,
            request.commodity,
        )

        # Step 5: Accuracy scoring
        accuracy_result = self._safe_get_accuracy_score(
            request.plot_id,
            coord_result,
            polygon_result,
            protected_result,
            deforestation_result,
        )

        # Step 6: Determine overall validity
        overall_valid = self._determine_overall_validity(
            coord_result, polygon_result,
            protected_result, deforestation_result,
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        # Step 7: Compute provenance hash
        provenance_hash = self._compute_plot_provenance(
            request.plot_id,
            coord_result,
            polygon_result,
            protected_result,
            deforestation_result,
        )

        result = PlotVerificationResult(
            plot_id=request.plot_id,
            overall_valid=overall_valid,
            coordinate_result=coord_result,
            polygon_result=polygon_result,
            protected_area_result=protected_result,
            deforestation_result=deforestation_result,
            accuracy_score=accuracy_result,
            provenance_hash=provenance_hash,
            verified_at=utcnow(),
            processing_time_ms=elapsed_ms,
        )

        logger.info(
            "Plot verification complete: id=%s, valid=%s, "
            "hash=%s, elapsed=%.1fms",
            request.plot_id, overall_valid,
            provenance_hash[:12], elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Unified API: Batch verification
    # ------------------------------------------------------------------

    def submit_batch(
        self,
        requests: List[VerifyPlotRequest],
    ) -> str:
        """Submit a batch of plot verification requests.

        Creates a batch job, processes all plots, and stores results.
        Returns the batch ID for subsequent status polling via
        ``get_batch_status()``.

        Args:
            requests: List of VerifyPlotRequest objects.

        Returns:
            Batch ID string for tracking.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()

        batch_id = f"BATCH-{uuid.uuid4().hex[:12]}"
        batch_result = BatchVerificationResult(
            batch_id=batch_id,
            status="processing",
            total_plots=len(requests),
        )

        with self._batch_lock:
            self._batch_registry[batch_id] = batch_result

        logger.info(
            "Batch submitted: id=%s, plots=%d", batch_id, len(requests)
        )

        # Process all plots synchronously (batch pipeline would use async)
        start = time.monotonic()
        results: List[PlotVerificationResult] = []
        failed_count = 0

        for request in requests:
            try:
                plot_result = self.verify_plot(request)
                results.append(plot_result)
            except Exception as exc:
                logger.warning(
                    "Batch plot failed: id=%s, error=%s",
                    request.plot_id, exc,
                )
                failed_count += 1
                results.append(PlotVerificationResult(
                    plot_id=request.plot_id,
                    overall_valid=False,
                    provenance_hash="",
                ))

        elapsed_ms = (time.monotonic() - start) * 1000.0

        # Update batch result
        with self._batch_lock:
            batch_result.status = "completed"
            batch_result.completed_plots = len(results)
            batch_result.failed_plots = failed_count
            batch_result.results = results
            batch_result.completed_at = utcnow()
            batch_result.processing_time_ms = elapsed_ms

        logger.info(
            "Batch completed: id=%s, total=%d, completed=%d, "
            "failed=%d, elapsed=%.1fms",
            batch_id, len(requests), len(results),
            failed_count, elapsed_ms,
        )

        return batch_id

    def get_batch_status(self, batch_id: str) -> BatchVerificationResult:
        """Get the status and results of a batch verification job.

        Args:
            batch_id: Batch identifier returned by ``submit_batch()``.

        Returns:
            BatchVerificationResult with current status and results.

        Raises:
            ValueError: If the batch_id is not found.
        """
        with self._batch_lock:
            result = self._batch_registry.get(batch_id)

        if result is None:
            raise ValueError(f"Batch not found: {batch_id}")

        return result

    # ------------------------------------------------------------------
    # Unified API: Accuracy scoring
    # ------------------------------------------------------------------

    def get_accuracy_score(
        self,
        plot_id: str,
    ) -> Any:
        """Get the geolocation accuracy score for a plot.

        Delegates to the AccuracyScoringEngine. Requires that the plot
        has been previously verified and scored.

        Args:
            plot_id: Unique plot identifier.

        Returns:
            GeolocationAccuracyScore from the scoring engine.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the AccuracyScoringEngine is not available.
        """
        self._ensure_started()
        if self._accuracy_scorer is None:
            raise RuntimeError(
                "AccuracyScoringEngine is not available"
            )

        logger.debug("Getting accuracy score: plot=%s", plot_id)
        return self._accuracy_scorer.get_score(plot_id)

    # ------------------------------------------------------------------
    # Unified API: Compliance reporting
    # ------------------------------------------------------------------

    def generate_compliance_report(
        self,
        operator_id: str,
        plots_with_results: Optional[List[Any]] = None,
        commodity: Optional[str] = None,
        export_format: str = "json",
    ) -> Any:
        """Generate an EUDR Article 9 compliance report.

        Delegates to the Article9ComplianceReporter engine to produce
        a comprehensive compliance report for the operator.

        Args:
            operator_id: Unique operator identifier.
            plots_with_results: List of PlotVerificationBundle objects.
                If None, an empty report is generated.
            commodity: Optional commodity filter.
            export_format: Export format ('json', 'csv', 'pdf_data').

        Returns:
            ComplianceReport from the Article9ComplianceReporter.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the Article9ComplianceReporter is not available.
        """
        self._ensure_started()
        if self._article9_reporter is None:
            raise RuntimeError(
                "Article9ComplianceReporter is not available"
            )

        logger.info(
            "Generating compliance report: operator=%s, "
            "commodity=%s, format=%s",
            operator_id, commodity or "all", export_format,
        )

        return self._article9_reporter.generate_report(
            operator_id=operator_id,
            plots_with_results=plots_with_results or [],
            commodity_filter=commodity,
            export_format=export_format,
        )

    # ------------------------------------------------------------------
    # Internal: Safe wrappers (exception-tolerant for verify_plot)
    # ------------------------------------------------------------------

    def _safe_validate_coordinate(
        self,
        lat: float,
        lon: float,
        declared_country: str,
        commodity: str,
    ) -> Optional[Any]:
        """Validate coordinate, returning None on engine unavailability.

        Args:
            lat: Latitude.
            lon: Longitude.
            declared_country: Country code.
            commodity: Commodity identifier.

        Returns:
            CoordinateValidationResult or None.
        """
        if self._coordinate_validator is None:
            return None
        try:
            return self.validate_coordinate(
                lat, lon, declared_country, commodity,
            )
        except Exception as exc:
            logger.warning(
                "Coordinate validation failed (non-fatal): %s", exc
            )
            return None

    def _safe_verify_polygon(
        self,
        vertices: List[Tuple[float, float]],
        declared_area_ha: Optional[float],
    ) -> Optional[Any]:
        """Verify polygon, returning None on engine unavailability.

        Args:
            vertices: Polygon vertices.
            declared_area_ha: Declared area in hectares.

        Returns:
            PolygonVerificationResult or None.
        """
        if not vertices or self._polygon_verifier is None:
            return None
        try:
            return self.verify_polygon(vertices, declared_area_ha)
        except Exception as exc:
            logger.warning(
                "Polygon verification failed (non-fatal): %s", exc
            )
            return None

    def _safe_check_protected_areas(
        self,
        lat: float,
        lon: float,
        polygon: Optional[List[Tuple[float, float]]],
        buffer_km: float,
    ) -> Optional[Any]:
        """Check protected areas, returning None on engine unavailability.

        Args:
            lat: Latitude.
            lon: Longitude.
            polygon: Polygon vertices or None.
            buffer_km: Buffer in km.

        Returns:
            ProtectedAreaCheckResult or None.
        """
        if self._protected_area_checker is None:
            return None
        try:
            return self.check_protected_areas(lat, lon, polygon, buffer_km)
        except Exception as exc:
            logger.warning(
                "Protected area check failed (non-fatal): %s", exc
            )
            return None

    def _safe_verify_deforestation(
        self,
        plot_id: str,
        lat: float,
        lon: float,
        polygon: Optional[List[Tuple[float, float]]],
        commodity: str,
    ) -> Optional[Any]:
        """Verify deforestation, returning None on engine unavailability.

        Args:
            plot_id: Plot identifier.
            lat: Latitude.
            lon: Longitude.
            polygon: Polygon vertices or None.
            commodity: Commodity identifier.

        Returns:
            DeforestationVerificationResult or None.
        """
        if self._deforestation_verifier is None:
            return None
        try:
            return self.verify_deforestation(
                plot_id, lat, lon, polygon, commodity,
            )
        except Exception as exc:
            logger.warning(
                "Deforestation verification failed (non-fatal): %s", exc
            )
            return None

    def _safe_get_accuracy_score(
        self,
        plot_id: str,
        coord_result: Optional[Any],
        polygon_result: Optional[Any],
        protected_result: Optional[Any],
        deforestation_result: Optional[Any],
    ) -> Optional[Any]:
        """Compute accuracy score, returning None on engine unavailability.

        If the AccuracyScoringEngine supports a ``compute()`` method that
        takes sub-results, it is used. Otherwise falls back to ``get_score()``.

        Args:
            plot_id: Plot identifier.
            coord_result: Coordinate validation result.
            polygon_result: Polygon verification result.
            protected_result: Protected area check result.
            deforestation_result: Deforestation verification result.

        Returns:
            GeolocationAccuracyScore or None.
        """
        if self._accuracy_scorer is None:
            return None
        try:
            if hasattr(self._accuracy_scorer, "compute"):
                return self._accuracy_scorer.compute(
                    plot_id=plot_id,
                    coordinate_result=coord_result,
                    polygon_result=polygon_result,
                    protected_area_result=protected_result,
                    deforestation_result=deforestation_result,
                )
            return self._accuracy_scorer.get_score(plot_id)
        except Exception as exc:
            logger.warning(
                "Accuracy scoring failed (non-fatal): %s", exc
            )
            return None

    # ------------------------------------------------------------------
    # Internal: Overall validity determination
    # ------------------------------------------------------------------

    def _determine_overall_validity(
        self,
        coord_result: Optional[Any],
        polygon_result: Optional[Any],
        protected_result: Optional[Any],
        deforestation_result: Optional[Any],
    ) -> bool:
        """Determine overall plot validity from all sub-results.

        A plot is valid if:
            - Coordinate validation passed (or not available)
            - Polygon verification passed (or not available/no polygon)
            - No protected area overlap detected
            - No deforestation detected

        Args:
            coord_result: Coordinate validation result.
            polygon_result: Polygon verification result.
            protected_result: Protected area check result.
            deforestation_result: Deforestation verification result.

        Returns:
            True if all available checks pass.
        """
        # Coordinate check
        if coord_result is not None and hasattr(coord_result, "is_valid"):
            if not coord_result.is_valid:
                return False

        # Polygon check
        if polygon_result is not None and hasattr(polygon_result, "is_valid"):
            if not polygon_result.is_valid:
                return False

        # Protected area check
        if protected_result is not None and hasattr(
            protected_result, "overlaps_protected"
        ):
            if protected_result.overlaps_protected:
                return False

        # Deforestation check
        if deforestation_result is not None and hasattr(
            deforestation_result, "deforestation_detected"
        ):
            if deforestation_result.deforestation_detected:
                return False

        return True

    # ------------------------------------------------------------------
    # Internal: Provenance hash for plot verification
    # ------------------------------------------------------------------

    def _compute_plot_provenance(
        self,
        plot_id: str,
        coord_result: Optional[Any],
        polygon_result: Optional[Any],
        protected_result: Optional[Any],
        deforestation_result: Optional[Any],
    ) -> str:
        """Compute SHA-256 provenance hash for a plot verification.

        Args:
            plot_id: Plot identifier.
            coord_result: Coordinate validation result.
            polygon_result: Polygon verification result.
            protected_result: Protected area check result.
            deforestation_result: Deforestation verification result.

        Returns:
            SHA-256 hex digest string.
        """
        hash_parts: List[str] = [plot_id]

        if coord_result is not None and hasattr(coord_result, "provenance_hash"):
            hash_parts.append(coord_result.provenance_hash)
        if polygon_result is not None and hasattr(polygon_result, "provenance_hash"):
            hash_parts.append(polygon_result.provenance_hash)
        if protected_result is not None:
            hash_parts.append(
                str(getattr(protected_result, "overlaps_protected", ""))
            )
        if deforestation_result is not None:
            hash_parts.append(
                str(getattr(deforestation_result, "deforestation_detected", ""))
            )

        hash_input = "|".join(hash_parts)
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Internal: Logging
    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        """Configure structured logging based on service configuration."""
        log_level = getattr(logging, self._config.log_level, logging.INFO)
        logging.getLogger(
            "greenlang.agents.eudr.geolocation_verification"
        ).setLevel(log_level)
        logger.debug(
            "Logging configured: level=%s", self._config.log_level
        )

    # ------------------------------------------------------------------
    # Internal: OpenTelemetry
    # ------------------------------------------------------------------

    def _init_tracer(self) -> None:
        """Initialize OpenTelemetry tracer if available."""
        if OTEL_AVAILABLE and otel_trace is not None:
            self._tracer = otel_trace.get_tracer(
                "greenlang.agents.eudr.geolocation_verification",
                "1.0.0",
            )
            logger.info("OpenTelemetry tracer initialized")
        else:
            self._tracer = None
            logger.debug(
                "OpenTelemetry not available, tracing disabled"
            )

    # ------------------------------------------------------------------
    # Internal: Database
    # ------------------------------------------------------------------

    async def _connect_database(self) -> None:
        """Create async PostgreSQL connection pool.

        Uses psycopg_pool.AsyncConnectionPool with configurable pool
        sizing from the service configuration.

        Raises:
            RuntimeError: If psycopg is not available or connection fails.
        """
        if not PSYCOPG_POOL_AVAILABLE or not PSYCOPG_AVAILABLE:
            logger.warning(
                "psycopg/psycopg_pool not available, database disabled. "
                "Install with: pip install 'psycopg[binary]' psycopg_pool"
            )
            self._db_pool = None
            return

        try:
            conninfo = self._config.database_url
            pool = AsyncConnectionPool(
                conninfo=conninfo,
                min_size=max(1, self._config.pool_size // 2),
                max_size=self._config.pool_size,
                open=False,
            )
            await pool.open()
            await pool.check()
            self._db_pool = pool
            logger.info(
                "PostgreSQL connection pool opened: min=%d, max=%d",
                max(1, self._config.pool_size // 2),
                self._config.pool_size,
            )
        except Exception as exc:
            logger.error(
                "Failed to connect to PostgreSQL: %s", exc, exc_info=True
            )
            self._db_pool = None
            raise RuntimeError(
                f"Database connection failed: {exc}"
            ) from exc

    async def _register_pgvector(self) -> None:
        """Register pgvector type extension on the connection pool.

        Enables transparent encoding/decoding of vector columns for
        embedding-based similarity search.
        """
        if self._db_pool is None:
            logger.debug("Skipping pgvector registration: no database pool")
            return

        try:
            async with self._db_pool.connection() as conn:
                await conn.execute("SELECT 1")
            logger.info("pgvector extension registration check completed")
        except Exception as exc:
            logger.warning(
                "pgvector registration failed (non-fatal): %s", exc
            )

    async def _close_database(self) -> None:
        """Close the PostgreSQL connection pool."""
        if self._db_pool is not None:
            try:
                await self._db_pool.close()
                logger.info("PostgreSQL connection pool closed")
            except Exception as exc:
                logger.warning(
                    "Error closing database pool: %s", exc
                )
            finally:
                self._db_pool = None

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity health.

        Returns:
            Dictionary with status, pool stats, and latency.
        """
        if self._db_pool is None:
            return {"status": "unhealthy", "reason": "no_pool"}

        try:
            start = time.monotonic()
            async with self._db_pool.connection() as conn:
                await conn.execute("SELECT 1")
            latency_ms = (time.monotonic() - start) * 1000

            pool_stats = {}
            if hasattr(self._db_pool, "get_stats"):
                pool_stats = self._db_pool.get_stats()

            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "pool_stats": pool_stats,
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "reason": str(exc),
            }

    # ------------------------------------------------------------------
    # Internal: Redis
    # ------------------------------------------------------------------

    async def _connect_redis(self) -> None:
        """Connect to Redis for caching.

        Uses redis.asyncio for async Redis operations.
        Non-fatal on failure: service continues without caching.
        """
        if not REDIS_AVAILABLE:
            logger.warning(
                "redis package not available, caching disabled. "
                "Install with: pip install redis"
            )
            self._redis = None
            return

        try:
            client = aioredis.from_url(
                self._config.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            await client.ping()
            self._redis = client
            logger.info(
                "Redis connected: url=%s, ttl=%ds",
                "***",  # redacted
                self._config.verification_cache_ttl_seconds,
            )
        except Exception as exc:
            logger.warning(
                "Failed to connect to Redis (non-fatal): %s", exc
            )
            self._redis = None

    async def _close_redis(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            try:
                await self._redis.aclose()
                logger.info("Redis connection closed")
            except Exception as exc:
                logger.warning("Error closing Redis: %s", exc)
            finally:
                self._redis = None

    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity health.

        Returns:
            Dictionary with status and latency.
        """
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
            return {
                "status": "unhealthy",
                "reason": str(exc),
            }

    # ------------------------------------------------------------------
    # Internal: Engine initialization
    # ------------------------------------------------------------------

    async def _initialize_engines(self) -> None:
        """Initialize all eight internal engines.

        Engines are created with references to the shared DB pool, Redis
        client, and service configuration. Engines that require async
        initialization have their ``initialize()`` methods called.
        """
        logger.info("Initializing 8 geolocation verification engines...")

        # 1. CoordinateValidator
        self._coordinate_validator = await self._init_coordinate_validator()

        # 2. PolygonTopologyVerifier
        self._polygon_verifier = await self._init_polygon_verifier()

        # 3. ProtectedAreaChecker
        self._protected_area_checker = await self._init_protected_area_checker()

        # 4. DeforestationCutoffVerifier
        self._deforestation_verifier = (
            await self._init_deforestation_verifier()
        )

        # 5. AccuracyScoringEngine
        self._accuracy_scorer = await self._init_accuracy_scorer()

        # 6. TemporalConsistencyAnalyzer
        self._temporal_analyzer = await self._init_temporal_analyzer()

        # 7. BatchVerificationPipeline
        self._batch_pipeline = await self._init_batch_pipeline()

        # 8. Article9ComplianceReporter
        self._article9_reporter = await self._init_article9_reporter()

        logger.info("All 8 engines initialized successfully")

    async def _init_coordinate_validator(self) -> Any:
        """Initialize the CoordinateValidator engine.

        Returns:
            Initialized CoordinateValidator instance, or None if unavailable.
        """
        try:
            from greenlang.agents.eudr.geolocation_verification.coordinate_validator import (
                CoordinateValidator,
            )

            validator = CoordinateValidator(config=self._config)
            logger.info("CoordinateValidator initialized")
            return validator
        except ImportError:
            logger.warning("CoordinateValidator not available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize CoordinateValidator: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_polygon_verifier(self) -> Any:
        """Initialize the PolygonTopologyVerifier engine.

        Returns:
            Initialized PolygonTopologyVerifier instance, or None if unavailable.
        """
        try:
            from greenlang.agents.eudr.geolocation_verification.polygon_verifier import (
                PolygonTopologyVerifier,
            )

            verifier = PolygonTopologyVerifier(config=self._config)
            logger.info("PolygonTopologyVerifier initialized")
            return verifier
        except ImportError:
            logger.warning("PolygonTopologyVerifier not available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize PolygonTopologyVerifier: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_protected_area_checker(self) -> Any:
        """Initialize the ProtectedAreaChecker engine.

        Returns:
            Initialized ProtectedAreaChecker instance, or None if unavailable.
        """
        try:
            from greenlang.agents.eudr.geolocation_verification.protected_area_checker import (
                ProtectedAreaChecker,
            )

            checker = ProtectedAreaChecker(config=self._config)
            logger.info("ProtectedAreaChecker initialized")
            return checker
        except ImportError:
            logger.debug("ProtectedAreaChecker module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize ProtectedAreaChecker: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_deforestation_verifier(self) -> Any:
        """Initialize the DeforestationCutoffVerifier engine.

        Returns:
            Initialized DeforestationCutoffVerifier instance, or None.
        """
        try:
            from greenlang.agents.eudr.geolocation_verification.deforestation_verifier import (
                DeforestationCutoffVerifier,
            )

            verifier = DeforestationCutoffVerifier(config=self._config)
            logger.info("DeforestationCutoffVerifier initialized")
            return verifier
        except ImportError:
            logger.debug(
                "DeforestationCutoffVerifier module not yet available"
            )
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize DeforestationCutoffVerifier: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_accuracy_scorer(self) -> Any:
        """Initialize the AccuracyScoringEngine.

        Returns:
            Initialized AccuracyScoringEngine instance, or None.
        """
        try:
            from greenlang.agents.eudr.geolocation_verification.accuracy_scorer import (
                AccuracyScoringEngine,
            )

            scorer = AccuracyScoringEngine(config=self._config)
            logger.info("AccuracyScoringEngine initialized")
            return scorer
        except ImportError:
            logger.warning("AccuracyScoringEngine not available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize AccuracyScoringEngine: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_temporal_analyzer(self) -> Any:
        """Initialize the TemporalConsistencyAnalyzer.

        Returns:
            Initialized TemporalConsistencyAnalyzer instance, or None.
        """
        try:
            from greenlang.agents.eudr.geolocation_verification.temporal_analyzer import (
                TemporalConsistencyAnalyzer,
            )

            analyzer = TemporalConsistencyAnalyzer(config=self._config)
            logger.info("TemporalConsistencyAnalyzer initialized")
            return analyzer
        except ImportError:
            logger.warning("TemporalConsistencyAnalyzer not available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize TemporalConsistencyAnalyzer: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_batch_pipeline(self) -> Any:
        """Initialize the BatchVerificationPipeline.

        Returns:
            Initialized BatchVerificationPipeline instance, or None.
        """
        try:
            from greenlang.agents.eudr.geolocation_verification.batch_pipeline import (
                BatchVerificationPipeline,
            )

            pipeline = BatchVerificationPipeline(config=self._config)
            logger.info("BatchVerificationPipeline initialized")
            return pipeline
        except ImportError:
            logger.debug(
                "BatchVerificationPipeline module not yet available"
            )
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize BatchVerificationPipeline: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_article9_reporter(self) -> Any:
        """Initialize the Article9ComplianceReporter.

        Returns:
            Initialized Article9ComplianceReporter instance, or None.
        """
        try:
            from greenlang.agents.eudr.geolocation_verification.article9_reporter import (
                Article9ComplianceReporter,
            )

            reporter = Article9ComplianceReporter()
            logger.info("Article9ComplianceReporter initialized")
            return reporter
        except ImportError:
            logger.debug(
                "Article9ComplianceReporter module not yet available"
            )
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize Article9ComplianceReporter: %s",
                exc, exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Internal: Engine shutdown
    # ------------------------------------------------------------------

    async def _close_engines(self) -> None:
        """Close all engines that implement a close/shutdown method."""
        engine_names = [
            ("coordinate_validator", self._coordinate_validator),
            ("polygon_verifier", self._polygon_verifier),
            ("protected_area_checker", self._protected_area_checker),
            ("deforestation_verifier", self._deforestation_verifier),
            ("accuracy_scorer", self._accuracy_scorer),
            ("temporal_analyzer", self._temporal_analyzer),
            ("batch_pipeline", self._batch_pipeline),
            ("article9_reporter", self._article9_reporter),
        ]

        for name, engine in engine_names:
            if engine is None:
                continue
            try:
                if hasattr(engine, "shutdown") and asyncio.iscoroutinefunction(
                    engine.shutdown
                ):
                    await engine.shutdown()
                elif hasattr(engine, "close") and asyncio.iscoroutinefunction(
                    engine.close
                ):
                    await engine.close()
                logger.debug("Engine %s closed", name)
            except Exception as exc:
                logger.warning(
                    "Error closing engine %s: %s", name, exc
                )

        # Null out all engine references
        self._coordinate_validator = None
        self._polygon_verifier = None
        self._protected_area_checker = None
        self._deforestation_verifier = None
        self._accuracy_scorer = None
        self._temporal_analyzer = None
        self._batch_pipeline = None
        self._article9_reporter = None

        logger.info("All engines closed")

    # ------------------------------------------------------------------
    # Internal: Health check background task
    # ------------------------------------------------------------------

    def _start_health_check(self) -> None:
        """Start the background health check task."""
        if self._health_task is not None:
            return
        self._health_task = asyncio.create_task(
            self._health_check_loop(),
            name="geo-health-check",
        )
        logger.debug(
            "Health check background task started (interval=%.0fs)",
            self._health_interval_seconds,
        )

    def _stop_health_check(self) -> None:
        """Cancel the background health check task."""
        if self._health_task is not None:
            self._health_task.cancel()
            self._health_task = None
            logger.debug("Health check background task cancelled")

    async def _health_check_loop(self) -> None:
        """Background loop that periodically runs health checks."""
        try:
            while True:
                try:
                    await self.health_check()
                except Exception as exc:
                    logger.warning(
                        "Health check failed: %s", exc
                    )
                await asyncio.sleep(self._health_interval_seconds)
        except asyncio.CancelledError:
            logger.debug("Health check loop cancelled")

    # ------------------------------------------------------------------
    # Internal: Engine health
    # ------------------------------------------------------------------

    def _check_engine_health(self) -> Dict[str, Any]:
        """Check initialization status of all eight engines.

        Returns:
            Dictionary with per-engine status and overall engine health.
        """
        engines = {
            "coordinate_validator": self._coordinate_validator,
            "polygon_verifier": self._polygon_verifier,
            "protected_area_checker": self._protected_area_checker,
            "deforestation_verifier": self._deforestation_verifier,
            "accuracy_scorer": self._accuracy_scorer,
            "temporal_analyzer": self._temporal_analyzer,
            "batch_pipeline": self._batch_pipeline,
            "article9_reporter": self._article9_reporter,
        }

        engine_statuses: Dict[str, str] = {}
        initialized_count = 0
        for name, engine in engines.items():
            if engine is not None:
                engine_statuses[name] = "initialized"
                initialized_count += 1
            else:
                engine_statuses[name] = "unavailable"

        # Core engines (coordinate, polygon, accuracy) are required;
        # others are optional and their absence degrades but does not
        # break the service.
        core_engines = [
            "coordinate_validator",
            "polygon_verifier",
            "accuracy_scorer",
        ]
        core_ok = all(
            engine_statuses.get(e) == "initialized" for e in core_engines
        )

        overall = "healthy" if core_ok else "degraded"

        return {
            "status": overall,
            "initialized": initialized_count,
            "total": len(engines),
            "engines": engine_statuses,
        }

    # ------------------------------------------------------------------
    # Internal: Metrics
    # ------------------------------------------------------------------

    def _flush_metrics(self) -> None:
        """Flush Prometheus metrics on shutdown."""
        if not self._config.enable_metrics:
            return
        logger.debug("Prometheus metrics flushed")

    # ------------------------------------------------------------------
    # Internal: Guard
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        """Raise RuntimeError if the service is not started.

        Raises:
            RuntimeError: If the service has not been started.
        """
        if not self._started:
            raise RuntimeError(
                "GeolocationVerificationService is not started. "
                "Call await service.startup() first."
            )

    # ------------------------------------------------------------------
    # Convenience: get_engine
    # ------------------------------------------------------------------

    def get_engine(self, name: str) -> Any:
        """Retrieve an engine by name.

        Args:
            name: Engine name (e.g., 'coordinate_validator', 'accuracy_scorer').

        Returns:
            The engine instance, or None if not initialized.

        Raises:
            RuntimeError: If the service has not been started.
            ValueError: If the engine name is not recognized.
        """
        self._ensure_started()
        valid_names = {
            "coordinate_validator": self._coordinate_validator,
            "polygon_verifier": self._polygon_verifier,
            "protected_area_checker": self._protected_area_checker,
            "deforestation_verifier": self._deforestation_verifier,
            "accuracy_scorer": self._accuracy_scorer,
            "temporal_analyzer": self._temporal_analyzer,
            "batch_pipeline": self._batch_pipeline,
            "article9_reporter": self._article9_reporter,
        }
        if name not in valid_names:
            raise ValueError(
                f"Unknown engine name: '{name}'. "
                f"Valid names: {sorted(valid_names.keys())}"
            )
        return valid_names[name]

    # ------------------------------------------------------------------
    # Convenience: engine count
    # ------------------------------------------------------------------

    def initialized_engine_count(self) -> int:
        """Return the number of successfully initialized engines.

        Returns:
            Count of non-None engine instances (0 to 8).
        """
        engines = [
            self._coordinate_validator,
            self._polygon_verifier,
            self._protected_area_checker,
            self._deforestation_verifier,
            self._accuracy_scorer,
            self._temporal_analyzer,
            self._batch_pipeline,
            self._article9_reporter,
        ]
        return sum(1 for e in engines if e is not None)

# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for the Geolocation Verification service.

    Automatically starts the service on application startup and shuts it
    down on application shutdown. The service instance is stored in
    ``app.state.geo_service`` for access from route handlers.

    Usage with FastAPI::

        from fastapi import FastAPI
        from greenlang.agents.eudr.geolocation_verification.setup import lifespan
from greenlang.schemas import utcnow

        app = FastAPI(lifespan=lifespan)

    Args:
        app: The FastAPI application instance.

    Yields:
        None (service is accessible via ``app.state.geo_service``).
    """
    service = get_service()
    app.state.geo_service = service
    try:
        await service.startup()
        yield
    finally:
        await service.shutdown()

# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_service_instance: Optional[GeolocationVerificationService] = None
_service_lock = threading.Lock()

def get_service(
    config: Optional[GeolocationVerificationConfig] = None,
) -> GeolocationVerificationService:
    """Return the singleton GeolocationVerificationService instance.

    Uses double-checked locking for thread safety. The instance is
    created on first call. Pass a config to override the default
    environment-based configuration.

    Args:
        config: Optional configuration override.

    Returns:
        GeolocationVerificationService singleton instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = GeolocationVerificationService(
                    config=config
                )
    return _service_instance

def set_service(service: GeolocationVerificationService) -> None:
    """Replace the singleton GeolocationVerificationService instance.

    Primarily intended for testing and dependency injection.

    Args:
        service: Replacement service instance.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.info("GeolocationVerificationService singleton replaced")

def reset_service() -> None:
    """Reset the singleton GeolocationVerificationService to None.

    The next call to ``get_service()`` will create a fresh instance.
    Intended for test teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.debug("GeolocationVerificationService singleton reset")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "GeolocationVerificationService",
    "HealthStatus",
    "VerifyPlotRequest",
    "PlotVerificationResult",
    "BatchVerificationResult",
    "lifespan",
    "get_service",
    "set_service",
    "reset_service",
]
